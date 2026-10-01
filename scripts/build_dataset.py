"""Build v1 short: canonical FM + FD labels/odds + ESPN odds -> JSONL por split.

Corre NO .venv (pandas+pyreadr): ../.venv/bin/python build_dataset.py
Fases: preload rows -> passada cronologica (rolling < cutoff) -> export
-> leakage_test.py. Tudo com resume parcial via manifest.
"""
import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import PROCESSED  # noqa: E402
from features import aggregates, h2h, last_n, rest_days, wdl  # noqa: E402
from loaders import (fotmob_detail, iter_espn_months, iter_fd,  # noqa: E402
                     iter_fotmob_days, parse_fd_date, season_end_year)
from questions import render as render_questions  # noqa: E402
from teams import normalize  # noqa: E402
from verbalize import verbalize  # noqa: E402

COMP = {"ENG": "Premier League", "ESP": "LaLiga", "GER": "Bundesliga",
        "ITA": "Serie A", "FRA": "Ligue 1"}


def game_key(lg, se, h, a):
    return (lg, se, h, a)


def load_all():
    games = []
    for g in iter_fotmob_days():
        if not g["finished"] or not g.get("ts"):
            continue
        dt = datetime.fromtimestamp(g["ts"] / 1000, tz=timezone.utc)
        se = season_end_year(dt)
        if se < 2016 or se > 2025:
            continue
        games.append({"lg": g["lg"], "se": se,
                      "home": normalize(g["home"]), "away": normalize(g["away"]),
                      "dh": g["home"], "da": g["away"],
                      "hg": g["hg"], "ag": g["ag"], "mid": g["mid"],
                      "kickoff": dt, "round": None, "venue": None})
    fd = {}
    for r in iter_fd():
        dt = parse_fd_date(r)
        if not dt:
            continue
        se = season_end_year(dt)
        if se < 2016 or se > 2025:
            continue
        fd[game_key(r["lg"], se, normalize(r["HomeTeam"]),
                    normalize(r["AwayTeam"]))] = r
    espn = {}
    for g in iter_espn_months():
        if not g["finished"] or not g.get("date"):
            continue
        try:
            dt = datetime.fromisoformat(g["date"].replace("Z", "+00:00"))
        except ValueError:
            continue
        se = season_end_year(dt)
        if se < 2016 or se > 2025:
            continue
        espn[game_key(g["lg"], se, normalize(g["home"]),
                      normalize(g["away"]))] = g["eid"]
    return games, fd, espn


def wfr_check(games):
    """Cross-check scores contra worldfootballR (so log, nao bloqueia)."""
    try:
        import pyreadr
    except ImportError:
        print("[build] pyreadr em falta, skip wfr check", flush=True)
        return
    from config import RAW
    codes = {"ENG": "ENG", "ESP": "ESP", "GER": "GER", "ITA": "ITA", "FRA": "FRA"}
    ref = {}
    for lg, code in codes.items():
        p = RAW / "worldfootballR_data" / "match_results" / f"{code}_match_results.rds"
        if not p.exists():
            continue
        df = pyreadr.read_r(str(p))[None]
        for _, r in df.iterrows():
            try:
                se = int(r["Season_End_Year"])
            except (ValueError, TypeError):
                continue
            if se < 2016 or se > 2025:
                continue
            try:
                hg_, ag_ = r["HomeGoals"], r["AwayGoals"]
                if hg_ != hg_ or ag_ != ag_:  # NaN = jogo por jogar
                    continue
                ref[(lg, se, normalize(str(r["Home"])), normalize(str(r["Away"])))] = \
                    (int(hg_), int(ag_))
            except (ValueError, TypeError):
                continue
    same = diff = missing = 0
    for g in games:
        k = game_key(g["lg"], g["se"], g["home"], g["away"])
        if k not in ref:
            missing += 1
        elif ref[k] == (g["hg"], g["ag"]):
            same += 1
        else:
            diff += 1
    print(f"[build] wfr check: iguais={same} diferentes={diff} sem-ref={missing}",
          flush=True)


def parse_espn_odds(eid_lg):
    return {}


def build_labels(g, det, fdrow):
    """Devolve (labels, dropped). None = sem dados -> pergunta cai."""
    L, dropped = {}, []
    hg, ag = g["hg"], g["ag"]
    tot = hg + ag

    def put(k, v):
        if v is None:
            dropped.append(k)
        else:
            L[k] = int(v) if isinstance(v, bool) else v

    put("match_result", "home" if hg > ag else ("draw" if hg == ag else "away"))
    put("home_win", int(hg > ag))
    put("away_win", int(hg < ag))
    put("draw", int(hg == ag))
    put("home_double", int(hg >= ag))
    put("away_double", int(hg <= ag))
    for line, key in ((0.5, "over_05"), (1.5, "over_15"), (2.5, "over_25"),
                      (3.5, "over_35"), (4.5, "over_45")):
        put(key, int(tot > line))
    put("under_25", int(tot < 3))
    put("under_35", int(tot < 4))
    put("btts", int(hg > 0 and ag > 0))
    put("home_scores", int(hg > 0))
    put("away_scores", int(ag > 0))
    put("home_over_05", int(hg > 0))
    put("home_over_15", int(hg >= 2))
    put("away_over_05", int(ag > 0))
    put("away_over_15", int(ag >= 2))
    put("away_over_25", int(ag >= 3))
    put("odd_total", int(tot % 2 == 1))
    put("home_clean_sheet", int(ag == 0))
    put("away_clean_sheet", int(hg == 0))
    put("away_win_by_1", int(ag - hg == 1))
    put("away_win_by_2plus", int(ag - hg >= 2))

    # HT: FD primario
    hth = hta = None
    if fdrow:
        try:
            hth, hta = int(fdrow["HTHG"]), int(fdrow["HTAG"])
        except (ValueError, TypeError, KeyError):
            pass
    if hth is None and det:
        hth = sum(1 for x in det["goals"]
                  if x["side"] == "h" and _first_half(x) and not x["owngoal"])
        hta = sum(1 for x in det["goals"]
                  if x["side"] == "a" and _first_half(x) and not x["owngoal"])
    if hth is None:
        for k in ("ht_over_05", "ht_over_15", "home_ht_scores", "away_ht_scores",
                  "ht_btts_no", "ht_result", "second_half_over_15",
                  "away_both_halves"):
            dropped.append(k)
    else:
        put("ht_over_05", int(hth + hta >= 1))
        put("ht_over_15", int(hth + hta >= 2))
        put("home_ht_scores", int(hth > 0))
        put("away_ht_scores", int(hta > 0))
        put("ht_btts_no", int(not (hth > 0 and hta > 0)))
        put("ht_result", "home" if hth > hta else ("draw" if hth == hta else "away"))
        put("second_half_over_15", int((hg - hth) + (ag - hta) >= 2))
        put("away_both_halves", int(hta > 0 and (ag - hta) > 0))

    # primeiro golo (detalhe FM)
    if det and det["goals"]:
        first = min(det["goals"], key=lambda e: (e["min"] or 999))
        put("first_goal_away", int(first["side"] == "a"))
    else:
        put("first_goal_away", int(hg == 0 and ag > 0))

    # cantos/remates/cartoes: FD primario, FM fallback
    hc = ac = hs = aws = hst = ast = yc = rc = None
    if fdrow:
        try:
            hc, ac = int(fdrow["HC"]), int(fdrow["AC"])
            hs, aws = int(fdrow["HS"]), int(fdrow["AS"])
            hst, ast = int(fdrow["HST"]), int(fdrow["AST"])
            yc = int(fdrow["HY"]) + int(fdrow["AY"])
            rc = int(fdrow["HR"]) + int(fdrow["AR"])
        except (ValueError, TypeError, KeyError):
            pass
    if hc is None and det:
        st = det["stats"].get("All", {})
        hc = _pair(st.get("corners"), 0)
        ac = _pair(st.get("corners"), 1)
        hs = _pair(st.get("total_shots"), 0)
        aws = _pair(st.get("total_shots"), 1)
        hst = _pair(st.get("ShotsOnTarget"), 0)
        ast = _pair(st.get("ShotsOnTarget"), 1)
        yc = _pair(st.get("yellow_cards"), 0, d=0) + _pair(st.get("yellow_cards"), 1, d=0)
        rc = ((det.get("red_h") or 0) + (det.get("red_a") or 0)) or None
    for k, v in (("corners_over_75", (hc or -1) + (ac or -1) > 7 if hc is not None else None),
                 ("corners_over_85", (hc + ac) > 8 if hc is not None else None),
                 ("corners_over_95", (hc + ac) > 9 if hc is not None else None),
                 ("corners_over_105", (hc + ac) > 10 if hc is not None else None),
                 ("corners_under_105", (hc + ac) < 11 if hc is not None else None),
                 ("corners_under_125", (hc + ac) < 13 if hc is not None else None),
                 ("home_corners_over_35", hc > 3 if hc is not None else None),
                 ("home_corners_over_45", hc > 4 if hc is not None else None),
                 ("away_corners_over_35", ac > 3 if ac is not None else None),
                 ("away_corners_over_45", ac > 4 if ac is not None else None),
                 ("away_corners_over_55", ac > 5 if ac is not None else None),
                 ("shots_over_265", (hs + aws) > 26 if hs is not None else None),
                 ("home_shots_over_105", hs > 10 if hs is not None else None),
                 ("away_shots_over_145", aws > 14 if aws is not None else None),
                 ("sot_over_75", (hst + ast) > 7 if hst is not None else None),
                 ("sot_over_85", (hst + ast) > 8 if hst is not None else None),
                 ("cards_over_35", yc > 3 if yc is not None else None),
                 ("cards_over_45", yc > 4 if yc is not None else None),
                 ("red_card", int(rc > 0) if rc is not None else None)):
        put(k, v)
    if hc is not None:
        put("most_corners", "home" if hc > ac else ("draw" if hc == ac else "away"))
    else:
        dropped.append("most_corners")

    # jogadores: golos nao-own-goal do detalhe
    scorers = {}
    if det:
        for x in det["goals"]:
            if not x["owngoal"] and x["player"]:
                scorers.setdefault(normalize(x["player"]), []).append(x["side"])
    for qid, name, is2, isht in g.get("_pjobs", []):
        n = normalize(name)
        c = len(scorers.get(n, []))
        if is2:
            put(qid, int(c >= 2))
        elif isht:
            ht_c = sum(1 for x in det["goals"] if normalize(x["player"] or "") == n
                       and not x["owngoal"] and _first_half(x)) if det else 0
            put(qid, int(ht_c >= 1))
        else:
            put(qid, int(c >= 1))
    return L, dropped


def _first_half(x):
    m = x.get("min") or 0
    ms = x.get("minstr", "")
    return m < 45 or (m == 45 and "+" not in ms)


def _pair(v, i, d=None):
    try:
        if isinstance(v, list) and len(v) == 2:
            f = float(v[i])
            return f
        return d
    except (ValueError, TypeError):
        return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--seasons", nargs="*", type=int, default=list(range(2016, 2026)))
    a = ap.parse_args()
    PROCESSED.mkdir(parents=True, exist_ok=True)

    print("[build] a carregar fontes...", flush=True)
    games, fd, espn = load_all()
    games = [g for g in games if g["se"] in a.seasons]
    if a.max_games:
        games = sorted(games, key=lambda g: g["kickoff"])[-a.max_games:]
    print(f"[build] jogos canonicos: {len(games)}", flush=True)
    wfr_check(games)

    games.sort(key=lambda g: g["kickoff"])
    hist = {}
    pair_hist = {}
    scorers = {}
    tables = {}
    disp = {}
    for g in games:
        disp.setdefault(g["home"], g["dh"])
        disp.setdefault(g["away"], g["da"])

    out = {"train": [], "calib": [], "test": []}
    cnt = Counter()
    for gi, g in enumerate(games):
        if gi % 2000 == 0:
            print(f"[build] {gi}/{len(games)}", flush=True)
        cutoff = g["kickoff"] - timedelta(minutes=50)
        key = (g["lg"], g["se"], g["home"], g["away"])
        det = fotmob_detail(g["mid"])
        fdrow = fd.get(key)
        if det:
            g["round"] = det.get("round")
        lh = [r for r in hist.get((g["lg"], g["home"]), []) if r[0] < cutoff][-5:]
        la = [r for r in hist.get((g["lg"], g["away"]), []) if r[0] < cutoff][-5:]
        tab, pos = tables.get((g["lg"], g["se"]), ({}, {}))
        feat = {
            "last_h": lh, "last_a": la,
            "agg_h": aggregates(lh), "agg_a": aggregates(la),
            "table": {g["home"]: tab.get(g["home"], {"pts": 0, "p": 0, "w": 0, "d": 0,
                                                     "l": 0, "gf": 0, "ga": 0}),
                      g["away"]: tab.get(g["away"], {"pts": 0, "p": 0, "w": 0, "d": 0,
                                                     "l": 0, "gf": 0, "ga": 0})},
            "pos": {g["home"]: pos.get(g["home"], 99),
                    g["away"]: pos.get(g["away"], 99)},
            "h2h": [x for x in
                    pair_hist.get((g["lg"], frozenset((g["home"], g["away"]))), [])
                    if x["kickoff"] < cutoff][-5:],
            "rest_h": (cutoff - max([r[0] for r in hist.get((g["lg"], g["home"]), [])
                                     if r[0] < cutoff], default=None)).days
            if any(r[0] < cutoff for r in hist.get((g["lg"], g["home"]), [])) else None,
            "rest_a": None,
        }
        ra = [r[0] for r in hist.get((g["lg"], g["away"]), []) if r[0] < cutoff]
        feat["rest_a"] = (cutoff - max(ra)).days if ra else None
        xi = None
        if det and det["lineup"].get("h", {}).get("xi"):
            xh, xa = det["lineup"]["h"], det["lineup"]["a"]
            xi = {"h_form": xh.get("formation"), "a_form": xa.get("formation"),
                  "h_names": [p["name"] for p in xh["xi"]],
                  "a_names": [p["name"] for p in xa["xi"]],
                  "missing": ([f"{m['name']} ({m['why']})" for m in xh["missing"]] +
                              [f"{m['name']} ({m['why']})" for m in xa["missing"]])}
        feat["xi"] = xi
        sc = scorers.get((g["lg"], g["home"]), {})
        sa = scorers.get((g["lg"], g["away"]), {})
        feat["top_scorer_h"] = max(sc.items(), key=lambda kv: kv[1], default=(None, 0))
        feat["top_scorer_a"] = max(sa.items(), key=lambda kv: kv[1], default=(None, 0))
        if feat["top_scorer_h"][1] == 0:
            feat["top_scorer_h"] = None
        if feat["top_scorer_a"][1] == 0:
            feat["top_scorer_a"] = None

        # jogadores p/ perguntas: XI (sem GR=idx0) + 1 sub; fallbacks p/ 70 fixas
        g["_pjobs"] = []
        pnames, ph2, ph = [], [], []
        full_xi = bool(xi and len(xi["h_names"]) >= 11 and len(xi["a_names"]) >= 11)
        if full_xi:
            out_ = (xi["h_names"][1:] + xi["a_names"][1:])
            subs = []
            if det:
                subs = (det["lineup"]["h"].get("subs", []) +
                        det["lineup"]["a"].get("subs", []))
            cands = (out_ + subs)[:11]
            while len(cands) < 11:
                cands.append(xi["h_names"][0])
            pnames = cands
            ranked = sorted(cands, key=lambda n: (-sc.get(n, 0) - sa.get(n, 0), n))
            ph2 = ranked[:2]
            ph = ranked[:2]
        else:
            cnt["sem_xi"] += 1
        if full_xi:
            for i, n in enumerate(pnames):
                g["_pjobs"].append((f"p{i}_anytime", n, False, False))
            for i, n in enumerate(ph2):
                g["_pjobs"].append((f"p2_{i}", n, True, False))
            for i, n in enumerate(ph):
                g["_pjobs"].append((f"pht_{i}", n, False, True))

        dispH, dispA = disp[g["home"]], disp[g["away"]]
        if not full_xi:
            pass  # sem XI completo: conta para rolling mas nao gera caso
        else:
            state = verbalize({**g, "home": g["home"], "away": g["away"]}, feat,
                              {**{k: (disp.get(k) or k) for k in
                                     set(list(disp) + [g["home"], g["away"]])},
                               g["home"]: dispH, g["away"]: dispA})
            from questions import render as _rq
            qs = _rq(dispH, dispA, {"anytime_11": pnames, "two": ph2, "ht": ph})
            labels, dropped = build_labels(g, det, fdrow)
            keep_qs = {k: v for k, v in qs.items() if k in labels}
            odds = {}
            if fdrow:
                try:
                    odds = {"1x2": [float(fdrow["B365H"]), float(fdrow["B365D"]),
                                    float(fdrow["B365A"])],
                            "ou25": [float(fdrow["B365>2.5"]), float(fdrow["B365<2.5"])]}
                except (ValueError, TypeError, KeyError):
                    pass
            case = {
                "event_id": f"fm{g['mid']}", "lg": g["lg"], "se": g["se"],
                "kickoff_utc": g["kickoff"].isoformat(),
                "cutoff_utc": cutoff.isoformat(),
                "round": g["round"],
                "state_text": state, "questions": keep_qs, "labels": labels,
                "odds_at_cutoff": odds,
                "provenance": {
                    "standings_round_used": None,
                    "last_game_included_home": lh[-1][0].isoformat() if lh else None,
                    "last_game_included_away": la[-1][0].isoformat() if la else None,
                    "lineup_source": "fotmob_confirmed",
                    "odds_timestamp": "closing_proxy",
                    "dropped_questions": dropped,
                    "fd_matched": fdrow is not None,
                },
            }
            split = "train" if g["se"] <= 2023 else ("calib" if g["se"] == 2024 else "test")
            out[split].append(case)
            cnt[split] += 1

        # atualiza rolling COM este jogo (para os proximos; cutoff ja passou)
        pair_hist.setdefault((g["lg"], frozenset((g["home"], g["away"]))), []).append(g)
        for team, gf, ga, adv, hc in ((g["home"], g["hg"], g["ag"], g["away"], True),
                                      (g["away"], g["ag"], g["hg"], g["home"], False)):
            hist.setdefault((g["lg"], team), []).append(
                (g["kickoff"], hc, gf, ga, adv))
            t = tables.setdefault((g["lg"], g["se"]), ({}, {}))[0]
            r_ = t.setdefault(team, {"p": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0})
            r_["p"] += 1
            r_["gf"] += gf
            r_["ga"] += ga
            if gf > ga:
                r_["w"] += 1
            elif gf == ga:
                r_["d"] += 1
            else:
                r_["l"] += 1
            for k in list(t):
                t[k]["pts"] = 3 * t[k]["w"] + t[k]["d"]
            order = sorted(t, key=lambda k: (-t[k]["pts"],
                                             -(t[k]["gf"] - t[k]["ga"]), -t[k]["gf"]))
            tables[(g["lg"], g["se"])] = (t, {k: i + 1 for i, k in enumerate(order)})
        if det:
            for x in det["goals"]:
                if x["player"] and not x["owngoal"]:
                    k = (g["lg"], g["home"] if x["side"] == "h" else g["away"])
                    d_ = scorers.setdefault(k, {})
                    d_[x["player"]] = d_.get(x["player"], 0) + 1

    for split, cases in out.items():
        p = PROCESSED / f"{split}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for c in cases:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"[build] {split}: {len(cases)} casos -> {p}", flush=True)
    print("[build] contagem:", dict(cnt), flush=True)
    r = subprocess.run([sys.executable, str(Path(__file__).parent / "leakage_test.py")],
                       capture_output=True, text=True)
    print(r.stdout[-1500:], flush=True)
    if r.returncode != 0:
        print(r.stderr[-500:], flush=True)
        raise SystemExit("leakage_test FALHOU")


if __name__ == "__main__":
    main()
