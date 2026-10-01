"""Servidor live do modelo n_2000: fixtures -> estado -> modelo -> odds -> EV.

Pipeline (espelha o treino, mas tudo live):
  1. FIXTURES: FotMob `matches?date=` dos proximos dias (Big5, sem auth).
  2. ESTADO: rolling point-in-time a partir do espelho `data/raw` + refresh
     live dos ultimos dias + `matchDetails` do proprio jogo (lineups
     confirmados, h2h). Verbalizado com a MESMA `verbalize()` do treino.
     Cutoff live = agora (so jogos terminados), equivalente pratico do
     kickoff-50min do treino.
  3. MODELO: forward dos mercados apostaveis (match_result, over/under 2.5)
     com os pesos n_2000 + temperaturas do `metrics_n_2000.json`.
  4. ODDS: Betclic + Betano (.pt, scrape do HTML server-rendered, sem auth)
     -> --manual-odds para testes.
  5. EV: edge = p*odd - 1 por lado e por casa; aposta se edge >= --min-edge;
     stake via Kelly fracionado (cap --max-stake).

Fontes (decisao, ver README_LIVE no fim do ficheiro):
  - Estado: FotMob (fixtures, resultados, lineups, h2h, tabela) + espelho
    local como base do rolling. ESPN como backup. Sorare desligado por
    defeito (precisa de key; so serve props de jogadores, fora do EV 1x2/OU).
    Sofascore excluida (403), Understat excluido (deprecated).
  - Odds: football-data sao CSVs historicos -> nao servem live. Betclic e
    Betano .pt fazem server-side render do HTML com odds 1X2 + totais
    (sem auth, GET simples) -> scrape direto dessas paginas, sem API paga
    e sem browser. Sem key nenhuma.

Uso:
  ../.venv/bin/python live_server.py scan --hours-ahead 72
  ../.venv/bin/python live_server.py scan --hours-ahead 72 --manual-odds "1x2:2.10,3.40,3.20;ou25:1.95,1.85"
  ../.venv/bin/python live_server.py serve --port 8080 --interval 900
  Decisions: data/live/decisions.jsonl | HTTP: /health /decisions

Sem dependencias novas: stdlib + torch/transformers/laya (as do projeto).
"""
import argparse
import json
import sys
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import TokenBucket, fetch_text, load_dotenv  # noqa: E402
from config import RAW  # noqa: E402
from features import aggregates, wdl  # noqa: E402
from teams import normalize  # noqa: E402
from verbalize import verbalize  # noqa: E402

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED = BASE_DIR / "data" / "processed"
LIVE_DIR = BASE_DIR / "data" / "live"

# Defaults de calibracao vindos do metrics_n_2000.json do Kaggle
# (temperature por qtype, tbo por bucket). --calibration-json sobrepoe.
DEFAULT_TEMPERATURE = [2.7913668155670166, 1.0, 2.584965944290161]
DEFAULT_TBO = {"choice:3-5": 2.7913668155670166, "noul:2": 2.584965944290161}

MODEL_CANDIDATES = [
    BASE_DIR / "data" / "models" / "curve_kaggle" / "n_2000",
    BASE_DIR / "kaggle_bundle" / "upload_n2000" / "n_2000",
]

FOTMOB = "https://www.fotmob.com/api/data"
BIG5 = {47: "ENG", 87: "ESP", 54: "GER", 55: "ITA", 53: "FRA"}
COMP = {"ENG": "Premier League", "ESP": "LaLiga", "GER": "Bundesliga",
        "ITA": "Serie A", "FRA": "Ligue 1"}

# Odds tratadas em odds_scrape.py (get_odds_for_fixture). Este modulo so
# construi o indice de ligas via _league_index.

_bucket = TokenBucket(1.5)


# ---------- fixtures + detalhes live (FotMob) ----------

def fotmob_day(day):
    p = RAW / "fotmob" / "days" / f"{day:%Y%m%d}.json"
    if p.exists():
        try:
            cached = json.loads(p.read_text(encoding="utf-8"))
            # Dias passados/hoje com jogos por acabar: o espelho congela os
            # resultados no primeiro fetch; refrescar para o rolling nao
            # perder a ultima jornada e para haver resultados p/ settlement.
            today = datetime.now(timezone.utc).date()
            stale = day <= today and any(
                not (m.get("status") or {}).get("finished")
                for lg in cached.get("leagues", []) for m in lg.get("matches", []))
            if not stale:
                return cached
        except (OSError, ValueError):
            pass
    code, text = fetch_text(f"{FOTMOB}/matches?date={day:%Y%m%d}")
    if code != 200 or not text or not text.startswith("{"):
        print(f"[live] FotMob dia {day} falhou ({code})", flush=True)
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
    # espelha no raw (mesmo formato do fetcher; sobrepoe para refrescar)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return json.loads(text)


def upcoming_fixtures(hours_ahead, leagues=None):
    """Jogos Big5 futuros (nao terminados) nas proximas `hours_ahead` horas."""
    now = datetime.now(timezone.utc)
    out = []
    days = (hours_ahead // 24) + 2
    for i in range(days):
        d = fotmob_day((now + timedelta(days=i)).date())
        for lg in d.get("leagues", []):
            code = BIG5.get(lg.get("id"))
            if not code or (leagues and code not in leagues):
                continue
            for m in lg.get("matches", []):
                if (m.get("status") or {}).get("finished"):
                    continue
                ts = m.get("timeTS")
                if not ts:
                    continue
                ko = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                if now < ko < now + timedelta(hours=hours_ahead):
                    out.append({"lg": code, "mid": m["id"],
                                "home": m["home"]["name"], "away": m["away"]["name"],
                                "kickoff": ko})
    out.sort(key=lambda g: g["kickoff"])
    return out


def live_detail(mid):
    _bucket.wait()
    code, text = fetch_text(f"{FOTMOB}/matchDetails?matchId={mid}", timeout=30)
    if code != 200 or not text or not text.startswith("{"):
        return None
    try:
        from loaders import fotmob_detail as _parse  # noqa
    except ImportError:
        return None
    # fotmob_detail le do espelho; grava primeiro para reutilizar o parser
    p = RAW / "fotmob" / f"match_{mid}.json"
    if not p.exists():
        p.write_text(text, encoding="utf-8")
    return _parse(mid)


# ---------- rolling live (espelho + refresh) ----------

def build_history():
    """Jogos terminados por equipa: (kickoff, foi_casa, gf, ga, adv)."""
    from loaders import iter_fotmob_days
    hist, pair, tables_order = {}, {}, {}
    games = []
    for g in iter_fotmob_days():
        if not g["finished"] or not g.get("ts") or g["hg"] is None:
            continue
        dt = datetime.fromtimestamp(g["ts"] / 1000, tz=timezone.utc)
        games.append({"lg": g["lg"], "se": None, "home": normalize(g["home"]),
                      "away": normalize(g["away"]), "hg": g["hg"], "ag": g["ag"],
                      "kickoff": dt})
    games.sort(key=lambda g: g["kickoff"])
    for g in games:
        pair.setdefault((g["lg"], frozenset((g["home"], g["away"]))), []).append(g)
        for team, gf, ga, adv, hc in ((g["home"], g["hg"], g["ag"], g["away"], True),
                                      (g["away"], g["ag"], g["hg"], g["home"], False)):
            hist.setdefault((g["lg"], team), []).append(
                (g["kickoff"], hc, gf, ga, adv))
    # tabela da epoca corrente por liga (so jogos passados)
    tables = {}
    for g in games:
        t = tables.setdefault(g["lg"], {})
        for team, gf, ga in ((g["home"], g["hg"], g["ag"]),
                             (g["away"], g["ag"], g["hg"])):
            r = t.setdefault(team, {"p": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0})
            r["p"] += 1
            r["gf"] += gf
            r["ga"] += ga
            r["w"] += gf > ga
            r["d"] += gf == ga
            r["l"] += gf < ga
    for lg, t in tables.items():
        for r in t.values():
            r["pts"] = 3 * r["w"] + r["d"]
        order = sorted(t, key=lambda k: (-t[k]["pts"], -(t[k]["gf"] - t[k]["ga"])))
        tables_order[lg] = (t, {k: i + 1 for i, k in enumerate(order)})
    return hist, pair, tables_order


def feat_for(fx, hist, pair, tables_order, det, now):
    lg, h, a = fx["lg"], normalize(fx["home"]), normalize(fx["away"])
    lh = [r for r in hist.get((lg, h), []) if r[0] < now][-5:]
    la = [r for r in hist.get((lg, a), []) if r[0] < now][-5:]
    tab, pos = tables_order.get(lg, ({}, {}))
    zero = {"pts": 0, "p": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0}
    feat = {
        "last_h": lh, "last_a": la,
        "agg_h": aggregates(lh), "agg_a": aggregates(la),
        "table": {h: tab.get(h, dict(zero)), a: tab.get(a, dict(zero))},
        "pos": {h: pos.get(h, 99), a: pos.get(a, 99)},
        "h2h": [{"hg": x["hg"], "ag": x["ag"],
                 "home": x["home"], "away": x["away"]}
                for x in pair.get((lg, frozenset((h, a))), [])
                if x["kickoff"] < now][-5:],
        "rest_h": None, "rest_a": None,
        "xi": None, "top_scorer_h": None, "top_scorer_a": None,
    }
    rh = [r[0] for r in hist.get((lg, h), []) if r[0] < now]
    ra = [r[0] for r in hist.get((lg, a), []) if r[0] < now]
    feat["rest_h"] = (now - max(rh)).days if rh else None
    feat["rest_a"] = (now - max(ra)).days if ra else None
    xi_confirmed = False
    if det and (det.get("lineup") or {}).get("h", {}).get("xi"):
        xh, xa = det["lineup"]["h"], det["lineup"]["a"]
        if len(xh["xi"]) >= 11 and len(xa["xi"]) >= 11:
            xi_confirmed = True
            feat["xi"] = {
                "h_form": xh.get("formation"), "a_form": xa.get("formation"),
                "h_names": [p["name"] for p in xh["xi"]],
                "a_names": [p["name"] for p in xa["xi"]],
                "missing": ([f"{m['name']} ({m['why']})" for m in xh["missing"]] +
                            [f"{m['name']} ({m['why']})" for m in xa["missing"]])}
    return feat, xi_confirmed


# ---------- modelo ----------

def load_model(model_dir, device):
    from transformers import AutoTokenizer
    from laya.agent import _fix_tokenizer_config
    from laya.common import build_model
    from safetensors.torch import load_file
    d = Path(model_dir)
    _fix_tokenizer_config(str(d))
    import json as _j
    with open(d / "rl_agent_config.json") as f:
        cfg = _j.load(f)
    tok = AutoTokenizer.from_pretrained(d / "tokenizer")
    enc = d / "encoder"
    model = build_model(cfg, encoder_dir=str(enc) if enc.exists() else None,
                        pretrained=False)
    model.load_state_dict(load_file(str(d / "model.safetensors")), strict=True)
    model.to(device)
    model.eval()
    return model, tok, cfg


def build_live_questions(disp_home, disp_away, det):
    """As 70 perguntas do treino (templates identicos a questions.py).

    Jogadores (15 qids) so com XI confirmado dos dois lados (receita igual
    ao build_dataset); sem XI, corre so as 55 de equipas. Devolve (qs, xi_ok).
    """
    from questions import (CARDS, CORNERS, HTM, PLAYER_2PLUS, PLAYER_ANYTIME,
                           PLAYER_HT, QUESTIONS, SHOTS, _fill)
    base = {**QUESTIONS,
            **{k: ({"type": "noul", "instructions": v}
                   if isinstance(v, str) else v)
               for k, v in {**CORNERS, **SHOTS, **CARDS, **HTM}.items()}}
    qs = {qid: _fill(dict(q), disp_home, disp_away) for qid, q in base.items()}
    xi_ok = False
    try:
        ln = (det or {}).get("lineup") or {}
        xh, xa = ln.get("h") or {}, ln.get("a") or {}
        if len(xh.get("xi", [])) >= 11 and len(xa.get("xi", [])) >= 11:
            out_ = ([p["name"] for p in xh["xi"][1:]] +
                    [p["name"] for p in xa["xi"][1:]])
            subs = (xh.get("subs", []) + xa.get("subs", []))
            cands = (out_ + subs)[:11]
            while len(cands) < 11:
                cands.append(xh["xi"][0]["name"])
            for i, n in enumerate(cands):
                qs[f"p{i}_anytime"] = {
                    "type": "noul",
                    "instructions": PLAYER_ANYTIME.replace("PLAYERNAME", n)}
            for i, n in enumerate(cands[:2]):
                qs[f"p2_{i}"] = {
                    "type": "noul",
                    "instructions": PLAYER_2PLUS.replace("PLAYERNAME", n)}
                qs[f"pht_{i}"] = {
                    "type": "noul",
                    "instructions": PLAYER_HT.replace("PLAYERNAME", n)}
            xi_ok = True
    except (KeyError, IndexError, TypeError):
        pass
    return qs, xi_ok


def predict(model, tok, cfg, state_text, questions, device, batch=8):
    """Forward em batch de N perguntas. Devolve ({qid: {lado: prob}}, skipped).

    `questions`: {qid: {"type": "choice"/"noul", "instructions":, "criteria":}}.
    Batching: 70 perguntas em ~9 forwards (batch 8) em vez de 70 — e o que
    torna o live por jogo uma questao de segundos em CPU.
    """
    import numpy as np
    import torch
    from laya.common import QTYPES, build_sequence, collate_items, temp_bucket
    max_len, hml = cfg.get("max_len", 1024), cfg.get("head_max_len", 256)
    items, keys = [], []
    skipped = []
    for qid, q in questions.items():
        qtype, crit = q["type"], q.get("criteria", {})
        labels = list(crit.keys()) if qtype == "choice" else ["false", "true"]
        seq, markers = build_sequence(
            tok, state_text, {"t": qtype, "ins": q["instructions"], "crit": crit},
            max_len, hml)
        if len(markers) != len(labels):
            skipped.append(qid)
            continue
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES[qtype]})
        keys.append((qid, labels))
    out = {}
    with torch.no_grad():
        for bi in range(0, len(items), batch):
            sel, skeys = items[bi:bi + batch], keys[bi:bi + batch]
            b = collate_items([sel], tok.pad_token_id)
            if device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits, _ = model(b["input_ids"].to(device),
                                      b["attention_mask"].to(device),
                                      b["marker_pos"].to(device),
                                      b["marker_mask"].to(device),
                                      b["qtype"].to(device))
            else:
                logits, _ = model(b["input_ids"].to(device),
                                  b["attention_mask"].to(device),
                                  b["marker_pos"].to(device),
                                  b["marker_mask"].to(device),
                                  b["qtype"].to(device))
            lg = logits.float().cpu().numpy()
            mm = b["marker_mask"].numpy()
            qt = b["qtype"].numpy()
            from laya.common import QTYPE_NAMES
            for j, (qid, kk) in enumerate(skeys):
                qtn = QTYPE_NAMES[int(qt[j])]
                k = int(mm[j].sum())
                t = TBO.get(temp_bucket(int(qt[j]), k), TEMPERATURE[int(qt[j])])
                z = lg[j, :k] / t
                z = z - z.max()
                e = np.exp(z)
                p = e / e.sum()
                out[qid] = {kk[i]: float(p[i]) for i in range(k)}
                out[qid]["_qtype"] = qtn
            del b, logits
            if device.type == "cuda":
                import torch as _t
                _t.cuda.empty_cache()
    return out, skipped


# ---------- odds live (ver odds_scrape.py) ----------


def parse_manual_odds(s):
    """'1x2:H,D,A;ou25:O,U' -> {qid: {outcome: odd}}. Para testes sem casas."""
    if not s:
        return None
    out = {}
    parts = {}
    for part in s.split(";"):
        k, v = part.split(":", 1)
        parts[k.strip()] = [float(x) for x in v.split(",")]
    if parts.get("1x2") and len(parts["1x2"]) == 3:
        h, d, a = parts["1x2"]
        out["match_result"] = {"home": h, "draw": d, "away": a}
        out["home_win"] = {"true": h}
        out["draw"] = {"true": d}
        out["away_win"] = {"true": a}
    if parts.get("ou25") and len(parts["ou25"]) == 2:
        out["over_25"] = {"true": parts["ou25"][0]}
        out["under_25"] = {"true": parts["ou25"][1]}
    return out


# ---------- EV ----------

def _novig(probs_dict):
    """Probs do mercado sem vig: 1/odd normalizado. So um lado -> 1/odd."""
    inv = {k: 1.0 / o for k, o in probs_dict.items() if o and o > 1.0}
    tot = sum(inv.values())
    if not tot:
        return {}
    if len(inv) == 1:
        return inv  # sem par: sem vig possivel, fica conservador
    return {k: v / tot for k, v in inv.items()}


def decide(probs, odds, min_edge, kelly, max_stake, casa, trust=0.7):
    """Edge por pergunta/lado. odds: {qid: {outcome: odd}} (formato odds_scrape).

    O modelo engana-se: p_usada = trust*p_modelo + (1-trust)*p_mercado(sem vig).
    trust=1.0 = comportamento antigo (confia cegamente); 0.7 = assume que em
    ~30% do desacordo com o mercado quem tem razao e o mercado. O edge e o
    Kelly usam SEMPRE a p_usada, nunca a crua.
    """
    bets = []
    for qid, pv in probs.items():
        if qid.startswith("_"):
            continue
        ov = odds.get(qid) or {}
        if not ov:
            continue
        pm = _novig(ov)
        cands = []
        if pv.get("_qtype") == "choice":
            for label, p in pv.items():
                if label.startswith("_") or label not in ov:
                    continue
                o = ov[label]
                if o and o > 1.0:
                    pu = trust * float(p) + (1.0 - trust) * pm.get(label, float(p))
                    cands.append((f"{qid}:{label}", float(p), pu, float(o)))
        else:
            for side in ("true", "false"):
                if side in pv and side in ov and ov[side] and ov[side] > 1.0:
                    pu = trust * float(pv[side]) + (1.0 - trust) * pm.get(
                        side, float(pv[side]))
                    cands.append((f"{qid}:{side}", float(pv[side]), pu,
                                  float(ov[side])))
        for mercado, p_model, p, o in cands:
            edge = p * o - 1.0
            if edge >= min_edge:
                f = edge / (o - 1.0) * kelly if o > 1.0 else 0.0
                bets.append({"casa": casa, "mercado": mercado,
                             "p_model": round(p_model, 4), "p": round(p, 4),
                             "odd": o, "edge": round(edge, 4),
                             "stake_u": round(max(0.0, min(f, max_stake)), 3)})
    return bets


# Jogos/dia por mes (Big5 juntas, 10 epocas FotMob 2015-2025). Serve para
# ritmar o teto mensal: reparte o orcamento pelos jogos que ainda vêm.
MONTHLY_RATE = {"01": 5.85, "02": 7.29, "03": 5.49, "04": 7.83,
                "05": 5.74, "06": 0.61, "07": 0.68, "08": 4.07,
                "09": 6.03, "10": 5.84, "11": 5.28, "12": 6.58}
BANK_FILE = LIVE_DIR / "bankroll.json"


def _bank_load(now):
    month = now.strftime("%Y-%m")
    try:
        b = json.loads(BANK_FILE.read_text(encoding="utf-8"))
        if b.get("month") == month:
            return b
    except (OSError, ValueError):
        pass
    return {"month": month, "turned": 0.0, "games_seen": 0,
            "games_with_bets": 0}


def _bank_save(b):
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = BANK_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(b, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(BANK_FILE)


def apply_monthly_budget(bets, cap, now, consume=True):
    """Teto de turnover mensal (unidades de banca). Ritmo por jogo esperado.

    Sem isto, ~185 jogos/mes x ~0.1u dariam ~1800% da banca. Aqui o mes tem
    `cap` unidades (defeito 4.0 = 400%) e cada jogo recebe
    restante/jogos_com_aposta_esperados. Dentro do jogo, as apostas entram
    por ordem de edge (as melhores primeiro); o resto fica como
    skipped_budget. Devolve (taken, info).
    """
    import calendar as _cal
    b = _bank_load(now)
    month = now.strftime("%Y-%m")
    days_left = _cal.monthrange(now.year, now.month)[1] - now.day + 1
    rate = MONTHLY_RATE.get(now.strftime("%m"), 5.0)
    seen, hit = b.get("games_seen", 0), b.get("games_with_bets", 0)
    hit_rate = (hit / seen) if seen >= 5 else 0.5
    exp_games = max(1.0, rate * days_left * hit_rate)
    remaining = max(0.0, cap - b.get("turned", 0.0))
    allowance = remaining / exp_games if exp_games else 0.0
    ordered = sorted(bets, key=lambda x: x["edge"], reverse=True)
    taken, used = [], 0.0
    for bt in ordered:
        if used + bt["stake_u"] <= allowance + 1e-9:
            taken.append(bt)
            used += bt["stake_u"]
    info = {"month": month, "cap": cap,
            "turned_before": round(b.get("turned", 0.0), 3),
            "allowance": round(allowance, 4),
            "expected_bet_games_left": round(exp_games, 1),
            "taken": len(taken), "skipped_budget": len(bets) - len(taken)}
    if consume:
        b["turned"] = round(b.get("turned", 0.0) + used, 4)
        b["games_seen"] = seen + 1
        b["games_with_bets"] = hit + (1 if taken else 0)
        _bank_save(b)
    info["turned_after"] = round(b.get("turned", 0.0) + (0.0 if consume else used), 4)
    return taken, info


def old_league_teams(hist, lg):
    """Equipas com jogos na liga na epoca 24/25 (ultima completa no espelho).

    Quem joga hoje e nao esta aqui subiu de divisao entretanto (ou voltou
    apos anos fora): dominou a divisao anterior, sem historico fiavel na
    atual. O modelo nao tem base para estes casos -> cortar.
    """
    out = set()
    for (l, team), rows in hist.items():
        if l != lg:
            continue
        n = sum(1 for r in rows
                if (r[0].year + 1 if r[0].month >= 8 else r[0].year) == 2025)
        if n >= 1:
            out.add(team)
    return out


# ---------- pipeline inteiro (um jogo) ----------

def run_pipeline(a, fx, model, tok, cfg, device, hist, pair, tables_order,
                 now, league_index=None, tag="live", consume=True):
    """Fluxo completo de UM jogo: dados live -> estado -> modelo -> odds -> EV.

    fx: {lg, mid, home, away, kickoff}. Devolve o registo de decisao (e grava
    em data/live/decisions.jsonl).
    """
    t_start = time.time()
    # Corte de recem-promovidos: sem jogos na liga em 24/25 -> fora, sem
    # previsao nem aposta. O modelo treinou em dinamicas da divisao atual;
    # quem dominou a divisao anterior chega sem historico fiavel.
    veterans = old_league_teams(hist, fx["lg"])
    promoted = [t for t in (normalize(fx["home"]), normalize(fx["away"]))
                if t not in veterans]
    if promoted:
        dec = {"ts": now.isoformat(), "tag": tag, "lg": fx["lg"],
               "home": fx["home"], "away": fx["away"],
               "kickoff_utc": fx["kickoff"].isoformat(),
               "status": "skipped_promoted", "promoted": promoted,
               "bets": [], "pipeline_secs": round(time.time() - t_start, 1)}
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        with open(LIVE_DIR / "decisions.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(dec, ensure_ascii=False) + "\n")
        print(f"[{tag}] SKIP {fx['lg']} {fx['home']}-{fx['away']}: "
              f"recem-promovido {promoted}", flush=True)
        return dec
    det = live_detail(fx["mid"])
    feat, xi_ok = feat_for(fx, hist, pair, tables_order, det, now)
    disp_home, disp_away = fx["home"], fx["away"]
    disp = {normalize(fx["home"]): disp_home, normalize(fx["away"]): disp_away}
    state = verbalize({"home": normalize(fx["home"]), "away": normalize(fx["away"]),
                       "lg": fx["lg"], "round": (det or {}).get("round"),
                       "kickoff": fx["kickoff"], "venue": None},
                      feat, disp)
    questions, xi_players = build_live_questions(disp_home, disp_away, det)
    probs, skipped = predict(model, tok, cfg, state, questions, device,
                             batch=a.batch)
    odds_by_casa, odds_errors = {}, {}
    if a.manual_odds:
        odds_by_casa["manual"] = parse_manual_odds(a.manual_odds) or {}
    else:
        from odds_scrape import get_odds_for_fixture  # noqa
        odds_by_casa, odds_errors = get_odds_for_fixture(
            fx["lg"], fx["home"], fx["away"], tuple(a.bookmakers),
            league_index)
    bets = []
    for casa, odds in odds_by_casa.items():
        bets += decide(probs, odds or {}, a.min_edge, a.kelly,
                       a.max_stake, casa, a.trust)
    bets, budget = apply_monthly_budget(bets, a.max_monthly_turnover, now,
                                        consume)
    dt = time.time() - t_start
    dec = {
        "ts": now.isoformat(), "tag": tag, "lg": fx["lg"],
        "home": fx["home"], "away": fx["away"],
        "kickoff_utc": fx["kickoff"].isoformat(),
        "lineup_confirmed": xi_ok, "xi_players": xi_players,
        "n_questions": len(questions), "skipped": skipped,
        "probs": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                   for k, v in probs.items()},
        "odds": odds_by_casa, "odds_errors": odds_errors,
        "bets": bets, "budget": budget, "pipeline_secs": round(dt, 1),
    }
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    with open(LIVE_DIR / "decisions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(dec, ensure_ascii=False) + "\n")
    mr = probs.get("match_result", {})
    ou = probs.get("over_25", {})
    print(f"[{tag}] {fx['lg']} {fx['home']}-{fx['away']} "
          f"[{'XI-OK' if xi_ok else 'XI-pendente'}] q={len(questions)} "
          f"skip={len(skipped)} "
          f"1x2={[round(mr.get(k, float('nan')), 3) for k in ('home', 'draw', 'away')]} "
          f"over25={round(ou.get('true', float('nan')), 3)} "
          f"odds_casas={sorted(odds_by_casa)} "
          f"bets={len(bets)} {dt:.0f}s", flush=True)
    for b in bets:
        print(f"[{tag}]   APOSTA {b['casa']} {b['mercado']} "
              f"p_model={b['p_model']} p_usada={b['p']} odd={b['odd']} "
              f"edge={b['edge']} stake={b['stake_u']}u", flush=True)
    if budget["skipped_budget"]:
        print(f"[{tag}]   teto mensal: {budget['skipped_budget']} cortadas "
              f"(allowance={budget['allowance']}u, usado {budget['turned_before']}/"
              f"{budget['cap']}u em {budget['month']})", flush=True)
    return dec


# ---------- scan ----------

def _ensure_model(a):
    if getattr(a, "_model", None) is not None:
        return a._model, a._tok, a._cfg, a._device
    import torch
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    md = a.model_dir or next((str(p) for p in MODEL_CANDIDATES if
                              (p / "model.safetensors").exists()), None)
    if not md:
        raise SystemExit("pesos n_2000 nao encontrados (usa --model-dir)")
    print(f"[live] a carregar {md} ...", flush=True)
    model, tok, cfg = load_model(md, device)
    try:
        torch.set_num_threads(max(1, torch.get_num_threads() or 4))
    except (AttributeError, RuntimeError):
        pass
    a._model, a._tok, a._cfg, a._device = model, tok, cfg, device
    return model, tok, cfg, device


def _league_index(leagues, bookmakers):
    """Indice {casa: {lg: [eventos]}} das paginas de liga (uma vez por ronda)."""
    from odds_scrape import (betano_league_events, betclic_league_events,  # noqa
                             pinnacle_league_odds)
    index = {}
    for casa in bookmakers:
        if casa == "ps3838":
            index[casa] = {}  # discovery por pesquisa, sem paginas de liga
            continue
        index[casa] = {}
        for lg in leagues:
            try:
                if casa == "pinnacle":
                    evs = pinnacle_league_odds(lg)
                else:
                    evs = (betano_league_events(lg) if casa == "betano"
                           else betclic_league_events(lg))
            except Exception as e:  # noqa - uma liga nao mata o scan
                print(f"[live] {casa}/{lg} falhou: {e}", flush=True)
                evs = []
            index[casa][lg] = evs
            print(f"[live] {casa}/{lg}: {len(evs)} eventos", flush=True)
    return index

def scan(a, model=None, tok=None, cfg=None, device=None):
    now = datetime.now(timezone.utc)
    # refresh rolling: ultimos 14 dias (resultados que o espelho ainda nao tem)
    for i in range(14, -1, -1):
        fotmob_day((now - timedelta(days=i)).date())
    from loaders import iter_fotmob_days  # noqa (pos-refresh)
    hist, pair, tables_order = build_history()
    _ = iter_fotmob_days  # (import valida o espelho)
    print(f"[live] historico: {sum(len(v) for v in hist.values())} jogos-equipa",
          flush=True)

    # indice de odds por casa+liga (5 paginas por casa, uma vez por scan)
    index = {} if a.manual_odds else _league_index(
        set(a.leagues or list(BIG5.values())), a.bookmakers)

    model, tok, cfg, device = _ensure_model(a)
    print(f"[live] modelo OK device={device}", flush=True)

    fixtures = upcoming_fixtures(a.hours_ahead, a.leagues)
    print(f"[live] fixtures proximas {a.hours_ahead}h: {len(fixtures)}", flush=True)
    decisions = []
    for fx in fixtures:
        if fx["kickoff"] < now + timedelta(minutes=30):
            continue  # a comecar: sem tempo de apostar
        try:
            decisions.append(run_pipeline(a, fx, model, tok, cfg, device,
                                          hist, pair, tables_order, now,
                                          index, tag="live"))
        except Exception as e:  # noqa - um jogo nao mata o scan
            print(f"[live] {fx['home']}-{fx['away']} falhou: {e}", flush=True)
    n_bets = sum(len(d["bets"]) for d in decisions)
    print(f"[live] DONE fixtures={len(decisions)} apostas={n_bets} "
          f"-> {LIVE_DIR / 'decisions.jsonl'}", flush=True)
    return decisions


# ---------- serve ----------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._send({"ok": True, "ts": datetime.now(timezone.utc).isoformat()})
        if self.path == "/decisions":
            p = LIVE_DIR / "decisions.jsonl"
            rows = []
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines()[-200:]:
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        pass
            return self._send(rows)
        return self._send({"error": "usa /health ou /decisions"}, 404)


def serve(a):
    # Modelo carregado UMA vez; as rondas reutilizam-no.
    _ensure_model(a)
    threading.Thread(target=_loop, args=(a,), daemon=True).start()
    srv = HTTPServer(("127.0.0.1", a.port), Handler)
    print(f"[live] serve na porta {a.port} (/health /decisions)", flush=True)
    srv.serve_forever()


def _loop(a):
    while True:
        try:
            scan(a)
        except Exception as e:  # noqa - servidor nao pode cair
            print(f"[live] scan falhou: {e}", flush=True)
        time.sleep(a.interval)


# ---------- schedule (refresh horario + trigger T-20min) ----------

SCHED_FILE = LIVE_DIR / "schedule.json"


def _load_sched():
    try:
        return json.loads(SCHED_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"games": {}, "done": {}}


def _save_sched(st):
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = SCHED_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(SCHED_FILE)


def _refresh_schedule(st, leagues):
    """Fixtures hoje+amanha (48h). Deteta novos, mudancas de hora, saidos."""
    now = datetime.now(timezone.utc)
    fixtures = upcoming_fixtures(48, leagues)
    seen, added, moved, gone = set(), 0, [], 0
    for fx in fixtures:
        key = f"{fx['lg']}:{fx['mid']}"
        seen.add(key)
        ko = fx["kickoff"].isoformat()
        g = st["games"].get(key)
        if g is None:
            st["games"][key] = {"lg": fx["lg"], "mid": fx["mid"],
                                "home": fx["home"], "away": fx["away"],
                                "kickoff_utc": ko}
            added += 1
        elif g["kickoff_utc"] != ko:
            moved.append(f"{fx['home']}-{fx['away']}: {g['kickoff_utc']} -> {ko}")
            g["kickoff_utc"] = ko
    for key in [k for k in st["games"] if k not in seen]:
        gone += 1
        del st["games"][key]
        st["done"].pop(key, None)
    print(f"[sched] refresh: {len(fixtures)} fixtures ({added} novos, "
          f"{len(moved)} mudados, {gone} saidos)", flush=True)
    for m in moved:
        print(f"[sched] hora mudou: {m}", flush=True)
    return fixtures


def schedule_loop(a):
    """Loop nonstop: refresh horario da agenda + pipeline inteiro a T-20min."""
    model, tok, cfg, device = _ensure_model(a)
    st = _load_sched()
    last_refresh, hist_cache = 0.0, None
    print(f"[sched] arranque: refresh={a.refresh}s tick={a.tick}s "
          f"trigger=T-{a.lead_minutes}min", flush=True)
    while True:
        try:
            now = datetime.now(timezone.utc)
            if time.monotonic() - last_refresh >= a.refresh or last_refresh == 0:
                for i in range(14, -1, -1):
                    fotmob_day((now - timedelta(days=i)).date())
                hist_cache = build_history()
                _refresh_schedule(st, a.leagues)
                last_refresh = time.monotonic()
                _save_sched(st)
            for key, g in list(st["games"].items()):
                try:
                    ko = datetime.fromisoformat(g["kickoff_utc"])
                except ValueError:
                    continue
                if st["done"].get(key) == g["kickoff_utc"]:
                    continue  # ja corrido para esta hora
                tmins = (ko - now).total_seconds() / 60
                if tmins <= a.lead_minutes and tmins > -120:
                    if now >= ko:
                        print(f"[sched] {g['home']}-{g['away']} ja comecou "
                              f"(atraso), a correr na mesma", flush=True)
                    print(f"[sched] TRIGGER T-{tmins:.0f}min: "
                          f"{g['home']}-{g['away']} ({g['lg']})", flush=True)
                    fx = {"lg": g["lg"], "mid": g["mid"], "home": g["home"],
                          "away": g["away"], "kickoff": ko}
                    try:
                        index = {} if a.manual_odds else _league_index(
                            {g["lg"]}, a.bookmakers)
                        run_pipeline(a, fx, model, tok, cfg, device,
                                     *hist_cache, now, index, tag="sched")
                        st["done"][key] = g["kickoff_utc"]
                    except Exception as e:  # noqa - falhou, tenta na proxima tick
                        print(f"[sched] pipeline falhou "
                              f"{g['home']}-{g['away']}: {e}", flush=True)
                    _save_sched(st)
                elif tmins <= -120:
                    st["done"][key] = g["kickoff_utc"]  # expirado, nao correr
                    _save_sched(st)
        except Exception as e:  # noqa - o relogio nao pode parar
            print(f"[sched] volta falhou: {e}", flush=True)
        time.sleep(a.tick)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("scan", "serve", "schedule"):
        s = sub.add_parser(name)
        s.add_argument("--model-dir", default=None)
        s.add_argument("--calibration-json", default=None)
        s.add_argument("--leagues", nargs="*", default=None)
        s.add_argument("--hours-ahead", type=int, default=72)
        s.add_argument("--min-edge", type=float, default=0.05)
        s.add_argument("--kelly", type=float, default=0.25)
        s.add_argument("--max-stake", type=float, default=0.05,
                       help="stake maxima em fracao de banca (Kelly ja devolve fracao)")
        s.add_argument("--bookmakers", nargs="*", default=["betclic"],
                       help="casas a usar (betclic)")
        s.add_argument("--manual-odds", default=None,
                       help="'1x2:H,D,A;ou25:O,U' (testes sem casas)")
        s.add_argument("--batch", type=int, default=8,
                       help="micro-batch do forward (CPU: 8; T4: 16)")
        s.add_argument("--trust", type=float, default=0.7,
                       help="humildade do modelo: p_usada = trust*p_modelo + "
                            "(1-trust)*p_mercado (1.0 = confia cegamente)")
        s.add_argument("--max-monthly-turnover", type=float, default=4.0,
                       help="teto mensal apostado, em bancas (4.0 = 400%%)")
        s.add_argument("--port", type=int, default=8080)
        s.add_argument("--interval", type=int, default=900)
        s.add_argument("--refresh", type=int, default=3600,
                       help="schedule: de quanto em quanto reler fixtures (s)")
        s.add_argument("--tick", type=int, default=60,
                       help="schedule: passo do relogio (s)")
        s.add_argument("--lead-minutes", type=int, default=20,
                       help="schedule: correr o pipeline a T-Xmin")
        s.add_argument("--test-run", action="store_true",
                       help="schedule: corre ja o pipeline do proximo jogo e sai")
    a = ap.parse_args()

    global TEMPERATURE, TBO
    if a.calibration_json:
        m = json.loads(Path(a.calibration_json).read_text(encoding="utf-8"))
        TEMPERATURE = [float(x) for x in m["temperature"]]
        TBO = {k: float(v) for k, v in m["temperature_by_options"].items()}
    else:
        TEMPERATURE, TBO = list(DEFAULT_TEMPERATURE), dict(DEFAULT_TBO)

    if a.cmd == "scan":
        scan(a)
    elif a.cmd == "serve":
        serve(a)
    elif a.cmd == "schedule":
        if a.test_run:
            # valida o pipeline inteiro ja, sem esperar pelo relogio
            model, tok, cfg, device = _ensure_model(a)
            now = datetime.now(timezone.utc)
            for i in range(14, -1, -1):
                fotmob_day((now - timedelta(days=i)).date())
            hist = build_history()
            fixtures = upcoming_fixtures(a.hours_ahead, a.leagues)
            upcoming = sorted(
                [f for f in fixtures if f["kickoff"] > now],
                key=lambda f: f["kickoff"])
            if not upcoming:
                raise SystemExit("sem fixtures futuras para test-run")
            fx = upcoming[0]
            print(f"[sched] TEST-RUN: {fx['home']}-{fx['away']} "
                  f"({fx['lg']}) ko={fx['kickoff'].isoformat()}", flush=True)
            index = {} if a.manual_odds else _league_index(
                {fx["lg"]}, a.bookmakers)
            run_pipeline(a, fx, model, tok, cfg, device, *hist, now,
                         index, tag="test", consume=False)
        else:
            schedule_loop(a)
    else:
        serve(a)


if __name__ == "__main__":
    main()

# README_LIVE (contrato das fontes)
# - Estado: FotMob tem tudo o que o state_text precisa (fixtures, resultados,
#   lineups confirmados ~1h antes, h2h, tabela). O rolling vem do espelho
#   data/raw (mesmos ficheiros do treino) + refresh dos ultimos 14 dias a
#   cada scan. Sem XI confirmado o modelo corre na mesma (frases do XI
#   omitidas, como no treino quando xi=None); os mercados EV (1x2, OU2.5)
#   nao precisam de nomes de jogadores.
# - Limpeza conhecida: top_scorer_* vem a None live (exigia goals de todos
#   os detalhes da epoca). Custa 1 frase de contexto; o grueso (forma,
#   tabela, h2h, XI) esta la.
# - Odds: Betclic + Betano .pt (HTML server-rendered: 1X2 nos cartoes de
#   liga, totais 2.5 nas paginas de evento). Sem auth, sem key, sem browser.
#   Matching por normalize() + substring; jogos nao listados ficam com
#   odds_errors (sem aposta). Sem key: --manual-odds ou modo sem-aposta
#   (mostra probs + lineup status).
# - Anti-leakage: cutoff = agora; odds nunca entram no state_text.
