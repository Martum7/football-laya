"""Leitores compactos das 4 fontes. Nada de pandas aqui (so stdlib);
o parquet final e feito no build_dataset (corre no .venv).
"""
import csv
import glob
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import RAW  # noqa: E402
from teams import normalize  # noqa: E402

FM = RAW / "fotmob"
ES = RAW / "espn"
FD = RAW / "odds_footballdata"
WFR = RAW / "worldfootballR_data"

FM_BIG5 = {47: "ENG", 87: "ESP", 54: "GER", 55: "ITA", 53: "FRA"}
ESPN_LG = {"ENG": "eng.1", "ESP": "esp.1", "GER": "ger.1",
           "ITA": "ita.1", "FRA": "fra.1"}
FD_LG = {"ENG": ["E0"], "ESP": ["SP1"], "GER": ["D1"],
         "ITA": ["I1"], "FRA": ["F1"]}


def season_end_year(dt):
    return dt.year + 1 if dt.month >= 8 else dt.year


def iter_fotmob_days():
    for p in sorted((FM / "days").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        for lg in d.get("leagues", []):
            code = FM_BIG5.get(lg.get("id"))
            if not code:
                continue
            for m in lg.get("matches", []):
                yield {"src": "fotmob", "lg": code,
                       "mid": m["id"],
                       "home": m["home"]["name"], "away": m["away"]["name"],
                       "hg": m["home"].get("score"), "ag": m["away"].get("score"),
                       "ts": m.get("timeTS"),
                       "finished": (m.get("status") or {}).get("finished", False)}


def fotmob_detail(mid):
    """Devolve dict compacto do matchDetails (None se falhar)."""
    p = FM / f"match_{mid}.json"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    h, c = d.get("header", {}), d.get("content", {})
    status = h.get("status", {})
    out = {
        "kickoff": status.get("utcTime"),
        "hs": h.get("homeScore"), "as_": h.get("awayScore"),
        "red_h": status.get("numberOfHomeRedCards", 0),
        "red_a": status.get("numberOfAwayRedCards", 0),
    }
    g = (d.get("general") or {})
    out["round"] = g.get("matchRound")
    out["venue"] = ((g.get("homeTeam") or {}).get("name"), None)
    ln = (c.get("lineup") or {})
    out["lineup"] = {}
    for side, key in (("h", "homeTeam"), ("a", "awayTeam")):
        t = ln.get(key) or {}
        out["lineup"][side] = {
            "formation": t.get("formation"),
            "coach": (t.get("coach") or {}).get("name")
            if isinstance(t.get("coach"), dict) else t.get("coach"),
            "rating": t.get("rating"),
            "xi": [{"name": p_["name"], "rating": (p_.get("performance") or {}).get("rating"),
                    "mv": p_.get("marketValue")}
                   for p_ in (t.get("starters") or [])],
            "missing": [{"name": u.get("name"),
                         "why": (u.get("unavailability") or {}).get("type")}
                        for u in (t.get("unavailable") or [])],
            "subs": [s.get("name") for s in (t.get("subs") or []) if s.get("name")],
        }
    out["stats"] = {}
    for per in ("All", "FirstHalf", "SecondHalf"):
        blk = ((c.get("stats") or {}).get("Periods") or {}).get(per) or {}
        vals = {}
        for grp in blk.get("stats", []):
            for s in grp.get("stats", []):
                v = s.get("stats")
                if isinstance(v, list) and len(v) == 2:
                    vals[s.get("key")] = v
        out["stats"][per] = vals
    goals = []
    evs = h.get("events") or {}
    for side, key in (("h", "homeTeamGoals"), ("a", "awayTeamGoals")):
        for _, lst in (evs.get(key) or {}).items():
            for e in lst or []:
                goals.append({"side": side, "min": e.get("time"),
                              "minstr": str(e.get("timeStr", "")),
                              "player": ((e.get("player") or {}).get("name")),
                              "pen": "Penalty" in str(e.get("goalDescriptionKey", "")),
                              "owngoal": bool(e.get("ownGoal"))})
    out["goals"] = sorted(goals, key=lambda e: (e["min"] or 0))
    out["h2h"] = (c.get("h2h") or {}).get("summary")
    return out


def iter_espn_months(now=None):
    from datetime import datetime as _dt
    now = now or _dt.now(timezone.utc)
    for p in sorted((ES / "months").glob("*.json")):
        lg = p.stem.split("_")[0]
        d = json.loads(p.read_text(encoding="utf-8"))
        for e in d.get("events", []):
            comp = (e.get("competitions") or [{}])[0]
            cs = comp.get("competitors", [])
            if len(cs) < 2:
                continue
            home = next((x for x in cs if x.get("homeAway") == "home"), cs[0])
            away = next((x for x in cs if x.get("homeAway") == "away"), cs[-1])
            hg, ag = _int(home.get("score")), _int(away.get("score"))
            try:
                dt = _dt.fromisoformat((e.get("date") or "").replace("Z", "+00:00"))
            except ValueError:
                dt = None
            # ESPN marca alguns acabados como completed=False: score valido
            # + data ha +3h conta como terminado (regra explicita no README).
            flag = (e.get("status", {}).get("type") or {}).get("completed", False)
            done = bool(flag) or (hg is not None and ag is not None
                                  and dt is not None and dt < now - timedelta(hours=3))
            yield {"src": "espn", "lg": lg, "eid": str(e.get("id")),
                   "home": home["team"]["displayName"],
                   "away": away["team"]["displayName"],
                   "hg": hg, "ag": ag,
                   "date": e.get("date"),
                   "finished": done}


def _int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def iter_fd():
    for p in sorted(FD.glob("fd_*.csv")):
        code, season = p.stem.split("_")[1], p.stem.split("_")[2]
        lg = next((k for k, v in FD_LG.items() if code in v), None)
        with open(p, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                if not row.get("HomeTeam") or row.get("FTHG") in ("", None):
                    continue
                yield {"src": "fd", "lg": lg, "code": code, "season": season,
                       **row}


def parse_fd_date(row):
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(row["Date"].strip(), fmt).replace(
                tzinfo=timezone.utc)
        except (ValueError, KeyError):
            continue
    return None
