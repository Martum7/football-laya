"""ESPN football Big5: scoreboard mensal -> summary por evento.

Endpoints publicos, sem auth, funcionam de qualquer rede:
  GET .../soccer/{liga}/scoreboard?dates=YYYYMM   -> eventos do mes
  GET .../soccer/{liga}/summary?event={id}        -> boxscore, odds Bet365,
     keyEvents, lastFiveGames, rosters, standings, gameInfo
Ligas: eng.1 esp.1 ger.1 ita.1 fra.1. Meses 2015-08 -> 2025-07 (120x5).
Uso:
  python3 fetch_espn.py --probe
  nohup python3 fetch_espn.py > ../data/logs/espn.log 2>&1 &
"""
import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import TokenBucket, already, append_manifest, fetch_text  # noqa: E402
from config import LOGS, RATE, RAW  # noqa: E402

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
OUT = RAW / "espn"
MANIFEST = OUT / "_manifest.jsonl"

LEAGUES = {"ENG": "eng.1", "ESP": "esp.1", "GER": "ger.1",
           "ITA": "ita.1", "FRA": "fra.1"}

_bucket = TokenBucket(2.0)
_print_lock = threading.Lock()


def _log(msg):
    with _print_lock:
        print(msg, flush=True)


def months(a_y, a_m, b_y, b_m):
    y, m = a_y, a_m
    while (y, m) <= (b_y, b_m):
        yield y, m
        m += 1
        if m > 12:
            m, y = 1, y + 1


def scoreboard(lg, y, m):
    p = OUT / "months" / f"{lg}_{y}{m:02d}.json"
    if already(p):
        return json.loads(p.read_text(encoding="utf-8"))
    _bucket.wait()
    code, text = fetch_text(f"{BASE}/{LEAGUES[lg]}/scoreboard?dates={y}{m:02d}")
    if code != 200 or not text or not text.startswith("{"):
        _log(f"[espn] {lg} {y}-{m:02d} FALHOU {code}")
        return {}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return json.loads(text)


def summary(lg, eid):
    p = OUT / f"event_{eid}.json"
    if already(p):
        return "skip"
    _bucket.wait()
    code, text = fetch_text(f"{BASE}/{LEAGUES[lg]}/summary?event={eid}",
                            timeout=30)
    if code != 200 or not text or not text.startswith("{"):
        return f"falha {code}"
    p.write_text(text, encoding="utf-8")
    append_manifest(MANIFEST, {"lg": lg, "id": eid, "bytes": len(text)})
    return "ok"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues", nargs="*", default=list(LEAGUES))
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)

    if a.probe:
        d = scoreboard("ENG", 2024, 9)
        evs = d.get("events", [])
        print(f"[espn] probe ENG 2024-09: {len(evs)} eventos", flush=True)
        if evs:
            print("[espn] summary:", summary("ENG", evs[0]["id"]), flush=True)
        raise SystemExit(0)

    jobs = [(lg, y, m) for lg in a.leagues
            for y, m in months(2015, 8, 2025, 7)]
    with ThreadPoolExecutor(max_workers=2) as ex:
        boards = list(ex.map(lambda j: (j, scoreboard(*j)), jobs))
    ids = []
    for (lg, _, _), b in boards:
        for e in b.get("events", []):
            ids.append((lg, e["id"]))
    ids = sorted(set(ids))
    print(f"[espn] eventos unicos: {len(ids)}, a sacar summaries...", flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(lambda t: summary(*t), ids))
    from collections import Counter
    print("[espn] resultado:", dict(Counter(res)), flush=True)
