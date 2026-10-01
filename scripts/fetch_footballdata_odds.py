"""Odds historicas football-data.co.uk (1X2 + O/U). Pequeno e rapido.

https://www.football-data.co.uk/mmz4281/{epoca}/{codigo}.csv
Uso:
  python3 fetch_footballdata_odds.py --codes E0 SP1 D1 I1 F1 --seasons 2324 2425
"""
import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import already  # noqa: E402
from config import FD_SEASONS, LEAGUES, LOGS, RAW  # noqa: E402

OUT = RAW / "odds_footballdata"


def fetch(code, season):
    dest = OUT / f"fd_{code}_{season}.csv"
    if already(dest):
        print(f"[fd] skip {code} {season}", flush=True)
        return
    url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "football-laya"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        print(f"[fd] ok {code} {season} ({len(data)} bytes)", flush=True)
    except Exception as e:
        print(f"[fd] FALHOU {code} {season}: {e}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", nargs="*", default=None)
    ap.add_argument("--seasons", nargs="*", default=None)
    a = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    codes = a.codes or [c for lg in LEAGUES.values() for c in lg["fd_codes"]]
    seasons = a.seasons or FD_SEASONS
    for se in seasons:
        for co in codes:
            fetch(co, se)
    print("[fd] DONE", flush=True)
