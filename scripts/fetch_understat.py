"""Understat Big5 2015..2024: HTML com JSON embebido -> raw + parsed.

Ligas: EPL, La_liga, Bundesliga, Serie_A, Ligue_1.
Uso:
  nohup python3 fetch_understat.py > ../data/logs/understat.log 2>&1 &
"""
import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import TokenBucket, already, fetch_text, save_text  # noqa: E402
from config import LEAGUES, LOGS, RATE, RAW, UNDERSTAT_SEASONS  # noqa: E402

OUT = RAW / "understat"
_bucket = TokenBucket(RATE["understat_rps"])


def fetch_league_season(lg, year):
    html_p = OUT / f"{lg}_{year}.html"
    json_p = OUT / f"{lg}_{year}.json"
    if already(json_p):
        return "skip"
    if not already(html_p):
        _bucket.wait()
        code, text = fetch_text(f"https://understat.com/league/{lg}/{year}")
        if code != 200 or not text or "datesData" not in text:
            print(f"[understat] FALHOU {lg} {year}: status={code}", flush=True)
            return "falha"
        save_text(html_p, text)
    else:
        text = html_p.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"var\s+datesData\s*=\s*JSON\.parse\('(.*?)'\)", text, re.S)
    if not m:
        print(f"[understat] sem datesData {lg} {year}", flush=True)
        return "sem-dados"
    try:
        data = json.loads(m.group(1).encode().decode("unicode_escape"))
    except Exception as e:
        print(f"[understat] parse falhou {lg} {year}: {e}", flush=True)
        return "parse-erro"
    save_text(json_p, json.dumps(data, ensure_ascii=False))
    print(f"[understat] ok {lg} {year} jogos={len(data)}", flush=True)
    return "ok"


if __name__ == "__main__":
    # DEPRECATED 2026-10-03: o Understat tirou o JSON embebido das paginas
    # de liga (o HTML vem sem datesData). Os shots por liga ja estao no
    # espelho worldfootballR_data/understat_shots/*.rds (fetch via
    # fetch_worldfootballr_data.py) e leem-se com Python 3.12 + pyreadr.
    print("[understat] DEPRECATED: usar worldfootballR_data/understat_shots/*.rds",
          flush=True)
    raise SystemExit(0)
