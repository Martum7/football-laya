"""FotMob Big5: datas -> jogos -> detalhe completo por jogo.

Endpoints (publicos, sem auth, funcionam com curl normal):
  GET /api/data/matches?date=YYYYMMDD          -> ligas + jogos do dia
  GET /api/data/matchDetails?matchId={id}      -> facts, stats, playerStats,
     shotmap, lineup, table, h2h + header.events (golos/cartoes)
  GET /api/data/match?id={id}                  -> resumo rapido
  GET /api/data/teams?id={teamId}               -> fixtures/squad/stats

Backfill: se corre de 2015-08-01 a 2025-07-31 (~3650 dias) + detalhe
dos jogos Big5 (~19k). Com resume por ficheiro.
Uso:
  python3 fetch_fotmob.py --probe                 # 1 data + 2 detalhes
  nohup python3 fetch_fotmob.py > ../data/logs/fotmob.log 2>&1 &
"""
import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import TokenBucket, already, append_manifest, fetch_text  # noqa: E402
from config import LOGS, RATE, RAW  # noqa: E402

BASE = "https://www.fotmob.com/api/data"
OUT = RAW / "fotmob"
MANIFEST = OUT / "_manifest.jsonl"

_bucket = TokenBucket(1.5)  # conservador no arranque; subir se nao bloquear
_print_lock = threading.Lock()

# FotMob leagueId e unico (nomes colidem: ha varias "Ligue 1"/"Premier League").
# Descobertos via /api/data/matches?date=2024-09-01 (fim-de-semana de liga).
BIG5_LEAGUES = {
    47: "ENG",   # Premier League
    87: "ESP",   # LaLiga
    54: "GER",   # Bundesliga
    55: "ITA",   # Serie A
    53: "FRA",   # Ligue 1
}


def _log(msg):
    with _print_lock:
        print(msg, flush=True)


def day_matches(day):
    """day: date. Devolve dict leagues->matches (raw)."""
    p = OUT / "days" / f"{day:%Y%m%d}.json"
    if already(p):
        return json.loads(p.read_text(encoding="utf-8"))
    _bucket.wait()
    code, text = fetch_text(f"{BASE}/matches?date={day:%Y%m%d}")
    if code != 200 or not text or not text.startswith("{"):
        _log(f"[fotmob] dia {day} FALHOU status={code}")
        return {}
    d = json.loads(text)
    pp = OUT / "days"
    pp.mkdir(parents=True, exist_ok=True)
    (pp / f"{day:%Y%m%d}.json").write_text(text, encoding="utf-8")
    return d


def big5_ids(day_data):
    """Extrai (match_id, league, home, away) Big5 do JSON do dia."""
    out = []
    for lg in day_data.get("leagues", []):
        code = BIG5_LEAGUES.get(lg.get("id"))
        if not code:
            continue
        for m in lg.get("matches", []):
            out.append((m["id"], code,
                        m.get("home", {}).get("name"),
                        m.get("away", {}).get("name")))
    return out


def details(mid):
    p = OUT / f"match_{mid}.json"
    if already(p):
        return "skip"
    _bucket.wait()
    code, text = fetch_text(f"{BASE}/matchDetails?matchId={mid}", timeout=30)
    if code != 200 or not text or not text.startswith("{"):
        return f"falha {code}"
    p.write_text(text, encoding="utf-8")
    append_manifest(MANIFEST, {"id": mid, "bytes": len(text)})
    return "ok"


def daterange(a, b):
    d = a
    while d <= b:
        yield d
        d += timedelta(days=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="dfrom", default="2015-08-01")
    ap.add_argument("--to", dest="dto", default="2025-07-31")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)

    if a.probe:
        d = day_matches(date(2024, 10, 12))
        ids = big5_ids(d)
        print(f"[fotmob] probe 2024-10-12: ligas={len(d.get('leagues', []))} "
              f"jogos-big5={len(ids)}", flush=True)
        for mid, lg, h, aw in ids[:4]:
            print(f"  {mid} [{lg}] {h} - {aw}: {details(mid)}", flush=True)
        raise SystemExit(0)

    days = list(daterange(date.fromisoformat(a.dfrom), date.fromisoformat(a.dto)))
    print(f"[fotmob] dias: {len(days)}, a listar jogos...", flush=True)
    all_ids = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        for d in ex.map(day_matches, days):
            all_ids.extend(big5_ids(d))
    all_ids = sorted(set(all_ids))
    print(f"[fotmob] jogos Big5 unicos: {len(all_ids)}, a sacar detalhe...",
          flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(lambda t: details(t[0]), all_ids))
    from collections import Counter
    print("[fotmob] resultado:", dict(Counter(res)), flush=True)
