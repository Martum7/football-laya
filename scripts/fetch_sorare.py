"""Sorare GraphQL: scores por jogador (precisa SORARE_API_KEY).

Limite da key: 200 req/min -> bucket 3.0/s partilhado entre workers.
Modos:
  python3 fetch_sorare.py --slugs-file ../../data/processed/players_wanted.jsonl
  python3 fetch_sorare.py --wait   # corre em background a espera da lista
                                   # (produzida quando o build correr)

Formato players_wanted.jsonl: {"slug": "harry-kane", "name": "Harry Kane"}
Saida: data/raw/sorare/player_{slug}.json (ultimos 20 game scores).
Referencia: https://github.com/sorare/api/ (endpoint https://api.sorare.com/graphql)
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import TokenBucket, already, load_dotenv  # noqa: E402
from config import LOGS, PROCESSED, RATE, RAW  # noqa: E402

load_dotenv()

GQL = "https://api.sorare.com/graphql"
OUT = RAW / "sorare"
_bucket = TokenBucket(RATE["sorare_rps"])

QUERY = """query($slug: String!, $first: Int!, $after: String) {
  football {
    player(slug: $slug) {
      displayName
      position
      playerGameScores(first: $first, after: $after) {
        pageInfo { hasNextPage endCursor }
        nodes {
          score
          scoringVersion
          game { date }
        }
      }
    }
  }
}"""


def fetch_all_pages(slug, max_pages=3, per_page=50):
    """Pagina para tras no tempo (cobrir jogos antigos, nao so last 20).
    O filtro game.date < cutoff faz-se no BUILD, aqui guarda-se tudo."""
    after = None
    nodes = []
    for _ in range(max_pages):
        res = gql(QUERY, {"slug": slug, "first": per_page, "after": after})
        if "_error" in res:
            return {"_error": res["_error"]}
        try:
            conn = res["data"]["football"]["player"]["playerGameScores"]
        except (KeyError, TypeError):
            # fallback: API antiga so com last:N
            res_old = gql(QUERY_LAST, {"slug": slug})
            if "_error" in res_old:
                return {"_error": res_old["_error"]}
            return res_old
        nodes.extend(conn.get("nodes", []))
        pi = conn.get("pageInfo", {})
        if not pi.get("hasNextPage"):
            break
        after = pi.get("endCursor")
    return {"slug": slug, "scores": nodes}


QUERY_LAST = """query($slug: String!) {
  football {
    player(slug: $slug) {
      displayName
      position
      playerGameScores(last: 50) {
        score
        scoringVersion
        game { date }
      }
    }
  }
}"""


def gql(query, variables):
    key = os.environ.get("SORARE_API_KEY", "")
    if not key:
        raise SystemExit("Falta SORARE_API_KEY no ambiente.")
    _bucket.wait()
    body = json.dumps({"query": query, "variables": variables}).encode()
    last = None
    for attempt in range(4):
        req = urllib.request.Request(GQL, data=body, headers={
            "Content-Type": "application/json",
            "APIKEY": key,
            "User-Agent": "football-laya",
        })
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            last = getattr(e, "code", None) or str(e)
            time.sleep(min(2 ** attempt, 8))
    return {"_error": str(last)}


def fetch_slug(slug, max_pages=3):
    dest = OUT / f"player_{slug}.json"
    if already(dest):
        return "skip"
    res = fetch_all_pages(slug, max_pages=max_pages)
    if "_error" in res:
        print(f"[sorare] FALHOU {slug}: {res['_error']}", flush=True)
        return "falha"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    n = len(res.get("scores", []))
    return "ok" if n else "sem-jogos"


def load_slugs(path):
    slugs = []
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    slugs.append(json.loads(line)["slug"])
                except (KeyError, ValueError):
                    pass
    return slugs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slugs-file",
                    default=str(PROCESSED / "players_wanted.jsonl"))
    ap.add_argument("--wait", action="store_true",
                    help="espera pela lista em vez de sair se vazia")
    ap.add_argument("--workers", type=int, default=RATE["sorare_workers"])
    ap.add_argument("--poll-secs", type=int, default=120)
    ap.add_argument("--max-pages", type=int, default=3,
                    help="paginas de ~50 scores por jogador (para tras no tempo)")
    a = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    while True:
        slugs = load_slugs(a.slugs_file)
        missing = [s for s in slugs
                    if not already(OUT / f"player_{s}.json")]
        print(f"[sorare] wanted={len(slugs)} em-falta={len(missing)}", flush=True)
        if missing:
            if not os.environ.get("SORARE_API_KEY"):
                print("[sorare] PARADO: mete a key em ../.env (SORARE_API_KEY=...)",
                      flush=True)
            else:
                with ThreadPoolExecutor(max_workers=a.workers) as ex:
                    res = list(ex.map(
                        lambda s: fetch_slug(s, max_pages=a.max_pages), missing))
                from collections import Counter
                print("[sorare] resultado:", dict(Counter(res)), flush=True)
        if not a.wait:
            break
        time.sleep(a.poll_secs)
