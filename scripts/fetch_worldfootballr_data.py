"""Espelho dos CSVs worldfootballR_data (GitHub) filtrados para Big5.

Nao faz scrape: lista a arvore via GitHub API e saca os CSVs em paralelo.
Resume por ficheiro existente. Max anos = tudo o que existir no repo.
Uso:
  nohup python3 fetch_worldfootballr_data.py > ../data/logs/wfr.log 2>&1 &
"""
import argparse
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import already, append_manifest  # noqa: E402
from config import LOGS, RATE, RAW  # noqa: E402

API_ROOT = "https://api.github.com/repos/JaseZiv/worldfootballR_data/contents"
API = API_ROOT + "/data"
RAWBASE = "https://raw.githubusercontent.com/JaseZiv/worldfootballR_data/master/data"
OUT = RAW / "worldfootballR_data"
MANIFEST = OUT / "_manifest.jsonl"

# keywords para filtrar Big5 (nomes de ficheiro/pasta variam; propositadamente largo)
KEEP = ["eng", "epl", "premier", "esp", "laliga", "la_liga", "ger", "bundesliga",
        "ita", "serie", "fra", "ligue", "big5", "match_results", "fotmob"]


def gh_get(url):
    token = os.environ.get("GITHUB_TOKEN")
    h = {"User-Agent": "football-laya", "Accept": "application/vnd.github+json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def walk(path="data"):
    """Devolve lista de (git_path, download_url) para .csv: 1 so chamada
    a git trees recursive (evita rate-limit da contents API)."""
    trees_url = ("https://api.github.com/repos/JaseZiv/"
                 "worldfootballR_data/git/trees/master?recursive=1")
    try:
        tree = gh_get(trees_url)
    except Exception as e:
        print(f"[wfr] git-trees falhou: {e}", flush=True)
        return []
    files = []
    for it in tree.get("tree", []):
        p = it.get("path", "")
        if it.get("type") == "blob" and p.endswith(".rds") and p.startswith("data/"):
            files.append((p, RAWBASE + "/" + p.replace("data/", "", 1)))
    return files


def wanted(git_path):
    """Filtro Big5. NOTA: fb_*_season_stats sao TOTAIS finais de epoca ->
    proibidos como feature direta (leakage); guardamos para baselines."""
    parts = git_path.split("/")
    if len(parts) < 2 or parts[1] not in (
            "match_results", "fotmob_match_details", "understat_shots",
            "tm_player_vals", "tm_transfers", "fb_big5_advanced_season_stats",
            "fb_match_summary", "fb_match_shooting"):
        return False
    if parts[1] == "match_results":
        return any(c in parts[2] for c in
                   ["ENG", "ESP", "GER", "ITA", "FRA"])
    return True


def fetch_one(args):
    git_path, url = args
    if not url:
        return "sem-url"
    rel = git_path.replace("data/", "", 1)
    dest = OUT / rel
    if already(dest):
        return "skip"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "football-laya"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(dest)
        append_manifest(MANIFEST, {"file": rel, "bytes": len(data)})
        return "ok"
    except Exception as e:
        return f"erro {e}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=RATE["wfr_workers"])
    ap.add_argument("--list-only", action="store_true")
    a = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    print("[wfr] a listar arvore data/ ...", flush=True)
    all_files = walk()
    print(f"[wfr] ficheiros .csv no repo: {len(all_files)}", flush=True)
    sel = [f for f in all_files if wanted(f[0])]
    print(f"[wfr] selecionados Big5: {len(sel)}", flush=True)
    if a.list_only:
        for p, _ in sel[:100]:
            print("  ", p)
    else:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            res = list(ex.map(fetch_one, sel))
        from collections import Counter
        print("[wfr] resultado:", dict(Counter(
            r if r in ("ok", "skip", "sem-url") else "erro" for r in res)), flush=True)
