"""Sofascore Big5: seasons (max disponivel) -> rounds -> events -> details.

Padrao portado de basket_predict/nba_monitor/scrapers/sofascore.py:
headers simples (Windows UA + Referer + Accept json), 0.6s entre requests,
coleta por intervalo de datas + filtro de torneio, resume por ficheiro.

NOTA DE REDE: api.sofascore.com da 403 Cloudflare a partir de IPs
marcados (ex: este proxy). Da rede de casa funciona com estes headers.
Detalhe por jogo via FotMob (fetch_fotmob.py) e o caminho que funciona
de qualquer rede; este fetcher e o caminho rico (statistics/lineups/odds)
para correr da rede de casa.
"""
import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import TokenBucket, already, append_manifest, save_text  # noqa: E402
from config import LEAGUES, LOGS, RATE, RAW  # noqa: E402

BASE = "https://api.sofascore.com/api/v1"
OUT = RAW / "sofascore"
MANIFEST = OUT / "_manifest.jsonl"

# Headers do basket_predict (provados na rede de casa)
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"),
    "Referer": "https://www.sofascore.com/",
    "Accept": "application/json",
}

# uniqueTournament.id por liga (filtro tipo is_nba do basket: so estes passam)
BIG5_UT_IDS = {17, 8, 54, 55, 34}  # EPL, LaLiga, Bundesliga, Serie A, Ligue 1

_bucket = TokenBucket(RATE["sofascore_rps"])
_print_lock = threading.Lock()


def _log(msg):
    with _print_lock:
        print(msg, flush=True)


def _get_plain(url, timeout=25):
    """GET simples estilo basket_predict (funciona da rede de casa)."""
    _bucket.wait()
    import urllib.request
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except Exception as e:
        return getattr(e, "code", str(e)), None


def _get(url, timeout=25):
    code, text = _get_plain(url, timeout)
    if code == 403:
        # retry via curl_cffi (impersonation) antes de desistir
        _bucket.wait()
        try:
            from curl_cffi import requests as cr
            r = cr.get(url, impersonate="chrome", timeout=timeout,
                       headers={"Accept": "application/json"}, verify=False)
            return r.status_code, r.text if r.status_code == 200 else None
        except Exception as e:
            return f"cf-{e}", None
    return code, text


def seasons_for(ut):
    p = OUT / f"seasons_ut{ut}.json"
    if already(p):
        return json.loads(p.read_text(encoding="utf-8"))
    code, text = _get(f"{BASE}/unique-tournament/{ut}/seasons")
    if code != 200 or not text:
        _log(f"[seasons ut={ut}] FALHOU status={code} "
             f"(rede com IP marcado? da rede de casa funciona)")
        return {}
    save_text(p, text)
    append_manifest(MANIFEST, {"kind": "seasons", "ut": ut})
    return json.loads(text)


def round_events(ut, season_id, rnd):
    p = OUT / f"events_ut{ut}_s{season_id}_r{rnd}.json"
    if already(p):
        return json.loads(p.read_text(encoding="utf-8"))
    code, text = _get(
        f"{BASE}/unique-tournament/{ut}/season/{season_id}/events/round/{rnd}")
    if code != 200 or not text:
        return {"events": []}
    save_text(p, text)
    return json.loads(text)


def event_details(eid):
    out = {}
    for kind in ("info", "incidents", "statistics", "lineups", "odds"):
        p = OUT / f"event_{eid}_{kind}.json"
        if already(p):
            out[kind] = True
            continue
        url = f"{BASE}/event/{eid}" if kind == "info" else f"{BASE}/event/{eid}/{kind}"
        code, text = _get(url)
        if code == 200 and text:
            save_text(p, text)
            out[kind] = True
        else:
            out[kind] = False  # 404 em odds antigas e normal
    append_manifest(MANIFEST, {"kind": "event", "id": eid, **out})
    return out


def is_big5(event):
    """Filtro tipo is_nba do basket: so Big5 passam (por uniqueTournament.id)."""
    try:
        return event.get("tournament", {}).get("uniqueTournament", {}).get("id") \
            in BIG5_UT_IDS
    except AttributeError:
        return False


def scheduled_day(day):
    """Jogos Big5 finalizados de um dia (date)."""
    p = OUT / "days" / f"{day:%Y%m%d}.json"
    if already(p):
        return json.loads(p.read_text(encoding="utf-8"))
    code, text = _get(f"{BASE}/sport/football/scheduled-events/{day:%Y-%m-%d}")
    if code != 200 or not text:
        _log(f"[sofascore] dia {day} FALHOU status={code}")
        return {"events": []}
    d = json.loads(text)
    (OUT / "days").mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return d


def collect_range(dfrom, dto, workers=3):
    """Coleta por datas (igual collect_games_range do basket): dias -> eventos
    -> detalhe (event, incidents, statistics, lineups)."""
    from datetime import timedelta
    day = dfrom
    all_ids = []
    while day <= dto:
        d = scheduled_day(day)
        for e in d.get("events", []):
            if is_big5(e) and e.get("status", {}).get("type") == "finished":
                all_ids.append(e["id"])
        day += timedelta(days=1)
    all_ids = sorted(set(all_ids))
    _log(f"[sofascore] eventos Big5 unicos: {len(all_ids)}")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda i: event_details(i), all_ids))


def backfill(leagues, max_round=38, workers=3, probe=False):
    for lg in leagues:
        ut = LEAGUES[lg]["sofascore_ut"]
        seasons = seasons_for(ut).get("seasons", [])
        _log(f"[{lg}] ut={ut} seasons={len(seasons)} "
             f"{[s.get('year') for s in seasons[:12]]}")
        if probe:
            if seasons:
                sid = seasons[0]["id"]
                ev = round_events(ut, sid, 1).get("events", [])
                _log(f"[{lg}] probe season={sid} round1 events={len(ev)}")
            continue
        jobs = [(ut, s["id"], r) for s in seasons for r in range(1, max_round + 1)]
        with ThreadPoolExecutor(max_workers=1) as ex:
            rounds = list(ex.map(lambda j: (j, round_events(*j)), jobs))
        eids = []
        for (_, events) in rounds:
            for e in events.get("events", []):
                if e.get("id"):
                    eids.append(e["id"])
        eids = sorted(set(eids))
        _log(f"[{lg}] total event_ids={len(eids)}; a sacar details com {workers} workers")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(event_details, eids))
        _log(f"[{lg}] DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues", nargs="*", default=list(LEAGUES),
                    choices=list(LEAGUES))
    ap.add_argument("--max-round", type=int, default=38)
    ap.add_argument("--workers", type=int, default=RATE["sofascore_workers"])
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--from", dest="dfrom", default=None,
                    help="modo datas: AAAA-MM-DD (igual --since do basket)")
    ap.add_argument("--to", dest="dto", default=None)
    a = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    if a.dfrom:
        from datetime import date
        dto = date.fromisoformat(a.dto) if a.dto else date.today()
        collect_range(date.fromisoformat(a.dfrom), dto, a.workers)
    else:
        backfill(a.leagues, a.max_round, a.workers, a.probe)
