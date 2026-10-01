"""Scrape das odds para as 70 perguntas do modelo (Betclic + Betano .pt).

Por jogo, devolve as odds tratadas por qid, prontas para o pipeline de EV
(modelo -> probs -> edge = p*odd-1). Formato de saida (JSONL):
  {"event": "Liverpool - Manchester City", "lg": "ENG",
   "kickoff_utc": "...", "casa": "betclic", "fetched_at": "...",
   "url": "...", "odds": {"match_result": {"home": 3.33, ...},
                           "over_25": {"true": 1.9, "false": 1.9}, ...},
   "missing": ["shots_over_265", ...]}

Fontes (sem auth, GET simples):
  - Betclic: pagina da liga (discovery) + ng-state da pagina do evento
    (tab Top: 17 mercados completos). Outras tabs (Golos, Marcadores,
    Estatisticas, Resultado) via XHR GetMatchSnapshot+category_id
    (por ligar quando o URL final estiver confirmado).
  - Betano: pagina da liga (discovery) + initial_state da pagina do evento
    (MRES, HCTG, DBLC, BTSC, OUHG/OUAG, OUH1, H1RS, CNOU, PSCR...).

Uso:
  ../.venv/bin/python odds_scrape.py --league ENG --hours-ahead 72
  ../.venv/bin/python odds_scrape.py --url https://www.betclic.pt/futebol-...-m123
  ../.venv/bin/python odds_scrape.py --league ENG --out ../data/live/odds.jsonl
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import TokenBucket, fetch_text  # noqa: E402
from teams import normalize  # noqa: E402

BASE_DIR = Path(__file__).resolve().parents[1]

_bucket = TokenBucket(1.5)
_HEADERS = {"Accept-Language": "pt-PT,pt;q=0.9"}
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

# Descoberta: paginas de liga (estaveis). Betclic por slug, Betano por URL.
BETCLIC_SLUGS = {
    "ENG": "inglaterra-premier-league-c3",
    "ESP": "espanha-la-liga-c7",
    "GER": "alemanha-bundesliga-c5",
    "ITA": "italia-serie-a-c6",
    "FRA": "franca-ligue-1-c4",
}
BETANO_LG_URLS = {
    "ENG": "https://www.betano.pt/sport/futebol/inglaterra/premier-league/1/",
    "ESP": "https://www.betano.pt/sport/futebol/espanha/laliga/5/",
    "GER": "https://www.betano.pt/sport/futebol/alemanha/bundesliga/216/",
    "ITA": "https://www.betano.pt/sport/futebol/italia/serie-a/1635/",
    "FRA": "https://www.betano.pt/sport/futebol/franca/ligue-1/215/",
}

# Pinnacle via The Odds API (mesmo feed do PS3838/BetInAsia). Uma chamada
# por liga devolve TODOS os eventos (eficiente). Precisa de ODDS_API_KEY
# (gratis, 500 creditos/mes). Livre de login wall.
ODDSAPI_SPORTS = {
    "ENG": "soccer_epl",
    "ESP": "soccer_spain_la_liga",
    "GER": "soccer_germany_bundesliga",
    "ITA": "soccer_italy_serie_a",
    "FRA": "soccer_france_ligue_one",
}
_PINN_CACHE = {}  # lg -> (ts, eventos, api_key)

# As 70 perguntas (3 choice + 67 noul). Ordem fixa para o relatorio.
QIDS_CHOICE = ["match_result", "most_corners", "ht_result"]
QIDS_NOUL = [
    "home_win", "away_win", "draw", "home_double", "away_double",
    "over_05", "over_15", "over_25", "over_35", "over_45",
    "under_25", "under_35", "btts", "home_scores", "away_scores",
    "home_over_05", "home_over_15", "away_over_05", "away_over_15",
    "away_over_25", "ht_over_05", "ht_over_15", "home_ht_scores",
    "away_ht_scores", "away_both_halves", "first_goal_away",
    "second_half_over_15", "odd_total",
    "corners_over_75", "corners_over_85", "corners_over_95",
    "corners_over_105", "corners_under_105", "corners_under_125",
    "home_corners_over_35", "home_corners_over_45",
    "away_corners_over_35", "away_corners_over_45", "away_corners_over_55",
    "shots_over_265", "home_shots_over_105", "away_shots_over_145",
    "sot_over_75", "sot_over_85", "cards_over_35", "cards_over_45",
    "red_card", "ht_btts_no", "home_clean_sheet", "away_clean_sheet",
    "away_win_by_1", "away_win_by_2plus",
] + [f"p{i}_anytime" for i in range(11)] + ["p2_0", "p2_1", "pht_0", "pht_1"]
ALL_QIDS = QIDS_CHOICE + QIDS_NOUL
assert len(ALL_QIDS) == 70, len(ALL_QIDS)

OU_LINES = {"over_05": 0.5, "over_15": 1.5, "over_25": 2.5,
            "over_35": 3.5, "over_45": 4.5,
            "under_25": 2.5, "under_35": 3.5}
CORNERS_LINES = {"corners_over_75": 7.5, "corners_over_85": 8.5,
                 "corners_over_95": 9.5, "corners_over_105": 10.5,
                 "corners_under_105": 10.5, "corners_under_125": 12.5}
HOME_CORNERS = {"home_corners_over_35": 3.5, "home_corners_over_45": 4.5}
AWAY_CORNERS = {"away_corners_over_35": 3.5, "away_corners_over_45": 4.5,
                "away_corners_over_55": 5.5}


def _get(url):
    _bucket.wait()
    code, text = fetch_text(url, timeout=30, extra_headers={
        "Accept-Language": "pt-PT,pt;q=0.9", "User-Agent": _UA})
    if code != 200 or not text:
        print(f"[odds] GET falhou {code}: {url}", flush=True)
        return None
    return text


def _pt_float(s):
    return float(str(s).replace(",", ".").strip())


# ---------- Betclic ----------

def _ng_state_match(html):
    m = re.search(r'<script id="ng-state"[^>]*>(.*?)</script>', html, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(1))
    except ValueError:
        return None
    for v in d.values():
        if isinstance(v, dict) and isinstance(v.get("response"), dict):
            pay = v["response"].get("payload", {})
            if isinstance(pay, dict) and isinstance(pay.get("match"), dict):
                return pay["match"]
    return None


def _flat_selections(mk):
    """Todas as selections (name, odds) de um market do ng-state."""
    out = []
    for s in mk.get("mainSelections", []):
        if s.get("status", 1) == 1 and s.get("odds"):
            out.append((s.get("name", ""), float(s["odds"])))
    for row in mk.get("selectionMatrix", []):
        for w in row.get("selections", []):
            s = ((w.get("selectionOneof") or {}).get("selection") or {})
            if s.get("status", 1) == 1 and s.get("odds"):
                out.append((s.get("name", ""), float(s["odds"])))
    return out


def betclic_league_events(lg):
    """Discovery: [(home, away, event_path)] da pagina da liga."""
    text = _get(f"https://www.betclic.pt/futebol-sfootball/{BETCLIC_SLUGS[lg]}")
    if not text:
        return []
    evs = []
    for label, href in re.findall(r'aria-label="([^"]+)"[^>]*href="([^"]+-m\d+)"', text):
        if " - " not in label:
            continue
        home, away = label.split(" - ", 1)
        evs.append({"home": home.strip(), "away": away.strip(), "url": href})
    return evs


def betclic_event_markets(event_url):
    """Mercados da tab Top (ng-state): {market_name: [(sel, odd)]} + meta."""
    text = _get("https://www.betclic.pt" + event_url
                if event_url.startswith("/") else event_url)
    if not text:
        return {}, {}
    mt = _ng_state_match(text)
    if not mt:
        return {}, {}
    meta = {"event": mt.get("name"), "kickoff_utc": mt.get("matchDateUtc"),
            "is_live": mt.get("isLive"),
            "open_markets": mt.get("openMarketCount"),
            "categories": [(c.get("id"), c.get("name"))
                           for c in mt.get("categories", [])]}
    markets = {}
    for sub in mt.get("subCategories", []):
        for mk in sub.get("markets", []):
            sels = _flat_selections(mk)
            if sels:
                markets.setdefault(mk.get("name", ""), sels)
    return markets, meta


def _ou_pick(sels, line, over=True):
    want = ("Acima" if over else "Abaixo") + f" de {line}".replace(".", ",")
    for name, odd in sels:
        if name.strip() == want:
            return odd
    return None


def map_betclic(home, away, markets):
    """ng-state Top -> {qid: {outcome: odd}}. So saidas tratadas."""
    odds, has = {}, set(markets)
    get = markets.get

    m = get("Resultado (Tempo Regulamentar)", [])
    if len(m) >= 3:
        o = [m[0][1], m[1][1], m[2][1]]
        odds["match_result"] = {"home": o[0], "draw": o[1], "away": o[2]}
        odds["home_win"] = {"true": o[0]}
        odds["draw"] = {"true": o[1]}
        odds["away_win"] = {"true": o[2]}

    for name, odd in get("Resultado duplo", []):
        n = name.lower()
        if "ou empate" in n:
            if normalize(home) in normalize(n) or normalize(n.split(" ou ")[0]) in normalize(home):
                odds["home_double"] = {"true": odd}
            else:
                odds["away_double"] = {"true": odd}

    tot = get("Total de golos - acima/abaixo", [])
    for qid, line in OU_LINES.items():
        over = qid.startswith("over")
        odd = _ou_pick(tot, line, over)
        if odd:
            odds[qid] = {"true": odd}

    for name, odd in get("As duas equipas marcam", []):
        if name.strip().lower() == "sim":
            odds["btts"] = {"true": odd}
        elif name.strip().lower() == "não":
            odds.setdefault("btts", {})["false"] = odd

    for team, keys in ((home, ("home_scores", "home_over_05", "home_over_15")),
                       (away, ("away_scores", "away_over_05", "away_over_15",
                               "away_over_25"))):
        sels = []
        for mk_name, mk_sels in markets.items():
            if mk_name.startswith(team + " - Total de golos"):
                sels = mk_sels
                break
        if not sels:
            continue
        for qid in keys:
            line = 0.5 if qid in ("home_scores", "away_scores", "home_over_05",
                                  "away_over_05") else (1.5 if "15" in qid else 2.5)
            odd = _ou_pick(sels, line, True)
            if odd:
                odds[qid] = {"true": odd}
    return odds


# ---------- Betano ----------

def _initial_state(html):
    m = re.search(r'window\["initial_state"\]=(\{.*?\});?\s*</script>', html, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))["data"]
    except (ValueError, KeyError):
        return None


def betano_league_events(lg):
    text = _get(BETANO_LG_URLS[lg])
    if not text:
        return []
    data = _initial_state(text)
    if not data:
        return []
    evs = []
    for blk in data.get("blocks", []):
        for e in blk.get("events", []):
            parts = e.get("participants", [])
            if len(parts) < 2:
                continue
            evs.append({"home": parts[0]["name"], "away": parts[1]["name"],
                        "url": e.get("url")})
    return evs


def betano_event_markets(event_url):
    text = _get("https://www.betano.pt" + event_url)
    if not text:
        return {}, {}
    data = _initial_state(text)
    if not data:
        return {}, {}
    ev = data.get("event", {})
    meta = {"event": ev.get("name"), "kickoff_utc": None,
            "is_live": None, "open_markets": ev.get("totalMarketsAvailable"),
            "categories": []}
    if ev.get("startTime"):
        try:
            meta["kickoff_utc"] = datetime.fromtimestamp(
                ev["startTime"] / 1000, tz=timezone.utc).isoformat()
        except (TypeError, ValueError):
            pass
    markets = {}
    for mk in ev.get("markets", []):
        sels = [(s.get("name", ""), float(s["price"]))
                for s in mk.get("selections", []) if s.get("price")]
        if sels:
            markets.setdefault(f"{mk.get('type')}:{mk.get('name')}", sels)
    # marcadores (tabela PSCR): nome -> odd anytime
    for mk in ev.get("markets", []):
        if mk.get("type") != "PSCR":
            continue
        table = mk.get("tableLayout") or {}
        for row in table.get("rows", []):
            title = row.get("title", "")
            for gs in row.get("groupSelections", []):
                for s in gs.get("selections", [])[:1]:
                    if s.get("price"):
                        markets.setdefault("PSCR:anytime", []).append(
                            (title, float(s["price"])))
    return markets, meta


def map_betano(home, away, markets):
    odds = {}

    def find_mkt(prefix):
        for k, v in markets.items():
            if k.startswith(prefix):
                return v
        return []

    mres = find_mkt("MRES:")
    if len(mres) >= 3:
        o = [mres[0][1], mres[1][1], mres[2][1]]
        odds["match_result"] = {"home": o[0], "draw": o[1], "away": o[2]}
        odds["home_win"] = {"true": o[0]}
        odds["draw"] = {"true": o[1]}
        odds["away_win"] = {"true": o[2]}

    for name, odd in find_mkt("DBLC:"):
        n = name.lower()
        if "ou empate" in n:
            if normalize(home) in normalize(n):
                odds["home_double"] = {"true": odd}
            else:
                odds["away_double"] = {"true": odd}

    hctg = find_mkt("HCTG:")
    for qid, line in OU_LINES.items():
        over = qid.startswith("over")
        for name, odd in hctg:
            if (name.startswith("Mais") if over else name.startswith("Menos")) \
                    and f"{line}" in name:
                odds[qid] = {"true": odd}
                break

    for name, odd in find_mkt("BTSC:"):
        if name.strip().lower() == "sim":
            odds["btts"] = {"true": odd}
        elif name.strip().lower() == "não":
            odds.setdefault("btts", {})["false"] = odd

    for team, keys in ((home, ("home_scores", "home_over_05", "home_over_15")),
                       (away, ("away_scores", "away_over_05", "away_over_15",
                               "away_over_25"))):
        sels = find_mkt("OUHG:") if team == home else find_mkt("OUAG:")
        for qid in keys:
            line = 0.5 if qid in ("home_scores", "away_scores", "home_over_05",
                                  "away_over_05") else (1.5 if "15" in qid else 2.5)
            for name, odd in sels:
                if name.startswith("Mais") and f"{line}" in name:
                    odds[qid] = {"true": odd}
                    break

    ouh1 = find_mkt("OUH1:")
    for qid, line in (("ht_over_05", 0.5), ("ht_over_15", 1.5)):
        for name, odd in ouh1:
            if name.startswith("Mais") and f"{line}" in name:
                odds[qid] = {"true": odd}
                break
    h1rs = find_mkt("H1RS:")
    if len(h1rs) >= 3:
        odds["ht_result"] = {"home": h1rs[0][1], "draw": h1rs[1][1],
                             "away": h1rs[2][1]}

    cnou = find_mkt("CNOU:")
    for qid, line in CORNERS_LINES.items():
        over = "over" in qid
        for name, odd in cnou:
            if (name.startswith("Mais") if over else name.startswith("Menos")) \
                    and f"{line}" in name:
                odds[qid] = {"true": odd}
                break

    for i, (name, odd) in enumerate(find_mkt("PSCR:anytime")[:11]):
        odds[f"p{i}_anytime"] = {"true": odd, "_player": name}
    return odds


# ---------- Pinnacle via The Odds API (feed PS3838) ----------

def _oddsapi_key():
    import os
    from common import load_dotenv  # noqa
    load_dotenv()
    return os.environ.get("ODDS_API_KEY")


def pinnacle_league_odds(lg, api_key=None, ttl=900):
    """TODA a liga numa chamada: [{home, away, commence, 1x2, ou25}].

    Filtra so book==pinnacle. Cache de `ttl`s: varios triggers na mesma
    janela partilham o snapshot (o tier gratis tem 500 creditos/mes).
    """
    import urllib.parse
    import urllib.request
    key = api_key or _oddsapi_key()
    if not key:
        raise RuntimeError("ODDS_API_KEY em falta (.env ou ambiente)")
    now_ts = time.time()
    hit = _PINN_CACHE.get(lg)
    if hit and now_ts - hit[0] < ttl and (api_key is None or hit[2] == api_key):
        return hit[1]
    qs = urllib.parse.urlencode({"apiKey": key, "regions": "eu",
                                 "markets": "h2h,totals",
                                 "bookmakers": "pinnacle",
                                 "oddsFormat": "decimal"})
    url = f"https://api.the-odds-api.com/v4/sports/{ODDSAPI_SPORTS[lg]}/odds?{qs}"
    _bucket.wait()
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        events = json.loads(r.read().decode("utf-8"))
    out = []
    for ev in events:
        h2h = ou = None
        for b in ev.get("bookmakers", []):
            if b.get("key") != "pinnacle":
                continue
            for mkt in b.get("markets", []):
                if mkt.get("key") == "h2h":
                    pr = {o.get("name"): o.get("price")
                          for o in mkt.get("outcomes", [])}
                    try:
                        h2h = [float(pr[ev["home_team"]]),
                               float(pr["Draw"]),
                               float(pr[ev["away_team"]])]
                    except (KeyError, TypeError, ValueError):
                        pass
                elif mkt.get("key") == "totals":
                    for o in mkt.get("outcomes", []):
                        try:
                            if float(o.get("point")) == 2.5:
                                if o.get("name") == "Over":
                                    ou = ou or [None, None]
                                    ou[0] = float(o.get("price"))
                                elif o.get("name") == "Under":
                                    ou = ou or [None, None]
                                    ou[1] = float(o.get("price"))
                        except (TypeError, ValueError):
                            pass
        out.append({"home": ev.get("home_team", ""), "away": ev.get("away_team", ""),
                    "commence": ev.get("commence_time"), "1x2": h2h, "ou25": ou})
    _PINN_CACHE[lg] = (now_ts, out, api_key)
    return out


def map_pinnacle(home, away, trio, ou):
    """1x2 + totais Pinnacle -> qids. Mesmo formato das outras casas."""
    odds = {}
    if trio and len(trio) == 3 and all(trio):
        odds["match_result"] = {"home": trio[0], "draw": trio[1], "away": trio[2]}
        odds["home_win"] = {"true": trio[0]}
        odds["draw"] = {"true": trio[1]}
        odds["away_win"] = {"true": trio[2]}
    if ou and len(ou) == 2 and ou[0]:
        odds["over_25"] = {"true": ou[0]}
    if ou and len(ou) == 2 and ou[1]:
        odds["under_25"] = {"true": ou[1]}
    return odds


# ---------- PS3838 prematch (via CDP: search -> ev= -> odds completas) ----------

PS_BASE = "https://www.ps3838.com/sports-service/sv"
PS_QS = ("btg=1&c=&cl=3&d=&ec=&g=&hle=false&ic=false&ice=false&inl=false"
         "&l=3&lang=&lg=&lv=0&me=0&me01=&mk=1&more=false&o=1&ot=1&pa=0"
         "&pn=-1&pv=1&ru=&sp=29&tm=0&v=0&locale=pt_BR")
# mercados: "0" = jogo inteiro, "1" = 1a parte. Por mercado:
# [spreads, totals, moneyline, ...]; moneyline = [1, X, 2, ...];
# totals row = [label, linha, over, under, ...].


def ps_search(team):
    """IDs de eventos por equipa: /search/v2 (sem auth)."""
    import urllib.parse
    import urllib.request
    qs = urllib.parse.urlencode({"s": 11, "t": team, "locale": "pt_BR"})
    req = urllib.request.Request(f"{PS_BASE}/search/v2?{qs}",
                                 headers={"User-Agent": _UA})
    _bucket.wait()
    with urllib.request.urlopen(req, timeout=25) as r:
        d = json.loads(r.read().decode("utf-8"))
    ids = []
    for t in d.get("teams", []):
        for e in t.get("events", []):
            ids.append(e["id"])
    return ids


def ps_event(ev_id):
    """Evento completo: liga, equipas, kickoff, live?, mercados 0/1."""
    import urllib.request
    _bucket.wait()
    req = urllib.request.Request(
        f"{PS_BASE}/compact/events?{PS_QS}&ev={ev_id}",
        headers={"User-Agent": _UA,
                 "Accept-Language": "pt-PT,pt;q=0.9"})
    with urllib.request.urlopen(req, timeout=25) as r:
        d = json.loads(r.read().decode("utf-8"))
    blocks = (d.get("n") or d.get("l") or [])
    if not blocks:
        return None
    for lg in blocks[0][2]:
        for e in lg[2]:
            if e[0] != ev_id:
                continue
            try:
                ko = datetime.fromtimestamp(e[4] / 1000,
                                            tz=timezone.utc).isoformat()
            except (TypeError, ValueError):
                ko = None
            mkts = {}
            for key, m in (e[8] if len(e) > 8 else {}).items():
                try:
                    ml = [float(m[2][0]), float(m[2][1]), float(m[2][2])]
                except (IndexError, TypeError, ValueError):
                    ml = None
                totals = {}
                for row in m[1] or []:
                    try:
                        totals[float(row[1])] = (float(row[2]), float(row[3]))
                    except (IndexError, TypeError, ValueError):
                        continue
                mkts[key] = {"ml": ml, "totals": totals}
            return {"league": lg[1], "home": e[1], "away": e[2],
                    "kickoff_utc": ko, "live": bool(e[5]), "markets": mkts}
    return None


def map_ps3838(home, away, ev):
    """Mercados 0 (jogo) + 1 (1a parte) -> qids."""
    odds = {}
    m0 = (ev.get("markets") or {}).get("0", {})
    m1 = (ev.get("markets") or {}).get("1", {})
    if m0.get("ml") and len(m0["ml"]) == 3:
        h, d, a = m0["ml"]
        odds["match_result"] = {"home": h, "draw": d, "away": a}
        odds["home_win"] = {"true": h}
        odds["draw"] = {"true": d}
        odds["away_win"] = {"true": a}
    for line, qid in ((0.5, "over_05"), (1.5, "over_15"), (2.5, "over_25"),
                      (3.5, "over_35"), (4.5, "over_45")):
        if line in m0.get("totals", {}):
            odds[qid] = {"true": m0["totals"][line][0]}
    for line, qid in ((2.5, "under_25"), (3.5, "under_35")):
        if line in m0.get("totals", {}):
            odds[qid] = {"true": m0["totals"][line][1]}
    if m1.get("ml") and len(m1["ml"]) == 3:
        odds["ht_result"] = {"home": m1["ml"][0], "draw": m1["ml"][1],
                             "away": m1["ml"][2]}
    for line, qid in ((0.5, "ht_over_05"), (1.5, "ht_over_15")):
        if line in m1.get("totals", {}):
            odds[qid] = {"true": m1["totals"][line][0]}
    return odds


# ---------- driver ----------

def _match_names(fot_home, fot_away, ev_home, ev_away):
    fh, fa = normalize(fot_home), normalize(fot_away)
    h, a = normalize(ev_home), normalize(ev_away)
    return (fh in h or h in fh) and (fa in a or a in fa)


def get_odds_for_fixture(lg, home, away, casas=("betclic", "betano"),
                         league_index=None):
    """Odds tratadas por qid para UM jogo: {casa: {qid: {outcome: odd}}}.

    league_index opcional: {casa: {lg: [eventos]}} para nao refazer as
    paginas de liga a cada jogo (o scheduler constroi uma vez por ronda).
    Devolve (odds_by_casa, errors).
    """
    odds_by_casa, errors = {}, {}
    for casa in casas:
        if casa == "ps3838":
            # Sem indice de ligas: pesquisa pela equipa -> IDs -> evento.
            try:
                ids = ps_search(home)
                hit = None
                for eid in ids[:6]:
                    try:
                        cand = ps_event(eid)
                    except Exception:
                        continue
                    if not cand or cand["live"]:
                        continue
                    import difflib
                    na = normalize(cand["away"])
                    if na == normalize(away) or na in normalize(away) \
                            or normalize(away) in na:
                        hit = cand
                        break
                if not hit:
                    errors[casa] = "jogo nao encontrado"
                    continue
                odds = map_ps3838(hit["home"], hit["away"], hit)
                if odds:
                    odds_by_casa[casa] = odds
                else:
                    errors[casa] = "sem mercados mapeaveis"
            except Exception as e:  # noqa
                errors[casa] = str(e)[:200]
            continue
        if casa == "pinnacle":
            # Sem pagina de evento: a chamada de liga JA traz 1x2 + totais.
            try:
                if league_index and lg in (league_index.get(casa) or {}):
                    evs = league_index[casa][lg]
                else:
                    evs = pinnacle_league_odds(lg)
            except Exception as e:  # noqa
                errors[casa] = str(e)[:200]
                continue
            hit = next((e for e in evs
                        if _match_names(home, away, e["home"], e["away"])), None)
            if not hit:
                errors[casa] = "jogo nao listado"
                continue
            odds = map_pinnacle(hit["home"], hit["away"],
                                hit.get("1x2"), hit.get("ou25"))
            if odds:
                odds_by_casa[casa] = odds
            else:
                errors[casa] = "sem mercados mapeaveis"
            continue
        try:
            if league_index and lg in (league_index.get(casa) or {}):
                evs = league_index[casa][lg]
            else:
                evs = (betclic_league_events(lg) if casa == "betclic"
                       else betano_league_events(lg))
        except Exception as e:  # noqa
            errors[casa] = str(e)[:200]
            continue
        hit = next((e for e in evs
                    if _match_names(home, away, e["home"], e["away"])), None)
        if not hit:
            errors[casa] = "jogo nao listado"
            continue
        try:
            if casa == "betclic":
                markets, meta = betclic_event_markets(hit["url"])
                odds = map_betclic(hit["home"], hit["away"], markets)
            else:
                markets, meta = betano_event_markets(hit["url"])
                odds = map_betano(hit["home"], hit["away"], markets)
            if odds:
                odds_by_casa[casa] = odds
            else:
                errors[casa] = "sem mercados mapeaveis"
        except Exception as e:  # noqa
            errors[casa] = str(e)[:200]
    return odds_by_casa, errors


def scrape_game(lg, fot_home, fot_away, casa, ev, out_f, run_id="-"):
    now = datetime.now(timezone.utc).isoformat()
    if casa == "betclic":
        markets, meta = betclic_event_markets(ev["url"])
        odds = map_betclic(ev["home"], ev["away"], markets)
    else:
        markets, meta = betano_event_markets(ev["url"])
        odds = map_betano(ev["home"], ev["away"], markets)
    missing = [q for q in ALL_QIDS if q not in odds]
    rec = {"run_id": run_id, "event": f"{ev['home']} - {ev['away']}", "lg": lg,
           "fotmob_home": fot_home, "fotmob_away": fot_away,
           "kickoff_utc": meta.get("kickoff_utc"), "casa": casa,
           "fetched_at": now, "url": ev["url"],
           "n_markets": len(markets),
           "categories": meta.get("categories", []),
           "odds": odds, "missing": missing,
           "coverage": f"{len(odds)}/{len(ALL_QIDS)}"}
    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out_f.flush()
    print(f"[odds] {casa} {rec['event']}: {rec['coverage']} "
          f"mercados={len(markets)}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues", nargs="*", default=["ENG", "ESP", "GER", "ITA", "FRA"])
    ap.add_argument("--casas", nargs="*", default=["betclic", "betano"])
    ap.add_argument("--hours-ahead", type=int, default=72)
    ap.add_argument("--url", default=None, help="pagina de evento unica (debug)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--watch", type=int, default=0,
                    help="loop nonstop: nova ronda de X em X segundos (0 = uma vez)")
    a = ap.parse_args()

    if a.url:
        casa = "betclic" if "betclic" in a.url else "betano"
        path = a.url.split(".pt", 1)[1]
        if casa == "betclic":
            markets, meta = betclic_event_markets(path)
            print(json.dumps(meta, ensure_ascii=False, indent=1)[:800])
            odds = map_betclic("", "", markets)
        else:
            markets, meta = betano_event_markets(path)
            print(json.dumps(meta, ensure_ascii=False, indent=1)[:800])
            odds = map_betano("", "", markets)
        print(f"mercados={len(markets)} qids={len(odds)}/{len(ALL_QIDS)}")
        print("qids:", sorted(odds))
        print("missing:", [q for q in ALL_QIDS if q not in odds])
        return

    sys.path.insert(0, str(Path(__file__).parent))
    from live_server import upcoming_fixtures  # noqa (fixtures FotMob)
    fixtures = upcoming_fixtures(a.hours_ahead,
                                 [l for l in a.leagues if l in BETCLIC_SLUGS])
    print(f"[odds] fixtures: {len(fixtures)}", flush=True)

    out = open(a.out, "a", encoding="utf-8") if a.out else sys.stdout
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    scan_once(a, out, run_id)
    while a.watch > 0:
        print(f"[odds] proxima ronda em {a.watch}s ...", flush=True)
        time.sleep(a.watch)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        try:
            scan_once(a, out, run_id)
        except Exception as e:  # noqa - o relogio nao pode parar
            print(f"[odds] ronda {run_id} falhou: {e}", flush=True)
    if a.out:
        out.close()


def scan_once(a, out, run_id):
    """Uma ronda completa: fixtures FotMob -> match nas casas -> snapshot JSONL."""
    sys.path.insert(0, str(Path(__file__).parent))
    from live_server import upcoming_fixtures  # noqa (fixtures FotMob)
    fixtures = upcoming_fixtures(a.hours_ahead,
                                 [l for l in a.leagues if l in BETCLIC_SLUGS])
    print(f"[odds] ronda {run_id}: fixtures={len(fixtures)}", flush=True)
    n = 0
    for casa in a.casas:
        for lg in a.leagues:
            try:
                evs = (betclic_league_events(lg) if casa == "betclic"
                       else betano_league_events(lg))
            except Exception as e:  # noqa - uma liga nao mata o scan
                print(f"[odds] {casa}/{lg} falhou: {e}", flush=True)
                continue
            print(f"[odds] {casa}/{lg}: {len(evs)} eventos", flush=True)
            for fx in [f for f in fixtures if f["lg"] == lg]:
                hit = next((e for e in evs
                            if _match_names(fx["home"], fx["away"],
                                            e["home"], e["away"])), None)
                if not hit:
                    print(f"[odds] {casa} sem {fx['home']}-{fx['away']}",
                          flush=True)
                    continue
                try:
                    rec = scrape_game(lg, fx["home"], fx["away"], casa, hit, out,
                                      run_id)
                    n += 1
                except Exception as e:  # noqa
                    print(f"[odds] {casa} {hit} falhou: {e}", flush=True)
                time.sleep(1)
    print(f"[odds] ronda {run_id} DONE: {n} snapshots", flush=True)


if __name__ == "__main__":
    main()
