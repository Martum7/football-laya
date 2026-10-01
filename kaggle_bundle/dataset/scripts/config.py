"""Config partilhada dos fetchers Big5 (max anos disponiveis)."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RAW = BASE_DIR / "data" / "raw"
LOGS = BASE_DIR / "data" / "logs"
PROCESSED = BASE_DIR / "data" / "processed"

# Big5: ids por fonte. seasons seaisonIds do Sofascore sao descobertos via API (max disponivel).
LEAGUES = {
    "ENG": {"name": "Premier League", "sofascore_ut": 17, "fd_codes": ["E0"],
            "understat": "EPL", "wfr_country": "ENG"},
    "ESP": {"name": "LaLiga", "sofascore_ut": 8, "fd_codes": ["SP1"],
            "understat": "La_liga", "wfr_country": "ESP"},
    "GER": {"name": "Bundesliga", "sofascore_ut": 35, "fd_codes": ["D1"],
            "understat": "Bundesliga", "wfr_country": "GER"},
    "ITA": {"name": "Serie A", "sofascore_ut": 23, "fd_codes": ["I1"],
            "understat": "Serie_A", "wfr_country": "ITA"},
    "FRA": {"name": "Ligue 1", "sofascore_ut": 34, "fd_codes": ["F1"],
            "understat": "Ligue_1", "wfr_country": "FRA"},
}

# football-data.co.uk: epocas 2015/16 -> 2024/25 (formato do site: 1516..2425)
FD_SEASONS = ["1516", "1617", "1718", "1819", "1920",
              "2021", "2122", "2223", "2324", "2425"]

# Understat usa ano de inicio: 2015 (=2015/16) .. 2024 (=2024/25)
UNDERSTAT_SEASONS = list(range(2015, 2025))

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.5 Safari/605.1.15",
]

# Rate limits (pedidos/segundo por fonte, conservadores para nao levar block).
# SoaScore: partilhado entre workers do fetcher sofascore.
RATE = {
    "sofascore_rps": 0.9,      # ~1 req / 1.1s
    "sofascore_workers": 3,
    "wfr_workers": 8,
    "understat_rps": 2.0,
    "understat_workers": 4,
    "sorare_rps": 3.0,         # 180/min, abaixo dos 200/min da key
    "sorare_workers": 4,
}
