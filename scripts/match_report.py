"""Report de matching entre fontes: FotMob x ESPN x FD por (data, home, away).

Corre no python do sistema (stdlib). Mostra cobertura + top nao-mapeados
para afinar teams.ALIASES antes do build final.
Uso: python3 match_report.py [--season 2024]   # season = end year
"""
import argparse
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from loaders import (iter_espn_months, iter_fd, iter_fotmob_days,  # noqa: E402
                     parse_fd_date, season_end_year)
from teams import normalize  # noqa: E402


def day_key(dt):
    return dt.strftime("%Y-%m-%d")


def main(season=None):
    # Chave por PAR+EPOCA (cada par joga casa+fora 1x por epoca nas Big5).
    # Datas divergem entre fontes em adiados; a data canonica vem do FotMob.
    fm, es, fd = defaultdict(list), defaultdict(list), defaultdict(list)
    for g in iter_fotmob_days():
        if not g["finished"] or not g.get("ts"):
            continue
        dt = datetime.fromtimestamp(g["ts"] / 1000, tz=timezone.utc)
        se = season_end_year(dt)
        if season and se != season:
            continue
        fm[(g["lg"], se, normalize(g["home"]), normalize(g["away"]))].append(g)
    for g in iter_espn_months():
        if not g["finished"] or not g.get("date"):
            continue
        try:
            dt = datetime.fromisoformat(g["date"].replace("Z", "+00:00"))
        except ValueError:
            continue
        se = season_end_year(dt)
        if season and se != season:
            continue
        es[(g["lg"], se, normalize(g["home"]), normalize(g["away"]))].append(g)
    for r in iter_fd():
        dt = parse_fd_date(r)
        if not dt or (season and season_end_year(dt) != season):
            continue
        fd[(r["lg"], season_end_year(dt), normalize(r["HomeTeam"]),
            normalize(r["AwayTeam"]))].append(r)

    print(f"chaves FotMob: {len(fm)}  ESPN: {len(es)}  FD: {len(fd)}")
    for name, a, b in (("FM-ESPN", fm, es), ("FM-FD", fm, fd)):
        only_a = set(a) - set(b)
        print(f"{name}: match={len(set(a) & set(b))} so-em-A={len(only_a)}")
        for k in sorted(only_a)[:15]:
            print("   so-A:", k)
    # golos FM vs FD (sanity de scores)
    same = diff = 0
    for k in set(fm) & set(fd):
        g, r = fm[k][0], fd[k][0]
        try:
            if int(r["FTHG"]) == g["hg"] and int(r["FTAG"]) == g["ag"]:
                same += 1
            else:
                diff += 1
                if diff <= 5:
                    print("   score-diff:", k, (g["hg"], g["ag"]),
                          (r["FTHG"], r["FTAG"]))
        except (ValueError, TypeError):
            pass
    print(f"scores FM==FD: {same}  diferentes: {diff}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2024)
    main(ap.parse_args().season)
