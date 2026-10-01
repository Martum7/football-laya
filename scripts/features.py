"""Rolling point-in-time: last5, tabela N-1, agregados. So passado < cutoff."""
from collections import defaultdict


def build_history(games):
    """games: lista dict(lg,se,home,away,hg,ag,kickoff). Devolve por equipa
    lista ordenada de (kickoff, foi_casa, gf, ga, adversario)."""
    by_team = defaultdict(list)
    for g in games:
        by_team[(g["lg"], g["home"])].append(
            (g["kickoff"], True, g["hg"], g["ag"], g["away"]))
        by_team[(g["lg"], g["away"])].append(
            (g["kickoff"], False, g["ag"], g["hg"], g["home"]))
    for v in by_team.values():
        v.sort()
    return by_team


def last_n(by_team, lg, team, cutoff, n=5):
    return [r for r in by_team.get((lg, team), []) if r[0] < cutoff][-n:]


def wdl(rows):
    w = sum(1 for r in rows if r[2] > r[3])
    d = sum(1 for r in rows if r[2] == r[3])
    return w, d, len(rows) - w - d


def aggregates(rows):
    gf = sum(r[2] for r in rows)
    ga = sum(r[3] for r in rows)
    btts = sum(1 for r in rows if r[2] > 0 and r[3] > 0)
    cs = sum(1 for r in rows if r[3] == 0)
    over25 = sum(1 for r in rows if r[2] + r[3] > 2)
    return {"gf": gf, "ga": ga, "btts": btts, "cs": cs, "over25": over25,
            "n": len(rows)}


def table_at(by_team_games, lg, se, cutoff, games):
    """Tabela da epoca com jogos kickoff < cutoff. Devolve dict equipa->linha
    + posicao. by_team_games nao usado (assinatura p/ futuro)."""
    t = {}
    for g in games:
        if g["lg"] != lg or g["se"] != se or g["kickoff"] >= cutoff:
            continue
        for team, gf, ga in ((g["home"], g["hg"], g["ag"]),
                            (g["away"], g["ag"], g["hg"])):
            r = t.setdefault(team, {"p": 0, "w": 0, "d": 0, "l": 0,
                                    "gf": 0, "ga": 0})
            r["p"] += 1
            r["gf"] += gf
            r["ga"] += ga
            if gf > ga:
                r["w"] += 1
            elif gf == ga:
                r["d"] += 1
            else:
                r["l"] += 1
    for r in t.values():
        r["pts"] = 3 * r["w"] + r["d"]
    order = sorted(t, key=lambda k: (-t[k]["pts"], -(t[k]["gf"] - t[k]["ga"]),
                                     -t[k]["gf"]))
    pos = {k: i + 1 for i, k in enumerate(order)}
    return t, pos


def h2h(games, home, away, cutoff, n=5):
    rows = [g for g in games
            if g["kickoff"] < cutoff and {g["home"], g["away"]} == {home, away}]
    rows.sort(key=lambda g: g["kickoff"])
    return rows[-n:]


def rest_days(by_team, lg, team, cutoff):
    past = [r[0] for r in by_team.get((lg, team), []) if r[0] < cutoff]
    if not past:
        return None
    return (cutoff - max(past)).days
