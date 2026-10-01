"""state_text curto (~500 tokens): ingles corrido, sem siglas."""
from features import aggregates, h2h, last_n, rest_days, table_at, wdl


def _res(gf, ga):
    return "won" if gf > ga else ("drew" if gf == ga else "lost")


def _form_str(rows, disp):
    # rows: [(kickoff, foi_casa, gf, ga, adv_canon)] mais recentes no fim
    parts = []
    for _, hc, gf, ga, adv in rows:
        where = "at home" if hc else "away"
        parts.append(f"{_res(gf, ga)} {gf}-{ga} {where} against {disp.get(adv, adv)}")
    return "; ".join(parts)


def verbalize(g, feat, disp):
    H, A = disp[g["home"]], disp[g["away"]]
    L = []
    comp = {"ENG": "Premier League", "ESP": "LaLiga", "GER": "Bundesliga",
            "ITA": "Serie A", "FRA": "Ligue 1"}[g["lg"]]
    L.append(f"{H} host {A} in {comp} round {g['round'] or '?'} on "
             f"{g['kickoff'].strftime('%Y-%m-%d')} at {g['venue'] or 'their home ground'}.")
    th, ph = feat["table"][g["home"]], feat["pos"][g["home"]]
    ta, pa = feat["table"][g["away"]], feat["pos"][g["away"]]
    L.append(f"{H} are {ph}th with {th['pts']} points after {th['p']} games, "
             f"with {th['w']} wins, {th['d']} draws and {th['l']} losses.")
    L.append(f"{A} are {pa}th with {ta['pts']} points after {ta['p']} games.")
    lh, la = feat["last_h"], feat["last_a"]
    wh, dh, lh_ = wdl(lh)
    wa, da, la_ = wdl(la)
    ah, aa = feat["agg_h"], feat["agg_a"]
    L.append(f"In the last {len(lh)} league games, {H} won {wh}, drew {dh} and lost "
             f"{lh_}, scoring {ah['gf']} goals and conceding {ah['ga']}. "
             f"Recent form: {_form_str(lh, disp)}.")
    L.append(f"In the last {len(la)} league games, {A} won {wa}, drew {da} and lost "
             f"{la_}, scoring {aa['gf']} goals and conceding {aa['ga']}. "
             f"Recent form: {_form_str(la, disp)}.")
    hh = feat["h2h"]
    if hh:
        def winner(x):
            if x["hg"] == x["ag"]:
                return None
            return x["home"] if x["hg"] > x["ag"] else x["away"]
        hw = sum(1 for x in hh if winner(x) == g["home"])
        aw = sum(1 for x in hh if winner(x) == g["away"])
        last = hh[-1]
        score = f"{last['hg']}-{last['ag']}"
        L.append(f"Head to head in the last {len(hh)} meetings: {H} won {hw}, "
                 f"{A} won {aw}, with {len(hh) - hw - aw} draws. "
                 f"Last meeting finished {score}.")
    L.append(f"Both teams scored in {ah['btts']} of {H}'s last {ah['n']}. "
             f"{H} kept {ah['cs']} clean sheets.")
    if feat["rest_h"] is not None:
        L.append(f"{H} last played {feat['rest_h']} days ago, "
                 f"{A} last played {feat['rest_a']} days ago.")
    xi = feat.get("xi")
    if xi:
        L.append(f"Confirmed {H} starters in a {xi['h_form'] or 'announced formation'}: "
                 f"{', '.join(xi['h_names'])}.")
        L.append(f"Confirmed {A} starters in a {xi['a_form'] or 'announced formation'}: "
                 f"{', '.join(xi['a_names'])}.")
        if xi["missing"]:
            L.append("Missing: " + "; ".join(xi["missing"]) + ".")
    if feat.get("top_scorer_h"):
        n, c = feat["top_scorer_h"]
        L.append(f"{n} leads {H} with {c} league goals this season.")
    if feat.get("top_scorer_a"):
        n, c = feat["top_scorer_a"]
        L.append(f"{n} leads {A} with {c} league goals this season.")
    return " ".join(L)
