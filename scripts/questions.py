"""As 70 perguntas congeladas (v1 short). NAO EDITAR sem versao nova:
treino e inferencia tem de usar os mesmos templates.
Tipos: 3 choice (match_result, most_corners, ht_result) + 67 noul.
"""
QUESTIONS = {
    "match_result": {"type": "choice", "instructions": "What is the full time result?",
                     "criteria": {"home": "HOME wins", "draw": "draw", "away": "AWAY wins"}},
    "home_win": {"type": "noul", "instructions": "Does HOME win the match?"},
    "away_win": {"type": "noul", "instructions": "Does AWAY win the match?"},
    "draw": {"type": "noul", "instructions": "Is the match a draw?"},
    "home_double": {"type": "noul", "instructions": "Does HOME avoid defeat?"},
    "away_double": {"type": "noul", "instructions": "Does AWAY avoid defeat?"},
    "over_05": {"type": "noul", "instructions": "Are there at least 1 goals scored in total?"},
    "over_15": {"type": "noul", "instructions": "Are there at least 2 goals scored in total?"},
    "over_25": {"type": "noul", "instructions": "Are there more than 2 goals scored in total?"},
    "over_35": {"type": "noul", "instructions": "Are there more than 3 goals scored in total?"},
    "over_45": {"type": "noul", "instructions": "Are there more than 4 goals scored in total?"},
    "under_25": {"type": "noul", "instructions": "Are there fewer than 3 goals scored in total?"},
    "under_35": {"type": "noul", "instructions": "Are there fewer than 4 goals scored in total?"},
    "btts": {"type": "noul", "instructions": "Do both teams score at least one goal?"},
    "home_scores": {"type": "noul", "instructions": "Does HOME score at least one goal?"},
    "away_scores": {"type": "noul", "instructions": "Does AWAY score at least one goal?"},
    "home_over_05": {"type": "noul", "instructions": "Does HOME score at least one goal?"},
    "home_over_15": {"type": "noul", "instructions": "Does HOME score at least 2 goals?"},
    "away_over_05": {"type": "noul", "instructions": "Does AWAY score at least one goal?"},
    "away_over_15": {"type": "noul", "instructions": "Does AWAY score at least 2 goals?"},
    "away_over_25": {"type": "noul", "instructions": "Does AWAY score at least 3 goals?"},
    "ht_over_05": {"type": "noul", "instructions": "Is there at least one goal in the first half?"},
    "ht_over_15": {"type": "noul", "instructions": "Are there at least 2 goals in the first half?"},
    "second_half_over_15": {"type": "noul", "instructions": "Are there at least 2 goals in the second half?"},
    "odd_total": {"type": "noul", "instructions": "Is the total number of goals odd?"},
    "home_ht_scores": {"type": "noul", "instructions": "Does HOME score in the first half?"},
    "away_ht_scores": {"type": "noul", "instructions": "Does AWAY score in the first half?"},
    "away_both_halves": {"type": "noul", "instructions": "Does AWAY score in both halves?"},
    "first_goal_away": {"type": "noul", "instructions": "Does AWAY score the first goal of the match?"},
}
# 11 anytime (preenchidos por jogo com os titulares ofensivos + top marcador)
PLAYER_ANYTIME = "Does PLAYERNAME score at least one goal?"
PLAYER_2PLUS = "Does PLAYERNAME score at least 2 goals?"
PLAYER_HT = "Does PLAYERNAME score in the first half?"

CORNERS = {
    "corners_over_75": "Are there more than 7 corners in total?",
    "corners_over_85": "Are there more than 8 corners in total?",
    "corners_over_95": "Are there more than 9 corners in total?",
    "corners_over_105": "Are there more than 10 corners in total?",
    "corners_under_105": "Are there fewer than 11 corners in total?",
    "corners_under_125": "Are there fewer than 13 corners in total?",
    "home_corners_over_35": "Does HOME have more than 3 corners?",
    "home_corners_over_45": "Does HOME have more than 4 corners?",
    "away_corners_over_35": "Does AWAY have more than 3 corners?",
    "away_corners_over_45": "Does AWAY have more than 4 corners?",
    "away_corners_over_55": "Does AWAY have more than 5 corners?",
    "most_corners": {"type": "choice",
                     "instructions": "Which team has the most corners?",
                     "criteria": {"home": "HOME have the most corners",
                                 "draw": "equal corners",
                                 "away": "AWAY have the most corners"}},
}
SHOTS = {
    "shots_over_265": "Are there more than 26 total shots?",
    "home_shots_over_105": "Does HOME have more than 10 shots?",
    "away_shots_over_145": "Does AWAY have more than 14 shots?",
    "sot_over_75": "Are there more than 7 shots on target in total?",
    "sot_over_85": "Are there more than 8 shots on target in total?",
}
CARDS = {
    "cards_over_35": "Are there more than 3 yellow cards in total?",
    "cards_over_45": "Are there more than 4 yellow cards in total?",
    "red_card": "Is there at least one red card?",
}
HTM = {
    "ht_result": {"type": "choice",
                  "instructions": "What is the half time result?",
                  "criteria": {"home": "HOME lead at half time",
                              "draw": "draw at half time",
                              "away": "AWAY lead at half time"}},
    "ht_btts_no": {"type": "noul",
                   "instructions": "Is at least one team scoreless in the first half?"},
    "home_clean_sheet": {"type": "noul", "instructions": "Does HOME keep a clean sheet?"},
    "away_clean_sheet": {"type": "noul", "instructions": "Does AWAY keep a clean sheet?"},
    "away_win_by_1": {"type": "noul", "instructions": "Does AWAY win by exactly 1 goal?"},
    "away_win_by_2plus": {"type": "noul", "instructions": "Does AWAY win by 2 or more goals?"},
}


def render(home, away, players):
    """players: dict com keys anytime_11 (lista 11 nomes), two (2 nomes), ht (2 nomes).
    Devolve dict qid->question (70 no total)."""
    qs = {}
    for qid, q in {**QUESTIONS,
                   **{k: ({"type": "noul", "instructions": v}
                           if isinstance(v, str) else v)
                      for k, v in {**CORNERS, **SHOTS, **CARDS, **HTM}.items()}}.items():
        qs[qid] = _fill(q, home, away)
    for i, name in enumerate(players["anytime_11"]):
        qs[f"p{i}_anytime"] = {"type": "noul",
                               "instructions": PLAYER_ANYTIME.replace("PLAYERNAME", name)}
    for i, name in enumerate(players["two"]):
        qs[f"p2_{i}"] = {"type": "noul",
                         "instructions": PLAYER_2PLUS.replace("PLAYERNAME", name)}
    for i, name in enumerate(players["ht"]):
        qs[f"pht_{i}"] = {"type": "noul",
                          "instructions": PLAYER_HT.replace("PLAYERNAME", name)}
    assert len(qs) == 70, f"esperava 70, deu {len(qs)}"
    return qs


def _fill(q, home, away):
    out = dict(q)
    if isinstance(out.get("instructions"), str):
        out["instructions"] = out["instructions"].replace("HOME", home).replace("AWAY", away)
    if isinstance(out.get("criteria"), dict):
        out["criteria"] = {k: v.replace("HOME", home).replace("AWAY", away)
                           for k, v in out["criteria"].items()}
    return out
