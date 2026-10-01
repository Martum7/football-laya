"""Normalizacao de nomes de equipas entre fontes (FotMob/ESPN/FD/wfr).

canonical = minusculas, so [a-z0-9]. Aliases manuais para os que nao batem.
Correr report: python3 teams.py  (lista nao-mapeados por liga)
"""
import json
import re
import sys
from pathlib import Path

# nome normalizado fonte -> canonico nosso
ALIASES = {
    # ENG (FD usa nomes curtos)
    "manunited": "manchesterunited",
    "manunitd": "manchesterunited",
    "mancity": "manchestercity",
    "spurs": "tottenhamhotspur",
    "tottenham": "tottenhamhotspur",
    "wolves": "wolverhamptonwanderers",
    "wolverhampton": "wolverhamptonwanderers",
    "westham": "westhamunited",
    "brighton": "brightonhovealbion",
    "brightonhovealbion": "brightonhovealbion",
    "sheffieldutd": "sheffieldunited",
    "sheffieldwed": "sheffieldwednesday",
    "leeds": "leedsunited",
    "leicester": "leicestercity",
    "norwich": "norwichcity",
    "qpr": "queensparkrangers",
    "rennais": "rennes",
    "staderennais": "rennes",
    # FD/ESPN nomes curtos ESP
    "athbilbao": "athleticbilbao",
    "athmadrid": "atleticomadrid",
    "betis": "realbetis",
    "celta": "celtavigo",
    "espanol": "espanyol",
    "espanhol": "espanyol",  # betclic.pt usa nomes em portugues
    "sociedad": "realsociedad",
    "vallecano": "rayovallecano",
    "valladolid": "realvalladolid",
    # GER nomes oficiais variados
    "cologne": "koln",
    "fccologne": "koln",
    "fckoln": "koln",
    "1fckoln": "koln",
    "fcaugsburg": "augsburg",
    "fcheidenheim": "heidenheim",
    "1fcheidenheim1846": "heidenheim",
    "svdarmstadt": "darmstadt",
    "svdarmstadt98": "darmstadt",
    "tsghoffenheim": "hoffenheim",
    "vfbstuttgart": "stuttgart",
    "vflbochum": "bochum",
    "vflwolfsburg": "wolfsburg",
    "scfreiburg": "freiburg",
    "mainz05": "mainz",
    "einfrankfurt": "eintrachtfrankfurt",
    "mgladbach": "borussiamonchengladbach",
    "leverkusen": "bayerleverkusen",
    "1fcunionberlin": "unionberlin",
    # FRA
    "clermont": "clermontfoot",
    "stetienne": "saintetienne",
    # relegados / variantes historicas 2015-2025
    "cardiff": "cardiffcity",
    "huddersfield": "huddersfieldtown",
    "sdhuesca": "huesca",
    "spgijon": "sportinggijon",
    "nancylorraine": "nancy",
    "bastia": "scbastia",
    "ajaccio": "gfcajaccio",
    "ajacciogfco": "gfcajaccio",
    "dijonfco": "dijon",
    "scamiens": "amiens",
    "scpaderborn": "paderborn",
    "scfreiburg": "freiburg",
    "hamburgsv": "hamburgersv",
    "hamburg": "hamburgersv",
    "hannover": "hannover96",
    "hertha": "herthaberlin",
    "herthab": "herthaberlin",
    "herthabsc": "herthaberlin",
    "fcingolstadt04": "ingolstadt",
    "rasenballsportleipzig": "rbleipzig",
    "1fcnurnberg": "nurnberg",
    "spvgggreutherfurth": "greutherfurth",
    "bielefeld": "arminiabielefeld",
    "chievo": "chievoverona",
    "parma": "parmacalcio1913",
    "spal": "spal2013",
    # FD short names
    "luton": "lutontown",
    "newcastle": "newcastleunited",
    "nottmforest": "nottinghamforest",
    "nottm": "nottinghamforest",
    "ipswich": "ipswichtown",
    "hull": "hullcity",
    "stoke": "stokecity",
    "swansea": "swanseacity",
    "westbrom": "westbromwichalbion",
    "wba": "westbromwichalbion",
    # ESP
    "atleticomadrid": "atleticomadrid",
    "athleticclub": "athleticbilbao",
    "athleticbilbao": "athleticbilbao",
    "realmadrid": "realmadrid",
    "realsociedad": "realsociedad",
    "realbetis": "realbetis",
    "villareal": "villarreal",
    "villarreal": "villarreal",
    "celtadevigo": "celtavigo",
    "celtavigo": "celtavigo",
    "deportivoalaves": "alaves",
    "alaves": "alaves",
    # GER
    "bayernmunich": "bayernmunchen",
    "bayernmunchen": "bayernmunchen",
    "bayern": "bayernmunchen",
    "dortmund": "borussiadortmund",
    "borussiadortmund": "borussiadortmund",
    "mgladbach": "borussiamonchengladbach",
    "borussiamgladbach": "borussiamonchengladbach",
    "borussiamonchengladbach": "borussiamonchengladbach",
    "leverkusen": "bayerleverkusen",
    "bayerleverkusen": "bayerleverkusen",
    "schalke": "schalke04",
    "schalke04": "schalke04",
    "stuttgart": "stuttgart",
    "cologne": "koln",
    "koln": "koln",
    "mainz": "mainz",
    "scfreiburg": "freiburg",
    "freiburg": "freiburg",
    "unionberlin": "unionberlin",
    "rbleipzig": "rbleipzig",
    # ITA
    "inter": "internazionale",
    "internazionale": "internazionale",
    "intermilano": "internazionale",
    "acmilan": "milan",
    "milan": "milan",
    "asroma": "roma",
    "roma": "roma",
    "sscnapoli": "napoli",
    "napoli": "napoli",
    "hellasverona": "verona",
    "verona": "verona",
    # FRA
    "parissg": "parissaintgermain",
    "parissaintgermain": "parissaintgermain",
    "psg": "parissaintgermain",
    "saintetienne": "saintetienne",
    "asse": "saintetienne",
    "olympiquelyonnais": "lyon",
    "lyon": "lyon",
    "olympiquedemarseille": "marseille",
    "marseille": "marseille",
    "asmonaco": "monaco",
    "monaco": "monaco",
}

SUFFIXES = ["fc", "afc", "cf", "sc", "ac", "ssc", "ud", "rcd", "sd"]
PREFIXES = ["afc", "rcd", "ud", "deportivo", "stade", "as", "ac", "ssc", "rc"]


def _strip_de(s):
    if s.startswith("de") and len(s) > 5 and s not in ("derby",):
        return s[2:]
    return s


def normalize(name):
    import unicodedata
    ascii_ = unicodedata.normalize("NFKD", name or "").encode(
        "ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]", "", ascii_.lower()).strip()
    s = re.sub(r"\s+", "", s)
    if s in ALIASES:
        return ALIASES[s]
    for pre in PREFIXES:
        if s.startswith(pre) and len(s) > len(pre) + 3:
            s = _strip_de(s[len(pre):])
            break
    if s in ALIASES:
        return ALIASES[s]
    for suf in SUFFIXES:
        if s.endswith(suf) and len(s) > len(suf) + 2:
            cand = s[: -len(suf)]
            return ALIASES.get(cand, cand)
    return s


def display(canon):
    return canon  # sobreposto depois com nomes oficiais FotMob
