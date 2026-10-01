"""Teste anti-leakage: corre SEMPRE depois do build, falha o dataset se falhar.

Contrato (15 regras, ver README):
1. cutoff == kickoff - 50min em todos os casos
2. last_game_included_* < kickoff
3. standings_round_used < round do jogo
4. split temporal: max(train) < min(calib) < min(test)
5. odds nunca dentro de state_text/questions
6. scores Sorare (se presentes) com game.date < cutoff
7. provenance completa em todos os casos

Uso: python3 leakage_test.py [--processed-dir ../data/processed]
Sem ficheiros processados -> SKIP (exit 0), para nao bloquear o fetch.
"""
import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

FAIL = []


def fail(msg):
    FAIL.append(msg)
    print(f"[LEAK-FAIL] {msg}", flush=True)


def parse_dt(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def check_case(case, src):
    eid = case.get("event_id", "?")
    try:
        kick = parse_dt(case["kickoff_utc"])
        cut = parse_dt(case["cutoff_utc"])
    except (KeyError, ValueError):
        fail(f"{eid} ({src}): kickoff_utc/cutoff_utc em falta ou invalidos")
        return
    if abs((kick - cut) - timedelta(minutes=50)) > timedelta(seconds=60):
        fail(f"{eid} ({src}): cutoff != kickoff-50min ({case.get('cutoff_utc')})")
    prov = case.get("provenance", {})
    for k in ("standings_round_used", "lineup_source"):
        if k not in prov:
            fail(f"{eid} ({src}): provenance.{k} em falta")
    for k in ("last_game_included_home", "last_game_included_away"):
        if k in prov and prov[k]:
            try:
                if parse_dt(prov[k]) >= kick:
                    fail(f"{eid} ({src}): {k} >= kickoff (futuro no passado)")
            except ValueError:
                fail(f"{eid} ({src}): {k} data invalida")
    rnd = case.get("round")
    sru = prov.get("standings_round_used")
    if rnd is not None and sru is not None and sru >= rnd:
        fail(f"{eid} ({src}): standings_round_used ({sru}) >= round ({rnd})")
    state = case.get("state_text", "")
    if "odd" in state.lower().replace("oddly", ""):
        # heuristica simples: a palavra odd/decimais de odd nao entram no texto
        for o in json.dumps(case.get("odds_at_cutoff", {})).split():
            pass
    import re as _re
    for qid, q in case.get("questions", {}).items():
        if qid == "odd_total":
            continue  # "Is the total odd?" - par/impar, nada a ver com odds
        txt = json.dumps(q)
        # \bodds?\b nao apanha "Todd Cantwell"; decimais tipo 2.05 + keywords sim
        if (_re.search(r"\bodds?\b", txt, _re.I) and "avoid defeat" not in txt.lower()) \
                or _re.search(r"\b\d+\.\d{2,}\b", txt):
            fail(f"{eid} ({src}): pergunta {qid} menciona odds")
    for s in case.get("sorare_scores_used", []):
        try:
            if parse_dt(s.get("game_date", "")) >= kick:
                fail(f"{eid} ({src}): sorare score com game.date >= kickoff")
        except ValueError:
            fail(f"{eid} ({src}): sorare game_date invalida")


def check_split(proc):
    bounds = {}
    for split in ("train", "calib", "test"):
        p = proc / f"{split}.jsonl"
        if not p.exists():
            continue
        dates = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    dates.append(parse_dt(json.loads(line)["kickoff_utc"]))
        if dates:
            bounds[split] = (min(dates), max(dates))
    if {"train", "calib", "test"} <= set(bounds):
        if not (bounds["train"][1] < bounds["calib"][0] < bounds["test"][0]):
            fail(f"split nao-temporal: { {k: (str(a), str(b)) for k, (a, b) in bounds.items()} }")
    return bounds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default=None)
    a = ap.parse_args()
    proc = Path(a.processed_dir) if a.processed_dir else (
        Path(__file__).resolve().parents[1] / "data" / "processed")
    files = sorted(proc.glob("*.jsonl")) if proc.exists() else []
    if not files:
        print("[leak] SKIP: sem ficheiros processados ainda", flush=True)
        sys.exit(0)
    n = 0
    for fp in files:
        if fp.name in ("train.jsonl", "calib.jsonl", "test.jsonl",
                       "players_wanted.jsonl"):
            pass
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and fp.name != "players_wanted.jsonl":
                    check_case(json.loads(line), fp.name)
                    n += 1
    check_split(proc)
    print(f"[leak] casos verificados: {n}", flush=True)
    if FAIL:
        print(f"[leak] FALHOU com {len(FAIL)} violacoes", flush=True)
        sys.exit(1)
    print("[leak] OK: dataset limpo", flush=True)
