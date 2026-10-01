"""Export train_items.pt no formato do script oficial MPS.

Le data/processed/train.jsonl -> tokeniza (state, pergunta) com
laya.common.build_sequence (mesma regra do oficial: skip se markers !=
n_options) -> torch.save items + meta p/ cache.
Corre NO .venv: ../.venv/bin/python export_laya_items.py [--split train]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from config import PROCESSED  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parents[1] / "data" / "models" / \
    "laya-multilingual-base"


def gold_probs(qtype, criteria, label):
    if qtype == "noul":
        return ({"false": 1.0, "true": 0.0} if label == 0
                else {"false": 0.0, "true": 1.0})
    if qtype == "choice":
        return {k: (1.0 if k == label else 0.0) for k in criteria}
    raise ValueError(qtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-items", type=int, default=0)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from laya.agent import _fix_tokenizer_config
    from laya.common import QTYPES, build_sequence, render_options

    _fix_tokenizer_config(str(MODEL_DIR))
    with open(MODEL_DIR / "rl_agent_config.json") as f:
        cfg = json.load(f)
    max_len, head_max_len = cfg.get("max_len", 1024), cfg.get("head_max_len", 256)
    print(f"[export] max_len={max_len} head_max_len={head_max_len}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR / "tokenizer")

    items, skipped = [], 0
    with open(PROCESSED / f"{a.split}.jsonl", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if a.max_items:
        rows = rows[:a.max_items]
    for ri, row in enumerate(rows):
        if ri % 2000 == 0:
            print(f"[export] {ri}/{len(rows)} items={len(items)} skip={skipped}",
                  flush=True)
        for qid, q in row["questions"].items():
            if qid not in row["labels"]:
                skipped += 1
                continue
            label = row["labels"][qid]
            qtype = q["type"]
            crit = q.get("criteria", {})
            try:
                probs = gold_probs(qtype, crit, label)
            except ValueError:
                skipped += 1
                continue
            tot = sum(probs.values())
            target = [float(v) / tot for v in probs.values()]
            keys = list(probs.keys())
            lab = keys.index(max(probs, key=probs.get))
            n_opt = len(render_options({"t": qtype, "crit": crit}))
            seq, markers = build_sequence(
                tok, row["state_text"],
                {"t": qtype, "ins": q["instructions"], "crit": crit},
                max_len, head_max_len)
            if len(markers) != n_opt:
                skipped += 1
                continue
            items.append({"ids": seq, "markers": markers,
                          "qtype": QTYPES[qtype], "target": target, "label": lab})
    out = PROCESSED / f"{a.split}_items.pt"
    torch.save(items, out)
    meta = {"model_dir": str(MODEL_DIR.resolve()), "max_len": max_len,
            "head_max_len": head_max_len, "dataset": "football-laya",
            "split": a.split, "n": len(items), "skipped": skipped}
    with open(str(out) + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[export] DONE {len(items)} items -> {out} (skip={skipped})", flush=True)


if __name__ == "__main__":
    main()
