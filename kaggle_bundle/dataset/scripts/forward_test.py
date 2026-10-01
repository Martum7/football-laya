"""Forward so-teste em streaming (smoke test do n_2000 no Kaggle).

Problema que resolve: `eval_curve.py:forward_rows` tokeniza os 1000 casos
(~70k items) todos de uma vez e depois faz forward com `--batch 128`.
128 x 1024 tokens rebenta a T4 16GB (o treino usa batch 32 com
--grad-checkpoint + --amp pelo mesmo motivo).

Este script faz o oposto:
- le o test.jsonl em blocos de casos (--case-chunk, defeito 50),
- tokeniza so o bloco corrente,
- forward em micro-batches pequenos (--batch, defeito 16) com no_grad
  (+ autocast fp16 em CUDA),
- liberta tudo antes do bloco seguinte.

Nao calibra, nao calcula metricas. So prova que o n_2000 carrega e faz
forward no teste.

Uso local:
  ../.venv/bin/python forward_test.py --checkpoint 2000 --max-cases 10 --batch 8

Uso Kaggle (paths do notebook curve_eval_only):
  !python /kaggle/working/fb/scripts/forward_test.py --checkpoint 2000 \
      --curve-dir /kaggle/working/fb/data/models/curve_kaggle \
      --processed-dir /kaggle/working/fb/data/processed \
      --batch 16 --case-chunk 50 --amp
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "data" / "models" / "laya-multilingual-base"
CURVE_DIR = BASE_DIR / "data" / "models" / "curve"


def ckpt_dir(n, curve_dir):
    return MODEL_DIR if n == 0 else Path(curve_dir) / f"n_{n}"


def load_model_and_tok(n, device, curve_dir):
    from transformers import AutoTokenizer
    from laya.agent import _fix_tokenizer_config
    from laya.common import build_model
    from safetensors.torch import load_file
    d = ckpt_dir(n, curve_dir)
    if not (d / "model.safetensors").exists():
        raise FileNotFoundError(f"sem pesos em {d / 'model.safetensors'}")
    _fix_tokenizer_config(str(d))
    with open(d / "rl_agent_config.json") as f:
        cfg = json.load(f)
    tok = AutoTokenizer.from_pretrained(d / "tokenizer")
    enc = d / "encoder"
    model = build_model(cfg, encoder_dir=str(enc) if enc.exists() else None, pretrained=False)
    model.load_state_dict(load_file(str(d / "model.safetensors")), strict=True)
    model.to(device)
    model.eval()
    return model, tok, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=int, default=2000)
    ap.add_argument("--curve-dir", default=str(CURVE_DIR))
    ap.add_argument("--processed-dir", default=None,
                    help="onde esta test.jsonl (defeito: data/processed)")
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16,
                    help="micro-batch GPU. T4 16GB: 16 seguro, 32 limite.")
    ap.add_argument("--case-chunk", type=int, default=50,
                    help="casos tokenizados de cada vez (defeito 50 -> ~3500 items)")
    ap.add_argument("--amp", action="store_true",
                    help="autocast fp16 em CUDA (recomendado na T4)")
    ap.add_argument("--no-amp", action="store_true")
    a = ap.parse_args()

    processed = Path(a.processed_dir) if a.processed_dir else BASE_DIR / "data" / "processed"
    with open(processed / "test.jsonl", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    rows.sort(key=lambda r: r["kickoff_utc"])
    if a.max_cases:
        rows = rows[:a.max_cases]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    use_amp = a.amp and device.type == "cuda" and not a.no_amp
    # no Kaggle queremos AMP por defeito quando ha CUDA e o user nao disse nada:
    if device.type == "cuda" and not a.amp and not a.no_amp:
        use_amp = True
    print(f"[fwd] device={device} amp={use_amp} casos={len(rows)} "
          f"ckpt={a.checkpoint} batch={a.batch} chunk={a.case_chunk}", flush=True)
    if device.type == "cuda":
        print(f"[fwd] gpu={torch.cuda.get_device_name(0)} "
              f"mem_total={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB",
              flush=True)

    from laya.common import QTYPES, build_sequence, collate_items

    model, tok, cfg = load_model_and_tok(a.checkpoint, device, a.curve_dir)
    max_len, hml = cfg.get("max_len", 1024), cfg.get("head_max_len", 256)
    pad_id = tok.pad_token_id
    print(f"[fwd] modelo OK ({'base' if a.checkpoint == 0 else f'n_{a.checkpoint}'}) "
          f"max_len={max_len} head={hml}", flush=True)

    total_items, total_skip, total_batches = 0, 0, 0
    t0 = time.time()
    sample = None

    with torch.no_grad():
        for c0 in range(0, len(rows), a.case_chunk):
            chunk = rows[c0:c0 + a.case_chunk]
            flat = []
            skip = 0
            for row in chunk:
                for qid, q in row["questions"].items():
                    if qid not in row["labels"]:
                        skip += 1
                        continue
                    label = row["labels"][qid]
                    qtype, crit = q["type"], q.get("criteria", {})
                    if qtype == "noul":
                        keys = ["false", "true"]
                    else:
                        keys = list(crit.keys())
                        if label not in keys:
                            skip += 1
                            continue
                    seq, markers = build_sequence(
                        tok, row["state_text"],
                        {"t": qtype, "ins": q["instructions"], "crit": crit},
                        max_len, hml)
                    if len(markers) != len(keys):
                        skip += 1
                        continue
                    flat.append({"ids": seq, "markers": markers,
                                 "qtype": QTYPES[qtype]})
            # forward do bloco em micro-batches; em OOM, desce o batch a metade e repete
            bsize = a.batch
            bi = 0
            while bi < len(flat):
                sel = flat[bi:bi + bsize]
                try:
                    b = collate_items([sel], pad_id)
                    if use_amp:
                        with torch.amp.autocast("cuda", dtype=torch.float16):
                            logits, _ = model(b["input_ids"].to(device),
                                              b["attention_mask"].to(device),
                                              b["marker_pos"].to(device),
                                              b["marker_mask"].to(device),
                                              b["qtype"].to(device))
                    else:
                        logits, _ = model(b["input_ids"].to(device),
                                          b["attention_mask"].to(device),
                                          b["marker_pos"].to(device),
                                          b["marker_mask"].to(device),
                                          b["qtype"].to(device))
                    if sample is None:
                        k = int(b["marker_mask"][0].sum())
                        sample = logits[0, :k].float().cpu().tolist()
                    if device.type == "cuda":
                        del logits
                    total_batches += 1
                    bi += bsize
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower() or bsize <= 1:
                        raise
                    torch.cuda.empty_cache()
                    bsize = max(1, bsize // 2)
                    print(f"[fwd] OOM -> batch agora {bsize}", flush=True)
            total_items += len(flat)
            total_skip += skip
            dt = time.time() - t0
            mem = (f" gpu_mem={torch.cuda.memory_allocated() / 1e9:.2f}GB"
                   if device.type == "cuda" else "")
            print(f"[fwd] casos {min(c0 + a.case_chunk, len(rows))}/{len(rows)} "
                  f"items={total_items} skip={total_skip} "
                  f"batches={total_batches} {dt:.0f}s{mem}", flush=True)
            del flat

    dt = time.time() - t0
    print(f"[fwd] DONE ckpt={a.checkpoint} casos={len(rows)} items={total_items} "
          f"skip={total_skip} batches={total_batches} tempo={dt:.1f}s", flush=True)
    if sample is not None:
        print(f"[fwd] sample_logits(k={len(sample)}): "
              f"{[round(x, 3) for x in sample]}", flush=True)


if __name__ == "__main__":
    main()
