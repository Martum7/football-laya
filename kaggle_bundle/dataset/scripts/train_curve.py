"""Treino da curva de aprendizagem: um modelo independente por checkpoint N.

Cada N = numero de CASOS (jogos/linhas do train.jsonl, cada um com ~70 perguntas).
Por defeito (--order suffix): os N casos MAIS RECENTES do train (janela que
termina mesmo antes do calib). N=0 = modelo base, nao treina.

Uso (correr NO .venv):
  ../.venv/bin/python train_curve.py --checkpoints 0,100,500,1000 --curve-dir ../data/models/curve_recent
  ../.venv/bin/python train_curve.py --checkpoints 100 --dry-run   # so conta items, sem treinar

Saida por N>0: data/models/curve/n_{N}/model.safetensors (+ configs copiadas do base
+ train_meta.json com hiperparams, sha do train.jsonl, n_items).
Resume: se model.safetensors ja existe, salta (a nao ser --overwrite).
Pausa/retoma a meio do N: o estado e guardado em n_{N}/resume.pt de
--save-every em --save-every steps (e no fim de cada epoca). Para parar,
mata o processo (Ctrl+C / kill); para continuar, corre o MESMO comando e
ele retoma sozinho. --fresh ignora o resume e recomeca.

Custo: treino independente => tempo total ~= soma(N) * epochs. N=14k ~ 1M items.
Comecar pequeno (100,500,1000) antes de lancar a curva toda.
"""
import argparse
import hashlib
import json
import random
import resource
import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="To copy construct from a tensor")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from config import PROCESSED  # noqa: E402

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "data" / "models" / "laya-multilingual-base"
CURVE_DIR = BASE_DIR / "data" / "models" / "curve"

DEFAULT_CHECKPOINTS = [0, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000,
                       7000, 8000, 9000, 10000, 12000, 14000]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gold_target(qtype, criteria, label):
    if qtype == "noul":
        probs = {"false": 1.0, "true": 0.0} if label == 0 else {"false": 0.0, "true": 1.0}
    elif qtype == "choice":
        probs = {k: (1.0 if k == label else 0.0) for k in criteria}
    else:
        raise ValueError(qtype)
    tot = sum(probs.values())
    keys = list(probs.keys())
    return [float(v) / tot for v in probs.values()], keys.index(max(probs, key=probs.get))


def build_items_for_cases(cases, tok, max_len, head_max_len):
    from laya.common import QTYPES, build_sequence, render_options
    items = []
    skipped = 0
    for row in cases:
        for qid, q in row["questions"].items():
            if qid not in row["labels"]:
                skipped += 1
                continue
            label = row["labels"][qid]
            qtype, crit = q["type"], q.get("criteria", {})
            try:
                target, lab = gold_target(qtype, crit, label)
            except ValueError:
                skipped += 1
                continue
            n_opt = len(render_options({"t": qtype, "crit": crit}))
            seq, markers = build_sequence(
                tok, row["state_text"],
                {"t": qtype, "ins": q["instructions"], "crit": crit},
                max_len, head_max_len)
            if len(markers) != n_opt:
                skipped += 1
                continue
            items.append({"ids": torch.LongTensor(seq), "markers": torch.LongTensor(markers),
                          "qtype": QTYPES[qtype], "target": target, "label": lab})
    # ids/markers como tensores CPU: 140k items em listas de int Python
    # rebentavam a RAM (~28B/int); em int64 sao ~8B/token.
    approx_mb = sum(it["ids"].numel() * 8 + it["markers"].numel() * 8
                    for it in items) / 1e6
    print(f"[export] items={len(items)} skip={skipped} ~{approx_mb:.0f}MB em ids/markers",
          flush=True)
    return items, skipped


def mps_gb():
    try:
        return torch.mps.current_allocated_memory() / 1e9
    except (AttributeError, RuntimeError):
        return float("nan")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one(n_cases, cases, tok, cfg, args, device, curve_dir):
    from laya.common import build_model, collate_items
    set_seed(args.seed)
    items, skipped = build_items_for_cases(cases, tok, cfg["max_len"], cfg["head_max_len"])
    print(f"[train n={n_cases}] casos={len(cases)} items={len(items)} skip={skipped}", flush=True)
    if args.dry_run or not items:
        return {"n_cases": n_cases, "n_items": len(items), "skipped": skipped,
                "dry_run": True}

    out = curve_dir / f"n_{n_cases}"
    out.mkdir(parents=True, exist_ok=True)
    resume_path = out / "resume.pt"

    from transformers import get_linear_schedule_with_warmup
    model = build_model(cfg, encoder_dir=str(MODEL_DIR / "encoder"), pretrained=True)
    from safetensors.torch import load_file
    resumed = resume_path.exists() and not args.fresh
    if not resumed:
        base_w = load_file(str(MODEL_DIR / "model.safetensors"))
        model.load_state_dict(base_w, strict=True)
    model.to(device)
    model.train()
    if args.grad_checkpoint:
        try:
            model.encoder.gradient_checkpointing_enable()
            print("[train] gradient checkpointing ON", flush=True)
        except (AttributeError, NotImplementedError) as exc:
            print(f"[train] gradient checkpointing NAO suportado ({exc}), a continuar sem",
                  flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    use_amp = bool(args.amp and device.type == "cuda")
    if args.amp and device.type != "cuda":
        print("[train] --amp so em CUDA, ignorado", flush=True)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("[train] AMP fp16 ON", flush=True)
    pad_id = tok.pad_token_id
    micro, accum, epochs = args.batch, args.accum, args.epochs
    eff_batch = micro * accum
    steps_per_epoch = (len(items) + eff_batch - 1) // eff_batch
    total_steps = steps_per_epoch * epochs
    print(f"[train n={n_cases}] micro_batch={micro} accum={accum} "
          f"eff_batch={eff_batch} steps/epoca={steps_per_epoch} total={total_steps}",
          flush=True)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)

    start_ep, start_pos, eff_step = 0, 0, 0
    carry_loss, carry_n = 0.0, 0
    if resumed:
        ck = torch.load(str(resume_path), map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        if scaler is not None and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        start_ep, start_pos = ck["epoch"], ck["pos"]
        eff_step, carry_loss, carry_n = ck["eff_step"], ck["run_loss"], ck["run_n"]
        try:
            torch.set_rng_state(ck["cpu_rng"])
        except (KeyError, RuntimeError):
            pass
        print(f"[train n={n_cases}] RETOMA: epoca {start_ep + 1}/{epochs} "
              f"a partir do item {start_pos}/{len(items)} (step {eff_step})", flush=True)

    def save_resume(epoch, pos, rl, rn):
        torch.save({
            "model": {k: v.cpu() for k, v in model.state_dict().items()},
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch, "pos": pos, "eff_step": eff_step,
            "run_loss": rl, "run_n": rn,
            "cpu_rng": torch.get_rng_state(),
            "args": {"lr": args.lr, "wd": args.wd, "seed": args.seed,
                     "batch": args.batch, "accum": args.accum, "epochs": args.epochs},
        }, str(resume_path))

    idx = np.arange(len(items))
    for ep in range(start_ep, epochs):
        rng = np.random.RandomState(args.seed + ep)
        rng.shuffle(idx)
        opt.zero_grad()
        if ep == start_ep and start_pos:
            run_loss, run_n = carry_loss, carry_n
        else:
            run_loss, run_n = 0.0, 0
        pending = 0
        s0 = start_pos if ep == start_ep else 0
        for s in range(s0, len(items), micro):
            sel = [items[i] for i in idx[s:s + micro]]
            b = collate_items([sel], pad_id)
            ids = b["input_ids"].to(device)
            att = b["attention_mask"].to(device)
            mpos = b["marker_pos"].to(device)
            mmsk = b["marker_mask"].to(device)
            qt = b["qtype"].to(device)
            tgt = b["target"].to(device)
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits, _ = model(ids, att, mpos, mmsk, qt)
            else:
                logits, _ = model(ids, att, mpos, mmsk, qt)
            logp = torch.log_softmax(logits.float(), -1)
            loss = -(tgt * logp * mmsk.float()).sum(-1).mean() / accum
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            run_loss += float(loss.detach()) * accum * len(sel)
            run_n += len(sel)
            pending += 1
            last = (s + micro >= len(items))
            if pending % accum == 0 or last:
                if use_amp:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()
                opt.zero_grad()
                eff_step += 1
                if eff_step % 10 == 0 or last:
                    from datetime import datetime as _dt
                    _ts = _dt.now().strftime("%H:%M:%S")
                    msg = (f"[{_ts}][train n={n_cases}] ep={ep + 1}/{epochs} "
                           f"step={eff_step}/{total_steps} loss={run_loss / max(run_n, 1):.4f}")
                    if device.type == "mps":
                        msg += f" mps={mps_gb():.2f}GB"
                    print(msg, flush=True)
                if eff_step % args.save_every == 0:
                    save_resume(ep, s + len(sel), run_loss, run_n)
                    print(f"[train n={n_cases}] checkpoint guardado (step {eff_step})",
                          flush=True)
                if device.type == "mps" and eff_step % 500 == 0:
                    torch.mps.empty_cache()
        peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
        print(f"[train n={n_cases}] epoca {ep + 1}/{epochs} loss={run_loss / run_n:.4f} "
              f"pico_ram={peak_gb:.1f}GB", flush=True)
        save_resume(ep + 1, 0, 0.0, 0)
        if device.type == "mps":
            torch.mps.empty_cache()

    out = curve_dir / f"n_{n_cases}"
    out.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    save_file(model.state_dict(), str(out / "model.safetensors"))
    resume_path.unlink(missing_ok=True)  # retoma ja nao e precisa
    shutil.copy(MODEL_DIR / "rl_agent_config.json", out / "rl_agent_config.json")
    shutil.copytree(MODEL_DIR / "tokenizer", out / "tokenizer", dirs_exist_ok=True)
    shutil.copytree(MODEL_DIR / "encoder", out / "encoder", dirs_exist_ok=True)
    meta = {"n_cases": n_cases, "n_items": len(items), "skipped": skipped,
            "order": args.order, "subset_from": cases[0]["kickoff_utc"] if cases else None,
            "subset_to": cases[-1]["kickoff_utc"] if cases else None,
            "epochs": epochs, "batch": args.batch, "accum": args.accum,
            "eff_batch": args.batch * args.accum,
            "lr": args.lr, "wd": args.wd,
            "seed": args.seed, "base": "laya-multilingual-base",
            "train_sha256": sha256_file(PROCESSED / "train.jsonl")}
    with open(out / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[train n={n_cases}] DONE -> {out}", flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", default=",".join(map(str, DEFAULT_CHECKPOINTS)))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1,
                    help="passos de acumulacao; batch efectivo = batch*accum")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="gradient checkpointing no encoder: menos RAM, ~30% mais lento/step")
    ap.add_argument("--amp", action="store_true",
                    help="fp16 mixed precision (so CUDA; 2-4x mais rapido na T4)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--fresh", action="store_true",
                    help="ignora resume.pt e recomeca o N do zero")
    ap.add_argument("--save-every", type=int, default=500,
                    help="guardar estado de retoma de X em X steps efectivos")
    ap.add_argument("--order", choices=["prefix", "suffix"], default="suffix",
                    help="prefix=os N mais antigos; suffix=os N mais recentes (defeito)")
    ap.add_argument("--curve-dir", default=None,
                    help="onde guardar n_N (defeito: data/models/curve)")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from laya.agent import _fix_tokenizer_config
    _fix_tokenizer_config(str(MODEL_DIR))
    with open(MODEL_DIR / "rl_agent_config.json") as f:
        cfg = json.load(f)
    cfg["max_len"], cfg["head_max_len"] = cfg.get("max_len", 1024), cfg.get("head_max_len", 256)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR / "tokenizer")

    with open(PROCESSED / "train.jsonl", encoding="utf-8") as f:
        all_cases = [json.loads(l) for l in f if l.strip()]
    # train.jsonl ja e cronologico (verificado); reforcar por seguranca:
    all_cases.sort(key=lambda r: r["kickoff_utc"])
    print(f"[train] train.jsonl: {len(all_cases)} casos "
          f"({all_cases[0]['kickoff_utc']} .. {all_cases[-1]['kickoff_utc']})", flush=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[train] device={device}", flush=True)

    want = [int(x) for x in a.checkpoints.split(",") if x.strip() != ""]
    curve_dir = Path(a.curve_dir) if a.curve_dir else CURVE_DIR
    curve_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] order={a.order} curve_dir={curve_dir}", flush=True)
    for n in want:
        if n == 0:
            print("[train n=0] baseline = modelo base, sem treino.", flush=True)
            continue
        n = min(n, len(all_cases))
        out = curve_dir / f"n_{n}" / "model.safetensors"
        if out.exists() and not a.overwrite and not a.dry_run:
            print(f"[train n={n}] SKIP (ja existe {out}, usa --overwrite)", flush=True)
            continue
        subset = all_cases[:n] if a.order == "prefix" else all_cases[-n:]
        print(f"[train n={n}] janela: {subset[0]['kickoff_utc']} .. {subset[-1]['kickoff_utc']}",
              flush=True)
        train_one(n, subset, tok, cfg, a, device, curve_dir)


if __name__ == "__main__":
    main()
