"""Avaliacao da curva: por checkpoint N, calibra no calib fixo e mede no test fixo.

Protocolo por checkpoint (N=0 = modelo base sem treino):
  1. forward em calib.jsonl -> fit_temperature_map (temperaturas por checkpoint)
  2. forward em test.jsonl com temperaturas -> metricas de accuracy, NLL,
     Brier, ECE (raw e calibradas) + foco em confianca dos nouls
     (acc, ECE, confianca media, tabela de reliability por bins)
  3. junta uma linha em data/processed/curve/metrics.jsonl (+ .csv)

EV/ROI nao e calculado aqui: fica para validacao posterior em jogos reais,
em tempo real. (Se um dia for preciso offline, usar --with-ev.)

Uso (NO .venv):
  ../.venv/bin/python eval_curve.py --checkpoints 0,100,500
  ../.venv/bin/python eval_curve.py --checkpoints 0,100 --max-cases 50  # smoke test rapido
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from config import PROCESSED  # noqa: E402

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "data" / "models" / "laya-multilingual-base"
CURVE_DIR = BASE_DIR / "data" / "models" / "curve"
OUT_DIR = BASE_DIR / "data" / "processed" / "curve"

EV_THRESHOLDS = [0.0, 0.02, 0.05, 0.10]


def ckpt_dir(n):
    return MODEL_DIR if n == 0 else CURVE_DIR / f"n_{n}"


def load_model_and_tok(n, device):
    from transformers import AutoTokenizer
    from laya.agent import _fix_tokenizer_config
    from laya.common import build_model
    from safetensors.torch import load_file
    d = ckpt_dir(n)
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


def forward_rows(model, tok, rows, max_len, head_max_len, device, batch=64):
    """Devolve lista de dicts por item: logits, target, qtype, k, qid, odds, label."""
    from laya.common import QTYPES, build_sequence, collate_items
    # monta items com indice para re-anexar qid/odds
    flat, meta = [], []
    for ri, row in enumerate(rows):
        if ri % 250 == 0:
            print(f"[eval] tokenizar caso {ri}/{len(rows)}...", flush=True)
        odds = row.get("odds_at_cutoff") or {}
        for qid, q in row["questions"].items():
            if qid not in row["labels"]:
                continue
            label = row["labels"][qid]
            qtype, crit = q["type"], q.get("criteria", {})
            if qtype == "noul":
                keys = ["false", "true"]
                tgt = [0.0, 1.0] if label == 1 else [1.0, 0.0]
            else:
                keys = list(crit.keys())
                try:
                    tgt = [1.0 if k == label else 0.0 for k in keys]
                except Exception:
                    continue
            seq, markers = build_sequence(
                tok, row["state_text"],
                {"t": qtype, "ins": q["instructions"], "crit": crit},
                max_len, head_max_len)
            if len(markers) != len(keys):
                continue
            flat.append({"ids": seq, "markers": markers,
                         "qtype": QTYPES[qtype], "target": tgt, "label": keys.index(
                             ("true" if label == 1 else "false") if qtype == "noul" else label)})
            meta.append({"qid": qid, "odds": odds, "keys": keys})
    out = []
    nb = (len(flat) + batch - 1) // batch
    with torch.no_grad():
        for bi, s in enumerate(range(0, len(flat), batch)):
            if bi % 50 == 0:
                print(f"[eval] forward batch {bi}/{nb}...", flush=True)
            sel = flat[s:s + batch]
            b = collate_items([sel], tok.pad_token_id)
            logits, _ = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                              b["marker_pos"].to(device), b["marker_mask"].to(device),
                              b["qtype"].to(device))
            lg = logits.float().cpu().numpy()
            tg = b["target"].numpy()
            mm = b["marker_mask"].numpy()
            qt = b["qtype"].numpy()
            for j in range(len(sel)):
                k = int(mm[j].sum())
                out.append({"logits": lg[j, :k], "target": tg[j, :k],
                            "qtype": int(qt[j]), "k": k,
                            "qid": meta[s + j]["qid"], "odds": meta[s + j]["odds"],
                            "keys": meta[s + j]["keys"]})
    return out


def softmax(z):
    z = np.asarray(z, float)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def metrics_for(items, temperature, tbo):
    from laya.common import ece_score, temp_bucket, QTYPE_NAMES
    conf, corr, nll, brier, is_noul = [], [], [], [], []
    acc_noul_c, acc_noul_n, acc_ch_c, acc_ch_n = 0, 0, 0, 0
    per_bucket = {}
    for it in items:
        t = tbo.get(temp_bucket(it["qtype"], it["k"]), temperature[it["qtype"]])
        p = softmax(it["logits"] / t)
        y = int(np.argmax(it["target"]))
        pred = int(np.argmax(p))
        ok = int(pred == y)
        c = float(p.max())
        conf.append(c)
        corr.append(ok)
        nll.append(float(-np.log(max(p[y], 1e-12))))
        oh = np.zeros_like(p)
        oh[y] = 1.0
        brier.append(float(np.mean((p - oh) ** 2)))
        qtn = QTYPE_NAMES[it["qtype"]]
        is_noul.append(qtn == "noul")
        if qtn == "noul":
            acc_noul_c += ok
            acc_noul_n += 1
        elif qtn == "choice":
            acc_ch_c += ok
            acc_ch_n += 1
        b = temp_bucket(it["qtype"], it["k"])
        per_bucket.setdefault(b, []).append((c, ok))
    conf, corr = np.array(conf), np.array(corr)
    nconf = np.array([c for c, q in zip(conf, is_noul) if q])
    ncorr = np.array([o for o, q in zip(corr, is_noul) if q])
    # reliability dos nouls: 5 bins de confianca com n, accuracy e confianca media
    noul_rel = []
    for lo, hi in ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)):
        sel = (nconf >= lo) & (nconf < hi)
        noul_rel.append({"bin": f"{lo:.1f}-{min(hi, 1.0):.1f}",
                         "n": int(sel.sum()),
                         "acc": float(ncorr[sel].mean()) if sel.any() else float("nan"),
                         "conf": float(nconf[sel].mean()) if sel.any() else float("nan")})
    return {
        "n": len(items),
        "acc": float(corr.mean()) if len(corr) else float("nan"),
        "acc_noul": acc_noul_c / acc_noul_n if acc_noul_n else float("nan"),
        "acc_choice": acc_ch_c / acc_ch_n if acc_ch_n else float("nan"),
        "nll": float(np.mean(nll)) if nll else float("nan"),
        "brier": float(np.mean(brier)) if brier else float("nan"),
        "ece": float(ece_score(conf, corr)),
        "ece_noul": float(ece_score(nconf, ncorr)) if len(nconf) else float("nan"),
        "noul_conf_mean": float(nconf.mean()) if len(nconf) else float("nan"),
        "noul_conf_correct": float(nconf[ncorr == 1].mean()) if (ncorr == 1).any() else float("nan"),
        "noul_conf_wrong": float(nconf[ncorr == 0].mean()) if (ncorr == 0).any() else float("nan"),
        "noul_reliability": noul_rel,
        "ece_by_bucket": {b: float(ece_score(np.array([c for c, _ in v]),
                                             np.array([o for _, o in v])))
                          for b, v in per_bucket.items()},
    }


def ev_for(items, temperature, tbo, thresholds=EV_THRESHOLDS):
    """EV so para qids com odds mapeaveis. Flat stake 1u por aposta com edge>thr."""
    from laya.common import temp_bucket
    recs = []  # (edge, win, odds)
    for it in items:
        t = tbo.get(temp_bucket(it["qtype"], it["k"]), temperature[it["qtype"]])
        p = softmax(it["logits"] / t)
        y = int(np.argmax(it["target"]))
        o = it["odds"] or {}
        odd = None
        if it["qid"] == "match_result" and "1x2" in o and len(o["1x2"]) == 3:
            # aposta no lado predito pelo modelo
            pred = int(np.argmax(p))
            key = it["keys"][pred]
            pos = {"home": 0, "draw": 1, "away": 2}[key]
            try:
                odd = float(o["1x2"][pos])
            except (ValueError, TypeError, IndexError):
                odd = None
            prob = float(p[pred])
            win = int(pred == y)
        elif it["qid"] in ("over_25", "under_25") and "ou25" in o and len(o["ou25"]) == 2:
            # noul: p[1] = prob true. over_25 true <-> ou25[0]; under_25 true <-> ...?
            # under_25 e o complementar de over_25: P(under)=1-P(over); odd under=ou25[1].
            prob_true = float(p[1])
            try:
                if it["qid"] == "over_25":
                    odd = float(o["ou25"][0])
                    prob, win = prob_true, int(y == 1)
                else:
                    odd = float(o["ou25"][1])
                    prob, win = 1.0 - prob_true, int(y == 1)
                    # prob aqui = prob da aposta under; win = acertou under
                    # mas p e 2-way: pred under = p[0]>p[1]? Para EV direccional simples:
                    # aposta sempre no lado do qid (under) com prob 1-p_true.
                    pass
            except (ValueError, TypeError, IndexError):
                odd = None
        else:
            continue
        if odd is None or not np.isfinite(odd) or odd <= 1.0:
            continue
        recs.append((float(prob * odd - 1.0), win, float(odd)))
    out = {"n_eligible": len(recs)}
    for thr in thresholds:
        sel = [(e, w, od) for e, w, od in recs if e > thr]
        if not sel:
            out[f"thr_{thr}"] = {"n_bets": 0, "winrate": float("nan"),
                                 "roi": float("nan"), "profit_u": 0.0}
            continue
        profit = sum((od - 1.0) if w else -1.0 for _, w, od in sel)
        out[f"thr_{thr}"] = {"n_bets": len(sel),
                             "winrate": float(sum(w for _, w, _ in sel) / len(sel)),
                             "roi": float(profit / len(sel)),
                             "profit_u": float(profit)}
    return out


def eval_one(n, device, max_cases=0, batch=64, with_ev=False):
    from laya.calibrate import fit_temperature_map
    model, tok, cfg = load_model_and_tok(n, device)
    max_len, hml = cfg.get("max_len", 1024), cfg.get("head_max_len", 256)

    def load(split):
        with open(PROCESSED / f"{split}.jsonl", encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        rows.sort(key=lambda r: r["kickoff_utc"])
        return rows[:max_cases] if max_cases else rows

    calib_rows, test_rows = load("calib"), load("test")
    print(f"[eval n={n}] calib={len(calib_rows)} test={len(test_rows)}", flush=True)

    calib_items = forward_rows(model, tok, calib_rows, max_len, hml, device, batch)
    records = [(it["qtype"], it["logits"], it["target"], it["k"]) for it in calib_items]
    fit = fit_temperature_map(records, compute_ece=True, seed=0)
    temperature = [float(x) for x in fit["temperature"]]
    tbo = {k: float(v) for k, v in fit["temperature_by_options"].items()}

    test_items = forward_rows(model, tok, test_rows, max_len, hml, device, batch)
    raw = metrics_for(test_items, [1.0, 1.0, 1.0], {})
    cal = metrics_for(test_items, temperature, tbo)
    ev = ev_for(test_items, temperature, tbo) if with_ev else None

    res = {"checkpoint": n,
           "model": "base" if n == 0 else f"n_{n}",
           "max_cases": max_cases, "batch": batch,
           "n_calib_items": len(calib_items), "n_test_items": len(test_items),
           "temperature": temperature, "temperature_by_options": tbo,
           "n_by_bucket": fit.get("n_by_bucket", {}),
           "ece_report_calib": fit.get("report", {}),
           "raw": raw, "calibrated": cal, "ev": ev}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"metrics_n_{n}.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"[eval n={n}] acc={cal['acc']:.4f} acc_noul={cal['acc_noul']:.4f} "
          f"brier={cal['brier']:.4f} ece={cal['ece']:.4f} ece_noul={cal['ece_noul']:.4f} "
          f"noul_conf={cal['noul_conf_mean']:.3f} "
          f"(ok={cal['noul_conf_correct']:.3f}/err={cal['noul_conf_wrong']:.3f})", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", default="0,100,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,12000,14000")
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--with-ev", action="store_true",
                    help="calcula EV offline com odds_at_cutoff (fora do protocolo actual)")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-avalia mesmo quem ja tem metrics_n_N.json")
    ap.add_argument("--curve-dir", default=None,
                    help="onde estao os n_N (defeito: data/models/curve)")
    ap.add_argument("--metrics-dir", default=None,
                    help="onde escrever metrics.* (defeito: data/processed/curve)")
    a = ap.parse_args()

    global CURVE_DIR, OUT_DIR
    if a.curve_dir:
        CURVE_DIR = Path(a.curve_dir)
    if a.metrics_dir:
        OUT_DIR = Path(a.metrics_dir)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[eval] device={device}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for n in [int(x) for x in a.checkpoints.split(",") if x.strip() != ""]:
        d = ckpt_dir(n)
        if not (d / "model.safetensors").exists():
            print(f"[eval n={n}] SKIP (sem {d / 'model.safetensors'}; treina primeiro)", flush=True)
            continue
        done = OUT_DIR / f"metrics_n_{n}.json"
        if done.exists() and not a.overwrite:
            try:
                prev = json.loads(open(done).read())
                if prev.get("max_cases", 0) == (a.max_cases or 0):
                    print(f"[eval n={n}] SKIP (ja avaliado em {done})", flush=True)
                    results.append(prev)
                    continue
            except (ValueError, OSError):
                pass
        try:
            results.append(eval_one(n, device, a.max_cases, a.batch, a.with_ev))
        except Exception as exc:
            print(f"[eval n={n}] ERRO: {exc}", flush=True)
    if results:
        with open(OUT_DIR / "metrics.jsonl", "w") as f:
            for r in sorted(results, key=lambda r: r["checkpoint"]):
                f.write(json.dumps(r) + "\n")
        keys = ["checkpoint", "acc_raw", "acc_noul_cal", "nll_cal", "brier_cal",
                "ece_raw", "ece_cal", "ece_noul_cal", "noul_conf_cal"]
        if a.with_ev:
            keys += ["ev_bets_05", "ev_roi_05"]
        with open(OUT_DIR / "metrics.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(keys)
            for r in sorted(results, key=lambda r: r["checkpoint"]):
                row = [r["checkpoint"],
                       round(r["raw"]["acc"], 4), round(r["calibrated"]["acc_noul"], 4),
                       round(r["calibrated"]["nll"], 4), round(r["calibrated"]["brier"], 4),
                       round(r["raw"]["ece"], 4), round(r["calibrated"]["ece"], 4),
                       round(r["calibrated"]["ece_noul"], 4),
                       round(r["calibrated"]["noul_conf_mean"], 4)]
                if a.with_ev and r["ev"]:
                    row += [r["ev"].get("thr_0.05", {}).get("n_bets"),
                            round(r["ev"].get("thr_0.05", {}).get("roi", float("nan")), 4)
                            if r["ev"].get("thr_0.05", {}).get("n_bets") else ""]
                w.writerow(row)
        print(f"[eval] DONE {len(results)} checkpoints -> {OUT_DIR / 'metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
