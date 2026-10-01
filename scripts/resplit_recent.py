"""Re-split por recencia: os casos mais recentes passam a calib+test.

Junta train/calib/test.jsonl, ordena por kickoff_utc e redefine:
  test  = ultimos 1000 casos (os mais recentes)
  calib = 1000 anteriores
  train = todo o resto (pool de onde a curva tira os N mais recentes)

Motivo: treinar/avaliar em 2015-2023 para apostar "hoje" e enviesado;
a distribuicao do futebol muda (equipas, jogadores, estilos).

Seguranca:
  - backup dos 3 ficheiros originais em data/processed/split_backup_<data>/
  - ordem temporal estrita train < calib < test (por construcao)
  - corre leakage_test.py no fim e falha se ele falhar
  - verifica que nenhum event_id aparece em dois splits

Uso (sem GPU, rapido):
  ../.venv/bin/python resplit_recent.py [--n-calib 1000] [--n-test 1000]
"""
import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import PROCESSED  # noqa: E402


def load(name):
    with open(PROCESSED / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-calib", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=1000)
    a = ap.parse_args()

    all_rows = []
    for name in ("train", "calib", "test"):
        for r in load(name):
            r["_old_split"] = name
            all_rows.append(r)
    all_rows.sort(key=lambda r: r["kickoff_utc"])
    print(f"[resplit] total: {len(all_rows)} casos "
          f"({all_rows[0]['kickoff_utc']} .. {all_rows[-1]['kickoff_utc']})", flush=True)

    # mesma mid FotMob reutilizada em 2 jogos distintos (3 casos em 18022):
    # ficam no mesmo split, mas para a curva nao pode haver o mesmo event_id
    # em dois splits -> dedup, fica o kickoff mais antigo.
    seen = {}
    deduped = []
    for r in all_rows:
        if r["event_id"] in seen:
            old = seen[r["event_id"]]
            keep = min(old, r, key=lambda x: x["kickoff_utc"])
            drop = r if keep is old else old
            print(f"[resplit] duplicado {r['event_id']}: fica {keep['kickoff_utc']} "
                  f"({keep['_old_split']}), cai {drop['kickoff_utc']} ({drop['_old_split']})",
                  flush=True)
            seen[r["event_id"]] = keep
        else:
            seen[r["event_id"]] = r
    all_rows = sorted(seen.values(), key=lambda r: r["kickoff_utc"])
    print(f"[resplit] apos dedup: {len(all_rows)} casos", flush=True)

    n = len(all_rows)
    assert a.n_calib + a.n_test < n, "calib+test maior que o dataset"
    new_test = all_rows[n - a.n_test:]
    new_calib = all_rows[n - a.n_test - a.n_calib:n - a.n_test]
    new_train = all_rows[:n - a.n_test - a.n_calib]

    # ordem temporal estrita
    assert new_train[-1]["kickoff_utc"] < new_calib[0]["kickoff_utc"] <= new_calib[-1]["kickoff_utc"]
    assert new_calib[-1]["kickoff_utc"] < new_test[0]["kickoff_utc"]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    backup = PROCESSED / f"split_backup_{stamp}"
    backup.mkdir(parents=True, exist_ok=True)
    for name in ("train", "calib", "test"):
        shutil.copy(PROCESSED / f"{name}.jsonl", backup / f"{name}.jsonl")
    print(f"[resplit] backup em {backup}", flush=True)

    for name, rows in (("train", new_train), ("calib", new_calib), ("test", new_test)):
        for r in rows:
            r.pop("_old_split", None)
        with open(PROCESSED / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[resplit] {name}: {len(rows)} casos "
              f"({rows[0]['kickoff_utc']} .. {rows[-1]['kickoff_utc']})", flush=True)

    r = subprocess.run([sys.executable, str(Path(__file__).parent / "leakage_test.py")],
                       capture_output=True, text=True)
    print(r.stdout[-800:], flush=True)
    if r.returncode != 0:
        print(r.stderr[-500:], flush=True)
        raise SystemExit("leakage_test FALHOU - splits novos invalidos")


if __name__ == "__main__":
    main()
