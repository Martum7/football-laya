"""Utilidades partilhadas: rate limiter, http, resume, manifest."""
import json
import random
import threading
import time
import urllib.request
from pathlib import Path

from config import USER_AGENTS


class TokenBucket:
    """Rate limiter thread-safe: max `rps` pedidos/segundo."""

    def __init__(self, rps):
        self._interval = 1.0 / max(rps, 0.01)
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            if now < self._next:
                time.sleep(self._next - now)
                now = time.monotonic()
            self._next = max(now, self._next) + self._interval


def load_dotenv(path=None):
    """Carrega .env simples (KEY=valor, sem dependencias).
    Variaveis de ambiente reais têm prioridade sobre o .env."""
    import os
    if path is None:
        path = Path(__file__).resolve().parents[1] / ".env"
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip("'\"")
        if k and k not in os.environ:
            os.environ[k] = v


def fetch_text(url, timeout=25, tries=4, extra_headers=None):
    """GET com urllib + retry/backoff. Devolve (status, texto|None)."""
    last = None
    for attempt in range(tries):
        req = urllib.request.Request(url, headers={
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "application/json,text/*,*/*",
            "Accept-Language": "en-GB,en;q=0.9",
            **(extra_headers or {}),
        })
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except Exception as e:  # HTTPError traz .code
            last = getattr(e, "code", None) or str(e)
            time.sleep(min(2 ** attempt, 8) + random.random())
    return last, None


def already(path):
    p = Path(path)
    return p.exists() and p.stat().st_size > 0


def save_text(path, text):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(p)


def append_manifest(manifest_path, record):
    p = Path(manifest_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
