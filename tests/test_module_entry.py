"""`python -m preferencefactory` 必须与 `main.py` 走同一条 CLI 契约。

⚠️ 自包含：不依赖 conftest 夹具。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_module_entry_emits_json() -> None:
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    r = subprocess.run(
        [sys.executable, "-m", "preferencefactory", "config", "--json"],
        cwd=ROOT, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    assert r.returncode == 0, r.stderr[-2000:]
    payload = json.loads(r.stdout)
    assert "seed" in payload and "filter" in payload
