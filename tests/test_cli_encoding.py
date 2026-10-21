"""回归：非 UTF-8 的 stdout 不能因为组合字符崩掉。

Windows 下管道/重定向的默认编码是 GBK，而验收详情里有 ``κ̄``（U+0304）。
``main.py`` 必须在写入前把非 tty 的 stdout 切到 UTF-8，否则输出直接丢失。

⚠️ 自包含：不依赖 conftest 夹具。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "main.py"


def _run_gbk(*args: str) -> subprocess.CompletedProcess[bytes]:
    env = dict(os.environ, PYTHONIOENCODING="gbk")
    return subprocess.run(
        [sys.executable, str(MAIN), *args],
        cwd=ROOT, env=env, capture_output=True,
    )


def test_acceptance_survives_gbk_redirect() -> None:
    r = _run_gbk("acceptance", "--json")
    assert r.returncode in (0, 1), r.stderr.decode("utf-8", "replace")[-2000:]
    payload = json.loads(r.stdout.decode("utf-8"))
    assert payload["total"] == payload["passed"], payload
