"""`--json` 必须**只**输出 JSON —— 否则 `python main.py ... --json | jq` 会挂。

这是坑 119 / 120 的回归闸门：漏挂 `--json` 或把说明文字一起写进 stdout，都会
让下游拿到「文字 + JSON」。这里对每个只读命令真的跑一遍并 `json.loads`。

⚠️ 自包含：不依赖 conftest 夹具。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "main.py"

READ_ONLY = [
    "summary",
    "judges",
    "bias",
    "probes",
    "agreement",
    "funnel",
    "noise",
    "records",
    "preferences",
    "dataset",
    "lineage",
    "config",
    "acceptance",
    "adversarial",
]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(MAIN), *args],
        cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )


@pytest.mark.parametrize("cmd", READ_ONLY)
def test_read_only_command_emits_pure_json(cmd: str) -> None:
    r = _run(cmd, "--json")
    # rc=1 表示某项质量闸门没过，不是 CLI 契约坏了；两者必须分开看。
    assert r.returncode in (0, 1), f"{cmd}: rc={r.returncode}\n{r.stderr[-2000:]}"
    json.loads(r.stdout)  # 有任何一个说明文字混进来都会在这里失败


def test_show_emits_pure_json() -> None:
    records = json.loads(_run("records", "--json").stdout)
    assert records, "records --json 没有返回任何记录"
    r = _run("show", str(records[0]["id"]), "--json")
    assert r.returncode == 0, r.stderr[-2000:]
    payload = json.loads(r.stdout)
    assert payload["id"] == records[0]["id"]
