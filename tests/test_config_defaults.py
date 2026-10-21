"""`configs/default.json` 与代码里的 ``DEFAULTS`` 必须说同一件事。

``load_config`` 用 ``deep_merge(DEFAULTS, default.json)``：文件里写错一个数，
代码就会用一个和文档不一样的口径跑报告。README 说「两边必须一致」，这里把它
变成一条会红的用例。

⚠️ 只检查文件里出现过的键（`families` / `judges` 只在代码里），不要求两边结构相同。
"""
from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from preferencefactory.config import DEFAULTS

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "default.json"

_MISSING = object()


def _leaves(node: Mapping[str, Any], prefix: str = "") -> Iterator[tuple[str, Any]]:
    for key, value in node.items():
        if key.startswith("_"):  # _comment / _families_note
            continue
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            yield from _leaves(value, path)
        else:
            yield path, value


def _lookup(cfg: Mapping[str, Any], path: str) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return _MISSING
        cur = cur[part]
    return cur


def test_default_json_matches_code_defaults() -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    problems: list[str] = []
    for path, expected in _leaves(raw):
        actual = _lookup(DEFAULTS, path)
        if actual is _MISSING:
            problems.append(f"{path}: DEFAULTS 里没有这个键")
        elif actual != expected:
            problems.append(f"{path}: 文件={expected!r} / DEFAULTS={actual!r}")
    assert not problems, "configs/default.json 与 DEFAULTS 不一致：\n  " + "\n  ".join(problems)
