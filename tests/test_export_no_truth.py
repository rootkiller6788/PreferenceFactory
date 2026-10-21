"""导出形态不能泄漏隐藏真值。

下游拿到 `quality` / `truth_winner` 之后，会把评估用的真值当特征喂进训练 ——
指标会「好得不像话」，而且很难查。真值只允许留在 `records.jsonl`。
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from preferencefactory.config import load_config
from preferencefactory.dataset import export_record
from preferencefactory.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parent.parent

TRUTH_KEYS = {
    "latent",
    "quality",
    "satisfies",
    "checks_passed",
    "truth_winner",
    "truth_margin",
    "is_noise",
}


def _keys(node: Any) -> Iterator[str]:
    if isinstance(node, Mapping):
        for key, value in node.items():
            yield str(key)
            yield from _keys(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            yield from _keys(value)


def test_export_record_has_no_truth() -> None:
    res = run_pipeline(load_config(None, root=ROOT), root=ROOT)
    assert res.kept, "没有保留记录，这条用例失去意义"
    for rec in res.kept:
        leaked = TRUTH_KEYS & set(_keys(export_record(rec)))
        assert not leaked, f"{rec.id} 的导出形态泄漏了：{sorted(leaked)}"
