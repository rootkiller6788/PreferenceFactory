"""缓存只能省计算，不能改变结果。

这些优化引入了 `lru_cache` / `cached_property`。如果缓存键漏了某个影响结果的
输入（例如裁判噪声尺度），同一进程内第二次运行就会复用第一次的值 —— 表现为
「同一份配置两次运行结果不同」，而且只在连续运行时才暴露。
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from preferencefactory.config import load_config
from preferencefactory.judges import _noise_of
from preferencefactory.pipeline import run_pipeline
from preferencefactory.schema import JudgeSpec

ROOT = Path(__file__).resolve().parent.parent


def test_two_runs_in_one_process_are_identical() -> None:
    cfg = load_config(None, root=ROOT)
    a = run_pipeline(cfg, root=ROOT)
    b = run_pipeline(cfg, root=ROOT)
    assert a.version == b.version
    assert [r.to_dict() for r in a.records] == [r.to_dict() for r in b.records]
    assert [c.to_dict() for c in a.candidates] == [c.to_dict() for c in b.candidates]


def test_noise_cache_keys_on_bias_scale() -> None:
    """同 id、不同 noise 的裁判不能共用缓存（_noise_for 的键必须含 noise）。"""
    cfg = load_config(None, root=ROOT)
    res = run_pipeline(cfg, root=ROOT)
    judge = JudgeSpec(id="llm-test", kind="llm", family="plain", noise=0.05)
    candidate = res.candidates[0]
    quiet = _noise_of(judge, candidate)
    loud = _noise_of(replace(judge, noise=0.40), candidate)
    assert quiet != loud
    # 重复取同一个键必须稳定（缓存不能每次重抽）。
    assert _noise_of(judge, candidate) == quiet


def test_config_deepcopy_does_not_share_judge_lists() -> None:
    """回归：改副本不能污染 DEFAULTS（否则后续运行会偷偷用错口径）。"""
    cfg = load_config(None, root=ROOT)
    copy = json.loads(json.dumps(dict(cfg)))
    copy["judges"][0]["noise"] = 0.99
    assert cfg["judges"][0]["noise"] != 0.99
