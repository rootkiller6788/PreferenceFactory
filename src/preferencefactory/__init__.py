"""PreferenceFactory —— 偏好证据的产生、验证、去噪与版本化。

核心命题：**一条偏好数据不是 `chosen/rejected` 两个字符串，而是一份自带可信度与
来源的证据。** 所以 `agreement` / `confidence` / `provenance` 是一等字段，
而「多裁判 + 一致性 + 置信度过滤」是主链 —— **不是单个 LLM Judge**。
"""

from __future__ import annotations

from .config import DEFAULTS, deep_merge, load_config, objectives_of, protocol_digest, version_digest
from .dataset import content_digest, export_record, version_id
from .pipeline import OUT_FILES, PipelineResult, load_result, run_pipeline, write_outputs
from .rng import Rng, canonical_json, derive_seed, sha256_text
from .schema import (
    BIAS_AXES,
    CHECK_KINDS,
    JUDGE_KINDS,
    PROMPT_KINDS,
    VERDICTS,
    BiasProfile,
    Candidate,
    Check,
    Evidence,
    JudgeSpec,
    PreferenceRecord,
    ProbeResult,
    Prompt,
    SchemaError,
    Verdict,
)

__all__ = [
    "BIAS_AXES",
    "CHECK_KINDS",
    "DEFAULTS",
    "JUDGE_KINDS",
    "OUT_FILES",
    "PROMPT_KINDS",
    "VERDICTS",
    "BiasProfile",
    "Candidate",
    "Check",
    "Evidence",
    "JudgeSpec",
    "PipelineResult",
    "PreferenceRecord",
    "ProbeResult",
    "Prompt",
    "Rng",
    "SchemaError",
    "Verdict",
    "canonical_json",
    "content_digest",
    "deep_merge",
    "derive_seed",
    "export_record",
    "load_config",
    "load_result",
    "objectives_of",
    "protocol_digest",
    "run_pipeline",
    "sha256_text",
    "version_digest",
    "version_id",
    "write_outputs",
]
