"""配置：默认值 + 深合并 + 校验 + 两类摘要。

摘要分两种，**不能混**（`L2-moefactory` 的坑 70 就是混了）：

- `protocol_digest` 只跟**口径**有关（置信度尺度、平局阈值、过滤门槛）。改这些会
  改变报告里的数字，但不会改变「判了什么」。
- `version_digest` 跟**被判的对象**有关（prompt 池 + 候选生成器族 + 裁判池的
  注入偏差率）。它变了说明「输入变了」，报告必须整体重跑。

**注入偏差率写在这里而不是代码里**，是本项目最硬的一条设计：偏差是**已知量**，
所以偏差检测器可以拿它校准。检测器测出来的数如果与注入量对不上，错的可能是
检测器 —— 这个对照是验收标准②里「可量化裁判偏差」的底气。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .rng import canonical_json, join_digest, sha256_text

__all__ = [
    "DEFAULTS",
    "deep_merge",
    "load_config",
    "validate_config",
    "protocol_digest",
    "version_digest",
    "objectives_of",
    "expectations_of",
    "judges_of",
    "families_of",
    "resolve_path",
]

 #: 候选生成器族。`skill` 是隐藏能力均值，`verbosity` 是长度倾向。
#:
#: 两条刻意的设计：
#:
#: 1. **`verbosity` 与 `skill` 不相关** —— 长度必须是「与真值无关的噪声」，
#:    否则长度偏差就没法跟「长答案确实更好」区分开。
#: 2. **`kind_bonus` 让每个族各有擅长与短板。** 第一版没有这一项，`careful` 的
#:    skill=0.90 全面碾压，于是「它产出的候选更差」的样本数为 0 —— 自我偏好这条
#:    轴**根本测不出来**（self 探针 n=0，检测器直接哑掉）。加成交错之后，每个族
#:    都在某些题型上更强、在另一些上更弱，三条偏差轴才都有样本。
#: **三个「裁判归属族」的 verbosity 刻意接近（0.85 / 1.00 / 1.10）。**
#:
#: 原因：`verbose` 族的候选比别人长 ~10 个字符，于是当某个裁判归属 `verbose` 时，
#: 它的「自我偏好」和它的「长度偏差」在自家族陷阱里指向同一个候选 —— 两条轴彻底
#: 纠缠，self 轴的 Spearman 只有 0.50（注入 0.38 的裁判测出来比注入 0.22 的低，
#: 因为后者顺带吃到了长度偏差的加成）。把三个裁判归到长度倾向接近的族上，
#: `verbose` 就只负责提供**长度方差**，不再污染 self 轴。
FAMILIES: tuple[dict[str, Any], ...] = (
    {
        "id": "terse",
        "skill": 0.74,
        "spread": 0.11,
        "verbosity": 0.85,
        "kind_bonus": {"arith": 0.10, "upper": -0.02, "reverse": -0.09, "sort": 0.01},
    },
    {
        "id": "plain",
        "skill": 0.74,
        "spread": 0.11,
        "verbosity": 1.00,
        "kind_bonus": {"arith": -0.02, "upper": 0.09, "reverse": 0.00, "sort": -0.06},
    },
    {
        "id": "verbose",
        "skill": 0.73,
        "spread": 0.11,
        "verbosity": 1.95,
        "kind_bonus": {"arith": -0.07, "upper": 0.00, "reverse": 0.11, "sort": -0.02},
    },
    {
        "id": "careful",
        "skill": 0.75,
        "spread": 0.09,
        "verbosity": 1.10,
        "kind_bonus": {"arith": 0.00, "upper": -0.08, "reverse": -0.03, "sort": 0.13},
    },
)

#: 裁判池。**`beta` / `lam` / `self_bias` 是我们亲手注入的偏差量**，不是估计值。
#:
#: - `rule`：只读可验证判据，三个偏差全为 0 —— 它是无偏锚点。
#: - `human`：看得到真值，但有噪声，三个偏差也为 0 —— 它是「吵但诚实」的对照。
#: - `llm-*`：位置偏差 / 长度偏差 / 自我偏好各不相同，噪声反而更小
#:   （真实 LLM 裁判的典型形态：**自信但有偏**）。
JUDGES: tuple[dict[str, Any], ...] = (
    {"id": "rule", "kind": "rule", "family": None, "beta": 0.0, "lam": 0.0, "self_bias": 0.0, "noise": 0.0},
    {"id": "human", "kind": "human", "family": None, "beta": 0.0, "lam": 0.0, "self_bias": 0.0, "noise": 0.11},
    # 三个 llm 裁判归属的族是 `plain` / `careful` / `terse` —— 这三个族的长度倾向
    # 接近（1.00 / 1.10 / 0.85），于是 self 轴不被 length 轴污染。
    # `verbose`（1.95）刻意**不给任何裁判**，它只负责提供长度方差。
    {"id": "llm-alpha", "kind": "llm", "family": "plain", "beta": 0.30, "lam": 0.28, "self_bias": 0.22, "noise": 0.05},
    {"id": "llm-beta", "kind": "llm", "family": "careful", "beta": 0.16, "lam": 0.40, "self_bias": 0.08, "noise": 0.05},
    {"id": "llm-gamma", "kind": "llm", "family": "terse", "beta": 0.06, "lam": 0.10, "self_bias": 0.38, "noise": 0.05},
)

DEFAULTS: dict[str, Any] = {
    "seed": 20260930,
    "prompts": {
        "index": "data/prompts.jsonl",
        "n_prompts": 96,
    },
    "candidates": {
        "n_candidates": 3,          # 每个 prompt 出几个候选（3 → 3 个无序对）
        "families": [dict(f) for f in FAMILIES],
    },
    "judges": [dict(j) for j in JUDGES],
    "judging": {
        "tie_margin": 0.05,         # 打分差小于它就算「平局」
        "confidence_scale": 0.60,    # 果断度到这个量级就当满分
        "weight_by_bias": True,      # 是否用「已检出偏差」给裁判降权再投一次票
    },
    "probes": {
        "n_swap": 96,               # 换序探针：同一对错序判两次
        "n_length": 96,             # 长度陷阱：更长的那个其实更差
        "n_self": 96,               # 自我偏好陷阱：裁判自家族产出的那个其实更差
        #: 真值差小于它算「质量相当」（换序探针的取样口径）。
        #:
        #: 0.02 是**真值还是分级的时候**定的，那时有大量「完全一样好」的对。真值
        #: 改成连续的之后，|Δq| <= 0.02 只剩 16/288 对 —— 探针样本太少，检出率的
        #: 置信区间宽到没法用。放宽到 0.10 拿到 ~75 对；放宽不会引入偏差
        #: （真值差越大越不容易「改口」，只会稀释检出率），所以次序仍然可比。
        "match_tolerance": 0.10,
        #: 陷阱的**真值差必须落在同一段**里。不设这一对值的时候，self 轴的
        #: Spearman 只有 0.50 —— 因为每个裁判的自家族不同，各自陷阱的「真值差」
        #: 分布也不同，难度不可比。
        "trap_gap_lo": 0.08,
        "trap_gap_hi": 0.30,
        #: 自家族陷阱额外要求两条候选**长度接近**。不设时，长度偏差会在 self 陷阱里
        #: 帮自家族说话；设太小（<= 6）时 `verbose` 族可用对数掉到 6，太少。
        "self_length_band": 8,
    },
    "filter": {
        #: 这两个门槛是**扫过一轮**的（288 条上跑 3×3×2 网格）：
        #: 0.60/0.30 只留 16%，而且精度被基础噪声率钉死在 0.43 附近；
        #: 0.35/0.20 留 58%，噪声率 0.295 → 0.030，精度 0.656、提升 2.22 倍。
        #: 0.35 的含义是「至少 3/5 干净多数（3-2 → 0.40）」。
        "min_agreement": 0.35,
        "min_confidence": 0.20,
        "bias_threshold": 0.10,     # 检出偏差超过它，该裁判在这条记录上算「有染」
        "require_rule_agreement": True,
        "require_untainted_support": True,
    },
    "acceptance": {
        #: 真值差小于它就视为「本来就没有偏好」。真值连续之后严格相等几乎不会发生，
        #: 但没有容差的话，「差 0.001」会被当成有偏好，过滤器的精度会被高估。
        "truth_tie_eps": 0.02,
        "min_records": 200,
        "min_noise_removed": 40,
        "require_agreement_in_unit": True,
        "require_kappa_le_agreement": True,
        "require_unbiased_judges_clean": True,
        "unbiased_bias_tolerance": 0.12,
        "require_noise_rate_drops": True,
        #: 每条偏差轴的探针样本必须 > 0，否则那条轴的偏差**无从测量**。
        "require_all_axes_sampled": True,
        #: 校准看**秩相关**：注入的是打分偏移量、检出的是踩中率，量纲不同，
        #: 能要求的只是一场次序一致。
        "min_rank_correlation": 0.60,
        #: 长度与真值的相关系数上限。超过它，「长度偏差」和「长答案确实更好」
        #: 就拆不开了。
        "length_quality_corr_bound": 0.30,
        #: **用提升倍数而不是绝对精度**：精度被基础噪声率钉死了 ——
        #: 丢掉 X 条时精度上限就是 noise_total / X。基础噪声率 29.5% 时，
        #: 丢掉 42% 的记录，精度再高也到不了 0.71。提升倍数才是「比随机丢好多少」。
        "min_filter_lift": 1.5,
    },
    "expectations": {
        # 设计期望。**允许被实测推翻** —— 被推翻的那条是附录 B 里最值钱的一行。
        "rule_judge_agrees_with_truth": True,
        "weighted_verdict_beats_majority": True,
        "confidence_correlates_with_true_margin": True,
        "filter_precision_above": 0.60,
        #: 「四个题型的保留率应当差不多」—— 过滤器不该系统性地偏袒某个题型。
        #: 实测差 18 个百分点（算术 51.4% vs 排序 69.4%），**被推翻**。
        "retention_spread_under": 0.10,
        #: 「裁判判平局应当主要发生在真值也平局的记录上」。
        #: 实测裁判平局 15.6% vs 真值平局 4.5%，**被推翻** ——
        #: 裁判的 `tie` 大多是「它判不出来」，不是「两个候选真的一样好」。
        "tie_verdicts_track_true_ties": True,
        #: 「按检出偏差降权之后，每个题型都该判得更准」。
        "weighted_verdict_beats_majority_in_every_kind": True,
    },
    "report": {
        "headline_records": 6,
        "max_matrix_rows": 24,
    },
}


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = deep_merge(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def resolve_path(root: Path | str, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else Path(root) / p


def load_config(path: Path | str | None = None, *, root: Path | str) -> dict[str, Any]:
    """读配置。附加 `_config_path` / `_root`，并在校验失败时**立刻报错**。"""
    root = Path(root)
    cfg = dict(DEFAULTS)
    if path is not None:
        p = Path(path)
        raw = json.loads(p.read_text(encoding="utf-8"))
        cfg = deep_merge(cfg, raw)
    else:
        default = root / "configs" / "default.json"
        if default.exists():
            cfg = deep_merge(cfg, json.loads(default.read_text(encoding="utf-8")))
        p = default
    cfg["_config_path"] = str(p)
    cfg["_root"] = str(root)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Mapping[str, Any]) -> None:
    """配置校验。**宁可在这里炸，也不要跑到一半用一个荒唐的口径出报告。**"""
    errs: list[str] = []

    n_prompts = (cfg.get("prompts") or {}).get("n_prompts")
    if not isinstance(n_prompts, int) or n_prompts < 4:
        errs.append(f"prompts.n_prompts 必须是 >= 4 的整数，得到 {n_prompts!r}")

    n_cand = (cfg.get("candidates") or {}).get("n_candidates")
    if not isinstance(n_cand, int) or not (2 <= n_cand <= 4):
        errs.append(f"candidates.n_candidates 必须在 [2, 4]，得到 {n_cand!r}")

    judges = cfg.get("judges") or []
    if not isinstance(judges, list) or len(judges) < 3:
        errs.append(f"judges 至少 3 个（rule / human / 至少一个 llm），得到 {len(judges) if isinstance(judges, list) else '非列表'}")
        judges = []
    kinds = [j.get("kind") for j in judges if isinstance(j, Mapping)]
    if "rule" not in kinds:
        errs.append("judges 必须含一个 kind=rule（无偏锚点，没有它偏差无从对照）")
    if "human" not in kinds:
        errs.append("judges 必须含一个 kind=human（吵但诚实的对照组）")
    fam_ids = {f["id"] for f in (cfg.get("candidates") or {}).get("families", [])}
    for j in judges:
        if not isinstance(j, Mapping):
            continue
        jid = j.get("id")
        if j.get("kind") == "rule" and (j.get("beta") or j.get("lam") or j.get("self_bias")):
            errs.append(f"裁判 {jid} 是 rule，不允许带任何偏差（它是无偏锚点）")
        if j.get("kind") == "human" and (j.get("beta") or j.get("lam") or j.get("self_bias")):
            errs.append(f"裁判 {jid} 是 human，不允许带偏差（它是无偏对照组）")
        if j.get("kind") == "llm":
            if j.get("family") not in fam_ids:
                errs.append(f"裁判 {jid} 的 family={j.get('family')!r} 不在候选族 {sorted(fam_ids)} 里（自我偏好无从定义）")
            for key in ("beta", "lam", "self_bias"):
                v = j.get(key)
                if not isinstance(v, (int, float)) or not (0.0 <= v <= 1.0):
                    errs.append(f"裁判 {jid} 的 {key} 应在 [0, 1]，得到 {v!r}")
        if not isinstance(j.get("noise"), (int, float)) or j.get("noise", -1) < 0:
            errs.append(f"裁判 {jid} 的 noise 必须是 >= 0 的数，得到 {j.get('noise')!r}")

    jd = cfg.get("judging") or {}
    tm = jd.get("tie_margin")
    if not isinstance(tm, (int, float)) or not (0.0 <= tm < 1.0):
        errs.append(f"judging.tie_margin 应在 [0, 1)，得到 {tm!r}")
    cs = jd.get("confidence_scale")
    if not isinstance(cs, (int, float)) or cs <= 0:
        errs.append(f"judging.confidence_scale 必须 > 0，得到 {cs!r}")

    fl = cfg.get("filter") or {}
    for key in ("min_agreement", "min_confidence"):
        v = fl.get(key)
        if not isinstance(v, (int, float)) or not (0.0 <= v <= 1.0):
            errs.append(f"filter.{key} 应在 [0, 1]，得到 {v!r}")
    bt = fl.get("bias_threshold")
    if not isinstance(bt, (int, float)) or not (0.0 <= bt <= 1.0):
        errs.append(f"filter.bias_threshold 应在 [0, 1]，得到 {bt!r}")

    pr = cfg.get("probes") or {}
    for key in ("n_swap", "n_length", "n_self"):
        v = pr.get(key)
        if not isinstance(v, int) or v < 0:
            errs.append(f"probes.{key} 必须是非负整数，得到 {v!r}")

    if errs:
        raise ValueError("配置校验失败：\n  - " + "\n  - ".join(errs))


def protocol_digest(cfg: Mapping[str, Any]) -> str:
    """口径摘要：只吃影响**数字口径**的配置，不吃裁判池（那是输入不是口径）。"""
    keep = {
        "judging": cfg.get("judging"),
        "filter": cfg.get("filter"),
        "probes": cfg.get("probes"),
        "acceptance": cfg.get("acceptance"),
        "report": cfg.get("report"),
    }
    return sha256_text(canonical_json(keep))


def version_digest(cfg: Mapping[str, Any], inputs: Iterable[tuple[str, Any]]) -> str:
    """版本摘要：吃**被判对象**的摘要（prompt 池 + 候选族 + 裁判池 + 种子）。"""
    parts = [
        sha256_text(canonical_json(cfg.get("seed"))),
        sha256_text(canonical_json(cfg.get("candidates") or {})),
        sha256_text(canonical_json(cfg.get("judges") or [])),
        sha256_text(canonical_json(cfg.get("prompts") or {})),
    ]
    for name, payload in inputs:
        parts.append(sha256_text(f"{name}:{canonical_json(payload)}"))
    return join_digest(parts)


def objectives_of(cfg: Mapping[str, Any]) -> tuple[()]:
    """本项目**没有**优化目标。

    一旦有了目标函数，「哪条偏好更好」就会混进一份只该回答「这条证据有多可信」
    的报告里。这里只做**过滤**，不做**挑选** —— 过滤的门槛是置信度，不是收益。
    """
    return ()


def expectations_of(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return dict(cfg.get("expectations") or {})


def judges_of(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(j) for j in (cfg.get("judges") or [])]


def families_of(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(f) for f in (cfg.get("candidates") or {}).get("families", [])]
