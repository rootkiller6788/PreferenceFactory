"""一致率 / kappa / 置信度 / 聚合结论。

**为什么是「多裁判 + 一致性」而不是「一个更好的裁判」**：单裁判的偏差会直接固化
进数据，而且无法被发现 —— 你没有任何参照物。多裁判的意义不在于「投票更准」，
而在于**裁判之间的分歧本身就是信号**：分歧大 → 这条记录本来就说不清 → 该丢。

四条性质（全部可证，报告里逐条核对）：

1. `agreement ∈ [0, 1]`，且对裁判顺序无关。
2. `kappa ≤ agreement` 恒成立（代数上必然，因为 κ=(P̄−Pe)/(1−Pe) 而 Pe(1−P̄)≥0）。
   κ 是**机会校正后**的一致率：大家都选 A 时一致率很高，但那可能只是因为 A 类本来就多。
3. `confidence` 对 `agreement` 与 `decisiveness` 都单调不减。
4. `tie` 是一等结论：投票打平时返回 `tie` 而不是硬选一个。
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

from .schema import VERDICTS, Verdict

__all__ = [
    "agreement_of",
    "category_distribution",
    "kappa_of",
    "decisiveness_of",
    "confidence_of",
    "aggregate",
    "weighted_aggregate",
    "bias_weight",
]

_CATS = tuple(VERDICTS)


def _counts(choices: Iterable[str]) -> dict[str, int]:
    out = {c: 0 for c in _CATS}
    for c in choices:
        if c not in out:
            raise ValueError(f"未知结论 {c!r}（可选：{list(_CATS)}）")
        out[c] += 1
    return out


def agreement_of(choices: Sequence[str]) -> float:
    """逐对一致率：任意两个裁判结论相同的比例。

    用组合数而不是双重循环 —— 双重循环在 5 个裁判时看不出差别，但一旦裁判池
    扩到几十个，它就是热点。
    """
    n = len(choices)
    if n < 2:
        return 1.0
    cnt = _counts(choices)
    agreeing = sum(v * (v - 1) // 2 for v in cnt.values())
    total = n * (n - 1) // 2
    return agreeing / total


def category_distribution(items: Iterable[Mapping[str, int]]) -> dict[str, float]:
    """全体记录上的类别分布 `p_j`（Fleiss' kappa 的机会项要用）。

    **必须在全量记录上算**，不能逐条算：κ 的分母是「随机裁判按总体分布乱猜也能
    撞上的概率」，而这个总体分布只有在看完所有记录之后才知道。
    """
    tot = {c: 0 for c in _CATS}
    n_items = 0
    for item in items:
        n_items += 1
        for c in _CATS:
            tot[c] += int(item.get(c, 0))
    n = sum(tot.values())
    if n == 0 or n_items == 0:
        return {c: 0.0 for c in _CATS}
    return {c: tot[c] / n for c in _CATS}


def kappa_of(counts: Mapping[str, int], p_j: Mapping[str, float], n_judges: int) -> tuple[float, float]:
    """单条记录的 `(观测一致率 P_i, Fleiss' κ)`。

    P_i 就是这一条上的逐对一致率；κ 减掉「按总体分布乱猜也能达到的一致率」。
    κ 可能为负 —— 那意味着**比随机还差**，是很强的噪声信号。
    """
    if n_judges < 2:
        return 1.0, 1.0
    sq = sum(int(counts.get(c, 0)) ** 2 for c in _CATS)
    p_i = (sq - n_judges) / (n_judges * (n_judges - 1))
    p_e = sum(float(p_j.get(c, 0.0)) ** 2 for c in _CATS)
    denom = 1.0 - p_e
    if denom <= 1e-12:
        # 总体分布退化到单一类别：κ 无定义。返回 0 而不是 1 —— 「没法衡量」不等于「很好」。
        return p_i, 0.0
    return p_i, (p_i - p_e) / denom


def decisiveness_of(verdicts: Sequence[Verdict]) -> float:
    """裁判们的平均果断度（分差的绝对值均值）。

    与 `agreement` 正交：一致率说「他们意见相同吗」，果断度说「他们各自有多确定」。
    两个都低 → 这条记录本来就说不清。
    """
    if not verdicts:
        return 0.0
    return sum(v.margin for v in verdicts) / len(verdicts)


def confidence_of(agreement: float, decisiveness: float, scale: float) -> float:
    """`confidence = agreement × min(1, decisiveness / scale)`。

    乘而不是加权和：任何一个为 0，置信度就必须是 0 —— 「全场意见一致但都很犹豫」
    和「都很确定但吵翻了」都不配进数据集。加法会让一项的高分掩盖另一项的 0。
    """
    if scale <= 0:
        raise ValueError(f"confidence 的 scale 必须 > 0，得到 {scale}")
    a = min(1.0, max(0.0, agreement))
    d = min(1.0, max(0.0, decisiveness / scale))
    return a * d


def aggregate(verdicts: Sequence[Verdict]) -> str:
    """多数票。**打平返回 `tie`** —— 硬选一个就是把裁判的噪声固化成训练数据。"""
    if not verdicts:
        return "tie"
    cnt = _counts(v.choice for v in verdicts)
    best = max(cnt.values())
    top = [c for c in _CATS if cnt[c] == best]
    return top[0] if len(top) == 1 else "tie"


def weighted_aggregate(verdicts: Sequence[Verdict], weights: Mapping[str, float]) -> str:
    """按「已检出偏差」降权后的投票。

    权重来自**探针实测**，不是来自注入量 —— 真实场景里没有注入量这一列。
    这一步的价值是可测量的：把 `weighted_verdict` 与 `verdict` 分别对着隐藏真值
    算准确率，就知道「知道谁有偏之后能不能判得更准」。
    """
    if not verdicts:
        return "tie"
    acc = {c: 0.0 for c in _CATS}
    for v in verdicts:
        acc[v.choice] += float(weights.get(v.judge_id, 1.0))
    best = max(acc.values())
    top = [c for c in _CATS if math.isclose(acc[c], best, rel_tol=1e-12, abs_tol=1e-12)]
    return top[0] if len(top) == 1 else "tie"


def bias_weight(profile_or_max: float) -> float:
    """把「检出的最大偏差」折成投票权重 ∈ [0, 1]。"""
    return max(0.0, min(1.0, 1.0 - max(0.0, min(1.0, profile_or_max))))
