"""候选生成器：四个族，能力不同、长度倾向不同。

三个刻意的设计：

1. **`verbosity` 与 `skill` 不相关。** 长度必须是与真值无关的噪声，否则「长度偏差」
   永远和「长答案确实更好」纠缠不清。`Candidate.pad_chars` 把「与真值无关的填充」
   单独记一列，报告里要能证明它与 `quality` 不相关。
2. **错误答案是「像样但错」的。** 一眼假的错误会让所有裁判都判对，一致率全部趋近 1，
   过滤器的价值就测不出来了。算术错 ±1..6、排序只交换相邻两个、倒写只交换两个字符。
3. **族的槽位每轮换。** 如果第 0 个候选永远是 `terse` 族，位置偏差就会与
   「偏好某个族」混在一起，自我偏好和位置偏差就拆不开了。
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from .prompts import criteria_score, render
from .rng import Rng, derive_seed
from .schema import Candidate, Prompt, SchemaError

__all__ = [
    "FamilyError",
    "normalize_families",
    "generate",
    "generate_for_prompt",
    "length_stats",
    "norm_length",
    "quality_of",
]


class FamilyError(SchemaError):
    """候选族配置有问题。"""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise FamilyError(msg)


def normalize_families(raw: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for f in raw:
        fid = str(f.get("id", ""))
        _require(bool(fid), f"候选族缺 id：{dict(f)}")
        _require(fid not in seen, f"候选族 id 重复：{fid}（重复会让族统计算重）")
        seen.add(fid)
        skill = float(f.get("skill", 0.5))
        spread = float(f.get("spread", 0.1))
        verb = float(f.get("verbosity", 1.0))
        bonus = {str(k): float(v) for k, v in dict(f.get("kind_bonus") or {}).items()}
        _require(0.0 < skill < 1.0, f"族 {fid} 的 skill 应在 (0, 1)，得到 {skill}")
        _require(spread >= 0.0, f"族 {fid} 的 spread 不能为负，得到 {spread}")
        _require(0.1 <= verb <= 5.0, f"族 {fid} 的 verbosity 应在 [0.1, 5]，得到 {verb}")
        out.append({"id": fid, "skill": skill, "spread": spread, "verbosity": verb, "kind_bonus": bonus})
    _require(len(out) >= 2, f"至少要 2 个候选族，得到 {len(out)}")
    return out


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def generate(prompt: Prompt, family: Mapping[str, Any], slot: int, rng: Rng) -> Candidate:
    """为一个 prompt 生成一个候选。

    `latent` 是隐藏能力，它决定**能通过几条判据**：拿到 `latent` 后，把判据按难度
    升序排，能过前 `n_pass` 条（`latent >= difficulty`）。于是「客观质量」
    （通过的判据占比）是 `latent` 的单调阶梯函数 —— 真值是分级的，平局是可能的，
    而平局正是「这条记录本来就没有合法偏好」的信号。
    """
    fid = str(family["id"])
    # 能力 = 族的基础能力 + 这个题型的加成。**必须有加成**：如果某个族在所有题型上
    # 都最强，那么「它产出的候选更差」的样本就是 0，自我偏好这条轴根本测不出来
    # （第一版 `careful` 的 skill=0.90 就是这样，self 探针 n=0，检测器直接哑掉）。
    bonus = float(dict(family.get("kind_bonus") or {}).get(prompt.kind, 0.0))
    latent = _clamp01(rng.normal(float(family["skill"]) + bonus, float(family["spread"])))
    ordered = sorted(prompt.checks, key=lambda c: (c.difficulty, c.kind))
    passed = tuple(c.kind for c in ordered if latent >= c.difficulty)
    violated = frozenset(c.kind for c in ordered if latent < c.difficulty)
    # 离 `exact` 及格线有多远 → 决定错得多离谱（部分分要能还原这个距离）。
    err = max((c.difficulty - latent for c in ordered if c.kind == "exact"), default=0.5)
    verbosity = float(family["verbosity"]) * rng.uniform(0.85, 1.15)
    text, pad = render(prompt, violated, verbosity, rng, err)
    polish = _clamp01(rng.normal(0.5, 0.15))
    return Candidate(
        id=f"{prompt.id}#c{slot}",
        prompt_id=prompt.id,
        family=fid,
        text=text,
        latent=round(latent, 6),
        checks_passed=passed,
        n_checks=prompt.n_checks,
        criteria=round(criteria_score(prompt, text), 6),
        polish=round(polish, 6),
        verbosity=round(verbosity, 6),
        length=len(text),
        pad_chars=pad,
    )


def generate_for_prompt(prompt: Prompt, families: Sequence[Mapping[str, Any]], rng: Rng, n: int) -> list[Candidate]:
    """生成一个 prompt 的 `n` 个候选，每个来自不同的族。

    族的槽位由 `rng.shuffled` 决定 —— 这是刻意的：如果槽位固定，位置偏差就会与
    「偏好第 0 个族」混为一谈，自我偏好这条轴就没法单独测量。
    """
    if n > len(families):
        raise FamilyError(f"要 {n} 个候选但只有 {len(families)} 个族（一个 prompt 上同族出两份，自我偏好就无法定义）")
    chosen = rng.sample(list(families), n)
    return [generate(prompt, fam, slot, rng.fork(prompt.id, fam["id"], slot)) for slot, fam in enumerate(chosen)]


def quality_of(c: Candidate) -> float:
    return c.quality


def all_pairs(
    prompts: Sequence[Prompt],
    by_prompt: Mapping[str, Sequence[Candidate]],
) -> list[tuple[Prompt, Candidate, Candidate]]:
    """把每个 prompt 的候选两两配对（无序对）。

    **只在同一个 prompt 内部配对**：把两道题的答案放一起比较是没有意义的，
    而一旦跨题配对，长度差异就会被题型差异污染（排序题天然比算术题长）。
    """
    out: list[tuple[Prompt, Candidate, Candidate]] = []
    for p in prompts:
        cs = list(by_prompt.get(p.id) or ())
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                out.append((p, cs[i], cs[j]))
    return out


def length_stats(candidates: Iterable[Candidate]) -> tuple[float, float]:
    """全池长度的均值与标准差（供长度偏差的 z 分数使用）。

    **用全池而不是逐对归一化**：逐对归一化会让「两个都很长」的一对看起来
    和「两个都很短」的一对一样，而长度偏差恰恰在这两种情形下表现不同。
    """
    xs = [float(c.length) for c in candidates]
    if not xs:
        return 0.0, 1.0
    mu = sum(xs) / len(xs)
    var = sum((x - mu) ** 2 for x in xs) / len(xs)
    return mu, math.sqrt(var) or 1.0


def norm_length(c: Candidate, mu: float, sigma: float, *, clip: float = 2.0) -> float:
    """长度 z 分数，截断到 ±clip。

    截断是必要的：不截断时一个极端长的候选能拿到 z=5，长度偏差项会盖过一切，
    于是「长度偏差 0.28」这个注入量对应的实际效应就不可解释了。
    """
    z = (c.length - mu) / sigma if sigma else 0.0
    return max(-clip, min(clip, z))
