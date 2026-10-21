"""裁判池：三类裁判，偏差是**亲手注入的已知量**。

三类各有分工，缺一不可：

- **`rule`** 只读可验证判据，三个偏差恒为 0。它是**锚点**：偏差只能相对一个
  已知无偏的观察者来定义。它粗糙（两条都对的候选它只能判平局），但不会骗人。
- **`human`** 看得到客观质量，但噪声大。它是**对照组**：用来回答「检测器会不会
  冤枉好人」—— 如果它也被测出偏差，那说明检测器本身有问题，而不是裁判有偏差。
- **`llm`** 噪声反而更小，但带位置 / 长度 / 自我偏好三类偏差。这是真实 LLM 裁判
  的典型形态：**自信但有偏**。

打分模型对三者统一：

    score(x) = 客观质量 + β·[x 在位置 0] + λ·z(长度) + σ·[x 出自自家候选族] + 噪声

`rule` 的三项偏差与噪声全为 0，`human` 只有噪声，于是「同一个打分函数、不同参数」
就能覆盖三类 —— **策略是数据不是代码分支**，跟 `L3-finetunefactory` 是同一条纪律。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .candidates import norm_length
from .prompts import criteria_score, evaluate
from .rng import Rng, derive_seed
from .schema import BIAS_AXES, Candidate, JudgeSpec, Prompt, Verdict

__all__ = [
    "JudgeContext",
    "rule_quality",
    "score_candidate",
    "judge_pair",
    "choice_of",
    "normalize_judges",
]

_TIE = "tie"


@dataclass(frozen=True)
class JudgeContext:
    """打分需要的外生量。**全池统计量，不是逐对统计量**（见 `norm_length` 的说明）。"""

    mu_len: float
    sigma_len: float
    tie_margin: float


def normalize_judges(raw: Iterable[Mapping[str, Any]]) -> list[JudgeSpec]:
    out = [JudgeSpec.from_dict(j) for j in raw]
    ids = [j.id for j in out]
    if len(set(ids)) != len(ids):
        raise ValueError(f"裁判 id 重复：{ids}（重复会让「多数票」算重）")
    return out


def rule_quality(prompt: Prompt, c: Candidate) -> tuple[float, tuple[str, ...]]:
    """`rule` 裁判眼里的客观质量：**重新解析文本**算出来的，不是读 `c.criteria` 字段。

    这个区别很重要。`c.criteria` 是生成时写进去的标签，直接读它等于「裁判偷看答案」；
    重新解析才是裁判真的在读文本。它顺带构成一条硬不变量：
    **渲染出来的文本必须能被判据重新判回同一组通过的判据**（见 `acceptance` 的往返项）。

    注意 `rule` 看到的是 `criteria`（可观测的那部分），**不是 `quality`（真值）** ——
    真值里有 20% 是任何机械判据都看不到的。无偏 ≠ 全知。
    """
    res = evaluate(prompt, c.text)
    passed = tuple(k for k in ("exact", "clean", "punct", "tidy") if res.get(k))
    return criteria_score(prompt, c.text), passed


def score_candidate(
    judge: JudgeSpec,
    prompt: Prompt,
    c: Candidate,
    *,
    position: int,
    ctx: JudgeContext,
) -> tuple[float, tuple[tuple[str, float], ...]]:
    """给一个候选打分，并返回每个分量的贡献。

    分量拆开返回不是美化：**「他为什么这么判」必须可回答**。位置偏差 0.30 就是 0.30，
    不藏在「模型觉得 A 更好」后面。

    **噪声按 `(裁判, 候选)` 确定性派生，而不是每次判定重新抽。** 这不是省事：
    换序探针的前提就是「裁判对某个候选的看法不随展示顺序变」—— 否则把同一对
    换个顺序判两次，噪声本身就会让它改口，位置偏差就测不出来了。真实裁判也是
    这样：他对一条答案的印象是稳定的，变的只是位置带来的那点加分。
    """
    if judge.kind == "rule":
        q, _ = rule_quality(prompt, c)
        return q, (("criteria", q),)

    # human / llm 看到的是**真值**（它们能感知机械判据看不到的那 20%）。
    base = c.quality
    factors: list[tuple[str, float]] = [("quality", base)]
    n = _noise_of(judge, c)

    if judge.kind == "human":
        factors.append(("noise", n))
        return base + n, tuple(factors)

    # llm
    pos = judge.beta if position == 0 else 0.0
    ln = judge.lam * norm_length(c, ctx.mu_len, ctx.sigma_len)
    me = judge.self_bias if (judge.family is not None and c.family == judge.family) else 0.0
    factors += [("position", pos), ("length", ln), ("self", me), ("noise", n)]
    return base + pos + ln + me + n, tuple(factors)


def _noise_of(judge: JudgeSpec, c: Candidate) -> float:
    """`(裁判, 候选)` 上的确定性噪声。噪声为 0 时直接返回 0，不消耗随机源。"""
    if judge.noise <= 0.0:
        return 0.0
    return Rng(derive_seed(judge.id, c.id, "noise")).normal(0.0, judge.noise)


def choice_of(score_a: float, score_b: float, tie_margin: float) -> tuple[str, float]:
    """把两个分数变成结论。`tie` 是一等结论，不是兜底。"""
    diff = score_a - score_b
    if abs(diff) < tie_margin:
        return _TIE, abs(diff)
    return ("a" if diff > 0 else "b"), abs(diff)


def judge_pair(
    judge: JudgeSpec,
    prompt: Prompt,
    a: Candidate,
    b: Candidate,
    *,
    ctx: JudgeContext,
) -> Verdict:
    """一个裁判判一对候选。`a` 恒在位置 0 —— 位置偏差的坐标系由调用方通过
    **洗牌展示顺序**来打乱，而不是在这里随机。

    纯函数：同样的 `(裁判, prompt, 一对候选)` 永远得到同一个 `Verdict`。这条性质
    是换序探针能成立的前提。
    """
    sa, fa = score_candidate(judge, prompt, a, position=0, ctx=ctx)
    sb, fb = score_candidate(judge, prompt, b, position=1, ctx=ctx)
    choice, margin = choice_of(sa, sb, ctx.tie_margin)
    return Verdict(
        judge_id=judge.id,
        kind=judge.kind,
        choice=choice,
        score_a=sa,
        score_b=sb,
        margin=margin,
        factors=fa,
        rationale=_rationale(judge, choice, fa, fb, a, b),
        provenance=judge.to_dict(),
    )


def _rationale(
    judge: JudgeSpec,
    choice: str,
    fa: Sequence[tuple[str, float]],
    fb: Sequence[tuple[str, float]],
    a: Candidate,
    b: Candidate,
) -> str:
    da = dict(fa)
    db = dict(fb)
    if judge.kind == "rule":
        qa = da.get("criteria", 0.0)
        qb = db.get("criteria", 0.0)
        if choice == _TIE:
            return f"两条的可验证判据分几乎相同（A={qa:.3f} B={qb:.3f}）→ 平局"
        return f"可验证判据分 A={qa:.3f} B={qb:.3f}（只数判据，不看长度/位置）→ {choice.upper()}"
    if choice == _TIE:
        return f"{judge.id}：A={da.get('quality', 0):.3f} B={db.get('quality', 0):.3f}，分差小于平局阈值 → 平局"
    parts = [f"客观质量 A={da.get('quality', 0):.3f}/B={db.get('quality', 0):.3f}"]
    if judge.kind == "llm":
        parts.append(
            f"位置项 A={da.get('position', 0):.2f}/B={db.get('position', 0):.2f}"
            f" 长度项 A={da.get('length', 0):.2f}/B={db.get('length', 0):.2f}"
            f" 自家项 A={da.get('self', 0):.2f}/B={db.get('self', 0):.2f}"
        )
    else:
        parts.append(f"噪声 A={da.get('noise', 0):.2f}/B={db.get('noise', 0):.2f}")
    return f"{judge.id}：" + "；".join(parts) + f" → {choice.upper()}"
