"""偏差探针集：把「裁判有偏」从一句断言变成**可测量的率**。

三个探针集，一一对应三条偏差轴。每个探针都是一道**陷阱**：正确答案与偏差指向的
答案相反，裁判踩中一次记一次。

1. **`swap`（位置偏差）** —— 同一对候选换个顺序判两次。无偏裁判两次会点名同一个
   候选（噪声是按 `(裁判, 候选)` 固定的，见 `judges._noise_of`）；只有位置加分才能
   让它改口。**这是最硬的一条**：它不需要真值，也不需要假设裁判诚实，它只需要
   裁判自洽。

2. **`length`（长度偏差）** —— 更长的那个**其实更差**。裁判点名更长的那个就是踩中。
   对照集是「更长的那个确实更好」，用来给出「这个裁判本来就有多准」的基线。

3. **`self`（自我偏好）** —— 裁判自家候选族产出的那个**其实更差**。同理配对照集。

**注入量与检出量不是同一个量纲**（注入的是打分偏移量，检出的是踩中率），所以校准
看的是**次序**而不是相等：注入越大的裁判，检出率必须越高。这条由 `acceptance` 逐轴核对。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .judges import JudgeContext, judge_pair
from .rng import Rng, derive_seed
from .schema import BIAS_AXES, BiasProfile, Candidate, JudgeSpec, ProbeResult, Prompt

__all__ = [
    "TrapUnit",
    "SwapUnit",
    "build_length_units",
    "build_self_units",
    "build_swap_units",
    "run_traps",
    "run_swaps",
    "run_probes",
]

_EPS = 1e-9


@dataclass(frozen=True)
class TrapUnit:
    """一道陷阱：展示顺序已定，`target` 是「偏差会指向、但其实是错的」那个。"""

    prompt: Prompt
    target: Candidate
    other: Candidate
    target_first: bool

    @property
    def a(self) -> Candidate:
        return self.target if self.target_first else self.other

    @property
    def b(self) -> Candidate:
        return self.other if self.target_first else self.target

    def chose_target(self, choice: str) -> bool:
        if choice == "tie":
            return False
        first = "a" if self.target_first else "b"
        return choice == first


@dataclass(frozen=True)
class SwapUnit:
    """一道换序探针：一对质量相当的候选，判两次（两种展示顺序）。"""

    prompt: Prompt
    x: Candidate
    y: Candidate


# --------------------------------------------------------------------------
# 陷阱集的构造
# --------------------------------------------------------------------------

def _take(xs: Sequence[Any], n: int, rng: Rng) -> list[Any]:
    """取至多 `n` 个。**不足时全取而不是报错** —— 探针集常常比可用样本还大，
    那种情况下「全用上」比「炸掉」有用，而且 `n` 会如实写进结果里。
    """
    if n >= len(xs):
        return list(xs)
    return rng.sample(list(xs), n)


def _in_gap_band(x: Candidate, y: Candidate, lo: float, hi: float) -> bool:
    """真值差是否落在 `[lo, hi]` 这一段里。

    **陷阱难度必须配平。** 这是 self 轴校准失败的真正原因：每个裁判的自家族不同，
    于是各自陷阱集里的「真值差」分布也不同 —— 一个裁判的陷阱是「差 0.30 明显更差」，
    另一个的是「差 0.05 险胜」。难度不一样，检出率就不可比，Spearman 只有 0.5。
    把真值差限制在同一段里之后，各裁判面对的是**同一难度**的诱惑。
    """
    return lo <= abs(x.quality - y.quality) <= hi


def build_length_units(
    pairs: Sequence[tuple[Prompt, Candidate, Candidate]],
    n: int,
    rng: Rng,
    *,
    gap_lo: float = 0.05,
    gap_hi: float = 0.30,
) -> tuple[list[TrapUnit], list[TrapUnit]]:
    """`(陷阱集, 对照集)`。陷阱 = 更长的其实更差；对照 = 更长的确实更好。

    展示顺序**逐条随机**：如果陷阱候选永远放在位置 0，长度偏差就测不出东西 ——
    位置偏差会把它吃掉（或者被它吃掉）。
    """
    trap_pool: list[tuple[Prompt, Candidate, Candidate]] = []
    ctrl_pool: list[tuple[Prompt, Candidate, Candidate]] = []
    for p, x, y in pairs:
        if x.length == y.length or not _in_gap_band(x, y, gap_lo, gap_hi):
            continue
        longer, shorter = (x, y) if x.length > y.length else (y, x)
        if longer.quality < shorter.quality:
            trap_pool.append((p, longer, shorter))
        else:
            ctrl_pool.append((p, longer, shorter))

    def wrap(pool: list[tuple[Prompt, Candidate, Candidate]]) -> list[TrapUnit]:
        return [
            TrapUnit(prompt=p, target=tgt, other=oth, target_first=rng.bernoulli(0.5))
            for p, tgt, oth in _take(pool, n, rng)
        ]

    return wrap(trap_pool), wrap(ctrl_pool)


def build_self_units(
    pairs: Sequence[tuple[Prompt, Candidate, Candidate]],
    family: str,
    n: int,
    rng: Rng,
    *,
    gap_lo: float = 0.08,
    gap_hi: float = 0.30,
    length_band: float = 8.0,
) -> tuple[list[TrapUnit], list[TrapUnit]]:
    """`(陷阱集, 对照集)`。陷阱 = 自家候选族产出的那个其实更差。

    额外要求两条候选**长度接近**（`length_band`）：否则长度偏差会在 self 陷阱里
    替自家族说话，self 轴的检出率就混进了 length 轴的成分。
    """
    trap_pool: list[tuple[Prompt, Candidate, Candidate]] = []
    ctrl_pool: list[tuple[Prompt, Candidate, Candidate]] = []
    for p, x, y in pairs:
        own = [c for c in (x, y) if c.family == family]
        if len(own) != 1:
            # 两个都出自自家族（或都不是）→ 这条上自我偏好无从定义，跳过。
            continue
        me = own[0]
        other = y if me is x else x
        if not _in_gap_band(me, other, gap_lo, gap_hi):
            continue
        if abs(me.length - other.length) > length_band:
            continue
        if me.quality < other.quality:
            trap_pool.append((p, me, other))
        else:
            ctrl_pool.append((p, me, other))

    def wrap(pool: list[tuple[Prompt, Candidate, Candidate]]) -> list[TrapUnit]:
        return [
            TrapUnit(prompt=p, target=tgt, other=oth, target_first=rng.bernoulli(0.5))
            for p, tgt, oth in _take(pool, n, rng)
        ]

    return wrap(trap_pool), wrap(ctrl_pool)


def build_swap_units(
    pairs: Sequence[tuple[Prompt, Candidate, Candidate]],
    n: int,
    tol: float,
    rng: Rng,
) -> list[SwapUnit]:
    """质量相当的候选对。

    必须用**质量相当**的对：如果两个候选客观差距本来就大，裁判两次都点名那个
    更好的，既可能是「没位置偏差」，也可能是「位置加分没大到能翻盘」。只有质量
    相当时「改口」才必然是位置带来的。
    """
    # 只看客观质量相当的对。长度**不设条件**：长度对某个候选是一个固定加分，
    # 换序不会改变它，所以它不会造成「改口」；反过来，如果在这里筛选长度，
    # 样本就会被人为地限制在「长度有差异」的对上，反而引入选择偏差。
    pool = [(p, x, y) for p, x, y in pairs if abs(x.quality - y.quality) <= tol + _EPS]
    return [SwapUnit(prompt=p, x=x, y=y) for p, x, y in _take(pool, n, rng)]


# --------------------------------------------------------------------------
# 跑探针
# --------------------------------------------------------------------------

def _run_one(judge: JudgeSpec, unit: TrapUnit, ctx: JudgeContext) -> bool:
    v = judge_pair(judge, unit.prompt, unit.a, unit.b, ctx=ctx)
    return unit.chose_target(v.choice)


def run_traps(
    judge: JudgeSpec,
    units: Sequence[TrapUnit],
    controls: Sequence[TrapUnit],
    ctx: JudgeContext,
    *,
    axis: str,
    definition: str,
    control_definition: str,
) -> tuple[ProbeResult, ProbeResult]:
    """返回 `(陷阱集结果, 对照集结果)`。

    **对照集的失误率会写进陷阱结果的 `baseline`**：一个噪声大的裁判在陷阱上踩中，
    可能只是它本来就会错，而不是它有偏差。用同难度的对照集把这份失误率减掉，
    「检出率」才是在说偏差。
    """
    hits = sum(1 for u in units if _run_one(judge, u, ctx))
    chits = sum(1 for u in controls if _run_one(judge, u, ctx))
    ctrl_rate = chits / len(controls) if controls else 0.0
    return (
        ProbeResult(
            axis=axis,
            judge_id=judge.id,
            n=len(units),
            hits=hits,
            definition=definition,
            baseline=(1.0 - ctrl_rate) if controls else 0.0,
            n_control=len(controls),
        ),
        ProbeResult(
            axis=axis, judge_id=judge.id, n=len(controls), hits=chits, definition=control_definition
        ),
    )


def run_swaps(judge: JudgeSpec, units: Sequence[SwapUnit], ctx: JudgeContext) -> ProbeResult:
    """换序探针：两次都给出确定结论、且点名了**不同**候选 → 改口，记一次。

    只统计「两次都确定」的对：平局不是改口，把它算进去会稀释检出率，
    也会让「噪声大所以爱平局」的裁判显得更没偏。
    """
    hits = 0
    n = 0
    for u in units:
        v1 = judge_pair(judge, u.prompt, u.x, u.y, ctx=ctx)   # x 在位置 0
        v2 = judge_pair(judge, u.prompt, u.y, u.x, ctx=ctx)   # y 在位置 0
        if v1.is_tie or v2.is_tie:
            continue
        n += 1
        picked1 = u.x if v1.choice == "a" else u.y
        picked2 = u.x if v2.choice == "b" else u.y
        if picked1.id != picked2.id:
            hits += 1
    return ProbeResult(
        axis="position",
        judge_id=judge.id,
        n=n,
        hits=hits,
        definition="质量相当的一对换序判两次，两次都确定且点名了不同候选 → 记一次改口",
    )


def run_probes(
    judges: Sequence[JudgeSpec],
    pairs: Sequence[tuple[Prompt, Candidate, Candidate]],
    ctx: JudgeContext,
    cfg: Mapping[str, Any],
    rng: Rng,
) -> tuple[list[ProbeResult], list[BiasProfile]]:
    """跑全部探针，产出 `(逐条结果, 每个裁判的偏差档案)`。

    探针在**主记录流之外**单独跑：它是测量仪器，不是数据。混在一起的话，
    「陷阱里的判定」也会进一致率统计，而那些判定是刻意构造的极端样本。
    """
    pcfg = dict(cfg.get("probes") or {})
    n_swap = int(pcfg.get("n_swap", 0))
    n_length = int(pcfg.get("n_length", 0))
    n_self = int(pcfg.get("n_self", 0))
    tol = float(pcfg.get("match_tolerance", 0.02))

    gap_lo = float(pcfg.get("trap_gap_lo", 0.05))
    gap_hi = float(pcfg.get("trap_gap_hi", 0.30))
    swap_units = build_swap_units(pairs, n_swap, tol, rng.fork("swap"))
    length_trap, length_ctrl = build_length_units(
        pairs, n_length, rng.fork("length"), gap_lo=gap_lo, gap_hi=gap_hi
    )

    results: list[ProbeResult] = []
    profiles: list[BiasProfile] = []

    for j in judges:
        axis_results: dict[str, ProbeResult] = {}

        pr = run_swaps(j, swap_units, ctx)
        axis_results["position"] = pr
        results.append(pr)

        lt, lc = run_traps(
            j,
            length_trap,
            length_ctrl,
            ctx,
            axis="length",
            definition="更长的那个其实更差，裁判点名更长的 → 记一次踩中",
            control_definition="对照：更长的那个确实更好，裁判点名更长的（这是判对的基线）",
        )
        axis_results["length"] = lt
        results.extend([lt, lc])

        if j.family is not None:
            st, sc = build_self_units(
                pairs,
                j.family,
                n_self,
                rng.fork("self", j.id),
                gap_lo=gap_lo,
                gap_hi=gap_hi,
                length_band=float(pcfg.get("self_length_band", 8.0)),
            )
            s_res, s_ctrl = run_traps(
                j,
                st,
                sc,
                ctx,
                axis="self",
                definition=f"自家候选族（{j.family}）产出的那个其实更差，裁判点名它 → 记一次踩中",
                control_definition="对照：自家候选族产出的那个确实更好，裁判点名它（判对的基线）",
            )
        else:
            s_res = ProbeResult(
                axis="self",
                judge_id=j.id,
                n=0,
                hits=0,
                definition="无候选族归属，自我偏好轴不适用（不参与检出，也不冤枉它）",
            )
            s_ctrl = s_res
        axis_results["self"] = s_res
        results.extend([s_res, s_ctrl])

        profiles.append(
            BiasProfile(
                judge_id=j.id,
                kind=j.kind,
                family=j.family,
                injected=tuple((a, float(j.injected[a])) for a in BIAS_AXES),
                # 换序探针本身就把噪声排除在外了（噪声按 (裁判, 候选) 固定），
                # 所以 position 轴用原始率；length / self 要扣掉自身失误率。
                detected=tuple(
                    (a, axis_results[a].rate if a == "position" else axis_results[a].corrected)
                    for a in BIAS_AXES
                ),
                n_probes=tuple((a, axis_results[a].n) for a in BIAS_AXES),
            )
        )

    return results, profiles
