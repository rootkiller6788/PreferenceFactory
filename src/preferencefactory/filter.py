"""过滤：把「哪条记录能进数据集」变成**可解释的逐条决定**。

四条规则按顺序执行，每条都留下 `Evidence`（**通过的也留** —— 「为什么保留」和
「为什么丢弃」一样值钱，只有留了通过项，事后才能解释「这条凭什么进了数据集」）。

漏斗记两种口径，**两个都要**，因为它们回答的是不同的问题：

- **首因归因**：每条被丢的记录只算在第一个卡住它的规则名下。用来回答
  「主要是被哪条卡住的」。
- **独立失败数**：每条规则各自拦下了多少（不管顺序）。用来回答
  「哪条规则最有杀伤力」—— 有些记录会被好几条同时卡住，首因归因会把它全给第一条。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .schema import BIAS_AXES, BiasProfile, Evidence, PreferenceRecord, Verdict

__all__ = [
    "FUNNEL_RULES",
    "Funnel",
    "NoiseReport",
    "tainted_axes",
    "evaluate_record",
    "filter_records",
    "build_funnel",
    "noise_report",
]

FUNNEL_RULES: tuple[str, ...] = (
    "agreement",
    "confidence",
    "rule_agreement",
    "untainted_support",
)


@dataclass(frozen=True)
class Funnel:
    total: int
    kept: int
    first_cause: tuple[tuple[str, int], ...]
    independent: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "kept": self.kept,
            "dropped": self.total - self.kept,
            "first_cause": dict(self.first_cause),
            "independent": dict(self.independent),
        }


@dataclass(frozen=True)
class NoiseReport:
    """「过滤掉了多少噪声样本」的完整账本。

    `precision` 与 `recall` 是这份报告最该被看的两个数：只报「过滤掉了 N 条」
    是没有意义的 —— 随机丢 N 条也能做到。
    """

    n_total: int
    noise_total: int
    n_kept: int
    noise_kept: int
    n_removed: int
    noise_removed: int

    @property
    def noise_rate_before(self) -> float:
        return self.noise_total / self.n_total if self.n_total else 0.0

    @property
    def noise_rate_after(self) -> float:
        return self.noise_kept / self.n_kept if self.n_kept else 0.0

    @property
    def precision(self) -> float:
        """丢掉的东西里有多少真的是噪声。"""
        return self.noise_removed / self.n_removed if self.n_removed else 0.0

    @property
    def recall(self) -> float:
        """噪声被抓到了多少。"""
        return self.noise_removed / self.noise_total if self.noise_total else 0.0

    @property
    def lift(self) -> float:
        """**比随机丢好多少倍** = 精度 / 基础噪声率。

        这是本报告最该被看的一个数。绝对精度是被基础噪声率钉死的：丢掉 X 条时
        精度上限就是 `noise_total / X`。所以「过滤掉了 N 条噪声」这句话本身没有
        —— 随机丢 N 条也能丢到噪声，命中率就是基础噪声率。提升倍数才回答
        「这个过滤器到底有没有在挑」。
        """
        base = self.noise_rate_before
        return self.precision / base if base > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_total": self.n_total,
            "noise_total": self.noise_total,
            "n_kept": self.n_kept,
            "noise_kept": self.noise_kept,
            "n_removed": self.n_removed,
            "noise_removed": self.noise_removed,
            "noise_rate_before": round(self.noise_rate_before, 6),
            "noise_rate_after": round(self.noise_rate_after, 6),
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "lift": round(self.lift, 6),
        }


def tainted_axes(
    judge_id: str,
    rec: PreferenceRecord,
    profiles: Mapping[str, BiasProfile],
    threshold: float,
) -> tuple[str, ...]:
    """这条记录上，这个裁判的哪几条偏差轴**真的会被触发**。

    「有偏差」不等于「在这条记录上被偏差影响」。一个长度偏差 0.4 的裁判，在两条
    一样长的候选上是不受影响的。按记录判断而不是按裁判判断，是为了不让过滤器
    一刀切地把有偏裁判全丢掉 —— 那样做会连它判对的部分一起赔进去。
    """
    if rec.verdict == "tie":
        return ()
    p = profiles.get(judge_id)
    if p is None:
        return ()
    winner = rec.a if rec.verdict == "a" else rec.b
    loser = rec.b if rec.verdict == "a" else rec.a
    out: list[str] = []
    if rec.verdict == "a" and p.detected_on("position") > threshold:
        out.append("position")
    if winner.length > loser.length and p.detected_on("length") > threshold:
        out.append("length")
    if p.family is not None and winner.family == p.family and p.detected_on("self") > threshold:
        out.append("self")
    return tuple(out)


def evaluate_record(
    rec: PreferenceRecord,
    cfg: Mapping[str, Any],
    profiles: Mapping[str, BiasProfile],
) -> tuple[bool, tuple[Evidence, ...], str | None]:
    """逐条评估。返回 `(保留?, 证据, 首因)`。"""
    f = dict(cfg.get("filter") or {})
    min_ag = float(f.get("min_agreement", 0.0))
    min_cf = float(f.get("min_confidence", 0.0))
    thr = float(f.get("bias_threshold", 1.0))

    ev: list[Evidence] = []
    first: str | None = None

    def note(rule: str, passed: bool, detail: str) -> None:
        nonlocal first
        ev.append(Evidence(rule=rule, passed=passed, detail=detail))
        if not passed and first is None:
            first = rule

    # 1) 一致率
    ok = rec.agreement >= min_ag
    note(
        "agreement",
        ok,
        f"一致率 {rec.agreement:.3f} {'>=' if ok else '<'} 门槛 {min_ag:.3f}"
        f"（{sum(1 for v in rec.verdicts if v.choice == rec.verdict)}/{len(rec.verdicts)} 个裁判同意）",
    )

    # 2) 置信度
    ok = rec.confidence >= min_cf
    note(
        "confidence",
        ok,
        f"置信度 {rec.confidence:.3f} {'>=' if ok else '<'} 门槛 {min_cf:.3f}"
        f"（一致率 {rec.agreement:.3f} × 果断度 {rec.provenance.get('decisiveness', 0.0):.3f}）",
    )

    # 3) rule 裁判（无偏锚点）不得反对
    if f.get("require_rule_agreement", True):
        rv = next((v for v in rec.verdicts if v.kind == "rule"), None)
        if rv is None:
            note("rule_agreement", False, "这条记录没有 rule 裁判的判定，无从对照")
        elif rv.is_tie:
            note("rule_agreement", True, "rule 裁判判平局（它太粗，两条都判不出差别）→ 不构成反对")
        elif rv.choice == rec.verdict:
            note("rule_agreement", True, f"rule 裁判也判 {rec.verdict.upper()}（无偏锚点同意）")
        else:
            note(
                "rule_agreement",
                False,
                f"rule 裁判判 {rv.choice.upper()}，聚合结论是 {rec.verdict.upper()} —— "
                "无偏锚点反对，说明这个结论多半是偏差投出来的",
            )

    # 4) 至少有一个「在这条记录上无染」的裁判支持结论
    if f.get("require_untainted_support", True):
        if rec.verdict == "tie":
            note("untainted_support", False, "结论是平局，不构成偏好（没有任何裁判支持一个具体选择）")
        else:
            clean: list[str] = []
            dirty: list[str] = []
            for v in rec.verdicts:
                if v.choice != rec.verdict:
                    continue
                bad = tainted_axes(v.judge_id, rec, profiles, thr)
                (dirty if bad else clean).append(f"{v.judge_id}({'+'.join(bad)})" if bad else v.judge_id)
            ok = bool(clean)
            note(
                "untainted_support",
                ok,
                (f"无染支持者 {clean}" if ok else f"支持者全部有染 {dirty}")
                + f"（有染门槛：检出偏差 > {thr:.2f}）",
            )

    kept = all(e.passed for e in ev)
    return kept, tuple(ev), first


def filter_records(
    records: Iterable[PreferenceRecord],
    cfg: Mapping[str, Any],
    profiles: Mapping[str, BiasProfile],
) -> list[PreferenceRecord]:
    recs = list(records)
    out: list[PreferenceRecord] = []
    for r in recs:
        kept, ev, _ = evaluate_record(r, cfg, profiles)
        out.append(r.with_kept(kept, ev))
    return out


def build_funnel(records: Sequence[PreferenceRecord]) -> Funnel:
    """漏斗。**要求证据已经写进记录**（`filter_records` 跑过），
    否则这里只能数出「丢了」而说不出「为什么」。
    """
    first_cause = {r: 0 for r in FUNNEL_RULES}
    independent = {r: 0 for r in FUNNEL_RULES}
    kept = 0
    for rec in records:
        if rec.kept:
            kept += 1
        fails = [e.rule for e in rec.evidence if not e.passed]
        for r in fails:
            independent[r] = independent.get(r, 0) + 1
        if fails:
            first_cause[fails[0]] = first_cause.get(fails[0], 0) + 1
    return Funnel(
        total=len(records),
        kept=kept,
        first_cause=tuple((r, first_cause.get(r, 0)) for r in FUNNEL_RULES),
        independent=tuple((r, independent.get(r, 0)) for r in FUNNEL_RULES),
    )


def noise_report(records: Sequence[PreferenceRecord]) -> NoiseReport:
    """噪声账本。真值来自 `Candidate.quality`（通过的判据占比），
    只在项目内部可得 —— 真实部署没有这一列，所以这里的数字是**对过滤器的评估**，
    不是过滤器本身的输入。
    """
    n_total = len(records)
    noise_total = sum(1 for r in records if r.is_noise)
    kept = [r for r in records if r.kept]
    dropped = [r for r in records if not r.kept]
    return NoiseReport(
        n_total=n_total,
        noise_total=noise_total,
        n_kept=len(kept),
        noise_kept=sum(1 for r in kept if r.is_noise),
        n_removed=len(dropped),
        noise_removed=sum(1 for r in dropped if r.is_noise),
    )
