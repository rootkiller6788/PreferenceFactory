"""偏好证据的对象模型。

**核心命题：一条偏好记录不是 `chosen/rejected` 两个字符串，而是一份自带可信度与
来源的证据。** 所以 `PreferenceRecord` 里 `agreement` / `confidence` / `rationale` /
`provenance` 与 `candidates` 一样是一等字段 —— 缺任何一个，这条数据就没法被辩护。

对象链（四层，与 `L2-modelgenomefactory` 的纪律一致）：

    Prompt + Family  →  Candidate（带隐藏真值 quality）
    Candidate × 2    →  Pair
    Pair × Judge     →  Verdict（带 rationale / provenance）
    Verdict × J      →  PreferenceRecord（带 agreement / confidence / evidence）

隐藏真值 `quality` 的口径：**通过的判据条数 / 判据总数**。它是客观的（每条判据都
能机械判定），也是分级的（于是「两个候选一样好」是可能的）。这个口径刻意保留了
平局的可能 —— 真值平局的记录**本来就没有合法偏好**，任何判定都是噪声，这正是
过滤器该拦下的东西。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

__all__ = [
    "PROMPT_KINDS",
    "CHECK_KINDS",
    "JUDGE_KINDS",
    "VERDICTS",
    "BIAS_AXES",
    "TRUTH_W_CRITERIA",
    "TRUTH_W_POLISH",
    "SchemaError",
    "Check",
    "Prompt",
    "Candidate",
    "JudgeSpec",
    "Verdict",
    "Evidence",
    "PreferenceRecord",
    "BiasProfile",
    "ProbeResult",
]

PROMPT_KINDS: tuple[str, ...] = ("arith", "upper", "reverse", "sort")

#: 四条**机械可判定**的判据。它们刻意都不涉及长度 —— 长度必须是与真值无关的噪声，
#: 否则「长度偏差」就永远和「长答案确实更好」纠缠不清。
CHECK_KINDS: tuple[str, ...] = ("exact", "clean", "punct", "tidy")

JUDGE_KINDS: tuple[str, ...] = ("rule", "human", "llm")

#: 判定结论。`tie` 是一等结论，不是「判不出来」的兜底 —— 两个候选一样好时，
#: 诚实的回答就是平局，硬选一个就是把裁判的噪声固化成数据。
VERDICTS: tuple[str, ...] = ("a", "b", "tie")

#: 三条偏差轴。**每一条都对应一个注入参数和一个探针集**，一一对应才好校准。
BIAS_AXES: tuple[str, ...] = ("position", "length", "self")


class SchemaError(ValueError):
    """对象构造期校验失败。"""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise SchemaError(msg)


@dataclass(frozen=True)
class Check:
    """一条机械可判定的判据。`difficulty` 越高，越难通过。"""

    kind: str
    difficulty: float

    def __post_init__(self) -> None:
        _require(self.kind in CHECK_KINDS, f"未知判据类型 {self.kind!r}（可选：{list(CHECK_KINDS)}）")
        _require(0.0 < self.difficulty < 1.0, f"判据难度应在 (0, 1)，得到 {self.difficulty}")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "difficulty": round(self.difficulty, 6)}


@dataclass(frozen=True)
class Prompt:
    """一个带**可验证判据**的 prompt。

    判据是 `rule` 裁判的唯一依据，也是整个项目唯一的无偏真值锚。没有它，
    「偏差」就只能靠投票数来定义，而投票数本身就被偏差污染了。
    """

    id: str
    kind: str
    question: str
    expected: str
    checks: tuple[Check, ...]

    def __post_init__(self) -> None:
        _require(self.kind in PROMPT_KINDS, f"未知 prompt 类型 {self.kind!r}（可选：{list(PROMPT_KINDS)}）")
        _require(bool(self.expected), f"prompt {self.id} 的 expected 不能为空")
        _require(len(self.checks) >= 1, f"prompt {self.id} 至少要有 1 条判据")
        kinds = [c.kind for c in self.checks]
        _require(len(set(kinds)) == len(kinds), f"prompt {self.id} 的判据有重复：{kinds}")

    @property
    def difficulty(self) -> float:
        return sum(c.difficulty for c in self.checks) / len(self.checks)

    @property
    def n_checks(self) -> int:
        return len(self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "question": self.question,
            "expected": self.expected,
            "checks": [c.to_dict() for c in self.checks],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Prompt":
        return cls(
            id=str(raw["id"]),
            kind=str(raw["kind"]),
            question=str(raw["question"]),
            expected=str(raw["expected"]),
            checks=tuple(Check(str(c["kind"]), float(c["difficulty"])) for c in raw["checks"]),
        )


#: 真值的构成权重。**刻意留出 20% 是任何机械判据都看不到的。**
#:
#: 如果没有这个残差，`rule` 裁判就等于真值本身，于是「rule 同意才保留」这条
#: 过滤规则会强得不像真的（实测会把精度顶到接近 1，而真实场景里验证器从来
#: 不是神谕）。留 20% 的盲区之后，rule 裁判的定位才对：**最可信的锚点，但不是答案。**
TRUTH_W_CRITERIA: float = 0.80
TRUTH_W_POLISH: float = 0.20


@dataclass(frozen=True)
class Candidate:
    """一个候选答案。

    三个「质量」要分清，混了就没法解释结果：

    - `criteria`：**可观测**的客观质量（`rule` 裁判重解析文本算出来的部分分均值）。
    - `quality`：**隐藏真值** = `0.8·criteria + 0.2·polish`。`criteria` 占大头，
      所以 rule 裁判与真值高度相关但不相等。
    - `polish`：无法被任何机械判据观测的残差（表达的得体程度）。它**不影响文本**，
      也不影响长度 —— 所以「长度与真值不相关」这条仍然成立。
    """

    id: str
    prompt_id: str
    family: str
    text: str
    latent: float
    checks_passed: tuple[str, ...]
    n_checks: int
    criteria: float
    polish: float
    verbosity: float
    length: int
    #: 与真值无关的长度贡献（客套话 / 落款 / 多余空格）。**单独记一列**，
    #: 是为了在报告里能证明「长度确实与真值无关」——这是长度偏差可被识别的前提。
    pad_chars: int

    def __post_init__(self) -> None:
        _require(0.0 <= self.latent <= 1.0, f"latent 应在 [0, 1]，得到 {self.latent}")
        _require(len(self.checks_passed) <= self.n_checks, "通过的判据数不能超过判据总数")
        _require(0.0 <= self.criteria <= 1.0, f"criteria 应在 [0, 1]，得到 {self.criteria}")
        _require(0.0 <= self.polish <= 1.0, f"polish 应在 [0, 1]，得到 {self.polish}")
        _require(self.length >= 0 and self.pad_chars >= 0, "长度不能为负")

    @property
    def quality(self) -> float:
        """隐藏真值 ∈ [0, 1]：`criteria` 为主，加 20% 不可观测的 `polish`。"""
        return TRUTH_W_CRITERIA * self.criteria + TRUTH_W_POLISH * self.polish

    @property
    def satisfies(self) -> bool:
        """核心答案是否完全正确（`exact` 判据）。"""
        return "exact" in self.checks_passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt_id": self.prompt_id,
            "family": self.family,
            "text": self.text,
            "latent": round(self.latent, 6),
            "checks_passed": list(self.checks_passed),
            "n_checks": self.n_checks,
            "criteria": round(self.criteria, 6),
            "polish": round(self.polish, 6),
            "quality": round(self.quality, 6),
            "satisfies": self.satisfies,
            "verbosity": round(self.verbosity, 6),
            "length": self.length,
            "pad_chars": self.pad_chars,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Candidate":
        return cls(
            id=str(raw["id"]),
            prompt_id=str(raw["prompt_id"]),
            family=str(raw["family"]),
            text=str(raw["text"]),
            latent=float(raw["latent"]),
            checks_passed=tuple(raw["checks_passed"]),
            n_checks=int(raw["n_checks"]),
            criteria=float(raw["criteria"]),
            polish=float(raw["polish"]),
            verbosity=float(raw["verbosity"]),
            length=int(raw["length"]),
            pad_chars=int(raw["pad_chars"]),
        )


@dataclass(frozen=True)
class JudgeSpec:
    """一个裁判。

    `beta` / `lam` / `self_bias` 是**我们亲手注入的偏差量**（不是估计值）。
    它们写进配置、也写进每条 `Verdict.provenance`，于是「谁判的、他有多偏」
    在同一个对象里就能回答。
    """

    id: str
    kind: str
    family: str | None = None
    beta: float = 0.0        # 位置偏差：偏向展示在第一个的候选
    lam: float = 0.0         # 长度偏差：偏向更长的候选
    self_bias: float = 0.0   # 自我偏好：偏向自家候选族产出的候选
    noise: float = 0.0

    def __post_init__(self) -> None:
        _require(self.kind in JUDGE_KINDS, f"未知裁判类型 {self.kind!r}（可选：{list(JUDGE_KINDS)}）")
        for name in ("beta", "lam", "self_bias"):
            v = getattr(self, name)
            _require(0.0 <= v <= 1.0, f"裁判 {self.id} 的 {name} 应在 [0, 1]，得到 {v}")
        _require(self.noise >= 0.0, f"裁判 {self.id} 的 noise 不能为负")
        if self.kind in ("rule", "human"):
            _require(
                self.beta == 0.0 and self.lam == 0.0 and self.self_bias == 0.0,
                f"裁判 {self.id} 是 {self.kind}，必须无偏（它是锚点/对照组）",
            )
            _require(self.family is None, f"裁判 {self.id} 是 {self.kind}，不该属于某个候选族")
        else:
            _require(self.family is not None, f"裁判 {self.id} 是 llm，必须指定 family（自我偏好要以它为定义）")

    @property
    def injected(self) -> dict[str, float]:
        return {"position": self.beta, "length": self.lam, "self": self.self_bias}

    @property
    def total_injected_bias(self) -> float:
        return self.beta + self.lam + self.self_bias

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "family": self.family,
            "beta": self.beta,
            "lam": self.lam,
            "self_bias": self.self_bias,
            "noise": self.noise,
            "injected": self.injected,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "JudgeSpec":
        return cls(
            id=str(raw["id"]),
            kind=str(raw["kind"]),
            family=(str(raw["family"]) if raw.get("family") is not None else None),
            beta=float(raw.get("beta", 0.0)),
            lam=float(raw.get("lam", 0.0)),
            self_bias=float(raw.get("self_bias", 0.0)),
            noise=float(raw.get("noise", 0.0)),
        )


@dataclass(frozen=True)
class Verdict:
    """一个裁判对一对候选的判定。

    `factors` 记下每个分量贡献了多少 —— 位置偏差 0.30 就是 0.30，不藏在结论后面。
    这样「他为什么这么判」是可回答的，而不是一句「模型觉得 A 更好」。
    """

    judge_id: str
    kind: str
    choice: str
    score_a: float
    score_b: float
    margin: float                       # 恒 >= 0：|score_a - score_b|
    factors: tuple[tuple[str, float], ...]
    rationale: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require(self.choice in VERDICTS, f"未知结论 {self.choice!r}（可选：{list(VERDICTS)}）")

    @property
    def is_tie(self) -> bool:
        return self.choice == "tie"

    def to_dict(self) -> dict[str, Any]:
        return {
            "judge_id": self.judge_id,
            "kind": self.kind,
            "choice": self.choice,
            "score_a": round(self.score_a, 6),
            "score_b": round(self.score_b, 6),
            "margin": round(self.margin, 6),
            "factors": {k: round(v, 6) for k, v in self.factors},
            "rationale": self.rationale,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Verdict":
        return cls(
            judge_id=str(raw["judge_id"]),
            kind=str(raw["kind"]),
            choice=str(raw["choice"]),
            score_a=float(raw["score_a"]),
            score_b=float(raw["score_b"]),
            margin=float(raw["margin"]),
            factors=tuple((str(k), float(v)) for k, v in raw["factors"].items()),
            rationale=str(raw["rationale"]),
            provenance=dict(raw.get("provenance") or {}),
        )


@dataclass(frozen=True)
class Evidence:
    """一条过滤证据。**通过的也要留** —— 「为什么保留」和「为什么丢弃」一样值钱，
    而且只有留了通过项，才能事后解释「这条凭什么进了数据集」。
    """

    rule: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"rule": self.rule, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True)
class PreferenceRecord:
    """一条偏好证据。字段与 README 建议的结构一一对应。"""

    id: str
    prompt_id: str
    prompt_kind: str
    question: str
    candidates: tuple[Candidate, ...]        # 恰好 2 个，按**展示顺序** [a, b]
    ranking: tuple[str, ...]                 # 该 prompt 上的 Copeland 排序（候选 id）
    judges: tuple[str, ...]
    scores: tuple[float, ...]                # 每个裁判的有符号分差（>0 表示 a 更好）
    verdicts: tuple[Verdict, ...]
    agreement: float
    kappa: float
    confidence: float
    verdict: str                             # 多数票结论
    weighted_verdict: str                    # 按检出偏差降权后的结论
    rationale: str
    provenance: Mapping[str, Any] = field(default_factory=dict)
    truth: Mapping[str, Any] = field(default_factory=dict)
    evidence: tuple[Evidence, ...] = ()
    kept: bool = True

    def __post_init__(self) -> None:
        _require(len(self.candidates) == 2, f"一条偏好记录必须恰好 2 个候选，得到 {len(self.candidates)}")
        _require(self.verdict in VERDICTS, f"未知聚合结论 {self.verdict!r}")
        _require(0.0 <= self.agreement <= 1.0, f"agreement 应在 [0, 1]，得到 {self.agreement}")
        _require(0.0 <= self.confidence <= 1.0, f"confidence 应在 [0, 1]，得到 {self.confidence}")

    # ---- 派生视图

    @property
    def a(self) -> Candidate:
        return self.candidates[0]

    @property
    def b(self) -> Candidate:
        return self.candidates[1]

    @property
    def truth_winner(self) -> str:
        """隐藏真值下的胜者。**优先读 `truth["winner"]`** —— 那里带平局容差。

        真值现在是连续的（`0.8·判据分 + 0.2·不可观测残差`），严格相等几乎不会发生，
        于是「两个候选一样好」必须由**容差**来定义（差不到 `truth_tie_eps` 就算平局）。
        容差写在配置里、由流水线算好存进 `truth`，这里只是读出来。
        """
        stored = self.truth.get("winner") if self.truth else None
        if stored in VERDICTS:
            return str(stored)
        qa, qb = self.a.quality, self.b.quality
        if qa > qb:
            return "a"
        if qb > qa:
            return "b"
        return "tie"

    @property
    def truth_margin(self) -> float:
        return abs(self.a.quality - self.b.quality)

    @property
    def is_noise(self) -> bool:
        """这条记录**本来就不该有偏好结论**。

        两种情形：真值平局（两个候选一样好，谁赢都是噪声），或结论与真值相反
        （判定错了）。过滤器的全部价值就是把这两类挑出去。
        """
        t = self.truth_winner
        return t == "tie" or self.verdict != t

    @property
    def chosen(self) -> str | None:
        """导出用：被选中的候选文本。`tie` 返回 None —— **不硬选**，
        硬选就是把裁判噪声固化成训练数据。"""
        if self.verdict == "tie":
            return None
        return (self.a if self.verdict == "a" else self.b).text

    @property
    def rejected(self) -> str | None:
        if self.verdict == "tie":
            return None
        return (self.b if self.verdict == "a" else self.a).text

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt_id": self.prompt_id,
            "prompt_kind": self.prompt_kind,
            "question": self.question,
            "candidates": [c.to_dict() for c in self.candidates],
            "ranking": list(self.ranking),
            "judges": list(self.judges),
            "scores": [round(s, 6) for s in self.scores],
            "verdicts": [v.to_dict() for v in self.verdicts],
            "agreement": round(self.agreement, 6),
            "kappa": round(self.kappa, 6),
            "confidence": round(self.confidence, 6),
            "verdict": self.verdict,
            "weighted_verdict": self.weighted_verdict,
            "rationale": self.rationale,
            "provenance": dict(self.provenance),
            "truth": dict(self.truth),
            "evidence": [e.to_dict() for e in self.evidence],
            "kept": self.kept,
        }

    def with_kept(self, kept: bool, evidence: tuple[Evidence, ...]) -> "PreferenceRecord":
        return replace(self, kept=kept, evidence=evidence)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PreferenceRecord":
        return cls(
            id=str(raw["id"]),
            prompt_id=str(raw["prompt_id"]),
            prompt_kind=str(raw["prompt_kind"]),
            question=str(raw["question"]),
            candidates=tuple(Candidate.from_dict(c) for c in raw["candidates"]),
            ranking=tuple(raw["ranking"]),
            judges=tuple(raw["judges"]),
            scores=tuple(float(s) for s in raw["scores"]),
            verdicts=tuple(Verdict.from_dict(v) for v in raw["verdicts"]),
            agreement=float(raw["agreement"]),
            kappa=float(raw["kappa"]),
            confidence=float(raw["confidence"]),
            verdict=str(raw["verdict"]),
            weighted_verdict=str(raw["weighted_verdict"]),
            rationale=str(raw["rationale"]),
            provenance=dict(raw.get("provenance") or {}),
            truth=dict(raw.get("truth") or {}),
            evidence=tuple(Evidence(**e) for e in raw.get("evidence") or ()),
            kept=bool(raw.get("kept", True)),
        )


@dataclass(frozen=True)
class BiasProfile:
    """一个裁判的三轴偏差档案：**注入量**与**检出量**并排放。

    两者并排是刻意的 —— 检测器的对错要能一眼看出来，而不是再写个脚本去对。
    """

    judge_id: str
    kind: str
    family: str | None
    injected: tuple[tuple[str, float], ...]
    detected: tuple[tuple[str, float], ...]
    n_probes: tuple[tuple[str, int], ...]

    @property
    def max_detected(self) -> float:
        return max((v for _, v in self.detected), default=0.0)

    @property
    def max_injected(self) -> float:
        return max((v for _, v in self.injected), default=0.0)

    def detected_on(self, axis: str) -> float:
        return dict(self.detected).get(axis, 0.0)

    def injected_on(self, axis: str) -> float:
        return dict(self.injected).get(axis, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "judge_id": self.judge_id,
            "kind": self.kind,
            "family": self.family,
            "injected": dict(self.injected),
            "detected": dict(self.detected),
            "n_probes": dict(self.n_probes),
            "max_detected": round(self.max_detected, 6),
            "max_injected": round(self.max_injected, 6),
        }


@dataclass(frozen=True)
class ProbeResult:
    """一个探针集的结果：`n` 次陷阱判定里，裁判踩中了几次。

    `baseline` 是**这个裁判自己的失误率**，在同难度对照集上测出来的。踩中率减去它
    才是「偏差贡献了多少」。

    没有这个校正会**冤枉好人**：噪声 0.11 的 `human` 在长度陷阱上照样会踩中
    （实测 0.125，一度超过了 0.12 的容差）—— 那不是偏差，那是它本来就有的失误率。
    用同难度的对照集把这一项减掉之后，`human` 掉到 0.025，而 λ=0.40 的裁判仍有 0.77。
    """

    axis: str
    judge_id: str
    n: int
    hits: int
    #: 陷阱的口径说明（写进报告，让「检出率」这个数字有定义）
    definition: str
    #: 该裁判在**同难度对照集**上的失误率（0 表示没有对照集，如换序探针）
    baseline: float = 0.0
    n_control: int = 0

    @property
    def rate(self) -> float:
        return self.hits / self.n if self.n else 0.0

    @property
    def corrected(self) -> float:
        """扣掉自身失误率之后的偏差量。**不可能为负**（负值说明对照集没配平好）。"""
        return max(0.0, self.rate - self.baseline)

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "judge_id": self.judge_id,
            "n": self.n,
            "hits": self.hits,
            "rate": round(self.rate, 6),
            "baseline": round(self.baseline, 6),
            "corrected": round(self.corrected, 6),
            "n_control": self.n_control,
            "definition": self.definition,
        }
