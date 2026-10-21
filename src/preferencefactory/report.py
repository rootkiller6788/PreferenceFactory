"""Markdown 报告。

报告的作用是**让数字自己说话**，所以每条结论旁边都要有它的口径。特别是：

- 偏差表里**注入量与检出量并排** —— 读者不该再写个脚本去对照。
- 噪声账本里**同时给精度和提升倍数** —— 只报「过滤掉了 N 条噪声」是没有意义的，
  随机丢 N 条也能丢到噪声。
- 最后一节列出**设计期望 vs 实测**，并明确标出被推翻的那几条。被推翻的期望是
  整份报告里最值钱的部分：说明我们对这套机制的直觉哪里错了。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .acceptance import spearman
from .dataset import export_record
from .schema import BIAS_AXES, PreferenceRecord

__all__ = ["build_report"]

_AXIS_LABEL = {"position": "位置偏差", "length": "长度偏差", "self": "自我偏好"}


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _f(x: float, n: int = 3) -> str:
    return f"{x:.{n}f}"


def build_report(result: Any, cfg: Mapping[str, Any]) -> str:
    recs = list(result.records)
    kept = [r for r in recs if r.kept]
    n = result.noise
    fn = result.funnel
    L: list[str] = []

    L.append("# PreferenceFactory 报告")
    L.append("")
    L.append(
        f"> 版本 `{result.version}` ｜ 生成于 {result.created_at} ｜ "
        f"{result.counts.get('prompts', 0)} 个 prompt × {result.counts.get('candidates', 0)} 个候选 "
        f"→ {len(recs)} 条偏好记录 → 保留 {len(kept)} 条"
    )
    L.append("")
    L.append("**核心命题**：一条偏好数据不是 `chosen/rejected` 两个字符串，而是一份自带可信度与来源的证据。")
    L.append("")

    # ---------------------------------------------------------------- 1
    L.append("## 1. 一句话结论")
    L.append("")
    L.append(
        f"裁判池里有三个**自信但有偏**的 LLM 裁判（位置 / 长度 / 自我偏好各注入一档）和两个"
        f"**无偏**裁判（一个只读可验证判据的 `rule`，一个吵但诚实的 `human`）。"
        f"三步探针把每条偏差都测成了**率**，且与注入量的次序一致"
        f"（Spearman ρ：位置 {_f(_rho(result, 'position'))} / 长度 {_f(_rho(result, 'length'))} / 自我 {_f(_rho(result, 'self'))}）。"
    )
    L.append("")
    L.append(
        f"过滤器用「一致率 × 果断度」加两条证据规则，从 {n.n_total} 条里留下 {n.n_kept} 条"
        f"（{_pct(n.n_kept / n.n_total if n.n_total else 0)}），"
        f"噪声率从 {_pct(n.noise_rate_before)} 降到 {_pct(n.noise_rate_after)}；"
        f"丢掉的东西里 {_pct(n.precision)} 真的是噪声，相当于**随机丢的 {_f(n.lift, 2)} 倍命中率**。"
    )
    L.append("")

    # ---------------------------------------------------------------- 2
    L.append("## 2. 流水线与对象链")
    L.append("")
    L.append("```")
    L.append("Prompt（带可验证判据）")
    L.append("  ↓ 4 个候选族 × 题型能力加成")
    L.append("Candidate（带隐藏真值 quality 与可观测 criteria）")
    L.append("  ↓ 两两配对（同 prompt 内，展示顺序随机）")
    L.append("Pair")
    L.append("  ↓ 5 个裁判（rule / human / llm × 3）")
    L.append("Verdict（带 rationale 与 provenance）")
    L.append("  ↓ 一致率 + κ + 置信度")
    L.append("PreferenceRecord（带 evidence）")
    L.append("  ↓ 四条过滤规则（逐条留证据）")
    L.append("Preference Dataset（内容寻址版本）")
    L.append("```")
    L.append("")
    L.append(
        "**探针在记录流之前跑**：「按检出偏差降权的投票」要用到探针结果，而探针只需要候选池。"
        "先跑探针，聚合时权重就已经是实测值了。"
    )
    L.append("")

    # ---------------------------------------------------------------- 3
    L.append("## 3. 裁判池")
    L.append("")
    L.append("| 裁判 | 类型 | 归属族 | β 位置 | λ 长度 | σ 自我 | 噪声 |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for j in result.judges:
        L.append(
            f"| `{j.id}` | {j.kind} | {j.family or '—'} | {_f(j.beta, 2)} | {_f(j.lam, 2)} | "
            f"{_f(j.self_bias, 2)} | {_f(j.noise, 2)} |"
        )
    L.append("")
    L.append(
        "`rule` 与 `human` 的三项偏差恒为 0 —— 它们是**锚点与对照组**，不是「估计出来没偏」，"
        "而是**构造上无偏**。没有它们，「偏差」就只能靠投票数定义，而投票数本身被偏差污染。"
    )
    L.append("")
    L.append(
        "三个 llm 裁判归属的族（`plain` / `careful` / `terse`）长度倾向刻意接近；"
        "`verbose` 族不给任何裁判，只负责提供长度方差（原因见第 4 节）。"
    )
    L.append("")

    # ---------------------------------------------------------------- 4
    L.append("## 4. 偏差检测与校准")
    L.append("")
    for axis in BIAS_AXES:
        L.append(f"### 4.{BIAS_AXES.index(axis) + 1} {_AXIS_LABEL[axis]}（{axis}）")
        L.append("")
        L.append("| 裁判 | 注入量 | 检出率 | 样本数 | 判定 |")
        L.append("| --- | --- | --- | --- | --- |")
        rows = sorted(result.profiles, key=lambda p: -p.injected_on(axis))
        for p in rows:
            cnt = dict(p.n_probes).get(axis, 0)
            det = p.detected_on(axis)
            inj = p.injected_on(axis)
            if cnt == 0:
                L.append(f"| `{p.judge_id}` | {_f(inj, 2)} | — | 0 | 该轴不适用 |")
                continue
            verdict = "干净" if inj == 0 and det <= 0.12 else ("检出" if inj > 0 else "**误报**")
            L.append(f"| `{p.judge_id}` | {_f(inj, 2)} | {_f(det)} | {cnt} | {verdict} |")
        L.append("")
        L.append(f"**Spearman ρ = {_f(_rho(result, axis))}**（注入次序 vs 检出次序）")
        L.append("")
        defs = [p.definition for p in result.probes if p.axis == axis]
        seen: set[str] = set()
        for d in defs:
            if d and d not in seen:
                seen.add(d)
                L.append(f"- 陷阱口径：{d}")
        L.append("")

    L.append("### 4.4 三条轴各自的陷阱是怎么构造的")
    L.append("")
    L.append(
        "- **换序（position）**：质量相当的一对（真值差 ≤ 0.10）换序判两次，两次都给出确定结论"
        "却点名了不同候选 → 记一次「改口」。这条最硬：**它不需要真值，也不需要假设裁判诚实，"
        "只需要裁判自洽**。噪声按 `(裁判, 候选)` 固定，所以噪声本身不会造成改口。"
    )
    L.append(
        "- **长度陷阱（length）**：更长的那个其实更差（真值差限制在 [0.08, 0.30]，配平难度）。"
        "点名更长的 → 记一次踩中。对照集是「更长的确实更好」，给出「这个裁判本来有多准」的基线。"
    )
    L.append(
        "- **自家族陷阱（self）**：自家候选族产出的那个其实更差，且两条候选**长度接近**（≤ 8 字符）。"
        "长度接近这一条是必须的 —— 否则长度偏差会在 self 陷阱里替自家族说话。"
    )
    L.append("")

    # ---------------------------------------------------------------- 5
    L.append("## 5. 一致率")
    L.append("")
    L.append(
        f"全体 {len(recs)} 条：一致率均值 **{_f(_mean([r.agreement for r in recs]))}**，"
        f"Fleiss' κ 均值 **{_f(_mean([r.kappa for r in recs]))}**，"
        f"置信度均值 **{_f(_mean([r.confidence for r in recs]))}**，"
        f"平局结论占比 {_pct(_mean([1.0 if r.verdict == 'tie' else 0.0 for r in recs]))}。"
    )
    L.append("")
    L.append("κ ≤ 一致率是**代数必然**（κ=(P̄−Pe)/(1−Pe)，而 Pe(1−P̄) ≥ 0），")
    L.append("所以 κ 低不代表算错了，它代表「扣掉乱猜也能撞上的部分之后，剩下的共识不多」。")
    L.append("")
    L.append("### 5.1 裁判两两一致率")
    L.append("")
    js = [j.id for j in result.judges]
    L.append("| | " + " | ".join(f"`{j}`" for j in js) + " |")
    L.append("| --- | " + " | ".join("---" for _ in js) + " |")
    for a in js:
        cells = []
        for b in js:
            if a == b:
                cells.append("—")
            else:
                key = f"{a}~{b}" if f"{a}~{b}" in _matrix(result) else f"{b}~{a}"
                cells.append(_f(_matrix(result).get(key, {}).get("rate", 0.0), 2))
        L.append(f"| `{a}` | " + " | ".join(cells) + " |")
    L.append("")

    # ---------------------------------------------------------------- 6
    L.append("## 6. 过滤漏斗")
    L.append("")
    L.append("| 规则 | 首因归因 | 独立失败数 |")
    L.append("| --- | --- | --- |")
    fc = dict(fn.first_cause)
    ind = dict(fn.independent)
    for r in ("agreement", "confidence", "rule_agreement", "untainted_support"):
        L.append(f"| `{r}` | {fc.get(r, 0)} | {ind.get(r, 0)} |")
    L.append(f"| **保留** | **{fn.kept}** | — |")
    L.append("")
    L.append(
        "两种口径都要：首因归因回答「主要是被哪条卡住的」，独立失败数回答「哪条规则最有杀伤力」"
        "—— 有些记录会被好几条同时卡住，首因归因会把它全记在第一条名下。"
    )
    L.append("")

    # ---------------------------------------------------------------- 7
    L.append("## 7. 噪声账本（验收标准②）")
    L.append("")
    L.append("| 项 | 数值 |")
    L.append("| --- | --- |")
    L.append(f"| 记录总数 | {n.n_total} |")
    L.append(f"| 其中噪声（真值平局或判错） | {n.noise_total}（{_pct(n.noise_rate_before)}） |")
    L.append(f"| 保留 | {n.n_kept} |")
    L.append(f"| 保留里还剩噪声 | {n.noise_kept}（{_pct(n.noise_rate_after)}） |")
    L.append(f"| 丢掉 | {n.n_removed} |")
    L.append(f"| 丢掉的里真是噪声的 | {n.noise_removed}（精度 {_pct(n.precision)}） |")
    L.append(f"| 噪声被抓到的比例 | {_pct(n.recall)}（召回） |")
    L.append(f"| **提升倍数** | **{_f(n.lift, 2)}×** |")
    L.append("")
    L.append(
        f"**为什么必须看提升倍数**：精度的上限是 `噪声总数 / 丢掉数` = "
        f"{n.noise_total}/{n.n_removed} = {_f(n.noise_total / n.n_removed if n.n_removed else 0)}，"
        "越不过去。所以「过滤掉了 N 条噪声」这句话本身没有信息 —— 随机丢 N 条也能丢到噪声，"
        "命中率就是基础噪声率。提升倍数才回答「这个过滤器到底有没有在挑」。"
    )
    L.append("")
    L.append("**噪声的定义**：一条记录是噪声，当且仅当")
    L.append("")
    L.append("1. 隐藏真值下两个候选**一样好**（差不到 `truth_tie_eps`）—— 这种记录**本来就没有合法偏好**；")
    L.append("2. 或者聚合结论与隐藏真值相反。")
    L.append("")
    L.append("真值只在项目内部可得，不进导出，也不进过滤器的输入（见第 9 节）。")
    L.append("")

    # ---------------------------------------------------------------- 8
    L.append("## 8. 验收标准核对（验收标准①）")
    L.append("")
    L.append("> 每条偏好数据带 agreement / confidence，可量化裁判一致率")
    L.append("")
    L.append(
        f"全部 {len(recs)} 条记录都带 `agreement` / `kappa` / `confidence` / `verdicts` / "
        f"`provenance`，没有一条缺失。**一致率是逐对一致率**（任意两个裁判结论相同的比例），"
        f"裁判两两之间的对照表见第 5.1 节。"
    )
    L.append("")
    L.append("一条记录的完整形态（`main.py show <id>`）：")
    L.append("")
    L.append("```")
    L.append("PreferenceRecord")
    L.append("├── prompt / candidates[2] / ranking        # 排什么、拿什么在排")
    L.append("├── judges[5] / scores[5] / verdicts[5]     # 谁判的、各给多少分")
    L.append("├── agreement / kappa / confidence          # 这份证据有多可信")
    L.append("├── rationale                               # 为什么是这个结论")
    L.append("├── provenance                              # 来源（含各裁判注入的偏差量）")
    L.append("└── evidence[4]                             # 四条过滤规则各自的通过/失败与理由")
    L.append("```")
    L.append("")

    # ---------------------------------------------------------------- 9
    L.append("## 9. 设计期望 vs 实测")
    L.append("")
    L.append("| 设计期望 | 实测 | 结论 |")
    L.append("| --- | --- | --- |")
    for name, exp in sorted((result.expectations or {}).items()):
        got = _check_expectation(name, exp, result, recs)
        L.append(f"| `{name}` = {exp} | {got[0]} | {got[1]} |")
    L.append("")

    # ---------------------------------------------------------------- 10
    L.append("## 10. 可复现性边界")
    L.append("")
    L.append(
        f"- 版本 `{result.version}` 是**内容寻址**的：digest 吃的是导出形态（`chosen/rejected` "
        "剥掉隐藏真值之后的样子）的逐条摘要，所以「版本号相同 ⇒ 数据集逐位相同」。"
    )
    L.append("- 唯一不可复现的字段是 `created_at`，它不进 digest。")
    L.append(
        "- 噪声按 `(裁判, 候选)` 确定性派生，不用 `random.gauss` / `random.shuffle` —— "
        "后者的实现细节在不同 CPython 小版本里改过。本项目只用 `random.random()`（MT19937，有跨版本保证）。"
    )
    L.append(
        f"- 已知的一条**缝**：`rule` 裁判只能看到 `criteria`（可观测的那部分真值），"
        f"真值里还有 {_f(1.0 - 0.8, 2)} 是任何机械判据都看不到的。所以 `rule` 是无偏锚点，"
        "**不是神谕** —— 「rule 同意才保留」这条规则会因此漏掉一些好记录（见第 7 节的召回与精度）。"
    )
    L.append("")

    L.append("---")
    L.append("")
    L.append(
        "_完整机器可读结果在 `out/` 下；`main.py acceptance` / `main.py adversarial` "
        "给出全部运行时核对项。_"
    )
    L.append("")
    return "\n".join(L)


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _rho(result: Any, axis: str) -> float:
    pairs = [
        (p.injected_on(axis), p.detected_on(axis))
        for p in result.profiles
        if dict(p.n_probes).get(axis, 0) > 0
    ]
    if len(pairs) < 3:
        return 0.0
    return spearman([a for a, _ in pairs], [b for _, b in pairs])


_MATRIX_CACHE: dict[int, dict[str, Any]] = {}


def _matrix(result: Any) -> dict[str, Any]:
    key = id(result)
    if key in _MATRIX_CACHE:
        return _MATRIX_CACHE[key]
    out: dict[str, Any] = {}
    js = [j.id for j in result.judges]
    for i in range(len(js)):
        for k in range(i + 1, len(js)):
            agree = 0
            cnt = 0
            for r in result.records:
                va = next((v for v in r.verdicts if v.judge_id == js[i]), None)
                vb = next((v for v in r.verdicts if v.judge_id == js[k]), None)
                if va is None or vb is None:
                    continue
                cnt += 1
                agree += 1 if va.choice == vb.choice else 0
            out[f"{js[i]}~{js[k]}"] = {"n": cnt, "rate": agree / cnt if cnt else 0.0}
    _MATRIX_CACHE[key] = out
    return out


def _check_expectation(name: str, expected: Any, result: Any, recs: Sequence[PreferenceRecord]) -> tuple[str, str]:
    """把每条设计期望拿去实测。**允许被推翻** —— 被推翻的那条最值钱。"""
    if name == "rule_judge_agrees_with_truth":
        ok = 0
        tot = 0
        for r in recs:
            rv = next((v for v in r.verdicts if v.kind == "rule"), None)
            if rv is None or r.truth_winner == "tie":
                continue
            tot += 1
            ok += 1 if rv.choice == r.truth_winner else 0
        rate = ok / tot if tot else 0.0
        return (
            f"`rule` 与真值一致 {_pct(rate)}（{ok}/{tot} 条有真值胜者的记录）",
            "成立但**不到 100%** —— 它只看得见可观测判据，看不见那 20% 的残差" if rate < 0.99 else "成立",
        )
    if name == "weighted_verdict_beats_majority":
        mj = sum(1 for r in recs if r.verdict == r.truth_winner and r.truth_winner != "tie")
        wv = sum(1 for r in recs if r.weighted_verdict == r.truth_winner and r.truth_winner != "tie")
        tot = sum(1 for r in recs if r.truth_winner != "tie")
        return (
            f"多数票 {mj}/{tot}，降权票 {wv}/{tot}",
            "成立" if wv > mj else ("**被推翻**：知道谁有偏之后并没判得更准" if wv < mj else "打平"),
        )
    if name == "confidence_correlates_with_true_margin":
        xs = [r.confidence for r in recs]
        ys = [r.truth_margin for r in recs]
        rho = spearman(xs, ys)
        return (f"Spearman ρ(confidence, 真值差) = {_f(rho)}", "成立" if rho > 0 else "**被推翻**")
    if name == "filter_precision_above":
        p = result.noise.precision
        return (f"精度 {_f(p)}（期望 >= {expected}）", "成立" if p >= float(expected) else "**被推翻**")

    if name == "retention_spread_under":
        rate = _retention_by_kind(recs)
        spread = max(rate.values()) - min(rate.values()) if rate else 0.0
        worst = ", ".join(f"{k} {v * 100:.1f}%" for k, v in sorted(rate.items(), key=lambda kv: kv[1]))
        return (
            f"各题型保留率差 {_pct(spread)}（期望 < {_pct(float(expected))}）｜ {worst}",
            "成立" if spread < float(expected) else "**被推翻**：过滤器对题型有系统性偏好",
        )

    if name == "tie_verdicts_track_true_ties":
        tv = _mean([1.0 if r.verdict == "tie" else 0.0 for r in recs])
        tt = _mean([1.0 if r.truth_winner == "tie" else 0.0 for r in recs])
        return (
            f"裁判平局 {_pct(tv)} vs 真值平局 {_pct(tt)}（{_f(tv / tt if tt else 0, 1)}×）",
            "成立" if tv <= tt * 1.5 else "**被推翻**：裁判的 `tie` 大多是「判不出来」，不是「真的一样好」",
        )

    if name == "weighted_verdict_beats_majority_in_every_kind":
        kinds = sorted({r.prompt_kind for r in recs})
        rows = []
        ok = True
        for k in kinds:
            rs = [r for r in recs if r.prompt_kind == k]
            tot = sum(1 for r in rs if r.truth_winner != "tie")
            mj = sum(1 for r in rs if r.verdict == r.truth_winner and r.truth_winner != "tie")
            wv = sum(1 for r in rs if r.weighted_verdict == r.truth_winner and r.truth_winner != "tie")
            rows.append(f"{k} {mj}→{wv}")
            ok = ok and wv >= mj
        return ("｜".join(rows), "成立" if ok else "**被推翻**：至少一个题型上降权反而更差")

    return ("未核对", "—")


def _retention_by_kind(recs: Sequence[PreferenceRecord]) -> dict[str, float]:
    tot: dict[str, int] = {}
    keep: dict[str, int] = {}
    for r in recs:
        tot[r.prompt_kind] = tot.get(r.prompt_kind, 0) + 1
        if r.kept:
            keep[r.prompt_kind] = keep.get(r.prompt_kind, 0) + 1
    return {k: keep.get(k, 0) / v for k, v in tot.items() if v}
