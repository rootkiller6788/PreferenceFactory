"""运行时核对：验收清单 + 对抗用例。

本工作区的约定是「不写 pytest，把验证放进流水线自身的运行时核对」，所以这里是
**唯一的质量闸门**：`main.py acceptance` / `main.py adversarial` 必须全绿，交付才算数。

两套的分工：

- **`acceptance`** 核对**这一批结果**是否满足 README 的两条验收标准，以及那些
  代数上必然成立的性质（κ ≤ 一致率、导出不含真值……）。它依赖产物。
- **`adversarial`** 用**刻意构造的极端裁判**去敲检测器：一个永远选第一个的裁判
  必须被位置探针抓到；一个完全干净的裁判一条都不许被抓到。它不依赖产物，
  所以产物过期时它仍然能跑。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import agreement as ag
from .candidates import length_stats, norm_length
from .judges import JudgeContext, judge_pair, rule_quality
from .probes import run_swaps, build_swap_units, build_length_units, build_self_units, run_traps
from .prompts import evaluate
from .rng import Rng, derive_seed
from .schema import (
    BIAS_AXES,
    Candidate,
    JudgeSpec,
    PreferenceRecord,
    Prompt,
    Check,
)

__all__ = ["acceptance_items", "adversarial_items", "summarize", "spearman"]


def _item(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman 秩相关（并列取平均秩）。

    **为什么是秩相关而不是线性相关**：注入的是打分偏移量、检出的是踩中率，
    两者量纲不同、也不该假定线性。校准能要求的只是一场**次序一致**。
    """

    def ranks(v: Sequence[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    rx, ry = ranks(xs), ranks(ys)
    return _pearson(rx, ry)


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


# --------------------------------------------------------------------------
# 验收清单
# --------------------------------------------------------------------------

def acceptance_items(result: Any, cfg: Mapping[str, Any], out_dir: Path | str) -> list[dict[str, Any]]:
    acc = dict(cfg.get("acceptance") or {})
    items: list[dict[str, Any]] = []
    recs = list(result.records)
    kept = [r for r in recs if r.kept]

    # 1) 记录数
    need = int(acc.get("min_records", 0))
    items.append(
        _item("记录规模", len(recs) >= need, f"{len(recs)} 条（要求 >= {need}）")
    )

    # 2) agreement 有界
    bad = [r.id for r in recs if not (0.0 <= r.agreement <= 1.0)]
    items.append(
        _item("一致率落在 [0, 1]", not bad, f"越界 {len(bad)} 条" + (f"，例：{bad[:3]}" if bad else ""))
    )

    # 3) κ ≤ agreement（代数必然：κ=(P̄−Pe)/(1−Pe)，而 Pe(1−P̄) ≥ 0）
    if acc.get("require_kappa_le_agreement", True):
        bad = [r.id for r in recs if r.kappa > r.agreement + 1e-9]
        items.append(
            _item(
                "κ ≤ 一致率",
                not bad,
                f"违反 {len(bad)} 条"
                + (f"，例：{bad[:3]}" if bad else f"（κ̄={_mean([r.kappa for r in recs]):.3f}，一致率̄={_mean([r.agreement for r in recs]):.3f}）"),
            )
        )

    # 4) 不冤枉好人：注入偏差为 0 的裁判，检出量必须很小
    tol = float(acc.get("unbiased_bias_tolerance", 0.12))
    if acc.get("require_unbiased_judges_clean", True):
        worst: list[str] = []
        for p in result.profiles:
            if p.max_injected > 0:
                continue
            for axis in BIAS_AXES:
                d = p.detected_on(axis)
                n = dict(p.n_probes).get(axis, 0)
                if n and d > tol:
                    worst.append(f"{p.judge_id}/{axis}={d:.3f}(n={n})")
        items.append(
            _item(
                "无偏裁判不被指控",
                not worst,
                "全部低于容差 " + f"{tol:.2f}" if not worst else "；".join(worst),
            )
        )

    # 5) 校准：每轴的「注入量 → 检出率」次序一致
    min_rho = float(acc.get("min_rank_correlation", 0.60))
    for axis in BIAS_AXES:
        pairs = [
            (p.injected_on(axis), p.detected_on(axis))
            for p in result.profiles
            if dict(p.n_probes).get(axis, 0) > 0
        ]
        if len(pairs) < 3:
            items.append(_item(f"校准[{axis}]", True, f"只有 {len(pairs)} 个可比裁判，跳过"))
            continue
        rho = spearman([a for a, _ in pairs], [b for _, b in pairs])
        items.append(
            _item(
                f"校准[{axis}]：注入次序 == 检出次序",
                rho >= min_rho,
                f"Spearman ρ={rho:.3f}（要求 >= {min_rho:.2f}），{len(pairs)} 个裁判",
            )
        )

    # 6) 渲染 ↔ 判定往返
    if result.candidates and result.prompts:
        by_id = {p.id: p for p in result.prompts}
        bad = 0
        for c in result.candidates:
            p = by_id.get(c.prompt_id)
            if p is None:
                continue
            res = evaluate(p, c.text)
            got = tuple(k for k in ("exact", "clean", "punct", "tidy") if res[k])
            if set(got) != set(c.checks_passed):
                bad += 1
        items.append(
            _item(
                "渲染 ↔ 判据往返一致",
                bad == 0,
                f"{len(result.candidates)} 个候选里 {bad} 个对不上（rule 裁判读的必须是候选真实写的东西）",
            )
        )

    # 7) 长度与真值不相关（长度偏差能被识别的前提）
    if result.candidates:
        rho = _pearson([float(c.length) for c in result.candidates], [c.quality for c in result.candidates])
        bound = float(acc.get("length_quality_corr_bound", 0.30))
        items.append(
            _item(
                "长度与真值弱相关",
                abs(rho) <= bound,
                f"corr(长度, 真值)={rho:+.3f}（|·| <= {bound:.2f}）。"
                "相关太强时「长度偏差」与「长答案确实更好」就拆不开了",
            )
        )

    # 8) 噪声率下降
    nr = result.noise
    if acc.get("require_noise_rate_drops", True):
        items.append(
            _item(
                "过滤后噪声率下降",
                nr.noise_rate_after < nr.noise_rate_before,
                f"{nr.noise_rate_before:.3f} → {nr.noise_rate_after:.3f}"
                f"（保留 {nr.n_kept}/{nr.n_total} 条）",
            )
        )

    # 9) 过滤掉的噪声规模
    need_rm = int(acc.get("min_noise_removed", 0))
    items.append(
        _item("过滤掉的噪声规模", nr.noise_removed >= need_rm, f"{nr.noise_removed} 条（要求 >= {need_rm}）")
    )

    # 10) 过滤的**提升倍数**（绝对精度被基础噪声率钉死，看它没意义）
    bound_l = float(acc.get("min_filter_lift", 1.5))
    items.append(
        _item(
            "过滤提升倍数（比随机丢好多少）",
            nr.lift >= bound_l,
            f"lift={nr.lift:.2f}×（要求 >= {bound_l:.1f}）；"
            f"precision={nr.precision:.3f}，recall={nr.recall:.3f}。"
            f"精度的上限是 noise_total/丢掉数 = {nr.noise_total}/{nr.n_removed} = "
            f"{(nr.noise_total / nr.n_removed if nr.n_removed else 0):.3f}，越不过去",
        )
    )

    # 10b) 每条轴的探针样本数
    if acc.get("require_all_axes_sampled", True):
        empty = [
            f"{p.judge_id}/{axis}"
            for p in result.profiles
            for axis in BIAS_AXES
            if p.kind == "llm" and dict(p.n_probes).get(axis, 0) == 0
        ]
        items.append(
            _item(
                "每条偏差轴都有探针样本",
                not empty,
                "全部 > 0" if not empty else f"样本为 0，该轴无从测量：{empty}",
            )
        )

    # 11) 导出不含隐藏真值
    pref = Path(out_dir) / "preferences.jsonl"
    if pref.exists():
        text = pref.read_text(encoding="utf-8")
        leaked = [k for k in ("latent", "quality", "satisfies", "checks_passed") if f'"{k}"' in text]
        items.append(
            _item(
                "导出不含隐藏真值",
                not leaked,
                "干净" if not leaked else f"泄漏字段：{leaked}（下游会拿真值当特征）",
            )
        )

    # 12) 产物 ≡ 内存
    from .dataset import read_jsonl

    on_disk = read_jsonl(Path(out_dir) / "records.jsonl")
    same = len(on_disk) == len(recs) and all(
        PreferenceRecord.from_dict(row).to_dict() == r.to_dict() for row, r in zip(on_disk, recs)
    )
    items.append(_item("产物 ≡ 内存（逐位）", same, f"{len(on_disk)} vs {len(recs)} 条"))

    # 13) 每条保留记录都带 agreement / confidence（验收标准①）
    missing = [r.id for r in kept if r.agreement is None or r.confidence is None]
    items.append(
        _item(
            "每条记录自带 agreement / confidence",
            not missing,
            f"{len(kept)} 条保留记录，缺失 {len(missing)} 条",
        )
    )

    # 14) 每条记录都能回答「谁判的」
    nojudge = [r.id for r in recs if not r.judges or len(r.verdicts) != len(r.judges)]
    items.append(
        _item("每条记录可溯源到具体裁判", not nojudge, f"{len(recs)} 条，缺失 {len(nojudge)} 条")
    )

    return items


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


# --------------------------------------------------------------------------
# 对抗用例
# --------------------------------------------------------------------------

def _synthetic_pool(rng: Rng, n: int = 12) -> tuple[list[Prompt], dict[str, list[Candidate]]]:
    """造一个小池子给对抗用例用（不依赖 `data/`，产物过期也能跑）。"""
    from .prompts import build_pool

    prompts = build_pool(rng, n)
    from .candidates import generate_for_prompt
    from .config import FAMILIES

    by: dict[str, list[Candidate]] = {}
    for p in prompts:
        by[p.id] = generate_for_prompt(p, FAMILIES, rng.fork("adv", p.id), 3)
    return prompts, by


def adversarial_items(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    rng = Rng(derive_seed("adversarial", int(cfg.get("seed", 0))))
    prompts, by = _synthetic_pool(rng)
    from .candidates import all_pairs, length_stats

    pairs = all_pairs(prompts, by)
    allc = [c for cs in by.values() for c in cs]
    mu, sd = length_stats(allc)
    ctx = JudgeContext(mu_len=mu, sigma_len=sd, tie_margin=float((cfg.get("judging") or {}).get("tie_margin", 0.05)))
    items: list[dict[str, Any]] = []

    swap_units = build_swap_units(pairs, 60, 0.02, rng.fork("adv-swap"))
    lt, lc = build_length_units(pairs, 60, rng.fork("adv-len"))

    def mk(jid: str, kind: str, **kw: Any) -> JudgeSpec:
        return JudgeSpec(id=jid, kind=kind, **kw)

    cases: list[tuple[str, JudgeSpec, Callable[[dict[str, float]], bool]]] = [
        ("永远选第一个", mk("adv-first", "llm", family="terse", beta=1.0, lam=0.0, self_bias=0.0, noise=0.0),
         lambda d: d["position"] >= 0.95),
        ("永远选更长的", mk("adv-long", "llm", family="terse", beta=0.0, lam=1.0, self_bias=0.0, noise=0.0),
         lambda d: d["length"] >= 0.95),
        ("永远选自家族", mk("adv-self", "llm", family="verbose", beta=0.0, lam=0.0, self_bias=1.0, noise=0.0),
         lambda d: d["self"] >= 0.90),
        ("完全干净的裁判", mk("adv-clean", "llm", family="terse", beta=0.0, lam=0.0, self_bias=0.0, noise=0.05),
         lambda d: max(d.values()) <= 0.35),
    ]

    for label, judge, ok_fn in cases:
        pr = run_swaps(judge, swap_units, ctx)
        lt_res, _ = run_traps(judge, lt, lc, ctx, axis="length", definition="", control_definition="")
        if judge.family is not None:
            st, sc = build_self_units(pairs, judge.family, 60, rng.fork("adv-self", judge.id))
            st_res, _ = run_traps(judge, st, sc, ctx, axis="self", definition="", control_definition="")
        else:
            st_res = pr
        detected = {
            "position": pr.rate,
            "length": lt_res.rate,
            "self": st_res.rate if judge.family else 0.0,
        }
        n = {"position": pr.n, "length": lt_res.n, "self": st_res.n}
        ok = ok_fn(detected)
        items.append(
            _item(
                f"对抗：{label}",
                ok,
                "检出 " + " ".join(f"{k}={detected[k]:.3f}(n={n[k]})" for k in ("position", "length", "self")),
            )
        )

    # ---- 结构性对抗

    # rule 裁判必须跟着判据走：把一条文本的判据全破坏，它的分必须掉到底
    p = prompts[0]
    c = by[p.id][0]
    good, _picked = rule_quality(p, c)
    broken = Candidate(
        id=c.id + "-broken", prompt_id=c.prompt_id, family=c.family,
        text=c.text + "〔?〕  ", latent=c.latent, checks_passed=(), n_checks=c.n_checks,
        criteria=c.criteria, polish=c.polish,
        verbosity=c.verbosity, length=len(c.text) + 5, pad_chars=c.pad_chars + 5,
    )
    bad_score, _ = rule_quality(p, broken)
    items.append(
        _item(
            "对抗：破坏判据后 rule 分数必须下降",
            bad_score < good,
            f"{good:.3f} → {bad_score:.3f}（rule 只读判据，不读长度）",
        )
    )

    # 两个完全一样的候选：任何无偏裁判都必须判平局
    same = judge_pair(mk("adv-tie", "rule"), p, c, c, ctx=ctx)
    items.append(
        _item("对抗：两条一模一样的候选 → 平局", same.choice == "tie", f"得到 {same.choice}（margin={same.margin:.4f}）")
    )

    # 一致率的端点
    items.append(
        _item(
            "对抗：一致率端点",
            ag.agreement_of(["a", "a", "a", "a"]) == 1.0 and ag.agreement_of(["a", "b", "tie", "a"]) < 1.0,
            f"全同={ag.agreement_of(['a'] * 4):.3f}，混合={ag.agreement_of(['a', 'b', 'tie', 'a']):.3f}",
        )
    )

    # 置信度单调性
    base = ag.confidence_of(0.5, 0.3, 0.6)
    up_a = ag.confidence_of(0.7, 0.3, 0.6)
    up_d = ag.confidence_of(0.5, 0.5, 0.6)
    items.append(
        _item(
            "对抗：置信度对两项都单调不减",
            up_a >= base and up_d >= base,
            f"base={base:.3f}，agreement↑={up_a:.3f}，decisiveness↑={up_d:.3f}",
        )
    )
    items.append(
        _item(
            "对抗：任一为 0 则置信度为 0",
            ag.confidence_of(0.0, 1.0, 0.6) == 0.0 and ag.confidence_of(1.0, 0.0, 0.6) == 0.0,
            "乘法口径：一项为 0 就归零（加法会让一项的高分掩盖另一项的 0）",
        )
    )

    # 过滤门槛的端点
    from .filter import evaluate_record, build_funnel, filter_records

    r0 = _dummy_record()
    loose = {"filter": {"min_agreement": 0.0, "min_confidence": 0.0, "bias_threshold": 1.0,
                        "require_rule_agreement": False, "require_untainted_support": False}}
    tight = {"filter": {"min_agreement": 1.0, "min_confidence": 1.0, "bias_threshold": 1.0,
                        "require_rule_agreement": False, "require_untainted_support": False}}
    ok_loose, _, _ = evaluate_record(r0, loose, {})
    ok_tight, _, _ = evaluate_record(r0, tight, {})
    items.append(
        _item("对抗：过滤门槛端点", ok_loose and not ok_tight, f"全开={ok_loose}，全关={ok_tight}")
    )

    # 血缘图拒绝成环
    from .lineage import Lineage, Node, Edge

    g = Lineage()
    g.add_node(Node("a", "record"))
    g.add_node(Node("b", "record"))
    g.add_edge(Edge("a", "b", "x"))
    try:
        g.add_edge(Edge("b", "a", "x"))
        cyc = False
    except ValueError:
        cyc = True
    items.append(_item("对抗：血缘图拒绝成环", cyc, "成环时抛 ValueError（静默成环会让 DOT 丢边）"))

    # 判据难度的端点
    easy = Prompt(id="t1", kind="arith", question="q", expected="1",
                  checks=(Check("exact", 0.01), Check("clean", 0.01), Check("punct", 0.01), Check("tidy", 0.01)))
    hard = Prompt(id="t2", kind="arith", question="q", expected="1",
                  checks=(Check("exact", 0.99), Check("clean", 0.99), Check("punct", 0.99), Check("tidy", 0.99)))
    fam = {"id": "terse", "skill": 0.7, "spread": 0.1, "verbosity": 1.0}
    from .candidates import generate

    ce = generate(easy, fam, 0, rng.fork("adv-easy"))
    ch = generate(hard, fam, 0, rng.fork("adv-hard"))
    items.append(
        _item(
            "对抗：判据难度端点",
            len(ce.checks_passed) == 4 and len(ch.checks_passed) == 0,
            f"全易={len(ce.checks_passed)}/4，全难={len(ch.checks_passed)}/4",
        )
    )

    return items


def _dummy_record() -> PreferenceRecord:
    from .schema import Evidence

    c = Candidate(
        id="x#c0", prompt_id="x", family="terse", text="1。", latent=0.5,
        checks_passed=("exact",), n_checks=4, criteria=1.0, polish=0.5,
        verbosity=1.0, length=2, pad_chars=1,
    )
    return PreferenceRecord(
        id="x#p00", prompt_id="x", prompt_kind="arith", question="q",
        candidates=(c, c), ranking=(), judges=("rule",), scores=(0.0,),
        verdicts=(), agreement=0.5, kappa=0.2, confidence=0.25,
        verdict="a", weighted_verdict="a", rationale="", evidence=(Evidence("agreement", True, ""),),
    )


def summarize(items: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    ok = sum(1 for i in items if i["ok"])
    return ok, len(items)
