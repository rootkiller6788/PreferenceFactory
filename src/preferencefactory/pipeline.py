"""编排：Prompt → 候选 → 配对 → 判定 → 一致率 → 置信度 → 偏差 → 过滤 → 版本。

**流水线顺序不是随意排的**：探针必须在主记录流**之前**跑。因为「按检出偏差降权的
投票」要用探针结果，而探针只需要候选池、不需要记录 —— 先跑探针，后面聚合时权重
就已经是实测值了。反过来（先判记录再补探针）会得到一个「用结论去算权重」的循环。

产物清单写在 `OUT_FILES` 里，**声明与实际必须一致**（由 `load_result` 的校验兜住：
缺任何一个文件它返回 `None`，于是只读命令会重跑而不是读半个产物）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .agreement import (
    agreement_of,
    category_distribution,
    confidence_of,
    decisiveness_of,
    kappa_of,
    aggregate,
    weighted_aggregate,
    bias_weight,
)
from .candidates import all_pairs, generate_for_prompt, length_stats, normalize_families
from .config import (
    expectations_of,
    judges_of,
    load_config,
    protocol_digest,
    resolve_path,
    version_digest,
)
from .dataset import build_manifest, content_digest, export_record, version_id, write_jsonl
from .filter import Funnel, NoiseReport, build_funnel, filter_records, noise_report
from .judges import JudgeContext, judge_pair, normalize_judges
from .lineage import Lineage, Edge, Node
from .prompts import build_pool, load_prompts, write_pool
from .probes import run_probes
from .rng import Rng, canonical_json, sha256_text
from .schema import (
    BiasProfile,
    Candidate,
    JudgeSpec,
    PreferenceRecord,
    ProbeResult,
    Prompt,
)

__all__ = [
    "OUT_FILES",
    "PipelineResult",
    "run_pipeline",
    "write_outputs",
    "load_result",
    "materialize",
]

OUT_FILES: tuple[str, ...] = (
    "version.json",
    "summary.json",
    #: prompt 与候选也落盘。它们是**输入**，但把它们快照进产物有两个好处：
    #: ① 产物自包含，拷给别人就能复算；② 验收清单里的「渲染↔判据往返」「长度与
    #: 真值弱相关」两条**从产物读时也能跑** —— 不落盘的话这两条会被静默跳过，
    #: 而「静默跳过」比「失败」危险得多。
    "prompts.jsonl",
    "candidates.jsonl",
    "records.jsonl",
    "preferences.jsonl",
    "dataset.json",
    "agreement.json",
    "bias.json",
    "probes.json",
    "funnel.json",
    "lineage.json",
    "lineage.dot",
)


@dataclass(frozen=True)
class PipelineResult:
    cfg: Mapping[str, Any]
    version: str
    created_at: str
    prompts: tuple[Prompt, ...]
    candidates: tuple[Candidate, ...]
    judges: tuple[JudgeSpec, ...]
    records: tuple[PreferenceRecord, ...]
    profiles: tuple[BiasProfile, ...]
    probes: tuple[ProbeResult, ...]
    funnel: Funnel
    noise: NoiseReport
    manifest: Any
    lineage: Lineage
    expectations: Mapping[str, Any] = field(default_factory=dict)
    counts: Mapping[str, Any] = field(default_factory=dict)

    # ---- 便捷视图

    @property
    def kept(self) -> tuple[PreferenceRecord, ...]:
        return tuple(r for r in self.records if r.kept)

    def record(self, rid: str) -> PreferenceRecord | None:
        for r in self.records:
            if r.id == rid:
                return r
        return None

    def profile_of(self, judge_id: str) -> BiasProfile | None:
        for p in self.profiles:
            if p.judge_id == judge_id:
                return p
        return None


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def run_pipeline(cfg: Mapping[str, Any], *, root: Path | str = ".") -> PipelineResult:
    root = Path(root)
    seed = int(cfg.get("seed", 0))
    rng = Rng(derive_key("root", seed))

    # ---- 1) prompt 池
    pool_path = resolve_path(root, str((cfg.get("prompts") or {}).get("index", "data/prompts.jsonl")))
    n_prompts = int((cfg.get("prompts") or {}).get("n_prompts", 96))
    if pool_path.exists():
        prompts = tuple(load_prompts(pool_path))
        if len(prompts) < n_prompts:
            prompts = prompts + tuple(build_pool(rng.fork("pool-extend"), n_prompts - len(prompts)))
            write_pool(pool_path, prompts)
        prompts = prompts[:n_prompts]
    else:
        prompts = tuple(build_pool(rng.fork("pool"), n_prompts))
        write_pool(pool_path, prompts)

    # ---- 2) 候选
    families = normalize_families((cfg.get("candidates") or {}).get("families", ()))
    n_cand = int((cfg.get("candidates") or {}).get("n_candidates", 3))
    by_prompt: dict[str, list[Candidate]] = {}
    candidates: list[Candidate] = []
    for p in prompts:
        cs = generate_for_prompt(p, families, rng.fork("cand", p.id), n_cand)
        by_prompt[p.id] = cs
        candidates.extend(cs)
    candidates_t = tuple(candidates)

    # ---- 3) 长度口径（全池，不是逐对）
    mu_len, sigma_len = length_stats(candidates_t)
    jcfg = dict(cfg.get("judging") or {})
    ctx = JudgeContext(
        mu_len=mu_len,
        sigma_len=sigma_len,
        tie_margin=float(jcfg.get("tie_margin", 0.05)),
    )

    # ---- 4) 裁判池
    judges = tuple(normalize_judges(judges_of(cfg)))
    pairs = all_pairs(prompts, by_prompt)

    # ---- 5) 探针（**在主记录流之前**）
    probe_results, profiles = run_probes(judges, pairs, ctx, cfg, rng.fork("probe"))
    profiles_by_id = {p.judge_id: p for p in profiles}
    weights = {jid: bias_weight(p.max_detected) for jid, p in profiles_by_id.items()}

    # ---- 6) 判定 → 记录
    spec_digest = protocol_digest(cfg)
    conf_scale = float(jcfg.get("confidence_scale", 0.60))
    tie_eps = float((cfg.get("acceptance") or {}).get("truth_tie_eps", 0.02))
    raw_records: list[PreferenceRecord] = []
    for idx, (p, x, y) in enumerate(pairs):
        order_rng = rng.fork("order", p.id, x.id, y.id)
        a, b = (x, y) if order_rng.bernoulli(0.5) else (y, x)
        verdicts = tuple(judge_pair(j, p, a, b, ctx=ctx) for j in judges)
        raw_records.append(
            _make_record(p, a, b, verdicts, judges, idx, spec_digest, weights, conf_scale, tie_eps)
        )

    # ---- 7) 一致率的两轮：先算总体类别分布，再逐条算 κ
    table = [_counts(r) for r in raw_records]
    p_j = category_distribution(table)
    records: list[PreferenceRecord] = []
    for r, cnt in zip(raw_records, table):
        pi, kap = kappa_of(cnt, p_j, len(judges))
        records.append(
            PreferenceRecord(
                **{**_asdict_shallow(r), "agreement": pi, "kappa": kap}
            )
        )

    # ---- 8) 排序（每个 prompt 的 Copeland 排名）
    records = _attach_rankings(records, by_prompt)

    # ---- 9) 过滤
    records_t = tuple(records)
    filtered = filter_records(records_t, cfg, profiles_by_id)
    funnel = build_funnel(filtered)
    noise = noise_report(filtered)

    # ---- 10) 版本与血缘
    kept = tuple(r for r in filtered if r.kept)
    manifest = build_manifest(
        kept,
        filtered,
        extra={
            "agreement_mean_all": _mean([r.agreement for r in filtered]),
            "confidence_mean_all": _mean([r.confidence for r in filtered]),
            "truth_tie_rate": _mean([1.0 if r.truth_winner == "tie" else 0.0 for r in filtered]),
        },
    )
    lineage = _build_lineage(prompts, by_prompt, judges, filtered, manifest.version)

    vd = version_digest(
        cfg,
        [("prompts", [p.to_dict() for p in prompts]), ("candidates", [c.to_dict() for c in candidates_t])],
    )
    counts = {
        "prompts": len(prompts),
        "candidates": len(candidates_t),
        "pairs": len(pairs),
        "records": len(filtered),
        "kept": len(kept),
        "judges": len(judges),
        "probes": len(probe_results),
        "families": len(families),
        "input_digest": vd[:12],
    }

    return PipelineResult(
        cfg=cfg,
        version=manifest.version,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        prompts=prompts,
        candidates=candidates_t,
        judges=judges,
        records=tuple(filtered),
        profiles=tuple(profiles),
        probes=tuple(probe_results),
        funnel=funnel,
        noise=noise,
        manifest=manifest,
        lineage=lineage,
        expectations=expectations_of(cfg),
        counts=counts,
    )


def derive_key(tag: str, seed: int) -> int:
    from .rng import derive_seed

    return derive_seed(tag, seed)


def _counts(rec: PreferenceRecord) -> dict[str, int]:
    out = {"a": 0, "b": 0, "tie": 0}
    for v in rec.verdicts:
        out[v.choice] = out.get(v.choice, 0) + 1
    return out


_RECORD_FIELDS = (
    "id",
    "prompt_id",
    "prompt_kind",
    "question",
    "candidates",
    "ranking",
    "judges",
    "scores",
    "verdicts",
    "agreement",
    "kappa",
    "confidence",
    "verdict",
    "weighted_verdict",
    "rationale",
    "provenance",
    "truth",
    "evidence",
    "kept",
)


def _asdict_shallow(rec: PreferenceRecord) -> dict[str, Any]:
    return {f: getattr(rec, f) for f in _RECORD_FIELDS}


def _make_record(
    prompt: Prompt,
    a: Candidate,
    b: Candidate,
    verdicts: tuple[Any, ...],
    judges: Sequence[JudgeSpec],
    idx: int,
    spec_digest: str,
    weights: Mapping[str, float],
    conf_scale: float,
    tie_eps: float = 0.02,
) -> PreferenceRecord:
    ag = agreement_of([v.choice for v in verdicts])
    dec = decisiveness_of(verdicts)
    conf = confidence_of(ag, dec, conf_scale)
    verdict = aggregate(verdicts)
    wverdict = weighted_aggregate(verdicts, weights)
    scores = tuple(v.score_a - v.score_b for v in verdicts)
    truth_winner = _truth_of(a, b, tie_eps)
    return PreferenceRecord(
        id=f"{prompt.id}#p{idx:02d}",
        prompt_id=prompt.id,
        prompt_kind=prompt.kind,
        question=prompt.question,
        candidates=(a, b),
        ranking=(),
        judges=tuple(j.id for j in judges),
        scores=scores,
        verdicts=verdicts,
        agreement=ag,
        kappa=0.0,  # 第二轮填
        confidence=conf,
        verdict=verdict,
        weighted_verdict=wverdict,
        rationale=_record_rationale(verdict, wverdict, verdicts),
        provenance={
            "spec_digest": spec_digest,
            "decisiveness": round(dec, 6),
            "weights": dict(weights),
        },
        truth={
            "winner": truth_winner,
            "margin": round(abs(a.quality - b.quality), 6),
            "quality_a": round(a.quality, 6),
            "quality_b": round(b.quality, 6),
            "families": [a.family, b.family],
            "lengths": [a.length, b.length],
        },
        evidence=(),
        kept=True,
    )


def _truth_of(a: Candidate, b: Candidate, eps: float = 0.02) -> str:
    """真值胜者。**带平局容差**：差不到 `eps` 就算「本来就没有偏好」。

    真值连续之后严格相等几乎不会发生，但「差 0.001」和「一样好」在语义上是同一件事
    —— 那种记录上任何判定都是噪声，必须能被算作噪声，否则过滤器的精度会被高估。
    """
    d = a.quality - b.quality
    if d > eps:
        return "a"
    if d < -eps:
        return "b"
    return "tie"


def _record_rationale(verdict: str, wverdict: str, verdicts: Sequence[Any]) -> str:
    votes = _counts_choices([v.choice for v in verdicts])
    parts = [f"{k}×{v}" for k, v in sorted(votes.items()) if v]
    s = f"多数票 {verdict.upper()}（{'/'.join(parts)}）"
    if wverdict != verdict:
        s += f"；按检出偏差降权后变为 {wverdict.upper()} —— 说明原结论靠有偏裁判撑着"
    return s


def _counts_choices(choices: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in choices:
        out[c] = out.get(c, 0) + 1
    return out


def _attach_rankings(
    records: list[PreferenceRecord], by_prompt: Mapping[str, Sequence[Candidate]]
) -> list[PreferenceRecord]:
    """每个 prompt 上的 Copeland 排名：赢一次 +1，输一次 −1，平局 0。

    排名**只用聚合结论**，不用真值 —— 真值若在排序里出现，这份数据就没法用来
    评估「排序错了多少」了。
    """
    wins: dict[str, dict[str, int]] = {}
    for r in records:
        d = wins.setdefault(r.prompt_id, {})
        for c in r.candidates:
            d.setdefault(c.id, 0)
        if r.verdict == "tie":
            continue
        winner = r.a if r.verdict == "a" else r.b
        loser = r.b if r.verdict == "a" else r.a
        d[winner.id] += 1
        d[loser.id] -= 1
    out: list[PreferenceRecord] = []
    for r in records:
        d = wins.get(r.prompt_id, {})
        ordered = sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))
        out.append(
            PreferenceRecord(**{**_asdict_shallow(r), "ranking": tuple(k for k, _ in ordered)})
        )
    return out


def _mean(xs: Sequence[float]) -> float:
    return round(sum(xs) / len(xs), 6) if xs else 0.0


def _build_lineage(
    prompts: Sequence[Prompt],
    by_prompt: Mapping[str, Sequence[Candidate]],
    judges: Sequence[JudgeSpec],
    records: Sequence[PreferenceRecord],
    version: str,
) -> Lineage:
    g = Lineage()
    g.add_node(Node("root", "root", f"PreferenceFactory {version}"))
    for p in prompts:
        g.add_node(Node(p.id, "prompt", f"{p.kind}: {p.question[:24]}", {"n_checks": p.n_checks}))
        g.add_edge(Edge("root", p.id, "contains"))
        for c in by_prompt.get(p.id, ()):
            nid = f"{c.id}"
            g.add_node(Node(nid, "candidate", f"{c.family} q={c.quality:.2f}", {"family": c.family, "length": c.length}))
            g.add_edge(Edge(p.id, nid, "generated", c.family))
    for j in judges:
        jid = f"judge:{j.id}"
        g.add_node(Node(jid, "judge", f"{j.kind}/{j.family or '-'}", j.to_dict()))
        g.add_edge(Edge("root", jid, "pool"))
    for r in records:
        rid = f"rec:{r.id}"
        g.add_node(
            Node(
                rid,
                "record",
                f"{r.verdict.upper()} ag={r.agreement:.2f} cf={r.confidence:.2f}",
                {"kept": r.kept, "noise": r.is_noise},
            )
        )
        g.add_edge(Edge(r.prompt_id, rid, "paired"))
        for c in r.candidates:
            g.add_edge(Edge(c.id, rid, "compared"))
        for v in r.verdicts:
            g.add_edge(Edge(f"judge:{v.judge_id}", rid, "judged", v.choice))
    return g


# --------------------------------------------------------------------------
# 落盘 / 读回
# --------------------------------------------------------------------------

def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


def write_outputs(result: PipelineResult, out_dir: Path | str) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def w(name: str, obj: Any) -> None:
        p = out / name
        _write_json(p, obj)
        written.append(p)

    w(
        "version.json",
        {
            "version": result.version,
            "digest": result.manifest.digest,
            "created_at": result.created_at,
            "protocol_digest": protocol_digest(result.cfg),
            "counts": dict(result.counts),
        },
    )
    w(
        "summary.json",
        {
            "version": result.version,
            "counts": dict(result.counts),
            "funnel": result.funnel.to_dict(),
            "noise": result.noise.to_dict(),
            "judges": [j.to_dict() for j in result.judges],
            "expectations": dict(result.expectations),
        },
    )
    prompts_path = out / "prompts.jsonl"
    write_jsonl(prompts_path, (p.to_dict() for p in result.prompts))
    written.append(prompts_path)

    cand_path = out / "candidates.jsonl"
    write_jsonl(cand_path, (c.to_dict() for c in result.candidates))
    written.append(cand_path)

    records_path = out / "records.jsonl"
    write_jsonl(records_path, (r.to_dict() for r in result.records))
    written.append(records_path)

    pref_path = out / "preferences.jsonl"
    write_jsonl(pref_path, (export_record(r) for r in result.kept))
    written.append(pref_path)

    w("dataset.json", result.manifest.to_dict())
    w(
        "agreement.json",
        {
            "n_records": len(result.records),
            "overall": _agreement_summary(result.records),
            "by_judge_pair": _pairwise_matrix(result),
            "distribution": _histogram([r.agreement for r in result.records]),
            "confidence_distribution": _histogram([r.confidence for r in result.records]),
        },
    )
    w(
        "bias.json",
        {
            "profiles": [p.to_dict() for p in result.profiles],
            "by_axis": _bias_by_axis(result.profiles),
        },
    )
    w("probes.json", [p.to_dict() for p in result.probes])
    w(
        "funnel.json",
        {"funnel": result.funnel.to_dict(), "noise": result.noise.to_dict()},
    )
    w("lineage.json", result.lineage.to_dict())
    dot = out / "lineage.dot"
    dot.parent.mkdir(parents=True, exist_ok=True)
    dot.write_text(result.lineage.to_dot(), encoding="utf-8", newline="\n")
    written.append(dot)
    return written


def _agreement_summary(records: Sequence[PreferenceRecord]) -> dict[str, Any]:
    return {
        "agreement_mean": _mean([r.agreement for r in records]),
        "kappa_mean": _mean([r.kappa for r in records]),
        "confidence_mean": _mean([r.confidence for r in records]),
        "tie_rate": _mean([1.0 if r.verdict == "tie" else 0.0 for r in records]),
    }


def _pairwise_matrix(result: PipelineResult) -> dict[str, Any]:
    """裁判两两之间的一致率。**这是「一致率」最直接的一张表**：
    谁跟谁老是不对付，一眼就能看出来。
    """
    out: dict[str, Any] = {}
    js = [j.id for j in result.judges]
    for i in range(len(js)):
        for k in range(i + 1, len(js)):
            agree = 0
            n = 0
            for r in result.records:
                va = next((v for v in r.verdicts if v.judge_id == js[i]), None)
                vb = next((v for v in r.verdicts if v.judge_id == js[k]), None)
                if va is None or vb is None:
                    continue
                n += 1
                agree += 1 if va.choice == vb.choice else 0
            out[f"{js[i]}~{js[k]}"] = {
                "n": n,
                "agree": agree,
                "rate": round(agree / n, 6) if n else 0.0,
            }
    return out


def _bias_by_axis(profiles: Sequence[BiasProfile]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for axis in ("position", "length", "self"):
        rows = []
        for p in profiles:
            rows.append(
                {
                    "judge_id": p.judge_id,
                    "kind": p.kind,
                    "injected": p.injected_on(axis),
                    "detected": p.detected_on(axis),
                    "n": dict(p.n_probes).get(axis, 0),
                }
            )
        # 按注入量排序：校准看的是**次序**，排序后一眼就能看出次序有没有乱。
        rows.sort(key=lambda r: (-r["injected"], r["judge_id"]))
        out[axis] = rows
    return out


def _histogram(xs: Sequence[float], bins: int = 10) -> list[dict[str, Any]]:
    if not xs:
        return []
    lo, hi = 0.0, 1.0
    width = (hi - lo) / bins
    counts = [0] * bins
    for x in xs:
        i = int((x - lo) / width)
        counts[min(bins - 1, max(0, i))] += 1
    return [
        {"lo": round(lo + i * width, 3), "hi": round(lo + (i + 1) * width, 3), "n": counts[i]}
        for i in range(bins)
    ]


def load_result(out_dir: Path | str) -> PipelineResult | None:
    """从产物重建 `PipelineResult`。

    **缺任何一个文件就返回 `None`** —— 半个产物比没有产物危险得多：读命令会
    以为自己读到了完整结果，然后在缺失的那部分上静默给出 0。
    """
    out = Path(out_dir)
    for name in OUT_FILES:
        if not (out / name).exists():
            return None
    from .dataset import read_jsonl

    version = json.loads((out / "version.json").read_text(encoding="utf-8"))
    records = tuple(
        PreferenceRecord.from_dict(row) for row in read_jsonl(out / "records.jsonl")
    )
    funnel_raw = json.loads((out / "funnel.json").read_text(encoding="utf-8"))
    f = funnel_raw["funnel"]
    n = funnel_raw["noise"]
    funnel = Funnel(
        total=int(f["total"]),
        kept=int(f["kept"]),
        first_cause=tuple((k, int(v)) for k, v in f["first_cause"].items()),
        independent=tuple((k, int(v)) for k, v in f["independent"].items()),
    )
    noise = NoiseReport(
        n_total=int(n["n_total"]),
        noise_total=int(n["noise_total"]),
        n_kept=int(n["n_kept"]),
        noise_kept=int(n["noise_kept"]),
        n_removed=int(n["n_removed"]),
        noise_removed=int(n["noise_removed"]),
    )
    bias_raw = json.loads((out / "bias.json").read_text(encoding="utf-8"))
    profiles = tuple(
        BiasProfile(
            judge_id=str(p["judge_id"]),
            kind=str(p["kind"]),
            family=(str(p["family"]) if p.get("family") is not None else None),
            injected=tuple((k, float(v)) for k, v in p["injected"].items()),
            detected=tuple((k, float(v)) for k, v in p["detected"].items()),
            n_probes=tuple((k, int(v)) for k, v in p["n_probes"].items()),
        )
        for p in bias_raw["profiles"]
    )
    probes = tuple(
        ProbeResult(
            axis=str(p["axis"]),
            judge_id=str(p["judge_id"]),
            n=int(p["n"]),
            hits=int(p["hits"]),
            definition=str(p["definition"]),
        )
        for p in json.loads((out / "probes.json").read_text(encoding="utf-8"))
    )
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    judges = tuple(JudgeSpec.from_dict(j) for j in summary["judges"])
    lineage = Lineage.from_dict(json.loads((out / "lineage.json").read_text(encoding="utf-8")))
    dataset = json.loads((out / "dataset.json").read_text(encoding="utf-8"))

    return PipelineResult(
        cfg={},
        version=str(version["version"]),
        created_at=str(version["created_at"]),
        prompts=tuple(Prompt.from_dict(row) for row in read_jsonl(out / "prompts.jsonl")),
        candidates=tuple(Candidate.from_dict(row) for row in read_jsonl(out / "candidates.jsonl")),
        judges=judges,
        records=records,
        profiles=profiles,
        probes=probes,
        funnel=funnel,
        noise=noise,
        manifest=dataset,
        lineage=lineage,
        expectations=dict(summary.get("expectations") or {}),
        counts=dict(version.get("counts") or {}),
    )


def materialize(cfg: Mapping[str, Any], *, root: Path | str = ".", out_dir: Path | str | None = None) -> tuple[PipelineResult, list[Path]]:
    res = run_pipeline(cfg, root=root)
    out = Path(out_dir) if out_dir is not None else Path(root) / "out"
    return res, write_outputs(res, out)
