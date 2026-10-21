"""命令行。

两条纪律：

1. **`--json` 只输出 JSON。** 之前的版本是「说明文字 + 一段 JSON」，于是
   `main.py zoo --json | jq` 这种用法直接坏掉。现在用 `_quiet_prose` 把命令自己的
   `print` 吞掉，`_emit_json` 直接写真正的 `sys.stdout`。
2. **只读命令从产物重建，不重跑。** 产物缺失时 `load_result` 返回 `None`，
   这时候才在内存里跑一次，并明确提示「产物不存在」—— 不能静默重跑，
   否则用户以为自己看的是磁盘上的东西。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .acceptance import acceptance_items, adversarial_items, summarize
from .config import load_config, objectives_of, protocol_digest
from .dataset import export_record
from .pipeline import OUT_FILES, load_result, materialize, run_pipeline, write_outputs
from .report import build_report
from .visualize import build_html

__all__ = ["build_parser", "main"]

_REAL_STDOUT = sys.stdout


def _emit_json(payload: Any) -> None:
    """只写 JSON。**走真正的 stdout**，因为此时 `sys.stdout` 已经被换成 StringIO 了。"""
    _REAL_STDOUT.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    _REAL_STDOUT.flush()


class _quiet_prose:
    """把命令自己 print 的说明文字吞掉，只留 `_emit_json` 的输出。"""

    def __enter__(self) -> "_quiet_prose":
        import io

        self._saved = sys.stdout
        sys.stdout = io.StringIO()
        return self

    def __exit__(self, *exc: object) -> bool:
        sys.stdout = self._saved
        return False


def _row(cells: Sequence[Any], widths: Sequence[int]) -> str:
    out = []
    for c, w in zip(cells, widths):
        s = str(c)
        out.append(s if len(s) >= w else s + " " * (w - len(s)))
    return "  ".join(out).rstrip()


# --------------------------------------------------------------------------
# 结果获取
# --------------------------------------------------------------------------

def _root_of(args: argparse.Namespace) -> Path:
    return Path(getattr(args, "root", ".") or ".").resolve()


def _out_of(args: argparse.Namespace) -> Path:
    return Path(getattr(args, "out", None) or (_root_of(args) / "out")).resolve()


def _result(args: argparse.Namespace) -> Any:
    """优先读产物；产物不全就在内存里跑一次（并提示）。"""
    out = _out_of(args)
    res = load_result(out)
    if res is not None:
        return res
    print(f"[提示] {out} 下产物不完整，本次在内存中跑一遍（先跑 `run` 可落盘）。", file=sys.stderr)
    return run_pipeline(load_config(getattr(args, "config", None), root=_root_of(args)), root=_root_of(args))


def _cfg_of(args: argparse.Namespace) -> Mapping[str, Any]:
    return load_config(getattr(args, "config", None), root=_root_of(args))


# --------------------------------------------------------------------------
# 子命令
# --------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    cfg = _cfg_of(args)
    res, written = materialize(cfg, root=_root_of(args), out_dir=_out_of(args))
    print(f"版本 {res.version}")
    c = dict(res.counts)
    print(
        f"{c.get('prompts', 0)} 个 prompt / {c.get('candidates', 0)} 个候选 / "
        f"{c.get('records', 0)} 条记录 / 保留 {c.get('kept', 0)} 条 / {c.get('judges', 0)} 个裁判"
    )
    print(f"噪声率 {res.noise.noise_rate_before:.3f} → {res.noise.noise_rate_after:.3f}"
          f"（提升 {res.noise.lift:.2f}×）")
    print(f"写入 {len(written)} 个文件 → {_out_of(args)}")
    return 0


def cmd_visualize(args: argparse.Namespace) -> int:
    res = _result(args)
    out = _out_of(args)
    p = out / "report.html"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(build_html(res, _cfg_of(args)), encoding="utf-8", newline="\n")
    print(f"写入 {p}（自包含，无外部资源）")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    res = _result(args)
    out = _out_of(args)
    p = out / "report.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(build_report(res, _cfg_of(args)), encoding="utf-8", newline="\n")
    print(f"写入 {p}（{p.stat().st_size} 字节）")
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    res = _result(args)
    n = res.noise
    payload = {
        "version": res.version,
        "created_at": res.created_at,
        "counts": dict(res.counts),
        "noise": n.to_dict(),
        "funnel": res.funnel.to_dict(),
        "objectives": list(objectives_of(_cfg_of(args))),
        "protocol_digest": protocol_digest(_cfg_of(args))[:12] if res.cfg else None,
    }
    if args.json:
        _emit_json(payload)
        return 0
    print(f"PreferenceFactory {res.version}（{res.created_at}）")
    c = dict(res.counts)
    print(f"  prompt {c.get('prompts', 0)} ｜ 候选 {c.get('candidates', 0)} ｜ 记录 {c.get('records', 0)}"
          f" ｜ 保留 {c.get('kept', 0)} ｜ 裁判 {c.get('judges', 0)}")
    print(f"  噪声率 {n.noise_rate_before:.3f} → {n.noise_rate_after:.3f}"
          f" ｜ 精度 {n.precision:.3f} ｜ 提升 {n.lift:.2f}× ｜ 召回 {n.recall:.3f}")
    print(f"  优化目标：{list(objectives_of(_cfg_of(args))) or '无（本项目只过滤，不挑选）'}")
    return 0


def cmd_judges(args: argparse.Namespace) -> int:
    res = _result(args)
    rows = [j.to_dict() for j in res.judges]
    if args.json:
        _emit_json(rows)
        return 0
    print(_row(["裁判", "类型", "归属族", "β位置", "λ长度", "σ自我", "噪声"], [12, 6, 10, 7, 7, 7, 6]))
    for j in res.judges:
        print(_row([j.id, j.kind, j.family or "—", f"{j.beta:.2f}", f"{j.lam:.2f}",
                    f"{j.self_bias:.2f}", f"{j.noise:.2f}"], [12, 6, 10, 7, 7, 7, 6]))
    print()
    print("rule / human 的三项偏差恒为 0：它们是锚点与对照组，是**构造上**无偏，不是估计出来没偏。")
    return 0


def cmd_bias(args: argparse.Namespace) -> int:
    res = _result(args)
    payload = {p.judge_id: p.to_dict() for p in res.profiles}
    if args.json:
        _emit_json(payload)
        return 0
    print("偏差档案：**注入量**（我们亲手设的）与**检出率**（探针测的）并排。")
    print()
    print(_row(["裁判", "轴", "注入", "检出", "样本"], [12, 10, 8, 8, 6]))
    for p in res.profiles:
        for axis in ("position", "length", "self"):
            cnt = dict(p.n_probes).get(axis, 0)
            det = "—" if cnt == 0 else f"{p.detected_on(axis):.3f}"
            print(_row([p.judge_id, axis, f"{p.injected_on(axis):.2f}", det, cnt], [12, 10, 8, 8, 6]))
    return 0


def cmd_probes(args: argparse.Namespace) -> int:
    res = _result(args)
    rows = [p.to_dict() for p in res.probes]
    if args.json:
        _emit_json(rows)
        return 0
    print(_row(["轴", "裁判", "样本", "踩中", "率", "口径"], [10, 12, 6, 6, 8, 40]))
    for p in res.probes:
        print(_row([p.axis, p.judge_id, p.n, p.hits, f"{p.rate:.3f}", p.definition[:40]],
                   [10, 12, 6, 6, 8, 40]))
    return 0


def cmd_agreement(args: argparse.Namespace) -> int:
    res = _result(args)
    recs = list(res.records)
    payload = {
        "n_records": len(recs),
        "agreement_mean": sum(r.agreement for r in recs) / len(recs) if recs else 0.0,
        "kappa_mean": sum(r.kappa for r in recs) / len(recs) if recs else 0.0,
        "confidence_mean": sum(r.confidence for r in recs) / len(recs) if recs else 0.0,
        "per_record": [{"id": r.id, "agreement": r.agreement, "kappa": r.kappa,
                        "confidence": r.confidence, "verdict": r.verdict} for r in recs],
    }
    if args.json:
        _emit_json(payload)
        return 0
    print(f"{len(recs)} 条记录：一致率均值 {payload['agreement_mean']:.3f} ｜ "
          f"κ 均值 {payload['kappa_mean']:.3f} ｜ 置信度均值 {payload['confidence_mean']:.3f}")
    print()
    print(_row(["记录", "一致率", "κ", "置信度", "结论"], [16, 8, 8, 8, 8]))
    for r in sorted(recs, key=lambda x: -x.confidence)[: int(args.top)]:
        print(_row([r.id, f"{r.agreement:.3f}", f"{r.kappa:.3f}", f"{r.confidence:.3f}", r.verdict],
                   [16, 8, 8, 8, 8]))
    return 0


def cmd_funnel(args: argparse.Namespace) -> int:
    res = _result(args)
    payload = {"funnel": res.funnel.to_dict(), "noise": res.noise.to_dict()}
    if args.json:
        _emit_json(payload)
        return 0
    print(_row(["规则", "首因归因", "独立失败数"], [22, 10, 10]))
    fc = dict(res.funnel.first_cause)
    ind = dict(res.funnel.independent)
    for r in ("agreement", "confidence", "rule_agreement", "untainted_support"):
        print(_row([r, fc.get(r, 0), ind.get(r, 0)], [22, 10, 10]))
    print(f"\n保留 {res.funnel.kept} / {res.funnel.total}")
    return 0


def cmd_noise(args: argparse.Namespace) -> int:
    res = _result(args)
    n = res.noise
    if args.json:
        _emit_json(n.to_dict())
        return 0
    print("「过滤掉了多少噪声样本」的完整账本")
    print()
    print(f"  记录总数            {n.n_total}")
    print(f"  其中噪声            {n.noise_total}（{n.noise_rate_before:.1%}）")
    print(f"  保留                {n.n_kept}（剩噪声 {n.noise_kept}，{n.noise_rate_after:.1%}）")
    print(f"  丢掉                {n.n_removed}（其中真是噪声 {n.noise_removed}）")
    print(f"  精度（丢的中真噪声）{n.precision:.3f}")
    print(f"  召回（噪声抓到多少）{n.recall:.3f}")
    print(f"  提升倍数            {n.lift:.2f}×（随机丢的命中率 = 基础噪声率 {n.noise_rate_before:.3f}）")
    return 0


def cmd_records(args: argparse.Namespace) -> int:
    res = _result(args)
    recs = list(res.records)
    if args.all:
        rows = [r.to_dict() for r in recs]
    else:
        rows = [{"id": r.id, "prompt": r.question, "verdict": r.verdict,
                 "agreement": r.agreement, "confidence": r.confidence, "kept": r.kept,
                 "noise": r.is_noise} for r in recs]
    if args.json:
        _emit_json(rows)
        return 0
    print(_row(["记录", "题型", "结论", "一致率", "置信度", "保留", "噪声"], [16, 8, 6, 8, 8, 6, 6]))
    for r in recs[: int(args.top)]:
        print(_row([r.id, r.prompt_kind, r.verdict, f"{r.agreement:.3f}", f"{r.confidence:.3f}",
                    "✓" if r.kept else "✗", "是" if r.is_noise else ""], [16, 8, 6, 8, 8, 6, 6]))
    if len(recs) > int(args.top):
        print(f"… 还有 {len(recs) - int(args.top)} 条（用 --top 调整）")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    res = _result(args)
    rec = res.record(args.record_id)
    if rec is None:
        print(f"没有这条记录：{args.record_id}", file=sys.stderr)
        return 2
    if args.json:
        _emit_json(rec.to_dict())
        return 0
    print(f"记录 {rec.id}（{rec.prompt_kind}）")
    print(f"  prompt   {rec.question}")
    for tag, c in (("A", rec.a), ("B", rec.b)):
        print(f"  候选 {tag}   [{c.family}] {c.text!r}")
        print(f"          判据分={c.criteria:.3f} 真值={c.quality:.3f} 长度={c.length} 填充={c.pad_chars}")
    print(f"  ranking  {list(rec.ranking)}")
    print(f"  真值胜者 {rec.truth_winner}（差 {rec.truth_margin:.3f}）→ 这条{'是' if rec.is_noise else '不是'}噪声")
    print()
    print("  判定：")
    for v in rec.verdicts:
        f = {k: round(x, 3) for k, x in v.factors}
        print(f"    {v.judge_id:10s} → {v.choice:4s}  A={v.score_a:.3f} B={v.score_b:.3f}  {f}")
    print()
    print(f"  一致率 {rec.agreement:.3f} ｜ κ {rec.kappa:.3f} ｜ 置信度 {rec.confidence:.3f}")
    print(f"  结论   {rec.verdict}（降权票 {rec.weighted_verdict}）")
    print(f"  理由   {rec.rationale}")
    print()
    print("  过滤证据：")
    for e in rec.evidence:
        print(f"    [{'通过' if e.passed else '丢弃'}] {e.rule}: {e.detail}")
    print(f"  → {'保留' if rec.kept else '丢弃'}")
    return 0


def cmd_preferences(args: argparse.Namespace) -> int:
    res = _result(args)
    rows = [export_record(r) for r in res.kept]
    if args.json:
        _emit_json(rows)
        return 0
    print(f"导出形态 {len(rows)} 条（**不含隐藏真值**）")
    print()
    print(_row(["记录", "chosen", "rejected", "置信度"], [16, 30, 30, 8]))
    for r in rows[: int(args.top)]:
        print(_row([r["id"], (r["chosen"] or "")[:29], (r["rejected"] or "")[:29],
                    f"{r['confidence']:.3f}"], [16, 30, 30, 8]))
    return 0


def cmd_dataset(args: argparse.Namespace) -> int:
    res = _result(args)
    m = res.manifest
    if args.json:
        _emit_json(m.to_dict() if hasattr(m, "to_dict") else m)
        return 0
    d = m.to_dict() if hasattr(m, "to_dict") else m
    print(f"数据集版本 {d['version']}（digest {d['digest'][:16]}…）")
    print(f"  {d['n_records']} 条保留 / 丢掉 {d['n_dropped']} 条")
    for k, v in d["stats"].items():
        print(f"  {k}: {v}")
    return 0


def cmd_lineage(args: argparse.Namespace) -> int:
    res = _result(args)
    g = res.lineage
    if args.json:
        _emit_json(g.to_dict())
        return 0
    print(f"血缘图：{len(g.nodes)} 个节点 / {len(g.edges)} 条边")
    if args.dot:
        print(g.to_dot())
        return 0
    for k in ("prompt", "candidate", "judge", "record"):
        ns = [n for n in g.nodes if n.kind == k]
        print(f"  {k:10s} {len(ns)}")
    print("\n用 --dot 输出 Graphviz；`out/lineage.dot` 里也有一份。")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    cfg = _cfg_of(args)
    if args.json:
        _emit_json({k: v for k, v in cfg.items() if not k.startswith("_")})
        return 0
    print(f"配置来源 {cfg.get('_config_path')}")
    print(f"protocol_digest {protocol_digest(cfg)[:16]}…")
    print(f"优化目标 {list(objectives_of(cfg)) or '无'}")
    return 0


def _print_items(items: Sequence[Mapping[str, Any]], title: str) -> int:
    print(title)
    print()
    for i in items:
        print(f"  [{'PASS' if i['ok'] else 'FAIL'}] {i['name']}")
        print(f"         {i['detail']}")
    ok, tot = summarize(items)
    print()
    print(f"{ok}/{tot} 通过")
    return 0 if ok == tot else 1


def cmd_acceptance(args: argparse.Namespace) -> int:
    cfg = _cfg_of(args)
    res = _result(args)
    items = acceptance_items(res, cfg, _out_of(args))
    if args.json:
        _emit_json({"items": list(items), "passed": summarize(items)[0], "total": len(items)})
        return 0 if summarize(items)[0] == len(items) else 1
    return _print_items(items, "验收清单（依赖产物）")


def cmd_adversarial(args: argparse.Namespace) -> int:
    cfg = _cfg_of(args)
    items = adversarial_items(cfg)
    if args.json:
        _emit_json({"items": list(items), "passed": summarize(items)[0], "total": len(items)})
        return 0 if summarize(items)[0] == len(items) else 1
    return _print_items(items, "对抗用例（不依赖产物，产物过期也能跑）")


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="preferencefactory",
        description="PreferenceFactory —— 偏好证据的产生、验证、去噪与版本化",
    )
    p.add_argument("--root", default=".", help="项目根目录（默认当前目录）")
    p.add_argument("--out", default=None, help="产物目录（默认 <root>/out）")
    p.add_argument("--config", default=None, help="配置文件（默认 <root>/configs/default.json）")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name: str, help_: str, func: Callable[[argparse.Namespace], int]) -> argparse.ArgumentParser:
        s = sub.add_parser(name, help=help_)
        s.set_defaults(func=func)
        return s

    add("run", "跑整条流水线并落盘（写）", cmd_run)
    add("visualize", "生成自包含 HTML 报告（写）", cmd_visualize)
    add("report", "生成 Markdown 报告（写）", cmd_report)

    add("summary", "总览", cmd_summary).add_argument("--json", action="store_true")
    add("judges", "裁判池与注入偏差", cmd_judges).add_argument("--json", action="store_true")
    add("bias", "偏差档案：注入 vs 检出", cmd_bias).add_argument("--json", action="store_true")
    add("probes", "探针逐条结果", cmd_probes).add_argument("--json", action="store_true")

    s = add("agreement", "一致率 / κ / 置信度", cmd_agreement)
    s.add_argument("--json", action="store_true")
    s.add_argument("--top", type=int, default=15)

    s = add("funnel", "过滤漏斗", cmd_funnel)
    s.add_argument("--json", action="store_true")

    s = add("noise", "噪声账本（验收标准②）", cmd_noise)
    s.add_argument("--json", action="store_true")

    s = add("records", "记录列表", cmd_records)
    s.add_argument("--json", action="store_true")
    s.add_argument("--all", action="store_true", help="输出完整记录（含真值）")
    s.add_argument("--top", type=int, default=20)

    s = add("show", "看一条记录的完整证据", cmd_show)
    s.add_argument("record_id")
    s.add_argument("--json", action="store_true")

    s = add("preferences", "导出形态（不含真值）", cmd_preferences)
    s.add_argument("--json", action="store_true")
    s.add_argument("--top", type=int, default=15)

    s = add("dataset", "数据集版本与统计", cmd_dataset)
    s.add_argument("--json", action="store_true")

    s = add("lineage", "血缘图", cmd_lineage)
    s.add_argument("--json", action="store_true")
    s.add_argument("--dot", action="store_true")

    s = add("config", "当前配置口径", cmd_config)
    s.add_argument("--json", action="store_true")

    s = add("acceptance", "验收清单", cmd_acceptance)
    s.add_argument("--json", action="store_true")

    s = add("adversarial", "对抗用例", cmd_adversarial)
    s.add_argument("--json", action="store_true")

    return p


def _harden_stdio() -> None:
    """让 stdout / stderr 不会因为一个字符就崩掉。

    Windows 下输出被重定向（管道、写文件）时编码是 GBK，报告里的 ``κ̄`` 带组合
    字符 U+0304，直接 ``UnicodeEncodeError``。终端里 Python 走宽字符 API 不受
    影响，所以只在非 tty 时切到 UTF-8，tty 只放宽错误处理。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            if stream.isatty():
                reconfigure(errors="replace")
            else:
                reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def main(argv: list[str] | None = None) -> int:
    _harden_stdio()
    args = build_parser().parse_args(argv)
    try:
        with _quiet_prose() if getattr(args, "json", False) else _null():
            return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover
        return 130


class _null:
    def __enter__(self) -> "_null":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False
