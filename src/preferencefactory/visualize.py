"""自包含 HTML 可视化。

**不引任何外部资源**（没有 CDN、没有 webfont、没有 JS 库）—— 产出的 `.html`
双击就能看，断网也能看，拷给别人也不会变成一片空白。图表全部是内联 SVG，
在构建时算好坐标。

配色跟随浅色主题：浅底深字。
"""

from __future__ import annotations

import html
from typing import Any, Mapping, Sequence

from .report import _mean, _rho
from .schema import BIAS_AXES, PreferenceRecord

__all__ = ["build_html"]

_BG = "#ffffff"
_INK = "#1f2328"
_MUTED = "#6b7280"
_GRID = "#e5e7eb"
_GOOD = "#2f7d4f"
_WARN = "#b45309"
_BAD = "#b42318"
_BLUE = "#2f5fd0"
_PURPLE = "#6d40c4"
_TEAL = "#0f766e"

_AXIS_LABEL = {"position": "位置偏差 β", "length": "长度偏差 λ", "self": "自我偏好 σ"}


def _esc(s: Any) -> str:
    return html.escape(str(s))


def build_html(result: Any, cfg: Mapping[str, Any]) -> str:
    recs = list(result.records)
    kept = [r for r in recs if r.kept]
    n = result.noise
    fn = result.funnel

    parts: list[str] = []
    parts.append(_head(result, len(recs), len(kept)))
    parts.append(_kpis(result, n))
    parts.append(_bias_section(result))
    parts.append(_funnel_section(fn))
    parts.append(_noise_section(n))
    parts.append(_dist_section(recs))
    parts.append(_matrix_section(result))
    parts.append(_records_section(kept))
    parts.append(_foot())
    return "\n".join(parts)


def _head(result: Any, n_rec: int, n_kept: int) -> str:
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PreferenceFactory · {_esc(result.version)}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin:0; padding:32px 28px 64px; background:{_BG}; color:{_INK};
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
         font-size:14px; line-height:1.65; }}
  h1 {{ font-size:24px; margin:0 0 4px; letter-spacing:-.01em; }}
  h2 {{ font-size:17px; margin:36px 0 12px; padding-bottom:6px; border-bottom:1px solid {_GRID}; }}
  h3 {{ font-size:14px; margin:20px 0 8px; color:{_MUTED}; font-weight:600; }}
  .sub {{ color:{_MUTED}; font-size:13px; margin-bottom:20px; }}
  .kpis {{ display:flex; flex-wrap:wrap; gap:12px; margin:20px 0 8px; }}
  .kpi {{ flex:1 1 150px; border:1px solid {_GRID}; border-radius:8px; padding:12px 14px; }}
  .kpi .v {{ font-size:22px; font-weight:600; letter-spacing:-.02em; }}
  .kpi .k {{ font-size:12px; color:{_MUTED}; }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; margin:8px 0 4px; }}
  th,td {{ border:1px solid {_GRID}; padding:6px 9px; text-align:right; }}
  th {{ background:#f7f8fa; font-weight:600; text-align:right; }}
  td:first-child, th:first-child {{ text-align:left; }}
  .note {{ color:{_MUTED}; font-size:12.5px; margin:6px 0 0; }}
  .mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:12px; }}
  .ok {{ color:{_GOOD}; font-weight:600; }} .warn {{ color:{_WARN}; font-weight:600; }} .bad {{ color:{_BAD}; font-weight:600; }}
  .grid {{ display:flex; flex-wrap:wrap; gap:22px; }}
  .cell {{ flex:1 1 300px; min-width:280px; }}
</style></head><body>
<h1>PreferenceFactory</h1>
<div class="sub">版本 <span class="mono">{_esc(result.version)}</span> ｜ {_esc(result.created_at)} ｜
{n_rec} 条偏好记录 → 保留 {n_kept} 条 ｜ 核心命题：<b>偏好数据要自带可信度与来源</b></div>
"""


def _kpis(result: Any, n: Any) -> str:
    recs = list(result.records)
    cards = [
        (f"{n.noise_rate_before * 100:.1f}%", "过滤前噪声率", ""),
        (f"{n.noise_rate_after * 100:.1f}%", "过滤后噪声率", "ok"),
        (f"{n.lift:.2f}×", "过滤提升倍数（vs 随机丢）", "ok"),
        (f"{n.recall * 100:.0f}%", "噪声召回", ""),
        (f"{_mean([r.agreement for r in recs]):.3f}", "一致率均值", ""),
        (f"{_mean([r.kappa for r in recs]):.3f}", "Fleiss' κ 均值", ""),
        (f"{_rho(result, 'position'):.2f}", "位置偏差校准 ρ", "ok"),
        (f"{_rho(result, 'length'):.2f}", "长度偏差校准 ρ", "ok"),
        (f"{_rho(result, 'self'):.2f}", "自我偏好校准 ρ", "ok"),
    ]
    items = "".join(
        f'<div class="kpi"><div class="v {cls}">{_esc(v)}</div><div class="k">{_esc(k)}</div></div>'
        for v, k, cls in cards
    )
    return f'<div class="kpis">{items}</div>'


def _bars(
    rows: Sequence[tuple[str, float, float]],
    *,
    width: int = 460,
    color_a: str = _BLUE,
    color_b: str = _PURPLE,
    legend: tuple[str, str] = ("A", "B"),
) -> str:
    """两列并排的横向条形图。"""
    row_h = 26
    label_w = 96
    val_w = 96
    plot_w = width - label_w - val_w
    h = row_h * len(rows) + 30
    out = [
        f'<svg viewBox="0 0 {width} {h}" width="100%" height="{h}" role="img">',
        f'<line x1="{label_w}" y1="4" x2="{label_w}" y2="{h - 26}" stroke="{_GRID}"/>',
    ]
    for i, (name, a, b) in enumerate(rows):
        y = 8 + i * row_h
        wa = max(0.0, min(1.0, a)) * (plot_w - 8)
        wb = max(0.0, min(1.0, b)) * (plot_w - 8)
        out.append(
            f'<text x="{label_w - 8}" y="{y + 15}" text-anchor="end" font-size="12" fill="{_INK}">{_esc(name)}</text>'
        )
        out.append(f'<rect x="{label_w + 2}" y="{y + 3}" width="{wa:.1f}" height="8" fill="{color_a}" rx="2"/>')
        out.append(f'<rect x="{label_w + 2}" y="{y + 13}" width="{wb:.1f}" height="8" fill="{color_b}" rx="2"/>')
        out.append(
            f'<text x="{width - 4}" y="{y + 15}" text-anchor="end" font-size="11" fill="{_MUTED}">'
            f"{a:.3f} / {b:.3f}</text>"
        )
    lx = label_w + 2
    out.append(f'<rect x="{lx}" y="{h - 18}" width="9" height="8" fill="{color_a}" rx="2"/>')
    out.append(f'<text x="{lx + 14}" y="{h - 11}" font-size="11" fill="{_MUTED}">{_esc(legend[0])}</text>')
    out.append(f'<rect x="{lx + 90}" y="{h - 18}" width="9" height="8" fill="{color_b}" rx="2"/>')
    out.append(f'<text x="{lx + 104}" y="{h - 11}" font-size="11" fill="{_MUTED}">{_esc(legend[1])}</text>')
    out.append("</svg>")
    return "".join(out)


def _bias_section(result: Any) -> str:
    out = ["<h2>偏差检测：注入量 vs 检出率</h2>",
           '<p class="note">每一组的两条横线：上＝我们<b>注入</b>的量，下＝探针<b>检出</b>的率。'
           '两者量纲不同（一个是打分偏移量、一个是踩中率），所以校准看的是<b>次序</b>而不是相等。</p>']
    for axis in BIAS_AXES:
        rows = []
        for p in sorted(result.profiles, key=lambda q: -q.injected_on(axis)):
            if dict(p.n_probes).get(axis, 0) == 0:
                continue
            rows.append((p.judge_id, p.injected_on(axis), p.detected_on(axis)))
        out.append(f"<h3>{_AXIS_LABEL[axis]}　Spearman ρ = {_rho(result, axis):.3f}</h3>")
        out.append(_bars(rows, legend=("注入量", "检出率")))
    out.append(
        '<p class="note"><b>不冤枉好人</b>是最关键的一条：<code>rule</code> 与 <code>human</code> '
        '的三项偏差构造上为 0，探针也不该测出东西。测出来就是检测器本身有问题。</p>'
    )
    return "".join(out)


def _hist(xs: Sequence[float], bins: int = 10, *, width: int = 300, height: int = 130) -> str:
    counts = [0] * bins
    for x in xs:
        i = int(max(0.0, min(0.999, x)) * bins)
        counts[i] += 1
    mx = max(counts) or 1
    bw = width / bins
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img">']
    for i, c in enumerate(counts):
        hgt = (c / mx) * (height - 26)
        out.append(
            f'<rect x="{i * bw + 1:.1f}" y="{height - 20 - hgt:.1f}" width="{bw - 2:.1f}" '
            f'height="{hgt:.1f}" fill="{_BLUE}" rx="2"/>'
        )
    for i in range(bins):
        out.append(
            f'<text x="{i * bw + bw / 2:.1f}" y="{height - 6}" text-anchor="middle" font-size="9" '
            f'fill="{_MUTED}">{i / bins:.1f}</text>'
        )
    out.append("</svg>")
    return "".join(out)


def _funnel_section(fn: Any) -> str:
    fc = dict(fn.first_cause)
    ind = dict(fn.independent)
    rows = "".join(
        f"<tr><td><code>{_esc(r)}</code></td><td>{fc.get(r, 0)}</td><td>{ind.get(r, 0)}</td></tr>"
        for r in ("agreement", "confidence", "rule_agreement", "untainted_support")
    )
    width = 420
    h = 20
    total = max(1, fn.total)
    out = [f'<svg viewBox="0 0 {width} 34" width="100%" height="34" role="img">']
    out.append(f'<rect x="0" y="6" width="{width * fn.kept / total:.1f}" height="{h}" fill="{_GOOD}" rx="3"/>')
    out.append(
        f'<rect x="{width * fn.kept / total:.1f}" y="6" width="{width * (1 - fn.kept / total):.1f}" '
        f'height="{h}" fill="{_GRID}" rx="3"/>'
    )
    out.append(
        f'<text x="{width / 2:.0f}" y="20" text-anchor="middle" font-size="12" fill="#ffffff">'
        f"保留 {fn.kept} / {fn.total}（{fn.kept / total * 100:.0f}%）</text>"
    )
    out.append("</svg>")
    return (
        "<h2>过滤漏斗</h2>"
        + "".join(out)
        + "<table><tr><th>规则</th><th>首因归因</th><th>独立失败数</th></tr>"
        + rows
        + "</table>"
        + '<p class="note">首因归因＝每条被丢的记录只算在第一个卡住它的规则名下；'
        "独立失败数＝每条规则各自拦下多少。两个都要：前者回答「主要被谁卡住」，后者回答「谁最有杀伤力」。</p>"
    )


def _noise_section(n: Any) -> str:
    cap = n.noise_total / n.n_removed if n.n_removed else 0.0
    return (
        "<h2>噪声账本</h2>"
        "<table>"
        "<tr><th>项</th><th>数值</th></tr>"
        f"<tr><td>记录总数</td><td>{n.n_total}</td></tr>"
        f"<tr><td>其中噪声</td><td>{n.noise_total}（{n.noise_rate_before * 100:.1f}%）</td></tr>"
        f"<tr><td>保留</td><td>{n.n_kept}</td></tr>"
        f"<tr><td>保留里还剩噪声</td><td>{n.noise_kept}（{n.noise_rate_after * 100:.1f}%）</td></tr>"
        f"<tr><td>丢掉</td><td>{n.n_removed}</td></tr>"
        f"<tr><td>丢掉的里真是噪声</td><td>{n.noise_removed}（精度 {n.precision * 100:.1f}%）</td></tr>"
        f"<tr><td>噪声召回</td><td>{n.recall * 100:.1f}%</td></tr>"
        f"<tr><td><b>提升倍数</b></td><td class='ok'><b>{n.lift:.2f}×</b></td></tr>"
        "</table>"
        f'<p class="note">精度的<b>上限</b>是 噪声总数/丢掉数 = {n.noise_total}/{n.n_removed} = {cap:.3f}，'
        "越不过去。所以只看精度会得出「过滤器一般」的错误印象 —— 提升倍数才回答「比随机丢好多少」。</p>"
    )


def _dist_section(recs: Sequence[PreferenceRecord]) -> str:
    return (
        "<h2>一致率 / 置信度分布</h2>"
        '<div class="grid">'
        f'<div class="cell"><h3>一致率（均值 {_mean([r.agreement for r in recs]):.3f}）</h3>'
        f"{_hist([r.agreement for r in recs])}</div>"
        f'<div class="cell"><h3>置信度（均值 {_mean([r.confidence for r in recs]):.3f}）</h3>'
        f"{_hist([r.confidence for r in recs])}</div>"
        f'<div class="cell"><h3>Fleiss\' κ（均值 {_mean([r.kappa for r in recs]):.3f}）</h3>'
        f"{_hist([max(0.0, min(1.0, r.kappa)) for r in recs])}</div>"
        f'<div class="cell"><h3>真值差 |Δq|（均值 {_mean([r.truth_margin for r in recs]):.3f}）</h3>'
        f"{_hist([min(1.0, r.truth_margin) for r in recs])}</div>"
        "</div>"
        '<p class="note">κ ≤ 一致率是代数必然（κ=(P̄−Pe)/(1−Pe)，而 Pe(1−P̄) ≥ 0），'
        "所以 κ 低不代表算错，它代表「扣掉乱猜也能撞上的部分之后，剩下的共识不多」。</p>"
    )


def _matrix_section(result: Any) -> str:
    js = [j.id for j in result.judges]
    cells: dict[tuple[str, str], float] = {}
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
            cells[(js[i], js[k])] = agree / cnt if cnt else 0.0

    def cell(a: str, b: str) -> str:
        if a == b:
            return '<td style="background:#f7f8fa">—</td>'
        v = cells.get((a, b), cells.get((b, a), 0.0))
        alpha = 0.10 + 0.55 * v
        return f'<td style="background:rgba(47,95,208,{alpha:.2f})">{v:.2f}</td>'

    head = "".join(f"<th>{_esc(j)}</th>" for j in js)
    body = "".join(
        f"<tr><th>{_esc(a)}</th>" + "".join(cell(a, b) for b in js) + "</tr>" for a in js
    )
    return (
        "<h2>裁判两两一致率</h2>"
        f"<table><tr><th>&nbsp;</th>{head}</tr>{body}</table>"
        '<p class="note">颜色越深越一致。「谁跟谁老是不对付」在这张表上一眼可见 —— '
        "这也是「多裁判」比「一个更好的裁判」强的地方：<b>分歧本身就是信号</b>。</p>"
    )


def _records_section(kept: Sequence[PreferenceRecord]) -> str:
    rows = []
    for r in sorted(kept, key=lambda x: -x.confidence)[:12]:
        ch = (r.chosen or "")[:34]
        rj = (r.rejected or "")[:34]
        rows.append(
            f"<tr><td class='mono'>{_esc(r.id)}</td><td>{_esc(r.question[:30])}</td>"
            f"<td class='mono'>{_esc(ch)}</td><td class='mono'>{_esc(rj)}</td>"
            f"<td>{r.agreement:.2f}</td><td>{r.confidence:.2f}</td>"
            f"<td>{_esc('/'.join(v.choice for v in r.verdicts))}</td></tr>"
        )
    return (
        "<h2>置信度最高的若干条（导出形态）</h2>"
        "<table><tr><th>记录</th><th>prompt</th><th>chosen</th><th>rejected</th>"
        "<th>agreement</th><th>confidence</th><th>各裁判结论</th></tr>"
        + "".join(rows)
        + "</table>"
        '<p class="note">这里已经是<b>剥掉隐藏真值</b>之后的形态 —— 下游真正能吃到的就是这些字段。'
        "真值留在 <code>records.jsonl</code> 里，只用于评估过滤器。</p>"
    )


def _foot() -> str:
    return "</body></html>\n"
