"""版本化导出：把过滤器放行的内容压成一份**可复现、可引用**的偏好数据集。

两条纪律：

1. **内容寻址版本**：`v1-<digest 前 12 位>`，digest 吃的是被选中记录的内容摘要。
   于是「版本号相同 ⇒ 数据集逐位相同」，而「版本号不同 ⇒ 确实换了东西」。
   唯一不可复现的字段是 `created_at`，它不进 digest。
2. **导出时剥掉隐藏真值。** `chosen/rejected` 是给下游训练用的，真值是给本项目
   评估过滤器用的。两者混在一起，下游就会拿真值当特征喂进去 —— 那种 bug 在
   评测指标上表现为「好得不像话」，很难查。真值留在 `records.jsonl` 里。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .rng import canonical_json, join_digest, sha256_text
from .schema import PreferenceRecord

__all__ = [
    "DatasetManifest",
    "content_digest",
    "version_id",
    "export_record",
    "build_manifest",
    "write_jsonl",
    "read_jsonl",
]

_TRUTH_KEYS = ("latent", "quality", "satisfies", "checks_passed")


@dataclass(frozen=True)
class DatasetManifest:
    version: str
    digest: str
    n_records: int
    n_dropped: int
    stats: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "digest": self.digest,
            "n_records": self.n_records,
            "n_dropped": self.n_dropped,
            "stats": dict(self.stats),
        }


def content_digest(records: Sequence[PreferenceRecord]) -> str:
    """数据集内容摘要。**只吃导出形态**（剥掉真值之后的样子）。

    如果吃的是带真值的形态，那么「真值变了但导出没变」也会换版本号 —— 而那对
    下游来说是同一份数据集。
    """
    parts = [sha256_text(canonical_json(export_record(r))) for r in records]
    return join_digest(parts) if parts else sha256_text("empty")


def version_id(digest: str, prefix: str = "v1") -> str:
    return f"{prefix}-{digest[:12]}"


def export_record(rec: PreferenceRecord) -> dict[str, Any]:
    """导出形态：下游真正能吃的那一份。**不含隐藏真值。**"""
    chosen = rec.chosen
    rejected = rec.rejected
    return {
        "id": rec.id,
        "prompt": rec.question,
        "chosen": chosen,
        "rejected": rejected,
        "agreement": round(rec.agreement, 6),
        "kappa": round(rec.kappa, 6),
        "confidence": round(rec.confidence, 6),
        "judges": list(rec.judges),
        "judge_choices": {v.judge_id: v.choice for v in rec.verdicts},
        "rationale": rec.rationale,
        "provenance": {
            "prompt_id": rec.prompt_id,
            "prompt_kind": rec.prompt_kind,
            "candidate_families": [c.family for c in rec.candidates],
            "spec_digest": rec.provenance.get("spec_digest", ""),
        },
    }


def build_manifest(
    kept: Sequence[PreferenceRecord],
    all_records: Sequence[PreferenceRecord],
    *,
    extra: Mapping[str, Any] | None = None,
) -> DatasetManifest:
    digest = content_digest(kept)
    stats: dict[str, Any] = {
        "by_prompt_kind": _count_by(kept, lambda r: r.prompt_kind),
        "by_verdict": _count_by(kept, lambda r: r.verdict),
        "by_family_pair": _count_by(kept, lambda r: "|".join(c.family for c in r.candidates)),
        "agreement_mean": _mean([r.agreement for r in kept]),
        "confidence_mean": _mean([r.confidence for r in kept]),
        "kappa_mean": _mean([r.kappa for r in kept]),
        "chosen_len_mean": _mean([float(len(r.chosen)) for r in kept if r.chosen is not None]),
        "rejected_len_mean": _mean([float(len(r.rejected)) for r in kept if r.rejected is not None]),
    }
    if extra:
        stats.update(extra)
    return DatasetManifest(
        version=version_id(digest),
        digest=digest,
        n_records=len(kept),
        n_dropped=len(all_records) - len(kept),
        stats=stats,
    )


def _count_by(records: Iterable[PreferenceRecord], key) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in records:
        k = str(key(r))
        out[k] = out.get(k, 0) + 1
    return {k: v for k, v in sorted(out.items())}


def _mean(xs: Sequence[float]) -> float:
    return round(sum(xs) / len(xs), 6) if xs else 0.0


def write_jsonl(path, rows: Iterable[Mapping[str, Any]]) -> int:
    import json
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def read_jsonl(path) -> list[dict[str, Any]]:
    import json
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
