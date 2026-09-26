"""Category coverage and deterministic manual-check sampling for G1.

:func:`coverage_report` summarizes a build's reaction-change summary rows: the
overall valid/rejected split, the failure-code histogram, and one
``{n_total, n_valid, coverage}`` entry per category (the seven boolean document
categories plus the index- and pairing-ambiguity categories).  Ambiguity is
also reported as explicit status histograms, so an ambiguous or truncated
index can never hide inside a single count.

:func:`manual_sample` picks, for every non-empty category, the ``n`` valid
reactions with the smallest ``sha256(f"{seed}:{reaction_id}")`` ranking and
returns one flat, sorted ``{reaction_id, category}`` entry per pick.  The
ranking is a pure function of seed and reaction id, so the same build always
yields the same human-check list.  :func:`write_coverage` stamps the report
with the shared G1 manifest schema version and a volatile generation timestamp.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from pes2ts_core.g1 import G1_MANIFEST_SCHEMA_VERSION
from pes2ts_core.g1.document import CATEGORY_NAMES
from pes2ts_core.utils.hashing import JSONValue, sha256_bytes
from pes2ts_core.utils.jsonio import write_json

#: Ambiguity categories derived from the summary status columns.
AMBIGUITY_CATEGORIES: Final[tuple[str, ...]] = ("index_ambiguous", "pairing_ambiguous")
#: Every category the coverage report and the manual sample cover.
SAMPLE_CATEGORIES: Final[tuple[str, ...]] = (*CATEGORY_NAMES, *AMBIGUITY_CATEGORIES)
#: Seed used when the configuration carries neither ``g1.sample_seed`` nor ``split.seed``.
DEFAULT_SEED: Final[int] = 42
#: Manual-check sample size used when the configuration carries none.
DEFAULT_SAMPLE_SIZE: Final[int] = 20


def _now() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def _settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the ``g1`` configuration block (or an empty mapping)."""
    raw = config.get("g1")
    settings: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
    return settings


def _in_category(row: Mapping[str, Any], category: str) -> bool:
    """Return whether one summary row belongs to *category*."""
    if category == "index_ambiguous":
        return str(row["index_status"]) in {"ambiguous", "truncated"}
    if category == "pairing_ambiguous":
        return str(row["pairing_status"]) == "ambiguous"
    return bool(row[category])


def _coverage(n_total: int, n_valid: int) -> float:
    """Return the valid fraction of *n_total*, rounded to four decimals."""
    return round(n_valid / n_total, 4) if n_total else 0.0


def coverage_report(summary_rows: Sequence[Mapping[str, Any]]) -> dict[str, JSONValue]:
    """Summarize *summary_rows* by outcome, failure code, and category.

    Coverage is the valid fraction of each category (rejected rows can never
    belong to a category, but they still count toward ``n_total`` when the
    category key is a status-derived one), and the status histograms count
    every row, so ``index_ambiguous`` plus ``index_truncated`` reconciles with
    the ``truncated``/``ambiguous`` histogram entries.
    """
    valid = [row for row in summary_rows if str(row["status"]) == "valid"]
    by_code = Counter(
        str(row["failure_code"]) for row in summary_rows if row["failure_code"]
    )
    categories: dict[str, JSONValue] = {}
    for category in SAMPLE_CATEGORIES:
        matching = [row for row in summary_rows if _in_category(row, category)]
        matching_valid = [row for row in valid if _in_category(row, category)]
        entry: dict[str, JSONValue] = {
            "n_total": len(matching),
            "n_valid": len(matching_valid),
            "coverage": _coverage(len(matching), len(matching_valid)),
        }
        categories[category] = entry
    index_status: dict[str, JSONValue] = {
        name: sum(1 for row in summary_rows if str(row["index_status"]) == name)
        for name in ("unique", "ambiguous", "truncated")
    }
    pairing_status: dict[str, JSONValue] = {
        name: sum(1 for row in summary_rows if str(row["pairing_status"]) == name)
        for name in ("unique", "ambiguous")
    }
    return {
        "overall": {
            "n_total": len(summary_rows),
            "n_valid": len(valid),
            "n_rejected": len(summary_rows) - len(valid),
            "coverage": _coverage(len(summary_rows), len(valid)),
        },
        "by_code": {code: by_code[code] for code in sorted(by_code)},
        "categories": categories,
        "index_status": index_status,
        "pairing_status": pairing_status,
    }


def manual_sample(
    summary_rows: Sequence[Mapping[str, Any]], *, n: int, seed: int
) -> list[dict[str, JSONValue]]:
    """Return the deterministic per-category manual-check sample.

    For every category with at least one valid member, the ``n`` valid
    reactions with the smallest ``sha256(f"{seed}:{reaction_id}")`` ranking are
    picked (reaction id breaks the tie deterministically).  The result is one
    flat list of ``{reaction_id, category}`` entries sorted by
    ``(category, ranking, reaction_id)``; a reaction may appear once per
    category it belongs to.
    """
    valid = [row for row in summary_rows if str(row["status"]) == "valid"]
    entries: list[tuple[str, str, str]] = []
    for category in SAMPLE_CATEGORIES:
        ranked = sorted(
            (
                sha256_bytes(f"{seed}:{row['reaction_id']}".encode("utf-8")),
                str(row["reaction_id"]),
            )
            for row in valid
            if _in_category(row, category)
        )
        for ranking, reaction_id in ranked[:n]:
            entries.append((category, ranking, reaction_id))
    entries.sort()
    return [
        {"reaction_id": reaction_id, "category": category}
        for category, _ranking, reaction_id in entries
    ]


def coverage_seed(config: Mapping[str, Any]) -> tuple[int, str]:
    """Return ``(seed, source)`` from ``g1.sample_seed`` else ``split.seed``."""
    settings = _settings(config)
    if "sample_seed" in settings:
        return int(settings["sample_seed"]), "g1.sample_seed"
    split_config = config.get("split")
    split_settings: Mapping[str, Any] = split_config if isinstance(split_config, Mapping) else {}
    return int(split_settings.get("seed", DEFAULT_SEED)), "split.seed"


def build_coverage_report(
    summary_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    sample_size: int | None = None,
) -> dict[str, JSONValue]:
    """Return the coverage report plus its seed metadata and manual sample."""
    seed, seed_source = coverage_seed(config)
    if sample_size is None:
        sample_size = int(_settings(config).get("sample_size", DEFAULT_SAMPLE_SIZE))
    report = coverage_report(summary_rows)
    report["seed"] = seed
    report["seed_source"] = seed_source
    report["sample_size"] = sample_size
    sample: list[JSONValue] = [dict(entry) for entry in manual_sample(summary_rows, n=sample_size, seed=seed)]
    report["manual_sample"] = sample
    return report


def write_coverage(report: Mapping[str, Any], path: str | Path) -> Path:
    """Atomically write *report* as ``g1_coverage.json`` and return its path."""
    document: dict[str, JSONValue] = dict(report)
    document["schema_version"] = G1_MANIFEST_SCHEMA_VERSION
    document["generated_at"] = _now()
    write_json(path, document)
    return Path(path)


__all__ = [
    "AMBIGUITY_CATEGORIES",
    "DEFAULT_SAMPLE_SIZE",
    "DEFAULT_SEED",
    "SAMPLE_CATEGORIES",
    "build_coverage_report",
    "coverage_report",
    "coverage_seed",
    "manual_sample",
    "write_coverage",
]
