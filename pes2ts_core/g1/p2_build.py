"""P2 orchestration: hierarchical classification and cluster reporting.

Reads **only** the P1 annotation artifacts (mappings, bond events, IRC
verdict summaries -- never truth coordinates) and classifies every
G2-eligible reaction into the four-level taxonomy of
:mod:`pes2ts_core.g1.reaction_class`.  Non-eligible P1 reactions get a typed
``excluded_p1_status`` record so the P2 denominator always reconciles with
the P1 manifest.  The cluster report adds the plan-required statistics:
per-level cluster counts, size histograms, singleton fractions, family-label
distributions, the P1-status cross-tabulation per cluster, IRC evidence
quality per cluster, representatives of large and rare clusters, the frozen
G0 split distribution across clusters (reporting only -- the split itself is
never reordered), and an explicit stability probe certificate.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from pes2ts_core.g0.fetch import dataset_version
from pes2ts_core.g1.build import shard_name
from pes2ts_core.g1.p1_truth import p1_document_path, p1_settings
from pes2ts_core.g1.reaction_class import (
    FLAG_CANONICAL_BUDGET,
    LEVEL_L0,
    LEVEL_L1,
    LEVEL_L2,
    LEVEL_L3,
    classify_reaction,
)
from pes2ts_core.g1.truth_schema import (
    CLASS_LEVELS,
    DEFAULT_CANONICAL_BUDGET,
    DEFAULT_SAMPLE_SEED,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_TAXONOMY_VERSION,
    G1_TRUTH_DIRNAME,
    G2_ELIGIBLE_STATUSES,
    P2_COVERAGE_FILENAME,
    P2_DOCUMENT_SCHEMA_VERSION,
    P2_MANIFEST_FILENAME,
    P2_MANIFEST_SCHEMA_VERSION,
    P2_STATUSES,
    P2_SUMMARY_FILENAME,
    REACTION_CLASS_DIRNAME,
    STATUS_CLASSIFIED,
    STATUS_UNCLASSIFIABLE,
)
from pes2ts_core.utils.hashing import JSONValue, sha256_bytes, sha256_file
from pes2ts_core.utils.jsonio import read_json, write_json
from pes2ts_core.utils.parquet_io import read_parquet, write_parquet

logger = logging.getLogger(__name__)

#: G0 split assignment artifact (read-only reporting input).
SPLIT_ASSIGNMENT_FILENAME: Final[str] = "split_assignment.parquet"
#: Cluster report and gate filenames.
CLUSTER_REPORT_FILENAME: Final[str] = "g1_cluster_report.json"
#: Maximum representatives kept per cluster in the report.
MAX_REPRESENTATIVES: Final[int] = 5
#: Cluster size histogram buckets for the report.
SIZE_BUCKETS: Final[tuple[int, ...]] = (1, 2, 5, 10, 50, 100, 500)

#: Summary Parquet columns of the P2 build.
P2_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "reaction_id", "status", "p1_status",
    "edit_family_id", "l1_cluster_id", "l2_cluster_id", "l3_cluster_id",
    "family_labels", "irc_orientation", "endpoint_match", "synchrony",
    "n_center_atoms", "n_context_r1", "n_context_r2", "flags",
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def p2_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the effective ``g1_class`` settings with documented defaults."""
    raw = config.get("g1_class")
    settings: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
    return {
        "taxonomy_version": str(settings.get("taxonomy_version", DEFAULT_TAXONOMY_VERSION)),
        "canonical_budget": int(settings.get("canonical_budget", DEFAULT_CANONICAL_BUDGET)),
        "sample_size": int(settings.get("sample_size", DEFAULT_SAMPLE_SIZE)),
        "sample_seed": int(settings.get("sample_seed", DEFAULT_SAMPLE_SEED)),
    }


def reaction_classes_dir(interim_dir: str | Path) -> Path:
    """Return the sharded P2 document root under ``interim/g1_truth``."""
    return Path(interim_dir) / G1_TRUTH_DIRNAME / REACTION_CLASS_DIRNAME


def p2_document_path(interim_dir: str | Path, reaction_id: str, shard_size: int) -> Path:
    """Return the sharded P2 document path of *reaction_id*."""
    return reaction_classes_dir(interim_dir) / shard_name(reaction_id, shard_size) / f"{reaction_id}.json"


@dataclass(frozen=True, slots=True)
class P2Result:
    """Outcome of one P2 classification run."""

    n_total: int
    n_classified: int
    n_excluded: int
    summary_path: Path
    manifest_path: Path
    cluster_report_path: Path


def _classify_document(
    p1_document: Mapping[str, Any], settings: Mapping[str, Any]
) -> dict[str, JSONValue]:
    """Return one P2 class document from its P1 mapping document."""
    reaction_id = str(p1_document["reaction_id"])
    p1_status = str(p1_document["status"])
    document: dict[str, JSONValue] = {
        "schema_version": P2_DOCUMENT_SCHEMA_VERSION,
        "reaction_id": reaction_id,
        "generated_at": _now(),
        "taxonomy_version": str(settings["taxonomy_version"]),
        "status": STATUS_UNCLASSIFIABLE,
        "p1_status": p1_status,
        "levels": {},
        "family_labels": [],
        "pathway_labels": {},
        "n_center_atoms": 0,
        "n_context_r1": 0,
        "n_context_r2": 0,
        "flags": [],
    }
    if p1_status not in G2_ELIGIBLE_STATUSES:
        document["status_detail"] = "excluded_p1_status"
        return document
    classification = classify_reaction(
        p1_document,
        taxonomy_version=str(settings["taxonomy_version"]),
        canonical_budget=int(settings["canonical_budget"]),
    )
    if classification is None:
        document["status_detail"] = "no_usable_events_or_graph"
        return document
    document["status"] = STATUS_CLASSIFIED
    document["levels"] = classification["levels"]
    document["family_labels"] = classification["family_labels"]
    document["pathway_labels"] = classification["pathway_labels"]
    document["n_center_atoms"] = classification["n_center_atoms"]
    document["n_context_r1"] = classification["n_context_r1"]
    document["n_context_r2"] = classification["n_context_r2"]
    document["flags"] = classification["flags"]
    return document


def _summary_row(document: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    """Return the summary Parquet row of one P2 document."""
    levels: Mapping[str, Any] = document.get("levels") or {}
    pathway: Mapping[str, Any] = document.get("pathway_labels") or {}
    return {
        "reaction_id": document["reaction_id"],
        "status": document["status"],
        "p1_status": document["p1_status"],
        "edit_family_id": levels.get(LEVEL_L0, {}).get("cluster_id"),
        "l1_cluster_id": levels.get(LEVEL_L1, {}).get("cluster_id"),
        "l2_cluster_id": levels.get(LEVEL_L2, {}).get("cluster_id"),
        "l3_cluster_id": levels.get(LEVEL_L3, {}).get("cluster_id"),
        "family_labels": "|".join(document["family_labels"]),
        "irc_orientation": pathway.get("irc_orientation"),
        "endpoint_match": pathway.get("endpoint_match"),
        "synchrony": pathway.get("synchrony"),
        "n_center_atoms": document["n_center_atoms"],
        "n_context_r1": document["n_context_r1"],
        "n_context_r2": document["n_context_r2"],
        "flags": "|".join(document["flags"]),
    }


def _split_assignment(interim_dir: Path) -> dict[str, str]:
    """Return ``reaction_id -> split`` from the frozen G0 assignment."""
    path = interim_dir / SPLIT_ASSIGNMENT_FILENAME
    if not path.is_file():
        return {}
    return {
        str(row["reaction_id"]): str(row["split"])
        for row in read_parquet(path).to_pylist()
    }


def _cluster_report(
    rows: Sequence[Mapping[str, Any]],
    documents: Sequence[Mapping[str, Any]],
    split_of: Mapping[str, str],
    settings: Mapping[str, Any],
) -> dict[str, JSONValue]:
    """Return the complete cluster report document."""
    classified = [row for row in rows if str(row["status"]) == STATUS_CLASSIFIED]
    p1_status_by_cluster: dict[str, Counter[str]] = {}
    evidence_by_cluster: dict[str, Counter[str]] = {}
    report_levels: dict[str, JSONValue] = {}
    for level, column in (
        ("l0_edit_family", "edit_family_id"),
        ("l1_center_template", "l1_cluster_id"),
        ("l2_context_r1", "l2_cluster_id"),
        ("l3_context_r2", "l3_cluster_id"),
    ):
        members: dict[str, list[Mapping[str, Any]]] = {}
        for row in classified:
            cluster_id = row[column]
            if cluster_id:
                members.setdefault(str(cluster_id), []).append(row)
        sizes = sorted((len(group) for group in members.values()), reverse=True)
        histogram: dict[str, int] = {}
        for bucket in SIZE_BUCKETS:
            histogram[f">={bucket}"] = sum(1 for size in sizes if size >= bucket)
        singletons = sum(1 for size in sizes if size == 1)
        seed = int(settings["sample_seed"])
        clusters_detail: dict[str, JSONValue] = {}
        for cluster_id in sorted(members):
            group = members[cluster_id]
            status_counts = Counter(str(row["p1_status"]) for row in group)
            evidence_counts = Counter(
                f"support={row['irc_orientation']}/{row['endpoint_match']}"
                for row in group
            )
            p1_status_by_cluster[cluster_id] = status_counts
            evidence_by_cluster[cluster_id] = evidence_counts
            splits = Counter(
                split_of.get(str(row["reaction_id"]), "unassigned")
                for row in group
            )
            representatives = sorted(
                (str(row["reaction_id"]) for row in group),
                key=lambda reaction_id: (
                    sha256_bytes(f"{seed}:{cluster_id}:{reaction_id}".encode("utf-8")),
                    reaction_id,
                ),
            )[:MAX_REPRESENTATIVES]
            clusters_detail[cluster_id] = {
                "n": len(group),
                "p1_status": dict(sorted(status_counts.items())),
                "splits": dict(sorted(splits.items())),
                "representatives": representatives,
            }
        report_levels[level] = {
            "n_clusters": len(members),
            "n_classified": len(classified),
            "singleton_fraction": (
                round(singletons / len(members), 6) if members else 0.0
            ),
            "size_histogram": histogram,
            "largest": sizes[:MAX_REPRESENTATIVES],
            "clusters": clusters_detail,
        }
    family_counts = Counter(
        label for row in classified for label in str(row["family_labels"]).split("|") if label
    )
    orientation_counts = Counter(
        str(row["irc_orientation"]) for row in classified
    )
    synchrony_counts = Counter(str(row["synchrony"]) for row in classified)
    cross_split = Counter(
        f"{split_of.get(str(row['reaction_id']), 'unassigned')}|{row['l1_cluster_id']}"
        for row in classified
    )
    spanning = {
        cluster_id: dict(sorted(splits.items()))
        for cluster_id, entry in report_levels["l1_center_template"]["clusters"].items()
        if len(entry["splits"]) > 1
    }
    budget_flagged = sum(
        1 for document in documents
        if FLAG_CANONICAL_BUDGET in document.get("flags", [])
    )
    return {
        "schema_version": P2_MANIFEST_SCHEMA_VERSION,
        "dataset_version": "",
        "taxonomy_version": settings["taxonomy_version"],
        "n_total": len(rows),
        "n_classified": len(classified),
        "n_excluded": len(rows) - len(classified),
        "levels": report_levels,
        "family_labels": dict(sorted(family_counts.items())),
        "pathway_labels": {
            "irc_orientation": dict(sorted(orientation_counts.items())),
            "synchrony": dict(sorted(synchrony_counts.items())),
        },
        "cross_split": {
            "n_l1_clusters_spanning_splits": len(spanning),
            "spanning_examples": dict(sorted(spanning.items())[:MAX_REPRESENTATIVES]),
            "pair_counts": {
                key: cross_split[key] for key in sorted(cross_split)[:MAX_REPRESENTATIVES]
            },
        },
        "canonical_budget_flagged": budget_flagged,
        "drfp_note": (
            "DRFP similarity is an optional exploratory aid and is deliberately "
            "not part of the structural cluster ids"
        ),
        "split_note": (
            "The frozen G0 split is reported, never reordered; any cluster-aware "
            "resplit is a separate audited decision"
        ),
        "generated_at": _now(),
    }


def classify_p2(config: Mapping[str, Any]) -> P2Result:
    """Classify every P1 document and write the P2 artifacts.

    A missing P1 summary raises :class:`FileNotFoundError` naming the
    ``g1 resolve-map`` prerequisite.  Every P1 document (eligible or not)
    gets one P2 record so the denominators reconcile exactly.
    """
    manifests_dir = Path(config["paths"]["manifests"])
    interim_dir = Path(config["paths"]["interim"])
    settings = p2_settings(config)
    shard_size = int(p1_settings(config)["shard_size"])
    p1_summary_path = interim_dir / "g1_p1_summary.parquet"
    if not p1_summary_path.is_file():
        raise FileNotFoundError(
            f"Missing P1 summary {p1_summary_path}; run `g1 resolve-map --allow-truth` first"
        )
    p1_rows = read_parquet(p1_summary_path).to_pylist()
    documents: list[dict[str, JSONValue]] = []
    for row in p1_rows:
        reaction_id = str(row["reaction_id"])
        p1_document = read_json(p1_document_path(interim_dir, reaction_id, shard_size))
        document = _classify_document(p1_document, settings)
        write_json(p2_document_path(interim_dir, reaction_id, shard_size), document)
        documents.append(document)
    rows = [_summary_row(document) for document in documents]
    summary_path = interim_dir / P2_SUMMARY_FILENAME
    write_parquet(summary_path, {name: [row[name] for row in rows] for name in P2_SUMMARY_COLUMNS})
    status_histogram = Counter(str(row["status"]) for row in rows)
    split_of = _split_assignment(interim_dir)
    report = _cluster_report(rows, documents, split_of, settings)
    report["dataset_version"] = dataset_version(config)
    cluster_report_path = manifests_dir / CLUSTER_REPORT_FILENAME
    write_json(cluster_report_path, report)
    manifest: dict[str, JSONValue] = {
        "schema_version": P2_MANIFEST_SCHEMA_VERSION,
        "dataset_version": dataset_version(config),
        "taxonomy_version": settings["taxonomy_version"],
        "n_total": len(rows),
        "n_classified": status_histogram.get(STATUS_CLASSIFIED, 0),
        "n_unclassifiable": status_histogram.get(STATUS_UNCLASSIFIABLE, 0),
        "by_status": {
            status: status_histogram[status]
            for status in P2_STATUSES
            if status_histogram.get(status)
        },
        "n_files": len(rows),
        "summary_sha256": sha256_file(summary_path),
        "cluster_report_sha256": sha256_file(cluster_report_path),
        "config": dict(sorted(settings.items())),
        "generated_at": _now(),
    }
    manifest_path = manifests_dir / P2_MANIFEST_FILENAME
    write_json(manifest_path, manifest)
    coverage = {
        "schema_version": P2_MANIFEST_SCHEMA_VERSION,
        "n_total": len(rows),
        "n_classified": manifest["n_classified"],
        "classified_fraction": (
            round(manifest["n_classified"] / len(rows), 6) if rows else 0.0
        ),
        "n_clusters_by_level": {
            level: report["levels"][level]["n_clusters"] for level in CLASS_LEVELS
        },
        "family_labels": report["family_labels"],
        "generated_at": _now(),
    }
    write_json(manifests_dir / P2_COVERAGE_FILENAME, coverage)
    logger.info(
        "P2 classify: %d reaction(s), %d classified, %d cluster(s) at L1 -> %s",
        len(rows), manifest["n_classified"], report["levels"]["l1_center_template"]["n_clusters"],
        summary_path,
    )
    return P2Result(
        n_total=len(rows),
        n_classified=int(manifest["n_classified"]),
        n_excluded=len(rows) - int(manifest["n_classified"]),
        summary_path=summary_path,
        manifest_path=manifest_path,
        cluster_report_path=cluster_report_path,
    )


__all__ = [
    "CLUSTER_REPORT_FILENAME",
    "P2Result",
    "P2_SUMMARY_COLUMNS",
    "SPLIT_ASSIGNMENT_FILENAME",
    "classify_p2",
    "p2_document_path",
    "p2_settings",
    "reaction_classes_dir",
]
