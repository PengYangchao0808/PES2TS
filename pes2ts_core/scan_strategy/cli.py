"""CLI-stage implementation for ``g1 v2-scan-plan|verify|freeze`` (todo 4).

Skeleton proposal/freeze pipeline for the graph-theoretic scan-strategy
selector:

- ``scan_plan_proposals`` iterates the proposal input set (initially empty),
  writes per-reaction ``g1_strategy_proposal_v1`` documents under the
  proposals tree, plus a summary parquet and a manifest.  With an empty input
  set the written manifest is deterministic (identical bytes except the
  volatile ``generated_at``).
- ``verify_scan_proposals`` re-reads the proposals tree, validates every
  document through :func:`pes2ts_core.scan_strategy.contracts_v2.validate_v2_document`
  plus a digest re-seal load check, and reconciles the manifest counts.
  Clean → exit 0 with ``problems=0``; dirty → exit 22 listing problems.
- ``freeze_scan_plans`` is the gated freeze: verification runs first; the
  freeze gate then requires verification cleanliness **and** at least one
  execution-eligible proposal.  When the gate fails (including the empty-input
  case, typed ``NO_ELIGIBLE_PROPOSALS``), the freeze manifest records
  ``plan_gate_pass=false`` and **no** consumable export is written under the
  plans tree (an empty directory skeleton is permitted).  Only a passing gate
  writes frozen ``g1_generation_plan_v2`` documents, the plan manifest, and
  the plan summary.

Artifact paths come from the ``scan_strategy`` config section, resolved
against ``paths.interim`` / ``paths.manifests`` (g1_v2 dirname idiom).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from pes2ts_core.contracts import seal_document
from pes2ts_core.scan_strategy.contracts_v2 import validate_v2_document
from pes2ts_core.utils.hashing import sha256_file
from pes2ts_core.utils.jsonio import read_json, write_json
from pes2ts_core.utils.parquet_io import write_parquet

# allow: SIZE_OK -- the plan freezes one CLI-stage module owning the
# plan/verify/freeze artifact contract (same idiom as g2/pipeline.py);
# todo 23 extends freeze inside this module rather than splitting it.

#: Exit code for a failed scan verification or a refused freeze gate
#: (``EXIT_G1_BUILD_FAILED``, same family as the v1/v2 g1 verifiers).
EXIT_SCAN_GATE_FAILED: Final[int] = 22

#: Typed freeze-gate refusal reason when the proposals tree holds no
#: execution-eligible proposal (the correct empty-input outcome, not a stub).
REASON_NO_ELIGIBLE_PROPOSALS: Final[str] = "NO_ELIGIBLE_PROPOSALS"

#: Typed freeze-gate refusal reason when proposal verification found problems.
REASON_VERIFICATION_FAILED: Final[str] = "VERIFICATION_FAILED"

PROPOSAL_MANIFEST_SCHEMA: Final[str] = "g1_v2_scan_proposal_manifest_v1"
FREEZE_MANIFEST_SCHEMA: Final[str] = "g1_v2_scan_freeze_v1"
PLAN_MANIFEST_SCHEMA: Final[str] = "g1_v2_scan_plan_manifest_v1"

#: Stable summary column order (write_parquet sorts columns by name anyway;
#: the tuple documents the intended schema for future populated runs).
PROPOSAL_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "reaction_id", "status", "family", "execution_eligible", "n_candidates",
    "content_sha256",
)
PLAN_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "reaction_id", "plan_id", "plan_version", "n_candidates", "content_sha256",
)


@dataclass(frozen=True, slots=True)
class ScanArtifactPaths:
    """Resolved artifact locations for the scan-strategy stage."""

    proposals_dir: Path
    plans_dir: Path
    proposal_summary: Path
    plan_summary: Path
    proposal_manifest: Path
    plan_manifest: Path
    freeze_manifest: Path


@dataclass(frozen=True, slots=True)
class ScanProposalResult:
    """Outcome of one ``v2-scan-plan`` invocation."""

    n_total: int
    n_written: int
    summary_path: Path
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class ScanVerifyResult:
    """Outcome of one ``v2-scan-verify`` invocation."""

    n_total: int
    n_clean: int
    problems: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScanFreezeResult:
    """Outcome of one ``v2-scan-freeze`` invocation."""

    plan_gate_pass: bool
    reasons: tuple[str, ...]
    n_proposals: int
    n_plans: int
    freeze_manifest_path: Path


def scan_artifact_paths(config: Mapping[str, Any]) -> ScanArtifactPaths:
    """Resolve the scan-strategy artifact trees from the merged config."""
    settings = config["scan_strategy"]
    interim = Path(config["paths"]["interim"])
    manifests = Path(config["paths"]["manifests"])
    return ScanArtifactPaths(
        proposals_dir=interim / str(settings["proposals_dir"]),
        plans_dir=interim / str(settings["plans_dir"]),
        proposal_summary=interim / str(settings["proposal_summary"]),
        plan_summary=interim / str(settings["plan_summary"]),
        proposal_manifest=manifests / str(settings["proposal_manifest"]),
        plan_manifest=manifests / str(settings["plan_manifest"]),
        freeze_manifest=manifests / str(settings["freeze_manifest"]),
    )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _proposal_inputs(_config: Mapping[str, Any]) -> list[str]:
    """Return the reaction ids to propose over.

    The selector core (todo 17) wires this to the approved proposal input
    population; the todo-4 skeleton deliberately iterates an empty set so the
    plan/verify/freeze artifact contract lands before any chemistry input.
    """
    return []


def _iter_proposal_documents(paths: ScanArtifactPaths) -> list[Path]:
    """Return every proposal JSON under the tree in sorted path order."""
    if not paths.proposals_dir.is_dir():
        return []
    return sorted(paths.proposals_dir.rglob("*.json"))


def scan_plan_proposals(config: Mapping[str, Any]) -> ScanProposalResult:
    """Write the proposals tree skeleton, summary, and manifest."""
    logger = logging.getLogger(__name__)
    paths = scan_artifact_paths(config)
    paths.proposals_dir.mkdir(parents=True, exist_ok=True)

    inputs = sorted(_proposal_inputs(config))
    if inputs:
        raise NotImplementedError(
            "per-reaction proposal documents arrive with the selector core (todo 17)"
        )
    n_written = 0

    write_parquet(
        paths.proposal_summary,
        {column: [] for column in PROPOSAL_SUMMARY_COLUMNS},
    )
    manifest: dict[str, Any] = {
        "schema_version": PROPOSAL_MANIFEST_SCHEMA,
        "n_total": len(inputs),
        "n_files": n_written,
        "n_summary_rows": len(inputs),
        "summary_sha256": sha256_file(paths.proposal_summary),
        "generated_at": _now(),
    }
    write_json(paths.proposal_manifest, manifest)
    logger.info(
        "v2-scan-plan: proposals=%d tree=%s manifest=%s",
        len(inputs), paths.proposals_dir, paths.proposal_manifest,
    )
    return ScanProposalResult(
        n_total=len(inputs),
        n_written=n_written,
        summary_path=paths.proposal_summary,
        manifest_path=paths.proposal_manifest,
    )


def verify_scan_proposals(config: Mapping[str, Any]) -> ScanVerifyResult:
    """Re-read the proposals tree and reconcile it with the manifest."""
    paths = scan_artifact_paths(config)
    problems: list[str] = []

    documents = _iter_proposal_documents(paths)
    n_clean = 0
    for document_path in documents:
        try:
            document = read_json(document_path)
        except (OSError, ValueError) as exc:
            problems.append(f"{document_path}: unreadable ({exc})")
            continue
        if not isinstance(document, dict):
            problems.append(f"{document_path}: document root must be an object")
            continue
        issues = validate_v2_document(document)
        if issues:
            problems.append(f"{document_path}: {'; '.join(issues)}")
            continue
        if seal_document(document).get("content_sha256") != document.get("content_sha256"):
            problems.append(f"{document_path}: content_sha256 digest mismatch")
            continue
        n_clean += 1

    if not paths.proposal_manifest.is_file():
        problems.append(
            f"Missing scan proposal manifest {paths.proposal_manifest}; "
            "run `g1 v2-scan-plan` first"
        )
    else:
        manifest = read_json(paths.proposal_manifest)
        if not isinstance(manifest, dict):
            problems.append(f"{paths.proposal_manifest}: manifest is not a JSON object")
        else:
            if manifest.get("n_total") != len(documents):
                problems.append(
                    f"proposal manifest n_total={manifest.get('n_total')!r} "
                    f"but the tree holds {len(documents)} document(s)"
                )
            if manifest.get("n_files") != len(documents):
                problems.append(
                    f"proposal manifest n_files={manifest.get('n_files')!r} "
                    f"but the tree holds {len(documents)} document(s)"
                )
            if not paths.proposal_summary.is_file():
                problems.append(
                    f"Missing scan proposal summary {paths.proposal_summary}"
                )
            elif manifest.get("summary_sha256") != sha256_file(paths.proposal_summary):
                problems.append(
                    "proposal manifest summary_sha256 does not match the summary parquet"
                )

    return ScanVerifyResult(
        n_total=len(documents),
        n_clean=n_clean,
        problems=tuple(problems),
    )


def freeze_scan_plans(config: Mapping[str, Any]) -> ScanFreezeResult:
    """Gate proposals into frozen plans; refuse without consumable exports."""
    logger = logging.getLogger(__name__)
    paths = scan_artifact_paths(config)

    verification = verify_scan_proposals(config)
    reasons: list[str] = []
    if verification.problems:
        reasons.append(REASON_VERIFICATION_FAILED)

    eligible = [
        document_path
        for document_path in _iter_proposal_documents(paths)
        if _is_execution_eligible(document_path)
    ]
    if not eligible:
        reasons.append(REASON_NO_ELIGIBLE_PROPOSALS)

    if reasons:
        paths.plans_dir.mkdir(parents=True, exist_ok=True)
        freeze_manifest: dict[str, Any] = {
            "schema_version": FREEZE_MANIFEST_SCHEMA,
            "plan_gate_pass": False,
            "reasons": reasons,
            "n_proposals": verification.n_total,
            "n_plans": 0,
            "n_verification_problems": len(verification.problems),
            "generated_at": _now(),
        }
        write_json(paths.freeze_manifest, freeze_manifest)
        logger.info(
            "v2-scan-freeze: plan_gate_pass=false reasons=%s", reasons,
        )
        return ScanFreezeResult(
            plan_gate_pass=False,
            reasons=tuple(reasons),
            n_proposals=verification.n_total,
            n_plans=0,
            freeze_manifest_path=paths.freeze_manifest,
        )

    raise NotImplementedError(
        "frozen plan export (g1_generation_plan_v2 + plan manifest + plan "
        "summary) arrives with the GenerationPlanV2 freeze todo 23"
    )


def _is_execution_eligible(document_path: Path) -> bool:
    """True when the proposal document is valid and execution-eligible."""
    try:
        document = read_json(document_path)
    except (OSError, ValueError):
        return False
    if not isinstance(document, dict):
        return False
    if validate_v2_document(document):
        return False
    return document.get("execution_eligible") is True


__all__ = [
    "EXIT_SCAN_GATE_FAILED",
    "FREEZE_MANIFEST_SCHEMA",
    "PLAN_MANIFEST_SCHEMA",
    "PROPOSAL_MANIFEST_SCHEMA",
    "PROPOSAL_SUMMARY_COLUMNS",
    "PLAN_SUMMARY_COLUMNS",
    "REASON_NO_ELIGIBLE_PROPOSALS",
    "REASON_VERIFICATION_FAILED",
    "ScanArtifactPaths",
    "ScanFreezeResult",
    "ScanProposalResult",
    "ScanVerifyResult",
    "freeze_scan_plans",
    "scan_artifact_paths",
    "scan_plan_proposals",
    "verify_scan_proposals",
]
