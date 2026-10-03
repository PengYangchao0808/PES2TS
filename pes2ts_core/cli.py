"""Command-line interface for PES2TS.

Usage::

    pes2ts [--config PATH] [--log-level LEVEL] [--log-file PATH] {g0,g1} <SUBCOMMAND>

Every ``g0`` subcommand is implemented. ``fetch`` downloads and checksum-verifies
the configured Reaction-QM source files; ``inventory`` builds the TS-free R/P
inventory; ``quarantine`` extracts and relocates the TS/IRC ground truth and
writes its manifest; ``dedup`` detects exact duplicates and written reverses;
``split`` adopts the authors' official split; ``audit`` runs the mandatory DRFP
near-duplicate cross-split leakage audit; ``freeze`` freezes the split manifest
under the explicit leak decision policy; ``cohorts`` selects the deterministic
trial and stratified cohorts; ``run-all`` orchestrates the idempotent pipeline;
and ``truth-index`` is the audited accessor for the resulting IRC index.

Every ``g1`` subcommand is implemented as well. ``build`` assembles the
per-reaction bond-change documents, summary, manifest, and coverage report for
a cohort or the whole inventory; ``sample`` prints the deterministic
manual-check sample; ``strata`` re-derives the authoritative strata and
cohorts from a full-build summary; ``verify`` re-reads the written tree
(``--stage build|p1|p2``) and reconciles it with the summary and manifest;
``join-audit`` and ``resolve-map`` run the truth-assisted P1 layer (both
demand an explicit ``--allow-truth`` and log every audited truth read);
``classify`` builds the P2 hierarchical reaction taxonomy; and ``gate``
writes the G2 eligibility decision.

Every ``g2`` subcommand is implemented as well. ``prepare`` assembles the
deterministic endpoints for the selected reactions, ``run`` executes the
GFN2-xTB PATH step and writes the per-reaction path artifacts, and ``verify``
re-reads the whole G2 tree and reconciles it; infrastructure failures (missing
inputs, a missing xTB executable, or verification problems) exit 24.
``--help`` works for every subcommand and exits 0.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from pes2ts_core.config_loader import EXIT_CONFIG_ERROR, ConfigError, load_config
from pes2ts_core.g0.dedup import detect_duplicates
from pes2ts_core.g0.fetch import (
    EXIT_CHECKSUM_MISMATCH,
    ChecksumMismatch,
    fetch_sources,
    verify_sources,
    write_source_manifest,
)
from pes2ts_core.g0.inventory import INVENTORY_PARQUET_FILENAME, build_inventory
from pes2ts_core.g0.reader import H5SchemaError
from pes2ts_core.g0.split import (
    EXIT_AUDIT_INCOMPLETE,
    EXIT_LEAK_FOUND,
    EXIT_REMEDIATION_FAILED,
    EXIT_SPLIT_SCHEMA_ERROR,
    REMEDIATION_CHOICES,
    REMEDIATION_NONE,
    AuditIncompleteError,
    RemediationFailedError,
    SplitSchemaError,
    adopt_official_split,
    freeze_split,
)
from pes2ts_core.g0.strata import (
    EXIT_AUTHORITATIVE_STRATA_REQUIRED,
    AuthoritativeStrataRequiredError,
    select_cohorts,
)
from pes2ts_core.g0.truth_quarantine import (
    EXIT_QUARANTINE_ERROR,
    quarantine_truth,
)
from pes2ts_core.g1.build import (
    EXIT_G1_BUILD_FAILED,
    SUMMARY_FILENAME,
    build_g1,
    cohort_member_ids,
)
from pes2ts_core.g1.coverage import coverage_seed, manual_sample
from pes2ts_core.g1.gate import GATE_HINT, run_gate
from pes2ts_core.g1.p1_truth import resolve_p1, run_join_audit
from pes2ts_core.g1.p1_verify import verify_p1
from pes2ts_core.g1.p2_build import classify_p2
from pes2ts_core.g1.p2_verify import verify_p2
from pes2ts_core.g1.strata_auth import rebuild_authoritative_strata
from pes2ts_core.g1.truth_schema import EXIT_TRUTH_FLAG_REQUIRED
from pes2ts_core.g1.verify import verify_g1
from pes2ts_core.generation.execution.xtb_path import EXIT_G2_FAILED
from pes2ts_core.generation.execution.xtb_path.pipeline import (
    InfrastructureError,
    prepare_ids,
    run_ids,
    select_reaction_ids,
)
from pes2ts_core.generation.execution.xtb_path.verify import verify_g2
from pes2ts_core.integration.acp.cli_backend import ACPCLIError
from pes2ts_core.logging_setup import setup_logging
from pes2ts_core.utils.parquet_io import read_parquet
from pes2ts_core.version import __version__

PROG = "pes2ts"

#: Order is user-visible in ``g0 --help``.
G0_SUBCOMMANDS: tuple[str, ...] = (
    "fetch",
    "inventory",
    "quarantine",
    "dedup",
    "audit",
    "split",
    "freeze",
    "cohorts",
    "run-all",
    "truth-index",
)

#: Order is user-visible in ``g1 --help``.
G1_SUBCOMMANDS: tuple[str, ...] = (
    "build",
    "sample",
    "strata",
    "verify",
    "join-audit",
    "resolve-map",
    "classify",
    "gate",
    "v2-build",
    "v2-classify",
    "v2-verify",
    "v2-gate",
    "v2-sanitize-exports",
    "v2-scan-plan",
    "v2-scan-verify",
    "v2-scan-freeze",
)

#: ``g1 verify --stage`` choices.
VERIFY_STAGES: tuple[str, ...] = ("build", "p1", "p2")

#: Order is user-visible in ``g2 --help``.
G2_SUBCOMMANDS: tuple[str, ...] = ("prepare", "run", "verify")

G0Handler = Callable[[argparse.Namespace, dict[str, Any]], int]
G1Handler = Callable[[argparse.Namespace, dict[str, Any]], int]
G2Handler = Callable[[argparse.Namespace, dict[str, Any]], int]


def _fetch_handler(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Acquire the configured source files, verify them, then record the manifest."""
    logger = logging.getLogger(__name__)
    try:
        fetched = fetch_sources(config, force=bool(getattr(args, "force", False)))
        digests = verify_sources(config)
        manifest_path = write_source_manifest(config, fetched, digests)
    except ChecksumMismatch as exc:
        logger.error("%s", exc)
        return EXIT_CHECKSUM_MISMATCH
    logger.info("Source manifest written: %s", manifest_path)
    return 0


def _g0_inventory_handler(_args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Build the TS-free reactant/product inventory and report the artifacts."""
    logger = logging.getLogger(__name__)
    try:
        result = build_inventory(config)
    except (ChecksumMismatch, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return EXIT_CHECKSUM_MISMATCH
    logger.info(
        "Inventory: %d row(s), %d rejection(s) -> %s",
        result.n_rows,
        result.n_rejections,
        result.parquet_path,
    )
    return 0


def _g0_dedup_handler(_args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Detect exact duplicates and written reverses in the inventory."""
    logger = logging.getLogger(__name__)
    inventory_path = Path(config["paths"]["interim"]) / INVENTORY_PARQUET_FILENAME
    try:
        result = detect_duplicates(inventory_path, config)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return EXIT_CHECKSUM_MISMATCH
    logger.info(
        "Dedup: %d row(s), %d unique, %d group(s), %d duplicate(s), %d reverse(s) -> %s",
        result.n_rows,
        result.n_unique,
        result.n_duplicate_groups,
        result.n_exact_duplicates,
        result.n_reverse_pairs,
        result.ledger_path,
    )
    print(
        f"dedup: rows={result.n_rows} unique={result.n_unique} "
        f"groups={result.n_duplicate_groups} "
        f"exact_duplicates={result.n_exact_duplicates} "
        f"reverse_pairs={result.n_reverse_pairs}"
    )
    return 0


def _g0_audit_handler(_args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Run the full cross-split near-duplicate audit; exit 6 on budget abort."""
    from pes2ts_core.g0.neardup import (
        EXIT_AUDIT_BUDGET_EXCEEDED, AuditBudgetExceeded, compute_fingerprints,
        cross_split_leak_audit,
    )

    logger = logging.getLogger(__name__)
    try:
        fingerprints = compute_fingerprints(config)
        result = cross_split_leak_audit(config, fingerprints.fingerprints)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return EXIT_CHECKSUM_MISMATCH
    except AuditBudgetExceeded as exc:
        logger.error("Leak audit aborted: %s", exc)
        return EXIT_AUDIT_BUDGET_EXCEEDED
    logger.info(
        "Leak audit: %d pair(s) over threshold, max_similarity=%.6f -> %s",
        result.n_pairs_over_threshold,
        result.max_similarity,
        result.manifest_path,
    )
    print(
        f"audit: pairs_over_threshold={result.n_pairs_over_threshold} "
        f"max_similarity={result.max_similarity:.6f} complete={result.complete}"
    )
    return 0


def _g0_split_handler(_args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Adopt the official split; exit 5 on schema violation, 3 on missing input."""
    logger = logging.getLogger(__name__)
    try:
        result = adopt_official_split(config)
    except SplitSchemaError as exc:
        logger.error("Split schema error: %s", exc)
        return EXIT_SPLIT_SCHEMA_ERROR
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return EXIT_CHECKSUM_MISMATCH
    logger.info(
        "Split adoption: covered=%d/%d not_in_split=%d -> %s",
        result.covered,
        result.covered + result.n_not_in_split,
        result.n_not_in_split,
        result.manifest_path,
    )
    print(
        f"split: train={result.counts['train']} valid={result.counts['valid']} "
        f"test={result.counts['test']} total={result.counts['total']} "
        f"covered={result.covered} not_in_split={result.n_not_in_split}"
    )
    return 0


def _g0_freeze_handler(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Freeze the split manifest under the explicit leak decision policy."""
    from pes2ts_core.g0.neardup import EXIT_AUDIT_BUDGET_EXCEEDED, AuditBudgetExceeded

    logger = logging.getLogger(__name__)
    remediation = cast(
        Literal["none", "exclude-leaky", "rebuild"],
        getattr(args, "remediation", REMEDIATION_NONE),
    )
    try:
        result = freeze_split(config, remediation=remediation)
    except AuditIncompleteError as exc:
        logger.error("Split freeze refused: %s", exc)
        return EXIT_AUDIT_INCOMPLETE
    except RemediationFailedError as exc:
        logger.error("Split remediation failed: %s", exc)
        return EXIT_REMEDIATION_FAILED
    except AuditBudgetExceeded as exc:
        logger.error("Leak audit aborted during split freeze: %s", exc)
        return EXIT_AUDIT_BUDGET_EXCEEDED
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return EXIT_CHECKSUM_MISMATCH
    if result.halted:
        logger.error(
            "Split freeze halted with leak_status=%s: %d offense(s); re-run "
            "with --remediation exclude-leaky or --remediation rebuild to "
            "remediate",
            result.leak_status,
            result.n_pairs_over_threshold + result.n_cross_split_known_duplicates,
        )
        return EXIT_LEAK_FOUND
    logger.info(
        "Split freeze: leak_status=%s skipped=%s counts=%s excluded=%d -> %s",
        result.leak_status,
        result.skipped,
        result.counts,
        len(result.excluded_ids),
        result.manifest_path,
    )
    print(
        f"freeze: leak_status={result.leak_status} skipped={result.skipped} "
        f"train={result.counts.get('train', 0)} "
        f"valid={result.counts.get('valid', 0)} "
        f"test={result.counts.get('test', 0)} "
        f"excluded={len(result.excluded_ids)}"
    )
    return 0


def _g0_cohorts_handler(_args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Select the deterministic trial and stratified cohorts from the inventory."""
    logger = logging.getLogger(__name__)
    try:
        result = select_cohorts(config)
    except AuthoritativeStrataRequiredError as exc:
        logger.error("Cohorts halted: %s", exc)
        return EXIT_AUTHORITATIVE_STRATA_REQUIRED
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return EXIT_CHECKSUM_MISMATCH
    logger.info(
        "Cohorts: trial=%d, stratified=%d -> %s",
        result.trial_size,
        result.stratified_size,
        result.report_path,
    )
    print(
        f"cohorts: trial={result.trial_size} "
        f"stratified={result.stratified_size} report={result.report_path}"
    )
    return 0


def _g0_quarantine_handler(_args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Extract, relocate, and manifest the TS/IRC ground truth."""
    logger = logging.getLogger(__name__)
    try:
        result = quarantine_truth(config)
    except ConfigError as exc:
        logger.error("Quarantine configuration error: %s", exc)
        return EXIT_CONFIG_ERROR
    except (OSError, H5SchemaError) as exc:
        logger.error("Quarantine failed: %s", exc)
        return EXIT_QUARANTINE_ERROR
    logger.info(
        "Quarantine %s: %d TS row(s), %d IRC index row(s), %d reaction(s) without TS",
        "verified (skipped)" if result.skipped else "extracted",
        result.n_ts_rows,
        result.n_irc_rows,
        result.n_reactions_without_ts,
    )
    print(
        f"quarantine: ts_rows={result.n_ts_rows} "
        f"irc_index_rows={result.n_irc_rows} "
        f"reactions_without_ts={result.n_reactions_without_ts} "
        f"relocated={list(result.relocated)} skipped={result.skipped}"
    )
    return 0


def _g0_run_all_handler(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Run the full idempotent G0 pipeline; exit 21 on a stage failure or halt."""
    from pes2ts_core.g0.pipeline import EXIT_PIPELINE_FAILED, run_pipeline, with_data_root

    logger = logging.getLogger(__name__)
    data_root = getattr(args, "data_root", None)
    if data_root:
        config = with_data_root(config, data_root)
    report = run_pipeline(
        config,
        skip_fetch=bool(getattr(args, "skip_fetch", False)),
        force=bool(getattr(args, "force", False)),
        remediation=str(getattr(args, "remediation", REMEDIATION_NONE)),
    )
    if not report.ok:
        failed = next(
            (stage for stage in report.stages if stage.status == "failed"), None
        )
        if failed is not None:
            logger.error(
                "g0 run-all failed at stage %s: %s",
                failed.name,
                failed.error or failed.reason,
            )
        else:
            logger.error("g0 run-all did not complete every stage")
        return EXIT_PIPELINE_FAILED
    ran = sum(1 for stage in report.stages if stage.status == "ran")
    skipped = sum(1 for stage in report.stages if stage.status == "skipped")
    logger.info(
        "g0 run-all complete: %d ran, %d skipped, leak_status=%s -> %s",
        ran,
        skipped,
        report.leak_status,
        report.report_path,
    )
    print(
        f"run-all: stages={len(report.stages)} ran={ran} skipped={skipped} "
        f"leak_status={report.leak_status} report={report.report_path}"
    )
    return 0


def _g0_truth_index_handler(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Audited access to the quarantined IRC index (the single truth-index path)."""
    from pes2ts_core.g0.truth import truth_reader

    logger = logging.getLogger(__name__)
    manifests_dir = config["paths"]["manifests"]
    reaction_id = getattr(args, "reaction_id", None)
    if reaction_id:
        frames = truth_reader.load_irc_frames(
            reaction_id, allow_truth=True, manifests_dir=manifests_dir
        )
        logger.info("Truth-index read reaction %s", frames["reaction_id"])
        print(
            f"truth-index: {frames['reaction_id']} n_atoms={frames['n_atoms']} "
            f"n_frames={frames['n_frames']} has_forces={frames['has_forces']} "
            f"EHG={'present' if frames['EHG'] is not None else 'absent'}"
        )
    else:
        rows = truth_reader.load_irc_index(allow_truth=True, manifests_dir=manifests_dir)
        with_forces = sum(1 for row in rows if row["has_forces"])
        logger.info("Truth-index summary over %d reaction(s)", len(rows))
        print(f"truth-index: reactions={len(rows)} with_forces={with_forces}")
    return 0


def _g1_build_handler(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Build the G1 documents for a cohort; exit 3 on a missing cohort/inventory."""
    logger = logging.getLogger(__name__)
    try:
        ids = cohort_member_ids(
            config,
            str(getattr(args, "cohort", "all")),
            limit=getattr(args, "limit", None),
        )
        result = build_g1(config, reaction_ids=ids)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return EXIT_CHECKSUM_MISMATCH
    except ValueError as exc:
        logger.error("G1 build failed: %s", exc)
        return EXIT_G1_BUILD_FAILED
    logger.info(
        "G1 build: %d reaction(s), %d valid, %d rejected -> %s",
        result.n_total,
        result.n_valid,
        result.n_rejected,
        result.summary_path,
    )
    return 0


def _g1_sample_handler(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Print the deterministic manual-check sample read from the summary parquet."""
    logger = logging.getLogger(__name__)
    summary_path = Path(config["paths"]["interim"]) / SUMMARY_FILENAME
    if not summary_path.is_file():
        logger.error("Missing summary %s; run `g1 build` first", summary_path)
        return EXIT_CHECKSUM_MISMATCH
    rows = read_parquet(summary_path).to_pylist()
    g1_config = config.get("g1")
    settings = g1_config if isinstance(g1_config, dict) else {}
    n = getattr(args, "n", None)
    if n is None:
        n = int(settings.get("sample_size", 20))
    seed, seed_source = coverage_seed(config)
    sample = manual_sample(rows, n=int(n), seed=seed)
    category = getattr(args, "category", None)
    if category:
        sample = [entry for entry in sample if entry["category"] == category]
    by_id = {str(row["reaction_id"]): row for row in rows}
    for entry in sample:
        row = by_id[str(entry["reaction_id"])]
        print(
            f"{entry['reaction_id']} {entry['category']} formed={row['n_formed']} "
            f"broken={row['n_broken']} order_changed={row['n_order_changed']} "
            f"h_migration={row['n_h_migration']} index={row['index_status']} "
            f"pairing={row['pairing_status']}"
        )
    logger.info(
        "G1 sample: %d row(s) (n=%d, seed=%d from %s)",
        len(sample),
        n,
        seed,
        seed_source,
    )
    return 0


def _g1_strata_handler(_args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Re-derive the authoritative strata and cohorts; exit 3 on missing input."""
    logger = logging.getLogger(__name__)
    try:
        result = rebuild_authoritative_strata(config)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return EXIT_CHECKSUM_MISMATCH
    except ValueError as exc:
        logger.error("G1 strata failed: %s", exc)
        return EXIT_G1_BUILD_FAILED
    logger.info(
        "G1 strata: %d inventory row(s), %d authoritative, report -> %s",
        result.n_inventory,
        result.n_valid,
        result.report_path,
    )
    return 0


def _g1_verify_handler(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Re-read the written documents of one stage; exit 22 on any problem."""
    logger = logging.getLogger(__name__)
    stage = str(getattr(args, "stage", "build") or "build")
    try:
        if stage == "p1":
            result = verify_p1(config)
            n_ok, n_bad = result.n_eligible, result.n_total - result.n_eligible
            problems = result.problems
        elif stage == "p2":
            result = verify_p2(config)
            n_ok, n_bad = result.n_classified, result.n_total - result.n_classified
            problems = result.problems
        else:
            result = verify_g1(config)
            n_ok, n_bad = result.n_valid, result.n_rejected
            problems = result.problems
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return EXIT_CHECKSUM_MISMATCH
    except ValueError as exc:
        logger.error("G1 verify failed: %s", exc)
        return EXIT_G1_BUILD_FAILED
    if problems:
        for problem in problems[:10]:
            logger.error("G1 verify: %s", problem)
        logger.error("G1 verify failed with %d problem(s)", len(problems))
        return EXIT_G1_BUILD_FAILED
    logger.info(
        "G1 verify (%s) OK: %d document(s), %d ok, %d other",
        stage,
        n_ok + n_bad,
        n_ok,
        n_bad,
    )
    return 0


def _g1_join_audit_handler(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Write the P1 join audit; exit 23 without ``--allow-truth``."""
    logger = logging.getLogger(__name__)
    if not bool(getattr(args, "allow_truth", False)):
        logger.error("join-audit reads the quarantined truth; pass --allow-truth")
        return EXIT_TRUTH_FLAG_REQUIRED
    try:
        path = run_join_audit(config, allow_truth=True)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return EXIT_CHECKSUM_MISMATCH
    except PermissionError as exc:
        logger.error("Truth access refused: %s", exc)
        return EXIT_TRUTH_FLAG_REQUIRED
    logger.info("P1 join audit written: %s", path)
    return 0


def _g1_resolve_map_handler(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Build the P1 mapping documents; exit 23 without ``--allow-truth``."""
    logger = logging.getLogger(__name__)
    if not bool(getattr(args, "allow_truth", False)):
        logger.error("resolve-map reads the quarantined truth; pass --allow-truth")
        return EXIT_TRUTH_FLAG_REQUIRED
    try:
        ids = cohort_member_ids(
            config,
            str(getattr(args, "cohort", "all")),
            limit=getattr(args, "limit", None),
        )
        result = resolve_p1(
            config, allow_truth=True, reaction_ids=ids,
            limit=getattr(args, "limit", None),
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return EXIT_CHECKSUM_MISMATCH
    except PermissionError as exc:
        logger.error("Truth access refused: %s", exc)
        return EXIT_TRUTH_FLAG_REQUIRED
    except ValueError as exc:
        logger.error("P1 resolve failed: %s", exc)
        return EXIT_G1_BUILD_FAILED
    logger.info(
        "P1 resolve-map: %d reaction(s), %d eligible -> %s",
        result.n_total,
        result.n_eligible,
        result.summary_path,
    )
    print(
        f"resolve-map: total={result.n_total} eligible={result.n_eligible} "
        f"by_status={result.by_status}"
    )
    return 0


def _g1_classify_handler(_args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Classify the P1 documents into the hierarchical reaction taxonomy."""
    logger = logging.getLogger(__name__)
    try:
        result = classify_p2(config)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return EXIT_CHECKSUM_MISMATCH
    except ValueError as exc:
        logger.error("P2 classify failed: %s", exc)
        return EXIT_G1_BUILD_FAILED
    logger.info(
        "P2 classify: %d reaction(s), %d classified -> %s",
        result.n_total,
        result.n_classified,
        result.summary_path,
    )
    print(
        f"classify: total={result.n_total} classified={result.n_classified} "
        f"excluded={result.n_excluded}"
    )
    return 0


def _g1_gate_handler(_args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Write the G2 eligibility gate manifest; exit 22 on failed verification."""
    logger = logging.getLogger(__name__)
    try:
        result = run_gate(config)
    except FileNotFoundError as exc:
        logger.error("%s; %s", exc, GATE_HINT)
        return EXIT_CHECKSUM_MISMATCH
    if result.refusals:
        for refusal in result.refusals[:10]:
            logger.error("Gate refusal: %s", refusal)
        return EXIT_G1_BUILD_FAILED
    logger.info(
        "G1 gate: %d/%d eligible (%.4f) -> %s",
        result.n_eligible,
        result.n_denominator,
        result.eligible_fraction,
        result.manifest_path,
    )
    print(
        f"gate: eligible={result.n_eligible}/{result.n_denominator} "
        f"fraction={result.eligible_fraction}"
    )
    return 0


def _g1_v2_build_handler(_args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Freeze the v1 baseline and rebuild the v2 exclusive-edit tree."""
    logger = logging.getLogger(__name__)
    from pes2ts_core.g1.v2_build import build_v2

    try:
        result = build_v2(config)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return EXIT_CHECKSUM_MISMATCH
    print(
        f"v2-build: total={result.n_total} clean={result.n_clean} "
        f"issues={result.n_issues} excluded={result.n_excluded}"
    )
    return 0


def _g1_v2_classify_handler(_args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Classify every v2 edit document and write the migration report."""
    logger = logging.getLogger(__name__)
    from pes2ts_core.g1.v2_classify import classify_v2

    try:
        manifest = classify_v2(config)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return EXIT_CHECKSUM_MISMATCH
    print(
        f"v2-classify: total={manifest['n_total']} "
        f"classified={manifest['n_classified']} "
        f"l0_clusters={manifest['n_clusters_by_level']['l0_edit_family']}"
    )
    return 0


def _g1_v2_verify_handler(_args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Recompute the v2 trees; exit 22 on any verification problem."""
    logger = logging.getLogger(__name__)
    from pes2ts_core.g1.v2_verify import verify_v2

    try:
        result = verify_v2(config)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return EXIT_CHECKSUM_MISMATCH
    if result.problems:
        for problem in result.problems[:10]:
            logger.error("v2 verify: %s", problem)
        return EXIT_G1_BUILD_FAILED
    print(
        f"v2-verify: total={result.n_total} clean={result.n_clean} "
        f"issues={result.n_issues} excluded={result.n_excluded} problems=0"
    )
    return 0


def _g1_v2_gate_handler(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Write the v2 gate manifests and the whitelisted G2 export."""
    logger = logging.getLogger(__name__)
    from pes2ts_core.g1.v2_gate import run_v2_gate

    assume_verified = bool(getattr(args, "assume_verified", False))
    try:
        result = run_v2_gate(config, assume_verified=assume_verified)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return EXIT_CHECKSUM_MISMATCH
    if result.refusals:
        for refusal in result.refusals[:10]:
            logger.error("v2 gate refusal: %s", refusal)
        return EXIT_G1_BUILD_FAILED
    print(
        f"v2-gate: scan_ready={result.n_scan_ready}/{result.n_denominator} "
        f"needs_review={result.n_needs_review} excluded={result.n_excluded}"
    )
    return 0


def _g1_v2_sanitize_exports_handler(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Non-destructively migrate the legacy G2 export tree without truth fields."""
    from pes2ts_core.g1.export_sanitize import migrate_export_tree
    from pes2ts_core.g1.v2_schema import G2_SCAN_READY_FILENAME, EXPORT_DIRNAME, V2_DIRNAME
    from pes2ts_core.utils.jsonio import read_json

    interim = Path(config["paths"]["interim"])
    source_root = interim / V2_DIRNAME / EXPORT_DIRNAME
    output_root = Path(args.output_root) if args.output_root else source_root.with_name("export_contracts_v1")
    ready = read_json(interim / G2_SCAN_READY_FILENAME)
    reaction_ids = set(ready.get("reaction_ids", []))
    manifest = migrate_export_tree(
        source_root, output_root, expected_reaction_ids=reaction_ids,
        resume_staging=bool(args.resume_staging),
    )
    print(
        f"v2-sanitize-exports: exports={manifest['n_exports']} bytes={manifest['n_bytes']} "
        f"removed={manifest['removed_truth_key_occurrences']} output={output_root}"
    )
    return 0


def _g1_v2_scan_plan_handler(_args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Write the scan-proposal tree skeleton, summary, and manifest."""
    from pes2ts_core.generation.planning.cli import scan_plan_proposals

    result = scan_plan_proposals(config)
    print(
        f"v2-scan-plan: proposals={result.n_total} written={result.n_written} "
        f"summary={result.summary_path} manifest={result.manifest_path}"
    )
    return 0


def _g1_v2_scan_verify_handler(_args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Re-read the scan-proposal tree; exit 22 on any verification problem."""
    from pes2ts_core.generation.planning.cli import verify_scan_proposals

    result = verify_scan_proposals(config)
    if result.problems:
        for problem in result.problems[:10]:
            logging.getLogger(__name__).error("v2-scan-verify: %s", problem)
        print(
            f"v2-scan-verify: total={result.n_total} clean={result.n_clean} "
            f"problems={len(result.problems)}"
        )
        return EXIT_G1_BUILD_FAILED
    print(
        f"v2-scan-verify: total={result.n_total} clean={result.n_clean} problems=0"
    )
    return 0


def _g1_v2_scan_freeze_handler(_args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Gate scan proposals into frozen plans; refuse without consumable exports."""
    from pes2ts_core.generation.planning.cli import freeze_scan_plans

    result = freeze_scan_plans(config)
    if not result.plan_gate_pass:
        print(
            f"v2-scan-freeze: plan_gate_pass=false reasons={list(result.reasons)} "
            f"manifest={result.freeze_manifest_path}"
        )
        return EXIT_G1_BUILD_FAILED
    print(
        f"v2-scan-freeze: plan_gate_pass=true plans={result.n_plans} "
        f"manifest={result.freeze_manifest_path}"
    )
    return 0


#: The handler for every ``g0`` subcommand; all are implemented.
SUBCOMMAND_HANDLERS: dict[str, G0Handler] = {
    "fetch": _fetch_handler,
    "inventory": _g0_inventory_handler,
    "quarantine": _g0_quarantine_handler,
    "dedup": _g0_dedup_handler,
    "audit": _g0_audit_handler,
    "split": _g0_split_handler,
    "freeze": _g0_freeze_handler,
    "cohorts": _g0_cohorts_handler,
    "run-all": _g0_run_all_handler,
    "truth-index": _g0_truth_index_handler,
}

#: The handler for every ``g1`` subcommand; all are implemented.
G1_SUBCOMMAND_HANDLERS: dict[str, G1Handler] = {
    "build": _g1_build_handler,
    "sample": _g1_sample_handler,
    "strata": _g1_strata_handler,
    "verify": _g1_verify_handler,
    "join-audit": _g1_join_audit_handler,
    "resolve-map": _g1_resolve_map_handler,
    "classify": _g1_classify_handler,
    "gate": _g1_gate_handler,
    "v2-build": _g1_v2_build_handler,
    "v2-classify": _g1_v2_classify_handler,
    "v2-verify": _g1_v2_verify_handler,
    "v2-gate": _g1_v2_gate_handler,
    "v2-sanitize-exports": _g1_v2_sanitize_exports_handler,
    "v2-scan-plan": _g1_v2_scan_plan_handler,
    "v2-scan-verify": _g1_v2_scan_verify_handler,
    "v2-scan-freeze": _g1_v2_scan_freeze_handler,
}


def _select_g2_ids(args: argparse.Namespace, config: dict[str, Any]) -> list[str]:
    """Resolve the g2 prepare/run selection (cohort/limit/--reaction filters)."""
    return select_reaction_ids(
        config,
        cohort=str(getattr(args, "cohort", "trial")),
        limit=getattr(args, "limit", None),
        reactions=tuple(getattr(args, "reaction", None) or ()),
    )


def _g2_prepare_handler(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Assemble the endpoints for the selected reactions; exit 24 on infrastructure errors."""
    logger = logging.getLogger(__name__)
    try:
        ids = _select_g2_ids(args, config)
        report = prepare_ids(ids, config=config)
    except (InfrastructureError, ACPCLIError, ValueError) as exc:
        logger.error("%s", exc)
        return EXIT_G2_FAILED
    logger.info(
        "G2 prepare: %d selected, %d prepared, %d ledgered",
        report.n_selected,
        report.n_attempted,
        report.n_skipped,
    )
    return 0


def _g2_run_handler(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Run the xTB PATH stage; batch completion is 0, infrastructure errors exit 24."""
    logger = logging.getLogger(__name__)
    try:
        ids = _select_g2_ids(args, config)
        report = run_ids(ids, config=config, force=bool(getattr(args, "force", False)))
    except (InfrastructureError, ACPCLIError, ValueError) as exc:
        logger.error("%s", exc)
        return EXIT_G2_FAILED
    logger.info(
        "G2 run: %d selected, %d attempted, %d skipped",
        report.n_selected,
        report.n_attempted,
        report.n_skipped,
    )
    return 0


def _g2_verify_handler(_args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Re-read the G2 tree; exit 24 on any problem or missing batch artifacts."""
    logger = logging.getLogger(__name__)
    try:
        result = verify_g2(config=config)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return EXIT_G2_FAILED
    if result.problems:
        for problem in result.problems:
            print(f"g2 verify: {problem}")
        logger.error("G2 verify failed with %d problem(s)", len(result.problems))
        return EXIT_G2_FAILED
    logger.info(
        "G2 verify OK: %d path(s), %d valid, %d failed",
        result.n_total,
        result.n_valid,
        result.n_failed,
    )
    return 0


#: The handler for every ``g2`` subcommand; all are implemented.
G2_SUBCOMMAND_HANDLERS: dict[str, G2Handler] = {
    "prepare": _g2_prepare_handler,
    "run": _g2_run_handler,
    "verify": _g2_verify_handler,
}


def _add_common_options(parser: argparse.ArgumentParser, *, suppress_defaults: bool) -> None:
    """Attach common options to *parser*.

    Sub-parser levels use ``argparse.SUPPRESS`` defaults so a ``--config`` (or
    ``--log-level``/``--log-file``) given before the subcommand is not reset by
    the sub-parser's own default.
    """
    if suppress_defaults:
        config_default: Any = argparse.SUPPRESS
        log_level_default: Any = argparse.SUPPRESS
        log_file_default: Any = argparse.SUPPRESS
    else:
        config_default = None
        log_level_default = "INFO"
        log_file_default = None

    parser.add_argument(
        "--config",
        metavar="PATH",
        default=config_default,
        help="User YAML file deep-merged over config/defaults.yaml",
    )
    parser.add_argument(
        "--log-level",
        metavar="LEVEL",
        default=log_level_default,
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        metavar="PATH",
        default=log_file_default,
        help="Optional file to also write log records to",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the full ``pes2ts`` argument parser."""
    parser = argparse.ArgumentParser(prog=PROG, description="PES2TS data pipeline")
    _add_common_options(parser, suppress_defaults=False)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    top_subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    demo_parser = top_subparsers.add_parser(
        "demo", help="write the offline synthetic seven-contract demonstration bundle",
    )
    _add_common_options(demo_parser, suppress_defaults=True)
    demo_parser.add_argument("--output", default="examples/contracts_v1", metavar="PATH",
                             help="output directory (default: examples/contracts_v1)")
    demo_parser.set_defaults(handler=_demo_handler)
    acp_run_parser = top_subparsers.add_parser(
        "acp-run", help="execute one frozen ScanPlan attempt through the local ACP CLI",
    )
    _add_common_options(acp_run_parser, suppress_defaults=True)
    acp_run_parser.add_argument("--case", required=True, metavar="PATH", help="sealed ReactionCase JSON")
    acp_run_parser.add_argument("--review-record", required=True, metavar="PATH",
                                help="accepted ReviewRecord linked to this exact ready ReactionCase")
    acp_run_parser.add_argument("--plan", required=True, metavar="PATH", help="sealed ScanPlan JSON")
    acp_run_parser.add_argument("--acp-root", required=True, metavar="PATH", help="ACP source checkout")
    acp_run_parser.add_argument("--python", dest="python_executable", metavar="PATH",
                                help="Python executable from the configured ACP environment")
    acp_run_parser.add_argument("--acp-config", metavar="PATH", help="ACP QC/resource configuration")
    acp_run_parser.add_argument("--output-root", required=True, metavar="PATH",
                                help="PES2TS output root; each attempt gets a separate directory")
    acp_run_parser.add_argument("--execution-id", required=True, help="PES2TS execution identifier")
    acp_run_parser.add_argument("--attempt-id", required=True, help="immutable attempt identifier")
    acp_run_parser.add_argument("--candidate-id", help="frozen candidate ID (default: first candidate)")
    acp_run_parser.add_argument("--timeout", type=float, help="wall-time limit in seconds (cannot exceed plan budget)")
    acp_run_parser.add_argument("--nproc", type=int, help="ACP process core count")
    acp_run_parser.add_argument("--memory", help="ACP memory limit, e.g. 8GB")
    acp_run_parser.set_defaults(handler=_acp_run_handler)
    demo_run_parser = top_subparsers.add_parser(
        "acp-demo-run", help="execute a frozen train-first Demo cohort through the ACP CLI",
    )
    _add_common_options(demo_run_parser, suppress_defaults=True)
    demo_run_parser.add_argument("--manifest", required=True, metavar="PATH",
                                 help="frozen DemoExecutionManifest JSON")
    demo_run_parser.add_argument("--acp-root", required=True, metavar="PATH", help="ACP source checkout")
    demo_run_parser.add_argument("--python", dest="python_executable", metavar="PATH",
                                 help="Python executable from the configured ACP environment")
    demo_run_parser.add_argument("--acp-config", metavar="PATH", help="ACP QC/resource configuration")
    demo_run_parser.add_argument("--output-root", required=True, metavar="PATH",
                                 help="new or empty output directory for this immutable cohort run")
    demo_run_parser.add_argument("--include-valid", action="store_true",
                                 help="explicitly execute accepted valid-split cases for frozen evaluation")
    demo_run_parser.add_argument("--train-index", metavar="PATH",
                                 help="matching train-only DemoExecutionIndex.json required with --include-valid")
    demo_run_parser.add_argument("--nproc", type=int, help="ACP process core count")
    demo_run_parser.add_argument("--memory", help="ACP memory limit, e.g. 8GB")
    demo_run_parser.add_argument("--evaluation-split", choices=("train", "valid", "test", "unassigned"),
                                 default="valid")
    demo_run_parser.add_argument("--ranking-labels", metavar="PATH",
                                 help="bundle-relative JSON array of independently sourced frame labels")
    demo_run_parser.set_defaults(handler=_acp_demo_run_handler)
    validation_run_parser = top_subparsers.add_parser(
        "acp-validate-run",
        help="run ACP TS optimization/frequency then two-way IRC for one accepted proposal",
    )
    _add_common_options(validation_run_parser, suppress_defaults=True)
    validation_run_parser.add_argument("--case", required=True, metavar="PATH", help="reviewed ReactionCase JSON")
    validation_run_parser.add_argument("--review-record", required=True, metavar="PATH", help="accepted ReviewRecord JSON")
    validation_run_parser.add_argument("--path", required=True, metavar="PATH", help="usable PathBundle JSON")
    validation_run_parser.add_argument("--proposal", required=True, metavar="PATH", help="accepted SeedProposal JSON")
    validation_run_parser.add_argument("--source-frame-id", required=True)
    validation_run_parser.add_argument("--acp-root", required=True, metavar="PATH", help="ACP source checkout")
    validation_run_parser.add_argument("--python", dest="python_executable", metavar="PATH",
                                       help="Python executable from the ACP environment")
    validation_run_parser.add_argument("--acp-config", metavar="PATH", help="ACP QC/resource configuration")
    validation_run_parser.add_argument("--output-root", required=True, metavar="PATH", help="stage attempt output root")
    validation_run_parser.add_argument("--batch-execution-id", required=True)
    validation_run_parser.add_argument("--batch-attempt-id", required=True)
    validation_run_parser.add_argument("--irc-execution-id", required=True)
    validation_run_parser.add_argument("--irc-attempt-id", required=True)
    validation_run_parser.add_argument("--batch-item-id", default="pes2ts_ts")
    validation_run_parser.add_argument("--method", required=True, help="frozen OptTS/frequency/IRC method")
    validation_run_parser.add_argument("--basis", required=True, help="frozen basis; empty string for built-in basis")
    validation_run_parser.add_argument("--batch-timeout", type=float, required=True, help="BatchOptimize wall-time budget in seconds")
    validation_run_parser.add_argument("--irc-timeout", type=float, required=True, help="IRC wall-time budget in seconds")
    validation_run_parser.add_argument("--nproc", type=int)
    validation_run_parser.add_argument("--memory", help="ACP memory limit, e.g. 8GB")
    validation_run_parser.add_argument("--irc-maxpoints", type=int, default=100)
    validation_run_parser.add_argument("--irc-step", type=float, default=0.1)
    validation_run_parser.add_argument("--endpoint-tolerance", type=float, default=0.35)
    validation_run_parser.add_argument("--validation-id", required=True)
    validation_run_parser.add_argument("--output", required=True, metavar="PATH", help="ValidationResult JSON output")
    validation_run_parser.set_defaults(handler=_acp_validate_run_handler)
    validation_parser = top_subparsers.add_parser(
        "acp-collect-validation",
        help="verify ACP TS/frequency/IRC products and assemble one ValidationResult",
    )
    _add_common_options(validation_parser, suppress_defaults=True)
    validation_parser.add_argument("--case", required=True, metavar="PATH", help="reviewed ReactionCase JSON")
    validation_parser.add_argument("--review-record", required=True, metavar="PATH", help="accepted ReviewRecord JSON")
    validation_parser.add_argument("--path", required=True, metavar="PATH", help="usable PathBundle JSON")
    validation_parser.add_argument("--proposal", required=True, metavar="PATH", help="accepted SeedProposal JSON")
    validation_parser.add_argument("--batch-result", required=True, metavar="DIR", help="completed ACP BatchOptimize task directory")
    validation_parser.add_argument("--batch-item", required=True, help="TS-tagged BatchOptimize item ID")
    validation_parser.add_argument("--batch-execution-id", required=True)
    validation_parser.add_argument("--batch-attempt-id", required=True)
    validation_parser.add_argument("--irc-result", required=True, metavar="DIR", help="completed ACP IRC task directory")
    validation_parser.add_argument("--irc-execution-id", required=True)
    validation_parser.add_argument("--irc-attempt-id", required=True)
    validation_parser.add_argument("--method", required=True, help="frozen validation method")
    validation_parser.add_argument("--basis", required=True, help="frozen validation basis (empty string for built-in basis)")
    validation_parser.add_argument("--endpoint-tolerance", type=float, default=0.35,
                                    help="R/P endpoint aligned RMSD limit in Angstrom (default: 0.35)")
    validation_parser.add_argument("--validation-id", required=True)
    validation_parser.add_argument("--output", required=True, metavar="PATH", help="ValidationResult JSON output")
    validation_parser.set_defaults(handler=_acp_collect_validation_handler)
    g0_parser = top_subparsers.add_parser(
        "g0",
        help="G0: data entry, ground-truth quarantine, and split freezing",
    )
    _add_common_options(g0_parser, suppress_defaults=True)
    g0_subparsers = g0_parser.add_subparsers(
        dest="g0_command", required=True, metavar="SUBCOMMAND"
    )
    help_texts: dict[str, str] = {
        "fetch": "download and checksum-verify the configured source files",
        "inventory": "build the TS-free reactant/product inventory",
        "quarantine": "extract and relocate TS/IRC ground truth behind the audited accessor",
        "dedup": "detect exact duplicates and written reverses across the inventory",
        "audit": "run the mandatory DRFP near-duplicate cross-split leakage audit",
        "split": "adopt the authors' official train/valid/test split",
        "freeze": "freeze the split manifest under the explicit leak decision policy",
        "cohorts": "select the deterministic trial and stratified cohorts",
        "run-all": "run the full idempotent G0 pipeline and write the run report",
        "truth-index": "audited summary of the quarantined IRC index",
    }
    for name in G0_SUBCOMMANDS:
        sub_parser = g0_subparsers.add_parser(name, help=help_texts[name])
        _add_common_options(sub_parser, suppress_defaults=True)
        if name == "fetch":
            sub_parser.add_argument(
                "--force",
                action="store_true",
                help="Re-download even when the local MD5 already matches",
            )
        elif name == "truth-index":
            sub_parser.add_argument(
                "--reaction-id",
                metavar="RXN_ID",
                default=None,
                help="Reaction whose IRC frames to read (omitted: index summary)",
            )
        elif name == "freeze":
            sub_parser.add_argument(
                "--remediation",
                choices=REMEDIATION_CHOICES,
                default=REMEDIATION_NONE,
                help=(
                    "Leak remediation applied only when the audit found "
                    "leakage (default: none halts with LEAK_FOUND)"
                ),
            )
        elif name == "run-all":
            sub_parser.add_argument(
                "--data-root",
                metavar="PATH",
                default=None,
                help=(
                    "Override paths.data_root and rebuild every derived path "
                    "under it"
                ),
            )
            sub_parser.add_argument(
                "--force",
                action="store_true",
                help="Re-run every stage regardless of recorded input digests",
            )
            sub_parser.add_argument(
                "--skip-fetch",
                action="store_true",
                help="Verify existing sources instead of downloading them",
            )
            sub_parser.add_argument(
                "--remediation",
                choices=REMEDIATION_CHOICES,
                default=REMEDIATION_NONE,
                help=(
                    "Leak remediation passed to the freeze stage "
                    "(default: none halts with LEAK_FOUND)"
                ),
            )
        sub_parser.set_defaults(handler=SUBCOMMAND_HANDLERS[name])

    g1_parser = top_subparsers.add_parser(
        "g1",
        help="G1: authoritative bond changes and unified reaction indexes",
    )
    _add_common_options(g1_parser, suppress_defaults=True)
    g1_subparsers = g1_parser.add_subparsers(
        dest="g1_command", required=True, metavar="SUBCOMMAND"
    )
    g1_help: dict[str, str] = {
        "build": "build the per-reaction bond-change documents, summary, and manifest",
        "sample": "print the deterministic manual-check sample from the summary",
        "strata": "re-derive the authoritative strata and cohorts from a full build",
        "verify": "re-read the written documents of one stage and reconcile them",
        "join-audit": "write the P1 inventory/TS/IRC join audit (requires --allow-truth)",
        "resolve-map": "build the truth-assisted P1 TS/IRC mapping documents (requires --allow-truth)",
        "classify": "classify the P1 documents into the hierarchical reaction taxonomy",
        "gate": "write the G2 eligibility gate manifest and eligible id list",
        "v2-build": "freeze the v1 baseline and rebuild exclusive v2 edits with the audit",
        "v2-classify": "classify the v2 documents and write the v1->v2 migration report",
        "v2-verify": "recompute the v2 edit/class trees and the export whitelist",
        "v2-gate": "write scan_ready/needs_review/excluded manifests and the G2 export",
        "v2-sanitize-exports": "write a separate truth-clean copy of legacy G2 exports",
        "v2-scan-plan": "write the scan-strategy proposal tree, summary, and manifest",
        "v2-scan-verify": "re-read the scan-proposal tree and reconcile it with the manifest",
        "v2-scan-freeze": "gate scan proposals into frozen plans (refuses without eligible proposals)",
    }
    for name in G1_SUBCOMMANDS:
        sub_parser = g1_subparsers.add_parser(name, help=g1_help[name])
        _add_common_options(sub_parser, suppress_defaults=True)
        if name == "build":
            sub_parser.add_argument(
                "--cohort",
                choices=("trial", "stratified", "all"),
                default="all",
                help="Reaction subset to build (default: all inventory rows)",
            )
            sub_parser.add_argument(
                "--limit",
                type=int,
                metavar="N",
                default=None,
                help="Truncate the sorted cohort to its first N ids",
            )
        elif name == "sample":
            sub_parser.add_argument(
                "--category",
                metavar="NAME",
                default=None,
                help="Only print sample entries of this category",
            )
            sub_parser.add_argument(
                "--n",
                type=int,
                metavar="N",
                default=None,
                help="Per-category sample size (default: g1.sample_size)",
            )
        elif name == "verify":
            sub_parser.add_argument(
                "--stage",
                choices=VERIFY_STAGES,
                default="build",
                help="Which tree to verify (default: build, the classic G1 documents)",
            )
        elif name == "v2-gate":
            sub_parser.add_argument(
                "--assume-verified",
                action="store_true",
                help="Skip the built-in v2 verification (after an explicit `g1 v2-verify`)",
            )
        elif name == "v2-sanitize-exports":
            sub_parser.add_argument("--output-root", default=None, metavar="PATH",
                                    help="new destination (must not exist; default: sibling export_contracts_v1)")
            sub_parser.add_argument("--resume-staging", action="store_true",
                                    help="resume this command's existing .staging directory after checking it")
        elif name in ("join-audit", "resolve-map"):
            sub_parser.add_argument(
                "--allow-truth",
                action="store_true",
                help="Explicitly permit audited ground-truth reads for this run",
            )
            if name == "resolve-map":
                sub_parser.add_argument(
                    "--cohort",
                    choices=("trial", "stratified", "all"),
                    default="all",
                    help="Reaction subset to resolve (default: all inventory rows)",
                )
                sub_parser.add_argument(
                    "--limit",
                    type=int,
                    metavar="N",
                    default=None,
                    help="Truncate the sorted cohort to its first N ids",
                )
        sub_parser.set_defaults(handler=G1_SUBCOMMAND_HANDLERS[name])

    g2_parser = top_subparsers.add_parser(
        "g2",
        help="G2: GFN2-xTB reaction-path generation and verification",
    )
    _add_common_options(g2_parser, suppress_defaults=True)
    g2_subparsers = g2_parser.add_subparsers(
        dest="g2_command", required=True, metavar="SUBCOMMAND"
    )
    g2_help: dict[str, str] = {
        "prepare": "assemble and persist the deterministic endpoints for the selection",
        "run": "run the GFN2-xTB PATH step and write the per-reaction path artifacts",
        "verify": "re-read the G2 tree and reconcile summary, manifest, and documents",
    }
    for name in G2_SUBCOMMANDS:
        sub_parser = g2_subparsers.add_parser(name, help=g2_help[name])
        _add_common_options(sub_parser, suppress_defaults=True)
        if name in ("prepare", "run"):
            sub_parser.add_argument(
                "--reaction",
                metavar="RXN_ID",
                action="append",
                default=None,
                help="Restrict the batch to this reaction (repeatable)",
            )
            sub_parser.add_argument(
                "--cohort",
                choices=("trial", "stratified", "all"),
                default="trial",
                help="Reaction subset to select (default: trial cohort)",
            )
            sub_parser.add_argument(
                "--limit",
                type=int,
                metavar="N",
                default=None,
                help="Truncate the sorted selection to its first N ids",
            )
            sub_parser.add_argument(
                "--force",
                action="store_true",
                help="Re-run even when the terminal artifacts already exist",
            )
        sub_parser.set_defaults(handler=G2_SUBCOMMAND_HANDLERS[name])
    return parser


def _demo_handler(args: argparse.Namespace, _config: dict[str, Any]) -> int:
    """Write the local interface demonstration; never submits external jobs."""
    from pes2ts_core.demo import write_synthetic_bundle

    root = write_synthetic_bundle(args.output)
    print(f"synthetic contract bundle written: {root}")
    return 0


def _acp_run_handler(args: argparse.Namespace, _config: dict[str, Any]) -> int:
    """Execute and collect one ACP CLI attempt; never infer TS validation."""
    import json

    from pes2ts_core.contracts import ContractError, dumps_document, loads_document
    from pes2ts_core.integration.acp.adapter import scan_plan_to_acp_request
    from pes2ts_core.integration.acp.cli_backend import (
        ACPCLIBackend, ACPCLIError, cli_result_to_execution_record,
        collect_cli_path_bundle, write_cli_result_json,
    )
    from pes2ts_core.integration.acp.quality import apply_scan_path_quality

    try:
        case = loads_document(Path(args.case).read_text(encoding="utf-8"))
        review = loads_document(Path(args.review_record).read_text(encoding="utf-8"))
        plan = loads_document(Path(args.plan).read_text(encoding="utf-8"))
        source = case.get("source", {})
        review_output = review.get("extensions", {}).get("pes2ts.review_output.v1", {})
        if (case.get("status") != "ready" or review.get("schema_name") != "ReviewRecord"
                or review.get("status") != "accepted"
                or (review.get("reaction_id"), review.get("case_id"), review.get("dataset_version"), review.get("split"))
                   != (case.get("reaction_id"), case.get("case_id"), case.get("dataset_version"), case.get("split"))
                or source.get("review_record_id") != review.get("object_id")
                or source.get("reviewed_from_case_sha256") != review.get("case_sha256")
                or review_output.get("reviewed_case_sha256") != case.get("content_sha256")):
            raise ACPCLIError("calculation requires an accepted human review bound to this exact ready ReactionCase")
        preflight = scan_plan_to_acp_request(case, plan, args.candidate_id)
        candidate_id = preflight["metadata"]["candidate_id"]
        candidate = next(row for row in plan["candidates"] if row["candidate_id"] == candidate_id)
        frozen_timeout = float(candidate["budget"]["max_wall_seconds"])
        timeout = args.timeout if args.timeout is not None else frozen_timeout
        if timeout > frozen_timeout:
            raise ACPCLIError("requested timeout exceeds the frozen ScanPlan wall-time budget")
        backend = ACPCLIBackend(acp_root=args.acp_root, python_executable=args.python_executable,
                                config_path=args.acp_config)
        result = backend.run_scan(execution_id=args.execution_id, attempt_id=args.attempt_id,
            scan_request=preflight["scan_request"], output_root=args.output_root,
            timeout_seconds=timeout, nproc=args.nproc, memory=args.memory)
        attempt_root = Path(result.attempt_dir)
        audit_dir = attempt_root / "PES2TS"
        audit_dir.mkdir(parents=True, exist_ok=True)
        execution = cli_result_to_execution_record(case=case, plan=plan,
            candidate_id=candidate_id, result=result)
        (audit_dir / "ExecutionRecord.json").write_text(dumps_document(execution), encoding="utf-8")
        summary = {"execution_id": args.execution_id, "attempt_id": args.attempt_id,
                   "acp_task_id": None, "request_sha256": result.request_sha256,
                   "manifest_sha256": result.manifest_sha256, "process_status": result.status,
                   "returncode": result.returncode, "wall_seconds": result.wall_seconds,
                   "execution_record": "ExecutionRecord.json", "path_bundle": None,
                   "path_quality": None, "error": result.error}
        if result.status == "completed":
            path = collect_cli_path_bundle(output_dir=attempt_root, case=case, plan=plan,
                                           execution_id=args.execution_id, candidate_id=candidate_id)
            assessed = apply_scan_path_quality(plan=plan, execution=execution, path=path)
            path = assessed["path_bundle"]
            (audit_dir / "PathBundle.json").write_text(dumps_document(path), encoding="utf-8")
            (audit_dir / "PathQuality.json").write_text(
                json.dumps(assessed["quality_assessment"], ensure_ascii=False, sort_keys=True, indent=2),
                encoding="utf-8")
            summary["path_bundle"] = "PathBundle.json"
            summary["path_quality"] = path["status"]
        write_cli_result_json(result, audit_dir / "cli_attempt_summary.json")
        (audit_dir / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0 if result.status == "completed" else 1
    except (OSError, ValueError, ContractError, ACPCLIError) as exc:
        logging.getLogger(__name__).error("ACP CLI attempt failed: %s", exc)
        return 2


def _acp_demo_run_handler(args: argparse.Namespace, _config: dict[str, Any]) -> int:
    """Run the train-first frozen cohort and persist its index/metrics bundle."""
    import json

    from pes2ts_core.contracts import ContractError
    from pes2ts_core.demo_runner import run_demo_execution_manifest

    try:
        result = run_demo_execution_manifest(manifest_path=args.manifest,
            output_root=args.output_root, acp_root=args.acp_root,
            python_executable=args.python_executable, config_path=args.acp_config,
            nproc=args.nproc, memory=args.memory, include_valid=args.include_valid,
            train_index_path=args.train_index, evaluation_split=args.evaluation_split,
            ranking_labels_path=args.ranking_labels)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (OSError, ValueError, ContractError) as exc:
        logging.getLogger(__name__).error("ACP Demo cohort failed: %s", exc)
        return 2


def _acp_collect_validation_handler(args: argparse.Namespace, _config: dict[str, Any]) -> int:
    """Verify existing ACP stage artifacts and persist the derived result."""
    import json

    from pes2ts_core.contracts import ContractError, loads_document
    from pes2ts_core.integration.acp.cli_backend import ACPCLIError
    from pes2ts_core.integration.acp.stage_results import collect_acp_validation_result
    from pes2ts_core.utils.jsonio import write_json

    try:
        case = loads_document(Path(args.case).read_text(encoding="utf-8"))
        review = loads_document(Path(args.review_record).read_text(encoding="utf-8"))
        path = loads_document(Path(args.path).read_text(encoding="utf-8"))
        proposal = loads_document(Path(args.proposal).read_text(encoding="utf-8"))
        result = collect_acp_validation_result(
            validation_id=args.validation_id, case=case, review_record=review,
            path=path, proposal=proposal, batch_task_root=args.batch_result,
            batch_item_id=args.batch_item, batch_execution_id=args.batch_execution_id,
            batch_attempt_id=args.batch_attempt_id, irc_task_root=args.irc_result,
            irc_execution_id=args.irc_execution_id, irc_attempt_id=args.irc_attempt_id,
            expected_method=args.method, expected_basis=args.basis,
            endpoint_tolerance_angstrom=args.endpoint_tolerance)
        write_json(args.output, result)
        print(json.dumps({"validation_id": result["object_id"],
                          "status": result["status"], "output": str(Path(args.output))},
                         ensure_ascii=False, sort_keys=True))
        return 0
    except (OSError, ValueError, ContractError, ACPCLIError) as exc:
        logging.getLogger(__name__).error("ACP validation collection failed: %s", exc)
        return 2


def _acp_validate_run_handler(args: argparse.Namespace, _config: dict[str, Any]) -> int:
    """Run and verify one selected proposal through ACP OptTS/Freq and IRC."""
    import json

    from pes2ts_core.contracts import ContractError, loads_document
    from pes2ts_core.integration.acp.cli_backend import ACPCLIError
    from pes2ts_core.integration.acp.stage_cli import ACPValidationCLIBackend
    from pes2ts_core.integration.acp.stage_results import collect_acp_validation_result
    from pes2ts_core.utils.jsonio import write_json

    try:
        case = loads_document(Path(args.case).read_text(encoding="utf-8"))
        review = loads_document(Path(args.review_record).read_text(encoding="utf-8"))
        path = loads_document(Path(args.path).read_text(encoding="utf-8"))
        proposal = loads_document(Path(args.proposal).read_text(encoding="utf-8"))
        backend = ACPValidationCLIBackend(acp_root=args.acp_root,
            python_executable=args.python_executable, config_path=args.acp_config)
        stage = backend.run_validation(case=case, review_record=review, path=path,
            proposal=proposal, source_frame_id=args.source_frame_id,
            validation_id=args.validation_id,
            expected_method=args.method, expected_basis=args.basis,
            output_root=args.output_root, batch_execution_id=args.batch_execution_id,
            batch_attempt_id=args.batch_attempt_id, irc_execution_id=args.irc_execution_id,
            irc_attempt_id=args.irc_attempt_id, batch_item_id=args.batch_item_id,
            batch_timeout_seconds=args.batch_timeout, irc_timeout_seconds=args.irc_timeout,
            nproc=args.nproc, memory=args.memory, irc_maxpoints=args.irc_maxpoints,
            irc_step=args.irc_step, endpoint_tolerance_angstrom=args.endpoint_tolerance)
        result = stage.get("validation_result")
        if result is None:
            result = collect_acp_validation_result(validation_id=args.validation_id,
                case=case, review_record=review, path=path, proposal=proposal,
                batch_task_root=stage["batch"].task_root, batch_item_id=args.batch_item_id,
                batch_execution_id=args.batch_execution_id, batch_attempt_id=args.batch_attempt_id,
                irc_task_root=stage["irc"].task_root, irc_execution_id=args.irc_execution_id,
                irc_attempt_id=args.irc_attempt_id, expected_method=args.method,
                expected_basis=args.basis, endpoint_tolerance_angstrom=args.endpoint_tolerance)
        result = _add_stage_receipt_extension(result, stage, args.output_root)
        write_json(args.output, result)
        print(json.dumps({"validation_id": result["object_id"],
            "status": result["status"], "output": str(Path(args.output)),
            "batch_attempt": stage["batch"].attempt_id,
            "irc_attempt": stage["irc"].attempt_id if stage["irc"] else None},
            ensure_ascii=False, sort_keys=True))
        return 0 if result["status"] in {"passed", "failed"} else 1
    except (OSError, ValueError, ContractError, ACPCLIError) as exc:
        logging.getLogger(__name__).error("ACP validation run failed: %s", exc)
        return 2


def _add_stage_receipt_extension(result: dict[str, Any], stage: dict[str, Any],
                                 output_root: str | Path) -> dict[str, Any]:
    """Bind local stage receipts and wall-time accounting into ValidationResult."""
    from pes2ts_core.contracts import ContractError, seal_document

    root = Path(output_root).expanduser().resolve()
    receipts = []
    for name in ("batch", "irc"):
        item = stage.get(name)
        if item is None:
            continue
        task_root = Path(item.task_root).resolve()
        try:
            task_ref = task_root.relative_to(root).as_posix()
        except ValueError as exc:
            raise ContractError("ACP validation attempt escaped its output root") from exc
        receipts.append({"workflow": item.workflow, "execution_id": item.execution_id,
            "attempt_id": item.attempt_id, "status": item.status,
            "returncode": item.returncode, "wall_seconds": item.wall_seconds,
            "cpu_seconds": None, "request_sha256": item.request_sha256,
            "result_manifest_sha256": item.manifest_sha256,
            "task_ref": task_ref, "receipt_ref": f"{task_ref}/WORK/pes2ts/stage_cli_receipt.json",
            "log_ref": f"{task_ref}/WORK/pes2ts/acp_cli.log", "error": item.error})
    extensions = dict(result.get("extensions", {}))
    extensions["pes2ts.acp_stage_receipts.v1"] = {
        "schema": "pes2ts_acp_stage_receipts_v1",
        "wall_time_unit": "second", "cpu_time_available": False,
        "attempts": receipts,
    }
    return seal_document({**result, "extensions": extensions})


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_logging(
        level=getattr(args, "log_level", "INFO") or "INFO",
        log_file=getattr(args, "log_file", None),
    )
    logger = logging.getLogger(__name__)

    try:
        config = load_config(getattr(args, "config", None))
    except ConfigError as exc:
        logger.error("Failed to load configuration: %s", exc)
        return EXIT_CONFIG_ERROR

    handler: G0Handler = args.handler
    return handler(args, config)
