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
cohorts from a full-build summary; and ``verify`` re-reads the written tree and
reconciles it with the summary and manifest.
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
from pes2ts_core.g0.neardup import (
    EXIT_AUDIT_BUDGET_EXCEEDED,
    AuditBudgetExceeded,
    compute_fingerprints,
    cross_split_leak_audit,
)
from pes2ts_core.g0.pipeline import (
    EXIT_PIPELINE_FAILED,
    run_pipeline,
    with_data_root,
)
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
from pes2ts_core.g1.strata_auth import rebuild_authoritative_strata
from pes2ts_core.g1.verify import verify_g1
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
G1_SUBCOMMANDS: tuple[str, ...] = ("build", "sample", "strata", "verify")

G0Handler = Callable[[argparse.Namespace, dict[str, Any]], int]
G1Handler = Callable[[argparse.Namespace, dict[str, Any]], int]


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


def _g1_verify_handler(_args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Re-read every written document; exit 22 on any reconciliation problem."""
    logger = logging.getLogger(__name__)
    try:
        result = verify_g1(config)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return EXIT_CHECKSUM_MISMATCH
    except ValueError as exc:
        logger.error("G1 verify failed: %s", exc)
        return EXIT_G1_BUILD_FAILED
    if result.problems:
        for problem in result.problems[:10]:
            logger.error("G1 verify: %s", problem)
        logger.error("G1 verify failed with %d problem(s)", len(result.problems))
        return EXIT_G1_BUILD_FAILED
    logger.info(
        "G1 verify OK: %d document(s), %d valid, %d rejected",
        result.n_total,
        result.n_valid,
        result.n_rejected,
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
        "verify": "re-read every written document and reconcile it with the manifest",
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
        sub_parser.set_defaults(handler=G1_SUBCOMMAND_HANDLERS[name])
    return parser


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
