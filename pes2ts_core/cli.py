"""Command-line interface for PES2TS.

Usage::

    pes2ts [--config PATH] [--log-level LEVEL] [--log-file PATH] g0 <SUBCOMMAND>

``g0 fetch`` downloads and checksum-verifies the configured Reaction-QM
source files; ``g0 quarantine`` extracts and relocates the TS/IRC ground truth
and writes its manifest; ``g0 truth-index`` is the audited accessor for the
resulting IRC index.  The remaining ``g0`` subcommands are still stubs: they
parse their arguments, load the (merged) configuration, log that they are not
implemented yet, and exit with a subcommand-specific non-zero code.
``--help`` works for every subcommand and exits 0.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

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
from pes2ts_core.logging_setup import setup_logging
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

#: Each stub exits with its own code so callers can tell which stage ran.
NOT_IMPLEMENTED_EXIT_CODES: dict[str, int] = {
    name: 10 + index for index, name in enumerate(G0_SUBCOMMANDS)
}

G0Handler = Callable[[argparse.Namespace, dict[str, Any]], int]


def _make_stub(name: str) -> G0Handler:
    """Build the placeholder handler for g0 subcommand *name*."""
    exit_code = NOT_IMPLEMENTED_EXIT_CODES[name]

    def _handler(_args: argparse.Namespace, _config: dict[str, Any]) -> int:
        logging.getLogger(__name__).warning("g0 %s: not implemented yet", name)
        return exit_code

    _handler.__name__ = f"{name.replace('-', '_')}_handler"
    return _handler


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
    remediation = str(getattr(args, "remediation", REMEDIATION_NONE))
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


#: Populated lazily so tests can import the module without side effects.
SUBCOMMAND_HANDLERS: dict[str, G0Handler] = {
    name: _make_stub(name) for name in G0_SUBCOMMANDS
}

#: Implemented handlers; the remaining entries are still stubs.
SUBCOMMAND_HANDLERS["fetch"] = _fetch_handler
SUBCOMMAND_HANDLERS["inventory"] = _g0_inventory_handler
SUBCOMMAND_HANDLERS["dedup"] = _g0_dedup_handler
SUBCOMMAND_HANDLERS["audit"] = _g0_audit_handler
SUBCOMMAND_HANDLERS["split"] = _g0_split_handler
SUBCOMMAND_HANDLERS["freeze"] = _g0_freeze_handler
SUBCOMMAND_HANDLERS["cohorts"] = _g0_cohorts_handler
SUBCOMMAND_HANDLERS["quarantine"] = _g0_quarantine_handler
SUBCOMMAND_HANDLERS["truth-index"] = _g0_truth_index_handler


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
    for name in G0_SUBCOMMANDS:
        if name == "fetch":
            help_text = "download and checksum-verify the configured source files"
        elif name == "quarantine":
            help_text = "extract and relocate TS/IRC ground truth behind the audited accessor"
        elif name == "truth-index":
            help_text = "audited summary of the quarantined IRC index"
        elif name == "split":
            help_text = "adopt the authors' official train/valid/test split"
        elif name == "audit":
            help_text = "run the mandatory DRFP near-duplicate cross-split leakage audit"
        elif name == "freeze":
            help_text = "freeze the split manifest under the explicit leak decision policy"
        elif name == "cohorts":
            help_text = "select the deterministic trial and stratified cohorts"
        else:
            help_text = f"g0 {name} (not implemented yet)"
        sub_parser = g0_subparsers.add_parser(name, help=help_text)
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
        sub_parser.set_defaults(handler=SUBCOMMAND_HANDLERS[name])
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
