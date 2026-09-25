"""Command-line interface for PES2TS.

Usage::

    pes2ts [--config PATH] [--log-level LEVEL] [--log-file PATH] g0 <SUBCOMMAND>

``g0 fetch`` downloads and checksum-verifies the configured Reaction-QM
source files.  The remaining ``g0`` subcommands are stubs in this scaffold
release: they parse their arguments, load the (merged) configuration, log that
they are not implemented yet, and exit with a subcommand-specific non-zero
code.  ``--help`` works for every subcommand and exits 0.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Sequence
from typing import Any

from pes2ts_core.config_loader import EXIT_CONFIG_ERROR, ConfigError, load_config
from pes2ts_core.g0.fetch import (
    EXIT_CHECKSUM_MISMATCH,
    ChecksumMismatch,
    fetch_sources,
    verify_sources,
    write_source_manifest,
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


#: Populated lazily so tests can import the module without side effects.
SUBCOMMAND_HANDLERS: dict[str, G0Handler] = {
    name: _make_stub(name) for name in G0_SUBCOMMANDS
}

#: ``g0 fetch`` is implemented; the remaining entries are still stubs.
SUBCOMMAND_HANDLERS["fetch"] = _fetch_handler


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
