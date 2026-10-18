"""Explicit subcommand router for the predictor product.

Offline commands (``inventory``, ``split``, ``build-teacher``, ``train``,
``evaluate``, ``export``) route to :mod:`pes2ts_core.pesn.offline.cli` with
lazy imports, so the offline toolchain never enters the runtime import chain.
``predict`` is the runtime-side entry and loads a model bundle.

Exit codes (predictor-owned, documented here):
  0  success
  2  configuration error (unreadable/invalid YAML, unknown keys)
  3  artifact/checksum error (missing or hash-mismatched upstream artifact)
  30 input or contract rejection (typed)
  31 optional ML backend missing (typed unsupported, never a crash)
  32 pipeline failure (stage did not complete)
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

EXIT_OK = 0
EXIT_CONFIG_ERROR = 2
EXIT_CHECKSUM_ERROR = 3
EXIT_CONTRACT_REJECT = 30
EXIT_BACKEND_MISSING = 31
EXIT_PIPELINE_FAILED = 32


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pes2ts_core.pesn",
        description="PES-N predictor product entry (research scope, G3-P1)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_inventory = sub.add_parser(
        "inventory", help="offline: source/metadata join with typed qualification"
    )
    p_inventory.add_argument("--config", required=True, help="predictor YAML config path")
    p_inventory.add_argument("--output-root", default=None, help="override output root")

    p_split = sub.add_parser("split", help="offline: canonical grouping + freeze split")
    p_split.add_argument("--config", required=True)
    p_split.add_argument("--output-root", default=None)

    p_build = sub.add_parser(
        "build-teacher", help="offline: scoped audited teacher dataset build"
    )
    p_build.add_argument("--config", required=True)
    p_build.add_argument("--output-root", default=None)
    p_build.add_argument(
        "--allow-truth",
        action="store_true",
        help="explicit audited ground-truth access flag (required)",
    )

    p_train = sub.add_parser("train", help="offline: train the teacher model")
    p_train.add_argument("--config", required=True)
    p_train.add_argument("--output-root", default=None)
    p_train.add_argument("--resume", default=None, help="run_id to resume")

    p_eval = sub.add_parser("evaluate", help="offline: frozen evaluation of a run/bundle")
    p_eval.add_argument("--config", required=True)
    p_eval.add_argument("--output-root", default=None)

    p_export = sub.add_parser("export", help="offline: whitelist runtime bundle export")
    p_export.add_argument("--config", required=True)
    p_export.add_argument("--output-root", default=None)

    p_predict = sub.add_parser("predict", help="runtime: predict with a model bundle")
    p_predict.add_argument("--bundle", required=True)
    p_predict.add_argument("--input", required=True, help="PESNInput JSON path")
    p_predict.add_argument("--output", default=None)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    from pes2ts_core.pesn.offline import cli as offline_cli

    handlers = {
        "inventory": offline_cli.cmd_inventory,
        "split": offline_cli.cmd_split,
        "build-teacher": offline_cli.cmd_build_teacher,
        "train": offline_cli.cmd_train,
        "evaluate": offline_cli.cmd_evaluate,
        "export": offline_cli.cmd_export,
    }
    if args.command == "predict":
        from pes2ts_core.pesn.inference import cli as inference_cli

        return inference_cli.cmd_predict(args)
    handler = handlers[args.command]
    result = handler(args)
    return int(result)


if __name__ == "__main__":
    sys.exit(main())
