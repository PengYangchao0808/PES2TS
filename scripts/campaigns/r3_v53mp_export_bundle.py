"""V5.3-MP S3 deliverable CLI: campaign-disciplined bundle export.

Wraps ``pes2ts_core.pesn.offline.publishing.export.export_move_process_bundle``
(contract §6 ``pes2ts_move_process_bundle_v1``) for ONE selected V5.3-MP run:

1. run_id parse — the run driver's ``_parse_run_id`` discipline
   (``v53mp_{tier}_{arm}_s{seed}``; ValueError -> exit 2);
2. campaign-root discipline — ``_assert_v53mp_root`` on BOTH the output root
   and the output dir (must normalize under ``outputs/pesn/v53mp/**``;
   ``CampaignRootError`` -> exit 1);
3. **ledger binding** — the run's LATEST ``run_records.jsonl`` entry must be
   ``status == "done"`` and its ``checkpoint_sha256`` must equal metrics.json's
   ``checkpoint_best_sha256``; any drift refuses the export (no export from
   unverified runs);
4. ``tau_end = metrics.endpoint_head.tau_end`` (must lie in [0, 1]); arm A/B
   exports are allowed but carry a LOUD stderr note that their endpoint head
   is untrained — arm C is the expected export arm;
5. metrics payload = the FULL ``metrics.json`` dict (the exporter
   JSON-serializes it into the bundle);
6. provenance = the driver's ``_build_training_provenance(...)``, which
   returns every ``MOVE_PROCESS_REQUIRED_PROVENANCE`` key plus extras;
7. post-export whitelist: files in ``output_dir`` must EQUAL
   ``MOVE_PROCESS_BUNDLE_FILES`` exactly — extra/missing names refuse with the
   drift spelled out; then the bundle path + ``checksums.json`` sha256 entries
   are printed;
8. exit codes: 0 ok · 1 refused (ledger / checkpoint / tau / campaign-root /
   whitelist / export error) · 2 usage / parse (argparse, bad run_id, bad
   campaign config — mirroring the run driver's config handling).

Fail-closed throughout: every refusal happens BEFORE the export call unless it
is the whitelist post-check itself, and nothing outside
``outputs/pesn/v53mp/**`` is ever created.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.offline.publishing.export import (  # noqa: E402
    MOVE_PROCESS_BUNDLE_FILES,
    ExportError,
    export_move_process_bundle,
)
from scripts.agent_ops.supervisor.campaign_root import (  # noqa: E402
    CampaignRootError,
)
from scripts.campaigns.r3_v53mp_run import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_SPLIT_BASENAME,
    DEFAULT_TABLES_DIR,
    DEFAULT_V53MP_ROOT,
    V53MPConfigError,
    _assert_v53mp_root,
    _build_training_provenance,
    _load_campaign_config,
    _parse_run_id,
)

#: ledger lives under the output root (same resolution as the run driver)
LEDGER_BASENAME = "run_records.jsonl"
#: C is the only arm whose endpoint head was trained (lambda_endpoint > 0)
EXPORT_ARM = "C"


class ExportBundleRefusal(RuntimeError):
    """Typed fail-closed refusal (ledger / metrics / tau / whitelist)."""


# --------------------------------------------------------------------------
# fail-closed readers
# --------------------------------------------------------------------------


def _read_json(path: Path, what: str) -> Dict[str, Any]:
    """Read a JSON object or refuse with a typed error (never returns junk)."""
    if not path.is_file():
        raise ExportBundleRefusal("{}_MISSING: {}".format(what, path))
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ExportBundleRefusal("{}_UNREADABLE: {}: {}".format(what, path, exc)) from exc
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExportBundleRefusal("{}_CORRUPT: {}: {}".format(what, path, exc)) from exc
    if not isinstance(doc, dict):
        raise ExportBundleRefusal("{}_CORRUPT: {} is not a JSON object".format(what, path))
    return doc


def _latest_ledger_entry(ledger_path: Path, run_id: str) -> Dict[str, Any]:
    """Last JSONL line bound to ``run_id`` (append-only: file order = time)."""
    if not ledger_path.is_file():
        raise ExportBundleRefusal("LEDGER_MISSING: {}".format(ledger_path))
    try:
        lines = ledger_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ExportBundleRefusal(
            "LEDGER_UNREADABLE: {}: {}".format(ledger_path, exc)
        ) from exc
    latest: Optional[Dict[str, Any]] = None
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExportBundleRefusal(
                "LEDGER_CORRUPT: {} line {}: {}".format(ledger_path, lineno, exc)
            ) from exc
        if not isinstance(entry, dict):
            raise ExportBundleRefusal(
                "LEDGER_CORRUPT: {} line {} is not a JSON object".format(
                    ledger_path, lineno
                )
            )
        # v53mp records carry the human-readable id in plan_run_id (the
        # ledger's own run_id is the computed/hash scheme — accept either).
        if entry.get("plan_run_id") == run_id or entry.get("run_id") == run_id:
            latest = entry
    if latest is None:
        raise ExportBundleRefusal(
            "LEDGER_NO_ENTRY: no ledger entry for run_id {!r} in {}".format(
                run_id, ledger_path
            )
        )
    return latest


def _assert_ledger_binding(
    entry: Dict[str, Any], metrics: Dict[str, Any], run_id: str
) -> None:
    """Latest entry must be ``done`` and pin the SAME checkpoint as metrics."""
    status = entry.get("status")
    if status != "done":
        raise ExportBundleRefusal(
            "LEDGER_STATUS_NOT_DONE: latest ledger entry for {!r} has status "
            "{!r} (need 'done'; no export from unverified runs)".format(
                run_id, status
            )
        )
    ledger_sha = entry.get("checkpoint_sha256")
    metrics_sha = metrics.get("checkpoint_best_sha256")
    if not (isinstance(metrics_sha, str) and metrics_sha.strip()):
        raise ExportBundleRefusal(
            "METRICS_CHECKPOINT_SHA_MISSING: {} metrics.json has no usable "
            "checkpoint_best_sha256 ({!r})".format(run_id, metrics_sha)
        )
    if ledger_sha != metrics_sha:
        raise ExportBundleRefusal(
            "LEDGER_CHECKPOINT_MISMATCH: {} ledger checkpoint_sha256={} but "
            "metrics checkpoint_best_sha256={} (no export from unverified "
            "runs)".format(run_id, ledger_sha, metrics_sha)
        )


def _extract_tau_end(metrics: Dict[str, Any], run_id: str) -> float:
    endpoint = metrics.get("endpoint_head")
    if not isinstance(endpoint, dict) or "tau_end" not in endpoint:
        raise ExportBundleRefusal(
            "TAU_END_MISSING: {} metrics.json has no endpoint_head.tau_end".format(
                run_id
            )
        )
    raw = endpoint["tau_end"]
    try:
        tau = float(raw)
    except (TypeError, ValueError) as exc:
        raise ExportBundleRefusal(
            "TAU_END_INVALID: {} endpoint_head.tau_end={!r} must be a number "
            "in [0, 1]".format(run_id, raw)
        ) from exc
    if not math.isfinite(tau) or not (0.0 <= tau <= 1.0):
        raise ExportBundleRefusal(
            "TAU_END_OUT_OF_RANGE: {} endpoint_head.tau_end={!r} must lie in "
            "[0, 1]".format(run_id, raw)
        )
    return tau


def _assert_bundle_whitelist(bundle: Path) -> None:
    """Post-check: the bundle must contain EXACTLY the whitelist (spec 7)."""
    if not bundle.is_dir():
        raise ExportBundleRefusal("BUNDLE_DIR_MISSING: {}".format(bundle))
    present = {p.name for p in bundle.iterdir() if p.is_file()}
    expected = set(MOVE_PROCESS_BUNDLE_FILES)
    if present != expected:
        raise ExportBundleRefusal(
            "BUNDLE_WHITELIST_DRIFT: {} must contain exactly {}; "
            "unexpected={} missing={}".format(
                bundle,
                list(MOVE_PROCESS_BUNDLE_FILES),
                sorted(present - expected),
                sorted(expected - present),
            )
        )


# --------------------------------------------------------------------------
# export command
# --------------------------------------------------------------------------


def cmd_export(args: argparse.Namespace) -> int:
    # -- 1. run_id parse (ValueError -> exit 2 via main) --------------------
    tier, arm, seed = _parse_run_id(args.run_id)

    # -- 2. campaign-root discipline (CampaignRootError -> exit 1) ----------
    _assert_v53mp_root(args.output_root)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.output_root) / "exports" / "{}_bundle".format(args.run_id)
    )
    _assert_v53mp_root(output_dir)

    # -- 3. loud note: only arm C has a trained endpoint head ---------------
    if arm != EXPORT_ARM:
        sys.stderr.write(
            "WARNING: {} is arm {} — its endpoint head is UNTRAINED "
            "(lambda_endpoint=0), so endpoint_head.tau_end is diagnostic only; "
            "export proceeds anyway, but arm {} is the expected export arm.\n"
            .format(args.run_id, arm, EXPORT_ARM)
        )

    # -- 4. run outputs (fail-closed: never export an unverified run) -------
    run_dir = Path(args.output_root) / args.run_id
    if not run_dir.is_dir():
        raise ExportBundleRefusal("RUN_DIR_MISSING: {}".format(run_dir))
    checkpoint = run_dir / "checkpoint_best.pt"
    if not checkpoint.is_file():
        raise ExportBundleRefusal("CHECKPOINT_MISSING: {}".format(checkpoint))
    metrics = _read_json(run_dir / "metrics.json", "METRICS")

    # -- 5. ledger binding ---------------------------------------------------
    ledger_path = Path(args.output_root) / LEDGER_BASENAME
    entry = _latest_ledger_entry(ledger_path, args.run_id)
    _assert_ledger_binding(entry, metrics, args.run_id)

    # -- 6. tau_end ----------------------------------------------------------
    tau_end = _extract_tau_end(metrics, args.run_id)

    # -- 7. provenance (same builder + same paths as the training driver) ---
    config_path = Path(args.config)
    try:
        config = _load_campaign_config(config_path)
    except V53MPConfigError as exc:
        # bad --config / config schema = usage (exit 2), mirrors the driver
        sys.stderr.write("error: {}\n".format(exc))
        return 2
    tables_dir = Path(args.tables_dir)
    split_path = Path(args.split) if args.split else tables_dir / DEFAULT_SPLIT_BASENAME
    try:
        provenance = _build_training_provenance(
            config, config_path, tables_dir, split_path,
            run_id=args.run_id, tier=tier,
        )
    except V53MPConfigError as exc:
        # missing/unhashable provenance artifacts = export refusal (exit 1)
        raise ExportBundleRefusal(
            "PROVENANCE_REFUSED: {}".format(exc)
        ) from exc

    model_id = args.model_id or "pes2ts-v53mp-{}-{}-s{}".format(arm, tier, seed)

    # -- 8. export -----------------------------------------------------------
    bundle = export_move_process_bundle(
        run_dir=run_dir,
        output_dir=output_dir,
        model_id=model_id,
        tau_end=tau_end,
        metrics=metrics,          # the FULL metrics.json payload
        provenance=provenance,
    )

    # -- 9. whitelist post-check + checksum printout -------------------------
    _assert_bundle_whitelist(bundle)
    checksums = _read_json(bundle / "checksums.json", "CHECKSUMS")
    print("bundle: {}".format(bundle))
    for name in sorted(checksums):
        print("{}  {}".format(checksums[name], name))
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="r3_v53mp_export_bundle",
        description="V5.3-MP S3 deliverable: export the MOVE+PROCESS whitelist "
                    "bundle for ONE verified run (fail-closed)",
    )
    parser.add_argument(
        "--run-id", required=True,
        help="v53mp_<2k|5k|20k>_<A|B|C>_s<seed>",
    )
    parser.add_argument("--output-root", default=DEFAULT_V53MP_ROOT)
    parser.add_argument(
        "--output-dir", default=None,
        help="default: <output-root>/exports/<run-id>_bundle "
             "(must stay under outputs/pesn/v53mp/**)",
    )
    parser.add_argument(
        "--model-id", default=None,
        help="default: pes2ts-v53mp-<arm>-<tier>-s<seed>",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--tables-dir", default=DEFAULT_TABLES_DIR)
    parser.add_argument(
        "--split", default=None,
        help="default: <tables-dir>/" + DEFAULT_SPLIT_BASENAME,
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return cmd_export(args)
    except ValueError as exc:
        # _parse_run_id: bad run_id syntax -> usage/parse
        sys.stderr.write("error: {}\n".format(exc))
        return 2
    except CampaignRootError as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 1
    except ExportBundleRefusal as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 1
    except ExportError as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 1
    except V53MPConfigError as exc:
        # config / provenance artifact refusal (mirrors the run driver's exit 2)
        sys.stderr.write("error: {}\n".format(exc))
        return 2
    except OSError as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
