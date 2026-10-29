#!/usr/bin/env python3
"""V5.4.1 WP7 deliverable CLI: embedding champion bundle export.

Wraps ``pes2ts_core.pesn.offline.publishing.export.export_embedding_bundle``
(plan §12.3 ``pes2ts_embedding_bundle_v1``) for the single frozen V5.4.1
champion (E*, which may be E0).  This is a NEW entry on purpose — the B6
``export_bundle`` and the V5.3-MP ``export_move_process_bundle`` (and their
CLI ``r3_v53mp_export_bundle.py``) are untouched.

Pipeline (fail-closed throughout; nothing is written before every audit
passes):

1. ``--run-id`` must use the ``V541_`` namespace (D05);
2. campaign-root discipline — ``assert_v541_root`` on BOTH the output root
   and the output dir (must normalize under ``outputs/pesn/v541_embed_50k/**``);
3. **ledger binding** — the run's LATEST ``run_records.jsonl`` entry must
   carry ``outcome == "done"`` and a ``checkpoint_best_sha256`` equal to the
   sha256 of ``<run_dir>/checkpoint_best.pt``; any drift refuses the export;
4. the champion metrics must agree (``embedding_id`` / ``run_outcome``);
5. provenance / benchmark summary are supplied as JSON files
   (``--provenance`` / ``--benchmark-summary``) — the WP7 evidence blocks
   (three seeds, screen, dev exposure, unread confirm, cost, failures and
   censoring, unclosed capacity diagnostics) are NEVER invented here; the
   exporter additionally re-audits every required key, the three-seed
   coverage, ``confirm_read=false`` and the checkpoint binding;
6. ``--feature-schema`` defaults to the built unified schema
   (node 48 / legacy edge 22 / chem+spatial+angle recipes / tie / mask /
   eps / cache fingerprint), overridable for audit replay;
7. ``--embedding-config`` defaults to the registry row + residual +
   core-consumption + feature-recipe slice of the frozen campaign yaml;
8. post-export whitelist + SHA re-audit
   (``verify_embedding_bundle``), then the bundle path and every
   ``checksums.json`` digest are printed.

Exit codes: ``0`` ok · ``1`` refused (ledger / checkpoint / root / schema /
export error) · ``2`` usage / parse (argparse, bad run_id, unreadable
config or evidence JSON).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.offline.publishing.export import (  # noqa: E402
    EMBEDDING_BUNDLE_FILES,
    EMBEDDING_REQUIRED_PROVENANCE,
    EmbeddingBundleError,
    ExportError,
    build_embedding_feature_schema,
    export_embedding_bundle,
    verify_embedding_bundle,
)
from scripts.agent_ops.supervisor.campaign_root import (  # noqa: E402
    CampaignRootError,
)
from scripts.agent_ops.v541_config_schema import (  # noqa: E402
    DEFAULT_CONFIG,
    V541SchemaError,
    load_config,
)
from scripts.campaigns.v541_run import (  # noqa: E402
    DEFAULT_ROOT,
    V541RunnerError,
    assert_v541_root,
    run_records_path,
)

#: v541 ledger filename (owned by v541_runplan; imported via v541_run's
#: module namespace so the two can never disagree).
LEDGER_BASENAME = "run_records.jsonl"
#: formal run dirs live under <root>/runs/<run_id> (mode_out_dir formal).
FORMAL_RUN_SUBDIR = "runs"


class ExportBundleRefusal(RuntimeError):
    """Typed fail-closed refusal (ledger / metrics / root / evidence)."""


def _read_json(path: Path, what: str) -> Dict[str, Any]:
    """Read a JSON object or refuse with a typed error (never returns junk)."""
    if not path.is_file():
        raise ExportBundleRefusal("{}_MISSING: {}".format(what, path))
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ExportBundleRefusal(
            "{}_UNREADABLE: {}: {}".format(what, path, exc)) from exc
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExportBundleRefusal(
            "{}_CORRUPT: {}: {}".format(what, path, exc)) from exc
    if not isinstance(doc, dict):
        raise ExportBundleRefusal(
            "{}_CORRUPT: {} is not a JSON object".format(what, path))
    return doc


def _latest_ledger_entry(ledger_path: Path, run_id: str) -> Dict[str, Any]:
    """Last JSONL line bound to ``run_id`` (append-only: file order = time)."""
    if not ledger_path.is_file():
        raise ExportBundleRefusal("LEDGER_MISSING: {}".format(ledger_path))
    try:
        lines = ledger_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ExportBundleRefusal(
            "LEDGER_UNREADABLE: {}: {}".format(ledger_path, exc)) from exc
    latest: Optional[Dict[str, Any]] = None
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExportBundleRefusal(
                "LEDGER_CORRUPT: {} line {}: {}".format(
                    ledger_path, lineno, exc)) from exc
        if not isinstance(entry, dict):
            raise ExportBundleRefusal(
                "LEDGER_CORRUPT: {} line {} is not a JSON object".format(
                    ledger_path, lineno))
        if entry.get("run_id") == run_id:
            latest = entry
    if latest is None:
        raise ExportBundleRefusal(
            "LEDGER_NO_ENTRY: no ledger entry for run_id {!r} in {}".format(
                run_id, ledger_path))
    return latest


def _assert_ledger_binding(entry: Dict[str, Any], checkpoint_sha256: str,
                           run_id: str) -> None:
    """Latest entry must be ``done`` and pin the SAME champion checkpoint."""
    outcome = entry.get("outcome")
    if outcome != "done":
        raise ExportBundleRefusal(
            "LEDGER_OUTCOME_NOT_DONE: latest ledger entry for {!r} has "
            "outcome {!r} (need 'done'; no export from an unfinished run)"
            .format(run_id, outcome))
    ledger_sha = entry.get("checkpoint_best_sha256")
    if not (isinstance(ledger_sha, str) and ledger_sha.strip()):
        raise ExportBundleRefusal(
            "LEDGER_CHECKPOINT_SHA_MISSING: {} ledger entry has no usable "
            "checkpoint_best_sha256 ({!r})".format(run_id, ledger_sha))
    if ledger_sha != checkpoint_sha256:
        raise ExportBundleRefusal(
            "LEDGER_CHECKPOINT_MISMATCH: {} ledger checkpoint_best_sha256={} "
            "but the champion checkpoint file is {} (no export from an "
            "unverified run)".format(run_id, ledger_sha, checkpoint_sha256))


def _assert_metrics_agreement(metrics: Dict[str, Any], *, run_id: str,
                              embedding_id: str, checkpoint_sha256: str,
                              seed: int) -> None:
    """metrics.json must agree with the CLI identity and the checkpoint."""
    metric_eid = metrics.get("embedding_id")
    if str(metric_eid) != str(embedding_id):
        raise ExportBundleRefusal(
            "METRICS_EMBEDDING_DRIFT: {} metrics embedding_id={!r} != export "
            "embedding_id={!r}".format(run_id, metric_eid, embedding_id))
    outcome = metrics.get("run_outcome")
    if outcome != "done":
        raise ExportBundleRefusal(
            "METRICS_OUTCOME_NOT_DONE: {} metrics run_outcome={!r}".format(
                run_id, outcome))
    metric_seed = metrics.get("seed")
    if metric_seed is not None and int(metric_seed) != int(seed):
        raise ExportBundleRefusal(
            "METRICS_SEED_DRIFT: {} metrics seed={} != run_id seed={}".format(
                run_id, metric_seed, seed))
    metric_sha = metrics.get("checkpoint_best_sha256")
    if isinstance(metric_sha, str) and metric_sha.strip() \
            and metric_sha != checkpoint_sha256:
        raise ExportBundleRefusal(
            "METRICS_CHECKPOINT_MISMATCH: {} metrics checkpoint_best_sha256="
            "{} but the champion checkpoint file is {}".format(
                run_id, metric_sha, checkpoint_sha256))


def _parse_run_id(run_id: str) -> int:
    """``V541_<...>_S<seed>`` -> seed (ValueError -> exit 2 via main)."""
    if not run_id.startswith("V541_"):
        raise ValueError(
            "run_id must use the V541_ namespace (D05), got {!r}".format(run_id))
    for token in reversed(run_id.split("_")):
        if token.startswith("S") and token[1:].isdigit():
            seed = int(token[1:])
            if seed in (17, 29, 43):
                return seed
            raise ValueError(
                "run_id seed {} is not preregistered (17/29/43)".format(seed))
    raise ValueError(
        "run_id {!r} carries no _S<seed> suffix (expected V541_..._S17/29/43)"
        .format(run_id))


def _embedding_id_from_run_id(run_id: str) -> Optional[str]:
    """Best-effort ``E<digit>`` token from a V541 run id (else ``None``)."""
    for token in run_id.split("_"):
        if len(token) == 2 and token[0] == "E" and token[1].isdigit():
            return token
    return None


def _default_embedding_config(cfg: Dict[str, Any], embedding_id: str,
                              model_id: str) -> Dict[str, Any]:
    """The frozen config slice exported as ``embedding_config.yaml``."""
    contract = cfg.get("embedding_contract") or {}
    registry = (contract.get("embedding_registry") or {}).get(
        str(embedding_id))
    if not isinstance(registry, dict):
        raise V541SchemaError(
            "V541_EMBEDDING_ROW_MISSING",
            "embedding_registry has no row for {!r}".format(embedding_id))
    return {
        "schema": cfg.get("schema"),
        "campaign_id": cfg.get("campaign_id"),
        "objective": cfg.get("objective"),
        "model_id": str(model_id),
        "embedding_id": str(embedding_id),
        "registry_row": registry,
        "residual": contract.get("residual"),
        "core_consumption": contract.get("core_consumption"),
        "feature_recipe": contract.get("feature_recipe"),
    }


def cmd_export(args: argparse.Namespace) -> int:
    # -- 1. run_id parse (ValueError -> exit 2 via main) --------------------
    seed = _parse_run_id(args.run_id)
    embedding_id = args.embedding or _embedding_id_from_run_id(args.run_id)
    if not embedding_id:
        raise ExportBundleRefusal(
            "EMBEDDING_ID_UNRESOLVED: cannot derive the embedding id from "
            "{!r}; pass --embedding explicitly".format(args.run_id))

    # -- 2. campaign-root discipline (CampaignRootError -> exit 1) ----------
    assert_v541_root(args.output_root)
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else Path(args.output_root) / "exports" / "embedding_champion_bundle"
    )
    assert_v541_root(output_dir)

    # -- 3. champion run outputs (fail-closed) ------------------------------
    run_dir = (Path(args.run_dir) if args.run_dir
               else Path(args.output_root) / FORMAL_RUN_SUBDIR / args.run_id)
    if not run_dir.is_dir():
        raise ExportBundleRefusal("RUN_DIR_MISSING: {}".format(run_dir))
    checkpoint = run_dir / "checkpoint_best.pt"
    if not checkpoint.is_file():
        raise ExportBundleRefusal("CHECKPOINT_MISSING: {}".format(checkpoint))

    from pes2ts_core.utils.hashing import sha256_file

    checkpoint_sha = sha256_file(checkpoint)

    # -- 4. ledger binding (append-only; LAST entry wins) -------------------
    ledger_path = Path(args.output_root) / LEDGER_BASENAME
    if run_records_path(Path(args.output_root)).name != LEDGER_BASENAME:
        raise ExportBundleRefusal(  # pragma: no cover - ownership drift guard
            "LEDGER_OWNERSHIP_DRIFT: v541_runplan and this CLI disagree on "
            "the ledger filename")
    entry = _latest_ledger_entry(ledger_path, args.run_id)
    _assert_ledger_binding(entry, checkpoint_sha, args.run_id)

    # -- 5. metrics agreement -----------------------------------------------
    metrics = _read_json(run_dir / "metrics.json", "METRICS")
    _assert_metrics_agreement(
        metrics, run_id=args.run_id, embedding_id=embedding_id,
        checkpoint_sha256=checkpoint_sha, seed=seed)

    # -- 6. evidence JSON files (WP7; NEVER invented here) ------------------
    provenance = _read_json(Path(args.provenance), "PROVENANCE")
    missing = [k for k in EMBEDDING_REQUIRED_PROVENANCE
               if k not in provenance]
    if missing:
        raise ExportBundleRefusal(
            "PROVENANCE_INCOMPLETE: {} is missing required key(s) {}; the "
            "WP7 evidence blocks are supplied by the campaign, never "
            "fabricated by this CLI".format(args.provenance, missing))
    benchmark_summary = _read_json(
        Path(args.benchmark_summary), "BENCHMARK_SUMMARY")

    # -- 7. config slice + feature schema -----------------------------------
    config_path = Path(args.config)
    try:
        cfg = load_config(config_path)
    except V541SchemaError as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 2
    embedding_config: Dict[str, Any]
    if args.embedding_config:
        embedding_config = _read_json(
            Path(args.embedding_config), "EMBEDDING_CONFIG")
    else:
        embedding_config = _default_embedding_config(
            cfg, embedding_id, args.model_id or "")
    feature_schema = (
        _read_json(Path(args.feature_schema), "FEATURE_SCHEMA")
        if args.feature_schema
        else build_embedding_feature_schema(embedding_id))

    model_id = args.model_id or "pes2ts-v541-{}-s{}".format(
        embedding_id.lower(), seed)

    # -- 8. export ----------------------------------------------------------
    bundle = export_embedding_bundle(
        run_dir=run_dir,
        output_dir=output_dir,
        model_id=model_id,
        embedding_id=embedding_id,
        embedding_config=embedding_config,
        feature_schema=feature_schema,
        provenance=provenance,
        benchmark_summary=benchmark_summary,
        checkpoint_name="checkpoint_best.pt",
        seeds_required=(17, 29, 43),
    )

    # -- 9. post-hoc whitelist + SHA audit + printout -----------------------
    verify_embedding_bundle(bundle)
    checksums = _read_json(bundle / "checksums.json", "CHECKSUMS")
    present = {p.name for p in bundle.iterdir() if p.is_file()}
    if present != set(EMBEDDING_BUNDLE_FILES):  # pragma: no cover - belt+braces
        raise ExportBundleRefusal(
            "BUNDLE_WHITELIST_DRIFT: unexpected={} missing={}".format(
                sorted(present - set(EMBEDDING_BUNDLE_FILES)),
                sorted(set(EMBEDDING_BUNDLE_FILES) - present)))
    print("bundle: {}".format(bundle))
    for name in sorted(checksums):
        print("{}  {}".format(checksums[name], name))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="v541_export_bundle",
        description="V5.4.1 WP7: export the embedding champion whitelist "
                    "bundle (pes2ts_embedding_bundle_v1) for ONE verified "
                    "champion run (fail-closed)",
    )
    parser.add_argument(
        "--run-id", required=True,
        help="V541_<...>_S<17|29|43> champion run id",
    )
    parser.add_argument("--output-root", default=DEFAULT_ROOT)
    parser.add_argument(
        "--output-dir", default=None,
        help="default: <output-root>/exports/embedding_champion_bundle "
             "(must stay under outputs/pesn/v541_embed_50k/**)",
    )
    parser.add_argument(
        "--run-dir", default=None,
        help="default: <output-root>/runs/<run-id> (formal lane)",
    )
    parser.add_argument(
        "--embedding", default=None,
        help="embedding id E0–E5 (default: parsed from --run-id)",
    )
    parser.add_argument(
        "--model-id", default=None,
        help="default: pes2ts-v541-<eid>-s<seed>",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--provenance", required=True,
        help="WP7 provenance JSON (three seeds / screen / dev exposure / "
             "unread confirm / cost / failures and censoring / unclosed "
             "capacity diagnostics + §12.2 identity SHAs)",
    )
    parser.add_argument(
        "--benchmark-summary", required=True,
        help="WP7 benchmark summary JSON (winner / per-candidate ledger / "
             "H3 / confirm sealed)",
    )
    parser.add_argument(
        "--feature-schema", default=None,
        help="optional feature-schema JSON override (default: built unified "
             "schema, node 48 / legacy edge 22)",
    )
    parser.add_argument(
        "--embedding-config", default=None,
        help="optional embedding-config JSON override (default: frozen "
             "campaign-yaml registry row + residual + core-consumption + "
             "feature recipe)",
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
    except (CampaignRootError, V541RunnerError) as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 1
    except ExportBundleRefusal as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 1
    except (EmbeddingBundleError, ExportError) as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 1
    except V541SchemaError as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 2
    except OSError as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
