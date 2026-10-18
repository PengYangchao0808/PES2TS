"""Runtime whitelist control-bundle export (``pesn_model_bundle_v2``).

Exports the SAME whitelist file set as the v1 teacher bundle (so the reader
and product-boundary tooling stay unchanged) with a v2 manifest: unified
encoder + control heads, the frozen ControlSpec hash, the environment
domain (E_native), the reward-spec hash, and the v2 research capability map
(``residual_head`` / ``q_proxy`` research-only; every formal decision cap
disabled).  The v1 export path is untouched and v1 bundles stay readable.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Mapping

from pes2ts_core.pesn.bundle.reader import SUPPORTED_BUNDLE_FORMAT_V2
from pes2ts_core.pesn.contracts.artifacts import (
    BUNDLE_FORBIDDEN_KEYS,
    BUNDLE_RUNTIME_FILES,
)
from pes2ts_core.pesn.contracts.control import (
    CONTROL_BUNDLE_SCHEMA_VERSION,
    ENVIRONMENT_NATIVE,
    control_capability_map,
    validate_control_bundle_manifest,
)
from pes2ts_core.pesn.offline.experiments.runs import sha256_text
from pes2ts_core.pesn.offline.publishing.export import build_feature_spec_document
from pes2ts_core.utils.hashing import JSONValue, sha256_file
from pes2ts_core.utils.jsonio import write_json

EXPORT_SCHEMA: Final[str] = "pesn_control_export_record_v1"


class ControlExportError(RuntimeError):
    """Typed control-bundle export refusal."""


def build_control_model_card(
    *,
    model_id: str,
    manifest: Mapping[str, JSONValue],
    metrics: Mapping[str, JSONValue],
    run_summary: Mapping[str, JSONValue],
) -> str:
    caps = manifest["capabilities"]
    lines = [
        f"# {model_id} — research control model card (draft)",
        "",
        f"- generated: {datetime.now(UTC).isoformat(timespec='seconds')}",
        "- research package: bounded per-coordinate residual proposals and",
        "  action-conditioned Q_proxy estimates only; Q_proxy is not Q_MOVE,",
        "  not a success probability, and production control stays disabled.",
        f"- environment domain: {manifest.get('environment_id')} (frozen identity",
        "  hash in the manifest); policies must be compared on the same",
        "  environment and budget.",
        "- data provenance: REAL G2/ACP macro-step transitions collected under",
        "  the frozen E_native environment on train-qualified reactions; IRC",
        "  teacher frames contributed zero transition records.",
        "",
        "## Frozen metrics (control run)",
        "```json",
        json.dumps(dict(metrics), sort_keys=True),
        "```",
        "",
        "## Run summary",
        "```json",
        json.dumps(dict(run_summary), sort_keys=True),
        "```",
        "",
        "## Capabilities",
        "```json",
        json.dumps(dict(caps), sort_keys=True),
        "```",
        "",
        "No claim is made about TS success, refinement savings, or transfer",
        "beyond the frozen environment.",
    ]
    return "\n".join(lines) + "\n"


def _validate_export_inputs(
    *,
    run_path: Path,
    checkpoint_payload: Mapping[str, Any],
    run_summary: Mapping[str, JSONValue],
    open_q_proxy: bool,
) -> bool:
    """Cross-check checkpoint / run record / summary; return effective q flag."""
    checkpoint_mode = str(checkpoint_payload.get("training_mode", ""))
    summary_mode = str(run_summary.get("training_mode", ""))
    if not checkpoint_mode:
        raise ControlExportError("checkpoint carries no training_mode")
    if checkpoint_mode != summary_mode:
        raise ControlExportError(
            f"checkpoint training_mode {checkpoint_mode!r} != summary mode "
            f"{summary_mode!r}; refusing to export a mislabelled run"
        )
    run_document_path = run_path / "run.json"
    if not run_document_path.is_file():
        raise ControlExportError(f"run record missing: {run_document_path}")
    run_document = json.loads(run_document_path.read_text(encoding="utf-8"))
    run_mode = run_document.get("training_mode")
    if run_mode is None:
        config = run_document.get("config") or {}
        run_mode = (config.get("control_train") or {}).get("training_mode")
    if str(run_mode) != checkpoint_mode:
        raise ControlExportError(
            f"run record training_mode {run_mode!r} != checkpoint mode "
            f"{checkpoint_mode!r}; the mode must be verifiable from the run"
        )
    if str(run_document.get("run_id", "")) != str(run_summary.get("run_id", "")):
        raise ControlExportError(
            f"run.json run_id {run_document.get('run_id')!r} != summary run_id "
            f"{run_summary.get('run_id')!r}; refusing to mix runs"
        )
    expected_config_sha = run_summary.get("effective_config_sha256")
    if expected_config_sha and checkpoint_payload.get("config_sha256") != expected_config_sha:
        raise ControlExportError(
            "checkpoint config digest does not match the trained run's "
            "effective config; refusing to export"
        )
    expected_transitions = run_summary.get("transitions_digest")
    if expected_transitions and checkpoint_payload.get("transitions_digest") != expected_transitions:
        raise ControlExportError(
            "checkpoint transitions digest does not match the trained run's "
            "data digest; refusing to export"
        )
    if checkpoint_mode == "bc_only" and open_q_proxy:
        return False
    return open_q_proxy


def export_control_bundle(
    *,
    run_dir: str | Path,
    output_dir: str | Path,
    model_id: str,
    metrics: Mapping[str, JSONValue],
    run_summary: Mapping[str, JSONValue],
    model_config: Mapping[str, Any],
    control_spec_sha256: str,
    reward_spec_sha256: str,
    environment_id: str = ENVIRONMENT_NATIVE,
    open_residual: bool = True,
    open_q_proxy: bool = True,
    support_registry_doc: Mapping[str, Any] | None = None,
) -> Path:
    """Assemble and validate the whitelist v2 runtime bundle.

    The bundle is staged under a temporary sibling directory and moved into
    place atomically, so a failed export can never leave a half-written
    package that looks usable; an existing non-empty target is refused.

    When ``support_registry_doc`` is supplied the bundle becomes a
    registry-bound package: the exact registry document ships inside the
    bundle and the manifest carries its hash, environment/reward identity,
    and fit-transition digest (review F3).  Without it the export is an
    explicit legacy source-gate package.
    """
    run_path = Path(run_dir)
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise ControlExportError(
            f"target bundle directory already exists and is not empty: {out}; "
            "published model packages are never overwritten"
        )

    best_checkpoint = run_path / "checkpoint_best.pt"
    if not best_checkpoint.is_file():
        raise ControlExportError(f"best control checkpoint missing: {best_checkpoint}")

    from pes2ts_core.pesn.models import BackendMissingError

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise BackendMissingError("torch", "control bundle export") from exc

    payload = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    open_q_proxy = _validate_export_inputs(
        run_path=run_path,
        checkpoint_payload=payload,
        run_summary=run_summary,
        open_q_proxy=open_q_proxy,
    )
    model_state = payload["model_state"]

    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.parent / f".{out.name}.staging"
    if staging.exists():
        import shutil

        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        _write_bundle_files(
            staging=staging,
            model_id=model_id,
            metrics=metrics,
            run_summary=run_summary,
            model_config=model_config,
            control_spec_sha256=control_spec_sha256,
            reward_spec_sha256=reward_spec_sha256,
            environment_id=environment_id,
            open_residual=open_residual,
            open_q_proxy=open_q_proxy,
            model_state=model_state,
            torch=torch,
            support_registry_doc=support_registry_doc,
        )
        staging.rename(out)
    except BaseException:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)
        raise
    return out


def _write_bundle_files(
    *,
    staging: Path,
    model_id: str,
    metrics: Mapping[str, JSONValue],
    run_summary: Mapping[str, JSONValue],
    model_config: Mapping[str, Any],
    control_spec_sha256: str,
    reward_spec_sha256: str,
    environment_id: str,
    open_residual: bool,
    open_q_proxy: bool,
    model_state,
    torch,
    support_registry_doc: Mapping[str, Any] | None = None,
) -> None:
    from pes2ts_core.pesn.contracts.control import (
        CONTROL_SUPPORT_REGISTRY_FILE,
        CONTROL_SUPPORT_REGISTRY_SCHEMA,
    )

    weights_path = staging / "weights.pt"
    torch.save({"model_state": model_state}, weights_path)

    feature_spec = build_feature_spec_document()
    write_json(staging / "feature_spec.json", feature_spec)

    label_spec = {
        "schema_version": "pesn_control_labelspec_v1",
        "note": "control bundles train on real macro-step transitions, not teacher labels",
        "content_sha256": "",
    }
    label_spec["content_sha256"] = sha256_text(
        json.dumps(label_spec, sort_keys=True)
    )
    write_json(staging / "label_spec.json", label_spec)

    normalization = {
        "schema_version": "pesn_normalization_v1",
        "note": "encoder consumes scale-free invariant bases; no fitted normalizer",
        "sealed_samples_contributed": 0,
    }
    write_json(staging / "normalization.json", normalization)

    split_reference = {
        "split_schema": "control_round_selection_v1",
        "note": "transitions restricted to the frozen train-qualified selection",
        "selection_manifest_sha256": run_summary.get("selection_manifest_sha256"),
    }
    write_json(staging / "split_reference.json", split_reference)
    write_json(staging / "training_runs.json", {"runs": [dict(run_summary)]})
    write_json(staging / "metrics.json", dict(metrics))

    capabilities = control_capability_map(
        residual=open_residual, q_proxy=open_q_proxy
    )

    checksums: dict[str, str] = {}
    for name in (
        "weights.pt",
        "feature_spec.json",
        "normalization.json",
        "label_spec.json",
        "split_reference.json",
        "training_runs.json",
        "metrics.json",
    ):
        checksums[name] = sha256_file(staging / name)

    manifest: dict[str, JSONValue] = {
        "schema_version": CONTROL_BUNDLE_SCHEMA_VERSION,
        "bundle_format": SUPPORTED_BUNDLE_FORMAT_V2,
        "model_id": model_id,
        "model_hash": checksums["weights.pt"],
        "feature_spec_sha256": feature_spec["content_sha256"],
        "label_spec_sha256": label_spec["content_sha256"],
        "normalization_sha256": checksums["normalization.json"],
        "weights_sha256": checksums["weights.pt"],
        "training_run_ids": [run_summary.get("run_id", "unknown")],
        "capabilities": capabilities,
        "model_config": dict(model_config),
        "control_spec_sha256": control_spec_sha256,
        "environment_id": environment_id,
        "reward_spec_sha256": reward_spec_sha256,
        "model_card": "model_card.md",
        "checksums": "checksums.json",
        "research_only": True,
    }
    if support_registry_doc is not None:
        built_from = support_registry_doc.get("built_from") or {}
        fit_ids = [str(tid) for tid in built_from.get("fit_transition_ids", [])]
        canonical = json.dumps(support_registry_doc, sort_keys=True)
        (staging / CONTROL_SUPPORT_REGISTRY_FILE).write_text(
            canonical + "\n",
            encoding="utf-8",
        )
        checksums[CONTROL_SUPPORT_REGISTRY_FILE] = sha256_file(
            staging / CONTROL_SUPPORT_REGISTRY_FILE
        )
        manifest["support_registry"] = {
            "schema": str(support_registry_doc.get("schema_version", "")),
            "file": CONTROL_SUPPORT_REGISTRY_FILE,
            "sha256": sha256_text(canonical),
            "environment_hash": str(support_registry_doc["environment_hash"]),
            "reward_spec_sha256": str(support_registry_doc["reward_spec_sha256"]),
            "fit_transitions_digest": sha256_text(
                json.dumps(sorted(fit_ids))
            ),
            "n_fit_transitions": int(built_from.get("n_fit_transitions", len(fit_ids))),
        }
        if manifest["support_registry"]["schema"] != CONTROL_SUPPORT_REGISTRY_SCHEMA:
            raise ControlExportError(
                "support registry doc schema "
                f"{manifest['support_registry']['schema']!r} is not "
                f"{CONTROL_SUPPORT_REGISTRY_SCHEMA!r}"
            )
    for key in BUNDLE_FORBIDDEN_KEYS:
        if key in manifest:
            raise ControlExportError(f"manifest would carry forbidden key {key!r}")
    model_card = build_control_model_card(
        model_id=model_id,
        manifest=manifest,
        metrics=metrics,
        run_summary=run_summary,
    )
    (staging / "model_card.md").write_text(model_card, encoding="utf-8")
    checksums["model_card.md"] = sha256_file(staging / "model_card.md")

    manifest["model_card_sha256"] = checksums["model_card.md"]
    validate_control_bundle_manifest(manifest)
    write_json(staging / "model_bundle.json", manifest)
    checksums["model_bundle.json"] = sha256_file(staging / "model_bundle.json")
    write_json(staging / "checksums.json", checksums)

    present = {entry.name for entry in staging.iterdir() if entry.is_file()}
    if present - set(BUNDLE_RUNTIME_FILES):
        raise ControlExportError(
            f"export produced non-whitelisted files: {sorted(present - set(BUNDLE_RUNTIME_FILES))}"
        )


__all__ = [
    "EXPORT_SCHEMA",
    "ControlExportError",
    "build_control_model_card",
    "export_control_bundle",
]
