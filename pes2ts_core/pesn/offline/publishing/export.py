"""Runtime whitelist bundle export (B6).

Exports exactly ``BUNDLE_RUNTIME_FILES`` from a finished run: model weights
(model state only — never optimizer/trainer state), feature/label spec,
normalization, split reference (hashes only), run summary, frozen metrics,
generated model card, checksums, and the manifest binding them all.  The
export refuses forbidden keys and any attempt to include teacher data,
offline entrypoints, or source paths.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Mapping, Sequence

from pes2ts_core.pesn.bundle.reader import SUPPORTED_BUNDLE_FORMAT
from pes2ts_core.pesn.contracts.artifacts import (
    BUNDLE_FORBIDDEN_KEYS,
    BUNDLE_RUNTIME_FILES,
    BUNDLE_SCHEMA_VERSION,
    validate_bundle_manifest,
)
from pes2ts_core.pesn.contracts.predictions import (
    research_capability_map,
)
from pes2ts_core.pesn.offline.experiments.runs import sha256_text
from pes2ts_core.utils.hashing import JSONValue, sha256_file, stable_json_dumps
from pes2ts_core.utils.jsonio import read_json, write_json

EXPORT_SCHEMA: Final[str] = "pesn_export_record_v1"


class ExportError(RuntimeError):
    """Typed export refusal (forbidden content / missing run outputs)."""


def build_feature_spec_document() -> dict[str, JSONValue]:
    from pes2ts_core.pesn.features.graph import (
        DEFAULT_N_BASIS,
        DEFAULT_SPATIAL_CUTOFF,
        ROLE_NAMES,
        FEATURE_SPEC_VERSION,
    )
    from pes2ts_core.pesn.features.normalization import DESCRIPTOR_NAMES

    document: dict[str, JSONValue] = {
        "schema_version": "pesn_feature_spec_v1",
        "spec_version": FEATURE_SPEC_VERSION,
        "allowed_fields": [
            "input_id",
            "atomic_numbers",
            "coordinates",
            "units",
            "charge",
            "multiplicity",
            "method",
            "task_graph",
            "coordinate_spec",
        ],
        "graph_definition": {
            "edge_roles": list(ROLE_NAMES),
            "spatial_cutoff_angstrom": DEFAULT_SPATIAL_CUTOFF,
            "distance_basis": DEFAULT_N_BASIS,
            "task_edges_uncut": True,
            "driver_coordinates": "exclusive edit-pair distances",
        },
        "units": {"length": "angstrom", "direction": "unitless_scaled_internal"},
        "invariances": ["translation", "rotation", "atom-map-consistent-relabeling"],
        "support_domain": {
            "atomic_numbers": "Z 1..86 (Cordero table coverage)",
            "n_driver_coords": ">=1",
            "note": "inputs outside the frozen support raise typed rejections",
        },
        "descriptor_names": list(DESCRIPTOR_NAMES),
    }
    body = dict(document)
    body.pop("content_sha256", None)
    document["content_sha256"] = sha256_text(stable_json_dumps(body))
    return document


def build_model_card(
    *,
    model_id: str,
    manifest: Mapping[str, JSONValue],
    metrics: Mapping[str, JSONValue],
    run_summary: Mapping[str, JSONValue],
) -> str:
    caps = manifest["capabilities"]
    lines = [
        f"# {model_id} — research model card (draft)",
        "",
        f"- generated: {datetime.now(UTC).isoformat(timespec='seconds')}",
        "- research package: predictions are IRC-teacher research outputs only;",
        "  STOP/MOVE/Q heads and production control are disabled by contract.",
        "- data provenance: B3LYP-D3/TZVP IRC trajectories (quarantined source,",
        "  audited scope-checked bridge reads); truth-assisted P1 atom mapping;",
        "  endpoint graphs from the G1 authority. Not end-to-end truth-free.",
        "",
        "## Frozen metrics (initial run)",
        "```json",
        stable_json_dumps(dict(metrics)),
        "```",
        "",
        "## Run summary",
        "```json",
        stable_json_dumps(dict(run_summary)),
        "```",
        "",
        "## Capabilities",
        "```json",
        stable_json_dumps(dict(caps)),
        "```",
        "",
        "No claim is made about refinement success, off-path transfer, or QC cost.",
    ]
    return "\n".join(lines) + "\n"


def export_bundle(
    *,
    run_dir: str | Path,
    dataset_dir: str | Path,
    output_dir: str | Path,
    model_id: str,
    metrics: Mapping[str, JSONValue],
    open_direction: bool,
    open_activity: bool,
    run_summary: Mapping[str, JSONValue],
    model_config: Mapping[str, int],
) -> Path:
    """Assemble and validate the whitelist runtime bundle."""
    run_path = Path(run_dir)
    dataset_path = Path(dataset_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    best_checkpoint = run_path / "checkpoint_best.pt"
    if not best_checkpoint.is_file():
        raise ExportError(f"best checkpoint missing: {best_checkpoint}")

    from pes2ts_core.pesn.models import BackendMissingError

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise BackendMissingError("torch", "bundle export") from exc

    payload = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    model_state = payload["model_state"]
    weights_path = out / "weights.pt"
    torch.save({"model_state": model_state}, weights_path)

    feature_spec = build_feature_spec_document()
    write_json(out / "feature_spec.json", feature_spec)

    label_spec_path = dataset_path / "label_spec.json"
    if label_spec_path.is_file():
        label_spec = read_json(label_spec_path)
    else:
        raise ExportError(f"label spec missing: {label_spec_path}")
    write_json(out / "label_spec.json", label_spec)

    normalization_path = run_path / "normalization.json"
    normalization: dict[str, JSONValue] = {
        "schema_version": "pesn_normalization_v1",
        "note": "encoder consumes scale-free invariant bases; no fitted normalizer in v1",
        "sealed_samples_contributed": 0,
    }
    if normalization_path.is_file():
        normalization = read_json(normalization_path)
    write_json(out / "normalization.json", normalization)

    dataset_manifest = read_json(dataset_path / "dataset_manifest.json")
    split_reference = {
        "split_schema": "pesn_irc_split_v1",
        "parent_split_manifest_sha256": dataset_manifest.get("parent_split_manifest_sha256"),
        "dataset_manifest_sha256": dataset_manifest.get("dataset_manifest_sha256")
        or _dataset_hash(dataset_path),
        "subsets_used": ["train_fit", "train_dev"],
        "note": "sealed parent valid/test contributed zero training samples",
    }
    write_json(out / "split_reference.json", split_reference)

    write_json(out / "training_runs.json", {"runs": [dict(run_summary)]})
    write_json(out / "metrics.json", dict(metrics))

    capabilities = research_capability_map(
        progress=True,
        path_proximity=True,
        teacher_direction=open_direction,
        coordinate_activity=open_activity,
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
        checksums[name] = sha256_file(out / name)

    manifest: dict[str, JSONValue] = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "bundle_format": SUPPORTED_BUNDLE_FORMAT,
        "model_id": model_id,
        "model_hash": checksums["weights.pt"],
        "feature_spec_sha256": feature_spec["content_sha256"],
        "label_spec_sha256": label_spec.get("content_sha256", ""),
        "normalization_sha256": checksums["normalization.json"],
        "weights_sha256": checksums["weights.pt"],
        "training_run_ids": [run_summary.get("run_id", "unknown")],
        "capabilities": capabilities,
        "model_config": dict(model_config),
        "model_card": "model_card.md",
        "checksums": "checksums.json",
        "research_only": True,
    }
    for key in BUNDLE_FORBIDDEN_KEYS:
        if key in manifest:
            raise ExportError(f"manifest would carry forbidden key {key!r}")
    model_card = build_model_card(
        model_id=model_id,
        manifest=manifest,
        metrics=metrics,
        run_summary=run_summary,
    )
    (out / "model_card.md").write_text(model_card, encoding="utf-8")
    checksums["model_card.md"] = sha256_file(out / "model_card.md")

    manifest["model_card_sha256"] = checksums["model_card.md"]
    validate_bundle_manifest(manifest)
    write_json(out / "model_bundle.json", manifest)
    checksums["model_bundle.json"] = sha256_file(out / "model_bundle.json")
    write_json(out / "checksums.json", checksums)

    present = {entry.name for entry in out.iterdir() if entry.is_file()}
    if present - set(BUNDLE_RUNTIME_FILES):
        raise ExportError(
            f"export produced non-whitelisted files: {sorted(present - set(BUNDLE_RUNTIME_FILES))}"
        )
    return out


def _dataset_hash(dataset_dir: Path) -> str:
    return sha256_file(dataset_dir / "dataset_manifest.json")


# ---------------------------------------------------------------------------
# V5.3-MP MOVE+PROCESS whitelist bundle (contract §6 export
# ``pes2ts_move_process_bundle_v1``).  A NEW function on purpose: the B6
# ``export_bundle`` above is the G3-P1 pesn_irc bundle and stays untouched.
# ---------------------------------------------------------------------------

MOVE_PROCESS_BUNDLE_SCHEMA: Final[str] = "pes2ts_move_process_bundle_v1"
#: No-Move threshold is calibrated on the inner-val split at this precision
#: target (MoveProcessContract §8/§9) and frozen before export.
MOVE_PROCESS_TAU_TARGET_PRECISION: Final[float] = 0.9
#: Provenance keys a caller MUST provide; the export refuses without them.
MOVE_PROCESS_REQUIRED_PROVENANCE: Final[tuple[str, ...]] = (
    "config_sha256",
    "data_manifest_sha256",
    "teacher_manifest_sha256",
    "split_sha256",
    "label_spec_sha256",
    "code_commit",
)
#: Exact whitelist — nothing else may exist in the output directory.
MOVE_PROCESS_BUNDLE_FILES: Final[tuple[str, ...]] = (
    "weights.pt",
    "tau_end.json",
    "metrics.json",
    "provenance.json",
    "model_card.md",
    "model_bundle.json",
    "checksums.json",
)
_MOVE_PROCESS_HEADS_VERSION: Final[str] = "pesn_key_atom_v53mp_heads_v1"
_MOVE_PROCESS_BACKBONE: Final[str] = "StarEGNNV3"
_MOVE_PROCESS_BACKBONE_PARAMS: Final[int] = 55441
_MOVE_PROCESS_CONTRACT_DOCS: Final[tuple[str, ...]] = (
    "docs/contracts/PES2TS_MoveProcessContract_v1.md",
    "docs/plans/PES2TS_V5.3-MP_大规模MOVE-PROCESS联合训练计划_v1_20261021.md",
    "docs/design/decisions/ADR-0031-MOVE-PROCESS统一学习合同.md",
)


def _validate_move_process_inputs(
    tau_end: float, provenance: Mapping[str, str]
) -> float:
    """Fail-closed pre-checks (no filesystem / torch access yet)."""
    try:
        tau = float(tau_end)
    except (TypeError, ValueError) as exc:
        raise ExportError(f"tau_end must be a number in [0, 1], got {tau_end!r}") from exc
    import math

    if not math.isfinite(tau) or not (0.0 <= tau <= 1.0):
        raise ExportError(f"tau_end must lie in [0, 1], got {tau_end!r}")
    if not isinstance(provenance, Mapping):
        raise ExportError("provenance must be a mapping of str -> str")
    missing = [k for k in MOVE_PROCESS_REQUIRED_PROVENANCE if k not in provenance]
    if missing:
        raise ExportError(
            "provenance is missing required key(s): " + ", ".join(missing)
        )
    for key in MOVE_PROCESS_REQUIRED_PROVENANCE:
        value = provenance[key]
        if not (isinstance(value, str) and value.strip()):
            raise ExportError(
                f"provenance[{key!r}] must be a non-empty string, got {value!r}"
            )
    return tau


def _build_move_process_model_card(
    *,
    model_id: str,
    tau_end: float,
    metrics: Mapping[str, JSONValue],
    provenance: Mapping[str, JSONValue],
) -> str:
    lines = [
        f"# {model_id} — V5.3-MP MOVE+PROCESS research model card",
        "",
        f"- generated: {datetime.now(UTC).isoformat(timespec='seconds')}",
        f"- contract schema: {MOVE_PROCESS_BUNDLE_SCHEMA}",
        f"- heads: {_MOVE_PROCESS_HEADS_VERSION}",
        f"- backbone: {_MOVE_PROCESS_BACKBONE} "
        f"({_MOVE_PROCESS_BACKBONE_PARAMS} parameters, frozen architecture)",
        "- scope: MOVE (key-atom displacement) + PROCESS (progress q / endpoint",
        "  completion / directed ordering) research training outputs only.",
        "- No-Move rule: [p_end >= tau_end] AND [target chemical-state",
        "  consistency]. tau_end is calibrated on the inner-val split at target",
        f"  precision {MOVE_PROCESS_TAU_TARGET_PRECISION} and then frozen.",
        "- DEPLOYMENT: training qualification is NOT deployment qualification.",
        "  The No-Move head is NOT deployment-qualified until independent",
        "  calibration, false-stop-rate and closed-loop validation pass; no",
        "  online control authority is granted by this bundle.",
        "- claims: limited to TS_candidate / endpoint_reached; no verified TS",
        "  or verified reaction path is claimed (OptTS + single imaginary",
        "  frequency + bidirectional IRC + endpoint identity required).",
        "- research_only: true — not for production control.",
        "",
        "## Frozen metrics",
        "```json",
        stable_json_dumps(dict(metrics)),
        "```",
        "",
        "## Provenance (abridged)",
        "```json",
        stable_json_dumps(dict(provenance)),
        "```",
        "",
        "Contract documents:",
    ]
    lines.extend(f"- {path}" for path in _MOVE_PROCESS_CONTRACT_DOCS)
    return "\n".join(lines) + "\n"


def export_move_process_bundle(
    *,
    run_dir: str | Path,
    output_dir: str | Path,
    model_id: str,
    tau_end: float,
    metrics: Mapping[str, JSONValue],
    provenance: Mapping[str, str],
) -> Path:
    """Assemble the V5.3-MP ``pes2ts_move_process_bundle_v1`` whitelist bundle.

    Writes exactly :data:`MOVE_PROCESS_BUNDLE_FILES`:

    * ``weights.pt`` — ``{"model_state": ...}`` taken from
      ``checkpoint_best.pt`` payload key ``"model"``; optimizer/trainer state
      is never exported;
    * ``tau_end.json`` — the frozen inner-val-calibrated No-Move threshold;
    * ``metrics.json`` / ``provenance.json`` / ``model_card.md``;
    * ``model_bundle.json`` (schema binding + checksums + capabilities) and
      ``checksums.json``.

    Refuses (:class:`ExportError`, fail-closed) when ``tau_end`` is outside
    ``[0, 1]`` or ``provenance`` misses any of
    :data:`MOVE_PROCESS_REQUIRED_PROVENANCE`.
    """
    tau = _validate_move_process_inputs(tau_end, provenance)
    run_path = Path(run_dir)
    out = Path(output_dir)

    best_checkpoint = run_path / "checkpoint_best.pt"
    if not best_checkpoint.is_file():
        raise ExportError(f"best checkpoint missing: {best_checkpoint}")

    from pes2ts_core.pesn.models import BackendMissingError

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise BackendMissingError(
            "torch", "move-process bundle export"
        ) from exc

    payload = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "model" not in payload:
        raise ExportError(
            "checkpoint payload must carry the model state under the 'model' "
            "key (V5.3-MP trainer contract)"
        )
    model_state = payload["model"]
    if not isinstance(model_state, Mapping):
        raise ExportError("checkpoint payload['model'] is not a state mapping")
    out.mkdir(parents=True, exist_ok=True)
    weights_path = out / "weights.pt"
    torch.save({"model_state": model_state}, weights_path)

    tau_doc: dict[str, JSONValue] = {
        "tau_end": tau,
        "target_precision": MOVE_PROCESS_TAU_TARGET_PRECISION,
        "calibrated_on": "inner-val",
        "frozen": True,
    }
    write_json(out / "tau_end.json", tau_doc)
    write_json(out / "metrics.json", dict(metrics))

    provenance_doc: dict[str, JSONValue] = dict(provenance)
    provenance_doc.update(
        {
            "heads_version": _MOVE_PROCESS_HEADS_VERSION,
            "backbone": _MOVE_PROCESS_BACKBONE,
            "backbone_params": _MOVE_PROCESS_BACKBONE_PARAMS,
            "contract_docs": list(_MOVE_PROCESS_CONTRACT_DOCS),
            "export_schema": MOVE_PROCESS_BUNDLE_SCHEMA,
        }
    )
    write_json(out / "provenance.json", provenance_doc)

    model_card = _build_move_process_model_card(
        model_id=model_id, tau_end=tau, metrics=metrics, provenance=provenance_doc
    )
    (out / "model_card.md").write_text(model_card, encoding="utf-8")

    checksums: dict[str, str] = {}
    for name in (
        "weights.pt",
        "tau_end.json",
        "metrics.json",
        "provenance.json",
        "model_card.md",
    ):
        checksums[name] = sha256_file(out / name)

    bundle: dict[str, JSONValue] = {
        "schema_version": MOVE_PROCESS_BUNDLE_SCHEMA,
        "model_id": model_id,
        "model_hash": checksums["weights.pt"],
        "weights_sha256": checksums["weights.pt"],
        "tau_end_sha256": checksums["tau_end.json"],
        "metrics_sha256": checksums["metrics.json"],
        "provenance_sha256": checksums["provenance.json"],
        "model_card_sha256": checksums["model_card.md"],
        "tau_end": tau,
        "heads_version": _MOVE_PROCESS_HEADS_VERSION,
        "backbone": _MOVE_PROCESS_BACKBONE,
        "backbone_params": _MOVE_PROCESS_BACKBONE_PARAMS,
        "capabilities": {
            "move": True,
            "progress_q": "diagnostic",
            "endpoint": "diagnostic_not_deployment",
            "ordering": "diagnostic",
        },
        "research_only": True,
        "model_card": "model_card.md",
        "checksums": dict(checksums),
    }
    for key in BUNDLE_FORBIDDEN_KEYS:
        if key in bundle:
            raise ExportError(f"bundle manifest would carry forbidden key {key!r}")
    write_json(out / "model_bundle.json", bundle)
    checksums["model_bundle.json"] = sha256_file(out / "model_bundle.json")
    write_json(out / "checksums.json", checksums)

    present = {entry.name for entry in out.iterdir() if entry.is_file()}
    if present != set(MOVE_PROCESS_BUNDLE_FILES):
        unexpected = sorted(present - set(MOVE_PROCESS_BUNDLE_FILES))
        missing = sorted(set(MOVE_PROCESS_BUNDLE_FILES) - present)
        raise ExportError(
            "move-process bundle must contain exactly {}; unexpected={} missing={}"
            .format(list(MOVE_PROCESS_BUNDLE_FILES), unexpected, missing)
        )
    return out


__all__ = [
    "EXPORT_SCHEMA",
    "ExportError",
    "MOVE_PROCESS_BUNDLE_FILES",
    "MOVE_PROCESS_BUNDLE_SCHEMA",
    "MOVE_PROCESS_REQUIRED_PROVENANCE",
    "MOVE_PROCESS_TAU_TARGET_PRECISION",
    "build_feature_spec_document",
    "build_model_card",
    "export_bundle",
    "export_move_process_bundle",
]
