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
from typing import Any, Dict, Final, Mapping, Sequence

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


# ---------------------------------------------------------------------------
# V5.4.1 embedding champion bundle (plan §12.1/§12.3
# ``pes2ts_embedding_bundle_v1``).  A NEW function on purpose: the B6
# ``export_bundle`` and the V5.3-MP ``export_move_process_bundle`` above stay
# untouched.  Real export runs only AFTER the WP7 dev evaluation the main
# agent performs; this exporter is fail-closed and evidence-driven.
# ---------------------------------------------------------------------------

EMBEDDING_BUNDLE_SCHEMA: Final[str] = "pes2ts_embedding_bundle_v1"
EMBEDDING_WEIGHTS_SCHEMA: Final[str] = "pes2ts_embedding_weights_v1"
EMBEDDING_HANDOFF_SCHEMA: Final[str] = "pes2ts_v541_downstream_handoff_v1"
EMBEDDING_FEATURE_SCHEMA_ID: Final[str] = "pes2ts_v541_feature_schema_v1"
#: Frozen node width / legacy edge channels the schema must declare
#: (plan §12.3: feature_schema 固定节点 48、legacy edge 22).
EMBEDDING_NODE_DIM: Final[int] = 48
EMBEDDING_LEGACY_EDGE_DIM: Final[int] = 22
#: Exact whitelist — nothing else may exist in the output directory.
EMBEDDING_BUNDLE_FILES: Final[tuple[str, ...]] = (
    "embedding_config.yaml",
    "embedding_weights.pt",
    "feature_schema.json",
    "provenance.json",
    "benchmark_summary.json",
    "model_card.md",
    "downstream_handoff.json",
    "checksums.json",
)
#: Provenance keys the export REFUSES to run without (plan §12.3: 三 seed /
#: screen / 训练与 dev 暴露 / 未读 confirm / 成本 / 失败删失 / 未闭合容量诊断,
#: plus the frozen identity SHAs of §12.2).
EMBEDDING_REQUIRED_PROVENANCE: Final[tuple[str, ...]] = (
    "config_sha256",
    "common_protocol_sha256",
    "embedding_spec_sha256",
    "feature_recipe_sha256",
    "code_sha256",
    "environment_sha256",
    "git_commit",
    "seeds",
    "screen",
    "dev_exposure",
    "confirm_read",
    "cost",
    "failures_and_censoring",
    "capacity_diagnostics",
)
#: benchmark_summary keys required to render the model card honestly
#: (§9/§10: winner, per-candidate ledger, H3, confirm sealed).
EMBEDDING_REQUIRED_BENCHMARK_KEYS: Final[tuple[str, ...]] = (
    "winner_embedding_id",
    "candidates",
    "h3",
    "confirm_read",
)
#: feature_schema keys required (plan §12.3).
EMBEDDING_FEATURE_SCHEMA_REQUIRED: Final[tuple[str, ...]] = (
    "node_dim",
    "legacy_edge_dim",
    "chem_recipe",
    "spatial_recipe",
    "angle_recipe",
    "tie",
    "mask",
    "eps",
    "cache_fingerprint",
)


class EmbeddingBundleError(ExportError):
    """Typed refusal of the V5.4.1 embedding-bundle exporter.

    Carries a stable ``code`` + ``detail`` (repo convention, cf.
    ``V541InterfaceError``); every refusal is fail-closed.
    """

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = str(code)
        self.detail = str(detail)


def _require_embedding(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise EmbeddingBundleError(code, detail)


def build_embedding_feature_schema(embedding_id: str) -> dict[str, JSONValue]:
    """The unified frozen feature schema (plan §12.3 / §6.6–§6.7).

    ONE schema for every winner including E0 (E0 按空分支统一 schema 导出):
    node width 48, legacy edge channels 22 in legacy order, the internal
    chem/spatial/angle recipes with their tie policy, masks, eps values and
    a cache fingerprint hashing the whole recipe body.
    """
    from pes2ts_core.pesn.contracts.key_atom_v3 import EDGE_INPUT_DIM_V3
    from pes2ts_core.pesn.models.v541 import feature_providers as prov
    from pes2ts_core.pesn.models.v541.feature_providers.common import (
        ANGLE_COS_TOL,
        COS_CLAMP_MAX,
        COS_CLAMP_MIN,
        COS_DENOM_EPS,
        RBF_T_CLAMP,
    )
    from pes2ts_core.pesn.models.v541.interfaces import HIDDEN_V541

    schema: dict[str, JSONValue] = {
        "schema_version": EMBEDDING_FEATURE_SCHEMA_ID,
        "embedding_id": str(embedding_id),
        "node_dim": int(HIDDEN_V541),
        "legacy_edge_dim": int(EDGE_INPUT_DIM_V3),
        "edge_channel_order": "legacy",
        "chem_recipe": {
            "fields": list(prov.CHEM_EDGE_FIELDS),
            "dim": int(prov.CHEM_EDGE_DIM),
        },
        "spatial_recipe": {
            "rbf_n": int(prov.RBF_N),
            "center_start_angstrom": float(prov.RBF_CENTER_START),
            "center_end_angstrom": float(prov.RBF_CENTER_END),
            "sigma_angstrom": float(prov.RBF_SIGMA_ANGSTROM),
            "f2_masked_slots_never_read": True,
        },
        "angle_recipe": {
            "max_neighbors_per_center": int(prov.ANGLE_MAX_NEIGHBORS),
            "max_unordered_pairs_per_center": int(prov.ANGLE_MAX_PAIRS),
            "basis": ["1", "c", "c2", "c3"],
            "cos_clamp": [float(COS_CLAMP_MIN), float(COS_CLAMP_MAX)],
            "chirality_labels": False,
        },
        "tie": {
            "angle_tie_policy": str(prov.ANGLE_TIE_POLICY),
            "angle_shell_tie_tol_angstrom": float(prov.ANGLE_SHELL_TIE_TOL),
            "spatial_tie_tol_angstrom": 1e-6,
        },
        "mask": {
            "edge_mask_level": "F2",
            "node_vector_consumed": False,
            "no_angle_token": "explicit mask (zero aggregate + valid mask)",
        },
        "eps": {
            "cos_denom_eps": float(COS_DENOM_EPS),
            "angle_cos_tol": float(ANGLE_COS_TOL),
            "rbf_t_clamp": float(RBF_T_CLAMP),
        },
    }
    body = {k: v for k, v in schema.items() if k != "cache_fingerprint"}
    schema["cache_fingerprint"] = sha256_text(stable_json_dumps(body))
    return schema


def build_downstream_handoff(
    *, model_id: str, embedding_id: str, checkpoint_path: str,
    checkpoint_sha256: str,
) -> dict[str, JSONValue]:
    """The frozen-E* / joint-finetune boundary handed to V5.4.2 (§12.3/§16.2).

    ``frozen-E*`` is the DEFAULT: read-only, ``requires_grad=False``,
    strict-load entry through the exported state mapping.  A
    ``joint-finetune-E*`` run is a SEPARATE preregistered experiment and is
    never implied by this bundle.
    """
    return {
        "schema_version": EMBEDDING_HANDOFF_SCHEMA,
        "model_id": str(model_id),
        "embedding_id": str(embedding_id),
        "frozen": {
            "mode": "frozen-{}".format(embedding_id),
            "read_only": True,
            "requires_grad": False,
            "strict_load": True,
            "strict_load_entry": (
                "V541Model(embedding_id=...).load_state_dict("
                "legacy_state | branch_state, strict=True)"
            ),
            "weights_file": "embedding_weights.pt",
            "weights_payload": (
                "legacy_state (backbone.* + process_head.*) merged with "
                "branch_state (embedding.* / w_out.weight / alpha; EMPTY for "
                "E0 — the empty-branch unified schema)"
            ),
        },
        "joint_finetune": {
            "requires_separate_experiment": True,
            "experiment": "V5.4.2 joint-finetune-E* (separate preregistration)",
            "not_implied_by_this_bundle": True,
        },
        "reference_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": str(checkpoint_sha256),
            "note": (
                "immutable V541 champion checkpoint reference — the full "
                "MOVE/PROCESS model lives there; embedding weights alone are "
                "not a full model reproduction package (§12.3)"
            ),
        },
    }


def build_embedding_model_card(
    *, model_id: str, embedding_id: str,
    provenance: Mapping[str, JSONValue],
    benchmark_summary: Mapping[str, JSONValue],
    checkpoint_sha256: str,
) -> str:
    """Model card stating every disclosure the plan §12.3 requires.

    Renders the per-candidate ledger, H2/H3 outcomes, cost gate and stage
    skips FROM THE PASSED EVIDENCE (never hard-coded), and always carries
    the claim boundaries: common-core warmstart, dev exposure, unread
    confirm, failures/censoring, unclosed capacity diagnostics, and the
    explicit non-claims (no full original-paper model reproduction, no
    verified/strict TS, no deployment qualification).
    """
    candidates = benchmark_summary.get("candidates")
    stages = benchmark_summary.get("stages")
    lines = [
        "# {} — V5.4.1 embedding champion research model card".format(model_id),
        "",
        "- generated: {}".format(
            datetime.now(UTC).isoformat(timespec="seconds")),
        "- bundle schema: {}".format(EMBEDDING_BUNDLE_SCHEMA),
        "- winner embedding: {} (E* may be E0 — no stable,足够大的收益时 "
        "E0 留任, plan §9.5/§10.5)".format(embedding_id),
        "- reference checkpoint sha256: {}".format(checkpoint_sha256),
        "- common-core warmstart: R3 frozen StarEGNNV3 init + legacy MOVE "
        "readout + ReactionProcessHeadV53MP; the embedding branch is a "
        "zero-gate residual (alpha_0=0) on h_legacy (plan §5.2/§5.4).",
        "- scope: MOVE + PROCESS research outputs only (TS_candidate / "
        "endpoint_reached); research_only, not for production control.",
        "",
        "## Seeds (three-seed ledger)",
        "```json",
        stable_json_dumps(dict(provenance.get("seeds") or {})),
        "```",
        "",
        "## Inner-val screen (single-seed初筛; dev 不参与排序, §9.3)",
        "```json",
        stable_json_dumps(dict(provenance.get("screen") or {})),
        "```",
        "",
        "## Per-candidate outcomes (seeds / gates / censoring)",
    ]
    if isinstance(candidates, Mapping):
        for eid in sorted(candidates):
            lines.append("- {}: {}".format(
                eid, stable_json_dumps(dict(candidates[eid]))))
    else:
        lines.append("- (none supplied)")
    if isinstance(stages, Mapping):
        lines.append("")
        lines.append("## Stage dispositions")
        for name in sorted(stages):
            lines.append("- {}: {}".format(
                name, stable_json_dumps(dict(stages[name]))
                if isinstance(stages[name], Mapping)
                else str(stages[name])))
    lines += [
        "",
        "## H3 (three-seed)",
        "```json",
        stable_json_dumps(dict(benchmark_summary.get("h3") or {})),
        "```",
        "",
        "## Dev exposure",
        "```json",
        stable_json_dumps(dict(provenance.get("dev_exposure") or {})),
        "```",
        "- confirm curves were NEVER read (confirm_read=false); confirm "
        "stays sealed until separate approval (§10.1/§17).",
        "",
        "## Resource cost",
        "```json",
        stable_json_dumps(dict(provenance.get("cost") or {})),
        "```",
        "",
        "## Failures and censoring (full ledger, not averages over "
        "survivors only, §10.3)",
        "```json",
        stable_json_dumps(dict(provenance.get("failures_and_censoring") or {})),
        "```",
        "",
        "## Capacity/depth diagnostics (T4; 未闭合混杂单列, §9.4)",
        "```json",
        stable_json_dumps(dict(provenance.get("capacity_diagnostics") or {})),
        "```",
        "- if these diagnostics are not closed, capacity/depth confounds "
        "remain UNEXCLUDED and the winner attribution is narrowed "
        "accordingly; a diagnostic group is never silently upgraded to the "
        "V5.4.1 winner.",
        "",
        "## Claim boundaries (explicit non-claims)",
        "- This bundle does NOT reproduce any full original-paper GNN "
        "(GINE/PNA/SchNet-like/Angle-MPNN are lightweight adaptations; E6 "
        "is scalar-projected diagnostics only, plan §1.1/§6.5).",
        "- NO strict/verified TS claim: verified TS requires OptTS + single "
        "imaginary frequency + bidirectional IRC + endpoint identity; only "
        "TS_candidate / endpoint_reached are claimed.",
        "- NOT deployment qualification: training qualification is not "
        "deployment qualification; no online control authority is granted.",
        "- frozen-E* handoff is read-only (requires_grad=false); a "
        "joint-finetune-E* run requires a SEPARATE preregistered experiment "
        "(V5.4.2) and never silently changes this stage's winner (§16.1).",
        "- research_only: true.",
    ]
    return "\n".join(lines) + "\n"


def _partition_v541_state(
        state: Mapping[str, Any], embedding_id: str,
        checkpoint: Path) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Split a V541 champion state dict into (legacy_state, branch_state).

    Fail-closed: every key must be a tensor and must belong to the legacy
    view (``backbone.*`` / ``process_head.*``) or to the residual branch
    (``embedding.*`` / ``w_out.*`` / ``alpha``).  E0 MUST have an EMPTY
    branch (空分支统一 schema); E1–E5 MUST carry ``alpha`` + ``w_out.weight``
    + at least one ``embedding.*`` key.
    """
    from pes2ts_core.pesn.models.v541.model import LEGACY_STATE_PREFIXES

    import torch

    legacy: Dict[str, Any] = {}
    branch: Dict[str, Any] = {}
    for key, value in state.items():
        _require_embedding(
            isinstance(value, torch.Tensor),
            "V541_WEIGHTS_TENSOR",
            "state entry {!r} in {} is not a tensor".format(
                key, checkpoint))
        if key.startswith(LEGACY_STATE_PREFIXES):
            legacy[key] = value
        elif (key.startswith("embedding.") or key.startswith("w_out.")
              or key == "alpha"):
            branch[key] = value
        else:
            raise EmbeddingBundleError(
                "V541_WEIGHTS_UNEXPECTED_KEY",
                "state entry {!r} belongs to neither the legacy view nor "
                "the residual branch (plan §5.2/§12.3)".format(key))
    _require_embedding(
        bool(legacy), "V541_WEIGHTS_LEGACY_EMPTY",
        "checkpoint carries no legacy backbone/process_head state")
    if str(embedding_id) == "E0":
        _require_embedding(
            not branch, "V541_WEIGHTS_E0_BRANCH",
            "E0 must export the EMPTY-branch unified schema (e0_has_no_branch); "
            "found branch keys {}".format(sorted(branch)))
    else:
        _require_embedding(
            "alpha" in branch, "V541_WEIGHTS_GATE_MISSING",
            "{} checkpoint has no 'alpha' gate parameter".format(embedding_id))
        _require_embedding(
            "w_out.weight" in branch, "V541_WEIGHTS_W_MISSING",
            "{} checkpoint has no 'w_out.weight' projection".format(embedding_id))
        _require_embedding(
            any(k.startswith("embedding.") for k in branch),
            "V541_WEIGHTS_BRANCH_MISSING",
            "{} checkpoint has no embedding.* branch parameters".format(
                embedding_id))
    return legacy, branch


def _validate_embedding_provenance(
        provenance: Mapping[str, JSONValue], *, checkpoint_sha256: str,
        seeds_required: Sequence[int]) -> None:
    """Fail-closed provenance audit (plan §12.3 evidence blocks)."""
    _require_embedding(
        isinstance(provenance, Mapping),
        "V541_PROVENANCE_TYPE", "provenance must be a mapping")
    missing = [k for k in EMBEDDING_REQUIRED_PROVENANCE
               if k not in provenance]
    _require_embedding(
        not missing, "V541_PROVENANCE_MISSING",
        "provenance is missing required key(s): {}".format(
            ", ".join(missing)))
    for key in EMBEDDING_REQUIRED_PROVENANCE:
        _require_embedding(
            provenance[key] is not None,
            "V541_PROVENANCE_NULL",
            "provenance[{!r}] must not be null".format(key))
    _require_embedding(
        provenance["confirm_read"] is False,
        "V541_PROVENANCE_CONFIRM",
        "confirm_read must be exactly False (confirm 永封存, §17); got "
        "{!r}".format(provenance["confirm_read"]))
    seeds = provenance["seeds"]
    _require_embedding(
        isinstance(seeds, Mapping),
        "V541_PROVENANCE_SEEDS",
        "provenance['seeds'] must be a mapping keyed by seed; got "
        "{}".format(type(seeds).__name__))
    covered = {int(k) for k in seeds}
    needed = {int(s) for s in seeds_required}
    _require_embedding(
        needed <= covered, "V541_PROVENANCE_SEEDS",
        "provenance['seeds'] covers {} but the three-seed ledger requires "
        "{}".format(sorted(covered), sorted(needed)))
    given = provenance.get("checkpoint_sha256")
    if given is not None:
        _require_embedding(
            str(given) == str(checkpoint_sha256),
            "V541_PROVENANCE_CHECKPOINT_DRIFT",
            "provenance checkpoint_sha256={} disagrees with the champion "
            "checkpoint file {}".format(given, checkpoint_sha256))


def _validate_embedding_benchmark(
        benchmark_summary: Mapping[str, JSONValue], embedding_id: str,
        *, confirm_read: JSONValue) -> None:
    _require_embedding(
        isinstance(benchmark_summary, Mapping),
        "V541_BENCHMARK_TYPE", "benchmark_summary must be a mapping")
    missing = [k for k in EMBEDDING_REQUIRED_BENCHMARK_KEYS
               if k not in benchmark_summary]
    _require_embedding(
        not missing, "V541_BENCHMARK_MISSING",
        "benchmark_summary is missing required key(s): {}".format(
            ", ".join(missing)))
    _require_embedding(
        str(benchmark_summary["winner_embedding_id"]) == str(embedding_id),
        "V541_BENCHMARK_WINNER",
        "benchmark_summary.winner_embedding_id={!r} != export embedding_id="
        "{!r}".format(benchmark_summary["winner_embedding_id"], embedding_id))
    _require_embedding(
        isinstance(benchmark_summary["candidates"], Mapping),
        "V541_BENCHMARK_CANDIDATES",
        "benchmark_summary['candidates'] must be a per-candidate mapping")
    _require_embedding(
        benchmark_summary["confirm_read"] is False
        and confirm_read is False,
        "V541_BENCHMARK_CONFIRM",
        "benchmark_summary.confirm_read must be exactly False")


def verify_embedding_bundle(bundle_dir: str | Path) -> dict[str, str]:
    """Post-hoc whitelist + SHA audit of an exported bundle (§12.3).

    Re-reads the directory: it must contain EXACTLY
    :data:`EMBEDDING_BUNDLE_FILES`, ``checksums.json`` must cover every
    other file, and every recorded digest must match the bytes on disk.
    Returns the checksum map; raises :class:`EmbeddingBundleError` on any
    drift (extra/missing file, unhashed file, digest mismatch).
    """
    bundle = Path(bundle_dir)
    _require_embedding(
        bundle.is_dir(), "V541_BUNDLE_DIR_MISSING", str(bundle))
    present = {p.name for p in bundle.iterdir() if p.is_file()}
    expected = set(EMBEDDING_BUNDLE_FILES)
    _require_embedding(
        present == expected, "V541_BUNDLE_WHITELIST_DRIFT",
        "{} must contain exactly {}; unexpected={} missing={}".format(
            bundle, list(EMBEDDING_BUNDLE_FILES),
            sorted(present - expected), sorted(expected - present)))
    from pes2ts_core.utils.jsonio import read_json

    checksums = read_json(bundle / "checksums.json")
    _require_embedding(
        isinstance(checksums, Mapping),
        "V541_BUNDLE_CHECKSUMS_CORRUPT",
        "checksums.json is not a mapping")
    covered = expected - {"checksums.json"}
    _require_embedding(
        set(checksums) == covered, "V541_BUNDLE_CHECKSUMS_COVERAGE",
        "checksums.json must cover exactly {}; extra={} missing={}".format(
            sorted(covered),
            sorted(set(checksums) - covered),
            sorted(covered - set(checksums))))
    for name in sorted(covered):
        actual = sha256_file(bundle / name)
        _require_embedding(
            actual == checksums[name], "V541_BUNDLE_CHECKSUM_MISMATCH",
            "{}: recorded {} but file is {}".format(
                name, checksums[name], actual))
    return {str(k): str(v) for k, v in checksums.items()}


def export_embedding_bundle(
    *,
    run_dir: str | Path,
    output_dir: str | Path,
    model_id: str,
    embedding_id: str,
    embedding_config: Mapping[str, JSONValue],
    feature_schema: Mapping[str, JSONValue],
    provenance: Mapping[str, JSONValue],
    benchmark_summary: Mapping[str, JSONValue],
    checkpoint_name: str = "checkpoint_best.pt",
    seeds_required: Sequence[int] = (17, 29, 43),
) -> Path:
    """Assemble the V5.4.1 ``pes2ts_embedding_bundle_v1`` whitelist bundle.

    Writes exactly :data:`EMBEDDING_BUNDLE_FILES`:

    * ``embedding_config.yaml`` — the frozen per-candidate config slice
      (registry row + residual + core-consumption + feature recipe);
    * ``embedding_weights.pt`` — ``legacy_state`` (backbone.* +
      process_head.*, the legacy 必需映射) merged with ``branch_state``
      (``embedding.*`` / ``w_out.weight`` / ``alpha``; EMPTY dict for E0 —
      the empty-branch unified schema) PLUS the immutable full-V541
      checkpoint reference ``{path, sha256}`` (never lost, §12.3);
    * ``feature_schema.json`` — node 48 / legacy edge 22 / internal
      chem+spatial+angle recipes / tie / mask / eps / cache fingerprint;
    * ``provenance.json`` — the §12.2 identity SHAs + §12.3 evidence
      blocks (three seeds, screen, dev exposure, unread confirm, cost,
      failures/censoring, unclosed capacity diagnostics);
    * ``benchmark_summary.json`` / ``model_card.md`` /
      ``downstream_handoff.json`` (frozen-E* requires_grad=false
      strict-load entry; joint-finetune = separate experiment);
    * ``checksums.json`` — then a POST-hoc whitelist + SHA re-audit.

    Typed refusals (:class:`EmbeddingBundleError`, a subclass of
    :class:`ExportError`): unknown/diagnostic-only embedding id, missing or
    non-tensor state, E0 with a branch (or a candidate without gate/W/
    branch), incomplete provenance, benchmark/provenance drift, winner
    mismatch, and any whitelist/checksum drift.
    """
    from pes2ts_core.pesn.models.v541.embedding_registry import (
        FROZEN_EMBEDDING_IDS,
    )

    eid = str(embedding_id)
    _require_embedding(
        eid in FROZEN_EMBEDDING_IDS, "V541_EMBEDDING_UNKNOWN",
        "unknown embedding id {!r}; frozen ids {}".format(
            embedding_id, list(FROZEN_EMBEDDING_IDS)))
    _require_embedding(
        eid != "E6", "V541_EMBEDDING_DIAGNOSTIC",
        "E6 is a scalar-projected diagnostic and can never be a champion "
        "bundle winner (config embedding_registry.E6.diagnostic_only)")

    run_path = Path(run_dir)
    checkpoint = run_path / str(checkpoint_name)
    _require_embedding(
        checkpoint.is_file(), "V541_CHECKPOINT_MISSING", str(checkpoint))

    from pes2ts_core.pesn.models import BackendMissingError

    try:
        import torch  # noqa: F811 - local alias for the payload load
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise BackendMissingError("torch", "embedding bundle export") from exc

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _require_embedding(
        isinstance(payload, Mapping) and isinstance(payload.get("model"),
                                                     Mapping),
        "V541_CHECKPOINT_PAYLOAD",
        "checkpoint payload must carry the model state under the 'model' "
        "key (V541 trainer contract)")
    state = payload["model"]
    legacy_state, branch_state = _partition_v541_state(state, eid, checkpoint)
    checkpoint_sha = sha256_file(checkpoint)

    # -- evidence audits (fail-closed BEFORE writing anything) --------------
    _validate_embedding_provenance(
        provenance, checkpoint_sha256=checkpoint_sha,
        seeds_required=seeds_required)
    _validate_embedding_benchmark(
        benchmark_summary, eid, confirm_read=provenance["confirm_read"])
    _require_embedding(
        isinstance(feature_schema, Mapping),
        "V541_FEATURE_SCHEMA_TYPE", "feature_schema must be a mapping")
    missing_schema = [k for k in EMBEDDING_FEATURE_SCHEMA_REQUIRED
                      if k not in feature_schema]
    _require_embedding(
        not missing_schema, "V541_FEATURE_SCHEMA_MISSING",
        "feature_schema is missing required key(s): {}".format(
            ", ".join(missing_schema)))
    _require_embedding(
        int(feature_schema["node_dim"]) == EMBEDDING_NODE_DIM
        and int(feature_schema["legacy_edge_dim"]) == EMBEDDING_LEGACY_EDGE_DIM,
        "V541_FEATURE_SCHEMA_DIMS",
        "feature_schema must declare node_dim={} and legacy_edge_dim={}; "
        "got {}/{}".format(
            EMBEDDING_NODE_DIM, EMBEDDING_LEGACY_EDGE_DIM,
            feature_schema["node_dim"], feature_schema["legacy_edge_dim"]))
    _require_embedding(
        isinstance(embedding_config, Mapping) and bool(embedding_config),
        "V541_EMBEDDING_CONFIG",
        "embedding_config must be a non-empty mapping")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # -- 1. embedding_weights.pt (legacy必需映射 + branch + checkpoint ref) --
    weights_doc: dict[str, JSONValue] = {
        "schema": EMBEDDING_WEIGHTS_SCHEMA,
        "embedding_id": eid,
        "model_id": str(model_id),
        "has_branch": bool(branch_state),
        "legacy_state": legacy_state,
        "branch_state": branch_state,
        "checkpoint_ref": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha,
        },
    }
    for key in BUNDLE_FORBIDDEN_KEYS:
        _require_embedding(
            key not in weights_doc, "V541_WEIGHTS_FORBIDDEN_KEY",
            "weights payload would carry forbidden key {!r}".format(key))
    torch.save(weights_doc, out / "embedding_weights.pt")

    # -- 2. embedding_config.yaml -------------------------------------------
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise BackendMissingError("yaml", "embedding bundle export") from exc
    config_doc: dict[str, JSONValue] = {
        "schema_version": "pes2ts_v541_embedding_config_v1",
        "bundle_schema": EMBEDDING_BUNDLE_SCHEMA,
        "model_id": str(model_id),
        "embedding_id": eid,
        "config": dict(embedding_config),
    }
    (out / "embedding_config.yaml").write_text(
        yaml.safe_dump(config_doc, sort_keys=True, allow_unicode=True),
        encoding="utf-8")

    # -- 3. feature_schema.json ---------------------------------------------
    from pes2ts_core.utils.jsonio import write_json

    write_json(out / "feature_schema.json", dict(feature_schema))

    # -- 4. provenance.json (+ checkpoint binding) ---------------------------
    provenance_doc: dict[str, JSONValue] = dict(provenance)
    provenance_doc.update({
        "export_schema": EMBEDDING_BUNDLE_SCHEMA,
        "model_id": str(model_id),
        "embedding_id": eid,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
    })
    write_json(out / "provenance.json", provenance_doc)

    # -- 5. benchmark_summary.json ------------------------------------------
    write_json(out / "benchmark_summary.json", dict(benchmark_summary))

    # -- 6. downstream_handoff.json -----------------------------------------
    handoff = build_downstream_handoff(
        model_id=str(model_id), embedding_id=eid,
        checkpoint_path=str(checkpoint), checkpoint_sha256=checkpoint_sha)
    write_json(out / "downstream_handoff.json", handoff)

    # -- 7. model_card.md ----------------------------------------------------
    model_card = build_embedding_model_card(
        model_id=str(model_id), embedding_id=eid, provenance=provenance_doc,
        benchmark_summary=benchmark_summary, checkpoint_sha256=checkpoint_sha)
    (out / "model_card.md").write_text(model_card, encoding="utf-8")

    # -- 8. checksums.json + POST-hoc whitelist/SHA audit --------------------
    checksums: dict[str, str] = {}
    for name in EMBEDDING_BUNDLE_FILES:
        if name == "checksums.json":
            continue
        checksums[name] = sha256_file(out / name)
    write_json(out / "checksums.json", checksums)
    verify_embedding_bundle(out)
    return out


__all__ = [
    "EMBEDDING_BUNDLE_FILES",
    "EMBEDDING_BUNDLE_SCHEMA",
    "EMBEDDING_FEATURE_SCHEMA_ID",
    "EMBEDDING_FEATURE_SCHEMA_REQUIRED",
    "EMBEDDING_HANDOFF_SCHEMA",
    "EMBEDDING_LEGACY_EDGE_DIM",
    "EMBEDDING_NODE_DIM",
    "EMBEDDING_REQUIRED_BENCHMARK_KEYS",
    "EMBEDDING_REQUIRED_PROVENANCE",
    "EMBEDDING_WEIGHTS_SCHEMA",
    "EXPORT_SCHEMA",
    "EmbeddingBundleError",
    "ExportError",
    "MOVE_PROCESS_BUNDLE_FILES",
    "MOVE_PROCESS_BUNDLE_SCHEMA",
    "MOVE_PROCESS_REQUIRED_PROVENANCE",
    "MOVE_PROCESS_TAU_TARGET_PRECISION",
    "build_downstream_handoff",
    "build_embedding_feature_schema",
    "build_embedding_model_card",
    "build_feature_spec_document",
    "build_model_card",
    "export_bundle",
    "export_embedding_bundle",
    "export_move_process_bundle",
    "verify_embedding_bundle",
]
