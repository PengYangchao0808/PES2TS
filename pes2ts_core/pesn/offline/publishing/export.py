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


__all__ = [
    "EXPORT_SCHEMA",
    "ExportError",
    "build_feature_spec_document",
    "build_model_card",
    "export_bundle",
]
