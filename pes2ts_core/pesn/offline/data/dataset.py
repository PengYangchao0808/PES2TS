"""Isolated teacher dataset construction and shard export (B2).

For every selected train-side reaction this builder

1. adapts the frozen G1 authority documents into the plain task-graph /
   coordinate-spec contracts (``endpoint_adapter``);
2. reads the IRC trajectory through the single audited, scope-checked truth
   bridge (``truth_source``) — never directly;
3. re-verifies the claimed branch orientation empirically against the frozen
   endpoint bond pattern (covalent-radius bondedness at the branch terminus)
   and records a typed quality flag on disagreement;
4. computes the progress / direction / activity labels (``labels``);
5. exports per-shard state views (plain float arrays), label views, and a
   provenance view that keeps frame indices, branch truth, TS references, and
   exclusion reasons on the isolated side.

Sample identity: ``sample_id = f"{reaction_id}#f{frame_index}"``.  The state
view stores only what online feature building may see (elements + current
geometry); references to neighbouring frames, TS geometry, and any label
material live in the label/provenance views.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Mapping, Sequence

import numpy as np

from pes2ts_core.g1.index_map import COVALENT_RADII
from pes2ts_core.pesn.contracts.artifacts import (
    DATASET_SCHEMA_VERSION,
    LABELS_SCHEMA_VERSION,
    LABEL_SPEC_SCHEMA_VERSION,
    validate_dataset_manifest,
    validate_labels_manifest,
)
from pes2ts_core.pesn.offline.data.endpoint_adapter import (
    adapt_reaction,
    load_authority_document,
    to_coordinate_spec,
    to_task_graph,
)
from pes2ts_core.pesn.offline.data.truth_source import (
    TruthBridgeError,
    load_scoped_irc_frames,
    stream_scoped_irc_frames,
)
from pes2ts_core.pesn.offline.labels.direction import (
    ACTIVITY_RULE_VERSION,
    DIFFERENCE_RULE_VERSION,
    DirectionLabelError,
    compute_direction_labels,
    evaluate_distance_coords,
)
from pes2ts_core.pesn.offline.labels.progress import (
    MASSES_VERSION,
    ProgressLabelError,
    compute_progress_labels,
    masses_for,
)
from pes2ts_core.utils.hashing import JSONValue, sha256_file, stable_json_dumps
from pes2ts_core.utils.jsonio import write_json

DATASET_VERSION: Final[str] = "0.1.0"
LABEL_SPEC_VERSION: Final[str] = "pesn_irc_teacher_labelspec_v1.0.0"
BOND_TOLERANCE_ANGSTROM: Final[float] = 0.45
SHARD_SIZE: Final[int] = 200
#: Above this selection size, per-reaction audited reads (each an O(archive)
#: group scan) are replaced by one audited streaming pass through the bridge.
PER_REACTION_LOOKUP_THRESHOLD: Final[int] = 512

ORIENTATION_CONFIRMED: Final[str] = "confirmed"
ORIENTATION_FLAGGED: Final[str] = "flagged_terminus_pattern_mismatch"


class DatasetBuildError(RuntimeError):
    """Typed failure of teacher dataset construction."""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deterministic_selection(candidates: Sequence[str], *, seed: int, limit: int) -> list[str]:
    """Rank by sha256(seed:rid) and take the first *limit* (sorted output)."""
    ranked = sorted(candidates, key=lambda rid: (int(_sha256_text(f"{seed}:{rid}")[:16], 16), rid))
    return sorted(ranked[:limit])


def build_label_spec() -> dict[str, JSONValue]:
    return {
        "schema_version": LABEL_SPEC_SCHEMA_VERSION,
        "label_spec_version": LABEL_SPEC_VERSION,
        "masses_version": MASSES_VERSION,
        "arc_definition": "mass_weighted_adjacent_kabsch_sqrt_m_delta_sq",
        "arc_units": "angstrom_sqrt_u",
        "registration": "mass_weighted_kabsch_adjacent_frames",
        "signed_progress": {
            "definition": "-d_arc/L_R on R side, 0 at TS, +d_arc/L_P on P side",
            "swap_semantics": "sign flip under R/P exchange; |s| invariant",
        },
        "branch_layouts": ["ts_first_dual_branch", "single_branch"],
        "endpoint_semantics": "recorded_terminus_only_never_reaction_complete",
        "direction": {
            "z_convention": "z = L_branch - d_arc (increases toward TS)",
            "difference_rule": DIFFERENCE_RULE_VERSION,
            "normalization": "unit_norm_with_validity_check",
            "ts_anchor": "masked",
        },
        "activity": {"rule": ACTIVITY_RULE_VERSION, "threshold": "1/n_coords"},
        "driver_coordinates": "exclusive F/B/O edit-pair distances (map-order rows)",
        "masking": "missing labels carry typed reasons, never zero-filled",
        "content_sha256": "pending",
    }


def _finalize_label_spec_hash(spec: dict[str, JSONValue]) -> dict[str, JSONValue]:
    body = {k: v for k, v in spec.items() if k != "content_sha256"}
    spec["content_sha256"] = _sha256_text(stable_json_dumps(body))
    return spec


def _terminus_bonded_pattern_ok(
    terminus_coords: np.ndarray,
    atomic_numbers: np.ndarray,
    bonded_pairs: set[tuple[int, int]],
    broken_pairs: set[tuple[int, int]],
) -> bool:
    """Check the branch terminus satisfies its side's frozen bond pattern."""
    for i, j in bonded_pairs:
        radius = COVALENT_RADII.get(int(atomic_numbers[i]), 0.0) + COVALENT_RADII.get(
            int(atomic_numbers[j]), 0.0
        )
        distance = float(np.linalg.norm(terminus_coords[i] - terminus_coords[j]))
        if distance > radius + BOND_TOLERANCE_ANGSTROM:
            return False
    for i, j in broken_pairs:
        radius = COVALENT_RADII.get(int(atomic_numbers[i]), 0.0) + COVALENT_RADII.get(
            int(atomic_numbers[j]), 0.0
        )
        distance = float(np.linalg.norm(terminus_coords[i] - terminus_coords[j]))
        if distance <= radius + BOND_TOLERANCE_ANGSTROM:
            return False
    return True


def _verify_orientation(
    frames: np.ndarray,
    atomic_numbers: np.ndarray,
    task_graph,
    progress_labels,
    orientation: str,
) -> str:
    """Empirically confirm the claimed orientation at the branch terminus."""
    side_of_b1 = 1 if orientation == "P_first" else -1
    edits = task_graph.edits
    formed = {
        (e.i, e.j) for e in edits if e.kind == "formed"
    }
    broken = {
        (e.i, e.j) for e in edits if e.kind == "broken"
    }
    side_arr = progress_labels.side
    for target_side in (-1, 1):
        frames_on_side = np.where(side_arr == target_side)[0]
        if frames_on_side.size == 0:
            continue
        terminus = int(frames_on_side[np.argmax(progress_labels.d_arc[frames_on_side])])
        if target_side == side_of_b1:
            bonded, unbonded = formed, broken
        else:
            bonded, unbonded = broken, formed
        expected_ok = _terminus_bonded_pattern_ok(
            frames[terminus], atomic_numbers, bonded, unbonded
        )
        if not expected_ok:
            return ORIENTATION_FLAGGED
    return ORIENTATION_CONFIRMED


@dataclass(frozen=True, slots=True)
class ReactionRecord:
    """Everything one reaction contributes to the teacher dataset."""

    reaction_id: str
    coords: np.ndarray  # (F, N, 3) float32
    atomic_numbers: np.ndarray  # (N,) int
    task_graph_doc: dict[str, JSONValue]
    coordinate_spec_doc: dict[str, JSONValue]
    progress: object
    direction: object
    provenance: dict[str, JSONValue]
    exclusion: str | None


def prepare_authority(
    reaction_id: str,
    *,
    interim_dir: Path,
    shard_size: int,
) -> dict[str, object] | tuple[str, str]:
    """Load and adapt the G1 authority documents for one reaction.

    Returns the assembled parts dict, or a ``(exclusion_code, reason)``
    tuple as a typed exclusion.
    """
    try:
        docs = load_authority_document(interim_dir, reaction_id, shard_size=shard_size)
        adapter = adapt_reaction(docs)
        task_graph = to_task_graph(adapter)
        coord_spec = to_coordinate_spec(adapter)
    except Exception as exc:
        return ("ADAPTER_FAILED", f"{type(exc).__name__}:{exc}")
    return {
        "reaction_id": reaction_id,
        "adapter": adapter,
        "task_graph": task_graph,
        "coord_spec": coord_spec,
    }


def record_from_frames(parts: dict[str, object], frames_doc: Mapping[str, object]) -> ReactionRecord:
    """Compute labels for one prepared reaction from scoped plain frames."""
    reaction_id = str(parts["reaction_id"])
    adapter = parts["adapter"]
    task_graph = parts["task_graph"]
    coord_spec = parts["coord_spec"]
    coords = np.asarray(frames_doc["coordinates"], dtype=np.float64)
    n_atoms = int(frames_doc["n_atoms"])
    if coords.shape != (int(frames_doc["n_frames"]), n_atoms, 3) or n_atoms != task_graph.n_atoms:
        return ReactionRecord(
            reaction_id=reaction_id,
            coords=np.zeros(0, dtype=np.float32),
            atomic_numbers=np.zeros(0, dtype=int),
            task_graph_doc={},
            coordinate_spec_doc={},
            progress=None,
            direction=None,
            provenance={"reason": f"frame_shape_mismatch:{coords.shape}/{n_atoms}"},
            exclusion="FRAME_SHAPE_MISMATCH",
        )

    atomic_numbers = np.asarray(task_graph.atomic_numbers, dtype=int)
    masses = masses_for(atomic_numbers)
    try:
        progress = compute_progress_labels(
            coords,
            masses,
            ts_frame_index=int(adapter["ts_frame_index"]),
            splice_index=(
                int(adapter["splice_index"])
                if adapter["splice_index"] is not None
                else None
            ),
            layout=str(adapter["layout"]),
            orientation=str(adapter["orientation"]),
        )
    except ProgressLabelError as exc:
        return ReactionRecord(
            reaction_id=reaction_id,
            coords=coords.astype(np.float32),
            atomic_numbers=atomic_numbers,
            task_graph_doc=_graph_doc(task_graph),
            coordinate_spec_doc=_coord_doc(coord_spec),
            progress=None,
            direction=None,
            provenance={"reason": f"progress_error:{exc}"},
            exclusion="PROGRESS_LABEL_FAILED",
        )

    coord_pairs = [(c.atoms[0], c.atoms[1]) for c in coord_spec.coords]
    try:
        q = evaluate_distance_coords(coords, coord_pairs)
        direction = compute_direction_labels(
            q,
            progress,
            scale_a=np.array([c.scale_a for c in coord_spec.coords]),
            ref_q=np.array([c.ref_q for c in coord_spec.coords]),
        )
    except DirectionLabelError as exc:
        return ReactionRecord(
            reaction_id=reaction_id,
            coords=coords.astype(np.float32),
            atomic_numbers=atomic_numbers,
            task_graph_doc=_graph_doc(task_graph),
            coordinate_spec_doc=_coord_doc(coord_spec),
            progress=None,
            direction=None,
            provenance={"reason": f"direction_error:{exc}"},
            exclusion="DIRECTION_LABEL_FAILED",
        )

    orientation_check = _verify_orientation(
        coords, atomic_numbers, task_graph, progress, str(adapter["orientation"])
    )
    provenance = {
        "source_ref": "B3LYPD3_TZVP_IRC",
        "source_method": "B3LYP-D3/TZVP",
        "electronic_state": "S0",
        "layout": adapter["layout"],
        "ts_frame_index": adapter["ts_frame_index"],
        "splice_index": adapter["splice_index"],
        "orientation": adapter["orientation"],
        "orientation_check": orientation_check,
        "endpoint_status": progress.endpoint_status,
        "branch_totals": {
            "L_R": progress.branch_total_r,
            "L_P": progress.branch_total_p,
        },
        "frame_reasons_progress": progress.frame_reasons,
        "frame_reasons_direction": direction.frame_reasons,
        "mapping_provenance": adapter["mapping_provenance"],
        "atom_rows_follow_map_order": True,
        "n_frames": int(coords.shape[0]),
        "l0u_family": adapter.get("l0u_family"),
    }
    return ReactionRecord(
        reaction_id=reaction_id,
        coords=coords.astype(np.float32),
        atomic_numbers=atomic_numbers,
        task_graph_doc=_graph_doc(task_graph),
        coordinate_spec_doc=_coord_doc(coord_spec),
        progress=progress,
        direction=direction,
        provenance=provenance,
        exclusion=None,
    )


def build_reaction_record(
    reaction_id: str,
    *,
    interim_dir: Path,
    manifests_dir: Path,
    split_dir: Path,
    allow_truth: bool,
    shard_size: int,
) -> ReactionRecord:
    """Build labels for one reaction, or return a typed exclusion record."""
    parts = prepare_authority(reaction_id, interim_dir=interim_dir, shard_size=shard_size)
    if isinstance(parts, tuple):
        code, reason = parts
        return ReactionRecord(
            reaction_id=reaction_id,
            coords=np.zeros(0, dtype=np.float32),
            atomic_numbers=np.zeros(0, dtype=int),
            task_graph_doc={},
            coordinate_spec_doc={},
            progress=None,
            direction=None,
            provenance={"reason": reason},
            exclusion=code,
        )
    try:
        frames_doc = load_scoped_irc_frames(
            reaction_id,
            allow_truth=allow_truth,
            manifests_dir=manifests_dir,
            split_dir=split_dir,
        )
    except (TruthBridgeError, KeyError) as exc:
        return ReactionRecord(
            reaction_id=reaction_id,
            coords=np.zeros(0, dtype=np.float32),
            atomic_numbers=np.zeros(0, dtype=int),
            task_graph_doc={},
            coordinate_spec_doc={},
            progress=None,
            direction=None,
            provenance={"reason": f"truth_access_refused:{exc}"},
            exclusion="TRUTH_ACCESS_REFUSED",
        )
    return record_from_frames(parts, frames_doc)


def build_records_streaming(
    reaction_ids: Sequence[str],
    *,
    interim_dir: Path,
    manifests_dir: Path,
    split_dir: Path,
    allow_truth: bool,
    shard_size: int,
) -> list[ReactionRecord]:
    """One audited scoped stream pass; selection order defines output order."""
    prepared: dict[str, dict[str, object]] = {}
    records: list[ReactionRecord] = []
    for reaction_id in reaction_ids:
        parts = prepare_authority(reaction_id, interim_dir=interim_dir, shard_size=shard_size)
        if isinstance(parts, tuple):
            code, reason = parts
            records.append(
                ReactionRecord(
                    reaction_id=reaction_id,
                    coords=np.zeros(0, dtype=np.float32),
                    atomic_numbers=np.zeros(0, dtype=int),
                    task_graph_doc={},
                    coordinate_spec_doc={},
                    progress=None,
                    direction=None,
                    provenance={"reason": reason},
                    exclusion=code,
                )
            )
        else:
            prepared[reaction_id] = parts
    seen: set[str] = set()
    for frames_doc in stream_scoped_irc_frames(
        frozenset(prepared),
        allow_truth=allow_truth,
        manifests_dir=manifests_dir,
        split_dir=split_dir,
    ):
        reaction_id = str(frames_doc["reaction_id"])
        seen.add(reaction_id)
        records.append(record_from_frames(prepared[reaction_id], frames_doc))
    for reaction_id in prepared:
        if reaction_id not in seen:
            records.append(
                ReactionRecord(
                    reaction_id=reaction_id,
                    coords=np.zeros(0, dtype=np.float32),
                    atomic_numbers=np.zeros(0, dtype=int),
                    task_graph_doc={},
                    coordinate_spec_doc={},
                    progress=None,
                    direction=None,
                    provenance={"reason": "absent_from_irc_archive"},
                    exclusion="TRUTH_ACCESS_REFUSED",
                )
            )
    order = {rid: index for index, rid in enumerate(reaction_ids)}
    records.sort(key=lambda record: order.get(record.reaction_id, len(order)))
    return records


def _graph_doc(task_graph) -> dict[str, JSONValue]:
    return {
        "atomic_numbers": list(task_graph.atomic_numbers),
        "atom_map": list(task_graph.atom_map),
        "reactant_bonds": [[t.i, t.j, t.order] for t in task_graph.reactant_bonds],
        "product_bonds": [[t.i, t.j, t.order] for t in task_graph.product_bonds],
        "edits": [
            [e.kind, e.i, e.j, e.reactant_order, e.product_order]
            for e in task_graph.edits
        ],
        "source": task_graph.source,
        "provenance": dict(task_graph.provenance),
        "graph_hash": task_graph.content_hash(),
    }


def _coord_doc(coord_spec) -> dict[str, JSONValue]:
    return {
        "spec_version": coord_spec.spec_version,
        "coords": [
            {
                "kind": c.kind,
                "atoms": list(c.atoms),
                "units": c.units,
                "role": c.role,
                "scale_a": c.scale_a,
                "ref_q": c.ref_q,
                "label": c.label,
            }
            for c in coord_spec.coords
        ],
        "spec_hash": coord_spec.content_hash(),
    }


def write_shards(
    records: Sequence[ReactionRecord],
    output_dir: Path,
    *,
    shard_size: int = SHARD_SIZE,
) -> list[dict[str, JSONValue]]:
    """Write state/label/provenance shard triples; return shard descriptors."""
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptors: list[dict[str, JSONValue]] = []
    for start in range(0, len(records), shard_size):
        chunk = records[start : start + shard_size]
        shard_index = start // shard_size
        states: dict[str, np.ndarray] = {}
        labels: dict[str, np.ndarray] = {}
        provenance: dict[str, JSONValue] = {"reactions": {}}
        n_samples = 0
        for record in chunk:
            rid = record.reaction_id
            if record.exclusion is not None or record.progress is None:
                provenance["reactions"][rid] = {
                    "exclusion": record.exclusion,
                    **record.provenance,
                }
                continue
            states[f"coords__{rid}"] = record.coords
            states[f"atomic_numbers__{rid}"] = record.atomic_numbers.astype(np.int16)
            labels[f"s__{rid}"] = record.progress.signed_progress.astype(np.float32)
            labels[f"d__{rid}"] = record.progress.path_proximity.astype(np.float32)
            labels[f"s_mask__{rid}"] = record.progress.progress_mask
            labels[f"side__{rid}"] = record.progress.side
            labels[f"d_arc__{rid}"] = record.progress.d_arc.astype(np.float32)
            labels[f"direction__{rid}"] = record.direction.direction.astype(np.float32)
            labels[f"activity__{rid}"] = record.direction.activity.astype(np.float32)
            labels[f"dir_mask__{rid}"] = record.direction.direction_mask
            labels[f"act_mask__{rid}"] = record.direction.activity_mask
            provenance["reactions"][rid] = {
                "exclusion": None,
                "task_graph": record.task_graph_doc,
                "coordinate_spec": record.coordinate_spec_doc,
                **record.provenance,
            }
            n_samples += int(record.progress.progress_mask.sum())
        shard_tag = f"{shard_index:04d}"
        states_path = output_dir / f"states_{shard_tag}.npz"
        labels_path = output_dir / f"labels_{shard_tag}.npz"
        prov_path = output_dir / f"provenance_{shard_tag}.json"
        np.savez_compressed(states_path, **states)
        np.savez_compressed(labels_path, **labels)
        write_json(prov_path, provenance)
        descriptors.append(
            {
                "shard": shard_tag,
                "n_reactions": len(chunk),
                "n_samples": n_samples,
                "states_sha256": sha256_file(states_path),
                "labels_sha256": sha256_file(labels_path),
                "provenance_sha256": sha256_file(prov_path),
            }
        )
    return descriptors


def build_dataset_manifest(
    *,
    dataset_dir: Path,
    shards: Sequence[dict[str, JSONValue]],
    source_hashes: Mapping[str, str],
    parent_split_sha256: str,
    label_spec_sha256: str,
    n_reactions: int,
    n_excluded: int,
    task_coverage: dict[str, int],
) -> dict[str, JSONValue]:
    manifest: dict[str, JSONValue] = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_sha256": dict(source_hashes),
        "source_ref": "B3LYPD3_TZVP_IRC (quarantined, audited bridge reads)",
        "source_method": "B3LYP-D3/TZVP",
        "electronic_state": "S0",
        "parent_split_manifest_sha256": parent_split_sha256,
        "label_spec_sha256": label_spec_sha256,
        "shards": list(shards),
        "n_reactions": n_reactions,
        "n_excluded": n_excluded,
        "n_samples": int(sum(int(s["n_samples"]) for s in shards)),
        "task_coverage": task_coverage,
        "units": {"coordinates": "angstrom", "arc": "angstrom_sqrt_u"},
    }
    validate_dataset_manifest(manifest)
    return manifest


def build_labels_manifest(
    *,
    dataset_manifest_sha256: str,
    label_spec_sha256: str,
    shards: Sequence[dict[str, JSONValue]],
    n_samples: int,
) -> dict[str, JSONValue]:
    manifest: dict[str, JSONValue] = {
        "schema_version": LABELS_SCHEMA_VERSION,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "label_spec_sha256": label_spec_sha256,
        "shards": [
            {"shard": s["shard"], "labels_sha256": s["labels_sha256"]} for s in shards
        ],
        "n_samples": n_samples,
        "join_key": "sample_id = reaction_id#f<frame_index>",
    }
    validate_labels_manifest(manifest)
    return manifest


def task_coverage_of(records: Sequence[ReactionRecord]) -> dict[str, int]:
    coverage = {
        "progress_frames": 0,
        "proximity_frames": 0,
        "direction_frames": 0,
        "activity_frames": 0,
        "ts_anchor_frames": 0,
    }
    for record in records:
        if record.exclusion is not None or record.progress is None:
            continue
        coverage["progress_frames"] += int(record.progress.progress_mask.sum())
        coverage["proximity_frames"] += int(record.progress.progress_mask.sum())
        coverage["direction_frames"] += int(record.direction.direction_mask.sum())
        coverage["activity_frames"] += int(record.direction.activity_mask.sum())
        coverage["ts_anchor_frames"] += int((record.progress.side == 0).sum())
    return coverage


__all__ = [
    "DATASET_VERSION",
    "DatasetBuildError",
    "LABEL_SPEC_VERSION",
    "ORIENTATION_CONFIRMED",
    "ORIENTATION_FLAGGED",
    "ReactionRecord",
    "build_dataset_manifest",
    "build_label_spec",
    "build_labels_manifest",
    "build_reaction_record",
    "deterministic_selection",
    "task_coverage_of",
    "write_shards",
]
