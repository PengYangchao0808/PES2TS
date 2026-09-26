"""P1 orchestration: the truth-assisted TS/IRC mapping annotation build.

This is the **only** module under ``pes2ts_core/g1/`` that reads the
quarantined ground truth, and it is the only one the static guard
allowlists for that purpose (plan section 7: the exception is scoped to a
named P1 annotation module).  Every truth byte flows through the audited
accessors :func:`pes2ts_core.g0.truth.truth_reader.load_ts_table`,
``load_irc_index``, ``load_irc_frames``, and ``iter_irc_trajectories`` --
this module never opens the relocated HDF5 archives itself and never builds
their paths.  Every successful read lands in ``truth_access_log.jsonl``
with this module recorded as the caller.

The per-reaction pipeline follows the plan's P1.0-P1.3 order: join gate,
TS-row correspondence, side candidate resolution, IRC layout/orientation,
and event validation.  Reactions that fail any gate keep a typed status and
their evidence; only ``resolved_unique`` / ``resolved_symmetry_collapsed``
reactions are marked G2-eligible in the summary.  Persisted documents carry
mappings, event verdicts, and audit digests -- never TS/IRC coordinates or
energy arrays.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
from pes2ts_core.g0.fetch import dataset_version
from pes2ts_core.g0.inventory import INVENTORY_PARQUET_FILENAME
from pes2ts_core.g0.rejections import (
    LEDGER_FILENAME, Rejection, RejectionCode, RejectionLedger,
)
from pes2ts_core.g0.rp_checks import ELEMENT_SYMBOLS
from pes2ts_core.g0.truth import truth_reader
from pes2ts_core.g0.truth.truth_reader import TRUTH_MANIFEST_FILENAME
from pes2ts_core.g1.build import (
    BUILD_HINT as G1_BUILD_HINT,
    SUMMARY_FILENAME as G1_SUMMARY_FILENAME,
    reaction_change_path as g1_document_path,
    shard_name,
)
from pes2ts_core.g1.graph_payload import graph_payload
from pes2ts_core.g1.irc_bond_evidence import bond_pattern_orientation, validate_events
from pes2ts_core.g1.truth_alignment import (
    FLAG_TS_FRAME_UNMATCHED,
    analyze_irc_layout,
    endpoint_verdict,
    solve_side_assignment,
    solve_ts_correspondence,
)
from pes2ts_core.g1.truth_join import build_join_audit
from pes2ts_core.g1.truth_schema import (
    DEFAULT_BOND_DISTANCE_TOLERANCE,
    DEFAULT_ENDPOINT_PASS,
    DEFAULT_ENDPOINT_WEAK,
    DEFAULT_MAX_COMBINATIONS,
    DEFAULT_PERMUTATION_BUDGET,
    DEFAULT_RMSD_CLASS_TOLERANCE,
    DEFAULT_SAMPLE_SEED,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_SHARD_SIZE,
    DEFAULT_SPLICE_FACTOR,
    DEFAULT_TS_FRAME_TOLERANCE,
    ENDPOINT_FAIL,
    G1_TRUTH_DIRNAME,
    G2_ELIGIBLE_STATUSES,
    JOIN_AUDIT_FILENAME,
    LAYOUT_ALGORITHM,
    MAPPING_ALGORITHM,
    ORIENTATION_P_FIRST,
    ORIENTATION_R_FIRST,
    ORIENTATION_UNRESOLVED,
    P1_COVERAGE_FILENAME,
    P1_DOCUMENT_SCHEMA_VERSION,
    P1_MANIFEST_FILENAME,
    P1_MANIFEST_SCHEMA_VERSION,
    P1_MAPPING_DIRNAME,
    P1_STATUSES,
    P1_SUMMARY_FILENAME,
    STATUS_ATOM_ELEMENT_MISMATCH,
    STATUS_ENDPOINT_MISMATCH,
    STATUS_MISSING_TRUTH_JOIN,
    STATUS_RESOLVED_SYMMETRY,
    STATUS_RESOLVED_UNIQUE,
    STATUS_SOURCE_MISMATCH,
    STATUS_UNRESOLVED_CENTER,
    STATUS_UNRESOLVED_TRUNCATED,
)
from pes2ts_core.utils.hashing import (
    JSONValue, sha256_bytes, sha256_file, stable_json_dumps,
)
from pes2ts_core.utils.jsonio import read_json, write_json
from pes2ts_core.utils.parquet_io import read_parquet, write_parquet

logger = logging.getLogger(__name__)

#: Selection size at/below which IRC frames are fetched per reaction instead
#: of streaming the whole archive once.
DEFAULT_PER_REACTION_THRESHOLD: Final[int] = 512
#: Rejection-ledger stage stamped by the P1 build.
P1_STAGE: Final[str] = "g1_p1_truth"
#: Rejection codes reused for P1 exclusions (typed, never silent).
P1_REJECTION_CODES: Final[dict[str, RejectionCode]] = {
    STATUS_UNRESOLVED_CENTER: RejectionCode.G1_MAP_ERROR,
    STATUS_UNRESOLVED_TRUNCATED: RejectionCode.G1_MAP_ERROR,
    STATUS_ENDPOINT_MISMATCH: RejectionCode.G1_INDEX_MISMATCH,
    STATUS_ATOM_ELEMENT_MISMATCH: RejectionCode.ELEMENT_MISMATCH,
    STATUS_MISSING_TRUTH_JOIN: RejectionCode.MISSING_IN_H5,
    STATUS_SOURCE_MISMATCH: RejectionCode.G1_COMPONENT_MISMATCH,
}
#: New rejection codes this stage needs (added to the shared enum lazily is
#: forbidden; the enum is extended in rejections.py directly -- see imports).

#: Summary Parquet columns of the P1 build.
P1_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "reaction_id", "status", "g2_eligible", "ts_frame_index", "layout",
    "orientation", "endpoint_match", "n_combinations_r", "n_combinations_p",
    "n_equivalence_classes_r", "n_equivalence_classes_p", "symmetry_collapsed",
    "selected_rmsd_r", "selected_rmsd_p", "n_formed", "n_broken",
    "n_order_changed", "n_h_migration", "n_event_support", "n_event_weak",
    "n_event_mismatch", "n_quality_flags",
)

_Z_TO_SYMBOL: Final[dict[int, str]] = {
    index + 1: symbol for index, symbol in enumerate(ELEMENT_SYMBOLS)
}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def p1_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the effective ``g1_truth`` settings with documented defaults."""
    raw = config.get("g1_truth")
    settings: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
    return {
        "shard_size": int(settings.get("shard_size", DEFAULT_SHARD_SIZE)),
        "max_combinations": int(settings.get("max_combinations", DEFAULT_MAX_COMBINATIONS)),
        "rmsd_class_tolerance": float(settings.get("rmsd_class_tolerance", DEFAULT_RMSD_CLASS_TOLERANCE)),
        "endpoint_rmsd_pass": float(settings.get("endpoint_rmsd_pass", DEFAULT_ENDPOINT_PASS)),
        "endpoint_rmsd_weak": float(settings.get("endpoint_rmsd_weak", DEFAULT_ENDPOINT_WEAK)),
        "ts_frame_tolerance": float(settings.get("ts_frame_tolerance", DEFAULT_TS_FRAME_TOLERANCE)),
        "splice_factor": float(settings.get("splice_factor", DEFAULT_SPLICE_FACTOR)),
        "bond_distance_tolerance": float(settings.get("bond_distance_tolerance", DEFAULT_BOND_DISTANCE_TOLERANCE)),
        "permutation_budget": int(settings.get("permutation_budget", DEFAULT_PERMUTATION_BUDGET)),
        "per_reaction_lookup_threshold": int(settings.get("per_reaction_lookup_threshold", DEFAULT_PER_REACTION_THRESHOLD)),
        "sample_size": int(settings.get("sample_size", DEFAULT_SAMPLE_SIZE)),
        "sample_seed": int(settings.get("sample_seed", DEFAULT_SAMPLE_SEED)),
    }


def p1_mapping_dir(interim_dir: str | Path) -> Path:
    """Return the sharded P1 document root under ``interim/g1_truth``."""
    return Path(interim_dir) / G1_TRUTH_DIRNAME / P1_MAPPING_DIRNAME


def p1_document_path(interim_dir: str | Path, reaction_id: str, shard_size: int) -> Path:
    """Return the sharded P1 document path of *reaction_id*."""
    return p1_mapping_dir(interim_dir) / shard_name(reaction_id, shard_size) / f"{reaction_id}.json"


def _truth_digests(manifests_dir: Path) -> dict[str, str]:
    """Return the truth-manifest digest plus its recorded artifact digests."""
    manifest_path = manifests_dir / TRUTH_MANIFEST_FILENAME
    document = read_json(manifest_path) if manifest_path.is_file() else {}
    digests: dict[str, str] = {"truth_manifest": sha256_file(manifest_path)}
    if isinstance(document, dict):
        ts_entry = document.get("ts_parquet")
        if isinstance(ts_entry, dict) and "sha256" in ts_entry:
            digests["ts_parquet"] = str(ts_entry["sha256"])
        irc_entry = document.get("irc_index")
        if isinstance(irc_entry, dict) and "sha256" in irc_entry:
            digests["irc_index"] = str(irc_entry["sha256"])
    return digests


def _side_geometry_inputs(
    inventory_row: Mapping[str, Any], prefix: str
) -> tuple[dict[str, list[list[float]]], dict[str, str]]:
    """Return ``(tag -> coordinates, tag -> smiles)`` of one side."""
    geometries: dict[str, list[list[float]]] = {}
    smiles: dict[str, str] = {}
    for component in inventory_row["components"]:
        tag = str(component["tag"])
        if tag.startswith(prefix):
            geometries[tag] = [
                [float(value) for value in point] for point in component["coordinates"]
            ]
            smiles[tag] = str(component["smiles"])
    return geometries, smiles


def _first_bijection_reference(
    g1_document: Mapping[str, Any],
    r_geometries: Mapping[str, list[list[float]]],
    maps: Sequence[int],
) -> np.ndarray | None:
    """Return the R-side geometry in map order under the first G1 bijection.

    The TS-row permutation fallback needs *some* deterministic reference
    geometry in map order; the first stored G1 isomorphism (the one that
    filled ``rows``) supplies it.  ``None`` when the blocks do not cover
    every map (the side solve will surface the real failure).
    """
    blocks = sorted(g1_document["reactants"], key=lambda block: int(block["index_base"]))
    rows = {
        int(row["map"]): (str(block["tag"]), int(row["local_index"]))
        for block in blocks
        for row in block["rows"]
    }
    if sorted(rows) != list(maps):
        return None
    try:
        return np.vstack([
            np.asarray(r_geometries[rows[m][0]], dtype=np.float64)[rows[m][1]]
            for m in maps
        ])
    except KeyError:
        return None


def _reject_document(
    base: dict[str, JSONValue], status: str, detail: str
) -> dict[str, JSONValue]:
    """Finalize *base* with a typed non-eligible status and its detail."""
    base["status"] = status
    base["status_detail"] = detail
    base["validation"] = {"status": "unresolved" if "unresolved" in status else "rejected", "reasons": [detail]}
    return base


@dataclass(frozen=True, slots=True)
class P1Result:
    """Outcome of one P1 resolve run."""

    n_total: int
    n_eligible: int
    by_status: dict[str, int]
    summary_path: Path
    manifest_path: Path
    join_audit_path: Path
    documents_dir: Path


def resolve_reaction(
    inventory_row: Mapping[str, Any],
    g1_document: Mapping[str, Any],
    ts_row: Mapping[str, Any] | None,
    irc_frames: Mapping[str, Any] | None,
    settings: Mapping[str, Any],
    sources: Mapping[str, JSONValue],
) -> dict[str, JSONValue]:
    """Resolve one reaction's P1 mapping document.

    Typed failure order: rejected G1 source first, then truth joins, atom/
    element conservation, candidate enumeration completeness, TS-row
    ambiguity, and endpoint correspondence.  Eligible documents carry the
    full mapping table, the graph payload, and the IRC evidence summary.
    """
    reaction_id = str(inventory_row["reaction_id"])
    base: dict[str, JSONValue] = {
        "schema_version": P1_DOCUMENT_SCHEMA_VERSION,
        "reaction_id": reaction_id,
        "generated_at": _now(),
        "truth_assisted": True,
        "sources": dict(sources),
        "status": STATUS_RESOLVED_UNIQUE,
        "status_detail": "",
        "mapping": {"algorithm": MAPPING_ALGORITHM, "map_to_atoms": []},
        "bond_events": {"formed": [], "broken": [], "order_changed": [], "hydrogen_migration": []},
        "irc_validation": None,
        "reaction_center": {"core_maps": [], "shell1_maps": []},
        "graph": None,
        "validation": {"status": "pass", "reasons": []},
    }
    if str(g1_document.get("validation", {}).get("status")) != "valid":
        failure = g1_document.get("validation", {}).get("failure_code") or "g1_rejected"
        return _reject_document(base, STATUS_SOURCE_MISMATCH, f"g1_build_{failure}")
    if ts_row is None or irc_frames is None:
        missing = []
        if ts_row is None:
            missing.append("ts")
        if irc_frames is None:
            missing.append("irc")
        return _reject_document(base, STATUS_MISSING_TRUTH_JOIN, "missing:" + "+".join(missing))
    payload = graph_payload(str(inventory_row["reaction_smiles"]))
    if payload is None:
        return _reject_document(base, STATUS_SOURCE_MISMATCH, "reaction_smiles_unparseable")
    base["graph"] = payload
    elements_by_map = {int(k): str(v) for k, v in payload["elements"].items()}
    maps = sorted(elements_by_map)
    if maps != list(range(1, len(maps) + 1)):
        return _reject_document(base, STATUS_ATOM_ELEMENT_MISMATCH, "map_space_not_contiguous")
    r_geometries, r_smiles = _side_geometry_inputs(inventory_row, "R")
    p_geometries, p_smiles = _side_geometry_inputs(inventory_row, "P")
    ts_coordinates = np.asarray(ts_row["coordinates"], dtype=np.float64)
    ts_atomic_numbers = [int(z) for z in ts_row["atomic_numbers"]]
    elements_in_map_order = [elements_by_map[m] for m in maps]
    reference_for_fallback = _first_bijection_reference(
        g1_document, r_geometries, maps
    )
    correspondence = solve_ts_correspondence(
        elements_in_map_order, ts_atomic_numbers, _Z_TO_SYMBOL,
        reference_by_map=reference_for_fallback,
        ts_coordinates=ts_coordinates,
        permutation_budget=int(settings["permutation_budget"]),
    )
    if correspondence is None:
        return _reject_document(base, STATUS_ATOM_ELEMENT_MISMATCH, "no_element_compatible_ts_rows")
    ts_by_map = ts_coordinates[list(correspondence.row_of_map)]
    r_solution = solve_side_assignment(
        g1_document["reactants"], r_geometries, r_smiles, ts_by_map,
        max_combinations=int(settings["max_combinations"]),
        rmsd_class_tolerance=float(settings["rmsd_class_tolerance"]),
    )
    p_solution = solve_side_assignment(
        g1_document["products"], p_geometries, p_smiles, ts_by_map,
        max_combinations=int(settings["max_combinations"]),
        rmsd_class_tolerance=float(settings["rmsd_class_tolerance"]),
    )
    if r_solution is None or p_solution is None:
        return _reject_document(base, STATUS_ENDPOINT_MISMATCH, "no_usable_side_bijection")
    if r_solution.truncated or p_solution.truncated:
        return _reject_document(base, STATUS_UNRESOLVED_TRUNCATED, "candidate_enumeration_capped")
    if correspondence.algorithm == "greedy_distance" and correspondence.n_ambiguous_alternatives > 0:
        return _reject_document(
            base, STATUS_UNRESOLVED_CENTER,
            f"greedy_ts_permutation_ambiguous:{correspondence.n_ambiguous_alternatives}",
        )
    r_by_map = np.vstack(
        [np.asarray(r_geometries[geometry_tag], dtype=np.float64)[local_row]
         for _block, geometry_tag, local_row in r_solution.rows_by_map]
    )
    p_by_map = np.vstack(
        [np.asarray(p_geometries[geometry_tag], dtype=np.float64)[local_row]
         for _block, geometry_tag, local_row in p_solution.rows_by_map]
    )
    frames = irc_frames
    frame_coordinates = np.asarray(frames["coordinates"], dtype=np.float64)
    layout = analyze_irc_layout(
        frame_coordinates, ts_coordinates,
        splice_factor=float(settings["splice_factor"]),
        ts_tolerance=float(settings["ts_frame_tolerance"]),
    )
    if FLAG_TS_FRAME_UNMATCHED in layout.quality_flags:
        # The exact TS-frame equality is the trajectory's identity proof; a
        # trajectory containing no frame that matches the quarantined TS
        # cannot belong to this reaction under any candidate assignment.
        base["irc_validation"] = {
            "layout_algorithm": LAYOUT_ALGORITHM,
            "n_frames": int(frames["n_frames"]),
            "n_atoms": int(frames["n_atoms"]),
            "ts_frame_index": layout.ts_frame_index,
            "ts_frame_rmsd": layout.ts_frame_rmsd,
            "layout": layout.layout,
            "splice_index": layout.splice_index,
            "orientation": ORIENTATION_UNRESOLVED,
            "orientation_basis": "ts_frame_unmatched",
            "endpoint_match": ENDPOINT_FAIL,
            "endpoint_rmsd": {},
            "event_support": [],
            "synchrony": "unresolved",
            "max_progress_delta": None,
            "n_support": 0,
            "n_weak": 0,
            "n_mismatch": 0,
            "quality_flags": list(layout.quality_flags),
        }
        return _reject_document(
            base, STATUS_ENDPOINT_MISMATCH,
            f"ts_frame_rmsd={layout.ts_frame_rmsd:.3f}exceeds_tolerance",
        )
    # Graph structure decides the branch-to-side orientation; whole-molecule
    # RMSD at IRC termini is recorded as evidence but demonstrably mislabels
    # orientations when conformer noise dominates (see bond_pattern_orientation).
    bond_orientation, bond_scores = bond_pattern_orientation(
        layout, frame_coordinates,
        payload["r_bonds"], payload["p_bonds"], elements_by_map,
        list(correspondence.row_of_map),
        tolerance=float(settings["bond_distance_tolerance"]),
    )
    verdict = endpoint_verdict(
        layout, frame_coordinates, r_by_map, p_by_map,
        pass_rmsd=float(settings["endpoint_rmsd_pass"]),
        weak_rmsd=float(settings["endpoint_rmsd_weak"]),
        orientation_hint=bond_orientation,
    )
    orientation_basis = (
        "bond_pattern"
        if bond_orientation in (ORIENTATION_R_FIRST, ORIENTATION_P_FIRST)
        else "rmsd_fallback"
    )
    events: dict[str, JSONValue] = {
        "formed": list(g1_document["bond_changes"]["formed"]),
        "broken": list(g1_document["bond_changes"]["broken"]),
        "order_changed": list(g1_document["bond_changes"]["order_changed"]),
        "hydrogen_migration": list(g1_document["bond_changes"]["hydrogen_migration"]),
    }
    base["bond_events"] = events
    validation = validate_events(
        events, layout, frame_coordinates, list(correspondence.row_of_map),
        elements_by_map, verdict.orientation,
        tolerance=float(settings["bond_distance_tolerance"]),
    )
    base["irc_validation"] = {
        "layout_algorithm": LAYOUT_ALGORITHM,
        "n_frames": int(frames["n_frames"]),
        "n_atoms": int(frames["n_atoms"]),
        "ts_frame_index": layout.ts_frame_index,
        "ts_frame_rmsd": layout.ts_frame_rmsd,
        "layout": layout.layout,
        "splice_index": layout.splice_index,
        "orientation": verdict.orientation,
        "orientation_basis": orientation_basis,
        "orientation_bond_scores": {
            branch: {
                side: [satisfied, total]
                for side, (satisfied, total) in scores.items()
            }
            for branch, scores in bond_scores.items()
        },
        "endpoint_match": verdict.endpoint_match,
        "endpoint_rmsd": {
            "branch_a_to_r": verdict.rmsd_a_to_r,
            "branch_a_to_p": verdict.rmsd_a_to_p,
            "branch_b_to_r": verdict.rmsd_b_to_r,
            "branch_b_to_p": verdict.rmsd_b_to_p,
        },
        "event_support": [
            {
                "kind": item.kind, "maps": list(item.maps), "branch": item.branch,
                "d_r_end": item.d_r_end, "d_ts": item.d_ts, "d_p_end": item.d_p_end,
                "d_min": item.d_min, "d_max": item.d_max, "trend": item.trend,
                "verdict": item.verdict,
                "progress_fraction": item.progress_fraction,
                "detail": item.detail,
            }
            for item in validation.event_support
        ],
        "synchrony": validation.synchrony,
        "max_progress_delta": validation.max_progress_delta,
        "n_support": validation.n_support,
        "n_weak": validation.n_weak,
        "n_mismatch": validation.n_mismatch,
        "quality_flags": list(
            dict.fromkeys([*layout.quality_flags, *verdict.quality_flags, *validation.quality_flags])
        ),
    }
    base["reaction_center"] = {
        "core_maps": list(g1_document["reaction_center"]["core"]),
        "shell1_maps": [
            map_number for map_number in g1_document["reaction_center"]["with_shell"]
            if map_number not in set(g1_document["reaction_center"]["core"])
        ],
    }
    if verdict.endpoint_match == ENDPOINT_FAIL:
        return _reject_document(
            base, STATUS_ENDPOINT_MISMATCH,
            f"endpoint_rmsd_exceeds_weak:r={min(verdict.rmsd_a_to_r, verdict.rmsd_b_to_r or verdict.rmsd_a_to_r):.3f}",
        )
    collapsed = r_solution.selected_class_size > 1 or p_solution.selected_class_size > 1
    status = STATUS_RESOLVED_SYMMETRY if collapsed else STATUS_RESOLVED_UNIQUE
    base["status"] = status
    base["status_detail"] = (
        f"classes r={r_solution.n_equivalence_classes}/p={p_solution.n_equivalence_classes} "
        f"selected r={r_solution.selected_class_size}/p={p_solution.selected_class_size}"
    )
    map_to_atoms: list[JSONValue] = []
    for index, map_number in enumerate(maps):
        r_block, r_geometry, r_local = r_solution.rows_by_map[index]
        p_block, p_geometry, p_local = p_solution.rows_by_map[index]
        map_to_atoms.append({
            "map": map_number,
            "element": elements_by_map[map_number],
            "r_component": r_block, "r_geometry_tag": r_geometry, "r_local_index": r_local,
            "p_component": p_block, "p_geometry_tag": p_geometry, "p_local_index": p_local,
            "ts_irc_index": correspondence.row_of_map[index],
        })
    base["mapping"] = {
        "algorithm": MAPPING_ALGORITHM,
        "ts_row_algorithm": correspondence.algorithm,
        "map_to_atoms": map_to_atoms,
        "n_combinations_r": r_solution.n_combinations,
        "n_combinations_p": p_solution.n_combinations,
        "n_equivalence_classes_r": r_solution.n_equivalence_classes,
        "n_equivalence_classes_p": p_solution.n_equivalence_classes,
        "selected_class_size_r": r_solution.selected_class_size,
        "selected_class_size_p": p_solution.selected_class_size,
        "selected_rmsd_r": r_solution.selected_rmsd,
        "selected_rmsd_p": p_solution.selected_rmsd,
        "symmetry_collapsed": collapsed,
        "resolution_reason": "best kabsch class per side; lexicographic representative",
    }
    reasons: list[str] = []
    if verdict.orientation == ORIENTATION_UNRESOLVED:
        reasons.append("orientation_unresolved")
    reasons.extend(base["irc_validation"]["quality_flags"])
    if validation.n_mismatch:
        reasons.append(f"{validation.n_mismatch}_irc_event_mismatch")
    base["validation"] = {"status": "pass" if not reasons else "pass_with_flags", "reasons": reasons}
    return base


def _summary_row(document: Mapping[str, Any]) -> dict[str, JSONValue]:
    """Return the summary Parquet row of one P1 document."""
    irc = document.get("irc_validation")
    mapping = document.get("mapping")
    events = document.get("bond_events")
    status = str(document["status"])
    irc_mapping: Mapping[str, Any] = mapping if isinstance(mapping, Mapping) else {}
    irc_block: Mapping[str, Any] = irc if isinstance(irc, Mapping) else {}
    events_block: Mapping[str, Any] = events if isinstance(events, Mapping) else {}
    flags = irc_block.get("quality_flags") if isinstance(irc_block.get("quality_flags"), list) else []
    return {
        "reaction_id": document["reaction_id"],
        "status": status,
        "g2_eligible": status in G2_ELIGIBLE_STATUSES,
        "ts_frame_index": irc_block.get("ts_frame_index"),
        "layout": irc_block.get("layout"),
        "orientation": irc_block.get("orientation"),
        "endpoint_match": irc_block.get("endpoint_match"),
        "n_combinations_r": irc_mapping.get("n_combinations_r"),
        "n_combinations_p": irc_mapping.get("n_combinations_p"),
        "n_equivalence_classes_r": irc_mapping.get("n_equivalence_classes_r"),
        "n_equivalence_classes_p": irc_mapping.get("n_equivalence_classes_p"),
        "symmetry_collapsed": bool(irc_mapping.get("symmetry_collapsed")),
        "selected_rmsd_r": irc_mapping.get("selected_rmsd_r"),
        "selected_rmsd_p": irc_mapping.get("selected_rmsd_p"),
        "n_formed": len(events_block.get("formed") or []),
        "n_broken": len(events_block.get("broken") or []),
        "n_order_changed": len(events_block.get("order_changed") or []),
        "n_h_migration": len(events_block.get("hydrogen_migration") or []),
        "n_event_support": irc_block.get("n_support"),
        "n_event_weak": irc_block.get("n_weak"),
        "n_event_mismatch": irc_block.get("n_mismatch"),
        "n_quality_flags": len(flags),
    }


def _load_inputs(
    config: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
    """Load inventory rows, digests, and the G1 summary rows.

    The per-reaction G1 documents are **not** eagerly loaded: a full run
    touches ~200k JSON files and only the selected reactions' documents are
    needed, so :func:`_g1_document` reads them lazily one at a time.
    """
    interim_dir = Path(config["paths"]["interim"])
    inventory_path = interim_dir / INVENTORY_PARQUET_FILENAME
    if not inventory_path.is_file():
        raise FileNotFoundError(f"Missing inventory {inventory_path}; run `g0 inventory` first")
    g1_summary_path = interim_dir / G1_SUMMARY_FILENAME
    if not g1_summary_path.is_file():
        raise FileNotFoundError(f"Missing G1 summary {g1_summary_path}; {G1_BUILD_HINT}")
    inventory_rows = {
        str(row["reaction_id"]): row
        for row in read_parquet(inventory_path).to_pylist()
    }
    g1_rows = read_parquet(g1_summary_path).to_pylist()
    digests = {
        "inventory": sha256_file(inventory_path),
        "g1_summary": sha256_file(g1_summary_path),
    }
    return inventory_rows, digests, g1_rows


def _g1_document(
    interim_dir: Path, reaction_id: str, shard_size: int
) -> dict[str, Any] | None:
    """Return one reaction's G1 document (``None`` when never built)."""
    path = g1_document_path(interim_dir, reaction_id, shard_size)
    if not path.is_file():
        return None
    return read_json(path)


def run_join_audit(
    config: Mapping[str, Any], *, allow_truth: bool
) -> Path:
    """Write ``g1_p1_join_audit.json`` and return its path.

    Reads the TS table and the IRC index through the audited truth
    accessors (recording this module as the caller) and the plain interim
    artifacts directly.  ``allow_truth=False`` raises
    :class:`PermissionError` before any truth byte is touched.
    """
    manifests_dir = Path(config["paths"]["manifests"])
    interim_dir = Path(config["paths"]["interim"])
    inventory_rows, digests, g1_rows = _load_inputs(config)
    ts_table = truth_reader.load_ts_table(allow_truth, manifests_dir=manifests_dir)
    ts_rows = ts_table.to_pylist()
    irc_rows = truth_reader.load_irc_index(allow_truth, manifests_dir=manifests_dir)
    truth_digests = _truth_digests(manifests_dir)
    audit = build_join_audit(
        inventory_rows=list(inventory_rows.values()),
        ts_rows=ts_rows,
        irc_rows=irc_rows,
        g1_summary_rows=g1_rows,
        digests={**digests, **truth_digests},
        dataset_version=dataset_version(config),
    )
    audit["caller_module"] = __name__
    audit["generated_at"] = _now()
    path = manifests_dir / JOIN_AUDIT_FILENAME
    write_json(path, audit)
    logger.info(
        "P1 join audit: inventory=%d ts=%d irc=%d joined=%d -> %s",
        audit["n_inventory"], audit["n_ts"], audit["n_irc_index"],
        audit["n_joined_all"], path,
    )
    return path


def resolve_p1(
    config: Mapping[str, Any],
    *,
    allow_truth: bool,
    reaction_ids: Sequence[str] | None = None,
    limit: int | None = None,
) -> P1Result:
    """Build every P1 mapping document, summary, manifest, and coverage.

    ``reaction_ids=None`` processes every inventory row (a full run streams
    the IRC archive once); an explicit small selection fetches frames per
    reaction.  Rejected or unresolved reactions are persisted with typed
    statuses and append one ledger entry each.
    """
    manifests_dir = Path(config["paths"]["manifests"])
    interim_dir = Path(config["paths"]["interim"])
    settings = p1_settings(config)
    shard_size = int(settings["shard_size"])
    inventory_rows, digests, g1_rows = _load_inputs(config)
    ts_table = truth_reader.load_ts_table(allow_truth, manifests_dir=manifests_dir)
    ts_rows = {str(row["reaction_id"]): row for row in ts_table.to_pylist()}
    irc_index_rows = truth_reader.load_irc_index(allow_truth, manifests_dir=manifests_dir)
    truth_digests = _truth_digests(manifests_dir)
    sources: dict[str, JSONValue] = {
        "dataset_version": dataset_version(config),
        "inventory_digest": digests["inventory"],
        "g1_summary_digest": digests["g1_summary"],
        "truth_manifest_digest": truth_digests["truth_manifest"],
        "ts_parquet_digest": truth_digests.get("ts_parquet", ""),
        "irc_index_digest": truth_digests.get("irc_index", ""),
        "mapping_algorithm": MAPPING_ALGORITHM,
        "mapping_config_digest": sha256_bytes(
            stable_settings_text(settings).encode("utf-8")
        ),
    }
    selected = (
        sorted(inventory_rows) if reaction_ids is None else sorted(set(reaction_ids))
    )
    unknown = [reaction_id for reaction_id in selected if reaction_id not in inventory_rows]
    if unknown:
        raise ValueError(f"unknown reaction id(s): {unknown[:5]}")
    if limit is not None:
        selected = selected[:limit]
    selected_set = set(selected)
    irc_by_reaction: dict[str, dict[str, Any]] = {}
    if len(selected) <= int(settings["per_reaction_lookup_threshold"]):
        for reaction_id in selected:
            try:
                frames = truth_reader.load_irc_frames(
                    reaction_id, allow_truth, manifests_dir=manifests_dir, caller=__name__
                )
            except KeyError:
                continue
            irc_by_reaction[reaction_id] = frames
    else:
        logger.info("P1 streaming IRC archive once for %d reaction(s)", len(selected))
        for frames in truth_reader.iter_irc_trajectories(
            allow_truth, manifests_dir=manifests_dir, caller=__name__
        ):
            if frames.reaction_id in selected_set:
                irc_by_reaction[frames.reaction_id] = {
                    "reaction_id": frames.reaction_id,
                    "n_atoms": frames.n_atoms,
                    "n_frames": frames.n_frames,
                    "has_forces": frames.has_forces,
                    "coordinates": frames.coordinates,
                    "EHG": frames.ehg,
                }
            if len(irc_by_reaction) == len(selected_set):
                break
    audit = build_join_audit(
        inventory_rows=list(inventory_rows.values()),
        ts_rows=list(ts_rows.values()),
        irc_rows=irc_index_rows,
        g1_summary_rows=g1_rows,
        digests={**digests, **truth_digests},
        dataset_version=dataset_version(config),
    )
    join_audit_path = manifests_dir / JOIN_AUDIT_FILENAME
    write_json(join_audit_path, audit)
    documents: list[dict[str, JSONValue]] = []
    document_paths: dict[str, Path] = {}
    for reaction_id in selected:
        g1_document = _g1_document(interim_dir, reaction_id, shard_size)
        if g1_document is None:
            g1_document = {
                "validation": {"status": "rejected", "failure_code": "G1_UNBUILT"},
                "bond_changes": {"formed": [], "broken": [], "order_changed": [], "hydrogen_migration": []},
                "reaction_center": {"core": [], "with_shell": []},
                "reactants": [], "products": [],
            }
        document = resolve_reaction(
            inventory_rows[reaction_id],
            g1_document,
            ts_rows.get(reaction_id),
            irc_by_reaction.get(reaction_id),
            settings,
            sources,
        )
        path = p1_document_path(interim_dir, reaction_id, shard_size)
        write_json(path, document)
        documents.append(document)
        document_paths[reaction_id] = path
    rows = [_summary_row(document) for document in documents]
    summary_path = interim_dir / P1_SUMMARY_FILENAME
    write_parquet(summary_path, {name: [row[name] for row in rows] for name in P1_SUMMARY_COLUMNS})
    status_histogram = Counter(str(row["status"]) for row in rows)
    n_eligible = sum(1 for row in rows if bool(row["g2_eligible"]))
    manifest: dict[str, JSONValue] = {
        "schema_version": P1_MANIFEST_SCHEMA_VERSION,
        "dataset_version": dataset_version(config),
        "n_total": len(rows),
        "n_eligible": n_eligible,
        "by_status": {status: status_histogram[status] for status in P1_STATUSES if status_histogram.get(status)},
        "n_files": len(rows),
        "summary_sha256": sha256_file(summary_path),
        "config": {key: value for key, value in sorted(settings.items())},
        "sources": dict(sources),
        "generated_at": _now(),
    }
    manifest_path = manifests_dir / P1_MANIFEST_FILENAME
    write_json(manifest_path, manifest)
    coverage = build_p1_coverage(rows, settings)
    write_json(manifests_dir / P1_COVERAGE_FILENAME, coverage)
    ledger = RejectionLedger.load(manifests_dir)
    existing = _ledger_signatures(manifests_dir)
    for document in documents:
        status = str(document["status"])
        if status in G2_ELIGIBLE_STATUSES:
            continue
        code = P1_REJECTION_CODES.get(status)
        if code is None:
            continue
        detail = f"{status}:{document['status_detail']}"
        reaction_id = str(document["reaction_id"])
        if (reaction_id, P1_STAGE, code.value, detail) in existing:
            continue
        ledger.add(Rejection(
            reaction_id=reaction_id, stage=P1_STAGE, code=code, detail=detail,
            source_pointer=str(document_paths[reaction_id]),
        ))
    ledger.write()
    logger.info(
        "P1 resolve: %d reaction(s), %d eligible, statuses=%s -> %s",
        len(rows), n_eligible, dict(manifest["by_status"]), summary_path,
    )
    return P1Result(
        n_total=len(rows), n_eligible=n_eligible,
        by_status={status: status_histogram[status] for status in P1_STATUSES if status_histogram.get(status)},
        summary_path=summary_path, manifest_path=manifest_path,
        join_audit_path=join_audit_path, documents_dir=p1_mapping_dir(interim_dir),
    )


def stable_settings_text(settings: Mapping[str, Any]) -> str:
    """Return the canonical JSON text of the effective P1 settings."""
    return stable_json_dumps(dict(sorted(settings.items())))


def build_p1_coverage(
    rows: Sequence[Mapping[str, Any]], settings: Mapping[str, Any]
) -> dict[str, JSONValue]:
    """Return the P1 coverage report with per-status manual-check samples."""
    seed = int(settings.get("sample_seed", DEFAULT_SAMPLE_SEED))
    sample_size = int(settings.get("sample_size", DEFAULT_SAMPLE_SIZE))
    by_status: dict[str, JSONValue] = {}
    for status in P1_STATUSES:
        members = sorted(
            str(row["reaction_id"]) for row in rows if str(row["status"]) == status
        )
        entry: dict[str, JSONValue] = {"n": len(members)}
        ranked = sorted(
            members,
            key=lambda reaction_id: (
                sha256_bytes(f"{seed}:{reaction_id}".encode("utf-8")), reaction_id,
            ),
        )
        entry["manual_sample"] = ranked[:sample_size]
        by_status[status] = entry
    return {
        "schema_version": P1_MANIFEST_SCHEMA_VERSION,
        "n_total": len(rows),
        "n_eligible": sum(1 for row in rows if bool(row["g2_eligible"])),
        "eligible_fraction": (
            round(sum(1 for row in rows if bool(row["g2_eligible"])) / len(rows), 6)
            if rows else 0.0
        ),
        "by_status": by_status,
        "endpoint_match_histogram": {
            str(verdict): sum(1 for row in rows if str(row["endpoint_match"]) == verdict)
            for verdict in ("pass", "weak", "fail", "None")
            if any(str(row["endpoint_match"]) == verdict for row in rows)
        },
        "orientation_histogram": {
            str(value): sum(1 for row in rows if str(row["orientation"]) == value)
            for value in sorted({str(row["orientation"]) for row in rows})
        },
        "layout_histogram": {
            str(value): sum(1 for row in rows if str(row["layout"]) == value)
            for value in sorted({str(row["layout"]) for row in rows})
        },
        "seed": seed,
        "sample_size": sample_size,
        "generated_at": _now(),
    }


def _ledger_signatures(manifests_dir: Path) -> set[tuple[str, str, str, str]]:
    """Return the persisted ``(reaction_id, stage, code, detail)`` signatures."""
    import json

    path = manifests_dir / LEDGER_FILENAME
    if not path.is_file():
        return set()
    signatures: set[tuple[str, str, str, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record: object = json.loads(line)
        if isinstance(record, dict):
            signatures.add((
                str(record.get("reaction_id")), str(record.get("stage")),
                str(record.get("code")), str(record.get("detail")),
            ))
    return signatures


__all__ = [
    "DEFAULT_PER_REACTION_THRESHOLD",
    "P1Result",
    "P1_STAGE",
    "P1_SUMMARY_COLUMNS",
    "build_p1_coverage",
    "p1_document_path",
    "p1_mapping_dir",
    "p1_settings",
    "resolve_p1",
    "resolve_reaction",
    "run_join_audit",
    "stable_settings_text",
]
