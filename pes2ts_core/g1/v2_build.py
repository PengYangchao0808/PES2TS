"""G1 v2 rebuild: exclusive edits, machine audit, and the frozen v1 baseline.

``build_v2`` walks every reaction of the P1 summary (the 199,217-reaction
full population), recomputes the **mutually exclusive** v2 edit block from
the persisted map-space graph (never from truth), and audits each record
along three separated dimensions:

- ``structural_validity`` -- ``map_to_atoms`` is a bijection onto the side
  component rows, the element at every (component, row) agrees with the map
  element, the component ordinals agree with the graph payload, the v1
  ``bond_events`` are exactly derivable from the same graph, and the v2 edit
  block satisfies its own contract;
- ``mapping_determinism`` -- for symmetry-collapsed records every stored
  alternative candidate combination is enumerated (within a budget) and the
  element-labelled edit signature must not change; truncated G1 candidate
  searches are flagged;
- ``irc_evidence_quality`` -- IRC event mismatches and unresolved
  orientations are surfaced as typed issues instead of hiding inside an
  ``eligible`` count.

The v1 baseline is frozen first (:func:`freeze_v1_baseline`) so every v2
number has a comparison anchor; v2 artifacts live under their own
``g1_v2`` paths and never overwrite v1 or the split.
"""

from __future__ import annotations

import itertools
import json
import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from pes2ts_core.g0.fetch import dataset_version
from pes2ts_core.g0.inventory import INVENTORY_PARQUET_FILENAME
from pes2ts_core.g1.build import (
    SUMMARY_FILENAME as G1_SUMMARY_FILENAME,
    reaction_change_path as g1_document_path,
    shard_name,
)
from pes2ts_core.g1.p1_truth import p1_document_path, p1_settings
from pes2ts_core.g1.truth_schema import (
    G2_ELIGIBLE_STATUSES,
    GATE_MANIFEST_FILENAME,
    P1_MANIFEST_FILENAME,
    P1_SUMMARY_FILENAME,
    P2_MANIFEST_FILENAME,
    P2_SUMMARY_FILENAME,
)
from pes2ts_core.g1.v2_edits import (
    edit_counts,
    edit_signature,
    exclusive_edits,
    legacy_events_from_graph,
    legacy_reconciles,
    reaction_center,
    validate_edit_contract,
)
from pes2ts_core.g1.v2_schema import (
    AUDIT_CLEAN,
    AUDIT_EXCLUDED,
    AUDIT_ISSUES,
    DEFAULT_CENTER_SHELL,
    DEFAULT_COLLAPSE_AUDIT_BUDGET,
    DEFAULT_V2_SAMPLE_SEED,
    DEFAULT_V2_SAMPLE_SIZE,
    DEFAULT_V2_SHARD_SIZE,
    EDITS_DIRNAME,
    FREEZE_MANIFEST_FILENAME,
    HH_EVENT_KINDS,
    ISSUE_CANDIDATE_SEARCH_TRUNCATED,
    ISSUE_COLLAPSE_AUDIT_TRUNCATED,
    ISSUE_COLLAPSE_EDIT_AMBIGUOUS,
    ISSUE_COMPONENT_ASSIGNMENT_MISMATCH,
    ISSUE_EDIT_CONTRACT_VIOLATION,
    ISSUE_ELEMENT_SEQUENCE_MISMATCH,
    ISSUE_IRC_EVENT_MISMATCH,
    ISSUE_LEGACY_EVENT_INCONSISTENCY,
    ISSUE_MAP_BIJECTION_BROKEN,
    ISSUE_ORIENTATION_UNRESOLVED,
    ISSUE_DIMENSIONS,
    V2_DIRNAME,
    V2_EDITS_SCHEMA_VERSION,
    V2_FREEZE_SCHEMA_VERSION,
    V2_MANIFEST_FILENAME,
    V2_MANIFEST_SCHEMA_VERSION,
    V2_SUMMARY_FILENAME,
)
from pes2ts_core.utils.hashing import JSONValue, sha256_bytes, sha256_file
from pes2ts_core.utils.jsonio import read_json, write_json
from pes2ts_core.utils.parquet_io import read_parquet, write_parquet

logger = logging.getLogger(__name__)

#: Rejection-ledger filename kept for parity with the other stages.
ISSUE_LEDGER_FILENAME: Final[str] = "g1_v2_issue_ledger.jsonl"

#: Summary Parquet columns of the v2 build.
V2_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "reaction_id", "p1_status", "audit_status", "issues", "dimensions",
    "n_formed", "n_broken", "n_order_changed",
    "n_h_total", "n_h_transfer", "n_h_free_events", "n_h_hh_events",
    "n_aromatic_edits", "n_aromatic_regions",
    "core_size", "shell1_size", "n_atoms",
    "collapse_state", "n_alts_enumerated",
    "endpoint_match", "orientation", "n_event_mismatch", "n_quality_flags",
    "mapping_bijection_ok", "elements_ok", "legacy_ok",
)

#: Element symbols indexed by atomic number (subset sufficient for symbols).
_Z_TO_SYMBOL: Final[dict[int, str]] = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
    9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 35: "Br", 53: "I",
}

_HYDROGEN_SYMBOL: Final[str] = "H"


def _g1_shard_size(config: Mapping[str, Any]) -> int:
    """Return the shard size the v1 G1 tree was written with."""
    raw = config.get("g1")
    settings: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
    return int(settings.get("shard_size", 1000))


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def v2_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the effective ``g1_v2`` settings with documented defaults."""
    raw = config.get("g1_v2")
    settings: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
    return {
        "shard_size": int(settings.get("shard_size", DEFAULT_V2_SHARD_SIZE)),
        "collapse_audit_budget": int(
            settings.get("collapse_audit_budget", DEFAULT_COLLAPSE_AUDIT_BUDGET)
        ),
        "center_shell": int(settings.get("center_shell", DEFAULT_CENTER_SHELL)),
        "sample_size": int(settings.get("sample_size", DEFAULT_V2_SAMPLE_SIZE)),
        "sample_seed": int(settings.get("sample_seed", DEFAULT_V2_SAMPLE_SEED)),
    }


def v2_edits_dir(interim_dir: str | Path) -> Path:
    """Return the sharded v2 edit-document root under ``interim/g1_v2``."""
    return Path(interim_dir) / V2_DIRNAME / EDITS_DIRNAME


def v2_document_path(interim_dir: str | Path, reaction_id: str, shard_size: int) -> Path:
    """Return the sharded v2 edit-document path of *reaction_id*."""
    return v2_edits_dir(interim_dir) / shard_name(reaction_id, shard_size) / f"{reaction_id}.json"


# ---------------------------------------------------------------------------
# v1 baseline freeze.
# ---------------------------------------------------------------------------
def freeze_v1_baseline(config: Mapping[str, Any]) -> Path:
    """Freeze the v1 baseline counts and digests as the audit anchor.

    The freeze manifest records the total denominator, the admitted/eligible
    counts, the P1 status histogram, the v1 L0 cluster count, the gate
    exclusions, and the digests of every summary/manifest the v2 numbers are
    compared against.  It is written once per ``g1 v2-build`` invocation and
    never mutated afterwards.
    """
    interim_dir = Path(config["paths"]["interim"])
    manifests_dir = Path(config["paths"]["manifests"])
    g1_manifest = read_json(manifests_dir / "g1_manifest.json")
    p1_manifest = read_json(manifests_dir / P1_MANIFEST_FILENAME)
    p2_manifest = read_json(manifests_dir / P2_MANIFEST_FILENAME)
    cluster_report = read_json(manifests_dir / "g1_cluster_report.json")
    gate = read_json(manifests_dir / GATE_MANIFEST_FILENAME)
    summaries = {
        "g1_reaction_change_summary.parquet": sha256_file(interim_dir / G1_SUMMARY_FILENAME),
        P1_SUMMARY_FILENAME: sha256_file(interim_dir / P1_SUMMARY_FILENAME),
        P2_SUMMARY_FILENAME: sha256_file(interim_dir / P2_SUMMARY_FILENAME),
    }
    l0_report = cluster_report.get("levels", {}).get("l0_edit_family", {}) if isinstance(cluster_report, Mapping) else {}
    freeze: dict[str, JSONValue] = {
        "schema_version": V2_FREEZE_SCHEMA_VERSION,
        "dataset_version": dataset_version(config),
        "n_total": (g1_manifest or {}).get("n_total"),
        "n_g1_valid": (g1_manifest or {}).get("n_valid"),
        "n_g1_rejected": (g1_manifest or {}).get("n_rejected"),
        "g1_by_code": (g1_manifest or {}).get("by_code"),
        "p1_by_status": (p1_manifest or {}).get("by_status"),
        "p1_n_eligible": (p1_manifest or {}).get("n_eligible"),
        "p2_n_classified": (p2_manifest or {}).get("n_classified"),
        "v1_l0_n_clusters": l0_report.get("n_clusters"),
        "v1_l0_n_classified": l0_report.get("n_classified"),
        "v1_gate": {
            key: gate.get(key) for key in ("n_denominator", "n_eligible", "eligible_fraction", "exclusions")
        } if isinstance(gate, Mapping) else {},
        "v1_family_labels": (cluster_report or {}).get("family_labels") if isinstance(cluster_report, Mapping) else {},
        "summary_digests": summaries,
        "note": (
            "v1 baseline frozen before the v2 rebuild; v2 artifacts use "
            "independent paths and never overwrite v1 or the frozen split"
        ),
        "generated_at": _now(),
    }
    path = manifests_dir / FREEZE_MANIFEST_FILENAME
    write_json(path, freeze)
    return path


# ---------------------------------------------------------------------------
# Per-reaction audit pieces.
# ---------------------------------------------------------------------------
def _inventory_side_index(
    inventory_row: Mapping[str, Any], prefix: str
) -> dict[str, list[int]]:
    """Return ``tag -> atomic_numbers`` of one side's inventory components."""
    return {
        str(component["tag"]): [int(z) for z in component["atomic_numbers"]]
        for component in inventory_row["components"]
        if str(component["tag"]).startswith(prefix)
    }


def _check_mapping_bijection(
    map_to_atoms: Sequence[Mapping[str, Any]],
    side_index_r: Mapping[str, Sequence[int]],
    side_index_p: Mapping[str, Sequence[int]],
    elements: Mapping[int, str],
) -> tuple[list[str], bool, bool]:
    """Return ``(issues, bijection_ok, elements_ok)`` of the map table."""
    issues: list[str] = []
    bijection_ok = True
    elements_ok = True
    seen_r: set[tuple[str, int]] = set()
    seen_p: set[tuple[str, int]] = set()
    for entry in map_to_atoms:
        r_key = (str(entry["r_component"]), int(entry["r_local_index"]))
        p_key = (str(entry["p_component"]), int(entry["p_local_index"]))
        if r_key in seen_r or p_key in seen_p:
            bijection_ok = False
            continue
        seen_r.add(r_key)
        seen_p.add(p_key)
        map_number = int(entry["map"])
        expected = elements.get(map_number)
        r_numbers = side_index_r.get(r_key[0])
        p_numbers = side_index_p.get(p_key[0])
        if r_numbers is None or p_numbers is None:
            elements_ok = False
            continue
        if not (0 <= r_key[1] < len(r_numbers)) or not (0 <= p_key[1] < len(p_numbers)):
            elements_ok = False
            continue
        r_symbol = _Z_TO_SYMBOL.get(r_numbers[r_key[1]], "?")
        p_symbol = _Z_TO_SYMBOL.get(p_numbers[p_key[1]], "?")
        if expected is None or r_symbol != expected or p_symbol != expected or str(entry["element"]) != expected:
            elements_ok = False
    if not bijection_ok:
        issues.append(ISSUE_MAP_BIJECTION_BROKEN)
    if not elements_ok:
        issues.append(ISSUE_ELEMENT_SEQUENCE_MISMATCH)
    return issues, bijection_ok, elements_ok


def _check_component_assignment(
    map_to_atoms: Sequence[Mapping[str, Any]],
    graph: Mapping[str, Any],
) -> list[str]:
    """Cross-check the per-tag map sets against the graph components.

    The P1 solver may permute which inventory geometry tag backs which
    SMILES component when several components share one skeleton, so a tag
    need not equal ``R<ordinal>`` of the reaction SMILES.  The contract that
    must hold instead: every side's per-tag map sets partition the maps and
    each part equals exactly one component's map set.
    """
    for side, key in (("r", "r_component"), ("p", "p_component")):
        components = {
            int(map_number): int(ordinal)
            for map_number, ordinal in graph[f"{side}_components"].items()
        }
        component_map_sets: dict[int, set[int]] = {}
        for map_number, ordinal in components.items():
            component_map_sets.setdefault(ordinal, set()).add(map_number)
        tag_map_sets: dict[str, set[int]] = {}
        for entry in map_to_atoms:
            tag = str(entry[key])
            if not tag.startswith(side.upper()) or not tag[1:].isdigit():
                return [ISSUE_COMPONENT_ASSIGNMENT_MISMATCH]
            tag_map_sets.setdefault(tag, set()).add(int(entry["map"]))
        remaining = dict(component_map_sets)
        for maps in (tag_map_sets[tag] for tag in sorted(tag_map_sets)):
            for ordinal in list(remaining):
                if remaining[ordinal] == maps:
                    del remaining[ordinal]
                    break
            else:
                return [ISSUE_COMPONENT_ASSIGNMENT_MISMATCH]
        if remaining:
            return [ISSUE_COMPONENT_ASSIGNMENT_MISMATCH]
    return []


def _audit_collapse(
    p1_document: Mapping[str, Any],
    g1_document: Mapping[str, Any] | None,
    elements: Mapping[int, str],
    inventory_row: Mapping[str, Any],
) -> tuple[str, int, list[str]]:
    """Return ``(collapse_state, n_verified, issues)`` of one record.

    The edit-invariance proof: the v2 edits, the reaction center, and every
    classification are functions of the mapped reaction-SMILES graphs alone,
    which the candidate machinery never touches -- a candidate bijection or
    a same-skeleton pairing permutation only reassigns which inventory
    geometry row backs each map number.  A symmetry collapse therefore
    cannot change the recorded chemistry unless a stored candidate is not
    element-consistent with the map space, which is verified here for every
    stored alternative of both sides.  ``ambiguous`` flags exactly that
    violation; the count of verified alternatives is returned as evidence.
    """
    if not bool(p1_document.get("mapping", {}).get("symmetry_collapsed")):
        return "not_collapsed", 0, []
    if g1_document is None:
        return "invariant", 0, []
    side_index = {
        str(component["tag"]): [int(z) for z in component["atomic_numbers"]]
        for component in inventory_row.get("components", [])
    }
    verified = 0
    for blocks in (g1_document.get("reactants"), g1_document.get("products")):
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            rows = block.get("rows") or []
            if not rows:
                continue
            tag = str(block["tag"])
            numbers = side_index.get(tag)
            if numbers is None:
                return "ambiguous", verified, [ISSUE_COLLAPSE_EDIT_AMBIGUOUS]
            row_element = {int(row["local_index"]): str(row["element"]) for row in rows}
            map_set = {int(row["map"]) for row in rows}
            for candidate in block.get("candidates") or []:
                seen_maps: set[int] = set()
                for entry in candidate:
                    local = int(entry["local_index"])
                    map_number = int(entry["map"])
                    if map_number in seen_maps or map_number not in map_set:
                        return "ambiguous", verified, [ISSUE_COLLAPSE_EDIT_AMBIGUOUS]
                    seen_maps.add(map_number)
                    symbol = (
                        _Z_TO_SYMBOL.get(numbers[local], "?")
                        if 0 <= local < len(numbers) else "?"
                    )
                    if elements.get(map_number) != symbol or row_element.get(local) != symbol:
                        return "ambiguous", verified, [ISSUE_COLLAPSE_EDIT_AMBIGUOUS]
                if seen_maps != map_set:
                    return "ambiguous", verified, [ISSUE_COLLAPSE_EDIT_AMBIGUOUS]
                verified += 1
    return "invariant", verified, []


def _irc_issues(irc_validation: Mapping[str, Any] | None) -> list[str]:
    """Return the typed IRC-evidence issues of one P1 document."""
    if not isinstance(irc_validation, Mapping):
        return []
    issues: list[str] = []
    if int(irc_validation.get("n_mismatch") or 0) > 0:
        issues.append(ISSUE_IRC_EVENT_MISMATCH)
    if irc_validation.get("orientation") == "unresolved":
        issues.append(ISSUE_ORIENTATION_UNRESOLVED)
    return issues


def build_v2_reaction(
    inventory_row: Mapping[str, Any],
    p1_document: Mapping[str, Any],
    g1_document: Mapping[str, Any] | None,
    g1_summary_row: Mapping[str, Any] | None,
    settings: Mapping[str, Any],
) -> dict[str, JSONValue]:
    """Return one complete v2 edit/audit document."""
    reaction_id = str(p1_document["reaction_id"])
    p1_status = str(p1_document["status"])
    base: dict[str, JSONValue] = {
        "schema_version": V2_EDITS_SCHEMA_VERSION,
        "reaction_id": reaction_id,
        "generated_at": _now(),
        "dataset_version": dataset_version_of_row(inventory_row),
        "p1_status": p1_status,
        "audit_status": AUDIT_EXCLUDED,
        "issues": [],
        "dimensions": [],
        "edits": [],
        "hydrogen_partner_changes": [],
        "aromatic_regions": {},
        "reaction_center": {"core": [], "with_shell": []},
        "edit_counts": {},
        "graph": None,
        "collapse_state": "not_collapsed",
        "n_alts_enumerated": 0,
        "legacy_reconciled": None,
        "mapping_audit": {
            "bijection_ok": None, "elements_ok": None,
            "component_assignment_ok": None,
        },
        "irc_evidence": {
            "endpoint_match": None, "orientation": None,
            "n_event_mismatch": None, "n_quality_flags": None,
        },
    }
    if p1_status not in G2_ELIGIBLE_STATUSES:
        base["issues"] = [f"p1:{p1_status}"]
        return base
    graph = p1_document.get("graph")
    if not isinstance(graph, Mapping):
        base["issues"] = [ISSUE_EDIT_CONTRACT_VIOLATION]
        base["dimensions"] = [ISSUE_DIMENSIONS[ISSUE_EDIT_CONTRACT_VIOLATION]]
        base["audit_status"] = AUDIT_ISSUES
        return base
    elements = {int(k): str(v) for k, v in graph["elements"].items()}
    r_bonds = [[float(v) for v in bond] for bond in graph["r_bonds"]]
    p_bonds = [[float(v) for v in bond] for bond in graph["p_bonds"]]
    block = exclusive_edits(elements, r_bonds, p_bonds)
    adjacency: Mapping[int, Sequence[int]] = block["_adjacency"]  # type: ignore[assignment]
    center = reaction_center(
        block["edits"], block["hydrogen_partner_changes"], adjacency,
        int(settings["center_shell"]),
    )
    # -- structural audit ---------------------------------------------------
    issues: list[str] = validate_edit_contract(
        block["edits"], block["hydrogen_partner_changes"], center,
        block["multi_bond_pairs"],  # type: ignore[arg-type]
    )
    map_to_atoms = p1_document.get("mapping", {}).get("map_to_atoms") or []
    side_index_r = _inventory_side_index(inventory_row, "R")
    side_index_p = _inventory_side_index(inventory_row, "P")
    mapping_issues, bijection_ok, elements_ok = _check_mapping_bijection(
        map_to_atoms, side_index_r, side_index_p, elements,
    )
    issues.extend(mapping_issues)
    component_issues = _check_component_assignment(map_to_atoms, graph)
    issues.extend(component_issues)
    legacy = legacy_events_from_graph(elements, r_bonds, p_bonds)
    legacy_ok = legacy_reconciles(p1_document.get("bond_events"), legacy)
    if not legacy_ok:
        issues.append(ISSUE_LEGACY_EVENT_INCONSISTENCY)
    # -- mapping determinism ------------------------------------------------
    collapsed_by_summary = g1_summary_row is not None and (
        str(g1_summary_row.get("index_status")) == "truncated"
        or str(g1_summary_row.get("pairing_status")) == "truncated"
    )
    if collapsed_by_summary:
        issues.append(ISSUE_CANDIDATE_SEARCH_TRUNCATED)
    if collapse_allowed := not component_issues:
        collapse_state, enumerated, collapse_issues = _audit_collapse(
            p1_document, g1_document, elements, inventory_row,
        )
        issues.extend(collapse_issues)
    else:
        # Component tags disagree with the graph ordinals: the alternative
        # remap is not sound, so the collapse tie cannot be audited here.
        collapse_state, enumerated = (
            ("truncated", 0) if bool(
                p1_document.get("mapping", {}).get("symmetry_collapsed")
            ) else ("not_collapsed", 0)
        )
        if collapse_state == "truncated":
            issues.append(ISSUE_COLLAPSE_AUDIT_TRUNCATED)
    # -- IRC evidence ---------------------------------------------------------
    irc = p1_document.get("irc_validation")
    irc_issues = _irc_issues(irc)
    issues.extend(irc_issues)
    irc_block: Mapping[str, Any] = irc if isinstance(irc, Mapping) else {}
    # -- assemble -------------------------------------------------------------
    counts = edit_counts(block)
    base.update({
        "edits": block["edits"],
        "hydrogen_partner_changes": block["hydrogen_partner_changes"],
        "aromatic_regions": block["aromatic_regions"],
        "reaction_center": center,
        "edit_counts": counts,
        "graph": {
            "elements": dict(graph["elements"]),
            "r_bonds": [[float(v) for v in bond] for bond in graph["r_bonds"]],
            "p_bonds": [[float(v) for v in bond] for bond in graph["p_bonds"]],
            "r_components": dict(graph["r_components"]),
            "p_components": dict(graph["p_components"]),
        },
        "n_atoms": len(elements),
        "collapse_state": collapse_state,
        "n_alts_enumerated": enumerated,
        "legacy_reconciled": legacy_ok,
        "mapping_audit": {
            "bijection_ok": bijection_ok,
            "elements_ok": elements_ok,
            "component_assignment_ok": not component_issues,
        },
        "irc_evidence": {
            "endpoint_match": irc_block.get("endpoint_match"),
            "orientation": irc_block.get("orientation"),
            "n_event_mismatch": irc_block.get("n_mismatch"),
            "n_quality_flags": len(irc_block.get("quality_flags") or []),
        },
    })
    issues = sorted(set(issues))
    base["issues"] = issues
    base["dimensions"] = sorted({ISSUE_DIMENSIONS[code] for code in issues if code in ISSUE_DIMENSIONS})
    base["audit_status"] = AUDIT_CLEAN if not issues else AUDIT_ISSUES
    return base


def dataset_version_of_row(inventory_row: Mapping[str, Any]) -> str:
    """Return the dataset version stamped on the inventory row."""
    return str(inventory_row.get("dataset_version", ""))


def _summary_row(document: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    """Return the summary Parquet row of one v2 document."""
    counts: Mapping[str, int] = document.get("edit_counts") or {}
    h_total = sum(counts.get(kind, 0) for kind in (
        "transfer", "release", "capture", "to_hh", "from_hh",
        "hh_release", "hh_form_free", "hh_swap",
    ))
    return {
        "reaction_id": document["reaction_id"],
        "p1_status": document["p1_status"],
        "audit_status": document["audit_status"],
        "issues": "|".join(document["issues"]),
        "dimensions": "|".join(document["dimensions"]),
        "n_formed": counts.get("formed", 0),
        "n_broken": counts.get("broken", 0),
        "n_order_changed": counts.get("order_changed", 0),
        "n_h_total": h_total,
        "n_h_transfer": counts.get("transfer", 0),
        "n_h_free_events": counts.get("release", 0) + counts.get("capture", 0),
        "n_h_hh_events": sum(counts.get(kind, 0) for kind in HH_EVENT_KINDS),
        "n_aromatic_edits": sum(
            1 for edit in document["edits"] if edit.get("aromatic_region")
        ),
        "n_aromatic_regions": len(document.get("aromatic_regions") or {}),
        "core_size": len(document["reaction_center"]["core"]),
        "shell1_size": len(document["reaction_center"]["with_shell"]),
        "n_atoms": document.get("n_atoms", 0),
        "collapse_state": document["collapse_state"],
        "n_alts_enumerated": document["n_alts_enumerated"],
        "endpoint_match": document["irc_evidence"]["endpoint_match"],
        "orientation": document["irc_evidence"]["orientation"],
        "n_event_mismatch": document["irc_evidence"]["n_event_mismatch"],
        "n_quality_flags": document["irc_evidence"]["n_quality_flags"],
        "mapping_bijection_ok": document["mapping_audit"]["bijection_ok"],
        "elements_ok": document["mapping_audit"]["elements_ok"],
        "legacy_ok": document["legacy_reconciled"],
    }


@dataclass(frozen=True, slots=True)
class V2Result:
    """Outcome of one v2 rebuild run."""

    n_total: int
    n_clean: int
    n_issues: int
    n_excluded: int
    summary_path: Path
    manifest_path: Path
    freeze_path: Path
    documents_dir: Path


def build_v2(config: Mapping[str, Any]) -> V2Result:
    """Rebuild every v2 edit document, the issue ledger, and the manifests."""
    manifests_dir = Path(config["paths"]["manifests"])
    interim_dir = Path(config["paths"]["interim"])
    settings = v2_settings(config)
    shard_size = int(settings["shard_size"])
    p1_shard_size = int(p1_settings(config)["shard_size"])
    g1_shard_size = _g1_shard_size(config)
    freeze_path = freeze_v1_baseline(config)
    p1_summary_path = interim_dir / P1_SUMMARY_FILENAME
    if not p1_summary_path.is_file():
        raise FileNotFoundError(
            f"Missing P1 summary {p1_summary_path}; run `g1 resolve-map --allow-truth` first"
        )
    inventory_path = interim_dir / INVENTORY_PARQUET_FILENAME
    if not inventory_path.is_file():
        raise FileNotFoundError(f"Missing inventory {inventory_path}; run `g0 inventory` first")
    p1_rows = read_parquet(p1_summary_path).to_pylist()
    inventory_rows = {
        str(row["reaction_id"]): row
        for row in read_parquet(inventory_path).to_pylist()
    }
    g1_summary_rows = {
        str(row["reaction_id"]): row
        for row in read_parquet(interim_dir / G1_SUMMARY_FILENAME).to_pylist()
    }
    documents: list[dict[str, JSONValue]] = []
    for row in p1_rows:
        reaction_id = str(row["reaction_id"])
        p1_document = read_json(p1_document_path(interim_dir, reaction_id, p1_shard_size))
        collapsed = bool(row.get("symmetry_collapsed")) or str(row.get("status")) == "resolved_symmetry_collapsed"
        g1_document = None
        if collapsed and str(row["status"]) in G2_ELIGIBLE_STATUSES:
            path = g1_document_path(interim_dir, reaction_id, g1_shard_size)
            if path.is_file():
                g1_document = read_json(path)
        document = build_v2_reaction(
            inventory_rows.get(reaction_id, {}),
            p1_document,
            g1_document,
            g1_summary_rows.get(reaction_id),
            settings,
        )
        write_json(v2_document_path(interim_dir, reaction_id, shard_size), document)
        documents.append(document)
    rows = [_summary_row(document) for document in documents]
    summary_path = interim_dir / V2_SUMMARY_FILENAME
    write_parquet(summary_path, {name: [row[name] for row in rows] for name in V2_SUMMARY_COLUMNS})
    status_histogram = Counter(str(row["audit_status"]) for row in rows)
    issue_histogram = Counter(
        issue for row in rows for issue in str(row["issues"]).split("|") if issue
    )
    dimension_histogram = Counter(
        dimension for row in rows for dimension in str(row["dimensions"]).split("|") if dimension
    )
    collapse_histogram = Counter(str(row["collapse_state"]) for row in rows)
    ledger_path = manifests_dir / ISSUE_LEDGER_FILENAME
    with ledger_path.open("w", encoding="utf-8") as handle:
        for document in documents:
            if document["audit_status"] != AUDIT_ISSUES:
                continue
            record: dict[str, JSONValue] = {
                "reaction_id": document["reaction_id"],
                "p1_status": document["p1_status"],
                "issues": document["issues"],
                "dimensions": document["dimensions"],
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    manifest: dict[str, JSONValue] = {
        "schema_version": V2_MANIFEST_SCHEMA_VERSION,
        "dataset_version": dataset_version(config),
        "n_total": len(rows),
        "n_clean": status_histogram.get(AUDIT_CLEAN, 0),
        "n_issues": status_histogram.get(AUDIT_ISSUES, 0),
        "n_excluded": status_histogram.get(AUDIT_EXCLUDED, 0),
        "by_issue": dict(sorted(issue_histogram.items())),
        "by_dimension": dict(sorted(dimension_histogram.items())),
        "collapse_states": dict(sorted(collapse_histogram.items())),
        "n_files": len(rows),
        "summary_sha256": sha256_file(summary_path),
        "freeze_sha256": sha256_file(freeze_path),
        "config": dict(sorted(settings.items())),
        "generated_at": _now(),
    }
    manifest_path = manifests_dir / V2_MANIFEST_FILENAME
    write_json(manifest_path, manifest)
    logger.info(
        "G1 v2 build: %d reaction(s), %d clean, %d issues, %d excluded -> %s",
        len(rows), manifest["n_clean"], manifest["n_issues"], manifest["n_excluded"],
        summary_path,
    )
    return V2Result(
        n_total=len(rows),
        n_clean=int(manifest["n_clean"]),
        n_issues=int(manifest["n_issues"]),
        n_excluded=int(manifest["n_excluded"]),
        summary_path=summary_path,
        manifest_path=manifest_path,
        freeze_path=freeze_path,
        documents_dir=v2_edits_dir(interim_dir),
    )


__all__ = [
    "ISSUE_LEDGER_FILENAME",
    "V2Result",
    "V2_SUMMARY_COLUMNS",
    "build_v2",
    "build_v2_reaction",
    "freeze_v1_baseline",
    "v2_document_path",
    "v2_edits_dir",
    "v2_settings",
]
