"""Deterministic minimal G1 plans for single-bond changes and H transfers."""
from __future__ import annotations

import hashlib
import math
from typing import Any

from pes2ts_core.contracts import ContractError, dumps_document, make_document
from pes2ts_core.utils.hashing import stable_json_dumps


def _short_id(prefix: str, payload: Any) -> str:
    return f"{prefix}-{hashlib.sha256(stable_json_dumps(payload).encode()).hexdigest()[:12]}"


def build_minimal_scan_plan(case: dict[str, Any], *, experiment_id: str = "demo-experiment-v1",
                            method: dict[str, Any] | None = None, n_points: int = 9,
                            budget: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a frozen one-coordinate candidate for a simple bond/H transfer.

    The generated scan is an initial interface milestone: it interpolates the
    measured endpoint distance and is never represented as a validated TS path.
    """
    # Explicit validation before constructing IDs or indices.
    if case.get("schema_name") != "ReactionCase":
        raise ContractError("expected a ReactionCase")
    if case.get("status") != "ready":
        raise ContractError("ReactionCase must be ready; resolve review reasons and endpoint spin first")
    dumps_document(case)
    atoms = case.get("atoms", [])
    atom_maps = [a["atom_map_id"] for a in atoms]
    map_to_index = {atom_map: i for i, atom_map in enumerate(atom_maps)}
    edits = case.get("edits", [])
    bond_edits = [e for e in edits if e.get("kind") in {"formed", "broken"}]
    if not bond_edits:
        raise ContractError("minimal plan supports only single bond formation/breaking and H transfers")
    transfers = case.get("hydrogen_transfers", [])
    if len(transfers) > 1:
        raise ContractError("multiple H transfers require a synchronized or multidimensional plan")
    element_by_map = {a["atom_map_id"]: a["element"] for a in atoms}
    if transfers:
        transfer = transfers[0]
        hydrogen = int(transfer["hydrogen_map_id"])
        old_partner = transfer.get("old_partner_map_id")
        new_partner = transfer.get("new_partner_map_id")
        if transfer.get("kind", "transfer") != "transfer" or old_partner is None or new_partner is None:
            raise ContractError("only a complete heavy-atom H transfer has a 1D plan")
        if element_by_map.get(hydrogen) != "H" or any(
            element_by_map.get(int(partner)) == "H" for partner in (old_partner, new_partner)
        ):
            raise ContractError("H2 formation/dissociation needs the dedicated H2 path strategy")
        required_pairs = {
            tuple(sorted((hydrogen, int(old_partner)))),
            tuple(sorted((hydrogen, int(new_partner)))),
        }
        actual_h_pairs = {tuple(sorted(map(int, edit["atom_map_ids"]))) for edit in bond_edits
                          if hydrogen in map(int, edit.get("atom_map_ids", []))}
        if actual_h_pairs != required_pairs:
            raise ContractError("H transfer bond edits do not match its old/new partner maps")
        unrelated_bond_changes = [edit for edit in bond_edits
                                  if tuple(sorted(map(int, edit["atom_map_ids"]))) not in required_pairs]
        if unrelated_bond_changes:
            raise ContractError("H transfer is coupled to additional bond formation/breaking; use a complex-path strategy")
    else:
        if len(bond_edits) != 1:
            raise ContractError("multiple bond formation/breaking edits require a complex-path strategy")
        pair_check = list(map(int, bond_edits[0]["atom_map_ids"]))
        if len(pair_check) == 2 and all(element_by_map.get(atom_map) == "H" for atom_map in pair_check):
            raise ContractError("H2 formation/dissociation needs the dedicated H2 path strategy")
    # Prefer H-partner changes where available; otherwise choose the first edit
    # in canonical map-pair order. A broader strategy is deferred to G1 v2.
    hydrogen_pairs = [tuple(sorted((int(e["hydrogen_map_id"]), int(e["new_partner_map_id"]))))
                      for e in case.get("hydrogen_transfers", []) if e.get("new_partner_map_id") is not None]
    selected = sorted(bond_edits, key=lambda e: (0 if tuple(sorted(e["atom_map_ids"])) in hydrogen_pairs else 1,
                                                  tuple(sorted(e["atom_map_ids"]))))[0]
    pair = sorted(map(int, selected["atom_map_ids"]))
    if any(m not in map_to_index for m in pair) or pair[0] == pair[1]:
        raise ContractError("bond edit references invalid or identical atom maps")
    start_endpoint = "product" if selected["kind"] == "formed" else "reactant"
    other_endpoint = "reactant" if start_endpoint == "product" else "product"
    start_xyz = case[start_endpoint].get("geometry")
    end_xyz = case[other_endpoint].get("geometry")
    if start_xyz is None or end_xyz is None:
        raise ContractError("endpoint geometries are required to construct a scan plan")
    i, j = (map_to_index[m] for m in pair)
    distance = lambda xyz: sum((xyz[i][k] - xyz[j][k]) ** 2 for k in range(3)) ** 0.5
    start, end = distance(start_xyz), distance(end_xyz)
    if not (0.45 <= start <= 8 and 0.45 <= end <= 8):
        raise ContractError("selected coordinate has implausible endpoint distance")
    if not isinstance(n_points, int) or isinstance(n_points, bool) or not 3 <= n_points <= 101:
        raise ContractError("n_points must be an integer in 3..101")
    points = [round(start + (end - start) * p / (n_points - 1), 6) for p in range(n_points)]
    method = method or {"engine": "xtb", "method": "GFN2-xTB", "basis": None, "solvent": None,
                        "engine_version": None, "parameter_sha256": None}
    frozen_budget = {"max_attempts": 1, "max_cpu_hours": 8.0, "max_wall_seconds": 14400}
    if budget is not None:
        if not isinstance(budget, dict) or set(budget) != set(frozen_budget):
            raise ContractError("budget must define max_attempts, max_cpu_hours, and max_wall_seconds")
        attempts = budget.get("max_attempts")
        if not isinstance(attempts, int) or isinstance(attempts, bool) or not 1 <= attempts <= 5:
            raise ContractError("budget.max_attempts must be an integer in 1..5")
        for key in ("max_cpu_hours", "max_wall_seconds"):
            value = budget.get(key)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value) or value <= 0):
                raise ContractError(f"budget.{key} must be finite and positive")
        frozen_budget = {"max_attempts": attempts, "max_cpu_hours": float(budget["max_cpu_hours"]),
                         "max_wall_seconds": float(budget["max_wall_seconds"])}
    edited_pairs = {tuple(sorted(map(int, edit["atom_map_ids"]))) for edit in bond_edits}
    for transfer in transfers:
        hydrogen = int(transfer["hydrogen_map_id"])
        for partner_key in ("old_partner_map_id", "new_partner_map_id"):
            partner = transfer.get(partner_key)
            if partner is not None:
                edited_pairs.add(tuple(sorted((hydrogen, int(partner)))))
    # Include order changes and any other endpoint edit as a measured observer,
    # even if the one-dimensional driver only scans a formed/broken bond.
    for edit in edits:
        if "atom_map_ids" in edit:
            edited_pairs.add(tuple(sorted(map(int, edit["atom_map_ids"]))))
    observer_pairs = [list(p) for p in sorted(edited_pairs - {tuple(pair)})]
    candidate = {
        "rank": 1, "start_endpoint": start_endpoint, "direction": "P_to_R" if start_endpoint == "product" else "R_to_P",
        "observer_pairs_map": observer_pairs,
        "retry_policy": {"failure_policy": "retry_previous",
                         "scan_retry_count": max(0, frozen_budget["max_attempts"] - 1) if budget is not None else 1,
                         "optimizer_retries": 1, "reuse_previous_geometry": True},
        "method": method, "budget": frozen_budget,
        "coordinates": [{"coordinate_id": "driver-01", "role": "driver", "atom_map_ids": pair,
                         "atom_indices": [map_to_index[m] for m in pair], "unit": "angstrom",
                         "direction": "stretch" if end >= start else "contract", "points": points}],
        "fallback_candidate_ids": [],
    }
    # The plan identity must change whenever any frozen execution policy changes,
    # not only when the driver coordinate, point count, or method changes.
    plan_id = _short_id("plan", {"case_id": case["case_id"], "source_case_sha256": case["content_sha256"],
        "candidate_strategy": "single_bond_linear_endpoint_interpolation_v1", "candidate": candidate})
    candidate["candidate_id"] = f"{plan_id}:c001"
    candidate["request_id"] = f"{plan_id}:c001:attempt1"
    return make_document("ScanPlan", plan_id, "ready", experiment_id=experiment_id,
        dataset_version=case["dataset_version"], reaction_id=case["reaction_id"], case_id=case["case_id"],
        plan_id=plan_id, split=case["split"], plan_version=1, atom_map_ids=atom_maps, n_atoms=len(atom_maps),
        candidate_strategy="single_bond_linear_endpoint_interpolation_v1", candidates=[candidate],
        source_case_sha256=case["content_sha256"], plan_frozen=True)


def build_scan_plan_with_rejection(case: dict[str, Any], *, experiment_id: str = "demo-experiment-v1",
                                   method: dict[str, Any] | None = None, n_points: int = 9,
                                   budget: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return either a minimal executable plan or a durable, reasoned refusal.

    Invalid ReactionCase documents still raise: a rejection record must point
    to a valid, traceable input. Strategy limitations become ScanPlan records
    so downstream coverage reports can count them without parsing log text.
    """
    dumps_document(case)
    if case.get("status") != "ready":
        raise ContractError("ReactionCase must be ready before building a ScanPlan")
    try:
        return build_minimal_scan_plan(case, experiment_id=experiment_id, method=method,
                                       n_points=n_points, budget=budget)
    except ContractError as exc:
        strategy = "single_bond_or_h_transfer_v1"
        reason = str(exc)
        plan_id = _short_id("plan", {"case_id": case["case_id"], "source_case_sha256": case["content_sha256"],
                                      "strategy": strategy, "reject_reason": reason})
        return make_document("ScanPlan", plan_id, "rejected", experiment_id=experiment_id,
            dataset_version=case["dataset_version"], reaction_id=case["reaction_id"], case_id=case["case_id"],
            plan_id=plan_id, split=case["split"], plan_version=1,
            atom_map_ids=[atom["atom_map_id"] for atom in case["atoms"]], n_atoms=len(case["atoms"]),
            candidate_strategy=strategy, candidates=[], source_case_sha256=case["content_sha256"],
            plan_frozen=True, reject_reasons=[reason])
