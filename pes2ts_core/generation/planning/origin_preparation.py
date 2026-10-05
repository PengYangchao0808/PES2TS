"""Versioned, topology-preserving input repairs and torsional hypotheses.

G2-AB1 WP-3 adds the versioned origin-preparation path migrated from the
archived experiment scripts: it screens the input, drives an unconstrained
optimization through an injected (ACP-backed in production) evaluator, and
emits ``endpoint_method_evidence_v1`` with typed structured failures — a lost
connection names the exact atom pairs with before/after distances and the
original structure hash instead of a generic invalid-geometry code.

Repairs (overlap separation, torsion hypotheses) are SOURCED INPUT repairs:
they produce a NEW input identity (new hash + provenance) and never overwrite
an endpoint. Nothing here reads ground truth.
"""
from __future__ import annotations

import time

import numpy as np

from pes2ts_core.generation.planning.synchronized_path import digest

#: Proposed interface registered under constitution §9.6 (P1: existence and
#: validation only — fields are explicitly NOT frozen).
ENDPOINT_METHOD_EVIDENCE_SCHEMA = "endpoint_method_evidence_v1"

#: Typed origin-preparation failure codes (WP-3); the generic
#: PREPARED_ORIGIN_GEOMETRY_INVALID is retired.
ORIGIN_FAILURE_CODES = ("ORIGIN_CONNECTION_LOST", "ORIGIN_OVERLAP_INPUT",
                        "ORIGIN_SCF_BRANCH", "ORIGIN_BUDGET_EXHAUSTED",
                        "ORIGIN_PREPARED_COLLISION")


def _bond_screen(x, elements, bond_indices):
    """Bonded-pair geometry screen shared by the input and prepared checks."""
    from rdkit import Chem
    periodic = Chem.GetPeriodicTable()
    radii = [periodic.GetRcovalent(periodic.GetAtomicNumber(e)) for e in elements]
    bonds = {tuple(sorted(p)) for p in bond_indices}
    lost, collisions = [], []
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            distance = float(np.linalg.norm(np.asarray(x[i])-x[j]))
            total = radii[i]+radii[j]
            if (i, j) in bonds:
                if distance > total+.8 or distance < .45*total:
                    lost.append({"atoms": [i, j], "distance": distance,
                                 "covalent_radius_sum": total})
            elif distance < .55*total:
                collisions.append({"atoms": [i, j], "distance": distance,
                                   "covalent_radius_sum": total})
    return lost, collisions


def _collapsed_bonds(x, elements, bond_indices):
    """Bonded pairs squeezed below the bonded-geometry floor (overlap class)."""
    from rdkit import Chem
    periodic = Chem.GetPeriodicTable()
    radii = [periodic.GetRcovalent(periodic.GetAtomicNumber(e)) for e in elements]
    collapsed = []
    for i, j in {tuple(sorted(p)) for p in bond_indices}:
        distance = float(np.linalg.norm(np.asarray(x[i])-x[j]))
        total = radii[i]+radii[j]
        if distance < .45*total:
            collapsed.append({"atoms": [i, j], "distance": distance,
                              "covalent_radius_sum": total})
    return collapsed


def prepare_origin(geometry, elements, bond_indices, evaluate_free, *,
                   method_context=None, budget_seconds=None, parent=None):
    """Versioned origin preparation through an injected free-optimization backend.

    ``evaluate_free(geometry, evaluation_id)`` runs the unconstrained
    optimization (ACP-backed in production; test doubles must be marked and
    never enter production paths, R3) and returns a record with
    ``success``/``failure_class``/``coordinates``/``energy`` and optional
    ``gradient_hartree_per_angstrom``/``duration_seconds``/``acp_receipt_ref``.

    Returns ``endpoint_method_evidence_v1``.  Failure taxonomy (typed, never a
    single generic code): ``ORIGIN_OVERLAP_INPUT`` (terminal — an overlapped
    input is an invalid origin, never "repaired" into the original),
    ``ORIGIN_SCF_BRANCH``, ``ORIGIN_BUDGET_EXHAUSTED``,
    ``ORIGIN_CONNECTION_LOST`` (exact atom pairs, before/after distances,
    original structure hash), ``ORIGIN_PREPARED_COLLISION``.
    ``minimum_status`` stays ``unknown`` — curvature is not computed here, so
    no minimum is ever claimed verified.
    """
    started = time.monotonic()
    x = np.asarray(geometry, float)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all() or len(x) != len(elements):
        raise ValueError("INVALID_ORIGIN_INPUT")
    source_geometry_hash = digest(x.tolist())
    method_context_hash = digest(dict(method_context or {}))

    def evidence(*, status, prepared=None, failure=None, detail=None, evaluator=None):
        prepared_hash = digest(np.asarray(prepared, float).tolist()) if prepared is not None else None
        return {"schema_version": ENDPOINT_METHOD_EVIDENCE_SCHEMA,
                "status": status,
                "failure_code": failure,
                "failure_detail": detail,
                "source_geometry_hash": source_geometry_hash,
                "prepared_geometry_hash": prepared_hash,
                "method_context_hash": method_context_hash,
                "stationarity_status": ("gradient_bound_converged"
                                        if status == "prepared" and evaluator
                                        and evaluator.get("gradient_hartree_per_angstrom") is not None
                                        else "not_certified" if status == "prepared"
                                        else "not_reached"),
                "minimum_status": "unknown",
                "identity_status": ("connection_preserved" if status == "prepared"
                                    else "not_verified"),
                "gradient_ref": (evaluator or {}).get("gradient_ref"),
                "frequency_ref": {"value": None,
                                  "reason": "frequency_not_computed_by_origin_preparation"},
                "acp_receipt_ref": (evaluator or {}).get("acp_receipt_ref"),
                "parent": parent,
                "duration_seconds": time.monotonic()-started}

    _, input_collisions = _bond_screen(x, elements, bond_indices)
    input_collapse = _collapsed_bonds(x, elements, bond_indices)
    if input_collisions or input_collapse:
        return evidence(status="invalid_input", failure="ORIGIN_OVERLAP_INPUT",
                        detail={"collisions": input_collisions,
                                "collapsed_bonds": input_collapse})

    evaluation_id = f"origin-0000/eval-{digest({'geometry': x.tolist(), 'operation': 'origin_free_optimization', 'method_context': dict(method_context or {})})[:16]}"
    record = evaluate_free(x, evaluation_id)
    duration = float(record.get("duration_seconds") or 0.)
    if budget_seconds is not None and duration > budget_seconds:
        return evidence(status="failed", failure="ORIGIN_BUDGET_EXHAUSTED",
                        detail={"allowed_seconds": budget_seconds,
                                "observed_seconds": duration}, evaluator=record)
    if not record.get("success"):
        failure = record.get("failure_class") or "ORIGIN_EVALUATION_FAILED"
        if "SCF" in str(failure).upper():
            failure = "ORIGIN_SCF_BRANCH"
        return evidence(status="failed", failure=failure,
                        detail={"evaluator_failure_class": record.get("failure_class")},
                        evaluator=record)
    prepared = np.asarray(record.get("coordinates"), float)
    if prepared.shape != x.shape or not np.isfinite(prepared).all():
        return evidence(status="failed", failure="ORIGIN_EVALUATION_FAILED",
                        detail={"reason": "INVALID_PREPARED_GEOMETRY"}, evaluator=record)
    before = {tuple(sorted(p)): float(np.linalg.norm(x[p[0]]-x[p[1]])) for p in bond_indices}
    lost, collisions = _bond_screen(prepared, elements, bond_indices)
    if lost:
        detail = [{"atoms": row["atoms"],
                   "distance_before_angstrom": before.get(tuple(row["atoms"])),
                   "distance_after_angstrom": row["distance"]}
                  for row in lost]
        return evidence(status="failed", failure="ORIGIN_CONNECTION_LOST",
                        detail={"lost_connections": detail,
                                "source_geometry_hash": source_geometry_hash},
                        prepared=prepared, evaluator=record)
    if collisions:
        return evidence(status="failed", failure="ORIGIN_PREPARED_COLLISION",
                        detail={"collisions": collisions}, prepared=prepared,
                        evaluator=record)
    result = evidence(status="prepared", prepared=prepared, evaluator=record)
    result["coordinates"] = prepared.tolist()
    result["energy"] = record.get("energy")
    return result


def apply_input_repair(source_geometry, repaired_geometry, repair_kind, repair_detail):
    """Bless a repaired geometry as a NEW input identity with provenance.

    The source endpoint is never modified or re-labelled: the repair output
    carries its own hash and full provenance back to the source hash.
    """
    source = np.asarray(source_geometry, float)
    repaired = np.asarray(repaired_geometry, float)
    if source.shape != repaired.shape:
        raise ValueError("INVALID_REPAIR_SHAPE")
    return {"geometry": repaired.tolist(),
            "provenance": {"repair_kind": repair_kind,
                           "repair_detail": repair_detail,
                           "source_geometry_hash": digest(source.tolist()),
                           "repaired_geometry_hash": digest(repaired.tolist()),
                           "original_geometry_unchanged": True}}


def repair_component_overlap(geometry, elements, components, *, max_iterations=200):
    from rdkit import Chem
    x = np.asarray(geometry,float).copy()
    original = x.copy()
    table=Chem.GetPeriodicTable()
    covalent=[table.GetRcovalent(e) for e in elements]
    vdw=[table.GetRvdw(e) for e in elements]
    groups=[list(group) for group in components]
    if sorted(i for group in groups for i in group) != list(range(len(x))):
        raise ValueError("INVALID_COMPONENT_PARTITION")
    translations=np.zeros((len(groups),3))
    for iteration in range(max_iterations):
        worst=None
        for a in range(len(groups)):
            for b in range(a+1,len(groups)):
                for i in groups[a]:
                    for j in groups[b]:
                        separation=max(covalent[i]+covalent[j]+.8,.75*(vdw[i]+vdw[j]))
                        distance=np.linalg.norm(x[j]-x[i])
                        shortage=separation-distance
                        if shortage > 1e-6 and (worst is None or shortage>worst[0]):
                            worst=(shortage,a,b,i,j,distance)
        if worst is None:
            return x,{"status":"repaired" if np.any(translations) else "unchanged",
                      "iterations":iteration,"component_translations_angstrom":translations.tolist(),
                      "internal_geometry_preserved":all(np.allclose(
                          x[group]-x[group[0]],original[group]-original[group[0]]) for group in groups)}
        shortage,a,b,i,j,distance=worst
        direction=(x[j]-x[i])/distance if distance>1e-10 else np.eye(3)[(a+b)%3]
        movement=(shortage+.001)*direction
        x[groups[b]]+=movement
        translations[b]+=movement
    return None,{"status":"component_placement_failed","iterations":max_iterations}


def torsion_hypotheses(geometry, bonds, *, maximum=2, rotatable_bonds=None):
    """Rotate a disconnected side of a non-ring internal bond; no atom jitter."""
    x=np.asarray(geometry,float)
    neighbours={i:set() for i in range(len(x))}
    for a,b in bonds:
        neighbours[a].add(b);neighbours[b].add(a)
    rows=[]
    eligible = {tuple(sorted(pair)) for pair in (rotatable_bonds if rotatable_bonds is not None else bonds)}
    for a,b in sorted(tuple(sorted(pair)) for pair in bonds):
        if (a,b) not in eligible:
            continue
        if len(neighbours[a])<2 or len(neighbours[b])<2:
            continue
        visited={b};queue=[b]
        for node in queue:
            for other in neighbours[node]:
                if {node,other}=={a,b} or other in visited:
                    continue
                visited.add(other);queue.append(other)
        if a in visited:
            continue
        axis=x[b]-x[a];norm=np.linalg.norm(axis)
        if norm<1e-10:
            continue
        axis/=norm
        for degrees in (60.,-60.):
            angle=np.radians(degrees)
            ids=sorted(visited)
            v=x[ids]-x[a]
            rotated=v*np.cos(angle)+np.cross(np.tile(axis,(len(ids),1)),v)*np.sin(angle)+np.outer(v@axis,axis)*(1-np.cos(angle))
            trial=x.copy();trial[ids]=rotated+x[a]
            rows.append((trial,{"rotation_bond_atoms":[a,b],"rotation_degrees":degrees,"rotated_atoms":ids}))
            if len(rows)>=maximum:
                return rows
    return rows
