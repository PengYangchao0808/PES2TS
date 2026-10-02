"""Mapped displacement and local-coordinate evidence, without TS references."""
from __future__ import annotations

from itertools import combinations, product
import numpy as np

from pes2ts_core.generation.planning.continuation import _jacobian
from pes2ts_core.generation.planning.local_corrector import motion_evidence
from pes2ts_core.generation.planning.synchronized_path import align, residual, value


def local_monitors(plan, bonds):
    neighbours = {i: set() for i in range(len(plan["elements"]))}
    for a, b in bonds:
        neighbours[a].add(b)
        neighbours[b].add(a)
    monitors = []
    centres = {a for d in plan["drivers"] for a in d["atoms"]}
    for b, ns in neighbours.items():
        for a, c in combinations(sorted(ns), 2):
            if centres & {a, b, c}:
                monitors.append({"kind": "angle", "atoms": [a, b, c]})
    for b, c in sorted(tuple(sorted(pair)) for pair in bonds):
        for a, d in product(sorted(neighbours[b]-{c}), sorted(neighbours[c]-{b})):
            if len({a, b, c, d}) == 4 and centres & {a, b, c, d}:
                monitors.append({"kind": "dihedral", "atoms": [a, b, c, d]})
    return monitors


def segment_diagnostics(previous, predictor, corrected, plan, targets, bonds):
    previous, predictor, corrected = [np.asarray(x, float) for x in (previous, predictor, corrected)]
    movement = np.linalg.norm(align(corrected, predictor)-predictor, axis=1)
    top = sorted(range(len(movement)), key=lambda i: (-movement[i], i))[:8]
    rows = []
    for monitor in local_monitors(plan, bonds):
        with np.errstate(invalid="ignore", divide="ignore"):
            a, b = value(predictor, monitor["kind"], monitor["atoms"]), value(corrected, monitor["kind"], monitor["atoms"])
            # Dihedral is undefined when one adjacent angle is nearly linear.
            angles = [value(predictor, "angle", monitor["atoms"][s:s+3])
                      for s in (0, 1)] if monitor["kind"] == "dihedral" else [90.]
        if not np.isfinite([a, b]+angles).all() or any(t < 5. or t > 175. for t in angles):
            continue
        rows.append({**monitor, "maps": [plan["atom_map_order"][i] for i in monitor["atoms"]],
                     "predictor_degrees": a, "corrected_degrees": b,
                     "change_degrees": residual(b, a, monitor["kind"])})
    rows.sort(key=lambda r: -abs(r["change_degrees"]))
    coordinates = plan["drivers"]+plan.get("guards", [])
    j = _jacobian(previous, coordinates)
    scales = np.linalg.norm(j, axis=1)
    scaled = j/np.maximum(scales[:, None], 1e-12)
    s = np.linalg.svd(scaled, compute_uv=False)
    rank = int(np.sum(s > max(1e-9, s[0]*1e-7)))
    increment = [residual(t, value(previous, d["kind"], d["atoms"]), d["kind"])
                 for d, t in zip(coordinates, targets)]
    achieved = [residual(value(corrected, d["kind"], d["atoms"]), t, d["kind"])
                for d, t in zip(coordinates, targets)]
    return {"predictor_motion": motion_evidence(predictor, previous),
            "corrector_motion": motion_evidence(corrected, predictor),
            "total_motion": motion_evidence(corrected, previous),
            "largest_moving_atoms": [{"atom_index": i, "map": plan["atom_map_order"][i],
                                      "element": plan["elements"][i], "displacement_angstrom": float(movement[i])} for i in top],
            "local_coordinate_changes": rows[:12], "actual_target_increments": increment,
            "final_constraint_residuals": achieved,
            "scaled_jacobian_rank": rank, "scaled_jacobian_singular_values": s.tolist(),
            "scaled_jacobian_condition": float(s[0]/s[-1]) if s[-1] > 1e-12 else None,
            "physical_bifurcation_confirmed": False}
