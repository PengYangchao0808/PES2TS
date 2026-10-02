"""Finite-difference Lagrangian curvature probes, with explicit limitations."""
from __future__ import annotations

import numpy as np

from pes2ts_core.generation.planning.continuation import _jacobian
from pes2ts_core.generation.planning.local_corrector import tangent_projector


def directional_curvature(x, coordinates, gradient, direction, evaluate, *, step=.001, name="probe"):
    x, gradient = np.asarray(x, float), np.asarray(gradient, float)
    projector, rank = tangent_projector(x, coordinates)
    vector = projector@np.asarray(direction, float).ravel()
    norm = np.linalg.norm(vector)
    if norm < 1e-10:
        return {"status": "no_free_direction", "physical_bifurcation_confirmed": False}
    vector /= norm
    multipliers = np.linalg.lstsq(_jacobian(x, coordinates).T, -gradient.ravel(), rcond=1e-7)[0]
    rows = []
    for scale in (step, 2*step):
        lagrangian_gradients = []
        for sign in (-1, 1):
            y = x+sign*scale*vector.reshape(x.shape)
            result = evaluate(y, f"{name}-{scale:g}-{'minus' if sign < 0 else 'plus'}")
            if not result.get("success"):
                return {"status": "qc_failed", "failure_class": result.get("failure_class"),
                        "physical_bifurcation_confirmed": False}
            g = np.asarray(result["gradient_hartree_per_angstrom"], float).ravel()
            lagrangian_gradients.append(g+_jacobian(y, coordinates).T@multipliers)
        h_v = (lagrangian_gradients[1]-lagrangian_gradients[0])/(2*scale)
        rows.append({"step_angstrom": scale, "curvature_hartree_per_angstrom2": float(vector@h_v)})
    return {"status": "measured", **rank, "directions_checked": 1, "measurements": rows,
            "negative_direction_detected": all(r["curvature_hartree_per_angstrom2"] < -.0001 for r in rows),
            "positive_definite_hessian_verified": False, "physical_bifurcation_confirmed": False}
