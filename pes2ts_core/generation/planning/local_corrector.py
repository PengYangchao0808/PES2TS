"""Local, physical-gradient correction on a fixed constraint manifold.

Every evaluated trial stays in the SAME predictor-centred neighbourhood.
No restraint potential is added to the electronic energy. Boundary/stalled
solutions remain failures, even when all driven distances are exact.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import time

import numpy as np

from pes2ts_core.generation.planning.continuation import (
    ContinuationPolicy, _jacobian, project_geometry,
)
from pes2ts_core.generation.planning.synchronized_path import align, residual, value


@dataclass(frozen=True)
class LocalCorrectorPolicy:
    rmsd_radius: float = .15
    atom_radius: float = .35
    iteration_atom_step: float = .05
    gradient_rms_tolerance: float = .0001  # hartree / angstrom
    gradient_max_tolerance: float = .0003
    constraint_tolerance: float = .00001
    max_evaluations: int = 80
    max_iterations: int = 60
    max_seconds: float = 120.
    max_backtracks: int = 12
    initial_inverse_curvature: float = 5.

    def __post_init__(self):
        for name, v in asdict(self).items():
            if isinstance(v, bool) or not math.isfinite(v) or v <= 0:
                raise ValueError(f"INVALID_LOCAL_POLICY:{name}")
        for name in ("max_evaluations", "max_iterations", "max_backtracks"):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f"INVALID_LOCAL_POLICY:{name}")
        if self.iteration_atom_step > self.atom_radius:
            raise ValueError("LOCAL_ITERATION_EXCEEDS_RADIUS")


def motion_evidence(x, reference):
    movement = np.linalg.norm(align(x, reference)-reference, axis=1)
    return {"rmsd_angstrom": float(np.sqrt(np.mean(movement**2))),
            "maximum_atom_step_angstrom": float(movement.max())}


def tangent_projector(x, coordinates):
    """Remove constraints and whole-system rigid modes, never fragment modes."""
    x = np.asarray(x, float)
    j = _jacobian(x, coordinates)
    norms = np.linalg.norm(j, axis=1)
    if np.any(norms < 1e-12) or not np.isfinite(j).all():
        raise ValueError("COORDINATE_DEGENERATE")
    scaled = j / norms[:, None]
    singular = np.linalg.svd(scaled, compute_uv=False)
    rank = int(np.sum(singular > max(1e-9, singular[0]*1e-7)))
    if rank != len(coordinates):
        raise ValueError("DEPENDENT_CONSTRAINTS")
    rigid = []
    centred = x-x.mean(0)
    for axis in np.eye(3):
        rigid.append(np.tile(axis, (len(x), 1)).ravel())
        rotational = np.cross(np.tile(axis, (len(x), 1)), centred).ravel()
        if np.linalg.norm(rotational) > 1e-10:
            rigid.append(rotational)
    rows = np.vstack([scaled] + [r[None, :]/np.linalg.norm(r) for r in rigid])
    u, s, vt = np.linalg.svd(rows, full_matrices=False)
    keep = s > max(1e-9, s[0]*1e-7)
    p = np.eye(x.size)-vt[keep].T@vt[keep]
    return p, {"constraint_rank": rank, "n_constraints": len(coordinates),
               "scaled_jacobian_singular_values": singular.tolist(),
               "scaled_jacobian_condition": float(singular[0]/singular[-1])}


def gradient_quality(g, projector, shape):
    free = (projector@np.asarray(g, float).ravel()).reshape(shape)
    return free, {"free_gradient_rms_hartree_per_angstrom": float(np.sqrt(np.mean(free**2))),
                  "free_gradient_max_hartree_per_angstrom": float(np.abs(free).max()),
                  "physical_gradient_norm_hartree_per_angstrom": float(np.linalg.norm(g))}


def correct_local(guess, coordinates, targets, masses, evaluate, *, policy=None,
                  geometry_check=None, save=None, initial_inverse_curvature=None):
    """Projected quasi-Newton correction with retraction and Armijo search.

    evaluate(X, evaluation_id) returns physical energy and a geometry-bound
    gradient_hartree_per_angstrom. A QC failure never yields an accepted frame.
    The inverse-curvature update is an optimizer approximation, NOT a physical
    Hessian or a certification that the constrained stationary point is a minimum.
    """
    policy = policy or LocalCorrectorPolicy()
    reference = np.asarray(guess, float).copy()
    if reference.ndim != 2 or reference.shape[1] != 3 or not np.isfinite(reference).all():
        raise ValueError("INVALID_LOCAL_GEOMETRY")
    if len(targets) != len(coordinates) or not coordinates:
        raise ValueError("INVALID_LOCAL_TARGETS")
    projection_policy = replace(ContinuationPolicy(), distance_tolerance=policy.constraint_tolerance,
                                angle_tolerance=.001, projection_tolerance=.1,
                                projection_iterations=48, predictor_atom_step=policy.iteration_atom_step)
    x, projection = project_geometry(reference, coordinates, targets, masses, projection_policy)
    started = time.monotonic()
    history, evaluations = [], []
    h_inv = (np.eye(reference.size)*policy.initial_inverse_curvature if initial_inverse_curvature is None
             else np.asarray(initial_inverse_curvature, float).copy())
    if h_inv.shape != (reference.size, reference.size) or not np.isfinite(h_inv).all():
        raise ValueError("INVALID_OPTIMIZER_CURVATURE_STATE")
    if not np.allclose(h_inv, h_inv.T, atol=1e-8) or np.linalg.eigvalsh(h_inv)[0] <= 0:
        raise ValueError("NONPOSITIVE_OPTIMIZER_CURVATURE_STATE")
    energy = None
    physical_gradient = None
    final_quality = {}
    n_evaluations = 0
    reason = projection["reason"]

    def within(y):
        q = motion_evidence(y, reference)
        return (q["rmsd_angstrom"] < policy.rmsd_radius and
                q["maximum_atom_step_angstrom"] < policy.atom_radius), q

    def finish(failure):
        record = {"success": failure is None, "converged": failure is None,
                  "failure_class": failure, "coordinates": x.tolist() if x is not None else None,
                  "energy": energy, "duration_seconds": time.monotonic()-started,
                  "physical_gradient_status": "bound" if physical_gradient is not None else "not_collected",
                  "physical_gradient_hartree_per_angstrom": physical_gradient.tolist() if physical_gradient is not None else None,
                  "local_policy": asdict(policy), "locality_reference": reference.tolist(),
                  "optimizer": "projected_bfgs_fixed_neighbourhood_v1", "local_history": history,
                  "evaluations": evaluations, "n_gradient_evaluations": n_evaluations,
                  "quality": final_quality, "restraint_energy_added": False,
                  "optimizer_state": {"inverse_curvature": h_inv.tolist()} if failure is None else None,
                  "stationary_point_claimed": False, "constrained_minimum_verified": False}
        if save:
            save(record)
        return record

    def sample(y):
        nonlocal n_evaluations
        if n_evaluations >= policy.max_evaluations or time.monotonic()-started >= policy.max_seconds:
            return None, "LOCAL_BUDGET_EXHAUSTED"
        ok, motion = within(y)
        if not ok:
            return None, "LOCALITY_LIMIT"
        if geometry_check:
            failure = geometry_check(y)
            if failure:
                return None, failure
        evaluation_id = f"gradient-{n_evaluations:04d}"
        n_evaluations += 1
        result = evaluate(y, evaluation_id)
        evaluations.append({"evaluation_id": evaluation_id, "success": result.get("success", False),
                            "request_sha256": result.get("request_sha256"),
                            "gradient_evidence": result.get("gradient_evidence"), **motion})
        e = result.get("energy")
        g = result.get("gradient_hartree_per_angstrom")
        if not result.get("success"):
            return None, result.get("failure_class") or "GRADIENT_FAILED"
        if e is None or not math.isfinite(e) or g is None:
            return None, "INVALID_PHYSICAL_GRADIENT"
        g = np.asarray(g, float)
        if g.shape != y.shape or not np.isfinite(g).all():
            return None, "INVALID_PHYSICAL_GRADIENT"
        return (float(e), g), None

    if reason:
        return finish(reason)
    sampled, reason = sample(x)
    if reason:
        return finish(reason)
    energy, physical_gradient = sampled
    for iteration in range(policy.max_iterations+1):
        try:
            p, rank = tangent_projector(x, coordinates)
        except (ValueError, np.linalg.LinAlgError) as exc:
            return finish(str(exc))
        free, quality = gradient_quality(physical_gradient, p, x.shape)
        final_quality = {**quality, **rank, **motion_evidence(x, reference)}
        errors = [abs(residual(value(x, d["kind"], d["atoms"]), t, d["kind"]))
                  for d, t in zip(coordinates, targets)]
        final_quality["constraint_residuals"] = errors
        history.append({"iteration": iteration, "energy_hartree": energy, **final_quality})
        if save:
            save({"status": "running", "history": history, "n_gradient_evaluations": n_evaluations})
        converged = (quality["free_gradient_rms_hartree_per_angstrom"] <= policy.gradient_rms_tolerance
                     and quality["free_gradient_max_hartree_per_angstrom"] <= policy.gradient_max_tolerance
                     and all(e <= (policy.constraint_tolerance if d["kind"] == "distance" else .001)
                             for e, d in zip(errors, coordinates)))
        if converged:
            return finish(None)
        if iteration == policy.max_iterations:
            return finish("LOCAL_ITERATION_LIMIT")
        direction = -(p@h_inv@free.ravel()).reshape(x.shape)
        if float(np.sum(direction*free)) >= -1e-16:
            h_inv = np.eye(x.size)*policy.initial_inverse_curvature
            direction = -free*policy.initial_inverse_curvature
        largest = np.linalg.norm(direction, axis=1).max()
        if largest > policy.iteration_atom_step:
            direction *= policy.iteration_atom_step/largest
        accepted = None
        saw_boundary = False
        for backtrack in range(policy.max_backtracks):
            alpha = .5**backtrack
            trial, evidence = project_geometry(x+alpha*direction, coordinates, targets, masses, projection_policy)
            if trial is None:
                continue
            trial = align(trial, reference)
            if not within(trial)[0]:
                saw_boundary = True
                continue
            trial_sample, failure = sample(trial)
            if failure:
                return finish(failure)
            trial_energy, trial_gradient = trial_sample
            displacement = trial-x
            armijo = 1e-4*float(np.sum(physical_gradient*displacement))
            if trial_energy <= energy+armijo+1e-11:
                accepted = trial, trial_energy, trial_gradient
                break
        if accepted is None:
            return finish("LOCALITY_LIMIT" if saw_boundary else "LOCAL_STATIONARITY_STALLED")
        trial, trial_energy, trial_gradient = accepted
        s = (trial-x).ravel()
        new_p, _ = tangent_projector(trial, coordinates)
        y = new_p@trial_gradient.ravel()-free.ravel()
        sy = float(s@y)
        if sy > 1e-10 and sy > 1e-5*np.linalg.norm(s)*np.linalg.norm(y):
            v = np.eye(x.size)-np.outer(s, y)/sy
            h_inv = v@h_inv@v.T+np.outer(s, s)/sy
        x, energy, physical_gradient = trial, trial_energy, trial_gradient
    return finish("LOCAL_ITERATION_LIMIT")
