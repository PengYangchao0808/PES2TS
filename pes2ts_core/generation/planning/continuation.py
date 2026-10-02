"""Finite-budget, connectivity-only predictor/corrector continuation.

The backend consumes one geometry and one complete target set at a time.
Only accepted frames update state. No endpoint or TS geometries are read here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable

import numpy as np

from pes2ts_core.generation.planning.connectivity import validate_driver_pairs
from pes2ts_core.generation.planning.synchronized_path import align, coordinate_jacobian, digest, residual, value


@dataclass(frozen=True)
class ContinuationPolicy:
    initial_step: float = .04
    min_step: float = .000625
    max_step: float = .08
    max_frames: int = 80
    max_attempts: int = 160
    max_seconds: float = 600.
    rmsd_limit: float = .30
    atom_step_limit: float = .60
    smooth_rmsd: float = .15
    distance_tolerance: float = .01
    angle_tolerance: float = .5
    projection_tolerance: float = .001
    projection_iterations: int = 24
    predictor_atom_step: float = .12
    secant_weight: float = .5
    predictor_enabled: bool = True
    adaptive_enabled: bool = True
    require_physical_gradient: bool = False
    free_gradient_rms_tolerance: float = .0003
    free_gradient_max_tolerance: float = .001
    relative_constraint_tolerance: float = 0.
    constraint_precision_floor: float = .00001

    def __post_init__(self):
        numeric = [v for v in asdict(self).values()]
        if not all(math.isfinite(v) for v in numeric):
            raise ValueError("NONFINITE_POLICY")
        if not (0 < self.min_step <= self.initial_step <= self.max_step <= 1):
            raise ValueError("INVALID_STEP_POLICY")
        if self.max_frames < 2 or self.max_attempts < 1 or self.max_seconds <= 0:
            raise ValueError("INVALID_BUDGET")
        if any(v <= 0 for v in (self.rmsd_limit, self.atom_step_limit, self.distance_tolerance,
                                self.angle_tolerance, self.projection_tolerance,
                                self.projection_iterations, self.predictor_atom_step)):
            raise ValueError("INVALID_TOLERANCE")
        if not 0 <= self.secant_weight <= 1:
            raise ValueError("INVALID_SECANT_WEIGHT")
        if not 0 <= self.relative_constraint_tolerance <= 1:
            raise ValueError("INVALID_RELATIVE_CONSTRAINT_TOLERANCE")
        if min(self.free_gradient_rms_tolerance, self.free_gradient_max_tolerance,
               self.constraint_precision_floor) <= 0:
            raise ValueError("INVALID_GRADIENT_OR_PRECISION_TOLERANCE")


def validate_plan(plan):
    if plan.get("scope") != "connectivity_only" or plan.get("parameter_dimension") != 1:
        raise ValueError("INVALID_CONTINUATION_SCOPE")
    maps = plan["atom_map_order"]
    x = np.asarray(plan["start_geometry"], float)
    if len(set(maps)) != len(maps) or len(plan["elements"]) != len(maps):
        raise ValueError("INVALID_ATOM_IDENTITY")
    if x.shape != (len(maps), 3) or not np.isfinite(x).all():
        raise ValueError("INVALID_GEOMETRY")
    active = plan["active_edits"]
    required = {tuple(sorted(e["maps"])) for e in active}
    if len(required) != len(active) or any(e["edit_kind"] not in {"formed", "broken"} for e in active):
        raise ValueError("INVALID_ACTIVE_EDIT")
    validate_driver_pairs(plan["drivers"], required)
    kinds = {tuple(sorted(e["maps"])): e["edit_kind"] for e in active}
    ids = set()
    for d in plan["drivers"] + plan.get("guards", []):
        if d["id"] in ids:
            raise ValueError("DUPLICATE_COORDINATE")
        ids.add(d["id"])
        if d in plan["drivers"] and d["edit_kind"] != kinds[tuple(sorted(d["maps"]))]:
            raise ValueError("DRIVER_EDIT_KIND_MISMATCH")
        atoms = d["atoms"]
        if len(atoms) != {"distance": 2, "angle": 3, "dihedral": 4}[d["kind"]] or len(set(atoms)) != len(atoms):
            raise ValueError("INVALID_COORDINATE")
        if any(not isinstance(i, int) or i < 0 or i >= len(maps) for i in atoms):
            raise ValueError("INVALID_ATOM_INDEX")
        if d in plan["drivers"] and [maps[i] for i in atoms] != d["maps"]:
            raise ValueError("DRIVER_MAP_INDEX_MISMATCH")
        grid = np.asarray(d["lambda_values"], float)
        targets = np.asarray(d["values"], float)
        if grid.ndim != 1 or len(grid) < 2 or targets.shape != grid.shape or not np.isfinite(grid).all() or not np.isfinite(targets).all():
            raise ValueError("INVALID_TARGET_FUNCTION")
        if grid[0] != 0 or grid[-1] != 1 or np.any(np.diff(grid) <= 0):
            raise ValueError("INVALID_TARGET_GRID")
        if d["kind"] == "distance" and np.any(targets <= 0):
            raise ValueError("INVALID_DISTANCE_TARGET")
    centres = {a for e in active for a in e["maps"]}
    support = centres | set(plan.get("local_support_maps", []))
    for g in plan.get("guards", []):
        if g["kind"] not in {"angle", "dihedral"} or not set(g["maps"]) <= support or not set(g["maps"]) & centres:
            raise ValueError("NONLOCAL_OR_DISTANCE_GUARD")
        if [maps[i] for i in g["atoms"]] != g["maps"]:
            raise ValueError("GUARD_MAP_INDEX_MISMATCH")
    if plan.get("guards"):
        # Reject redundant or singular guards before spending a QC call.
        from pes2ts_core.generation.planning.synchronized_path import coordinate_rank
        rank = coordinate_rank(x, plan["drivers"] + plan["guards"])
        if rank["rank"] != len(plan["drivers"])+len(plan["guards"]):
            raise ValueError("DEPENDENT_OR_SINGULAR_GUARD")
    ContinuationPolicy(**plan["policy"])
    return plan


def target_values(plan, lam):
    values = []
    for d in plan["drivers"] + plan.get("guards", []):
        q = np.asarray(d["values"], float)
        if d["kind"] == "dihedral":
            q = np.degrees(np.unwrap(np.radians(q)))
        values.append(float(np.interp(lam, d["lambda_values"], q)))
    return values


def _jacobian(x, coordinates):
    # Analytic B rows avoid 6N finite-difference evaluations per projection.
    if any(d["kind"] != "distance" for d in coordinates):
        return coordinate_jacobian(x, coordinates)
    j = np.zeros((len(coordinates), x.size))
    for row, d in enumerate(coordinates):
        a, b = d["atoms"]
        v = x[a] - x[b]
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            raise ValueError("COORDINATE_DEGENERATE")
        j[row, 3*a:3*a+3] = v / norm
        j[row, 3*b:3*b+3] = -v / norm
    return j


def project_geometry(geometry, coordinates, targets, masses, policy, tangent=None):
    """Iterative mass-weighted projection, with explicit infeasibility evidence."""
    x = np.asarray(geometry, float).copy()
    mass = np.repeat(np.asarray(masses, float), 3)
    if mass.shape != (x.size,) or np.any(mass <= 0) or not np.isfinite(mass).all():
        raise ValueError("INVALID_MASSES")
    scales = np.asarray([policy.distance_tolerance if d["kind"] == "distance"
                         else np.radians(policy.angle_tolerance) for d in coordinates])
    try:
        if tangent is not None:
            j = _jacobian(x, coordinates)
            v = np.asarray(tangent, float).ravel()
            v = v - np.linalg.pinv(j, rcond=1e-7) @ (j @ v)
            v = v.reshape(x.shape)
            largest = np.linalg.norm(v, axis=1).max()
            if largest > policy.predictor_atom_step:
                v *= policy.predictor_atom_step / largest
            x += policy.secant_weight * v
        for iteration in range(policy.projection_iterations + 1):
            delta = np.asarray([residual(t, value(x, d["kind"], d["atoms"]), d["kind"]) *
                                (1 if d["kind"] == "distance" else np.pi/180)
                                for d, t in zip(coordinates, targets)])
            error = float(np.max(np.abs(delta) / scales))
            if not math.isfinite(error):
                return None, {"reason": "COORDINATE_DEGENERATE", "iterations": iteration}
            if error <= policy.projection_tolerance:
                return x, {"reason": None, "iterations": iteration, "normalized_residual": error}
            if iteration == policy.projection_iterations:
                break
            j = _jacobian(x, coordinates)
            a = j / scales[:, None] / np.sqrt(mass)[None, :]
            rhs = delta / scales
            u, singular, vt = np.linalg.svd(a, full_matrices=False)
            keep = singular > max(1e-9, singular[0]*1e-7)
            dx = (vt[keep].T @ ((u[:, keep].T @ rhs) / singular[keep])) / np.sqrt(mass)
            unreachable = rhs - a @ (dx * np.sqrt(mass))
            if np.max(np.abs(unreachable)) > 1.:
                return None, {"reason": "CONSTRAINT_INFEASIBLE", "rank": int(sum(keep)),
                              "unreachable_residual": float(np.max(np.abs(unreachable)))}
            dx = dx.reshape(x.shape)
            largest = np.linalg.norm(dx, axis=1).max()
            if largest > policy.predictor_atom_step:
                dx *= policy.predictor_atom_step / largest
            x += dx
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return None, {"reason": "COORDINATE_DEGENERATE"}
    return None, {"reason": "PROJECTION_LIMIT", "iterations": iteration, "normalized_residual": error}


def geometry_quality(previous, x, coordinates, targets, policy, previous_targets=None):
    aligned = align(x, previous)
    motion = np.linalg.norm(aligned - previous, axis=1)
    rmsd = float(np.sqrt(np.mean(motion**2)))
    atom_step = float(motion.max())
    tolerances = [policy.distance_tolerance if d["kind"] == "distance" else policy.angle_tolerance
                  for d in coordinates]
    if policy.relative_constraint_tolerance and previous_targets is not None:
        tolerances = [min(tol, max(policy.constraint_precision_floor,
                          policy.relative_constraint_tolerance*abs(residual(t, old, d["kind"]))))
                      if d["kind"] == "distance" else tol
                      for tol, t, old, d in zip(tolerances, targets, previous_targets, coordinates)]
    absolute_errors = [abs(residual(value(x, d["kind"], d["atoms"]), t, d["kind"]))
                       for d, t in zip(coordinates, targets)]
    errors = [err/tol for err, tol in zip(absolute_errors, tolerances)]
    q = {"rmsd_angstrom": rmsd, "maximum_atom_step_angstrom": atom_step,
         "normalized_constraint_residual": max(errors), "constraint_residuals": absolute_errors,
         "effective_constraint_tolerances": tolerances}
    q["reason"] = ("CONSTRAINT_RESIDUAL" if max(errors) > 1 or not np.isfinite(errors).all()
                   else "PATH_DISCONTINUITY" if rmsd > policy.rmsd_limit or atom_step > policy.atom_step_limit
                   else None)
    return q


def collision_reason(x, checks):
    for row in checks:
        a, b = row["atoms"]
        if np.linalg.norm(x[a] - x[b]) < row["minimum_distance"]:
            return "NONBONDED_COLLISION"
    return None


def run_continuation(plan, initial_result, corrector: Callable, save: Callable | None = None):
    """Run/replay a deterministic sequence; cached backend calls prevent duplication.

    initial_result is a separately audited origin preparation at this method.
    corrector returns success, converged, coordinates, energy, duration and evidence.
    save receives the whole checkpoint after every attempted node.
    """
    validate_plan(plan)
    if plan.get("content_sha256") != digest({k: v for k, v in plan.items() if k != "content_sha256"}):
        raise ValueError("FROZEN_PLAN_HASH_MISMATCH")
    policy = ContinuationPolicy(**plan["policy"])
    coordinates = plan["drivers"] + plan.get("guards", [])
    x = np.asarray(initial_result["coordinates"], float)
    initial_energy = initial_result.get("energy")
    if not initial_result.get("success") or not initial_result.get("converged") or initial_energy is None or not math.isfinite(initial_energy):
        raise ValueError("INVALID_INITIAL_RESULT")
    if x.shape != (len(plan["elements"]), 3) or not np.isfinite(x).all():
        raise ValueError("INVALID_INITIAL_GEOMETRY")
    if collision_reason(x, plan.get("collision_checks", [])):
        raise ValueError("INVALID_INITIAL_COLLISION")
    q0 = geometry_quality(x, x, coordinates, target_values(plan, 0), policy)
    if q0["reason"]:
        raise ValueError("INITIAL_TARGET_MISMATCH")
    frames = [{"frame_id": "accepted-0000", "lambda": 0., "geometry": x.tolist(),
               "energy_hartree": initial_energy, "quality": q0, "parent_frame_id": None,
               "frame_role": "prepared_origin", "targets": target_values(plan, 0)}]
    attempts = []
    lam = 0.
    step = policy.initial_step
    smooth = 0
    elapsed = float(initial_result.get("duration_seconds", 0))
    terminal = "BUDGET_EXHAUSTED"

    def snapshot():
        n_bound = sum(frame.get("physical_gradient_status") == "bound" for frame in frames)
        return {"schema_version": "pes2ts_continuation_result_v1", "reaction_id": plan["reaction_id"],
                "plan_sha256": plan["content_sha256"], "frames": frames, "attempts": attempts,
                "status": terminal, "last_lambda": lam, "next_step": step,
                "duration_seconds": elapsed, "completed_interval": lam >= 1.-1e-12,
                "physical_gradient_status": ("bound" if n_bound == len(frames) else "partially_bound" if n_bound else "not_collected"),
                "n_bound_frame_gradients": n_bound, "stationary_point_claimed": False}

    while lam < 1.-1e-12:
        if len(frames) >= policy.max_frames or len(attempts) >= policy.max_attempts or elapsed >= policy.max_seconds:
            terminal = "BUDGET_EXHAUSTED"
            break
        next_lam = min(1., lam + step)
        actual_step = next_lam - lam
        targets = target_values(plan, next_lam)
        tangent = None
        if len(frames) > 1:
            previous = frames[-2]
            tangent = (x - align(previous["geometry"], x)) * actual_step / (lam - previous["lambda"])
        if policy.predictor_enabled:
            guess, projection = project_geometry(x, coordinates, targets, plan["masses"], policy, tangent)
        else:
            guess, projection = x.copy(), {"reason": None, "mode": "previous_frame_control"}
        attempt = {"attempt_id": f"attempt-{len(attempts):04d}", "parent_frame_id": frames[-1]["frame_id"],
                   "lambda": next_lam, "actual_step": actual_step, "projection": projection,
                   "accepted": False, "targets": targets}
        reason = projection["reason"]
        if guess is not None:
            result = corrector(guess, targets, attempt["attempt_id"])
            elapsed += float(result.get("duration_seconds", 0))
            attempt["backend_evidence"] = {k: v for k, v in result.items() if k not in {"coordinates", "optimizer_state"}}
            energy = result.get("energy")
            output = result.get("coordinates")
            if not result.get("success") or not result.get("converged"):
                reason = result.get("failure_class") or "OPT_FAILED"
            elif energy is None or not math.isfinite(energy):
                reason = "NONFINITE_ENERGY"
            elif output is None or np.asarray(output).shape != x.shape or not np.isfinite(output).all():
                reason = "INVALID_BACKEND_GEOMETRY"
            else:
                output = np.asarray(output, float)
                quality = geometry_quality(x, output, coordinates, targets, policy, frames[-1]["targets"])
                quality["corrector_rmsd_angstrom"] = float(np.sqrt(np.mean(np.sum((align(output, guess)-guess)**2, axis=1))))
                attempt["quality"] = quality
                from pes2ts_core.generation.planning.structure_checks import structure_issues
                identity_issues = structure_issues(output, plan)
                quality["structure_issues"] = identity_issues
                reason = (quality["reason"] or collision_reason(output, plan.get("collision_checks", []))
                          or (identity_issues[0]["reason"] if identity_issues else None))
                gradient = result.get("physical_gradient_hartree_per_angstrom")
                if gradient is None and result.get("physical_gradient") is not None:
                    from pes2ts_core.generation.planning.gradient_evidence import BOHR_ANGSTROM
                    gradient = np.asarray(result["physical_gradient"])/BOHR_ANGSTROM
                if gradient is not None and result.get("physical_gradient_status") == "bound":
                    from pes2ts_core.generation.planning.local_corrector import tangent_projector, gradient_quality
                    gradient = np.asarray(gradient, float)
                    if gradient.shape != output.shape or not np.isfinite(gradient).all():
                        reason = reason or "INVALID_PHYSICAL_GRADIENT"
                    else:
                        try:
                            p, rank_evidence = tangent_projector(output, coordinates)
                            _, force_quality = gradient_quality(gradient, p, output.shape)
                            quality.update(force_quality)
                            quality.update(rank_evidence)
                            if policy.require_physical_gradient and (
                                force_quality["free_gradient_rms_hartree_per_angstrom"] > policy.free_gradient_rms_tolerance
                                or force_quality["free_gradient_max_hartree_per_angstrom"] > policy.free_gradient_max_tolerance):
                                reason = reason or "FREE_GRADIENT_NOT_CONVERGED"
                        except (ValueError, np.linalg.LinAlgError):
                            if policy.require_physical_gradient:
                                reason = reason or "COORDINATE_DEGENERATE"
                elif policy.require_physical_gradient:
                    reason = reason or "PHYSICAL_GRADIENT_MISSING"
                if reason is None:
                    attempt["accepted"] = True
                    if hasattr(corrector, "on_accept"):
                        corrector.on_accept(result)
                    frames.append({"frame_id": f"accepted-{len(frames):04d}", "lambda": next_lam,
                                   "geometry": output.tolist(), "energy_hartree": energy,
                                   "quality": quality, "parent_frame_id": frames[-1]["frame_id"],
                                   "attempt_id": attempt["attempt_id"], "targets": targets,
                                   "physical_gradient_status": result.get("physical_gradient_status", "not_collected"),
                                   "physical_gradient_hartree_per_angstrom": np.asarray(gradient).tolist() if gradient is not None else None,
                                   "frame_role": "constrained_optimization"})
                    x = output
                    lam = next_lam
                    smooth = smooth + 1 if quality["rmsd_angstrom"] < policy.smooth_rmsd and quality["corrector_rmsd_angstrom"] < policy.smooth_rmsd else 0
                    if smooth >= 2 and policy.adaptive_enabled:
                        step = min(step * 1.5, policy.max_step)
                        smooth = 0
        attempt["reason"] = reason
        attempts.append(attempt)
        if reason is not None:
            smooth = 0
            if not policy.adaptive_enabled:
                terminal = "FIXED_STEP_FAILED:" + reason
                if save:
                    save(snapshot())
                break
            if reason in {"BACKEND_UNAVAILABLE", "INVALID_BACKEND_GEOMETRY", "NONFINITE_ENERGY"}:
                terminal = reason
                if save:
                    save(snapshot())
                break
            reduced = actual_step / 2
            if reduced < policy.min_step - 1e-14:
                attempt["minimum_step_attempted"] = actual_step <= policy.min_step+1e-14
                attempt["next_half_step"] = reduced
                attempt["configured_minimum_step"] = policy.min_step
                terminal = "STEP_LIMIT:" + reason
                if save:
                    save(snapshot())
                break
            step = reduced
        terminal = "completed" if lam >= 1.-1e-12 else "running"
        if save:
            save(snapshot())
    result = snapshot()
    if save:
        save(result)
    return result
