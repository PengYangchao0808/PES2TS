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
from pes2ts_core.generation.planning.synchronized_path import align, digest, residual, value

#: Execution-identity scheme (G2-AB1 WP-1): every physical evaluation is
#: addressed as ``trial-<NNNN>/eval-<content16>`` — the trial counter is the
#: run-time namespace, the content digest is the immutable identity of
#: (geometry, constraints, targets, method context, operation).  Same content
#: under a replayed trial reuses the receipt; same key with different content
#: is a typed conflict and is never silently recomputed in place.
EVALUATION_IDENTITY_SCHEME = "pes2ts_execution_identity_v1"


def evaluation_content_digest(geometry, coordinates, targets, method_context=None,
                              operation="local_correction"):
    """Content identity of one physical evaluation (content-addressed).

    The digest binds the geometry, the mathematical constraint set
    (kind + atoms), the targets, the working method context and the operation
    name.  Execution identity (trial counter) is deliberately NOT part of the
    content digest.
    """
    canonical = {"operation": str(operation),
                 "geometry": np.asarray(geometry, float).tolist(),
                 "coordinates": [{"kind": d["kind"], "atoms": [int(i) for i in d["atoms"]]}
                                 for d in coordinates],
                 "targets": [float(t) for t in targets],
                 "method_context": dict(method_context or {})}
    return digest(canonical)[:16]


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


@dataclass(frozen=True)
class BranchPolicy:
    """Branch-stability knobs (G2-AB1 WP-2).

    Mathematical standing: within a neighbourhood where A has full row rank and
    the reduced Hessian is positive definite, the KKT solution branch is
    locally unique and smooth; the branch is NOT guaranteed to extend to
    lambda=1 nor to belong to the target chemical channel.  The numeric anchor
    below only suppresses non-essential local drift inside a trust boundary —
    it never adds a penalty to the electronic energy and it cannot lock
    slippage, eliminate bifurcation, or guarantee the target TS.  Acceptance is
    always re-checked with the raw physical free gradient after the anchor is
    released; a frame stationary only under the anchor is reported
    ``biased_only``, never ``physical``.
    """

    branch_rmsd_radius: float = .25      # frozen via config g2.continuation.branch_policy
    branch_atom_radius: float = .50      # frozen via config g2.continuation.branch_policy
    cumulative_rmsd_radius: float = .40  # frozen via config g2.continuation.branch_policy
    bias_kappa: float = .05              # frozen via config g2.continuation.branch_policy
    release_iterations: int = 6          # frozen via config g2.continuation.branch_policy

    def __post_init__(self):
        for name, v in asdict(self).items():
            if isinstance(v, bool) or not math.isfinite(v) or v <= 0:
                raise ValueError(f"INVALID_BRANCH_POLICY:{name}")
        if not isinstance(self.release_iterations, int):
            raise ValueError("INVALID_BRANCH_POLICY:release_iterations")


#: Locality failure taxonomy (G2-AB1 WP-2 split of the old LOCALITY_LIMIT).
#: Every code keeps the ``LOCALITY`` prefix so upper-layer prefix matching
#: keeps working; new codes only refine the failure shape.
LOCALITY_CODES = ("LOCALITY_STEP", "LOCALITY_ATOM", "LOCALITY_BRANCH",
                  "LOCALITY_CUMULATIVE", "LOCALITY_RANK")


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
                  geometry_check=None, save=None, initial_inverse_curvature=None,
                  method_context=None, operation="local_correction",
                  branch_reference=None, branch_policy=None):
    """Projected quasi-Newton correction with retraction and Armijo search.

    evaluate(X, evaluation_id) returns physical energy and a geometry-bound
    gradient_hartree_per_angstrom. A QC failure never yields an accepted frame.
    The inverse-curvature update is an optimizer approximation, NOT a physical
    Hessian or a certification that the constrained stationary point is a minimum.
    evaluation_id is content-addressed (trial-<NNNN>/eval-<content16>): replays
    with identical geometry/constraints/targets/method reuse receipts, changed
    content recomputes under a fresh address instead of colliding.

    G2-AB1 WP-2: when ``branch_reference`` (the previous accepted frame) is
    supplied, every trial is checked against the predictor neighbourhood AND
    the branch radii around the frozen reference ``T_*`` AND the cumulative
    displacement gate.  A releasable numeric anchor biases the optimizer
    direction only — the electronic energy is never penalized — and acceptance
    is confirmed with the raw physical free gradient; anchor-only stationarity
    is returned as ``convergence_mode="biased_only"`` with released evidence,
    never as a physical stationarity claim.
    """
    policy = policy or LocalCorrectorPolicy()
    branch_policy = branch_policy or BranchPolicy()
    reference = np.asarray(guess, float).copy()
    if reference.ndim != 2 or reference.shape[1] != 3 or not np.isfinite(reference).all():
        raise ValueError("INVALID_LOCAL_GEOMETRY")
    if len(targets) != len(coordinates) or not coordinates:
        raise ValueError("INVALID_LOCAL_TARGETS")
    if branch_reference is not None:
        branch_reference = np.asarray(branch_reference, float)
        if branch_reference.shape != reference.shape or not np.isfinite(branch_reference).all():
            raise ValueError("INVALID_BRANCH_REFERENCE")
        branch_reference = align(branch_reference, reference)
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
    frozen_basis = None
    anchor_active = branch_reference is not None and branch_policy.bias_kappa > 0

    def locality(y):
        """Typed locality gates: predictor, branch reference, cumulative drift."""
        predictor = motion_evidence(y, reference)
        if predictor["maximum_atom_step_angstrom"] >= policy.atom_radius:
            return "LOCALITY_ATOM", predictor
        if predictor["rmsd_angstrom"] >= policy.rmsd_radius:
            return "LOCALITY_STEP", predictor
        if branch_reference is not None:
            branch = motion_evidence(y, branch_reference)
            if (branch["maximum_atom_step_angstrom"] >= branch_policy.branch_atom_radius
                    or branch["rmsd_angstrom"] >= branch_policy.branch_rmsd_radius):
                return "LOCALITY_BRANCH", {**predictor, "branch_motion": branch}
        cumulative = motion_evidence(y, correction_start)
        if cumulative["rmsd_angstrom"] >= branch_policy.cumulative_rmsd_radius:
            return "LOCALITY_CUMULATIVE", {**predictor, "cumulative_motion": cumulative}
        return None, predictor

    def finish(failure, convergence_mode=None, released=None):
        record = {"success": failure is None, "converged": failure is None,
                  "failure_class": failure, "coordinates": x.tolist() if x is not None else None,
                  "energy": energy, "duration_seconds": time.monotonic()-started,
                  "physical_gradient_status": "bound" if physical_gradient is not None else "not_collected",
                  "physical_gradient_hartree_per_angstrom": physical_gradient.tolist() if physical_gradient is not None else None,
                  "local_policy": asdict(policy), "branch_policy": asdict(branch_policy),
                  "locality_reference": reference.tolist(),
                  "branch_locality_reference": branch_reference.tolist() if branch_reference is not None else None,
                  "numeric_anchor_active": bool(anchor_active and failure is None),
                  "convergence_mode": convergence_mode,
                  "released_evidence": released,
                  "optimizer": "projected_bfgs_fixed_neighbourhood_v1", "local_history": history,
                  "evaluations": evaluations, "n_gradient_evaluations": n_evaluations,
                  "evaluation_identity_scheme": EVALUATION_IDENTITY_SCHEME,
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
        gate, motion = locality(y)
        if gate:
            return None, gate
        if geometry_check:
            failure = geometry_check(y)
            if failure:
                return None, failure
        trial_id = f"trial-{n_evaluations:04d}"
        content16 = evaluation_content_digest(y, coordinates, targets, method_context, operation)
        evaluation_id = f"{trial_id}/eval-{content16}"
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
    correction_start = x.copy()
    try:
        # Frozen tangent basis at the correction start: the anchor never sees a
        # projector recomputed at a biased iterate (pseudo-gradient avoidance).
        frozen_basis, frozen_rank = tangent_projector(x, coordinates)
    except (ValueError, np.linalg.LinAlgError):
        frozen_basis = None

    def anchor_gradient(y):
        if not anchor_active or frozen_basis is None:
            return np.zeros_like(y)
        return (branch_policy.bias_kappa*(frozen_basis@(y-branch_reference).ravel())).reshape(y.shape)

    def anchor_merit(y, physical_energy):
        """Optimizer-internal merit during the anchored phase only.

        M(x) = E(x) + kappa/2 * ||P0(x - T_*)||^2 has exactly the effective
        gradient g + kappa*P0(x - T_*) as its stationary condition (P0 is an
        idempotent symmetric projector).  The electronic energy itself never
        carries a restraint: the recorded ``energy`` stays the physical value
        and ``restraint_energy_added`` stays False; the anchor exists only to
        steer the optimizer, and acceptance is re-verified after release.
        """
        if frozen_basis is None or branch_reference is None:
            return physical_energy
        offset = (frozen_basis@(y-branch_reference).ravel())
        return physical_energy + .5*branch_policy.bias_kappa*float(offset@offset)

    def released_evidence(free, errors, anchor):
        return {"free_gradient_rms_hartree_per_angstrom":
                    float(np.sqrt(np.mean(free**2))),
                "free_gradient_max_hartree_per_angstrom": float(np.abs(free).max()),
                "constraint_residuals": list(errors),
                "anchor_gradient_norm_hartree_per_angstrom": float(np.linalg.norm(anchor))}

    def optimization_loop(max_iterations, release_mode):
        """One anchored or released phase.

        Returns a finished record (all terminal outcomes inside), the marker
        ``"anchored_stationary"`` with released evidence when the anchor holds
        the frame stationary, or ``None`` when the iteration budget ran out.
        """
        nonlocal x, energy, physical_gradient, h_inv, final_quality
        for iteration in range(max_iterations+1):
            try:
                p, rank = tangent_projector(x, coordinates)
            except (ValueError, np.linalg.LinAlgError) as exc:
                return finish("LOCALITY_RANK:"+str(exc))
            free, quality = gradient_quality(physical_gradient, p, x.shape)
            anchor = np.zeros_like(x) if release_mode else anchor_gradient(x)
            if release_mode or frozen_basis is None:
                effective = free+anchor
            else:
                # Anchored phase works entirely on the frozen tangent basis so
                # the merit, its gradient, the search direction, and the BFGS
                # curvature are one exactly consistent system; the raw
                # physical check above keeps the current projector.
                effective = ((frozen_basis@physical_gradient.ravel()).reshape(x.shape)
                             + anchor)
            effective_quality = {"effective_free_gradient_rms_hartree_per_angstrom":
                                     float(np.sqrt(np.mean(effective**2))),
                                 "effective_free_gradient_max_hartree_per_angstrom":
                                     float(np.abs(effective).max())}
            final_quality = {**quality, **effective_quality, **rank, **motion_evidence(x, reference)}
            errors = [abs(residual(value(x, d["kind"], d["atoms"]), t, d["kind"]))
                      for d, t in zip(coordinates, targets)]
            final_quality["constraint_residuals"] = errors
            history.append({"iteration": iteration, "energy_hartree": energy,
                            "phase": "released" if release_mode else "anchored", **final_quality})
            if save:
                save({"status": "running", "history": history, "n_gradient_evaluations": n_evaluations})
            residuals_ok = all(e <= (policy.constraint_tolerance if d["kind"] == "distance" else .001)
                               for e, d in zip(errors, coordinates))
            converged = (quality["free_gradient_rms_hartree_per_angstrom"] <= policy.gradient_rms_tolerance
                         and quality["free_gradient_max_hartree_per_angstrom"] <= policy.gradient_max_tolerance
                         and residuals_ok)
            if converged:
                return finish(None, convergence_mode="physical")
            anchored_converged = (effective_quality["effective_free_gradient_rms_hartree_per_angstrom"]
                                  <= policy.gradient_rms_tolerance
                                  and effective_quality["effective_free_gradient_max_hartree_per_angstrom"]
                                  <= policy.gradient_max_tolerance
                                  and residuals_ok)
            if not release_mode and anchor_active and anchored_converged:
                return "anchored_stationary", released_evidence(free, errors, anchor)
            if iteration == max_iterations:
                if release_mode:
                    # Release budget exhausted without physical stationarity:
                    # the frame is stable only under the anchor — report honestly.
                    return "release_unconverged", released_evidence(free, errors, anchor)
                return None
            descent = effective if not release_mode else free
            if release_mode or frozen_basis is None:
                direction = -(p@h_inv@descent.ravel()).reshape(x.shape)
            else:
                direction = -(frozen_basis@h_inv@descent.ravel()).reshape(x.shape)
            if float(np.sum(direction*descent)) >= -1e-16:
                h_inv = np.eye(x.size)*policy.initial_inverse_curvature
                direction = -descent*policy.initial_inverse_curvature
            largest = np.linalg.norm(direction, axis=1).max()
            if largest > policy.iteration_atom_step:
                direction *= policy.iteration_atom_step/largest
            accepted = None
            boundary_reason = None
            current_merit = (energy if release_mode
                             else anchor_merit(x, energy))
            for backtrack in range(policy.max_backtracks):
                alpha = .5**backtrack
                trial, evidence = project_geometry(x+alpha*direction, coordinates, targets, masses, projection_policy)
                if trial is None:
                    continue
                trial = align(trial, reference)
                gate, _ = locality(trial)
                if gate:
                    boundary_reason = boundary_reason or gate
                    continue
                trial_sample, failure = sample(trial)
                if failure:
                    if failure in LOCALITY_CODES or failure == "LOCAL_BUDGET_EXHAUSTED":
                        boundary_reason = boundary_reason or failure
                        if failure == "LOCAL_BUDGET_EXHAUSTED":
                            return finish(failure)
                        continue
                    return finish(failure)
                trial_energy, trial_gradient = trial_sample
                displacement = trial-x
                trial_merit = (trial_energy if release_mode
                               else anchor_merit(trial, trial_energy))
                armijo = 1e-4*float(np.sum(effective*displacement))
                # Absolute slack 1e-14 (not 1e-11): the anchored crawl region
                # needs acceptance of ~1e-12-scale merit decreases, which a
                # 1e-11 slack would block and turn into a false stall.
                if trial_merit <= current_merit+armijo+1e-14:
                    accepted = trial, trial_energy, trial_gradient
                    break
            if accepted is None:
                if release_mode:
                    return finish(None, convergence_mode="biased_only",
                                  released=released_evidence(free, errors, anchor))
                return finish(boundary_reason or "LOCAL_STATIONARITY_STALLED")
            trial, trial_energy, trial_gradient = accepted
            s = (trial-x).ravel()
            if release_mode:
                new_p, _ = tangent_projector(trial, coordinates)
                y = new_p@trial_gradient.ravel()-free.ravel()
            else:
                # Keep the BFGS curvature consistent with the anchored search
                # direction: both gradients carry the same frozen-basis anchor.
                y = ((frozen_basis@trial_gradient.ravel()).reshape(x.shape)
                     + anchor_gradient(trial)).ravel()-effective.ravel()
            sy = float(s@y)
            if sy > 1e-10 and sy > 1e-5*np.linalg.norm(s)*np.linalg.norm(y):
                v = np.eye(x.size)-np.outer(s, y)/sy
                h_inv = v@h_inv@v.T+np.outer(s, s)/sy
            x, energy, physical_gradient = trial, trial_energy, trial_gradient
        return None

    sampled, reason = sample(x)
    if reason:
        return finish(reason)
    energy, physical_gradient = sampled
    outcome = optimization_loop(policy.max_iterations, release_mode=False)
    if isinstance(outcome, tuple) and outcome[0] == "anchored_stationary":
        release = optimization_loop(branch_policy.release_iterations, release_mode=True)
        if isinstance(release, tuple) and release[0] == "release_unconverged":
            return finish(None, convergence_mode="biased_only", released=release[1])
        return release
    if outcome is None:
        return finish("LOCAL_ITERATION_LIMIT")
    return outcome
