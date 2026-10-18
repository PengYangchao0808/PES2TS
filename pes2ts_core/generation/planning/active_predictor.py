"""Key-atom-only geometric seed predictor (G2-AB3 WP-3).

Given the last ACCEPTED geometry ``X`` and the next scan-point targets, this
predictor produces a pure-geometry seed by moving ONLY the key (active) atoms
``K``:

.. code-block:: text

    min ||delta_X_K||^2   subject to   J_K delta_X_K = dq,   delta_X_notK = 0

where ``J_K`` is the driver Jacobian restricted to the Cartesian columns of
``K`` (for a distance driver ``d(a, b)`` the row is ``+u`` on ``a`` and
``-u`` on ``b`` with ``u`` the unit vector from ``a`` to ``b``) and
``dq = q_target - q(X)``.

Invariants enforced here (and asserted by tests):

* non-K atom coordinates are bit-for-bit unchanged — the seed writes only
  ``K`` rows and no rigid-body transform is ever applied;
* rank-deficient / collinear K-Jacobians, unreachable targets (displacement
  beyond the per-atom cap), or non-finite geometry return typed refusals —
  the predictor never shrinks the driver set and never touches other atoms
  to "make it work";
* zero energy/gradient calls: this is geometry only, no QM, no BFGS.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

SEED_OK = "predicted"
SEED_RANK_DEFICIENT = "refused_rank_deficient"
SEED_TARGET_UNREACHABLE = "refused_target_unreached"
SEED_NONFINITE = "refused_nonfinite_geometry"
SEED_AT_TARGET = "at_target"


class SeedPredictionError(ValueError):
    """Raised on misuse (shape mismatches), never for typed refusals."""


@dataclass(frozen=True)
class SeedPrediction:
    """Typed outcome of one key-atom seed prediction."""

    status: str
    geometry: tuple[tuple[float, float, float], ...] | None
    active_atom_indices: tuple[int, ...]
    delta_active: tuple[tuple[float, float, float], ...] | None
    target_values: tuple[float, ...]
    achieved_values: tuple[float, ...] | None
    residuals: tuple[float, ...] | None
    max_atom_displacement: float | None
    jacobian_rank: int | None
    singular_values: tuple[float, ...] | None
    reason: str | None = None


def _distance_jacobian_row(geometry: np.ndarray, atoms: Sequence[int]) -> np.ndarray:
    u = geometry[atoms[0]] - geometry[atoms[1]]
    norm = float(np.linalg.norm(u))
    if norm < 1e-9:
        raise SeedPredictionError(
            f"driver atoms {atoms} are coincident; distance Jacobian is undefined"
        )
    u /= norm
    row = np.zeros((1, geometry.shape[0], 3), dtype=float)
    row[0, atoms[0]] = u
    row[0, atoms[1]] = -u
    return row.reshape(1, -1)[0]


def _angle_jacobian_row(geometry: np.ndarray, atoms: Sequence[int]) -> np.ndarray:
    i, j, k = atoms
    b1 = geometry[i] - geometry[j]
    b2 = geometry[k] - geometry[j]
    r1 = float(np.linalg.norm(b1))
    r2 = float(np.linalg.norm(b2))
    if r1 < 1e-9 or r2 < 1e-9:
        raise SeedPredictionError("angle Jacobian undefined for coincident atoms")
    e1, e2 = b1 / r1, b2 / r2
    cos = float(np.clip(np.dot(e1, e2), -1.0, 1.0))
    sin = math.sqrt(max(0.0, 1.0 - cos * cos))
    if sin < 1e-8:
        raise SeedPredictionError("angle Jacobian degenerate for collinear atoms")
    grad = np.zeros((geometry.shape[0], 3), dtype=float)
    dcos_du = (e2 - cos * e1) / r1
    grad[i] = dcos_du
    grad[j] = -(dcos_du + (e1 - cos * e2) / r2)
    grad[k] = (e1 - cos * e2) / r2
    dtheta_dcos = -1.0 / math.sin(math.acos(cos))
    return (grad * dtheta_dcos).reshape(-1)


def _driver_jacobian_row(geometry: np.ndarray, kind: str, atoms: Sequence[int]) -> np.ndarray:
    if kind == "distance":
        return _distance_jacobian_row(geometry, atoms)
    if kind == "angle":
        return _angle_jacobian_row(geometry, atoms)
    raise SeedPredictionError(
        f"seed predictor supports distance/angle drivers in v1; got {kind!r} "
        "(dihedral drivers need an explicit support-set registration first)"
    )


def predict_seed(
    geometry: Sequence[Sequence[float]],
    *,
    driver_kinds: Sequence[str],
    driver_atoms: Sequence[Sequence[int]],
    target_values: Sequence[float],
    active_atom_indices: Sequence[int],
    max_atom_displacement: float = 0.6,
    rank_tolerance: float = 1e-8,
) -> SeedPrediction:
    """Predict the next seed by moving only the active atoms.

    Args:
        geometry: Parent (last accepted) geometry, Å, frozen atom order.
        driver_kinds/driver_atoms/target_values: The scan point's driver
            definitions and targets (units implicit per kind).
        active_atom_indices: The key set ``K`` (0-based positions).
        max_atom_displacement: Per-atom seed displacement cap (Å).
        rank_tolerance: Smallest singular value accepted as rank-relevant.

    Returns:
        :class:`SeedPrediction` — never raises for geometric refusals.

    Raises:
        SeedPredictionError: shape/index misuse only.
    """
    n_atoms = len(geometry)
    if len(driver_kinds) != len(driver_atoms) or len(driver_kinds) != len(target_values):
        raise SeedPredictionError("driver kind/atom/target lists must be parallel")
    active = sorted({int(i) for i in active_atom_indices})
    for index in active:
        if not 0 <= index < n_atoms:
            raise SeedPredictionError(f"active index {index} out of range")
    for atoms in driver_atoms:
        for index in atoms:
            if not 0 <= index < n_atoms:
                raise SeedPredictionError(f"driver atom {index} out of range")
            if index not in active:
                raise SeedPredictionError(
                    f"driver atom {index} is outside the active set K; the seed "
                    "predictor may only move key atoms"
                )

    x = np.asarray(geometry, dtype=float)
    if not np.isfinite(x).all():
        return SeedPrediction(
            status=SEED_NONFINITE, geometry=None, active_atom_indices=tuple(active),
            delta_active=None, target_values=tuple(float(t) for t in target_values),
            achieved_values=None, residuals=None, max_atom_displacement=None,
            jacobian_rank=None, singular_values=None,
            reason="parent geometry contains non-finite coordinates",
        )

    rows: list[np.ndarray] = []
    for kind, atoms in zip(driver_kinds, driver_atoms):
        rows.append(_driver_jacobian_row(x, kind, tuple(atoms)))
    jac_full = np.vstack(rows)
    column_indices = [atom * 3 + axis for atom in active for axis in range(3)]
    jac_k = jac_full[:, column_indices]

    achieved = np.asarray(
        [float(np.linalg.norm(x[atoms[0]] - x[atoms[1]]))
         if kind == "distance" else float("nan")
         for kind, atoms in zip(driver_kinds, driver_atoms)],
        dtype=float,
    )
    targets = np.asarray(target_values, dtype=float)
    if np.isfinite(achieved).all():
        residual_now = achieved - targets
        if float(np.abs(residual_now).max()) <= 1e-9:
            return SeedPrediction(
                status=SEED_AT_TARGET, geometry=tuple(tuple(float(v) for v in row) for row in x),
                active_atom_indices=tuple(active),
                delta_active=tuple((0.0, 0.0, 0.0) for _ in active),
                target_values=tuple(float(t) for t in target_values),
                achieved_values=tuple(float(v) for v in achieved),
                residuals=tuple(float(v) for v in residual_now),
                max_atom_displacement=0.0, jacobian_rank=int(np.linalg.matrix_rank(jac_k)),
                singular_values=None, reason=None,
            )
    # The linear system solves for the DISPLACEMENT dq = q_target - q(X), not
    # for the targets themselves.
    dq = np.where(np.isfinite(achieved), targets - achieved, targets)

    u_matrix, s_values, vt_matrix = np.linalg.svd(jac_k, full_matrices=False)
    rank = int(np.sum(s_values > rank_tolerance * max(1.0, float(s_values.max()) if s_values.size else 1.0)))
    if rank < len(driver_kinds):
        return SeedPrediction(
            status=SEED_RANK_DEFICIENT, geometry=None, active_atom_indices=tuple(active),
            delta_active=None, target_values=tuple(float(t) for t in target_values),
            achieved_values=None, residuals=None, max_atom_displacement=None,
            jacobian_rank=rank, singular_values=tuple(float(v) for v in s_values),
            reason=(
                f"K-restricted Jacobian rank {rank} < {len(driver_kinds)} drivers; "
                "the coupled targets are not linearly reachable by moving K alone"
            ),
        )

    # Minimum-norm least-squares solution of J_K dx = dq (scaled SVD pinverse).
    tolerance = rank_tolerance * max(1.0, float(s_values.max()))
    s_safe = np.where(s_values > tolerance, s_values, np.inf)
    dx_active = (vt_matrix.T * (1.0 / s_safe)) @ (u_matrix.T @ dq)
    if not np.isfinite(dx_active).all():
        return SeedPrediction(
            status=SEED_NONFINITE, geometry=None, active_atom_indices=tuple(active),
            delta_active=None, target_values=tuple(float(t) for t in target_values),
            achieved_values=None, residuals=None, max_atom_displacement=None,
            jacobian_rank=rank, singular_values=tuple(float(v) for v in s_values),
            reason="seed solve produced non-finite displacements",
        )
    max_displacement = float(np.abs(dx_active.reshape(-1, 3)).max()) if dx_active.size else 0.0
    if max_displacement > float(max_atom_displacement):
        return SeedPrediction(
            status=SEED_TARGET_UNREACHABLE, geometry=None, active_atom_indices=tuple(active),
            delta_active=None, target_values=tuple(float(t) for t in target_values),
            achieved_values=None, residuals=None, max_atom_displacement=max_displacement,
            jacobian_rank=rank, singular_values=tuple(float(v) for v in s_values),
            reason=(
                f"required key-atom displacement {max_displacement:.3f} Å exceeds the "
                f"per-atom cap {float(max_atom_displacement):.3f} Å; subdivide the step "
                "or refuse — never move non-key atoms to compensate"
            ),
        )

    seed = x.copy()
    for row_index, atom_index in enumerate(active):
        seed[atom_index] += dx_active.reshape(-1, 3)[row_index]

    achieved_seed = np.asarray(
        [float(np.linalg.norm(seed[atoms[0]] - seed[atoms[1]]))
         if kind == "distance" else float("nan")
         for kind, atoms in zip(driver_kinds, driver_atoms)],
        dtype=float,
    )
    residuals_seed = achieved_seed - targets
    achieved_ok = np.isfinite(achieved_seed)

    return SeedPrediction(
        status=SEED_OK,
        geometry=tuple(tuple(float(v) for v in row) for row in seed),
        active_atom_indices=tuple(active),
        delta_active=tuple(
            tuple(float(v) for v in (seed[atom] - x[atom])) for atom in active
        ),
        target_values=tuple(float(t) for t in target_values),
        achieved_values=tuple(float(v) for v in achieved_seed),
        residuals=tuple(float(v) if ok else None
                        for v, ok in zip(residuals_seed, achieved_ok)),
        max_atom_displacement=max_displacement,
        jacobian_rank=rank,
        singular_values=tuple(float(v) for v in s_values),
        reason=None,
    )


__all__ = [
    "SeedPrediction",
    "SeedPredictionError",
    "SEED_AT_TARGET",
    "SEED_NONFINITE",
    "SEED_OK",
    "SEED_RANK_DEFICIENT",
    "SEED_TARGET_UNREACHABLE",
    "predict_seed",
]
