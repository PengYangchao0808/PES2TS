"""Outcome normalization for ACP ``XtbConstrainedOpt`` results (G2-AB3 WP-4).

Separates *calculation status* (ACP's job: did xTB run, converge, and
produce evidence?) from *chemical acceptance* (the controller's job).  This
layer:

* loads the evidence product of a completed attempt and verifies identity
  (request digest, atom-order digest, geometry rows) so a stale attempt can
  never be mistaken for this point's result;
* reads the optimized geometry from the atom-ordered ``optimized.xyz``;
* measures the ACTUAL driver coordinate values on the final geometry and
  records per-driver residuals against the targets — xTB restraints are
  finite-strength potentials, so the actual distance may deviate from the
  requested one and must be checked, never assumed;
* splits energy channels: ``E_effective`` (restraint-including TOTAL ENERGY
  from the optimization) vs ``E_physical`` (unrestrained single point at the
  same geometry).  ``E_effective`` is never reported as the physical energy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pes2ts_core.integration.acp.xtb_constraints_request import (
    atom_order_digest,
    request_sha256,
)

EVIDENCE_SCHEMA = "pes2ts_xtb_constrained_opt_evidence_v1"

OUTCOME_COMPUTED = "computed"
OUTCOME_NOT_CONVERGED = "rejected_not_converged"
OUTCOME_SCC_FAILURE = "rejected_scc_failure"
OUTCOME_NO_GEOMETRY = "rejected_no_geometry"
OUTCOME_BACKEND_FAILED = "rejected_backend_failed"
OUTCOME_STALE_IDENTITY = "rejected_stale_identity"
OUTCOME_ABNORMAL_TERMINATION = "rejected_abnormal_termination"


class NativeConstraintResultError(ValueError):
    """Raised when an attempt's evidence cannot be normalized."""


@dataclass(frozen=True)
class DriverMeasurement:
    """One driver's target vs actual value on the final geometry."""

    driver_id: str
    kind: str
    atoms: tuple[int, ...]
    unit: str
    target: float
    actual: float | None
    residual: float | None


@dataclass(frozen=True)
class NativeConstraintOutcome:
    """Normalized outcome of one native constrained-optimization attempt."""

    status: str
    attempt_dir: str
    request_sha256: str
    geometry: tuple[tuple[float, float, float], ...] | None
    measurements: tuple[DriverMeasurement, ...]
    energy_effective_hartree: float | None
    energy_physical_hartree: float | None
    converged_geometry: bool
    normal_termination: bool
    scc_failure_detected: bool
    n_optimization_cycles_logged: int | None
    wall_seconds: float
    evidence: dict[str, Any] = field(default_factory=dict)
    failure_reason: str | None = None


def _parse_xyz(path: Path) -> tuple[tuple[str, ...], np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise NativeConstraintResultError(f"optimized xyz is truncated: {path}")
    try:
        n_atoms = int(lines[0].split()[0])
    except (ValueError, IndexError) as exc:
        raise NativeConstraintResultError(f"optimized xyz header is malformed: {path}") from exc
    body = [line for line in lines[2:] if line.strip()]
    if len(body) != n_atoms:
        raise NativeConstraintResultError(
            f"optimized xyz declares {n_atoms} atoms but has {len(body)} rows: {path}"
        )
    symbols: list[str] = []
    rows: list[list[float]] = []
    for line in body:
        parts = line.split()
        if len(parts) < 4:
            raise NativeConstraintResultError(f"optimized xyz row is malformed: {line!r}")
        symbols.append(parts[0])
        try:
            rows.append([float(parts[1]), float(parts[2]), float(parts[3])])
        except ValueError as exc:
            raise NativeConstraintResultError(
                f"optimized xyz coordinate is non-numeric: {line!r}"
            ) from exc
    for row in rows:
        for value in row:
            if not math.isfinite(value):
                raise NativeConstraintResultError("optimized geometry has non-finite coordinates")
    return tuple(symbols), np.asarray(rows, dtype=float)


def distance_value(geometry: np.ndarray, atoms: Sequence[int]) -> float:
    return float(np.linalg.norm(geometry[atoms[0]] - geometry[atoms[1]]))


def angle_value(geometry: np.ndarray, atoms: Sequence[int]) -> float:
    b1 = geometry[atoms[1]] - geometry[atoms[0]]
    b2 = geometry[atoms[2]] - geometry[atoms[1]]
    cos = float(np.dot(b1, b2) / (np.linalg.norm(b1) * np.linalg.norm(b2)))
    return float(math.degrees(math.acos(max(-1.0, min(1.0, cos)))))


def dihedral_value(geometry: np.ndarray, atoms: Sequence[int]) -> float:
    p0, p1, p2, p3 = (geometry[i] for i in atoms[:4])
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1n, v), w))
    return float(math.degrees(math.atan2(y, x)))


_COORDINATE_FUNCTIONS = {
    "distance": distance_value,
    "angle": angle_value,
    "dihedral": dihedral_value,
}


def measure_coordinate(kind: str, geometry: np.ndarray, atoms: Sequence[int]) -> float:
    try:
        function = _COORDINATE_FUNCTIONS[kind]
    except KeyError as exc:
        raise NativeConstraintResultError(f"unsupported coordinate kind {kind!r}") from exc
    return function(geometry, atoms)


def normalize_outcome(
    *,
    attempt_result: Any,
    request: dict[str, Any],
    driver_ids: Sequence[str] | None = None,
) -> NativeConstraintOutcome:
    """Normalize one attempt result against its request.

    Args:
        attempt_result: ``XtbConstrainedOptAttemptResult`` from the transport
            (``status`` there means "the calculation ran"; acceptance is
            decided here).
        request: The frozen request that produced this attempt.  The evidence
            must bind to this request's digest, atom order, and geometry size;
            mismatches yield ``rejected_stale_identity``.
        driver_ids: Optional per-constraint driver labels (parallel to
            ``request.recipe.constraints``).

    Raises:
        NativeConstraintResultError: when the transport status is completed
            but the evidence/geometry cannot even be read (contract break).
    """
    if attempt_result.status != "completed":
        return NativeConstraintOutcome(
            status=OUTCOME_BACKEND_FAILED,
            attempt_dir=str(attempt_result.attempt_dir),
            request_sha256=request_sha256(request),
            geometry=None,
            measurements=(),
            energy_effective_hartree=None,
            energy_physical_hartree=None,
            converged_geometry=False,
            normal_termination=False,
            scc_failure_detected=False,
            n_optimization_cycles_logged=None,
            wall_seconds=float(attempt_result.wall_seconds or 0.0),
            failure_reason=attempt_result.error or "backend attempt did not complete",
        )

    evidence_path = Path(attempt_result.evidence_path or "")
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeConstraintResultError(
            f"completed attempt has unreadable evidence: {exc}"
        ) from exc
    if evidence.get("schema_version") != EVIDENCE_SCHEMA:
        raise NativeConstraintResultError("evidence schema mismatch")

    expected_request_sha = request_sha256(request)
    evidence_request_sha = evidence.get("request_sha256")
    elements = request["source"]["elements"]
    expected_atom_order = atom_order_digest(elements, len(request["source"]["geometry"]))
    if evidence_request_sha is not None and evidence_request_sha != expected_request_sha:
        return NativeConstraintOutcome(
            status=OUTCOME_STALE_IDENTITY,
            attempt_dir=str(attempt_result.attempt_dir),
            request_sha256=expected_request_sha,
            geometry=None, measurements=(),
            energy_effective_hartree=None, energy_physical_hartree=None,
            converged_geometry=False, normal_termination=False,
            scc_failure_detected=False, n_optimization_cycles_logged=None,
            wall_seconds=float(attempt_result.wall_seconds or 0.0),
            evidence=evidence,
            failure_reason=(
                f"evidence binds request {evidence_request_sha} but this point is {expected_request_sha}"
            ),
        )
    if evidence.get("input_geometry_rows") not in (None, len(elements)):
        return NativeConstraintOutcome(
            status=OUTCOME_STALE_IDENTITY,
            attempt_dir=str(attempt_result.attempt_dir),
            request_sha256=expected_request_sha,
            geometry=None, measurements=(),
            energy_effective_hartree=None, energy_physical_hartree=None,
            converged_geometry=False, normal_termination=False,
            scc_failure_detected=False, n_optimization_cycles_logged=None,
            wall_seconds=float(attempt_result.wall_seconds or 0.0),
            evidence=evidence,
            failure_reason="evidence geometry row count does not match this request",
        )

    opt = evidence.get("opt") or {}
    sp = evidence.get("physical_sp") or {}
    converged = bool(opt.get("converged_geometry"))
    normal = bool(opt.get("normal_termination"))
    scc_failure = bool(opt.get("scc_failure_detected"))
    energy_effective = opt.get("energy_effective_hartree")
    energy_physical = sp.get("energy_physical_hartree") if isinstance(sp, dict) else None

    geometry = None
    measurements: tuple[DriverMeasurement, ...] = ()
    geometry_ok = False
    stale_reason: str | None = None
    geometry_path = attempt_result.optimized_geometry_path
    if geometry_path:
        try:
            symbols, coords = _parse_xyz(Path(geometry_path))
        except NativeConstraintResultError as exc:
            stale_reason = str(exc)
        else:
            if tuple(elements) != symbols:
                stale_reason = "optimized geometry element order differs from the request"
            else:
                geometry = tuple(tuple(float(v) for v in row) for row in coords)
                geometry_ok = True
                constraints = request["recipe"]["constraints"]
                measured: list[DriverMeasurement] = []
                coords_array = np.asarray(geometry, dtype=float)
                for index, constraint in enumerate(constraints):
                    label = (
                        driver_ids[index]
                        if driver_ids and index < len(driver_ids)
                        else f"constraint_{index}"
                    )
                    actual = measure_coordinate(constraint["kind"], coords_array, constraint["atoms"])
                    measured.append(
                        DriverMeasurement(
                            driver_id=label,
                            kind=constraint["kind"],
                            atoms=tuple(constraint["atoms"]),
                            unit=constraint["unit"],
                            target=float(constraint["target"]),
                            actual=actual,
                            residual=actual - float(constraint["target"]),
                        )
                    )
                measurements = tuple(measured)
    if stale_reason is not None:
        return NativeConstraintOutcome(
            status=OUTCOME_STALE_IDENTITY,
            attempt_dir=str(attempt_result.attempt_dir),
            request_sha256=expected_request_sha,
            geometry=None, measurements=(),
            energy_effective_hartree=None, energy_physical_hartree=None,
            converged_geometry=converged, normal_termination=normal,
            scc_failure_detected=scc_failure,
            n_optimization_cycles_logged=opt.get("n_optimization_cycles_logged"),
            wall_seconds=float(opt.get("wall_seconds") or attempt_result.wall_seconds or 0.0),
            evidence=evidence,
            failure_reason=stale_reason,
        )

    if not geometry_ok:
        status = OUTCOME_NO_GEOMETRY
        reason = "no usable optimized geometry in the completed attempt"
    elif not normal:
        status = OUTCOME_ABNORMAL_TERMINATION
        reason = "xTB did not terminate normally"
    elif scc_failure:
        status = OUTCOME_SCC_FAILURE
        reason = "SCC failure detected in the optimization log"
    elif not converged:
        status = OUTCOME_NOT_CONVERGED
        marker = opt.get("convergence_marker")
        reason = f"geometry optimization did not converge ({marker})"
    else:
        status = OUTCOME_COMPUTED
        reason = None

    return NativeConstraintOutcome(
        status=status,
        attempt_dir=str(attempt_result.attempt_dir),
        request_sha256=expected_request_sha,
        geometry=geometry,
        measurements=measurements,
        energy_effective_hartree=(float(energy_effective)
                                  if isinstance(energy_effective, (int, float)) else None),
        energy_physical_hartree=(float(energy_physical)
                                 if isinstance(energy_physical, (int, float)) else None),
        converged_geometry=converged,
        normal_termination=normal,
        scc_failure_detected=scc_failure,
        n_optimization_cycles_logged=opt.get("n_optimization_cycles_logged"),
        wall_seconds=float(opt.get("wall_seconds") or attempt_result.wall_seconds or 0.0),
        evidence=evidence,
        failure_reason=reason,
    )


__all__ = [
    "EVIDENCE_SCHEMA",
    "NativeConstraintOutcome",
    "NativeConstraintResultError",
    "OUTCOME_ABNORMAL_TERMINATION",
    "OUTCOME_BACKEND_FAILED",
    "OUTCOME_COMPUTED",
    "OUTCOME_NOT_CONVERGED",
    "OUTCOME_NO_GEOMETRY",
    "OUTCOME_SCC_FAILURE",
    "OUTCOME_STALE_IDENTITY",
    "angle_value",
    "dihedral_value",
    "distance_value",
    "measure_coordinate",
    "normalize_outcome",
]
