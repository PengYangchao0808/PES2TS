"""Native constrained-opt outcome normalization tests (G2-AB3 WP-4).

Crafts fake attempt results + ``evidence.json`` / ``optimized.xyz`` products
under the real ACP layout (``RESULT/pes2ts/xtb_constopt/``) in ``tmp_path``
and locks ``normalize_outcome``: backend failure, converged+computed with
real measurements, not-converged, SCC failure, stale request identity, and
element-order mismatch.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from pes2ts_core.integration.acp.native_constraint_results import (
    EVIDENCE_SCHEMA,
    OUTCOME_ABNORMAL_TERMINATION,
    OUTCOME_BACKEND_FAILED,
    OUTCOME_COMPUTED,
    OUTCOME_NOT_CONVERGED,
    OUTCOME_NO_GEOMETRY,
    OUTCOME_SCC_FAILURE,
    OUTCOME_STALE_IDENTITY,
    NativeConstraintResultError,
    angle_value,
    dihedral_value,
    distance_value,
    measure_coordinate,
    normalize_outcome,
)
from pes2ts_core.integration.acp.xtb_constraints_request import (
    DriverTarget,
    build_constraint_request,
)

ELEMENTS = ["C", "H", "H"]
SEED_GEOMETRY = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.5, 0.0, 0.0]]


@dataclass
class FakeAttemptResult:
    """Mirror of ``XtbConstrainedOptAttemptResult`` fields used here."""

    status: str
    attempt_dir: str
    evidence_path: str | None
    optimized_geometry_path: str | None
    wall_seconds: float
    error: str | None = None
    request_sha256: str = ""
    returncode: int | None = None
    timed_out: bool = False
    log_ref: str = "WORK/pes2ts/acp_cli.log"


def make_request(*, target=2.0, elements=ELEMENTS, geometry=SEED_GEOMETRY) -> dict:
    return build_constraint_request(
        geometry=geometry,
        elements=elements,
        charge=0,
        multiplicity=1,
        targets=[DriverTarget("d0", "distance", (0, 1), target, "angstrom")],
        reaction_id="RXN_TEST",
        point_id="lam0.5",
    )


def format_xyz(elements, coords) -> str:
    lines = [str(len(elements)), "optimized geometry"]
    for symbol, row in zip(elements, coords):
        lines.append(f"{symbol} {row[0]:.6f} {row[1]:.6f} {row[2]:.6f}")
    return "\n".join(lines) + "\n"


def coords_with_distance(distance: float) -> list[list[float]]:
    return [[0.0, 0.0, 0.0], [float(distance), 0.0, 0.0], [float(distance) + 1.5, 0.0, 0.0]]


def write_attempt(
    tmp_path: Path,
    *,
    request: dict,
    coords: list[list[float]] | None = None,
    elements_override: list[str] | None = None,
    status: str = "completed",
    converged: bool = True,
    normal: bool = True,
    scc_failure: bool = False,
    request_sha_override: str | None = None,
    input_geometry_rows=None,
    energy_effective: float | None = -1.234,
    energy_physical: float | None = -1.2345,
    write_geometry: bool = True,
) -> FakeAttemptResult:
    attempt_dir = tmp_path / "attempt"
    result_root = attempt_dir / "RESULT" / "pes2ts" / "xtb_constopt"
    result_root.mkdir(parents=True, exist_ok=True)
    elements = elements_override or request["source"]["elements"]
    if coords is None:
        coords = coords_with_distance(2.03)
    evidence = {
        "schema_version": EVIDENCE_SCHEMA,
        "request_sha256": (
            request_sha_override
            if request_sha_override is not None
            else request["provenance"]["request_sha256"]
        ),
        "input_geometry_rows": (
            len(elements) if input_geometry_rows is None else input_geometry_rows
        ),
        "opt": {
            "converged_geometry": converged,
            "normal_termination": normal,
            "scc_failure_detected": scc_failure,
            "energy_effective_hartree": energy_effective,
            "n_optimization_cycles_logged": 7,
            "wall_seconds": 0.02,
        },
        "physical_sp": {"energy_physical_hartree": energy_physical},
    }
    evidence_path = result_root / "evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    xyz_path = result_root / "optimized.xyz"
    if write_geometry:
        xyz_path.write_text(format_xyz(elements, coords), encoding="utf-8")
        optimized_path: str | None = str(xyz_path)
    else:
        optimized_path = None
    return FakeAttemptResult(
        status=status,
        attempt_dir=str(attempt_dir),
        evidence_path=str(evidence_path),
        optimized_geometry_path=optimized_path,
        wall_seconds=0.02,
        error=None if status == "completed" else "ACP CLI exited with return code 1",
    )


def test_backend_failure_is_rejected_without_reading_evidence(tmp_path):
    request = make_request()
    attempt = write_attempt(tmp_path, request=request, status="failed")
    outcome = normalize_outcome(attempt_result=attempt, request=request, driver_ids=["d0"])
    assert outcome.status == OUTCOME_BACKEND_FAILED
    assert outcome.geometry is None
    assert outcome.measurements == ()
    assert outcome.converged_geometry is False
    # The transport's typed error is surfaced; absence falls back to the default.
    assert outcome.failure_reason == "ACP CLI exited with return code 1"
    attempt = write_attempt(tmp_path / "noerr", request=request, status="failed")
    attempt.error = None
    outcome = normalize_outcome(attempt_result=attempt, request=request, driver_ids=["d0"])
    assert outcome.status == OUTCOME_BACKEND_FAILED
    assert outcome.failure_reason == "backend attempt did not complete"
    assert outcome.request_sha256 == request["provenance"]["request_sha256"]


def test_converged_normal_attempt_computes_measurements(tmp_path):
    request = make_request(target=2.0)
    attempt = write_attempt(
        tmp_path, request=request, coords=coords_with_distance(2.03),
        energy_effective=-1.234, energy_physical=-1.2345,
    )
    outcome = normalize_outcome(attempt_result=attempt, request=request, driver_ids=["d0"])
    assert outcome.status == OUTCOME_COMPUTED
    assert outcome.failure_reason is None
    assert outcome.converged_geometry is True
    assert outcome.normal_termination is True
    assert outcome.scc_failure_detected is False
    assert outcome.n_optimization_cycles_logged == 7
    assert outcome.energy_effective_hartree == pytest.approx(-1.234)
    assert outcome.energy_physical_hartree == pytest.approx(-1.2345)
    # Actual distance measured from optimized.xyz; residual vs the target.
    (measurement,) = outcome.measurements
    assert measurement.driver_id == "d0"
    assert measurement.kind == "distance"
    assert measurement.atoms == (0, 1)
    assert measurement.unit == "angstrom"
    assert measurement.target == pytest.approx(2.0)
    assert measurement.actual == pytest.approx(2.03, abs=1e-6)
    assert measurement.residual == pytest.approx(0.03, abs=1e-6)
    assert len(outcome.geometry) == 3


def test_evidence_not_converged_is_rejected(tmp_path):
    request = make_request()
    attempt = write_attempt(tmp_path, request=request, converged=False)
    outcome = normalize_outcome(attempt_result=attempt, request=request, driver_ids=["d0"])
    assert outcome.status == OUTCOME_NOT_CONVERGED
    assert outcome.converged_geometry is False
    assert "did not converge" in (outcome.failure_reason or "")


def test_scc_failure_is_rejected(tmp_path):
    request = make_request()
    attempt = write_attempt(tmp_path, request=request, converged=False, scc_failure=True)
    outcome = normalize_outcome(attempt_result=attempt, request=request, driver_ids=["d0"])
    assert outcome.status == OUTCOME_SCC_FAILURE
    assert outcome.scc_failure_detected is True
    assert "SCC" in (outcome.failure_reason or "")


def test_request_sha_mismatch_is_stale_identity(tmp_path):
    request = make_request()
    attempt = write_attempt(
        tmp_path, request=request, request_sha_override="0" * 64,
    )
    outcome = normalize_outcome(attempt_result=attempt, request=request, driver_ids=["d0"])
    assert outcome.status == OUTCOME_STALE_IDENTITY
    assert outcome.geometry is None
    assert "binds request" in (outcome.failure_reason or "")


def test_element_order_mismatch_in_xyz_is_stale_identity(tmp_path):
    request = make_request()
    attempt = write_attempt(
        tmp_path, request=request, elements_override=["H", "C", "H"],
        coords=[[0.0, 0.0, 0.0], [2.03, 0.0, 0.0], [3.53, 0.0, 0.0]],
    )
    outcome = normalize_outcome(attempt_result=attempt, request=request, driver_ids=["d0"])
    assert outcome.status == OUTCOME_STALE_IDENTITY
    assert "element order" in (outcome.failure_reason or "")


def test_input_geometry_rows_mismatch_is_stale_identity(tmp_path):
    request = make_request()
    attempt = write_attempt(tmp_path, request=request, input_geometry_rows=99)
    outcome = normalize_outcome(attempt_result=attempt, request=request, driver_ids=["d0"])
    assert outcome.status == OUTCOME_STALE_IDENTITY
    assert "row count" in (outcome.failure_reason or "")


def test_abnormal_termination_and_missing_geometry_paths(tmp_path):
    request = make_request()
    attempt = write_attempt(tmp_path, request=request, normal=False)
    outcome = normalize_outcome(attempt_result=attempt, request=request, driver_ids=["d0"])
    assert outcome.status == OUTCOME_ABNORMAL_TERMINATION
    assert "terminate normally" in (outcome.failure_reason or "")

    attempt = write_attempt(tmp_path / "no_geo", request=request, write_geometry=False)
    outcome = normalize_outcome(attempt_result=attempt, request=request, driver_ids=["d0"])
    assert outcome.status == OUTCOME_NO_GEOMETRY


def test_driver_ids_are_parallel_to_constraints(tmp_path):
    request = make_request()
    request["recipe"]["constraints"] = [
        {"kind": "distance", "atoms": [0, 1], "target": 2.0, "unit": "angstrom"},
        {"kind": "distance", "atoms": [1, 2], "target": 1.5, "unit": "angstrom"},
    ]
    attempt = write_attempt(tmp_path, request=request, coords=coords_with_distance(2.0))
    outcome = normalize_outcome(attempt_result=attempt, request=request,
                                driver_ids=["first", "second"])
    assert [m.driver_id for m in outcome.measurements] == ["first", "second"]
    outcome = normalize_outcome(attempt_result=attempt, request=request, driver_ids=None)
    assert [m.driver_id for m in outcome.measurements] == ["constraint_0", "constraint_1"]


def test_completed_attempt_with_unreadable_evidence_raises(tmp_path):
    request = make_request()
    attempt = write_attempt(tmp_path, request=request)
    Path(attempt.evidence_path).write_text("{not json", encoding="utf-8")
    with pytest.raises(NativeConstraintResultError, match="unreadable evidence"):
        normalize_outcome(attempt_result=attempt, request=request, driver_ids=["d0"])


def test_evidence_schema_mismatch_raises(tmp_path):
    request = make_request()
    attempt = write_attempt(tmp_path, request=request)
    evidence = json.loads(Path(attempt.evidence_path).read_text(encoding="utf-8"))
    evidence["schema_version"] = "pes2ts_other_evidence_v1"
    Path(attempt.evidence_path).write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(NativeConstraintResultError, match="schema mismatch"):
        normalize_outcome(attempt_result=attempt, request=request, driver_ids=["d0"])


def test_measure_coordinate_kinds():
    geometry = np.asarray(coords_with_distance(1.0), dtype=float)
    assert distance_value(geometry, [0, 1]) == pytest.approx(1.0)
    bent = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=float
    )
    assert angle_value(bent, [0, 1, 2]) == pytest.approx(90.0, abs=1e-9)
    assert measure_coordinate("distance", geometry, [0, 1]) == pytest.approx(1.0)
    assert measure_coordinate("angle", bent, [0, 1, 2]) == pytest.approx(90.0, abs=1e-9)
    # Planar trans dihedral p0-p1-p2-p3 = 180 deg.
    dihedral_geom = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, -1.0, 0.0],
        ],
        dtype=float,
    )
    assert abs(abs(dihedral_value(dihedral_geom, [0, 1, 2, 3])) - 180.0) < 1e-6
    with pytest.raises(NativeConstraintResultError, match="unsupported"):
        measure_coordinate("coordinate", geometry, [0, 1])


def test_nonfinite_optimized_coordinates_are_contract_breaks(tmp_path):
    request = make_request()
    attempt = write_attempt(tmp_path, request=request)
    bad_xyz = "3\ncomment\nC 0.0 0.0 0.0\nH nan 0.0 0.0\nH 3.5 0.0 0.0\n"
    Path(attempt.optimized_geometry_path).write_text(bad_xyz, encoding="utf-8")
    # Geometry parse fails -> stale_identity (typed), not a crash.
    outcome = normalize_outcome(attempt_result=attempt, request=request, driver_ids=["d0"])
    assert outcome.status == OUTCOME_STALE_IDENTITY
    assert "non-finite" in (outcome.failure_reason or "")
