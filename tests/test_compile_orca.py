"""Tests for pes2ts_core.scan_strategy.compile_orca (plan todo 19).

Covers: capability-gate-first ordering, atom_rows index resolution (never
map−1), B/A/D units + D periodic shortest-arc, Simul_Scan probe-receipt gate,
3/4-coordinate boundary, SCHEDULED per-point recipes with per-point hashes,
per-point constraint syntax, minimal NEB shape, every-driver recovery,
determinism, and typed refusals.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from pes2ts_core.scan_strategy import capabilities as caps
from pes2ts_core.scan_strategy import compile_orca as co

# ---------------------------------------------------------------------------
# Capability-registry fixtures (probe-receipt injection; todo-16 vocabulary).
# ---------------------------------------------------------------------------
ALL_MODES = ["SINGLE_1D", "COUPLED_1D", "SCHEDULED_1D", "PATH_NEB"]
ALL_KINDS = ["B", "A", "D"]


def _receipt(
    modes: list[str],
    *,
    probe_id: str = "probe-t19",
    status: str = "pass",
    engine_version: str = "6.1.0",
    adapter_version: str | None = "acp-pes-scan-v0",
) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "probe_id": probe_id,
        "date": "2026-10-01",
        "engine_version": engine_version,
        "status": status,
        "evidence_ref": "evidence/task-19",
        "modes": list(modes),
    }
    if adapter_version is not None:
        doc["adapter_version"] = adapter_version
    return doc


def _entry(
    entry_id: str,
    layer: str,
    *,
    modes: list[str],
    kinds: list[str],
    max_coords: int,
    receipts: list[dict[str, Any]],
    simul_scan: bool,
    per_point_constraints: bool,
    custom_schedule: bool,
    native_scan: bool = True,
    limits: tuple[int, int] = (2, 101),
    engine_version: str = "6.1.0",
    adapter_version: str = "acp-pes-scan-v0",
) -> dict[str, Any]:
    return {
        "entry_id": entry_id,
        "layer": layer,
        "engine": "orca",
        "adapter": "acp",
        "capability": {
            "engine_version": engine_version,
            "adapter_version": adapter_version,
            "supported_modes": list(modes),
            "coordinate_kinds": list(kinds),
            "max_scan_coordinates": max_coords,
            "point_limits": {"baseline": limits[0], "max": limits[1]},
            "custom_schedule_support": custom_schedule,
            "constraint_support": {
                "native_scan": native_scan,
                "per_point_constraints": per_point_constraints,
                "simul_scan": simul_scan,
            },
            "method_element_coverage": {"B3LYP-D3": ["C", "H", "N", "O"]},
            "probe_receipts": receipts,
        },
        "notes": "todo-19 synthetic stack",
    }


def _smoke(
    tmp_path: Path,
    *,
    modes: list[str] | None = None,
    kinds: list[str] | None = None,
    max_coords: int = 3,
    limits: tuple[int, int] = (2, 101),
    probed: bool = True,
    simul_scan: bool = True,
    per_point_constraints: bool = True,
    custom_schedule: bool = True,
    native_scan: bool = True,
    adapter_modes: list[str] | None = None,
    adapter_simul_scan: bool | None = None,
    adapter_per_point: bool | None = None,
    adapter_max: int | None = None,
) -> caps.EffectiveCapability:
    """Build a synthetic three-layer registry and return its effective capability."""
    modes = list(modes if modes is not None else ALL_MODES)
    kinds = list(kinds if kinds is not None else ALL_KINDS)
    receipts = [_receipt(modes)] if probed else []
    entries = [
        _entry(
            "eng",
            "engine",
            modes=modes,
            kinds=kinds,
            max_coords=max_coords,
            receipts=receipts,
            simul_scan=simul_scan,
            per_point_constraints=per_point_constraints,
            custom_schedule=custom_schedule,
            native_scan=native_scan,
            limits=limits,
        ),
        _entry(
            "adp",
            "adapter",
            modes=list(adapter_modes if adapter_modes is not None else modes),
            kinds=kinds,
            max_coords=adapter_max if adapter_max is not None else max_coords,
            receipts=receipts,
            simul_scan=simul_scan if adapter_simul_scan is None else adapter_simul_scan,
            per_point_constraints=(
                per_point_constraints if adapter_per_point is None else adapter_per_point
            ),
            custom_schedule=custom_schedule,
            native_scan=native_scan,
            limits=limits,
        ),
        _entry(
            "dep",
            "deployment",
            modes=modes,
            kinds=kinds,
            max_coords=max_coords,
            receipts=receipts,
            simul_scan=simul_scan,
            per_point_constraints=per_point_constraints,
            custom_schedule=custom_schedule,
            native_scan=native_scan,
            limits=limits,
        ),
    ]
    payload = {
        "schema_version": caps.REGISTRY_SCHEMA_VERSION,
        "registry_id": "todo19-test-stack",
        "description": "synthetic probed stack for compile_orca tests",
        "entries": entries,
    }
    path = tmp_path / "orca_capabilities_v1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    registry = caps.load_capability_registry(path)
    return caps.effective_capability("eng", "adp", "dep", registry=registry)


# ---------------------------------------------------------------------------
# Candidate / geometry fixtures.
# ---------------------------------------------------------------------------
def _driver(
    kind: str,
    maps: list[int],
    *,
    unit: str | None = None,
    schedule_values: list[float] | None = None,
    index0: list[int] | None = None,
) -> dict[str, Any]:
    resolved_unit = unit if unit is not None else ("angstrom" if kind == "B" else "degree")
    return {
        "kind": kind,
        "maps": list(maps),
        "unit": resolved_unit,
        "schedule_values": schedule_values,
        "index0": index0,
    }


def _scan_candidate(
    mode: str,
    drivers: list[dict[str, Any]],
    lambda_values: list[float],
    *,
    start_endpoint: str = "R",
    schedule_kind: str | None = None,
    candidate_id: str = "cand-t19",
) -> dict[str, Any]:
    candidate: dict[str, Any] = {
        "candidate_kind": "ScanCandidateV2",
        "candidate_id": candidate_id,
        "mode": mode,
        "start_endpoint": start_endpoint,
        "direction": "R_to_P" if start_endpoint == "R" else "P_to_R",
        "assembly_id": "asm-t19",
        "anchor_reason": "HAND_TEST",
        "drivers": drivers,
        "monitors": [],
        "guards": [],
        "event_coverage": [],
        "lambda_values": list(lambda_values),
        "schedule_id": "sched:t19",
        "required_capabilities": [mode],
        "budget": {"max_attempts": 1, "max_cpu_hours": 1.0, "max_wall_seconds": 60},
        "failure_reasons": [],
    }
    if schedule_kind is not None:
        candidate["schedule_kind"] = schedule_kind
    return candidate


def _path_candidate(
    n_atoms: int,
    reactant: list[list[float]],
    product: list[list[float]],
    n_images: int,
    *,
    candidate_id: str = "cand-path-t19",
) -> dict[str, Any]:
    return {
        "candidate_kind": "PathCandidateV1",
        "candidate_id": candidate_id,
        "n_atoms": n_atoms,
        "endpoint_geometries": {"reactant": reactant, "product": product},
        "image_chain": {"n_images": n_images},
        "start_endpoint": "R",
        "direction": "R_to_P",
        "anchor_reason": "HAND_TEST",
        "failure_reasons": [],
    }


def _b_geometry() -> dict[str, dict[int, tuple[float, float, float]]]:
    """B(1,2): 1.5 Å at R → 2.5 Å at P; B(1,3): 1.0 → 2.0."""
    return {
        "R": {1: (0.0, 0.0, 0.0), 2: (1.5, 0.0, 0.0), 3: (0.0, 1.0, 0.0), 4: (0.0, 0.0, 1.0)},
        "P": {1: (0.0, 0.0, 0.0), 2: (2.5, 0.0, 0.0), 3: (0.0, 2.0, 0.0), 4: (0.0, 0.0, 1.0)},
    }


def _a_geometry() -> dict[str, dict[int, tuple[float, float, float]]]:
    """A(2,1,3): 90° at R → 45° at P."""
    return {
        "R": {1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0), 3: (0.0, 1.0, 0.0), 4: (0.0, 0.0, 1.0)},
        "P": {1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0), 3: (1.0, 1.0, 0.0), 4: (0.0, 0.0, 1.0)},
    }


def _d_geometry(
    theta_r: float = 170.0, theta_p: float = -170.0
) -> dict[str, dict[int, tuple[float, float, float]]]:
    """D(2,1,3,4): dihedral θ_r at R → θ_p at P."""
    tr = math.radians(theta_r)
    tp = math.radians(theta_p)
    return {
        "R": {
            1: (0.0, 0.0, 0.0),
            2: (0.0, 1.0, 0.0),
            3: (1.0, 0.0, 0.0),
            4: (1.0, math.cos(tr), math.sin(tr)),
        },
        "P": {
            1: (0.0, 0.0, 0.0),
            2: (0.0, 1.0, 0.0),
            3: (1.0, 0.0, 0.0),
            4: (1.0, math.cos(tp), math.sin(tp)),
        },
    }


def _lambdas(n: int) -> list[float]:
    return [index / (n - 1) for index in range(n)]


def _lin_values(start: float, end: float, s_list: list[float]) -> list[float]:
    return [round(start + s * (end - start), 6) for s in s_list]


def _d_values(start: float, end: float, s_list: list[float]) -> list[float]:
    delta = (end - start + 180.0) % 360.0 - 180.0
    out: list[float] = []
    for s in s_list:
        value = (start + s * delta + 180.0) % 360.0 - 180.0
        if value <= -180.0:
            value += 360.0
        out.append(round(value, 6))
    return out


# ---------------------------------------------------------------------------
# SINGLE_1D: indices from atom_rows (never map−1), units, no implicit grid.
# ---------------------------------------------------------------------------
def test_single_1d_b_indices_from_atom_rows_not_map_minus_one(tmp_path) -> None:
    """Permuted atom order: maps [1,2] → indices (2,1), not map−1 (0,1)."""
    capability = _smoke(tmp_path, probed=True)
    candidate = _scan_candidate(
        "SINGLE_1D", [_driver("B", [1, 2])], _lambdas(9)
    )
    request = co.compile_orca(
        candidate, (4, 2, 1, 3), capability, geometry=_b_geometry()
    )
    coord = request.coordinates[0]
    assert coord.maps == (1, 2)
    # atom_rows (4,2,1,3): map1 is at position 2, map2 at position 1 (0-based).
    assert coord.indices == (2, 1)
    assert coord.indices != (1 - 1, 2 - 1)
    fragment = request.geom_fragments[0]
    assert "Scan B 2 1" in fragment
    assert "Scan B 0 1" not in fragment


def test_single_1d_b_units_and_fragment_and_no_implicit_grid(tmp_path) -> None:
    capability = _smoke(tmp_path)
    lambdas = _lambdas(9)
    candidate = _scan_candidate("SINGLE_1D", [_driver("B", [1, 2])], lambdas)
    request = co.compile_orca(candidate, (1, 2, 3, 4), capability, geometry=_b_geometry())
    coord = request.coordinates[0]
    assert coord.unit == "angstrom"
    assert coord.start_value == 1.5
    assert coord.end_value == 2.5
    assert list(coord.values) == _lin_values(1.5, 2.5, lambdas)
    fragment = request.geom_fragments[0]
    # steps = N-1; start/end in Å; single coordinate only.
    assert "Scan B 0 1 = 1.5, 2.5, 8 end" in fragment
    assert fragment.count("Scan ") == 1
    assert "Simul_Scan" not in fragment
    # No implicit grids: total points == λ count; one compiled input hash.
    assert request.total_points == len(lambdas)
    assert len(request.point_input_sha256) == 1
    assert request.compiled_kind == "hashes"
    assert request.simultaneous is False


def test_single_1d_a_units_are_degrees(tmp_path) -> None:
    capability = _smoke(tmp_path)
    lambdas = _lambdas(5)
    candidate = _scan_candidate("SINGLE_1D", [_driver("A", [2, 1, 3])], lambdas)
    request = co.compile_orca(candidate, (1, 2, 3, 4), capability, geometry=_a_geometry())
    coord = request.coordinates[0]
    assert coord.kind == "A"
    assert coord.unit == "degree"
    assert coord.start_value == 90.0
    assert coord.end_value == 45.0
    assert list(coord.values) == _lin_values(90.0, 45.0, lambdas)
    assert "Scan A 1 0 2 = 90, 45, 4 end" in request.geom_fragments[0]


def test_single_1d_d_periodic_shortest_arc_native_scan(tmp_path) -> None:
    """D 170°→−170° interpolates on the +20° short arc, never through 0°."""
    capability = _smoke(tmp_path)
    lambdas = _lambdas(5)
    candidate = _scan_candidate("SINGLE_1D", [_driver("D", [2, 1, 3, 4])], lambdas)
    request = co.compile_orca(candidate, (1, 2, 3, 4), capability, geometry=_d_geometry())
    coord = request.coordinates[0]
    assert coord.unit == "degree"
    assert coord.start_value == 170.0
    assert coord.end_value == -170.0
    expected = _d_values(170.0, -170.0, lambdas)
    assert list(coord.values) == expected
    mid = coord.values[2]
    assert abs(mid) == pytest.approx(180.0)
    assert mid != pytest.approx(0.0)
    assert "Scan D 1 0 2 3 = 170, -170, 4 end" in request.geom_fragments[0]


def test_d_periodic_per_point_values_use_shortest_arc(tmp_path) -> None:
    """Non-linear D schedule compiles per-point with periodic interpolation."""
    capability = _smoke(tmp_path)
    lambdas = _lambdas(5)
    s_values = [0.0, 0.1, 0.5, 0.9, 1.0]
    candidate = _scan_candidate(
        "SINGLE_1D",
        [_driver("D", [2, 1, 3, 4], schedule_values=s_values)],
        lambdas,
        schedule_kind="smoothstep",
    )
    request = co.compile_orca(candidate, (1, 2, 3, 4), capability, geometry=_d_geometry())
    coord = request.coordinates[0]
    assert list(coord.values) == _d_values(170.0, -170.0, s_values)
    assert abs(coord.values[2]) == pytest.approx(180.0)
    assert request.compiled_kind == "recipe"
    assert len(request.point_input_sha256) == 5


# ---------------------------------------------------------------------------
# COUPLED_1D: Simul_Scan only with a passing probe receipt.
# ---------------------------------------------------------------------------
def test_coupled_simul_scan_compiled_with_passing_probe_receipt(tmp_path) -> None:
    capability = _smoke(tmp_path, probed=True, simul_scan=True)
    lambdas = _lambdas(5)
    candidate = _scan_candidate(
        "COUPLED_1D",
        [_driver("B", [1, 2]), _driver("B", [1, 3])],
        lambdas,
    )
    request = co.compile_orca(candidate, (1, 2, 3, 4), capability, geometry=_b_geometry())
    assert request.simultaneous is True
    fragment = request.geom_fragments[0]
    assert fragment.count("Scan B") == 2
    assert "Simul_Scan true" in fragment
    assert "Scan B 0 1 = 1.5, 2.5, 4 end" in fragment
    assert "Scan B 0 2 = 1, 2, 4 end" in fragment
    # Simultaneous-step count: both coordinates share steps = N-1.
    assert fragment.count(", 4 end") == 2
    # λ alignment: every driver's values[i] is evaluated at λ_i.
    assert request.total_points == len(lambdas)
    assert len(request.coordinates) == 2
    assert list(request.coordinates[0].values) == _lin_values(1.5, 2.5, lambdas)
    assert list(request.coordinates[1].values) == _lin_values(1.0, 2.0, lambdas)
    assert len(request.point_input_sha256) == 1


def test_coupled_refused_without_probe_receipt_simul_scan_unprobed(tmp_path) -> None:
    capability = _smoke(tmp_path, probed=False, simul_scan=True)
    candidate = _scan_candidate(
        "COUPLED_1D",
        [_driver("B", [1, 2]), _driver("B", [1, 3])],
        _lambdas(5),
    )
    with pytest.raises(co.OrcaCompileError) as excinfo:
        co.compile_orca(candidate, (1, 2, 3, 4), capability, geometry=_b_geometry())
    assert excinfo.value.code == co.CODE_SIMUL_SCAN_UNPROBED
    assert "probe receipt" in str(excinfo.value)


def test_coupled_multi_scan_rejected_when_simul_scan_capability_missing(tmp_path) -> None:
    """Mode enabled by receipts, but the simul_scan capability claim is false."""
    capability = _smoke(tmp_path, probed=True, adapter_simul_scan=False)
    assert "COUPLED_1D" in capability.enabled_modes
    assert capability.constraint_support.simul_scan is False
    candidate = _scan_candidate(
        "COUPLED_1D",
        [_driver("B", [1, 2]), _driver("B", [1, 3])],
        _lambdas(5),
    )
    with pytest.raises(co.OrcaCompileError) as excinfo:
        co.compile_orca(candidate, (1, 2, 3, 4), capability, geometry=_b_geometry())
    assert excinfo.value.code == co.CODE_SIMUL_SCAN_UNSUPPORTED


def test_coupled_rejected_when_mode_undeclared_by_adapter(tmp_path) -> None:
    capability = _smoke(
        tmp_path, probed=True, adapter_modes=["SINGLE_1D", "SCHEDULED_1D", "PATH_NEB"]
    )
    candidate = _scan_candidate(
        "COUPLED_1D",
        [_driver("B", [1, 2]), _driver("B", [1, 3])],
        _lambdas(5),
    )
    with pytest.raises(co.OrcaCompileError) as excinfo:
        co.compile_orca(candidate, (1, 2, 3, 4), capability, geometry=_b_geometry())
    assert excinfo.value.code == co.CODE_BACKEND_CAPABILITY_MISSING


def test_coupled_rejected_when_schedule_is_non_linear(tmp_path) -> None:
    """Simul_Scan compiles the shared linear λ grid only — no implicit grids."""
    capability = _smoke(tmp_path, probed=True)
    candidate = _scan_candidate(
        "COUPLED_1D",
        [
            _driver("B", [1, 2], schedule_values=[0.0, 0.2, 0.5, 0.8, 1.0]),
            _driver("B", [1, 3]),
        ],
        _lambdas(5),
    )
    with pytest.raises(co.OrcaCompileError) as excinfo:
        co.compile_orca(candidate, (1, 2, 3, 4), capability, geometry=_b_geometry())
    assert excinfo.value.code == co.CODE_SIMUL_SCAN_UNSUPPORTED


def test_three_coordinates_compile_with_simul_scan_boundary(tmp_path) -> None:
    capability = _smoke(tmp_path, probed=True)
    lambdas = _lambdas(5)
    candidate = _scan_candidate(
        "COUPLED_1D",
        [_driver("B", [1, 2]), _driver("B", [1, 3]), _driver("A", [2, 1, 3])],
        lambdas,
    )
    request = co.compile_orca(candidate, (1, 2, 3, 4), capability, geometry=_b_geometry())
    assert request.simultaneous is True
    assert len(request.coordinates) == 3
    fragment = request.geom_fragments[0]
    scan_lines = [line for line in fragment.splitlines() if line.startswith("Scan ")]
    assert len(scan_lines) == 3
    assert fragment.count(", 4 end") == 3
    assert "Simul_Scan true" in fragment


def test_four_coordinates_rejected(tmp_path) -> None:
    capability = _smoke(tmp_path, probed=True, max_coords=3)
    candidate = _scan_candidate(
        "COUPLED_1D",
        [
            _driver("B", [1, 2]),
            _driver("B", [1, 3]),
            _driver("A", [2, 1, 3]),
            _driver("D", [2, 1, 3, 4]),
        ],
        _lambdas(5),
    )
    with pytest.raises(co.OrcaCompileError) as excinfo:
        co.compile_orca(candidate, (1, 2, 3, 4), capability, geometry=_b_geometry())
    assert excinfo.value.code == co.CODE_BACKEND_CAPABILITY_MISSING
    assert "max_scan_coordinates" in str(excinfo.value) or "upper bound 3" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Capability gate ordering (gate before any compilation output).
# ---------------------------------------------------------------------------
def test_capability_gate_runs_before_compilation(tmp_path) -> None:
    """Unprobed + missing geometry + unknown map → the capability code wins."""
    capability = _smoke(tmp_path, probed=False)
    candidate = _scan_candidate(
        "SINGLE_1D", [_driver("B", [1, 99])], _lambdas(5)
    )
    with pytest.raises(co.OrcaCompileError) as excinfo:
        co.compile_orca(candidate, (1, 2, 3, 4), capability, geometry=None)
    assert excinfo.value.code == co.CODE_CAPABILITY_UNPROBED


def test_capability_gate_runs_before_compilation_coupled(tmp_path) -> None:
    capability = _smoke(tmp_path, probed=False)
    candidate = _scan_candidate(
        "COUPLED_1D",
        [_driver("B", [1, 2]), _driver("B", [1, 3])],
        _lambdas(5),
    )
    with pytest.raises(co.OrcaCompileError) as excinfo:
        co.compile_orca(candidate, (1, 2, 3, 4), capability, geometry=None)
    assert excinfo.value.code == co.CODE_SIMUL_SCAN_UNPROBED


def test_shipped_baseline_registry_refuses_everything(tmp_path) -> None:
    """Shipped registry is unprobed (todo 16) — compile is honestly refused."""
    registry = caps.load_capability_registry()
    effective = caps.effective_capability(
        "orca-acp-engine", "acp-adapter-v0", "deployment-baseline", registry=registry
    )
    candidate = _scan_candidate("SINGLE_1D", [_driver("B", [1, 2])], _lambdas(9))
    with pytest.raises(co.OrcaCompileError) as excinfo:
        co.compile_orca(candidate, (1, 2, 3, 4), effective, geometry=_b_geometry())
    assert excinfo.value.code == co.CODE_CAPABILITY_UNPROBED


# ---------------------------------------------------------------------------
# SCHEDULED_1D custom lists: per-point recipe + per-point hashes.
# ---------------------------------------------------------------------------
def test_scheduled_per_point_hashes_count_and_values_match_schedule(tmp_path) -> None:
    capability = _smoke(tmp_path, probed=True)
    lambdas = _lambdas(5)
    s_a = [0.0, 0.1, 0.5, 0.9, 1.0]  # non-linear window
    s_b = [0.0, 0.3, 0.5, 0.7, 1.0]
    candidate = _scan_candidate(
        "SCHEDULED_1D",
        [
            _driver("B", [1, 2], schedule_values=s_a),
            _driver("B", [1, 3], schedule_values=s_b),
        ],
        lambdas,
        schedule_kind="smoothstep",
    )
    request = co.compile_orca(candidate, (1, 2, 3, 4), capability, geometry=_b_geometry())
    assert request.compiled_kind == "recipe"
    assert request.recipe_kind == co.RECIPE_PER_POINT
    # Per-point hashes: one sealed generation recipe, N hashes — not N inline files.
    assert len(request.point_input_sha256) == request.total_points == 5
    assert len(set(request.point_input_sha256)) == 5
    assert "recipe_kind" in request.recipe and "generation" in request.recipe
    coord_a, coord_b = request.coordinates
    assert list(coord_a.values) == _lin_values(1.5, 2.5, s_a)
    assert list(coord_b.values) == _lin_values(1.0, 2.0, s_b)
    assert len(request.geom_fragments) == 5


def test_single_driver_non_linear_compiles_per_point_constraints(tmp_path) -> None:
    capability = _smoke(tmp_path, probed=True)
    lambdas = _lambdas(5)
    s_values = [0.0, 0.15625, 0.5, 0.84375, 1.0]  # smoothstep(λ) — non-linear
    candidate = _scan_candidate(
        "SINGLE_1D",
        [_driver("B", [1, 2], schedule_values=s_values)],
        lambdas,
        schedule_kind="smoothstep",
    )
    request = co.compile_orca(candidate, (1, 2, 3, 4), capability, geometry=_b_geometry())
    assert request.compiled_kind == "recipe"
    assert len(request.point_input_sha256) == 5
    assert list(request.coordinates[0].values) == _lin_values(1.5, 2.5, s_values)


def test_per_point_constraints_block_syntax(tmp_path) -> None:
    capability = _smoke(tmp_path, probed=True)
    lambdas = _lambdas(3)
    s_values = [0.0, 0.5, 1.0]
    candidate = _scan_candidate(
        "SCHEDULED_1D",
        [
            _driver("B", [1, 2], schedule_values=s_values),
            _driver("A", [2, 1, 3], schedule_values=s_values),
        ],
        lambdas,
        schedule_kind="linear",
    )
    request = co.compile_orca(candidate, (1, 2, 3, 4), capability, geometry=_a_geometry())
    assert len(request.geom_fragments) == 3
    for index, fragment in enumerate(request.geom_fragments):
        assert fragment.startswith("%geom\nConstraints\n")
        assert fragment.endswith("\nend")
        # Both drivers appear in every point input — no first-driver projection.
        assert "{ B 0 1 " in fragment
        assert "{ A 1 0 2 " in fragment
    # Point-0 constraint values equal the drivers' start values.
    assert "{ B 0 1 1.5 }" in request.geom_fragments[0] or "{ B 0 1 1 }" in request.geom_fragments[0]


def test_scheduled_refused_without_per_point_constraints_capability(tmp_path) -> None:
    capability = _smoke(tmp_path, probed=True, adapter_per_point=False)
    candidate = _scan_candidate(
        "SCHEDULED_1D",
        [
            _driver("B", [1, 2], schedule_values=[0.0, 0.5, 1.0]),
            _driver("B", [1, 3], schedule_values=[0.0, 0.5, 1.0]),
        ],
        _lambdas(3),
        schedule_kind="linear",
    )
    with pytest.raises(co.OrcaCompileError) as excinfo:
        co.compile_orca(candidate, (1, 2, 3, 4), capability, geometry=_b_geometry())
    assert excinfo.value.code == co.CODE_PER_POINT_CONSTRAINTS_UNSUPPORTED


# ---------------------------------------------------------------------------
# Path / NEB minimal shape.
# ---------------------------------------------------------------------------
def _path_geometry(n_atoms: int = 4) -> tuple[list[list[float]], list[list[float]]]:
    reactant = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    product = [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.1, 1.0, 0.0], [0.0, 0.0, 1.0]]
    return reactant[:n_atoms], product[:n_atoms]


def test_neb_minimal_shape(tmp_path) -> None:
    capability = _smoke(tmp_path, probed=True)
    reactant, product = _path_geometry(4)
    candidate = _path_candidate(4, reactant, product, 5)
    request = co.compile_orca(
        candidate,
        (1, 2, 3, 4),
        capability,
        elements=["C", "C", "O", "H"],
    )
    assert request.mode == "PATH_NEB"
    assert request.total_points == 5  # image count
    assert request.coordinates == ()
    fragments = request.geom_fragments
    assert any("Path" in fragment and "n_images 5" in fragment for fragment in fragments)
    xyz_fragments = [fragment for fragment in fragments if fragment.startswith("* xyz")]
    assert len(xyz_fragments) == 2
    assert "C 0 0 0" in xyz_fragments[0]
    assert len(request.point_input_sha256) == 1
    assert request.recipe["n_images"] == 5
    assert request.recipe["endpoint_geometries"]["reactant"] == reactant


def test_neb_refused_without_path_probe(tmp_path) -> None:
    capability = _smoke(tmp_path, probed=False)
    reactant, product = _path_geometry(4)
    candidate = _path_candidate(4, reactant, product, 5)
    with pytest.raises(co.OrcaCompileError) as excinfo:
        co.compile_orca(candidate, (1, 2, 3, 4), capability)
    assert excinfo.value.code == co.CODE_CAPABILITY_UNPROBED


def test_neb_rejects_bad_image_count(tmp_path) -> None:
    capability = _smoke(tmp_path, probed=True)
    reactant, product = _path_geometry(4)
    candidate = _path_candidate(4, reactant, product, 0)
    with pytest.raises(co.OrcaCompileError) as excinfo:
        co.compile_orca(candidate, (1, 2, 3, 4), capability)
    assert excinfo.value.code == co.CODE_PATH_GEOMETRY_INVALID


# ---------------------------------------------------------------------------
# Every-driver recovery + determinism + typed refusals.
# ---------------------------------------------------------------------------
def test_every_driver_coordinate_present_no_first_driver_projection(tmp_path) -> None:
    capability = _smoke(tmp_path, probed=True)
    lambdas = _lambdas(5)
    drivers = [
        _driver("B", [1, 2]),
        _driver("B", [1, 3]),
        _driver("A", [2, 1, 3]),
    ]
    candidate = _scan_candidate("COUPLED_1D", drivers, lambdas)
    request = co.compile_orca(candidate, (1, 2, 3, 4), capability, geometry=_a_geometry())
    assert len(request.coordinates) == len(drivers)
    fragment = request.geom_fragments[0]
    for coord in request.coordinates:
        index_token = " ".join(str(index) for index in coord.indices)
        assert f"Scan {coord.kind} {index_token}" in fragment
    recorded_maps = [tuple(row.maps) for row in request.coordinates]
    assert recorded_maps == [(1, 2), (1, 3), (2, 1, 3)]


def test_determinism_stable_json(tmp_path) -> None:
    capability = _smoke(tmp_path, probed=True)
    candidate = _scan_candidate(
        "SCHEDULED_1D",
        [
            _driver("B", [1, 2], schedule_values=[0.0, 0.5, 1.0]),
            _driver("B", [1, 3], schedule_values=[0.0, 0.5, 1.0]),
        ],
        _lambdas(3),
        schedule_kind="linear",
    )
    first = co.compile_orca(candidate, (1, 2, 3, 4), capability, geometry=_b_geometry())
    second = co.compile_orca(candidate, (1, 2, 3, 4), capability, geometry=_b_geometry())
    assert first.to_json() == second.to_json()
    doc = json.loads(first.to_json())
    assert doc["schema_version"] == co.SCHEMA_ORCA_COMPILE
    assert doc["total_points"] == 3
    assert len(doc["point_input_sha256"]) == 3


def test_typed_refusal_driver_map_unknown(tmp_path) -> None:
    capability = _smoke(tmp_path, probed=True)
    candidate = _scan_candidate("SINGLE_1D", [_driver("B", [1, 99])], _lambdas(5))
    with pytest.raises(co.OrcaCompileError) as excinfo:
        co.compile_orca(candidate, (1, 2, 3, 4), capability, geometry=_b_geometry())
    assert excinfo.value.code == co.CODE_DRIVER_MAP_UNKNOWN


def test_typed_refusal_index_order_mismatch(tmp_path) -> None:
    capability = _smoke(tmp_path, probed=True)
    candidate = _scan_candidate(
        "SINGLE_1D", [_driver("B", [1, 2], index0=[0, 1])], _lambdas(5)
    )
    # atom_rows (1,2,3,4): computed indices (0,1) == frozen index0 → compiles.
    ok = co.compile_orca(candidate, (1, 2, 3, 4), capability, geometry=_b_geometry())
    assert ok.coordinates[0].indices == (0, 1)
    # atom_rows (4,2,1,3): computed indices (2,1) ≠ frozen index0 → typed refusal.
    with pytest.raises(co.OrcaCompileError) as excinfo:
        co.compile_orca(candidate, (4, 2, 1, 3), capability, geometry=_b_geometry())
    assert excinfo.value.code == co.CODE_INDEX_ORDER_MISMATCH


def test_typed_refusal_unit_unsupported(tmp_path) -> None:
    capability = _smoke(tmp_path, probed=True)
    candidate = _scan_candidate(
        "SINGLE_1D", [_driver("A", [2, 1, 3], unit="angstrom")], _lambdas(5)
    )
    with pytest.raises(co.OrcaCompileError) as excinfo:
        co.compile_orca(candidate, (1, 2, 3, 4), capability, geometry=_a_geometry())
    assert excinfo.value.code == co.CODE_UNIT_UNSUPPORTED


def test_typed_refusal_point_window(tmp_path) -> None:
    capability = _smoke(tmp_path, probed=True, limits=(9, 101))
    candidate = _scan_candidate("SINGLE_1D", [_driver("B", [1, 2])], _lambdas(5))
    with pytest.raises(co.OrcaCompileError) as excinfo:
        co.compile_orca(candidate, (1, 2, 3, 4), capability, geometry=_b_geometry())
    assert excinfo.value.code == co.CODE_POINT_WINDOW_INVALID


def test_typed_refusal_geometry_required_after_gate(tmp_path) -> None:
    capability = _smoke(tmp_path, probed=True)
    candidate = _scan_candidate("SINGLE_1D", [_driver("B", [1, 2])], _lambdas(5))
    with pytest.raises(co.OrcaCompileError) as excinfo:
        co.compile_orca(candidate, (1, 2, 3, 4), capability, geometry=None)
    assert excinfo.value.code == co.CODE_GEOMETRY_REQUIRED


def test_typed_refusal_atom_order_invalid(tmp_path) -> None:
    capability = _smoke(tmp_path, probed=True)
    candidate = _scan_candidate("SINGLE_1D", [_driver("B", [1, 2])], _lambdas(5))
    with pytest.raises(co.OrcaCompileError) as excinfo:
        co.compile_orca(candidate, (1, 2, 2, 3), capability, geometry=_b_geometry())
    assert excinfo.value.code == co.CODE_ATOM_ORDER_INVALID


def test_typed_refusal_unknown_candidate_kind(tmp_path) -> None:
    capability = _smoke(tmp_path, probed=True)
    candidate = {"candidate_kind": "SomethingElse", "candidate_id": "x"}
    with pytest.raises(co.OrcaCompileError) as excinfo:
        co.compile_orca(candidate, (1, 2, 3, 4), capability)
    assert excinfo.value.code == co.CODE_UNKNOWN_CANDIDATE_KIND
