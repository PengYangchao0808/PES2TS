"""Tests for pes2ts_core.integration.acp.multicoord (plan todo 20).

Covers: multi-coordinate payloads carrying ALL drivers (units+maps+indices+
per-point values, never first-driver-only), A/D kinds, non-uniform per-point
lists, capability refusal on the shipped default registry
(``BACKEND_CAPABILITY_MISSING``), payload building under synthetic probe
receipts, the per-frame all-driver result parse surface, explicit refusals on
the current single-distance adapter, and a byte-identical regression pin of
``scan_plan_to_acp_request`` derived from existing tests.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from pes2ts_core.contracts import dumps_document, make_document, seal_document
from pes2ts_core.integration.acp.adapter import ACPMappingError, scan_plan_to_acp_request
from pes2ts_core.integration.acp.multicoord import (
    ADAPTER_VERSION_MULTICOORD,
    CODE_BACKEND_CAPABILITY_MISSING,
    CODE_CAPABILITY_UNPROBED,
    CODE_CUSTOM_SCHEDULE_UNSUPPORTED,
    CODE_SIMUL_SCAN_UNSUPPORTED,
    SCHEMA_MULTICOORD_REQUEST,
    SCHEMA_MULTICOORD_RESULT,
    MulticoordAdapterError,
    driver_id_for,
    multicoord_request_payload,
    parse_multicoord_result,
)
from pes2ts_core.planning import build_minimal_scan_plan
from pes2ts_core.scan_strategy import capabilities as caps
from pes2ts_core.scan_strategy import compile_orca as co
from pes2ts_core.utils.hashing import stable_json_dumps

# ---------------------------------------------------------------------------
# Capability-registry fixtures (synthetic probe receipts; todo-16 vocabulary).
# ---------------------------------------------------------------------------
ALL_MODES = ["SINGLE_1D", "COUPLED_1D", "SCHEDULED_1D", "PATH_NEB"]
ALL_KINDS = ["B", "A", "D"]
# Golden digest of stable_json_dumps(scan_plan_to_acp_request(simple_case,
# build_minimal_scan_plan(sealed simple_case))) — derived from the existing
# test_contracts.py fixture; adapter.py is byte-frozen so this must not move.
GOLDEN_ADAPTER_REQUEST_SHA256 = "4ee2bbeb51cb649220b1617fe537e58fb93b49ca6f31b1eec171a608985228b4"


def _receipt(modes: list[str], *, status: str = "pass") -> dict[str, Any]:
    return {
        "probe_id": "probe-t20",
        "date": "2026-10-01",
        "engine_version": "6.1.0",
        "adapter_version": "acp-pes-scan-v0",
        "status": status,
        "evidence_ref": "evidence/task-20",
        "modes": list(modes),
    }


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
    limits: tuple[int, int] = (2, 101),
) -> dict[str, Any]:
    return {
        "entry_id": entry_id,
        "layer": layer,
        "engine": "orca",
        "adapter": "acp",
        "capability": {
            "engine_version": "6.1.0",
            "adapter_version": "acp-pes-scan-v0",
            "supported_modes": list(modes),
            "coordinate_kinds": list(kinds),
            "max_scan_coordinates": max_coords,
            "point_limits": {"baseline": limits[0], "max": limits[1]},
            "custom_schedule_support": custom_schedule,
            "constraint_support": {
                "native_scan": True,
                "per_point_constraints": per_point_constraints,
                "simul_scan": simul_scan,
            },
            "method_element_coverage": {"B3LYP-D3": ["C", "H", "N", "O"]},
            "probe_receipts": receipts,
        },
        "notes": "todo-20 synthetic stack",
    }


def _smoke(
    tmp_path: Path,
    *,
    modes: list[str] | None = None,
    kinds: list[str] | None = None,
    max_coords: int = 3,
    probed: bool = True,
    simul_scan: bool = True,
    per_point_constraints: bool = True,
    custom_schedule: bool = True,
    adapter_modes: list[str] | None = None,
    adapter_simul_scan: bool | None = None,
    adapter_per_point: bool | None = None,
    adapter_custom_schedule: bool | None = None,
    adapter_max: int | None = None,
) -> caps.EffectiveCapability:
    modes = list(modes if modes is not None else ALL_MODES)
    kinds = list(kinds if kinds is not None else ALL_KINDS)
    receipts = [_receipt(modes)] if probed else []
    entries = [
        _entry(
            "eng", "engine", modes=modes, kinds=kinds, max_coords=max_coords,
            receipts=receipts, simul_scan=simul_scan,
            per_point_constraints=per_point_constraints, custom_schedule=custom_schedule,
        ),
        _entry(
            "adp", "adapter",
            modes=list(adapter_modes if adapter_modes is not None else modes),
            kinds=kinds,
            max_coords=adapter_max if adapter_max is not None else max_coords,
            receipts=receipts,
            simul_scan=simul_scan if adapter_simul_scan is None else adapter_simul_scan,
            per_point_constraints=(
                per_point_constraints if adapter_per_point is None else adapter_per_point
            ),
            custom_schedule=custom_schedule if adapter_custom_schedule is None
            else adapter_custom_schedule,
        ),
        _entry(
            "dep", "deployment", modes=modes, kinds=kinds, max_coords=max_coords,
            receipts=receipts, simul_scan=simul_scan,
            per_point_constraints=per_point_constraints, custom_schedule=custom_schedule,
        ),
    ]
    payload = {
        "schema_version": caps.REGISTRY_SCHEMA_VERSION,
        "registry_id": "todo20-test-stack",
        "description": "synthetic probed stack for multicoord adapter tests",
        "entries": entries,
    }
    path = tmp_path / "orca_capabilities_v1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    registry = caps.load_capability_registry(path)
    return caps.effective_capability("eng", "adp", "dep", registry=registry)


def _default_capability() -> caps.EffectiveCapability:
    """Shipped registry (todo 16): adapter SINGLE_1D/B only, everything unprobed."""
    registry = caps.load_capability_registry()
    return caps.effective_capability(
        "orca-acp-engine", "acp-adapter-v0", "deployment-baseline", registry=registry
    )


# ---------------------------------------------------------------------------
# Candidate / geometry / materials fixtures.
# ---------------------------------------------------------------------------
ATOM_ROWS = (1, 2, 3, 4)
ELEMENTS = ["C", "C", "H", "H"]


def _b2_geometry() -> dict[str, dict[int, tuple[float, float, float]]]:
    """B(1,2): 1.5→2.5 Å; B(3,4): 1.0→2.0 Å."""
    return {
        "R": {1: (0.0, 0.0, 0.0), 2: (1.5, 0.0, 0.0), 3: (0.0, 0.0, 2.0), 4: (0.0, 0.0, 3.0)},
        "P": {1: (0.0, 0.0, 0.0), 2: (2.5, 0.0, 0.0), 3: (0.0, 0.0, 2.0), 4: (0.0, 0.0, 4.0)},
    }


def _ad_geometry() -> dict[str, dict[int, tuple[float, float, float]]]:
    """A(2,1,3)=90° and D(2,1,3,4)=170°→−170° (todo-19 proven geometry)."""
    import math

    tr = math.radians(170.0)
    tp = math.radians(-170.0)
    return {
        "R": {
            1: (0.0, 0.0, 0.0), 2: (0.0, 1.0, 0.0), 3: (1.0, 0.0, 0.0),
            4: (1.0, math.cos(tr), math.sin(tr)),
        },
        "P": {
            1: (0.0, 0.0, 0.0), 2: (0.0, 1.0, 0.0), 3: (1.0, 0.0, 0.0),
            4: (1.0, math.cos(tp), math.sin(tp)),
        },
    }


def _materials(geom: dict[str, dict[int, tuple[float, float, float]]]) -> dict[str, Any]:
    return {
        "atom_map_ids": list(ATOM_ROWS),
        "elements": list(ELEMENTS),
        "r_coordinates": [list(geom["R"][m]) for m in ATOM_ROWS],
        "p_coordinates": [list(geom["P"][m]) for m in ATOM_ROWS],
        "charge": 0,
        "multiplicity": 1,
    }


def _driver(kind: str, maps: list[int], *, schedule_values: list[float] | None = None) -> dict[str, Any]:
    return {
        "kind": kind,
        "maps": list(maps),
        "unit": "angstrom" if kind == "B" else "degree",
        "schedule_values": schedule_values,
        "index0": None,
    }


def _scan_candidate(
    mode: str,
    drivers: list[dict[str, Any]],
    lambda_values: list[float],
    *,
    schedule_kind: str | None = None,
    candidate_id: str = "cand-t20",
) -> dict[str, Any]:
    candidate: dict[str, Any] = {
        "candidate_kind": "ScanCandidateV2",
        "candidate_id": candidate_id,
        "mode": mode,
        "start_endpoint": "R",
        "direction": "R_to_P",
        "assembly_id": "asm-t20",
        "anchor_reason": "HAND_TEST",
        "drivers": drivers,
        "monitors": [],
        "guards": [],
        "event_coverage": [],
        "lambda_values": list(lambda_values),
        "schedule_id": "sched:t20",
        "required_capabilities": [mode],
        "budget": {"max_attempts": 1, "max_cpu_hours": 1.0, "max_wall_seconds": 60},
        "failure_reasons": [],
    }
    if schedule_kind is not None:
        candidate["schedule_kind"] = schedule_kind
    return candidate


def _lambdas(n: int) -> list[float]:
    return [index / (n - 1) for index in range(n)]


def _compile_2b(tmp_path: Path) -> Any:
    capability = _smoke(tmp_path)
    candidate = _scan_candidate(
        "COUPLED_1D", [_driver("B", [1, 2]), _driver("B", [3, 4])], _lambdas(9)
    )
    return co.compile_orca(candidate, ATOM_ROWS, capability, geometry=_b2_geometry())


def _compile_ad(tmp_path: Path) -> Any:
    capability = _smoke(tmp_path)
    candidate = _scan_candidate(
        "COUPLED_1D", [_driver("A", [2, 1, 3]), _driver("D", [2, 1, 3, 4])], _lambdas(9)
    )
    return co.compile_orca(candidate, ATOM_ROWS, capability, geometry=_ad_geometry())


def _compile_scheduled_nonuniform(tmp_path: Path) -> Any:
    capability = _smoke(tmp_path)
    lambdas = _lambdas(4)
    candidate = _scan_candidate(
        "SCHEDULED_1D",
        [
            _driver("B", [1, 2], schedule_values=[0.0, 0.25, 0.6, 1.0]),
            _driver("B", [3, 4], schedule_values=[0.0, 0.5, 0.75, 1.0]),
        ],
        lambdas,
        schedule_kind="event_A_early",
    )
    return co.compile_orca(candidate, ATOM_ROWS, capability, geometry=_b2_geometry())


def _compile_single_nonuniform(tmp_path: Path) -> Any:
    capability = _smoke(tmp_path)
    candidate = _scan_candidate(
        "SINGLE_1D",
        [_driver("B", [1, 2], schedule_values=[0.0, 0.2, 0.55, 1.0])],
        _lambdas(4),
        schedule_kind="smoothstep",
    )
    return co.compile_orca(candidate, ATOM_ROWS, capability, geometry=_b2_geometry())


def _simple_case() -> dict[str, Any]:
    """Exact test_contracts.py simple_case fixture (byte-identical pin source)."""
    case = make_document(
        "ReactionCase", "case:rxn-demo-v1", "ready",
        dataset_version="synthetic-v1", reaction_id="RXN_DEMO_0001",
        case_id="case:rxn-demo-v1", split="train",
        atoms=[{"atom_map_id": 2, "element": "C"}, {"atom_map_id": 7, "element": "H"},
               {"atom_map_id": 12, "element": "N"}],
        reactant={"charge": 0, "multiplicity": 1, "geometry": [[0, 0, 0], [1.1, 0, 0], [3.5, 0, 0]]},
        product={"charge": 0, "multiplicity": 1, "geometry": [[0, 0, 0], [1.8, 0, 0], [2.8, 0, 0]]},
        edits=[{"kind": "broken", "atom_map_ids": [2, 7]},
               {"kind": "formed", "atom_map_ids": [7, 12]}],
        hydrogen_transfers=[{"hydrogen_map_id": 7, "old_partner_map_id": 2,
                             "new_partner_map_id": 12}],
        source={"dataset": "synthetic", "mapping_provenance": "endpoint_only"},
    )
    return seal_document(case)


def _tampered_ready_plan(case: dict[str, Any], mutate) -> dict[str, Any]:
    plan = copy.deepcopy(build_minimal_scan_plan(case))
    mutate(plan)
    return seal_document(plan)


# ---------------------------------------------------------------------------
# Payload: all drivers, units, maps, per-point values.
# ---------------------------------------------------------------------------
def test_multicoord_payload_carries_all_drivers_with_units_maps_values(tmp_path) -> None:
    """2-B COUPLED payload carries BOTH drivers — kind/maps/indices/units/values."""
    compiled = _compile_2b(tmp_path)
    materials = _materials(_b2_geometry())
    payload = multicoord_request_payload(compiled, materials, {"capability": _smoke(tmp_path)})

    assert payload["adapter_version"] == ADAPTER_VERSION_MULTICOORD
    assert payload["schema_version"] == SCHEMA_MULTICOORD_REQUEST
    assert payload["metadata"]["n_drivers"] == 2
    assert payload["metadata"]["driver_ids"] == ["driver-01", "driver-02"]
    assert payload["metadata"]["simultaneous"] is True
    assert payload["metadata"]["mode"] == "COUPLED_1D"
    assert payload["metadata"]["start_endpoint"] == "R"
    assert len(payload["coordinates"]) == 2  # never first-driver-only
    first, second = payload["coordinates"]
    assert first["coordinate_id"] == driver_id_for(0) == "driver-01"
    assert first["kind"] == "B"
    assert first["atom_map_ids"] == [1, 2]
    assert first["atom_indices"] == [0, 1]
    assert first["unit"] == "angstrom"
    assert first["values"] == list(compiled.coordinates[0].values)
    assert len(first["values"]) == 9
    assert second["atom_map_ids"] == [3, 4]
    assert second["atom_indices"] == [2, 3]
    assert second["unit"] == "angstrom"
    assert second["values"] == list(compiled.coordinates[1].values)
    assert second["values"] != first["values"]  # distinct per-driver schedules present
    assert payload["lambda_values"] == list(compiled.lambda_values)
    assert payload["source"]["source_type"] == "xyz_text"
    assert payload["source"]["xyz_text"].startswith("4\nPES2TS multicoord cand-t20\n")
    assert any("Simul_Scan true" in fragment for fragment in payload["geom_fragments"])
    assert len(payload["point_input_sha256"]) == 1
    assert payload["recipe"]["simul_scan"] is True


def test_multicoord_payload_represents_a_and_d_kinds(tmp_path) -> None:
    """A and D drivers are represented with degree units and their values."""
    compiled = _compile_ad(tmp_path)
    payload = multicoord_request_payload(
        compiled, _materials(_ad_geometry()), {"capability": _smoke(tmp_path)}
    )
    kinds = [row["kind"] for row in payload["coordinates"]]
    assert kinds == ["A", "D"]
    units = [row["unit"] for row in payload["coordinates"]]
    assert units == ["degree", "degree"]
    a_row, d_row = payload["coordinates"]
    assert a_row["atom_map_ids"] == [2, 1, 3]
    assert a_row["atom_indices"] == [1, 0, 2]
    assert d_row["atom_map_ids"] == [2, 1, 3, 4]
    assert d_row["atom_indices"] == [1, 0, 2, 3]
    assert a_row["values"] == list(compiled.coordinates[0].values)
    assert d_row["values"] == list(compiled.coordinates[1].values)
    assert all(len(row["values"]) == 9 for row in payload["coordinates"])


def test_multicoord_payload_preserves_nonuniform_per_point_lists(tmp_path) -> None:
    """SCHEDULED_1D per-point recipe: every driver keeps its non-uniform list."""
    compiled = _compile_scheduled_nonuniform(tmp_path)
    payload = multicoord_request_payload(
        compiled, _materials(_b2_geometry()), {"capability": _smoke(tmp_path)}
    )
    assert payload["metadata"]["recipe_kind"] == co.RECIPE_PER_POINT
    assert payload["protocol"]["custom_schedule"] is True
    assert len(payload["coordinates"]) == 2
    values_1 = payload["coordinates"][0]["values"]
    values_2 = payload["coordinates"][1]["values"]
    assert values_1 == list(compiled.coordinates[0].values)
    assert values_2 == list(compiled.coordinates[1].values)
    steps_1 = [round(b - a, 6) for a, b in zip(values_1, values_1[1:])]
    assert len(set(steps_1)) > 1
    assert len(payload["point_input_sha256"]) == compiled.total_points == 4
    assert len(set(payload["point_input_sha256"])) == 4  # one hash per point, distinct
    assert len(payload["geom_fragments"]) == 4
    assert all("Constraints" in fragment for fragment in payload["geom_fragments"])
    # Every constraint block carries BOTH drivers — no first-driver projection.
    for fragment in payload["geom_fragments"]:
        assert fragment.count("{ B ") == 2


# ---------------------------------------------------------------------------
# Capability gate: default registry refuses; probed receipts build.
# ---------------------------------------------------------------------------
def test_default_registry_refuses_coupled_multicoord_backend_capability_missing(tmp_path) -> None:
    """Default registry (adapter SINGLE_1D/B): COUPLED 2-B → BACKEND_CAPABILITY_MISSING."""
    compiled = _compile_2b(tmp_path)
    with pytest.raises(MulticoordAdapterError) as excinfo:
        multicoord_request_payload(
            compiled, _materials(_b2_geometry()), {"capability": _default_capability()}
        )
    assert excinfo.value.code == CODE_BACKEND_CAPABILITY_MISSING
    assert isinstance(excinfo.value, ACPMappingError)
    assert "COUPLED_1D" in str(excinfo.value) or "max_scan_coordinates" in str(excinfo.value)


def test_default_registry_refuses_a_and_d_kinds_backend_capability_missing(tmp_path) -> None:
    """A/D kinds are refused on the default registry — explicit typed refusal."""
    compiled_ad = _compile_ad(tmp_path)
    with pytest.raises(MulticoordAdapterError) as excinfo:
        multicoord_request_payload(
            compiled_ad, _materials(_ad_geometry()), {"capability": _default_capability()}
        )
    assert excinfo.value.code == CODE_BACKEND_CAPABILITY_MISSING

    capability = _smoke(tmp_path)
    single_a = _scan_candidate("SINGLE_1D", [_driver("A", [2, 1, 3])], _lambdas(9))
    compiled_a = co.compile_orca(single_a, ATOM_ROWS, capability, geometry=_ad_geometry())
    with pytest.raises(MulticoordAdapterError) as excinfo:
        multicoord_request_payload(
            compiled_a, _materials(_ad_geometry()), {"capability": _default_capability()}
        )
    assert excinfo.value.code == CODE_BACKEND_CAPABILITY_MISSING
    assert "coordinate kinds" in str(excinfo.value) or "A" in str(excinfo.value)


def test_default_registry_refuses_nonuniform_scheduled_backend_capability_missing(tmp_path) -> None:
    """SCHEDULED_1D non-uniform multi-coordinate: BACKEND_CAPABILITY_MISSING on default."""
    compiled = _compile_scheduled_nonuniform(tmp_path)
    with pytest.raises(MulticoordAdapterError) as excinfo:
        multicoord_request_payload(
            compiled, _materials(_b2_geometry()), {"capability": _default_capability()}
        )
    assert excinfo.value.code == CODE_BACKEND_CAPABILITY_MISSING


def test_default_registry_refuses_single_nonuniform_with_capability_unprobed(tmp_path) -> None:
    """Single-driver non-uniform B on default registry: declared-but-unprobed refusal."""
    compiled = _compile_single_nonuniform(tmp_path)
    with pytest.raises(MulticoordAdapterError) as excinfo:
        multicoord_request_payload(
            compiled, _materials(_b2_geometry()), {"capability": _default_capability()}
        )
    assert excinfo.value.code == CODE_CAPABILITY_UNPROBED
    assert "unprobed" in str(excinfo.value)


def test_probed_registry_builds_payload_and_refusal_never_downgrades(tmp_path) -> None:
    """With synthetic probe receipts the payload builds; without, refusal (no projection)."""
    probed = _smoke(tmp_path)
    compiled = _compile_2b(tmp_path)
    materials = _materials(_b2_geometry())
    payload = multicoord_request_payload(compiled, materials, {"capability": probed})
    assert len(payload["coordinates"]) == 2

    unprobed = _smoke(tmp_path, probed=False)
    with pytest.raises(MulticoordAdapterError) as excinfo:
        multicoord_request_payload(compiled, materials, {"capability": unprobed})
    assert excinfo.value.code in {CODE_BACKEND_CAPABILITY_MISSING, CODE_CAPABILITY_UNPROBED}
    assert "BACKEND_CAPABILITY_MISSING" in str(excinfo.value) or "CAPABILITY_UNPROBED" in str(
        excinfo.value
    )


def test_payload_refusal_when_simul_scan_flag_false_for_coupled(tmp_path) -> None:
    """Probed modes but adapter simul_scan=false → SIMUL_SCAN_UNSUPPORTED at payload layer."""
    compiled = _compile_2b(tmp_path)
    weak = _smoke(tmp_path, adapter_simul_scan=False)
    with pytest.raises(MulticoordAdapterError) as excinfo:
        multicoord_request_payload(compiled, _materials(_b2_geometry()), {"capability": weak})
    assert excinfo.value.code == CODE_SIMUL_SCAN_UNSUPPORTED


def test_payload_refusal_when_custom_schedule_false_for_per_point_recipe(tmp_path) -> None:
    """Per-point recipe + custom_schedule_support=false → CUSTOM_SCHEDULE_UNSUPPORTED."""
    compiled = _compile_scheduled_nonuniform(tmp_path)
    weak = _smoke(tmp_path, adapter_custom_schedule=False)
    with pytest.raises(MulticoordAdapterError) as excinfo:
        multicoord_request_payload(compiled, _materials(_b2_geometry()), {"capability": weak})
    assert excinfo.value.code == CODE_CUSTOM_SCHEDULE_UNSUPPORTED


def test_payload_refuses_compiled_atom_order_mismatch(tmp_path) -> None:
    """Compiled atom_rows must equal case materials atom_map_ids — no silent reorder."""
    compiled = _compile_2b(tmp_path)
    materials = _materials(_b2_geometry())
    reordered = {**materials, "atom_map_ids": [4, 3, 2, 1],
                 "r_coordinates": list(reversed(materials["r_coordinates"])),
                 "p_coordinates": list(reversed(materials["p_coordinates"]))}
    with pytest.raises(MulticoordAdapterError) as excinfo:
        multicoord_request_payload(compiled, reordered, {"capability": _smoke(tmp_path)})
    assert excinfo.value.code == "ATOM_ORDER_INVALID"


# ---------------------------------------------------------------------------
# Result parse surface: per-frame all-driver actuals (todo 21 extends).
# ---------------------------------------------------------------------------
def _raw_frames_2b(*, drop_driver02_actual_at: int | None = None) -> list[dict[str, Any]]:
    frames = []
    for index in range(3):
        actual = {"driver-01": 1.5 + 0.5 * index / 2, "driver-02": 1.0 + 1.0 * index / 2}
        if drop_driver02_actual_at is not None and index == drop_driver02_actual_at:
            actual.pop("driver-02")
        frames.append({
            "index": index,
            "optimization_converged": True,
            "target_coordinates": {"driver-01": 1.5 + 0.5 * index / 2,
                                   "driver-02": 1.0 + 1.0 * index / 2},
            "actual_coordinates": actual,
            "constraint_residuals": {"driver-01": 0.01, "driver-02": 0.02},
            "scan_energy_hartree": -40.0 - index / 10,
            "single_point_energy_hartree": None,
            "geometry_path": f"frames/f{index}.xyz",
            "retry_history": [],
        })
    return frames


def test_parse_multicoord_result_recovers_all_driver_actuals(tmp_path) -> None:
    compiled = _compile_2b(tmp_path)
    result = parse_multicoord_result(
        compiled=compiled, execution_id="exec-t20", acp_task_id="job-t20",
        raw_frames=_raw_frames_2b(),
    )
    assert result.schema_version == SCHEMA_MULTICOORD_RESULT
    assert result.candidate_id == "cand-t20"
    assert result.mode == "COUPLED_1D"
    assert result.driver_ids == ("driver-01", "driver-02")
    assert result.driver_units == {"driver-01": "angstrom", "driver-02": "angstrom"}
    assert result.n_frames == 3
    assert result.complete is True
    frame0 = result.frames[0]
    assert frame0.complete is True
    assert frame0.missing_driver_ids == ()
    assert frame0.target_by_driver == {"driver-01": 1.5, "driver-02": 1.0}
    assert frame0.actual_by_driver == {"driver-01": 1.5, "driver-02": 1.0}
    assert frame0.residual_by_driver == {"driver-01": 0.01, "driver-02": 0.02}
    assert frame0.scan_energy_hartree == pytest.approx(-40.0)
    assert frame0.single_point_energy_hartree is None
    # Every frame carries BOTH drivers — never first-driver-only.
    for frame in result.frames:
        assert set(frame.actual_by_driver) == {"driver-01", "driver-02"}


def test_parse_multicoord_result_marks_missing_driver_incomplete(tmp_path) -> None:
    """A frame missing one driver's actual → explicit incomplete, value None."""
    compiled = _compile_2b(tmp_path)
    result = parse_multicoord_result(
        compiled=compiled, execution_id="exec-t20", acp_task_id=None,
        raw_frames=_raw_frames_2b(drop_driver02_actual_at=1),
    )
    assert result.complete is False
    frame1 = next(frame for frame in result.frames if frame.frame_index == 1)
    assert frame1.complete is False
    assert frame1.missing_driver_ids == ("driver-02",)
    assert frame1.actual_by_driver["driver-02"] is None
    assert frame1.actual_by_driver["driver-01"] is not None


def test_parse_multicoord_result_rejects_structural_frame_issues(tmp_path) -> None:
    compiled = _compile_2b(tmp_path)
    base = _raw_frames_2b()
    duplicated = copy.deepcopy(base)
    duplicated[1]["index"] = 0
    with pytest.raises(MulticoordAdapterError) as excinfo:
        parse_multicoord_result(
            compiled=compiled, execution_id="exec", acp_task_id=None, raw_frames=duplicated
        )
    assert excinfo.value.code == "FRAMES_INVALID"
    bad_index = copy.deepcopy(base)
    bad_index[0]["index"] = -1
    with pytest.raises(MulticoordAdapterError) as excinfo:
        parse_multicoord_result(
            compiled=compiled, execution_id="exec", acp_task_id=None, raw_frames=bad_index
        )
    assert excinfo.value.code == "FRAMES_INVALID"
    bad_value = copy.deepcopy(base)
    bad_value[0]["actual_coordinates"]["driver-01"] = "not-a-number"
    with pytest.raises(MulticoordAdapterError) as excinfo:
        parse_multicoord_result(
            compiled=compiled, execution_id="exec", acp_task_id=None, raw_frames=bad_value
        )
    assert excinfo.value.code == "FRAMES_INVALID"


def test_multicoord_surfaces_are_deterministic_and_truth_free(tmp_path) -> None:
    compiled = _compile_2b(tmp_path)
    materials = _materials(_b2_geometry())
    capability = _smoke(tmp_path)
    payload_a = multicoord_request_payload(compiled, materials, {"capability": capability})
    payload_b = multicoord_request_payload(compiled, materials, {"capability": capability})
    assert stable_json_dumps(payload_a) == stable_json_dumps(payload_b)
    result = parse_multicoord_result(
        compiled=compiled, execution_id="exec", acp_task_id="job",
        raw_frames=_raw_frames_2b(),
    )
    assert result.to_json() == result.to_json()
    blob = stable_json_dumps(payload_a) + result.to_json()
    for forbidden in ("orientation", "irc_evidence", "endpoint_match", "ground_truth"):
        assert forbidden not in blob


# ---------------------------------------------------------------------------
# Existing adapter regression: explicit refusals + byte-identical pin.
# ---------------------------------------------------------------------------
def test_current_adapter_refuses_multicoordinate_scan_plan() -> None:
    """Existing adapter refuses multi-coordinate plans explicitly (no downgrade)."""
    case = _simple_case()

    def add_second_coordinate(plan: dict[str, Any]) -> None:
        plan["candidates"][0]["coordinates"].append({
            "coordinate_id": "driver-02", "role": "driver", "atom_map_ids": [7, 12],
            "atom_indices": [1, 2], "unit": "angstrom", "direction": "stretch",
            "points": list(plan["candidates"][0]["coordinates"][0]["points"]),
        })

    plan = _tampered_ready_plan(case, add_second_coordinate)
    with pytest.raises(ACPMappingError, match="single distance coordinate"):
        scan_plan_to_acp_request(case, plan)


def test_current_adapter_refuses_nonuniform_points_scan_plan() -> None:
    """Existing adapter refuses non-uniform point lists explicitly."""
    case = _simple_case()

    def make_nonuniform(plan: dict[str, Any]) -> None:
        points = plan["candidates"][0]["coordinates"][0]["points"]
        plan["candidates"][0]["coordinates"][0]["points"] = [
            points[0], points[0] + 0.05, points[-1] - 0.2, points[-1] - 0.05, points[-1],
        ]

    plan = _tampered_ready_plan(case, make_nonuniform)
    with pytest.raises(ACPMappingError, match="non-uniform plan needs an adapter extension"):
        scan_plan_to_acp_request(case, plan)


def test_adapter_regression_scan_plan_to_acp_request_byte_identical() -> None:
    """scan_plan_to_acp_request output equals the pre-change frozen structure.

    Expected values are derived from existing tests (test_contracts.py
    simple_case + build_minimal_scan_plan); the adapter module is byte-frozen
    so both the structure and the stable-serialization digest must hold.
    """
    case = _simple_case()
    plan = build_minimal_scan_plan(case)
    request = scan_plan_to_acp_request(case, plan)

    expected = {
        "adapter_version": "pes2ts_acp_mapping_v1",
        "metadata": {
            "atom_map_ids": [2, 7, 12],
            "budget": {"max_attempts": 1, "max_cpu_hours": 8.0, "max_wall_seconds": 14400},
            "candidate_id": "plan-1a106bad5c60:c001",
            "case_id": "case:rxn-demo-v1",
            "elements": ["C", "H", "N"],
            "experiment_id": "demo-experiment-v1",
            "plan_id": "plan-1a106bad5c60",
            "planned_method": {
                "basis": None, "engine": "xtb", "engine_version": None,
                "method": "GFN2-xTB", "parameter_sha256": None, "solvent": None,
            },
            "reaction_id": "RXN_DEMO_0001",
            "request_id": "plan-1a106bad5c60:c001:attempt1",
            "retry_policy": {
                "failure_policy": "retry_previous", "optimizer_retries": 1,
                "reuse_previous_geometry": True, "scan_retry_count": 1,
            },
        },
        "method_levels": {
            "scan_coordinate": {
                "scan_bond_type": "auto", "scan_coordinate_end": 2.4,
                "scan_coordinate_kind": "distance", "scan_coordinate_points": 9,
                "scan_coordinate_start": 1.0,
            },
            "scan_driver": {
                "scan_failure_policy": "retry_previous", "scan_mode": "relaxed_scan",
                "scan_retry_count": 1, "scan_reuse_previous_geometry": True,
            },
            "scan_optimizer": {
                "scan_optimizer_method": "GFN2-xTB", "scan_optimizer_retries": 1,
                "scan_optimizer_retry_strategy": "previous_geometry",
            },
        },
        "scan_request": {
            "coordinate": {
                "atoms": [1, 2], "end": 2.4, "kind": "distance", "n_points": 9,
                "start": 1.0, "unit": "angstrom",
            },
            "mode": "bond_length_scan",
            "protocol": {
                "name": "pes2ts_scan_plan_v1",
                "scan_driver": {
                    "failure_policy": "retry_previous", "mode": "relaxed_scan",
                    "software": "xtb",
                },
                "scan_optimizer": {
                    "method": "GFN2-xTB", "retry_count": 1,
                    "retry_strategy": "previous_geometry",
                },
                "scan_type": "bond_length",
                "single_point": {"enabled": False},
            },
            "source": {
                "charge": 0, "multiplicity": 1, "source_type": "xyz_text",
                "xyz_text": (
                    "3\nPES2TS plan-1a106bad5c60 plan-1a106bad5c60:c001\n"
                    "C 0.0000000000 0.0000000000 0.0000000000\n"
                    "H 1.8000000000 0.0000000000 0.0000000000\n"
                    "N 2.8000000000 0.0000000000 0.0000000000\n"
                ),
            },
        },
    }
    assert request == expected
    digest = hashlib.sha256(stable_json_dumps(request).encode("utf-8")).hexdigest()
    assert digest == GOLDEN_ADAPTER_REQUEST_SHA256
    # Existing single-distance assertions from test_contracts.py stay valid.
    assert request["scan_request"]["coordinate"]["atoms"] == [1, 2]
    assert request["method_levels"]["scan_driver"]["scan_retry_count"] == 1
    assert request["metadata"]["retry_policy"]["reuse_previous_geometry"] is True


def test_shipped_registry_capability_check_shape_for_multi_and_ad(tmp_path) -> None:
    """Default-registry capability_check shapes back the typed refusals above."""
    default = _default_capability()
    assert default.max_scan_coordinates == 1
    assert default.coordinate_kinds == frozenset({"B"})
    assert default.supported_modes == frozenset({"SINGLE_1D"})
    assert caps.capability_check(default, "COUPLED_1D", ["B", "B"], 2)["status"] == "fail"
    assert caps.capability_check(default, "SINGLE_1D", ["A"], 1)["status"] == "fail"
    assert caps.capability_check(default, "SINGLE_1D", ["D"], 1)["status"] == "fail"
    assert caps.capability_check(default, "SCHEDULED_1D", ["B", "B"], 2)["status"] == "fail"
    # Declared-but-unprobed single B remains "unknown" — never optimistic "pass".
    assert caps.capability_check(default, "SINGLE_1D", ["B"], 1)["status"] == "unknown"


def test_multicoord_module_does_not_modify_adapter_contract_surface() -> None:
    """New module lives beside adapter.py; adapter public names stay importable."""
    import pes2ts_core.integration.acp.adapter as adapter_module
    import pes2ts_core.integration.acp.multicoord as multicoord_module

    assert hasattr(adapter_module, "scan_plan_to_acp_request")
    assert adapter_module.scan_plan_to_acp_request is not None
    assert multicoord_module.multicoord_request_payload is not None
    # Payload projection is a distinct function, not a monkeypatch of the old one.
    assert multicoord_module.multicoord_request_payload is not adapter_module.scan_plan_to_acp_request
