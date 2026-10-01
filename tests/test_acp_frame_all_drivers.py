"""Tests for pes2ts_core.integration.acp.frame_recovery (plan todo 21).

Covers: multi-driver per-frame target/actual/residual traces (targets from the
compiled schedule, actuals from the todo-20 parse surface); incomplete marking
when ANY driver residual is missing (never silently passed); monitor
quantities per frame; non-target contacts; separated scan vs single-point
energy channels; electronic-state diagnostics placeholder; retry-history
channel; single-driver backward-compatible PathBundle extension shape; multi-
driver additive-only extension; determinism; truth-key purity; and the frozen
single-distance module public surfaces staying intact.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from pes2ts_core.integration.acp.frame_recovery import (
    CODE_DRIVER_COVERAGE,
    CODE_DRIVER_MISMATCH,
    ENERGY_CHANNEL_NONE,
    ENERGY_CHANNEL_SCAN,
    ENERGY_CHANNEL_SINGLE_POINT,
    PATH_BUNDLE_EXTENSION_KEY,
    SCHEMA_FRAME_RECOVERY,
    SCHEMA_PATH_BUNDLE_EXTENSION,
    FrameRecoveryError,
    FrameRecoveryReport,
    recover_frames,
    to_path_bundle_extension,
)
from pes2ts_core.integration.acp.multicoord import (
    SCHEMA_MULTICOORD_RESULT,
    driver_id_for,
    parse_multicoord_result,
)
from pes2ts_core.scan_strategy import capabilities as caps
from pes2ts_core.scan_strategy import compile_orca as co
from pes2ts_core.utils.hashing import stable_json_dumps

# ---------------------------------------------------------------------------
# Capability-registry fixture (synthetic probe receipts; todo-16/20 pattern).
# ---------------------------------------------------------------------------
ALL_MODES = ["SINGLE_1D", "COUPLED_1D", "SCHEDULED_1D", "PATH_NEB"]
ALL_KINDS = ["B", "A", "D"]


def _receipt(modes: list[str]) -> dict[str, Any]:
    return {
        "probe_id": "probe-t21",
        "date": "2026-10-01",
        "engine_version": "6.1.0",
        "adapter_version": "acp-pes-scan-v0",
        "status": "pass",
        "evidence_ref": "evidence/task-21",
        "modes": list(modes),
    }


def _entry(entry_id: str, layer: str) -> dict[str, Any]:
    receipts = [_receipt(ALL_MODES)]
    return {
        "entry_id": entry_id,
        "layer": layer,
        "engine": "orca",
        "adapter": "acp",
        "capability": {
            "engine_version": "6.1.0",
            "adapter_version": "acp-pes-scan-v0",
            "supported_modes": list(ALL_MODES),
            "coordinate_kinds": list(ALL_KINDS),
            "max_scan_coordinates": 3,
            "point_limits": {"baseline": 2, "max": 101},
            "custom_schedule_support": True,
            "constraint_support": {
                "native_scan": True,
                "per_point_constraints": True,
                "simul_scan": True,
            },
            "method_element_coverage": {"B3LYP-D3": ["C", "H", "N", "O"]},
            "probe_receipts": receipts,
        },
        "notes": "todo-21 synthetic stack",
    }


def _capability(tmp_path: Path) -> caps.EffectiveCapability:
    payload = {
        "schema_version": caps.REGISTRY_SCHEMA_VERSION,
        "registry_id": "todo21-test-stack",
        "description": "synthetic probed stack for frame recovery tests",
        "entries": [_entry("eng", "engine"), _entry("adp", "adapter"),
                    _entry("dep", "deployment")],
    }
    path = tmp_path / "orca_capabilities_v1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    registry = caps.load_capability_registry(path)
    return caps.effective_capability("eng", "adp", "dep", registry=registry)


# ---------------------------------------------------------------------------
# Candidate / geometry / materials fixtures (todo-20 shapes).
# ---------------------------------------------------------------------------
ATOM_ROWS = (1, 2, 3, 4)
ELEMENTS = ["C", "C", "H", "H"]


def _b2_geometry() -> dict[str, dict[int, tuple[float, float, float]]]:
    """B(1,2): 1.5→2.5 Å; B(3,4): 1.0→2.0 Å."""
    return {
        "R": {1: (0.0, 0.0, 0.0), 2: (1.5, 0.0, 0.0), 3: (0.0, 0.0, 2.0), 4: (0.0, 0.0, 3.0)},
        "P": {1: (0.0, 0.0, 0.0), 2: (2.5, 0.0, 0.0), 3: (0.0, 0.0, 2.0), 4: (0.0, 0.0, 4.0)},
    }


def _geometry_rows(
    geom: dict[str, dict[int, tuple[float, float, float]]], side: str
) -> list[list[float]]:
    return [list(geom[side][map_id]) for map_id in ATOM_ROWS]


def _materials(geom: dict[str, dict[int, tuple[float, float, float]]]) -> dict[str, Any]:
    return {
        "atom_map_ids": list(ATOM_ROWS),
        "elements": list(ELEMENTS),
        "r_coordinates": _geometry_rows(geom, "R"),
        "p_coordinates": _geometry_rows(geom, "P"),
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
    candidate_id: str = "cand-t21",
) -> dict[str, Any]:
    candidate: dict[str, Any] = {
        "candidate_kind": "ScanCandidateV2",
        "candidate_id": candidate_id,
        "mode": mode,
        "start_endpoint": "R",
        "direction": "R_to_P",
        "assembly_id": "asm-t21",
        "anchor_reason": "HAND_TEST",
        "drivers": drivers,
        "monitors": [],
        "guards": [],
        "event_coverage": [],
        "lambda_values": list(lambda_values),
        "schedule_id": "sched:t21",
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
    candidate = _scan_candidate(
        "COUPLED_1D", [_driver("B", [1, 2]), _driver("B", [3, 4])], _lambdas(5)
    )
    return co.compile_orca(
        candidate, ATOM_ROWS, _capability(tmp_path), geometry=_b2_geometry()
    )


def _compile_single(tmp_path: Path) -> Any:
    candidate = _scan_candidate("SINGLE_1D", [_driver("B", [1, 2])], _lambdas(5))
    return co.compile_orca(
        candidate, ATOM_ROWS, _capability(tmp_path), geometry=_b2_geometry()
    )


def _raw_frames_2b(
    *,
    drop_driver02_actual_at: int | None = None,
    with_single_point: bool = False,
    with_retry: bool = False,
    energies_none_at: int | None = None,
) -> list[dict[str, Any]]:
    frames = []
    for index in range(3):
        actual: dict[str, float | None] = {
            "driver-01": 1.5 + 0.5 * index / 2,
            "driver-02": 1.0 + 1.0 * index / 2,
        }
        if drop_driver02_actual_at is not None and index == drop_driver02_actual_at:
            actual.pop("driver-02")
        scan_energy = None if energies_none_at == index else -40.0 - index / 10
        frames.append({
            "index": index,
            "optimization_converged": True,
            "target_coordinates": {
                "driver-01": 1.5 + 0.5 * index / 2,
                "driver-02": 1.0 + 1.0 * index / 2,
            },
            "actual_coordinates": actual,
            "constraint_residuals": {"driver-01": 0.01, "driver-02": 0.02},
            "scan_energy_hartree": scan_energy,
            "single_point_energy_hartree": (-39.9 - index / 10) if with_single_point else None,
            "geometry_path": f"frames/f{index}.xyz",
            "retry_history": (
                [{"attempt": 1, "note": "scf_restart"}] if with_retry and index == 1 else []
            ),
        })
    return frames


def _raw_frames_single() -> list[dict[str, Any]]:
    frames = []
    for index in range(3):
        value = 1.5 + 0.25 * index
        frames.append({
            "index": index,
            "optimization_converged": True,
            "target_coordinates": {"driver-01": value},
            "actual_coordinates": {"driver-01": value - 0.01},
            "constraint_residuals": {"driver-01": 0.01},
            "scan_energy_hartree": -40.0 - index / 10,
            "single_point_energy_hartree": None,
            "geometry_path": f"frames/s{index}.xyz",
            "retry_history": [],
        })
    return frames


def _parse(compiled: Any, raw_frames: list[dict[str, Any]]) -> Any:
    return parse_multicoord_result(
        compiled=compiled,
        execution_id="exec-t21",
        acp_task_id="job-t21",
        raw_frames=raw_frames,
    )


def _monitor_records() -> list[dict[str, Any]]:
    """Coordinate-pool-shaped monitor records (B distance + A angle)."""
    return [
        {"coordinate_id": "coord-0001", "role": "monitor", "kind": "B",
         "atom_maps": (1, 3), "units": "angstrom", "event_ids": ("ev-1",)},
        {"coordinate_id": "coord-0002", "role": "monitor", "kind": "A",
         "atom_maps": (2, 1, 3), "units": "degree", "event_ids": ("ev-2",)},
    ]


# ---------------------------------------------------------------------------
# All-driver target/actual/residual traces.
# ---------------------------------------------------------------------------
def test_recover_frames_traces_all_driver_targets_actuals_residuals(tmp_path) -> None:
    """Every frame × every driver carries target, actual, and residual."""
    compiled = _compile_2b(tmp_path)
    result = _parse(compiled, _raw_frames_2b())
    report = recover_frames(result, compiled)

    assert isinstance(report, FrameRecoveryReport)
    assert report.schema_version == SCHEMA_FRAME_RECOVERY
    assert report.driver_ids == ("driver-01", "driver-02")
    assert report.n_frames == 3
    assert report.complete is True
    assert report.incomplete_frame_indices == ()
    for record in report.frames:
        assert set(record.targets_by_driver) == {"driver-01", "driver-02"}
        assert set(record.actuals_by_driver) == {"driver-01", "driver-02"}
        assert set(record.residuals_by_driver) == {"driver-01", "driver-02"}
        for driver_id in report.driver_ids:
            target = record.targets_by_driver[driver_id]
            actual = record.actuals_by_driver[driver_id]
            residual = record.residuals_by_driver[driver_id]
            assert target is not None and actual is not None and residual is not None
            assert residual == pytest.approx(target - actual)
        assert record.incomplete is False


def test_recover_frames_targets_come_from_compiled_schedule(tmp_path) -> None:
    """Schedule-λ targets come from the compiled per-point values, not first-driver."""
    compiled = _compile_2b(tmp_path)
    result = _parse(compiled, _raw_frames_2b())
    report = recover_frames(result, compiled)
    for record in report.frames:
        for position, driver_id in enumerate(report.driver_ids):
            row = compiled.coordinates[position]
            assert record.targets_by_driver[driver_id] == pytest.approx(
                row.values[record.frame_index]
            )
        assert record.lambda_value == pytest.approx(
            compiled.lambda_values[record.frame_index]
        )
        assert record.stage_id is None  # plain scan has no stage id
        assert record.geometry_ref == f"frames/f{record.frame_index}.xyz"


def test_recover_frames_driver_mismatch_refused(tmp_path) -> None:
    """Result drivers must equal compiled drivers — no first-driver projection."""
    compiled_2b = _compile_2b(tmp_path)
    compiled_single = _compile_single(tmp_path)
    result_2b = _parse(compiled_2b, _raw_frames_2b())
    with pytest.raises(FrameRecoveryError) as excinfo:
        recover_frames(result_2b, compiled_single)
    assert excinfo.value.code == CODE_DRIVER_MISMATCH

    result_single = _parse(compiled_single, _raw_frames_single())
    with pytest.raises(FrameRecoveryError) as excinfo:
        recover_frames(result_single, compiled_2b)
    assert excinfo.value.code == CODE_DRIVER_MISMATCH


def test_recover_frames_never_first_driver_only_outputs(tmp_path) -> None:
    """Outputs structurally cover every driver (assertion tripwire armed)."""
    compiled = _compile_2b(tmp_path)
    report = recover_frames(_parse(compiled, _raw_frames_2b()), compiled)
    assert report.source["driver_ids"] == ["driver-01", "driver-02"]
    assert report.source["n_drivers"] == 2
    for record in report.frames:
        doc = record.to_doc()
        assert set(doc["targets_by_driver"]) == {"driver-01", "driver-02"}
        assert set(doc["actuals_by_driver"]) == {"driver-01", "driver-02"}
        assert set(doc["residuals_by_driver"]) == {"driver-01", "driver-02"}
        assert set(doc["backend_residuals_by_driver"]) == {"driver-01", "driver-02"}


def test_recover_frames_zero_frames_is_incomplete(tmp_path) -> None:
    compiled = _compile_2b(tmp_path)
    report = recover_frames(_parse(compiled, []), compiled)
    assert report.n_frames == 0
    assert report.complete is False
    assert report.frames == ()


# ---------------------------------------------------------------------------
# Incomplete marking: missing ANY driver residual → incomplete, never silent.
# ---------------------------------------------------------------------------
def test_recover_frames_marks_incomplete_when_any_driver_residual_missing(tmp_path) -> None:
    """Frame missing one driver's actual → incomplete; residual not silently passed."""
    compiled = _compile_2b(tmp_path)
    raw = _raw_frames_2b(drop_driver02_actual_at=1)
    result = _parse(compiled, raw)
    assert result.complete is False  # todo-20 parse surface already flags it

    report = recover_frames(result, compiled)
    assert report.complete is False
    assert report.incomplete_frame_indices == (1,)
    frame1 = next(record for record in report.frames if record.frame_index == 1)
    assert frame1.incomplete is True
    # Todo-20 missing_driver_ids semantics: target or actual absent.
    assert frame1.missing_driver_ids == ("driver-02",)
    # Residual channel: uncomputable for the missing driver — named, not silent.
    assert frame1.missing_residual_driver_ids == ("driver-02",)
    assert frame1.residuals_by_driver["driver-02"] is None
    assert frame1.residuals_by_driver["driver-01"] is not None
    # Other frames stay complete — one missing driver does not blank the path.
    frame0 = next(record for record in report.frames if record.frame_index == 0)
    assert frame0.incomplete is False
    assert frame0.missing_driver_ids == ()


def test_recover_frames_incomplete_when_backend_residual_only_but_actual_missing(
    tmp_path,
) -> None:
    """Backend residual present but actual absent → still incomplete (target−actual undefined)."""
    compiled = _compile_2b(tmp_path)
    raw = _raw_frames_2b(drop_driver02_actual_at=2)
    raw[2]["constraint_residuals"] = {"driver-01": 0.01, "driver-02": 0.02}
    report = recover_frames(_parse(compiled, raw), compiled)
    frame2 = next(record for record in report.frames if record.frame_index == 2)
    assert frame2.incomplete is True
    assert frame2.missing_residual_driver_ids == ("driver-02",)
    assert frame2.backend_residuals_by_driver["driver-02"] == 0.02
    assert frame2.residuals_by_driver["driver-02"] is None
    assert report.incomplete_frame_indices == (2,)


# ---------------------------------------------------------------------------
# Monitor quantities per frame.
# ---------------------------------------------------------------------------
def test_recover_frames_records_monitors_per_frame_with_geometry(tmp_path) -> None:
    compiled = _compile_2b(tmp_path)
    result = _parse(compiled, _raw_frames_2b())
    geom = _b2_geometry()
    report = recover_frames(
        result,
        compiled,
        _monitor_records(),
        geometries={0: _geometry_rows(geom, "R")},
    )
    assert report.monitor_coordinate_ids == ("coord-0001", "coord-0002")
    frame0 = report.frames[0]
    assert [row.coordinate_id for row in frame0.monitors] == ["coord-0001", "coord-0002"]
    b_row = frame0.monitors[0]
    assert b_row.kind == "B" and b_row.role == "monitor"
    assert b_row.measured is True
    # B(1,3) on the R geometry: map1 (0,0,0) → map3 (0,0,2) = 2.0 Å.
    assert b_row.value == pytest.approx(2.0)
    a_row = frame0.monitors[1]
    assert a_row.kind == "A" and a_row.measured is True
    # A(2,1,3): (1.5,0,0)-(0,0,0)-(0,0,2) is a right angle.
    assert a_row.value == pytest.approx(90.0)
    # Frames without geometry keep explicit unmeasured rows — never dropped.
    frame1 = report.frames[1]
    assert [row.measured for row in frame1.monitors] == [False, False]
    assert all(row.value is None for row in frame1.monitors)


def test_recover_frames_monitors_unmeasured_without_geometry(tmp_path) -> None:
    compiled = _compile_2b(tmp_path)
    report = recover_frames(
        _parse(compiled, _raw_frames_2b()), compiled, _monitor_records()
    )
    assert report.n_frames == 3
    for record in report.frames:
        assert len(record.monitors) == 2
        assert all(row.measured is False and row.value is None for row in record.monitors)


# ---------------------------------------------------------------------------
# Non-target contacts.
# ---------------------------------------------------------------------------
def test_recover_frames_records_non_target_contacts(tmp_path) -> None:
    """Nonbonded pairs below threshold that are not driver targets are recorded."""
    compiled = _compile_2b(tmp_path)
    # Crafted geometry: driver pairs stay far apart, but maps 2 and 3 are 0.5 Å.
    close_geometry = {
        1: [0.0, 0.0, 0.0],
        2: [1.5, 0.0, 0.0],
        3: [1.5, 0.5, 0.0],   # 0.5 Å from map 2 — not a driver pair ({1,2},{3,4})
        4: [0.0, 0.0, 5.0],
    }
    report = recover_frames(
        _parse(compiled, _raw_frames_2b()),
        compiled,
        geometries={0: [close_geometry[map_id] for map_id in ATOM_ROWS]},
        contact_threshold_angstrom=0.8,
    )
    frame0 = report.frames[0]
    assert frame0.contacts_measured is True
    contacts = {(row.map_a, row.map_b): row.distance_angstrom for row in frame0.non_target_contacts}
    assert (2, 3) in contacts
    assert contacts[(2, 3)] == pytest.approx(0.5)
    # Driver target pairs are excluded even if close.
    assert (1, 2) not in contacts
    assert (3, 4) not in contacts


def test_recover_frames_no_contacts_without_geometry(tmp_path) -> None:
    compiled = _compile_2b(tmp_path)
    report = recover_frames(_parse(compiled, _raw_frames_2b()), compiled)
    for record in report.frames:
        assert record.contacts_measured is False
        assert record.non_target_contacts == ()


# ---------------------------------------------------------------------------
# Energy channels separated; electronic-state placeholder; retry history.
# ---------------------------------------------------------------------------
def test_energy_channels_separate_scan_and_single_point(tmp_path) -> None:
    compiled = _compile_2b(tmp_path)
    report = recover_frames(
        _parse(compiled, _raw_frames_2b(with_single_point=True)), compiled
    )
    for record in report.frames:
        channels = {row.energy_channel: row.value for row in record.energies}
        assert set(channels) == {ENERGY_CHANNEL_SCAN, ENERGY_CHANNEL_SINGLE_POINT}
        assert channels[ENERGY_CHANNEL_SCAN] == pytest.approx(-40.0 - record.frame_index / 10)
        assert channels[ENERGY_CHANNEL_SINGLE_POINT] == pytest.approx(
            -39.9 - record.frame_index / 10
        )
        # Values live in distinct rows — never merged into one channel.
        scan_rows = [row for row in record.energies if row.energy_channel == ENERGY_CHANNEL_SCAN]
        sp_rows = [row for row in record.energies if row.energy_channel == ENERGY_CHANNEL_SINGLE_POINT]
        assert len(scan_rows) == 1 and len(sp_rows) == 1


def test_energy_channels_none_when_no_energies(tmp_path) -> None:
    compiled = _compile_2b(tmp_path)
    report = recover_frames(
        _parse(compiled, _raw_frames_2b(energies_none_at=0)), compiled
    )
    frame0 = report.frames[0]
    assert [row.energy_channel for row in frame0.energies] == [ENERGY_CHANNEL_NONE]
    assert frame0.energies[0].value is None
    # Other frames keep their scan channel.
    frame1 = report.frames[1]
    assert [row.energy_channel for row in frame1.energies] == [ENERGY_CHANNEL_SCAN]


def test_electronic_state_diagnostics_placeholder_channel(tmp_path) -> None:
    compiled = _compile_2b(tmp_path)
    report = recover_frames(_parse(compiled, _raw_frames_2b()), compiled)
    assert report.electronic_state_diagnostics["status"] == "not_recorded"
    assert report.electronic_state_diagnostics["charge"] == 0
    assert report.electronic_state_diagnostics["multiplicity"] == 1
    for record in report.frames:
        assert record.electronic_state_diagnostics["status"] == "not_recorded"
        assert record.electronic_state_diagnostics["channel"] == "electronic_state_diagnostics"
        assert record.electronic_state_diagnostics["spin_contamination"] is None


def test_retry_history_channel_carried(tmp_path) -> None:
    compiled = _compile_2b(tmp_path)
    report = recover_frames(
        _parse(compiled, _raw_frames_2b(with_retry=True)), compiled
    )
    frame1 = report.frames[1]
    assert frame1.retry_history == ({"attempt": 1, "note": "scf_restart"},)
    frame0 = report.frames[0]
    assert frame0.retry_history == ()


# ---------------------------------------------------------------------------
# PathBundle v2 extension: single-driver backward compat + additive multi.
# ---------------------------------------------------------------------------
def test_single_driver_backward_compatible_extension(tmp_path) -> None:
    """Single-B report → extension keeps OLD PathBundle frame field semantics."""
    compiled = _compile_single(tmp_path)
    report = recover_frames(_parse(compiled, _raw_frames_single()), compiled)
    extension = to_path_bundle_extension(report)

    assert extension["schema_version"] == SCHEMA_PATH_BUNDLE_EXTENSION
    assert extension["extension_key"] == PATH_BUNDLE_EXTENSION_KEY
    assert extension["n_drivers"] == 1
    assert extension["driver_ids"] == ["driver-01"]
    assert extension["complete"] is True
    assert extension["semantics"]["additive_only"] is True

    for frame_doc, record in zip(extension["frames"], report.frames, strict=True):
        # OLD single-driver fields — historical PathBundle meaning intact.
        assert frame_doc["target_coordinate"] == pytest.approx(
            record.targets_by_driver["driver-01"]
        )
        assert frame_doc["actual_coordinate"] == pytest.approx(
            record.actuals_by_driver["driver-01"]
        )
        assert frame_doc["constraint_residuals"] == record.backend_residuals_by_driver
        assert frame_doc["converged"] is True
        assert frame_doc["geometry_ref"] == record.geometry_ref
        # NEW named fields exist even for a single driver (additive surface).
        assert frame_doc["all_driver_targets"] == {"driver-01": record.targets_by_driver["driver-01"]}
        assert frame_doc["all_driver_actuals"] == {"driver-01": record.actuals_by_driver["driver-01"]}
        assert frame_doc["all_driver_residuals"] == {
            "driver-01": record.residuals_by_driver["driver-01"]
        }
        assert frame_doc["incomplete"] is False
        assert frame_doc["energy_channels"] == [
            {"energy_channel": ENERGY_CHANNEL_SCAN,
             "value": record.energies[0].value, "method_id": None}
        ]


def test_multi_driver_extension_additive_only(tmp_path) -> None:
    """Multi-driver: OLD first-driver fields unchanged; new named fields added."""
    compiled = _compile_2b(tmp_path)
    report = recover_frames(_parse(compiled, _raw_frames_2b()), compiled)
    extension = to_path_bundle_extension(report)

    assert extension["n_drivers"] == 2
    assert extension["driver_ids"] == ["driver-01", "driver-02"]
    old_fields = set(extension["semantics"]["old_single_driver_fields"])
    new_fields = set(extension["semantics"]["new_named_fields"])
    assert old_fields & new_fields == set()  # additive names never collide

    for frame_doc, record in zip(extension["frames"], report.frames, strict=True):
        # OLD semantics preserved: singular fields follow the FIRST driver.
        assert frame_doc["target_coordinate"] == pytest.approx(
            record.targets_by_driver["driver-01"]
        )
        assert frame_doc["actual_coordinate"] == pytest.approx(
            record.actuals_by_driver["driver-01"]
        )
        assert frame_doc["constraint_residuals"] == record.backend_residuals_by_driver
        # NEW named multi-driver fields cover ALL drivers.
        assert set(frame_doc["all_driver_targets"]) == {"driver-01", "driver-02"}
        assert set(frame_doc["all_driver_actuals"]) == {"driver-01", "driver-02"}
        assert set(frame_doc["all_driver_residuals"]) == {"driver-01", "driver-02"}
        assert frame_doc["all_driver_targets"]["driver-02"] == pytest.approx(
            record.targets_by_driver["driver-02"]
        )
        # Old field names appear exactly once in the frame dict (no overwrite).
        assert list(frame_doc).count("target_coordinate") == 1
        assert list(frame_doc).count("all_driver_targets") == 1


def test_extension_never_first_driver_only(tmp_path) -> None:
    compiled = _compile_2b(tmp_path)
    report = recover_frames(_parse(compiled, _raw_frames_2b()), compiled)
    extension = to_path_bundle_extension(report)
    for frame_doc in extension["frames"]:
        for field in ("all_driver_targets", "all_driver_actuals", "all_driver_residuals"):
            assert set(frame_doc[field]) == {"driver-01", "driver-02"}


def test_extension_carries_incomplete_frame(tmp_path) -> None:
    compiled = _compile_2b(tmp_path)
    report = recover_frames(
        _parse(compiled, _raw_frames_2b(drop_driver02_actual_at=1)), compiled
    )
    extension = to_path_bundle_extension(report)
    assert extension["complete"] is False
    assert extension["incomplete_frame_indices"] == [1]
    frame1 = next(doc for doc in extension["frames"] if doc["frame_index"] == 1)
    assert frame1["incomplete"] is True
    assert frame1["missing_driver_ids"] == ["driver-02"]
    assert frame1["missing_residual_driver_ids"] == ["driver-02"]
    assert frame1["all_driver_residuals"]["driver-02"] is None


def test_extension_refuses_non_report_input(tmp_path) -> None:
    compiled = _compile_2b(tmp_path)
    report = recover_frames(_parse(compiled, _raw_frames_2b()), compiled)
    with pytest.raises(FrameRecoveryError):
        to_path_bundle_extension(report.to_doc())  # mapping is not a report
    # Tampered report (empty drivers) is refused, not silently first-driver.
    tampered = FrameRecoveryReport(
        schema_version=report.schema_version,
        candidate_id=report.candidate_id,
        mode=report.mode,
        execution_id=report.execution_id,
        acp_task_id=report.acp_task_id,
        driver_ids=(),
        driver_units={},
        n_frames=report.n_frames,
        complete=False,
        frames=report.frames,
        incomplete_frame_indices=report.incomplete_frame_indices,
        monitor_coordinate_ids=report.monitor_coordinate_ids,
        electronic_state_diagnostics=report.electronic_state_diagnostics,
        source=report.source,
    )
    with pytest.raises(FrameRecoveryError) as excinfo:
        to_path_bundle_extension(tampered)
    assert excinfo.value.code == CODE_DRIVER_COVERAGE


# ---------------------------------------------------------------------------
# Determinism, purity, frozen-module surface.
# ---------------------------------------------------------------------------
def test_deterministic_recovery_and_extension(tmp_path) -> None:
    compiled = _compile_2b(tmp_path)
    raw = _raw_frames_2b(with_single_point=True, with_retry=True)
    report_a = recover_frames(_parse(compiled, copy.deepcopy(raw)), compiled, _monitor_records())
    report_b = recover_frames(_parse(compiled, copy.deepcopy(raw)), compiled, _monitor_records())
    assert report_a.to_json() == report_b.to_json()
    extension_a = to_path_bundle_extension(report_a)
    extension_b = to_path_bundle_extension(report_b)
    assert stable_json_dumps(extension_a) == stable_json_dumps(extension_b)
    # Frame order is sorted by frame_index regardless of input order.
    shuffled = list(reversed(raw))
    report_c = recover_frames(_parse(compiled, shuffled), compiled)
    assert [record.frame_index for record in report_c.frames] == [0, 1, 2]


def test_surfaces_are_truth_free(tmp_path) -> None:
    compiled = _compile_2b(tmp_path)
    report = recover_frames(
        _parse(compiled, _raw_frames_2b(with_single_point=True)), compiled, _monitor_records()
    )
    blob = report.to_json() + stable_json_dumps(to_path_bundle_extension(report))
    for forbidden in ("orientation", "irc_evidence", "endpoint_match", "ground_truth",
                      "ts_coordinates", "irc_frames"):
        assert forbidden not in blob


def test_frozen_acp_module_surfaces_stay_intact() -> None:
    """trajectory.py / quality.py / multicoord.py public surfaces unchanged."""
    from pes2ts_core.integration.acp import quality, trajectory
    from pes2ts_core.integration.acp import multicoord as mc

    assert trajectory.path_bundle_to_acp_pes_profile is not None
    assert trajectory.project_path_bundle_to_acp_graph is not None
    assert quality.assess_scan_path_quality is not None
    assert quality.apply_scan_path_quality is not None
    assert mc.multicoord_request_payload is not None
    assert mc.parse_multicoord_result is not None
    assert mc.driver_id_for(0) == "driver-01" == driver_id_for(0)
    assert mc.SCHEMA_MULTICOORD_RESULT == SCHEMA_MULTICOORD_RESULT
