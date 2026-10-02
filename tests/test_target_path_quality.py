"""Tests for pes2ts_core.generation.planning.target_path (plan todo 22).

Covers the design §11.2 four independent result tiers:
- numerically usable + unfinished target edits → target_path_compatible=false
  (the plan's failure scenario — technical usable ≠ chemical completion);
- validated_ts absent without OptTS + first-order imaginary + two-way IRC
  evidence references; missing evidence listed per artifact;
- continuous bond-length labels with unknown/transition (never forced binary);
- no-internal-peak association/dissociation paths are not reported as TS;
- endpoint maximum energy is never reported as TS;
- stereo / region target checks; path continuity; determinism; purity.
"""

from __future__ import annotations

from typing import Any

import pytest

from pes2ts_core.contracts import FORBIDDEN_TRUTH_KEYS
from pes2ts_core.g1.v2_verify import FORBIDDEN_EXPORT_KEYS
from pes2ts_core.integration.acp.frame_recovery import (
    FrameRecoveryRecord,
    FrameRecoveryReport,
    MonitorRecord,
    EnergyChannelRow,
    SCHEMA_FRAME_RECOVERY,
)
from pes2ts_core.generation.planning.target_path import (
    BOND_BROKEN,
    BOND_FORMED,
    BOND_NO_ORDER_EVIDENCE,
    BOND_TRANSITION,
    CODE_BOND_INCOMPLETE,
    CODE_BOND_MEASUREMENT_MISSING,
    CODE_ENDPOINT_MAX_NOT_TS,
    CODE_EVIDENCE_INCOMPLETE,
    CODE_FRAME_JUMP,
    CODE_NO_FRAMES,
    CODE_NO_INTERNAL_PEAK,
    CODE_ORDER_EVIDENCE_MISSING,
    CODE_TARGET_INCOMPLETE,
    EVIDENCE_FIRST_ORDER_IMAGINARY,
    EVIDENCE_IRC_FORWARD,
    EVIDENCE_IRC_REVERSE,
    EVIDENCE_OPTTS,
    LABEL_ACHIEVED,
    LABEL_NOT_ACHIEVED,
    LABEL_NOT_EVALUABLE,
    PATH_CLASS_ENDPOINT_MAX,
    PATH_CLASS_INTERNAL_MAXIMUM,
    PATH_CLASS_NO_INTERNAL_PEAK,
    SCHEMA_TARGET_PATH_ASSESSMENT,
    TIER_EXECUTION_COMPLETE,
    TIER_NUMERICALLY_USABLE,
    TIER_TARGET_PATH_COMPATIBLE,
    TIER_VALIDATED_TS,
    TIERS,
    TargetPathError,
    assess_target_path,
    parse_evidence_bundle,
    parse_target_definitions,
)
from pes2ts_core.utils.hashing import stable_json_dumps

FORBIDDEN = {key.lower() for key in (FORBIDDEN_TRUTH_KEYS | FORBIDDEN_EXPORT_KEYS)}


# ---------------------------------------------------------------------------
# Fixtures: a 3-frame recovered path with all-driver residuals + a monitor.
# ---------------------------------------------------------------------------
DRIVER = "driver-01"


def _energy_row(value: float | None, method_id: str | None = "B3LYP-D3") -> EnergyChannelRow:
    return EnergyChannelRow(energy_channel="scan", value=value, method_id=method_id)


def _record(
    index: int,
    *,
    actual: float,
    target: float = 1.5,
    converged: bool = True,
    scan_energy: float | None = -40.0,
    monitor_value: float | None = None,
    incomplete: bool = False,
    missing_residual: tuple[str, ...] = (),
    frame_index: int | None = None,
) -> FrameRecoveryRecord:
    residual = None if actual is None or target is None else target - actual
    return FrameRecoveryRecord(
        frame_index=index if frame_index is None else frame_index,
        converged=converged,
        lambda_value=index / 2,
        stage_id=None,
        geometry_ref=f"frames/f{index}.xyz",
        targets_by_driver={DRIVER: target},
        actuals_by_driver={DRIVER: actual},
        residuals_by_driver={DRIVER: residual},
        backend_residuals_by_driver={DRIVER: residual},
        missing_driver_ids=() if actual is not None else (DRIVER,),
        missing_residual_driver_ids=missing_residual,
        incomplete=incomplete,
        monitors=(
            MonitorRecord(
                coordinate_id="coord-mon-b",
                kind="B",
                atom_maps=(1, 2),
                units="angstrom",
                role="monitor",
                value=monitor_value if monitor_value is not None else actual,
                measured=True,
            ),
        ),
        non_target_contacts=(),
        contacts_measured=True,
        electronic_state_diagnostics={"status": "not_recorded"},
        retry_history=(),
        energies=(_energy_row(scan_energy),),
    )


def _report(
    records: list[FrameRecoveryRecord],
    *,
    complete: bool | None = None,
    candidate_id: str = "cand-t22",
) -> FrameRecoveryReport:
    incomplete = tuple(row.frame_index for row in records if row.incomplete)
    if complete is None:
        complete = bool(records) and not incomplete
    return FrameRecoveryReport(
        schema_version=SCHEMA_FRAME_RECOVERY,
        candidate_id=candidate_id,
        mode="SINGLE_1D",
        execution_id="exec-t22",
        acp_task_id="job-t22",
        driver_ids=(DRIVER,),
        driver_units={DRIVER: "angstrom"},
        n_frames=len(records),
        complete=complete,
        frames=tuple(records),
        incomplete_frame_indices=incomplete,
        monitor_coordinate_ids=("coord-mon-b",),
        electronic_state_diagnostics={"status": "not_recorded"},
        source={"n_drivers": 1},
    )


def _formed_target(*, driver_id: str | None = DRIVER, target_id: str = "t-formed") -> dict[str, Any]:
    # C–C radius sum ≈ 1.52 Å; formed bond needs d ≤ 1.52+0.45 = 1.97 Å.
    return {
        "target_id": target_id,
        "atom_maps": [1, 2],
        "edit_kind": "formed",
        "element_a": "C",
        "element_b": "C",
        "r_distance_angstrom": 3.0,
        "p_distance_angstrom": 1.54,
        "driver_id": driver_id,
    }


def _broken_target(*, driver_id: str | None = None, target_id: str = "t-broken") -> dict[str, Any]:
    return {
        "target_id": target_id,
        "atom_maps": [3, 4],
        "edit_kind": "broken",
        "element_a": "C",
        "element_b": "H",
        "r_distance_angstrom": 1.09,
        "p_distance_angstrom": 3.0,
        "driver_id": driver_id,
    }


def _order_target(*, target_id: str = "t-order") -> dict[str, Any]:
    return {
        "target_id": target_id,
        "atom_maps": [5, 6],
        "edit_kind": "order_changed",
        "element_a": "C",
        "element_b": "O",
        "r_distance_angstrom": 1.43,
        "p_distance_angstrom": 1.22,
        "driver_id": None,
    }


def _full_evidence() -> dict[str, str]:
    return {
        EVIDENCE_OPTTS: "receipts/optts/abc.json",
        EVIDENCE_FIRST_ORDER_IMAGINARY: "receipts/freq/abc.json",
        EVIDENCE_IRC_FORWARD: "receipts/irc/fwd.json",
        EVIDENCE_IRC_REVERSE: "receipts/irc/rev.json",
    }


# ---------------------------------------------------------------------------
# Tier independence (plan failure scenario).
# ---------------------------------------------------------------------------
def test_numerically_usable_but_target_incomplete_is_not_target_path_compatible() -> None:
    """Plan failure scenario: technical usable ≠ chemical target completion."""
    # Constrained scan followed its schedule (residuals ~0) but the driven
    # coordinate never reaches the formed-bond distance — 2.8 Å stays above
    # the C–C bonded upper (~1.97 Å).
    records = [
        _record(0, actual=3.0, target=3.0, monitor_value=3.0, scan_energy=-40.0),
        _record(1, actual=2.9, target=2.9, monitor_value=2.9, scan_energy=-39.0),
        _record(2, actual=2.8, target=2.8, monitor_value=2.8, scan_energy=-38.0),
    ]
    assessment = assess_target_path(
        _report(records),
        {"bonds": [_formed_target()], "regions": [], "stereo": []},
        {},
    )
    assert assessment.tiers[TIER_EXECUTION_COMPLETE].status == LABEL_ACHIEVED
    assert assessment.tiers[TIER_NUMERICALLY_USABLE].status == LABEL_ACHIEVED
    assert assessment.tiers[TIER_TARGET_PATH_COMPATIBLE].status == LABEL_NOT_ACHIEVED
    assert CODE_TARGET_INCOMPLETE in assessment.tiers[TIER_TARGET_PATH_COMPATIBLE].reason_codes
    # Tiers are independent keys, never collapsed.
    assert set(assessment.tiers) == set(TIERS)


def test_target_path_compatible_achieved_when_edit_realized() -> None:
    records = [
        _record(0, actual=3.0, monitor_value=3.0),
        _record(1, actual=2.0, monitor_value=2.0),
        _record(2, actual=1.54, monitor_value=1.54),
    ]
    assessment = assess_target_path(
        _report(records),
        {"bonds": [_formed_target()], "regions": [], "stereo": []},
        {},
    )
    assert assessment.tiers[TIER_TARGET_PATH_COMPATIBLE].status == LABEL_ACHIEVED
    bond = assessment.bond_labels[0]
    assert bond.label == BOND_FORMED
    assert bond.completed is True


# ---------------------------------------------------------------------------
# validated_ts evidence requirements.
# ---------------------------------------------------------------------------
def test_validated_ts_absent_without_evidence_and_lists_missing() -> None:
    records = [
        _record(0, actual=3.0, monitor_value=3.0, scan_energy=-40.0),
        _record(1, actual=2.0, monitor_value=2.0, scan_energy=-38.0),
        _record(2, actual=1.54, monitor_value=1.54, scan_energy=-39.5),
    ]
    assessment = assess_target_path(
        _report(records),
        {"bonds": [_formed_target()], "regions": [], "stereo": []},
        {},
    )
    ts = assessment.tiers[TIER_VALIDATED_TS]
    assert ts.status == LABEL_NOT_ACHIEVED
    assert CODE_EVIDENCE_INCOMPLETE in ts.reason_codes
    assert set(ts.missing_evidence) == set(
        (EVIDENCE_OPTTS, EVIDENCE_FIRST_ORDER_IMAGINARY, EVIDENCE_IRC_FORWARD, EVIDENCE_IRC_REVERSE)
    )
    # Internal maximum alone never grants validated_ts.
    assert assessment.path_shape.classification == PATH_CLASS_INTERNAL_MAXIMUM


def test_validated_ts_partial_evidence_lists_only_missing_artifacts() -> None:
    records = [
        _record(0, actual=3.0, monitor_value=3.0, scan_energy=-40.0),
        _record(1, actual=2.0, monitor_value=2.0, scan_energy=-38.0),
        _record(2, actual=1.54, monitor_value=1.54, scan_energy=-39.5),
    ]
    evidence = {
        EVIDENCE_OPTTS: "receipts/optts/abc.json",
        EVIDENCE_IRC_FORWARD: "receipts/irc/fwd.json",
    }
    assessment = assess_target_path(
        _report(records),
        {"bonds": [_formed_target()], "regions": [], "stereo": []},
        evidence,
    )
    ts = assessment.tiers[TIER_VALIDATED_TS]
    assert ts.status == LABEL_NOT_ACHIEVED
    assert set(ts.missing_evidence) == {
        EVIDENCE_FIRST_ORDER_IMAGINARY,
        EVIDENCE_IRC_REVERSE,
    }
    assert EVIDENCE_OPTTS not in ts.missing_evidence


def test_validated_ts_achieved_when_all_evidence_refs_present() -> None:
    records = [
        _record(0, actual=3.0, monitor_value=3.0, scan_energy=-40.0),
        _record(1, actual=2.0, monitor_value=2.0, scan_energy=-38.0),
        _record(2, actual=1.54, monitor_value=1.54, scan_energy=-39.5),
    ]
    assessment = assess_target_path(
        _report(records),
        {"bonds": [_formed_target()], "regions": [], "stereo": []},
        _full_evidence(),
    )
    assert assessment.tiers[TIER_VALIDATED_TS].status == LABEL_ACHIEVED
    assert assessment.tiers[TIER_VALIDATED_TS].missing_evidence == ()
    # Evidence refs recorded as opaque strings only.
    assert assessment.evidence[EVIDENCE_OPTTS] == "receipts/optts/abc.json"


def test_evidence_bundle_refuses_forbidden_truth_keys() -> None:
    with pytest.raises(TargetPathError) as excinfo:
        parse_evidence_bundle({"irc_evidence": "something", EVIDENCE_OPTTS: "x"})
    assert excinfo.value.code == "EVIDENCE_INVALID"


# ---------------------------------------------------------------------------
# Continuous bond labels + transition zone.
# ---------------------------------------------------------------------------
def test_transition_zone_bond_is_unknown_not_forced_binary() -> None:
    # 2.2 Å is between bonded_upper (~1.97) and separated_lower (~2.47).
    records = [
        _record(0, actual=3.0, monitor_value=3.0),
        _record(1, actual=2.5, monitor_value=2.5),
        _record(2, actual=2.2, monitor_value=2.2),
    ]
    assessment = assess_target_path(
        _report(records),
        {"bonds": [_formed_target()], "regions": [], "stereo": []},
        {},
    )
    bond = assessment.bond_labels[0]
    assert bond.label == BOND_TRANSITION
    assert bond.completed is False
    assert CODE_BOND_INCOMPLETE in bond.reason_codes
    assert assessment.tiers[TIER_TARGET_PATH_COMPATIBLE].status == LABEL_NOT_ACHIEVED


def test_broken_bond_continuous_labels() -> None:
    # Broken C–H: bonded upper ≈ 1.07+0.45=1.52; separated lower ≈ 2.02.
    records = [
        _record(0, actual=1.09, monitor_value=1.09),
        _record(1, actual=1.8, monitor_value=1.8),
        _record(2, actual=3.0, monitor_value=3.0),
    ]
    # Provide the broken pair as a monitor on maps (3,4) via an extra monitor.
    # Reuse driver actual for measurement by attaching driver_id=None → measurement
    # missing unless we put the pair in monitors. Build custom records:
    def rec(index: int, value: float, energy: float) -> FrameRecoveryRecord:
        row = _record(index, actual=value, monitor_value=value, scan_energy=energy)
        return FrameRecoveryRecord(
            frame_index=row.frame_index,
            converged=row.converged,
            lambda_value=row.lambda_value,
            stage_id=row.stage_id,
            geometry_ref=row.geometry_ref,
            targets_by_driver=row.targets_by_driver,
            actuals_by_driver=row.actuals_by_driver,
            residuals_by_driver=row.residuals_by_driver,
            backend_residuals_by_driver=row.backend_residuals_by_driver,
            missing_driver_ids=row.missing_driver_ids,
            missing_residual_driver_ids=row.missing_residual_driver_ids,
            incomplete=row.incomplete,
            monitors=(
                MonitorRecord(
                    coordinate_id="coord-broken",
                    kind="B",
                    atom_maps=(3, 4),
                    units="angstrom",
                    role="monitor",
                    value=value,
                    measured=True,
                ),
            ),
            non_target_contacts=(),
            contacts_measured=True,
            electronic_state_diagnostics=row.electronic_state_diagnostics,
            retry_history=(),
            energies=(_energy_row(energy),),
        )

    assessment = assess_target_path(
        _report([rec(0, 1.09, -40.0), rec(1, 1.8, -39.0), rec(2, 3.0, -38.5)]),
        {"bonds": [_broken_target()], "regions": [], "stereo": []},
        {},
    )
    bond = assessment.bond_labels[0]
    assert bond.label == BOND_BROKEN
    assert bond.completed is True
    assert assessment.tiers[TIER_TARGET_PATH_COMPATIBLE].status == LABEL_ACHIEVED


def test_order_change_without_electronic_evidence_is_not_completed() -> None:
    def rec(index: int, value: float, energy: float) -> FrameRecoveryRecord:
        return FrameRecoveryRecord(
            frame_index=index,
            converged=True,
            lambda_value=index / 2,
            stage_id=None,
            geometry_ref=f"frames/o{index}.xyz",
            targets_by_driver={DRIVER: value},
            actuals_by_driver={DRIVER: value},
            residuals_by_driver={DRIVER: 0.0},
            backend_residuals_by_driver={DRIVER: 0.0},
            missing_driver_ids=(),
            missing_residual_driver_ids=(),
            incomplete=False,
            monitors=(
                MonitorRecord(
                    coordinate_id="coord-order",
                    kind="B",
                    atom_maps=(5, 6),
                    units="angstrom",
                    role="monitor",
                    value=value,
                    measured=True,
                ),
            ),
            non_target_contacts=(),
            contacts_measured=True,
            electronic_state_diagnostics={"status": "not_recorded"},
            retry_history=(),
            energies=(_energy_row(energy),),
        )

    assessment = assess_target_path(
        _report([rec(0, 1.43, -40.0), rec(1, 1.3, -39.2), rec(2, 1.22, -38.8)]),
        {
            "bonds": [_order_target()],
            "regions": [],
            "stereo": [],
            "electronic_order_evidence": {},
        },
        {},
    )
    bond = assessment.bond_labels[0]
    assert bond.label == BOND_NO_ORDER_EVIDENCE
    assert bond.completed is False
    assert CODE_ORDER_EVIDENCE_MISSING in bond.reason_codes

    with_evidence = assess_target_path(
        _report([rec(0, 1.43, -40.0), rec(1, 1.3, -39.2), rec(2, 1.22, -38.8)]),
        {
            "bonds": [_order_target()],
            "regions": [],
            "stereo": [],
            "electronic_order_evidence": {"t-order": "receipts/mulliken/order.json"},
        },
        {},
    )
    assert with_evidence.bond_labels[0].label == BOND_FORMED
    assert with_evidence.bond_labels[0].completed is True


def test_unmeasurable_bond_is_not_evaluable_measurement() -> None:
    records = [
        _record(0, actual=1.5, monitor_value=1.5),
        _record(1, actual=1.5, monitor_value=1.5),
        _record(2, actual=1.5, monitor_value=1.5),
    ]
    # Broken target on maps (3,4) with no monitor and no driver_id.
    assessment = assess_target_path(
        _report(records),
        {"bonds": [_broken_target()], "regions": [], "stereo": []},
        {},
    )
    bond = assessment.bond_labels[0]
    assert bond.label == BOND_TRANSITION
    assert bond.end_distance_angstrom is None
    assert CODE_BOND_MEASUREMENT_MISSING in bond.reason_codes


# ---------------------------------------------------------------------------
# Path shape: endpoint-max and no-peak are never TS.
# ---------------------------------------------------------------------------
def test_endpoint_max_energy_is_never_reported_as_ts() -> None:
    records = [
        _record(0, actual=3.0, monitor_value=3.0, scan_energy=-38.0),  # max at endpoint
        _record(1, actual=2.2, monitor_value=2.2, scan_energy=-39.0),
        _record(2, actual=1.54, monitor_value=1.54, scan_energy=-40.0),
    ]
    assessment = assess_target_path(
        _report(records),
        {"bonds": [_formed_target()], "regions": [], "stereo": []},
        {},
    )
    assert assessment.path_shape.classification == PATH_CLASS_ENDPOINT_MAX
    assert assessment.path_shape.max_is_endpoint is True
    assert assessment.ts_candidate_from_path_shape is False
    assert assessment.tiers[TIER_VALIDATED_TS].status == LABEL_NOT_ACHIEVED
    assert CODE_ENDPOINT_MAX_NOT_TS in assessment.tiers[TIER_VALIDATED_TS].reason_codes


def test_no_internal_peak_dissociation_path_is_not_ts() -> None:
    # Monotonic rise to dissociation endpoint — no interior barrier.
    # Design §11.2: such association/dissociation paths may have no applicable
    # TS seed; the endpoint maximum is never force-reported as TS.
    records = [
        _record(0, actual=1.09, target=1.09, monitor_value=1.09, scan_energy=-40.0),
        _record(1, actual=2.0, target=2.0, monitor_value=2.0, scan_energy=-39.0),
        _record(2, actual=3.0, target=3.0, monitor_value=3.0, scan_energy=-38.0),
    ]
    assessment = assess_target_path(
        _report(records),
        {"bonds": [_broken_target(driver_id=DRIVER)], "regions": [], "stereo": []},
        {},
    )
    assert assessment.path_shape.has_internal_maximum is False
    assert assessment.path_shape.classification in (
        PATH_CLASS_ENDPOINT_MAX,
        PATH_CLASS_NO_INTERNAL_PEAK,
    )
    assert assessment.ts_candidate_from_path_shape is False
    assert assessment.tiers[TIER_VALIDATED_TS].status == LABEL_NOT_ACHIEVED
    shape_codes = set(assessment.tiers[TIER_VALIDATED_TS].reason_codes)
    assert shape_codes & {CODE_NO_INTERNAL_PEAK, CODE_ENDPOINT_MAX_NOT_TS}


def test_flat_energy_path_without_barrier_is_not_ts() -> None:
    """No-peak path: flat energies never grant validated_ts from path shape."""
    records = [
        _record(0, actual=2.0, target=2.0, monitor_value=2.0, scan_energy=-40.0),
        _record(1, actual=2.0, target=2.0, monitor_value=2.0, scan_energy=-40.0),
        _record(2, actual=2.0, target=2.0, monitor_value=2.0, scan_energy=-40.0),
    ]
    assessment = assess_target_path(
        _report(records),
        {"bonds": [_broken_target(driver_id=DRIVER)], "regions": [], "stereo": []},
        {},
    )
    assert assessment.path_shape.has_internal_maximum is False
    assert assessment.ts_candidate_from_path_shape is False
    assert assessment.tiers[TIER_VALIDATED_TS].status == LABEL_NOT_ACHIEVED
    assert CODE_NO_INTERNAL_PEAK in assessment.tiers[TIER_VALIDATED_TS].reason_codes or (
        CODE_ENDPOINT_MAX_NOT_TS in assessment.tiers[TIER_VALIDATED_TS].reason_codes
    )


def test_internal_maximum_with_full_evidence_achieves_validated_ts() -> None:
    records = [
        _record(0, actual=3.0, monitor_value=3.0, scan_energy=-40.0),
        _record(1, actual=2.0, monitor_value=2.0, scan_energy=-37.5),
        _record(2, actual=1.54, monitor_value=1.54, scan_energy=-39.0),
    ]
    assessment = assess_target_path(
        _report(records),
        {"bonds": [_formed_target()], "regions": [], "stereo": []},
        _full_evidence(),
    )
    assert assessment.path_shape.classification == PATH_CLASS_INTERNAL_MAXIMUM
    assert assessment.tiers[TIER_VALIDATED_TS].status == LABEL_ACHIEVED
    assert assessment.ts_candidate_from_path_shape is True


# ---------------------------------------------------------------------------
# Path continuity + numerical checks.
# ---------------------------------------------------------------------------
def test_frame_index_gap_breaks_continuity_and_numerical_tiers() -> None:
    records = [
        _record(0, actual=3.0, monitor_value=3.0),
        _record(2, actual=1.54, monitor_value=1.54, frame_index=2),
    ]
    assessment = assess_target_path(
        _report(records, complete=False),
        {"bonds": [_formed_target()], "regions": [], "stereo": []},
        {},
    )
    assert CODE_FRAME_JUMP in assessment.tiers[TIER_TARGET_PATH_COMPATIBLE].reason_codes
    assert assessment.tiers[TIER_NUMERICALLY_USABLE].status == LABEL_NOT_ACHIEVED


def test_empty_report_yields_no_frames_on_all_applicable_tiers() -> None:
    assessment = assess_target_path(
        _report([], complete=False),
        {"bonds": [_formed_target()], "regions": [], "stereo": []},
        {},
    )
    assert assessment.tiers[TIER_EXECUTION_COMPLETE].status == LABEL_NOT_ACHIEVED
    assert CODE_NO_FRAMES in assessment.tiers[TIER_EXECUTION_COMPLETE].reason_codes
    assert assessment.tiers[TIER_NUMERICALLY_USABLE].status == LABEL_NOT_ACHIEVED
    assert assessment.tiers[TIER_TARGET_PATH_COMPATIBLE].status == LABEL_NOT_ACHIEVED
    assert assessment.tiers[TIER_VALIDATED_TS].status == LABEL_NOT_ACHIEVED


def test_unconverged_frames_block_numerically_usable() -> None:
    records = [
        _record(0, actual=3.0, monitor_value=3.0, converged=True),
        _record(1, actual=2.2, monitor_value=2.2, converged=False),
        _record(2, actual=1.54, monitor_value=1.54, converged=True),
    ]
    assessment = assess_target_path(
        _report(records),
        {"bonds": [_formed_target()], "regions": [], "stereo": []},
        {},
    )
    assert assessment.tiers[TIER_NUMERICALLY_USABLE].status == LABEL_NOT_ACHIEVED
    # Chemical tier may still see the edit realized — independence.
    assert assessment.tiers[TIER_TARGET_PATH_COMPATIBLE].status == LABEL_ACHIEVED


# ---------------------------------------------------------------------------
# Stereo and region checks.
# ---------------------------------------------------------------------------
def _stereo_record(index: int, energy: float) -> FrameRecoveryRecord:
    value = 3.0 - 1.46 * index / 2
    return _record(index, actual=value, monitor_value=value, scan_energy=energy)


def test_stereo_target_achieved_when_final_label_matches_product() -> None:
    assessment = assess_target_path(
        _report([_stereo_record(0, -40.0), _stereo_record(1, -39.0), _stereo_record(2, -38.5)]),
        {
            "bonds": [_formed_target()],
            "regions": [],
            "stereo": [
                {
                    "target_id": "st-1",
                    "center_maps": [2],
                    "r_label": "R",
                    "p_label": "S",
                    "final_label": "S",
                }
            ],
        },
        {},
    )
    assert assessment.tiers[TIER_TARGET_PATH_COMPATIBLE].status == LABEL_ACHIEVED


def test_stereo_target_at_reactant_is_incomplete() -> None:
    assessment = assess_target_path(
        _report([_stereo_record(0, -40.0), _stereo_record(1, -39.0), _stereo_record(2, -38.5)]),
        {
            "bonds": [_formed_target()],
            "regions": [],
            "stereo": [
                {
                    "target_id": "st-1",
                    "center_maps": [2],
                    "r_label": "R",
                    "p_label": "S",
                    "final_label": "R",
                }
            ],
        },
        {},
    )
    assert assessment.tiers[TIER_TARGET_PATH_COMPATIBLE].status == LABEL_NOT_ACHIEVED
    details = assessment.tiers[TIER_TARGET_PATH_COMPATIBLE].details
    assert details["stereo_failures"][0]["code"] == "STEREO_AT_REACTANT"


def test_stereo_without_final_label_is_not_evaluable() -> None:
    assessment = assess_target_path(
        _report([_stereo_record(0, -40.0), _stereo_record(1, -39.0), _stereo_record(2, -38.5)]),
        {
            "bonds": [_formed_target()],
            "regions": [],
            "stereo": [
                {
                    "target_id": "st-1",
                    "center_maps": [2],
                    "r_label": "R",
                    "p_label": "S",
                    "final_label": None,
                }
            ],
        },
        {},
    )
    assert assessment.tiers[TIER_TARGET_PATH_COMPATIBLE].status == LABEL_NOT_ACHIEVED
    details = assessment.tiers[TIER_TARGET_PATH_COMPATIBLE].details
    assert details["stereo_failures"][0]["code"] == "STEREO_LABEL_MISSING"


def test_region_target_with_member_bonds_follows_bond_completion() -> None:
    def rec(index: int, value: float, energy: float) -> FrameRecoveryRecord:
        return FrameRecoveryRecord(
            frame_index=index,
            converged=True,
            lambda_value=index / 2,
            stage_id=None,
            geometry_ref=f"frames/r{index}.xyz",
            targets_by_driver={DRIVER: value},
            actuals_by_driver={DRIVER: value},
            residuals_by_driver={DRIVER: 0.0},
            backend_residuals_by_driver={DRIVER: 0.0},
            missing_driver_ids=(),
            missing_residual_driver_ids=(),
            incomplete=False,
            monitors=(
                MonitorRecord(
                    coordinate_id="coord-region",
                    kind="B",
                    atom_maps=(7, 8),
                    units="angstrom",
                    role="monitor",
                    value=value,
                    measured=True,
                ),
            ),
            non_target_contacts=(),
            contacts_measured=True,
            electronic_state_diagnostics={"status": "not_recorded"},
            retry_history=(),
            energies=(_energy_row(energy),),
        )

    member = {
        "target_id": "t-region-bond",
        "atom_maps": [7, 8],
        "edit_kind": "formed",
        "element_a": "C",
        "element_b": "C",
        "driver_id": None,
    }
    incomplete = assess_target_path(
        _report([rec(0, 3.0, -40.0), rec(1, 2.5, -39.0), rec(2, 2.2, -38.5)]),
        {
            "bonds": [_formed_target()],
            "regions": [{"target_id": "rg-1", "region_id": "ar-1", "bond_targets": [member]}],
            "stereo": [],
        },
        {},
    )
    assert incomplete.tiers[TIER_TARGET_PATH_COMPATIBLE].status == LABEL_NOT_ACHIEVED

    complete = assess_target_path(
        _report([rec(0, 3.0, -40.0), rec(1, 2.0, -39.0), rec(2, 1.54, -38.5)]),
        {
            "bonds": [_formed_target()],
            "regions": [{"target_id": "rg-1", "region_id": "ar-1", "bond_targets": [member]}],
            "stereo": [],
        },
        {},
    )
    assert complete.tiers[TIER_TARGET_PATH_COMPATIBLE].status == LABEL_ACHIEVED


def test_electronic_region_without_evidence_is_not_evaluable() -> None:
    records = [
        _record(0, actual=1.5, monitor_value=1.5),
        _record(1, actual=1.5, monitor_value=1.5),
        _record(2, actual=1.5, monitor_value=1.5),
    ]
    assessment = assess_target_path(
        _report(records),
        {
            "bonds": [_formed_target(driver_id=DRIVER)],
            "regions": [
                {
                    "target_id": "rg-1",
                    "region_id": "ar-1",
                    "bond_targets": [],
                    "requires_electronic_evidence": True,
                }
            ],
            "stereo": [],
        },
        {},
    )
    assert assessment.tiers[TIER_TARGET_PATH_COMPATIBLE].status == LABEL_NOT_ACHIEVED


# ---------------------------------------------------------------------------
# Determinism + purity + mapping round-trip.
# ---------------------------------------------------------------------------
def test_assessment_is_deterministic() -> None:
    records = [
        _record(0, actual=3.0, monitor_value=3.0, scan_energy=-40.0),
        _record(1, actual=2.0, monitor_value=2.0, scan_energy=-38.0),
        _record(2, actual=1.54, monitor_value=1.54, scan_energy=-39.5),
    ]
    report = _report(records)
    definitions = {"bonds": [_formed_target()], "regions": [], "stereo": []}
    evidence = _full_evidence()
    first = assess_target_path(report, definitions, evidence)
    second = assess_target_path(report, definitions, evidence)
    assert first.to_json() == second.to_json()
    assert first.to_json() == stable_json_dumps(first.to_doc())


def test_to_doc_carries_no_forbidden_truth_keys() -> None:
    records = [
        _record(0, actual=3.0, monitor_value=3.0, scan_energy=-40.0),
        _record(1, actual=2.0, monitor_value=2.0, scan_energy=-38.0),
        _record(2, actual=1.54, monitor_value=1.54, scan_energy=-39.5),
    ]
    assessment = assess_target_path(
        _report(records),
        {"bonds": [_formed_target()], "regions": [], "stereo": []},
        _full_evidence(),
    )

    def walk(value: Any, hits: list[str]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in FORBIDDEN:
                    hits.append(str(key))
                walk(child, hits)
        elif isinstance(value, list):
            for item in value:
                walk(item, hits)

    hits: list[str] = []
    walk(assessment.to_doc(), hits)
    assert hits == []
    assert "orientation" not in assessment.to_json()
    assert "irc_evidence" not in assessment.to_json()
    assert "endpoint_match" not in assessment.to_json()
    assert "ts_coordinates" not in assessment.to_json()


def test_mapping_inputs_round_trip_like_dataclasses() -> None:
    records = [
        _record(0, actual=3.0, monitor_value=3.0, scan_energy=-40.0),
        _record(1, actual=2.0, monitor_value=2.0, scan_energy=-38.0),
        _record(2, actual=1.54, monitor_value=1.54, scan_energy=-39.5),
    ]
    report = _report(records)
    from_mapping = assess_target_path(
        report.to_doc(),
        {"bonds": [_formed_target()], "regions": [], "stereo": []},
        _full_evidence(),
    )
    from_dataclass = assess_target_path(
        report,
        parse_target_definitions({"bonds": [_formed_target()], "regions": [], "stereo": []}),
        parse_evidence_bundle(_full_evidence()),
    )
    assert from_mapping.to_json() == from_dataclass.to_json()
    assert from_mapping.schema_version == SCHEMA_TARGET_PATH_ASSESSMENT
    assert set(from_mapping.tiers) == set(TIERS)
    # checks block is present and typed.
    assert "bond_targets" in from_mapping.checks
    assert "path_continuity" in from_mapping.checks
    assert from_mapping.checks["ts_evidence"].status == LABEL_ACHIEVED


def test_invalid_target_definition_is_typed_error() -> None:
    with pytest.raises(TargetPathError) as excinfo:
        parse_target_definitions({"bonds": [{"target_id": "x", "atom_maps": [1], "edit_kind": "formed",
                                             "element_a": "C", "element_b": "C"}]})
    assert excinfo.value.code == "TARGET_DEFINITION_INVALID"
