"""Technical ACP scan-path quality gates remain separate from TS validation."""
from __future__ import annotations

import pytest

from pes2ts_core.contracts import ContractError, seal_document
from pes2ts_core.demo import synthetic_objects
from pes2ts_core.integration.acp.quality import apply_scan_path_quality, assess_scan_path_quality
from pes2ts_core.ranking import rank_path_bundle


def _quality_inputs():
    docs = synthetic_objects()
    plan = docs["ScanPlan"]
    path = docs["PathBundle"]
    execution = docs["ExecutionRecord"]
    points = plan["candidates"][0]["coordinates"][0]["points"]
    frames = []
    for index, target in enumerate(points):
        frame = path["frames"][min(index, len(path["frames"]) - 1)]
        geometry = [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [1.1 + target, 0.0, 0.0]]
        row = {**frame, "frame_id": f"quality-frame-{index}", "frame_index": index,
               "geometry": geometry, "actual_coordinate": target, "constraint_residuals": {"driver-01": 0.0},
               "target_coordinate": target, "converged": True,
               "energies": {"scan_electronic": {"value": -20.0 + index / 100,
                                                   "unit": "hartree", "method_id": "xtb:gfn2-xTB"}}}
        frames.append(row)
    path = seal_document({**path, "frames": frames})
    return plan, execution, path


def test_completed_full_converged_scan_is_usable_but_not_physically_validated():
    plan, execution, path = _quality_inputs()

    result = assess_scan_path_quality(plan=plan, execution=execution, path=path)

    assert result["status"] == "usable"
    assert result["physical_validation"] == "not_run"
    assert all(result["checks"].values())
    assert result["issues"] == []

    applied = apply_scan_path_quality(plan=plan, execution=execution, path=path)
    assert applied["path_bundle"]["status"] == "usable"
    assert applied["path_bundle"]["content_sha256"] != path["content_sha256"]
    assert applied["quality_assessment"]["input_sha256"]["path"] == path["content_sha256"]
    assert applied["quality_assessment"]["result_path_sha256"] == applied["path_bundle"]["content_sha256"]
    assert rank_path_bundle(applied["path_bundle"])["status"] == "accepted"


def test_partial_completed_scan_requires_review():
    plan, execution, path = _quality_inputs()
    path = seal_document({**path, "frames": path["frames"][:-1]})

    result = assess_scan_path_quality(plan=plan, execution=execution, path=path)

    assert result["status"] == "needs_review"
    assert result["checks"]["frame_count_matches_plan"] is False


def test_unconverged_frame_needs_review():
    plan, execution, path = _quality_inputs()
    frames = [dict(frame) for frame in path["frames"]]
    frames[2] = {**frames[2], "converged": False}
    path = seal_document({**path, "frames": frames})

    result = assess_scan_path_quality(plan=plan, execution=execution, path=path)

    assert result["status"] == "needs_review"
    assert result["checks"]["all_frames_converged"] is False


def test_missing_target_coordinate_needs_review_but_wrong_coordinate_is_unusable():
    plan, execution, path = _quality_inputs()
    frames = [dict(frame) for frame in path["frames"]]
    frames[1] = {key: value for key, value in frames[1].items() if key != "target_coordinate"}
    path_missing = seal_document({**path, "frames": frames})
    assert assess_scan_path_quality(plan=plan, execution=execution,
                                    path=path_missing)["status"] == "needs_review"

    frames[1] = {**frames[1], "target_coordinate": 9.9}
    path_misaligned = seal_document({**path, "frames": frames})
    assert assess_scan_path_quality(plan=plan, execution=execution,
                                    path=path_misaligned)["status"] == "unusable"


def test_impossible_geometry_and_constraint_residual_cannot_be_ranked():
    plan, execution, path = _quality_inputs()
    frames = [dict(frame) for frame in path["frames"]]
    frames[0] = {**frames[0], "geometry":[[0, 0, 0], [0, 0, 0], [99, 0, 0]],
                 "actual_coordinate":99.0, "constraint_residuals":{"driver-01":99.0}}
    invalid_path = seal_document({**path, "frames":frames})

    result = assess_scan_path_quality(plan=plan, execution=execution, path=invalid_path)

    assert result["status"] == "unusable"
    assert result["checks"]["actual_coordinates_satisfy_constraints"] is False
    assert result["checks"]["no_atom_collisions"] is False
    assert any("violates the frozen distance constraint" in issue for issue in result["issues"])


def test_active_execution_remains_unchecked():
    plan, execution, path = _quality_inputs()
    execution = seal_document({**execution, "status": "running"})

    result = assess_scan_path_quality(plan=plan, execution=execution, path=path)

    assert result["status"] == "unchecked"
    assert result["physical_validation"] == "not_run"


def test_failed_execution_without_frames_is_unusable():
    plan, execution, path = _quality_inputs()
    failed_attempts = [dict(attempt) for attempt in execution["attempts"]]
    failed_attempts[-1] = {**failed_attempts[-1], "status": "failed", "failure_code": "synthetic_failure"}
    execution = seal_document({**execution, "status": "failed", "attempts": failed_attempts})
    path = seal_document({**path, "frames": []})

    result = assess_scan_path_quality(plan=plan, execution=execution, path=path)

    assert result["status"] == "unusable"


def test_identity_mismatch_and_bad_tolerance_are_rejected():
    plan, execution, path = _quality_inputs()
    wrong_execution = seal_document({**execution, "acp_task_id": "other-task"})
    with pytest.raises(ContractError, match="identities do not match"):
        assess_scan_path_quality(plan=plan, execution=wrong_execution, path=path)
    with pytest.raises(ContractError, match="finite and positive"):
        assess_scan_path_quality(plan=plan, execution=execution, path=path, coordinate_tolerance=0)
