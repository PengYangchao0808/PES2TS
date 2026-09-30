"""Technical completeness checks for ACP scan paths (not physical TS validation)."""
from __future__ import annotations

import math
from typing import Any

from pes2ts_core.contracts import ContractError, dumps_document, seal_document


def assess_scan_path_quality(*, plan: dict[str, Any], execution: dict[str, Any],
                             path: dict[str, Any], coordinate_tolerance: float = 1e-4,
                             constraint_tolerance_angstrom: float = 0.05,
                             collision_threshold_angstrom: float = 0.45) -> dict[str, Any]:
    """Classify whether an ACP scan path is technically complete and rankable.

    ``usable`` means the completed scan contains one converged, comparable
    energy frame for every frozen target coordinate in plan order. It does not
    mean that the path contains a transition state or is physically validated.
    Partial, missing, or unconverged data is ``needs_review``; terminal failure
    without usable frames is ``unusable``; active executions remain ``unchecked``.
    """
    try:
        dumps_document(plan)
        dumps_document(execution)
        dumps_document(path)
    except ContractError as exc:
        raise ContractError(f"path quality inputs: {exc}") from exc
    if plan.get("status") != "ready" or not plan.get("candidates"):
        raise ContractError("path quality requires an executable ScanPlan")
    if (not isinstance(coordinate_tolerance, (int, float)) or isinstance(coordinate_tolerance, bool)
            or not math.isfinite(coordinate_tolerance) or coordinate_tolerance <= 0):
        raise ContractError("coordinate_tolerance must be finite and positive")
    for name, value in (("constraint_tolerance_angstrom", constraint_tolerance_angstrom),
                        ("collision_threshold_angstrom", collision_threshold_angstrom)):
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or value <= 0):
            raise ContractError(f"{name} must be finite and positive")
    candidate = next((item for item in plan["candidates"]
                      if item.get("candidate_id") == path.get("candidate_id")), None)
    if candidate is None:
        raise ContractError("PathBundle candidate_id does not exist in ScanPlan")
    if (path.get("plan_id") != plan.get("plan_id")
            or path.get("reaction_id") != plan.get("reaction_id")
            or path.get("case_id") != plan.get("case_id")
            or execution.get("plan_id") != plan.get("plan_id")
            or execution.get("candidate_id") != candidate.get("candidate_id")
            or execution.get("object_id") != path.get("execution_id")
            or execution.get("acp_task_id") != path.get("acp_task_id")
            or execution.get("reaction_id") != path.get("reaction_id")
            or execution.get("case_id") != path.get("case_id")):
        raise ContractError("ExecutionRecord, ScanPlan, and PathBundle identities do not match")
    coordinates = candidate.get("coordinates", [])
    if not coordinates or not coordinates[0].get("points"):
        raise ContractError("path quality requires frozen target coordinates")
    points = coordinates[0]["points"]
    frames = path.get("frames", [])
    execution_status = execution.get("status")
    checks: dict[str, bool | None] = {
        "execution_completed": execution_status == "completed" if execution_status in {
            "completed", "failed", "cancelled"} else None,
        "frame_count_matches_plan": len(frames) == len(points) if execution_status == "completed" else None,
        "target_coordinates_complete": None,
        "target_coordinates_match_plan": None,
        "actual_coordinates_match_geometry": None,
        "actual_coordinates_satisfy_constraints": None,
        "no_atom_collisions": None,
        "all_frames_converged": None,
        "scan_energies_complete": None,
        "scan_energy_channel_comparable": None,
    }
    issues: list[str] = []
    if execution_status in {"queued", "running"}:
        issues.append("execution is still active")
        status = "unchecked"
    elif execution_status in {"failed", "cancelled"}:
        status = "needs_review" if frames else "unusable"
        issues.append(f"execution ended with status {execution_status}")
    elif execution_status != "completed":
        status = "unchecked"
        issues.append(f"execution status {execution_status!r} is not terminal")
    elif not frames:
        status = "unusable"
        issues.append("completed execution returned no trajectory frames")
    else:
        if len(frames) > len(points):
            issues.append("trajectory has more frames than frozen scan points")
        elif len(frames) < len(points):
            issues.append("trajectory is missing one or more frozen scan points")
        coordinate_ok = True
        coordinate_data_complete = True
        coordinate_mismatch = False
        actual_complete = True
        actual_mismatch = False
        constraint_satisfied = True
        constraint_data_complete = True
        collision_free = True
        pair = coordinates[0].get("atom_indices", [])
        if len(pair) != 2 or any(not isinstance(index, int) or index < 0 or index >= path["n_atoms"] for index in pair):
            raise ContractError("scan coordinate must identify two valid atom indices")
        for index, (frame, expected) in enumerate(zip(frames, points)):
            actual = frame.get("target_coordinate")
            if (not isinstance(actual, (int, float)) or isinstance(actual, bool)
                    or not math.isfinite(actual)):
                coordinate_ok = False
                coordinate_data_complete = False
                issues.append(f"frame {index} target coordinate does not match frozen scan point")
            elif abs(actual - expected) > coordinate_tolerance:
                coordinate_ok = False
                coordinate_mismatch = True
                issues.append(f"frame {index} target coordinate does not match frozen scan point")
            geometry = frame.get("geometry", [])
            try:
                delta = [geometry[pair[0]][axis] - geometry[pair[1]][axis] for axis in range(3)]
                measured = math.sqrt(sum(component * component for component in delta))
            except (IndexError, TypeError):
                measured = float("nan")
            actual = frame.get("actual_coordinate")
            if not math.isfinite(measured):
                actual_complete = False
                constraint_data_complete = False
                issues.append(f"frame {index} driver distance cannot be recomputed from geometry")
            elif not isinstance(actual, (int, float)) or isinstance(actual, bool) or not math.isfinite(actual):
                actual_complete = False
                issues.append(f"frame {index} actual coordinate is missing or non-finite")
            elif abs(actual - measured) > max(coordinate_tolerance, 0.005):
                actual_mismatch = True
                issues.append(f"frame {index} actual coordinate disagrees with its geometry")
            if math.isfinite(measured):
                residual = abs(measured - expected)
                residual_rows = frame.get("constraint_residuals", {})
                if residual_rows is not None and not isinstance(residual_rows, dict):
                    constraint_data_complete = False
                    issues.append(f"frame {index} constraint residuals are malformed")
                elif isinstance(residual_rows, dict) and any(
                    not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
                    for value in residual_rows.values()
                ):
                    constraint_data_complete = False
                    issues.append(f"frame {index} constraint residuals contain invalid values")
                if residual > constraint_tolerance_angstrom:
                    constraint_satisfied = False
                    issues.append(f"frame {index} actual geometry violates the frozen distance constraint ({residual:.4g} Å)")
                if isinstance(residual_rows, dict) and any(abs(value) > constraint_tolerance_angstrom for value in residual_rows.values()
                                                           if isinstance(value, (int, float)) and not isinstance(value, bool)):
                    constraint_satisfied = False
                    issues.append(f"frame {index} ACP reports a constraint residual above tolerance")
            geom = frame.get("geometry", [])
            for a in range(len(geom)):
                for b in range(a + 1, len(geom)):
                    try:
                        separation = math.sqrt(sum((geom[a][axis] - geom[b][axis]) ** 2 for axis in range(3)))
                    except (IndexError, TypeError):
                        separation = float("nan")
                    if math.isfinite(separation) and separation < collision_threshold_angstrom:
                        collision_free = False
                        issues.append(f"frame {index} atoms {a} and {b} overlap ({separation:.4g} Å)")
        checks["target_coordinates_complete"] = coordinate_data_complete
        checks["target_coordinates_match_plan"] = coordinate_ok
        checks["actual_coordinates_match_geometry"] = actual_complete and not actual_mismatch
        checks["actual_coordinates_satisfy_constraints"] = constraint_data_complete and constraint_satisfied
        checks["no_atom_collisions"] = collision_free
        converged = all(frame.get("converged") is True for frame in frames)
        checks["all_frames_converged"] = converged
        if not converged:
            issues.append("one or more frames are unconverged or have unknown convergence")
        energy_rows = [frame.get("energies", {}).get("scan_electronic") for frame in frames]
        energies_complete = all(
            isinstance(row, dict)
            and isinstance(row.get("value"), (int, float))
            and not isinstance(row.get("value"), bool)
            and math.isfinite(row["value"])
            for row in energy_rows
        )
        checks["scan_energies_complete"] = energies_complete
        if not energies_complete:
            issues.append("one or more scan energies are missing or non-finite")
        method_ids = {row.get("method_id") for row in energy_rows if isinstance(row, dict)}
        comparable = energies_complete and len(method_ids) == 1 and None not in method_ids and all(
            row.get("unit") == "hartree" and isinstance(row.get("method_id"), str) and row["method_id"]
            for row in energy_rows if isinstance(row, dict)
        )
        checks["scan_energy_channel_comparable"] = comparable
        if not comparable:
            issues.append("scan energy channel is not comparable across all frames")
        if (len(frames) > len(points) or coordinate_mismatch or actual_mismatch
                or not constraint_satisfied or not collision_free):
            status = "unusable"
        elif (len(frames) == len(points) and coordinate_data_complete
              and actual_complete and constraint_data_complete and constraint_satisfied and collision_free
              and converged and energies_complete and comparable):
            status = "usable"
        else:
            status = "needs_review"
    return {
        "quality_version": "acp-scan-path-v1",
        "status": status,
        "physical_validation": "not_run",
        "reaction_id": path["reaction_id"],
        "case_id": path["case_id"],
        "plan_id": plan["plan_id"],
        "candidate_id": path["candidate_id"],
        "path_id": path["object_id"],
        "execution_id": path["execution_id"],
        "acp_task_id": path["acp_task_id"],
        "input_sha256": {"plan": plan["content_sha256"],
                         "execution": execution["content_sha256"],
                         "path": path["content_sha256"]},
        "checks": checks,
        "issues": issues,
        "expected_frame_count": len(points),
        "observed_frame_count": len(frames),
        "coordinate_tolerance_angstrom": coordinate_tolerance,
        "constraint_tolerance_angstrom": constraint_tolerance_angstrom,
        "collision_threshold_angstrom": collision_threshold_angstrom,
    }


def apply_scan_path_quality(*, plan: dict[str, Any], execution: dict[str, Any],
                            path: dict[str, Any], coordinate_tolerance: float = 1e-4) -> dict[str, Any]:
    """Return the quality report and a re-sealed PathBundle carrying its status."""
    report = assess_scan_path_quality(plan=plan, execution=execution, path=path,
                                      coordinate_tolerance=coordinate_tolerance)
    updated_path = seal_document({**path, "status": report["status"]})
    report["result_path_sha256"] = updated_path["content_sha256"]
    return {"path_bundle": updated_path, "quality_assessment": report}
