"""Static ACP request/result projections for PES2TS contract objects."""
from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any

from pes2ts_core.contracts import ContractError, dumps_document, make_document


class ACPMappingError(ValueError):
    """The frozen scientific plan cannot be represented by the ACP baseline."""


_EXECUTION_STATUS = {
    "queued": "queued", "pending": "queued", "submitted": "queued",
    "running": "running", "completed": "completed", "success": "completed",
    "failed": "failed", "error": "failed", "cancelled": "cancelled", "canceled": "cancelled",
}


def _portable_acp_ref(value: str, prefix: str, field: str) -> str:
    if (not isinstance(value, str) or not value.startswith(prefix + "/")
            or value.startswith(("/", "\\")) or ":" in value
            or ".." in value.replace("\\", "/").split("/")):
        raise ACPMappingError(f"{field} must be a portable path beneath {prefix}/")
    return value.replace("\\", "/")


def acp_execution_to_record(*, case: dict[str, Any], plan: dict[str, Any], candidate_id: str,
                            execution_id: str, acp_task_id: str | None, task_status: str,
                            attempts: list[dict[str, Any]], work_ref: str, result_ref: str) -> dict[str, Any]:
    """Project a normalized ACP task/attempt summary into an auditable record.

    This is a pure result adapter. An ACP-specific reader must normalize its
    task and retry fields before calling it; unknown states are rejected and
    missing per-attempt CPU costs remain explicit nulls, never estimates.
    """
    try:
        dumps_document(case)
        dumps_document(plan)
    except ContractError as exc:
        raise ContractError(f"ACP execution inputs: {exc}") from exc
    if plan.get("status") != "ready" or (case.get("case_id"), case.get("reaction_id"), case.get("dataset_version"),
            case.get("split"), case.get("content_sha256"), [a["atom_map_id"] for a in case.get("atoms", [])]) != (
            plan.get("case_id"), plan.get("reaction_id"), plan.get("dataset_version"), plan.get("split"),
            plan.get("source_case_sha256"), plan.get("atom_map_ids")):
        raise ACPMappingError("execution plan and ReactionCase identities/source digest do not match")
    candidate = next((row for row in plan["candidates"] if row.get("candidate_id") == candidate_id), None)
    if candidate is None:
        raise ACPMappingError(f"unknown candidate_id: {candidate_id}")
    if acp_task_id is not None and (not isinstance(acp_task_id, str) or not acp_task_id):
        raise ACPMappingError("acp_task_id must be a non-empty string or null for CLI execution")
    if not isinstance(execution_id, str) or not execution_id:
        raise ACPMappingError("execution_id is required")
    state = _EXECUTION_STATUS.get(str(task_status).lower())
    if state is None:
        raise ACPMappingError(f"unsupported ACP task status: {task_status!r}")
    if not isinstance(attempts, list):
        raise ACPMappingError("ACP attempts must be supplied as a list")
    normalized_attempts = []
    seen_ids = set()
    cpu_total = 0.0
    cost_complete = True
    failed = False
    for index, attempt in enumerate(attempts, start=1):
        if not isinstance(attempt, dict):
            raise ACPMappingError(f"ACP attempt {index} must be an object")
        raw_status = str(attempt.get("status", "")).lower()
        attempt_status = _EXECUTION_STATUS.get(raw_status)
        if attempt_status is None:
            raise ACPMappingError(f"unsupported ACP attempt status: {raw_status!r}")
        attempt_id = attempt.get("attempt_id") or f"{candidate['request_id']}:attempt-{index:03d}"
        if not isinstance(attempt_id, str) or not attempt_id or attempt_id in seen_ids:
            raise ACPMappingError("ACP attempt IDs must be non-empty and unique")
        seen_ids.add(attempt_id)
        cpu_seconds = attempt.get("cpu_seconds")
        if cpu_seconds is not None and (not isinstance(cpu_seconds, (int, float)) or isinstance(cpu_seconds, bool)
                or not math.isfinite(cpu_seconds) or cpu_seconds < 0):
            raise ACPMappingError(f"ACP attempt {attempt_id} CPU cost must be null or finite and non-negative")
        if cpu_seconds is None:
            cost_complete = False
        else:
            cpu_total += cpu_seconds
        record_attempt = {"attempt_id": attempt_id, "status": attempt_status, "cpu_seconds": cpu_seconds}
        native_attempt_id = attempt.get("acp_attempt_id")
        if native_attempt_id is not None:
            record_attempt["acp_attempt_id"] = native_attempt_id
        if attempt_status == "failed":
            failure_code = attempt.get("failure_code")
            if not isinstance(failure_code, str) or not failure_code:
                raise ACPMappingError(f"failed ACP attempt {attempt_id} requires a failure_code")
            record_attempt["failure_code"] = failure_code
            failed = True
        for optional in ("log_ref", "started_at", "finished_at"):
            if attempt.get(optional) is not None:
                record_attempt[optional] = attempt[optional]
        normalized_attempts.append(record_attempt)
    if state == "completed" and (not normalized_attempts or normalized_attempts[-1]["status"] != "completed"):
        raise ACPMappingError("completed ACP task must include a completed final attempt")
    if state == "failed" and (not normalized_attempts or normalized_attempts[-1]["status"] != "failed"):
        raise ACPMappingError("failed ACP task must include a failed final attempt")
    return make_document("ExecutionRecord", execution_id, state,
        reaction_id=case["reaction_id"], case_id=case["case_id"], plan_id=plan["plan_id"],
        candidate_id=candidate_id, request_id=candidate["request_id"], acp_task_id=acp_task_id,
        attempts=normalized_attempts, total_cpu_seconds=cpu_total if cost_complete else None,
        cost_complete=cost_complete, failure_retained=failed,
        work_ref=_portable_acp_ref(work_ref, "WORK", "work_ref"),
        result_ref=_portable_acp_ref(result_ref, "RESULT", "result_ref"))


def scan_plan_to_acp_job_payload(case: dict[str, Any], plan: dict[str, Any], *,
                                 resources: dict[str, Any], candidate_id: str | None = None,
                                 project_id: str | None = None) -> dict[str, Any]:
    """Project a preflighted scan into the ACP V1 job-create request shape.

    ``resources`` is required so the caller must choose ACP allocation
    explicitly. CPU-hour and wall-time caps remain in the PES2TS remark because
    the ACP job-create resource object does not expose equivalent hard limits.
    ``output_dir`` is deliberately omitted; ACP owns its WORK/RESULT layout.
    This function only builds a request body and never submits it.
    """
    if not isinstance(resources, dict):
        raise ACPMappingError("ACP resources must be supplied explicitly as an object")
    preflight = scan_plan_to_acp_request(case, plan, candidate_id)
    scan = preflight["scan_request"]
    metadata = preflight["metadata"]
    remark = json.dumps({"pes2ts": metadata}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    candidate_token = metadata["candidate_id"].replace(":", "-")
    payload = {
        "workflow": "PESsearch",
        "name": f"PES2TS {metadata['reaction_id']} {candidate_token}",
        "task_name": f"PES2TS {metadata['reaction_id']} {candidate_token}",
        "molecule_name": metadata["reaction_id"],
        "remark": remark,
        "tags": ["PES2TS", metadata["experiment_id"], metadata["reaction_id"],
                 metadata["plan_id"], metadata["candidate_id"]],
        "input": {"source": scan["source"], "coordinate": scan["coordinate"], "protocol": scan["protocol"]},
        "method": {"mode": scan["mode"], "schema_id": "pes_scan", "levels": preflight["method_levels"]},
        "resources": dict(resources),
        "execution_mode": "local",
    }
    if project_id is not None:
        payload["project_id"] = project_id
    return payload


def scan_plan_to_acp_request(case: dict[str, Any], plan: dict[str, Any], candidate_id: str | None = None) -> dict[str, Any]:
    """Convert a validated ScanPlan to a preflighted ACP scan request.

    This returns a data request only. It does not submit a job or claim the
    local ACP service has accepted a request.
    """
    try:
        dumps_document(case)
    except ContractError as exc:
        raise ContractError(f"ReactionCase: {exc}") from exc
    try:
        dumps_document(plan)
    except ContractError as exc:
        raise ContractError(f"ScanPlan: {exc}") from exc
    if plan.get("status") != "ready":
        raise ACPMappingError("only a ready ScanPlan can be projected to an ACP request")
    identity_fields = ("case_id", "reaction_id", "dataset_version", "split")
    if any(plan.get(field) != case.get(field) for field in identity_fields):
        raise ACPMappingError("plan and ReactionCase identities/split do not match")
    case_atom_maps = [atom["atom_map_id"] for atom in case["atoms"]]
    if (plan.get("atom_map_ids") != case_atom_maps or plan.get("n_atoms") != len(case_atom_maps)
            or plan.get("source_case_sha256") != case.get("content_sha256")):
        raise ACPMappingError("plan atom order or source ReactionCase digest does not match")
    candidates = plan["candidates"]
    candidate = next((x for x in candidates if x["candidate_id"] == candidate_id), None) if candidate_id else candidates[0]
    if candidate is None:
        raise ACPMappingError(f"unknown candidate_id: {candidate_id}")
    scan_engine = str(candidate.get("method", {}).get("engine") or "").lower()
    if scan_engine not in {"orca", "xtb"}:
        raise ACPMappingError(f"ACP relaxed-scan backend is unsupported: {scan_engine or '<missing>'}")
    retry_policy = candidate.get("retry_policy")
    if not isinstance(retry_policy, dict):
        raise ACPMappingError("ScanPlan must freeze its per-point retry policy")
    coords = candidate["coordinates"]
    if not 3 <= len(coords[0]["points"]) <= 101 or len(coords) > 4:
        raise ACPMappingError("candidate exceeds ACP scan point/coordinate limits")
    if len({len(c["points"]) for c in coords}) != 1:
        raise ACPMappingError("ACP synchronized coordinates require equal point counts")
    atom_maps = plan["atom_map_ids"]
    atoms = case["atoms"]
    elements = [a["element"] for a in atoms]
    endpoint_name = candidate["start_endpoint"]
    geometry = case[endpoint_name]["geometry"]
    if len(geometry) != len(atom_maps) or len(elements) != len(atom_maps):
        raise ACPMappingError("endpoint geometry, elements, and atom maps do not align")
    xyz_lines = [str(len(elements)), f"PES2TS {plan['plan_id']} {candidate['candidate_id']}"]
    xyz_lines.extend(f"{element} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}"
                     for element, xyz in zip(elements, geometry, strict=True))
    xyz_text = "\n".join(xyz_lines) + "\n"
    coord = coords[0]
    coord_maps = coord.get("atom_map_ids", [])
    map_to_index = {atom_map: index for index, atom_map in enumerate(case_atom_maps)}
    if (len(coord_maps) != 2 or len(set(coord_maps)) != 2
            or any(atom_map not in map_to_index for atom_map in coord_maps)
            or coord.get("atom_indices") != [map_to_index[atom_map] for atom_map in coord_maps]):
        raise ACPMappingError("coordinate atom maps and zero-based indices do not identify the same atoms")
    point_values = coord["points"]
    expected = [point_values[0] + (point_values[-1] - point_values[0]) * i / (len(point_values) - 1)
                for i in range(len(point_values))]
    if any(abs(a - b) > 1e-6 for a, b in zip(point_values, expected, strict=True)):
        raise ACPMappingError("ACP PesScanRequest represents uniform points only; non-uniform plan needs an adapter extension")
    if len(coords) != 1:
        raise ACPMappingError("this first ACP projection supports a single distance coordinate")
    method_spec = candidate.get("method", {})
    optimizer_field_map = {
        "basis": "basis", "dispersion": "dispersion", "solvent_model": "solvent_model",
        "solvent": "solvent", "grid": "grid", "scf_convergence": "scf_convergence",
        "scf_max_iterations": "scf_max_iterations", "ri_approximation": "ri_approximation",
        "aux_j_basis": "aux_j_basis", "aux_c_basis": "aux_c_basis",
        "max_iterations": "max_iterations", "convergence": "convergence",
    }
    supported_method_keys = {"engine", "engine_version", "method", "parameter_sha256", *optimizer_field_map}
    unknown_method = {key: value for key, value in method_spec.items()
                      if key not in supported_method_keys and value not in (None, "", False)}
    if unknown_method:
        raise ACPMappingError(f"ACP scan method fields are unsupported and cannot be silently dropped: {sorted(unknown_method)}")
    if method_spec.get("solvent") and not method_spec.get("solvent_model"):
        raise ACPMappingError("a solvent requires an explicit ACP-supported solvent_model")
    acp_optimizer = {"method": method_spec["method"]}
    for source_key, acp_key in optimizer_field_map.items():
        value = method_spec.get(source_key)
        if value is not None and value != "":
            acp_optimizer[acp_key] = value
    acp_optimizer["retry_count"] = retry_policy["optimizer_retries"]
    acp_optimizer["retry_strategy"] = "previous_geometry" if retry_policy["reuse_previous_geometry"] else "original_geometry"
    method_level_fields = {"scan_optimizer_method": method_spec["method"]}
    for source_key, acp_key in optimizer_field_map.items():
        value = method_spec.get(source_key)
        if value is not None and value != "":
            method_level_fields[f"scan_optimizer_{acp_key}"] = value
    scan_request = {
        "mode": "bond_length_scan",
        "source": {"source_type": "xyz_text", "xyz_text": xyz_text,
                   "charge": case[endpoint_name]["charge"], "multiplicity": case[endpoint_name]["multiplicity"]},
        "coordinate": {"kind": "distance", "atoms": coord["atom_indices"], "unit": "angstrom",
                       "start": point_values[0], "end": point_values[-1], "n_points": len(point_values)},
        "protocol": {"scan_type": "bond_length",
                     "scan_driver": {"software": scan_engine, "mode": "relaxed_scan",
                                     "failure_policy": retry_policy["failure_policy"]},
                     "scan_optimizer": acp_optimizer,
                     "single_point": {"enabled": False}, "name": "pes2ts_scan_plan_v1"},
    }
    acp_method_levels = {
        "scan_coordinate": {"scan_coordinate_kind": "distance", "scan_bond_type": "auto",
            "scan_coordinate_start": point_values[0], "scan_coordinate_end": point_values[-1],
            "scan_coordinate_points": len(point_values)},
        "scan_driver": {"scan_mode": "relaxed_scan", "scan_reuse_previous_geometry": retry_policy["reuse_previous_geometry"],
            "scan_failure_policy": retry_policy["failure_policy"], "scan_retry_count": retry_policy["scan_retry_count"]},
        "scan_optimizer": {**method_level_fields,
            "scan_optimizer_retries": retry_policy["optimizer_retries"],
            "scan_optimizer_retry_strategy": acp_optimizer["retry_strategy"]},
    }
    return {
        "adapter_version": "pes2ts_acp_mapping_v1",
        "metadata": {"experiment_id": plan.get("experiment_id"), "reaction_id": plan["reaction_id"],
                     "case_id": plan["case_id"], "plan_id": plan["plan_id"],
                     "candidate_id": candidate["candidate_id"], "request_id": candidate["request_id"],
                     "atom_map_ids": atom_maps, "elements": elements,
                     "budget": candidate["budget"], "retry_policy": retry_policy,
                     "planned_method": method_spec},
        "method_levels": acp_method_levels,
        "scan_request": scan_request,
    }


def acp_result_to_path_bundle(*, case: dict[str, Any], plan: dict[str, Any], execution_id: str,
                              acp_task_id: str, frames: list[dict[str, Any]], candidate_id: str | None = None,
                              path_status: str = "unchecked",
                              refined_method_id: str | None = None,
                              geometry_loader: Callable[[str], list[list[float]]] | None = None) -> dict[str, Any]:
    """Project ACP ScanFrame records plus resolved geometries to PathBundle.

    A geometry loader is injected by ACP's local/remote result parser so this
    adapter never opens machine paths itself.
    """
    try:
        dumps_document(case)
    except ContractError as exc:
        raise ContractError(f"ReactionCase: {exc}") from exc
    try:
        dumps_document(plan)
    except ContractError as exc:
        raise ContractError(f"ScanPlan: {exc}") from exc
    if (case.get("case_id") != plan.get("case_id")
            or case.get("reaction_id") != plan.get("reaction_id")
            or case.get("dataset_version") != plan.get("dataset_version")
            or case.get("split") != plan.get("split")
            or plan.get("source_case_sha256") != case.get("content_sha256")
            or [atom["atom_map_id"] for atom in case.get("atoms", [])] != plan.get("atom_map_ids")):
        raise ACPMappingError("result plan and ReactionCase identities/source digest do not match")
    if plan.get("status") != "ready" or not plan.get("candidates"):
        raise ACPMappingError("only a ready ScanPlan with executable candidates can produce a PathBundle")
    selected_candidate = next((item for item in plan["candidates"]
                               if item.get("candidate_id") == candidate_id), None) if candidate_id else plan["candidates"][0]
    if selected_candidate is None:
        raise ACPMappingError(f"unknown candidate_id: {candidate_id}")
    candidate_id = selected_candidate["candidate_id"]
    normalized = []
    planned_method = selected_candidate.get("method", {})
    def actual_method_id(level: Any) -> str | None:
        if not isinstance(level, dict):
            return None
        engine = level.get("engine")
        method = level.get("method", level.get("functional"))
        if not isinstance(engine, str) or not engine or not isinstance(method, str) or not method:
            return None
        parts = [f"{engine}:{method}"]
        for key in ("engine_version", "basis", "dispersion", "solvent_model", "solvent", "grid", "scf_convergence", "ri_approximation"):
            value = level.get(key)
            if value not in (None, "", "none"):
                parts.append(f"{key}={value}")
        return "|".join(parts)

    for index, frame in enumerate(frames):
        if frame.get("atom_map_ids") not in (None, plan["atom_map_ids"]):
            raise ACPMappingError(f"frame {index} atom map identity/order differs from plan")
        geometry = frame.get("geometry_angstrom")
        if geometry is None and frame.get("geometry_path") and geometry_loader is not None:
            geometry = geometry_loader(frame["geometry_path"])
        if geometry is None:
            raise ACPMappingError(f"frame {index} geometry must be resolved by ACP's geometry parser")
        energies = {}
        scan_energy = frame.get("scan_energy_hartree")
        refined_energy = frame.get("single_point_energy_hartree", frame.get("refined_energy_hartree"))
        raw_level = frame.get("optimizer_level") or frame.get("scan_optimizer_level")
        actual_level = dict(raw_level) if isinstance(raw_level, dict) else None
        if actual_level is not None:
            actual_level.setdefault("engine", frame.get("optimizer_engine") or frame.get("engine"))
            actual_level.setdefault("engine_version", frame.get("optimizer_engine_version"))
        effective_scan_method = frame.get("scan_method_id") or actual_method_id(actual_level)
        if scan_energy is not None:
            energies["scan_electronic"] = {"value": scan_energy, "unit": "hartree",
                                            "method_id": effective_scan_method}
        effective_refined_method = frame.get("refined_method_id") or refined_method_id
        if refined_energy is not None:
            if not effective_refined_method:
                raise ACPMappingError("single-point energy is present but its effective method is unknown")
            energies["refined_electronic"] = {"value": refined_energy, "unit": "hartree",
                                                "method_id": effective_refined_method}
        native_index = frame.get("index", index)
        geometry_ref = _portable_result_geometry_ref(frame.get("geometry_path"))
        normalized.append({"frame_id": frame.get("frame_id", f"{execution_id}:f{native_index:05d}"),
                           "frame_index": index, "atom_map_ids": plan["atom_map_ids"],
                           "elements": [atom["element"] for atom in case["atoms"]],
                           "geometry": geometry, "geometry_ref": geometry_ref, "energies": energies,
                           "converged": frame.get("optimization_converged", frame.get("converged")),
                           "target_coordinate": frame.get("target_coordinate"),
                           "actual_coordinate": frame.get("actual_coordinate"),
                           "constraint_residuals": frame.get("constraint_residuals", {}),
                           "retry_history": frame.get("retry_history", []),
                           "actual_optimizer_level": actual_level,
                           "source_acp_frame_id": frame.get("acp_frame_id", native_index)})
    document = make_document("PathBundle", f"path:{execution_id}", path_status,
        reaction_id=case["reaction_id"], case_id=case["case_id"], plan_id=plan["plan_id"],
        candidate_id=candidate_id, execution_id=execution_id, acp_task_id=acp_task_id,
        atom_map_ids=plan["atom_map_ids"],
        n_atoms=len(plan["atom_map_ids"]), energy_reference="absolute_hartree",
        frames=normalized, supersedes=None,
        extensions={"pes2ts.acp_method_provenance.v1": {
            "planned_method": planned_method,
            "actual_optimizer_levels": [frame.get("actual_optimizer_level") for frame in normalized],
            "actual_method_ids": [frame.get("energies", {}).get("scan_electronic", {}).get("method_id")
                                  for frame in normalized],
        }})
    return document


def acp_s2_profile_to_path_bundle(*, case: dict[str, Any], plan: dict[str, Any],
                                  execution_id: str, acp_task_id: str | None,
                                  profile: dict[str, Any],
                                  geometry_angstrom_by_index: Mapping[int, list[list[float]]],
                                  candidate_id: str | None = None,
                                  path_status: str = "needs_review",
                                  refined_method_id: str | None = None) -> dict[str, Any]:
    """Project ACP's S2 profile plus frame geometry responses to a PathBundle.

    ``profile`` is the response from ``GET /jobs/{job_id}/s2/profile`` and
    geometries are the matching ``xyz`` responses from
    ``GET /jobs/{job_id}/s2/frame/{frame_index}``, already parsed by the ACP
    geometry reader. A successful HTTP/profile response does not imply a
    usable path or a validated transition state; callers must pass the
    separately evaluated path status.
    """
    if not isinstance(profile, dict) or profile.get("mode") != "bond_length_scan":
        raise ACPMappingError("ACP S2 profile must describe a bond_length_scan")
    if ((acp_task_id is not None and profile.get("job_id") != acp_task_id)
            or (acp_task_id is None and profile.get("job_id") not in (None, ""))):
        raise ACPMappingError("ACP S2 profile job_id does not match acp_task_id")
    profile_frames = profile.get("frames")
    if not isinstance(profile_frames, list) or not profile_frames:
        raise ACPMappingError("ACP S2 profile must contain at least one frame")
    indexed: dict[int, dict[str, Any]] = {}
    for frame in profile_frames:
        if not isinstance(frame, dict):
            raise ACPMappingError("ACP S2 profile frames must be objects")
        index = frame.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0 or index in indexed:
            raise ACPMappingError("ACP S2 frame indices must be unique non-negative integers")
        indexed[index] = frame
    expected_indices = list(range(len(indexed)))
    if sorted(indexed) != expected_indices:
        raise ACPMappingError("ACP S2 frame indices must be contiguous and zero-based")
    if set(geometry_angstrom_by_index) != set(indexed):
        raise ACPMappingError("ACP frame geometry indices must exactly match profile frame indices")

    frames = []
    for index in expected_indices:
        frame = indexed[index]
        frames.append({
            "index": index,
            "frame_id": f"{acp_task_id or execution_id}:frame:{index:05d}",
            "acp_frame_id": index,
            "atom_map_ids": plan.get("atom_map_ids"),
            "geometry_angstrom": geometry_angstrom_by_index[index],
            "geometry_path": frame.get("geometry_path", ""),
            "scan_energy_hartree": frame.get("scan_energy_hartree"),
            "single_point_energy_hartree": frame.get("single_point_energy_hartree"),
            "optimization_converged": frame.get("optimization_converged"),
            "target_coordinate": frame.get("target_coordinate"),
            "actual_coordinate": frame.get("actual_coordinate"),
            "constraint_residuals": frame.get("constraint_residuals", {}),
            "retry_history": frame.get("retry_history", []),
            "optimizer_level": frame.get("optimizer_level"),
            "optimizer_engine": frame.get("optimizer_engine"),
            "optimizer_engine_version": frame.get("optimizer_engine_version"),
        })
    bundle = acp_result_to_path_bundle(
        case=case, plan=plan, execution_id=execution_id, acp_task_id=acp_task_id,
        frames=frames, candidate_id=candidate_id, path_status=path_status,
        refined_method_id=refined_method_id)
    dumps_document(bundle)
    return bundle


def _portable_result_geometry_ref(raw_path: Any) -> str:
    """Normalize an ACP geometry path to a safe, task-relative RESULT reference.

    ACP trajectory readers can expose either a RESULT-relative path or a path
    relative to RESULT. WORK paths are intentionally not promoted to
    consumable result references; geometry remains available in the inline
    PathBundle payload in that case.
    """
    if not isinstance(raw_path, str) or not raw_path.strip():
        return ""
    value = raw_path.strip().replace("\\", "/")
    parts = value.split("/")
    if (value.startswith("/") or ":" in value or any(part in {"", ".", ".."} for part in parts)):
        return ""
    if parts[0] == "WORK":
        return ""
    if parts[0] == "RESULT":
        return value
    return "RESULT/" + value


def build_acp_result_manifest(*, task_id: str, workflow: str, status: str,
                              products: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the verified local ACP `ResultManifest` v2 wire shape."""
    seen: set[str] = set()
    for product in products:
        if not product.get("id") or product["id"] in seen:
            raise ACPMappingError("ACP product ids must be non-empty and unique")
        seen.add(product["id"])
        path = product.get("path", "")
        if not path or path.startswith(("/", "\\")) or ":" in path or ".." in path.replace("\\", "/").split("/"):
            raise ACPMappingError("ACP Product.path must be relative to RESULT and stay inside it")
        if product.get("kind", "file") not in {"structure", "frequency_modes", "energy_report", "ensemble", "trajectory", "report", "file", "pes_profile", "irc_endpoint", "thermo_report", "multireference_report", "wavefunction", "spin_diagnostics", "active_space", "state_comparison"}:
            raise ACPMappingError(f"unsupported ACP Product.kind: {product.get('kind')}")
    return {"version": 2, "task_id": task_id, "workflow": workflow, "status": status,
            "products": products}


def verify_acp_result_manifest_files(manifest: dict[str, Any], result_dir: str | Path) -> dict[str, Any]:
    """Verify RESULT files referenced by an ACP v2 manifest and their SHA256 metadata.

    This is a read-only consumer-side integrity check; ACP remains the sole
    writer of ``result_manifest.json``. Every product must carry the digest of
    its packaged file in ``metadata.sha256``. The returned summary is safe to
    attach to an execution audit record and contains no machine-local paths.
    """
    if (not isinstance(manifest, dict) or manifest.get("version") != 2
            or not isinstance(manifest.get("task_id"), str) or not manifest["task_id"]
            or not isinstance(manifest.get("workflow"), str) or not manifest["workflow"]
            or not isinstance(manifest.get("status"), str) or not manifest["status"]
            or not isinstance(manifest.get("products"), list)):
        raise ACPMappingError("invalid ACP ResultManifest v2 header or products")
    root = Path(result_dir).resolve(strict=True)
    if not root.is_dir():
        raise ACPMappingError("ACP RESULT root must be a directory")
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    verified = []
    for product in manifest["products"]:
        if not isinstance(product, dict):
            raise ACPMappingError("ACP manifest products must be objects")
        product_id = product.get("id")
        raw_path = product.get("path")
        if not isinstance(product_id, str) or not product_id or product_id in seen_ids:
            raise ACPMappingError("ACP product ids must be non-empty and unique")
        if not isinstance(raw_path, str) or not raw_path:
            raise ACPMappingError(f"ACP product {product_id} requires a RESULT-relative path")
        normalized_raw = raw_path.replace("\\", "/")
        rel = PurePosixPath(normalized_raw)
        if (rel.is_absolute() or any(part in {"", ".", ".."} for part in normalized_raw.split("/"))
                or ":" in raw_path):
            raise ACPMappingError(f"ACP product {product_id} has an unsafe RESULT path")
        portable_path = rel.as_posix()
        if portable_path in seen_paths:
            raise ACPMappingError("ACP product paths must be unique")
        try:
            path = (root / Path(*rel.parts)).resolve(strict=True)
        except OSError as exc:
            raise ACPMappingError(f"ACP product {product_id} file is missing or inaccessible") from exc
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ACPMappingError(f"ACP product {product_id} resolves outside RESULT") from exc
        if not path.is_file():
            raise ACPMappingError(f"ACP product {product_id} does not reference a regular file")
        metadata = product.get("metadata")
        expected_sha = metadata.get("sha256") if isinstance(metadata, dict) else None
        if (not isinstance(expected_sha, str) or len(expected_sha) != 64
                or any(char not in "0123456789abcdef" for char in expected_sha.lower())):
            raise ACPMappingError(f"ACP product {product_id} requires metadata.sha256")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_sha = digest.hexdigest()
        if actual_sha != expected_sha.lower():
            raise ACPMappingError(f"ACP product {product_id} SHA256 does not match its RESULT file")
        expected_size = metadata.get("size_bytes")
        actual_size = path.stat().st_size
        if expected_size is not None and (isinstance(expected_size, bool) or expected_size != actual_size):
            raise ACPMappingError(f"ACP product {product_id} size_bytes does not match its RESULT file")
        seen_ids.add(product_id)
        seen_paths.add(portable_path)
        verified.append({"id": product_id, "path": portable_path, "sha256": actual_sha,
                         "size_bytes": actual_size})
    return {"task_id": manifest["task_id"], "workflow": manifest["workflow"],
            "status": manifest["status"], "products_verified": len(verified), "products": verified}
