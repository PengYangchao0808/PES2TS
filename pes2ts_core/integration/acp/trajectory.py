"""Project PES2TS paths through ACP's native trajectory wire contracts."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

from pes2ts_core.contracts import ContractError, dumps_document
from pes2ts_core.utils.hashing import stable_json_dumps

HARTREE_TO_KCAL_MOL = 627.5094740631


class ACPFrameContractError(ValueError):
    """The ACP trajectory-frame API or PES2TS projection inputs are invalid."""


def _load_frame_contract(acp_source_root: str | Path) -> dict[str, Any]:
    """Import ACP's frame contract from the specified source checkout."""
    source_root = Path(acp_source_root).resolve(strict=True)
    module_path = source_root / "src" / "acp" / "results" / "frames.py"
    if not module_path.is_file():
        raise ACPFrameContractError(
            f"ACP TrajectoryFrame contract not found under {source_root}"
        )
    source_path = str(source_root / "src")
    inserted = source_path not in sys.path
    if inserted:
        sys.path.insert(0, source_path)
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        from acp.results.frames import (  # type: ignore[import-not-found]
            TrajectoryAnnotation,
            TrajectoryFrame,
            view_spec,
        )
    except ImportError as exc:
        raise ACPFrameContractError(f"cannot import ACP frame contract from {source_root}") from exc
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
        if inserted:
            try:
                sys.path.remove(source_path)
            except ValueError:
                pass
    module = sys.modules.get("acp.results.frames")
    loaded_path = Path(module.__file__).resolve() if module and getattr(module, "__file__", None) else None
    if loaded_path != module_path.resolve():
        raise ACPFrameContractError(
            f"ACP frame contract resolved to {loaded_path}, expected {module_path}"
        )
    return {"TrajectoryFrame": TrajectoryFrame, "TrajectoryAnnotation": TrajectoryAnnotation,
            "view_spec": view_spec,
            "source_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest()}


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _safe_geometry_ref(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return ""
    normalized = value.replace("\\", "/")
    parts = normalized.split("/")
    if (normalized.startswith("/") or ":" in normalized
            or any(part in {"", ".", ".."} for part in parts)):
        return ""
    if parts[0] == "WORK":
        return ""
    return normalized if parts[0] == "RESULT" else f"RESULT/{normalized}"


def path_bundle_to_acp_pes_profile(
    path_bundle: dict[str, Any],
    scan_plan: dict[str, Any],
    *,
    proposals: list[dict[str, Any]] | None = None,
    relative_reference: str = "minimum",
) -> dict[str, Any]:
    """Convert PES2TS records to ACP's canonical ``pes_profile_v2`` input.

    The result is suitable for ACP's existing PESsearch energy-graph builder
    after the caller places it at ``RESULT/pes_search/pes_profile.json`` under
    an ACP scheduler job. This function does not create or rewrite ACP files.
    """
    try:
        dumps_document(path_bundle)
        dumps_document(scan_plan)
    except ContractError as exc:
        raise ACPFrameContractError(str(exc)) from exc
    candidate = next((item for item in scan_plan.get("candidates", [])
                      if item.get("candidate_id") == path_bundle.get("candidate_id")), None)
    if (candidate is None or path_bundle.get("plan_id") != scan_plan.get("plan_id")
            or path_bundle.get("reaction_id") != scan_plan.get("reaction_id")
            or path_bundle.get("case_id") != scan_plan.get("case_id")):
        raise ACPFrameContractError("PathBundle and ScanPlan identities do not match")
    if not path_bundle.get("acp_task_id"):
        raise ACPFrameContractError("PathBundle requires an ACP task ID for profile projection")
    if relative_reference not in {"minimum", "first"}:
        raise ACPFrameContractError("relative_reference must be 'minimum' or 'first'")
    driver = next((coordinate for coordinate in candidate.get("coordinates", [])
                   if coordinate.get("role") == "driver"), None)
    if driver is None:
        raise ACPFrameContractError("ScanPlan candidate has no driver coordinate")
    planned_points = driver.get("points", [])
    path_frames = path_bundle["frames"]
    scan_values = [_finite_number(frame.get("energies", {}).get("scan_electronic", {}).get("value"))
                   for frame in path_frames]
    refined_values = [_finite_number(frame.get("energies", {}).get("refined_electronic", {}).get("value"))
                      for frame in path_frames]

    def relative(values: list[float | None]) -> list[float | None]:
        finite = [value for value in values if value is not None]
        if not finite:
            return [None] * len(values)
        reference = min(finite) if relative_reference == "minimum" else next(
            value for value in values if value is not None
        )
        return [None if value is None else (value - reference) * HARTREE_TO_KCAL_MOL
                for value in values]

    profile_frames: list[dict[str, Any]] = []
    for position, frame in enumerate(path_frames):
        target = _finite_number(frame.get("target_coordinate"))
        if target is None and position < len(planned_points):
            target = _finite_number(planned_points[position])
        actual = _finite_number(frame.get("actual_coordinate"))
        scan_channel = frame.get("energies", {}).get("scan_electronic") or {}
        refined_channel = frame.get("energies", {}).get("refined_electronic") or {}
        target_coordinates = {driver["coordinate_id"]: target} if target is not None else {}
        actual_coordinates = {driver["coordinate_id"]: actual} if actual is not None else {}
        profile_frames.append({
            "index": frame["frame_index"],
            "target_coordinate": target,
            "actual_coordinate": actual,
            "target_coordinates": target_coordinates,
            "actual_coordinates": actual_coordinates,
            "coordinate_unit": driver["unit"],
            "geometry_path": _safe_geometry_ref(frame.get("geometry_ref")),
            "scan_energy_hartree": scan_values[position],
            "scan_method_id": scan_channel.get("method_id"),
            "single_point_energy_hartree": refined_values[position],
            "single_point_method_id": refined_channel.get("method_id"),
            "single_point_status": "completed" if refined_values[position] is not None else None,
            "optimization_converged": frame.get("converged"),
            "pes2ts_frame_id": frame["frame_id"],
            "pes2ts_geometry_sha256": hashlib.sha256(
                stable_json_dumps(frame["geometry"]).encode("utf-8")
            ).hexdigest(),
        })

    recommendations: list[dict[str, Any]] = []
    recommendation_sources: dict[str, dict[str, Any]] = {}
    frame_by_id = {frame["frame_id"]: frame for frame in path_frames}
    for proposal in proposals or []:
        try:
            dumps_document(proposal)
        except ContractError as exc:
            raise ACPFrameContractError(f"SeedProposal: {exc}") from exc
        if (proposal.get("path_id") != path_bundle["object_id"]
                or proposal.get("path_content_sha256") != path_bundle["content_sha256"]
                or proposal.get("reaction_id") != path_bundle["reaction_id"]
                or proposal.get("case_id") != path_bundle["case_id"]):
            raise ACPFrameContractError("SeedProposal does not match the projected PathBundle snapshot")
        for selected in proposal.get("selected_frames", []):
            frame = frame_by_id.get(selected.get("frame_id"))
            if frame is None:
                raise ACPFrameContractError("SeedProposal references a frame absent from PathBundle")
            geometry_digest = hashlib.sha256(
                stable_json_dumps(frame["geometry"]).encode("utf-8")
            ).hexdigest()
            if selected.get("geometry_sha256") != geometry_digest:
                raise ACPFrameContractError("SeedProposal structure digest differs from its graph frame")
            recommendation_id = f"{proposal['object_id']}:{frame['frame_id']}"
            recommendations.append({
                "candidate_id": recommendation_id,
                "frame_index": frame["frame_index"],
                "confidence": None,
                "reason": selected.get("reason"),
                "recommended_type": "ts",
            })
            recommendation_sources[recommendation_id] = {
                "selection_source": proposal["selection_source"],
                "proposal_id": proposal["object_id"],
                "rule": proposal.get("rule"),
                "rank": selected["rank"],
                "score": selected.get("score"),
                "score_unit": selected.get("score_unit"),
                "geometry_sha256": geometry_digest,
            }

    kind = "distance" if driver["unit"] == "angstrom" else "coordinate"
    coordinate = {
        "coordinate_id": driver["coordinate_id"],
        "kind": kind,
        "atom_map_ids": driver["atom_map_ids"],
        "atom_indices": driver["atom_indices"],
        "unit": driver["unit"],
    }
    return {
        "schema_version": "pes_profile_v2",
        "workflow": "PESsearch",
        "mode": "bond_length_scan",
        "status": "completed" if path_bundle["status"] == "usable" else "ready_for_review",
        "stationary_point_claimed": False,
        "reaction_id": path_bundle["reaction_id"],
        "case_id": path_bundle["case_id"],
        "plan_id": path_bundle["plan_id"],
        "candidate_id": path_bundle["candidate_id"],
        "path_id": path_bundle["object_id"],
        "execution_id": path_bundle["execution_id"],
        "acp_task_id": path_bundle["acp_task_id"],
        "protocol": {"coordinate": coordinate},
        "coordinate": coordinate,
        "coordinates": [coordinate],
        "scan": {
            "frame_count": len(profile_frames),
            "frames": profile_frames,
            "quality": {
                "scan_complete": path_bundle["status"] == "usable",
                "path_status": path_bundle["status"],
            },
        },
        "energy_profile": {
            "relative_energies_kcal_mol": relative(scan_values),
            "relative_reference": relative_reference,
            "scan_energy_method_id": next((channel.get("method_id") for path_frame in path_frames
                                             for channel in [path_frame.get("energies", {}).get("scan_electronic") or {}]
                                             if channel.get("method_id")), None),
            "single_point_energy_method_id": next((channel.get("method_id") for path_frame in path_frames
                                                     for channel in [path_frame.get("energies", {}).get("refined_electronic") or {}]
                                                     if channel.get("method_id")), None),
        },
        "recommendations": {"ts": recommendations, "intermediates": []},
        "review": {"status": "pending"},
        "provenance": {
            "producer": "PES2TS",
            "reaction_id": path_bundle["reaction_id"],
            "case_id": path_bundle["case_id"],
            "plan_id": path_bundle["plan_id"],
            "candidate_id": path_bundle["candidate_id"],
            "path_id": path_bundle["object_id"],
            "path_content_sha256": path_bundle["content_sha256"],
            "execution_id": path_bundle["execution_id"],
            "acp_task_id": path_bundle["acp_task_id"],
            "recommendation_sources": recommendation_sources,
            "frame_id_map": {str(frame["frame_index"]): frame["frame_id"] for frame in path_frames},
        },
    }


def project_path_bundle_to_acp_graph(
    path_bundle: dict[str, Any],
    scan_plan: dict[str, Any],
    *,
    acp_source_root: str | Path,
    proposals: list[dict[str, Any]] | None = None,
    relative_reference: str = "minimum",
) -> dict[str, Any]:
    """Build an ACP Energy & Trajectory graph using ACP's frame/annotation classes.

    Geometry stays in the returned ``geometries`` table, indexed by the exact
    graph node ID. Existing ACP RESULT geometry references are carried through
    for remote resolution; callers that publish inline frames must first write
    and manifest the XYZ files in RESULT, then replace the empty reference.
    """
    try:
        dumps_document(path_bundle)
        dumps_document(scan_plan)
    except ContractError as exc:
        raise ACPFrameContractError(str(exc)) from exc
    if (path_bundle.get("plan_id") != scan_plan.get("plan_id")
            or path_bundle.get("candidate_id") not in {
                candidate.get("candidate_id") for candidate in scan_plan.get("candidates", [])
            }
            or path_bundle.get("reaction_id") != scan_plan.get("reaction_id")
            or path_bundle.get("case_id") != scan_plan.get("case_id")):
        raise ACPFrameContractError("PathBundle and ScanPlan identities do not match")
    task_id = path_bundle.get("acp_task_id")
    if not isinstance(task_id, str) or not task_id:
        raise ACPFrameContractError("PathBundle requires an ACP task ID for graph projection")
    if relative_reference not in {"minimum", "first"}:
        raise ACPFrameContractError("relative_reference must be 'minimum' or 'first'")

    contract = _load_frame_contract(acp_source_root)
    view = contract["view_spec"]("scan")
    frames = path_bundle["frames"]
    candidate = next(item for item in scan_plan["candidates"]
                     if item["candidate_id"] == path_bundle["candidate_id"])
    driver = next((coord for coord in candidate["coordinates"] if coord.get("role") == "driver"), None)
    if driver is None:
        raise ACPFrameContractError("ScanPlan candidate has no driver coordinate")
    planned_points = driver.get("points", [])
    scan_values = [_finite_number(frame.get("energies", {}).get("scan_electronic", {}).get("value"))
                   for frame in frames]
    refined_values = [_finite_number(frame.get("energies", {}).get("refined_electronic", {}).get("value"))
                      for frame in frames]

    def relative(values: list[float | None]) -> list[float | None]:
        finite = [value for value in values if value is not None]
        if not finite:
            return [None] * len(values)
        reference = min(finite) if relative_reference == "minimum" else next(
            value for value in values if value is not None
        )
        return [None if value is None else (value - reference) * HARTREE_TO_KCAL_MOL
                for value in values]

    relative_scan = relative(scan_values)
    relative_refined = relative(refined_values)
    nodes: list[dict[str, Any]] = []
    geometry_table: dict[str, dict[str, Any]] = {}
    for position, frame in enumerate(frames):
        frame_index = frame["frame_index"]
        target = _finite_number(frame.get("target_coordinate"))
        if target is None and position < len(planned_points):
            target = _finite_number(planned_points[position])
        actual = _finite_number(frame.get("actual_coordinate"))
        x = target if target is not None else actual
        if x is None:
            x = float(frame_index)
        geometry = frame["geometry"]
        geometry_sha256 = hashlib.sha256(stable_json_dumps(geometry).encode("utf-8")).hexdigest()
        node_id = frame["frame_id"]
        geometry_table[node_id] = {
            "frame_id": node_id,
            "frame_index": frame_index,
            "atom_map_ids": frame["atom_map_ids"],
            "elements": frame["elements"],
            "geometry_angstrom": geometry,
            "geometry_sha256": geometry_sha256,
        }
        status = ("converged" if frame.get("converged") is True else
                  "failed" if frame.get("converged") is False else "unknown")
        nodes.append(contract["TrajectoryFrame"](
            frame_id=node_id,
            label=f"Frame {frame_index + 1}",
            frame_index=frame_index,
            x=x,
            energy=relative_scan[position],
            status=status,
            geometry_ref=_safe_geometry_ref(frame.get("geometry_ref")),
            metadata={
                "reaction_id": path_bundle["reaction_id"],
                "case_id": path_bundle["case_id"],
                "plan_id": path_bundle["plan_id"],
                "candidate_id": path_bundle["candidate_id"],
                "path_id": path_bundle["object_id"],
                "path_content_sha256": path_bundle["content_sha256"],
                "acp_task_id": task_id,
                "source_acp_frame_id": frame.get("source_acp_frame_id"),
                "atom_map_ids": frame["atom_map_ids"],
                "elements": frame["elements"],
                "geometry_sha256": geometry_sha256,
                "target_coordinate": target,
                "actual_coordinate": actual,
                "scan_energy_hartree": scan_values[position],
                "scan_method_id": frame.get("energies", {}).get("scan_electronic", {}).get("method_id"),
                "refined_energy_hartree": refined_values[position],
                "refined_method_id": frame.get("energies", {}).get("refined_electronic", {}).get("method_id"),
            },
        ).to_node(view.node_type))

    node_by_id = {node["id"]: node for node in nodes}
    annotations: list[dict[str, Any]] = []
    for proposal in proposals or []:
        try:
            dumps_document(proposal)
        except ContractError as exc:
            raise ACPFrameContractError(f"SeedProposal: {exc}") from exc
        if (proposal.get("path_id") != path_bundle["object_id"]
                or proposal.get("path_content_sha256") != path_bundle["content_sha256"]
                or proposal.get("reaction_id") != path_bundle["reaction_id"]
                or proposal.get("case_id") != path_bundle["case_id"]):
            raise ACPFrameContractError("SeedProposal does not match the projected PathBundle snapshot")
        for selected in proposal.get("selected_frames", []):
            node = node_by_id.get(selected.get("frame_id"))
            if node is None:
                raise ACPFrameContractError("SeedProposal references a frame absent from PathBundle")
            geometry_digest = geometry_table[node["id"]]["geometry_sha256"]
            if selected.get("geometry_sha256") != geometry_digest:
                raise ACPFrameContractError("SeedProposal structure digest differs from its graph frame")
            rank = selected["rank"]
            annotations.append(contract["TrajectoryAnnotation"](
                id=f"{proposal['object_id']}:{node['id']}",
                type="ts",
                label=f"{proposal.get('rule', 'ranking')} #{rank}",
                frame_index=node["frame_index"],
                x=node["x"],
                y=node["energy"],
                status=proposal["status"],
                geometry_ref=node["geometry_ref"],
                selected=proposal["status"] == "accepted",
                metadata={
                    "selection_source": proposal["selection_source"],
                    "proposal_id": proposal["object_id"],
                    "rule": proposal.get("rule"),
                    "rank": rank,
                    "score": selected.get("score"),
                    "score_unit": selected.get("score_unit"),
                    "reason": selected.get("reason"),
                    "geometry_sha256": geometry_digest,
                    "node_id": node["id"],
                    "frame_number": node["frame_index"] + 1,
                    "scan_step": node["frame_index"],
                },
            ).to_annotation())

    finite_scan = [value for value in scan_values if value is not None]
    if finite_scan:
        minimum_index = min((i for i, value in enumerate(scan_values) if value is not None),
                            key=lambda i: scan_values[i])
        node = nodes[minimum_index]
        annotations.append(contract["TrajectoryAnnotation"](
            id=f"minimum:{node['id']}", type="minimum", label="最低扫描能",
            frame_index=node["frame_index"], x=node["x"], y=node["energy"],
            status=node["status"], geometry_ref=node["geometry_ref"],
        ).to_annotation())
    for node in nodes:
        if node["status"] == "failed":
            annotations.append(contract["TrajectoryAnnotation"](
                id=f"failed:{node['id']}", type="failed", label="未收敛",
                frame_index=node["frame_index"], x=node["x"], y=node["energy"],
                status="failed", geometry_ref=node["geometry_ref"],
            ).to_annotation())

    series: list[dict[str, Any]] = []
    for series_id, label, unit, values, method_ids in (
        ("relative_energy", "相对扫描能", "kcal/mol", relative_scan,
         [frame.get("energies", {}).get("scan_electronic", {}).get("method_id") for frame in frames]),
        ("scan_energy", "扫描能", "Eh", scan_values,
         [frame.get("energies", {}).get("scan_electronic", {}).get("method_id") for frame in frames]),
        ("single_point_energy", "精化单点能", "Eh", refined_values,
         [frame.get("energies", {}).get("refined_electronic", {}).get("method_id") for frame in frames]),
        ("relative_single_point_energy", "相对精化单点能", "kcal/mol", relative_refined,
         [frame.get("energies", {}).get("refined_electronic", {}).get("method_id") for frame in frames]),
    ):
        if any(value is not None for value in values):
            method = next((value for value in method_ids if value), None)
            series.append({"id": series_id, "label": label, "unit": unit, "axis": "left",
                           "values": values, "method_id": method})

    driver_unit = driver.get("unit") or "angstrom"
    unit_label = "Å" if driver_unit in {"angstrom", "A", "Å"} else driver_unit
    graph = {
        "job_id": task_id,
        "view_type": "scan",
        "title": "PES2TS 反应路径",
        "status": path_bundle["status"],
        "complete": path_bundle["status"] == "usable",
        "revision": path_bundle["content_sha256"][:16],
        "default_series": "relative_energy" if relative_scan and any(v is not None for v in relative_scan)
        else "scan_energy",
        "available_views": ["scan"],
        "x_axis": {"label": "扫描距离" if driver_unit == "angstrom" else "扫描坐标",
                   "unit": unit_label},
        "series": series,
        "nodes": nodes,
        "edges": [],
        "annotations": annotations,
        "source": "PES2TS PathBundle",
        "provenance": {"reaction_id": path_bundle["reaction_id"], "case_id": path_bundle["case_id"],
                       "plan_id": scan_plan["plan_id"], "candidate_id": path_bundle["candidate_id"],
                       "execution_id": path_bundle["execution_id"], "acp_task_id": task_id,
                       "acp_frame_contract_sha256": contract["source_sha256"]},
        "metadata": {"path_id": path_bundle["object_id"], "path_content_sha256": path_bundle["content_sha256"],
                     "frame_count": len(nodes), "atom_map_ids": path_bundle["atom_map_ids"],
                     "geometry_refs_missing": sum(not node["geometry_ref"] for node in nodes),
                     "relative_reference": relative_reference,
                     "structure_payload_key": "geometries[node.id]"},
        "geometries": geometry_table,
    }
    try:
        json.dumps(graph, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ACPFrameContractError(f"ACP trajectory projection is not strict JSON: {exc}") from exc
    return graph
