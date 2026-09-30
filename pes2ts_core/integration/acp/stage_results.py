"""Read-only verification and evidence extraction for ACP TS/IRC results.

The collector consumes ACP-owned RESULT manifests and never edits them.  It
returns evidence shaped for ``build_validation_result`` only after checking
the registered files, geometry identities, and the electronic/IRC signatures.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any

from pes2ts_core.integration.acp.cli_backend import ACPCLIError
from pes2ts_core.utils.hashing import stable_json_dumps


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(raw: Any) -> PurePosixPath:
    if not isinstance(raw, str) or not raw or raw.startswith(("/", "\\")) or ":" in raw:
        raise ACPCLIError("ACP product path must be relative to RESULT")
    portable = raw.replace("\\", "/")
    if any(part in {"", ".", ".."} for part in portable.split("/")):
        raise ACPCLIError("ACP product path contains an unsafe component")
    return PurePosixPath(portable)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ACPCLIError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise ACPCLIError(f"{label} must contain a JSON object")
    return payload


def _verified_manifest(task_root: str | Path, workflow: str) -> dict[str, Any]:
    root = Path(task_root).expanduser().resolve(strict=True)
    result_root = root / "RESULT"
    manifest_path = result_root / "result_manifest.json"
    try:
        manifest = _read_json(manifest_path, "ACP result manifest")
    except FileNotFoundError as exc:
        raise ACPCLIError("ACP result manifest is missing") from exc
    if (manifest.get("version") != 2 or manifest.get("workflow") != workflow
            or manifest.get("status") != "completed"
            or not isinstance(manifest.get("products"), list)):
        raise ACPCLIError(f"ACP result is not a completed v2 {workflow} manifest")
    products: dict[str, dict[str, Any]] = {}
    paths: set[str] = set()
    for row in manifest["products"]:
        if not isinstance(row, dict):
            raise ACPCLIError("ACP result manifest products must be objects")
        product_id = row.get("id")
        kind = row.get("kind", "file")
        if not isinstance(product_id, str) or not product_id or product_id in products:
            raise ACPCLIError("ACP result manifest has missing or duplicate product IDs")
        relative = _safe_relative_path(row.get("path"))
        relative_text = relative.as_posix()
        if relative_text in paths:
            raise ACPCLIError("ACP result manifest has duplicate product paths")
        paths.add(relative_text)
        try:
            target = (result_root / Path(*relative.parts)).resolve(strict=True)
        except OSError as exc:
            raise ACPCLIError(f"ACP result product is missing: {relative_text}") from exc
        if not target.is_relative_to(result_root.resolve()) or not target.is_file():
            raise ACPCLIError(f"ACP product is missing or resolves outside RESULT: {relative_text}")
        actual_hash = _sha256_file(target)
        metadata = row.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ACPCLIError(f"ACP product metadata is malformed: {product_id}")
        declared_hash = metadata.get("sha256")
        if declared_hash is not None and declared_hash != actual_hash:
            raise ACPCLIError(f"ACP product hash does not match: {product_id}")
        products[product_id] = {**row, "path": relative_text, "kind": kind,
                                "sha256": actual_hash, "absolute_path": target}
    return {"root": root, "result_root": result_root, "manifest": manifest,
            "manifest_sha256": _sha256_file(manifest_path), "products": products}


def _xyz(path: Path, expected_elements: list[str] | None = None) -> tuple[list[str], list[list[float]]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        count = int(lines[0].strip())
    except (OSError, IndexError, ValueError) as exc:
        raise ACPCLIError(f"not a readable XYZ geometry: {path.name}") from exc
    if count <= 0 or len(lines) < count + 2:
        raise ACPCLIError(f"XYZ atom count/rows are invalid: {path.name}")
    elements: list[str] = []
    geometry: list[list[float]] = []
    for line in lines[2:2 + count]:
        fields = line.split()
        if len(fields) < 4:
            raise ACPCLIError(f"XYZ coordinate row is incomplete: {path.name}")
        try:
            row = [float(value) for value in fields[1:4]]
        except ValueError as exc:
            raise ACPCLIError(f"XYZ coordinates are non-numeric: {path.name}") from exc
        if any(not math.isfinite(value) for value in row):
            raise ACPCLIError(f"XYZ coordinates are non-finite: {path.name}")
        elements.append(fields[0])
        geometry.append(row)
    if expected_elements is not None and elements != expected_elements:
        raise ACPCLIError(f"XYZ element order differs from ReactionCase: {path.name}")
    return elements, geometry


def _geometry_digest(geometry: list[list[float]]) -> str:
    return _sha256_bytes(stable_json_dumps(geometry).encode("utf-8"))


def _geometry_max_delta(left: list[list[float]], right: list[list[float]]) -> float:
    if len(left) != len(right) or any(len(a) != 3 or len(b) != 3 for a, b in zip(left, right)):
        return math.inf
    return max((abs(a[k] - b[k]) for a, b in zip(left, right) for k in range(3)), default=0.0)


def _stage_protocol_digest(stage: str, manifest_sha256: str,
                           provenance: dict[str, Any], product_hashes: list[str]) -> str:
    payload = {"stage": stage, "manifest_sha256": manifest_sha256,
               "provenance": provenance, "product_sha256": product_hashes}
    return _sha256_bytes(stable_json_dumps(payload).encode("utf-8"))


def _expected_vibrational_mode_count(geometry: list[list[float]]) -> int:
    """Return 3N-6 for nonlinear or 3N-5 for linear molecules."""
    atom_count = len(geometry)
    if atom_count < 2:
        return 0
    origin = geometry[0]
    vectors = [[point[k] - origin[k] for k in range(3)] for point in geometry[1:]]
    axis = next((row for row in vectors if math.sqrt(sum(x * x for x in row)) > 1e-8), None)
    if axis is None:
        return 0
    axis_norm = math.sqrt(sum(x * x for x in axis))
    is_linear = True
    for row in vectors:
        cross = [axis[1] * row[2] - axis[2] * row[1],
                 axis[2] * row[0] - axis[0] * row[2],
                 axis[0] * row[1] - axis[1] * row[0]]
        if math.sqrt(sum(x * x for x in cross)) > 1e-7 * axis_norm:
            is_linear = False
            break
    return 3 * atom_count - (5 if is_linear else 6)


def collect_batch_ts_frequency_evidence(
    *, task_root: str | Path,
    item_id: str,
    expected_elements: list[str],
    source_frame_id: str,
    source_geometry: list[list[float]],
    source_geometry_sha256: str,
    execution_id: str,
    attempt_id: str,
    expected_method: str | None = None,
    expected_basis: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract OptTS + frequency evidence from one completed TS BatchOptimize item."""
    verified = _verified_manifest(task_root, "BatchOptimize")
    products = verified["products"]
    structure_id = f"batch_{item_id}"
    modes_id = f"batch_{item_id}_normal_modes"
    structure = products.get(structure_id)
    modes_product = products.get(modes_id)
    if (structure is None or structure.get("kind") != "structure"
            or modes_product is None or modes_product.get("kind") != "frequency_modes"):
        raise ACPCLIError("completed BatchOptimize result lacks the selected TS geometry or frequency product")
    if (modes_product.get("metadata", {}).get("geometry_product_id") != structure_id):
        raise ACPCLIError("ACP normal-modes product is not bound to the selected optimized TS geometry")

    modes_doc = _read_json(modes_product["absolute_path"], "ACP normal-modes product")
    if (modes_doc.get("schema_version") != "normal_modes_v1"
            or modes_doc.get("atom_count") != len(expected_elements)
            or not isinstance(modes_doc.get("modes"), list)):
        raise ACPCLIError("ACP normal-modes product does not satisfy normal_modes_v1")
    warnings = modes_doc.get("warnings", [])
    if not isinstance(warnings, list) or warnings:
        raise ACPCLIError("ACP normal-modes product contains parser warnings")
    elements, optimized_geometry = _xyz(structure["absolute_path"], expected_elements)
    if len(elements) != len(expected_elements):
        raise ACPCLIError("optimized TS atom count does not match ReactionCase")

    result_root = verified["result_root"]
    provenance_path = result_root / "batch_provenance.json"
    provenance_doc = _read_json(provenance_path, "ACP batch provenance")
    items = provenance_doc.get("items")
    if provenance_doc.get("workflow") != "BatchOptimize" or not isinstance(items, list):
        raise ACPCLIError("ACP batch provenance has an unexpected schema")
    item_provenance = [row for row in items if isinstance(row, dict) and row.get("item_id") == item_id]
    if len(item_provenance) != 1:
        raise ACPCLIError("ACP batch provenance must contain exactly one selected item")
    item_provenance = item_provenance[0]
    effective = item_provenance.get("effective_config")
    if (item_provenance.get("role") != "ts" or not isinstance(effective, dict)
            or not isinstance(effective.get("method"), str) or not effective["method"]):
        raise ACPCLIError("ACP provenance does not confirm a TS optimization method")
    if expected_method is not None and effective.get("method") != expected_method:
        raise ACPCLIError("ACP executed method differs from the frozen validation protocol")
    actual_basis = str(effective.get("basis") or "")
    if expected_basis is not None and actual_basis != expected_basis:
        raise ACPCLIError("ACP executed basis differs from the frozen validation protocol")

    input_path = verified["root"] / "input.xyz"
    _, input_geometry = _xyz(input_path, expected_elements)
    if _geometry_digest(source_geometry) != source_geometry_sha256:
        raise ACPCLIError("supplied proposal geometry does not match its recorded content digest")
    if _geometry_max_delta(input_geometry, source_geometry) > 1e-6:
        raise ACPCLIError("ACP OptTS input geometry differs from the ranked proposal frame")

    mode_frequencies: list[float] = []
    mode_indices: set[int] = set()
    for mode in modes_doc["modes"]:
        if not isinstance(mode, dict):
            raise ACPCLIError("ACP normal-modes entries must be objects")
        value = mode.get("frequency_cm1")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ACPCLIError("ACP normal-modes product contains an invalid frequency")
        if mode.get("imaginary") is not (value < 0):
            raise ACPCLIError("ACP imaginary-mode flags disagree with the signed frequencies")
        mode_index = mode.get("mode_index")
        if not isinstance(mode_index, int) or isinstance(mode_index, bool) or mode_index in mode_indices:
            raise ACPCLIError("ACP normal-modes product has missing or duplicate mode indices")
        mode_indices.add(mode_index)
        vectors = mode.get("vectors")
        if not isinstance(vectors, list) or len(vectors) != len(expected_elements):
            raise ACPCLIError("ACP normal-mode displacement does not match the TS atom count")
        for vector in vectors:
            if (not isinstance(vector, list) or len(vector) != 3
                    or any(not isinstance(component, (int, float)) or isinstance(component, bool)
                           or not math.isfinite(component) for component in vector)):
                raise ACPCLIError("ACP normal-mode displacement contains invalid coordinates")
        if sum(component * component for vector in vectors for component in vector) <= 1e-12:
            raise ACPCLIError("ACP normal-mode displacement is identically zero")
        mode_frequencies.append(float(value))
    imaginary_count = sum(value < 0 for value in mode_frequencies)
    expected_mode_count = _expected_vibrational_mode_count(optimized_geometry)
    if expected_mode_count <= 0 or len(mode_frequencies) != expected_mode_count:
        raise ACPCLIError("ACP normal-modes product does not contain the expected 3N-6/3N-5 modes")

    provenance_hash = _sha256_file(provenance_path)
    manifest_digest = verified["manifest_sha256"]
    common_identity = {"execution_id": execution_id, "attempt_id": attempt_id,
                       "result_manifest_sha256": manifest_digest}
    optimized_digest = _geometry_digest(optimized_geometry)
    optimization_protocol = _stage_protocol_digest(
        "optts", manifest_digest, item_provenance,
        [structure["sha256"], provenance_hash])
    frequency_protocol = _stage_protocol_digest(
        "frequency", manifest_digest, {"item_id": item_id, "role": "ts",
        "effective_config": effective}, [modes_product["sha256"], provenance_hash])
    optts = {"status": "converged", **common_identity,
        "source_frame_id": source_frame_id,
        "source_geometry_sha256": source_geometry_sha256,
        "first_order_saddle": imaginary_count == 1,
        "optimized_geometry_sha256": optimized_digest,
        "optimized_structure_ref": f"RESULT/{structure['path']}",
        "optimized_structure_file_sha256": structure["sha256"],
        "method": effective["method"], "basis": actual_basis,
        "protocol_sha256": optimization_protocol}
    frequency = {"status": "passed" if imaginary_count == 1 else "failed",
        **common_identity, "imaginary_mode_count": imaginary_count,
        "source_ts_geometry_sha256": optimized_digest,
        "normal_modes_ref": f"RESULT/{modes_product['path']}",
        "normal_modes_file_sha256": modes_product["sha256"],
        "protocol_sha256": frequency_protocol}
    return optts, frequency


def _pair_distance_rms(left: list[list[float]], right: list[list[float]]) -> float:
    if len(left) != len(right) or not left:
        return math.inf
    squared: list[float] = []
    for i in range(len(left)):
        for j in range(i):
            left_distance = math.sqrt(sum((left[i][k] - left[j][k]) ** 2 for k in range(3)))
            right_distance = math.sqrt(sum((right[i][k] - right[j][k]) ** 2 for k in range(3)))
            squared.append((left_distance - right_distance) ** 2)
    return math.sqrt(sum(squared) / len(squared)) if squared else 0.0


def _aligned_rmsd(left: list[list[float]], right: list[list[float]]) -> float:
    """Atom-order-preserving Kabsch RMSD with proper rotations only."""
    if len(left) != len(right) or not left:
        return math.inf
    import numpy as np

    moving = np.asarray(left, dtype=float)
    reference = np.asarray(right, dtype=float)
    if moving.ndim != 2 or moving.shape[1] != 3 or reference.shape != moving.shape:
        return math.inf
    moving_center = moving.mean(axis=0)
    reference_center = reference.mean(axis=0)
    centered_moving = moving - moving_center
    centered_reference = reference - reference_center
    covariance = centered_moving.T @ centered_reference
    u, _, vt = np.linalg.svd(covariance)
    orientation = float(np.linalg.det(u @ vt))
    correction = np.diag([1.0, 1.0, 1.0 if orientation >= 0 else -1.0])
    rotation = u @ correction @ vt
    aligned = centered_moving @ rotation
    delta = aligned - centered_reference
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def _match_endpoint(geometry: list[list[float]], reactant: list[list[float]],
                    product: list[list[float]], tolerance_angstrom: float) -> str | None:
    reactant_rms = _aligned_rmsd(geometry, reactant)
    product_rms = _aligned_rmsd(geometry, product)
    if min(reactant_rms, product_rms) > tolerance_angstrom:
        return None
    if math.isclose(reactant_rms, product_rms, rel_tol=0.0, abs_tol=1e-8):
        return None
    return "reactant" if reactant_rms < product_rms else "product"


def _resolve_report_path(task_root: Path, raw: Any) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ACPCLIError("ACP IRC report endpoint path is missing")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = task_root / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ACPCLIError("ACP IRC report endpoint file is missing") from exc
    if not resolved.is_relative_to(task_root.resolve()) or not resolved.is_file():
        raise ACPCLIError("ACP IRC report endpoint resolves outside its task directory")
    return resolved


def collect_irc_evidence(
    *, task_root: str | Path,
    case: dict[str, Any],
    optimized_ts_geometry_sha256: str,
    optimized_ts_file_sha256: str,
    execution_id: str,
    attempt_id: str,
    expected_method: str | None = None,
    expected_basis: str | None = None,
    tolerance_angstrom: float = 0.35,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify both ACP IRC endpoints against the frozen ReactionCase geometry."""
    if (not isinstance(tolerance_angstrom, (int, float)) or isinstance(tolerance_angstrom, bool)
            or not math.isfinite(tolerance_angstrom) or tolerance_angstrom <= 0):
        raise ACPCLIError("IRC endpoint tolerance must be positive and finite")
    verified = _verified_manifest(task_root, "irc")
    products = verified["products"]
    report_product = products.get("irc_report")
    if report_product is None or report_product.get("kind") != "report":
        raise ACPCLIError("completed ACP IRC result does not register its report")
    report = _read_json(report_product["absolute_path"], "ACP IRC report")
    if report.get("workflow") != "irc" or report.get("status") != "completed":
        raise ACPCLIError("ACP IRC report does not confirm completion")
    ts_source = report.get("ts_source")
    if (not isinstance(ts_source, dict) or ts_source.get("schema") != "irc_ts_source_v1"
            or ts_source.get("geometry_sha256") != optimized_ts_file_sha256):
        raise ACPCLIError("ACP IRC source proof does not bind the exact optimized TS XYZ file")
    if not isinstance(report.get("method"), str) or not report["method"]:
        raise ACPCLIError("ACP IRC report is missing its actual method")
    if (report.get("method") != ts_source.get("method")
            or str(report.get("basis") or "") != str(ts_source.get("basis") or "")):
        raise ACPCLIError("ACP IRC method/basis differs from the verified TS source")
    reactant_charge = case["reactant"].get("charge")
    product_charge = case["product"].get("charge")
    if reactant_charge != product_charge or ts_source.get("charge") != reactant_charge:
        raise ACPCLIError("ACP IRC charge differs from the frozen ReactionCase endpoints")
    reactant_multiplicity = case["reactant"].get("multiplicity")
    product_multiplicity = case["product"].get("multiplicity")
    if (reactant_multiplicity != product_multiplicity
            or ts_source.get("multiplicity") != reactant_multiplicity):
        raise ACPCLIError("ACP IRC multiplicity differs from the frozen ReactionCase endpoints")
    if expected_method is not None and report.get("method") != expected_method:
        raise ACPCLIError("ACP IRC method differs from the frozen validation protocol")
    if expected_basis is not None and str(report.get("basis") or "") != expected_basis:
        raise ACPCLIError("ACP IRC basis differs from the frozen validation protocol")
    endpoints = report.get("endpoints")
    if not isinstance(endpoints, dict) or set(endpoints) != {"forward", "reverse"}:
        raise ACPCLIError("ACP IRC report must contain forward and reverse endpoints")
    expected_elements = [atom["element"] for atom in case["atoms"]]
    reactant = case["reactant"]["geometry"]
    product = case["product"]["geometry"]
    output: list[dict[str, Any]] = []
    manifest_digest = verified["manifest_sha256"]
    for direction in ("forward", "reverse"):
        product_id = f"irc_{direction}_endpoint"
        item = products.get(product_id)
        if item is None or item.get("kind") != "irc_endpoint":
            raise ACPCLIError(f"ACP IRC result does not register its {direction} endpoint")
        elements, geometry = _xyz(item["absolute_path"], expected_elements)
        if len(elements) != len(expected_elements):
            raise ACPCLIError(f"ACP IRC {direction} endpoint atom count is invalid")
        reported_path = _resolve_report_path(verified["root"], endpoints[direction])
        reported_elements, reported_geometry = _xyz(reported_path, expected_elements)
        if (reported_elements != elements
                or _geometry_max_delta(reported_geometry, geometry) > 1e-5):
            raise ACPCLIError(f"ACP IRC report and registered {direction} endpoint differ")
        matched = _match_endpoint(geometry, reactant, product, float(tolerance_angstrom))
        protocol = {key: report.get(key) for key in (
            "directions", "method", "basis", "maxpoints", "step", "input_role")}
        protocol["endpoint_tolerance_angstrom"] = float(tolerance_angstrom)
        protocol_sha256 = _stage_protocol_digest(
            f"irc_{direction}", manifest_digest,
            {"protocol": protocol, "ts_source": ts_source},
            [report_product["sha256"], item["sha256"]])
        output.append({"status": "matched" if matched else "failed",
            "execution_id": execution_id, "attempt_id": attempt_id,
            "result_manifest_sha256": manifest_digest,
            "endpoint_reached": matched,
            "endpoint_ref": f"RESULT/{item['path']}",
            "endpoint_file_sha256": item["sha256"],
            "source_ts_geometry_sha256": optimized_ts_geometry_sha256,
            "protocol_sha256": protocol_sha256,
            "endpoint_pair_distance_rms_angstrom": {
                "reactant": _pair_distance_rms(geometry, reactant),
                "product": _pair_distance_rms(geometry, product)},
            "endpoint_aligned_rmsd_angstrom": {
                "reactant": _aligned_rmsd(geometry, reactant),
                "product": _aligned_rmsd(geometry, product)},
        })
    return output[0], output[1]


def collect_acp_validation_result(
    *, validation_id: str,
    case: dict[str, Any],
    review_record: dict[str, Any],
    path: dict[str, Any],
    proposal: dict[str, Any],
    batch_task_root: str | Path,
    batch_item_id: str,
    batch_execution_id: str,
    batch_attempt_id: str,
    irc_task_root: str | Path,
    irc_execution_id: str,
    irc_attempt_id: str,
    expected_method: str,
    expected_basis: str,
    endpoint_tolerance_angstrom: float = 0.35,
) -> dict[str, Any]:
    """Collect ACP artifacts and assemble one content-bound ValidationResult."""
    from pes2ts_core.contracts import ContractError, dumps_document
    from pes2ts_core.integration.validation import build_validation_result

    try:
        dumps_document(case)
        dumps_document(review_record)
        dumps_document(path)
        dumps_document(proposal)
    except ContractError as exc:
        raise ACPCLIError(f"validation input contract failed: {exc}") from exc
    if case.get("status") != "ready":
        raise ACPCLIError("ACP validation requires a ready ReactionCase")
    source = case.get("source", {})
    review_output = review_record.get("extensions", {}).get("pes2ts.review_output.v1", {})
    if (review_record.get("schema_name") != "ReviewRecord"
            or review_record.get("status") != "accepted"
            or (review_record.get("reaction_id"), review_record.get("case_id"),
                review_record.get("dataset_version"), review_record.get("split"))
               != (case.get("reaction_id"), case.get("case_id"),
                   case.get("dataset_version"), case.get("split"))
            or not isinstance(source, dict)
            or source.get("review_record_id") != review_record.get("object_id")
            or source.get("reviewed_from_case_sha256") != review_record.get("case_sha256")
            or not isinstance(review_output, dict)
            or review_output.get("reviewed_case_sha256") != case.get("content_sha256")):
        raise ACPCLIError("ACP validation requires accepted human review bound to this exact ReactionCase")
    if (case.get("reaction_id"), case.get("case_id")) != (
            path.get("reaction_id"), path.get("case_id")):
        raise ACPCLIError("ReactionCase and PathBundle identities do not match")
    selected = proposal.get("selected_frames")
    if not isinstance(selected, list) or not selected or not isinstance(selected[0], dict):
        raise ACPCLIError("accepted proposal must contain a selected source frame")
    frame_id = selected[0].get("frame_id")
    frame = next((row for row in path.get("frames", [])
                  if isinstance(row, dict) and row.get("frame_id") == frame_id), None)
    if frame is None:
        raise ACPCLIError("proposal source frame is absent from the PathBundle")
    source_geometry = frame.get("geometry")
    source_digest = selected[0].get("geometry_sha256")
    if not isinstance(source_geometry, list) or not isinstance(source_digest, str):
        raise ACPCLIError("proposal source geometry and digest are required")

    elements = [atom["element"] for atom in case["atoms"]]
    optts, frequency = collect_batch_ts_frequency_evidence(
        task_root=batch_task_root, item_id=batch_item_id,
        expected_elements=elements, source_frame_id=frame_id,
        source_geometry=source_geometry, source_geometry_sha256=source_digest,
        execution_id=batch_execution_id, attempt_id=batch_attempt_id,
        expected_method=expected_method, expected_basis=expected_basis)
    irc_forward, irc_reverse = collect_irc_evidence(
        task_root=irc_task_root, case=case,
        optimized_ts_geometry_sha256=optts["optimized_geometry_sha256"],
        optimized_ts_file_sha256=optts["optimized_structure_file_sha256"],
        execution_id=irc_execution_id, attempt_id=irc_attempt_id,
        expected_method=expected_method, expected_basis=expected_basis,
        tolerance_angstrom=endpoint_tolerance_angstrom)
    return build_validation_result(validation_id=validation_id, path=path,
        proposal=proposal, optts=optts, frequency=frequency,
        irc_forward=irc_forward, irc_reverse=irc_reverse,
        validation_note=None)


__all__ = ["collect_acp_validation_result", "collect_batch_ts_frequency_evidence",
           "collect_irc_evidence"]
