"""Versioned scientific data contracts shared by PES generation and ranking.

The module intentionally uses only the standard library.  ACP-specific objects
are projected in ``integration.acp`` and never become authoritative records.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import UTC, datetime
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

from pes2ts_core.utils.hashing import stable_json_dumps

SCHEMA_VERSION = "pes2ts_contracts_v1"
OBJECTS = {
    "ExperimentManifest", "ReactionCase", "ScanPlan", "ExecutionRecord",
    "PathBundle", "SeedProposal", "ValidationResult", "ReviewRecord",
}
STATUSES = {
    "ExperimentManifest": {"frozen", "draft"},
    "ReactionCase": {"ready", "needs_review", "rejected"},
    "ScanPlan": {"ready", "needs_review", "rejected"},
    "ExecutionRecord": {"queued", "running", "completed", "failed", "cancelled"},
    "PathBundle": {"unchecked", "usable", "unusable", "needs_review"},
    "SeedProposal": {"accepted", "rejected", "needs_review"},
    "ValidationResult": {"not_run", "incomplete", "passed", "failed"},
    "ReviewRecord": {"accepted", "rejected", "needs_more_info", "replacement_required"},
}
FORBIDDEN_TRUTH_KEYS = {
    "ts_coordinates", "ts_geometry", "ts_energy", "irc_frames", "irc_coordinates",
    "endpoint_match", "orientation", "irc_evidence", "validation_label", "target_ts",
}
BASE_REQUIRED = {"schema_name", "schema_version", "object_id", "created_at", "producer", "input_refs", "content_sha256", "status"}
OBJECT_FIELDS = {
    "ExperimentManifest": {"experiment_id", "dataset_version", "splits", "policy_version", "budgets", "validation_protocol"},
    "ReactionCase": {"dataset_version", "reaction_id", "case_id", "split", "atoms", "reactant", "product", "edits", "source", "hydrogen_transfers", "review_reasons"},
    "ScanPlan": {"experiment_id", "dataset_version", "reaction_id", "case_id", "plan_id", "split", "plan_version", "atom_map_ids", "n_atoms", "candidate_strategy", "candidates", "source_case_sha256", "plan_frozen", "reject_reasons"},
    "ExecutionRecord": {"reaction_id", "case_id", "plan_id", "candidate_id", "request_id", "acp_task_id", "attempts", "total_cpu_seconds", "cost_complete", "failure_retained", "work_ref", "result_ref"},
    "PathBundle": {"reaction_id", "case_id", "plan_id", "candidate_id", "execution_id", "acp_task_id", "atom_map_ids", "n_atoms", "energy_reference", "frames", "supersedes"},
    "SeedProposal": {"reaction_id", "case_id", "path_id", "path_content_sha256", "path_status", "ranking_version", "rule", "top_k", "selection_source", "selected_frames", "rejected_reason", "review_note"},
    "ValidationResult": {"reaction_id", "case_id", "path_id", "proposal_id", "path_content_sha256", "proposal_content_sha256", "source_frame_id", "source_geometry_sha256", "optts", "frequency", "irc_forward", "irc_reverse", "validation_note"},
    "ReviewRecord": {"dataset_version", "reaction_id", "case_id", "split", "case_sha256", "workbook_sha256", "reviewers", "adjudication", "decision_source"},
}
OPTIONAL_FIELDS = {"ExperimentManifest": set(), "ReactionCase": {"review_reasons"}, "ScanPlan": {"reject_reasons"},
                   "ExecutionRecord": set(), "PathBundle": set(), "SeedProposal": {"review_note", "path_content_sha256"},
                   "ValidationResult": {"path_content_sha256", "proposal_content_sha256", "source_frame_id", "source_geometry_sha256"}, "ReviewRecord": set()}


class ContractError(ValueError):
    """Raised when a contract document cannot safely be consumed."""


def _finite_tree(value: Any, path: str = "$", issues: list[str] | None = None) -> list[str]:
    issues = [] if issues is None else issues
    if isinstance(value, float) and not math.isfinite(value):
        issues.append(f"{path}: non-finite number")
    elif isinstance(value, dict):
        for key, child in value.items():
            _finite_tree(child, f"{path}.{key}", issues)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            _finite_tree(child, f"{path}[{i}]", issues)
    return issues


def _truth_keys(value: Any, path: str = "$", issues: list[str] | None = None) -> list[str]:
    issues = [] if issues is None else issues
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_TRUTH_KEYS:
                issues.append(f"{path}.{key}: forbidden evaluation/truth field")
            _truth_keys(child, f"{path}.{key}", issues)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            _truth_keys(child, f"{path}[{i}]", issues)
    return issues


def _spin_provenance_issues(document: dict[str, Any]) -> list[str]:
    """Cross-check source-derived endpoint spin against its ReactionCase."""
    source = document.get("source")
    if not isinstance(source, dict):
        return []
    for field in ("g1_export_sha256", "g1_payload_sha256"):
        digest = source.get(field)
        if digest is not None and (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            return [f"$.source.{field}: must be 64 lowercase hexadecimal characters"]
    provenance = source.get("spin_provenance")
    if provenance is None:
        return []
    issues: list[str] = []
    if not isinstance(provenance, dict) or set(provenance) != {"reactant", "product"}:
        return ["$.source.spin_provenance: must contain reactant and product records"]
    atoms = document.get("atoms", [])
    for side in ("reactant", "product"):
        entry = provenance.get(side)
        path = f"$.source.spin_provenance.{side}"
        if not isinstance(entry, dict):
            issues.append(f"{path}: must be an object")
            continue
        expected_prefix = "R" if side == "reactant" else "P"
        if entry.get("side") != side:
            issues.append(f"{path}.side: does not match its endpoint")
        if entry.get("source_field") != "sanitized_g1_export.components":
            issues.append(f"{path}.source_field: unsupported provenance source")
        components = entry.get("components")
        if not isinstance(components, list) or not components:
            issues.append(f"{path}.components: requires at least one source component")
            continue
        component_ids = [item.get("component_id") if isinstance(item, dict) else None for item in components]
        if any(not isinstance(tag, str) or not tag.startswith(expected_prefix) for tag in component_ids):
            issues.append(f"{path}.components: IDs must be unique and match endpoint prefix {expected_prefix}")
        elif len(set(component_ids)) != len(component_ids):
            issues.append(f"{path}.components: IDs must be unique and match endpoint prefix {expected_prefix}")
        atom_counts = []
        charges = []
        multiplicities = []
        for index, component in enumerate(components):
            cpath = f"{path}.components[{index}]"
            if not isinstance(component, dict):
                continue
            count, charge, multiplicity = (component.get("atom_count"), component.get("charge"),
                                           component.get("multiplicity"))
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                issues.append(f"{cpath}.atom_count: must be a positive integer")
            else:
                atom_counts.append(count)
            if not isinstance(charge, int) or isinstance(charge, bool):
                issues.append(f"{cpath}.charge: must be an integer")
            else:
                charges.append(charge)
            if multiplicity is not None and (not isinstance(multiplicity, int) or isinstance(multiplicity, bool)
                                             or multiplicity < 1):
                issues.append(f"{cpath}.multiplicity: must be null or a positive integer")
            else:
                multiplicities.append(multiplicity)
        if len(atom_counts) == len(components) and sum(atom_counts) != len(atoms):
            issues.append(f"{path}.components: atom counts do not cover the ReactionCase atoms")
        endpoint = document.get(side)
        endpoint_charge = endpoint.get("charge") if isinstance(endpoint, dict) else None
        if len(charges) == len(components) and isinstance(endpoint_charge, int) and not isinstance(endpoint_charge, bool):
            if sum(charges) != endpoint_charge:
                issues.append(f"{path}.components: charges do not sum to the endpoint charge")
        if len(multiplicities) != len(components):
            continue
        unknown = any(value is None for value in multiplicities)
        non_singlets = [value for value in multiplicities if value not in (None, 1)]
        if unknown:
            expected_value, expected_status, expected_reason = None, "unresolved", "component_multiplicity_missing"
        elif len(non_singlets) > 1:
            expected_value, expected_status, expected_reason = None, "unresolved", "multiple_non_singlet_components_require_spin_coupling"
        elif non_singlets:
            expected_value, expected_status, expected_reason = non_singlets[0], "resolved", "source_components_define_total_spin"
        else:
            expected_value, expected_status, expected_reason = 1, "resolved", "all_source_components_are_singlets"
        if entry.get("multiplicity") != expected_value:
            issues.append(f"{path}.multiplicity: disagrees with source-component spin resolution")
        if entry.get("status") != expected_status or entry.get("reason") != expected_reason:
            issues.append(f"{path}.status/reason: disagrees with source-component spin resolution")
        case_multiplicity = endpoint.get("multiplicity") if isinstance(endpoint, dict) else None
        if case_multiplicity != expected_value:
            issues.append(f"$.{side}.multiplicity: disagrees with source spin provenance")
    return issues


def validate_document(document: dict[str, Any], *, production_input: bool = False) -> list[str]:
    """Return all structural issues in one v1 contract document."""
    issues: list[str] = []
    kind = document.get("schema_name")
    if kind not in OBJECTS:
        return ["$.schema_name: unknown contract object"]
    for key in sorted(BASE_REQUIRED - document.keys()):
        issues.append(f"$.{key}: required field missing")
    for key in sorted((OBJECT_FIELDS[kind] - OPTIONAL_FIELDS[kind]) - document.keys()):
        issues.append(f"$.{key}: required field missing")
    allowed = BASE_REQUIRED | OBJECT_FIELDS[kind] | {"extensions"}
    for key in sorted(document.keys() - allowed):
        issues.append(f"$.{key}: unknown field; use a namespaced extension if needed")
    if document.get("schema_version") != SCHEMA_VERSION:
        issues.append("$.schema_version: unsupported version")
    if not isinstance(document.get("object_id"), str) or not document.get("object_id"):
        issues.append("$.object_id: must be a non-empty string")
    if document.get("status") not in STATUSES[kind]:
        issues.append(f"$.status: invalid {kind} status")
    if not isinstance(document.get("producer"), dict):
        issues.append("$.producer: must be an object")
    if not isinstance(document.get("input_refs"), list):
        issues.append("$.input_refs: must be an array")
    stamp = document.get("created_at")
    try:
        datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        issues.append("$.created_at: must be ISO-8601")
    issues.extend(_finite_tree(document))
    if production_input or kind in {"ReactionCase", "ScanPlan"}:
        issues.extend(_truth_keys(document))

    if kind == "ReactionCase":
        for field in ("dataset_version", "reaction_id", "case_id", "split", "atoms", "reactant", "product", "edits", "source"):
            if field not in document:
                issues.append(f"$.{field}: required field missing")
        atoms = document.get("atoms")
        if isinstance(atoms, list):
            maps = [a.get("atom_map_id") for a in atoms if isinstance(a, dict)]
            valid_maps = len(maps) == len(atoms) and all(
                isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in maps)
            if not valid_maps or len(set(maps)) != len(maps):
                issues.append("$.atoms: atom_map_id values must be unique positive integers")
            for side in ("reactant", "product"):
                endpoint = document.get(side)
                if not isinstance(endpoint, dict) or endpoint.get("charge") is None:
                    issues.append(f"$.{side}: charge is required")
                elif not isinstance(endpoint.get("charge"), int) or isinstance(endpoint.get("charge"), bool):
                    issues.append(f"$.{side}.charge: must be an integer")
                elif endpoint.get("multiplicity") is None and document.get("status") == "ready":
                    issues.append(f"$.{side}.multiplicity: required for a ready ReactionCase")
                elif endpoint.get("multiplicity") is not None and (
                    not isinstance(endpoint.get("multiplicity"), int)
                    or isinstance(endpoint.get("multiplicity"), bool)
                    or endpoint.get("multiplicity") < 1
                ):
                    issues.append(f"$.{side}.multiplicity: must be positive")
                if isinstance(endpoint, dict) and endpoint.get("geometry") is not None:
                    geom = endpoint["geometry"]
                    if not isinstance(geom, list) or len(geom) != len(atoms) or any(not isinstance(row, list) or len(row) != 3 for row in geom):
                        issues.append(f"$.{side}.geometry: expected one 3D coordinate per atom")
                    elif any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
                             for row in geom for value in row):
                        issues.append(f"$.{side}.geometry: coordinates must be finite numbers")
        if document.get("split") not in {"train", "valid", "test", "unassigned"}:
            issues.append("$.split: invalid split")
        if document.get("status") == "needs_review" and not document.get("review_reasons"):
            issues.append("$.review_reasons: required when ReactionCase needs review")
        if document.get("status") == "ready" and document.get("review_reasons"):
            issues.append("$.review_reasons: resolve review items before marking ReactionCase ready")
        issues.extend(_spin_provenance_issues(document))

    if kind == "ScanPlan":
        candidates = document.get("candidates")
        if not isinstance(candidates, list):
            issues.append("$.candidates: must be an array")
        elif document.get("status") == "rejected":
            if candidates:
                issues.append("$.candidates: rejected plan cannot have executable candidates")
            if not document.get("reject_reasons"):
                issues.append("$.reject_reasons: rejected plan must explain refusal")
        elif not candidates:
            issues.append("$.candidates: at least one candidate is required")
        else:
            for ci, candidate in enumerate(candidates):
                coords = candidate.get("coordinates", [])
                if not coords:
                    issues.append(f"$.candidates[{ci}].coordinates: at least one coordinate is required")
                for j, coord in enumerate(coords):
                    points = coord.get("points")
                    if not isinstance(points, list) or len(points) < 3 or len(points) > 101:
                        issues.append(f"$.candidates[{ci}].coordinates[{j}].points: requires 3..101 scan points")
                    if coord.get("unit") != "angstrom":
                        issues.append(f"$.candidates[{ci}].coordinates[{j}].unit: expected angstrom")
                    if coord.get("atom_indices") and max(coord["atom_indices"]) >= document.get("n_atoms", 0):
                        issues.append(f"$.candidates[{ci}].coordinates[{j}].atom_indices: index out of bounds")
                if len(coords) > 4:
                    issues.append(f"$.candidates[{ci}].coordinates: ACP baseline supports at most 4 coordinates")

    if kind == "PathBundle":
        if not isinstance(document.get("candidate_id"), str) or not document["candidate_id"]:
            issues.append("$.candidate_id: must identify the executed ScanPlan candidate")
        frames = document.get("frames")
        if not isinstance(frames, list):
            issues.append("$.frames: must be an array")
        else:
            seen = set()
            n_atoms = document.get("n_atoms")
            top_maps = document.get("atom_map_ids")
            valid_top_maps = isinstance(top_maps, list) and len(top_maps) == n_atoms and all(
                isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in top_maps)
            if not valid_top_maps or len(set(top_maps)) != len(top_maps):
                issues.append("$.atom_map_ids: must be unique positive atom identities matching n_atoms")
            channel_methods: dict[str, set[str]] = {}
            expected_elements = None
            for i, frame in enumerate(frames):
                if not isinstance(frame, dict):
                    issues.append(f"$.frames[{i}]: must be an object")
                    continue
                fid = frame.get("frame_id")
                if not isinstance(fid, str) or not fid:
                    issues.append(f"$.frames[{i}].frame_id: must be a non-empty identity")
                elif fid in seen:
                    issues.append(f"$.frames[{i}].frame_id: duplicate frame identity")
                elif isinstance(fid, str):
                    seen.add(fid)
                if frame.get("frame_index") != i:
                    issues.append(f"$.frames[{i}].frame_index: must follow stable zero-based order")
                if frame.get("atom_map_ids") != top_maps:
                    issues.append(f"$.frames[{i}].atom_map_ids: order must exactly match PathBundle atom_map_ids")
                geometry = frame.get("geometry")
                if (not isinstance(geometry, list) or len(geometry) != n_atoms
                        or any(not isinstance(row, list) or len(row) != 3 for row in geometry)):
                    issues.append(f"$.frames[{i}].geometry: expected finite N×3 coordinates")
                elif any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
                         for row in geometry for value in row):
                    issues.append(f"$.frames[{i}].geometry: expected finite N×3 coordinates")
                elements = frame.get("elements")
                if not isinstance(elements, list) or len(elements) != n_atoms or any(not isinstance(x, str) or not x for x in elements):
                    issues.append(f"$.frames[{i}].elements: must identify every atom in map order")
                elif expected_elements is None:
                    expected_elements = elements
                elif elements != expected_elements:
                    issues.append(f"$.frames[{i}].elements: element order differs from other frames")
                channels = frame.get("energies", {})
                if not isinstance(channels, dict):
                    issues.append(f"$.frames[{i}].energies: must be channel keyed")
                else:
                    for channel, value in channels.items():
                        if value is not None and (not isinstance(value, dict) or value.get("unit") != "hartree"
                                or (value.get("method_id") is not None and (not isinstance(value.get("method_id"), str) or not value.get("method_id")))):
                            issues.append(f"$.frames[{i}].energies.{channel}: method must be identified or explicitly unknown; unit must be hartree")
                        elif value is not None:
                            if value.get("method_id") is not None:
                                channel_methods.setdefault(channel, set()).add(value["method_id"])
            for channel, methods in channel_methods.items():
                    if len(methods) > 1:
                        issues.append(f"$.frames: mixed methods in energy channel {channel}")
    if kind == "ExecutionRecord":
        attempts = document.get("attempts")
        if not isinstance(attempts, list):
            issues.append("$.attempts: must be an array")
        else:
            seen_attempts = set()
            failed_count = 0
            cpu_sum = 0.0
            valid_attempt_statuses = {"queued", "running", "completed", "failed", "cancelled"}
            for i, attempt in enumerate(attempts):
                if not isinstance(attempt, dict):
                    issues.append(f"$.attempts[{i}]: must be an object")
                    continue
                attempt_id = attempt.get("attempt_id")
                if not isinstance(attempt_id, str) or not attempt_id:
                    issues.append(f"$.attempts[{i}].attempt_id: required")
                elif attempt_id in seen_attempts:
                    issues.append(f"$.attempts[{i}].attempt_id: duplicate attempt identity")
                seen_attempts.add(attempt_id)
                if attempt.get("status") not in valid_attempt_statuses:
                    issues.append(f"$.attempts[{i}].status: invalid status")
                if attempt.get("status") == "failed":
                    failed_count += 1
                    if not isinstance(attempt.get("failure_code"), str) or not attempt["failure_code"]:
                        issues.append(f"$.attempts[{i}].failure_code: required for failed attempts")
                cpu_seconds = attempt.get("cpu_seconds")
                if cpu_seconds is not None and (not isinstance(cpu_seconds, (int, float))
                        or isinstance(cpu_seconds, bool) or cpu_seconds < 0):
                    issues.append(f"$.attempts[{i}].cpu_seconds: must be null or non-negative numeric cost")
                elif cpu_seconds is not None:
                    cpu_sum += cpu_seconds
            expected_cost_complete = all(attempt.get("cpu_seconds") is not None for attempt in attempts if isinstance(attempt, dict))
            if document.get("cost_complete") is not expected_cost_complete:
                issues.append("$.cost_complete: must reflect whether every attempt cost is known")
            total_cpu = document.get("total_cpu_seconds")
            if expected_cost_complete:
                if not isinstance(total_cpu, (int, float)) or isinstance(total_cpu, bool) or total_cpu < 0:
                    issues.append("$.total_cpu_seconds: non-negative numeric cost required when costs are complete")
                elif abs(total_cpu - cpu_sum) > 1e-6:
                    issues.append("$.total_cpu_seconds: must equal the sum of attempt CPU costs")
            elif total_cpu is not None:
                issues.append("$.total_cpu_seconds: must be null when any attempt cost is unknown")
            if document.get("failure_retained") is not (failed_count > 0):
                issues.append("$.failure_retained: must reflect whether failed attempts are retained")
            if document.get("status") == "completed" and (not attempts or attempts[-1].get("status") != "completed"):
                issues.append("$.attempts: completed execution must end with a completed attempt")
            if document.get("status") == "failed" and (not attempts or attempts[-1].get("status") != "failed"):
                issues.append("$.attempts: failed execution must end with a failed attempt")
        for key, prefix in (("work_ref", "WORK"), ("result_ref", "RESULT")):
            ref = document.get(key)
            if not isinstance(ref, str) or not ref.startswith(prefix + "/") or ".." in ref.replace("\\", "/").split("/"):
                issues.append(f"$.{key}: must be relative to {prefix} without path traversal")
    if kind == "ReviewRecord":
        for key in ("case_sha256", "workbook_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(document.get(key, ""))):
                issues.append(f"$.{key}: must be a lowercase SHA256")
        reviewers = document.get("reviewers")
        if not isinstance(reviewers, list) or len(reviewers) != 2:
            issues.append("$.reviewers: exactly two independent reviewer records are required")
        else:
            names = []
            allowed_decisions = {"pending", "accept", "reject", "replace", "needs_more_info"}
            allowed_dimension = {"not_reviewed", "confirmed", "issue"}
            allowed_scan = {"pending", "1D", "synchronized", "path_or_neb", "reject"}
            for i, reviewer in enumerate(reviewers):
                if not isinstance(reviewer, dict):
                    issues.append(f"$.reviewers[{i}]: must be an object")
                    continue
                name = reviewer.get("reviewer")
                names.append(name)
                if not isinstance(name, str) or not name.strip():
                    issues.append(f"$.reviewers[{i}].reviewer: required")
                if reviewer.get("decision") not in allowed_decisions:
                    issues.append(f"$.reviewers[{i}].decision: invalid decision")
                dimensions = reviewer.get("dimensions")
                if not isinstance(dimensions, dict) or set(dimensions) != {"reaction_center", "atom_mapping", "charge_spin", "geometry_assembly"}:
                    issues.append(f"$.reviewers[{i}].dimensions: must contain the four review dimensions")
                elif any(value not in allowed_dimension for value in dimensions.values()):
                    issues.append(f"$.reviewers[{i}].dimensions: invalid review value")
                if reviewer.get("scan_feasibility") not in allowed_scan:
                    issues.append(f"$.reviewers[{i}].scan_feasibility: invalid feasibility")
            if len(names) == 2 and names[0] == names[1]:
                issues.append("$.reviewers: reviewer identities must be distinct")
        decision = document.get("adjudication")
        if not isinstance(decision, dict):
            issues.append("$.adjudication: must be an object")
        else:
            if decision.get("decision") != document.get("status"):
                issues.append("$.adjudication.decision: must match ReviewRecord status")
            if not isinstance(decision.get("adjudicator"), str) or not decision["adjudicator"].strip():
                issues.append("$.adjudication.adjudicator: required")
            if decision.get("decision") == "accepted":
                for endpoint in ("reactant_multiplicity", "product_multiplicity"):
                    value = decision.get(endpoint)
                    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                        issues.append(f"$.adjudication.{endpoint}: positive integer required for acceptance")
                if isinstance(reviewers, list) and len(reviewers) == 2:
                    for i, reviewer in enumerate(reviewers):
                        if not isinstance(reviewer, dict):
                            continue
                        dimensions = reviewer.get("dimensions", {})
                        if reviewer.get("decision") != "accept":
                            issues.append(f"$.reviewers[{i}].decision: accepted record requires reviewer acceptance")
                        if isinstance(dimensions, dict) and any(dimensions.get(key) != "confirmed" for key in
                                ("reaction_center", "atom_mapping", "charge_spin", "geometry_assembly")):
                            issues.append(f"$.reviewers[{i}].dimensions: accepted record requires all dimensions confirmed")
                        if reviewer.get("scan_feasibility") != "1D":
                            issues.append(f"$.reviewers[{i}].scan_feasibility: accepted M1 record requires 1D")
                        multiplicities = reviewer.get("multiplicities", {})
                        if isinstance(multiplicities, dict):
                            for reviewer_endpoint, final_endpoint in (("reactant", "reactant_multiplicity"),
                                                                      ("product", "product_multiplicity")):
                                if multiplicities.get(reviewer_endpoint) != decision.get(final_endpoint):
                                    issues.append(f"$.reviewers[{i}].multiplicities.{reviewer_endpoint}: must match adjudication")
        if document.get("decision_source") != "human":
            issues.append("$.decision_source: must be human")
    if kind == "SeedProposal":
        source_digest = document.get("path_content_sha256")
        if source_digest is not None and not re.fullmatch(r"[0-9a-f]{64}", str(source_digest)):
            issues.append("$.path_content_sha256: must be a lowercase SHA256")
        if document.get("selection_source") not in {"ranking", "human", "validation"}:
            issues.append("$.selection_source: must identify ranking, human, or validation")
        if document.get("path_status") not in STATUSES["PathBundle"]:
            issues.append("$.path_status: must preserve the source PathBundle status")
        selected = document.get("selected_frames", [])
        if not isinstance(selected, list):
            issues.append("$.selected_frames: must be an array")
        elif document.get("status") == "accepted" and not selected:
            issues.append("$.selected_frames: accepted proposal must contain a candidate")
        elif document.get("status") == "rejected" and selected:
            issues.append("$.selected_frames: rejected proposal cannot contain candidates")
    if kind == "ValidationResult":
        for field in ("path_content_sha256", "proposal_content_sha256", "source_frame_id", "source_geometry_sha256"):
            value = document.get(field)
            if document.get("status") == "passed" and not value:
                issues.append(f"$.{field}: required to bind passed validation to its exact input content")
        for field in ("path_content_sha256", "proposal_content_sha256", "source_geometry_sha256"):
            value = document.get(field)
            if value is not None and (not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)):
                issues.append(f"$.{field}: must be a lowercase SHA256")
        if document.get("source_frame_id") is not None and (
                not isinstance(document["source_frame_id"], str) or not document["source_frame_id"].strip()):
            issues.append("$.source_frame_id: must be a non-empty frame identity")
        stage_fields = ("optts", "frequency", "irc_forward", "irc_reverse")
        for field in stage_fields:
            if not isinstance(document.get(field), dict):
                issues.append(f"$.{field}: must be a stage evidence object")
        for field in stage_fields:
            stage = document.get(field)
            if (isinstance(stage, dict)
                    and stage.get("status") in {"converged", "passed", "matched"}):
                task_id = stage.get("acp_task_id")
                has_task_id = isinstance(task_id, str) and bool(task_id.strip())
                if not has_task_id:
                    execution_id = stage.get("execution_id")
                    attempt_id = stage.get("attempt_id")
                    if not isinstance(execution_id, str) or not execution_id.strip():
                        issues.append(f"$.{field}.execution_id: CLI success requires an execution ID when acp_task_id is absent")
                    if not isinstance(attempt_id, str) or not attempt_id.strip():
                        issues.append(f"$.{field}.attempt_id: CLI success requires an immutable attempt ID when acp_task_id is absent")
                digest = stage.get("result_manifest_sha256")
                if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                    issues.append(f"$.{field}.result_manifest_sha256: successful stage requires a result manifest SHA256")
        if document.get("status") == "passed" and all(
            isinstance(document.get(field), dict) for field in stage_fields
        ):
            optts = document["optts"]
            frequency = document["frequency"]
            forward = document["irc_forward"]
            reverse = document["irc_reverse"]
            if optts.get("status") != "converged" or optts.get("first_order_saddle") is not True:
                issues.append("$.optts: passed validation requires a converged first-order saddle")
            if not optts.get("source_frame_id"):
                issues.append("$.optts.source_frame_id: must identify the optimized proposal frame")
            if document.get("source_frame_id") != optts.get("source_frame_id"):
                issues.append("$.source_frame_id: must match the OptTS source frame")
            optimized_digest = optts.get("optimized_geometry_sha256")
            if not isinstance(optimized_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", optimized_digest):
                issues.append("$.optts.optimized_geometry_sha256: passed validation requires the optimized structure digest")
            for field, stage in (("frequency", frequency), ("irc_forward", forward), ("irc_reverse", reverse)):
                if not isinstance(stage.get("protocol_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", stage["protocol_sha256"]):
                    issues.append(f"$.{field}.protocol_sha256: passed validation requires the executed protocol digest")
                if stage.get("source_ts_geometry_sha256") != optimized_digest:
                    issues.append(f"$.{field}.source_ts_geometry_sha256: must match the OptTS output structure")
            if not isinstance(optts.get("protocol_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", optts["protocol_sha256"]):
                issues.append("$.optts.protocol_sha256: passed validation requires the executed protocol digest")
            if frequency.get("status") != "passed" or frequency.get("imaginary_mode_count") != 1:
                issues.append("$.frequency: passed validation requires exactly one imaginary mode")
            expected_endpoints = {"reactant", "product"}
            irc_endpoints = {forward.get("endpoint_reached"), reverse.get("endpoint_reached")}
            if (forward.get("status") != "matched" or reverse.get("status") != "matched"
                    or irc_endpoints != expected_endpoints):
                issues.append("$.irc_forward/irc_reverse: both directions must match distinct R/P endpoints")
    return issues


def seal_document(document: dict[str, Any]) -> dict[str, Any]:
    """Set a stable scientific-content digest, excluding volatile metadata."""
    result = dict(document)
    payload = dict(result)
    payload.pop("content_sha256", None)
    payload.pop("created_at", None)
    payload.pop("producer", None)
    payload.pop("generated_at", None)
    result["content_sha256"] = hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()
    return result


def make_document(schema_name: str, object_id: str, status: str, **fields: Any) -> dict[str, Any]:
    doc = {"schema_name": schema_name, "schema_version": SCHEMA_VERSION, "object_id": object_id,
           "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
           "producer": {"name": "pes2ts", "version": "0.1"}, "input_refs": [], "status": status, **fields}
    doc = seal_document(doc)
    problems = validate_document(doc)
    if problems:
        raise ContractError("; ".join(problems))
    return doc


def dumps_document(document: dict[str, Any]) -> str:
    problems = validate_document(document)
    if problems:
        raise ContractError("; ".join(problems))
    if seal_document(document)["content_sha256"] != document["content_sha256"]:
        raise ContractError("$.content_sha256: digest mismatch")
    return stable_json_dumps(document)


def loads_document(payload: str) -> dict[str, Any]:
    try:
        doc = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ContractError(str(exc)) from exc
    if not isinstance(doc, dict):
        raise ContractError("contract root must be an object")
    problems = validate_document(doc)
    if problems:
        raise ContractError("; ".join(problems))
    if seal_document(doc)["content_sha256"] != doc["content_sha256"]:
        raise ContractError("$.content_sha256: digest mismatch")
    return doc


def validate_artifact_ref(ref: dict[str, Any]) -> list[str]:
    """Validate a portable package-relative ArtifactRef."""
    issues = []
    path = ref.get("relative_path", "")
    raw = str(path)
    portable = raw.replace("\\", "/")
    pure = PurePosixPath(portable)
    windows = PureWindowsPath(raw)
    if (not path or pure.is_absolute() or windows.is_absolute() or windows.drive
            or ".." in pure.parts or re.match(r"^[A-Za-z]:", portable)):
        issues.append("relative_path must be package-relative and may not escape its root")
    if not re.fullmatch(r"[0-9a-f]{64}", str(ref.get("sha256", ""))):
        issues.append("sha256 must be 64 lowercase hexadecimal characters")
    if not ref.get("artifact_id") or not ref.get("media_type"):
        issues.append("artifact_id and media_type are required")
    return issues
