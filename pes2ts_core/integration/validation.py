"""Assemble replayable physical-validation records from independent stage outputs."""
from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from pes2ts_core.contracts import ContractError, dumps_document, make_document
from pes2ts_core.utils.hashing import stable_json_dumps


_STAGES = ("optts", "frequency", "irc_forward", "irc_reverse")
_STAGE_STATES = {
    "optts": {"not_run", "queued", "running", "converged", "failed"},
    "frequency": {"not_run", "queued", "running", "passed", "failed"},
    "irc_forward": {"not_run", "queued", "running", "matched", "failed"},
    "irc_reverse": {"not_run", "queued", "running", "matched", "failed"},
}
_ACTIVE = {"queued", "running"}


def _validation_costs(stage_attempts: dict[str, list[dict[str, Any]] | None] | None,
                      stage_evidence: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if stage_attempts is not None and not isinstance(stage_attempts, dict):
        raise ContractError("stage_attempts must be a stage-keyed object or null")
    unknown = set(stage_attempts or {}) - set(_STAGES)
    if unknown:
        raise ContractError(f"unknown validation cost stage(s): {sorted(unknown)}")
    attempt_ids: set[str] = set()
    normalized: dict[str, Any] = {}
    total = 0.0
    complete = True
    for stage in _STAGES:
        if stage_attempts is None:
            rows = [] if stage_evidence[stage]["status"] == "not_run" else None
        elif stage not in stage_attempts:
            rows = [] if stage_evidence[stage]["status"] == "not_run" else None
        else:
            rows = stage_attempts[stage]
        if rows is None:
            normalized[stage] = {"attempts": None}
            complete = False
            continue
        if not isinstance(rows, list):
            raise ContractError(f"cost attempts for {stage} must be a list or null")
        stage_state = stage_evidence[stage]["status"]
        if stage_state == "not_run" and rows:
            raise ContractError(f"{stage} is marked not_run but has cost attempts")
        normalized_rows = []
        for index, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                raise ContractError(f"{stage} cost attempt {index} must be an object")
            attempt_id = row.get("attempt_id")
            if not isinstance(attempt_id, str) or not attempt_id or attempt_id in attempt_ids:
                raise ContractError("validation attempt IDs must be non-empty and globally unique")
            attempt_ids.add(attempt_id)
            state = row.get("status")
            if (not isinstance(state, str)
                    or state not in {"queued", "running", "completed", "failed", "cancelled"}):
                raise ContractError(f"invalid validation attempt status for {attempt_id}")
            cpu = row.get("cpu_seconds")
            if (cpu is not None and (not isinstance(cpu, (int, float)) or isinstance(cpu, bool)
                    or not math.isfinite(cpu) or cpu < 0)):
                raise ContractError(f"invalid CPU cost for validation attempt {attempt_id}")
            if cpu is None:
                complete = False
            else:
                total += cpu
            item = {"attempt_id": attempt_id, "status": state, "cpu_seconds": cpu}
            if state == "failed":
                failure_code = row.get("failure_code")
                if not isinstance(failure_code, str) or not failure_code:
                    raise ContractError(f"failed validation attempt {attempt_id} needs failure_code")
                item["failure_code"] = failure_code
            normalized_rows.append(item)
        if (normalized_rows and stage_state in {"converged", "passed", "matched"}
                and normalized_rows[-1]["status"] != "completed"):
            raise ContractError(f"successful {stage} evidence must end with a completed attempt")
        if stage_evidence[stage]["status"] != "not_run" and not normalized_rows:
            complete = False
        normalized[stage] = {"attempts": normalized_rows}
    return {"stages": normalized, "total_cpu_seconds": total if complete else None,
            "cost_complete": complete}


def _check_stage_identity(stage_name: str, item: dict[str, Any], state: str) -> None:
    """Require scheduler identity or a content-addressed direct-CLI result."""
    if state == "not_run":
        return
    task_id = item.get("acp_task_id")
    has_task_id = isinstance(task_id, str) and bool(task_id.strip())
    result_digest = item.get("result_manifest_sha256")
    needs_result_digest = state in {"converged", "passed", "matched"}
    if needs_result_digest and (not isinstance(result_digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", result_digest)):
        raise ContractError(f"{stage_name} successful evidence must bind a result manifest SHA256")
    if result_digest is not None and (not isinstance(result_digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", result_digest)):
        raise ContractError(f"{stage_name} result manifest digest must be a lowercase SHA256")
    if has_task_id:
        return
    execution_id = item.get("execution_id")
    attempt_id = item.get("attempt_id")
    if not isinstance(execution_id, str) or not execution_id.strip():
        raise ContractError(f"{stage_name} evidence must retain its ACP task ID or CLI execution ID")
    if not isinstance(attempt_id, str) or not attempt_id.strip():
        raise ContractError(f"{stage_name} CLI evidence must retain its immutable attempt ID")
    if not has_task_id and state not in _ACTIVE and not needs_result_digest:
        # Failed/cancelled CLI attempts have no completed manifest, so their
        # immutable receipt attempt ID is the available execution evidence.
        return


def build_validation_result(*, validation_id: str, path: dict[str, Any], proposal: dict[str, Any],
                             optts: dict[str, Any], frequency: dict[str, Any],
                             irc_forward: dict[str, Any], irc_reverse: dict[str, Any],
                             stage_attempts: dict[str, list[dict[str, Any]] | None] | None = None,
                             validation_note: str | None = None) -> dict[str, Any]:
    """Create a strict ValidationResult without promoting incomplete evidence.

    Stage evidence is preserved verbatim apart from contract serialization. A
    proposal must be accepted from a usable path, and OptTS must start from its
    highest-ranked proposed frame. The final status is derived from all four
    stages; callers cannot request ``passed`` directly.
    """
    for name, document in (("PathBundle", path), ("SeedProposal", proposal)):
        try:
            dumps_document(document)
        except ContractError as exc:
            raise ContractError(f"{name}: {exc}") from exc
    if path.get("status") != "usable":
        raise ContractError("physical validation requires a technically usable PathBundle")
    if (proposal.get("status") != "accepted" or proposal.get("path_status") != "usable"
            or proposal.get("path_id") != path.get("object_id")
            or proposal.get("path_content_sha256") != path.get("content_sha256")
            or proposal.get("reaction_id") != path.get("reaction_id")
            or proposal.get("case_id") != path.get("case_id")):
        raise ContractError("ValidationResult requires an accepted proposal bound to this usable path")
    selected = proposal.get("selected_frames", [])
    if not selected:
        raise ContractError("accepted proposal has no initial-guess frame")
    source_frame_id = selected[0].get("frame_id")
    source_frame = next((frame for frame in path.get("frames", [])
                         if frame.get("frame_id") == source_frame_id), None)
    if source_frame is None:
        raise ContractError("proposal initial-guess frame is absent from its PathBundle")
    actual_geometry_digest = hashlib.sha256(stable_json_dumps(source_frame["geometry"]).encode()).hexdigest()
    if selected[0].get("geometry_sha256") != actual_geometry_digest:
        raise ContractError("proposal initial-guess geometry is stale relative to its PathBundle")
    evidence = {"optts": optts, "frequency": frequency,
                "irc_forward": irc_forward, "irc_reverse": irc_reverse}
    for stage, item in evidence.items():
        if not isinstance(item, dict):
            raise ContractError(f"{stage} evidence must be an object")
        state = item.get("status")
        if not isinstance(state, str) or state not in _STAGE_STATES[stage]:
            raise ContractError(f"unsupported {stage} evidence status: {state!r}")
        _check_stage_identity(stage, item, state)
    hard_failures: list[str] = []
    pending: list[str] = []
    optts_state = optts["status"]
    if optts_state == "failed":
        hard_failures.append("OptTS failed")
    elif optts_state in _ACTIVE or optts_state == "not_run":
        pending.append(f"OptTS is {optts_state}")
    elif optts.get("first_order_saddle") is not True:
        hard_failures.append("OptTS did not confirm a first-order saddle")
    elif optts.get("source_frame_id") != source_frame_id:
        hard_failures.append("OptTS source frame differs from the proposal initial guess")
    optimized_geometry_digest = optts.get("optimized_geometry_sha256")
    if optts_state == "converged" and (not isinstance(optimized_geometry_digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", optimized_geometry_digest)):
        hard_failures.append("OptTS output geometry digest is missing or invalid")
    if optts_state == "converged" and (not isinstance(optts.get("protocol_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", optts["protocol_sha256"])):
        hard_failures.append("OptTS executed protocol digest is missing or invalid")

    frequency_state = frequency["status"]
    if frequency_state == "failed":
        hard_failures.append("frequency analysis failed")
    elif frequency_state in _ACTIVE or frequency_state == "not_run":
        pending.append(f"frequency analysis is {frequency_state}")
    elif frequency.get("imaginary_mode_count") != 1:
        hard_failures.append("frequency analysis did not find exactly one imaginary mode")
    if frequency_state == "passed" and frequency.get("source_ts_geometry_sha256") != optimized_geometry_digest:
        hard_failures.append("frequency analysis is not bound to the OptTS output geometry")
    if frequency_state == "passed" and (not isinstance(frequency.get("protocol_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", frequency["protocol_sha256"])):
        hard_failures.append("frequency executed protocol digest is missing or invalid")

    expected_endpoints = {"reactant", "product"}
    matched_endpoints = []
    for stage in ("irc_forward", "irc_reverse"):
        state = evidence[stage]["status"]
        if state == "matched" and evidence[stage].get("source_ts_geometry_sha256") != optimized_geometry_digest:
            hard_failures.append(f"{stage} is not bound to the OptTS output geometry")
        if state == "matched" and (not isinstance(evidence[stage].get("protocol_sha256"), str) or not re.fullmatch(
                r"[0-9a-f]{64}", evidence[stage]["protocol_sha256"])):
            hard_failures.append(f"{stage} executed protocol digest is missing or invalid")
        if state == "failed":
            hard_failures.append(f"{stage} failed")
        elif state in _ACTIVE or state == "not_run":
            pending.append(f"{stage} is {state}")
        else:
            endpoint = evidence[stage].get("endpoint_reached")
            matched_endpoints.append(endpoint)
            if endpoint not in expected_endpoints:
                hard_failures.append(f"{stage} did not match a reactant/product endpoint")
    if len(matched_endpoints) == 2 and set(matched_endpoints) != expected_endpoints:
        hard_failures.append("IRC directions did not reach distinct reactant and product endpoints")

    if all(evidence[stage]["status"] == "not_run" for stage in _STAGES):
        status = "not_run"
    elif hard_failures:
        status = "failed"
    elif pending:
        status = "incomplete"
    else:
        status = "passed"
    costs = _validation_costs(stage_attempts, evidence)
    # G2-AB1 WP-4: partial-path proposals are accepted; completed_interval is
    # a REPORT field (path integrity and candidate validation are reported
    # separately), and the missing constitution-§8 preparation layer is
    # explicitly typed instead of silently assumed.
    gradient_extension = {}
    if isinstance(proposal.get("extensions"), dict):
        gradient_extension = proposal["extensions"].get("pes2ts.gradient_seed_proposals.v1", {})
    completed_interval = None
    if isinstance(gradient_extension, dict) and "completed_interval" in gradient_extension:
        completed_interval = gradient_extension["completed_interval"]
    elif isinstance(proposal.get("completed_interval"), bool):
        completed_interval = proposal["completed_interval"]
    elif isinstance(selected, list) and selected and isinstance(selected[0], dict) \
            and isinstance(selected[0].get("completed_interval"), bool):
        completed_interval = selected[0]["completed_interval"]
    path_provenance = {"completed_interval": completed_interval,
                       "completed_interval_is_report_only": True,
                       "path_integrity_reported_separately": True,
                       "preparation_layer_status": "absent"}
    if validation_note is None:
        notes = hard_failures or pending
        validation_note = "; ".join(notes) if notes else (
            "All OptTS, frequency, and bidirectional IRC criteria passed." if status == "passed"
            else "Physical validation has not been run." if status == "not_run"
            else "Validation stage evidence is incomplete.")
    if not isinstance(validation_note, str) or not validation_note.strip():
        raise ContractError("validation_note must be a non-empty string")
    return make_document("ValidationResult", validation_id, status,
        reaction_id=path["reaction_id"], case_id=path["case_id"], path_id=path["object_id"],
        proposal_id=proposal["object_id"], path_content_sha256=path["content_sha256"],
        proposal_content_sha256=proposal["content_sha256"], source_frame_id=source_frame_id,
        source_geometry_sha256=actual_geometry_digest, optts=dict(optts), frequency=dict(frequency),
        irc_forward=dict(irc_forward), irc_reverse=dict(irc_reverse), validation_note=validation_note,
        extensions={"pes2ts.validation_costs.v1": costs,
                    "pes2ts.path_provenance.v1": path_provenance})
