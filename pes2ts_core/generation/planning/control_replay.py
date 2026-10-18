"""Control-state replay branch (WP-3 §6.1 / IRC plan §5 完整回放, §8.2).

Resumes ONE E_native control-scan decision point from a
``pes2ts_control_state_snapshot_v1`` document and executes exactly one
injected candidate (zero residual or a legal nonzero command) through the
SHARED macro-step seam ``control_step._execute_logical_target`` — no
policy re-decision for the executed target, no duplicated optimizer logic,
no writes to the round's transition store or episode index (paired
branches are diagnostic, never main-episode transitions).

Budget fairness: the branch inherits the REMAINING wall/request budget
from the snapshot — wall elapsed counts from the snapshot's
``wall_elapsed_seconds`` and host requests continue from
``request_count``.  A fresh-budget replay is forbidden.  Replay refuses
(typed ``missing_history``) when any required history/budget field is
absent; budget and pending state are never fabricated.

Caller-supplied binds: ``control`` and ``spec`` are the compiled case the
snapshot was captured under; both are verified against the snapshot's
control/spec digests before anything executes.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pes2ts_core.generation.planning import control_step as control_step_mod
from pes2ts_core.generation.planning.chemical_control import ChemicalControl
from pes2ts_core.generation.planning.control_snapshot import (
    ControlSnapshotError,
    control_code_identity,
    control_document_digest,
    pending_from_payload,
    snapshot_content_sha256,
    validate_snapshot,
)
from pes2ts_core.generation.planning.control_step import (
    NativeScanPolicy,
    ScanFrame,
    TransportCallable,
    _check_scan_budget,
    _decision_context_vector,
    _execute_logical_target,
    freeze_control_environment,
)
from pes2ts_core.pesn.contracts.control import PolicyRewardSpec, ResidualControlSpec

REPLAY_BRANCH_SCHEMA = "pes2ts_control_replay_branch_v1"
REPLAY_INJECTED_POLICY_ID = "replay_injected_v1"
REPLAY_BRANCH_DOCUMENT_NAME = "control_replay_branch.json"

BRANCH_OUTCOME_ACCEPTED = "accepted"
BRANCH_OUTCOME_REJECTED = "rejected"
BRANCH_OUTCOME_SUBDIVIDED = "subdivided"
BRANCH_OUTCOME_BUDGET_STOP = "budget_stop"

#: History/budget leaves that must be present AND non-null for a snapshot
#: to be resumable.  Absence/null => typed ``missing_history``.  Nullable
#: exceptions (present but legitimately null) are checked separately.
_REQUIRED_HISTORY_PATHS: tuple[tuple[str, str], ...] = (
    ("episode", "episode_id"),
    ("episode", "reaction_id"),
    ("episode", "decision_index"),
    ("control_identity", "control_digest"),
    ("control_identity", "spec_digest"),
    ("control_identity", "environment_hash"),
    ("control_identity", "reward_spec_sha256"),
    ("physical_state", "parent_geometry"),
    ("physical_state", "parent_bonds"),
    ("physical_state", "parent_index"),
    ("physical_state", "last_accepted_lambda"),
    ("physical_state", "target_lambda"),
    ("physical_state", "pending"),
    ("budget", "max_wall_seconds"),
    ("budget", "max_host_requests"),
    ("budget", "wall_elapsed_seconds"),
    ("budget", "request_count"),
    ("counters", "decision_index"),
    ("counters", "accepted_prefix_frame_indices"),
    ("counters", "recent_rejections"),
    ("counters", "candidate_already_awarded"),
    ("history", "subdivision"),
    ("history", "rejection_memory"),
    ("rng", "kind"),
)

#: Keys that must be PRESENT but may legitimately be null
#: (``parent_request_sha256`` is null at the origin frame).
_REQUIRED_PRESENCE_PATHS: tuple[tuple[str, str], ...] = (
    ("physical_state", "parent_request_sha256"),
)


class ControlReplayError(ValueError):
    """Typed replay refusal (never a silent default)."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"control replay refused [{code}]: {detail}")


def load_snapshot(path: str | Path) -> dict[str, Any]:
    """Read + structurally validate one snapshot document from disk."""
    snapshot_path = Path(path)
    if not snapshot_path.is_file():
        raise ControlReplayError("snapshot_missing", f"no snapshot at {snapshot_path}")
    try:
        document = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ControlReplayError("snapshot_unreadable", str(exc)) from exc
    if not isinstance(document, dict):
        raise ControlReplayError("snapshot_invalid", "snapshot is not a mapping")
    try:
        validate_snapshot(document)
    except ControlSnapshotError as exc:
        raise ControlReplayError(exc.code, exc.detail) from exc
    return document


def require_complete_history(snapshot: Mapping[str, Any]) -> None:
    """Refuse replay when any required history/budget field is absent.

    Typed error ``missing_history``.  Budget and pending state are never
    fabricated: a null/absent leaf blocks replay instead of being defaulted.
    """
    if not isinstance(snapshot, Mapping):
        raise ControlReplayError("missing_history", "snapshot is not a mapping")
    if "parent_state_sha256" not in snapshot:
        raise ControlReplayError("missing_history", "snapshot lacks 'parent_state_sha256'")
    for section, key in (*_REQUIRED_HISTORY_PATHS, *_REQUIRED_PRESENCE_PATHS):
        block = snapshot.get(section)
        if not isinstance(block, Mapping):
            raise ControlReplayError(
                "missing_history", f"snapshot lacks section {section!r}"
            )
        if key not in block:
            raise ControlReplayError(
                "missing_history", f"snapshot lacks {section}.{key}"
            )
    for section, key in _REQUIRED_HISTORY_PATHS:
        if snapshot[section][key] is None:
            raise ControlReplayError(
                "missing_history", f"snapshot lacks {section}.{key}"
            )
    subdivision = snapshot["history"]["subdivision"]
    if not isinstance(subdivision, Mapping):
        raise ControlReplayError(
            "missing_history", "history.subdivision must be a mapping"
        )
    for key in ("scale", "min_scale", "max_subdivisions", "max_rejections_per_target"):
        if key not in subdivision or subdivision[key] is None:
            raise ControlReplayError(
                "missing_history", f"snapshot lacks history.subdivision.{key}"
            )
    pending = snapshot["physical_state"]["pending"]
    if not isinstance(pending, list):
        raise ControlReplayError(
            "missing_history", "physical_state.pending must be a list"
        )


def run_replay_branch(
    *,
    control: ChemicalControl,
    snapshot: Mapping[str, Any],
    candidate_u: Sequence[float],
    candidate_source: str,
    transport: TransportCallable,
    save_dir: str | Path,
    reward_spec: PolicyRewardSpec,
    spec: ResidualControlSpec,
    policy: NativeScanPolicy | None = None,
    branch_label: str = "b00",
) -> dict[str, Any]:
    """Execute ONE injected-candidate continuation from a saved decision point.

    Resumes the scan loop at the snapshot's open logical target, executes
    the provided candidate through the shared macro-step seam, then
    continues the SAME target under main-loop continuation semantics
    (midpoint acceptance keeps the residual; no policy re-decision) until
    the target is resolved, terminally failed, or the INHERITED budget
    stops the branch.  Writes artifacts under ``save_dir`` only; never
    appends to the round's transition store or episode index.
    """
    active_policy = policy or NativeScanPolicy()

    require_complete_history(snapshot)
    try:
        validate_snapshot(dict(snapshot))
    except ControlSnapshotError as exc:
        raise ControlReplayError(exc.code, exc.detail) from exc

    episode = snapshot["episode"]
    control_identity = snapshot["control_identity"]
    physical = snapshot["physical_state"]
    budget = snapshot["budget"]
    counters = snapshot["counters"]
    history = snapshot["history"]
    rng_block = snapshot["rng"]

    current_control_digest = control_document_digest(control)
    if current_control_digest != str(control_identity["control_digest"]):
        raise ControlReplayError(
            "control_mismatch",
            "compiled control digest does not match the snapshot's "
            f"control_identity.control_digest ({current_control_digest[:12]} != "
            f"{str(control_identity['control_digest'])[:12]}); refusing to replay "
            "under a different case",
        )
    current_spec_digest = str(spec.content_hash())
    if current_spec_digest != str(control_identity["spec_digest"]):
        raise ControlReplayError(
            "spec_mismatch",
            "residual spec digest does not match the snapshot's "
            f"spec_digest ({current_spec_digest[:12]} != "
            f"{str(control_identity['spec_digest'])[:12]})",
        )
    current_reward_sha = str(reward_spec.content_hash())
    if current_reward_sha != str(control_identity["reward_spec_sha256"]):
        raise ControlReplayError(
            "reward_spec_mismatch",
            "reward spec digest does not match the snapshot's "
            f"reward_spec_sha256 ({current_reward_sha[:12]} != "
            f"{str(control_identity['reward_spec_sha256'])[:12]})",
        )
    environment = freeze_control_environment(policy=active_policy, reward_spec=reward_spec)
    if str(environment["environment_hash"]) != str(control_identity["environment_hash"]):
        raise ControlReplayError(
            "environment_mismatch",
            "scan policy + reward spec freeze to a different environment hash "
            f"than the snapshot ({environment['environment_hash'][:12]} != "
            f"{str(control_identity['environment_hash'])[:12]}); the branch must "
            "run under the snapshot's frozen environment",
        )
    if str(control_identity["code_identity"]) != control_code_identity():
        raise ControlReplayError(
            "code_identity_mismatch",
            "execution-code identity drifted from the snapshot's capture; "
            "refusing to replay under different engine constants",
        )

    parent_geometry = np.asarray(physical["parent_geometry"], dtype=float)
    parent_bonds = {
        tuple(sorted((int(pair[0]), int(pair[1]))))
        for pair in physical["parent_bonds"]
    }
    parent_index = int(physical["parent_index"])
    parent_request_sha256 = physical["parent_request_sha256"]
    last_accepted_lambda = float(physical["last_accepted_lambda"])
    target_lambda = float(physical["target_lambda"])
    pending = pending_from_payload(physical["pending"])
    if not pending:
        raise ControlReplayError(
            "missing_history", "snapshot pending queue is empty; nothing to resume"
        )
    if abs(pending[0][0] - target_lambda) > 1e-12:
        raise ControlReplayError(
            "snapshot_inconsistent",
            f"pending head lambda {pending[0][0]} != physical_state.target_lambda "
            f"{target_lambda}",
        )

    request_count0 = int(budget["request_count"])
    max_wall = float(budget["max_wall_seconds"])
    max_requests = int(budget["max_host_requests"])
    wall_elapsed0 = float(budget["wall_elapsed_seconds"])

    candidate_u_t = tuple(float(v) for v in candidate_u)
    if len(candidate_u_t) != len(spec.driver_ids):
        raise ControlReplayError(
            "candidate_invalid",
            "candidate_u is not parallel to the snapshot spec's driver_ids",
        )
    base_targets = control.targets_at(target_lambda)
    legal, reason = spec.residual_is_legal(candidate_u_t, base_targets)
    if not legal:
        raise ControlReplayError(
            "candidate_illegal",
            f"injected candidate violates A(H): {reason}",
        )

    source_episode_id = str(episode["episode_id"])
    decision_index = int(episode["decision_index"])
    identity_prefix = f"{source_episode_id}:{branch_label}"
    reaction_id = str(episode["reaction_id"])

    state_for_context = control_step_mod.ControlScanState(
        control=control,
        spec=spec,
        parent_geometry=parent_geometry,
        last_accepted_lambda=last_accepted_lambda,
        target_lambda=target_lambda,
        remaining_wall_fraction=(
            max(0.0, 1.0 - wall_elapsed0 / max_wall) if max_wall > 0 else 0.0
        ),
        remaining_request_fraction=(
            max(0.0, 1.0 - request_count0 / max_requests) if max_requests > 0 else 0.0
        ),
        recent_rejections=int(counters["recent_rejections"]),
        candidate_already_awarded=bool(counters["candidate_already_awarded"]),
        decision_index=decision_index,
    )
    fresh = control_step_mod.FreshDecision(
        candidates=(candidate_u_t,),
        candidate_sources=(str(candidate_source),),
        executed_candidate_index=0,
        policy_id=REPLAY_INJECTED_POLICY_ID,
        policy_hash=str(control_identity.get("policy_hash") or "none"),
        decision_context_vector=_decision_context_vector(state_for_context),
    )

    driver_pairs = {
        tuple(sorted(pair))
        for pair in (list(driver.atoms) for driver in control.drivers)
        if len(pair) == 2
    }

    save_root = Path(save_dir)
    attempts_root = save_root / "attempts"
    save_root.mkdir(parents=True, exist_ok=True)

    frames: list[ScanFrame] = []
    events: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    decision_meta: list[dict[str, Any]] = []

    branch_wall_start = time.monotonic()
    request_count = request_count0
    open_transition: dict[str, Any] | None = None
    result: control_step_mod.LogicalTargetResult | None = None
    budget_stop_code: str | None = None
    budget_stop_detail: str | None = None

    while True:
        elapsed_total = wall_elapsed0 + (time.monotonic() - branch_wall_start)
        exhausted, _b_status, budget_code, budget_detail = _check_scan_budget(
            elapsed_total=elapsed_total,
            request_count=request_count,
            policy=active_policy,
            n_accepted=len(frames),
        )
        if exhausted:
            budget_stop_code = budget_code
            budget_stop_detail = budget_detail
            break

        pending_head = pending[0]
        result = _execute_logical_target(
            control=control,
            spec=spec,
            active_policy=active_policy,
            transport=transport,
            attempts_root=attempts_root,
            reaction_id=reaction_id,
            driver_pairs=driver_pairs,
            frames=frames,
            transitions=transitions,
            decision_meta=decision_meta,
            events=events,
            target_lambda=target_lambda,
            is_grid_point=bool(pending_head[1]),
            rejections=int(pending_head[2]),
            scale=1.0,
            chosen_u=candidate_u_t,
            chosen_source=str(candidate_source),
            identity_prefix=identity_prefix,
            decision_index=decision_index,
            fresh_decision=fresh,
            open_transition=open_transition,
            parent_geometry=parent_geometry,
            parent_bonds=parent_bonds,
            parent_index=parent_index,
            last_accepted_lambda=last_accepted_lambda,
            request_count=request_count,
            parent_request_sha256=(
                parent_request_sha256 if not frames else frames[-1].request_sha256 or None
            ),
        )
        request_count = result.request_count
        parent_geometry = result.parent_geometry
        parent_bonds = result.parent_bonds
        parent_index = result.parent_index
        last_accepted_lambda = result.last_accepted_lambda
        fresh = None
        if result.stop_scan or result.target_resolved:
            open_transition = None
            break
        open_transition = result.open_transition
        if result.pending_entry is not None:
            pending[0] = result.pending_entry

    branch_wall = time.monotonic() - branch_wall_start
    return _finalize_branch_document(
        control=control,
        snapshot=snapshot,
        episode=episode,
        control_identity=control_identity,
        rng_block=rng_block,
        environment=environment,
        candidate_u_t=candidate_u_t,
        candidate_source=str(candidate_source),
        branch_label=branch_label,
        source_episode_id=source_episode_id,
        decision_index=decision_index,
        target_lambda=target_lambda,
        last_accepted_lambda=last_accepted_lambda,
        frames=frames,
        transitions=transitions,
        open_transition=open_transition,
        result=result,
        request_count0=request_count0,
        request_count=request_count,
        branch_wall=branch_wall,
        wall_elapsed0=wall_elapsed0,
        max_wall=max_wall,
        max_requests=max_requests,
        budget_stop_code=budget_stop_code,
        budget_stop_detail=budget_stop_detail,
        save_root=save_root,
        pending=pending,
    )


def _branch_cost(
    *,
    transitions: Sequence[Mapping[str, Any]],
    open_transition: Mapping[str, Any] | None,
    request_count0: int,
    request_count: int,
) -> dict[str, Any]:
    """Macro cost of the branch ONLY (finalized + still-open transition)."""
    records = list(transitions)
    if open_transition is not None:
        records = [*records, open_transition]
    wall = sum(float(record.get("macro_wall", 0.0)) for record in records)
    cycles = sum(int(record.get("macro_cycles", 0)) for record in records)
    refusals = sum(len(record.get("refusal_reasons", ())) for record in records)
    return {
        "wall_seconds": wall,
        "host_requests": int(request_count - request_count0),
        "n_optimization_cycles": cycles,
        "n_refusals": refusals,
        "scope": "branch_only",
    }


def _finalize_branch_document(
    *,
    control: ChemicalControl,
    snapshot: Mapping[str, Any],
    episode: Mapping[str, Any],
    control_identity: Mapping[str, Any],
    rng_block: Mapping[str, Any],
    environment: Mapping[str, Any],
    candidate_u_t: tuple[float, ...],
    candidate_source: str,
    branch_label: str,
    source_episode_id: str,
    decision_index: int,
    target_lambda: float,
    last_accepted_lambda: float,
    frames: Sequence[ScanFrame],
    transitions: Sequence[Mapping[str, Any]],
    open_transition: Mapping[str, Any] | None,
    result: control_step_mod.LogicalTargetResult | None,
    request_count0: int,
    request_count: int,
    branch_wall: float,
    wall_elapsed0: float,
    max_wall: float,
    max_requests: int,
    budget_stop_code: str | None,
    budget_stop_detail: str | None,
    save_root: Path,
    pending: Sequence[tuple[float, bool, int]],
) -> dict[str, Any]:
    if budget_stop_code is not None:
        status = BRANCH_OUTCOME_BUDGET_STOP
        failure_code = budget_stop_code
        failure_detail = budget_stop_detail
    elif result is None:
        status = BRANCH_OUTCOME_SUBDIVIDED
        failure_code = "REPLAY_NO_PROGRESS"
        failure_detail = "branch ended without executing the logical target"
    elif result.stop_scan:
        status = BRANCH_OUTCOME_REJECTED
        failure_code = result.failure_code
        failure_detail = result.failure_detail
    elif result.target_resolved and result.accepted:
        status = BRANCH_OUTCOME_ACCEPTED
        failure_code = None
        failure_detail = None
    else:
        status = BRANCH_OUTCOME_SUBDIVIDED
        failure_code = None
        failure_detail = (
            "open logical target still pending without a typed terminal "
            "(midpoint progress retained; continuation budget exhausted)"
        )

    quality_frames: list[dict[str, Any]] = []
    for frame in frames:
        nominal = control.targets_at(frame.lambda_value)
        actual = [
            frame.actual_values[driver.driver_id] for driver in control.drivers
        ]
        quality_frames.append({
            "frame_index": frame.frame_index,
            "lambda": float(frame.lambda_value),
            "is_grid_point": bool(frame.is_grid_point),
            "max_abs_deviation": max(
                abs(a - n) for a, n in zip(actual, nominal)
            ),
        })

    cost = _branch_cost(
        transitions=transitions,
        open_transition=open_transition,
        request_count0=request_count0,
        request_count=request_count,
    )
    wall_elapsed_after = wall_elapsed0 + branch_wall
    budget_after = {
        "max_wall_seconds": max_wall,
        "max_host_requests": max_requests,
        "wall_elapsed_seconds": wall_elapsed_after,
        "request_count": request_count,
        "remaining_wall_seconds": max(0.0, max_wall - wall_elapsed_after),
        "remaining_host_requests": max(0, max_requests - request_count),
        "inherited_from_snapshot": True,
        "snapshot_wall_elapsed_seconds": wall_elapsed0,
        "snapshot_request_count": request_count0,
    }

    attempt_refs: list[str] = []
    for record in [*transitions, *( [open_transition] if open_transition else [])]:
        attempt_refs.extend(str(ref) for ref in record.get("child_attempt_refs", ()))
    artifacts = [REPLAY_BRANCH_DOCUMENT_NAME]
    artifacts.extend(
        str(Path(ref).relative_to(save_root))
        for ref in attempt_refs
        if Path(ref).is_absolute() and str(Path(ref)).startswith(str(save_root))
    )

    branch_doc: dict[str, Any] = {
        "schema_version": REPLAY_BRANCH_SCHEMA,
        "branch": {
            "branch_label": branch_label,
            "branch_id": f"{source_episode_id}:{branch_label}",
            "source_episode_id": source_episode_id,
            "source_round_id": episode.get("round_id"),
            "source_reaction_id": episode.get("reaction_id"),
            "source_decision_index": int(decision_index),
            "source_branch": episode.get("branch"),
            "source_snapshot_sha256": snapshot_content_sha256(dict(snapshot)),
        },
        "environment": {
            "environment_id": environment["environment_id"],
            "environment_hash": environment["environment_hash"],
            "reward_spec_sha256": str(control_identity["reward_spec_sha256"]),
            "code_identity": str(control_identity["code_identity"]),
        },
        "executed_candidate": {
            "residual_u": [float(v) for v in candidate_u_t],
            "source_label": candidate_source,
            "policy_id": REPLAY_INJECTED_POLICY_ID,
        },
        "outcome": {
            "status": status,
            "failure_code": failure_code,
            "failure_detail": failure_detail,
            "accepted": bool(result.accepted) if result is not None else False,
            "target_resolved": bool(result.target_resolved) if result is not None else False,
            "final_scale": float(result.scale) if result is not None else 1.0,
            "target_lambda": float(target_lambda),
            "last_accepted_lambda": float(last_accepted_lambda),
            "pending_head_lambda": float(pending[0][0]) if pending else None,
            "n_accepted_frames": len(frames),
            "n_refusals": cost["n_refusals"],
            "continuation_semantics": (
                "shared_macro_step_injected_candidate_no_policy_redecide"
            ),
        },
        "quality": {
            "frames": quality_frames,
            "max_abs_deviation": (
                max((entry["max_abs_deviation"] for entry in quality_frames),
                    default=None)
            ),
        },
        "cost": cost,
        "budget_after": budget_after,
        "accepted_frames": [frame.to_payload() for frame in frames],
        "rng": {
            "source_policy_rng": dict(rng_block),
            "note": (
                "the injected candidate makes the policy RNG unused for the "
                "executed decision; the continuation path performs no policy "
                "sampling"
            ),
        },
        "artifacts": artifacts,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }

    document_path = save_root / REPLAY_BRANCH_DOCUMENT_NAME
    document_path.write_text(
        json.dumps(branch_doc, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return branch_doc


__all__ = [
    "BRANCH_OUTCOME_ACCEPTED",
    "BRANCH_OUTCOME_BUDGET_STOP",
    "BRANCH_OUTCOME_REJECTED",
    "BRANCH_OUTCOME_SUBDIVIDED",
    "REPLAY_BRANCH_SCHEMA",
    "REPLAY_INJECTED_POLICY_ID",
    "ControlReplayError",
    "load_snapshot",
    "require_complete_history",
    "run_replay_branch",
]
