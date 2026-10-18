"""Shared G2 single-step execution seam + E_native control scan (ADR-0017).

This module is the ONE place where a scan point is executed (plan §5):

    nominal targets -> (optional bounded residual) -> key-atom seed
      -> one ACP ``XtbConstrainedOpt`` request -> normalized outcome
      -> residual + chemistry acceptance -> accept / typed refusal
      -> recovery (midpoint subdivision, original target stays queued)

``classic_scan.run_native_scan`` drives these primitives with a zero residual
(byte-compatible with the pre-extraction engine); ``run_control_scan`` drives
them with policy-chosen bounded residuals and records one
``pes2ts_control_transition_v2`` per decision macro-step.

The ONE logical-target macro step (decision execution + acceptance +
subdivision recovery + cost accounting + transition finalization) lives in
``_execute_logical_target`` — the shared seam that ``run_control_scan`` and
``control_replay.run_replay_branch`` (paired-branch diagnostics) both call,
so Classic/exploration/model/branch paths never duplicate optimizer logic.
``run_control_scan`` can also capture one
``pes2ts_control_state_snapshot_v1`` document per decision point
(``snapshot_dir`` / ``snapshot_sink``) BEFORE ``decide`` runs there, which
is what replay resumes from.

Discipline shared by both drivers:

* a rejected/unconverged structure NEVER becomes the parent frame;
* subdivision inserts a midpoint between the same accepted parent and the
  still-queued target; the chosen residual persists across the internal
  retries of its logical target (parent target identity continues);
* the acceptance residual gate compares the ACTUAL coordinate against the
  COMMANDED target (the real constraint residual); nominal progress is
  tracked separately and never masqueraded as achieved;
* E_native never performs an ORCA rescue (``rescue_used`` stays False);
  ``E_hybrid`` is a separate frozen environment and is not implemented here;
* per-attempt costs are aggregated once per decision — internal receipts are
  never re-billed as training steps.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from pes2ts_core.generation.planning.active_predictor import (
    SEED_AT_TARGET,
    SEED_OK,
    predict_seed,
)
from pes2ts_core.generation.planning.chemical_control import ChemicalControl
from pes2ts_core.generation.planning.control_snapshot import (
    build_snapshot_document,
    control_code_identity,
    control_document_digest,
    origin_state_sha256,
    policy_identity_from_decide,
    snapshot_content_sha256,
    validate_snapshot,
)
from pes2ts_core.integration.acp.native_constraint_results import (
    NativeConstraintOutcome,
    OUTCOME_COMPUTED,
    normalize_outcome,
)
from pes2ts_core.integration.acp.xtb_constraints_request import (
    XTC_REQUEST_SCHEMA,
    build_constraint_request,
    DriverTarget,
)
from pes2ts_core.integration.acp.native_constraint_results import (
    EVIDENCE_SCHEMA,
)
from pes2ts_core.pesn.contracts.control import (
    CONTROL_TRANSITION_SCHEMA_VERSION,
    ControlAction,
    ControlTransition,
    ENVIRONMENT_NATIVE,
    MacroCost,
    PolicyRewardSpec,
    ResidualControlSpec,
    TERMINAL_COMPLETE,
    TERMINAL_STEP_LIMIT,
    TRUNCATED_BUDGET,
)

CONTROL_RESULT_SCHEMA = "pes2ts_native_scan_result_v3"
CONTROL_TRACE_SCHEMA = "pes2ts_native_scan_trace_v3"
CONTROL_TRANSITIONS_SCHEMA = CONTROL_TRANSITION_SCHEMA_VERSION

STATUS_COMPLETE = "complete_interval"
STATUS_PARTIAL_PREFIX = "partial_prefix"
STATUS_REJECTED_INPUT = "rejected_input"
STATUS_BUDGET_EXHAUSTED = "budget_exhausted"

ENGINE_MODE_CONTROL = "control_residual_e_native"
CONTROL_ENGINE_VERSION = 1
CANDIDATE_GENERATOR_VERSION = "bounded_candidates_v1"
ACP_WORKFLOW_NAME = "XtbConstrainedOpt"

_COVALENT_RADII: dict[str, float] = {
    "H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39,
}
_BOND_TOLERANCE = 0.45
_COLLISION_DISTANCE = 0.8


class ControlScanError(ValueError):
    """Raised on control-scan misuse or contract breaks (not typed refusals)."""


# ---------------------------------------------------------------------------
# Shared frozen execution recipe + accepted frame (moved from classic_scan;
# classic_scan re-exports them so external importers stay untouched).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NativeScanPolicy:
    """Frozen execution recipe for one native scan (classic or control)."""

    n_grid_points: int = 21
    force_constant: float | None = 50.0
    opt_level: str = "normal"
    max_cycle: int | None = 500
    threads: int = 4
    point_timeout_seconds: int | None = 300
    physical_sp: bool = True
    max_seed_atom_displacement: float = 0.6
    distance_residual_tolerance: float = 0.05
    max_subdivisions: int = 2
    max_rejections_per_target: int = 4
    max_wall_seconds: float = 900.0
    max_host_requests: int = 120
    freeze_environment: bool = False

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


TransportCallable = Callable[[dict[str, Any], Path, str, str], Any]


@dataclass
class ScanFrame:
    """One accepted scan point (v1-compatible payload)."""

    frame_index: int
    lambda_value: float
    is_grid_point: bool
    requested_targets: dict[str, float]
    actual_values: dict[str, float]
    residuals: dict[str, float]
    geometry: list[list[float]]
    energy_effective_hartree: float | None
    energy_physical_hartree: float | None
    parent_frame_index: int | None
    attempt_dir: str
    request_sha256: str
    n_optimization_cycles: int | None
    max_seed_displacement: float | None

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def bonded_pairs(elements: Sequence[str], geometry: np.ndarray) -> set[tuple[int, int]]:
    n = len(elements)
    pairs: set[tuple[int, int]] = set()
    for i in range(n):
        for j in range(i + 1, n):
            ri = _COVALENT_RADII.get(elements[i])
            rj = _COVALENT_RADII.get(elements[j])
            if ri is None or rj is None:
                continue
            if float(np.linalg.norm(geometry[i] - geometry[j])) <= ri + rj + _BOND_TOLERANCE:
                pairs.add((i, j))
    return pairs


def chemistry_check(
    elements: Sequence[str],
    geometry: np.ndarray,
    common_bond_checks: Sequence[Any],
    endpoint_union_pairs: frozenset[tuple[int, int]],
    driver_pairs: set[tuple[int, int]],
    parent_pairs: set[tuple[int, int]],
    local_support_indices: frozenset[int] = frozenset(),
) -> tuple[bool, str | None]:
    """Post-optimization chemistry sanity with role-separated criteria.

    * collision: any non-driver pair closer than 0.8 A;
    * spectator guards: a bond common to both endpoints must stay inside its
      plan-frozen distance window (F/B pairs are exempt);
    * unsolicited bond formation: newly bonded, unknown to both endpoint
      graphs, and outside the reaction-center local support set.
    """
    n = len(elements)
    for i in range(n):
        for j in range(i + 1, n):
            distance = float(np.linalg.norm(geometry[i] - geometry[j]))
            if distance < _COLLISION_DISTANCE and (i, j) not in driver_pairs:
                return False, f"collision: atoms {i},{j} at {distance:.3f} Å"
    current = bonded_pairs(elements, geometry)
    for check in common_bond_checks:
        pair = tuple(sorted(check.atoms))
        if pair in driver_pairs:
            continue
        distance = float(np.linalg.norm(geometry[pair[0]] - geometry[pair[1]]))
        if not check.minimum_distance <= distance <= check.maximum_distance:
            return False, (
                f"spectator bond {pair[0]},{pair[1]} at {distance:.3f} Å outside its "
                f"window [{check.minimum_distance:.3f}, {check.maximum_distance:.3f}]"
            )
    formed_unknown = {
        pair for pair in current - parent_pairs
        if pair not in driver_pairs
        and pair not in endpoint_union_pairs
        and not (pair[0] in local_support_indices and pair[1] in local_support_indices)
    }
    if formed_unknown:
        first = sorted(formed_unknown)[0]
        return False, (
            f"unsolicited bond formed outside both endpoint graphs and the "
            f"reaction-center support set: atoms {first[0]},{first[1]}"
        )
    return True, None


# ---------------------------------------------------------------------------
# Single-step primitives (shared by classic and control drivers)
# ---------------------------------------------------------------------------


def classify_step_refusal(
    outcome: NativeConstraintOutcome,
    control: ChemicalControl,
    policy: NativeScanPolicy,
    driver_pairs: set[tuple[int, int]],
    parent_pairs: set[tuple[int, int]],
) -> str | None:
    """Return a refusal reason, or None when the point is acceptable.

    The residual gate compares the actual coordinate with the COMMANDED
    constraint target (the real constraint residual reported by the engine);
    nominal schedule progress is judged separately by the drivers.
    """
    if outcome.status != OUTCOME_COMPUTED:
        return f"{outcome.status}: {outcome.failure_reason}"
    residual_bad = [
        measurement for measurement in outcome.measurements
        if measurement.residual is None
        or abs(measurement.residual) > policy.distance_residual_tolerance
    ]
    if residual_bad:
        worst = max(residual_bad, key=lambda m: abs(m.residual or 0.0))
        return (
            f"residual {worst.driver_id} {worst.residual:+.4f} exceeds "
            f"{policy.distance_residual_tolerance:g}"
        )
    final_geometry = np.asarray(outcome.geometry, dtype=float)
    chemistry_ok, chemistry_reason = chemistry_check(
        control.elements, final_geometry,
        control.common_bond_checks, control.endpoint_union_pairs,
        driver_pairs, parent_pairs,
        local_support_indices=control.local_support_indices,
    )
    if not chemistry_ok:
        return f"chemistry: {chemistry_reason}"
    return None


def scaled_step_targets(
    control: ChemicalControl,
    last_accepted_lambda: float,
    target_lambda: float,
    scale: float,
) -> tuple[float, tuple[float, ...]]:
    """Nominal lambda + targets for one attempt at the given subdivision."""
    span = target_lambda - last_accepted_lambda
    scaled_lambda = last_accepted_lambda + span * scale
    return scaled_lambda, control.targets_at(scaled_lambda)


@dataclass
class StepPreparation:
    """Everything one attempt produced (seed refusal => no request)."""

    seed: Any
    request: dict[str, Any] | None
    attempt_id: str
    attempt_dir: Path
    outcome: NativeConstraintOutcome | None
    transport_used: bool


def _lambda_tag(lam: float) -> str:
    return f"{lam:.6f}".rstrip("0").rstrip(".")


def prepare_step_attempt(
    *,
    control: ChemicalControl,
    policy: NativeScanPolicy,
    parent_geometry: np.ndarray,
    scaled_lambda: float,
    command_targets: Sequence[float],
    transport: TransportCallable,
    attempts_root: Path,
    reaction_id: str,
    rejections: int,
    parent_request_sha256: str | None,
) -> StepPreparation:
    """Seed -> request -> transport -> normalized outcome for ONE attempt."""
    driver_kinds = [driver.kind for driver in control.drivers]
    driver_atoms = [list(driver.atoms) for driver in control.drivers]
    driver_ids = [driver.driver_id for driver in control.drivers]
    n_atoms = len(control.elements)

    seed = predict_seed(
        [tuple(float(v) for v in row) for row in parent_geometry],
        driver_kinds=driver_kinds,
        driver_atoms=driver_atoms,
        target_values=tuple(float(v) for v in command_targets),
        active_atom_indices=control.active_atom_indices,
        max_atom_displacement=policy.max_seed_atom_displacement,
    )
    attempt_id = f"lam{_lambda_tag(scaled_lambda)}-try{rejections:02d}"
    attempt_dir = attempts_root / f"lam{_lambda_tag(scaled_lambda)}" / attempt_id

    if seed.status not in (SEED_OK, SEED_AT_TARGET):
        return StepPreparation(
            seed=seed, request=None, attempt_id=attempt_id,
            attempt_dir=attempt_dir, outcome=None, transport_used=False,
        )

    seed_geometry = (
        seed.geometry
        if seed.geometry is not None
        else [tuple(float(v) for v in row) for row in parent_geometry]
    )
    targets_payload = [
        DriverTarget(
            driver_id=driver.driver_id,
            kind=driver.kind,
            atoms=driver.atoms,
            target=float(value),
            unit=driver.unit,
        )
        for driver, value in zip(control.drivers, command_targets)
    ]
    fix_atoms = (
        [i for i in range(n_atoms) if i not in control.active_atom_indices]
        if policy.freeze_environment
        else None
    )
    request = build_constraint_request(
        geometry=seed_geometry,
        elements=control.elements,
        charge=control.charge,
        multiplicity=control.multiplicity,
        targets=targets_payload,
        reaction_id=reaction_id,
        point_id=f"lam{_lambda_tag(scaled_lambda)}",
        force_constant=policy.force_constant,
        opt_level=policy.opt_level,
        max_cycle=policy.max_cycle,
        fix_atoms=fix_atoms,
        physical_sp=policy.physical_sp,
        threads=policy.threads,
        timeout_seconds=policy.point_timeout_seconds,
        plan_sha256=control.plan_sha256,
        parent_frame_sha256=parent_request_sha256,
    )
    attempt = transport(request, attempt_dir, f"{reaction_id}-native", attempt_id)
    outcome = normalize_outcome(attempt_result=attempt, request=request, driver_ids=driver_ids)
    return StepPreparation(
        seed=seed, request=request, attempt_id=attempt_id, attempt_dir=attempt_dir,
        outcome=outcome, transport_used=True,
    )


# ---------------------------------------------------------------------------
# Environment freeze (E_native)
# ---------------------------------------------------------------------------


def freeze_control_environment(
    *,
    policy: NativeScanPolicy,
    reward_spec: PolicyRewardSpec,
    environment_id: str = ENVIRONMENT_NATIVE,
) -> dict[str, Any]:
    """Freeze and hash the E_native environment identity for one round."""
    if environment_id != ENVIRONMENT_NATIVE:
        raise ControlScanError(
            "only E_native is implemented; E_hybrid is gated on the AB4 rescue gate"
        )
    payload = {
        "engine_mode": ENGINE_MODE_CONTROL,
        "engine_version": CONTROL_ENGINE_VERSION,
        "environment_id": environment_id,
        "scan_policy": policy.to_payload(),
        "reward_spec": reward_spec.to_payload(),
        "request_schema": XTC_REQUEST_SCHEMA,
        "evidence_schema": EVIDENCE_SCHEMA,
        "acp_workflow": ACP_WORKFLOW_NAME,
        "candidate_generator_version": CANDIDATE_GENERATOR_VERSION,
        "rescue": "disabled_in_E_native",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        "environment_id": environment_id,
        "environment_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "freeze_payload": payload,
    }


# ---------------------------------------------------------------------------
# Decision-time state handed to the policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ControlScanState:
    """Read-only decision-time observation (no truth, no future)."""

    control: ChemicalControl
    spec: ResidualControlSpec
    parent_geometry: np.ndarray
    last_accepted_lambda: float
    target_lambda: float
    remaining_wall_fraction: float
    remaining_request_fraction: float
    recent_rejections: int
    candidate_already_awarded: bool
    decision_index: int


@dataclass(frozen=True)
class ControlDecision:
    """The policy's answer for one decision: candidates + explicit choice."""

    candidates: tuple[tuple[float, ...], ...]
    candidate_sources: tuple[str, ...]
    executed_candidate_index: int
    policy_id: str
    policy_hash: str


DecisionFn = Callable[[ControlScanState], ControlDecision]


# ---------------------------------------------------------------------------
# E_native control scan (macro-step environment)
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8"
    )


def _is_zero_residual(u: Sequence[float], tolerance: float = 1e-9) -> bool:
    return all(abs(float(v)) <= tolerance for v in u)


def _episode_status_for(failure_code: str | None, status: str) -> str:
    if status == STATUS_COMPLETE:
        return TERMINAL_COMPLETE
    if status == STATUS_BUDGET_EXHAUSTED:
        return TRUNCATED_BUDGET
    if status == STATUS_PARTIAL_PREFIX:
        return TERMINAL_STEP_LIMIT
    return "ongoing"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write JSON (temp file + ``os.replace``), sort_keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".part-{os.getpid()}")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _check_scan_budget(
    *,
    elapsed_total: float,
    request_count: int,
    policy: NativeScanPolicy,
    n_accepted: int,
) -> tuple[bool, str | None, str | None, str | None]:
    """Shared wall/request budget gate. Returns (exhausted, status, code, detail)."""
    if elapsed_total > policy.max_wall_seconds:
        return (
            True,
            STATUS_BUDGET_EXHAUSTED,
            "WALL_BUDGET_EXHAUSTED",
            f"{elapsed_total:.1f}s exceeded {policy.max_wall_seconds:g}s after "
            f"{n_accepted} accepted frames",
        )
    if request_count >= policy.max_host_requests:
        return (
            True,
            STATUS_BUDGET_EXHAUSTED,
            "REQUEST_BUDGET_EXHAUSTED",
            f"{request_count} host requests reached the cap "
            f"{policy.max_host_requests}",
        )
    return False, None, None, None


@dataclass(frozen=True)
class FreshDecision:
    """Policy decision bundle for a NOT-yet-open logical target.

    Produced by ``decide`` (main scan) or synthesized from an injected
    candidate (replay branch); the shared macro-step seam builds the
    transition record from it.
    """

    candidates: tuple[tuple[float, ...], ...]
    candidate_sources: tuple[str, ...]
    executed_candidate_index: int
    policy_id: str
    policy_hash: str
    decision_context_vector: tuple[float, ...]


@dataclass
class LogicalTargetResult:
    """Outcome of ONE logical-target macro step (shared execution seam)."""

    accepted: bool
    stop_scan: bool
    scale: float
    target_resolved: bool
    status: str | None
    failure_code: str | None
    failure_detail: str | None
    request_count: int
    last_accepted_lambda: float
    parent_geometry: np.ndarray
    parent_bonds: set[tuple[int, int]]
    parent_index: int
    pending_popped: bool
    pending_entry: tuple[float, bool, int] | None
    new_frame_indices: tuple[int, ...]
    open_transition: dict[str, Any] | None


def _execute_logical_target(
    *,
    control: ChemicalControl,
    spec: ResidualControlSpec,
    active_policy: NativeScanPolicy,
    transport: TransportCallable,
    attempts_root: Path,
    reaction_id: str,
    driver_pairs: set[tuple[int, int]],
    frames: list[ScanFrame],
    transitions: list[dict[str, Any]],
    decision_meta: list[dict[str, Any]],
    events: list[dict[str, Any]],
    target_lambda: float,
    is_grid_point: bool,
    rejections: int,
    scale: float,
    chosen_u: tuple[float, ...],
    chosen_source: str,
    identity_prefix: str,
    decision_index: int,
    fresh_decision: FreshDecision | None,
    open_transition: dict[str, Any] | None,
    parent_geometry: np.ndarray,
    parent_bonds: set[tuple[int, int]],
    parent_index: int,
    last_accepted_lambda: float,
    request_count: int,
    parent_request_sha256: str | None,
) -> LogicalTargetResult:
    """Execute + accept/recover ONE logical target (the shared macro step).

    Shared by :func:`run_control_scan` (policy-chosen residual) and
    ``control_replay.run_replay_branch`` (injected candidate): the
    constrained-opt request, refusal classification, midpoint subdivision
    recovery, per-decision cost accounting, and transition finalization
    live ONLY here — no caller duplicates optimizer logic.

    ``fresh_decision`` builds the open transition record (a NOT-yet-open
    logical target); ``open_transition`` continues an existing one
    (midpoint-acceptance retry — the residual is NOT re-chosen).
    """
    def record(event: dict[str, Any]) -> None:
        event["ts"] = _now()
        events.append(event)

    min_scale = 0.5 ** active_policy.max_subdivisions
    transition = open_transition
    if transition is None:
        if fresh_decision is None:
            raise ControlScanError(
                "logical-target execution needs either an open transition "
                "or a fresh decision"
            )
        base_targets = control.targets_at(target_lambda)
        action = ControlAction(
            schema_version="pes2ts_control_action_v2",
            action_id=f"{identity_prefix}-d{decision_index:03d}",
            driver_ids=spec.driver_ids,
            base_targets=tuple(float(v) for v in base_targets),
            residual_u=chosen_u,
            epsilon=spec.epsilon,
            command_targets=spec.command_targets(base_targets, chosen_u),
            source=chosen_source,
            policy_id=fresh_decision.policy_id,
        )
        transition = {
            "transition_id": f"{identity_prefix}-d{decision_index:03d}",
            "decision_index": decision_index,
            "decision_start_frame_index": parent_index,
            "decision_context_vector": fresh_decision.decision_context_vector,
            "action": action,
            "candidates": fresh_decision.candidates,
            "candidate_sources": fresh_decision.candidate_sources,
            "executed_candidate_index": fresh_decision.executed_candidate_index,
            "child_attempt_refs": [],
            "macro_wall": 0.0,
            "macro_requests": 0,
            "macro_cycles": 0,
            "refusal_reasons": [],
            "accepted": False,
            "accepted_frame_index": None,
            "pending_target": None,
            "policy_hash": str(fresh_decision.policy_hash),
        }
        record({
            "kind": "decision", "decision_index": decision_index,
            "target_lambda": target_lambda,
            "residual_u": list(chosen_u),
            "source": chosen_source,
            "n_candidates": len(fresh_decision.candidates),
            "policy_id": fresh_decision.policy_id,
            "policy_hash": str(fresh_decision.policy_hash),
        })

    accepted = False
    stop_scan = False
    status: str | None = None
    failure_code: str | None = None
    failure_detail: str | None = None
    new_frames: list[ScanFrame] = []

    while not accepted:
        scaled_lambda, nominal_targets = scaled_step_targets(
            control, last_accepted_lambda, target_lambda, scale
        )
        command_targets = _command_with_residual(
            spec, nominal_targets, chosen_u
        )
        if command_targets is None:
            # The residual is illegal against the midpoint nominal (near
            # a frozen range edge): fall back to the nominal midpoint.
            record({
                "kind": "residual_dropped_at_midpoint",
                "lambda": scaled_lambda, "decision_index": transition["decision_index"],
            })
            command_targets = nominal_targets
        prep = prepare_step_attempt(
            control=control,
            policy=active_policy,
            parent_geometry=parent_geometry,
            scaled_lambda=scaled_lambda,
            command_targets=command_targets,
            transport=transport,
            attempts_root=attempts_root,
            reaction_id=reaction_id,
            rejections=rejections,
            parent_request_sha256=parent_request_sha256,
        )
        if prep.seed.status not in (SEED_OK, SEED_AT_TARGET):
            if scale > min_scale:
                scale *= 0.5
                record({"kind": "subdivision_after_seed_refusal",
                        "lambda": scaled_lambda, "seed_status": prep.seed.status,
                        "new_scale": scale})
                continue
            status = STATUS_PARTIAL_PREFIX
            failure_code = f"STEP_LIMIT:SEED_{prep.seed.status.upper()}"
            failure_detail = prep.seed.reason
            record({"kind": "seed_refusal_terminal", "lambda": scaled_lambda,
                    "status": prep.seed.status, "reason": prep.seed.reason})
            stop_scan = True
            break

        request_count += 1
        transition["macro_requests"] += 1
        transition["child_attempt_refs"].append(str(prep.attempt_dir))
        assert prep.outcome is not None and prep.request is not None
        outcome = prep.outcome
        attempt_wall = float(outcome.wall_seconds or 0.0)
        transition["macro_wall"] += attempt_wall
        transition["macro_cycles"] += int(outcome.n_optimization_cycles_logged or 0)
        record({
            "kind": "point_attempt", "lambda": scaled_lambda,
            "is_grid_point": is_grid_point, "attempt_id": prep.attempt_id,
            "attempt_dir": str(prep.attempt_dir),
            "request_sha256": prep.request["provenance"]["request_sha256"],
            "outcome": outcome.status, "converged": outcome.converged_geometry,
            "wall_seconds": outcome.wall_seconds,
            "n_optimization_cycles_logged": outcome.n_optimization_cycles_logged,
            "seed_status": prep.seed.status,
            "max_seed_displacement": prep.seed.max_atom_displacement,
            "failure_reason": outcome.failure_reason,
            "decision_index": transition["decision_index"],
        })

        refusal_reason = classify_step_refusal(
            outcome, control, active_policy, driver_pairs, parent_bonds,
        )
        if refusal_reason is None:
            final_geometry = np.asarray(outcome.geometry, dtype=float)
            frame = ScanFrame(
                frame_index=parent_index + 1,
                lambda_value=float(scaled_lambda),
                is_grid_point=bool(
                    is_grid_point and abs(scaled_lambda - target_lambda) <= 1e-9
                ),
                requested_targets={
                    driver.driver_id: float(value)
                    for driver, value in zip(control.drivers, command_targets)
                },
                actual_values={
                    m.driver_id: float(m.actual) for m in outcome.measurements
                },
                residuals={m.driver_id: float(m.residual) for m in outcome.measurements},
                geometry=[[float(v) for v in row] for row in final_geometry],
                energy_effective_hartree=outcome.energy_effective_hartree,
                energy_physical_hartree=outcome.energy_physical_hartree,
                parent_frame_index=parent_index,
                attempt_dir=str(prep.attempt_dir),
                request_sha256=str(prep.request["provenance"]["request_sha256"]),
                n_optimization_cycles=outcome.n_optimization_cycles_logged,
                max_seed_displacement=prep.seed.max_atom_displacement,
            )
            frames.append(frame)
            new_frames.append(frame)
            parent_geometry = final_geometry
            parent_bonds = bonded_pairs(control.elements, final_geometry)
            parent_index = frame.frame_index
            last_accepted_lambda = float(scaled_lambda)
            accepted = True
            transition["accepted"] = True
            transition["accepted_frame_index"] = frame.frame_index
            record({
                "kind": "accepted", "lambda": float(scaled_lambda),
                "frame_index": frame.frame_index,
                "is_grid_point": frame.is_grid_point,
                "energy_effective_hartree": frame.energy_effective_hartree,
                "energy_physical_hartree": frame.energy_physical_hartree,
            })
            continue

        record({"kind": "refusal", "lambda": scaled_lambda,
                "attempt_id": prep.attempt_id, "reason": refusal_reason})
        transition["refusal_reasons"].append(refusal_reason)
        rejections += 1
        if rejections > active_policy.max_rejections_per_target:
            status = STATUS_PARTIAL_PREFIX
            failure_code = "STEP_LIMIT:REJECTIONS_EXHAUSTED"
            failure_detail = refusal_reason
            stop_scan = True
            break
        if scale > min_scale:
            scale *= 0.5
            record({"kind": "subdivision_after_refusal", "lambda": scaled_lambda,
                    "new_scale": scale})
            continue
        status = STATUS_PARTIAL_PREFIX
        failure_code = "STEP_LIMIT:SUBDIVISIONS_EXHAUSTED"
        failure_detail = refusal_reason
        stop_scan = True
        break

    target_resolved = accepted and scale >= 1.0 - 1e-12
    open_transition_out: dict[str, Any] | None = None
    if stop_scan or target_resolved:
        transition["pending_target"] = (
            float(target_lambda) if stop_scan else None
        )
        transition["final_scale"] = scale
        transition["episode_ended_here"] = bool(stop_scan)
        transition["state_after_geometry"] = [
            [float(x) for x in row] for row in parent_geometry
        ]
        transition["state_after_frame_index"] = parent_index
        transitions.append(transition)
        decision_meta.append({
            "decision_index": transition["decision_index"],
            "target_lambda": float(target_lambda),
            "source": chosen_source,
            "accepted": bool(accepted),
            "macro_wall": transition["macro_wall"],
            "macro_requests": transition["macro_requests"],
        })
    else:
        open_transition_out = transition

    pending_popped = bool(accepted and target_resolved)
    pending_entry: tuple[float, bool, int] | None = None
    if accepted and not target_resolved:
        pending_entry = (float(target_lambda), bool(is_grid_point), int(rejections))

    return LogicalTargetResult(
        accepted=accepted,
        stop_scan=stop_scan,
        scale=scale,
        target_resolved=target_resolved,
        status=status,
        failure_code=failure_code,
        failure_detail=failure_detail,
        request_count=request_count,
        last_accepted_lambda=last_accepted_lambda,
        parent_geometry=parent_geometry,
        parent_bonds=parent_bonds,
        parent_index=parent_index,
        pending_popped=pending_popped,
        pending_entry=pending_entry,
        new_frame_indices=tuple(frame.frame_index for frame in new_frames),
        open_transition=open_transition_out,
    )


def run_control_scan(
    *,
    control: ChemicalControl,
    origin_geometry: Sequence[Sequence[float]],
    transport: TransportCallable,
    save_dir: str | Path,
    spec: ResidualControlSpec,
    reward_spec: PolicyRewardSpec,
    decide: DecisionFn,
    policy: NativeScanPolicy | None = None,
    reaction_id: str | None = None,
    episode_tag: str | None = None,
    round_id: str | None = None,
    snapshot_sink: Callable[[Mapping[str, Any]], None] | None = None,
    snapshot_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run one E_native episode with policy-chosen bounded residuals.

    Returns a ``pes2ts_native_scan_result_v3`` document carrying the event
    trace under ``_events`` and the transition records under
    ``_transitions`` for :func:`persist_control_outputs`.

    ``episode_tag`` makes the transition/action ids episode-unique
    (``<reaction>-<tag>-d<idx>``); without it the legacy id format is kept
    for backward-compatible stores.

    ``snapshot_dir`` / ``snapshot_sink`` capture one
    ``pes2ts_control_state_snapshot_v1`` document per decision point,
    written BEFORE ``decide`` is invoked there (``snapshot_dir`` gets
    ``d{decision_index:03d}.json`` atomically; ``snapshot_sink`` receives
    the same document).  When neither is passed the scan behaves exactly
    as before.
    """
    active_policy = policy or NativeScanPolicy()
    reaction_id = reaction_id or control.reaction_id
    identity_prefix = (
        f"{reaction_id}-{episode_tag}" if episode_tag else reaction_id
    )
    episode_id = f"{reaction_id}:{episode_tag}" if episode_tag else reaction_id
    n_atoms = len(control.elements)
    if len(origin_geometry) != n_atoms:
        raise ControlScanError("origin geometry does not match the control atom order")
    if tuple(driver.driver_id for driver in control.drivers) != spec.driver_ids:
        raise ControlScanError("residual spec drivers do not match the control order")

    environment = freeze_control_environment(
        policy=active_policy, reward_spec=reward_spec
    )

    save_root = Path(save_dir)
    attempts_root = save_root / "attempts"
    (save_root / "control").mkdir(parents=True, exist_ok=True)

    snapshot_root = Path(snapshot_dir) if snapshot_dir is not None else None
    if snapshot_root is not None:
        snapshot_root.mkdir(parents=True, exist_ok=True)

    grid = list(control.lambda_grid)
    frames: list[ScanFrame] = []
    events: list[dict[str, Any]] = []
    request_count = 0
    wall_start = time.monotonic()
    decision_meta: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []

    def record(event: dict[str, Any]) -> None:
        event["ts"] = _now()
        events.append(event)

    origin_array = np.asarray(origin_geometry, dtype=float)
    if not np.isfinite(origin_array).all():
        return _control_terminal_result(
            control=control,
            reaction_id=reaction_id,
            scan_policy=active_policy,
            reward_spec=reward_spec,
            environment=environment,
            frames=frames,
            events=events,
            transitions=transitions,
            decision_meta=decision_meta,
            request_count=request_count,
            wall_seconds=0.0,
            status=STATUS_REJECTED_INPUT,
            failure_code="INPUT_GEOMETRY_INVALID",
            failure_detail="origin geometry contains non-finite coordinates",
        )

    driver_pairs = {tuple(sorted(pair)) for pair in
                    (list(driver.atoms) for driver in control.drivers) if len(pair) == 2}

    origin_achieved = {
        driver.driver_id: driver.target_at(0.0) for driver in control.drivers
    }
    frames.append(
        ScanFrame(
            frame_index=0, lambda_value=0.0, is_grid_point=True,
            requested_targets=dict(origin_achieved),
            actual_values=dict(origin_achieved),
            residuals={key: 0.0 for key in origin_achieved},
            geometry=[[float(v) for v in row] for row in origin_array],
            energy_effective_hartree=None, energy_physical_hartree=None,
            parent_frame_index=None, attempt_dir="", request_sha256="",
            n_optimization_cycles=None, max_seed_displacement=None,
        )
    )
    record({"kind": "origin_frame", "lambda": 0.0, "frame_index": 0})

    parent_geometry = origin_array.copy()
    parent_bonds = bonded_pairs(control.elements, origin_array)
    parent_index = 0
    last_accepted_lambda = 0.0

    pending: deque[tuple[float, bool, int]] = deque(
        (lam, True, 0) for lam in grid[1:]
    )
    status = STATUS_COMPLETE
    failure_code: str | None = None
    failure_detail: str | None = None
    min_scale = 0.5 ** active_policy.max_subdivisions
    decision_index = 0
    transition: dict[str, Any] | None = None
    open_target: float | None = None

    snapshot_prev_sha: str | None = None
    origin_sha = origin_state_sha256(
        reaction_id=reaction_id, episode_id=episode_id, origin_geometry=origin_geometry
    )
    policy_probe = policy_identity_from_decide(decide)
    environment_hash = str(environment["environment_hash"])
    reward_sha = str(reward_spec.content_hash())
    control_digest = control_document_digest(control)
    spec_digest = str(spec.content_hash())
    code_identity = control_code_identity()

    while pending:
        elapsed = time.monotonic() - wall_start
        exhausted, budget_status, budget_code, budget_detail = _check_scan_budget(
            elapsed_total=elapsed,
            request_count=request_count,
            policy=active_policy,
            n_accepted=len(frames) - 1,
        )
        if exhausted:
            status, failure_code = budget_status, budget_code
            failure_detail = budget_detail
            break

        target_lambda, is_grid_point, rejections = pending[0]

        # ---- decision point: ONE policy decision per LOGICAL target ----
        # A midpoint acceptance keeps the same target queued; the retry
        # CONTINUES the open decision (parent target identity persists
        # across internal subdivisions; the residual is not re-chosen).
        continuing = (
            transition is not None and target_lambda == open_target
        )
        fresh: FreshDecision | None = None
        this_decision_index = (
            int(transition["decision_index"]) if continuing else decision_index
        )
        if not continuing:
            base_targets = control.targets_at(target_lambda)
            state = ControlScanState(
                control=control,
                spec=spec,
                parent_geometry=parent_geometry,
                last_accepted_lambda=last_accepted_lambda,
                target_lambda=target_lambda,
                remaining_wall_fraction=max(
                    0.0, 1.0 - elapsed / active_policy.max_wall_seconds
                ),
                remaining_request_fraction=max(
                    0.0, 1.0 - request_count / active_policy.max_host_requests
                ),
                recent_rejections=0,
                candidate_already_awarded=(
                    _first_bracketed_peak_decision(transitions, frames) is not None
                ),
                decision_index=decision_index,
            )
            if snapshot_root is not None or snapshot_sink is not None:
                elapsed_snapshot = time.monotonic() - wall_start
                snapshot_doc = build_snapshot_document(
                    round_id=round_id,
                    episode_id=episode_id,
                    reaction_id=reaction_id,
                    branch=None,
                    decision_index=decision_index,
                    control_digest=control_digest,
                    spec_digest=spec_digest,
                    policy_id=policy_probe["policy_id"],
                    policy_hash=policy_probe["policy_hash"],
                    environment_hash=environment_hash,
                    reward_spec_sha256=reward_sha,
                    code_identity=code_identity,
                    parent_geometry=parent_geometry,
                    parent_bonds=parent_bonds,
                    parent_index=parent_index,
                    parent_request_sha256=frames[-1].request_sha256 or None,
                    last_accepted_lambda=last_accepted_lambda,
                    target_lambda=target_lambda,
                    pending=pending,
                    scale=1.0,
                    open_target=None,
                    open_transition=None,
                    max_wall_seconds=active_policy.max_wall_seconds,
                    max_host_requests=active_policy.max_host_requests,
                    wall_elapsed_seconds=elapsed_snapshot,
                    request_count=request_count,
                    accepted_prefix_frame_indices=[
                        frame.frame_index for frame in frames
                    ],
                    recent_rejections=0,
                    candidate_already_awarded=(
                        _first_bracketed_peak_decision(transitions, frames) is not None
                    ),
                    subdivision={
                        "scale": 1.0,
                        "min_scale": min_scale,
                        "max_subdivisions": active_policy.max_subdivisions,
                        "max_rejections_per_target": (
                            active_policy.max_rejections_per_target
                        ),
                    },
                    open_refusal_reasons=(
                        list(transition["refusal_reasons"]) if transition else None
                    ),
                    rng=policy_probe["rng"],
                    parent_state_sha256=(
                        snapshot_prev_sha if snapshot_prev_sha is not None else origin_sha
                    ),
                )
                if snapshot_root is not None:
                    _write_json_atomic(
                        snapshot_root / f"d{decision_index:03d}.json", snapshot_doc
                    )
                if snapshot_sink is not None:
                    snapshot_sink(snapshot_doc)
                snapshot_prev_sha = snapshot_content_sha256(snapshot_doc)
            decision = decide(state)
            _validate_decision(decision, spec, base_targets)
            chosen_u = tuple(
                float(v) for v in decision.candidates[decision.executed_candidate_index]
            )
            chosen_source = decision.candidate_sources[decision.executed_candidate_index]
            fresh = FreshDecision(
                candidates=tuple(
                    tuple(float(v) for v in u) for u in decision.candidates
                ),
                candidate_sources=tuple(decision.candidate_sources),
                executed_candidate_index=int(decision.executed_candidate_index),
                policy_id=str(decision.policy_id),
                policy_hash=str(decision.policy_hash),
                decision_context_vector=_decision_context_vector(state),
            )
            decision_index += 1
            open_target = target_lambda
        else:
            chosen_u = tuple(
                float(v) for v in transition["action"].residual_u
            )
            chosen_source = str(transition["action"].source)

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
            is_grid_point=is_grid_point,
            rejections=rejections,
            scale=1.0,
            chosen_u=chosen_u,
            chosen_source=chosen_source,
            identity_prefix=identity_prefix,
            decision_index=this_decision_index,
            fresh_decision=fresh,
            open_transition=None if fresh is not None else transition,
            parent_geometry=parent_geometry,
            parent_bonds=parent_bonds,
            parent_index=parent_index,
            last_accepted_lambda=last_accepted_lambda,
            request_count=request_count,
            parent_request_sha256=frames[-1].request_sha256 or None,
        )
        parent_geometry = result.parent_geometry
        parent_bonds = result.parent_bonds
        parent_index = result.parent_index
        last_accepted_lambda = result.last_accepted_lambda
        request_count = result.request_count
        if result.stop_scan or result.target_resolved:
            transition = None
            open_target = None
        else:
            transition = result.open_transition
        if result.stop_scan:
            status = result.status or status
            failure_code = result.failure_code
            failure_detail = result.failure_detail
            break
        if result.pending_popped:
            pending.popleft()
        elif result.pending_entry is not None:
            pending[0] = result.pending_entry

    covered = {round(frame.lambda_value, 9) for frame in frames}
    grid_covered = all(round(lam, 9) in covered for lam in grid)
    completed_interval = status == STATUS_COMPLETE and grid_covered and not pending
    if completed_interval:
        failure_code = None
        failure_detail = None
    elif status == STATUS_COMPLETE:
        status = STATUS_PARTIAL_PREFIX
        if failure_code is None:
            failure_code = "STEP_LIMIT:GRID_INCOMPLETE"

    wall_seconds = time.monotonic() - wall_start
    result = _control_terminal_result(
        control=control,
        reaction_id=reaction_id,
        scan_policy=active_policy,
        reward_spec=reward_spec,
        environment=environment,
        frames=frames,
        events=events,
        transitions=transitions,
        decision_meta=decision_meta,
        request_count=request_count,
        wall_seconds=wall_seconds,
        status=status,
        failure_code=failure_code,
        failure_detail=failure_detail,
        completed_interval=completed_interval,
    )
    return result


def _decision_context_vector(state: ControlScanState) -> tuple[float, ...]:
    from pes2ts_core.pesn.contracts.control import ControlContext

    context = ControlContext(
        nominal_lambda=float(state.last_accepted_lambda),
        next_target_lambda=float(state.target_lambda),
        remaining_wall_fraction=float(state.remaining_wall_fraction),
        remaining_request_fraction=float(state.remaining_request_fraction),
        recent_rejections=int(state.recent_rejections),
        rescue_quota_remaining=0,
        candidate_already_awarded=bool(state.candidate_already_awarded),
    )
    context.validate()
    return context.to_vector()


def _first_bracketed_peak_decision(
    transitions: list[dict[str, Any]], frames: list[ScanFrame]
) -> int | None:
    """Decision whose accepted frame is the FIRST bracketed physical peak."""
    if len(frames) < 3:
        return None
    for index in range(1, len(frames) - 1):
        e_prev = frames[index - 1].energy_physical_hartree
        e_here = frames[index].energy_physical_hartree
        e_next = frames[index + 1].energy_physical_hartree
        if None in (e_prev, e_here, e_next):
            continue
        if e_here > e_prev and e_here >= e_next:
            for transition in transitions:
                if transition["accepted_frame_index"] == frames[index].frame_index:
                    return int(transition["decision_index"])
    return None


def _validate_decision(
    decision: ControlDecision, spec: ResidualControlSpec, base_targets: Sequence[float]
) -> None:
    if not decision.candidates:
        raise ControlScanError("policy returned an empty candidate list")
    if len(decision.candidates) != len(decision.candidate_sources):
        raise ControlScanError("candidate sources not parallel to candidates")
    has_zero = any(_is_zero_residual(u) for u in decision.candidates)
    if not has_zero:
        raise ControlScanError(
            "candidate list must contain the zero residual (frozen baseline)"
        )
    index = decision.executed_candidate_index
    if not 0 <= index < len(decision.candidates):
        raise ControlScanError("executed candidate index out of range")
    legal, reason = spec.residual_is_legal(
        decision.candidates[index], base_targets
    )
    if not legal:
        raise ControlScanError(f"executed candidate violates A(H): {reason}")


def _command_with_residual(
    spec: ResidualControlSpec,
    nominal_targets: Sequence[float],
    u: Sequence[float],
) -> tuple[float, ...] | None:
    """Command for one attempt; None when the residual is illegal here."""
    legal, _reason = spec.residual_is_legal(u, nominal_targets)
    if not legal:
        return None
    return tuple(
        float(nominal) + float(u_j) * eps
        for nominal, u_j, eps in zip(nominal_targets, u, spec.epsilon)
    )


def _finalize_transitions(
    *,
    transitions: list[dict[str, Any]],
    decision_meta: list[dict[str, Any]],
    frames: list[ScanFrame],
    control: ChemicalControl,
    environment: dict[str, Any],
    reaction_id: str,
    reward_spec: PolicyRewardSpec,
    policy_id: str,
    policy_hash: str,
    status: str,
    failure_code: str | None,
) -> list[ControlTransition]:
    """Compute candidate awards + rewards and freeze transition objects."""
    # Candidate award: the FIRST bracketed physical-energy peak among accepted
    # frames (frozen predicate; at most one bonus per episode).
    awarded_decision = _first_bracketed_peak_decision(transitions, frames)

    finalized: list[ControlTransition] = []
    for position, transition in enumerate(transitions):
        is_last = position == len(transitions) - 1
        episode_status = (
            _episode_status_for(failure_code if is_last else None, status)
            if is_last
            else "ongoing"
        )
        action: ControlAction = transition["action"]
        accepted_frame = (
            frames[transition["accepted_frame_index"]]
            if transition["accepted_frame_index"] is not None
            else None
        )
        actual_values: tuple[float | None, ...] = tuple(
            float(v) for v in accepted_frame.actual_values.values()
        ) if accepted_frame is not None else tuple(
            None for _ in action.driver_ids
        )
        macro_cost = MacroCost(
            wall_seconds=float(transition["macro_wall"]),
            host_requests=int(transition["macro_requests"]),
            n_optimization_cycles_logged=int(transition["macro_cycles"]),
            native_wall_seconds=float(transition["macro_wall"]),
        )
        # Frozen TD semantics (Demo24 plan §5): the recorded mask is the ONLY
        # bootstrap authority.  A final transition — terminal_complete,
        # terminal_step_limit, AND truncated_budget — never bootstraps: under
        # the undiscounted finite-budget objective no further return can
        # accrue after the episode ended, so a "truncated" status name alone
        # must not fabricate a successor value.  Only an ongoing transition
        # with a genuinely recorded successor decision (non-empty
        # next_candidates) bootstraps.
        bootstrap_mask = bool(
            (not is_last)
            and transitions[position + 1]["candidates"]
        )
        candidate_awarded = (
            awarded_decision is not None
            and transition["decision_index"] == awarded_decision
        )
        hard_failure = bool(
            is_last
            and failure_code is not None
            and failure_code in reward_spec.hard_failure_codes
        )
        reward = reward_spec.step_reward(
            macro_cost=macro_cost,
            candidate_awarded=candidate_awarded,
            hard_failure=hard_failure,
        )
        next_candidates = (
            tuple(
                tuple(float(v) for v in u)
                for u in transitions[position + 1]["candidates"]
            )
            if not is_last
            else ()
        )
        start_frame = frames[0] if not frames else None
        for frame in frames:
            if frame.frame_index == transition["decision_start_frame_index"]:
                start_frame = frame
                break
        finalized.append(
            ControlTransition(
                schema_version=CONTROL_TRANSITION_SCHEMA_VERSION,
                transition_id=transition["transition_id"],
                environment_id=environment["environment_id"],
                environment_hash=environment["environment_hash"],
                policy_id=policy_id,
                policy_hash=policy_hash,
                reaction_id=reaction_id,
                decision_index=transition["decision_index"],
                decision_start_frame_index=transition["decision_start_frame_index"],
                action=action,
                candidates=tuple(
                    tuple(float(v) for v in u) for u in transition["candidates"]
                ),
                candidate_sources=tuple(transition["candidate_sources"]),
                executed_candidate_index=transition["executed_candidate_index"],
                command_targets=action.command_targets,
                actual_values=actual_values,
                accepted=transition["accepted"],
                accepted_frame_index=transition["accepted_frame_index"],
                pending_target=transition["pending_target"],
                child_attempt_refs=tuple(transition["child_attempt_refs"]),
                macro_cost=macro_cost,
                rescue_used=False,
                refusal_reasons=tuple(transition["refusal_reasons"]),
                episode_status=episode_status,
                reward=reward,
                reward_events={
                    "candidate_awarded": bool(candidate_awarded),
                    "hard_failure": hard_failure,
                },
                bootstrap_mask=bootstrap_mask,
                decision_start_geometry=tuple(
                    tuple(float(x) for x in row)
                    for row in (start_frame.geometry if start_frame else [])
                ),
                state_after_geometry=tuple(
                    tuple(float(x) for x in row)
                    for row in transition["state_after_geometry"]
                ),
                next_candidates=next_candidates,
                decision_context_vector=tuple(
                    float(v) for v in transition["decision_context_vector"]
                ),
            )
        )
    return finalized


def _control_terminal_result(
    *,
    control: ChemicalControl,
    reaction_id: str,
    scan_policy: NativeScanPolicy,
    reward_spec: PolicyRewardSpec,
    environment: dict[str, Any],
    frames: list[ScanFrame],
    events: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    decision_meta: list[dict[str, Any]],
    request_count: int,
    wall_seconds: float,
    status: str,
    failure_code: str | None,
    failure_detail: str | None,
    completed_interval: bool = False,
) -> dict[str, Any]:
    policy_id = transitions[0]["action"].policy_id if transitions else "none"
    policy_hash = str(transitions[0]["policy_hash"]) if transitions else "none"
    finalized = _finalize_transitions(
        transitions=transitions,
        decision_meta=decision_meta,
        frames=frames,
        control=control,
        environment=environment,
        reaction_id=reaction_id,
        reward_spec=reward_spec,
        policy_id=policy_id,
        policy_hash=policy_hash,
        status=status,
        failure_code=failure_code,
    )

    accepted = frames[1:]
    total_cycles = sum(frame.n_optimization_cycles or 0 for frame in accepted)
    energies_physical = [
        frame.energy_physical_hartree for frame in accepted
        if frame.energy_physical_hartree is not None
    ]
    last_lambda = frames[-1].lambda_value if frames else None

    deviations: list[dict[str, Any]] = []
    for frame in accepted:
        nominal = control.targets_at(frame.lambda_value)
        actual = [frame.actual_values[driver.driver_id] for driver in control.drivers]
        deviations.append({
            "frame_index": frame.frame_index,
            "lambda": frame.lambda_value,
            "max_abs_deviation": max(
                abs(a - n) for a, n in zip(actual, nominal)
            ),
        })

    result = {
        "schema_version": CONTROL_RESULT_SCHEMA,
        "engine_mode": ENGINE_MODE_CONTROL,
        "engine_version": CONTROL_ENGINE_VERSION,
        "reaction_id": reaction_id,
        "status": status,
        "failure_code": failure_code,
        "failure_detail": failure_detail,
        "completed_interval": bool(completed_interval),
        "commanded_interval_complete": bool(completed_interval),
        "last_accepted_lambda": last_lambda,
        "n_accepted_frames_beyond_origin": len(accepted),
        "n_grid_points": len(control.lambda_grid),
        "frames": [frame.to_payload() for frame in frames],
        "energy_channel": {
            "physical_source": "unrestrained_xTB_single_point_same_geometry",
            "effective_source": "restraint_including_TOTAL_ENERGY",
            "physical_available": bool(energies_physical),
            "physical_min_hartree": min(energies_physical) if energies_physical else None,
            "physical_max_hartree": max(energies_physical) if energies_physical else None,
        },
        "environment": {
            "environment_id": environment["environment_id"],
            "environment_hash": environment["environment_hash"],
            "reward_spec_sha256": reward_spec.content_hash(),
            "candidate_generator_version": CANDIDATE_GENERATOR_VERSION,
            "rescue": "disabled_in_E_native",
        },
        "decisions": [
            {
                "decision_index": entry["decision_index"],
                "target_lambda": entry["target_lambda"],
                "source": entry["source"],
                "accepted": entry["accepted"],
                "macro_wall_seconds": entry["macro_wall"],
                "macro_host_requests": entry["macro_requests"],
            }
            for entry in decision_meta
        ],
        "actual_vs_nominal": {
            "note": "commanded targets may deviate from the nominal schedule; "
                    "nominal coverage is never claimed from perturbed frames",
            "max_abs_deviation": max(
                (entry["max_abs_deviation"] for entry in deviations), default=0.0
            ),
            "per_frame": deviations,
        },
        "active_atom_indices": list(control.active_atom_indices),
        "active_atom_maps": list(control.active_atom_maps),
        "execution_direction": control.execution_direction,
        "origin_endpoint": control.origin_endpoint,
        "policy": scan_policy.to_payload(),
        "plan_sha256": control.plan_sha256,
        "atom_order_sha256": control.atom_order_sha256,
        "cost": {
            "host_requests": request_count,
            "n_accepted_frames": len(accepted),
            "n_optimization_cycles_logged": total_cycles,
            "wall_seconds": wall_seconds,
        },
        "generated_at": _now(),
        "_events": events,
        "_transitions": [transition.to_payload() for transition in finalized],
    }
    return result


def persist_control_outputs(save_dir: str | Path, result: dict[str, Any]) -> None:
    """Write result + trace + transitions under ``<save_dir>/control/``."""
    root = Path(save_dir) / "control"
    root.mkdir(parents=True, exist_ok=True)
    events = result.pop("_events", [])
    transitions = result.pop("_transitions", [])
    trace = {
        "schema_version": CONTROL_TRACE_SCHEMA,
        "engine_mode": result.get("engine_mode"),
        "reaction_id": result.get("reaction_id"),
        "events": events,
        "generated_at": _now(),
    }
    _write_json(root / "control_scan_result.json", result)
    _write_json(root / "control_scan_trace.json", trace)
    with (root / "control_transitions.jsonl").open("a", encoding="utf-8") as handle:
        for payload in transitions:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


__all__ = [
    "ACP_WORKFLOW_NAME",
    "CANDIDATE_GENERATOR_VERSION",
    "CONTROL_ENGINE_VERSION",
    "CONTROL_RESULT_SCHEMA",
    "CONTROL_TRACE_SCHEMA",
    "CONTROL_TRANSITIONS_SCHEMA",
    "ControlDecision",
    "ControlScanError",
    "ControlScanState",
    "DecisionFn",
    "ENGINE_MODE_CONTROL",
    "FreshDecision",
    "LogicalTargetResult",
    "MacroCost",
    "NativeScanPolicy",
    "ScanFrame",
    "STATUS_BUDGET_EXHAUSTED",
    "STATUS_COMPLETE",
    "STATUS_PARTIAL_PREFIX",
    "STATUS_REJECTED_INPUT",
    "bonded_pairs",
    "chemistry_check",
    "classify_step_refusal",
    "freeze_control_environment",
    "persist_control_outputs",
    "prepare_step_attempt",
    "run_control_scan",
    "scaled_step_targets",
]
