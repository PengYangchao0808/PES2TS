"""Control contracts for the unified residual-decision round (ADR-0017).

Six registered draft schemas become concrete here (plan §6):

* ``pesn_control_input_v1``       — :class:`ControlInput` (observation +
  execution context + frozen :class:`ResidualControlSpec`).
* ``pesn_control_prediction_v1``  — :class:`ControlPrediction` (bounded
  per-coordinate residual proposal + per-candidate Q_proxy estimates).
* ``pes2ts_control_action_v2``    — :class:`ControlAction` (bounded k-dim
  residual, command targets, provenance; a vector, never a candidate index).
* ``pes2ts_control_transition_v2``— :class:`ControlTransition` (one decision
  macro-step: accepted prefix, pending target, child attempts, macro cost,
  rescue flag, terminated/truncated/censored + bootstrap mask).
* ``pes2ts_policy_reward_spec_v2``— :class:`PolicyRewardSpec` (undiscounted
  finite-budget proxy objective with typed terminal semantics).
* ``pesn_model_bundle_v2``        — capability-map extension (residual /
  q_proxy stay ``research_only``; every formal decision cap stays disabled).

Semantics frozen by the plan and enforced by ``validate``:

* the residual is bounded per driver: ``|delta_q_j| <= epsilon_j`` with
  ``u_j = delta_q_j / epsilon_j`` in ``[-1, 1]``; monitor/guard coordinates
  cannot be activated because the spec only covers driver coordinates;
* the command is an absolute target ``q_base + epsilon * u``; a zero residual
  reproduces the Classic baseline target exactly;
* nominal progress (``base_targets``) and the commanded/actual values are
  separate fields — a perturbed structure never masquerades as having
  reached the nominal lambda;
* ``Q_proxy`` is a research-only candidate-efficiency proxy; it is not
  Q_MOVE, not a success probability, and never unlocks production control.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final, Mapping, Sequence

CONTROL_INPUT_SCHEMA_VERSION: Final[str] = "pesn_control_input_v1"
CONTROL_PREDICTION_SCHEMA_VERSION: Final[str] = "pesn_control_prediction_v1"
CONTROL_ACTION_SCHEMA_VERSION: Final[str] = "pes2ts_control_action_v2"
CONTROL_TRANSITION_SCHEMA_VERSION: Final[str] = "pes2ts_control_transition_v2"
REWARD_SPEC_SCHEMA_VERSION: Final[str] = "pes2ts_policy_reward_spec_v2"
CONTROL_BUNDLE_SCHEMA_VERSION: Final[str] = "pesn_model_bundle_v2"

#: Optional v2 manifest block binding a registry-trained package to the
#: exact action-support registry it was trained with (review F3).  Bundles
#: without the block are legacy source-gate packages and stay readable.
CONTROL_SUPPORT_REGISTRY_FILE: Final[str] = "action_support_registry.json"
CONTROL_SUPPORT_REGISTRY_SCHEMA: Final[str] = "pesn_action_support_registry_v1"
CONTROL_SUPPORT_REGISTRY_BLOCK_KEYS: Final[tuple[str, ...]] = (
    "schema",
    "file",
    "sha256",
    "environment_hash",
    "reward_spec_sha256",
    "fit_transitions_digest",
    "n_fit_transitions",
)

ENVIRONMENT_NATIVE: Final[str] = "E_native"
ENVIRONMENT_HYBRID: Final[str] = "E_hybrid"

ACTION_SOURCE_RULE_ZERO: Final[str] = "rule_zero"
ACTION_SOURCE_MODEL_RESIDUAL: Final[str] = "model_residual"
ACTION_SOURCE_SCALED: Final[str] = "scaled_model_residual"
ACTION_SOURCE_PERTURB: Final[str] = "registered_perturbation"
ACTION_SOURCE_EXPLORE: Final[str] = "bounded_exploration"

ACTION_SOURCES: Final[frozenset[str]] = frozenset(
    {
        ACTION_SOURCE_RULE_ZERO,
        ACTION_SOURCE_MODEL_RESIDUAL,
        ACTION_SOURCE_SCALED,
        ACTION_SOURCE_PERTURB,
        ACTION_SOURCE_EXPLORE,
    }
)

#: Undiscounted finite-budget objective (plan §2): gamma is fixed at 1.
REWARD_GAMMA: Final[float] = 1.0


class ControlContractError(ValueError):
    """Typed contract violation (validation, never a silent default)."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"control contract violated [{code}]: {detail}")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise ControlContractError(code, detail)


# ---------------------------------------------------------------------------
# Frozen residual control spec (epsilon, allowed signs, command bounds)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResidualControlSpec:
    """Frozen per-driver residual bounds and legal command ranges.

    Rows are parallel to the compiled ``ChemicalControl.drivers`` order.  The
    spec is the single authority for "which coordinates may move, how far,
    in which direction" (plan §3); it never activates a monitor/guard.
    """

    schema_version: str
    driver_ids: tuple[str, ...]
    epsilon: tuple[float, ...]
    allowed_sign: tuple[int, ...]  # -1 / 0 / +1 (0 = both directions)
    min_target: tuple[float, ...]
    max_target: tuple[float, ...]

    def validate(self) -> None:
        _require(
            self.schema_version == "pesn_residual_control_spec_v1",
            "SCHEMA_UNKNOWN",
            f"residual spec schema {self.schema_version!r} unsupported",
        )
        n = len(self.driver_ids)
        _require(n > 0, "SPEC_EMPTY", "residual spec covers no driver")
        for name in ("epsilon", "allowed_sign", "min_target", "max_target"):
            _require(
                len(getattr(self, name)) == n,
                "SPEC_LENGTH_MISMATCH",
                f"{name} is not parallel to driver_ids",
            )
        for index, driver_id in enumerate(self.driver_ids):
            _require(bool(driver_id), "SPEC_DRIVER_ID_EMPTY", f"row {index}")
            _require(
                self.epsilon[index] > 0.0,
                "SPEC_EPSILON_INVALID",
                f"driver {driver_id} needs epsilon > 0",
            )
            _require(
                self.allowed_sign[index] in (-1, 0, 1),
                "SPEC_SIGN_INVALID",
                f"driver {driver_id} allowed_sign must be -1/0/+1",
            )
            _require(
                self.min_target[index] < self.max_target[index],
                "SPEC_RANGE_INVALID",
                f"driver {driver_id} target range is empty",
            )

    def content_hash(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "driver_ids": list(self.driver_ids),
            "epsilon": [float(v) for v in self.epsilon],
            "allowed_sign": [int(v) for v in self.allowed_sign],
            "min_target": [float(v) for v in self.min_target],
            "max_target": [float(v) for v in self.max_target],
        }
        import json

        return _sha256_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":"))
        )

    def residual_is_legal(
        self, u: Sequence[float], base_targets: Sequence[float]
    ) -> tuple[bool, str | None]:
        """Legality of one bounded residual against ``A(H)`` bounds."""
        if len(u) != len(self.driver_ids) or len(base_targets) != len(self.driver_ids):
            return False, "SPEC_LENGTH_MISMATCH"
        for index, driver_id in enumerate(self.driver_ids):
            u_j = float(u[index])
            if not -1.0 - 1e-12 <= u_j <= 1.0 + 1e-12:
                return False, f"residual {driver_id} |u|>1"
            delta = u_j * self.epsilon[index]
            sign = self.allowed_sign[index]
            if sign == 1 and delta < -1e-12:
                return False, f"residual {driver_id} sign forbidden"
            if sign == -1 and delta > 1e-12:
                return False, f"residual {driver_id} sign forbidden"
            target = float(base_targets[index]) + delta
            if not self.min_target[index] <= target <= self.max_target[index]:
                return False, f"command target {driver_id} outside frozen range"
        return True, None

    def command_targets(
        self, base_targets: Sequence[float], u: Sequence[float]
    ) -> tuple[float, ...]:
        legal, reason = self.residual_is_legal(u, base_targets)
        _require(legal, "ACTION_ILLEGAL", reason or "illegal residual")
        return tuple(
            float(base) + float(u_j) * eps
            for base, u_j, eps in zip(base_targets, u, self.epsilon)
        )


# ---------------------------------------------------------------------------
# Control input / prediction
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ControlContext:
    """Typed execution-context scalars entering the model with the state.

    Everything here is a *frozen observation* of budget / recovery memory
    (plan §2); no field may leak truth labels or future information.
    """

    nominal_lambda: float
    next_target_lambda: float
    remaining_wall_fraction: float
    remaining_request_fraction: float
    recent_rejections: int
    rescue_quota_remaining: int
    candidate_already_awarded: bool

    def validate(self) -> None:
        _require(
            -1e-9 <= self.nominal_lambda <= 1.0 + 1e-9,
            "CONTEXT_INVALID",
            "nominal_lambda outside [0, 1]",
        )
        _require(
            -1e-9 <= self.next_target_lambda <= 1.0 + 1e-9,
            "CONTEXT_INVALID",
            "next_target_lambda outside [0, 1]",
        )
        for name in ("remaining_wall_fraction", "remaining_request_fraction"):
            _require(
                -1e-9 <= getattr(self, name) <= 1.0 + 1e-9,
                "CONTEXT_INVALID",
                f"{name} outside [0, 1]",
            )
        _require(
            self.recent_rejections >= 0 and self.rescue_quota_remaining >= 0,
            "CONTEXT_INVALID",
            "negative budget/recovery memory",
        )

    def to_vector(self) -> tuple[float, ...]:
        return (
            float(self.nominal_lambda),
            float(self.next_target_lambda),
            float(self.remaining_wall_fraction),
            float(self.remaining_request_fraction),
            float(min(self.recent_rejections, 8)) / 8.0,
            float(self.rescue_quota_remaining),
            1.0 if self.candidate_already_awarded else 0.0,
        )


#: Dimension of :meth:`ControlContext.to_vector` — the model context width.
CONTROL_CONTEXT_DIM: Final[int] = 7


@dataclass(frozen=True, slots=True)
class ControlInput:
    """``pesn_control_input_v1``: state observation + execution context.

    ``base_input`` is the qualified :class:`PESNInput` of the ACCEPTED
    geometry (the state observation); the residual spec and nominal targets
    freeze what the policy may propose for the next decision.
    """

    schema_version: str
    base_input: object  # PESNInput (typed loosely to avoid a runtime-cycle)
    context: ControlContext
    spec: ResidualControlSpec
    base_targets: tuple[float, ...]

    @property
    def input_hash(self) -> str:
        return self.base_input.input_hash  # type: ignore[attr-defined]

    def validate(self) -> None:
        _require(
            self.schema_version == CONTROL_INPUT_SCHEMA_VERSION,
            "SCHEMA_UNKNOWN",
            f"control input schema {self.schema_version!r} unsupported",
        )
        _require(
            hasattr(self.base_input, "input_hash"),
            "INPUT_INVALID",
            "base_input must be a qualified PESNInput",
        )
        self.context.validate()
        self.spec.validate()
        _require(
            len(self.base_targets) == len(self.spec.driver_ids),
            "INPUT_LENGTH_MISMATCH",
            "base_targets not parallel to the residual spec",
        )


@dataclass(frozen=True, slots=True)
class CandidateQEstimate:
    """One candidate's Q_proxy estimate (parallel to the candidate list)."""

    available: bool
    value: float | None
    reason: str | None = None

    def validate(self) -> None:
        if self.available:
            _require(
                self.value is not None and self.reason is None,
                "HEAD_INVALID",
                "available Q estimate needs a value and no reason",
            )
        else:
            _require(
                self.reason in ("head_disabled", "backend_missing", "no_action_support"),
                "HEAD_INVALID",
                f"unknown Q unavailability reason {self.reason!r}",
            )


@dataclass(frozen=True, slots=True)
class ControlPrediction:
    """``pesn_control_prediction_v1``: proposals + candidate Q_proxy values.

    The prediction only *proposes*; G2 constructs the legal command and
    executes it (plan §3).  ``q_proxy`` is parallel to the caller's frozen
    candidate list and is research-only.
    """

    schema_version: str
    input_hash: str
    model_id: str
    model_hash: str
    feature_hash: str
    residual: tuple[float, ...]
    residual_available: bool
    residual_reason: str | None
    q_proxy: tuple[CandidateQEstimate, ...]
    research_only: bool = True

    def validate(self) -> None:
        _require(
            self.schema_version == CONTROL_PREDICTION_SCHEMA_VERSION,
            "SCHEMA_UNKNOWN",
            f"control prediction schema {self.schema_version!r} unsupported",
        )
        _require(
            self.research_only,
            "HEAD_INVALID",
            "control predictions must stay research_only",
        )
        if self.residual_available:
            _require(
                len(self.residual) > 0 and self.residual_reason is None,
                "HEAD_INVALID",
                "available residual needs components and no reason",
            )
            for value in self.residual:
                _require(-1.0 <= value <= 1.0, "HEAD_INVALID", "residual |u|>1")
        else:
            _require(
                self.residual_reason in ("head_disabled", "backend_missing"),
                "HEAD_INVALID",
                f"unknown residual unavailability reason {self.residual_reason!r}",
            )
        for estimate in self.q_proxy:
            estimate.validate()


# ---------------------------------------------------------------------------
# Action / transition
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ControlAction:
    """``pes2ts_control_action_v2``: the bounded residual actually commanded."""

    schema_version: str
    action_id: str
    driver_ids: tuple[str, ...]
    base_targets: tuple[float, ...]
    residual_u: tuple[float, ...]
    epsilon: tuple[float, ...]
    command_targets: tuple[float, ...]
    source: str
    policy_id: str

    def validate(self) -> None:
        _require(
            self.schema_version == CONTROL_ACTION_SCHEMA_VERSION,
            "SCHEMA_UNKNOWN",
            f"control action schema {self.schema_version!r} unsupported",
        )
        n = len(self.driver_ids)
        _require(n > 0, "ACTION_EMPTY", "action covers no driver")
        _require(
            len(self.base_targets) == n
            and len(self.residual_u) == n
            and len(self.epsilon) == n
            and len(self.command_targets) == n,
            "ACTION_LENGTH_MISMATCH",
            "action rows are not parallel",
        )
        _require(self.source in ACTION_SOURCES, "ACTION_SOURCE_UNKNOWN", self.source)
        _require(bool(self.action_id), "ACTION_ID_MISSING", "action_id empty")
        _require(bool(self.policy_id), "ACTION_ID_MISSING", "policy_id empty")
        for index, driver_id in enumerate(self.driver_ids):
            _require(
                -1.0 - 1e-12 <= self.residual_u[index] <= 1.0 + 1e-12,
                "ACTION_ILLEGAL",
                f"residual {driver_id} |u|>1",
            )
            expected = self.base_targets[index] + (
                self.residual_u[index] * self.epsilon[index]
            )
            _require(
                abs(expected - self.command_targets[index]) <= 1e-9,
                "ACTION_ILLEGAL",
                f"command target {driver_id} inconsistent with base+epsilon*u",
            )

    def content_hash(self) -> str:
        import json

        payload = {
            "schema_version": self.schema_version,
            "action_id": self.action_id,
            "driver_ids": list(self.driver_ids),
            "base_targets": [float(v) for v in self.base_targets],
            "residual_u": [float(v) for v in self.residual_u],
            "epsilon": [float(v) for v in self.epsilon],
            "command_targets": [float(v) for v in self.command_targets],
            "source": self.source,
            "policy_id": self.policy_id,
        }
        return _sha256_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":"))
        )


@dataclass(frozen=True, slots=True)
class MacroCost:
    """Aggregated per-decision cost; internal receipts are never re-billed."""

    wall_seconds: float
    host_requests: int
    n_optimization_cycles_logged: int
    native_wall_seconds: float = 0.0
    rescue_wall_seconds: float = 0.0
    bare_sp_wall_seconds: float = 0.0
    subdivision_wall_seconds: float = 0.0

    def validate(self) -> None:
        _require(
            self.wall_seconds >= 0.0 and self.host_requests >= 0,
            "COST_INVALID",
            "negative macro cost",
        )
        parts = (
            self.native_wall_seconds
            + self.rescue_wall_seconds
            + self.bare_sp_wall_seconds
            + self.subdivision_wall_seconds
        )
        _require(
            abs(parts - self.wall_seconds) <= 1e-6,
            "COST_INVALID",
            "macro wall cost does not equal the sum of its internal parts",
        )

    def to_payload(self) -> dict[str, float | int]:
        return {
            "wall_seconds": float(self.wall_seconds),
            "host_requests": int(self.host_requests),
            "n_optimization_cycles_logged": int(self.n_optimization_cycles_logged),
            "native_wall_seconds": float(self.native_wall_seconds),
            "rescue_wall_seconds": float(self.rescue_wall_seconds),
            "bare_sp_wall_seconds": float(self.bare_sp_wall_seconds),
            "subdivision_wall_seconds": float(self.subdivision_wall_seconds),
        }


TERMINAL_COMPLETE: Final[str] = "terminal_complete"
TERMINAL_STEP_LIMIT: Final[str] = "terminal_step_limit"
TRUNCATED_BUDGET: Final[str] = "truncated_budget"
CENSORED_MANUAL: Final[str] = "censored_manual"


@dataclass(frozen=True, slots=True)
class ControlTransition:
    """``pes2ts_control_transition_v2``: one decision macro-step record.

    Records base/proposal/command/target/actual separately, keeps every child
    attempt reference, the pending-target state, rescue usage, the macro cost
    breakdown, and the terminal/truncated/censored + bootstrap semantics
    (plan §5.2).  A rejected step without a new accepted frame keeps the
    parent geometry while budget / recovery memory still advance.
    """

    schema_version: str
    transition_id: str
    environment_id: str
    environment_hash: str
    policy_id: str
    policy_hash: str
    reaction_id: str
    decision_index: int
    decision_start_frame_index: int
    action: ControlAction
    candidates: tuple[tuple[float, ...], ...]
    candidate_sources: tuple[str, ...]
    executed_candidate_index: int
    command_targets: tuple[float, ...]
    actual_values: tuple[float | None, ...]
    accepted: bool
    accepted_frame_index: int | None
    pending_target: float | None
    child_attempt_refs: tuple[str, ...]
    macro_cost: MacroCost
    rescue_used: bool
    refusal_reasons: tuple[str, ...]
    episode_status: str  # one of TERMINAL_*/TRUNCATED_*/CENSORED_* or "ongoing"
    reward: float
    reward_events: Mapping[str, bool]
    bootstrap_mask: bool
    decision_start_geometry: tuple[tuple[float, ...], ...] = ()
    state_after_geometry: tuple[tuple[float, ...], ...] = ()
    next_candidates: tuple[tuple[float, ...], ...] = ()
    decision_context_vector: tuple[float, ...] = ()

    def validate(self) -> None:
        _require(
            self.schema_version == CONTROL_TRANSITION_SCHEMA_VERSION,
            "SCHEMA_UNKNOWN",
            f"transition schema {self.schema_version!r} unsupported",
        )
        self.action.validate()
        self.macro_cost.validate()
        _require(
            self.environment_id in (ENVIRONMENT_NATIVE, ENVIRONMENT_HYBRID),
            "ENVIRONMENT_UNKNOWN",
            f"environment {self.environment_id!r} unknown",
        )
        _require(
            self.action.command_targets == self.command_targets,
            "TRANSITION_INCONSISTENT",
            "transition command targets differ from the action",
        )
        _require(
            0 <= self.executed_candidate_index < max(1, len(self.candidates)),
            "TRANSITION_INCONSISTENT",
            "executed candidate index out of range",
        )
        _require(
            self.actual_values is None
            or len(self.actual_values) == len(self.action.driver_ids),
            "TRANSITION_INCONSISTENT",
            "actual values not parallel to drivers",
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "transition_id": self.transition_id,
            "environment_id": self.environment_id,
            "environment_hash": self.environment_hash,
            "policy_id": self.policy_id,
            "policy_hash": self.policy_hash,
            "reaction_id": self.reaction_id,
            "decision_index": self.decision_index,
            "decision_start_frame_index": self.decision_start_frame_index,
            "action": {
                "schema_version": self.action.schema_version,
                "action_id": self.action.action_id,
                "driver_ids": list(self.action.driver_ids),
                "base_targets": [float(v) for v in self.action.base_targets],
                "residual_u": [float(v) for v in self.action.residual_u],
                "epsilon": [float(v) for v in self.action.epsilon],
                "command_targets": [float(v) for v in self.action.command_targets],
                "source": self.action.source,
                "policy_id": self.action.policy_id,
                "content_sha256": self.action.content_hash(),
            },
            "candidates": [list(map(float, u)) for u in self.candidates],
            "candidate_sources": list(self.candidate_sources),
            "executed_candidate_index": self.executed_candidate_index,
            "command_targets": [float(v) for v in self.command_targets],
            "actual_values": [
                None if v is None else float(v) for v in self.actual_values
            ],
            "accepted": bool(self.accepted),
            "accepted_frame_index": self.accepted_frame_index,
            "pending_target": self.pending_target,
            "child_attempt_refs": list(self.child_attempt_refs),
            "macro_cost": self.macro_cost.to_payload(),
            "rescue_used": bool(self.rescue_used),
            "refusal_reasons": list(self.refusal_reasons),
            "episode_status": self.episode_status,
            "reward": float(self.reward),
            "reward_events": dict(self.reward_events),
            "bootstrap_mask": bool(self.bootstrap_mask),
            "decision_start_geometry": [
                [float(x) for x in row] for row in self.decision_start_geometry
            ],
            "state_after_geometry": [
                [float(x) for x in row] for row in self.state_after_geometry
            ],
            "next_candidates": [list(map(float, u)) for u in self.next_candidates],
            "decision_context": {
                "vector": [float(v) for v in self.decision_context_vector],
            },
        }


# ---------------------------------------------------------------------------
# Reward spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PolicyRewardSpec:
    """``pes2ts_policy_reward_spec_v2``: undiscounted finite-budget proxy.

    ``R_episode = B*I(candidate submitted) - sum_step C_step/C_ref
    - kappa*I(known hard failure)``.  Each decision pays its own macro cost;
    the candidate bonus is awarded at most once per episode; budget
    exhaustion is a finite-budget truncation (not a failure penalty); manual
    interruption is censored and never fabricates a reward.  This is a
    candidate-efficiency proxy — it is NOT a TS-success or production
    objective (plan §4).
    """

    schema_version: str
    candidate_bonus: float
    cost_reference_seconds: float
    hard_failure_penalty: float
    hard_failure_codes: frozenset[str]
    cost_unit: str = "wall_seconds"
    gamma: float = REWARD_GAMMA

    def validate(self) -> None:
        _require(
            self.schema_version == REWARD_SPEC_SCHEMA_VERSION,
            "SCHEMA_UNKNOWN",
            f"reward spec schema {self.schema_version!r} unsupported",
        )
        _require(
            self.candidate_bonus >= 0.0 and self.hard_failure_penalty >= 0.0,
            "REWARD_INVALID",
            "negative bonus/penalty",
        )
        _require(
            self.cost_reference_seconds > 0.0,
            "REWARD_INVALID",
            "cost reference must be positive",
        )
        _require(
            self.cost_unit == "wall_seconds",
            "REWARD_INVALID",
            f"unknown cost unit {self.cost_unit!r}",
        )
        _require(self.gamma == 1.0, "REWARD_INVALID", "gamma must stay 1.0")

    def step_reward(
        self,
        *,
        macro_cost: MacroCost,
        candidate_awarded: bool,
        hard_failure: bool,
    ) -> float:
        cost = float(macro_cost.wall_seconds) / self.cost_reference_seconds
        return (
            (self.candidate_bonus if candidate_awarded else 0.0)
            - cost
            - (self.hard_failure_penalty if hard_failure else 0.0)
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "candidate_bonus": float(self.candidate_bonus),
            "cost_reference_seconds": float(self.cost_reference_seconds),
            "hard_failure_penalty": float(self.hard_failure_penalty),
            "hard_failure_codes": sorted(self.hard_failure_codes),
            "cost_unit": self.cost_unit,
            "gamma": self.gamma,
        }

    def content_hash(self) -> str:
        import json

        return _sha256_text(
            json.dumps(self.to_payload(), sort_keys=True, separators=(",", ":"))
        )


#: Default known hard-failure codes for the native environment (typed
#: STEP_LIMIT family; budget exhaustion is a truncation, never a penalty).
DEFAULT_HARD_FAILURE_CODES: Final[frozenset[str]] = frozenset(
    {
        "STEP_LIMIT:REJECTIONS_EXHAUSTED",
        "STEP_LIMIT:SUBDIVISIONS_EXHAUSTED",
    }
)


# ---------------------------------------------------------------------------
# Capability map v2 (research-only residual / q_proxy)
# ---------------------------------------------------------------------------

CAP_RESIDUAL: Final[str] = "residual_head"
CAP_Q_PROXY: Final[str] = "q_proxy"

#: v2 key set = v1 keys + the two research control heads.
CONTROL_CAPABILITY_KEYS: Final[tuple[str, ...]] = (
    "encoder",
    "progress",
    "path_proximity",
    "teacher_direction",
    "coordinate_activity",
    CAP_RESIDUAL,
    CAP_Q_PROXY,
    "p_seed",
    "c_full",
    "q_stop",
    "q_move",
    "production_control",
)

#: Formal caps that must stay disabled in EVERY bundle (v1 and v2).
_V1_MUST_BE_DISABLED: Final[frozenset[str]] = frozenset(
    {"p_seed", "c_full", "q_stop", "q_move", "production_control"}
)


def validate_control_capability_map(caps: Mapping[str, str]) -> None:
    """v2 validator: exact v2 key set; residual/q_proxy research_only only."""
    from pes2ts_core.pesn.contracts.predictions import UnsupportedCapability

    unknown = set(caps) - set(CONTROL_CAPABILITY_KEYS)
    if unknown:
        raise UnsupportedCapability(
            sorted(unknown)[0], f"unknown capability keys: {sorted(unknown)}"
        )
    missing = set(CONTROL_CAPABILITY_KEYS) - set(caps)
    if missing:
        raise UnsupportedCapability(
            sorted(missing)[0], f"missing capability keys: {sorted(missing)}"
        )
    for key in _V1_MUST_BE_DISABLED:
        if caps[key] != "disabled":
            raise UnsupportedCapability(
                key, f"capability must stay disabled, got {caps[key]!r}"
            )
    for key in (CAP_RESIDUAL, CAP_Q_PROXY):
        if caps[key] not in ("disabled", "research_only"):
            raise UnsupportedCapability(
                key,
                "control heads are research_only at most; formal enabling is "
                "not unlocked by name proximity",
            )


def control_capability_map(
    *, residual: bool, q_proxy: bool
) -> dict[str, str]:
    """Build the v2 research capability map (formal caps all disabled)."""
    caps: dict[str, str] = {
        "encoder": "enabled",
        "progress": "disabled",
        "path_proximity": "disabled",
        "teacher_direction": "disabled",
        "coordinate_activity": "disabled",
        CAP_RESIDUAL: "research_only" if residual else "disabled",
        CAP_Q_PROXY: "research_only" if q_proxy else "disabled",
    }
    for key in _V1_MUST_BE_DISABLED:
        caps[key] = "disabled"
    validate_control_capability_map(caps)
    return caps


def validate_support_registry_binding(block: object) -> None:
    """Strict validator for the optional ``support_registry`` manifest block."""
    from pes2ts_core.pesn.contracts.artifacts import ArtifactError, _is_sha256

    if not isinstance(block, Mapping):
        raise ArtifactError(
            "SUPPORT_REGISTRY_INVALID", "support_registry block must be a mapping"
        )
    missing = [
        key for key in CONTROL_SUPPORT_REGISTRY_BLOCK_KEYS if key not in block
    ]
    if missing:
        raise ArtifactError(
            "SUPPORT_REGISTRY_INVALID",
            f"support_registry block missing keys: {missing}",
        )
    if block["schema"] != CONTROL_SUPPORT_REGISTRY_SCHEMA:
        raise ArtifactError(
            "SUPPORT_REGISTRY_INVALID",
            f"support_registry schema {block['schema']!r} != "
            f"{CONTROL_SUPPORT_REGISTRY_SCHEMA!r}",
        )
    if block["file"] != CONTROL_SUPPORT_REGISTRY_FILE:
        raise ArtifactError(
            "SUPPORT_REGISTRY_INVALID",
            f"support_registry file must be {CONTROL_SUPPORT_REGISTRY_FILE!r}",
        )
    for key in ("sha256", "environment_hash", "reward_spec_sha256", "fit_transitions_digest"):
        if not _is_sha256(block.get(key)):
            raise ArtifactError(
                "SUPPORT_REGISTRY_INVALID",
                f"support_registry {key} is not a sha256 hex digest",
            )
    if not isinstance(block.get("n_fit_transitions"), int) or block[
        "n_fit_transitions"
    ] < 0:
        raise ArtifactError(
            "SUPPORT_REGISTRY_INVALID",
            "support_registry n_fit_transitions must be a non-negative int",
        )


def validate_control_bundle_manifest(doc: Mapping[str, object]) -> None:
    """``pesn_model_bundle_v2`` manifest validator (v1 fields + control)."""
    from pes2ts_core.pesn.contracts.artifacts import (
        BUNDLE_FORBIDDEN_KEYS,
        ArtifactError,
    )

    if doc.get("schema_version") != CONTROL_BUNDLE_SCHEMA_VERSION:
        raise ArtifactError(
            "SCHEMA_UNKNOWN",
            f"bundle schema {doc.get('schema_version')!r} != "
            f"{CONTROL_BUNDLE_SCHEMA_VERSION!r}",
        )
    required = (
        "model_id",
        "model_hash",
        "feature_spec_sha256",
        "label_spec_sha256",
        "normalization_sha256",
        "weights_sha256",
        "training_run_ids",
        "capabilities",
        "model_card",
        "checksums",
        "control_spec_sha256",
        "environment_id",
        "reward_spec_sha256",
    )
    missing = [key for key in required if key not in doc]
    if missing:
        raise ArtifactError("MISSING_KEY", f"bundle missing keys: {missing}")
    present = [key for key in BUNDLE_FORBIDDEN_KEYS if key in doc]
    if present:
        raise ArtifactError("FORBIDDEN_KEY", f"bundle carries forbidden keys: {present}")
    caps = doc.get("capabilities")
    if not isinstance(caps, dict):
        raise ArtifactError("MISSING_KEY", "bundle capabilities must be a mapping")
    try:
        validate_control_capability_map({str(k): str(v) for k, v in caps.items()})
    except Exception as exc:
        raise ArtifactError("CAPABILITY_INVALID", str(exc)) from exc
    if doc.get("environment_id") not in (ENVIRONMENT_NATIVE, ENVIRONMENT_HYBRID):
        raise ArtifactError(
            "ENVIRONMENT_UNKNOWN",
            f"environment_id {doc.get('environment_id')!r} unknown",
        )
    if doc.get("research_only") is not True:
        raise ArtifactError(
            "RESEARCH_ONLY_REQUIRED",
            "control bundles must declare research_only=true",
        )
    if "support_registry" in doc:
        validate_support_registry_binding(doc["support_registry"])


__all__ = [
    "ACTION_SOURCES",
    "ACTION_SOURCE_EXPLORE",
    "ACTION_SOURCE_MODEL_RESIDUAL",
    "ACTION_SOURCE_PERTURB",
    "ACTION_SOURCE_RULE_ZERO",
    "ACTION_SOURCE_SCALED",
    "CENSORED_MANUAL",
    "CONTROL_ACTION_SCHEMA_VERSION",
    "CONTROL_BUNDLE_SCHEMA_VERSION",
    "CONTROL_CAPABILITY_KEYS",
    "CONTROL_CONTEXT_DIM",
    "CONTROL_INPUT_SCHEMA_VERSION",
    "CONTROL_PREDICTION_SCHEMA_VERSION",
    "CONTROL_SUPPORT_REGISTRY_BLOCK_KEYS",
    "CONTROL_SUPPORT_REGISTRY_FILE",
    "CONTROL_SUPPORT_REGISTRY_SCHEMA",
    "CONTROL_TRANSITION_SCHEMA_VERSION",
    "CandidateQEstimate",
    "ControlAction",
    "ControlContractError",
    "ControlContext",
    "ControlInput",
    "ControlPrediction",
    "ControlTransition",
    "DEFAULT_HARD_FAILURE_CODES",
    "ENVIRONMENT_HYBRID",
    "ENVIRONMENT_NATIVE",
    "MacroCost",
    "PolicyRewardSpec",
    "REWARD_GAMMA",
    "REWARD_SPEC_SCHEMA_VERSION",
    "ResidualControlSpec",
    "TERMINAL_COMPLETE",
    "TERMINAL_STEP_LIMIT",
    "TRUNCATED_BUDGET",
    "control_capability_map",
    "validate_control_bundle_manifest",
    "validate_control_capability_map",
]
