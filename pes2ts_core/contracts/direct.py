"""Direct-displacement training contracts (WP0, plan v3 §3–§5, §10).

Typed, validated data contracts for the ``E_direct_xtb_v1`` MDP family.
This module follows the pesn contracts discipline: typed dataclasses plus
loud rejections, no imports from training/data/backend modules, numpy only.

Schemas introduced here (all content-hashable via ``stable_json_dumps``):

* ``pesn_direct_state_v1``       — observable task/environment state
* ``pesn_direct_action_v1``      — latent/executed action record
* ``pes2ts_direct_transition_v1``— one full transition with masks/receipts
* ``pes2ts_direct_reward_spec_v1``— frozen reward/potential coefficients

Boundary statements (plan §3/§4): 900-second operational limits are censoring
events, not MDP failure terminals. Logical budget is 100 actions; every
sampled action (including rejections) consumes one unit. Expert actions carry
``null`` log-probability with ``expert_not_probabilistic`` — never a fabricated
sampling probability.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Mapping, Sequence

import numpy as np

from pes2ts_core.utils.hashing import JSONValue, sha256_bytes, stable_json_dumps


def _sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))

STATE_SCHEMA_VERSION: Final[str] = "pesn_direct_state_v1"
ACTION_SCHEMA_VERSION: Final[str] = "pesn_direct_action_v1"
TRANSITION_SCHEMA_VERSION: Final[str] = "pes2ts_direct_transition_v1"
REWARD_SPEC_SCHEMA_VERSION: Final[str] = "pes2ts_direct_reward_spec_v1"
DIRECT_MDP_SCHEMA_VERSION: Final[str] = "pes2ts_direct_mdp_spec_v1"


class DirectContractError(ValueError):
    """Typed rejection for direct-contract violations."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise DirectContractError(code, detail)


# ---------------------------------------------------------------------------
# Frozen MDP specification (reward + budget + thresholds; plan §3.1, §5)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DirectRewardSpec:
    """``reward_direct_proxy_v1`` coefficients. Fitted only via fit50 review."""

    schema_version: str = REWARD_SPEC_SCHEMA_VERSION
    budget_steps: int = 100            # H — logical action budget
    step_cost: float = 0.01            # c
    success_bonus: float = 2.0         # B
    failure_penalty: float = 1.0       # F
    shaping_scale: float = 0.1         # eta
    gamma: float = 1.0
    no_progress_window: int = 20
    no_progress_tolerance: float = 1e-3
    goal_consecutive_hits: int = 2

    def validate(self) -> None:
        _require(self.budget_steps > 0, "BUDGET_INVALID", "budget_steps must be positive")
        _require(self.step_cost >= 0.0, "REWARD_INVALID", "step_cost must be >= 0")
        _require(self.gamma == 1.0, "REWARD_INVALID", "v1 fixes gamma=1")
        _require(
            self.no_progress_window > 0 and self.goal_consecutive_hits > 0,
            "REWARD_INVALID", "windows must be positive",
        )

    def to_payload(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "budget_steps": self.budget_steps,
            "step_cost": self.step_cost,
            "success_bonus": self.success_bonus,
            "failure_penalty": self.failure_penalty,
            "shaping_scale": self.shaping_scale,
            "gamma": self.gamma,
            "no_progress_window": self.no_progress_window,
            "no_progress_tolerance": self.no_progress_tolerance,
            "goal_consecutive_hits": self.goal_consecutive_hits,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, JSONValue]) -> "DirectRewardSpec":
        _require(
            payload.get("schema_version") == REWARD_SPEC_SCHEMA_VERSION,
            "SCHEMA_UNKNOWN", f"expected {REWARD_SPEC_SCHEMA_VERSION}",
        )
        spec = cls(
            budget_steps=int(payload["budget_steps"]),
            step_cost=float(payload["step_cost"]),
            success_bonus=float(payload["success_bonus"]),
            failure_penalty=float(payload["failure_penalty"]),
            shaping_scale=float(payload["shaping_scale"]),
            gamma=float(payload["gamma"]),
            no_progress_window=int(payload["no_progress_window"]),
            no_progress_tolerance=float(payload["no_progress_tolerance"]),
            goal_consecutive_hits=int(payload["goal_consecutive_hits"]),
        )
        spec.validate()
        return spec


DEFAULT_REWARD_SPEC: Final[DirectRewardSpec] = DirectRewardSpec()


# ---------------------------------------------------------------------------
# State (plan §3.2): observable environment state, no identity shortcuts
# ---------------------------------------------------------------------------

FORBIDDEN_STATE_FIELDS: Final[tuple[str, ...]] = (
    "reference_ts", "irc_frames", "future_frames", "reference_path",
    "signed_progress", "path_proximity", "source_reader", "file_paths",
)


@dataclass(frozen=True, slots=True)
class DirectState:
    """Observable state of ``E_direct_xtb_v1`` (numpy arrays are copies)."""

    atomic_numbers: np.ndarray            # (N,) int
    coordinates: np.ndarray               # (N,3) float, angstrom
    task_graph: Mapping[str, JSONValue]   # H_direction target-graph contract doc
    atom_map_ids: tuple[str, ...]         # N identity labels (R-side atom map)
    charge: int = 0
    multiplicity: int = 1
    method_id: str = "GFN2-xTB"
    energy_hartree: float | None = None   # E(X); None until first observation
    energy_reference_hartree: float | None = None  # E(reset) for delta_E
    gradient_hartree_per_angstrom: np.ndarray | None = None  # (N,3) or None
    observation_mask: bool = True         # False when E/g unavailable
    steps_remaining: int = 100
    event_state: Mapping[str, float] = field(default_factory=dict)
    # event_state keys (plan §3.2): goal_consecutive_hits,
    # no_progress_count, best_potential
    schema_version: str = STATE_SCHEMA_VERSION

    def validate(self) -> None:
        _require(
            self.schema_version == STATE_SCHEMA_VERSION,
            "SCHEMA_UNKNOWN", f"expected {STATE_SCHEMA_VERSION}",
        )
        z = np.asarray(self.atomic_numbers)
        x = np.asarray(self.coordinates, dtype=float)
        _require(z.ndim == 1 and z.size > 0, "ATOM_COUNT_INVALID", "atomic_numbers must be (N,)")
        _require(
            x.shape == (z.size, 3), "ATOM_COUNT_MISMATCH",
            f"coordinates {x.shape} incompatible with {z.size} atoms",
        )
        _require(bool(np.isfinite(x).all()), "NONFINITE_COORDINATES", "coordinates contain NaN/Inf")
        _require(len(self.atom_map_ids) == z.size, "MAP_MISMATCH", "atom_map_ids length mismatch")
        _require(
            self.steps_remaining >= 0 and self.steps_remaining <= DEFAULT_REWARD_SPEC.budget_steps,
            "BUDGET_INVALID", "steps_remaining outside [0, budget]",
        )
        for key in self.task_graph:
            _require(
                key not in FORBIDDEN_STATE_FIELDS, "FORBIDDEN_FIELD",
                f"state task_graph carries forbidden identity field {key!r}",
            )

    def delta_e_hartree(self) -> float | None:
        if self.energy_hartree is None or self.energy_reference_hartree is None:
            return None
        return self.energy_hartree - self.energy_reference_hartree

    def n_atoms(self) -> int:
        return int(np.asarray(self.atomic_numbers).size)

    def to_payload(self) -> dict[str, JSONValue]:
        """Content-hashable payload (arrays as nested lists, rounded)."""
        grad = None
        if self.gradient_hartree_per_angstrom is not None:
            grad = np.asarray(self.gradient_hartree_per_angstrom, dtype=float).round(10).tolist()
        return {
            "schema_version": self.schema_version,
            "atomic_numbers": np.asarray(self.atomic_numbers).astype(int).tolist(),
            "coordinates": np.asarray(self.coordinates, dtype=float).round(6).tolist(),
            "task_graph": dict(self.task_graph),
            "atom_map_ids": list(self.atom_map_ids),
            "charge": int(self.charge),
            "multiplicity": int(self.multiplicity),
            "method_id": self.method_id,
            "energy_hartree": self.energy_hartree,
            "energy_reference_hartree": self.energy_reference_hartree,
            "gradient": grad,
            "observation_mask": bool(self.observation_mask),
            "steps_remaining": int(self.steps_remaining),
            "event_state": {k: float(v) for k, v in self.event_state.items()},
        }

    def content_hash(self) -> str:
        return _sha256_text(stable_json_dumps(self.to_payload()))

    @classmethod
    def from_payload(cls, payload: Mapping[str, JSONValue]) -> "DirectState":
        grad = payload.get("gradient")
        state = cls(
            atomic_numbers=np.asarray(payload["atomic_numbers"], dtype=int),
            coordinates=np.asarray(payload["coordinates"], dtype=float),
            task_graph=dict(payload["task_graph"]),
            atom_map_ids=tuple(payload["atom_map_ids"]),
            charge=int(payload["charge"]),
            multiplicity=int(payload["multiplicity"]),
            method_id=str(payload["method_id"]),
            energy_hartree=payload["energy_hartree"],
            energy_reference_hartree=payload["energy_reference_hartree"],
            gradient_hartree_per_angstrom=(
                None if grad is None else np.asarray(grad, dtype=float)),
            observation_mask=bool(payload["observation_mask"]),
            steps_remaining=int(payload["steps_remaining"]),
            event_state={k: float(v) for k, v in payload["event_state"].items()},
        )
        state.validate()
        return state


# ---------------------------------------------------------------------------
# Action (plan §6.1): latent u -> normalized z -> physical a; rejection record
# ---------------------------------------------------------------------------

REJECTION_REASONS: Final[frozenset[str]] = frozenset({
    "nonfinite_latent", "support_boundary_violation", "identity_violation",
    "catastrophic_overlap", "wrong_atom_count", "not_executed_censored",
    "expert_not_probabilistic",
})


@dataclass(frozen=True, slots=True)
class DirectAction:
    """One sampled/executed displacement action.

    ``latent_u``/``normalized_z``/``proposed_angstrom``/``executed_angstrom``
    are recorded separately; unexecuted fields are ``None`` with a rejection
    reason (plan §10).
    """

    schema_version: str = ACTION_SCHEMA_VERSION
    action_max_angstrom: float = 0.1
    latent_u: np.ndarray | None = None        # (N,3) raw Gaussian latent
    normalized_z: np.ndarray | None = None    # (N,3) tanh-radial normalized
    proposed_angstrom: np.ndarray | None = None   # (N,3) = a_max * z
    executed_angstrom: np.ndarray | None = None   # (N,3) actually applied
    executed: bool = False
    rejection_reason: str | None = None
    log_prob_z: float | None = None           # log pi_z; None for experts
    log_prob_source: str = "policy"           # policy | expert_not_probabilistic

    def validate(self) -> None:
        _require(
            self.schema_version == ACTION_SCHEMA_VERSION,
            "SCHEMA_UNKNOWN", f"expected {ACTION_SCHEMA_VERSION}",
        )
        _require(self.action_max_angstrom > 0, "ACTION_MAX_INVALID", "a_max must be positive")
        # Non-finite latent_u is an accepted *numeric fault* record: it flows
        # through the advancer as rejection "nonfinite_latent" (td_eligible
        # false, collection version stops). The execution carriers below must
        # always be finite when present.
        for name in ("normalized_z", "proposed_angstrom", "executed_angstrom"):
            arr = getattr(self, name)
            if arr is not None:
                a = np.asarray(arr, dtype=float)
                _require(a.ndim == 2 and a.shape[1] == 3, "ACTION_SHAPE_INVALID",
                         f"{name} must be (N,3), got {a.shape}")
                _require(bool(np.isfinite(a).all()), "NONFINITE_ACTION",
                         f"{name} contains NaN/Inf")
        if self.rejection_reason is not None:
            _require(
                self.rejection_reason in REJECTION_REASONS,
                "REJECTION_REASON_UNKNOWN", f"{self.rejection_reason!r} not registered",
            )
            _require(not self.executed, "REJECTED_BUT_EXECUTED",
                     "a rejected action cannot carry an executed displacement")
        if self.executed:
            _require(self.rejection_reason is None, "EXECUTED_WITH_REJECTION",
                     "executed actions carry no rejection reason")
        if self.log_prob_source == "expert_not_probabilistic":
            _require(self.log_prob_z is None, "EXPERT_LOGPROB_FORBIDDEN",
                     "expert actions carry null log-probability")

    def to_payload(self) -> dict[str, JSONValue]:
        def _r(arr):
            return None if arr is None else np.asarray(arr, dtype=float).round(8).tolist()
        return {
            "schema_version": self.schema_version,
            "action_max_angstrom": float(self.action_max_angstrom),
            "latent_u": _r(self.latent_u),
            "normalized_z": _r(self.normalized_z),
            "proposed_angstrom": _r(self.proposed_angstrom),
            "executed_angstrom": _r(self.executed_angstrom),
            "executed": bool(self.executed),
            "rejection_reason": self.rejection_reason,
            "log_prob_z": None if self.log_prob_z is None else float(self.log_prob_z),
            "log_prob_source": self.log_prob_source,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, JSONValue]) -> "DirectAction":
        def _arr(name):
            value = payload.get(name)
            return None if value is None else np.asarray(value, dtype=float)
        action = cls(
            action_max_angstrom=float(payload["action_max_angstrom"]),
            latent_u=_arr("latent_u"), normalized_z=_arr("normalized_z"),
            proposed_angstrom=_arr("proposed_angstrom"),
            executed_angstrom=_arr("executed_angstrom"),
            executed=bool(payload["executed"]),
            rejection_reason=payload["rejection_reason"],
            log_prob_z=(None if payload["log_prob_z"] is None
                        else float(payload["log_prob_z"])),
            log_prob_source=str(payload["log_prob_source"]),
        )
        action.validate()
        return action


# ---------------------------------------------------------------------------
# Transition (plan §4): the full supervised signal with masks and receipts
# ---------------------------------------------------------------------------

TERMINATION_REASONS: Final[frozenset[str]] = frozenset({
    "continued",                     # non-terminal step
    "goal_connected_proxy",          # success terminal
    "illegal_action",                # pre-execution rejection terminal
    "qc_failure_terminal",           # deterministic geometry-related QC failure
    "budget_exhausted",              # 100 steps consumed without goal
    "no_progress_terminal",          # frozen no-progress termination
    "censored_operational_limit",    # 900s/cancel/scheduling — NOT a failure
    "canceled",
    "data_slice_boundary",           # teacher data boundary; not a task terminal
})


@dataclass(frozen=True, slots=True)
class DirectTransition:
    """One ``pes2ts_direct_transition_v1`` record (plan §4 checklist)."""

    transition_id: str
    reaction_id: str
    group_id: str
    subset: str                        # e.g. direct_fit50
    episode_id: str
    step_index: int
    state: DirectState
    action: DirectAction
    next_state: DirectState | None    # None only for pre-execution rejections
    reward: float
    reward_components: Mapping[str, float]
    terminated: bool
    bootstrap: float                  # 0 for terminals, 1 to bootstrap s'
    termination_reason: str
    td_eligible: bool
    logical_cost: int                 # 1 per sampled action (rejections included)
    qc_cost_evidence: Mapping[str, JSONValue] | None = None
    environment_version: str = "E_direct_xtb_v1"
    advancer_version: str = ""
    reward_spec_sha256: str = ""
    parent_node_hash: str = ""
    schema_version: str = TRANSITION_SCHEMA_VERSION

    def validate(self) -> None:
        _require(
            self.schema_version == TRANSITION_SCHEMA_VERSION,
            "SCHEMA_UNKNOWN", f"expected {TRANSITION_SCHEMA_VERSION}",
        )
        _require(self.step_index >= 0, "STEP_INVALID", "step_index must be >= 0")
        _require(
            self.termination_reason in TERMINATION_REASONS,
            "TERMINATION_UNKNOWN", f"{self.termination_reason!r} not registered",
        )
        _require(self.logical_cost == 1, "LOGICAL_COST_INVALID",
                 "v1 charges exactly one logical unit per sampled action")
        _require(self.bootstrap in (0.0, 1.0), "BOOTSTRAP_INVALID", "bootstrap must be 0 or 1")
        if self.termination_reason.startswith("censored"):
            _require(not self.terminated, "CENSOR_NOT_TERMINAL",
                     "censoring is not an MDP terminal state")
        if self.termination_reason == "data_slice_boundary":
            _require(not self.terminated, "SLICE_NOT_TERMINAL",
                     "teacher slice boundary is not a task terminal")
        if self.terminated:
            _require(self.bootstrap == 0.0, "BOOTSTRAP_ON_TERMINAL",
                     "terminals must not bootstrap")
        else:
            _require(self.next_state is not None, "NEXT_STATE_MISSING",
                     "non-terminal transitions carry next_state")
        if self.action.log_prob_source == "expert_not_probabilistic":
            _require(self.action.log_prob_z is None, "EXPERT_LOGPROB_FORBIDDEN",
                     "expert transitions never carry fabricated log-probability")
        _require(bool(np.isfinite(self.reward)), "REWARD_NONFINITE",
                 "non-finite reward is a model/numeric fault, not an MDP event")
        if self.action.rejection_reason == "not_executed_censored":
            _require(self.termination_reason.startswith("censored"),
                     "CENSOR_MISMATCH", "censored rejection requires censored reason")

    def to_payload(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "transition_id": self.transition_id,
            "reaction_id": self.reaction_id,
            "group_id": self.group_id,
            "subset": self.subset,
            "episode_id": self.episode_id,
            "step_index": int(self.step_index),
            "state": self.state.to_payload(),
            "action": self.action.to_payload(),
            "next_state": None if self.next_state is None else self.next_state.to_payload(),
            "reward": float(self.reward),
            "reward_components": {k: float(v) for k, v in self.reward_components.items()},
            "terminated": bool(self.terminated),
            "bootstrap": float(self.bootstrap),
            "termination_reason": self.termination_reason,
            "td_eligible": bool(self.td_eligible),
            "logical_cost": int(self.logical_cost),
            "qc_cost_evidence": dict(self.qc_cost_evidence) if self.qc_cost_evidence else None,
            "environment_version": self.environment_version,
            "advancer_version": self.advancer_version,
            "reward_spec_sha256": self.reward_spec_sha256,
            "parent_node_hash": self.parent_node_hash,
        }

    def content_hash(self) -> str:
        return _sha256_text(stable_json_dumps(self.to_payload()))

    @classmethod
    def from_payload(cls, payload: Mapping[str, JSONValue]) -> "DirectTransition":
        nxt = payload.get("next_state")
        transition = cls(
            transition_id=str(payload["transition_id"]),
            reaction_id=str(payload["reaction_id"]),
            group_id=str(payload["group_id"]),
            subset=str(payload["subset"]),
            episode_id=str(payload["episode_id"]),
            step_index=int(payload["step_index"]),
            state=DirectState.from_payload(payload["state"]),
            action=DirectAction.from_payload(payload["action"]),
            next_state=(None if nxt is None
                        else DirectState.from_payload(nxt)),
            reward=float(payload["reward"]),
            reward_components={k: float(v)
                               for k, v in payload["reward_components"].items()},
            terminated=bool(payload["terminated"]),
            bootstrap=float(payload["bootstrap"]),
            termination_reason=str(payload["termination_reason"]),
            td_eligible=bool(payload["td_eligible"]),
            logical_cost=int(payload["logical_cost"]),
            qc_cost_evidence=payload.get("qc_cost_evidence"),
            environment_version=str(payload.get("environment_version", "")),
            advancer_version=str(payload.get("advancer_version", "")),
            reward_spec_sha256=str(payload.get("reward_spec_sha256", "")),
            parent_node_hash=str(payload.get("parent_node_hash", "")),
        )
        transition.validate()
        return transition


__all__ = [
    "ACTION_SCHEMA_VERSION",
    "DEFAULT_REWARD_SPEC",
    "DIRECT_MDP_SCHEMA_VERSION",
    "DirectAction",
    "DirectContractError",
    "DirectRewardSpec",
    "DirectState",
    "DirectTransition",
    "FORBIDDEN_STATE_FIELDS",
    "REJECTION_REASONS",
    "REWARD_SPEC_SCHEMA_VERSION",
    "STATE_SCHEMA_VERSION",
    "TERMINATION_REASONS",
    "TRANSITION_SCHEMA_VERSION",
]
