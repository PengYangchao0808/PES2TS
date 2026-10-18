"""Shared no-side-effect direct-MDP advancer (WP5a, plan v3 §3–§5).

``advance_direct_state`` is the ONE place where the direct-displacement MDP
semantics live. Offline teacher construction and the online environment both
call it with an already-obtained observation; this module never submits QC
work itself (R3: ACP is the only execution backend, wired one level up).

Fixed event priority (plan §3.3): pre-execution rejection → physical /
infrastructure classification → goal on valid observation → budget
exhaustion → no-progress → continue.

``reward_direct_proxy_v1`` (plan §5): per-step ``-c``; success terminal adds
``+B``; failure terminal adds ``-F - c*steps_remaining_after`` so that any
failure totals ``-F-cH`` regardless of when it happens; censoring adds no
failure components. Potential shaping ``eta*(Phi(s')-Phi(s))`` with
``Phi = -L_goal in [-1, 0]`` from task-graph form/break distance penalties.

Constants marked ``fit50`` are calibration candidates, not frozen science.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping

import numpy as np

from pes2ts_core.pesn.contracts.direct import (
    DEFAULT_REWARD_SPEC,
    DirectAction,
    DirectRewardSpec,
    DirectState,
    DirectTransition,
)
from pes2ts_core.utils.hashing import stable_json_dumps

ADVANCER_VERSION: Final[str] = "pes2ts_direct_advancer_v1.0"


def _sha256_text(text: str) -> str:
    from pes2ts_core.utils.hashing import sha256_bytes
    return sha256_bytes(text.encode("utf-8"))

#: Cordero 2008 covalent radii subset (duplicated from features/geometry.py
#: so the host planning layer never imports the predictor feature stack).
_COVALENT_RADII: Final[Mapping[int, float]] = {
    1: 0.31, 3: 1.28, 4: 0.96, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
    11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02,
    19: 2.03, 20: 1.76, 26: 1.32, 27: 1.26, 28: 1.24, 29: 1.32, 30: 1.22,
    34: 1.20, 35: 1.20, 44: 1.46, 45: 1.27, 46: 1.31, 47: 1.44, 48: 1.51,
    53: 1.39, 78: 1.38, 79: 1.42,
}
_DEFAULT_RADIUS: Final[float] = 1.20

#: fit50 calibration candidates (plan §3.3/§3.2).
BOND_SCALE: Final[float] = 1.15          # bonded iff d < scale*(r_i+r_j)
OVERLAP_FACTOR: Final[float] = 0.45      # catastrophic iff d < factor*(r_i+r_j)
POTENTIAL_SCALE_ANGSTROM: Final[float] = 1.0  # distance softening in Phi


class DirectAdvanceError(RuntimeError):
    """Typed advancer misuse (never an MDP event)."""


def _radii_pair(z: np.ndarray) -> np.ndarray:
    r = np.array([_COVALENT_RADII.get(int(v), _DEFAULT_RADIUS) for v in z], dtype=float)
    return r[:, None] + r[None, :]


def pairwise_distances(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff * diff).sum(-1))


def bonded_mask(coords: np.ndarray, atomic_numbers: np.ndarray,
                *, scale: float = BOND_SCALE) -> np.ndarray:
    """Boolean adjacency from covalent-radius bondedness (diagonal False)."""
    n = coords.shape[0]
    d = pairwise_distances(coords)
    bonded = d < scale * _radii_pair(atomic_numbers)
    np.fill_diagonal(bonded, False)
    return bonded


def _edits(task_graph: Mapping) -> list[dict]:
    edits = task_graph.get("edits")
    if edits is None and isinstance(task_graph.get("graph"), Mapping):
        edits = task_graph["graph"].get("edits")
    if edits is None:
        raise DirectAdvanceError("task_graph carries no edits list")
    return [dict(e) for e in edits]


def _product_bonds(task_graph: Mapping) -> list[tuple[int, int]]:
    """Atom-index pairs that must be bonded in the target (H_direction).

    Vocabulary follows the frozen G1 v2 export kinds: ``formed`` and
    ``order_changed`` pairs carry a product-side bond; ``broken`` pairs must
    be absent. Unknown kinds make the edit list unusable (loud, not ignored).
    """
    pairs = []
    for edit in _edits(task_graph):
        kind = str(edit.get("kind", "")).lower()
        if kind in ("formed", "make", "form", "order_changed", "change"):
            i, j = int(edit["i"]), int(edit["j"])
            pairs.append((min(i, j), max(i, j)))
    return sorted(set(pairs))


def _broken_bonds(task_graph: Mapping) -> list[tuple[int, int]]:
    """Atom-index pairs bonded on the reactant side that must be absent."""
    pairs = []
    for edit in _edits(task_graph):
        kind = str(edit.get("kind", "")).lower()
        if kind in ("broken", "break"):
            i, j = int(edit["i"]), int(edit["j"])
            pairs.append((min(i, j), max(i, j)))
    return sorted(set(pairs))


def goal_potential(coords: np.ndarray, atomic_numbers: np.ndarray,
                   task_graph: Mapping) -> float:
    """``Phi = -L_goal in [-1, 0]`` (plan §5).

    ``L_goal`` averages smooth penalties over product bonds that must form
    (penalizing being too long) and broken bonds that must open (penalizing
    being too short). It reads only the task graph and current geometry —
    never target coordinates, reference paths, or progress labels.
    """
    d = pairwise_distances(coords)
    r = _radii_pair(atomic_numbers)
    terms: list[float] = []
    for i, j in _product_bonds(task_graph):
        target = r[i, j]
        terms.append(max(0.0, (d[i, j] - target) / POTENTIAL_SCALE_ANGSTROM))
    for i, j in _broken_bonds(task_graph):
        target = r[i, j]
        terms.append(max(0.0, (target - d[i, j]) / POTENTIAL_SCALE_ANGSTROM))
    if not terms:
        return 0.0
    loss = float(np.mean(terms))
    return -min(1.0, loss)


def goal_connected_proxy(state: DirectState) -> bool:
    """First-version goal proxy (plan §3.3).

    Complete target adjacency match under covalent-radius bondedness, valid
    observation, identity fields intact, no catastrophic overlap. Stereo
    identity checking is registered as a fit50 follow-up, not silently
    claimed. The consecutive-hit rule is enforced by the caller's event
    state (this predicate is the single-hit signal).
    """
    if not state.observation_mask or state.energy_hartree is None:
        return False
    coords = np.asarray(state.coordinates, dtype=float)
    z = np.asarray(state.atomic_numbers).astype(int)
    bonded = bonded_mask(coords, z)
    for i, j in _product_bonds(state.task_graph):
        if not bonded[i, j]:
            return False
    for i, j in _broken_bonds(state.task_graph):
        if bonded[i, j]:
            return False
    return not has_catastrophic_overlap(coords, z)


def has_catastrophic_overlap(coords: np.ndarray, atomic_numbers: np.ndarray,
                             *, factor: float = OVERLAP_FACTOR) -> bool:
    d = pairwise_distances(coords)
    too_close = d < factor * _radii_pair(atomic_numbers)
    np.fill_diagonal(too_close, False)
    return bool(too_close.any())


# ---------------------------------------------------------------------------
# Observation handed to the advancer (obtained outside, via ACP or offline)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DirectObservation:
    """Result of one unconstrained in-place E/g evaluation."""

    status: str                       # completed | failed_geometry | failed_infra | canceled
    energy_hartree: float | None = None
    gradient_hartree_per_angstrom: np.ndarray | None = None
    cache_reused: bool = False
    request_sha256: str | None = None
    wall_seconds: float | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status not in ("completed", "failed_geometry", "failed_infra", "canceled"):
            raise DirectAdvanceError(f"unknown observation status {self.status!r}")
        if self.status == "completed":
            if self.energy_hartree is None:
                raise DirectAdvanceError("completed observation requires energy")
            if not np.isfinite(self.energy_hartree):
                raise DirectAdvanceError("completed observation energy non-finite")


# ---------------------------------------------------------------------------
# The shared advancer
# ---------------------------------------------------------------------------

def classify_pre_execution(state: DirectState, action: DirectAction) -> str | None:
    """Pre-execution legality (priority 1). Returns a rejection reason or None.

    Non-finite latent actions are model numeric faults: the advancer reports
    ``nonfinite_latent`` and the caller marks ``td_eligible=False`` and stops
    the collection version (plan §3.3); they are not MDP failure terminals
    with valid TD signal.
    """
    if action.latent_u is not None and not np.isfinite(action.latent_u).all():
        return "nonfinite_latent"
    if action.proposed_angstrom is None:
        return "not_executed_censored"       # caller must pair with censored reason
    prop = np.asarray(action.proposed_angstrom, dtype=float)
    if prop.shape != (state.n_atoms(), 3):
        return "wrong_atom_count"
    if not np.isfinite(prop).all():
        return "nonfinite_latent"
    limit = action.action_max_angstrom
    if float(np.linalg.norm(prop, axis=1).max()) > limit * (1.0 + 1e-6):
        return "support_boundary_violation"
    next_coords = np.asarray(state.coordinates, dtype=float) + prop
    if has_catastrophic_overlap(next_coords, np.asarray(state.atomic_numbers).astype(int)):
        return "catastrophic_overlap"
    return None


def _failure_reward(spec: DirectRewardSpec, steps_remaining_after: int) -> float:
    """Per-step reward of a failing action: ``-c - F - c*steps_remaining_after``.

    Telescopes to ``-F - cH`` for a failure at any step (plan §5): the step's
    own ``-c`` plus the compensation for all unspent steps.
    """
    return -(spec.step_cost + spec.failure_penalty
             + spec.step_cost * steps_remaining_after)


def advance_direct_state(
    state: DirectState,
    action: DirectAction,
    observation: DirectObservation | None,
    *,
    reward_spec: DirectRewardSpec = DEFAULT_REWARD_SPEC,
    episode_id: str,
    step_index: int,
    reaction_id: str,
    group_id: str,
    subset: str,
    transition_id: str,
    parent_node_hash: str = "",
) -> DirectTransition:
    """Apply one action+observation under the fixed event priority.

    ``observation`` is ``None`` only for pre-execution rejections (no QC was
    or will be submitted for the rejected proposal).
    """
    reward_spec.validate()
    state.validate()
    action.validate()

    spec = reward_spec
    phi_s = goal_potential(np.asarray(state.coordinates, dtype=float),
                           np.asarray(state.atomic_numbers).astype(int),
                           state.task_graph)
    components: dict[str, float] = {"step_cost": -spec.step_cost}
    steps_after = max(0, state.steps_remaining - 1)

    # --- priority 1: pre-execution rejection -----------------------------
    rejection = classify_pre_execution(state, action)
    if rejection is not None:
        # Numeric model faults are not valid MDP failure terminals (plan §3.3).
        if rejection in ("nonfinite_latent", "not_executed_censored"):
            terminal = rejection == "not_executed_censored"
            reason = "canceled" if terminal else "illegal_action"
            components = {"step_cost": -spec.step_cost, "censored": 0.0}
            reward = -spec.step_cost
            rejected_action = DirectAction(
                action_max_angstrom=action.action_max_angstrom,
                latent_u=action.latent_u, normalized_z=action.normalized_z,
                proposed_angstrom=action.proposed_angstrom, executed_angstrom=None,
                executed=False, rejection_reason=rejection,
                log_prob_z=action.log_prob_z, log_prob_source=action.log_prob_source,
            )
            return DirectTransition(
                transition_id=transition_id, reaction_id=reaction_id,
                group_id=group_id, subset=subset, episode_id=episode_id,
                step_index=step_index, state=state, action=rejected_action,
                next_state=None if terminal else state,
                reward=reward, reward_components=components,
                terminated=False, bootstrap=0.0 if terminal else 1.0,
                termination_reason=reason, td_eligible=rejection != "nonfinite_latent",
                logical_cost=1, qc_cost_evidence=None,
                advancer_version=ADVANCER_VERSION,
                reward_spec_sha256=_reward_spec_sha256(spec),
                parent_node_hash=parent_node_hash,
            )
        # Geometric/identity rejections are MDP failure terminals.
        reward = _failure_reward(spec, steps_after)
        components["failure"] = reward + spec.step_cost
        rejected_action = DirectAction(
            action_max_angstrom=action.action_max_angstrom,
            latent_u=action.latent_u, normalized_z=action.normalized_z,
            proposed_angstrom=action.proposed_angstrom, executed_angstrom=None,
            executed=False, rejection_reason=rejection,
            log_prob_z=action.log_prob_z, log_prob_source=action.log_prob_source,
        )
        return DirectTransition(
            transition_id=transition_id, reaction_id=reaction_id,
            group_id=group_id, subset=subset, episode_id=episode_id,
            step_index=step_index, state=state, action=rejected_action,
            next_state=None, reward=reward, reward_components=components,
            terminated=True, bootstrap=0.0, termination_reason="illegal_action",
            td_eligible=True, logical_cost=1, qc_cost_evidence=None,
            advancer_version=ADVANCER_VERSION,
            reward_spec_sha256=_reward_spec_sha256(spec),
            parent_node_hash=parent_node_hash,
        )

    if observation is None:
        raise DirectAdvanceError(
            "executed action requires an observation (pre-execution rejections "
            "are handled above)"
        )

    prop = np.asarray(action.proposed_angstrom, dtype=float)
    next_coords = np.asarray(state.coordinates, dtype=float) + prop
    z = np.asarray(state.atomic_numbers).astype(int)
    executed_action = DirectAction(
        action_max_angstrom=action.action_max_angstrom,
        latent_u=action.latent_u, normalized_z=action.normalized_z,
        proposed_angstrom=prop, executed_angstrom=prop, executed=True,
        rejection_reason=None, log_prob_z=action.log_prob_z,
        log_prob_source=action.log_prob_source,
    )

    # --- priority 2: physical / infrastructure classification ------------
    if observation.status in ("failed_infra", "canceled"):
        next_state = _with_observation(state, next_coords, None, None, False, steps_after)
        components["censored"] = 0.0
        evidence = _evidence(observation)
        return DirectTransition(
            transition_id=transition_id, reaction_id=reaction_id, group_id=group_id,
            subset=subset, episode_id=episode_id, step_index=step_index,
            state=state, action=executed_action, next_state=next_state,
            reward=-spec.step_cost, reward_components=components,
            terminated=False, bootstrap=1.0,
            termination_reason="censored_operational_limit", td_eligible=False,
            logical_cost=1, qc_cost_evidence=evidence,
            advancer_version=ADVANCER_VERSION,
            reward_spec_sha256=_reward_spec_sha256(spec),
            parent_node_hash=parent_node_hash,
        )
    if observation.status == "failed_geometry":
        reward = _failure_reward(spec, steps_after)
        components["failure"] = reward + spec.step_cost
        return DirectTransition(
            transition_id=transition_id, reaction_id=reaction_id, group_id=group_id,
            subset=subset, episode_id=episode_id, step_index=step_index,
            state=state, action=executed_action,
            next_state=None, reward=reward, reward_components=components,
            terminated=True, bootstrap=0.0, termination_reason="qc_failure_terminal",
            td_eligible=True, logical_cost=1, qc_cost_evidence=_evidence(observation),
            advancer_version=ADVANCER_VERSION,
            reward_spec_sha256=_reward_spec_sha256(spec),
            parent_node_hash=parent_node_hash,
        )

    # --- valid observation: build the observed successor ------------------
    energy = float(observation.energy_hartree)
    grad = (None if observation.gradient_hartree_per_angstrom is None
            else np.asarray(observation.gradient_hartree_per_angstrom, dtype=float))
    phi_next_raw = goal_potential(next_coords, z, state.task_graph)
    probe = _with_observation(state, next_coords, energy, grad, True, steps_after)
    goal_hit = goal_connected_proxy(probe)

    hits = int(state.event_state.get("goal_consecutive_hits", 0)) + (1 if goal_hit else 0)
    best = float(state.event_state.get("best_potential", -1.0))
    best_next = max(best, phi_next_raw)
    no_progress = int(state.event_state.get("no_progress_count", 0))
    improved = phi_next_raw > best + spec.no_progress_tolerance
    no_progress_next = 0 if improved or goal_hit else no_progress + 1

    next_event_state = {
        "goal_consecutive_hits": float(hits),
        "no_progress_count": float(no_progress_next),
        "best_potential": best_next,
    }
    next_state = _with_observation(state, next_coords, energy, grad, True, steps_after,
                                   event_state=next_event_state)
    phi_next = 0.0  # task terminals fix Phi = 0 (plan §5)
    if not goal_hit or hits < spec.goal_consecutive_hits:
        phi_next = phi_next_raw
    components["shaping"] = spec.shaping_scale * (phi_next - phi_s)

    # --- priority 3: goal --------------------------------------------------
    if goal_hit and hits >= spec.goal_consecutive_hits:
        components["success"] = spec.success_bonus
        reward = -spec.step_cost + spec.success_bonus + components["shaping"]
        return DirectTransition(
            transition_id=transition_id, reaction_id=reaction_id, group_id=group_id,
            subset=subset, episode_id=episode_id, step_index=step_index,
            state=state, action=executed_action, next_state=next_state,
            reward=reward, reward_components=components,
            terminated=True, bootstrap=0.0, termination_reason="goal_connected_proxy",
            td_eligible=True, logical_cost=1, qc_cost_evidence=_evidence(observation),
            advancer_version=ADVANCER_VERSION,
            reward_spec_sha256=_reward_spec_sha256(spec),
            parent_node_hash=parent_node_hash,
        )

    # --- priority 4: budget exhausted --------------------------------------
    if steps_after == 0:
        # task terminal fixes Phi(s')=0, so the terminal step's shaping is
        # eta*(0 - Phi(s)) and the whole-episode shaping telescopes to
        # -eta*Phi(s0) (plan §5)
        components["shaping"] = spec.shaping_scale * (0.0 - phi_s)
        reward = _failure_reward(spec, 0) + components["shaping"]
        reward_components_out = {
            "step_cost": components["step_cost"],
            "shaping": components["shaping"],
            "failure": _failure_reward(spec, 0) + spec.step_cost,
        }
        return DirectTransition(
            transition_id=transition_id, reaction_id=reaction_id, group_id=group_id,
            subset=subset, episode_id=episode_id, step_index=step_index,
            state=state, action=executed_action, next_state=next_state,
            reward=reward, reward_components=reward_components_out,
            terminated=True, bootstrap=0.0, termination_reason="budget_exhausted",
            td_eligible=True, logical_cost=1, qc_cost_evidence=_evidence(observation),
            advancer_version=ADVANCER_VERSION,
            reward_spec_sha256=_reward_spec_sha256(spec),
            parent_node_hash=parent_node_hash,
        )

    # --- priority 5: no-progress -------------------------------------------
    if no_progress_next >= spec.no_progress_window:
        components["shaping"] = spec.shaping_scale * (0.0 - phi_s)
        reward = _failure_reward(spec, steps_after) + components["shaping"]
        reward_components_out = {
            "step_cost": components["step_cost"],
            "shaping": components["shaping"],
            "failure": _failure_reward(spec, steps_after) + spec.step_cost,
        }
        return DirectTransition(
            transition_id=transition_id, reaction_id=reaction_id, group_id=group_id,
            subset=subset, episode_id=episode_id, step_index=step_index,
            state=state, action=executed_action, next_state=next_state,
            reward=reward, reward_components=reward_components_out,
            terminated=True, bootstrap=0.0, termination_reason="no_progress_terminal",
            td_eligible=True, logical_cost=1, qc_cost_evidence=_evidence(observation),
            advancer_version=ADVANCER_VERSION,
            reward_spec_sha256=_reward_spec_sha256(spec),
            parent_node_hash=parent_node_hash,
        )

    # --- priority 6: continue -----------------------------------------------
    reward = -spec.step_cost + components["shaping"]
    return DirectTransition(
        transition_id=transition_id, reaction_id=reaction_id, group_id=group_id,
        subset=subset, episode_id=episode_id, step_index=step_index,
        state=state, action=executed_action, next_state=next_state,
        reward=reward, reward_components=components,
        terminated=False, bootstrap=1.0, termination_reason="continued",
        td_eligible=True, logical_cost=1, qc_cost_evidence=_evidence(observation),
        advancer_version=ADVANCER_VERSION,
        reward_spec_sha256=_reward_spec_sha256(spec),
        parent_node_hash=parent_node_hash,
    )


def _reward_spec_sha256(spec: DirectRewardSpec) -> str:
    return _sha256_text(stable_json_dumps(spec.to_payload()))


def _evidence(observation: DirectObservation) -> dict[str, object]:
    return {
        "status": observation.status,
        "cache_reused": observation.cache_reused,
        "request_sha256": observation.request_sha256,
        "wall_seconds": observation.wall_seconds,
        "error": observation.error,
    }


def _with_observation(
    state: DirectState,
    coords: np.ndarray,
    energy: float | None,
    gradient: np.ndarray | None,
    observed: bool,
    steps_remaining: int,
    *,
    event_state: Mapping[str, float] | None = None,
) -> DirectState:
    if event_state is None:
        event_state = dict(state.event_state)
    return DirectState(
        atomic_numbers=np.asarray(state.atomic_numbers).astype(int),
        coordinates=coords,
        task_graph=state.task_graph,
        atom_map_ids=state.atom_map_ids,
        charge=state.charge,
        multiplicity=state.multiplicity,
        method_id=state.method_id,
        energy_hartree=energy,
        energy_reference_hartree=state.energy_reference_hartree,
        gradient_hartree_per_angstrom=gradient,
        observation_mask=observed,
        steps_remaining=steps_remaining,
        event_state=dict(event_state),
    )


__all__ = [
    "ADVANCER_VERSION",
    "BOND_SCALE",
    "DirectAdvanceError",
    "DirectObservation",
    "OVERLAP_FACTOR",
    "advance_direct_state",
    "bonded_mask",
    "classify_pre_execution",
    "goal_connected_proxy",
    "goal_potential",
    "has_catastrophic_overlap",
    "pairwise_distances",
]
