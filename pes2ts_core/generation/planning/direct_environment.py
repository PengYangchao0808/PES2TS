"""Online direct environment shell (WP5a, plan v3 §3).

``DirectEnvironment`` owns episode state and delegates every semantic decision
to the shared advancer (:mod:`direct_mdp`). QC work is submitted through an
``ObservationSource`` protocol; the fake source in this module is a test
double only — production observation comes from ACP via
``integration/acp/direct_observation.py`` (R3).

Censoring (timeouts, cancellation, scheduling faults) surfaces as
``censored_operational_limit`` transitions that carry a real successor when
one was obtained; no artificial failure reward is added (plan §3.1).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Final, Protocol

import numpy as np

from pes2ts_core.generation.planning.direct_mdp import (
    ADVANCER_VERSION,
    DirectAdvanceError,
    DirectObservation,
    advance_direct_state,
    goal_potential,
)
from pes2ts_core.pesn.contracts.direct import (
    DEFAULT_REWARD_SPEC,
    DirectAction,
    DirectRewardSpec,
    DirectState,
    DirectTransition,
)

ENVIRONMENT_VERSION: Final[str] = "E_direct_xtb_v1.0"
SNAPSHOT_SCHEMA: Final[str] = "pes2ts_direct_env_snapshot_v1"


class ObservationSource(Protocol):
    """One unconstrained in-place E/g evaluation (ACP in production)."""

    def __call__(
        self,
        coordinates: np.ndarray,
        atomic_numbers: np.ndarray,
        *,
        charge: int,
        multiplicity: int,
        evaluation_id: str,
    ) -> DirectObservation: ...


@dataclass(frozen=True, slots=True)
class FakeObservationSource:
    """Deterministic test double: harmonic E/g around a target geometry.

    Energy in hartree-scale arbitrary units; gradient points toward the
    target. Cache semantics mirror the real source via ``cache_reused``.
    """

    target_coordinates: np.ndarray
    force_constant: float = 0.05
    status_for_step: Callable[[int], str] | None = None

    def __call__(self, coordinates, atomic_numbers, *, charge, multiplicity,
                 evaluation_id) -> DirectObservation:
        step = _step_from_evaluation_id(evaluation_id)
        if self.status_for_step is not None:
            status = self.status_for_step(step)
            if status != "completed":
                return DirectObservation(status=status, error=f"fake-{status}")
        delta = np.asarray(coordinates, dtype=float) - np.asarray(self.target_coordinates, dtype=float)
        energy = float(self.force_constant * (delta * delta).sum())
        gradient = 2.0 * self.force_constant * delta
        return DirectObservation(
            status="completed", energy_hartree=energy,
            gradient_hartree_per_angstrom=gradient,
            request_sha256=f"fake-{evaluation_id}", wall_seconds=0.0,
        )


def _step_from_evaluation_id(evaluation_id: str) -> int:
    tail = evaluation_id.rsplit("s", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return -1


class DirectEnvironment:
    """Episode holder for one direct-displacement task."""

    def __init__(
        self,
        *,
        observation_source: ObservationSource,
        atomic_numbers: np.ndarray,
        task_graph: dict,
        atom_map_ids: tuple[str, ...],
        charge: int = 0,
        multiplicity: int = 1,
        method_id: str = "GFN2-xTB",
        reward_spec: DirectRewardSpec = DEFAULT_REWARD_SPEC,
        episode_id: str,
        reaction_id: str,
        group_id: str,
        subset: str,
        origin: str = "deployment_origin",
    ) -> None:
        if origin not in ("teacher_origin", "deployment_origin"):
            raise DirectAdvanceError(f"unknown origin {origin!r}")
        self._source = observation_source
        self._reward_spec = reward_spec
        self._episode_id = episode_id
        self._reaction_id = reaction_id
        self._group_id = group_id
        self._subset = subset
        self._origin = origin
        self._base = dict(
            atomic_numbers=np.asarray(atomic_numbers).astype(int),
            task_graph=task_graph,
            atom_map_ids=tuple(atom_map_ids),
            charge=int(charge),
            multiplicity=int(multiplicity),
            method_id=method_id,
        )
        self._state: DirectState | None = None
        self._step_index = 0
        self._done = False

    # -- lifecycle ---------------------------------------------------------

    @property
    def state(self) -> DirectState:
        if self._state is None:
            raise DirectAdvanceError("environment not reset")
        return self._state

    @property
    def done(self) -> bool:
        return self._done

    def reset(self, initial_coordinates: np.ndarray) -> DirectState:
        """Evaluate the reset geometry and establish the energy reference."""
        coords = np.asarray(initial_coordinates, dtype=float).copy()
        observation = self._source(
            coords, self._base["atomic_numbers"],
            charge=self._base["charge"], multiplicity=self._base["multiplicity"],
            evaluation_id=f"{self._episode_id}-reset",
        )
        if observation.status != "completed":
            raise DirectAdvanceError(
                f"reset observation failed ({observation.status}); episode cannot start"
            )
        phi0 = goal_potential(coords, self._base["atomic_numbers"], self._base["task_graph"])
        self._state = DirectState(
            atomic_numbers=self._base["atomic_numbers"].copy(),
            coordinates=coords,
            task_graph=self._base["task_graph"],
            atom_map_ids=self._base["atom_map_ids"],
            charge=self._base["charge"],
            multiplicity=self._base["multiplicity"],
            method_id=self._base["method_id"],
            energy_hartree=float(observation.energy_hartree),
            energy_reference_hartree=float(observation.energy_hartree),
            gradient_hartree_per_angstrom=(
                None if observation.gradient_hartree_per_angstrom is None
                else np.asarray(observation.gradient_hartree_per_angstrom, dtype=float)
            ),
            observation_mask=True,
            steps_remaining=self._reward_spec.budget_steps,
            event_state={"goal_consecutive_hits": 0.0, "no_progress_count": 0.0,
                         "best_potential": phi0},
        )
        self._state.validate()
        self._step_index = 0
        self._done = False
        return self._state

    def step(self, action: DirectAction) -> DirectTransition:
        """One logical action; fixed cost 1, single E/g per legal node."""
        state = self.state
        if self._done:
            raise DirectAdvanceError("step() called on finished episode")
        transition_id = f"{self._episode_id}-t{self._step_index:04d}"
        evaluation_id = f"{self._episode_id}-s{self._step_index:04d}"

        rejection = None
        from pes2ts_core.generation.planning.direct_mdp import classify_pre_execution
        rejection = classify_pre_execution(state, action)

        observation: DirectObservation | None = None
        t0 = time.perf_counter()
        if rejection is None:
            proposed = np.asarray(action.proposed_angstrom, dtype=float)
            next_coords = np.asarray(state.coordinates, dtype=float) + proposed
            observation = self._source(
                next_coords, self._base["atomic_numbers"],
                charge=self._base["charge"], multiplicity=self._base["multiplicity"],
                evaluation_id=evaluation_id,
            )
            if observation.wall_seconds is None:
                object.__setattr__(observation, "wall_seconds", time.perf_counter() - t0)

        transition = advance_direct_state(
            state, action, observation,
            reward_spec=self._reward_spec,
            episode_id=self._episode_id,
            step_index=self._step_index,
            reaction_id=self._reaction_id,
            group_id=self._group_id,
            subset=self._subset,
            transition_id=transition_id,
            parent_node_hash=state.content_hash(),
        )
        transition.validate()
        self._step_index += 1
        if transition.terminated:
            self._done = True
        elif transition.next_state is not None and transition.td_eligible:
            self._state = transition.next_state
        elif transition.next_state is not None:
            # censored successor: keep the observed state, caller decides to resume
            self._state = transition.next_state
        if transition.action.rejection_reason == "nonfinite_latent":
            # model numeric fault: stop this collection version loudly (plan §3.3)
            self._done = True
        return transition

    # -- recovery ----------------------------------------------------------

    def snapshot(self) -> dict:
        """Recovery snapshot (episode identity + state + counters)."""
        state = self.state
        return {
            "schema_version": SNAPSHOT_SCHEMA,
            "environment_version": ENVIRONMENT_VERSION,
            "advancer_version": ADVANCER_VERSION,
            "episode_id": self._episode_id,
            "reaction_id": self._reaction_id,
            "group_id": self._group_id,
            "subset": self._subset,
            "origin": self._origin,
            "step_index": self._step_index,
            "done": self._done,
            "state": state.to_payload(),
        }

    def restore(self, snapshot: dict, *, observation_source: ObservationSource) -> None:
        if snapshot.get("schema_version") != SNAPSHOT_SCHEMA:
            raise DirectAdvanceError("snapshot schema mismatch")
        if snapshot.get("episode_id") != self._episode_id:
            raise DirectAdvanceError("snapshot belongs to another episode")
        payload = snapshot["state"]
        self._state = DirectState(
            atomic_numbers=np.asarray(payload["atomic_numbers"], dtype=int),
            coordinates=np.asarray(payload["coordinates"], dtype=float),
            task_graph=payload["task_graph"],
            atom_map_ids=tuple(payload["atom_map_ids"]),
            charge=int(payload["charge"]),
            multiplicity=int(payload["multiplicity"]),
            method_id=str(payload["method_id"]),
            energy_hartree=payload["energy_hartree"],
            energy_reference_hartree=payload["energy_reference_hartree"],
            gradient_hartree_per_angstrom=(
                None if payload.get("gradient") is None
                else np.asarray(payload["gradient"], dtype=float)
            ),
            observation_mask=bool(payload["observation_mask"]),
            steps_remaining=int(payload["steps_remaining"]),
            event_state={k: float(v) for k, v in payload["event_state"].items()},
        )
        self._state.validate()
        self._step_index = int(snapshot["step_index"])
        self._done = bool(snapshot["done"])
        self._source = observation_source


__all__ = [
    "ENVIRONMENT_VERSION",
    "DirectEnvironment",
    "FakeObservationSource",
    "ObservationSource",
    "SNAPSHOT_SCHEMA",
]
