"""Direct path candidate tests (WP7a): identity, budget, bundle adaptation."""
from __future__ import annotations

import numpy as np
import pytest

from pes2ts_core.generation.planning.direct_candidates import (
    DirectCandidateError,
    build_direct_path_bundle,
    candidate_from_transitions,
)
from pes2ts_core.pesn.contracts.direct import (
    DirectAction,
    DirectState,
    DirectTransition,
)
from pes2ts_core.generation.planning.direct_mdp import (
    DirectObservation,
    advance_direct_state,
)

TASK_GRAPH = {"edits": [{"kind": "broken", "i": 0, "j": 1},
                        {"kind": "formed", "i": 1, "j": 2}]}
Z = np.array([6, 1, 6])


def _state(hx: float, steps: int) -> DirectState:
    from pes2ts_core.generation.planning.direct_mdp import goal_potential
    coords = np.array([[0.0, 0, 0], [hx, 0, 0], [2.5, 0, 0]])
    return DirectState(
        atomic_numbers=Z, coordinates=coords, task_graph=TASK_GRAPH,
        atom_map_ids=("a0", "a1", "a2"), energy_hartree=-10.0,
        energy_reference_hartree=-10.0,
        gradient_hartree_per_angstrom=np.zeros((3, 3)),
        observation_mask=True, steps_remaining=steps,
        event_state={"goal_consecutive_hits": 0.0, "no_progress_count": 0.0,
                     "best_potential": goal_potential(coords, Z, TASK_GRAPH)},
    )


def _action(hx_next: float, hx: float) -> DirectAction:
    prop = np.zeros((3, 3))
    prop[1, 0] = hx_next - hx
    return DirectAction(action_max_angstrom=0.1, latent_u=None, normalized_z=None,
                        proposed_angstrom=prop, executed=True,
                        rejection_reason=None, log_prob_z=None,
                        log_prob_source="policy")


def _transitions(moves) -> list[DirectTransition]:
    state = _state(1.1, 100)
    out = []
    hx = 1.1
    obs = DirectObservation(status="completed", energy_hartree=-10.0,
                            gradient_hartree_per_angstrom=np.zeros((3, 3)))
    for t, target in enumerate(moves):
        tr = advance_direct_state(state, _action(target, hx), obs,
                                  episode_id="ep-c", step_index=t,
                                  reaction_id="RXN_C", group_id="G_C",
                                  subset="direct_monitor8",
                                  transition_id=f"t{t}")
        out.append(tr)
        if tr.terminated or tr.next_state is None:
            break
        state = tr.next_state
        hx = target
    return out


def test_candidate_from_goal_rollout():
    transitions = _transitions([1.2, 1.3, 1.4])
    candidate = candidate_from_transitions(
        transitions, reset_coordinates=_state(1.1, 100).coordinates,
        reset_energy_hartree=-10.0, policy_version="bc-v0")
    assert candidate.reached_goal_proxy
    assert candidate.termination_reason == "goal_connected_proxy"
    assert candidate.logical_steps_used == 3
    assert len(candidate.frames_hash) == len(candidate.frames_geometry)
    # all frame hashes distinct along a moving path
    assert len(set(candidate.frames_hash)) == len(candidate.frames_hash)


def test_candidate_rejects_goal_flag_mismatch():
    transitions = _transitions([1.2, 1.3, 1.4])
    candidate = candidate_from_transitions(
        transitions, reset_coordinates=_state(1.1, 100).coordinates,
        reset_energy_hartree=-10.0)
    broken = type(candidate)(
        reaction_id=candidate.reaction_id, group_id=candidate.group_id,
        episode_id=candidate.episode_id, subset=candidate.subset,
        frames_geometry=candidate.frames_geometry,
        frames_energy_hartree=candidate.frames_energy_hartree,
        frames_hash=candidate.frames_hash,
        termination_reason="budget_exhausted",
        reached_goal_proxy=True,
        logical_steps_used=candidate.logical_steps_used,
    )
    with pytest.raises(DirectCandidateError):
        broken.validate()


def test_empty_rollout_returns_none():
    assert candidate_from_transitions(
        [], reset_coordinates=np.zeros((3, 3)), reset_energy_hartree=0.0) is None


def test_bundle_adaptation_structure():
    transitions = _transitions([1.2, 1.3, 1.4])
    candidate = candidate_from_transitions(
        transitions, reset_coordinates=_state(1.1, 100).coordinates,
        reset_energy_hartree=-10.0, policy_version="bc-v0")
    bundle = build_direct_path_bundle(
        candidate, elements=["C", "H", "C"], atom_map_ids=["a0", "a1", "a2"])
    assert bundle["n_frames"] == len(bundle["frames"]) >= 2
    frame = bundle["frames"][0]
    assert frame["frame_index"] == 0
    assert frame["energies"]["xtp"]["unit"] == "hartree"
    assert len(frame["geometry"]) == 3
    assert "not a validated transition state" in bundle["note"]
