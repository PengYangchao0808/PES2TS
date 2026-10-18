"""Direct environment shell tests (fake observation source; no ACP, no truth)."""
from __future__ import annotations

import numpy as np
import pytest

from pes2ts_core.generation.planning.direct_environment import (
    DirectEnvironment,
    FakeObservationSource,
)
from pes2ts_core.pesn.contracts.direct import (
    DEFAULT_REWARD_SPEC,
    DirectAction,
    DirectRewardSpec,
)

TASK_GRAPH = {
    "edits": [
        {"kind": "broken", "i": 0, "j": 1, "reactant_order": 1, "product_order": 0},
        {"kind": "formed", "i": 1, "j": 2, "reactant_order": 0, "product_order": 1},
    ]
}
Z = np.array([6, 1, 6])
START = np.array([[0.0, 0, 0], [1.1, 0, 0], [2.5, 0, 0]])
TARGET = np.array([[0.0, 0, 0], [1.5, 0, 0], [2.5, 0, 0]])


def make_env(source=None, spec=DEFAULT_REWARD_SPEC) -> DirectEnvironment:
    return DirectEnvironment(
        observation_source=source or FakeObservationSource(target_coordinates=TARGET),
        atomic_numbers=Z, task_graph=TASK_GRAPH, atom_map_ids=("a0", "a1", "a2"),
        reward_spec=spec, episode_id="ep-1", reaction_id="RXN_TEST",
        group_id="G_TEST", subset="direct_fit50", origin="deployment_origin",
    )


def step_action(h_x_next: float, h_x: float, *, a_max: float = 0.1) -> DirectAction:
    prop = np.zeros((3, 3))
    prop[1, 0] = h_x_next - h_x
    return DirectAction(action_max_angstrom=a_max, latent_u=prop / a_max,
                        normalized_z=prop / a_max, proposed_angstrom=prop,
                        executed=True, rejection_reason=None, log_prob_z=-3.0)


def test_reset_establishes_energy_reference():
    env = make_env()
    state = env.reset(START)
    assert state.energy_hartree == state.energy_reference_hartree
    assert state.delta_e_hartree() == 0.0
    assert state.steps_remaining == DEFAULT_REWARD_SPEC.budget_steps
    assert state.event_state["goal_consecutive_hits"] == 0.0


def test_step_threads_state_and_charges_logical_budget():
    env = make_env()
    state = env.reset(START)
    h = 1.1
    for i in range(2):
        tr = env.step(step_action(h + 0.1, h))
        h += 0.1
        assert not tr.terminated
        assert tr.logical_cost == 1
        assert tr.state.steps_remaining == DEFAULT_REWARD_SPEC.budget_steps - i
        assert tr.next_state.coordinates[1, 0] == pytest.approx(h)
    assert env.state.coordinates[1, 0] == pytest.approx(h)
    # third step completes the second consecutive goal hit -> success terminal
    tr = env.step(step_action(h + 0.1, h))
    assert tr.terminated and tr.termination_reason == "goal_connected_proxy"


def test_faked_censoring_surfaces_as_censored_transition():
    calls = {"n": 0}

    def status_for_step(step: int) -> str:
        return "failed_infra" if step == 1 else "completed"

    source = FakeObservationSource(target_coordinates=TARGET,
                                   status_for_step=status_for_step)
    env = make_env(source=source)
    env.reset(START)
    tr0 = env.step(step_action(1.2, 1.1))
    assert tr0.termination_reason == "continued"
    tr1 = env.step(step_action(1.3, 1.2))
    assert tr1.termination_reason == "censored_operational_limit"
    assert not tr1.terminated and not env.done
    # the observed successor is retained; the episode can be resumed
    assert env.state.steps_remaining == DEFAULT_REWARD_SPEC.budget_steps - 2


def test_snapshot_restore_roundtrip_preserves_visible_state():
    env = make_env()
    env.reset(START)
    env.step(step_action(1.2, 1.1))
    snap = env.snapshot()
    env2 = make_env()
    env2.restore(snap, observation_source=FakeObservationSource(target_coordinates=TARGET))
    assert env2.state.content_hash() == env.state.content_hash()
    assert env2.snapshot() == snap
    # restored env continues stepping
    tr = env2.step(step_action(1.3, 1.2))
    assert tr.state.content_hash() == env.state.content_hash()


def test_snapshot_refuses_foreign_episode():
    env = make_env()
    env.reset(START)
    snap = env.snapshot()
    other = DirectEnvironment(
        observation_source=FakeObservationSource(target_coordinates=TARGET),
        atomic_numbers=Z, task_graph=TASK_GRAPH, atom_map_ids=("a0", "a1", "a2"),
        episode_id="ep-2", reaction_id="RXN_OTHER", group_id="G", subset="s",
    )
    with pytest.raises(Exception):
        other.restore(snap, observation_source=FakeObservationSource(TARGET))


def test_nonfinite_action_stops_episode_loudly():
    env = make_env()
    env.reset(START)
    nan = DirectAction(action_max_angstrom=0.1, latent_u=np.full((3, 3), np.nan),
                       normalized_z=None, proposed_angstrom=None, executed=False,
                       rejection_reason=None)
    tr = env.step(nan)
    assert tr.action.rejection_reason == "nonfinite_latent"
    assert tr.td_eligible is False
    assert env.done  # collection version must stop (plan §3.3)


def test_state_sufficiency_same_visible_state_same_successor():
    """Different histories reaching the same *visible* state (including
    event_state) advance identically (plan §3.2): the advancer's output is a
    pure function of the visible state, action, and observation."""
    from pes2ts_core.generation.planning.direct_mdp import (
        DirectObservation,
        advance_direct_state,
    )
    env = make_env()
    same = env.reset(START)
    action = step_action(1.2, 1.1)
    obs = DirectObservation(status="completed", energy_hartree=-10.0,
                            gradient_hartree_per_angstrom=np.zeros((3, 3)))
    outs = []
    for episode, step_idx in (("history-A", 1), ("history-B", 37)):
        tr = advance_direct_state(same, action, obs, episode_id=episode,
                                  step_index=step_idx, reaction_id="RXN_TEST",
                                  group_id="G_TEST", subset="direct_fit50",
                                  transition_id=f"{episode}-t")
        outs.append(tr)
    assert outs[0].reward == pytest.approx(outs[1].reward)
    assert outs[0].next_state.content_hash() == outs[1].next_state.content_hash()
    assert outs[0].termination_reason == outs[1].termination_reason
