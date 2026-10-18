"""Direct-MDP advancer tests: the plan §5 reward hand-check table and the
plan §3.3 event priority. Synthetic H-transfer only; no QC, no truth."""
from __future__ import annotations

import numpy as np
import pytest

from pes2ts_core.generation.planning.direct_mdp import (
    ADVANCER_VERSION,
    DirectObservation,
    advance_direct_state,
    goal_connected_proxy,
    goal_potential,
)
from pes2ts_core.pesn.contracts.direct import (
    DEFAULT_REWARD_SPEC,
    DirectAction,
    DirectRewardSpec,
    DirectState,
    DirectTransition,
)

TASK_GRAPH = {
    "edits": [
        {"kind": "broken", "i": 0, "j": 1, "reactant_order": 1, "product_order": 0},
        {"kind": "formed", "i": 1, "j": 2, "reactant_order": 0, "product_order": 1},
    ]
}
Z = np.array([6, 1, 6])


def make_state(h_x: float, *, steps: int = 100, hits: float = 0.0,
               no_progress: float = 0.0, best: float | None = None) -> DirectState:
    coords = np.array([[0.0, 0.0, 0.0], [h_x, 0.0, 0.0], [2.5, 0.0, 0.0]])
    phi = goal_potential(coords, Z, TASK_GRAPH)
    return DirectState(
        atomic_numbers=Z, coordinates=coords, task_graph=TASK_GRAPH,
        atom_map_ids=("a0", "a1", "a2"), charge=0, multiplicity=1,
        method_id="GFN2-xTB", energy_hartree=-10.0, energy_reference_hartree=-10.0,
        gradient_hartree_per_angstrom=np.zeros((3, 3)), observation_mask=True,
        steps_remaining=steps,
        event_state={"goal_consecutive_hits": hits, "no_progress_count": no_progress,
                     "best_potential": phi if best is None else best},
    )


def action_for(h_x_next: float, h_x: float, *, a_max: float = 0.1,
               expert: bool = False) -> DirectAction:
    prop = np.zeros((3, 3))
    prop[1, 0] = h_x_next - h_x
    latent = prop / a_max  # tanh radial inverse at these magnitudes
    return DirectAction(
        action_max_angstrom=a_max, latent_u=latent.copy(),
        normalized_z=(prop / a_max), proposed_angstrom=prop,
        executed=True if not expert else True,
        rejection_reason=None,
        log_prob_z=None if expert else -3.2,
        log_prob_source="expert_not_probabilistic" if expert else "policy",
    )


OBS = DirectObservation(status="completed", energy_hartree=-10.0,
                        gradient_hartree_per_angstrom=np.zeros((3, 3)))


def run(spec: DirectRewardSpec, seed_hx: float, moves: list[float]):
    """Thread states through the advancer; return all transitions."""
    state = make_state(seed_hx)
    out: list[DirectTransition] = []
    h_x = seed_hx
    for t, target in enumerate(moves):
        tr = advance_direct_state(
            state, action_for(target, h_x), OBS, reward_spec=spec,
            episode_id="ep-test", step_index=t, reaction_id="RXN_TEST",
            group_id="G_TEST", subset="direct_fit50",
            transition_id=f"t{t}", parent_node_hash=state.content_hash(),
        )
        out.append(tr)
        if tr.terminated or tr.next_state is None:
            break
        state = tr.next_state
        h_x = target
    return out


def test_reward_hand_check_success():
    """T-step success totals B - cT - eta*Phi(s0); Phi(terminal)=0."""
    plan_spec = DEFAULT_REWARD_SPEC
    transitions = run(plan_spec, 1.1, [1.2, 1.3, 1.4, 1.5])
    # goal single-hit starts at x=1.3 (both edit conditions met), terminal on
    # the second consecutive hit -> executed steps 1.2,1.3,1.4,1.5 => T=4
    assert transitions[-1].termination_reason == "goal_connected_proxy"
    phi0 = goal_potential(np.array([[0, 0, 0], [1.1, 0, 0], [2.5, 0, 0]]), Z, TASK_GRAPH)
    total = sum(t.reward for t in transitions)
    expected = (plan_spec.success_bonus - plan_spec.step_cost * len(transitions)
                - plan_spec.shaping_scale * phi0)
    assert total == pytest.approx(expected, abs=1e-12)
    assert transitions[-1].bootstrap == 0.0


def test_reward_hand_check_immediate_failure_equals_exhausted():
    """Failure at any step totals -F - cH - eta*Phi(s0) (plan §5): the
    terminal fixes Phi=0 so the shaping telescopes to -eta*Phi(s0)."""
    spec = DEFAULT_REWARD_SPEC
    # immediate: first action drives H into catastrophic overlap with C0
    state = make_state(0.55)  # already inside the overlap danger zone
    prop = np.zeros((3, 3))
    prop[1, 0] = -0.1  # H1 slides to 0.45 A < 0.45*(r_C+r_H) overlap line
    bad = DirectAction(action_max_angstrom=0.1, latent_u=prop / 0.1,
                       normalized_z=prop / 0.1,
                       proposed_angstrom=prop,
                       executed=False, rejection_reason=None, log_prob_z=-3.0)
    tr = advance_direct_state(state, bad, None, reward_spec=spec, episode_id="e",
                              step_index=0, reaction_id="r", group_id="g",
                              subset="s", transition_id="t0")
    assert tr.termination_reason == "illegal_action"
    assert tr.action.rejection_reason == "catastrophic_overlap"
    assert tr.qc_cost_evidence is None
    assert tr.reward == pytest.approx(-(spec.failure_penalty + spec.step_cost * spec.budget_steps))
    # exhausted: the last logical step without a goal terminates as budget
    # exhaustion; the terminal shaping jump equals eta*(0 - Phi(s))
    last = make_state(1.1, steps=1)
    phi_s = goal_potential(last.coordinates, Z, TASK_GRAPH)
    tr_last = advance_direct_state(last, action_for(1.2, 1.1), OBS, reward_spec=spec,
                                   episode_id="e2", step_index=99, reaction_id="r",
                                   group_id="g", subset="s", transition_id="t99")
    assert tr_last.termination_reason == "budget_exhausted"
    assert tr_last.reward == pytest.approx(
        -(spec.step_cost + spec.failure_penalty)
        + spec.shaping_scale * (0.0 - phi_s))
    # no-progress variant: episode total = -F - cH - eta*Phi(s0)
    spec_np = DirectRewardSpec(no_progress_window=5)
    moves_np = [1.1 + 1e-9 * (i + 1) for i in range(6)]
    transitions = run(spec_np, 1.1, moves_np)
    assert transitions[-1].termination_reason == "no_progress_terminal"
    phi0 = goal_potential(np.array([[0, 0, 0], [1.1, 0, 0], [2.5, 0, 0]]), Z, TASK_GRAPH)
    assert sum(t.reward for t in transitions) == pytest.approx(
        -(spec_np.failure_penalty + spec_np.step_cost * spec_np.budget_steps)
        - spec_np.shaping_scale * phi0)


def test_censoring_is_not_terminal_and_adds_no_failure():
    state = make_state(1.1)
    censored_obs = DirectObservation(status="failed_infra", error="timeout")
    tr = advance_direct_state(state, action_for(1.2, 1.1), censored_obs,
                              reward_spec=DEFAULT_REWARD_SPEC, episode_id="e",
                              step_index=0, reaction_id="r", group_id="g",
                              subset="s", transition_id="t0")
    assert not tr.terminated
    assert tr.termination_reason == "censored_operational_limit"
    assert tr.td_eligible is False
    assert tr.bootstrap == 1.0
    assert "failure" not in tr.reward_components
    assert tr.next_state is not None


def test_event_priority_goal_beats_budget_exhaustion():
    """A first goal hit on the last logical step still terminates as success."""
    spec = DEFAULT_REWARD_SPEC
    state = make_state(1.24, steps=1, hits=1.0)  # one hit already; next hit wins
    tr = advance_direct_state(state, action_for(1.34, 1.24), OBS, reward_spec=spec,
                              episode_id="e", step_index=99, reaction_id="r",
                              group_id="g", subset="s", transition_id="t99")
    assert tr.termination_reason == "goal_connected_proxy"
    assert tr.terminated


def test_qc_geometry_failure_is_terminal_with_receipt():
    state = make_state(1.1)
    obs = DirectObservation(status="failed_geometry", error="SCF no convergence")
    tr = advance_direct_state(state, action_for(1.2, 1.1), obs,
                              reward_spec=DEFAULT_REWARD_SPEC, episode_id="e",
                              step_index=0, reaction_id="r", group_id="g",
                              subset="s", transition_id="t0")
    assert tr.terminated and tr.termination_reason == "qc_failure_terminal"
    assert tr.qc_cost_evidence["status"] == "failed_geometry"
    assert tr.td_eligible


def test_nonfinite_action_is_numeric_fault_not_failure():
    state = make_state(1.1)
    nan_action = DirectAction(action_max_angstrom=0.1,
                              latent_u=np.full((3, 3), np.nan),
                              normalized_z=None, proposed_angstrom=None,
                              executed=False, rejection_reason=None)
    tr = advance_direct_state(state, nan_action, None,
                              reward_spec=DEFAULT_REWARD_SPEC, episode_id="e",
                              step_index=0, reaction_id="r", group_id="g",
                              subset="s", transition_id="t0")
    assert not tr.terminated
    assert tr.td_eligible is False
    assert tr.action.rejection_reason == "nonfinite_latent"


def test_goal_proxy_requires_two_consecutive_hits_by_default():
    coords_hit = np.array([[0.0, 0, 0], [1.4, 0, 0], [2.5, 0, 0]])
    state = make_state(1.4)
    assert goal_connected_proxy(state) is True
    # single hit must not terminate when consecutive window is 2 (default)
    transitions = run(DEFAULT_REWARD_SPEC, 1.25, [1.35])
    assert transitions[0].termination_reason == "continued"


def test_potential_bounds_and_monotone_path():
    for hx in (0.9, 1.1, 1.3, 1.5, 2.0):
        coords = np.array([[0, 0, 0], [hx, 0, 0], [2.5, 0, 0]])
        phi = goal_potential(coords, Z, TASK_GRAPH)
        assert -1.0 <= phi <= 0.0
    assert goal_potential(np.array([[0, 0, 0], [1.55, 0, 0], [2.5, 0, 0]]),
                          Z, TASK_GRAPH) == 0.0


def test_expert_actions_carry_null_logprob():
    transitions = run(DEFAULT_REWARD_SPEC, 1.1, [1.2])
    expert = DirectAction(action_max_angstrom=0.1, latent_u=np.zeros((3, 3)),
                          normalized_z=np.zeros((3, 3)),
                          proposed_angstrom=np.zeros((3, 3)), executed=True,
                          rejection_reason=None, log_prob_z=None,
                          log_prob_source="expert_not_probabilistic")
    tr = advance_direct_state(transitions[0].state, expert, OBS,
                              reward_spec=DEFAULT_REWARD_SPEC, episode_id="e",
                              step_index=1, reaction_id="r", group_id="g",
                              subset="s", transition_id="t1")
    assert tr.action.log_prob_z is None
    with pytest.raises(Exception):
        DirectAction(action_max_angstrom=0.1, executed=False,
                     rejection_reason=None, log_prob_z=-1.0,
                     log_prob_source="expert_not_probabilistic").validate()


def test_advancer_version_stable():
    assert ADVANCER_VERSION.startswith("pes2ts_direct_advancer_v1")
