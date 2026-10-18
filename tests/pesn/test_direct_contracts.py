"""Direct contract tests (WP0): schema rejections, hashes, expert rules."""
from __future__ import annotations

import numpy as np
import pytest

from pes2ts_core.pesn.contracts.direct import (
    ACTION_SCHEMA_VERSION,
    DEFAULT_REWARD_SPEC,
    REWARD_SPEC_SCHEMA_VERSION,
    STATE_SCHEMA_VERSION,
    DirectAction,
    DirectContractError,
    DirectRewardSpec,
    DirectState,
    DirectTransition,
)

TASK_GRAPH = {"edits": [{"kind": "make", "i": 1, "j": 2, "reactant_order": 0,
                         "product_order": 1}]}


def make_state(**over) -> DirectState:
    base = dict(
        atomic_numbers=np.array([6, 1, 6]),
        coordinates=np.array([[0.0, 0, 0], [1.1, 0, 0], [2.5, 0, 0]]),
        task_graph=TASK_GRAPH, atom_map_ids=("a0", "a1", "a2"),
        steps_remaining=100,
        event_state={"goal_consecutive_hits": 0.0, "no_progress_count": 0.0,
                     "best_potential": -0.5},
    )
    base.update(over)
    return DirectState(**base)


def test_state_validates_and_hashes_stably():
    s = make_state()
    s.validate()
    assert s.content_hash() == make_state().content_hash()
    bumped = make_state(coordinates=np.array([[0.0, 0, 0], [1.1000009, 0, 0], [2.5, 0, 0]]))
    assert bumped.content_hash() != s.content_hash()


@pytest.mark.parametrize("over,code", [
    (dict(coordinates=np.array([[0.0, 0, 0], [np.nan, 0, 0], [2.5, 0, 0]])),
     "NONFINITE_COORDINATES"),
    (dict(atom_map_ids=("a0", "a1")), "MAP_MISMATCH"),
    (dict(steps_remaining=101), "BUDGET_INVALID"),
    (dict(task_graph={"edits": [], "reference_ts": [[0.0]]}), "FORBIDDEN_FIELD"),
])
def test_state_rejections(over, code):
    with pytest.raises(DirectContractError) as err:
        make_state(**over).validate()
    assert err.value.code == code


def test_action_rejections():
    ok = DirectAction(action_max_angstrom=0.1, latent_u=np.zeros((3, 3)),
                      normalized_z=np.zeros((3, 3)), proposed_angstrom=np.zeros((3, 3)),
                      executed=True, rejection_reason=None, log_prob_z=-1.0)
    ok.validate()
    with pytest.raises(DirectContractError):
        DirectAction(action_max_angstrom=0.0).validate()
    with pytest.raises(DirectContractError):
        DirectAction(action_max_angstrom=0.1, executed=False,
                     rejection_reason="made_up").validate()
    # non-finite latent is an accepted fault record (classified downstream);
    # non-finite execution carriers stay contract rejections
    DirectAction(action_max_angstrom=0.1,
                 latent_u=np.full((3, 3), np.inf)).validate()
    with pytest.raises(DirectContractError):
        DirectAction(action_max_angstrom=0.1,
                     normalized_z=np.full((3, 3), np.nan)).validate()


def test_reward_spec_roundtrip_and_default_payload():
    spec = DEFAULT_REWARD_SPEC
    assert spec.gamma == 1.0 and spec.budget_steps == 100
    payload = spec.to_payload()
    assert payload["schema_version"] == REWARD_SPEC_SCHEMA_VERSION
    cloned = DirectRewardSpec.from_payload(payload)
    assert cloned.to_payload() == payload
    with pytest.raises(DirectContractError):
        DirectRewardSpec(gamma=0.9).validate()
    with pytest.raises(DirectContractError):
        DirectRewardSpec.from_payload({"schema_version": "wrong"})


def make_transition(**over) -> DirectTransition:
    state = make_state()
    action = DirectAction(action_max_angstrom=0.1, latent_u=np.zeros((3, 3)),
                          normalized_z=np.zeros((3, 3)),
                          proposed_angstrom=np.zeros((3, 3)), executed=True,
                          rejection_reason=None, log_prob_z=-1.0)
    nxt = make_state(steps_remaining=99)
    base = dict(
        transition_id="t0", reaction_id="RXN_T", group_id="G_T",
        subset="direct_fit50", episode_id="e0", step_index=0,
        state=state, action=action, next_state=nxt, reward=-0.01,
        reward_components={"step_cost": -0.01}, terminated=False,
        bootstrap=1.0, termination_reason="continued", td_eligible=True,
        logical_cost=1, qc_cost_evidence={"status": "completed"},
    )
    base.update(over)
    return DirectTransition(**base)


def test_transition_rules():
    make_transition().validate()
    with pytest.raises(DirectContractError):
        make_transition(terminated=True, bootstrap=1.0).validate()
    with pytest.raises(DirectContractError):
        make_transition(termination_reason="exploded").validate()
    with pytest.raises(DirectContractError):
        make_transition(termination_reason="censored_operational_limit",
                        terminated=True).validate()
    with pytest.raises(DirectContractError):
        make_transition(terminated=False, next_state=None).validate()
    with pytest.raises(DirectContractError):
        make_transition(logical_cost=0).validate()
    payload = make_transition().to_payload()
    assert payload["schema_version"] == "pes2ts_direct_transition_v1"
    assert payload["action"]["schema_version"] == ACTION_SCHEMA_VERSION
    assert payload["state"]["schema_version"] == STATE_SCHEMA_VERSION


def test_schema_versions_registered():
    assert STATE_SCHEMA_VERSION == "pesn_direct_state_v1"
    assert ACTION_SCHEMA_VERSION == "pesn_direct_action_v1"
