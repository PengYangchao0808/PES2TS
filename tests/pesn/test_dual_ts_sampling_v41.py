"""v4.1 WP2 tests: unified sampler equivalence, exposure accounting, and the
BC checkpoint-v2 → IQL ``load_bc_actor`` wiring (P06 / F5 / finding 3).

These are the WP2 acceptance tests on top of the 287 pesn tests:
* BC and IQL produce the IDENTICAL draw sequence for the same seed + dataset
  (recorded state ids compared literally);
* ``draw_batch`` group-round-robin semantics + carry-over;
* exposure accounting (coverage / repeat rate / near-TS fraction, per group);
* the BC checkpoint-v2 identity block loads through IQL ``load_bc_actor``;
* the λ/budget registration gate (arm B unfreeze + diagnostic budgets).
"""
from __future__ import annotations

import numpy as np
import pytest

from pes2ts_core.pesn.contracts.dual_ts import (
    ACTION_SCHEMA_VERSION,
    DualTSAction,
    DualTSActionKind,
    DualTSTransition,
)
from pes2ts_core.pesn.models import require_torch
from pes2ts_core.pesn.models.dual_egnn import DualTSModelConfig
from pes2ts_core.pesn.models.dual_ts_policy import DualTSPolicy
from pes2ts_core.pesn.offline.training.dual_ts_bc import (
    DualTSBCSettings,
    DualTSBCTrainer,
    resolve_lambda_registration,
)
from pes2ts_core.pesn.offline.training.dual_ts_iql import (
    DualTSTransitionRow,
    EpisodeKeyedBuffer,
    load_bc_actor,
)
from pes2ts_core.pesn.offline.training.dual_ts_sampling import (
    DualTSGroupBranchSampler,
    ExposureTracker,
    build_sampler_groups,
)
from tests.pesn.test_dual_ts_bc import make_population

torch = require_torch()
A_MAX = 0.1


# ---------------------------------------------------------------------------
# Fixtures: an IQL buffer built from the SAME synthetic population as BC
# ---------------------------------------------------------------------------

def _fallback_action(n_atoms: int) -> DualTSAction:
    rng = np.random.default_rng(0)
    a = rng.normal(size=(n_atoms, 3)) * 0.01
    return DualTSAction(kind=DualTSActionKind.ACCEPTED_FALLBACK,
                        a_max_angstrom=A_MAX, action_angstrom=a,
                        reason="fixture_accepted_fallback")


def _buffer_from_states(states: list) -> EpisodeKeyedBuffer:
    """One terminal transition per state; one episode per (group, branch).

    Rows are inserted in sorted (group, branch) then remaining-arc order, so
    ``head_groups()`` arc order (episode, chain position) coincides with the
    BC arc order (remaining_arc_length).
    """
    cells: dict[tuple, list] = {}
    for s in states:
        cells.setdefault((s["reaction_group_id"], s["branch_id"]), []).append(s)
    rows = []
    for (g, b) in sorted(cells):
        episode = f"{g}|{b}"
        for s in sorted(cells[(g, b)], key=lambda x: x["remaining_arc_length"]):
            n_atoms = int(np.asarray(s["atomic_numbers"]).size)
            t = DualTSTransition(
                transition_id=f"{s['state_id']}-t",
                state_id=s["state_id"],
                next_state_id=None,
                episode_id=episode,
                branch_id=b,
                proposed_action=_fallback_action(n_atoms),
                next_observation_missing_reason="teacher_frame_isolated",
                boundary="teacher_endpoint",
                terminal=True,
                head_eligibility={
                    "Q": {"eligible": True, "reason": "fixture"},
                    "V": {"eligible": True, "reason": "fixture"},
                    "actor": {"eligible": True, "reason": "fixture"},
                },
            )
            rows.append(DualTSTransitionRow(
                reaction_group_id=g, transition=t,
                state_inputs={"atomic_numbers": s["atomic_numbers"]},
                reward=0.1,
                remaining_arc_length=s["remaining_arc_length"]))
    return EpisodeKeyedBuffer(rows)


# ---------------------------------------------------------------------------
# 1. Sampler equivalence (BC vs IQL — the P06/F5 acceptance test)
# ---------------------------------------------------------------------------

def test_bc_iql_identical_draw_sequence():
    """UNIT-LEVEL (synthetic fixture): BC and IQL produce the IDENTICAL draw
    sequence for the same seed + dataset.  This is the fixture-shaped
    acceptance test; the REAL-shard production equivalence (finding 3) lives
    in ``test_g2_remediation_v41.py::test_real_shard_bc_iql_identical_draw_sequence``."""
    states = make_population()
    seed = 20261017

    groups_bc = build_sampler_groups(
        states,
        group_key=lambda s: s["reaction_group_id"],
        branch_key=lambda s: s["branch_id"],
        arc_key=lambda s, _i: s["remaining_arc_length"],
    )
    buf = _buffer_from_states(states)
    groups_iql = buf.head_groups()

    # the (group, branch) → [state_id] layout must be identical
    assert set(groups_bc) == set(groups_iql)
    for g in groups_bc:
        assert set(groups_bc[g]) == set(groups_iql[g])
        for b in groups_bc[g]:
            bc_ids = [states[i]["state_id"] for i in groups_bc[g][b]]
            iql_ids = [buf.rows[i].transition.state_id
                       for i in groups_iql[g][b]]
            assert bc_ids == iql_ids, (g, b)

    bc_sampler = DualTSGroupBranchSampler(seed=seed, groups=groups_bc,
                                          rows_per_group=1)
    iql_sampler = DualTSGroupBranchSampler(seed=seed, groups=groups_iql,
                                           rows_per_group=1)

    # record first N drawn state ids in both trainers and assert equality
    n = 40
    bc_draws = bc_sampler.draw_batch(n)
    iql_draws = iql_sampler.draw_batch(n)
    assert len(bc_draws) == len(iql_draws) == n
    bc_ids = [states[i]["state_id"] for i in bc_draws]
    iql_ids = [buf.rows[i].transition.state_id for i in iql_draws]
    assert bc_ids == iql_ids
    # and a second batch continues identically (carry-over)
    bc_draws2 = bc_sampler.draw_batch(7)
    iql_draws2 = iql_sampler.draw_batch(7)
    assert [states[i]["state_id"] for i in bc_draws2] == \
        [buf.rows[i].transition.state_id for i in iql_draws2]


def test_draw_batch_round_robin_carries_over():
    groups = {
        "A": {"b0": [0, 1], "b1": [2]},
        "B": {"b0": [3, 4]},
    }
    sampler = DualTSGroupBranchSampler(seed=7, groups=groups, rows_per_group=1)
    assert sampler.n_groups == 2 and sampler.rows_per_round == 2
    # 3 draws = one full round (A, B) + the first group of the next round (A)
    first = sampler.draw_batch(3)
    second = sampler.draw_batch(1)
    third = sampler.draw_batch(2)
    # concatenation of chunks == one contiguous stream
    all_rows = first + second + third
    assert len(all_rows) == 6
    # group-round-robin: every group exactly once per round, deterministic order
    full = DualTSGroupBranchSampler(seed=7, groups=groups, rows_per_group=1)
    stream = full.draw_batch(6)
    assert stream == all_rows
    assert full.rounds_completed == 3


def test_sampler_state_dict_roundtrip_preserves_stream():
    groups = {
        "A": {"b0": [0, 1], "b1": [2]},
        "B": {"b0": [3, 4]},
    }
    s1 = DualTSGroupBranchSampler(seed=3, groups=groups, rows_per_group=1)
    s1.draw_batch(5)
    state = s1.state_dict()
    s2 = DualTSGroupBranchSampler(seed=3, groups=groups, rows_per_group=1)
    s2.load_state_dict(state)
    assert s2.draw_batch(10) == s1.draw_batch(10)
    assert s2.rounds_completed == s1.rounds_completed


# ---------------------------------------------------------------------------
# 2. Exposure accounting
# ---------------------------------------------------------------------------

def test_exposure_tracker_accounting():
    row_meta = []
    for g in ("G1", "G2"):
        for k, ral in enumerate((8.0, 4.0, 1.0, 0.5)):
            row_meta.append({
                "group_id": g, "branch_id": "b",
                "state_id": f"{g}_s{k}",
                "remaining_arc_length": ral,
            })
    tracker = ExposureTracker(row_meta=row_meta)
    # indices: G1 s0..s3 (0..3), G1 s0,s1 again (0,1), G2 s0,s1 (4,5)
    tracker.record_batch([0, 1, 2, 3, 0, 1, 4, 5])
    tracker.snapshot(kind="epoch", update=1, epoch=1)
    payload = tracker.payload()
    assert payload["drawable_states"] == 8 and payload["n_groups"] == 2
    assert payload["total_draws"] == 8
    assert payload["distinct_states"] == 6
    assert payload["coverage"] == pytest.approx(6 / 8)
    assert payload["repeat_rate"] == pytest.approx(2 / 8)
    # near-TS: ral <= 0.2 * 8 = 1.6 -> rows ral 1.0 and 0.5 (drawn once each)
    assert payload["near_ts_draws"] == 2
    assert payload["near_ts_fraction"] == pytest.approx(2 / 8)
    per_group = payload["per_group"]
    assert per_group["G1"]["draws"] == 6
    assert per_group["G1"]["distinct_states"] == 4
    assert per_group["G1"]["coverage"] == pytest.approx(1.0)
    assert per_group["G1"]["repeat_rate"] == pytest.approx(2 / 6)
    assert per_group["G1"]["near_ts_draws"] == 2
    assert per_group["G2"]["draws"] == 2
    assert per_group["G2"]["coverage"] == pytest.approx(0.5)
    assert per_group["G2"]["repeat_rate"] == pytest.approx(0.0)
    # interval snapshot recorded (per-group interval deltas)
    assert len(payload["history"]) == 1
    h = payload["history"][0]
    assert h["kind"] == "epoch" and h["draws_in_interval"] == 8


# ---------------------------------------------------------------------------
# 3. BC checkpoint v2 -> IQL load_bc_actor
# ---------------------------------------------------------------------------

def test_bc_checkpoint_v2_loads_via_iql_load_bc_actor(tmp_path):
    states = make_population()
    spec = "f" * 64
    settings = DualTSBCSettings(
        recipe="bc_a", budget_updates=30, batch_states=8, seed=42,
        device="cpu", require_cuda=False, curve_every=30, eval_every=0)
    trainer = DualTSBCTrainer(
        settings=settings, states=states,
        data_meta={"spec_digest": spec, "shard_digests": ["a" * 64],
                   "norm_params": {"s_g_hartree_per_angstrom": 0.0154,
                                   "s_delta_e": 0.0789}},
        run_dir=tmp_path)
    trainer.train()
    ckpt = torch.load(tmp_path / "checkpoint_v2.pt", map_location="cpu",
                      weights_only=False)
    assert ckpt["schema_version"] == "pesn_dual_ts_bc_checkpoint_v2"
    ident = ckpt["identity"]
    assert ident["action_version"] == ACTION_SCHEMA_VERSION
    assert ident["seed"] == 42 and ident["recipe"] == "bc_a"
    assert ident["lambda_a"] == 1.0 and ident["lambda_dir"] == 1.0
    assert ident["features_spec_digest"] == spec

    policy = DualTSPolicy(
        config=DualTSModelConfig(light=True, signed_vector_messages=False),
        action_max=0.1, sigma_range=(0.02, 1.0), signed_mu_gate=False)
    prov = load_bc_actor(
        torch, policy=policy, payload=ckpt,
        expected_features_spec_digest=spec,
        expected_action_version=ACTION_SCHEMA_VERSION,
        expected_seed=42,
        expected_checkpoint_digest=str(ckpt["content_digest"]))
    assert prov["seed"] == 42
    assert prov["features_spec_digest"] == spec
    assert prov["bc_checkpoint_digest"] == ckpt["content_digest"]
    # strict load: policy head matches the checkpoint head exactly
    for k, v in policy.module.state_dict().items():
        assert torch.equal(v, ckpt["model_state"]["policy"][k])


# ---------------------------------------------------------------------------
# 4. λ / budget registration gate (arm B unfreeze)
# ---------------------------------------------------------------------------

_PROTOCOL = {
    "registered_budgets": {
        "fit1": {"updates": None},
        "fit5": {"updates": None},
        "short": {"updates": 2000},
        "full": {"updates": 20000},
    },
    "lambda_registration": {
        "slots": {
            "arm_a": {"lambda_a": 1.0, "lambda_dir": 1.0, "status": "frozen"},
            "arm_b": None,
        },
    },
}


def test_resolve_lambda_registration_gate():
    # registered arm A / stage full / correct budget → accepted
    d = resolve_lambda_registration(_PROTOCOL, recipe="bc_a", lambda_a=1.0,
                                    lambda_dir=1.0, stage="full",
                                    budget_updates=20000,
                                    allow_diagnostic_budget=False)
    assert d["accepted"] and not d["diagnostic"] and not d["problems"]
    # budget mismatch → refused
    d = resolve_lambda_registration(_PROTOCOL, recipe="bc_a", lambda_a=1.0,
                                    lambda_dir=1.0, stage="full",
                                    budget_updates=2000,
                                    allow_diagnostic_budget=False)
    assert not d["accepted"]
    # unregistered λ (arm B) → refused without the flag
    d = resolve_lambda_registration(_PROTOCOL, recipe="bc_d", lambda_a=1.0,
                                    lambda_dir=2.0, stage="full",
                                    budget_updates=20000,
                                    allow_diagnostic_budget=False)
    assert not d["accepted"]
    # ... accepted as a diagnostic run with the flag
    d = resolve_lambda_registration(_PROTOCOL, recipe="bc_d", lambda_a=1.0,
                                    lambda_dir=2.0, stage="full",
                                    budget_updates=20000,
                                    allow_diagnostic_budget=True)
    assert d["accepted"] and d["diagnostic"]
    # unknown stage → refused unless diagnostic
    d = resolve_lambda_registration(_PROTOCOL, recipe="bc_a", lambda_a=1.0,
                                    lambda_dir=1.0, stage="bogus",
                                    budget_updates=20000,
                                    allow_diagnostic_budget=False)
    assert not d["accepted"]
    # canary stage accepts any budget
    d = resolve_lambda_registration(_PROTOCOL, recipe="bc_a", lambda_a=1.0,
                                    lambda_dir=1.0, stage="fit1",
                                    budget_updates=50,
                                    allow_diagnostic_budget=False)
    assert d["accepted"] and not d["diagnostic"]
