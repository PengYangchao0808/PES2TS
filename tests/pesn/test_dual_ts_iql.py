"""Dual-TS IQL semantics fixtures (WP5-prep, plan v4 §I.11/§I.12).

Fixture-level, synthetic toy data, tiny nets, CPU-quick — NEVER value
training on real data (WP6 gate: BC frozen + RL eligibility first).

Coverage map (deliverable letter → test):

(a) Q action-direction sensitivity
    → test_geometric_action_features_flip_with_action
    → test_q_critic_action_direction_sensitive
(b) warmup: actor bit-identical after the frozen 2,000 Q/V updates,
    no policy updates
    → test_warmup_actor_frozen_through_2000_updates
(c) phase transition only after the warmup count; joint budget honored
    → test_phase_transition_only_after_warmup_count
(d) BC-actor loader identity gates (digest/spec/action/seed refusals)
    → test_bc_actor_loader_identity_gates
(e) multi-episode indexing: no cross-episode successor; per-episode TD
    chains correct
    → test_cross_episode_successor_refused
    → test_td_chains_resolve_within_episode
(f) resume bit-consistency (full state round-trip)
    → test_resume_bit_consistency
(g) per-head eligibility masks skip + count
    → test_eligibility_masks_skip_and_count
(h) AWR weight formula + clip at 100
    → test_awr_weights_formula_and_clip
(i) expectile loss hand-check
    → test_expectile_loss_hand_check
(j) unit discipline: mis-scaled a (Å vs z) detected/refused via version
    flags and the a_max bound
    → test_unit_discipline_version_flags_and_bounds
plus: shared §I.10 sampling/reduction import discipline and checkpoint
tamper/frozen-identity refusals.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from pes2ts_core.pesn.models import require_torch
from pes2ts_core.pesn.contracts.dual_ts import (
    ACTION_SCHEMA_VERSION,
    DualTSAction,
    DualTSActionKind,
    DualTSBundle,
    DualTSTransition,
)
from pes2ts_core.pesn.models.dual_egnn import (
    DualTSModelConfig,
    collate_dual_states,
)
from pes2ts_core.pesn.models.dual_ts_critics import (
    Q_ACTION_INPUT_VERSION,
    S_A_RULE,
    DualTSCriticError,
    DualTSQCritic,
    DualTSVCritic,
    geometric_action_features,
    validate_physical_action,
)
from pes2ts_core.pesn.models.dual_ts_policy import DualTSPolicy
from pes2ts_core.pesn.offline.training import dual_ts_iql, dual_ts_sampling
from pes2ts_core.pesn.offline.training.dual_ts_iql import (
    BC_CHECKPOINT_SCHEMA,
    BCActorIdentityError,
    CHECKPOINT_SCHEMA,
    DualTSIQLSettings,
    DualTSIQLTrainer,
    DualTSIQLTrainerError,
    DualTSTransitionRow,
    EpisodeKeyedBuffer,
    awr_weights,
    checkpoint_content_digest,
    expectile_loss,
    load_bc_actor,
    restore_dual_ts_iql_checkpoint,
)
from tests.pesn.test_dual_ts_model import synthetic_states

torch = require_torch()

SPEC_DIGEST = "f" * 64
A_MAX = 0.1
N_ATOMS = 6
PAIRS = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 3], [1, 4]])
DIRECTED = np.concatenate([PAIRS, PAIRS[:, ::-1]], axis=0).T


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _state(seed: int) -> dict:
    return synthetic_states(1, seed=seed)[0]


def _radial_inverse_np(z: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(z, axis=-1, keepdims=True)
    r_c = np.clip(r, 1e-12, 1.0 - 1e-7)
    return z * (np.arctanh(r_c) / np.maximum(r, 1e-12))


def _move(rng: np.random.Generator, scale: float = 0.04) -> DualTSAction:
    a = rng.normal(size=(N_ATOMS, 3))
    norms = np.linalg.norm(a, axis=-1, keepdims=True)
    a = a / norms * (scale * (0.5 + rng.random((N_ATOMS, 1))))
    z = a / A_MAX
    act = DualTSAction(
        kind=DualTSActionKind.MOVE, a_max_angstrom=A_MAX,
        latent_u=_radial_inverse_np(z), normalized_z=z, action_angstrom=a,
        density_version="fixture:radial_tanh_v0")
    act.validate()
    return act


def _fallback(rng: np.random.Generator) -> DualTSAction:
    """Accepted fallback: physical carrier present, NO latent (not sampled)."""
    a = rng.normal(size=(N_ATOMS, 3))
    norms = np.linalg.norm(a, axis=-1, keepdims=True)
    a = a / norms * 0.03
    act = DualTSAction(kind=DualTSActionKind.ACCEPTED_FALLBACK,
                       a_max_angstrom=A_MAX, action_angstrom=a,
                       reason="fixture_accepted_fallback")
    act.validate()
    return act


def _ok(head: str = "fixture"):
    return {"eligible": True, "reason": head}


def _chain(group: str, branch: str, episode: str, n_steps: int, seed: int,
           *, rewards: bool = True, q_flag: bool = True,
           v_declared: bool = True, fallback: bool = False,
           reward_values: list[float] | None = None) -> list:
    rows = []
    rng = np.random.default_rng(seed)
    for k in range(n_steps):
        last = k == n_steps - 1
        heads = {"Q": _ok("fixture_q"), "actor": _ok("fixture_actor")}
        if v_declared:
            heads["V"] = _ok("fixture_v")
        if not q_flag:
            heads["Q"] = {"eligible": False, "reason": "fixture_q_blocked"}
        action = _fallback(rng) if fallback else _move(rng)
        reward = None
        if rewards:
            reward = (reward_values[k] if reward_values is not None
                      else 0.1 * (k + 1))
        t = DualTSTransition(
            transition_id=f"{episode}-t{k}",
            state_id=f"{episode}-s{k}",
            next_state_id=None if last else f"{episode}-s{k + 1}",
            episode_id=episode, branch_id=branch,
            proposed_action=action,
            next_observation_digest=None if last else "ab" * 32,
            next_observation_missing_reason=(
                "teacher_frame_isolated" if last else None),
            boundary="teacher_endpoint" if last else "none",
            terminal=last,
            head_eligibility=heads,
        )
        rows.append(DualTSTransitionRow(
            reaction_group_id=group, transition=t,
            state_inputs=_state(seed * 100 + k),
            reward=reward,
            reward_missing_reason=(None if rewards else "outcome_unknown"),
        ))
    return rows


def _bc_checkpoint(path: Path, seed: int, spec_digest: str, *, hidden: int = 16,
                   n_layers: int = 1, d_model: int = 16, d_element: int = 8,
                   action_version: str = ACTION_SCHEMA_VERSION) -> dict:
    torch.manual_seed(0)
    pol = DualTSPolicy(
        config=DualTSModelConfig(hidden=hidden, n_layers=n_layers,
                                 d_model=d_model, d_element=d_element),
        action_max=A_MAX, sigma_range=(0.02, 1.0))
    payload = {
        "schema_version": BC_CHECKPOINT_SCHEMA,
        "trainer_version": "pesn_dual_ts_bc_v2.0",
        "identity": {
            "features_spec_digest": spec_digest,
            "action_version": action_version,
            "seed": int(seed),
        },
        "model_state": pol.state_dicts(),
    }
    payload["content_digest"] = checkpoint_content_digest(torch, payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def _settings(**kw) -> DualTSIQLSettings:
    base = dict(warmup_updates=4, joint_max_updates=8, batch_reactions=2,
                rows_per_group_per_draw=1, seed=42, hidden=16, n_layers=1,
                d_model=16, d_element=8, device="cpu", require_cuda=False,
                cpu_intra_threads=1, cpu_inter_threads=1, log_every=1,
                diagnostic_run=True)
    base.update(kw)
    return DualTSIQLSettings(**base)


def _bundle(bc_path: Path, seed: int = 42) -> DualTSBundle:
    payload = torch.load(bc_path, map_location="cpu", weights_only=False)
    bundle = DualTSBundle(
        run_id="wp5-fixture-run",
        bc_parent_checkpoint_digest=str(payload["content_digest"]),
        seeds={"training": int(seed), "bc": int(seed)},
        dataset_version="fixture_dataset_v0",
        norm_version="fixture_norm_v0",
        split_version="fixture_split_v0",
        action_version=ACTION_SCHEMA_VERSION,
        environment_version="fixture_env_v0",
        reward_version="fixture_reward_v0")
    bundle.validate()
    return bundle


def _standard_buffer() -> EpisodeKeyedBuffer:
    rows = (_chain("R1", "A", "R1|E1", 3, 1)
            + _chain("R1", "B", "R1|E2", 2, 2)
            + _chain("R2", "A", "R2|E1", 2, 3))
    return EpisodeKeyedBuffer(rows)


def _trainer(tmp_path: Path, buffer: EpisodeKeyedBuffer,
             settings: DualTSIQLSettings, *, seed: int = 42,
             spec: str = SPEC_DIGEST, name: str = "run",
             bc_path: Path | None = None) -> DualTSIQLTrainer:
    bc = bc_path or (tmp_path / "bc_actor.pt")
    if not bc.is_file():
        _bc_checkpoint(bc, seed, spec, hidden=settings.hidden,
                       n_layers=settings.n_layers, d_model=settings.d_model,
                       d_element=settings.d_element)
    return DualTSIQLTrainer(
        settings=settings, buffer=buffer, run_dir=tmp_path / name,
        bc_checkpoint_path=bc, bundle=_bundle(bc, seed),
        features_spec_digest=spec, s_g=0.015440, s_delta_e=0.0788881)


def _snapshot(trainer: DualTSIQLTrainer) -> dict:
    nets = {}
    for name in ("q1", "q2", "v", "q1_target", "q2_target"):
        nets[name] = {k: v.detach().cpu().clone()
                      for k, v in getattr(trainer, name).state_dict().items()}
    nets["actor"] = {
        block: {k: v.detach().cpu().clone() for k, v in sd.items()}
        for block, sd in trainer.actor.state_dicts().items()}
    opt = {}
    for name in ("q", "v", "pi"):
        st = getattr(trainer, f"opt_{name}").state_dict()
        opt[name] = {
            k: {kk: (vv.clone() if hasattr(vv, "clone") else vv)
                for kk, vv in v.items()}
            for k, v in st["state"].items()}
    return {"nets": nets, "opt": opt,
            "torch_rng": torch.get_rng_state().clone(),
            "phase": trainer.phase,
            "counts": (trainer.warmup_updates_done,
                       trainer.joint_updates_done),
            "sampler_draws": trainer.sampler.draws,
            "history_len": len(trainer.history)}


def _assert_snapshots_equal(a: dict, b: dict, label: str) -> None:
    def walk(x, y, path):
        if hasattr(x, "shape"):
            assert torch.equal(x, y), f"{label}: tensor mismatch at {path}"
        elif isinstance(x, dict):
            assert set(x) == set(y), f"{label}: keys differ at {path}"
            for k in x:
                walk(x[k], y[k], f"{path}.{k}")
        elif isinstance(x, (list, tuple)):
            assert len(x) == len(y), f"{label}: len differs at {path}"
            for i, (xi, yi) in enumerate(zip(x, y)):
                walk(xi, yi, f"{path}[{i}]")
        else:
            assert x == y, f"{label}: {path}: {x!r} != {y!r}"

    walk(a, b, label)


# ---------------------------------------------------------------------------
# (h) AWR weights + (i) expectile
# ---------------------------------------------------------------------------


def test_awr_weights_formula_and_clip():
    A = torch.tensor([-3.0, 0.0, 3.0, 30.0, -100.0])
    w = awr_weights(torch, A, temperature=3.0, clip=100.0)
    assert float(w[0]) == pytest.approx(float(np.exp(-1.0)), rel=1e-6)
    assert float(w[1]) == pytest.approx(1.0, rel=1e-6)
    assert float(w[2]) == pytest.approx(float(np.e), rel=1e-6)
    assert float(w[3]) == 100.0          # exp(10) clipped at the cap
    assert float(w[4]) < 1.0             # negative advantage → down-weight
    assert not w.requires_grad           # A is stop-grad by construction
    a_grad = torch.zeros(4, requires_grad=True)
    assert not awr_weights(torch, a_grad).requires_grad
    assert dual_ts_iql.awr_weights is not None


def test_expectile_loss_hand_check():
    diff = torch.tensor([1.0, -1.0, 2.0])
    tau = 0.7
    expected = (tau * 1.0 + (1.0 - tau) * 1.0 + tau * 4.0) / 3.0
    assert float(expectile_loss(torch, diff, tau)) == pytest.approx(
        expected, rel=1e-6)
    # τ = 0.5 reduces to half the MSE
    half = expectile_loss(torch, diff, 0.5)
    assert float(half) == pytest.approx(0.5 * float((diff ** 2).mean()),
                                        rel=1e-6)


# ---------------------------------------------------------------------------
# (a) Q action-direction sensitivity + (j) unit discipline
# ---------------------------------------------------------------------------


def test_geometric_action_features_flip_with_action():
    batch = collate_dual_states(torch, [_state(0)])
    rng = np.random.default_rng(0)
    a = torch.as_tensor(_move(rng).action_angstrom, dtype=torch.float64)
    atom_pos, edge_pos = geometric_action_features(torch, batch, a, s_a=A_MAX)
    atom_neg, edge_neg = geometric_action_features(torch, batch, -a,
                                                   s_a=A_MAX)
    # per-atom: ||ã|| unchanged; ã·g̃ flips sign exactly
    assert torch.allclose(atom_pos[:, 0], atom_neg[:, 0])
    assert torch.allclose(atom_pos[:, 1], -atom_neg[:, 1], atol=1e-12)
    # per-edge: r̂ projections flip; Δd̃ changes but is NOT a sign flip
    assert torch.allclose(edge_pos[:, 0], -edge_neg[:, 0], atol=1e-12)
    assert torch.allclose(edge_pos[:, 1], -edge_neg[:, 1], atol=1e-12)
    assert not torch.allclose(edge_pos[:, 2], edge_neg[:, 2])
    assert not torch.allclose(edge_pos[:, 2], -edge_neg[:, 2])
    # features are finite and the atom block is exactly 2-dim / edge 3-dim
    assert atom_pos.shape == (int(batch["z"].shape[0]), 2)
    assert edge_pos.shape == (int(batch["edge_index"].shape[1]), 3)
    assert torch.isfinite(atom_pos).all() and torch.isfinite(edge_pos).all()


def test_q_critic_action_direction_sensitive():
    critic = DualTSQCritic(
        config=DualTSModelConfig(hidden=16, n_layers=1, d_model=16,
                                 d_element=8),
        s_a_angstrom=A_MAX, a_max=A_MAX)
    batch = collate_dual_states(torch, synthetic_states(2, seed=3))
    rng = np.random.default_rng(5)
    move = torch.as_tensor(_move(rng).action_angstrom, dtype=torch.float64)
    # two graphs in the batch → per-graph identical physical actions
    a1 = torch.cat([move, move], dim=0)
    with torch.no_grad():
        q_a = critic(batch, a1, Q_ACTION_INPUT_VERSION)
        q_neg = critic(batch, -a1, Q_ACTION_INPUT_VERSION)
        q_b = critic(batch, a1 * 0.5, Q_ACTION_INPUT_VERSION)
    assert torch.isfinite(q_a).all()
    # NOT invariant under action reversal — direction matters
    assert float((q_a - q_neg).abs().max()) > 1e-6
    # magnitude matters too
    assert float((q_a - q_b).abs().max()) > 1e-6
    # V has no action channel at all
    v = DualTSVCritic(config=critic.config, s_a_angstrom=A_MAX, a_max=A_MAX)
    with torch.no_grad():
        v0 = v(batch)
    assert v0.shape == q_a.shape and torch.isfinite(v0).all()
    # independent init: Q and V networks differ from birth
    assert critic.init_state_digest != v.init_state_digest


def test_unit_discipline_version_flags_and_bounds():
    critic = DualTSQCritic(
        config=DualTSModelConfig(hidden=16, n_layers=1, d_model=16,
                                 d_element=8),
        s_a_angstrom=A_MAX, a_max=A_MAX)
    batch = collate_dual_states(torch, [_state(7)])
    rng = np.random.default_rng(9)
    action = _move(rng)
    a = torch.as_tensor(action.action_angstrom, dtype=torch.float64)

    def expect_error(fn, code):
        with pytest.raises(DualTSCriticError) as err:
            fn()
        assert err.value.code == code

    # correct version flag + physical Å works
    with torch.no_grad():
        critic(batch, a, Q_ACTION_INPUT_VERSION)
    # dimensionless-z unit flag → typed refusal (never silently rescaled)
    z_version = "pesn_dual_ts_action_v2:radial_tanh_v0:a_unit=dimensionless_z"
    expect_error(lambda: critic(batch, a, z_version),
                 "ACTION_VERSION_MISMATCH")
    # ambiguous version without a unit declaration → refusal
    expect_error(lambda: critic(batch, a, "radial_tanh_v0"),
                 "ACTION_VERSION_MISMATCH")
    # the z CARRIER passed with a lying Å flag is still caught by the bound:
    z = torch.as_tensor(action.normalized_z, dtype=torch.float64)
    assert float(z.norm(dim=-1).max()) > A_MAX  # z norms ≫ 0.1
    expect_error(lambda: critic(batch, z, Q_ACTION_INPUT_VERSION),
                 "ACTION_BOUND_EXCEEDED")
    # non-finite → refusal
    bad = a.clone()
    bad[0, 0] = float("nan")
    expect_error(lambda: validate_physical_action(torch, batch, bad,
                                                  a_max=A_MAX,
                                                  action_version=Q_ACTION_INPUT_VERSION),
                 "NONFINITE_ACTION")
    # wrong atom count → refusal
    expect_error(lambda: critic(collate_dual_states(torch, [_state(8), _state(9)]),
                                a, Q_ACTION_INPUT_VERSION),
                 "ACTION_SHAPE_INVALID")
    # s_a scale rule: frozen bound reuse, recorded for the checkpoint
    assert S_A_RULE == "action_bound_a_max_v0"
    assert critic.identity_payload()["s_a_angstrom"] == A_MAX
    assert critic.identity_payload()["s_a_rule"] == S_A_RULE


# ---------------------------------------------------------------------------
# (d) BC-actor loader identity gates
# ---------------------------------------------------------------------------


def _fresh_policy() -> DualTSPolicy:
    return DualTSPolicy(
        config=DualTSModelConfig(hidden=16, n_layers=1, d_model=16,
                                 d_element=8),
        action_max=A_MAX, sigma_range=(0.02, 1.0))


def test_bc_actor_loader_identity_gates(tmp_path):
    bc_path = tmp_path / "bc.pt"
    payload = _bc_checkpoint(bc_path, seed=42, spec_digest=SPEC_DIGEST)

    def expect_error(**kw):
        defaults = dict(policy=_fresh_policy(), payload=payload,
                        expected_features_spec_digest=SPEC_DIGEST,
                        expected_action_version=ACTION_SCHEMA_VERSION,
                        expected_seed=42,
                        expected_checkpoint_digest=str(payload["content_digest"]))
        defaults.update(kw)
        with pytest.raises(BCActorIdentityError) as err:
            load_bc_actor(torch, **defaults)
        return err.value.code

    # happy path: loads, strict, provenance returned
    pol = _fresh_policy()
    prov = load_bc_actor(
        torch, policy=pol, payload=payload,
        expected_features_spec_digest=SPEC_DIGEST,
        expected_action_version=ACTION_SCHEMA_VERSION, expected_seed=42,
        expected_checkpoint_digest=str(payload["content_digest"]))
    assert prov["seed"] == 42
    ref = payload["model_state"]["policy"]
    for k, v in pol.module.state_dict().items():
        assert torch.equal(v, ref[k])

    assert expect_error(
        payload={"schema_version": "pesn_dual_ts_bc_v1"}) == "BC_SCHEMA_UNKNOWN"
    no_digest = copy.deepcopy(payload)
    no_digest.pop("content_digest")
    assert expect_error(payload=no_digest) == "BC_DIGEST_MISSING"
    tampered = copy.deepcopy(payload)
    key = next(iter(tampered["model_state"]["policy"]))
    tampered["model_state"]["policy"][key] = \
        tampered["model_state"]["policy"][key] + 0.5
    assert expect_error(payload=tampered) == "BC_SELF_DIGEST_MISMATCH"
    assert expect_error(
        expected_checkpoint_digest="0" * 64) == "BC_BUNDLE_DIGEST_MISMATCH"
    assert expect_error(
        expected_features_spec_digest="0" * 64) == "BC_FEATURES_SPEC_MISMATCH"
    assert expect_error(
        expected_action_version="radial_tanh_v0_OLD") == \
        "BC_ACTION_VERSION_MISMATCH"
    assert expect_error(expected_seed=7) == "BC_SEED_MISMATCH"

    # end-to-end: a SELF-consistent BC checkpoint whose content digest is
    # NOT the one the bundle binds (bc_parent_checkpoint_digest) is refused
    # inside the trainer constructor — identity comes from the bundle.
    other = tmp_path / "bc_other.pt"
    _bc_checkpoint(other, seed=42, spec_digest=SPEC_DIGEST)
    payload_other = torch.load(other, map_location="cpu", weights_only=False)
    original_digest = str(payload_other["content_digest"])
    payload_other["model_state"]["policy"][key] = \
        payload_other["model_state"]["policy"][key] - 1.0
    payload_other["content_digest"] = checkpoint_content_digest(
        torch, payload_other)
    other2 = tmp_path / "bc_other2.pt"
    torch.save(payload_other, other2)
    rows = _chain("R1", "A", "E1", 2, 1)
    buf = EpisodeKeyedBuffer(rows)
    stale_bundle = DualTSBundle(
        run_id="wp5-fixture-run",
        bc_parent_checkpoint_digest=original_digest,  # pre-tamper binding
        seeds={"training": 42, "bc": 42},
        dataset_version="d", norm_version="n", split_version="s",
        action_version=ACTION_SCHEMA_VERSION, environment_version="e",
        reward_version="r")
    with pytest.raises(BCActorIdentityError) as err:
        DualTSIQLTrainer(
            settings=_settings(), buffer=buf, run_dir=tmp_path / "r",
            bc_checkpoint_path=other2, bundle=stale_bundle,
            features_spec_digest=SPEC_DIGEST, s_g=0.015, s_delta_e=0.08)
    assert err.value.code == "BC_BUNDLE_DIGEST_MISMATCH"


# ---------------------------------------------------------------------------
# (e) multi-episode indexing + per-episode TD chains
# ---------------------------------------------------------------------------


def test_cross_episode_successor_refused():
    rows = _chain("R1", "A", "E1", 2, 11)
    other = _chain("R1", "A", "E2", 2, 12)
    rogue = DualTSTransition(
        transition_id="E3-t0", state_id="E3-s0", next_state_id="E1-s1",
        episode_id="E3", branch_id="A",
        proposed_action=_move(np.random.default_rng(13)),
        next_observation_missing_reason="observation_not_exported",
        head_eligibility={"Q": _ok(), "V": _ok(), "actor": _ok()},
    )
    rogue_row = DualTSTransitionRow(reaction_group_id="R1", transition=rogue,
                                    state_inputs=_state(99), reward=0.1)
    with pytest.raises(DualTSIQLTrainerError) as err:
        EpisodeKeyedBuffer(rows + other + [rogue_row])
    assert err.value.code == "CROSS_EPISODE_SUCCESSOR"

    # unresolved successor (not present anywhere) is refused too
    dangling = DualTSTransitionRow(
        reaction_group_id="R1",
        transition=DualTSTransition(
            transition_id="E4-t0", state_id="E4-s0",
            next_state_id="NOWHERE", episode_id="E4", branch_id="A",
            proposed_action=_move(np.random.default_rng(14)),
            next_observation_digest="cd" * 32,
            head_eligibility={"Q": _ok(), "V": _ok(), "actor": _ok()}),
        state_inputs=_state(98), reward=0.1)
    with pytest.raises(DualTSIQLTrainerError) as err:
        EpisodeKeyedBuffer(rows + [dangling])
    assert err.value.code == "SUCCESSOR_UNRESOLVED"

    # the good buffer: chains resolve WITHIN the episode; terminals stop
    buf = EpisodeKeyedBuffer(_chain("R1", "A", "E1", 3, 11)
                             + _chain("R1", "A", "E2", 2, 12))
    ids = {r.transition.transition_id: i
           for i, r in enumerate(buf.rows)}
    for tid, idx in ids.items():
        row = buf.rows[idx]
        nxt = buf.next_row_of(idx)
        if row.transition.next_state_id is None:
            assert nxt is None and row.transition.terminal
        else:
            assert nxt is not None
            assert buf.rows[nxt].transition.episode_id == \
                row.transition.episode_id, "successor left the episode"
    # E1's terminal does NOT chain into E2's first state
    e1_last = ids["E1-t2"]
    assert buf.next_row_of(e1_last) is None
    assert buf.rows[ids["E2-t0"]].transition.state_id == "E2-s0"


def test_td_chains_resolve_within_episode(tmp_path):
    buf = EpisodeKeyedBuffer(
        _chain("R1", "A", "R1|E1", 3, 21, reward_values=[0.1, 0.2, 0.3])
        + _chain("R1", "B", "R1|E2", 2, 22, reward_values=[1.0, 2.0]))
    settings = _settings(warmup_updates=2, joint_max_updates=2)
    trainer = _trainer(tmp_path, buf, settings)
    ids = {r.transition.transition_id: i for i, r in enumerate(buf.rows)}
    report = trainer.inspect_td_targets(
        [ids["R1|E1-t1"], ids["R1|E1-t2"], ids["R1|E2-t0"]])
    # mid-episode row bootstraps from the SAME-episode successor
    mid = report[0]
    assert mid["episode_id"] == "R1|E1"
    assert mid["successor_episode_id"] == "R1|E1"
    assert mid["bootstrap_mask"] == 1.0
    successor_state = buf.rows[buf.next_row_of(
        ids["R1|E1-t1"])].state_inputs
    with torch.no_grad():
        v_next = trainer.v(collate_dual_states(torch, [successor_state]))
    assert mid["y"] == pytest.approx(0.2 + float(settings.gamma) * float(v_next[0]),
                                     rel=1e-5)
    # terminal row: no bootstrap, y == reward exactly
    term = report[1]
    assert term["bootstrap_mask"] == 0.0
    assert term["successor_episode_id"] is None
    assert term["y"] == pytest.approx(0.3)
    # the OTHER episode's rows never appear as successors of E1
    assert all(r["successor_episode_id"] in (None, "R1|E1")
               for r in report[:2])


# ---------------------------------------------------------------------------
# (g) per-head eligibility masks
# ---------------------------------------------------------------------------


def test_eligibility_masks_skip_and_count(tmp_path):
    rows = (_chain("R1", "A", "E", 4, 31, rewards=False, q_flag=False,
                   v_declared=False, fallback=False, reward_values=None),)
    # build the exact mixed-eligibility buffer by hand:
    base = _chain("R1", "A", "E", 4, 31)
    # r0: fully eligible (default)
    # r1: reward unknown → Q skipped, V/actor fine
    base[1] = DualTSTransitionRow(
        reaction_group_id="R1", transition=base[1].transition,
        state_inputs=base[1].state_inputs, reward=None,
        reward_missing_reason="outcome_unknown")
    # r2: accepted fallback (no latent) → actor skipped, Q/V fine
    fb = _fallback(np.random.default_rng(32))
    base[2] = DualTSTransitionRow(
        reaction_group_id="R1",
        transition=DualTSTransition(
            transition_id=base[2].transition.transition_id,
            state_id=base[2].transition.state_id,
            next_state_id=base[2].transition.next_state_id,
            episode_id=base[2].transition.episode_id,
            branch_id=base[2].transition.branch_id,
            proposed_action=fb,
            next_observation_digest="ab" * 32,
            head_eligibility={"Q": _ok(), "V": _ok(), "actor": _ok()}),
        state_inputs=base[2].state_inputs, reward=0.3)
    # r3: V eligibility not declared → V skipped, Q/actor fine
    t3 = base[3].transition
    base[3] = DualTSTransitionRow(
        reaction_group_id="R1",
        transition=DualTSTransition(
            transition_id=t3.transition_id, state_id=t3.state_id,
            next_state_id=t3.next_state_id, episode_id=t3.episode_id,
            branch_id=t3.branch_id, proposed_action=t3.proposed_action,
            next_observation_digest=None,
            next_observation_missing_reason="teacher_frame_isolated",
            boundary=t3.boundary, terminal=t3.terminal,
            head_eligibility={"Q": _ok(), "actor": _ok()}),
        state_inputs=base[3].state_inputs, reward=0.4)
    buf = EpisodeKeyedBuffer(base)
    trainer = _trainer(tmp_path, buf, _settings(warmup_updates=2,
                                                joint_max_updates=2))
    counts = trainer.eligibility_counts
    assert counts["Q"]["indexed"] == 3 and counts["Q"]["skipped"] == 1
    assert counts["Q"]["skip_reasons"] == {"reward_unknown": 1}
    assert counts["V"]["indexed"] == 3 and counts["V"]["skipped"] == 1
    assert counts["V"]["skip_reasons"] == {"eligibility_not_declared": 1}
    assert counts["actor"]["indexed"] == 3
    assert counts["actor"]["skip_reasons"] == {"latent_carrier_missing": 1}
    # per-head masks follow the same flags inside drawn batches
    idx = trainer.sampler.draw()
    batch = trainer._build_batch([buf.rows[i] for i in idx], idx)
    for row_i, global_i in enumerate(idx):
        flags = trainer.eligibility[global_i]
        assert float(batch["q_mask"][row_i]) == float(flags["Q"])
        assert float(batch["v_mask"][row_i]) == float(flags["V"])
        assert float(batch["pi_mask"][row_i]) == float(flags["actor"])
    hist = trainer.run_updates(4)
    assert len(hist) == 4
    assert set(trainer.skip_stats) == {"Q", "V", "actor"}


# ---------------------------------------------------------------------------
# (b) warmup freeze through the frozen 2,000 updates
# ---------------------------------------------------------------------------


def test_warmup_actor_frozen_through_2000_updates(tmp_path):
    buf = _standard_buffer()
    settings = _settings(warmup_updates=2000, joint_max_updates=4)
    trainer = _trainer(tmp_path, buf, settings, name="warmup2k")
    assert trainer.phase == "warmup"
    assert all(not p.requires_grad for p in trainer.actor.parameters())
    reference = {k: v.detach().cpu().clone()
                 for k, v in trainer.actor.module.state_dict().items()}
    q_ref = {k: v.detach().cpu().clone()
             for k, v in trainer.q1.state_dict().items()}
    trainer.run_updates(2000)
    # actor bit-identical through ALL 2,000 warmup updates + the transition
    for k, v in trainer.actor.module.state_dict().items():
        assert torch.equal(v.detach().cpu(), reference[k]), \
            f"actor tensor {k} moved during warmup"
    # Q/V actually trained (the warmup is Q/V updates, not a no-op)
    q_now = trainer.q1.state_dict()
    assert any(not torch.equal(q_now[k].cpu(), q_ref[k]) for k in q_ref)
    assert trainer.phase == "joint"
    assert trainer.warmup_updates_done == 2000
    assert trainer.joint_updates_done == 0
    # NO policy updates: no π loss anywhere, actor optimizer never stepped
    assert len(trainer.history) == 2000
    assert all(h["loss_pi"] is None and h["phase"] == "warmup"
               for h in trainer.history)
    assert len(trainer.opt_pi.state) == 0
    assert len(trainer.phase_transitions) == 1
    assert trainer.phase_transitions[0]["warmup_updates_done"] == 2000


# ---------------------------------------------------------------------------
# (c) phase machine transitions exactly at the warmup count
# ---------------------------------------------------------------------------


def test_phase_transition_only_after_warmup_count(tmp_path):
    buf = _standard_buffer()
    settings = _settings(warmup_updates=6, joint_max_updates=3)
    trainer = _trainer(tmp_path, buf, settings, name="phases")
    trainer.run_updates(5)
    assert trainer.phase == "warmup"
    assert trainer.phase_transitions == []
    trainer.run_updates(1)
    assert trainer.phase == "joint"
    assert trainer.warmup_updates_done == 6
    assert len(trainer.phase_transitions) == 1
    # joint budget is honored: 3 joint updates exhaust it
    trainer.run_updates(3)
    assert trainer.joint_updates_done == 3
    with pytest.raises(DualTSIQLTrainerError) as err:
        trainer.run_updates(1)
    assert err.value.code == "JOINT_BUDGET_EXHAUSTED"


# ---------------------------------------------------------------------------
# (f) resume bit-consistency + checkpoint identity refusals
# ---------------------------------------------------------------------------


def test_resume_bit_consistency(tmp_path):
    buf = _standard_buffer()
    settings = _settings(warmup_updates=5, joint_max_updates=40)
    bc = tmp_path / "bc_actor.pt"
    spec = SPEC_DIGEST
    trainer1 = _trainer(tmp_path, buf, settings, name="cont", bc_path=bc)
    trainer1.run_updates(12)
    snap1 = _snapshot(trainer1)

    trainer2 = _trainer(tmp_path, buf, settings, name="part", bc_path=bc)
    trainer2.run_updates(5)
    ck = tmp_path / "ckpt.pt"
    trainer2.save_checkpoint(ck)

    trainer3 = restore_dual_ts_iql_checkpoint(
        ck, buffer=buf, bundle=_bundle(bc), bc_checkpoint_path=bc,
        features_spec_digest=spec, s_g=0.015440, s_delta_e=0.0788881,
        run_dir=tmp_path / "resumed")
    # prove restore overwrites fresh state: perturb, then restore again
    with torch.no_grad():
        next(iter(trainer3.q1.parameters())).fill_(1.234)
    trainer3 = restore_dual_ts_iql_checkpoint(
        ck, buffer=buf, bundle=_bundle(bc), bc_checkpoint_path=bc,
        features_spec_digest=spec, s_g=0.015440, s_delta_e=0.0788881,
        run_dir=tmp_path / "resumed")
    assert float(next(iter(trainer3.q1.parameters()))
                 .detach().flatten()[0]) != 1.234
    assert trainer3.phase == "joint"          # warmup(5) done, 7 joint done
    trainer3.run_updates(7)
    snap3 = _snapshot(trainer3)
    _assert_snapshots_equal(snap1, snap3, "resume")
    assert snap3["counts"] == (5, 7)
    assert snap3["history_len"] == 12


def test_checkpoint_tamper_and_frozen_identity_refusals(tmp_path):
    buf = _standard_buffer()
    settings = _settings(warmup_updates=3, joint_max_updates=10)
    bc = tmp_path / "bc_actor.pt"
    trainer = _trainer(tmp_path, buf, settings, name="main", bc_path=bc)
    trainer.run_updates(5)
    ck = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(ck)

    # (1) tampered payload → content-digest refusal
    payload = torch.load(ck, map_location="cpu", weights_only=False)
    tampered = copy.deepcopy(payload)
    tampered["warmup_updates_done"] = 999
    ck_bad = tmp_path / "ckpt_bad.pt"
    torch.save(tampered, ck_bad)
    with pytest.raises(DualTSIQLTrainerError) as err:
        restore_dual_ts_iql_checkpoint(
            ck_bad, buffer=buf, bundle=_bundle(bc), bc_checkpoint_path=bc,
            features_spec_digest=SPEC_DIGEST, s_g=0.015440,
            s_delta_e=0.0788881, run_dir=tmp_path / "r_bad")
    assert err.value.code == "CHECKPOINT_DIGEST_MISMATCH"

    # (2) identity-consistent digest but a DIFFERENT run's spec digest →
    #     frozen-identity refusal (never resume across identities)
    forged = copy.deepcopy(payload)
    forged["identity"] = {**forged["identity"],
                          "features_spec_digest": "a" * 64}
    forged["content_digest"] = checkpoint_content_digest(torch, forged)
    ck_forged = tmp_path / "ckpt_forged.pt"
    torch.save(forged, ck_forged)
    with pytest.raises(DualTSIQLTrainerError) as err:
        restore_dual_ts_iql_checkpoint(
            ck_forged, buffer=buf, bundle=_bundle(bc), bc_checkpoint_path=bc,
            features_spec_digest=SPEC_DIGEST, s_g=0.015440,
            s_delta_e=0.0788881, run_dir=tmp_path / "r_forged")
    assert err.value.code == "FROZEN_IDENTITY_MISMATCH"


# ---------------------------------------------------------------------------
# pipeline / shared-surface discipline
# ---------------------------------------------------------------------------


def test_shared_sampling_reduction_surface_and_semantics(tmp_path):
    # §I.11 "BC/IQL 采样和 log-prob 归约不同" prevention: the trainer imports
    # the SHARED module — no local re-implementation exists.
    assert dual_ts_iql.hierarchical_logprob_reduction is \
        dual_ts_sampling.hierarchical_logprob_reduction
    assert dual_ts_iql.DualTSGroupBranchSampler is \
        dual_ts_sampling.DualTSGroupBranchSampler
    assert dual_ts_iql.GROUP_BRANCH_SAMPLING_RULE == \
        dual_ts_sampling.GROUP_BRANCH_SAMPLING_RULE
    assert dual_ts_iql.LOGPROB_REDUCTION_RULE == \
        dual_ts_sampling.LOGPROB_REDUCTION_RULE

    buf = _standard_buffer()
    trainer = _trainer(tmp_path, buf, _settings())
    ident = trainer.identity_payload()
    assert ident["action_version"] == ACTION_SCHEMA_VERSION
    assert ident["action_input_version"] == Q_ACTION_INPUT_VERSION
    assert ident["features_spec_digest"] == SPEC_DIGEST
    assert ident["frozen_scales"]["s_a_rule"] == S_A_RULE
    assert ident["frozen_scales"]["s_a_angstrom"] == A_MAX
    assert ident["targets"]["expectile"] == 0.7
    assert ident["targets"]["advantage_temperature"] == 3.0
    assert ident["targets"]["weight_clip"] == 100.0
    assert ident["targets"]["polyak_registry"]["pairs"] == ["q1", "q2"]
    # per-reaction multi-episode sampling view: group R1 carries 2 episodes
    groups = buf.head_groups()
    assert set(groups["R1"]) == {"A", "B"}
    assert groups["R1"]["A"] == [0, 1, 2] and groups["R1"]["B"] == [3, 4]
    record = trainer.run_record()
    assert record["research_only"] is True
    assert record["phase"] == "warmup"
    hist = trainer.run_updates(6)
    losses = [(h["loss_v"], h["loss_q"]) for h in hist]
    for lv, lq in losses:
        if lv is not None:
            assert np.isfinite(lv)
        if lq is not None:
            assert np.isfinite(lq)
    assert trainer.run_record()["phase"] == "joint"
