"""Lane C campaign trainer tests (synthetic; CPU only, <10s each).

Covers the full flow on a tiny synthetic population: legacy-head training
(50 steps) with per-update independent checkpoints + run_record v3, resume
determinism vs an uninterrupted run, epoch_traversal vs group_balanced
sampler interface parity, budget censoring, timing_only, the displacement
head loss (L_xyz/L_dir/L_mag) + dev evaluation against a duck-typed fake
model, the physical-prior action, and unified npz label loading.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pes2ts_core.pesn.offline.training.dual_ts_bc import (
    DualTSBCSettings,
    DualTSBCTrainer,
)
from pes2ts_core.pesn.offline.training.dual_ts_campaign import (
    CampaignArmConfig,
    CampaignDatasetSpec,
    CampaignTrainer,
    CampaignTrainerError,
    EpochTraversalSampler,
    build_displacement_targets,
    compute_a_s,
    compute_physical_prior_action,
    load_campaign_states,
    per_atom_cos_dir_loss,
    per_atom_mag_loss,
)
from pes2ts_core.pesn.offline.training.dual_ts_sampling import (
    DualTSGroupBranchSampler,
    build_sampler_groups,
)
from tests.pesn.test_dual_ts_bc import make_population, make_state

RUN_V3 = "pesn_dual_ts_bc_run_v3"
RUN_V3_KEYS = (
    "schema_version", "trainer_version", "arm_id", "preset", "recipe",
    "settings", "data", "accounting", "exposure", "history", "diagnostics",
    "fit_diagnostics_final", "wall_seconds", "checkpoints", "dev_evals",
    "ema", "timing", "resume_events", "budget_state",
)


def _meta() -> dict:
    return {
        "spec_digest": "0" * 64,
        "shard_digests": ["a" * 64],
        "split_sha256": "s" * 64,
        "norm_params": {"s_g_hartree_per_angstrom": 0.0154,
                        "s_delta_e": 0.0789},
    }


def _legacy_arm(**overrides) -> CampaignArmConfig:
    base = dict(
        arm_id="D0",
        dataset=CampaignDatasetSpec(kind="legacy"),
        head="bc_nll_legacy",
        sampler="group_balanced_v41",
        preset="legacy_v41",
        seed=42,
        updates=40,
        batch_states=8,
        eval_points=(20, 40),
        curve_every=1,
        device="cpu",
        require_cuda=False,
        light=True,
    )
    base.update(overrides)
    return CampaignArmConfig(**base)


def make_dev_population(seed: int = 99) -> list[dict]:
    rng = np.random.default_rng(seed)
    states = []
    for g in ("RXN_DEV_A", "RXN_DEV_B"):
        for br in ("brA", "brB"):
            for k in range(2):
                direction = rng.normal(size=(6, 3))
                direction /= np.maximum(
                    np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
                magnitude = rng.uniform(0.008, 0.045, size=(6, 1))
                action = direction * magnitude
                states.append(make_state(
                    rng, g, br, state_id=f"{g}_{br}_{k}", action=action,
                    remaining_arc=8.0 - 2.0 * k))
    return states


# ---------------------------------------------------------------------------
# 1. Legacy numeric path vs the old DualTSBCTrainer (bc_nll)
# ---------------------------------------------------------------------------

def test_legacy_preset_matches_old_trainer(tmp_path):
    states = make_population()
    old_settings = DualTSBCSettings(
        recipe="bc_nll", budget_updates=50, batch_states=8, seed=42,
        device="cpu", require_cuda=False, curve_every=1, eval_every=0)
    old = DualTSBCTrainer(
        settings=old_settings, states=states,
        data_meta={"spec_digest": "0" * 64, "shard_digests": [],
                   "norm_params": {"s_g_hartree_per_angstrom": 0.0154,
                                   "s_delta_e": 0.0789}},
        run_dir=tmp_path / "old")
    old_record = old.train()

    arm = _legacy_arm(updates=50, eval_points=(), seed=42, batch_states=8)
    new = CampaignTrainer(
        arm=arm, states=states, data_meta=_meta(),
        run_dir=tmp_path / "new")
    new_record = new.train()

    old_nll = [h["nll_loss"] for h in old_record["history"]]
    new_nll = [h["nll_loss"] for h in new_record["history"]]
    assert len(old_nll) == len(new_nll) == 50
    assert np.allclose(old_nll, new_nll, atol=1e-5, rtol=1e-5)
    assert np.allclose([h["loss"] for h in old_record["history"]],
                       [h["loss"] for h in new_record["history"]],
                       atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# 2. Full flow: 50 steps -> checkpoints -> run_record v3 -> dev metrics
# ---------------------------------------------------------------------------

def test_full_flow_run_record_v3(tmp_path):
    states = make_population()
    dev = make_dev_population()
    arm = _legacy_arm(updates=50, eval_points=(20, 40, 50))
    trainer = CampaignTrainer(
        arm=arm, states=states, data_meta=_meta(), run_dir=tmp_path,
        dev_states=dev)
    record = trainer.train()

    assert record["schema_version"] == RUN_V3
    for key in RUN_V3_KEYS:
        assert key in record, f"missing run_record v3 key {key}"
    assert record["arm_id"] == "D0" and record["preset"] == "legacy_v41"
    assert record["accounting"]["updates_run"] == 50

    # independent per-update checkpoints (raw + EMA identities)
    assert len(record["checkpoints"]) == 3
    for ck in record["checkpoints"]:
        assert ck["update"] in (20, 40, 50)
        assert (tmp_path / ck["dir"] / "raw.pt").is_file()
        assert (tmp_path / ck["dir"] / "ema.pt").is_file()
    assert {ck["update"] for ck in record["checkpoints"]} == {20, 40, 50}

    # dev evals at each checkpoint point
    assert set(record["dev_evals"]) == {"u20", "u40", "u50"}
    summary = record["dev_evals"]["u50"]["summary"]
    for key in ("mse_ratio_median", "direction_cosine_median",
                "near_ts_cosine_median", "pooled_mse_ratio",
                "near_zero_rmse_pooled_A"):
        assert key in summary
    assert summary["n_groups"] == 2
    # EMA + budget state present
    assert record["ema"]["decay"] == 0.999
    assert record["budget_state"]["status"] == "completed"
    assert record["resume_events"] == []


# ---------------------------------------------------------------------------
# 3. Resume determinism (interrupted + resumed == uninterrupted)
# ---------------------------------------------------------------------------

def test_resume_is_deterministic(tmp_path):
    states = make_population()
    dev = make_dev_population()

    # uninterrupted reference
    ref = CampaignTrainer(
        arm=_legacy_arm(updates=40, eval_points=(20, 40)),
        states=states, data_meta=_meta(), run_dir=tmp_path / "ref",
        dev_states=dev)
    ref_record = ref.train()

    # interrupted run (20 steps -> checkpoint u20)
    run_dir = tmp_path / "resume"
    first = CampaignTrainer(
        arm=_legacy_arm(updates=20, eval_points=(20,)),
        states=states, data_meta=_meta(), run_dir=run_dir, dev_states=dev)
    first_record = first.train()
    assert first_record["accounting"]["updates_run"] == 20

    # resume in a fresh trainer (same seed/config/run_dir)
    second = CampaignTrainer(
        arm=_legacy_arm(updates=40, eval_points=(40,)),
        states=states, data_meta=_meta(), run_dir=run_dir, dev_states=dev)
    second.resume()
    second_record = second.train()
    assert second_record["accounting"]["updates_run"] == 40
    assert len(second.resume_events) == 1

    # concatenated loss sequence matches the uninterrupted run exactly
    ref_loss = [h["loss"] for h in ref_record["history"]]
    merged = ([h["loss"] for h in first_record["history"]]
              + [h["loss"] for h in second_record["history"]])
    assert len(merged) == len(ref_loss) == 40
    assert np.allclose(merged, ref_loss, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# 4. Sampler interface parity (epoch_traversal vs group_balanced)
# ---------------------------------------------------------------------------

def test_sampler_parity_and_epoch_traversal_resume():
    states = make_population()
    groups = build_sampler_groups(
        [s for s in states if not s["is_teacher"]],
        group_key=lambda s: s["reaction_group_id"],
        branch_key=lambda s: s["branch_id"],
        arc_key=lambda s, _i: s["remaining_arc_length"])
    gb = DualTSGroupBranchSampler(seed=7, groups=groups, rows_per_group=1)
    et = EpochTraversalSampler(seed=7, states=states, batch_states=8)

    gb_rows = gb.draw_batch(8)
    et_rows = et.draw_batch(8)
    assert len(gb_rows) == len(et_rows) == 8
    assert all(isinstance(i, int) for i in gb_rows)
    assert all(isinstance(i, int) for i in et_rows)
    assert all(0 <= i < 18 for i in et_rows)

    # epoch traversal: one full epoch draws every drawable state exactly once
    stream = et.draw_batch(18 - 8)  # complete the first 18 draws
    first_epoch = et_rows + stream
    assert sorted(first_epoch) == list(range(18))
    # resume round-trip
    et2 = EpochTraversalSampler(seed=7, states=states, batch_states=8)
    et2.load_state_dict(et.state_dict())
    assert et2.draw_batch(10) == et.draw_batch(10)


# ---------------------------------------------------------------------------
# 5. Budget censoring (hard deadline -> save then stop)
# ---------------------------------------------------------------------------

def test_budget_censored(tmp_path):
    states = make_population()
    trainer = CampaignTrainer(
        arm=_legacy_arm(updates=50, eval_points=()),
        states=states, data_meta=_meta(), run_dir=tmp_path,
        deadline=time_time_minus(1))
    record = trainer.train()
    assert record["budget_state"]["status"] == "budget_censored"
    assert record["accounting"]["updates_run"] == 0
    # a checkpoint was saved before terminating
    assert (tmp_path / "checkpoints" / "u0" / "raw.pt").is_file()


def time_time_minus(seconds: float) -> float:
    import time
    return time.time() - seconds


# ---------------------------------------------------------------------------
# 6. timing_only probe
# ---------------------------------------------------------------------------

def test_timing_only(tmp_path):
    states = make_population()
    dev = make_dev_population()
    trainer = CampaignTrainer(
        arm=_legacy_arm(updates=40, eval_points=(20, 40)),
        states=states, data_meta=_meta(), run_dir=tmp_path, dev_states=dev)
    record = trainer.timing_probe(full_updates=40)
    assert record["timing_only"] is True
    assert record["schema_version"] == RUN_V3
    assert "steps_per_sec" in record["timing"]
    assert "estimated_full_seconds" in record["estimate"]
    assert "dev_eval_sample" in record
    # the probe ran 100 warmup + 500 timed steps
    assert record["timing"]["steps_per_sec"] > 0


# ---------------------------------------------------------------------------
# 7. Displacement head loss + dev eval (duck-typed fake model)
# ---------------------------------------------------------------------------

class FakeDisplacementModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.readout = torch.nn.Linear(5, 3)

    def forward(self, inputs, a_phys=None):
        ns = inputs["node_scalars"]
        a_hat = self.readout(ns)
        if a_phys is not None:
            a_hat = a_hat + a_phys
        return {"a_hat": a_hat}


def test_displacement_loss_and_dev_eval(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "pes2ts_core.pesn.offline.training.dual_ts_campaign.build_displacement_policy",
        lambda *a, **k: FakeDisplacementModel())
    states = make_population()
    dev = make_dev_population()
    arm = CampaignArmConfig(
        arm_id="H2", dataset=CampaignDatasetSpec(kind="native"),
        head="displacement", sampler="group_balanced_v41",
        preset="campaign_v1", seed=7, updates=10, batch_states=8,
        eval_points=(5, 10), lambda_dir=0.3, lambda_mag=0.1,
        curve_every=1, device="cpu", require_cuda=False)
    trainer = CampaignTrainer(
        arm=arm, states=states, data_meta=_meta(), run_dir=tmp_path,
        dev_states=dev)
    record = trainer.train()
    assert record["accounting"]["updates_run"] == 10
    # auxiliary loss components present when lambda is set
    for h in record["history"]:
        assert "l_xyz" in h and "l_dir" in h and "l_mag" in h
    assert record["accounting"]["a_s_angstrom"] > 0
    summary = record["dev_evals"]["u10"]["summary"]
    assert summary["direction_cosine_median"] is not None


def test_per_atom_loss_building_blocks():
    torch_ = torch
    a_star = torch_.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    # cosine loss: aligned -> 0, anti-aligned -> 2, orthogonal -> 1
    aligned = per_atom_cos_dir_loss(a_star, a_star)
    assert torch_.allclose(aligned, torch_.zeros(2), atol=1e-6)
    flipped = per_atom_cos_dir_loss(-a_star, a_star)
    assert torch_.allclose(flipped, torch_.full((2,), 2.0), atol=1e-6)
    # magnitude loss: identical magnitude -> 0; 2x -> 0.5 log(2)^2 (Huber quad)
    same = per_atom_mag_loss(a_star, a_star)
    assert torch_.allclose(same, torch_.zeros(2), atol=1e-6)
    doubled = per_atom_mag_loss(2.0 * a_star, a_star)
    import math
    expected = 0.5 * (math.log(2.0)) ** 2
    assert torch_.allclose(doubled, torch_.full((2,), expected), atol=1e-5)


def test_compute_a_s():
    states = make_population()
    a_s = compute_a_s([s for s in states if not s["is_teacher"]])
    assert a_s > 0
    # a_s is the median non-zero atom action norm of the population
    norms = []
    for s in states:
        if s["action"] is None:
            continue
        n = np.linalg.norm(s["action"], axis=1)
        norms.extend(n[n > 0].tolist())
    assert a_s == pytest.approx(float(np.median(norms)), rel=1e-6)


def test_physical_prior_action():
    masses = lambda z: np.ones(len(z), dtype=float)
    state = {"g_scaled": np.array([[2.0, 0, 0], [0.0, 2.0, 0]]),
             "atomic_numbers": np.array([1, 1])}
    out = compute_physical_prior_action(
        state, s_g=1.0, eps_g=0.0, tau_g=0.0, delta_s_fit=0.5,
        masses_for=masses)
    assert out.shape == (2, 3)
    g = np.array([[2.0, 0, 0], [0.0, 2.0, 0]])
    cos = (out * g).sum() / (np.linalg.norm(out) * np.linalg.norm(g))
    assert cos == pytest.approx(1.0)  # direction preserved
    assert np.linalg.norm(out) == pytest.approx(0.5)  # scaled by delta_s_fit
    # below tau_g -> zero action
    zero = compute_physical_prior_action(
        state, s_g=1.0, eps_g=0.0, tau_g=1e6, delta_s_fit=0.5,
        masses_for=masses)
    assert np.allclose(zero, 0.0)


# ---------------------------------------------------------------------------
# 8. Unified npz label loading (native + legacy)
# ---------------------------------------------------------------------------

def _write_shard(features_dir: Path, gid: str, rid: str, *,
                 n_states: int = 3, n_atoms: int = 4):
    rng = np.random.default_rng(123)
    z = np.array([1, 6, 8, 1], dtype=np.int64)[:n_atoms]
    ns = np.zeros((n_states, n_atoms, 5))
    ns[..., 0] = z[None, :]
    x = rng.normal(size=(n_states, n_atoms, 3))
    g = rng.normal(size=(n_states, n_atoms, 3))
    atom_a = rng.normal(size=(n_atoms, 4))
    atom_b = rng.normal(size=(n_atoms, 4))
    edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
    edge_a = rng.normal(size=(4, 4))
    edge_b = rng.normal(size=(4, 4))
    action = rng.normal(size=(n_states, n_atoms, 3)) * 0.02
    is_terminal = np.zeros(n_states, dtype=bool)
    is_terminal[-1] = True
    ral = np.linspace(5.0, 0.0, n_states)
    state_id = np.array([f"{rid}_s{i}" for i in range(n_states)])
    branch_id = np.array(["brA"] * n_states)
    episode_id = np.array([f"EP_{rid}_brA"] * n_states)
    step_index = np.arange(n_states)
    np.savez(
        features_dir / f"{gid}.npz",
        reaction_id=np.array(rid), reaction_group_id=np.array(gid),
        atomic_numbers=z, node_scalars=ns, x_angstrom=x, g_scaled=g,
        atom_feats_a=atom_a, atom_feats_b=atom_b,
        edge_feats_a=edge_a, edge_feats_b=edge_b, edge_index=edge_index,
        state_id=state_id, branch_id=branch_id, episode_id=episode_id,
        step_index=step_index,
        action_angstrom=action, is_terminal=is_terminal,
        remaining_arc_length=ral)
    return {"reaction_group_id": gid, "npz_file": f"{gid}.npz"}


def _write_manifest(features_dir: Path, rows: list[dict]) -> None:
    (features_dir / "features_manifest.json").write_text(
        json.dumps({"per_reaction": rows}))


def test_load_campaign_states_native(tmp_path):
    features = tmp_path / "native"
    features.mkdir()
    rows = [_write_shard(features, "G1", "RXN_N1", n_states=3)]
    _write_manifest(features, rows)
    dataset = CampaignDatasetSpec(kind="native", features_dir=str(features),
                                  labels_dir="", split_path="")
    states, meta = load_campaign_states(dataset, group_ids=["G1"])
    assert meta["n_states"] == 3
    assert meta["n_action_states"] == 2  # last state terminal
    assert states[0]["is_teacher"] is False
    assert states[2]["is_teacher"] is True and states[2]["action"] is None
    assert states[0]["action"].shape == (4, 3)
    assert states[0]["near_zero_mask"].shape == (4,)


def test_load_campaign_states_legacy(tmp_path):
    features = tmp_path / "features"
    labels = tmp_path / "legacy_labels"
    features.mkdir()
    labels.mkdir()
    rows = [_write_shard(features, "G2", "RXN_L1", n_states=3)]
    _write_manifest(features, rows)
    # legacy labels live in a separate npz keyed by reaction id
    rng = np.random.default_rng(1)
    np.savez(labels / "legacy_labels_RXN_L1.npz",
             action_angstrom=rng.normal(size=(3, 4, 3)) * 0.02,
             is_terminal=np.array([False, False, True]),
             remaining_arc_length=np.array([5.0, 2.5, 0.0]))
    dataset = CampaignDatasetSpec(kind="legacy", features_dir=str(features),
                                  labels_dir=str(labels), split_path="")
    states, meta = load_campaign_states(dataset, group_ids=["G2"])
    assert meta["dataset_kind"] == "legacy"
    assert meta["n_states"] == 3
    assert states[2]["is_teacher"] is True and states[2]["action"] is None
    assert states[0]["remaining_arc_length"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# 9. Checkpoint raw + EMA dual identity
# ---------------------------------------------------------------------------

def test_checkpoint_dual_identity(tmp_path):
    states = make_population()
    dev = make_dev_population()
    arm = _legacy_arm(updates=20, eval_points=(10, 20))
    trainer = CampaignTrainer(
        arm=arm, states=states, data_meta=_meta(), run_dir=tmp_path,
        dev_states=dev)
    record = trainer.train()
    assert len(record["checkpoints"]) == 2
    for ck in record["checkpoints"]:
        raw = torch.load(tmp_path / ck["dir"] / "raw.pt",
                         map_location="cpu", weights_only=False)
        ema = torch.load(tmp_path / ck["dir"] / "ema.pt",
                         map_location="cpu", weights_only=False)
        assert raw["schema_version"] == "pesn_dual_ts_campaign_checkpoint_v1"
        assert ema["schema_version"] == "pesn_dual_ts_campaign_ema_v1"
        assert raw["update"] == ck["update"]
        assert "model_state" in raw and "optimizer_state" in raw
        assert "ema_state" in raw and "sampler_state" in raw
        assert "model_state" in ema  # ema-applied weights


# ---------------------------------------------------------------------------
# 10. E2 key-pair conditioning (reconstruction + wrapper)
# ---------------------------------------------------------------------------

def test_keypair_reconstruct_and_records():
    from pes2ts_core.pesn.offline.training.dual_ts_campaign import (
        _reconstruct_endpoint_graphs,
    )
    from pes2ts_core.pesn.features.dual_ts_graph_v4 import (
        build_key_pair_records,
        select_key_pairs,
    )
    n = 4
    edge_pairs = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)

    def edge_feats(bonds: dict) -> np.ndarray:
        E = len(edge_pairs)
        rows = np.zeros((E, 4))
        for k, (i, j) in enumerate(edge_pairs):
            o = bonds.get((i, j))
            if o is not None:
                rows[k, 0] = 1.0       # bond_present
                rows[k, 1] = float(o)  # bond_order
        return np.concatenate([rows, rows], axis=0)  # directed expansion

    state = {
        "atomic_numbers": np.array([6, 1, 8, 1], dtype=np.int64),
        "edge_pairs": edge_pairs,
        "edge_feats_a": edge_feats({(0, 1): 1.0, (1, 2): 1.0}),
        "edge_feats_b": edge_feats({(0, 1): 1.0, (1, 2): 2.0, (2, 3): 1.0}),
    }
    ga, gb = _reconstruct_endpoint_graphs(state)
    key_pairs = select_key_pairs(ga, gb)
    # (1,2) order changed 1.0->2.0; (2,3) absent->present
    assert key_pairs.shape == (2, 2)
    assert set(map(tuple, key_pairs.tolist())) == {(1, 2), (2, 3)}

    x = np.random.default_rng(0).normal(size=(n, 3))
    records = build_key_pair_records(
        key_pairs=key_pairs, x=x, topology_a=ga, topology_b=gb)
    assert records.shape == (2, 40)


def test_keypair_wrapped_encoder():
    from pes2ts_core.pesn.offline.training.dual_ts_campaign import (
        KeyPairWrappedEncoder,
    )

    class FakeEncoder(torch.nn.Module):
        def forward(self, inputs):
            return {"h": inputs["h"], "v": inputs["v"]}

    hidden = 8
    wrapped = KeyPairWrappedEncoder(FakeEncoder(), hidden=hidden)
    assert wrapped.hidden == hidden
    h = torch.randn(4, hidden)
    v = torch.randn(4, 3)
    empty = {"h": h, "v": v, "pair_records": torch.zeros(0, 40),
             "pair_indices": torch.zeros(0, 2, dtype=torch.long)}
    out = wrapped(empty)
    assert torch.allclose(out["h"], h) and torch.allclose(out["v"], v)
    # with a key pair: h is modulated (FiLM identity at init => ~h) and shaped
    one = {"h": h, "v": v, "pair_records": torch.randn(1, 40),
           "pair_indices": torch.tensor([[0, 1]])}
    out2 = wrapped(one)
    assert out2["h"].shape == h.shape and torch.isfinite(out2["h"]).all()
    # conditioner parameters are registered and counted
    n_extra = sum(p.numel() for p in wrapped.conditioner.parameters())
    assert n_extra > 0


# ---------------------------------------------------------------------------
# 11. Config guards
# ---------------------------------------------------------------------------

def test_arm_config_guards():
    with pytest.raises(CampaignTrainerError):
        CampaignArmConfig(head="bogus")
    with pytest.raises(CampaignTrainerError):
        CampaignArmConfig(head="displacement", preset="legacy_v41")
    with pytest.raises(CampaignTrainerError):
        CampaignArmConfig(lambda_dir=-1.0)
    with pytest.raises(CampaignTrainerError):
        CampaignArmConfig(eval_points=(5000,), updates=4000)
    # round-trip
    arm = CampaignArmConfig(
        arm_id="D2", dataset=CampaignDatasetSpec(kind="legacy"),
        head="displacement", lambda_dir=0.3, lambda_mag=0.1,
        features_version="v4", encoder="painn", keypair_condition=True)
    arm2 = CampaignArmConfig.from_payload(arm.to_payload())
    assert arm2.to_payload() == arm.to_payload()


# ---------------------------------------------------------------------------
# 12. Driver alias resolution (reuse arms -> concrete run arm)
# ---------------------------------------------------------------------------

def test_resolve_run_arm_alias_chain():
    import scripts.run_v42_campaign as drv
    decisions = {"sampler_winner": None, "winner_of": {}, "configs": {}}
    # reuse arms resolve to the concrete arm holding the run dirs
    assert drv._resolve_run_arm("H0", decisions, drv.ARMS) == "D3"
    assert drv._resolve_run_arm("D3", decisions, drv.ARMS) == "D3"
    assert drv._resolve_run_arm("H2", decisions, drv.ARMS) == "H2"
    # stage-winner labels resolve via the arm that stores the winner
    decisions["winner_of"]["E1"] = "H2"
    assert drv._resolve_run_arm("E0", decisions, drv.ARMS) == "H2"
    decisions["winner_of"]["W96"] = "V1"
    decisions["winner_of"]["W128"] = "V1"
    assert drv._resolve_run_arm("W64", decisions, drv.ARMS) == "V1"
    decisions["winner_of"]["V1"] = "E2"
    assert drv._resolve_run_arm("V0", decisions, drv.ARMS) == "E2"
    # unresolved reference -> SystemExit (clear error, chain included)
    with pytest.raises(SystemExit):
        drv._resolve_run_arm("BOGUS", decisions, drv.ARMS)
    # cycle detection
    decisions["winner_of"]["E1"] = "E0"  # E0 -> E1 -> E0
    with pytest.raises(SystemExit):
        drv._resolve_run_arm("E0", decisions, drv.ARMS)


def test_apply_promotion_with_reuse_alias(tmp_path, monkeypatch):
    """H0 (reuse D3) has no H0_seed* run dirs; apply_promotion must resolve
    it to D3 and return a decision, never NO_SEEDS."""
    import scripts.run_v42_campaign as drv
    # route run dirs into the temp dir
    monkeypatch.setattr(drv, "_run_dir",
                        lambda arm_id, seed: tmp_path / f"{arm_id}_seed{seed}")
    # fake Lane-D promotion (deterministic promote, asserts seeds present)
    def fake_promo(parent_dev, child_dev, seeds):
        assert len(seeds) >= 1, "NO_SEEDS: promotion got no seeds"
        return {"promote": True, "decision": "PROMOTE",
                "reasons": [], "per_seed": {}}
    monkeypatch.setattr(
        "pes2ts_core.pesn.offline.evaluation.dual_ts_promotion"
        ".apply_promotion_rule", fake_promo)

    summary = {"direction_cosine_median": 0.5, "near_ts_cosine_median": 0.5,
               "mse_ratio_median": 0.8, "n_groups_ratio_below_one": 5,
               "pooled_mse_ratio": 0.7, "near_zero_rmse_pooled_A": 0.01}
    # synthetic D3 (H0 reuse target) + H2 double-seed run records
    for arm_id in ("D3", "H2"):
        for seed in (20261017, 20261018):
            d = tmp_path / f"{arm_id}_seed{seed}"
            d.mkdir(parents=True)
            (d / "run_record.json").write_text(json.dumps({
                "schema_version": "pesn_dual_ts_bc_run_v3",
                "dev_evals": {"u20000": {"summary": summary}}}))

    driver = drv.CampaignDriver(out_root=tmp_path)
    # resolve the reuse/config chain so _config_payload("H2") is valid
    driver.decisions["configs"]["D3"] = driver._config_payload("D3")
    driver.decisions["configs"]["H0"] = driver._config_payload("H0")
    driver.decisions["winner_of"]["H2"] = "H0"

    winner = driver.apply_promotion("H0", "H2", "E1")
    assert winner == "H2"


# ---------------------------------------------------------------------------
# 13. H2 λ injection (CALIBRATED placeholder -> canary values)
# ---------------------------------------------------------------------------

def test_h2_lambda_injection():
    import scripts.run_v42_campaign as drv
    lam_dir, lam_mag = 16311.868760193809, 56.20006672262769
    decisions = {"sampler_winner": "group_balanced_v41",
                 "winner_of": {}, "configs": {}}
    # resolve D3 then H0 (reuse D3), and set H2's parent to H0
    d3 = drv._resolve_arm(drv.ARMS["D3"], decisions, seed=20261017)
    decisions["configs"]["D3"] = drv._flat_config(d3)
    decisions["configs"]["H0"] = drv._flat_config(
        drv._resolve_arm(drv.ARMS["H0"], decisions, seed=20261017))
    decisions["winner_of"]["H2"] = "H0"
    # inject the canary lambdas (mirrors run_campaign's direct assignment)
    decisions["configs"]["H2"] = dict(decisions["configs"]["H0"])
    decisions["configs"]["H2"]["lambda_dir"] = lam_dir
    decisions["configs"]["H2"]["lambda_mag"] = lam_mag
    h2 = drv._resolve_arm(drv.ARMS["H2"], decisions, seed=20261017)
    assert h2.lambda_dir == pytest.approx(lam_dir)
    assert h2.lambda_mag == pytest.approx(lam_mag)
    # regression: WITHOUT the driver's injected config, H2 resolves to
    # lambda None (the pre-fix silent H0-replica bug)
    no_lam = {"sampler_winner": "group_balanced_v41",
              "winner_of": {"H2": "H0"}, "configs": dict(decisions["configs"])}
    no_lam["configs"].pop("H2", None)
    h2_none = drv._resolve_arm(drv.ARMS["H2"], no_lam, seed=20261017)
    assert h2_none.lambda_dir is None and h2_none.lambda_mag is None


def test_resolve_lambdas_missing_canary(monkeypatch, tmp_path):
    import scripts.run_v42_campaign as drv
    monkeypatch.setattr(drv, "LAMBDA_CANARY_PATH",
                        tmp_path / "nonexistent_lambda_canary.json")
    with pytest.raises(SystemExit):
        drv._resolve_lambdas()
