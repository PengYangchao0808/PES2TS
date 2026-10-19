"""V5.1a trainer gates (plan §3.2–§3.4; G5.1 fixture for the v51 lane).

CPU-fast gates for the core v51 lane:

* inner-val split: determinism, sha256 hash, 32/8-style stratification,
  real-manifest behaviour on the frozen V5 5-fold split;
* best-checkpoint selection rule (frozen plan §3.4: rmse ≤ R*×1.02 →
  lowest SRE → highest near-TS cos) exercised directly on synthetic eval
  sequences;
* patience stopper (8 checks / min_delta 0.005 / floor update 500);
* arm resolution: B0–B3 table-driven entries (explicit anchor bypasses
  the A/C prefix rule), magnitude-head ⇒ anchor-on, pair_loss switch;
* trainer construction on CPU for modes screen|formal|v51 (both config
  schemas), CLI ``--mode v51``, gv campaign-id plumbing;
* a tiny SYNTHETIC end-to-end v51 fit (Lane B monkeypatched) verifying:
  run_record / metrics.json / eval.jsonl / checkpoint_best.pt contracts
  (driver-lane key spellings), ``selection_exposed: false``, post-hoc
  outer-dev rows tagged ``dev_dense_best`` / ``dev_dense_final``, and a
  legacy screen-mode regression fit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pes2ts_core.pesn.contracts.key_atom import (
    EDGE_DIM,
    EDGE_SLOTS,
    NODE_INV_DIM,
    TASK_DIM,
    KeyAtomBatch,
)
from pes2ts_core.pesn.models import require_torch
from pes2ts_core.pesn.offline import training as _training_pkg  # noqa: F401
from pes2ts_core.pesn.offline.training import key_atom_bc as kbc

torch = require_torch()

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Pure gates: inner split
# ---------------------------------------------------------------------------

def _synthetic_clusters(n: int = 40):
    ids = [f"RXN_{i:04d}" for i in range(n)]
    family_of = {g: f"family_{i % 7}" for i, g in enumerate(ids)}
    size_of = {g: (i % 3) + 1 for i, g in enumerate(ids)}
    return ids, family_of, size_of


def test_inner_split_determinism_and_hash():
    ids, fam, size = _synthetic_clusters(40)
    tr1, va1, h1 = kbc.inner_val_split(
        ids, family_of=fam, size_of=size, n_val=8, seed=20261019)
    tr2, va2, h2 = kbc.inner_val_split(
        ids, family_of=fam, size_of=size, n_val=8, seed=20261019)
    assert tr1 == tr2 and va1 == va2 and h1 == h2
    assert len(va1) == 8 and len(tr1) == 32
    assert not set(tr1) & set(va1)
    assert sorted(tr1 + va1) == sorted(ids)
    assert len(h1) == 64 and all(c in "0123456789abcdef" for c in h1)
    # different seed → different split (extremely likely) and hash
    tr3, va3, h3 = kbc.inner_val_split(
        ids, family_of=fam, size_of=size, n_val=8, seed=1)
    assert h3 != h1
    assert va3 != va1
    # input order must not matter (sorted inside)
    tr4, va4, h4 = kbc.inner_val_split(
        list(reversed(ids)), family_of=fam, size_of=size, n_val=8,
        seed=20261019)
    assert tr4 == tr1 and va4 == va1 and h4 == h1


def test_inner_split_stratifies_families():
    """Every family with ≥1 val slot keeps its stride spacing — the val
    set touches the MAJORITY of families (systematic sampling over the
    family-ordered list)."""
    ids, fam, size = _synthetic_clusters(40)
    _tr, va, _h = kbc.inner_val_split(
        ids, family_of=fam, size_of=size, n_val=8, seed=20261019)
    families_in_val = {fam[g] for g in va}
    assert len(families_in_val) >= 5, families_in_val


def test_inner_split_on_real_manifest():
    """The frozen V5 split manifest (50 fit50 clusters) yields a
    deterministic 40→32/8 inner split for fold 0."""
    manifest = REPO / "data/manifests/v5/split_manifest.json"
    if not manifest.is_file():
        pytest.skip("V5 split manifest not present")
    family_of, size_of = kbc._cluster_family_size(manifest)
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    group_fold = doc["group_fold"]
    train_ids = sorted(g for g, f in group_fold.items() if int(f) != 0)
    assert len(train_ids) == 40
    tr, va, h = kbc.inner_val_split(
        train_ids, family_of=family_of, size_of=size_of, n_val=8,
        seed=20261019)
    tr2, va2, h2 = kbc.inner_val_split(
        train_ids, family_of=family_of, size_of=size_of, n_val=8,
        seed=20261019)
    assert (tr, va, h) == (tr2, va2, h2)
    assert len(va) == 8 and not set(tr) & set(va)
    assert set(va) <= set(train_ids)


def test_inner_split_invalid_n_val():
    ids, fam, size = _synthetic_clusters(10)
    with pytest.raises(kbc.KeyAtomBCTrainerError):
        kbc.inner_val_split(ids, family_of=fam, size_of=size, n_val=10)
    with pytest.raises(kbc.KeyAtomBCTrainerError):
        kbc.inner_val_split(ids, family_of=fam, size_of=size, n_val=0)


# ---------------------------------------------------------------------------
# Pure gates: selection rule + patience (plan §3.4, frozen)
# ---------------------------------------------------------------------------

P = kbc.V51EvalPoint


def test_select_best_within_1p02_band_prefers_sre():
    points = [
        P(100, 0.60, 0.30, 0.70),
        P(200, 0.50, 0.28, 0.71),          # R* = 0.50
        P(300, 0.51, 0.10, 0.60),          # ≤ 0.50*1.02, much lower SRE
        P(400, 0.55, 0.05, 0.90),          # OUTSIDE the band (0.55 > 0.51)
    ]
    best = kbc.select_best_v51(points)
    assert best.update == 300


def test_select_best_new_rstar_evicts_old_band():
    points = [
        P(100, 0.60, 0.30, 0.70),
        P(200, 0.50, 0.28, 0.71),
        P(300, 0.51, 0.10, 0.60),          # in band of R*=0.50
        P(400, 0.40, 0.25, 0.75),          # R* drops to 0.40 → 300 evicted
    ]
    best = kbc.select_best_v51(points)
    assert best.update == 400


def test_select_best_cos_tiebreak_then_update():
    points = [
        P(100, 0.50, 0.20, 0.70),
        P(150, 0.50, 0.20, 0.80),          # equal SRE → higher cos wins
    ]
    assert kbc.select_best_v51(points).update == 150
    tied = [
        P(200, 0.50, 0.20, 0.75),
        P(180, 0.50, 0.20, 0.75),          # full tie → earliest update
    ]
    assert kbc.select_best_v51(tied).update == 180


def test_select_best_none_metrics_sort_worst():
    points = [
        P(100, 0.50, None, None),          # SRE undefined → worst
        P(200, 0.501, 0.40, None),         # in band; defined SRE wins
    ]
    assert kbc.select_best_v51(points).update == 200
    only_none = [P(100, 0.50, None, None), P(200, 0.49, None, None)]
    assert kbc.select_best_v51(only_none).update == 200  # best rmse only
    assert kbc.select_best_v51([]) is None
    assert kbc.select_best_v51([P(1, None, None, None)]) is None
    assert kbc.select_best_v51([P(1, float("nan"), 0.1, 0.1)]) is None


def test_best_tracker_recomputes_selection():
    tracker = kbc.V51BestTracker(qualify_factor=1.02)
    b1 = tracker.add(P(100, 0.60, 0.30, 0.70))
    assert b1.update == 100 and tracker.r_star == 0.60
    b2 = tracker.add(P(200, 0.50, 0.28, 0.71))
    assert b2.update == 200 and tracker.r_star == 0.50
    b3 = tracker.add(P(300, 0.505, 0.10, 0.60))
    assert b3.update == 300
    b4 = tracker.add(P(400, 0.40, 0.25, 0.75))
    assert b4.update == 400  # 300 evicted when R* drops


def test_early_stopper_floor_and_patience():
    stopper = kbc.V51EarlyStopper(patience=3, min_delta=0.005,
                                  floor_update=500)
    assert not stopper.update(0.60, at_update=100)   # first check: baseline
    assert not stopper.update(0.599, at_update=150)  # bad 1
    assert not stopper.update(0.598, at_update=300)  # bad 2
    assert not stopper.update(0.597, at_update=450)  # bad 3 (below floor)
    assert not stopper.update(0.596, at_update=499)  # bad 4 (below floor)
    assert stopper.update(0.595, at_update=500)      # bad 5 ≥ patience+floor
    # improvement ≥ min_delta resets the counter
    stopper2 = kbc.V51EarlyStopper(patience=2, min_delta=0.005,
                                   floor_update=500)
    assert not stopper2.update(0.60, at_update=100)
    assert not stopper2.update(0.598, at_update=200)
    assert not stopper2.update(0.50, at_update=300)  # big improvement
    assert stopper2.bad_checks == 0
    assert not stopper2.update(0.499, at_update=400)          # bad 1
    assert stopper2.update(0.4985, at_update=500)             # bad 2 + floor
    # non-finite / None checks never stop
    stopper3 = kbc.V51EarlyStopper(patience=1, min_delta=0.005,
                                   floor_update=500)
    assert not stopper3.update(None, at_update=600)
    assert not stopper3.update(float("nan"), at_update=700)
    assert stopper3.bad_checks == 0


# ---------------------------------------------------------------------------
# Arm resolution (driver contract: B0–B3 table-driven)
# ---------------------------------------------------------------------------

def _arm_config(**arm_over):
    table = {
        "A05": {"d0": 0.05, "anchor": "on"},
        "C05": {"d0": 0.05, "anchor": "off"},
        "B0": {"d0": 0.05, "anchor": "on", "magnitude_head": False,
               "pair_loss": "legacy_j"},
        "B1": {"d0": 0.05, "anchor": "on", "magnitude_head": True,
               "pair_loss": "legacy_j"},
        "B2": {"d0": 0.05, "anchor": "on", "magnitude_head": False,
               "pair_loss": "exact_trio"},
        "B3": {"d0": 0.05, "anchor": "on", "magnitude_head": True,
               "pair_loss": "exact_trio"},
    }
    for arm_id, over in arm_over.items():
        table[arm_id] = {**table[arm_id], **over}
    return {"d0_angstrom": 0.08, "sprint": {"arm_table": table}}


def test_resolve_arm_legacy_prefix_rule_intact():
    cfg = _arm_config()
    a = kbc.resolve_arm(cfg, "A05")
    assert (a.anchor, a.d0, a.magnitude_head, a.pair_loss) == (
        "on", 0.05, False, "legacy_j")
    c = kbc.resolve_arm(cfg, "C05")
    assert c.anchor == "off"
    with pytest.raises(kbc.KeyAtomBCTrainerError) as excinfo:
        kbc.resolve_arm(_arm_config(A05={"anchor": "off"}), "A05")
    assert "ARM_PREFIX_CONFLICT" in str(excinfo.value)


def test_resolve_arm_b_arms_table_driven():
    cfg = _arm_config()
    b3 = kbc.resolve_arm(cfg, "B3")
    assert (b3.anchor, b3.magnitude_head, b3.pair_loss) == (
        "on", True, "exact_trio")
    b0 = kbc.resolve_arm(cfg, "B0")
    assert (b0.magnitude_head, b0.pair_loss) == (False, "legacy_j")
    assert b3.to_payload()["magnitude_head"] is True
    assert b3.to_payload()["pair_loss"] == "exact_trio"


def test_resolve_arm_head_requires_anchor_on():
    cfg = _arm_config(B1={"anchor": "off"})
    with pytest.raises(kbc.KeyAtomBCTrainerError) as excinfo:
        kbc.resolve_arm(cfg, "B1")
    assert "ARM_HEAD_ANCHOR_REQUIRED" in str(excinfo.value)


def test_resolve_arm_pair_loss_validation():
    with pytest.raises(kbc.KeyAtomBCTrainerError) as excinfo:
        kbc.resolve_arm(_arm_config(B2={"pair_loss": "hinge"}), "B2")
    assert "ARM_PAIR_LOSS_INVALID" in str(excinfo.value)
    # config-level default when the entry omits pair_loss
    cfg = {"pair_loss": "exact_trio",
           "sprint": {"arm_table": {"B0": {"d0": 0.05, "anchor": "on"}}}}
    assert kbc.resolve_arm(cfg, "B0").pair_loss == "exact_trio"


# ---------------------------------------------------------------------------
# gv campaign-id plumbing (driver reconciliation item 4)
# ---------------------------------------------------------------------------

def test_build_gv_block_campaign_from_config():
    block = kbc.build_gv_block(
        config_path=REPO / "config/campaigns/v5_sprint.yaml",
        repo_root=REPO, campaign_id="V5_1A_SPRINT")
    # V5_1A_SPRINT is registered in g_v_map (2026-10-19) — resolves directly
    assert block["campaign_id"] == "V5_1A_SPRINT"
    assert block["campaign_id_configured"] == "V5_1A_SPRINT"
    assert block["primary_g"] == "G3.P.2"
    # a genuinely unregistered id still falls back to the V5_SPRINT entry
    fallback = kbc.build_gv_block(
        config_path=REPO / "config/campaigns/v5_sprint.yaml",
        repo_root=REPO, campaign_id="V9_UNREGISTERED")
    assert fallback["campaign_id"] == "V5_SPRINT"
    assert fallback["campaign_id_configured"] == "V9_UNREGISTERED"
    legacy = kbc.build_gv_block(
        config_path=REPO / "config/campaigns/v5_sprint.yaml",
        repo_root=REPO)
    assert legacy["campaign_id"] == "V5_SPRINT"
    assert legacy["campaign_id_configured"] == "V5_SPRINT"


# ---------------------------------------------------------------------------
# Metrics additions: alpha_median, dead-zone fraction, G5.3 enumeration
# ---------------------------------------------------------------------------

def test_compute_cluster_metrics_alpha_median():
    rng = np.random.default_rng(0)
    per_sample = []
    for gid, scale in (("g1", 0.5), ("g2", 1.0)):
        y = rng.normal(scale=0.05, size=(3, 3))
        u = y * scale
        per_sample.append({"group_id": gid, "u": u, "y": y,
                           "u_classic": None, "near_ts": 0.1})
    per_cluster, agg = kbc.compute_cluster_metrics(per_sample)
    assert abs(per_cluster["g1"]["alpha_median"] - 0.5) < 1e-9
    assert abs(per_cluster["g2"]["alpha_median"] - 1.0) < 1e-9
    assert agg["alpha_median"] == pytest.approx(0.75)
    assert agg["alpha_q25"] is not None and agg["alpha_q75"] is not None


def test_g53_dead_zone_fraction_and_constant():
    from pes2ts_core.pesn.offline.evaluation import key_atom_protocols as prot
    assert prot.G53_DEAD_ZONE_MAX == 0.10
    # one active pair correctly signed but INSIDE the dead zone
    pos = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    y = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]])   # bond stretches
    u_dead = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]])  # |Δd|<0.005
    sample = {"group_id": "g", "reaction_id": "g", "branch": 0,
              "pos": pos, "key_rows": [0, 1], "pairs": [(0, 1)],
              "u": u_dead, "y": y}
    rep = prot.evaluate_g53([sample])
    assert rep["aggregate"]["n_active_pairs"] == 1
    assert rep["aggregate"]["n_dead_pairs"] == 1
    assert rep["aggregate"]["dead_zone_fraction"] == 1.0
    u_far = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]])
    rep2 = prot.evaluate_g53([{**sample, "u": u_far}])
    assert rep2["aggregate"]["dead_zone_fraction"] == 0.0
    # no active pairs → None (never a fabricated 0)
    rep3 = prot.evaluate_g53([{**sample, "y": np.zeros_like(y)}])
    assert rep3["aggregate"]["dead_zone_fraction"] is None


def test_g53_sample_enumeration_matches_static_diagnostics_rule():
    """Both endpoints in-frame AND pair_edit ≠ 0 (frozen V5 rule)."""
    from pes2ts_core.pesn.offline.evaluation.key_atom_protocols import (
        g53_sample_enumeration,
    )
    # 2 frames × 4 nodes; keys rows {0,1} and {4,5}
    pos = np.arange(24, dtype=np.float64).reshape(8, 3) * 0.1
    ptr = np.array([0, 4, 8])
    key_index = np.array([0, 1, 4, 5])
    key_ptr = np.array([0, 2, 4])
    pair_index = np.array([
        [0, 1, 4, 0],     # (0,1) in-frame f0; (4,5) in f1; (0,4) cross
        [1, 5, 5, 4]])
    pair_edit = np.array([1.0, 0.0, -1.0, 1.0])  # (1,5) zero-edit → drop
    u = np.ones((4, 3)) * 0.01
    y = np.ones((4, 3)) * 0.02
    samples = g53_sample_enumeration(
        pos=pos, ptr=ptr, key_index=key_index, key_ptr=key_ptr,
        pair_index=pair_index, pair_edit=pair_edit, u=u, y=y,
        group_ids=("RXN_A", "RXN_B"), branches=(0, 1))
    assert len(samples) == 2
    assert samples[0]["pairs"] == [(0, 1)]       # (1,5) dropped: edit=0
    assert samples[1]["pairs"] == [(0, 1)]       # re-based rows of (4,5)
    assert samples[0]["key_rows"] == [0, 1]
    assert samples[1]["key_rows"] == [0, 1]
    assert samples[1]["group_id"] == "RXN_B"
    assert samples[1]["branch"] == 1


# ---------------------------------------------------------------------------
# Trainer construction / CLI
# ---------------------------------------------------------------------------

def _base_config(tmp_path: Path, *, schema: str = "pesn_v5_1_sprint_v1"):
    return {
        "schema": schema,
        "d0_angstrom": 0.05,
        "a_ref_angstrom": 0.08,
        "lambda_pair": 0.5,
        "lambda_cos": 0.02,
        "cos_start_update": 250,
        "cos_ramp_updates": 250,
        "tau_norm_angstrom": 0.02,
        "cos_soft_norm_angstrom": 1e-4,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "warmup_updates": 2,
        "min_lr": 1e-4,
        "batch_size": 8,
        "reactions_per_batch": 2,
        "frames_per_reaction": 2,
        "grad_clip_norm": 1.0,
        "eval_every_updates": 2,
        "early_stop_patience_checks": 8,
        "early_stop_min_delta": 0.005,
        "paths": {"split_manifest": "data/manifests/v5/split_manifest.json",
                  "feature_root": "outputs/pesn/v5/features"},
        "sprint": {"campaign_id": "V5_1A_SPRINT", "arm_table": {
            "B0": {"d0": 0.05, "anchor": "on", "magnitude_head": False,
                   "pair_loss": "legacy_j"},
            "B3": {"d0": 0.05, "anchor": "on", "magnitude_head": True,
                   "pair_loss": "exact_trio"},
            "A05": {"d0": 0.05, "anchor": "on"},
        }},
        "v51": {
            "inner_val": {
                "n_val_clusters": 3,
                "seed": 20261019,
                "selection": {"rmse_tolerance": 1.02, "patience_checks": 8,
                              "min_delta": 0.005, "min_update_floor": 500},
            },
            "magnitude_head": {"mode": "per_atom", "lambda_mag": 0.5},
            "pair_exact": {"lambda_sign": 0.2, "lambda_dead": 0.1,
                           "tau_sign_angstrom": 0.005,
                           "tau_dead_angstrom": 0.005,
                           "active_threshold_angstrom": 0.005},
            "ramp": {"start_update": 250, "ramp_updates": 250},
        },
        "device": {"device": "cpu", "require_cuda": False},
    }


def _make_trainer(tmp_path, *, mode: str, arm_id: str, updates: int = 4,
                  schema: str = "pesn_v5_1_sprint_v1", config=None):
    cfg = config if config is not None else _base_config(
        tmp_path, schema=schema)
    arm = kbc.resolve_arm(cfg, arm_id)
    return kbc.KeyAtomBCTrainer(
        config=cfg, config_path=tmp_path / "cfg.yaml", arm=arm, fold=0,
        seed=17, updates=updates, mode=mode, run_dir=tmp_path / "run",
        device_override="cpu", repo_root=REPO)


def test_trainer_accepts_both_schemas_and_v51_mode(tmp_path):
    trainer = _make_trainer(tmp_path, mode="v51", arm_id="B3")
    assert trainer.mode == "v51"
    assert trainer.loss_config.pair_loss == "exact_trio"
    assert trainer.model.magnitude_head is True
    assert trainer.magnitude_mode == "per_atom"
    assert trainer.inner_val_n == 3
    assert trainer.inner_val_seed == 20261019
    assert trainer.v51_qualify_factor == 1.02
    legacy = _make_trainer(tmp_path, mode="screen", arm_id="A05",
                           schema="pesn_v5_sprint_draft2")
    assert legacy.model.magnitude_head is False
    assert legacy.loss_config.pair_loss == "legacy_j"
    with pytest.raises(kbc.KeyAtomBCTrainerError) as excinfo:
        _make_trainer(tmp_path, mode="oof", arm_id="B3")
    assert "MODE_UNKNOWN" in str(excinfo.value)
    # trainer-level belt: an arm spec with head-on + anchor-off is refused
    bad_arm = kbc.KeyAtomArmSpec(
        arm_id="X9", anchor="off", d0=0.05, magnitude_head=True)
    with pytest.raises(kbc.KeyAtomBCTrainerError) as excinfo:
        kbc.KeyAtomBCTrainer(
            config=_base_config(tmp_path), config_path=tmp_path / "cfg.yaml",
            arm=bad_arm, fold=0, seed=17, updates=2, mode="v51",
            run_dir=tmp_path / "run_bad", device_override="cpu",
            repo_root=REPO)
    assert "ARM_HEAD_ANCHOR_REQUIRED" in str(excinfo.value)


def test_cli_mode_choices_include_v51():
    parser = kbc.build_arg_parser()
    args = parser.parse_args(
        ["--arm", "B3", "--fold", "0", "--seed", "17", "--mode", "v51"])
    assert args.mode == "v51"
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--arm", "B3", "--fold", "0", "--seed", "17", "--mode", "oof"])


def test_checkpoint_fields_head_provenance(tmp_path):
    trainer = _make_trainer(tmp_path, mode="v51", arm_id="B3")
    (trainer.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    trainer.updates_run = 7
    trainer._save_checkpoint("checkpoint_last.pt")
    import torch as _torch
    doc = _torch.load(trainer.run_dir / "checkpoints" / "checkpoint_last.pt",
                      map_location="cpu", weights_only=False)
    assert doc["schema_version"] == kbc.CHECKPOINT_SCHEMA
    assert doc["magnitude_head"] is True
    assert doc["magnitude_mode"] == "per_atom"
    assert doc["pair_loss"] == "exact_trio"
    assert doc["update"] == 7
    model = trainer.model
    model.load_state_dict(doc["model"])  # round-trips


# ---------------------------------------------------------------------------
# Synthetic end-to-end v51 fit (Lane B monkeypatched; CPU-fast)
# ---------------------------------------------------------------------------

def _graph_batch(rng, *, n: int = 9, n_keys: int = 3, rid: str,
                 branch: int, d0: float = 0.05, near: float = 0.1,
                 y_scale: float = 0.08) -> KeyAtomBatch:
    pos = rng.uniform(-2.5, 2.5, size=(n, 3))
    key_rows = sorted(rng.choice(n, size=n_keys, replace=False).tolist())
    pairs = [(key_rows[i], key_rows[i + 1]) for i in range(n_keys - 1)]
    edges = []
    for i in range(n - 1):
        edges.append((i, i + 1, True))
        edges.append((i + 1, i, True))
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) == 1:
                continue
            if float(np.linalg.norm(pos[i] - pos[j])) < 4.5:
                edges.append((i, j, False))
                edges.append((j, i, False))
    for a, b in pairs:
        edges.append((a, b, True))
        edges.append((b, a, True))
    e_src = [a for a, _b, _c in edges]
    e_dst = [b for _a, b, _c in edges]
    e_attr = np.zeros((len(edges), EDGE_DIM), dtype=np.float32)
    for k, (_a, _b, cov) in enumerate(edges):
        if cov:
            e_attr[k, EDGE_SLOTS["covalent_start"]] = 1.0
            e_attr[k, EDGE_SLOTS["covalent_other"]] = 1.0
    pos0 = pos + rng.normal(scale=0.3, size=pos.shape)
    anchor = (pos - pos.mean(0)) - (pos0 - pos0.mean(0))
    h_inv = rng.normal(size=(n, NODE_INV_DIM)).astype(np.float32)
    task = np.zeros(TASK_DIM, dtype=np.float32)
    task[0 if branch == 0 else 1] = 1.0
    task[16] = d0 / 0.08
    y = rng.normal(scale=y_scale, size=(n_keys, 3)).astype(np.float32)
    return KeyAtomBatch(
        z_atomic=torch.as_tensor(
            rng.integers(1, 10, size=n).astype(np.int64)),
        h_node_inv=torch.as_tensor(h_inv),
        pos=torch.as_tensor(pos, dtype=torch.float32),
        pos0=torch.as_tensor(pos0, dtype=torch.float32),
        anchor_vec=torch.as_tensor(anchor, dtype=torch.float32),
        edge_index=torch.as_tensor(np.stack([e_src, e_dst]), dtype=torch.int64),
        edge_attr=torch.as_tensor(e_attr),
        r_ij=torch.as_tensor(
            (pos[np.asarray(e_src)] - pos[np.asarray(e_dst)]),
            dtype=torch.float32),
        r0_ij=torch.as_tensor(
            (pos0[np.asarray(e_src)] - pos0[np.asarray(e_dst)]),
            dtype=torch.float32),
        task19=torch.as_tensor(task).unsqueeze(0),
        batch=torch.zeros(n, dtype=torch.int64),
        ptr=torch.tensor([0, n], dtype=torch.int64),
        key_index=torch.as_tensor(key_rows, dtype=torch.int64),
        key_ptr=torch.tensor([0, n_keys], dtype=torch.int64),
        pair_index=torch.as_tensor(np.stack(pairs), dtype=torch.int64),
        pair_edit=torch.as_tensor(
            np.array([1.0 if i % 2 == 0 else -1.0 for i in range(len(pairs))],
                     dtype=np.float32)),
        degK=torch.full((n_keys,), 2.0, dtype=torch.float32),
        y_teacher=torch.as_tensor(y),
        q_weight=torch.ones(1, dtype=torch.float32),
        near_ts=torch.tensor([near], dtype=torch.float32),
        terminal=torch.tensor([False]),
        capped=torch.tensor([False]),
        d0=torch.tensor([d0], dtype=torch.float32),
        group_ids=(rid,),
        reaction_ids=(rid,),
        branch=torch.tensor([branch], dtype=torch.int64),
    )


def _collate_single_graphs(items):
    z, h, pos, pos0, anc, ei, ea, rij, r0ij = [], [], [], [], [], [], [], [], []
    task, batch_vec, ptr = [], [], [0]
    key_index, key_ptr, pi, pj, pe, degk = [], [0], [], [], [], []
    y, qw, near, term, cap, d0s, br = [], [], [], [], [], [], []
    node_off = key_off = 0
    for g, item in enumerate(items):
        n = int(item.pos.shape[0])
        m = int(item.key_index.numel())
        z.append(item.z_atomic)
        h.append(item.h_node_inv)
        pos.append(item.pos)
        pos0.append(item.pos0)
        anc.append(item.anchor_vec)
        ei.append(item.edge_index + node_off)
        ea.append(item.edge_attr)
        rij.append(item.r_ij)
        r0ij.append(item.r0_ij)
        task.append(item.task19[0])
        batch_vec.append(torch.full((n,), g, dtype=torch.int64))
        ptr.append(ptr[-1] + n)
        key_index.append(item.key_index + node_off)
        key_ptr.append(key_ptr[-1] + m)
        pi.append(item.pair_index[0] + node_off)
        pj.append(item.pair_index[1] + node_off)
        pe.append(item.pair_edit)
        degk.append(item.degK)
        y.append(item.y_teacher)
        qw.append(item.q_weight[0])
        near.append(item.near_ts[0])
        term.append(item.terminal[0])
        cap.append(item.capped[0])
        d0s.append(item.d0[0])
        br.append(item.branch[0])
        node_off += n
        key_off += m
    return KeyAtomBatch(
        z_atomic=torch.cat(z), h_node_inv=torch.cat(h),
        pos=torch.cat(pos), pos0=torch.cat(pos0), anchor_vec=torch.cat(anc),
        edge_index=torch.stack([torch.cat([t[0] for t in ei]),
                                torch.cat([t[1] for t in ei])]),
        edge_attr=torch.cat(ea), r_ij=torch.cat(rij), r0_ij=torch.cat(r0ij),
        task19=torch.stack(task), batch=torch.cat(batch_vec),
        ptr=torch.as_tensor(ptr, dtype=torch.int64),
        key_index=torch.cat(key_index),
        key_ptr=torch.as_tensor(key_ptr, dtype=torch.int64),
        pair_index=torch.stack([torch.cat(pi), torch.cat(pj)]),
        pair_edit=torch.cat(pe), degK=torch.cat(degk),
        y_teacher=torch.cat(y), q_weight=torch.stack(qw),
        near_ts=torch.stack(near), terminal=torch.stack(term),
        capped=torch.stack(cap), d0=torch.stack(d0s),
        group_ids=tuple(it.reaction_ids[0] for it in items),
        reaction_ids=tuple(it.reaction_ids[0] for it in items),
        branch=torch.stack(br),
    )


@pytest.fixture()
def synthetic_lane(tmp_path, monkeypatch):
    """Monkeypatched Lane B over tiny synthetic graphs (10 train clusters
    × 2 branches × 2 frames + 1 dev cluster)."""
    rng = np.random.default_rng(20261019)
    samples: dict[str, KeyAtomBatch] = {}
    train: dict[str, dict[int, list[dict]]] = {}
    inner_rows: dict[str, list[dict]] = {}
    for i in range(10):
        rid = f"RXN_T{i:02d}"
        train[rid] = {}
        inner_rows[rid] = []
        for branch in (0, 1):
            rows = []
            for j in range(2):
                sid = f"{rid}.b{branch}.i{j}"
                samples[sid] = _graph_batch(rng, rid=rid, branch=branch)
                rows.append({"sample_id": sid, "q_weight": 1.0,
                             "group_id": rid})
            train[rid][branch] = rows
            inner_rows[rid].extend(rows)
    dev_rows = []
    for branch in (0, 1):
        sid = f"RXN_DEV.b{branch}.i0"
        samples[sid] = _graph_batch(rng, rid="RXN_DEV", branch=branch)
        dev_rows.append({"sample_id": sid, "q_weight": 1.0,
                         "group_id": "RXN_DEV"})
    family_of = {g: f"fam{i % 3}" for i, g in enumerate(train)}
    size_of = {g: (i % 2) + 1 for i, g in enumerate(train)}

    def fake_pools(config, *, fold, d0, split_path, inner_val=None):
        out = {"train": train, "dev_dense": dev_rows}
        if inner_val is not None:
            train_ids, val_ids, split_hash = kbc.inner_val_split(
                sorted(train), family_of=family_of, size_of=size_of,
                n_val=int(inner_val.get("n_val", 3)),
                seed=int(inner_val.get("seed", 20261019)))
            out["inner_train"] = {g: train[g] for g in train_ids}
            out["inner_val_dense"] = [r for g in val_ids
                                      for r in inner_rows[g]]
            out["inner_split"] = {
                "train_ids": list(train_ids), "val_ids": list(val_ids),
                "n_val": int(inner_val.get("n_val", 3)),
                "seed": int(inner_val.get("seed", 20261019)),
                "hash": split_hash,
                "train_rids": list(train_ids), "val_rids": list(val_ids),
                "n_val_clusters": len(val_ids),
            }
        return out

    def fake_view(*, config, samples: list, fold, d0, anchor, split_path,
                  device):
        view = [SAMPLES_BY_ID[row["sample_id"]] for row in samples]
        return view, _collate_single_graphs

    SAMPLES_BY_ID = samples
    monkeypatch.setattr(kbc, "load_lane_b_pools", fake_pools)
    monkeypatch.setattr(kbc, "load_lane_b_view", fake_view)
    return {"samples": samples, "train": train, "dev_rows": dev_rows}


def test_v51_fit_end_to_end_synthetic(tmp_path, synthetic_lane):
    cfg = _base_config(tmp_path)
    trainer = _make_trainer(tmp_path, mode="v51", arm_id="B3", updates=4,
                            config=cfg)
    status = trainer.fit()
    assert status == kbc.STATUS_COMPLETE
    run_dir = trainer.run_dir

    # --- eval.jsonl: inner-val during training, post-hoc outer rows last
    lines = [json.loads(l) for l in
             (run_dir / "eval.jsonl").read_text(encoding="utf-8").splitlines()]
    splits = [l.get("split") for l in lines]
    assert splits == ["inner_val", "inner_val", "dev_dense_best",
                      "dev_dense_final"]
    assert lines[2]["checkpoint"] == "checkpoint_best.pt"
    assert lines[3]["checkpoint"] == "checkpoint_final.pt"
    # g53 scalars merged into the aggregate the driver reads
    assert "sre_cluster_median" in lines[0]["aggregate"]
    assert "dead_zone_fraction" in lines[0]["aggregate"]
    assert "alpha_median" in lines[0]["aggregate"]

    # --- metrics.json driver contract
    metrics = json.loads((run_dir / "metrics.json").read_text("utf-8"))
    assert metrics["best_update"] in (2, 4)
    score = metrics["inner_val_score"]
    assert set(score) == {"rmse_over_zero", "sre_median", "near_ts_cos"}
    assert score["rmse_over_zero"] is not None
    assert metrics["alpha_median"] is not None
    assert metrics["dead_zone_fraction"] is not None
    assert metrics["sign_accuracy"] is not None
    assert metrics["selection_exposed"] is False
    assert metrics["loss_schema_version"] == kbc.LOSS_SCHEMA_VERSION
    # raw series: inner-val training curve + tagged post-hoc outer rows
    assert [r.get("split") for r in metrics["raw"]] == [
        "inner_val", "inner_val", "dev_dense_best", "dev_dense_final"]

    # --- run_record.json
    record = json.loads((run_dir / "run_record.json").read_text("utf-8"))
    assert record["schema_version"] == kbc.RUN_RECORD_SCHEMA
    assert record["loss_schema_version"] == kbc.LOSS_SCHEMA_VERSION
    assert record["campaign_id"] == "V5_1A_SPRINT"
    # registered in g_v_map since 2026-10-19 — no V5_SPRINT fallback
    assert record["gv"]["campaign_id"] == "V5_1A_SPRINT"
    assert record["gv"]["campaign_id_configured"] == "V5_1A_SPRINT"
    assert record["status"] == kbc.STATUS_COMPLETE
    assert record["selection_exposed"] is False
    assert record["best_update"] == metrics["best_update"]
    assert set(record["inner_val_score"]) == {
        "rmse_over_zero", "sre_median", "near_ts_cos"}
    assert record["alpha_median"] is not None
    assert record["dead_zone_fraction"] is not None
    split_hash = record["inner_val_split_hash"]
    assert len(split_hash) == 64
    assert record["inner_val_split"]["n_val"] == 3
    assert len(record["inner_val_split"]["val_ids"]) == 3
    assert record["arm"]["magnitude_head"] is True
    assert record["arm"]["pair_loss"] == "exact_trio"
    assert record["mode"] == "v51"

    # --- checkpoints
    best_path = run_dir / "checkpoints" / "checkpoint_best.pt"
    final_path = run_dir / "checkpoints" / "checkpoint_final.pt"
    last_path = run_dir / "checkpoints" / "checkpoint_last.pt"
    assert best_path.is_file() and final_path.is_file() and last_path.is_file()
    import torch as _torch
    best = _torch.load(best_path, map_location="cpu", weights_only=False)
    assert best["update"] == metrics["best_update"]
    assert best["magnitude_head"] is True
    assert best["magnitude_mode"] == "per_atom"
    assert best["pair_loss"] == "exact_trio"
    assert best["v51"]["selection_exposed"] is False
    assert best["v51"]["inner_val_split_hash"] == split_hash
    assert best["v51"]["best_update"] == metrics["best_update"]
    model = trainer.model  # head-on construction round-trips the state
    model.load_state_dict(best["model"])

    # --- status.json
    status_doc = json.loads((run_dir / "status.json").read_text("utf-8"))
    assert status_doc["status"] == kbc.STATUS_COMPLETE
    assert status_doc["best_update"] == metrics["best_update"]
    assert status_doc["selection_exposed"] is False
    assert status_doc["inner_val_score"] == score


def test_v51_fit_never_selects_on_outer_dev(tmp_path, synthetic_lane):
    """All rows BEFORE the post-hoc block are inner_val; the outer rows
    come only after training ended (selection_exposed=false)."""
    cfg = _base_config(tmp_path)
    trainer = _make_trainer(tmp_path, mode="v51", arm_id="B0", updates=4,
                            config=cfg)
    assert trainer.fit() == kbc.STATUS_COMPLETE
    lines = [json.loads(l) for l in
             (trainer.run_dir / "eval.jsonl").read_text().splitlines()]
    outer = [l for l in lines if l.get("split") in (
        "dev_dense_best", "dev_dense_final")]
    inner = [l for l in lines if l.get("split") == "inner_val"]
    assert len(inner) == 2 and len(outer) == 2
    assert lines.index(outer[0]) > max(
        lines.index(i) for i in inner)


def test_legacy_screen_fit_unchanged(tmp_path, synthetic_lane):
    """V5 screen-mode fit on the same synthetic data: no split tags, no
    v51 keys, legacy loss schema stamp."""
    cfg = _base_config(tmp_path, schema="pesn_v5_sprint_draft2")
    cfg["sprint"] = {"arm_table": {"A05": {"d0": 0.05, "anchor": "on"}}}
    trainer = _make_trainer(tmp_path, mode="screen", arm_id="A05",
                            updates=2, config=cfg)
    status = trainer.fit()
    assert status == kbc.STATUS_COMPLETE
    lines = [json.loads(l) for l in
             (trainer.run_dir / "eval.jsonl").read_text().splitlines()]
    assert all("split" not in l for l in lines)
    assert all("g53_aggregate" not in l for l in lines)
    metrics = json.loads(
        (trainer.run_dir / "metrics.json").read_text("utf-8"))
    assert "best_update" not in metrics
    assert "inner_val_score" not in metrics
    record = json.loads(
        (trainer.run_dir / "run_record.json").read_text("utf-8"))
    assert record["loss_schema_version"] == kbc.SCHEMA_VERSION
    assert record["campaign_id"] == "V5_SPRINT"
    assert record["mode"] == "screen"
    assert record["arm"] == {"arm_id": "A05", "anchor": "on", "d0": 0.05,
                             "magnitude_head": False, "pair_loss": "legacy_j"}
    assert (trainer.run_dir / "checkpoints" / "checkpoint_final.pt").is_file()
    assert not (
        trainer.run_dir / "checkpoints" / "checkpoint_best.pt").exists()


def test_status_values_keep_v5_and_add_patience():
    assert kbc.STATUS_COMPLETE == "complete"
    assert kbc.STATUS_CENSORED_SCREEN == "censored_screen"
    assert kbc.STATUS_ABORTED_DATA == "aborted_data"
    assert kbc.STATUS_ABORTED_NAN == "aborted_nan"
    assert kbc.STATUS_CENSORED_PATIENCE == "censored_patience"
    assert set(kbc.STATUS_VALUES) == {
        "complete", "censored_screen", "aborted_data", "aborted_nan",
        "censored_patience"}
    assert kbc.LOSS_SCHEMA_VERSION == "pesn_key_atom_bc_loss_v2"
    assert kbc.CHECKPOINT_SCHEMA == "pesn_key_atom_bc_checkpoint_v1"
    assert kbc.ACCEPTED_CONFIG_SCHEMAS == (
        "pesn_v5_sprint_draft2", "pesn_v5_1_sprint_v1")
    assert kbc.V51_INNER_SPLIT_SEED == 20261019
    assert kbc.V51_QUALIFY_FACTOR == 1.02
    assert kbc.V51_PATIENCE_FLOOR_UPDATE == 500
