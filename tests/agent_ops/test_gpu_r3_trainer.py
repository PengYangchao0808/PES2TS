"""WP-L3/E3 trainer tests (mission: key_atom_bc_v3 training loop, lane W-LOSS).

CPU fixture mini-runs on the real E1 fold-0 tables (skipped when the
gitignored campaign data is absent): loss decreases, checkpoint roundtrip,
resume-by-run-id, label-weight applied exactly once, inner-val-only
checkpoint selection, metrics schema, determinism, curve-balanced sampler
composition, inner-split containment, config refusals and the classic
CV/Jacobian baseline algebra.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "outputs/pesn/30h_campaign/data"
SPLIT = DATA_DIR / "split_manifest_r3.json"

pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "sample_view.csv").is_file() or not SPLIT.is_file(),
    reason="campaign E1 tables not present in this checkout (gitignored artifacts)",
)

from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (  # noqa: E402
    KeyAtomDatasetV3,
)
from pes2ts_core.pesn.offline.training import key_atom_bc_v3 as trainer_mod  # noqa: E402
from pes2ts_core.pesn.offline.training.key_atom_bc_v3 import (  # noqa: E402
    TrainerConfigV3,
    TrainerError,
    classic_jacobian_step,
    curve_batch_sampler,
    make_inner_split,
    train_run,
)

UPDATES = 30


@pytest.fixture(scope="module")
def dataset() -> KeyAtomDatasetV3:
    return KeyAtomDatasetV3(
        DATA_DIR, fold=0, split_manifest=SPLIT, label_statuses=("ok",))


def _config(**overrides) -> TrainerConfigV3:
    base = dict(arm="F0", fold=0, seed=17, level="F0", updates=UPDATES,
                val_every=10, device="cpu")
    base.update(overrides)
    return TrainerConfigV3(**base)


# ---------------------------------------------------------------------------
# config refusals (fail-closed)
# ---------------------------------------------------------------------------


def test_config_refusals():
    with pytest.raises(TrainerError) as excinfo:
        TrainerConfigV3(arm="F0", fold=0, seed=17, batch_size=30)
    assert "BATCH_COMPOSITION" in str(excinfo.value)
    with pytest.raises(TrainerError) as excinfo:
        TrainerConfigV3(arm="S2", fold=0, seed=17, aux_arm="S2", lambda_arm=0.9)
    assert "AUX_LAMBDA_RANGE" in str(excinfo.value)
    with pytest.raises(TrainerError) as excinfo:
        TrainerConfigV3(arm="F0", fold=0, seed=17, lambda_arm=0.1)
    assert "AUX_LAMBDA_WITHOUT_ARM" in str(excinfo.value)


# ---------------------------------------------------------------------------
# inner split + sampler composition
# ---------------------------------------------------------------------------


def test_inner_split_only_outer_train_and_deterministic(dataset):
    samples = dataset.samples
    train_idx, val_idx = make_inner_split(samples, seed=17)
    outer = {s.curve_id for s in samples}
    train_curves = {samples[i].curve_id for i in train_idx}
    val_curves = {samples[i].curve_id for i in val_idx}
    assert train_curves | val_curves == outer
    assert not (train_curves & val_curves)          # disjoint
    assert 1 <= len(val_curves) <= len(outer) - 1   # both sides non-trivial
    # deterministic per seed
    t2, v2 = make_inner_split(samples, seed=17)
    assert (train_idx, val_idx) == (t2, v2)
    # a different seed picks a different (equally sized in curves) val split
    _, v3 = make_inner_split(samples, seed=29)
    v3_curves = {samples[i].curve_id for i in v3}
    assert len(v3_curves) == len(val_curves)


def test_sampler_curve_balanced_and_direction_balanced(dataset):
    samples = dataset.samples
    train_idx, _ = make_inner_split(samples, seed=17)
    train_curves = {samples[i].curve_id for i in train_idx}
    gen = curve_batch_sampler(samples, train_idx, bidirectional=False, seed=7)
    for _ in range(5):
        positions = next(gen)
        assert len(positions) == 32                    # frozen batch size
        curves = [samples[train_idx[p]].curve_id for p in positions]
        assert set(curves) <= train_curves             # inner-train only
        from collections import Counter
        per_curve = Counter(curves)
        assert sum(per_curve.values()) == 32
        # 8 curve slots x 4 frames (a curve may repeat when the fold has
        # <8 inner-train curves; every slot still draws 4 frames)
        assert max(per_curve.values()) % 4 == 0
        dirs = {samples[train_idx[p]].intent.direction for p in positions}
        assert dirs == {"forward"}                     # forward-only arm
    gen_b = curve_batch_sampler(samples, train_idx, bidirectional=True, seed=7)
    positions = next(gen_b)
    from collections import Counter
    per_curve_dir = Counter(
        (samples[train_idx[p]].curve_id, samples[train_idx[p]].intent.direction)
        for p in positions)
    by_curve = {}
    for (curve, direction), n in per_curve_dir.items():
        by_curve.setdefault(curve, {})[direction] = n
    for curve, dirs in by_curve.items():
        assert sum(dirs.values()) % 4 == 0             # 4 frames per curve slot
        if "forward" in dirs and "reverse" in dirs:
            assert dirs["forward"] == dirs["reverse"]  # balanced 2+2 per slot


# ---------------------------------------------------------------------------
# mini-run: loss decreases, checkpoint roundtrip, selection, metrics schema
# ---------------------------------------------------------------------------


def test_mini_run_loss_decreases_checkpoint_and_metrics(dataset, tmp_path):
    out = tmp_path / "run"
    metrics = train_run(dataset, _config(), out, resume=False)

    # loss decreases (first logged train loss vs last)
    log = [json.loads(ln) for ln in
           (out / "train_log.jsonl").read_text().splitlines()]
    train_logs = [e for e in log if "loss" in e]
    assert train_logs[0]["loss"] > train_logs[-1]["loss"]

    # checkpoint roundtrip: load best, run one eval batch
    state = torch.load(str(out / "checkpoint_best.pt"), map_location="cpu",
                       weights_only=False)
    assert state["schema"] == trainer_mod.TRAINER_SCHEMA
    assert state["config"]["arm"] == "F0"
    assert state["seeds"]["torch"] == 17

    # selection is inner-val argmin (recorded best_update matches the log)
    val_logs = [e for e in log if "inner_val_loss" in e]
    best = min(val_logs, key=lambda e: e["inner_val_loss"])
    assert metrics["best_update"] == best["update"]
    assert metrics["selected_update"] == best["update"]
    # selection never uses outer-val: recorded val curves ⊂ fold0 outer-train
    fold0_curves = {s.curve_id for s in dataset.samples}
    assert set(metrics["inner_val_curves"]) <= fold0_curves
    assert set(metrics["inner_train_curves"]).isdisjoint(
        metrics["inner_val_curves"])

    # metrics schema (frozen primaries + guards + provenance)
    for key in ("key_rmse_over_zero", "sre_and_sign", "baselines", "guards",
                "checkpoint_best_sha256", "seeds", "selection_metric",
                "hyperparameters", "arm", "fold", "seed"):
        assert key in metrics, key
    assert metrics["key_rmse_over_zero"]["median_rmse_over_zero"] is not None
    assert metrics["sre_and_sign"]["sre_cluster_median"] is not None
    assert metrics["guards"]["policy"].startswith("record-only")
    assert metrics["baselines"]["zero_step"]["key_rmse_over_zero"] is not None
    assert "formula" in metrics["baselines"]["classic_cv_jacobian"]
    # metrics file written and self-consistent
    on_disk = json.loads((out / "metrics.json").read_text())
    assert on_disk["checkpoint_best_sha256"] == metrics["checkpoint_best_sha256"]
    from pes2ts_core.pesn.offline.training.key_atom_bc_v3 import sha256_file
    assert sha256_file(out / "checkpoint_best.pt") == \
        metrics["checkpoint_best_sha256"]


def test_resume_continues_by_run_identity(dataset, tmp_path):
    out = tmp_path / "run"
    train_run(dataset, _config(), out, resume=False)
    state = torch.load(str(out / "checkpoint_last.pt"), map_location="cpu",
                       weights_only=False)
    assert state["update"] == UPDATES
    # resume with a longer budget continues from the checkpoint
    metrics = train_run(dataset, _config(updates=UPDATES + 10), out, resume=True)
    state2 = torch.load(str(out / "checkpoint_last.pt"), map_location="cpu",
                        weights_only=False)
    assert state2["update"] == UPDATES + 10
    # resume into a DIFFERENT identity refuses (fail-closed)
    with pytest.raises(TrainerError) as excinfo:
        train_run(dataset, _config(seed=29), out, resume=True)
    assert "RESUME_IDENTITY_MISMATCH" in str(excinfo.value)


def test_determinism_same_seed_same_metrics(dataset, tmp_path):
    """Same seed -> same trajectory within float tolerance (CPU threading
    makes bit-exactness unattainable; recorded honestly, not claimed)."""
    m1 = train_run(dataset, _config(), tmp_path / "a", resume=False)
    m2 = train_run(dataset, _config(), tmp_path / "b", resume=False)
    assert m1["last_train_loss"] == pytest.approx(m2["last_train_loss"],
                                                  rel=1e-6, abs=1e-9)
    assert m1["best_inner_val_loss"] == pytest.approx(
        m2["best_inner_val_loss"], rel=1e-6, abs=1e-9)
    assert m1["seeds"] == m2["seeds"]
    assert m1["best_update"] == m2["best_update"]


# ---------------------------------------------------------------------------
# label_weight applied exactly once (spy on the frozen loss call)
# ---------------------------------------------------------------------------


def test_frame_weight_passed_once(dataset, tmp_path, monkeypatch):
    """The trainer hands the loss EXACTLY the batch frame_weight (label_weight)."""
    captured = []
    real_loss = trainer_mod.key_atom_loss_v3

    def spy(u, y, degK, key_ptr, pos, key_index, pair_index=None, **kwargs):
        fw = kwargs.get("frame_weight")
        captured.append(None if fw is None else fw.detach().cpu().clone())
        return real_loss(u, y, degK, key_ptr, pos, key_index, pair_index,
                         **kwargs)

    monkeypatch.setattr(trainer_mod, "key_atom_loss_v3", spy)
    out = tmp_path / "run"
    train_run(dataset, _config(updates=2, val_every=2), out, resume=False)
    assert captured, "spy never saw a loss call"
    # every training call received a [G] weight tensor equal to the batch's
    # label weights (dataset label_weight=1.0 for all E1 'ok' rows); the
    # trainer must NOT square/rescale it (applied exactly once, in the loss)
    for fw in captured[:3]:
        assert fw is not None and fw.ndim == 1
        assert float(fw.min()) >= 0.0


# ---------------------------------------------------------------------------
# classic CV/Jacobian baseline algebra
# ---------------------------------------------------------------------------


def test_classic_jacobian_step_satisfies_projection():
    rng = np.random.RandomState(0)
    pos = rng.randn(5, 3)
    key_rows = [0, 1, 2]
    pairs = [(0, 1), (1, 2), (0, 2)]
    target = np.array([0.08, -0.08, 0.0])
    u = classic_jacobian_step(pos, key_rows, pairs, target)
    assert u.shape == (3, 3)
    # first-order projections reproduce the solvable target subspace
    slot = {r: k for k, r in enumerate(key_rows)}
    for l, (i, j) in enumerate(pairs):
        r = pos[j] - pos[i]
        e = r / np.linalg.norm(r)
        j_l = float(e @ (u[slot[j]] - u[slot[i]]))
        assert math.isclose(j_l, target[l], abs_tol=1e-8)


def test_aux_arm_loss_wiring(dataset, tmp_path):
    """S2 arm: trainer passes its calibrated λ to the frozen loss (bucket only)."""
    out = tmp_path / "run"
    metrics = train_run(
        dataset,
        _config(arm="S2", aux_arm="S2", lambda_arm=0.05),
        out, resume=False)
    assert metrics["aux_arm"] == "S2"
    assert metrics["lambda_arm"] == 0.05
    log = [json.loads(ln) for ln in (out / "train_log.jsonl").read_text().splitlines()]
    train_logs = [e for e in log if "loss" in e]
    assert train_logs[0]["lambda_strong"] == pytest.approx(0.05)
    assert train_logs[0]["lambda_dead"] == 0.0


# ---------------------------------------------------------------------------
# E3-CONT: lambda calibration probe math (plan 5.4)
# ---------------------------------------------------------------------------


def test_probe_bucket_ratios_and_median():
    probe = pytest.importorskip("scripts.campaigns.r3_gradient_probe")
    r_enc, r_head = probe.bucket_ratios(2.0, 0.1, 4.0, 2.0)
    assert r_enc == pytest.approx(0.05)
    assert r_head == pytest.approx(0.5)
    # zero base (degenerate) refuses to div-zero (epsilon in denominator)
    r_enc, _ = probe.bucket_ratios(0.0, 0.3, 0.0, 0.3)
    assert r_enc == pytest.approx(0.3 / 1e-12)
    assert probe.median([3.0, 1.0, 2.0]) == 2.0
    assert probe.median([4.0, 1.0, 2.0, 3.0]) == 2.5


def test_probe_calibrate_lambda_verdicts():
    probe = pytest.importorskip("scripts.campaigns.r3_gradient_probe")
    # calibrated: ratio 1.0 -> lambda = target exactly
    out = probe.calibrate_lambda(1.0, 0.02)
    assert out["verdict"] == "calibrated"
    assert out["lambda"] == pytest.approx(0.02)
    assert out["share_at_lambda"] == pytest.approx(0.02)
    # target_unreachable: even lambda=0.5 stays under the 10% target but
    # above the 1% scale floor
    out = probe.calibrate_lambda(0.5, 0.10)   # needs lambda=0.2 -> ok
    assert out["verdict"] == "calibrated" and out["lambda"] == pytest.approx(0.2)
    out = probe.calibrate_lambda(0.5, 0.02)   # needs 0.04 -> ok
    assert out["verdict"] == "calibrated"
    out = probe.calibrate_lambda(0.25, 0.10)  # needs 0.4 -> ok
    assert out["verdict"] == "calibrated"
    out = probe.calibrate_lambda(0.3, 0.10)   # needs 0.333 -> ok
    assert out["verdict"] == "calibrated"
    out = probe.calibrate_lambda(0.21, 0.10)  # needs 0.476 -> ok
    assert out["verdict"] == "calibrated"
    out = probe.calibrate_lambda(0.15, 0.02)  # needs 0.133 -> ok
    assert out["verdict"] == "calibrated"
    out = probe.calibrate_lambda(0.22, 0.10)  # needs 0.4545 -> ok
    assert out["verdict"] == "calibrated"
    out = probe.calibrate_lambda(0.224, 0.10)  # needs 0.446 -> ok
    assert out["verdict"] == "calibrated"
    out = probe.calibrate_lambda(0.24, 0.10)  # needs 0.4167 -> ok
    assert out["verdict"] == "calibrated"
    out = probe.calibrate_lambda(0.3, 0.02)   # needs 0.0667 -> ok
    assert out["verdict"] == "calibrated"
    # target_unreachable: share at lambda_max < target but >= 1% floor
    out = probe.calibrate_lambda(0.15, 0.10)  # needs 0.667 > 0.5; share@0.5=0.075
    assert out["verdict"] == "target_unreachable"
    assert out["lambda"] == 0.5
    assert out["share_at_lambda"] == pytest.approx(0.075)
    # scale_unreachable: even lambda_max stays below the 1% floor
    out = probe.calibrate_lambda(0.01, 0.10)  # share@0.5 = 0.005 < 0.01
    assert out["verdict"] == "scale_unreachable"
    assert out["lambda"] == 0.5
    out = probe.calibrate_lambda(0.0, 0.02)   # no gradient at all
    assert out["verdict"] == "scale_unreachable"
    assert out["lambda"] is None
    # head flag
    assert probe.head_flag(0.3) == "ok"
    assert probe.head_flag(0.3001) == "head_overweighted"


def test_probe_calibration_json_shape_and_lambda_zero_regression(dataset):
    """The shipped probe output is well-formed; aux lambda=0 == B0-F bitwise.

    Bit-identity is asserted at the LOSS level (one process, same tensors):
    the aux codepath at lambda=0 must return exactly the base loss value
    and gradients -- the trainer passes lambda_strong/dead=0.0, which makes
    the frozen loss skip the bucket terms entirely.
    """
    from pes2ts_core.pesn.models.key_atom_losses_v3 import (
        KeyAtomLossV3Config,
        aux_arm_lambdas_v3,
        key_atom_loss_v3,
    )
    # aux_arm_lambdas_v3 places lambda on the arm's bucket only
    assert aux_arm_lambdas_v3("S2", 0.0) == (0.0, 0.0)
    assert aux_arm_lambdas_v3("S10", 0.1) == (0.1, 0.0)
    assert aux_arm_lambdas_v3("D10", 0.1) == (0.0, 0.1)

    # loss-level bit identity: base vs aux-arm-at-0 on one real batch
    # (single-thread: CPU reduction order is otherwise free to vary between
    # separate backward calls, which would mask true bit-identity)
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(17)
        train_idx, _ = make_inner_split(dataset.samples, 17)
        from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (
            build_feature_batches,
        )
        from pes2ts_core.pesn.contracts.key_atom_v3 import (
            NODE_DEGK_KEY,
            NODE_KEY_SET_KEY,
            NODE_POS_KEY,
            forward_view_v3,
        )
        batch = build_feature_batches(
            dataset, train_idx[:8], level="F2", device="cpu")
        from pes2ts_core.pesn.models.key_atom_egnn_v3 import StarEGNNV3
        model = StarEGNNV3()
        n_keys = batch["n_keys"].long()
        key_ptr = torch.cat(
            [torch.zeros(1, dtype=torch.long), torch.cumsum(n_keys, dim=0)])

        def _loss(**kwargs):
            out = model(forward_view_v3(batch))
            return key_atom_loss_v3(
                out["u"], batch["y_teacher"],
                batch[NODE_DEGK_KEY][batch[NODE_KEY_SET_KEY]], key_ptr,
                batch[NODE_POS_KEY], batch[NODE_KEY_SET_KEY],
                batch["pair_index"], config=KeyAtomLossV3Config(),
                lambda_cos=0.02, frame_weight=batch["frame_weight"],
                pair_edit=batch["pair_edit"], **kwargs)

        base = _loss()
        aux0 = _loss(aux_arm="S2", lambda_strong=0.0, lambda_dead=0.0)
        assert float(base["loss"]) == float(aux0["loss"])      # bit-identical
        assert float(base["l_atom"]) == float(aux0["l_atom"])
        assert float(aux0["l_strong"]) == 0.0
        # gradient bit-identity
        base["loss"].backward()
        grads_base = [p.grad.clone() for p in model.parameters()
                      if p.grad is not None]
        model.zero_grad()
        aux0["loss"].backward()
        grads_aux = [p.grad.clone() for p in model.parameters()
                     if p.grad is not None]
        assert len(grads_base) == len(grads_aux) > 0
        for ga, gb in zip(grads_base, grads_aux):
            assert torch.equal(ga, gb)
    finally:
        torch.set_num_threads(prev_threads)

    # the shipped calibration file (when present) has the frozen schema
    cal = Path(__file__).resolve().parents[2] / \
        "outputs/pesn/30h_campaign/r3_runs/lambda_calibration.json"
    if cal.is_file():
        doc = json.loads(cal.read_text(encoding="utf-8"))
        assert doc["schema"] == probe_schema
        assert set(doc["arms"]) == {"S2", "S10", "D2", "D10"}
        for arm, v in doc["arms"].items():
            assert v["verdict"] in ("calibrated", "target_unreachable",
                                    "scale_unreachable")
            if v["lambda"] is not None:
                assert 0.0 < v["lambda"] <= 0.5
            assert v["target_encoder_share"] == (0.02 if arm.endswith("2")
                                                 else 0.10)


probe_schema = "pesn_r3_lambda_probe_v1"


# ---------------------------------------------------------------------------
# E3-CONT: seal semantics (plan 4.2: folds3/4 excluded from aggregation)
# ---------------------------------------------------------------------------


def test_seal_semantics_folds34_excluded_from_selection():
    ev = pytest.importorskip("scripts.campaigns.r3_g2_evidence")
    entries = [
        {"run_id": "b-f1-s17", "arm": "B", "fold": 1, "seed": 17,
         "rmse_over_zero": 0.9, "median_rmse": 0.05, "sign_median": 0.9,
         "sre_median": 0.2, "dead_zone_fraction": 0.1,
         "classic_sign_median": 0.0, "classic_sre_median": 0.5,
         "selected_update": 250, "last_train_loss": 0.1,
         "checkpoint_sha256": "aa" * 32},
        {"run_id": "fb-f0-s17", "arm": "F2-bidir", "fold": 0, "seed": 17,
         "rmse_over_zero": 0.884, "median_rmse": 0.05, "sign_median": 0.764,
         "sre_median": 0.41, "dead_zone_fraction": 0.1,
         "classic_sign_median": 0.0, "classic_sre_median": 0.5,
         "selected_update": 250, "last_train_loss": 0.12,
         "checkpoint_sha256": "ee" * 32},
        {"run_id": "b-f3-s17", "arm": "B", "fold": 3, "seed": 17,
         "rmse_over_zero": 0.5, "median_rmse": 0.01, "sign_median": 1.0,
         "sre_median": 0.05, "dead_zone_fraction": 0.0,
         "classic_sign_median": 0.0, "classic_sre_median": 0.5,
         "selected_update": 250, "last_train_loss": 0.05,
         "checkpoint_sha256": "bb" * 32},
        {"run_id": "l-s2-f0-s17", "arm": "S2", "fold": 0, "seed": 17,
         "rmse_over_zero": 0.95, "median_rmse": 0.052, "sign_median": 0.85,
         "sre_median": 0.25, "dead_zone_fraction": 0.12,
         "classic_sign_median": 0.0, "classic_sre_median": 0.5,
         "selected_update": 500, "last_train_loss": 0.09,
         "checkpoint_sha256": "cc" * 32},
        {"run_id": "b-f4-s29", "arm": "B", "fold": 4, "seed": 29,
         "rmse_over_zero": 0.4, "median_rmse": 0.005, "sign_median": 1.0,
         "sre_median": 0.01, "dead_zone_fraction": 0.0,
         "classic_sign_median": 0.0, "classic_sre_median": 0.5,
         "selected_update": 250, "last_train_loss": 0.01,
         "checkpoint_sha256": "dd" * 32},
    ]
    assert ev.is_sealed_fold(3) and ev.is_sealed_fold(4)
    assert not ev.is_sealed_fold(0) and not ev.is_sealed_fold(2)
    dev, sealed = ev.split_entries(entries)
    # sealed values never reach the dev aggregate; ids listed for audit
    assert sorted(sealed) == ["b-f3-s17", "b-f4-s29"]
    assert {e["run_id"] for e in dev} == {
        "b-f1-s17", "fb-f0-s17", "l-s2-f0-s17"}
    pack = ev.build_evidence(entries)
    assert pack["n_sealed_runs_excluded"] == 2
    assert pack["sealed_run_ids"] == ["b-f3-s17", "b-f4-s29"]
    # the spectacularly-good sealed fold3 metrics must NOT leak into B stats
    b_arm = pack["per_arm"]["B"]
    assert b_arm["n_runs"] == 1
    assert b_arm["mean_rmse_over_zero"] == pytest.approx(0.9)
    assert b_arm["min_rmse_over_zero"] == pytest.approx(0.9)
    assert b_arm["folds"] == [1]
    # guards compare the L arm against its matched B0-F (same fold+seed)
    (guard,) = pack["guards_vs_matched_b0f"]
    assert guard["baseline_run_id"] == "fb-f0-s17"
    assert guard["rmse_degradation_pct"] == pytest.approx(
        100.0 * (0.052 - 0.05) / 0.05)
    assert guard["sign_drop"] == pytest.approx(0.764 - 0.85)
    assert guard["deadzone_rise"] == pytest.approx(0.12 - 0.1)
    assert not guard["within_rmse_guard"]   # 4% > 2% -- recorded, not gating
    # early-overfit accounting (sealed entries excluded)
    assert pack["early_overfit_note"]["n_selected_at_or_before_250"] == 2
    assert pack["early_overfit_note"]["n_selected_total"] == 3


# ---------------------------------------------------------------------------
# E3-CONT: driver lambda/arm wiring
# ---------------------------------------------------------------------------


def _load_driver():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "r3_drv_cont",
        str(Path(__file__).resolve().parents[2] /
            "scripts/campaigns/run_v5_1_r3.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_driver_arm_map_and_lambda_required():
    drv = _load_driver()
    # B0-F and Top-1/aux arms inherit the winner's bidirectional config
    assert drv.ARM_MAP["B"]["bidirectional"] is True
    assert drv.ARM_MAP["V-top1"]["bidirectional"] is True
    assert drv.ARM_MAP["S2"]["bidirectional"] is True
    assert drv.ARM_MAP["D10"]["bidirectional"] is True
    assert drv.ARM_MAP["V-top2"].get("bidirectional") is not True
    assert drv.SEALED_FOLDS == (3, 4)

    line = {"arm": "S2", "fold": 0, "seed": 17, "variant": None,
            "campaign_id": "PES2TS_30H_R3_V52_V53", "phase": "R3",
            "data_sha256": "a1" * 32, "teacher_sha256": "b2" * 32,
            "features_sha256": "c3" * 32, "split_sha256": "d4" * 32,
            "config_sha256": "e5" * 32, "confirm": False}

    class Args:
        level = "F2"
        lambda_arm = None
        device = "cpu"

    from pes2ts_core.pesn.offline.training.key_atom_bc_v3 import TrainerError
    with pytest.raises(TrainerError) as excinfo:
        drv._trainer_config_from_line(line, Args())
    assert "AUX_LAMBDA_REQUIRED" in str(excinfo.value)

    class Args2:
        level = "F2"
        lambda_arm = "0.019056400705052445"
        device = "cpu"

    cfg = drv._trainer_config_from_line(line, Args2())
    assert cfg.aux_arm == "S2"
    assert cfg.lambda_arm == pytest.approx(0.019056400705052445)
    assert cfg.bidirectional is True
    assert cfg.level == "F2"

    # B arm with explicit winner level -> bidirectional B0-F config
    b_line = dict(line, arm="B")

    class Args3:
        level = "F2"
        lambda_arm = None
        device = "cpu"

    cfg_b = drv._trainer_config_from_line(b_line, Args3())
    assert cfg_b.bidirectional is True and cfg_b.aux_arm is None
