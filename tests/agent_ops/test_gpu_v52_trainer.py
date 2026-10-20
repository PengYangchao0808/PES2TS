"""E5 V5.2 trainer tests (mission: key_atom_v52_train_v3, lane W-LOSS).

CPU tests on the real E1 tables (skip-if-absent): λ off-term refusals
(regression for the LAMBDA_RANGE queue burn), E4-coherent split protocol,
frozen-encoder stage actually freezing backbone params, R3-backbone
init verbatim load, frozen stop class weights, label mapping (Mask/q),
AUC math, metrics schema from a 4-update fixture run, and the G3
evidence-pack criteria helpers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "outputs/pesn/30h_campaign/data"
SPLIT = DATA_DIR / "split_manifest_r3.json"
E4_REPORT = PROJECT_ROOT / \
    "outputs/pesn/30h_campaign/e4_labels/e4_label_report.json"
R3_CKPT = PROJECT_ROOT / \
    "outputs/pesn/30h_campaign/r3_runs/5372d9b12807c3f0/checkpoint_best.pt"

pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "sample_view.csv").is_file() or not SPLIT.is_file(),
    reason="campaign E1 tables not present in this checkout")

from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (  # noqa: E402
    KeyAtomDatasetV3,
)
from pes2ts_core.pesn.offline.training import key_atom_v52_train_v3 as tr  # noqa: E402
from pes2ts_core.pesn.offline.training.key_atom_v52_train_v3 import (  # noqa: E402
    FROZEN_STOP_WEIGHTS_V52,
    V52TrainerConfig,
    V52TrainerError,
    auc_mann_whitney,
    load_backbone_from_r3,
    pearson,
    stop_class_weights_for_fold,
    v52_curve_split,
)

LQ = 0.42104980687872556
LM = 0.0132012267552582
INIT = "5372d9b12807c3f0"


@pytest.fixture(scope="module")
def dataset() -> KeyAtomDatasetV3:
    return KeyAtomDatasetV3(
        DATA_DIR, split_manifest=SPLIT, label_statuses=("ok",))


@pytest.fixture(scope="module")
def labels(dataset):
    return dataset.v52_labels()


# ---------------------------------------------------------------------------
# config refusals (regression: the queue-burn bug)
# ---------------------------------------------------------------------------


def test_config_lambda_refusals():
    # single-term arms: active term needs calibrated λ in (0, 0.5]
    ok_q = V52TrainerConfig(arm="Q", fold=0, seed=17, lambda_q=LQ,
                            lambda_m=0.0, backbone_run_id=INIT)
    assert ok_q.train_progress and not ok_q.train_stop
    ok_t = V52TrainerConfig(arm="T", fold=1, seed=17, lambda_q=0.0,
                            lambda_m=LM, backbone_run_id=INIT)
    assert ok_t.train_stop and not ok_t.train_progress
    # REGRESSION: off-term λ=0 on an ACTIVE check used to burn the queue
    with pytest.raises(V52TrainerError) as excinfo:
        V52TrainerConfig(arm="Q", fold=0, seed=17, lambda_q=LQ, lambda_m=0.9,
                         backbone_run_id=INIT)
    assert excinfo.value.code == "LAMBDA_OFF_TERM"
    with pytest.raises(V52TrainerError) as excinfo:
        V52TrainerConfig(arm="Q", fold=0, seed=17, lambda_q=0.0, lambda_m=0.0,
                         backbone_run_id=INIT)
    assert excinfo.value.code == "LAMBDA_RANGE"      # active λq missing
    with pytest.raises(V52TrainerError) as excinfo:
        V52TrainerConfig(arm="QT", fold=0, seed=17, lambda_q=0.9, lambda_m=LM,
                         backbone_run_id=INIT)
    assert excinfo.value.code == "LAMBDA_RANGE"      # > 0.5 cap
    with pytest.raises(V52TrainerError):
        V52TrainerConfig(arm="Q", fold=3, seed=17, lambda_q=LQ, lambda_m=0.0,
                         backbone_run_id=INIT)       # confirm fold refused


# ---------------------------------------------------------------------------
# split protocol (E4-coherent: train folds != F; eval fold F)
# ---------------------------------------------------------------------------


def test_split_train_folds_never_include_eval_fold(dataset):
    samples = dataset.samples
    train, val, test = v52_curve_split(samples, fold=0, seed=17)
    assert train and val and test
    train_folds = {int(samples[i].fold) for i in train + val}
    eval_folds = {int(samples[i].fold) for i in test}
    assert train_folds == {1, 2}                     # dev minus fold 0
    assert eval_folds == {0}
    train_curves = {samples[i].curve_id for i in train}
    val_curves = {samples[i].curve_id for i in val}
    assert not (train_curves & val_curves)           # selection disjoint
    assert train_curves | val_curves == {samples[i].curve_id
                                         for i in train + val}
    # deterministic; every dev fold usable as eval
    t2, v2, e2 = v52_curve_split(samples, fold=0, seed=17)
    assert (train, val, test) == (t2, v2, e2)
    for fold in (1, 2):
        tr_, va_, te_ = v52_curve_split(samples, fold=fold, seed=29)
        assert {int(samples[i].fold) for i in tr_ + va_} == \
            ({0, 2} if fold == 1 else {0, 1})
        assert {int(samples[i].fold) for i in te_} == {fold}


# ---------------------------------------------------------------------------
# frozen stage schedule + backbone init
# ---------------------------------------------------------------------------


def test_encoder_frozen_phase_actually_frozen(dataset, labels, tmp_path):
    """Stage A (freeze_encoder_updates) leaves backbone grads None."""
    from pes2ts_core.pesn.features.key_atom_graph_v3 import (
        collate_v3, to_torch_batch,
    )
    from pes2ts_core.pesn.offline.training.key_atom_bc_v3 import (
        build_feature_cache, set_seed,
    )
    from pes2ts_core.pesn.offline.training.key_atom_v52_train_v3 import (
        _labels_for_batch, _loss_on_batch, _set_encoder_trainable,
        _set_heads_trainable, stop_class_weights_for_fold,
    )
    set_seed(17)
    model = load_backbone_from_r3(R3_CKPT, device="cpu")
    _set_heads_trainable(model, "QT")
    _set_encoder_trainable(model, False)
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.progress_head.parameters())
    assert all(p.requires_grad for p in model.ts_stop_head.parameters())
    train_idx, _, _ = v52_curve_split(dataset.samples, 0, 17)
    cache = build_feature_cache(dataset, train_idx[:8], level="F2")
    batch = to_torch_batch(collate_v3(cache), device="cpu")
    cfg = V52TrainerConfig(arm="QT", fold=0, seed=17, lambda_q=LQ,
                           lambda_m=LM, backbone_run_id=INIT, device="cpu")
    loss = _loss_on_batch(model, batch, labels, cfg,
                          stop_class_weights_for_fold(0))
    loss["loss"].backward()
    assert all(p.grad is None for p in model.backbone.parameters())  # frozen
    assert any(p.grad is not None for p in model.progress_head.parameters())
    assert any(p.grad is not None for p in model.ts_stop_head.parameters())
    # joint phase unfreezes
    _set_encoder_trainable(model, True)
    assert any(p.requires_grad for p in model.backbone.parameters())


def test_backbone_init_is_verbatim_r3():
    payload = torch.load(str(R3_CKPT), map_location="cpu", weights_only=False)
    model = load_backbone_from_r3(R3_CKPT, device="cpu")
    r3_state = payload["model"]
    v52_backbone = {k[len("backbone."):]: v
                    for k, v in model.state_dict().items()
                    if k.startswith("backbone.")}
    assert set(v52_backbone) == set(r3_state)
    for key, tensor in r3_state.items():
        assert torch.equal(v52_backbone[key], tensor), key
    counts = model.head_parameter_counts()
    assert counts == {"backbone": 55441, "progress_head": 2113,
                      "ts_stop_head": 65}


def test_frozen_stop_class_weights(dataset):
    w0 = stop_class_weights_for_fold(0)
    assert w0.pos_weight == pytest.approx(20.3109756097561)
    assert (w0.n_pos, w0.n_neg) == (164, 3331)
    assert w0.source == "inner-train = dev folds minus fold 0"
    # compiled constants equal the G3a report (drift detection)
    report = json.loads(E4_REPORT.read_text(encoding="utf-8")) \
        if E4_REPORT.is_file() else None
    if report is not None:
        for entry in report["stop_labels"]["class_weights_frozen"]:
            fold = entry.get("fold")
            if fold in (0, 1, 2):
                pw, n_pos, n_neg = FROZEN_STOP_WEIGHTS_V52[fold]
                assert entry["pos_weight"] == pytest.approx(pw)
                assert (entry["n_pos"], entry["n_neg"]) == (n_pos, n_neg)
    with pytest.raises(V52TrainerError):
        stop_class_weights_for_fold(4)   # confirm folds have no frozen weight


def test_label_mapping_masks(dataset, labels):
    """stop=-1 => masked (not in denominator); q=None => q_masked."""
    from pes2ts_core.pesn.offline.training.key_atom_v52_train_v3 import (
        _labels_for_batch,
    )
    masked = [sid for sid, lab in labels.by_sample.items()
              if lab.stop == -1][:4]
    assert masked, "real data must contain middle-window Mask rows"
    batch = {"sample_ids": tuple(masked)}
    lab = _labels_for_batch(batch, labels, "cpu")
    assert not bool(lab["stop_mask"].any())           # all masked out
    assert lab["y_stop"].sum() == 0                   # masked rows are 0-target
    q_masked = [sid for sid in labels.q_masked_sample_ids()][:4]
    if q_masked:
        lab2 = _labels_for_batch({"sample_ids": tuple(q_masked)}, labels, "cpu")
        assert not bool(lab2["q_mask"].any())


# ---------------------------------------------------------------------------
# AUC / pearson math
# ---------------------------------------------------------------------------


def test_auc_and_pearson_math():
    assert auc_mann_whitney([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0
    assert auc_mann_whitney([0.1, 0.2, 0.9, 0.8], [1, 1, 0, 0]) == 0.0
    # ties: all scores equal -> 0.5
    assert auc_mann_whitney([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == 0.5
    # degenerate: no positives -> None
    assert auc_mann_whitney([0.1, 0.2], [0, 0]) is None
    assert pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)
    assert pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None  # constant input


# ---------------------------------------------------------------------------
# fixture mini-run: metrics schema + schedule log
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mini_run(dataset, labels, tmp_path_factory):
    """4-update QT fixture run (freeze=2, joint=2) on real fold0 split."""
    if not R3_CKPT.is_file():
        pytest.skip("R3 winner checkpoint absent")
    out = tmp_path_factory.mktemp("v52mini")
    cfg = V52TrainerConfig(
        arm="QT", fold=0, seed=17, lambda_q=LQ, lambda_m=LM,
        backbone_run_id=INIT, device="cpu",
        freeze_encoder_updates=2, joint_updates=2, val_every=2)
    metrics = tr.train_run_v52(
        dataset, labels, cfg, out, r3_ckpt=R3_CKPT, resume=False)
    log = [json.loads(ln)
           for ln in (out / "train_log.jsonl").read_text().splitlines()]
    return metrics, log, out


def test_mini_run_metrics_schema(mini_run):
    metrics, log, out = mini_run
    for key in ("q_head", "stop_head", "move_guard_vs_r3_winner",
                "stop_class_weights", "schedule", "selected_update",
                "checkpoint_best_sha256", "seeds", "backbone_run_id",
                "lambda_q", "lambda_m", "train_folds", "eval_fold"):
        assert key in metrics, key
    assert metrics["train_folds"] == [1, 2]
    assert metrics["eval_fold"] == 0
    assert metrics["schedule"]["freeze_encoder_updates"] == 2
    assert metrics["schedule"]["joint_updates"] == 2
    q = metrics["q_head"]
    assert q["n_eval_curves"] >= 5
    assert q["constant_q_baseline_mae"] > 0
    s = metrics["stop_head"]
    assert s["mask_excluded"] is True
    assert s["n_eval_unmasked"] > 0
    assert s["auc"] is None or 0.0 <= s["auc"] <= 1.0
    assert (out / "checkpoint_best.pt").is_file()
    assert (out / "checkpoint_last.pt").is_file()
    from pes2ts_core.pesn.offline.training.key_atom_bc_v3 import sha256_file
    assert sha256_file(out / "checkpoint_best.pt") == \
        metrics["checkpoint_best_sha256"]


def test_mini_run_schedule_log_phase_switch(mini_run):
    _, log, _ = mini_run
    train_logs = [e for e in log if "loss" in e and "phase" in e]
    assert train_logs, "no training log lines"
    freeze_lines = [e for e in train_logs if e["phase"] == "freeze"]
    joint_lines = [e for e in train_logs if e["phase"] == "joint"]
    assert freeze_lines and joint_lines
    assert all(not e["encoder_trainable"] for e in freeze_lines)
    assert all(e["encoder_trainable"] for e in joint_lines)
    assert all(np.isfinite(e["loss"]) for e in train_logs)
    val_lines = [e for e in log if "inner_val_loss" in e]
    assert val_lines


# ---------------------------------------------------------------------------
# G3 evidence criteria helpers
# ---------------------------------------------------------------------------


def test_g3_evidence_criteria_helpers():
    ev = pytest.importorskip("scripts.campaigns.r3_g3_evidence")
    assert ev.q_criterion({"mae_curve_median": 0.3,
                           "constant_q_baseline_mae": 0.47,
                           "learnable_above_constant": True})["pass"] is True
    assert ev.q_criterion({"mae_curve_median": 0.5,
                           "constant_q_baseline_mae": 0.47,
                           "learnable_above_constant": False})["pass"] is False
    assert ev.stop_criterion({"auc": 0.57, "n_eval_unmasked": 1525,
                              "n_eval_pos": 87})["pass"] is True
    assert ev.stop_criterion({"auc": 0.5, "n_eval_unmasked": 10,
                              "n_eval_pos": 1})["pass"] is False
    assert ev.stop_criterion({"auc": None})["pass"] is None
    assert ev.move_criterion({"rmse_degradation_pct": 1.2,
                              "within_rmse_guard": True})["pass"] is True
    assert ev.move_criterion({"rmse_degradation_pct": 5.0,
                              "within_rmse_guard": False})["pass"] is False
    # aggregate: failures listed honestly, never dropped
    entries = [
        {"run_id": "q1", "arm": "Q", "fold": 0, "seed": 17,
         "q": {"mae_curve_median": 0.3, "corr_curve_median": 0.4,
               "learnable_above_constant": True},
         "stop": {"auc": 0.55, "precision_at_0.5": 0.2, "recall_at_0.5": 0.5},
         "move": {"rmse_degradation_pct": 0.5, "within_rmse_guard": True},
         "selected_update": 400, "wall_clock_s": 60.0,
         "checkpoint_sha256": "aa" * 32},
        {"run_id": "q2", "arm": "Q", "fold": 1, "seed": 29,
         "q": {"mae_curve_median": 0.5, "corr_curve_median": 0.0,
               "learnable_above_constant": False},
         "stop": {"auc": 0.49, "precision_at_0.5": 0.1, "recall_at_0.5": 0.4},
         "move": {"rmse_degradation_pct": 5.0, "within_rmse_guard": False},
         "selected_update": 200, "wall_clock_s": 60.0,
         "checkpoint_sha256": "bb" * 32},
    ]
    blocked = [{"run_id": "qt-c", "arm": "QT", "fold": 3, "seed": 17,
                "blocked_with_reason": "confirm_awaiting_user_authorization"}]
    pack = ev.build_pack(entries, blocked)
    q_arm = pack["per_arm"]["Q"]
    assert q_arm["n_runs"] == 2
    assert q_arm["failures"]["q_not_learnable"] == ["q2"]
    assert q_arm["failures"]["stop_at_or_below_chance"] == ["q2"]
    assert q_arm["failures"]["move_guard_breach"] == ["q2"]
    assert pack["blocked_runs"][0]["run_id"] == "qt-c"
    # arm-level criteria use aggregates: q learnable rate 1/2 > 0 -> True;
    # per-run failure q2 is still listed honestly above
    assert pack["per_arm_criteria"]["Q"]["q"]["pass"] is True
    # move: median degradation (0.5+5.0)/2 = 2.75% > 2% and rate 1/2 < 1
    assert pack["per_arm_criteria"]["Q"]["move"]["pass"] is False
