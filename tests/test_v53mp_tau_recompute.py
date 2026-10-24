"""τ_end offline recalibration gate unit tests (plan §9.3).

Driver under test: ``scripts/agent_ops/feature_gate/v53mp_tau_recompute.py``
(decision-layer-only recompute over the frozen S3 checkpoints).

Coverage:

* NEW-formula math ``tau_recalibrate``: τ_prec / τ_fpr(q99) / max /
  binding label, precision@τ_new and FPR@τ_new, the NO-absolute-0.50-floor
  guarantee, and both support gates (support < 30, positive curves < 10);
* ``insufficient_support`` path incl. the no-threshold edge;
* OLD-formula reproduction on a fixture: ``calibrate_tau_old`` is the
  trainer's ``calibrate_tau_end`` (known-answer sweep + max-F1 fallback)
  and ``anchor_check`` tolerance semantics (reproduce / drift / None);
* mis-stop metric parity vs a HAND-COMPUTED case AND vs an inline mirror
  of ``evaluate_v53mp`` :1269-1290 (exact formulas: pred_stop over all
  rows, rates averaged over ``keep & start/ts`` where
  ``keep = endpoint_mask & y ∈ {0,1}``, no label_status filter);
* population derivation ``capped_inner_val_positions`` mirrors the
  trainer's v53mp_pool_split + ``seed ^ 0x1E7A1`` cap exactly;
* ``load_run_artifact`` sha binding + τ_old extraction on a tmp fixture;
* markdown rendering states the pre-registered-decision-layer framing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agent_ops.feature_gate.v53mp_tau_recompute import (  # noqa: E402
    anchor_check,
    calibrate_tau_old,
    capped_inner_val_positions,
    load_run_artifact,
    misstop_metrics,
    render_markdown,
    sigmoid_scores,
    tau_recalibrate,
)
from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (  # noqa: E402
    _sigmoid,
    calibrate_tau_end,
    v53mp_pool_split,
)


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


# ---------------------------------------------------------------------------
# sigmoid parity
# ---------------------------------------------------------------------------


def test_sigmoid_scores_matches_trainer_sigmoid() -> None:
    logits = np.linspace(-8.0, 8.0, 33)
    np.testing.assert_array_equal(sigmoid_scores(logits), _sigmoid(logits))


# ---------------------------------------------------------------------------
# NEW formula: math correctness
# ---------------------------------------------------------------------------

def test_tau_recalibrate_prec_binding_separated_classes() -> None:
    # 50 positives @ +2, 100 negatives @ -2: fully separated.
    logits = [2.0] * 50 + [-2.0] * 100
    y = [1] * 50 + [0] * 100
    mask = [True] * 150
    curves = ["RX{:02d}".format(i % 12) for i in range(150)]
    out = tau_recalibrate(
        logits, y, mask, target_precision=0.9, curve_ids=curves)
    tau_prec = float(_sigmoid(np.array([2.0]))[0])
    tau_fpr = float(_sigmoid(np.array([-2.0]))[0])
    assert out["tau_prec"] == pytest.approx(tau_prec)
    assert out["tau_fpr"] == pytest.approx(tau_fpr)
    assert out["tau_new"] == pytest.approx(tau_prec)   # max binding
    assert out["binding"] == "tau_prec"
    assert out["support_pos_at_tau_new"] == 50
    assert out["precision_at_tau_new"] == pytest.approx(1.0)
    assert out["fpr_at_tau_new"] == pytest.approx(0.0)
    assert out["n_pos_curves"] == 12
    assert out["status"] == "ok"
    assert out["status_reason"] is None


def test_tau_recalibrate_fpr_binding_heavy_tail_negatives() -> None:
    # 4 high-logit negatives (q99 lands in the tail) above the positives:
    # τ_fpr must dominate and FPR@τ_new = 4/100.
    logits = [2.0] * 50 + [4.0] * 4 + [-4.0] * 96
    y = [1] * 50 + [0] * 100
    mask = [True] * 150
    out = tau_recalibrate(logits, y, mask, target_precision=0.9)
    tau_hi = float(_sigmoid(np.array([4.0]))[0])
    assert out["tau_fpr"] == pytest.approx(tau_hi)
    assert out["tau_new"] == pytest.approx(tau_hi)
    assert out["binding"] == "tau_fpr"
    assert out["fpr_at_tau_new"] == pytest.approx(0.04)
    # positives at 2.0 < tau_new → support collapses → gate fires
    assert out["support_pos_at_tau_new"] == 0
    assert out["status"] == "insufficient_support"
    assert "support(0)" in out["status_reason"]


def test_tau_recalibrate_no_absolute_floor() -> None:
    # ALL logits below 0.5-sigmoid: τ_new must stay below 0.5 (no floor).
    logits = [-3.0] * 40 + [-6.0] * 100
    y = [1] * 40 + [0] * 100
    mask = [True] * 140
    out = tau_recalibrate(logits, y, mask, target_precision=0.9)
    assert out["tau_new"] is not None
    assert out["tau_new"] < 0.5
    assert out["tau_new"] == pytest.approx(max(out["tau_prec"], out["tau_fpr"]))


def test_tau_recalibrate_precision_actual_values() -> None:
    # 30 positives @ +2, 3 negatives @ +1 (score≈0.732 < positive score),
    # 97 negatives @ -4. τ_prec = min positive score; at τ_new the 3 mild
    # negatives sit below → precision 1.0, FPR 0; verify by hand.
    logits = [2.0] * 30 + [1.0] * 3 + [-4.0] * 97
    y = [1] * 30 + [0] * 100
    mask = [True] * 130
    out = tau_recalibrate(logits, y, mask, target_precision=0.9)
    scores = sigmoid_scores(logits)
    yy = np.asarray(y)
    pred = scores >= out["tau_new"]
    tp = int((pred & (yy == 1)).sum())
    fp = int((pred & (yy == 0)).sum())
    assert out["support_pos_at_tau_new"] == tp == 30
    assert out["precision_at_tau_new"] == pytest.approx(tp / (tp + fp))
    assert out["fpr_at_tau_new"] == pytest.approx(fp / 100)
    assert out["status"] == "ok"


# ---------------------------------------------------------------------------
# NEW formula: support gates / insufficient_support
# ---------------------------------------------------------------------------


def test_tau_recalibrate_support_gate_below_30() -> None:
    # 25 positives (well separated) → support 25 < 30 → gate fires even
    # though precision is perfect and curves are plenty.
    logits = [2.0] * 25 + [-2.0] * 100
    y = [1] * 25 + [0] * 100
    mask = [True] * 125
    curves = ["RX{:02d}".format(i) for i in range(125)]
    out = tau_recalibrate(
        logits, y, mask, target_precision=0.9, curve_ids=curves)
    assert out["support_pos_at_tau_new"] == 25
    assert out["status"] == "insufficient_support"
    assert "support(25) < 30" in out["status_reason"]


def test_tau_recalibrate_pos_curve_gate_below_10() -> None:
    # 40 positives from only 3 curves: support ok, curves gate fires.
    logits = [2.0] * 40 + [-2.0] * 100
    y = [1] * 40 + [0] * 100
    mask = [True] * 140
    curves = (["A"] * 15 + ["B"] * 15 + ["C"] * 10
              + ["N{:03d}".format(i) for i in range(100)])
    out = tau_recalibrate(
        logits, y, mask, target_precision=0.9, curve_ids=curves)
    assert out["support_pos_at_tau_new"] == 40
    assert out["n_pos_curves"] == 3
    assert out["status"] == "insufficient_support"
    assert "positive_curves(3) < 10" in out["status_reason"]


def test_tau_recalibrate_both_gates_reasons() -> None:
    logits = [2.0] * 5 + [-2.0] * 100
    y = [1] * 5 + [0] * 100
    mask = [True] * 105
    curves = ["A"] * 5 + ["N{:03d}".format(i) for i in range(100)]
    out = tau_recalibrate(
        logits, y, mask, target_precision=0.9, curve_ids=curves)
    assert out["status"] == "insufficient_support"
    assert "support(5) < 30" in out["status_reason"]
    assert "positive_curves(1) < 10" in out["status_reason"]


def test_tau_recalibrate_masked_rows_excluded() -> None:
    # Masked high-logit rows must not count as support or FPR mass.
    logits = [6.0] * 10 + [2.0] * 40 + [-2.0] * 100
    y = [1] * 10 + [1] * 40 + [0] * 100
    mask = [False] * 10 + [True] * 140
    out = tau_recalibrate(logits, y, mask, target_precision=0.9)
    assert out["n_unmasked"] == 140
    assert out["n_pos"] == 40
    assert out["support_pos_at_tau_new"] == 40


def test_tau_recalibrate_no_negatives_tau_fpr_none() -> None:
    logits = [2.0] * 40
    y = [1] * 40
    mask = [True] * 40
    out = tau_recalibrate(logits, y, mask, target_precision=0.9)
    assert out["tau_fpr"] is None
    assert out["fpr_at_tau_new"] is None
    assert out["tau_new"] == out["tau_prec"]


def test_tau_recalibrate_curve_ids_length_mismatch() -> None:
    with pytest.raises(ValueError):
        tau_recalibrate(
            [2.0] * 10 + [-2.0] * 10, [1] * 10 + [0] * 10, [True] * 20,
            curve_ids=["only-one"])


def test_tau_recalibrate_max_f1_fallback_reported() -> None:
    # Genuinely non-separable: 11 negatives above the 90 positives —
    # every threshold yields precision < 0.9 (top score is a negative),
    # so the existing logic falls back to max-F1; τ_prec must carry the
    # marker, and τ_new must still equal max(τ_prec, τ_fpr).
    logits = [3.0] * 11 + [1.0] * 90 + [-1.0] * 90
    y = [0] * 11 + [1] * 90 + [0] * 90
    mask = [True] * 191
    out = tau_recalibrate(logits, y, mask, target_precision=0.9)
    assert out["tau_prec_fallback"] == "max_f1"
    assert out["tau_new"] == pytest.approx(max(out["tau_prec"], out["tau_fpr"]))
    assert out["tau_fpr"] == pytest.approx(
        float(_sigmoid(np.array([3.0]))[0]))   # q99 lands in the tail negs


# ---------------------------------------------------------------------------
# OLD-formula reproduction (fixture) + anchor_check
# ---------------------------------------------------------------------------


def test_calibrate_tau_old_fixture_known_answer() -> None:
    # Known sweep: scores [0.1, 0.2, 0.8, 0.9]; smallest τ with
    # precision ≥ 0.9 is 0.8 (tp=2, fp=0).
    logits = [_logit(p) for p in (0.1, 0.2, 0.8, 0.9)]
    y = [0, 0, 1, 1]
    mask = [True] * 4
    old = calibrate_tau_old(logits, y, mask, target_precision=0.9)
    assert old["tau_end"] == pytest.approx(0.8, abs=1e-12)
    assert old["fallback"] is None
    assert old["achieved_precision"] == pytest.approx(1.0)
    # the wrapper IS the trainer function (same object semantics)
    direct = calibrate_tau_end(
        logits, y, mask, target_precision=0.9)
    assert old == direct


def test_calibrate_tau_old_deterministic_reproduction() -> None:
    rng = np.random.RandomState(7)
    logits = rng.normal(0.0, 2.0, size=500).tolist()
    y = (rng.uniform(size=500) < 0.3).astype(int).tolist()
    mask = [True] * 500
    first = calibrate_tau_old(logits, y, mask, target_precision=0.9)
    second = calibrate_tau_old(logits, y, mask, target_precision=0.9)
    assert first["tau_end"] == second["tau_end"]
    anchor = anchor_check(first["tau_end"], second["tau_end"], tol=1e-6)
    assert anchor["reproduced"] is True
    assert anchor["abs_diff"] == 0.0


def test_anchor_check_tolerance_semantics() -> None:
    assert anchor_check(0.5, 0.5, tol=1e-6)["reproduced"] is True
    within = anchor_check(0.5 + 5e-7, 0.5, tol=1e-6)
    assert within["reproduced"] is True
    assert within["abs_diff"] == pytest.approx(5e-7)
    drift = anchor_check(0.5 + 1e-5, 0.5, tol=1e-6)
    assert drift["reproduced"] is False
    assert drift["abs_diff"] == pytest.approx(1e-5)
    none_r = anchor_check(None, 0.5, tol=1e-6)
    assert none_r["reproduced"] is False
    assert none_r["reason"] == "recomputed tau is None"
    none_e = anchor_check(0.5, None, tol=1e-6)
    assert none_e["reproduced"] is False
    assert none_e["reason"] == "expected tau is None"


# ---------------------------------------------------------------------------
# mis-stop metric formula parity
# ---------------------------------------------------------------------------


def test_misstop_metrics_hand_computed_case() -> None:
    logits = [2.0, -2.0, 1.0, -1.0, 3.0, -3.0, 0.0, 5.0]
    y =      [1,    0,    1,    0,    1,    0,    0,   1]
    mask =   [True, True, True, True, True, True, False, True]
    start =  [True, True, False, False, True, False, False, False]
    ts =     [False, False, True, True, False, True, False, True]
    out = misstop_metrics(
        logits, y, mask, start, ts, tau=0.7)
    # pred_stop = sigmoid >= 0.7 → rows 0,2,4,7; keep excludes masked row 6
    assert out["n_unmasked"] == 7
    assert out["n_start_rows"] == 3          # rows 0,1,4 (all unmasked)
    assert out["n_ts_rows"] == 4             # rows 2,3,5,7
    # mis_start: pred_stop over rows 0,1,4 = [1,0,1] → 2/3
    assert out["start_misstop_rate_at_tau"] == pytest.approx(2.0 / 3.0)
    # mis_ts: pred_stop over rows 2,3,5,7 = [1,0,0,1] → 1/2
    assert out["ts_misstop_rate_at_tau"] == pytest.approx(0.5)
    # overall: pred_stop over keep rows 0-5,7 = [1,0,1,0,1,0,1] → 4/7
    assert out["stop_rate_overall_at_tau"] == pytest.approx(4.0 / 7.0)


def test_misstop_metrics_parity_with_trainer_formula() -> None:
    # Random fixture; compare against an inline mirror of
    # evaluate_v53mp :1269-1290 (the exact code the report must reflect).
    rng = np.random.RandomState(20261023)
    logits = rng.normal(0.0, 1.5, size=400).tolist()
    y = (rng.uniform(size=400) < 0.2).astype(int).tolist()
    mask = rng.uniform(size=400) < 0.9
    mask = mask.astype(bool).tolist()
    start = (rng.uniform(size=400) < 0.05).astype(bool).tolist()
    ts = (rng.uniform(size=400) < 0.05).astype(bool).tolist()
    tau = 0.42

    out = misstop_metrics(logits, y, mask, start, ts, tau=tau)

    # --- inline mirror of the trainer block (verbatim formulas) ---
    log = np.asarray(logits, dtype=np.float64)
    ya = np.asarray(y, dtype=np.int64)
    m = np.asarray(mask, dtype=np.bool_)
    keep = m & np.isin(ya, (0, 1))
    pred_stop = _sigmoid(log) >= float(tau)
    start_arr = np.asarray(start, dtype=np.bool_)
    ts_arr = np.asarray(ts, dtype=np.bool_)
    n_start = int(np.sum(keep & start_arr))
    n_ts = int(np.sum(keep & ts_arr))
    mis_start = (float(pred_stop[keep & start_arr].mean())
                 if n_start else None)
    mis_ts = float(pred_stop[keep & ts_arr].mean()) if n_ts else None

    assert out["n_unmasked"] == int(keep.sum())
    assert out["n_start_rows"] == n_start
    assert out["n_ts_rows"] == n_ts
    assert out["start_misstop_rate_at_tau"] == pytest.approx(mis_start)
    assert out["ts_misstop_rate_at_tau"] == pytest.approx(mis_ts)
    assert out["stop_rate_overall_at_tau"] == pytest.approx(
        float(pred_stop[keep].mean()))


def test_misstop_metrics_tau_none_reports_counts_only() -> None:
    out = misstop_metrics(
        [2.0, -2.0], [1, 0], [True, True], [True, False], [False, True],
        tau=None)
    assert out["tau_end"] is None
    assert out["start_misstop_rate_at_tau"] is None
    assert out["ts_misstop_rate_at_tau"] is None
    assert out["stop_rate_overall_at_tau"] is None
    assert out["n_unmasked"] == 2
    assert out["n_start_rows"] == 1
    assert out["n_ts_rows"] == 1


def test_misstop_metrics_tau_zero_all_stop() -> None:
    # sigmoid >= 0 is true for every finite logit → all rates are 1.0.
    out = misstop_metrics(
        [5.0, -5.0, 0.0], [1, 0, 0], [True] * 3,
        [True, False, False], [False, True, False], tau=0.0)
    assert out["start_misstop_rate_at_tau"] == pytest.approx(1.0)
    assert out["ts_misstop_rate_at_tau"] == pytest.approx(1.0)
    assert out["stop_rate_overall_at_tau"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# population derivation: capped inner-val (trainer :1489-1511 mirror)
# ---------------------------------------------------------------------------


def _synthetic_samples(n_curves: int, per_curve: int):
    return [
        SimpleNamespace(curve_id="C{:04d}".format(c), fold=0)
        for c in range(n_curves) for _ in range(per_curve)
    ]


def test_capped_inner_val_positions_mirrors_trainer() -> None:
    samples = _synthetic_samples(50, 10)          # 500 positions
    pool = ["C{:04d}".format(c) for c in range(50)]
    seed = 29
    cap = 40
    got = capped_inner_val_positions(samples, pool, seed, cap)

    _, inner_val = v53mp_pool_split(samples, pool_curve_ids=pool, seed=seed)
    rng = np.random.RandomState(seed ^ 0x1E7A1)
    expected = sorted(rng.choice(
        list(inner_val), size=cap, replace=False).tolist())

    assert got == expected
    assert len(got) == cap
    # cap leaves a deterministic, seed-dependent subset of inner-val
    assert set(got) <= set(int(p) for p in inner_val)


def test_capped_inner_val_positions_no_cap_when_small() -> None:
    samples = _synthetic_samples(20, 5)
    pool = ["C{:04d}".format(c) for c in range(20)]
    got = capped_inner_val_positions(samples, pool, 17, 10_000)
    _, inner_val = v53mp_pool_split(samples, pool_curve_ids=pool, seed=17)
    assert got == [int(p) for p in inner_val]


def test_capped_inner_val_positions_none_cap() -> None:
    samples = _synthetic_samples(20, 5)
    pool = ["C{:04d}".format(c) for c in range(20)]
    got = capped_inner_val_positions(samples, pool, 17, None)
    _, inner_val = v53mp_pool_split(samples, pool_curve_ids=pool, seed=17)
    assert got == [int(p) for p in inner_val]


# ---------------------------------------------------------------------------
# load_run_artifact: sha binding + tau_old extraction
# ---------------------------------------------------------------------------


def _write_fake_run(root: Path, tier: str, arm: str, seed: int,
                    payload: dict) -> Path:
    run_dir = root / "v53mp_{}_{}_s{}".format(tier, arm, seed)
    run_dir.mkdir(parents=True)
    ckpt = run_dir / "checkpoint_best.pt"
    ckpt.write_bytes(b"fake-checkpoint-bytes")
    from pes2ts_core.utils.hashing import sha256_file
    doc = dict(payload)
    doc.setdefault("checkpoint_best_sha256", sha256_file(str(ckpt)))
    (run_dir / "metrics.json").write_text(
        json.dumps(doc), encoding="utf-8")
    return run_dir


def test_load_run_artifact_extracts_tau_old(tmp_path: Path) -> None:
    _write_fake_run(tmp_path, "20k", "C", 29, {
        "endpoint_head": {"tau_end": 0.218748534226792},
        "tau_end_calibration": {
            "target_precision": 0.9, "fallback": "max_f1",
            "n_inner_val_unmasked": 6842},
        "inner_val_max_samples": 8192,
        "n_dev_samples_total": 135926,
    })
    art = load_run_artifact(tmp_path, "20k", "C", 29)
    assert art["run_id"] == "v53mp_20k_C_s29"
    assert art["tau_old"] == pytest.approx(0.218748534226792)
    assert art["tau_target_precision"] == 0.9
    assert art["inner_val_max_samples"] == 8192
    assert art["metrics_n_inner_val_unmasked"] == 6842
    assert len(art["ckpt_sha256"]) == 64


def test_load_run_artifact_sha_drift_refused(tmp_path: Path) -> None:
    _write_fake_run(tmp_path, "20k", "A", 17, {
        "endpoint_head": {"tau_end": 0.5}})
    # tamper the checkpoint AFTER the metrics binding was written
    run_dir = tmp_path / "v53mp_20k_A_s17"
    (run_dir / "checkpoint_best.pt").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="sha256"):
        load_run_artifact(tmp_path, "20k", "A", 17)


def test_load_run_artifact_missing_run_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_run_artifact(tmp_path, "20k", "B", 43)


def test_load_run_artifact_falls_back_to_tau_calibration(tmp_path: Path) -> None:
    _write_fake_run(tmp_path, "20k", "B", 17, {
        "tau_end_calibration": {
            "tau_end": 0.6168, "target_precision": 0.9}})
    art = load_run_artifact(tmp_path, "20k", "B", 17)
    assert art["tau_old"] == pytest.approx(0.6168)


# ---------------------------------------------------------------------------
# markdown rendering: framing discipline
# ---------------------------------------------------------------------------


def test_render_markdown_states_pre_registered_framing() -> None:
    doc = {
        "schema": "pesn_v53mp_tau_recompute_v1",
        "created_at": "2026-10-23T00:00:00Z",
        "config": {
            "tier": "20k", "target_precision": 0.9, "fpr_quantile": 0.99,
            "min_support": 30, "min_pos_curves": 10, "anchor_tol": 1e-6,
            "n_dev_positions": 135926,
            "code_commit": "abc123def456", "code_commit_dirty": False,
            "dataset_manifest_sha256": "d" * 64,
            "split_manifest_sha256": "s" * 64,
            "label_cache_json_sha256": "j" * 64,
            "label_cache_npz_sha256": "n" * 64,
        },
        "runs": [{
            "run_id": "v53mp_20k_C_s29", "arm": "C", "seed": 29,
            "status": "ok",
            "tau_old_metrics": 0.218748534226792,
            "tau_old_recomputed": 0.218748534226792,
            "anchor": {"reproduced": True, "abs_diff": 0.0},
            "anchor_n_unmasked_recomputed": 6842,
            "anchor_n_unmasked_metrics": 6842,
            "recalibration": {
                "tau_new": 0.91, "precision_at_tau_new": 1.0,
                "fpr_at_tau_new": 0.009, "support_pos_at_tau_new": 35,
                "n_pos_curves": 12, "status": "ok"},
            "misstop": {
                "at_tau_old": {"start_misstop_rate_at_tau": 0.6314,
                               "ts_misstop_rate_at_tau": 0.6224},
                "at_tau_new": {"start_misstop_rate_at_tau": 0.0,
                               "ts_misstop_rate_at_tau": 0.0}},
        }],
        "summary": {
            "sanity_anchor": {"all_reproduced": True, "n_runs": 1,
                              "n_failed": 0, "max_abs_diff": 0.0,
                              "tolerance": 1e-6},
            "plan_target_s29": {
                "run_id": "v53mp_20k_C_s29",
                "start_misstop_old": 0.6314, "start_misstop_new": 0.0,
                "ts_misstop_old": 0.6224, "ts_misstop_new": 0.0,
                "target": 0.01, "start_met": True, "ts_met": True,
                "status": "ok"},
        },
    }
    md = render_markdown(doc)
    assert "预注册" in md
    assert "不触碰训练与 scaling 可比性" in md
    assert "S4 启动前需要单独签字" in md
    assert "insufficient_support" in md
    assert "0.6314" in md
    assert "PASS" in md


def test_render_markdown_anchor_failure_banner() -> None:
    doc = {
        "schema": "pesn_v53mp_tau_recompute_v1",
        "created_at": "2026-10-23T00:00:00Z",
        "config": {
            "tier": "20k", "target_precision": 0.9, "fpr_quantile": 0.99,
            "min_support": 30, "min_pos_curves": 10, "anchor_tol": 1e-6,
            "n_dev_positions": 100,
            "code_commit": "abc123def456", "code_commit_dirty": False,
            "dataset_manifest_sha256": "d" * 64,
            "split_manifest_sha256": "s" * 64,
            "label_cache_json_sha256": "j" * 64,
            "label_cache_npz_sha256": "n" * 64,
        },
        "runs": [{
            "run_id": "v53mp_20k_A_s17", "arm": "A", "seed": 17,
            "status": "anchor_failed",
            "tau_old_metrics": 0.436,
            "anchor": {"reproduced": False, "abs_diff": 0.01},
            "anchor_n_unmasked_recomputed": 6800,
            "anchor_n_unmasked_metrics": 6854,
        }],
        "summary": {
            "sanity_anchor": {"all_reproduced": False, "n_runs": 1,
                              "n_failed": 1, "max_abs_diff": 0.01,
                              "tolerance": 1e-6},
        },
    }
    md = render_markdown(doc)
    assert "**FAIL**" in md
    assert "不予采信" in md
    assert "ANCHOR FAILED" in md
