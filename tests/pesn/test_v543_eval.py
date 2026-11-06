"""V5.4.3 — frozen V543 evaluator, CPU-fast synthetic certificate.

Every metric of ``pes2ts_core/pesn/offline/evaluation/v543_eval.py`` is checked
against a HAND-COMPUTED value on a tiny synthetic frame set (2–3 curves, a
handful of frames, known teacher top-3 sets).  The three structural guarantees
the campaign depends on are asserted explicitly:

1. **``fixed_coverage_direction@3`` is NOT the V542 direction leaf** — on the
   same synthetic data its value differs from
   ``sign_accuracy_cluster_median`` (plan §1.5 :108 / §2 F3:
   ``scripts/agent_ops/v542_stats.py`` :177–190 records that V542's direction
   IS ``sre_and_sign.sign_accuracy_cluster_median``; a rename would reproduce
   the duplicate-leaf defect);
2. **the q-validity rule of plan §9.4 :1029–1033** — a curve with fewer than
   3 valid q points never enters a correlation, and ``frac_curves_ge3 < 0.70``
   marks the correlation diagnostic-only;
3. **determinism** — two runs of the same input produce byte-identical JSON
   (the property that lets a baseline be SHA-frozen, §9.5 :1037).

No data, no checkpoints, no torch model: ``pytest`` alone, CPU, milliseconds.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from pes2ts_core.pesn.offline.evaluation.v543_eval import (
    ATOM_DEAD_THRESHOLD,
    FIXED_COVERAGE_TOPK,
    FPR_CONSTRAINT,
    Q_CORR_MIN_POINTS,
    Q_FRAC_GE3_MIN,
    REPORT_SCHEMA,
    SIGN_PAIR_THRESHOLD,
    EvalFrame,
    V543EvalError,
    evaluate_baseline,
    evaluate_v543,
    evaluator_spec,
    evaluator_spec_sha256,
    exact_delta_d,
    fixed_coverage_at_k,
    frame_terms,
    paired_curve_bootstrap,
    paired_curve_bootstrap_multiseed,
    recall_at_fpr,
    soft_cosine,
    topk_pair_indices,
    zero_residual_frames,
)

#: A pair's teacher Δd equals the displacement difference along x when
#: ``r = (1,0,0)``: ``Δd = |1 + u_j,x − u_i,x| − 1 = u_j,x − u_i,x`` for any
#: ``|u_j,x − u_i,x| <= 1``.  The helper below builds exactly that geometry, so
#: every hand-computed Δd below is exact in float64.
PAIR_DISTANCE = 1.0


def _pair_frame(curve_id: str, deltas_y, deltas_u, *, direction: str = "forward",
                **kwargs) -> EvalFrame:
    """One frame whose ``n`` pairs produce ``Δd_y = deltas_y``/``Δd_u = deltas_u``.

    Pair ``l`` uses key slots ``(2l, 2l+1)`` with ``r_l = (1,0,0)``; only the
    x component of the displacement is set, so ``Δd = Δu_x`` exactly (see
    :data:`PAIR_DISTANCE`).
    """
    n = len(deltas_y)
    assert len(deltas_u) == n
    y = np.zeros((2 * n, 3), dtype=np.float64)
    u = np.zeros((2 * n, 3), dtype=np.float64)
    slots = np.zeros((n, 2), dtype=np.int64)
    offsets = np.zeros((n, 3), dtype=np.float64)
    for l, (dy, du) in enumerate(zip(deltas_y, deltas_u)):
        slots[l] = (2 * l, 2 * l + 1)
        offsets[l] = (PAIR_DISTANCE, 0.0, 0.0)
        y[2 * l + 1, 0] = float(dy)
        u[2 * l + 1, 0] = float(du)
    return EvalFrame(curve_id=curve_id, u=u, y=y, pair_slots=slots,
                     pair_offsets=offsets, direction=direction, **kwargs)


def _plain_frame(curve_id: str, u, y, *, direction: str = "forward",
                 **kwargs) -> EvalFrame:
    return EvalFrame(curve_id=curve_id, u=np.asarray(u, dtype=np.float64),
                     y=np.asarray(y, dtype=np.float64),
                     direction=direction, **kwargs)


# ---------------------------------------------------------------------------
# geometry: the plan §9.2 Δd
# ---------------------------------------------------------------------------
def test_exact_delta_d_matches_the_plan_formula_by_hand() -> None:
    # r = (2,0,0); u_i = (0.1,0,0), u_j = (0,0,0)
    # ‖r + u_j − u_i‖ − ‖r‖ = ‖(1.9,0,0)‖ − 2 = −0.1   (plan §9.2 :901–902)
    dd = exact_delta_d(np.asarray([[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]]),
                       np.asarray([[0, 1]], dtype=np.int64),
                       np.asarray([[2.0, 0.0, 0.0]]))
    assert dd.shape == (1,)
    assert dd[0] == pytest.approx(-0.1, abs=1e-12)

    # independent check of the plan's own written form:
    # ‖x_i + u_i − (x_j + u_j)‖ − ‖x_i − x_j‖ with x_i = 0, x_j = (2,0,0)
    x_i = np.zeros(3)
    x_j = np.asarray([2.0, 0.0, 0.0])
    manual = (np.linalg.norm(x_i + np.asarray([0.1, 0, 0])
                             - (x_j + np.zeros(3)))
              - np.linalg.norm(x_i - x_j))
    assert manual == pytest.approx(dd[0], abs=1e-12)


def test_exact_delta_d_empty_pair_set_is_empty() -> None:
    dd = exact_delta_d(np.zeros((1, 3)), np.zeros((0, 2), dtype=np.int64),
                       np.zeros((0, 3)))
    assert dd.shape == (0,)


# ---------------------------------------------------------------------------
# the new metric: plan §9.2 affirmations
# ---------------------------------------------------------------------------
def test_topk_pair_indices_ties_break_by_pair_index() -> None:
    # plan §9.2 :904 "ties 按 pair 索引"
    assert topk_pair_indices([0.03, 0.03, 0.09], 3).tolist() == [2, 0, 1]
    assert topk_pair_indices([0.05, -0.05, 0.01], 2).tolist() == [0, 1]
    assert topk_pair_indices([0.1, 0.2], 5).tolist() == [1, 0]


def test_fixed_coverage_at_3_by_hand() -> None:
    dd_y = np.asarray([0.10, -0.05, 0.03, -0.02])
    dd_u = np.asarray([0.12, -0.01, -0.05, 0.02])
    out = fixed_coverage_at_k(dd_y, dd_u)
    # T_3 = pairs 0 (|0.10|), 1 (|0.05|), 2 (|0.03|) of the teacher
    assert out["teacher_topk"] == (0, 1, 2)
    # P_3 = pairs 0 (|0.12|), 2 (|0.05|), 3 (|0.02|) of the prediction
    assert out["prediction_topk"] == (0, 2, 3)
    # pair 0: sign +/+ and |0.12| >= 0.005  -> hit
    # pair 1: sign -/- and |0.01| >= 0.005  -> hit
    # pair 2: sign -/+                      -> miss
    assert out["hits"] == (True, True, False)
    assert out["direction"] == pytest.approx(2.0 / 3.0)
    assert out["topk_overlap"] == pytest.approx(2.0 / 3.0)
    assert out["k"] == 3


def test_fixed_coverage_at_3_dead_zone_rule() -> None:
    # |Δd_u| < 0.005 Å counts INCORRECT even with the right sign (§9.2 :906)
    out = fixed_coverage_at_k(np.asarray([0.5, 0.5]),
                              np.asarray([0.5, 0.002]))
    assert out["hits"] == (True, False)
    assert out["direction"] == pytest.approx(0.5)


def test_fixed_coverage_at_3_k_is_min_of_three_and_n_pairs() -> None:
    out = fixed_coverage_at_k(np.asarray([0.2, -0.1]), np.asarray([0.2, -0.1]))
    assert out["k"] == 2                      # min(3, n_pairs) — §9.2 :903
    assert out["direction"] == pytest.approx(1.0)
    out = fixed_coverage_at_k(np.asarray([0.2]), np.asarray([-0.2]))
    assert out["k"] == 1
    assert out["direction"] == pytest.approx(0.0)
    empty = fixed_coverage_at_k(np.zeros(0), np.zeros(0))
    assert empty["direction"] is None and empty["topk_overlap"] is None


def test_fixed_coverage_at_3_curved_geometry_agrees_with_exact_delta_d() -> None:
    """The metric consumes ``exact_delta_d`` — no second geometry path."""
    frame = _pair_frame("c", [0.10, -0.05, 0.03], [0.12, -0.01, -0.05])
    terms = frame_terms(frame)
    direct = fixed_coverage_at_k(
        exact_delta_d(frame.y, frame.pair_slots, frame.pair_offsets),
        exact_delta_d(frame.u, frame.pair_slots, frame.pair_offsets))
    assert terms.fixed_direction == pytest.approx(direct["direction"])
    assert terms.topk_overlap == pytest.approx(direct["topk_overlap"])


# ---------------------------------------------------------------------------
# the anti-duplicate-leaf guarantee (plan §1.5 :108 / §2 F3)
# ---------------------------------------------------------------------------
def test_new_direction_is_not_the_v542_sign_leaf() -> None:
    """Hand-computed data where the two metrics MUST disagree.

    Pairs (teacher Δd, prediction Δd):
        0  ( 0.50, -0.50)   key pair, wrong sign
        1  ( 0.20,  0.20)   right sign
        2  ( 0.10, -0.10)   key pair, wrong sign
        3  ( 0.02,  0.02)   right sign
        4  ( 0.01,  0.01)   right sign
        5  (0.008, 0.008)   right sign
    V542 leaf: all six are teacher-significant (|Δd_y| >= 0.005) -> 4/6.
    New metric: T_3 = {0,1,2} (the teacher's largest |Δd|) -> 1/3.
    """
    frame = _pair_frame(
        "c",
        [0.50, 0.20, 0.10, 0.02, 0.01, 0.008],
        [-0.50, 0.20, -0.10, 0.02, 0.01, 0.008],
        q_pred=0.0, q_true=0.0, q_mask=True,
    )
    terms = frame_terms(frame)
    assert terms.n_active_pairs == 6
    assert terms.sign_accuracy == pytest.approx(4.0 / 6.0)
    assert terms.fixed_direction == pytest.approx(1.0 / 3.0)
    assert terms.sign_accuracy != pytest.approx(terms.fixed_direction)

    report = evaluate_v543([frame], train_q_mean=0.0)
    move = report["pooled"]["move"]
    assert move["sign_accuracy_cluster_median"] == pytest.approx(4.0 / 6.0)
    assert move["fixed_coverage"]["direction_at_3"] == pytest.approx(1.0 / 3.0)
    # and the two are DIFFERENT numbers from the same single frame
    assert (move["sign_accuracy_cluster_median"]
            != move["fixed_coverage"]["direction_at_3"])


def test_fixed_coverage_direction_is_a_distinct_implementation() -> None:
    """Structural guard: the new metric has its own tail (top-k) and threshold."""
    import inspect

    import pes2ts_core.pesn.offline.evaluation.v543_eval as module

    source = inspect.getsource(module.fixed_coverage_at_k)
    assert "topk_pair_indices" in source          # the teacher top-k tail
    assert "sign(" in source                      # sign agreement
    # it must not delegate to the frozen V542 sign leaf
    assert "sign_accuracy" not in source


# ---------------------------------------------------------------------------
# MOVE: rmse_zero / dead zone / cosine / pair MAE
# ---------------------------------------------------------------------------
def test_rmse_zero_by_hand() -> None:
    """Frame A perfect; frame B predicts zero against ‖y‖=2.

    Per curve: sqrt(mean_f mse) = sqrt((0 + 4)/2) = sqrt(2),
               sqrt(mean_f mse_zero) = sqrt((0.5 + 4)/2) = 1.5
               → rmse_zero = sqrt(2)/1.5 = 0.9428090415820634.
    """
    frame_a = _plain_frame("c", [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                           [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    frame_b = _plain_frame("c", [[0.0, 0.0, 0.0]], [[0.0, 2.0, 0.0]])
    report = evaluate_v543([frame_a, frame_b])
    row = report["per_curve"][0]
    assert row["curve_id"] == "c"
    assert row["move"]["rmse"] == pytest.approx(math.sqrt(2.0))
    assert row["move"]["rmse_zero_denominator"] == pytest.approx(1.5)
    assert row["move"]["rmse_zero"] == pytest.approx(math.sqrt(2.0) / 1.5)
    # one curve -> the V542 median over clusters is the same number
    assert report["pooled"]["move"]["rmse_zero"] == pytest.approx(
        math.sqrt(2.0) / 1.5)


def test_rmse_zero_is_undefined_when_the_teacher_is_zero() -> None:
    frame = _plain_frame("c", np.zeros((2, 3)), np.zeros((2, 3)))
    report = evaluate_v543([frame])
    assert report["per_curve"][0]["move"]["rmse_zero"] is None
    assert report["pooled"]["move"]["rmse_zero"] is None


def test_dead_zone_fraction_and_sign_by_hand() -> None:
    frame = _pair_frame("c", [0.1, 0.1], [0.001, 0.1])
    report = evaluate_v543([frame])
    move = report["pooled"]["move"]
    # both pairs are teacher-active; one prediction stays in the dead zone
    assert move["dead_zone_fraction"] == pytest.approx(0.5)
    assert move["dead_fraction"] == pytest.approx(0.5)
    # the dead-zone pair is also a WRONG sign decision (|Δd_u| < 0.005)
    assert move["sign_accuracy_cluster_median"] == pytest.approx(0.5)


def test_atom_dead_fraction_by_hand() -> None:
    u = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.004, 0.0],
                    [0.0, 0.0, 0.02]])
    y = np.asarray([[0.1, 0, 0], [0.5, 0, 0], [0.1, 0, 0], [0.1, 0, 0]])
    report = evaluate_v543([_plain_frame("c", u, y)])
    move = report["pooled"]["move"]
    # ‖u‖ < 0.005 for rows 0 and 2 → 2/4
    assert move["atom_dead_fraction"] == pytest.approx(0.5)
    assert ATOM_DEAD_THRESHOLD == pytest.approx(0.005)


def test_cos_eff_median_counts_only_effective_atoms() -> None:
    # atom 0: y = (0.5,0,0) effective, prediction aligned -> cos ≈ +1
    # atom 1: y = (0,0,0.005) below 0.02 Å -> NOT an effective atom
    u = np.asarray([[0.5, 0.0, 0.0], [0.0, 0.0, 0.005]])
    y = np.asarray([[0.5, 0.0, 0.0], [0.0, 0.0, 0.005]])
    report = evaluate_v543([_plain_frame("c", u, y)])
    move = report["pooled"]["move"]
    assert move["n_effective_atoms"] == 1
    assert move["cos_eff_median"] == pytest.approx(1.0)
    # the module's cosine is the frozen G5.2 formula
    from pes2ts_core.pesn.offline.evaluation.key_atom_protocols import (
        _soft_cosine_matrix,
    )

    frozen = _soft_cosine_matrix(u, y)
    assert soft_cosine(u, y) == pytest.approx(frozen, abs=1e-15)


def test_pair_delta_d_mae_by_hand() -> None:
    frame = _pair_frame("c", [0.10, -0.05], [0.12, -0.07])
    report = evaluate_v543([frame])
    # |0.12−0.10| + |−0.07−(−0.05)| = 0.02 + 0.02 → mean 0.02
    assert report["pooled"]["move"]["pair_delta_d_mae"] == pytest.approx(0.02)
    assert report["per_curve"][0]["move"]["pair_delta_d_mae"] == pytest.approx(
        0.02)


def test_basis_projection_residual_is_reported_when_key_v_is_given() -> None:
    # key_v channel 0 = (1,0,0); teacher residual is along x → residual 0
    key_v = np.zeros((1, 8, 3))
    key_v[0, 0] = (1.0, 0.0, 0.0)
    key_v[0, 1] = (0.0, 1.0, 0.0)
    y = np.asarray([[0.3, 0.0, 0.0]])
    u0 = np.asarray([[0.0, 0.0, 0.0]])
    frame = _plain_frame("c", u0, y, u0=u0, key_v=key_v)
    report = evaluate_v543([frame])
    assert report["pooled"]["move"]["n_basis_rows"] == 1
    assert report["pooled"]["move"][
        "basis_projection_residual_median"] == pytest.approx(0.0, abs=1e-12)
    # without key_v the diagnostic is absent, never fabricated
    plain = evaluate_v543([_plain_frame("c", u0, y)])
    assert plain["pooled"]["move"]["basis_projection_residual_median"] is None
    assert plain["pooled"]["move"]["n_basis_rows"] == 0


def test_move_ineligible_frames_stay_out_of_move_but_keep_process() -> None:
    """§4.1 ``_frame_mask_for``: only ``label_status == \"ok\"`` frames have y."""
    eligible = _pair_frame("c", [0.10], [0.10], q_pred=0.0, q_true=0.0,
                           q_mask=True, endpoint_logit=0.0, endpoint_label=0,
                           endpoint_mask=True)
    terminal = _pair_frame("c", [99.0], [0.0], q_pred=1.0, q_true=1.0,
                           q_mask=True, endpoint_logit=0.0, endpoint_label=1,
                           endpoint_mask=True, move_eligible=False)
    report = evaluate_v543([eligible, terminal])
    row = report["per_curve"][0]
    assert row["move"]["n_frames_move_eligible"] == 1
    assert report["validity"]["move"]["n_frames"] == 2
    assert report["validity"]["move"]["n_frames_move_eligible"] == 1
    assert report["validity"]["move"][
        "n_frames_excluded_by_label_status"] == 1
    # the terminal frame's y=99 never reaches a MOVE denominator:
    # rmse_zero would be enormous if it did
    assert row["move"]["rmse_zero"] == pytest.approx(0.0, abs=1e-12)
    # PROCESS still sees both frames (terminal rows keep endpoint labels)
    assert report["pooled"]["process"]["q"]["n_eval_curves"] == 1
    assert report["pooled"]["process"]["q"]["total_points"] == 2
    assert report["pooled"]["process"]["endpoint"]["n_unmasked"] == 2


def test_fixed_coverage_stratified_breakdown() -> None:
    frame_ts = _pair_frame("c", [0.1, 0.05], [0.1, -0.05],
                           stratum="post_TS", intent_class="breaking")
    frame_start = _pair_frame("c", [0.1, 0.05], [0.1, 0.05],
                              stratum="start", intent_class="forming")
    report = evaluate_v543([frame_ts, frame_start])
    block = report["pooled"]["move"]["fixed_coverage"]
    assert block["by_stratum"]["post_TS"]["n_frames"] == 1
    assert block["by_stratum"]["post_TS"][
        "direction_at_3_frame_mean"] == pytest.approx(0.5)
    assert block["by_stratum"]["start"][
        "direction_at_3_frame_mean"] == pytest.approx(1.0)
    assert block["by_intent_class"]["breaking"]["n_frames"] == 1
    assert block["by_intent_class"]["forming"]["n_frames"] == 1
    # frame mean -> curve mean: the curve value is the mean of the two frames
    assert block["direction_at_3"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# PROCESS: q metrics + the §9.4 validity rule
# ---------------------------------------------------------------------------
def _q_report():
    frames = []
    # curve A: 3 points, perfect
    for pred, true in ((0.0, 0.0), (0.5, 0.5), (1.0, 1.0)):
        frames.append(_plain_frame("a", [[0.0, 0, 0]], [[0.0, 0, 0]],
                                   q_pred=pred, q_true=true, q_mask=True))
    # curve B: 2 points only -> never enters a correlation (§9.4 :1031)
    for pred, true in ((0.0, 0.0), (0.0, 1.0)):
        frames.append(_plain_frame("b", [[0.0, 0, 0]], [[0.0, 0, 0]],
                                   q_pred=pred, q_true=true, q_mask=True))
    # curve C: 1 point, q_true 0.5
    frames.append(_plain_frame("c", [[0.0, 0, 0]], [[0.0, 0, 0]],
                               q_pred=0.5, q_true=0.5, q_mask=True))
    # a q-masked-out frame must never enter anything
    frames.append(_plain_frame("c", [[0.0, 0, 0]], [[0.0, 0, 0]],
                               q_pred=0.0, q_true=0.0, q_mask=False))
    return evaluate_v543(frames, train_q_mean=0.5)


def test_q_validity_rule_by_hand() -> None:
    report = _q_report()
    q = report["pooled"]["process"]["q"]
    assert q["n_eval_curves"] == 3
    assert q["median_points_per_curve"] == pytest.approx(2.0)
    assert q["total_points"] == 6
    assert q["frac_curves_ge3"] == pytest.approx(1.0 / 3.0)
    assert q["frac_curves_ge5"] == pytest.approx(0.0)
    # only curve A (3 points) may contribute a correlation
    assert q["n_curves_corr"] == 1
    assert q["corr_curve_median"] == pytest.approx(1.0)
    # the V542 any-2-point reading is kept as a historical anchor only
    assert q["corr_curve_median_v542_anchor"] is not None
    # frac_curves_ge3 = 1/3 < 0.70 → diagnostic only (§9.4 :1032)
    assert q["corr_diagnostic_only"] is True
    assert report["validity"]["q"]["corr_diagnostic_only"] is True
    assert report["validity"]["q"]["frac_ge3_threshold"] == Q_FRAC_GE3_MIN


def test_q_mae_curve_median_and_constant_baseline_by_hand() -> None:
    report = _q_report()
    q = report["pooled"]["process"]["q"]
    # per-curve MAE: A 0, B (|0−0|+|0−1|)/2 = 0.5, C 0 → median 0, mean 1/6
    assert q["mae_curve_median"] == pytest.approx(0.0)
    assert q["mae_curve_mean"] == pytest.approx(0.5 / 3.0)
    # constant baseline over ALL masked points: |0.5−0|+|0.5−0.5|+|0.5−1|
    #   + |0.5−0|+|0.5−1| + |0.5−0.5| = 1 + 1 + 0 = 2 over 6 points
    assert q["constant_q_baseline_mae"] == pytest.approx(2.0 / 6.0)
    assert q["constant_q_value"] == pytest.approx(0.5)
    assert q["learnable_above_constant"] is True


def test_q_curve_with_three_points_can_correlate_negative() -> None:
    frames = [
        _plain_frame("c", [[0.0, 0, 0]], [[0.0, 0, 0]], q_pred=1.0, q_true=0.0,
                     q_mask=True),
        _plain_frame("c", [[0.0, 0, 0]], [[0.0, 0, 0]], q_pred=0.5, q_true=0.5,
                     q_mask=True),
        _plain_frame("c", [[0.0, 0, 0]], [[0.0, 0, 0]], q_pred=0.0, q_true=1.0,
                     q_mask=True),
    ]
    row = evaluate_v543(frames)["per_curve"][0]
    assert row["q"]["n_points"] == 3
    assert row["q"]["corr"] == pytest.approx(-1.0)
    assert row["q"]["mae"] == pytest.approx(2.0 / 3.0)


# ---------------------------------------------------------------------------
# PROCESS: ordering (frozen V542 pair counting)
# ---------------------------------------------------------------------------
def test_ordering_pairwise_accuracy_by_hand() -> None:
    frames = [
        _plain_frame("c", [[0.0, 0, 0]], [[0.0, 0, 0]], q_pred=0.0, q_true=0.0,
                     q_mask=True),
        _plain_frame("c", [[0.0, 0, 0]], [[0.0, 0, 0]], q_pred=0.6, q_true=0.5,
                     q_mask=True),
        _plain_frame("c", [[0.0, 0, 0]], [[0.0, 0, 0]], q_pred=0.9, q_true=1.0,
                     q_mask=True),
    ]
    report = evaluate_v543(frames)
    ordering = report["pooled"]["process"]["ordering"]
    assert ordering["n_pairs_full"] == 3
    assert ordering["pairwise_accuracy_full"] == pytest.approx(1.0)
    # q_true never changes side here → no cross-TS pair (§9.1 :893)
    assert ordering["n_pairs_cross_ts"] == 0
    assert ordering["pairwise_accuracy_cross_ts"] is None


def test_ordering_cross_ts_counts_only_opposite_side_pairs() -> None:
    frames = [
        _plain_frame("c", [[0.0, 0, 0]], [[0.0, 0, 0]], q_pred=-0.4,
                     q_true=-0.5, q_mask=True),
        _plain_frame("c", [[0.0, 0, 0]], [[0.0, 0, 0]], q_pred=0.2, q_true=0.5,
                     q_mask=True),
    ]
    ordering = evaluate_v543(frames)["pooled"]["process"]["ordering"]
    assert ordering["n_pairs_full"] == 1
    assert ordering["n_pairs_cross_ts"] == 1
    assert ordering["pairwise_accuracy_cross_ts"] == pytest.approx(1.0)


def test_ordering_is_grouped_by_direction() -> None:
    """Pairs only exist inside one (curve, intent direction) group."""
    frames = [
        _plain_frame("c", [[0.0, 0, 0]], [[0.0, 0, 0]], q_pred=0.0, q_true=0.0,
                     q_mask=True, direction="forward"),
        _plain_frame("c", [[0.0, 0, 0]], [[0.0, 0, 0]], q_pred=0.0, q_true=0.0,
                     q_mask=True, direction="reverse"),
        _plain_frame("c", [[0.0, 0, 0]], [[0.0, 0, 0]], q_pred=1.0, q_true=1.0,
                     q_mask=True, direction="reverse"),
    ]
    ordering = evaluate_v543(frames)["pooled"]["process"]["ordering"]
    assert ordering["n_pairs_full"] == 1     # only inside "reverse"


# ---------------------------------------------------------------------------
# PROCESS: endpoint AUPRC / Brier / ECE / FPR-constrained recall
# ---------------------------------------------------------------------------
def test_endpoint_brier_and_ece_by_hand() -> None:
    logit = math.log(4.0)                     # sigmoid = 0.8 exactly
    frames = [
        _plain_frame("c", [[0.0, 0, 0]], [[0.0, 0, 0]], endpoint_logit=logit,
                     endpoint_label=1, endpoint_mask=True),
        _plain_frame("c", [[0.0, 0, 0]], [[0.0, 0, 0]], endpoint_logit=logit,
                     endpoint_label=0, endpoint_mask=True),
        # masked endpoint row: never enters a denominator
        _plain_frame("c", [[0.0, 0, 0]], [[0.0, 0, 0]], endpoint_logit=0.0,
                     endpoint_label=-1, endpoint_mask=False),
    ]
    endpoint = evaluate_v543(frames)["pooled"]["process"]["endpoint"]
    assert endpoint["n_unmasked"] == 2
    assert endpoint["n_pos"] == 1 and endpoint["n_neg"] == 1
    # brier = ((0.8−1)² + (0.8−0)²)/2 = (0.04 + 0.64)/2 = 0.34
    assert endpoint["brier"] == pytest.approx(0.34)
    # one bin: accuracy 0.5, confidence 0.8 → ECE = |0.5 − 0.8| = 0.3
    assert endpoint["ece"] == pytest.approx(0.3)
    assert endpoint["ece_bins"] == 10
    assert endpoint["tau_end"] is None
    assert endpoint["start_misstop_rate_at_tau"] is None
    assert endpoint["ts_misstop_rate_at_tau"] is None


def test_endpoint_auprc_auroc_by_hand() -> None:
    # scores 1.0/0.5 with labels 1/0 → AUPRC = 1.0, AUROC = 1.0
    frames = [
        _plain_frame("c", [[0.0, 0, 0]], [[0.0, 0, 0]], endpoint_logit=40.0,
                     endpoint_label=1, endpoint_mask=True),
        _plain_frame("c", [[0.0, 0, 0]], [[0.0, 0, 0]], endpoint_logit=-40.0,
                     endpoint_label=0, endpoint_mask=True),
    ]
    endpoint = evaluate_v543(frames)["pooled"]["process"]["endpoint"]
    assert endpoint["auprc"] == pytest.approx(1.0)
    assert endpoint["auroc"] == pytest.approx(1.0)


def test_recall_at_fpr_constraint_by_hand() -> None:
    out = recall_at_fpr([0.9, 0.8, 0.7, 0.6], [1, 1, 0, 0])
    assert out["fpr_constraint"] == pytest.approx(FPR_CONSTRAINT)
    # threshold 0.8: tp=2, fp=0 → recall 1.0, fpr 0.0
    assert out["threshold"] == pytest.approx(0.8)
    assert out["recall"] == pytest.approx(1.0)
    assert out["fpr"] == pytest.approx(0.0)
    # 0.7 would buy nothing and break the constraint (fp=1 of 2 negatives)
    assert out["threshold"] != pytest.approx(0.7)


def test_recall_at_fpr_degrades_gracefully_without_both_classes() -> None:
    out = recall_at_fpr([0.9, 0.8], [1, 1])
    assert out["recall"] is None and out["threshold"] is None
    assert out["n_pos"] == 2 and out["n_neg"] == 0


def test_endpoint_recall_appears_in_the_report() -> None:
    frames = [
        _plain_frame("c", [[0.0, 0, 0]], [[0.0, 0, 0]], endpoint_logit=40.0,
                     endpoint_label=1, endpoint_mask=True),
        _plain_frame("c", [[0.0, 0, 0]], [[0.0, 0, 0]], endpoint_logit=-40.0,
                     endpoint_label=0, endpoint_mask=True),
    ]
    endpoint = evaluate_v543(frames)["pooled"]["process"]["endpoint"]
    assert endpoint["recall_at_fpr_0p05"] == pytest.approx(1.0)
    assert endpoint["fpr_at_threshold"] == pytest.approx(0.0)
    assert endpoint["threshold_at_fpr_constraint"] is not None


# ---------------------------------------------------------------------------
# structure: per-curve rows + pooled block
# ---------------------------------------------------------------------------
def test_per_curve_and_pooled_structure() -> None:
    frames = [
        _pair_frame("a", [0.1, 0.05], [0.1, -0.05], q_pred=0.0, q_true=0.0,
                    q_mask=True),
        _pair_frame("b", [0.1], [0.1], q_pred=0.5, q_true=0.5, q_mask=True),
        _pair_frame("b", [0.2], [0.2], q_pred=1.0, q_true=1.0, q_mask=True),
    ]
    report = evaluate_v543(frames, train_q_mean=0.25)
    assert report["schema"] == REPORT_SCHEMA
    assert len(report["per_curve"]) == 2
    assert [row["curve_id"] for row in report["per_curve"]] == ["a", "b"]
    for row in report["per_curve"]:
        for key in ("curve_id", "n_frames", "n_key_rows", "move", "q",
                    "ordering", "endpoint"):
            assert key in row
        for key in ("rmse_zero", "sign_accuracy",
                    "fixed_coverage_direction_at_3",
                    "fixed_coverage_topk_overlap_at_3", "dead_zone_fraction",
                    "pair_delta_d_mae"):
            assert key in row["move"], key
        assert "fixed_coverage_direction_at_3" in row["move"]
    pooled = report["pooled"]
    for key in ("move", "process"):
        assert key in pooled
    for key in ("q", "ordering", "endpoint"):
        assert key in pooled["process"]
    assert pooled["move"]["n_curves"] == 2
    assert pooled["move"]["n_frames"] == 3
    # §9.1 leaf naming is mirrored under fixed_coverage
    assert pooled["move"]["fixed_coverage"]["direction_at_3"] == pytest.approx(
        pooled["move"]["fixed_coverage_direction_at_3"])
    assert pooled["move"]["fixed_coverage"]["k"] == FIXED_COVERAGE_TOPK
    # the per-curve mean of curve "b" is the mean of its two frames
    row_b = [row for row in report["per_curve"] if row["curve_id"] == "b"][0]
    assert row_b["move"]["fixed_coverage_direction_at_3"] == pytest.approx(1.0)
    # pooled = mean over the two curves (frame mean -> curve mean, §9.2 :910)
    assert pooled["move"]["fixed_coverage"]["direction_at_3"] == pytest.approx(
        (0.5 + 1.0) / 2.0)
    assert pooled["move"]["fixed_coverage"][
        "direction_at_3_curve_median"] == pytest.approx(0.75)
    ish = report["validity"]
    assert ish["move"]["n_pairs"] == 4
    assert ish["pair_set"]["n_pairs"] == 4


def test_report_carries_frozen_definitions_and_identity() -> None:
    report = evaluate_v543([_plain_frame("c", [[0.0, 0, 0]], [[0.0, 0, 0]])])
    assert report["evaluator_spec_sha256"] == evaluator_spec_sha256()
    assert report["plan_section"]
    metrics = evaluator_spec()["metrics"]
    for key in ("move.sign_accuracy", "move.fixed_coverage.direction_at_3",
                "process.q.corr_curve_median",
                "process.endpoint.recall_at_fpr_0p05"):
        assert key in metrics, key
    assert report["bond_scale_note"]["applied_here"] is False
    assert report["ambiguities"]
    assert report["frozen_definitions"]["fixed_coverage_direction_at_3"]


def test_evaluator_spec_is_frozen_and_complete() -> None:
    spec = evaluator_spec()
    assert spec["schema"] == REPORT_SCHEMA
    constants = spec["constants"]
    assert constants["fixed_coverage_topk"] == 3
    assert constants["sign_pair_threshold_angstrom"] == pytest.approx(0.005)
    assert constants["fpr_constraint"] == pytest.approx(0.05)
    assert constants["ece_bins"] == 10
    assert constants["q_corr_min_points"] == 3
    assert constants["q_frac_ge3_min"] == pytest.approx(0.70)
    assert constants["bootstrap_draws"] == 10_000
    assert constants["bootstrap_seed_root"] == 20261102
    # the digest is stable across calls (a frozen identity)
    assert evaluator_spec_sha256() == evaluator_spec_sha256()


# ---------------------------------------------------------------------------
# determinism (the SHA-freeze property of §9.5)
# ---------------------------------------------------------------------------
def test_same_input_produces_identical_json() -> None:
    frames = [
        _pair_frame("a", [0.10, -0.05, 0.03], [0.12, -0.01, -0.05],
                    q_pred=0.1, q_true=0.2, q_mask=True,
                    endpoint_logit=0.3, endpoint_label=1, endpoint_mask=True,
                    stratum="post_TS", intent_class="breaking"),
        _pair_frame("b", [0.2, 0.1], [0.2, 0.1], q_pred=0.4, q_true=0.4,
                    q_mask=True, endpoint_logit=-0.2, endpoint_label=0,
                    endpoint_mask=True),
    ]
    first = json.dumps(evaluate_v543(frames, train_q_mean=0.15),
                       indent=2, sort_keys=True)
    second = json.dumps(evaluate_v543(frames, train_q_mean=0.15),
                        indent=2, sort_keys=True)
    assert first == second
    # a different prediction must change the numbers (the test above is not
    # vacuous: the report really is a function of the input)
    changed = [
        _pair_frame("a", [0.10, -0.05, 0.03], [0.0, 0.0, 0.0],
                    q_pred=0.1, q_true=0.2, q_mask=True,
                    endpoint_logit=0.3, endpoint_label=1, endpoint_mask=True,
                    stratum="post_TS", intent_class="breaking"),
        frames[1],
    ]
    assert json.dumps(evaluate_v543(changed, train_q_mean=0.15),
                      sort_keys=True) != first


# ---------------------------------------------------------------------------
# the zero-residual baseline (M0/P0; §4.3 :226 / §9.5 :1037)
# ---------------------------------------------------------------------------
def test_baseline_uses_u0_and_q0_raw_exactly() -> None:
    frame = _pair_frame(
        "c", [0.1, 0.05], [0.1, 0.05],
        q_pred=0.9, q_true=0.5, q_mask=True,
        endpoint_logit=1.5, endpoint_label=1, endpoint_mask=True,
        u0=np.zeros((4, 3)), q0_raw=math.atanh(0.5),
        endpoint0_logit=-0.5)
    frames = zero_residual_frames([frame])
    baseline = frames[0]
    assert np.array_equal(baseline.u, frame.u0)
    assert baseline.q_pred == pytest.approx(0.5)          # tanh(q0_raw)
    assert baseline.endpoint_logit == pytest.approx(-0.5)

    report = evaluate_baseline([frame], train_q_mean=0.5)
    assert report["baseline"]["kind"] == "M0P0_zero_residual"
    move = report["pooled"]["move"]
    # u = u0 = 0 against teacher Δd (0.1, 0.05): both directions missed
    assert move["fixed_coverage"]["direction_at_3"] == pytest.approx(0.0)
    assert move["sign_accuracy_cluster_median"] == pytest.approx(0.0)
    # the frame set and the teacher side are IDENTICAL to the candidate's
    candidate = evaluate_v543([frame], train_q_mean=0.5)
    assert report["validity"]["move"]["n_frames"] == \
        candidate["validity"]["move"]["n_frames"]
    assert report["validity"]["move"]["n_pairs"] == \
        candidate["validity"]["move"]["n_pairs"]
    # M0's q is the encoded q0 (= tanh(q0_raw)) and P0's logit is e0
    assert report["pooled"]["process"]["q"]["mae_curve_median"] == \
        pytest.approx(0.0)


def test_baseline_requires_the_zero_residual_quantities() -> None:
    frame = _pair_frame("c", [0.1], [0.1])
    with pytest.raises(V543EvalError) as error:
        evaluate_baseline([frame])
    assert error.value.code == "V543_EVAL_BASELINE_INPUT"


# ---------------------------------------------------------------------------
# typed refusals
# ---------------------------------------------------------------------------
def test_empty_frame_set_is_refused() -> None:
    with pytest.raises(V543EvalError) as error:
        evaluate_v543([])
    assert error.value.code == "V543_EVAL_EMPTY"


def test_shape_mismatch_is_refused() -> None:
    frame = EvalFrame(curve_id="c", u=np.zeros((2, 3)), y=np.zeros((3, 3)))
    with pytest.raises(V543EvalError) as error:
        evaluate_v543([frame])
    assert error.value.code == "V543_EVAL_FRAME_SHAPE"


def test_non_finite_prediction_is_refused() -> None:
    frame = _plain_frame("c", [[float("nan"), 0, 0]], [[0.0, 0, 0]])
    with pytest.raises(V543EvalError) as error:
        evaluate_v543([frame])
    assert error.value.code == "V543_EVAL_FRAME_NONFINITE"


def test_pair_slot_out_of_range_is_refused() -> None:
    frame = EvalFrame(
        curve_id="c", u=np.zeros((1, 3)), y=np.zeros((1, 3)),
        pair_slots=np.asarray([[0, 1]], dtype=np.int64),
        pair_offsets=np.asarray([[1.0, 0.0, 0.0]]))
    with pytest.raises(V543EvalError) as error:
        evaluate_v543([frame])
    assert error.value.code == "V543_EVAL_PAIR_SLOT"


def test_masked_q_without_values_is_refused() -> None:
    frame = _plain_frame("c", [[0.0, 0, 0]], [[0.0, 0, 0]], q_mask=True)
    with pytest.raises(V543EvalError) as error:
        evaluate_v543([frame])
    assert error.value.code == "V543_EVAL_Q"


def test_masked_in_endpoint_row_with_bad_label_is_refused() -> None:
    frame = _plain_frame("c", [[0.0, 0, 0]], [[0.0, 0, 0]],
                         endpoint_logit=0.0, endpoint_label=-1,
                         endpoint_mask=True)
    with pytest.raises(V543EvalError) as error:
        evaluate_v543([frame])
    assert error.value.code == "V543_EVAL_ENDPOINT"


# ---------------------------------------------------------------------------
# paired curve bootstrap (plan §9.4; the statistic v543_stats will consume)
# ---------------------------------------------------------------------------
def test_paired_curve_bootstrap_is_curve_level_and_paired() -> None:
    candidate = {"a": 1.0, "b": 2.0, "c": 3.0}
    baseline = {"a": 0.0, "b": 0.0, "c": 0.0}
    out = paired_curve_bootstrap(candidate, baseline, higher_is_better=True,
                                 metric="direction", draws=200)
    assert out["n_curves"] == 3 and out["n_seeds"] == 1
    assert out["point_estimate"] == pytest.approx(2.0)
    assert out["unit"].startswith("curve")
    # every draw improves (all three curves improve) → p = 1/(draws+1)
    assert out["p_one_sided"] == pytest.approx(1.0 / 201.0)
    assert out["ci95"][0] > 0.0
    assert out["dropped_curves"]["n_candidate_only"] == 0


def test_paired_curve_bootstrap_drops_missing_clusters_on_both_sides() -> None:
    candidate = {"a": 1.0, "b": 2.0, "only_candidate": 9.0}
    baseline = {"a": 0.0, "b": 0.0, "only_baseline": 9.0}
    out = paired_curve_bootstrap(candidate, baseline, higher_is_better=True,
                                 metric="rmse", draws=100)
    assert out["n_curves"] == 2
    assert out["dropped_curves"]["n_candidate_only"] == 1
    assert out["dropped_curves"]["n_baseline_only"] == 1


def test_paired_curve_bootstrap_seed_averages_per_draw() -> None:
    seed_1 = {"a": 1.0, "b": 1.0}
    seed_2 = {"a": 3.0, "b": 3.0}
    baseline = {"a": 0.0, "b": 0.0}
    out = paired_curve_bootstrap([seed_1, seed_2], [baseline, baseline],
                                 higher_is_better=True, metric="q", draws=100)
    assert out["n_seeds"] == 2
    # seed averages equal 2.0 in every draw → a degenerate, exact interval
    assert out["point_estimate"] == pytest.approx(2.0)
    assert out["ci95"] == pytest.approx([2.0, 2.0])


def test_paired_curve_bootstrap_refuses_unpairable_input() -> None:
    with pytest.raises(V543EvalError) as error:
        paired_curve_bootstrap({"a": 1.0}, {"b": 1.0}, higher_is_better=True)
    assert error.value.code == "V543_EVAL_BOOTSTRAP_EMPTY"


def test_lower_is_better_flips_the_improvement_sign() -> None:
    # rmse: candidate 0.9 vs baseline 1.0 → improvement +0.1 per curve
    out = paired_curve_bootstrap({"a": 0.9, "b": 0.8}, {"a": 1.0, "b": 1.0},
                                 higher_is_better=False, metric="rmse_zero",
                                 draws=100)
    assert out["point_estimate"] == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# the confirmatory multi-seed bootstrap (per-seed resampling, §9.4 :1019/:1023)
# ---------------------------------------------------------------------------
def test_multiseed_bootstrap_draws_are_reproducible_by_hand() -> None:
    """Every draw is rebuilt in the test from the documented seed derivation.

    2 seeds x 3 curves; the test recomputes each per-seed resample with
    ``np.random.default_rng([seed_root, draw, seed_index])`` — if the module
    changed either the derivation or the seed-average order, this fails draw by
    draw (not just on the CI).
    """
    candidate = [{"a": 2.0, "b": 4.0, "c": 6.0},
                 {"a": 1.0, "b": 3.0, "c": 5.0}]
    baseline = [{"a": 1.0, "b": 1.0, "c": 1.0},
                {"a": 0.0, "b": 0.0, "c": 0.0}]
    draws, seed_root = 64, 12345
    out = paired_curve_bootstrap_multiseed(
        candidate, baseline, higher_is_better=True, metric="direction",
        draws=draws, seed_root=seed_root)

    improvements = [
        np.asarray([1.0, 3.0, 5.0]),      # seed 0: 2-1, 4-1, 6-1
        np.asarray([1.0, 3.0, 5.0]),      # seed 1: 1-0, 3-0, 5-0
    ]
    expected = []
    for d in range(draws):
        per_seed = []
        for index, improvement in enumerate(improvements):
            generator = np.random.default_rng([seed_root, d, index])
            pick = generator.integers(0, improvement.size,
                                      size=improvement.size)
            per_seed.append(float(improvement[pick].mean()))
        expected.append(float(np.mean(per_seed)))
    assert out["draw_values"] == pytest.approx(expected, abs=1e-12)
    # CI and p come from exactly those draws
    assert out["ci95"][0] == pytest.approx(float(np.percentile(expected, 2.5)))
    assert out["ci95"][1] == pytest.approx(float(np.percentile(expected, 97.5)))
    p_expected = (1 + sum(1 for v in expected if v <= 0.0)) / (draws + 1)
    assert out["p_one_sided"] == pytest.approx(p_expected)
    assert out["mode"] == "per_seed_independent"
    assert out["seed_derivation"].startswith("np.random.default_rng")


def test_multiseed_bootstrap_resamples_each_seed_independently() -> None:
    """Two seeds with DIFFERENT cluster counts get independent resamples."""
    candidate = [{"a": 1.0, "b": 2.0, "c": 3.0}, {"x": 10.0, "y": 20.0}]
    baseline = [{"a": 0.0, "b": 0.0, "c": 0.0}, {"x": 0.0, "y": 0.0}]
    out = paired_curve_bootstrap_multiseed(
        candidate, baseline, higher_is_better=True, metric="m", draws=32,
        seed_root=7)
    assert out["n_curves_per_seed"] == [3, 2]
    assert out["n_curves_min"] == 2 and out["n_curves_max"] == 3
    # point estimate = mean over seeds of each seed's own mean improvement
    assert out["point_estimate"] == pytest.approx((2.0 + 15.0) / 2.0)
    assert out["per_seed"][0]["point_estimate"] == pytest.approx(2.0)
    assert out["per_seed"][1]["point_estimate"] == pytest.approx(15.0)


def test_multiseed_bootstrap_keeps_power_when_seeds_disagree_on_clusters() -> None:
    """The collapsed-intersection defect: near-disjoint seeds keep their clusters."""
    candidate = [
        {"shared": 2.0, "a1": 1.0, "a2": 1.0, "a3": 1.0},
        {"shared": 2.0, "b1": 1.0, "b2": 1.0, "b3": 1.0},
        {"shared": 2.0, "c1": 1.0, "c2": 1.0, "c3": 1.0},
    ]
    baseline = [
        {"shared": 0.0, "a1": 0.0, "a2": 0.0, "a3": 0.0},
        {"shared": 0.0, "b1": 0.0, "b2": 0.0, "b3": 0.0},
        {"shared": 0.0, "c1": 0.0, "c2": 0.0, "c3": 0.0},
    ]
    multiseed = paired_curve_bootstrap_multiseed(
        candidate, baseline, higher_is_better=True, metric="dir", draws=200)
    shared = paired_curve_bootstrap(candidate, baseline,
                                    higher_is_better=True, metric="dir",
                                    draws=200)
    # the shared-intersection reading collapses to the single common curve
    assert shared["n_curves"] == 1
    assert shared["point_estimate"] == pytest.approx(2.0)
    assert shared["ci95"] == pytest.approx([2.0, 2.0])
    # the confirmatory mode keeps every seed's own clusters: 3 x 4 = 12
    assert multiseed["n_curves_per_seed"] == [4, 4, 4]
    assert multiseed["n_curves_total"] == 12
    # per-seed statistic: the shared curve (+2) plus three +1 curves → 1.25
    assert multiseed["point_estimate"] == pytest.approx(1.25)
    # and it is an INTERVAL, not a degenerate point: 12 clusters ≠ 1 cluster
    lower, upper = multiseed["ci95"]
    assert lower < upper
    assert lower >= 1.0 and upper <= 2.0


def test_multiseed_bootstrap_matches_shared_mode_on_identical_seed_sets() -> None:
    """Degenerate case: identical cluster sets AND constant improvement.

    Then every resample of every seed returns the same statistic, so the two
    modes must produce the SAME CI and p exactly — the audit path and the
    confirmatory path cannot disagree when the data makes them equivalent.
    """
    candidate = {"a": 2.0, "b": 2.0, "c": 2.0}
    baseline = {"a": 1.0, "b": 1.0, "c": 1.0}
    three = [candidate, candidate, candidate]
    three_base = [baseline, baseline, baseline]
    multiseed = paired_curve_bootstrap_multiseed(
        three, three_base, higher_is_better=True, metric="dir", draws=500)
    shared = paired_curve_bootstrap(three, three_base,
                                    higher_is_better=True, metric="dir",
                                    draws=500)
    assert multiseed["point_estimate"] == pytest.approx(1.0)
    assert shared["point_estimate"] == pytest.approx(1.0)
    assert multiseed["ci95"] == pytest.approx(shared["ci95"])
    assert multiseed["p_one_sided"] == pytest.approx(shared["p_one_sided"])
    assert multiseed["p_one_sided"] == pytest.approx(1.0 / 501.0)


def test_multiseed_bootstrap_agrees_with_shared_mode_on_a_fixed_seed_set() -> None:
    """Same cluster set: point estimates agree; the interval is NARROWER.

    With an identical cluster set the two modes must agree on the (unsampled)
    point estimate.  Their intervals cannot be equal when the values vary: the
    confirmatory mode averages three INDEPENDENT resamples per draw (§9.4 :1023)
    while the audit mode averages the (identical) seeds inside one resample, so
    the confirmatory interval is tighter by construction.  That is the intended
    behaviour for the campaign — the three formal seeds are independent samples
    — and it is asserted here as a RELATION, not as equality; the equality case
    is the constant-improvement degenerate test above.
    """
    candidate = {"a": 2.0, "b": 4.0, "c": 3.0, "d": 5.0}
    baseline = {"a": 1.0, "b": 1.0, "c": 2.0, "d": 2.0}
    tables = [candidate] * 3
    bases = [baseline] * 3
    multiseed = paired_curve_bootstrap_multiseed(
        tables, bases, higher_is_better=True, metric="m", draws=4000)
    shared = paired_curve_bootstrap(tables, bases, higher_is_better=True,
                                    metric="m", draws=4000)
    assert multiseed["point_estimate"] == pytest.approx(
        shared["point_estimate"])
    assert multiseed["n_curves_per_seed"] == [4, 4, 4]
    # independent per-seed resampling: the interval nests inside the single-set
    # one (variance reduction of the seed-averaged estimator)
    assert multiseed["ci95"][0] >= shared["ci95"][0] - 1e-12
    assert multiseed["ci95"][1] <= shared["ci95"][1] + 1e-12
    assert (multiseed["ci95"][1] - multiseed["ci95"][0]
            < shared["ci95"][1] - shared["ci95"][0])
    # both are "all draws positive" for a strictly positive improvement
    assert multiseed["p_one_sided"] == pytest.approx(1.0 / 4001.0)
    assert shared["p_one_sided"] == pytest.approx(1.0 / 4001.0)


def test_multiseed_bootstrap_drops_missing_clusters_per_seed() -> None:
    candidate = [{"a": 1.0, "only_c": 5.0}, {"b": 1.0, "c": 1.0}]
    baseline = [{"a": 0.0, "only_b": 5.0}, {"b": 0.0, "c": 0.0}]
    out = paired_curve_bootstrap_multiseed(
        candidate, baseline, higher_is_better=True, metric="m", draws=50)
    assert out["n_curves_per_seed"] == [1, 2]
    dropped = out["dropped_curves"]
    assert dropped["n_candidate_only"] == 1     # only_c in seed 0
    assert dropped["n_baseline_only"] == 1      # only_b in seed 0
    assert dropped["per_seed"][0]["n_candidate_only"] == 1
    assert dropped["per_seed"][1]["n_candidate_only"] == 0


def test_multiseed_bootstrap_refuses_a_seed_without_shared_clusters() -> None:
    with pytest.raises(V543EvalError) as error:
        paired_curve_bootstrap_multiseed(
            [{"a": 1.0}, {"x": 1.0}], [{"a": 0.0}, {"y": 0.0}],
            higher_is_better=True, draws=10)
    assert error.value.code == "V543_EVAL_BOOTSTRAP_EMPTY"


def test_multiseed_bootstrap_refuses_mismatched_seed_counts() -> None:
    with pytest.raises(V543EvalError) as error:
        paired_curve_bootstrap_multiseed(
            [{"a": 1.0}], [{"a": 0.0}, {"a": 0.0}], higher_is_better=True)
    assert error.value.code == "V543_EVAL_BOOTSTRAP_SEEDS"


# ---------------------------------------------------------------------------
# the V542 口径 constants are the frozen ones (no re-invented thresholds)
# ---------------------------------------------------------------------------
def test_frozen_thresholds_match_the_plan() -> None:
    assert SIGN_PAIR_THRESHOLD == pytest.approx(0.005)     # G53 / §9.2 :906
    assert FIXED_COVERAGE_TOPK == 3                        # §9.2 :903
    assert FPR_CONSTRAINT == pytest.approx(0.05)           # 附录 B :1603
    assert Q_CORR_MIN_POINTS == 3                          # §9.4 :1031
    assert Q_FRAC_GE3_MIN == pytest.approx(0.70)           # §9.4 :1032
