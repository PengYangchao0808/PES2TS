"""Small unit tests for the V5.2-R D1 probe helpers (pure functions only).

Covers the pre-registered threshold rule, directed pairwise ordering math,
tabular padding, cluster bootstrap determinism, Spearman ties, and the gate
verdict logic — no dataset/model loads, no network, milliseconds-fast.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.agent_ops.data_gate.v52r_d1_probe import (  # noqa: E402
    curve_cluster_bootstrap_auprc,
    evaluate_gate,
    k_pair_distances,
    pad_tabular_features,
    pairwise_ordering_accuracy,
    pick_precision_threshold,
    spearman_corr,
)


class TestPickPrecisionThreshold:
    def test_picks_smallest_qualifying_threshold(self):
        # scores: 0.9(pos), 0.8(neg), 0.7(pos), 0.2(neg)
        scores = np.array([0.9, 0.8, 0.7, 0.2])
        labels = np.array([1, 0, 1, 0])
        # thr=0.7 → preds {0.9,0.8,0.7}: tp=2 fp=1 prec=2/3 <0.8
        # thr=0.9 → preds {0.9}: tp=1 fp=0 prec=1.0 ≥0.8 ✓
        thr = pick_precision_threshold(scores, labels, 0.8)
        assert thr == pytest.approx(0.9)

    def test_none_when_target_unreachable(self):
        # top score is a negative; best achievable precision is 0.5
        scores = np.array([0.9, 0.8, 0.7, 0.6])
        labels = np.array([0, 1, 0, 1])
        assert pick_precision_threshold(scores, labels, 0.8) is None

    def test_none_without_positives(self):
        assert pick_precision_threshold(
            np.array([0.1, 0.2]), np.array([0, 0]), 0.8) is None

    def test_full_recall_when_all_positive_tops(self):
        scores = np.array([0.95, 0.9, 0.1, 0.05])
        labels = np.array([1, 1, 0, 0])
        # thr=0.9 → tp=2 fp=0 prec=1.0; smallest such thr is 0.9 (0.95 also
        # qualifies but is larger)
        assert pick_precision_threshold(scores, labels, 0.8) == pytest.approx(0.9)


class TestPairwiseOrderingAccuracy:
    def test_perfect_prediction(self):
        q_true = np.array([-1.0, 0.0, 1.0])
        q_pred = np.array([-0.9, 0.1, 0.8])
        res = pairwise_ordering_accuracy(q_true, q_pred, np.array([0, 5, 9]), 5)
        assert res["n_pairs"] == 3
        assert res["accuracy"] == pytest.approx(1.0)
        # cross-TS pairs: (0,9) and (0,5)? (0,5): (0-5)*(5-5)=0 → not straddle
        assert res["n_pairs_cross_ts"] == 1
        assert res["cross_ts_accuracy"] == pytest.approx(1.0)

    def test_inverted_prediction_is_zero(self):
        q_true = np.array([-1.0, 0.0, 1.0])
        q_pred = np.array([1.0, 0.0, -1.0])
        res = pairwise_ordering_accuracy(q_true, q_pred, np.array([0, 5, 9]), 5)
        assert res["accuracy"] == pytest.approx(0.0)

    def test_ties_in_q_true_excluded(self):
        q_true = np.array([0.5, 0.5, -1.0])
        q_pred = np.array([0.5, 0.6, 0.0])
        res = pairwise_ordering_accuracy(q_true, q_pred, np.array([1, 2, 3]), 2)
        # the (0.5, 0.5) pair has no strict q difference → excluded; the two
        # mixed pairs must be correct
        assert res["n_pairs"] == 2
        assert res["n_correct"] == 2

    def test_empty_group(self):
        res = pairwise_ordering_accuracy(
            np.array([1.0]), np.array([0.2]), np.array([3]), 1)
        assert res["accuracy"] is None
        assert res["n_pairs"] == 0


class TestTabularFeatures:
    def test_k_pair_distances_triangle(self):
        pts = np.array([[0.0, 0, 0], [1.0, 0, 0], [0, 2.0, 0]])
        d = k_pair_distances(pts, [0, 1, 2])
        assert d == pytest.approx([1.0, 2.0, np.sqrt(5.0)])

    def test_padding_and_onehot(self):
        mat = pad_tabular_features(
            [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]],
            ["forward", "reverse", "forward"], max_pairs=3)
        assert mat.shape == (3, 5)
        assert mat[0] == pytest.approx([1.0, 2.0, 0.0, 1.0, 0.0])
        assert mat[1] == pytest.approx([3.0, 0.0, 0.0, 0.0, 1.0])
        assert mat[2] == pytest.approx([4.0, 5.0, 6.0, 1.0, 0.0])

    def test_oversize_row_refused(self):
        with pytest.raises(ValueError):
            pad_tabular_features([[1.0, 2.0, 3.0]], ["forward"], max_pairs=2)

    def test_unknown_direction_refused(self):
        with pytest.raises(ValueError):
            pad_tabular_features([[1.0]], ["sideways"], max_pairs=1)


class TestBootstrap:
    def test_deterministic_and_brackets_point(self):
        rng = np.random.RandomState(42)
        n = 400
        curve_ids = np.array([f"C{i % 20}" for i in range(n)], dtype=object)
        labels = rng.randint(0, 2, size=n)
        scores = labels * 0.6 + rng.uniform(0, 0.4, size=n)
        out1 = curve_cluster_bootstrap_auprc(
            scores, labels, curve_ids, n_resamples=200, seed=0)
        out2 = curve_cluster_bootstrap_auprc(
            scores, labels, curve_ids, n_resamples=200, seed=0)
        assert out1 == out2                              # deterministic
        assert out1["ci95"]["lower"] is not None
        assert out1["ci95"]["lower"] <= out1["point"] <= out1["ci95"]["upper"]
        assert out1["n_curves"] == 20

    def test_prevalence_reported(self):
        labels = np.array([1, 0, 1, 0, 0, 0, 1, 0], dtype=np.int64)
        scores = np.linspace(0, 1, 8)
        curves = np.array([f"c{i}" for i in range(8)], dtype=object)
        out = curve_cluster_bootstrap_auprc(
            scores, labels, curves, n_resamples=20, seed=0)
        assert out["prevalence"] == pytest.approx(3 / 8)


class TestSpearman:
    def test_monotone_is_one(self):
        assert spearman_corr([1, 2, 3, 4], [10, 20, 30, 40]) == \
            pytest.approx(1.0)

    def test_ties_average_ranks(self):
        rho = spearman_corr([1, 1, 2, 3], [1, 2, 3, 4])
        assert -1.0 <= rho <= 1.0

    def test_constant_is_none(self):
        assert spearman_corr([1, 1, 1], [1, 2, 3]) is None


class TestGate:
    def test_pass_verdict(self):
        bootstrap = {"ci95": {"lower": 0.25}, "prevalence": 0.15}
        false_stop = {"start_pooled_rate": 0.0, "ts_pooled_rate": 0.01}
        ordering = {"accuracy_mean": 0.72, "cross_ts_accuracy_mean": 0.65}
        gate = evaluate_gate(bootstrap, false_stop, ordering)
        assert gate["verdict"] == "PASS"
        assert all(c["pass"] for c in gate["checks"])

    def test_fail_on_ci_below_prevalence(self):
        bootstrap = {"ci95": {"lower": 0.10}, "prevalence": 0.15}
        false_stop = {"start_pooled_rate": 0.0, "ts_pooled_rate": 0.01}
        ordering = {"accuracy_mean": 0.72, "cross_ts_accuracy_mean": 0.65}
        gate = evaluate_gate(bootstrap, false_stop, ordering)
        assert gate["verdict"] == "FAIL"
        failed = [c["id"] for c in gate["checks"] if not c["pass"]]
        assert failed == ["auprc_ci_lower_above_prevalence"]

    def test_fail_on_ordering_and_null_ci(self):
        bootstrap = {"ci95": {"lower": None}, "prevalence": 0.15}
        false_stop = {"start_pooled_rate": 0.05, "ts_pooled_rate": 0.06}
        ordering = {"accuracy_mean": 0.60, "cross_ts_accuracy_mean": 0.55}
        gate = evaluate_gate(bootstrap, false_stop, ordering)
        assert gate["verdict"] == "FAIL"
        failed = {c["id"] for c in gate["checks"] if not c["pass"]}
        assert "auprc_ci_lower_above_prevalence" in failed
        assert "start_false_stop_le_2pct" in failed
        assert "ts_false_stop_le_5pct" in failed
        assert "ordering_mean_above_065" in failed
        assert "ordering_cross_ts_above_060" in failed
