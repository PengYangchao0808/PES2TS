"""Unit tests for the V5.2-R D2 phase-1 probe (synthetic only).

Covers the geometry/stereo sign conventions (mirror flips chirality signs,
rotation preserves them, planarity flag is not chiral), the K-angle/dihedral
combination builders against ``evaluate_internal_coordinate``, padding
correctness for variable K, the P3 topology featurizer's coordinate
independence, the paired-delta cluster bootstrap (determinism + CI
bracketing), the P2 self-check / P3 leak / MME gate-rule logic, the goal-free
frame guard, and pre-registration spec-hash stability — no dataset loads, no
``outputs/`` reads, no network, milliseconds-fast.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pes2ts_core.pesn.contracts.key_atom_v3 import IntentRecordV3  # noqa: E402
from pes2ts_core.pesn.features.geometry import (  # noqa: E402
    evaluate_internal_coordinate,
)
from pes2ts_core.pesn.offline.data.geometry_stereo import (  # noqa: E402
    MAX_K_ANGLES,
    MAX_K_DIHEDRALS,
    MAX_K_PAIRS,
    MAX_K_QUADS,
    cis_trans_flag,
    improper_dihedral,
    k_angle_triples,
    k_dihedral_quads,
    k_feature_blocks,
    k_pair_values,
    pad_vector,
    signed_tetrahedral_volume,
    sorted_combinations,
)
from scripts.agent_ops.data_gate.v52r_d2_probe import (  # noqa: E402
    P3_N_FEATURES,
    _FrameRecorder,
    assert_goal_free,
    build_probe_spec,
    compare_p0_to_d1,
    evaluate_phase1_gate,
    feature_layout,
    paired_cluster_bootstrap_auprc,
    p3_topology_features,
    spec_sha256,
    spec_text,
)


def _random_coords(k: int, seed: int = 7) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.uniform(-2.0, 2.0, size=(k, 3))


def _random_rotation(seed: int = 11) -> np.ndarray:
    rng = np.random.RandomState(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _wrap_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class TestSignedTetrahedralVolume:
    def test_mirror_flips_sign(self):
        coords = _random_coords(6)
        quads = k_dihedral_quads(6)
        vols = signed_tetrahedral_volume(coords, quads)
        mirrored = coords * np.array([-1.0, 1.0, 1.0])
        vols_m = signed_tetrahedral_volume(mirrored, quads)
        assert np.allclose(vols_m, -vols, atol=1e-12)
        assert np.any(vols > 1e-6) and np.any(vols < -1e-6)   # actually chiral

    def test_rotation_and_translation_preserve(self):
        coords = _random_coords(6)
        quads = k_dihedral_quads(6)
        vols = signed_tetrahedral_volume(coords, quads)
        rot = _random_rotation() @ coords.T
        vols_r = signed_tetrahedral_volume(rot.T + 5.0, quads)
        assert np.allclose(vols_r, vols, atol=1e-9)

    def test_planar_quad_is_zero(self):
        planar = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        v = signed_tetrahedral_volume(planar, np.array([[0, 1, 2, 3]]))
        assert v[0] == pytest.approx(0.0, abs=1e-12)

    def test_empty_quads(self):
        v = signed_tetrahedral_volume(_random_coords(3), np.zeros((0, 4), int))
        assert v.shape == (0,)


class TestImproperDihedral:
    def test_matches_evaluate_internal_coordinate(self):
        coords = _random_coords(6, seed=3)
        quads = k_dihedral_quads(6)
        mine = improper_dihedral(coords, quads)
        ref = np.array([
            evaluate_internal_coordinate(coords, list(q), "dihedral")
            for q in quads])
        assert np.allclose(mine, ref, atol=1e-12)

    def test_mirror_flips_sign(self):
        coords = _random_coords(6, seed=13)
        quads = k_dihedral_quads(6)
        tau = improper_dihedral(coords, quads)
        mirrored = coords * np.array([1.0, -1.0, 1.0])
        tau_m = improper_dihedral(mirrored, quads)
        assert np.allclose(_wrap_pi(tau_m + tau), 0.0, atol=1e-9)

    def test_rotation_preserves(self):
        coords = _random_coords(6, seed=17)
        quads = k_dihedral_quads(6)
        tau = improper_dihedral(coords, quads)
        rot = (_random_rotation(seed=19) @ coords.T).T
        tau_r = improper_dihedral(rot, quads)
        assert np.allclose(_wrap_pi(tau_r - tau), 0.0, atol=1e-9)


class TestCisTransFlag:
    @staticmethod
    def _quad(dz: float, dy: float) -> np.ndarray:
        # chain a-b-c-d with b–c along +x; (dy, dz) place d beyond c
        return np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0],
                         [1.0, 0.0, 0.0], [2.0, dy, dz]])

    def test_cis_trans_perpendicular(self):
        quads = np.array([[0, 1, 2, 3]])
        # d beyond c in the b–c plane, same side as a → τ=0 → cis (+1)
        assert cis_trans_flag(self._quad(0.0, 1.0), quads)[0] == 1.0
        # d mirrored to the opposite side → τ=π → trans (−1)
        assert cis_trans_flag(self._quad(0.0, -1.0), quads)[0] == -1.0
        # d exactly out-of-plane (τ=π/2) → perpendicular → 0
        assert cis_trans_flag(self._quad(1.0, 0.0), quads)[0] == 0.0

    def test_mirror_and_rotation_invariant(self):
        coords = _random_coords(6, seed=23)
        quads = k_dihedral_quads(6)
        flags = cis_trans_flag(coords, quads)
        assert set(np.unique(flags)) <= {-1.0, 0.0, 1.0}
        mirrored = coords * np.array([-1.0, 1.0, 1.0])
        assert np.array_equal(cis_trans_flag(mirrored, quads), flags)
        rot = (_random_rotation(seed=29) @ coords.T).T
        assert np.array_equal(cis_trans_flag(rot, quads), flags)


class TestKCombinationBuilders:
    def test_counts_per_k(self):
        assert sorted_combinations(2, 2).shape == (1, 2)
        assert sorted_combinations(8, 2).shape == (MAX_K_PAIRS, 2)
        assert sorted_combinations(8, 3).shape == (MAX_K_ANGLES, 3)
        assert sorted_combinations(8, 4).shape == (MAX_K_QUADS, 4)
        assert sorted_combinations(4, 3).shape == (4, 3)   # C(4,3)=4
        assert k_angle_triples(2).shape == (0, 3)
        assert k_dihedral_quads(3).shape == (0, 4)

    def test_lexicographic_order(self):
        triples = k_angle_triples(5)
        assert tuple(triples[0]) == (0, 1, 2)
        assert tuple(triples[-1]) == (2, 3, 4)
        assert tuple(k_dihedral_quads(5)[0]) == (0, 1, 2, 3)

    def test_angles_match_evaluate_internal_coordinate(self):
        coords = _random_coords(6, seed=5)
        triples = k_angle_triples(6)
        from pes2ts_core.pesn.offline.data.geometry_stereo import k_angle_values
        mine = k_angle_values(coords)
        ref = np.array([
            evaluate_internal_coordinate(coords, list(t), "angle")
            for t in triples])
        assert np.allclose(mine, ref, atol=1e-12)

    def test_dihedrals_match_evaluate_internal_coordinate(self):
        coords = _random_coords(8, seed=9)
        from pes2ts_core.pesn.offline.data.geometry_stereo import k_dihedral_values
        mine = k_dihedral_values(coords)
        ref = np.array([
            evaluate_internal_coordinate(coords, list(q), "dihedral")
            for q in k_dihedral_quads(8)])
        assert np.allclose(mine, ref, atol=1e-12)

    def test_pairs_match_d1_formula(self):
        coords = _random_coords(8, seed=2)
        d = k_pair_values(coords)
        iu = np.triu_indices(8, k=1)
        diff = coords[iu[0]] - coords[iu[1]]
        assert np.array_equal(d, np.sqrt((diff * diff).sum(axis=1)))


class TestPadding:
    def test_pad_vector_zero_fills(self):
        out = pad_vector([1.0, 2.0], 4)
        assert out == pytest.approx([1.0, 2.0, 0.0, 0.0])

    def test_pad_vector_refuses_oversize(self):
        with pytest.raises(ValueError):
            pad_vector([1.0, 2.0, 3.0], 2)

    def test_blocks_variable_k(self):
        rng = np.random.RandomState(0)
        frame = rng.uniform(-1, 1, size=(8, 3))
        for k in (2, 4, 8):
            keys = list(range(k))
            blocks = k_feature_blocks(frame, keys)
            assert blocks["pairs"].shape == (MAX_K_PAIRS,)
            assert blocks["angles"].shape == (MAX_K_ANGLES,)
            assert blocks["dihedrals"].shape == (MAX_K_DIHEDRALS,)
            assert blocks["volumes"].shape == (MAX_K_QUADS,)
            n_pairs = k * (k - 1) // 2
            assert np.count_nonzero(blocks["pairs"][:n_pairs]) >= 1
            assert np.all(blocks["pairs"][n_pairs:] == 0.0)
            if k >= 3:
                n_ang = k * (k - 1) * (k - 2) // 6
                assert np.count_nonzero(~np.isnan(blocks["angles"][:n_ang])) == n_ang
                assert np.all(blocks["angles"][n_ang:] == 0.0)
            else:
                assert np.all(blocks["angles"] == 0.0)
            if k >= 4:
                n_q = k * (k - 1) * (k - 2) * (k - 3) // 24
                assert np.all(blocks["volumes"][n_q:] == 0.0)
            else:
                assert np.all(blocks["volumes"] == 0.0)

    def test_oversize_k_refused(self):
        frame = np.zeros((9, 3))
        with pytest.raises(ValueError):
            k_feature_blocks(frame, list(range(9)))


def _synthetic_intent(direction: str = "forward") -> IntentRecordV3:
    row = {
        "intent_id": "INT_TEST_1",
        "curve_id": "RXN_TEST_1",
        "direction": direction,
        "reverse_of_intent_id": "INT_TEST_2",
        "intent_pair_id": "PAIR_1",
        "g_src": json.dumps({"atoms": [6, 6, 8, 1],
                             "bonds": [[0, 1, 1.0], [1, 2, 2.0], [2, 3, 1.5]]}),
        "g_tgt": json.dumps({"atoms": [6, 6, 8, 1],
                             "bonds": [[0, 1, 1.0], [1, 2, 1.0], [0, 2, 1.0]]}),
        "cv_spec": json.dumps({
            "kind": "graph_edit_bond_lengths",
            "coordinates": [[0, 1], [1, 2], [0, 2]]}),
    }
    return IntentRecordV3.from_table_row(row)


class TestP3TopologyFeatures:
    def test_content(self):
        intent = _synthetic_intent()
        f = p3_topology_features(intent)
        assert f.shape == (P3_N_FEATURES,)
        assert f[0] == 4.0                       # n_atoms
        assert f[1] == 3.0 and f[2] == 3.0       # n_bonds src/tgt
        assert f[3:8] == pytest.approx([1.0, 1.0, 1.0, 0.0, 0.0])   # src hist
        assert f[8:13] == pytest.approx([3.0, 0.0, 0.0, 0.0, 0.0])  # tgt hist
        assert f[13] == 3.0                      # changed bonds (1,2) (2,3) (0,2)
        assert f[14] == 3.0                      # cv pairs
        # |ΔB| over union {(0,1):0, (1,2):1, (2,3):1.5, (0,2):1}
        assert f[15] == pytest.approx(3.5 / 4)   # mean
        assert f[16] == pytest.approx(1.5)       # max
        assert f[17] == pytest.approx(3.5)       # sum
        assert f[18] == 3.0                      # n_nonzero
        assert f[19] == 1.0 and f[20] == 0.0     # direction one-hot
        assert f[21] == 4.0                      # |K|

    def test_no_coordinate_dependence(self):
        intent = _synthetic_intent()
        rng = np.random.RandomState(0)
        coords_a = rng.uniform(-1, 1, size=(4, 3))
        coords_b = rng.permutation(coords_a.reshape(-1)).reshape(4, 3)
        base = p3_topology_features(intent)
        assert np.array_equal(
            p3_topology_features(intent, _frame_xyz=coords_a), base)
        assert np.array_equal(
            p3_topology_features(intent, _frame_xyz=coords_b), base)

    def test_direction_flips_onehot_only(self):
        fwd = p3_topology_features(_synthetic_intent("forward"))
        rev = p3_topology_features(_synthetic_intent("reverse"))
        assert fwd[19:21] == pytest.approx([1.0, 0.0])
        assert rev[19:21] == pytest.approx([0.0, 1.0])
        keep = np.arange(P3_N_FEATURES) < 19
        assert np.array_equal(fwd[keep], rev[keep])


class TestGoalFreeGuard:
    class _FakeDataset:
        def frame_xyz(self, curve_id, node_index):
            return ((0.0, 0.0, 0.0),)

    def test_allows_current_only(self):
        rec = _FrameRecorder(self._FakeDataset())
        rec.frame_xyz("C1", 3)
        assert_goal_free(rec, {("C1", 3)}, "P0")

    def test_rejects_goal_frame(self):
        rec = _FrameRecorder(self._FakeDataset())
        rec.frame_xyz("C1", 3)
        rec.frame_xyz("C1", 9)          # e.g. a target-endpoint fetch
        with pytest.raises(AssertionError):
            assert_goal_free(rec, {("C1", 3)}, "P4")


class TestPairedBootstrap:
    def test_deterministic_and_ci_brackets_point(self):
        rng = np.random.RandomState(1)
        n = 400
        curves = np.array([f"C{i % 20}" for i in range(n)], dtype=object)
        labels = rng.randint(0, 2, size=n)
        base = labels * 0.15 + rng.uniform(0, 0.7, size=n)   # overlapping
        stronger = base + 0.2 * labels   # ranking shift → strictly better AP
        out1 = paired_cluster_bootstrap_auprc(
            stronger, base, labels, curves, n_resamples=200, seed=0)
        out2 = paired_cluster_bootstrap_auprc(
            stronger, base, labels, curves, n_resamples=200, seed=0)
        assert out1 == out2                                  # deterministic
        assert out1["point"] is not None and out1["point"] > 0.0
        assert out1["ci95"]["lower"] is not None
        assert (out1["ci95"]["lower"] <= out1["point"]
                <= out1["ci95"]["upper"])                    # brackets point

    def test_identical_scores_give_zero_delta(self):
        rng = np.random.RandomState(2)
        n = 100
        curves = np.array([f"c{i % 10}" for i in range(n)], dtype=object)
        labels = rng.randint(0, 2, size=n)
        scores = rng.uniform(0, 1, size=n)
        out = paired_cluster_bootstrap_auprc(
            scores, scores, labels, curves, n_resamples=50, seed=0)
        assert out["point"] == pytest.approx(0.0, abs=1e-12)
        assert out["ci95"]["lower"] == pytest.approx(0.0, abs=1e-12)
        assert out["ci95"]["upper"] == pytest.approx(0.0, abs=1e-12)


class TestGateRules:
    _P0_CHECK = {"status": "SKIPPED", "max_abs_diff": None}

    def test_p2_rule_boundary(self):
        ok = evaluate_phase1_gate(
            p0_check=self._P0_CHECK, p2_median=0.991, p3_median=0.30,
            p4_median=0.60, mme_p1={"median_delta": None, "ci_lower": None},
            mme_p4={"median_delta": None, "ci_lower": None})
        by_id = {c["id"]: c for c in ok["checks"]}
        assert by_id["p2_self_check"]["status"] == "PASS"
        bad = evaluate_phase1_gate(
            p0_check=self._P0_CHECK, p2_median=0.985, p3_median=0.30,
            p4_median=0.60, mme_p1={"median_delta": None, "ci_lower": None},
            mme_p4={"median_delta": None, "ci_lower": None})
        assert {c["id"] for c in bad["checks"] if c["status"] == "FAIL"} == \
            {"p2_self_check"}
        assert bad["verdict"] == "FAIL"

    def test_p3_leak_rule(self):
        leak = evaluate_phase1_gate(
            p0_check=self._P0_CHECK, p2_median=1.0, p3_median=0.60,
            p4_median=0.62, mme_p1={"median_delta": None, "ci_lower": None},
            mme_p4={"median_delta": None, "ci_lower": None})
        by_id = {c["id"]: c for c in leak["checks"]}
        assert by_id["p3_leak_rule"]["status"] == "LEAK"      # 0.60 ≥ 0.62−0.05
        assert leak["verdict"] == "FAIL"
        clean = evaluate_phase1_gate(
            p0_check=self._P0_CHECK, p2_median=1.0, p3_median=0.55,
            p4_median=0.62, mme_p1={"median_delta": None, "ci_lower": None},
            mme_p4={"median_delta": None, "ci_lower": None})
        by_id = {c["id"]: c for c in clean["checks"]}
        assert by_id["p3_leak_rule"]["status"] == "NO_LEAK"

    def test_mme_pass_and_pending(self):
        passed = evaluate_phase1_gate(
            p0_check=self._P0_CHECK, p2_median=1.0, p3_median=0.30,
            p4_median=0.60,
            mme_p1={"median_delta": 0.12, "ci_lower": 0.02, "point": 0.11},
            mme_p4={"median_delta": 0.06, "ci_lower": 0.01, "point": 0.05})
        by_id = {c["id"]: c for c in passed["checks"]}
        assert by_id["mme_p1_minus_p0"]["status"] == "PASS"
        assert by_id["mme_p4_minus_p0"]["status"] == "PASS"
        assert by_id["mme_p5_minus_p0"]["status"] == "PENDING"
        assert passed["verdict"] == "PASS"
        assert passed["training_authorized"] is False
        # CI lower bound ≤ 0 → PENDING-GATE even with a large median delta
        pending = evaluate_phase1_gate(
            p0_check=self._P0_CHECK, p2_median=1.0, p3_median=0.30,
            p4_median=0.60,
            mme_p1={"median_delta": 0.15, "ci_lower": -0.01, "point": 0.14},
            mme_p4={"median_delta": 0.01, "ci_lower": 0.0, "point": 0.01})
        by_id = {c["id"]: c for c in pending["checks"]}
        assert by_id["mme_p1_minus_p0"]["status"] == "PENDING-GATE"
        assert by_id["mme_p4_minus_p0"]["status"] == "PENDING-GATE"
        assert pending["verdict"] == "PASS"      # pending ≠ integrity failure


class TestSpecHashStability:
    _HASHES = {
        "code_commit": "abc123",
        "data_manifest_sha256": "d0",
        "dataset_sha256": "d1",
        "campaign_config_sha256": "c1",
        "label_builder_sha256": "l1",
        "geometry_stereo_sha256": "g1",
        "d1_probe_script_sha256": "s1",
        "d2_probe_script_sha256": "s2",
        "embeddings_sha256": {"embeddings_0_17.npz": "e1"},
    }

    def test_stable_across_builds(self):
        a = build_probe_spec(hashes=dict(self._HASHES), n_bootstrap=1000,
                             d1_pack_path="x.json")
        b = build_probe_spec(hashes=dict(self._HASHES), n_bootstrap=1000,
                             d1_pack_path="x.json")
        assert spec_sha256(a) == spec_sha256(b)
        assert spec_text(a) == spec_text(b)      # canonical file bytes

    def test_hash_changes_with_inputs(self):
        a = build_probe_spec(hashes=dict(self._HASHES), n_bootstrap=1000,
                             d1_pack_path="x.json")
        other = dict(self._HASHES)
        other["label_builder_sha256"] = "CHANGED"
        b = build_probe_spec(hashes=other, n_bootstrap=1000,
                             d1_pack_path="x.json")
        assert spec_sha256(a) != spec_sha256(b)
        c = build_probe_spec(hashes=dict(self._HASHES), n_bootstrap=500,
                             d1_pack_path="x.json")
        assert spec_sha256(a) != spec_sha256(c)

    def test_spec_contains_required_gates(self):
        spec = build_probe_spec(hashes=dict(self._HASHES), n_bootstrap=1000,
                                d1_pack_path="x.json")
        assert spec["seeds"]["encoder_seeds"] == [17, 29, 43]
        assert "P1-P0" in spec["mme_gates"]
        assert "P4-P0" in spec["mme_gates"]
        assert spec["mme_gates"]["P5-P0"]["status"].startswith("PENDING")
        assert spec["mme_gates"]["P2_self_check"]
        assert spec["mme_gates"]["P3_leak_rule"]
        assert "COUNTED ONLY" in spec["sealed_firewall"]
        assert spec["bootstrap"]["n_resamples"] == 1000
        layout = feature_layout()
        for probe, entry in layout.items():
            assert spec["probes"][probe]["n_features"] == entry["n_features"]
            last = max(c["stop"] for c in entry["columns"])
            assert last == entry["n_features"]


class TestCompareP0ToD1:
    @staticmethod
    def _units_and_d1():
        units = [{"fold": 0, "seed": 17, "probes": {
            "P0/hgb": {"auprc": 0.382801, "auroc": 0.722123},
            "P0/logistic": {"auprc": 0.257553, "auroc": 0.493585},
            "P0/constant": {"auprc": 0.198502, "auroc": 0.5}}}]
        d1_units = [{"fold": 0, "seed": 17, "probes": {
            "hgb_tab": {"auprc": 0.382801, "auroc": 0.722123},
            "logistic_tab": {"auprc": 0.257553, "auroc": 0.493585},
            "constant": {"auprc": 0.198502, "auroc": 0.5}}}]

        def _pack(u):
            return {"endpoint_probes": {"units": u}}
        return units, d1_units, _pack(d1_units)

    def test_skipped_without_inputs(self):
        out = compare_p0_to_d1([], None, None)
        assert out["status"] == "SKIPPED"
        assert out["in_process"]["status"] == "SKIPPED"
        assert out["stored_pack"]["status"] == "SKIPPED"

    def test_pass_paths(self):
        units, d1_units, d1_pack = self._units_and_d1()
        out = compare_p0_to_d1(units, d1_units, d1_pack)
        assert out["status"] == "PASS"
        assert out["in_process"]["status"] == "PASS"
        assert out["stored_pack"]["status"] == "PASS"
        assert out["in_process"]["n_compared"] >= 6

    def test_inprocess_fail_overrides(self):
        units, d1_units, d1_pack = self._units_and_d1()
        units[0]["probes"]["P0/hgb"]["auprc"] = 0.5
        out = compare_p0_to_d1(units, d1_units, d1_pack)
        assert out["in_process"]["status"] == "FAIL"
        assert out["in_process"]["n_mismatches"] == 1
        assert out["status"] == "FAIL"

    def test_stored_pack_drift_reported_but_not_gating(self):
        units, d1_units, _ = self._units_and_d1()
        drifted = [{"fold": 0, "seed": 17, "probes": {
            "hgb_tab": {"auprc": 0.35, "auroc": 0.62}}}]
        out = compare_p0_to_d1(
            units, d1_units, {"endpoint_probes": {"units": drifted}})
        assert out["status"] == "PASS"                 # in-process is hard
        assert out["stored_pack"]["status"] == "FAIL"  # drift is audited
        assert "drift" in out["stored_pack"]["note"]
