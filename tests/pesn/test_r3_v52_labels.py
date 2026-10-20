"""V5.2 E4 label math tests: q arcs, stop window edges, mask semantics.

Frozen campaign ``v52_training.qlabel`` / ``stop_labels`` + 30h plan
§5.2–§5.3.  Covers: mass-weighted arc-length q on synthetic arcs
(A=−1/TS=0/B=+1), zero-arc masking (never divide by zero), reverse
``q = −q`` exactness, Kabsch-aligned ``d_key_TS`` and the three-way stop
window including BOTH inclusive edges (0.04 / 0.12 Å) with the middle
region masked — plus fixture-table end-to-end label derivation through
``KeyAtomDatasetV3.v52_labels``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (
    ATOMIC_MASS_TABLE_V52,
    STOP_NEGATIVE_WINDOW_ANGSTROM_V52,
    STOP_POSITIVE_WINDOW_ANGSTROM_V52,
    ZERO_ARC_EPS_ANGSTROM_V52,
    KeyAtomDatasetV3,
    atomic_mass_v52,
    kabsch_rmsd_v52,
    mass_weighted_arc_lengths_v52,
    progress_q_from_arcs_v52,
    stop_label_from_distance_v52,
)


# ---------------------------------------------------------------------------
# q math on synthetic arcs
# ---------------------------------------------------------------------------
def test_arc_lengths_linear_path_mass_weighted() -> None:
    # 4 atoms: 1 heavy (Z=16, index 0), 3 light (Z=1); each step moves ONLY
    # the heavy atom by 1 Å along x → ds = sqrt(m_S·1²/ (m_S + 3·m_H))
    frames = np.tile(np.array([[0.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0],
                               [1.0, 0, 0]]), (3, 1, 1))
    frames[1, 0, 0] = 1.0
    frames[2, 0, 0] = 2.0
    s = mass_weighted_arc_lengths_v52(frames, (16, 1, 1, 1))
    m_h, m_s = atomic_mass_v52(1), atomic_mass_v52(16)
    ds_expected = math.sqrt(m_s / (m_s + 3 * m_h))
    assert s.shape == (3,)
    assert s[0] == 0.0
    np.testing.assert_allclose(s[1], ds_expected, rtol=1e-12)
    np.testing.assert_allclose(s[2], 2 * ds_expected, rtol=1e-12)


def test_q_anchors_and_midpoints_on_synthetic_arcs() -> None:
    # symmetric 5-frame path: s = [0, 1, 2, 3, 4], TS at position 2
    s = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    q, reason = progress_q_from_arcs_v52(s, 2)
    assert reason == ""
    np.testing.assert_allclose(q, [-1.0, -0.5, 0.0, 0.5, 1.0], atol=1e-12)


def test_q_asymmetric_segment_lengths() -> None:
    # TS splits the path 1:3 — left/right segments normalize INDEPENDENTLY
    s = np.array([0.0, 1.0, 4.0])
    q, reason = progress_q_from_arcs_v52(s, 1)
    assert reason == ""
    np.testing.assert_allclose(q, [-1.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(q[0], -(s[1] - s[0]) / (s[1] - s[0]))
    # the last frame: (s_B - s_TS)/(s_B - s_TS) = +1 by construction
    s2 = np.array([0.0, 0.5, 1.0, 3.0])
    q2, _ = progress_q_from_arcs_v52(s2, 1)
    np.testing.assert_allclose(
        q2, [-1.0, 0.0, 0.5 / 2.5, 1.0], atol=1e-12)


def test_zero_arc_left_and_right_masked_not_divided() -> None:
    # A..TS arc ~0 → masked (zero_arc_a_ts), never inf/nan
    q, reason = progress_q_from_arcs_v52(np.array([0.0, 0.0, 1.0]), 1)
    assert q is None and reason == "zero_arc_a_ts"
    # TS..B arc ~0 → masked (zero_arc_ts_b)
    q, reason = progress_q_from_arcs_v52(np.array([0.0, 1.0, 1.0]), 1)
    assert q is None and reason == "zero_arc_ts_b"
    # exactly-zero eps boundary: arc == eps still masked (<= eps)
    q, reason = progress_q_from_arcs_v52(
        np.array([0.0, ZERO_ARC_EPS_ANGSTROM_V52, 1.0]), 1)
    assert q is None and reason == "zero_arc_a_ts"
    # just above eps is valid
    q, reason = progress_q_from_arcs_v52(
        np.array([0.0, 10 * ZERO_ARC_EPS_ANGSTROM_V52, 1.0]), 1)
    assert q is not None and reason == ""


def test_q_ts_index_out_of_range_refused() -> None:
    with pytest.raises(Exception):
        progress_q_from_arcs_v52(np.array([0.0, 1.0]), 5)


# ---------------------------------------------------------------------------
# d_key_TS (Kabsch) + stop window edges
# ---------------------------------------------------------------------------
def test_kabsch_rmsd_translation_rotation_invariance() -> None:
    rng = np.random.default_rng(7)
    mobile = rng.normal(size=(5, 3))
    # proper rotation (rot about z by θ) + translation
    theta = 0.7
    rot = np.array([[math.cos(theta), -math.sin(theta), 0],
                    [math.sin(theta), math.cos(theta), 0],
                    [0, 0, 1.0]])
    target = mobile @ rot.T + np.array([3.0, -2.0, 1.5])
    assert kabsch_rmsd_v52(mobile, target) < 1e-10


def test_kabsch_rmsd_reflects_no_reflection() -> None:
    # chirally asymmetric geometry: its mirror image is NOT related by any
    # proper rotation, so legal rigid alignment (det=+1 only) keeps a large
    # residual — reflections must never be aligned away
    mobile = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.2, 2.0, 0], [0.1, 0.3, 3.0]])
    target = mobile * np.array([1.0, 1.0, -1.0])
    assert kabsch_rmsd_v52(mobile, target) > 0.5


def test_kabsch_rmsd_single_atom_translation_only() -> None:
    # one atom: any pair is related by pure translation ⇒ aligned RMSD = 0
    assert kabsch_rmsd_v52(np.array([[1.0, 0, 0]]),
                            np.array([[2.0, 0, 0]])) == pytest.approx(0.0)
    assert kabsch_rmsd_v52(np.array([[1.0, 0, 0]]),
                            np.array([[1.0, 0, 0]])) == 0.0


def test_stop_window_edges_inclusive_both_sides() -> None:
    # frozen windows (yaml): positive d<=0.04, negative d>=0.12, middle Mask
    assert STOP_POSITIVE_WINDOW_ANGSTROM_V52 == 0.04
    assert STOP_NEGATIVE_WINDOW_ANGSTROM_V52 == 0.12
    assert stop_label_from_distance_v52(0.0) == 1
    assert stop_label_from_distance_v52(0.04) == 1        # inclusive edge
    assert stop_label_from_distance_v52(0.04 - 1e-9) == 1
    assert stop_label_from_distance_v52(0.04 + 1e-9) == -1  # middle → Mask
    assert stop_label_from_distance_v52(0.08) == -1
    assert stop_label_from_distance_v52(0.12 - 1e-9) == -1
    assert stop_label_from_distance_v52(0.12) == 0        # inclusive edge
    assert stop_label_from_distance_v52(1.5) == 0


def test_stop_window_widening_refused() -> None:
    with pytest.raises(Exception):
        stop_label_from_distance_v52(0.1, positive_window=0.13,
                                     negative_window=0.12)


def test_atomic_mass_table_frozen_shape() -> None:
    assert len(ATOMIC_MASS_TABLE_V52) == 118     # Z = 1..118
    assert atomic_mass_v52(6) == pytest.approx(12.011)
    with pytest.raises(Exception):
        atomic_mass_v52(0)
    with pytest.raises(Exception):
        atomic_mass_v52(119)


# ---------------------------------------------------------------------------
# end-to-end on the deterministic six-table fixture
# ---------------------------------------------------------------------------
@pytest.fixture()
def fixture_dataset(tmp_path):
    from scripts.agent_ops.feature_gate.v3_fixtures import write_fixture_tables
    write_fixture_tables(tmp_path, n_frames=9, n_curves=2, fold=1)
    return KeyAtomDatasetV3(tmp_path)


def test_fixture_labels_reverse_negation_exact(fixture_dataset) -> None:
    table = fixture_dataset.v52_labels()
    fwd = {(l.curve_id, l.node_index): l.q for l in table.by_sample.values()
           if l.direction == "forward" and l.q_mask}
    rev = {(l.curve_id, l.node_index): l.q for l in table.by_sample.values()
           if l.direction == "reverse" and l.q_mask}
    assert set(fwd) == set(rev) and fwd
    for key in fwd:
        assert abs(fwd[key] + rev[key]) <= 1e-15   # exact negation, not approx


def test_fixture_labels_q_anchors_and_status(fixture_dataset) -> None:
    table = fixture_dataset.v52_labels()
    assert table.zero_arc_curves == ()     # jittered frames ⇒ no zero arcs
    for lab in table.by_sample.values():
        assert set((lab.stop,)) <= {-1, 0, 1}
        if lab.q_mask:
            assert -1.0 - 1e-12 <= lab.q <= 1.0 + 1e-12
    # TS frame is a near-TS positive for BOTH intents (d_key_TS = 0)
    ts_rows = [l for l in table.by_sample.values() if l.d_key_ts < 1e-12]
    assert ts_rows and all(l.stop == 1 for l in ts_rows)


def test_fixture_mask_semantics_table_fields(fixture_dataset) -> None:
    table = fixture_dataset.v52_labels()
    for lab in table.by_sample.values():
        assert lab.stop_mask == (lab.stop in (0, 1))
        if not lab.q_mask:
            assert lab.q is None            # masked q is None, never NaN


def test_labels_only_from_six_tables_no_truth_read(fixture_dataset) -> None:
    """The label path never touches ground truth (static + runtime check)."""
    import inspect

    from pes2ts_core.pesn.offline.data import key_atom_dataset_v3 as mod
    src = inspect.getsource(mod.build_v52_labels)
    for forbidden in ("ground_truth", "truth_reader", "pes2ts_core.g0",
                      "pes2ts_core.g1", "data/ground"):
        assert forbidden not in src, forbidden
    table = fixture_dataset.v52_labels()
    assert len(table.by_sample) == len(fixture_dataset.samples)
