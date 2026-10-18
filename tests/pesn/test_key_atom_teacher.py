"""Synthetic tests for the V5 key-atom teacher lane (ADR-0026).

Covers the ADR §5 math obligations with hand-built fixtures only — no truth
access, no real data files, no writes outside ``tmp_path``:

* weighted Kabsch: proper-rotation recovery, planar references, refusal to
  absorb reflections, weight sensitivity;
* the first-contact teacher: exact analytic contact / cap / terminal events
  on 1-D motions (contact solved on RMS_A² — the mean-square form of the
  ADR §2.3 quadratic, identical to D²=d0² for one key atom);
* flat-tail marking: exact prefix flagging, q=0.25, 20% cap / tail_review,
  and monotone branches staying uncut;
* the v5 split builder: fold disjointness/uniqueness, seeded reproducibility,
  union-find duplicate closure;
* KeyAtomTask: build from a tiny hand-built wp2-task + p1-mapping pair with
  one broken and one formed bond — key rows, R/P endpoint matching, signed
  ΔB, strict save/load roundtrip.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.contracts.key_atom import (  # noqa: E402
    REASON_NO_FUTURE_STEP,
    REASON_OK,
    KeyAtomTaskError,
    build_key_atom_task,
    load_key_atom_task,
)
from pes2ts_core.pesn.offline.data.key_atom_teacher import (  # noqa: E402
    NATIVE_SCHEMA,
    TEACHER_CAP_ANGSTROM,
    apply_rigid_transform,
    build_branch_native,
    build_frame_contract,
    conflict_audit,
    compute_reference_block,
    key_path_arc,
    mark_flat_tail,
    reference_gate,
    solve_teacher,
    weighted_kabsch,
    write_native_shard,
    write_teacher_shard,
)
from pes2ts_core.pesn.offline.splits.v5_fold import (  # noqa: E402
    V5SplitError,
    build_v5_split,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _proper_rotation(rng: np.random.Generator) -> np.ndarray:
    """Random proper rotation via QR with det fix (float64)."""
    m = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(m)
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _random_cloud(rng: np.random.Generator, n: int = 12) -> np.ndarray:
    """Non-degenerate 3-D point cloud (float64)."""
    return rng.normal(size=(n, 3))


def _single_mover_branch(
    steps, n_static_keys: int = 0, n_ref: int = 4, seed: int = 0,
):
    """Branch where key row 0 walks ``steps`` (per-frame +x displacement)
    and every other atom stays fixed at deterministic positions.

    Returns ``(pos [T, N, 3], key_rows, ref_rows, ref_weights)``.  With the
    default ``n_static_keys=0`` the key set is exactly the moving atom
    (m=1), so RMS_A equals the atom's displacement — matching the
    orchestrator's analytic 1-D fixtures.
    """
    steps = np.asarray(steps, dtype=np.float64)
    t_frames = len(steps) + 1
    n_keys = 1 + n_static_keys
    n_atoms = n_keys + n_ref
    rng = np.random.default_rng(seed)
    base = np.zeros((n_atoms, 3), dtype=np.float64)
    base[1:n_keys] = rng.normal(scale=1.5, size=(n_static_keys, 3))
    base[n_keys:] = rng.normal(scale=1.5, size=(n_ref, 3))
    # give the reference block a solid scatter (gate-friendly)
    base[n_keys] += np.array([2.0, 0.0, 0.0])
    base[n_keys + 1] += np.array([0.0, 2.0, 0.0])
    pos = np.repeat(base[None, :, :], t_frames, axis=0)
    pos[:, 0, 0] = np.concatenate([[0.0], np.cumsum(steps)])
    key_rows = np.arange(n_keys, dtype=np.int64)
    ref_rows = np.arange(n_keys, n_atoms, dtype=np.int64)
    ref_weights = np.zeros(n_atoms, dtype=np.float64)
    ref_weights[ref_rows] = 0.25
    return pos, key_rows, ref_rows, ref_weights


# ---------------------------------------------------------------------------
# weighted Kabsch (ADR-0026 §2.2)
# ---------------------------------------------------------------------------


def test_weighted_kabsch_recovers_proper_rotation():
    rng = np.random.default_rng(7)
    x = _random_cloud(rng, 12)
    rot = _proper_rotation(rng)
    shift = np.array([1.5, -2.0, 0.7])
    y = x @ rot.T + shift
    aligned, (r_found, _muy, _mux) = weighted_kabsch(y, x, np.ones(len(x)))
    assert np.max(np.abs(aligned - x)) < 1e-8
    assert np.linalg.det(r_found) == pytest.approx(1.0, abs=1e-10)
    assert np.max(np.abs(r_found - rot)) < 1e-8


def test_weighted_kabsch_planar_reference_works():
    rng = np.random.default_rng(11)
    x = np.column_stack([
        rng.normal(size=8), rng.normal(size=8), np.zeros(8)])
    rot = _proper_rotation(rng)
    y = x @ rot.T
    aligned, (r_found, _muy, _mux) = weighted_kabsch(y, x, np.ones(8))
    assert np.max(np.abs(aligned - x)) < 1e-8
    assert np.linalg.det(r_found) == pytest.approx(1.0, abs=1e-10)


def test_weighted_kabsch_does_not_absorb_reflection():
    rng = np.random.default_rng(13)
    x = _random_cloud(rng, 10)
    mirror = x * np.array([1.0, 1.0, -1.0])  # improper (det -1) image
    aligned, (r_found, _muy, _mux) = weighted_kabsch(
        mirror, x, np.ones(len(x)))
    rmsd = np.sqrt(np.mean(np.sum((aligned - x) ** 2, axis=1)))
    assert np.linalg.det(r_found) == pytest.approx(1.0, abs=1e-10)
    # a chiral cloud cannot be superposed on its mirror by a proper rotation
    assert rmsd > 1e-3


def test_weighted_kabsch_weight_variation_changes_solution():
    rng = np.random.default_rng(17)
    x = _random_cloud(rng, 9)
    rot = _proper_rotation(rng)
    y = x @ rot.T
    y[0] = y[0] + np.array([0.8, 0.0, 0.0])  # displace one atom in Y
    uniform, _ = weighted_kabsch(y, x, np.ones(9))
    # concentrate ALL weight on the displaced atom: centroids coincide with
    # that atom's pair, so the fit maps it onto x[0] exactly
    w = np.zeros(9)
    w[0] = 1.0
    heavy, _ = weighted_kabsch(y, x, w)
    diff = np.linalg.norm(uniform - heavy, axis=1).sum()
    assert diff > 1e-3  # weight distribution changes the fitted rotation
    assert np.linalg.norm(heavy[0] - x[0]) < 1e-8  # and follows that atom
    assert np.linalg.norm(uniform[0] - x[0]) > 1e-3
    # both fits stay proper rotations
    _, (r_u, _, _) = weighted_kabsch(y, x, np.ones(9))
    _, (r_h, _, _) = weighted_kabsch(y, x, w)
    assert np.linalg.det(r_u) == pytest.approx(1.0, abs=1e-10)
    assert np.linalg.det(r_h) == pytest.approx(1.0, abs=1e-10)


def test_weighted_kabsch_rejects_bad_weights():
    x = np.zeros((3, 3))
    with pytest.raises(Exception):
        weighted_kabsch(x, x, np.zeros(3))


def test_apply_rigid_transform_matches_kabsch_output():
    rng = np.random.default_rng(19)
    x = _random_cloud(rng, 6)
    rot = _proper_rotation(rng)
    y = x @ rot.T + np.array([0.3, 0.1, -0.2])
    aligned, transform = weighted_kabsch(y, x, np.ones(6))
    reapplied = apply_rigid_transform(y, transform)
    assert np.max(np.abs(aligned - reapplied)) < 1e-12


# ---------------------------------------------------------------------------
# first-contact teacher (ADR-0026 §2.3)
# ---------------------------------------------------------------------------


def test_contact_exact_frame5_d0_005():
    """0.01 A/frame walk (m=1): d0=0.05 hits exactly at frame 5.

    The earliest root is alpha=1 of segment 4 — the same physical event as
    "alpha=0 at frame 5" — with u=[0.05,0,0] and d_eff=0.05.
    """
    steps = np.full(9, 0.01)
    pos, key_rows, ref_rows, ref_w = _single_mover_branch(steps)
    out = solve_teacher(pos, key_rows, ref_rows, ref_w, 0.05)
    assert out["eligible"][0]
    assert not out["terminal"][0]
    assert not out["capped"][0]
    assert out["d_eff"][0] == pytest.approx(0.05, abs=1e-8)
    np.testing.assert_allclose(out["u_raw"][0][0], [0.05, 0.0, 0.0], atol=1e-8)
    assert out["event_segment"][0] == 4
    assert out["event_alpha"][0] == pytest.approx(1.0, abs=1e-8)
    assert out["reason_id"][0] == REASON_OK


def test_contact_midpoint_root_d0_012():
    """0.1 A/frame walk: d0=0.12 has its exact root alpha=0.2 in segment 1."""
    steps = np.full(5, 0.1)
    pos, key_rows, ref_rows, ref_w = _single_mover_branch(steps)
    out = solve_teacher(pos, key_rows, ref_rows, ref_w, 0.12)
    assert out["eligible"][0]
    assert not out["capped"][0]
    assert out["event_segment"][0] == 1
    assert out["event_alpha"][0] == pytest.approx(0.2, abs=1e-8)
    assert out["d_eff"][0] == pytest.approx(0.12, abs=1e-8)
    np.testing.assert_allclose(out["u_raw"][0][0], [0.12, 0.0, 0.0], atol=1e-8)


def test_cap_event_preempts_contact():
    """One fast key atom among 24 static ones (m=25): RMS_A stays below d0
    while the moving atom's norm hits cap=0.20 first — the cap event lands
    at exactly 0.20 A, alpha=1/3 of the second segment (plan §1.3)."""
    steps = np.full(4, 0.15)
    pos, key_rows, ref_rows, ref_w = _single_mover_branch(
        steps, n_static_keys=24)
    out = solve_teacher(pos, key_rows, ref_rows, ref_w, 0.05)
    assert out["eligible"][0]
    assert out["capped"][0]
    assert not out["terminal"][0]
    assert out["event_segment"][0] == 1
    assert out["event_alpha"][0] == pytest.approx(1.0 / 3.0, abs=1e-8)
    np.testing.assert_allclose(
        np.linalg.norm(out["u_raw"][0][0]), 0.20, atol=1e-8)
    np.testing.assert_allclose(out["u_raw"][0][1:], 0.0, atol=1e-12)
    # d_eff = RMS over all key atoms at the cap point = 0.20 / sqrt(25)
    expected_rms = 0.20 / np.sqrt(len(key_rows))
    assert out["d_eff"][0] == pytest.approx(expected_rms, abs=1e-8)


def test_terminal_when_ts_below_d0():
    """Short branch whose TS is still inside d0: terminal label = residual."""
    steps = np.array([0.01, 0.01, 0.01])
    pos, key_rows, ref_rows, ref_w = _single_mover_branch(steps)
    out = solve_teacher(pos, key_rows, ref_rows, ref_w, 0.05)
    assert out["terminal"][0]
    assert not out["capped"][0]
    np.testing.assert_allclose(out["u_raw"][0][0], [0.03, 0.0, 0.0], atol=1e-8)
    assert out["d_eff"][0] == pytest.approx(0.03, abs=1e-8)
    assert out["eligible"].all()
    # near_ts is the REMAINING key-arc fraction (plan §1.3): 1 at the start
    assert out["near_ts"][0] == pytest.approx(1.0, abs=1e-12)
    assert out["near_ts"][-1] < out["near_ts"][0]


def test_no_future_step_row_is_ineligible():
    """Identical future frames -> no MOVE label; never a fabricated STOP."""
    steps = np.array([0.01, 0.01, 0.0, 0.0, 0.0])
    pos, key_rows, ref_rows, ref_w = _single_mover_branch(steps)
    out = solve_teacher(pos, key_rows, ref_rows, ref_w, 0.05)
    for t in (2, 3, 4):  # start rows with a fully identical future
        assert not out["eligible"][t]
        assert out["reason_id"][t] == REASON_NO_FUTURE_STEP
    assert out["eligible"][0]
    assert out["reason_id"][0] == REASON_OK


def test_gauge_unresolved_branch_exports_ineligible_rows():
    steps = np.full(4, 0.05)
    pos, key_rows, ref_rows, ref_w = _single_mover_branch(steps)
    out = solve_teacher(pos, key_rows, ref_rows, ref_w, 0.05,
                        gauge_ok=False)
    assert not out["eligible"].any()
    np.testing.assert_allclose(out["u_raw"], 0.0)


def test_teacher_row_count_is_t_minus_1():
    steps = np.full(6, 0.02)
    pos, key_rows, ref_rows, ref_w = _single_mover_branch(steps)
    out = solve_teacher(pos, key_rows, ref_rows, ref_w, 0.05)
    assert len(out["start_row"]) == len(pos) - 1
    assert out["u_raw"].shape == (len(pos) - 1, len(key_rows), 3)


# ---------------------------------------------------------------------------
# flat-tail marking (plan §1.1)
# ---------------------------------------------------------------------------


def _pair_branch(q_of_frame, drift=None):
    """Two key atoms: separation ``q`` along x, both share ``drift`` along y
    (rigid co-motion keeps q_l constant while the key arc grows)."""
    t_frames = len(q_of_frame)
    pos = np.zeros((t_frames, 3, 3), dtype=np.float64)
    q = np.asarray(q_of_frame, dtype=np.float64)
    if drift is None:
        drift = np.zeros(t_frames)
    drift = np.asarray(drift, dtype=np.float64)
    pos[:, 0, 1] = drift
    pos[:, 1, 0] = q
    pos[:, 1, 1] = drift
    # a spectator reference atom keeps arrays non-degenerate
    pos[:, 2] = np.array([0.4, 1.8, -0.9])
    key_rows = [0, 1]
    k_pairs = [(0, 1, 1.0, 0.0, -1.0)]
    return pos, key_rows, k_pairs


def test_flat_prefix_is_tailed_with_q025():
    """Key distance constant for 8 frames then rising: the contiguous low-f
    prefix is flagged with q=0.25.  The central-difference vq makes the LAST
    flat frame (f already sees the rise) part of the transition, so rows
    0..6 — 7 of the 8 flat frames — are the flat tail."""
    q = [1.5] * 8 + [1.5 + 0.05 * (j - 7) for j in range(8, 16)]
    pos, key_rows, k_pairs = _pair_branch(q)
    q_frame, tail_flag, meta = mark_flat_tail(pos, key_rows, k_pairs)
    assert meta["reason"] == "flat_tail_candidate"
    assert not meta["tail_review"]
    assert tail_flag[:7].all()
    assert not tail_flag[7:].any()
    assert (q_frame[:7] == np.float32(0.25)).all()
    assert (q_frame[7:] == np.float32(1.0)).all()
    assert meta["prefix_q_range"] < 0.02
    assert meta["tail_arc_fraction"] <= 0.20


def test_monotone_branch_has_no_tail():
    q = [1.0 + 0.05 * j for j in range(12)]
    pos, key_rows, k_pairs = _pair_branch(q)
    q_frame, tail_flag, meta = mark_flat_tail(pos, key_rows, k_pairs)
    assert not tail_flag.any()
    assert (q_frame == np.float32(1.0)).all()
    assert meta["reason"] != "flat_tail_candidate"


def test_tail_review_when_arc_cap_exceeded():
    """Rigid co-motion prefix keeps q_l constant but eats >20% of the key
    arc: the whole branch becomes tail_review — frames kept, weights
    unchanged (never auto-extended)."""
    t_frames = 16
    drift = np.concatenate([np.linspace(0.0, 0.7, 8), np.full(8, 0.7)])
    q = [1.5] * 8 + [1.5 + 0.2 * (j - 7) for j in range(8, 16)]
    pos, key_rows, k_pairs = _pair_branch(q, drift=drift)
    q_frame, tail_flag, meta = mark_flat_tail(pos, key_rows, k_pairs)
    assert meta["tail_review"]
    assert not tail_flag.any()
    assert (q_frame == np.float32(1.0)).all()
    assert meta["tail_arc_fraction"] > 0.20


def test_key_path_arc_and_progress_monotone():
    q = [1.0 + 0.05 * j for j in range(6)]
    pos, key_rows, _k = _pair_branch(q)
    ds = key_path_arc(pos, key_rows)
    assert len(ds) == len(pos) - 1
    assert (ds > 0).all()
    total = ds.sum()
    frac = np.concatenate([[0.0], np.cumsum(ds)]) / total
    assert frac[-1] == pytest.approx(1.0)
    assert np.all(np.diff(frac) >= 0)


def test_reference_block_keeps_rigid_heavy_neighbors():
    """Chain C(key)-C-C=C-C-H: bond 1-2 (both degrees >=2, non-ring single)
    is a free rotor and gets cut, so only row 1 stays within the rigid block
    of the key atom; rows 2+ are excluded despite hop <= 2."""
    z = [6, 6, 6, 6, 6, 1]
    bonds = [
        (0, 1, 1.0), (1, 2, 1.0), (2, 3, 2.0),
        (3, 4, 1.0), (4, 5, 1.0),
    ]
    ref = compute_reference_block(z, bonds, key_rows=[0])
    rows = set(int(r) for r in ref["rows"])
    assert rows == {1}


def test_reference_block_keeps_terminal_tips_attached():
    """A degree-1 heavy tip (F) attached to a KEPT atom stays in S; the
    tip's own degree-1 bond is never a rotor cut.  The C0=C1 bond is a
    double bond, so C1 (and its F tip) share the key atom's rigid block."""
    z = [6, 6, 9, 1, 1]
    bonds = [
        (0, 1, 2.0),  # C0(key)=C1: rigid
        (1, 2, 1.0),  # C1-F2: degree-1 tip, stays attached to kept C1
        (1, 3, 1.0),  # C1-H3: hydrogen, never reference
        (0, 4, 1.0),  # C0-H4
    ]
    ref = compute_reference_block(z, bonds, key_rows=[0])
    rows = set(int(r) for r in ref["rows"])
    assert rows == {1, 2}


def test_reference_gate_rejects_collinear_block():
    pos0 = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0],
    ])
    rows = np.array([1, 2, 3])
    weights = np.full(4, 0.25)
    ok, evals = reference_gate(pos0, rows, weights)
    assert not ok  # lambda2 == 0 on a collinear block
    assert evals[0] > 0.01


def test_reference_gate_accepts_spread_block():
    pos0 = np.array([
        [0.0, 0.0, 0.0], [1.5, 0.2, 0.1], [-0.4, 1.4, -0.3], [0.3, -0.5, 1.3],
    ])
    rows = np.array([1, 2, 3])
    weights = np.full(4, 0.25)
    ok, evals = reference_gate(pos0, rows, weights)
    assert ok
    assert evals[0] > evals[1] >= 0


def test_build_branch_native_end_to_end_synthetic():
    """Reference + gate + tail + progress on one synthetic branch."""

    class _TinyTask:
        atomic_numbers = (6, 6, 8, 8, 6)
        key_atom_rows = (0, 1)

        def start_bonds(self, branch):
            # C0(key)-O2, C1(key)-N3, O2=C4, N3=C4: no free single rotor
            # between the key set and rows 2-4 (2-4 double, 3-4 double)
            return ((0, 2, 1.0), (1, 3, 1.0), (2, 4, 2.0), (3, 4, 2.0))

        def K_pairs(self, branch):
            return ((0, 1, 0.0, 0.0, 0.0),)

    q = [1.4] * 8 + [1.4 + 0.04 * (j - 7) for j in range(8, 14)]
    t_frames = len(q)
    pos = np.zeros((t_frames, 5, 3))
    pos[:, 1, 0] = q
    pos[:, 2] = [1.2, 0.3, -0.8]
    pos[:, 3] = [-1.1, 0.9, 0.4]
    pos[:, 4] = [0.2, -1.3, 1.1]
    record = build_branch_native(pos, _TinyTask(), branch=0)
    assert record["gate_ok"]
    assert set(int(r) for r in record["ref_rows"]) == {2, 3, 4}
    # low-f prefix rows flagged (central-difference bleed at frame 7)
    assert record["tail_flag"][:7].all()
    assert not record["tail_flag"][7:].any()
    assert record["key_progress"][-1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# shard writers (contract shapes/dtypes; tmp_path only)
# ---------------------------------------------------------------------------


def test_native_and_teacher_shard_contract(tmp_path):
    steps = np.full(6, 0.02)
    pos, key_rows, ref_rows, ref_w = _single_mover_branch(steps)
    q_frame = np.ones(len(pos), dtype=np.float32)
    q_frame[:2] = np.float32(0.25)
    tail_flag = np.zeros(len(pos), dtype=bool)
    tail_flag[:2] = True
    progress = np.linspace(0.0, 1.0, len(pos))
    artifacts = write_native_shard(
        tmp_path / "native", "RXN_SYNTH1", 0, pos, key_rows,
        q_frame, progress, tail_flag,
        {"schema_version": NATIVE_SCHEMA, "reaction_id": "RXN_SYNTH1",
         "branch": 0, "n_frames": int(len(pos))})
    npz_path = Path(artifacts["native_npz"]["path"])
    assert npz_path.parent == tmp_path / "native"
    with np.load(npz_path) as data:
        assert set(data.files) == {
            "pos", "key_rows", "q_frame", "key_progress", "tail_flag"}
        assert data["pos"].dtype == np.float64
        assert data["key_rows"].dtype == np.int64
        assert data["q_frame"].dtype == np.float32
        assert data["key_progress"].dtype == np.float64
        assert data["tail_flag"].dtype == bool
        assert data["pos"].shape == (len(pos), pos.shape[1], 3)

    out = solve_teacher(pos, key_rows, ref_rows, ref_w, 0.05)
    teacher = write_teacher_shard(
        tmp_path / "teacher", 0.05, "RXN_SYNTH1", 0, out)
    assert Path(teacher["path"]).parent == tmp_path / "teacher" / "d0_0.05"
    with np.load(teacher["path"]) as data:
        assert set(data.files) == {
            "start_row", "u_raw", "eligible", "reason_id", "terminal",
            "capped", "d_eff", "eta", "near_ts"}
        assert data["start_row"].dtype == np.int64
        assert data["u_raw"].dtype == np.float32
        assert data["eligible"].dtype == bool
        assert data["reason_id"].dtype == np.uint8
        assert data["terminal"].dtype == bool
        assert data["capped"].dtype == bool
        assert data["d_eff"].dtype == np.float32
        assert data["eta"].dtype == np.float32
        assert data["near_ts"].dtype == np.float32
        s_rows = len(pos) - 1
        assert data["start_row"].shape == (s_rows,)
        assert data["u_raw"].shape == (s_rows, len(key_rows), 3)


# ---------------------------------------------------------------------------
# conflict audit (plan §1.2 bounded; synthetic records)
# ---------------------------------------------------------------------------


def _audit_record(reaction_id, branch, pos0, key_rows, u0, elements,
                  degrees, is_ring):
    return {
        "reaction_id": reaction_id,
        "branch": branch,
        "pos0": np.asarray(pos0, dtype=np.float64),
        "key_rows": np.asarray(key_rows, dtype=np.int64),
        "u0": np.asarray(u0, dtype=np.float64),
        "atomic_numbers": list(elements),
        "deg_start": list(degrees),
        "is_ring": list(is_ring),
    }


def test_conflict_audit_flags_antipodal_labels():
    """Identical X0 but antipodal teacher labels on the same branch: the
    conflict component is zero-weighted as a whole."""
    rng = np.random.default_rng(3)
    pos0 = rng.normal(size=(4, 3))
    elements = [6, 7, 8, 9]  # distinct elements -> identity orbit only
    degrees = [1, 2, 2, 1]
    is_ring = [False, False, False, False]
    key_rows = [0, 1, 2, 3]
    u = np.full((4, 3), 0.05)
    records = [
        _audit_record("RXN_A", 0, pos0, key_rows, u, elements, degrees,
                      is_ring),
        _audit_record("RXN_B", 0, pos0, key_rows, -u, elements, degrees,
                      is_ring),
        _audit_record("RXN_C", 1, pos0, key_rows, u, elements, degrees,
                      is_ring),
    ]
    audit, weights = conflict_audit(records)
    assert audit["scope"] == "fit50_internal"
    assert audit["pair_outcomes"]["conflict"] == 1
    # the conflicting pair (same branch, antipodal labels) zeroes both
    assert weights["RXN_A"]["0"] == 0.0
    assert weights["RXN_B"]["0"] == 0.0
    # the other-branch twin is context_resolved, weight stays 1
    assert weights["RXN_C"]["1"] == 1.0
    assert 0.0 < audit["isolation_rate"] <= 1.0
    assert audit["residual_conflict_rate"] == 0.0


def test_conflict_audit_near_conflict_weight():
    """Labels with flattened cosine in [0, 0.25) -> near-conflict, 0.25."""
    rng = np.random.default_rng(5)
    pos0 = rng.normal(size=(4, 3))
    elements = [6, 7, 8, 9]
    degrees = [1, 2, 2, 1]
    is_ring = [False] * 4
    key_rows = [0, 1, 2, 3]
    u_a = np.full((4, 3), 0.05)
    # per-atom rows [2, -0.9, -1] * 0.05: sum 0.1*c vs norm sqrt(5.81)*c
    # -> flattened cosine = 0.1 / (sqrt(3)*sqrt(5.81)) ~ 0.024 in [0, 0.25)
    u_b = np.tile(np.array([[2.0, -0.9, -1.0]]) * 0.05, (4, 1))
    records = [
        _audit_record("RXN_A", 0, pos0, key_rows, u_a, elements, degrees,
                      is_ring),
        _audit_record("RXN_B", 0, pos0, key_rows, u_b, elements, degrees,
                      is_ring),
    ]
    audit, weights = conflict_audit(records)
    assert audit["pair_outcomes"]["near_conflict"] == 1
    assert weights["RXN_A"]["0"] == pytest.approx(0.25)
    assert weights["RXN_B"]["0"] == pytest.approx(0.25)


def test_conflict_audit_overflow_is_recorded_not_guessed():
    """Class symmetry beyond the orbit cap -> mapping_group_overflow."""
    rng = np.random.default_rng(9)
    n = 8
    pos0 = rng.normal(size=(n, 3))
    # all atoms in ONE invariant class: 8! = 40320 >> 128
    elements = [6] * n
    degrees = [2] * n
    is_ring = [True] * n
    key_rows = list(range(n))
    u = rng.normal(size=(n, 3)) * 0.05
    records = [
        _audit_record("RXN_A", 0, pos0, key_rows, u, elements, degrees,
                      is_ring),
        _audit_record("RXN_B", 0, pos0 + 1e-4, key_rows, -u, elements,
                      degrees, is_ring),
    ]
    audit, weights = conflict_audit(records)
    assert audit["pair_outcomes"]["mapping_group_overflow"] == 1
    assert len(audit["overflow_pairs"]) == 1
    # overflow never zeroes by guesswork
    assert weights["RXN_A"]["0"] == 1.0
    assert weights["RXN_B"]["0"] == 1.0


def test_conflict_audit_size_mismatch_skipped():
    a = _audit_record("RXN_A", 0, np.zeros((3, 3)), [0, 1, 2],
                      np.ones((3, 3)) * 0.05, [6, 7, 8], [1, 2, 1],
                      [False] * 3)
    b = _audit_record("RXN_B", 0, np.zeros((4, 3)), [0, 1, 2, 3],
                      np.ones((4, 3)) * 0.05, [6, 7, 8, 9], [1, 2, 2, 1],
                      [False] * 4)
    audit, weights = conflict_audit([a, b])
    assert audit["pair_outcomes"]["size_mismatch"] == 1
    assert weights["RXN_A"]["0"] == 1.0


def test_frame_contract_documents_adr_semantics():
    doc = build_frame_contract()
    assert doc["schema_version"] == "pesn_v5_frame_contract_v1"
    assert doc["cap_angstrom"] == TEACHER_CAP_ANGSTROM
    assert doc["root_finding"]["root_tolerance"] == 1e-8
    assert "proper only" in doc["kabsch"]["rotation"]
    assert doc["d0_ablation_angstrom"] == [0.05, 0.08, 0.12]


# ---------------------------------------------------------------------------
# KeyAtomTask (ADR-0026 §2.1)
# ---------------------------------------------------------------------------


def _tiny_wp2_and_p1():
    """6-atom hand-built pair: bond (1,2) broken, bond (2,3) formed.

    Endpoint 0 of the wp2 task is the P graph and endpoint 1 the R graph
    (deliberately swapped vs file order) to exercise bond-set matching.
    """
    p1 = {
        "schema_version": "g1_p1_truth_v1",
        "reaction_id": "RXN_TEST000042",
        "graph": {
            "r_bonds": [[1, 2, 1.0], [4, 5, 1.0], [5, 6, 1.0]],
            "p_bonds": [[2, 3, 1.0], [4, 5, 1.0], [5, 6, 1.0]],
        },
        "bond_events": {
            "broken": [{"atoms": [1, 2], "order_r": 1.0, "order_p": None}],
            "formed": [{"atoms": [2, 3], "order_p": 1.0, "order_r": None}],
            "order_changed": [],
            "hydrogen_migration": [],
        },
        "mapping": {
            "map_to_atoms": [
                {"map": m, "ts_irc_index": m - 1, "element": el}
                for m, el in enumerate(
                    ["C", "N", "O", "C", "C", "H"], start=1)
            ],
        },
        "irc_validation": {
            "layout": "ts_first_dual_branch",
            "orientation": "P_first",
            "ts_frame_index": 0,
            "splice_index": 3,
        },
    }

    def _endpoint(bonds):
        return {
            "atomic_numbers": [6, 7, 8, 6, 6, 1],
            "atom_map_ids": [f"m{m:04d}" for m in range(1, 7)],
            "bonds": [
                {"i": i - 1, "j": j - 1, "bond_order": o}
                for i, j, o in bonds
            ],
        }

    wp2 = {
        "schema_version": "pesn_dual_endpoint_task_v2",
        "reaction_id": "RXN_TEST000042",
        "reaction_group_id": "RXN_TEST000042",
        # endpoint 0 = PRODUCT graph, endpoint 1 = REACTANT graph
        "endpoints": [
            _endpoint([(2, 3, 1.0), (4, 5, 1.0), (5, 6, 1.0)]),
            _endpoint([(1, 2, 1.0), (4, 5, 1.0), (5, 6, 1.0)]),
        ],
        "charge": 0,
        "multiplicity": 1,
        "method_id": "B3LYP-D3/def2-TZVP",
    }
    return wp2, p1


def test_key_atom_task_build_and_signs():
    wp2, p1 = _tiny_wp2_and_p1()
    task = build_key_atom_task(wp2, p1)
    # key rows: event maps {1,2,3} -> rows {0,1,2} in ts_irc_index order
    assert task.key_atom_rows == (0, 1, 2)
    assert task.atomic_numbers == (6, 7, 8, 6, 6, 1)
    assert task.atom_map_ids == tuple(f"m{m:04d}" for m in range(1, 7))
    # R/P endpoint matching by bond sets (endpoints deliberately swapped)
    assert task.endpoint_sides == ("P", "R")
    ev = task.endpoint_match_evidence
    assert ev["endpoint_0"]["side"] == "P"
    assert ev["endpoint_1"]["side"] == "R"
    # branch 1 STARTS at the R endpoint: bond (0,1) present at start and
    # absent at the other (P) end -> dB = -1 (breaking); bond (1,2) forms
    k1 = {(i, j): db for i, j, _bs, _bo, db in task.K_pairs(1)}
    assert k1[(0, 1)] == pytest.approx(-1.0)
    assert k1[(1, 2)] == pytest.approx(1.0)
    # branch 0 STARTS at the P endpoint: signs flip
    k0 = {(i, j): db for i, j, _bs, _bo, db in task.K_pairs(0)}
    assert k0[(0, 1)] == pytest.approx(1.0)
    assert k0[(1, 2)] == pytest.approx(-1.0)
    # both endpoints' orders are exposed per pair
    pair_01_b1 = [p for p in task.K_pairs(1) if (p[0], p[1]) == (0, 1)][0]
    assert pair_01_b1[2] == 1.0   # B_start (R endpoint has the bond)
    assert pair_01_b1[3] == 0.0   # B_other
    # union edges = G_R u G_P as sorted row pairs
    assert task.union_edges() == ((0, 1), (1, 2), (3, 4), (4, 5))
    assert task.charge == 0 and task.multiplicity == 1
    assert task.group_id == "RXN_TEST000042"
    assert len(task.sources["wp2_task"]["sha256"]) == 64
    assert len(task.sources["p1_mapping"]["sha256"]) == 64


def test_key_atom_task_roundtrip_and_strict_load(tmp_path):
    wp2, p1 = _tiny_wp2_and_p1()
    task = build_key_atom_task(wp2, p1)
    path = tmp_path / "task.json"
    task.save(path)
    loaded = load_key_atom_task(path)
    assert loaded.to_payload() == task.to_payload()

    doc = json.loads(path.read_text())
    doc["surprise_field"] = 1
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(doc))
    with pytest.raises(KeyAtomTaskError):
        load_key_atom_task(bad)

    doc2 = json.loads(path.read_text())
    doc2["schema_version"] = "pesn_key_atom_task_v99"
    bad2 = tmp_path / "bad2.json"
    bad2.write_text(json.dumps(doc2))
    with pytest.raises(KeyAtomTaskError):
        load_key_atom_task(bad2)


def test_key_atom_task_rejects_ambiguous_side_match():
    wp2, p1 = _tiny_wp2_and_p1()
    # make both endpoints identical to the R graph: P side cannot be matched
    wp2["endpoints"][0] = wp2["endpoints"][1]
    with pytest.raises(KeyAtomTaskError):
        build_key_atom_task(wp2, p1)


# ---------------------------------------------------------------------------
# v5 split builder (plan §1.4; synthetic fixtures, tmp_path only)
# ---------------------------------------------------------------------------


def _write_class_file(interim: Path, rid: str, family: str) -> None:
    """Place the family label under the G1 shard dir the loader expects."""
    shard = f"{int(rid.split('_')[-1]) // 1000:05d}"
    path = interim / "g1_v2" / "classes" / shard / f"{rid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "levels": {"l0u_undirected_family": {"cluster_id": family}},
    }))


def _write_split_fixtures(tmp_path, n_families=6, groups_per_family=10):
    """Fabricate a fit50 selection + lineage + interim family tree."""
    interim = tmp_path / "interim"
    group_ids = []
    families = [f"g1p2_taxonomy_v2:l0u_undirected_family:family{i:02d}"
                for i in range(n_families)]
    for i in range(50):
        rid = f"RXN_{i + 1:010d}"
        group_ids.append(rid)
        _write_class_file(interim, rid, families[i % n_families])

    # extra fit-pool groups (beyond fit50) for confirm candidates
    fit_pool = list(group_ids)
    for i in range(50, 130):
        rid = f"RXN_{i + 1:010d}"
        fit_pool.append(rid)
        _write_class_file(interim, rid, families[i % n_families])

    # family holdout: 4 sealed families x 12 groups
    holdout = []
    sealed_families = []
    sealed_reactions = []
    for fi in range(4):
        fam = f"g1p2_taxonomy_v2:l0u_undirected_family:sealed{fi:02d}"
        sealed_families.append(fam)
        for gi in range(12):
            rid = f"RXN_{900000 + fi * 100 + gi:010d}"
            holdout.append(rid)
            sealed_reactions.append(rid)
            _write_class_file(interim, rid, fam)

    # duplicate ledger: groups 1 and 2 are one identity
    (interim / "duplicate_groups.json").write_text(json.dumps({
        "schema_version": "g0_manifest_v1",
        "groups": [{
            "canonical_reaction_id": group_ids[0],
            "identity": "deadbeef",
            "member_reaction_ids": [group_ids[0], group_ids[1]],
        }],
    }))

    selection_path = tmp_path / "selection_direct_fit50.json"
    selection_path.write_text(json.dumps({
        "schema_version": "pes2ts_direct_selection_v1",
        "selection": "direct_fit50",
        "group_ids": group_ids,
        "reaction_ids": group_ids,
    }))
    lineage_path = tmp_path / "dual_ts_split_lineage_v1.json"
    lineage_path.write_text(json.dumps({
        "schema_version": "pesn_dual_ts_split_lineage_v1",
        "partitions": {
            "fit": {"group_ids": fit_pool, "n_groups": len(fit_pool)},
            "family_holdout": {
                "group_ids": holdout, "n_groups": len(holdout)},
        },
        "family_audit": {
            "sealed_families": sealed_families,
            "sealed_reaction_ids": sealed_reactions,
        },
        "v1_binding": {
            "reaction_to_group": {gid: gid for gid in fit_pool}},
    }))
    return selection_path, lineage_path, interim


def test_v5_split_folds_disjoint_unique_reproducible(tmp_path):
    selection, lineage, interim = _write_split_fixtures(tmp_path)
    manifest = build_v5_split(
        selection, lineage, seed=20261018, interim_dir=interim,
        v1_split_manifest_path=None, parent_manifest_path=None)
    assert manifest["schema_version"] == "pesn_v5_split_v1"
    folds = manifest["folds"]
    assert set(folds) == {"0", "1", "2", "3", "4"}
    all_groups = [g for k in folds for g in folds[k]]
    assert len(all_groups) == 50
    assert len(set(all_groups)) == 50
    sizes = [len(folds[k]) for k in folds]
    assert max(sizes) - min(sizes) <= 2
    # duplicate-merged groups land in one fold
    assert manifest["group_fold"]["RXN_0000000001"] == \
        manifest["group_fold"]["RXN_0000000002"]
    merged = [c for c in manifest["components"] if len(c["group_ids"]) > 1]
    assert len(merged) == 1
    assert sorted(merged[0]["group_ids"]) == [
        "RXN_0000000001", "RXN_0000000002"]
    # sealed candidates
    confirm = manifest["confirm_candidates"]
    assert confirm["sealed"] is True
    assert confirm["n_groups"] >= 30
    assert not (set(confirm["group_ids"]) & set(all_groups))
    ood = manifest["ood_candidates"]
    assert ood["sealed"] is True
    assert ood["n_families"] >= 3
    assert all(len(v) >= 10 for v in ood["families"].values())
    # parent hashes recorded
    assert "fit50_selection" in manifest["parents"]
    assert "dual_ts_lineage" in manifest["parents"]
    assert len(manifest["history_exclusion_sha256"]) == 64
    assert manifest["assertions"]["folds_pairwise_disjoint"] is True
    assert manifest["assertions"]["group_fold_unique"] is True
    assert manifest["frozen"] is True

    # same seed -> identical assignment and sealed pools
    manifest2 = build_v5_split(
        selection, lineage, seed=20261018, interim_dir=interim,
        v1_split_manifest_path=None, parent_manifest_path=None)
    assert manifest2["group_fold"] == manifest["group_fold"]
    assert manifest2["confirm_candidates"] == confirm
    assert manifest2["ood_candidates"] == ood


def test_v5_split_rejects_bad_selection(tmp_path):
    selection, lineage, interim = _write_split_fixtures(tmp_path)
    bad = tmp_path / "bad_selection.json"
    bad.write_text(json.dumps({
        "schema_version": "pes2ts_direct_selection_v1",
        "group_ids": ["RXN_0000000001"],  # != 50
        "reaction_ids": ["RXN_0000000001"],
    }))
    with pytest.raises(V5SplitError):
        build_v5_split(bad, lineage, interim_dir=interim,
                       v1_split_manifest_path=None,
                       parent_manifest_path=None)


def test_v5_split_insufficient_ood_pool_refuses(tmp_path):
    selection, lineage, interim = _write_split_fixtures(tmp_path)
    doc = json.loads(lineage.read_text())
    # shrink holdout to a single family with 1 group
    doc["partitions"]["family_holdout"] = {
        "group_ids": ["RXN_0000900000"], "n_groups": 1}
    doc["family_audit"] = {
        "sealed_families": [
            "g1p2_taxonomy_v2:l0u_undirected_family:sealed00"],
        "sealed_reaction_ids": ["RXN_0000900000"],
    }
    bad_lineage = tmp_path / "lineage_thin_ood.json"
    bad_lineage.write_text(json.dumps(doc))
    with pytest.raises(V5SplitError):
        build_v5_split(selection, bad_lineage, interim_dir=interim,
                       v1_split_manifest_path=None,
                       parent_manifest_path=None)
