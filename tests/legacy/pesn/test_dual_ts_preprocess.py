"""Dual-TS teacher preprocessing tests (WP1) — SYNTHETIC ONLY, no truth h5.

Covers the v4 plan §I.4 WP1 scope through the core functions: two-branch
split + ordering, no cross-TS chaining, drift-removal identity preservation,
vector-transform synchronization, unordered A/B swap digest identity,
a_max exceedance flagging (never insertion), gaps registry structure,
contract validation of both records, and the task↔teacher identity check.
"""
from __future__ import annotations

import numpy as np
import pytest

from pes2ts_core.pesn.contracts.dual_ts import (
    BOUNDARY_TEACHER_ENDPOINT,
    DualEndpointTask,
    DualTSTeacher,
    TASK_SCHEMA_VERSION,
    TEACHER_SCHEMA_VERSION,
    validate_record,
)
from pes2ts_core.pesn.offline.data.dual_ts_preprocess import (
    A_MAX_ANGSTROM,
    BOND_PERCEPTION_COVALENT,
    DRIFT_RULE,
    DualTSPreprocessError,
    TS_RULE_AUTHORITY,
    TS_RULE_MAX_ENERGY,
    apply_rigid_transform,
    assert_task_teacher_identity,
    build_dual_ts_reaction,
    compute_branch_labels,
    endpoint_graph_from_bond_list,
    endpoint_graph_from_geometry,
    flag_a_max_exceedances,
    identify_ts_frame,
    order_frames_and_energy,
    remove_rigid_drift,
    rotation_matrix_about_z,
    split_dual_branches,
    ts_position_in_ordered_path,
)

Z = (8, 1, 6, 1)  # O, H, C, H — all in the v1 mass table

R_BONDS = [(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)]  # map space, 1-based
P_BONDS = [(1, 2, 1.0), (2, 4, 2.0), (3, 4, 1.0)]  # distinct bond set


def graphs():
    z = np.asarray(Z, dtype=int)
    return (endpoint_graph_from_bond_list(z, R_BONDS),
            endpoint_graph_from_bond_list(z, P_BONDS))


def synthetic_ordered_path(step_scale: float = 0.2) -> np.ndarray:
    """Ordered R→TS→P path: 5 frames, ts_position=2, 4 atoms, mild 3D shape.

    Branch R (endpoint→TS) = frames [0,1,2]; branch P (endpoint→TS) =
    frames [4,3,2] after splitting.  One deliberate large step can be
    injected via ``step_scale`` per-frame multipliers in other tests.
    """
    base = np.array([
        [[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [1.8, 0.0, 0.3], [2.7, 0.1, 0.0]],
        [[0.1, 0.0, 0.0], [1.0, 0.1, 0.1], [1.7, 0.1, 0.2], [2.6, 0.0, 0.1]],
        [[0.2, 0.1, 0.0], [1.3, 0.2, 0.0], [1.5, 0.0, 0.1], [2.4, 0.1, 0.2]],
        [[0.1, 0.0, 0.2], [1.6, 0.1, 0.0], [1.4, 0.1, 0.2], [2.5, 0.0, 0.0]],
        [[0.0, 0.1, 0.1], [1.9, 0.0, 0.1], [1.2, 0.2, 0.0], [2.6, 0.1, 0.1]],
    ], dtype=float)
    return base * step_scale


def synthetic_energies() -> np.ndarray:
    return np.array([0.0, 0.2, 1.0, 0.3, 0.05])  # argmax at the TS frame 2


def build_outcome(path=None, ts_position: int | None = 2, **overrides):
    graph_r, graph_p = graphs()
    kwargs = dict(
        reaction_id="RXN_SYNTH0001",
        reaction_group_id="GRP_SYNTH0001",
        partition="fit",
        atomic_numbers=Z,
        ordered_path=synthetic_ordered_path() if path is None else path,
        authority_ts_position=ts_position,
        energies=synthetic_energies(),
        graph_r=graph_r,
        graph_p=graph_p,
        bond_perception="g1_p1_mapping_endpoint_bonds_v1",
        method_id="synthetic_method",
        source="synthetic fixture",
    )
    kwargs.update(overrides)
    return build_dual_ts_reaction(**kwargs)


def states_of_branch(teacher: DualTSTeacher, label: str):
    return [s for s in teacher.states if s.branch_id.endswith("br" + label)]


# ---------------------------------------------------------------------------
# Two-branch split + ordering
# ---------------------------------------------------------------------------


def test_two_branch_split_and_ordering():
    path = synthetic_ordered_path()
    branch_r, branch_p = split_dual_branches(path, 2)
    # Each branch is internally ordered endpoint → TS; TS is terminal.
    assert branch_r.shape == (3, 4, 3) and branch_p.shape == (3, 4, 3)
    assert np.array_equal(branch_r[0], path[0])   # R endpoint first
    assert np.array_equal(branch_r[-1], path[2])  # TS last
    assert np.array_equal(branch_p[0], path[-1])  # P endpoint first
    assert np.array_equal(branch_p[-1], path[2])  # same shared TS frame last
    # no cross-TS chaining in the split itself: only frame 2 is shared
    assert np.array_equal(branch_r[1], path[1])
    assert np.array_equal(branch_p[1], path[3])


def test_order_frames_and_energy_matches_v1_layout_rule():
    frames = np.arange(5 * 2 * 3, dtype=float).reshape(5, 2, 3)
    energies = np.arange(5, dtype=float)
    # ts_first_dual_branch, P_first: prefix = branch2[::-1] (frames 3..4)
    ordered, ordered_e, ts_pos = order_frames_and_energy(
        frames, energies, layout="ts_first_dual_branch",
        orientation="P_first", ts_frame_index=0, splice_index=2)
    assert ts_pos == 2
    assert np.array_equal(ordered, np.concatenate(
        [frames[3:][::-1], frames[0:1], frames[1:3]]))
    assert np.array_equal(ordered_e, np.concatenate(
        [energies[3:][::-1], energies[0:1], energies[1:3]]))
    assert ts_position_in_ordered_path(
        layout="ts_first_dual_branch", orientation="R_first",
        ts_frame_index=0, splice_index=2) == 2


def test_single_branch_layout_is_a_registered_miss_not_a_fabricated_side():
    with pytest.raises(DualTSPreprocessError) as excinfo:
        ts_position_in_ordered_path(
            layout="single_branch", orientation="R_first",
            ts_frame_index=0, splice_index=None)
    assert excinfo.value.code == "LAYOUT_UNSUPPORTED"


# ---------------------------------------------------------------------------
# Teacher structure: no cross-TS chaining, shared TS, terminal semantics
# ---------------------------------------------------------------------------


def test_teacher_has_two_branches_and_never_chains_across_ts():
    outcome = build_outcome()
    assert outcome.status == "ok"
    teacher = outcome.teacher
    assert {b.endpoint_label for b in teacher.branches} == {"A", "B"}
    assert len(teacher.episodes) == 2
    by_id = {s.state_id: s for s in teacher.states}
    for state in teacher.states:
        if state.boundary_type == BOUNDARY_TEACHER_ENDPOINT:
            assert state.next_state_id is None
            assert by_id[state.state_id].coordinates is not None
        else:
            nxt = by_id[state.next_state_id]
            assert nxt.branch_id == state.branch_id  # never leaves the branch
    # Both branches end at the SAME exported TS geometry (shared frame).
    ts_a = states_of_branch(teacher, "A")[-1]
    ts_b = states_of_branch(teacher, "B")[-1]
    assert np.array_equal(ts_a.coordinates, ts_b.coordinates)
    assert ts_a.boundary_type == ts_b.boundary_type == BOUNDARY_TEACHER_ENDPOINT
    assert len(teacher.states) == 6  # 3 + 3 frames, no inserted nodes


def test_labels_and_boundaries_hold_contract_identities():
    outcome = build_outcome()
    teacher = outcome.teacher
    labels = {lab.state_id: lab for lab in teacher.labels}
    for state in teacher.states:
        lab = labels[state.state_id]
        if state.boundary_type == BOUNDARY_TEACHER_ENDPOINT:
            assert lab.action_delta is None
            assert float(np.max(np.abs(lab.ts_displacement))) == 0.0
            assert lab.remaining_arc_length == 0.0
            assert lab.teacher_endpoint
        else:
            nxt = {s.state_id: s for s in teacher.states}[state.next_state_id]
            a = np.asarray(lab.action_delta)
            assert np.allclose(
                np.asarray(state.coordinates) + a,
                np.asarray(nxt.coordinates), atol=1e-10)
            assert np.allclose(
                np.asarray(lab.ts_displacement)
                - np.asarray(labels[state.next_state_id].ts_displacement)
                - a, 0.0, atol=1e-9)
            assert lab.remaining_arc_length >= labels[
                state.next_state_id].remaining_arc_length


# ---------------------------------------------------------------------------
# Drift removal preserves identities
# ---------------------------------------------------------------------------


def test_drift_removal_is_rigid_and_preserves_label_identities():
    rng = np.random.default_rng(7)
    path = synthetic_ordered_path()
    drifted = np.empty_like(path)
    for i in range(len(path)):
        angle = 0.05 * float(rng.uniform(-1, 1))
        r = rotation_matrix_about_z(angle)
        t = rng.uniform(-0.3, 0.3, size=3)
        drifted[i] = apply_rigid_transform(path[i], r, t)
    # drift actually happened
    assert float(np.max(np.abs(drifted - path))) > 0.1
    masses = np.array([15.999, 1.008, 12.011, 1.008])
    removed = remove_rigid_drift(drifted, masses)
    # rigid invariance: per-frame internal distance matrices are preserved
    for i in range(len(path)):
        d0 = np.linalg.norm(path[i][:, None] - path[i][None], axis=-1)
        d1 = np.linalg.norm(drifted[i][:, None] - drifted[i][None], axis=-1)
        d2 = np.linalg.norm(removed[i][:, None] - removed[i][None], axis=-1)
        assert np.allclose(d0, d1, atol=1e-12)
        assert np.allclose(d0, d2, atol=1e-8)
    # labels differenced from the drift-removed path still satisfy the
    # identities (compute_branch_labels raises LABEL_IDENTITY_BROKEN else)
    actions, d, ells = compute_branch_labels(
        np.round(removed[:3], 10))
    assert actions.shape == (2, 4, 3) and ells[-1] == 0.0
    assert DRIFT_RULE.startswith("chained_mass_weighted_kabsch")


# ---------------------------------------------------------------------------
# Vector-transform synchronization
# ---------------------------------------------------------------------------


def test_rotation_synchronizes_coordinates_and_all_vector_labels():
    outcome = build_outcome()
    teacher = outcome.teacher
    r = rotation_matrix_about_z(0.7)
    t = np.array([1.0, -2.0, 0.5])
    labels = {lab.state_id: lab for lab in teacher.labels}
    for label in ("A", "B"):
        states = states_of_branch(teacher, label)
        coords = np.asarray([s.coordinates for s in states])
        rotated = apply_rigid_transform(coords, r, t)
        # rebuild labels from the rotated exported chain
        a_rot, d_rot, ell_rot = compute_branch_labels(rotated)
        a_orig = np.asarray([labels[s.state_id].action_delta
                             for s in states[:-1]])
        d_orig = np.asarray([labels[s.state_id].ts_displacement
                             for s in states])
        # labels rotated by the SAME transform, translation cancels
        assert np.allclose(a_rot, a_orig @ r.T, atol=1e-12)
        assert np.allclose(d_rot, d_orig @ r.T, atol=1e-12)
        assert np.allclose(ell_rot[:-1],
                           [labels[s.state_id].remaining_arc_length
                            for s in states[:-1]], atol=1e-9)
        # identities hold post-transform
        assert np.allclose(rotated[:-1] + a_rot, rotated[1:], atol=1e-12)
        assert np.allclose(d_rot[:-1] - d_rot[1:] - a_rot, 0.0, atol=1e-9)
        assert float(np.max(np.abs(d_rot[-1]))) == 0.0


# ---------------------------------------------------------------------------
# Unordered identity: A/B swap gives the same task digest
# ---------------------------------------------------------------------------


def test_endpoint_swap_gives_identical_task_digest():
    outcome = build_outcome()
    payload = outcome.task_payload
    swapped_payload = dict(payload)
    swapped_payload["endpoints"] = [payload["endpoints"][1],
                                    payload["endpoints"][0]]
    task_a = DualEndpointTask.from_payload(payload)
    task_b = DualEndpointTask.from_payload(swapped_payload)
    assert task_a.content_digest() == task_b.content_digest()
    assert outcome.gaps["task_digest"] == task_a.content_digest()


# ---------------------------------------------------------------------------
# a_max exceedance: flagged, never inserted
# ---------------------------------------------------------------------------


def test_a_max_exceedance_flagged_not_inserted():
    path = synthetic_ordered_path()
    # inject one huge per-atom step on the R side (frame 0 -> frame 1)
    path = path.copy()
    path[1, 2, 0] += 0.5
    outcome = build_outcome(path=path)
    assert outcome.status == "ok"
    exceed = outcome.gaps["items"]["a_max_exceedances"]
    # The spike is non-rigid, so chained rigid registration legitimately
    # smears it; both adjacent steps of the injected branch must be flagged,
    # and nothing may be inserted anywhere.
    total_flagged = exceed["A"]["count"] + exceed["B"]["count"]
    assert total_flagged >= 2
    for side in ("A", "B"):
        assert exceed[side]["inserted"] is False
        assert exceed[side]["a_max_angstrom"] == 0.1
        if exceed[side]["count"]:
            assert len(exceed[side]["step_indices"]) == exceed[side]["count"]
            assert exceed[side]["max_abs_per_atom"] > 0.1
    # no insertion happened: state count == frame count per branch
    assert outcome.gaps["items"]["branch_step_counts"] == {"A": 3, "B": 3}
    assert len(outcome.teacher.states) == 6
    # direct helper marks exactly the injected step (no registration here)
    actions = np.zeros((3, 2, 3))
    actions[1, 0] = [0.2, 0.0, 0.0]
    flags = flag_a_max_exceedances(actions, A_MAX_ANGSTROM)
    assert flags["step_indices"] == [1] and flags["count"] == 1
    assert flags["inserted"] is False


# ---------------------------------------------------------------------------
# Gaps registry structure + WP1 observation policy
# ---------------------------------------------------------------------------


def test_gaps_registry_structure_and_missing_observations():
    outcome = build_outcome()
    gaps = outcome.gaps
    assert gaps["reaction_group_id"] == "GRP_SYNTH0001"
    assert gaps["partition"] == "fit"
    assert gaps["status"] == "ok"
    items = gaps["items"]
    for key in ("missing_branches", "mapping_problems",
                "geometry_quality_issues", "branch_step_counts",
                "a_max_exceedances", "missing_observations"):
        assert key in items
    assert gaps["ts_rule"] == TS_RULE_AUTHORITY
    assert gaps["drift_rule"] == DRIFT_RULE
    assert gaps["bond_perception"] == "g1_p1_mapping_endpoint_bonds_v1"
    assert gaps["arc_length_metric_version"].endswith("PLACEHOLDER")
    # WP1: every state's observation is null + registered reason (xTB = WP2)
    for side, count in (("A", 3), ("B", 3)):
        entry = items["missing_observations"][side]
        assert entry["count"] == count
        assert entry["reasons"]["non_ts"] == "observation_not_exported"
        assert entry["reasons"]["ts"] == "teacher_frame_isolated"
        assert entry["xtb"] == "wp2_scope"
    for state in outcome.teacher.states:
        assert state.observation is None
        assert state.observation_missing_reason in (
            "observation_not_exported", "teacher_frame_isolated")


# ---------------------------------------------------------------------------
# Contract validation of both payloads
# ---------------------------------------------------------------------------


def test_both_records_validate_and_round_trip():
    outcome = build_outcome()
    validate_record(TASK_SCHEMA_VERSION, outcome.task_payload)
    validate_record(TEACHER_SCHEMA_VERSION, outcome.teacher_payload)
    task = DualEndpointTask.from_payload(outcome.task_payload)
    teacher = DualTSTeacher.from_payload(outcome.teacher_payload)
    assert task.content_digest() == outcome.diagnostics["task_digest"]
    assert teacher.content_digest() == outcome.diagnostics["teacher_digest"]
    # digests are stable across a reload of the serialized payloads
    assert DualEndpointTask.from_payload(outcome.task_payload).content_digest() \
        == task.content_digest()
    assert DualTSTeacher.from_payload(outcome.teacher_payload).content_digest() \
        == teacher.content_digest()


# ---------------------------------------------------------------------------
# Task↔teacher cross-record identity (WP2 checklist item)
# ---------------------------------------------------------------------------


def test_task_teacher_identity_mismatch_is_typed_rejection():
    outcome = build_outcome()
    payload = outcome.teacher_payload
    payload["states"][0]["atomic_numbers"][0] = 7  # O -> N corruption
    tampered = DualTSTeacher.from_payload(payload)  # state itself stays valid
    with pytest.raises(DualTSPreprocessError) as excinfo:
        assert_task_teacher_identity(outcome.task, tampered)
    assert excinfo.value.code == "TASK_TEACHER_IDENTITY_MISMATCH"
    # and the honest records pass the same check
    assert_task_teacher_identity(outcome.task, outcome.teacher)


# ---------------------------------------------------------------------------
# TS identification + covalent fallback perception
# ---------------------------------------------------------------------------


def test_ts_rule_authority_wins_and_max_energy_is_recorded_fallback():
    idx, rule = identify_ts_frame(synthetic_energies(), authority_ts_position=2)
    assert (idx, rule) == (2, TS_RULE_AUTHORITY)
    idx, rule = identify_ts_frame(synthetic_energies(), authority_ts_position=None)
    assert (idx, rule) == (2, TS_RULE_MAX_ENERGY)
    with pytest.raises(DualTSPreprocessError) as excinfo:
        identify_ts_frame(None, authority_ts_position=None)
    assert excinfo.value.code == "TS_FRAME_UNIDENTIFIED"


def test_build_falls_back_to_max_energy_ts_and_records_the_rule():
    outcome = build_outcome(ts_position=None)
    assert outcome.status == "ok"
    assert outcome.gaps["ts_rule"] == TS_RULE_MAX_ENERGY
    assert outcome.diagnostics["ts_position"] == 2.0


def test_covalent_fallback_bond_perception_is_deterministic():
    z = np.array([6, 6])
    close = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    far = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    g1 = endpoint_graph_from_geometry(z, close)
    g2 = endpoint_graph_from_geometry(z, close)
    assert g1.content_digest() == g2.content_digest()
    assert len(g1.bonds) == 1
    assert len(endpoint_graph_from_geometry(z, far).bonds) == 0
    assert BOND_PERCEPTION_COVALENT.startswith("covalent_radius")
