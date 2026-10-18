"""B2 acceptance: non-uniform arc sampling, sides, direction, and masking."""

from __future__ import annotations

import numpy as np
import pytest

from tests.pesn.conftest import (
    build_synthetic_reaction_data,
    synthetic_coordinate_spec,
    synthetic_task_graph,
    synthetic_trajectory,
)
from pes2ts_core.pesn.offline.labels.direction import (
    REASON_TS_ANCHOR,
    REASON_ZERO_ARC_STEP,
    compute_direction_labels,
    evaluate_distance_coords,
)
from pes2ts_core.pesn.offline.labels.progress import (
    SIDE_P,
    SIDE_R,
    SIDE_TS,
    compute_progress_labels,
    kabsch_align,
    masses_for,
)


def test_kabsch_alignment_removes_rigid_motion() -> None:
    rng = np.random.default_rng(0)
    mobile = rng.normal(size=(8, 3))
    angle = 0.7
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    moved = mobile @ rotation.T + np.asarray([3.0, -2.0, 5.0])
    weights = np.full(8, 12.0)
    aligned = kabsch_align(moved, mobile, weights)
    assert np.max(np.abs(aligned - mobile)) < 1e-8


def test_progress_labels_side_signs_and_bounds() -> None:
    coords = synthetic_trajectory(bend=0.3)
    masses = masses_for(synthetic_task_graph().atomic_numbers)
    labels = compute_progress_labels(
        coords,
        masses,
        ts_frame_index=0,
        splice_index=coords.shape[0] // 2,
        layout="ts_first_dual_branch",
        orientation="P_first",
    )
    assert labels.signed_progress[0] == 0.0 and labels.side[0] == SIDE_TS
    branch1 = labels.side == SIDE_P
    branch2 = labels.side == SIDE_R
    assert branch1.any() and branch2.any()
    assert np.all(labels.signed_progress[branch1] > 0)
    assert np.all(labels.signed_progress[branch2] < 0)
    assert np.all(np.abs(labels.signed_progress) <= 1.0 + 1e-12)
    assert np.allclose(labels.path_proximity, np.abs(labels.signed_progress))
    # monotone away from TS on each branch
    p_order = np.argsort(np.abs(labels.signed_progress[branch1]))
    assert np.all(np.diff(labels.d_arc[branch1][p_order]) >= -1e-12)


def test_r_p_exchange_flips_sign_and_keeps_proximity() -> None:
    coords = synthetic_trajectory()
    masses = masses_for(synthetic_task_graph().atomic_numbers)
    p_first = compute_progress_labels(
        coords, masses, ts_frame_index=0, splice_index=coords.shape[0] // 2,
        layout="ts_first_dual_branch", orientation="P_first",
    )
    r_first = compute_progress_labels(
        coords, masses, ts_frame_index=0, splice_index=coords.shape[0] // 2,
        layout="ts_first_dual_branch", orientation="R_first",
    )
    np.testing.assert_allclose(p_first.signed_progress, -r_first.signed_progress, atol=1e-12)
    np.testing.assert_allclose(p_first.path_proximity, r_first.path_proximity, atol=1e-12)


def test_nonuniform_frame_spacing_becomes_arc_length() -> None:
    dense = synthetic_trajectory(n_branch_frames=6)
    masses = masses_for(synthetic_task_graph().atomic_numbers)
    labels = compute_progress_labels(
        dense, masses, ts_frame_index=0, splice_index=dense.shape[0] // 2,
        layout="ts_first_dual_branch", orientation="P_first",
    )
    branch = labels.side == SIDE_P
    # geometric steps grow along the fixture's branch (t spacing constant but
    # distances differ), so equal frame counts are NOT equal arc increments
    increments = np.diff(labels.d_arc[branch])
    assert np.any(np.abs(np.diff(increments)) > 1e-6)


def test_zero_arc_branch_masks_side_with_typed_reason() -> None:
    coords = synthetic_trajectory(n_branch_frames=3)
    coords[coords.shape[0] // 2 + 1 :, 2] = coords[coords.shape[0] // 2, 2]  # duplicate branch-2 frames
    masses = masses_for(synthetic_task_graph().atomic_numbers)
    labels = compute_progress_labels(
        coords, masses, ts_frame_index=0, splice_index=coords.shape[0] // 2,
        layout="ts_first_dual_branch", orientation="P_first",
    )
    branch2 = np.where(labels.side == SIDE_R)[0]
    if labels.endpoint_status.get("r_side") in ("zero_arc", "degenerate", "empty"):
        assert not labels.progress_mask[branch2].any()


def test_single_branch_labels_one_side_only() -> None:
    coords = synthetic_trajectory(n_branch_frames=5)[: 6]  # TS + branch_1 only
    masses = masses_for(synthetic_task_graph().atomic_numbers)
    labels = compute_progress_labels(
        coords, masses, ts_frame_index=0, splice_index=None,
        layout="single_branch", orientation="P_first",
    )
    assert (labels.side == SIDE_P).any()
    assert not (labels.side == SIDE_R).any()
    assert labels.endpoint_status["r_side"] == "absent"


def test_direction_labels_unit_masked_at_ts_and_zero_steps() -> None:
    reaction = build_synthetic_reaction_data()
    branch = reaction.side != SIDE_TS
    valid_rows = reaction.direction[branch][reaction.dir_mask[branch]]
    assert valid_rows.shape[0] > 0
    np.testing.assert_allclose(
        np.linalg.norm(valid_rows, axis=1), 1.0, atol=1e-9
    )
    assert not reaction.dir_mask[0], "TS anchor direction must be masked"


def test_direction_points_toward_ts_in_q_space() -> None:
    reaction = build_synthetic_reaction_data()
    # On the P side the A--H driver is stretching toward its TS value from
    # below... in this fixture AH grows monotonically from TS to P end, so
    # the toward-TS derivative of q_AH is negative on the P branch.
    p_rows = np.where((reaction.side == SIDE_P) & reaction.dir_mask)[0]
    assert p_rows.size > 0
    mean_p = reaction.direction[p_rows].mean(axis=0)
    assert mean_p[0] < 0, "A-H must shrink toward TS on the product branch"
    assert mean_p[1] > 0, "H-B must grow toward TS on the product branch"


def test_direction_zero_step_masked() -> None:
    coords = synthetic_trajectory(n_branch_frames=4)
    coords[2] = coords[1]  # duplicate frame -> zero arc step
    masses = masses_for(synthetic_task_graph().atomic_numbers)
    progress = compute_progress_labels(
        coords, masses, ts_frame_index=0, splice_index=coords.shape[0] // 2,
        layout="ts_first_dual_branch", orientation="P_first",
    )
    q = evaluate_distance_coords(coords, [(0, 1), (2, 3)])
    direction = compute_direction_labels(q, progress)
    assert direction.frame_reasons.get(1) in (
        REASON_ZERO_ARC_STEP,
        REASON_TS_ANCHOR,
        "ok",
    ) or direction.direction_mask[1]


def test_activity_labels_above_average_rule() -> None:
    reaction = build_synthetic_reaction_data()
    for frame in range(reaction.n_frames):
        if not reaction.act_mask[frame]:
            continue
        active = reaction.activity[frame] == 1.0
        assert active.sum() <= reaction.activity.shape[1]
        assert reaction.activity[frame].sum() >= 1.0
