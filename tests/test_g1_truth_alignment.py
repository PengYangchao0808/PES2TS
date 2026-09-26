"""Geometry solver fixtures: Kabsch, IRC layout, TS rows, side candidates."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from pes2ts_core.g1.truth_alignment import (
    FLAG_ENDPOINT_AMBIGUOUS,
    FLAG_NO_SPLICE,
    FLAG_SINGLE_FRAME,
    FLAG_TS_FRAME_UNMATCHED,
    LAYOUT_SINGLE_BRANCH,
    LAYOUT_TS_FIRST,
    LAYOUT_TS_LAST,
    LAYOUT_UNUSUAL,
    analyze_irc_layout,
    endpoint_verdict,
    kabsch_rmsd,
    solve_side_assignment,
    solve_ts_correspondence,
)

SYMBOLS = {1: "H", 8: "O", 9: "F", 17: "Cl", 6: "C"}


def _chain(n: int, step: float = 1.5) -> NDArray[np.float64]:
    coordinates = np.zeros((n, 3), dtype=np.float64)
    coordinates[:, 0] = np.arange(n) * step
    return coordinates


def _trajectory(reactant: NDArray[np.float64], product: NDArray[np.float64], ts: NDArray[np.float64], *, frames_per_branch: int = 8) -> NDArray[np.float64]:
    branch_a = np.linspace(0.0, 1.0, frames_per_branch)[:, None, None] * (reactant - ts) + ts
    branch_b = np.linspace(0.1, 1.0, frames_per_branch)[:, None, None] * (product - ts) + ts
    return np.vstack([branch_a, branch_b])


def test_kabsch_is_rotation_and_translation_invariant() -> None:
    base = _chain(4)
    theta = 1.1
    rotation = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    moved = (base - base.mean(axis=0)) @ rotation.T + np.array([4.0, -7.0, 2.0])
    assert kabsch_rmsd(base, moved) == pytest.approx(0.0, abs=1e-9)


def test_kabsch_penalizes_permutation() -> None:
    base = _chain(4)
    permuted = base[[1, 0, 3, 2]]
    assert kabsch_rmsd(base, permuted) > 0.1


def test_layout_ts_first_dual_branch_is_detected() -> None:
    reactant = _chain(3)
    product = _chain(3, step=1.7)[::-1].copy()
    ts = (reactant + product) / 2
    frames = _trajectory(reactant, product, ts, frames_per_branch=8)
    layout = analyze_irc_layout(frames, ts)
    assert layout.layout == LAYOUT_TS_FIRST
    assert layout.ts_frame_index == 0
    assert layout.splice_index == 7
    assert layout.branch_a_indices[-1] == 7
    assert layout.branch_b_indices == tuple(range(8, 16))
    assert layout.quality_flags == ()


def test_layout_ts_last_is_detected_with_reversed_branches() -> None:
    reactant = _chain(3)
    product = _chain(3, step=1.7)[::-1].copy()
    ts = (reactant + product) / 2
    frames = _trajectory(reactant, product, ts, frames_per_branch=8)[::-1].copy()
    layout = analyze_irc_layout(frames, ts)
    assert layout.layout == LAYOUT_TS_LAST
    assert layout.ts_frame_index == len(frames) - 1
    # Every branch still runs from the TS end to its terminus.
    assert layout.branch_a_indices[0] == layout.splice_index
    assert layout.branch_b_indices[0] == layout.ts_frame_index


def test_layout_single_frame_trajectory() -> None:
    ts = _chain(3)
    layout = analyze_irc_layout(ts[None, :, :].copy(), ts)
    assert FLAG_SINGLE_FRAME in layout.quality_flags


def test_layout_without_splice_is_single_branch() -> None:
    reactant = _chain(3)
    ts = reactant + 0.05
    frames = np.linspace(0.0, 1.0, 10)[:, None, None] * (reactant - ts) + ts
    layout = analyze_irc_layout(frames, ts)
    assert layout.layout == LAYOUT_SINGLE_BRANCH
    assert FLAG_NO_SPLICE in layout.quality_flags


def test_layout_flags_unmatched_ts_geometry() -> None:
    reactant = _chain(3)
    product = _chain(3, step=1.7)[::-1].copy()
    ts = (reactant + product) / 2
    frames = _trajectory(reactant, product, ts, frames_per_branch=8)
    layout = analyze_irc_layout(frames, _chain(3, step=3.0))
    assert FLAG_TS_FRAME_UNMATCHED in layout.quality_flags


def test_layout_recovers_ts_frame_from_rigidly_rotated_trajectory() -> None:
    # A minority of real archive trajectories are stored in a different
    # rigid frame than ts.parquet; the frame search must align before
    # comparing, and the found TS frame stays at index 0.
    reactant = _chain(3)
    product = _chain(3, step=1.7)[::-1].copy()
    ts = (reactant + product) / 2
    frames = _trajectory(reactant, product, ts, frames_per_branch=8)
    theta = 0.9
    rotation = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    rotated = (frames - frames.mean(axis=1, keepdims=True)) @ rotation.T + np.array([8.0, -4.0, 6.0])
    layout = analyze_irc_layout(rotated, ts)
    assert FLAG_TS_FRAME_UNMATCHED not in layout.quality_flags
    assert layout.ts_frame_index == 0
    assert layout.ts_frame_rmsd < 1e-6


def test_layout_interior_ts_frame_is_unusual() -> None:
    reactant = _chain(3)
    ts = reactant + 0.05
    frames = np.linspace(0.0, 1.0, 10)[:, None, None] * (reactant - ts) + ts
    shifted = np.roll(frames, 3, axis=0).copy()  # TS-like frame now interior
    layout = analyze_irc_layout(shifted, ts)
    assert layout.layout == LAYOUT_UNUSUAL


def test_endpoint_verdict_orientations_and_quality() -> None:
    reactant = _chain(3)
    product = _chain(3, step=1.7)[::-1].copy()
    ts = (reactant + product) / 2
    frames = _trajectory(reactant, product, ts, frames_per_branch=8)
    layout = analyze_irc_layout(frames, ts)
    verdict = endpoint_verdict(
        layout, frames, reactant, product, pass_rmsd=0.3, weak_rmsd=0.8,
    )
    assert verdict.orientation == "R_first"
    assert verdict.endpoint_match == "pass"
    swapped = endpoint_verdict(
        layout, frames, product, reactant, pass_rmsd=0.3, weak_rmsd=0.8,
    )
    assert swapped.orientation == "P_first"
    far = endpoint_verdict(
        layout, frames, reactant * 3.0, product * 3.0, pass_rmsd=0.3, weak_rmsd=0.8,
    )
    assert far.endpoint_match == "fail"
    # A decisive bond-pattern hint overrides the RMSD label (graph structure
    # outranks conformer-noisy geometry) while every RMSD stays recorded and
    # no ambiguity flag is raised.
    hinted = endpoint_verdict(
        layout, frames, reactant, product, pass_rmsd=0.3, weak_rmsd=0.8,
        orientation_hint="P_first",
    )
    assert hinted.orientation == "P_first"
    assert hinted.rmsd_a_to_r == verdict.rmsd_a_to_r
    assert hinted.endpoint_match == verdict.endpoint_match
    assert FLAG_ENDPOINT_AMBIGUOUS not in hinted.quality_flags


def test_ts_correspondence_identity_and_greedy_fallback() -> None:
    identity = solve_ts_correspondence(["O", "H", "H"], [8, 1, 1], SYMBOLS)
    assert identity is not None and identity.algorithm == "identity"
    reference = _chain(3)
    ts_coordinates = reference[::-1].copy()
    greedy = solve_ts_correspondence(
        ["O", "H", "H"], [1, 1, 8], SYMBOLS,
        reference_by_map=reference, ts_coordinates=ts_coordinates,
    )
    assert greedy is not None
    assert greedy.algorithm == "greedy_distance"
    assert list(greedy.row_of_map) == [2, 1, 0]
    mismatch = solve_ts_correspondence(["O", "H"], [8, 8], SYMBOLS)
    assert mismatch is None


def _block(tag: str, base: int, candidates: list[list[dict[str, int]]]) -> dict[str, object]:
    return {
        "tag": tag, "index_base": base, "n_atoms": len(candidates[0]),
        "candidates": [
            [{"map": pair["map"], "local_index": pair["local_index"]} for pair in candidate]
            for candidate in candidates
        ],
    }


def _bijection(pairs: dict[int, int]) -> list[dict[str, int]]:
    return [{"map": m, "local_index": row} for m, row in sorted(pairs.items())]


def test_side_assignment_picks_unique_best_candidate() -> None:
    geometry = np.array([
        [0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.9, 0.1, 0.3], [4.4, 0.0, 0.1],
    ])
    ts = geometry + 0.05
    identity = _bijection({1: 0, 2: 1, 3: 2, 4: 3})
    swapped = _bijection({1: 0, 2: 1, 3: 3, 4: 2})
    blocks = [_block("R0", 0, [identity, swapped])]
    solution = solve_side_assignment(
        blocks, {"R0": geometry}, {"R0": "[X:1]"}, ts,
    )
    assert solution is not None
    assert solution.n_combinations == 2
    assert solution.selected_class_size == 1
    assert solution.rows_by_map == (("R0", "R0", 0), ("R0", "R0", 1), ("R0", "R0", 2), ("R0", "R0", 3))


def test_side_assignment_collapses_symmetric_ties() -> None:
    geometry = np.array([
        [0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.2, 0.0], [3.0, 0.2, 0.0],
    ])
    identity = _bijection({1: 0, 2: 1, 3: 2, 4: 3})
    swapped = _bijection({1: 0, 2: 1, 3: 3, 4: 2})
    blocks = [_block("R0", 0, [identity, swapped])]
    solution = solve_side_assignment(
        blocks, {"R0": geometry}, {"R0": "[X:1]"}, geometry,
    )
    assert solution is not None
    assert solution.selected_class_size == 2
    assert solution.n_equivalence_classes == 1


def test_side_assignment_candidate_order_never_changes_the_winner() -> None:
    geometry = np.array([
        [0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.9, 0.1, 0.3], [4.4, 0.0, 0.1],
    ])
    ts = geometry + 0.05
    identity = _bijection({1: 0, 2: 1, 3: 2, 4: 3})
    swapped = _bijection({1: 0, 2: 1, 3: 3, 4: 2})
    forward = solve_side_assignment(
        [_block("R0", 0, [identity, swapped])], {"R0": geometry}, {"R0": "s"}, ts,
    )
    backward = solve_side_assignment(
        [_block("R0", 0, [swapped, identity])], {"R0": geometry}, {"R0": "s"}, ts,
    )
    assert forward is not None and backward is not None
    assert forward.rows_by_map == backward.rows_by_map
    assert forward.selected_rmsd == backward.selected_rmsd
    assert forward.bucket_histogram == backward.bucket_histogram


def test_side_assignment_marks_truncated_enumeration() -> None:
    geometry = _chain(4)
    identity = _bijection({1: 0, 2: 1, 3: 2, 4: 3})
    alternative = _bijection({1: 1, 2: 0, 3: 2, 4: 3})
    blocks = [_block("R0", 0, [identity, alternative])]
    solution = solve_side_assignment(
        blocks, {"R0": geometry}, {"R0": "s"}, geometry, max_combinations=1,
    )
    assert solution is not None
    assert solution.truncated is True
    assert solution.n_combinations == 1


def test_side_assignment_permutes_identical_smiles_geometries() -> None:
    left = _chain(3)
    right = _chain(3) + np.array([0.0, 5.0, 0.0])
    ts = left + 0.05
    block_left = _block("R0", 0, [_bijection({1: 0, 2: 1, 3: 2})])
    block_right = _block("R1", 3, [_bijection({4: 0, 5: 1, 6: 2})])
    # Stored tag pairing is deliberately wrong (R0 owns the far geometry);
    # only the identical-SMILES geometry permutation rescues the assignment.
    solution = solve_side_assignment(
        [block_right, block_left],
        {"R0": right, "R1": left},
        {"R0": "same", "R1": "same"},
        np.vstack([ts, right + 0.05]),
    )
    assert solution is not None
    assert solution.selected_rmsd < 0.2
