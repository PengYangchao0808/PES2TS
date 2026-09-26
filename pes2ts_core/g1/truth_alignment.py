"""P1.1/P1.2 geometry: rigid alignment, IRC layout, and candidate mapping.

Empirically established on the real archive (and re-verified per reaction at
run time, never assumed):

* IRC frame order is ``[TS, branch_1 (TS -> end A), branch_2 (near-TS -> end
  B)]``; the two branches are spliced where one consecutive-frame step is an
  outlier (``> splice_factor`` times the median step), and the stored
  ``ts_index`` constant is confirmed against ``ts.parquet`` instead of being
  trusted.
* TS/IRC atom rows follow the reaction map-number order (row ``i`` carries
  the atom with map ``i + 1``) on every sampled reaction; the solver
  therefore *proposes* the identity correspondence, verifies it with the
  element sequence, and falls back to a budget-capped element-preserving
  permutation search only when the sequence check fails.

The R/P side assignment re-uses the G1 candidate bijections (never a fresh
"first graph match"): every stored candidate combination -- including
permutations of identical-skeleton component geometries -- is scored by the
Kabsch residual of the TS geometry against the side coordinates ordered by
map number, then folded into equivalence classes one tolerance wide.
Buckets are absolute (``round(rmsd / tolerance)``), so candidate input order
can never change the classes or the selected representative.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from pes2ts_core.g1.truth_schema import (
    DEFAULT_MAX_COMBINATIONS,
    DEFAULT_RMSD_CLASS_TOLERANCE,
    DEFAULT_SPLICE_FACTOR,
    DEFAULT_TS_FRAME_TOLERANCE,
    ENDPOINT_FAIL,
    ENDPOINT_PASS,
    ENDPOINT_WEAK,
    LAYOUT_SINGLE_BRANCH,
    LAYOUT_TS_FIRST,
    LAYOUT_TS_LAST,
    LAYOUT_UNUSUAL,
    ORIENTATION_P_FIRST,
    ORIENTATION_R_FIRST,
    ORIENTATION_UNRESOLVED,
)

logger = logging.getLogger(__name__)

#: Quality flags recorded on the layout / endpoint analysis.
FLAG_NON_FINITE: Final[str] = "non_finite_coordinates"
FLAG_SINGLE_FRAME: Final[str] = "single_frame_trajectory"
FLAG_NO_SPLICE: Final[str] = "no_splice_outlier_found"
FLAG_TS_FRAME_UNMATCHED: Final[str] = "ts_frame_not_matched"
FLAG_TS_FRAME_INTERIOR: Final[str] = "ts_frame_interior"
FLAG_BRANCH_MISSING: Final[str] = "branch_missing"
FLAG_ENDPOINT_AMBIGUOUS: Final[str] = "endpoint_orientation_ambiguous"

#: Orientation decision margin (Angstrom of aligned RMSD difference).
ORIENTATION_MARGIN: Final[float] = 0.1


# ---------------------------------------------------------------------------
# Rigid alignment
# ---------------------------------------------------------------------------

def kabsch_rmsd(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    """Return the RMSD of *first* onto *second* after optimal rotation.

    Both inputs are ``(n, 3)`` coordinate arrays in a shared row
    correspondence.  The optimal right-handed rotation (Kabsch algorithm,
    reflection-corrected) and the centroids are removed; the result is the
    root-mean-square deviation over all coordinates.
    """
    centered_first = first - first.mean(axis=0)
    centered_second = second - second.mean(axis=0)
    covariance = centered_first.T @ centered_second
    left, _singular, right_t = np.linalg.svd(covariance)
    determinant = np.sign(np.linalg.det(right_t.T @ left.T))
    rotation = right_t.T @ np.diag([1.0, 1.0, determinant]) @ left.T
    rotated = (rotation @ centered_first.T).T
    return float(np.sqrt(((rotated - centered_second) ** 2).sum() / len(first)))


def direct_rmsd(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    """Return the plain RMSD without any realignment."""
    difference = first - second
    return float(np.sqrt((difference * difference).sum() / len(first)))


def all_finite(coordinates: NDArray[np.float64]) -> bool:
    """Return whether every coordinate value is finite."""
    return bool(np.isfinite(coordinates).all())


# ---------------------------------------------------------------------------
# IRC layout
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class IrcLayout:
    """Decomposition of one IRC trajectory into TS frame and branches.

    ``branch_a_indices``/``branch_b_indices`` list frame indices ordered from
    the TS-adjacent end to the branch terminus (so ``[-1]`` is the endpoint
    geometry of the branch); ``branch_b`` is ``None`` for single-branch or
    degenerate trajectories.  ``quality_flags`` carries every anomaly the
    caller must surface instead of silently trusting the layout.
    """

    n_frames: int
    n_atoms: int
    ts_frame_index: int
    ts_frame_rmsd: float
    layout: str
    splice_index: int | None
    branch_a_indices: tuple[int, ...]
    branch_b_indices: tuple[int, ...] | None
    quality_flags: tuple[str, ...] = field(default_factory=tuple)


def analyze_irc_layout(
    frames: NDArray[np.float64],
    ts_coordinates: NDArray[np.float64],
    *,
    splice_factor: float = DEFAULT_SPLICE_FACTOR,
    ts_tolerance: float = DEFAULT_TS_FRAME_TOLERANCE,
) -> IrcLayout:
    """Locate the TS frame and split the trajectory into its two branches.

    The TS frame is the frame closest to the quarantined TS geometry (the
    archive stores both in one Cartesian frame, so a direct RMSD suffices);
    the branch splice is the consecutive-frame step that exceeds
    ``splice_factor`` times the median step.  The documented ``ts_first``
    layout has the TS at frame 0 and branch B starting one step from it; the
    mirrored ``ts_last`` layout and unusual interiors are detected and
    flagged, never forced into an unsupported shape.
    """
    n_frames, n_atoms, _width = frames.shape
    flags: list[str] = []
    if not all_finite(frames) or not all_finite(ts_coordinates):
        flags.append(FLAG_NON_FINITE)
    if n_frames == 1:
        return IrcLayout(
            n_frames=n_frames, n_atoms=n_atoms, ts_frame_index=0,
            ts_frame_rmsd=direct_rmsd(frames[0], ts_coordinates),
            layout=LAYOUT_SINGLE_BRANCH, splice_index=None,
            branch_a_indices=(0,), branch_b_indices=None,
            quality_flags=(*flags, FLAG_SINGLE_FRAME),
        )
    per_atom = np.sqrt(((frames - ts_coordinates[None, :, :]) ** 2).sum(axis=2))
    distances = per_atom.mean(axis=1)
    # The archive stores most IRC trajectories in the TS's own Cartesian
    # frame, but a minority is rigidly rotated/translated; the identity
    # check must therefore align each frame before comparing (a direct RMSD
    # would misreport a rotated trajectory as "no TS frame").
    if distances.min() > ts_tolerance:
        distances = np.array([
            kabsch_rmsd(frame, ts_coordinates) for frame in frames
        ])
    ts_frame_index = int(np.nanargmin(distances))
    ts_frame_rmsd = float(distances[ts_frame_index])
    if ts_frame_rmsd > ts_tolerance:
        flags.append(FLAG_TS_FRAME_UNMATCHED)
    steps = np.sqrt(((frames[1:] - frames[:-1]) ** 2).sum(axis=2).mean(axis=1))
    finite_steps = steps[np.isfinite(steps)]
    splice = int(np.nanargmax(steps)) if len(finite_steps) else 0
    splice_found = bool(
        len(finite_steps)
        and steps[splice] > splice_factor * float(np.median(finite_steps))
        and 0 <= splice < n_frames - 1
    )
    if not splice_found:
        flags.append(FLAG_NO_SPLICE)
        if ts_frame_index == 0:
            return IrcLayout(
                n_frames=n_frames, n_atoms=n_atoms, ts_frame_index=0,
                ts_frame_rmsd=ts_frame_rmsd, layout=LAYOUT_SINGLE_BRANCH,
                splice_index=None, branch_a_indices=tuple(range(n_frames)),
                branch_b_indices=None, quality_flags=tuple(flags),
            )
        if ts_frame_index == n_frames - 1:
            return IrcLayout(
                n_frames=n_frames, n_atoms=n_atoms,
                ts_frame_index=ts_frame_index, ts_frame_rmsd=ts_frame_rmsd,
                layout=LAYOUT_SINGLE_BRANCH, splice_index=None,
                branch_a_indices=tuple(range(n_frames - 1, -1, -1)),
                branch_b_indices=None, quality_flags=tuple(flags),
            )
        flags.append(FLAG_TS_FRAME_INTERIOR)
        return IrcLayout(
            n_frames=n_frames, n_atoms=n_atoms, ts_frame_index=ts_frame_index,
            ts_frame_rmsd=ts_frame_rmsd, layout=LAYOUT_UNUSUAL,
            splice_index=None, branch_a_indices=tuple(range(n_frames)),
            branch_b_indices=None, quality_flags=tuple(flags),
        )
    if ts_frame_index == 0:
        return IrcLayout(
            n_frames=n_frames, n_atoms=n_atoms, ts_frame_index=0,
            ts_frame_rmsd=ts_frame_rmsd, layout=LAYOUT_TS_FIRST,
            splice_index=splice,
            branch_a_indices=tuple(range(0, splice + 1)),
            branch_b_indices=tuple(range(splice + 1, n_frames)),
            quality_flags=tuple(flags),
        )
    if ts_frame_index == n_frames - 1:
        return IrcLayout(
            n_frames=n_frames, n_atoms=n_atoms,
            ts_frame_index=ts_frame_index, ts_frame_rmsd=ts_frame_rmsd,
            layout=LAYOUT_TS_LAST, splice_index=splice,
            branch_a_indices=tuple(range(splice, -1, -1)),
            branch_b_indices=tuple(range(n_frames - 1, splice, -1)),
            quality_flags=tuple(flags),
        )
    flags.append(FLAG_TS_FRAME_INTERIOR)
    segment_first = tuple(range(0, splice + 1))
    segment_second = tuple(range(splice + 1, n_frames))
    orient_first = segment_first if segment_first[0] == ts_frame_index or abs(segment_first[0] - ts_frame_index) < abs(segment_second[0] - ts_frame_index) else tuple(reversed(segment_first))
    orient_second = segment_second if abs(segment_second[0] - ts_frame_index) < abs(segment_first[-1] - ts_frame_index) else tuple(reversed(segment_second))
    return IrcLayout(
        n_frames=n_frames, n_atoms=n_atoms, ts_frame_index=ts_frame_index,
        ts_frame_rmsd=ts_frame_rmsd, layout=LAYOUT_UNUSUAL, splice_index=splice,
        branch_a_indices=orient_first, branch_b_indices=orient_second,
        quality_flags=tuple(flags),
    )


@dataclass(frozen=True, slots=True)
class EndpointVerdict:
    """Branch-endpoint match against the R/P geometries ordered by map."""

    orientation: str
    endpoint_match: str
    rmsd_a_to_r: float
    rmsd_a_to_p: float
    rmsd_b_to_r: float | None
    rmsd_b_to_p: float | None
    quality_flags: tuple[str, ...] = field(default_factory=tuple)


def endpoint_verdict(
    layout: IrcLayout,
    frames: NDArray[np.float64],
    r_by_map: NDArray[np.float64],
    p_by_map: NDArray[np.float64],
    *,
    pass_rmsd: float,
    weak_rmsd: float,
    orientation_hint: str | None = None,
) -> EndpointVerdict:
    """Decide the trajectory orientation and the endpoint match quality.

    ``orientation`` names the side whose geometry the *first* branch
    terminates at.  The decision hierarchy follows the plan's
    graph-over-geometry rule: a decisive *orientation_hint* (the bond-pattern
    verdict from :func:`pes2ts_core.g1.irc_bond_evidence.bond_pattern_orientation`)
    is used verbatim, because whole-molecule aligned RMSD at IRC termini is
    dominated by conformer noise and demonstrably mislabels orientations;
    without a decisive hint the aligned-RMSD margins decide, and near-ties
    are reported as unresolved rather than guessed.  ``endpoint_match`` is
    always the worst per-side verdict over the branch endpoints
    (``pass`` / ``weak`` / ``fail`` against the configured thresholds), and
    every RMSD stays recorded as evidence regardless of who decided.
    """
    end_a = frames[layout.branch_a_indices[-1]]
    rmsd_a_r = kabsch_rmsd(end_a, r_by_map)
    rmsd_a_p = kabsch_rmsd(end_a, p_by_map)
    rmsd_b_r: float | None = None
    rmsd_b_p: float | None = None
    if layout.branch_b_indices:
        end_b = frames[layout.branch_b_indices[-1]]
        rmsd_b_r = kabsch_rmsd(end_b, r_by_map)
        rmsd_b_p = kabsch_rmsd(end_b, p_by_map)
    flags: list[str] = list(layout.quality_flags)
    decisive_hint = orientation_hint in (ORIENTATION_R_FIRST, ORIENTATION_P_FIRST)
    if decisive_hint:
        orientation = str(orientation_hint)
    elif layout.branch_b_indices is None:
        orientation = (
            ORIENTATION_R_FIRST if rmsd_a_r + ORIENTATION_MARGIN < rmsd_a_p
            else ORIENTATION_P_FIRST if rmsd_a_p + ORIENTATION_MARGIN < rmsd_a_r
            else ORIENTATION_UNRESOLVED
        )
        if orientation == ORIENTATION_UNRESOLVED:
            flags.append(FLAG_ENDPOINT_AMBIGUOUS)
    else:
        assert rmsd_b_r is not None and rmsd_b_p is not None
        a_r_first = rmsd_a_r + ORIENTATION_MARGIN < rmsd_a_p
        a_p_first = rmsd_a_p + ORIENTATION_MARGIN < rmsd_a_r
        if a_r_first:
            orientation = ORIENTATION_R_FIRST
        elif a_p_first:
            orientation = ORIENTATION_P_FIRST
        else:
            orientation = ORIENTATION_UNRESOLVED
            flags.append(FLAG_ENDPOINT_AMBIGUOUS)
    if layout.branch_b_indices is None:
        side_best = (rmsd_a_r, rmsd_a_p)
    else:
        assert rmsd_b_r is not None and rmsd_b_p is not None
        side_best = (min(rmsd_a_r, rmsd_b_r), min(rmsd_a_p, rmsd_b_p))
    verdicts = [
        ENDPOINT_PASS if value <= pass_rmsd
        else ENDPOINT_WEAK if value <= weak_rmsd
        else ENDPOINT_FAIL
        for value in side_best
    ]
    endpoint_match = (
        ENDPOINT_FAIL if ENDPOINT_FAIL in verdicts
        else ENDPOINT_WEAK if ENDPOINT_WEAK in verdicts
        else ENDPOINT_PASS
    )
    return EndpointVerdict(
        orientation=orientation,
        endpoint_match=endpoint_match,
        rmsd_a_to_r=rmsd_a_r,
        rmsd_a_to_p=rmsd_a_p,
        rmsd_b_to_r=rmsd_b_r,
        rmsd_b_to_p=rmsd_b_p,
        quality_flags=tuple(flags),
    )


# ---------------------------------------------------------------------------
# TS-row correspondence
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TsCorrespondence:
    """Row-of-TS/IRC assignment for every reaction map number."""

    row_of_map: tuple[int, ...]
    algorithm: str
    n_ambiguous_alternatives: int


def _greedy_distance_match(
    reference_by_map: NDArray[np.float64],
    ts_coordinates: NDArray[np.float64],
    elements_by_map: Sequence[str],
    ts_atomic_numbers: Sequence[int],
    symbol_of_z: Mapping[int, str],
    budget: int,
) -> tuple[tuple[int, ...] | None, int]:
    """Greedy element-preserving row matching by distance-matrix similarity.

    Fallback path for a TS whose atom rows are *not* in map order: each map
    (most connected first) claims the free TS row of the same element whose
    distances to already-claimed partners best match the reference geometry.
    Returns the permutation and how many alternative first assignments were
    within ``1e-9`` of the greedy pick (a coarse ambiguity signal); the
    permutation is ``None`` when no element-compatible assignment exists.
    """
    n = len(elements_by_map)
    free_rows: set[int] = set(range(n))
    row_of_map: dict[int, int] = {}
    reference_distance = np.linalg.norm(
        reference_by_map[:, None, :] - reference_by_map[None, :, :], axis=2
    )
    ts_distance = np.linalg.norm(
        ts_coordinates[:, None, :] - ts_coordinates[None, :, :], axis=2
    )
    element_rows: dict[str, list[int]] = {}
    for row, atomic_number in enumerate(ts_atomic_numbers):
        element_rows.setdefault(str(symbol_of_z.get(int(atomic_number), "")), []).append(row)
    order = sorted(range(n), key=lambda m: -int((reference_distance[m] > 0).sum()))
    ambiguous = 0
    for map_index in order:
        element = elements_by_map[map_index]
        candidates = [row for row in element_rows.get(element, []) if row in free_rows]
        if not candidates:
            return None, ambiguous
        scored: list[tuple[float, int]] = []
        for row in candidates:
            mismatch = 0.0
            for other_map, other_row in row_of_map.items():
                mismatch += abs(
                    reference_distance[map_index, other_map]
                    - ts_distance[row, other_row]
                )
            scored.append((mismatch, row))
        scored.sort()
        if len(scored) > 1 and scored[1][0] - scored[0][0] < 1e-9:
            ambiguous += 1
        row_of_map[map_index] = scored[0][1]
        free_rows.discard(scored[0][1])
        if len(row_of_map) > budget:
            return None, ambiguous
    if free_rows:
        return None, ambiguous
    return tuple(row_of_map[m] for m in range(n)), ambiguous


# ---------------------------------------------------------------------------
# Side candidate assignment
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BlockChoice:
    """One block's chosen geometry tag and its map-to-row bijection."""

    block_tag: str
    geometry_tag: str
    map_to_local: Mapping[int, int]


@dataclass(frozen=True, slots=True)
class SideSolution:
    """Candidate-combination resolution of one side against the TS geometry.

    ``rows_by_map`` is ordered by map number 1..n; each entry is the
    ``(block_tag, geometry_tag, local_row)`` the winning combination assigns.
    """

    n_combinations: int
    truncated: bool
    n_equivalence_classes: int
    selected_class_size: int
    selected_rmsd: float
    selected: tuple[BlockChoice, ...]
    rows_by_map: tuple[tuple[str, str, int], ...]
    bucket_histogram: tuple[tuple[int, int], ...]


def _block_candidates(block: Mapping[str, Any]) -> list[dict[int, int]]:
    """Return the block's stored bijections (first fills ``rows``).

    The G1 document stores ``candidates`` as ``[{map, local_index}, ...]``
    lists; the invariant is at least one entry (the one that filled
    ``rows``).  A block without stored candidates falls back to its ``rows``
    so a hand-trimmed document still resolves deterministically.
    """
    raw = block.get("candidates")
    parsed: list[dict[int, int]] = []
    if isinstance(raw, list) and raw:
        for candidate in raw:
            if isinstance(candidate, list) and candidate:
                parsed.append(
                    {int(pair["map"]): int(pair["local_index"]) for pair in candidate}
                )
    if not parsed:
        rows = block.get("rows")
        if isinstance(rows, list) and rows:
            parsed.append(
                {int(row["map"]): int(row["local_index"]) for row in rows}
            )
    return parsed


def _geometry_groups(
    blocks: Sequence[Mapping[str, Any]], component_smiles: Mapping[str, str]
) -> list[tuple[str, ...]]:
    """Group block tags whose inventory SMILES are literally identical.

    Identical SMILES (maps included) guarantee identical skeletons, so their
    geometries may be exchanged inside a group; anything else is kept fixed,
    a deliberately conservative rule (skeleton-equal but SMILES-different
    components never swap).
    """
    by_smiles: dict[str, list[str]] = {}
    for block in blocks:
        tag = str(block["tag"])
        by_smiles.setdefault(str(component_smiles.get(tag, tag)), []).append(tag)
    return [tuple(sorted(group)) for group in by_smiles.values()]


def solve_side_assignment(
    side_blocks: Sequence[Mapping[str, Any]],
    component_geometries: Mapping[str, Sequence[Sequence[float]]],
    component_smiles: Mapping[str, str],
    ts_by_map: NDArray[np.float64],
    *,
    max_combinations: int = DEFAULT_MAX_COMBINATIONS,
    rmsd_class_tolerance: float = DEFAULT_RMSD_CLASS_TOLERANCE,
) -> SideSolution | None:
    """Resolve one side's map-to-geometry assignment against the TS.

    Enumerates the product of (identical-SMILES geometry permutations) x
    (stored G1 candidate bijections) per block, scores every combination by
    the Kabsch RMSD between the TS geometry (rows already in map order) and
    the side coordinates ordered by map, folds combinations into absolute
    RMSD buckets (``round(rmsd / tolerance)``), and selects the best
    bucket's lexicographically smallest member.  Buckets are absolute, so
    candidate input order can never change the classes or the winner.  A
    total space larger than ``max_combinations`` marks the result
    ``truncated`` and only the lexicographic prefix of each dimension is
    enumerated; completeness is then explicitly unproven.  Returns ``None``
    when a block carries no usable bijection or no combination yields a
    contiguous 1..n map space compatible with the identity TS rows.
    """
    ordered_blocks = sorted(side_blocks, key=lambda block: int(block["index_base"]))
    candidate_lists = [_block_candidates(block) for block in ordered_blocks]
    if not candidate_lists or any(not candidates for candidates in candidate_lists):
        return None
    geometries = {
        tag: np.asarray(coordinates, dtype=np.float64)
        for tag, coordinates in component_geometries.items()
    }
    tags = [str(block["tag"]) for block in ordered_blocks]
    groups = _geometry_groups(ordered_blocks, component_smiles)
    group_members = {tag: group for group in groups for tag in group}
    # Every geometry assignment = one permutation per group, in lexicographic
    # order; block tags outside any multi-member group map to themselves.
    permutation_products = list(itertools.product(
        *(itertools.permutations(group) for group in groups)
    ))
    candidate_product = list(itertools.product(
        *(range(len(candidates)) for candidates in candidate_lists)
    ))
    n_total = len(permutation_products) * len(candidate_product)
    truncated = n_total > max_combinations
    permutation_cap = min(len(permutation_products), max_combinations)
    candidate_cap = max(1, min(len(candidate_product), max_combinations // permutation_cap))
    scored: list[tuple[float, tuple[tuple[str, int], ...], tuple[BlockChoice, ...]]] = []
    for permutation_index in range(permutation_cap):
        geometry_tag_of_block: dict[str, str] = {}
        for group, permutation in zip(groups, permutation_products[permutation_index], strict=True):
            for block_tag, geometry_tag in zip(
                sorted(tag for tag in tags if tag in group),
                permutation,
                strict=True,
            ):
                geometry_tag_of_block[block_tag] = geometry_tag
        for candidate_choice in candidate_product[:candidate_cap]:
            choice_key: list[tuple[str, int]] = []
            choices: list[BlockChoice] = []
            entries: list[tuple[int, str, str, int]] = []
            usable = True
            for block, candidates, candidate_index, block_tag in zip(
                ordered_blocks, candidate_lists, candidate_choice, tags, strict=True
            ):
                bijection = candidates[candidate_index]
                geometry_tag = geometry_tag_of_block.get(block_tag, block_tag)
                if geometry_tag not in geometries:
                    usable = False
                    break
                choice_key.append((geometry_tag, candidate_index))
                choices.append(BlockChoice(
                    block_tag=block_tag,
                    geometry_tag=geometry_tag,
                    map_to_local=bijection,
                ))
                entries.extend(
                    (map_number, block_tag, geometry_tag, local_row)
                    for map_number, local_row in bijection.items()
                )
            if not usable:
                continue
            entries.sort(key=lambda entry: entry[0])
            maps = [entry[0] for entry in entries]
            if maps != list(range(1, len(maps) + 1)):
                continue  # identity TS rows require the contiguous 1..n map space
            coordinates_by_map = np.vstack(
                [geometries[geometry_tag][local_row] for _m, _b, geometry_tag, local_row in entries]
            )
            if coordinates_by_map.shape != ts_by_map.shape:
                continue
            rmsd = kabsch_rmsd(ts_by_map, coordinates_by_map)
            scored.append((rmsd, tuple(choice_key), tuple(choices)))
    if not scored:
        return None
    scored.sort(key=lambda entry: (entry[0], entry[1]))
    best_rmsd = scored[0][0]
    buckets: dict[int, int] = {}
    for rmsd, _key, _choices in scored:
        bucket = int(round(rmsd / rmsd_class_tolerance))
        buckets[bucket] = buckets.get(bucket, 0) + 1
    best_bucket = min(buckets)
    selected_class = [
        entry for entry in scored
        if int(round(entry[0] / rmsd_class_tolerance)) == best_bucket
    ]
    winner = selected_class[0]
    rows_by_map = tuple(
        (block_tag, geometry_tag, local_row)
        for _map_number, block_tag, geometry_tag, local_row in sorted(
            (
                (map_number, choice.block_tag, choice.geometry_tag, local_row)
                for choice in winner[2]
                for map_number, local_row in sorted(choice.map_to_local.items())
            ),
            key=lambda entry: entry[0],
        )
    )
    return SideSolution(
        n_combinations=len(scored),
        truncated=truncated,
        n_equivalence_classes=len(buckets),
        selected_class_size=len(selected_class),
        selected_rmsd=best_rmsd,
        selected=winner[2],
        rows_by_map=rows_by_map,
        bucket_histogram=tuple(sorted(buckets.items())),
    )


def solve_ts_correspondence(
    elements_by_map: Sequence[str],
    ts_atomic_numbers: Sequence[int],
    symbol_of_z: Mapping[int, str],
    *,
    reference_by_map: NDArray[np.float64] | None = None,
    ts_coordinates: NDArray[np.float64] | None = None,
    permutation_budget: int = 64,
) -> TsCorrespondence | None:
    """Return the TS-row correspondence for every map number.

    The identity ``row = map - 1`` is returned when the TS element sequence
    matches the map-ordered elements exactly (the archive's documented and
    sampled invariant).  Otherwise, when a reference geometry is supplied,
    a budget-capped element-preserving greedy distance-matrix search runs;
    its coarse ambiguity count is recorded for the caller to judge.  ``None``
    means no element-compatible correspondence exists at all.
    """
    n = len(elements_by_map)
    if len(ts_atomic_numbers) != n:
        return None
    ts_elements = [str(symbol_of_z.get(int(z), "")) for z in ts_atomic_numbers]
    if ts_elements == list(elements_by_map):
        return TsCorrespondence(
            row_of_map=tuple(range(n)), algorithm="identity",
            n_ambiguous_alternatives=0,
        )
    if reference_by_map is None or ts_coordinates is None:
        return None
    permutation, ambiguous = _greedy_distance_match(
        reference_by_map, ts_coordinates, elements_by_map, ts_atomic_numbers,
        symbol_of_z, permutation_budget,
    )
    if permutation is None:
        return None
    return TsCorrespondence(
        row_of_map=permutation, algorithm="greedy_distance",
        n_ambiguous_alternatives=ambiguous,
    )


__all__ = [
    "EndpointVerdict",
    "IrcLayout",
    "SideSolution",
    "TsCorrespondence",
    "FLAG_BRANCH_MISSING",
    "FLAG_ENDPOINT_AMBIGUOUS",
    "FLAG_NO_SPLICE",
    "FLAG_NON_FINITE",
    "FLAG_SINGLE_FRAME",
    "FLAG_TS_FRAME_INTERIOR",
    "FLAG_TS_FRAME_UNMATCHED",
    "all_finite",
    "analyze_irc_layout",
    "direct_rmsd",
    "endpoint_verdict",
    "kabsch_rmsd",
    "solve_side_assignment",
    "solve_ts_correspondence",
]
