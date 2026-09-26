"""P1.3 IRC path evidence for mapped bond events.

The discrete formed/broken/order-changed/hydrogen-migration events are
*defined* by the mapped R/P graph difference (they arrive here already
computed in map space).  This module adds the independent path-level check
the plan requires: for every changed atom pair, the pair distance is traced
along the IRC branch that terminates at the side the bond belongs to, and a
typed verdict -- ``support`` / ``weak`` / ``mismatch`` -- records whether
the geometry actually moves the way the graph edit claims.  A single-frame
distance never overrides the graph: verdicts annotate, they never rewrite
the event lists, and contradictions surface as ``mismatch`` instead of
being silently absorbed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from pes2ts_core.g1.index_map import COVALENT_RADII
from pes2ts_core.g1.truth_alignment import IrcLayout
from pes2ts_core.g1.truth_schema import (
    ENDPOINT_WEAK,
    LAYOUT_SINGLE_BRANCH,
    ORIENTATION_P_FIRST,
    ORIENTATION_R_FIRST,
    ORIENTATION_UNRESOLVED,
    SUPPORT_MISMATCH,
    SUPPORT_SUPPORT,
    SUPPORT_WEAK,
)

#: Trend thresholds (Angstrom difference between TS and branch end).
TREND_MARGIN: Final[float] = 0.05
#: Maximum progress-fraction spread for a "synchronous" label.
SYNCHRONY_MARGIN: Final[float] = 0.25
#: Synchrony label when fewer than two events carry a progress fraction.
SYNCHRONY_SINGLE: Final[str] = "single_event"
SYNCHRONY_SYNCHRONOUS: Final[str] = "synchronous"
SYNCHRONY_ASYNCHRONOUS: Final[str] = "asynchronous"
#: Quality flags of the event validation.
FLAG_BRANCH_MISSING: Final[str] = "branch_missing_for_event"
FLAG_ORIENTATION_UNRESOLVED: Final[str] = "orientation_unresolved"
FLAG_NON_FINITE_CURVE: Final[str] = "non_finite_curve"


def _atomic_number_of(symbol: str) -> int | None:
    """Return Z of *symbol* via the first match in the shared symbol table."""
    from pes2ts_core.g0.rp_checks import ELEMENT_SYMBOLS

    for index, element in enumerate(ELEMENT_SYMBOLS, start=1):
        if element == symbol:
            return index
    return None


def bonded_cutoff(
    first_symbol: str, second_symbol: str, *, tolerance: float
) -> float:
    """Return the covalent-radius-sum distance cutoff of an atom pair."""
    first = _atomic_number_of(first_symbol) or 1
    second = _atomic_number_of(second_symbol) or 1
    return COVALENT_RADII.get(first, 1.0) + COVALENT_RADII.get(second, 1.0) + tolerance


def terminus_bond_scores(
    r_bonds: Sequence[Sequence[float]],
    p_bonds: Sequence[Sequence[float]],
    elements_by_map: Mapping[int, str],
    terminus: NDArray[np.float64],
    row_of_map: Sequence[int],
    *,
    tolerance: float,
) -> dict[str, tuple[int, int]]:
    """Return per-side ``(satisfied, total)`` bond counts at one terminus.

    Local pair distances are immune to the conformer noise that dominates
    whole-molecule RMSD at IRC termini, and the identity map-order rows make
    the check permutation-safe.  At the R-side terminus the R bond set is
    fully satisfied while the P set misses exactly the reaction's edits.
    """
    scores: dict[str, tuple[int, int]] = {}
    for side, bonds in (("r", r_bonds), ("p", p_bonds)):
        satisfied = 0
        for bond in bonds:
            first, second = int(bond[0]), int(bond[1])
            distance = float(np.linalg.norm(
                terminus[row_of_map[first - 1]] - terminus[row_of_map[second - 1]]
            ))
            cutoff = bonded_cutoff(
                str(elements_by_map.get(first, "H")),
                str(elements_by_map.get(second, "H")),
                tolerance=tolerance,
            )
            if distance <= cutoff:
                satisfied += 1
        scores[side] = (satisfied, len(bonds))
    return scores


def bond_pattern_orientation(
    layout: IrcLayout,
    frames: NDArray[np.float64],
    r_bonds: Sequence[Sequence[float]],
    p_bonds: Sequence[Sequence[float]],
    elements_by_map: Mapping[int, str],
    row_of_map: Sequence[int],
    *,
    tolerance: float,
    margin: float = 0.5,
) -> tuple[str, dict[str, dict[str, tuple[int, int]]]]:
    """Decide which branch terminates at which side from the bond patterns.

    Graph structure decides, geometry only scores: each terminus's
    R-likeness is ``p_missing - r_missing`` (positive when the R bond set is
    satisfied more completely than the P set), and the terminus with the
    larger R-likeness is the R-side terminus.  The mapped graph difference
    guarantees at least one bond set differs on reactions with formed or
    broken events, so the pattern is decisive exactly where the branch
    assignment matters; pure order-change reactions (identical bond sets on
    both sides) return ``unresolved`` and fall back to the RMSD decision.
    """
    if not r_bonds or not p_bonds or layout.branch_b_indices is None:
        return ORIENTATION_UNRESOLVED, {}
    terminus_a = frames[layout.branch_a_indices[-1]]
    terminus_b = frames[layout.branch_b_indices[-1]]
    scores_a = terminus_bond_scores(
        r_bonds, p_bonds, elements_by_map, terminus_a, row_of_map,
        tolerance=tolerance,
    )
    scores_b = terminus_bond_scores(
        r_bonds, p_bonds, elements_by_map, terminus_b, row_of_map,
        tolerance=tolerance,
    )

    def r_likeness(scores: Mapping[str, tuple[int, int]]) -> int:
        r_satisfied, r_total = scores["r"]
        p_satisfied, p_total = scores["p"]
        return (p_total - p_satisfied) - (r_total - r_satisfied)

    likeness_a = r_likeness(scores_a)
    likeness_b = r_likeness(scores_b)
    if likeness_a > likeness_b + margin:
        orientation = ORIENTATION_R_FIRST
    elif likeness_b > likeness_a + margin:
        orientation = ORIENTATION_P_FIRST
    else:
        orientation = ORIENTATION_UNRESOLVED
    return orientation, {"branch_a": scores_a, "branch_b": scores_b}


@dataclass(frozen=True, slots=True)
class EventSupport:
    """One event's IRC distance evidence."""

    kind: str
    maps: tuple[int, int]
    branch: str
    d_r_end: float | None
    d_ts: float | None
    d_p_end: float | None
    d_min: float | None
    d_max: float | None
    trend: str
    verdict: str
    progress_fraction: float | None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class EventValidation:
    """Every event verdict plus the aggregate quality summary."""

    event_support: tuple[EventSupport, ...]
    n_support: int
    n_weak: int
    n_mismatch: int
    synchrony: str
    max_progress_delta: float | None
    quality_flags: tuple[str, ...] = field(default_factory=tuple)


def _branch_for(
    layout: IrcLayout, side: str, orientation: str, flags: list[str]
) -> tuple[int, ...] | None:
    """Return the frame indices (TS-adjacent -> far end) of *side*'s branch."""
    if layout.branch_b_indices is None:
        if layout.layout == LAYOUT_SINGLE_BRANCH:
            flags.append(FLAG_BRANCH_MISSING)
            return None
        flags.append(FLAG_BRANCH_MISSING)
        return None
    first_side = (
        ORIENTATION_R_FIRST if orientation == ORIENTATION_R_FIRST
        else ORIENTATION_P_FIRST if orientation == ORIENTATION_P_FIRST
        else None
    )
    if first_side is None:
        flags.append(FLAG_ORIENTATION_UNRESOLVED)
        # Both branches exist; fall back to a side guess by index order is
        # forbidden -- use both ends as evidence and let verdicts stay weak.
        return layout.branch_a_indices if side == "r" else layout.branch_b_indices
    if side == "r":
        return (
            layout.branch_a_indices if first_side == ORIENTATION_R_FIRST
            else layout.branch_b_indices
        )
    return (
        layout.branch_a_indices if first_side == ORIENTATION_P_FIRST
        else layout.branch_b_indices
    )


def _curve(
    frames: NDArray[np.float64],
    indices: Sequence[int],
    row_i: int,
    row_j: int,
) -> NDArray[np.float64] | None:
    """Return the pair distance along one branch (``None`` on non-finite)."""
    values = np.linalg.norm(
        frames[list(indices), row_i, :] - frames[list(indices), row_j, :], axis=1
    )
    if not np.isfinite(values).all():
        return None
    return values


def _trend(delta: float) -> str:
    """Classify the branch-end minus TS distance difference."""
    if delta < -TREND_MARGIN:
        return "decreasing"
    if delta > TREND_MARGIN:
        return "increasing"
    return "flat"


def _progress(curve: NDArray[np.float64]) -> float | None:
    """Fraction along the branch where the curve crosses its TS/end midpoint."""
    midpoint = (curve[0] + curve[-1]) / 2.0
    index = int(np.argmin(np.abs(curve - midpoint)))
    return index / max(len(curve) - 1, 1)


def _verdict_event(
    kind: str,
    maps: tuple[int, int],
    elements: Mapping[int, str],
    frames: NDArray[np.float64],
    layout: IrcLayout,
    row_of_map: Sequence[int],
    orientation: str,
    *,
    tolerance: float,
    flags: list[str],
) -> EventSupport:
    """Validate one formed/broken/order_changed event on its branches."""
    side = "p" if kind == "formed" else "r"
    opposite = "r" if side == "p" else "p"
    branch = _branch_for(layout, side, orientation, flags)
    opposite_branch = _branch_for(layout, opposite, orientation, flags)
    row_i, row_j = row_of_map[maps[0] - 1], row_of_map[maps[1] - 1]
    d_ts = float(np.linalg.norm(
        frames[layout.ts_frame_index, row_i] - frames[layout.ts_frame_index, row_j]
    ))
    d_side: float | None = None
    d_opposite: float | None = None
    curve: NDArray[np.float64] | None = None
    progress: float | None = None
    if branch is not None:
        curve = _curve(frames, branch, row_i, row_j)
        if curve is not None:
            d_side = float(curve[-1])
            progress = _progress(curve)
    if opposite_branch is not None:
        opposite_curve = _curve(frames, opposite_branch, row_i, row_j)
        if opposite_curve is not None:
            d_opposite = float(opposite_curve[-1])
    d_r_end = d_side if side == "r" else d_opposite
    d_p_end = d_side if side == "p" else d_opposite
    finite_curve = curve is not None
    if not finite_curve:
        flags.append(FLAG_NON_FINITE_CURVE)
    cutoff = bonded_cutoff(
        str(elements.get(maps[0], "H")), str(elements.get(maps[1], "H")),
        tolerance=tolerance,
    )
    trend = _trend((d_side - d_ts)) if d_side is not None else "n/a"
    if kind == "order_changed":
        # Distance cannot decide bond order; consistency means bonded-range
        # endpoints on both sides with the TS in between.
        bonded_r = d_r_end is not None and d_r_end <= cutoff
        bonded_p = d_p_end is not None and d_p_end <= cutoff
        if bonded_r and bonded_p:
            verdict = SUPPORT_SUPPORT
            detail = "bonded at both endpoints; order itself is graph-defined"
        elif bonded_r or bonded_p:
            verdict = SUPPORT_WEAK
            detail = "only one endpoint within the bonded cutoff"
        else:
            verdict = SUPPORT_MISMATCH
            detail = "neither endpoint within the bonded cutoff"
        return EventSupport(
            kind=kind, maps=maps, branch="both" if branch and opposite_branch else side,
            d_r_end=d_r_end, d_ts=d_ts, d_p_end=d_p_end,
            d_min=None, d_max=None, trend=trend, verdict=verdict,
            progress_fraction=progress, detail=detail,
        )
    if d_side is None:
        return EventSupport(
            kind=kind, maps=maps, branch="none" if branch is None else side,
            d_r_end=d_r_end, d_ts=d_ts, d_p_end=d_p_end, d_min=None, d_max=None,
            trend="n/a", verdict=SUPPORT_WEAK,
            progress_fraction=None,
            detail="branch curve unavailable (missing branch or non-finite)",
        )
    assert curve is not None
    shortened = d_side < d_ts - TREND_MARGIN
    bonded = d_side <= cutoff
    if kind == "formed":
        if shortened and bonded:
            verdict = SUPPORT_SUPPORT
        elif shortened or bonded:
            verdict = SUPPORT_WEAK
        else:
            verdict = SUPPORT_MISMATCH
    else:  # broken: the bond re-forms toward its own side's endpoint
        if shortened and bonded:
            verdict = SUPPORT_SUPPORT
        elif shortened or bonded:
            verdict = SUPPORT_WEAK
        else:
            verdict = SUPPORT_MISMATCH
    return EventSupport(
        kind=kind, maps=maps, branch=side,
        d_r_end=d_r_end, d_ts=d_ts, d_p_end=d_p_end,
        d_min=float(curve.min()), d_max=float(curve.max()),
        trend=trend, verdict=verdict, progress_fraction=progress,
    )


def _verdict_migration(
    entry: Mapping[str, Any],
    elements: Mapping[int, str],
    frames: NDArray[np.float64],
    layout: IrcLayout,
    row_of_map: Sequence[int],
    orientation: str,
    *,
    tolerance: float,
    flags: list[str],
) -> EventSupport:
    """Validate one hydrogen migration by its two partner-distance curves."""
    h_map = int(entry["h"])
    from_map = entry.get("from")
    to_map = entry.get("to")
    h_row = row_of_map[h_map - 1]
    curves: list[tuple[str, float, float, float | None]] = []
    for label, partner in (("from", from_map), ("to", to_map)):
        if partner is None:
            continue
        partner_row = row_of_map[int(partner) - 1]
        branch_side = "r" if label == "from" else "p"
        branch = _branch_for(layout, branch_side, orientation, flags)
        if branch is None:
            curves.append((label, math.inf, math.inf, None))
            continue
        curve = _curve(frames, branch, h_row, partner_row)
        if curve is None:
            flags.append(FLAG_NON_FINITE_CURVE)
            curves.append((label, math.inf, math.inf, None))
            continue
        curves.append((label, float(curve[0]), float(curve[-1]), _progress(curve)))
    ts_values = [value for _label, value, _end, _p in curves if math.isfinite(value)]
    end_values = [value for _label, _ts_value, value, _p in curves if math.isfinite(value)]
    d_ts_mean = float(np.mean(ts_values)) if ts_values else None
    cutoff = bonded_cutoff(
        str(elements.get(h_map, "H")),
        str(elements.get(int(from_map if from_map is not None else to_map or h_map), "H")),
        tolerance=tolerance,
    )
    checks: list[bool] = []
    for label, ts_value, end_value, _progress_value in curves:
        if not math.isfinite(end_value):
            checks.append(False)
            continue
        partner_map = from_map if label == "from" else to_map
        assert partner_map is not None
        partner_cutoff = bonded_cutoff(
            str(elements.get(h_map, "H")), str(elements.get(int(partner_map), "H")),
            tolerance=tolerance,
        )
        checks.append(end_value < ts_value and end_value <= partner_cutoff)
    true_checks = sum(1 for check in checks if check)
    if not checks:
        verdict = SUPPORT_WEAK
    elif true_checks == len(checks):
        verdict = SUPPORT_SUPPORT
    elif true_checks == 0:
        verdict = SUPPORT_MISMATCH
    else:
        verdict = SUPPORT_WEAK
    progress_values = [
        progress for _label, _ts_value, _end, progress in curves if progress is not None
    ]
    d_r_end = next(
        (end for label, _ts_value, end, _p in curves if label == "from"), None
    )
    d_p_end = next(
        (end for label, _ts_value, end, _p in curves if label == "to"), None
    )
    trend = _trend(min(
        (end - ts_value for _label, ts_value, end, _p in curves if math.isfinite(end)),
        default=0.0,
    ))
    partner_for_maps = int(from_map) if from_map is not None else int(to_map or h_map)
    return EventSupport(
        kind="hydrogen_migration",
        maps=(h_map, partner_for_maps),
        branch="both" if len(curves) > 1 else ("r" if from_map is not None else "p"),
        d_r_end=d_r_end if d_r_end is not None and math.isfinite(d_r_end) else None,
        d_ts=d_ts_mean,
        d_p_end=d_p_end if d_p_end is not None and math.isfinite(d_p_end) else None,
        d_min=min(
            (end for _label, _ts_value, end, _p in curves if math.isfinite(end)),
            default=None,
        ),
        d_max=max(
            (end for _label, _ts_value, end, _p in curves if math.isfinite(end)),
            default=None,
        ),
        trend=trend,
        verdict=verdict,
        progress_fraction=(
            float(np.mean(progress_values)) if progress_values else None
        ),
        detail=f"cutoff={cutoff:.3f}",
    )


def validate_events(
    bond_events: Mapping[str, Sequence[Mapping[str, Any]]],
    layout: IrcLayout,
    frames: NDArray[np.float64],
    row_of_map: Sequence[int],
    elements_by_map: Mapping[int, str],
    orientation: str,
    *,
    tolerance: float,
) -> EventValidation:
    """Validate every mapped bond event against the IRC trajectory.

    ``bond_events`` is the map-space event mapping (``formed``/``broken``/
    ``order_changed`` lists of ``{atoms, order_r, order_p}`` plus the
    ``hydrogen_migration`` list of ``{h, from, to}``), ``row_of_map`` maps
    each map number to its TS/IRC row, and *orientation* decides which
    branch terminates at which side.  Under an unresolved orientation the
    curves run on a best-guess branch, so a claimed contradiction could be
    a branch-assignment artifact; ``mismatch`` verdicts are therefore
    capped to ``weak`` (never reported as contradictions).  Aggregate
    counters and the synchrony label complete the record.
    """
    flags: list[str] = []
    supports: list[EventSupport] = []
    for kind in ("formed", "broken", "order_changed"):
        for entry in bond_events.get(kind, ()):
            atoms = [int(value) for value in entry["atoms"]]
            supports.append(_verdict_event(
                kind, (atoms[0], atoms[1]), elements_by_map, frames, layout,
                row_of_map, orientation, tolerance=tolerance, flags=flags,
            ))
    for entry in bond_events.get("hydrogen_migration", ()):
        supports.append(_verdict_migration(
            entry, elements_by_map, frames, layout, row_of_map, orientation,
            tolerance=tolerance, flags=flags,
        ))
    if orientation == ORIENTATION_UNRESOLVED:
        capped: list[EventSupport] = []
        for item in supports:
            if item.verdict == SUPPORT_MISMATCH:
                note = "capped: a contradiction verdict requires a known branch"
                detail = f"{item.detail}; {note}" if item.detail else note
                capped.append(replace(item, verdict=SUPPORT_WEAK, detail=detail))
            else:
                capped.append(item)
        supports = capped
    n_support = sum(1 for item in supports if item.verdict == SUPPORT_SUPPORT)
    n_weak = sum(1 for item in supports if item.verdict == SUPPORT_WEAK)
    n_mismatch = sum(1 for item in supports if item.verdict == SUPPORT_MISMATCH)
    fractions = [
        item.progress_fraction for item in supports
        if item.progress_fraction is not None
    ]
    if len(fractions) < 2:
        synchrony = SYNCHRONY_SINGLE
        max_delta = (
            0.0 if not fractions else None
        )
        if len(fractions) == 1:
            max_delta = 0.0
    else:
        max_delta = max(fractions) - min(fractions)
        synchrony = (
            SYNCHRONY_SYNCHRONOUS if max_delta <= SYNCHRONY_MARGIN
            else SYNCHRONY_ASYNCHRONOUS
        )
    return EventValidation(
        event_support=tuple(supports),
        n_support=n_support,
        n_weak=n_weak,
        n_mismatch=n_mismatch,
        synchrony=synchrony,
        max_progress_delta=max_delta,
        quality_flags=tuple(sorted(set(flags))),
    )


__all__ = [
    "EventSupport",
    "EventValidation",
    "FLAG_BRANCH_MISSING",
    "FLAG_NON_FINITE_CURVE",
    "FLAG_ORIENTATION_UNRESOLVED",
    "SYNCHRONY_ASYNCHRONOUS",
    "SYNCHRONY_SINGLE",
    "SYNCHRONY_SYNCHRONOUS",
    "bond_pattern_orientation",
    "bonded_cutoff",
    "terminus_bond_scores",
    "validate_events",
]
