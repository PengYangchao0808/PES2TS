"""IRC bond-event evidence fixtures: verdicts, trends, and quality flags."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from pes2ts_core.g1.irc_bond_evidence import (
    FLAG_BRANCH_MISSING,
    FLAG_ORIENTATION_UNRESOLVED,
    bond_pattern_orientation,
    bonded_cutoff,
    validate_events,
)
from pes2ts_core.g1.truth_alignment import analyze_irc_layout


def _transfer_frames() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Physically consistent H transfer: rows in map order 1(H),2(O),3(H)."""
    reactant = np.array([[0.95, 0.0, 0.0], [0.0, 0.0, 0.0], [3.5, 0.0, 0.0]])
    product = np.array([[3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    ts = np.array([[1.3, 0.0, 0.0], [0.0, 0.0, 0.0], [1.3, 0.0, 0.0]])
    branch_a = np.linspace(0.0, 1.0, 11)[:, None, None] * (reactant - ts) + ts
    branch_b = np.linspace(0.1, 1.0, 11)[:, None, None] * (product - ts) + ts
    return np.vstack([branch_a, branch_b]), ts


def _events() -> dict[str, list[dict[str, object]]]:
    return {
        "formed": [{"atoms": [2, 3], "order_r": None, "order_p": 1.0}],
        "broken": [{"atoms": [1, 2], "order_r": 1.0, "order_p": None}],
        "order_changed": [],
        "hydrogen_migration": [
            {"h": 1, "from": 2, "to": None},
            {"h": 3, "from": None, "to": 2},
        ],
    }


def test_h_transfer_events_all_support() -> None:
    frames, ts = _transfer_frames()
    layout = analyze_irc_layout(frames, ts)
    validation = validate_events(
        _events(), layout, frames, [0, 1, 2], {1: "H", 2: "O", 3: "H"},
        "R_first", tolerance=0.45,
    )
    assert validation.n_support == 4
    assert validation.n_weak == 0
    assert validation.n_mismatch == 0
    formed = validation.event_support[0]
    assert formed.trend == "decreasing"
    assert formed.d_p_end == pytest.approx(1.0, abs=1e-6)
    assert formed.d_r_end > 3.0


def test_wrong_direction_event_is_mismatch() -> None:
    frames, ts = _transfer_frames()
    layout = analyze_irc_layout(frames, ts)
    events = {
        "formed": [{"atoms": [1, 3], "order_r": None, "order_p": 1.0}],
        "broken": [], "order_changed": [], "hydrogen_migration": [],
    }
    validation = validate_events(
        events, layout, frames, [0, 1, 2], {1: "H", 2: "O", 3: "H"},
        "R_first", tolerance=0.45,
    )
    assert validation.n_mismatch == 1


def test_orientation_unresolved_flags_and_weak_verdicts() -> None:
    frames, ts = _transfer_frames()
    layout = analyze_irc_layout(frames, ts)
    validation = validate_events(
        _events(), layout, frames, [0, 1, 2], {1: "H", 2: "O", 3: "H"},
        "unresolved", tolerance=0.45,
    )
    assert FLAG_ORIENTATION_UNRESOLVED in validation.quality_flags
    # A claimed contradiction could be a branch-assignment artifact under an
    # unresolved orientation, so mismatch verdicts are capped to weak.
    assert validation.n_mismatch == 0
    assert validation.n_support + validation.n_weak == 4


def test_single_branch_trajectory_reports_missing_branch() -> None:
    reactant = np.array([[0.95, 0.0, 0.0], [0.0, 0.0, 0.0], [3.5, 0.0, 0.0]])
    ts = reactant + 0.2
    frames = np.linspace(0.0, 1.0, 8)[:, None, None] * (reactant - ts) + ts
    layout = analyze_irc_layout(frames, ts)
    validation = validate_events(
        _events(), layout, frames, [0, 1, 2], {1: "H", 2: "O", 3: "H"},
        "R_first", tolerance=0.45,
    )
    assert FLAG_BRANCH_MISSING in validation.quality_flags
    assert validation.n_support == 0


def test_non_finite_curve_is_weak_and_flagged() -> None:
    frames, ts = _transfer_frames()
    frames = frames.copy()
    # Poison a coordinate an event curve actually touches: atom 2 (map 3)
    # on the product branch is the formed O-H pair's hydrogen.
    frames[15, 2, 1] = np.inf
    layout = analyze_irc_layout(frames, ts)
    validation = validate_events(
        _events(), layout, frames, [0, 1, 2], {1: "H", 2: "O", 3: "H"},
        "R_first", tolerance=0.45,
    )
    assert "non_finite_curve" in validation.quality_flags
    affected = [e for e in validation.event_support if e.verdict == "weak"]
    assert affected


def test_order_changed_requires_bonded_endpoints() -> None:
    # A genuine order change stays bonded on both sides: build a short
    # trajectory whose pair distance is 1.0 at R, 1.1 at TS, 0.98 at P.
    reactant = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 4.0]])
    product = np.array([[0.98, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 4.0]])
    ts = np.array([[1.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 4.0]])
    branch_a = np.linspace(0.0, 1.0, 8)[:, None, None] * (reactant - ts) + ts
    branch_b = np.linspace(0.1, 1.0, 8)[:, None, None] * (product - ts) + ts
    frames = np.vstack([branch_a, branch_b])
    layout = analyze_irc_layout(frames, ts)
    events = {
        "formed": [], "broken": [],
        "order_changed": [{"atoms": [1, 2], "order_r": 1.0, "order_p": 2.0}],
        "hydrogen_migration": [],
    }
    validation = validate_events(
        events, layout, frames, [0, 1, 2], {1: "H", 2: "O", 3: "H"},
        "R_first", tolerance=0.45,
    )
    order_change = validation.event_support[0]
    assert order_change.verdict == "support"
    assert "order itself is graph-defined" in order_change.detail


def test_synchrony_labels() -> None:
    frames, ts = _transfer_frames()
    layout = analyze_irc_layout(frames, ts)
    validation = validate_events(
        _events(), layout, frames, [0, 1, 2], {1: "H", 2: "O", 3: "H"},
        "R_first", tolerance=0.45,
    )
    assert validation.synchrony == "synchronous"
    only_one = {
        "formed": [{"atoms": [2, 3], "order_r": None, "order_p": 1.0}],
        "broken": [], "order_changed": [], "hydrogen_migration": [],
    }
    single = validate_events(
        only_one, layout, frames, [0, 1, 2], {1: "H", 2: "O", 3: "H"},
        "R_first", tolerance=0.45,
    )
    assert single.synchrony == "single_event"


def test_bonded_cutoff_uses_covalent_radii() -> None:
    assert bonded_cutoff("H", "H", tolerance=0.45) == pytest.approx(0.31 + 0.31 + 0.45)
    assert bonded_cutoff("C", "O", tolerance=0.0) == pytest.approx(0.76 + 0.66)


def test_bond_pattern_orientation_decides_from_graph() -> None:
    # Branch A of the transfer fixture ends at the reactant, so the R bond
    # set is fully satisfied there and the P set is missing exactly its
    # edits; local pair distances decide even where whole-molecule RMSD is
    # conformer-noisy.
    frames, ts = _transfer_frames()
    layout = analyze_irc_layout(frames, ts)
    elements = {1: "H", 2: "O", 3: "H"}
    orientation, scores = bond_pattern_orientation(
        layout, frames,
        [[1, 2, 1.0]], [[2, 3, 1.0]],
        elements, [0, 1, 2], tolerance=0.45,
    )
    assert orientation == "R_first"
    assert scores["branch_a"]["r"] == (1, 1)
    assert scores["branch_a"]["p"] == (0, 1)
    assert scores["branch_b"]["r"] == (0, 1)
    assert scores["branch_b"]["p"] == (1, 1)
    swapped, _ = bond_pattern_orientation(
        layout, frames,
        [[2, 3, 1.0]], [[1, 2, 1.0]],
        elements, [0, 1, 2], tolerance=0.45,
    )
    assert swapped == "P_first"


def test_bond_pattern_orientation_is_unresolved_for_identical_bond_sets() -> None:
    # Pure order-change reactions share the bond pairs on both sides, so no
    # pattern signal exists and the caller falls back to the RMSD decision.
    frames, ts = _transfer_frames()
    layout = analyze_irc_layout(frames, ts)
    orientation, scores = bond_pattern_orientation(
        layout, frames,
        [[1, 2, 1.0]], [[1, 2, 2.0]],
        {1: "H", 2: "O", 3: "H"}, [0, 1, 2], tolerance=0.45,
    )
    assert orientation == "unresolved"
    assert scores["branch_a"]["r"] == scores["branch_a"]["p"]
