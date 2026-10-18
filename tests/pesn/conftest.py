"""Shared fixtures: synthetic reactions, teacher datasets, and split trees.

The synthetic "H-transfer" trajectory moves one hydrogen from atom A to atom
B along monotone interpolations, giving well-defined arc lengths, sides, and
driver coordinates without touching any real data source.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# The optional ML backend is deliberately not a base requirement.  Without
# torch, skip this directory instead of failing collection: several modules
# call require_torch() at import time and raise BackendMissingError.
collect_ignore_glob: list[str] = []
if importlib.util.find_spec("torch") is None:
    collect_ignore_glob = ["test_*.py"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.contracts.inputs import (  # noqa: E402
    BondTerm,
    CoordinateDef,
    CoordinateSpec,
    TaskEdit,
    TaskGraph,
)
from pes2ts_core.pesn.offline.labels.progress import compute_progress_labels, masses_for  # noqa: E402
from pes2ts_core.pesn.offline.labels.direction import (  # noqa: E402
    compute_direction_labels,
    evaluate_distance_coords,
)
from pes2ts_core.pesn.offline.training.loader import ReactionData  # noqa: E402


def synthetic_task_graph() -> TaskGraph:
    """A--H broken, H--B formed, with a spectator C=O bond on both sides."""
    return TaskGraph(
        schema_version="pesn_task_graph_v1",
        atomic_numbers=(8, 1, 1, 8),
        atom_map=(1, 2, 3, 4),
        reactant_bonds=(BondTerm(0, 1, 1.0), BondTerm(2, 3, 1.0), BondTerm(0, 3, 1.0)),
        product_bonds=(BondTerm(0, 1, 1.0), BondTerm(2, 3, 1.0), BondTerm(0, 3, 1.0)),
        edits=(
            TaskEdit("broken", 0, 1, 1.0, 0.0),
            TaskEdit("formed", 2, 3, 0.0, 1.0),
        ),
        source="synthetic_fixture",
        provenance={"fixture": "tests/pesn"},
    )


def synthetic_coordinate_spec() -> CoordinateSpec:
    return CoordinateSpec(
        schema_version="pesn_coordinate_spec_v1",
        coords=(
            CoordinateDef("distance", (0, 1), "angstrom", "driver", 1.0, 0.0, "A-H"),
            CoordinateDef("distance", (2, 3), "angstrom", "driver", 1.0, 0.0, "H-B"),
        ),
        spec_version="synthetic_v1",
    )


def synthetic_trajectory(
    n_branch_frames: int = 6, bend: float = 0.0
) -> np.ndarray:
    """Frames ``(1 + 2*n_branch_frames, 4, 3)`` in archive order.

    Layout ``ts_first_dual_branch``: frame 0 is the TS, frames ``1..n`` walk
    branch_1 (product side for ``orientation="P_first"``) from near-TS to the
    terminus, frames ``n+1..2n`` walk branch_2 (reactant side) the same way.
    On branch_1 the broken-edit distance A--H stretches toward the terminus
    while the formed-edit distance H--B shrinks to bonding; branch_2 mirrors.
    A slight *bend* breaks collinearity so Kabsch registration is exercised,
    and frame spacing grows sub-quadratically toward each terminus so arc
    length is non-uniform in frame index.
    """
    ah_ts, hb_ts = 1.25, 1.25
    frames = [_h_transfer_frame(ah_ts, hb_ts, bend)]
    for k in range(1, n_branch_frames + 1):
        t = (k / n_branch_frames) ** 1.5
        frames.append(_h_transfer_frame(ah_ts + 0.55 * t, hb_ts - 0.45 * t, bend))
    for k in range(1, n_branch_frames + 1):
        t = (k / n_branch_frames) ** 1.5
        frames.append(_h_transfer_frame(ah_ts - 0.45 * t, hb_ts + 0.55 * t, -bend))
    return np.asarray(frames, dtype=np.float64)


def _h_transfer_frame(ah: float, hb: float, bend: float) -> np.ndarray:
    """A(0)-H(1) at *ah*; the H(2)-B(3) pair keeps distance *hb*."""
    return np.asarray(
        [
            [0.0, 0.0, 0.0],
            [ah, bend, 0.0],
            [3.0, 1.2, bend * 0.5],
            [3.0 + hb, 1.2, bend * 0.5],
        ],
        dtype=np.float64,
    )


def build_synthetic_reaction_data(reaction_id: str = "RXN_TEST0000001") -> ReactionData:
    coords = synthetic_trajectory()
    task_graph = synthetic_task_graph()
    masses = masses_for(task_graph.atomic_numbers)
    progress = compute_progress_labels(
        coords,
        masses,
        ts_frame_index=0,
        splice_index=coords.shape[0] // 2,
        layout="ts_first_dual_branch",
        orientation="P_first",
    )
    q = evaluate_distance_coords(coords, [(0, 1), (2, 3)])
    direction = compute_direction_labels(q, progress)
    return ReactionData(
        reaction_id=reaction_id,
        atomic_numbers=np.asarray(task_graph.atomic_numbers),
        coords=coords,
        task_graph=task_graph,
        coordinate_spec=synthetic_coordinate_spec(),
        s=progress.signed_progress,
        d=progress.path_proximity,
        s_mask=progress.progress_mask,
        side=progress.side,
        d_arc=progress.d_arc,
        direction=direction.direction,
        activity=direction.activity,
        dir_mask=direction.direction_mask,
        act_mask=direction.activity_mask,
        method="synthetic",
    )


@pytest.fixture()
def synthetic_reaction() -> ReactionData:
    return build_synthetic_reaction_data()


@pytest.fixture()
def synthetic_reactions() -> list[ReactionData]:
    return [
        build_synthetic_reaction_data(f"RXN_TEST{i:010d}") for i in range(4)
    ]
