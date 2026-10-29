"""Shared fixtures/helpers for the V5.4.1 WP3a feature-provider tests
(plan §6.2–§6.8; config ``feature_recipe``).

Every fixture produces a FORWARD-VIEW torch batch at feature level F2 —
exactly what ``model.forward`` and the WP3b branches consume — built with
the REAL legacy pipeline (``build_sample_features_v3`` → ``collate_v3`` →
``to_torch_batch`` → ``forward_view_v3``) so the providers are exercised
against genuine inputs: F2-masked ``edge_attr`` (first 12 RBF slots
zero), 22 legacy edge channels, bidirectional main graph.

Custom geometries cover the plan's §6.4/§6.8 acceptance matrix: single /
double atom, no-neighbour, stretched-beyond-cutoff covalent, distance
ties, dense >32 degree, zero-length angle sides, collinear / near-
collinear / zero-angle / right-angle triplets, and extreme-but-finite
coordinates.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import torch

from pes2ts_core.pesn.contracts.key_atom_v3 import (
    EDGE_ABS_DELTA_KEY,
    EDGE_ATTR_KEY,
    EDGE_B_SRC_KEY,
    EDGE_B_TGT_KEY,
    EDGE_DELTA_B_KEY,
    EDGE_INDEX_KEY,
    EDGE_IS_CHANGED_KEY,
    FORWARD_VIEW_KEYS_V3,
    IntentRecordV3,
    KeyAtomSampleV3,
    NODE_DEGK_KEY,
    NODE_GRAPH_KEY,
    NODE_INTENT_ROLE_KEY,
    NODE_IS_KEY_KEY,
    NODE_KEY_SET_KEY,
    NODE_POS_KEY,
    NODE_Z_KEY,
    forward_view_v3,
)
from pes2ts_core.pesn.features.key_atom_graph_v3 import (
    build_sample_features_v3,
    collate_v3,
    to_torch_batch,
)
from pes2ts_core.pesn.models.v541.feature_providers import (
    AngleTripletView,
    ChemGraphView,
    SpatialGraphView,
)

# --- batch construction -----------------------------------------------------
def forward_batch(
        samples: Sequence[KeyAtomSampleV3], *, level: str = "F2",
) -> dict:
    """Build + collate + torch-convert + strip to the forward view."""
    built = [build_sample_features_v3(s, level=level) for s in samples]
    return forward_view_v3(to_torch_batch(collate_v3(built)))


def make_intent(
        g_src: Iterable[tuple[int, int, float]],
        g_tgt: Iterable[tuple[int, int, float]],
        *, cv_deltas: Sequence[float] = (),
        cv_coords: Sequence[tuple[int, int]] = (),
        direction: str = "forward",
        curve_id: str = "curve_x",
) -> IntentRecordV3:
    """Minimal legal intent record (at least one edit or CV is required
    downstream by ``KeyAtomSampleV3`` — the helper fails fast otherwise)."""
    g_src = tuple(tuple(b) for b in g_src)
    g_tgt = tuple(tuple(b) for b in g_tgt)
    if not (g_src != g_tgt or cv_coords or cv_deltas):
        raise ValueError("fixture intent needs an edit or a CV (key set)")
    return IntentRecordV3(
        intent_id=f"{curve_id}::{direction}",
        curve_id=curve_id,
        direction=direction,
        reverse_of_intent_id=(
            f"{curve_id}::rev" if direction == "forward"
            else f"{curve_id}::fwd"),
        intent_pair_id=f"{curve_id}::pair",
        g_src_bonds=g_src,
        g_tgt_bonds=g_tgt,
        cv_deltas=tuple(float(d) for d in cv_deltas),
        cv_coordinates=tuple(tuple(c) for c in cv_coords),
        intent_version="v3",
    )


def make_sample(
        atomic_numbers: Sequence[int],
        xyz: Sequence[Sequence[float]],
        intent: IntentRecordV3, *, node_index: int = 0,
) -> KeyAtomSampleV3:
    return KeyAtomSampleV3(
        curve_id=intent.curve_id, node_index=node_index, intent=intent,
        atomic_numbers=tuple(int(z) for z in atomic_numbers),
        xyz=tuple(tuple(float(v) for v in row) for row in xyz))


# --- geometry fixtures ------------------------------------------------------
def single_atom_batch() -> dict:
    """N=1, E=0: every view empty but well-shaped."""
    sample = make_sample(
        (6,), [(0.0, 0.0, 0.0)],
        make_intent((), (), cv_deltas=(0.1,), cv_coords=((0, 0),)))
    return forward_batch([sample])


def bonded_pair_batch(dist: float = 1.0) -> dict:
    """Two atoms, one covalent bond at ``dist`` Å (also within cutoff)."""
    sample = make_sample(
        (6, 8), [(0.0, 0.0, 0.0), (dist, 0.0, 0.0)],
        make_intent(((0, 1, 1.0),), ((0, 1, 1.0),),
                    cv_deltas=(0.05,), cv_coords=((0, 1),)))
    return forward_batch([sample])


def no_neighbor_batch() -> dict:
    """Two atoms 10 Å apart, NO bond, NO spatial contact (both views empty)."""
    sample = make_sample(
        (6, 8), [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
        make_intent((), (), cv_deltas=(0.05,), cv_coords=((0, 1),)))
    return forward_batch([sample])


def stretched_bond_batch(dist: float = 5.0) -> dict:
    """Covalent pair BEYOND the 4.5 Å cutoff (legacy keeps it in the main
    graph; G_spatial must NOT treat it as a spatial edge, §6.6.2)."""
    sample = make_sample(
        (6, 8), [(0.0, 0.0, 0.0), (dist, 0.0, 0.0)],
        make_intent(((0, 1, 1.0),), ((0, 1, 1.0),),
                    cv_deltas=(0.05,), cv_coords=((0, 1),)))
    return forward_batch([sample])


def _ring_xyz(count: int, radius: float) -> list[tuple[float, float, float]]:
    """``count`` points on a circle of ``radius`` centred on the origin."""
    out = [(0.0, 0.0, 0.0)]
    for k in range(count):
        theta = 2.0 * math.pi * k / count
        out.append((radius * math.cos(theta), radius * math.sin(theta), 0.0))
    return out


def sphere_center_batch(
        radii: Sequence[float], *, cv_keys: tuple[int, int] = (0, 1),
) -> dict:
    """Centre atom 0 with ``len(radii)`` chem neighbours at the given
    radii (one per ring position).  Key set comes from a CV so NO extra
    edit edge disturbs the neighbour counts.  Equal radii ⇒ one tied
    distance shell (the ``reject_boundary_tie_shell`` fixtures)."""
    xyz = _ring_xyz(len(radii), 1.0)
    for k, radius in enumerate(radii):
        x, y, _ = xyz[k + 1]
        norm = math.hypot(x, y) or 1.0
        xyz[k + 1] = (x / norm * radius, y / norm * radius, 0.0)
    bonds_src = tuple((0, k + 1, 1.0) for k in range(len(radii)))
    sample = make_sample(
        (6,) + (1,) * len(radii), xyz,
        make_intent(bonds_src, bonds_src,
                    cv_deltas=(0.1,), cv_coords=(cv_keys,)))
    return forward_batch([sample])


def dense_sphere_batch(n_shell: int = 39) -> dict:
    """Centre + ``n_shell`` atoms on a UNIT sphere: every pair is within
    the cutoff, the centre's radius selection retains ALL shell atoms by
    the 1e-6 Å tie rule (realised degree > nominal 32, §6.6.2)."""
    xyz = _ring_xyz(n_shell, 1.0)
    bonds_src = tuple((0, k + 1, 1.0) for k in range(n_shell))
    sample = make_sample(
        (6,) + (1,) * n_shell, xyz,
        make_intent(bonds_src, bonds_src,
                    cv_deltas=(0.1,), cv_coords=((0, 1),)))
    return forward_batch([sample])


def zero_length_batch() -> dict:
    """Atoms 0 and 1 at the SAME position (zero-length chem bond) plus a
    normal neighbour: legal for G_chem (finite distance/RBF), typed
    refusal for angle geometry (§6.3.6)."""
    sample = make_sample(
        (6, 1, 1),
        [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        make_intent(((0, 1, 1.0), (0, 2, 1.0)),
                    ((0, 1, 1.0), (0, 2, 1.0)),
                    cv_deltas=(0.1,), cv_coords=((1, 2),)))
    return forward_batch([sample])


def angle_geometry_batch(kind: str) -> dict:
    """Centre 0 with two chem neighbours placed to hit one cos value.

    ``kind``: ``"pi"`` (cos = −1), ``"near_pi"`` (cos ≈ −1 + 5e-7),
    ``"zero"`` (cos = +1, near-zero angle), ``"right"`` (cos = 0).
    """
    placements = {
        "pi": [(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)],
        "near_pi": [(1.0, 0.0, 0.0), (-1.0, 1e-3, 0.0)],
        "zero": [(1.0, 0.0, 0.0), (2.0, 0.0, 0.0)],
        "right": [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
    }
    if kind not in placements:
        raise ValueError(f"unknown angle geometry {kind!r}")
    a, b = placements[kind]
    sample = make_sample(
        (6, 1, 1), [(0.0, 0.0, 0.0), a, b],
        make_intent(((0, 1, 1.0), (0, 2, 1.0)),
                    ((0, 1, 1.0), (0, 2, 1.0)),
                    cv_deltas=(0.1,), cv_coords=((1, 2),)))
    return forward_batch([sample])


def one_chem_one_spatial_batch() -> dict:
    """Centre 0: ONE chem neighbour (atom 1) and one spatial-only contact
    (atom 2, non-bonded at 1.5 Å).  Angle pool is chem-only (§6.3.2): no
    triplet may be formed through the spatial contact."""
    sample = make_sample(
        (6, 1, 8),
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.5, 0.0, 0.0)],
        make_intent(((0, 1, 1.0),), ((0, 1, 1.0),),
                    cv_deltas=(0.1,), cv_coords=((0, 1),)))
    return forward_batch([sample])


def extreme_coordinate_batch(scale: float = 1e4, huge: float = 0.0) -> dict:
    """SN2-like geometry scaled by ``scale`` (all pairs leave the spatial
    cutoff; chem edges remain) and, when ``huge > 0``, atom 0 is moved to
    ``huge`` Å — extreme but finite (plan §6.4)."""
    from scripts.agent_ops.feature_gate.v3_fixtures import (
        ATOMIC_NUMBERS_SN2,
        BASE_XYZ_SN2,
        make_sn2_intent,
    )

    xyz = [tuple(v * scale for v in row) for row in BASE_XYZ_SN2]
    if huge:
        xyz[0] = (huge, 0.0, 0.0)
    sample = KeyAtomSampleV3(
        curve_id="curve_x", node_index=0,
        intent=make_sn2_intent("forward"),
        atomic_numbers=ATOMIC_NUMBERS_SN2,
        xyz=tuple(xyz))
    return forward_batch([sample])


def sn2_forward_batch() -> dict:
    """The canonical 6-atom SN2 fixture, single forward intent, F2."""
    from scripts.agent_ops.feature_gate.v3_fixtures import make_sn2_sample

    return forward_batch([make_sn2_sample("forward")])


def sn2_reverse_batch() -> dict:
    """Same geometry with the REVERSED intent (source/target swapped)."""
    from scripts.agent_ops.feature_gate.v3_fixtures import make_sn2_sample

    return forward_batch([make_sn2_sample("reverse")])


def sn2_bidirectional_batch(n_graphs: int = 2) -> dict:
    """Multi-graph batch mixing forward and reverse intents."""
    from scripts.agent_ops.feature_gate.v3_fixtures import make_v3_gate_batch

    return make_v3_gate_batch(n_graphs=n_graphs, level="F2")


# --- batch transforms -------------------------------------------------------
_NODE_KEYS = (NODE_Z_KEY, NODE_POS_KEY, NODE_GRAPH_KEY, NODE_IS_KEY_KEY,
              NODE_DEGK_KEY, NODE_INTENT_ROLE_KEY)
_EDGE_ALIGNED_KEYS = (EDGE_ATTR_KEY, EDGE_B_SRC_KEY, EDGE_B_TGT_KEY,
                      EDGE_DELTA_B_KEY, EDGE_ABS_DELTA_KEY,
                      EDGE_IS_CHANGED_KEY)


def permute_forward_batch(batch: Mapping[str, torch.Tensor],
                          perm: torch.Tensor) -> dict:
    """Atom-permute a forward batch: ``pos_new[k] = pos_old[perm[k]]``.

    Node-aligned keys are gathered, ``edge_index``/``key_index`` are
    remapped through the inverse permutation, and edge-aligned attribute
    rows FOLLOW their edge (row k still describes edge k).
    """
    perm = perm.long()
    inverse = torch.argsort(perm)
    out: dict = {}
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            out[key] = value
        elif key in _NODE_KEYS:
            out[key] = value[perm]
        elif key == NODE_KEY_SET_KEY:
            out[key] = inverse[value]
        elif key == EDGE_INDEX_KEY:
            out[key] = inverse[value]
        elif key in _EDGE_ALIGNED_KEYS:
            out[key] = value            # rows are edge-aligned, not node-
        else:                           # aligned — unchanged by atom order
            out[key] = value
    return out


def transform_forward_batch(
        batch: Mapping[str, torch.Tensor], *,
        rotate: torch.Tensor | None = None,
        translate: torch.Tensor | None = None,
        scale: float | None = None,
) -> dict:
    """Move/scale ONLY ``pos`` (edge attrs are geometry-independent and
    O(3)/translation leave the distance-based selection unchanged)."""
    pos = batch[NODE_POS_KEY]
    if scale is not None:
        pos = pos * float(scale)
    if rotate is not None:
        pos = pos @ rotate.to(dtype=pos.dtype).T
    if translate is not None:
        pos = pos + translate.to(dtype=pos.dtype)
    out = dict(batch)
    out[NODE_POS_KEY] = pos
    return out


def random_rotation(seed: int = 0) -> torch.Tensor:
    """Deterministic O(3) matrix (rotation with det = ±1 both allowed —
    reflections are exercised separately via explicit sign flips)."""
    generator = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    q = q @ torch.diag(torch.sign(torch.diag(r)))
    return q


def reflection_matrix(axis: int = 2) -> torch.Tensor:
    matrix = torch.eye(3)
    matrix[axis, axis] = -1.0
    return matrix


# --- view comparison helpers ------------------------------------------------
def chem_pair_map(view: ChemGraphView) -> dict[tuple[int, int], tuple]:
    """``{(a, b): (8 attrs, distance, rbf12 tuple)}`` of a chem view."""
    out: dict[tuple[int, int], tuple] = {}
    for p in range(int(view.pair_index.shape[1])):
        a = int(view.pair_index[0, p])
        b = int(view.pair_index[1, p])
        attrs = tuple(float(v) for v in view.pair_attr[p])
        dist = float(view.pair_distance[p])
        rbf = tuple(float(v) for v in view.pair_rbf12[p])
        out[(a, b)] = (attrs, dist, rbf)
    return out


def spatial_pair_map(view: SpatialGraphView) -> dict[tuple[int, int], tuple]:
    """``{(a, b): (distance, rbf12 tuple)}`` of a spatial view."""
    out: dict[tuple[int, int], tuple] = {}
    for p in range(int(view.pair_index.shape[1])):
        a = int(view.pair_index[0, p])
        b = int(view.pair_index[1, p])
        out[(a, b)] = (float(view.pair_distance[p]),
                       tuple(float(v) for v in view.pair_rbf12[p]))
    return out


def canonical_triplets(
        view: AngleTripletView, node_map: Mapping[int, int] | None = None,
) -> set[tuple]:
    """Set of canonical triplet rows for order-independent comparison.

    Each row is ``(center, side0, side1)`` with the two sides sorted by
    neighbour index (the pair is unordered, §6.3).  ``node_map`` remaps
    the indices OF THIS VIEW (old → new): pass it for the ORIGINAL view
    and nothing for the permuted rebuild — the two sets must be equal.
    """
    node_map = node_map or {}
    rows: set[tuple] = set()
    for t in range(int(view.center_index.shape[0])):
        center = int(view.center_index[t])
        center = node_map.get(center, center)
        sides = []
        for s in range(2):
            nbr = int(view.side_index[t, s])
            nbr = node_map.get(nbr, nbr)
            sides.append((
                nbr,
                float(view.side_distance[t, s]),
                tuple(float(v) for v in view.side_rbf12[t, s]),
                tuple(float(v) for v in view.side_chem_attr[t, s]),
            ))
        sides.sort(key=lambda side: side[0])
        cos = float(view.cos_theta[t])
        basis = tuple(float(v) for v in view.cos_basis[t])
        rows.add((center, sides[0], sides[1], cos, basis))
    return rows


def mapped_chem_pairs(
        view: ChemGraphView, node_map: Mapping[int, int],
) -> dict[tuple[int, int], tuple]:
    """Chem pairs of an ORIGINAL view re-keyed through ``node_map``
    (old → new), canonically sorted so keys compare directly against a
    permuted rebuild."""
    out: dict[tuple[int, int], tuple] = {}
    for (a, b), value in chem_pair_map(view).items():
        a2, b2 = node_map.get(a, a), node_map.get(b, b)
        out[(min(a2, b2), max(a2, b2))] = value
    return out


def mapped_spatial_pairs(
        view: SpatialGraphView, node_map: Mapping[int, int],
) -> dict[tuple[int, int], tuple]:
    """Spatial pairs of an ORIGINAL view re-keyed through ``node_map``."""
    out: dict[tuple[int, int], tuple] = {}
    for (a, b), value in spatial_pair_map(view).items():
        a2, b2 = node_map.get(a, a), node_map.get(b, b)
        out[(min(a2, b2), max(a2, b2))] = value
    return out


def inverse_map(perm: torch.Tensor) -> dict[int, int]:
    """``{old_index: new_index}`` for ``permute_forward_batch``'s gather."""
    inverse = torch.argsort(perm)
    return {int(old): int(inverse[old]) for old in range(int(perm.numel()))}


def all_finite(view: object) -> bool:
    """Every tensor output of a view is finite (counts are plain floats)."""
    for name, value in vars(view).items():
        if name == "counts":
            continue
        if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
            if not bool(torch.isfinite(value).all()):
                return False
    return True


def counts_are_floats(view: object) -> bool:
    counts = getattr(view, "counts")
    return bool(counts) and all(
        isinstance(v, float) and not isinstance(v, bool)
        for v in counts.values())


# --- a couple of shared fixture-level sanity tests --------------------------
def test_fixture_batches_are_forward_view_f2() -> None:
    """Fixtures are genuine F2 forward batches (17-slot pack, RBF slots
    masked, 22-channel edge input present)."""
    from pes2ts_core.pesn.contracts.key_atom_v3 import (
        EDGE_INPUT_DIM_V3,
        EDGE_PACK_DIM_V3,
    )

    batch = sn2_forward_batch()
    assert set(batch) <= set(FORWARD_VIEW_KEYS_V3)
    assert batch[EDGE_ATTR_KEY].shape[1] == EDGE_PACK_DIM_V3 == 17
    assert batch[EDGE_ATTR_KEY].shape[1] + 5 == EDGE_INPUT_DIM_V3 == 22
    # F2 keeps slots 12–16 and MASKS the first 12 (RBF) to zero.
    assert bool((batch[EDGE_ATTR_KEY][:, :12] == 0).all())
    assert bool((batch[EDGE_ATTR_KEY][:, 12:] != 0).any())
    for key in (EDGE_B_SRC_KEY, EDGE_B_TGT_KEY, EDGE_DELTA_B_KEY,
                EDGE_ABS_DELTA_KEY, EDGE_IS_CHANGED_KEY):
        assert batch[key].shape[0] == batch[EDGE_INDEX_KEY].shape[1]


def test_fixture_intent_reversal_pairs() -> None:
    from scripts.agent_ops.feature_gate.v3_fixtures import make_sn2_intent

    intent = make_sn2_intent("forward")
    reverse = intent.reversed_intent()
    assert intent.g_src_bonds == reverse.g_tgt_bonds
    assert intent.g_tgt_bonds == reverse.g_src_bonds
    assert intent.pair_orders() != reverse.pair_orders()
