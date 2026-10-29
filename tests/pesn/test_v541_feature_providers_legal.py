"""V5.4.1 WP3a — ``G_chem`` / ``G_spatial`` semantics tests (plan §6.2,
§6.6.2, §6.8 rows 1/2/6).

Covers: chem membership (covalent union + edit, spatial-only excluded),
the §6.6.2 beyond-cutoff-covalent rule, legacy-selection subset +
tie-expanded degree > 32, empty/single/double-atom cases, batch
independent-vs-mixed consistency without cross-graph leakage, extreme
finite coordinates, zero-length pairs (legal for the distance view),
gradient finiteness, pure-function/no-cache behaviour and exact chunk
equivalence.
"""

from __future__ import annotations

import pytest
import torch

from pes2ts_core.pesn.contracts.key_atom_v3 import EDGE_INDEX_KEY
from pes2ts_core.pesn.models.v541.feature_providers import (
    FeatureProviderError,
    build_angle_triplets,
    build_chem_graph,
    build_legal_neighbor_views,
    build_spatial_graph,
)
from tests.pesn.test_v541_feature_providers_fixtures import (
    bonded_pair_batch,
    chem_pair_map,
    dense_sphere_batch,
    extreme_coordinate_batch,
    forward_batch,
    mapped_chem_pairs,
    mapped_spatial_pairs,
    inverse_map,
    no_neighbor_batch,
    one_chem_one_spatial_batch,
    permute_forward_batch,
    single_atom_batch,
    sn2_forward_batch,
    spatial_pair_map,
    stretched_bond_batch,
    zero_length_batch,
)
from scripts.agent_ops.feature_gate.v3_fixtures import make_sn2_sample


def _input_pair_keys(batch: dict) -> set[tuple[int, int]]:
    edge = batch[EDGE_INDEX_KEY]
    return {(min(int(edge[0, e]), int(edge[1, e])),
             max(int(edge[0, e]), int(edge[1, e])))
            for e in range(int(edge.shape[1]))}


# ---------------------------------------------------------------------------
# G_chem membership (§6.2 / §6.6.4)
# ---------------------------------------------------------------------------
def test_chem_is_covalent_union_plus_edit() -> None:
    batch = sn2_forward_batch()
    view = build_chem_graph(batch)
    pairs = set(chem_pair_map(view))
    # SN2 covalent union: C–Br, C–O (both edits) + 3× C–H spectators.
    assert pairs == {(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)}
    # spatial-only contacts (Br–O, Br–H, …) are NOT chem edges
    assert (1, 2) not in pairs
    assert (3, 4) not in pairs
    assert view.counts["n_chem_pairs"] == 5
    assert view.counts["n_chem_edges"] == 10
    assert view.counts["n_non_chem_pairs"] == 10   # 15 input pairs − 5


def test_chem_excludes_spatial_only_pairs() -> None:
    views = build_legal_neighbor_views(one_chem_one_spatial_batch())
    assert set(chem_pair_map(views.chem)) == {(0, 1)}
    # the non-bonded 1.5 Å contact exists ONLY in the spatial view
    assert set(spatial_pair_map(views.spatial)) == {(0, 1), (0, 2), (1, 2)}


def test_chem_is_subset_of_input_and_bidirectional() -> None:
    batch = sn2_forward_batch()
    view = build_chem_graph(batch)
    inputs = _input_pair_keys(batch)
    for p in range(int(view.pair_index.shape[1])):
        pair = (int(view.pair_index[0, p]), int(view.pair_index[1, p]))
        assert pair in inputs
        assert pair[0] < pair[1]
    assert int(view.edge_index.shape[1]) == 2 * int(view.pair_index.shape[1])
    assert torch.equal(view.edge_attr[0::2], view.edge_attr[1::2])


# ---------------------------------------------------------------------------
# G_spatial = legacy selection subset (§6.6.2 — THE acceptance rule)
# ---------------------------------------------------------------------------
def test_stretched_covalent_is_chem_but_never_spatial() -> None:
    """Covalent pair kept beyond the 4.5 Å cutoff: chem YES, spatial NO
    (E3 must not mistake it for a spatial edge, plan §6.6.2)."""
    batch = stretched_bond_batch(5.0)
    views = build_legal_neighbor_views(batch)
    assert set(chem_pair_map(views.chem)) == {(0, 1)}
    assert set(spatial_pair_map(views.spatial)) == set()
    assert views.spatial.counts["n_input_pairs_beyond_cutoff"] == 1
    assert views.spatial.counts["n_spatial_pairs"] == 0
    # distance stays physical even beyond the cutoff
    assert abs(float(views.chem.pair_distance[0]) - 5.0) < 1e-5


def test_spatial_is_legacy_selection_subset_of_main_graph() -> None:
    batch = sn2_forward_batch()
    views = build_legal_neighbor_views(batch)
    inputs = _input_pair_keys(batch)
    spatial = set(spatial_pair_map(views.spatial))
    # every SN2 pair lies within the cutoff ⇒ the selection recovers the
    # whole main graph (covalent-within-cutoff pairs are spatial too)
    assert spatial == inputs
    assert views.spatial.counts["n_selected_not_in_input"] == 0
    assert views.spatial.counts["n_input_pairs_beyond_cutoff"] == 0
    # chem ⊆ spatial here (all bonds within cutoff) — but NOT in general
    # (see the stretched-covalent test above).
    assert set(chem_pair_map(views.chem)) <= spatial


def test_spatial_selection_never_crosses_graphs() -> None:
    batch = forward_batch([
        make_sn2_sample("forward", graph_index=0),
        make_sn2_sample("forward", graph_index=1)])
    graph_index = batch["graph_index"]
    views = build_legal_neighbor_views(batch)
    for view in (views.chem, views.spatial):
        edge = view.edge_index
        assert int(edge.shape[1]) > 0
        same = graph_index[edge[0]] == graph_index[edge[1]]
        assert bool(same.all())
    assert views.spatial.counts["n_graphs"] == 2
    assert views.chem.counts["n_graphs"] == 2
    # graphs are 12 Å apart in x ⇒ no spatial pair may bridge them
    assert int(views.spatial.counts["n_spatial_pairs"]) == 15 * 2


def test_dense_tie_retention_exceeds_nominal_32() -> None:
    """39 shell atoms at radius 1.0: the centre keeps ALL of them via the
    1e-6 Å tie rule — realised degree > nominal 32, counted not
    truncated (§6.6.2)."""
    batch = dense_sphere_batch(39)
    views = build_legal_neighbor_views(batch)
    spatial = views.spatial
    assert spatial.counts["n_selected_not_in_input"] == 0
    assert spatial.counts["n_over_cap_directed"] > 0
    degree = torch.bincount(spatial.edge_index[0], minlength=40)
    assert int(degree.max()) > 32
    # centre keeps the FULL tied shell (39 pairs), never cut to 32
    centre_pairs = sum(
        1 for p in range(int(spatial.pair_index.shape[1]))
        if 0 in (int(spatial.pair_index[0, p]), int(spatial.pair_index[1, p])))
    assert centre_pairs == 39


# ---------------------------------------------------------------------------
# degenerate sizes / empty cases (§6.4)
# ---------------------------------------------------------------------------
def test_single_atom_and_no_neighbour_views() -> None:
    single = build_legal_neighbor_views(single_atom_batch())
    assert int(single.chem.pair_index.shape[1]) == 0
    assert int(single.spatial.pair_index.shape[1]) == 0

    noneigh = build_legal_neighbor_views(no_neighbor_batch())
    # 10 Å apart, unbounded pair: in NEITHER view
    assert int(noneigh.chem.pair_index.shape[1]) == 0
    assert int(noneigh.spatial.pair_index.shape[1]) == 0
    assert noneigh.spatial.counts["n_input_pairs_beyond_cutoff"] == 0


def test_double_atom_views() -> None:
    views = build_legal_neighbor_views(bonded_pair_batch(1.0))
    assert int(views.chem.pair_index.shape[1]) == 1
    assert int(views.spatial.pair_index.shape[1]) == 1
    assert torch.equal(views.chem.edge_index, views.spatial.edge_index)


def test_zero_length_pair_is_legal_for_distance_views() -> None:
    """Two coincident atoms: G_chem keeps the pair with distance 0 Å and
    FINITE RBF/mask (only the ANGLE provider refuses zero sides)."""
    base = zero_length_batch()
    views = build_legal_neighbor_views(base)
    zero_pairs = [
        p for p in range(int(views.chem.pair_index.shape[1]))
        if float(views.chem.pair_distance[p]) == 0.0]
    assert zero_pairs, "expected the coincident-atom chem pair"
    assert torch.isfinite(views.chem.rbf12).all()
    assert torch.isfinite(views.chem.distance).all()
    assert bool(views.chem.valid.all())
    # gradient through a zero distance stays finite (safe norm)
    pos = base["pos"].detach().clone().requires_grad_(True)
    batch = dict(base)
    batch["pos"] = pos
    out = build_chem_graph(batch)
    out.rbf12.sum().backward()
    assert torch.isfinite(pos.grad).all()


# ---------------------------------------------------------------------------
# batch independence / mixing (§6.8 row 6)
# ---------------------------------------------------------------------------
def test_batch_independent_vs_mixed_chem_spatial() -> None:
    batch_a = forward_batch([make_sn2_sample("forward")])
    batch_b = forward_batch([make_sn2_sample("reverse", jitter_seed=7)])
    batch_ab = forward_batch([
        make_sn2_sample("forward"), make_sn2_sample("reverse", jitter_seed=7)])
    view_a = build_legal_neighbor_views(batch_a)
    view_b = build_legal_neighbor_views(batch_b)
    view_ab = build_legal_neighbor_views(batch_ab)

    merged_chem = dict(chem_pair_map(view_a.chem))
    merged_chem.update(
        {(a + 6, b + 6): v for (a, b), v in chem_pair_map(view_b.chem).items()})
    merged_spatial = dict(spatial_pair_map(view_a.spatial))
    merged_spatial.update(
        {(a + 6, b + 6): v
         for (a, b), v in spatial_pair_map(view_b.spatial).items()})
    assert merged_chem == chem_pair_map(view_ab.chem)
    assert merged_spatial == spatial_pair_map(view_ab.spatial)
    # per-graph counts add up (no cross-graph double counting)
    assert (view_ab.chem.counts["n_chem_pairs"]
            == view_a.chem.counts["n_chem_pairs"]
            + view_b.chem.counts["n_chem_pairs"])
    assert (view_ab.spatial.counts["n_spatial_pairs"]
            == view_a.spatial.counts["n_spatial_pairs"]
            + view_b.spatial.counts["n_spatial_pairs"])
    # nothing bridges the graphs
    graph_index = batch_ab["graph_index"]
    for view in (view_ab.chem, view_ab.spatial):
        edge = view.edge_index
        assert bool((graph_index[edge[0]] == graph_index[edge[1]]).all())


# ---------------------------------------------------------------------------
# permutation of the input batch leaves the SETS unchanged (§6.4)
# ---------------------------------------------------------------------------
def test_permuted_input_yields_same_chem_spatial_sets() -> None:
    batch = sn2_forward_batch()
    perm = torch.randperm(6, generator=torch.Generator().manual_seed(3))
    permuted = permute_forward_batch(batch, perm)
    ref = build_legal_neighbor_views(batch)
    out = build_legal_neighbor_views(permuted)
    inv = inverse_map(perm)
    assert mapped_chem_pairs(ref.chem, inv) == chem_pair_map(out.chem)
    assert mapped_spatial_pairs(ref.spatial, inv) == spatial_pair_map(out.spatial)


# ---------------------------------------------------------------------------
# extreme-but-legal coordinates (§6.4)
# ---------------------------------------------------------------------------
def test_extreme_finite_coordinates_stay_finite() -> None:
    for batch in (extreme_coordinate_batch(1e4),
                  extreme_coordinate_batch(1.0, huge=1e17)):
        views = build_legal_neighbor_views(batch)
        for view in (views.chem, views.spatial):
            assert torch.isfinite(view.distance).all()
            assert torch.isfinite(view.rbf12).all()
            assert torch.isfinite(view.pair_rbf12).all()
            assert bool(view.valid.all())
            assert all(isinstance(v, float) for v in view.counts.values())
    # scaled SN2: every pair leaves the cutoff ⇒ empty spatial, chem intact
    scaled = build_legal_neighbor_views(extreme_coordinate_batch(1e4))
    assert int(scaled.spatial.pair_index.shape[1]) == 0
    assert int(scaled.chem.pair_index.shape[1]) == 5


# ---------------------------------------------------------------------------
# gradients / purity / chunking (§6.3.8, §6.4 finiteness)
# ---------------------------------------------------------------------------
def test_gradients_flow_and_stay_finite() -> None:
    base = sn2_forward_batch()
    pos = base["pos"].detach().clone().requires_grad_(True)
    batch = dict(base)
    batch["pos"] = pos
    views = build_legal_neighbor_views(batch)
    loss = (views.chem.rbf12.sum() + views.spatial.rbf12.sum()
            + views.chem.distance.sum() + views.spatial.distance.sum())
    loss.backward()
    assert pos.grad is not None
    assert torch.isfinite(pos.grad).all()
    assert float(pos.grad.abs().sum()) > 0.0


def test_outputs_follow_pos_no_stale_cache() -> None:
    """Providers are pure: no cross-call cache may freeze geometry."""
    base = sn2_forward_batch()
    moved = dict(base)
    moved["pos"] = base["pos"] * 1.05 + 0.07
    ref = build_chem_graph(base)
    out = build_chem_graph(moved)
    assert not torch.equal(ref.pair_distance, out.pair_distance)
    assert not torch.equal(ref.rbf12, out.rbf12)
    # same input twice ⇒ identical output (determinism, no hidden state)
    again = build_chem_graph(base)
    assert torch.equal(ref.pair_distance, again.pair_distance)
    assert torch.equal(ref.rbf12, again.rbf12)
    assert ref.counts == again.counts


def test_spatial_chunking_is_exact() -> None:
    batch = dense_sphere_batch(15)
    tiny = build_spatial_graph(batch, chunk_size=1)
    huge = build_spatial_graph(batch, chunk_size=100_000)
    assert torch.equal(tiny.pair_index, huge.pair_index)
    assert torch.equal(tiny.distance, huge.distance)
    assert torch.equal(tiny.rbf12, huge.rbf12)
    assert tiny.counts == huge.counts


def test_invalid_chunk_size_is_typed_error() -> None:
    batch = sn2_forward_batch()
    for builder in (build_spatial_graph,):
        with pytest.raises(FeatureProviderError) as excinfo:
            builder(batch, chunk_size=0)
        assert excinfo.value.code == "V541_PROVIDER_PARAM"


def test_spatial_only_batch_has_empty_chem_view() -> None:
    """Non-bonded pair inside the cutoff: the input carries edges but
    G_chem is empty (well-shaped 0-row views) while G_spatial keeps the
    pair — the P=0-with-E>0 path must not crash (§6.2 membership)."""
    from tests.pesn.test_v541_feature_providers_fixtures import (
        forward_batch,
        make_intent,
        make_sample,
    )

    sample = make_sample(
        (6, 8), [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)],
        make_intent((), (), cv_deltas=(0.05,), cv_coords=((0, 1),)))
    batch = forward_batch([sample])
    assert int(batch[EDGE_INDEX_KEY].shape[1]) == 2   # spatial pair present
    views = build_legal_neighbor_views(batch)
    assert int(views.chem.pair_index.shape[1]) == 0
    assert tuple(views.chem.edge_index.shape) == (2, 0)
    assert tuple(views.chem.edge_attr.shape) == (0, 8)
    assert int(views.spatial.pair_index.shape[1]) == 1
    trip = build_angle_triplets(batch)
    assert int(trip.center_index.shape[0]) == 0
    assert trip.counts["n_chem_pairs"] == 0.0
    assert not bool(trip.center_has_angle.any())
