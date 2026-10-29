"""V5.4.1 WP3a — O(3)/permutation/batch/finiteness battery (plan §6.4).

Runs the plan's per-candidate geometry matrix against ALL three legal
views (G_chem, G_spatial, G_angle):

* translation: every distance/RBF/cos/basis/attr invariant;
* rotation AND reflection: same (all outputs are scalars — the views
  carry no vector output, so invariance is the requirement);
* full-atom permutation: mapped SETS of pairs/triplets and all counts
  are identical (exact — the provider math is permutation-equivariant
  up to row order, which canonicalisation removes);
* batch independence vs mixing: the mixed build equals the union of the
  single builds (offset), with no cross-graph edge in any view;
* every output finite over the whole fixture matrix (incl. extreme
  coordinates), gradients finite, counts detached floats.
"""

from __future__ import annotations

import pytest
import torch

from pes2ts_core.pesn.models.v541.feature_providers import (
    build_angle_triplets,
    build_legal_neighbor_views,
)
from tests.pesn.test_v541_feature_providers_fixtures import (
    all_finite,
    angle_geometry_batch,
    bonded_pair_batch,
    canonical_triplets,
    chem_pair_map,
    counts_are_floats,
    dense_sphere_batch,
    extreme_coordinate_batch,
    forward_batch,
    inverse_map,
    mapped_chem_pairs,
    mapped_spatial_pairs,
    no_neighbor_batch,
    one_chem_one_spatial_batch,
    permute_forward_batch,
    reflection_matrix,
    random_rotation,
    single_atom_batch,
    sn2_bidirectional_batch,
    sn2_forward_batch,
    sn2_reverse_batch,
    spatial_pair_map,
    sphere_center_batch,
    stretched_bond_batch,
    transform_forward_batch,
    zero_length_batch,
)
from scripts.agent_ops.feature_gate.v3_fixtures import make_sn2_sample

#: fp32 tolerances for transformed-geometry comparisons (the PERMUTATION
#: tests below are exact — no tolerance needed there).
ABS_TOL = 1e-5
REL_TOL = 1e-5


def _all_fixture_batches() -> dict[str, dict]:
    """The full §6.4/§6.8 fixture matrix."""
    return {
        "single_atom": single_atom_batch(),
        "bonded_pair": bonded_pair_batch(),
        "no_neighbor": no_neighbor_batch(),
        "stretched": stretched_bond_batch(),
        "sn2_forward": sn2_forward_batch(),
        "sn2_reverse": sn2_reverse_batch(),
        "sn2_bidir": sn2_bidirectional_batch(n_graphs=2),
        "one_chem_one_spatial": one_chem_one_spatial_batch(),
        "shell_tie": sphere_center_batch([1.0] * 11 + [2.0] * 3),
        "shell_over_cap": sphere_center_batch([1.5] * 13),
        "dense": dense_sphere_batch(15),
        "angle_pi": angle_geometry_batch("pi"),
        "angle_near_pi": angle_geometry_batch("near_pi"),
        "angle_zero": angle_geometry_batch("zero"),
        "angle_right": angle_geometry_batch("right"),
        "zero_length": zero_length_batch(),
        "extreme_scaled": extreme_coordinate_batch(1e4),
        "extreme_huge": extreme_coordinate_batch(1.0, huge=1e17),
    }


def _assert_close(a: torch.Tensor, b: torch.Tensor, what: str) -> None:
    assert a.shape == b.shape, what
    if a.numel() == 0:
        return
    assert torch.allclose(a, b, rtol=REL_TOL, atol=ABS_TOL), (
        f"{what}: max abs diff {float((a - b).abs().max())}")


#: Count keys that are purely structural (isometry must preserve them
#: exactly).  The three fp-boundary diagnostics (``n_zero_angle``,
#: ``n_collinear``, ``n_clamped_cos``) sit ON rounding boundaries under a
#: rotation, so they are compared with a small slack instead.
_STRUCTURAL_COUNTS = (
    "n_atoms", "n_graphs", "n_input_edges", "n_input_undirected_pairs",
    "n_chem_pairs", "n_chem_edges", "n_duplicate_edges_merged",
    "n_non_chem_pairs", "n_directions_synthesized", "n_spatial_pairs",
    "n_spatial_edges", "n_over_cap_directed", "n_selected_not_in_input",
    "n_input_pairs_beyond_cutoff", "n_centers", "n_centers_with_angle",
    "n_angle_neighbors_selected", "n_triplets",
    "n_boundary_shells_rejected", "n_boundary_tie_shells_rejected",
    "n_boundary_neighbors_rejected", "n_boundary_tie_neighbors_rejected",
)
_DIAGNOSTIC_COUNTS = ("n_zero_angle", "n_collinear", "n_clamped_cos")


def _ordered_triplets(view):
    """Triplets canonicalised for isometry comparison.

    fp noise can reorder the ``(distance, index)`` side ordering of tied
    neighbours and the pair-enumeration order after a rotation, so rows
    are re-sorted by the LABEL-invariant key ``(center, min(j,k),
    max(j,k))`` with the two sides swapped into index order.  The
    resulting tensors are directly comparable elementwise (the atom
    LABELS are unchanged by a rotation — only the order was unstable).
    """
    n0 = view.side_index[:, 0]
    n1 = view.side_index[:, 1]
    swap_col = (n0 > n1).unsqueeze(1)          # [T, 1]  for [T, 2]
    swap_deep = (n0 > n1).view(-1, 1, 1)       # [T,1,1] for [T,2,12/8]
    lo = torch.minimum(n0, n1)
    hi = torch.maximum(n0, n1)
    side_index = torch.stack((lo, hi), dim=1)
    side_distance = torch.where(swap_col, view.side_distance.flip(1),
                                view.side_distance)
    side_rbf12 = torch.where(swap_deep, view.side_rbf12.flip(1),
                             view.side_rbf12)
    side_attr = torch.where(swap_deep, view.side_chem_attr.flip(1),
                            view.side_chem_attr)
    base = int(view.center_has_angle.shape[0]) + 2
    key = ((view.center_index * base + lo) * base + hi)
    order = torch.argsort(key)
    return (view.center_index[order], side_index[order],
            side_distance[order], side_rbf12[order], side_attr[order],
            view.cos_theta[order], view.cos_basis[order])


def _compare_counts(ref_counts: dict, out_counts: dict) -> None:
    for key in _STRUCTURAL_COUNTS:
        if key in ref_counts:
            assert ref_counts[key] == out_counts[key], key
    for key in _DIAGNOSTIC_COUNTS:
        if key in ref_counts:
            # a rotation may flip a value sitting on the fp boundary of
            # the 1e-6 clamp/zero/collinear diagnostics — allow a tiny
            # slack, never a structural change
            assert abs(ref_counts[key] - out_counts[key]) <= 2.0, key


def _transform_invariance(batch: dict, matrix: torch.Tensor | None,
                          shift: torch.Tensor | None) -> None:
    moved = transform_forward_batch(batch, rotate=matrix, translate=shift)
    ref = build_legal_neighbor_views(batch)
    out = build_legal_neighbor_views(moved)
    ref_trip = build_angle_triplets(batch)
    out_trip = build_angle_triplets(moved)

    # discrete structure: unchanged (distance-based selection is
    # isometry-invariant; attrs are geometry-independent)
    assert torch.equal(ref.chem.pair_index, out.chem.pair_index)
    assert torch.equal(ref.spatial.pair_index, out.spatial.pair_index)
    assert torch.equal(ref.chem.pair_attr, out.chem.pair_attr)
    _compare_counts(ref.chem.counts, out.chem.counts)
    _compare_counts(ref.spatial.counts, out.spatial.counts)
    _compare_counts(ref_trip.counts, out_trip.counts)

    # scalar features: invariant within fp32 tolerance
    _assert_close(ref.chem.pair_distance, out.chem.pair_distance,
                  "chem distance")
    _assert_close(ref.chem.rbf12, out.chem.rbf12, "chem rbf12")
    _assert_close(ref.spatial.pair_distance, out.spatial.pair_distance,
                  "spatial distance")
    _assert_close(ref.spatial.rbf12, out.spatial.rbf12, "spatial rbf12")

    # triplets: canonical row/side order (fp noise may reorder tied
    # distances — the SET and the labels are invariant, the raw order is
    # not guaranteed across a rotation)
    r = _ordered_triplets(ref_trip)
    o = _ordered_triplets(out_trip)
    assert torch.equal(r[0], o[0]), "triplet centres"
    assert torch.equal(r[1], o[1]), "triplet side labels"
    assert torch.equal(ref_trip.center_has_angle,
                       out_trip.center_has_angle)
    assert torch.equal(ref_trip.angle_neighbor_count,
                       out_trip.angle_neighbor_count)
    _assert_close(r[2], o[2], "side distance")
    _assert_close(r[3], o[3], "side rbf12")
    _assert_close(r[5], o[5], "cos")
    _assert_close(r[6], o[6], "cos basis")
    # side chem attrs do not depend on geometry at all
    assert torch.equal(r[4], o[4])


# ---------------------------------------------------------------------------
# O(3) + translation (§6.4)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", [
    "sn2_forward", "sn2_bidir", "shell_tie", "dense", "angle_near_pi",
    "one_chem_one_spatial", "extreme_scaled",
])
def test_translation_invariance(kind: str) -> None:
    batch = _all_fixture_batches()[kind]
    shift = torch.tensor([12.5, -3.25, 7.75])
    _transform_invariance(batch, None, shift)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("kind", [
    "sn2_forward", "sn2_bidir", "shell_tie", "angle_right", "dense",
])
def test_rotation_invariance(kind: str, seed: int) -> None:
    batch = _all_fixture_batches()[kind]
    _transform_invariance(batch, random_rotation(seed), None)


@pytest.mark.parametrize("kind", [
    "sn2_forward", "shell_tie", "angle_near_pi",
])
def test_reflection_invariance(kind: str) -> None:
    """Reflections (improper O(3)) leave every scalar view invariant —
    no signed/chiral label is ever produced (§6.3.7)."""
    batch = _all_fixture_batches()[kind]
    _transform_invariance(batch, reflection_matrix(axis=2), None)
    _transform_invariance(
        batch,
        random_rotation(3) @ reflection_matrix(axis=0),
        torch.tensor([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# full-atom permutation (§6.4 / §6.8 row 4)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", [
    "sn2_forward", "sn2_bidir", "shell_tie", "shell_over_cap", "dense",
    "angle_zero", "one_chem_one_spatial",
])
def test_permutation_equivariance_sets_and_counts(kind: str) -> None:
    batch = _all_fixture_batches()[kind]
    n = int(batch["pos"].shape[0])
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(17))
    permuted = permute_forward_batch(batch, perm)
    inv = inverse_map(perm)

    ref = build_legal_neighbor_views(batch)
    out = build_legal_neighbor_views(permuted)
    ref_trip = build_angle_triplets(batch)
    out_trip = build_angle_triplets(permuted)

    # sets of pairs/triplets (mapped) are IDENTICAL — exact values
    assert mapped_chem_pairs(ref.chem, inv) == chem_pair_map(out.chem)
    assert mapped_spatial_pairs(ref.spatial, inv) == spatial_pair_map(
        out.spatial)
    assert canonical_triplets(ref_trip, inv) == canonical_triplets(out_trip)
    # per-atom counts travel with their atom; dict counts are label-free
    assert torch.equal(ref_trip.angle_neighbor_count[perm],
                       out_trip.angle_neighbor_count)
    assert torch.equal(ref_trip.center_has_angle[perm],
                       out_trip.center_has_angle)
    assert ref.chem.counts == out.chem.counts
    assert ref.spatial.counts == out.spatial.counts
    assert ref_trip.counts == out_trip.counts


# ---------------------------------------------------------------------------
# batch independence vs mixing + no cross-graph leakage (§6.8 row 6)
# ---------------------------------------------------------------------------
def test_batch_independent_vs_mixed_all_views() -> None:
    batch_a = forward_batch([make_sn2_sample("forward")])
    batch_b = forward_batch([make_sn2_sample("reverse", jitter_seed=7)])
    batch_ab = forward_batch([
        make_sn2_sample("forward"), make_sn2_sample("reverse", jitter_seed=7)])
    view_a = build_legal_neighbor_views(batch_a)
    view_b = build_legal_neighbor_views(batch_b)
    view_ab = build_legal_neighbor_views(batch_ab)
    trip_a = build_angle_triplets(batch_a)
    trip_b = build_angle_triplets(batch_b)
    trip_ab = build_angle_triplets(batch_ab)

    merged_chem = dict(chem_pair_map(view_a.chem))
    merged_chem.update({(a + 6, b + 6): v
                        for (a, b), v in chem_pair_map(view_b.chem).items()})
    merged_spatial = dict(spatial_pair_map(view_a.spatial))
    merged_spatial.update({(a + 6, b + 6): v
                           for (a, b), v in spatial_pair_map(
                               view_b.spatial).items()})
    assert merged_chem == chem_pair_map(view_ab.chem)
    assert merged_spatial == spatial_pair_map(view_ab.spatial)
    # triplets: union with the +6 atom offset
    offset = {i: i + 6 for i in range(6)}
    merged_trip = canonical_triplets(trip_a) | canonical_triplets(
        trip_b, offset)
    assert merged_trip == canonical_triplets(trip_ab)

    # no cross-graph edge in ANY view of the mixed batch
    graph_index = batch_ab["graph_index"]
    for edge in (view_ab.chem.edge_index, view_ab.spatial.edge_index):
        assert bool((graph_index[edge[0]] == graph_index[edge[1]]).all())
    trip_center = trip_ab.center_index
    trip_sides = trip_ab.side_index                  # [T, 2]
    assert bool((graph_index[trip_center].unsqueeze(1)
                 == graph_index[trip_sides]).all())
    # counts add exactly
    assert (trip_ab.counts["n_triplets"]
            == trip_a.counts["n_triplets"] + trip_b.counts["n_triplets"])


def test_multi_graph_bidirectional_batch_no_leakage() -> None:
    # bidirectional fixture: n_graphs families × fwd/rev = 2·n_graphs
    # samples (graphs), 6 atoms each
    batch = sn2_bidirectional_batch(n_graphs=3)
    views = build_legal_neighbor_views(batch)
    trip = build_angle_triplets(batch)
    graph_index = batch["graph_index"]
    for edge in (views.chem.edge_index, views.spatial.edge_index):
        assert bool((graph_index[edge[0]] == graph_index[edge[1]]).all())
    assert bool((graph_index[trip.center_index].unsqueeze(1)
                 == graph_index[trip.side_index]).all())
    assert views.chem.counts["n_graphs"] == 6
    # 6 graphs × 5 chem pairs
    assert views.chem.counts["n_chem_pairs"] == 30


# ---------------------------------------------------------------------------
# finiteness sweep + gradients (§6.4)
# ---------------------------------------------------------------------------
def test_all_outputs_finite_across_fixture_matrix() -> None:
    for name, batch in _all_fixture_batches().items():
        views = build_legal_neighbor_views(batch)
        if name == "zero_length":
            # degenerate angle input: the ANGLE provider must refuse it
            # with its typed error while the distance views stay finite
            from pes2ts_core.pesn.models.v541.feature_providers import (
                FeatureProviderError,
            )

            with pytest.raises(FeatureProviderError) as excinfo:
                build_angle_triplets(batch)
            assert excinfo.value.code == "V541_PROVIDER_ZERO_DISTANCE"
            trip = None
        else:
            trip = build_angle_triplets(batch)
        for label, view in (("chem", views.chem), ("spatial", views.spatial)):
            assert all_finite(view), f"{name}/{label} has non-finite output"
            assert counts_are_floats(view), f"{name}/{label} counts"
            # masks are boolean and explicit
            if label == "chem":
                assert view.valid.dtype == torch.bool
                assert bool(view.valid.all())
            else:
                assert view.valid.dtype == torch.bool
                assert bool(view.valid.all())
        if trip is None:
            continue
        assert all_finite(trip), f"{name}/angle has non-finite output"
        assert counts_are_floats(trip), f"{name}/angle counts"
        assert trip.triplet_valid.dtype == torch.bool
        assert trip.center_has_angle.dtype == torch.bool
        assert bool(trip.triplet_valid.all())
        # angle caps hold everywhere
        assert bool((trip.angle_neighbor_count <= 12).all())
        per_center = torch.bincount(
            trip.center_index, minlength=int(trip.center_has_angle.shape[0]))
        assert int(per_center.max()) <= 66


def test_gradients_finite_across_views() -> None:
    base = sn2_bidirectional_batch(n_graphs=2)
    pos = base["pos"].detach().clone().requires_grad_(True)
    batch = dict(base)
    batch["pos"] = pos
    views = build_legal_neighbor_views(batch)
    trip = build_angle_triplets(batch)
    loss = (views.chem.rbf12.sum() + views.spatial.rbf12.sum()
            + trip.cos_basis.sum() + trip.side_rbf12.sum())
    loss.backward()
    assert pos.grad is not None
    assert torch.isfinite(pos.grad).all()
    assert float(pos.grad.abs().sum()) > 0.0


def test_no_vector_outputs_and_detached_counts() -> None:
    """Views are invariant scalars only (no equivariant vector could be
    misread as E(3)-safe), and diagnostics are plain floats."""
    views = build_legal_neighbor_views(sn2_forward_batch())
    trip = build_angle_triplets(sn2_forward_batch())
    for view in (views.chem, views.spatial, trip):
        for name, value in vars(view).items():
            if name == "counts" or not isinstance(value, torch.Tensor):
                continue
            if (torch.is_floating_point(value) and value.ndim >= 1
                    and value.shape[-1] == 3):
                # no float [*, 3] vector tensor is ever produced — the
                # views are invariant scalars only
                pytest.fail(f"{type(view).__name__}.{name} looks like a vector")
        assert all(isinstance(v, float) for v in view.counts.values())
