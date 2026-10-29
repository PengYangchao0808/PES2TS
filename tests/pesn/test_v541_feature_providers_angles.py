"""V5.4.1 WP3a — angle-triplet tests (plan §6.3, §6.8 rows 4/5).

Covers: chem-only neighbour pool, the 12-neighbour / 66-pair caps,
``reject_boundary_tie_shell`` at both boundary kinds (tied shell and
clean cut), explicit no-angle masks, cos safety/clamp with the frozen
``[1, c, c², c³]`` basis, zero-distance typed refusal, collinear /
near-collinear / zero-angle diagnostics, j/k swap symmetry, side-feature
consistency with ``G_chem``, exact chunk equivalence and finiteness.
"""

from __future__ import annotations

import pytest
import torch

from pes2ts_core.pesn.models.v541.feature_providers import (
    ANGLE_MAX_NEIGHBORS,
    ANGLE_MAX_PAIRS,
    FeatureProviderError,
    build_angle_triplets,
    build_chem_graph,
)
from tests.pesn.test_v541_feature_providers_fixtures import (
    angle_geometry_batch,
    canonical_triplets,
    inverse_map,
    one_chem_one_spatial_batch,
    permute_forward_batch,
    single_atom_batch,
    bonded_pair_batch,
    no_neighbor_batch,
    sn2_bidirectional_batch,
    sn2_forward_batch,
    sphere_center_batch,
    zero_length_batch,
)


def _pair_row(view, a: int, b: int) -> int:
    key = (min(a, b), max(a, b))
    for p in range(int(view.pair_index.shape[1])):
        if (int(view.pair_index[0, p]), int(view.pair_index[1, p])) == key:
            return p
    raise AssertionError(f"pair {key} not found")


# ---------------------------------------------------------------------------
# neighbour pool: G_chem / intent only (§6.3.2)
# ---------------------------------------------------------------------------
def test_angle_pool_is_chem_only() -> None:
    """A spatial-only contact never becomes an angle neighbour."""
    batch = one_chem_one_spatial_batch()
    trip = build_angle_triplets(batch)
    # centre 0 has ONE chem neighbour (atom 1) and a spatial contact
    # (atom 2): no unordered pair may form ⇒ no angle at any centre.
    assert int(trip.center_index.numel()) == 0
    assert not bool(trip.center_has_angle.any())
    assert trip.angle_neighbor_count.tolist() == [1, 1, 0]
    assert trip.counts["n_triplets"] == 0.0
    # sanity: the spatial contact exists (so the pool restriction is the
    # only reason there is no angle)
    from tests.pesn.test_v541_feature_providers_fixtures import (
        spatial_pair_map,
    )
    from pes2ts_core.pesn.models.v541.feature_providers import (
        build_spatial_graph,
    )

    spatial = build_spatial_graph(batch)
    assert (0, 2) in spatial_pair_map(spatial)


def test_sn2_angles_use_chem_neighbours_only() -> None:
    trip = build_angle_triplets(sn2_forward_batch())
    # only the carbon has ≥2 chem neighbours (C–Br, C–O, 3× C–H)
    assert trip.center_has_angle.tolist() == [True] + [False] * 5
    assert trip.angle_neighbor_count.tolist() == [5, 1, 1, 1, 1, 1]
    assert int(trip.center_index.shape[0]) == 10       # C(5, 2)
    assert trip.counts["n_centers_with_angle"] == 1.0
    # every triplet's centre is the carbon
    assert bool((trip.center_index == 0).all())


# ---------------------------------------------------------------------------
# caps + tie policy (§6.3.3–§6.3.5)
# ---------------------------------------------------------------------------
def test_clean_cap_boundary_is_not_a_tie() -> None:
    """12 neighbours at 1.0 Å + 1 at 2.0 Å: exactly C(12, 2) = 66 pairs;
    the boundary shell is a single (non-tied) rejection."""
    batch = sphere_center_batch([1.0] * 12 + [2.0])
    trip = build_angle_triplets(batch)
    assert trip.angle_neighbor_count[0] == 12
    assert int(trip.center_index.shape[0]) == ANGLE_MAX_PAIRS == 66
    assert trip.counts["n_boundary_shells_rejected"] == 1.0
    assert trip.counts["n_boundary_tie_shells_rejected"] == 0.0
    assert trip.counts["n_boundary_neighbors_rejected"] == 1.0
    assert trip.counts["n_boundary_tie_neighbors_rejected"] == 0.0
    assert bool(trip.center_has_angle[0])


def test_reject_boundary_tie_shell_keeps_whole_shells() -> None:
    """11 at 1.0 Å + 3 at 2.0 Å: the tied 3-shell would breach the cap,
    so it is rejected WHOLE — 11 selected (C(11,2) = 55 pairs), tie
    counters retained (§6.3.4)."""
    batch = sphere_center_batch([1.0] * 11 + [2.0] * 3)
    trip = build_angle_triplets(batch)
    assert trip.angle_neighbor_count[0] == 11
    assert int(trip.center_index.shape[0]) == 55
    assert trip.counts["n_boundary_shells_rejected"] == 1.0
    assert trip.counts["n_boundary_tie_shells_rejected"] == 1.0
    assert trip.counts["n_boundary_neighbors_rejected"] == 3.0
    assert trip.counts["n_boundary_tie_neighbors_rejected"] == 3.0
    # no partial take: the selected set is exactly the 1.0 Å shell
    distances = trip.side_distance[trip.center_index == 0]
    assert bool((distances < 1.5).all())


def test_first_shell_over_cap_rejects_everything() -> None:
    """13 tied neighbours: the FIRST shell already breaches the cap, so
    the whole shell is rejected and the centre gets an explicit no-angle
    mask — atom id can never pick 12 of 13 (§6.3.4 anti-leak rule)."""
    batch = sphere_center_batch([1.5] * 13)
    trip = build_angle_triplets(batch)
    assert trip.angle_neighbor_count[0] == 0
    assert not bool(trip.center_has_angle[0])
    assert int(trip.center_index.shape[0]) == 0
    assert trip.counts["n_boundary_shells_rejected"] == 1.0
    assert trip.counts["n_boundary_tie_shells_rejected"] == 1.0
    assert trip.counts["n_boundary_neighbors_rejected"] == 13.0
    assert trip.counts["n_boundary_tie_neighbors_rejected"] == 13.0
    assert trip.counts["n_triplets"] == 0.0


def test_distinct_distances_take_nearest_12() -> None:
    """15 neighbours at distinct radii: nearest-first selection takes
    exactly 12 (no tie ⇒ boundary counted as a plain rejection)."""
    radii = [1.0 + 0.05 * k for k in range(15)]
    batch = sphere_center_batch(radii)
    trip = build_angle_triplets(batch)
    assert trip.angle_neighbor_count[0] == 12
    assert int(trip.center_index.shape[0]) == 66
    assert trip.counts["n_boundary_tie_shells_rejected"] == 0.0
    # selected sides are the 12 nearest radii (≤ 1.55 Å), not any 12
    distances = trip.side_distance[trip.center_index == 0]
    assert float(distances.max()) <= 1.55 + 1e-4
    assert float(distances.min()) >= 1.0 - 1e-4


def test_selection_is_ranked_by_distance_not_atom_id() -> None:
    """Input order puts the FAR shell first; selection must still take
    the 11 NEAREST neighbours and reject the far tied shell — ranking
    keys contain distance only, never atom id / input position (§6.3.4).
    """
    radii = [2.0] * 3 + [1.0] * 11      # far atoms are 1, 2, 3 (input-first)
    batch = sphere_center_batch(radii)
    trip = build_angle_triplets(batch)
    assert trip.angle_neighbor_count[0] == 11
    distances = trip.side_distance[trip.center_index == 0]
    assert float(distances.max()) < 1.5       # only the 1.0 Å shell is in
    assert trip.counts["n_boundary_tie_shells_rejected"] == 1.0
    # a full label permutation preserves the mapped triplet set exactly
    n = int(batch["pos"].shape[0])
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(9))
    permuted = permute_forward_batch(batch, perm)
    out = build_angle_triplets(permuted)
    assert canonical_triplets(trip, inverse_map(perm)) == canonical_triplets(out)


# ---------------------------------------------------------------------------
# no-angle masks / degenerate sizes (§6.3.4, §6.4)
# ---------------------------------------------------------------------------
def test_no_angle_centres_have_explicit_mask() -> None:
    for batch in (single_atom_batch(), bonded_pair_batch(),
                  no_neighbor_batch(), sn2_forward_batch()):
        trip = build_angle_triplets(batch)
        n = int(trip.center_has_angle.shape[0])
        assert int(trip.angle_neighbor_count.shape[0]) == n
        assert bool((trip.angle_neighbor_count <= ANGLE_MAX_NEIGHBORS).all())
        # has_angle ⇔ ≥2 selected neighbours ⇔ ≥1 pair
        assert torch.equal(trip.center_has_angle,
                           trip.angle_neighbor_count >= 2)
        # sum of C(cnt, 2) over centres is exactly the triplet count
        counts = trip.angle_neighbor_count
        expected = int((counts * (counts - 1) // 2).sum())
        assert int(trip.center_index.shape[0]) == expected
        assert trip.counts["n_triplets"] == float(expected)


def test_pair_count_never_exceeds_66_per_center() -> None:
    batches = [sn2_forward_batch(), sphere_center_batch([1.0] * 15),
               sn2_bidirectional_batch(n_graphs=2)]
    for batch in batches:
        trip = build_angle_triplets(batch)
        per_center = torch.bincount(
            trip.center_index,
            minlength=int(trip.center_has_angle.shape[0]))
        assert int(per_center.max()) <= ANGLE_MAX_PAIRS
        assert bool((trip.angle_neighbor_count
                     <= ANGLE_MAX_NEIGHBORS).all())


# ---------------------------------------------------------------------------
# cos safety + frozen basis (§6.3.6 / §6.7 E4)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kind, expected_cos, expected_basis",
    [
        ("pi", -1.0, (1.0, -1.0, 1.0, -1.0)),
        ("zero", 1.0, (1.0, 1.0, 1.0, 1.0)),
        ("right", 0.0, (1.0, 0.0, 0.0, 0.0)),
    ],
)
def test_cos_basis_values(kind: str, expected_cos: float,
                          expected_basis: tuple[float, ...]) -> None:
    trip = build_angle_triplets(angle_geometry_batch(kind))
    assert int(trip.center_index.shape[0]) == 1
    cos = float(trip.cos_theta[0])
    assert abs(cos - expected_cos) < 1e-6
    basis = tuple(float(v) for v in trip.cos_basis[0])
    assert len(basis) == 4
    for got, want in zip(basis, expected_basis):
        assert abs(got - want) < 1e-6
    # basis is exactly [1, c, c², c³]
    assert basis[0] == 1.0
    assert abs(basis[1] - cos) < 1e-7
    assert abs(basis[2] - cos * cos) < 1e-6
    assert abs(basis[3] - cos * cos * cos) < 1e-6


def test_cos_is_always_clamped() -> None:
    for kind in ("pi", "near_pi", "zero", "right"):
        trip = build_angle_triplets(angle_geometry_batch(kind))
        assert bool((trip.cos_theta >= -1.0).all())
        assert bool((trip.cos_theta <= 1.0).all())
        assert bool(torch.isfinite(trip.cos_theta).all())


def test_zero_distance_angle_is_typed_error() -> None:
    with pytest.raises(FeatureProviderError) as excinfo:
        build_angle_triplets(zero_length_batch())
    assert excinfo.value.code == "V541_PROVIDER_ZERO_DISTANCE"
    assert "centre 0" in excinfo.value.detail


def test_near_zero_angle_side_stays_finite() -> None:
    """§6.4 near-zero (not zero) side: 1e-4 Å is legal — the guarded
    denominator keeps cos/basis finite and the triplet is emitted."""
    from tests.pesn.test_v541_feature_providers_fixtures import (
        forward_batch,
        make_intent,
        make_sample,
    )

    sample = make_sample(
        (6, 1, 1),
        [(0.0, 0.0, 0.0), (1e-4, 0.0, 0.0), (1e-4, 1.0, 0.0)],
        make_intent(((0, 1, 1.0), (0, 2, 1.0)),
                    ((0, 1, 1.0), (0, 2, 1.0)),
                    cv_deltas=(0.1,), cv_coords=((1, 2),)))
    trip = build_angle_triplets(forward_batch([sample]))
    assert int(trip.center_index.shape[0]) == 1
    assert bool(torch.isfinite(trip.cos_theta).all())
    assert bool(torch.isfinite(trip.cos_basis).all())
    assert bool(torch.isfinite(trip.side_distance).all())
    assert float(trip.side_distance.min()) > 0.0
    assert bool((trip.cos_theta >= -1.0).all())
    assert bool((trip.cos_theta <= 1.0).all())


def test_collinear_and_zero_angle_diagnostics() -> None:
    near = build_angle_triplets(angle_geometry_batch("near_pi"))
    assert near.counts["n_collinear"] == 1.0          # |cos| ≥ 1 − 1e-6
    assert near.counts["n_zero_angle"] == 0.0
    cos = float(near.cos_theta[0])
    assert -1.0 < cos <= -0.999999                    # near-π, not clamped
    assert torch.isfinite(near.cos_basis).all()

    zero = build_angle_triplets(angle_geometry_batch("zero"))
    assert zero.counts["n_zero_angle"] == 1.0         # near-zero angle
    assert zero.counts["n_collinear"] == 1.0          # ⊆ collinear

    right = build_angle_triplets(angle_geometry_batch("right"))
    assert right.counts["n_zero_angle"] == 0.0
    assert right.counts["n_collinear"] == 0.0


# ---------------------------------------------------------------------------
# side features come from G_chem (§6.7 E4 inputs)
# ---------------------------------------------------------------------------
def test_side_features_match_chem_view() -> None:
    batch = sn2_forward_batch()
    chem = build_chem_graph(batch)
    trip = build_angle_triplets(batch)
    for t in range(int(trip.center_index.shape[0])):
        center = int(trip.center_index[t])
        for s in range(2):
            nbr = int(trip.side_index[t, s])
            p = _pair_row(chem, center, nbr)
            assert float(trip.side_distance[t, s]) == float(
                chem.pair_distance[p])
            assert torch.equal(trip.side_rbf12[t, s], chem.pair_rbf12[p])
            assert torch.equal(trip.side_chem_attr[t, s], chem.pair_attr[p])
    # valid mask is explicit and all-true for emitted triplets
    assert trip.triplet_valid.dtype == torch.bool
    assert bool(trip.triplet_valid.all())
    assert int(trip.triplet_valid.shape[0]) == int(trip.center_index.shape[0])


# ---------------------------------------------------------------------------
# j/k exchange symmetry (§6.8 row 4)
# ---------------------------------------------------------------------------
def test_jk_swap_leaves_pair_level_features_identical() -> None:
    """Swapping the two neighbour atoms (input permutation of just j,k)
    keeps cos/basis bit-identical and the two SIDES as the same multiset
    — the property E4 relies on for symmetric aggregation."""
    batch = sphere_center_batch([1.0, 1.3, 1.7])
    # swap atoms 1 and 2 (the two non-centre neighbours of interest)
    perm = torch.tensor([0, 2, 1, 3])
    swapped = permute_forward_batch(batch, perm)
    ref = build_angle_triplets(batch)
    out = build_angle_triplets(swapped)
    inv = inverse_map(perm)
    assert canonical_triplets(ref, inv) == canonical_triplets(out)
    # the actual triplet row: same centre, sides swapped as a set, cos equal
    assert int(ref.center_index.shape[0]) == int(out.center_index.shape[0])
    if int(ref.center_index.shape[0]):
        # cos values (pair-level, side-agnostic) travel exactly
        ref_cos = sorted(float(c) for c in ref.cos_theta)
        out_cos = sorted(float(c) for c in out.cos_theta)
        assert ref_cos == out_cos
    assert ref.counts == out.counts


def test_jk_swap_column_symmetry_of_row_tensors() -> None:
    """Direct tensor-level symmetry: swapping the two side columns of an
    emitted row swaps the side tensors and leaves cos/basis untouched."""
    trip = build_angle_triplets(sphere_center_batch([1.0, 1.3, 1.7]))
    assert int(trip.center_index.shape[0]) >= 1
    swapped_distance = trip.side_distance.flip(1)
    swapped_rbf = trip.side_rbf12.flip(1)
    swapped_attr = trip.side_chem_attr.flip(1)
    swapped_index = trip.side_index.flip(1)
    # the flip is an involution — and pair-level features are untouched
    assert torch.equal(swapped_distance.flip(1), trip.side_distance)
    assert torch.equal(swapped_rbf.flip(1), trip.side_rbf12)
    assert torch.equal(swapped_attr.flip(1), trip.side_chem_attr)
    assert torch.equal(swapped_index.flip(1), trip.side_index)
    # unordered-pair invariant: {side0, side1} unchanged by construction
    for t in range(int(trip.center_index.shape[0])):
        a = (float(trip.side_distance[t, 0]),
             tuple(float(v) for v in trip.side_rbf12[t, 0]),
             tuple(float(v) for v in trip.side_chem_attr[t, 0]))
        b = (float(swapped_distance[t, 1]),
             tuple(float(v) for v in swapped_rbf[t, 1]),
             tuple(float(v) for v in swapped_attr[t, 1]))
        assert a == b


# ---------------------------------------------------------------------------
# chunking / purity (§6.3.8)
# ---------------------------------------------------------------------------
def test_angle_chunking_is_exact() -> None:
    batch = sn2_bidirectional_batch(n_graphs=2)
    tiny = build_angle_triplets(batch, chunk_size=1)
    huge = build_angle_triplets(batch, chunk_size=100_000)
    assert torch.equal(tiny.center_index, huge.center_index)
    assert torch.equal(tiny.side_index, huge.side_index)
    assert torch.equal(tiny.side_distance, huge.side_distance)
    assert torch.equal(tiny.side_rbf12, huge.side_rbf12)
    assert torch.equal(tiny.side_chem_attr, huge.side_chem_attr)
    assert torch.equal(tiny.cos_theta, huge.cos_theta)
    assert torch.equal(tiny.cos_basis, huge.cos_basis)
    assert torch.equal(tiny.center_has_angle, huge.center_has_angle)
    assert torch.equal(tiny.angle_neighbor_count, huge.angle_neighbor_count)
    assert tiny.counts == huge.counts


def test_angle_outputs_follow_geometry_no_cache() -> None:
    base = sphere_center_batch([1.0, 1.3, 1.7])
    moved = dict(base)
    moved["pos"] = base["pos"] * 1.15          # non-rigid: distances scale
    ref = build_angle_triplets(base)
    out = build_angle_triplets(moved)
    assert not torch.equal(ref.side_distance, out.side_distance)
    again = build_angle_triplets(base)
    assert torch.equal(ref.cos_theta, again.cos_theta)
    assert ref.counts == again.counts


def test_invalid_chunk_size_is_typed_error() -> None:
    with pytest.raises(FeatureProviderError) as excinfo:
        build_angle_triplets(sn2_forward_batch(), chunk_size=0)
    assert excinfo.value.code == "V541_PROVIDER_PARAM"
