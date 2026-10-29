"""V5.4.1 WP3a — feature-provider CONTRACT tests (plan §6.4/§6.6/§6.8).

Freezes the schema the WP3b branches build against:

* F2 pack = 17 slots with the first 12 (RBF) MASKED, legacy edge input =
  22 channels;
* config ``feature_recipe`` (chem_edge_fields / rbf12 / angle policy)
  equals the provider constants — drift fails closed;
* distances are physical Å recomputed from ``pos``; the F2-masked slots
  are never read (mask unchanged ⇒ outputs unchanged, geometry change ⇒
  RBF changes);
* bidirectional edges carry identical attributes (message direction never
  flips the signed delta) and the forward/reverse intent hand
  expectations of §6.6.4;
* duplicate merge / mismatch / cross-graph / self-loop / missing-key /
  non-finite refusals are typed errors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from pes2ts_core.pesn.contracts.key_atom_v3 import (
    EDGE_ABS_DELTA_KEY,
    EDGE_ATTR_KEY,
    EDGE_B_SRC_KEY,
    EDGE_B_TGT_KEY,
    EDGE_DELTA_B_KEY,
    EDGE_INDEX_KEY,
    EDGE_INPUT_DIM_V3,
    EDGE_IS_CHANGED_KEY,
    EDGE_PACK_DIM_V3,
    EDGE_PACK_SLOTS_V3,
    edge_pack_slot_mask_v3,
)
from pes2ts_core.pesn.features.key_atom_spec_v3 import gaussian_rbf_v3
from pes2ts_core.pesn.models.v541.feature_providers import (
    CHEM_EDGE_DIM,
    CHEM_EDGE_FIELDS,
    FeatureProviderError,
    build_chem_graph,
    build_legal_neighbor_views,
    build_angle_triplets,
    build_spatial_graph,
    rbf12,
)
from tests.pesn.test_v541_feature_providers_fixtures import (
    bonded_pair_batch,
    sn2_bidirectional_batch,
    sn2_forward_batch,
    sn2_reverse_batch,
    transform_forward_batch,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_YAML = PROJECT_ROOT / "config/campaigns/pes2ts_v541_embed_50k.yaml"

_CHEM = {name: i for i, name in enumerate(CHEM_EDGE_FIELDS)}


def _pair_row(view, a: int, b: int) -> tuple[int, int]:
    """Column pair (2p, 2p+1) of the undirected pair ``(a, b)``."""
    key = (min(a, b), max(a, b))
    for p in range(int(view.pair_index.shape[1])):
        pair = (int(view.pair_index[0, p]), int(view.pair_index[1, p]))
        if pair == key:
            return 2 * p, 2 * p + 1
    raise AssertionError(f"pair {key} not in chem view")


# ---------------------------------------------------------------------------
# frozen schema: 17 slots / 22 channels / F2 mask
# ---------------------------------------------------------------------------
def test_f2_pack_is_17_slots_with_12_masked_rbf() -> None:
    assert EDGE_PACK_DIM_V3 == 17
    assert EDGE_INPUT_DIM_V3 == 17 + 5 == 22
    assert EDGE_PACK_SLOTS_V3["is_spatial"] == 12
    assert EDGE_PACK_SLOTS_V3["covalent_union"] == 13
    assert EDGE_PACK_SLOTS_V3["aromatic_union"] == 14
    assert EDGE_PACK_SLOTS_V3["b_min"] == 15
    assert EDGE_PACK_SLOTS_V3["b_max"] == 16
    mask_f2 = edge_pack_slot_mask_v3("F2")
    assert mask_f2[:12].sum() == 0          # RBF slots masked at F2
    assert mask_f2[12:].tolist() == [1, 1, 1, 1, 1]
    mask_f3 = edge_pack_slot_mask_v3("F3")
    assert mask_f3[:12].sum() == 12         # RBF only at F3


def test_config_feature_recipe_matches_provider_constants() -> None:
    config = yaml.safe_load(CAMPAIGN_YAML.read_text(encoding="utf-8"))
    recipe = config["embedding_contract"]["feature_recipe"]
    assert list(recipe["chem_edge_fields"]) == list(CHEM_EDGE_FIELDS)
    assert CHEM_EDGE_DIM == 8 == len(recipe["chem_edge_fields"])
    rbf = recipe["rbf12"]
    assert int(rbf["n_bins"]) == 12
    assert float(rbf["center_start_angstrom"]) == 0.0
    assert float(rbf["center_end_angstrom"]) == 4.5
    assert float(rbf["sigma_angstrom"]) == 0.3
    angle = recipe["angle"]
    assert int(angle["max_neighbors_per_center"]) == 12
    assert int(angle["max_unordered_pairs_per_center"]) == 66
    assert angle["tie_policy"] == "reject_boundary_tie_shell"
    assert [str(x) for x in angle["basis"]] == ["1", "c", "c2", "c3"]
    assert list(angle["cos_clamp"]) == [-1.0, 1.0]
    assert angle["chirality_labels"] is False


def test_rbf12_matches_frozen_numpy_definition() -> None:
    """Provider math ≡ ``gaussian_rbf_v3`` (spec v3 / config recipe)."""
    distances = torch.tensor([0.0, 0.25, 1.0, 1.5, 3.0, 4.5, 4.4999])
    ours = rbf12(distances).numpy()
    reference = gaussian_rbf_v3(distances.numpy())
    assert ours.shape == (7, 12)
    np.testing.assert_allclose(ours, reference, rtol=1e-6, atol=1e-7)
    # frozen centers: uniform 0–4.5, sigma 0.3
    centers = np.linspace(0.0, 4.5, 12)
    manual = np.exp(-0.5 * ((distances.numpy()[:, None] - centers) / 0.3) ** 2)
    np.testing.assert_allclose(ours, manual, rtol=1e-6, atol=1e-7)


def test_chem_view_schema_and_direction_layout() -> None:
    batch = sn2_forward_batch()
    view = build_chem_graph(batch)
    n_pairs = int(view.pair_index.shape[1])
    assert n_pairs == 5                       # SN2: 4 C–H + C–Br + C–O
    assert tuple(view.pair_attr.shape) == (n_pairs, CHEM_EDGE_DIM)
    assert tuple(view.edge_index.shape) == (2, 2 * n_pairs)
    assert tuple(view.edge_attr.shape) == (2 * n_pairs, CHEM_EDGE_DIM)
    assert tuple(view.rbf12.shape) == (2 * n_pairs, 12)
    assert view.valid.dtype == torch.bool
    assert bool(view.valid.all())
    # legacy layout: 2p = a→b (a<b), 2p+1 = b→a; identical attrs.
    for p in range(n_pairs):
        a = int(view.pair_index[0, p])
        b = int(view.pair_index[1, p])
        assert a < b
        assert (int(view.edge_index[0, 2 * p]), int(view.edge_index[1, 2 * p])) == (a, b)
        assert (int(view.edge_index[0, 2 * p + 1]), int(view.edge_index[1, 2 * p + 1])) == (b, a)
        assert torch.equal(view.edge_attr[2 * p], view.edge_attr[2 * p + 1])
        assert float(view.distance[2 * p]) == float(view.distance[2 * p + 1])
        assert torch.equal(view.rbf12[2 * p], view.rbf12[2 * p + 1])


def test_distances_are_physical_angstrom_from_pos() -> None:
    batch = sn2_forward_batch()
    view = build_chem_graph(batch)
    pos = batch["pos"].numpy().astype(np.float64)
    for p in range(int(view.pair_index.shape[1])):
        a = int(view.pair_index[0, p])
        b = int(view.pair_index[1, p])
        expected = float(np.linalg.norm(pos[a] - pos[b]))
        assert abs(float(view.pair_distance[p]) - expected) < 1e-5
        # NOT read from the F2-masked pack slots: those are all zero.
    assert bool((batch[EDGE_ATTR_KEY][:, :12] == 0).all())


# ---------------------------------------------------------------------------
# F2 mask unchanged / geometry sensitivity (§6.8 row 1)
# ---------------------------------------------------------------------------
def test_f2_mask_unchanged_but_current_rbf_changes() -> None:
    """Two fixtures with the SAME F2 topology/mask but different current
    distances: every legacy edge tensor is bit-identical while the
    provider's recomputed distance/RBF changes (§6.8 acceptance)."""
    base = sn2_forward_batch()
    moved = transform_forward_batch(base, scale=1.2)
    for key in (EDGE_INDEX_KEY, EDGE_ATTR_KEY, EDGE_B_SRC_KEY, EDGE_B_TGT_KEY,
                EDGE_DELTA_B_KEY, EDGE_ABS_DELTA_KEY, EDGE_IS_CHANGED_KEY):
        assert torch.equal(base[key], moved[key]), key
    assert bool((base[EDGE_ATTR_KEY][:, :12] == 0).all())
    assert bool((moved[EDGE_ATTR_KEY][:, :12] == 0).all())

    view_a = build_chem_graph(base)
    view_b = build_chem_graph(moved)
    assert torch.equal(view_a.pair_index, view_b.pair_index)
    assert not torch.equal(view_a.pair_distance, view_b.pair_distance)
    assert not torch.equal(view_a.rbf12, view_b.rbf12)
    # chem ATTRS are intent/topology, not geometry — unchanged.
    assert torch.equal(view_a.pair_attr, view_b.pair_attr)
    # geometry change is visible in RBF (mask did not freeze it away)
    assert float((view_a.rbf12 - view_b.rbf12).abs().max()) > 1e-3


def test_masked_rbf_slots_are_never_read() -> None:
    """Garbage in the F2-masked first 12 pack slots changes NOTHING."""
    base = sn2_forward_batch()
    poisoned = dict(base)
    generator = torch.Generator().manual_seed(0)
    poisoned[EDGE_ATTR_KEY] = (
        base[EDGE_ATTR_KEY].clone())
    poisoned[EDGE_ATTR_KEY][:, :12] = torch.randn(
        base[EDGE_ATTR_KEY][:, :12].shape, generator=generator) * 7.0
    assert not torch.equal(base[EDGE_ATTR_KEY], poisoned[EDGE_ATTR_KEY])

    ref = build_legal_neighbor_views(base)
    out = build_legal_neighbor_views(poisoned)
    assert torch.equal(ref.chem.edge_index, out.chem.edge_index)
    assert torch.equal(ref.chem.edge_attr, out.chem.edge_attr)
    assert torch.equal(ref.chem.rbf12, out.chem.rbf12)
    assert torch.equal(ref.chem.distance, out.chem.distance)
    assert torch.equal(ref.spatial.edge_index, out.spatial.edge_index)
    assert torch.equal(ref.spatial.rbf12, out.spatial.rbf12)
    ref_trip = build_angle_triplets(base)
    out_trip = build_angle_triplets(poisoned)
    assert torch.equal(ref_trip.cos_theta, out_trip.cos_theta)
    assert torch.equal(ref_trip.side_chem_attr, out_trip.side_chem_attr)


# ---------------------------------------------------------------------------
# intent semantics: direction invariance + reversal hand expectations
# ---------------------------------------------------------------------------
def test_bidirectional_edges_never_flip_bond_delta() -> None:
    """Message direction i→j vs j→i carries the SAME signed delta; only a
    reversed INTENT changes it (plan §6.6.3)."""
    view = build_chem_graph(sn2_forward_batch())
    delta = view.edge_attr[:, _CHEM["delta_b_intent"]]
    b_src = view.edge_attr[:, _CHEM["b_src"]]
    b_tgt = view.edge_attr[:, _CHEM["b_tgt"]]
    assert torch.equal(delta[0::2], delta[1::2])
    assert torch.equal(b_src[0::2], b_src[1::2])
    assert torch.equal(b_tgt[0::2], b_tgt[1::2])
    # every chem pair is an edit or a covalent bond with delta == b_tgt-b_src
    for p in range(int(view.pair_index.shape[1])):
        assert abs(float(delta[2 * p])
                   - (float(b_tgt[2 * p]) - float(b_src[2 * p]))) < 1e-6


def test_intent_reversal_hand_expectations() -> None:
    """§6.6.4 hand fixture: breaking/edit fields under forward vs reverse
    intent (SN2 C–Br breaks, C–O forms, C–H spectators)."""
    forward = build_chem_graph(sn2_forward_batch())
    reverse = build_chem_graph(sn2_reverse_batch())
    assert torch.equal(forward.pair_index, reverse.pair_index)

    third = 1.0 / 3.0                       # legacy BOND_SCALE = 3
    # breaking bond (0, 1): C–Br present in G_src only
    fwd_br = _pair_row(forward, 0, 1)
    rev_br = _pair_row(reverse, 0, 1)
    for col in fwd_br:
        row = forward.edge_attr[col]
        assert abs(float(row[_CHEM["b_src"]]) - third) < 1e-6
        assert abs(float(row[_CHEM["b_tgt"]]) - 0.0) < 1e-6
        assert abs(float(row[_CHEM["delta_b_intent"]]) - (-third)) < 1e-6
        assert abs(float(row[_CHEM["abs_delta"]]) - third) < 1e-6
        assert float(row[_CHEM["is_changed"]]) == 1.0
        assert float(row[_CHEM["covalent_union"]]) == 1.0
        assert abs(float(row[_CHEM["b_min"]]) - 0.0) < 1e-6
        assert abs(float(row[_CHEM["b_max"]]) - third) < 1e-6
    for col in rev_br:
        row = reverse.edge_attr[col]
        assert abs(float(row[_CHEM["b_src"]]) - 0.0) < 1e-6
        assert abs(float(row[_CHEM["b_tgt"]]) - third) < 1e-6
        assert abs(float(row[_CHEM["delta_b_intent"]]) - third) < 1e-6   # sign flips
        assert abs(float(row[_CHEM["abs_delta"]]) - third) < 1e-6       # invariant
        assert float(row[_CHEM["is_changed"]]) == 1.0
        assert float(row[_CHEM["covalent_union"]]) == 1.0
        assert abs(float(row[_CHEM["b_min"]]) - 0.0) < 1e-6             # invariant
        assert abs(float(row[_CHEM["b_max"]]) - third) < 1e-6           # invariant

    # forming bond (0, 2): C–O present in G_tgt only — mirrored picture
    fwd_o = _pair_row(forward, 0, 2)
    rev_o = _pair_row(reverse, 0, 2)
    assert abs(float(forward.edge_attr[fwd_o[0]][_CHEM["delta_b_intent"]])
               - third) < 1e-6
    assert abs(float(reverse.edge_attr[rev_o[0]][_CHEM["delta_b_intent"]])
               - (-third)) < 1e-6

    # spectator bond (0, 3): identical under both intents (bitwise)
    fwd_h = _pair_row(forward, 0, 3)
    rev_h = _pair_row(reverse, 0, 3)
    assert torch.equal(forward.edge_attr[fwd_h[0]], reverse.edge_attr[rev_h[0]])
    assert torch.equal(forward.edge_attr[fwd_h[1]], reverse.edge_attr[rev_h[1]])


# ---------------------------------------------------------------------------
# duplicate merge / typed refusals (§6.8 row 2 + §6.2)
# ---------------------------------------------------------------------------
def _append_edge(batch: dict, column: int, *, delta_scale: float = 1.0,
                 flip_delta: bool = False) -> dict:
    out = dict(batch)
    out[EDGE_INDEX_KEY] = torch.cat(
        [batch[EDGE_INDEX_KEY], batch[EDGE_INDEX_KEY][:, [column]]], dim=1)
    out[EDGE_ATTR_KEY] = torch.cat(
        [batch[EDGE_ATTR_KEY], batch[EDGE_ATTR_KEY][[column]]], dim=0)
    for key in (EDGE_B_SRC_KEY, EDGE_B_TGT_KEY, EDGE_DELTA_B_KEY,
                EDGE_ABS_DELTA_KEY, EDGE_IS_CHANGED_KEY):
        value = batch[key][[column]]
        if flip_delta and key == EDGE_DELTA_B_KEY:
            value = -value
        out[key] = torch.cat([batch[key], value], dim=0)
    return out


def test_duplicate_directed_edges_merge_once() -> None:
    base = sn2_forward_batch()
    ref = build_chem_graph(base)
    assert ref.counts["n_duplicate_edges_merged"] == 30 - 15   # 15 pairs
    # append a third copy of column 0 (same pair, identical attributes)
    duplicated = _append_edge(base, 0)
    view = build_chem_graph(duplicated)
    assert int(view.pair_index.shape[1]) == int(ref.pair_index.shape[1])
    assert view.counts["n_input_edges"] == base[EDGE_INDEX_KEY].shape[1] + 1
    assert view.counts["n_duplicate_edges_merged"] == 30 + 1 - 15
    assert view.counts["n_chem_pairs"] == ref.counts["n_chem_pairs"]
    assert torch.equal(view.pair_attr, ref.pair_attr)
    assert torch.equal(view.pair_distance, ref.pair_distance)


def test_inconsistent_duplicate_attrs_are_typed_error() -> None:
    base = sn2_forward_batch()
    # append the REVERSE column of edge 0 with a flipped signed delta:
    # same undirected pair, inconsistent attribute ⇒ refuse (§6.6.4).
    bad = _append_edge(base, 1, flip_delta=True)
    with pytest.raises(FeatureProviderError) as excinfo:
        build_chem_graph(bad)
    assert excinfo.value.code == "V541_PROVIDER_EDGE_ATTR_MISMATCH"


def test_cross_graph_edge_is_typed_error() -> None:
    # bidirectional fixture: 2 graph families × fwd/rev = 4 samples × 6
    # atoms = 24 rows in 6-atom graph blocks
    batch = sn2_bidirectional_batch(n_graphs=2)
    n_total = int(batch["pos"].shape[0])
    assert n_total == 24
    bad = dict(batch)
    edge = batch[EDGE_INDEX_KEY].clone()
    edge[1, 0] = (int(edge[0, 0]) + 6) % n_total   # always another graph
    assert int(batch["graph_index"][edge[0, 0]]) != int(batch["graph_index"][edge[1, 0]])
    bad[EDGE_INDEX_KEY] = edge
    for builder in (build_chem_graph, build_spatial_graph, build_angle_triplets):
        with pytest.raises(FeatureProviderError) as excinfo:
            builder(bad)
        assert excinfo.value.code == "V541_PROVIDER_CROSS_GRAPH"


def test_self_loop_is_typed_error() -> None:
    batch = sn2_forward_batch()
    bad = dict(batch)
    edge = batch[EDGE_INDEX_KEY].clone()
    edge[1, 0] = edge[0, 0]
    bad[EDGE_INDEX_KEY] = edge
    with pytest.raises(FeatureProviderError) as excinfo:
        build_chem_graph(bad)
    assert excinfo.value.code == "V541_PROVIDER_SELF_LOOP"


def test_missing_forward_key_is_typed_error() -> None:
    batch = sn2_forward_batch()
    for key in (EDGE_B_SRC_KEY, EDGE_ATTR_KEY):
        bad = dict(batch)
        bad.pop(key)
        with pytest.raises(FeatureProviderError) as excinfo:
            build_chem_graph(bad)
        assert excinfo.value.code == "V541_PROVIDER_MISSING_KEY"


def test_nonfinite_pos_is_typed_error() -> None:
    batch = sn2_forward_batch()
    bad = dict(batch)
    pos = batch["pos"].clone()
    pos[2, 1] = float("nan")
    bad["pos"] = pos
    for builder in (build_chem_graph, build_spatial_graph, build_angle_triplets):
        with pytest.raises(FeatureProviderError) as excinfo:
            builder(bad)
        assert excinfo.value.code == "V541_PROVIDER_GEOM_NONFINITE"


def test_counts_are_detached_python_floats() -> None:
    views = build_legal_neighbor_views(sn2_forward_batch())
    triplets = build_angle_triplets(sn2_forward_batch())
    for view in (views.chem, views.spatial, triplets):
        assert view.counts, "counts must not be empty"
        for name, value in view.counts.items():
            assert isinstance(value, float), (name, type(value))
            assert not isinstance(value, torch.Tensor)
        # monotone sanity: atoms/edges are non-negative
        assert all(v >= 0.0 for v in view.counts.values())


def test_empty_views_are_well_shaped_single_atom() -> None:
    from tests.pesn.test_v541_feature_providers_fixtures import (
        single_atom_batch,
    )

    batch = single_atom_batch()
    views = build_legal_neighbor_views(batch)
    trip = build_angle_triplets(batch)
    for view in (views.chem, views.spatial):
        assert tuple(view.pair_index.shape) == (2, 0)
        assert tuple(view.edge_index.shape) == (2, 0)
        assert tuple(view.distance.shape) == (0,)
        assert tuple(view.rbf12.shape) == (0, 12)
        assert tuple(view.valid.shape) == (0,)
        assert view.valid.dtype == torch.bool
    assert tuple(trip.center_index.shape) == (0,)
    assert tuple(trip.side_index.shape) == (0, 2)
    assert tuple(trip.center_has_angle.shape) == (1,)
    assert not bool(trip.center_has_angle.any())
    assert trip.counts["n_triplets"] == 0.0
    assert trip.counts["n_centers_with_angle"] == 0.0


def test_bonded_pair_views() -> None:
    views = build_legal_neighbor_views(bonded_pair_batch(1.2))
    assert int(views.chem.pair_index.shape[1]) == 1
    assert int(views.spatial.pair_index.shape[1]) == 1
    assert abs(float(views.chem.pair_distance[0]) - 1.2) < 1e-5


def test_near_zero_bond_length_stays_finite() -> None:
    """§6.4 near-zero bond length: 1e-3 Å is legal for the distance
    views — finite distance/RBF/mask and finite gradients (only an
    EXACT zero is refused, and only by the angle provider)."""
    base = bonded_pair_batch(1e-3)
    views = build_legal_neighbor_views(base)
    assert abs(float(views.chem.pair_distance[0]) - 1e-3) < 1e-7
    assert torch.isfinite(views.chem.rbf12).all()
    assert bool(views.chem.valid.all())
    assert int(views.spatial.pair_index.shape[1]) == 1   # still < cutoff

    pos = base["pos"].detach().clone().requires_grad_(True)
    batch = dict(base)
    batch["pos"] = pos
    out = build_chem_graph(batch)
    out.rbf12.sum().backward()
    assert torch.isfinite(pos.grad).all()
