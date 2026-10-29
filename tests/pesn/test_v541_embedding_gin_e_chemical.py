"""V5.4.1 WP3b — E1 ``gin_e_chemical`` candidate tests (plan §6.7 E1,
§6.6.3/§6.6.4, §6.8 rows 2/3/4/6).

Covered here (candidate-level; the model-level gate/parity battery lives
in ``test_v541_embedding_chemical_model.py``):

* the frozen §6.7 formula — hand-computed small graphs (bonded pair,
  3-atom chain, isolated node, single atom) plus an INDEPENDENT literal
  loop over a random-weight module;
* ``epsilon = 0`` FIXED (a constant, never a parameter) and the explicit
  self term: an isolated node aggregates exactly ``MLP(h)``;
* zero-neighbour / single-neighbour / multi-neighbour finiteness and
  finite gradients;
* §6.8 row 2 — message direction never flips the signed bond delta:
  bitwise-identical attrs in both view directions, direction-flip
  invariance of the delta, duplicate-edge merge equivalence and typed
  refusal of an inconsistent duplicate;
* E(3)/permutation invariance, independent-vs-mixed batch consistency,
  F2 discipline (the masked slots are never read, the input batch is
  never mutated) and the ``diagnostics`` switch (no count-only host sync
  on the cheap path).

Registry discipline: the E1 module is imported INSIDE an autouse fixture
(its import registers ``E1``) and the registry is restored afterwards, so
nothing leaks into the WP2 registry tests running in the same session.
"""

from __future__ import annotations

import math

import pytest
import torch

from pes2ts_core.pesn.models.v541.embedding_registry import (
    build_embedding,
    embedding_spec,
    is_embedding_implemented,
    registry_restore,
    registry_snapshot,
)
from pes2ts_core.pesn.models.v541.feature_providers import build_chem_graph
from pes2ts_core.pesn.models.v541.interfaces import (
    HIDDEN_V541,
    normalize_branch_output,
)
from tests.pesn.test_v541_embedding_chemical_fixtures import (
    assert_all_floats,
    assert_close,
    bonded_pair_batch,
    chain_batch,
    chemical_batch_matrix,
    controlled_gine,
    duplicate_first_edge,
    flip_edge_directions,
    isolated_plus_bonded_batch,
    load_e1,
    no_neighbor_batch,
    patch_count_only_helpers,
    poison_first_edge_attr,
    random_h,
    sn2_bidirectional_batch,
    sn2_forward_batch,
    single_atom_batch,
    sphere_center_batch,
)
from tests.pesn.test_v541_feature_providers_fixtures import (
    forward_batch,
    permute_forward_batch,
    random_rotation,
    reflection_matrix,
    transform_forward_batch,
)

SEED = 11


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def e1():
    """Import (⇒ register E1) inside the test, restore afterwards."""
    snap = registry_snapshot()
    module = load_e1()
    yield module
    registry_restore(snap)


def _chem_pairs(view) -> set[tuple[int, int]]:
    return {(int(view.pair_index[0, p]), int(view.pair_index[1, p]))
            for p in range(int(view.pair_index.shape[1]))}


# ---------------------------------------------------------------------------
# construction / registry contract
# ---------------------------------------------------------------------------
def test_e1_shapes_registry_and_frozen_epsilon(e1) -> None:
    mod = e1.GINEChemicalEmbedding()
    assert mod.width == HIDDEN_V541 == 48
    # P_e: linear 8 → 48; MLP: 48 → 48 → 48 (ReLU middle, linear out)
    assert tuple(mod.pe_proj.weight.shape) == (48, 8)
    assert tuple(mod.out_mlp[0].weight.shape) == (48, 48)
    assert tuple(mod.out_mlp[2].weight.shape) == (48, 48)
    assert isinstance(mod.out_mlp[1], torch.nn.ReLU)
    # epsilon = 0 FIXED: a constant, never a parameter, never trainable
    assert mod.eps == 0.0
    assert all("eps" not in name for name, _ in mod.named_parameters())
    # no dropout / BatchNorm / pooling anywhere (plan §6.7 公共项)
    forbidden = (torch.nn.Dropout, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                 torch.nn.AvgPool1d, torch.nn.MaxPool1d, torch.nn.AvgPool2d,
                 torch.nn.MaxPool2d)
    assert not any(isinstance(m, forbidden) for m in mod.modules())
    # param count: P_e 9w + MLP 2(w²+w) = 2w² + 11w
    w = HIDDEN_V541
    assert sum(p.numel() for p in mod.parameters()) == 2 * w * w + 11 * w
    # the import registered the REAL E1 builder (WP3 path)
    assert is_embedding_implemented("E1") is True
    built = build_embedding("E1")
    assert isinstance(built, e1.GINEChemicalEmbedding)
    spec = embedding_spec("E1")
    assert spec.family == "gin_e_chemical"
    assert spec.output == "residual_node_delta"
    assert spec.in_main_matrix is True
    # diagnostics switch defaults to the frozen WP3a default
    assert mod.diagnostics_enabled is True
    assert mod.last_diagnostics == {}
    # width parameter is free (E5 will build the same class at 24)
    narrow = e1.GINEChemicalEmbedding(width=24)
    assert narrow.width == 24
    assert tuple(narrow.pe_proj.weight.shape) == (24, 8)


# ---------------------------------------------------------------------------
# hand-computed small graphs (§6.7 E1 formula)
# ---------------------------------------------------------------------------
def test_hand_computed_message_bonded_pair(e1) -> None:
    """2 atoms, 1 chem pair: ``f_i = ReLU(h_i + ReLU(h_j)) = h_i + h_j``."""
    batch = bonded_pair_batch(1.0)
    mod = e1.GINEChemicalEmbedding()
    controlled_gine(mod)
    h = random_h(2, SEED)
    with torch.no_grad():
        f = mod(h, batch)
    assert_close(f[0], h[0] + h[1], "bonded f0")
    assert_close(f[1], h[1] + h[0], "bonded f1")
    # the hand-test premise: exactly one chem pair
    view = build_chem_graph(batch)
    assert _chem_pairs(view) == {(0, 1)}
    assert tuple(view.pair_index.shape) == (2, 1)


def test_hand_computed_message_chain_against_literal_loop(e1) -> None:
    """3-atom chain vs an INDEPENDENT literal loop of the §6.7 formula."""
    batch = chain_batch()
    mod = e1.GINEChemicalEmbedding()
    controlled_gine(mod)
    h = random_h(3, SEED)
    view = build_chem_graph(batch)
    assert _chem_pairs(view) == {(0, 1), (1, 2)}, "hand-test premise"
    with torch.no_grad():
        f = mod(h, batch)
    # literal: f_i = ReLU(h_i + Σ_j ReLU(h_j)) with P_e = 0
    assert_close(f[0], h[0] + h[1], "chain f0")
    assert_close(f[1], h[1] + h[0] + h[2], "chain f1")
    assert_close(f[2], h[2] + h[1], "chain f2")


def test_literal_loop_matches_random_weights(e1) -> None:
    """Random-init module vs an independent literal loop over the view."""
    batch = sn2_bidirectional_batch(n_graphs=2)
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    mod = e1.GINEChemicalEmbedding()
    view = build_chem_graph(batch)
    # literal reference of §6.7 with the module's OWN pe_proj/MLP
    pe = mod.pe_proj(view.edge_attr.to(h.dtype))
    expected = torch.zeros_like(h)
    for e in range(int(view.edge_index.shape[1])):
        receiver = int(view.edge_index[0, e])
        neighbour = int(view.edge_index[1, e])
        expected[receiver] += torch.relu(h[neighbour] + pe[e])
    expected = mod.out_mlp(h + expected)          # (1+0)·h + Σ_j …
    with torch.no_grad():
        f = mod(h, batch)
    assert_close(f, expected, "E1 vs literal loop")


def test_isolated_node_aggregates_self_only(e1) -> None:
    """A node with no chem neighbour sums EXACTLY ``MLP(h)`` (no graph
    self-loop is ever added — plan §6.7 E1)."""
    batch = isolated_plus_bonded_batch()
    mod = e1.GINEChemicalEmbedding()
    controlled_gine(mod)
    h = random_h(3, SEED)
    with torch.no_grad():
        f = mod(h, batch)
    # node 2 is isolated: only the self term survives
    assert_close(f[2], torch.relu(h[2]), "isolated self-only")
    assert_close(f[0], h[0] + h[1], "bonded node0")
    diag = mod.last_diagnostics
    assert diag["zero_neighbor"] == 1.0
    assert diag["n_edges"] == 2.0          # one pair, bidirectional
    # fully empty view: single atom and the no-neighbour pair
    for batch2, n_nodes in ((single_atom_batch(), 1), (no_neighbor_batch(), 2)):
        h2 = random_h(n_nodes, SEED)
        mod2 = e1.GINEChemicalEmbedding()
        controlled_gine(mod2)
        with torch.no_grad():
            out = mod2(h2, batch2)
        assert_close(out, torch.relu(h2), "empty-view f = ReLU(h)")
        assert mod2.last_diagnostics["n_edges"] == 0.0
        assert mod2.last_diagnostics["zero_neighbor"] == float(n_nodes)


def test_multi_neighbor_degree_counts(e1) -> None:
    """Sphere centre with 3 chem neighbours: 6 directed edges, zero
    isolated nodes; the centre aggregates all three neighbours."""
    batch = sphere_center_batch([1.0, 1.0, 1.0])
    mod = e1.GINEChemicalEmbedding()
    controlled_gine(mod)
    h = random_h(4, SEED)
    view = build_chem_graph(batch)
    assert _chem_pairs(view) == {(0, 1), (0, 2), (0, 3)}
    with torch.no_grad():
        f = mod(h, batch)
    assert_close(f[0], h[0] + h[1] + h[2] + h[3], "centre sum")
    assert_close(f[1], h[1] + h[0], "leaf sum")
    diag = mod.last_diagnostics
    assert diag["n_edges"] == 6.0
    assert diag["zero_neighbor"] == 0.0


# ---------------------------------------------------------------------------
# §6.8 row 2: direction never flips the bond delta; duplicates merge
# ---------------------------------------------------------------------------
def test_view_directions_carry_identical_signed_attrs(e1) -> None:
    """Both message directions of every pair carry bitwise-identical
    attrs — the SIGNED bond delta is never negated by direction (§6.6.3)."""
    batch = sn2_forward_batch()
    view = build_chem_graph(batch)
    n_pairs = int(view.pair_index.shape[1])
    assert n_pairs > 0
    fwd = view.edge_attr[0::2]
    rev = view.edge_attr[1::2]
    assert torch.equal(fwd, rev)               # bitwise, incl. delta field
    # the SN2 fixture has a real edit: delta_b_intent is NOT all zero
    delta_idx = 2                              # chem_edge_fields[2]
    assert float(view.pair_attr[:, delta_idx].abs().max()) > 0.0


def test_direction_flip_leaves_delta_unchanged(e1) -> None:
    """Reversing every message direction leaves the delta BITWISE equal:
    the view is rebuilt from a symmetric edge set with identical attrs."""
    batch = sn2_forward_batch()
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    mod = e1.GINEChemicalEmbedding()
    flipped = flip_edge_directions(batch)
    with torch.no_grad():
        a = mod(h, batch)
        b = mod(h, flipped)
    assert torch.equal(a, b)
    # and the signed delta itself is unchanged under the flip
    view_a = build_chem_graph(batch)
    view_b = build_chem_graph(flipped)
    assert torch.equal(view_a.pair_attr, view_b.pair_attr)


def test_duplicate_edge_merges_and_inconsistent_duplicate_refuses(e1) -> None:
    """Bitwise-equal duplicates merge (identical delta); an inconsistent
    duplicate is a typed provider refusal that E1 propagates."""
    batch = sn2_forward_batch()
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    mod = e1.GINEChemicalEmbedding()
    with torch.no_grad():
        base = mod(h, batch)
        merged = mod(h, duplicate_first_edge(batch))
    assert torch.equal(base, merged)
    with pytest.raises(Exception) as exc:
        mod(h, poison_first_edge_attr(batch))
    assert getattr(exc.value, "code", "") == "V541_PROVIDER_EDGE_ATTR_MISMATCH"


# ---------------------------------------------------------------------------
# E(3)/permutation invariance + F2 discipline (§6.4 / §6.6.1 / §6.8)
# ---------------------------------------------------------------------------
def test_translation_rotation_reflection_invariance(e1) -> None:
    """E1 reads NO geometry: the delta is invariant under any pos move."""
    batch = sn2_bidirectional_batch(n_graphs=2)
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    mod = e1.GINEChemicalEmbedding()
    with torch.no_grad():
        ref = mod(h, batch)
        shifted = mod(h, transform_forward_batch(
            batch, translate=torch.tensor([12.5, -3.25, 7.75])))
        rotated = mod(h, transform_forward_batch(
            batch, rotate=random_rotation(0)))
        reflected = mod(h, transform_forward_batch(
            batch, rotate=reflection_matrix(2)))
        both = mod(h, transform_forward_batch(
            batch, rotate=random_rotation(3) @ reflection_matrix(0),
            translate=torch.tensor([1.0, 2.0, 3.0])))
    for label, moved in (("translation", shifted), ("rotation", rotated),
                         ("reflection", reflected), ("rot+refl+trans", both)):
        assert torch.equal(ref, moved), f"E1 {label} must be bitwise"


def test_permutation_equivariance(e1) -> None:
    batch = sn2_forward_batch()
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    gen = torch.Generator().manual_seed(17)
    perm = torch.randperm(n, generator=gen)
    permuted = permute_forward_batch(batch, perm)
    mod = e1.GINEChemicalEmbedding()
    with torch.no_grad():
        ref = mod(h, batch)
        out = mod(h[perm], permuted)
    assert_close(ref[perm], out, "E1 permutation")


def test_f2_masked_slots_never_read_and_input_untouched(e1) -> None:
    """Garbage in the F2-masked first 12 ``edge_attr`` slots changes
    NOTHING (E1 reads only the 8 chem fields), and forward never mutates
    the input batch."""
    batch = sn2_forward_batch()
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    mod = e1.GINEChemicalEmbedding()
    before = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
              for k, v in batch.items()}
    with torch.no_grad():
        base = mod(h, batch)
        poisoned = dict(batch)
        attr = batch["edge_attr"].clone()
        attr[:, :12] = 1234.5
        poisoned["edge_attr"] = attr
        out = mod(h, poisoned)
    assert torch.equal(base, out)              # bitwise: slots unread
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, before[key]), key


# ---------------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------------
def test_gradients_finite_over_matrix(e1) -> None:
    for name, batch in chemical_batch_matrix().items():
        n = int(batch["pos"].shape[0])
        h = random_h(n, SEED).requires_grad_(True)
        mod = e1.GINEChemicalEmbedding()
        out = mod(h, batch)
        assert bool(torch.isfinite(out).all()), name
        out.sum().backward()
        assert h.grad is not None and bool(torch.isfinite(h.grad).all()), name
        for pname, p in mod.named_parameters():
            assert p.grad is not None, f"{name}:{pname}"
            assert bool(torch.isfinite(p.grad).all()), f"{name}:{pname}"
        # the delta is a legal branch output for every fixture
        normalize_branch_output(
            out.detach(), n_nodes=n, hidden=HIDDEN_V541,
            ref_dtype=torch.float32)


# ---------------------------------------------------------------------------
# independent vs mixed batch (§6.8 row 6)
# ---------------------------------------------------------------------------
def test_independent_vs_mixed_batch(e1) -> None:
    from scripts.agent_ops.feature_gate.v3_fixtures import make_sn2_sample

    batch_a = forward_batch([make_sn2_sample("forward")])
    batch_b = forward_batch([make_sn2_sample("reverse", jitter_seed=7)])
    batch_ab = forward_batch([
        make_sn2_sample("forward"), make_sn2_sample("reverse", jitter_seed=7)])
    n_a = int(batch_a["pos"].shape[0])
    n_b = int(batch_b["pos"].shape[0])
    torch.manual_seed(SEED)
    h_a = torch.randn(n_a, 48)
    h_b = torch.randn(n_b, 48)
    mod = e1.GINEChemicalEmbedding()
    with torch.no_grad():
        alone = mod(h_a, batch_a)
        alone_b = mod(h_b, batch_b)
        mixed = mod(torch.cat([h_a, h_b]), batch_ab)
    assert int(batch_ab["graph_index"][:n_a].max()) == 0
    assert_close(alone, mixed[:n_a], "E1 independent-vs-mixed (A)")
    assert_close(alone_b, mixed[n_a:n_a + n_b], "E1 independent-vs-mixed (B)")


# ---------------------------------------------------------------------------
# diagnostics switch (WP3a pattern)
# ---------------------------------------------------------------------------
def test_diagnostics_false_skips_count_only_helpers(
        monkeypatch: pytest.MonkeyPatch, e1) -> None:
    batch = sn2_bidirectional_batch(n_graphs=2)
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    cheap = e1.GINEChemicalEmbedding(diagnostics_enabled=False)
    patch_count_only_helpers(monkeypatch)          # helpers now RAISE
    with torch.no_grad():
        out = cheap(h, batch)                      # must not raise
    assert bool(torch.isfinite(out).all())
    assert cheap.last_diagnostics == {}            # no counts at all

    # control: the default path really routes through those helpers
    loud = e1.GINEChemicalEmbedding(diagnostics_enabled=True)
    with pytest.raises(AssertionError):
        with torch.no_grad():
            loud(h, batch)


def test_diagnostics_off_keeps_features_bitwise_identical(e1) -> None:
    batch = sn2_bidirectional_batch(n_graphs=2)
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    on = e1.GINEChemicalEmbedding(diagnostics_enabled=True)
    off = e1.GINEChemicalEmbedding(diagnostics_enabled=False)
    off.load_state_dict(on.state_dict())
    with torch.no_grad():
        a = on(h, batch)
        b = off(h, batch)
    assert torch.equal(a, b)
    assert on.last_diagnostics and off.last_diagnostics == {}


def test_diagnostics_are_detached_floats_and_copied(e1) -> None:
    batch = sn2_bidirectional_batch(n_graphs=2)
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    mod = e1.GINEChemicalEmbedding()
    with torch.no_grad():
        mod(h, batch)
    diag = mod.last_diagnostics
    assert_all_floats(diag, "E1.last_diagnostics")
    for key in ("n_nodes", "n_edges", "zero_neighbor", "zero_neighbor_frac"):
        assert key in diag, key
    assert diag["n_atoms"] == float(n)
    assert diag["n_chem_edges"] == diag["n_edges"]
    assert 0.0 <= diag["zero_neighbor_frac"] <= 1.0
    # a returned snapshot never aliases live state
    snapshot = mod.last_diagnostics
    snapshot["n_edges"] = -1.0
    assert mod.last_diagnostics["n_edges"] != -1.0
    # disabled ⇒ empty (and no tensor anywhere)
    mod.diagnostics_enabled = False
    with torch.no_grad():
        mod(h, batch)
    assert mod.last_diagnostics == {}
