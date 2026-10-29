"""V5.4.1 WP3b — E5 ``dual_stream`` candidate tests (plan §6.7 E5,
registry family ``dual_stream`` / conditional row on E1/E4).

Covered here (candidate-level; the model-level gate/parity battery lives
in ``test_v541_embedding_dual_stream_model.py``):

* the frozen §6.7 E5 formula — exact shapes, exact parameter count, and
  HAND-COMPUTED fusion values on a 3-atom chain with controlled weights
  (chem block first, angle block last, fusion selectors per block);
* the two 48→24 input projections are INDEPENDENT parameters and the
  two 24-wide stream inputs never cross-contaminate; the internal
  GINE24/Angle24 branches are fresh width-24 rebuilds, not copies of a
  standalone E1/E4 construction;
* STRICT ``j/k`` unordered-pair symmetry is inherited bitwise from the
  Angle24 stream;
* zero-neighbour / zero-angle / degenerate-geometry finiteness and the
  typed ``V541_PROVIDER_ZERO_DISTANCE`` refusal;
* E(3)/permutation invariance, independent-vs-mixed batch consistency,
  F2 discipline (masked slots never read, input never mutated);
* the ``diagnostics`` switch — shared keys ``n_edges`` / ``n_triplets`` /
  ``zero_neighbor_frac`` / ``zero_angle_frac`` plus ``chem.`` / ``angle.``
  prefixed per-source provider counts, all detached floats;
* exactly ONE provider call per view per forward (chem view once for the
  GINE stream, angle view once for the Angle stream — no duplicated
  operator).

Registry discipline: the E5 module is imported INSIDE an autouse fixture
(its import registers ``E5``, and transitively the already-real E1/E4
modules) and the registry is restored afterwards, so nothing leaks into
the WP2 registry tests running in the same session.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from pes2ts_core.pesn.models.v541.embedding_registry import (
    build_embedding,
    embedding_spec,
    is_embedding_implemented,
    registry_restore,
    registry_snapshot,
)
from pes2ts_core.pesn.models.v541.feature_providers import (
    FeatureProviderError,
    build_angle_triplets,
    build_chem_graph,
)
from pes2ts_core.pesn.models.v541.interfaces import (
    HIDDEN_V541,
    normalize_branch_output,
)
from tests.pesn.test_v541_embedding_dual_stream_fixtures import (
    ABS_TOL,
    BRANCH_WIDTH,
    REL_TOL,
    assert_all_floats,
    assert_close,
    chain_batch,
    controlled_angle24,
    controlled_e5,
    controlled_fusion,
    controlled_gine24,
    controlled_projections,
    dual_stream_batch_matrix,
    load_e5,
    no_neighbor_batch,
    one_chem_one_spatial_batch,
    patch_count_only_helpers,
    random_h,
    sn2_bidirectional_batch,
    sn2_forward_batch,
    single_atom_batch,
    sphere_center_batch,
    swap_sides,
)
from tests.pesn.test_v541_feature_providers_fixtures import (
    forward_batch,
    permute_forward_batch,
    random_rotation,
    reflection_matrix,
    transform_forward_batch,
    zero_length_batch,
)

SEED = 31

#: Exact §6.7 E5 parameter count: two 48→24 projections (2·1176) +
#: GINE24 (1416) + Angle24 (5928) + fusion 48→48 (2352).
E5_PARAMS = 12_048


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def e5():
    """Import (⇒ register E5) inside the test, restore afterwards."""
    snap = registry_snapshot()
    module = load_e5()
    yield module
    registry_restore(snap)


def _chem_pairs(view) -> set[tuple[int, int]]:
    return {(int(view.pair_index[0, p]), int(view.pair_index[1, p]))
            for p in range(int(view.pair_index.shape[1]))}


def _shell3_batch() -> dict:
    return sphere_center_batch([1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# construction / registry contract
# ---------------------------------------------------------------------------
def test_e5_shapes_registry_and_param_count(e5) -> None:
    mod = e5.DualStreamEmbedding()
    assert mod.width == HIDDEN_V541 == 48
    assert mod.branch_width == BRANCH_WIDTH == 24
    # independent per-stream projections: 48 → 24
    assert tuple(mod.chem_in.weight.shape) == (24, 48)
    assert tuple(mod.angle_in.weight.shape) == (24, 48)
    # the streams REUSE the exact E1/E4 classes at width 24
    assert isinstance(mod.chem_branch, e5.GINEChemicalEmbedding)
    assert isinstance(mod.angle_branch, e5.AngleMPNNEmbedding)
    assert mod.chem_branch.width == 24
    assert mod.angle_branch.width == 24
    assert tuple(mod.chem_branch.pe_proj.weight.shape) == (24, 8)
    assert tuple(mod.angle_branch.pair_mlp[0].weight.shape) == (24, 60)
    assert tuple(mod.angle_branch.triplet_mlp[0].weight.shape) == (24, 76)
    # fusion: frozen concat order [f_c ‖ f_a] → 48, plain linear
    assert tuple(mod.fusion.weight.shape) == (48, 48)
    assert isinstance(mod.fusion, torch.nn.Linear)
    # no dropout / BatchNorm / pooling anywhere (plan §6.7 公共项)
    forbidden = (torch.nn.Dropout, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                 torch.nn.AvgPool1d, torch.nn.MaxPool1d, torch.nn.AvgPool2d,
                 torch.nn.MaxPool2d)
    assert not any(isinstance(m, forbidden) for m in mod.modules())
    # exact parameter count (§6.7 E5)
    assert sum(p.numel() for p in mod.parameters()) == E5_PARAMS
    assert sum(p.numel() for p in mod.parameters()) == (
        2 * (48 * 24 + 24)                                   # chem_in+angle_in
        + sum(p.numel() for p in mod.chem_branch.parameters())
        + sum(p.numel() for p in mod.angle_branch.parameters())
        + (48 * 48 + 48))                                    # fusion
    # NO internal gate: the single §5.2 W/alpha lives in V541Model only
    assert not hasattr(mod, "alpha")
    assert not hasattr(mod, "w_out")
    assert all("alpha" not in name and "gate" not in name
               for name, _ in mod.named_parameters())
    # the import registered the REAL E5 builder (WP3 path)
    assert is_embedding_implemented("E5") is True
    built = build_embedding("E5")
    assert isinstance(built, e5.DualStreamEmbedding)
    assert e5.EMBEDDING_ID_E5 == "E5"
    spec = embedding_spec("E5")
    assert spec.family == "dual_stream"
    assert spec.internal_view == "E1_and_E4_shared_inputs"
    assert spec.output == "residual_node_delta"
    assert spec.conditional_parent == ("E1", "E4")
    assert spec.in_main_matrix is False      # conditional launch row (§9.2)
    # diagnostics switch defaults to the frozen WP3a default
    assert mod.diagnostics_enabled is True
    assert mod.last_diagnostics == {}
    # width knobs are free (construction sanity only)
    narrow = e5.DualStreamEmbedding(branch_width=8)
    assert narrow.branch_width == 8
    assert tuple(narrow.chem_in.weight.shape) == (8, 48)
    assert tuple(narrow.fusion.weight.shape) == (48, 16)


# ---------------------------------------------------------------------------
# hand-computed fusion values (§6.7 E5 formula)
# ---------------------------------------------------------------------------
def test_hand_computed_chain_fusion(e5) -> None:
    """3-atom chain with fully controlled weights:

    ``h_c = h[:, :24]``, ``h_a = h[:, 24:]``; GINE24 gives
    ``f_c_i = h_c_i + Σ_j h_c_j``; Angle24 gives
    ``f_a_1 = h_a_0 + h_a_2`` (only centre 1 carries an angle); fusion
    = identity ⇒ ``delta = [f_c ‖ f_a]`` block-exact by hand.
    """
    batch = chain_batch()
    mod = e5.DualStreamEmbedding()
    controlled_e5(mod)
    h = random_h(3, SEED)                    # positive (rand + 0.5)
    h_c, h_a = h[:, :24], h[:, 24:]
    view = build_chem_graph(batch)
    assert _chem_pairs(view) == {(0, 1), (1, 2)}, "hand-test premise"
    aview = build_angle_triplets(batch)
    assert int(aview.center_index.shape[0]) == 1, "hand-test premise"
    assert int(aview.center_index[0]) == 1
    with torch.no_grad():
        f = mod(h, batch)
    expected0 = torch.cat((h_c[0] + h_c[1], torch.zeros(24)))
    expected1 = torch.cat((h_c[0] + h_c[1] + h_c[2], h_a[0] + h_a[2]))
    expected2 = torch.cat((h_c[1] + h_c[2], torch.zeros(24)))
    assert_close(f[0], expected0, "chain node0 = [f_c0 | 0]")
    assert_close(f[1], expected1, "chain node1 = [f_c1 | h_a0+h_a2]")
    assert_close(f[2], expected2, "chain node2 = [f_c2 | 0]")
    assert tuple(f.shape) == (3, HIDDEN_V541)


def test_hand_computed_shell3_angle_block(e5) -> None:
    """Sphere centre with 3 tied neighbours: C(3,2) = 3 triplets ⇒
    ``f_a_0 = 2·(h_a1 + h_a2 + h_a3)``; every leaf has no angle ⇒ its
    angle block is exactly 0 even with a BIASED fusion."""
    batch = _shell3_batch()
    mod = e5.DualStreamEmbedding()
    controlled_projections(mod)
    controlled_gine24(mod.chem_branch)
    controlled_angle24(mod.angle_branch)
    controlled_fusion(mod, block="angle")     # keep ONLY the angle block
    with torch.no_grad():
        mod.fusion.bias.copy_(torch.full((48,), 0.25))  # tempting constant
    h = random_h(4, SEED)
    h_a = h[:, 24:]
    with torch.no_grad():
        f = mod(h, batch)
    # leaves: angle stream masked to 0 ⇒ fusion output = the bias only
    for leaf in (1, 2, 3):
        assert_close(f[leaf], torch.full((48,), 0.25), f"leaf {leaf} bias-only")
    centre = 2.0 * (h_a[1] + h_a[2] + h_a[3])
    assert_close(f[0][24:], centre + 0.25, "shell3 centre angle block")
    assert float(f[0][:24].max()) == 0.25     # chem block masked out


def test_fusion_block_selectors_are_exact(e5) -> None:
    """Fusion selector ``chem`` zeroes the angle block, ``angle`` zeroes
    the chem block — the concat order [f_c ‖ f_a] is frozen and exact."""
    batch = chain_batch()
    h = random_h(3, SEED)
    views = (build_chem_graph(batch), build_angle_triplets(batch))
    mod = e5.DualStreamEmbedding()
    controlled_e5(mod)
    with torch.no_grad():
        f_c, f_a = mod.stream_deltas(h, *views)
    # selector = chem: keep the first block, zero the second
    ctrl = e5.DualStreamEmbedding()
    controlled_e5(ctrl)
    controlled_fusion(ctrl, block="chem")
    with torch.no_grad():
        out_chem = ctrl(h, batch)
    assert_close(out_chem[:, :24], f_c, "chem selector: chem half verbatim")
    assert float(out_chem[:, 24:].abs().max()) == 0.0
    # selector = angle: keep the second block, zero the first
    ctrl = e5.DualStreamEmbedding()
    controlled_e5(ctrl)
    controlled_fusion(ctrl, block="angle")
    with torch.no_grad():
        out_angle = ctrl(h, batch)
    assert_close(out_angle[:, 24:], f_a, "angle selector: angle half verbatim")
    assert float(out_angle[:, :24].abs().max()) == 0.0


# ---------------------------------------------------------------------------
# independent branch weights; 24-wide streams never mix
# ---------------------------------------------------------------------------
def test_branch_weights_are_independent(e5) -> None:
    """chem_in / angle_in are SEPARATE parameters; mutating one stream's
    projection never touches the other stream's output."""
    batch = sn2_forward_batch()
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    mod = e5.DualStreamEmbedding()
    assert mod.chem_in.weight.data_ptr() != mod.angle_in.weight.data_ptr()
    views = (build_chem_graph(batch), build_angle_triplets(batch))
    with torch.no_grad():
        f_c0, f_a0 = mod.stream_deltas(h, *views)
        mod.chem_in.weight.zero_()
        mod.chem_in.bias.zero_()
        f_c1, f_a1 = mod.stream_deltas(h, *views)
    assert not torch.equal(f_c0, f_c1)        # chem stream moved…
    assert torch.equal(f_a0, f_a1)            # …angle stream did not
    # sub-branch parameter sets are disjoint (no shared storage)
    chem_ptrs = {p.data_ptr() for p in mod.chem_branch.parameters()}
    angle_ptrs = {p.data_ptr() for p in mod.angle_branch.parameters()}
    assert chem_ptrs.isdisjoint(angle_ptrs)
    assert mod.chem_in.weight.data_ptr() not in angle_ptrs


def test_branches_are_not_copies_of_standalone_e1_e4(e5) -> None:
    """E5's internal GINE24/Angle24 are fresh width-24 rebuilds with
    their OWN RNG draws — not copies of a same-seed standalone E1/E4
    construction (which is also width 48)."""
    torch.manual_seed(SEED)
    mod = e5.DualStreamEmbedding()
    torch.manual_seed(SEED)
    standalone_gine = e5.GINEChemicalEmbedding(width=24)
    torch.manual_seed(SEED)
    standalone_angle = e5.AngleMPNNEmbedding(width=24)
    assert all(
        not torch.equal(pa, pb) for pa, pb in
        zip(mod.chem_branch.parameters(), standalone_gine.parameters()))
    assert all(
        not torch.equal(pa, pb) for pa, pb in
        zip(mod.angle_branch.parameters(), standalone_angle.parameters()))
    # …and not width-48 parent copies either (shapes differ by design)
    torch.manual_seed(SEED)
    full_gine = e5.GINEChemicalEmbedding()
    full_angle = e5.AngleMPNNEmbedding()
    assert tuple(mod.chem_branch.pe_proj.weight.shape) != tuple(
        full_gine.pe_proj.weight.shape)
    assert tuple(mod.angle_branch.pair_mlp[0].weight.shape) != tuple(
        full_angle.pair_mlp[0].weight.shape)


def test_24_wide_stream_inputs_do_not_mix(e5) -> None:
    """With disjoint projections, perturbing one 24-slice of ``h`` moves
    ONLY that stream — the two branch inputs never cross-contaminate."""
    batch = chain_batch()
    h = random_h(3, SEED)
    mod = e5.DualStreamEmbedding()
    controlled_projections(mod)
    views = (build_chem_graph(batch), build_angle_triplets(batch))
    h_angle_shift = h.clone()
    h_angle_shift[:, 24:] += 10.0
    h_chem_shift = h.clone()
    h_chem_shift[:, :24] += 10.0
    with torch.no_grad():
        f_c0, f_a0 = mod.stream_deltas(h, *views)
        f_c1, f_a1 = mod.stream_deltas(h_angle_shift, *views)
        f_c2, f_a2 = mod.stream_deltas(h_chem_shift, *views)
    assert tuple(f_c0.shape) == (3, BRANCH_WIDTH)
    assert tuple(f_a0.shape) == (3, BRANCH_WIDTH)
    assert torch.equal(f_c0, f_c1)            # chem blind to the angle slice
    assert not torch.equal(f_a0, f_a1)
    assert torch.equal(f_a0, f_a2)            # angle blind to the chem slice
    assert not torch.equal(f_c0, f_c2)


# ---------------------------------------------------------------------------
# inherited j/k unordered-pair symmetry (bitwise)
# ---------------------------------------------------------------------------
def test_jk_swap_is_bitwise_invariant(e5) -> None:
    """Swapping the two sides of EVERY triplet changes nothing — not one
    bit (the Angle24 stream inherits E4's commutative sum/product)."""
    for name, batch in (("chain", chain_batch()),
                        ("shell3", _shell3_batch()),
                        ("sn2", sn2_forward_batch()),
                        ("bidir", sn2_bidirectional_batch(n_graphs=2))):
        n = int(batch["pos"].shape[0])
        h = random_h(n, SEED)
        chem_view = build_chem_graph(batch)
        angle_view = build_angle_triplets(batch)
        mod = e5.DualStreamEmbedding()
        with torch.no_grad():
            base = mod.aggregate_delta(h, chem_view, angle_view)
            swapped = mod.aggregate_delta(h, chem_view, swap_sides(angle_view))
            twice = mod.aggregate_delta(
                h, chem_view, swap_sides(swap_sides(angle_view)))
        assert torch.equal(base, swapped), name
        assert torch.equal(base, twice), name


# ---------------------------------------------------------------------------
# F2 discipline + geometry sensitivity
# ---------------------------------------------------------------------------
def test_f2_masked_slots_never_read_and_input_untouched(e5) -> None:
    """Garbage in the F2-masked first 12 ``edge_attr`` slots changes
    NOTHING (both streams read only the 8 chem fields / recomputed
    geometry), and forward never mutates the input batch."""
    batch = sn2_forward_batch()
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    mod = e5.DualStreamEmbedding()
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


def test_distance_change_moves_delta_at_live_fusion(e5) -> None:
    """Same F2 topology, two current radii ⇒ different E5 delta (the
    Angle24 stream consumes recomputed RBF12 + cosine)."""
    near, far = sphere_center_batch([1.0, 1.0, 1.0]), sphere_center_batch(
        [1.2, 1.2, 1.2])
    assert torch.equal(near["edge_attr"], far["edge_attr"])
    assert torch.equal(near["edge_index"], far["edge_index"])
    h = random_h(4, SEED)
    mod = e5.DualStreamEmbedding()
    with torch.no_grad():
        a = mod(h, near)
        b = mod(h, far)
    assert not torch.allclose(a, b, rtol=REL_TOL, atol=ABS_TOL)
    assert float((a - b).abs().max()) > 1e-4


# ---------------------------------------------------------------------------
# E(3)/permutation invariance + independent vs mixed batch (§6.4/§6.8)
# ---------------------------------------------------------------------------
def test_translation_rotation_reflection_invariance(e5) -> None:
    batch = sn2_bidirectional_batch(n_graphs=2)
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    mod = e5.DualStreamEmbedding()
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
        assert_close(ref, moved, f"E5 {label}")


def test_permutation_equivariance(e5) -> None:
    batch = sn2_forward_batch()
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    gen = torch.Generator().manual_seed(17)
    perm = torch.randperm(n, generator=gen)
    permuted = permute_forward_batch(batch, perm)
    mod = e5.DualStreamEmbedding()
    with torch.no_grad():
        ref = mod(h, batch)
        out = mod(h[perm], permuted)
    assert_close(ref[perm], out, "E5 permutation")


def test_independent_vs_mixed_batch(e5) -> None:
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
    mod = e5.DualStreamEmbedding()
    with torch.no_grad():
        alone = mod(h_a, batch_a)
        alone_b = mod(h_b, batch_b)
        mixed = mod(torch.cat([h_a, h_b]), batch_ab)
    assert int(batch_ab["graph_index"][:n_a].max()) == 0
    assert_close(alone, mixed[:n_a], "E5 independent-vs-mixed (A)")
    assert_close(alone_b, mixed[n_a:n_a + n_b], "E5 independent-vs-mixed (B)")


# ---------------------------------------------------------------------------
# one provider call per view per forward (no duplicated operator)
# ---------------------------------------------------------------------------
def test_one_provider_call_per_view_per_forward(
        e5, monkeypatch: pytest.MonkeyPatch) -> None:
    """The chem view is built EXACTLY once (for GINE24) and the angle
    view EXACTLY once (for Angle24) — E5 never rebuilds a view."""
    seen: dict[str, object] = {}
    real_chem = e5.build_chem_graph
    real_angle = e5.build_angle_triplets

    def chem_spy(batch, *, diagnostics=True):
        seen["chem"] = seen.get("chem", 0) + 1
        seen["chem_diagnostics"] = diagnostics
        return real_chem(batch, diagnostics=diagnostics)

    def angle_spy(batch, *, chunk_size=4096, diagnostics=True):
        seen["angle"] = seen.get("angle", 0) + 1
        seen["angle_diagnostics"] = diagnostics
        seen["chunk_size"] = chunk_size
        return real_angle(batch, chunk_size=chunk_size,
                          diagnostics=diagnostics)

    monkeypatch.setattr(e5, "build_chem_graph", chem_spy)
    monkeypatch.setattr(e5, "build_angle_triplets", angle_spy)
    batch = sn2_bidirectional_batch(n_graphs=2)
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    mod = e5.DualStreamEmbedding(chunk_size=13, diagnostics_enabled=False)
    with torch.no_grad():
        out = mod(h, batch)
    assert bool(torch.isfinite(out).all())
    assert seen["chem"] == 1 and seen["angle"] == 1
    assert seen["chem_diagnostics"] is False
    assert seen["angle_diagnostics"] is False
    assert seen["chunk_size"] == 13
    mod.diagnostics_enabled = True
    with torch.no_grad():
        mod(h, batch)
    assert seen["chem"] == 2 and seen["angle"] == 2
    assert seen["chem_diagnostics"] is True
    assert seen["angle_diagnostics"] is True


# ---------------------------------------------------------------------------
# diagnostics switch (shared keys + per-source prefixes)
# ---------------------------------------------------------------------------
def test_diagnostics_false_skips_count_only_helpers(
        e5, monkeypatch: pytest.MonkeyPatch) -> None:
    batch = sn2_bidirectional_batch(n_graphs=2)
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    cheap = e5.DualStreamEmbedding(diagnostics_enabled=False)
    patch_count_only_helpers(monkeypatch)          # helpers now RAISE
    with torch.no_grad():
        out = cheap(h, batch)                      # must not raise
    assert bool(torch.isfinite(out).all())
    assert cheap.last_diagnostics == {}            # no counts at all

    # control: the default path really routes through those helpers
    loud = e5.DualStreamEmbedding(diagnostics_enabled=True)
    with pytest.raises(AssertionError):
        with torch.no_grad():
            loud(h, batch)


def test_diagnostics_off_keeps_features_bitwise_identical(e5) -> None:
    batch = sn2_bidirectional_batch(n_graphs=2)
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    on = e5.DualStreamEmbedding(diagnostics_enabled=True)
    off = e5.DualStreamEmbedding(diagnostics_enabled=False)
    off.load_state_dict(on.state_dict())
    with torch.no_grad():
        a = on(h, batch)
        b = off(h, batch)
    assert torch.equal(a, b)
    assert on.last_diagnostics and off.last_diagnostics == {}


def test_diagnostics_keys_prefixes_and_detached_floats(e5) -> None:
    batch = sn2_forward_batch()
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    mod = e5.DualStreamEmbedding()
    with torch.no_grad():
        mod(h, batch)
    diag = mod.last_diagnostics
    assert_all_floats(diag, "E5.last_diagnostics")
    # shared vocabulary
    for key in ("n_nodes", "n_edges", "n_triplets",
                "zero_neighbor_frac", "zero_angle_frac"):
        assert key in diag, key
    assert diag["chem.n_atoms"] == float(n)
    assert 0.0 <= diag["zero_neighbor_frac"] <= 1.0
    assert 0.0 <= diag["zero_angle_frac"] <= 1.0
    # per-source prefixed provider counts agree with the providers
    chem_view = build_chem_graph(batch)
    angle_view = build_angle_triplets(batch)
    assert diag["n_edges"] == float(int(chem_view.edge_index.shape[1]))
    assert diag["n_triplets"] == float(int(angle_view.center_index.shape[0]))
    assert diag["chem.n_chem_edges"] == diag["n_edges"]
    assert diag["chem.n_chem_pairs"] == float(int(chem_view.pair_index.shape[1]))
    assert diag["angle.n_triplets"] == diag["n_triplets"]
    assert diag["angle.n_atoms"] == float(n)
    # a returned snapshot never aliases live state
    snapshot = mod.last_diagnostics
    snapshot["n_edges"] = -1.0
    assert mod.last_diagnostics["n_edges"] != -1.0
    # disabled ⇒ empty (and no tensor anywhere)
    mod.diagnostics_enabled = False
    with torch.no_grad():
        mod(h, batch)
    assert mod.last_diagnostics == {}


# ---------------------------------------------------------------------------
# degenerate geometry: typed refusal + finiteness matrix
# ---------------------------------------------------------------------------
def test_zero_length_angle_side_is_typed_refusal(e5) -> None:
    """Degenerate geometry fails CLOSED through E5 — the provider's
    typed refusal is never swallowed or patched over."""
    batch = zero_length_batch()
    h = random_h(int(batch["pos"].shape[0]), SEED)
    mod = e5.DualStreamEmbedding()
    with pytest.raises(FeatureProviderError) as excinfo:
        mod(h, batch)
    assert excinfo.value.code == "V541_PROVIDER_ZERO_DISTANCE"


@pytest.mark.parametrize("name", sorted(dual_stream_batch_matrix()))
def test_outputs_and_gradients_finite(e5, name: str) -> None:
    batch = dual_stream_batch_matrix()[name]
    if name == "zero_length":
        h = random_h(int(batch["pos"].shape[0]), SEED)
        mod = e5.DualStreamEmbedding()
        with pytest.raises(FeatureProviderError) as excinfo:
            mod(h, batch)
        assert excinfo.value.code == "V541_PROVIDER_ZERO_DISTANCE"
        return
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    pos = batch["pos"].detach().clone().requires_grad_(True)
    work = dict(batch)
    work["pos"] = pos
    mod = e5.DualStreamEmbedding()
    out = mod(h, work)
    assert tuple(out.shape) == (n, HIDDEN_V541)
    assert out.dtype == torch.float32
    assert bool(torch.isfinite(out).all())
    delta, node_vector = normalize_branch_output(out, n_nodes=n)
    assert node_vector is None and tuple(delta.shape) == (n, HIDDEN_V541)
    (out * out).sum().backward()
    if pos.grad is not None:
        assert bool(torch.isfinite(pos.grad).all())
    for pname, p in mod.named_parameters():
        if p.grad is None:
            continue
        assert bool(torch.isfinite(p.grad).all()), pname


def test_no_angle_stream_is_masked_to_zero(e5) -> None:
    """Every centre of this fixture has ≤1 chem neighbour ⇒ no angle:
    the Angle24 block is exactly 0 for ALL nodes even with a BIASED
    fusion — no learned missing-angle token leaks through E5."""
    batch = one_chem_one_spatial_batch()
    angle_view = build_angle_triplets(batch)
    assert not bool(angle_view.center_has_angle.any())
    assert int(angle_view.center_index.shape[0]) == 0
    h = random_h(3, SEED)
    mod = e5.DualStreamEmbedding()
    with torch.no_grad():
        mod.fusion.bias.copy_(torch.randn(48))    # a tempting bias
        bias = mod.fusion.bias.detach().clone()
        f_c, f_a = mod.stream_deltas(
            h, build_chem_graph(batch), angle_view)
        f = mod(h, batch)
    assert float(f_a.abs().max()) == 0.0
    assert_close(f, f_c @ mod.fusion.weight[:, :24].T + bias,
                 "no-angle E5 = chem block through fusion")


def test_empty_views_stay_finite(e5) -> None:
    """Single atom and no-neighbour batches: BOTH views empty ⇒ the
    delta is the fusion of two pure self-terms, always finite."""
    for batch, n_nodes in ((single_atom_batch(), 1), (no_neighbor_batch(), 2)):
        h = random_h(n_nodes, SEED)
        mod = e5.DualStreamEmbedding()
        with torch.no_grad():
            out = mod(h, batch)
        assert tuple(out.shape) == (n_nodes, HIDDEN_V541)
        assert bool(torch.isfinite(out).all())
        diag = mod.last_diagnostics
        assert diag["n_edges"] == 0.0
        assert diag["n_triplets"] == 0.0
        assert diag["zero_neighbor_frac"] == 1.0
        assert diag["zero_angle_frac"] == 1.0


def test_dense_tie_shell_rejection_is_finite(e5) -> None:
    """Dense fixture: the 15 equal-radius neighbours form ONE tie shell
    > 12, rejected WHOLE — no angle, finite output, counted."""
    from tests.pesn.test_v541_feature_providers_fixtures import (
        dense_sphere_batch,
    )

    batch = dense_sphere_batch(15)
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    mod = e5.DualStreamEmbedding()
    with torch.no_grad():
        out = mod(h, batch)
    assert bool(torch.isfinite(out).all())
    diag = mod.last_diagnostics
    assert diag["n_triplets"] == 0.0
    assert diag["angle.n_boundary_shells_rejected"] == 1.0
    assert diag["zero_angle_frac"] == 1.0


def test_all_parameter_gradients_present_and_finite(e5) -> None:
    """With chem pairs AND angles present every parameter is in the
    graph and receives a finite gradient."""
    batch = sn2_forward_batch()
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    mod = e5.DualStreamEmbedding()
    out = mod(h, batch)
    diag = mod.last_diagnostics
    assert diag["n_edges"] > 0.0 and diag["n_triplets"] > 0.0
    (out * out).sum().backward()
    total = 0.0
    for pname, p in mod.named_parameters():
        assert p.grad is not None, pname
        assert bool(torch.isfinite(p.grad).all()), pname
        total += float(p.grad.abs().sum())
    assert total > 0.0
    # both stream projections really carry gradient (both views live)
    assert float(mod.chem_in.weight.grad.abs().sum()) > 0.0
    assert float(mod.angle_in.weight.grad.abs().sum()) > 0.0


def test_eval_determinism(e5) -> None:
    batch = sn2_bidirectional_batch(n_graphs=2)
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    mod = e5.DualStreamEmbedding()
    with torch.no_grad():
        a = mod(h, batch)
        b = mod(h, batch)
    assert torch.equal(a, b)
