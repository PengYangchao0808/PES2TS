"""V5.4.1 WP3b — E4 ``angle_mpnn`` candidate tests (plan §6.7 E4, §6.3,
§6.6.1, §6.8 rows 1/4/5/6).

Covered here (candidate-level; the model-level gate/parity battery lives
in ``test_v541_embedding_geometric_model.py``):

* hand-computed small graphs (right angle, 3-neighbour tie shell, 3-atom
  chain) against the literal §6.7 formula with controlled weights;
* STRICT ``j/k`` swap invariance (bitwise) and no atom-id feature: the
  unordered pair enters only through ``a_ij + a_ik`` / ``a_ij ⊙ a_ik``;
* every pair input (``h_j``, RBF12, the 8 chem fields, the cosine basis)
  is actually consumed;
* the explicit ``center_has_angle`` mask: a no-angle centre aggregates 0
  and outputs exactly 0 even with a biased output MLP (no learned
  missing-angle token);
* typed refusal of a zero-length angle side;
* F2 discipline (masked slots never read, input never mutated, a pure
  distance change moves the delta);
* E(3)/permutation invariance, independent-vs-mixed batch consistency,
  chunking equivalence (bitwise), the diagnostics switch, extreme and
  degenerate finiteness, finite gradients, small-batch CPU budget.

Registry discipline: the E4 module is imported INSIDE an autouse fixture
(its import registers ``E4``) and the registry is restored afterwards.
"""

from __future__ import annotations

import dataclasses
import time

import pytest
import torch

from pes2ts_core.pesn.models.v541.embedding_registry import (
    build_embedding,
    is_embedding_implemented,
    registry_restore,
    registry_snapshot,
)
from pes2ts_core.pesn.models.v541.feature_providers import (
    FeatureProviderError,
    build_angle_triplets,
)
from pes2ts_core.pesn.models.v541.interfaces import (
    HIDDEN_V541,
    normalize_branch_output,
)
from tests.pesn.test_v541_embedding_geometric_fixtures import (
    ABS_TOL,
    REL_TOL,
    assert_all_floats,
    assert_close,
    geometric_batch_matrix,
    load_e4,
    patch_count_only_helpers,
    random_h,
    sn2_bidirectional_batch,
    sn2_forward_batch,
)
from tests.pesn.test_v541_feature_providers_fixtures import (
    angle_geometry_batch,
    forward_batch,
    make_intent,
    make_sample,
    one_chem_one_spatial_batch,
    permute_forward_batch,
    random_rotation,
    reflection_matrix,
    sphere_center_batch,
    transform_forward_batch,
    zero_length_batch,
)

SEED = 23


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def e4():
    """Import (⇒ register E4) inside the test, restore afterwards."""
    snap = registry_snapshot()
    module = load_e4()
    yield module
    registry_restore(snap)


def _controlled(module) -> None:
    """``P_e = 0``, ``a_ij = h_j`` (identity pair MLP), triplet message =
    ``a_ij + a_ik``, output MLP = ReLU (zero biases).

    With a POSITIVE ``h`` the pipeline collapses to the hand formula
    ``f_i = Σ_{{j,k}} (h_j + h_k)`` for angle centres and ``0`` elsewhere.
    """
    with torch.no_grad():
        module.pe_proj.weight.zero_()
        module.pe_proj.bias.zero_()
        pair_w = torch.zeros(48, 108)
        pair_w[:, :48] = torch.eye(48)              # select h_j only
        module.pair_mlp[0].weight.copy_(pair_w)
        module.pair_mlp[0].bias.zero_()
        module.pair_mlp[2].weight.copy_(torch.eye(48))
        module.pair_mlp[2].bias.zero_()
        trip_w = torch.zeros(48, 148)
        trip_w[:, 48:96] = torch.eye(48)            # select a_ij + a_ik
        module.triplet_mlp[0].weight.copy_(trip_w)
        module.triplet_mlp[0].bias.zero_()
        module.triplet_mlp[2].weight.copy_(torch.eye(48))
        module.triplet_mlp[2].bias.zero_()
        module.out_mlp[0].weight.copy_(torch.eye(48))
        module.out_mlp[0].bias.zero_()
        module.out_mlp[2].weight.copy_(torch.eye(48))
        module.out_mlp[2].bias.zero_()


def _chain_batch() -> dict:
    """3-atom chain: centre 1 has exactly two chem neighbours (one angle)."""
    sample = make_sample(
        (6, 1, 1),
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (5.0, 0.0, 0.0)],
        make_intent(((0, 1, 1.0), (1, 2, 1.0)), ((0, 1, 1.0), (1, 2, 1.0)),
                    cv_deltas=(0.1,), cv_coords=((0, 1),)))
    return forward_batch([sample])


def _swap_sides(view):
    """The same triplet view with the unordered pair's columns exchanged."""
    return dataclasses.replace(
        view,
        side_index=view.side_index.flip(1),
        side_distance=view.side_distance.flip(1),
        side_rbf12=view.side_rbf12.flip(1),
        side_chem_attr=view.side_chem_attr.flip(1),
    )


# ---------------------------------------------------------------------------
# construction / registry contract
# ---------------------------------------------------------------------------
def test_e4_shapes_and_registry(e4) -> None:
    mod = e4.AngleMPNNEmbedding()
    assert mod.width == HIDDEN_V541 == 48
    assert tuple(mod.pe_proj.weight.shape) == (48, 8)        # P_e: 8→48
    assert tuple(mod.pair_mlp[0].weight.shape) == (48, 108)  # 48+48+12
    assert tuple(mod.pair_mlp[2].weight.shape) == (48, 48)
    assert tuple(mod.triplet_mlp[0].weight.shape) == (48, 148)  # 48·3+4
    assert tuple(mod.triplet_mlp[2].weight.shape) == (48, 48)
    assert tuple(mod.out_mlp[0].weight.shape) == (48, 48)
    assert tuple(mod.out_mlp[2].weight.shape) == (48, 48)
    for mlp in (mod.pair_mlp, mod.triplet_mlp, mod.out_mlp):
        assert isinstance(mlp[1], torch.nn.ReLU)     # ReLU in the middle
        assert isinstance(mlp[0], torch.nn.Linear)
        assert isinstance(mlp[2], torch.nn.Linear)
    assert is_embedding_implemented("E4") is True
    built = build_embedding("E4")
    assert isinstance(built, e4.AngleMPNNEmbedding)
    assert mod.diagnostics_enabled is True
    assert mod.last_diagnostics == {}


def test_e5_width_24_is_constructible(e4) -> None:
    """E5 later reuses this class at width 24 (§6.7 E5) — the width knob
    exists and keeps the two-layer MLP hidden width equal to it."""
    mod = e4.AngleMPNNEmbedding(24)
    assert mod.width == 24
    assert tuple(mod.pair_mlp[0].weight.shape) == (24, 24 + 24 + 12)
    assert tuple(mod.triplet_mlp[0].weight.shape) == (24, 24 * 3 + 4)


# ---------------------------------------------------------------------------
# hand-computed small graphs
# ---------------------------------------------------------------------------
def test_hand_computed_right_angle(e4) -> None:
    """Centre 0 with two 1.0 Å neighbours: ``f_0 = h_1 + h_2``; the two
    terminal atoms have no angle ⇒ exactly 0."""
    batch = angle_geometry_batch("right")
    mod = e4.AngleMPNNEmbedding()
    _controlled(mod)
    h = torch.rand(3, 48) + 0.5
    with torch.no_grad():
        f = mod(h, batch)
    assert_close(f[0], h[1] + h[2], "right-angle centre")
    assert float(f[1].abs().max()) == 0.0
    assert float(f[2].abs().max()) == 0.0
    assert mod.last_diagnostics["n_triplets"] == 1.0
    assert mod.last_diagnostics["zero_angle"] == 2.0


def test_hand_computed_three_neighbour_tie_shell(e4) -> None:
    """Equal-radius shell: C(3,2) = 3 triplets ⇒ ``f_0 = 2·(h_1+h_2+h_3)``
    (each leaf appears in two of the three unordered pairs)."""
    batch = sphere_center_batch([1.0, 1.0, 1.0])
    mod = e4.AngleMPNNEmbedding()
    _controlled(mod)
    h = torch.rand(4, 48) + 0.5
    with torch.no_grad():
        f = mod(h, batch)
    assert_close(f[0], 2.0 * (h[1] + h[2] + h[3]), "tie-shell centre")
    for leaf in (1, 2, 3):
        assert float(f[leaf].abs().max()) == 0.0, leaf
    diag = mod.last_diagnostics
    assert diag["n_triplets"] == 3.0
    assert diag["zero_angle"] == 3.0


def test_hand_computed_chain_uses_neighbour_hidden_state(e4) -> None:
    """Only centre 1 carries an angle: ``f_1 = h_0 + h_2`` — the message
    is built from the SIDE neighbours (``h_j``), never from ``h_i``."""
    batch = _chain_batch()
    mod = e4.AngleMPNNEmbedding()
    _controlled(mod)
    h = torch.rand(3, 48) + 0.5
    with torch.no_grad():
        f = mod(h, batch)
    assert_close(f[1], h[0] + h[2], "chain centre")
    assert float(f[0].abs().max()) == 0.0
    assert float(f[2].abs().max()) == 0.0
    assert mod.last_diagnostics["n_triplets"] == 1.0
    assert mod.last_diagnostics["zero_angle"] == 2.0


def test_cosine_channel_changes_output(e4) -> None:
    """The right-angle and straight-angle fixtures share topology AND
    side distances (1.0 Å) — only ``c`` differs (0 vs −1), so with the
    cosine basis wired the deltas must differ."""
    right = angle_geometry_batch("right")
    straight = angle_geometry_batch("pi")
    assert torch.equal(right["edge_attr"], straight["edge_attr"])
    h = random_h(3, SEED)
    mod = e4.AngleMPNNEmbedding()
    with torch.no_grad():
        a = mod(h, right)
        b = mod(h, straight)
    assert not torch.allclose(a, b, rtol=REL_TOL, atol=ABS_TOL)
    # and the provider really only differs in the cosine channel
    va = build_angle_triplets(right)
    vb = build_angle_triplets(straight)
    assert torch.equal(va.side_distance, vb.side_distance)
    assert torch.equal(va.side_rbf12, vb.side_rbf12)
    assert not torch.equal(va.cos_theta, vb.cos_theta)


# ---------------------------------------------------------------------------
# unordered pair: strict j/k swap, no atom-id feature (§6.8 row 4)
# ---------------------------------------------------------------------------
def test_jk_swap_is_bitwise_invariant(e4) -> None:
    """Swapping the two sides of EVERY triplet changes nothing — not one
    bit.  ``j/k`` enter only through commutative sum/product."""
    for name, batch in (("sn2", sn2_forward_batch()),
                        ("tie", sphere_center_batch([1.0] * 5)),
                        ("bidir", sn2_bidirectional_batch(n_graphs=2))):
        n = int(batch["pos"].shape[0])
        h = random_h(n, SEED)
        view = build_angle_triplets(batch)
        mod = e4.AngleMPNNEmbedding()
        with torch.no_grad():
            base = mod.aggregate_delta(h, view)
            swapped = mod.aggregate_delta(h, _swap_sides(view))
        assert torch.equal(base, swapped), name
        # …and a DOUBLE swap is identity too
        with torch.no_grad():
            twice = mod.aggregate_delta(h, _swap_sides(_swap_sides(view)))
        assert torch.equal(base, twice), name


def test_pair_inputs_are_consumed(e4) -> None:
    """RBF12 / the 8 chem fields of each side really feed ``a_ij`` (a
    perturbation of either moves the delta); so does the cosine basis."""
    batch = sn2_forward_batch()
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    view = build_angle_triplets(batch)
    mod = e4.AngleMPNNEmbedding()
    with torch.no_grad():
        base = mod.aggregate_delta(h, view)
        rbf = mod.aggregate_delta(
            h, dataclasses.replace(view, side_rbf12=view.side_rbf12 + 0.25))
        chem = mod.aggregate_delta(
            h, dataclasses.replace(view,
                                   side_chem_attr=view.side_chem_attr + 0.5))
        cos = mod.aggregate_delta(
            h, dataclasses.replace(view, cos_basis=view.cos_basis * 0.5))
    assert not torch.allclose(base, rbf, rtol=REL_TOL, atol=ABS_TOL)
    assert not torch.allclose(base, chem, rtol=REL_TOL, atol=ABS_TOL)
    assert not torch.allclose(base, cos, rtol=REL_TOL, atol=ABS_TOL)


def test_no_angle_centre_is_masked_to_zero(e4) -> None:
    """``agg = 0`` AND the explicit ``center_has_angle`` mask: even a
    BIASED output MLP cannot put a learned missing-angle token on a
    centre without an angle."""
    batch = one_chem_one_spatial_batch()            # every centre: ≤1 nbr
    view = build_angle_triplets(batch)
    assert not bool(view.center_has_angle.any())
    assert int(view.center_index.shape[0]) == 0
    h = random_h(3, SEED)
    mod = e4.AngleMPNNEmbedding()
    with torch.no_grad():
        mod.out_mlp[2].bias.fill_(7.0)              # a tempting constant
        f = mod(h, batch)
    assert float(f.abs().max()) == 0.0              # …never reaches output
    diag = mod.last_diagnostics
    assert diag["zero_angle"] == 3.0
    assert diag["n_triplets"] == 0.0


def test_zero_length_angle_side_is_typed_refusal(e4) -> None:
    """Degenerate geometry fails CLOSED through the candidate — the
    provider's typed refusal is never swallowed or patched over."""
    batch = zero_length_batch()
    h = random_h(int(batch["pos"].shape[0]), SEED)
    mod = e4.AngleMPNNEmbedding()
    with pytest.raises(FeatureProviderError) as excinfo:
        mod(h, batch)
    assert excinfo.value.code == "V541_PROVIDER_ZERO_DISTANCE"


# ---------------------------------------------------------------------------
# F2 discipline + geometry sensitivity (§6.8 row 1)
# ---------------------------------------------------------------------------
def test_f2_masked_slots_are_never_read(e4) -> None:
    """Garbage in the F2-masked first 12 ``edge_attr`` slots changes NOTHING
    — both side distances and RBF12 come from ``pos`` (§6.6.1)."""
    batch = sn2_forward_batch()
    h = random_h(int(batch["pos"].shape[0]), SEED)
    mod = e4.AngleMPNNEmbedding()
    with torch.no_grad():
        base = mod(h, batch)
        poisoned = dict(batch)
        attr = batch["edge_attr"].clone()
        attr[:, :12] = 1234.5
        poisoned["edge_attr"] = attr
        out = mod(h, poisoned)
    assert torch.equal(base, out)                  # bitwise: slots unread


def test_distance_change_moves_delta_while_f2_stays_untouched(e4) -> None:
    """Same F2 topology, two current radii ⇒ different E4 delta (§6.8
    row 1), with the input batch left byte-identical."""
    near = sphere_center_batch([1.0, 1.0, 1.0])
    far = sphere_center_batch([1.2, 1.2, 1.2])
    assert torch.equal(near["edge_attr"], far["edge_attr"])
    assert torch.equal(near["edge_index"], far["edge_index"])
    assert bool((near["edge_attr"][:, :12] == 0).all())
    assert bool((far["edge_attr"][:, :12] == 0).all())

    h = random_h(4, SEED)
    mod = e4.AngleMPNNEmbedding()
    before = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
              for k, v in near.items()}
    with torch.no_grad():
        d_near = mod(h, near)
        d_far = mod(h, far)
    assert not torch.allclose(d_near, d_far, rtol=REL_TOL, atol=ABS_TOL)
    assert float((d_near - d_far).abs().max()) > 1e-4
    for key, value in near.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, before[key]), key


# ---------------------------------------------------------------------------
# E(3) + permutation + batch consistency (§6.4 / §6.8)
# ---------------------------------------------------------------------------
def test_translation_rotation_reflection_invariance(e4) -> None:
    batch = sn2_bidirectional_batch(n_graphs=2)
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    mod = e4.AngleMPNNEmbedding()
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
        assert_close(ref, moved, f"E4 {label}")


def test_permutation_equivariance(e4) -> None:
    batch = sn2_forward_batch()
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    gen = torch.Generator().manual_seed(17)
    perm = torch.randperm(n, generator=gen)
    permuted = permute_forward_batch(batch, perm)
    mod = e4.AngleMPNNEmbedding()
    with torch.no_grad():
        ref = mod(h, batch)
        out = mod(h[perm], permuted)
    assert_close(ref[perm], out, "E4 permutation")


def test_independent_vs_mixed_batch(e4) -> None:
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
    mod = e4.AngleMPNNEmbedding()
    with torch.no_grad():
        alone = mod(h_a, batch_a)
        mixed = mod(torch.cat([h_a, h_b]), batch_ab)
        alone_b = mod(h_b, batch_b)
    assert int(batch_ab["graph_index"][:n_a].max()) == 0
    assert_close(alone, mixed[:n_a], "E4 independent-vs-mixed (A)")
    assert_close(alone_b, mixed[n_a:n_a + n_b], "E4 independent-vs-mixed (B)")


def test_chunking_equivalence_is_bitwise(e4) -> None:
    batch = sn2_bidirectional_batch(n_graphs=2)
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    default = e4.AngleMPNNEmbedding()
    tiny = e4.AngleMPNNEmbedding(chunk_size=1)
    tiny.load_state_dict(default.state_dict())
    other = e4.AngleMPNNEmbedding(chunk_size=7)
    other.load_state_dict(default.state_dict())
    with torch.no_grad():
        a = default(h, batch)
        b = tiny(h, batch)
        c = other(h, batch)
    assert torch.equal(a, b)
    assert torch.equal(a, c)


# ---------------------------------------------------------------------------
# diagnostics switch (WP3a pattern)
# ---------------------------------------------------------------------------
def test_diagnostics_false_skips_count_only_helpers(
        monkeypatch: pytest.MonkeyPatch, e4) -> None:
    batch = sn2_bidirectional_batch(n_graphs=2)
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    cheap = e4.AngleMPNNEmbedding(diagnostics_enabled=False)
    patch_count_only_helpers(monkeypatch)          # helpers now RAISE
    with torch.no_grad():
        out = cheap(h, batch)                      # must not raise
    assert bool(torch.isfinite(out).all())
    assert cheap.last_diagnostics == {}

    loud = e4.AngleMPNNEmbedding(diagnostics_enabled=True)
    with pytest.raises(AssertionError):
        with torch.no_grad():
            loud(h, batch)


def test_diagnostics_off_keeps_features_bitwise_identical(e4) -> None:
    batch = sn2_bidirectional_batch(n_graphs=2)
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    on = e4.AngleMPNNEmbedding(diagnostics_enabled=True)
    off = e4.AngleMPNNEmbedding(diagnostics_enabled=False)
    off.load_state_dict(on.state_dict())
    with torch.no_grad():
        a = on(h, batch)
        b = off(h, batch)
    assert torch.equal(a, b)
    assert on.last_diagnostics and off.last_diagnostics == {}


def test_diagnostics_are_detached_floats_and_copied(e4) -> None:
    batch = sn2_bidirectional_batch(n_graphs=2)
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    mod = e4.AngleMPNNEmbedding()
    with torch.no_grad():
        mod(h, batch)
    diag = mod.last_diagnostics
    assert_all_floats(diag, "E4.last_diagnostics")
    for key in ("n_nodes", "n_triplets", "zero_angle"):
        assert key in diag, key
    # provider vocabulary merged in, and the counts agree with the provider
    view = build_angle_triplets(batch)
    assert diag["n_triplets"] == float(view.center_index.shape[0])
    assert diag["n_atoms"] == float(n)
    assert diag["zero_angle"] == float(
        int((~view.center_has_angle).sum()))
    snapshot = mod.last_diagnostics
    snapshot["n_triplets"] = -1.0
    assert mod.last_diagnostics["n_triplets"] != -1.0
    mod.diagnostics_enabled = False
    with torch.no_grad():
        mod(h, batch)
    assert mod.last_diagnostics == {}


def test_provider_receives_chunk_size_and_flag(e4, monkeypatch) -> None:
    seen: dict[str, object] = {}
    real = e4.build_angle_triplets

    def spy(batch, *, chunk_size=4096, diagnostics=True):
        seen["chunk_size"] = chunk_size
        seen["diagnostics"] = diagnostics
        return real(batch, chunk_size=chunk_size, diagnostics=diagnostics)

    monkeypatch.setattr(e4, "build_angle_triplets", spy)
    batch = sn2_forward_batch()
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    mod = e4.AngleMPNNEmbedding(chunk_size=13, diagnostics_enabled=False)
    with torch.no_grad():
        mod(h, batch)
    assert seen == {"chunk_size": 13, "diagnostics": False}
    mod.diagnostics_enabled = True
    with torch.no_grad():
        mod(h, batch)
    assert seen == {"chunk_size": 13, "diagnostics": True}


# ---------------------------------------------------------------------------
# finiteness / gradients / contract / budget (§6.4)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(geometric_batch_matrix()))
def test_outputs_and_gradients_finite(e4, name: str) -> None:
    batch = geometric_batch_matrix()[name]
    if name == "zero_length":
        # degenerate angle side: refused by the provider, fail-closed
        h = random_h(int(batch["pos"].shape[0]), SEED)
        mod = e4.AngleMPNNEmbedding()
        with pytest.raises(FeatureProviderError) as excinfo:
            mod(h, batch)
        assert excinfo.value.code == "V541_PROVIDER_ZERO_DISTANCE"
        return
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    pos = batch["pos"].detach().clone().requires_grad_(True)
    work = dict(batch)
    work["pos"] = pos
    mod = e4.AngleMPNNEmbedding()
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


def test_all_parameter_gradients_present_and_finite(e4) -> None:
    """With angles present every parameter is in the graph and finite."""
    batch = sn2_forward_batch()
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    mod = e4.AngleMPNNEmbedding()
    out = mod(h, batch)
    assert mod.last_diagnostics["n_triplets"] > 0.0
    (out * out).sum().backward()
    total = 0.0
    for pname, p in mod.named_parameters():
        assert p.grad is not None, pname
        assert bool(torch.isfinite(p.grad).all()), pname
        total += float(p.grad.abs().sum())
    assert total > 0.0


def test_small_batch_cpu_budget(e4) -> None:
    """Bounded per-forward CPU cost (measured ≈3 ms; 2 s is a smoke
    guard against an accidental per-centre O(N²) regression)."""
    batch = sn2_bidirectional_batch(n_graphs=3)
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    mod = e4.AngleMPNNEmbedding()
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(3):
            out = mod(h, batch)
        elapsed = (time.perf_counter() - start) / 3.0
    assert bool(torch.isfinite(out).all())
    assert elapsed < 2.0, f"E4 forward took {elapsed * 1000:.0f} ms"


def test_eval_determinism(e4) -> None:
    batch = sn2_bidirectional_batch(n_graphs=2)
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    mod = e4.AngleMPNNEmbedding()
    with torch.no_grad():
        a = mod(h, batch)
        b = mod(h, batch)
    assert torch.equal(a, b)


def test_dense_equal_radius_shell_is_rejected_whole(e4) -> None:
    """Dense fixture: the centre's 15 equal-radius neighbours form ONE
    tie shell > 12, so ``reject_boundary_tie_shell`` drops it WHOLE —
    no partial, atom-id-dependent selection, no angle, output exactly 0."""
    from tests.pesn.test_v541_feature_providers_fixtures import (
        dense_sphere_batch,
    )

    batch = dense_sphere_batch(15)
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    mod = e4.AngleMPNNEmbedding()
    with torch.no_grad():
        out = mod(h, batch)
    assert bool(torch.isfinite(out).all())
    assert float(out.abs().max()) == 0.0
    view = build_angle_triplets(batch)
    assert int(view.center_index.shape[0]) == 0
    per_center = torch.bincount(
        view.center_index, minlength=int(view.center_has_angle.shape[0]))
    assert int(per_center.max()) <= 66
    diag = mod.last_diagnostics
    assert diag["n_triplets"] == 0.0
    assert diag["n_boundary_shells_rejected"] == 1.0
    assert diag["zero_angle"] == float(n)


def test_twelve_neighbour_cap_yields_66_triplets(e4) -> None:
    """Exactly 12 tied neighbours survive the cap ⇒ C(12,2) = 66
    triplets for the centre (the frozen ANGLE_MAX_PAIRS bound)."""
    batch = sphere_center_batch([1.0] * 12)
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    mod = e4.AngleMPNNEmbedding()
    with torch.no_grad():
        out = mod(h, batch)
    assert bool(torch.isfinite(out).all())
    view = build_angle_triplets(batch)
    assert int(view.center_index.shape[0]) == 66
    per_center = torch.bincount(
        view.center_index, minlength=int(view.center_has_angle.shape[0]))
    assert int(per_center.max()) == 66
    assert mod.last_diagnostics["n_triplets"] == 66.0
    assert mod.last_diagnostics["zero_angle"] == float(n - 1)
