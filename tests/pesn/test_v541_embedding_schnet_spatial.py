"""V5.4.1 WP3b — E3 ``schnet_spatial`` candidate tests (plan §6.7 E3,
§6.6.1/§6.6.2, §6.8 rows 1/3/4/5/6).

Covered here (candidate-level; the model-level gate/parity battery lives
in ``test_v541_embedding_geometric_model.py``):

* the frozen cosine envelope — hand values, the exact ``r = 4.5 Å``
  boundary, finiteness and a zero gradient beyond the cutoff;
* hand-computed small-graph messages (bonded pair and a 3-atom chain)
  against an INDEPENDENT literal loop implementation of
  ``m_i = Σ_j A(h_j) ⊙ F(RBF12) · c(r)`` with controlled weights;
* ``F`` really consumes the recomputed RBF12 (12→48→48, ReLU middle);
* zero-neighbour aggregation is exactly 0 (``zero_neighbor`` diagnostic);
* F2 discipline: the masked first 12 ``edge_attr`` slots are never READ
  (garbage there changes nothing) and never WRITTEN (input batch
  untouched), while a pure distance change DOES change the delta;
* E(3)/permutation invariance, independent-vs-mixed batch consistency,
  chunking equivalence (bitwise), the diagnostics switch (no count-only
  host sync on the cheap path), extreme/degenerate finiteness, finite
  gradients and a small-batch CPU budget.

Registry discipline: the E3 module is imported INSIDE an autouse fixture
(its import registers ``E3``) and the registry is restored afterwards, so
nothing leaks into the WP2 registry tests running in the same session.
"""

from __future__ import annotations

import math
import time

import pytest
import torch

from pes2ts_core.pesn.models.v541.embedding_registry import (
    build_embedding,
    is_embedding_implemented,
    registry_restore,
    registry_snapshot,
)
from pes2ts_core.pesn.models.v541.embeddings.geometric_common import (
    ENVELOPE_CUTOFF,
    cosine_envelope,
)
from pes2ts_core.pesn.models.v541.feature_providers import build_spatial_graph
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
    load_e3,
    patch_count_only_helpers,
    random_h,
    sn2_bidirectional_batch,
    sn2_forward_batch,
)
from tests.pesn.test_v541_feature_providers_fixtures import (
    bonded_pair_batch,
    forward_batch,
    make_intent,
    make_sample,
    no_neighbor_batch,
    permute_forward_batch,
    random_rotation,
    reflection_matrix,
    transform_forward_batch,
    zero_length_batch,
)

SEED = 11


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def e3():
    """Import (⇒ register E3) inside the test, restore afterwards."""
    snap = registry_snapshot()
    module = load_e3()
    yield module
    registry_restore(snap)


def _envelope(r: float) -> float:
    """Independent literal evaluation of the frozen §6.7 envelope."""
    return 0.5 * (math.cos(math.pi * r / 4.5) + 1.0) if r < 4.5 else 0.0


def _controlled(module) -> None:
    """A = identity, F ≡ ones, atomwise MLP = ReLU (zero biases).

    With a POSITIVE ``h`` the whole pipeline becomes hand-computable:
    ``f = ReLU(Σ_j h_j · c(r_ij)) = Σ_j h_j · c(r_ij)``.
    """
    with torch.no_grad():
        module.atom_proj.weight.copy_(torch.eye(HIDDEN_V541))
        module.atom_proj.bias.zero_()
        module.filter_mlp[0].weight.zero_()
        module.filter_mlp[0].bias.fill_(1.0)      # hidden = 1 → ReLU → 1
        module.filter_mlp[2].weight.zero_()
        module.filter_mlp[2].bias.copy_(torch.ones(HIDDEN_V541))
        module.out_mlp[0].weight.copy_(torch.eye(HIDDEN_V541))
        module.out_mlp[0].bias.zero_()
        module.out_mlp[2].weight.copy_(torch.eye(HIDDEN_V541))
        module.out_mlp[2].bias.zero_()


def _chain_batch() -> dict:
    """3-atom chain 0–1 (1.0 Å) –2 (4.0 Å apart from 1).

    Both bonds are covalent AND inside the 4.5 Å cutoff, so ``G_spatial``
    is exactly the two bonds; the 0–2 pair (5.0 Å) is neither a bond nor
    a spatial contact.  Hand expectations below rely on this.
    """
    sample = make_sample(
        (6, 1, 1),
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (5.0, 0.0, 0.0)],
        make_intent(((0, 1, 1.0), (1, 2, 1.0)), ((0, 1, 1.0), (1, 2, 1.0)),
                    cv_deltas=(0.1,), cv_coords=((0, 1),)))
    return forward_batch([sample])


# ---------------------------------------------------------------------------
# envelope (frozen §6.7): values, 4.5 Å boundary, gradients
# ---------------------------------------------------------------------------
def test_envelope_hand_values_and_boundary() -> None:
    r = torch.tensor([0.0, 1.0, 2.25, 4.0, 4.4, 4.5, 4.5 + 1e-6, 6.0, 100.0])
    c = cosine_envelope(r)
    assert c.shape == r.shape and c.dtype == r.dtype
    assert float(c[0]) == 1.0                      # c(0) = 1
    assert abs(float(c[2]) - 0.5) <= 1e-6          # c(cutoff/2) = 0.5
    assert float(c[1]) == pytest.approx(_envelope(1.0), abs=1e-6)
    assert float(c[3]) == pytest.approx(_envelope(4.0), abs=1e-6)
    assert float(c[4]) > 0.0                       # still inside < 4.5
    # the boundary itself and everything beyond is EXACTLY zero
    for idx in (5, 6, 7, 8):
        assert float(c[idx]) == 0.0, idx
    assert ENVELOPE_CUTOFF == 4.5


def test_envelope_is_continuous_up_to_the_cutoff() -> None:
    left = float(cosine_envelope(torch.tensor([4.5 - 1e-3]))[0])
    assert left <= 1e-4 and left >= 0.0            # → 0 as r → 4.5⁻
    assert float(cosine_envelope(torch.tensor([2.25]))[0]) == pytest.approx(
        0.5, abs=1e-6)


def test_envelope_gradient_finite_and_zero_beyond_cutoff() -> None:
    r = torch.tensor([0.5, 2.0, 4.4, 4.5, 7.0], requires_grad=True)
    c = cosine_envelope(r)
    c.sum().backward()
    assert r.grad is not None
    assert bool(torch.isfinite(r.grad).all())
    # inside: the analytic derivative −(π/2·4.5)·sin(π r/4.5) is nonzero
    assert abs(float(r.grad[0])) > 0.0
    assert abs(float(r.grad[2])) > 0.0
    # at/beyond the cutoff the hard-zero branch contributes NO gradient
    assert float(r.grad[3]) == 0.0
    assert float(r.grad[4]) == 0.0


def test_envelope_follows_dtype() -> None:
    for dtype in (torch.float32, torch.float64):
        c = cosine_envelope(torch.tensor([1.0, 5.0], dtype=dtype))
        assert c.dtype == dtype
        assert bool(torch.isfinite(c).all())


# ---------------------------------------------------------------------------
# construction / registry contract
# ---------------------------------------------------------------------------
def test_e3_shapes_and_registry(e3) -> None:
    assert e3.ENVELOPE_CUTOFF == 4.5               # frozen §6.7 cutoff
    mod = e3.SchnetSpatialEmbedding()
    assert mod.width == HIDDEN_V541 == 48
    assert tuple(mod.atom_proj.weight.shape) == (48, 48)   # A: 48→48
    assert tuple(mod.filter_mlp[0].weight.shape) == (48, 12)   # F: 12→48
    assert tuple(mod.filter_mlp[2].weight.shape) == (48, 48)   # →48
    assert tuple(mod.out_mlp[0].weight.shape) == (48, 48)
    assert tuple(mod.out_mlp[2].weight.shape) == (48, 48)
    # ReLU between, linear output, no dropout/normalisation
    assert isinstance(mod.filter_mlp[1], torch.nn.ReLU)
    assert isinstance(mod.out_mlp[1], torch.nn.ReLU)
    # the import registered the REAL E3 builder (WP3 path)
    assert is_embedding_implemented("E3") is True
    built = build_embedding("E3")
    assert isinstance(built, e3.SchnetSpatialEmbedding)
    # diagnostics switch defaults to the frozen WP3a default
    assert mod.diagnostics_enabled is True
    assert mod.last_diagnostics == {}


# ---------------------------------------------------------------------------
# hand-computed small graphs
# ---------------------------------------------------------------------------
def test_hand_computed_message_bonded_pair(e3) -> None:
    """2 atoms at 1.0 Å: ``f_0 = h_1·c(1.0)``, ``f_1 = h_0·c(1.0)``."""
    batch = bonded_pair_batch(1.0)
    mod = e3.SchnetSpatialEmbedding()
    _controlled(mod)
    h = torch.rand(2, 48) + 0.5                    # positive ⇒ ReLU inert
    with torch.no_grad():
        f = mod(h, batch)
    c = _envelope(1.0)
    assert c == pytest.approx(0.5 * (math.cos(math.pi / 4.5) + 1.0), abs=1e-9)
    assert_close(f[0], h[1] * c, "bonded f0")
    assert_close(f[1], h[0] * c, "bonded f1")
    # the view premise: exactly one spatial pair
    view = build_spatial_graph(batch)
    assert tuple(view.pair_index.shape) == (2, 1)


def test_hand_computed_message_chain_against_literal_loop(e3) -> None:
    """3-atom chain vs an INDEPENDENT literal loop of the §6.7 formula."""
    batch = _chain_batch()
    mod = e3.SchnetSpatialEmbedding()
    _controlled(mod)
    h = torch.rand(3, 48) + 0.5

    view = build_spatial_graph(batch)
    pairs = {(int(view.pair_index[0, p]), int(view.pair_index[1, p]))
             for p in range(int(view.pair_index.shape[1]))}
    assert pairs == {(0, 1), (1, 2)}, pairs       # the hand-test premise

    # literal reference: m_i = Σ_j A(h_j) ⊙ F · c(r) with A = I, F = 1
    expected = torch.zeros(3, 48)
    for p in range(int(view.pair_index.shape[1])):
        a, b = int(view.pair_index[0, p]), int(view.pair_index[1, p])
        r = float(view.pair_distance[p])
        weight = _envelope(r)
        expected[a] += h[b] * weight
        expected[b] += h[a] * weight              # bidirectional view
    with torch.no_grad():
        f = mod(h, batch)
    assert_close(f, torch.relu(expected), "chain vs literal loop")
    # and the explicit numbers: 0–1 = 1.0 Å, 1–2 = 4.0 Å
    assert_close(f[0], h[1] * _envelope(1.0), "chain f0")
    assert_close(f[1], h[0] * _envelope(1.0) + h[2] * _envelope(4.0), "chain f1")
    assert_close(f[2], h[1] * _envelope(4.0), "chain f2")


def test_filter_mlp_consumes_recomputed_rbf12(e3) -> None:
    """F is exactly 12→48→48 with a ReLU in the middle."""
    mod = e3.SchnetSpatialEmbedding()
    with torch.no_grad():
        # hidden = rbf (first 12 rows identity, rest zero), then identity
        w1 = torch.zeros(48, 12)
        w1[:12] = torch.eye(12)
        mod.filter_mlp[0].weight.copy_(w1)
        mod.filter_mlp[0].bias.zero_()
        mod.filter_mlp[2].weight.copy_(torch.eye(48))
        mod.filter_mlp[2].bias.zero_()
    rbf = torch.rand(5, 12)
    out = mod.filter_mlp(rbf)
    assert_close(out[:, :12], rbf, "filter first 12 channels")
    assert float(out[:, 12:].detach().abs().max()) == 0.0  # nothing else
    # negative pre-activation is really gated by ReLU
    with torch.no_grad():
        mod.filter_mlp[0].weight.copy_(-torch.eye(48, 12))
    neg = mod.filter_mlp(rbf)
    assert float(neg.detach().abs().max()) == 0.0


def test_no_neighbor_aggregates_exactly_zero(e3) -> None:
    """``m = 0`` for a node without a spatial neighbour (controlled MLP)."""
    batch = no_neighbor_batch()                    # 2 atoms, 10 Å apart
    mod = e3.SchnetSpatialEmbedding()
    _controlled(mod)
    h = torch.rand(2, 48) + 0.5
    with torch.no_grad():
        f = mod(h, batch)
    assert float(f.abs().max()) == 0.0             # m = 0 → ReLU(0) = 0
    diag = mod.last_diagnostics
    assert diag["zero_neighbor"] == 2.0
    assert diag["n_edges"] == 0.0


# ---------------------------------------------------------------------------
# F2 discipline + geometry sensitivity (§6.8 row 1)
# ---------------------------------------------------------------------------
def test_f2_masked_slots_are_never_read(e3) -> None:
    """Garbage in the F2-masked first 12 ``edge_attr`` slots changes NOTHING
    — distances/RBF come from ``pos`` only (§6.6.1)."""
    batch = sn2_forward_batch()
    h = random_h(int(batch["pos"].shape[0]), SEED)
    mod = e3.SchnetSpatialEmbedding()
    with torch.no_grad():
        base = mod(h, batch)
        poisoned = dict(batch)
        attr = batch["edge_attr"].clone()
        attr[:, :12] = 1234.5
        poisoned["edge_attr"] = attr
        out = mod(h, poisoned)
    assert torch.equal(base, out)                  # bitwise: slots unread


def test_distance_change_moves_delta_while_f2_stays_untouched(e3) -> None:
    """Same F2 topology (identical ``edge_attr`` incl. the zero mask), two
    current distances ⇒ different E3 delta (the plan's §6.8 row 1 check)."""
    near = bonded_pair_batch(1.0)
    far = bonded_pair_batch(2.0)
    assert torch.equal(near["edge_attr"], far["edge_attr"])
    assert torch.equal(near["edge_index"], far["edge_index"])
    assert bool((near["edge_attr"][:, :12] == 0).all())
    assert bool((far["edge_attr"][:, :12] == 0).all())

    h = random_h(2, SEED)
    mod = e3.SchnetSpatialEmbedding()
    before = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
              for k, v in near.items()}
    with torch.no_grad():
        d_near = mod(h, near)
        d_far = mod(h, far)
    assert not torch.allclose(d_near, d_far, rtol=REL_TOL, atol=ABS_TOL)
    assert float((d_near - d_far).abs().max()) > 1e-4
    # the forward NEVER mutates its input (F2 mask untouched by construction)
    for key, value in near.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, before[key]), key


# ---------------------------------------------------------------------------
# E(3) + permutation + batch consistency (§6.4 / §6.8)
# ---------------------------------------------------------------------------
def test_translation_rotation_reflection_invariance(e3) -> None:
    batch = sn2_bidirectional_batch(n_graphs=2)
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    mod = e3.SchnetSpatialEmbedding()
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
        assert_close(ref, moved, f"E3 {label}")


def test_permutation_equivariance(e3) -> None:
    batch = sn2_forward_batch()
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    gen = torch.Generator().manual_seed(17)
    perm = torch.randperm(n, generator=gen)
    permuted = permute_forward_batch(batch, perm)
    mod = e3.SchnetSpatialEmbedding()
    with torch.no_grad():
        ref = mod(h, batch)
        out = mod(h[perm], permuted)
    assert_close(ref[perm], out, "E3 permutation")


def test_independent_vs_mixed_batch(e3) -> None:
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
    mod = e3.SchnetSpatialEmbedding()
    with torch.no_grad():
        alone = mod(h_a, batch_a)
        mixed = mod(torch.cat([h_a, h_b]), batch_ab)
    # graph A occupies the first rows of the mixed batch
    assert int(batch_ab["graph_index"][:n_a].max()) == 0
    assert_close(alone, mixed[:n_a], "E3 independent-vs-mixed (A)")
    # graph B matches too (offset rows)
    with torch.no_grad():
        alone_b = mod(h_b, batch_b)
    assert_close(alone_b, mixed[n_a:n_a + n_b], "E3 independent-vs-mixed (B)")


def test_chunking_equivalence_is_bitwise(e3) -> None:
    """chunk_size only re-buckets the provider work: same weights ⇒
    bitwise-identical delta (WP3a guarantees provider bitwise identity)."""
    batch = sn2_bidirectional_batch(n_graphs=2)
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    default = e3.SchnetSpatialEmbedding()
    tiny = e3.SchnetSpatialEmbedding(chunk_size=1)
    tiny.load_state_dict(default.state_dict())
    with torch.no_grad():
        a = default(h, batch)
        b = tiny(h, batch)
        c = e3.SchnetSpatialEmbedding(chunk_size=7)
        c.load_state_dict(default.state_dict())
        d = c(h, batch)
    assert torch.equal(a, b)
    assert torch.equal(a, d)


# ---------------------------------------------------------------------------
# diagnostics switch (WP3a pattern)
# ---------------------------------------------------------------------------
def test_diagnostics_false_skips_count_only_helpers(
        monkeypatch: pytest.MonkeyPatch, e3) -> None:
    batch = sn2_bidirectional_batch(n_graphs=2)
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    cheap = e3.SchnetSpatialEmbedding(diagnostics_enabled=False)
    patch_count_only_helpers(monkeypatch)          # helpers now RAISE
    with torch.no_grad():
        out = cheap(h, batch)                      # must not raise
    assert bool(torch.isfinite(out).all())
    assert cheap.last_diagnostics == {}            # no counts at all

    # control: the default path really routes through those helpers
    loud = e3.SchnetSpatialEmbedding(diagnostics_enabled=True)
    with pytest.raises(AssertionError):
        with torch.no_grad():
            loud(h, batch)


def test_diagnostics_off_keeps_features_bitwise_identical(e3) -> None:
    batch = sn2_bidirectional_batch(n_graphs=2)
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    on = e3.SchnetSpatialEmbedding(diagnostics_enabled=True)
    off = e3.SchnetSpatialEmbedding(diagnostics_enabled=False)
    off.load_state_dict(on.state_dict())
    with torch.no_grad():
        a = on(h, batch)
        b = off(h, batch)
    assert torch.equal(a, b)
    assert on.last_diagnostics and off.last_diagnostics == {}


def test_diagnostics_are_detached_floats_and_copied(e3) -> None:
    batch = sn2_bidirectional_batch(n_graphs=2)
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    mod = e3.SchnetSpatialEmbedding()
    with torch.no_grad():
        mod(h, batch)
    diag = mod.last_diagnostics
    assert_all_floats(diag, "E3.last_diagnostics")
    for key in ("n_nodes", "n_edges", "zero_neighbor"):
        assert key in diag, key
    # provider counts are merged in (spatial view vocabulary)
    assert diag["n_atoms"] == float(n)
    assert diag["n_spatial_edges"] == diag["n_edges"]
    # a returned snapshot never aliases live state
    snapshot = mod.last_diagnostics
    snapshot["n_edges"] = -1.0
    assert mod.last_diagnostics["n_edges"] != -1.0
    # disabled ⇒ empty (and no tensor anywhere)
    mod.diagnostics_enabled = False
    with torch.no_grad():
        mod(h, batch)
    assert mod.last_diagnostics == {}


def test_provider_receives_chunk_size_and_flag(e3, monkeypatch) -> None:
    """The candidate forwards ``chunk_size`` / ``diagnostics`` verbatim."""
    seen: dict[str, object] = {}
    real = e3.build_spatial_graph

    def spy(batch, *, chunk_size=4096, diagnostics=True):
        seen["chunk_size"] = chunk_size
        seen["diagnostics"] = diagnostics
        return real(batch, chunk_size=chunk_size, diagnostics=diagnostics)

    monkeypatch.setattr(e3, "build_spatial_graph", spy)
    batch = sn2_forward_batch()
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    mod = e3.SchnetSpatialEmbedding(chunk_size=13, diagnostics_enabled=False)
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
def test_outputs_and_gradients_finite(e3, name: str) -> None:
    batch = geometric_batch_matrix()[name]
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    pos = batch["pos"].detach().clone().requires_grad_(True)
    work = dict(batch)
    work["pos"] = pos
    mod = e3.SchnetSpatialEmbedding()
    out = mod(h, work)
    assert tuple(out.shape) == (n, HIDDEN_V541)
    assert out.dtype == torch.float32
    assert bool(torch.isfinite(out).all())
    # the output satisfies the frozen residual contract
    delta, node_vector = normalize_branch_output(out, n_nodes=n)
    assert node_vector is None and tuple(delta.shape) == (n, HIDDEN_V541)
    (out * out).sum().backward()
    if pos.grad is not None:
        assert bool(torch.isfinite(pos.grad).all())
    for pname, p in mod.named_parameters():
        if p.grad is None:
            # a view with no spatial edge never reaches A/F — allowed here
            continue
        assert bool(torch.isfinite(p.grad).all()), pname


def test_all_parameter_gradients_present_and_finite(e3) -> None:
    """On a batch WITH spatial edges every parameter is in the graph and
    carries a finite (nonzero) gradient."""
    batch = sn2_forward_batch()
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    mod = e3.SchnetSpatialEmbedding()
    out = mod(h, batch)
    (out * out).sum().backward()
    total = 0.0
    for pname, p in mod.named_parameters():
        assert p.grad is not None, pname
        assert bool(torch.isfinite(p.grad).all()), pname
        total += float(p.grad.abs().sum())
    assert total > 0.0


def test_zero_length_pair_stays_finite(e3) -> None:
    """A degenerate (0 Å) chem pair is legal for the DISTANCE view: the
    envelope/RBF stay finite — E3 never needs a direction (§6.3.6 is an
    angle-provider rule)."""
    batch = zero_length_batch()
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    mod = e3.SchnetSpatialEmbedding()
    with torch.no_grad():
        out = mod(h, batch)
    assert bool(torch.isfinite(out).all())


def test_small_batch_cpu_budget(e3) -> None:
    """Bounded per-forward CPU cost on a small multi-graph batch.

    Measured ≈3 ms/forward on the dev box; the 2 s bound is a smoke
    guard against an accidental O(N²)-per-node regression, not a
    benchmark.
    """
    batch = sn2_bidirectional_batch(n_graphs=3)     # 18 atoms, 6 graphs
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    mod = e3.SchnetSpatialEmbedding()
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(3):
            out = mod(h, batch)
        elapsed = (time.perf_counter() - start) / 3.0
    assert bool(torch.isfinite(out).all())
    assert elapsed < 2.0, f"E3 forward took {elapsed * 1000:.0f} ms"


def test_eval_determinism(e3) -> None:
    batch = sn2_bidirectional_batch(n_graphs=2)
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    mod = e3.SchnetSpatialEmbedding()
    with torch.no_grad():
        a = mod(h, batch)
        b = mod(h, batch)
    assert torch.equal(a, b)
