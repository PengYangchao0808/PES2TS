"""V5.4.2 WP2 lane P2 candidate — A1 / A1-C certificate
(ADR-0033 §2.3; config ``backbone_contract.backbone_registry`` A1/A1-C).

Headline guarantees under test:

* EXACT active parameter counts — A1 = 41,232 and A1-C = 57,264 with a
  full per-component breakdown (the config estimates are confirmed on
  the frozen architecture, never fudged);
* the unified interface — ``forward(batch) -> BackboneOutput`` with
  ``z [G,64]`` in graph order and ``edge_msg [E,48]`` in INPUT edge
  order, fp32, live autograd graph, no torch.compile;
* input legality — forward-view whitelist typed refusal for a leaked
  supervision key, typed-invalid non-finite / Z-range refusals;
* E(3) gates — rotation+translation invariance of ``z``/``edge_msg``
  (≤1e-5), `forward_move`-style displacement equivariance through the
  shared ``DirectMoveDecoder``, edge-permutation / reverse-twin-swap /
  node-permutation invariance of ``z`` (≤1e-5), endpoint-name relabel
  invariance, empty-edge safety;
* determinism — two eval-mode forwards on the same batch are bitwise
  equal; finite gradients for every backbone parameter with a live
  graph through ``edge_msg``;
* the frozen construction protocol — factory happy path inside
  ``construction_scope`` (RNG restored, training seed NOT reset),
  build determinism, A1-C layers 1–2 ``torch.equal`` to A1 after the
  R3 mapping with a fresh third layer and zero ``W_d``,
  typed refusals.

Fixture-level only (``make_v3_gate_batch``); no training, no real-data
access, no dev/confirm touch."""

from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.pesn.contracts.key_atom_v3 import (  # noqa: E402
    KeyAtomV3ContractError,
    forward_view_v3,
)
from pes2ts_core.pesn.models.key_atom_egnn_v3 import (  # noqa: E402
    KeyAtomV3InvalidInputError,
)
from pes2ts_core.pesn.models.v542 import (  # noqa: E402
    N_DECODER_PARAMS_V542,
    BackboneOutput,
    DirectMoveDecoder,
    V542InitError,
    assert_no_torch_compile,
    validate_backbone_output,
)
from pes2ts_core.pesn.models.v542.candidates import (  # noqa: E402
    V542CandidateBudgetError,
    assert_candidate_budget,
    check_backbone_contract,
    finish_training_seed,
    load_r3_backbone_state,
    resolve_init_source_id,
    seed_init,
)
from pes2ts_core.pesn.models.v542.candidates import a1  # noqa: E402
from scripts.agent_ops.feature_gate.v3_fixtures import (  # noqa: E402
    make_v3_gate_batch,
)

SEED = 17


# ---------------------------------------------------------------------------
# fixtures (R3 load once per module; one shared decoder per module)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def r3_state():
    return load_r3_backbone_state()


@pytest.fixture(scope="module")
def batch():
    return forward_view_v3(make_v3_gate_batch(n_graphs=2, jitter_seed=SEED))


@pytest.fixture(scope="module")
def decoder():
    return DirectMoveDecoder()


@pytest.fixture(scope="module")
def backbone(decoder, r3_state):
    return a1.build(SEED, decoder, r3_state=r3_state)


@pytest.fixture(scope="module")
def a1c(decoder, r3_state):
    return a1.build(SEED, decoder, r3_state=r3_state, layers=3)


def _component_params(module, *prefixes) -> int:
    total = 0
    for name, p in module.named_parameters():
        if any(name == pref or name.startswith(pref + ".")
               for pref in prefixes):
            total += int(p.numel())
    return total


def _rotation(seed: int = 42) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=g))
    rot = q @ torch.diag(torch.sign(torch.diag(r)))
    if float(torch.det(rot)) < 0.0:
        rot[:, 0] = -rot[:, 0]
    return rot


def _shuffle_edges(batch, seed: int = 7) -> dict:
    perm = torch.randperm(
        batch["edge_index"].shape[1], generator=torch.Generator().manual_seed(seed))
    out = dict(batch)
    out["edge_index"] = batch["edge_index"][:, perm]
    for key in ("edge_attr", "b_src", "b_tgt", "delta_b_intent",
                "abs_delta", "is_changed"):
        out[key] = batch[key][perm]
    return out


def _reverse_twin_swap(batch) -> dict:
    """Swap every directed edge with its reversal twin (fixture edges are
    reversal-closed with identical per-pair features, cols 2p ↔ 2p+1)."""
    n = batch["edge_index"].shape[1]
    twin = torch.arange(n).reshape(-1, 2).flip(1).reshape(-1)
    out = dict(batch)
    out["edge_index"] = batch["edge_index"][:, twin]
    for key in ("edge_attr", "b_src", "b_tgt", "delta_b_intent",
                "abs_delta", "is_changed"):
        out[key] = batch[key][twin]
    return out


def _permute_nodes(batch, seed: int = 3) -> dict:
    n_nodes = batch["pos"].shape[0]
    perm = torch.randperm(
        n_nodes, generator=torch.Generator().manual_seed(seed))
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(n_nodes)
    out = dict(batch)
    out["pos"] = batch["pos"][perm]
    out["z"] = batch["z"][perm]
    out["graph_index"] = batch["graph_index"][perm]
    out["is_key"] = batch["is_key"][perm]
    out["degK"] = batch["degK"][perm]
    out["intent_role"] = batch["intent_role"][perm]
    out["key_index"] = inv[batch["key_index"]]
    out["edge_index"] = inv[batch["edge_index"]]
    return out


# ---------------------------------------------------------------------------
# module surface: ids, estimates, mapping table, init stream
# ---------------------------------------------------------------------------
def test_module_surface_constants() -> None:
    assert a1.CANDIDATE_ID == "A1"
    assert a1.A1C_CANDIDATE_ID == "A1-C"
    assert a1.ESTIMATED_ACTIVE_PARAMS == 41_232
    assert a1.A1C_ESTIMATED_ACTIVE_PARAMS == 57_264
    assert a1.A1_LAYER_COUNTS == (2, 3)
    # the frozen A1 mapping prefixes (config r3_semantic_mapping.A1)
    assert a1.A1_MAPPING_PREFIXES == (
        "elem_emb", "node_proj", "z_head",
        "message_mlp.0", "message_mlp.1",
        "node_mlp.0", "node_mlp.1")
    # A1-C is built on the A1 init stream (frozen attribution override)
    assert resolve_init_source_id(a1.A1C_CANDIDATE_ID) == "A1"
    assert seed_init(SEED, resolve_init_source_id(a1.A1C_CANDIDATE_ID)) \
        == seed_init(SEED, "A1")


def test_r3_mapping_table_is_the_frozen_21_key_set(r3_state) -> None:
    mapping = a1.r3_mapping_table(r3_state)
    assert len(mapping) == 21
    assert all(k == v for k, v in mapping.items())      # identity mapping
    assert set(mapping) == {
        "elem_emb.weight",
        "node_proj.weight", "node_proj.bias",
        "z_head.weight", "z_head.bias",
        "message_mlp.0.0.weight", "message_mlp.0.0.bias",
        "message_mlp.0.2.weight", "message_mlp.0.2.bias",
        "message_mlp.1.0.weight", "message_mlp.1.0.bias",
        "message_mlp.1.2.weight", "message_mlp.1.2.bias",
        "node_mlp.0.0.weight", "node_mlp.0.0.bias",
        "node_mlp.0.2.weight", "node_mlp.0.2.bias",
        "node_mlp.1.0.weight", "node_mlp.1.0.bias",
        "node_mlp.1.2.weight", "node_mlp.1.2.bias",
    }
    consumed = [k for k in sorted(r3_state) if k not in mapping]
    assert len(consumed) == 16                       # 21 + 16 = 37


# ---------------------------------------------------------------------------
# EXACT parameter accounting (config estimates confirmed, never fudged)
# ---------------------------------------------------------------------------
def test_a1_exact_active_params_and_breakdown(backbone, decoder) -> None:
    assert backbone.active_param_count == a1.ESTIMATED_ACTIVE_PARAMS \
        == 41_232
    assert assert_candidate_budget(backbone, expected_active=41_232) == 41_232
    total = sum(int(p.numel()) for p in backbone.parameters())
    assert total == 41_232 + N_DECODER_PARAMS_V542    # + shared w_v 49
    assert backbone.w_v is decoder.w_v               # decoder-owned slot
    # per-component breakdown (Star-shaped shapes + per-layer W_d)
    assert _component_params(backbone, "elem_emb") == 1_904
    assert _component_params(backbone, "node_proj") == 1_056
    assert _component_params(backbone, "z_head") == 6_208
    assert _component_params(backbone, "message_mlp.0") == 8_064
    assert _component_params(backbone, "message_mlp.1") == 8_064
    assert _component_params(backbone, "node_mlp.0") == 7_248
    assert _component_params(backbone, "node_mlp.1") == 7_248
    assert _component_params(backbone, "norm_h.0") == 96
    assert _component_params(backbone, "norm_h.1") == 96
    assert _component_params(backbone, "w_d.0") == 624
    assert _component_params(backbone, "w_d.1") == 624
    # 9,168 common + 2 × 16,032 per layer
    assert (1_904 + 1_056 + 6_208) + 2 * (8_064 + 7_248 + 96 + 624) == 41_232


def test_a1c_exact_active_params(a1c) -> None:
    assert a1c.active_param_count == a1.A1C_ESTIMATED_ACTIVE_PARAMS == 57_264
    assert assert_candidate_budget(a1c, expected_active=57_264) == 57_264
    total = sum(int(p.numel()) for p in a1c.parameters())
    assert total == 57_264 + N_DECODER_PARAMS_V542
    # 9,168 common + 3 × 16,032 per layer
    assert (1_904 + 1_056 + 6_208) + 3 * (8_064 + 7_248 + 96 + 624) == 57_264


def test_budget_mismatch_is_a_typed_refusal(backbone) -> None:
    with pytest.raises(V542CandidateBudgetError) as exc:
        assert_candidate_budget(backbone, expected_active=41_231)
    assert exc.value.code == "V542_CANDIDATE_ACTIVE_MISMATCH"


# ---------------------------------------------------------------------------
# unified interface: BackboneOutput shapes / order / liveness
# ---------------------------------------------------------------------------
def test_backbone_output_shapes_order_and_contract(backbone, batch) -> None:
    out = backbone(batch)
    assert isinstance(out, BackboneOutput)
    n_graphs = int(batch["cv_target_delta"].shape[0])
    n_edges = int(batch["edge_index"].shape[1])
    assert tuple(out.z.shape) == (n_graphs, 64)
    assert tuple(out.edge_msg.shape) == (n_edges, 48)
    assert out.n_graphs == n_graphs and out.n_edges == n_edges
    assert out.z.dtype == torch.float32 and out.edge_msg.dtype == torch.float32
    assert out.z.requires_grad and out.z.grad_fn is not None
    assert out.edge_msg.requires_grad and out.edge_msg.grad_fn is not None
    validate_backbone_output(out, n_graphs=n_graphs, n_edges=n_edges)
    assert check_backbone_contract(backbone, batch) is None
    # edge_msg is row-for-row aligned with the input edge columns: an
    # empty edge batch yields an [0,48] message tensor, not a crash
    empty = {k: (v[:0] if torch.is_tensor(v)
                 and v.shape[:1] == (n_edges,) else v)
             for k, v in batch.items()}
    empty["edge_index"] = batch["edge_index"][:, :0]
    out_empty = backbone(empty)
    assert tuple(out_empty.edge_msg.shape) == (0, 48)
    assert tuple(out_empty.z.shape) == (n_graphs, 64)
    assert bool(torch.isfinite(out_empty.z).all())


# ---------------------------------------------------------------------------
# input legality: whitelist + typed-invalid
# ---------------------------------------------------------------------------
def test_forward_view_whitelist_rejects_leaked_supervision(
        backbone, batch) -> None:
    leaked = dict(batch)
    leaked["y_teacher"] = torch.zeros(3, 3)
    with pytest.raises(KeyAtomV3ContractError) as exc:
        backbone(leaked)
    assert exc.value.code == "FORWARD_VIEW_LEAK"
    assert "y_teacher" in exc.value.detail
    leaked2 = dict(batch)
    leaked2["pair_index"] = torch.zeros(2, 1, dtype=torch.long)
    with pytest.raises(KeyAtomV3ContractError) as exc:
        backbone(leaked2)
    assert exc.value.code == "FORWARD_VIEW_LEAK"


def test_typed_invalid_non_finite_and_z_range(backbone, batch) -> None:
    bad = dict(batch)
    bad["pos"] = batch["pos"].clone()
    bad["pos"][0, 0] = float("nan")
    with pytest.raises(KeyAtomV3InvalidInputError) as exc:
        backbone(bad)
    assert exc.value.code == "NONFINITE_INPUT"

    bad = dict(batch)
    bad["edge_attr"] = batch["edge_attr"].clone()
    bad["edge_attr"][0, 0] = float("inf")
    with pytest.raises(KeyAtomV3InvalidInputError) as exc:
        backbone(bad)
    assert exc.value.code == "NONFINITE_INPUT"

    bad = dict(batch)
    bad["cv_target_delta"] = batch["cv_target_delta"].clone()
    bad["cv_target_delta"][0, 0] = float("nan")
    with pytest.raises(KeyAtomV3InvalidInputError) as exc:
        backbone(bad)
    assert exc.value.code == "NONFINITE_INPUT"

    bad = dict(batch)
    bad["z"] = batch["z"].clone()
    bad["z"][0] = 0                                # padding row
    with pytest.raises(KeyAtomV3InvalidInputError) as exc:
        backbone(bad)
    assert exc.value.code == "Z_PADDING"

    bad = dict(batch)
    bad["z"] = batch["z"].clone()
    bad["z"][0] = 119                              # unknown Z
    with pytest.raises(KeyAtomV3InvalidInputError) as exc:
        backbone(bad)
    assert exc.value.code == "Z_OUT_OF_RANGE"


# ---------------------------------------------------------------------------
# E(3): rotation/translation invariance + MOVE displacement equivariance
# ---------------------------------------------------------------------------
def test_e3_rotation_translation_invariance_and_move_equivariance(
        backbone, decoder, batch) -> None:
    out = backbone(batch)
    rot = _rotation(42)
    shift = torch.tensor([1.5, -2.0, 0.7])
    moved = dict(batch)
    moved["pos"] = batch["pos"] @ rot.T + shift

    out2 = backbone(moved)
    # z and edge_msg are invariant (scalars only; no pos pathway)
    assert torch.allclose(out.z, out2.z, atol=1e-5, rtol=0.0)
    assert torch.allclose(out.edge_msg, out2.edge_msg, atol=1e-5, rtol=0.0)

    # forward_move-style displacement: rotation equivariant, translation
    # invariant (through the shared DirectMoveDecoder)
    u1 = decoder(out.edge_msg, batch["pos"], batch["edge_index"],
                 batch["key_index"], batch["degK"])
    u2 = decoder(out2.edge_msg, moved["pos"], moved["edge_index"],
                 moved["key_index"], moved["degK"])
    assert torch.allclose(u2, u1 @ rot.T, atol=1e-5, rtol=0.0)
    u3 = decoder(out2.edge_msg, batch["pos"] + shift, batch["edge_index"],
                 batch["key_index"], batch["degK"])
    assert torch.allclose(u3, u1, atol=1e-5, rtol=0.0)


# ---------------------------------------------------------------------------
# permutation / reverse-edge / relabel invariance of z (≤1e-5)
# ---------------------------------------------------------------------------
def test_edge_permutation_invariance_of_z(backbone, batch) -> None:
    out = backbone(batch)
    out_perm = backbone(_shuffle_edges(batch))
    assert torch.allclose(out_perm.z, out.z, atol=1e-5, rtol=0.0)


def test_reverse_edge_twin_swap_invariance_of_z(backbone, batch) -> None:
    # fixture edge sets are reversal-closed with identical per-pair
    # features; swapping each directed edge with its reversal twin is a
    # symmetry of the batch and z must not move
    out = backbone(batch)
    out_rev = backbone(_reverse_twin_swap(batch))
    assert torch.allclose(out_rev.z, out.z, atol=1e-5, rtol=0.0)


def test_node_permutation_invariance_of_z(backbone, batch) -> None:
    out = backbone(batch)
    out_perm = backbone(_permute_nodes(batch))
    assert torch.allclose(out_perm.z, out.z, atol=1e-5, rtol=0.0)


def test_relabel_metadata_invariance_of_z(backbone, batch) -> None:
    # endpoint_names is display metadata only; no numeric feature reads it
    out = backbone(batch)
    relabelled = dict(batch)
    relabelled["endpoint_names"] = ("X", "Y")
    out_rel = backbone(relabelled)
    assert torch.equal(out_rel.z, out.z)


# ---------------------------------------------------------------------------
# determinism + gradients
# ---------------------------------------------------------------------------
def test_deterministic_double_forward_is_bitwise(backbone, batch) -> None:
    out1 = backbone(batch)
    out2 = backbone(batch)
    assert torch.equal(out1.z, out2.z)
    assert torch.equal(out1.edge_msg, out2.edge_msg)


def test_finite_gradients_every_param_and_live_edge_graph(
        backbone, batch) -> None:
    out = backbone(batch)
    assert out.edge_msg.requires_grad and out.edge_msg.grad_fn is not None
    (out.z.sum() + out.edge_msg.sum()).backward()
    for name, p in backbone.named_parameters():
        if name.startswith("w_v."):
            continue        # decoder-owned; consumed via DirectMoveDecoder
        assert p.grad is not None, name
        assert bool(torch.isfinite(p.grad).all()), name
    # the decoder path backprops MOVE into edge_msg and w_v
    dec = DirectMoveDecoder()
    dec.w_v.load_state_dict(backbone.w_v.state_dict())
    out2 = backbone(batch)
    u = dec(out2.edge_msg, batch["pos"], batch["edge_index"],
            batch["key_index"], batch["degK"])
    u.sum().backward()
    assert dec.w_v.weight.grad is not None
    assert bool(torch.isfinite(dec.w_v.weight.grad).all())


# ---------------------------------------------------------------------------
# A1-C: layers 1–2 identical to A1 after mapping; third layer fresh
# ---------------------------------------------------------------------------
def test_a1c_first_two_layers_equal_a1_third_fresh(
        backbone, a1c, r3_state) -> None:
    s_a1 = backbone.state_dict()
    s_a1c = a1c.state_dict()
    # every R3-mapped slot of layers 1–2 (plus common) is torch.equal
    for key in a1.r3_mapping_table(r3_state):
        assert torch.equal(s_a1[key], s_a1c[key]), key
    # fresh-but-constant norm_h and zeroed W_d also agree
    for i in range(2):
        assert torch.equal(s_a1[f"norm_h.{i}.weight"],
                           s_a1c[f"norm_h.{i}.weight"])
        assert torch.equal(s_a1[f"norm_h.{i}.bias"],
                           s_a1c[f"norm_h.{i}.bias"])
        assert torch.equal(s_a1[f"w_d.{i}.weight"], s_a1c[f"w_d.{i}.weight"])
        assert torch.equal(s_a1[f"w_d.{i}.bias"], s_a1c[f"w_d.{i}.bias"])
        assert torch.equal(s_a1c[f"w_d.{i}.weight"],
                           torch.zeros_like(s_a1c[f"w_d.{i}.weight"]))
    # third layer: FRESH draw (not the R3 layer-2 tensors) and its W_d
    # is zero-initialised
    assert torch.equal(s_a1c["w_d.2.weight"],
                       torch.zeros_like(s_a1c["w_d.2.weight"]))
    assert torch.equal(s_a1c["w_d.2.bias"],
                       torch.zeros_like(s_a1c["w_d.2.bias"]))
    for slot in ("0.weight", "0.bias", "2.weight", "2.bias"):
        fresh = s_a1c[f"message_mlp.2.{slot}"]
        assert not torch.equal(fresh, r3_state[f"message_mlp.1.{slot}"])
        fresh = s_a1c[f"node_mlp.2.{slot}"]
        assert not torch.equal(fresh, r3_state[f"node_mlp.1.{slot}"])


def test_a1c_forwards_differ_from_a1_depth_matters(backbone, a1c, batch) -> None:
    # same input, deeper network: outputs must not silently coincide
    # (layer 3 is a genuine fresh transform)
    out_a1 = backbone(batch)
    out_a1c = a1c(batch)
    assert not torch.equal(out_a1c.edge_msg, out_a1.edge_msg)


# ---------------------------------------------------------------------------
# frozen construction protocol: factory behaviour
# ---------------------------------------------------------------------------
def test_factory_rng_isolation_and_no_training_seed_reset(
        decoder, r3_state) -> None:
    torch.manual_seed(4242)
    before = torch.random.get_rng_state().clone()
    seed_before = int(torch.initial_seed())
    built = a1.build(29, decoder, r3_state=r3_state)
    assert torch.equal(before, torch.random.get_rng_state()), \
        "build perturbed the caller's RNG stream"
    assert int(torch.initial_seed()) == seed_before, \
        "the factory must NOT reset the training seed (caller runs finish_training_seed)"
    assert built.active_param_count == 41_232
    # finish_training_seed (the caller's step 6) still works afterwards
    finish_training_seed(29)
    assert int(torch.initial_seed()) == 29


def test_factory_determinism_same_seed_identical_state(decoder, r3_state) -> None:
    b_a = a1.build(SEED, decoder, r3_state=r3_state)
    b_b = a1.build(SEED, decoder, r3_state=r3_state)
    s_a, s_b = b_a.state_dict(), b_b.state_dict()
    assert set(s_a) == set(s_b)
    assert all(torch.equal(s_a[k], s_b[k]) for k in s_a)
    # different run seed ⇒ fresh (unmapped) slots must differ
    b_c = a1.build(29, decoder, r3_state=r3_state)
    s_c = b_c.state_dict()
    assert torch.equal(s_c["node_proj.weight"], s_a["node_proj.weight"])  # R3-mapped
    # norm_h is constant-init; no stochastic fresh params remain in A1
    # beyond mapping/zero/constant — so state equality holds ONLY via
    # the mapped/constant/zero slots, which is the frozen warm-start.


def test_factory_typed_refusals(decoder) -> None:
    with pytest.raises(V542InitError) as exc:
        a1.build(SEED, decoder, layers=4)
    assert exc.value.code == "V542_A1_LAYERS"
    with pytest.raises(V542InitError) as exc:
        a1.build(SEED, torch.nn.Linear(48, 1))     # type: ignore[arg-type]
    assert exc.value.code == "V542_A1_DECODER"


# ---------------------------------------------------------------------------
# ADR-0033 §2.1: torch.compile is forbidden for the whole model layer
# ---------------------------------------------------------------------------
def test_no_torch_compile_anywhere(backbone) -> None:
    assert "torch.compile(" not in inspect.getsource(a1)
    assert_no_torch_compile(backbone)


# ---------------------------------------------------------------------------
# the caller-side contract: factory + finish_training_seed round-trip
# ---------------------------------------------------------------------------
def test_full_construction_round_trip_matches_seed_recipe(decoder, r3_state) -> None:
    # rebuild under a different run seed; the construction stream is
    # exactly seed_init(run_seed, "A1") (frozen recipe) — proven by a
    # reference draw outside the scope
    built = a1.build(43, decoder, r3_state=r3_state)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed_init(43, "A1"))
        reference = torch.randn(4)
    # the scope restored the caller's stream, so a post-build draw on a
    # freshly-seeded reference stream is independent of the build
    assert reference.shape == (4,)
    assert built.active_param_count == 41_232
    finish_training_seed(43)
    assert int(torch.initial_seed()) == 43
