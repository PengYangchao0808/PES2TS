"""V5.4.2 WP2 lane P2-3 — candidate A4 certificate (ADR-0033 §2.3;
Oracle architecture §2.4; config ``candidates.A4`` /
``init_mapping.r3_semantic_mapping.A4`` /
``per_candidate_zero_init.A4``).

Headline guarantees under test:

* EXACT active parameters — measured ``55,840`` (config estimate,
  asserted at build; per-slot breakdown reported, never padded);
* the frozen R3 semantic mapping — ``elem_emb``/``node_proj``/``z_head``
  identity + ``edge_readout`` ← R3 ``message_mlp.2``, all 37 payload
  keys accounted exactly once (strict audit), shared ``w_v`` equality;
* zero-init special parameters (``W_g``, attention output projection,
  FFN output projection, ``b_h`` RBF12 bias) and the resulting IDENTITY
  property of the transformer stack at init;
* the ``BackboneOutput`` contract — ``z [G,64]`` in graph order,
  ``edge_msg [E,48]`` in INPUT edge order, fp32, finite, live autograd,
  no torch.compile anywhere;
* graph-boundary isolation (joint batch ≡ separate runs), node
  permutation invariance of ``z``, edge-order-consistent ``edge_msg``,
  E(3) invariance, bitwise determinism;
* forward-view whitelist rejection and the legacy typed-invalid
  taxonomy (NONFINITE_INPUT / Z_PADDING / Z_OUT_OF_RANGE);
* finite gradients for EVERY parameter (including the reaction token
  and all attention params), with the token/attention paths proven LIVE
  once the zero-init projections are moved off zero;
* the construction protocol — RNG isolation, no training-seed reset
  inside ``build``, seed determinism.

Fixture-level only (``make_v3_gate_batch`` + one hand-crafted
empty-edge batch); no training, no real-data access, no dev/confirm
touch."""

from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.pesn.contracts.key_atom_v3 import (  # noqa: E402
    EDGE_ABS_DELTA_KEY,
    EDGE_ATTR_KEY,
    EDGE_B_SRC_KEY,
    EDGE_B_TGT_KEY,
    EDGE_DELTA_B_KEY,
    EDGE_INDEX_KEY,
    EDGE_IS_CHANGED_KEY,
    GRAPH_CV_DELTA_KEY,
    GRAPH_SIGNED_INTENT_KEY,
    KeyAtomV3ContractError,
    NODE_DEGK_KEY,
    NODE_GRAPH_KEY,
    NODE_INTENT_ROLE_KEY,
    NODE_IS_KEY_KEY,
    NODE_KEY_SET_KEY,
    NODE_POS_KEY,
    NODE_Z_KEY,
    forward_view_v3,
)
from pes2ts_core.pesn.models.key_atom_egnn_v3 import (  # noqa: E402
    KeyAtomV3InvalidInputError,
    StarEGNNV3,
)
from pes2ts_core.pesn.models.v542 import (  # noqa: E402
    N_DECODER_PARAMS_V542,
    N_PROCESS_PARAMS_V542,
    BackboneOutput,
    DirectMoveDecoder,
    V542InitError,
    assert_no_torch_compile,
)
from pes2ts_core.pesn.models.v542.candidates import (  # noqa: E402
    V542CandidateBudgetError,
    apply_r3_mapping,
    assert_candidate_budget,
    check_backbone_contract,
    load_r3_backbone_state,
)
from pes2ts_core.pesn.models.v542.candidates.a4 import (  # noqa: E402
    A4_CONSUMED_SOURCE_KEYS,
    A4_LAYER_PARAMS,
    A4_N_HEADS,
    A4_N_LAYERS,
    A4_R3_MAPPING,
    CANDIDATE_ID,
    ESTIMATED_ACTIVE_PARAMS,
    SparseGeometryTransformerA4,
    active_param_breakdown,
    build,
)
from scripts.agent_ops.feature_gate.v3_fixtures import (  # noqa: E402
    make_v3_gate_batch,
)

SEED = 17
#: Frozen node-row keys (batch concatenation / permutation helpers).
_NODE_ROW_KEYS = (NODE_Z_KEY, NODE_POS_KEY, NODE_GRAPH_KEY, NODE_IS_KEY_KEY,
                  NODE_DEGK_KEY, NODE_INTENT_ROLE_KEY)
#: Frozen edge-row keys aligned with the ``edge_index`` columns.
_EDGE_ROW_KEYS = (EDGE_ATTR_KEY, EDGE_B_SRC_KEY, EDGE_B_TGT_KEY,
                  EDGE_DELTA_B_KEY, EDGE_ABS_DELTA_KEY, EDGE_IS_CHANGED_KEY)
#: Frozen graph-row keys.
_GRAPH_ROW_KEYS = (GRAPH_CV_DELTA_KEY, GRAPH_SIGNED_INTENT_KEY)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def r3_state():
    return load_r3_backbone_state()


@pytest.fixture()
def move_decoder():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31337)
        return DirectMoveDecoder()


@pytest.fixture()
def backbone(move_decoder):
    return build(SEED, move_decoder)


@pytest.fixture()
def batch():
    return forward_view_v3(make_v3_gate_batch(n_graphs=2, jitter_seed=SEED))


# ---------------------------------------------------------------------------
# batch helpers (block concatenation, node permutation, edge shuffle, E(3))
# ---------------------------------------------------------------------------
def _concat_batches(b1: dict, b2: dict) -> dict:
    """Block-concatenate two independent batches into ONE batch.

    Node/edge/graph row tensors are concatenated in order; ``graph_index``
    of the second batch is offset by G1, ``key_index``/``edge_index`` by
    N1.  ``endpoint_names`` (harness metadata, never a model input) is
    dropped.  Relative row orders inside each block are preserved, so
    per-segment accumulation order matches the separate runs.
    """
    n1 = int(b1[NODE_POS_KEY].shape[0])
    g1 = int(b1[GRAPH_CV_DELTA_KEY].shape[0])
    out: dict = {}
    for key in _NODE_ROW_KEYS:
        out[key] = torch.cat([b1[key], b2[key]], dim=0)
    out[NODE_GRAPH_KEY] = torch.cat(
        [b1[NODE_GRAPH_KEY], b2[NODE_GRAPH_KEY] + g1], dim=0)
    out[NODE_KEY_SET_KEY] = torch.cat(
        [b1[NODE_KEY_SET_KEY], b2[NODE_KEY_SET_KEY] + n1], dim=0)
    out[EDGE_INDEX_KEY] = torch.cat(
        [b1[EDGE_INDEX_KEY], b2[EDGE_INDEX_KEY] + n1], dim=1)
    for key in _EDGE_ROW_KEYS:
        out[key] = torch.cat([b1[key], b2[key]], dim=0)
    for key in _GRAPH_ROW_KEYS:
        out[key] = torch.cat([b1[key], b2[key]], dim=0)
    return out


def _permute_batch_nodes(batch: dict, perm: torch.Tensor) -> dict:
    """Relabel nodes: new node row ``i`` is old node row ``perm[i]``.

    ``edge_index``/``key_index`` are remapped through the inverse; every
    node-row tensor follows ``perm``.  Edge columns keep their order.
    """
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(int(perm.numel()))
    out = dict(batch)
    for key in _NODE_ROW_KEYS:
        out[key] = batch[key][perm]
    out[NODE_KEY_SET_KEY] = inv[batch[NODE_KEY_SET_KEY]]
    out[EDGE_INDEX_KEY] = inv[batch[EDGE_INDEX_KEY]]
    return out


def _shuffle_batch_edges(batch: dict, seed: int) -> tuple[dict, torch.Tensor]:
    """Shuffle edge COLUMNS (all edge-row tensors synchronously)."""
    generator = torch.Generator().manual_seed(seed)
    n_edges = int(batch[EDGE_INDEX_KEY].shape[1])
    perm = torch.randperm(n_edges, generator=generator)
    out = dict(batch)
    out[EDGE_INDEX_KEY] = batch[EDGE_INDEX_KEY][:, perm]
    for key in _EDGE_ROW_KEYS:
        out[key] = batch[key][perm]
    return out, perm


def _transform_positions(batch: dict, seed: int, *, reflect: bool) -> dict:
    """Apply a random O(3) transform (+ translation) to ``pos`` only."""
    generator = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    q = q @ torch.diag(torch.sign(torch.diag(r)))
    if reflect:
        q = q @ torch.diag(torch.tensor([1.0, 1.0, -1.0]))
    shift = torch.randn(3, generator=generator)
    out = dict(batch)
    out[NODE_POS_KEY] = batch[NODE_POS_KEY] @ q.T + shift
    return out


def _handcrafted_batch() -> dict:
    """Minimal legal batch: graph 0 (3 nodes, one edge 0→1) + graph 1
    (ONE isolated node, ZERO edges) — exercises empty real-in-edge
    segments (nodes attend only their token) and a zero-edge graph."""
    return {
        NODE_Z_KEY: torch.tensor([6, 7, 8, 10], dtype=torch.long),
        NODE_POS_KEY: torch.tensor(
            [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0],
             [0.0, 1.1, 0.0], [0.5, -0.7, 0.3]]),
        NODE_GRAPH_KEY: torch.tensor([0, 0, 0, 1], dtype=torch.long),
        NODE_KEY_SET_KEY: torch.tensor([0, 1, 3], dtype=torch.long),
        NODE_IS_KEY_KEY: torch.tensor([1.0, 1.0, 0.0, 1.0]),
        NODE_DEGK_KEY: torch.tensor([1.0, 1.0, 0.0, 1.0]),
        NODE_INTENT_ROLE_KEY: torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        EDGE_INDEX_KEY: torch.tensor([[0], [1]], dtype=torch.long),
        EDGE_ATTR_KEY: torch.zeros(1, 17),
        EDGE_B_SRC_KEY: torch.tensor([1.0]),
        EDGE_B_TGT_KEY: torch.tensor([0.0]),
        EDGE_DELTA_B_KEY: torch.tensor([-1.0]),
        EDGE_ABS_DELTA_KEY: torch.tensor([1.0]),
        EDGE_IS_CHANGED_KEY: torch.tensor([1.0]),
        GRAPH_CV_DELTA_KEY: torch.zeros(2, 4),
        GRAPH_SIGNED_INTENT_KEY: torch.tensor([1.0, -1.0]),
    }


# ---------------------------------------------------------------------------
# frozen identity constants + exact parameter count
# ---------------------------------------------------------------------------
def test_candidate_identity_constants_are_frozen() -> None:
    assert CANDIDATE_ID == "A4"
    assert ESTIMATED_ACTIVE_PARAMS == 55_840            # config estimate
    assert A4_N_LAYERS == 2 and A4_N_HEADS == 4
    assert A4_LAYER_PARAMS == 19_016
    # joint mirrors the Oracle budget table (58,067)
    assert (ESTIMATED_ACTIVE_PARAMS + N_DECODER_PARAMS_V542
            + N_PROCESS_PARAMS_V542) == 58_067


def test_exact_active_params_and_breakdown(backbone) -> None:
    assert isinstance(backbone, SparseGeometryTransformerA4)
    measured = backbone.active_param_count
    assert measured == ESTIMATED_ACTIVE_PARAMS == 55_840
    # the frozen per-slot breakdown (Oracle §2.4 accounting)
    breakdown = active_param_breakdown(backbone)
    assert breakdown == {
        "edge_readout": 8_064,      # (118→48→48) ← R3 message_mlp.2
        "elem_emb": 1_904,          # Embedding(119,16)
        "layers": 38_032,           # 2 × 19,016
        "node_proj": 1_056,         # Linear(21→48)
        "token_init": 288,          # Linear(5→48)
        "w_g": 288,                 # Linear(5→48) zero-init
        "z_head": 6_208,            # Linear(96→64)
    }
    assert sum(breakdown.values()) == 55_840
    for layer in backbone.layers:
        assert sum(int(p.numel()) for p in layer.parameters()) == \
            A4_LAYER_PARAMS
    # the construction-time budget gate asserts the EXACT count …
    assert assert_candidate_budget(backbone,
                                   expected_active=55_840) == 55_840
    # … and a drifted expectation is a typed STOP (no padding/shrinking)
    with pytest.raises(V542CandidateBudgetError) as exc:
        assert_candidate_budget(backbone, expected_active=41_232)
    assert exc.value.code == "V542_CANDIDATE_ACTIVE_MISMATCH"


# ---------------------------------------------------------------------------
# R3 semantic mapping, zero-inits, shared w_v (config init_policy)
# ---------------------------------------------------------------------------
def test_mapping_applies_r3_tensors_and_zero_inits(
        backbone, move_decoder, r3_state) -> None:
    state = backbone.state_dict()
    # mapped targets are torch.equal to the R3 payload …
    for target, source in A4_R3_MAPPING.items():
        assert torch.equal(state[target], r3_state[source]), target
    # … and the R3-mapped edge readout is the Star message MLP.2
    assert torch.equal(backbone.edge_readout[0].weight,
                       r3_state["message_mlp.2.0.weight"])
    assert torch.equal(backbone.edge_readout[2].bias,
                       r3_state["message_mlp.2.2.bias"])
    # special zero-inits (config per_candidate_zero_init.A4 + Oracle §2.4)
    assert torch.equal(backbone.w_g.weight, torch.zeros(48, 5))
    assert torch.equal(backbone.w_g.bias, torch.zeros(48))
    for layer in backbone.layers:
        assert torch.equal(layer.out_proj.weight, torch.zeros(48, 48))
        assert torch.equal(layer.out_proj.bias, torch.zeros(48))
        assert torch.equal(layer.ffn_out.weight, torch.zeros(48, 96))
        assert torch.equal(layer.ffn_out.bias, torch.zeros(48))
        assert torch.equal(layer.edge_bias.weight, torch.zeros(4, 12))
        assert torch.equal(layer.edge_bias.bias, torch.zeros(4))
    # shared w_v: module-object identity + R3 values + dual-key storage
    assert backbone.w_v is move_decoder.w_v
    assert torch.equal(backbone.w_v.weight, r3_state["w_v.weight"])
    assert torch.equal(backbone.w_v.bias, r3_state["w_v.bias"])
    sd = backbone.state_dict()
    assert sd["w_v.weight"].data_ptr() == move_decoder.w_v.weight.data_ptr()
    # active params exclude the decoder-owned 49
    assert (backbone.active_param_count
            == sum(int(p.numel()) for p in backbone.parameters())
            - N_DECODER_PARAMS_V542)


def test_mapping_audit_accounts_all_37_keys(backbone, r3_state) -> None:
    assert len(r3_state) == 37
    assert len(A4_R3_MAPPING) == 9
    assert len(A4_CONSUMED_SOURCE_KEYS) == 28
    # every payload key accounted exactly once: mapping ∪ consumed
    assert set(A4_R3_MAPPING.values()) | set(A4_CONSUMED_SOURCE_KEYS) == \
        set(r3_state)
    assert not (set(A4_R3_MAPPING.values()) & set(A4_CONSUMED_SOURCE_KEYS))
    # the audit passes with all four loader-mirroring categories empty
    audit = apply_r3_mapping(
        backbone, r3_state, A4_R3_MAPPING,
        consumed_source_keys=list(A4_CONSUMED_SOURCE_KEYS))
    assert audit.missing == ()
    assert audit.unexpected == ()
    assert audit.shape_mismatch == ()
    assert audit.dtype_mismatch == ()
    assert len(audit.torch_equal_verified) == 9


def test_fresh_attention_params_are_seed_dependent(r3_state) -> None:
    decoder_a, decoder_b = DirectMoveDecoder(), DirectMoveDecoder()
    bb_a = build(17, decoder_a)
    bb_b = build(29, decoder_b)
    # R3-mapped slots identical across seeds (warm-start) …
    assert torch.equal(bb_a.edge_readout[0].weight,
                       bb_b.edge_readout[0].weight)
    assert torch.equal(bb_a.z_head.weight, bb_b.z_head.weight)
    # … fresh attention/token params follow the seed_init stream …
    assert not torch.equal(bb_a.layers[0].qkv.weight,
                           bb_b.layers[0].qkv.weight)
    assert not torch.equal(bb_a.token_init.weight, bb_b.token_init.weight)
    # … and the synthetic-state injection path mirrors the guarantees
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4242)
        synthetic = StarEGNNV3().state_dict()
    bb_c = build(31, DirectMoveDecoder(), r3_state=synthetic)
    assert torch.equal(bb_c.edge_readout[2].weight,
                       synthetic["message_mlp.2.2.weight"])
    assert torch.equal(bb_c.w_v.weight, synthetic["w_v.weight"])
    assert torch.equal(bb_c.elem_emb.weight, synthetic["elem_emb.weight"])


def test_build_refuses_non_decoder(move_decoder) -> None:
    with pytest.raises(V542InitError) as exc:
        build(17, torch.nn.Linear(48, 1))      # type: ignore[arg-type]
    assert exc.value.code == "V542_DECODER_TYPE"


# ---------------------------------------------------------------------------
# output shapes / order / backbone contract
# ---------------------------------------------------------------------------
def test_output_shapes_order_and_contract(backbone, batch) -> None:
    n_graphs = int(batch[GRAPH_CV_DELTA_KEY].shape[0])
    n_edges = int(batch[EDGE_INDEX_KEY].shape[1])
    out = backbone(batch)
    assert isinstance(out, BackboneOutput)
    assert tuple(out.z.shape) == (n_graphs, 64)
    assert tuple(out.edge_msg.shape) == (n_edges, 48)
    assert out.z.dtype == torch.float32 and out.edge_msg.dtype == torch.float32
    assert bool(torch.isfinite(out.z).all())
    assert bool(torch.isfinite(out.edge_msg).all())
    assert out.n_graphs == n_graphs and out.n_edges == n_edges
    # live autograd graph + no torch.compile: the frozen wrapper check
    assert check_backbone_contract(backbone, batch) is None
    assert out.z.requires_grad and out.z.grad_fn is not None
    assert out.edge_msg.requires_grad and out.edge_msg.grad_fn is not None


def test_graph_boundary_isolation(backbone) -> None:
    batch_a = forward_view_v3(
        make_v3_gate_batch(n_graphs=1, jitter_seed=101))
    batch_b = forward_view_v3(
        make_v3_gate_batch(n_graphs=1, jitter_seed=202))
    joint = _concat_batches(batch_a, batch_b)
    out_a, out_b = backbone(batch_a), backbone(batch_b)
    out_j = backbone(joint)
    g_a = int(batch_a[GRAPH_CV_DELTA_KEY].shape[0])
    e_a = int(batch_a[EDGE_INDEX_KEY].shape[1])
    # per-graph z identical to the separate runs (no cross-graph mixing)
    assert torch.allclose(out_j.z[:g_a], out_a.z, atol=1e-6, rtol=0.0)
    assert torch.allclose(out_j.z[g_a:], out_b.z, atol=1e-6, rtol=0.0)
    # per-edge edge_msg identical, in the concatenated input edge order
    assert torch.allclose(out_j.edge_msg[:e_a], out_a.edge_msg,
                          atol=1e-6, rtol=0.0)
    assert torch.allclose(out_j.edge_msg[e_a:], out_b.edge_msg,
                          atol=1e-6, rtol=0.0)


def test_node_permutation_invariance(backbone, batch) -> None:
    out = backbone(batch)
    generator = torch.Generator().manual_seed(7)
    n_nodes = int(batch[NODE_POS_KEY].shape[0])
    perm = torch.randperm(n_nodes, generator=generator)
    out_p = backbone(_permute_batch_nodes(batch, perm))
    # z is permutation-INVARIANT; edge_msg (edge columns unmoved) is
    # edge-order-consistent — both within the frozen 1e-5 tolerance
    assert torch.allclose(out_p.z, out.z, atol=1e-5, rtol=0.0)
    assert torch.allclose(out_p.edge_msg, out.edge_msg,
                          atol=1e-5, rtol=0.0)


def test_edge_permutation_consistency(backbone, batch) -> None:
    out = backbone(batch)
    shuffled, perm = _shuffle_batch_edges(batch, seed=11)
    out_s = backbone(shuffled)
    # edge_msg rows follow the shuffled input edge order exactly
    assert torch.allclose(out_s.edge_msg, out.edge_msg[perm],
                          atol=1e-6, rtol=0.0)
    assert torch.allclose(out_s.z, out.z, atol=1e-5, rtol=0.0)


def test_e3_invariance(backbone, batch) -> None:
    out = backbone(batch)
    for seed, reflect in ((3, False), (4, True)):   # rotation; reflection
        out_t = backbone(_transform_positions(batch, seed, reflect=reflect))
        # backbone outputs are pure invariants (positions are never
        # consumed by the math; no absolute coordinates, no vector
        # channels) — MOVE equivariance belongs to the shared decoder
        assert torch.allclose(out_t.z, out.z, atol=1e-7, rtol=0.0)
        assert torch.allclose(out_t.edge_msg, out.edge_msg,
                              atol=1e-7, rtol=0.0)


def test_determinism_and_no_dropout(backbone, batch) -> None:
    assert not any(isinstance(m, torch.nn.Dropout)
                   for m in backbone.modules())
    out1, out2 = backbone(batch), backbone(batch)
    assert torch.equal(out1.z, out2.z)
    assert torch.equal(out1.edge_msg, out2.edge_msg)
    backbone.eval()
    out3 = backbone(batch)
    assert torch.equal(out1.z, out3.z)
    assert torch.equal(out1.edge_msg, out3.edge_msg)


# ---------------------------------------------------------------------------
# input legality: whitelist + typed invalid (legacy v3 taxonomy)
# ---------------------------------------------------------------------------
def test_forward_view_whitelist_rejection(backbone, batch) -> None:
    leaked = dict(batch)
    leaked["y_teacher"] = torch.zeros(3, 3)         # supervision key!
    with pytest.raises(KeyAtomV3ContractError) as exc:
        backbone(leaked)
    assert exc.value.code == "FORWARD_VIEW_LEAK"


def test_typed_invalid_nonfinite_and_z_range(backbone, batch) -> None:
    bad = dict(batch)
    bad[NODE_POS_KEY] = batch[NODE_POS_KEY].clone()
    bad[NODE_POS_KEY][0, 0] = float("nan")
    with pytest.raises(KeyAtomV3InvalidInputError) as exc:
        backbone(bad)
    assert exc.value.code == "NONFINITE_INPUT"

    bad = dict(batch)
    bad[EDGE_ATTR_KEY] = batch[EDGE_ATTR_KEY].clone()
    bad[EDGE_ATTR_KEY][0, 0] = float("inf")
    with pytest.raises(KeyAtomV3InvalidInputError) as exc:
        backbone(bad)
    assert exc.value.code == "NONFINITE_INPUT"

    bad = dict(batch)
    bad[NODE_Z_KEY] = batch[NODE_Z_KEY].clone()
    bad[NODE_Z_KEY][0] = 0                          # padding row
    with pytest.raises(KeyAtomV3InvalidInputError) as exc:
        backbone(bad)
    assert exc.value.code == "Z_PADDING"

    bad = dict(batch)
    bad[NODE_Z_KEY] = batch[NODE_Z_KEY].clone()
    bad[NODE_Z_KEY][0] = 119                        # unknown Z
    with pytest.raises(KeyAtomV3InvalidInputError) as exc:
        backbone(bad)
    assert exc.value.code == "Z_OUT_OF_RANGE"


def test_empty_in_edges_and_zero_edge_graph_finite(backbone) -> None:
    tiny = _handcrafted_batch()     # isolated nodes + a zero-edge graph
    out = backbone(tiny)
    assert tuple(out.z.shape) == (2, 64)
    assert tuple(out.edge_msg.shape) == (1, 48)
    assert bool(torch.isfinite(out.z).all())
    assert bool(torch.isfinite(out.edge_msg).all())
    assert check_backbone_contract(backbone, tiny) is None
    # a batch with NO edges at all must also stay finite
    no_edges = dict(tiny)
    no_edges[EDGE_INDEX_KEY] = torch.zeros(2, 0, dtype=torch.long)
    for key in _EDGE_ROW_KEYS:
        no_edges[key] = tiny[key][:0]
    out0 = backbone(no_edges)
    assert tuple(out0.edge_msg.shape) == (0, 48)
    assert bool(torch.isfinite(out0.z).all())


# ---------------------------------------------------------------------------
# gradients: finite for every param; token/attention paths live
# ---------------------------------------------------------------------------
def test_finite_gradients_every_parameter(backbone, batch) -> None:
    out = backbone(batch)
    (out.z.sum() + out.edge_msg.sum()).backward()
    active = [(n, p) for n, p in backbone.named_parameters()
              if not n.startswith("w_v.")]
    assert active, "no active parameters?"
    for name, param in active:
        assert param.grad is not None, name
        assert bool(torch.isfinite(param.grad).all()), name
    # the token and attention params receive (possibly zero-at-init but
    # well-defined) gradients through the zero-init output projections
    grads = dict(active)
    for name in ("token_init.weight", "token_init.bias",
                 "layers.0.qkv.weight", "layers.0.qkv.bias",
                 "layers.0.token_bias", "layers.1.qkv.weight",
                 "w_g.weight", "w_g.bias", "edge_readout.0.weight"):
        assert name in grads, name
        assert grads[name].grad is not None, name


def test_token_and_attention_paths_live_after_unzeroing(backbone, batch) -> None:
    # simulate a trained state: move the zero-init output projections off
    # zero, then prove gradients REACH the token and attention params
    with torch.no_grad():
        for layer in backbone.layers:
            layer.out_proj.weight.copy_(
                torch.randn_like(layer.out_proj.weight))
            layer.ffn_out.weight.copy_(
                torch.randn_like(layer.ffn_out.weight))
    out = backbone(batch)
    (out.z.sum() + out.edge_msg.sum()).backward()
    grads = dict(backbone.named_parameters())
    for name in ("layers.0.qkv.weight", "layers.1.qkv.weight",
                 "token_init.weight", "layers.0.token_bias",
                 "layers.0.edge_bias.weight", "w_g.weight"):
        grad = grads[name].grad
        assert grad is not None and bool(torch.isfinite(grad).all()), name
        assert float(grad.abs().sum()) > 0.0, f"{name} path is dead"


# ---------------------------------------------------------------------------
# zero-init identity property at init (frozen recipe implication)
# ---------------------------------------------------------------------------
def test_zero_init_transformer_is_identity_at_init(backbone, batch) -> None:
    out = backbone(batch)
    # manual reference: h0 (W_g = 0) → edge_readout → z_head, no
    # transformer contribution at all
    with torch.no_grad():
        dtype = torch.float32
        z_atomic = batch[NODE_Z_KEY]
        is_key = batch[NODE_IS_KEY_KEY].to(dtype)
        degk = batch[NODE_DEGK_KEY].to(dtype)
        role = batch[NODE_INTENT_ROLE_KEY].to(dtype)
        degk_norm = degk.clamp(min=0.0, max=4.0) / 4.0
        h0 = backbone.node_proj(torch.cat(
            [backbone.elem_emb(z_atomic),
             torch.cat([is_key.unsqueeze(-1), degk_norm.unsqueeze(-1),
                        role], dim=-1)], dim=-1))
        src, dst = batch[EDGE_INDEX_KEY][0], batch[EDGE_INDEX_KEY][1]
        e22 = backbone._edge_input(batch, dtype)
        edge_ref = backbone.edge_readout(
            torch.cat([h0[src], h0[dst], e22], dim=-1))
        graph_index = batch[NODE_GRAPH_KEY].long()
        key_index = batch[NODE_KEY_SET_KEY].long()
        n_graphs = int(batch[GRAPH_CV_DELTA_KEY].shape[0])
        key_graph = graph_index[key_index]
        k_sum = torch.zeros(n_graphs, 48).index_add(
            0, key_graph, h0[key_index])
        k_count = torch.zeros(n_graphs).index_add(
            0, key_graph, torch.ones(int(key_index.shape[0])))
        g_sum = torch.zeros(n_graphs, 48).index_add(0, graph_index, h0)
        g_count = torch.zeros(n_graphs).index_add(
            0, graph_index, torch.ones_like(graph_index, dtype=dtype))
        z_ref = torch.tanh(backbone.z_head(torch.cat(
            [g_sum / g_count.clamp(min=1.0).unsqueeze(-1),
             k_sum / k_count.clamp(min=1.0).unsqueeze(-1)], dim=-1)))
    assert torch.allclose(out.edge_msg, edge_ref, atol=1e-6, rtol=0.0)
    assert torch.allclose(out.z, z_ref, atol=1e-6, rtol=0.0)
    # the attention/FFN/token branches are structurally IDENTITY at init:
    # perturbing every fresh-seeded non-mapped weight cannot move z or
    # edge_msg (the zero-init output projections gate them out exactly)
    with torch.no_grad():
        for layer in backbone.layers:
            layer.qkv.weight.add_(torch.randn_like(layer.qkv.weight))
            layer.ffn_in.weight.add_(torch.randn_like(layer.ffn_in.weight))
            layer.edge_bias.weight.add_(
                torch.randn_like(layer.edge_bias.weight))
            layer.token_bias.add_(0.1)
        backbone.token_init.weight.add_(
            torch.randn_like(backbone.token_init.weight))
    out_perturbed = backbone(batch)
    assert torch.allclose(out_perturbed.z, out.z, atol=1e-7, rtol=0.0)
    assert torch.allclose(out_perturbed.edge_msg, out.edge_msg,
                          atol=1e-7, rtol=0.0)


# ---------------------------------------------------------------------------
# construction protocol: RNG isolation, no seed reset, determinism
# ---------------------------------------------------------------------------
def test_build_rng_isolation_no_seed_reset_and_determinism(
        move_decoder, r3_state) -> None:
    import pes2ts_core.pesn.models.v542.candidates.a4 as a4_mod

    assert "torch.compile(" not in inspect.getsource(a4_mod)
    # decoders are RUNNER-owned: construct them BEFORE measuring the
    # stream (their Linear init legitimately consumes global RNG)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31337)
        decoders = [DirectMoveDecoder() for _ in range(3)]
    torch.manual_seed(4242)
    before_state = torch.random.get_rng_state().clone()
    before_seed = int(torch.initial_seed())
    bb1 = build(17, decoders[0])
    assert torch.equal(before_state, torch.random.get_rng_state()), \
        "build perturbed the caller's RNG stream"
    assert int(torch.initial_seed()) == before_seed, \
        "build must NOT reset the training seed (step 6 is the runner's)"
    # same run seed → bitwise-identical fresh modules
    bb2 = build(17, decoders[1])
    sd1, sd2 = bb1.state_dict(), bb2.state_dict()
    assert set(sd1) == set(sd2)
    assert all(torch.equal(sd1[k], sd2[k]) for k in sd1)
    # different run seed → different fresh attention stream
    bb3 = build(29, decoders[2])
    assert not torch.equal(bb1.layers[0].qkv.weight,
                           bb3.layers[0].qkv.weight)


def test_no_torch_compile_anywhere(backbone) -> None:
    assert_no_torch_compile(backbone)
    import pes2ts_core.pesn.models.v542.candidates.a4 as a4_mod

    assert "torch.compile(" not in inspect.getsource(a4_mod)
    assert "_orig_mod" not in inspect.getsource(a4_mod)


def test_constructor_refuses_width_drift() -> None:
    with pytest.raises(V542InitError) as exc:
        SparseGeometryTransformerA4(hidden=64)
    assert exc.value.code == "V542_A4_CONFIG"
