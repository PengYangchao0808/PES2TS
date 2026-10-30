"""V5.4.2 WP2 lane P2-4 — A5 / A5-C GraphGRU candidate certificate
(ADR-0033 §2.1/§2.3; config ``backbone_contract.candidates.A5``/``A5-C``).

Headline guarantees under test:

* EXACT preregistered active-parameter counts — A5 = 31,584 and
  A5-C = 53,712 (config ``backbone_active``; plan §5.2 table; joint
  33,811 / 55,939 with decoder 49 + process head 2,178) — plus the
  per-module breakdown that reproduces them, never padded/shrunk;
* the frozen backbone contract — ``forward(batch) -> BackboneOutput``
  with ``z [G,64]`` in ``graph_index`` graph order and ``edge_msg
  [E,48]`` row-for-row aligned with the input ``edge_index`` columns,
  fp32, live autograd graph, no torch.compile;
* graph-boundary isolation (perturbing every OTHER graph leaves one
  graph's ``z`` row and its edges bitwise identical), permutation
  invariance of ``z`` (1e-5) with edge-order-consistent ``edge_msg``,
  E(3) invariance (rotation/translation/reflection), determinism,
  forward-view whitelist rejection and legacy typed-invalid refusals;
* finite gradients on EVERY parameter with the GRUCell parameters
  proven to receive nonzero gradient (both cells for A5-C);
* the frozen init protocol — A5-C's first cell (and block) equals A5's
  bit-for-bit under the shared A5 init stream, the second cell is
  fresh, every ``W_g`` is exact-zero at init, ``build`` is RNG-neutral
  and never resets the training seed, and the strict R3 mapping is
  applied (9 mapped keys / 28 declared consumed of the 37-key payload);
* T=4 is really applied (per-cell forward-hook counters: A5 = 4 calls
  on the shared cell, A5-C = 2 calls per alternating cell) and
  ``edge_msg`` IS the final iteration's message tensor.

Fixture-level only (``make_v3_gate_batch``); no training, no real-data
access, no dev/confirm touch."""

from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.pesn.contracts.key_atom_v3 import (  # noqa: E402
    EDGE_ATTR_KEY,
    EDGE_B_SRC_KEY,
    EDGE_B_TGT_KEY,
    EDGE_DELTA_B_KEY,
    EDGE_INDEX_KEY,
    EDGE_IS_CHANGED_KEY,
    EDGE_ABS_DELTA_KEY,
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
)
from pes2ts_core.pesn.models.v542 import (  # noqa: E402
    N_DECODER_PARAMS_V542,
    N_PROCESS_PARAMS_V542,
    BackboneOutput,
    DirectMoveDecoder,
    V542InitError,
    assert_no_torch_compile,
)
from pes2ts_core.pesn.models.v542.candidates.a5 import (  # noqa: E402
    CANDIDATE_ID,
    CANDIDATE_ID_2CELL,
    ESTIMATED_ACTIVE_PARAMS,
    N_ITERATIONS,
    A5Backbone,
    a5_mapping_table,
    build,
    candidate_id_for_cells,
)
from pes2ts_core.pesn.models.v542.candidates.common import (  # noqa: E402
    V542CandidateBudgetError,
    assert_candidate_budget,
    check_backbone_contract,
    load_r3_backbone_state,
    resolve_init_source_id,
)
from scripts.agent_ops.feature_gate.v3_fixtures import make_v3_gate_batch  # noqa: E402

SEED = 17
#: Frozen per-candidate widths (config candidate_common_widths).
HIDDEN = 48
Z_DIM = 64

#: Frozen preregistered counts (config backbone_active + plan §5.2).
A5_ACTIVE = 31_584
A5C_ACTIVE = 53_712


@pytest.fixture(scope="module")
def r3_state():
    return load_r3_backbone_state()


@pytest.fixture(scope="module")
def batch():
    return forward_view_v3(make_v3_gate_batch(n_graphs=2, jitter_seed=SEED))


@pytest.fixture(scope="module")
def a5(r3_state):
    return build(SEED, DirectMoveDecoder(), r3_state=r3_state, cells=1)


@pytest.fixture(scope="module")
def a5c(r3_state):
    return build(SEED, DirectMoveDecoder(), r3_state=r3_state, cells=2)


def _clone_batch(batch) -> dict:
    return {k: (v.clone() if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()}


# ---------------------------------------------------------------------------
# frozen candidate identity + exact preregistered parameter counts
# ---------------------------------------------------------------------------
def test_candidate_ids_and_estimated_params_are_frozen() -> None:
    assert CANDIDATE_ID == "A5"
    assert CANDIDATE_ID_2CELL == "A5-C"
    assert ESTIMATED_ACTIVE_PARAMS == {"A5": A5_ACTIVE, "A5-C": A5C_ACTIVE}
    assert N_ITERATIONS == 4
    assert candidate_id_for_cells(1) == "A5"
    assert candidate_id_for_cells(2) == "A5-C"
    # A5-C attribution: init stream override A5-C → A5 (config
    # init_policy.candidate_fresh_init.init_source_id_overrides)
    assert resolve_init_source_id(candidate_id_for_cells(2)) == "A5"


def test_a5_exact_active_params_and_breakdown(a5, r3_state) -> None:
    assert isinstance(a5, A5Backbone)
    assert a5.candidate_id == "A5" and a5.n_cells == 1
    assert a5.active_param_count == A5_ACTIVE == 31_584
    assert assert_candidate_budget(a5, expected_active=A5_ACTIVE) == A5_ACTIVE
    # the frozen budget gate is EXACT — never padded or shrunk
    with pytest.raises(V542CandidateBudgetError) as exc:
        assert_candidate_budget(a5, expected_active=A5_ACTIVE + 1)
    assert exc.value.code == "V542_CANDIDATE_ACTIVE_MISMATCH"
    # per-module arithmetic that reproduces the estimate exactly
    per_module = {
        "elem_emb": sum(int(p.numel()) for p in a5.elem_emb.parameters()),
        "node_proj": sum(int(p.numel()) for p in a5.node_proj.parameters()),
        "z_head": sum(int(p.numel()) for p in a5.z_head.parameters()),
        "message_mlp": sum(
            int(p.numel()) for p in a5.message_mlps.parameters()),
        "gru_cell": sum(int(p.numel()) for p in a5.gru_cells.parameters()),
        "w_g": sum(int(p.numel()) for p in a5.w_gs.parameters()),
    }
    assert per_module == {
        "elem_emb": 1_904, "node_proj": 1_056, "z_head": 6_208,
        "message_mlp": 8_064,        # 118→48→48 (R3 message_mlp.0 shape)
        "gru_cell": 14_112,          # nn.GRUCell(48, 48) with both biases
        "w_g": 240,                  # Linear(5, 48, bias=False), zero-init
    }
    assert sum(per_module.values()) == A5_ACTIVE
    # joint arithmetic of the plan §5.2 table (w_v 49 + head 2,178)
    assert (a5.active_param_count + N_DECODER_PARAMS_V542
            + N_PROCESS_PARAMS_V542) == 33_811
    assert len(r3_state) == 37


def test_a5c_exact_active_params_and_breakdown(a5c) -> None:
    assert a5c.candidate_id == "A5-C" and a5c.n_cells == 2
    assert a5c.active_param_count == A5C_ACTIVE == 53_712
    assert assert_candidate_budget(
        a5c, expected_active=A5C_ACTIVE) == A5C_ACTIVE
    per_module = {
        "message_mlps": sum(
            int(p.numel()) for p in a5c.message_mlps.parameters()),
        "gru_cells": sum(int(p.numel()) for p in a5c.gru_cells.parameters()),
        "w_gs": sum(int(p.numel()) for p in a5c.w_gs.parameters()),
    }
    assert per_module == {
        "message_mlps": 16_128,      # 2 × 8,064 (per-cell shared-shape MLP)
        "gru_cells": 14_112 + 13_824,  # cell 0 full biases, cell 1 bias-free
        "w_gs": 480,                 # 2 × Linear(5, 48, bias=False)
    }
    assert sum(per_module.values()) == 44_544  # all recurrent-block params
    assert a5c.active_param_count - A5_ACTIVE == 22_128  # block-1 delta
    assert (a5c.active_param_count + N_DECODER_PARAMS_V542
            + N_PROCESS_PARAMS_V542) == 55_939


# ---------------------------------------------------------------------------
# unified interface: shapes, order, fp32, live graph (config interface)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cells", [1, 2])
def test_forward_output_shapes_order_and_contract(a5, a5c, batch, cells) -> None:
    backbone = a5 if cells == 1 else a5c
    n_graphs = int(batch[GRAPH_CV_DELTA_KEY].shape[0])
    n_edges = int(batch[EDGE_INDEX_KEY].shape[1])
    with torch.no_grad():
        out = backbone(batch)
    assert isinstance(out, BackboneOutput)
    assert set(BackboneOutput.__slots__) == {"z", "edge_msg"}
    assert tuple(out.z.shape) == (n_graphs, Z_DIM)
    assert tuple(out.edge_msg.shape) == (n_edges, HIDDEN)
    assert out.z.dtype == torch.float32
    assert out.edge_msg.dtype == torch.float32
    assert out.n_graphs == n_graphs and out.n_edges == n_edges
    # full frozen contract: shapes/isfinite/fp32 + LIVE autograd graph +
    # no torch.compile anywhere in the tree
    check_backbone_contract(backbone, batch)
    out2 = backbone(batch)
    assert out2.z.requires_grad and out2.z.grad_fn is not None
    assert out2.edge_msg.requires_grad and out2.edge_msg.grad_fn is not None


# ---------------------------------------------------------------------------
# graph-boundary isolation (H0: no cross-graph coupling)
# ---------------------------------------------------------------------------
def _graph_node_mask(batch, graph: int) -> torch.Tensor:
    return batch[NODE_GRAPH_KEY] == graph


def test_graph_boundary_isolation_per_graph(a5, batch) -> None:
    # for EVERY graph g: perturbing only g's node/edge features changes
    # ONLY g's z row and g's edge_msg rows — every other slot is
    # BITWISE identical (no cross-graph coupling anywhere in the T=4
    # loop, pooling or injection)
    n_graphs = int(batch[GRAPH_CV_DELTA_KEY].shape[0])
    dst = batch[EDGE_INDEX_KEY][1]
    with torch.no_grad():
        base = a5(batch)
        for g in range(n_graphs):
            perturbed = _clone_batch(batch)
            gmask = _graph_node_mask(perturbed, g)
            perturbed[NODE_DEGK_KEY] = torch.where(
                gmask, perturbed[NODE_DEGK_KEY] + 0.5,
                perturbed[NODE_DEGK_KEY])
            emask = gmask[dst]                     # edges received by g
            perturbed[EDGE_ATTR_KEY] = torch.where(
                emask.unsqueeze(-1),
                perturbed[EDGE_ATTR_KEY] * 1.7 + 0.03,
                perturbed[EDGE_ATTR_KEY])
            out = a5(perturbed)
            # g's own slots changed (the perturbation is active)
            assert not torch.equal(out.z[g], base.z[g]), g
            assert not torch.equal(out.edge_msg[emask],
                                   base.edge_msg[emask]), g
            # every OTHER graph's slots are bitwise identical
            g_others = torch.ones(n_graphs, dtype=torch.bool)
            g_others[g] = False
            assert torch.equal(out.z[g_others], base.z[g_others]), g
            assert torch.equal(out.edge_msg[~emask],
                               base.edge_msg[~emask]), g


def test_zero_w_g_masks_graph_conditioning_at_init(a5, batch) -> None:
    # W_g is exact-zero at init ⇒ the graph conditioning (cv_target_delta
    # / signed_intent) cannot reach the outputs until trained
    with torch.no_grad():
        base = a5(batch)
        perturbed = _clone_batch(batch)
        perturbed[GRAPH_CV_DELTA_KEY] = (
            perturbed[GRAPH_CV_DELTA_KEY] + 0.37)
        perturbed[GRAPH_SIGNED_INTENT_KEY] = torch.where(
            perturbed[GRAPH_SIGNED_INTENT_KEY] >= 0,
            torch.full_like(perturbed[GRAPH_SIGNED_INTENT_KEY], -1.0),
            torch.full_like(perturbed[GRAPH_SIGNED_INTENT_KEY], 1.0))
        out = a5(perturbed)
    assert torch.equal(out.z, base.z)
    assert torch.equal(out.edge_msg, base.edge_msg)


# ---------------------------------------------------------------------------
# permutation gates (H0: node permutation / edge permutation)
# ---------------------------------------------------------------------------
def test_node_permutation_invariance(a5, batch) -> None:
    n_nodes = int(batch[NODE_POS_KEY].shape[0])
    gen = torch.Generator().manual_seed(3)
    perm = torch.randperm(n_nodes, generator=gen)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(n_nodes)

    shuffled = _clone_batch(batch)
    shuffled[NODE_Z_KEY] = batch[NODE_Z_KEY][perm]
    shuffled[NODE_POS_KEY] = batch[NODE_POS_KEY][perm]
    shuffled[NODE_GRAPH_KEY] = batch[NODE_GRAPH_KEY][perm]
    shuffled[NODE_KEY_SET_KEY] = inv[batch[NODE_KEY_SET_KEY]]
    shuffled[NODE_IS_KEY_KEY] = batch[NODE_IS_KEY_KEY][perm]
    shuffled[NODE_DEGK_KEY] = batch[NODE_DEGK_KEY][perm]
    shuffled[NODE_INTENT_ROLE_KEY] = batch[NODE_INTENT_ROLE_KEY][perm]
    shuffled[EDGE_INDEX_KEY] = inv[batch[EDGE_INDEX_KEY]]

    with torch.no_grad():
        base = a5(batch)
        out = a5(shuffled)
    assert torch.allclose(out.z, base.z, atol=1e-5, rtol=1e-5)
    # edge COLUMN order is unchanged ⇒ edge_msg stays aligned rowwise
    assert torch.allclose(out.edge_msg, base.edge_msg,
                          atol=1e-5, rtol=1e-5)


def test_edge_permutation_edge_msg_alignment(a5, batch) -> None:
    n_edges = int(batch[EDGE_INDEX_KEY].shape[1])
    gen = torch.Generator().manual_seed(5)
    eperm = torch.randperm(n_edges, generator=gen)

    shuffled = _clone_batch(batch)
    shuffled[EDGE_INDEX_KEY] = batch[EDGE_INDEX_KEY][:, eperm]
    for key in (EDGE_ATTR_KEY, EDGE_B_SRC_KEY, EDGE_B_TGT_KEY,
                EDGE_DELTA_B_KEY, EDGE_ABS_DELTA_KEY, EDGE_IS_CHANGED_KEY):
        shuffled[key] = batch[key][eperm]

    with torch.no_grad():
        base = a5(batch)
        out = a5(shuffled)
    # edge_msg follows the shuffled column order row-for-row
    assert torch.allclose(out.edge_msg, base.edge_msg[eperm],
                          atol=1e-5, rtol=1e-5)
    assert torch.allclose(out.z, base.z, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# E(3) invariance (H0: translation / rotation / reflection)
# ---------------------------------------------------------------------------
def test_e3_invariance(a5, batch) -> None:
    gen = torch.Generator().manual_seed(11)
    mat = torch.randn(3, 3, generator=gen)
    q, r = torch.linalg.qr(mat)
    q = q * torch.sign(torch.diag(r)).unsqueeze(0)   # proper rotation
    trans = torch.tensor([1.5, -2.0, 0.7])
    reflect = torch.tensor([-1.0, 1.0, 1.0])

    pos = batch[NODE_POS_KEY]
    rotated = pos @ q.T + trans
    reflected = (pos @ q.T + trans) * reflect
    assert not torch.allclose(rotated, pos)          # the transform moved

    with torch.no_grad():
        base = a5(batch)
        for moved in (rotated, reflected):
            bad = _clone_batch(batch)
            bad[NODE_POS_KEY] = moved
            out = a5(bad)
            assert torch.allclose(out.z, base.z, atol=1e-6, rtol=1e-5)
            assert torch.allclose(out.edge_msg, base.edge_msg,
                                  atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# determinism (no dropout / stochastic layers; two calls bitwise equal)
# ---------------------------------------------------------------------------
def test_determinism_two_calls_bitwise_equal(a5, batch) -> None:
    assert not any(isinstance(m, torch.nn.Dropout) for m in a5.modules())
    assert not any(isinstance(m, torch.nn.BatchNorm1d) for m in a5.modules())
    with torch.no_grad():
        a = a5(batch)
        b = a5(batch)
    assert torch.equal(a.z, b.z)
    assert torch.equal(a.edge_msg, b.edge_msg)


# ---------------------------------------------------------------------------
# input legality: forward-view whitelist + legacy typed invalid
# ---------------------------------------------------------------------------
def test_forward_view_whitelist_rejection(a5, batch) -> None:
    bad = _clone_batch(batch)
    bad["y_teacher"] = torch.zeros(2, 3)          # supervision leak attempt
    with pytest.raises(KeyAtomV3ContractError) as exc:
        a5(bad)
    assert exc.value.code == "FORWARD_VIEW_LEAK"
    bad2 = _clone_batch(batch)
    bad2["energy_gradient"] = torch.zeros(1, 3)   # forbidden input class
    with pytest.raises(KeyAtomV3ContractError) as exc:
        a5(bad2)
    assert exc.value.code == "FORWARD_VIEW_LEAK"


@pytest.mark.parametrize("key,bad_value", [
    (EDGE_ATTR_KEY, float("nan")),
    (NODE_DEGK_KEY, float("inf")),
    (NODE_POS_KEY, float("nan")),
    (GRAPH_CV_DELTA_KEY, float("nan")),
    (NODE_INTENT_ROLE_KEY, float("inf")),
    (GRAPH_SIGNED_INTENT_KEY, float("nan")),
])
def test_typed_invalid_nonfinite(a5, batch, key, bad_value) -> None:
    bad = _clone_batch(batch)
    bad[key] = torch.full_like(batch[key], bad_value)
    with pytest.raises(KeyAtomV3InvalidInputError) as exc:
        a5(bad)
    assert exc.value.code == "NONFINITE_INPUT"
    assert key in exc.value.detail


def test_typed_invalid_z_range(a5, batch) -> None:
    bad = _clone_batch(batch)
    bad[NODE_Z_KEY] = batch[NODE_Z_KEY].clone()
    bad[NODE_Z_KEY][0] = 0                          # padding row
    with pytest.raises(KeyAtomV3InvalidInputError) as exc:
        a5(bad)
    assert exc.value.code == "Z_PADDING"
    bad[NODE_Z_KEY][0] = 119                        # unknown element
    with pytest.raises(KeyAtomV3InvalidInputError) as exc:
        a5(bad)
    assert exc.value.code == "Z_OUT_OF_RANGE"


# ---------------------------------------------------------------------------
# gradients: finite on every param; GRUCell params proven live
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fixture_name", ["a5", "a5c"])
def test_finite_gradients_and_gru_cell_grads(
        request, batch, fixture_name) -> None:
    backbone = request.getfixturevalue(fixture_name)
    backbone.zero_grad(set_to_none=True)
    out = backbone(batch)
    loss = out.z.sum() + out.edge_msg.sum()
    loss.backward()
    for name, param in backbone.named_parameters():
        assert param.grad is not None, name
        assert bool(torch.isfinite(param.grad).all()), name
        if name.startswith("gru_cells."):
            # the recurrent cell parameters MUST receive real gradient
            # (A5-C: BOTH alternating cells)
            assert float(param.grad.abs().sum()) > 0.0, name
    # the zero-initialised W_g is still trainable (grad reaches it)
    for name, param in backbone.named_parameters():
        if name.startswith("w_gs."):
            assert param.grad is not None, name
            assert bool(torch.isfinite(param.grad).all()), name


# ---------------------------------------------------------------------------
# A5-C attribution: first cell == A5 (A5 init stream), second is fresh
# ---------------------------------------------------------------------------
def test_a5c_first_block_equals_a5_second_cell_fresh(a5, a5c, r3_state) -> None:
    sd_a5, sd_a5c = a5.state_dict(), a5c.state_dict()
    # EVERY A5 parameter slot exists in A5-C and is bit-for-bit equal:
    # mapped slots (elem/node/z/message_mlp 0) plus the fresh block-0
    # cell/W_g drawn on the SHARED A5 init stream (init_source_id=A5)
    assert set(sd_a5) <= set(sd_a5c)
    for key, value in sd_a5.items():
        assert key in sd_a5c, key
        assert torch.equal(sd_a5c[key], value), key
    # the second cell is FRESH: its weights differ from cell 0 and were
    # never mapped from R3
    cell0, cell1 = a5c.gru_cells[0], a5c.gru_cells[1]
    for attr in ("weight_ih", "weight_hh"):
        assert not torch.equal(getattr(cell0, attr),
                               getattr(cell1, attr)), attr
    assert not torch.equal(a5c.message_mlps[0][0].weight,
                           a5c.message_mlps[1][0].weight)
    assert not torch.equal(a5c.message_mlps[1][0].weight,
                           r3_state["message_mlp.0.0.weight"])
    assert not torch.equal(a5c.message_mlps[1][0].weight,
                           r3_state["message_mlp.1.0.weight"])


def test_w_g_zero_at_init_for_both_cells(a5, a5c) -> None:
    assert len(a5.w_gs) == 1 and len(a5c.w_gs) == 2
    for backbone in (a5, a5c):
        for slot in backbone.w_gs:
            assert slot.bias is None              # bias-free Linear(5,48)
            assert torch.equal(slot.weight, torch.zeros(HIDDEN, 5))
            assert slot.weight.dtype == torch.float32


# ---------------------------------------------------------------------------
# T=4 iteration count really applied + edge_msg is the FINAL message
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fixture_name,expected_counts", [
    ("a5", [4]),
    ("a5c", [2, 2]),
])
def test_t4_iterations_applied_and_final_edge_msg(
        request, batch, fixture_name, expected_counts) -> None:
    backbone = request.getfixturevalue(fixture_name)
    assert backbone.n_iterations == N_ITERATIONS == 4
    cell_calls = [0] * backbone.n_cells
    mlp_calls = [0] * backbone.n_cells
    mlp_outputs: list[torch.Tensor] = []
    for idx, cell in enumerate(backbone.gru_cells):
        cell.register_forward_hook(
            lambda m, i, o, k=idx: cell_calls.__setitem__(
                k, cell_calls[k] + 1))
    for idx, mlp in enumerate(backbone.message_mlps):
        def _record(_m, _i, output, k=idx):
            mlp_calls[k] += 1
            mlp_outputs.append(output)
        mlp.register_forward_hook(_record)
    with torch.no_grad():
        out = backbone(batch)
    # T=4 is actually executed: the shared cell fires 4× (A5) or the two
    # alternating cells fire 2× each (A5-C)
    assert cell_calls == expected_counts
    assert sum(mlp_calls) == 4
    # edge_msg IS the final iteration's per-edge message tensor
    assert len(mlp_outputs) == 4
    assert torch.equal(out.edge_msg, mlp_outputs[-1])
    assert int(out.edge_msg.shape[0]) == int(batch[EDGE_INDEX_KEY].shape[1])


# ---------------------------------------------------------------------------
# frozen construction protocol (build: RNG isolation + mapping + zero)
# ---------------------------------------------------------------------------
def test_build_is_rng_neutral_and_never_resets_training_seed(
        r3_state) -> None:
    decoder = DirectMoveDecoder()      # built BEFORE the RNG snapshot
    torch.manual_seed(4242)
    before = torch.random.get_rng_state().clone()
    build(29, decoder, r3_state=r3_state)
    assert torch.equal(before, torch.random.get_rng_state()), \
        "build perturbed the caller's RNG stream"
    # the training seed is NEVER reset inside build (step 6 is the
    # caller's job): torch.initial_seed() stays exactly where it was
    torch.manual_seed(999)
    build(17, decoder, r3_state=r3_state)
    assert int(torch.initial_seed()) == 999


def test_build_is_deterministic_per_run_seed(r3_state) -> None:
    b1 = build(SEED, DirectMoveDecoder(), r3_state=r3_state)
    b2 = build(SEED, DirectMoveDecoder(), r3_state=r3_state)
    sd1, sd2 = b1.state_dict(), b2.state_dict()
    assert set(sd1) == set(sd2)
    assert all(torch.equal(sd1[k], sd2[k]) for k in sd1)
    b3 = build(SEED + 12, DirectMoveDecoder(), r3_state=r3_state)
    # the FRESH cell really is seeded by run_seed (mapped slots stay
    # identical because R3 overwrites them)
    assert not torch.equal(b1.gru_cells[0].weight_ih,
                           b3.gru_cells[0].weight_ih)


def test_build_applies_frozen_mapping_and_declares_consumed(a5, a5c,
                                                            r3_state) -> None:
    mapping, consumed = a5_mapping_table(r3_state)
    # 9 mapped keys (A1 common set identity + shared msg MLP from R3
    # message_mlp.0); 28 declared consumed; 9 + 28 = the frozen 37
    assert len(mapping) == 9 and len(consumed) == 28
    assert len(set(mapping.values())) == 9
    assert set(mapping.values()) | set(consumed) == set(r3_state)
    sd = a5.state_dict()
    for target, source in mapping.items():
        assert torch.equal(sd[target], r3_state[source]), target
    # A5-C maps the SAME set (only block 0 comes from R3)
    mapping_c, consumed_c = a5_mapping_table(r3_state)
    assert mapping_c == mapping and consumed_c == consumed
    sd_c = a5c.state_dict()
    for target, source in mapping.items():
        assert torch.equal(sd_c[target], r3_state[source]), target
    # unmapped fresh slots were NOT silently mapped from R3
    assert not torch.equal(sd_c["message_mlps.1.0.weight"],
                           r3_state["message_mlp.0.0.weight"])
    assert not torch.equal(sd_c["message_mlps.1.0.weight"],
                           r3_state["message_mlp.1.0.weight"])


def test_build_loads_r3_w_v_into_fresh_decoder(r3_state) -> None:
    # gate-2 B1: A5/A5-C (the former blockers) load the SHA-verified R3
    # w_v into the SHARED decoder on every build — the backbone itself
    # still owns no w_v slot (the decoder is the single authority)
    for cells in (1, 2):
        decoder = DirectMoveDecoder()
        random_before = decoder.w_v.weight.detach().clone()
        build(SEED, decoder, r3_state=r3_state, cells=cells)
        assert not torch.equal(decoder.w_v.weight, random_before), cells
        assert torch.equal(decoder.w_v.weight,
                           r3_state["w_v.weight"]), cells
        assert torch.equal(decoder.w_v.bias, r3_state["w_v.bias"]), cells
        built = build(SEED, DirectMoveDecoder(), r3_state=r3_state,
                      cells=cells)
        assert not any(name.split(".")[0] == "w_v"
                       for name, _ in built.named_parameters()), cells
    # gate-2 B2: the frozen a_ref seam holds on A5's own build too
    tampered = DirectMoveDecoder()
    tampered._a_ref = 0.1
    with pytest.raises(V542InitError) as exc:
        build(SEED, tampered, r3_state=r3_state)
    assert exc.value.code == "V542_DECODER_A_REF"


def test_build_typed_refusals(a5, r3_state) -> None:
    # cells outside the frozen {1, 2} (incl. bool / str)
    for bad_cells in (3, 0, -1, True, "2", 2.0):  # type: ignore[assignment]
        with pytest.raises(V542InitError) as exc:
            build(SEED, DirectMoveDecoder(), r3_state=r3_state,
                  cells=bad_cells)                # type: ignore[arg-type]
        assert exc.value.code == "V542_A5_CELLS"
    # the assembly seam requires THE shared decoder
    with pytest.raises(V542InitError) as exc:
        A5Backbone(torch.nn.Linear(1, 1))         # type: ignore[arg-type]
    assert exc.value.code == "V542_A5_DECODER"
    # gate-1 B1: frozen a_ref=0.08 on the adopted decoder
    class _WrongARefDecoder(DirectMoveDecoder):
        @property
        def a_ref(self) -> float:
            return 0.1

    with pytest.raises(V542InitError) as exc:
        A5Backbone(_WrongARefDecoder())
    assert exc.value.code == "V542_DECODER_A_REF"
    # the backbone never owns w_v: the decoder stays a separate module
    assert not any("w_v" in name for name, _ in a5.named_parameters())


# ---------------------------------------------------------------------------
# no torch.compile (ADR-0033 §2.1)
# ---------------------------------------------------------------------------
def test_a5_never_calls_torch_compile(a5, a5c) -> None:
    from pes2ts_core.pesn.models.v542.candidates import a5 as a5_mod

    assert "torch.compile(" not in inspect.getsource(a5_mod)
    assert_no_torch_compile(a5)
    assert_no_torch_compile(a5c)
    for name, child in a5.named_modules():
        assert not hasattr(child, "_orig_mod"), name
