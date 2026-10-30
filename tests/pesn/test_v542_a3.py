"""V5.4.2 WP2 lane P2 candidate — A3 certificate
(ADR-0033 §2.3; config ``backbone_contract.backbone_registry`` A3 row;
Oracle architecture §2.3).

Headline guarantees under test:

* EXACT active parameter count — A3 = 53,328 = A1 41,232 + query
  attention 12,096, with the full per-module breakdown;
* **A3 == A1 at initialisation** — built from the same R3 state and
  seeds, the two backbones produce BITWISE-equal ``z`` and ``edge_msg``
  (``W_o`` zero-init makes the attention branch exactly zero; observed
  tolerance: bitwise, i.e. max-abs-diff 0.0 — the stronger of the
  allowed ≤1e-6);
* the attention is structurally real — zeroing ``W_o`` is the only thing
  separating A3 from A1 at init (a non-zero ``W_o`` changes
  ``edge_msg``);
* the same interface / input-legality / E(3) / determinism / gradient /
  no-compile / construction-protocol gates as A1 (see
  ``tests/pesn/test_v542_a1.py``), including empty-in-edge safety.

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
    assert_no_torch_compile,
    validate_backbone_output,
)
from pes2ts_core.pesn.models.v542.candidates import (  # noqa: E402
    V542CandidateBudgetError,
    assert_candidate_budget,
    check_backbone_contract,
    load_r3_backbone_state,
)
from pes2ts_core.pesn.models.v542.candidates import a1, a3  # noqa: E402
from scripts.agent_ops.feature_gate.v3_fixtures import (  # noqa: E402
    make_v3_gate_batch,
)

SEED = 17


# ---------------------------------------------------------------------------
# fixtures
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
    return a3.build(SEED, decoder, r3_state=r3_state)


@pytest.fixture(scope="module")
def a1_baseline(decoder, r3_state):
    return a1.build(SEED, decoder, r3_state=r3_state)


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
# module surface + EXACT parameter accounting
# ---------------------------------------------------------------------------
def test_module_surface_constants() -> None:
    assert a3.CANDIDATE_ID == "A3"
    assert a3.ESTIMATED_ACTIVE_PARAMS == 53_328
    assert a3.N_HEADS == 2 and a3.HEAD_DIM == 24
    assert a3.QUERY_IN_DIM == 53                    # 48 ⊕ cv 4 ⊕ sign 1


def test_a3_exact_active_params_and_breakdown(backbone, a1_baseline) -> None:
    assert backbone.active_param_count == a3.ESTIMATED_ACTIVE_PARAMS == 53_328
    assert assert_candidate_budget(backbone, expected_active=53_328) == 53_328
    total = sum(int(p.numel()) for p in backbone.parameters())
    assert total == 53_328 + N_DECODER_PARAMS_V542
    # the delta over A1 is EXACTLY the frozen query-attention block
    assert (backbone.active_param_count
            - a1_baseline.active_param_count) == 12_096
    assert _component_params(backbone, "query_g") == 2_592     # 53→48
    assert _component_params(backbone, "w_q") == 2_352         # 48→48
    assert _component_params(backbone, "w_k") == 2_352
    assert _component_params(backbone, "w_value") == 2_352
    assert _component_params(backbone, "w_o") == 2_352         # zero-init
    assert _component_params(backbone, "attn_norm") == 96      # LN(48)
    assert 2_592 + 3 * 2_352 + 2_352 + 96 == 12_096
    # A1 core intact: same 41,232 underneath
    assert _component_params(backbone, "message_mlp.0") == 8_064
    assert _component_params(backbone, "w_d.0") == 624


def test_budget_mismatch_is_a_typed_refusal(backbone) -> None:
    with pytest.raises(V542CandidateBudgetError) as exc:
        assert_candidate_budget(backbone, expected_active=53_327)
    assert exc.value.code == "V542_CANDIDATE_ACTIVE_MISMATCH"


# ---------------------------------------------------------------------------
# the headline: A3 == A1 at initialisation (W_o zero-init, ADR-0033 §2.3)
# ---------------------------------------------------------------------------
def test_a3_equals_a1_at_init_same_r3_state_and_seed(
        backbone, a1_baseline, batch) -> None:
    """Observed tolerance: BITWISE (max-abs-diff 0.0) — well within the
    allowed ≤1e-6 fp32.  W_o = 0 kills the attention branch exactly, and
    every shared slot is R3-mapped / constant / zero, so the forwards
    coincide op-for-op."""
    out_a3 = backbone(batch)
    out_a1 = a1_baseline(batch)
    assert torch.equal(out_a3.z, out_a1.z)
    assert torch.equal(out_a3.edge_msg, out_a1.edge_msg)
    assert float((out_a3.z - out_a1.z).detach().abs().max()) == 0.0
    assert float((out_a3.edge_msg - out_a1.edge_msg).detach().abs().max()) == 0.0


def test_a3_equals_a1_at_init_even_under_different_run_seeds(
        decoder, r3_state, batch) -> None:
    # structural claim: the equality comes from the frozen init recipe
    # (R3 mapping + zero W_d/W_o + constant LayerNorm), NOT from a
    # coincidental seed match
    b_a1 = a1.build(SEED, decoder, r3_state=r3_state)
    b_a3 = a3.build(43, decoder, r3_state=r3_state)
    out_a1 = b_a1(batch)
    out_a3 = b_a3(batch)
    assert torch.equal(out_a3.z, out_a1.z)
    assert torch.equal(out_a3.edge_msg, out_a1.edge_msg)


def test_w_o_is_zero_initialised_and_attention_activates_when_wired(
        backbone, decoder, r3_state, batch) -> None:
    assert torch.equal(backbone.w_o.weight,
                       torch.zeros_like(backbone.w_o.weight))
    assert torch.equal(backbone.w_o.bias,
                       torch.zeros_like(backbone.w_o.bias))
    # every W_d is zero too (A3 ⊇ A1 distance injection)
    for slot in backbone.w_d:
        assert torch.equal(slot.weight, torch.zeros_like(slot.weight))
        assert torch.equal(slot.bias, torch.zeros_like(slot.bias))
    # a non-zero W_o makes the attention branch act on edge_msg
    fresh = a3.build(SEED, decoder, r3_state=r3_state)
    out_a1 = a1.build(SEED, decoder, r3_state=r3_state)(batch)
    with torch.no_grad():
        fresh.w_o.weight.add_(0.05)
    out_active = fresh(batch)
    assert not torch.equal(out_active.edge_msg, out_a1.edge_msg)
    assert not torch.equal(out_active.z, out_a1.z)


def test_segment_softmax_rows_matches_dense_reference() -> None:
    logits = torch.tensor([[1.0, 0.5], [2.0, -1.0], [0.0, 3.0]])
    index = torch.tensor([0, 0, 2])
    out = a3.segment_softmax_rows(logits, index, 4)
    ref0 = torch.softmax(logits[:2], dim=0)
    assert torch.allclose(out[:2], ref0, atol=1e-6)
    assert torch.allclose(out[2], torch.softmax(logits[2:3], dim=0))
    # each attended SEGMENT (receiver) sums to 1 per head
    assert torch.allclose(out[:2].sum(dim=0), torch.ones(2), atol=1e-6)
    assert torch.allclose(out[2:].sum(dim=0), torch.ones(2), atol=1e-6)
    # empty edges pass through
    empty = torch.zeros(0, 2)
    assert tuple(a3.segment_softmax_rows(empty, torch.zeros(
        0, dtype=torch.long), 3).shape) == (0, 2)


# ---------------------------------------------------------------------------
# unified interface + input legality (same contract as A1)
# ---------------------------------------------------------------------------
def test_backbone_output_shapes_order_and_contract(backbone, batch) -> None:
    out = backbone(batch)
    assert isinstance(out, BackboneOutput)
    n_graphs = int(batch["cv_target_delta"].shape[0])
    n_edges = int(batch["edge_index"].shape[1])
    assert tuple(out.z.shape) == (n_graphs, 64)
    assert tuple(out.edge_msg.shape) == (n_edges, 48)
    assert out.z.dtype == torch.float32 and out.edge_msg.dtype == torch.float32
    validate_backbone_output(out, n_graphs=n_graphs, n_edges=n_edges)
    assert check_backbone_contract(backbone, batch) is None


def test_forward_view_whitelist_rejects_leaked_supervision(
        backbone, batch) -> None:
    leaked = dict(batch)
    leaked["y_teacher"] = torch.zeros(3, 3)
    with pytest.raises(KeyAtomV3ContractError) as exc:
        backbone(leaked)
    assert exc.value.code == "FORWARD_VIEW_LEAK"


def test_typed_invalid_non_finite(backbone, batch) -> None:
    bad = dict(batch)
    bad["signed_intent"] = batch["signed_intent"].clone()
    bad["signed_intent"][0] = float("inf")
    with pytest.raises(KeyAtomV3InvalidInputError) as exc:
        backbone(bad)
    assert exc.value.code == "NONFINITE_INPUT"


def test_empty_in_edges_are_safe(backbone, batch) -> None:
    n_edges = batch["edge_index"].shape[1]
    empty = {k: (v[:0] if torch.is_tensor(v)
                 and v.shape[:1] == (n_edges,) else v)
             for k, v in batch.items()}
    empty["edge_index"] = batch["edge_index"][:, :0]
    out = backbone(empty)
    assert tuple(out.edge_msg.shape) == (0, 48)
    assert bool(torch.isfinite(out.z).all())
    assert bool(torch.isfinite(out.edge_msg).all())


# ---------------------------------------------------------------------------
# E(3) / permutation / relabel invariance (≤1e-5)
# ---------------------------------------------------------------------------
def test_e3_rotation_translation_invariance_and_move_equivariance(
        backbone, decoder, batch) -> None:
    out = backbone(batch)
    rot = _rotation(42)
    shift = torch.tensor([1.5, -2.0, 0.7])
    moved = dict(batch)
    moved["pos"] = batch["pos"] @ rot.T + shift
    out2 = backbone(moved)
    assert torch.allclose(out.z, out2.z, atol=1e-5, rtol=0.0)
    assert torch.allclose(out.edge_msg, out2.edge_msg, atol=1e-5, rtol=0.0)
    u1 = decoder(out.edge_msg, batch["pos"], batch["edge_index"],
                 batch["key_index"], batch["degK"])
    u2 = decoder(out2.edge_msg, moved["pos"], moved["edge_index"],
                 moved["key_index"], moved["degK"])
    assert torch.allclose(u2, u1 @ rot.T, atol=1e-5, rtol=0.0)


def test_edge_and_node_permutation_invariance_of_z(backbone, batch) -> None:
    out = backbone(batch)
    assert torch.allclose(backbone(_shuffle_edges(batch)).z, out.z,
                          atol=1e-5, rtol=0.0)
    assert torch.allclose(backbone(_permute_nodes(batch)).z, out.z,
                          atol=1e-5, rtol=0.0)


def test_relabel_metadata_invariance_of_z(backbone, batch) -> None:
    out = backbone(batch)
    relabelled = dict(batch)
    relabelled["endpoint_names"] = ("X", "Y")
    assert torch.equal(backbone(relabelled).z, out.z)


# ---------------------------------------------------------------------------
# determinism + gradients (attention params included)
# ---------------------------------------------------------------------------
def test_deterministic_double_forward_is_bitwise(backbone, batch) -> None:
    out1 = backbone(batch)
    out2 = backbone(batch)
    assert torch.equal(out1.z, out2.z)
    assert torch.equal(out1.edge_msg, out2.edge_msg)


def test_finite_gradients_including_attention_params(backbone, batch) -> None:
    out = backbone(batch)
    assert out.edge_msg.requires_grad and out.edge_msg.grad_fn is not None
    (out.z.sum() + out.edge_msg.sum()).backward()
    attention = ("query_g", "w_q", "w_k", "w_value", "w_o", "attn_norm")
    seen_attention = False
    for name, p in backbone.named_parameters():
        if name.startswith("w_v."):
            continue        # decoder-owned; consumed via DirectMoveDecoder
        assert p.grad is not None, name
        assert bool(torch.isfinite(p.grad).all()), name
        if name.startswith(attention):
            seen_attention = True
    assert seen_attention


# ---------------------------------------------------------------------------
# frozen construction protocol
# ---------------------------------------------------------------------------
def test_factory_determinism_and_seed_isolation(decoder, r3_state) -> None:
    torch.manual_seed(4242)
    before = torch.random.get_rng_state().clone()
    seed_before = int(torch.initial_seed())
    b_a = a3.build(SEED, decoder, r3_state=r3_state)
    assert torch.equal(before, torch.random.get_rng_state())
    assert int(torch.initial_seed()) == seed_before, \
        "the factory must NOT reset the training seed"
    b_b = a3.build(SEED, decoder, r3_state=r3_state)
    s_a, s_b = b_a.state_dict(), b_b.state_dict()
    assert set(s_a) == set(s_b)
    assert all(torch.equal(s_a[k], s_b[k]) for k in s_a)
    # the A3 mapping set is A1's: query attention is 100% fresh seeded
    mapped = a1.r3_mapping_table(r3_state)
    assert not any(k.startswith(attention)
                   for k in mapped
                   for attention in ("query_g", "w_q", "w_k",
                                     "w_value", "w_o", "attn_norm"))


def test_factory_shares_decoder_w_v(decoder, r3_state) -> None:
    built = a3.build(29, decoder, r3_state=r3_state)
    assert built.w_v is decoder.w_v


# ---------------------------------------------------------------------------
# ADR-0033 §2.1: torch.compile is forbidden
# ---------------------------------------------------------------------------
def test_no_torch_compile_anywhere(backbone) -> None:
    assert "torch.compile(" not in inspect.getsource(a3)
    assert_no_torch_compile(backbone)
