"""V5.4.2 WP2 lane P2 — candidate A2 single-frame PaiNN-lite v32 certificate
(ADR-0033 §2.3/§4; config ``backbone_registry.A2``,
``init_mapping.r3_semantic_mapping.A2``,
``per_candidate_zero_init.A2``).

Headline guarantees under test:

* EXACT active parameters 54,896 — measured count plus the full
  per-component breakdown (config estimate, ``estimated: true`` until
  this assertion pins it);
* :class:`BackboneOutput` shapes ``z [G,64]`` / ``edge_msg [E,48]``,
  fp32, finite, LIVE autograd graph, ``edge_msg`` rows in input
  ``edge_index`` column order;
* the frozen zero-init row: ``V_0`` buffer all-zero, ``W_g`` zero,
  ``phi_u``/``phi_g`` final output projections zero, while ``W_s``/``W_v``
  keep their default seeded init;
* the frozen A2 R3 mapping — A1 common set (``elem_emb``/``node_proj``/
  ``z_head``) + two same-shape ``phi_s`` ← ``message_mlp.0/1`` (13
  targets, 24 declared-consumed payload keys, strict audit clean), with
  the build running inside ``construction_scope`` WITHOUT touching the
  caller's training seed;
* forward-view whitelist typed rejection (``FORWARD_VIEW_LEAK``) and the
  legacy typed-invalid taxonomy (``NONFINITE_INPUT`` / ``Z_PADDING`` /
  ``Z_OUT_OF_RANGE``);
* E(3): ``z``/``edge_msg`` invariant within 1e-5 under translation,
  rotation AND reflection, and the shared decoder displacement
  equivariant — checked both at the frozen cold start and with the
  zero-init ``phi_u`` gate opened, so the vector channel is exercised;
* node permutation / edge permutation / edge-order reversal / endpoint
  relabel invariance;
* determinism: repeat forward bitwise-equal, repeat build bitwise-equal,
  and ``build`` leaves the caller's global RNG stream byte-identical
  (no training-seed reset inside);
* finite gradients on EVERY parameter including the vector-channel
  params (the cold-start V→s gate is closed by the frozen zero-init, so
  those grads are present, finite and exactly zero; once ``phi_u`` is
  opened the vector channel demonstrably receives NONZERO gradient);
* ``edge_msg`` DEPENDS on the vector channel — the intentional A2
  semantics of ADR-0033 §4 (recompute from ``s^L``) guarded against a
  silent "fix" back to a Star-style capture;
* no ``torch.compile`` anywhere (source-level + structural check).

Fixture-level only (``make_v3_gate_batch``); no training, no real-data
access, no dev/confirm touch."""

from __future__ import annotations

import ast
import copy
import dataclasses
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
    Backbone,
    BackboneOutput,
    DirectMoveDecoder,
    V542InitError,
    V542InterfaceError,
    assert_no_torch_compile,
    validate_backbone_output,
)
from pes2ts_core.pesn.models.v542.candidates import (  # noqa: E402
    apply_r3_mapping,
    assert_candidate_budget,
    check_backbone_contract,
    load_r3_backbone_state,
)
from pes2ts_core.pesn.models.v542.candidates import a2 as a2_mod  # noqa: E402
from pes2ts_core.pesn.models.v542.candidates.a2 import (  # noqa: E402
    CANDIDATE_ID,
    ESTIMATED_ACTIVE_PARAMS,
    VECTOR_CHANNELS,
    A2Backbone,
    A2_R3_MAPPING,
    build,
)
from scripts.agent_ops.feature_gate.v3_fixtures import make_v3_gate_batch  # noqa: E402

SEED = 17
#: E(3)/permutation tolerance mandated by the A2 contract.
TOL = 1e-5

#: The frozen per-component breakdown of the 54,896 estimate
#: (Oracle §2.2: common 9,168 + W_g 288 + 2×22,720).
EXPECTED_BREAKDOWN = {
    "elem_emb": 1_904,
    "node_proj": 1_056,
    "z_head": 6_208,
    "W_g": 288,
    "phi_s": 16_128,   # 2 × 8,064
    "W_s": 3_136,      # 2 × 1,568
    "W_v": 3_136,      # 2 × 1,568
    "phi_u": 12_480,   # 2 × 6,240
    "phi_g": 10_368,   # 2 × 5,184
    "norm": 192,       # 2 × 96
}


# ---------------------------------------------------------------------------
# fixtures + helpers
# ---------------------------------------------------------------------------
@pytest.fixture()
def r3_state():
    return load_r3_backbone_state()


@pytest.fixture()
def batch():
    return forward_view_v3(make_v3_gate_batch(n_graphs=2, jitter_seed=SEED))


@pytest.fixture()
def backbone(r3_state):
    return build(SEED, DirectMoveDecoder(), r3_state=r3_state)


def _open_phi_u(model: A2Backbone, *, scale: float = 0.02,
                seed: int = 20_261_030) -> None:
    """Open the frozen zero-init ``phi_u`` output gate (as training would).

    The cold start closes the V→s pathway on purpose (config
    ``per_candidate_zero_init.A2``); tests that must OBSERVE the vector
    channel's influence (E(3) with V live, gradient reachability,
    ``edge_msg`` dependency) open it deterministically first.
    """
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for seq in model.phi_u:
            final = seq[-1]
            final.weight.copy_(
                torch.randn(final.weight.shape, generator=gen) * scale)
            final.bias.zero_()


def _frame(seed: int, *, det: int = 1) -> torch.Tensor:
    """Deterministic 3×3 orthonormal frame with the requested determinant."""
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(3, 3, dtype=torch.float64, generator=gen)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diag(r)).reshape(1, 3)
    if det * float(torch.linalg.det(q)) < 0.0:
        q[:, -1] = -q[:, -1]
    return q.to(torch.float32)


def _shifted(batch, delta) -> dict:
    out = dict(batch)
    out["pos"] = batch["pos"] + delta
    return out


def _rotated(batch, rotation: torch.Tensor) -> dict:
    out = dict(batch)
    out["pos"] = batch["pos"] @ rotation.T
    return out


def _shuffle_edges(batch, order: torch.Tensor) -> dict:
    out = dict(batch)
    out["edge_index"] = batch["edge_index"][:, order]
    for key in ("edge_attr", "b_src", "b_tgt", "delta_b_intent",
                "abs_delta", "is_changed"):
        out[key] = batch[key][order]
    return out


def _decoder_u(model: A2Backbone, batch, decoder: DirectMoveDecoder):
    with torch.no_grad():
        out = model(batch)
        return decoder(out.edge_msg, batch["pos"], batch["edge_index"],
                       batch["key_index"], batch["degK"])


def _max_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


# ---------------------------------------------------------------------------
# frozen constants + module discipline
# ---------------------------------------------------------------------------
def test_candidate_constants_are_frozen() -> None:
    assert CANDIDATE_ID == "A2"
    assert ESTIMATED_ACTIVE_PARAMS == 54_896
    assert VECTOR_CHANNELS == 32
    assert a2_mod.N_LAYERS == 2
    assert a2_mod.HIDDEN == 48
    assert a2_mod.A2_BACKBONE_VERSION == (
        "pesn_key_atom_v542_a2_painn_lite_v32")
    # the frozen mapping is the A1 common set + two same-shape phi_s
    assert len(A2_R3_MAPPING) == 13
    assert set(A2_R3_MAPPING.values()) == {
        "elem_emb.weight",
        "node_proj.weight", "node_proj.bias",
        "z_head.weight", "z_head.bias",
        "message_mlp.0.0.weight", "message_mlp.0.0.bias",
        "message_mlp.0.2.weight", "message_mlp.0.2.bias",
        "message_mlp.1.0.weight", "message_mlp.1.0.bias",
        "message_mlp.1.2.weight", "message_mlp.1.2.bias",
    }
    assert [t for t in A2_R3_MAPPING if t.startswith("phi_s.")] == [
        "phi_s.0.0.weight", "phi_s.0.0.bias",
        "phi_s.0.2.weight", "phi_s.0.2.bias",
        "phi_s.1.0.weight", "phi_s.1.0.bias",
        "phi_s.1.2.weight", "phi_s.1.2.bias",
    ]


def test_module_discipline_no_compile_no_dual_and_a2_semantics_doc() -> None:
    source = inspect.getsource(a2_mod)
    # ADR-0033 §2.1: torch.compile is forbidden for the whole model layer
    assert "torch.compile(" not in source
    # Oracle §2.2: A2 must not reuse dual-* code
    assert "from pes2ts_core.pesn.models.dual" not in source
    tree = ast.parse(source)
    from_modules = [n.module or "" for n in ast.walk(tree)
                    if isinstance(n, ast.ImportFrom)]
    assert not any(m.startswith("pes2ts_core.pesn.models.dual")
                   for m in from_modules), from_modules
    # ADR-0033 §4: the edge_msg recompute difference is documented as an
    # INTENTIONAL semantic difference, not a bug, in the module docstring
    doc = inspect.getdoc(a2_mod) or ""
    lowered = doc.lower()
    assert "edge_msg" in doc
    assert "intentional" in lowered
    assert "not a bug" in lowered
    assert "adr-0033 §4" in lowered
    assert "fixed" in lowered
    assert "recomput" in lowered


# ---------------------------------------------------------------------------
# exact parameter budget (config backbone_registry.A2: 54,896)
# ---------------------------------------------------------------------------
def test_exact_active_params_and_component_breakdown(backbone) -> None:
    measured = backbone.active_param_count
    assert measured == ESTIMATED_ACTIVE_PARAMS == 54_896
    assert isinstance(measured, int)
    # naive total equals active: A2 owns no decoder-shared w_v
    total = sum(int(p.numel()) for p in backbone.parameters())
    assert total == 54_896
    # full per-component breakdown — the report to give if this ever drifts
    breakdown = backbone.param_breakdown()
    assert breakdown == EXPECTED_BREAKDOWN
    assert sum(breakdown.values()) == 54_896
    # the construction-time gate accepts exactly this count
    assert assert_candidate_budget(
        backbone, expected_active=ESTIMATED_ACTIVE_PARAMS) == 54_896
    # and a mismatched expectation is a typed refusal
    from pes2ts_core.pesn.models.v542.candidates import (
        V542CandidateBudgetError,
    )
    with pytest.raises(V542CandidateBudgetError) as exc:
        assert_candidate_budget(backbone, expected_active=54_895)
    assert exc.value.code == "V542_CANDIDATE_ACTIVE_MISMATCH"
    # protocol conformance (forward + exact active_param_count)
    assert isinstance(backbone, Backbone)


# ---------------------------------------------------------------------------
# BackboneOutput surface: shapes, order, fp32, live graph
# ---------------------------------------------------------------------------
def test_backbone_output_shapes_order_and_live_graph(backbone, batch) -> None:
    out = backbone(batch)
    assert isinstance(out, BackboneOutput)
    assert {f.name for f in dataclasses.fields(BackboneOutput)} == {
        "z", "edge_msg"}
    n_graphs = int(batch["cv_target_delta"].shape[0])
    n_edges = int(batch["edge_index"].shape[1])
    assert tuple(out.z.shape) == (n_graphs, 64)
    assert tuple(out.edge_msg.shape) == (n_edges, 48)
    assert out.z.dtype == torch.float32
    assert out.edge_msg.dtype == torch.float32
    # live autograd graph on both fields (training backprops MOVE)
    for tensor in (out.z, out.edge_msg):
        assert tensor.requires_grad
        assert tensor.grad_fn is not None
    assert out.n_graphs == n_graphs and out.n_edges == n_edges
    # the frozen wrapper helper accepts it (shapes/fp32/finiteness)
    validate_backbone_output(out, n_graphs=n_graphs, n_edges=n_edges)
    # the full candidate contract: no-compile + shapes + live graph
    assert check_backbone_contract(backbone, batch) is None
    # forward-view keys only: a fresh build reads no supervision key
    source = inspect.getsource(A2Backbone.forward)
    for forbidden in ("y_teacher", "frame_weight", "pair_index",
                      "endpoint_geometry", "true_progress"):
        assert f'["{forbidden}"]' not in source


# ---------------------------------------------------------------------------
# frozen zero-init row (config per_candidate_zero_init.A2)
# ---------------------------------------------------------------------------
def test_v0_buffer_all_zero_at_init(backbone) -> None:
    buffers = dict(backbone.named_buffers())
    assert "V_0" in buffers
    assert "V_0" not in dict(backbone.named_parameters()), \
        "V_0 must be a zero buffer — the Oracle estimate has no V_0 term"
    v0 = buffers["V_0"]
    assert tuple(v0.shape) == (3, VECTOR_CHANNELS)
    assert torch.equal(v0, torch.zeros(3, VECTOR_CHANNELS))
    # and the forward really seeds V from this buffer (V_0 = 0 ⇒ cold V = 0)
    assert torch.count_nonzero(backbone.V_0).item() == 0


def test_frozen_zero_init_slots(backbone) -> None:
    # W_g: zero (graph-cond injection enters only through training)
    assert torch.equal(backbone.W_g.weight,
                       torch.zeros_like(backbone.W_g.weight))
    assert torch.equal(backbone.W_g.bias,
                       torch.zeros_like(backbone.W_g.bias))
    # phi_g final output: zero (both layers)
    for gate in backbone.phi_g:
        assert torch.equal(gate.weight, torch.zeros_like(gate.weight))
        assert torch.equal(gate.bias, torch.zeros_like(gate.bias))
    # phi_u FINAL output projection: zero; its input layer stays seeded
    for seq in backbone.phi_u:
        final = seq[-1]
        assert torch.equal(final.weight, torch.zeros_like(final.weight))
        assert torch.equal(final.bias, torch.zeros_like(final.bias))
        first = seq[0]
        assert not torch.equal(first.weight,
                               torch.zeros_like(first.weight)), \
            "phi_u input layer must keep the default seeded init"
    # W_s / W_v: default seeded (NOT zero) — config W_s_W_v: default_seeded
    for module in list(backbone.W_s) + list(backbone.W_v):
        assert torch.count_nonzero(module.weight).item() > 0
        assert torch.count_nonzero(module.bias).item() > 0
    # scalar-only LayerNorm keeps its affine defaults (not zeroed)
    for norm in backbone.norm:
        assert torch.count_nonzero(norm.weight).item() > 0


# ---------------------------------------------------------------------------
# R3 warm-start mapping + construction protocol
# ---------------------------------------------------------------------------
def test_build_applies_a2_mapping_with_clean_strict_audit(
        backbone, r3_state) -> None:
    state = backbone.state_dict()
    # every mapped target is torch.equal to its R3 source
    for target, source_key in A2_R3_MAPPING.items():
        assert target in state, target
        assert torch.equal(state[target], r3_state[source_key]), target
    # phi_s really is the R3 message MLP (same shape, same values)
    assert state["phi_s.0.0.weight"].shape == (48, 118)
    assert state["phi_s.1.2.weight"].shape == (48, 48)
    # A2 carries NO node_mlp / message_mlp / norm_h / w_v slots at all
    leaked = [k for k in state
              if k.split(".")[0] in
              ("node_mlp", "message_mlp", "norm_h", "w_v")]
    assert leaked == []
    # the strict audit runs clean over the injected payload (13 targets,
    # the other 24 payload keys declared consumed: node_mlp 12 +
    # message_mlp.2 4 + norm_h 6 + w_v 2)
    consumed = [k for k in sorted(r3_state)
                if k not in set(A2_R3_MAPPING.values())]
    assert len(consumed) == 24
    with torch.random.fork_rng(devices=[]):
        fresh = A2Backbone()
    audit = apply_r3_mapping(fresh, r3_state, dict(A2_R3_MAPPING),
                             consumed_source_keys=consumed)
    assert audit.missing == ()
    assert audit.unexpected == ()
    assert audit.shape_mismatch == ()
    assert audit.dtype_mismatch == ()
    assert audit.dual_write_aliases == ()
    assert len(audit.torch_equal_verified) == 13
    assert set(audit.torch_equal_verified) == set(A2_R3_MAPPING)
    # an UNdeclared payload key is still refused (strict full accounting)
    drifted = dict(r3_state)
    drifted["ghost.weight"] = torch.zeros(4)
    with torch.random.fork_rng(devices=[]):
        fresh2 = A2Backbone()
    from pes2ts_core.pesn.models.v542.candidates import V542InitMapError
    with pytest.raises(V542InitMapError) as exc:
        apply_r3_mapping(fresh2, drifted, dict(A2_R3_MAPPING),
                         consumed_source_keys=consumed)
    assert exc.value.code == "V542_INIT_MAP_UNEXPECTED"


def test_build_default_r3_path_typed_decoder_and_rng_neutrality(
        r3_state) -> None:
    # default path: SHA-pinned load inside build (no injected state)
    from_default = build(SEED + 1, DirectMoveDecoder())
    assert from_default.active_param_count == 54_896
    # and it equals the injected-state build parameter-for-parameter
    injected = build(SEED + 1, DirectMoveDecoder(), r3_state=r3_state)
    sd_a, sd_b = from_default.state_dict(), injected.state_dict()
    assert set(sd_a) == set(sd_b)
    assert all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a)
    # typed refusal for a non-decoder seam
    with pytest.raises(V542InitError) as exc:
        build(SEED, torch.nn.Linear(1, 1))            # type: ignore[arg-type]
    assert exc.value.code == "V542_A2_DECODER"
    # build NEVER resets/perturbs the caller's training seed: the whole
    # construction runs inside construction_scope (fork_rng restore)
    decoder = DirectMoveDecoder()                    # constructed OUTSIDE
    torch.manual_seed(4242)
    before = torch.random.get_rng_state().clone()
    built = build(SEED + 2, decoder)
    assert torch.equal(before, torch.random.get_rng_state()), \
        "build perturbed the caller's global RNG stream"
    assert int(torch.initial_seed()) == 4242, \
        "build must not reset the training seed (finish_training_seed is the runner's step 6)"
    assert built.active_param_count == 54_896


def test_build_refuses_tampered_a_ref(r3_state) -> None:
    # gate-2 B2: A2's own build enforces the frozen a_ref seam too
    # (not only build_candidate's pre-dispatch check)
    decoder = DirectMoveDecoder()
    decoder._a_ref = 0.1                       # tamper attempt
    with pytest.raises(V542InitError) as exc:
        build(SEED, decoder, r3_state=r3_state)
    assert exc.value.code == "V542_DECODER_A_REF"


def test_build_loads_r3_w_v_into_fresh_decoder(r3_state) -> None:
    # gate-2 B1: A2 (the former blocker) loads the SHA-verified R3 w_v
    # into the SHARED decoder on every build — order-independent
    decoder = DirectMoveDecoder()
    random_before = decoder.w_v.weight.detach().clone()
    build(SEED, decoder, r3_state=r3_state)
    assert not torch.equal(decoder.w_v.weight, random_before)
    assert torch.equal(decoder.w_v.weight, r3_state["w_v.weight"])
    assert torch.equal(decoder.w_v.bias, r3_state["w_v.bias"])
    # the backbone still owns NO w_v slot (decoder is the authority)
    built = build(SEED, DirectMoveDecoder(), r3_state=r3_state)
    assert not any(name.split(".")[0] == "w_v"
                   for name, _ in built.named_parameters())


# ---------------------------------------------------------------------------
# input legality: forward-view whitelist + typed invalid
# ---------------------------------------------------------------------------
def test_forward_view_whitelist_typed_rejection(backbone, batch) -> None:
    bad = dict(batch)
    bad["y_teacher"] = torch.zeros(2, 3)             # supervision leak attempt
    with pytest.raises(KeyAtomV3ContractError) as exc:
        backbone(bad)
    assert exc.value.code == "FORWARD_VIEW_LEAK"
    assert isinstance(exc.value, ValueError)         # typed, not bare RuntimeError
    bad2 = dict(batch)
    bad2["endpoint_geometry"] = torch.zeros(1, 3)     # forbidden input class
    with pytest.raises(KeyAtomV3ContractError) as exc:
        backbone(bad2)
    assert exc.value.code == "FORWARD_VIEW_LEAK"


def test_typed_invalid_non_finite_and_atomic_number(backbone, batch) -> None:
    bad = dict(batch)
    bad["pos"] = batch["pos"].clone()
    bad["pos"][0, 0] = float("inf")
    with pytest.raises(KeyAtomV3InvalidInputError) as exc:
        backbone(bad)
    assert exc.value.code == "NONFINITE_INPUT"
    bad = dict(batch)
    bad["edge_attr"] = batch["edge_attr"].clone()
    bad["edge_attr"][0, 0] = float("nan")
    with pytest.raises(KeyAtomV3InvalidInputError) as exc:
        backbone(bad)
    assert exc.value.code == "NONFINITE_INPUT"
    bad = dict(batch)
    bad["degK"] = torch.full_like(batch["degK"], float("nan"))
    with pytest.raises(KeyAtomV3InvalidInputError) as exc:
        backbone(bad)
    assert exc.value.code == "NONFINITE_INPUT"
    # Z legality (legacy taxonomy, same codes as StarEGNNV3)
    bad = dict(batch)
    bad["z"] = torch.zeros_like(batch["z"])
    with pytest.raises(KeyAtomV3InvalidInputError) as exc:
        backbone(bad)
    assert exc.value.code == "Z_PADDING"
    bad = dict(batch)
    bad["z"] = torch.full_like(batch["z"], 119)
    with pytest.raises(KeyAtomV3InvalidInputError) as exc:
        backbone(bad)
    assert exc.value.code == "Z_OUT_OF_RANGE"


# ---------------------------------------------------------------------------
# E(3): invariance of z/edge_msg + decoder displacement equivariance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("opened", [False, True],
                         ids=["cold_start", "phi_u_opened"])
def test_e3_invariance_and_decoder_equivariance(
        backbone, batch, opened: bool) -> None:
    model = copy.deepcopy(backbone)
    if opened:
        _open_phi_u(model)
    decoder = DirectMoveDecoder()
    rotation = _frame(3, det=1)
    reflection = _frame(5, det=-1)
    shift = torch.tensor([0.7, -1.3, 2.1])
    variants = {
        "translation": _shifted(batch, shift),
        "rotation": _rotated(batch, rotation),
        "reflection": _rotated(batch, reflection),
    }

    with torch.no_grad():
        base = model(batch)
        u_base = decoder(base.edge_msg, batch["pos"], batch["edge_index"],
                         batch["key_index"], batch["degK"])
        for name, variant in variants.items():
            out = model(variant)
            assert _max_delta(out.z, base.z) <= TOL, f"z not invariant: {name}"
            assert _max_delta(out.edge_msg, base.edge_msg) <= TOL, \
                f"edge_msg not invariant: {name}"
            # decoder displacement: invariant input ⇒ u follows pos only
            u_var = decoder(out.edge_msg, variant["pos"],
                            variant["edge_index"], variant["key_index"],
                            variant["degK"])
            if name == "translation":
                assert _max_delta(u_var, u_base) <= TOL, \
                    "u not translation-invariant"
            else:
                expected = u_base @ (rotation if name == "rotation"
                                     else reflection).T
                assert _max_delta(u_var, expected) <= TOL, \
                    f"u not equivariant: {name}"


# ---------------------------------------------------------------------------
# permutation / edge reversal / relabel invariance
# ---------------------------------------------------------------------------
def test_edge_permutation_and_edge_reversal_invariance(
        backbone, batch) -> None:
    n_edges = int(batch["edge_index"].shape[1])
    order = torch.randperm(n_edges)
    reversed_order = torch.arange(n_edges).flip(0)
    with torch.no_grad():
        base = backbone(batch)
        shuffled = backbone(_shuffle_edges(batch, order))
        # edge_msg rows follow the input edge order row-for-row
        assert _max_delta(shuffled.edge_msg, base.edge_msg[order]) <= TOL
        assert _max_delta(shuffled.z, base.z) <= TOL
        reversed_out = backbone(_shuffle_edges(batch, reversed_order))
        assert _max_delta(reversed_out.edge_msg,
                          base.edge_msg[reversed_order]) <= TOL
        assert _max_delta(reversed_out.z, base.z) <= TOL
    # decoder u is invariant to edge ordering (same edge set, same sum)
    decoder = DirectMoveDecoder()
    u_base = _decoder_u(backbone, batch, decoder)
    u_shuffled = _decoder_u(backbone, _shuffle_edges(batch, order), decoder)
    u_reversed = _decoder_u(backbone,
                            _shuffle_edges(batch, reversed_order), decoder)
    assert _max_delta(u_shuffled, u_base) <= TOL
    assert _max_delta(u_reversed, u_base) <= TOL


def test_node_permutation_invariance(backbone, batch) -> None:
    n_nodes = int(batch["pos"].shape[0])
    gen = torch.Generator().manual_seed(SEED + 7)
    perm = torch.randperm(n_nodes, generator=gen)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(n_nodes)
    permuted = dict(batch)
    for key in ("pos", "z", "is_key", "degK", "intent_role", "graph_index"):
        permuted[key] = batch[key][perm]
    permuted["edge_index"] = inv[batch["edge_index"].long()]
    # key ORDER is preserved (values re-point at the permuted rows)
    permuted["key_index"] = inv[batch["key_index"].long()]
    with torch.no_grad():
        base = backbone(batch)
        out = backbone(permuted)
    assert _max_delta(out.z, base.z) <= TOL, "z not permutation-invariant"
    assert _max_delta(out.edge_msg, base.edge_msg) <= TOL, \
        "edge_msg not permutation-invariant"
    # decoder rows follow key_index order ⇒ u unchanged
    decoder = DirectMoveDecoder()
    assert _max_delta(_decoder_u(backbone, permuted, decoder),
                      _decoder_u(backbone, batch, decoder)) <= TOL


def test_endpoint_relabel_invariance_is_bitwise(backbone, batch) -> None:
    # display-name swap must leave outputs BIT-identical (T1.1): A2 never
    # reads endpoint_names
    swapped = dict(batch)
    swapped["endpoint_names"] = tuple(
        reversed(tuple(batch["endpoint_names"])))
    with torch.no_grad():
        base = backbone(batch)
        out = backbone(swapped)
    assert torch.equal(out.z, base.z)
    assert torch.equal(out.edge_msg, base.edge_msg)


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------
def test_forward_determinism_two_calls_bitwise(backbone, batch) -> None:
    backbone.eval()
    with torch.no_grad():
        first = backbone(batch)
        second = backbone(batch)
    assert torch.equal(first.z, second.z)
    assert torch.equal(first.edge_msg, second.edge_msg)


def test_build_determinism_same_seed_bitwise(r3_state) -> None:
    a = build(SEED, DirectMoveDecoder(), r3_state=r3_state)
    b = build(SEED, DirectMoveDecoder(), r3_state=r3_state)
    sd_a, sd_b = a.state_dict(), b.state_dict()
    assert set(sd_a) == set(sd_b)
    assert all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a)
    # a different run seed changes the fresh (unmapped) params
    c = build(SEED + 1, DirectMoveDecoder(), r3_state=r3_state)
    sd_c = c.state_dict()
    assert not torch.equal(sd_c["W_v.0.weight"], sd_a["W_v.0.weight"]), \
        "fresh vector params must come from the candidate seed_init stream"
    # mapped slots stay R3-identical regardless of the run seed
    assert torch.equal(sd_c["phi_s.0.0.weight"], sd_a["phi_s.0.0.weight"])


# ---------------------------------------------------------------------------
# gradients: finite everywhere; the vector channel really receives gradient
# ---------------------------------------------------------------------------
def test_finite_gradients_on_every_param_cold_start(backbone, batch) -> None:
    backbone.zero_grad(set_to_none=True)
    out = backbone(batch)
    loss = out.z.sum() + out.edge_msg.sum()
    loss.backward()
    for name, param in backbone.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite grad"
    # Cold start (frozen zero-init phi_u output) closes the V→s path:
    # vector-path gradients are present but EXACTLY zero — never NaN.
    named = dict(backbone.named_parameters())
    for name in ("W_v.0.weight", "W_v.1.weight", "W_s.1.weight",
                 "phi_g.0.weight", "phi_g.1.weight", "phi_s.0.0.weight"):
        grad = named[name].grad
        assert torch.equal(grad, torch.zeros_like(grad)), \
            f"{name}: expected a closed-gate zero grad at cold start"
    # the live paths are not all zero (sanity: the loss really trains)
    assert torch.count_nonzero(named["phi_s.1.0.weight"].grad).item() > 0
    assert torch.count_nonzero(named["z_head.weight"].grad).item() > 0
    assert torch.count_nonzero(named["W_g.weight"].grad).item() > 0


def test_vector_channel_receives_nonzero_gradient(backbone, batch) -> None:
    model = copy.deepcopy(backbone)
    _open_phi_u(model)                    # open the frozen V→s gate
    model.zero_grad(set_to_none=True)
    out = model(batch)
    (out.z.sum() + out.edge_msg.sum()).backward()
    named = dict(model.named_parameters())
    # radial vector coefficients + neighbour gate + vector gates all get
    # nonzero, finite gradient through ‖V‖ → phi_u → s^L → z/edge_msg
    for name in ("W_v.0.weight", "W_v.0.bias",
                 "W_v.1.weight", "W_v.1.bias",
                 "W_s.1.weight",
                 "phi_g.0.weight", "phi_g.1.weight"):
        grad = named[name].grad
        assert grad is not None, name
        assert torch.isfinite(grad).all(), name
        assert torch.count_nonzero(grad).item() > 0, \
            f"{name} received no gradient — the vector channel is dead"
    # documented structural zero: layer 1 multiplies V_0 = 0, so W_s[0]
    # can never receive gradient (still counted in the frozen 54,896)
    w_s0 = named["W_s.0.weight"].grad
    assert w_s0 is not None and torch.isfinite(w_s0).all()
    assert torch.equal(w_s0, torch.zeros_like(w_s0))


# ---------------------------------------------------------------------------
# ADR-0033 §4 guard: edge_msg DEPENDS on the vector channel
# ---------------------------------------------------------------------------
def test_edge_msg_depends_on_vector_channel(backbone, batch) -> None:
    def perturb_v_path(model: A2Backbone) -> None:
        with torch.no_grad():
            model.W_v[0].weight.add_(0.25)   # radial coefficients
            model.W_s[1].weight.add_(0.25)   # neighbour scalar gate
            model.phi_g[0].weight.add_(0.25) # vector gates

    # control — frozen cold start: the zero-init phi_u output closes the
    # V→s path, so V-path perturbations provably CANNOT reach edge_msg
    # (bitwise): the dependency below is not an artefact.
    cold = copy.deepcopy(backbone)
    with torch.no_grad():
        base_cold = cold(batch).edge_msg
    perturb_v_path(cold)
    with torch.no_grad():
        after_cold = cold(batch).edge_msg
    assert torch.equal(base_cold, after_cold), \
        "cold start: edge_msg must be bitwise independent of the V path"

    # the intentional A2 semantics: with the gate open (as training
    # would), perturbing ONLY V-path params changes the edge_msg that
    # the shared MOVE decoder consumes — the vector channel acts on MOVE
    # through the s^L recompute (ADR-0033 §4; must not be "fixed").
    opened = copy.deepcopy(backbone)
    _open_phi_u(opened)
    with torch.no_grad():
        base_open = opened(batch).edge_msg
    perturb_v_path(opened)
    with torch.no_grad():
        after_open = opened(batch).edge_msg
    delta = _max_delta(after_open, base_open)
    assert delta > 1e-6, \
        f"edge_msg does not depend on the vector channel (delta={delta})"

    # ablation direction: zeroing the radial path entirely also moves it
    ablated = copy.deepcopy(backbone)
    _open_phi_u(ablated)
    with torch.no_grad():
        base_abl = ablated(batch).edge_msg
        for module in ablated.W_v:
            module.weight.zero_()
            module.bias.zero_()
    with torch.no_grad():
        after_abl = ablated(batch).edge_msg
    assert _max_delta(after_abl, base_abl) > 1e-6


# ---------------------------------------------------------------------------
# no torch.compile (ADR-0033 §2.1)
# ---------------------------------------------------------------------------
def test_no_torch_compile_anywhere(backbone) -> None:
    source = inspect.getsource(a2_mod)
    assert "torch.compile(" not in source
    # structural: the built tree is clean…
    assert assert_no_torch_compile(backbone) is None
    for name, child in backbone.named_modules():
        assert not hasattr(child, "_orig_mod"), name
    # …and a compile-wrapped child is a typed refusal

    class _FakeOptimized(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._orig_mod = torch.nn.Linear(2, 2)

    with pytest.raises(V542InterfaceError) as exc:
        assert_no_torch_compile(_FakeOptimized())
    assert exc.value.code == "V542_TORCH_COMPILE_FORBIDDEN"
