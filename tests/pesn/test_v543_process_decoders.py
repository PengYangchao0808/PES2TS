"""V5.4.3 WP6 — PROCESS-track candidate certificate (plan §6.3 :538–556,
§6.4 :557–571, §4.5 :246, §5.2 置换 column, §5.2.1 :346–352, §11.5 :1242).

What is pinned here, per candidate (P1 / P2 / P3 / P4 / P4-C):

* the MEASURED ``sum(p.numel())`` equals the arithmetic count in the module and
  lands inside the frozen band 34,200–37,800 (§6.4 :563–569) — P4-C instead
  lands inside its own «P4 measured ±2 %» band (§6.3 :547), never mixed with it;
* the §6.1 protocol (``forward(ctx)`` only) and the §6.1 13-field record;
* registration into a FRESH :class:`DecoderRegistry` (never the process-wide
  WP4 object — its M0/P0-only state is asserted by ``test_v543_registry.py``);
* zero-residual initialisation EXACTLY zero (§4.5 :246) so V543 == V542 A0;
* forward shape/dtype ``[G]`` fp32 finite (§6.1 :496, §4.3 :224);
* pair-permutation and token-permutation INVARIANCE (§5.2 置换 column: the
  PROCESS output is a graph-level scalar pair);
* the ``P = 0`` context of §5.2 :309;
* the §4.2/§4.3 flow through the real ``V543Assembly`` + ``V543Model``
  (``q = tanh(q0_raw + δq)``, ``e = e0 + δe``);
* §7.2 :713–726 gradient isolation (encoder/other-track grads ``None``);
* the P4-C-vs-P4 parameter comparison (the H-P2 capacity control, §3.2 :167).

It also asserts the two plan ↔ WP4 conflicts WP6 found:
``P4-C`` is not a registry-declarable id (:547 vs §6.1/§6.3 :541–548), and the
PROCESS view carries no ``key_edge_index`` (:349) although §6.3 :554 mentions it
for the local attention bias."""

from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.pesn.models.v543.contracts import (  # noqa: E402
    DECODER_CONTEXT_SCHEMA_V543,
    MoveContextV1,
    ProcessContextV1,
    ProcessResidual,
    V543ParamBudgetError,
)
from pes2ts_core.pesn.models.v543.process import (  # noqa: E402
    DECODER_ID_P1,
    DECODER_ID_P2,
    DECODER_ID_P3,
    DECODER_ID_P4,
    DECODER_ID_P4C,
    FORMAL_PROCESS_DECODER_IDS_V543,
    MEASURED_PARAMS_V543,
    P1_EXPECTED_PARAMS,
    P2_EXPECTED_PARAMS,
    P3_EXPECTED_PARAMS,
    P4C_BAND_PCT,
    P4C_EXPECTED_PARAMS,
    P4C_SPEC,
    P4_EXPECTED_PARAMS,
    PLAN_PARAM_ESTIMATES_V543,
    PROCESS_CANDIDATE_IDS_V543,
    V543ProcessDecoderError,
    assert_p4c_matches_p4,
    build_process_candidate,
    p4_reference_params,
    p4c_band,
    process_candidate_report,
    process_candidate_specs,
    register_process_candidates,
)
from pes2ts_core.pesn.models.v543.process.deepsets import (  # noqa: E402
    P2_FEATURE_DIM,
    P2_READOUT,
    P2_TOKEN_HIDDEN,
    ProcessDeepSets,
)
from pes2ts_core.pesn.models.v543.process.graph_gnn import (  # noqa: E402
    P3_HIDDEN,
    P3_LAYERS,
    ProcessLatentGraphGNN,
)
from pes2ts_core.pesn.models.v543.process.graph_transformer import (  # noqa: E402
    P4_DIM,
    P4_FFN,
    P4_HEAD_DIM,
    P4_HEADS,
    P4_LAYERS,
    P4ControlSpec,
    ProcessGraphTransformer,
    ProcessP4Control,
)
from pes2ts_core.pesn.models.v543.process.residual_mlp import (  # noqa: E402
    P1_HIDDEN,
    ProcessResidualMLP,
)
from pes2ts_core.pesn.models.v543.process.tokens import (  # noqa: E402
    PAIR_EDGE_FEATURE_DIM_V543,
    local_token_edges,
    token_tensors,
)
from pes2ts_core.pesn.models.v543.registry import (  # noqa: E402
    P0_SPEC,
    PARAM_BAND_MAX_V543,
    PARAM_BAND_MIN_V543,
    PARAM_HARD_CAP_V543,
    REGISTRY_RECORD_FIELDS,
    REGISTRY_V543,
    TRACK_PROCESS,
    DecoderRegistry,
    DecoderSpec,
    ProcessDecoderProtocol,
    V543RegistryError,
    assert_decoder_protocol,
    count_trainable_params,
    default_registry,
)
from tests.pesn._v543_fixtures_wp6 import (  # noqa: E402
    batch_with_status,
    drop_all_pairs,
    permute_key_tokens,
    permute_pairs,
    process_model,
    real_process_context,
)

#: Every implemented candidate id (the four formal ones + the control).
ALL_CANDIDATE_IDS = PROCESS_CANDIDATE_IDS_V543
#: The four candidates that ARE campaign-declared registry ids (§6.3 :541–546).
REGISTRABLE_IDS = FORMAL_PROCESS_DECODER_IDS_V543


def _decoder(decoder_id: str) -> torch.nn.Module:
    return build_process_candidate(decoder_id)


@pytest.fixture(scope="module")
def process_ctx() -> ProcessContextV1:
    return real_process_context()[2]


# ---------------------------------------------------------------------------
# parameter budget (§6.4 :557–571)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("decoder_id", ALL_CANDIDATE_IDS)
def test_measured_parameter_count_is_exact_and_in_the_band(
        decoder_id: str) -> None:
    """``sum(p.numel())`` is the module's own arithmetic AND inside the band."""
    decoder = _decoder(decoder_id)
    measured = count_trainable_params(decoder)
    assert measured == MEASURED_PARAMS_V543[decoder_id]
    assert decoder.n_trainable_params == measured
    assert PARAM_BAND_MIN_V543 <= measured <= PARAM_BAND_MAX_V543, (
        f"{decoder_id}: {measured} outside "
        f"[{PARAM_BAND_MIN_V543}, {PARAM_BAND_MAX_V543}]")
    assert measured <= PARAM_HARD_CAP_V543


@pytest.mark.parametrize("decoder_id", ALL_CANDIDATE_IDS)
def test_measured_count_is_close_to_the_plan_estimate(
        decoder_id: str) -> None:
    """The reported estimate column: within 5 % of the §6.3 table.

    The estimate is a REPORTING target only (§6.4 :569 "估算值不得作为实现前
    exact 断言"); the frozen band is the assertion.
    """
    if decoder_id == DECODER_ID_P4C:
        pytest.skip("P4-C has no §6.3 estimate — it is matched to P4 (:547)")
    measured = MEASURED_PARAMS_V543[decoder_id]
    estimate = PLAN_PARAM_ESTIMATES_V543[decoder_id]
    deviation = 100.0 * (measured - estimate) / float(estimate)
    assert abs(deviation) <= 5.0, (decoder_id, measured, estimate, deviation)


def test_parameter_counts_match_the_plan_table() -> None:
    """The exact counts the WP6 report quotes (bases: §6.3 :543–547)."""
    assert P1_EXPECTED_PARAMS == 36_482      # 10,400 + 25,760 + 322
    assert P2_EXPECTED_PARAMS == 35_418      # 4,312 + 30,848 + 258
    assert P3_EXPECTED_PARAMS == 36_210
    assert P4_EXPECTED_PARAMS == 37_546
    assert P4C_EXPECTED_PARAMS == 37_762
    for decoder_id, expected in MEASURED_PARAMS_V543.items():
        assert count_trainable_params(_decoder(decoder_id)) == expected, \
            decoder_id


# ---------------------------------------------------------------------------
# structure: each candidate is the §6.3 table row, not an approximation
# ---------------------------------------------------------------------------
def test_p1_is_the_plan_mlp() -> None:
    """§6.3 :543 "``graph_s[64]`` → 160 → 160 → 2"."""
    decoder = ProcessResidualMLP()
    assert P1_HIDDEN == 160
    assert tuple(decoder.fc1.weight.shape) == (160, 64)
    assert tuple(decoder.fc2.weight.shape) == (160, 160)
    assert tuple(decoder.head.weight.shape) == (2, 160)


def test_p2_is_the_plan_deepsets() -> None:
    """§6.3 :544 "token 编码 48→88；mean+max 池化；拼接（240）→ 128 → 2"."""
    decoder = ProcessDeepSets()
    assert P2_TOKEN_HIDDEN == 88
    assert P2_READOUT == 128
    assert P2_FEATURE_DIM == 240 == 64 + 88 + 88
    assert tuple(decoder.encode.weight.shape) == (88, 48)
    assert tuple(decoder.fc1.weight.shape) == (128, 240)
    assert tuple(decoder.head.weight.shape) == (2, 128)


def test_p3_is_the_plan_graph_gnn(process_ctx: ProcessContextV1) -> None:
    """§6.3 :545 "同一 token 图 2 层消息传递；hidden 56；边特征 8"."""
    decoder = ProcessLatentGraphGNN()
    assert P3_HIDDEN == 56
    assert P3_LAYERS == 2
    assert len(decoder.blocks) == 2
    assert PAIR_EDGE_FEATURE_DIM_V543 == 8
    tokens = token_tensors(process_ctx)
    edges = local_token_edges(process_ctx, tokens)
    # the edge feature is the frozen pair_s[20:28] slice, never a coordinate
    assert tuple(edges.edge_feature.shape)[-1] == 8
    assert edges.n_edges > 0
    assert int(edges.edge_index.max()) < tokens.n_tokens


def test_p4_is_the_plan_graph_transformer() -> None:
    """§6.3 :546 "2 层、d=48、4 heads（head_dim 12）、FFN 80"."""
    decoder = ProcessGraphTransformer()
    assert (P4_DIM, P4_HEADS, P4_HEAD_DIM, P4_FFN, P4_LAYERS) == \
        (48, 4, 12, 80, 2)
    assert len(decoder.blocks) == 2
    block = decoder.blocks[0]
    assert block.heads == 4
    assert block.head_dim == 12
    assert tuple(block.qkv.weight.shape) == (3 * 48, 48)
    assert tuple(block.ffn_up.weight.shape) == (80, 48)
    assert tuple(block.ffn_down.weight.shape) == (48, 80)
    # §6.3 :554 "局部图 bias ... 不用可学习全局帧": one scalar per head
    assert tuple(block.local_bias.shape) == (4,)


def test_p4c_replaces_only_the_mixing_operator() -> None:
    """§6.3 :547 same width/tokens/params, attention → MLP+pooling."""
    control = ProcessP4Control()
    transformer = ProcessGraphTransformer()
    assert control.dim == transformer.dim == 48
    assert control.layers == transformer.layers == 2
    assert control.ffn == transformer.ffn == 80
    # the attention machinery is GONE, the token input and head are shared
    assert not hasattr(control.blocks[0], "qkv")
    assert hasattr(control.blocks[0], "mix1")
    assert tuple(control.head.weight.shape) == \
        tuple(transformer.head.weight.shape)
    assert tuple(control.input.graph_proj.weight.shape) == \
        tuple(transformer.input.graph_proj.weight.shape)


# ---------------------------------------------------------------------------
# P4-C: the independent ±2 % matched band (§6.3 :547)
# ---------------------------------------------------------------------------
def test_p4c_is_within_two_percent_of_p4() -> None:
    p4 = count_trainable_params(ProcessGraphTransformer())
    control = count_trainable_params(ProcessP4Control())
    assert p4 == p4_reference_params()
    deviation = 100.0 * (control - p4) / float(p4)
    assert abs(deviation) <= P4C_BAND_PCT, (control, p4, deviation)
    assert assert_p4c_matches_p4(control) == control
    band = p4c_band()
    assert band["reference_params"] == p4
    assert band["band_pct"] == 2.0
    assert band["min"] == 36_795 and band["max"] == 38_297
    assert band["min"] <= control <= band["max"]


@pytest.mark.parametrize("count", [34_200, 36_000, 39_000])
def test_p4c_refuses_a_count_outside_the_matched_band(count: int) -> None:
    """The ±2 % band is an assertion, and it is NOT the 36K ±5 % band."""
    with pytest.raises(V543ParamBudgetError) as error:
        assert_p4c_matches_p4(count)
    assert error.value.code == "V543_P4C_PARAM_MATCH"
    assert "±2" in error.value.detail or "2.0" in error.value.detail


def test_p4c_construction_asserts_the_matched_band() -> None:
    """A widened control must be refused at CONSTRUCTION (§6.4 :569 habit)."""
    with pytest.raises(V543ParamBudgetError) as error:
        ProcessP4Control(mix_hidden=120)
    assert error.value.code == "V543_P4C_PARAM_MATCH"


def test_p4c_record_is_the_13_fields_plus_band_provenance() -> None:
    """§6.1 :503–517 fields, in order, plus the band the control froze."""
    measured = count_trainable_params(ProcessP4Control())
    record = P4C_SPEC.record(measured)
    assert tuple(record) == REGISTRY_RECORD_FIELDS
    assert record["decoder_id"] == "P4-C"
    assert record["track"] == TRACK_PROCESS
    assert record["contract_version"] == DECODER_CONTEXT_SCHEMA_V543
    assert record["n_trainable_params"] == measured
    assert record["status"] == "diagnostic"          # §6.3 :547 诊断
    assert record["nfe"] == 1 and record["stochastic"] is False
    assert record["zero_init"] is True
    extra = P4C_SPEC.record_extra(measured)
    assert extra["band"]["band_pct"] == P4C_BAND_PCT == 2.0
    assert extra["band"]["reference_decoder_id"] == "P4"
    assert extra["band"]["min"] <= measured <= extra["band"]["max"]
    assert extra["init_seed"] >= 0
    assert list(extra["record_fields"]) == list(REGISTRY_RECORD_FIELDS)


# ---------------------------------------------------------------------------
# §6.1 protocol / record / registration (P1–P4)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("decoder_id", REGISTRABLE_IDS)
def test_formal_candidates_satisfy_the_process_protocol(
        decoder_id: str, process_ctx: ProcessContextV1) -> None:
    decoder = _decoder(decoder_id)
    assert isinstance(decoder, ProcessDecoderProtocol)
    assert assert_decoder_protocol(decoder, decoder_id=decoder_id,
                                   track=TRACK_PROCESS) is decoder
    names = [name for name in inspect.signature(decoder.forward).parameters]
    assert names == ["ctx"], names
    assert isinstance(decoder(process_ctx), ProcessResidual)


@pytest.mark.parametrize("decoder_id", ALL_CANDIDATE_IDS)
def test_candidates_refuse_a_move_context(decoder_id: str) -> None:
    """§5.2.1 :348–352 — PROCESS may not read MOVE state, and vice versa."""
    move = MoveContextV1(
        graph_s=torch.zeros(2, 64), key_s=torch.zeros(3, 48),
        key_v=torch.zeros(3, 8, 3), u0=torch.ones(3, 3),
        key_edge_index=torch.zeros(2, 0, dtype=torch.long),
        key_edge_attr=torch.zeros(0, 8),
        key_graph=torch.zeros(3, dtype=torch.long),
        key_ptr=torch.tensor([0, 3], dtype=torch.long))
    decoder = _decoder(decoder_id)
    with pytest.raises((V543ProcessDecoderError, AttributeError)) as error:
        decoder(move)
    code = getattr(error.value, "code", "")
    assert code in ("V543_PROCESS_CONTEXT_TYPE", "V543_TRACK_ISOLATION")


@pytest.mark.parametrize("decoder_id", REGISTRABLE_IDS)
def test_formal_candidates_register_with_the_13_field_record(
        decoder_id: str) -> None:
    """§6.1 :503–517 record + §6.4 construction assertion through the registry."""
    registry = default_registry()          # a FRESH registry (M0/P0 only)
    register_process_candidates(registry)
    assert decoder_id in registry.ids(track=TRACK_PROCESS)
    built = registry.build(decoder_id, track=TRACK_PROCESS)
    assert count_trainable_params(built) == MEASURED_PARAMS_V543[decoder_id]
    record = registry.describe(decoder_id, build=True)
    for name in REGISTRY_RECORD_FIELDS:
        assert name in record, name
    assert record["decoder_id"] == decoder_id
    assert record["track"] == TRACK_PROCESS
    assert record["n_trainable_params"] == MEASURED_PARAMS_V543[decoder_id]
    assert record["contract_version"] == DECODER_CONTEXT_SCHEMA_V543
    assert record["equivariance"] == "invariant"
    assert record["nfe"] == 1 and record["stochastic"] is False
    assert record["zero_init"] is True
    assert record["status"] in ("proposed", "smoke")


def test_process_candidate_specs_are_the_formal_table() -> None:
    specs = process_candidate_specs()
    assert tuple(spec.decoder_id for spec in specs) == REGISTRABLE_IDS
    for spec in specs:
        assert spec.track == TRACK_PROCESS
        assert spec.equivariance == "invariant"
        assert spec.nfe == 1 and spec.stochastic is False and spec.zero_init
        assert spec.contract_version == DECODER_CONTEXT_SCHEMA_V543
        assert tuple(spec.record(MEASURED_PARAMS_V543[spec.decoder_id])) == \
            REGISTRY_RECORD_FIELDS


def test_registration_does_not_touch_the_wp4_registry() -> None:
    """The process-wide registry is WP4's: it stays M0/P0 (§11.2)."""
    before = REGISTRY_V543.ids()
    registry = DecoderRegistry()
    registry.register(P0_SPEC)
    register_process_candidates(registry)
    assert registry.ids() == ("P0",) + REGISTRABLE_IDS
    assert REGISTRY_V543.ids() == before == ("M0", "P0")


def test_p4c_is_not_a_registry_declarable_id() -> None:
    """PLAN ↔ WP4 CONFLICT, asserted rather than hidden.

    §6.3 :547 names ``P4-C``; §6.1/§6.3 :541–548 and
    ``registry.DECODER_IDS_V543`` declare only ``P0…P5``, so a DecoderSpec with
    that id is a typed refusal.  WP6 keeps the plan's id for the DECODER and
    records it through :class:`P4ControlSpec` instead of editing WP4.
    """
    with pytest.raises(V543RegistryError) as error:
        DecoderSpec(decoder_id="P4-C", track=TRACK_PROCESS,
                    model_version="v543_p4c_param_matched_control_v1",
                    factory=ProcessP4Control)
    assert error.value.code == "V543_REGISTRY_UNKNOWN"
    with pytest.raises(V543RegistryError) as error:
        assert_decoder_protocol(ProcessP4Control(), decoder_id="P4-C",
                                track=TRACK_PROCESS)
    assert error.value.code == "V543_REGISTRY_UNKNOWN"
    assert P4ControlSpec().decoder_id == "P4-C"


# ---------------------------------------------------------------------------
# forward contract: shape / dtype / zero-residual init (§6.1 :496, §4.5 :246)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("decoder_id", ALL_CANDIDATE_IDS)
def test_forward_shape_dtype_and_finiteness(
        decoder_id: str, process_ctx: ProcessContextV1) -> None:
    residual = _decoder(decoder_id)(process_ctx)
    assert isinstance(residual, ProcessResidual)
    for name in ("delta_q_raw", "delta_e"):
        value = getattr(residual, name)
        assert tuple(value.shape) == (process_ctx.graph_s.shape[0],)
        assert value.dtype == torch.float32
        assert bool(torch.isfinite(value).all())


@pytest.mark.parametrize("decoder_id", ALL_CANDIDATE_IDS)
def test_zero_residual_initialisation_is_exact(
        decoder_id: str, process_ctx: ProcessContextV1) -> None:
    """§4.5 :246 "最后一层 ``Linear(·,2)`` 的 weight/bias" zeroed."""
    residual = _decoder(decoder_id)(process_ctx)
    assert torch.equal(residual.delta_q_raw, torch.zeros_like(
        residual.delta_q_raw))
    assert torch.equal(residual.delta_e, torch.zeros_like(residual.delta_e))
    q = torch.tanh(process_ctx.q0_raw + residual.delta_q_raw)
    assert torch.equal(q, torch.tanh(process_ctx.q0_raw))
    assert torch.equal(process_ctx.endpoint0_logit + residual.delta_e,
                       process_ctx.endpoint0_logit)


# ---------------------------------------------------------------------------
# permutation invariance (§5.2 置换 column / §11.5 :1231)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("decoder_id", ALL_CANDIDATE_IDS)
def test_pair_permutation_invariance(
        decoder_id: str, process_ctx: ProcessContextV1) -> None:
    """Permuting pair ROWS must not change the graph-level residual.

    Exact mathematics, floating-point reassociation allowed: the tolerance is
    1e-6 (fp32) and is asserted on the DIFFERENCE, never on a relaxed "close
    enough" output.
    """
    decoder = _decoder(decoder_id)
    base = decoder(process_ctx)
    for seed in (0, 1, 7):
        permuted = decoder(permute_pairs(process_ctx, seed=seed))
        assert float((permuted.delta_q_raw - base.delta_q_raw)
                     .detach().abs().max()) <= 1e-6, (decoder_id, seed)
        assert float((permuted.delta_e - base.delta_e)
                     .detach().abs().max()) <= 1e-6


@pytest.mark.parametrize("decoder_id", ALL_CANDIDATE_IDS)
def test_token_permutation_invariance(
        decoder_id: str, process_ctx: ProcessContextV1) -> None:
    """Permuting the KEY rows (the token order) must not change the residual."""
    decoder = _decoder(decoder_id)
    base = decoder(process_ctx)
    for seed in (0, 3, 11):
        permuted = decoder(permute_key_tokens(process_ctx, seed=seed))
        assert float((permuted.delta_q_raw - base.delta_q_raw)
                     .detach().abs().max()) <= 1e-6, (decoder_id, seed)
        assert float((permuted.delta_e - base.delta_e)
                     .detach().abs().max()) <= 1e-6


@pytest.mark.parametrize("decoder_id", ALL_CANDIDATE_IDS)
def test_empty_pair_context_is_legal(
        decoder_id: str, process_ctx: ProcessContextV1) -> None:
    """§5.2 形状约束 :309 "允许 ``P=0`` … Decoder 必须能处理空 pair"."""
    empty = drop_all_pairs(process_ctx)
    assert int(empty.pair_s.shape[0]) == 0
    assert int(empty.token_ptr[-1]) == empty.graph_s.shape[0] + empty.key_s.shape[0]
    residual = _decoder(decoder_id)(empty)
    assert tuple(residual.delta_q_raw.shape) == (empty.graph_s.shape[0],)
    assert bool(torch.isfinite(residual.delta_q_raw).all())


def test_p4_attention_is_within_graph_only() -> None:
    """§6.3 :554 "token 级 attention"，不做跨图/原子级稠密 N²."""
    _, context, process_ctx = real_process_context()
    tokens = token_tensors(process_ctx)
    same_graph = (tokens.token_graph.unsqueeze(0)
                  == tokens.token_graph.unsqueeze(1))
    # every graph is [graph, 3 keys, 2 pairs] -> 6 tokens -> 36 in-graph pairs
    spans = [(int(tokens.token_ptr[g + 1]) - int(tokens.token_ptr[g]))
             for g in range(int(context.n_graphs))]
    assert spans == [6, 6, 6, 6]
    assert int(same_graph.sum()) == sum(span * span for span in spans)
    assert not bool(same_graph.all())          # cross-graph pairs are masked


# ---------------------------------------------------------------------------
# determinism / RNG discipline (§7.2 :667)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("decoder_id", ALL_CANDIDATE_IDS)
def test_initialisation_is_deterministic(decoder_id: str) -> None:
    torch.manual_seed(1234)
    first = _decoder(decoder_id).state_dict()
    torch.manual_seed(999)
    second = _decoder(decoder_id).state_dict()
    assert list(first) == list(second)
    for name in first:
        assert torch.equal(first[name], second[name]), name


@pytest.mark.parametrize("decoder_id", ALL_CANDIDATE_IDS)
def test_construction_does_not_disturb_the_global_rng(decoder_id: str) -> None:
    """fork_rng discipline: building a candidate cannot shift the sampler."""
    torch.manual_seed(2024)
    before = torch.random.get_rng_state().clone()
    _decoder(decoder_id)
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)


# ---------------------------------------------------------------------------
# the real §4.2 flow: assembly + V543Model
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("decoder_id", ALL_CANDIDATE_IDS)
def test_model_forward_emits_q_and_endpoint(decoder_id: str) -> None:
    """§4.3 :223–224 — ``q = tanh(q0_raw + δq)``, ``e = e0 + δe``."""
    batch = batch_with_status(("ok",) * 4, jitter_seed=0)
    from pes2ts_core.pesn.contracts.key_atom_v3 import forward_view_v3

    view = forward_view_v3(dict(batch))
    decoder = _decoder(decoder_id)
    model = process_model(decoder, decoder_id=decoder_id)
    out = model(view)
    assert set(out) == {"u", "z", "q", "endpoint_logit", "order_score"}
    debug = model.forward_debug(view)
    n_graphs = int(debug["n_graphs"])
    assert tuple(out["q"].shape) == (n_graphs,)
    assert tuple(out["endpoint_logit"].shape) == (n_graphs,)
    assert bool(torch.isfinite(out["q"]).all())
    assert out["order_score"] is out["q"]
    assert torch.equal(out["q"], torch.tanh(
        debug["q0_raw"] + debug["delta_q_raw"]))
    assert torch.equal(out["endpoint_logit"],
                       debug["endpoint0_logit"] + debug["delta_e"])
    # a freshly built candidate is still the exact V542 A0 baseline (§4.5)
    assert torch.equal(out["q"], torch.tanh(debug["q0_raw"]))
    assert int(debug["encoder_forward_count"]) == 2


@pytest.mark.parametrize("decoder_id", ALL_CANDIDATE_IDS)
def test_gradients_reach_only_the_process_decoder(decoder_id: str) -> None:
    """§7.2 :713–726 — encoder and inactive-track grads stay ``None``."""
    batch = batch_with_status(("ok",) * 4, jitter_seed=0)
    from pes2ts_core.pesn.contracts.key_atom_v3 import forward_view_v3

    view = forward_view_v3(dict(batch))
    decoder = _decoder(decoder_id)
    model = process_model(decoder, decoder_id=decoder_id)
    out = model(view)
    loss = out["q"].pow(2).sum() + out["endpoint_logit"].pow(2).sum()
    assert bool(torch.isfinite(loss))
    model.zero_grad()
    loss.backward()
    isolation = model.assert_gradient_isolation(require_active_grad=True)
    assert isolation["encoder_params_with_grad"] == 0
    assert isolation["inactive_params_with_grad"] == 0
    assert isolation["active_params_with_grad"] > 0
    for parameter in model.assembly.encoder.parameters():
        assert parameter.grad is None
    assert any(p.grad is not None for p in decoder.parameters())


# ---------------------------------------------------------------------------
# the WP6 parameter report
# ---------------------------------------------------------------------------
def test_process_candidate_report_matches_the_measurements() -> None:
    report = process_candidate_report()
    assert report["track"] == TRACK_PROCESS
    assert report["target"] == 36_000
    rows = {row["decoder_id"]: row for row in report["candidates"]}
    assert set(rows) == set(ALL_CANDIDATE_IDS)
    for decoder_id in REGISTRABLE_IDS:
        row = rows[decoder_id]
        assert row["measured"] == MEASURED_PARAMS_V543[decoder_id]
        assert row["plan_estimate"] == PLAN_PARAM_ESTIMATES_V543[decoder_id]
        assert row["in_frozen_band"] is True
    control = rows[DECODER_ID_P4C]
    assert control["in_matched_band"] is True
    assert control["matched_band"]["band_pct"] == 2.0
    assert "independent" in control["band_rule"]
