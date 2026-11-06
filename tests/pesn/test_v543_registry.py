"""V5.4.3 WP4 — decoder registry CPU certificate (plan §6.1 :485–519,
§6.2 :522, §6.3 :539, §6.4 :557–571, §11.2 ``registry.py``).

What is pinned here:

* the two §6.1 protocols are ``typing.Protocol``-based and structural: an
  object with exactly ``forward(ctx)`` satisfies them, an object with a second
  argument does NOT (a decoder that accepts more can read data or supervision);
* the §6.1 record emits EXACTLY the plan's field list, in the plan's order;
* the §6.4 parameter band is a construction-time typed refusal at 34,199 /
  37,801 / 40,001 and accepts the band edges 34,200 / 37,800;
* M0/P0 are zero-parameter baselines whose ``forward`` returns a zero residual
  of the right shape and refuses the other track's context;
* the §6.1 prohibition (:519 "Decoder 私自实现数据读取、损失定义、checkpoint
  选择") is enforced structurally, not by convention."""

from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.pesn.models.v543.contracts import (  # noqa: E402
    DECODER_CONTEXT_SCHEMA_V543,
    MoveContextV1,
    ProcessContextV1,
    ProcessResidual,
)
from pes2ts_core.pesn.models.v543.registry import (  # noqa: E402
    BASELINE_DECODER_BY_TRACK,
    DECODER_IDS_V543,
    MOVE_DECODER_IDS_V543,
    M0_SPEC,
    P0_SPEC,
    PARAM_BAND_MAX_V543,
    PARAM_BAND_MIN_V543,
    PARAM_HARD_CAP_V543,
    PARAM_TARGET_V543,
    PROCESS_DECODER_IDS_V543,
    REGISTRY_RECORD_FIELDS,
    REGISTRY_V543,
    TRACK_MOVE,
    TRACK_PROCESS,
    DecoderRegistry,
    DecoderSpec,
    MoveDecoderProtocol,
    ProcessDecoderProtocol,
    V543DecoderProtocolError,
    V543RegistryError,
    ZeroMoveDecoder,
    ZeroProcessDecoder,
    assert_budget,
    assert_decoder_protocol,
    assert_parameter_matching_report,
    baseline_decoder_id,
    build_baseline,
    build_decoder,
    count_trainable_params,
    default_registry,
    describe_decoder,
    param_band_tolerance,
    track_of_decoder_id,
)
from pes2ts_core.pesn.models.v543.contracts import V543ParamBudgetError  # noqa: E402
from tests.pesn._v543_fixtures_wp4 import (  # noqa: E402
    SynthMoveRefiner,
    SynthProcessRefiner,
    synth_move_spec,
    synth_process_spec,
)

# ---------------------------------------------------------------------------
# protocol shape
# ---------------------------------------------------------------------------
def test_move_protocol_is_forward_only() -> None:
    """The §6.1 MOVE protocol is ``forward(MoveContextV1) -> Tensor``."""
    signature = inspect.signature(MoveDecoderProtocol.forward)
    names = [name for name in signature.parameters]
    assert names == ["self", "ctx"], names
    assert isinstance(ZeroMoveDecoder(), MoveDecoderProtocol)


def test_process_protocol_is_forward_only() -> None:
    """The §6.1 PROCESS protocol is ``forward(ProcessContextV1) ->
    ProcessResidual``."""
    signature = inspect.signature(ProcessDecoderProtocol.forward)
    names = [name for name in signature.parameters]
    assert names == ["self", "ctx"], names
    assert isinstance(ZeroProcessDecoder(), ProcessDecoderProtocol)


def test_decoder_ids_are_the_plan_table() -> None:
    assert MOVE_DECODER_IDS_V543 == ("M0", "M1", "M2", "M3", "M4", "M5")
    assert PROCESS_DECODER_IDS_V543 == ("P0", "P1", "P2", "P3", "P4", "P5")
    assert DECODER_IDS_V543 == MOVE_DECODER_IDS_V543 + PROCESS_DECODER_IDS_V543
    assert track_of_decoder_id("M3") == TRACK_MOVE
    assert track_of_decoder_id("P4") == TRACK_PROCESS
    with pytest.raises(V543RegistryError) as error:
        track_of_decoder_id("X9")
    assert error.value.code == "V543_REGISTRY_UNKNOWN"


# ---------------------------------------------------------------------------
# §6.1 record
# ---------------------------------------------------------------------------
def test_record_has_the_plan_fields() -> None:
    """§6.1 :503–517 — the record's field list, in the plan's order."""
    assert REGISTRY_RECORD_FIELDS == (
        "decoder_id", "track", "model_version", "contract_version",
        "n_trainable_params", "equivariance", "stochastic", "nfe",
        "zero_init", "status", "checkpoint_sha256", "config_sha256",
        "acceptance")
    record = M0_SPEC.record(0)
    assert tuple(record) == REGISTRY_RECORD_FIELDS
    assert record["contract_version"] == DECODER_CONTEXT_SCHEMA_V543
    assert record["n_trainable_params"] == 0
    assert record["track"] == TRACK_MOVE


def test_describe_payload_is_complete() -> None:
    described = describe_decoder("P0")
    assert described["decoder_id"] == "P0"
    assert described["track"] == TRACK_PROCESS
    assert described["n_trainable_params"] == 0
    assert described["is_baseline"] is True
    assert described["param_band"]["target"] == PARAM_TARGET_V543
    assert described["param_band"]["min"] == PARAM_BAND_MIN_V543
    assert described["param_band"]["max"] == PARAM_BAND_MAX_V543
    assert described["param_band"]["hard_cap"] == PARAM_HARD_CAP_V543
    assert described["record_fields"] == list(REGISTRY_RECORD_FIELDS)
    # a describe(build=True) re-measures instead of quoting the registration
    assert describe_decoder("P0", build=True)["n_trainable_params_source"] == \
        "measured_at_describe"
    assert described["n_trainable_params_source"] == "measured_at_register"


def test_registry_holds_only_the_wp4_baselines() -> None:
    assert REGISTRY_V543.ids() == ("M0", "P0")
    assert REGISTRY_V543.ids(track=TRACK_MOVE) == ("M0",)
    assert REGISTRY_V543.ids(track=TRACK_PROCESS) == ("P0",)
    for absent in ("M1", "M2", "M3", "M4", "M5", "P1", "P2", "P3", "P4", "P5"):
        with pytest.raises(V543RegistryError) as error:
            REGISTRY_V543.spec(absent)
        assert error.value.code == "V543_REGISTRY_UNKNOWN"
        assert "WP5" in error.value.detail or "not registered" in \
            error.value.detail


# ---------------------------------------------------------------------------
# §6.4 parameter band
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("count,code", [
    (34_199, "V543_REGISTRY_PARAMS_BAND"),
    (37_801, "V543_REGISTRY_PARAMS_BAND"),
    (40_001, "V543_REGISTRY_PARAMS_CAP"),
])
def test_out_of_band_is_refused(count: int, code: str) -> None:
    with pytest.raises(V543ParamBudgetError) as error:
        assert_budget(count, decoder_id="M1", track=TRACK_MOVE)
    assert error.value.code == code
    assert str(count) in error.value.detail


@pytest.mark.parametrize("count", [34_200, 36_000, 37_800])
def test_band_edges_are_accepted(count: int) -> None:
    assert assert_budget(count, decoder_id="M2", track=TRACK_MOVE) == count
    assert assert_budget(count, decoder_id="P4",
                         track=TRACK_PROCESS) == count


def test_baseline_must_be_zero_parameter() -> None:
    assert assert_budget(0, decoder_id="M0", track=TRACK_MOVE) == 0
    assert assert_budget(0, decoder_id="P0", track=TRACK_PROCESS) == 0
    with pytest.raises(V543ParamBudgetError) as error:
        assert_budget(1, decoder_id="M0", track=TRACK_MOVE)
    assert error.value.code == "V543_REGISTRY_BASELINE_PARAMS"


def test_band_constants_match_the_plan() -> None:
    assert PARAM_TARGET_V543 == 36_000
    assert PARAM_BAND_MIN_V543 == 34_200
    assert PARAM_BAND_MAX_V543 == 37_800
    assert PARAM_HARD_CAP_V543 == 40_000
    assert param_band_tolerance() == 1_800
    report = assert_parameter_matching_report()
    # plan §6.4 :566–567: 57,619 + 36K + 36K = 129,619 / band max 133,219
    assert report["final_target_total"] == 129_619
    assert report["final_band_max_total"] == 133_219
    assert report["global_hard_cap"] == 150_000


def test_synthetic_candidates_land_in_the_band() -> None:
    """The WP4 stand-in candidates are really inside 34,200–37,800."""
    move = SynthMoveRefiner()
    process = SynthProcessRefiner()
    assert PARAM_BAND_MIN_V543 <= count_trainable_params(move) <= \
        PARAM_BAND_MAX_V543
    assert PARAM_BAND_MIN_V543 <= count_trainable_params(process) <= \
        PARAM_BAND_MAX_V543
    registry = DecoderRegistry()
    registry.register(synth_move_spec())
    registry.register(synth_process_spec())
    built = registry.build("M1", track=TRACK_MOVE)
    assert count_trainable_params(built) == count_trainable_params(move)


def test_registration_measures_and_refuses_a_bad_factory() -> None:
    class TooBig(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(PARAM_HARD_CAP_V543 + 1))

        def forward(self, ctx):  # noqa: ANN001 - protocol shape
            return torch.zeros_like(ctx.u0)

    registry = DecoderRegistry()
    with pytest.raises(V543ParamBudgetError) as error:
        registry.register(DecoderSpec(
            decoder_id="M1", track=TRACK_MOVE, model_version="too_big",
            factory=TooBig))
    assert error.value.code == "V543_REGISTRY_PARAMS_CAP"
    assert registry.ids() == ()


# ---------------------------------------------------------------------------
# M0 / P0 behaviour
# ---------------------------------------------------------------------------
def _move_context(n_keys: int = 3) -> MoveContextV1:
    return MoveContextV1(
        graph_s=torch.zeros(2, 64), key_s=torch.zeros(n_keys, 48),
        key_v=torch.zeros(n_keys, 8, 3), u0=torch.ones(n_keys, 3),
        key_edge_index=torch.zeros(2, 0, dtype=torch.long),
        key_edge_attr=torch.zeros(0, 8),
        key_graph=torch.zeros(n_keys, dtype=torch.long),
        key_ptr=torch.tensor([0, n_keys], dtype=torch.long))


def _process_context(n_graphs: int = 2) -> ProcessContextV1:
    n_keys = 2 * n_graphs
    return ProcessContextV1(
        graph_s=torch.zeros(n_graphs, 64), key_s=torch.zeros(n_keys, 48),
        pair_s=torch.zeros(0, 48),
        intent_pair_index=torch.zeros(2, 0, dtype=torch.long),
        key_graph=torch.repeat_interleave(
            torch.arange(n_graphs, dtype=torch.long), 2),
        pair_graph=torch.zeros(0, dtype=torch.long),
        key_ptr=torch.arange(0, 2 * (n_graphs + 1), 2, dtype=torch.long),
        pair_ptr=torch.zeros(n_graphs + 1, dtype=torch.long),
        token_index=torch.zeros(3 * n_graphs, dtype=torch.long),
        token_type=torch.zeros(3 * n_graphs, dtype=torch.int8),
        token_graph=torch.repeat_interleave(
            torch.arange(n_graphs, dtype=torch.long), 3),
        token_ptr=torch.arange(0, 3 * (n_graphs + 1), 3, dtype=torch.long),
        q0_raw=torch.ones(n_graphs), endpoint0_logit=torch.zeros(n_graphs))


def test_m0_returns_a_zero_residual() -> None:
    decoder = build_decoder("M0", track=TRACK_MOVE)
    ctx = _move_context()
    delta = decoder(ctx)
    assert tuple(delta.shape) == (3, 3)
    assert float(delta.abs().sum()) == 0.0
    assert decoder.n_trainable_params == 0
    assert count_trainable_params(decoder) == 0
    with pytest.raises(V543DecoderProtocolError) as error:
        decoder(_process_context())
    assert error.value.code == "V543_DECODER_CONTEXT_TYPE"


def test_p0_returns_a_zero_residual() -> None:
    decoder = build_decoder("P0", track=TRACK_PROCESS)
    ctx = _process_context()
    residual = decoder(ctx)
    assert isinstance(residual, ProcessResidual)
    assert tuple(residual.delta_q_raw.shape) == (2,)
    assert tuple(residual.delta_e.shape) == (2,)
    assert float(residual.delta_q_raw.abs().sum()) == 0.0
    assert float(residual.delta_e.abs().sum()) == 0.0
    assert count_trainable_params(decoder) == 0
    with pytest.raises(V543DecoderProtocolError) as error:
        decoder(_move_context())
    assert error.value.code == "V543_DECODER_CONTEXT_TYPE"


def test_baselines_are_the_declared_ids() -> None:
    assert BASELINE_DECODER_BY_TRACK == {TRACK_MOVE: "M0",
                                        TRACK_PROCESS: "P0"}
    assert baseline_decoder_id(TRACK_MOVE) == "M0"
    assert baseline_decoder_id(TRACK_PROCESS) == "P0"
    assert isinstance(build_baseline(TRACK_MOVE), ZeroMoveDecoder)
    assert isinstance(build_baseline(TRACK_PROCESS), ZeroProcessDecoder)
    # a baseline has no parameter at all, so its state_dict is empty
    assert build_baseline(TRACK_MOVE).state_dict() == {}


# ---------------------------------------------------------------------------
# §6.1 prohibitions (data / loss / checkpoint selection)
# ---------------------------------------------------------------------------
def test_forward_with_two_arguments_is_refused() -> None:
    class Grabbing(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def forward(self, ctx, batch):  # noqa: ANN001 - deliberate breach
            return torch.zeros_like(ctx.u0)

    with pytest.raises(V543DecoderProtocolError) as error:
        assert_decoder_protocol(Grabbing(), decoder_id="M1",
                                track=TRACK_MOVE)
    assert error.value.code == "V543_DECODER_SIGNATURE"


def test_varargs_forward_is_refused() -> None:
    class Variadic(torch.nn.Module):
        def forward(self, *args):  # noqa: ANN001 - deliberate breach
            return torch.zeros(1)

    with pytest.raises(V543DecoderProtocolError) as error:
        assert_decoder_protocol(Variadic(), decoder_id="M1",
                                track=TRACK_MOVE)
    assert error.value.code == "V543_DECODER_SIGNATURE"


@pytest.mark.parametrize("name", ["dataset", "labels", "labels_v52",
                                  "endpoint_labels", "loss", "criterion",
                                  "pos", "y_teacher", "pair_index"])
def test_decoder_may_not_hold_data_state(name: str) -> None:
    class Hoarder(torch.nn.Module):
        def forward(self, ctx):  # noqa: ANN001 - protocol shape
            return torch.zeros_like(ctx.u0)

    hoarder = Hoarder()
    object.__setattr__(hoarder, name, torch.zeros(2))
    with pytest.raises(V543DecoderProtocolError) as error:
        assert_decoder_protocol(hoarder, decoder_id="M1", track=TRACK_MOVE)
    assert error.value.code == "V543_DECODER_PRIVATE_STATE"
    assert name in error.value.detail


@pytest.mark.parametrize("name", ["compute_loss", "select_checkpoint",
                                  "load_data", "score_checkpoint"])
def test_decoder_may_not_implement_loss_or_selection(name: str) -> None:
    class Cheater(torch.nn.Module):
        def forward(self, ctx):  # noqa: ANN001 - protocol shape
            return torch.zeros_like(ctx.u0)

    cheater = Cheater()
    setattr(cheater, name, lambda *args, **kwargs: None)
    with pytest.raises(V543DecoderProtocolError) as error:
        assert_decoder_protocol(cheater, decoder_id="M1", track=TRACK_MOVE)
    assert error.value.code == "V543_DECODER_PRIVATE_METHOD"


def test_track_ownership_is_enforced() -> None:
    with pytest.raises(V543RegistryError) as error:
        assert_decoder_protocol(ZeroMoveDecoder(), decoder_id="M1",
                                track=TRACK_PROCESS)
    assert error.value.code == "V543_REGISTRY_TRACK"
    with pytest.raises(V543RegistryError) as error:
        assert_decoder_protocol(ZeroProcessDecoder(), decoder_id="P1",
                                track=TRACK_MOVE)
    assert error.value.code == "V543_REGISTRY_TRACK"
    # the spec's own track must agree with the id
    with pytest.raises(V543RegistryError) as error:
        DecoderSpec(decoder_id="M1", track=TRACK_PROCESS,
                    model_version="x", factory=lambda: None)
    assert error.value.code == "V543_REGISTRY_TRACK"


def test_duplicate_and_cross_track_build_are_refused() -> None:
    registry = default_registry()
    with pytest.raises(V543RegistryError) as error:
        registry.register(M0_SPEC)
    assert error.value.code == "V543_REGISTRY_DUPLICATE"
    with pytest.raises(V543RegistryError) as error:
        registry.build("M0", track=TRACK_PROCESS)
    assert error.value.code == "V543_REGISTRY_TRACK"
    with pytest.raises(V543RegistryError) as error:
        registry.register(DecoderSpec(decoder_id="M9", track=TRACK_MOVE,
                                      model_version="x", factory=lambda: None))
    assert error.value.code == "V543_REGISTRY_UNKNOWN"


def test_spec_validation_vocabulary() -> None:
    with pytest.raises(V543RegistryError) as error:
        DecoderSpec(decoder_id="M1", track=TRACK_MOVE, model_version="x",
                    factory=SynthMoveRefiner, equivariance="scalar")
    assert error.value.code == "V543_REGISTRY_EQUIVARIANCE"
    with pytest.raises(V543RegistryError) as error:
        DecoderSpec(decoder_id="M1", track=TRACK_MOVE, model_version="x",
                    factory=SynthMoveRefiner, status="winner")
    assert error.value.code == "V543_REGISTRY_STATUS"
    with pytest.raises(V543RegistryError) as error:
        DecoderSpec(decoder_id="M1", track=TRACK_MOVE, model_version="x",
                    factory=SynthMoveRefiner, nfe=0)
    assert error.value.code == "V543_REGISTRY_NFE"
    with pytest.raises(V543RegistryError) as error:
        DecoderSpec(decoder_id="M1", track=TRACK_MOVE, model_version="x",
                    factory=SynthMoveRefiner, contract_version="decoder_v0")
    assert error.value.code == "V543_REGISTRY_CONTRACT"


def test_describe_all_reports_the_boundary() -> None:
    payload = REGISTRY_V543.describe_all()
    assert payload["baselines"] == {"move": "M0", "process": "P0"}
    assert [entry["decoder_id"] for entry in payload["registered"]] == \
        ["M0", "P0"]
    assert payload["declared_ids"] == list(DECODER_IDS_V543)
    for entry in payload["registered"]:
        assert entry["n_trainable_params"] == 0
        assert entry["is_baseline"] is True
