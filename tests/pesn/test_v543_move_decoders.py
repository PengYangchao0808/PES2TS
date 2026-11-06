"""V5.4.3 WP5 — MOVE-track candidates M1/M2/M3 CPU certificate (plan §6.2
:521–536, §4.5 :241–243, §6.4 :557–571, §11.1 ``move/*`` :1097–1103, §11.5
:1229–1231 ``test_v543_rotation_equivariance`` / ``translation_invariance`` /
``permutation_equivariance``).

What is pinned here, per candidate (M1 Residual MLP, M2 GVP, M3 Vector-GNN):

* the exact measured parameter count (``sum(p.numel())`` at construction), the
  §6.4 band 34,200–37,800, the 40,000 hard cap and the distance to the plan's
  own estimate (§6.2 :526 "约 36.6K" for M1, :528 "约 34.5K" for M3; M2 has no
  estimate, so the 36,000 target is used);
* the §6.1 protocol (``forward(MoveContextV1) -> δu [K,3]``), the typed
  context refusal and the 13-field §6.1 registry record;
* the §4.5 zero-residual initialisation (``δu ≡ 0`` at construction);
* rotation equivariance / translation invariance measured through the REAL
  WP1 encoder + WP2 Bridge (rotate/translate the fixture ``pos`` and rebuild the
  context with the SAME encoder weights) — the §11.5 H0 gates;
* key-permutation equivariance and ``key_edge`` column-order insensitivity
  (the plan does not freeze the edge row order, §5.2 :288);
* the M3 ``Ek=0`` degenerate path (§5.2 :310);
* assembly integration with ``V543Model`` (``u = u0 + δu``) and the §7.2
  gradient isolation (encoder parameters keep ``.grad is None``);
* a 3-update WP4-trainer smoke run on synthetic data (loss finite, trainable).

Floating-point tolerances, and why
----------------------------------

* rotation/translation enter the decoder through the Bridge, whose arithmetic
  is fp32: measured drift is ~1e-7 of ``|δu|``, so ``atol = 2e-6`` (the
  encoder's own rotation error, measured 1.5e-8, is the floor).
* the ``key_edge`` shuffle test compares two SUMS over the same 24 edges in a
  different order.  Summation is commutative in exact arithmetic but not in
  fp32, so the tolerance is set from the summation-order bound
  (``n·eps·Σ|x|`` ≈ 24·1.2e-7·O(1) ≈ 3e-6); ``atol = 1e-5`` is used and the
  measured drift is reported in the test's message.  The point of the test is
  that the decoder does not depend on the COLUMN ORDER, not that fp32 addition
  is associative.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.pesn.contracts.key_atom_v3 import forward_view_v3  # noqa: E402
from pes2ts_core.pesn.models.v543 import LatentBridge  # noqa: E402
from pes2ts_core.pesn.models.v543.assembly import (  # noqa: E402
    assert_packet_matches_context,
    unpermute_to_key_rows,
)
from pes2ts_core.pesn.models.v543.contracts import (  # noqa: E402
    DECODER_CONTEXT_SCHEMA_V543,
    MoveContextV1,
    V543ParamBudgetError,
)
from pes2ts_core.pesn.models.v543.move import gvp, residual_mlp, vector_gnn  # noqa: E402
from pes2ts_core.pesn.models.v543.move.gvp import (  # noqa: E402
    GVPRefinerMoveDecoder,
    M2_N_TRAINABLE_PARAMS,
)
from pes2ts_core.pesn.models.v543.move.residual_mlp import (  # noqa: E402
    M1_N_TRAINABLE_PARAMS,
    ResidualMLPMoveDecoder,
)
from pes2ts_core.pesn.models.v543.move.vector_gnn import (  # noqa: E402
    M3_N_TRAINABLE_PARAMS,
    VectorGNNMoveDecoder,
)
from pes2ts_core.pesn.models.v543.registry import (  # noqa: E402
    PARAM_BAND_MAX_V543,
    PARAM_BAND_MIN_V543,
    PARAM_HARD_CAP_V543,
    PARAM_TARGET_V543,
    REGISTRY_RECORD_FIELDS,
    REGISTRY_V543,
    TRACK_MOVE,
    DecoderRegistry,
    MoveDecoderProtocol,
    V543DecoderProtocolError,
    assert_decoder_protocol,
    count_trainable_params,
)
from pes2ts_core.pesn.offline.training.v543_train import (  # noqa: E402
    V543TrainerConfig,
    train_run_v543,
)
from tests.pesn._v543_fixtures_wp4 import (  # noqa: E402
    batch_with_status,
    make_encoder,
    make_model,
    synthetic_tables,
)

#: ``(class, module, decoder_id, plan estimate, exact measured)``.
CANDIDATES: tuple[tuple[type, object, str, int, int], ...] = (
    (ResidualMLPMoveDecoder, residual_mlp, "M1", 36_600, M1_N_TRAINABLE_PARAMS),
    (GVPRefinerMoveDecoder, gvp, "M2", PARAM_TARGET_V543, M2_N_TRAINABLE_PARAMS),
    (VectorGNNMoveDecoder, vector_gnn, "M3", 34_500, M3_N_TRAINABLE_PARAMS),
)

#: ``(decoder_id, spec-factory, class)`` for the registry tests.
SPECS: tuple[tuple[str, object, type], ...] = (
    ("M1", residual_mlp.m1_spec, ResidualMLPMoveDecoder),
    ("M2", gvp.m2_spec, GVPRefinerMoveDecoder),
    ("M3", vector_gnn.m3_spec, VectorGNNMoveDecoder),
)


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
def _rotation_matrix() -> torch.Tensor:
    """A fixed proper rotation (axis-angle, fp64 built then cast to fp32)."""
    axis = torch.tensor([0.3, -0.7, 0.5], dtype=torch.float64)
    axis = axis / axis.norm()
    angle = torch.tensor(0.7, dtype=torch.float64)
    skew = torch.zeros(3, 3, dtype=torch.float64)
    skew[0, 1], skew[0, 2], skew[1, 2] = -axis[2], axis[1], -axis[0]
    skew[1, 0], skew[2, 0], skew[2, 1] = axis[2], -axis[1], axis[0]
    rotation = (torch.eye(3, dtype=torch.float64)
                + torch.sin(angle) * skew
                + (1 - torch.cos(angle)) * (skew @ skew))
    return rotation.to(torch.float32)


TRANSLATION = torch.tensor([0.31, -1.17, 2.03], dtype=torch.float32)


def _perturbed(decoder: torch.nn.Module, *, seed: int, scale: float = 0.02
               ) -> torch.nn.Module:
    """Deterministically perturb every parameter off the zero-init point.

    The §4.5 zero-residual initialisation makes ``δu ≡ 0`` and also kills the
    gradient of the trunk at the first update, so the equivariance /
    gradient-isolation / value checks are run on a perturbed decoder — the
    zero-init property itself is asserted separately and exactly.
    """
    generator = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        for parameter in decoder.parameters():
            parameter.add_(scale * torch.randn(
                parameter.shape, generator=generator))
    return decoder


@pytest.fixture(scope="module")
def real_contexts() -> dict[str, object]:
    """One shared encoder → base / rotated / translated real contexts.

    The encoder is created ONCE: the rotation/translation statement is about
    the same weights (``context_of`` in the WP4 fixtures re-instantiates a
    random encoder per call, which would compare two different models).
    """
    batch = batch_with_status(("ok",) * 4, jitter_seed=17)
    encoder = make_encoder()
    bridge = LatentBridge()

    def context_of(data: dict) -> object:
        view = forward_view_v3(dict(data))
        packet = encoder(view)
        return bridge.build_context(packet, view)

    rotation = _rotation_matrix()
    rotated = dict(batch)
    rotated["pos"] = batch["pos"] @ rotation.T
    translated = dict(batch)
    translated["pos"] = batch["pos"] + TRANSLATION
    return {
        "batch": batch,
        "context": context_of(batch),
        "rotated": context_of(rotated),
        "translated": context_of(translated),
        "rotation": rotation,
    }


def _decode(decoder: torch.nn.Module, ctx: MoveContextV1) -> torch.Tensor:
    with torch.no_grad():
        return decoder(ctx)


def _move_view(contexts: dict[str, object], name: str = "context"
               ) -> MoveContextV1:
    return contexts[name].move_view()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# §6.4 / §6.2 parameters
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cls,module,decoder_id,estimate,expected", CANDIDATES)
def test_measured_parameter_count(cls: type, module: object, decoder_id: str,
                                  estimate: int, expected: int) -> None:
    """``sum(p.numel())`` at construction: exact, in band, near the estimate."""
    decoder = cls()
    measured = count_trainable_params(decoder)
    assert measured == expected, (decoder_id, measured, expected)
    assert decoder.n_trainable_params == expected
    assert PARAM_BAND_MIN_V543 <= measured <= PARAM_BAND_MAX_V543, measured
    assert measured <= PARAM_HARD_CAP_V543
    assert abs(measured - estimate) <= 1_800, (
        decoder_id, measured, estimate,
        "deviates from the plan estimate by more than the ±5 % band")
    # the module-level constant is the same number (record source of truth)
    assert getattr(module, f"{decoder_id}_N_TRAINABLE_PARAMS").__int__() == \
        measured


def test_out_of_band_construction_is_a_typed_refusal() -> None:
    """The §6.4 assertion runs AT CONSTRUCTION, and it refuses the band edges.

    The frozen widths are asserted against the module constant inside
    ``__init__`` (``V543_M1/M2/M3_PARAMS``) and against the §6.4 band by
    :func:`~...registry.assert_budget`; both are typed refusals, never a
    warning.  Here the band refusal itself is exercised on an in-band /
    out-of-band pair so the wiring is pinned.
    """
    from pes2ts_core.pesn.models.v543.registry import assert_budget

    for _, _, decoder_id, _, count in CANDIDATES:
        assert assert_budget(count, decoder_id=decoder_id,
                             track=TRACK_MOVE) == count
    with pytest.raises(V543ParamBudgetError) as error:
        assert_budget(M1_N_TRAINABLE_PARAMS + 1_000, decoder_id="M1",
                      track=TRACK_MOVE)
    # +1,000 from 37,272 is still inside the band (≤ 37,800)? No: 38,272 > max.
    assert error.value.code == "V543_REGISTRY_PARAMS_BAND"


def test_structure_constants_match_the_plan() -> None:
    """§6.2 :526/:527/:528 — the declared widths and depths."""
    assert residual_mlp.M1_HIDDEN_DIM == 136
    assert residual_mlp.M1_COEFF_DIM == 8
    assert gvp.M2_N_BLOCKS == 2
    assert gvp.M2_SCALAR_DIM == 96
    assert gvp.M2_VECTOR_CHANNELS == 8
    assert vector_gnn.M3_N_LAYERS == 2
    assert vector_gnn.M3_HIDDEN_DIM == 56
    assert vector_gnn.M3_EDGE_DIM == 8


# ---------------------------------------------------------------------------
# §6.1 protocol + typed refusals
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cls,module,decoder_id,estimate,expected", CANDIDATES)
def test_protocol_and_declaration(cls: type, module: object, decoder_id: str,
                                  estimate: int, expected: int) -> None:
    decoder = cls()
    names = list(inspect.signature(type(decoder).forward).parameters)
    assert names == ["self", "ctx"], names
    assert isinstance(decoder, MoveDecoderProtocol)
    assert assert_decoder_protocol(
        decoder, decoder_id=decoder_id, track=TRACK_MOVE) is decoder
    assert decoder.decoder_id == decoder_id
    assert decoder.track == TRACK_MOVE
    assert isinstance(decoder.model_version, str) and decoder.model_version
    # no data / loss / checkpoint-selection surface (§6.1 :519)
    for forbidden in ("compute_loss", "select_checkpoint", "load_data",
                      "dataset", "labels", "pos"):
        assert not hasattr(decoder, forbidden), forbidden


@pytest.mark.parametrize("cls,module,decoder_id,estimate,expected", CANDIDATES)
def test_wrong_track_view_is_refused(cls: type, module: object,
                                     decoder_id: str, estimate: int,
                                     expected: int,
                                     real_contexts: dict) -> None:
    decoder = cls()
    process_view = real_contexts["context"].process_view()
    with pytest.raises(V543DecoderProtocolError) as error:
        decoder(process_view)
    assert error.value.code == "V543_DECODER_CONTEXT_TYPE"
    assert decoder_id in error.value.detail
    with pytest.raises(V543DecoderProtocolError) as error:
        decoder({"key_s": torch.zeros(1, 48)})
    assert error.value.code == "V543_DECODER_CONTEXT_TYPE"


# ---------------------------------------------------------------------------
# §6.1 registry record (13 fields) and registration
# ---------------------------------------------------------------------------
def test_move_candidates_register_with_the_plan_record() -> None:
    registry = DecoderRegistry()
    for decoder_id, factory, _cls in SPECS:
        registry.register(factory())
    assert registry.ids() == ("M1", "M2", "M3")
    assert registry.ids(track=TRACK_MOVE) == ("M1", "M2", "M3")
    for decoder_id, _factory, cls in SPECS:
        spec = registry.spec(decoder_id)
        assert type(spec.factory()) is cls
        described = registry.describe(decoder_id)
        record = spec.record(described["n_trainable_params"])
        assert tuple(record) == REGISTRY_RECORD_FIELDS
        assert set(described) >= set(REGISTRY_RECORD_FIELDS)
        assert record["decoder_id"] == decoder_id
        assert record["track"] == TRACK_MOVE
        assert record["contract_version"] == DECODER_CONTEXT_SCHEMA_V543
        assert record["n_trainable_params"] == count_trainable_params(cls())
        assert record["equivariance"] == "polar"
        assert record["stochastic"] is False
        assert record["nfe"] == 1
        assert record["zero_init"] is True
        assert record["status"] == "proposed"
        assert described["n_trainable_params_source"] == "measured_at_register"
        assert described["is_baseline"] is False
        # build() re-measures and the declared count matches (§6.4 :569)
        built = registry.build(decoder_id, track=TRACK_MOVE)
        assert count_trainable_params(built) == record["n_trainable_params"]


def test_importing_the_candidates_does_not_touch_the_wp4_registry() -> None:
    """The WP5 modules never register into the process-wide registry.

    The ``models/v543/__init__.py`` surface and the global registry are owned by
    the integrating agent (WP5 hard constraint: no shared file is modified), so
    importing ``move.*`` must leave ``REGISTRY_V543`` at the WP4 boundary.
    """
    assert REGISTRY_V543.ids() == ("M0", "P0")
    for spec in (residual_mlp.M1_SPEC, gvp.M2_SPEC, vector_gnn.M3_SPEC):
        assert spec.decoder_id not in REGISTRY_V543.ids()


# ---------------------------------------------------------------------------
# §4.5 zero residual
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cls,module,decoder_id,estimate,expected", CANDIDATES)
def test_zero_residual_initialisation(cls: type, module: object,
                                      decoder_id: str, estimate: int,
                                      expected: int, real_contexts: dict
                                      ) -> None:
    """§4.5 :241–243 — the freshly built decoder predicts ``δu ≡ 0`` exactly."""
    decoder = cls()
    view = _move_view(real_contexts)
    delta = _decode(decoder, view)
    assert tuple(delta.shape) == (int(view.key_s.shape[0]), 3)
    assert delta.dtype == torch.float32
    assert torch.equal(delta, torch.zeros_like(delta)), decoder_id


@pytest.mark.parametrize("cls,module,decoder_id,estimate,expected", CANDIDATES)
def test_zero_init_locations(cls: type, module: object, decoder_id: str,
                             estimate: int, expected: int) -> None:
    """§4.5 :241–243 — WHICH tensor is zero at construction."""
    decoder = cls()
    assert torch.equal(decoder.coeff.weight,
                       torch.zeros_like(decoder.coeff.weight)), decoder_id
    assert torch.equal(decoder.coeff.bias,
                       torch.zeros_like(decoder.coeff.bias)), decoder_id
    if decoder_id == "M2":
        # one zero factor is enough for δu = 0; a zero GATE as well would make
        # the product stationary in BOTH factors (dead gradient, see gvp.py).
        assert torch.equal(decoder.out_gate,
                           torch.ones_like(decoder.out_gate))


@pytest.mark.parametrize("cls,module,decoder_id,estimate,expected", CANDIDATES)
def test_construction_is_seeded_and_rng_neutral(cls: type, module: object,
                                                decoder_id: str,
                                                estimate: int, expected: int
                                                ) -> None:
    """WP1 ``fork_rng`` discipline: deterministic + caller RNG preserved."""
    before = torch.get_rng_state()
    first = cls()
    after = torch.get_rng_state()
    assert torch.equal(before, after), f"{decoder_id} perturbed the global RNG"
    second = cls(seed=getattr(module, f"{decoder_id}_INIT_SEED_V543"))
    assert first.init_seed == getattr(
        module, f"{decoder_id}_INIT_SEED_V543")
    for (name, a), (name_b, b) in zip(first.state_dict().items(),
                                      second.state_dict().items()):
        assert name == name_b
        assert torch.equal(a, b), f"{decoder_id}:{name} is not seed-stable"


# ---------------------------------------------------------------------------
# E(3): rotation equivariance / translation invariance (§11.5 :1229–1230)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cls,module,decoder_id,estimate,expected", CANDIDATES)
def test_rotation_equivariance(cls: type, module: object, decoder_id: str,
                               estimate: int, expected: int,
                               real_contexts: dict) -> None:
    """Rotating the fixture geometry rotates ``δu`` (through the real Bridge)."""
    rotation = real_contexts["rotation"]
    decoder = _perturbed(cls(), seed=11)
    base = _decode(decoder, _move_view(real_contexts, "context"))
    rotated = _decode(decoder, _move_view(real_contexts, "rotated"))
    assert float(base.abs().max()) > 0.0, "the perturbed decoder is silent"
    expected_delta = base @ rotation.T
    error = float((rotated - expected_delta).abs().max())
    assert error <= 2e-6, (decoder_id, error, float(base.abs().max()))
    # and the polar inputs really do rotate (the test is not vacuous).  The
    # key_v bound is 1e-5: channel 0 is u0/a_ref, so the encoder's own fp32
    # rotation error (~1.5e-8) is amplified by 1/0.08 = 12.5, and the remaining
    # channels are unit-vector differences of the same magnitude.
    ctx = real_contexts["context"]
    rot = real_contexts["rotated"]
    assert float((rot.u0 - ctx.u0 @ rotation.T).detach().abs().max()) <= 1e-6
    key_v_error = float((rot.key_v - torch.einsum(
        "kcd,ed->kce", ctx.key_v, rotation)).detach().abs().max())
    assert key_v_error <= 1e-5, key_v_error


@pytest.mark.parametrize("cls,module,decoder_id,estimate,expected", CANDIDATES)
def test_translation_invariance(cls: type, module: object, decoder_id: str,
                                estimate: int, expected: int,
                                real_contexts: dict) -> None:
    """Translating the fixture geometry leaves ``δu`` unchanged (§5.2 平移)."""
    decoder = _perturbed(cls(), seed=11)
    base = _decode(decoder, _move_view(real_contexts, "context"))
    translated = _decode(decoder, _move_view(real_contexts, "translated"))
    error = float((translated - base).abs().max())
    assert error <= 2e-6, (decoder_id, error)
    ctx = real_contexts["context"]
    moved = real_contexts["translated"]
    assert float((moved.u0 - ctx.u0).detach().abs().max()) <= 1e-6
    assert float((moved.key_v - ctx.key_v).detach().abs().max()) <= 1e-5


# ---------------------------------------------------------------------------
# permutation equivariance / edge-column order (§11.5 :1231, §5.2 :288/:310)
# ---------------------------------------------------------------------------
def _within_graph_permutation(view: MoveContextV1, *, seed: int) -> torch.Tensor:
    """A permutation that keeps every graph's keys inside its own ``key_ptr``.

    Keeping the permutation graph-local keeps ``key_ptr``/``key_graph``
    consistent with the §5.2 :307 layout, so the test permutes the key ROWS the
    way a re-indexed dataset would, not an arbitrary global shuffle.
    """
    generator = torch.Generator().manual_seed(int(seed))
    n_keys = int(view.key_s.shape[0])
    order = torch.arange(n_keys)
    key_ptr = view.key_ptr.tolist()
    for graph in range(len(key_ptr) - 1):
        low, high = int(key_ptr[graph]), int(key_ptr[graph + 1])
        if high - low <= 1:
            continue
        order[low:high] = torch.randperm(high - low, generator=generator) + low
    return order


def _permuted(view: MoveContextV1, order: torch.Tensor) -> MoveContextV1:
    """Re-index every key row (and every edge endpoint) by ``order``."""
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(int(order.numel()))
    return MoveContextV1(
        graph_s=view.graph_s,
        key_s=view.key_s[order],
        key_v=view.key_v[order],
        u0=view.u0[order],
        key_edge_index=inverse[view.key_edge_index],
        key_edge_attr=view.key_edge_attr,
        key_graph=view.key_graph[order],
        key_ptr=view.key_ptr,
    )


@pytest.mark.parametrize("cls,module,decoder_id,estimate,expected", CANDIDATES)
def test_key_permutation_equivariance(cls: type, module: object,
                                      decoder_id: str, estimate: int,
                                      expected: int, real_contexts: dict
                                      ) -> None:
    """``δu`` rows follow the key rows (plan §5.2 :283/:285 原子置换等变)."""
    view = _move_view(real_contexts)
    order = _within_graph_permutation(view, seed=23)
    permuted = _permuted(view, order)
    decoder = _perturbed(cls(), seed=13)
    base = _decode(decoder, view)
    shuffled = _decode(decoder, permuted)
    error = float((shuffled - base[order]).abs().max())
    assert error <= 2e-6, (decoder_id, error, float(base.abs().max()))


@pytest.mark.parametrize("cls,module,decoder_id,estimate,expected", CANDIDATES)
def test_key_edge_column_order_is_irrelevant(cls: type, module: object,
                                             decoder_id: str, estimate: int,
                                             expected: int,
                                             real_contexts: dict) -> None:
    """The plan freezes no edge row order (§5.2 :288) → a shuffle must not move.

    ``atol = 1e-5`` is the fp32 summation-order bound for the 24 fixture edges
    (``n·eps·Σ|x|``), NOT a modelling tolerance: M1/M2 never read ``key_edge_*``
    so they are bit-identical, and M3 aggregates with ``index_add`` (sum), so
    only the addition ORDER changes.
    """
    view = _move_view(real_contexts)
    generator = torch.Generator().manual_seed(37)
    n_edges = int(view.key_edge_index.shape[1])
    order = torch.randperm(n_edges, generator=generator)
    shuffled = MoveContextV1(
        graph_s=view.graph_s,
        key_s=view.key_s,
        key_v=view.key_v,
        u0=view.u0,
        key_edge_index=view.key_edge_index[:, order],
        key_edge_attr=view.key_edge_attr[order],
        key_graph=view.key_graph,
        key_ptr=view.key_ptr,
    )
    decoder = _perturbed(cls(), seed=17)
    base = _decode(decoder, view)
    other = _decode(decoder, shuffled)
    error = float((other - base).abs().max())
    assert error <= 1e-5, (decoder_id, error, float(base.abs().max()))


def test_vector_gnn_degenerates_without_edges(real_contexts: dict) -> None:
    """§5.2 :310 "``Ek=0`` 允许；M3 必须能退化为无消息传递"."""
    view = _move_view(real_contexts)
    no_edges = MoveContextV1(
        graph_s=view.graph_s,
        key_s=view.key_s,
        key_v=view.key_v,
        u0=view.u0,
        key_edge_index=torch.zeros(2, 0, dtype=torch.long),
        key_edge_attr=torch.zeros(0, 8, dtype=torch.float32),
        key_graph=view.key_graph,
        key_ptr=view.key_ptr,
    )
    decoder = _perturbed(VectorGNNMoveDecoder(), seed=19)
    delta = _decode(decoder, no_edges)
    assert tuple(delta.shape) == (int(view.key_s.shape[0]), 3)
    assert bool(torch.isfinite(delta).all())
    # the isolated-key path is a pure per-key function of V (no messages): it
    # must differ from the message-passing result, otherwise the test is vacuous.
    with_edges = _decode(decoder, view)
    assert float((delta - with_edges).abs().max()) > 0.0
    zero_init = _decode(VectorGNNMoveDecoder(), no_edges)
    assert torch.equal(zero_init, torch.zeros_like(zero_init))


# ---------------------------------------------------------------------------
# assembly integration + §7.2 gradient isolation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cls,module,decoder_id,estimate,expected", CANDIDATES)
def test_assembly_forward_is_u0_plus_delta_u(cls: type, module: object,
                                             decoder_id: str, estimate: int,
                                             expected: int,
                                             real_contexts: dict) -> None:
    """``V543Model`` MOVE assembly: ``u = u0 + δu`` in ``key_index`` row order."""
    decoder = _perturbed(cls(), seed=29)
    model = make_model(track=TRACK_MOVE, decoder=decoder,
                       decoder_id=decoder_id)
    view = forward_view_v3(dict(real_contexts["batch"]))
    out = model.forward_debug(view)
    assert set(("u", "z", "q", "endpoint_logit", "order_score")) <= set(out)
    n_keys = int(real_contexts["context"].n_keys)
    assert tuple(out["u"].shape) == (n_keys, 3)
    assert bool(torch.isfinite(out["u"]).all())

    packet = model.assembly.encoder(view)
    ctx = model.assembly.bridge.build_context(packet, view)
    delta = _decode(decoder, ctx.move_view())
    assert float((out["delta_u"] - delta).detach().abs().max()) <= 1e-7
    permutation = assert_packet_matches_context(packet, ctx)
    expected_u = unpermute_to_key_rows(ctx.u0 + delta, permutation)
    assert float((out["u"] - expected_u).detach().abs().max()) <= 1e-7
    assert int(model.encoder_forward_count) >= 1


@pytest.mark.parametrize("cls,module,decoder_id,estimate,expected", CANDIDATES)
def test_gradient_only_reaches_the_decoder(cls: type, module: object,
                                           decoder_id: str, estimate: int,
                                           expected: int,
                                           real_contexts: dict) -> None:
    """plan §7.2 :713–726 — encoder parameters keep ``.grad is None``."""
    decoder = _perturbed(cls(), seed=31)
    model = make_model(track=TRACK_MOVE, decoder=decoder,
                       decoder_id=decoder_id)
    view = forward_view_v3(dict(real_contexts["batch"]))
    model.zero_grad()
    out = model.forward_debug(view)
    loss = out["delta_u"].pow(2).sum()
    loss.backward()
    for name, parameter in model.assembly.encoder.named_parameters():
        assert parameter.grad is None, f"{decoder_id} leaked grad to {name}"
    assert all(parameter.grad is None
               for parameter in model.assembly.inactive.decoder.parameters())
    report = model.assembly.assert_gradient_isolation(require_active_grad=True)
    assert report["encoder_params_with_grad"] == 0
    assert report["active_params_with_grad"] > 0
    assert math.isfinite(float(loss.detach()))


# ---------------------------------------------------------------------------
# WP4-trainer smoke (3 updates per candidate, synthetic data)
# ---------------------------------------------------------------------------
def _smoke_source():
    from tests.pesn._v543_fixtures_wp4 import SynthSource

    batch = batch_with_status(("ok",) * 4, jitter_seed=17)
    tables = synthetic_tables(batch, endpoint_labels=(1, 0, 1, 0))
    return SynthSource(tables, label_status=("ok",) * 4)


@pytest.mark.parametrize("cls,module,decoder_id,estimate,expected", CANDIDATES)
def test_trainer_smoke_three_updates(cls: type, module: object,
                                     decoder_id: str, estimate: int,
                                     expected: int, tmp_path: Path) -> None:
    """The WP4 trainer drives each candidate for 3 finite-loss updates."""
    decoder = cls()
    before = {name: value.detach().clone()
              for name, value in decoder.state_dict().items()}
    model = make_model(track=TRACK_MOVE, decoder=decoder,
                       decoder_id=decoder_id)
    source = _smoke_source()
    config = V543TrainerConfig(track="move", decoder_id=decoder_id, seed=17,
                               stop_after=3)
    out_dir = tmp_path / decoder_id
    metrics = train_run_v543(
        model, config, source, labels_v52=source.tables.labels,
        endpoint_labels=source.tables.endpoints, out_dir=out_dir)
    assert metrics["outcome"] == "done"
    assert int(metrics["updates"]) == 3
    assert metrics["best_val_loss"] is not None
    assert math.isfinite(float(metrics["best_val_loss"]))
    assert (out_dir / "metrics.json").is_file()
    moved = [name for name, value in decoder.state_dict().items()
             if not torch.equal(value.detach(), before[name])]
    assert moved, f"{decoder_id}: 3 updates changed no parameter"
