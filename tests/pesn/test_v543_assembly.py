"""V5.4.3 WP4 — assembly + model CPU certificate (plan §4.1 :182–188,
§4.2/§4.3 :190–227, §4.4 :229–235, §6.4 :557–571, §7.2 :713–726,
§11.2 ``assembly.py`` / ``model.py``).

Pinned here:

* the assembly freezes every encoder parameter and keeps the Bridge
  parameter-free (§4.1 :185 "0 trainable");
* the NON-active track is its zero-residual baseline and a non-baseline in that
  slot is a typed refusal (§4.3 :226);
* the §6.4 global budget (57,619 + active + inactive ≤ 150,000) is asserted;
* the §4.3 five-key external contract, ``order_score is q`` (same tensor), and
  the ``u = u0 + δu`` / ``q = tanh(q0_raw+δq)`` / ``e = e0+δe`` identity;
* the encoder runs EXACTLY once per forward view (§4.4 boundary 1);
* §7.2 :713–726 gradient isolation on BOTH tracks: after a real backward the
  encoder holds ``None`` grads (never a zero tensor), the inactive Refiner holds
  ``None`` grads, and the active Refiner holds real grads."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.pesn.contracts.key_atom_v3 import forward_view_v3  # noqa: E402
from pes2ts_core.pesn.models.v543.assembly import (  # noqa: E402
    V543AssemblyError,
    assert_packet_matches_context,
    assemble_baseline_v543,
    assemble_v543,
    bind_decoder,
    unpermute_to_key_rows,
)
from pes2ts_core.pesn.models.v543.contracts import (  # noqa: E402
    ENCODER_JOINT_PARAMS_V543,
    PRECISION_VERSION_V543,
)
from pes2ts_core.pesn.models.v543.model import (  # noqa: E402
    DEBUG_OUTPUT_KEYS,
    EXTERNAL_OUTPUT_KEYS,
    V543Model,
    V543ModelError,
    check_external_output,
    identity_outputs_v543,
)
from tests.pesn._v543_fixtures_wp4 import (  # noqa: E402
    SynthMoveRefiner,
    SynthProcessRefiner,
    batch_with_status,
    make_assembly,
    make_encoder,
    make_model,
)


@pytest.fixture(scope="module")
def batch() -> dict:
    return batch_with_status(("ok",) * 4, jitter_seed=3)


@pytest.fixture(scope="module")
def view(batch: dict) -> dict:
    return forward_view_v3(dict(batch))


# ---------------------------------------------------------------------------
# wiring / budgets
# ---------------------------------------------------------------------------
def test_baseline_assembly_freezes_and_counts() -> None:
    assembly = assemble_baseline_v543(make_encoder())
    report = assembly.param_report()
    assert report["encoder_joint"] == ENCODER_JOINT_PARAMS_V543 == 57_619
    assert report["bridge_trainable"] == 0
    assert report["inactive_params"] == 0
    assert report["total_model_params"] == 57_619
    assert report["active"]["decoder_id"] == "M0"
    assert report["inactive"]["decoder_id"] == "P0"
    assert all(not p.requires_grad for p in assembly.encoder.parameters())
    assembly.assert_encoder_frozen()
    assembly.assert_param_budget()


def test_encoder_not_frozen_is_refused() -> None:
    encoder = make_encoder()
    assembly = assemble_baseline_v543(encoder)
    next(iter(encoder.parameters())).requires_grad_(True)
    with pytest.raises(V543AssemblyError) as error:
        assembly.assert_encoder_frozen()
    assert error.value.code == "V543_ENCODER_NOT_FROZEN"


def test_active_candidate_is_trainable_inactive_is_not() -> None:
    refiner = SynthMoveRefiner()
    assembly = make_assembly(track="move", decoder=refiner, decoder_id="M1")
    report = assembly.param_report()
    assert report["active"]["n_trainable_params"] == 35_340
    assert report["active"]["role"] == "active"
    assert report["inactive"]["decoder_id"] == "P0"
    assert report["inactive"]["role"] == "baseline"
    assert report["total_model_params"] == 57_619 + 35_340
    assert all(p.requires_grad for p in refiner.parameters())
    # trunk.weight + trunk.bias + coeff.weight
    assert len(assembly.trainable_parameters()) == 3
    assembly.assert_param_budget()


def test_non_baseline_inactive_track_is_refused() -> None:
    with pytest.raises(V543AssemblyError) as error:
        assemble_v543(
            make_encoder(), active_track="process",
            move_refiner=SynthMoveRefiner(), move_decoder_id="M1")
    assert error.value.code == "V543_ASSEMBLY_INACTIVE"
    assert "zero-residual" in error.value.detail


def test_global_budget_is_asserted() -> None:
    """57,619 + active + inactive must stay under 150,000 (plan §6.4 :568).

    With the band (≤37,800) and a zero-parameter inactive track the cap cannot
    be reached by a legal assembly, so the check is exercised on a report that
    is over the line: the refusal must be the GLOBAL budget, not a band code.
    """
    assembly = make_assembly(track="move", decoder=SynthMoveRefiner(),
                             decoder_id="M1")
    report = dict(assembly.param_report())
    report["total_model_params"] = 150_001
    original = assembly.param_report

    assembly.param_report = lambda: dict(report)      # type: ignore[assignment]
    try:
        with pytest.raises(V543AssemblyError) as error:
            assembly.assert_param_budget()
        assert error.value.code == "V543_GLOBAL_BUDGET"
    finally:
        assembly.param_report = original              # type: ignore[assignment]


def test_bind_decoder_reports_the_measured_count() -> None:
    binding = bind_decoder(SynthProcessRefiner(), decoder_id="P1",
                           track="process")
    assert binding.n_trainable_params == 36_482
    assert binding.is_baseline is False
    assert binding.describe()["role"] == "active"

    class Liar(torch.nn.Module):
        n_trainable_params = 12

        def forward(self, ctx):  # noqa: ANN001 - protocol shape
            return torch.zeros_like(ctx.u0)

    with pytest.raises(V543AssemblyError) as error:
        bind_decoder(Liar(), decoder_id="M1", track="move")
    assert error.value.code == "V543_ASSEMBLY_PARAM_REPORT"


def test_track_switch_refreezes_the_boundary() -> None:
    assembly = make_assembly(track="move", decoder=SynthMoveRefiner(),
                             decoder_id="M1")
    assembly.set_active_track("process")
    assert assembly.active.decoder_id == "P0"
    assert assembly.inactive.decoder_id == "M1"
    assert not any(p.requires_grad
                   for p in assembly.move.decoder.parameters())
    with pytest.raises(V543AssemblyError) as error:
        assembly.set_active_track("both")
    assert error.value.code == "V543_ASSEMBLY_TRACK"


# ---------------------------------------------------------------------------
# the unique forward flow (§4.2 / §4.3)
# ---------------------------------------------------------------------------
def test_forward_returns_exactly_the_five_keys(view: dict) -> None:
    model = make_model(track="move")
    out = model(view)
    assert tuple(sorted(out)) == tuple(sorted(EXTERNAL_OUTPUT_KEYS))
    check_external_output(out)
    # plan §4.3 :223 — "``order_score IS q``": the SAME tensor object
    assert out["order_score"] is out["q"]
    assert out["u"].dtype == torch.float32
    assert PRECISION_VERSION_V543 == "fp32_v1"


def test_zero_residual_identities(view: dict) -> None:
    """M0/P0 reproduce V542 A0 exactly (plan §5.8 :464–471)."""
    model = make_model(track="move")
    out = model.forward_debug(view)
    identity = identity_outputs_v543(out)
    assert identity["order_score_is_q"] is True
    assert identity["q_equals_tanh_q0_raw"] is True
    assert identity["u_equals_u0"] is True
    assert identity["endpoint_equals_endpoint0"] is True
    # "零残差 V543(u,q,e) == V542(u,q,e)": u/p_end are the encoder's own values
    assert torch.equal(out["q"],
                       torch.tanh(out["q0_raw"] + out["delta_q_raw"]))
    assert float(out["delta_u"].abs().sum()) == 0.0
    assert float(out["delta_e"].abs().sum()) == 0.0


def test_decoder_residual_is_added(view: dict) -> None:
    model = make_model(track="move", decoder=SynthMoveRefiner(),
                       decoder_id="M1")
    out = model.forward_debug(view)
    assert not torch.equal(out["delta_u"], torch.zeros_like(out["delta_u"]))
    # u = u0 + δu in key_index row order (§4.3 :222)
    expected = unpermute_to_key_rows(out["u0"] + out["delta_u"],
                                     out["key_permutation"])
    assert torch.equal(out["u"], expected)


def test_encoder_runs_exactly_once_per_view(view: dict) -> None:
    model = make_model(track="move")
    assert model.encoder_forward_count == 0
    model(view)
    assert model.encoder_forward_count == 1
    model(view)
    assert model.encoder_forward_count == 2
    encoder = model.assembly.encoder
    for label, fires in encoder.last_hook_fires.items():
        assert fires == 1, label


def test_forward_refuses_supervision_at_the_boundary(batch: dict) -> None:
    model = make_model(track="move")
    with pytest.raises(Exception) as error:
        model(dict(batch))          # a raw batch, supervision included
    assert "SUPERVISION" in str(error.value).upper()


def test_key_permutation_is_proven_not_assumed(batch: dict, view: dict) -> None:
    encoder = make_encoder()
    packet = encoder(view)
    from pes2ts_core.pesn.models.v543 import LatentBridge

    context = LatentBridge().build_context(packet, view)
    permutation = assert_packet_matches_context(packet, context)
    assert torch.equal(packet.u0[permutation], context.u0)


def test_forward_context_runs_only_the_decoder(view: dict) -> None:
    model = make_model(track="move", decoder=SynthMoveRefiner(),
                       decoder_id="M1")
    encoder = model.assembly.encoder
    packet = encoder(view)
    context = model.assembly.bridge.build_context(packet, view)
    calls = model.encoder_forward_count
    out = model.forward_context(context)
    assert model.encoder_forward_count == calls      # no extra encoder call
    assert tuple(sorted(out)) == tuple(sorted(EXTERNAL_OUTPUT_KEYS))
    with pytest.raises(V543ModelError) as error:
        model.forward_context("not a context")
    assert error.value.code == "V543_MODEL_CONTEXT"


def test_model_holds_no_parameter_of_its_own() -> None:
    model = make_model(track="move")
    assert list(model.named_parameters()) == []
    described = model.describe()
    assert described["external_output_keys"] == list(EXTERNAL_OUTPUT_KEYS)
    assert described["debug_output_keys"] == list(DEBUG_OUTPUT_KEYS)
    assert described["encoder_forward_calls_per_view"] == 1


def test_debug_keys_are_the_plan_diagnostics(view: dict) -> None:
    model = make_model(track="move")
    out = model.forward_debug(view)
    for key in ("u0", "delta_u", "q0_raw", "delta_q_raw", "endpoint0_logit",
                "delta_e"):
        assert key in out, key
    # the EXTERNAL contract must not have grown the diagnostics (§4.3 :225)
    external = model(view)
    for key in ("u0", "delta_u", "q0_raw", "delta_q_raw"):
        assert key not in external


# ---------------------------------------------------------------------------
# §7.2 gradient isolation (both tracks)
# ---------------------------------------------------------------------------
def _backward(model: V543Model, view: dict) -> None:
    model.zero_grad()
    out = model(view)
    loss = out["u"].pow(2).mean() + out["q"].pow(2).mean() \
        + out["endpoint_logit"].pow(2).mean()
    loss.backward()


def test_gradient_isolation_move_track(view: dict) -> None:
    refiner = SynthMoveRefiner()
    model = make_model(track="move", decoder=refiner, decoder_id="M1")
    _backward(model, view)
    for name, parameter in model.assembly.encoder.named_parameters():
        assert parameter.grad is None, name
    assert all(p.grad is None
               for p in model.assembly.process.decoder.parameters())
    assert any(p.grad is not None for p in refiner.parameters())
    report = model.assert_gradient_isolation(require_active_grad=True)
    assert report["encoder_params_with_grad"] == 0
    assert report["active_params_with_grad"] >= 1


def test_gradient_isolation_process_track(view: dict) -> None:
    refiner = SynthProcessRefiner()
    model = make_model(track="process", decoder=refiner, decoder_id="P1")
    _backward(model, view)
    for name, parameter in model.assembly.encoder.named_parameters():
        assert parameter.grad is None, name
    assert all(p.grad is None
               for p in model.assembly.move.decoder.parameters())
    assert any(p.grad is not None for p in refiner.parameters())
    model.assert_gradient_isolation(require_active_grad=True)


def test_zero_grad_returns_the_encoder_to_none(view: dict) -> None:
    model = make_model(track="move", decoder=SynthMoveRefiner(),
                       decoder_id="M1")
    _backward(model, view)
    model.zero_grad()
    assert all(p.grad is None for p in model.assembly.encoder.parameters())


def test_missing_active_grad_is_refused(view: dict) -> None:
    """The plan's third assertion (:723) fires when no grad is produced."""
    model = make_model(track="move", decoder=SynthMoveRefiner(),
                       decoder_id="M1")
    model.zero_grad()
    with pytest.raises(V543AssemblyError) as error:
        model.assert_gradient_isolation(require_active_grad=True)
    assert error.value.code == "V543_GRADIENT_ACTIVE_MISSING"


def test_baseline_run_skips_the_active_grad_requirement(view: dict) -> None:
    """A zero-parameter baseline has no parameter gradient by construction."""
    model = make_model(track="move")
    model.zero_grad()
    out = model(view)
    # the decoder half is a no-op, so the loss carries no graph at all
    assert not out["u"].requires_grad
    report = model.assert_gradient_isolation(require_active_grad=False)
    assert report["active_trainable_params"] == 0
    assert report["active_params_with_grad"] == 0
    assert all(p.grad is None for p in model.assembly.encoder.parameters())
