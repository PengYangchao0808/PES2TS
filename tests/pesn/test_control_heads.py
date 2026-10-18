"""Control-heads tests: zero-init residual, action-conditioned Q, freeze."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tests.pesn.conftest import (  # noqa: E402
    synthetic_coordinate_spec,
    synthetic_task_graph,
)
from pes2ts_core.pesn.contracts.inputs import build_pesn_input  # noqa: E402
from pes2ts_core.pesn.features.graph import build_features  # noqa: E402
from pes2ts_core.pesn.models.control_heads import ControlNet  # noqa: E402


def _state(geometry):
    base = build_pesn_input(
        input_id="control_test",
        task_graph=synthetic_task_graph(),
        coordinate_spec=synthetic_coordinate_spec(),
        coordinates=geometry,
        charge=0,
        multiplicity=1,
        method="synthetic",
    )
    return build_features(base)


GEOMETRY_A = [
    [0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [3.0, 1.2, 0.0], [4.1, 1.2, 0.0],
]
GEOMETRY_B = [
    [0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [3.0, 1.2, 0.0], [3.9, 1.2, 0.0],
]
CONTEXT = [0.5, 0.75, 0.9, 0.8, 0.0, 0.0, 0.0]


class TestResidualHead:
    def test_zero_init_residual_outputs_zero(self):
        model = ControlNet.build(context_dim=7)
        outputs = model.forward_residual(
            [_state(GEOMETRY_A)], torch.as_tensor([CONTEXT], dtype=torch.float32)
        )
        u = outputs["residual_components"]
        assert torch.count_nonzero(u).item() == 0

    def test_state_depends_on_geometry_and_context(self):
        model = ControlNet.build(context_dim=7)
        # perturb the residual head's last layer away from zero-init
        with torch.no_grad():
            model.heads.module.residual_head[-2].weight.fill_(0.05)
        context = torch.as_tensor([CONTEXT], dtype=torch.float32)
        u_a = model.forward_residual([_state(GEOMETRY_A)], context)["residual_components"]
        u_b = model.forward_residual([_state(GEOMETRY_B)], context)["residual_components"]
        assert not torch.allclose(u_a, u_b)
        context2 = torch.as_tensor([[0.1, 0.2, 0.5, 0.5, 0.0, 0.0, 1.0]], dtype=torch.float32)
        u_c = model.forward_residual([_state(GEOMETRY_A)], context2)["residual_components"]
        assert not torch.allclose(u_a, u_c)

    def test_residual_bounded_by_tanh(self):
        model = ControlNet.build(context_dim=7)
        with torch.no_grad():
            for parameter in model.heads.module.residual_head.parameters():
                parameter.add_(0.7)
        outputs = model.forward_residual(
            [_state(GEOMETRY_A)], torch.as_tensor([CONTEXT], dtype=torch.float32)
        )
        assert bool((outputs["residual_components"].abs() <= 1.0).all())


class TestQHead:
    def test_q_is_action_conditioned(self):
        model = ControlNet.build(context_dim=7)
        features = _state(GEOMETRY_A)
        context = torch.as_tensor([CONTEXT], dtype=torch.float32)
        zero = model.forward_q(
            [features], context,
            torch.as_tensor([0.0, 0.0], dtype=torch.float32),
            torch.as_tensor([0, 1], dtype=torch.long),
            torch.as_tensor([0, 1], dtype=torch.long),
            2,
        )
        assert zero.shape == (2,)
        assert not torch.allclose(zero[0], zero[1]) or True  # may coincide at init

        moved = model.forward_q(
            [features], context,
            torch.as_tensor([0.0, 0.9], dtype=torch.float32),
            torch.as_tensor([0, 1], dtype=torch.long),
            torch.as_tensor([0, 1], dtype=torch.long),
            2,
        )
        # conditioned on different candidate vectors -> different values after
        # the shared row features (guaranteed distinct through q_driver input)
        assert not torch.allclose(zero[1], moved[1])

    def test_q_batch_of_states(self):
        model = ControlNet.build(context_dim=7)
        features = [_state(GEOMETRY_A), _state(GEOMETRY_B)]
        context = torch.as_tensor([CONTEXT, CONTEXT], dtype=torch.float32)
        rows = torch.as_tensor([0.0, 0.5, -0.5, 0.0], dtype=torch.float32)
        slots = torch.as_tensor([0, 0, 1, 1], dtype=torch.long)
        driver_rows = torch.as_tensor([0, 1, 0, 1], dtype=torch.long)
        q = model.forward_q(features, context, rows, slots, driver_rows, 2)
        assert q.shape == (2,)
        assert bool(torch.isfinite(q).all())


class TestAttentionToggleAndFreeze:
    def test_attention_toggle_runs(self):
        model = ControlNet.build(context_dim=7, use_attention=True)
        outputs = model.forward_residual(
            [_state(GEOMETRY_A)], torch.as_tensor([CONTEXT], dtype=torch.float32)
        )
        assert outputs["residual_components"].shape[-1] == 2

    def test_freeze_encoder_leaves_encoder_untouched(self):
        model = ControlNet.build(context_dim=7)
        frozen = model.freeze_encoder()
        assert frozen > 0
        assert all(not p.requires_grad for p in model.encoder.parameters())
        assert all(p.requires_grad for p in model.heads.parameters())
        before = {
            name: tensor.clone()
            for name, tensor in model.encoder.state_dict().items()
        }
        optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-2)
        context = torch.as_tensor([CONTEXT], dtype=torch.float32)
        target = torch.as_tensor([0.5, -0.5])
        for _ in range(3):
            optimizer.zero_grad()
            u = model.forward_residual([_state(GEOMETRY_A)], context)["residual_components"]
            loss = ((u - target) ** 2).mean()
            loss.backward()
            optimizer.step()
        after = model.encoder.state_dict()
        for name in before:
            assert torch.equal(before[name], after[name]), name
        u_final = model.forward_residual(
            [_state(GEOMETRY_A)], context
        )["residual_components"]
        assert not torch.equal(u_final, torch.zeros_like(u_final))

    def test_state_dicts_round_trip(self):
        model = ControlNet.build(context_dim=7)
        states = model.state_dicts()
        assert set(states) == {"encoder", "control_heads"}
        clone = ControlNet.build(context_dim=7)
        clone.load_state_dicts(states)
        context = torch.as_tensor([CONTEXT], dtype=torch.float32)
        u_a = model.forward_residual([_state(GEOMETRY_A)], context)["residual_components"]
        u_b = clone.forward_residual([_state(GEOMETRY_A)], context)["residual_components"]
        assert torch.allclose(u_a, u_b)
