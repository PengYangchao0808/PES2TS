"""V5.2 head math tests: shapes, exact param counts, bitwise backbone.

Campaign ``v52_training.heads`` (FROZEN) + 30h plan §5.1.  The headline
guarantee under test: attaching/running the V5.2 heads leaves the R3
backbone — and therefore the Move output ``u`` — BIT-for-bit unchanged
(``torch.equal``), and the head parameter counts are exactly 2,113 + 65,
reported SEPARATELY from ``MEASURED_PARAMS_V3 = 55,441``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.pesn.contracts.key_atom_v3 import (  # noqa: E402
    KeyAtomV3ContractError,
    forward_view_v3,
)
from pes2ts_core.pesn.models.key_atom_egnn_v3 import (  # noqa: E402
    MEASURED_PARAMS_V3,
    StarEGNNV3,
)
from pes2ts_core.pesn.models.key_atom_heads_v3 import (  # noqa: E402
    MEASURED_PARAMS_HEADS_V52,
    MEASURED_PARAMS_PROGRESS_HEAD_V52,
    MEASURED_PARAMS_TS_STOP_HEAD_V52,
    MEASURED_PARAMS_V52_JOINT,
    ProgressHeadV3,
    TSStopHeadV3,
    KeyAtomV52Model,
    measured_params_v52_joint,
)
from scripts.agent_ops.feature_gate.v3_fixtures import make_v3_batch  # noqa: E402

SEED = 29


@pytest.fixture()
def batch():
    return forward_view_v3(make_v3_batch(n_graphs=3))


@pytest.fixture()
def shared():
    torch.manual_seed(SEED)
    backbone = StarEGNNV3()
    model = KeyAtomV52Model(backbone)
    model.eval()
    return backbone, model


def test_progress_head_shape_and_range(batch) -> None:
    torch.manual_seed(SEED)
    head = ProgressHeadV3()
    z = torch.randn(3, 64)
    q = head(z)
    assert tuple(q.shape) == (3,)
    assert bool((q.abs() <= 1.0).all())


def test_progress_head_exact_params() -> None:
    head = ProgressHeadV3()
    n = sum(p.numel() for p in head.parameters())
    assert n == 2113 == MEASURED_PARAMS_PROGRESS_HEAD_V52
    # (64·32 + 32) + (32·1 + 1)
    assert n == (64 * 32 + 32) + (32 + 1)


def test_progress_head_frozen_architecture() -> None:
    with pytest.raises(RuntimeError):
        ProgressHeadV3(hidden=16)          # width is frozen
    with pytest.raises(RuntimeError):
        ProgressHeadV3(z_dim=32)


def test_ts_stop_head_shape_and_raw_logit() -> None:
    torch.manual_seed(SEED)
    head = TSStopHeadV3()
    n = sum(p.numel() for p in head.parameters())
    assert n == 65 == MEASURED_PARAMS_TS_STOP_HEAD_V52
    logits = head(torch.randn(4, 64))
    assert tuple(logits.shape) == (4,)
    # raw logit: the module applies NO sigmoid (frozen recipe: train with
    # BCEWithLogitsLoss — plan §5.1 不要训练时预先 sigmoid 两次)
    assert isinstance(head.fc, torch.nn.Linear) and head.fc.out_features == 1
    assert not any(isinstance(m, torch.nn.Sigmoid) for m in head.modules())


def test_param_counts_separate_from_backbone() -> None:
    counts = KeyAtomV52Model().head_parameter_counts()
    assert counts["backbone"] == MEASURED_PARAMS_V3 == 55441
    assert counts["progress_head"] == MEASURED_PARAMS_PROGRESS_HEAD_V52
    assert counts["ts_stop_head"] == MEASURED_PARAMS_TS_STOP_HEAD_V52
    assert (counts["progress_head"] + counts["ts_stop_head"]
            == MEASURED_PARAMS_HEADS_V52 == 2178)
    joint = measured_params_v52_joint()
    assert joint == 55441 + 2178 == MEASURED_PARAMS_V52_JOINT == 57619


def test_backbone_u_bitwise_unchanged_by_heads(shared, batch) -> None:
    """THE attachability guarantee: heads never perturb ``u``/``z``."""
    backbone, model = shared
    with torch.no_grad():
        out_bb = backbone(batch)
        out = model(batch)
        out_move = model.forward_move(batch)
    assert torch.equal(out["u"], out_bb["u"])
    assert torch.equal(out["z"], out_bb["z"])
    assert torch.equal(out_move["u"], out_bb["u"])
    assert torch.equal(out_move["z"], out_bb["z"])


def test_backbone_weights_untouched_after_attach_and_head_forward(
        shared, batch) -> None:
    backbone, model = shared
    before = {k: v.clone() for k, v in backbone.state_dict().items()}
    with torch.no_grad():
        model(batch)                    # head forward passes through backbone
    after = backbone.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_heads_do_not_share_parameters_with_backbone() -> None:
    backbone = StarEGNNV3()
    model = KeyAtomV52Model(backbone)
    bb_params = {id(p) for p in backbone.parameters()}
    head_params = {id(p) for p in model.progress_head.parameters()} | \
        {id(p) for p in model.ts_stop_head.parameters()}
    assert not (bb_params & head_params)


def test_full_forward_shapes(shared, batch) -> None:
    _, model = shared
    with torch.no_grad():
        out = model(batch)
    n_keys = int(batch["key_index"].numel())
    n_graphs = int(batch["cv_target_delta"].shape[0])
    assert tuple(out["u"].shape) == (n_keys, 3)
    assert tuple(out["z"].shape) == (n_graphs, 64)
    assert tuple(out["q"].shape) == (n_graphs,)
    assert tuple(out["stop_logit"].shape) == (n_graphs,)


def test_heads_keep_forward_view_contract(shared, batch) -> None:
    """Supervision keys still refused with heads attached (R3 unchanged)."""
    _, model = shared
    leaked = dict(batch)
    leaked["y_teacher"] = torch.zeros(1, 3)
    with pytest.raises(KeyAtomV3ContractError):
        model(leaked)


def test_eval_determinism_with_heads(shared, batch) -> None:
    _, model = shared
    with torch.no_grad():
        a = model(batch)
        b = model(batch)
    for key in ("u", "z", "q", "stop_logit"):
        assert torch.equal(a[key], b[key]), key


def test_grads_flow_to_heads_and_backbone() -> None:
    torch.manual_seed(SEED)
    model = KeyAtomV52Model(StarEGNNV3())
    batch = forward_view_v3(make_v3_batch(n_graphs=2))
    out = model(batch)
    loss = out["q"].sum() + out["stop_logit"].sum() + out["u"].sum()
    loss.backward()
    assert all(p.grad is not None for p in model.progress_head.parameters())
    assert all(p.grad is not None for p in model.ts_stop_head.parameters())
    # q/stop gradients reach the shared backbone (joint fine-tune path) …
    assert model.backbone.z_head.weight.grad is not None
