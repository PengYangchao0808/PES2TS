"""V5.3-MP head math tests: shapes, exact params, bitwise backbone, LEGACY.

MoveProcessContract v1 §6 (模型与头) + V5.3-MP plan §3/§8.1/§8.2.  The
headline guarantees under test:

* attaching/running the V5.3-MP ``process_head`` leaves the R3 backbone —
  and therefore the Move output ``u`` — BIT-for-bit unchanged
  (``torch.equal``), same test caliber as ``tests/pesn/test_r3_v52_heads.py``;
* ``forward_move`` is bitwise identical to a direct backbone call;
* exact parameter accounting: 55,441 backbone + 2,113 progress + 65
  endpoint = 57,619 joint (reported SEPARATELY per head);
* ``TSStopHeadV3.legacy_frozen`` marks LEGACY_TS_STOP (contract §2/§6);
* V5.2 import-compat regression: ``KeyAtomV52Model`` still constructs and
  forwards (nothing renamed/removed).
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
    LEGACY_TS_STOP_HEAD,
    MEASURED_PARAMS_ENDPOINT_HEAD_V53MP,
    MEASURED_PARAMS_HEADS_V52,
    MEASURED_PARAMS_HEADS_V53MP,
    MEASURED_PARAMS_PROGRESS_HEAD_V52,
    MEASURED_PARAMS_TS_STOP_HEAD_V52,
    MEASURED_PARAMS_V52_JOINT,
    MEASURED_PARAMS_V53MP_JOINT,
    V53MP_HEADS_VERSION,
    EndpointHeadV53MP,
    KeyAtomV52Model,
    KeyAtomV53MPModel,
    ProgressHeadV3,
    ReactionProcessHeadV53MP,
    TSStopHeadV3,
    measured_params_v53mp_joint,
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
    model = KeyAtomV53MPModel(backbone)
    model.eval()
    return backbone, model


# ---------------------------------------------------------------------------
# Shapes / frozen architecture
# ---------------------------------------------------------------------------
def test_endpoint_head_shape_and_raw_logit() -> None:
    torch.manual_seed(SEED)
    head = EndpointHeadV53MP()
    n = sum(p.numel() for p in head.parameters())
    assert n == 65 == MEASURED_PARAMS_ENDPOINT_HEAD_V53MP
    logits = head(torch.randn(4, 64))
    assert tuple(logits.shape) == (4,)
    # raw logit: NO sigmoid in the module (contract §3.2/§7 BCEWithLogits)
    assert isinstance(head.fc, torch.nn.Linear) and head.fc.out_features == 1
    assert not any(isinstance(m, torch.nn.Sigmoid) for m in head.modules())


def test_endpoint_head_frozen_architecture() -> None:
    with pytest.raises(RuntimeError):
        EndpointHeadV53MP(z_dim=32)      # width is frozen (Linear(64,1))


def test_process_head_outputs_and_shared_order_score(batch) -> None:
    torch.manual_seed(SEED)
    head = ReactionProcessHeadV53MP()
    z = torch.randn(3, 64)
    out = head(z)
    assert set(out) == {"q", "endpoint_logit", "order_score"}
    for key in ("q", "endpoint_logit", "order_score"):
        assert tuple(out[key].shape) == (3,), key
    assert bool((out["q"].abs() <= 1.0).all())           # tanh
    # frozen wiring choice: order_score IS q (contract §6 可与 q 共享)
    assert torch.equal(out["order_score"], out["q"])


def test_full_forward_shapes(shared, batch) -> None:
    _, model = shared
    with torch.no_grad():
        out = model(batch)
    n_keys = int(batch["key_index"].numel())
    n_graphs = int(batch["cv_target_delta"].shape[0])
    assert set(out) == {"u", "z", "q", "endpoint_logit", "order_score"}
    assert tuple(out["u"].shape) == (n_keys, 3)
    assert tuple(out["z"].shape) == (n_graphs, 64)
    for key in ("q", "endpoint_logit", "order_score"):
        assert tuple(out[key].shape) == (n_graphs,), key
    assert torch.equal(out["order_score"], out["q"])


# ---------------------------------------------------------------------------
# Parameter accounting (exact)
# ---------------------------------------------------------------------------
def test_param_counts_separate_from_backbone() -> None:
    counts = KeyAtomV53MPModel().head_parameter_counts()
    assert counts["backbone"] == MEASURED_PARAMS_V3 == 55441
    assert counts["process_head_progress"] == MEASURED_PARAMS_PROGRESS_HEAD_V52 == 2113
    assert counts["process_head_endpoint"] == MEASURED_PARAMS_ENDPOINT_HEAD_V53MP == 65
    assert (counts["process_head_progress"] + counts["process_head_endpoint"]
            == counts["process_head"] == MEASURED_PARAMS_HEADS_V53MP == 2178)
    # V5.2 heads total unchanged (endpoint head is shape-identical to stop)
    assert MEASURED_PARAMS_HEADS_V52 == 2178 and MEASURED_PARAMS_V52_JOINT == 57619


def test_joint_param_count_exact() -> None:
    assert KeyAtomV53MPModel().n_parameters == 57619 == MEASURED_PARAMS_V53MP_JOINT
    assert measured_params_v53mp_joint() == 57619
    assert MEASURED_PARAMS_V53MP_JOINT == MEASURED_PARAMS_V3 + 2178


def test_heads_do_not_share_parameters_with_backbone() -> None:
    backbone = StarEGNNV3()
    model = KeyAtomV53MPModel(backbone)
    bb_params = {id(p) for p in backbone.parameters()}
    head_params = {id(p) for p in model.process_head.parameters()}
    assert not (bb_params & head_params)


def test_backbone_shared_by_reference() -> None:
    backbone = StarEGNNV3()
    model = KeyAtomV53MPModel(backbone)
    assert model.backbone is backbone


# ---------------------------------------------------------------------------
# Attachability: u bit-for-bit unchanged (contract §6)
# ---------------------------------------------------------------------------
def test_backbone_u_bitwise_unchanged_by_v53mp_heads(batch) -> None:
    """THE attachability guarantee: capture u BEFORE attaching heads."""
    torch.manual_seed(SEED)
    backbone = StarEGNNV3()
    with torch.no_grad():
        out_before = backbone(batch)
    model = KeyAtomV53MPModel(backbone)     # attach process_head
    model.eval()
    with torch.no_grad():
        out = model(batch)
        out_move = model.forward_move(batch)
    assert torch.equal(out["u"], out_before["u"])
    assert torch.equal(out["z"], out_before["z"])
    assert torch.equal(out_move["u"], out_before["u"])
    assert torch.equal(out_move["z"], out_before["z"])


def test_backbone_weights_untouched_after_attach_and_head_forward(
        shared, batch) -> None:
    backbone, model = shared
    before = {k: v.clone() for k, v in backbone.state_dict().items()}
    with torch.no_grad():
        model(batch)                    # head forward passes through backbone
    after = backbone.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_forward_move_bitwise_equals_backbone_direct(shared, batch) -> None:
    backbone, model = shared
    with torch.no_grad():
        out_bb = backbone(batch)
        out_move = model.forward_move(batch)
    assert torch.equal(out_move["u"], out_bb["u"])
    assert torch.equal(out_move["z"], out_bb["z"])
    # backbone-only view emits no head outputs
    assert set(out_move) == {"u", "z"}


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
    for key in ("u", "z", "q", "endpoint_logit", "order_score"):
        assert torch.equal(a[key], b[key]), key


def test_grads_flow_to_heads_and_backbone() -> None:
    torch.manual_seed(SEED)
    model = KeyAtomV53MPModel(StarEGNNV3())
    batch = forward_view_v3(make_v3_batch(n_graphs=2))
    out = model(batch)
    loss = (out["q"].sum() + out["endpoint_logit"].sum()
            + out["order_score"].sum() + out["u"].sum())
    loss.backward()
    assert all(p.grad is not None for p in model.process_head.parameters())
    assert model.process_head.progress_head.fc1.weight.grad is not None
    assert model.process_head.endpoint_head.fc.weight.grad is not None
    # q/endpoint gradients reach the shared backbone (joint fine-tune path)
    assert model.backbone.z_head.weight.grad is not None


# ---------------------------------------------------------------------------
# Legacy discipline + V5.2 import-compat regression
# ---------------------------------------------------------------------------
def test_ts_stop_head_marked_legacy_frozen() -> None:
    assert TSStopHeadV3.legacy_frozen is True
    assert LEGACY_TS_STOP_HEAD == "TSStopHeadV3"
    # docstring carries the LEGACY_TS_STOP prohibition note
    assert "LEGACY_TS_STOP" in (TSStopHeadV3.__doc__ or "")
    assert "forbidden for new training" in (TSStopHeadV3.__doc__ or "")


def test_v53mp_version_constant() -> None:
    assert V53MP_HEADS_VERSION == "pesn_key_atom_v53mp_heads_v1"


def test_v52_model_still_constructs_and_forwards() -> None:
    """Import-compat regression: nothing renamed/removed for V5.2."""
    torch.manual_seed(SEED)
    model = KeyAtomV52Model()
    model.eval()
    batch = forward_view_v3(make_v3_batch(n_graphs=2))
    with torch.no_grad():
        out = model(batch)
    assert "q" in out and "stop_logit" in out
    assert tuple(out["q"].shape) == tuple(out["stop_logit"].shape)
    assert ProgressHeadV3().fc1.in_features == 64
    n_stop = sum(p.numel() for p in TSStopHeadV3().parameters())
    assert n_stop == MEASURED_PARAMS_TS_STOP_HEAD_V52 == 65
