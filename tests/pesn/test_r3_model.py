"""R3 v3 model math tests: param budget, fp32, determinism, typed invalid.

PD-2 tolerances apply where numeric (campaign ``feature_contract``;
preset ``30h_v1.1_normative``).  All results are fixture-level and remain
``not_admissible_until_gate_pass`` until G0+G1 run on real data.
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
    PARAM_BUDGET_V3,
    STAR_EGNN_V3_VERSION,
    KeyAtomV3InvalidInputError,
    StarEGNNV3,
)
from scripts.agent_ops.feature_gate.v3_fixtures import make_v3_batch  # noqa: E402

SEED = 17


@pytest.fixture()
def model():
    torch.manual_seed(SEED)
    return StarEGNNV3()


@pytest.fixture()
def batch():
    return forward_view_v3(make_v3_batch(n_graphs=2))


def test_registered_version(model) -> None:
    assert STAR_EGNN_V3_VERSION == "pesn_key_atom_star_egnn_v3"


def test_param_count_reported_and_under_budget(model) -> None:
    """~59k-class backbone: assert <70k, report the exact count."""
    n = model.n_parameters
    assert n == MEASURED_PARAMS_V3, (n, MEASURED_PARAMS_V3)
    assert n < PARAM_BUDGET_V3
    # exact count recorded for the handoff / run records
    print(f"MEASURED_PARAMS_V3 = {n}")


def test_fp32_params_and_outputs(model, batch) -> None:
    for name, p in model.named_parameters():
        assert p.dtype == torch.float32, (name, p.dtype)
    out = model(batch)
    assert out["u"].dtype == torch.float32
    assert out["z"].dtype == torch.float32


def test_forward_determinism_bitwise(model, batch) -> None:
    """No dropout/batchnorm: two eval-mode calls are bitwise equal."""
    model.eval()
    a = model(batch)
    b = model(batch)
    assert torch.equal(a["u"], b["u"])
    assert torch.equal(a["z"], b["z"])


def test_no_stochastic_modules(model) -> None:
    banned = tuple(
        m for m in model.modules()
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout1d,
                          torch.nn.Dropout2d, torch.nn.BatchNorm1d,
                          torch.nn.BatchNorm2d, torch.nn.InstanceNorm1d)))
    assert banned == ()


def test_output_shapes(model, batch) -> None:
    out = model(batch)
    n_keys = int(batch["key_index"].numel())
    n_graphs = int(batch["cv_target_delta"].shape[0])
    assert out["u"].shape == (n_keys, 3)
    assert out["z"].shape == (n_graphs, 64)
    assert bool(torch.isfinite(out["u"]).all())


def test_nan_input_typed_invalid(model, batch) -> None:
    bad = dict(batch)
    bad["pos"] = torch.full_like(batch["pos"], float("nan"))
    with pytest.raises(KeyAtomV3InvalidInputError) as exc:
        model(bad)
    assert exc.value.code == "NONFINITE_INPUT"


def test_z_out_of_range_typed_invalid(model, batch) -> None:
    bad = dict(batch)
    bad["z"] = batch["z"].clone()
    bad["z"][0] = 200
    with pytest.raises(KeyAtomV3InvalidInputError) as exc:
        model(bad)
    assert exc.value.code == "Z_OUT_OF_RANGE"


def test_supervision_key_refused(model, batch) -> None:
    """Forward legality: a supervision key in the batch is a typed refusal."""
    bad = dict(batch)
    bad["y_teacher"] = torch.zeros(int(batch["key_index"].numel()), 3)
    with pytest.raises(KeyAtomV3ContractError) as exc:
        model(bad)
    assert exc.value.code == "FORWARD_VIEW_LEAK"


def test_fixed_shape_refuses_other_widths() -> None:
    with pytest.raises(RuntimeError):
        StarEGNNV3(hidden=64)
