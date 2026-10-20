"""R3 T1.1 — relabel invariance (display-name swap ⇒ identical everything).

Hard gates (PD-2 / campaign ``feature_contract.tolerances``):

* features BIT-identical under an endpoint display-name swap (and under the
  double swap — involution);
* eval-mode model outputs bit-identical with ``max|Δu| ≤ 1e-6``.

Fixture-level results — ``not_admissible_until_gate_pass`` until G0+G1.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.pesn.contracts.key_atom_v3 import (  # noqa: E402
    ENDPOINT_NAMES_KEY,
    forward_view_v3,
)
from pes2ts_core.pesn.models.key_atom_egnn_v3 import StarEGNNV3  # noqa: E402
from scripts.agent_ops.feature_gate.equivariance_harness import (  # noqa: E402
    RELABEL_ABS_TOL,
    check_feature_relabel_identity,
    check_relabel_invariance,
)
from scripts.agent_ops.feature_gate.v3_fixtures import (  # noqa: E402
    build_fixture_features,
    make_v3_gate_batch,
)

PRESET = "30h_v1.1_normative"
LEVELS = ("F0", "F1", "F2", "F3")
NAMES = (("A", "B"), ("reactant", "product"), ("X", "Y"))


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("names", NAMES)
def test_relabel_features_bit_identical(level, names) -> None:
    a = build_fixture_features("forward", level=level, endpoint_names=names)
    b = build_fixture_features(
        "forward", level=level, endpoint_names=(names[1], names[0]))
    assert tuple(a[ENDPOINT_NAMES_KEY]) != tuple(b[ENDPOINT_NAMES_KEY])
    for key, value in a.items():
        if key == ENDPOINT_NAMES_KEY:
            continue
        other = b[key]
        if isinstance(value, np.ndarray):
            assert np.array_equal(value, other), f"{key} changed on relabel"
        else:
            assert value == other
    report = check_feature_relabel_identity(a, b, preset=PRESET)
    assert report["passed"], report


@pytest.mark.parametrize("level", LEVELS)
def test_relabel_involution_double_swap_restores(level) -> None:
    a = build_fixture_features("forward", level=level, endpoint_names=("A", "B"))
    once = build_fixture_features("forward", level=level, endpoint_names=("B", "A"))
    twice = build_fixture_features("forward", level=level, endpoint_names=("A", "B"))
    assert check_feature_relabel_identity(once, twice, preset=PRESET)["passed"]
    for key, value in a.items():
        if key == ENDPOINT_NAMES_KEY:
            continue
        if isinstance(value, np.ndarray):
            assert np.array_equal(value, twice[key]), key


def test_relabel_swap_only_preserves_signed_intent() -> None:
    """The swap is an INVOLUTION on display names only: signed features do
    not move (same physical task ⇒ same G_rxn)."""
    a = build_fixture_features("forward", endpoint_names=("A", "B"))
    b = build_fixture_features("forward", endpoint_names=("B", "A"))
    for key in ("b_src", "b_tgt", "delta_b_intent", "cv_target_delta",
                "signed_intent", "intent_role"):
        assert np.array_equal(a[key], b[key]), key
    assert not np.array_equal(a["delta_b_intent"], np.zeros_like(
        a["delta_b_intent"]))  # sanity: intent is actually present


def test_relabel_model_outputs_bit_identical_within_pd2() -> None:
    torch.manual_seed(17)
    model = StarEGNNV3()
    batch = make_v3_gate_batch(n_graphs=2)
    report = check_relabel_invariance(model, batch, preset=PRESET)
    assert report["passed"], report
    assert report["bit_identical"] is True
    assert report["max_abs_delta"] <= RELABEL_ABS_TOL
    assert report["relabel_abs_tol"] == 1e-6


def test_relabel_gate_blocked_without_endpoint_names() -> None:
    torch.manual_seed(17)
    model = StarEGNNV3()
    batch = make_v3_gate_batch(n_graphs=2)
    batch.pop(ENDPOINT_NAMES_KEY)
    report = check_relabel_invariance(model, batch, preset=PRESET)
    assert report["passed"] is False
    assert report["state"] == "blocked"
    # forward view is what the model consumes; metadata absence is legal
    out = model(forward_view_v3(batch))
    assert out["u"].shape[0] > 0
