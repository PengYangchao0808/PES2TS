"""W-FEAT E2 — v3 feature-gate integration (G1 report over the real stack).

Runs the WP-F2 harness (normative PD-2 preset bound to the campaign yaml)
against the R3 v3 StarEGNN + F3-superset fixture batch and asserts the full
T1.1–T1.4c gate report; the 1e-4 record-only reference can never widen a
pass (PD-2).  Downstream G1 consumption note: use the report ``state`` /
``passed`` fields (``gate_admissible`` alone is not the gate criterion).

Fixture-level results — ``not_admissible_until_gate_pass`` until G0+G1.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from scripts.agent_ops.feature_gate.equivariance_harness import (  # noqa: E402
    EDGE_ALIGNED_TENSOR_KEYS,
    NONSPATIAL_TENSOR_KEYS,
    build_gate_report,
    check_feature_relabel_identity,
    check_feature_reverse_change,
)
from scripts.agent_ops.feature_gate.v3_fixtures import (  # noqa: E402
    build_fixture_features,
    make_intent_off_batch,
    make_v3_gate_batch,
)
from pes2ts_core.pesn.models.key_atom_egnn_v3 import StarEGNNV3  # noqa: E402

NORMATIVE = "30h_v1.1_normative"
RECORD_ONLY = "r3_v3_record"


@pytest.fixture()
def model():
    torch.manual_seed(17)
    return StarEGNNV3()


@pytest.fixture()
def gate_inputs():
    batch = make_v3_gate_batch(n_graphs=2)
    intent_off = make_intent_off_batch({"n_graphs": 2}, mode="full")
    return batch, intent_off


def test_full_gate_report_passes(model, gate_inputs) -> None:
    batch, intent_off = gate_inputs
    report = build_gate_report(
        model, batch, preset=NORMATIVE, intent_off=intent_off)
    states = {t["test_id"]: t["state"] for t in report["tests"]}
    assert states == {
        "T1.1": "pass", "T1.2": "pass", "T1.3": "pass",
        "T1.4": "pass", "T1.4b": "pass", "T1.4c": "pass"}, states
    assert report["state"] == "pass"
    assert report["passed"] is True
    assert report["gate_admissible"] is True
    assert report["yaml_sha256"]  # tolerance binding recorded
    # downstream consumption contract: state + passed carry the verdict
    assert report["state"] in ("pass", "fail", "blocked", "partial", "not_run")
    json.dumps(report)  # report stays JSON-serializable


def test_record_only_report_never_passes(model, gate_inputs) -> None:
    batch, intent_off = gate_inputs
    report = build_gate_report(
        model, batch, preset=RECORD_ONLY, intent_off=intent_off)
    assert report["passed"] is False
    assert report["gate_admissible"] is False
    assert report["banner"] == "record-only, not a pass gate"
    assert report["state"] == "blocked"   # never a silent pass


def test_feature_level_checks_on_v3_features() -> None:
    fwd = build_fixture_features("forward")
    swapped = build_fixture_features("forward", endpoint_names=("B", "A"))
    rev = build_fixture_features("reverse")
    assert check_feature_relabel_identity(fwd, swapped, preset=NORMATIVE)["passed"]
    assert check_feature_reverse_change(fwd, rev, preset=NORMATIVE)["passed"]


def test_v3_harness_extension_constants() -> None:
    """v3 extensions: non-spatial [*,3] channels and edge-aligned rows are
    excluded from the spatial transforms / node gathers."""
    assert "intent_role" in NONSPATIAL_TENSOR_KEYS
    assert "pos" not in NONSPATIAL_TENSOR_KEYS
    for key in ("edge_attr", "b_src", "b_tgt", "delta_b_intent",
                "abs_delta", "is_changed"):
        assert key in EDGE_ALIGNED_TENSOR_KEYS
    assert "intent_role" not in EDGE_ALIGNED_TENSOR_KEYS


def test_permutation_gather_ignores_edge_aligned_even_when_e_equals_n() -> None:
    """Guard: an edge-aligned [E,3] tensor must not be node-gathered even in
    the degenerate E == N case (v3 batch bookkeeping safety)."""
    from scripts.agent_ops.feature_gate.equivariance_harness import (
        check_permutation_equivariance,
    )

    n = 4
    torch.manual_seed(0)
    pos = torch.randn(n, 3, dtype=torch.float64)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_attr = torch.randn(4, 3, dtype=torch.float64)   # E == N degenerate
    calls = {"n": 0}

    class _Echo(torch.nn.Module):
        def forward(self, batch):
            calls["n"] += 1
            return {"u": batch["pos"][batch["key_index"]]}

    model = _Echo()
    batch = {
        "pos": pos, "edge_index": edge_index, "edge_attr": edge_attr,
        "key_index": torch.arange(n, dtype=torch.long),
    }
    report = check_permutation_equivariance(model, batch, preset=NORMATIVE)
    assert calls["n"] >= 2
    assert report["passed"], report
