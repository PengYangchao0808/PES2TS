"""R3 T1.3 — rotation/translation/permutation equivariance (PD-2 tolerances).

Normative preset ``30h_v1.1_normative`` (rel 1e-5 + abs 1e-6, bound to the
campaign yaml feature_contract.tolerances; PD-2 forbids auto-relaxation —
the 1e-4 record-only reference can never widen a pass).  Reports carry
median/P99/max errors.  Fixture-level — ``not_admissible_until_gate_pass``
until G0+G1 run on real data.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.pesn.models.key_atom_egnn_v3 import StarEGNNV3  # noqa: E402
from scripts.agent_ops.feature_gate.equivariance_harness import (  # noqa: E402
    PRESETS,
    PresetContractError,
    check_permutation_equivariance,
    check_rotation_equivariance,
    check_translation_invariance,
    load_campaign_feature_contract,
    resolve_preset,
    run_feature_gates,
    verify_preset_against_yaml,
    DEFAULT_CAMPAIGN_YAML,
)
from scripts.agent_ops.feature_gate.v3_fixtures import make_v3_gate_batch  # noqa: E402

NORMATIVE = "30h_v1.1_normative"
RECORD_ONLY = "r3_v3_record"


@pytest.fixture()
def model():
    torch.manual_seed(17)
    return StarEGNNV3()


@pytest.fixture()
def batch():
    return make_v3_gate_batch(n_graphs=2)


def test_translation_invariance_pd2(model, batch) -> None:
    report = check_translation_invariance(model, batch, preset=NORMATIVE)
    assert report["passed"], report
    assert report["max_abs_delta"] <= report["threshold"]
    print("translation median/p99/max:", report["median"], report["p99"],
          report["max"])


def test_rotation_equivariance_pd2(model, batch) -> None:
    report = check_rotation_equivariance(model, batch, preset=NORMATIVE)
    assert report["passed"], report
    errs = report["per_atom_errors"]
    assert max(errs) <= report["threshold"]
    print("rotation median/p99/max:", report["median"], report["p99"],
          report["max"])


def test_permutation_equivariance_pd2(model, batch) -> None:
    report = check_permutation_equivariance(model, batch, preset=NORMATIVE)
    assert report["passed"], report
    print("permutation median/p99/max:", report["median"], report["p99"],
          report["max"])


def test_intent_role_not_rotated_as_spatial_vector(model, batch) -> None:
    """v3 extension: ``intent_role`` is a [N,3] CHANNEL tensor, so the
    rotation transform must leave it untouched (equivalence still passes)."""
    from scripts.agent_ops.feature_gate.equivariance_harness import (
        NONSPATIAL_TENSOR_KEYS,
        _random_rotation,
    )

    assert "intent_role" in NONSPATIAL_TENSOR_KEYS
    assert "pos" not in NONSPATIAL_TENSOR_KEYS
    R = _random_rotation(0)
    report = check_rotation_equivariance(
        model, batch, preset=NORMATIVE, rotation=R)
    assert report["passed"], report


def test_record_only_preset_never_admits(model, batch) -> None:
    """The 1e-4 record reference is diagnostics only: a numerically-small
    error can satisfy the loose check, but the preset is never admissible
    and a run/report built on it can never pass (PD-2)."""
    report = check_rotation_equivariance(model, batch, preset=RECORD_ONLY)
    assert report["record_only"] is True
    assert report["gate_admissible"] is False
    assert report["banner"] == "record-only, not a pass gate"
    assert report["tolerance"]["abs"] == 1e-4  # recorded, never a pass gate
    run = run_feature_gates(model, batch, preset=RECORD_ONLY)
    assert run["numeric_pass"] is True    # errors are below even 1e-4 …
    assert run["passed"] is False         # … and it still never passes
    assert run["gate_admissible"] is False


def test_pd2_thresholds_stricter_than_record_reference(model, batch) -> None:
    """The normative pass threshold must stay below the 1e-4 record value
    for the fixture scale — no auto-widening path exists."""
    norm = check_rotation_equivariance(model, batch, preset=NORMATIVE)
    record = check_rotation_equivariance(model, batch, preset=RECORD_ONLY)
    assert norm["threshold"] < record["threshold"] or norm["scale"] > 10.0
    assert PRESETS[NORMATIVE].abs == 1e-6
    assert PRESETS[NORMATIVE].rel == 1e-5


def test_yaml_binding_refuses_widened_preset() -> None:
    contract, sha = load_campaign_feature_contract(DEFAULT_CAMPAIGN_YAML)
    resolved, explicit = resolve_preset(NORMATIVE, strict=True)
    assert explicit is True
    block = verify_preset_against_yaml(resolved, contract, sha)
    assert block["record_only_reference"] == 1e-4
    widened = type(resolved)(
        name="widened", rel=1e-4, abs=1e-4, source="test",
        relabel_max_abs=1e-6)
    with pytest.raises(PresetContractError):
        verify_preset_against_yaml(widened, contract, sha)
