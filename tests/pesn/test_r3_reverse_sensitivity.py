"""R3 T1.2 — reverse sensitivity (TRUE Reverse ⇒ signed features change).

Contract semantics (ReactionIntent Contract v1 §3.2 operation B): reversing
the reaction goal swaps ``G_src/G_tgt``, negates the signed intent edits,
swaps the intent-role channels, negates the directed CV descriptors and
flips the derived direction sign, while the unsigned/undirected context is
preserved.  Exact-negative label tangents are NOT required at finite step
(diagnostic only).  Fixture-level — ``not_admissible_until_gate_pass``.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.pesn.models.key_atom_egnn_v3 import StarEGNNV3  # noqa: E402
from scripts.agent_ops.feature_gate.equivariance_harness import (  # noqa: E402
    PRESERVED_UNORDERED_CONTEXT_KEYS,
    check_feature_reverse_change,
    check_reverse_sensitivity,
)
from scripts.agent_ops.feature_gate.v3_fixtures import (  # noqa: E402
    build_fixture_features,
    make_sn2_sample,
    make_sn2_teacher,
    make_v3_gate_batch,
)

PRESET = "30h_v1.1_normative"


@pytest.fixture()
def pair():
    fwd = build_fixture_features("forward")
    rev = build_fixture_features("reverse")
    return fwd, rev


def test_true_reverse_signed_features_change(pair) -> None:
    fwd, rev = pair
    report = check_feature_reverse_change(fwd, rev, preset=PRESET)
    assert report["passed"], report
    assert report["exact_negative_required"] is False


def test_true_reverse_swaps_ordered_endpoints_exactly(pair) -> None:
    fwd, rev = pair
    assert np.array_equal(fwd["b_src"], rev["b_tgt"])
    assert np.array_equal(fwd["b_tgt"], rev["b_src"])


def test_true_reverse_negates_signed_edits_exactly(pair) -> None:
    fwd, rev = pair
    assert np.array_equal(fwd["delta_b_intent"], -rev["delta_b_intent"])
    assert np.array_equal(fwd["abs_delta"], rev["abs_delta"])
    assert np.array_equal(fwd["is_changed"], rev["is_changed"])


def test_true_reverse_swaps_intent_role_channels(pair) -> None:
    fwd, rev = pair
    role_f, role_r = fwd["intent_role"], rev["intent_role"]
    assert np.array_equal(role_f[:, 0], role_r[:, 1])   # breaking <-> forming
    assert np.array_equal(role_f[:, 1], role_r[:, 0])
    assert np.array_equal(role_f[:, 2], role_r[:, 2])   # spectator invariant


def test_true_reverse_negates_directed_cv_and_sign(pair) -> None:
    fwd, rev = pair
    assert np.array_equal(fwd["cv_target_delta"], -rev["cv_target_delta"])
    assert float(fwd["signed_intent"][0]) == 1.0
    assert float(rev["signed_intent"][0]) == -1.0


def test_true_reverse_preserves_undirected_context(pair) -> None:
    fwd, rev = pair
    for key in PRESERVED_UNORDERED_CONTEXT_KEYS:
        if key in fwd:
            assert np.array_equal(fwd[key], rev[key]), key


def test_reverse_involution_restores_original_features() -> None:
    """reverse(reverse(x)) == x at FEATURE level (label involution T0.2
    companion; ids differ, numeric features must not)."""
    from pes2ts_core.pesn.features.key_atom_graph_v3 import (
        build_sample_features_v3,
    )

    fwd = build_fixture_features("forward")
    rev_sample = make_sn2_sample("reverse")
    back = build_sample_features_v3(
        _with_intent(rev_sample, rev_sample.intent.reversed_intent()))
    assert check_feature_reverse_change(
        fwd, build_fixture_features("reverse"), preset=PRESET)["passed"]
    from pes2ts_core.pesn.contracts.key_atom_v3 import FORWARD_VIEW_KEYS_V3

    # involution holds on the INPUT features (supervision legitimately
    # depends on the intent direction and is excluded)
    for key in FORWARD_VIEW_KEYS_V3:
        if key == "endpoint_names":
            continue
        value = fwd[key]
        if isinstance(value, np.ndarray):
            assert np.array_equal(value, back[key]), key


def _with_intent(sample, intent):
    import dataclasses

    return dataclasses.replace(sample, intent=intent)


def test_reverse_model_outputs_change_no_exact_negative() -> None:
    torch.manual_seed(17)
    model = StarEGNNV3()
    batch = make_v3_gate_batch(n_graphs=2)
    report = check_reverse_sensitivity(model, batch, preset=PRESET)
    assert report["passed"], report
    assert report["outputs_changed"] is True
    assert report["exact_negative_required"] is False


def test_tangent_diagnostic_approximately_opposite_not_exact() -> None:
    """Small-step label tangents of the two intents are approximately
    opposite — a DIAGNOSTIC, never a hard exact-negative requirement."""
    y_fwd = np.asarray(make_sn2_teacher("forward"), dtype=np.float64)
    y_rev = np.asarray(make_sn2_teacher("reverse", asymmetric=True),
                       dtype=np.float64)
    cos = float((y_fwd * y_rev).sum()
                / (np.linalg.norm(y_fwd) * np.linalg.norm(y_rev)))
    assert cos < -0.9, cos                       # approximately opposite
    assert not np.array_equal(y_fwd, -y_rev)     # NOT exact negatives
