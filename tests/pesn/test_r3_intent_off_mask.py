"""R3 T1.4c — PD-6 intent-off mask completeness (five leak classes).

Gates:

* all five PD-6 classes are PRESENT in the intent-ON v3 features and are
  individually detectable;
* the full mask (the legal ``F2-intent-off`` arm) masks every class while
  PRESERVING the legal undirected chemical topology bit-identically;
* a partial mask that leaves ANY single class unmasked FAILS the harness
  check (parametrized over all five classes);
* the F3 slot superset is FIXED: F0/F1/F2/F3 arms differ only by zero
  masks — identical shapes, dtypes, and one shared model.

Fixture-level — ``not_admissible_until_gate_pass`` until G0+G1.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.pesn.contracts.key_atom_v3 import (  # noqa: E402
    FEATURE_LEVELS,
    FORWARD_VIEW_KEYS_V3,
    INTENT_OFF_ALL_V3,
    INTENT_OFF_CLASSES_V3,
    PRESERVED_UNORDERED_KEYS_V3,
    forward_view_v3,
)
from pes2ts_core.pesn.features.key_atom_graph_v3 import (  # noqa: E402
    apply_intent_off,
    build_sample_features_v3,
)
from pes2ts_core.pesn.models.key_atom_egnn_v3 import StarEGNNV3  # noqa: E402
from scripts.agent_ops.feature_gate.equivariance_harness import (  # noqa: E402
    PRESERVED_UNORDERED_CONTEXT_KEYS,
    check_intent_off_mask,
)
from scripts.agent_ops.feature_gate.v3_fixtures import (  # noqa: E402
    make_intent_off_batch,
    make_sn2_sample,
    make_v3_gate_batch,
    make_v3_batch,
)

PRESET = "30h_v1.1_normative"
NORMATIVE_ON = make_v3_gate_batch(n_graphs=2)


def _on_sample():
    return build_sample_features_v3(make_sn2_sample("forward"))


def test_five_classes_enumerated() -> None:
    assert len(INTENT_OFF_CLASSES_V3) == 5
    assert set(INTENT_OFF_ALL_V3) == {
        "ordered_source_target", "signed_graph_edits", "intent_role",
        "directed_cv_descriptors", "implicit_direction_encodings"}


@pytest.mark.parametrize("cls,keys", INTENT_OFF_CLASSES_V3)
def test_each_class_present_and_nonzero_in_intent_on(cls, keys) -> None:
    on = _on_sample()
    for key in keys:
        assert key in on, (cls, key)
        value = on[key]
        assert isinstance(value, np.ndarray)
        assert value.any(), f"{cls}/{key} is all-zero in the intent-ON build"


def test_full_mask_masks_every_class_and_preserves_context() -> None:
    on = _on_sample()
    off = apply_intent_off(on, classes=INTENT_OFF_ALL_V3)
    for cls, keys in INTENT_OFF_CLASSES_V3:
        for key in keys:
            assert not np.any(off[key]), f"{cls}/{key} not masked"
    for key in PRESERVED_UNORDERED_KEYS_V3:
        assert key in on
        assert np.array_equal(on[key], off[key]), key


def test_harness_pd6_check_passes_for_full_mask() -> None:
    batch = NORMATIVE_ON
    off = forward_view_v3(make_intent_off_batch({"n_graphs": 2}, mode="full"))
    report = check_intent_off_mask(batch, off, preset=PRESET)
    assert report["state"] == "pass", report
    assert report["class_coverage"] == "5/5"
    assert report["preserved_ok"] is True


@pytest.mark.parametrize(
    "leaky_cls", [cls for cls, _ in INTENT_OFF_CLASSES_V3])
def test_partial_mask_fails_on_each_leaking_class(leaky_cls) -> None:
    """Leaving ANY single class unmasked must FAIL (class detectability)."""
    batch = NORMATIVE_ON
    leaky = forward_view_v3(make_intent_off_batch(
        {"n_graphs": 2}, mode="partial", unmask_classes=(leaky_cls,)))
    report = check_intent_off_mask(batch, leaky, preset=PRESET)
    assert report["state"] == "fail", (leaky_cls, report)
    assert report["classes_unmasked"] == [leaky_cls]
    assert report["classes"][leaky_cls]["masked"] is False


def test_nuke_mask_fails_because_context_destroyed() -> None:
    batch = NORMATIVE_ON
    nuked = forward_view_v3(make_intent_off_batch({"n_graphs": 2}, mode="nuke"))
    report = check_intent_off_mask(batch, nuked, preset=PRESET)
    assert report["state"] == "fail", report
    assert report["preserved_ok"] is False


def test_undirected_context_keys_preserved_by_pd6_definition() -> None:
    """The v3 preserved keys are a subset of the harness PD-6 preserved
    topology set (extra names like ``elements``/``adjacency`` are aliases
    checked only when present)."""
    for key in ("pos", "z", "edge_index", "graph_index", "abs_delta",
                "is_changed", "degK"):
        assert key in PRESERVED_UNORDERED_CONTEXT_KEYS
    assert set(PRESERVED_UNORDERED_KEYS_V3) <= set(
        PRESERVED_UNORDERED_CONTEXT_KEYS)
    assert set(PRESERVED_UNORDERED_KEYS_V3) <= set(FORWARD_VIEW_KEYS_V3)


@pytest.mark.parametrize("level", FEATURE_LEVELS)
def test_slot_superset_fixed_shapes_per_level(level) -> None:
    """F0/F1/F2/F3: same keys, shapes and dtypes — mask-only differences."""
    f3 = _on_sample()
    lv = build_sample_features_v3(make_sn2_sample("forward"), level=level)
    assert set(FORWARD_VIEW_KEYS_V3) <= set(lv)
    for key in FORWARD_VIEW_KEYS_V3:
        if key == "endpoint_names":
            continue
        assert lv[key].shape == f3[key].shape, key
        assert lv[key].dtype == f3[key].dtype, key


def test_level_masks_are_cumulative_zero_masks() -> None:
    f3 = _on_sample()
    f0 = build_sample_features_v3(make_sn2_sample("forward"), level="F0")
    # F0: no intent features, no RBF12, no unordered bond context
    assert not f0["b_src"].any() and not f0["b_tgt"].any()
    assert not f0["delta_b_intent"].any()
    assert not f0["intent_role"].any()
    assert not f0["cv_target_delta"].any()
    assert not f0["signed_intent"].any()
    assert not f0["abs_delta"].any() and not f0["is_changed"].any()
    rbf = slice(0, 12)
    assert not f0["edge_attr"][:, rbf].any()
    assert f3["edge_attr"][:, rbf].any()   # F3 keeps the RBF(12) slots
    # structural keys identical at every level
    for key in ("z", "pos", "edge_index", "graph_index", "key_index"):
        assert np.array_equal(f0[key], f3[key]), key
    f1 = build_sample_features_v3(make_sn2_sample("forward"), level="F1")
    assert f1["is_key"].any() and f1["degK"].any()
    assert not f1["delta_b_intent"].any()
    f2 = build_sample_features_v3(make_sn2_sample("forward"), level="F2")
    assert f2["delta_b_intent"].any() and not f2["edge_attr"][:, rbf].any()


def test_one_shared_model_runs_every_level() -> None:
    """No arm may add parameters: ONE model instance evaluates all arms."""
    torch.manual_seed(17)
    model = StarEGNNV3()
    n_params = model.n_parameters
    for level in FEATURE_LEVELS:
        batch = forward_view_v3(make_v3_batch(n_graphs=1, level=level))
        out = model(batch)
        assert out["u"].shape[0] > 0
        assert model.n_parameters == n_params


def test_intent_off_batch_still_runs_model() -> None:
    torch.manual_seed(17)
    model = StarEGNNV3()
    off = forward_view_v3(make_intent_off_batch({"n_graphs": 1}, mode="full"))
    out = model(off)
    assert bool(torch.isfinite(out["u"]).all())
