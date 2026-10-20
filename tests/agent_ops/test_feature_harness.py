"""Feature-gate property tests (T1.1-T1.4) against the dummy E(n) model.

Deterministic: every stochastic element (batch geometry, model init, harness
rotations/permutations) is seeded.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from scripts.agent_ops.feature_gate.dummy_equivariant_model import (
    make_dummy_batch,
    make_dummy_model,
)
from scripts.agent_ops.feature_gate.equivariance_harness import (
    PRESETS,
    RELABEL_ABS_TOL,
    TypedInvalidError,
    check_input_legality,
    check_nan_typed_invalid,
    check_relabel_invariance,
    check_reverse_sensitivity,
    check_translation_invariance,
    resolve_preset,
    run_feature_gates,
)

SEED = 1234
ALL_CHECKS = {
    "input_legality",
    "nan_typed_invalid",
    "translation_invariance",
    "rotation_equivariance",
    "permutation_equivariance",
    "relabel_invariance",
    "reverse_sensitivity",
}


def _fixture(seed: int = SEED):
    torch.manual_seed(seed)
    model = make_dummy_model(seed=seed)
    batch = make_dummy_batch(seed=seed)
    return model, batch


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

def test_preset_catalog_matches_open_decision() -> None:
    """Both disagreeing docs are registered; none is a silent default."""
    assert set(PRESETS) == {"30h_v1.1", "r3_v3"}
    assert PRESETS["30h_v1.1"].rel == 1e-5 and PRESETS["30h_v1.1"].abs == 1e-6
    assert PRESETS["r3_v3"].rel == 0.0 and PRESETS["r3_v3"].abs == 1e-4
    for preset in PRESETS.values():
        assert preset.source  # every preset names its (conflicting) source


def test_explicit_preset_no_warning_and_records_explicit_flag() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        preset, explicit = resolve_preset("r3_v3")
    assert preset.name == "r3_v3" and explicit is True


def test_omitted_preset_warns_listing_both_candidates() -> None:
    with pytest.warns(UserWarning, match="30h_v1\\.1") as record:
        preset, explicit = resolve_preset(None)
    assert explicit is False and preset.name == "30h_v1.1"
    msg = str(record[0].message)
    assert "r3_v3" in msg and "1e-4" in msg  # warning lists BOTH docs' lines


def test_unknown_preset_rejected() -> None:
    with pytest.raises(KeyError):
        resolve_preset("no_such_preset")


# ---------------------------------------------------------------------------
# Correct model passes every gate, under BOTH presets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_correct_model_passes_all_gates(preset: str) -> None:
    model, batch = _fixture()
    report = run_feature_gates(model, batch, preset=preset, seed=SEED)

    assert set(report["checks"]) == ALL_CHECKS
    failed = [k for k, r in report["checks"].items() if not r["passed"]]
    assert failed == [], f"unexpected failures under preset {preset}: {failed}"
    assert report["passed"] is True

    # every sub-report records the used preset (never a silent default)
    for sub in report["checks"].values():
        assert sub["preset"] == preset
        assert sub["preset_explicit"] is True


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_correct_model_translation_error_is_tiny(preset: str) -> None:
    model, batch = _fixture()
    rep = check_translation_invariance(model, batch, preset=preset)
    assert rep["max_abs_delta"] <= RELABEL_ABS_TOL
    assert rep["median"] <= rep["p99"] <= rep["max"]


def test_relabel_is_bit_identical() -> None:
    model, batch = _fixture()
    rep = check_relabel_invariance(model, batch, preset="30h_v1.1")
    assert rep["passed"] and rep["bit_identical"] is True
    assert rep["max_abs_delta"] == 0.0


def test_reverse_sensitivity_changes_without_exact_negative() -> None:
    model, batch = _fixture()
    rep = check_reverse_sensitivity(model, batch, preset="30h_v1.1")
    assert rep["passed"] is True
    assert rep["outputs_changed"] is True
    assert rep["exact_negative_required"] is False


# ---------------------------------------------------------------------------
# Bug modes fail exactly the expected gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "bug_flags, expected_failures",
    [
        ({"use_raw_coords": True}, {"translation_invariance"}),
        ({"use_atom_index": True}, {"permutation_equivariance"}),
        ({"use_endpoint_names": True}, {"relabel_invariance"}),
        ({"ignore_intent": True}, {"reverse_sensitivity"}),
    ],
    ids=["raw_coords", "atom_index", "endpoint_names", "ignore_intent"],
)
@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_bug_mode_fails_exactly_expected_gate(
    bug_flags: dict, expected_failures: set[str], preset: str
) -> None:
    model = make_dummy_model(seed=SEED, **bug_flags)
    batch = make_dummy_batch(seed=SEED)
    report = run_feature_gates(model, batch, preset=preset, seed=SEED)

    failed = {k for k, r in report["checks"].items() if not r["passed"]}
    assert failed == expected_failures, (
        f"bug {bug_flags} under {preset}: expected {expected_failures}, "
        f"got {failed}"
    )
    assert report["passed"] is False


def test_raw_coords_bug_reports_large_translation_delta() -> None:
    model = make_dummy_model(seed=SEED, use_raw_coords=True)
    batch = make_dummy_batch(seed=SEED)
    rep = check_translation_invariance(model, batch, preset="r3_v3")
    # shift is ~O(1) A and enters u additively -> error is O(1), not O(1e-4)
    assert rep["max_abs_delta"] > 0.1


# ---------------------------------------------------------------------------
# NaN -> typed invalid, never a fabricated action
# ---------------------------------------------------------------------------

def test_nan_input_raises_typed_invalid_directly() -> None:
    model, batch = _fixture()
    batch["pos"] = batch["pos"].clone()
    batch["pos"][0, 0] = float("nan")
    with pytest.raises(TypedInvalidError) as excinfo:
        model(batch)
    assert excinfo.value.code == "NAN_INPUT"


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_nan_gate_passes_for_correct_model(preset: str) -> None:
    model, batch = _fixture()
    rep = check_nan_typed_invalid(model, batch, preset=preset)
    assert rep["passed"] is True
    assert rep["outcome"] == "typed_invalid_raised"
    assert rep["code"] == "NAN_INPUT"


def test_nan_gate_fails_for_fabricating_model() -> None:
    """A model that swallows NaN and returns an action must be flagged."""

    class Fabricator(torch.nn.Module):
        def forward(self, batch):
            return {"u": torch.zeros_like(batch["pos"][:2])}

    model, batch = _fixture()
    rep = check_nan_typed_invalid(Fabricator(), batch, preset="30h_v1.1")
    assert rep["passed"] is False
    assert rep["outcome"] == "fabricated_action"


def test_nan_gate_flags_untyped_exception() -> None:
    class RawErrorModel(torch.nn.Module):
        def forward(self, batch):
            raise RuntimeError("boom")  # no typed code attribute

    model, batch = _fixture()
    rep = check_nan_typed_invalid(RawErrorModel(), batch, preset="30h_v1.1")
    assert rep["passed"] is False
    assert rep["outcome"] == "untyped_exception"


# ---------------------------------------------------------------------------
# Forbidden-feature legality
# ---------------------------------------------------------------------------

def test_clean_batch_is_legal() -> None:
    _, batch = _fixture()
    rep = check_input_legality(batch, preset="30h_v1.1")
    assert rep["passed"] is True and rep["violations"] == []


@pytest.mark.parametrize(
    "key, category",
    [
        ("pos_future", "future_frames"),
        ("next_frame", "future_frames"),
        ("pos_ts", "ts_coords"),
        ("true_ts", "ts_coords"),
        ("true_progress", "true_progress_arc"),
        ("remaining_arc", "true_progress_arc"),
        ("branch_id", "branch_id"),
        ("split", "split_family"),
        ("family", "split_family"),
        ("energy", "energy_gradient"),
        ("gradient", "energy_gradient"),
        ("near_ts", "near_ts_capped_flags"),
        ("capped", "near_ts_capped_flags"),
    ],
)
def test_forbidden_keys_are_detected(key: str, category: str) -> None:
    _, batch = _fixture()
    batch[key] = torch.tensor([1.0])
    rep = check_input_legality(batch, preset="30h_v1.1")
    assert rep["passed"] is False
    assert key in rep["violations"]
    assert category in rep["violation_categories"]


def test_illegality_raise_mode() -> None:
    from scripts.agent_ops.feature_gate.equivariance_harness import (
        IllegalFeatureError,
    )

    _, batch = _fixture()
    batch["near_ts"] = torch.tensor([0.5])
    with pytest.raises(IllegalFeatureError) as excinfo:
        check_input_legality(batch, raise_on_violation=True)
    assert "near_ts" in excinfo.value.violations


# ---------------------------------------------------------------------------
# Reports carry the open-decision provenance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_reports_record_preset_and_source_provenance(preset: str) -> None:
    model, batch = _fixture()
    report = run_feature_gates(model, batch, preset=preset, seed=SEED)
    assert report["preset"] == preset
    assert report["preset_explicit"] is True
    assert report["preset_source"]
    if preset == "30h_v1.1":
        assert "ABSENT" in report["preset_source"]
