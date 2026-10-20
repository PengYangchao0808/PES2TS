"""Feature-gate property tests v2 (T1.1-T1.4) against the dummy E(n) model.

Deterministic: every stochastic element (batch geometry, model init, harness
rotations/permutations) is seeded.

Requires torch; when the interpreter has none (the repo's system ``python3``)
the module is skipped instead of erroring at collection.
"""

from __future__ import annotations

import warnings

import pytest

torch = pytest.importorskip(
    "torch",
    reason="feature-gate tests need torch; run with an interpreter that has "
           "it (e.g. /tmp/opencode/venv/bin/python)",
)

from scripts.agent_ops.feature_gate.dummy_equivariant_model import (  # noqa: E402
    INTENT_FIELD_KEYS,
    make_dummy_batch,
    make_dummy_model,
    make_intent_off_batch,
)
from scripts.agent_ops.feature_gate.equivariance_harness import (  # noqa: E402
    HARNESS_VERSION,
    PRESETS,
    RECORD_ONLY_BANNER,
    RELABEL_ABS_TOL,
    PresetRequiredError,
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
NORMATIVE = "30h_v1.1_normative"
ALL_CHECKS = {
    "input_legality",
    "input_allowlist",
    "nan_typed_invalid",
    "translation_invariance",
    "rotation_equivariance",
    "permutation_equivariance",
    "relabel_invariance",
    "reverse_sensitivity",
    "intent_off_mask",
}


def _fixture(seed: int = SEED):
    torch.manual_seed(seed)
    model = make_dummy_model(seed=seed)
    batch = make_dummy_batch(seed=seed)
    return model, batch


# ---------------------------------------------------------------------------
# v2 preset semantics: normative binding, record-only, fail-closed refusal
# ---------------------------------------------------------------------------

def test_preset_catalog_is_normative_plus_record_only() -> None:
    assert set(PRESETS) == {NORMATIVE, "r3_v3_record"}
    norm = PRESETS[NORMATIVE]
    assert norm.rel == 1e-5 and norm.abs == 1e-6
    assert norm.relabel_max_abs == 1e-6
    assert norm.gate_admissible is True and norm.record_only is False

    rec = PRESETS["r3_v3_record"]
    assert rec.abs == 1e-4 and rec.rel == 0.0
    assert rec.gate_admissible is False and rec.record_only is True
    assert rec.banner == RECORD_ONLY_BANNER
    for preset in PRESETS.values():
        assert preset.source  # every preset cites its yaml/doc provenance


@pytest.mark.parametrize("name", [NORMATIVE, "r3_v3_record"])
def test_explicit_preset_is_admissible_flagged(name: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        preset, explicit = resolve_preset(name)
    assert explicit is True
    if name == NORMATIVE:
        assert preset.gate_admissible is True
    else:
        assert preset.gate_admissible is False


def test_omitted_preset_refused_in_strict_mode() -> None:
    """v2 ORA-2: no preset -> refusal, not warn-and-pass."""
    with pytest.raises(PresetRequiredError):
        resolve_preset(None)


def test_legacy_fallback_is_opt_in_and_marks_inadmissible() -> None:
    with pytest.warns(UserWarning, match="gate_admissible=false"):
        preset, explicit = resolve_preset(None, strict=False)
    assert explicit is False and preset.name == NORMATIVE


@pytest.mark.parametrize("legacy", ["30h_v1.1", "r3_v3"])
def test_legacy_preset_names_rejected_with_rename_hint(legacy: str) -> None:
    with pytest.raises(KeyError, match="retired in harness v2"):
        resolve_preset(legacy)


def test_unknown_preset_rejected() -> None:
    with pytest.raises(KeyError):
        resolve_preset("no_such_preset")


def test_harness_version_is_v2() -> None:
    assert HARNESS_VERSION == "2"


# ---------------------------------------------------------------------------
# Correct model passes every gate, under BOTH admissible+record presets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_correct_model_passes_all_checks(preset: str) -> None:
    model, batch = _fixture()
    report = run_feature_gates(
        model, batch, preset=preset, seed=SEED,
        intent_off=make_intent_off_batch(batch))

    assert set(report["checks"]) == ALL_CHECKS
    failed = [k for k, r in report["checks"].items() if not r["passed"]]
    assert failed == [], f"unexpected failures under preset {preset}: {failed}"
    assert report["numeric_pass"] is True
    # v2 fail-closed: record-only preset can never be an overall pass
    assert report["gate_admissible"] is (preset == NORMATIVE)
    assert report["passed"] is (preset == NORMATIVE)
    for sub in report["checks"].values():
        assert sub["preset"] == preset
        assert sub["preset_explicit"] is True
        assert sub["harness_version"] == HARNESS_VERSION


def test_record_only_report_carries_banner_and_inadmissible() -> None:
    model, batch = _fixture()
    report = run_feature_gates(model, batch, preset="r3_v3_record", seed=SEED)
    assert report["banner"] == RECORD_ONLY_BANNER
    assert report["gate_admissible"] is False
    assert report["passed"] is False
    # individual check reports carry the banner too
    sub = report["checks"]["rotation_equivariance"]
    assert sub["banner"] == RECORD_ONLY_BANNER
    assert sub["record_only"] is True
    assert sub["gate_admissible"] is False


def test_non_strict_legacy_run_never_passes() -> None:
    model, batch = _fixture()
    with pytest.warns(UserWarning):
        report = run_feature_gates(model, batch, preset=None, strict=False,
                                   seed=SEED)
    assert report["preset_explicit"] is False
    assert report["gate_admissible"] is False
    assert report["passed"] is False
    assert report["numeric_pass"] is True  # diagnostics still informative


def test_strict_run_with_no_preset_raises() -> None:
    model, batch = _fixture()
    with pytest.raises(PresetRequiredError):
        run_feature_gates(model, batch, preset=None, seed=SEED)


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_correct_model_translation_error_is_tiny(preset: str) -> None:
    model, batch = _fixture()
    rep = check_translation_invariance(model, batch, preset=preset)
    assert rep["max_abs_delta"] <= RELABEL_ABS_TOL
    assert rep["median"] <= rep["p99"] <= rep["max"]


def test_relabel_is_bit_identical() -> None:
    model, batch = _fixture()
    rep = check_relabel_invariance(model, batch, preset=NORMATIVE)
    assert rep["passed"] and rep["bit_identical"] is True
    assert rep["max_abs_delta"] == 0.0
    assert rep["relabel_abs_tol"] == 1e-6


def test_relabel_blocked_without_endpoint_names() -> None:
    model, batch = _fixture()
    del batch["endpoint_names"]
    rep = check_relabel_invariance(model, batch, preset=NORMATIVE)
    assert rep["passed"] is False and rep["state"] == "blocked"


def test_reverse_sensitivity_changes_without_exact_negative() -> None:
    model, batch = _fixture()
    rep = check_reverse_sensitivity(model, batch, preset=NORMATIVE)
    assert rep["passed"] is True
    assert rep["outputs_changed"] is True
    assert rep["exact_negative_required"] is False


def test_reverse_blocked_without_signed_intent() -> None:
    model, batch = _fixture()
    del batch["signed_intent"]
    rep = check_reverse_sensitivity(model, batch, preset=NORMATIVE)
    assert rep["passed"] is False and rep["state"] == "blocked"


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
    report = run_feature_gates(
        model, batch, preset=preset, seed=SEED,
        intent_off=make_intent_off_batch(batch))

    failed = {k for k, r in report["checks"].items() if not r["passed"]}
    assert failed == expected_failures, (
        f"bug {bug_flags} under {preset}: expected {expected_failures}, "
        f"got {failed}"
    )
    assert report["passed"] is False


def test_raw_coords_bug_reports_large_translation_delta() -> None:
    model = make_dummy_model(seed=SEED, use_raw_coords=True)
    batch = make_dummy_batch(seed=SEED)
    rep = check_translation_invariance(model, batch, preset=NORMATIVE)
    # shift is ~O(1) A and enters u additively -> error is O(1), not O(1e-6)
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
    rep = check_nan_typed_invalid(Fabricator(), batch, preset=NORMATIVE)
    assert rep["passed"] is False
    assert rep["outcome"] == "fabricated_action"


def test_nan_gate_flags_untyped_exception() -> None:
    class RawErrorModel(torch.nn.Module):
        def forward(self, batch):
            raise RuntimeError("boom")  # no typed code attribute

    model, batch = _fixture()
    rep = check_nan_typed_invalid(RawErrorModel(), batch, preset=NORMATIVE)
    assert rep["passed"] is False
    assert rep["outcome"] == "untyped_exception"


# ---------------------------------------------------------------------------
# Forbidden-feature blacklist — every class of ReactionIntent L321-343
# ---------------------------------------------------------------------------

def test_clean_batch_is_legal() -> None:
    _, batch = _fixture()
    rep = check_input_legality(batch, preset=NORMATIVE)
    assert rep["passed"] is True and rep["violations"] == []
    assert rep["enforcement"] == "blacklist"


@pytest.mark.parametrize(
    "key, category",
    [
        # -- original classes ------------------------------------------------
        ("pos_future", "future_frames"),
        ("next_frame", "future_frames"),
        ("prev_frame", "future_frames"),
        ("pos_ts", "ts_coords"),
        ("true_ts", "ts_coords"),
        ("true_progress", "true_progress_arc"),
        ("remaining_arc", "true_progress_arc"),
        ("branch_id", "branch_id"),
        ("split", "split_family"),
        ("family", "split_family"),
        ("conflict_class", "split_family"),
        ("energy", "energy_gradient"),
        ("gradient", "energy_gradient"),
        ("near_ts", "near_ts_capped_flags"),
        ("capped", "near_ts_capped_flags"),
        ("d_eff", "near_ts_capped_flags"),
        # -- v2: complete forbidden classes (ORA-2 item 2) -------------------
        ("history", "history_hidden_state"),
        ("hidden_state", "history_hidden_state"),
        ("h_prev", "history_hidden_state"),
        ("lstm_state", "history_hidden_state"),
        ("x_a", "endpoint_geometry_XA_XB"),
        ("x_b", "endpoint_geometry_XA_XB"),
        ("pos_a", "endpoint_geometry_XA_XB"),
        ("endpoint_geom", "endpoint_geometry_XA_XB"),
        ("d_a_from_endpoint", "endpoint_derived_dA_dB_s"),
        ("d_b_from_endpoint", "endpoint_derived_dA_dB_s"),
        ("s_arc", "endpoint_derived_dA_dB_s"),
        ("d_endpoint", "endpoint_derived_dA_dB_s"),
        ("travel_direction", "travel_direction"),
        ("direction_token", "travel_direction"),
        ("is_reverse", "travel_direction"),
    ],
)
def test_forbidden_keys_are_detected(key: str, category: str) -> None:
    _, batch = _fixture()
    batch[key] = torch.tensor([1.0])
    rep = check_input_legality(batch, preset=NORMATIVE)
    assert rep["passed"] is False
    assert key in rep["violations"]
    assert category in rep["violation_categories"]


def test_every_forbidden_class_is_covered() -> None:
    """The parametrized table above exercises every registered class."""
    from scripts.agent_ops.feature_gate.equivariance_harness import (
        FORBIDDEN_FEATURE_PATTERNS,
    )

    covered = {
        "future_frames", "ts_coords", "true_progress_arc", "branch_id",
        "split_family", "energy_gradient", "near_ts_capped_flags",
        "history_hidden_state", "endpoint_geometry_XA_XB",
        "endpoint_derived_dA_dB_s", "travel_direction",
    }
    assert covered <= {c for c, _ in FORBIDDEN_FEATURE_PATTERNS}
    assert covered == {c for c, _ in FORBIDDEN_FEATURE_PATTERNS}


def test_intent_fields_are_not_blacklisted() -> None:
    """Legal signed intent is NOT forbidden (contract S6)."""
    _, batch = _fixture()
    for key in INTENT_FIELD_KEYS:
        batch[key] = torch.tensor([0.0])
    rep = check_input_legality(batch, preset=NORMATIVE)
    assert rep["passed"] is True, rep["violations"]


def test_illegality_raise_mode() -> None:
    from scripts.agent_ops.feature_gate.equivariance_harness import (
        IllegalFeatureError,
    )

    _, batch = _fixture()
    batch["near_ts"] = torch.tensor([0.5])
    with pytest.raises(IllegalFeatureError) as excinfo:
        check_input_legality(batch, raise_on_violation=True, preset=NORMATIVE)
    assert "near_ts" in excinfo.value.violations


def test_forbidden_patterns_extensible() -> None:
    _, batch = _fixture()
    batch["weird_leak"] = 1.0
    ok = check_input_legality(batch, preset=NORMATIVE)
    assert ok["passed"] is True
    flagged = check_input_legality(
        batch, preset=NORMATIVE,
        extra_patterns=(("custom_leak", ("weird_leak",)),))
    assert flagged["passed"] is False
    assert flagged["violation_categories"] == ["custom_leak"]
