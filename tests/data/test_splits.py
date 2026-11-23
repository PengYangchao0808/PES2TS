"""Tests for the R6 split firewall (``pes2ts_data.splits`` + config YAML)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pes2ts_data.splits import (
    CONFIRM_N,
    CONFIRM_SPLIT,
    REQUIRED_LAYERS,
    SPLIT_POLICY_SCHEMA_ID,
    TEST_SET_NAMES,
    SplitPolicy,
    SplitPolicyError,
    SplitViolation,
    assert_no_family_leak,
    assert_sealed,
    check_split_frame,
    default_split_policy_path,
    family_leaks,
    load_split_policy,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def test_load_real_split_policy():
    path = default_split_policy_path()
    assert path == REPO_ROOT / "config" / "data" / "splits.yaml"
    policy = load_split_policy(path)
    assert isinstance(policy, SplitPolicy)
    assert policy.schema_id == SPLIT_POLICY_SCHEMA_ID
    assert policy.policy == "family_holdout_v1"
    assert policy.confirm.n == CONFIRM_N == 2000
    assert policy.confirm.sealed is True
    assert policy.test_set_names == TEST_SET_NAMES
    assert policy.layer_names == REQUIRED_LAYERS
    assert policy.sealed_split_names == frozenset({CONFIRM_SPLIT})
    assert policy.layer_of("family_ood") == "L3"
    assert policy.layer_of("confirm") == "L3"
    assert policy.is_sealed("confirm")
    assert not policy.is_sealed("id_dev")


def test_load_split_policy_with_explicit_root_path(tmp_path):
    src = default_split_policy_path()
    target = tmp_path / "splits.yaml"
    target.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    assert load_split_policy(target).policy == "family_holdout_v1"


def _policy_doc() -> dict:
    return yaml.safe_load(default_split_policy_path().read_text(encoding="utf-8"))


def _write_yaml(tmp_path: Path, doc) -> Path:
    target = tmp_path / "splits_malformed.yaml"
    target.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    return target


def test_malformed_policies_are_typed_rejections(tmp_path):
    doc = _policy_doc()
    del doc["layers"]["L4"]
    with pytest.raises(SplitPolicyError) as excinfo:
        load_split_policy(_write_yaml(tmp_path, doc))
    assert excinfo.value.code == "POLICY_LAYER_MISSING"

    doc = _policy_doc()
    doc["confirm"]["n"] = 1999
    with pytest.raises(SplitPolicyError) as excinfo:
        load_split_policy(_write_yaml(tmp_path, doc))
    assert excinfo.value.code == "POLICY_CONFIRM_INVALID"

    doc = _policy_doc()
    doc["confirm"]["sealed"] = False
    with pytest.raises(SplitPolicyError) as excinfo:
        load_split_policy(_write_yaml(tmp_path, doc))
    assert excinfo.value.code == "POLICY_CONFIRM_INVALID"

    doc = _policy_doc()
    doc["test_sets"]["leaky_set"] = dict(doc["test_sets"]["id_dev"])
    with pytest.raises(SplitPolicyError) as excinfo:
        load_split_policy(_write_yaml(tmp_path, doc))
    assert excinfo.value.code == "POLICY_TEST_SET_UNKNOWN"

    doc = _policy_doc()
    doc["schema"] = "pes2ts_split_policy_v0"
    with pytest.raises(SplitPolicyError) as excinfo:
        load_split_policy(_write_yaml(tmp_path, doc))
    assert excinfo.value.code == "POLICY_SCHEMA_MISMATCH"

    doc = _policy_doc()
    doc["policy"] = "random_split_v1"
    with pytest.raises(SplitPolicyError) as excinfo:
        load_split_policy(_write_yaml(tmp_path, doc))
    assert excinfo.value.code == "POLICY_NAME_MISMATCH"

    bad_yaml = tmp_path / "splits_broken.yaml"
    bad_yaml.write_text("schema: [unclosed\n", encoding="utf-8")
    with pytest.raises(SplitPolicyError) as excinfo:
        load_split_policy(bad_yaml)
    assert excinfo.value.code == "POLICY_PARSE_ERROR"

    with pytest.raises(SplitPolicyError) as excinfo:
        load_split_policy(tmp_path / "does_not_exist.yaml")
    assert excinfo.value.code == "POLICY_FILE_MISSING"


# ---------------------------------------------------------------------------
# Family leakage
# ---------------------------------------------------------------------------

FAMILIES = {
    "RXN_A1": "cycloaddition",
    "RXN_A2": "cycloaddition",
    "RXN_B1": "rearrangement",
    "RXN_C1": "radical",
    "RXN_C2": "radical",
}


def test_family_leak_clean_split_passes():
    assert_no_family_leak(["RXN_A1", "RXN_A2", "RXN_B1"], ["RXN_C1", "RXN_C2"],
                          FAMILIES)
    assert family_leaks(["RXN_A1"], ["RXN_C1"], FAMILIES) == {}


def test_family_leak_detected_with_family_evidence():
    with pytest.raises(SplitViolation) as excinfo:
        assert_no_family_leak(["RXN_A1", "RXN_A2"], ["RXN_A1", "RXN_C1"], FAMILIES)
    # Same id on both sides is reported first, as the sharper violation.
    assert excinfo.value.code == "REACTION_LEAK"

    with pytest.raises(SplitViolation) as excinfo:
        assert_no_family_leak(["RXN_A1"], ["RXN_A2"], FAMILIES)
    assert excinfo.value.code == "FAMILY_LEAK"
    assert excinfo.value.families is not None
    assert "cycloaddition" in excinfo.value.families

    leaks = family_leaks(["RXN_A1"], ["RXN_A2", "RXN_C1"], FAMILIES)
    assert set(leaks) == {"cycloaddition"}
    assert set(leaks["cycloaddition"]) == {"RXN_A1", "RXN_A2"}


def test_unlabelled_family_is_typed_rejection_not_silent_drop():
    with pytest.raises(SplitViolation) as excinfo:
        assert_no_family_leak(["RXN_A1"], ["RXN_UNKNOWN"], FAMILIES)
    assert excinfo.value.code == "FAMILY_LABEL_MISSING"

    # Explicit opt-out: unlabelled ids are not silently treated as leaking.
    assert_no_family_leak(["RXN_A1"], ["RXN_UNKNOWN"], FAMILIES,
                          require_labels=False)


def test_family_guard_rejects_bad_values():
    with pytest.raises(SplitViolation) as excinfo:
        assert_no_family_leak(None, ["RXN_C1"], FAMILIES)
    assert excinfo.value.code == "BAD_SPLIT_VALUE"

    with pytest.raises(SplitViolation) as excinfo:
        assert_no_family_leak(["RXN_A1"], [object()], FAMILIES)
    assert excinfo.value.code == "BAD_SPLIT_VALUE"


# ---------------------------------------------------------------------------
# Sealed confirm set
# ---------------------------------------------------------------------------

def test_sealed_confirm_set_is_refused():
    with pytest.raises(SplitViolation) as excinfo:
        assert_sealed(CONFIRM_SPLIT)
    assert excinfo.value.code == "SEALED_SPLIT_ACCESS"

    # The explicit final-run escape hatch is allowed and silent.
    assert_sealed(CONFIRM_SPLIT, allow_confirm=True)

    policy = load_split_policy()
    with pytest.raises(SplitViolation) as excinfo:
        assert_sealed("confirm", policy=policy)
    assert excinfo.value.code == "SEALED_SPLIT_ACCESS"
    assert_sealed("confirm", allow_confirm=True, policy=policy)

    # Ordinary evaluation suites are not sealed.
    assert_sealed("id_dev", policy=policy)
    assert_sealed("scaffold_ood", policy=policy)

    with pytest.raises(SplitViolation) as excinfo:
        assert_sealed("made_up_split")
    assert excinfo.value.code == "UNKNOWN_SPLIT"


# ---------------------------------------------------------------------------
# Split frame validation
# ---------------------------------------------------------------------------

def test_check_split_frame_normalises_and_passes():
    frame = check_split_frame({
        "train": ["RXN_A1", "RXN_A2"],
        "valid": ["RXN_B1"],
        "test": ["RXN_C1"],
    })
    assert frame["train"] == frozenset({"RXN_A1", "RXN_A2"})
    assert frame["valid"] == frozenset({"RXN_B1"})
    assert frame["test"] == frozenset({"RXN_C1"})


def test_check_split_frame_rejects_missing_overlap_and_bad_values():
    with pytest.raises(SplitViolation) as excinfo:
        check_split_frame({"train": ["a"], "valid": ["b"]})
    assert excinfo.value.code == "MISSING_SPLIT_KEY"

    with pytest.raises(SplitViolation) as excinfo:
        check_split_frame({"train": ["a"], "valid": ["a"], "test": ["c"]})
    assert excinfo.value.code == "SPLIT_OVERLAP"

    with pytest.raises(SplitViolation) as excinfo:
        check_split_frame({"train": [], "valid": ["b"], "test": ["c"]})
    assert excinfo.value.code == "EMPTY_SPLIT"

    with pytest.raises(SplitViolation) as excinfo:
        check_split_frame({"train": ["a"], "valid": None, "test": ["c"]})
    assert excinfo.value.code == "BAD_SPLIT_VALUE"

    with pytest.raises(SplitViolation) as excinfo:
        check_split_frame(["not", "a", "mapping"])
    assert excinfo.value.code == "MALFORMED_SPLIT_FRAME"


def test_check_split_frame_checks_sealed_confirm_overlap():
    with pytest.raises(SplitViolation) as excinfo:
        check_split_frame({
            "train": ["a", "b"],
            "valid": ["c"],
            "test": ["d"],
            "confirm": ["b"],
        })
    assert excinfo.value.code == "SPLIT_OVERLAP"
