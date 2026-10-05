"""G2-AB1 WP-7: R6 split-firewall guards and holdout freeze discipline."""
from __future__ import annotations

import pytest

from pes2ts_core.flywheel import (
    assert_no_holdout_tuning, freeze_holdout, split_firewall_violations,
)


def assignments(**overrides):
    base = {"RXN_A1": "train", "RXN_A2": "train", "RXN_A3": "train",
            "RXN_B1": "valid", "RXN_C1": "test"}
    base.update(overrides)
    return base


FAMILIES = {"RXN_A1": "cycloaddition", "RXN_A2": "cycloaddition",
            "RXN_A3": "cycloaddition", "RXN_B1": "rearrangement", "RXN_C1": "radical"}


def test_reaction_family_never_crosses_splits():
    assert split_firewall_violations(assignments(), reaction_family=FAMILIES) == []
    broken = split_firewall_violations(assignments(RXN_A3="valid"), reaction_family=FAMILIES)
    assert broken[0]["kind"] == "FAMILY_CROSSES_SPLITS"
    assert set(broken[0]["splits"]) == {"train", "valid"}


def test_reverse_reaction_directions_share_one_split():
    pairs = [("RXN_B1", "RXN_B1_rev")]
    with_reverse = assignments(RXN_B1_rev="train")
    broken = split_firewall_violations(with_reverse, reaction_family={
        **FAMILIES, "RXN_B1_rev": "rearrangement"}, reverse_pairs=pairs)
    # Both couplings fire: the family straddles AND the reverse twin crosses.
    kinds = {violation["kind"] for violation in broken}
    assert "REVERSE_REACTION_CROSSES_SPLITS" in kinds
    assert "FAMILY_CROSSES_SPLITS" in kinds
    same = split_firewall_violations(assignments(RXN_B1_rev="valid"), reaction_family={
        **FAMILIES, "RXN_B1_rev": "rearrangement"}, reverse_pairs=pairs)
    assert same == []


def test_adjacent_frames_mapping_and_symmetry_copies_never_straddle():
    copies = [("RXN_A1:frame-3", "RXN_A1:frame-4"),
              ("RXN_A1:mapping-copy", "RXN_A1:symmetry-copy")]
    # Frame-level copies inherit their reaction's split; straddling assignments
    # are the leak the firewall exists to catch.
    with_frames = dict(assignments())
    with_frames["RXN_A1:frame-3"] = "train"
    with_frames["RXN_A1:frame-4"] = "test"
    broken = split_firewall_violations(with_frames, reaction_family=FAMILIES,
                                       frame_sibling_pairs=copies)
    assert broken[0]["kind"] == "COPIES_CROSS_SPLITS"
    safe = dict(with_frames)
    safe["RXN_A1:frame-4"] = "train"
    assert split_firewall_violations(safe, reaction_family=FAMILIES,
                                     frame_sibling_pairs=copies) == []


def test_invalid_split_is_reported_not_dropped():
    broken = split_firewall_violations(assignments(RXN_B1="holdout"),
                                       reaction_family=FAMILIES)
    assert broken[0]["kind"] == "INVALID_SPLIT"
    assert broken[0]["reaction_id"] == "RXN_B1"


def test_holdout_freezes_before_truth_and_tuning_stays_on_train():
    frozen = freeze_holdout({"cycloaddition": "train", "rearrangement": "valid",
                             "radical": "test"}, seed=42)
    assert frozen["frozen_before_truth_access"] is True
    assert frozen["tuning_allowed_on"] == ["cycloaddition"]
    assert_no_holdout_tuning(frozen, ["cycloaddition"])
    with pytest.raises(ValueError, match="HOLDOUT_TUNING_FORBIDDEN:radical"):
        assert_no_holdout_tuning(frozen, ["cycloaddition", "radical"])
    with pytest.raises(ValueError, match="HOLDOUT_TUNING_FORBIDDEN:rearrangement"):
        assert_no_holdout_tuning(frozen, ["rearrangement"])


def test_holdout_freeze_rejects_bad_splits_and_foreign_manifests():
    with pytest.raises(ValueError, match="INVALID_HOLDOUT_SPLIT"):
        freeze_holdout({"cycloaddition": "everything"}, seed=42)
    with pytest.raises(ValueError, match="INVALID_HOLDOUT_FREEZE_MANIFEST"):
        assert_no_holdout_tuning({"schema_version": "not_a_freeze"}, ["cycloaddition"])
