"""R6 split firewall (Zone 2): policy loader plus pure guard functions.

Public API::

    from pes2ts_data.splits import (
        SplitPolicy, load_split_policy, validate_split_policy_doc,
        assert_no_family_leak, assert_sealed, check_split_frame,
        family_leaks, SplitViolation, SplitPolicyError,
    )
"""

from __future__ import annotations

from .policy import (
    BASE_SPLIT_NAMES,
    CONFIRM_N,
    CONFIRM_SPLIT,
    REQUIRED_LAYERS,
    SPLIT_POLICY_SCHEMA_ID,
    TEST_SET_NAMES,
    ConfirmSpec,
    LayerSpec,
    RulesSpec,
    SplitPolicy,
    SplitPolicyError,
    SplitViolation,
    TestSetSpec,
    assert_no_family_leak,
    assert_sealed,
    check_split_frame,
    default_split_policy_path,
    family_leaks,
    load_split_policy,
    validate_split_policy_doc,
)

__all__ = [
    "BASE_SPLIT_NAMES",
    "CONFIRM_N",
    "CONFIRM_SPLIT",
    "REQUIRED_LAYERS",
    "SPLIT_POLICY_SCHEMA_ID",
    "TEST_SET_NAMES",
    "ConfirmSpec",
    "LayerSpec",
    "RulesSpec",
    "SplitPolicy",
    "SplitPolicyError",
    "SplitViolation",
    "TestSetSpec",
    "assert_no_family_leak",
    "assert_sealed",
    "check_split_frame",
    "default_split_policy_path",
    "family_leaks",
    "load_split_policy",
    "validate_split_policy_doc",
]
