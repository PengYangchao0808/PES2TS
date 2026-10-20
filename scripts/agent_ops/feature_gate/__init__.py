"""Feature-gate property-test harness (T1.1-T1.4 + T1.4b/T1.4c), v2.

Public names are re-exported lazily (PEP 562) so that running

    python -m scripts.agent_ops.feature_gate.equivariance_harness

does not trip runpy's "found in sys.modules" warning: importing the package
no longer imports the submodule eagerly.
"""

from typing import Any

_EXPORTS = (
    "DEFAULT_CAMPAIGN_YAML",
    "DEFAULT_PRESET",
    "EDGE_ALIGNED_TENSOR_KEYS",
    "FORWARD_INPUT_ALLOWLIST",
    "FORWARD_INPUT_ALLOWLIST_GROUPS",
    "FORWARD_INPUT_ALLOWLIST_SOURCE",
    "FORBIDDEN_FEATURE_PATTERNS",
    "GATE_REPORT_TESTS",
    "HARNESS_VERSION",
    "INTENT_OFF_MASK_CLASSES",
    "IllegalFeatureError",
    "NONSPATIAL_TENSOR_KEYS",
    "PRESETS",
    "PRESET_SOURCES",
    "PRESERVED_UNORDERED_CONTEXT_KEYS",
    "RECORD_ONLY_BANNER",
    "RELABEL_ABS_TOL",
    "PresetContractError",
    "PresetRequiredError",
    "TypedInvalidError",
    "build_gate_report",
    "check_feature_relabel_identity",
    "check_feature_reverse_change",
    "check_input_allowlist",
    "check_input_legality",
    "check_intent_off_mask",
    "check_nan_typed_invalid",
    "check_permutation_equivariance",
    "check_relabel_invariance",
    "check_reverse_sensitivity",
    "check_rotation_equivariance",
    "check_translation_invariance",
    "load_campaign_feature_contract",
    "resolve_preset",
    "run_feature_gates",
    "verify_preset_against_yaml",
)

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        from . import equivariance_harness as _h
        return getattr(_h, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
