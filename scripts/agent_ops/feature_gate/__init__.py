"""Feature-gate property-test harness (T1.1-T1.4) — dry-run lane B."""

from .equivariance_harness import (  # noqa: F401
    DEFAULT_PRESET,
    FORBIDDEN_FEATURE_PATTERNS,
    IllegalFeatureError,
    PRESETS,
    PRESET_SOURCES,
    RELABEL_ABS_TOL,
    TypedInvalidError,
    check_input_legality,
    check_nan_typed_invalid,
    check_permutation_equivariance,
    check_relabel_invariance,
    check_reverse_sensitivity,
    check_rotation_equivariance,
    check_translation_invariance,
    resolve_preset,
    run_feature_gates,
)
