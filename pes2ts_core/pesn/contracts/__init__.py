"""Predictor contract layer: inputs, predictions, and artifact declarations.

This sub-package defines typed, validated data contracts only.  It must not
import training, data-source, or backend modules; ``contracts.artifacts``
validates file/field structure without loading any model or dataset.
"""

from __future__ import annotations

from pes2ts_core.pesn.contracts.dual_ts import (
    SCHEMAS as DUAL_TS_SCHEMAS,
    DualEndpointTask,
    DualTSAction,
    DualTSActionKind,
    DualTSBundle,
    DualTSContractError,
    DualTSFeaturesSpec,
    DualTSTeacher,
    DualTSTransition,
    DualTSReward,
    SchemaInfo,
    canonical_endpoint_order,
    validate_record,
    ACTION_SCHEMA_VERSION as DUAL_TS_ACTION_SCHEMA_VERSION,
    BUNDLE_SCHEMA_VERSION as DUAL_TS_BUNDLE_SCHEMA_VERSION,
    FEATURES_SCHEMA_VERSION as DUAL_TS_FEATURES_SCHEMA_VERSION,
    REWARD_SCHEMA_VERSION as DUAL_TS_REWARD_SCHEMA_VERSION,
    TASK_SCHEMA_VERSION as DUAL_TS_TASK_SCHEMA_VERSION,
    TEACHER_SCHEMA_VERSION as DUAL_TS_TEACHER_SCHEMA_VERSION,
    TRANSITION_SCHEMA_VERSION as DUAL_TS_TRANSITION_SCHEMA_VERSION,
)


class PESNError(Exception):
    """Base type for every predictor-product error."""


__all__ = [
    "PESNError",
    # dual-endpoint TS teacher contracts (WP0a, contract doc v2)
    "DUAL_TS_ACTION_SCHEMA_VERSION",
    "DUAL_TS_BUNDLE_SCHEMA_VERSION",
    "DUAL_TS_FEATURES_SCHEMA_VERSION",
    "DUAL_TS_REWARD_SCHEMA_VERSION",
    "DUAL_TS_SCHEMAS",
    "DUAL_TS_TASK_SCHEMA_VERSION",
    "DUAL_TS_TEACHER_SCHEMA_VERSION",
    "DUAL_TS_TRANSITION_SCHEMA_VERSION",
    "DualEndpointTask",
    "DualTSAction",
    "DualTSActionKind",
    "DualTSBundle",
    "DualTSContractError",
    "DualTSFeaturesSpec",
    "DualTSTeacher",
    "DualTSTransition",
    "DualTSReward",
    "SchemaInfo",
    "canonical_endpoint_order",
    "validate_record",
]
