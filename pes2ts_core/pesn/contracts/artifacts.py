"""Artifact declarations and validators for the predictor product.

Covers the manifest layer of the teacher dataset, labels, split, feature
spec, training run, and model bundle contracts.  Validation is structural
(schema version, required/unknown keys, forbidden keys, hash format) — this
module never loads datasets, weights, or training state.
"""

from __future__ import annotations

import re
from typing import Final, Mapping

from pes2ts_core.pesn.contracts import PESNError
from pes2ts_core.pesn.contracts.inputs import (
    COORDINATE_SPEC_SCHEMA_VERSION,
    INPUT_SCHEMA_VERSION,
    TASK_GRAPH_SCHEMA_VERSION,
)
from pes2ts_core.pesn.contracts.predictions import PREDICTION_SCHEMA_VERSION

DATASET_SCHEMA_VERSION: Final[str] = "pesn_irc_teacher_dataset_v1"
LABELS_SCHEMA_VERSION: Final[str] = "pesn_irc_teacher_labels_v1"
SPLIT_SCHEMA_VERSION: Final[str] = "pesn_irc_split_v1"
FEATURE_SPEC_SCHEMA_VERSION: Final[str] = "pesn_feature_spec_v1"
LABEL_SPEC_SCHEMA_VERSION: Final[str] = "pesn_irc_teacher_labelspec_v1"
RUN_SCHEMA_VERSION: Final[str] = "pesn_training_run_v1"
BUNDLE_SCHEMA_VERSION: Final[str] = "pesn_model_bundle_v1"
NORMALIZATION_SCHEMA_VERSION: Final[str] = "pesn_normalization_v1"

SCHEMA_VERSIONS: Final[dict[str, str]] = {
    "input": INPUT_SCHEMA_VERSION,
    "task_graph": TASK_GRAPH_SCHEMA_VERSION,
    "coordinate_spec": COORDINATE_SPEC_SCHEMA_VERSION,
    "prediction": PREDICTION_SCHEMA_VERSION,
    "dataset": DATASET_SCHEMA_VERSION,
    "labels": LABELS_SCHEMA_VERSION,
    "split": SPLIT_SCHEMA_VERSION,
    "feature_spec": FEATURE_SPEC_SCHEMA_VERSION,
    "label_spec": LABEL_SPEC_SCHEMA_VERSION,
    "run": RUN_SCHEMA_VERSION,
    "bundle": BUNDLE_SCHEMA_VERSION,
    "normalization": NORMALIZATION_SCHEMA_VERSION,
}

_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")


class ArtifactError(PESNError):
    """Typed structural failure of an artifact manifest."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"artifact invalid [{code}]: {detail}")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.match(value))


def _check_schema(doc: Mapping[str, object], expected: str, kind: str) -> None:
    got = doc.get("schema_version")
    if got != expected:
        raise ArtifactError(
            "SCHEMA_UNKNOWN", f"{kind} schema_version {got!r} != {expected!r}"
        )


def _check_required(doc: Mapping[str, object], required: tuple[str, ...], kind: str) -> None:
    missing = [key for key in required if key not in doc]
    if missing:
        raise ArtifactError("MISSING_KEY", f"{kind} missing keys: {missing}")


def _check_no_forbidden(doc: Mapping[str, object], forbidden: tuple[str, ...], kind: str) -> None:
    present = [key for key in forbidden if key in doc]
    if present:
        raise ArtifactError("FORBIDDEN_KEY", f"{kind} carries forbidden keys: {present}")


def validate_dataset_manifest(doc: Mapping[str, object]) -> None:
    _check_schema(doc, DATASET_SCHEMA_VERSION, "dataset")
    _check_required(
        doc,
        (
            "dataset_version",
            "source_sha256",
            "parent_split_manifest_sha256",
            "label_spec_sha256",
            "shards",
            "n_reactions",
            "n_samples",
            "task_coverage",
        ),
        "dataset",
    )


def validate_labels_manifest(doc: Mapping[str, object]) -> None:
    _check_schema(doc, LABELS_SCHEMA_VERSION, "labels")
    _check_required(
        doc,
        ("dataset_manifest_sha256", "label_spec_sha256", "shards", "n_samples"),
        "labels",
    )


def validate_split_manifest(doc: Mapping[str, object]) -> None:
    _check_schema(doc, SPLIT_SCHEMA_VERSION, "split")
    _check_required(
        doc,
        (
            "parent_split_sha256",
            "grouping_version",
            "group_members_sha256",
            "subsets",
            "family_holdout",
            "frozen",
        ),
        "split",
    )
    if doc.get("frozen") is not True:
        raise ArtifactError("NOT_FROZEN", "split manifest must be frozen=true")
    known_cross_conflicts = doc.get("known_group_cross_split_conflicts", 0)
    if known_cross_conflicts != 0:
        raise ArtifactError(
            "GROUP_CROSS_SPLIT_CONFLICT",
            f"split reports {known_cross_conflicts} cross-split group conflicts",
        )


def validate_feature_spec(doc: Mapping[str, object]) -> None:
    _check_schema(doc, FEATURE_SPEC_SCHEMA_VERSION, "feature_spec")
    _check_required(
        doc,
        (
            "spec_version",
            "allowed_fields",
            "graph_definition",
            "units",
            "invariances",
            "support_domain",
        ),
        "feature_spec",
    )
    _check_no_forbidden(
        doc,
        ("reference_ts", "neighbor_frames", "future_frames", "labels"),
        "feature_spec",
    )


def validate_run_record(doc: Mapping[str, object]) -> None:
    _check_schema(doc, RUN_SCHEMA_VERSION, "run")
    _check_required(
        doc,
        ("run_id", "status", "config_sha256", "dataset_manifest_sha256", "events"),
        "run",
    )


#: Files a runtime model bundle may contain (export whitelist).
BUNDLE_RUNTIME_FILES: Final[tuple[str, ...]] = (
    "model_bundle.json",
    "weights.pt",
    "feature_spec.json",
    "normalization.json",
    "label_spec.json",
    "split_reference.json",
    "training_runs.json",
    "metrics.json",
    "model_card.md",
    "checksums.json",
    "action_support_registry.json",
)

#: Keys that must never appear inside a runtime bundle manifest.
BUNDLE_FORBIDDEN_KEYS: Final[tuple[str, ...]] = (
    "reference_ts",
    "neighbor_frames",
    "future_frames",
    "teacher_labels",
    "teacher_states",
    "optimizer_state",
    "offline_entrypoint",
    "source_paths",
    "dataset_paths",
)


def validate_bundle_manifest(doc: Mapping[str, object]) -> None:
    _check_schema(doc, BUNDLE_SCHEMA_VERSION, "bundle")
    _check_required(
        doc,
        (
            "model_id",
            "model_hash",
            "feature_spec_sha256",
            "label_spec_sha256",
            "normalization_sha256",
            "weights_sha256",
            "training_run_ids",
            "capabilities",
            "model_card",
            "checksums",
        ),
        "bundle",
    )
    _check_no_forbidden(doc, BUNDLE_FORBIDDEN_KEYS, "bundle")
    for key in ("model_hash", "feature_spec_sha256", "weights_sha256"):
        if not _is_sha256(doc.get(key)):
            raise ArtifactError("HASH_INVALID", f"bundle {key} is not a sha256 hex digest")
    from pes2ts_core.pesn.contracts.predictions import validate_capability_map

    caps = doc.get("capabilities")
    if not isinstance(caps, dict):
        raise ArtifactError("MISSING_KEY", "bundle capabilities must be a mapping")
    validate_capability_map({str(k): str(v) for k, v in caps.items()})


__all__ = [
    "BUNDLE_FORBIDDEN_KEYS",
    "BUNDLE_RUNTIME_FILES",
    "ArtifactError",
    "DATASET_SCHEMA_VERSION",
    "FEATURE_SPEC_SCHEMA_VERSION",
    "LABELS_SCHEMA_VERSION",
    "LABEL_SPEC_SCHEMA_VERSION",
    "NORMALIZATION_SCHEMA_VERSION",
    "RUN_SCHEMA_VERSION",
    "SPLIT_SCHEMA_VERSION",
    "SCHEMA_VERSIONS",
    "validate_bundle_manifest",
    "validate_dataset_manifest",
    "validate_feature_spec",
    "validate_labels_manifest",
    "validate_run_record",
    "validate_split_manifest",
]
