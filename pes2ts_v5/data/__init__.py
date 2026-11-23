"""Data layer for V5.6: manifest/spec, group split, trajectories, labels.

The batch contract itself is **not** duplicated here: :mod:`pes2ts_v5.data.contracts`
lazily bridges to ``pes2ts_data.contracts`` (StandardBatch / REQUIRED_FIELDS).
"""
from pes2ts_v5.data.contracts import (
    assert_batch_fields,
    contract_version,
    contracts_available,
    provenance_tokens,
    standard_batch_view,
)
from pes2ts_v5.data.labels import (
    LABEL_SPEC,
    PROCESS_CLASSES,
    build_move_labels,
    build_process_labels,
    build_trust_labels,
)
from pes2ts_v5.data.manifest import (
    ARTIFACT_FILES,
    SCHEMA_VERSION,
    DatasetManifest,
    V55ContractNotAvailable,
    V55DatasetSpec,
    adapt_v55_manifest,
    compute_dataset_hash,
    sha256_file,
)
from pes2ts_v5.data.split import SPLIT_NAMES, check_group_leak, compute_normalization_stats
from pes2ts_v5.data.trajectories import (
    V5TrajectoryDataset,
    build_frame_features,
    build_synthetic_dataset,
    collate_samples,
)

__all__ = [
    "SCHEMA_VERSION",
    "ARTIFACT_FILES",
    "V55DatasetSpec",
    "DatasetManifest",
    "V55ContractNotAvailable",
    "adapt_v55_manifest",
    "compute_dataset_hash",
    "sha256_file",
    "SPLIT_NAMES",
    "check_group_leak",
    "compute_normalization_stats",
    "V5TrajectoryDataset",
    "build_frame_features",
    "build_synthetic_dataset",
    "collate_samples",
    "LABEL_SPEC",
    "PROCESS_CLASSES",
    "build_move_labels",
    "build_process_labels",
    "build_trust_labels",
    "assert_batch_fields",
    "contract_version",
    "contracts_available",
    "provenance_tokens",
    "standard_batch_view",
]
