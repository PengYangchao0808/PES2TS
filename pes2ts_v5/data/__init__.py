"""Data layer for V5.6: manifest/spec, group split, trajectories, labels, sources.

The batch contract itself is **not** duplicated here: :mod:`pes2ts_v5.data.contracts`
lazily bridges to ``pes2ts_data.contracts`` (contract v2 ``SequenceBatch``, with
the v1 ``StandardBatch`` compatibility backend while data-zone v2 is not on disk)
and there is no ``pes2ts_v5.data.idx_v2`` import surface — the real-data reader is
lazily imported from ``pes2ts_data.readers.idx_v2``.
"""
from pes2ts_v5.data.contracts import (
    IDX_V2_PREFIX,
    DataContractError,
    DataContractUnavailable,
    DataError,
    DataSourceUnreachable,
    assert_batch_fields,
    contract_backend,
    contract_version,
    contracts_available,
    legacy_mapping,
    make_sequence_batch,
    provenance_tokens,
    to_sequence_batches,
    validate_sequence_batch,
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
    "IDX_V2_PREFIX",
    "DataContractError",
    "DataContractUnavailable",
    "DataError",
    "DataSourceUnreachable",
    "assert_batch_fields",
    "contract_backend",
    "contract_version",
    "contracts_available",
    "legacy_mapping",
    "make_sequence_batch",
    "provenance_tokens",
    "to_sequence_batches",
    "validate_sequence_batch",
]
