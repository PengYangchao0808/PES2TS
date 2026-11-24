"""Zone 2: data engineering for PES2TS.

This package owns the *data side* of the repository restructuring:

``pes2ts_data.contracts``
    The single data-batch contract v2: :class:`BaseBatch` / :class:`SequenceBatch`
    / :class:`GraphBatch`, :data:`FIELD_SPECS`, :func:`validate_batch`, plus the
    deprecated :class:`StandardBatch` compat shim.
``pes2ts_data.readers``
    Read-only numpy readers ported from the frozen tree: idx_v2 memmap reader,
    V5.5-shaped numpy dataset view, six-table streaming view, label cache and
    light-index loader (no torch).
``pes2ts_data.splits``
    R6 split firewall: validated ``config/data/splits.yaml`` policy and the pure
    guards (:func:`assert_no_family_leak`, :func:`assert_sealed`,
    :func:`check_split_frame`).
``pes2ts_data.ingest``
    Tombstone / cost-ledger readers for the retired data campaigns.
``pes2ts_data.registry``
    Datastore version-lock registry contract.
``pes2ts_data.builders``
    Moved build scripts (``scripts/datastore/*.py``); read-only here and not
    imported at package import time to keep this package light.

The R1 truth bridge (:mod:`pes2ts_data.truth_bridge`) is deliberately NOT
re-exported here: it must only be imported explicitly by offline isolation
builders, never by online/generation paths.
"""

from __future__ import annotations

from . import contracts, ingest, readers, registry, splits
from .contracts import (
    BASE_FIELDS,
    CODES,
    COMPAT_FIELDS,
    CONTRACT_REQUIRED_FIELDS,
    CONTRACT_VERSION,
    DTYPE_FAMILIES,
    FIELD_SPECS,
    GRAPH_FIELDS,
    IDENTITY_FIELDS,
    LEGACY_CONTRACT_VERSION,
    LEGACY_FIELD_MAP,
    SPLIT_TAGS,
    TRUTH_FIELD_PATTERNS,
    BaseBatch,
    ContractViolation,
    FieldSpec,
    GraphBatch,
    Provenance,
    SequenceBatch,
    StandardBatch,
    assert_fields,
    from_standard,
    is_truth_field,
    required_fields,
    to_standard,
    validate_batch,
)
from .ingest import (
    CampaignEntry,
    CampaignIndex,
    Ledger,
    LedgerError,
    RetiredEntry,
    Verification,
    VerifyStatus,
    load_campaign_index,
    load_ledger,
    summarize,
    verify_entry,
)
from .readers import (
    READER_VERSION,
    IdxV2Dataset,
    IdxV2Error,
    IdxV2Reader,
    IdxV2Spec,
    IntentRow,
    KeyAtomStreamView,
    LabelTables,
    LightIndexState,
    ReaderError,
    StreamColumns,
    build_frame_features,
    idx_v2_available,
    light_index_available,
    load_label_cache,
    load_light_index,
    read_light_index_metadata,
    read_staging_chunk,
)
from .registry import (
    REGISTRY_SCHEMA_ID,
    REGISTRY_STATUSES,
    DatastoreEntry,
    DatastoreRegistry,
    RegistryEntry,
    RegistryError,
    load_registry,
    validate_registry,
)
from .splits import (
    CONFIRM_N,
    CONFIRM_SPLIT,
    TEST_SET_NAMES,
    SplitPolicy,
    SplitPolicyError,
    SplitViolation,
    assert_no_family_leak,
    assert_sealed,
    check_split_frame,
    load_split_policy,
)

__version__ = "2.0.0"

__all__ = [
    # -- version -----------------------------------------------------------
    "__version__",
    # -- submodules --------------------------------------------------------
    "contracts",
    "ingest",
    "readers",
    "registry",
    "splits",
    # -- batch contract v2 (pes2ts_data.contracts) -------------------------
    "BASE_FIELDS",
    "CODES",
    "COMPAT_FIELDS",
    "CONTRACT_REQUIRED_FIELDS",
    "CONTRACT_VERSION",
    "DTYPE_FAMILIES",
    "FIELD_SPECS",
    "GRAPH_FIELDS",
    "IDENTITY_FIELDS",
    "LEGACY_CONTRACT_VERSION",
    "LEGACY_FIELD_MAP",
    "SPLIT_TAGS",
    "TRUTH_FIELD_PATTERNS",
    "BaseBatch",
    "ContractViolation",
    "FieldSpec",
    "GraphBatch",
    "Provenance",
    "SequenceBatch",
    "StandardBatch",
    "assert_fields",
    "from_standard",
    "is_truth_field",
    "required_fields",
    "to_standard",
    "validate_batch",
    # -- readers (pes2ts_data.readers) ------------------------------------
    "READER_VERSION",
    "IdxV2Dataset",
    "IdxV2Error",
    "IdxV2Reader",
    "IdxV2Spec",
    "IntentRow",
    "KeyAtomStreamView",
    "LabelTables",
    "LightIndexState",
    "ReaderError",
    "StreamColumns",
    "build_frame_features",
    "idx_v2_available",
    "light_index_available",
    "load_label_cache",
    "load_light_index",
    "read_light_index_metadata",
    "read_staging_chunk",
    # -- split firewall (pes2ts_data.splits) -------------------------------
    "CONFIRM_N",
    "CONFIRM_SPLIT",
    "TEST_SET_NAMES",
    "SplitPolicy",
    "SplitPolicyError",
    "SplitViolation",
    "assert_no_family_leak",
    "assert_sealed",
    "check_split_frame",
    "load_split_policy",
    # -- ledger (pes2ts_data.ingest) --------------------------------------
    "CampaignEntry",
    "CampaignIndex",
    "Ledger",
    "LedgerError",
    "RetiredEntry",
    "Verification",
    "VerifyStatus",
    "load_campaign_index",
    "load_ledger",
    "summarize",
    "verify_entry",
    # -- registry (pes2ts_data.registry) ----------------------------------
    "REGISTRY_SCHEMA_ID",
    "REGISTRY_STATUSES",
    "DatastoreEntry",
    "DatastoreRegistry",
    "RegistryEntry",
    "RegistryError",
    "load_registry",
    "validate_registry",
]
