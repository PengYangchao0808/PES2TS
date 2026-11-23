"""Zone 2: data engineering for PES2TS.

This package owns the *data side* of the repository restructuring:

``pes2ts_data.contracts``
    The single data-batch contract (:class:`StandardBatch`, :data:`FIELD_SPECS`,
    :func:`validate_batch`) plus the legacy V5.6 trajectory compat bridge.
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
"""

from __future__ import annotations

from . import contracts, ingest, registry, splits
from .contracts import (
    COMPAT_FIELDS,
    CONTRACT_REQUIRED_FIELDS,
    CONTRACT_VERSION,
    FIELD_SPECS,
    LEGACY_FIELD_MAP,
    TRUTH_FIELD_PATTERNS,
    ContractViolation,
    FieldSpec,
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

__version__ = "1.0.0"

__all__ = [
    # -- version -----------------------------------------------------------
    "__version__",
    # -- submodules --------------------------------------------------------
    "contracts",
    "ingest",
    "registry",
    "splits",
    # -- batch contract (pes2ts_data.contracts) ----------------------------
    "COMPAT_FIELDS",
    "CONTRACT_REQUIRED_FIELDS",
    "CONTRACT_VERSION",
    "FIELD_SPECS",
    "LEGACY_FIELD_MAP",
    "TRUTH_FIELD_PATTERNS",
    "ContractViolation",
    "FieldSpec",
    "StandardBatch",
    "assert_fields",
    "from_standard",
    "is_truth_field",
    "required_fields",
    "to_standard",
    "validate_batch",
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
