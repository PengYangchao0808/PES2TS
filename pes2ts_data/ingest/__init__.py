"""Data-ingest layer (Zone 2): tombstone/cost-ledger readers and ingest contracts.

Public API::

    from pes2ts_data.ingest import (
        load_campaign_index, load_ledger, verify_entry, summarize,
        VerifyStatus, LedgerError,
    )
"""

from __future__ import annotations

from .ledger import (
    CAMPAIGN_INDEX_RELPATH,
    CAMPAIGN_INDEX_SCHEMA,
    RETIRED_LEDGER_RELPATH,
    RETIRED_LEDGER_SCHEMA,
    CampaignEntry,
    CampaignIndex,
    Ledger,
    LedgerError,
    RetiredEntry,
    Verification,
    VerifyStatus,
    load_campaign_index,
    load_ledger,
    resolve_artifact_path,
    sha256_file,
    summarize,
    verify_entry,
)

__all__ = [
    "CAMPAIGN_INDEX_RELPATH",
    "CAMPAIGN_INDEX_SCHEMA",
    "RETIRED_LEDGER_RELPATH",
    "RETIRED_LEDGER_SCHEMA",
    "CampaignEntry",
    "CampaignIndex",
    "Ledger",
    "LedgerError",
    "RetiredEntry",
    "Verification",
    "VerifyStatus",
    "load_campaign_index",
    "load_ledger",
    "resolve_artifact_path",
    "sha256_file",
    "summarize",
    "verify_entry",
]
