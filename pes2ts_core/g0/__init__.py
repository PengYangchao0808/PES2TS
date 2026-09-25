"""G0 package: data entry, ground-truth quarantine, and split freezing.

This module is the single source of truth for the G0 manifest contract
constants shared by every G0 stage writer and reader.
"""

from __future__ import annotations

from typing import Final

#: Schema version stamped into every G0 manifest.
MANIFEST_SCHEMA_VERSION: Final[str] = "g0_manifest_v1"

#: Manifest keys that legitimately change between runs and must therefore be
#: excluded from any hash-input structure.
VOLATILE_KEYS: Final[frozenset[str]] = frozenset(
    {"generated_at", "downloaded_at", "duration_seconds"}
)

__all__ = ["MANIFEST_SCHEMA_VERSION", "VOLATILE_KEYS"]
