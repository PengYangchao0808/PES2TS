"""G1 package: authoritative bond-change analysis and unified reaction indexing.

This module owns the G1 manifest contract constants shared by every G1 stage
writer and reader.
"""

from __future__ import annotations

from typing import Final

#: Schema version stamped into every G1 manifest.
G1_MANIFEST_SCHEMA_VERSION: Final[str] = "g1_manifest_v1"

__all__ = ["G1_MANIFEST_SCHEMA_VERSION"]
