"""G2 package: cheap GFN2-xTB reaction-path generation.

This module owns the G2 directory, schema, and exit-code contract constants
shared by every G2 stage writer and reader, and re-exports the status and
failure-code contracts from :mod:`pes2ts_core.generation.execution.xtb_path.status`.
"""

from __future__ import annotations

from typing import Final

from pes2ts_core.generation.execution.xtb_path.status import (
    DEFAULT_REVERSE_RETRY_TRIGGERS,
    FAILURE_PRECEDENCE,
    G2_FAILURE_CODES,
    G2_STATUSES,
)

#: Directory (under ``paths.interim``) holding every G2 path artifact.
G2_DIRNAME: Final[str] = "g2"
#: Subdirectory (under the G2 directory) holding the per-shard path documents.
G2_PATHS_DIRNAME: Final[str] = "paths"
#: Schema version stamped into every G2 per-reaction path document.
SCHEMA_PATH: Final[str] = "g2_path_v1"
#: Schema version stamped into every G2 manifest artifact.
SCHEMA_MANIFEST: Final[str] = "g2_manifest_v1"
#: Schema version stamped into every G2 coverage artifact.
SCHEMA_COVERAGE: Final[str] = "g2_coverage_v1"
#: Exit code: a G2 stage hit an infrastructure error (missing inputs, a
#: missing xTB executable, or a failed ``g2 verify``).
EXIT_G2_FAILED: Final[int] = 24

__all__ = [
    "DEFAULT_REVERSE_RETRY_TRIGGERS",
    "EXIT_G2_FAILED",
    "FAILURE_PRECEDENCE",
    "G2_DIRNAME",
    "G2_FAILURE_CODES",
    "G2_PATHS_DIRNAME",
    "G2_STATUSES",
    "SCHEMA_COVERAGE",
    "SCHEMA_MANIFEST",
    "SCHEMA_PATH",
]
