"""Per-frame recovery and target-path quality for PES generation.

Target occupants: :mod:`pes2ts_core.generation.planning.target_path` (four
quality layers) and :mod:`pes2ts_core.integration.acp.frame_recovery` (per-frame
per-driver target/actual/residual). They are re-homed here in a later sync tier
once the X3 result projection exists; this package is a marker so the canonical
import path exists.
"""

from __future__ import annotations

__all__: list[str] = []
