"""Runtime feature construction for the PES-N predictor.

The feature layer converts a qualified :class:`~pes2ts_core.pesn.contracts.
inputs.PESNInput` (topology-authority task graph + current geometry) into
plain numeric arrays.  It never re-decides edits, roles, or permissions, and
it can only see the current observation — reference TS, future frames, and
labels are structurally unreachable here.
"""

from __future__ import annotations
