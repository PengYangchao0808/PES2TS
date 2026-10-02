"""Isolated evaluation plane (truth/DQ).

Target role: hold the quarantined truth readers, the OptTS / frequency / IRC
result collection, and the denominator + cost accounting. It is **read-only
with respect to generation and ranking** — nothing here may flow back into a
generation or inference input (four red lines, 整体开发与发布方案 v2 §1).

Migration status (code sync, additive): package marker only. The evaluation
surface currently lives in :mod:`pes2ts_core.g1` (P1/P2 truth) and
:mod:`pes2ts_core.integration.acp` (validation stages); those are relocated in
a later sync tier once the X3 result projection exists. No existing module is
imported here, so this package has zero effect on current behaviour.
"""

from __future__ import annotations

__all__: list[str] = []
