"""Deterministic, truth-free endpoint assembly for PES generation.

Holds the multi-component rigid assembly, clash separation, and scoring that
both generation backends share. Currently :mod:`pes2ts_core.generation.assembly.endpoints`
(the former ``pes2ts_core.g2.endpoints``); the ``XTB_PATH`` backend imports it
from here.
"""

from __future__ import annotations

__all__: list[str] = []
