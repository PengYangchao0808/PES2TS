"""Shared contract layer of the torch-free kernel.

This package hosts two things, both pure-Python:

* the generic document/schema contract helpers (``core.py``, formerly
  ``pes2ts_core/contracts.py``) — R1 truth-key rejection, document sealing,
  artifact-reference validation;
* the active task contracts still used by the G2/G4 planning code:
  ``inputs.py`` (task graph / coordinate specs), ``control.py`` (control
  context / reward / residual specs) and ``direct.py`` (direct-MDP contracts).

Predictor-model contracts (key-atom batches, dual-TS bundles, exported
predictions/artifacts) live with the frozen V4/V5.x predictor tree on the
read-only branch ``frozen/v4-v54-history``; they are deliberately NOT part
of the active kernel because they import the ML stack.
"""

from __future__ import annotations

class PESNError(Exception):
    """Base type for predictor-domain contract errors (kept for submodules)."""

from pes2ts_core.contracts import core as _core
from pes2ts_core.contracts.core import *  # noqa: F401,F403

try:  # core.py does not declare __all__: fall back to its public names
    from pes2ts_core.contracts.core import __all__ as _core_all
except ImportError:  # pragma: no cover - explicit fallback
    _core_all = [name for name in dir(_core) if not name.startswith('_')]

__all__ = [name for name in _core_all if not name.startswith('_')]
