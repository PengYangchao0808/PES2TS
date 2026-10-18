"""PES-N predictor product package (research scope, batch G3-P1).

Ownership note (ADR-0014): every new business line for the predictor lives in
this package.  The runtime-reachable set is ``api`` / ``contracts`` /
``features`` / ``models`` / ``inference`` / ``bundle``; the ``offline``
sub-package (data, splits, labels, training, evaluation, experiments,
publishing) may depend on the runtime set but never the reverse.

Importing this package is intentionally cheap: no ML backend, no offline
module, and no data source is imported at package-import time.  Heavy
submodules are loaded lazily through :func:`__getattr__` (PEP 562).
"""

from __future__ import annotations

from typing import Final

#: Product identity (predictor product, distinct from the software version).
PESN_PRODUCT_NAME: Final[str] = "pesn"
#: Contract family implemented by this package (draft status per ADR-0013).
PESN_CONTRACT_FAMILY: Final[str] = "pesn_v1_draft"
#: First-cycle research model identity this package targets (weights not yet
#: frozen; the identity itself carries no capability claim).
PESN_IRC_MODEL_TARGET: Final[str] = "pesn-irc-v0.1.0"

__all__ = [
    "PESN_PRODUCT_NAME",
    "PESN_CONTRACT_FAMILY",
    "PESN_IRC_MODEL_TARGET",
    "__version__",
]


def __getattr__(name: str) -> object:
    """Lazily resolve public runtime symbols without importing them eagerly."""
    if name == "__version__":
        from pes2ts_core.version import SOFTWARE_VERSION

        return SOFTWARE_VERSION
    lazy_runtime = {
        "PESNInput": ("pes2ts_core.pesn.contracts.inputs", "PESNInput"),
        "TaskGraph": ("pes2ts_core.pesn.contracts.inputs", "TaskGraph"),
        "CoordinateSpec": ("pes2ts_core.pesn.contracts.inputs", "CoordinateSpec"),
        "TeacherPrediction": ("pes2ts_core.pesn.contracts.predictions", "TeacherPrediction"),
        "load_bundle": ("pes2ts_core.pesn.api", "load_bundle"),
        "describe_bundle": ("pes2ts_core.pesn.api", "describe_bundle"),
    }
    entry = lazy_runtime.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = entry
    # Plain module import (never importlib): the static truth guard flags
    # dynamic-execution primitives, and this package stays fully guarded.
    if module_name == "pes2ts_core.pesn.contracts.inputs":
        from pes2ts_core.pesn.contracts import inputs as _mod
    elif module_name == "pes2ts_core.pesn.contracts.predictions":
        from pes2ts_core.pesn.contracts import predictions as _mod
    elif module_name == "pes2ts_core.pesn.api":
        from pes2ts_core.pesn import api as _mod
    else:  # pragma: no cover - defensive, all entries mapped above
        raise AttributeError(f"unmapped lazy module {module_name!r}")
    return getattr(_mod, attr)
