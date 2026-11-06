"""V5.4.3 MOVE-track decoder candidates (plan §11.1 directory tree :1097–1103;
WP5).

This package holds the formal MOVE candidates of the campaign, one module per
candidate as §11.1 prescribes:

* :mod:`~pes2ts_core.pesn.models.v543.move.residual_mlp` — ``M1`` Residual MLP
  (§6.2 :526, 128 → 136 → 136 → 8 coefficients, 37,272 params; the literal
  123-d ``u0/a_ref`` block is fed as its invariant ``key_v`` overlaps — see the
  module docstring's deviation note);
* :mod:`~pes2ts_core.pesn.models.v543.move.gvp` — ``M2`` GVP Refiner
  (§6.2 :527, 2 GVP blocks, scalars 96 / vectors 8, 35,600 params);
* :mod:`~pes2ts_core.pesn.models.v543.move.vector_gnn` — ``M3`` Vector-GNN
  (§6.2 :528, 2 scalar–vector layers, hidden 56, edge 8, 34,448 params);
* ``flow.py`` (``M4``) and ``diffusion.py`` (``M5``) are the CONDITIONAL
  candidates of §6.5 :574–583 and are deliberately absent: §6.5 gates their
  activation behind S1 diagnostics, so WP5 registers only M1–M3.

Package-level exports are intentionally NOT declared here: the integrating
agent owns the ``v543`` package surface (``models/v543/__init__.py``), and the
registry records of this WP live in each module (``M1_SPEC`` / ``M2_SPEC`` /
``M3_SPEC`` plus the ``register_m1`` / ``register_m2`` / ``register_m3``
helpers) so the shared ``registry.py`` never has to be edited by WP5.
"""

# --- integrator append (2026-11-03): mirror the process/ package surface ----
# Campaign wiring (WP7/WP8) registers the three formal MOVE candidates into a
# fresh registry; the module-global REGISTRY_V543 intentionally keeps only the
# M0/P0 baselines (WP4 contract).
from pes2ts_core.pesn.models.v543.move.gvp import (
    M2_SPEC,
    m2_spec,
    register_m2,
)
from pes2ts_core.pesn.models.v543.move.residual_mlp import (
    M1_SPEC,
    m1_spec,
    register_m1,
)
from pes2ts_core.pesn.models.v543.move.vector_gnn import (
    M3_SPEC,
    m3_spec,
    register_m3,
)

__all__ = [
    "M1_SPEC",
    "M2_SPEC",
    "M3_SPEC",
    "m1_spec",
    "m2_spec",
    "m3_spec",
    "register_m1",
    "register_m2",
    "register_m3",
]
