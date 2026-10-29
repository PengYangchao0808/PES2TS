"""V5.4.1 WP3a feature providers (plan §5.1 layout, §6.2–§6.8).

Lazy, pure-PyTorch builders of the LEGAL internal views the WP3b
candidate embeddings (E1–E5) consume.  This package is additive only: it
does not modify ``V541Model``, the embedding registry, campaign config,
legacy features or any training file, and it ships no E1–E5 operator
math — those land in ``...v541/embeddings/`` during WP3b.

Public surface::

    from pes2ts_core.pesn.models.v541.feature_providers import (
        build_legal_neighbor_views,   # G_chem + G_spatial (§6.2/§6.6.2)
        build_chem_graph,             # G_chem alone (E1/E2 view)
        build_spatial_graph,          # G_spatial alone (E3 view)
        build_angle_triplets,         # G_angle (E4 view, §6.3)
        CHEM_EDGE_FIELDS, CHEM_EDGE_DIM,   # frozen 8-field schema
        FeatureProviderError,              # typed refusals
    )

Conventions (frozen by WP3a, see module docstrings):

* inputs are forward-view batches only (``FORWARD_VIEW_KEYS_V3`` subset);
  no new forward key is introduced and the F2-masked ``edge_attr`` RBF
  slots are never read — distances come from the current ``pos`` in Å;
* outputs are dataclasses of tensors plus ``counts`` of DETACHED Python
  floats (no autograd graph is ever retained by a provider);
* cross-graph edges, duplicate-attribute mismatches, zero-length angle
  sides and non-finite geometry raise ``FeatureProviderError`` with a
  stable ``code`` (fail closed).
"""

from pes2ts_core.pesn.models.v541.feature_providers.angle_triplets import (
    ANGLE_MAX_NEIGHBORS,
    ANGLE_MAX_PAIRS,
    ANGLE_SHELL_TIE_TOL,
    ANGLE_TIE_POLICY,
    AngleTripletView,
    build_angle_triplets,
)
from pes2ts_core.pesn.models.v541.feature_providers.common import (
    CHEM_EDGE_DIM,
    CHEM_EDGE_FIELDS,
    FeatureProviderError,
    RBF_CENTER_END,
    RBF_CENTER_START,
    RBF_N,
    RBF_SIGMA_ANGSTROM,
    rbf12,
    safe_distances,
)
from pes2ts_core.pesn.models.v541.feature_providers.legal_neighbors import (
    ChemGraphView,
    LegalNeighborViews,
    SpatialGraphView,
    build_chem_graph,
    build_legal_neighbor_views,
    build_spatial_graph,
)

__all__ = [
    "ANGLE_MAX_NEIGHBORS",
    "ANGLE_MAX_PAIRS",
    "ANGLE_SHELL_TIE_TOL",
    "ANGLE_TIE_POLICY",
    "CHEM_EDGE_DIM",
    "CHEM_EDGE_FIELDS",
    "AngleTripletView",
    "ChemGraphView",
    "FeatureProviderError",
    "LegalNeighborViews",
    "RBF_CENTER_END",
    "RBF_CENTER_START",
    "RBF_N",
    "RBF_SIGMA_ANGSTROM",
    "SpatialGraphView",
    "build_angle_triplets",
    "build_chem_graph",
    "build_legal_neighbor_views",
    "build_spatial_graph",
    "rbf12",
    "safe_distances",
]
