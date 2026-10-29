"""V5.4.1 WP3a shared primitives for the legal-view feature providers
(plan §6.2–§6.8; config ``embedding_contract.feature_recipe``; ADR-0032).

Scope discipline: this package ONLY adds
``pes2ts_core/pesn/models/v541/feature_providers/**``.  It does not touch
``V541Model``/registry/config/legacy features/training, implements no
E1–E5 operator (those are WP3b), and performs no training/QC.

Normative references — imported, never re-typed:

* ``contracts.key_atom_v3`` — forward-view key names, ``EDGE_PACK_DIM_V3``
  = 17 packed edge slots (F2 keeps slots 12–16: ``is_spatial``,
  ``covalent_union``, ``aromatic_union``, ``b_min``, ``b_max``; the first
  12 RBF slots are MASKED to zero at F2) and ``EDGE_INPUT_DIM_V3`` = 22
  legacy edge channels (17 pack + 5 explicit scalars);
* ``features.key_atom_spec_v3`` — Gaussian RBF(12) centers uniform in
  ``[0, 4.5]`` Å with σ = 0.3 Å, spatial cutoff 4.5 Å, ≤32 neighbours per
  atom, 1e-6 Å boundary-tie retention (the legacy main-graph recipe of
  ``key_atom_graph_v3``: ``G_src ∪ G_tgt`` covalent union ∪ current-radius
  selection; covalent pairs are ALWAYS kept, even beyond the cutoff);
* campaign yaml ``feature_recipe`` — ``chem_edge_fields`` (the 8 frozen
  fields mirrored by :data:`CHEM_EDGE_FIELDS`) and the angle policy
  (``max_neighbors_per_center: 12``, ``max_unordered_pairs_per_center: 66``,
  ``tie_policy: reject_boundary_tie_shell``, ``basis: [1, c, c2, c3]``,
  ``cos_clamp: [-1.0, 1.0]``); the WP3a tests assert this module's
  constants against the yaml so drift fails closed.

Units and legality (plan §6.6):

* coordinates are the CURRENT frame in Å exactly as carried by the
  forward view ``pos`` — providers never rescale and never read a
  reference/X0/future frame;
* the F2-masked ``edge_attr`` RBF slots (first 12) are NEVER read: all
  distances are recomputed from ``pos`` and re-expanded through the
  frozen Gaussian definition (plan §6.6.1);
* providers read ONLY forward-view whitelist keys; no new forward key is
  introduced;
* all math is pure PyTorch (no torch_geometric / no third-party GNN), the
  functions are lazy/stateless (no module-level caches, no retained
  autograd graphs), and every count diagnostic is a detached Python
  float (cf. ``v541.interfaces`` diagnostics convention).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping

import torch

from pes2ts_core.pesn.contracts.key_atom_v3 import (
    EDGE_ABS_DELTA_KEY,
    EDGE_ATTR_KEY,
    EDGE_B_SRC_KEY,
    EDGE_B_TGT_KEY,
    EDGE_DELTA_B_KEY,
    EDGE_INDEX_KEY,
    EDGE_IS_CHANGED_KEY,
    EDGE_PACK_DIM_V3,
    EDGE_PACK_SLOTS_V3,
    NODE_GRAPH_KEY,
    NODE_POS_KEY,
)
from pes2ts_core.pesn.features.key_atom_spec_v3 import (
    N_RBF_V3,
    RBF_CUTOFF,
    RBF_SIGMA,
    SPATIAL_TIE_TOL,
)

# --- frozen chem-edge schema (config feature_recipe.chem_edge_fields) -------
#: The 8 frozen chemical edge fields, in config order.  ``b_src``/``b_tgt``/
#: ``delta_b_intent``/``abs_delta``/``is_changed`` are the forward-view edge
#: scalars (already bond-order/3 scaled by the legacy builder); the three
#: context fields come from ``edge_attr`` pack slots 13/15/16.  E1/E2 must
#: consume EXACTLY this set — no RDKit-derived extras (plan §6.6.4).
CHEM_EDGE_FIELDS: Final[tuple[str, ...]] = (
    "b_src", "b_tgt", "delta_b_intent", "abs_delta", "is_changed",
    "covalent_union", "b_min", "b_max",
)
CHEM_EDGE_DIM: Final[int] = len(CHEM_EDGE_FIELDS)   # 8 (WP3a fixture freeze)
_CHEM_FIELD_INDEX: Final[dict[str, int]] = {
    name: i for i, name in enumerate(CHEM_EDGE_FIELDS)}

# --- frozen RBF12 definition (spec v3 / config feature_recipe.rbf12) --------
RBF_N: Final[int] = N_RBF_V3                        # 12
RBF_CENTER_START: Final[float] = 0.0                # Å
RBF_CENTER_END: Final[float] = RBF_CUTOFF           # 4.5 Å
RBF_SIGMA_ANGSTROM: Final[float] = RBF_SIGMA        # 0.3 Å
#: Clamp on ``(r − c)/σ`` before squaring.  With σ = 0.3 and centers in
#: ``[0, 4.5]`` every IN-SPEC distance (r ≤ 4.5 Å) has |t| ≤ 15, so the
#: clamp never changes an in-spec value; it only keeps the exponent finite
#: (and its backward finite — no ``0·inf``) for extreme-but-legal
#: coordinates whose true Gaussian underflows float32 anyway.
RBF_T_CLAMP: Final[float] = 15.0

# --- frozen angle policy (config feature_recipe.angle) ----------------------
ANGLE_MAX_NEIGHBORS: Final[int] = 12                # §6.3.3
ANGLE_MAX_PAIRS: Final[int] = 66                    # C(12, 2), §6.3.3
#: Boundary-tie tolerance for grouping equal-distance neighbour shells.
#: The plan freezes no separate angle value, so the frozen legacy spatial
#: tie tolerance (1e-6 Å, spec v3 ``SPATIAL_TIE_TOL``) is reused; recorded
#: as a WP3a decision for WP0 sign-off.
ANGLE_SHELL_TIE_TOL: Final[float] = SPATIAL_TIE_TOL  # 1e-6 Å
ANGLE_TIE_POLICY: Final[str] = "reject_boundary_tie_shell"
COS_CLAMP_MIN: Final[float] = -1.0
COS_CLAMP_MAX: Final[float] = 1.0
#: Guard on ``‖u‖·‖v‖`` in the cosine denominator.  Cauchy–Schwarz keeps
#: ``|dot| ≤ ‖u‖·‖v‖``, so clamping the denominator can never push the
#: raw cosine beyond 1 by more than float rounding; the final clamp to
#: [-1, 1] absorbs that rounding (plan §6.3.6).
COS_DENOM_EPS: Final[float] = 1e-12
#: ``|cos| ≥ 1 − tol`` marks a collinear/near-collinear triplet; ``cos ≥
#: 1 − tol`` additionally marks a near-zero angle (diagnostics only —
#: never a mask, never a rejection).
ANGLE_COS_TOL: Final[float] = 1e-6

# --- chunking (bounded memory, plan §6.3.8) --------------------------------
#: Default centres (angle) / query atoms (spatial) per chunk.
DEFAULT_CHUNK_SIZE: Final[int] = 4096
#: Element budget for one spatial distance block ``[chunk, M]`` — the
#: query chunk shrinks automatically so peak memory stays bounded even
#: for a single huge graph.
DIST_BLOCK_BUDGET: Final[int] = 4_000_000


class FeatureProviderError(RuntimeError):
    """Typed refusal of the V5.4.1 feature-provider layer.

    Repo convention (cf. ``KeyAtomV3ContractError`` /
    ``V541InterfaceError``): every refusal carries a stable ``code`` and a
    human ``detail``; providers fail closed instead of fabricating a
    degraded view.
    """

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def require_v541(condition: bool, code: str, detail: str) -> None:
    """Raise :class:`FeatureProviderError` unless ``condition`` holds."""
    if not condition:
        raise FeatureProviderError(code, detail)


# --- count-only reductions (diagnostics=True paths ONLY) --------------------
def count_graphs(graph_index: torch.Tensor) -> int:
    """Graph count — a COUNTS-ONLY reduction.

    ``torch.unique`` forces a GPU→host sync; ``read_forward_batch``
    therefore skips it entirely under ``diagnostics=False`` (the value is
    consumed only by ``counts``).  Tests monkeypatch this symbol to
    prove the cheap path never calls it.  Legality checks never route
    through this helper.
    """
    return int(torch.unique(graph_index).numel())


def diagnostic_sum(values: torch.Tensor) -> int:
    """GPU→host reduction used ONLY to fill ``counts`` diagnostics.

    Every call site is gated on ``diagnostics=True``; legality checks
    (cross-graph, duplicate mismatch, non-finite, zero-distance angle
    sides) never route through this helper, so turning diagnostics off
    cannot weaken validation.  Tests monkeypatch this symbol to prove
    the ``diagnostics=False`` build pays no count-only sync.
    """
    return int(values.sum())


# --- geometry primitives ---------------------------------------------------
def safe_distances(diff: torch.Tensor) -> torch.Tensor:
    """Euclidean norm over the last dim of ``diff`` (Å).

    Forward value is EXACT (``0.0`` for a zero vector, identical to
    ``diff.norm(-1)`` elsewhere) while the backward pass stays finite at
    zero-length inputs: the zero branch is routed through a constant so
    ``sqrt``'s ``1/(2√0)`` singularity never reaches ``diff``.  This lets
    G_chem tolerate a degenerate pair (outputs stay finite) while the
    angle provider — which needs a direction — refuses zero-length sides
    with its own typed error (plan §6.3.6).
    """
    sq = (diff * diff).sum(dim=-1)
    positive = sq > 0
    sq_safe = torch.where(positive, sq, torch.ones_like(sq))
    mask = positive.to(dtype=sq.dtype)
    return sq_safe.sqrt() * mask


def rbf12(distances: torch.Tensor) -> torch.Tensor:
    """Frozen Gaussian RBF(12) expansion of CURRENT distances (Å).

    ``exp(−0.5 ((r − c_k)/0.3)²)`` over 12 centers uniform in ``[0, 4.5]``
    Å — the exact math of ``features.key_atom_spec_v3.gaussian_rbf_v3``
    (and of the F3 pack slots), recomputed here from physical distances.
    The providers NEVER read the F2-masked first 12 ``edge_attr`` slots
    (plan §6.6.1).  Output ``[..., 12]``, dtype/device follow
    ``distances``; always finite.
    """
    centers = torch.linspace(
        RBF_CENTER_START, RBF_CENTER_END, RBF_N,
        dtype=distances.dtype, device=distances.device)
    t = (distances.unsqueeze(-1) - centers) / RBF_SIGMA_ANGSTROM
    t = t.clamp(min=-RBF_T_CLAMP, max=RBF_T_CLAMP)
    return torch.exp(-0.5 * t * t)


def expand_pair_directions(pair_index: torch.Tensor) -> torch.Tensor:
    """``[2, P]`` undirected pairs → ``[2, 2P]`` directed edges.

    Column layout ``2p = (a→b)``, ``2p+1 = (b→a)`` — the legacy
    ``key_atom_graph_v3`` layout.  Both directions of a pair always carry
    IDENTICAL attributes downstream (message direction never flips the
    bond delta, plan §6.6.3).
    """
    if int(pair_index.shape[1]) == 0:
        return pair_index.new_zeros(2, 0)
    fwd = pair_index
    rev = pair_index.flip(0)
    return torch.stack((fwd, rev), dim=2).reshape(2, -1)


def undirected_pair_keys(edge_index: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """Sorted unique undirected keys ``min(i,j)·N + max(i,j)`` of ``edge_index``."""
    src, tgt = edge_index[0], edge_index[1]
    a = torch.minimum(src, tgt)
    b = torch.maximum(src, tgt)
    return torch.unique(a * int(n_nodes) + b)


# --- forward-view reading + validation -------------------------------------
@dataclass(frozen=True)
class ForwardBatchData:
    """Validated forward-view tensors a provider consumes.

    Only whitelist keys (``contracts.key_atom_v3.FORWARD_VIEW_KEYS_V3``)
    are ever read; edge-aligned scalars are optional because the spatial
    view does not need them.

    ``n_graphs`` is a COUNTS-ONLY field: it is ``0`` when the batch was
    read with ``diagnostics=False`` (the ``torch.unique`` sync is
    skipped); spatial iteration computes its own graph list regardless.
    """

    pos: torch.Tensor            # [N, 3] float, current frame (Å)
    graph_index: torch.Tensor    # [N] int64
    edge_index: torch.Tensor     # [2, E] int64
    edge_attr: torch.Tensor | None    # [E, 17] float
    b_src: torch.Tensor | None
    b_tgt: torch.Tensor | None
    delta_b_intent: torch.Tensor | None
    abs_delta: torch.Tensor | None
    is_changed: torch.Tensor | None
    n_nodes: int
    n_edges: int
    n_graphs: int


def _read_tensor(
        batch: Mapping[str, object], key: str) -> torch.Tensor:
    require_v541(isinstance(batch, Mapping),
                 "V541_PROVIDER_BATCH_TYPE",
                 f"batch must be a mapping, got {type(batch).__name__}")
    require_v541(key in batch,
                 "V541_PROVIDER_MISSING_KEY",
                 f"forward-view key {key!r} is required but absent")
    value = batch[key]
    require_v541(isinstance(value, torch.Tensor),
                 "V541_PROVIDER_BATCH_TYPE",
                 f"key {key!r} must be a tensor, got {type(value).__name__}")
    return value  # type: ignore[return-value]


def read_forward_batch(
        batch: Mapping[str, object], *, with_chem_scalars: bool,
        diagnostics: bool = True,
) -> ForwardBatchData:
    """Read + validate the forward-view tensors a provider needs.

    Typed refusals: ``V541_PROVIDER_MISSING_KEY`` / ``_BATCH_TYPE`` /
    ``_SHAPE`` / ``_DTYPE`` / ``_EMPTY`` / ``_GEOM_NONFINITE`` /
    ``_EDGE_RANGE`` / ``_SELF_LOOP`` / ``_CROSS_GRAPH``.  Cross-graph
    edges are rejected here for every view (plan §6.2: the legal views
    never mix graphs; the legacy collate guarantees this, the provider
    re-checks fail-closed).

    ``diagnostics=False`` skips the counts-only ``torch.unique`` graph
    count (host sync) and stores ``n_graphs=0``; every legality check
    above still runs in both modes.
    """
    pos = _read_tensor(batch, NODE_POS_KEY)
    require_v541(pos.ndim == 2 and pos.shape[-1] == 3,
                 "V541_PROVIDER_SHAPE",
                 f"pos must be [N,3], got {tuple(pos.shape)}")
    require_v541(bool(torch.is_floating_point(pos)),
                 "V541_PROVIDER_DTYPE",
                 f"pos must be floating point, got {pos.dtype}")
    n_nodes = int(pos.shape[0])
    require_v541(n_nodes >= 1,
                 "V541_PROVIDER_EMPTY", "batch has no atoms")
    require_v541(bool(torch.isfinite(pos).all()),
                 "V541_PROVIDER_GEOM_NONFINITE",
                 "pos contains non-finite coordinates (current frame, Å)")

    graph_index = _read_tensor(batch, NODE_GRAPH_KEY).reshape(-1).long()
    require_v541(int(graph_index.shape[0]) == n_nodes,
                 "V541_PROVIDER_SHAPE",
                 f"graph_index must be [{n_nodes}], got {tuple(graph_index.shape)}")
    require_v541(bool((graph_index >= 0).all()),
                 "V541_PROVIDER_EDGE_RANGE",
                 "graph_index contains negative graph ids")

    edge_index = _read_tensor(batch, EDGE_INDEX_KEY).long()
    require_v541(edge_index.ndim == 2 and int(edge_index.shape[0]) == 2,
                 "V541_PROVIDER_SHAPE",
                 f"edge_index must be [2,E], got {tuple(edge_index.shape)}")
    n_edges = int(edge_index.shape[1])
    if n_edges:
        require_v541(bool((edge_index >= 0).all()
                          and (edge_index < n_nodes).all()),
                     "V541_PROVIDER_EDGE_RANGE",
                     f"edge_index endpoints must lie in [0,{n_nodes})")
        self_loop = edge_index[0] == edge_index[1]
        require_v541(not bool(self_loop.any()),
                     "V541_PROVIDER_SELF_LOOP",
                     "edge_index contains a self loop (legacy main graph "
                     "never emits one)")
        g_src = graph_index[edge_index[0]]
        g_tgt = graph_index[edge_index[1]]
        cross = g_src != g_tgt
        if bool(cross.any()):
            e = int(torch.nonzero(cross, as_tuple=False)[0, 0])
            require_v541(
                False, "V541_PROVIDER_CROSS_GRAPH",
                f"edge {e} = ({int(edge_index[0, e])}, {int(edge_index[1, e])}) "
                f"spans graphs {int(g_src[e])} -> {int(g_tgt[e])}; legal "
                f"views are strictly within-graph (plan §6.2)")

    n_graphs = count_graphs(graph_index) if diagnostics else 0

    if not with_chem_scalars:
        return ForwardBatchData(
            pos=pos, graph_index=graph_index, edge_index=edge_index,
            edge_attr=None, b_src=None, b_tgt=None, delta_b_intent=None,
            abs_delta=None, is_changed=None,
            n_nodes=n_nodes, n_edges=n_edges, n_graphs=n_graphs)

    edge_attr = _read_tensor(batch, EDGE_ATTR_KEY)
    require_v541(edge_attr.ndim == 2
                 and tuple(edge_attr.shape) == (n_edges, EDGE_PACK_DIM_V3),
                 "V541_PROVIDER_SHAPE",
                 f"edge_attr must be [{n_edges},{EDGE_PACK_DIM_V3}], "
                 f"got {tuple(edge_attr.shape)}")
    require_v541(bool(torch.is_floating_point(edge_attr)),
                 "V541_PROVIDER_DTYPE",
                 f"edge_attr must be floating point, got {edge_attr.dtype}")

    scalars: dict[str, torch.Tensor] = {}
    for key in (EDGE_B_SRC_KEY, EDGE_B_TGT_KEY, EDGE_DELTA_B_KEY,
                EDGE_ABS_DELTA_KEY, EDGE_IS_CHANGED_KEY):
        value = _read_tensor(batch, key).reshape(-1)
        require_v541(int(value.shape[0]) == n_edges,
                     "V541_PROVIDER_SHAPE",
                     f"{key} must be [{n_edges}], got {tuple(value.shape)}")
        require_v541(bool(torch.is_floating_point(value)),
                     "V541_PROVIDER_DTYPE",
                     f"{key} must be floating point, got {value.dtype}")
        scalars[key] = value

    return ForwardBatchData(
        pos=pos, graph_index=graph_index, edge_index=edge_index,
        edge_attr=edge_attr,
        b_src=scalars[EDGE_B_SRC_KEY], b_tgt=scalars[EDGE_B_TGT_KEY],
        delta_b_intent=scalars[EDGE_DELTA_B_KEY],
        abs_delta=scalars[EDGE_ABS_DELTA_KEY],
        is_changed=scalars[EDGE_IS_CHANGED_KEY],
        n_nodes=n_nodes, n_edges=n_edges, n_graphs=n_graphs)


def chem_attr_matrix(data: ForwardBatchData) -> torch.Tensor:
    """``[E, 8]`` the frozen chem-edge fields, in ``CHEM_EDGE_FIELDS`` order.

    Sources: the five forward-view edge scalars plus ``edge_attr`` pack
    slots ``covalent_union`` (13), ``b_min`` (15), ``b_max`` (16).  The
    first 12 ``edge_attr`` slots (F2-masked RBF) are structurally never
    touched here.
    """
    require_v541(data.edge_attr is not None and data.b_src is not None,
                 "V541_PROVIDER_MISSING_KEY",
                 "chem scalars were not read (with_chem_scalars=False)")
    slots = EDGE_PACK_SLOTS_V3
    return torch.stack((
        data.b_src,                      # type: ignore[misc]
        data.b_tgt,                      # type: ignore[misc]
        data.delta_b_intent,             # type: ignore[misc]
        data.abs_delta,                  # type: ignore[misc]
        data.is_changed,                 # type: ignore[misc]
        data.edge_attr[:, slots["covalent_union"]],
        data.edge_attr[:, slots["b_min"]],
        data.edge_attr[:, slots["b_max"]],
    ), dim=1)


def chem_qualifies(attrs: torch.Tensor) -> torch.Tensor:
    """Per-row predicate: pair belongs to G_chem (source/target covalent
    union OR reaction edit)."""
    i = _CHEM_FIELD_INDEX
    return ((attrs[:, i["b_src"]] > 0)
            | (attrs[:, i["b_tgt"]] > 0)
            | (attrs[:, i["is_changed"]] > 0)
            | (attrs[:, i["covalent_union"]] > 0))


def distances_of_pairs(
        pos: torch.Tensor, pair_index: torch.Tensor) -> torch.Tensor:
    """Physical Å distances ``‖pos[a] − pos[b]‖`` for ``[2, P]`` pairs."""
    if int(pair_index.shape[1]) == 0:
        return pos.new_zeros((0,))
    diff = pos[pair_index[0]] - pos[pair_index[1]]
    dist = safe_distances(diff)
    if not bool(torch.isfinite(dist).all()):
        bad = int(torch.nonzero(~torch.isfinite(dist),
                                as_tuple=False)[0, 0])
        a = int(pair_index[0, bad])
        b = int(pair_index[1, bad])
        raise FeatureProviderError(
            "V541_PROVIDER_DIST_NONFINITE",
            f"distance for pair ({a},{b}) overflows the float32 range; "
            f"coordinates are finite but pairwise distances are not "
            f"(plan §6.4 extreme-but-legal bound)")
    return dist


__all__ = [
    "ANGLE_COS_TOL",
    "ANGLE_MAX_NEIGHBORS",
    "ANGLE_MAX_PAIRS",
    "ANGLE_SHELL_TIE_TOL",
    "ANGLE_TIE_POLICY",
    "CHEM_EDGE_DIM",
    "CHEM_EDGE_FIELDS",
    "COS_CLAMP_MAX",
    "COS_CLAMP_MIN",
    "COS_DENOM_EPS",
    "DEFAULT_CHUNK_SIZE",
    "DIST_BLOCK_BUDGET",
    "FeatureProviderError",
    "ForwardBatchData",
    "RBF_CENTER_END",
    "RBF_CENTER_START",
    "RBF_N",
    "RBF_SIGMA_ANGSTROM",
    "RBF_T_CLAMP",
    "chem_attr_matrix",
    "chem_qualifies",
    "count_graphs",
    "diagnostic_sum",
    "distances_of_pairs",
    "expand_pair_directions",
    "rbf12",
    "read_forward_batch",
    "require_v541",
    "safe_distances",
    "undirected_pair_keys",
]
