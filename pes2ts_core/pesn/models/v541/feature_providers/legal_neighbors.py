"""V5.4.1 WP3a — legal neighbour views ``G_chem`` / ``G_spatial``
(plan §6.2, §6.6.2; config ``embedding_contract.feature_recipe``).

Pure-PyTorch, lazy (stateless functions), no third-party GNN dependency.
Both views are DERIVED from the forward-view batch only — no new forward
key, no supervision/metadata read, no reference/X0 geometry.

``G_chem`` (plan §6.2 / §6.6.4 — the shared E1/E2 internal view)
    The legal source/target covalent union (``G_src ∪ G_tgt``) plus the
    reaction-edit pairs, deduplicated to one undirected pair per directed
    atom pair and expanded to BOTH message directions with identical
    attributes: swapping the message direction never flips the signed
    bond delta — only a reversed INTENT (swapped ``b_src``/``b_tgt`` in
    the input) changes it (plan §6.6.3).  Duplicate directed edges are
    merged only when all 8 frozen fields are bitwise equal; any
    inconsistency is a typed refusal.  Cross-graph edges, self loops and
    out-of-range endpoints are refused up front.

``G_spatial`` (plan §6.6.2 — the E3 internal view)
    The subset of the main graph obtained by RE-RUNNING the legacy
    spatial selection on the current ``pos``: per atom, candidates with
    ``r < 4.5 Å``, the ≤32 nearest, and 1e-6 Å boundary-tie retention
    (``key_atom_graph_v3`` recipe), then bidirectionalized.  Consequences
    frozen by the plan:

    * covalent pairs kept by the legacy builder BEYOND the cutoff are
      never spatial edges (they fail ``r < 4.5`` here) — E3 must not
      mistake them for spatial neighbours (§6.6.2);
    * covalent pairs WITHIN the cutoff WERE part of the legacy radius
      selection and therefore are spatial edges (the selection, not the
      ``is_spatial`` provenance flag, defines membership);
    * bidirectionalization + tie retention can push the actual degree
      above the nominal 32 — the excess is counted, never re-truncated
      by atom id;
    * the recomputed selection is intersected with the input edge set so
      the view is a SUBSET of the main graph by construction; selected
      pairs missing from the input (float32-vs-float64 rank flips at the
      truncation boundary) are counted in
      ``counts["n_selected_not_in_input"]`` instead of silently
      accepted or hard-failed.

Every view carries physical-Å distances, the frozen RBF12 expansion, an
explicit validity mask and detached Python-float counts.  Distances are
recomputed from ``pos`` — the F2-masked first 12 ``edge_attr`` slots are
never read (plan §6.6.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Mapping

import torch

from pes2ts_core.pesn.features.key_atom_spec_v3 import (
    SPATIAL_CUTOFF,
    SPATIAL_MAX_NEIGHBORS,
    SPATIAL_TIE_TOL,
)
from pes2ts_core.pesn.models.v541.feature_providers.common import (
    CHEM_EDGE_DIM,
    DEFAULT_CHUNK_SIZE,
    DIST_BLOCK_BUDGET,
    FeatureProviderError,
    ForwardBatchData,
    chem_attr_matrix,
    chem_qualifies,
    diagnostic_sum,
    distances_of_pairs,
    expand_pair_directions,
    rbf12,
    read_forward_batch,
    require_v541,
    undirected_pair_keys,
)

#: Frozen legacy spatial recipe (single source: spec v3 constants above;
#: mirrored here so tests/docs can print the exact view definition).
SPATIAL_RECIPE: Final[dict[str, float | int]] = {
    "cutoff_angstrom": SPATIAL_CUTOFF,
    "max_neighbors": SPATIAL_MAX_NEIGHBORS,
    "tie_tolerance_angstrom": SPATIAL_TIE_TOL,
}


def _counts(**values: float) -> dict[str, float]:
    """Detached Python-float diagnostics (never tensors — plan §6.4 /
    ``v541.interfaces`` convention: no autograd graph may be retained)."""
    return {name: float(value) for name, value in values.items()}


@dataclass(frozen=True)
class ChemGraphView:
    """``G_chem``: deduplicated bidirectional covalent-union + edit graph.

    Schema (``pair_*`` = one row per undirected pair, sorted by
    ``(a, b)``; ``edge_*`` = both message directions, legacy layout
    ``2p = a→b``, ``2p+1 = b→a`` with bitwise-identical attributes):

    * ``pair_index`` ``[2, P]`` int64 (row0 < row1);
    * ``pair_attr`` ``[P, 8]`` float — ``CHEM_EDGE_FIELDS`` order;
    * ``pair_distance`` ``[P]`` float (physical Å);
    * ``pair_rbf12`` ``[P, 12]`` float (frozen Gaussian expansion);
    * ``edge_index`` ``[2, 2P]``, ``edge_attr`` ``[2P, 8]``,
      ``distance`` ``[2P]``, ``rbf12`` ``[2P, 12]``, ``valid`` ``[2P]``
      bool (every emitted edge is real and finite);
    * ``counts`` — detached Python floats.
    """

    pair_index: torch.Tensor
    pair_attr: torch.Tensor
    pair_distance: torch.Tensor
    pair_rbf12: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    distance: torch.Tensor
    rbf12: torch.Tensor
    valid: torch.Tensor
    counts: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SpatialGraphView:
    """``G_spatial``: legacy radius-selection subset of the main graph.

    Same row layout as :class:`ChemGraphView` minus the chem attributes
    (E3's continuous filter consumes distance/RBF only, plan §6.7):
    ``pair_index`` ``[2, Q]``, ``pair_distance`` ``[Q]``,
    ``pair_rbf12`` ``[Q, 12]``, ``edge_index`` ``[2, 2Q]``,
    ``distance`` ``[2Q]``, ``rbf12`` ``[2Q, 12]``, ``valid`` ``[2Q]``,
    ``counts``.
    """

    pair_index: torch.Tensor
    pair_distance: torch.Tensor
    pair_rbf12: torch.Tensor
    edge_index: torch.Tensor
    distance: torch.Tensor
    rbf12: torch.Tensor
    valid: torch.Tensor
    counts: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class LegalNeighborViews:
    """Both legal views of one batch (``build_legal_neighbor_views``)."""

    chem: ChemGraphView
    spatial: SpatialGraphView


def _pair_features(
        pos: torch.Tensor, pair_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Physical-Å distances + RBF12 + validity for ``[2, P]`` pairs."""
    dist = distances_of_pairs(pos, pair_index)
    return dist, rbf12(dist), torch.isfinite(dist)


# ---------------------------------------------------------------------------
# G_chem
# ---------------------------------------------------------------------------
def build_chem_view(
        data: ForwardBatchData, *, diagnostics: bool = True,
) -> ChemGraphView:
    """Build ``G_chem`` from an already-validated forward batch.

    Grouping: directed input edges are keyed by the undirected pair
    ``min(i,j)·N + max(i,j)``.  Within a group all 8 frozen fields must
    be BITWISE equal (duplicates merge; any mismatch is
    ``V541_PROVIDER_EDGE_ATTR_MISMATCH``).  A group qualifies iff its
    attributes say covalent union OR reaction edit; qualifying groups are
    emitted once as an undirected pair and expanded to both directions
    with identical attributes (plan §6.6.3 — the signed delta belongs to
    the intent, not to the message direction).

    ``diagnostics=False`` returns ``counts={}`` and skips every
    count-only reduction (the ``n_directions_synthesized`` host sync);
    features, row order, dtypes and gradients are bitwise identical and
    all legality checks above still run.
    """
    pos, edge_index = data.pos, data.edge_index
    n_nodes, n_edges = data.n_nodes, data.n_edges

    if n_edges == 0:
        empty_counts = _counts(
            n_atoms=n_nodes, n_graphs=data.n_graphs, n_input_edges=n_edges,
            n_input_undirected_pairs=0, n_chem_pairs=0, n_chem_edges=0,
            n_duplicate_edges_merged=0, n_non_chem_pairs=0,
            n_directions_synthesized=0) if diagnostics else {}
        return ChemGraphView(
            pair_index=torch.zeros(2, 0, dtype=torch.long, device=pos.device),
            pair_attr=pos.new_zeros((0, CHEM_EDGE_DIM)),
            pair_distance=pos.new_zeros((0,)),
            pair_rbf12=pos.new_zeros((0, 12)),
            edge_index=torch.zeros(2, 0, dtype=torch.long, device=pos.device),
            edge_attr=pos.new_zeros((0, CHEM_EDGE_DIM)),
            distance=pos.new_zeros((0,)),
            rbf12=pos.new_zeros((0, 12)),
            valid=torch.zeros(0, dtype=torch.bool, device=pos.device),
            counts=empty_counts)

    attrs = chem_attr_matrix(data)
    require_v541(bool(torch.isfinite(attrs).all()),
                 "V541_PROVIDER_EDGE_ATTR_NONFINITE",
                 "chem edge fields (b_src/b_tgt/delta_b_intent/abs_delta/"
                 "is_changed/covalent_union/b_min/b_max) contain "
                 "non-finite values")

    src, tgt = edge_index[0], edge_index[1]
    keys = torch.minimum(src, tgt) * n_nodes + torch.maximum(src, tgt)
    order = torch.argsort(keys, stable=True)
    keys_sorted = keys[order]
    attrs_sorted = attrs[order]
    uniq, group_counts = torch.unique_consecutive(
        keys_sorted, return_counts=True)
    starts = torch.cumsum(group_counts, dim=0) - group_counts
    first_attrs = attrs_sorted[starts]                # [G, 8]
    reference = first_attrs.repeat_interleave(group_counts, dim=0)
    mismatch = (attrs_sorted != reference).any(dim=1)
    if bool(mismatch.any()):
        at = int(torch.nonzero(mismatch, as_tuple=False)[0, 0])
        a, b = divmod(int(keys_sorted[at]), n_nodes)
        raise FeatureProviderError(
            "V541_PROVIDER_EDGE_ATTR_MISMATCH",
            f"duplicate directed edges of pair ({a},{b}) carry "
            f"inconsistent chem-edge attributes; identical duplicates "
            f"merge, inconsistent ones are refused (plan §6.6.3/§6.6.4)")

    qualifies = chem_qualifies(first_attrs)           # [G]
    pair_keys = uniq[qualifies]
    pair_attr = first_attrs[qualifies]
    pair_index = torch.stack(
        (pair_keys // n_nodes, pair_keys % n_nodes), dim=0)
    dist, rbf, valid = _pair_features(pos, pair_index)
    n_pairs = int(pair_index.shape[1])
    if diagnostics:
        n_groups = int(uniq.numel())
        counts_out = _counts(
            n_atoms=n_nodes, n_graphs=data.n_graphs, n_input_edges=n_edges,
            n_input_undirected_pairs=n_groups, n_chem_pairs=n_pairs,
            n_chem_edges=2 * n_pairs,
            n_duplicate_edges_merged=n_edges - n_groups,
            n_non_chem_pairs=n_groups - n_pairs,
            n_directions_synthesized=2 * n_pairs
            - diagnostic_sum(group_counts[qualifies]))
    else:
        counts_out = {}
    return ChemGraphView(
        pair_index=pair_index, pair_attr=pair_attr, pair_distance=dist,
        pair_rbf12=rbf,
        edge_index=expand_pair_directions(pair_index),
        edge_attr=pair_attr.repeat_interleave(2, dim=0),
        distance=dist.repeat_interleave(2),
        rbf12=rbf.repeat_interleave(2, dim=0),
        valid=valid.repeat_interleave(2),
        counts=counts_out)


def build_chem_graph(
        batch: Mapping[str, object], *, diagnostics: bool = True,
) -> ChemGraphView:
    """Public entry: ``G_chem`` from a forward-view batch mapping.

    ``diagnostics=True`` (default) fills ``counts`` with detached Python
    floats at the cost of GPU→host reductions; ``diagnostics=False``
    skips all count-only work — features/order/gradients bitwise
    identical, ``counts={}``, every legality check unchanged.
    """
    data = read_forward_batch(
        batch, with_chem_scalars=True, diagnostics=diagnostics)
    return build_chem_view(data, diagnostics=diagnostics)


# ---------------------------------------------------------------------------
# G_spatial
# ---------------------------------------------------------------------------
def build_spatial_view(
        data: ForwardBatchData, *, chunk_size: int = DEFAULT_CHUNK_SIZE,
        diagnostics: bool = True,
) -> SpatialGraphView:
    """Build ``G_spatial`` from an already-validated forward batch.

    Per graph (never across graphs): every atom queries its co-graph
    atoms; candidates are ``r < 4.5 Å`` (strict, as legacy), the
    ``min(32, M)``-th smallest distance is the threshold, and everything
    within ``threshold + 1e-6 Å`` is retained (boundary ties — realised
    degree may exceed 32).  Query atoms are processed in chunks bounded
    by :data:`DIST_BLOCK_BUDGET` so one distance block never exceeds the
    budget regardless of graph size.  The resulting undirected pairs are
    intersected with the input edge set (subset discipline, plan §6.6.2)
    and expanded to both directions.

    ``diagnostics=False`` returns ``counts={}`` and skips every
    count-only reduction: the O(E) input-pair beyond-cutoff inventory,
    the ``n_over_cap`` chain and the ``n_selected_not_in_input`` sum all
    disappear (three GPU→host syncs).  The selection itself, the
    membership intersection and all features stay bitwise identical, and
    every legality check still runs.
    """
    require_v541(int(chunk_size) >= 1,
                 "V541_PROVIDER_PARAM",
                 f"chunk_size must be >= 1, got {chunk_size!r}")
    pos, edge_index = data.pos, data.edge_index
    n_nodes, n_edges = data.n_nodes, data.n_edges
    # Selection is discrete (cutoff / k-th value / tie comparisons): run
    # it on a DETACHED view so no autograd graph is ever built for it.
    # Emitted-pair features below recompute distances from the ATTACHED
    # pos so the outputs stay differentiable (plan §6.3.8 — nothing is
    # cached either way).
    pos_sel = pos.detach()

    # Membership keys are ALWAYS needed (subset discipline); the
    # beyond-cutoff inventory of the input pairs is counts-only.
    input_keys = undirected_pair_keys(edge_index, n_nodes)
    n_input_pairs = 0
    n_beyond = 0
    if diagnostics:
        n_input_pairs = int(input_keys.numel())
        if n_input_pairs:
            input_index = torch.stack(
                (input_keys // n_nodes, input_keys % n_nodes), dim=0)
            input_dist = distances_of_pairs(pos_sel, input_index)
            n_beyond = diagnostic_sum(input_dist >= SPATIAL_CUTOFF)

    selected_blocks: list[torch.Tensor] = []
    n_over_cap = 0
    for graph_id in torch.unique(data.graph_index).tolist():
        nodes = torch.nonzero(
            data.graph_index == graph_id, as_tuple=False).reshape(-1)
        m = int(nodes.numel())
        if m < 2:
            continue
        k = min(SPATIAL_MAX_NEIGHBORS, m)
        pos_graph = pos_sel[nodes]                     # [m, 3]
        step = max(1, min(int(chunk_size), DIST_BLOCK_BUDGET // m))
        for start in range(0, m, step):
            queries = nodes[start:start + step]
            # Legacy candidate math on the CURRENT frame (Å): strict
            # cutoff, k-th-smallest threshold, 1e-6 Å tie retention.
            # Manual per-axis accumulation (no cdist): exact for extreme
            # coordinates (no x²−2xy+y² cancellation), one [c, m] block.
            sq = pos_sel[queries].new_zeros((int(queries.numel()), m))
            for axis in range(3):
                delta = (pos_sel[queries][:, axis:axis + 1]
                         - pos_graph[None, :, axis])
                sq += delta * delta
            dist = sq.sqrt()                              # [c, m]
            self_mask = queries[:, None] == nodes[None, :]
            keep = (dist < SPATIAL_CUTOFF) & (~self_mask)
            values = torch.where(
                keep, dist, torch.full_like(dist, float("inf")))
            threshold = torch.kthvalue(values, k, dim=1).values  # [c]
            selected = keep & (
                dist <= (threshold + SPATIAL_TIE_TOL).unsqueeze(1))
            if diagnostics:
                # count-only: realised degree above the nominal cap
                # comes from ties at/around the k-th value — counted,
                # never re-truncated (§6.6.2).
                n_valid = keep.sum(dim=1)
                cap = torch.minimum(torch.full_like(n_valid, k), n_valid)
                n_over_cap += diagnostic_sum(
                    (selected.sum(dim=1) - cap).clamp_min(0))
            q_local, g_local = torch.nonzero(selected, as_tuple=True)
            sel_src = queries[q_local]
            sel_tgt = nodes[g_local]
            a = torch.minimum(sel_src, sel_tgt)
            b = torch.maximum(sel_src, sel_tgt)
            selected_blocks.append(a * n_nodes + b)

    if selected_blocks:
        selected_keys = torch.unique(torch.cat(selected_blocks))
    else:
        selected_keys = torch.zeros(0, dtype=torch.long, device=pos.device)
    in_input = torch.isin(selected_keys, input_keys)
    n_missing = diagnostic_sum(~in_input) if diagnostics else 0
    kept_keys = selected_keys[in_input]
    pair_index = torch.stack(
        (kept_keys // n_nodes, kept_keys % n_nodes), dim=0)
    dist, rbf, valid = _pair_features(pos, pair_index)
    n_pairs = int(pair_index.shape[1])
    counts_out = _counts(
        n_atoms=n_nodes, n_graphs=data.n_graphs, n_input_edges=n_edges,
        n_input_undirected_pairs=n_input_pairs,
        n_spatial_pairs=n_pairs, n_spatial_edges=2 * n_pairs,
        n_over_cap_directed=n_over_cap,
        n_selected_not_in_input=n_missing,
        n_input_pairs_beyond_cutoff=n_beyond) if diagnostics else {}
    if n_pairs == 0:
        return SpatialGraphView(
            pair_index=torch.zeros(2, 0, dtype=torch.long, device=pos.device),
            pair_distance=pos.new_zeros((0,)),
            pair_rbf12=pos.new_zeros((0, 12)),
            edge_index=torch.zeros(2, 0, dtype=torch.long, device=pos.device),
            distance=pos.new_zeros((0,)),
            rbf12=pos.new_zeros((0, 12)),
            valid=torch.zeros(0, dtype=torch.bool, device=pos.device),
            counts=counts_out)
    return SpatialGraphView(
        pair_index=pair_index, pair_distance=dist, pair_rbf12=rbf,
        edge_index=expand_pair_directions(pair_index),
        distance=dist.repeat_interleave(2),
        rbf12=rbf.repeat_interleave(2, dim=0),
        valid=valid.repeat_interleave(2),
        counts=counts_out)


def build_spatial_graph(
        batch: Mapping[str, object], *, chunk_size: int = DEFAULT_CHUNK_SIZE,
        diagnostics: bool = True,
) -> SpatialGraphView:
    """Public entry: ``G_spatial`` from a forward-view batch mapping.

    ``diagnostics=False`` ⇒ ``counts={}`` and no count-only host sync;
    features/order/gradients bitwise identical, legality unchanged.
    """
    data = read_forward_batch(
        batch, with_chem_scalars=False, diagnostics=diagnostics)
    return build_spatial_view(
        data, chunk_size=chunk_size, diagnostics=diagnostics)


def build_legal_neighbor_views(
        batch: Mapping[str, object], *, chunk_size: int = DEFAULT_CHUNK_SIZE,
        diagnostics: bool = True,
) -> LegalNeighborViews:
    """Both legal views of one batch (single read, one validation pass).

    ``diagnostics=False`` propagates to the shared read and both views:
    ``counts={}`` on each, no count-only GPU→host reduction, bitwise
    identical features/order/gradients, all legality checks unchanged.
    """
    data = read_forward_batch(
        batch, with_chem_scalars=True, diagnostics=diagnostics)
    return LegalNeighborViews(
        chem=build_chem_view(data, diagnostics=diagnostics),
        spatial=build_spatial_view(
            data, chunk_size=chunk_size, diagnostics=diagnostics))


__all__ = [
    "ChemGraphView",
    "LegalNeighborViews",
    "SPATIAL_RECIPE",
    "SpatialGraphView",
    "build_chem_graph",
    "build_chem_view",
    "build_legal_neighbor_views",
    "build_spatial_graph",
    "build_spatial_view",
]
