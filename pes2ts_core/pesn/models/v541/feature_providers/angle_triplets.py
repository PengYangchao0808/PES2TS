"""V5.4.1 WP3a — preregistered angle triplets ``G_angle`` (plan §6.3;
config ``embedding_contract.feature_recipe.angle``).

Pure-PyTorch, lazy (stateless), no third-party GNN dependency, bounded
memory via centre chunking, and NO cache of tensors carrying an autograd
graph (plan §6.3.8 — callers own any caching and must detach).

Neighbour pool (plan §6.3.2): ONLY the legal ``G_chem`` / intent
neighbours — the source/target covalent union plus reaction-edit pairs
produced by :mod:`...feature_providers.legal_neighbors`.  Spatial-only
pairs are never angle neighbours; the spatial-angle extension is out of
the first batch (§6.3.2).

Selection (plan §6.3.3–§6.3.5):

* per centre at most :data:`ANGLE_MAX_NEIGHBORS` = 12 angle neighbours;
* neighbours are ranked by CURRENT distance only — never by atom id,
  curve id or input position;
* equal-distance neighbours form shells (ties within
  :data:`ANGLE_SHELL_TIE_TOL` = 1e-6 Å, the frozen legacy tie tolerance);
* shells are taken nearest-first while the running total stays ≤ 12;
  the first shell that would exceed the cap stops selection and is NOT
  partially taken — ``reject_boundary_tie_shell``: a shell whose
  inclusion would breach the cap is rejected whole, so no atom id can
  decide which tied neighbour survives.  Rejected-shell diagnostics are
  retained (``counts["n_boundary_*"]``).
* every unordered pair ``{j, k}`` of the selected set is emitted
  (≤ C(12, 2) = 66 per centre); neighbour ranking uses distance ONLY —
  never atom id / curve id / input position.  The PAIR is unordered and
  all pair-level features are strictly symmetric in j/k.

Geometry (plan §6.3.6 / §6.7 E4 spec):

* ``cos θ_jik`` from the centre-to-neighbour vectors with a guarded
  denominator and a final clamp to ``[-1, 1]``; a ZERO-length side is a
  typed refusal (``V541_PROVIDER_ZERO_DISTANCE``) — never a silent drop;
* the frozen basis ``[1, c, c², c³]``;
* both sides carry their physical-Å distance, frozen RBF12 and the
  8-field chem-edge attributes (recomputed/looked up from ``G_chem``,
  never from the F2-masked ``edge_attr`` slots);
* centres without a usable angle carry an explicit
  ``center_has_angle`` mask (plan §6.3.4 — no learned missing-angle
  token, no id-dependent truncation).

Diagnostics: ``counts`` are detached Python floats — boundary/tie
rejections, near-zero-angle and collinear (degenerate) triplet counts,
clamp events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch

from pes2ts_core.pesn.models.v541.feature_providers.common import (
    ANGLE_COS_TOL,
    ANGLE_MAX_NEIGHBORS,
    ANGLE_MAX_PAIRS,
    ANGLE_SHELL_TIE_TOL,
    ANGLE_TIE_POLICY,
    CHEM_EDGE_DIM,
    COS_CLAMP_MAX,
    COS_CLAMP_MIN,
    COS_DENOM_EPS,
    DEFAULT_CHUNK_SIZE,
    FeatureProviderError,
    diagnostic_sum,
    read_forward_batch,
    require_v541,
)
from pes2ts_core.pesn.models.v541.feature_providers.legal_neighbors import (
    build_chem_view,
)


@dataclass(frozen=True)
class AngleTripletView:
    """Preregistered angle triplets of one batch.

    Row schema (``T`` emitted triplets over ``N`` atoms):

    * ``center_index`` ``[T]`` int64 — the angle centre ``i``;
    * ``side_index`` ``[T, 2]`` int64 — neighbours ``(j, k)``.  Their
      storage order has no semantic role: the pair itself is unordered;
      swap j/k leaves every pair-level feature unchanged;
    * ``side_distance`` ``[T, 2]`` float — physical Å ``r_ij``, ``r_ik``;
    * ``side_rbf12`` ``[T, 2, 12]`` float — frozen Gaussian expansion of
      each side distance;
    * ``side_chem_attr`` ``[T, 2, 8]`` float — ``CHEM_EDGE_FIELDS`` of
      edges ``(i,j)`` / ``(i,k)`` (identical on both message directions);
    * ``cos_theta`` ``[T]`` float — clamped to ``[-1, 1]``;
    * ``cos_basis`` ``[T, 4]`` — ``[1, c, c², c³]``;
    * ``triplet_valid`` ``[T]`` bool — every emitted triplet is real and
      finite by construction (explicit contract surface);
    * ``center_has_angle`` ``[N]`` bool — explicit no-angle mask;
    * ``angle_neighbor_count`` ``[N]`` int64 — selected neighbours
      (0…12) per centre;
    * ``counts`` — detached Python floats.
    """

    center_index: torch.Tensor
    side_index: torch.Tensor
    side_distance: torch.Tensor
    side_rbf12: torch.Tensor
    side_chem_attr: torch.Tensor
    cos_theta: torch.Tensor
    cos_basis: torch.Tensor
    triplet_valid: torch.Tensor
    center_has_angle: torch.Tensor
    angle_neighbor_count: torch.Tensor
    counts: dict[str, float] = field(default_factory=dict)


def _counts(**values: float) -> dict[str, float]:
    return {name: float(value) for name, value in values.items()}


def _select_shells(
        items: list[tuple[float, int, int]],
) -> tuple[list[tuple[float, int, int]], int, int, int, int]:
    """Distance-shell selection with ``reject_boundary_tie_shell``.

    ``items`` are ``(distance, atom_index, chem_edge_row)`` sorted by
    distance ONLY (never atom id / input position).  Returns ``(selected, boundary_shells,
    boundary_tie_shells, boundary_neighbors, boundary_tie_neighbors)``.
    Shells group consecutive items whose distance stays within
    :data:`ANGLE_SHELL_TIE_TOL` of the shell's first member; selection is
    distance-monotone and STOPS at the first shell that would exceed the
    cap — the boundary shell is rejected whole (never partially taken),
    and shells beyond it are not examined.
    """
    shells: list[list[tuple[float, int, int]]] = []
    for item in items:
        if shells and (item[0] - shells[-1][0][0]) <= ANGLE_SHELL_TIE_TOL:
            shells[-1].append(item)
        else:
            shells.append([item])
    selected: list[tuple[float, int, int]] = []
    acc = 0
    b_shells = b_tie_shells = b_neighbors = b_tie_neighbors = 0
    for shell in shells:
        if acc + len(shell) <= ANGLE_MAX_NEIGHBORS:
            selected.extend(shell)
            acc += len(shell)
        else:
            # Boundary shell rejected WHOLE (tie policy) + stop.
            b_shells = 1
            b_neighbors = len(shell)
            if len(shell) > 1:
                b_tie_shells = 1
                b_tie_neighbors = len(shell)
            break
    return selected, b_shells, b_tie_shells, b_neighbors, b_tie_neighbors


def build_angle_triplets(
        batch: Mapping[str, object], *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        diagnostics: bool = True,
) -> AngleTripletView:
    """Build the preregistered angle triplets of one forward-view batch.

    ``G_chem`` is rebuilt from the same batch on every call (lazy, no
    cross-view cache — staleness by construction is impossible, plan
    §6.6.6).  Centres are processed in chunks of ``chunk_size`` so peak
    intermediate memory stays bounded; nothing is cached between calls
    and no tensor with a live autograd graph is retained.

    ``diagnostics=False`` returns ``counts={}`` and skips every
    count-only reduction (boundary/tie accumulation, the zero-angle /
    collinear / clamp host sums, the ``has_angle``/``neighbor_count``
    sums, the shared graph-count sync): every emitted tensor — order,
    shape, dtype and gradient — is bitwise identical to the
    ``diagnostics=True`` build, while the ZERO-LENGTH-side typed refusal
    and all other legality checks still run in both modes.
    """
    require_v541(int(chunk_size) >= 1,
                 "V541_PROVIDER_PARAM",
                 f"chunk_size must be >= 1, got {chunk_size!r}")
    data = read_forward_batch(
        batch, with_chem_scalars=True, diagnostics=diagnostics)
    chem = build_chem_view(data, diagnostics=diagnostics)
    pos = data.pos
    n_nodes, n_graphs = data.n_nodes, data.n_graphs

    # CSR over the DIRECTED chem edges grouped by message source.
    src = chem.edge_index[0]
    dst = chem.edge_index[1]
    order = torch.argsort(src, stable=True)
    dst_sorted = dst[order]
    dist_sorted = chem.distance[order]
    # Shell selection is discrete — read distances detached (no scalar
    # conversion of a grad-tracking tensor, no graph retention).
    dist_rank = dist_sorted.detach()
    rbf_sorted = chem.rbf12[order]
    attr_sorted = chem.edge_attr[order]
    degree = torch.bincount(src, minlength=n_nodes)
    offsets = torch.cat(
        (torch.zeros(1, dtype=torch.long, device=pos.device),
         torch.cumsum(degree, dim=0)))

    selected_counts: list[int] = []
    n_boundary_shells = 0
    n_boundary_tie_shells = 0
    n_boundary_neighbors = 0
    n_boundary_tie_neighbors = 0
    n_zero_angle = 0
    n_collinear = 0
    n_clamped_cos = 0
    chunks: list[tuple[torch.Tensor, ...]] = []

    for chunk_start in range(0, n_nodes, int(chunk_size)):
        chunk_end = min(chunk_start + int(chunk_size), n_nodes)
        centers: list[int] = []
        nbr0: list[int] = []
        nbr1: list[int] = []
        row0: list[int] = []
        row1: list[int] = []
        for i in range(chunk_start, chunk_end):
            lo = int(offsets[i])
            hi = int(offsets[i + 1])
            items = [(float(dist_rank[r]), int(dst_sorted[r]), int(r))
                     for r in range(lo, hi)]
            items.sort(key=lambda t: t[0])
            selected, b_sh, b_tie_sh, b_nb, b_tie_nb = _select_shells(items)
            if diagnostics:
                n_boundary_shells += b_sh
                n_boundary_tie_shells += b_tie_sh
                n_boundary_neighbors += b_nb
                n_boundary_tie_neighbors += b_tie_nb
            selected_counts.append(len(selected))
            for x in range(len(selected)):
                for y in range(x + 1, len(selected)):
                    centers.append(i)
                    nbr0.append(selected[x][1])
                    nbr1.append(selected[y][1])
                    row0.append(selected[x][2])
                    row1.append(selected[y][2])
        if not centers:
            continue
        device = pos.device
        c_t = torch.tensor(centers, dtype=torch.long, device=device)
        n0_t = torch.tensor(nbr0, dtype=torch.long, device=device)
        n1_t = torch.tensor(nbr1, dtype=torch.long, device=device)
        r0_t = torch.tensor(row0, dtype=torch.long, device=device)
        r1_t = torch.tensor(row1, dtype=torch.long, device=device)

        side_distance = torch.stack(
            (dist_sorted[r0_t], dist_sorted[r1_t]), dim=1)
        side_rbf12 = torch.stack((rbf_sorted[r0_t], rbf_sorted[r1_t]), dim=1)
        side_attr = torch.stack((attr_sorted[r0_t], attr_sorted[r1_t]), dim=1)

        # Safe cosine: zero-length sides are a typed refusal (§6.3.6).
        u = pos[n0_t] - pos[c_t]
        v = pos[n1_t] - pos[c_t]
        su = (u * u).sum(dim=-1)
        sv = (v * v).sum(dim=-1)
        zero_u = su <= 0
        zero_v = sv <= 0
        zero = zero_u | zero_v
        if bool(zero.any()):
            at = int(torch.nonzero(zero, as_tuple=False)[0, 0])
            side = 0 if bool(zero_u[at]) else 1
            center = centers[at]
            neighbor = nbr0[at] if side == 0 else nbr1[at]
            raise FeatureProviderError(
                "V541_PROVIDER_ZERO_DISTANCE",
                f"zero-length angle side: centre {center} -> neighbour "
                f"{neighbor} has distance 0 Å; angle geometry requires "
                f"non-zero bond vectors (plan §6.3.6)")
        denom = (su.sqrt() * sv.sqrt()).clamp_min(COS_DENOM_EPS)
        raw = (u * v).sum(dim=-1) / denom
        cos = raw.clamp(COS_CLAMP_MIN, COS_CLAMP_MAX)
        ones = torch.ones_like(cos)
        basis = torch.stack((ones, cos, cos * cos, cos * cos * cos), dim=-1)

        if diagnostics:
            # count-only host reductions over the (already computed)
            # feature tensors — never legality, never features.
            n_clamped_cos += diagnostic_sum((raw > 1.0) | (raw < -1.0))
            n_zero_angle += diagnostic_sum(cos >= 1.0 - ANGLE_COS_TOL)
            n_collinear += diagnostic_sum(cos.abs() >= 1.0 - ANGLE_COS_TOL)
        chunks.append((c_t, n0_t, n1_t, side_distance, side_rbf12,
                       side_attr, cos, basis))

    # Per-centre selection always ran for every atom (even when no
    # triplet was emitted), so the count vector is complete here.
    neighbor_count = torch.tensor(
        selected_counts, dtype=torch.long, device=pos.device)
    has_angle = neighbor_count >= 2
    if not chunks:
        center_index = torch.zeros(0, dtype=torch.long, device=pos.device)
        side_index = torch.zeros(0, 2, dtype=torch.long, device=pos.device)
        side_distance = pos.new_zeros((0, 2))
        side_rbf12 = pos.new_zeros((0, 2, 12))
        side_attr = pos.new_zeros((0, 2, CHEM_EDGE_DIM))
        cos_theta = pos.new_zeros((0,))
        cos_basis = pos.new_zeros((0, 4))
    else:
        center_index = torch.cat([c[0] for c in chunks])
        side_index = torch.stack(
            (torch.cat([c[1] for c in chunks]),
             torch.cat([c[2] for c in chunks])), dim=1)
        side_distance = torch.cat([c[3] for c in chunks])
        side_rbf12 = torch.cat([c[4] for c in chunks])
        side_attr = torch.cat([c[5] for c in chunks])
        cos_theta = torch.cat([c[6] for c in chunks])
        cos_basis = torch.cat([c[7] for c in chunks])
    n_triplets = int(center_index.shape[0])
    if diagnostics:
        counts_out = _counts(
            n_atoms=n_nodes, n_graphs=n_graphs,
            n_chem_pairs=int(chem.pair_index.shape[1]),
            n_centers=n_nodes,
            n_centers_with_angle=diagnostic_sum(has_angle),
            n_angle_neighbors_selected=diagnostic_sum(neighbor_count),
            n_triplets=n_triplets,
            n_boundary_shells_rejected=n_boundary_shells,
            n_boundary_tie_shells_rejected=n_boundary_tie_shells,
            n_boundary_neighbors_rejected=n_boundary_neighbors,
            n_boundary_tie_neighbors_rejected=n_boundary_tie_neighbors,
            n_zero_angle=n_zero_angle, n_collinear=n_collinear,
            n_clamped_cos=n_clamped_cos)
    else:
        counts_out = {}
    return AngleTripletView(
        center_index=center_index, side_index=side_index,
        side_distance=side_distance, side_rbf12=side_rbf12,
        side_chem_attr=side_attr, cos_theta=cos_theta,
        cos_basis=cos_basis,
        triplet_valid=torch.ones(
            n_triplets, dtype=torch.bool, device=pos.device),
        center_has_angle=has_angle,
        angle_neighbor_count=neighbor_count,
        counts=counts_out)


__all__ = [
    "ANGLE_MAX_NEIGHBORS",
    "ANGLE_MAX_PAIRS",
    "ANGLE_SHELL_TIE_TOL",
    "ANGLE_TIE_POLICY",
    "AngleTripletView",
    "build_angle_triplets",
]
