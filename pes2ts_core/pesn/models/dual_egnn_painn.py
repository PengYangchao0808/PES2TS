"""PaiNN-style multi-channel equivariant encoder over the dual-topology union
graph (plan v4 §7 V1/V2, contract §5 V1/V2, ADR-0024 item 4).

Replaces the single-vector EGNN accumulation with 16 PaiNN-style vector
channels that propagate between layers and feed back into the scalar update:

* vector message: neighbour vector channels ``U_j`` transferred through
  radial scalar gates, plus a radial vector message ``rhat_ij · W(h_j)``;
* scalar coupling: ``rhat_ij·U`` and ``U_i·U_j`` inner products feed the
  scalar update (PaiNN mixing);
* scalar-gated vector update: each channel is scaled by a per-node scalar
  gate after the residual vector message.

E(3) polar-vector discipline: no single cross products anywhere (a single
cross product is an axial vector and would break reflection covariance), and
``LayerNorm`` is applied only to the invariant scalar axis ``hidden`` — never
to the spatial axes of the vector channels.

Input interface matches ``DualTSEGNN`` exactly (``edge_index`` bidirectional,
``node_scalars`` 5ch, ``edge_feats`` 4ch or 7ch endpoint blocks, ``g_scaled``
as an extra vector input channel alongside ``X``).  ``contact_edges=True``
adds per-state radius-5 Å contact edges (V2): chemical union edges are never
deleted, the smooth envelope multiplies ONLY the added contact branch, and
each atom keeps at most 16 nearest contact neighbours (sorted by
``(dist, atom_idx)``, deterministic).

Outputs ``(h, v)`` where ``v`` is the 16 vector channels aggregated through a
readout gate into a single polar vector ``(N, 3)`` (the raw ``(N, 16, 3)``
channels are also returned under ``v_channels`` for inspection).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from pes2ts_core.pesn.features.dual_ts_graph import (
    N_NODE_SCALARS,
    DualTSEncoderConfig,
    DualTSFusionEncoder,
)
from pes2ts_core.pesn.models.dual_egnn import gaussian_distance_basis

DUAL_EGNN_PAINN_VERSION: Final[str] = "dual_egnn_painn_v1"
MAX_Z: Final[int] = 32


@dataclass(frozen=True, slots=True)
class DualTSPaiNNEncoderConfig:
    """Sizes for the PaiNN-style dual-TS encoder.

    ``light`` is NOT supported here (V1/V2 are built with explicit
    ``hidden``/``n_layers`` per the width sweep — 64/96/128 × 3 layers);
    the fusion encoder sizes stay at their WP3-A defaults.
    """

    d_model: int = 32
    d_element: int = 16
    max_z: int = MAX_Z
    hidden: int = 64
    n_layers: int = 3
    n_basis: int = 32
    cutoff: float = 5.0
    vector_channels: int = 16
    contact_edges: bool = False
    contact_cutoff: float = 5.0
    max_contact_neighbors: int = 16
    # V4.2 E1 (features_v4): per-endpoint edge feature width (4=v3, 7=v4).
    edge_feat_dim: int = 4

    def __post_init__(self) -> None:
        if self.cutoff <= 0.0 or self.n_basis <= 0:
            raise ValueError("cutoff/n_basis must be positive")
        if self.n_layers <= 0 or self.hidden <= 0:
            raise ValueError("hidden/n_layers must be positive")
        if self.vector_channels <= 0:
            raise ValueError("vector_channels must be positive")
        if self.contact_cutoff <= 0.0 or self.max_contact_neighbors <= 0:
            raise ValueError("contact cutoff / max neighbours must be positive")


def _build_contact_edges(torch, *, x, chem_edge_index, node_counts, n_graphs,
                         cutoff, max_neighbors):
    """Deterministic per-state radius-``cutoff`` contact edges (V2).

    Only ADDS edges: a contact pair ``(a, j)`` is emitted iff its undirected
    pair is not already a chemical union edge (chemical edges are never
    deleted).  Each atom keeps at most ``max_neighbors`` contact neighbours;
    candidates are sorted by ``(dist, atom_idx)`` ascending and selected in
    ascending atom order (a candidate is skipped when either endpoint is
    already saturated).  Returns ``(src, dst, envelope)`` — directed both
    ways with a cosine smooth envelope ``0.5(1 + cos(π d / cutoff))``.
    """
    device = x.device
    dtype = x.dtype
    chem_set: set[tuple[int, int]] = set()
    for e in range(int(chem_edge_index.shape[1])):
        s = int(chem_edge_index[0, e])
        d = int(chem_edge_index[1, e])
        chem_set.add((min(s, d), max(s, d)))

    contact_pairs: list[tuple[int, int, float]] = []  # (a, j, dist)
    pair_set: set[tuple[int, int]] = set()
    degrees: dict[int, int] = {}
    row_idx = torch.arange(int(x.shape[0]), device=device)
    for gi in range(int(n_graphs)):
        start = int(node_counts[:gi].sum())
        end = int(start + int(node_counts[gi]))
        xi = x[start:end]
        ni = int(end - start)
        if ni < 2:
            continue
        dmat = torch.cdist(xi, xi)                       # (ni, ni)
        for a in range(ni):
            da = dmat[a]
            mask = (da <= cutoff) & (row_idx[:ni] != a)
            idx = torch.nonzero(mask).squeeze(-1)
            if idx.numel() == 0:
                continue
            dists = da[idx]
            # lexicographic (dist, idx) ascending via two stable sorts.
            order = torch.argsort(idx.float(), stable=True)     # secondary: idx
            order = order[torch.argsort(dists[order], stable=True)]  # primary: dist
            for k in range(int(order.numel())):
                j_local = int(idx[int(order[k])])
                ga = int(a + start)
                gj = int(j_local + start)
                pair = (min(ga, gj), max(ga, gj))
                if pair in chem_set or pair in pair_set:
                    continue
                if degrees.get(ga, 0) >= max_neighbors or \
                        degrees.get(gj, 0) >= max_neighbors:
                    continue
                d = float(dists[int(order[k])].item())
                contact_pairs.append((ga, gj, d))
                pair_set.add(pair)
                degrees[ga] = degrees.get(ga, 0) + 1
                degrees[gj] = degrees.get(gj, 0) + 1

    if not contact_pairs:
        return {
            "src": torch.zeros(0, dtype=torch.long, device=device),
            "dst": torch.zeros(0, dtype=torch.long, device=device),
            "envelope": torch.zeros(0, 1, dtype=dtype, device=device),
        }
    srcs: list[int] = []
    dsts: list[int] = []
    envs: list[float] = []
    for (a, j, d) in contact_pairs:
        env = 0.5 * (1.0 + math.cos(math.pi * d / float(cutoff)))
        srcs += [a, j]
        dsts += [j, a]
        envs += [env, env]
    return {
        "src": torch.as_tensor(srcs, dtype=torch.long, device=device),
        "dst": torch.as_tensor(dsts, dtype=torch.long, device=device),
        "envelope": torch.as_tensor(envs, dtype=dtype, device=device).unsqueeze(-1),
    }


class _TorchDualPaiNNFactory:
    """Builds the torch.nn.Module once torch is confirmed available."""

    def build(self, torch, config: DualTSPaiNNEncoderConfig):
        nn = torch.nn
        hidden = config.hidden
        C = config.vector_channels

        class _DualPaiNN(nn.Module):
            def __init__(inner) -> None:
                super().__init__()
                inner.config = config
                inner.fusion = DualTSFusionEncoder(DualTSEncoderConfig(
                    d_model=config.d_model, d_element=config.d_element,
                    max_z=config.max_z,
                    edge_feat_dim=getattr(config, "edge_feat_dim", 4)))
                inner.node_proj = nn.Linear(
                    2 * config.d_model + N_NODE_SCALARS, hidden)
                inner.edge_proj = nn.Linear(2 * config.d_model, hidden)
                inner.basis_proj = nn.Linear(config.n_basis, hidden)
                # vector init: C channels seeded from the gradient g (a
                # translation-invariant polar vector).  X is NOT used in the
                # absolute seed (it is translation-covariant); X enters only
                # through translation-invariant rel distances / rhat_ij.
                # No bias: a bias would break translation equivariance.
                inner.vec_init = nn.Linear(1, C, bias=False)
                # contact-edge marker embedding (V2 only — created only when
                # the contact branch exists so V1 carries no dangling param).
                inner.contact_emb = (
                    nn.Parameter(torch.zeros(hidden))
                    if config.contact_edges else None)
                inner.s_mlp = nn.ModuleList([
                    nn.Sequential(nn.Linear(3 * hidden, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden))
                    for _ in range(config.n_layers)])
                inner.v_radial = nn.ModuleList([
                    nn.Linear(config.n_basis, C)
                    for _ in range(config.n_layers)])
                inner.v_r = nn.ModuleList([
                    nn.Linear(hidden, C) for _ in range(config.n_layers)])
                inner.v2s = nn.ModuleList([
                    nn.Linear(2 * C, hidden) for _ in range(config.n_layers)])
                inner.node_mlp = nn.ModuleList([
                    nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden))
                    for _ in range(config.n_layers)])
                inner.v_gate = nn.ModuleList([
                    nn.Linear(hidden, C) for _ in range(config.n_layers)])
                inner.norm_h = nn.ModuleList([
                    nn.LayerNorm(hidden) for _ in range(config.n_layers)])
                inner.vec_readout = nn.Linear(hidden, C)

            def _apply(inner, fn, recurse=True):
                # The WP3-A fusion encoder pins float32; a module-wide DTYPE
                # cast must skip it (same discipline as dual_egnn), while
                # device moves must not.
                probe = torch.zeros(1)
                try:
                    after = fn(probe)
                except (TypeError, ValueError):
                    after = probe
                if after.dtype != probe.dtype:
                    fusion = inner._modules.pop("fusion")
                    try:
                        return super()._apply(fn, recurse)
                    finally:
                        inner._modules["fusion"] = fusion
                return super()._apply(fn, recurse)

            def _graph_meta(inner, inputs):
                if "node_counts" in inputs and inputs["node_counts"] is not None:
                    counts = inputs["node_counts"].to(inputs["z"].device).long()
                    n_graphs = int(counts.shape[0])
                else:
                    n_graphs = 1
                    counts = torch.as_tensor(
                        [int(inputs["z"].shape[0])], device=inputs["z"].device)
                return counts, n_graphs

            def forward(inner, inputs: dict) -> dict:
                dtype = inner.node_proj.weight.dtype
                z = inputs["z"]
                fused = inner.fusion(
                    z=z,
                    atom_feats_a=inputs["atom_feats_a"],
                    atom_feats_b=inputs["atom_feats_b"],
                    edge_feats_a=inputs["edge_feats_a"],
                    edge_feats_b=inputs["edge_feats_b"],
                )
                h = inner.node_proj(torch.cat(
                    [fused["fused_atom"].to(dtype),
                     inputs["node_scalars"].to(dtype)], dim=-1))
                x = inputs["x"].to(dtype)
                g = inputs["g"].to(dtype)

                # vector seed: C channels from g (polar, translation-invariant).
                U = inner.vec_init(g.unsqueeze(1).transpose(1, 2)).transpose(1, 2)  # (N, C, 3)

                edge_index = inputs["edge_index"]
                src = edge_index[0].long()
                dst = edge_index[1].long()
                rel = x[src] - x[dst]
                dist = rel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                unit = rel / dist
                rbf_chem = gaussian_distance_basis(torch, dist, config)
                e_static_chem = inner.edge_proj(
                    fused["fused_edge"].to(dtype)) + inner.basis_proj(rbf_chem)
                n_chem = int(src.shape[0])

                if config.contact_edges:
                    counts, n_graphs = inner._graph_meta(inputs)
                    contact = _build_contact_edges(
                        torch, x=x, chem_edge_index=edge_index,
                        node_counts=counts, n_graphs=n_graphs,
                        cutoff=config.contact_cutoff,
                        max_neighbors=config.max_contact_neighbors)
                    src = torch.cat([src, contact["src"]], dim=0)
                    dst = torch.cat([dst, contact["dst"]], dim=0)
                    rel = x[src] - x[dst]
                    dist = rel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                    unit = rel / dist
                    rbf = gaussian_distance_basis(torch, dist, config)
                    e_static_contact = inner.basis_proj(
                        rbf[n_chem:]) + inner.contact_emb
                    e_static = torch.cat([e_static_chem, e_static_contact], dim=0)
                    envelope = torch.cat([
                        torch.ones(n_chem, 1, dtype=dtype, device=x.device),
                        contact["envelope"],
                    ], dim=0)
                else:
                    rbf = rbf_chem
                    e_static = e_static_chem
                    envelope = torch.ones(
                        int(dist.shape[0]), 1, dtype=dtype, device=x.device)

                for layer in range(config.n_layers):
                    # scalar message (smooth envelope only on contact branch)
                    msg_in = torch.cat([h[src], h[dst], e_static], dim=-1)
                    s_msg = inner.s_mlp[layer](msg_in) * envelope
                    s_agg = torch.zeros_like(h).index_add_(0, dst, s_msg)

                    # vector message: neighbour transfer + radial message
                    v_neigh_gate = inner.v_radial[layer](rbf) * envelope   # (E, C)
                    U_msg = torch.zeros_like(U).index_add_(
                        0, dst, U[src] * v_neigh_gate.unsqueeze(-1))
                    v_r_gate = inner.v_r[layer](h[src]) * envelope         # (E, C)
                    U_msg = U_msg.index_add_(
                        0, dst, unit.unsqueeze(1) * v_r_gate.unsqueeze(-1))

                    # vector -> scalar coupling (PaiNN mixing)
                    inner_prod = (U[src] * U[dst]).sum(-1)                  # (E, C)
                    rhat_prod = (U[dst] * unit.unsqueeze(1)).sum(-1)        # (E, C)
                    vec_scalar = inner.v2s[layer](
                        torch.cat([inner_prod, rhat_prod], dim=-1)) * envelope
                    s_agg = s_agg.index_add_(0, dst, vec_scalar)

                    # scalar update (LayerNorm on the invariant axis only)
                    h = h + inner.node_mlp[layer](torch.cat([h, s_agg], dim=-1))
                    h = inner.norm_h[layer](h)

                    # scalar-gated vector update
                    U = U + U_msg
                    v_gate = inner.v_gate[layer](h)                          # (N, C)
                    U = U * v_gate.unsqueeze(-1)

                v_gate_out = inner.vec_readout(h)                           # (N, C)
                v_3d = (v_gate_out.unsqueeze(-1) * U).sum(dim=1)            # (N, 3)
                return {"h": h, "v": v_3d, "v_channels": U}

        return _DualPaiNN()


class DualTSPaiNNEncoder:
    """Wrapper mirroring ``DualTSEGNN``'s module-holder interface."""

    def __init__(self, config: DualTSPaiNNEncoderConfig | None = None) -> None:
        from pes2ts_core.pesn.models import require_torch
        self.torch = require_torch()
        self.config = config or DualTSPaiNNEncoderConfig()
        self.module = _TorchDualPaiNNFactory().build(self.torch, self.config)

    def __call__(self, inputs: dict) -> dict:
        return self.module(inputs)

    def parameters(self):
        return self.module.parameters()

    def train_mode(self, flag: bool) -> None:
        self.module.train(flag)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state) -> None:
        self.module.load_state_dict(state)

    def to(self, device) -> "DualTSPaiNNEncoder":
        self.module.to(device)
        return self

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.module.parameters())


def build_painn_encoder(config: DualTSPaiNNEncoderConfig | None = None) -> DualTSPaiNNEncoder:
    """Factory for the PaiNN-style dual-TS encoder (V1/V2)."""
    return DualTSPaiNNEncoder(config)


__all__ = [
    "DUAL_EGNN_PAINN_VERSION",
    "DualTSPaiNNEncoder",
    "DualTSPaiNNEncoderConfig",
    "build_painn_encoder",
]
