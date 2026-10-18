"""Equivariant EGNN over the dual-topology union graph (WP3-B, plan v4 §D WP3).

Consumes the WP3-A exchange-symmetric dual-graph encoder outputs (node
blocks, union edge list, per-endpoint edge embeddings — already fused by
``DualTSFusionEncoder`` exactly as frozen in the ``dual_ts_spec`` fusion
rule) plus the equivariant vector channels ``X`` (centroid removed at
consume time) and ``g̃ = g / s_g``, and runs equivariant message passing
over the dual-topology union graph.  Both endpoint graphs' edges enter as
message channels through the fused per-edge embeddings; bond orders are
never merged or sorted across endpoints (that happened nowhere — the
fused edge block is exchange-symmetric by construction).

Equivariance discipline (v1 ``models/egnn.py`` patterns reused):

* node scalars ``h`` are invariant: updates consume only distances
  (Gaussian basis, v1-reused parameters: cutoff 5.0 Å, 32 centers on
  ``linspace(0, cutoff, n_basis)``, scale ``cutoff / n_basis``) and
  invariant scalars (the frozen 5-channel node block holds no direction);
* the equivariant vector field ``v`` accumulates unit relative directions
  ``(x_src − x_dst)/‖·‖`` scaled by per-edge scalar gates
  (default: ``sigmoid`` — positive gating, the v1 structure that the WP3-B
  vector-head diagnostics target; a signed-weight variant is available
  behind ``signed_vector_messages=True`` and is a FLAGGED alternative,
  never the default);
* the vector seed is ``v0 = g̃ * sigmoid(w(h))`` — again positive gating.

Torch is resolved lazily through ``require_torch()``; the module is built
inside a factory (same discipline as v1).  No v1 file is modified.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from pes2ts_core.pesn.features.dual_ts_graph import (
    N_NODE_SCALARS,
    DualTSEncoderConfig,
    DualTSFusionEncoder,
    DualTSGraphError,
)
from pes2ts_core.pesn.models import require_torch

DUAL_TS_EGNN_VERSION: Final[str] = "pesn_dual_ts_egnn_v1.0"
MAX_Z: Final[int] = 32


@dataclass(frozen=True, slots=True)
class DualTSModelConfig:
    """Sizes for the full dual-TS model (fusion encoder + EGNN + head).

    ``light=True`` caps the EGNN at hidden 64 / 3 layers (small-data
    regime); the fusion encoder sizes stay at their WP3-A defaults.
    ``signed_vector_messages`` is the FLAGGED expressiveness alternative
    for the diagnostics lane — off by default, never silently flipped.
    """

    d_model: int = 32
    d_element: int = 16
    max_z: int = MAX_Z
    hidden: int = 128
    n_layers: int = 4
    n_basis: int = 32
    cutoff: float = 5.0
    light: bool = False
    signed_vector_messages: bool = False
    # V4.2 E1 (features_v4): per-endpoint edge feature width (4=v3, 7=v4).
    edge_feat_dim: int = 4

    def __post_init__(self) -> None:
        if self.light:
            object.__setattr__(self, "hidden", min(self.hidden, 64))
            object.__setattr__(self, "n_layers", min(self.n_layers, 3))
        if self.cutoff <= 0.0 or self.n_basis <= 0:
            raise ValueError("cutoff/n_basis must be positive")
        if self.n_layers <= 0 or self.hidden <= 0:
            raise ValueError("hidden/n_layers must be positive")


def gaussian_distance_basis(torch, dist: "torch.Tensor", config: DualTSModelConfig):
    """v1-reused Gaussian expansion of distances (frozen parameters).

    ``centers = linspace(0, cutoff, n_basis)``, ``scale = cutoff/n_basis``;
    ``basis_k(d) = exp(-0.5 ((d − center_k)/scale)^2)``.
    """
    centers = torch.linspace(0.0, float(config.cutoff), int(config.n_basis),
                             dtype=dist.dtype, device=dist.device)
    scale = float(config.cutoff) / int(config.n_basis)
    return torch.exp(-0.5 * ((dist - centers) / scale) ** 2)


class _TorchDualEGNNFactory:
    """Builds the torch.nn.Module once torch is confirmed available."""

    def build(self, torch, config: DualTSModelConfig):
        nn = torch.nn
        hidden = config.hidden

        class _DualEGNN(nn.Module):
            def __init__(inner) -> None:
                super().__init__()
                inner.config = config
                # WP3-A shared endpoint encoder + exchange-symmetric fusion
                # (imported unmodified; trained here as part of this model).
                inner.fusion = DualTSFusionEncoder(DualTSEncoderConfig(
                    d_model=config.d_model, d_element=config.d_element,
                    max_z=config.max_z,
                    edge_feat_dim=getattr(config, "edge_feat_dim", 4)))
                # input layer sync: fused dual-topology atom block + the
                # frozen 5-channel node scalar block -> hidden
                inner.node_proj = nn.Linear(
                    2 * config.d_model + N_NODE_SCALARS, hidden)
                inner.edge_proj = nn.Linear(2 * config.d_model, hidden)
                inner.basis_proj = nn.Linear(config.n_basis, hidden)
                # equivariant vector seed: positive gate on g̃ (v1 pattern)
                inner.vector_seed_gate = nn.Linear(hidden, 1)
                inner.edge_mlp = nn.ModuleList([
                    nn.Sequential(nn.Linear(3 * hidden, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden))
                    for _ in range(config.n_layers)])
                inner.node_mlp = nn.ModuleList([
                    nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden))
                    for _ in range(config.n_layers)])
                # per-edge vector gates: default positive (sigmoid); the
                # signed variant uses the raw linear head (flagged only)
                inner.vector_gate = nn.ModuleList([
                    nn.Linear(hidden, 1) for _ in range(config.n_layers)])
                inner.norm_h = nn.ModuleList([
                    nn.LayerNorm(hidden) for _ in range(config.n_layers)])

            def _apply(inner, fn, recurse=True):
                # The WP3-A fusion encoder pins float32 (.float() casts on
                # its inputs); a module-wide DTYPE cast must skip it (its
                # outputs are cast to the compute dtype in forward), while
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

            def forward(inner, inputs: dict) -> dict:
                # cast inputs to the module dtype (float64 tests cast the
                # module via .to(torch.float64); float32 GPU runs cast here)
                dtype = inner.node_proj.weight.dtype
                z = inputs["z"]
                fused = inner.fusion(
                    z=z,
                    atom_feats_a=inputs["atom_feats_a"],
                    atom_feats_b=inputs["atom_feats_b"],
                    edge_feats_a=inputs["edge_feats_a"],
                    edge_feats_b=inputs["edge_feats_b"],
                )
                # invariant node init: fused dual-topology block + frozen
                # 5-channel node scalars (all per-atom invariants).  The
                # fusion inputs are pure topology features (rotation- and
                # translation-invariant; permuted row-wise exactly), so the
                # float32 fusion cannot break the vector symmetry gates.
                h = inner.node_proj(torch.cat(
                    [fused["fused_atom"].to(dtype),
                     inputs["node_scalars"].to(dtype)], dim=-1))
                x = inputs["x"].to(dtype)
                g = inputs["g"].to(dtype)
                src, dst = inputs["edge_index"][0], inputs["edge_index"][1]
                rel = x[src] - x[dst]
                dist = rel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                unit = rel / dist
                e_static = inner.edge_proj(
                    fused["fused_edge"].to(dtype)) + inner.basis_proj(
                    gaussian_distance_basis(torch, dist, config))
                # vector seed: positive-gated gradient channel (v1 pattern)
                v = g * torch.sigmoid(inner.vector_seed_gate(h))
                signed = config.signed_vector_messages
                for layer in range(config.n_layers):
                    msg_in = torch.cat([h[src], h[dst], e_static], dim=-1)
                    messages = inner.edge_mlp[layer](msg_in)
                    agg = torch.zeros_like(h).index_add_(0, dst, messages)
                    h = inner.norm_h[layer](
                        h + inner.node_mlp[layer](torch.cat([h, agg], dim=-1)))
                    raw_gate = inner.vector_gate[layer](messages)
                    gate = raw_gate if signed else torch.sigmoid(raw_gate)
                    v = v.index_add(0, dst, unit * gate)
                return {"h": h, "v": v}

        return _DualEGNN()


class DualTSEGNN:
    """Wrapper mirroring the v1 encoder's module-holder interface."""

    def __init__(self, config: DualTSModelConfig | None = None) -> None:
        self.torch = require_torch()
        self.config = config or DualTSModelConfig()
        self.module = _TorchDualEGNNFactory().build(self.torch, self.config)

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

    def to(self, device) -> "DualTSEGNN":
        self.module.to(device)
        return self

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.module.parameters())


# ---------------------------------------------------------------------------
# Shard adapter: numpy state arrays -> torch input dict
# ---------------------------------------------------------------------------

REQUIRED_STATE_KEYS: Final[tuple[str, ...]] = (
    "atomic_numbers", "node_scalars", "x_angstrom", "g_scaled",
    "atom_feats_a", "atom_feats_b", "edge_feats_a", "edge_feats_b",
    "edge_index",
)


def state_inputs_to_tensors(torch, state: dict) -> dict:
    """One state's numpy arrays (shard rows) -> torch float input dict.

    Consumes the shard arrays AS-IS; the frozen consume-time translation
    rule (unweighted centroid removal) is re-applied here — it is idempotent
    on shard X (already centroid-removed at build time) and is what makes
    raw synthetic geometries translation-safe.
    """
    x = np.asarray(state["x_angstrom"], dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    node_scalars = np.asarray(state["node_scalars"], dtype=np.float64)
    # v4.1 WP1 load-time width guard: a 7-channel pre-v3 shard fed to the new
    # 5-channel model must fail loudly (old checkpoints are intentionally
    # incompatible — digest + shape gated), never silently mis-slice.
    if node_scalars.shape[-1] != N_NODE_SCALARS:
        raise DualTSGraphError(
            "NODE_SCALAR_WIDTH_MISMATCH",
            f"node_scalars last dim {node_scalars.shape[-1]} != "
            f"{N_NODE_SCALARS} (v3 observation-only layout); rebuild the "
            "shard with the v4.1 builder")
    return {
        "z": torch.as_tensor(np.ascontiguousarray(
            state["atomic_numbers"])).long(),
        "node_scalars": torch.as_tensor(np.ascontiguousarray(
            node_scalars)).to(torch.float64),
        "x": torch.as_tensor(np.ascontiguousarray(x)).to(torch.float64),
        "g": torch.as_tensor(np.ascontiguousarray(
            state["g_scaled"], dtype=np.float64)).to(torch.float64),
        "atom_feats_a": torch.as_tensor(np.ascontiguousarray(
            state["atom_feats_a"], dtype=np.float64)).to(torch.float64),
        "atom_feats_b": torch.as_tensor(np.ascontiguousarray(
            state["atom_feats_b"], dtype=np.float64)).to(torch.float64),
        "edge_feats_a": torch.as_tensor(np.ascontiguousarray(
            state["edge_feats_a"], dtype=np.float64)).to(torch.float64),
        "edge_feats_b": torch.as_tensor(np.ascontiguousarray(
            state["edge_feats_b"], dtype=np.float64)).to(torch.float64),
        "edge_index": torch.as_tensor(np.ascontiguousarray(
            state["edge_index"], dtype=np.int64)).long(),
    }


def collate_dual_states(torch, states: list[dict]) -> dict:
    """Concatenated-graph batch of dual states (offset edges, no padding).

    Each state contributes its own union graph; edge indices are offset by
    the running atom count.  All tensors share one dtype (float64 here —
    callers may cast/downcast afterwards for GPU runs).
    """
    for s in states:
        missing = [k for k in REQUIRED_STATE_KEYS if k not in s]
        if missing:
            raise KeyError(f"state input missing keys: {missing}")
    inputs = [state_inputs_to_tensors(torch, s) for s in states]
    n_atoms = [t["z"].shape[0] for t in inputs]
    offsets = np.cumsum([0] + n_atoms).tolist()
    edge_blocks = []
    for t, off in zip(inputs, offsets[:-1]):
        edge_blocks.append(t["edge_index"] + int(off))
    cat = torch.cat
    return {
        "z": cat([t["z"] for t in inputs], dim=0),
        "node_scalars": cat([t["node_scalars"] for t in inputs], dim=0),
        "x": cat([t["x"] for t in inputs], dim=0),
        "g": cat([t["g"] for t in inputs], dim=0),
        "atom_feats_a": cat([t["atom_feats_a"] for t in inputs], dim=0),
        "atom_feats_b": cat([t["atom_feats_b"] for t in inputs], dim=0),
        "edge_feats_a": cat([t["edge_feats_a"] for t in inputs], dim=0),
        "edge_feats_b": cat([t["edge_feats_b"] for t in inputs], dim=0),
        "edge_index": cat(edge_blocks, dim=1),
        "node_counts": torch.as_tensor(n_atoms),
        "n_graphs": len(states),
    }


__all__ = [
    "DUAL_TS_EGNN_VERSION",
    "DualTSModelConfig",
    "DualTSEGNN",
    "REQUIRED_STATE_KEYS",
    "collate_dual_states",
    "gaussian_distance_basis",
    "state_inputs_to_tensors",
]
