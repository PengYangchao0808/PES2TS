"""Dual-TS Q/V critics with PHYSICAL action input (WP5-prep, plan v4 §I.11).

Fixes of the v1 defects (plan §I.11 items 1-2; contract v2 doc §4):

* v1 fed Q the DIMENSIONLESS latent ``u`` and treated it as a physical
  displacement.  Here Q receives the PHYSICAL action ``a`` in Å (the
  ``action_angstrom`` carrier of ``pesn_dual_ts_action_v2``) and every call
  is gated by the explicit version flag ``Q_ACTION_INPUT_VERSION``.  An
  input claiming the dimensionless-z unit (or any unknown version string)
  is REFUSED with a typed error — never silently rescaled.
* v1 mixed units (``a + g`` with ``g`` in Hartree/Å; ``|a + g|``).  Every
  feature built here is DIMENSIONLESS: ``ã = a / s_a`` with the frozen
  scale rule ``s_a = a_max`` (``action_bound_a_max_v0`` — reuses the frozen
  action bound already registered in the features spec support domain,
  ``a_max_angstrom=0.1``; no new fit-only estimation is introduced by this
  preparation lane), ``g̃ = g / s_g`` arrives pre-scaled in the shard, and
  the pair distance change ``Δd_ij`` is divided by ``s_a`` as well.
* Feature set per plan §I.11 (joint with atom / task-edge encoding, never
  mainly the mean action length): per atom ``[||ã_i||, ã_i · g̃_i]``; per
  DIRECTED union (task) edge ``i→j`` ``[ã_i·r̂_ij, ã_j·r̂_ij, Δd̃_ij]``.
  Atom action features enter the node input layer; edge action features
  enter the static edge block — both are then refined by the same
  equivariant message-passing structure as ``dual_egnn`` (invariant node
  scalars, positive vector gates, the v1 Gaussian distance basis reused
  verbatim).  ``a`` is NEVER added to a gradient vector or any
  other unit-carrying tensor.
* Q1 / Q2 / V are fully independent networks (each owns its fusion encoder
  + EGNN + readout).  Each records the CONTENT DIGEST of its INIT state
  for provenance (plan §I.11: independent init recorded); target copies are
  registered by the trainer's polyak registry.

Torch is resolved lazily through ``require_torch()``; modules are built
inside a factory (v1 discipline).  No v1 file is modified.
"""
from __future__ import annotations

import hashlib
from typing import Final

import numpy as np

from pes2ts_core.pesn.features.dual_ts_graph import (
    N_NODE_SCALARS,
    DualTSEncoderConfig,
    DualTSFusionEncoder,
)
from pes2ts_core.pesn.models import require_torch
from pes2ts_core.pesn.models.dual_egnn import (
    DualTSModelConfig,
    gaussian_distance_basis,
)

CRITICS_VERSION: Final[str] = "pesn_dual_ts_critics_v2.0"

#: Version flag a caller MUST pass with every physical action given to a
#: Q critic.  It names the action schema, the registered radial transform,
#: and the UNIT of the tensor being passed.  Anything else is refused.
Q_ACTION_INPUT_VERSION: Final[str] = (
    "pesn_dual_ts_action_v2:radial_tanh_v0:a_unit=angstrom")

#: Frozen action-scale rule: s_a = a_max (Å).  Reuses the frozen action
#: bound of the features spec support domain; no new fitted scale.
S_A_RULE: Final[str] = "action_bound_a_max_v0"

COORDINATES_UNIT: Final[str] = "angstrom"
A_BOUND_TOL: Final[float] = 1e-6


class DualTSCriticError(RuntimeError):
    """Typed dual-TS critic misuse (unit/version violations, bad shapes)."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code


# ---------------------------------------------------------------------------
# Content digests over torch state (deterministic, byte-level)
# ---------------------------------------------------------------------------


def model_state_digest(state) -> str:
    """Deterministic sha256 over a (nested) torch state dict.

    Walks the tree in sorted key order; each tensor contributes its path,
    dtype, shape, and raw little-endian-free platform bytes.  Used for the
    recorded init provenance of every independent critic and for the
    checkpoint content digests (plan §I.11: real content digests only).
    """
    h = hashlib.sha256()

    def walk(prefix: str, obj) -> None:
        if isinstance(obj, dict):
            for key in sorted(obj.keys(), key=str):
                walk(f"{prefix}/{key}", obj[key])
        elif hasattr(obj, "detach"):
            t = obj.detach().to("cpu", copy=True).contiguous()
            h.update(prefix.encode("utf-8"))
            h.update(str(t.dtype).encode("utf-8"))
            h.update(str(tuple(t.shape)).encode("utf-8"))
            h.update(t.numpy().tobytes())
        else:
            h.update(prefix.encode("utf-8"))
            h.update(repr(obj).encode("utf-8"))

    walk("", dict(state))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Geometric action features (all DIMENSIONLESS)
# ---------------------------------------------------------------------------


def validate_physical_action(torch, batch, a, *, a_max: float,
                             action_version: str) -> None:
    """Typed refusal for anything that is not physical-Å within a_max.

    * the version flag must equal ``Q_ACTION_INPUT_VERSION`` exactly
      (a carrier passed with ``a_unit=dimensionless_z`` or an unknown /
      ambiguous version is refused — test (j) discipline);
    * the tensor must be finite, (N_total, 3), matching the batch atoms;
    * every per-atom norm must respect ``||a_i|| <= a_max (1 + tol)`` —
      this also catches a z carrier (norms up to 1.0 ≫ a_max=0.1) even if
      a caller lies about the version flag.
    """
    if action_version != Q_ACTION_INPUT_VERSION:
        raise DualTSCriticError(
            "ACTION_VERSION_MISMATCH",
            f"Q requires the physical-angstrom action input version "
            f"{Q_ACTION_INPUT_VERSION!r}; got {action_version!r}")
    n_atoms = int(batch["z"].shape[0])
    if a.dim() != 2 or a.shape[1] != 3 or int(a.shape[0]) != n_atoms:
        raise DualTSCriticError(
            "ACTION_SHAPE_INVALID",
            f"physical action must be (N,3) with N={n_atoms}, got "
            f"{tuple(a.shape)}")
    if not bool(torch.isfinite(a).all()):
        raise DualTSCriticError("NONFINITE_ACTION",
                                "physical action contains NaN/Inf")
    norms = a.float().norm(dim=-1)
    worst = float(norms.max()) if norms.numel() else 0.0
    if worst > float(a_max) * (1.0 + A_BOUND_TOL):
        raise DualTSCriticError(
            "ACTION_BOUND_EXCEEDED",
            f"physical action norm {worst:.6f} exceeds a_max={a_max} "
            f"(a dimensionless z carrier passed as angstrom is refused)")


def geometric_action_features(torch, batch, a, *, s_a: float):
    """Dimensionless per-atom (N,2) and per-directed-edge (E,3) features.

    Per atom: ``[||a_i||/s_a, (a_i/s_a)·g̃_i]``.
    Per directed union (task) edge i→j: ``[(a_i/s_a)·r̂_ij, (a_j/s_a)·r̂_ij,
    (||x_j+a_j−x_i−a_i|| − ||x_j−x_i||)/s_a]``.
    ``g̃`` is the pre-scaled shard gradient channel (``g_scaled``); ``x`` is
    Å (centroid removal is distance-invariant).  No unit mixing anywhere.
    """
    if not (np.isfinite(s_a) and s_a > 0.0):
        raise DualTSCriticError("SCALE_INVALID", "s_a must be finite > 0")
    dtype = batch["x"].dtype
    a = a.to(dtype)
    a_hat = a / float(s_a)
    a_norm = a.norm(dim=-1, keepdim=True) / float(s_a)
    g_dot = (a_hat * batch["g"].to(dtype)).sum(-1, keepdim=True)
    atom_feats = torch.cat([a_norm, g_dot], dim=-1)

    x = batch["x"].to(dtype)
    src, dst = batch["edge_index"][0], batch["edge_index"][1]
    rel = x[dst] - x[src]
    dist = rel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    unit = rel / dist
    moved = x + a
    rel_m = moved[dst] - moved[src]
    dist_m = rel_m.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    edge_feats = torch.cat([
        (a_hat[src] * unit).sum(-1, keepdim=True),
        (a_hat[dst] * unit).sum(-1, keepdim=True),
        (dist_m - dist) / float(s_a),
    ], dim=-1)
    return atom_feats, edge_feats


# ---------------------------------------------------------------------------
# Torch module factory (mirrors dual_egnn structure + action channels)
# ---------------------------------------------------------------------------


class _TorchDualCriticFactory:
    """Builds the torch.nn.Module once torch is confirmed available.

    ``use_action=False`` yields the V body (no action channels anywhere);
    ``use_action=True`` yields the Q body.  Both share the structural
    discipline of ``dual_egnn``: invariant scalar updates, equivariant
    vector field with POSITIVE gates, frozen Gaussian distance basis.
    """

    def build(self, torch, config: DualTSModelConfig, options: dict):
        nn = torch.nn
        hidden = config.hidden
        use_action = bool(options["use_action"])
        atom_act_dim = 2 if use_action else 0
        edge_act_dim = 3 if use_action else 0

        class _DualCritic(nn.Module):
            def __init__(inner) -> None:
                super().__init__()
                inner.config = config
                inner.use_action = use_action
                inner.fusion = DualTSFusionEncoder(DualTSEncoderConfig(
                    d_model=config.d_model, d_element=config.d_element,
                    max_z=config.max_z))
                inner.node_proj = nn.Linear(
                    2 * config.d_model + N_NODE_SCALARS + atom_act_dim,
                    hidden)
                inner.edge_proj = nn.Linear(2 * config.d_model, hidden)
                inner.basis_proj = nn.Linear(config.n_basis, hidden)
                if use_action:
                    inner.edge_act_proj = nn.Linear(edge_act_dim, hidden)
                inner.vector_seed_gate = nn.Linear(hidden, 1)
                inner.edge_mlp = nn.ModuleList([
                    nn.Sequential(nn.Linear(3 * hidden, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden))
                    for _ in range(config.n_layers)])
                inner.node_mlp = nn.ModuleList([
                    nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden))
                    for _ in range(config.n_layers)])
                inner.vector_gate = nn.ModuleList([
                    nn.Linear(hidden, 1) for _ in range(config.n_layers)])
                inner.norm_h = nn.ModuleList([
                    nn.LayerNorm(hidden) for _ in range(config.n_layers)])
                inner.readout = nn.Sequential(
                    nn.Linear(hidden, hidden), nn.SiLU(),
                    nn.Linear(hidden, 1))

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

            def _graph_id(inner, batch):
                n_graphs = int(batch["n_graphs"])
                counts = batch["node_counts"].to(batch["z"].device).long()
                return torch.repeat_interleave(
                    torch.arange(n_graphs, device=batch["z"].device), counts)

            def _encode_pass(inner, batch, atom_act_feats=None,
                             edge_act_feats=None):
                dtype = inner.node_proj.weight.dtype
                fused = inner.fusion(
                    z=batch["z"],
                    atom_feats_a=batch["atom_feats_a"],
                    atom_feats_b=batch["atom_feats_b"],
                    edge_feats_a=batch["edge_feats_a"],
                    edge_feats_b=batch["edge_feats_b"],
                )
                node_blocks = [fused["fused_atom"].to(dtype),
                               batch["node_scalars"].to(dtype)]
                if inner.use_action:
                    node_blocks.append(atom_act_feats.to(dtype))
                h = inner.node_proj(torch.cat(node_blocks, dim=-1))
                x = batch["x"].to(dtype)
                g = batch["g"].to(dtype)
                src, dst = batch["edge_index"][0], batch["edge_index"][1]
                rel = x[src] - x[dst]
                dist = rel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                unit = rel / dist
                e_static = inner.edge_proj(
                    fused["fused_edge"].to(dtype)) + inner.basis_proj(
                    gaussian_distance_basis(torch, dist, inner.config))
                if inner.use_action:
                    e_static = e_static + inner.edge_act_proj(
                        edge_act_feats.to(dtype))
                v = g * torch.sigmoid(inner.vector_seed_gate(h))
                signed = inner.config.signed_vector_messages
                for layer in range(inner.config.n_layers):
                    msg_in = torch.cat([h[src], h[dst], e_static], dim=-1)
                    messages = inner.edge_mlp[layer](msg_in)
                    agg = torch.zeros_like(h).index_add_(0, dst, messages)
                    h = inner.norm_h[layer](
                        h + inner.node_mlp[layer](torch.cat([h, agg], dim=-1)))
                    raw_gate = inner.vector_gate[layer](messages)
                    gate = raw_gate if signed else torch.sigmoid(raw_gate)
                    v = v.index_add(0, dst, unit * gate)
                return h

            def forward(inner, batch, a=None, action_version=None,
                        s_a=None, a_max=None):
                if inner.use_action:
                    if a is None:
                        raise DualTSCriticError(
                            "ACTION_MISSING",
                            "Q requires the physical action tensor")
                    validate_physical_action(
                        torch, batch, a, a_max=a_max,
                        action_version=action_version)
                    atom_act, edge_act = geometric_action_features(
                        torch, batch, a, s_a=s_a)
                    h = inner._encode_pass(batch, atom_act, edge_act)
                else:
                    h = inner._encode_pass(batch)
                gid = inner._graph_id(batch)
                pooled = torch.zeros(int(batch["n_graphs"]), h.shape[1],
                                     device=h.device, dtype=h.dtype)
                pooled = pooled.index_add_(0, gid, h)
                denom = batch["node_counts"].to(h.device).to(h.dtype)
                pooled = pooled / denom.clamp(min=1).unsqueeze(1)
                return inner.readout(pooled).squeeze(-1)

        return _DualCritic()


# ---------------------------------------------------------------------------
# Wrapper classes (v1-style module holders)
# ---------------------------------------------------------------------------


class _DualTSCriticBase:
    role: str = "?"

    def __init__(self, *, config: DualTSModelConfig | None = None,
                 s_a_angstrom: float | None = None,
                 a_max: float = 0.1) -> None:
        self.torch = require_torch()
        self.config = config or DualTSModelConfig()
        self.a_max = float(a_max)
        # frozen scale rule: s_a = a_max unless an explicit (frozen) value
        # is registered; the rule string travels with every checkpoint.
        self.s_a = float(s_a_angstrom) if s_a_angstrom is not None \
            else self.a_max
        if not (np.isfinite(self.s_a) and self.s_a > 0.0):
            raise DualTSCriticError("SCALE_INVALID",
                                    "s_a_angstrom must be finite > 0")
        self.module = _TorchDualCriticFactory().build(
            self.torch, self.config,
            {"use_action": self.role == "Q"})
        # independent-init provenance (recorded; NOT part of state_dict)
        self.init_state_digest = model_state_digest(
            self.module.state_dict())

    def parameters(self):
        return self.module.parameters()

    def train_mode(self, flag: bool) -> None:
        self.module.train(flag)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state) -> None:
        self.module.load_state_dict(state)

    def to(self, device):
        self.module.to(device)
        return self

    def to_dtype(self, dtype):
        self.module.to(dtype)
        return self

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.module.parameters())

    def identity_payload(self) -> dict:
        return {
            "critics_version": CRITICS_VERSION,
            "role": self.role,
            "s_a_angstrom": float(self.s_a),
            "s_a_rule": S_A_RULE,
            "a_max_angstrom": float(self.a_max),
            "action_input_version": (Q_ACTION_INPUT_VERSION
                                     if self.role == "Q" else None),
            "coordinates_unit": COORDINATES_UNIT,
            "config": {
                "d_model": self.config.d_model,
                "d_element": self.config.d_element,
                "max_z": self.config.max_z,
                "hidden": self.config.hidden,
                "n_layers": self.config.n_layers,
                "n_basis": self.config.n_basis,
                "cutoff": self.config.cutoff,
                "light": self.config.light,
                "signed_vector_messages": self.config.signed_vector_messages,
            },
            "init_state_digest": self.init_state_digest,
        }


class DualTSQCritic(_DualTSCriticBase):
    """Q(s, a_physical): one independent encoder + geometric action head."""

    role = "Q"

    def __call__(self, batch, a_angstrom, action_version: str):
        return self.module(batch, a_angstrom,
                           action_version=action_version, s_a=self.s_a,
                           a_max=self.a_max)


class DualTSVCritic(_DualTSCriticBase):
    """V(s): one independent encoder + readout (no action channels)."""

    role = "V"

    def __call__(self, batch):
        return self.module(batch)


__all__ = [
    "A_BOUND_TOL",
    "COORDINATES_UNIT",
    "CRITICS_VERSION",
    "DualTSCriticError",
    "DualTSQCritic",
    "DualTSVCritic",
    "Q_ACTION_INPUT_VERSION",
    "S_A_RULE",
    "geometric_action_features",
    "model_state_digest",
    "validate_physical_action",
]
