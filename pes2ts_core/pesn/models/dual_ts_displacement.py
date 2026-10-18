"""Unbounded dual-TS displacement head (contract §5, ADR-0024 item 1/2).

``dual_ts_unbounded_displacement_v1``: ``a_hat = readout(h, v)`` in Å, with
NO ``0.1·tanh`` upper bound and NO inverse-tanh / log-Jacobian — the readout
reuses the existing mu path (a per-atom scalar gate on the encoder's
equivariant vector field) but drops the radial transform entirely.  The gate
is a RAW linear readout (signed, unbounded) so the magnitude is not clamped
and the direction may oppose ``v``.

H1 residual form: ``use_prior_skip=True`` adds a frozen analytic prior
``a_phys`` (precomputed from ``(g, m, Δs_fit, eps_g, tau_g)``) outside the
parameter graph — ``a_hat = a_phys + readout(h, v)``.  The prior is passed as
a plain ``(N, 3)`` tensor to :meth:`forward`, never registered as a
parameter.  Per-atom ``residual_norm`` / ``skip_norm`` (and their ratio, with
``NaN`` where the skip norm is zero) are reported alongside the composite.

The encoder is duck-typed: anything with ``forward(inputs) -> (h, v)``
(a dict ``{"h", "v"}`` or a ``(h, v)`` tuple), where ``v`` is ``(N, 3)`` or
``(N, C, 3)``.  Inject the existing ``DualTSEGNN`` module or this lane's
``DualTSPaiNNEncoder`` module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import nn

from pes2ts_core.pesn.models.dual_egnn import DualTSModelConfig, DualTSEGNN
from pes2ts_core.pesn.models.dual_egnn_painn import (
    DualTSPaiNNEncoder,
    DualTSPaiNNEncoderConfig,
)

DUAL_TS_UNBOUNDED_DISPLACEMENT_VERSION: Final[str] = (
    "dual_ts_unbounded_displacement_v1"
)


@dataclass(frozen=True, slots=True)
class DisplacementHeadConfig:
    """Head sizes for the unbounded displacement readout."""

    readout_hidden: int = 64
    use_prior_skip: bool = False


class DualTSDisplacementPolicy(nn.Module):
    """Unbounded displacement policy: encoder + Å readout (+ optional skip)."""

    def __init__(self, encoder: nn.Module, config, head_config: DisplacementHeadConfig) -> None:
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.head_config = head_config
        hidden = int(config.hidden)
        self.readout_hidden = int(head_config.readout_hidden)
        self.use_prior_skip = bool(head_config.use_prior_skip)
        # Raw linear gate (signed, unbounded): a = gate(h) ⊙ v.  The bias is
        # an invariant scalar and does not break translation/rotation symmetry.
        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden, self.readout_hidden), nn.SiLU(),
            nn.Linear(self.readout_hidden, 1))

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _readout(self, h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        gate = self.readout_mlp(h)                       # (N, 1)
        if v.dim() == 3:
            # (N, C, 3): one shared scalar gate across channels, summed.
            return (gate.unsqueeze(-1) * v).sum(dim=1)   # (N, 3)
        return gate * v                                  # (N, 1) * (N, 3)

    def forward(self, inputs: dict, a_phys: torch.Tensor | None = None) -> dict:
        enc = self.encoder(inputs)
        if isinstance(enc, dict):
            h, v = enc["h"], enc["v"]
        else:
            h, v = enc
        residual = self._readout(h, v)
        if not self.use_prior_skip:
            return {"a_hat": residual, "h": h, "v": v, "residual": residual}
        if a_phys is None:
            raise ValueError(
                "use_prior_skip=True requires the frozen a_phys (N,3) tensor")
        a_phys = a_phys.to(residual.dtype)
        a_hat = a_phys + residual
        residual_norm = residual.norm(dim=-1)             # (N,)
        skip_norm = a_phys.norm(dim=-1)                   # (N,)
        # Zero-denominator ratio is NaN (single column, explicit), never inf.
        ratio = torch.where(
            skip_norm > 0.0,
            residual_norm / skip_norm,
            torch.full_like(skip_norm, float("nan")))
        return {
            "a_hat": a_hat, "h": h, "v": v,
            "residual": residual, "a_phys": a_phys,
            "residual_norm": residual_norm, "skip_norm": skip_norm,
            "residual_skip_ratio": ratio,
        }


def build_displacement_policy(
    model_config: DualTSModelConfig,
    head_config: DisplacementHeadConfig,
    encoder_kind: str = "egnn",
    vector_channels: int = 16,
    contact_edges: bool = False,
    hidden: int | None = None,
    n_layers: int | None = None,
    edge_feat_dim: int = 4,
) -> DualTSDisplacementPolicy:
    """Build an unbounded displacement policy on an egnn or painn encoder.

    The width sweep (hidden ∈ {64, 96, 128}, n_layers=3) is applied with
    ``light=False`` and explicit ``hidden``/``n_layers`` so the values are
    never clamped by the legacy light cap.  ``edge_feat_dim`` widens the
    per-endpoint edge input (4=v3, 7=v4/E1) in both encoders.
    """
    hidden = int(model_config.hidden if hidden is None else hidden)
    n_layers = int(model_config.n_layers if n_layers is None else n_layers)
    if encoder_kind == "egnn":
        cfg = DualTSModelConfig(
            d_model=model_config.d_model, d_element=model_config.d_element,
            max_z=model_config.max_z, hidden=hidden, n_layers=n_layers,
            n_basis=model_config.n_basis, cutoff=model_config.cutoff,
            light=False, signed_vector_messages=model_config.signed_vector_messages,
            edge_feat_dim=edge_feat_dim)
        encoder = DualTSEGNN(cfg)
    elif encoder_kind == "painn":
        cfg = DualTSPaiNNEncoderConfig(
            d_model=model_config.d_model, d_element=model_config.d_element,
            max_z=model_config.max_z, hidden=hidden, n_layers=n_layers,
            n_basis=model_config.n_basis, cutoff=model_config.cutoff,
            vector_channels=vector_channels, contact_edges=contact_edges,
            edge_feat_dim=edge_feat_dim)
        encoder = DualTSPaiNNEncoder(cfg)
    else:
        raise ValueError(f"unknown encoder_kind {encoder_kind!r} (egnn|painn)")
    return DualTSDisplacementPolicy(encoder.module, cfg, head_config)


__all__ = [
    "DUAL_TS_UNBOUNDED_DISPLACEMENT_VERSION",
    "DisplacementHeadConfig",
    "DualTSDisplacementPolicy",
    "build_displacement_policy",
]
