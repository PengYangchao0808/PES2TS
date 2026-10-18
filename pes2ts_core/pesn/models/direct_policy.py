"""Dimensionless radial direct policy (WP2, plan v3 §6.1).

``u_i ~ N(mu_i, sigma_i^2 I3)`` in the dimensionless latent; ``r_i = |u_i|``;
``z_i = tanh(r_i) * u_i / r_i`` in the unit ball; physical action
``a_i = a_max * z_i`` (angstrom). Per-atom Jacobian eigenvalues are
``sech^2(r)`` (radial, multiplicity 1) and ``tanh(r)/r`` (tangential,
multiplicity 2), so

``log pi_z = log p(u) - sum_i [ log sech^2(r_i) + 2 log(tanh(r_i)/r_i) ]``

with small-``r`` series keeping both the transform and its log-det finite
(the naive ``tanh(r)/r`` quotient is 0/0 at the origin). The inverse map is
``u = atanh(|z|) z / |z|``; boundary/non-finite inputs are rejected loudly,
never clipped probabilistically. ``log pi_a = log pi_z - 3N log(a_max)``;
BC/IQL/SAC train on ``log pi_z`` (plan §6.1) — both are recorded, never
mixed. ``sigma_i`` is one scalar per atom shared across directions.
"""
from __future__ import annotations

import math
from typing import Final

from pes2ts_core.pesn.models import require_torch
from pes2ts_core.pesn.models.egnn import DirectEGNNEncoder

POLICY_VERSION: Final[str] = "pesn_direct_policy_v1.0"
SMALL_R: Final[float] = 1e-4
DEFAULT_SIGMA_RANGE: Final[tuple[float, float]] = (0.02, 1.0)


class RadialPolicyError(RuntimeError):
    """Typed policy misuse (boundary violations, non-finite latents)."""


# ---------------------------------------------------------------------------
# Pure transforms (torch; float64-safe for the contract tests)
# ---------------------------------------------------------------------------

def _safe_ratio(torch, r):
    """``tanh(r)/r`` with the small-r series ``1 - r^2/3``."""
    series = 1.0 - r * r / 3.0
    naive = torch.tanh(r) / r.clamp(min=1e-12)
    return torch.where(r < SMALL_R, series, naive)


def radial_transform(torch, u):
    """Alias of :func:`radial_transform_stable` (kept for call-site clarity)."""
    return radial_transform_stable(torch, u)


def radial_transform_stable(torch, u):
    """``z = tanh(r) * u / r`` computed via the safe ratio directly."""
    r = u.norm(dim=-1, keepdim=True)
    ratio = _safe_ratio(torch, r)
    return ratio * u


def radial_log_det_per_atom(torch, u):
    """``log sech^2(r) + 2 log(tanh(r)/r)`` per atom (..., )."""
    r = u.norm(dim=-1)
    r2 = r * r
    # small-r series: log(1 - r^2) + 2 log(1 - r^2/3) ~ -(5/3) r^2
    small = -(5.0 / 3.0) * r2
    sech2 = 1.0 - torch.tanh(r) ** 2
    log_sech2 = torch.log(sech2.clamp(min=1e-30))
    ratio = _safe_ratio(torch, r.unsqueeze(-1)).squeeze(-1)
    naive = log_sech2 + 2.0 * torch.log(ratio.clamp(min=1e-30))
    return torch.where(r < SMALL_R, small, naive)


def radial_inverse(torch, z):
    """Unit-ball ``z`` -> latent ``u``; loud rejection outside the boundary."""
    rn = z.norm(dim=-1, keepdim=True)
    if bool((rn >= 1.0).any()):
        raise RadialPolicyError(
            f"inverse transform requires |z| < 1, got max {float(rn.max()):.6f}"
        )
    r = torch.atanh(rn.clamp(max=1.0 - 1e-7))
    u = r * z / rn.clamp(min=1e-12)
    if not bool(torch.isfinite(u).all()):
        raise RadialPolicyError("non-finite latent from inverse transform")
    return u


# ---------------------------------------------------------------------------
# Policy module
# ---------------------------------------------------------------------------

class _TorchPolicyFactory:
    def build(self, torch, encoder_module, config: dict):
        nn = torch.nn
        hidden = config["hidden"]

        class _Policy(nn.Module):
            def __init__(inner) -> None:
                super().__init__()
                inner.encoder = encoder_module
                # equivariant mu: the encoder's vector channel scaled by a
                # learned per-atom gate (rotates with the geometry by design)
                inner.mu_gate = nn.Linear(hidden, 1)
                # sigma = sigma_min + softplus(raw): never upper-saturates and
                # never has zero gradient (the previous hard clamp left the
                # head stuck at sigma_max with exactly zero gradient)
                inner.raw_sigma_head = nn.Linear(hidden, 1)
                nn.init.normal_(inner.raw_sigma_head.weight, std=0.01)
                sigma0 = 0.1
                target = max(sigma0 - config["sigma_min"], 1e-6)
                with torch.no_grad():
                    inner.raw_sigma_head.bias.fill_(math.log(math.expm1(target)))
                inner.action_max = config["action_max"]
                inner.sigma_min = config["sigma_min"]

            def _sigma(inner, h):
                raw = inner.raw_sigma_head(h)
                return inner.sigma_min + nn.functional.softplus(raw)

            def forward(inner, batch: dict) -> dict:
                enc = inner.encoder(batch)
                gate = torch.nn.functional.softplus(inner.mu_gate(enc["h"]))
                mu = enc["v"] * gate                       # (N,3) equivariant
                sigma = inner._sigma(h=enc["h"])           # (N,1) > sigma_min
                log_sigma = sigma.log().squeeze(-1)
                return {"mu": mu, "log_sigma": log_sigma, "sigma": sigma.squeeze(-1),
                        "encoder": enc}

            def log_prob_z(inner, mu, log_sigma, u) -> dict:
                sigma = log_sigma.exp().unsqueeze(-1)              # (N,1)
                resid = (u - mu) / sigma
                gaussian = (-1.5 * math.log(2.0 * math.pi)
                            - 3.0 * torch.log(sigma).squeeze(-1)
                            - 0.5 * (resid * resid).sum(-1))
                log_det = radial_log_det_per_atom(torch, u)
                per_atom = gaussian - log_det
                return {"per_atom": per_atom, "log_det": log_det}

            def sample(inner, mu, log_sigma, generator=None):
                sigma = log_sigma.exp().unsqueeze(-1)
                eps = torch.randn(mu.shape, generator=generator,
                                  device=mu.device, dtype=mu.dtype)
                u = mu + sigma * eps
                z = radial_transform_stable(torch, u)
                a = inner.action_max * z
                return {"u": u, "z": z, "a": a}

            def deterministic(inner, mu):
                z = radial_transform_stable(torch, mu)
                return inner.action_max * z

        return _Policy()


class RadialDirectPolicy:
    """Holder wrapper (same interface discipline as the encoder)."""

    def __init__(self, *, encoder: DirectEGNNEncoder | None = None,
                 action_max: float = 0.1,
                 sigma_range: tuple[float, float] = DEFAULT_SIGMA_RANGE,
                 light: bool = False) -> None:
        self.torch = require_torch()
        from pes2ts_core.pesn.models.egnn import EGNNConfig
        self.encoder = encoder or DirectEGNNEncoder(EGNNConfig(light=light))
        config = {
            "hidden": self.encoder.config.hidden,
            "action_max": float(action_max),
            "sigma_min": float(sigma_range[0]),
            "sigma_max": float(sigma_range[1]),
        }
        self.action_max = float(action_max)
        self.module = _TorchPolicyFactory().build(self.torch, self.encoder.module, config)

    def __call__(self, batch: dict) -> dict:
        return self.module(batch)

    def parameters(self):
        return self.module.parameters()

    def train_mode(self, flag: bool) -> None:
        self.module.train(flag)

    def state_dicts(self) -> dict:
        return {"policy": self.module.state_dict()}

    def load_state_dicts(self, state: dict) -> None:
        self.module.load_state_dict(state["policy"])

    def to(self, device) -> "RadialDirectPolicy":
        self.module.to(device)
        return self

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.module.parameters())


__all__ = [
    "DEFAULT_SIGMA_RANGE",
    "POLICY_VERSION",
    "RadialDirectPolicy",
    "RadialPolicyError",
    "radial_inverse",
    "radial_log_det_per_atom",
    "radial_transform_stable",
]
