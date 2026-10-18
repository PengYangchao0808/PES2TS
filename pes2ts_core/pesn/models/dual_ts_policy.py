"""Dual-TS radial policy head (WP3-B, plan v4 §I.8 action contract).

Action semantics exactly as the v1 fix and the contract
``pesn_dual_ts_action_v2`` freeze them (a_max = 0.1 Å first round):

* per-atom latent ``u_i ~ N(mu_i, sigma_i^2 I3)`` (dimensionless);
* radial squash ``z_i = tanh(||u_i||) u_i / ||u_i||`` (continuous limit 0
  at the origin), ``a_i = a_max * z_i`` (Å);
* ``sigma = sigma_min + softplus(raw)`` — never upper-saturates and never
  has zero gradient (the v1 hard-clamp regression fix, reused verbatim);
* ``log pi_z = log N(u; mu, sigma) − [log sech^2(r) + 2 log(tanh(r)/r)]``
  per atom with the small-``r`` series keeping the log-det finite;
  ``log pi_a = log pi_z − 3N log(a_max)`` — BC trains on ``log pi_z``;
* deterministic mode ``a = a_max * tanh(||mu||) mu / ||mu||``.

Head expressiveness (plan §I.9): the DEFAULT ``mu = softplus(w(h)) ⊙ v``
is a POSITIVE per-atom gate on the encoder's equivariant vector field —
the structure the diagnostics lane must probe.  A signed per-atom gate
``mu = (w(h)) ⊙ v`` is available behind ``signed_mu_gate=True`` as a
FLAGGED alternative; it is never the default and never silently enabled.
"""
from __future__ import annotations

import math
from typing import Final

from pes2ts_core.pesn.models import require_torch
from pes2ts_core.pesn.models.dual_egnn import DualTSEGNN, DualTSModelConfig
from pes2ts_core.pesn.models.direct_policy import (
    radial_log_det_per_atom,
    radial_transform_stable,
)

DUAL_TS_POLICY_VERSION: Final[str] = "pesn_dual_ts_policy_v1.0"
DEFAULT_SIGMA_RANGE: Final[tuple[float, float]] = (0.02, 1.0)
DEFAULT_A_MAX_ANGSTROM: Final[float] = 0.1


class DualTSPolicyError(RuntimeError):
    """Typed dual-TS policy misuse."""


class _TorchDualPolicyFactory:
    def build(self, torch, encoder_module, config: dict):
        nn = torch.nn

        class _DualPolicy(nn.Module):
            def __init__(inner) -> None:
                super().__init__()
                inner.encoder = encoder_module
                inner.signed_mu_gate = bool(config["signed_mu_gate"])
                # default: positive per-atom gate (v1 structure — the
                # diagnostics lane's target); signed variant: raw linear.
                inner.mu_gate = nn.Linear(config["hidden"], 1)
                # sigma = sigma_min + softplus(raw): the v1 fix reused
                inner.raw_sigma_head = nn.Linear(config["hidden"], 1)
                nn.init.normal_(inner.raw_sigma_head.weight, std=0.01)
                sigma0 = 0.1
                target = max(sigma0 - config["sigma_min"], 1e-6)
                with torch.no_grad():
                    inner.raw_sigma_head.bias.fill_(math.log(math.expm1(target)))
                inner.action_max = config["action_max"]
                inner.sigma_min = config["sigma_min"]

            def _mu(inner, h, v):
                raw = inner.mu_gate(h)
                gate = raw if inner.signed_mu_gate else nn.functional.softplus(raw)
                return v * gate

            def _sigma(inner, h):
                raw = inner.raw_sigma_head(h)
                return inner.sigma_min + nn.functional.softplus(raw)

            def forward(inner, inputs: dict) -> dict:
                enc = inner.encoder(inputs)
                h, v = enc["h"], enc["v"]
                mu = inner._mu(h, v)
                sigma = inner._sigma(h)
                log_sigma = sigma.log().squeeze(-1)
                return {"mu": mu, "log_sigma": log_sigma,
                        "sigma": sigma.squeeze(-1), "h": h, "v": v}

            def log_prob_z(inner, mu, log_sigma, u) -> dict:
                sigma = log_sigma.exp().unsqueeze(-1)
                resid = (u - mu) / sigma
                gaussian = (-1.5 * math.log(2.0 * math.pi)
                            - 3.0 * torch.log(sigma).squeeze(-1)
                            - 0.5 * (resid * resid).sum(-1))
                log_det = radial_log_det_per_atom(torch, u)
                per_atom = gaussian - log_det
                return {"per_atom": per_atom, "log_det": log_det,
                        "gaussian": gaussian}

            def sample(inner, mu, log_sigma, generator=None):
                sigma = log_sigma.exp().unsqueeze(-1)
                eps = torch.randn(mu.shape, generator=generator,
                                  device=mu.device, dtype=mu.dtype)
                u = mu + sigma * eps
                z = radial_transform_stable(torch, u)
                return {"u": u, "z": z, "a": inner.action_max * z}

            def deterministic(inner, mu):
                z = radial_transform_stable(torch, mu)
                return inner.action_max * z

        return _DualPolicy()


class DualTSPolicy:
    """Full dual-TS policy: dual-topology EGNN encoder + radial head."""

    def __init__(self, *, config: DualTSModelConfig | None = None,
                 action_max: float = DEFAULT_A_MAX_ANGSTROM,
                 sigma_range: tuple[float, float] = DEFAULT_SIGMA_RANGE,
                 signed_mu_gate: bool = False) -> None:
        self.torch = require_torch()
        self.config = config or DualTSModelConfig()
        self.action_max = float(action_max)
        self.sigma_min = float(sigma_range[0])
        self.signed_mu_gate = bool(signed_mu_gate)
        self.encoder = DualTSEGNN(self.config)
        self.module = _TorchDualPolicyFactory().build(
            self.torch, self.encoder.module,
            {
                "hidden": self.config.hidden,
                "action_max": self.action_max,
                "sigma_min": self.sigma_min,
                "signed_mu_gate": self.signed_mu_gate,
            })

    def __call__(self, inputs: dict) -> dict:
        return self.module(inputs)

    def parameters(self):
        return self.module.parameters()

    def train_mode(self, flag: bool) -> None:
        self.module.train(flag)

    def log_prob_z(self, mu, log_sigma, u) -> dict:
        return self.module.log_prob_z(mu, log_sigma, u)

    def sample(self, mu, log_sigma, generator=None) -> dict:
        return self.module.sample(mu, log_sigma, generator=generator)

    def deterministic(self, mu):
        return self.module.deterministic(mu)

    def state_dicts(self) -> dict:
        return {"encoder": self.encoder.state_dict(),
                "policy": self.module.state_dict()}

    def load_state_dicts(self, state: dict) -> None:
        self.encoder.load_state_dict(state["encoder"])
        self.module.load_state_dict(state["policy"])

    def to(self, device) -> "DualTSPolicy":
        self.encoder.to(device)
        self.module.to(device)
        return self

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = [
    "DEFAULT_A_MAX_ANGSTROM",
    "DEFAULT_SIGMA_RANGE",
    "DUAL_TS_POLICY_VERSION",
    "DualTSPolicy",
    "DualTSPolicyError",
]
