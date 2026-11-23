"""M10 Latent Neural ODE — fixed-step Euler latent flow (SIMPLIFIED, NFE recorded).

Simplification: fixed Euler steps (default 2) of a latent vector field applied
per frame, no adaptive solver and no external `torchdiffeq`.  ``last_nfe``
exposes the number of vector-field evaluations for the deployment-cost report.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pes2ts_v5.models.cores.base import CoreBase


class _ODEFunc(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh())

    def forward(self, z):
        return self.net(z)


class LatentNeuralODE(CoreBase):
    name = "M10"
    paradigm = "continuous-depth"
    stateful = False
    note = "SIMPLIFIED latent ODE: fixed Euler steps; NFE recorded per forward"

    def __init__(self, hidden_dim: int, layers: int = 2, steps: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.steps = int(steps)
        self.func = _ODEFunc(hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.last_nfe = 0

    def forward(self, x, lengths=None, delta_s=None, state=None):
        z = x
        dt = 1.0 / self.steps
        for _ in range(self.steps):
            z = z + dt * self.func(z)
        self.last_nfe = self.steps
        return self.drop(self.out(z)), None
