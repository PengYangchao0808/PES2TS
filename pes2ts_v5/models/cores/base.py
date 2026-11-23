"""Unified memory-core interface for the V5.6 architecture benchmark.

Every core maps a hidden sequence ``x [B, T, H]`` to ``y [B, T, H]`` and is
strictly **causal** (``y_t`` may only depend on ``x_{<=t}``).  Stateful cores
expose an opaque ``state`` object; the trainer keeps it only within a single
trajectory (never across reactions), and free-run rollout carries it step by
step.  ``step()`` is the deployment/inference entry (``batch=1`` friendly).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CoreBase(nn.Module):
    name: str = "core"
    paradigm: str = "unknown"
    stateful: bool = False
    note: str = ""

    def init_state(self, batch_size: int, device=None):
        """Return a fresh state for a new trajectory batch (None = stateless)."""
        return None

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        delta_s: torch.Tensor | None = None,
        state=None,
    ) -> tuple[torch.Tensor, object]:
        raise NotImplementedError

    def step(self, x_t: torch.Tensor, state=None, delta_s_t: torch.Tensor | None = None):
        """Single-step inference: ``x_t [B, H]`` -> (``y_t [B, H]``, next_state)."""
        ds = None if delta_s_t is None else delta_s_t.unsqueeze(1)
        y, state = self.forward(x_t.unsqueeze(1), lengths=None, delta_s=ds, state=state)
        return y[:, -1], state


def count_params(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def causal_left_pad(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Left-pad [B, T, H] along time (causal convolution helper)."""
    if pad <= 0:
        return x
    return torch.cat([x.new_zeros(x.shape[0], pad, x.shape[2]), x], dim=1)
