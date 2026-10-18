"""V4.2 campaign optimizer + EMA (Lane C; contract §6).

Two presets drive every arm:

* ``campaign_v1`` — ``AdamW`` peak lr 3e-4 / weight_decay 1e-5 with a
  ``LambdaLR`` that does a linear warmup over the first 5% of updates
  (20k→1000 steps, 40k→2000 steps) and then cosine-decays to 1e-5 over the
  remainder.  Clip 1.0 is applied by the trainer, not here.
* ``legacy_v41`` — ``Adam`` lr 1e-3 with no scheduler (D0/D1 only; batch 64
  matches the old runs).  No weight decay (reproduces the old trainer's
  ``Adam(params, lr=1e-3)`` exactly).

:class:`EMATracker` keeps a shadow copy of the model parameters.  It starts
tracking only after ``start_after_warmup`` updates (during warmup the shadow
is simply re-synced to the raw weights, so ``raw`` and ``ema`` identities are
identical until EMA is "on").  ``state_dict()`` / ``load_state_dict()`` are
full (shadow tensors included) so a resumed run restores the exact EMA.
"""
from __future__ import annotations

import math
from typing import Final, Iterable, Sequence

from pes2ts_core.pesn.models import require_torch

#: Campaign preset hyper-parameters (contract §6 / ADR-0024 decision 5).
CAMPAIGN_PEAK_LR: Final[float] = 3e-4
CAMPAIGN_MIN_LR: Final[float] = 1e-5
CAMPAIGN_WEIGHT_DECAY: Final[float] = 1e-5
CAMPAIGN_WARMUP_FRACTION: Final[float] = 0.05

#: Legacy preset hyper-parameters (D0/D1 only).
LEGACY_LR: Final[float] = 1e-3

#: Frozen EMA decay (contract §6).
EMA_DECAY: Final[float] = 0.999

PRESETS: Final[tuple[str, ...]] = ("campaign_v1", "legacy_v41")


def campaign_warmup_steps(total_updates: int) -> int:
    """Warmup length = 5% of the update budget (>= 1)."""
    return max(1, int(round(CAMPAIGN_WARMUP_FRACTION * int(total_updates))))


def campaign_lr(step: int, *, total_updates: int) -> float:
    """Absolute lr at 0-indexed update ``step`` for the campaign schedule.

    ``step`` in ``[0, warmup)`` ramps linearly 0 → peak; then cosine decays
    peak → ``CAMPAIGN_MIN_LR`` by the last update.  Exposed (not just buried
    in a lambda) so tests can assert the exact warmup/cosine shape.
    """
    warmup = campaign_warmup_steps(total_updates)
    peak = CAMPAIGN_PEAK_LR
    floor = CAMPAIGN_MIN_LR
    total = max(1, int(total_updates))
    if step < warmup:
        return peak * (float(step) / float(max(1, warmup - 1)))
    progress = float(step - warmup) / float(max(1, total - 1 - warmup))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return floor + (peak - floor) * cosine


def build_optim(params: Iterable, *, preset: str, total_updates: int):
    """Return ``(optimizer, scheduler)`` for ``preset``.

    ``params`` is any iterable of ``torch.nn.Parameter``.  ``scheduler`` is
    ``None`` for ``legacy_v41`` (no scheduling, old semantics preserved).
    ``total_updates`` is ignored by ``legacy_v41``.
    """
    torch = require_torch()
    if preset not in PRESETS:
        raise ValueError(
            f"preset {preset!r} not in {PRESETS} (campaign_v1|legacy_v41)")
    params = list(params)
    if preset == "legacy_v41":
        return torch.optim.Adam(params, lr=LEGACY_LR), None
    optimizer = torch.optim.AdamW(
        params, lr=CAMPAIGN_PEAK_LR, weight_decay=CAMPAIGN_WEIGHT_DECAY)
    total = int(total_updates)

    def lr_lambda(step: int) -> float:
        return campaign_lr(step, total_updates=total) / CAMPAIGN_PEAK_LR

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


class EMATracker:
    """Exponential moving average of model parameters (raw + EMA identities).

    ``register(params)`` snapshots the shadow from the current weights.
    ``update(params)`` is called once per optimizer step; while
    ``steps <= start_after_warmup`` the shadow re-syncs to the raw weights
    (EMA off), afterwards ``shadow = decay*shadow + (1-decay)*raw``.
    """

    def __init__(self, *, decay: float = EMA_DECAY,
                 start_after_warmup: int = 0) -> None:
        if not (0.0 < decay <= 1.0):
            raise ValueError("decay must be in (0, 1]")
        if start_after_warmup < 0:
            raise ValueError("start_after_warmup must be >= 0")
        self.decay = float(decay)
        self.start_after_warmup = int(start_after_warmup)
        self._shadow: list = []
        self._registered = False
        self.steps = 0

    # -- lifecycle -----------------------------------------------------------

    def register(self, params: Iterable) -> None:
        params = list(params)
        if not params:
            raise ValueError("EMATracker.register requires non-empty params")
        self._shadow = [p.detach().clone() for p in params]
        self._registered = True

    def update(self, params: Iterable) -> None:
        params = list(params)
        if not self._registered:
            self.register(params)
        if len(params) != len(self._shadow):
            raise ValueError("EMA parameter count changed (config mismatch)")
        self.steps += 1
        for shadow, p in zip(self._shadow, params):
            if self.steps <= self.start_after_warmup:
                shadow.copy_(p.detach())
            else:
                shadow.mul_(self.decay).add_(
                    p.detach(), alpha=1.0 - self.decay)

    def apply_to(self, params: Iterable) -> None:
        """Copy the shadow weights back into ``params`` (EMA eval identity)."""
        params = list(params)
        if not self._registered:
            raise ValueError("EMATracker not registered")
        if len(params) != len(self._shadow):
            raise ValueError("EMA parameter count changed (config mismatch)")
        for p, shadow in zip(params, self._shadow):
            p.data.copy_(shadow)

    # -- resume state --------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "start_after_warmup": self.start_after_warmup,
            "steps": int(self.steps),
            "registered": bool(self._registered),
            "shadow": [t.detach().to("cpu").clone() for t in self._shadow],
        }

    def load_state_dict(self, state: dict, params: Iterable) -> None:
        params = list(params)
        shadow = list(state["shadow"])
        if len(shadow) != len(params):
            raise ValueError(
                "EMA shadow parameter count mismatch on restore: "
                f"{len(shadow)} != {len(params)}")
        for t, p in zip(shadow, params):
            if tuple(t.shape) != tuple(p.shape):
                raise ValueError(
                    "EMA shadow parameter shape mismatch on restore: "
                    f"{tuple(t.shape)} != {tuple(p.shape)}")
        self.decay = float(state["decay"])
        self.start_after_warmup = int(state["start_after_warmup"])
        self.steps = int(state["steps"])
        self._registered = bool(state.get("registered", True))
        self._shadow = [
            t.to(device=p.device, dtype=p.dtype).clone()
            for t, p in zip(shadow, params)
        ]


__all__ = [
    "CAMPAIGN_MIN_LR",
    "CAMPAIGN_PEAK_LR",
    "CAMPAIGN_WARMUP_FRACTION",
    "CAMPAIGN_WEIGHT_DECAY",
    "EMA_DECAY",
    "EMATracker",
    "LEGACY_LR",
    "PRESETS",
    "build_optim",
    "campaign_lr",
    "campaign_warmup_steps",
]
