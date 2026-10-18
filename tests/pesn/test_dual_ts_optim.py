"""Lane C optimizer/EMA tests (synthetic; CPU only).

Covers the campaign_v1 warmup→cosine schedule shape, the legacy_v41 no-
scheduler preset, and the EMATracker shadow update / warmup / resume
round-trip.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pes2ts_core.pesn.offline.training.dual_ts_optim import (
    CAMPAIGN_MIN_LR,
    CAMPAIGN_PEAK_LR,
    EMA_DECAY,
    EMATracker,
    build_optim,
    campaign_lr,
    campaign_warmup_steps,
)


def test_campaign_warmup_steps():
    assert campaign_warmup_steps(20000) == 1000
    assert campaign_warmup_steps(40000) == 2000
    assert campaign_warmup_steps(1) == 1  # floor


def test_campaign_lr_warmup_then_cosine():
    total = 20000
    warmup = campaign_warmup_steps(total)
    # warmup: ramps 0 -> peak
    assert campaign_lr(0, total_updates=total) == pytest.approx(0.0)
    assert campaign_lr(warmup - 1, total_updates=total) == pytest.approx(
        CAMPAIGN_PEAK_LR)
    # cosine decay: monotonically non-increasing after warmup, ends at min
    peak = campaign_lr(warmup, total_updates=total)
    assert peak == pytest.approx(CAMPAIGN_PEAK_LR, rel=1e-3)
    tail = peak
    for step in range(warmup, total, 1000):
        val = campaign_lr(step, total_updates=total)
        assert val <= tail + 1e-12
        tail = val
    assert campaign_lr(total - 1, total_updates=total) == pytest.approx(
        CAMPAIGN_MIN_LR, rel=1e-3)
    # warmup is monotonic non-decreasing 0 -> peak; post-warmup in [min, peak]
    warmup_tail = 0.0
    for step in range(warmup):
        val = campaign_lr(step, total_updates=total)
        assert val >= warmup_tail - 1e-12
        warmup_tail = val
    for step in (1000, 10000, 19999):
        assert CAMPAIGN_MIN_LR - 1e-12 <= campaign_lr(
            step, total_updates=total) <= CAMPAIGN_PEAK_LR + 1e-12


def test_build_optim_presets():
    params = [torch.nn.Parameter(torch.zeros(3))]
    opt, scheduler = build_optim(params, preset="campaign_v1",
                                 total_updates=20000)
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.defaults["lr"] == CAMPAIGN_PEAK_LR
    assert opt.defaults["weight_decay"] == 1e-5
    assert scheduler is not None

    opt2, scheduler2 = build_optim(params, preset="legacy_v41",
                                   total_updates=20000)
    assert isinstance(opt2, torch.optim.Adam)
    assert opt2.defaults["lr"] == 1e-3
    assert scheduler2 is None

    with pytest.raises(ValueError):
        build_optim(params, preset="bogus", total_updates=20000)


def test_scheduler_actually_ramps():
    params = [torch.nn.Parameter(torch.ones(3))]
    opt, scheduler = build_optim(params, preset="campaign_v1",
                                 total_updates=100)
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        scheduler.step()
    # lr reaches the peak during warmup and decays to the min at the end
    assert max(lrs) == pytest.approx(CAMPAIGN_PEAK_LR, rel=1e-3)
    assert lrs[-1] <= CAMPAIGN_PEAK_LR
    # the final scheduled lr is near the cosine floor
    assert opt.param_groups[0]["lr"] <= CAMPAIGN_MIN_LR * 1.5


def test_ema_shadow_tracks_decay():
    params = [torch.nn.Parameter(torch.zeros(2))]
    ema = EMATracker(decay=0.5)
    ema.register(params)
    with torch.no_grad():
        params[0].copy_(torch.tensor([1.0, 1.0]))
    ema.update(params)  # steps=1: shadow = 0.5*0 + 0.5*1 = 0.5
    assert torch.allclose(ema._shadow[0], torch.tensor([0.5, 0.5]))
    ema.update(params)  # steps=2: shadow = 0.5*0.5 + 0.5*1 = 0.75
    assert torch.allclose(ema._shadow[0], torch.tensor([0.75, 0.75]))


def test_ema_start_after_warmup_syncs_then_tracks():
    params = [torch.nn.Parameter(torch.zeros(2))]
    ema = EMATracker(decay=0.5, start_after_warmup=2)
    ema.register(params)
    with torch.no_grad():
        params[0].copy_(torch.tensor([1.0, 1.0]))
    ema.update(params)  # steps=1 <= 2 -> shadow = raw = [1,1]
    ema.update(params)  # steps=2 <= 2 -> shadow = raw = [1,1]
    assert torch.allclose(ema._shadow[0], torch.tensor([1.0, 1.0]))
    with torch.no_grad():
        params[0].copy_(torch.tensor([2.0, 2.0]))
    ema.update(params)  # steps=3 > 2 -> 0.5*1 + 0.5*2 = 1.5
    assert torch.allclose(ema._shadow[0], torch.tensor([1.5, 1.5]))


def test_ema_state_dict_roundtrip():
    params = [torch.nn.Parameter(torch.zeros(2)),
              torch.nn.Parameter(torch.zeros(3))]
    ema = EMATracker(decay=0.5, start_after_warmup=1)
    ema.register(params)
    for k in range(5):
        with torch.no_grad():
            for p in params:
                p.add_(torch.full_like(p, k * 0.1))
        ema.update(params)
    state = ema.state_dict()
    assert state["decay"] == 0.5 and state["steps"] == 5

    params2 = [torch.nn.Parameter(torch.zeros(2)),
               torch.nn.Parameter(torch.zeros(3))]
    ema2 = EMATracker(decay=0.5, start_after_warmup=1)
    ema2.load_state_dict(state, params2)
    assert ema2.steps == 5
    for sh1, sh2 in zip(ema._shadow, ema2._shadow):
        assert torch.allclose(sh1, sh2)


def test_ema_apply_to_copies_shadow():
    params = [torch.nn.Parameter(torch.zeros(2))]
    ema = EMATracker(decay=0.5)
    ema.register(params)
    with torch.no_grad():
        params[0].copy_(torch.tensor([1.0, 1.0]))
    ema.update(params)  # shadow -> 0.5
    with torch.no_grad():
        params[0].copy_(torch.tensor([9.0, 9.0]))
    ema.apply_to(params)  # restore shadow [0.5, 0.5]
    assert torch.allclose(params[0], torch.tensor([0.5, 0.5]))


def test_ema_rejects_bad_decay_and_mismatch():
    with pytest.raises(ValueError):
        EMATracker(decay=0.0)
    with pytest.raises(ValueError):
        EMATracker(decay=1.1)
    params = [torch.nn.Parameter(torch.zeros(2))]
    ema = EMATracker(decay=0.5)
    ema.register(params)
    with pytest.raises(ValueError):
        ema.load_state_dict(
            {"decay": 0.5, "start_after_warmup": 0, "steps": 0,
             "registered": True, "shadow": [torch.zeros(3)]}, params)
