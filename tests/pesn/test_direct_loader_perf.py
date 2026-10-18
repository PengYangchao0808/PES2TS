"""WP-INFRA1 loader/performance tests (workers 0 vs 2 equivalence)."""
from __future__ import annotations

import numpy as np
import pytest

from pes2ts_core.pesn.offline.training.direct_loader import (
    LoaderSettings,
    ReactionBatchSampler,
    build_direct_dataloader,
)
from pes2ts_core.pesn.offline.training.performance import StepTimer
from tests.pesn.test_direct_bc import _MemDataset  # noqa: F401


def test_sampler_group_balanced_and_seeded():
    ds = _MemDataset()
    ids = ds.reaction_ids
    s1 = list(ReactionBatchSampler(ds, ids, batch_reactions=2, seed=1))
    s2 = list(ReactionBatchSampler(ds, ids, batch_reactions=2, seed=1))
    assert s1 == s2
    s3 = list(ReactionBatchSampler(ds, ids, batch_reactions=2, seed=2))
    assert s1 != s3
    # all five transitions of each reaction stay in the same batch
    for batch in s1:
        assert len(batch) % 5 == 0


@pytest.mark.filterwarnings("ignore")
def test_dataloader_workers_zero_vs_two_same_samples_cpu():
    ds = _MemDataset()
    ids = ds.reaction_ids
    results = {}
    for workers in (0, 2):
        settings = LoaderSettings(workers=workers, batch_reactions=2, seed=5,
                                  persistent_workers=False, prefetch_factor=1)
        loader = build_direct_dataloader(ds, ids, settings=settings,
                                         device_is_cuda=False, epoch=0)
        batches = []
        for batch in loader:
            batches.append({
                "actions": batch["actions"].numpy(),
                "n_graphs": int(batch["n_graphs"]),
                "z": batch["atomic_numbers"].numpy(),
            })
        results[workers] = batches
    assert len(results[0]) == len(results[2])
    for b0, b2 in zip(results[0], results[2]):
        assert b0["n_graphs"] == b2["n_graphs"]
        assert np.allclose(b0["actions"], b2["actions"])
        assert np.array_equal(b0["z"], b2["z"])


def test_step_timer_summary_synchronized_cpu_only():
    timer = StepTimer(sync_cuda=True)
    for _ in range(5):
        with timer:
            _ = sum(range(1000))
    summary = timer.summary()
    assert summary["n"] == 5
    assert summary["p50_ms"] <= summary["p95_ms"] <= summary["max_ms"]
