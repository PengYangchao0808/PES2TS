"""Direct IQL trainer smoke on synthetic transitions (CPU, tiny)."""
from __future__ import annotations

import numpy as np
import pytest

from pes2ts_core.pesn.offline.training.direct_bc import DirectBCSettings  # noqa: F401
from pes2ts_core.pesn.offline.training.direct_iql import (
    DirectIQLSettings,
    DirectIQLTrainer,
    episode_mc_returns,
)
from tests.pesn.test_direct_bc import _MemDataset, _sample  # noqa: F401


def test_episode_mc_returns_telescope():
    ds = _MemDataset()
    mc = episode_mc_returns(ds)
    for rid, g in mc.items():
        rewards = np.asarray([s.reward for s in ds.get_reaction(rid)])
        assert g[0] == pytest.approx(rewards.sum())
        assert g[-1] == pytest.approx(rewards[-1])


def test_iql_trainer_runs_and_losses_finite_on_cpu():
    ds = _MemDataset()
    settings = DirectIQLSettings(
        total_updates=6, batch_transitions=8, seed=3,
        hidden=16, n_layers=1, device="cpu", require_cuda=False,
        cpu_intra_threads=1, cpu_inter_threads=1, log_every=2, eval_every=5)
    trainer = DirectIQLTrainer(
        settings=settings, train_dataset=ds,
        run_dir="/tmp/opencode/iql_smoke_test", config_sha256="cfg-test",
        dataset_manifest_sha256="ds-test")
    result = trainer.train()
    assert len(result["history"]) >= 2
    for entry in result["history"]:
        assert np.isfinite(entry["loss_v"])
        assert np.isfinite(entry["loss_q"])
        assert np.isfinite(entry["loss_pi"])
        assert entry["q_finite"]
        assert entry["awr_ess"] > 0
    report = trainer.calibration_report()
    assert report["v_finite_all"]
    trainer.write_run_record()
    from pathlib import Path
    assert Path("/tmp/opencode/iql_smoke_test/checkpoint_last.pt").is_file()
