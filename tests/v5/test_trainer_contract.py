"""Trainer contract: overfits, determinism, callbacks, checkpoints, ledgers.

Also covers the shared batch contract: a batch missing a model-declared field
must raise the typed ``pes2ts_data.contracts.ContractViolation`` (``MISSING_FIELD``)
instead of a bare ``KeyError``.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from pes2ts_v5.training._util import set_seed, weights_sha256
from pes2ts_v5.training.callbacks import Callback, build_callbacks
from pes2ts_v5.training.trainer import TrainConfig, V5Trainer

CORE_IDS = [f"M{i:02d}" for i in range(1, 15)]


class RecordingCallback(Callback):
    name = "recording"

    def __init__(self) -> None:
        self.events: list[tuple] = []

    def on_train_begin(self, state) -> None:
        self.events.append(("train_begin", None))

    def on_epoch_begin(self, state) -> None:
        self.events.append(("epoch_begin", int(state.epoch)))

    def on_batch_end(self, state) -> None:
        self.events.append(("batch_end", (int(state.epoch), int(state.epoch_step))))

    def on_epoch_end(self, state) -> None:
        self.events.append(("epoch_end", (int(state.epoch), float(state.val_loss))))

    def on_train_end(self, state) -> None:
        self.events.append(("train_end", len(state.history)))


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------

def test_single_batch_overfits_gru(build_trainer) -> None:
    trainer = build_trainer("gru", max_epochs=5, lr=0.02)
    report = trainer.train()
    losses = [row["train_loss"] for row in report["history"]]
    assert report["epochs_run"] == 5
    assert math.isfinite(losses[0]) and math.isfinite(losses[-1])
    assert losses[-1] < losses[0], losses


def test_at_least_8_of_14_cores_overfit(build_trainer) -> None:
    decreased: list[str] = []
    details: dict[str, tuple[float, float]] = {}
    for core in CORE_IDS:
        trainer = build_trainer(core, max_epochs=5, lr=0.02)
        report = trainer.train()
        losses = [row["train_loss"] for row in report["history"]]
        details[core] = (losses[0], losses[-1])
        if losses[-1] < losses[0]:
            decreased.append(core)
    assert len(decreased) >= 8, f"only {len(decreased)}/14 cores overfit: {details}"


def test_loss_fn_from_config_is_used(build_trainer, dataset) -> None:
    from pes2ts_v5.configs.schema import load_experiment_config
    from pes2ts_v5.losses.components import build_weighted_loss_from_config

    cfg = load_experiment_config(
        Path(__file__).resolve().parents[2] / "config/experiments/smoke.yaml")
    loss_fn = build_weighted_loss_from_config(cfg.loss)
    trainer = build_trainer("gru", max_epochs=1, loss_fn=loss_fn)
    report = trainer.train()
    assert math.isfinite(report["history"][0]["train_loss"])
    assert loss_fn.describe()[0]["implemented"] is True


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_same_seed_gives_bit_identical_initial_weights(build_model) -> None:
    set_seed(17)
    first = build_model("gru")
    first_sha = weights_sha256(first)
    set_seed(17)
    second = build_model("gru")
    assert weights_sha256(second) == first_sha
    for key, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[key]), key


def test_amp_is_guarded_on_cpu(build_trainer) -> None:
    trainer = build_trainer("gru", max_epochs=1, amp=True)
    report = trainer.train()
    assert report["amp"]["requested"] is True
    assert report["amp"]["enabled"] is False
    assert "CPU" in report["amp"]["reason"]


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def test_callbacks_fire_in_order(build_trainer, tmp_path) -> None:
    recorder = RecordingCallback()
    trainer = build_trainer("gru", max_epochs=2, callbacks=[recorder])
    trainer.train()
    events = recorder.events
    assert events[0] == ("train_begin", None)
    assert events[1] == ("epoch_begin", 1)
    assert events[-1][0] == "train_end"
    epoch_ends = [event for event in events if event[0] == "epoch_end"]
    assert [event[1][0] for event in epoch_ends] == [1, 2]
    batches = [event for event in events if event[0] == "batch_end"]
    assert batches and batches[0][1] == (1, 1)
    # epoch_begin(2) must come after epoch_end(1)
    end_first = events.index(("epoch_end", epoch_ends[0][1]))
    begin_second = events.index(("epoch_begin", 2))
    assert end_first < begin_second


def test_early_stop_callback_requests_stop(build_trainer) -> None:
    callbacks = build_callbacks(["early_stop"], out_dir=None, patience=1)
    # lr=0 => weights frozen => val loss never improves after epoch 1 => stop
    trainer = build_trainer("gru", max_epochs=6, patience=1, lr=0.0,
                            callbacks=callbacks)
    report = trainer.train()
    assert report["stopped_early"] is True
    assert report["epochs_run"] < 6


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def test_checkpoint_save_load_roundtrip(build_trainer, tmp_path) -> None:
    run_dir = tmp_path / "run"
    trainer = build_trainer("gru", max_epochs=1, out_dir=run_dir)
    report = trainer.train()
    sha_after_training = weights_sha256(trainer.model)
    checkpoint = run_dir / "checkpoints" / "test-run" / "last.pt"
    assert checkpoint.exists()

    restored = build_trainer("gru", max_epochs=1, out_dir=run_dir)
    payload = restored.load_checkpoint(checkpoint)
    assert payload["epoch"] == report["epochs_run"]
    assert weights_sha256(restored.model) == sha_after_training
    assert restored.best_val_loss == pytest.approx(report["best_val_loss"])


# ---------------------------------------------------------------------------
# Provenance / cost ledgers (R5 + P3)
# ---------------------------------------------------------------------------

def test_provenance_and_cost_ledger_written(build_trainer, tmp_path, json_read) -> None:
    run_dir = tmp_path / "ledger-run"
    callbacks = build_callbacks(
        ["checkpoint_best", "cost_ledger", "provenance_ledger", "log"],
        out_dir=run_dir, patience=5)
    trainer = build_trainer("gru", max_epochs=2, callbacks=callbacks, out_dir=run_dir)
    trainer.train()

    cost_path = run_dir / "cost_ledger.jsonl"
    assert cost_path.exists()
    rows = [json_read_line(line) for line in
            cost_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2
    required = {"wall_time_s", "epoch", "train_loss", "val_loss", "params",
                "device", "seed", "config_hash"}
    assert required <= set(rows[0])
    assert rows[0]["seed"] == 17
    assert rows[0]["params"] > 0
    assert all(row["epoch"] in (1, 2) for row in rows)

    prov = json_read(run_dir / "provenance.json")
    for key in ("git_commit", "config_hash", "dataset_hash", "dataset_version",
                "seed", "versions", "param_counts", "weights_sha256"):
        assert key in prov
    assert prov["seed"] == 17
    assert prov["versions"]["python"] and prov["versions"]["torch"]
    assert prov["config_hash"] == "test-config-hash"
    assert prov["dataset_hash"]
    assert prov["param_counts"]["trainable_total"] > 0
    assert prov["weights_sha256"]  # written at train end
    assert (run_dir / "checkpoints" / "test-run" / "best.pt").exists()
    assert (run_dir / "train_log.jsonl").exists()


def test_oom_guard_requests_smaller_batch() -> None:
    """OomGuard documents+requests a batch reduction; CPU degrades gracefully."""
    from pes2ts_v5.training import callbacks as callbacks_module
    from pes2ts_v5.training.callbacks import OomGuard, TrainerState

    guard = OomGuard(max_memory_mb=1000.0, reduce_factor=0.5, min_batch_size=1)
    state = TrainerState(device="cuda:0", batch_size=8)
    original_peak = callbacks_module.peak_memory_mb
    callbacks_module.peak_memory_mb = lambda device=None: 4000.0  # above limit
    try:
        guard.on_batch_end(state)
    finally:
        callbacks_module.peak_memory_mb = original_peak
    assert state.suggested_batch_size == 4
    assert state.metrics["peak_device_mem_mb"] == 4000.0

    cpu_state = TrainerState(device="cpu", batch_size=8)
    OomGuard().on_batch_end(cpu_state)
    assert cpu_state.suggested_batch_size is None  # graceful CPU no-op
    assert cpu_state.metrics["oom_guard"] in {"unobservable_on_cpu", "no_device_limit"}


def json_read_line(line: str) -> dict:
    import json

    return json.loads(line)


# ---------------------------------------------------------------------------
# Shared batch contract
# ---------------------------------------------------------------------------

class _StubDataset:
    """Dataset yielding one modified batch (contract-violation probe)."""

    dataset_hash = "stub-hash"
    dataset_version = "stub-v1"

    def __init__(self, batch: dict, modify):
        self._batch = batch
        self._modify = modify

    def iter_batches(self, split, **kwargs):
        batch = self._modify(dict(self._batch))
        meta = [{"trajectory_id": "T0"}, {"trajectory_id": "T1"}]
        yield batch, meta


def _build_stub_trainer(dataset, batch: dict, modify, core: str = "M05") -> V5Trainer:
    from pes2ts_v5.configs.schema import CoreConfig, HeadConfig, ModelConfig
    from pes2ts_v5.models.registry import build_model_from_config

    model = build_model_from_config(
        ModelConfig(core=CoreConfig(name=core, hidden_dim=32), head=HeadConfig()))
    cfg = TrainConfig(batch_size=2, max_epochs=1, max_train_batches=1,
                      max_val_batches=1, device="cpu", cpu_threads=1, seed=17)
    return V5Trainer(model, dataset, cfg, out_dir=None, validate_contract=True)


def test_missing_declared_field_raises_typed_contract_violation(one_batch) -> None:
    contracts = pytest.importorskip("pes2ts_data.contracts")
    batch, _ = one_batch
    dataset = _StubDataset(batch, lambda b: {k: v for k, v in b.items() if k != "delta_s"})
    trainer = _build_stub_trainer(dataset, batch, lambda b: b)
    with pytest.raises(contracts.ContractViolation) as excinfo:
        trainer.train()
    assert excinfo.value.code == "MISSING_FIELD"
    assert "delta_s" in str(excinfo.value)


def test_truth_field_in_batch_raises_typed_contract_violation(one_batch) -> None:
    contracts = pytest.importorskip("pes2ts_data.contracts")
    batch, _ = one_batch
    dataset = _StubDataset(batch, lambda b: {**b, "ts_xyz": torch.zeros(1)})
    trainer = _build_stub_trainer(dataset, batch, lambda b: b)
    with pytest.raises(contracts.ContractViolation) as excinfo:
        trainer.train()
    assert excinfo.value.code == "TRUTH_FIELD_FORBIDDEN"
