"""Unified V5.6 trainer — the **only** training loop in the repository.

This module is the single place where ``loss.backward()`` and
``optimizer.step()`` are allowed to appear (enforced by
``tests/v5/test_single_train_loop.py`` and the repo arch-guard).  Every model
(any encoder/core/head combination from the model registry) is trained by this
one loop so the architecture comparison stays fair.

Discipline (plan §3.4):
* hidden state of stateful cores only ever lives **within one trajectory** — the
  trainer never carries state across batches/reactions;
* the same optimizer/early-stopping/checkpoint rules apply to all architectures;
* checkpoint + ledger files record full reproducibility identity (config hash,
  dataset hash, seed, parameter counts, weights checksum).

Extension points (all optional, backward compatible):
* ``callbacks``: :class:`~pes2ts_v5.training.callbacks.Callback` list driven by
  ``training.callbacks`` in the experiment YAML (cost ledger, provenance,
  checkpoint-best, OOM guard, logging, early stop);
* ``loss_fn``: named weighted loss from ``loss.components`` (defaults to the
  model's legacy masked multi-task loss);
* ``TrainConfig.from_experiment(cfg)``: derive trainer knobs from an
  :class:`~pes2ts_v5.configs.schema.ExperimentConfig`.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

from pes2ts_v5.data.contracts import standard_batch_view
from pes2ts_v5.evaluation.single_step import evaluate_model_batches
from pes2ts_v5.training._util import (
    json_safe,
    param_count,
    resolve_device,
    set_seed,
    utc_now,
    weights_sha256,
)
from pes2ts_v5.training.callbacks import Callback, TrainerState

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pes2ts_v5.configs.schema import ExperimentConfig

__all__ = [
    "TrainConfig",
    "V5Trainer",
    "preflight_forward_backward",
    "resolve_device",
    "set_seed",
    "weights_sha256",
]


def _optimizer_for(name: str) -> type[torch.optim.Optimizer]:
    key = str(name or "adamw").strip().lower()
    table: dict[str, type[torch.optim.Optimizer]] = {
        "adamw": torch.optim.AdamW,
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
    }
    if key not in table:
        raise ValueError(f"unknown optimizer {name!r}; choices={sorted(table)}")
    return table[key]


@dataclass
class TrainConfig:
    """Flat trainer knobs (backward-compatible; ``from_experiment`` maps YAML)."""

    batch_size: int = 4
    lr: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    grad_accum: int = 1
    max_epochs: int = 5
    patience: int = 3
    seed: int = 17
    device: str = "auto"
    history_len: int = 16
    subset_fraction: float = 1.0
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    loss_weights: dict = field(default_factory=lambda: {
        "move": 1.0, "process": 0.5, "progress": 0.2, "trust": 0.2})
    cpu_threads: int = 1
    # -- V5.6 additions ----------------------------------------------------
    amp: bool = False
    ema_decay: float | None = None
    optimizer: str = "adamw"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_experiment(
        cls,
        cfg: "ExperimentConfig",
        **overrides: Any,
    ) -> "TrainConfig":
        """Derive trainer knobs from a validated ``ExperimentConfig``.

        ``overrides`` (e.g. ``max_epochs``/``max_train_batches``/``device`` from
        the CLI) take precedence over the YAML values.
        """
        from pes2ts_v5.losses.components import legacy_loss_weights

        training = cfg.training
        optimizer = training.optimizer
        kwargs: dict[str, Any] = {
            "batch_size": int(training.batch_size),
            "lr": float(optimizer.lr if optimizer.lr is not None else training.lr),
            "weight_decay": float(
                optimizer.weight_decay
                if optimizer.weight_decay is not None else training.weight_decay),
            "grad_clip": float(training.grad_clip),
            "grad_accum": int(training.grad_accum),
            "max_epochs": int(training.epochs),
            "patience": int(training.patience),
            "seed": int(cfg.seed),
            "device": str(training.device),
            "cpu_threads": int(training.cpu_threads),
            "amp": bool(training.amp),
            "ema_decay": training.ema_decay,
            "optimizer": str(optimizer.name),
            "loss_weights": legacy_loss_weights(cfg.loss),
        }
        kwargs.update({k: v for k, v in overrides.items() if v is not None})
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in kwargs.items() if k in known})


class V5Trainer:
    """Train one model on a V5 trajectory dataset with the shared protocol."""

    def __init__(
        self,
        model: torch.nn.Module,
        dataset,
        cfg: TrainConfig,
        *,
        train_split: str = "train",
        val_split: str = "valid",
        out_dir: str | Path | None = None,
        experiment_id: str = "exp",
        callbacks: list[Callback] | None = None,
        loss_fn: nn.Module | None = None,
        config_hash: str = "",
        provenance: dict | None = None,
        validate_contract: bool = True,
    ):
        self.model = model
        self.dataset = dataset
        self.cfg = cfg
        self.train_split = train_split
        self.val_split = val_split
        self.out_dir = Path(out_dir) if out_dir else None
        self.experiment_id = experiment_id
        self.device = resolve_device(cfg.device)
        self.model.to(self.device)
        set_seed(cfg.seed)
        if cfg.cpu_threads != 0:
            torch.set_num_threads(int(cfg.cpu_threads))

        optimizer_cls = _optimizer_for(cfg.optimizer)
        self.optimizer = optimizer_cls(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        # ---- optional AMP (guarded: no CPU autocast claim) -----------------
        self._amp_requested = bool(cfg.amp)
        self._amp_enabled = bool(cfg.amp) and self.device.type == "cuda"
        self._amp_reason = ""
        if self._amp_requested and not self._amp_enabled:
            self._amp_reason = (
                "AMP requested but device is CPU; disabled (no CPU autocast in "
                "this engine — V5.6 computes in float32)")
            warnings.warn(self._amp_reason, RuntimeWarning, stacklevel=2)
        self._scaler = None
        if self._amp_enabled:  # pragma: no cover - CPU-only environment
            try:
                self._scaler = torch.amp.GradScaler("cuda", enabled=True)
            except (AttributeError, TypeError):
                self._scaler = torch.cuda.amp.GradScaler(enabled=True)

        # ---- optional EMA --------------------------------------------------
        self._ema_state: dict[str, torch.Tensor] | None = None
        if cfg.ema_decay is not None:
            self._ema_state = {
                name: tensor.detach().clone()
                for name, tensor in self.model.state_dict().items()}

        self.callbacks: list[Callback] = list(callbacks or [])
        self._has_checkpoint_callback = any(
            bool(getattr(cb, "handles_checkpoints", False)) for cb in self.callbacks)
        self.loss_fn = loss_fn
        self.config_hash = str(config_hash or "")
        self.provenance: dict = dict(provenance or {})
        #: validate every batch against the shared ``pes2ts_data.contracts``
        #: StandardBatch contract (no-op when that package is absent)
        self.validate_contract = bool(validate_contract)
        self.contract_fields = frozenset(
            getattr(model, "REQUIRED_FIELDS", ()) or
            getattr(model, "required_fields", ()) or ())

        self.history: list[dict] = []
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self.start_epoch = 0
        self.state = self._make_state()

    # ---------------------------------------------------------------- state
    def _make_state(self) -> TrainerState:
        return TrainerState(
            experiment_id=self.experiment_id,
            seed=int(self.cfg.seed),
            device=str(self.device),
            max_epochs=int(self.cfg.max_epochs),
            batch_size=int(self.cfg.batch_size),
            params=param_count(self.model),
            config_hash=self.config_hash,
            dataset_hash=str(getattr(self.dataset, "dataset_hash", "") or ""),
            dataset_version=str(getattr(self.dataset, "dataset_version", "") or ""),
            out_dir=self.out_dir,
            model=self.model,
            optimizer=self.optimizer,
            trainer=self,
            started_at=utc_now(),
            extra={"provenance": json_safe(self.provenance),
                   "loss_components": self._loss_components()},
        )

    def _loss_components(self) -> list[dict]:
        describe = getattr(self.loss_fn, "describe", None)
        return list(describe()) if callable(describe) else []

    def _fire(self, hook: str) -> None:
        for callback in self.callbacks:
            getattr(callback, hook)(self.state)

    # ------------------------------------------------------------- batches
    def _with_contract(self, batches):
        """Yield ``(batch, meta)`` after the shared StandardBatch contract check.

        Raises ``pes2ts_data.contracts.ContractViolation`` (code ``MISSING_FIELD``
        / ``TRUTH_FIELD_FORBIDDEN`` / ...) instead of a bare ``KeyError`` when the
        dataset does not provide a field the model declared.  When
        ``pes2ts_data`` is not installed the check is a documented no-op.
        """
        for batch, metas in batches:
            if self.validate_contract:
                standard_batch_view(batch, metas,
                                    required_fields=self.contract_fields)
            yield batch, metas

    def _train_batches(self):
        return self._with_contract(self.dataset.iter_batches(
            self.train_split, batch_size=self.cfg.batch_size, shuffle=True,
            seed=self.cfg.seed, fraction=self.cfg.subset_fraction,
            max_batches=self.cfg.max_train_batches, drop_last=False))

    def _val_batches(self):
        return self._with_contract(self.dataset.iter_batches(
            self.val_split, batch_size=self.cfg.batch_size, shuffle=False,
            seed=self.cfg.seed, fraction=self.cfg.subset_fraction,
            max_batches=self.cfg.max_val_batches, drop_last=False))

    @staticmethod
    def _to_device(batch: dict, device) -> dict:
        return {k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in batch.items()}

    def _loss_parts(self, out: dict, batch: dict) -> dict:
        if self.loss_fn is not None:
            return self.loss_fn(out, batch)
        return self.model.losses(out, batch, self.cfg.loss_weights)

    def _autocast(self):
        if self._amp_enabled:  # pragma: no cover - CPU-only environment
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return _NullContext()

    # --------------------------------------------------------- optimisation
    def _backward(self, loss: torch.Tensor) -> None:
        if self._scaler is not None:  # pragma: no cover - CPU-only environment
            self._scaler.scale(loss).backward()
        else:
            loss.backward()

    def _optimizer_step(self) -> None:
        if self._scaler is not None:  # pragma: no cover - CPU-only environment
            self._scaler.unscale_(self.optimizer)
            if self.cfg.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            if self.cfg.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def _ema_update(self) -> None:
        if self._ema_state is None:
            return
        decay = float(self.cfg.ema_decay or 0.0)
        with torch.no_grad():
            for name, tensor in self.model.state_dict().items():
                shadow = self._ema_state.get(name)
                if shadow is None:
                    self._ema_state[name] = tensor.detach().clone()
                elif tensor.dtype.is_floating_point:
                    shadow.mul_(decay).add_(tensor.detach(), alpha=1.0 - decay)
                else:
                    shadow.copy_(tensor.detach())

    def _apply_ema(self) -> None:
        if self._ema_state is not None:
            self.model.load_state_dict(self._ema_state)

    def current_lr(self) -> float:
        if not self.optimizer.param_groups:
            return float(self.cfg.lr)
        return float(self.optimizer.param_groups[0]["lr"])

    # --------------------------------------------------------------- epochs
    def train(self) -> dict:
        t0 = time.time()
        stale = 0
        stopped = False
        state = self.state
        state.started_at = utc_now()
        self._fire("on_train_begin")
        for epoch in range(self.start_epoch, self.cfg.max_epochs):
            epoch_t0 = time.time()
            self.model.train()
            losses: list[float] = []
            self.optimizer.zero_grad(set_to_none=True)
            n_batches = 0
            state.epoch = epoch + 1
            state.epoch_step = 0
            self._fire("on_epoch_begin")
            for step, (batch, _meta) in enumerate(self._train_batches()):
                batch = self._to_device(batch, self.device)
                with self._autocast():
                    out = self.model(batch)
                    parts = self._loss_parts(out, batch)
                    loss = parts["total"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss at epoch {epoch} step {step}: {float(loss)}")
                self._backward(loss / max(self.cfg.grad_accum, 1))
                n_batches += 1
                accum = max(self.cfg.grad_accum, 1)
                if (step + 1) % accum == 0:
                    self._optimizer_step()
                    state.global_step += 1
                    if self._ema_state is not None:
                        self._ema_update()
                losses.append(float(loss.detach().cpu()))
                state.epoch_step = n_batches
                state.train_loss = float(np.mean(losses))
                state.elapsed_s = time.time() - t0
                self._fire("on_batch_end")
            if n_batches == 0:
                raise RuntimeError(f"no training batches for split {self.train_split!r}")
            accum = max(self.cfg.grad_accum, 1)
            if n_batches % accum != 0:  # flush the tail of a partial accumulation window
                self._optimizer_step()
                state.global_step += 1
                if self._ema_state is not None:
                    self._ema_update()

            val_metrics = self.evaluate()
            val_loss = float(val_metrics.get("val_loss", float("nan")))
            record = {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "val_loss": val_loss,
                "val_move_rmse": val_metrics.get("move_rmse"),
                "val_move_cosine": val_metrics.get("move_cosine"),
                "n_train_batches": n_batches,
                "elapsed_s": round(time.time() - t0, 2),
            }
            self.history.append(record)
            improved = bool(np.isfinite(val_loss) and val_loss < self.best_val_loss - 1e-8)
            if improved:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                stale = 0
            else:
                stale += 1

            state.history = self.history
            state.train_loss = record["train_loss"]
            state.val_loss = val_loss
            state.best_val_loss = self.best_val_loss
            state.best_epoch = self.best_epoch
            state.improved = improved
            state.metrics = dict(val_metrics)
            state.epoch_seconds = time.time() - epoch_t0
            state.elapsed_s = time.time() - t0
            state.extra["lr"] = self.current_lr()

            if not self._has_checkpoint_callback and improved:
                self.save_checkpoint("best.pt", epoch + 1)
            self.save_checkpoint("last.pt", epoch + 1)
            self._fire("on_epoch_end")

            if state.suggested_batch_size and state.suggested_batch_size != self.cfg.batch_size:
                self.cfg.batch_size = int(state.suggested_batch_size)
                state.batch_size = int(self.cfg.batch_size)
                state.metrics["batch_size_reduced_to"] = int(self.cfg.batch_size)
                state.suggested_batch_size = None
            if state.should_stop or stale >= max(self.cfg.patience, 1):
                stopped = True
                break

        ema_applied = False
        if self._ema_state is not None:
            self._apply_ema()
            ema_applied = True
            self.save_checkpoint("ema.pt", self.history[-1]["epoch"] if self.history else 0)

        state.elapsed_s = time.time() - t0
        state.metrics["train_seconds"] = round(time.time() - t0, 4)
        self._fire("on_train_end")
        return {
            "experiment_id": self.experiment_id,
            "epochs_run": len(self.history),
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "train_seconds": round(time.time() - t0, 2),
            "device": str(self.device),
            "stopped_early": stopped,
            "optimizer": self.cfg.optimizer,
            "amp": {
                "requested": self._amp_requested,
                "enabled": self._amp_enabled,
                "reason": self._amp_reason,
            },
            "ema": {"decay": self.cfg.ema_decay, "applied": ema_applied},
            "config_hash": self.config_hash,
            "dataset_hash": str(getattr(self.dataset, "dataset_hash", "") or ""),
            "callbacks": [cb.name for cb in self.callbacks],
        }

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        cfg = self.cfg
        batches = list(self._with_contract(self.dataset.iter_batches(
            self.val_split, batch_size=cfg.batch_size, shuffle=False,
            seed=cfg.seed, fraction=cfg.subset_fraction,
            max_batches=cfg.max_val_batches)))
        if not batches:
            return {"val_loss": float("nan"), "move_rmse": float("nan")}
        metrics = evaluate_model_batches(self.model, batches, device=self.device)
        total, n = 0.0, 0
        for batch, _meta in batches:
            batch = self._to_device(batch, self.device)
            out = self.model(batch)
            parts = self._loss_parts(out, batch)
            total += float(parts["total"].detach().cpu())
            n += 1
        metrics["val_loss"] = total / max(n, 1)
        return metrics

    # ----------------------------------------------------------- checkpoints
    def _checkpoint_dir(self) -> Path | None:
        if self.out_dir is None:
            return None
        path = self.out_dir / "checkpoints" / self.experiment_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_checkpoint(self, name: str, epoch: int) -> Path | None:
        ckpt_dir = self._checkpoint_dir()
        if ckpt_dir is None:
            return None
        path = ckpt_dir / name
        payload = {
            "experiment_id": self.experiment_id,
            "arch": getattr(self.model, "arch", "unknown"),
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "weights_sha256": weights_sha256(self.model),
            "train_config": self.cfg.to_dict(),
            "config_hash": self.config_hash,
            "callbacks": [cb.name for cb in self.callbacks],
            "provenance": json_safe(self.provenance),
            "dataset_hash": getattr(self.dataset, "dataset_hash", ""),
            "dataset_version": getattr(self.dataset, "dataset_version", ""),
            "history": self.history,
        }
        torch.save(payload, path)
        return path

    def load_checkpoint(self, path: str | Path) -> dict:
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state"])
        if payload.get("optimizer_state"):
            self.optimizer.load_state_dict(payload["optimizer_state"])
        self.start_epoch = int(payload["epoch"])
        self.best_val_loss = float(payload.get("best_val_loss", float("inf")))
        self.best_epoch = int(payload.get("best_epoch", -1))
        self.history = list(payload.get("history", []))
        state = self.state
        state.history = self.history
        state.best_val_loss = self.best_val_loss
        state.best_epoch = self.best_epoch
        state.epoch = self.start_epoch
        return payload


class _NullContext:
    """No-op context manager (CPU fallback for ``torch.autocast``)."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> bool:
        return False


def preflight_forward_backward(
    model: torch.nn.Module,
    batch: dict,
    *,
    loss_fn: nn.Module | None = None,
    loss_weights: dict | None = None,
    device: str = "cpu",
    grad_clip: float = 1.0,
) -> dict:
    """One forward + one backward on a batch (CLI ``smoke`` preflight only).

    Lives here because this module owns every ``backward`` call in the
    repository; the runner's ``smoke`` subcommand delegates to it.
    """
    model.to(device)
    model.train()
    moved = {key: (value.to(device) if torch.is_tensor(value) else value)
             for key, value in batch.items()}
    out = model(moved)
    parts = (loss_fn(out, moved) if loss_fn is not None
             else model.losses(out, moved, loss_weights))
    loss = parts["total"]
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite preflight loss: {float(loss)}")
    loss.backward()
    if grad_clip:
        nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
    n_grad = 0
    grads_finite = True
    for param in model.parameters():
        if param.grad is None:
            continue
        n_grad += 1
        if not bool(torch.isfinite(param.grad).all().item()):
            grads_finite = False
    return {
        "loss": float(loss.detach().cpu()),
        "loss_finite": True,
        "grads_finite": bool(grads_finite),
        "n_grad_params": int(n_grad),
        "device": str(device),
    }
