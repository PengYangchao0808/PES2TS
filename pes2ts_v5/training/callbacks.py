"""Training callbacks: checkpointing, cost/provenance ledgers, OOM guard, logging,
early stopping.

The trainer owns the optimisation loop (it is the only place with
``loss.backward()``/``optimizer.step()``); callbacks only observe and record.
Every hook receives a :class:`TrainerState` view: a plain dataclass populated by
the trainer that callbacks may read (and, for ``should_stop`` /
``suggested_batch_size``, request changes through documented fields).

Registered names (``CALLBACK_REGISTRY``) are what ``training.callbacks`` in an
experiment YAML refers to, e.g. ``[checkpoint_best, cost_ledger, oom_guard]``.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from pes2ts_v5.training._util import (
    append_jsonl,
    device_total_mb,
    git_commit_sha,
    git_is_dirty,
    json_safe,
    peak_memory_mb,
    runtime_versions,
    utc_now,
    weights_sha256,
    write_json,
)


@dataclass
class TrainerState:
    """Trainer view handed to every callback hook.

    Read-only by convention, except for:

    * ``should_stop`` — a callback may set it to ``True`` to request stopping
      after the current epoch (honoured by :class:`V5Trainer`).
    * ``suggested_batch_size`` — a callback may set a smaller batch size; the
      trainer applies it for subsequent epochs (documented in ``OomGuard``).
    """

    experiment_id: str = "exp"
    seed: int = 17
    device: str = "cpu"
    max_epochs: int = 1
    epoch: int = 0                 # 1-based, current epoch
    global_step: int = 0           # optimizer steps since training began
    epoch_step: int = 0            # batches seen in the current epoch
    batch_size: int = 1
    train_loss: float = float("nan")
    val_loss: float = float("nan")
    best_val_loss: float = float("inf")
    best_epoch: int = -1
    improved: bool = False
    params: int = 0
    config_hash: str = ""
    dataset_hash: str = ""
    dataset_version: str = ""
    out_dir: Path | None = None
    elapsed_s: float = 0.0
    epoch_seconds: float = 0.0
    history: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    should_stop: bool = False
    suggested_batch_size: int | None = None
    model: torch.nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
    trainer: Any = None            # owning V5Trainer (documented: save_checkpoint)
    started_at: str = ""
    extra: dict = field(default_factory=dict)


class Callback:
    """Base class; override only the hooks you need."""

    name: str = "callback"
    #: ``True`` when the callback writes ``best.pt`` itself (the trainer then
    #: stops writing it internally, avoiding a double save).
    handles_checkpoints: bool = False

    def on_train_begin(self, state: TrainerState) -> None: ...

    def on_epoch_begin(self, state: TrainerState) -> None: ...

    def on_batch_end(self, state: TrainerState) -> None: ...

    def on_epoch_end(self, state: TrainerState) -> None: ...

    def on_train_end(self, state: TrainerState) -> None: ...


CALLBACK_REGISTRY: dict[str, type[Callback]] = {}
#: accepted YAML spellings that are not the canonical registered name
CALLBACK_ALIASES: dict[str, str] = {
    "provenance": "provenance_ledger",
    "provenance_ledger": "provenance_ledger",
    "checkpoint": "checkpoint_best",
    "logging": "log",
    "jsonl_log": "log",
    "early_stopping": "early_stop",
    "oom": "oom_guard",
}


def register_callback(name: str):
    """Class decorator registering a callback under ``name`` (duplicates fail)."""

    key = str(name).strip()

    def _decorator(cls: type[Callback]) -> type[Callback]:
        if not key:
            raise ValueError("callback name must be non-empty")
        if key in CALLBACK_REGISTRY:
            raise ValueError(
                f"callback {key!r} already registered as "
                f"{CALLBACK_REGISTRY[key].__name__}")
        if not issubclass(cls, Callback):
            raise TypeError(f"{cls!r} must subclass Callback")
        cls.name = key
        CALLBACK_REGISTRY[key] = cls
        return cls

    return _decorator


def available_callbacks() -> list[str]:
    return sorted(set(CALLBACK_REGISTRY) | set(CALLBACK_ALIASES))


def resolve_callback_name(name: str) -> str:
    """Resolve a YAML callback name (canonical or documented alias)."""
    key = str(name).strip().lower()
    if key in CALLBACK_REGISTRY:
        return key
    alias = CALLBACK_ALIASES.get(key)
    if alias is not None:
        return alias
    raise KeyError(
        f"unknown callback {name!r}; registered={sorted(CALLBACK_REGISTRY)}; "
        f"aliases={sorted(CALLBACK_ALIASES)}")



# ---------------------------------------------------------------------------
# Ledgers (R5 cost retention / P3 replayability)
# ---------------------------------------------------------------------------

@register_callback("checkpoint_best")
class CheckpointBest(Callback):
    """Write ``best.pt`` whenever validation loss improves.

    ``last.pt`` is always written by the trainer itself (resume semantics), so
    this callback only adds the best-weight write.
    """

    handles_checkpoints = True

    def __init__(self, out_dir: str | Path | None = None):
        self.out_dir = Path(out_dir) if out_dir is not None else None

    def on_epoch_end(self, state: TrainerState) -> None:
        if state.trainer is None or not state.improved:
            return
        state.trainer.save_checkpoint("best.pt", state.epoch)


@register_callback("cost_ledger")
class CostLedger(Callback):
    """Append one JSONL row per epoch: compute cost + loss + identity.

    Required row keys (R5 cost retention): ``wall_time_s``, ``epoch``,
    ``train_loss``, ``val_loss``, ``params``, ``device``, ``seed``,
    ``config_hash``.
    """

    def __init__(self, out_dir: str | Path | None = None,
                 filename: str = "cost_ledger.jsonl"):
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.filename = str(filename)

    def _path(self) -> Path | None:
        return None if self.out_dir is None else self.out_dir / self.filename

    def on_epoch_end(self, state: TrainerState) -> None:
        path = self._path()
        if path is None:
            return
        append_jsonl(path, {
            "timestamp": utc_now(),
            "experiment_id": state.experiment_id,
            "epoch": int(state.epoch),
            "wall_time_s": round(float(state.elapsed_s), 4),
            "epoch_time_s": round(float(state.epoch_seconds), 4),
            "train_loss": float(state.train_loss),
            "val_loss": float(state.val_loss),
            "params": int(state.params),
            "device": str(state.device),
            "seed": int(state.seed),
            "config_hash": str(state.config_hash),
            "dataset_hash": str(state.dataset_hash),
            "batch_size": int(state.batch_size),
            "peak_device_mem_mb": json_safe(state.metrics.get("peak_device_mem_mb")),
        })


@register_callback("provenance_ledger")
class ProvenanceLedger(Callback):
    """Write a run-provenance JSON document (P3 replayability).

    Written at ``on_train_begin`` (pre-training identity) and rewritten at
    ``on_train_end`` with the final weights hash and epoch results.
    """

    def __init__(self, out_dir: str | Path | None = None,
                 filename: str = "provenance.json",
                 repo_root: str | Path | None = None):
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.filename = str(filename)
        self.repo_root = (Path(repo_root) if repo_root is not None
                          else Path(__file__).resolve().parents[2])
        self._started = utc_now()

    def _doc(self, state: TrainerState, *, finished: bool) -> dict:
        model = state.model
        param_report: dict = {}
        if model is not None:
            report = getattr(model, "param_report", None)
            if callable(report):
                param_report = json_safe(report())
        doc: dict[str, Any] = {
            "schema": "pes2ts_v5_provenance_v1",
            "experiment_id": state.experiment_id,
            "started_at": self._started,
            "finished_at": utc_now() if finished else None,
            "git_commit": git_commit_sha(self.repo_root),
            "git_dirty": git_is_dirty(self.repo_root),
            "config_hash": state.config_hash,
            "dataset_hash": state.dataset_hash,
            "dataset_version": state.dataset_version,
            "seed": int(state.seed),
            "device": str(state.device),
            "versions": runtime_versions(),
            "param_counts": {
                **param_report,
                "trainable_total": int(state.params),
            },
            "weights_sha256": (weights_sha256(model)
                               if (finished and model is not None) else ""),
            "epochs_run": len(state.history),
            "best_epoch": int(state.best_epoch),
            "best_val_loss": float(state.best_val_loss),
            "extra": json_safe(state.extra),
        }
        return doc

    def _write(self, state: TrainerState, *, finished: bool) -> None:
        if self.out_dir is None:
            return
        write_json(self.out_dir / self.filename, self._doc(state, finished=finished))

    def on_train_begin(self, state: TrainerState) -> None:
        self._write(state, finished=False)

    def on_train_end(self, state: TrainerState) -> None:
        self._write(state, finished=True)


# ---------------------------------------------------------------------------
# Memory guard
# ---------------------------------------------------------------------------

@register_callback("oom_guard")
class OomGuard(Callback):
    """Monitor peak device memory and request a smaller batch size.

    On CUDA the peak allocated memory is compared with ``max_memory_mb`` (or a
    fraction of total device memory); when exceeded, ``state.suggested_batch_size``
    is set to a documented reduced value and the trainer applies it from the next
    epoch on.  On CPU there is no enforceable per-device limit: the guard records
    the process RSS peak for observability and is a graceful no-op.
    """

    def __init__(self, max_memory_mb: float | None = None,
                 peak_fraction: float = 0.9,
                 reduce_factor: float = 0.5,
                 min_batch_size: int = 1):
        self.max_memory_mb = None if max_memory_mb is None else float(max_memory_mb)
        self.peak_fraction = float(peak_fraction)
        self.reduce_factor = float(reduce_factor)
        self.min_batch_size = max(1, int(min_batch_size))

    def _limit_mb(self, state: TrainerState) -> float | None:
        if self.max_memory_mb is not None:
            return self.max_memory_mb
        total = device_total_mb(state.device)
        return None if total is None else self.peak_fraction * total

    def on_batch_end(self, state: TrainerState) -> None:
        peak = peak_memory_mb(state.device)
        if peak is None:
            state.metrics["peak_device_mem_mb"] = None
            state.metrics["oom_guard"] = "unobservable_on_cpu"
            return
        state.metrics["peak_device_mem_mb"] = round(peak, 2)
        limit = self._limit_mb(state)
        if limit is None:
            state.metrics["oom_guard"] = "no_device_limit"
            return
        if peak > limit and state.batch_size > self.min_batch_size:
            reduced = max(self.min_batch_size, int(state.batch_size * self.reduce_factor))
            if reduced < state.batch_size:
                state.suggested_batch_size = reduced
                state.metrics["oom_guard"] = f"reduce_batch_to_{reduced}"
        else:
            state.metrics.setdefault("oom_guard", "ok")


# ---------------------------------------------------------------------------
# Logging / stopping
# ---------------------------------------------------------------------------

@register_callback("log")
class LogCallback(Callback):
    """Append per-epoch metrics to ``train_log.jsonl``."""

    def __init__(self, out_dir: str | Path | None = None,
                 filename: str = "train_log.jsonl"):
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.filename = str(filename)

    def on_epoch_end(self, state: TrainerState) -> None:
        if self.out_dir is None:
            return
        append_jsonl(self.out_dir / self.filename, {
            "timestamp": utc_now(),
            "epoch": int(state.epoch),
            "train_loss": float(state.train_loss),
            "val_loss": float(state.val_loss),
            "best_val_loss": float(state.best_val_loss),
            "improved": bool(state.improved),
            "lr": json_safe(state.extra.get("lr")),
            "elapsed_s": round(float(state.elapsed_s), 4),
            "metrics": json_safe(state.metrics),
        })


@register_callback("early_stop")
class EarlyStop(Callback):
    """Request ``state.should_stop`` after ``patience`` epochs without improvement."""

    def __init__(self, patience: int = 3, min_delta: float = 1e-8):
        self.patience = max(1, int(patience))
        self.min_delta = float(min_delta)
        self._best = float("inf")
        self._stale = 0

    def on_train_begin(self, state: TrainerState) -> None:
        self._best = float("inf")
        self._stale = 0

    def on_epoch_end(self, state: TrainerState) -> None:
        val = float(state.val_loss)
        if not torch.isfinite(torch.tensor(val)):
            return
        if val < self._best - self.min_delta:
            self._best = val
            self._stale = 0
        else:
            self._stale += 1
        if self._stale >= self.patience:
            state.should_stop = True


# ---------------------------------------------------------------------------
# Construction from YAML names
# ---------------------------------------------------------------------------

def build_callbacks(
    names: list[str] | None,
    *,
    out_dir: str | Path | None = None,
    patience: int = 3,
    repo_root: str | Path | None = None,
) -> list[Callback]:
    """Instantiate registered callbacks by name (constructor kwargs by signature)."""
    if not names:
        return []
    resolved: list[str] = []
    for name in names:
        try:
            resolved.append(resolve_callback_name(name))
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
    context: dict[str, Any] = {
        "out_dir": Path(out_dir) if out_dir is not None else None,
        "patience": int(patience),
        "repo_root": repo_root,
    }
    callbacks: list[Callback] = []
    for name in resolved:
        cls = CALLBACK_REGISTRY[name]
        params = inspect.signature(cls.__init__).parameters
        kwargs = {key: context[key] for key in params
                  if key != "self" and key in context}
        callbacks.append(cls(**kwargs))
    return callbacks


__all__ = [
    "CALLBACK_ALIASES",
    "CALLBACK_REGISTRY",
    "Callback",
    "CheckpointBest",
    "CostLedger",
    "EarlyStop",
    "LogCallback",
    "OomGuard",
    "ProvenanceLedger",
    "TrainerState",
    "available_callbacks",
    "build_callbacks",
    "register_callback",
    "resolve_callback_name",
]


# keep the import used (mypy/lint) - time is helpful for callers measuring cost
_ = time
