"""Unified V5.6 trainer: AdamW, early stop, seed, checkpoint/resume, variable
length trajectories, gradient accumulation, GPU/CPU.

Discipline (plan §3.4):
* hidden state of stateful cores only ever lives **within one trajectory** — the
  trainer never carries state across batches/reactions;
* the same optimizer/early-stopping rule is used for all architectures;
* checkpoint files record full reproducibility identity (config, dataset hash,
  seed, parameter counts, weights checksum).
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from pes2ts_v5.evaluation.single_step import evaluate_model_batches


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def weights_sha256(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, param in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(np.ascontiguousarray(param.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


@dataclass
class TrainConfig:
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

    def to_dict(self) -> dict:
        return asdict(self)


class V5Trainer:
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
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.history: list[dict] = []
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self.start_epoch = 0
        if cfg.cpu_threads != 0:
            torch.set_num_threads(int(cfg.cpu_threads))

    # ------------------------------------------------------------- batches
    def _train_batches(self):
        return self.dataset.iter_batches(
            self.train_split, batch_size=self.cfg.batch_size, shuffle=True,
            seed=self.cfg.seed, fraction=self.cfg.subset_fraction,
            max_batches=self.cfg.max_train_batches, drop_last=False)

    def _val_batches(self):
        return self.dataset.iter_batches(
            self.val_split, batch_size=self.cfg.batch_size, shuffle=False,
            seed=self.cfg.seed, fraction=self.cfg.subset_fraction,
            max_batches=self.cfg.max_val_batches, drop_last=False)

    @staticmethod
    def _to_device(batch: dict, device) -> dict:
        return {k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in batch.items()}

    # --------------------------------------------------------------- epochs
    def train(self) -> dict:
        t0 = time.time()
        stale = 0
        for epoch in range(self.start_epoch, self.cfg.max_epochs):
            self.model.train()
            losses = []
            self.optimizer.zero_grad(set_to_none=True)
            n_batches = 0
            for step, (batch, _meta) in enumerate(self._train_batches()):
                batch = self._to_device(batch, self.device)
                out = self.model(batch)
                parts = self.model.losses(out, batch, self.cfg.loss_weights)
                loss = parts["total"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss at epoch {epoch} step {step}: {float(loss)}")
                (loss / max(self.cfg.grad_accum, 1)).backward()
                n_batches += 1
                if (step + 1) % max(self.cfg.grad_accum, 1) == 0:
                    if self.cfg.grad_clip:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.cfg.grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach().cpu()))
            if n_batches == 0:
                raise RuntimeError(f"no training batches for split {self.train_split!r}")
            val_metrics = self.evaluate()
            val_loss = val_metrics.get("val_loss", float("nan"))
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
            improved = np.isfinite(val_loss) and val_loss < self.best_val_loss - 1e-8
            if improved:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                stale = 0
                self.save_checkpoint("best.pt", epoch + 1)
            else:
                stale += 1
            self.save_checkpoint("last.pt", epoch + 1)
            if stale >= max(self.cfg.patience, 1):
                break
        return {
            "experiment_id": self.experiment_id,
            "epochs_run": len(self.history),
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "train_seconds": round(time.time() - t0, 2),
            "device": str(self.device),
        }

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        cfg = self.cfg
        batches = list(self.dataset.iter_batches(
            self.val_split, batch_size=cfg.batch_size, shuffle=False,
            seed=cfg.seed, fraction=cfg.subset_fraction,
            max_batches=cfg.max_val_batches))
        if not batches:
            return {"val_loss": float("nan"), "move_rmse": float("nan")}
        metrics = evaluate_model_batches(self.model, batches, device=self.device)
        total, n = 0.0, 0
        for batch, _meta in batches:
            batch = self._to_device(batch, self.device)
            out = self.model(batch)
            parts = self.model.losses(out, batch, cfg.loss_weights)
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
            "dataset_hash": getattr(self.dataset, "dataset_hash", ""),
            "dataset_version": getattr(self.dataset, "dataset_version", ""),
            "history": self.history,
        }
        torch.save(payload, path)
        return path

    def load_checkpoint(self, path: str | Path) -> dict:
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.start_epoch = int(payload["epoch"])
        self.best_val_loss = float(payload.get("best_val_loss", float("inf")))
        self.best_epoch = int(payload.get("best_epoch", -1))
        self.history = list(payload.get("history", []))
        return payload
