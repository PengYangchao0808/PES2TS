"""Small-trial training loop with early stop, resume, and run records (B4).

Implements the frozen v1 recipe (G3-P1 §5.1): AdamW (lr 3e-4, weight decay
1e-4, gradient clip 1.0), reaction-equal batches (8 reactions x up to 12
frames), FP32, at most 50 epochs with train_dev patience 8.  Model selection
uses train_dev progress MAE only — sealed subsets are never consulted.  Every
attempt appends to its run record; resume restores model/optimizer/RNG state
and refuses mismatched config or dataset hashes (a new run starts instead).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Sequence

import numpy as np

from pes2ts_core.pesn.models.encoder import SmallGraphEncoder
from pes2ts_core.pesn.models.heads import MultiTaskTeacher, ResearchHeads
from pes2ts_core.pesn.offline.experiments.runs import RunRecord
from pes2ts_core.pesn.offline.training.checkpoints import (
    CheckpointError,
    load_checkpoint,
    save_checkpoint,
)
from pes2ts_core.pesn.offline.training.loader import (
    ReactionData,
    epoch_batch_indices,
    make_batch,
)
from pes2ts_core.pesn.offline.training.losses import (
    DEFAULT_TAU,
    DEFAULT_WEIGHTS,
    compute_teacher_loss,
)

BEST_CHECKPOINT_NAME: Final[str] = "checkpoint_best.pt"
LAST_CHECKPOINT_NAME: Final[str] = "checkpoint_last.pt"
DRIVER_FEAT_DIM_MULTIPLIER: Final[int] = 3


class TrainerError(RuntimeError):
    """Typed trainer failure."""


@dataclass(slots=True)
class TrainSettings:
    batch_reactions: int = 8
    frames_per_reaction: int = 12
    pairs_per_reaction: int = 8
    pair_margin: float = 0.05
    max_epochs: int = 50
    patience: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    clip: float = 1.0
    seed: int = 42
    tau: float = DEFAULT_TAU
    loss_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    hidden: int = 64
    n_layers: int = 3
    n_basis: int = 16
    apply_symmetry: bool = True

    @classmethod
    def from_config(cls, config: dict) -> "TrainSettings":
        train_cfg = config.get("train", {})
        model_cfg = config.get("model", {})
        weights = dict(DEFAULT_WEIGHTS)
        weights.update(train_cfg.get("loss_weights", {}))
        return cls(
            batch_reactions=int(train_cfg.get("batch_reactions", 8)),
            frames_per_reaction=int(train_cfg.get("frames_per_reaction", 12)),
            pairs_per_reaction=int(train_cfg.get("pairs_per_reaction", 8)),
            pair_margin=float(train_cfg.get("pair_margin", 0.05)),
            max_epochs=int(train_cfg.get("max_epochs", 50)),
            patience=int(train_cfg.get("patience", 8)),
            lr=float(train_cfg.get("lr", 3e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
            clip=float(train_cfg.get("clip", 1.0)),
            seed=int(train_cfg.get("seed", config.get("seed", 42))),
            tau=float(train_cfg.get("tau", DEFAULT_TAU)),
            loss_weights=weights,
            hidden=int(model_cfg.get("hidden", 64)),
            n_layers=int(model_cfg.get("n_layers", 3)),
            n_basis=int(model_cfg.get("n_basis", 16)),
            apply_symmetry=bool(train_cfg.get("apply_symmetry", True)),
        )


def build_teacher_model(settings: TrainSettings) -> MultiTaskTeacher:
    encoder = SmallGraphEncoder(
        hidden=settings.hidden,
        n_layers=settings.n_layers,
        n_basis=settings.n_basis,
    )
    heads = ResearchHeads(
        feature_dim=DRIVER_FEAT_DIM_MULTIPLIER * settings.hidden + 1,
        hidden=settings.hidden,
    )
    return MultiTaskTeacher(encoder, heads)


class Trainer:
    """One training attempt bound to a run record and a run directory.

    ``run_record`` may be ``None`` for evaluation/export-only use, where no
    journal writes happen.
    """

    def __init__(
        self,
        *,
        settings: TrainSettings,
        run_record: RunRecord | None,
        config_sha256: str,
        dataset_manifest_sha256: str,
    ) -> None:
        self.settings = settings
        self.run_record = run_record
        self.config_sha256 = config_sha256
        self.dataset_manifest_sha256 = dataset_manifest_sha256
        self.model = build_teacher_model(settings)
        self.torch = self.model.torch
        self.optimizer = self.torch.optim.AdamW(
            self.model.parameters(),
            lr=settings.lr,
            weight_decay=settings.weight_decay,
        )
        self.torch.manual_seed(settings.seed)
        self.best_dev_mae = float("inf")
        self.best_epoch = -1
        self._epochs_without_improvement = 0

    # -- training ---------------------------------------------------------
    def train(
        self,
        train_reactions: Sequence[ReactionData],
        dev_reactions: Sequence[ReactionData],
        *,
        resume_from: Path | None = None,
    ) -> dict[str, object]:
        start_epoch = 0
        if resume_from is not None:
            payload = self._resume(resume_from)
            start_epoch = int(payload["epoch"])
            self.run_record.append_event("resumed", checkpoint=str(resume_from), epoch=start_epoch)
        if not train_reactions:
            raise TrainerError("no train reactions with valid labels")
        self.run_record.append_event("train_start", n_train=len(train_reactions), n_dev=len(dev_reactions))

        rng_master = np.random.default_rng(self.settings.seed)
        for epoch in range(start_epoch, self.settings.max_epochs):
            epoch_started = time.perf_counter()
            self.model.train_mode(True)
            parts_sum = {}
            index_chunks = epoch_batch_indices(
                len(train_reactions),
                batch_reactions=self.settings.batch_reactions,
                seed=self.settings.seed,
                epoch=epoch,
            )
            n_samples = 0
            for indices in index_chunks:
                chunk = [train_reactions[int(i)] for i in indices]
                batch = make_batch(
                    chunk,
                    frames_per_reaction=self.settings.frames_per_reaction,
                    pairs_per_reaction=self.settings.pairs_per_reaction,
                    pair_margin=self.settings.pair_margin,
                    rng=rng_master,
                )
                if not batch.samples:
                    continue
                self.optimizer.zero_grad()
                parts = compute_teacher_loss(
                    self.model,
                    batch,
                    weights=self.settings.loss_weights,
                    tau=self.settings.tau,
                    apply_symmetry=self.settings.apply_symmetry,
                )
                parts.total.backward()
                self.torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.settings.clip
                )
                self.optimizer.step()
                n_samples += batch.size
                for key in ("s", "d", "rank", "direction", "activity", "symmetry"):
                    value = float(getattr(parts, key).item())
                    parts_sum[key] = parts_sum.get(key, 0.0) + value
            n_batches = max(1, len(index_chunks))
            epoch_losses = {f"loss_{k}": round(v / n_batches, 6) for k, v in parts_sum.items()}

            dev_metrics = self.evaluate(dev_reactions) if dev_reactions else {"mae_s": float("nan")}
            dev_mae = float(dev_metrics.get("mae_s", float("nan")))
            epoch_seconds = round(time.perf_counter() - epoch_started, 2)

            self.run_record.add_epoch(
                {
                    "epoch": epoch,
                    "n_samples": n_samples,
                    "seconds": epoch_seconds,
                    **epoch_losses,
                    "dev_mae_s": None if np.isnan(dev_mae) else round(dev_mae, 6),
                    "dev_metrics": dev_metrics,
                }
            )
            self.run_record.record_resources(samples_per_second=round(n_samples / max(epoch_seconds, 1e-6), 2))

            if not np.isnan(dev_mae) and dev_mae < self.best_dev_mae:
                self.best_dev_mae = dev_mae
                self.best_epoch = epoch
                self._epochs_without_improvement = 0
                save_checkpoint(
                    self.run_record.run_dir / BEST_CHECKPOINT_NAME,
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch + 1,
                    best_metric=dev_mae,
                    config_sha256=self.config_sha256,
                    dataset_manifest_sha256=self.dataset_manifest_sha256,
                )
            else:
                self._epochs_without_improvement += 1

            save_checkpoint(
                self.run_record.run_dir / LAST_CHECKPOINT_NAME,
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch + 1,
                best_metric=self.best_dev_mae,
                config_sha256=self.config_sha256,
                dataset_manifest_sha256=self.dataset_manifest_sha256,
            )
            if self._epochs_without_improvement >= self.settings.patience:
                self.run_record.append_event(
                    "early_stop", epoch=epoch, best_epoch=self.best_epoch
                )
                break

        self.run_record.record_outputs(
            best_checkpoint=str(self.run_record.run_dir / BEST_CHECKPOINT_NAME),
            last_checkpoint=str(self.run_record.run_dir / LAST_CHECKPOINT_NAME),
        )
        return {
            "best_epoch": self.best_epoch,
            "best_dev_mae_s": None if np.isinf(self.best_dev_mae) else round(self.best_dev_mae, 6),
            "n_parameters": self.model.n_parameters(),
        }

    def _resume(self, checkpoint_path: Path) -> dict[str, object]:
        try:
            return load_checkpoint(
                checkpoint_path,
                model=self.model,
                optimizer=self.optimizer,
                expected_config_sha256=self.config_sha256,
                expected_dataset_sha256=self.dataset_manifest_sha256,
            )
        except CheckpointError as exc:
            self.run_record.record_failure("resume", str(exc))
            raise TrainerError(f"resume failed: {exc}") from exc

    # -- evaluation-side forward -----------------------------------------
    def evaluate(self, reactions: Sequence[ReactionData]) -> dict[str, object]:
        """Deterministic forward metrics over full label-valid frames."""
        from pes2ts_core.pesn.offline.evaluation.metrics import (
            activity_metrics,
            direction_metrics,
            pair_concordance,
            progress_metrics,
        )

        self.model.train_mode(False)
        torch = self.torch
        per_reaction: dict[str, dict[str, list[float]]] = {}
        index_chunks = epoch_batch_indices(
            len(reactions),
            batch_reactions=self.settings.batch_reactions,
            seed=0,
            epoch=0,
            shuffle=False,
        )
        rng = np.random.default_rng(0)
        for indices in index_chunks:
            chunk = [reactions[int(i)] for i in indices]
            batch = make_batch(
                chunk,
                frames_per_reaction=self.settings.frames_per_reaction,
                pairs_per_reaction=self.settings.pairs_per_reaction,
                pair_margin=self.settings.pair_margin,
                rng=rng,
            )
            if not batch.samples:
                continue
            with torch.no_grad():
                outputs = self.model.forward([s.features for s in batch.samples])
            s_hat = outputs["signed_progress"].cpu().numpy()
            counts = [int(c) for c in outputs["driver_counts"].cpu().numpy()]
            u_flat = outputs["direction_components"].cpu().numpy()
            act_logits_flat = outputs["activity_logits"].cpu().numpy()
            start = 0
            for local_id, sample in enumerate(batch.samples):
                count = counts[local_id]
                end = start + count
                store = per_reaction.setdefault(sample.reaction_id, {"s": [], "s_hat": [], "d": [], "cos": [], "act_y": [], "act_p": [], "scores": []})
                if sample.valid:
                    store["s"].append(sample.s)
                    store["s_hat"].append(float(s_hat[local_id]))
                    store["d"].append(sample.d)
                    store["scores"].append(float(outputs["path_score"][local_id].item()))
                if sample.dir_valid and count > 0:
                    segment = u_flat[start:end]
                    norm = float(np.linalg.norm(segment))
                    if norm > 1e-6:
                        t_hat = segment / norm
                        t_star = sample.direction
                        store["cos"].append(
                            float(np.dot(t_hat, t_star) / (np.linalg.norm(t_star) + 1e-12))
                        )
                if sample.act_valid and count > 0:
                    store["act_y"].extend(sample.activity.tolist())
                    store["act_p"].extend(
                        (1.0 / (1.0 + np.exp(-act_logits_flat[start:end]))).tolist()
                    )
                start = end

        progress = progress_metrics(per_reaction)
        direction = direction_metrics(per_reaction)
        activity = activity_metrics(per_reaction)
        concordance = pair_concordance(per_reaction, self.settings.pair_margin)
        return {**progress, **direction, **activity, **concordance}


__all__ = [
    "BEST_CHECKPOINT_NAME",
    "LAST_CHECKPOINT_NAME",
    "Trainer",
    "TrainerError",
    "TrainSettings",
    "build_teacher_model",
]
