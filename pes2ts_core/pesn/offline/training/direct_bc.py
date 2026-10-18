"""Direct BC trainer (WP3, plan v3 §7.1).

Loss ``L_BC = -E_group[log pi_z(a_expert | s)]``: per-sample log-probs sum
over valid atom dims, are averaged within each reaction group first, then
across groups (plan §6.3) — never a flat per-transition mean that would let
long paths dominate short ones. Group-balanced sampling draws reactions,
not transitions. Model selection uses the fit-monitor NLL only.

Device chain (P2/§11.5.2): one ``resolve_device`` entry; model, batches,
and every loss-side tensor live on the resolved device; checkpoints save
CPU-mapped state via ``state_dicts()`` + optimizer/RNG exactly like the
legacy teacher trainer.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Sequence

import numpy as np

from pes2ts_core.pesn.contracts.direct import DEFAULT_REWARD_SPEC, DirectRewardSpec
from pes2ts_core.pesn.features.direct_graph import build_direct_features
from pes2ts_core.pesn.models.egnn import (
    EGNNConfig,
    collate_direct_features,
)
from pes2ts_core.pesn.models.direct_policy import (
    RadialDirectPolicy,
    radial_inverse,
)
from pes2ts_core.pesn.offline.data.direct_dataset import (
    DirectSample,
    DirectTransitionDataset,
)
from pes2ts_core.pesn.offline.training.device import (
    module_summary,
    move_batch,
    resolve_device,
    seed_everything,
    set_cpu_threads,
)
from pes2ts_core.utils.jsonio import write_json

BC_VERSION: Final[str] = "pes2ts_direct_bc_v1.0"


class DirectBCTrainerError(RuntimeError):
    """Typed trainer refusal."""


@dataclass(frozen=True, slots=True)
class DirectBCSettings:
    max_epochs: int = 50
    patience: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    clip: float = 1.0
    seed: int = 42
    batch_reactions: int = 8
    hidden: int = 128
    n_layers: int = 4
    light: bool = False
    action_max: float = 0.1
    device: str = "cuda:0"
    require_cuda: bool = True
    cpu_intra_threads: int = 2
    cpu_inter_threads: int = 1
    eval_batch_reactions: int = 16
    sigma_range: tuple[float, float] = (0.02, 1.0)

    def to_payload(self) -> dict:
        return {
            "max_epochs": self.max_epochs, "patience": self.patience,
            "lr": self.lr, "weight_decay": self.weight_decay,
            "clip": self.clip, "seed": self.seed,
            "batch_reactions": self.batch_reactions,
            "hidden": self.hidden, "n_layers": self.n_layers,
            "light": self.light, "action_max": self.action_max,
            "device": self.device, "require_cuda": self.require_cuda,
            "cpu_threads": [self.cpu_intra_threads, self.cpu_inter_threads],
            "sigma_range": list(self.sigma_range),
        }


def sample_to_feature_inputs(sample: DirectSample,
                             reward_spec: DirectRewardSpec = DEFAULT_REWARD_SPEC):
    return build_direct_features(
        atomic_numbers=sample.atomic_numbers,
        coordinates=sample.coords,
        gradient=sample.gradient,
        task_graph=sample.task_graph,
        delta_e_hartree=sample.delta_e,
        steps_remaining=sample.steps_remaining,
        budget_steps=reward_spec.budget_steps,
        goal_consecutive_hits=float(sample.event_state[0]),
        goal_hits_required=reward_spec.goal_consecutive_hits,
        no_progress_count=float(sample.event_state[1]),
        no_progress_window=reward_spec.no_progress_window,
        best_potential=float(sample.event_state[2]),
    )


class DirectBCTrainer:
    """Group-balanced BC over direct expert transitions."""

    def __init__(
        self,
        *,
        settings: DirectBCSettings,
        train_dataset: DirectTransitionDataset,
        monitor_dataset: DirectTransitionDataset | None,
        run_dir: str | Path,
        config_sha256: str,
        dataset_manifest_sha256: str,
        reward_spec: DirectRewardSpec = DEFAULT_REWARD_SPEC,
    ) -> None:
        self.settings = settings
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config_sha256 = config_sha256
        self.dataset_manifest_sha256 = dataset_manifest_sha256
        self.reward_spec = reward_spec
        self.train_dataset = train_dataset
        self.monitor_dataset = monitor_dataset

        self.torch, self.device = resolve_device(
            device=settings.device, require_cuda=settings.require_cuda)
        set_cpu_threads(intra=settings.cpu_intra_threads,
                        inter=settings.cpu_inter_threads)
        seed_everything(settings.seed)
        from pes2ts_core.pesn.models.egnn import DirectEGNNEncoder
        encoder = DirectEGNNEncoder(EGNNConfig(
            hidden=settings.hidden, n_layers=settings.n_layers,
            light=settings.light))
        self.policy = RadialDirectPolicy(
            encoder=encoder, action_max=settings.action_max,
            sigma_range=settings.sigma_range).to(self.device)
        self.optimizer = self.torch.optim.AdamW(
            self.policy.parameters(), lr=settings.lr,
            weight_decay=settings.weight_decay)
        self.history: list[dict] = []
        self.best_monitor_nll = float("inf")
        self._epochs_without_improvement = 0

    # -- data ---------------------------------------------------------------

    def _reaction_batches(self, reaction_ids: Sequence[str], batch_size: int,
                          epoch: int) -> list[list[str]]:
        rng = np.random.default_rng([self.settings.seed, epoch])
        order = rng.permutation(len(reaction_ids))
        ids = [reaction_ids[i] for i in order]
        return [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

    def _build_batch(self, reaction_ids: Sequence[str], dataset) -> dict:
        torch = self.torch
        features, expert_u, sample_group_index = [], [], []
        atom_sample_index: list[int] = []
        samples_by_reaction = {rid: dataset.get_reaction(rid) for rid in reaction_ids}
        for g_idx, rid in enumerate(reaction_ids):
            for sample in samples_by_reaction[rid]:
                s_idx = len(sample_group_index)
                features.append(sample_to_feature_inputs(sample, self.reward_spec))
                a = torch.as_tensor(sample.action.astype(np.float32))
                z = (a / self.settings.action_max)
                u = radial_inverse(torch, z)
                expert_u.append(u)
                sample_group_index.append(g_idx)
                atom_sample_index.extend([s_idx] * int(sample.atomic_numbers.size))
        batch = collate_direct_features(torch, features)
        batch["expert_u"] = torch.cat(expert_u, dim=0)
        batch["sample_group_index"] = torch.as_tensor(
            np.asarray(sample_group_index, dtype=np.int64))
        batch["atom_sample_index"] = torch.as_tensor(
            np.asarray(atom_sample_index, dtype=np.int64))
        batch["n_groups"] = len(reaction_ids)
        batch["n_samples"] = len(sample_group_index)
        batch["reaction_ids"] = list(reaction_ids)
        return move_batch(batch, self.device)

    # -- training -----------------------------------------------------------

    def compute_loss(self, batch: dict) -> dict:
        torch = self.torch
        out = self.policy(batch)
        lp = self.policy.module.log_prob_z(out["mu"], out["log_sigma"],
                                           batch["expert_u"])
        per_atom = lp["per_atom"]                        # (Ntot,)
        # atom -> sample reduction (per-sample log pi_z sums valid atom dims)
        per_sample = torch.zeros(batch["n_samples"],
                                 device=per_atom.device).index_add_(
            0, batch["atom_sample_index"], per_atom)
        ones = torch.ones_like(per_atom)
        atoms_per_sample = torch.zeros(batch["n_samples"],
                                       device=per_atom.device).index_add_(
            0, batch["atom_sample_index"], ones)
        # sample -> group reduction (mean within reaction group)
        group_sum = torch.zeros(batch["n_groups"],
                                device=per_sample.device).index_add_(
            0, batch["sample_group_index"], per_sample)
        samples_per_group = torch.zeros(batch["n_groups"],
                                        device=per_sample.device).index_add_(
            0, batch["sample_group_index"], torch.ones_like(per_sample))
        group_mean = group_sum / samples_per_group.clamp(min=1)
        nll = -group_mean.mean()
        return {"loss": nll, "group_nll": group_mean.detach(),
                "mean_log_det": lp["log_det"].mean().detach()}

    def train(self) -> dict:
        torch = self.torch
        reaction_ids = self.train_dataset.reaction_ids
        if not reaction_ids:
            raise DirectBCTrainerError("empty train dataset")
        started = time.perf_counter()
        for epoch in range(self.settings.max_epochs):
            self.policy.train_mode(True)
            epoch_losses = []
            for batch_ids in self._reaction_batches(
                    reaction_ids, self.settings.batch_reactions, epoch):
                batch = self._build_batch(batch_ids, self.train_dataset)
                loss_parts = self.compute_loss(batch)
                self.optimizer.zero_grad(set_to_none=True)
                loss_parts["loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.settings.clip)
                self.optimizer.step()
                epoch_losses.append(float(loss_parts["loss"].detach()))
            monitor = self.evaluate(self.monitor_dataset) if self.monitor_dataset else None
            train_nll = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            record = {
                "epoch": epoch,
                "train_group_nll": train_nll,
                "monitor_group_nll": None if monitor is None else monitor["group_nll"],
                "seconds": round(time.perf_counter() - started, 2),
            }
            self.history.append(record)
            self._save_checkpoint(epoch, last=True)
            if monitor is not None:
                if monitor["group_nll"] < self.best_monitor_nll - 1e-6:
                    self.best_monitor_nll = monitor["group_nll"]
                    self._epochs_without_improvement = 0
                    self._save_checkpoint(epoch, best=True)
                else:
                    self._epochs_without_improvement += 1
                if self._epochs_without_improvement >= self.settings.patience:
                    break
        return {"history": self.history, "best_monitor_nll": self.best_monitor_nll,
                "model_summary": module_summary(self.policy.module)}

    def evaluate(self, dataset: DirectTransitionDataset | None) -> dict:
        if dataset is None:
            return {"group_nll": float("nan")}
        torch = self.torch
        self.policy.train_mode(False)
        reaction_ids = dataset.reaction_ids
        group_nlls = []
        with torch.no_grad():
            for start in range(0, len(reaction_ids),
                               self.settings.eval_batch_reactions):
                ids = reaction_ids[start:start + self.settings.eval_batch_reactions]
                batch = self._build_batch(ids, dataset)
                parts = self.compute_loss(batch)
                group_nlls.extend(parts["group_nll"].tolist())
        return {"group_nll": float(np.mean(group_nlls)),
                "n_groups": len(group_nlls)}

    def _save_checkpoint(self, epoch: int, *, best: bool = False,
                         last: bool = False) -> None:
        torch = self.torch
        from pes2ts_core.pesn.offline.training.checkpoints import (
            CHECKPOINT_SCHEMA, save_checkpoint)
        payload = {
            "schema_version": CHECKPOINT_SCHEMA,
            "epoch": epoch,
            "best_metric": self.best_monitor_nll,
            "config_sha256": self.config_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "model_state": {"policy": {
                k: v.detach().to("cpu") for k, v in
                self.policy.module.state_dict().items()}},
            "optimizer_state": self.optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "extra": {
                "trainer_version": BC_VERSION,
                "settings": self.settings.to_payload(),
                "model_summary": module_summary(self.policy.module),
            },
        }
        import torch as _torch
        name = "checkpoint_best.pt" if best else "checkpoint_last.pt"
        _torch.save(payload, self.run_dir / name)

    def write_run_record(self) -> None:
        write_json(self.run_dir / "run_record.json", {
            "schema_version": "pes2ts_training_run_v1",
            "trainer": "direct_bc",
            "trainer_version": BC_VERSION,
            "config_sha256": self.config_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "settings": self.settings.to_payload(),
            "history": self.history,
            "best_monitor_nll": self.best_monitor_nll,
        })


__all__ = [
    "BC_VERSION",
    "DirectBCSettings",
    "DirectBCTrainer",
    "DirectBCTrainerError",
    "sample_to_feature_inputs",
]
