"""Spawn-safe direct training batch pipeline (WP-INFRA1, plan v3 §11.5.4).

Top-level, picklable worker/collate functions only (spawn start method);
prefetch/persistent options are passed only when ``workers > 0``; workers
return CPU tensors and the main process moves them (pin-memory + async
only when CUDA targets exist). Sampling is group-balanced by construction:
the sampler yields reaction ids (groups), not transitions — matching the
BC/IQL empirical objectives (§6.3). Per-worker seeds derive from
``(base_seed, worker_id, epoch)`` so worker completion order never
determines randomness (§11.6).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from pes2ts_core.pesn.contracts.direct import DEFAULT_REWARD_SPEC, DirectRewardSpec
from pes2ts_core.pesn.models import require_torch

LOADER_VERSION: Final[str] = "pes2ts_direct_loader_v1.0"


class DirectLoaderError(RuntimeError):
    """Typed loader misuse."""


@dataclass(frozen=True, slots=True)
class LoaderSettings:
    workers: int = 2
    start_method: str = "spawn"
    threads_per_worker: int = 1
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    batch_reactions: int = 8
    seed: int = 42


def worker_init(worker_id: int, base_seed: int) -> None:
    """Top-level spawn-safe worker initializer."""
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        import os
        os.environ.setdefault(key, "1")
    np.random.seed((base_seed + worker_id) % 2**31)


def collate_feature_dicts(batch: list) -> dict:
    """Concatenate prebuilt feature dicts into one batch (top-level, picklable)."""
    import torch
    from pes2ts_core.pesn.models.egnn import collate_direct_features
    features = [item["features"] for item in batch]
    out = collate_direct_features(torch, features)
    out["actions"] = torch.stack([item["action"] for item in batch], dim=0)
    out["sample_group_index"] = torch.as_tensor(
        np.asarray([item["group_index"] for item in batch], dtype=np.int64))
    return out


def _sample_item(sample, group_index: int, reward_spec: DirectRewardSpec) -> dict:
    import torch
    from pes2ts_core.pesn.offline.training.direct_bc import sample_to_feature_inputs
    return {
        "features": sample_to_feature_inputs(sample, reward_spec),
        "action": torch.as_tensor(sample.action.astype(np.float32)),
        "group_index": int(group_index),
    }


@dataclass(frozen=True, slots=True)
class _WorkerInit:
    """Top-level picklable worker initializer (spawn-safe; no lambdas)."""

    base_seed: int

    def __call__(self, worker_id: int) -> None:
        worker_init(worker_id, self.base_seed)


class _FlatSampleSequence:
    """Sequence over (reaction, transition) pairs for torch DataLoader."""

    def __init__(self, dataset, reaction_ids, reward_spec: DirectRewardSpec):
        self._dataset = dataset
        self._reward_spec = reward_spec
        self._index: list[tuple[str, int, int]] = []
        for g_idx, rid in enumerate(reaction_ids):
            for i in range(len(dataset.get_reaction(rid))):
                self._index.append((rid, i, g_idx))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        rid, i, g_idx = self._index[idx]
        sample = self._dataset.get_reaction(rid)[i]
        return _sample_item(sample, g_idx, self._reward_spec)


class ReactionBatchSampler:
    """Yields flattened sample index batches grouped by reaction batches."""

    def __init__(self, dataset, reaction_ids, *, batch_reactions: int, seed: int):
        self.reaction_ids = list(reaction_ids)
        self.batch_reactions = int(batch_reactions)
        self.seed = int(seed)
        self.per_reaction = {
            rid: len(dataset.get_reaction(rid)) for rid in self.reaction_ids}

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(self.reaction_ids))
        cursor = 0
        offsets = np.cumsum([0] + [self.per_reaction[rid] for rid in self.reaction_ids])
        while cursor < len(order):
            chunk = order[cursor:cursor + self.batch_reactions]
            cursor += self.batch_reactions
            idxs: list[int] = []
            for pos in chunk:
                rid = self.reaction_ids[int(pos)]
                base = int(offsets[int(pos)])
                idxs.extend(range(base, base + self.per_reaction[rid]))
            yield idxs

    def __len__(self) -> int:
        return max(1, -(-len(self.reaction_ids) // self.batch_reactions))


def build_direct_dataloader(dataset, reaction_ids, *,
                            settings: LoaderSettings,
                            reward_spec: DirectRewardSpec = DEFAULT_REWARD_SPEC,
                            device_is_cuda: bool = True,
                            epoch: int = 0):
    """Build a torch DataLoader with the plan §11.5.4 option discipline."""
    torch = require_torch()
    if settings.workers < 0:
        raise DirectLoaderError("workers must be >= 0")
    sequence = _FlatSampleSequence(dataset, reaction_ids, reward_spec)
    sampler = ReactionBatchSampler(
        dataset, reaction_ids, batch_reactions=settings.batch_reactions,
        seed=settings.seed + epoch)
    kwargs = {
        "dataset": sequence,
        "batch_sampler": sampler,
        "collate_fn": collate_feature_dicts,
        "num_workers": settings.workers,
    }
    if settings.workers > 0:
        kwargs["prefetch_factor"] = settings.prefetch_factor
        kwargs["persistent_workers"] = settings.persistent_workers
        kwargs["multiprocessing_context"] = settings.start_method
        kwargs["worker_init_fn"] = _WorkerInit(settings.seed + epoch)
    # pin only useful when the consumer moves to an accelerator
    kwargs["pin_memory"] = bool(settings.pin_memory and device_is_cuda
                                and settings.workers > 0)
    return torch.utils.data.DataLoader(**kwargs)


__all__ = [
    "LOADER_VERSION",
    "DirectLoaderError",
    "LoaderSettings",
    "ReactionBatchSampler",
    "build_direct_dataloader",
    "collate_feature_dicts",
    "worker_init",
]
