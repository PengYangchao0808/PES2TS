"""Dataset resolution for the V5.6 engine (shared by the runner and the sweep).

One place decides where data comes from, so the train path and the architecture
sweep can never diverge:

* a ``store_manifest`` with a ``manifest.json`` is loaded as a real V5.5 store;
* a ``synthetic_fixture*`` split policy builds the deterministic synthetic
  fixture **at that path** when the manifest is absent (the only fallback);
* any other missing store raises :class:`~pes2ts_v5.data.manifest.V55ContractNotAvailable`
  — no data is ever fabricated;
* optional ``data.frame_feat_dim`` / ``data.static_feat_dim`` are explicit
  contract assertions against the loaded dataset (never magic defaults).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from pes2ts_v5.configs.schema import SYNTHETIC_SPLIT_POLICIES, ExperimentConfig
from pes2ts_v5.data.manifest import V55ContractNotAvailable
from pes2ts_v5.data.trajectories import V5TrajectoryDataset, build_synthetic_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
#: batch keys living on the time axis (used by the ``data.max_frames`` cap)
SEQUENCE_KEYS = (
    "frame_feats", "delta_s", "frame_mask", "move_target", "move_mask",
    "process_label", "process_mask", "process_progress", "progress_mask",
    "trust_label", "trust_mask", "q", "s",
)


class DataDimensionMismatch(RuntimeError):
    """Explicit config data dimensions disagree with the loaded dataset."""

    code = "DATA_DIM_MISMATCH"


def resolve_repo_path(path: str | Path) -> Path:
    target = Path(path)
    return target if target.is_absolute() else (REPO_ROOT / target)


def _truncate_batch(batch: dict, max_frames: int) -> dict:
    out = dict(batch)
    for key in SEQUENCE_KEYS:
        value = out.get(key)
        if torch.is_tensor(value) and value.dim() >= 2:
            out[key] = value[:, :max_frames]
    lengths = out.get("lengths")
    if torch.is_tensor(lengths):
        out["lengths"] = lengths.clamp(max=int(max_frames))
    return out


class CappedDataset:
    """Read-only dataset view applying ``data.limit`` / ``data.max_frames`` caps.

    * ``limit``: cap on reactions used per split (group-level subset fraction, so
      all trajectories of a chosen reaction stay together);
    * ``max_frames``: truncate every sequence to the first N frames (debug cap).
    Both are recorded in the run summary; the underlying dataset is untouched.
    """

    def __init__(self, dataset, *, limit: int | None = None,
                 max_frames: int | None = None):
        self._dataset = dataset
        self.limit = None if limit is None else int(limit)
        self.max_frames = None if max_frames is None else int(max_frames)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dataset, name)

    def _capped_fraction(self, split: str, fraction: float) -> float:
        if self.limit is None:
            return float(fraction)
        n_reactions = len(self._dataset.reaction_ids_for_split(split))
        if n_reactions == 0:
            return float(fraction)
        return min(float(fraction), self.limit / n_reactions)

    def iter_batches(self, split: str, **kwargs: Any):
        kwargs = dict(kwargs)
        kwargs["fraction"] = self._capped_fraction(split, kwargs.get("fraction", 1.0))
        for batch, meta in self._dataset.iter_batches(split, **kwargs):
            if self.max_frames is not None:
                batch = _truncate_batch(batch, self.max_frames)
            yield batch, meta


def _check_declared_dims(cfg: ExperimentConfig, dataset: Any) -> dict[str, int]:
    dims = {
        "frame_feat_dim": int(dataset.frame_feat_dim),
        "static_feat_dim": int(dataset.static_feat_dim),
        "q_dim": int(dataset.q_dim),
    }
    declared = {
        "frame_feat_dim": cfg.data.frame_feat_dim,
        "static_feat_dim": cfg.data.static_feat_dim,
    }
    for key, value in declared.items():
        if value is not None and int(value) != dims[key]:
            raise DataDimensionMismatch(
                f"data.{key}={value} declared in the config but the dataset at "
                f"{cfg.data.store_manifest!r} reports {dims[key]}; fix the explicit "
                "override (it is never used as a magic model default)")
    return dims


def resolve_dataset(cfg: ExperimentConfig) -> tuple[Any, dict]:
    """Load the configured store, or build the documented synthetic fixture."""
    store = resolve_repo_path(cfg.data.store_manifest)
    info: dict[str, Any] = {
        "store_manifest": str(store),
        "split_policy": cfg.data.split_policy,
    }
    if (store / "manifest.json").exists():
        dataset = V5TrajectoryDataset(store)
        info["source"] = "store_manifest"
    elif cfg.is_synthetic():
        build_synthetic_dataset(store, seed=int(cfg.seed))
        dataset = V5TrajectoryDataset(store)
        info["source"] = "synthetic_fixture"
        info["note"] = ("no store manifest at store_manifest; built the "
                        "deterministic synthetic fixture there (synthetic split "
                        "policy)")
    else:
        raise V55ContractNotAvailable(
            f"no dataset manifest at {store} and split_policy="
            f"{cfg.data.split_policy!r} is not one of "
            f"{list(SYNTHETIC_SPLIT_POLICIES)}; refusing to fabricate data — "
            "build the V5.5 store or use a synthetic policy explicitly")
    info["dims"] = _check_declared_dims(cfg, dataset)
    if cfg.data.limit is not None or cfg.data.max_frames is not None:
        info["caps"] = {"limit": cfg.data.limit, "max_frames": cfg.data.max_frames}
        dataset = CappedDataset(dataset, limit=cfg.data.limit,
                                max_frames=cfg.data.max_frames)
    return dataset, info


__all__ = [
    "CappedDataset",
    "DataDimensionMismatch",
    "REPO_ROOT",
    "SEQUENCE_KEYS",
    "resolve_dataset",
    "resolve_repo_path",
]
