"""Dataset resolution for the V5.6 engine (shared by the runner and the sweep).

One place decides where data comes from, so train/sweep/eval never diverge:

* ``data.store_manifest`` starting with ``idx_v2:`` -> **real data**: the frozen
  ``v53mp_idx_v2`` index is read through the data package
  (``pes2ts_data.readers.idx_v2.IdxV2Reader``) and the loader emits
  contract-validated ``SequenceBatch`` batches with per-sample ``provenance_id``;
* any other value -> the V5.5-shaped store directory: a ``manifest.json`` is
  loaded as a real store; with a ``synthetic_fixture*`` split policy the
  deterministic synthetic fixture is built at that path (the only fallback);
* any other missing store raises ``V55ContractNotAvailable`` — no data is ever
  fabricated;
* optional ``data.frame_feat_dim`` / ``data.static_feat_dim`` are explicit
  contract assertions against the loaded dataset (never magic defaults).

Typed failures (stable ``code``):
``DATA_SOURCE_UNREACHABLE`` (configured real source absent on this host) and
``DATA_CONTRACT_UNAVAILABLE`` (data-zone readers/contract not on the checkout).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from pes2ts_v5.configs.schema import SYNTHETIC_SPLIT_POLICIES, ExperimentConfig
from pes2ts_v5.data.contracts import IDX_V2_PREFIX, DataSourceUnreachable
from pes2ts_v5.data.manifest import V55ContractNotAvailable
from pes2ts_v5.data.trajectories import V5TrajectoryDataset, build_synthetic_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
#: batch keys living on the time axis (used by the ``data.max_frames`` cap)
SEQUENCE_KEYS = (
    "frame_feats", "delta_s", "frame_mask", "move_target", "move_mask",
    "process_label", "process_mask", "process_progress", "progress_mask",
    "trust_label", "trust_mask", "q", "s",
)
SOURCE_STORE = "store_manifest"
SOURCE_SYNTHETIC = "synthetic_fixture"
SOURCE_IDX_V2 = "idx_v2"


class DataDimensionMismatch(RuntimeError):
    """Explicit config data dimensions disagree with the loaded dataset."""

    code = "DATA_DIM_MISMATCH"


def resolve_repo_path(path: str | Path) -> Path:
    target = Path(path)
    return target if target.is_absolute() else (REPO_ROOT / target)


def parse_store_spec(store_manifest: str) -> tuple[str, str]:
    """Split ``idx_v2:<path>`` / ``<path>`` into ``(source_kind, path)``."""
    text = str(store_manifest).strip()
    lowered = text.lower()
    if lowered.startswith(IDX_V2_PREFIX):
        return SOURCE_IDX_V2, text[len(IDX_V2_PREFIX):].strip()
    return SOURCE_STORE, text


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


def _resolve_idx_v2(cfg: ExperimentConfig, path: str) -> tuple[Any, dict]:
    """Real-data ``idx_v2:<path>`` source through the data-zone reader."""
    root = resolve_repo_path(path)
    info: dict[str, Any] = {
        "store_manifest": str(cfg.data.store_manifest),
        "split_policy": cfg.data.split_policy,
        "source": SOURCE_IDX_V2,
        "idx_v2_root": str(root),
    }
    if not str(path):
        raise DataSourceUnreachable(
            "data.store_manifest uses the 'idx_v2:' prefix but has no path")
    if not root.exists():
        raise DataSourceUnreachable(
            f"idx_v2 source {root} does not exist on this host; set "
            "data.store_manifest to an existing idx_v2 directory (or use the "
            "synthetic fixture policy). No data is fabricated.")
    if not (root / "curves.json").exists():
        raise DataSourceUnreachable(
            f"idx_v2 source {root} has no curves.json; not a readable v53mp_idx_v2 "
            "index directory")
    from pes2ts_v5.data.idx_v2_source import (
        IdxV2SequenceDataset,
        load_idx_v2_reader,
        load_idx_v2_view,
    )

    reader = load_idx_v2_reader(root)  # DataContractUnavailable if readers absent
    view = load_idx_v2_view(reader, seed=int(cfg.seed),
                            max_curves=_idx_v2_max_curves(cfg))
    dataset = IdxV2SequenceDataset(view)
    info["dataset_hash"] = str(dataset.dataset_hash)
    info["dataset_version"] = str(dataset.dataset_version)
    info["counts"] = dict(dataset.counts)
    info["split_counts"] = dataset.split_counts()
    info["contract_backend"] = str(dataset.contract_backend)
    try:
        info["idx_v2_manifest"] = reader.manifest_doc()
    except Exception:  # optional provenance detail; not a source failure
        info["idx_v2_manifest"] = {"note": "reader.manifest_doc() unavailable"}
    return dataset, info


def _idx_v2_max_curves(cfg: ExperimentConfig) -> dict[str, int] | None:
    if cfg.data.limit is None:
        return None
    limit = int(cfg.data.limit)
    return {"train": limit, "valid": limit, "id": limit}


def resolve_dataset(cfg: ExperimentConfig) -> tuple[Any, dict]:
    """Load the configured source, or build the documented synthetic fixture."""
    kind, path = parse_store_spec(cfg.data.store_manifest)
    if kind == SOURCE_IDX_V2:
        dataset, info = _resolve_idx_v2(cfg, path)
        if cfg.data.max_frames is not None:
            info["caps"] = {"max_frames": cfg.data.max_frames}
            dataset = CappedDataset(dataset, max_frames=cfg.data.max_frames)
        info["dims"] = _check_declared_dims(cfg, dataset)
        return dataset, info

    store = resolve_repo_path(path)
    info: dict[str, Any] = {
        "store_manifest": str(store),
        "split_policy": cfg.data.split_policy,
    }
    if (store / "manifest.json").exists():
        dataset = V5TrajectoryDataset(store)
        info["source"] = SOURCE_STORE
    elif cfg.is_synthetic():
        build_synthetic_dataset(store, seed=int(cfg.seed))
        dataset = V5TrajectoryDataset(store)
        info["source"] = SOURCE_SYNTHETIC
        info["note"] = ("no store manifest at store_manifest; built the "
                        "deterministic synthetic fixture there (synthetic split "
                        "policy)")
    else:
        raise V55ContractNotAvailable(
            f"no dataset manifest at {store} and split_policy="
            f"{cfg.data.split_policy!r} is not one of "
            f"{list(SYNTHETIC_SPLIT_POLICIES)}; refusing to fabricate data — "
            "build the V5.5 store or use a synthetic policy explicitly")
    if cfg.data.limit is not None or cfg.data.max_frames is not None:
        info["caps"] = {"limit": cfg.data.limit, "max_frames": cfg.data.max_frames}
        dataset = CappedDataset(dataset, limit=cfg.data.limit,
                                max_frames=cfg.data.max_frames)
    info["dims"] = _check_declared_dims(cfg, dataset)
    return dataset, info


__all__ = [
    "CappedDataset",
    "DataDimensionMismatch",
    "REPO_ROOT",
    "SEQUENCE_KEYS",
    "SOURCE_IDX_V2",
    "SOURCE_STORE",
    "SOURCE_SYNTHETIC",
    "parse_store_spec",
    "resolve_dataset",
    "resolve_repo_path",
]
