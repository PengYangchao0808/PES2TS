"""Real-data source: ``idx_v2`` store -> contract-validated sequence batches.

The V5.6 engine reads the frozen ``v53mp_idx_v2`` trajectory index through the
data package (Zone 2):

* ``pes2ts_data.readers.idx_v2.IdxV2Reader`` — memmap reader;
* ``pes2ts_data.readers.idx_v2_dataset.IdxV2Dataset`` — the V5.5-shaped numpy
  dataset view (sequence construction / MOVE-PROCESS-TRUST targets live there;
  this module never re-implements them).

This module only *wraps* that dataset for the training engine: chunk trajectories
into batches, collate them with the existing V5 collate function (the model/loss
transport keeps masks/labels), then build the shared contract batch
(``SequenceBatch`` when data-zone contract v2 is installed, else the documented v1
``StandardBatch`` compat backend) with a per-sample ``provenance_id`` and validate
it via ``assert_fields`` + ``validate_batch(require_core=False)``.

Typed failures: :class:`DataContractUnavailable` (readers/contract not on the
checkout) and :class:`DataSourceUnreachable` (configured index absent/broken).
The smoke split is the curve-hash split from Zone 2, explicitly **not** the frozen
V5.5 ID/OOD protocol.
"""

from __future__ import annotations

from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from pes2ts_v5.data.contracts import (
    DataContractUnavailable,
    DataSourceUnreachable,
    contract_backend,
    provenance_tokens,
    to_sequence_batches,
    validate_sequence_batch,
)
from pes2ts_v5.data.trajectories import collate_samples

#: fields every trajectory batch must provide to the model/registry contract
BASE_REQUIRED_FIELDS = ("frame_feats", "static_feats", "lengths")
META_KEYS = ("traj_idx", "trajectory_id", "reaction_id", "reaction_row",
             "direction", "length")


def load_idx_v2_reader(root: str | Any):
    """Lazily import and construct the data-zone ``IdxV2Reader``.

    Raises :class:`DataContractUnavailable` when ``pes2ts_data.readers.idx_v2`` is
    not importable; :class:`DataSourceUnreachable` when the reader cannot open the
    configured root (missing/incompatible artifacts).
    """
    try:
        from pes2ts_data.readers.idx_v2 import IdxV2Reader
    except ImportError as exc:
        raise DataContractUnavailable(
            "real-data source requested but the data-zone reader is not "
            "available: `pes2ts_data.readers.idx_v2.IdxV2Reader` could not be "
            "imported (Zone-2 readers not on this checkout yet)") from exc
    try:
        return IdxV2Reader(root)
    except Exception as exc:  # reader failures are data-source failures
        raise DataSourceUnreachable(
            f"idx_v2 reader could not open {root!r}: {type(exc).__name__}: {exc}"
        ) from exc


def load_idx_v2_view(reader, *, seed: int = 17,
                     max_curves: Mapping[str, int] | None = None):
    """Lazily construct the data-zone ``IdxV2Dataset`` view over ``reader``."""
    try:
        from pes2ts_data.readers.idx_v2_dataset import IdxV2Dataset
    except ImportError as exc:
        raise DataContractUnavailable(
            "real-data source requested but the data-zone dataset view is not "
            "available: `pes2ts_data.readers.idx_v2_dataset.IdxV2Dataset` could "
            "not be imported") from exc
    try:
        return IdxV2Dataset(reader, seed=int(seed), max_curves=dict(max_curves or {}))
    except Exception as exc:  # dataset view failures are data-source failures
        raise DataSourceUnreachable(
            f"idx_v2 dataset view could not be built: {type(exc).__name__}: {exc}"
        ) from exc


class IdxV2SequenceDataset:
    """Training-engine view over the Zone-2 ``IdxV2Dataset``.

    Delegates sequence/target construction to the data package and adds only the
    batching + shared-contract boundary required by the trainer.
    """

    source_kind = "idx_v2"

    def __init__(self, view, *, required_fields: Sequence[str] = BASE_REQUIRED_FIELDS):
        self.view = view
        self.required_fields = tuple(str(field) for field in required_fields)
        self.contract_backend = contract_backend()
        self.sequence_batch_count = 0
        self.last_sequence_batch: Any = None
        self.last_sequence_batches: list[Any] = []
        self.last_validation_report: dict[str, Any] = {}
        self.last_validation_reports: list[dict[str, Any]] = []
        self.splits: dict[str, list[str]] = {
            split: list(view.reaction_ids_for_split(split))
            for split in ("train", "valid", "id")}
        self._offsets = self._split_offsets()

    def _split_offsets(self) -> dict[str, int]:
        """Map per-split trajectory index -> Zone-2 global trajectory index."""
        trajs = getattr(self.view, "_trajs", None)
        if not isinstance(trajs, Mapping):
            return {}
        offsets: dict[str, int] = {}
        running = 0
        for split, refs in trajs.items():
            offsets[str(split)] = running
            running += len(refs)
        return offsets

    # ------------------------------------------------------------ delegation
    def __getattr__(self, name: str) -> Any:
        return getattr(self.view, name)

    @property
    def reader(self):
        return getattr(self.view, "reader", None)

    @property
    def dataset_hash(self) -> str:
        return str(self.view.dataset_hash)

    @property
    def dataset_version(self) -> str:
        return str(self.view.dataset_version)

    @property
    def counts(self) -> dict[str, int]:
        return dict(self.view.counts)

    @property
    def frame_feat_dim(self) -> int:
        return int(self.view.frame_feat_dim)

    @property
    def static_feat_dim(self) -> int:
        return int(self.view.static_feat_dim)

    @property
    def q_dim(self) -> int:
        return int(self.view.q_dim)

    @property
    def spec(self):
        return self.view.spec

    def reaction_ids_for_split(self, split: str) -> list[str]:
        return list(self.view.reaction_ids_for_split(split))

    def split_counts(self) -> dict[str, int]:
        return dict(self.view.split_counts())

    def manifest_doc(self) -> dict:
        return self.view.reader.manifest_doc()

    def median_ds(self, split: str, position: int) -> float:
        return float(self.view.median_ds(self._global_index(split, position)))

    def reference_q(self, split: str, position: int) -> np.ndarray:
        return self.view.reference_q(self._global_index(split, position))

    def geom_for(self, reaction_row: int):
        return self.view.geom_for(reaction_row)

    # ------------------------------------------------------------- sequences
    def _global_index(self, split: str, position: int) -> int:
        return int(self._offsets.get(split, 0)) + int(position)

    def build_sequence(self, split: str, position: int) -> dict:
        """One V5.5-shaped sequence dict from the Zone-2 dataset view."""
        return self.view.build_sequence(self._global_index(split, position))

    def iter_sequences(self, split: str, *, fraction: float = 1.0,
                       seed: int = 17, max_samples: int | None = None):
        """Yield ``(samples, metas)`` for a split (delegates to Zone 2)."""
        return self.view.iter_sequences(split, fraction=fraction, seed=seed,
                                        max_samples=max_samples)

    # --------------------------------------------------------------- batches
    def _contract_batch(self, batch: dict, metas: list[dict]) -> list[Any]:
        tokens = provenance_tokens(metas, len(metas))
        sequences = to_sequence_batches(batch, provenance_id=tokens)
        reports = [validate_sequence_batch(item, self.required_fields)
                   for item in sequences]
        self.sequence_batch_count += len(sequences)
        self.last_sequence_batches = sequences
        self.last_sequence_batch = sequences[0] if sequences else None
        self.last_validation_reports = reports
        self.last_validation_report = reports[0] if reports else {}
        return sequences

    def trajectory_indices(self, split: str, *, fraction: float = 1.0,
                           seed: int = 17, min_reactions: int = 1) -> list[int]:
        return self.view.trajectory_indices(
            split, fraction=fraction, seed=seed, min_reactions=min_reactions)

    def iter_batches(self, split: str, *, batch_size: int = 4,
                     shuffle: bool = False, seed: int = 17,
                     fraction: float = 1.0, max_batches: int | None = None,
                     drop_last: bool = False) -> Iterator[tuple[dict, list[dict]]]:
        indices = self.trajectory_indices(split, fraction=fraction, seed=seed)
        if shuffle:
            rng = np.random.default_rng(seed)
            indices = [indices[i] for i in rng.permutation(len(indices))]
        for start in range(0, len(indices), batch_size):
            chunk = indices[start:start + batch_size]
            if drop_last and len(chunk) < batch_size:
                break
            samples = [self.build_sequence(split, position) for position in chunk]
            batch = collate_samples(samples)
            # Scalar (q_dim=1) real data: collate yields [B, T]; the shared heads
            # expect [B, T, q_dim].  Add the trailing feature axis explicitly.
            scalar_targets = (
                ("move_target", "float"),
                ("progress_target", "float"),
                ("trust_target", "float"),
            )
            for key, _dtype in scalar_targets:
                value = batch.get(key)
                if hasattr(value, "dim") and value.dim() == 2:
                    batch[key] = value.unsqueeze(-1)
            metas = [{key: sample[key] for key in META_KEYS} for sample in samples]
            self._contract_batch(batch, metas)
            yield batch, metas
            if max_batches is not None and (start // batch_size + 1) >= max_batches:
                break


__all__ = [
    "BASE_REQUIRED_FIELDS",
    "IdxV2SequenceDataset",
    "load_idx_v2_reader",
    "load_idx_v2_view",
]
