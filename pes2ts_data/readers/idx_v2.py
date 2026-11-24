"""Memmap reader over the frozen ``v53mp_idx_v2`` trajectory index (Zone 2).

Ported from ``pes2ts_v5/data/idx_v2.py`` (moved into Zone-2 ownership during the
repo restructuring) and made torch-free: this module keeps only the read-only
memmap index (:class:`IdxV2Reader`) and the staging-chunk reader.  The
V5.5-shaped dataset view that used torch collate lives in
:mod:`pes2ts_data.readers.idx_v2_dataset`.

Layout (``/pes2ts_data/v53mp_50k_data/idx_v2``)::

    curves.json                  # schema v53mp_idx_v2: binding/counts/per-curve
                                 # metadata + file hashes
    samples/*.npy                # per-decision-sample columns (14,685,162)
    label_curves/*.npy           # per-node: q_canonical, arc_s, node_order, offsets
    labels_ep/*.npy              # per-sample: ts_row, start_row, terminal_row,
                                 # d_key_ts_ep, degenerate_curve, ...
    labels_v52/*.npy             # per-sample: q progress, q_mask
    curve_node_energies.npy      # per-node energies (G1 cached-QC only)
    staging_chunks/chunk_*.npz   # real XYZ geometry shards

Everything is memory-mapped (``mode="r"``); nothing is written or loaded
eagerly, and no torch/pandas import is required.  Refusals are
:class:`IdxV2Error` (a :class:`~._common.ReaderError`) with a stable ``.code``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Final

import numpy as np

from ._common import ReaderError

__all__ = [
    "DEFAULT_IDX_ROOT",
    "DEFAULT_STAGING_ROOT",
    "SCHEMA",
    "STATUS_CONFLICT",
    "STATUS_MASKED",
    "STATUS_OK",
    "STATUS_TERMINAL",
    "IdxV2Error",
    "IdxV2Reader",
    "idx_v2_available",
    "read_staging_chunk",
]

DEFAULT_IDX_ROOT: Final[str] = "/pes2ts_data/v53mp_50k_data/idx_v2"
DEFAULT_STAGING_ROOT: Final[str] = "/pes2ts_data/v53mp_50k_data/staging_chunks"
SCHEMA: Final[str] = "v53mp_idx_v2"

#: Label-status codes from ``curves.json["label_status_table"]``.
STATUS_CONFLICT, STATUS_MASKED, STATUS_OK, STATUS_TERMINAL = 0, 1, 2, 3

#: Split names of the frozen dataset lineage (V5.5 contract names).
SPLIT_NAMES: Final[tuple[str, ...]] = (
    "train", "valid", "id", "scaffold_ood", "family_ood", "size_ood")


class IdxV2Error(ReaderError):
    """Typed refusal for missing/incompatible ``idx_v2`` artifacts."""


def idx_v2_available(root: str | Path | None = None) -> bool:
    """Return ``True`` when ``curves.json`` exists under the idx root."""
    base = Path(root or os.environ.get("PES2TS_V5_IDX_V2_DIR", DEFAULT_IDX_ROOT))
    return (base / "curves.json").exists()


def read_staging_chunk(path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """Read one real-geometry staging chunk: ``{reaction_id: {coordinates, ...}}``."""
    data = np.load(path, allow_pickle=False)
    out: dict[str, dict[str, np.ndarray]] = {}
    for key in data.files:
        reaction_id, _, field = key.partition("::")
        out.setdefault(reaction_id, {})[field] = np.asarray(data[key])
    return out


class IdxV2Reader:
    """Memmap reader over ``v53mp_idx_v2`` (read-only).

    Parameters
    ----------
    root:
        Index root.  Defaults to ``$PES2TS_V5_IDX_V2_DIR`` /
        :data:`DEFAULT_IDX_ROOT` (the WSL datastore path).

    Raises
    ------
    IdxV2Error
        ``IDX_V2_MISSING`` when ``curves.json`` is absent, ``IDX_V2_SCHEMA`` on
        an unsupported schema and ``IDX_V2_ARRAY`` when a required array is
        missing.
    """

    #: Array groups and members loaded (mmap) by :meth:`_load_arrays`.
    ARRAY_GROUPS: Final[dict[str, tuple[str, ...]]] = {
        "samples": ("curve_idx", "intent_idx", "node_index", "label_status",
                    "fold", "d0", "action_off", "action_len", "sample_weight",
                    "label_weight", "curve_weight"),
        "label_curves": ("q_canonical", "arc_s", "node_order", "offsets"),
        "labels_ep": ("ts_row", "start_row", "terminal_row", "d_key_ts_ep",
                      "degenerate_curve", "ep_label", "geometry_legal"),
        "labels_v52": ("q", "q_mask"),
    }

    #: Arrays without which the reader cannot serve a curve/sample.
    REQUIRED_ARRAYS: Final[tuple[str, ...]] = (
        "label_curves/q_canonical", "label_curves/arc_s", "label_curves/offsets",
        "samples/curve_idx", "labels_ep/ts_row", "labels_ep/d_key_ts_ep")

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or os.environ.get("PES2TS_V5_IDX_V2_DIR",
                                                DEFAULT_IDX_ROOT))
        curves_path = self.root / "curves.json"
        if not curves_path.exists():
            raise IdxV2Error(
                "IDX_V2_MISSING",
                f"idx_v2 not found at {self.root} (set PES2TS_V5_IDX_V2_DIR); "
                "real-data smoke requires the WSL 50k index, the synthetic "
                "fixture is used otherwise")
        try:
            self.curves_doc = json.loads(curves_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise IdxV2Error("IDX_V2_SCHEMA", f"{curves_path}: {exc}") from exc
        if self.curves_doc.get("schema") != SCHEMA:
            raise IdxV2Error("IDX_V2_SCHEMA",
                             f"unexpected idx schema {self.curves_doc.get('schema')!r}")
        self.binding: dict = self.curves_doc.get("binding", {})
        self.curve_ids: list[str] = list(self.curves_doc.get("curve_ids", []))
        self.curves: list[dict] = list(self.curves_doc.get("curves", []))
        self.label_status_table: list[str] = list(
            self.curves_doc.get("label_status_table", []))
        self._arrays: dict[str, np.ndarray] = {}
        self._load_arrays()

    # ------------------------------------------------------------------ load
    def _load_arrays(self) -> None:
        for group, names in self.ARRAY_GROUPS.items():
            for name in names:
                path = self.root / group / f"{name}.npy"
                if path.exists():
                    self._arrays[f"{group}/{name}"] = np.load(path, mmap_mode="r")
        energies = self.root / "curve_node_energies.npy"
        if energies.exists():
            self._arrays["curve_node_energies"] = np.load(energies, mmap_mode="r")
        for required in self.REQUIRED_ARRAYS:
            if required not in self._arrays:
                raise IdxV2Error("IDX_V2_ARRAY",
                                 f"missing required idx_v2 array: {required}")

    def __getitem__(self, name: str) -> np.ndarray:
        if name not in self._arrays:
            raise IdxV2Error("IDX_V2_ARRAY",
                             f"array {name!r} not available in idx_v2")
        return self._arrays[name]

    def array_names(self) -> tuple[str, ...]:
        """Names of every loaded (memmapped) array group member."""
        return tuple(sorted(self._arrays))

    # ------------------------------------------------------------ properties
    @property
    def n_curves(self) -> int:
        return int(self.binding.get("n_curves", len(self.curves)))

    @property
    def n_intents(self) -> int:
        return int(self.binding.get("n_intents", 0))

    @property
    def n_samples(self) -> int:
        return int(self.binding.get("n_samples",
                                    self["samples/curve_idx"].shape[0]))

    @property
    def n_nodes(self) -> int:
        return int(self["label_curves/q_canonical"].shape[0])

    @property
    def data_manifest_sha256(self) -> str | None:
        return self.binding.get("data_manifest_sha256")

    @property
    def dataset_hash(self) -> str:
        """Hash recorded from the frozen file registry in ``curves.json``."""
        files = self.curves_doc.get("files") or {}
        lines = []
        for name in sorted(files):
            record = files[name]
            sha = record.get("sha256") if isinstance(record, dict) else str(record)
            lines.append(f"{name}:{sha}")
        payload = "\n".join(lines) + ("\n" + str(self.binding.get("data_manifest_sha256")))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def dataset_version(self) -> str:
        created = str(self.curves_doc.get("created_at", "unknown"))[:10]
        commit = str(self.curves_doc.get("builder_git_commit", ""))[:8]
        return f"idx_v2-{created}-{commit}"

    def counts(self) -> dict[str, int]:
        return {"n_curves": self.n_curves, "n_intents": self.n_intents,
                "n_frames": self.n_nodes, "n_samples": self.n_samples}

    def manifest_doc(self) -> dict:
        """JSON-friendly provenance record of this index (no array payloads)."""
        return {
            "schema": SCHEMA,
            "source": str(self.root),
            "binding": self.binding,
            "builder_git_commit": self.curves_doc.get("builder_git_commit"),
            "created_at": self.curves_doc.get("created_at"),
            "dataset_version": self.dataset_version,
            "dataset_hash": self.dataset_hash,
            "counts": self.counts(),
            "note": "idx_v2 is an index over the frozen 50k tables; frame-level "
                    "counts are nodes, decision-level counts are samples",
        }

    # -------------------------------------------------------------- per curve
    def curve_meta(self, curve_idx: int) -> dict:
        return self.curves[int(curve_idx)]

    def curve_node_arrays(self, curve_idx: int) -> dict[str, np.ndarray]:
        """Return per-node arrays in node-index order for one curve."""
        offsets = self["label_curves/offsets"]
        lo, hi = int(offsets[curve_idx]), int(offsets[curve_idx + 1])
        order = self["label_curves/node_order"][lo:hi]
        size = hi - lo
        row_for_node = np.empty(size, dtype=np.int64)
        row_for_node[np.asarray(order, dtype=np.int64)] = np.arange(size)
        out = {
            "q": np.asarray(self["label_curves/q_canonical"][lo:hi],
                            dtype=np.float32)[row_for_node],
            "s": np.asarray(self["label_curves/arc_s"][lo:hi],
                            dtype=np.float32)[row_for_node],
        }
        if "curve_node_energies" in self._arrays:
            out["energy"] = np.asarray(
                self._arrays["curve_node_energies"][lo:hi],
                dtype=np.float32)[row_for_node]
        return out

    def sample_rows(self, curve_idx: int) -> tuple[int, int]:
        """Row range ``[lo, hi)`` of the samples belonging to one curve."""
        curve_index = self["samples/curve_idx"]
        return (int(np.searchsorted(curve_index, curve_idx, side="left")),
                int(np.searchsorted(curve_index, curve_idx, side="right")))

    def split_curves(self, split: str, *, seed: int = 17,
                     max_curves: int | None = None) -> list[int]:
        """Curve-level hash split for smoke runs (NOT the frozen V5.5 OOD split)."""
        hashed = (np.arange(self.n_curves, dtype=np.int64) * 2654435761
                  + int(seed)) % 100
        if split == "train":
            keep = hashed < 80
        elif split == "valid":
            keep = (hashed >= 80) & (hashed < 90)
        elif split == "id":
            keep = hashed >= 90
        else:
            raise IdxV2Error(
                "IDX_V2_SPLIT",
                f"idx_v2 smoke split {split!r} unsupported; use train/valid/id "
                "(the frozen V5.5 OOD splits await the contract adapter)")
        indices = np.flatnonzero(keep).tolist()
        if max_curves is not None:
            indices = indices[: int(max_curves)]
        return indices
