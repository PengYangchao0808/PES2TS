"""Read-only loader for the V5.3-MP label cache (Zone 2).

Source mapping (ported from the frozen ``pes2ts_core/pesn`` tree, removed from
master):

* ``pes2ts_core/pesn/offline/data/key_atom_label_cache_v53mp.py`` — the
  ``labels_cache_v53mp.npz`` + ``labels_cache_v53mp.json`` read path
  (:func:`load_label_cache`); its label classes (``V52SampleLabel`` /
  ``CurveV52Labels`` / ``EndpointSampleLabelV1`` ...) are replaced by the frozen
  dataclasses below with the same attribute names.

The cache is a provenance-bound accelerator: it is bound to the dataset
``data_manifest.json`` sha256, so a rebuilt dataset is never served from a stale
cache (fail-closed, typed :class:`~._common.ReaderError`).  Read-only: nothing
here writes, imports torch or reads the truth tree.  The companion light-index
pickle reader lives in :mod:`pes2ts_data.readers.light_index`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from ._common import DATA_MANIFEST_FILENAME, ReaderError, require, sha256_file

__all__ = [
    "CACHE_ARRAYS",
    "CACHE_JSON",
    "CACHE_NPZ",
    "CACHE_SCHEMA",
    "CurveLabels",
    "EndpointSampleLabel",
    "LabelTables",
    "V52SampleLabel",
    "load_label_cache",
]

#: Label cache filenames + schema id (frozen ``v53mp_label_cache_v1``).
CACHE_NPZ: Final[str] = "labels_cache_v53mp.npz"
CACHE_JSON: Final[str] = "labels_cache_v53mp.json"
CACHE_SCHEMA: Final[str] = "v53mp_label_cache_v1"

#: npz members consumed by :func:`load_label_cache` (schema of the frozen writer).
CACHE_ARRAYS: Final[tuple[str, ...]] = (
    "sample_ids", "q", "q_mask", "stop", "d_key_ts", "fold", "label_status",
    "direction", "curve_of", "node_index", "intent_of", "ep_label", "d_end",
    "bond_edit_complete", "geometry_legal", "ts_row", "start_row",
    "terminal_row", "degenerate_curve", "energy_ev_within_window",
    "d_key_ts_ep", "node_order", "arc_s", "q_canonical",
)


@dataclass(frozen=True)
class V52SampleLabel:
    """Per-sample V5.2 diagnostic labels (progress/stop/fold)."""

    sample_id: str
    curve_id: str
    node_index: int
    intent_id: str
    direction: str
    fold: int | None
    label_status: str
    q: float | None
    q_mask: bool
    stop: int
    d_key_ts: float


@dataclass(frozen=True)
class CurveLabels:
    """Per-curve canonical label vectors."""

    curve_id: str
    ts_index: int
    arc_s: tuple[float, ...]
    q_canonical: tuple[float, ...] | None
    q_zero_arc_reason: str
    node_order: tuple[int, ...]


@dataclass(frozen=True)
class EndpointSampleLabel:
    """Per-sample endpoint-label-spec-v1 fields."""

    sample_id: str
    curve_id: str
    node_index: int
    intent_id: str
    direction: str
    fold: int | None
    label_status: str
    endpoint_label: int
    d_end_angstrom: float
    bond_edit_complete: bool
    geometry_legal: bool
    ts_row: bool
    start_row: bool
    terminal_row: bool
    degenerate_curve: bool
    energy_ev_within_window: bool | None
    d_key_ts_angstrom: float


@dataclass(frozen=True)
class LabelTables:
    """Validated label-cache contents for one tables directory."""

    by_sample: dict[str, V52SampleLabel]
    endpoints: dict[str, EndpointSampleLabel]
    curves: dict[str, CurveLabels]
    positive_window: float
    negative_window: float
    doc: dict[str, Any]
    tables_dir: Path

    @property
    def sample_ids(self) -> list[str]:
        return sorted(self.by_sample)

    @property
    def n_samples(self) -> int:
        return len(self.by_sample)

    @property
    def n_curves(self) -> int:
        return len(self.curves)

    def sample(self, sample_id: str) -> dict[str, Any]:
        """Joined label view of one sample (typed refusal when absent)."""
        v52 = self.by_sample.get(sample_id)
        endpoint = self.endpoints.get(sample_id)
        require(v52 is not None and endpoint is not None, "LABEL_SAMPLE_MISSING",
                f"sample {sample_id!r} is not in the label cache")
        assert v52 is not None and endpoint is not None
        return {
            "sample_id": sample_id,
            "curve_id": v52.curve_id,
            "node_index": v52.node_index,
            "intent_id": v52.intent_id,
            "direction": v52.direction,
            "fold": v52.fold,
            "label_status": v52.label_status,
            "q": v52.q,
            "q_mask": v52.q_mask,
            "stop": v52.stop,
            "d_key_ts": v52.d_key_ts,
            "endpoint_label": endpoint.endpoint_label,
            "d_end_angstrom": endpoint.d_end_angstrom,
            "bond_edit_complete": endpoint.bond_edit_complete,
            "geometry_legal": endpoint.geometry_legal,
            "ts_row": endpoint.ts_row,
            "start_row": endpoint.start_row,
            "terminal_row": endpoint.terminal_row,
            "degenerate_curve": endpoint.degenerate_curve,
            "energy_ev_within_window": endpoint.energy_ev_within_window,
            "d_key_ts_angstrom": endpoint.d_key_ts_angstrom,
        }

    def curve(self, curve_id: str) -> CurveLabels:
        curve = self.curves.get(curve_id)
        require(curve is not None, "LABEL_CURVE_MISSING",
                f"curve {curve_id!r} is not in the label cache")
        assert curve is not None
        return curve


def _fold_value(value: Any) -> int | None:
    return None if int(value) < 0 else int(value)


def load_label_cache(
    tables_dir: str | Path,
    *,
    dataset_manifest_sha256: str | None = None,
    verify_manifest: bool = True,
) -> LabelTables:
    """Load ``labels_cache_v53mp.npz`` + ``.json`` (fail-closed, read-only).

    Parameters
    ----------
    tables_dir:
        Directory holding the six tables and the label-cache artifacts.
    dataset_manifest_sha256:
        Expected dataset ``data_manifest.json`` sha256.  When omitted and
        ``verify_manifest`` is true, the on-disk manifest is hashed and compared
        with the cache binding (a rebuilt dataset is refused as stale).
    verify_manifest:
        Set ``False`` to load without the dataset binding check (structural
        smoke tests only).
    """
    tables_dir = Path(tables_dir)
    json_path = tables_dir / CACHE_JSON
    npz_path = tables_dir / CACHE_NPZ
    require(json_path.is_file() and npz_path.is_file(), "CACHE_MISSING",
            f"{tables_dir}: expected {CACHE_NPZ} and {CACHE_JSON}")
    try:
        doc = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReaderError("CACHE_PARSE", f"{json_path}: {exc}") from exc
    require(doc.get("schema") == CACHE_SCHEMA, "CACHE_SCHEMA",
            f"{json_path}: schema {doc.get('schema')!r} != {CACHE_SCHEMA!r}")

    expected = dataset_manifest_sha256 or doc.get("dataset_manifest_sha256") or ""
    manifest_path = tables_dir / DATA_MANIFEST_FILENAME
    if verify_manifest and expected and manifest_path.is_file():
        on_disk = sha256_file(manifest_path)
        require(on_disk == expected, "CACHE_STALE",
                f"cache is bound to data_manifest sha256 {expected[:12]}... but "
                f"the tables dir hashes to {on_disk[:12]}...")

    try:
        with np.load(npz_path, allow_pickle=False) as data:
            missing = [name for name in CACHE_ARRAYS if name not in data.files]
            require(not missing, "CACHE_ARRAYS",
                    f"{npz_path}: missing members {missing}")
            arrays = {name: np.asarray(data[name]) for name in CACHE_ARRAYS}
    except (OSError, ValueError) as exc:
        raise ReaderError("CACHE_PARSE", f"{npz_path}: {exc}") from exc

    sample_ids = [str(value) for value in arrays["sample_ids"]]
    n = len(sample_ids)
    require(int(doc.get("n_samples", -1)) == n, "CACHE_COUNT",
            f"sample count drift: json {doc.get('n_samples')} != npz {n}")

    by_sample: dict[str, V52SampleLabel] = {}
    endpoints: dict[str, EndpointSampleLabel] = {}
    for index, sample_id in enumerate(sample_ids):
        q_value = float(arrays["q"][index])
        fold = _fold_value(arrays["fold"][index])
        by_sample[sample_id] = V52SampleLabel(
            sample_id=sample_id,
            curve_id=str(arrays["curve_of"][index]),
            node_index=int(arrays["node_index"][index]),
            intent_id=str(arrays["intent_of"][index]),
            direction=str(arrays["direction"][index]),
            fold=fold,
            label_status=str(arrays["label_status"][index]),
            q=None if np.isnan(q_value) else q_value,
            q_mask=bool(arrays["q_mask"][index]),
            stop=int(arrays["stop"][index]),
            d_key_ts=float(arrays["d_key_ts"][index]),
        )
        within = int(arrays["energy_ev_within_window"][index])
        endpoints[sample_id] = EndpointSampleLabel(
            sample_id=sample_id,
            curve_id=str(arrays["curve_of"][index]),
            node_index=int(arrays["node_index"][index]),
            intent_id=str(arrays["intent_of"][index]),
            direction=str(arrays["direction"][index]),
            fold=fold,
            label_status=str(arrays["label_status"][index]),
            endpoint_label=int(arrays["ep_label"][index]),
            d_end_angstrom=float(arrays["d_end"][index]),
            bond_edit_complete=bool(arrays["bond_edit_complete"][index]),
            geometry_legal=bool(arrays["geometry_legal"][index]),
            ts_row=bool(arrays["ts_row"][index]),
            start_row=bool(arrays["start_row"][index]),
            terminal_row=bool(arrays["terminal_row"][index]),
            degenerate_curve=bool(arrays["degenerate_curve"][index]),
            energy_ev_within_window=None if within < 0 else bool(within),
            d_key_ts_angstrom=float(arrays["d_key_ts_ep"][index]),
        )

    curve_meta = doc.get("curve_meta")
    require(isinstance(curve_meta, list), "CACHE_SCHEMA",
            f"{json_path}: curve_meta missing")
    curves: dict[str, CurveLabels] = {}
    pos = pos_q = 0
    for meta in curve_meta:
        size = int(meta["n"])
        has_q = not meta.get("q_zero_arc_reason")
        curves[str(meta["curve_id"])] = CurveLabels(
            curve_id=str(meta["curve_id"]),
            ts_index=int(meta.get("ts_index", 0)),
            arc_s=tuple(float(v) for v in arrays["arc_s"][pos:pos + size]),
            q_canonical=(tuple(float(v) for v in
                               arrays["q_canonical"][pos_q:pos_q + size])
                         if has_q else None),
            q_zero_arc_reason=str(meta.get("q_zero_arc_reason", "")),
            node_order=tuple(int(v) for v in arrays["node_order"][pos:pos + size]),
        )
        pos += size
        pos_q += size if has_q else 0
    require(len(curves) == int(doc.get("n_curves", len(curves))), "CACHE_COUNT",
            "curve count drift")
    windows = doc.get("endpoint_windows") or {}
    return LabelTables(
        by_sample=by_sample,
        endpoints=endpoints,
        curves=curves,
        positive_window=float(windows.get("positive_window_angstrom", 0.0)),
        negative_window=float(windows.get("negative_window_angstrom", 0.0)),
        doc=doc,
        tables_dir=tables_dir,
    )
