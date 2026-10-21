"""R3 v3 dataset: manifest-driven, ``intent_id``-keyed six-table loader.

Consumes the PD-1 six-table layout (30h plan §3.1 / campaign ``data_contract``)
under ``outputs/pesn/30h_campaign/data/`` (fixtures are built in tests until
E1 delivers the real tables):

```text
irc_curve.csv  irc_node.csv  irc_edge.csv
reaction_intent.csv  curve_quality.csv  sample_view.csv
data_manifest.json  teacher_manifest.json  split_manifest_r3.json
```

Column expectations are mirrored from the E1 schema
``scripts/agent_ops/data_gate/r3_table_schema.py`` (``r3_6table_v2``); a
machine check in ``tests/pesn/test_r3_dataset.py`` pins the mirror so schema
drift in E1 fails loudly here instead of silently mis-parsing.

Design rules:

* **sample identity** = ``(curve_id, node_index, intent_id, d0)``
  (ReactionIntent Contract v1 §4.5); ``d0`` defaults to the frozen primary
  0.08 Å; non-primary rows are dropped unless explicitly requested;
* **intent-keyed**: every sample resolves its directed ``G_rxn`` through the
  ``reaction_intent`` row (``IntentRecordV3``); forward/reverse share one
  curve and one fold — a reverse sample is never an independent curve;
* **weights once**: the dataset exposes the ORA-2 #3 weight triple as-is —
  ``curve_weight`` (sampler-side, sums to 1.0 per ``(curve_id, d0)``,
  verified), ``sample_weight`` and ``label_weight``; the LOSS applies
  ``label_weight`` exactly once via ``frame_weight`` (30h plan §4.1:
  sampling is curve-balanced, so no inverse-curve re-multiplication);
* **legality**: supervision fields (``target_action_u``,
  ``target_progress_q``, ``stop_label``, weights, ``label_status``) stay on
  the record / in the collated batch's SUPERVISION section and are dropped
  by :func:`~pes2ts_core.pesn.contracts.key_atom_v3.forward_view_v3` —
  ``model.forward`` can never see them (ReactionIntent §5);
* ``curve_quality`` is loaded only for presence (provenance-only columns
  must never become inputs, targets or split keys — 30h plan §3.1);
* manifests are verified fail-closed when present (sha256 per file).

E4 (V5.2) extension — read-compatible: the loader additionally exposes
label-side accessors (:meth:`KeyAtomDatasetV3.frame_xyz`,
:meth:`curve_ts_index`, :meth:`curve_endpoints`) and
:meth:`KeyAtomDatasetV3.v52_labels`, which derive the frozen V5.2 q and
TSStop labels from the SAME six-table teacher columns (mass-weighted arc
segmentation A=−1/TS=0/B=+1 with reverse q=−q and zero-arc masking;
three-way stop window 0.04/0.12 Å with Kabsch-aligned K-geometry RMSD).
No column, invariant or filter of the R3 path changed.

V5.3-MP S0-D extension (plan §4.4) — APPENDED, read-compatible:
:class:`ShardedKeyAtomDatasetV3` reads the same six tables LAZILY (light
sample index at init; per-curve geometry on demand through a bounded
LRU, with an optional ``shard_index_v53mp.json`` npz fast path built by
:func:`write_geometry_shards`); the in-memory ``KeyAtomDatasetV3`` above
is unchanged and remains the reference loader for small deliveries.
"""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

import numpy as np

from pes2ts_core.pesn.contracts.key_atom_v3 import (
    D0_PRIMARY_ANGSTROM,
    IntentRecordV3,
    KeyAtomSampleV3,
    KeyAtomV3ContractError,
)

# --- E4 (V5.2) frozen label constants (campaign ``v52_training``) -----------
#: Stop positive window: ``d_key_TS <= 0.04 Å`` (stop_label=1; 禁止偷扩窗).
STOP_POSITIVE_WINDOW_ANGSTROM_V52: Final[float] = 0.04
#: Stop negative window: ``d_key_TS >= 0.12 Å`` (stop_label=0; middle=Mask).
STOP_NEGATIVE_WINDOW_ANGSTROM_V52: Final[float] = 0.12
#: Zero-arc mask threshold (plan §5.2: mask instead of dividing by zero).
ZERO_ARC_EPS_ANGSTROM_V52: Final[float] = 1.0e-8
#: Label-table schema id (G3a report provenance).
STOP_LABEL_SCHEMA_V52: Final[str] = "pes2ts_e4_labels_v1"

#: Six tables + manifests (PD-1; plan §3.1: 共六张，不是五张).
TABLE_FILENAMES: dict[str, str] = {
    "irc_curve": "irc_curve.csv",
    "irc_node": "irc_node.csv",
    "irc_edge": "irc_edge.csv",
    "reaction_intent": "reaction_intent.csv",
    "curve_quality": "curve_quality.csv",
    "sample_view": "sample_view.csv",
}
DATA_MANIFEST_FILENAME = "data_manifest.json"
SPLIT_MANIFEST_FILENAME = "split_manifest_r3.json"

#: Required columns mirrored from ``r3_6table_v2`` (E1 schema; machine-pinned
#: by ``tests/pesn/test_r3_dataset.py``).
REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "irc_curve": ("curve_id", "canonical_irc_id", "n_nodes", "ts_index"),
    "irc_node": ("curve_id", "node_index", "x", "y", "z",
                 "atomic_numbers"),
    "reaction_intent": ("intent_id", "curve_id", "direction",
                        "reverse_of_intent_id", "intent_pair_id",
                        "g_src", "g_tgt", "cv_spec"),
    "sample_view": ("sample_id", "curve_id", "node_index", "intent_id", "d0",
                    "label_weight", "curve_weight", "sample_weight",
                    "label_status"),
    "irc_edge": ("curve_id", "intent_id", "edge_index", "delta_xyz_flat"),
    "curve_quality": ("curve_id", "quality_status"),
}

#: Label statuses accepted by default (ORA-2 #3 enum).
LABEL_STATUSES = ("ok", "masked", "terminal", "conflict")

WEIGHT_SUM_TOL = 1e-9


class KeyAtomDatasetV3Error(RuntimeError):
    """Typed refusal of the v3 dataset loader."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def _require(cond: bool, code: str, detail: str) -> None:
    if not cond:
        raise KeyAtomDatasetV3Error(code, detail)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise KeyAtomDatasetV3Error("TABLE_MISSING", f"{path} not found")
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        rows = [dict(r) for r in reader]
    return rows


def _check_columns(table: str, rows: Sequence[Mapping[str, str]]) -> None:
    required = REQUIRED_COLUMNS.get(table, ())
    if not rows:
        return
    header = set(rows[0])
    missing = [c for c in required if c not in header]
    _require(not missing, "TABLE_COLUMNS",
             f"{table} is missing required columns {missing}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_data_manifest_at(tables_dir: Path) -> str:
    """Fail-closed sha256 verification of ``data_manifest.json``.

    Returns the manifest's own sha256 when a manifest is present, or
    ``""`` when E1 has not delivered one yet (mirrors the historical
    in-class behavior; shared by the full and sharded loaders).
    """
    path = Path(tables_dir) / DATA_MANIFEST_FILENAME
    if not path.is_file():
        return ""                         # E1 may deliver manifests later
    doc = json.loads(path.read_text(encoding="utf-8"))
    files = doc.get("files", {})
    if isinstance(files, list):
        entries = {str(e["path"]): str(e["sha256"]) for e in files}
    else:
        entries = {str(k): str(v) for k, v in files.items()}
    for rel, sha in entries.items():
        target = Path(tables_dir) / rel
        _require(target.is_file(), "MANIFEST_FILE",
                 f"{rel} listed in {DATA_MANIFEST_FILENAME} not found")
        actual = _sha256(target)
        _require(actual == sha, "MANIFEST_SHA",
                 f"{rel}: manifest sha256 {sha[:12]}… != actual "
                 f"{actual[:12]}…")
    return _sha256(path)


def _load_folds_at(
    tables_dir: Path,
    split_manifest: str | Path | None,
    fold: int | None,
) -> dict[str, int]:
    """curve_id → fold from a split manifest (shared with the sharded loader).

    Accepted shapes mirror the R3 deliveries (``curve_folds`` mapping,
    ``assignments`` list, ``folds`` mapping, or a bare list).
    """
    if split_manifest is None:
        candidate = Path(tables_dir) / SPLIT_MANIFEST_FILENAME
        split_manifest = candidate if candidate.is_file() else None
    if split_manifest is None:
        _require(fold is None, "SPLIT_MANIFEST_MISSING",
                 "fold filter requested but no split manifest found")
        return {}
    doc = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
    folds: dict[str, int] = {}
    if isinstance(doc, Mapping) and "curve_folds" in doc:
        folds = {str(k): int(v) for k, v in doc["curve_folds"].items()}
    elif isinstance(doc, Mapping) and "assignments" in doc:
        folds = {str(a["curve_id"]): int(a["fold"])
                 for a in doc["assignments"]}
    elif isinstance(doc, Mapping) and isinstance(doc.get("folds"), Mapping):
        for f, curve_ids in doc["folds"].items():
            for cid in curve_ids:
                folds[str(cid)] = int(f)
    elif isinstance(doc, list):
        folds = {str(a["curve_id"]): int(a["fold"]) for a in doc}
    else:
        raise KeyAtomDatasetV3Error(
            "SPLIT_MANIFEST_SHAPE",
            f"unrecognized split manifest shape in {split_manifest}")
    _require(folds, "SPLIT_MANIFEST_EMPTY", "split manifest has no folds")
    return folds


@dataclass(frozen=True)
class SampleRecordV3:
    """One ``sample_view`` row joined with its intent + node geometry."""

    sample_id: str
    curve_id: str
    node_index: int
    intent: IntentRecordV3
    d0: float
    x: float
    y: float
    z: float
    label_weight: float
    curve_weight: float
    sample_weight: float
    label_status: str
    atomic_numbers: tuple[int, ...]
    target_action_u: tuple[tuple[float, float, float], ...] | None
    target_node_index: int | None
    fold: int | None
    feature_sha: str = ""
    label_sha: str = ""


class KeyAtomDatasetV3:
    """Manifest-driven six-table dataset keyed by ``intent_id``."""

    def __init__(
        self,
        tables_dir: str | Path,
        *,
        d0: float = D0_PRIMARY_ANGSTROM,
        label_statuses: Iterable[str] = ("ok",),
        fold: int | None = None,
        split_manifest: str | Path | None = None,
        verify_manifest: bool = True,
        directions: Iterable[str] = ("forward", "reverse"),
    ) -> None:
        self.tables_dir = Path(tables_dir)
        self.d0 = float(d0)
        self.fold = fold
        self.label_statuses = frozenset(label_statuses)
        self.directions = frozenset(directions)
        _require(self.tables_dir.is_dir(), "TABLES_DIR",
                 f"{self.tables_dir} is not a directory")

        # --- presence of all six tables (PD-1; curve_quality provenance) ---
        for table, filename in TABLE_FILENAMES.items():
            _require((self.tables_dir / filename).is_file(),
                     "TABLE_MISSING", f"{table} -> {filename} not found")

        if verify_manifest:
            self._verify_data_manifest()

        raw: dict[str, list[dict[str, str]]] = {}
        for table, filename in TABLE_FILENAMES.items():
            rows = _read_csv(self.tables_dir / filename)
            _check_columns(table, rows)
            raw[table] = rows
        self._raw = raw

        # --- index -----------------------------------------------------------
        self._curves: dict[str, dict[str, str]] = {
            r["curve_id"]: r for r in raw["irc_curve"]}
        require_len = len(self._curves)
        _require(require_len == len(raw["irc_curve"]),
                 "CURVE_ID_DUP", "duplicate curve_id in irc_curve")
        # Per-frame geometry: ``r3_6table_v2`` declares single-float x/y/z
        # (used by the E1 finiteness gates), but the v3 feature builder needs
        # the FULL [N,3] frame.  The loader accepts a JSON [N,3] payload in
        # the x cell (y/z then ignored) and falls back to the single triple
        # (only usable for N==1).  E1 must deliver one of the two — recorded
        # as a downstream interface note in the W-FEAT handoff.
        self._frame_xyz: dict[tuple[str, int], np.ndarray] = {}
        self._curve_atomic: dict[str, tuple[int, ...]] = {}
        for row in raw["irc_node"]:
            key = (row["curve_id"], int(row["node_index"]))
            _require(key not in self._frame_xyz, "NODE_DUP",
                     f"duplicate node row {key}")
            self._frame_xyz[key] = self._parse_frame_xyz(row)
            payload = row.get("atomic_numbers") or ""
            if payload:
                try:
                    self._curve_atomic.setdefault(
                        row["curve_id"],
                        tuple(int(v) for v in json.loads(payload)))
                except (ValueError, json.JSONDecodeError) as exc:
                    raise KeyAtomDatasetV3Error(
                        "ATOMIC_NUMBERS_PARSE",
                        f"{row['curve_id']}: {exc}") from exc

        self._intents: dict[tuple[str, str], IntentRecordV3] = {}
        for row in raw["reaction_intent"]:
            try:
                intent = IntentRecordV3.from_table_row(row)
            except (KeyAtomV3ContractError, ValueError, json.JSONDecodeError) as exc:
                raise KeyAtomDatasetV3Error(
                    "INTENT_PARSE", f"{row.get('intent_id')}: {exc}") from exc
            key = (intent.curve_id, intent.intent_id)
            _require(key not in self._intents, "INTENT_DUP",
                     f"duplicate (curve_id, intent_id) {key}")
            _require(intent.curve_id in self._curves, "INTENT_FK",
                     f"intent {intent.intent_id} references unknown curve")
            self._intents[key] = intent

        folds = self._load_folds(split_manifest, fold)
        self._folds = folds

        self._samples: list[SampleRecordV3] = []
        weight_totals: dict[str, float] = {}
        for row in raw["sample_view"]:
            # ORA-2 #3 / T0.1a invariant is defined over the FULL
            # (curve_id, d0) group — including masked/non-ok rows — so it is
            # accumulated BEFORE any status/fold filtering.
            try:
                if abs(float(row["d0"]) - self.d0) <= 1e-12:
                    cid = str(row["curve_id"])
                    weight_totals[cid] = (
                        weight_totals.get(cid, 0.0)
                        + float(row["curve_weight"]))
            except (KeyError, ValueError) as exc:
                raise KeyAtomDatasetV3Error(
                    "SAMPLE_WEIGHT_PARSE", f"{row.get('sample_id')}: {exc}"
                ) from exc
            rec = self._parse_sample(row, folds)
            if rec is None:
                continue
            self._samples.append(rec)
        _require(len(self._samples) > 0, "NO_SAMPLES",
                 "no sample_view rows survive the dataset filters "
                 f"(d0={self.d0}, statuses={sorted(self.label_statuses)}, "
                 f"fold={fold})")
        self._verify_curve_weights(weight_totals)
        self._verify_intent_pairs()

    # -- manifest -------------------------------------------------------------
    def _verify_data_manifest(self) -> None:
        sha = _verify_data_manifest_at(self.tables_dir)
        if sha:
            self.manifest_sha256 = sha

    # -- folds ----------------------------------------------------------------
    def _load_folds(
        self,
        split_manifest: str | Path | None,
        fold: int | None,
    ) -> dict[str, int]:
        return _load_folds_at(self.tables_dir, split_manifest, fold)

    # -- sample parsing -------------------------------------------------------
    @staticmethod
    def _parse_frame_xyz(row: Mapping[str, str]) -> np.ndarray:
        """Parse one ``irc_node`` row's geometry.

        Encodings accepted (in priority order):

        1. ``coords_flat`` + ``coords_shape`` columns — the E1 six-table
           delivery (PD-9 flat float array + ``[N,3]`` descriptor);
        2. a JSON ``[N,3]`` payload in the ``x`` cell (test fixtures);
        3. plain float ``x/y/z`` (representative triple, N==1 only).

        Non-finite coordinates are a typed refusal (gate T0.1c).
        """
        flat = (row.get("coords_flat") or "").strip()
        shape = (row.get("coords_shape") or "").strip()
        if flat and shape:
            try:
                arr = np.asarray(json.loads(flat), dtype=np.float64)
                shape_t = tuple(int(v) for v in json.loads(shape))
            except (ValueError, json.JSONDecodeError) as exc:
                raise KeyAtomDatasetV3Error(
                    "NODE_COORD_PARSE", f"coords_flat/shape: {exc}") from exc
            if shape_t and np.prod(shape_t) != arr.size:
                raise KeyAtomDatasetV3Error(
                    "NODE_COORD_SHAPE",
                    f"coords_flat has {arr.size} values, shape says {shape_t}")
            arr = arr.reshape(shape_t) if shape_t else arr.reshape(-1, 3)
            if arr.ndim != 2 or arr.shape[1] != 3:
                raise KeyAtomDatasetV3Error(
                    "NODE_COORD_SHAPE",
                    f"coords payload must reshape to [N,3], got {tuple(arr.shape)}")
            _require(bool(np.all(np.isfinite(arr))), "NODE_COORD_NONFINITE",
                     "frame coordinates contain non-finite values")
            return arr
        x_cell = (row.get("x") or "").strip()
        if x_cell.startswith("["):
            try:
                arr = np.asarray(json.loads(x_cell), dtype=np.float64)
            except (ValueError, json.JSONDecodeError) as exc:
                raise KeyAtomDatasetV3Error(
                    "NODE_COORD_PARSE", f"JSON x payload: {exc}") from exc
            if arr.ndim != 2 or arr.shape[1] != 3:
                raise KeyAtomDatasetV3Error(
                    "NODE_COORD_SHAPE",
                    f"JSON x payload must be [N,3], got {tuple(arr.shape)}")
            _require(bool(np.all(np.isfinite(arr))), "NODE_COORD_NONFINITE",
                     "frame coordinates contain non-finite values")
            return arr
        try:
            triple = np.asarray(
                [float(row["x"]), float(row["y"]), float(row["z"])],
                dtype=np.float64).reshape(1, 3)
        except ValueError as exc:
            raise KeyAtomDatasetV3Error(
                "NODE_COORD_PARSE", f"node row: {exc}") from exc
        _require(bool(np.all(np.isfinite(triple))), "NODE_COORD_NONFINITE",
                 "node coordinates contain non-finite values")
        return triple

    def _parse_sample(
        self,
        row: Mapping[str, str],
        folds: Mapping[str, int],
    ) -> SampleRecordV3 | None:
        try:
            d0 = float(row["d0"])
        except ValueError as exc:
            raise KeyAtomDatasetV3Error(
                "SAMPLE_D0_PARSE", f"{row.get('sample_id')}: {exc}") from exc
        if abs(d0 - self.d0) > 1e-12:
            return None                 # non-primary d0 rows are out of scope
        status = str(row.get("label_status", ""))
        if status not in self.label_statuses:
            return None
        curve_id = row["curve_id"]
        node_index = int(row["node_index"])
        intent_key = (curve_id, row["intent_id"])
        intent = self._intents.get(intent_key)
        _require(intent is not None, "SAMPLE_INTENT_FK",
                 f"sample {row.get('sample_id')} references unknown intent "
                 f"{intent_key}")
        _require(intent.direction in self.directions, "SAMPLE_DIRECTION",
                 f"sample {row.get('sample_id')} direction "
                 f"{intent.direction} filtered out")
        node_key = (curve_id, node_index)
        _require(node_key in self._frame_xyz, "SAMPLE_NODE_FK",
                 f"sample {row.get('sample_id')} references unknown node "
                 f"{node_key}")
        if folds:
            curve_fold = folds.get(curve_id)
            _require(curve_fold is not None, "SPLIT_CURVE_MISSING",
                     f"curve {curve_id} missing from split manifest")
            if self.fold is not None and int(curve_fold) != int(self.fold):
                return None
        atomic = self._curve_atomic.get(curve_id) or intent.g_atoms
        _require(bool(atomic), "ATOMIC_NUMBERS_MISSING",
                 f"curve {curve_id} has no atomic_numbers payload (irc_node "
                 "nor endpoint-graph atoms)")

        target_u = None
        target_node = None
        payload = (row.get("target_action_u") or "").strip()
        if payload:
            try:
                doc = json.loads(payload)
                flat = [float(v) for v in doc] if (
                    doc and isinstance(doc[0], (int, float))) else None
                if flat is not None:
                    if len(flat) % 3 != 0:
                        raise ValueError(
                            f"flat target_action_u length {len(flat)} % 3 != 0")
                    target_u = tuple(
                        (flat[k], flat[k + 1], flat[k + 2])
                        for k in range(0, len(flat), 3))
                else:
                    target_u = tuple(
                        (float(v[0]), float(v[1]), float(v[2])) for v in doc)
            except (ValueError, json.JSONDecodeError, IndexError) as exc:
                raise KeyAtomDatasetV3Error(
                    "TARGET_U_PARSE",
                    f"{row.get('sample_id')}: {exc}") from exc
            _require(len(target_u) == len(intent.key_rows()), "TARGET_U_SHAPE",
                     f"{row.get('sample_id')}: target_action_u has "
                     f"{len(target_u)} rows != |K|={len(intent.key_rows())}")
        tn = (row.get("target_node_index") or "").strip()
        if tn:
            target_node = int(tn)

        fold_value = (row.get("fold") or "").strip()
        curve = self._curves[curve_id]
        return SampleRecordV3(
            sample_id=str(row["sample_id"]),
            curve_id=curve_id,
            node_index=node_index,
            intent=intent,
            d0=d0,
            x=float(row["x"]),
            y=float(row["y"]),
            z=float(row["z"]),
            label_weight=float(row["label_weight"]),
            curve_weight=float(row["curve_weight"]),
            sample_weight=float(row["sample_weight"]),
            label_status=status,
            atomic_numbers=atomic,
            target_action_u=target_u,
            target_node_index=target_node,
            fold=int(fold_value) if fold_value else folds.get(curve_id),
            feature_sha=str(row.get("feature_sha", "")),
            label_sha=str(row.get("label_sha", "")),
        )

    # -- invariants -----------------------------------------------------------
    def _verify_curve_weights(self, totals: Mapping[str, float]) -> None:
        """ORA-2 #3 / T0.1a: Σ curve_weight == 1.0 per ``(curve_id, d0)``
        over the full sample_view group (pre-filter)."""
        for curve_id, total in totals.items():
            _require(abs(total - 1.0) <= WEIGHT_SUM_TOL, "CURVE_WEIGHT_SUM",
                     f"curve {curve_id}: sum(curve_weight)={total!r} != 1.0")

    def _verify_intent_pairs(self) -> None:
        """Every non-underdetermined intent has a mutual reverse (T0.2a)."""
        for (curve_id, intent_id), intent in self._intents.items():
            if intent.intent_underdetermined:
                continue
            rev = self._intents.get((curve_id, intent.reverse_of_intent_id))
            _require(rev is not None, "INTENT_REVERSE_MISSING",
                     f"intent {intent_id} has no reverse row "
                     f"{intent.reverse_of_intent_id!r}")
            _require(rev.reverse_of_intent_id == intent_id, "INTENT_INVOLUTION",
                     f"reverse(reverse({intent_id})) != {intent_id}")

    # -- public API -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self._samples)

    @property
    def samples(self) -> tuple[SampleRecordV3, ...]:
        return tuple(self._samples)

    @property
    def intents(self) -> Mapping[tuple[str, str], IntentRecordV3]:
        return dict(self._intents)

    @property
    def folds(self) -> Mapping[str, int]:
        return dict(self._folds)

    # -- V5.2 label-side accessors (E4; read-only views over loaded tables) --
    def frame_xyz(self, curve_id: str, node_index: int) -> tuple:
        """Full ``[N,3]`` current-frame geometry (public E4 accessor)."""
        key = (str(curve_id), int(node_index))
        if key not in self._frame_xyz:
            raise KeyAtomDatasetV3Error(
                "NODE_UNKNOWN", f"curve {curve_id} has no node {node_index}")
        arr = self._frame_xyz[key]
        return tuple(tuple(float(v) for v in row) for row in arr)

    def curve_ts_index(self, curve_id: str) -> int:
        """Trusted TS frame index of the curve (irc_curve.ts_index, PD-9)."""
        curve = self._curves.get(str(curve_id))
        if curve is None:
            raise KeyAtomDatasetV3Error(
                "CURVE_UNKNOWN", f"curve {curve_id} not in irc_curve")
        return int(curve["ts_index"])

    def curve_endpoints(self, curve_id: str) -> tuple[int, int]:
        """``(endpoint_a_index, endpoint_b_index)`` of the canonical A→B path."""
        curve = self._curves.get(str(curve_id))
        if curve is None:
            raise KeyAtomDatasetV3Error(
                "CURVE_UNKNOWN", f"curve {curve_id} not in irc_curve")
        a = int(curve.get("endpoint_a_index", 0) or 0)
        b = int(curve.get("endpoint_b_index", "") or
                (int(curve["n_nodes"]) - 1))
        return a, b

    def v52_labels(
        self,
        *,
        stop_positive_window: float = STOP_POSITIVE_WINDOW_ANGSTROM_V52,
        stop_negative_window: float = STOP_NEGATIVE_WINDOW_ANGSTROM_V52,
        zero_arc_eps: float = ZERO_ARC_EPS_ANGSTROM_V52,
    ) -> "V52LabelTable":
        """E4: compute q / TSStop labels for every LOADED sample row.

        Labels derive ONLY from the six-table teacher columns (irc_curve
        ``ts_index``/endpoints, irc_node geometry + ``atomic_numbers``,
        sample_view identity/weights) — no ground-truth reads.  See
        :func:`build_v52_labels` for the frozen math.
        """
        return build_v52_labels(
            self,
            stop_positive_window=stop_positive_window,
            stop_negative_window=stop_negative_window,
            zero_arc_eps=zero_arc_eps)

    def records(self, indices: Iterable[int] | None = None) -> list[SampleRecordV3]:
        if indices is None:
            return list(self._samples)
        return [self._samples[int(i)] for i in indices]

    def to_sample_v3(self, rec: SampleRecordV3) -> KeyAtomSampleV3:
        """Convert a table record into the buildable v3 sample.

        ``frame_weight`` carries ``label_weight`` — the label-quality weight
        the loss applies exactly once (curve weight stays sampler-side, per
        30h plan §4.1: 采样已 curve-balanced 时不再叠乘等效 inverse-curve
        权重；只在 loss 层施加 label-quality 权重).
        """
        xyz = self._frame_xyz_for(rec)
        return KeyAtomSampleV3(
            curve_id=rec.curve_id,
            node_index=rec.node_index,
            intent=rec.intent,
            atomic_numbers=rec.atomic_numbers,
            xyz=xyz,
            d0=rec.d0,
            endpoint_names=("A", "B"),
            y_teacher=rec.target_action_u,
            frame_weight=rec.label_weight,
            label_status=rec.label_status,
            sample_id=rec.sample_id,
            extras={"curve_weight": rec.curve_weight,
                    "sample_weight": rec.sample_weight,
                    "fold": rec.fold,
                    "feature_sha": rec.feature_sha,
                    "label_sha": rec.label_sha},
        )

    def _frame_xyz_for(self, rec: SampleRecordV3) -> tuple:
        """Full [N,3] current-frame geometry for the record's curve/node.

        Prefers the JSON [N,3] payload of the frame's ``irc_node`` row; falls
        back to a single representative triple (only legal for N==1).
        """
        n = len(rec.atomic_numbers)
        frame = self._frame_xyz[(rec.curve_id, rec.node_index)]
        if frame.shape == (n, 3):
            return tuple(tuple(float(v) for v in row) for row in frame)
        if frame.shape == (1, 3) and n == 1:
            return tuple(tuple(float(v) for v in frame[0]),)
        raise KeyAtomDatasetV3Error(
            "GEOMETRY_INCOMPLETE",
            f"curve {rec.curve_id} node {rec.node_index}: frame geometry "
            f"{tuple(frame.shape)} cannot cover N={n} atoms; E1 must deliver "
            "per-frame [N,3] coordinates (see handoff interface note)")


# ---------------------------------------------------------------------------
# E4 (V5.2) label machinery — q progress + TSStop three-way labels
# (frozen window constants are defined at module top, next to D0)
# ---------------------------------------------------------------------------

#: Conventional atomic masses (amu; most-abundant-isotope values), Z = 1..118.
#: Frozen E4 artifact — the mass UNIT cancels inside each curve's q
#: normalization, only the relative mass WEIGHTS matter (plan §5.2 requires
#: the mass set/units to be frozen; this table is that freeze).
ATOMIC_MASS_TABLE_V52: Final[tuple[float, ...]] = (
    1.008, 4.0026, 6.94, 9.0122, 10.81, 12.011, 14.007, 15.999, 18.998,
    20.180, 22.990, 24.305, 26.982, 28.085, 30.974, 32.06, 35.45, 39.948,
    39.098, 40.078, 44.956, 47.867, 50.942, 51.996, 54.938, 55.845, 58.933,
    58.693, 63.546, 65.38, 69.723, 72.630, 74.922, 78.971, 79.904, 83.798,
    85.468, 87.62, 88.906, 91.224, 92.906, 95.95, 98.0, 101.07, 102.91,
    106.42, 107.87, 112.41, 114.82, 118.71, 121.76, 127.60, 126.90, 131.29,
    132.91, 137.33, 138.91, 140.12, 140.91, 144.24, 145.0, 150.36, 151.96,
    157.25, 158.93, 162.50, 164.93, 167.26, 168.93, 173.05, 174.97, 178.49,
    180.95, 183.84, 186.21, 190.23, 192.22, 195.08, 196.97, 200.59, 204.38,
    207.2, 208.98, 209.0, 210.0, 222.0, 223.0, 226.0, 227.0, 232.04, 231.04,
    238.03, 237.0, 244.0, 243.0, 247.0, 247.0, 251.0, 252.0, 257.0, 258.0,
    259.0, 266.0, 267.0, 268.0, 269.0, 270.0, 269.0, 278.0, 281.0, 282.0,
    285.0, 286.0, 289.0, 290.0, 293.0, 294.0, 294.0,
)


def atomic_mass_v52(atomic_number: int) -> float:
    """Frozen amu mass for ``Z`` (1..118); out-of-range is a typed refusal."""
    z = int(atomic_number)
    if not (1 <= z <= len(ATOMIC_MASS_TABLE_V52)):
        raise KeyAtomDatasetV3Error(
            "ATOMIC_MASS_RANGE", f"Z={z} outside 1..{len(ATOMIC_MASS_TABLE_V52)}")
    return float(ATOMIC_MASS_TABLE_V52[z - 1])


def mass_weighted_arc_lengths_v52(
    frames: np.ndarray,
    atomic_numbers: Sequence[int],
) -> np.ndarray:
    """Cumulative mass-weighted arc length ``s [N]`` along an IRC node order.

    ``frames`` is ``[N_nodes, N_atoms, 3]`` in PATH order (already oriented
    A→B).  Per-step increment (frozen E4 definition, plan §5.2 质量加权弧长):

    ``ds_i = sqrt( Σ_a m_a·‖x_a^{(i)} − x_a^{(i−1)}‖² / Σ_a m_a )``

    (mass-weighted RMS displacement; amu).  No additional rigid alignment
    is applied between frames: the E1 six tables store every frame in ONE
    verified reference frame (``irc_edge.same_ref_frame``, PD-9; the R3
    teacher itself is built Kabsch-free in the native frame), so a Kabsch
    step would be a no-op here.  Translation already cancels inside the
    displacement difference.  ``s[0] = 0``.
    """
    arr = np.asarray(frames, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise KeyAtomDatasetV3Error(
            "ARC_FRAMES_SHAPE", f"frames must be [N,3], got {tuple(arr.shape)}")
    z = np.asarray(tuple(int(v) for v in atomic_numbers), dtype=np.int64)
    if int(z.shape[0]) != int(arr.shape[1]):
        raise KeyAtomDatasetV3Error(
            "ARC_ATOMS_MISMATCH",
             f"{int(z.shape[0])} atomic numbers vs {int(arr.shape[1])} atoms")
    masses = np.asarray([atomic_mass_v52(int(v)) for v in z], dtype=np.float64)
    m_sum = float(masses.sum())
    if not (m_sum > 0.0):
        raise KeyAtomDatasetV3Error("ARC_MASS_ZERO", "total mass is zero")
    if arr.shape[0] == 1:
        return np.zeros(1, dtype=np.float64)
    delta = np.diff(arr, axis=0)                       # [N-1, A, 3]
    sq = (delta * delta).sum(axis=-1)                  # [N-1, A]
    ds = np.sqrt((sq * masses[None, :]).sum(axis=1) / m_sum)
    return np.concatenate(([0.0], np.cumsum(ds)))


def progress_q_from_arcs_v52(
    arc_s: np.ndarray,
    ts_index: int,
    *,
    zero_arc_eps: float = ZERO_ARC_EPS_ANGSTROM_V52,
) -> tuple[np.ndarray | None, str]:
    """Frozen q math (plan §5.2): A=−1, TS=0, B=+1, zero-arc ⇒ masked.

    ``arc_s`` is indexed by PATH position (A→B order); ``ts_index`` is the
    path position of the TS frame.  Returns ``(q, "")`` or ``(None, reason)``
    where reason ∈ {``zero_arc_a_ts``, ``zero_arc_ts_b``} — the caller MUST
    mask (never divide by zero).
    """
    s = np.asarray(arc_s, dtype=np.float64).reshape(-1)
    ts = int(ts_index)
    if not (0 <= ts < int(s.shape[0])):
        raise KeyAtomDatasetV3Error(
            "Q_TS_INDEX", f"ts_index {ts} outside 0..{int(s.shape[0]) - 1}")
    s_a, s_ts, s_b = float(s[0]), float(s[ts]), float(s[-1])
    left, right = s_ts - s_a, s_b - s_ts
    if left <= float(zero_arc_eps):
        return None, "zero_arc_a_ts"
    if right <= float(zero_arc_eps):
        return None, "zero_arc_ts_b"
    idx = np.arange(int(s.shape[0]), dtype=np.float64)
    ts_f = float(ts)
    q = np.where(
        idx <= ts_f,
        -(s_ts - s) / left,
        (s - s_ts) / right)
    return q.astype(np.float64), ""


def kabsch_rmsd_v52(
    mobile: np.ndarray,
    target: np.ndarray,
) -> float:
    """RMSD after LEGAL rigid alignment (proper rotation, det=+1; no reflection).

    ``d_key_TS`` building block (plan §5.3: 合法刚体对齐后当前 K 几何与 TS
    的差距).  Centroids are removed, the optimal proper rotation comes from
    the SVD of the covariance with the reflection-fix ``d ← sign(det(VUᵀ))``,
    and the returned RMSD is ``sqrt(mean_a ‖R m_a − t_a‖²)`` over the K atoms.
    """
    m = np.asarray(mobile, dtype=np.float64).reshape(-1, 3)
    t = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if m.shape != t.shape or m.shape[0] == 0:
        raise KeyAtomDatasetV3Error(
            "KABSCH_SHAPE", f"mobile {m.shape} vs target {t.shape}")
    mc = m - m.mean(axis=0)
    tc = t - t.mean(axis=0)
    if m.shape[0] == 1:
        # single atom: alignment is translation-only by construction
        diff = mc - tc
        return float(np.sqrt((diff * diff).sum()))
    cov = mc.T @ tc
    u_mat, _s, vt = np.linalg.svd(cov)
    d_sign = np.sign(np.linalg.det(u_mat @ vt))
    if d_sign == 0.0:
        d_sign = 1.0
    correction = np.diag([1.0, 1.0, d_sign])
    rot = u_mat @ correction @ vt
    diff = mc @ rot - tc
    return float(np.sqrt((diff * diff).sum(axis=1).mean()))


def stop_label_from_distance_v52(
    d_key_ts: float,
    *,
    positive_window: float = STOP_POSITIVE_WINDOW_ANGSTROM_V52,
    negative_window: float = STOP_NEGATIVE_WINDOW_ANGSTROM_V52,
) -> int:
    """Frozen three-way stop label: +1 (near-TS), 0 (far), −1 (Mask).

    ``d ≤ positive_window`` ⇒ 1; ``d ≥ negative_window`` ⇒ 0; otherwise the
    middle region is MASKED (campaign ``v52_training.stop_labels``; windows
    are inclusive at both edges; 禁止偷扩窗 — widening either window is a
    contract violation, not a knob).
    """
    if not (float(positive_window) <= float(negative_window)):
        raise KeyAtomDatasetV3Error(
            "STOP_WINDOW_ORDER",
            f"positive window {positive_window} > negative {negative_window}")
    d = float(d_key_ts)
    if d <= float(positive_window):
        return 1
    if d >= float(negative_window):
        return 0
    return -1


@dataclass(frozen=True)
class CurveV52Labels:
    """Per-curve E4 label backbone (intent-independent geometry labels)."""

    curve_id: str
    ts_index: int
    #: cumulative mass-weighted arc length in A→B PATH order [N]
    arc_s: tuple[float, ...]
    #: canonical q per node in A→B path order (None when zero-arc masked)
    q_canonical: tuple[float, ...] | None
    q_zero_arc_reason: str
    #: path-order → node_index mapping (identity unless endpoints reversed)
    node_order: tuple[int, ...]


@dataclass(frozen=True)
class V52SampleLabel:
    """One sample row's E4 labels (intent-signed q; three-way stop)."""

    sample_id: str
    curve_id: str
    node_index: int
    intent_id: str
    direction: str
    fold: int | None
    label_status: str
    #: intent-signed progress label (A→B canonical q, negated for reverse)
    q: float | None
    #: True when q is usable (False ⇒ zero-arc / curve masked)
    q_mask: bool
    #: +1 near-TS / 0 far / −1 Mask (middle window)
    stop: int
    #: aligned K-geometry RMSD to TS (Å); provenance for the stop label
    d_key_ts: float

    @property
    def stop_mask(self) -> bool:
        """True when the row participates in the TSStop loss denominator."""
        return self.stop in (0, 1)


@dataclass(frozen=True)
class V52LabelTable:
    """E4 label table over a dataset load (keyed by ``sample_id``)."""

    by_sample: Mapping[str, V52SampleLabel]
    curves: Mapping[str, CurveV52Labels]

    @property
    def zero_arc_curves(self) -> tuple[str, ...]:
        return tuple(sorted(
            cid for cid, c in self.curves.items() if c.q_canonical is None))

    def q_masked_sample_ids(self) -> tuple[str, ...]:
        return tuple(sorted(
            sid for sid, lab in self.by_sample.items() if not lab.q_mask))


def build_v52_labels(
    dataset: "KeyAtomDatasetV3",
    *,
    stop_positive_window: float = STOP_POSITIVE_WINDOW_ANGSTROM_V52,
    stop_negative_window: float = STOP_NEGATIVE_WINDOW_ANGSTROM_V52,
    zero_arc_eps: float = ZERO_ARC_EPS_ANGSTROM_V52,
) -> V52LabelTable:
    """E4 core: derive q + TSStop labels from the six-table teacher columns.

    Provenance discipline: ONLY ``irc_curve`` (``ts_index``/endpoints),
    ``irc_node`` (geometry + ``atomic_numbers``), ``sample_view`` (identity,
    status, fold, weights) and ``reaction_intent`` (direction sign + K set)
    are read — no ground-truth access anywhere in this path.

    * **q**: mass-weighted arc-length segmentation
      (:func:`mass_weighted_arc_lengths_v52` + :func:`progress_q_from_arcs_v52`)
      normalized A=−1 / TS=0 / B=+1 on the canonical A→B path; the reverse
      intent label is exactly ``−q`` (frozen yaml ``qlabel.reverse q=-q``);
      zero-arc curves are MASKED (``q=None``), never divided by zero;
    * **stop**: ``d_key_TS`` = Kabsch-aligned K-geometry RMSD to the TS
      frame (per node, computed ONCE per curve — fwd/rev intents share the
      K set); three-way window label with Mask in the middle;
    * every loaded row gets labels (including non-``ok`` rows) so the G3a
      coverage report can account for the FULL denominator; the training
      layer applies its own ``label_status``/fold filters on top.
    """
    curves: dict[str, CurveV52Labels] = {}
    # (curve_id, node_index) → d_key_ts cache (intent-independent)
    d_key_cache: dict[tuple[str, int], float] = {}
    labels: dict[str, V52SampleLabel] = {}

    for rec in dataset.samples:
        cid = rec.curve_id
        curve_labels = curves.get(cid)
        if curve_labels is None:
            a_idx, b_idx = dataset.curve_endpoints(cid)
            curve_row = dataset._curves[cid]
            n_nodes = (int(curve_row["n_nodes"]) if curve_row.get("n_nodes")
                       else max(key[1] for key in dataset._frame_xyz
                                if key[0] == cid) + 1)
            ascending = list(range(n_nodes))
            node_order = (list(reversed(ascending))
                          if a_idx > b_idx else ascending)
            frames = np.stack([
                np.asarray(dataset.frame_xyz(cid, node), dtype=np.float64)
                for node in node_order])
            atomic = rec.atomic_numbers
            arc_s = mass_weighted_arc_lengths_v52(frames, atomic)
            ts_node = dataset.curve_ts_index(cid)
            # ts_node is a node_index; find its PATH position
            try:
                ts_pos = node_order.index(int(ts_node))
            except ValueError as exc:
                raise KeyAtomDatasetV3Error(
                    "Q_TS_NODE", f"curve {cid}: ts_index {ts_node} is not a "
                    "node of the curve") from exc
            q_path, reason = progress_q_from_arcs_v52(
                arc_s, ts_pos, zero_arc_eps=zero_arc_eps)
            curve_labels = CurveV52Labels(
                curve_id=cid,
                ts_index=int(ts_node),
                arc_s=tuple(float(v) for v in arc_s),
                q_canonical=(tuple(float(v) for v in q_path)
                             if q_path is not None else None),
                q_zero_arc_reason=reason,
                node_order=tuple(int(v) for v in node_order))
            curves[cid] = curve_labels

        # --- d_key_TS (cached per (curve, node); K = intent key set) --------
        cache_key = (cid, int(rec.node_index))
        d_key = d_key_cache.get(cache_key)
        if d_key is None:
            keys = np.asarray(
                [int(k) for k in rec.intent.key_rows()], dtype=np.int64)
            mobile = np.asarray(
                dataset.frame_xyz(cid, rec.node_index), dtype=np.float64)[keys]
            target = np.asarray(
                dataset.frame_xyz(cid, curve_labels.ts_index),
                dtype=np.float64)[keys]
            d_key = kabsch_rmsd_v52(mobile, target)
            d_key_cache[cache_key] = d_key

        # --- canonical q for this node (path order; reverse ⇒ −q) ----------
        q: float | None = None
        q_valid = curve_labels.q_canonical is not None
        if q_valid:
            try:
                pos = curve_labels.node_order.index(int(rec.node_index))
            except ValueError as exc:
                raise KeyAtomDatasetV3Error(
                    "Q_NODE_ORDER", f"curve {cid}: node {rec.node_index} not "
                    "in the curve's path order") from exc
            q = float(curve_labels.q_canonical[pos])
            if rec.intent.direction == "reverse":
                q = -q
        labels[rec.sample_id] = V52SampleLabel(
            sample_id=rec.sample_id,
            curve_id=cid,
            node_index=int(rec.node_index),
            intent_id=rec.intent.intent_id,
            direction=rec.intent.direction,
            fold=rec.fold,
            label_status=rec.label_status,
            q=q,
            q_mask=q_valid,
            stop=stop_label_from_distance_v52(
                d_key,
                positive_window=stop_positive_window,
                negative_window=stop_negative_window),
            d_key_ts=float(d_key))

    return V52LabelTable(by_sample=labels, curves=curves)


# ---------------------------------------------------------------------------
# V5.3-MP S0-D (plan §4.4): sharded lazy dataset + bounded per-curve cache
# ---------------------------------------------------------------------------
#: Shard index filename written by :func:`write_geometry_shards` and read
#: by :class:`ShardedKeyAtomDatasetV3` (plan §4.4 分片数据集 + 惰性索引).
SHARD_INDEX_FILENAME: Final[str] = "shard_index_v53mp.json"
#: Schema id of the shard index document.
SHARD_INDEX_VERSION: Final[str] = "v53mp_shard_index_v1"

#: Light-index cache files (V5.3-MP training-loop accelerator): the light
#: init scan (irc_node + sample_view) is a pure function of the frozen
#: tables; the cache is bound to the data-manifest + split-manifest shas and
#: never served when either drifts (fail-closed; see
#: :class:`ShardedKeyAtomDatasetV3` ``light_cache``).
LIGHT_CACHE_PKL: Final[str] = ".light_index_v53mp.pkl"
LIGHT_CACHE_JSON: Final[str] = ".light_index_v53mp.json"
LIGHT_CACHE_SCHEMA: Final[str] = "v53mp_light_index_v1"


@dataclass(frozen=True)
class SampleRecordV3Lite:
    """Light sample record for the sharded loader (no frame geometry).

    Carries everything supervision/sampling need — identity, intent,
    weights, status, fold, teacher target — but NOT the per-frame
    coordinate payload; geometry is fetched on demand via
    :meth:`ShardedKeyAtomDatasetV3.frame_xyz` /
    :meth:`ShardedKeyAtomDatasetV3.to_sample_v3`.  ``atomic_numbers`` is
    a per-curve SHARED tuple (one object per curve, referenced by every
    row), so it costs a pointer per record, not a copy.
    """

    sample_id: str
    curve_id: str
    node_index: int
    intent: IntentRecordV3
    d0: float
    label_weight: float
    curve_weight: float
    sample_weight: float
    label_status: str
    atomic_numbers: tuple[int, ...]
    target_action_u: tuple[tuple[float, float, float], ...] | None
    target_node_index: int | None
    fold: int | None


class ShardedKeyAtomDatasetV3:
    """V5.3-MP S0-D lazy six-table dataset with a bounded LRU frame cache.

    Reads the SAME PD-1 six tables as :class:`KeyAtomDatasetV3`, but at
    init parses only LIGHT columns (no ``coords_flat``/JSON geometry
    payloads of ``irc_node``) into an in-memory sample index; per-curve
    geometry is loaded on first access and kept in an LRU cache bounded
    to ``cache_curves`` curves (plan §4.4: 惰性加载 + LRU 缓存上限可配置).

    **Memory bound** (plan §4.4 hard-gate discipline): resident memory is
    ~ ``#samples × small light record`` (identity/weights/intent
    references — no coordinates) + ``#curves × (node-index set +
    atomic-numbers tuple)`` + ``cache_curves × per-curve frame arrays``.
    ``irc_edge``/``curve_quality`` payloads are checked for
    presence/columns and then discarded — never retained.

    Geometry source per curve (checked in this order):

    1. ``shard_index_v53mp.json`` fast path — when present, per-curve
       ``[n_nodes, n_atoms, 3]`` float64 arrays come from
       ``np.load(shard)`` (random access; the 20k-scale route);
    2. CSV fallback — ``irc_node.csv`` is streamed once per cache miss
       and only the requested curve's rows are parsed (correctness
       fallback for deliveries without shards).

    Curve-weight verification (Σ curve_weight == 1.0 per (curve, d0)
    over the FULL pre-filter group) and the intent-pair involution
    (T0.2) checks run at init exactly as in the full loader; manifests
    are verified fail-closed when present.  Frame-coordinate validity
    (finiteness/shape) is checked lazily at first geometry access
    (:meth:`frame_xyz` / :meth:`to_sample_v3`).
    """

    def __init__(
        self,
        tables_dir: str | Path,
        *,
        d0: float = D0_PRIMARY_ANGSTROM,
        label_statuses: Iterable[str] = ("ok", "terminal", "masked",
                                         "conflict"),
        cache_curves: int = 512,
        verify_manifest: bool = True,
        split_manifest: str | Path | None = None,
        fold: int | None = None,
        directions: Iterable[str] = ("forward", "reverse"),
        light_cache: bool = False,
    ) -> None:
        self.tables_dir = Path(tables_dir)
        self.d0 = float(d0)
        self.fold = fold
        self.label_statuses = frozenset(label_statuses)
        self.directions = frozenset(directions)
        self.cache_curves = int(cache_curves)
        self.verify_manifest = bool(verify_manifest)
        self.light_cache = bool(light_cache)
        self._split_manifest_input = split_manifest
        _require(self.tables_dir.is_dir(), "TABLES_DIR",
                 f"{self.tables_dir} is not a directory")
        _require(self.cache_curves >= 1, "CACHE_CURVES",
                 f"cache_curves must be >= 1, got {self.cache_curves}")

        # --- presence of all six tables (PD-1; curve_quality provenance) ---
        for table, filename in TABLE_FILENAMES.items():
            _require((self.tables_dir / filename).is_file(),
                     "TABLE_MISSING", f"{table} -> {filename} not found")

        self.manifest_sha256 = ""
        if verify_manifest:
            self.manifest_sha256 = _verify_data_manifest_at(self.tables_dir)

        # --- light: irc_curve (small) ----------------------------------------
        curve_rows = _read_csv(self.tables_dir / TABLE_FILENAMES["irc_curve"])
        _check_columns("irc_curve", curve_rows)
        self._curves: dict[str, dict[str, str]] = {
            r["curve_id"]: r for r in curve_rows}
        _require(len(self._curves) == len(curve_rows), "CURVE_ID_DUP",
                 "duplicate curve_id in irc_curve")

        # --- light: reaction_intent (small) ----------------------------------
        intent_rows = _read_csv(
            self.tables_dir / TABLE_FILENAMES["reaction_intent"])
        _check_columns("reaction_intent", intent_rows)
        self._intents: dict[tuple[str, str], IntentRecordV3] = {}
        for row in intent_rows:
            try:
                intent = IntentRecordV3.from_table_row(row)
            except (KeyAtomV3ContractError, ValueError,
                    json.JSONDecodeError) as exc:
                raise KeyAtomDatasetV3Error(
                    "INTENT_PARSE", f"{row.get('intent_id')}: {exc}") from exc
            key = (intent.curve_id, intent.intent_id)
            _require(key not in self._intents, "INTENT_DUP",
                     f"duplicate (curve_id, intent_id) {key}")
            _require(intent.curve_id in self._curves, "INTENT_FK",
                     f"intent {intent.intent_id} references unknown curve")
            self._intents[key] = intent

        # --- presence-only tables (payloads are NEVER retained) --------------
        self._check_header_only("irc_edge")
        self._check_header_only("curve_quality")

        # --- folds + shard fast-path (cheap; loaded before the cacheable
        # --- section so the cache key can bind them) --------------------------
        self._folds = _load_folds_at(self.tables_dir, split_manifest, fold)

        self._shard_curve_path: dict[str, str] = {}
        self._shard_sha: dict[str, str] = {}
        self._shard_verified: set[str] = set()
        self._load_shard_index()

        # --- light state: irc_node scan + sample index (optionally cached) ---
        # ``light_cache=True`` (V5.3-MP training loop): the light init scan is
        # a pure function of the frozen tables; the cache is bound to the
        # data-manifest + split-manifest content shas and never served when
        # either drifts (fail-closed).  The invariant checks below re-run
        # over the cached state as well (never skipped).
        state = self._try_load_light_cache() if self.light_cache else None
        if state is not None:
            (self._node_index, self._curve_atomic, self._curve_energy,
             self._samples, weight_totals) = state
        else:
            # --- light scan of irc_node: node sets, atomic numbers, energies -
            # Geometry payloads are read past, never parsed or retained here.
            self._node_index: dict[str, set[int]] = {}
            self._curve_atomic: dict[str, tuple[int, ...]] = {}
            self._curve_energy: dict[str, dict[int, float]] = {}
            self._scan_irc_node_light()

            # --- light sample index (no geometry) ----------------------------
            self._samples: list[SampleRecordV3Lite] = []
            weight_totals: dict[str, float] = {}
            path = self.tables_dir / TABLE_FILENAMES["sample_view"]
            with path.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                _require_columns_in("sample_view", reader.fieldnames)
                for row in reader:
                    # ORA-2 #3 / T0.1a invariant is defined over the FULL
                    # (curve_id, d0) group — accumulated BEFORE any filtering.
                    try:
                        if abs(float(row["d0"]) - self.d0) <= 1e-12:
                            cid = str(row["curve_id"])
                            weight_totals[cid] = (
                                weight_totals.get(cid, 0.0)
                                + float(row["curve_weight"]))
                    except (KeyError, ValueError) as exc:
                        raise KeyAtomDatasetV3Error(
                            "SAMPLE_WEIGHT_PARSE",
                            f"{row.get('sample_id')}: {exc}",
                        ) from exc
                    rec = self._parse_sample_lite(row, self._folds)
                    if rec is None:
                        continue
                    self._samples.append(rec)
            if self.light_cache:
                self._write_light_cache(weight_totals)
        _require(len(self._samples) > 0, "NO_SAMPLES",
                 "no sample_view rows survive the dataset filters "
                 f"(d0={self.d0}, statuses={sorted(self.label_statuses)}, "
                 f"fold={fold})")
        for curve_id, total in weight_totals.items():
            _require(abs(total - 1.0) <= WEIGHT_SUM_TOL, "CURVE_WEIGHT_SUM",
                     f"curve {curve_id}: sum(curve_weight)={total!r} != 1.0")
        for (curve_id, intent_id), intent in self._intents.items():
            if intent.intent_underdetermined:
                continue
            rev = self._intents.get((curve_id, intent.reverse_of_intent_id))
            _require(rev is not None, "INTENT_REVERSE_MISSING",
                     f"intent {intent_id} has no reverse row "
                     f"{intent.reverse_of_intent_id!r}")
            _require(rev.reverse_of_intent_id == intent_id, "INTENT_INVOLUTION",
                     f"reverse(reverse({intent_id})) != {intent_id}")

        # --- geometry LRU (EMPTY at init — laziness is testable) -------------
        self._frames: "OrderedDict[str, dict[int, np.ndarray]]" = OrderedDict()
        self._v52_cache: dict[tuple[float, float, float], V52LabelTable] = {}

    # -- light table helpers ---------------------------------------------------
    def _check_header_only(self, table: str) -> None:
        """Presence + required-column check without retaining any row."""
        path = self.tables_dir / TABLE_FILENAMES[table]
        if not path.is_file():
            raise KeyAtomDatasetV3Error("TABLE_MISSING", f"{path} not found")
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            _require_columns_in(table, reader.fieldnames)

    # -- light-index cache (V5.3-MP training-loop accelerator) ---------------
    def _light_cache_binding(self) -> dict:
        """Content identity the cache is bound to (fail-closed on drift)."""
        split_input = self._split_manifest_input
        split_path = (Path(split_input) if split_input is not None
                      else self.tables_dir / "split_manifest_v53mp.json")
        return {
            "schema": LIGHT_CACHE_SCHEMA,
            "data_manifest_sha256": self.manifest_sha256,
            "split_manifest_sha256": (_sha256(split_path)
                                      if split_path.is_file() else ""),
            "d0": float(self.d0),
            "label_statuses": sorted(self.label_statuses),
            "directions": sorted(self.directions),
            "fold": self.fold,
        }

    def _try_load_light_cache(self):
        """Load the cached light state iff the binding matches exactly."""
        json_path = self.tables_dir / LIGHT_CACHE_JSON
        pkl_path = self.tables_dir / LIGHT_CACHE_PKL
        if not json_path.is_file() or not pkl_path.is_file():
            return None
        try:
            doc = json.loads(json_path.read_text(encoding="utf-8"))
            if doc.get("binding") != self._light_cache_binding():
                return None
            import pickle
            with pkl_path.open("rb") as fh:
                state = pickle.load(fh)
        except Exception:  # noqa: BLE001 - any cache error falls back to CSV
            return None
        keys = ("node_index", "curve_atomic", "curve_energy", "samples",
                "weight_totals")
        if not isinstance(state, dict) or any(k not in state for k in keys):
            return None
        return (state["node_index"], state["curve_atomic"],
                state["curve_energy"], state["samples"],
                state["weight_totals"])

    def _write_light_cache(self, weight_totals: Mapping[str, float]) -> None:
        """Persist the light state + binding (best-effort; never fatal)."""
        import pickle
        state = {
            "node_index": self._node_index,
            "curve_atomic": self._curve_atomic,
            "curve_energy": self._curve_energy,
            "samples": self._samples,
            "weight_totals": dict(weight_totals),
        }
        doc = {"binding": self._light_cache_binding(),
               "n_samples": len(self._samples)}
        try:
            with (self.tables_dir / LIGHT_CACHE_PKL).open("wb") as fh:
                pickle.dump(state, fh, protocol=4)
            (self.tables_dir / LIGHT_CACHE_JSON).write_text(
                json.dumps(doc, indent=1, sort_keys=True) + "\n",
                encoding="utf-8")
        except OSError:
            pass  # read-only tables dirs degrade to the CSV path

    def _scan_irc_node_light(self) -> None:
        """One streaming pass over ``irc_node.csv`` keeping only light state.

        Per row: node-set membership (FK checks), the curve's shared
        ``atomic_numbers`` tuple, and — when the (20k-build) ``energy``
        column is populated — the per-node energy.  Coordinates are never
        parsed or retained here (deferred to the LRU loader).
        """
        path = self.tables_dir / TABLE_FILENAMES["irc_node"]
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            _require_columns_in("irc_node", reader.fieldnames)
            has_energy = "energy" in (reader.fieldnames or ())
            for row in reader:
                key = (row["curve_id"], int(row["node_index"]))
                nodes = self._node_index.setdefault(key[0], set())
                _require(key[1] not in nodes, "NODE_DUP",
                         f"duplicate node row {key}")
                nodes.add(key[1])
                payload = row.get("atomic_numbers") or ""
                if payload and key[0] not in self._curve_atomic:
                    try:
                        self._curve_atomic[key[0]] = tuple(
                            int(v) for v in json.loads(payload))
                    except (ValueError, json.JSONDecodeError) as exc:
                        raise KeyAtomDatasetV3Error(
                            "ATOMIC_NUMBERS_PARSE",
                            f"{row['curve_id']}: {exc}") from exc
                if has_energy:
                    cell = (row.get("energy") or "").strip()
                    if cell:
                        try:
                            self._curve_energy.setdefault(
                                key[0], {})[key[1]] = float(cell)
                        except ValueError as exc:
                            raise KeyAtomDatasetV3Error(
                                "NODE_ENERGY_PARSE",
                                f"{key}: energy {cell!r}") from exc

    def _parse_sample_lite(
        self,
        row: Mapping[str, str],
        folds: Mapping[str, int],
    ) -> SampleRecordV3Lite | None:
        """Mirror of ``KeyAtomDatasetV3._parse_sample`` without geometry."""
        try:
            d0 = float(row["d0"])
        except ValueError as exc:
            raise KeyAtomDatasetV3Error(
                "SAMPLE_D0_PARSE", f"{row.get('sample_id')}: {exc}") from exc
        if abs(d0 - self.d0) > 1e-12:
            return None                 # non-primary d0 rows are out of scope
        status = str(row.get("label_status", ""))
        if status not in self.label_statuses:
            return None
        curve_id = row["curve_id"]
        node_index = int(row["node_index"])
        intent_key = (curve_id, row["intent_id"])
        intent = self._intents.get(intent_key)
        _require(intent is not None, "SAMPLE_INTENT_FK",
                 f"sample {row.get('sample_id')} references unknown intent "
                 f"{intent_key}")
        _require(intent.direction in self.directions, "SAMPLE_DIRECTION",
                 f"sample {row.get('sample_id')} direction "
                 f"{intent.direction} filtered out")
        _require(node_index in self._node_index.get(curve_id, ()),
                 "SAMPLE_NODE_FK",
                 f"sample {row.get('sample_id')} references unknown node "
                 f"({curve_id}, {node_index})")
        if folds:
            curve_fold = folds.get(curve_id)
            _require(curve_fold is not None, "SPLIT_CURVE_MISSING",
                     f"curve {curve_id} missing from split manifest")
            if self.fold is not None and int(curve_fold) != int(self.fold):
                return None
        atomic = self._curve_atomic.get(curve_id) or intent.g_atoms
        _require(bool(atomic), "ATOMIC_NUMBERS_MISSING",
                 f"curve {curve_id} has no atomic_numbers payload (irc_node "
                 "nor endpoint-graph atoms)")

        target_u = None
        target_node = None
        payload = (row.get("target_action_u") or "").strip()
        if payload:
            try:
                doc_u = json.loads(payload)
                flat = [float(v) for v in doc_u] if (
                    doc_u and isinstance(doc_u[0], (int, float))) else None
                if flat is not None:
                    if len(flat) % 3 != 0:
                        raise ValueError(
                            f"flat target_action_u length {len(flat)} % 3 != 0")
                    target_u = tuple(
                        (flat[k], flat[k + 1], flat[k + 2])
                        for k in range(0, len(flat), 3))
                else:
                    target_u = tuple(
                        (float(v[0]), float(v[1]), float(v[2]))
                        for v in doc_u)
            except (ValueError, json.JSONDecodeError, IndexError) as exc:
                raise KeyAtomDatasetV3Error(
                    "TARGET_U_PARSE",
                    f"{row.get('sample_id')}: {exc}") from exc
            _require(len(target_u) == len(intent.key_rows()), "TARGET_U_SHAPE",
                     f"{row.get('sample_id')}: target_action_u has "
                     f"{len(target_u)} rows != |K|={len(intent.key_rows())}")
        tn = (row.get("target_node_index") or "").strip()
        if tn:
            target_node = int(tn)
        fold_value = (row.get("fold") or "").strip()
        return SampleRecordV3Lite(
            sample_id=str(row["sample_id"]),
            curve_id=curve_id,
            node_index=node_index,
            intent=intent,
            d0=d0,
            label_weight=float(row["label_weight"]),
            curve_weight=float(row["curve_weight"]),
            sample_weight=float(row["sample_weight"]),
            label_status=status,
            atomic_numbers=atomic,
            target_action_u=target_u,
            target_node_index=target_node,
            fold=int(fold_value) if fold_value else folds.get(curve_id),
        )

    # -- shard fast path --------------------------------------------------------
    def _load_shard_index(self) -> None:
        """Parse ``shard_index_v53mp.json`` when present (fail-closed)."""
        index_path = self.tables_dir / SHARD_INDEX_FILENAME
        if not index_path.is_file():
            return                          # CSV fallback for every curve
        doc = json.loads(index_path.read_text(encoding="utf-8"))
        _require(str(doc.get("version")) == SHARD_INDEX_VERSION,
                 "SHARD_INDEX_VERSION",
                 f"{SHARD_INDEX_FILENAME} has version "
                 f"{doc.get('version')!r}, expected {SHARD_INDEX_VERSION!r}")
        shards = doc.get("shards")
        _require(isinstance(shards, list), "SHARD_INDEX_SHAPE",
                 f"{SHARD_INDEX_FILENAME}: 'shards' must be a list")
        for entry in shards:
            rel = str(entry.get("path") or "")
            fmt = str(entry.get("format") or "")
            _require(bool(rel) and fmt == "npz", "SHARD_INDEX_SHAPE",
                     f"bad shard entry {entry!r}")
            _require((self.tables_dir / rel).is_file(), "SHARD_FILE",
                     f"shard file {rel} listed in {SHARD_INDEX_FILENAME} "
                     "not found")
            self._shard_sha[rel] = str(entry.get("sha256") or "")
            for cid in entry.get("curve_ids", ()):
                _require(str(cid) not in self._shard_curve_path,
                         "SHARD_CURVE_DUP",
                         f"curve {cid} appears in more than one shard")
                self._shard_curve_path[str(cid)] = rel

    def _load_curve_frames_shard(
        self, curve_id: str, shard_path: Path,
    ) -> dict[int, np.ndarray]:
        rel = shard_path.name
        if self.verify_manifest and rel not in self._shard_verified:
            expected = self._shard_sha.get(rel, "")
            actual = _sha256(shard_path)
            _require(bool(expected) and actual == expected, "SHARD_SHA",
                     f"{rel}: index sha256 {expected[:12]}… != actual "
                     f"{actual[:12]}…")
            self._shard_verified.add(rel)
        try:
            with np.load(shard_path, allow_pickle=False) as data:
                coords_key = f"{curve_id}::coords"
                nodes_key = f"{curve_id}::node_indices"
                _require(
                    coords_key in data.files and nodes_key in data.files,
                    "SHARD_CURVE_MISSING",
                    f"shard {rel} has no geometry for curve {curve_id}")
                coords = np.asarray(data[coords_key], dtype=np.float64)
                node_indices = [int(v) for v in data[nodes_key]]
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            raise KeyAtomDatasetV3Error(
                "SHARD_READ", f"{rel}:{curve_id}: {exc}") from exc
        _require(coords.ndim == 3 and coords.shape[-1] == 3
                 and coords.shape[0] == len(node_indices),
                 "SHARD_COORDS_SHAPE",
                 f"{rel}:{curve_id}: coords {tuple(coords.shape)} vs "
                 f"{len(node_indices)} node indices")
        expected_nodes = self._node_index.get(curve_id, set())
        _require(set(node_indices) == expected_nodes, "SHARD_NODE_SET",
                 f"{rel}:{curve_id}: shard nodes {sorted(node_indices)} != "
                 f"irc_node nodes {sorted(expected_nodes)}")
        return {node: coords[pos]
                for pos, node in enumerate(node_indices)}

    def _load_curve_frames_csv(
        self, curve_id: str,
    ) -> dict[int, np.ndarray]:
        """CSV fallback: one streaming pass, parsing ONLY this curve."""
        path = self.tables_dir / TABLE_FILENAMES["irc_node"]
        frames: dict[int, np.ndarray] = {}
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row["curve_id"] != curve_id:
                    continue
                node = int(row["node_index"])
                _require(node not in frames, "NODE_DUP",
                         f"duplicate node row ({curve_id}, {node})")
                frames[node] = KeyAtomDatasetV3._parse_frame_xyz(row)
        return frames

    def _load_curve_frames(self, curve_id: str) -> dict[int, np.ndarray]:
        rel = self._shard_curve_path.get(curve_id)
        if rel is not None:
            return self._load_curve_frames_shard(curve_id,
                                                 self.tables_dir / rel)
        return self._load_curve_frames_csv(curve_id)

    def _ensure_curve(self, curve_id: str) -> dict[int, np.ndarray]:
        """Load a curve's geometry ONCE; serve from the bounded LRU."""
        frames = self._frames.get(curve_id)
        if frames is not None:
            self._frames.move_to_end(curve_id)
            return frames
        frames = self._load_curve_frames(curve_id)
        self._frames[curve_id] = frames
        while len(self._frames) > self.cache_curves:
            self._frames.popitem(last=False)
        return frames

    # -- public API (mirrors KeyAtomDatasetV3) ----------------------------------
    def __len__(self) -> int:
        return len(self._samples)

    @property
    def samples(self) -> tuple[SampleRecordV3Lite, ...]:
        return tuple(self._samples)

    @property
    def intents(self) -> Mapping[tuple[str, str], IntentRecordV3]:
        return dict(self._intents)

    @property
    def folds(self) -> Mapping[str, int]:
        return dict(self._folds)

    @property
    def n_cached_curves(self) -> int:
        """Curves currently resident in the geometry LRU (0 right after init)."""
        return len(self._frames)

    @property
    def cached_curve_ids(self) -> tuple[str, ...]:
        """LRU-resident curve ids (oldest first) — monitoring/tests."""
        return tuple(self._frames.keys())

    @property
    def uses_shards(self) -> bool:
        """True when a valid ``shard_index_v53mp.json`` mapped any curve."""
        return bool(self._shard_curve_path)

    @property
    def _frame_xyz(self) -> dict[tuple[str, int], np.ndarray]:
        """Live view of the geometry cache (``build_v52_labels`` compat)."""
        return {(cid, node): arr
                for cid, frames in self._frames.items()
                for node, arr in frames.items()}

    def frame_xyz(self, curve_id: str, node_index: int) -> tuple:
        """Full ``[N,3]`` current-frame geometry (lazy; LRU-cached)."""
        frames = self._ensure_curve(str(curve_id))
        arr = frames.get(int(node_index))
        if arr is None:
            raise KeyAtomDatasetV3Error(
                "NODE_UNKNOWN", f"curve {curve_id} has no node {node_index}")
        return tuple(tuple(float(v) for v in row) for row in arr)

    def curve_ts_index(self, curve_id: str) -> int:
        """Trusted TS frame index of the curve (irc_curve.ts_index, PD-9)."""
        curve = self._curves.get(str(curve_id))
        if curve is None:
            raise KeyAtomDatasetV3Error(
                "CURVE_UNKNOWN", f"curve {curve_id} not in irc_curve")
        return int(curve["ts_index"])

    def curve_endpoints(self, curve_id: str) -> tuple[int, int]:
        """``(endpoint_a_index, endpoint_b_index)`` of the canonical A→B path."""
        curve = self._curves.get(str(curve_id))
        if curve is None:
            raise KeyAtomDatasetV3Error(
                "CURVE_UNKNOWN", f"curve {curve_id} not in irc_curve")
        a = int(curve.get("endpoint_a_index", 0) or 0)
        b = int(curve.get("endpoint_b_index", "") or
                (int(curve["n_nodes"]) - 1))
        return a, b

    def curve_energies(self, curve_id: str) -> dict[int, float] | None:
        """Per-node energies (Hartree), or None unless COMPLETE for the curve.

        Consumed by the ``endpoint_label_spec_v1`` diagnostic energy flag
        (V5.3-MP plan §4.3).  Partial/empty coverage ⇒ None — diagnostics
        must never fail closed on missing columns.
        """
        energies = self._curve_energy.get(str(curve_id))
        if not energies:
            return None
        nodes = self._node_index.get(str(curve_id), set())
        if set(energies) != nodes:
            return None
        return dict(energies)

    def v52_labels(
        self,
        *,
        stop_positive_window: float = STOP_POSITIVE_WINDOW_ANGSTROM_V52,
        stop_negative_window: float = STOP_NEGATIVE_WINDOW_ANGSTROM_V52,
        zero_arc_eps: float = ZERO_ARC_EPS_ANGSTROM_V52,
    ) -> "V52LabelTable":
        """V5.2 q/TSStop labels computed from lazily loaded geometry.

        Uses the same frozen :func:`build_v52_labels` math as the full
        loader; the per-curve frames pulled in by the computation are
        served (and bounded) by the same LRU.  The resulting table is
        cached per window triple.
        """
        key = (float(stop_positive_window), float(stop_negative_window),
               float(zero_arc_eps))
        table = self._v52_cache.get(key)
        if table is None:
            table = build_v52_labels(
                self,
                stop_positive_window=stop_positive_window,
                stop_negative_window=stop_negative_window,
                zero_arc_eps=zero_arc_eps)
            self._v52_cache[key] = table
        return table

    def records(
        self, indices: Iterable[int] | None = None,
    ) -> list[SampleRecordV3Lite]:
        if indices is None:
            return list(self._samples)
        return [self._samples[int(i)] for i in indices]

    def to_sample_v3(self, rec: SampleRecordV3Lite) -> KeyAtomSampleV3:
        """Assemble the full v3 sample, fetching geometry on demand.

        Mirrors ``KeyAtomDatasetV3.to_sample_v3`` (``frame_weight``
        carries ``label_weight`` exactly once — 30h plan §4.1).
        """
        frames = self._ensure_curve(rec.curve_id)
        arr = frames.get(int(rec.node_index))
        if arr is None:
            raise KeyAtomDatasetV3Error(
                "NODE_UNKNOWN",
                f"curve {rec.curve_id} has no node {rec.node_index}")
        n = len(rec.atomic_numbers)
        if arr.shape == (n, 3):
            xyz = tuple(tuple(float(v) for v in row) for row in arr)
        elif arr.shape == (1, 3) and n == 1:
            xyz = (tuple(float(v) for v in arr[0]),)
        else:
            raise KeyAtomDatasetV3Error(
                "GEOMETRY_INCOMPLETE",
                f"curve {rec.curve_id} node {rec.node_index}: frame "
                f"geometry {tuple(arr.shape)} cannot cover N={n} atoms")
        return KeyAtomSampleV3(
            curve_id=rec.curve_id,
            node_index=rec.node_index,
            intent=rec.intent,
            atomic_numbers=rec.atomic_numbers,
            xyz=xyz,
            d0=rec.d0,
            endpoint_names=("A", "B"),
            y_teacher=rec.target_action_u,
            frame_weight=rec.label_weight,
            label_status=rec.label_status,
            sample_id=rec.sample_id,
            extras={"curve_weight": rec.curve_weight,
                    "sample_weight": rec.sample_weight,
                    "fold": rec.fold,
                    "feature_sha": "",
                    "label_sha": ""},
        )


def _require_columns_in(table: str, header_fields: Sequence[str] | None) -> None:
    """Typed column check against a CSV header (no rows materialized)."""
    required = REQUIRED_COLUMNS.get(table, ())
    header = set(header_fields or ())
    missing = [c for c in required if c not in header]
    _require(not missing, "TABLE_COLUMNS",
             f"{table} is missing required columns {missing}")


def _savez_deterministic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Deterministic ``.npz`` writer (sorted keys, fixed zip timestamps).

    ``np.savez`` stamps each zip entry with the current time, so its
    bytes (and thus the shard sha256) differ per run; this writer pins
    ``date_time=(1980,1,1)`` so re-running
    :func:`write_geometry_shards` on unchanged tables reproduces
    byte-identical shards.
    """
    import io as _io
    import zipfile as _zip

    with _zip.ZipFile(path, "w", compression=_zip.ZIP_DEFLATED) as zf:
        for key in sorted(arrays):
            buf = _io.BytesIO()
            np.save(buf, arrays[key], allow_pickle=False)
            info = _zip.ZipInfo(filename=f"{key}.npy",
                                date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = _zip.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, buf.getvalue())


def write_geometry_shards(
    tables_dir: str | Path,
    out_dir: str | Path | None = None,
    curves_per_shard: int = 256,
) -> dict:
    """Build per-curve geometry shards + ``shard_index_v53mp.json`` (S0-D).

    Reads the six-table delivery via the SAME CSV parse path as the
    loaders (:meth:`KeyAtomDatasetV3._parse_frame_xyz`), so shard arrays
    are bit-identical to a full CSV load (the shard-equivalence hard
    gate of plan §4.4).  Output per shard file ``shard_NNNNN.npz``:

    * ``<curve_id>::coords`` — ``[n_nodes, n_atoms, 3]`` float64 in
      ascending node_index order;
    * ``<curve_id>::node_indices`` — ``[n_nodes]`` int64;
    * ``<curve_id>::atomic_numbers`` — ``[n_atoms]`` int64.

    plus the index document ``{"version": "v53mp_shard_index_v1",
    "shards": [{"path", "curve_ids", "format": "npz", "sha256"}]}``.
    ``irc_node`` rows MUST be contiguous per curve (one physical block
    per curve, as delivered); a non-contiguous curve is a typed refusal
    (``CURVE_BLOCK_ORDER``) so the build can never silently drop frames.
    Writes are deterministic (sorted keys, fixed zip timestamps,
    float64): rerunning on unchanged tables reproduces byte-identical
    shards and index.
    """
    tables = Path(tables_dir)
    _require(tables.is_dir(), "TABLES_DIR", f"{tables} is not a directory")
    out = Path(out_dir) if out_dir is not None else tables
    out.mkdir(parents=True, exist_ok=True)
    _require(int(curves_per_shard) >= 1, "SHARD_SIZE",
             f"curves_per_shard must be >= 1, got {curves_per_shard}")

    curve_rows = _read_csv(tables / TABLE_FILENAMES["irc_curve"])
    _check_columns("irc_curve", curve_rows)
    known_curves = {r["curve_id"] for r in curve_rows}

    entries: list[dict] = []
    buffers: dict[str, dict] = {}
    pending: list[str] = []
    flushed: set[str] = set()
    n_curves = 0

    def _flush_shard() -> None:
        nonlocal pending
        if not pending:
            return
        arrays: dict[str, np.ndarray] = {}
        for cid in pending:
            buf = buffers.pop(cid)
            order = sorted(buf["frames"])
            arrays[f"{cid}::coords"] = np.stack(
                [buf["frames"][node] for node in order]).astype(np.float64)
            arrays[f"{cid}::node_indices"] = np.asarray(order, dtype=np.int64)
            arrays[f"{cid}::atomic_numbers"] = np.asarray(
                buf["atomic"], dtype=np.int64)
        name = f"shard_{len(entries):05d}.npz"
        shard_path = out / name
        _savez_deterministic(shard_path, arrays)
        entries.append({
            "path": name,
            "curve_ids": list(pending),
            "format": "npz",
            "sha256": _sha256(shard_path),
        })
        pending = []

    path = tables / TABLE_FILENAMES["irc_node"]
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        _require_columns_in("irc_node", reader.fieldnames)
        current_curve: str | None = None
        for row in reader:
            cid = row["curve_id"]
            if current_curve is not None and cid != current_curve:
                flushed.add(current_curve)
                pending.append(current_curve)
                n_curves += 1
                if len(pending) >= int(curves_per_shard):
                    _flush_shard()
                current_curve = None
            _require(cid in known_curves, "NODE_CURVE_FK",
                     f"irc_node row references unknown curve {cid}")
            _require(cid not in flushed, "CURVE_BLOCK_ORDER",
                     f"irc_node rows for curve {cid} are not contiguous; "
                     "sort irc_node.csv by curve_id before sharding")
            if cid != current_curve:
                current_curve = cid
                buffers[cid] = {
                    "frames": {},
                    "atomic": (),
                }
            buf = buffers[current_curve]
            node = int(row["node_index"])
            _require(node not in buf["frames"], "NODE_DUP",
                     f"duplicate node row ({cid}, {node})")
            buf["frames"][node] = KeyAtomDatasetV3._parse_frame_xyz(row)
            if not buf["atomic"]:
                payload = row.get("atomic_numbers") or ""
                if payload:
                    try:
                        buf["atomic"] = tuple(int(v)
                                              for v in json.loads(payload))
                    except (ValueError, json.JSONDecodeError) as exc:
                        raise KeyAtomDatasetV3Error(
                            "ATOMIC_NUMBERS_PARSE",
                            f"{cid}: {exc}") from exc
        if current_curve is not None:
            pending.append(current_curve)
            n_curves += 1
    _flush_shard()

    index_doc = {"version": SHARD_INDEX_VERSION, "shards": entries}
    index_path = out / SHARD_INDEX_FILENAME
    index_path.write_text(
        json.dumps(index_doc, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return {
        "version": SHARD_INDEX_VERSION,
        "index_path": str(index_path),
        "n_shards": len(entries),
        "n_curves": n_curves,
        "shards": [dict(e) for e in entries],
    }


def build_feature_batches(
    dataset: KeyAtomDatasetV3,
    indices: Sequence[int],
    *,
    level: str = "F3",
    intent_off: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    """Build one collated torch batch from dataset indices (forward+super)."""
    from pes2ts_core.pesn.features.key_atom_graph_v3 import (
        build_sample_features_v3,
        collate_v3,
        to_torch_batch,
    )

    built = [
        build_sample_features_v3(
            dataset.to_sample_v3(dataset.samples[int(i)]),
            level=level, intent_off=intent_off)
        for i in indices
    ]
    return to_torch_batch(collate_v3(built), device=device)


__all__ = [
    "ATOMIC_MASS_TABLE_V52",
    "DATA_MANIFEST_FILENAME",
    "CurveV52Labels",
    "KeyAtomDatasetV3",
    "KeyAtomDatasetV3Error",
    "LABEL_STATUSES",
    "REQUIRED_COLUMNS",
    "SPLIT_MANIFEST_FILENAME",
    "STOP_LABEL_SCHEMA_V52",
    "STOP_NEGATIVE_WINDOW_ANGSTROM_V52",
    "STOP_POSITIVE_WINDOW_ANGSTROM_V52",
    "SampleRecordV3",
    "TABLE_FILENAMES",
    "V52LabelTable",
    "V52SampleLabel",
    "WEIGHT_SUM_TOL",
    "ZERO_ARC_EPS_ANGSTROM_V52",
    "atomic_mass_v52",
    "build_feature_batches",
    "build_v52_labels",
    "kabsch_rmsd_v52",
    "mass_weighted_arc_lengths_v52",
    "progress_q_from_arcs_v52",
    "stop_label_from_distance_v52",
    # V5.3-MP S0-D sharded lazy dataset (plan §4.4)
    "SHARD_INDEX_FILENAME",
    "SHARD_INDEX_VERSION",
    "SampleRecordV3Lite",
    "ShardedKeyAtomDatasetV3",
    "write_geometry_shards",
]
