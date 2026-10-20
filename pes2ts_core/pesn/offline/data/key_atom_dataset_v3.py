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
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from pes2ts_core.pesn.contracts.key_atom_v3 import (
    D0_PRIMARY_ANGSTROM,
    IntentRecordV3,
    KeyAtomSampleV3,
    KeyAtomV3ContractError,
)

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
        path = self.tables_dir / DATA_MANIFEST_FILENAME
        if not path.is_file():
            return                      # E1 may deliver manifests later
        doc = json.loads(path.read_text(encoding="utf-8"))
        files = doc.get("files", {})
        if isinstance(files, list):
            entries = {str(e["path"]): str(e["sha256"]) for e in files}
        else:
            entries = {str(k): str(v) for k, v in files.items()}
        for rel, sha in entries.items():
            target = self.tables_dir / rel
            _require(target.is_file(), "MANIFEST_FILE",
                     f"{rel} listed in {DATA_MANIFEST_FILENAME} not found")
            actual = _sha256(target)
            _require(actual == sha, "MANIFEST_SHA",
                     f"{rel}: manifest sha256 {sha[:12]}… != actual "
                     f"{actual[:12]}…")
        self.manifest_sha256 = _sha256(path)

    # -- folds ----------------------------------------------------------------
    def _load_folds(
        self,
        split_manifest: str | Path | None,
        fold: int | None,
    ) -> dict[str, int]:
        if split_manifest is None:
            candidate = self.tables_dir / SPLIT_MANIFEST_FILENAME
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
    "DATA_MANIFEST_FILENAME",
    "KeyAtomDatasetV3",
    "KeyAtomDatasetV3Error",
    "LABEL_STATUSES",
    "REQUIRED_COLUMNS",
    "SPLIT_MANIFEST_FILENAME",
    "SampleRecordV3",
    "TABLE_FILENAMES",
    "WEIGHT_SUM_TOL",
    "build_feature_batches",
]
