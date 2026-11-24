"""Shared constants and pure-numpy helpers for the six-table readers.

Source mapping (ported from the frozen ``pes2ts_core/pesn`` tree into Zone 2;
the originals are being removed from master):

* ``pes2ts_core/pesn/offline/data/key_atom_dataset_v3.py``
  -> :data:`TABLE_FILENAMES`, :data:`REQUIRED_COLUMNS`,
     :func:`read_csv_rows`, :func:`check_columns`, :func:`load_folds`,
     :func:`verify_data_manifest`, :class:`ReaderError`.
* ``pes2ts_core/pesn/contracts/key_atom_v3.py``
  -> :data:`D0_PRIMARY_ANGSTROM`.

Nothing here imports torch, writes to disk or touches the quarantined truth
tree; the six CSV tables are the only input.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

__all__ = [
    "D0_PRIMARY_ANGSTROM",
    "DATA_MANIFEST_FILENAME",
    "LABEL_STATUSES",
    "IntentRow",
    "ReaderError",
    "REQUIRED_COLUMNS",
    "SPLIT_MANIFEST_FILENAME",
    "SPLIT_MANIFEST_V53MP_FILENAME",
    "TABLE_FILENAMES",
    "WEIGHT_SUM_TOL",
    "check_columns",
    "load_folds",
    "read_csv_rows",
    "require",
    "sha256_file",
    "verify_data_manifest",
]

#: Six-table filenames (PD-1: exactly six tables, not five).
TABLE_FILENAMES: Final[dict[str, str]] = {
    "irc_curve": "irc_curve.csv",
    "irc_node": "irc_node.csv",
    "irc_edge": "irc_edge.csv",
    "reaction_intent": "reaction_intent.csv",
    "curve_quality": "curve_quality.csv",
    "sample_view": "sample_view.csv",
}

#: Required columns mirrored from the ``r3_6table_v2`` schema (E1).
REQUIRED_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "irc_curve": ("curve_id", "canonical_irc_id", "n_nodes", "ts_index"),
    "irc_node": ("curve_id", "node_index", "x", "y", "z", "atomic_numbers"),
    "reaction_intent": (
        "intent_id", "curve_id", "direction", "reverse_of_intent_id",
        "intent_pair_id", "g_src", "g_tgt", "cv_spec"),
    "sample_view": (
        "sample_id", "curve_id", "node_index", "intent_id", "d0",
        "label_weight", "curve_weight", "sample_weight", "label_status"),
    "irc_edge": ("curve_id", "intent_id", "edge_index", "delta_xyz_flat"),
    "curve_quality": ("curve_id", "quality_status"),
}

#: The primary base-offset of the frozen dataset (V5 contract).
D0_PRIMARY_ANGSTROM: Final[float] = 0.08
#: Tolerance on the d0 equality test (matches the frozen loader).
D0_TOLERANCE: Final[float] = 1e-12

DATA_MANIFEST_FILENAME: Final[str] = "data_manifest.json"
SPLIT_MANIFEST_FILENAME: Final[str] = "split_manifest_r3.json"
SPLIT_MANIFEST_V53MP_FILENAME: Final[str] = "split_manifest_v53mp.json"

#: Label-status enum accepted by the frozen ``sample_view`` rows.
LABEL_STATUSES: Final[tuple[str, ...]] = ("ok", "masked", "terminal", "conflict")

#: Invariant tolerance for ``sum(curve_weight) == 1.0`` per curve group.
WEIGHT_SUM_TOL: Final[float] = 1e-9

#: Read chunk size for streaming sha256 (bounded memory on multi-GB CSVs).
_SHA256_CHUNK_BYTES: Final[int] = 1 << 20


class ReaderError(RuntimeError):
    """Typed refusal shared by every Zone-2 reader (P2 honest rejection).

    Attributes
    ----------
    code:
        Stable machine code (``TABLE_MISSING``, ``TABLE_COLUMNS``,
        ``MANIFEST_SHA``, ``NPZ_MISSING`` ...).
    detail:
        Human-readable explanation.
    """

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}" if self.detail else self.code)


def require(condition: bool, code: str, detail: str) -> None:
    """Raise :class:`ReaderError` unless ``condition`` holds."""
    if not condition:
        raise ReaderError(code, detail)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Read one CSV table fully into ``list[dict[str, str]]`` (small tables only)."""
    path = Path(path)
    if not path.is_file():
        raise ReaderError("TABLE_MISSING", f"{path} not found")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def check_columns(table: str, header_or_rows: Sequence[str] | Iterable[Mapping[str, Any]] | None) -> None:
    """Typed required-column check against a CSV header or parsed rows."""
    required = REQUIRED_COLUMNS.get(table, ())
    if header_or_rows is None:
        header: set[str] = set()
    else:
        items = list(header_or_rows)
        if items and isinstance(items[0], Mapping):
            header = {str(key) for key in items[0]}
        else:
            header = {str(item) for item in items}
    missing = [column for column in required if column not in header]
    require(not missing, "TABLE_COLUMNS",
            f"{table} is missing required columns {missing}")


def sha256_file(path: str | Path) -> str:
    """Streaming sha256 hex digest of ``path`` (same digest as one-shot hashing)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_SHA256_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_data_manifest(tables_dir: str | Path) -> str:
    """Fail-closed sha256 verification of ``data_manifest.json``.

    Returns the manifest's own sha256 when the manifest is present, or ``""``
    when the delivery has not produced one yet (mirrors the frozen loader).
    """
    tables_dir = Path(tables_dir)
    path = tables_dir / DATA_MANIFEST_FILENAME
    if not path.is_file():
        return ""
    doc = json.loads(path.read_text(encoding="utf-8"))
    files = doc.get("files", {})
    if isinstance(files, list):
        entries = {str(entry["path"]): str(entry["sha256"]) for entry in files}
    else:
        entries = {str(key): str(value) for key, value in files.items()}
    for relative, expected in entries.items():
        target = tables_dir / relative
        require(target.is_file(), "MANIFEST_FILE",
                f"{relative} listed in {DATA_MANIFEST_FILENAME} not found")
        actual = sha256_file(target)
        require(actual == expected, "MANIFEST_SHA",
                f"{relative}: manifest sha256 {expected[:12]}... != actual {actual[:12]}...")
    return sha256_file(path)


def load_folds(
    tables_dir: str | Path,
    split_manifest: str | Path | None = None,
    fold: int | None = None,
) -> dict[str, int]:
    """Return ``curve_id -> fold`` from a split manifest (frozen-loader shapes).

    Accepted shapes: ``curve_folds`` mapping, ``assignments`` list, ``folds``
    mapping or a bare assignment list.  When no manifest exists and ``fold`` is
    ``None`` the mapping is empty; a requested fold without a manifest is a
    typed refusal.
    """
    tables_dir = Path(tables_dir)
    if split_manifest is None:
        candidate = tables_dir / SPLIT_MANIFEST_FILENAME
        split_manifest = candidate if candidate.is_file() else None
    if split_manifest is None:
        require(fold is None, "SPLIT_MANIFEST_MISSING",
                "fold filter requested but no split manifest found")
        return {}
    doc = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
    folds: dict[str, int]
    if isinstance(doc, Mapping) and "curve_folds" in doc:
        folds = {str(key): int(value) for key, value in doc["curve_folds"].items()}
    elif isinstance(doc, Mapping) and "assignments" in doc:
        folds = {str(row["curve_id"]): int(row["fold"]) for row in doc["assignments"]}
    elif isinstance(doc, Mapping) and isinstance(doc.get("folds"), Mapping):
        folds = {}
        for fold_id, curve_ids in doc["folds"].items():
            for curve_id in curve_ids:
                folds[str(curve_id)] = int(fold_id)
    elif isinstance(doc, list):
        folds = {str(row["curve_id"]): int(row["fold"]) for row in doc}
    else:
        raise ReaderError("SPLIT_MANIFEST_SHAPE",
                          f"unrecognized split manifest shape in {split_manifest}")
    require(bool(folds), "SPLIT_MANIFEST_EMPTY", "split manifest has no folds")
    return folds


@dataclass(frozen=True)
class IntentRow:
    """Minimal ``reaction_intent.csv`` row needed by the streaming view.

    The frozen ``IntentRecordV3`` carries the full G_rxn contract; only the
    direction sign, the reverse-involution key and the endpoint-graph atoms are
    consumed by the reader/join paths, so the reader keeps exactly those.
    """

    intent_id: str
    curve_id: str
    direction: str
    reverse_of_intent_id: str
    intent_pair_id: str
    g_atoms: tuple[int, ...] = ()
    intent_underdetermined: bool = False

    @classmethod
    def from_table_row(cls, row: Mapping[str, Any]) -> "IntentRow":
        """Parse one ``reaction_intent.csv`` row (typed refusals)."""
        direction = str(row.get("direction", "")).strip()
        require(direction in ("forward", "reverse"), "INTENT_DIRECTION",
                f"intent {row.get('intent_id')!r}: unknown direction {direction!r}")
        atoms: tuple[int, ...] = ()
        for key in ("g_src", "g_tgt"):
            payload = row.get(key) or ""
            if not str(payload).strip():
                continue
            try:
                doc = json.loads(str(payload))
            except json.JSONDecodeError as exc:
                raise ReaderError("INTENT_GRAPH_PARSE",
                                  f"{row.get('intent_id')}: {key}: {exc}") from exc
            if isinstance(doc, Mapping) and doc.get("atoms"):
                atoms = tuple(int(atom) for atom in doc["atoms"])
            if atoms:
                break
        flag = str(row.get("intent_underdetermined", "false")).strip().lower()
        return cls(
            intent_id=str(row.get("intent_id", "")),
            curve_id=str(row.get("curve_id", "")),
            direction=direction,
            reverse_of_intent_id=str(row.get("reverse_of_intent_id", "") or ""),
            intent_pair_id=str(row.get("intent_pair_id", "") or ""),
            g_atoms=atoms,
            intent_underdetermined=flag in ("true", "1"),
        )
