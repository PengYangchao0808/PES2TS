"""Streaming read-only view over the frozen six-table key-atom tables (Zone 2).

Source mapping (ported from the frozen ``pes2ts_core/pesn`` tree, which is being
removed from master):

* ``pes2ts_core/pesn/offline/data/key_atom_stream_view.py`` — streaming passes
  over ``sample_view.csv`` and ``irc_node.csv`` (node sets + atomic numbers,
  energies deliberately not collected), compact per-position columns, sampler
  stub records, npz ``memmap`` bridge.
* ``pes2ts_core/pesn/offline/data/key_atom_dataset_v3.py`` — light-table init
  (``irc_curve`` / ``reaction_intent`` / folds / header-only ``irc_edge`` +
  ``curve_quality``), survival-filter semantics and the curve-weight invariant.
* ``pes2ts_core/pesn/contracts/key_atom_v3.py`` — the minimal intent-row fields
  (direction, involution, endpoint-graph atoms) used by the join.

Column semantics are unchanged: rows survive only when ``|d0 - 0.08| <= 1e-12``,
``label_status`` is in ``label_statuses``, the intent direction is in
``directions`` and (when a fold filter is set) the curve fold matches.  The
``sum(curve_weight) == 1.0`` invariant is accumulated over the FULL primary-d0
group BEFORE status filtering, exactly like the frozen loader.

Read-only: this module never writes files, never imports torch and never reads
the quarantined truth tree.  All refusals are :class:`~._common.ReaderError`
with a stable ``.code`` (P2).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np

from ._common import (
    D0_PRIMARY_ANGSTROM,
    D0_TOLERANCE,
    LABEL_STATUSES,
    TABLE_FILENAMES,
    WEIGHT_SUM_TOL,
    ReaderError,
    check_columns,
    load_folds,
    read_csv_rows,
    require,
    verify_data_manifest,
)

__all__ = [
    "IntentRow",
    "KeyAtomStreamView",
    "StreamColumns",
    "TABLE_FILENAMES",
]


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


@dataclass(slots=True)
class StreamColumns:
    """Compact per-position columns of the survival-filtered sample stream."""

    n_rows: int
    curve_idx: np.ndarray
    node_index: np.ndarray
    intent_idx: np.ndarray
    direction: np.ndarray
    fold: np.ndarray
    weight_curves_checked: int = 0
    elapsed_s: float = 0.0


class KeyAtomStreamView:
    """Bounded read-only view over the six tables (never materialises samples).

    Parameters
    ----------
    tables_dir:
        Directory holding the six ``*.csv`` tables (plus optional
        ``data_manifest.json`` / split manifest).
    d0:
        Primary base offset (default :data:`D0_PRIMARY_ANGSTROM`).
    label_statuses:
        Status values kept by the survival filter.
    directions:
        Intent directions kept by the survival filter.
    split_manifest, fold:
        Optional curve-fold manifest / fold filter.
    verify_manifest:
        Fail-closed sha256 verification of ``data_manifest.json`` when present.
    """

    def __init__(
        self,
        tables_dir: str | Path,
        *,
        d0: float = D0_PRIMARY_ANGSTROM,
        label_statuses: Iterable[str] = LABEL_STATUSES,
        directions: Iterable[str] = ("forward", "reverse"),
        split_manifest: str | Path | None = None,
        fold: int | None = None,
        verify_manifest: bool = False,
    ) -> None:
        self.tables_dir = Path(tables_dir)
        self.d0 = float(d0)
        self.fold = fold
        self.label_statuses = frozenset(label_statuses)
        self.directions = frozenset(directions)
        require(self.tables_dir.is_dir(), "TABLES_DIR",
                f"{self.tables_dir} is not a directory")
        for table, filename in TABLE_FILENAMES.items():
            require((self.tables_dir / filename).is_file(), "TABLE_MISSING",
                    f"{table} -> {filename} not found")
        self.manifest_sha256 = verify_data_manifest(self.tables_dir) if verify_manifest else ""

        curve_rows = read_csv_rows(self.tables_dir / TABLE_FILENAMES["irc_curve"])
        check_columns("irc_curve", curve_rows)
        self._curves: dict[str, dict[str, str]] = {
            str(row["curve_id"]): row for row in curve_rows}
        require(len(self._curves) == len(curve_rows), "CURVE_ID_DUP",
                "duplicate curve_id in irc_curve")

        intent_rows = read_csv_rows(self.tables_dir / TABLE_FILENAMES["reaction_intent"])
        check_columns("reaction_intent", intent_rows)
        self._intents: dict[tuple[str, str], IntentRow] = {}
        for row in intent_rows:
            intent = IntentRow.from_table_row(row)
            key = (intent.curve_id, intent.intent_id)
            require(key not in self._intents, "INTENT_DUP",
                    f"duplicate (curve_id, intent_id) {key}")
            require(intent.curve_id in self._curves, "INTENT_FK",
                    f"intent {intent.intent_id} references unknown curve")
            self._intents[key] = intent
        for (curve_id, intent_id), intent in self._intents.items():
            if intent.intent_underdetermined:
                continue
            reverse = self._intents.get((curve_id, intent.reverse_of_intent_id))
            require(reverse is not None, "INTENT_REVERSE_MISSING",
                    f"intent {intent_id} has no reverse row "
                    f"{intent.reverse_of_intent_id!r}")
            require(reverse.reverse_of_intent_id == intent_id, "INTENT_INVOLUTION",
                    f"reverse(reverse({intent_id})) != {intent_id}")
        self._intent_row = {
            key: index for index, key in enumerate(self._intents)}

        self._check_header_only("irc_edge")
        self._check_header_only("curve_quality")
        self._folds = load_folds(self.tables_dir, split_manifest, fold)
        require(not (fold is not None and not self._folds), "SPLIT_MANIFEST_MISSING",
                "fold filter requested but no split manifest found")

        self._node_index: dict[str, set[int]] = {}
        self._curve_atomic: dict[str, tuple[int, ...]] = {}
        self._scan_irc_node_light()

    # ------------------------------------------------------------- light init
    def _check_header_only(self, table: str) -> None:
        path = self.tables_dir / TABLE_FILENAMES[table]
        require(path.is_file(), "TABLE_MISSING", f"{path} not found")
        with path.open("r", encoding="utf-8", newline="") as handle:
            check_columns(table, csv.DictReader(handle).fieldnames)

    def _scan_irc_node_light(self) -> None:
        """One streaming pass: node sets + per-curve atomic numbers (no energies)."""
        path = self.tables_dir / TABLE_FILENAMES["irc_node"]
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            check_columns("irc_node", header)
            column = {name: index for index, name in enumerate(header or ())}
            for raw in reader:
                curve_id = raw[column["curve_id"]]
                node = int(raw[column["node_index"]])
                nodes = self._node_index.setdefault(curve_id, set())
                require(node not in nodes, "NODE_DUP",
                        f"duplicate node row ({curve_id}, {node})")
                nodes.add(node)
                payload_index = column["atomic_numbers"]
                payload = raw[payload_index] if len(raw) > payload_index else ""
                if payload and curve_id not in self._curve_atomic:
                    try:
                        self._curve_atomic[curve_id] = tuple(
                            int(value) for value in json.loads(payload))
                    except (ValueError, json.JSONDecodeError) as exc:
                        raise ReaderError("ATOMIC_NUMBERS_PARSE",
                                          f"{curve_id}: {exc}") from exc

    # -------------------------------------------------------------- properties
    @property
    def curve_ids(self) -> list[str]:
        """Curve ids in ``irc_curve.csv`` order."""
        return list(self._curves)

    @property
    def n_curves(self) -> int:
        return len(self._curves)

    @property
    def n_nodes(self) -> int:
        return sum(len(nodes) for nodes in self._node_index.values())

    def curve_meta(self, curve_id: str) -> dict[str, str]:
        """Raw ``irc_curve`` row for ``curve_id`` (typed refusal when unknown)."""
        require(curve_id in self._curves, "CURVE_UNKNOWN",
                f"curve {curve_id!r} not in irc_curve")
        return dict(self._curves[curve_id])

    def node_indices(self, curve_id: str) -> list[int]:
        """Sorted node indices of one curve (the frozen node set)."""
        require(curve_id in self._node_index, "CURVE_UNKNOWN",
                f"curve {curve_id!r} has no irc_node rows")
        return sorted(self._node_index[curve_id])

    def node_window(self, curve_id: str, start: int, stop: int | None = None) -> np.ndarray:
        """Contiguous node-index window ``[start, stop)`` of one curve.

        This is the "node window" primitive used by frame/sample joins; it is a
        slice of the frozen node set (never a re-sorted or re-indexed range).
        """
        nodes = np.asarray(self.node_indices(curve_id), dtype=np.int64)
        return nodes[start:stop]

    def atomic_numbers(self, curve_id: str) -> tuple[int, ...]:
        """Per-curve atomic-number tuple (typed refusal when absent)."""
        atoms = self._curve_atomic.get(curve_id)
        require(bool(atoms), "ATOMIC_NUMBERS_MISSING",
                f"curve {curve_id!r} has no atomic_numbers payload")
        return atoms

    def fold_of(self, curve_id: str) -> int | None:
        return self._folds.get(curve_id)

    def intent(self, curve_id: str, intent_id: str) -> IntentRow:
        intent = self._intents.get((curve_id, intent_id))
        require(intent is not None, "INTENT_FK",
                f"unknown intent ({curve_id!r}, {intent_id!r})")
        return intent

    # ------------------------------------------------------- streaming passes
    def iter_sample_rows(self) -> Iterator[dict[str, str]]:
        """Yield raw ``sample_view.csv`` rows as dicts (bounded memory)."""
        path = self.tables_dir / TABLE_FILENAMES["sample_view"]
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            check_columns("sample_view", reader.fieldnames)
            yield from reader

    def stream_columns(self) -> StreamColumns:
        """Pass 1: compact numpy columns after the frozen survival filters.

        Also enforces (verbatim) the full-group curve-weight invariant and the
        per-row intent/node foreign keys; every refusal is typed.
        """
        from array import array

        curve_row = {curve_id: index for index, curve_id in enumerate(self._curves)}
        c_idx, n_idx, i_idx = array("i"), array("i"), array("i")
        direction, folds = array("b"), array("b")
        weight_totals: dict[str, float] = {}
        for row in self.iter_sample_rows():
            try:
                row_d0 = float(row["d0"])
                curve_weight = float(row["curve_weight"])
            except (KeyError, ValueError) as exc:
                raise ReaderError("SAMPLE_WEIGHT_PARSE",
                                  f"{row.get('sample_id')}: {exc}") from exc
            curve_id = row["curve_id"]
            if abs(row_d0 - self.d0) <= D0_TOLERANCE:
                weight_totals[curve_id] = weight_totals.get(curve_id, 0.0) + curve_weight
            if abs(row_d0 - self.d0) > D0_TOLERANCE:
                continue
            if row["label_status"] not in self.label_statuses:
                continue
            node_index = int(row["node_index"])
            intent = self._intents.get((curve_id, row["intent_id"]))
            require(intent is not None, "SAMPLE_INTENT_FK",
                    f"sample {row.get('sample_id')} references unknown intent "
                    f"({curve_id!r}, {row.get('intent_id')!r})")
            require(intent.direction in self.directions, "SAMPLE_DIRECTION",
                    f"sample {row.get('sample_id')} direction filtered")
            require(node_index in self._node_index.get(curve_id, ()), "SAMPLE_NODE_FK",
                    f"sample {row.get('sample_id')} references unknown node "
                    f"({curve_id}, {node_index})")
            atomic = self._curve_atomic.get(curve_id) or intent.g_atoms
            require(bool(atomic), "ATOMIC_NUMBERS_MISSING",
                    f"curve {curve_id} has no atomic_numbers payload")
            if self.fold is not None:
                curve_fold = self._folds.get(curve_id)
                require(curve_fold is not None, "SPLIT_CURVE_MISSING",
                        f"curve {curve_id} missing from split manifest")
                if int(curve_fold) != int(self.fold):
                    continue
            fold_value = (row.get("fold") or "").strip()
            c_idx.append(curve_row[curve_id])
            n_idx.append(node_index)
            i_idx.append(self._intent_row[(curve_id, row["intent_id"])])
            direction.append(0 if intent.direction == "forward" else 1)
            folds.append(int(fold_value) if fold_value
                         else int(self._folds.get(curve_id, -1)))
        require(len(c_idx) > 0, "NO_SAMPLES",
                "no sample_view rows survive the dataset filters "
                f"(d0={self.d0}, statuses={sorted(self.label_statuses)}, "
                f"fold={self.fold})")
        for curve_id, total in weight_totals.items():
            require(abs(total - 1.0) <= WEIGHT_SUM_TOL, "CURVE_WEIGHT_SUM",
                    f"curve {curve_id}: sum(curve_weight)={total!r} != 1.0")
        return StreamColumns(
            n_rows=len(c_idx),
            curve_idx=np.asarray(c_idx, dtype=np.int32),
            node_index=np.asarray(n_idx, dtype=np.int32),
            intent_idx=np.asarray(i_idx, dtype=np.int32),
            direction=np.asarray(direction, dtype=np.int8),
            fold=np.asarray(folds, dtype=np.int8),
            weight_curves_checked=len(weight_totals),
        )

    def iter_label_joined_rows(self, labels: Any | None = None) -> Iterator[dict[str, Any]]:
        """Stream sample rows joined with intent labels and (optionally) the cache.

        ``labels`` is an optional :class:`pes2ts_data.readers.LabelTables` (or
        any mapping-like with ``sample(sample_id)``); when provided the joined
        row gains the cache fields ``q``/``q_mask``/``stop``/``ep_label``/
        ``d_end`` (label joins are read-only dict copies).
        """
        for row in self.iter_sample_rows():
            curve_id, intent_id = row["curve_id"], row["intent_id"]
            intent = self._intents.get((curve_id, intent_id))
            joined: dict[str, Any] = {
                "sample_id": row["sample_id"],
                "curve_id": curve_id,
                "node_index": int(row["node_index"]),
                "intent_id": intent_id,
                "direction": intent.direction if intent is not None else None,
                "d0": float(row["d0"]),
                "label_status": row["label_status"],
                "label_weight": float(row["label_weight"]),
                "curve_weight": float(row["curve_weight"]),
                "sample_weight": float(row["sample_weight"]),
                "fold": (int(row["fold"]) if (row.get("fold") or "").strip() else None),
            }
            if labels is not None:
                joined["labels"] = labels.sample(row["sample_id"])
            yield joined
