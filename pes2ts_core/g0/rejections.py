"""Unified rejection ledger for the G0 stage.

Every stage records why a reaction left the pipeline with a member of the
single :class:`RejectionCode` enum; defining codes anywhere else is forbidden.
The ledger is persisted append-semantics-safe: :meth:`RejectionLedger.write`
atomically rewrites the full in-memory state as JSONL and refreshes the
summary, so a crash can never leave a torn ledger line. Because ``write`` is a
full rewrite, a later stage that wants to append must first read the persisted
state back with :meth:`RejectionLedger.load` (which tolerates a missing file)
instead of starting from an empty ledger.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final, TypedDict

from pes2ts_core.g0 import MANIFEST_SCHEMA_VERSION
from pes2ts_core.utils.hashing import JSONValue, stable_json_dumps
from pes2ts_core.utils.jsonio import atomic_writer, write_json

LEDGER_FILENAME: Final[str] = "rejection_ledger.jsonl"
SUMMARY_FILENAME: Final[str] = "rejection_summary.json"


class UnknownRejectionCodeError(ValueError):
    """Raised when a rejection carries a code outside :class:`RejectionCode`."""


class RejectionCode(str, Enum):
    """Machine-readable reason a reaction was rejected."""

    BAD_ID = "BAD_ID"
    DUPLICATE_ID = "DUPLICATE_ID"
    BAD_SMILES = "BAD_SMILES"
    NO_ARROW = "NO_ARROW"
    UNPARSEABLE_MAPPED = "UNPARSEABLE_MAPPED"
    MISSING_IN_H5 = "MISSING_IN_H5"
    MISSING_IN_CSV = "MISSING_IN_CSV"
    ID_MISMATCH = "ID_MISMATCH"
    ATOM_COUNT_MISMATCH = "ATOM_COUNT_MISMATCH"
    ELEMENT_MISMATCH = "ELEMENT_MISMATCH"
    CHARGE_NOT_NEUTRAL = "CHARGE_NOT_NEUTRAL"
    SPIN_NOT_SINGLET = "SPIN_NOT_SINGLET"
    BAD_GEOMETRY_SHAPE = "BAD_GEOMETRY_SHAPE"
    H5_SCHEMA_ERROR = "H5_SCHEMA_ERROR"
    SPLIT_SCHEMA_ERROR = "SPLIT_SCHEMA_ERROR"
    NOT_IN_SPLIT = "NOT_IN_SPLIT"
    DUPLICATE_OF = "DUPLICATE_OF"
    REVERSE_OF = "REVERSE_OF"
    LEAK_EXCLUDED = "LEAK_EXCLUDED"
    AUDIT_BUDGET_EXCEEDED = "AUDIT_BUDGET_EXCEEDED"


class RejectionRecord(TypedDict):
    """Serialized shape of one rejection ledger line."""

    reaction_id: str
    stage: str
    code: str
    detail: str
    source_pointer: str


@dataclass(frozen=True, slots=True)
class Rejection:
    """One rejected reaction with its stage and source provenance."""

    reaction_id: str
    stage: str
    code: RejectionCode
    detail: str
    source_pointer: str

    def __post_init__(self) -> None:
        if not isinstance(self.code, RejectionCode):
            msg = f"Unknown rejection code: {self.code!r}"
            raise UnknownRejectionCodeError(msg)

    def to_record(self) -> RejectionRecord:
        """Return the JSONL-serializable view of this rejection."""
        return {
            "reaction_id": self.reaction_id,
            "stage": self.stage,
            "code": self.code.value,
            "detail": self.detail,
            "source_pointer": self.source_pointer,
        }


class RejectionLedger:
    """In-memory rejection accumulator that writes JSONL plus a summary.

    Parameters
    ----------
    manifests_dir:
        Directory receiving ``rejection_ledger.jsonl`` and
        ``rejection_summary.json`` (created when missing).
    """

    def __init__(self, manifests_dir: Path) -> None:
        self._manifests_dir = manifests_dir
        self._rejections: list[Rejection] = []

    @classmethod
    def load(cls, manifests_dir: Path) -> RejectionLedger:
        """Return a ledger preloaded with the rejections persisted there.

        A missing ``rejection_ledger.jsonl`` yields an empty ledger, so callers
        can always ``load(...).add(...).write()`` and preserve whatever an
        earlier stage recorded instead of clobbering it. Blank lines are
        ignored; a line that is not a JSON object or that misses a field raises
        :class:`ValueError` naming the line number, because silently dropping a
        recorded rejection would corrupt the pipeline's audit trail.
        """
        logger = logging.getLogger(__name__)
        ledger = cls(manifests_dir)
        path = Path(manifests_dir) / LEDGER_FILENAME
        if not path.is_file():
            return ledger
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    record: object = json.loads(text)
                except ValueError as exc:
                    msg = (
                        f"Malformed rejection ledger line {line_number} in "
                        f"{path}: {exc}"
                    )
                    raise ValueError(msg) from exc
                if not isinstance(record, dict):
                    msg = (
                        f"Malformed rejection ledger line {line_number} in "
                        f"{path}: expected a JSON object"
                    )
                    raise ValueError(msg)
                reaction_id = record.get("reaction_id")
                stage = record.get("stage")
                code = record.get("code")
                detail = record.get("detail")
                source_pointer = record.get("source_pointer")
                if not (
                    isinstance(reaction_id, str)
                    and isinstance(stage, str)
                    and isinstance(code, str)
                    and isinstance(detail, str)
                    and isinstance(source_pointer, str)
                ):
                    msg = (
                        f"Malformed rejection ledger line {line_number} in "
                        f"{path}: expected five string fields"
                    )
                    raise ValueError(msg)
                try:
                    rejection = Rejection(
                        reaction_id=reaction_id,
                        stage=stage,
                        code=RejectionCode(code),
                        detail=detail,
                        source_pointer=source_pointer,
                    )
                except ValueError as exc:
                    msg = (
                        f"Malformed rejection ledger line {line_number} in "
                        f"{path}: {exc}"
                    )
                    raise ValueError(msg) from exc
                ledger.add(rejection)
        logger.info(
            "Loaded %d rejection(s) from %s", len(ledger._rejections), path
        )
        return ledger

    def add(self, rejection: Rejection) -> None:
        """Validate and append *rejection* to the in-memory ledger."""
        if not isinstance(rejection.code, RejectionCode):
            msg = f"Unknown rejection code: {rejection.code!r}"
            raise UnknownRejectionCodeError(msg)
        self._rejections.append(rejection)

    def write(self) -> None:
        """Atomically persist the full ledger and its summary."""
        logger = logging.getLogger(__name__)
        ledger_path = self._manifests_dir / LEDGER_FILENAME
        lines = [stable_json_dumps(rejection.to_record()) for rejection in self._rejections]
        payload = ("\n".join(lines) + "\n" if lines else "").encode("utf-8")
        with atomic_writer(ledger_path) as handle:
            handle.write(payload)
        write_json(self._manifests_dir / SUMMARY_FILENAME, self._summary())
        logger.info("Wrote %d rejection(s) to %s", len(self._rejections), ledger_path)

    def _summary(self) -> dict[str, JSONValue]:
        """Return the rejection histogram keyed by code value, sorted by name."""
        counts = Counter(rejection.code.value for rejection in self._rejections)
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "total": len(self._rejections),
            "by_code": {code: counts[code] for code in sorted(counts)},
        }


__all__ = [
    "LEDGER_FILENAME",
    "SUMMARY_FILENAME",
    "Rejection",
    "RejectionCode",
    "RejectionLedger",
    "RejectionRecord",
    "UnknownRejectionCodeError",
]
