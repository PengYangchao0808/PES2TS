"""Unified rejection ledger for the G0 stage.

Every stage records why a reaction left the pipeline with a member of the
single :class:`RejectionCode` enum; defining codes anywhere else is forbidden.
The ledger is persisted append-semantics-safe: :meth:`RejectionLedger.write`
atomically rewrites the full in-memory state as JSONL and refreshes the
summary, so a crash can never leave a torn ledger line.
"""

from __future__ import annotations

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
