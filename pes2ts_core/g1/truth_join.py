"""P1.0 join audit: ID coverage between inventory, TS, IRC index, and G1.

Pure data function: every table arrives as plain rows (the orchestrator in
:mod:`pes2ts_core.g1.p1_truth` owns all file reads, including the audited
truth accessors), so this module runs in tests without any quarantined
artifact.  For each inventory reaction it decides whether the ID join is
complete and records every conflict with a typed reason; the aggregate
becomes ``g1_p1_join_audit.json``.

The audit answers exactly one question -- *can this reaction's endpoint data
be joined to its TS/IRC truth and to a built G1 document?* -- and never
 silently drops a denominator: every inventory row lands in ``n_joined_all``
 or in one named reason bucket.  Element consistency compares the inventory's
distinct ``elements`` set against the distinct set of the TS atomic numbers
(the per-count conservation is the atom-count check here and the exact
map-order sequence check in the resolver).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Final

from pes2ts_core.g0.rp_checks import ELEMENT_SYMBOLS
from pes2ts_core.g1.truth_schema import JOIN_AUDIT_SCHEMA_VERSION

#: Maximum example reaction ids kept per reason bucket.
MAX_EXAMPLES: Final[int] = 5

#: Typed join-audit reasons.
REASON_MISSING_TS: Final[str] = "missing_ts"
REASON_MISSING_IRC: Final[str] = "missing_irc"
REASON_MISSING_G1: Final[str] = "missing_g1_build"
REASON_TS_DUPLICATE_ID: Final[str] = "ts_duplicate_id"
REASON_IRC_DUPLICATE_ID: Final[str] = "irc_duplicate_id"
REASON_INVENTORY_DUPLICATE_ID: Final[str] = "inventory_duplicate_id"
REASON_ATOM_COUNT_CONFLICT: Final[str] = "atom_count_conflict"
REASON_ELEMENT_CONFLICT: Final[str] = "element_conflict"
REASON_CHARGE_CONFLICT: Final[str] = "charge_conflict"
REASON_SPIN_CONFLICT: Final[str] = "spin_conflict"
REASON_SMILES_CONFLICT: Final[str] = "reaction_smiles_conflict"
REASON_TS_BAD_SHAPE: Final[str] = "ts_bad_shape"
REASON_IRC_BAD_SHAPE: Final[str] = "irc_bad_shape"

#: Every reason the audit can report, in reporting order.
JOIN_REASONS: Final[tuple[str, ...]] = (
    REASON_INVENTORY_DUPLICATE_ID,
    REASON_TS_DUPLICATE_ID,
    REASON_IRC_DUPLICATE_ID,
    REASON_MISSING_TS,
    REASON_MISSING_IRC,
    REASON_MISSING_G1,
    REASON_TS_BAD_SHAPE,
    REASON_IRC_BAD_SHAPE,
    REASON_ATOM_COUNT_CONFLICT,
    REASON_ELEMENT_CONFLICT,
    REASON_CHARGE_CONFLICT,
    REASON_SPIN_CONFLICT,
    REASON_SMILES_CONFLICT,
)


def _symbol_of(atomic_number: int) -> str | None:
    """Return the element symbol of *atomic_number* (``None`` when unknown)."""
    if 1 <= atomic_number <= len(ELEMENT_SYMBOLS):
        return ELEMENT_SYMBOLS[atomic_number - 1]
    return None


def _duplicate_ids(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    """Return the reaction ids appearing more than once in *rows*."""
    counts = Counter(str(row["reaction_id"]) for row in rows)
    return {reaction_id for reaction_id, n in counts.items() if n > 1}


def _ts_shape_ok(ts_row: Mapping[str, Any]) -> bool:
    """Return whether the TS row's coordinates match its atomic-number list."""
    numbers = ts_row.get("atomic_numbers")
    coordinates = ts_row.get("coordinates")
    if not isinstance(numbers, list) or not isinstance(coordinates, list):
        return False
    return len(numbers) == len(coordinates) and all(
        isinstance(point, list) and len(point) == 3 for point in coordinates
    )


def _reaction_reasons(
    inventory_row: Mapping[str, Any],
    ts_row: Mapping[str, Any] | None,
    irc_row: Mapping[str, Any] | None,
    g1_status: str | None,
    duplicates: Mapping[str, set[str]],
) -> list[str]:
    """Return every join-audit reason of one inventory reaction (ordered)."""
    reasons: list[str] = []
    reaction_id = str(inventory_row["reaction_id"])
    for reason, table in (
        (REASON_INVENTORY_DUPLICATE_ID, "inventory"),
        (REASON_TS_DUPLICATE_ID, "ts"),
        (REASON_IRC_DUPLICATE_ID, "irc"),
    ):
        if reaction_id in duplicates[table]:
            reasons.append(reason)
    if ts_row is None:
        reasons.append(REASON_MISSING_TS)
    if irc_row is None:
        reasons.append(REASON_MISSING_IRC)
    if g1_status is None:
        reasons.append(REASON_MISSING_G1)
    ts_numbers: list[int] | None = None
    if ts_row is not None:
        if not _ts_shape_ok(ts_row):
            reasons.append(REASON_TS_BAD_SHAPE)
        else:
            ts_numbers = [int(z) for z in ts_row["atomic_numbers"]]
    if irc_row is not None:
        n_atoms = irc_row.get("n_atoms")
        n_frames = irc_row.get("n_frames")
        if not isinstance(n_atoms, int) or n_atoms < 1 or not isinstance(n_frames, int) or n_frames < 1:
            reasons.append(REASON_IRC_BAD_SHAPE)
    if ts_numbers is not None and irc_row is not None and isinstance(irc_row.get("n_atoms"), int):
        if len(ts_numbers) != int(irc_row["n_atoms"]):
            reasons.append(REASON_ATOM_COUNT_CONFLICT)
    n_reactant = int(inventory_row["total_atoms_reactants"])
    n_product = int(inventory_row["total_atoms_products"])
    if ts_numbers is not None and not (len(ts_numbers) == n_reactant == n_product):
        reasons.append(REASON_ATOM_COUNT_CONFLICT)
    if ts_numbers is not None:
        inventory_elements = {
            str(symbol) for symbol in inventory_row.get("elements", [])
        }
        ts_elements = {
            symbol
            for symbol in (_symbol_of(z) for z in ts_numbers)
            if symbol is not None
        }
        if ts_elements != inventory_elements:
            reasons.append(REASON_ELEMENT_CONFLICT)
    if ts_row is not None:
        ts_charge = ts_row.get("charge")
        if ts_charge is not None and int(ts_charge) != int(inventory_row["charge_total_reactants"]):
            reasons.append(REASON_CHARGE_CONFLICT)
        ts_multiplicity = ts_row.get("multiplicity")
        if ts_multiplicity is not None and int(ts_multiplicity) != int(inventory_row["multiplicity_max"]):
            reasons.append(REASON_SPIN_CONFLICT)
        ts_smiles = ts_row.get("reaction_smiles")
        if isinstance(ts_smiles, str) and ts_smiles and ts_smiles != str(inventory_row["reaction_smiles"]):
            reasons.append(REASON_SMILES_CONFLICT)
    return reasons


def build_join_audit(
    inventory_rows: Sequence[Mapping[str, Any]],
    ts_rows: Sequence[Mapping[str, Any]],
    irc_rows: Sequence[Mapping[str, Any]],
    g1_summary_rows: Sequence[Mapping[str, Any]],
    *,
    digests: Mapping[str, str] | None = None,
    dataset_version: str = "",
) -> dict[str, Any]:
    """Return the complete ``g1_p1_join_audit.json`` document.

    ``digests`` maps artifact names (``inventory``, ``ts``, ``irc_index``,
    ``g1_summary``) to their SHA-256 digests; they are recorded verbatim so
    the audit is reproducible against the exact bytes it judged.  A reaction
    joins cleanly exactly when its reason list is empty; ``n_joined_all``
    counts those, and every other inventory row appears in ``by_reason``.
    """
    ts_by_id: dict[str, Mapping[str, Any]] = {}
    for row in ts_rows:
        ts_by_id.setdefault(str(row["reaction_id"]), row)
    irc_by_id: dict[str, Mapping[str, Any]] = {}
    for row in irc_rows:
        irc_by_id.setdefault(str(row["reaction_id"]), row)
    g1_status_by_id = {
        str(row["reaction_id"]): str(row["status"]) for row in g1_summary_rows
    }
    duplicates = {
        "inventory": _duplicate_ids(inventory_rows),
        "ts": _duplicate_ids(ts_rows),
        "irc": _duplicate_ids(irc_rows),
    }
    reason_examples: dict[str, list[str]] = {reason: [] for reason in JOIN_REASONS}
    reason_counts: Counter[str] = Counter()
    n_joined = 0
    for inventory_row in inventory_rows:
        reaction_id = str(inventory_row["reaction_id"])
        reasons = _reaction_reasons(
            inventory_row,
            ts_by_id.get(reaction_id),
            irc_by_id.get(reaction_id),
            g1_status_by_id.get(reaction_id),
            duplicates,
        )
        if not reasons:
            n_joined += 1
            continue
        for reason in reasons:
            reason_counts[reason] += 1
            if len(reason_examples[reason]) < MAX_EXAMPLES:
                reason_examples[reason].append(reaction_id)
    by_reason = {reason: reason_counts[reason] for reason in JOIN_REASONS if reason_counts[reason]}
    inventory_ids = {str(row["reaction_id"]) for row in inventory_rows}
    extra_ts = sorted(set(ts_by_id) - inventory_ids)
    extra_irc = sorted(set(irc_by_id) - inventory_ids)
    return {
        "schema_version": JOIN_AUDIT_SCHEMA_VERSION,
        "dataset_version": dataset_version,
        "n_inventory": len(inventory_rows),
        "n_inventory_unique": len(inventory_ids),
        "n_ts": len(ts_rows),
        "n_ts_unique": len(ts_by_id),
        "n_irc_index": len(irc_rows),
        "n_irc_index_unique": len(irc_by_id),
        "n_g1_summary": len(g1_summary_rows),
        "n_joined_all": n_joined,
        "n_not_joined": len(inventory_rows) - n_joined,
        "n_ts_ids_not_in_inventory": len(extra_ts),
        "n_irc_ids_not_in_inventory": len(extra_irc),
        "ts_extra_examples": extra_ts[:MAX_EXAMPLES],
        "irc_extra_examples": extra_irc[:MAX_EXAMPLES],
        "by_reason": by_reason,
        "reason_examples": {
            reason: ids for reason, ids in reason_examples.items() if ids
        },
        "digests": dict(digests or {}),
    }


__all__ = [
    "JOIN_REASONS",
    "MAX_EXAMPLES",
    "build_join_audit",
]
