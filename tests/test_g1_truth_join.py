"""Join-audit fixtures for the P1.0 ID coverage layer.

Every fixture is a plain-rows table (the module is pure data logic), so the
tests cover the exact reason taxonomy: missing TS/IRC/G1 records, duplicate
ids, atom-count conflicts, element/charge/spin/smiles conflicts, and extra
truth ids -- the denominators the plan requires to reconcile exactly.
"""

from __future__ import annotations

from typing import Any

from pes2ts_core.g1.truth_join import (
    REASON_ATOM_COUNT_CONFLICT,
    REASON_CHARGE_CONFLICT,
    REASON_ELEMENT_CONFLICT,
    REASON_INVENTORY_DUPLICATE_ID,
    REASON_IRC_DUPLICATE_ID,
    REASON_MISSING_G1,
    REASON_MISSING_IRC,
    REASON_MISSING_TS,
    REASON_SMILES_CONFLICT,
    REASON_SPIN_CONFLICT,
    REASON_TS_DUPLICATE_ID,
    build_join_audit,
)


def _inventory(rid: str, *, n_r: int = 3, n_p: int = 3, elements: tuple[str, ...] = ("O", "H", "H"), smiles: str = "A>>B") -> dict[str, Any]:
    return {
        "reaction_id": rid, "reaction_smiles": smiles,
        "total_atoms_reactants": n_r, "total_atoms_products": n_p,
        "charge_total_reactants": 0, "charge_total_products": 0,
        "multiplicity_max": 1, "elements": list(elements),
    }


def _ts(rid: str, *, numbers: tuple[int, ...] = (8, 1, 1), coordinates: list[list[float]] | None = None, smiles: str = "A>>B", charge: int = 0, multiplicity: int = 1) -> dict[str, Any]:
    if coordinates is None:
        coordinates = [[0.0, 0.0, float(i)] for i in range(len(numbers))]
    return {
        "reaction_id": rid, "atomic_numbers": list(numbers),
        "coordinates": coordinates, "EHG": [-1.0, -1.0, -1.0],
        "charge": charge, "multiplicity": multiplicity, "reaction_smiles": smiles,
    }


def _irc(rid: str, *, n_atoms: int = 3, n_frames: int = 10) -> dict[str, Any]:
    return {"reaction_id": rid, "n_atoms": n_atoms, "n_frames": n_frames, "has_forces": False, "ts_index": 0}


def _g1(rid: str, status: str = "valid") -> dict[str, Any]:
    return {"reaction_id": rid, "status": status}


def test_happy_join_counts_every_reaction() -> None:
    audit = build_join_audit(
        [_inventory("RXN_1"), _inventory("RXN_2")],
        [_ts("RXN_1"), _ts("RXN_2")],
        [_irc("RXN_1"), _irc("RXN_2")],
        [_g1("RXN_1"), _g1("RXN_2")],
    )
    assert audit["n_joined_all"] == 2
    assert audit["n_not_joined"] == 0
    assert audit["by_reason"] == {}
    assert audit["schema_version"] == "g1_p1_join_audit_v1"


def test_missing_ts_irc_and_g1_are_separate_reasons() -> None:
    audit = build_join_audit(
        [_inventory("RXN_1"), _inventory("RXN_2"), _inventory("RXN_3")],
        [_ts("RXN_1")],
        [_irc("RXN_1"), _irc("RXN_2")],
        [_g1("RXN_1")],
    )
    assert audit["by_reason"][REASON_MISSING_TS] == 2
    assert audit["by_reason"][REASON_MISSING_IRC] == 1
    assert audit["by_reason"][REASON_MISSING_G1] == 2
    assert audit["n_joined_all"] == 1


def test_duplicate_ids_are_flagged_per_table() -> None:
    audit = build_join_audit(
        [_inventory("RXN_1"), _inventory("RXN_1")],
        [_ts("RXN_1"), _ts("RXN_1")],
        [_irc("RXN_1"), _irc("RXN_1")],
        [_g1("RXN_1")],
    )
    assert audit["by_reason"][REASON_INVENTORY_DUPLICATE_ID] == 2
    assert audit["by_reason"][REASON_TS_DUPLICATE_ID] == 2
    assert audit["by_reason"][REASON_IRC_DUPLICATE_ID] == 2


def test_atom_count_conflict_between_ts_irc_and_inventory() -> None:
    audit = build_join_audit(
        [_inventory("RXN_1")],
        [_ts("RXN_1", numbers=(8, 1))],
        [_irc("RXN_1", n_atoms=3)],
        [_g1("RXN_1")],
    )
    # Both independent checks fire: TS vs inventory totals and TS vs IRC.
    assert audit["by_reason"][REASON_ATOM_COUNT_CONFLICT] == 2


def test_element_set_conflict() -> None:
    # The inventory ``elements`` column is the distinct symbol set; the TS
    # numbers are compared as the same distinct set (counts are covered by
    # the separate atom-count check).
    audit = build_join_audit(
        [_inventory("RXN_1", elements=("O", "H"))],
        [_ts("RXN_1", numbers=(8, 6, 1))],
        [_irc("RXN_1")],
        [_g1("RXN_1")],
    )
    assert audit["by_reason"][REASON_ELEMENT_CONFLICT] == 1
    matching = build_join_audit(
        [_inventory("RXN_1", elements=("H", "O"))],
        [_ts("RXN_1", numbers=(8, 1, 1))],
        [_irc("RXN_1")],
        [_g1("RXN_1")],
    )
    assert matching["by_reason"].get(REASON_ELEMENT_CONFLICT, 0) == 0


def test_charge_spin_and_smiles_conflicts() -> None:
    audit = build_join_audit(
        [_inventory("RXN_1"), _inventory("RXN_2"), _inventory("RXN_3")],
        [
            _ts("RXN_1", charge=1),
            _ts("RXN_2", multiplicity=3),
            _ts("RXN_3", smiles="different>>smiles"),
        ],
        [_irc("RXN_1"), _irc("RXN_2"), _irc("RXN_3")],
        [_g1("RXN_1"), _g1("RXN_2"), _g1("RXN_3")],
    )
    assert audit["by_reason"][REASON_CHARGE_CONFLICT] == 1
    assert audit["by_reason"][REASON_SPIN_CONFLICT] == 1
    assert audit["by_reason"][REASON_SMILES_CONFLICT] == 1


def test_extra_truth_ids_are_counted_with_examples() -> None:
    audit = build_join_audit(
        [_inventory("RXN_1")],
        [_ts("RXN_1"), _ts("RXN_99")],
        [_irc("RXN_1"), _irc("RXN_98")],
        [_g1("RXN_1")],
    )
    assert audit["n_ts_ids_not_in_inventory"] == 1
    assert audit["n_irc_ids_not_in_inventory"] == 1
    assert audit["ts_extra_examples"] == ["RXN_99"]
    assert audit["irc_extra_examples"] == ["RXN_98"]


def test_bad_ts_shape_is_typed() -> None:
    audit = build_join_audit(
        [_inventory("RXN_1")],
        [_ts("RXN_1", coordinates=[[0.0, 0.0]])],
        [_irc("RXN_1")],
        [_g1("RXN_1")],
    )
    assert audit["by_reason"]["ts_bad_shape"] == 1


def test_digests_and_versions_are_recorded() -> None:
    audit = build_join_audit(
        [_inventory("RXN_1")], [_ts("RXN_1")], [_irc("RXN_1")], [_g1("RXN_1")],
        digests={"inventory": "abc", "ts": "def"}, dataset_version="zenodo-x-rev1",
    )
    assert audit["digests"] == {"inventory": "abc", "ts": "def"}
    assert audit["dataset_version"] == "zenodo-x-rev1"
