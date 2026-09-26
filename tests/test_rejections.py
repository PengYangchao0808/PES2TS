"""Unit tests for the unified G0 rejection ledger and manifest constants."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from pes2ts_core.g0 import MANIFEST_SCHEMA_VERSION, VOLATILE_KEYS
from pes2ts_core.g0.rejections import (
    LEDGER_FILENAME,
    SUMMARY_FILENAME,
    Rejection,
    RejectionCode,
    RejectionLedger,
)

EXPECTED_CODES = {
    "BAD_ID",
    "DUPLICATE_ID",
    "BAD_SMILES",
    "NO_ARROW",
    "UNPARSEABLE_MAPPED",
    "MISSING_IN_H5",
    "MISSING_IN_CSV",
    "ID_MISMATCH",
    "ATOM_COUNT_MISMATCH",
    "ELEMENT_MISMATCH",
    "CHARGE_NOT_NEUTRAL",
    "SPIN_NOT_SINGLET",
    "BAD_GEOMETRY_SHAPE",
    "H5_SCHEMA_ERROR",
    "SPLIT_SCHEMA_ERROR",
    "NOT_IN_SPLIT",
    "DUPLICATE_OF",
    "REVERSE_OF",
    "LEAK_EXCLUDED",
    "AUDIT_BUDGET_EXCEEDED",
    "G1_MAP_ERROR",
    "G1_COMPONENT_MISMATCH",
    "G1_INDEX_MISMATCH",
    "G1_BOND_GEOMETRY",
    "G1_PREVIEW_CONFLICT",
    "G1_NO_BOND_CHANGE",
}


def _rejection(code: RejectionCode, reaction_id: str = "RXN_0000000001") -> Rejection:
    return Rejection(
        reaction_id=reaction_id,
        stage="inventory",
        code=code,
        detail=f"detail for {code.value}",
        source_pointer="B3LYPD3_TZVP.h5:/bundle/RXN_0000000001/R0",
    )


def test_manifest_contract_constants() -> None:
    assert MANIFEST_SCHEMA_VERSION == "g0_manifest_v1"
    assert VOLATILE_KEYS == {"generated_at", "downloaded_at", "duration_seconds"}


def test_rejection_code_members_are_the_agreed_contract() -> None:
    assert {member.value for member in RejectionCode} == EXPECTED_CODES


def test_ledger_writes_two_jsonl_lines_and_summary(tmp_path: Path) -> None:
    ledger = RejectionLedger(tmp_path)
    ledger.add(_rejection(RejectionCode.BAD_SMILES))
    ledger.add(_rejection(RejectionCode.MISSING_IN_H5, reaction_id="RXN_0000000002"))
    ledger.write()

    lines = (tmp_path / LEDGER_FILENAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    for record in records:
        assert set(record) == {
            "reaction_id",
            "stage",
            "code",
            "detail",
            "source_pointer",
        }
    assert records[0]["code"] == "BAD_SMILES"
    assert records[1]["code"] == "MISSING_IN_H5"
    assert records[1]["reaction_id"] == "RXN_0000000002"

    summary = json.loads((tmp_path / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert summary["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert summary["total"] == 2
    assert summary["by_code"] == {"BAD_SMILES": 1, "MISSING_IN_H5": 1}
    assert list(summary["by_code"]) == ["BAD_SMILES", "MISSING_IN_H5"]


def test_ledger_write_is_full_rewrite_from_memory(tmp_path: Path) -> None:
    ledger = RejectionLedger(tmp_path)
    ledger.add(_rejection(RejectionCode.BAD_ID))
    ledger.write()
    ledger.add(_rejection(RejectionCode.NO_ARROW, reaction_id="RXN_0000000003"))
    ledger.write()

    lines = (tmp_path / LEDGER_FILENAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["code"] == "NO_ARROW"
    summary = json.loads((tmp_path / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert summary["total"] == 2
    assert summary["by_code"] == {"BAD_ID": 1, "NO_ARROW": 1}


def test_ledger_load_tolerates_a_missing_file(tmp_path: Path) -> None:
    # Given: no ledger has ever been written
    # When: a stage loads the ledger and adds its first rejection
    ledger = RejectionLedger.load(tmp_path)
    ledger.add(_rejection(RejectionCode.BAD_ID))
    ledger.write()

    # Then: exactly the new rejection is persisted
    lines = (tmp_path / LEDGER_FILENAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["code"] == "BAD_ID"


def test_ledger_load_then_add_appends_to_existing(tmp_path: Path) -> None:
    # Given: an earlier stage persisted one rejection
    first = RejectionLedger(tmp_path)
    first.add(_rejection(RejectionCode.BAD_SMILES))
    first.write()

    # When: a later stage loads that ledger and adds its own rejection
    loaded = RejectionLedger.load(tmp_path)
    loaded.add(_rejection(RejectionCode.DUPLICATE_OF, reaction_id="RXN_0000000002"))
    loaded.write()

    # Then: the earlier line survives and the new one is appended
    lines = (tmp_path / LEDGER_FILENAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["code"] == "BAD_SMILES"
    assert json.loads(lines[1])["code"] == "DUPLICATE_OF"
    assert json.loads(lines[1])["reaction_id"] == "RXN_0000000002"
    summary = json.loads((tmp_path / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert summary["total"] == 2
    assert summary["by_code"] == {"BAD_SMILES": 1, "DUPLICATE_OF": 1}


def test_rejection_with_unknown_code_raises_value_error() -> None:
    with pytest.raises(ValueError, match="NOT_A_CODE"):
        dataclasses.replace(_rejection(RejectionCode.BAD_ID), code="NOT_A_CODE")


def test_ledger_add_rejects_unknown_code(tmp_path: Path) -> None:
    ledger = RejectionLedger(tmp_path)
    invalid = _rejection(RejectionCode.BAD_ID)
    object.__setattr__(invalid, "code", "NOT_A_CODE")
    with pytest.raises(ValueError, match="NOT_A_CODE"):
        ledger.add(invalid)
