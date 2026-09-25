"""Offline tests for the stratification preview and deterministic cohorts.

Fixtures are synthetic: a 40-reaction inventory Parquet spanning four strata
(25/8/5/2 over the ``(element_set, n_components, heavy_atom_bucket)`` key)
plus small hand-written records for the mapped bond-diff preview and the
bucket boundaries.  No network and no real data are needed.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from pes2ts_core.cli import main
from pes2ts_core.g0 import VOLATILE_KEYS
from pes2ts_core.g0.inventory import INVENTORY_PARQUET_FILENAME
from pes2ts_core.g0.strata import (
    COHORT_STRATIFIED_FILENAME,
    COHORT_TRIAL_FILENAME,
    EXIT_AUTHORITATIVE_STRATA_REQUIRED,
    STRATA_REPORT_FILENAME,
    AuthoritativeStrataRequiredError,
    compute_bond_changes_preview,
    compute_strata,
    select_cohorts,
)
from pes2ts_core.utils.hashing import sha256_bytes
from pes2ts_core.utils.parquet_io import write_parquet

#: Unchanged mapped reaction used where only the row fields matter.
UNCHANGED_SMILES = "[CH3:1][OH:2]>>[CH3:1][OH:2]"
#: Mapped SN2 step: the C-Br bond breaks and a C-O bond forms.
SN2_SMILES = "[CH3:1][Br:2].[O-:3]>>[CH3:1][O-:3].[Br-:2]"
#: Mapped ethane -> ethene: one bond keeps its map pair but changes order.
ORDER_CHANGE_SMILES = "[CH3:1][CH3:2]>>[CH2:1]=[CH2:2]"
#: Small mapped reaction (6 total atoms with hydrogens).
SMALL_SMILES = "[CH3:1][OH:2]>>[CH2:1]=[O:2]"
#: Medium mapped reaction (10 total atoms with hydrogens).
MEDIUM_SMILES = "[CH3:1][CH2:2][NH2:3]>>[CH3:1][CH:2]=[NH:3]"
#: Large mapped reaction (21 total atoms with hydrogens).
LARGE_SMILES = (
    "[CH3:1][CH2:2][CH2:3][CH2:4][CH2:5][CH2:6][OH:7]"
    ">>[CH3:1][CH2:2][CH2:3][CH2:4][CH2:5][CH2:6][OH:7]"
)
#: Extra-large mapped reaction (38 total atoms with hydrogens).
XL_SMILES = (
    "[CH3:1][CH2:2][CH2:3][CH2:4][CH2:5][CH2:6][CH2:7][CH2:8][CH2:9]"
    "[CH2:10][CH2:11][CH3:12]"
    ">>[CH3:1][CH2:2][CH2:3][CH2:4][CH2:5][CH2:6][CH2:7][CH2:8][CH2:9]"
    "[CH2:10][CH2:11][CH3:12]"
)

SENTINELS = {
    "n_bonds_formed": -1,
    "n_bonds_broken": -1,
    "n_bond_order_changed": -1,
}

#: (count, elements, total_atoms_reactants, reaction_smiles) per fixture stratum.
STRATUM_SPECS: tuple[tuple[int, tuple[str, ...], int, str], ...] = (
    (25, ("C", "H", "O"), 6, SMALL_SMILES),
    (8, ("C", "H", "N"), 10, MEDIUM_SMILES),
    (5, ("C", "H"), 38, XL_SMILES),
    (2, ("C", "H", "O"), 21, LARGE_SMILES),
)
#: Hand-computed largest-remainder quotas for stratified=21 over the fixture.
EXPECTED_STRATIFIED_STRATA = {
    "C,H,O|2|small": 13,
    "C,H,N|2|medium": 4,
    "C,H|2|xl": 3,
    "C,H,O|2|large": 1,
}
#: Trial quotas recomputed from the stratified members with trial=4.
EXPECTED_TRIAL_STRATA = {
    "C,H,O|2|small": 2,
    "C,H,N|2|medium": 1,
    "C,H|2|xl": 1,
}
INVENTORY_FIELDS: tuple[str, ...] = (
    "reaction_id",
    "reaction_smiles",
    "elements",
    "n_reactant_components",
    "n_product_components",
    "total_atoms_reactants",
)


def _inventory_rows() -> list[dict[str, Any]]:
    """Build the 40-row fixture inventory (strata of 25/8/5/2)."""
    rows: list[dict[str, Any]] = []
    for count, elements, total_atoms, smiles in STRATUM_SPECS:
        for _ in range(count):
            rows.append(
                {
                    "reaction_id": f"RXN_{len(rows) + 1:010d}",
                    "reaction_smiles": smiles,
                    "elements": list(elements),
                    "n_reactant_components": 1,
                    "n_product_components": 1,
                    "total_atoms_reactants": total_atoms,
                }
            )
    return rows


def _table(rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    """Build the synthetic inventory table from row mappings."""
    return pa.table(
        {field: [row[field] for row in rows] for field in INVENTORY_FIELDS}
    )


def _write_inventory(
    tmp_path: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    """Write *rows* as the fixture inventory Parquet."""
    write_parquet(tmp_path / "interim" / INVENTORY_PARQUET_FILENAME, _table(rows))


def _fixture_config(
    tmp_path: Path,
    *,
    trial: int = 4,
    stratified: int = 21,
    seed: int = 42,
    require_authoritative: bool = False,
) -> dict[str, Any]:
    """Build the config consumed by :func:`select_cohorts`."""
    return {
        "source": {"zenodo_record": 18551029, "zenodo_revision": "1"},
        "paths": {
            "interim": str(tmp_path / "interim"),
            "manifests": str(tmp_path / "manifests"),
        },
        "split": {"seed": seed},
        "cohorts": {
            "trial": trial,
            "stratified": stratified,
            "require_authoritative_strata": require_authoritative,
        },
    }


def _load(path: Path) -> dict[str, Any]:
    """Read a JSON artifact written by the stage under test."""
    return json.loads(path.read_text(encoding="utf-8"))


def test_bond_preview_counts_formed_and_broken_bonds() -> None:
    # Given: a mapped SN2 step (C-Br breaks, C-O forms)
    record = {"reaction_smiles": SN2_SMILES}

    # When
    preview = compute_bond_changes_preview(record)

    # Then
    assert preview == {
        "n_bonds_formed": 1,
        "n_bonds_broken": 1,
        "n_bond_order_changed": 0,
    }


def test_bond_preview_counts_order_change_in_all_three_counters() -> None:
    # Given: ethane -> ethene (same map pair, order 1 -> 2)
    record = {"reaction_smiles": ORDER_CHANGE_SMILES}

    # When
    preview = compute_bond_changes_preview(record)

    # Then: the (pair, order) key sets are not exclusive, so the change shows
    # up as one broken key, one formed key, and one order-changed pair
    assert preview == {
        "n_bonds_formed": 1,
        "n_bonds_broken": 1,
        "n_bond_order_changed": 1,
    }


def test_bond_preview_unchanged_graph_is_zero() -> None:
    # Given / When
    preview = compute_bond_changes_preview({"reaction_smiles": UNCHANGED_SMILES})

    # Then
    assert preview == {
        "n_bonds_formed": 0,
        "n_bonds_broken": 0,
        "n_bond_order_changed": 0,
    }


@pytest.mark.parametrize(
    "record",
    [
        pytest.param({"reaction_smiles": "CCO>>CC=O"}, id="unmapped"),
        pytest.param({"reaction_smiles": "[CH3:1]CO>>[CH3:1]C=O"}, id="partially-mapped"),
        pytest.param({"reaction_smiles": "[CH3:1]C%^&>>[CH3:1]O"}, id="unparseable"),
        pytest.param({"reaction_smiles": "[CH3:1]O"}, id="no-arrow"),
        pytest.param({"reaction_smiles": "[CH3:1]O>>"}, id="empty-side"),
        pytest.param({"reaction_smiles": None}, id="none-cell"),
        pytest.param({}, id="missing-key"),
    ],
)
def test_bond_preview_unavailable_returns_sentinels(record: dict[str, Any]) -> None:
    # Given / When
    preview = compute_bond_changes_preview(record)

    # Then: a clean -1 sentinel instead of a crash
    assert preview == SENTINELS


def test_compute_strata_bucket_boundaries_and_fields() -> None:
    # Given: rows at every bucket boundary (8/9, 16/17, 28/29)
    totals = (8, 9, 16, 17, 28, 29)
    rows = [
        {
            "reaction_id": f"RXN_{index:010d}",
            "reaction_smiles": UNCHANGED_SMILES,
            "elements": ["C", "H", "O"],
            "n_reactant_components": 1,
            "n_product_components": 2,
            "total_atoms_reactants": total,
        }
        for index, total in enumerate(totals, start=1)
    ]

    # When: the Parquet-table input form is exercised here
    records = compute_strata(_table(rows))

    # Then
    assert [record["heavy_atom_bucket"] for record in records] == [
        "small",
        "medium",
        "medium",
        "large",
        "large",
        "xl",
    ]
    assert records[0]["element_set"] == "C,H,O"
    assert records[0]["n_components"] == 3
    assert records[0]["n_bonds_formed"] == 0
    assert records[0]["n_bonds_broken"] == 0
    assert records[0]["n_bond_order_changed"] == 0


def test_compute_strata_carries_preview_and_unmapped_sentinels() -> None:
    # Given: a mapped SN2 row and an unmapped row
    rows = [
        {
            "reaction_id": "RXN_0000000001",
            "reaction_smiles": SN2_SMILES,
            "elements": ["Br", "C", "H", "O"],
            "n_reactant_components": 2,
            "n_product_components": 2,
            "total_atoms_reactants": 7,
        },
        {
            "reaction_id": "RXN_0000000002",
            "reaction_smiles": "CCO>>CC=O",
            "elements": ["C", "H", "O"],
            "n_reactant_components": 1,
            "n_product_components": 1,
            "total_atoms_reactants": 9,
        },
    ]

    # When
    records = compute_strata(_table(rows))

    # Then
    assert set(records[0]) == {
        "reaction_id",
        "element_set",
        "n_components",
        "heavy_atom_bucket",
        "n_bonds_formed",
        "n_bonds_broken",
        "n_bond_order_changed",
    }
    assert records[0]["n_bonds_formed"] == 1
    assert records[0]["n_bonds_broken"] == 1
    assert records[0]["n_bond_order_changed"] == 0
    assert records[0]["heavy_atom_bucket"] == "small"
    assert records[1]["element_set"] == "C,H,O"
    assert records[1]["heavy_atom_bucket"] == "medium"
    assert records[1]["n_bonds_formed"] == -1
    assert records[1]["n_bonds_broken"] == -1
    assert records[1]["n_bond_order_changed"] == -1


def test_select_cohorts_sizes_quotas_and_nesting(tmp_path: Path) -> None:
    # Given: 40 inventory reactions in strata of 25/8/5/2
    _write_inventory(tmp_path, _inventory_rows())
    config = _fixture_config(tmp_path)

    # When
    result = select_cohorts(config)

    # Then: exact sizes from the hand-computed largest-remainder allocation
    assert result.trial_size == 4
    assert result.stratified_size == 21
    assert result.trial_path == tmp_path / "interim" / COHORT_TRIAL_FILENAME
    assert result.stratified_path == tmp_path / "interim" / COHORT_STRATIFIED_FILENAME
    assert result.report_path == tmp_path / "manifests" / STRATA_REPORT_FILENAME

    trial = _load(result.trial_path)
    stratified = _load(result.stratified_path)
    report = _load(result.report_path)
    assert trial["schema_version"] == "g0_manifest_v1"
    assert trial["dataset_version"] == "zenodo-18551029-rev1"
    assert trial["cohort"] == "trial"
    assert trial["size"] == len(trial["members"]) == 4
    assert trial["members"] == sorted(trial["members"])
    assert stratified["cohort"] == "stratified"
    assert stratified["size"] == len(stratified["members"]) == 21
    assert set(trial["members"]) <= set(stratified["members"])
    assert {
        key: count for key, count in trial["strata"].items() if count
    } == EXPECTED_TRIAL_STRATA
    assert {
        key: count for key, count in stratified["strata"].items() if count
    } == EXPECTED_STRATIFIED_STRATA
    assert sum(trial["strata"].values()) == 4
    assert sum(stratified["strata"].values()) == 21

    # And: the zero-cohort stratum is reported explicitly, never dropped
    assert trial["strata"]["C,H,O|2|large"] == 0


def test_select_cohorts_strata_report_contract(tmp_path: Path) -> None:
    # Given / When
    _write_inventory(tmp_path, _inventory_rows())
    result = select_cohorts(_fixture_config(tmp_path))
    report = _load(result.report_path)

    # Then: preview provenance and the mandatory G1 obligation note
    assert report["schema_version"] == "g0_manifest_v1"
    assert report["dataset_version"] == "zenodo-18551029-rev1"
    assert report["strata_source"] == "preview"
    assert report["authoritative"] is False
    assert "G1 must" in report["g1_obligation"]
    assert report["seed"] == 42
    assert report["seed_stratified"] == int(sha256_bytes(b"42:stratified"), 16)

    # And: every stratum is listed once, sorted, with counts summing to sizes
    keys = [row["key"] for row in report["strata"]]
    assert keys == sorted(keys)
    rows = {row["key"]: row for row in report["strata"]}
    assert rows["C,H,O|2|large"] == {
        "key": "C,H,O|2|large",
        "n_inventory": 2,
        "n_cohort_trial": 0,
        "n_cohort_stratified": 1,
    }
    assert report["totals"] == {
        "n_inventory": 40,
        "n_cohort_trial": 4,
        "n_cohort_stratified": 21,
        "n_strata": 4,
    }
    assert sum(row["n_inventory"] for row in report["strata"]) == 40
    assert sum(row["n_cohort_trial"] for row in report["strata"]) == 4
    assert sum(row["n_cohort_stratified"] for row in report["strata"]) == 21


def _docs(result: Any) -> list[dict[str, Any]]:
    """Return the three JSON documents of a CohortsResult."""
    return [
        _load(result.trial_path),
        _load(result.stratified_path),
        _load(result.report_path),
    ]


def _strip_volatile(document: dict[str, Any]) -> dict[str, Any]:
    """Drop the keys that legitimately change between runs."""
    return {
        key: value for key, value in document.items() if key not in VOLATILE_KEYS
    }


def test_select_cohorts_is_deterministic_modulo_timestamp(tmp_path: Path) -> None:
    # Given
    _write_inventory(tmp_path, _inventory_rows())
    config = _fixture_config(tmp_path)

    # When: the same config is run twice
    first = _docs(select_cohorts(config))
    second = _docs(select_cohorts(config))

    # Then: every artifact is identical except the generated_at timestamp
    for first_doc, second_doc in zip(first, second, strict=True):
        assert _strip_volatile(first_doc) == _strip_volatile(second_doc)
    assert first[0]["members"] == second[0]["members"]
    assert first[1]["members"] == second[1]["members"]


def test_select_cohorts_seed_changes_membership(tmp_path: Path) -> None:
    # Given: one inventory selected under two different seeds
    _write_inventory(tmp_path, _inventory_rows())

    # When
    base = _docs(select_cohorts(_fixture_config(tmp_path, seed=42)))
    other = _docs(select_cohorts(_fixture_config(tmp_path, seed=43)))

    # Then: the hash-ranked membership differs for both cohorts
    assert base[0]["members"] != other[0]["members"]
    assert base[1]["members"] != other[1]["members"]


def test_select_cohorts_smaller_inventory_selects_all(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Given: a 40-reaction inventory and requests for 100/1000
    rows = _inventory_rows()
    _write_inventory(tmp_path, rows)

    # When
    with caplog.at_level(logging.WARNING, logger="pes2ts_core.g0.strata"):
        result = select_cohorts(_fixture_config(tmp_path, trial=100, stratified=1000))

    # Then: both cohorts hold the whole inventory, sizes reported honestly
    all_ids = sorted(str(row["reaction_id"]) for row in rows)
    assert result.trial_size == result.stratified_size == 40
    assert _load(result.stratified_path)["members"] == all_ids
    assert _load(result.trial_path)["members"] == all_ids
    assert _load(result.report_path)["totals"]["n_cohort_trial"] == 40
    assert "selecting all" in caplog.text


def test_select_cohorts_require_authoritative_halts(tmp_path: Path) -> None:
    # Given: the authoritative-strata requirement is enabled
    _write_inventory(tmp_path, _inventory_rows())
    config = _fixture_config(tmp_path, require_authoritative=True)

    # When / Then: the typed error names the G1 obligation and nothing is written
    with pytest.raises(AuthoritativeStrataRequiredError, match="G1"):
        select_cohorts(config)

    assert not (tmp_path / "interim" / COHORT_TRIAL_FILENAME).exists()
    assert not (tmp_path / "interim" / COHORT_STRATIFIED_FILENAME).exists()
    assert not (tmp_path / "manifests" / STRATA_REPORT_FILENAME).exists()


def test_select_cohorts_missing_inventory_raises(tmp_path: Path) -> None:
    # Given: no inventory Parquet exists
    config = _fixture_config(tmp_path)

    # When / Then
    with pytest.raises(FileNotFoundError, match="g0 inventory"):
        select_cohorts(config)


def test_cli_cohorts_writes_artifacts_and_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Given: the CLI whose config loader returns the fixture config
    _write_inventory(tmp_path, _inventory_rows())
    config = _fixture_config(tmp_path)
    monkeypatch.setattr("pes2ts_core.cli.load_config", lambda _path=None: config)

    # When
    exit_code = main(["g0", "cohorts"])

    # Then
    assert exit_code == 0
    assert (tmp_path / "interim" / COHORT_TRIAL_FILENAME).is_file()
    assert (tmp_path / "interim" / COHORT_STRATIFIED_FILENAME).is_file()
    assert (tmp_path / "manifests" / STRATA_REPORT_FILENAME).is_file()
    assert "cohorts: trial=4 stratified=21" in capsys.readouterr().out


def test_cli_cohorts_authoritative_requirement_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: the authoritative-strata requirement is enabled
    _write_inventory(tmp_path, _inventory_rows())
    config = _fixture_config(tmp_path, require_authoritative=True)
    monkeypatch.setattr("pes2ts_core.cli.load_config", lambda _path=None: config)

    # When
    exit_code = main(["g0", "cohorts"])

    # Then
    assert exit_code == EXIT_AUTHORITATIVE_STRATA_REQUIRED
    assert not (tmp_path / "interim" / COHORT_TRIAL_FILENAME).exists()
