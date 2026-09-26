"""Offline tests for the authoritative G1 strata rebuild and ``g1 strata`` CLI.

Fixtures reuse the module-level synthetic builders from ``tests/test_g1_build``
(imported directly), so every reaction-change document, summary, and rejection
in the tree is produced by the real ``g1 build`` stage before the authoritative
records are derived.  No network and no real data are needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pes2ts_core.cli import main
from pes2ts_core.g0 import VOLATILE_KEYS
from pes2ts_core.g0.inventory import INVENTORY_PARQUET_FILENAME
from pes2ts_core.g0.strata import (
    COHORT_STRATIFIED_FILENAME,
    COHORT_TRIAL_FILENAME,
    STRATA_REPORT_FILENAME,
    AuthoritativeStrataRequiredError,
    compute_strata,
    select_cohorts,
)
from pes2ts_core.g1 import G1_MANIFEST_SCHEMA_VERSION
from pes2ts_core.g1.build import EXIT_G1_BUILD_FAILED, build_g1
from pes2ts_core.g1.strata_auth import (
    SENTINEL,
    STRATA_MANIFEST_FILENAME,
    STRATA_SOURCE_G1,
    authoritative_records,
    rebuild_authoritative_strata,
)
from pes2ts_core.utils.hashing import sha256_bytes
from pes2ts_core.utils.parquet_io import read_parquet
from test_g1_build import (
    SAME_SMILES,
    _Reaction,
    _config,
    _happy,
    _water,
    _write_inventory,
)


def _fixture_config(root: Path, *, require_authoritative: bool = False) -> dict[str, Any]:
    """Return the test_g1_build config plus the cohorts block selection needs."""
    config = _config(root)
    config["cohorts"] = {
        "trial": 2,
        "stratified": 3,
        "require_authoritative_strata": require_authoritative,
    }
    return config


def _mixed_reactions() -> list[_Reaction]:
    """Two valid bond changes plus one zero-change rejection."""
    return [
        _happy("RXN_0000000001"),
        _water("RXN_0000000002"),
        _happy("RXN_0000000003", smiles=SAME_SMILES),
    ]


def _counters(record: dict[str, Any]) -> tuple[Any, Any, Any]:
    """Return the three bond-change counters of a strata record."""
    return (
        record["n_bonds_formed"],
        record["n_bonds_broken"],
        record["n_bond_order_changed"],
    )


def _load(path: Path) -> dict[str, Any]:
    """Read a JSON artifact written by the stage under test."""
    return json.loads(path.read_text(encoding="utf-8"))


def _strip_volatile(document: dict[str, Any]) -> dict[str, Any]:
    """Drop the keys that legitimately change between runs."""
    return {key: value for key, value in document.items() if key not in VOLATILE_KEYS}


def test_authoritative_records_replace_preview_counters_and_keep_rejected(
    tmp_path: Path,
) -> None:
    # Given: a 3-row inventory fully built (2 valid, 1 zero-change rejection)
    config = _fixture_config(tmp_path)
    _write_inventory(config, _mixed_reactions())
    build = build_g1(config)
    assert (build.n_total, build.n_valid, build.n_rejected) == (3, 2, 1)
    interim = Path(config["paths"]["interim"])
    preview_by_id = {
        str(record["reaction_id"]): record
        for record in compute_strata(read_parquet(interim / INVENTORY_PARQUET_FILENAME))
    }
    summary_by_id = {
        str(row["reaction_id"]): row for row in read_parquet(build.summary_path).to_pylist()
    }

    # When
    records, basis = authoritative_records(config)

    # Then: one record per inventory row, in inventory order
    assert [str(record["reaction_id"]) for record in records] == [
        "RXN_0000000001",
        "RXN_0000000002",
        "RXN_0000000003",
    ]
    by_id = {str(record["reaction_id"]): record for record in records}

    # And: valid rows carry the build's counters, equal to the preview they
    # replace (both fixture reactions break exactly one O-H bond)
    for reaction_id in ("RXN_0000000001", "RXN_0000000002"):
        record, row = by_id[reaction_id], summary_by_id[reaction_id]
        assert row["status"] == "valid"
        assert _counters(record) == (
            row["n_formed"],
            row["n_broken"],
            row["n_order_changed"],
        )
        assert _counters(record) == _counters(preview_by_id[reaction_id])
        assert _counters(record) == (0, 1, 0)

    # And: the rejected row stays in the records with the sentinel counters,
    # replacing preview counters that were a genuine zero trio
    rejected = by_id["RXN_0000000003"]
    assert summary_by_id["RXN_0000000003"]["status"] == "rejected"
    assert _counters(preview_by_id["RXN_0000000003"]) == (0, 0, 0)
    assert _counters(rejected) == (SENTINEL, SENTINEL, SENTINEL)
    # And: its stratum axes are still G0's, so it stays row-comparable
    assert rejected["element_set"] == "C,Cl,F,H,O"
    assert rejected["n_components"] == 3
    assert rejected["heavy_atom_bucket"] == "small"

    # And: the basis reconciles the denominator with the build
    assert basis == {
        "n_inventory": 3,
        "n_built": 3,
        "n_valid": 2,
        "n_rejected": 1,
        "n_unbuilt": 0,
        "n_counter_agreements": 2,
    }


def test_rebuild_rewrites_artifacts_with_authoritative_provenance(tmp_path: Path) -> None:
    # Given: a preview selection already on disk, then a full build
    config = _fixture_config(tmp_path)
    _write_inventory(config, _mixed_reactions())
    build_g1(config)
    preview = select_cohorts(config)
    assert _load(preview.report_path)["g1_obligation"]

    # When
    result = rebuild_authoritative_strata(config)

    # Then: the result names every artifact
    assert (result.n_inventory, result.n_valid, result.n_rejected) == (3, 2, 1)
    assert result.report_path == tmp_path / "manifests" / STRATA_REPORT_FILENAME
    assert result.trial_path == tmp_path / "interim" / COHORT_TRIAL_FILENAME
    assert result.stratified_path == tmp_path / "interim" / COHORT_STRATIFIED_FILENAME
    assert result.manifest_path == tmp_path / "manifests" / STRATA_MANIFEST_FILENAME

    # And: the report is the authoritative rewrite of the preview report
    report = _load(result.report_path)
    assert report["strata_source"] == STRATA_SOURCE_G1
    assert report["authoritative"] is True
    assert "g1_obligation" not in report
    assert report["seed_stratified"] == int(sha256_bytes(b"42:stratified"), 16)
    assert report["totals"] == {
        "n_inventory": 3,
        "n_cohort_trial": 2,
        "n_cohort_stratified": 3,
        "n_strata": 2,
    }

    # And: both cohorts were rewritten and keep the nested trial invariant
    trial, stratified = _load(result.trial_path), _load(result.stratified_path)
    assert trial["cohort"] == "trial" and trial["size"] == 2
    assert stratified["cohort"] == "stratified" and stratified["size"] == 3
    assert set(trial["members"]) <= set(stratified["members"])

    # And: the side manifest records the basis, seeds, strata, and paths
    manifest = _load(result.manifest_path)
    assert manifest["schema_version"] == G1_MANIFEST_SCHEMA_VERSION
    assert manifest["dataset_version"] == "zenodo-1-rev1"
    assert manifest["basis"] == {
        "n_inventory": 3,
        "n_built": 3,
        "n_valid": 2,
        "n_rejected": 1,
        "n_unbuilt": 0,
        "n_counter_agreements": 2,
    }
    assert manifest["seed"] == 42
    assert manifest["seed_stratified"] == int(sha256_bytes(b"42:stratified"), 16)
    assert manifest["n_strata"] == report["totals"]["n_strata"] == 2
    assert manifest["artifacts"] == {
        "strata_report": str(result.report_path),
        "cohort_trial": str(result.trial_path),
        "cohort_stratified": str(result.stratified_path),
    }
    assert manifest["generated_at"]


def test_authoritative_selection_matches_preview_membership(tmp_path: Path) -> None:
    # Given: an all-valid inventory, so no sentinel is involved
    config = _fixture_config(tmp_path)
    _write_inventory(config, [_happy("RXN_0000000001"), _water("RXN_0000000002")])
    build_g1(config)
    preview = select_cohorts(config)
    preview_trial = _load(preview.trial_path)["members"]
    preview_stratified = _load(preview.stratified_path)["members"]

    # When: the authoritative path re-derives the same cohorts
    result = rebuild_authoritative_strata(config)

    # Then: membership is byte-equal -- the selection rules did not drift
    assert _load(result.trial_path)["members"] == preview_trial
    assert _load(result.stratified_path)["members"] == preview_stratified
    assert preview_trial == sorted(preview_trial)
    assert preview_stratified == sorted(preview_stratified)


def test_rebuild_ignores_require_authoritative_halt(tmp_path: Path) -> None:
    # Given: the config that halts the preview wrapper
    config = _fixture_config(tmp_path, require_authoritative=True)
    _write_inventory(config, [_happy("RXN_0000000001"), _water("RXN_0000000002")])
    build_g1(config)
    with pytest.raises(AuthoritativeStrataRequiredError):
        select_cohorts(config)

    # When
    result = rebuild_authoritative_strata(config)

    # Then: the authoritative path never applies the preview-only halt
    assert (result.n_inventory, result.n_valid) == (2, 2)
    assert _load(result.report_path)["authoritative"] is True


def test_partial_build_raises_and_cli_exits_22(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a full inventory but a summary covering only one row
    config = _fixture_config(tmp_path)
    _write_inventory(config, [_happy("RXN_0000000001"), _water("RXN_0000000002")])
    build_g1(config, reaction_ids=["RXN_0000000001"])
    monkeypatch.setattr("pes2ts_core.cli.load_config", lambda _path=None: config)

    # When / Then: the library refuses the partial summary with the hint
    with pytest.raises(ValueError, match="full build over every inventory row"):
        rebuild_authoritative_strata(config)

    # And: the CLI maps the refusal to exit 22
    assert main(["g1", "strata"]) == EXIT_G1_BUILD_FAILED


def test_cli_strata_happy_path_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Given: a built tree and the CLI wired to it
    config = _fixture_config(tmp_path)
    _write_inventory(config, _mixed_reactions())
    build_g1(config)
    monkeypatch.setattr("pes2ts_core.cli.load_config", lambda _path=None: config)

    # When
    exit_code = main(["g1", "strata"])

    # Then: exit 0, one INFO line, and every artifact on disk
    assert exit_code == 0
    assert "G1 strata: 3 inventory row(s), 2 authoritative, report ->" in capsys.readouterr().err
    assert (tmp_path / "manifests" / STRATA_MANIFEST_FILENAME).is_file()
    assert (tmp_path / "manifests" / STRATA_REPORT_FILENAME).is_file()
    assert (tmp_path / "interim" / COHORT_TRIAL_FILENAME).is_file()
    assert (tmp_path / "interim" / COHORT_STRATIFIED_FILENAME).is_file()


def test_missing_inputs_raise_typed_errors_and_cli_exits_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: an empty tree
    config = _fixture_config(tmp_path)
    monkeypatch.setattr("pes2ts_core.cli.load_config", lambda _path=None: config)
    with pytest.raises(FileNotFoundError, match="g0 inventory"):
        rebuild_authoritative_strata(config)
    assert main(["g1", "strata"]) == 3

    # Given: an inventory but no build summary
    _write_inventory(config, [_happy("RXN_0000000001")])
    with pytest.raises(FileNotFoundError, match="g1 build"):
        authoritative_records(config)
    with pytest.raises(FileNotFoundError, match="g1 build"):
        rebuild_authoritative_strata(config)
    assert main(["g1", "strata"]) == 3


def test_rebuild_is_deterministic_modulo_generated_at(tmp_path: Path) -> None:
    # Given: a built tree rebuilt once
    config = _fixture_config(tmp_path)
    _write_inventory(config, _mixed_reactions())
    build_g1(config)
    first = rebuild_authoritative_strata(config)
    first_paths = (
        first.report_path,
        first.trial_path,
        first.stratified_path,
        first.manifest_path,
    )
    before = [_strip_volatile(_load(path)) for path in first_paths]

    # When: the same config rebuilds again
    second = rebuild_authoritative_strata(config)

    # Then: every artifact is identical except generated_at
    after = [
        _strip_volatile(_load(path))
        for path in (second.report_path, second.trial_path, second.stratified_path, second.manifest_path)
    ]
    assert after == before
    assert _load(second.stratified_path)["members"] == _load(first.stratified_path)["members"]
