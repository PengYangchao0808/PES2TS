"""Offline end-to-end tests for the idempotent G0 pipeline and its run report.

Fixtures are synthetic and built under ``tmp_path``: one combined HDF5 with five
reactions (R/P plus TS species), one IRC HDF5 with two reactions, the
reaction-info CSV, and the three official split CSVs.  The four probe/train
SMILES reused from the freeze suite are the verified-clean audit fixture, so the
whole chain (inventory, quarantine relocation, dedup, official split adoption,
full DRFP audit, freeze, cohorts) runs for real; the chain-length ester analog
is reserved for the leak-halt case.  No network and no real data are involved.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from pes2ts_core.cli import main
from pes2ts_core.g0 import VOLATILE_KEYS
from pes2ts_core.g0.pipeline import (
    EXIT_PIPELINE_FAILED,
    STAGE_ORDER,
    run_pipeline,
    with_data_root,
)
from pes2ts_core.utils.hashing import md5_file
from pes2ts_core.utils.truth_guard import assert_no_truth_access

WATER_Z = (8, 1, 1)
WATER_X = ((0.0, 0.0, 0.117), (0.0, 0.757, -0.469), (0.0, -0.757, -0.469))
DEFAULT_EHG = (-76.4, -76.3, -76.2)
BUNDLE = "bundle_a"
MAIN_H5_NAME = "fixture_TZVP.h5"
IRC_H5_NAME = "fixture_TZVP_IRC.h5"
INFO_NAME = "fixture_reaction_info.csv"
SPLIT_NAMES: dict[str, str] = {
    "train": "fixture_train.csv",
    "valid": "fixture_valid.csv",
    "test": "fixture_test.csv",
}
SPLIT_FILENAMES: tuple[str, ...] = tuple(SPLIT_NAMES.values())


def _rid(number: int) -> str:
    """Return the canonical reaction ID for *number*."""
    return f"RXN_{number:010d}"


TRAIN_ANALOG = "CCCCCCCC(=O)O.CO>>CCCCCCCC(=O)OC.O"
TRAIN_AROMATIC = "c1ccc(cc1)C(=O)O.CCO>>c1ccc(cc1)C(=O)OCC.O"
TRAIN_DIELS = "C=CC=C.C=C>>C1CC=CCC1"
VALID_AMIDE = "CC(=O)O.CCN>>CC(=O)NCC.O"
TEST_UNRELATED = "c1ccccc1.Br>>Brc1ccccc1"
TEST_ANALOG = "CCCCCCCCC(=O)O.CO>>CCCCCCCCC(=O)OC.O"

#: Five reactions whose train/probe pairs the freeze suite proved audit-clean.
CLEAN_ROWS: tuple[tuple[str, str], ...] = (
    (_rid(1), TRAIN_ANALOG),
    (_rid(2), TRAIN_AROMATIC),
    (_rid(3), TRAIN_DIELS),
    (_rid(4), VALID_AMIDE),
    (_rid(5), TEST_UNRELATED),
)
CLEAN_SPLITS: dict[str, list[str]] = {
    "train": [_rid(1), _rid(2), _rid(3)],
    "valid": [_rid(4)],
    "test": [_rid(5)],
}
#: The chain-length ester analog (Tanimoto 1.0) leaks from train into test.
LEAK_ROWS: tuple[tuple[str, str], ...] = (
    (_rid(1), TRAIN_ANALOG),
    (_rid(2), TRAIN_AROMATIC),
    (_rid(3), TRAIN_DIELS),
    (_rid(4), TEST_ANALOG),
    (_rid(5), TEST_UNRELATED),
    (_rid(6), VALID_AMIDE),
)
LEAK_SPLITS: dict[str, list[str]] = {
    "train": [_rid(1), _rid(2), _rid(3)],
    "valid": [_rid(6)],
    "test": [_rid(4), _rid(5)],
}
RUN_REPORT_NAME = "g0_run_report.json"
STAGE_STATE_NAME = "g0_stage_state.json"


def _add_species(reaction: h5py.Group, tag: str, *, label: str = "O") -> None:
    """Add one synthetic R/P/TS species with balanced water geometry."""
    species = reaction.create_group(tag)
    species.create_dataset("smiles", data=np.bytes_(label))
    species.create_dataset("EHG", data=np.asarray(DEFAULT_EHG, dtype=np.float64))
    species.create_dataset("charge", data=0)
    species.create_dataset("multiplicity", data=1)
    species.create_dataset("atomic_numbers", data=np.asarray(WATER_Z, dtype=np.int64))
    species.create_dataset(
        "coordinates", data=np.asarray(WATER_X, dtype=np.float64)
    )


def _build_main_h5(path: Path, rows: Sequence[tuple[str, str]]) -> Path:
    """Build the combined archive with one R0/P0/TS set per fixture reaction."""
    with h5py.File(path, "w") as handle:
        bundle = handle.create_group(BUNDLE)
        for index, (reaction_id, _smiles) in enumerate(rows, start=1):
            reaction = bundle.create_group(reaction_id)
            _add_species(reaction, "R0", label=f"R-{index}")
            _add_species(reaction, "P0", label=f"P-{index}")
            _add_species(reaction, "TS", label=f"TS-{index}")
    return path


def _build_irc_h5(path: Path, reaction_ids: Sequence[str]) -> Path:
    """Build a shape-only IRC archive: direct coordinates under each reaction."""
    with h5py.File(path, "w") as handle:
        for index, reaction_id in enumerate(reaction_ids, start=1):
            frames = np.arange((index + 2) * len(WATER_Z) * 3, dtype=np.float64)
            frames = frames.reshape(index + 2, len(WATER_Z), 3)
            group = handle.create_group(f"{BUNDLE}/{reaction_id}")
            group.create_dataset("coordinates", data=frames)
    return path


def _write_info_csv(path: Path, rows: Sequence[tuple[str, str]]) -> None:
    """Write the reaction-info CSV consumed by the inventory and quarantine."""
    lines = ["reaction_id,reaction_smiles,dE_dagger,dE,dH_dagger,dH,dG_dagger,dG"]
    lines.extend(
        f"{reaction_id},{smiles},-1.0,-2.0,-3.0,-4.0,-5.0,-6.0"
        for reaction_id, smiles in rows
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_split_csv(path: Path, reaction_ids: Sequence[str]) -> None:
    """Write one official-split CSV carrying only reaction IDs."""
    path.write_text("\n".join(["reaction_id", *reaction_ids]) + "\n", encoding="utf-8")


def _fixture_config(root: Path) -> dict[str, Any]:
    """Return a config whose data root is ``root/data`` and files are unmapped."""
    data_root = root / "data"
    return {
        "source": {
            "zenodo_record": 18551029,
            "zenodo_revision": "1",
            "zenodo_doi": "10.5281/zenodo.18551029",
            "zenodo_modified": "2025-09-01",
            "files": {},
        },
        "paths": {
            "data_root": str(data_root),
            "raw": str(data_root / "raw" / "reaction_qm"),
            "interim": str(data_root / "interim"),
            "ground_truth": str(data_root / "ground_truth"),
            "truth_sources": str(data_root / "ground_truth" / "sources"),
            "manifests": str(data_root / "manifests"),
        },
        "split": {"seed": 42},
        "near_dup": {
            "threshold": 0.90,
            "cluster_threshold": 0.80,
            "fp_size": 2048,
            "block_size": 1000,
            "max_seconds": 7200.0,
            "max_pairs": 10000,
        },
        "cohorts": {
            "trial": 2,
            "stratified": 4,
            "require_authoritative_strata": False,
        },
    }


def _build_fixture(
    root: Path,
    rows: Sequence[tuple[str, str]] = CLEAN_ROWS,
    splits: Mapping[str, Sequence[str]] = CLEAN_SPLITS,
) -> dict[str, Any]:
    """Build the full source tree and return the fixture config."""
    config = _fixture_config(root)
    raw = Path(config["paths"]["raw"])
    raw.mkdir(parents=True, exist_ok=True)
    _build_main_h5(raw / MAIN_H5_NAME, rows)
    _build_irc_h5(raw / IRC_H5_NAME, [rows[0][0], rows[1][0]])
    _write_info_csv(raw / INFO_NAME, rows)
    for label, reaction_ids in splits.items():
        _write_split_csv(raw / SPLIT_NAMES[label], reaction_ids)
    config["source"]["files"] = {
        name: {
            "filename": name,
            "url": f"https://example.invalid/{name}",
            "md5": md5_file(raw / name),
        }
        for name in (MAIN_H5_NAME, IRC_H5_NAME, INFO_NAME, *SPLIT_FILENAMES)
    }
    return config


def _strip_volatile(document: Mapping[str, Any]) -> dict[str, Any]:
    """Drop the keys that legitimately change between two identical runs."""
    return {key: value for key, value in document.items() if key not in VOLATILE_KEYS}


def test_full_run_writes_every_manifest_and_relocates_sources(tmp_path: Path) -> None:
    # Given: the full synthetic dataset and a config with no manifest yet
    config = _build_fixture(tmp_path)

    # When: the pipeline runs with downloads skipped but sources verified
    report = run_pipeline(config, skip_fetch=True)

    # Then: every stage completed and every artifact exists
    assert report.ok is True
    assert report.leak_status == "clean"
    manifests = Path(config["paths"]["manifests"])
    interim = Path(config["paths"]["interim"])
    for name in (
        "source_manifest.json",
        "inventory_manifest.json",
        "truth_manifest.json",
        "duplicate_ledger.json",
        "leak_audit.json",
        "split_manifest.json",
        "strata_report.json",
        "rejection_summary.json",
        "rejection_ledger.jsonl",
        STAGE_STATE_NAME,
        RUN_REPORT_NAME,
    ):
        assert (manifests / name).is_file(), name
    for name in ("inventory.parquet", "cohort_trial.json", "cohort_stratified.json"):
        assert (interim / name).is_file(), name

    # And: fetch verified without downloading; the other stages all ran
    by_name = {stage.name: stage for stage in report.stages}
    assert [stage.name for stage in report.stages] == list(STAGE_ORDER)
    assert by_name["fetch"].status == "skipped"
    assert by_name["fetch"].reason == "flag"
    assert all(
        by_name[name].status == "ran" for name in STAGE_ORDER if name != "fetch"
    )

    # And: the audit verdict is complete and clean, and the split froze clean
    audit = json.loads((manifests / "leak_audit.json").read_text(encoding="utf-8"))
    assert audit["complete"] is True
    assert audit["n_pairs_over_threshold"] == 0
    split = json.loads((manifests / "split_manifest.json").read_text(encoding="utf-8"))
    assert split["leak_status"] == "clean"
    assert split["frozen"] is True
    assert split["covered"] == 5

    # And: the run report names every stage, its outputs, and the totals
    document = json.loads(report.report_path.read_text(encoding="utf-8"))
    assert document["schema_version"] == "g0_manifest_v1"
    assert [stage["name"] for stage in document["stages"]] == list(STAGE_ORDER)
    assert [stage["status"] for stage in document["stages"]] == [
        "skipped",
        *["ran"] * (len(STAGE_ORDER) - 1),
    ]
    assert document["totals"]["n_inventory_rows"] == 5
    assert document["totals"]["n_train"] == 3
    assert document["rejection_histogram"] == {}
    assert len(document["config_digest"]) == 64
    assert document["data_root"] == config["paths"]["data_root"]
    assert all(
        len(digest) == 64
        for stage in document["stages"]
        for digest in stage["outputs"].values()
    )
    assert all(stage["n_outputs"] == len(stage["outputs"]) for stage in document["stages"])

    # And: the two answer-bearing archives now live only under the relocated
    # sources tree, never in the raw tree
    raw = Path(config["paths"]["raw"])
    relocated = Path(config["paths"]["truth_sources"])
    assert (relocated / MAIN_H5_NAME).is_file()
    assert (relocated / IRC_H5_NAME).is_file()
    assert not (raw / MAIN_H5_NAME).exists()
    assert not (raw / IRC_H5_NAME).exists()
    assert sorted(path.name for path in raw.iterdir()) == sorted(
        [INFO_NAME, *SPLIT_FILENAMES]
    )

    # And: the pipeline itself never gained an unauthorized truth reference
    assert assert_no_truth_access() == []


def test_second_run_skips_every_stage_and_keeps_artifacts(tmp_path: Path) -> None:
    # Given: a completed pipeline run and a snapshot of every manifest
    config = _build_fixture(tmp_path)
    assert run_pipeline(config, skip_fetch=True).ok is True
    manifests = Path(config["paths"]["manifests"])
    volatile_names = {RUN_REPORT_NAME, STAGE_STATE_NAME}
    snapshot = {
        path.name: _strip_volatile(json.loads(path.read_text(encoding="utf-8")))
        for path in manifests.glob("*.json")
        if path.name not in volatile_names
    }
    ledger_before = (manifests / "rejection_ledger.jsonl").read_bytes()

    # When: the same config runs again
    second = run_pipeline(config, skip_fetch=True)

    # Then: every stage is skipped because its inputs are unchanged
    assert second.ok is True
    assert [stage.status for stage in second.stages] == ["skipped"] * len(STAGE_ORDER)
    assert all(stage.reason == "inputs unchanged" for stage in second.stages)
    assert second.leak_status == "clean"

    # And: no stage rewrote an artifact (manifests and ledger byte-stable
    # modulo the documented volatile keys)
    for path in manifests.glob("*.json"):
        if path.name in volatile_names:
            continue
        current = _strip_volatile(json.loads(path.read_text(encoding="utf-8")))
        assert current == snapshot[path.name], path.name
    assert (manifests / "rejection_ledger.jsonl").read_bytes() == ledger_before

    # And: the run report was rewritten with all-skipped stages
    document = json.loads(second.report_path.read_text(encoding="utf-8"))
    assert [stage["status"] for stage in document["stages"]] == [
        "skipped"
    ] * len(STAGE_ORDER)


def test_force_reruns_every_stage(tmp_path: Path) -> None:
    # Given: a fresh fixture and the force flag
    config = _build_fixture(tmp_path)

    # When
    report = run_pipeline(config, skip_fetch=True, force=True)

    # Then: nothing is skipped, so every stage reports real work
    assert report.ok is True
    assert [stage.status for stage in report.stages] == ["ran"] * len(STAGE_ORDER)
    assert all(stage.reason != "inputs unchanged" for stage in report.stages)
    assert report.leak_status == "clean"


def test_fetch_skip_is_honest_after_relocation(tmp_path: Path) -> None:
    # Given: a completed run whose fetch state entry is dropped (e.g. an older
    # sidecar) while the quarantine state entry remains
    config = _build_fixture(tmp_path)
    assert run_pipeline(config, skip_fetch=True).ok is True
    state_path = Path(config["paths"]["manifests"]) / STAGE_STATE_NAME
    document = json.loads(state_path.read_text(encoding="utf-8"))
    del document["stages"]["fetch"]
    state_path.write_text(json.dumps(document), encoding="utf-8")

    # When: the pipeline runs again with the raw archives already relocated
    report = run_pipeline(config, skip_fetch=True)

    # Then: fetch skips instead of re-verifying (or re-downloading) the moved
    # archives, and every later stage still skips normally
    assert report.ok is True
    by_name = {stage.name: stage for stage in report.stages}
    assert by_name["fetch"].status == "skipped"
    assert by_name["fetch"].reason == "sources relocated"
    assert all(
        by_name[name].status == "skipped" for name in STAGE_ORDER if name != "fetch"
    )


def test_corrupt_source_stops_at_fetch_without_split_manifest(tmp_path: Path) -> None:
    # Given: a fresh fixture whose combined archive is corrupted before run 1
    config = _build_fixture(tmp_path)
    main_h5 = Path(config["paths"]["raw"]) / MAIN_H5_NAME
    main_h5.write_bytes(main_h5.read_bytes()[:64])

    # When
    report = run_pipeline(config, skip_fetch=True)

    # Then: the run stops at the failing stage and exits non-zero
    assert report.ok is False
    assert [stage.name for stage in report.stages] == ["fetch"]
    failed = report.stages[0]
    assert failed.status == "failed"
    assert "ChecksumMismatch" in (failed.error or "")
    assert failed.traceback_tail
    assert failed.n_outputs == 0

    # And: no downstream artifact, and no partial split manifest, was written
    manifests = Path(config["paths"]["manifests"])
    assert not (manifests / "source_manifest.json").exists()
    assert not (manifests / "inventory_manifest.json").exists()
    assert not (manifests / "split_manifest.json").exists()

    # And: the run report names the failing stage with its boundary error
    document = json.loads(report.report_path.read_text(encoding="utf-8"))
    assert len(document["stages"]) == 1
    assert document["stages"][0]["name"] == "fetch"
    assert document["stages"][0]["status"] == "failed"
    assert document["stages"][0]["traceback_tail"]
    assert document["leak_status"] is None
    assert report.leak_status is None


def test_leak_halt_is_reported_without_cohorts(tmp_path: Path) -> None:
    # Given: the chain-length ester analog that leaks from train into test
    config = _build_fixture(tmp_path, rows=LEAK_ROWS, splits=LEAK_SPLITS)

    # When
    report = run_pipeline(config, skip_fetch=True)

    # Then: the freeze stage halts and the run is honestly unsuccessful
    assert report.ok is False
    assert report.leak_status == "LEAK_FOUND"
    failed = [stage for stage in report.stages if stage.status == "failed"]
    assert [stage.name for stage in failed] == ["freeze"]
    assert "LEAK_FOUND" in (failed[0].error or "")

    # And: the frozen manifest carries the verdict and no cohort was produced
    manifests = Path(config["paths"]["manifests"])
    split = json.loads((manifests / "split_manifest.json").read_text(encoding="utf-8"))
    assert split["leak_status"] == "LEAK_FOUND"
    assert split["frozen"] is True
    assert not (Path(config["paths"]["interim"]) / "cohort_trial.json").exists()
    assert not (manifests / "strata_report.json").exists()

    document = json.loads(report.report_path.read_text(encoding="utf-8"))
    assert document["leak_status"] == "LEAK_FOUND"
    assert document["stages"][-1]["name"] == "freeze"


def test_leak_remediation_exclude_leaky_completes(tmp_path: Path) -> None:
    # Given: the leaky fixture and the explicit exclusion remediation
    config = _build_fixture(tmp_path, rows=LEAK_ROWS, splits=LEAK_SPLITS)

    # When
    report = run_pipeline(config, skip_fetch=True, remediation="exclude-leaky")

    # Then: the pipeline completes, the held-out analog is excluded, and the
    # cohorts are produced from the remediated split
    assert report.ok is True
    assert report.leak_status == "clean"
    manifests = Path(config["paths"]["manifests"])
    split = json.loads((manifests / "split_manifest.json").read_text(encoding="utf-8"))
    assert split["leak_status"] == "clean"
    assert split["remediation"]["applied"] == "exclude-leaky"
    assert split["remediation"]["excluded_ids"] == [_rid(4)]
    assert (Path(config["paths"]["interim"]) / "cohort_trial.json").is_file()
    assert (manifests / "strata_report.json").is_file()


def test_cli_run_all_leak_halt_exits_non_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: the leaky fixture and the CLI wired to it
    config = _build_fixture(tmp_path, rows=LEAK_ROWS, splits=LEAK_SPLITS)
    monkeypatch.setattr("pes2ts_core.cli.load_config", lambda _path=None: config)

    # When / Then: the halt is reported and the frozen manifest keeps the verdict
    assert main(["g0", "run-all", "--skip-fetch"]) == EXIT_PIPELINE_FAILED
    manifest = Path(config["paths"]["manifests"]) / "split_manifest.json"
    assert json.loads(manifest.read_text(encoding="utf-8"))["leak_status"] == "LEAK_FOUND"


def test_cli_run_all_with_data_root_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Given: a fixture built at one root and copied to another
    config = _build_fixture(tmp_path / "first")
    shutil.copytree(tmp_path / "first" / "data", tmp_path / "second" / "data")
    monkeypatch.setattr("pes2ts_core.cli.load_config", lambda _path=None: config)

    # When: the CLI re-roots the config at the copied tree
    exit_code = main(
        [
            "g0",
            "run-all",
            "--data-root",
            str(tmp_path / "second" / "data"),
            "--skip-fetch",
        ]
    )

    # Then: only the copied tree received the artifacts
    assert exit_code == 0
    assert (tmp_path / "second" / "data" / "manifests" / RUN_REPORT_NAME).is_file()
    assert not (tmp_path / "first" / "data" / "manifests" / RUN_REPORT_NAME).exists()
    assert "run-all:" in capsys.readouterr().out


def test_cli_run_all_exits_non_zero_when_a_stage_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a corrupted fixture and the CLI wired to it
    config = _build_fixture(tmp_path)
    main_h5 = Path(config["paths"]["raw"]) / MAIN_H5_NAME
    main_h5.write_bytes(main_h5.read_bytes()[:32])
    monkeypatch.setattr("pes2ts_core.cli.load_config", lambda _path=None: config)

    # When / Then
    assert main(["g0", "run-all", "--skip-fetch"]) == EXIT_PIPELINE_FAILED


def test_with_data_root_swaps_relative_and_absolute_prefixes(tmp_path: Path) -> None:
    # Given: a default-style relative config
    relative: dict[str, Any] = {
        "paths": {
            "data_root": "data",
            "raw": "data/raw/reaction_qm",
            "interim": "data/interim",
            "ground_truth": "data/ground_truth",
            "truth_sources": "data/ground_truth/sources",
            "manifests": "data/manifests",
        }
    }

    # When
    swapped = with_data_root(relative, "fixtures/_fixture_data")

    # Then: every derived path moved with the root, and the caller is untouched
    assert swapped["paths"] == {
        "data_root": "fixtures/_fixture_data",
        "raw": "fixtures/_fixture_data/raw/reaction_qm",
        "interim": "fixtures/_fixture_data/interim",
        "ground_truth": "fixtures/_fixture_data/ground_truth",
        "truth_sources": "fixtures/_fixture_data/ground_truth/sources",
        "manifests": "fixtures/_fixture_data/manifests",
    }
    assert relative["paths"]["raw"] == "data/raw/reaction_qm"

    # And: an absolute data root is swapped the same way
    absolute: dict[str, Any] = {
        "paths": {
            "data_root": str(tmp_path / "data"),
            "raw": str(tmp_path / "data" / "raw"),
            "manifests": str(tmp_path / "data" / "manifests"),
        }
    }
    rooted = with_data_root(absolute, tmp_path / "other")
    assert rooted["paths"]["raw"] == str(tmp_path / "other" / "raw")
    assert rooted["paths"]["manifests"] == str(tmp_path / "other" / "manifests")
    assert absolute["paths"]["raw"] == str(tmp_path / "data" / "raw")


def test_run_pipeline_rejects_unknown_remediation(tmp_path: Path) -> None:
    # Given: a valid fixture with an unsupported policy value
    config = _build_fixture(tmp_path)

    # When / Then
    with pytest.raises(ValueError, match="Unknown remediation"):
        run_pipeline(config, skip_fetch=True, remediation="repair")
