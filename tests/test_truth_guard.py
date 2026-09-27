"""Tests for the ground-truth quarantine, its accessors, and the static guard.

The suite covers the isolation contract end to end: the quarantine extracts
TS geometries and a shape-only IRC index, relocates the truth-bearing HDF5
sources, the audited accessors refuse access without ``allow_truth=True`` and
log every successful read, and the AST guard fails any module outside the
allowlist that references the quarantined paths, imports the truth reader,
names the truth-bearing HDF5 files, or uses dynamic-execution primitives.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from pes2ts_core import cli
from pes2ts_core.g0 import MANIFEST_SCHEMA_VERSION
from pes2ts_core.g0.truth import truth_reader
from pes2ts_core.g0.truth_quarantine import (
    IRC_INDEX_COLUMNS,
    IRC_INDEX_FILENAME,
    TRUTH_MANIFEST_FILENAME,
    TS_COLUMNS,
    TS_PARQUET_FILENAME,
    quarantine_truth,
)
from pes2ts_core.utils.hashing import sha256_file
from pes2ts_core.utils.jsonio import read_json
from pes2ts_core.utils.parquet_io import read_parquet
from pes2ts_core.utils.truth_guard import (
    DEFAULT_ALLOWLIST,
    DYNAMIC_EXEC_RISK,
    TRUTH_FILE_REF,
    TRUTH_IMPORT,
    TRUTH_PATH_REF,
    TruthAccessViolation,
    assert_no_truth_access,
    scan_truth_access,
)

WATER_Z = (8, 1, 1)
WATER_X = ((0.0, 0.0, 0.117), (0.0, 0.757, -0.469), (0.0, -0.757, -0.469))
METHANE_Z = (6, 1, 1, 1, 1)
METHANE_X = (
    (0.0, 0.0, 0.0),
    (0.63, 0.63, 0.63),
    (-0.63, -0.63, 0.63),
    (0.63, -0.63, -0.63),
    (-0.63, 0.63, -0.63),
)
COMBINED_Z = (*METHANE_Z, *WATER_Z)
COMBINED_X = (*METHANE_X, *WATER_X)
TS_EHG = (-76.4, -76.3, -76.2)

MAIN_H5_NAME = "B3LYPD3_TZVP.h5"
IRC_H5_NAME = "B3LYPD3_TZVP_IRC.h5"
INFO_NAME = "B3LYPD3_TZVP_reaction_info.csv"
RXN_1 = "RXN_0000000001"
RXN_2 = "RXN_0000000002"

IRC_FRAMES_1 = np.arange(4 * 3 * 3, dtype=np.float64).reshape(4, 3, 3)
IRC_FRAMES_2 = np.arange(5 * 5 * 3, dtype=np.float64).reshape(5, 5, 3)
IRC_EHG_2 = np.arange(5 * 3, dtype=np.float64).reshape(5, 3)

REACTION_INFO_CSV = "reaction_id,reaction_smiles\n1,O>>O\n2,C.O>>C.O\n"


def _add_species(
    reaction: h5py.Group,
    tag: str,
    *,
    smiles: str,
    atomic_numbers: Sequence[int],
    coordinates: Sequence[Sequence[float]],
    EHG: Sequence[float] = TS_EHG,
) -> None:
    species = reaction.create_group(tag)
    species.create_dataset("smiles", data=np.bytes_(smiles))
    species.create_dataset("EHG", data=np.asarray(EHG, dtype=np.float64))
    species.create_dataset("charge", data=0)
    species.create_dataset("multiplicity", data=1)
    species.create_dataset(
        "atomic_numbers", data=np.asarray(atomic_numbers, dtype=np.int64)
    )
    species.create_dataset(
        "coordinates", data=np.asarray(coordinates, dtype=np.float64)
    )


def _build_main_h5(path: Path) -> Path:
    with h5py.File(path, "w") as handle:
        uni = handle.create_group(f"bundle_a/{RXN_1}")
        for tag in ("R0", "P0", "TS"):
            _add_species(
                uni,
                tag,
                smiles=f"label-{tag}",
                atomic_numbers=WATER_Z,
                coordinates=WATER_X,
            )
        bi = handle.create_group(f"bundle_b/{RXN_2}")
        _add_species(
            bi, "R0", smiles="C", atomic_numbers=METHANE_Z, coordinates=METHANE_X
        )
        _add_species(bi, "R1", smiles="O", atomic_numbers=WATER_Z, coordinates=WATER_X)
        _add_species(
            bi, "P0", smiles="C", atomic_numbers=METHANE_Z, coordinates=METHANE_X
        )
        _add_species(bi, "P1", smiles="O", atomic_numbers=WATER_Z, coordinates=WATER_X)
        _add_species(
            bi,
            "TS",
            smiles="label-TS",
            atomic_numbers=COMBINED_Z,
            coordinates=COMBINED_X,
        )
    return path


def _build_irc_h5(path: Path) -> Path:
    with h5py.File(path, "w") as handle:
        first = handle.create_group(f"bundle_a/{RXN_1}")
        first.create_dataset("coordinates", data=IRC_FRAMES_1)
        second = handle.create_group(f"bundle_a/{RXN_2}")
        nested = second.create_group("irc")
        nested.create_dataset("coordinates", data=IRC_FRAMES_2)
        nested.create_dataset("EHG", data=IRC_EHG_2)
        nested.create_dataset("forces", data=np.zeros_like(IRC_FRAMES_2))
    return path


def _fixture_config(tmp_path: Path) -> dict:
    ground_truth = tmp_path / "ground_truth"
    files = {
        name: {"filename": name, "url": "https://example.invalid/x", "md5": "0" * 32}
        for name in (MAIN_H5_NAME, IRC_H5_NAME, INFO_NAME)
    }
    return {
        "source": {"zenodo_record": 18551029, "zenodo_revision": "1", "files": files},
        "paths": {
            "raw": str(tmp_path / "raw"),
            "ground_truth": str(ground_truth),
            "truth_sources": str(ground_truth / "sources"),
            "manifests": str(tmp_path / "manifests"),
        },
    }


def _prepare_fixture(tmp_path: Path) -> dict:
    config = _fixture_config(tmp_path)
    raw = Path(config["paths"]["raw"])
    raw.mkdir(parents=True, exist_ok=True)
    _build_main_h5(raw / MAIN_H5_NAME)
    _build_irc_h5(raw / IRC_H5_NAME)
    (raw / INFO_NAME).write_text(REACTION_INFO_CSV, encoding="utf-8")
    return config


def test_real_package_has_no_unauthorized_truth_access() -> None:
    # Given: the shipped package plus bin scripts
    # When
    findings = assert_no_truth_access()
    # Then: no module outside the allowlist touches the quarantined artifacts
    assert findings == []


def test_guard_flags_truth_import_in_temp_module(tmp_path: Path) -> None:
    # Given: a module importing the truth reader
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "mod_bad.py").write_text(
        "from pes2ts_core.g0.truth import truth_reader\n", encoding="utf-8"
    )

    # When / Then
    with pytest.raises(TruthAccessViolation) as excinfo:
        assert_no_truth_access(package_root=package)
    kinds = {finding.kind for finding in excinfo.value.findings}
    assert TRUTH_IMPORT in kinds
    assert str(package / "mod_bad.py") in str(excinfo.value)


def test_guard_flags_paths_files_and_dynamic_exec(tmp_path: Path) -> None:
    # Given: a module naming the quarantined paths, the truth files, and
    # dynamic-execution primitives
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "mod_bad.py").write_text(
        "DATA = 'data/ground_truth/ts.parquet'\n"
        "MAIN = 'B3LYPD3_TZVP.h5'\n"
        "IRC = 'B3LYPD3_TZVP_IRC.h5'\n"
        "import subprocess\n"
        "import importlib\n"
        "VALUE = eval('1')\n",
        encoding="utf-8",
    )

    # When
    findings = scan_truth_access(package_root=package)

    # Then: every rule fires with a line number and a stable kind
    counts = Counter(finding.kind for finding in findings)
    assert counts[TRUTH_PATH_REF] == 1
    assert counts[TRUTH_FILE_REF] == 2
    assert counts[DYNAMIC_EXEC_RISK] == 3
    assert all(finding.line >= 1 for finding in findings)


def test_truth_access_without_allow_truth_raises_permission_error() -> None:
    # Given / When / Then: neither accessor opens anything without the flag
    with pytest.raises(PermissionError, match="allow_truth=True") as irc_exc:
        truth_reader.load_irc_frames(RXN_1, manifests_dir="/dev/null")
    assert RXN_1 in str(irc_exc.value)

    with pytest.raises(PermissionError, match="audited") as ts_exc:
        truth_reader.load_ts_geometry(RXN_1, manifests_dir="/dev/null")
    assert RXN_1 in str(ts_exc.value)

    with pytest.raises(PermissionError):
        truth_reader.load_irc_index(manifests_dir="/dev/null")


def test_allowlisted_module_is_not_flagged(tmp_path: Path) -> None:
    # Given: a temp module at the allowlisted relative path with truth strings
    root = tmp_path / "pes2ts_core"
    (root / "g0").mkdir(parents=True)
    (root / "g0" / "truth_quarantine.py").write_text(
        "PATH = 'data/ground_truth/ts.parquet'\n", encoding="utf-8"
    )

    # When / Then: the allowlist is honored
    assert scan_truth_access(package_root=root, allowlist=DEFAULT_ALLOWLIST) == []


def test_cli_truth_index_handler_is_exempt_but_rest_is_not(tmp_path: Path) -> None:
    # Given: a cli.py whose truth import sits inside the truth-index handler
    source = tmp_path / "src"
    source.mkdir()
    cli_path = source / "cli.py"
    handler = (
        "def _truth_index_handler(args):\n"
        "    from pes2ts_core.g0.truth import truth_reader\n"
        "    return truth_reader\n"
    )
    cli_path.write_text(handler, encoding="utf-8")

    # When / Then: the handler subtree is exempt
    assert scan_truth_access(package_root=source) == []

    # When: the same import appears outside the handler...
    cli_path.write_text(
        handler
        + "\ndef helper():\n"
        "    from pes2ts_core.g0.truth import truth_reader\n",
        encoding="utf-8",
    )

    # Then: the rest of cli.py is still checked
    findings = scan_truth_access(package_root=source)
    assert any(finding.kind == TRUTH_IMPORT for finding in findings)
    with pytest.raises(TruthAccessViolation):
        assert_no_truth_access(package_root=source)


def test_cli_truth_index_handler_rejects_subprocess(tmp_path: Path) -> None:
    # Given: a truth-index handler smuggling a subprocess import
    source = tmp_path / "src"
    source.mkdir()
    (source / "cli.py").write_text(
        "def _truth_index_handler(args):\n"
        "    import subprocess\n"
        "    return subprocess\n",
        encoding="utf-8",
    )

    # When
    findings = scan_truth_access(package_root=source)

    # Then
    assert [finding.kind for finding in findings] == [DYNAMIC_EXEC_RISK]


def test_g2_tree_is_clean_under_default_allowlists() -> None:
    # Given: the shipped g2 subtree (skeleton today; runner.py joins in task 5)
    g2_root = Path(__file__).resolve().parents[1] / "pes2ts_core" / "g2"
    # When / Then: with the default allowlists the whole tree passes — the
    # runner's dynamic-exec risk is exempt while its truth checks stay active
    assert assert_no_truth_access(package_root=g2_root) == []


def test_dynamic_exec_allowlist_suppresses_only_dynamic_exec(tmp_path: Path) -> None:
    # Given: a module at the dynamic-exec allowlisted relative path using
    # subprocess/importlib/eval — the sanctioned g2 runner shape
    root = tmp_path / "pes2ts_core"
    (root / "g2").mkdir(parents=True)
    runner = root / "g2" / "runner.py"
    runner.write_text(
        "import subprocess\n"
        "import importlib\n"
        "VALUE = eval('1')\n"
        "OTHER = getattr(importlib, '__import__')\n",
        encoding="utf-8",
    )

    # When: scanned with the default allowlists
    findings = scan_truth_access(package_root=root)

    # Then: every dynamic-exec finding is exempt and nothing else fires
    assert findings == []

    # When: the scoped exemption is disabled explicitly
    findings = scan_truth_access(package_root=root, dynamic_exec_allowlist=())

    # Then: the same module is flagged again, exclusively as DYNAMIC_EXEC_RISK
    assert {finding.kind for finding in findings} == {DYNAMIC_EXEC_RISK}
    assert all(finding.file == str(runner) for finding in findings)


def test_dynamic_exec_allowlist_module_still_reports_truth_refs(tmp_path: Path) -> None:
    # Given: a module on the dynamic-exec allowlist that also embeds a
    # quarantined-path string (the injection case)
    root = tmp_path / "pes2ts_core"
    (root / "g2").mkdir(parents=True)
    runner = root / "g2" / "runner.py"
    runner.write_text(
        "import subprocess\n"
        "TOKEN = 'data/ground_truth/ts.parquet'\n",
        encoding="utf-8",
    )

    # When
    findings = scan_truth_access(package_root=root)

    # Then: the truth reference is still reported and the suppression does
    # not leak to the truth checks
    assert [(finding.kind, finding.file) for finding in findings] == [
        (TRUTH_PATH_REF, str(runner))
    ]
    with pytest.raises(TruthAccessViolation):
        assert_no_truth_access(package_root=root)


@pytest.mark.parametrize("member", DEFAULT_ALLOWLIST)
def test_default_allowlist_members_remain_fully_exempt(
    tmp_path: Path, member: str
) -> None:
    # Given: a temp copy of each fully-exempt module containing truth strings
    # AND dynamic-exec primitives
    root = tmp_path / "pes2ts_core"
    module = tmp_path / member
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text(
        "import subprocess\n"
        "VALUE = eval('1')\n"
        "PATH = 'data/ground_truth/ts.parquet'\n",
        encoding="utf-8",
    )

    # When / Then: the full allowlist still exempts every check for its members
    assert scan_truth_access(package_root=root) == []


def test_real_g2_runner_when_present_is_fully_clean() -> None:
    # Given: the real runner module (absent until task 5 lands it)
    runner = (
        Path(__file__).resolve().parents[1] / "pes2ts_core" / "g2" / "runner.py"
    )
    if not runner.is_file():
        pytest.skip("pes2ts_core/g2/runner.py does not exist yet (task 5)")
    # When
    findings = [finding for finding in scan_truth_access() if finding.file == str(runner)]
    # Then: dynamic exec is exempt and no truth reference is present
    assert findings == []


def test_quarantine_writes_artifacts_relocates_sources_and_is_idempotent(
    tmp_path: Path,
) -> None:
    # Given: a raw tree with two tiny truth-bearing HDF5 files and a CSV
    config = _prepare_fixture(tmp_path)
    raw = Path(config["paths"]["raw"])
    ground_truth = Path(config["paths"]["ground_truth"])
    truth_sources = Path(config["paths"]["truth_sources"])
    manifests = Path(config["paths"]["manifests"])

    # When
    result = quarantine_truth(config)

    # Then: the TS table has one row per reaction with a TS species
    assert result.skipped is False
    assert set(result.relocated) == {MAIN_H5_NAME, IRC_H5_NAME}
    ts_rows = {
        row["reaction_id"]: row
        for row in read_parquet(ground_truth / TS_PARQUET_FILENAME).to_pylist()
    }
    assert set(ts_rows) == {RXN_1, RXN_2}
    assert set(read_parquet(ground_truth / TS_PARQUET_FILENAME).column_names) == set(
        TS_COLUMNS
    )
    assert ts_rows[RXN_1]["atomic_numbers"] == list(WATER_Z)
    assert ts_rows[RXN_1]["coordinates"] == [list(atom) for atom in WATER_X]
    assert ts_rows[RXN_1]["EHG"] == pytest.approx(list(TS_EHG))
    assert ts_rows[RXN_1]["charge"] == 0
    assert ts_rows[RXN_1]["multiplicity"] == 1
    assert ts_rows[RXN_1]["reaction_smiles"] == "O>>O"
    assert ts_rows[RXN_2]["reaction_smiles"] == "C.O>>C.O"
    assert len(ts_rows[RXN_2]["atomic_numbers"]) == len(COMBINED_Z)

    # And: the IRC index is shape-only (no frames, no energies)
    index_table = read_parquet(ground_truth / IRC_INDEX_FILENAME)
    index_rows = {row["reaction_id"]: row for row in index_table.to_pylist()}
    assert set(index_table.column_names) == set(IRC_INDEX_COLUMNS)
    assert index_rows[RXN_1] == {
        "reaction_id": RXN_1,
        "n_atoms": 3,
        "n_frames": 4,
        "has_forces": False,
        "ts_index": 0,
    }
    assert index_rows[RXN_2]["n_atoms"] == 5
    assert index_rows[RXN_2]["n_frames"] == 5
    assert index_rows[RXN_2]["has_forces"] is True
    assert "coordinates" not in index_table.column_names

    # And: both truth-bearing HDF5 files moved out of the raw tree
    assert (truth_sources / MAIN_H5_NAME).is_file()
    assert (truth_sources / IRC_H5_NAME).is_file()
    assert not (raw / MAIN_H5_NAME).exists()
    assert not (raw / IRC_H5_NAME).exists()
    # And: the non-truth CSV stays in the raw tree
    assert (raw / INFO_NAME).is_file()

    # And: the manifest records digests, sizes, and the TS-free reaction count
    manifest = read_json(manifests / TRUTH_MANIFEST_FILENAME)
    assert isinstance(manifest, dict)
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["dataset_version"] == "zenodo-18551029-rev1"
    assert manifest["ts_parquet"]["n_rows"] == 2
    assert manifest["ts_parquet"]["sha256"] == sha256_file(
        ground_truth / TS_PARQUET_FILENAME
    )
    assert manifest["irc_index"]["sha256"] == sha256_file(
        ground_truth / IRC_INDEX_FILENAME
    )
    sources = {entry["filename"]: entry for entry in manifest["sources"]}
    assert set(sources) == {MAIN_H5_NAME, IRC_H5_NAME}
    assert sources[MAIN_H5_NAME]["sha256"] == sha256_file(
        truth_sources / MAIN_H5_NAME
    )
    assert sources[MAIN_H5_NAME]["relocated_path"] == str(
        truth_sources / MAIN_H5_NAME
    )
    assert sources[MAIN_H5_NAME]["size_bytes"] == (
        truth_sources / MAIN_H5_NAME
    ).stat().st_size
    assert manifest["n_reactions_without_ts"] == 0

    # When: quarantine runs again on the already-relocated tree...
    quarantined = (
        ground_truth / TS_PARQUET_FILENAME,
        ground_truth / IRC_INDEX_FILENAME,
        truth_sources / MAIN_H5_NAME,
        truth_sources / IRC_H5_NAME,
    )
    before = {path.name: sha256_file(path) for path in quarantined}
    second = quarantine_truth(config)

    # Then: nothing is rewritten and the result reports the skip
    assert second.skipped is True
    assert second.n_ts_rows == 2
    assert second.n_irc_rows == 2
    assert second.relocated == ()
    assert {path.name: sha256_file(path) for path in quarantined} == before


def test_quarantine_reports_reactions_without_ts(tmp_path: Path) -> None:
    # Given: the fixture with the TS species removed from the second reaction
    config = _prepare_fixture(tmp_path)
    raw = Path(config["paths"]["raw"])
    with h5py.File(raw / MAIN_H5_NAME, "r+") as handle:
        del handle[f"bundle_b/{RXN_2}/TS"]

    # When
    result = quarantine_truth(config)

    # Then: the absence is counted, not rejected, and only one TS row exists
    assert result.n_reactions_without_ts == 1
    assert result.n_ts_rows == 1
    manifest = read_json(Path(config["paths"]["manifests"]) / TRUTH_MANIFEST_FILENAME)
    assert isinstance(manifest, dict)
    assert manifest["n_reactions_without_ts"] == 1


def test_reader_returns_geometry_and_appends_audit_log(tmp_path: Path) -> None:
    # Given: a completed quarantine
    config = _prepare_fixture(tmp_path)
    quarantine_truth(config)
    manifests = Path(config["paths"]["manifests"])
    log_path = manifests / truth_reader.TRUTH_ACCESS_LOG_FILENAME
    assert not log_path.exists()

    # When: the TS geometry is read with the explicit flag
    geometry = truth_reader.load_ts_geometry(
        RXN_1, allow_truth=True, manifests_dir=manifests
    )

    # Then: the row round-trips exactly
    assert geometry["reaction_id"] == RXN_1
    assert list(geometry["atomic_numbers"]) == list(WATER_Z)
    np.testing.assert_allclose(geometry["coordinates"], WATER_X)
    np.testing.assert_allclose(geometry["EHG"], TS_EHG)
    assert geometry["reaction_smiles"] == "O>>O"

    # And: one audit line was appended
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["reaction_id"] == RXN_1
    assert entry["function"] == "load_ts_geometry"
    assert entry["accessor_module"] == "pes2ts_core.g0.truth.truth_reader"
    assert entry["ts"]

    # When: the IRC frames and the index are read
    frames = truth_reader.load_irc_frames(
        RXN_1, allow_truth=True, manifests_dir=manifests
    )
    forced = truth_reader.load_irc_frames(
        RXN_2, allow_truth=True, manifests_dir=manifests
    )
    index_rows = truth_reader.load_irc_index(allow_truth=True, manifests_dir=manifests)

    # Then: frames are bounded to the requested reaction and the log grew
    assert frames["reaction_id"] == RXN_1
    assert frames["n_frames"] == 4
    assert frames["n_atoms"] == 3
    np.testing.assert_allclose(frames["coordinates"], IRC_FRAMES_1)
    assert frames["EHG"] is None
    assert forced["has_forces"] is True
    np.testing.assert_allclose(forced["EHG"], IRC_EHG_2)
    assert {row["reaction_id"] for row in index_rows} == {RXN_1, RXN_2}
    assert len(log_path.read_text(encoding="utf-8").splitlines()) == 4

    # And: IDs are normalized before lookup
    aliased = truth_reader.load_ts_geometry(
        "1", allow_truth=True, manifests_dir=manifests
    )
    assert aliased["reaction_id"] == RXN_1


def test_cli_quarantine_and_truth_index_roundtrip(tmp_path: Path) -> None:
    # Given: a raw fixture plus a config override written for the CLI
    config = _prepare_fixture(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"paths": config["paths"]}), encoding="utf-8"
    )

    # When / Then: the quarantine command succeeds and relocates the sources
    assert cli.main(["g0", "quarantine", "--config", str(config_path)]) == 0
    assert (Path(config["paths"]["truth_sources"]) / MAIN_H5_NAME).is_file()
    # And: the audited truth-index command reports the index
    assert cli.main(["g0", "truth-index", "--config", str(config_path)]) == 0
    # And: a reaction-scoped truth-index read succeeds
    assert (
        cli.main(
            ["g0", "truth-index", "--reaction-id", RXN_1, "--config", str(config_path)]
        )
        == 0
    )
