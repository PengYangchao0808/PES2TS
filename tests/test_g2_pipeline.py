"""TDD tests for the G2 run-orchestration pipeline (plan task 9).

Covers :func:`select_reaction_ids`, :func:`prepare_ids`, and :func:`run_ids`
against a self-contained fake xTB binary (scenario-selected via an env var):
the nine frozen scenarios (valid forward, forward-fail/reverse-success with
energy renormalization, both-fail, ineligible ledger, idempotent resume,
``--force`` rerun, failure+force history, missing ``xtbpath.xyz``, and the
no-retry-on-``G2_XTB_FAILED`` rule), the post-F-wave ``npath``-mismatch
scenario, and the selection/prepare typed failures.
All fixtures are synthetic and written under ``tmp_path`` roots.
"""

# allow: SIZE_OK -- the plan names exactly one test file for the pipeline
# task; the nine frozen fake-xTB scenarios plus selection/prepare unit
# coverage and their fixture builders.

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import pytest
from pes2ts_core.config_loader import load_config
from pes2ts_core.g0.rejections import LEDGER_FILENAME, RejectionCode
from pes2ts_core.g1.build import reaction_change_path, shard_name
from pes2ts_core.generation.execution.xtb_path import SCHEMA_PATH
from pes2ts_core.generation.execution.xtb_path.pipeline import (
    InfrastructureError,
    prepare_ids,
    run_ids,
    select_reaction_ids,
)
from pes2ts_core.utils.hashing import sha256_file
from pes2ts_core.utils.jsonio import read_json, write_json
from pes2ts_core.utils.parquet_io import read_parquet, write_parquet

#: Mapped O-H bond breaking: ``[F:1][C:2]([H:3])([Cl:4])[O:5][H:6]>>[F:1][C:2]([H:3])([Cl:4])[O:5].[H:6]``
SMILES: Final[str] = "[F:1][C:2]([H:3])([Cl:4])[O:5][H:6]>>[F:1][C:2]([H:3])([Cl:4])[O:5].[H:6]"
ELEMENTS: Final[tuple[str, ...]] = ("F", "C", "H", "Cl", "O", "H")
ATOMIC_NUMBERS: Final[tuple[int, ...]] = (9, 6, 1, 17, 8, 1)
COORDS: Final[tuple[tuple[float, float, float], ...]] = (
    (0.0, 0.0, 0.0), (1.35, 0.0, 0.0), (1.9, 0.6, 0.3),
    (1.7, -1.2, 0.6), (2.6, 0.7, -0.2), (2.2, 1.5, 0.2),
)
REACTION_A: Final[str] = "RXN_0000000001"
REACTION_B: Final[str] = "RXN_0000000002"
SHARD_SIZE: Final[int] = 1000
CATEGORY_NAMES: Final[tuple[str, ...]] = (
    "pure_formed", "pure_broken", "both", "order_change_only",
    "has_order_change", "h_migration", "multi_component",
)

FAKE_XTB: Final[str] = '''#!/usr/bin/env python3
"""Fake GFN2-xTB PATH binary; scenario selected via $FAKE_XTB_SCENARIO."""
import itertools
import math
import os
import sys

FRAME_COUNT = 8


def read_xyz(name):
    with open(name, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    count = int(lines[0].split()[0])
    atoms = []
    for line in lines[2:2 + count]:
        parts = line.split()
        atoms.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    return atoms


def spread(atoms):
    return max(math.dist(a[1:], b[1:]) for a, b in itertools.combinations(atoms, 2))


def interpolated(start, end, count):
    frames = []
    for index in range(count):
        t = index / (count - 1)
        atoms = []
        for (element, sx, sy, sz), (_, ex, ey, ez) in zip(start, end):
            atoms.append((element, sx + t * (ex - sx), sy + t * (ey - sy), sz + t * (ez - sz)))
        frames.append(atoms)
    return frames


def write_frames(path, frames, base):
    with open(path, "w", encoding="utf-8") as handle:
        for atoms in frames:
            energy = 25.0 * (spread(atoms) - base)
            handle.write(f"{len(atoms)}\\n energy: {energy:.6f} xtb: 6.7.1 (fake)\\n")
            for element, x, y, z in atoms:
                handle.write(f"{element} {x:.6f} {y:.6f} {z:.6f}\\n")


def main():
    if "--help" in sys.argv:
        print("fake xtb PATH binary")
        return 0
    scenario = os.environ.get("FAKE_XTB_SCENARIO", "good")
    counter = os.environ.get("FAKE_XTB_COUNTER")
    if counter:
        with open(counter, "a", encoding="utf-8") as handle:
            handle.write(os.path.basename(os.getcwd()) + "\\n")
    reverse = os.path.basename(os.getcwd()) == "run_reverse"
    if scenario == "bad" or (scenario == "both_fail" and reverse):
        print("fake xtb: intentional failure")
        return 3
    with open("xtb_path.log", "w", encoding="utf-8") as handle:
        handle.write("* xtb version 6.7.1 (fake)\\n")
        handle.write("   forward barrier (kcal): 12.500000\\n")
        handle.write("   backward barrier (kcal): 12.500000\\n")
        handle.write("   reaction energy  (kcal): 25.000000\\n")
        if scenario == "npath_mismatch":
            handle.write("   npath: 99\\n")
    if scenario == "missing":
        return 0
    start, end = read_xyz("start.xyz"), read_xyz("end.xyz")
    count = 4 if scenario in ("short", "both_fail") and not reverse else FRAME_COUNT
    frames = interpolated(start, end, count)
    base = min(spread(start), spread(end))
    write_frames("xtbpath.xyz", frames, base)
    write_frames("xtbpath_ts.xyz", [frames[len(frames) // 2]], base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def _rows(count: int, elements: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        {"local_index": i, "map": i + 1, "element": elements[i], "global_index": i}
        for i in range(count)
    ]


def _component(tag: str, numbers: tuple[int, ...], coords: tuple[tuple[float, float, float], ...]) -> dict[str, Any]:
    return {
        "tag": tag, "smiles": "", "atomic_numbers": list(numbers),
        "coordinates": [list(coord) for coord in coords],
    }


def _inventory_row(reaction_id: str) -> dict[str, Any]:
    return {
        "reaction_id": reaction_id, "reaction_smiles": SMILES,
        "charge_total_reactants": 0, "charge_total_products": 0, "multiplicity_max": 1,
        "components": [
            _component("R0", ATOMIC_NUMBERS, COORDS),
            _component("P0", ATOMIC_NUMBERS[:5], COORDS[:5]),
            _component("P1", (1,), (COORDS[5],)),
        ],
    }


def _g1_document(
    reaction_id: str, *, status: str = "valid", drop_product_map: int | None = None,
) -> dict[str, Any]:
    product_rows = _rows(5, ELEMENTS[:5])
    if drop_product_map is not None:
        product_rows = [row for row in product_rows if row["map"] != drop_product_map]
    categories = dict.fromkeys(CATEGORY_NAMES, False)
    categories["pure_broken"] = True
    categories["multi_component"] = True
    return {
        "schema_version": "g1_change_v1", "reaction_id": reaction_id, "reaction_smiles": SMILES,
        "reactants": [{"tag": "R0", "rows": _rows(6, ELEMENTS)}],
        "products": [
            {"tag": "P0", "rows": product_rows},
            {"tag": "P1", "rows": [{"local_index": 0, "map": 6, "element": "H", "global_index": 0}]},
        ],
        "reaction_center": {"core": [5, 6], "with_shell": [2, 5, 6]},
        "bond_changes": {
            "formed": [], "broken": [{"atoms": [5, 6]}],
            "order_changed": [], "hydrogen_migration": [],
        },
        "categories": categories,
        "validation": {"status": status, "failure_code": None, "failure_detail": None},
    }


def _write_inputs(
    interim: Path, *,
    eligible_ids: tuple[str, ...], cohort_ids: tuple[str, ...],
    inventory_ids: tuple[str, ...], g1_ids: tuple[str, ...],
) -> None:
    interim.mkdir(parents=True, exist_ok=True)
    write_json(interim / "g2_eligible.json", {
        "reaction_ids": sorted(eligible_ids), "n_eligible": len(eligible_ids),
    })
    write_json(interim / "cohort_trial.json", {
        "cohort": "trial", "size": len(cohort_ids), "members": sorted(cohort_ids),
    })
    ids = sorted(inventory_ids)
    write_parquet(interim / "inventory.parquet", {
        "reaction_id": ids,
        "reaction_smiles": [SMILES] * len(ids),
        "charge_total_reactants": [0] * len(ids),
        "charge_total_products": [0] * len(ids),
        "multiplicity_max": [1] * len(ids),
        "components": [_inventory_row(rid)["components"] for rid in ids],
    })
    for reaction_id in g1_ids:
        path = reaction_change_path(interim, reaction_id, SHARD_SIZE)
        write_json(path, _g1_document(reaction_id))


def _fake_xtb(tmp_path: Path) -> Path:
    script = tmp_path / "fake_xtb.py"
    script.write_text(FAKE_XTB, encoding="utf-8")
    script.chmod(0o755)
    return script


def _config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, scenario: str,
    eligible_ids: tuple[str, ...] = (REACTION_A,), cohort_ids: tuple[str, ...] | None = None,
    inventory_ids: tuple[str, ...] | None = None, g1_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    config = load_config()
    interim = tmp_path / "interim"
    manifests = tmp_path / "manifests"
    config["paths"]["interim"] = str(interim)
    config["paths"]["manifests"] = str(manifests)
    config["g2"]["xtb"]["executable"] = str(_fake_xtb(tmp_path))
    config["g2"]["xtb"]["timeout_seconds"] = 60
    monkeypatch.setenv("FAKE_XTB_SCENARIO", scenario)
    monkeypatch.setenv("FAKE_XTB_COUNTER", str(tmp_path / "xtb_calls.log"))
    _write_inputs(
        interim,
        eligible_ids=eligible_ids,
        cohort_ids=cohort_ids if cohort_ids is not None else eligible_ids,
        inventory_ids=inventory_ids if inventory_ids is not None else eligible_ids,
        g1_ids=g1_ids if g1_ids is not None else eligible_ids,
    )
    return config


def _reaction_dir(config: dict[str, Any], reaction_id: str) -> Path:
    interim = Path(config["paths"]["interim"])
    return interim / "g2" / "paths" / shard_name(reaction_id, SHARD_SIZE) / reaction_id


def _ledger_entries(manifests: Path) -> list[dict[str, str]]:
    path = manifests / LEDGER_FILENAME
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _frame_energies(path: Path) -> tuple[list[dict[str, Any]], list[float]]:
    rows = read_parquet(path).to_pylist()
    return rows, [float(row["energy_rel_kcal"]) for row in rows]


def _counter_calls(tmp_path: Path) -> int:
    path = tmp_path / "xtb_calls.log"
    return len(path.read_text(encoding="utf-8").splitlines()) if path.is_file() else 0


# --------------------------------------------------------------------------
# Integration scenarios 1-9 (fake xTB)
# --------------------------------------------------------------------------


def test_run_valid_forward_produces_terminal_artifacts(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch, scenario="good")
    prepare_ids([REACTION_A], config=config)
    report = run_ids([REACTION_A], config=config)
    rxn_dir = _reaction_dir(config, REACTION_A)
    assert (report.n_selected, report.n_attempted, report.n_skipped) == (1, 1, 0)
    document = read_json(rxn_dir / "reaction_path.json")
    assert document["schema_version"] == SCHEMA_PATH
    assert document["status"] == "valid"
    assert document["direction"] == "forward"
    assert document["direction_recovered"] is False
    assert document["failure_code"] is None
    attempt = document["attempts"][0]
    assert attempt["xtb_version_line"] == "* xtb version 6.7.1 (fake)"
    assert attempt["seed_supported"] is False
    assert attempt["seed"] is None
    assert attempt["omp_num_threads"] == "4"
    rows, energies = _frame_energies(rxn_dir / "frames.parquet")
    assert len(rows) >= 8
    assert all(row["energy_rel_kcal_raw"] == pytest.approx(row["energy_rel_kcal"]) for row in rows)
    for name in ("endpoints.json", "R.xyz", "P.xyz", "run/start.xyz", "run/end.xyz",
                 "run/path.inp", "run/xtb_path.log", "run/xtbpath.xyz", "run/xtbpath_ts.xyz"):
        assert (rxn_dir / name).is_file(), name
    assert (rxn_dir / "run/start.xyz").read_bytes() == (rxn_dir / "R.xyz").read_bytes()
    manifest = read_json(Path(config["paths"]["manifests"]) / "g2_path_manifest.json")
    assert manifest["run"] == {"n_selected": 1, "n_attempted": 1, "n_skipped": 0}
    xtb = manifest["xtb"]
    assert xtb["sha256"] == sha256_file(Path(config["g2"]["xtb"]["executable"]))
    assert xtb["version"] == "* xtb version 6.7.1 (fake)"
    assert xtb["argv"][0] == str(Path(config["g2"]["xtb"]["executable"]).resolve())
    assert xtb["argv"][1:4] == ["start.xyz", "--path", "end.xyz"]
    assert xtb["omp_num_threads"] == "4"
    assert xtb["seed_supported"] is False
    assert xtb["seed"] is None
    assert (Path(config["paths"]["manifests"]) / "g2_coverage.json").is_file()


def test_reverse_retry_recovers_and_normalizes_energy(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch, scenario="good")
    prepare_ids([REACTION_A], config=config)
    run_ids([REACTION_A], config=config)
    forward_rows, forward_profile = _frame_energies(_reaction_dir(config, REACTION_A) / "frames.parquet")
    monkeypatch.setenv("FAKE_XTB_SCENARIO", "short")
    report = run_ids([REACTION_A], config=config, force=True)
    assert report.n_attempted == 1
    rxn_dir = _reaction_dir(config, REACTION_A)
    document = read_json(rxn_dir / "reaction_path.json")
    assert document["status"] == "valid"
    assert document["direction"] == "reverse"
    assert document["direction_recovered"] is True
    assert len(document["attempts"]) == 2
    assert document["attempts"][0]["failure_code"] == RejectionCode.G2_PATH_DISCONTINUOUS.value
    reverse_rows, reverse_profile = _frame_energies(rxn_dir / "frames.parquet")
    assert len(reverse_rows) == len(forward_rows)
    assert reverse_rows[0]["energy_rel_kcal"] == pytest.approx(0.0, abs=1e-6)
    assert reverse_rows[0]["energy_rel_kcal_raw"] == pytest.approx(0.0, abs=1e-6)
    assert reverse_rows[-1]["energy_rel_kcal_raw"] > 0.0
    for normalized, original in zip(reverse_profile, forward_profile):
        assert normalized == pytest.approx(original, abs=1e-6)


def test_both_directions_fail_records_reverse_code_and_both_attempts(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch, scenario="both_fail")
    prepare_ids([REACTION_A], config=config)
    report = run_ids([REACTION_A], config=config)
    assert report.n_attempted == 1
    rxn_dir = _reaction_dir(config, REACTION_A)
    document = read_json(rxn_dir / "reaction_path.json")
    assert document["status"] == "failed"
    assert document["failure_code"] == RejectionCode.G2_XTB_FAILED.value
    assert document["direction"] == "reverse"
    assert document["direction_recovered"] is False
    assert len(document["attempts"]) == 2
    assert document["attempts"][0]["failure_code"] == RejectionCode.G2_PATH_DISCONTINUOUS.value
    assert document["attempts"][1]["failure_code"] == RejectionCode.G2_XTB_FAILED.value
    entries = [e for e in _ledger_entries(Path(config["paths"]["manifests"]))
               if e["reaction_id"] == REACTION_A]
    assert [entry["code"] for entry in entries] == [RejectionCode.G2_XTB_FAILED.value]


def test_prepare_ineligible_id_ledgers_exactly_once(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch, scenario="good", eligible_ids=(REACTION_A,))
    manifests = Path(config["paths"]["manifests"])
    report = prepare_ids([REACTION_B], config=config)
    assert report.n_attempted == 0
    entries = [e for e in _ledger_entries(manifests) if e["reaction_id"] == REACTION_B]
    assert [entry["code"] for entry in entries] == [RejectionCode.G2_NOT_ELIGIBLE.value]
    assert all(entry["stage"] == "g2_path" for entry in entries)
    prepare_ids([REACTION_B], config=config)
    entries = [e for e in _ledger_entries(manifests) if e["reaction_id"] == REACTION_B]
    assert len(entries) == 1


def test_run_is_idempotent_without_force(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch, scenario="good")
    prepare_ids([REACTION_A], config=config)
    run_ids([REACTION_A], config=config)
    rxn_dir = _reaction_dir(config, REACTION_A)
    before = {
        name: sha256_file(rxn_dir / name)
        for name in ("reaction_path.json", "frames.parquet", "run/xtb_path.log")
    }
    calls = _counter_calls(tmp_path)
    fresh_xtb = read_json(Path(config["paths"]["manifests"]) / "g2_path_manifest.json")["xtb"]
    report = run_ids([REACTION_A], config=config)
    assert report.n_attempted == 0
    assert report.n_skipped == 1
    assert _counter_calls(tmp_path) == calls
    manifest = read_json(Path(config["paths"]["manifests"]) / "g2_path_manifest.json")
    assert manifest["run"] == {"n_selected": 1, "n_attempted": 0, "n_skipped": 1}
    # Provenance is stable across an idempotent re-run: the skipped reaction's
    # persisted terminal document still supplies the xtb block, byte-equal to
    # the fresh run's (modulo the volatile keys, which the xtb block excludes).
    assert manifest["xtb"] == fresh_xtb
    assert manifest["xtb"]["version"] == "* xtb version 6.7.1 (fake)"
    assert manifest["xtb"]["argv"][0] == str(Path(config["g2"]["xtb"]["executable"]).resolve())
    assert manifest["xtb"]["omp_num_threads"] == "4"
    assert manifest["xtb"]["seed_supported"] is False
    assert manifest["xtb"]["seed"] is None
    assert manifest["xtb"]["sha256"] == sha256_file(Path(config["g2"]["xtb"]["executable"]))
    after = {
        name: sha256_file(rxn_dir / name)
        for name in ("reaction_path.json", "frames.parquet", "run/xtb_path.log")
    }
    assert after == before


def test_force_reruns_and_invokes_xtb_again(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch, scenario="good")
    prepare_ids([REACTION_A], config=config)
    run_ids([REACTION_A], config=config)
    assert _counter_calls(tmp_path) == 1
    report = run_ids([REACTION_A], config=config, force=True)
    assert report.n_attempted == 1
    assert report.n_skipped == 0
    assert _counter_calls(tmp_path) == 2
    document = read_json(_reaction_dir(config, REACTION_A) / "reaction_path.json")
    assert document["status"] == "valid"


def test_force_after_failure_keeps_ledger_history_and_updates_batch(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch, scenario="missing")
    manifests = Path(config["paths"]["manifests"])
    prepare_ids([REACTION_A], config=config)
    run_ids([REACTION_A], config=config)
    monkeypatch.setenv("FAKE_XTB_SCENARIO", "good")
    report = run_ids([REACTION_A], config=config, force=True)
    assert report.n_attempted == 1
    document = read_json(_reaction_dir(config, REACTION_A) / "reaction_path.json")
    assert document["status"] == "valid"
    entries = [e for e in _ledger_entries(manifests) if e["reaction_id"] == REACTION_A]
    assert [entry["code"] for entry in entries] == [RejectionCode.G2_XTB_FAILED.value]
    coverage = read_json(manifests / "g2_coverage.json")
    assert coverage["historical_failed_by_code"] == {RejectionCode.G2_XTB_FAILED.value: 1}
    assert coverage["current_by_code"] == {}
    manifest = read_json(manifests / "g2_path_manifest.json")
    assert manifest["n_valid"] == 1 and manifest["n_failed"] == 0
    assert manifest["by_code"] == {}


def test_missing_xtbpath_fails_typed_and_ledgers_once(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch, scenario="missing")
    manifests = Path(config["paths"]["manifests"])
    prepare_ids([REACTION_A], config=config)
    run_ids([REACTION_A], config=config)
    document = read_json(_reaction_dir(config, REACTION_A) / "reaction_path.json")
    assert document["status"] == "failed"
    assert document["failure_code"] == RejectionCode.G2_XTB_FAILED.value
    entries = [e for e in _ledger_entries(manifests) if e["reaction_id"] == REACTION_A]
    assert len(entries) == 1
    assert entries[0]["code"] == RejectionCode.G2_XTB_FAILED.value
    calls = _counter_calls(tmp_path)
    report = run_ids([REACTION_A], config=config)
    assert report.n_attempted == 0
    assert _counter_calls(tmp_path) == calls
    entries = [e for e in _ledger_entries(manifests) if e["reaction_id"] == REACTION_A]
    assert len(entries) == 1


def test_xtb_failure_does_not_trigger_reverse(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch, scenario="missing")
    prepare_ids([REACTION_A], config=config)
    run_ids([REACTION_A], config=config)
    assert not (_reaction_dir(config, REACTION_A) / "run_reverse").exists()


def test_log_npath_mismatch_fails_typed_without_reverse(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch, scenario="npath_mismatch")
    manifests = Path(config["paths"]["manifests"])
    prepare_ids([REACTION_A], config=config)
    run_ids([REACTION_A], config=config)
    document = read_json(_reaction_dir(config, REACTION_A) / "reaction_path.json")
    assert document["status"] == "failed"
    assert document["failure_code"] == RejectionCode.G2_XTB_FAILED.value
    assert "npath" in document["failure_detail"]
    entries = [e for e in _ledger_entries(manifests) if e["reaction_id"] == REACTION_A]
    assert [entry["code"] for entry in entries] == [RejectionCode.G2_XTB_FAILED.value]
    assert not (_reaction_dir(config, REACTION_A) / "run_reverse").exists()


# --------------------------------------------------------------------------
# Selection and prepare unit coverage
# --------------------------------------------------------------------------


def test_select_reaction_ids_filters_cohort_limit_and_reactions(tmp_path, monkeypatch):
    ids = ("RXN_0000000001", "RXN_0000000002", "RXN_0000000003", "RXN_0000000004")
    config = _config(
        tmp_path, monkeypatch, scenario="good",
        eligible_ids=ids, cohort_ids=ids[1:3], inventory_ids=ids, g1_ids=(),
    )
    assert select_reaction_ids(config, cohort="trial") == sorted(ids[1:3])
    assert select_reaction_ids(config, cohort="trial", limit=1) == [ids[1]]
    assert select_reaction_ids(config, cohort="all", reactions=(ids[3],)) == [ids[3]]
    assert select_reaction_ids(config, cohort="all", reactions=("RXN_0000000009",)) == []
    assert select_reaction_ids(config, cohort="all") == sorted(ids)


def test_select_reaction_ids_missing_inputs_raise(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch, scenario="good", eligible_ids=())
    (Path(config["paths"]["interim"]) / "g2_eligible.json").unlink()
    with pytest.raises(InfrastructureError, match="g1 gate"):
        select_reaction_ids(config, cohort="trial")
    _write_inputs(Path(config["paths"]["interim"]), eligible_ids=(), cohort_ids=(),
                  inventory_ids=(), g1_ids=())
    (Path(config["paths"]["interim"]) / "cohort_trial.json").unlink()
    with pytest.raises(InfrastructureError, match="g0 cohorts"):
        select_reaction_ids(config, cohort="trial")
    (Path(config["paths"]["interim"]) / "inventory.parquet").unlink()
    with pytest.raises(InfrastructureError, match="g0 inventory"):
        select_reaction_ids(config, cohort="all")


def test_prepare_missing_or_invalid_g1_doc_ledgers(tmp_path, monkeypatch):
    ids = (REACTION_A, REACTION_B)
    config = _config(tmp_path, monkeypatch, scenario="good", eligible_ids=ids, g1_ids=(REACTION_B,))
    invalid = reaction_change_path(Path(config["paths"]["interim"]), REACTION_B, SHARD_SIZE)
    document = read_json(invalid)
    document["validation"]["status"] = "rejected"
    document["validation"]["failure_code"] = "G1_MAP_ERROR"
    from pes2ts_core.utils.jsonio import write_json
    write_json(invalid, document)
    report = prepare_ids(ids, config=config)
    assert report.n_attempted == 0
    entries = {entry["reaction_id"]: entry["code"] for entry in _ledger_entries(Path(config["paths"]["manifests"]))}
    assert entries[REACTION_A] == RejectionCode.G2_MISSING_G1_DOC.value
    assert entries[REACTION_B] == RejectionCode.G2_G1_NOT_VALID.value


def test_prepare_endpoint_mismatch_ledgers(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch, scenario="good")
    from pes2ts_core.utils.jsonio import write_json
    doc_path = reaction_change_path(Path(config["paths"]["interim"]), REACTION_A, SHARD_SIZE)
    document = read_json(doc_path)
    document["products"][0]["rows"] = [
        row for row in document["products"][0]["rows"] if row["map"] != 3
    ]
    write_json(doc_path, document)
    report = prepare_ids([REACTION_A], config=config)
    assert report.n_attempted == 0
    entries = _ledger_entries(Path(config["paths"]["manifests"]))
    assert [entry["code"] for entry in entries] == [RejectionCode.G2_ENDPOINT_MISMATCH.value]
    assert not (_reaction_dir(config, REACTION_A) / "endpoints.json").exists()


def test_run_unprepared_id_raises_infrastructure_error_with_hint(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch, scenario="good")
    with pytest.raises(InfrastructureError) as excinfo:
        run_ids([REACTION_A], config=config)
    assert f"run `g2 prepare` first for {REACTION_A}" in str(excinfo.value)
