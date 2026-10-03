"""TDD tests for the G2 run-orchestration pipeline (plan task 9, ACP seam).

Covers :func:`select_reaction_ids`, :func:`prepare_ids`, and :func:`run_ids`
against a self-contained fake ACP ``XtbPathSearch`` helper (scenario-selected
via an env var; ``pes2ts_core.generation.execution.xtb_path.pipeline.
run_xtb_path_acp_attempt`` is monkeypatched, so no ACP checkout or xTB binary
is ever launched): the nine frozen scenarios (valid forward,
forward-fail/reverse-success with energy renormalization, both-fail,
ineligible ledger, idempotent resume, ``--force`` rerun, failure+force
history, missing raw trajectory, and the no-retry-on-``G2_XTB_FAILED`` rule),
the unusable-ACP-output scenario, the ACP-wiring infrastructure error, and
the selection/prepare typed failures.  All fixtures are synthetic and written
under ``tmp_path`` roots.
"""

# allow: SIZE_OK -- the plan names exactly one test file for the pipeline
# task; the nine frozen fake-ACP scenarios plus selection/prepare unit
# coverage and their fixture builders.

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, Final

import pytest
from pes2ts_core.config_loader import load_config
from pes2ts_core.g0.rejections import LEDGER_FILENAME, RejectionCode
from pes2ts_core.g1.build import reaction_change_path, shard_name
from pes2ts_core.generation.execution.xtb_path import SCHEMA_PATH
from pes2ts_core.generation.execution.xtb_path import pipeline as pipeline_module
from pes2ts_core.generation.execution.xtb_path.acp_backend import XtbPathAttemptOutcome
from pes2ts_core.generation.execution.xtb_path.pipeline import (
    InfrastructureError,
    prepare_ids,
    run_ids,
    select_reaction_ids,
)
from pes2ts_core.generation.execution.xtb_path.xtb_output import Frame
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
FRAME_COUNT: Final[int] = 8
_G2_NULL_ATTEMPT_KEYS: Final[tuple[str, ...]] = (
    "executable_sha256", "argv", "xtb_version_line", "seed_supported",
    "seed", "omp_num_threads",
)


def _read_xyz_text(text: str) -> list[tuple[str, float, float, float]]:
    lines = text.splitlines()
    count = int(lines[0].split()[0])
    atoms: list[tuple[str, float, float, float]] = []
    for line in lines[2:2 + count]:
        parts = line.split()
        atoms.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    return atoms


def _spread(atoms: list[tuple[str, float, float, float]]) -> float:
    return max(math.dist(a[1:], b[1:]) for a, b in itertools.combinations(atoms, 2))


def _interpolated(
    start: list[tuple[str, float, float, float]],
    end: list[tuple[str, float, float, float]],
    count: int,
) -> list[list[tuple[str, float, float, float]]]:
    frames = []
    for index in range(count):
        t = index / (count - 1) if count > 1 else 0.0
        atoms = []
        for (element, sx, sy, sz), (_, ex, ey, ez) in zip(start, end):
            atoms.append((element, sx + t * (ex - sx), sy + t * (ey - sy), sz + t * (ez - sz)))
        frames.append(atoms)
    return frames


def _write_acp_artifacts(
    run_dir: Path, *, request: dict[str, Any], trajectory: str,
) -> Path:
    work = run_dir / "WORK" / "pes2ts"
    work.mkdir(parents=True, exist_ok=True)
    (work / "path_config.json").write_text(
        json.dumps(request, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    result = run_dir / "RESULT" / "pes_search"
    result.mkdir(parents=True, exist_ok=True)
    trajectory_path = result / "xtbpath.xyz"
    trajectory_path.write_text(trajectory, encoding="utf-8")
    profile = {"schema_version": "pes_profile_v2", "workflow": "XtbPathSearch",
               "status": "completed"}
    (result / "pes_profile.json").write_text(
        json.dumps(profile, sort_keys=True), encoding="utf-8"
    )
    manifest = {"version": 2, "workflow": "XtbPathSearch", "status": "completed"}
    (run_dir / "RESULT" / "result_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return trajectory_path


def _fake_acp_attempt(
    *,
    reaction_dir: Path,
    config: dict[str, Any] | Any,
    direction: str,
    execution_id: str,
    attempt_id: str,
    timeout_seconds: float,
    charge: int,
    uhf: int,
    start_xyz_text: str,
    end_xyz_text: str,
    acp_root: Any = None,
    python_executable: Any = None,
    acp_config_path: Any = None,
    register: bool | None = None,
) -> XtbPathAttemptOutcome:
    """Fake ``run_xtb_path_acp_attempt``; scenario via ``$FAKE_ACP_SCENARIO``."""
    scenario = os.environ.get("FAKE_ACP_SCENARIO", "good")
    counter = os.environ.get("FAKE_ACP_COUNTER")
    if counter:
        with open(counter, "a", encoding="utf-8") as handle:
            handle.write(direction + "\n")
    run_dir = Path(reaction_dir) / ("run" if direction == "forward" else "run_reverse")
    request: dict[str, Any] = {
        "schema_version": "pes2ts_xtb_path_request_v1",
        "reaction_id": Path(reaction_dir).name,
        "source": {
            "source_type": "xyz_text_pair",
            "start_xyz": start_xyz_text,
            "end_xyz": end_xyz_text,
            "charge": charge,
            "multiplicity": uhf + 1,
        },
    }
    request_sha = hashlib.sha256(
        (direction + start_xyz_text + end_xyz_text).encode("utf-8")
    ).hexdigest()
    base_attempt: dict[str, Any] = {
        "direction": direction,
        "returncode": 0,
        "timed_out": False,
        "request_sha256": request_sha,
        "manifest_sha256": None,
        "acp_execution_id": execution_id,
        "acp_attempt_id": attempt_id,
        "wall_seconds": 0.05,
        "acp_status": "completed",
        "acp_reused": False,
        "acp_attempt_dir": str(run_dir),
        "acp_log_ref": "WORK/pes2ts/acp_cli.log",
        "acp_error": None,
        "native_frame_energy_unit": "relative_kcal_per_mol",
        "raw_trajectory_path": None,
        "raw_trajectory_sha256": None,
    }
    for key in _G2_NULL_ATTEMPT_KEYS:
        base_attempt[key] = None

    def _failure(
        detail: str, *, returncode: int | None = None, acp_status: str = "failed",
    ) -> XtbPathAttemptOutcome:
        attempt = dict(base_attempt)
        attempt.update(
            returncode=returncode,
            acp_status=acp_status,
            acp_error=detail,
            failure_code=RejectionCode.G2_XTB_FAILED.value,
        )
        return XtbPathAttemptOutcome(frames=None, failure_detail=detail, attempt=attempt)

    if scenario == "bad" or (scenario == "both_fail" and direction == "reverse"):
        return _failure(
            "ACP XtbPathSearch attempt status is 'failed' (fake)", returncode=3
        )
    if scenario == "missing":
        return _failure(
            "ACP completed attempt did not record a raw trajectory path",
            returncode=0, acp_status="completed",
        )
    if scenario == "npath_mismatch":
        return _failure(
            "unusable ACP xTB path output: npath mismatch (fake)",
            returncode=0, acp_status="completed",
        )
    start, end = _read_xyz_text(start_xyz_text), _read_xyz_text(end_xyz_text)
    short_forward = scenario in ("short", "both_fail") and direction == "forward"
    count = 4 if short_forward else FRAME_COUNT
    atoms_frames = _interpolated(start, end, count)
    base = min(_spread(start), _spread(end))
    frames: list[Frame] = []
    trajectory_lines: list[str] = []
    for atoms in atoms_frames:
        energy = 25.0 * (_spread(atoms) - base)
        frames.append(Frame(
            elements=tuple(atom[0] for atom in atoms),
            coordinates=tuple(tuple(atom[1:]) for atom in atoms),
            energy=energy,
        ))
        trajectory_lines.append(
            f"{len(atoms)}\n energy: {energy:.6f} xtb: 6.7.1 (fake-acp)\n"
        )
        for element, x, y, z in atoms:
            trajectory_lines.append(f"{element} {x:.6f} {y:.6f} {z:.6f}\n")
    trajectory = "".join(trajectory_lines)
    trajectory_path = _write_acp_artifacts(run_dir, request=request, trajectory=trajectory)
    trajectory_sha = hashlib.sha256(trajectory.encode("utf-8")).hexdigest()
    attempt = dict(base_attempt)
    attempt.update(
        manifest_sha256=trajectory_sha,
        raw_trajectory_path=str(trajectory_path),
        raw_trajectory_sha256=trajectory_sha,
    )
    return XtbPathAttemptOutcome(frames=tuple(frames), failure_detail=None, attempt=attempt)


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
    config["acp"] = {
        "root": str(tmp_path / "fake_acp_root"),
        "python": None,
        "config_path": None,
    }
    config["g2"]["xtb"]["timeout_seconds"] = 60
    monkeypatch.setenv("FAKE_ACP_SCENARIO", scenario)
    monkeypatch.setenv("FAKE_ACP_COUNTER", str(tmp_path / "acp_calls.log"))
    monkeypatch.setattr(pipeline_module, "run_xtb_path_acp_attempt", _fake_acp_attempt)
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
    path = tmp_path / "acp_calls.log"
    return len(path.read_text(encoding="utf-8").splitlines()) if path.is_file() else 0


# --------------------------------------------------------------------------
# Integration scenarios 1-9 (fake ACP)
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
    assert attempt["direction"] == "forward"
    assert attempt["acp_status"] == "completed"
    assert attempt["returncode"] == 0
    assert attempt["timed_out"] is False
    assert attempt["request_sha256"]
    assert attempt["manifest_sha256"]
    assert attempt["acp_execution_id"].startswith("pes2ts-g2-")
    assert attempt["acp_attempt_id"].startswith("attempt-")
    for key in _G2_NULL_ATTEMPT_KEYS:
        assert key in attempt, key
        assert attempt[key] is None, key
    rows, energies = _frame_energies(rxn_dir / "frames.parquet")
    assert len(rows) >= FRAME_COUNT
    assert all(row["energy_rel_kcal_raw"] == pytest.approx(row["energy_rel_kcal"]) for row in rows)
    for name in ("endpoints.json", "R.xyz", "P.xyz",
                 "run/WORK/pes2ts/path_config.json",
                 "run/RESULT/result_manifest.json",
                 "run/RESULT/pes_search/pes_profile.json",
                 "run/RESULT/pes_search/xtbpath.xyz"):
        assert (rxn_dir / name).is_file(), name
    request_on_disk = json.loads(
        (rxn_dir / "run/WORK/pes2ts/path_config.json").read_text(encoding="utf-8")
    )
    assert request_on_disk["source"]["start_xyz"] == (rxn_dir / "R.xyz").read_text(encoding="utf-8")
    assert request_on_disk["source"]["end_xyz"] == (rxn_dir / "P.xyz").read_text(encoding="utf-8")
    manifest = read_json(Path(config["paths"]["manifests"]) / "g2_path_manifest.json")
    assert manifest["run"] == {"n_selected": 1, "n_attempted": 1, "n_skipped": 0}
    xtb = manifest["xtb"]
    assert xtb["sha256"] is None
    assert xtb["request_sha256"] == attempt["request_sha256"]
    assert xtb["manifest_sha256"] == attempt["manifest_sha256"]
    assert xtb["version"] is None
    assert xtb["argv"] is None
    assert xtb["omp_num_threads"] is None
    assert xtb["seed_supported"] is None
    assert xtb["seed"] is None
    assert "$path" in xtb["path_inp"] and "npoint=50" in xtb["path_inp"]
    assert (Path(config["paths"]["manifests"]) / "g2_coverage.json").is_file()


def test_reverse_retry_recovers_and_normalizes_energy(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch, scenario="good")
    prepare_ids([REACTION_A], config=config)
    run_ids([REACTION_A], config=config)
    forward_rows, forward_profile = _frame_energies(_reaction_dir(config, REACTION_A) / "frames.parquet")
    monkeypatch.setenv("FAKE_ACP_SCENARIO", "short")
    report = run_ids([REACTION_A], config=config, force=True)
    assert report.n_attempted == 1
    rxn_dir = _reaction_dir(config, REACTION_A)
    document = read_json(rxn_dir / "reaction_path.json")
    assert document["status"] == "valid"
    assert document["direction"] == "reverse"
    assert document["direction_recovered"] is True
    assert len(document["attempts"]) == 2
    assert document["attempts"][0]["direction"] == "forward"
    assert document["attempts"][0]["failure_code"] == RejectionCode.G2_PATH_DISCONTINUOUS.value
    assert document["attempts"][1]["direction"] == "reverse"
    assert (rxn_dir / "run_reverse/RESULT/pes_search/xtbpath.xyz").is_file()
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
    stable_names = ("reaction_path.json", "frames.parquet",
                    "run/RESULT/pes_search/xtbpath.xyz")
    before = {name: sha256_file(rxn_dir / name) for name in stable_names}
    calls = _counter_calls(tmp_path)
    fresh_xtb = read_json(Path(config["paths"]["manifests"]) / "g2_path_manifest.json")["xtb"]
    report = run_ids([REACTION_A], config=config)
    assert report.n_attempted == 0
    assert report.n_skipped == 1
    assert _counter_calls(tmp_path) == calls
    manifest = read_json(Path(config["paths"]["manifests"]) / "g2_path_manifest.json")
    assert manifest["run"] == {"n_selected": 1, "n_attempted": 0, "n_skipped": 1}
    # Provenance is stable across an idempotent re-run: the skipped reaction's
    # persisted terminal document still supplies the ACP block, byte-equal to
    # the fresh run's (modulo the volatile keys, which the xtb block excludes).
    assert manifest["xtb"] == fresh_xtb
    assert manifest["xtb"]["version"] is None
    assert manifest["xtb"]["sha256"] is None
    assert manifest["xtb"]["request_sha256"] == fresh_xtb["request_sha256"]
    assert manifest["xtb"]["manifest_sha256"] == fresh_xtb["manifest_sha256"]
    assert "$path" in manifest["xtb"]["path_inp"]
    after = {name: sha256_file(rxn_dir / name) for name in stable_names}
    assert after == before


def test_force_reruns_and_invokes_acp_again(tmp_path, monkeypatch):
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
    monkeypatch.setenv("FAKE_ACP_SCENARIO", "good")
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


def test_missing_raw_trajectory_fails_typed_and_ledgers_once(tmp_path, monkeypatch):
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


def test_acp_failure_does_not_trigger_reverse(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch, scenario="missing")
    prepare_ids([REACTION_A], config=config)
    run_ids([REACTION_A], config=config)
    assert not (_reaction_dir(config, REACTION_A) / "run_reverse").exists()


def test_unusable_acp_output_fails_typed_without_reverse(tmp_path, monkeypatch):
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


def test_missing_acp_root_raises_infrastructure_error(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch, scenario="good")
    config["acp"]["root"] = None
    prepare_ids([REACTION_A], config=config)
    with pytest.raises(InfrastructureError, match="acp.root"):
        run_ids([REACTION_A], config=config)
    assert _counter_calls(tmp_path) == 0


def test_acp_wiring_threads_register_setting():
    wiring = pipeline_module._acp_wiring({"acp": {"root": "/acp", "register": False}})
    assert wiring["register"] is False
    wiring = pipeline_module._acp_wiring({"acp": {"root": "/acp", "register": True}})
    assert wiring["register"] is True
    wiring = pipeline_module._acp_wiring({"acp": {"root": "/acp"}})
    assert wiring["register"] is True


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
