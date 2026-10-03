"""TDD tests for the G2 CLI contract (plan task 11, ACP seam).

Covers the ``g2`` top-level command wired in :mod:`pes2ts_core.cli`:
``--help`` exits 0 for every level; ``g2 prepare``/``g2 run``/``g2 verify``
end-to-end against a self-contained fake ACP ``XtbPathSearch`` helper
(``pipeline.run_xtb_path_acp_attempt`` monkeypatched; no ACP checkout or xTB
binary is ever launched); batch completion is exit 0 even when the batch
contains failed reactions; tampered trees, missing inputs, missing ACP
wiring, and unprepared reactions all map to exit 24 (``EXIT_G2_FAILED``)
with the pipeline's actionable hint preserved; the default cohort is
``trial`` and ``--reaction``/``--force`` are forwarded.  All fixtures are
synthetic and written under ``tmp_path`` roots.
"""

# allow: SIZE_OK -- the plan names exactly one test file for the CLI task;
# the fake-ACP harness is deliberately self-contained (same pattern as
# tests/test_g2_pipeline.py) plus the seven frozen CLI contract scenarios.

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Final

import pytest
import yaml
from pes2ts_core.cli import main
from pes2ts_core.config_loader import load_config
from pes2ts_core.g0.rejections import RejectionCode
from pes2ts_core.g1.build import reaction_change_path, shard_name
from pes2ts_core.generation.execution.xtb_path import pipeline as pipeline_module
from pes2ts_core.generation.execution.xtb_path.acp_backend import XtbPathAttemptOutcome
from pes2ts_core.generation.execution.xtb_path.xtb_output import Frame
from pes2ts_core.utils.jsonio import read_json, write_json
from pes2ts_core.utils.parquet_io import write_parquet

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
FRAME_COUNT: Final[int] = 8
_G2_NULL_ATTEMPT_KEYS: Final[tuple[str, ...]] = (
    "executable_sha256", "argv", "xtb_version_line", "seed_supported",
    "seed", "omp_num_threads",
)
_UNSET: Any = object()


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


def _fake_acp_attempt(
    *,
    reaction_dir: Path,
    config: Any,
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

    if scenario == "missing":
        return _failure(
            "ACP completed attempt did not record a raw trajectory path",
            returncode=0, acp_status="completed",
        )
    start, end = _read_xyz_text(start_xyz_text), _read_xyz_text(end_xyz_text)
    atoms_frames = _interpolated(start, end, FRAME_COUNT)
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


def _component(
    tag: str, numbers: tuple[int, ...], coords: tuple[tuple[float, float, float], ...],
) -> dict[str, Any]:
    return {
        "tag": tag, "smiles": "", "atomic_numbers": list(numbers),
        "coordinates": [list(coord) for coord in coords],
    }


def _inventory_components() -> list[dict[str, Any]]:
    return [
        _component("R0", ATOMIC_NUMBERS, COORDS),
        _component("P0", ATOMIC_NUMBERS[:5], COORDS[:5]),
        _component("P1", (1,), (COORDS[5],)),
    ]


def _g1_document(reaction_id: str) -> dict[str, Any]:
    return {
        "schema_version": "g1_change_v1", "reaction_id": reaction_id, "reaction_smiles": SMILES,
        "reactants": [{"tag": "R0", "rows": _rows(6, ELEMENTS)}],
        "products": [
            {"tag": "P0", "rows": _rows(5, ELEMENTS[:5])},
            {"tag": "P1", "rows": [{"local_index": 0, "map": 6, "element": "H", "global_index": 0}]},
        ],
        "reaction_center": {"core": [5, 6], "with_shell": [2, 5, 6]},
        "bond_changes": {
            "formed": [], "broken": [{"atoms": [5, 6]}],
            "order_changed": [], "hydrogen_migration": [],
        },
        "categories": {
            "pure_formed": False, "pure_broken": True, "both": False,
            "order_change_only": False, "has_order_change": False,
            "h_migration": False, "multi_component": True,
        },
        "validation": {"status": "valid", "failure_code": None, "failure_detail": None},
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
        "components": [_inventory_components() for _ in ids],
    })
    for reaction_id in g1_ids:
        write_json(reaction_change_path(interim, reaction_id, SHARD_SIZE), _g1_document(reaction_id))


def _yaml_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, scenario: str,
    eligible_ids: tuple[str, ...] = (REACTION_A,),
    cohort_ids: tuple[str, ...] | None = None,
    acp_root: Any = _UNSET,
) -> Path:
    """Build the synthetic G0/G1 inputs and dump the merged config to YAML."""
    config = load_config()
    interim = tmp_path / "interim"
    config["paths"]["interim"] = str(interim)
    config["paths"]["manifests"] = str(tmp_path / "manifests")
    config["acp"] = {
        "root": str(tmp_path / "fake_acp_root") if acp_root is _UNSET else acp_root,
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
        inventory_ids=eligible_ids,
        g1_ids=eligible_ids,
    )
    config_path = tmp_path / "fixture.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def _cli(config_path: Path, *argv: str) -> int:
    return main(["--config", str(config_path), *argv])


def _reaction_dir(tmp_path: Path, reaction_id: str) -> Path:
    interim = tmp_path / "interim"
    return interim / "g2" / "paths" / shard_name(reaction_id, SHARD_SIZE) / reaction_id


def _acp_calls(tmp_path: Path) -> int:
    path = tmp_path / "acp_calls.log"
    return len(path.read_text(encoding="utf-8").splitlines()) if path.is_file() else 0


# --------------------------------------------------------------------------
# Help contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("argv", [
    ("g2",),
    ("g2", "prepare"),
    ("g2", "run"),
    ("g2", "verify"),
])
def test_help_exits_zero(argv: tuple[str, ...]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([*argv, "--help"])
    assert excinfo.value.code == 0


# --------------------------------------------------------------------------
# End-to-end happy paths (fake ACP)
# --------------------------------------------------------------------------


def test_prepare_run_verify_end_to_end_exit_zero(tmp_path, monkeypatch) -> None:
    config_path = _yaml_config(
        tmp_path, monkeypatch, scenario="good",
        eligible_ids=(REACTION_A, REACTION_B),
    )
    assert _cli(config_path, "g2", "prepare") == 0
    assert _cli(config_path, "g2", "run") == 0
    assert _cli(config_path, "g2", "verify") == 0
    for reaction_id in (REACTION_A, REACTION_B):
        document = read_json(_reaction_dir(tmp_path, reaction_id) / "reaction_path.json")
        assert document["status"] == "valid"


def test_run_batch_containing_failed_reaction_still_exits_zero(tmp_path, monkeypatch) -> None:
    # The "missing" scenario makes the fake ACP produce no raw trajectory: a
    # typed, ledgered per-reaction failure — the batch still completes.  (The
    # frame-less failed directories are consistent with the verifier contract
    # since 5ee6187: frames.parquet is required only when the document
    # declares frames.n_frames > 0.)
    config_path = _yaml_config(
        tmp_path, monkeypatch, scenario="missing",
        eligible_ids=(REACTION_A, REACTION_B),
    )
    assert _cli(config_path, "g2", "prepare") == 0
    assert _cli(config_path, "g2", "run") == 0
    document = read_json(_reaction_dir(tmp_path, REACTION_A) / "reaction_path.json")
    assert document["status"] == "failed"


# --------------------------------------------------------------------------
# Exit-24 contract
# --------------------------------------------------------------------------


def test_verify_tampered_tree_exits_24_and_prints_problems(tmp_path, monkeypatch, capsys) -> None:
    config_path = _yaml_config(tmp_path, monkeypatch, scenario="good")
    assert _cli(config_path, "g2", "prepare") == 0
    assert _cli(config_path, "g2", "run") == 0
    (_reaction_dir(tmp_path, REACTION_A) / "frames.parquet").unlink()
    assert _cli(config_path, "g2", "verify") == 24
    out = capsys.readouterr().out
    assert REACTION_A in out
    assert "frames.parquet" in out


def test_missing_inventory_eligible_and_cohort_exit_24_with_hints(
    tmp_path, monkeypatch, capsys,
) -> None:
    config_path = _yaml_config(tmp_path, monkeypatch, scenario="good")
    interim = tmp_path / "interim"
    (interim / "inventory.parquet").unlink()
    assert _cli(config_path, "g2", "prepare", "--cohort", "all") == 24
    assert "run `g0 inventory` first" in capsys.readouterr().err
    _write_inputs(
        interim, eligible_ids=(REACTION_A,), cohort_ids=(REACTION_A,),
        inventory_ids=(REACTION_A,), g1_ids=(REACTION_A,),
    )
    (interim / "g2_eligible.json").unlink()
    assert _cli(config_path, "g2", "prepare") == 24
    assert "run `g1 gate` first" in capsys.readouterr().err
    _write_inputs(
        interim, eligible_ids=(REACTION_A,), cohort_ids=(REACTION_A,),
        inventory_ids=(REACTION_A,), g1_ids=(REACTION_A,),
    )
    (interim / "cohort_trial.json").unlink()
    assert _cli(config_path, "g2", "prepare") == 24
    assert "run `g0 cohorts` first" in capsys.readouterr().err


def test_missing_acp_root_run_exits_24_without_traceback(
    tmp_path, monkeypatch, capsys,
) -> None:
    config_path = _yaml_config(
        tmp_path, monkeypatch, scenario="good", acp_root=None,
    )
    assert _cli(config_path, "g2", "prepare") == 0
    assert _cli(config_path, "g2", "run") == 24
    err = capsys.readouterr().err
    assert "ERROR" in err
    assert "acp.root" in err
    assert "Traceback" not in err


def test_malformed_eligible_json_exits_24_without_traceback(
    tmp_path, monkeypatch, capsys,
) -> None:
    config_path = _yaml_config(tmp_path, monkeypatch, scenario="good")
    (tmp_path / "interim" / "g2_eligible.json").write_text("{broken json", encoding="utf-8")
    assert _cli(config_path, "g2", "prepare") == 24
    err = capsys.readouterr().err
    assert "g2_eligible.json" in err
    assert "Traceback" not in err


def test_run_without_prepare_exits_24_with_prepare_hint(tmp_path, monkeypatch, capsys) -> None:
    config_path = _yaml_config(tmp_path, monkeypatch, scenario="good")
    assert _cli(config_path, "g2", "run") == 24
    assert "run `g2 prepare` first" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Selection defaults and flag forwarding
# --------------------------------------------------------------------------


def test_default_cohort_is_trial_reaction_filters_and_force_reruns(
    tmp_path, monkeypatch, capsys,
) -> None:
    config_path = _yaml_config(
        tmp_path, monkeypatch, scenario="good",
        eligible_ids=(REACTION_A, REACTION_B), cohort_ids=(REACTION_A,),
    )
    # Default cohort is trial: only the trial member gets prepared.
    assert _cli(config_path, "g2", "prepare") == 0
    assert (_reaction_dir(tmp_path, REACTION_A) / "endpoints.json").is_file()
    assert not (_reaction_dir(tmp_path, REACTION_B) / "endpoints.json").exists()
    assert capsys.readouterr() is not None
    # --reaction selects one prepared reaction; the second run resumes (no new
    # ACP call) and --force invokes the fake ACP helper again.
    assert _cli(config_path, "g2", "run", "--reaction", REACTION_A) == 0
    assert _acp_calls(tmp_path) == 1
    assert _cli(config_path, "g2", "run") == 0
    assert _acp_calls(tmp_path) == 1
    assert _cli(config_path, "g2", "run", "--reaction", REACTION_A, "--force") == 0
    assert _acp_calls(tmp_path) == 2
