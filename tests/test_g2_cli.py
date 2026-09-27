"""TDD tests for the G2 CLI contract (plan task 11).

Covers the ``g2`` top-level command wired in :mod:`pes2ts_core.cli`:
``--help`` exits 0 for every level; ``g2 prepare``/``g2 run``/``g2 verify``
end-to-end against the self-contained fake xTB binary; batch completion is
exit 0 even when the batch contains failed reactions; tampered trees,
missing inputs, missing xTB executables, and unprepared reactions all map to
exit 24 (``EXIT_G2_FAILED``) with the pipeline's actionable hint preserved;
the default cohort is ``trial`` and ``--reaction``/``--force`` are forwarded.
All fixtures are synthetic and written under ``tmp_path`` roots.
"""

# allow: SIZE_OK -- the plan names exactly one test file for the CLI task;
# the fake-xTB harness is deliberately self-contained (same pattern as
# tests/test_g2_pipeline.py) plus the seven frozen CLI contract scenarios.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Final

import pytest
import yaml
from pes2ts_core.cli import main
from pes2ts_core.config_loader import load_config
from pes2ts_core.g1.build import reaction_change_path, shard_name
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
    if scenario == "bad":
        print("fake xtb: intentional failure")
        return 3
    with open("xtb_path.log", "w", encoding="utf-8") as handle:
        handle.write("* xtb version 6.7.1 (fake)\\n")
        handle.write("   forward barrier (kcal): 12.500000\\n")
        handle.write("   backward barrier (kcal): 12.500000\\n")
        handle.write("   reaction energy  (kcal): 25.000000\\n")
    if scenario == "missing":
        return 0
    start, end = read_xyz("start.xyz"), read_xyz("end.xyz")
    frames = interpolated(start, end, FRAME_COUNT)
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
    executable: Path | None = None,
) -> Path:
    """Build the synthetic G0/G1 inputs and dump the merged config to YAML."""
    config = load_config()
    interim = tmp_path / "interim"
    config["paths"]["interim"] = str(interim)
    config["paths"]["manifests"] = str(tmp_path / "manifests")
    script = tmp_path / "fake_xtb.py"
    script.write_text(FAKE_XTB, encoding="utf-8")
    script.chmod(0o755)
    config["g2"]["xtb"]["executable"] = str(executable if executable is not None else script)
    config["g2"]["xtb"]["timeout_seconds"] = 60
    monkeypatch.setenv("FAKE_XTB_SCENARIO", scenario)
    monkeypatch.setenv("FAKE_XTB_COUNTER", str(tmp_path / "xtb_calls.log"))
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


def _xtb_calls(tmp_path: Path) -> int:
    path = tmp_path / "xtb_calls.log"
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
# End-to-end happy paths (fake xTB)
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
    # The "missing" scenario makes the fake xTB produce no path: a typed,
    # ledgered per-reaction failure -- the batch still completes.  (The
    # frame-less failed directories are a verify problem by the task-10
    # verifier contract; only the run batch completion is asserted here.)
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


def test_missing_xtb_executable_run_exits_24_without_traceback(
    tmp_path, monkeypatch, capsys,
) -> None:
    config_path = _yaml_config(
        tmp_path, monkeypatch, scenario="good",
        executable=tmp_path / "absent" / "xtb",
    )
    assert _cli(config_path, "g2", "prepare") == 0
    assert _cli(config_path, "g2", "run") == 24
    err = capsys.readouterr().err
    assert "ERROR" in err
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
    # xTB call) and --force invokes the fake xTB again.
    assert _cli(config_path, "g2", "run", "--reaction", REACTION_A) == 0
    assert _xtb_calls(tmp_path) == 1
    assert _cli(config_path, "g2", "run") == 0
    assert _xtb_calls(tmp_path) == 1
    assert _cli(config_path, "g2", "run", "--reaction", REACTION_A, "--force") == 0
    assert _xtb_calls(tmp_path) == 2
