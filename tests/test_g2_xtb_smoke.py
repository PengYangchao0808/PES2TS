"""Real-binary GFN2-xTB PATH smoke test (plan task 17, ``xtb`` marker).

Runs :func:`pes2ts_core.g2.runner.run_xtb_path` on the REAL GFN2-xTB binary
against the small C7OH8 endpoint fixture (15 atoms, single component, neutral
closed shell) with the default ``g2.path`` parameters from
``config/defaults.yaml`` (alp=0.5, npoint=50). The run takes roughly 10-30 s.

Validity judgment: the bond-pair sets (``r_pairs``/``p_pairs``) are derived
GEOMETRICALLY from the fixture endpoint coordinates with the distance
criterion "Cordero covalent-radius sum + 0.45 A" — the same tolerance the
pipeline uses — because the fixture carries no G1 change document. The
``events`` mapping is passed EMPTY, which disables the ``G2_TOPOLOGY_DRIFT``
predicate (no invented event lists); every other validity predicate (endpoints
reached, energies finite, frame count, continuity, collision) stays active.
The raw xTB run is metadynamics and NOT byte-reproducible, so nothing here
asserts byte equality with the pinned fixture outputs.
"""

import math
import os
import shutil
from collections.abc import Collection, Sequence
from pathlib import Path

import pytest
import yaml

from pes2ts_core.g0.rp_checks import ELEMENT_SYMBOLS
from pes2ts_core.g1.index_map import COVALENT_RADII
from pes2ts_core.g2.inspect import evaluate_validity
from pes2ts_core.g2.runner import run_xtb_path
from pes2ts_core.g2.xtb_output import (
    Frame,
    parse_path_log,
    parse_path_xyz,
    parse_start_xyz,
    parse_ts_xyz,
)

ENV_EXECUTABLE = "PES2TS_XTB_EXECUTABLE"
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "g2" / "real_path"
BOND_TOLERANCE = 0.45
MIN_FRAMES = 8


def _resolve_xtb_executable() -> str:
    """Env var first, then PATH; a miss is a FAILURE, never a skip."""
    candidate = os.environ.get(ENV_EXECUTABLE) or shutil.which("xtb")
    if candidate is None:
        pytest.fail(
            f"No GFN2-xTB binary found: set {ENV_EXECUTABLE} to the xtb "
            "executable or put 'xtb' on PATH (XtbNotFoundError condition); "
            "a skip here would be a false pass"
        )
    return candidate


def _default_config(executable: str) -> dict[str, object]:
    with open(Path(__file__).parent.parent / "config" / "defaults.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    assert isinstance(config, dict) and isinstance(config.get("g2"), dict)
    config["g2"]["xtb"]["executable"] = executable
    return config


def _covalent_radius(element: str) -> float:
    return COVALENT_RADII[ELEMENT_SYMBOLS.index(element) + 1]


def _bonded_pairs(frame: Frame) -> Collection[Sequence[int]]:
    """Pairs whose distance is within the Cordero radius sum + tolerance."""
    pairs: list[Sequence[int]] = []
    for first in range(len(frame.elements)):
        for second in range(first + 1, len(frame.elements)):
            radius_sum = _covalent_radius(frame.elements[first]) + _covalent_radius(
                frame.elements[second]
            )
            distance = math.dist(frame.coordinates[first], frame.coordinates[second])
            if distance <= radius_sum + BOND_TOLERANCE:
                pairs.append((first, second))
    return pairs


@pytest.mark.xtb
def test_real_xtb_path_smoke(tmp_path: Path) -> None:
    # Given: the real binary and the C7OH8 endpoint fixture in a fresh run dir
    executable = _resolve_xtb_executable()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for name in ("start.xyz", "end.xyz"):
        shutil.copy(FIXTURE_DIR / name, run_dir / name)
    config = _default_config(executable)

    # When: the real GFN2-xTB PATH metadynamics runs to completion
    result = run_xtb_path(run_dir, config, charge=0, uhf=0)

    # Then: normal exit, parseable outputs, and a valid path verdict
    assert result.returncode == 0, (result.returncode, result.timed_out)
    assert result.timed_out is False

    path_frames = parse_path_xyz(run_dir / "xtbpath.xyz", start_path=run_dir / "start.xyz")
    ts_frame = parse_ts_xyz(run_dir / "xtbpath_ts.xyz")
    log = parse_path_log(run_dir / "xtb_path.log")
    assert ts_frame.energy is not None and math.isfinite(ts_frame.energy)
    assert len(path_frames) >= MIN_FRAMES
    energies = [frame.energy for frame in path_frames]
    assert all(energy is not None and math.isfinite(energy) for energy in energies)

    start_frame = parse_start_xyz(run_dir / "start.xyz")
    end_frame = parse_start_xyz(run_dir / "end.xyz")
    elements = dict(enumerate(start_frame.elements))
    reactant_coords = dict(enumerate(start_frame.coordinates))
    product_coords = dict(enumerate(end_frame.coordinates))
    verdict = evaluate_validity(
        path_frames,
        reaction_id="smoke_c7oh8",
        reactant_coords=reactant_coords,
        product_coords=product_coords,
        elements=elements,
        r_pairs=_bonded_pairs(start_frame),
        p_pairs=_bonded_pairs(end_frame),
        events={},  # no G1 doc for the fixture: topology-drift predicate off
        config=config,
    )
    assert verdict.status == "valid", (verdict.failure_code, verdict.detail)
    assert log.reaction_energy_kcal is not None
