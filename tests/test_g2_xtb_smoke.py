"""Real-ACP GFN2-xTB PATH smoke test (``acp`` marker).

Runs :func:`pes2ts_core.generation.execution.xtb_path.acp_backend.run_xtb_path_acp_attempt`
against a REAL ACP checkout (env ``PES2TS_ACP_ROOT``; interpreter
``PES2TS_ACP_PYTHON``) executing the REAL ``XtbPathSearch`` workflow on the
small C7OH8 endpoint fixture (15 atoms, single component, neutral closed
shell) with the default ``g2.path`` parameters from ``config/defaults.yaml``
(alp=0.5, npoint=50).  The run takes minutes.  This replaces the retired
local-xTB smoke (ADR-0002 X2'-C): G2 executes xTB PATH through ACP only.

Never-skip semantics (as before): when neither the ACP checkout nor its
python interpreter resolves, the test FAILS naming the exact variables — a
skip here would be a false pass.

Validity judgment: the bond-pair sets (``r_pairs``/``p_pairs``) are derived
GEOMETRICALLY from the fixture endpoint coordinates with the distance
criterion "Cordero covalent-radius sum + 0.45 A" — the same tolerance the
pipeline uses — because the fixture carries no G1 change document.  The
``events`` mapping is passed EMPTY, which disables the ``G2_TOPOLOGY_DRIFT``
predicate (no invented event lists); every other validity predicate (endpoints
reached, energies finite, frame count, continuity, collision) stays active.
The raw xTB PATH run is metadynamics and NOT byte-reproducible, so nothing
here asserts byte equality with pinned fixture outputs.
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Collection, Sequence
from pathlib import Path

import pytest
import yaml

from pes2ts_core.g0.rp_checks import ELEMENT_SYMBOLS
from pes2ts_core.g1.index_map import COVALENT_RADII
from pes2ts_core.generation.execution.xtb_path.acp_backend import (
    run_xtb_path_acp_attempt,
)
from pes2ts_core.generation.execution.xtb_path.inspect import evaluate_validity
from pes2ts_core.generation.execution.xtb_path.xtb_output import (
    Frame,
    parse_start_xyz,
)

ENV_ACP_ROOT = "PES2TS_ACP_ROOT"
ENV_ACP_PYTHON = "PES2TS_ACP_PYTHON"
ENV_ACP_CONFIG = "PES2TS_ACP_CONFIG"
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "g2" / "real_path"
BOND_TOLERANCE = 0.45
MIN_FRAMES = 8


def _resolve_acp_wiring() -> tuple[Path, str, str | None]:
    """Env vars first; a miss is a FAILURE naming the variables, never a skip."""
    raw_root = os.environ.get(ENV_ACP_ROOT)
    raw_python = os.environ.get(ENV_ACP_PYTHON)
    missing = [
        name
        for name, value in ((ENV_ACP_ROOT, raw_root), (ENV_ACP_PYTHON, raw_python))
        if not value
    ]
    if missing:
        pytest.fail(
            "ACP smoke environment not ready: set "
            + " and ".join(missing)
            + f" ({ENV_ACP_ROOT} = ACP source checkout containing src/acp/cli.py; "
            f"{ENV_ACP_PYTHON} = python interpreter of that ACP environment, "
            f"which must provide the GFN2-xTB binary ACP launches); "
            "G2 xTB PATH executes only through ACP — a skip here would be a "
            "false pass"
        )
    root = Path(str(raw_root)).expanduser()
    if not root.is_dir():
        pytest.fail(f"{ENV_ACP_ROOT}={root} is not a directory; a skip here would be a false pass")
    cli_py = root / "src" / "acp" / "cli.py"
    if not cli_py.is_file():
        pytest.fail(
            f"{ENV_ACP_ROOT}={root} lacks src/acp/cli.py; a skip here would be a false pass"
        )
    python = str(Path(str(raw_python)).expanduser())
    config_path = os.environ.get(ENV_ACP_CONFIG)
    return root, python, config_path


def _default_config() -> dict[str, object]:
    with open(Path(__file__).parent.parent / "config" / "defaults.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    assert isinstance(config, dict) and isinstance(config.get("g2"), dict)
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


@pytest.mark.acp
def test_real_acp_xtb_path_smoke(tmp_path: Path) -> None:
    # Given: a real ACP checkout and the C7OH8 endpoint fixture
    acp_root, python, acp_config = _resolve_acp_wiring()
    reaction_dir = tmp_path / "RXN_smoke_c7oh8"
    reaction_dir.mkdir()
    start_text = (FIXTURE_DIR / "start.xyz").read_text(encoding="utf-8")
    end_text = (FIXTURE_DIR / "end.xyz").read_text(encoding="utf-8")
    config = _default_config()
    config["reaction_id"] = "smoke_c7oh8"
    # Chemistry-only smoke: registration needs a jobs store shared with a live
    # ACP server (ACP_RUN_ROOT) and is covered by the transport unit tests.
    config.setdefault("acp", {})["register"] = False
    g2_block = config["g2"]
    assert isinstance(g2_block, dict)
    xtb_block = g2_block["xtb"]
    assert isinstance(xtb_block, dict)
    timeout_seconds = xtb_block["timeout_seconds"]
    assert isinstance(timeout_seconds, (int, float))

    # When: the real ACP XtbPathSearch workflow runs the GFN2-xTB PATH step
    outcome = run_xtb_path_acp_attempt(
        reaction_dir=reaction_dir,
        config=config,
        direction="forward",
        execution_id="pes2ts-smoke-acp-001",
        attempt_id="attempt-smoke-001",
        timeout_seconds=float(timeout_seconds),
        charge=0,
        uhf=0,
        start_xyz_text=start_text,
        end_xyz_text=end_text,
        acp_root=acp_root,
        python_executable=python,
        acp_config_path=acp_config,
    )

    # Then: a completed attempt, parsed frames, and ACP provenance
    assert outcome.failure_detail is None, outcome.failure_detail
    assert outcome.frames is not None
    attempt = outcome.attempt
    assert attempt["acp_status"] == "completed"
    assert attempt["request_sha256"]
    assert attempt["manifest_sha256"]
    assert attempt["raw_trajectory_sha256"]

    # And: parseable energies and a valid path verdict
    path_frames = outcome.frames
    assert len(path_frames) >= MIN_FRAMES
    energies = [frame.energy for frame in path_frames]
    assert all(energy is not None and math.isfinite(energy) for energy in energies)

    start_frame = parse_start_xyz(FIXTURE_DIR / "start.xyz")
    end_frame = parse_start_xyz(FIXTURE_DIR / "end.xyz")
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
