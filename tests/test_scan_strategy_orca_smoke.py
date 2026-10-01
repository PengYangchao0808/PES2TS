"""Gated ORCA smoke test (plan todo 24, ``orca`` marker).

Never-skip semantics follow ``tests/test_g2_xtb_smoke.py:46-55``: with
``PES2TS_ORCA_EXECUTABLE`` set to a real ORCA binary, each representative
sub-family compiles through ``pes2ts_core.scan_strategy.compile_orca`` (todo
19), assembles a minimal runnable ORCA input from the compiled artifacts,
executes the real binary on a tiny molecule, and records the ORCA version
line plus run evidence as capability-registry probe receipts.

Sub-families (design §13.3, one representative case each):

- ``single_b``      — H2 stretch, SINGLE_1D native ``%geom Scan``
- ``double_b_simul``— water double O–H, COUPLED_1D ``Simul_Scan``
- ``ad_scan``       — H2O2-like 4-atom, COUPLED_1D angle+dihedral
- ``nonuniform_per_point`` — H2, SCHEDULED_1D custom λ list → per-point Constraints
- ``neb_minimal``   — H2, PathCandidateV1 minimal ``%geom Path`` (n_images=3)

Receipts are written under a configurable directory (env
``PES2TS_SMOKE_RECEIPTS_DIR``, default ``.omo/evidence/``) — never auto-merged
into the shipped registry and never auto-committed.

When ``PES2TS_ORCA_EXECUTABLE`` is missing the tests FAIL naming that variable
(never skips): a skip here would be a false pass.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from pes2ts_core.integration.acp.multicoord import ADAPTER_VERSION_MULTICOORD
from pes2ts_core.scan_strategy import capabilities as caps
from pes2ts_core.scan_strategy import compile_orca as co
from pes2ts_core.scan_strategy.capabilities import ProbeReceipt
from pes2ts_core.utils.hashing import stable_json_dumps

ENV_EXECUTABLE = "PES2TS_ORCA_EXECUTABLE"
ENV_RECEIPTS_DIR = "PES2TS_SMOKE_RECEIPTS_DIR"
ADAPTER_VERSION_ORCA_DIRECT = "orca-direct-smoke"
SMOKE_METHOD = "HF STO-3G"
SMOKE_NPROCS = 1
RUN_TIMEOUT_SECONDS = 180.0
VERSION_TIMEOUT_SECONDS = 30.0

ALL_MODES = ["SINGLE_1D", "COUPLED_1D", "SCHEDULED_1D", "PATH_NEB"]
ALL_KINDS = ["B", "A", "D"]


def _resolve_orca_executable() -> str:
    """Env var required; a miss is a FAILURE naming the variable, never a skip."""
    candidate = os.environ.get(ENV_EXECUTABLE)
    if not candidate:
        pytest.fail(
            f"No ORCA executable found: set {ENV_EXECUTABLE} to the real ORCA "
            "binary (ORCA install + version required for the gated orca smoke); "
            "a skip here would be a false pass"
        )
    path = Path(candidate).expanduser()
    if not path.is_file():
        pytest.fail(
            f"{ENV_EXECUTABLE}={path} is not a file; a skip here would be a false pass"
        )
    return str(path)


def _orca_version_line(executable: str) -> str:
    """Best-effort ORCA version banner; failure to read it is a loud fail."""
    collected: list[str] = []
    for args in (["--version"], ["-v"]):
        try:
            proc = subprocess.run(
                [executable, *args],
                capture_output=True,
                text=True,
                timeout=VERSION_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        text = ((proc.stdout or "") + (proc.stderr or "")).strip()
        if text:
            collected.append(text)
            for line in text.splitlines():
                lowered = line.lower()
                if "version" in lowered or "orca" in lowered:
                    return line.strip()
    if collected:
        first_line = collected[0].splitlines()[0].strip()
        if first_line:
            return first_line
    pytest.fail(
        f"Could not determine an ORCA version line from {executable} "
        f"({ENV_EXECUTABLE}); a skip here would be a false pass"
    )


def _receipts_dir() -> Path:
    raw = os.environ.get(ENV_RECEIPTS_DIR)
    if raw:
        return Path(raw).expanduser()
    return Path(__file__).resolve().parents[1] / ".omo" / "evidence"


def _session_capability(
    tmp_path: Path, *, engine_version: str, probe_id: str
) -> caps.EffectiveCapability:
    """Session-local probed stack carrying the REAL ORCA version string."""
    receipt = {
        "probe_id": probe_id,
        "date": date.today().isoformat(),
        "engine_version": engine_version,
        "adapter_version": ADAPTER_VERSION_ORCA_DIRECT,
        "status": "pass",
        "evidence_ref": f"smoke/{probe_id}",
        "modes": list(ALL_MODES),
    }

    def _entry(entry_id: str, layer: str) -> dict[str, Any]:
        return {
            "entry_id": entry_id,
            "layer": layer,
            "engine": "orca",
            "adapter": "acp",
            "capability": {
                "engine_version": engine_version,
                "adapter_version": ADAPTER_VERSION_ORCA_DIRECT,
                "supported_modes": list(ALL_MODES),
                "coordinate_kinds": list(ALL_KINDS),
                "max_scan_coordinates": 3,
                "point_limits": {"baseline": 3, "max": 21},
                "custom_schedule_support": True,
                "constraint_support": {
                    "native_scan": True,
                    "per_point_constraints": True,
                    "simul_scan": True,
                },
                "method_element_coverage": {SMOKE_METHOD: ["H", "O"]},
                "probe_receipts": [receipt],
            },
            "notes": "todo-24 ORCA smoke session stack (real captured version)",
        }

    payload = {
        "schema_version": caps.REGISTRY_SCHEMA_VERSION,
        "registry_id": "todo24-orca-smoke",
        "description": "session probed stack for gated ORCA smoke",
        "entries": [_entry("eng", "engine"), _entry("adp", "adapter"), _entry("dep", "deployment")],
    }
    path = tmp_path / "orca_capabilities_v1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    registry = caps.load_capability_registry(path)
    return caps.effective_capability("eng", "adp", "dep", registry=registry)


def _driver(kind: str, maps: list[int], *, schedule_values: list[float] | None = None) -> dict[str, Any]:
    return {
        "kind": kind,
        "maps": list(maps),
        "unit": "angstrom" if kind == "B" else "degree",
        "schedule_values": schedule_values,
        "index0": None,
    }


def _scan_candidate(
    mode: str,
    drivers: list[dict[str, Any]],
    lambda_values: list[float],
    *,
    candidate_id: str,
    schedule_kind: str | None = None,
) -> dict[str, Any]:
    candidate: dict[str, Any] = {
        "candidate_kind": "ScanCandidateV2",
        "candidate_id": candidate_id,
        "mode": mode,
        "start_endpoint": "R",
        "direction": "R_to_P",
        "assembly_id": "asm-orca-smoke",
        "anchor_reason": "SMOKE",
        "drivers": drivers,
        "monitors": [],
        "guards": [],
        "event_coverage": [],
        "lambda_values": list(lambda_values),
        "schedule_id": f"sched:{candidate_id}",
        "required_capabilities": [mode],
        "budget": {"max_attempts": 1, "max_cpu_hours": 1.0, "max_wall_seconds": 300},
        "failure_reasons": [],
    }
    if schedule_kind is not None:
        candidate["schedule_kind"] = schedule_kind
    return candidate


def _lambdas(n: int) -> list[float]:
    return [index / (n - 1) for index in range(n)]


def _geometry_to_rows(
    block: dict[int, tuple[float, float, float]], atom_rows: Sequence[int]
) -> list[list[float]]:
    return [list(block[map_id]) for map_id in atom_rows]


def _assemble_scan_input(
    compiled: co.CompiledRequest,
    *,
    elements: Sequence[str],
    start_rows: list[list[float]],
    fragment_index: int = 0,
) -> str:
    geom_block = compiled.geom_fragments[fragment_index]
    # todo-19 `_constraint_block` emits `%geom/Constraints/{...}/end` and leaves
    # the `%geom` block itself open (offline tests assert the fragment text only).
    # Close it at assembly so a real ORCA run is well-formed; record the gap for
    # todo 25/26 rather than silently shipping an unrunnable per-point input.
    if geom_block.startswith("%geom") and "Constraints" in geom_block:
        geom_block = geom_block + "\nend"
    recipe = compiled.recipe
    charge = int(recipe.get("charge", 0))
    multiplicity = int(recipe.get("multiplicity", 1))
    nprocs = int(recipe.get("nprocs", SMOKE_NPROCS))
    lines = [
        f"pes2ts-orca-smoke {compiled.candidate_id}",
        f"! {SMOKE_METHOD}",
        "%pal",
        f" nprocs {nprocs}",
        "end",
        geom_block,
        f"* xyz {charge} {multiplicity}",
    ]
    for symbol, row in zip(elements, start_rows, strict=True):
        lines.append(f"{symbol} {row[0]:.6f} {row[1]:.6f} {row[2]:.6f}")
    lines.append("*")
    lines.append("")
    return "\n".join(lines)


def _assemble_path_input(compiled: co.CompiledRequest) -> str:
    recipe = compiled.recipe
    nprocs = int(recipe.get("nprocs", SMOKE_NPROCS))
    lines = [
        f"pes2ts-orca-smoke {compiled.candidate_id}",
        f"! {SMOKE_METHOD}",
        "%pal",
        f" nprocs {nprocs}",
        "end",
    ]
    lines.extend(compiled.geom_fragments)
    lines.append("")
    return "\n".join(lines)


def _run_orca(executable: str, input_text: str, workdir: Path) -> tuple[int, str]:
    inp = workdir / "pes2ts_smoke.inp"
    out = workdir / "pes2ts_smoke.out"
    inp.write_text(input_text, encoding="utf-8")
    with out.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            [executable, str(inp)],
            cwd=workdir,
            stdout=handle,
            stderr=subprocess.STDOUT,
            timeout=RUN_TIMEOUT_SECONDS,
            check=False,
        )
    text = out.read_text(encoding="utf-8", errors="replace")
    return proc.returncode, text


def _output_excerpt(output: str, *, tail_lines: int = 12) -> dict[str, Any]:
    lines = [line for line in output.splitlines() if line.strip()]
    return {
        "terminated_normally": "ORCA TERMINATED NORMALLY" in output,
        "n_lines": len(lines),
        "head": lines[:6],
        "tail": lines[-tail_lines:],
    }


def _record_family_evidence(
    *,
    family: str,
    probe_id: str,
    modes: Sequence[str],
    version_line: str,
    compiled: co.CompiledRequest,
    run: dict[str, Any],
    directory: Path,
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    evidence_name = f"smoke_evidence_{probe_id}.json"
    evidence = {
        "probe_id": probe_id,
        "family": family,
        "date": date.today().isoformat(),
        "engine_version": version_line,
        "adapter_version": ADAPTER_VERSION_ORCA_DIRECT,
        "modes": list(modes),
        "method": SMOKE_METHOD,
        "compiled": {
            "candidate_id": compiled.candidate_id,
            "mode": compiled.mode,
            "compiled_kind": compiled.compiled_kind,
            "recipe_kind": compiled.recipe_kind,
            "total_points": compiled.total_points,
            "simultaneous": compiled.simultaneous,
            "atom_rows": list(compiled.atom_rows),
            "geom_fragments": list(compiled.geom_fragments),
            "point_input_sha256": list(compiled.point_input_sha256),
        },
        "run": run,
        "note": (
            "Probe receipt evidence for gated ORCA smoke (plan todo 24). "
            "Written to disk only; not merged into orca_capabilities_v1.json."
        ),
    }
    evidence_path = directory / evidence_name
    evidence_path.write_text(stable_json_dumps(evidence), encoding="utf-8")
    receipt = ProbeReceipt(
        probe_id=probe_id,
        date=date.today().isoformat(),
        engine_version=version_line,
        adapter_version=ADAPTER_VERSION_ORCA_DIRECT,
        status="pass",
        evidence_ref=evidence_name,
        modes=tuple(modes),
    )
    receipt_path = directory / f"probe_receipt_{probe_id}.json"
    receipt_path.write_text(stable_json_dumps(receipt.to_doc()), encoding="utf-8")
    return receipt_path, evidence_path


def _assert_normal_termination(returncode: int, output: str, *, family: str) -> None:
    excerpt = _output_excerpt(output)
    assert returncode == 0, (family, returncode, excerpt["tail"])
    assert excerpt["terminated_normally"], (family, excerpt)


# ---------------------------------------------------------------------------
# Representative molecules (tiny; smoke only — not chemistry certification).
# ---------------------------------------------------------------------------
H2_ROWS = (1, 2)
H2_ELEMENTS = ["H", "H"]
H2_GEOMETRY: dict[str, dict[int, tuple[float, float, float]]] = {
    "R": {1: (0.0, 0.0, 0.0), 2: (0.0, 0.0, 0.74)},
    "P": {1: (0.0, 0.0, 0.0), 2: (0.0, 0.0, 2.00)},
}

WATER_ROWS = (1, 2, 3)
WATER_ELEMENTS = ["O", "H", "H"]
WATER_GEOMETRY: dict[str, dict[int, tuple[float, float, float]]] = {
    "R": {1: (0.000, 0.000, 0.000), 2: (0.958, 0.000, 0.000), 3: (-0.239, 0.927, 0.000)},
    "P": {1: (0.000, 0.000, 0.000), 2: (1.300, 0.000, 0.000), 3: (-0.324, 1.255, 0.000)},
}

AD_ROWS = (1, 2, 3, 4)
AD_ELEMENTS = ["O", "O", "H", "H"]
AD_GEOMETRY: dict[str, dict[int, tuple[float, float, float]]] = {
    "R": {
        1: (0.00, 0.00, 0.00),
        2: (1.45, 0.00, 0.00),
        3: (-0.35, 0.90, 0.00),
        4: (1.80, 0.55, 0.75),
    },
    "P": {
        1: (0.00, 0.00, 0.00),
        2: (1.45, 0.00, 0.00),
        3: (-0.35, 0.90, 0.00),
        4: (1.80, 0.55, -0.75),
    },
}


@pytest.mark.orca
def test_orca_single_b_smoke(tmp_path: Path) -> None:
    # Given: a real ORCA executable (never skip on absence)
    executable = _resolve_orca_executable()
    version_line = _orca_version_line(executable)
    probe_id = "orca-single-b-smoke"
    capability = _session_capability(tmp_path, engine_version=version_line, probe_id=probe_id)

    # When: SINGLE_1D native Scan on H2 compiles and the real binary runs it
    lambdas = _lambdas(5)
    candidate = _scan_candidate(
        "SINGLE_1D", [_driver("B", [1, 2])], lambdas, candidate_id="cand-orca-smoke-single-b"
    )
    compiled = co.compile_orca(
        candidate, H2_ROWS, capability, geometry=H2_GEOMETRY, elements=H2_ELEMENTS,
        method=SMOKE_METHOD, charge=0, multiplicity=1, nprocs=SMOKE_NPROCS,
    )
    assert compiled.mode == "SINGLE_1D"
    assert compiled.total_points == 5
    assert "Scan B" in compiled.geom_fragments[0]
    assert "Simul_Scan" not in compiled.geom_fragments[0]
    input_text = _assemble_scan_input(
        compiled,
        elements=H2_ELEMENTS,
        start_rows=_geometry_to_rows(H2_GEOMETRY["R"], H2_ROWS),
    )
    returncode, output = _run_orca(executable, input_text, tmp_path)

    # Then: normal termination + version/run evidence recorded as probe receipts
    _assert_normal_termination(returncode, output, family="single_b")
    run = {
        "returncode": returncode,
        "input_sha256_prefix": compiled.point_input_sha256[0][:16],
        "n_points": compiled.total_points,
        "output": _output_excerpt(output),
    }
    receipt_path, evidence_path = _record_family_evidence(
        family="single_b",
        probe_id=probe_id,
        modes=["SINGLE_1D"],
        version_line=version_line,
        compiled=compiled,
        run=run,
        directory=_receipts_dir(),
    )
    receipt_doc = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt_doc["engine_version"] == version_line
    assert receipt_doc["modes"] == ["SINGLE_1D"]
    assert receipt_doc["status"] == "pass"
    assert evidence_path.is_file()


@pytest.mark.orca
def test_orca_double_b_simul_smoke(tmp_path: Path) -> None:
    # Given: a real ORCA executable
    executable = _resolve_orca_executable()
    version_line = _orca_version_line(executable)
    probe_id = "orca-double-b-simul-smoke"
    capability = _session_capability(tmp_path, engine_version=version_line, probe_id=probe_id)

    # When: COUPLED_1D double-B Simul_Scan on water compiles and runs
    lambdas = _lambdas(5)
    candidate = _scan_candidate(
        "COUPLED_1D",
        [_driver("B", [1, 2]), _driver("B", [1, 3])],
        lambdas,
        candidate_id="cand-orca-smoke-double-b",
    )
    compiled = co.compile_orca(
        candidate, WATER_ROWS, capability, geometry=WATER_GEOMETRY, elements=WATER_ELEMENTS,
        method=SMOKE_METHOD, charge=0, multiplicity=1, nprocs=SMOKE_NPROCS,
    )
    assert compiled.mode == "COUPLED_1D"
    assert compiled.simultaneous is True
    assert len(compiled.coordinates) == 2
    fragment = compiled.geom_fragments[0]
    assert fragment.count("Scan B") == 2
    assert "Simul_Scan true" in fragment
    input_text = _assemble_scan_input(
        compiled,
        elements=WATER_ELEMENTS,
        start_rows=_geometry_to_rows(WATER_GEOMETRY["R"], WATER_ROWS),
    )
    returncode, output = _run_orca(executable, input_text, tmp_path)

    # Then: normal termination + evidence
    _assert_normal_termination(returncode, output, family="double_b_simul")
    run = {
        "returncode": returncode,
        "n_drivers": len(compiled.coordinates),
        "simultaneous": compiled.simultaneous,
        "output": _output_excerpt(output),
    }
    receipt_path, evidence_path = _record_family_evidence(
        family="double_b_simul",
        probe_id=probe_id,
        modes=["COUPLED_1D"],
        version_line=version_line,
        compiled=compiled,
        run=run,
        directory=_receipts_dir(),
    )
    receipt_doc = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt_doc["modes"] == ["COUPLED_1D"]
    assert receipt_doc["engine_version"] == version_line
    assert evidence_path.is_file()


@pytest.mark.orca
def test_orca_ad_scan_smoke(tmp_path: Path) -> None:
    # Given: a real ORCA executable
    executable = _resolve_orca_executable()
    version_line = _orca_version_line(executable)
    probe_id = "orca-ad-scan-smoke"
    capability = _session_capability(tmp_path, engine_version=version_line, probe_id=probe_id)

    # When: COUPLED_1D A+D scan on a 4-atom system compiles and runs
    lambdas = _lambdas(5)
    candidate = _scan_candidate(
        "COUPLED_1D",
        [_driver("A", [3, 1, 2]), _driver("D", [3, 1, 2, 4])],
        lambdas,
        candidate_id="cand-orca-smoke-ad",
    )
    compiled = co.compile_orca(
        candidate, AD_ROWS, capability, geometry=AD_GEOMETRY, elements=AD_ELEMENTS,
        method=SMOKE_METHOD, charge=0, multiplicity=1, nprocs=SMOKE_NPROCS,
    )
    assert compiled.mode == "COUPLED_1D"
    kinds = [row.kind for row in compiled.coordinates]
    assert kinds == ["A", "D"]
    assert all(row.unit == "degree" for row in compiled.coordinates)
    fragment = compiled.geom_fragments[0]
    assert "Scan A" in fragment and "Scan D" in fragment
    assert "Simul_Scan true" in fragment
    input_text = _assemble_scan_input(
        compiled,
        elements=AD_ELEMENTS,
        start_rows=_geometry_to_rows(AD_GEOMETRY["R"], AD_ROWS),
    )
    returncode, output = _run_orca(executable, input_text, tmp_path)

    # Then: normal termination + evidence
    _assert_normal_termination(returncode, output, family="ad_scan")
    run = {
        "returncode": returncode,
        "kinds": kinds,
        "simultaneous": compiled.simultaneous,
        "output": _output_excerpt(output),
    }
    receipt_path, evidence_path = _record_family_evidence(
        family="ad_scan",
        probe_id=probe_id,
        modes=["COUPLED_1D"],
        version_line=version_line,
        compiled=compiled,
        run=run,
        directory=_receipts_dir(),
    )
    receipt_doc = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt_doc["modes"] == ["COUPLED_1D"]
    assert evidence_path.is_file()


@pytest.mark.orca
def test_orca_nonuniform_per_point_smoke(tmp_path: Path) -> None:
    # Given: a real ORCA executable
    executable = _resolve_orca_executable()
    version_line = _orca_version_line(executable)
    probe_id = "orca-nonuniform-per-point-smoke"
    capability = _session_capability(tmp_path, engine_version=version_line, probe_id=probe_id)

    # When: SCHEDULED_1D non-uniform λ list compiles to per-point Constraints
    schedule_values = [0.0, 0.15, 0.4, 0.75, 1.0]
    candidate = _scan_candidate(
        "SCHEDULED_1D",
        [_driver("B", [1, 2], schedule_values=schedule_values)],
        list(schedule_values),
        candidate_id="cand-orca-smoke-nonuniform",
        schedule_kind="custom",
    )
    compiled = co.compile_orca(
        candidate, H2_ROWS, capability, geometry=H2_GEOMETRY, elements=H2_ELEMENTS,
        method=SMOKE_METHOD, charge=0, multiplicity=1, nprocs=SMOKE_NPROCS,
    )
    assert compiled.mode == "SCHEDULED_1D"
    assert compiled.compiled_kind == "recipe"
    assert compiled.recipe_kind == co.RECIPE_PER_POINT
    assert compiled.total_points == len(schedule_values)
    assert len(compiled.geom_fragments) == compiled.total_points
    assert len(compiled.point_input_sha256) == compiled.total_points

    start_rows = _geometry_to_rows(H2_GEOMETRY["R"], H2_ROWS)
    runs: list[dict[str, Any]] = []
    for point_index in range(compiled.total_points):
        input_text = _assemble_scan_input(
            compiled,
            elements=H2_ELEMENTS,
            start_rows=start_rows,
            fragment_index=point_index,
        )
        point_dir = tmp_path / f"point_{point_index}"
        point_dir.mkdir()
        returncode, output = _run_orca(executable, input_text, point_dir)
        _assert_normal_termination(returncode, output, family=f"nonuniform[{point_index}]")
        runs.append(
            {
                "point_index": point_index,
                "returncode": returncode,
                "input_sha256": compiled.point_input_sha256[point_index],
                "output": _output_excerpt(output),
            }
        )

    # Then: every per-point run terminated normally + evidence recorded
    receipt_path, evidence_path = _record_family_evidence(
        family="nonuniform_per_point",
        probe_id=probe_id,
        modes=["SCHEDULED_1D"],
        version_line=version_line,
        compiled=compiled,
        run={"n_points": compiled.total_points, "points": runs},
        directory=_receipts_dir(),
    )
    receipt_doc = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt_doc["modes"] == ["SCHEDULED_1D"]
    assert receipt_doc["engine_version"] == version_line
    assert evidence_path.is_file()


@pytest.mark.orca
def test_orca_neb_minimal_smoke(tmp_path: Path) -> None:
    # Given: a real ORCA executable
    executable = _resolve_orca_executable()
    version_line = _orca_version_line(executable)
    probe_id = "orca-neb-minimal-smoke"
    capability = _session_capability(tmp_path, engine_version=version_line, probe_id=probe_id)

    # When: PathCandidateV1 minimal NEB shape compiles and the binary runs it
    n_images = 3
    candidate: dict[str, Any] = {
        "candidate_kind": "PathCandidateV1",
        "candidate_id": "cand-orca-smoke-neb",
        "n_atoms": 2,
        "endpoint_geometries": {
            "reactant": _geometry_to_rows(H2_GEOMETRY["R"], H2_ROWS),
            "product": _geometry_to_rows(H2_GEOMETRY["P"], H2_ROWS),
        },
        "image_chain": {"n_images": n_images},
        "start_endpoint": "R",
        "direction": "R_to_P",
        "anchor_reason": "SMOKE",
        "method_kind": "NEB",
        "failure_reasons": [],
    }
    compiled = co.compile_orca(
        candidate, H2_ROWS, capability, elements=H2_ELEMENTS,
        method=SMOKE_METHOD, charge=0, multiplicity=1, nprocs=SMOKE_NPROCS,
    )
    assert compiled.mode == "PATH_NEB"
    assert compiled.total_points == n_images
    assert compiled.coordinates == ()
    assert any("Path" in fragment for fragment in compiled.geom_fragments)
    assert len(compiled.geom_fragments) == 3  # path block + two endpoint xyz blocks
    input_text = _assemble_path_input(compiled)
    returncode, output = _run_orca(executable, input_text, tmp_path)

    # Then: normal termination + version/run evidence recorded
    _assert_normal_termination(returncode, output, family="neb_minimal")
    run = {
        "returncode": returncode,
        "n_images": n_images,
        "mode": compiled.mode,
        "output": _output_excerpt(output),
    }
    receipt_path, evidence_path = _record_family_evidence(
        family="neb_minimal",
        probe_id=probe_id,
        modes=["PATH_NEB"],
        version_line=version_line,
        compiled=compiled,
        run=run,
        directory=_receipts_dir(),
    )
    receipt_doc = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt_doc["modes"] == ["PATH_NEB"]
    assert receipt_doc["engine_version"] == version_line
    assert evidence_path.is_file()
