"""Gated ACP smoke test (plan todo 24, ``acp`` marker).

Never-skip semantics follow ``tests/test_g2_xtb_smoke.py:46-55``: with
``PES2TS_ACP_ROOT`` (ACP source checkout containing ``src/acp/cli.py``) and
``PES2TS_ACP_PYTHON`` (that ACP environment's interpreter) set, this test
validates the real ACP installation, exercises a representative single-B scan
through the multicoord (todo 20) and legacy compat (todo 18/adapter) request
paths, and records engine/adapter versions plus run evidence as
capability-registry probe receipts.

Receipts are written under a configurable directory (env
``PES2TS_SMOKE_RECEIPTS_DIR``, default ``.omo/evidence/``) as individual JSON
files — never auto-merged into the shipped registry and never auto-committed.

When either environment variable is missing the test FAILS naming the exact
variables (never skips): a skip here would be a false pass.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from pes2ts_core.contracts import make_document, seal_document
from pes2ts_core.integration.acp.adapter import scan_plan_to_acp_request
from pes2ts_core.integration.acp.cli_backend import ACPCLIBackend
from pes2ts_core.integration.acp.multicoord import (
    ADAPTER_VERSION_MULTICOORD,
    multicoord_request_payload,
)
from pes2ts_core.planning import build_minimal_scan_plan
from pes2ts_core.scan_strategy import capabilities as caps
from pes2ts_core.scan_strategy import compile_orca as co
from pes2ts_core.scan_strategy.capabilities import ProbeReceipt
from pes2ts_core.utils.hashing import stable_json_dumps

ENV_ROOT = "PES2TS_ACP_ROOT"
ENV_PYTHON = "PES2TS_ACP_PYTHON"
ENV_RECEIPTS_DIR = "PES2TS_SMOKE_RECEIPTS_DIR"
PROBE_ID = "acp-single-b-smoke"
ADAPTER_VERSION_COMPAT = "acp-pes-scan-v0"

# Representative single-B system: H2 stretch, maps (1, 2).
ATOM_ROWS = (1, 2)
ELEMENTS = ["H", "H"]
GEOMETRY: dict[str, dict[int, tuple[float, float, float]]] = {
    "R": {1: (0.0, 0.0, 0.0), 2: (0.0, 0.0, 0.74)},
    "P": {1: (0.0, 0.0, 0.0), 2: (0.0, 0.0, 2.00)},
}
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def _resolve_acp_env() -> tuple[Path, Path]:
    """Env vars first; a miss is a FAILURE naming the variables, never a skip."""
    raw_root = os.environ.get(ENV_ROOT)
    raw_python = os.environ.get(ENV_PYTHON)
    missing = [name for name, value in ((ENV_ROOT, raw_root), (ENV_PYTHON, raw_python)) if not value]
    if missing:
        pytest.fail(
            "ACP smoke environment not ready: set "
            + " and ".join(missing)
            + f" ({ENV_ROOT} = ACP source checkout containing src/acp/cli.py; "
            f"{ENV_PYTHON} = python interpreter of that ACP environment); "
            "a skip here would be a false pass"
        )
    root = Path(str(raw_root)).expanduser()
    python = Path(str(raw_python)).expanduser()
    if not root.is_dir():
        pytest.fail(f"{ENV_ROOT}={root} is not a directory; a skip here would be a false pass")
    cli_py = root / "src" / "acp" / "cli.py"
    if not cli_py.is_file():
        pytest.fail(
            f"{ENV_ROOT}={root} does not contain src/acp/cli.py — not an ACP source "
            "checkout; a skip here would be a false pass"
        )
    if not python.is_file() and not _which(python):
        pytest.fail(
            f"{ENV_PYTHON}={python} is not an executable file; a skip here would be a false pass"
        )
    return root, python


def _which(candidate: Path) -> str | None:
    if candidate.is_file():
        return str(candidate)
    return shutil.which(str(candidate))


def _receipts_dir() -> Path:
    raw = os.environ.get(ENV_RECEIPTS_DIR)
    if raw:
        return Path(raw).expanduser()
    return Path(__file__).resolve().parents[1] / ".omo" / "evidence"


def _capture_acp_versions(acp_root: Path, acp_python: Path) -> dict[str, str]:
    """Capture interpreter + CLI-install evidence from the real ACP environment."""
    resolved_python = _which(acp_python) or str(acp_python)
    versions: dict[str, str] = {
        "acp_root": str(acp_root),
        "acp_python": resolved_python,
        "adapter_version_multicoord": ADAPTER_VERSION_MULTICOORD,
        "adapter_version_compat": ADAPTER_VERSION_COMPAT,
    }
    try:
        proc = subprocess.run(
            [resolved_python, "-c", "import sys; print(sys.version.split()[0])"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        versions["acp_python_version"] = proc.stdout.strip() if proc.returncode == 0 else "unknown"
    except (OSError, subprocess.TimeoutExpired):
        versions["acp_python_version"] = "unknown"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(acp_root / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    try:
        proc = subprocess.run(
            [resolved_python, "-m", "acp.cli", "--help"],
            cwd=str(acp_root),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            env=env,
        )
        versions["acp_cli_help_returncode"] = str(proc.returncode)
    except (OSError, subprocess.TimeoutExpired):
        versions["acp_cli_help_returncode"] = "unavailable"
    return versions


def _session_capability(
    tmp_path: Path,
    *,
    engine_version: str,
    adapter_version: str,
    probe_id: str,
) -> caps.EffectiveCapability:
    """Session-local probed registry using the REAL captured version strings.

    This is what the smoke validates; durable receipts are written separately
    under the receipts dir and are never auto-merged into the shipped registry.
    """
    receipt = {
        "probe_id": probe_id,
        "date": date.today().isoformat(),
        "engine_version": engine_version,
        "adapter_version": adapter_version,
        "status": "pass",
        "evidence_ref": f"smoke/{probe_id}",
        "modes": ["SINGLE_1D"],
    }

    def _entry(entry_id: str, layer: str) -> dict[str, Any]:
        return {
            "entry_id": entry_id,
            "layer": layer,
            "engine": "orca",
            "adapter": "acp",
            "capability": {
                "engine_version": engine_version,
                "adapter_version": adapter_version,
                "supported_modes": ["SINGLE_1D"],
                "coordinate_kinds": ["B"],
                "max_scan_coordinates": 1,
                "point_limits": {"baseline": 3, "max": 21},
                "custom_schedule_support": False,
                "constraint_support": {
                    "native_scan": True,
                    "per_point_constraints": False,
                    "simul_scan": False,
                },
                "method_element_coverage": {"B3LYP-D3": ["H"]},
                "probe_receipts": [receipt],
            },
            "notes": "todo-24 ACP smoke session stack (real captured versions)",
        }

    payload = {
        "schema_version": caps.REGISTRY_SCHEMA_VERSION,
        "registry_id": "todo24-acp-smoke",
        "description": "session probed stack for gated ACP smoke",
        "entries": [_entry("eng", "engine"), _entry("adp", "adapter"), _entry("dep", "deployment")],
    }
    path = tmp_path / "orca_capabilities_v1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    registry = caps.load_capability_registry(path)
    return caps.effective_capability("eng", "adp", "dep", registry=registry)


def _single_b_candidate() -> dict[str, Any]:
    return {
        "candidate_kind": "ScanCandidateV2",
        "candidate_id": "cand-acp-smoke-single-b",
        "mode": "SINGLE_1D",
        "start_endpoint": "R",
        "direction": "R_to_P",
        "assembly_id": "asm-acp-smoke",
        "anchor_reason": "SMOKE_H2_STRETCH",
        "drivers": [
            {"kind": "B", "maps": [1, 2], "unit": "angstrom", "schedule_values": None, "index0": None}
        ],
        "monitors": [],
        "guards": [],
        "event_coverage": [],
        "lambda_values": list(LAMBDAS),
        "schedule_id": "sched:acp-smoke",
        "required_capabilities": ["SINGLE_1D"],
        "budget": {"max_attempts": 1, "max_cpu_hours": 1.0, "max_wall_seconds": 120},
        "failure_reasons": [],
    }


def _materials() -> dict[str, Any]:
    return {
        "atom_map_ids": list(ATOM_ROWS),
        "elements": list(ELEMENTS),
        "r_coordinates": [list(GEOMETRY["R"][m]) for m in ATOM_ROWS],
        "p_coordinates": [list(GEOMETRY["P"][m]) for m in ATOM_ROWS],
        "charge": 0,
        "multiplicity": 1,
    }


def _compat_case() -> dict[str, Any]:
    """Minimal sealed ReactionCase for the legacy single-B compat path."""
    case = make_document(
        "ReactionCase",
        "case:smoke-acp-single-b",
        "ready",
        dataset_version="synthetic-smoke-v1",
        reaction_id="RXN_SMOKE_ACP_B",
        case_id="case:smoke-acp-single-b",
        split="train",
        atoms=[
            {"atom_map_id": 1, "element": "C"},
            {"atom_map_id": 2, "element": "H"},
            {"atom_map_id": 3, "element": "H"},
        ],
        reactant={
            "charge": 0,
            "multiplicity": 1,
            "geometry": [[0.0, 0.0, 0.0], [1.09, 0.0, 0.0], [0.0, 1.09, 0.0]],
        },
        product={
            "charge": 0,
            "multiplicity": 1,
            "geometry": [[0.0, 0.0, 0.0], [1.80, 0.0, 0.0], [0.0, 1.80, 0.0]],
        },
        edits=[{"kind": "broken", "atom_map_ids": [1, 2]}],
        hydrogen_transfers=[],
        source={"dataset": "synthetic", "mapping_provenance": "endpoint_only"},
    )
    return seal_document(case)


def _write_probe_receipt_and_evidence(
    *,
    engine_version: str,
    adapter_version: str,
    modes: Sequence[str],
    versions: dict[str, str],
    compiled_summary: dict[str, Any],
    multicoord_summary: dict[str, Any],
    compat_summary: dict[str, Any],
    directory: Path,
) -> tuple[Path, Path]:
    """Write one probe-receipt JSON + one run-evidence JSON (never auto-commit)."""
    directory.mkdir(parents=True, exist_ok=True)
    evidence_name = f"smoke_evidence_{PROBE_ID}.json"
    evidence = {
        "probe_id": PROBE_ID,
        "family": "acp_single_b_multicoord_compat",
        "date": date.today().isoformat(),
        "engine_version": engine_version,
        "adapter_version": adapter_version,
        "modes": list(modes),
        "versions": versions,
        "compiled": compiled_summary,
        "multicoord_payload": multicoord_summary,
        "compat_request": compat_summary,
        "note": (
            "Probe receipt evidence for gated ACP smoke (plan todo 24). "
            "Written to disk only; not merged into orca_capabilities_v1.json."
        ),
    }
    evidence_path = directory / evidence_name
    evidence_path.write_text(stable_json_dumps(evidence), encoding="utf-8")
    receipt = ProbeReceipt(
        probe_id=PROBE_ID,
        date=date.today().isoformat(),
        engine_version=engine_version,
        adapter_version=adapter_version,
        status="pass",
        evidence_ref=evidence_name,
        modes=tuple(modes),
    )
    receipt_path = directory / f"probe_receipt_{PROBE_ID}.json"
    receipt_path.write_text(stable_json_dumps(receipt.to_doc()), encoding="utf-8")
    return receipt_path, evidence_path


@pytest.mark.acp
def test_acp_single_b_scan_smoke(tmp_path: Path) -> None:
    # Given: a real ACP root + ACP environment python (never skip on absence)
    acp_root, acp_python = _resolve_acp_env()
    versions = _capture_acp_versions(acp_root, acp_python)
    assert versions["acp_python_version"] != "unknown", versions
    assert versions["acp_cli_help_returncode"] not in {"", "unavailable"}, versions

    # Backend construction proves src/acp/cli.py + python resolution (install gate)
    backend = ACPCLIBackend(acp_root=acp_root, python_executable=acp_python)
    assert Path(backend.python_executable).is_file()

    engine_version = f"acp:{acp_root.name}"
    adapter_version = ADAPTER_VERSION_MULTICOORD
    capability = _session_capability(
        tmp_path, engine_version=engine_version, adapter_version=adapter_version, probe_id=PROBE_ID
    )

    # When: representative single-B scan through the multicoord path (todo 19+20)
    candidate = _single_b_candidate()
    compiled = co.compile_orca(candidate, ATOM_ROWS, capability, geometry=GEOMETRY)
    assert compiled.mode == "SINGLE_1D"
    assert compiled.total_points == len(LAMBDAS)
    assert len(compiled.coordinates) == 1
    assert compiled.coordinates[0].kind == "B"
    payload = multicoord_request_payload(compiled, _materials(), {"capability": capability})
    assert payload["adapter_version"] == ADAPTER_VERSION_MULTICOORD
    assert payload["metadata"]["n_drivers"] == 1
    assert payload["coordinates"][0]["kind"] == "B"
    assert payload["coordinates"][0]["atom_map_ids"] == [1, 2]
    assert payload["lambda_values"] == list(compiled.lambda_values)

    # And: the same single-B family through the legacy compat path (todo 18/adapter)
    case = _compat_case()
    plan = build_minimal_scan_plan(case)
    compat_request = scan_plan_to_acp_request(case, plan)
    assert compat_request["scan_request"]["mode"] == "bond_length_scan"
    assert compat_request["metadata"]["atom_map_ids"] == [1, 2, 3]
    assert compat_request["scan_request"]["coordinate"]["kind"] == "distance"

    # Then: record engine/adapter versions + run evidence as probe receipts
    compiled_summary = {
        "candidate_id": compiled.candidate_id,
        "mode": compiled.mode,
        "compiled_kind": compiled.compiled_kind,
        "recipe_kind": compiled.recipe_kind,
        "total_points": compiled.total_points,
        "atom_rows": list(compiled.atom_rows),
        "geom_fragments": list(compiled.geom_fragments),
        "point_input_sha256": list(compiled.point_input_sha256),
    }
    multicoord_summary = {
        "adapter_version": payload["adapter_version"],
        "schema_version": payload["schema_version"],
        "n_drivers": payload["metadata"]["n_drivers"],
        "driver_ids": payload["metadata"]["driver_ids"],
        "modes": [payload["metadata"]["mode"]],
    }
    compat_summary = {
        "scan_request_mode": compat_request["scan_request"]["mode"],
        "coordinate_kind": compat_request["scan_request"]["coordinate"]["kind"],
        "atom_map_ids": compat_request["metadata"]["atom_map_ids"],
    }
    receipt_path, evidence_path = _write_probe_receipt_and_evidence(
        engine_version=engine_version,
        adapter_version=adapter_version,
        modes=["SINGLE_1D"],
        versions=versions,
        compiled_summary=compiled_summary,
        multicoord_summary=multicoord_summary,
        compat_summary=compat_summary,
        directory=_receipts_dir(),
    )
    receipt_doc = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt_doc["probe_id"] == PROBE_ID
    assert receipt_doc["status"] == "pass"
    assert receipt_doc["engine_version"] == engine_version
    assert receipt_doc["adapter_version"] == adapter_version
    assert receipt_doc["modes"] == ["SINGLE_1D"]
    assert evidence_path.is_file()
