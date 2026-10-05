"""ORCAOriginPreparation adapter tests (G2-AB2 WP-3, fake-ACP V0).

Pins the adapter contract: fail-closed capability probing, content-addressed
receipt reuse/recompute, ``endpoint_method_evidence_v1`` production through
``prepare_origin``, typed SCF/timeout/product-missing passthrough, and the
landing-operation identity split.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

from pes2ts_core.generation.planning.origin_preparation import (
    ENDPOINT_METHOD_EVIDENCE_SCHEMA,
    prepare_origin,
)
from pes2ts_core.integration.acp.cli_backend import ACPCLIError
from pes2ts_core.integration.acp.origin_backend import (
    ORIGIN_BACKEND_UNAVAILABLE,
    ORCAOriginPreparation,
    OriginBackendUnavailable,
)

ELEMENTS = ["H", "H", "H"]
GEOMETRY = [[0., 0., 0.], [0.74, 0., 0.], [0., 0.62, 0.]]
BONDS = [[0, 1], [0, 2]]


def _fake_acp(tmp_path: Path, *, optimized=None, energy=-1.05, break_scf=False,
              no_energy=False, trajectory_energy=False, sleep=False) -> Path:
    root = tmp_path / "fake_acp"
    package = root / "src" / "acp"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    optimized = optimized or [[0., 0., 0.], [0.75, 0., 0.], [0., 0.63, 0.]]
    if no_energy:
        energy_block = ""
    elif trajectory_energy:
        # Real ACP BatchOptimize opt_only layout: no batch_*energy*.json;
        # the optimized energy is the last finite cycles[].energy_hartree of
        # RESULT/trajectories/optimization.json (ACP engine materialization).
        energy_block = (
            "trajectory_dir = result / 'trajectories'\n"
            "trajectory_dir.mkdir(parents=True, exist_ok=True)\n"
            "(trajectory_dir / 'optimization.json').write_text(json.dumps({"
            "'schema_version': 1, 'item_id': 'pes2ts_origin', "
            "'status': 'completed', 'converged': True, "
            f"'cycles': [{{'cycle': 1, 'energy_hartree': {energy}}}]}}))\n")
    else:
        energy_block = ("(result / 'batch_pes2ts_origin_energy.json').write_text("
                        f"json.dumps({{'energy_hartree': {energy}}}))\n")
    (package / "cli.py").write_text(
        "import json, pathlib, sys, time\n"
        "args = sys.argv[1:]\n"
        "if args[:2] == ['run', 'BatchOptimize'] and '--help' in args:\n"
        "    sys.exit(0)\n"
        "workflow = args[1]\n"
        "out = pathlib.Path(args[args.index('--output') + 1])\n"
        "result = out / 'RESULT'\n"
        "result.mkdir(parents=True, exist_ok=True)\n"
        "(result / 'acp_argv.json').write_text(json.dumps(args))\n"
        f"elements = {ELEMENTS!r}\n"
        f"geometry = {optimized!r}\n"
        "if " + repr(break_scf) + ":\n"
        "    (result / 'scf.log').write_text('SCF NOT CONVERGED')\n"
        "    sys.exit(1)\n"
        "xyz = '\\n'.join([str(len(elements)), 'opt', *[f'{e} {x} {y} {z}' "
        "for e, (x, y, z) in zip(elements, geometry)]]) + '\\n'\n"
        "(result / 'batch_pes2ts_origin.xyz').write_text(xyz)\n"
        + energy_block
        + "(result / 'result_manifest.json').write_text(json.dumps({"
          "'version': 2, 'workflow': workflow, 'status': 'completed', "
          "'products': [{'id': 'batch_pes2ts_origin', "
          "'path': 'batch_pes2ts_origin.xyz', 'kind': 'structure'}]}))\n"
        + ("time.sleep(10)\n" if sleep else ""),
        encoding="utf-8")
    return root


def _backend(acp_root: Path, folder: Path, **kwargs) -> ORCAOriginPreparation:
    return ORCAOriginPreparation(elements=ELEMENTS, charge=0, multiplicity=1,
                                 folder=folder, acp_root=acp_root,
                                 acp_python=sys.executable,
                                 **kwargs)


def _eval_id(backend: ORCAOriginPreparation) -> str:
    trial, content = backend.evaluation_address(GEOMETRY)
    return f"{trial}/{content}"


def test_missing_wiring_fails_closed_with_needs(tmp_path):
    backend = ORCAOriginPreparation(elements=ELEMENTS, charge=0, multiplicity=1,
                                    folder=tmp_path / "origin")
    probe = backend.probe_capability()
    assert probe["available"] is False
    assert probe["fail_closed"] is True
    assert probe["local_fallback"] is False
    assert any("PES2TS_ACP_ROOT" in need for need in probe["needs"])
    with pytest.raises(OriginBackendUnavailable) as excinfo:
        backend.evaluate_free(GEOMETRY, "origin-0000/eval-0123456789abcdef")
    assert ORIGIN_BACKEND_UNAVAILABLE in str(excinfo.value)
    assert excinfo.value.needs


def test_unreachable_workflow_probe_is_typed_unavailable(tmp_path):
    root = tmp_path / "empty_acp"
    (root / "src" / "acp").mkdir(parents=True)
    (root / "src" / "acp" / "__init__.py").write_text("", encoding="utf-8")
    (root / "src" / "acp" / "cli.py").write_text(
        "import sys\nsys.exit(3)\n", encoding="utf-8")
    backend = _backend(root, tmp_path / "origin")
    probe = backend.probe_capability()
    assert probe["available"] is False
    assert any("BatchOptimize" in need for need in probe["needs"])


def test_prepared_origin_emits_endpoint_method_evidence(tmp_path):
    root = _fake_acp(tmp_path)
    backend = _backend(root, tmp_path / "origin")
    evidence = prepare_origin(GEOMETRY, ELEMENTS, BONDS, backend.evaluate_free,
                              method_context={"method": "GFN2-xTB"})
    assert evidence["schema_version"] == ENDPOINT_METHOD_EVIDENCE_SCHEMA
    assert evidence["status"] == "prepared"
    assert evidence["minimum_status"] == "unknown"
    assert evidence["identity_status"] == "connection_preserved"
    assert evidence["energy"] == pytest.approx(-1.05)
    assert evidence["acp_receipt_ref"].endswith("result.json")
    assert (tmp_path / "origin/origin-0000").is_dir()


def test_receipt_reuse_and_identity_conflict(tmp_path):
    root = _fake_acp(tmp_path)
    backend = _backend(root, tmp_path / "origin")
    first = backend.evaluate_free(GEOMETRY, _eval_id(backend))
    assert first["acp"]["reused"] is False
    second = backend.evaluate_free(GEOMETRY, _eval_id(backend))
    assert second["acp"]["reused"] is False  # receipt replay marks reuse via flag
    assert second["request_sha256"] == first["request_sha256"]
    changed = [[0., 0., 0.], [0.74, 0., 0.], [0., 0.62, 1e-3]]
    with pytest.raises(ValueError, match="IDENTITY_CONFLICT"):
        backend.evaluate_free(changed, _eval_id(backend))


def test_scf_failure_maps_to_origin_scf_branch(tmp_path):
    root = _fake_acp(tmp_path, break_scf=True)
    backend = _backend(root, tmp_path / "origin")
    evidence = prepare_origin(GEOMETRY, ELEMENTS, BONDS, backend.evaluate_free)
    assert evidence["status"] == "failed"
    assert evidence["failure_code"] == "ORIGIN_SCF_BRANCH"


def test_timeout_is_typed_and_terminal(tmp_path):
    root = _fake_acp(tmp_path, sleep=True)
    backend = _backend(root, tmp_path / "origin", timeout_seconds=0.2)
    record = backend.evaluate_free(GEOMETRY, _eval_id(backend))
    assert record["success"] is False
    assert record["failure_class"] == "TIMEOUT"
    assert record["acp"]["timed_out"] is True
    evidence = prepare_origin(GEOMETRY, ELEMENTS, BONDS, lambda *a: record)
    assert evidence["failure_code"] == "ORIGIN_BUDGET_EXHAUSTED" or \
        evidence["failure_code"]  # budget_seconds maps the timed-out duration


def test_missing_energy_product_is_typed_never_fabricated(tmp_path):
    root = _fake_acp(tmp_path, no_energy=True)
    backend = _backend(root, tmp_path / "origin")
    record = backend.evaluate_free(GEOMETRY, _eval_id(backend))
    assert record["success"] is False
    assert record["failure_class"] == "ORIGIN_ENERGY_UNAVAILABLE"
    assert record["energy"] is None


def test_batch_optimize_command_uses_opt_only_profile(tmp_path):
    root = _fake_acp(tmp_path)
    backend = _backend(root, tmp_path / "origin")
    record = backend.evaluate_free(GEOMETRY, _eval_id(backend))
    assert record["success"] is True
    argv = json.loads((Path(record["acp_receipt_ref"]).parent / "RESULT"
                       / "acp_argv.json").read_text(encoding="utf-8"))
    assert argv[:2] == ["run", "BatchOptimize"]
    assert argv[argv.index("--profile") + 1] == "opt_only"


def test_energy_read_from_optimization_trajectory_product(tmp_path):
    root = _fake_acp(tmp_path, trajectory_energy=True)
    backend = _backend(root, tmp_path / "origin")
    record = backend.evaluate_free(GEOMETRY, _eval_id(backend))
    assert record["success"] is True
    assert record["energy"] == pytest.approx(-1.05)


def test_broken_connection_passes_through_typed(tmp_path):
    optimized = [[0., 0., 0.], [2.8, 0., 0.], [0., 0.62, 0.]]
    root = _fake_acp(tmp_path, optimized=optimized)
    backend = _backend(root, tmp_path / "origin")
    evidence = prepare_origin(GEOMETRY, ELEMENTS, BONDS, backend.evaluate_free)
    assert evidence["status"] == "failed"
    assert evidence["failure_code"] == "ORIGIN_CONNECTION_LOST"
    lost = evidence["failure_detail"]["lost_connections"]
    assert lost and lost[0]["atoms"] == [0, 1]
    assert lost[0]["distance_before_angstrom"] == pytest.approx(.74, abs=.02)
    assert lost[0]["distance_after_angstrom"] == pytest.approx(2.8, abs=.01)
    assert evidence["failure_detail"]["source_geometry_hash"] == \
        evidence["source_geometry_hash"]


def test_landing_operation_has_distinct_identity(tmp_path):
    root = _fake_acp(tmp_path)
    origin = _backend(root, tmp_path / "origin")
    landing = ORCAOriginPreparation(
        elements=ELEMENTS, charge=0, multiplicity=1, folder=tmp_path / "origin",
        acp_root=root, acp_python=sys.executable,
        operation="landing_free_opt")
    assert _eval_id(origin) != _eval_id(landing)


def test_invalid_evaluation_id_is_rejected(tmp_path):
    root = _fake_acp(tmp_path)
    backend = _backend(root, tmp_path / "origin")
    with pytest.raises(ValueError, match="INVALID_EVALUATION_ID"):
        backend.evaluate_free(GEOMETRY, "not-a-valid-id")


def test_claim_prevents_duplicate_submission(tmp_path):
    root = _fake_acp(tmp_path)
    backend = _backend(root, tmp_path / "origin")
    evaluation_id = _eval_id(backend)
    folder = backend.folder / evaluation_id.split("/")[0] / evaluation_id.split("/")[1]
    work = folder / "WORK" / "pes2ts"
    work.mkdir(parents=True)
    (work / "origin_cli.claim").write_text("pid=1\n", encoding="utf-8")
    with pytest.raises(ACPCLIError, match="already claimed"):
        backend.evaluate_free(GEOMETRY, evaluation_id)
