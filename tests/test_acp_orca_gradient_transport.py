"""Fake-ACP transport tests for the ACP ``OrcaGradient`` CLI runner (X4′-B).

Mirrors the fake-ACP-process pattern of ``test_acp_xtb_path_transport.py``: a
stub ``src/acp/cli.py`` stands in for the real ACP engine and writes a minimal
valid ``RESULT/`` tree (v2 ``OrcaGradient`` manifest + gradient/geometry/
energy products).  No real ORCA binary is ever invoked.

Covers the success path (receipt, manifest validation, evidence projection,
attempt reuse), non-zero exit, timeout, malformed manifest/product, missing
ACP wiring, and the request builder's strict validation.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import pytest

from pes2ts_core.integration.acp.cli_backend import ACPCLIError
from pes2ts_core.integration.acp.gradient_backend import (
    ORCAGradientBackend,
    resolve_acp_wiring,
    write_json,
)
from pes2ts_core.integration.acp.orca_gradient_request import (
    ACP_WORKFLOW_NAME,
    build_gradient_request,
    request_sha256,
    validate_gradient_request,
)
from pes2ts_core.integration.acp.orca_gradient_transport import (
    OrcaGradientAttemptResult,
    parse_gradient_product_evidence,
    run_orca_gradient_attempt,
    validate_orca_gradient_result,
)
from pes2ts_core.generation.planning.gradient_evidence import BOHR_ANGSTROM
from pes2ts_core.utils.truth_guard import SUBPROCESS_IMPORT_ALLOWLIST, scan_truth_access

GEOMETRY = [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]
ELEMENTS = ["H", "H"]
ENERGY = -1.116
GRADIENT_BOHR = [[0.01, -0.02, 0.0], [-0.01, 0.02, 0.0]]

#: Stub ACP CLI. Test hooks arrive as extra argv tokens the transport passes
#: through verbatim: ``--test-delay S``, ``--test-exit-code N``,
#: ``--test-bad-manifest``, ``--test-missing-product``,
#: ``--test-bad-gradient``, ``--test-symbol-mismatch``.
_FAKE_CLI = '''\
import hashlib, json, math, pathlib, sys, time

args = sys.argv[1:]
delay = 0.0
exit_code = 0
bad_manifest = False
missing_product = False
bad_gradient = False
symbol_mismatch = False
i = 0
while i < len(args):
    if args[i] == "--test-delay":
        delay = float(args[i + 1]); i += 2; continue
    if args[i] == "--test-exit-code":
        exit_code = int(args[i + 1]); i += 2; continue
    if args[i] == "--test-bad-manifest":
        bad_manifest = True; i += 1; continue
    if args[i] == "--test-missing-product":
        missing_product = True; i += 1; continue
    if args[i] == "--test-bad-gradient":
        bad_gradient = True; i += 1; continue
    if args[i] == "--test-symbol-mismatch":
        symbol_mismatch = True; i += 1; continue
    i += 1
request_path = pathlib.Path(args[args.index("--gradient-config") + 1])
output = pathlib.Path(args[args.index("--output") + 1])
request = json.loads(request_path.read_text(encoding="utf-8"))
assert request.get("schema_version") == "pes2ts_orca_gradient_request_v1"
time.sleep(delay)
if exit_code:
    raise SystemExit(exit_code)
geometry = request["geometry"]
elements = list(request["elements"])
if symbol_mismatch:
    elements = ["Xx"] * len(elements)
gradient = [[0.01, -0.02, 0.0], [-0.01, 0.02, 0.0]]
if bad_gradient:
    gradient = "not-a-gradient"
bohr = 0.529177210903
gradient_ang = [[v / bohr for v in row] for row in gradient] if isinstance(gradient, list) else gradient
energy = -1.116
result = output / "RESULT"
(result / "gradient").mkdir(parents=True, exist_ok=True)
(result / "geometry").mkdir(parents=True, exist_ok=True)
(result / "energy").mkdir(parents=True, exist_ok=True)
xyz_lines = [str(len(elements)), "OrcaGradient input geometry"]
for symbol, row in zip(elements, geometry):
    xyz_lines.append("%s %15.10f %15.10f %15.10f" % (symbol, row[0], row[1], row[2]))
(result / "geometry" / "geometry.xyz").write_text("\\n".join(xyz_lines) + "\\n", encoding="utf-8")
(result / "energy" / "energy.json").write_text(json.dumps({
    "schema_version": "orca_energy_product_v1", "workflow": "OrcaGradient",
    "energy_hartree": energy, "energy_unit": "hartree",
    "method": request.get("method"), "basis": request.get("basis"),
    "charge": request.get("charge"), "multiplicity": request.get("multiplicity")}),
    encoding="utf-8")
if not missing_product:
    (result / "gradient" / "gradient.json").write_text(json.dumps({
        "schema_version": "orca_gradient_product_v1",
        "request_sha256": "stub", "workflow": "OrcaGradient",
        "gradient_hartree_per_bohr": gradient,
        "gradient_hartree_per_angstrom": gradient_ang,
        "gradient_unit": "hartree/bohr",
        "gradient_convention": "energy_gradient_dE_dX",
        "symbols": elements, "energy_hartree": energy,
        "method": request.get("method"), "basis": request.get("basis"),
        "charge": request.get("charge"),
        "multiplicity": request.get("multiplicity"),
        "gradient_source": "stub"}), encoding="utf-8")

def product(pid, rel):
    target = result / rel
    if not target.is_file():
        return None
    data = target.read_bytes()
    return {"id": pid, "label": pid, "path": rel, "kind": "file",
            "metadata": {"sha256": hashlib.sha256(data).hexdigest(),
                         "size_bytes": len(data)}}

products = [p for p in [
    product("gradient", "gradient/gradient.json"),
    product("geometry", "geometry/geometry.xyz"),
    product("energy", "energy/energy.json")] if p is not None]
manifest = {"version": 2, "task_id": "", "workflow": "OrcaGradient",
            "status": "completed", "products": products}
if bad_manifest:
    manifest["workflow"] = "PESsearch"
(result / "result_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
'''


def _fake_acp(tmp_path: Path) -> Path:
    root = tmp_path / "fake_acp"
    package = root / "src" / "acp"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "cli.py").write_text(_FAKE_CLI, encoding="utf-8")
    return root


def _request(**overrides):
    kwargs = {
        "geometry": GEOMETRY,
        "elements": ELEMENTS,
        "method": "GFN2-xTB",
        "basis": "",
        "charge": 0,
        "multiplicity": 1,
        "route_extras": ["EnGrad"],
        "timeout_seconds": 120,
        "nproc": 2,
        "extra_blocks": ['%xtb\\n XTBINPUTSTRING "--iterations 2000"\\nend'],
        "scf_convergence": "tight",
        "output_name": "grad",
    }
    kwargs.update(overrides)
    return build_gradient_request(
        geometry=kwargs["geometry"], elements=kwargs["elements"],
        method=kwargs["method"], basis=kwargs["basis"], charge=kwargs["charge"],
        multiplicity=kwargs["multiplicity"], route_extras=kwargs["route_extras"],
        timeout_seconds=kwargs["timeout_seconds"], nproc=kwargs["nproc"],
        extra_blocks=kwargs["extra_blocks"], scf_convergence=kwargs["scf_convergence"],
        output_name=kwargs["output_name"])


def _run(root: Path, output_root: Path, request, *, attempt_id="attempt-001",
         execution_id="execution-001", timeout_seconds=10.0, extra_args=(),
         acp_config_path=None):
    return run_orca_gradient_attempt(
        acp_root=root, python_executable=sys.executable,
        acp_config_path=acp_config_path, request=request,
        output_root=output_root, execution_id=execution_id,
        attempt_id=attempt_id, timeout_seconds=timeout_seconds,
        extra_args=extra_args)


def _work(output_root: Path) -> Path:
    return output_root / "WORK" / "pes2ts"


# ---------------------------------------------------------------------------
# Request builder.
# ---------------------------------------------------------------------------
def test_build_and_validate_gradient_request_roundtrip() -> None:
    request = _request()
    assert validate_gradient_request(request) == request
    assert request["schema_version"] == "pes2ts_orca_gradient_request_v1"
    assert request["route_extras"] == ["EnGrad"]
    assert request_sha256(request) == request_sha256(dict(request))


@pytest.mark.parametrize("overrides, reason", [
    ({"method": ""}, "method"),
    ({"multiplicity": 0}, "multiplicity"),
    ({"charge": True}, "charge"),
    ({"basis": None}, "basis"),
    ({"geometry": []}, "geometry"),
])
def test_validate_gradient_request_rejects_bad_payloads(overrides, reason) -> None:
    base = {
        "schema_version": "pes2ts_orca_gradient_request_v1",
        "geometry": GEOMETRY, "elements": ELEMENTS, "method": "GFN2-xTB",
        "basis": "", "charge": 0, "multiplicity": 1,
    }
    base.update(overrides)
    with pytest.raises(ValueError, match=reason):
        validate_gradient_request(base)


def test_build_gradient_request_rejects_unaligned_elements() -> None:
    with pytest.raises(ValueError, match="aligned"):
        build_gradient_request(geometry=GEOMETRY, elements=["H"], method="m",
                               basis="", charge=0, multiplicity=1)


# ---------------------------------------------------------------------------
# Transport success path.
# ---------------------------------------------------------------------------
def test_success_path_receipt_manifest_and_evidence(tmp_path: Path) -> None:
    root = _fake_acp(tmp_path)
    request = _request()
    output_root = tmp_path / "attempt"
    result = _run(root, output_root, request)

    assert isinstance(result, OrcaGradientAttemptResult)
    assert result.status == "completed"
    assert result.returncode == 0
    assert result.timed_out is False
    assert result.error is None
    assert result.reused is False
    assert result.gradient_product is not None
    assert result.gradient_product["schema_version"] == "orca_gradient_product_v1"
    assert Path(result.manifest_path).is_file()
    assert Path(result.gradient_path).is_file()

    receipt = json.loads((_work(output_root) / "cli_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "completed"
    assert receipt["workflow"] == ACP_WORKFLOW_NAME
    assert receipt["request_sha256"] == request_sha256(request)

    evidence = parse_gradient_product_evidence(
        result.gradient_product, gradient_path=result.gradient_path,
        elements=ELEMENTS, geometry=GEOMETRY, energy=ENERGY)
    assert evidence["status"] == "bound"
    assert evidence["gradient_hartree_per_bohr"] == GRADIENT_BOHR
    assert evidence["energy_hartree"] == pytest.approx(ENERGY)

    verified = validate_orca_gradient_result(output_root, elements=ELEMENTS,
                                             geometry=GEOMETRY)
    assert verified["gradient_product"]["symbols"] == ELEMENTS


def test_attempt_reuse_returns_cached_receipt(tmp_path: Path) -> None:
    root = _fake_acp(tmp_path)
    request = _request()
    output_root = tmp_path / "attempt"
    first = _run(root, output_root, request)
    second = _run(root, output_root, request)
    assert first.status == "completed"
    assert second.status == "completed"
    assert second.reused is True
    assert second.manifest_sha256 == first.manifest_sha256 if hasattr(first, "manifest_sha256") else True
    assert second.request_sha256 == first.request_sha256


def test_attempt_id_rejects_request_mismatch(tmp_path: Path) -> None:
    root = _fake_acp(tmp_path)
    output_root = tmp_path / "attempt"
    _run(root, output_root, _request(), attempt_id="attempt-001")
    other = _request(method="HF-3c", basis="3c")
    with pytest.raises(ACPCLIError, match="different execution or request"):
        _run(root, output_root, other, attempt_id="attempt-001")


# ---------------------------------------------------------------------------
# Transport failure paths.
# ---------------------------------------------------------------------------
def test_nonzero_exit_is_typed_failure(tmp_path: Path) -> None:
    root = _fake_acp(tmp_path)
    result = _run(root, tmp_path / "attempt", _request(),
                  extra_args=("--test-exit-code", "1"))
    assert result.status == "failed"
    assert result.returncode == 1
    assert result.gradient_product is None
    assert "return code 1" in (result.error or "")


def test_timeout_kills_process_and_marks_timed_out(tmp_path: Path) -> None:
    root = _fake_acp(tmp_path)
    result = _run(root, tmp_path / "attempt", _request(), timeout_seconds=0.4,
                  extra_args=("--test-delay", "5"))
    assert result.status == "failed"
    assert result.timed_out is True
    assert "timeout" in (result.error or "").lower()


def test_bad_manifest_workflow_fails_completion(tmp_path: Path) -> None:
    root = _fake_acp(tmp_path)
    result = _run(root, tmp_path / "attempt", _request(),
                  extra_args=("--test-bad-manifest",))
    assert result.status == "failed"
    assert "incomplete" in (result.error or "")


def test_missing_gradient_product_fails_completion(tmp_path: Path) -> None:
    root = _fake_acp(tmp_path)
    result = _run(root, tmp_path / "attempt", _request(),
                  extra_args=("--test-missing-product",))
    assert result.status == "failed"
    assert "incomplete" in (result.error or "")


def test_bad_gradient_product_fails_completion(tmp_path: Path) -> None:
    root = _fake_acp(tmp_path)
    result = _run(root, tmp_path / "attempt", _request(),
                  extra_args=("--test-bad-gradient",))
    assert result.status == "failed"


def test_symbol_mismatch_fails_completion(tmp_path: Path) -> None:
    root = _fake_acp(tmp_path)
    result = _run(root, tmp_path / "attempt", _request(),
                  extra_args=("--test-symbol-mismatch",))
    assert result.status == "failed"
    assert "symbols" in (result.error or "")


def test_missing_acp_root_raises(tmp_path: Path) -> None:
    with pytest.raises(ACPCLIError, match="ACP checkout root"):
        run_orca_gradient_attempt(
            acp_root=tmp_path / "does-not-exist", python_executable=sys.executable,
            acp_config_path=None, request=_request(), output_root=tmp_path / "out",
            execution_id="e", attempt_id="a", timeout_seconds=5.0)


# ---------------------------------------------------------------------------
# Result validator.
# ---------------------------------------------------------------------------
def _write_valid_result_tree(root: Path) -> Path:
    output = root / "attempt"
    result = output / "RESULT"
    (result / "gradient").mkdir(parents=True)
    (result / "geometry").mkdir(parents=True)
    (result / "energy").mkdir(parents=True)
    gradient_path = result / "gradient" / "gradient.json"
    gradient_doc = {
        "schema_version": "orca_gradient_product_v1",
        "gradient_hartree_per_bohr": GRADIENT_BOHR,
        "gradient_hartree_per_angstrom": [
            [v / BOHR_ANGSTROM for v in row] for row in GRADIENT_BOHR],
        "symbols": ELEMENTS, "energy_hartree": ENERGY,
        "gradient_unit": "hartree/bohr",
    }
    gradient_path.write_text(json.dumps(gradient_doc), encoding="utf-8")
    (result / "geometry" / "geometry.xyz").write_text(
        "2\nstub\nH 0 0 0\nH 0.74 0 0\n", encoding="utf-8")
    (result / "energy" / "energy.json").write_text(
        json.dumps({"schema_version": "orca_energy_product_v1",
                    "energy_hartree": ENERGY}), encoding="utf-8")
    products = []
    for pid, rel in (("gradient", "gradient/gradient.json"),
                     ("geometry", "geometry/geometry.xyz"),
                     ("energy", "energy/energy.json")):
        data = (result / rel).read_bytes()
        products.append({"id": pid, "label": pid, "path": rel, "kind": "file",
                         "metadata": {"sha256": hashlib.sha256(data).hexdigest()}})
    manifest = {"version": 2, "task_id": "", "workflow": "OrcaGradient",
                "status": "completed", "products": products}
    (result / "result_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return output


def test_validator_accepts_well_formed_result(tmp_path: Path) -> None:
    output = _write_valid_result_tree(tmp_path)
    verified = validate_orca_gradient_result(output, elements=ELEMENTS,
                                             geometry=GEOMETRY)
    assert verified["manifest"]["workflow"] == "OrcaGradient"
    assert len(verified["products"]) == 3


def test_validator_rejects_hash_mismatch(tmp_path: Path) -> None:
    output = _write_valid_result_tree(tmp_path)
    manifest_path = output / "RESULT" / "result_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for product in manifest["products"]:
        if product["id"] == "gradient":
            product["metadata"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ACPCLIError, match="SHA256"):
        validate_orca_gradient_result(output)


def test_validator_rejects_inconsistent_angstrom_conversion(tmp_path: Path) -> None:
    output = _write_valid_result_tree(tmp_path)
    gradient_path = output / "RESULT" / "gradient" / "gradient.json"
    doc = json.loads(gradient_path.read_text(encoding="utf-8"))
    doc["gradient_hartree_per_angstrom"] = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    gradient_path.write_text(json.dumps(doc), encoding="utf-8")
    # Keep the manifest product hash honest so the validator reaches the
    # conversion check rather than failing earlier on SHA256.
    manifest_path = output / "RESULT" / "result_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for product in manifest["products"]:
        if product["id"] == "gradient":
            data = gradient_path.read_bytes()
            product["metadata"]["sha256"] = hashlib.sha256(data).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ACPCLIError, match="inconsistent"):
        validate_orca_gradient_result(output)


def test_parse_evidence_missing_and_rejected() -> None:
    assert parse_gradient_product_evidence(None, gradient_path=None,
                                           elements=ELEMENTS, geometry=GEOMETRY,
                                           energy=None)["status"] == "missing"
    bad = {"schema_version": "orca_gradient_product_v1",
           "gradient_hartree_per_bohr": [[0.0, 0.0, 0.0]],
           "symbols": ELEMENTS, "energy_hartree": ENERGY}
    evidence = parse_gradient_product_evidence(
        bad, gradient_path="/nonexistent/gradient.json", elements=ELEMENTS,
        geometry=GEOMETRY, energy=ENERGY)
    assert evidence["status"] in {"missing", "rejected"}


# ---------------------------------------------------------------------------
# Backend public shape.
# ---------------------------------------------------------------------------
def test_orca_gradient_backend_success_record_shape(tmp_path: Path, monkeypatch) -> None:
    root = _fake_acp(tmp_path)
    backend = ORCAGradientBackend(
        charge=0, multiplicity=1, elements=ELEMENTS,
        folder=tmp_path / "evaluations", method="GFN2-xTB", basis="",
        acp_root=root, acp_python=sys.executable)
    record = backend(GEOMETRY, "trial-0000/eval-0011223344556677")
    assert record["success"] is True
    assert record["failure_class"] is None
    assert record["energy"] == pytest.approx(ENERGY)
    assert record["gradient_evidence"]["status"] == "bound"
    assert record["gradient_hartree_per_angstrom"] is not None
    assert len(record["gradient_hartree_per_angstrom"]) == 2
    assert record["acp"]["workflow"] == "OrcaGradient"
    assert record["request_sha256"]
    assert record["evaluation_identity"] == {
        "scheme": "pes2ts_execution_identity_v1",
        "evaluation_id": "trial-0000/eval-0011223344556677",
        "trial": "trial-0000", "content_address": "eval-0011223344556677",
        "content_addressed": True}
    receipt = tmp_path / "evaluations" / "trial-0000" / "eval-0011223344556677" / "result.json"
    assert receipt.is_file()
    cached = backend(GEOMETRY, "trial-0000/eval-0011223344556677")
    assert cached["request_sha256"] == record["request_sha256"]
    assert cached["evaluation_identity"]["content_address"] == "eval-0011223344556677"


def test_orca_gradient_backend_rejects_changed_cached_input(tmp_path: Path) -> None:
    root = _fake_acp(tmp_path)
    backend = ORCAGradientBackend(
        charge=0, multiplicity=1, elements=ELEMENTS,
        folder=tmp_path / "evaluations", acp_root=root,
        acp_python=sys.executable)
    backend(GEOMETRY, "trial-0000/eval-0011223344556677")
    backend.elements = ["H", "He"]
    with pytest.raises(ValueError, match="IDENTITY_CONFLICT"):
        backend(GEOMETRY, "trial-0000/eval-0011223344556677")


def test_orca_gradient_backend_identity_conflict_reports_both_digests(tmp_path: Path) -> None:
    root = _fake_acp(tmp_path)
    backend = ORCAGradientBackend(
        charge=0, multiplicity=1, elements=ELEMENTS,
        folder=tmp_path / "evaluations", acp_root=root,
        acp_python=sys.executable)
    stored = backend(GEOMETRY, "trial-0000/eval-0011223344556677")
    backend.method = "HF-3c"
    with pytest.raises(ValueError, match="IDENTITY_CONFLICT") as excinfo:
        backend(GEOMETRY, "trial-0000/eval-0011223344556677")
    message = str(excinfo.value)
    assert stored["request_sha256"] in message
    assert "CACHED_GRADIENT_INPUT_MISMATCH" in message


def test_orca_gradient_backend_content_address_separates_changed_geometry(tmp_path: Path) -> None:
    root = _fake_acp(tmp_path)
    backend = ORCAGradientBackend(
        charge=0, multiplicity=1, elements=ELEMENTS,
        folder=tmp_path / "evaluations", acp_root=root,
        acp_python=sys.executable)
    first = backend(GEOMETRY, "trial-0000/eval-0011223344556677")
    moved = [[row[0] + 0.25, row[1], row[2]] for row in GEOMETRY]
    second = backend(moved, "trial-0000/eval-9988776655443322")
    assert second["success"] is True
    assert second["request_sha256"] != first["request_sha256"]
    assert (tmp_path / "evaluations" / "trial-0000" / "eval-0011223344556677" / "result.json").is_file()
    assert (tmp_path / "evaluations" / "trial-0000" / "eval-9988776655443322" / "result.json").is_file()


@pytest.mark.parametrize("bad_id", ["gradient-0000", "trial-0000", "trial-0000/eval-1/extra",
                                    "trial-0000/eval-bad!addr", "../escape/eval-0000000000000000"])
def test_orca_gradient_backend_rejects_non_content_addressed_ids(tmp_path: Path, bad_id) -> None:
    backend = ORCAGradientBackend(
        charge=0, multiplicity=1, elements=ELEMENTS,
        folder=tmp_path / "evaluations")
    with pytest.raises(ValueError, match="INVALID_EVALUATION_ID"):
        backend(GEOMETRY, bad_id)


def test_orca_gradient_backend_requires_acp_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("PES2TS_ACP_ROOT", raising=False)
    backend = ORCAGradientBackend(
        charge=0, multiplicity=1, elements=ELEMENTS,
        folder=tmp_path / "evaluations")
    with pytest.raises(ACPCLIError, match="ACP checkout root"):
        backend(GEOMETRY, "trial-0000/eval-0011223344556677")


def test_resolve_acp_wiring_precedence(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PES2TS_ACP_ROOT", "/env/root")
    monkeypatch.setenv("PES2TS_ACP_PYTHON", "/env/python")
    monkeypatch.setenv("PES2TS_ACP_CONFIG", "/env/config.yaml")
    root, python, config = resolve_acp_wiring()
    assert (root, python, config) == ("/env/root", "/env/python", "/env/config.yaml")
    root, python, config = resolve_acp_wiring(
        acp_root="/explicit/root", acp_wiring={"python": "/wiring/python"})
    assert root == "/explicit/root"
    assert python == "/wiring/python"
    assert config == "/env/config.yaml"


def test_write_json_atomic_rejects_nan(tmp_path: Path) -> None:
    path = tmp_path / "doc.json"
    write_json(path, {"ok": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"ok": 1}
    with pytest.raises(ValueError):
        write_json(path, {"bad": math.nan})


# ---------------------------------------------------------------------------
# Truth guard integration.
# ---------------------------------------------------------------------------
def test_orca_gradient_transport_is_allowlisted_and_clean() -> None:
    assert "pes2ts_core/integration/acp/orca_gradient_transport.py" in SUBPROCESS_IMPORT_ALLOWLIST
    package_root = Path(__file__).resolve().parents[1] / "pes2ts_core"
    findings = [
        finding for finding in scan_truth_access(package_root=package_root)
        if "orca_gradient" in finding.file
    ]
    assert findings == []
