"""Fake-ACP transport tests for the ACP ``XtbPathSearch`` CLI runner.

Mirrors the fake-ACP-process pattern of ``test_acp_cli_backend.py``: a stub
``src/acp/cli.py`` stands in for the real ACP engine and writes a minimal
valid ``RESULT/`` tree (v2 manifest + ``pes_profile_v2`` + path frames + raw
trajectory).  No real xTB binary is ever invoked.

Covers the success path (receipt, manifest validation, typed result fields,
attempt reuse), non-zero exit, malformed manifest, timeout kill, claim
conflict, attempt-ID/request binding, ACP-root validation, the manifest
validator's unsafe-path/hash/profile rejections, and the ``--register``
contract (flag present when configured, absent when disabled, and the typed
failure when registration fails after a valid result).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

from pes2ts_core.integration.acp.cli_backend import ACPCLIError
from pes2ts_core.integration.acp.xtb_path_request import (
    ACP_WORKFLOW_NAME,
    build_path_request,
)
from pes2ts_core.integration.acp.xtb_path_transport import (
    XtbPathAttemptResult,
    run_xtb_path_attempt,
    validate_xtb_path_result,
)

START_XYZ = "2\nPES2TS start\nH 0.000000 0.000000 0.000000\nH 0.740000 0.000000 0.000000\n"
END_XYZ = "2\nPES2TS end\nH 0.000000 0.000000 0.000000\nH 1.500000 0.000000 0.000000\n"

#: Stub ACP CLI. Test hooks arrive as extra argv tokens the transport passes
#: through verbatim: ``--test-delay S``, ``--test-exit-code N``,
#: ``--test-bad-manifest``, ``--test-register-fail`` (write a valid RESULT
#: then exit 1, mimicking ACP's post-workflow registration failure).  A real
#: ACP never sees these.
_FAKE_CLI = '''\
import hashlib, json, pathlib, sys, time

args = sys.argv[1:]
delay = 0.0
exit_code = 0
bad_manifest = False
register_fail = False
i = 0
while i < len(args):
    if args[i] == "--test-delay":
        delay = float(args[i + 1]); i += 2; continue
    if args[i] == "--test-exit-code":
        exit_code = int(args[i + 1]); i += 2; continue
    if args[i] == "--test-bad-manifest":
        bad_manifest = True; i += 1; continue
    if args[i] == "--test-register-fail":
        register_fail = True; i += 1; continue
    i += 1
request_path = pathlib.Path(args[args.index("--path-config") + 1])
output = pathlib.Path(args[args.index("--output") + 1])
request = json.loads(request_path.read_text(encoding="utf-8"))
assert request.get("schema_version") == "pes2ts_xtb_path_request_v1"
time.sleep(delay)
if exit_code:
    raise SystemExit(exit_code)
result = output / "RESULT" / "pes_search"
frames_dir = result / "path_frames"
frames_dir.mkdir(parents=True, exist_ok=True)
profile = {"schema_version": "pes_profile_v2", "workflow": "XtbPathSearch",
           "status": "completed", "frames": [
               {"index": 0, "energy_rel_kcal": 0.0},
               {"index": 1, "energy_rel_kcal": 2.5},
               {"index": 2, "energy_rel_kcal": 1.0}]}
(result / "pes_profile.json").write_text(json.dumps(profile), encoding="utf-8")
xyz = ("3\\nfake path frame\\n"
       "H 0.00000000 0.00000000 0.00000000\\n"
       "H 0.74000000 0.00000000 0.00000000\\n"
       "H 0.00000000 0.74000000 0.00000000\\n")
for index in range(3):
    (frames_dir / ("path_frame_%03d.xyz" % index)).write_text(xyz, encoding="utf-8")
(result / "xtbpath.xyz").write_text(xyz * 3, encoding="utf-8")

def product(pid, rel):
    target = output / "RESULT" / rel
    data = target.read_bytes()
    return {"id": pid, "label": pid, "path": rel, "kind": "file",
            "metadata": {"sha256": hashlib.sha256(data).hexdigest(),
                         "size_bytes": len(data)}}

products = [product("pes_profile", "pes_search/pes_profile.json"),
            product("raw_trajectory", "pes_search/xtbpath.xyz"),
            product("path_frame_000", "pes_search/path_frames/path_frame_000.xyz")]
manifest = {"version": 2, "task_id": "", "workflow": "XtbPathSearch",
            "status": "completed", "products": products}
if bad_manifest:
    manifest["workflow"] = "PESsearch"
(output / "RESULT" / "result_manifest.json").write_text(
    json.dumps(manifest), encoding="utf-8")
if register_fail:
    raise SystemExit(1)
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
        "reaction_id": "RXN_0000000001",
        "start_xyz_text": START_XYZ,
        "end_xyz_text": END_XYZ,
        "charge": 0,
        "multiplicity": 1,
        "path_config": {},
    }
    kwargs.update(overrides)
    return build_path_request(**kwargs)


def _run(root: Path, output_root: Path, request, *, attempt_id="attempt-001",
         execution_id="execution-001", timeout_seconds=10.0, extra_args=(),
         acp_config_path=None, register=False):
    return run_xtb_path_attempt(
        acp_root=root, python_executable=sys.executable,
        acp_config_path=acp_config_path, request=request,
        output_root=output_root, execution_id=execution_id,
        attempt_id=attempt_id, timeout_seconds=timeout_seconds,
        extra_args=extra_args, register=register)


def _work(output_root: Path) -> Path:
    return output_root / "WORK" / "pes2ts"


# ---------------------------------------------------------------------------
# Success path.
# ---------------------------------------------------------------------------
def test_success_path_receipt_manifest_and_result_fields(tmp_path: Path) -> None:
    root = _fake_acp(tmp_path)
    request = _request()
    output_root = tmp_path / "attempt"
    result = _run(root, output_root, request)

    assert isinstance(result, XtbPathAttemptResult)
    assert result.status == "completed"
    assert result.returncode == 0
    assert result.timed_out is False
    assert result.error is None
    assert result.reused is False
    assert result.request_sha256 == request["provenance"]["request_sha256"]
    assert result.attempt_dir == str(output_root.resolve())
    assert Path(result.manifest_path).is_file()
    assert Path(result.profile_path).is_file()
    assert Path(result.frames_dir).is_dir()
    assert Path(result.raw_trajectory_path).is_file()
    assert Path(result.frames_dir, "path_frame_000.xyz").is_file()

    work = _work(output_root)
    receipt = json.loads((work / "cli_receipt.json").read_text(encoding="utf-8"))
    assert receipt["execution_id"] == "execution-001"
    assert receipt["attempt_id"] == "attempt-001"
    assert receipt["status"] == "completed"
    assert receipt["returncode"] == 0
    assert receipt["timed_out"] is False
    assert receipt["request_sha256"] == request["provenance"]["request_sha256"]
    assert receipt["command"][1:6] == [
        "-m", "acp.cli", "run", ACP_WORKFLOW_NAME, "--path-config"]
    assert receipt["command"][-2:] == ["--output", str(output_root.resolve())]
    path_config = work / "path_config.json"
    assert path_config.is_file()
    assert receipt["command"][6] == str(path_config)
    assert (work / "acp_cli.log").is_file()
    assert not (work / "cli_attempt.claim").exists()
    on_disk = json.loads(path_config.read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == "pes2ts_xtb_path_request_v1"

    verified = validate_xtb_path_result(output_root)
    assert verified["manifest"]["workflow"] == ACP_WORKFLOW_NAME
    assert verified["manifest_sha256"]


def test_completed_attempt_is_reused_with_same_receipt(tmp_path: Path) -> None:
    root = _fake_acp(tmp_path)
    request = _request()
    output_root = tmp_path / "attempt"
    first = _run(root, output_root, request)
    second = _run(root, output_root, request)
    assert first.status == "completed"
    assert second.status == "completed"
    assert second.reused is True
    assert second.request_sha256 == first.request_sha256
    assert second.manifest_path == first.manifest_path


def test_config_flag_and_extra_args_reach_the_command(tmp_path: Path) -> None:
    root = _fake_acp(tmp_path)
    config_path = tmp_path / "acp_config.yaml"
    config_path.write_text("acp: {}\n", encoding="utf-8")
    output_root = tmp_path / "attempt"
    result = _run(root, output_root, _request(), acp_config_path=config_path,
                  extra_args=("--nproc", "2"))
    assert result.status == "completed"
    receipt = json.loads((_work(output_root) / "cli_receipt.json").read_text(encoding="utf-8"))
    assert receipt["command"][-4:] == ["--config", str(config_path.resolve()),
                                       "--nproc", "2"]


# ---------------------------------------------------------------------------
# --register contract (X5'-A).
# ---------------------------------------------------------------------------
def test_register_flag_appended_when_configured(tmp_path: Path) -> None:
    root = _fake_acp(tmp_path)
    output_root = tmp_path / "attempt"
    result = _run(root, output_root, _request(), register=True)
    assert result.status == "completed"
    receipt = json.loads((_work(output_root) / "cli_receipt.json").read_text(encoding="utf-8"))
    assert receipt["register"] is True
    assert "--register" in receipt["command"]
    assert receipt["command"].index("--register") > receipt["command"].index("--output")


def test_register_flag_absent_when_disabled(tmp_path: Path) -> None:
    root = _fake_acp(tmp_path)
    output_root = tmp_path / "attempt"
    result = _run(root, output_root, _request(), register=False)
    assert result.status == "completed"
    receipt = json.loads((_work(output_root) / "cli_receipt.json").read_text(encoding="utf-8"))
    assert receipt["register"] is False
    assert "--register" not in receipt["command"]


def test_register_failure_after_valid_result_fails_loudly(tmp_path: Path) -> None:
    root = _fake_acp(tmp_path)
    output_root = tmp_path / "attempt"
    result = _run(root, output_root, _request(), register=True,
                  extra_args=("--test-register-fail",))
    assert result.status == "failed"
    assert result.returncode == 1
    assert result.error is not None
    assert "--register" in result.error
    assert "ACP_RUN_ROOT" in result.error
    assert ACP_WORKFLOW_NAME in result.error
    assert result.manifest_path is None
    receipt = json.loads((_work(output_root) / "cli_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["register"] is True
    assert "--register" in (receipt.get("error") or "")
    verified = validate_xtb_path_result(output_root)
    assert verified["manifest"]["workflow"] == ACP_WORKFLOW_NAME


def test_register_workflow_failure_keeps_generic_exit_error(tmp_path: Path) -> None:
    root = _fake_acp(tmp_path)
    output_root = tmp_path / "attempt"
    result = _run(root, output_root, _request(), register=True,
                  extra_args=("--test-exit-code", "9"))
    assert result.status == "failed"
    assert result.returncode == 9
    assert result.error == "ACP CLI exited with return code 9"


def test_register_must_be_boolean(tmp_path: Path) -> None:
    root = _fake_acp(tmp_path)
    with pytest.raises(ACPCLIError, match="register"):
        run_xtb_path_attempt(
            acp_root=root, python_executable=sys.executable,
            acp_config_path=None, request=_request(),
            output_root=tmp_path / "attempt", execution_id="execution-001",
            attempt_id="attempt-001", timeout_seconds=10.0, register="yes")


# ---------------------------------------------------------------------------
# Failure paths: never a silent partial success.
# ---------------------------------------------------------------------------
def test_nonzero_exit_is_a_typed_failure(tmp_path: Path) -> None:
    root = _fake_acp(tmp_path)
    output_root = tmp_path / "failed"
    result = _run(root, output_root, _request(), extra_args=("--test-exit-code", "9"))
    assert result.status == "failed"
    assert result.returncode == 9
    assert result.timed_out is False
    assert "return code 9" in result.error
    assert result.manifest_path is None
    assert result.profile_path is None
    receipt = json.loads((_work(output_root) / "cli_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["timed_out"] is False


def test_malformed_manifest_is_a_typed_failure(tmp_path: Path) -> None:
    root = _fake_acp(tmp_path)
    output_root = tmp_path / "bad-manifest"
    result = _run(root, output_root, _request(), extra_args=("--test-bad-manifest",))
    assert result.status == "failed"
    assert result.returncode == 0
    assert "result is incomplete" in result.error
    assert ACP_WORKFLOW_NAME in result.error or "manifest" in result.error
    receipt = json.loads((_work(output_root) / "cli_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["manifest_sha256"] is None


def test_timeout_kills_the_process_group(tmp_path: Path) -> None:
    root = _fake_acp(tmp_path)
    output_root = tmp_path / "timeout"
    result = _run(root, output_root, _request(), timeout_seconds=0.2,
                  extra_args=("--test-delay", "30"))
    assert result.status == "failed"
    assert result.timed_out is True
    assert "exceeded timeout" in result.error
    receipt = json.loads((_work(output_root) / "cli_receipt.json").read_text(encoding="utf-8"))
    assert receipt["timed_out"] is True
    assert receipt["status"] == "failed"
    assert not (_work(output_root) / "cli_attempt.claim").exists()


# ---------------------------------------------------------------------------
# Claim / attempt-binding / checkout validation.
# ---------------------------------------------------------------------------
def test_claim_file_blocks_duplicate_submission(tmp_path: Path) -> None:
    root = _fake_acp(tmp_path)
    output_root = tmp_path / "claimed"
    work = _work(output_root)
    work.mkdir(parents=True)
    (work / "cli_attempt.claim").write_text("pid=1\n", encoding="utf-8")
    with pytest.raises(ACPCLIError, match="already claimed"):
        _run(root, output_root, _request())


def test_attempt_id_rejects_a_different_request(tmp_path: Path) -> None:
    root = _fake_acp(tmp_path)
    output_root = tmp_path / "bound"
    _run(root, output_root, _request())
    with pytest.raises(ACPCLIError, match="different execution or request"):
        _run(root, output_root, _request(charge=1))


def test_terminal_attempt_requires_a_new_attempt_id(tmp_path: Path) -> None:
    root = _fake_acp(tmp_path)
    output_root = tmp_path / "terminal"
    _run(root, output_root, _request(), extra_args=("--test-exit-code", "3"))
    with pytest.raises(ACPCLIError, match="new attempt_id"):
        _run(root, output_root, _request(), extra_args=("--test-exit-code", "3"))


def test_acp_root_without_cli_is_rejected(tmp_path: Path) -> None:
    broken = tmp_path / "broken_acp"
    broken.mkdir()
    with pytest.raises(ACPCLIError, match="src/acp/cli.py"):
        _run(broken, tmp_path / "attempt", _request())


def test_malformed_request_is_rejected_before_launch(tmp_path: Path) -> None:
    root = _fake_acp(tmp_path)
    bad = _request()
    bad["source"]["start_xyz"] = ""
    with pytest.raises(ValueError):
        _run(root, tmp_path / "attempt", bad)


def test_portable_identifier_enforced(tmp_path: Path) -> None:
    root = _fake_acp(tmp_path)
    with pytest.raises(ACPCLIError, match="attempt_id"):
        _run(root, tmp_path / "attempt", _request(), attempt_id="bad id!")
    with pytest.raises(ACPCLIError, match="execution_id"):
        _run(root, tmp_path / "attempt", _request(), execution_id="")


# ---------------------------------------------------------------------------
# validate_xtb_path_result unit checks (no subprocess).
# ---------------------------------------------------------------------------
def _write_valid_result_tree(root: Path) -> Path:
    result = root / "RESULT" / "pes_search"
    frames_dir = result / "path_frames"
    frames_dir.mkdir(parents=True)
    profile = {"schema_version": "pes_profile_v2", "workflow": ACP_WORKFLOW_NAME,
               "status": "completed", "frames": [{"index": 0}]}
    (result / "pes_profile.json").write_text(json.dumps(profile), encoding="utf-8")
    xyz = "1\nframe\nH 0.0 0.0 0.0\n"
    (frames_dir / "path_frame_000.xyz").write_text(xyz, encoding="utf-8")
    (result / "xtbpath.xyz").write_text(xyz, encoding="utf-8")
    def product(pid, rel):
        target = root / "RESULT" / rel
        data = target.read_bytes()
        return {"id": pid, "path": rel, "kind": "file",
                "metadata": {"sha256": hashlib.sha256(data).hexdigest(),
                             "size_bytes": len(data)}}
    manifest = {"version": 2, "task_id": "t1", "workflow": ACP_WORKFLOW_NAME,
                "status": "completed",
                "products": [product("pes_profile", "pes_search/pes_profile.json"),
                             product("raw_trajectory", "pes_search/xtbpath.xyz"),
                             product("path_frame_000", "pes_search/path_frames/path_frame_000.xyz")]}
    (root / "RESULT" / "result_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_validator_accepts_a_complete_result_tree(tmp_path: Path) -> None:
    root = _write_valid_result_tree(tmp_path / "attempt")
    verified = validate_xtb_path_result(root)
    assert verified["manifest"]["status"] == "completed"
    assert len(verified["products"]) == 3
    assert verified["profile"]["schema_version"] == "pes_profile_v2"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda manifest: manifest.update(workflow="PESsearch"),
        lambda manifest: manifest.update(status="failed"),
        lambda manifest: manifest.update(version=1),
        lambda manifest: manifest.update(products="not-a-list"),
        lambda manifest: manifest["products"].append(
            {"id": "evil", "path": "../outside.xyz", "kind": "file"}),
        lambda manifest: manifest["products"].append(
            {"id": "evil", "path": "/etc/passwd", "kind": "file"}),
        lambda manifest: manifest["products"].append(
            {"id": "pes_profile", "path": "pes_search/pes_profile.json", "kind": "file"}),
    ],
)
def test_validator_rejects_malformed_manifests(tmp_path: Path, mutate) -> None:
    root = _write_valid_result_tree(tmp_path / "attempt")
    manifest_path = root / "RESULT" / "result_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ACPCLIError):
        validate_xtb_path_result(root)


def test_validator_rejects_sha_mismatch_and_missing_artifacts(tmp_path: Path) -> None:
    root = _write_valid_result_tree(tmp_path / "attempt")
    manifest_path = root / "RESULT" / "result_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["products"][0]["metadata"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ACPCLIError, match="SHA256"):
        validate_xtb_path_result(root)

    root2 = _write_valid_result_tree(tmp_path / "attempt2")
    (root2 / "RESULT" / "pes_search" / "xtbpath.xyz").unlink()
    with pytest.raises(ACPCLIError, match="xtbpath.xyz"):
        validate_xtb_path_result(root2)

    root3 = _write_valid_result_tree(tmp_path / "attempt3")
    bad_profile = b'{"schema_version": "pes_profile_v1"}'
    (root3 / "RESULT" / "pes_search" / "pes_profile.json").write_bytes(bad_profile)
    manifest3_path = root3 / "RESULT" / "result_manifest.json"
    manifest3 = json.loads(manifest3_path.read_text(encoding="utf-8"))
    for product in manifest3["products"]:
        if product["id"] == "pes_profile":
            product["metadata"]["sha256"] = hashlib.sha256(bad_profile).hexdigest()
            product["metadata"]["size_bytes"] = len(bad_profile)
    manifest3_path.write_text(json.dumps(manifest3), encoding="utf-8")
    with pytest.raises(ACPCLIError, match="pes_profile_v2"):
        validate_xtb_path_result(root3)

    root4 = _write_valid_result_tree(tmp_path / "attempt4")
    (root4 / "RESULT" / "result_manifest.json").unlink()
    with pytest.raises(ACPCLIError, match="result_manifest"):
        validate_xtb_path_result(root4)
