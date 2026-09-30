"""Process and immutable-attempt tests for the ACP validation stage runner."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from pes2ts_core.integration.acp.cli_backend import ACPCLIError
from pes2ts_core.integration.acp.stage_cli import ACPStageResult, ACPValidationCLIBackend
from pes2ts_core.demo import synthetic_objects
from pes2ts_core.contracts import seal_document
from pes2ts_core.ranking import rank_path_bundle


def _fake_acp(tmp_path: Path) -> Path:
    root = tmp_path / "fake_acp"
    package = root / "src" / "acp"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "cli.py").write_text(
        "import json, pathlib, sys, time\n"
        "args = sys.argv[1:]\n"
        "if '--sleep' in args: time.sleep(10)\n"
        "workflow = args[1]\n"
        "out = pathlib.Path(args[args.index('--output') + 1])\n"
        "result = out / 'RESULT'\n"
        "result.mkdir(parents=True, exist_ok=True)\n"
        "(result / 'result_manifest.json').write_text(json.dumps({"
        "'version': 2, 'workflow': workflow, 'status': 'completed', 'products': []}))\n",
        encoding="utf-8")
    return root


def _backend(root: Path) -> ACPValidationCLIBackend:
    return ACPValidationCLIBackend(acp_root=root, python_executable=sys.executable)


def test_stage_cli_records_completion_and_recovers_same_attempt(tmp_path: Path) -> None:
    backend = _backend(_fake_acp(tmp_path))
    task_root = tmp_path / "attempt"
    first = backend._run_stage(workflow="BatchOptimize", execution_id="exec-1",
        attempt_id="attempt-1", task_root=task_root, request={"method": "B3LYP"},
        args=["BatchOptimize", "--output", str(task_root)], timeout_seconds=10)
    assert first.status == "completed"
    assert first.manifest_sha256
    resumed = backend._run_stage(workflow="BatchOptimize", execution_id="exec-1",
        attempt_id="attempt-1", task_root=task_root, request={"method": "B3LYP"},
        args=["BatchOptimize", "--output", str(task_root)], timeout_seconds=10)
    assert resumed.reused is True
    assert resumed.manifest_sha256 == first.manifest_sha256
    with pytest.raises(ACPCLIError, match="different request"):
        backend._run_stage(workflow="BatchOptimize", execution_id="exec-1",
            attempt_id="attempt-1", task_root=task_root, request={"method": "HF"},
            args=["BatchOptimize", "--output", str(task_root)], timeout_seconds=10)


def test_stage_cli_timeout_is_terminal_and_leaves_receipt(tmp_path: Path) -> None:
    backend = _backend(_fake_acp(tmp_path))
    task_root = tmp_path / "timeout"
    result = backend._run_stage(workflow="irc", execution_id="exec-2",
        attempt_id="attempt-2", task_root=task_root, request={"direction": "both"},
        args=["irc", "--sleep", "--output", str(task_root)], timeout_seconds=0.1)
    assert result.status == "failed"
    receipt = json.loads((task_root / "WORK/pes2ts/stage_cli_receipt.json").read_text())
    assert "exceeded timeout" in receipt["error"]
    with pytest.raises(ACPCLIError, match="terminal/interrupted"):
        backend._run_stage(workflow="irc", execution_id="exec-2",
            attempt_id="attempt-2", task_root=task_root, request={"direction": "both"},
            args=["irc", "--sleep", "--output", str(task_root)], timeout_seconds=10)


def test_failed_batch_attempt_still_assembles_failed_validation_result() -> None:
    docs = synthetic_objects()
    path = seal_document({**docs["PathBundle"], "status": "usable"})
    proposal = rank_path_bundle(path, rule="highest_scan_energy", top_k=3)
    failed = ACPStageResult("BatchOptimize", "exec-batch", "attempt-batch", "failed",
        2, "acp-output", "a" * 64, None, 1.2, error="fake failure")
    outcome = ACPValidationCLIBackend._failed_validation(validation_id="validation:failed",
        path=path, proposal=proposal, batch_result=failed, note="simulated batch failure")
    assert outcome["validation_result"]["status"] == "failed"
    assert outcome["validation_result"]["optts"]["attempt_id"] == "attempt-batch"
    assert outcome["validation_result"]["frequency"]["status"] == "not_run"
