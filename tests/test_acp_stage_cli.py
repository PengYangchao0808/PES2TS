"""Process and immutable-attempt tests for the ACP validation stage runner."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from pes2ts_core.integration.acp.cli_backend import ACPCLIError
from pes2ts_core.integration.acp.stage_cli import ACPStageResult, ACPValidationCLIBackend
from pes2ts_core.demo import synthetic_objects
from pes2ts_core.contracts import make_document, seal_document
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


def _reviewed_case_and_review():
    """Case + accepted ReviewRecord pair satisfying the stage_cli binding."""
    docs = synthetic_objects()
    case = docs["ReactionCase"]
    bound_case = seal_document({**case, "source": {
        **case["source"],
        "review_record_id": "review:unit-gradient-binding",
        "reviewed_from_case_sha256": case["content_sha256"]}})
    reviewer = {"reviewer": "", "decision": "accept",
                "dimensions": {"reaction_center": "confirmed", "atom_mapping": "confirmed",
                               "charge_spin": "confirmed", "geometry_assembly": "confirmed"},
                "scan_feasibility": "1D",
                "multiplicities": {"reactant": 1, "product": 1}}
    review = make_document("ReviewRecord", "review:unit-gradient-binding", "accepted",
        dataset_version=case["dataset_version"], reaction_id=case["reaction_id"],
        case_id=case["case_id"], split=case["split"],
        case_sha256=case["content_sha256"], workbook_sha256="0"*64,
        reviewers=[{**reviewer, "reviewer": "reviewer-one"},
                   {**reviewer, "reviewer": "reviewer-two"}],
        adjudication={"adjudicator": "adjudicator", "decision": "accepted",
                      "reactant_multiplicity": 1, "product_multiplicity": 1},
        decision_source="human",
        extensions={"pes2ts.review_output.v1": {
            "reviewed_case_sha256": bound_case["content_sha256"]}})
    return bound_case, review


def test_validation_attempt_binds_proposal_geometry_frame_and_plan(tmp_path, monkeypatch) -> None:
    docs = synthetic_objects()
    path = seal_document({**docs["PathBundle"], "status": "usable"})
    base = rank_path_bundle(path, rule="highest_scan_energy", top_k=3)
    selected = [dict(base["selected_frames"][0])]
    selected[0] = {**selected[0], "plan_sha256": "p"*64,
                   "completed_interval": False,
                   "reference_geometry_used": False,
                   "stationary_point_verified": False,
                   "reaction_connection_verified": False}
    proposal = seal_document({**base, "selected_frames": selected})
    case, review = _reviewed_case_and_review()
    backend = _backend(_fake_acp(tmp_path))
    captured = {}

    def capture(*, request, **kwargs):
        captured.update(request)
        return ACPStageResult("BatchOptimize", kwargs["execution_id"],
            kwargs["attempt_id"], "failed", 1, str(kwargs["task_root"]),
            "b"*64, None, .1, error="captured before execution")

    monkeypatch.setattr(backend, "_run_stage", capture)
    backend.run_validation(case=case, review_record=review, path=path, proposal=proposal,
        source_frame_id=selected[0]["frame_id"], validation_id="validation:binding",
        expected_method="HF-3c", expected_basis="", output_root=tmp_path/"out",
        batch_execution_id="exec-b", batch_attempt_id="attempt-b",
        irc_execution_id="exec-i", irc_attempt_id="attempt-i",
        batch_timeout_seconds=10, irc_timeout_seconds=10)
    assert captured["proposal_geometry_sha256"] == selected[0]["geometry_sha256"]
    assert captured["proposal_frame_id"] == selected[0]["frame_id"]
    assert captured["proposal_plan_sha256"] == "p"*64
    assert "ranking" not in json.dumps(captured)
