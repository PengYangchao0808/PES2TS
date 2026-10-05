"""G2-AB1 WP-1: execution identity namespaces never collide across attempts.

Covers the B1-1 failure: a replayed attempt (resume/retry/warm-start change)
reused the ``gradient-0000`` directory name for a different geometry and died
on ``CACHED_GRADIENT_INPUT_MISMATCH``.  Evaluation ids are now content
addressed (``trial-<NNNN>/eval-<content16>``), so replays with changed content
recompute under a fresh address, identical replays reuse receipts, and a
forged same-address/different-binding receipt stays a typed conflict.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from pes2ts_core.generation.planning.continuation import run_continuation
from pes2ts_core.generation.planning.local_corrector import LocalCorrectorPolicy
from pes2ts_core.integration.acp.gradient_backend import ORCALocalCorrector

from test_acp_orca_gradient_transport import _fake_acp
from test_connectivity_continuation import example, physical_backend

PLAN = {"charge": 0, "multiplicity": 1, "elements": ["H", "H"], "masses": [1., 1.],
        "drivers": [{"id": "d", "kind": "distance", "atoms": [0, 1], "maps": [1, 2],
                     "edit_kind": "broken", "lambda_values": [0., 1.], "values": [1., 1.]}],
        "guards": [], "content_sha256": "plan-execution-identity-fixture"}


def _corrector(tmp_path: Path) -> ORCALocalCorrector:
    return ORCALocalCorrector(PLAN, tmp_path/"attempts",
                              policy=LocalCorrectorPolicy(max_evaluations=2, max_iterations=1),
                              acp_root=_fake_acp(tmp_path), acp_python=sys.executable)


def _eval_dirs(attempt_root: Path):
    root = attempt_root/"physical_evaluations"
    return sorted(str(p.relative_to(root)) for p in root.glob("trial-*/*") if p.is_dir())


def test_replayed_attempt_with_new_geometry_recomputes_without_directory_collision(tmp_path: Path):
    corrector = _corrector(tmp_path)
    guess_a = [[0., 0., 0.], [1., 0., 0.]]
    # Same constraint target but a different starting guess: trial-0000 content differs.
    guess_b = [[0.1, 0., 0.], [1.05, 0., 0.]]
    corrector(guess_a, [1.], "attempt-0000")
    attempt_root = tmp_path/"attempts"/"attempt-0000"
    first = _eval_dirs(attempt_root)
    assert first and first[0].startswith("trial-0000/eval-")
    # Simulate the interrupted attempt: the attempt receipt is gone, the gradient
    # receipts remain. A replay with changed geometry must not collide with them.
    (attempt_root/"result.json").unlink()
    corrector(guess_b, [1.], "attempt-0000")
    second = _eval_dirs(attempt_root)
    trial_zero = [d for d in second if d.startswith("trial-0000/")]
    assert len(trial_zero) == 2
    assert len(set(trial_zero)) == 2


def test_identical_replay_reuses_gradient_receipts(tmp_path: Path):
    corrector = _corrector(tmp_path)
    guess = [[0., 0., 0.], [1., 0., 0.]]
    corrector(guess, [1.], "attempt-0000")
    attempt_root = tmp_path/"attempts"/"attempt-0000"
    before = _eval_dirs(attempt_root)
    receipt = json.loads((attempt_root/"result.json").read_text(encoding="utf-8"))
    replay = corrector(guess, [1.], "attempt-0000")
    assert _eval_dirs(attempt_root) == before
    assert replay["request_sha256"] == receipt["request_sha256"]
    assert replay["evaluation_identity_prefix"] == {
        "scheme": "pes2ts_execution_identity_v1",
        "plan_sha256": PLAN["content_sha256"], "attempt_id": "attempt-0000"}


def test_forged_same_address_receipt_is_a_typed_identity_conflict(tmp_path: Path):
    corrector = _corrector(tmp_path)
    corrector([[0., 0., 0.], [1., 0., 0.]], [1.], "attempt-0000")
    attempt_root = tmp_path/"attempts"/"attempt-0000"
    receipt_path = next((attempt_root/"physical_evaluations").glob("trial-*/*/result.json"))
    stored = json.loads(receipt_path.read_text(encoding="utf-8"))
    stored["request_sha256"] = "f"*64
    receipt_path.write_text(json.dumps(stored), encoding="utf-8")
    (attempt_root/"result.json").unlink()
    with pytest.raises(ValueError, match="IDENTITY_CONFLICT") as excinfo:
        corrector([[0., 0., 0.], [1., 0., 0.]], [1.], "attempt-0000")
    assert "f"*64 in str(excinfo.value)


def test_continuation_attempts_record_their_evaluation_identity_prefix():
    plan, initial = example(max_frames=3)
    result = run_continuation(plan, initial, physical_backend)
    assert result["attempts"]
    for attempt in result["attempts"]:
        prefix = attempt["evaluation_identity_prefix"]
        assert prefix["scheme"] == "pes2ts_execution_identity_v1"
        assert prefix["plan_sha256"] == plan["content_sha256"]
        assert prefix["attempt_id"] == attempt["attempt_id"]
        assert prefix["parent_frame_id"] == attempt["parent_frame_id"]
