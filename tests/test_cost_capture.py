"""R5 cost-capture counting rules (G2-AB2 WP-1).

Unit-locks the frozen rules: cache hit = true zero with source, EnGrad
energy==gradient count, probe gradients counted once (probe ledger, not the
parent attempt), missing measurement = null + reason (never 0), origin
overlap screen = zero-cost typed rejection, campaign totals dedup parents.
"""
from __future__ import annotations

import sys

import pytest

from pes2ts_core.flywheel import aggregate_costs
from pes2ts_core.generation.planning.cost_capture import (
    capture_continuation_costs,
    capture_origin_cost,
    total_gradient_calls,
)
from pes2ts_core.generation.planning.local_corrector import LocalCorrectorPolicy
from pes2ts_core.integration.acp.gradient_backend import ORCALocalCorrector

from test_acp_orca_gradient_transport import _fake_acp


def attempt(attempt_id, *, n_gradient=3, duration=1.5, reused=False, wall=None,
            probe=None):
    evidence = {"n_gradient_evaluations": n_gradient, "duration_seconds": duration}
    acp: dict = {"reused": reused}
    if wall is not None:
        acp["wall_seconds"] = wall
    evidence["acp"] = acp
    row = {"attempt_id": attempt_id, "backend_evidence": evidence}
    if probe is not None:
        row["curvature_probe"] = probe
    return row


def test_gradient_attempt_counts_energy_equal_and_prefers_receipt_wall():
    result = {"attempts": [attempt("attempt-0000", n_gradient=4, duration=9.,
                                   wall=3.25)]}
    bundle = capture_continuation_costs(result, reaction_id="RXN_1")
    ledger = bundle["ledgers"][0]
    assert ledger["N_gradient"] == 4
    assert ledger["N_energy"] == 4  # EnGrad returns energy+gradient in one call
    assert ledger["cpu_seconds"] == 3.25  # receipt-measured beats in-process
    assert ledger["cost_source"] == "acp_orca_gradient_attempt"
    assert bundle["aggregate"]["N_gradient"] == 4


def test_cached_receipt_is_a_true_zero_with_source():
    result = {"attempts": [attempt("attempt-0000", n_gradient=5, reused=True)]}
    bundle = capture_continuation_costs(result, reaction_id="RXN_1")
    ledger = bundle["ledgers"][0]
    assert (ledger["N_gradient"], ledger["N_energy"]) == (0, 0)
    assert ledger["cpu_seconds"] == 0.0
    assert ledger["cost_source"] == "cache_hit:attempt-0000"
    assert bundle["aggregate"]["N_gradient"] == 0


def test_missing_measurement_is_null_with_reason_never_zero():
    result = {"attempts": [attempt("attempt-0000", n_gradient=2, duration=None,
                                   wall=None)]}
    bundle = capture_continuation_costs(result, reaction_id="RXN_1")
    ledger = bundle["ledgers"][0]
    assert ledger["cpu_seconds"] is None
    assert ledger["cpu_seconds_reason"]
    assert ledger["allocated_core_seconds"] is None
    assert ledger["allocated_core_seconds_reason"]


def test_attempt_without_backend_evidence_keeps_a_typed_ledger():
    result = {"attempts": [{"attempt_id": "attempt-0000"}]}
    bundle = capture_continuation_costs(result, reaction_id="RXN_1")
    ledger = bundle["ledgers"][0]
    assert ledger["N_gradient"] == 0
    assert ledger["cpu_seconds"] is None
    assert ledger["cost_source"] == "missing_backend_evidence"


def test_curvature_probe_gradients_count_once_in_probe_ledger_only():
    probe = {"n_gradient_evaluations": 2, "duration_seconds": .4}
    result = {"attempts": [attempt("attempt-0000", n_gradient=3, probe=probe)]}
    bundle = capture_continuation_costs(result, reaction_id="RXN_1")
    by_source = {row["cost_source"]: row for row in bundle["ledgers"]}
    assert by_source["curvature_probe"]["N_gradient"] == 2
    assert by_source["acp_orca_gradient_attempt"]["N_gradient"] == 3
    # Probe gradients are NOT folded into the parent attempt's counter.
    assert bundle["aggregate"]["N_gradient"] == 5


def test_origin_cost_is_one_free_optimization_call():
    evidence = {"status": "prepared", "duration_seconds": 12.}
    ledger = capture_origin_cost(evidence, reaction_id="RXN_2")
    assert ledger["N_gradient"] == 1
    assert ledger["cpu_seconds"] == 12.
    assert ledger["cost_source"] == "origin_free_optimization"


def test_origin_overlap_screen_never_reached_qc_and_costs_nothing():
    evidence = {"status": "invalid_input", "failure_code": "ORIGIN_OVERLAP_INPUT",
                "duration_seconds": 0.001}
    assert capture_origin_cost(evidence, reaction_id="RXN_2") is None


def test_campaign_total_dedups_and_rejects_duplicate_ids():
    left = capture_continuation_costs(
        {"attempts": [attempt("attempt-0000", n_gradient=3)]}, reaction_id="RXN_A")
    right = capture_continuation_costs(
        {"attempts": [attempt("attempt-0000", n_gradient=2)]}, reaction_id="RXN_B")
    assert total_gradient_calls(left, right) == 5
    with pytest.raises(ValueError):
        aggregate_costs(left["ledgers"] + left["ledgers"])


def test_invalid_input_shapes_are_typed_rejections():
    with pytest.raises(ValueError, match="INVALID_CONTINUATION_RESULT"):
        capture_continuation_costs({"attempts": "nope"}, reaction_id="RXN_1")
    with pytest.raises(ValueError, match="INVALID_ATTEMPT_RECORD"):
        capture_continuation_costs({"attempts": [42]}, reaction_id="RXN_1")
    with pytest.raises(ValueError, match="INVALID_ORIGIN_EVIDENCE"):
        capture_origin_cost(["nope"], reaction_id="RXN_1")


# ---------------------------------------------------------------------------
# G2-T3 P1-3: a whole-receipt replay of ORCALocalCorrector is a cache hit.
# The corrector result becomes attempt["backend_evidence"] in run_continuation
# (continuation.py:369) — the dict cost capture reads — so the returned copy
# must carry the top-level acp.reused flag; the persisted receipt stays intact.
# ---------------------------------------------------------------------------
CORRECTOR_PLAN = {
    "charge": 0, "multiplicity": 1, "elements": ["H", "H"], "masses": [1., 1.],
    "drivers": [{"id": "d", "kind": "distance", "atoms": [0, 1], "maps": [1, 2],
                 "edit_kind": "broken", "lambda_values": [0., 1.], "values": [1., 1.]}],
    "guards": [], "content_sha256": "plan-cost-capture-reuse-fixture"}
CORRECTOR_GUESS = [[0., 0., 0.], [1., 0., 0.]]


def _corrector(tmp_path):
    return ORCALocalCorrector(
        CORRECTOR_PLAN, tmp_path / "attempts",
        policy=LocalCorrectorPolicy(max_evaluations=2, max_iterations=1),
        acp_root=_fake_acp(tmp_path), acp_python=sys.executable)


def _result_for_capture(corrector_result):
    return {"attempts": [{"attempt_id": "attempt-0000",
                          "backend_evidence": corrector_result}]}


def test_replayed_corrector_receipt_is_a_cache_hit_ledger(tmp_path):
    corrector = _corrector(tmp_path)
    first = corrector(CORRECTOR_GUESS, [1.], "attempt-0000")
    second = corrector(CORRECTOR_GUESS, [1.], "attempt-0000")

    assert first["n_gradient_evaluations"] >= 1
    assert second["acp"]["reused"] is True
    assert second["request_sha256"] == first["request_sha256"]

    first_ledger = capture_continuation_costs(
        _result_for_capture(first), reaction_id="RXN_REPLAY")["ledgers"][0]
    second_bundle = capture_continuation_costs(
        _result_for_capture(second), reaction_id="RXN_REPLAY")
    second_ledger = second_bundle["ledgers"][0]

    assert first_ledger["N_gradient"] == first["n_gradient_evaluations"] > 0
    assert first_ledger["N_energy"] == first["n_gradient_evaluations"]
    assert first_ledger["cost_source"] == "acp_orca_gradient_attempt"
    assert first_ledger["cpu_seconds"] > 0

    assert second_ledger["cost_source"] == "cache_hit:attempt-0000"
    assert (second_ledger["N_gradient"], second_ledger["N_energy"]) == (0, 0)
    assert second_ledger["cpu_seconds"] == 0.0
    assert second_bundle["aggregate"]["N_gradient"] == 0


def test_replayed_corrector_receipt_never_rewrites_the_persisted_result(tmp_path):
    corrector = _corrector(tmp_path)
    corrector(CORRECTOR_GUESS, [1.], "attempt-0000")
    receipt = tmp_path / "attempts" / "attempt-0000" / "result.json"
    before = receipt.read_text(encoding="utf-8")
    assert '"reused"' not in before

    replay = corrector(CORRECTOR_GUESS, [1.], "attempt-0000")
    assert replay["acp"]["reused"] is True
    assert receipt.read_text(encoding="utf-8") == before


def test_replayed_corrector_receipt_hash_mismatch_still_rejects(tmp_path):
    corrector = _corrector(tmp_path)
    corrector(CORRECTOR_GUESS, [1.], "attempt-0000")
    moved = [[0., 0., 0.], [1.2, 0., 0.]]
    with pytest.raises(ValueError, match="CACHED_LOCAL_CORRECTOR_INPUT_MISMATCH"):
        corrector(moved, [1.], "attempt-0000")
