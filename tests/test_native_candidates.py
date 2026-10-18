"""Energy-only native-scan candidate selection tests (G2-AB3 WP-6).

Locks physical-energy peak bracketing, ``edge_unbracketed_prefix`` on an
incomplete interval, empty-with-reason when no physical energies exist, and
that ``gradient_available`` is always False.
"""
from __future__ import annotations

import pytest

from pes2ts_core.generation.planning.native_candidates import (
    ENERGY_CHANNEL,
    NATIVE_CANDIDATES_SCHEMA,
    PEAK_RULE,
    select_native_energy_candidates,
)


def make_frame(index, lam, e_phys, *, e_eff=-1.2, geometry=None, request_sha=None):
    return {
        "frame_index": index,
        "lambda_value": lam,
        "energy_physical_hartree": e_phys,
        "energy_effective_hartree": e_eff,
        "request_sha256": request_sha or f"sha-{index}",
        "attempt_dir": f"/tmp/attempt-{index}",
        "geometry": geometry or [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    }


def make_result(frames, *, completed_interval=True, plan_sha256="plan-abc",
                reaction_id="RXN_CAND"):
    return {
        "schema_version": "pes2ts_native_scan_result_v1",
        "engine_mode": "classic_native_xtb_scan",
        "reaction_id": reaction_id,
        "frames": frames,
        "completed_interval": completed_interval,
        "plan_sha256": plan_sha256,
    }


def test_bracketed_physical_energy_peak_is_selected():
    frames = [
        make_frame(0, 0.0, -1.0),
        make_frame(1, 0.5, -0.8),
        make_frame(2, 1.0, -0.9),
    ]
    proposals = select_native_energy_candidates(make_result(frames))
    assert proposals["schema_version"] == NATIVE_CANDIDATES_SCHEMA
    assert proposals["rule"] == PEAK_RULE
    assert proposals["energy_channel"] == ENERGY_CHANNEL
    assert proposals["completed_interval"] is True
    assert proposals["gradient_available"] is False
    assert proposals["n_candidates"] == 1
    (candidate,) = proposals["candidates"]
    assert candidate["evidence_class"] == "bracketed_physical_energy_peak"
    assert candidate["frame_index"] == 1
    assert candidate["lambda_value"] == 0.5
    assert candidate["energy_physical_hartree"] == pytest.approx(-0.8)
    assert candidate["neighbor_energies_hartree"] == pytest.approx((-1.0, -0.9))
    assert candidate["request_sha256"] == "sha-1"
    assert candidate["gradient_available"] is False
    assert candidate["stationary_point_verified"] is False
    assert candidate["observation_mask"] == {
        "physical_energy": True,
        "physical_gradient": False,
        "effective_energy": True,
    }
    assert candidate["geometry"] == frames[1]["geometry"]
    assert len(candidate["geometry_sha256"]) == 64
    assert candidate["plan_sha256"] == "plan-abc"
    assert candidate["attempt_dir"] == "/tmp/attempt-1"
    assert proposals["validation_status"] == "blocked_on_human_review"


def test_edge_unbracketed_prefix_on_incomplete_interval():
    frames = [
        make_frame(0, 0.0, -1.0),
        make_frame(1, 0.5, -0.9),
        make_frame(2, 1.0, -0.8),  # highest energy on the LAST accepted frame
    ]
    proposals = select_native_energy_candidates(
        make_result(frames, completed_interval=False)
    )
    assert proposals["n_candidates"] == 1
    (candidate,) = proposals["candidates"]
    assert candidate["evidence_class"] == "edge_unbracketed_prefix"
    assert candidate["frame_index"] == 2
    assert candidate["energy_physical_hartree"] == pytest.approx(-0.8)
    assert candidate["neighbor_energies_hartree"] == pytest.approx((-0.9,))
    assert candidate["gradient_available"] is False


def test_no_physical_energy_channel_is_empty_with_reason():
    frames = [
        make_frame(0, 0.0, None),
        make_frame(1, 0.5, None),
        make_frame(2, 1.0, None),
    ]
    proposals = select_native_energy_candidates(make_result(frames))
    assert proposals["n_candidates"] == 0
    assert proposals["candidates"] == []
    assert proposals["no_candidate_reason"] == "physical_energy_channel_missing"
    assert proposals["gradient_available"] is False


def test_fewer_than_two_frames_is_empty_with_reason():
    proposals = select_native_energy_candidates(make_result([make_frame(0, 0.0, -1.0)]))
    assert proposals["n_candidates"] == 0
    assert proposals["no_candidate_reason"] == "no_accepted_frames"
    assert proposals["n_accepted_frames"] == 0
    assert proposals["gradient_available"] is False
    proposals = select_native_energy_candidates({"frames": []})
    assert proposals["no_candidate_reason"] == "no_accepted_frames"


def test_effective_energy_never_drives_peak_selection():
    # E_effective has a fake mid-curve bump; E_physical is flat -> no peak.
    frames = [
        make_frame(0, 0.0, -1.0, e_eff=-1.0),
        make_frame(1, 0.5, -1.0, e_eff=-0.5),
        make_frame(2, 1.0, -1.0, e_eff=-1.0),
    ]
    proposals = select_native_energy_candidates(make_result(frames))
    assert proposals["n_candidates"] == 0
    assert all(
        c["evidence_class"] != "bracketed_physical_energy_peak"
        for c in proposals["candidates"]
    )
    assert proposals["gradient_available"] is False


def test_top_k_truncates_and_ranks_by_physical_energy():
    frames = [
        make_frame(0, 0.0, -1.0),
        make_frame(1, 0.25, -0.5),
        make_frame(2, 0.5, -0.7),
        make_frame(3, 0.75, -0.4),
        make_frame(4, 1.0, -0.9),
    ]
    proposals = select_native_energy_candidates(make_result(frames), top_k=1)
    assert proposals["n_candidates"] == 1
    assert proposals["candidates"][0]["frame_index"] == 3
    proposals = select_native_energy_candidates(make_result(frames), top_k=5)
    assert proposals["n_candidates"] == 2
    assert [c["frame_index"] for c in proposals["candidates"]] == [3, 1]
    assert proposals["gradient_available"] is False


def test_missing_neighbor_energies_skip_the_peak_rule():
    frames = [
        make_frame(0, 0.0, -1.0),
        make_frame(1, 0.5, None),
        make_frame(2, 1.0, -0.8),
    ]
    proposals = select_native_energy_candidates(make_result(frames))
    assert proposals["n_candidates"] == 0
    # Interior frame with a numeric peak but a None neighbour is skipped too.
    frames = [
        make_frame(0, 0.0, None),
        make_frame(1, 0.5, -0.5),
        make_frame(2, 1.0, -0.8),
    ]
    proposals = select_native_energy_candidates(make_result(frames))
    assert proposals["n_candidates"] == 0


def test_incomplete_interval_can_report_both_peak_and_edge():
    frames = [
        make_frame(0, 0.0, -1.0),
        make_frame(1, 0.25, -0.5),
        make_frame(2, 0.5, -0.9),
        make_frame(3, 0.75, -0.4),  # highest, last frame, unbracketed
    ]
    proposals = select_native_energy_candidates(
        make_result(frames, completed_interval=False)
    )
    classes = [c["evidence_class"] for c in proposals["candidates"]]
    assert "bracketed_physical_energy_peak" in classes
    assert "edge_unbracketed_prefix" in classes
    assert proposals["n_candidates"] == 2
    assert [c["frame_index"] for c in proposals["candidates"]] == [1, 3]
    assert proposals["gradient_available"] is False


def test_completed_interval_suppresses_edge_candidate():
    frames = [
        make_frame(0, 0.0, -1.0),
        make_frame(1, 0.5, -0.9),
        make_frame(2, 1.0, -0.8),
    ]
    proposals = select_native_energy_candidates(
        make_result(frames, completed_interval=True)
    )
    assert proposals["n_candidates"] == 0
    assert "no_candidate_reason" not in proposals
