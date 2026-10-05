"""Pilot judge + branch calibration tests (G2-AB2 WP-2, offline V0)."""
from __future__ import annotations

import pytest

from pes2ts_core.generation.planning.branch_calibration import (
    CALIBRATION_SCHEMA,
    CHOSEN_BRANCH_POLICY,
    CHOSEN_CONTINUATION_KNOBS,
    branch_selection_case,
    biased_release_case,
    cumulative_gate_case,
    probe_budget_case,
    scan_branch_policy_grid,
)
from pes2ts_core.generation.planning.local_corrector import BranchPolicy
from pes2ts_core.generation.planning.replay_report import (
    PILOT_REPLAY_SCHEMA,
    judge_pilot,
    judge_pilot_case,
)


def terminal(**overrides):
    base = {"schema_version": "pes2ts_g2t_terminal_state_v1",
            "terminal_class": "partial_prefix",
            "identity_conflict_count": 0,
            "n_accepted_frames_beyond_origin": 3,
            "last_accepted_lambda": 0.7,
            "origin_status": "prepared",
            "origin_failure_code": None,
            "failure_code": None, "blocked_reason": None, "blocked_needs": [],
            "costs": {"n_gradient": 42}}
    base.update(overrides)
    return base


def origin_connection_lost(**overrides):
    base = {"failure_code": "ORIGIN_CONNECTION_LOST",
            "source_geometry_hash": "abc",
            "failure_detail": {"source_geometry_hash": "abc",
                               "lost_connections": [
                                   {"atoms": [8, 16],
                                    "distance_before_angstrom": 1.03,
                                    "distance_after_angstrom": 2.10}]}}
    base.update(overrides)
    return base


def origin_failed_terminal(**overrides):
    base = {"origin_status": "failed",
            "origin_failure_code": "ORIGIN_CONNECTION_LOST",
            "terminal_class": "rejected_typed",
            "n_accepted_frames_beyond_origin": 0,
            "last_accepted_lambda": None}
    base.update(overrides)
    return terminal(**base)


# -- replay judge -----------------------------------------------------------

def test_passed_case_requires_all_three_criteria():
    verdict = judge_pilot_case({"terminal": terminal(),
                                "reference": {"last_accepted_lambda": 0.5}})
    assert verdict["verdict"] == "passed"
    assert verdict["criteria"]["progressed_beyond_reference"] is True
    assert verdict["evidence"]["n_gradient"] == 42


def test_identity_conflict_fails_the_case_b1_1():
    verdict = judge_pilot_case({"terminal": terminal(identity_conflict_count=1)})
    assert verdict["verdict"] == "failed"
    assert verdict["criteria"]["identity_conflicts"] is False


def test_progression_must_go_beyond_reference_lambda_b1_2():
    verdict = judge_pilot_case({"terminal": terminal(last_accepted_lambda=0.5),
                                "reference": {"last_accepted_lambda": 0.5}})
    assert verdict["verdict"] == "failed"
    assert verdict["criteria"]["progressed_beyond_reference"] is False
    no_reference = judge_pilot_case({"terminal": terminal()})
    assert no_reference["criteria"]["progressed_beyond_reference"] is True


def test_typed_connection_lost_detail_is_required_b1_3():
    verdict = judge_pilot_case({"terminal": origin_failed_terminal(),
                                "origin_evidence": origin_connection_lost()})
    assert verdict["criteria"]["origin_failure_typed"] is True
    assert verdict["criteria"]["progressed_beyond_reference"] is None
    assert verdict["verdict"] == "passed"  # v2: typed origin failure IS the D2 criterion
    assert "not applicable" in verdict["criteria_note"]
    generic = {"failure_code": "PREPARED_ORIGIN_GEOMETRY_INVALID",
               "failure_detail": {}}
    loose = judge_pilot_case({"terminal": origin_failed_terminal(
        origin_failure_code="PREPARED_ORIGIN_GEOMETRY_INVALID"),
        "origin_evidence": generic})
    assert loose["criteria"]["origin_failure_typed"] is False
    assert loose["verdict"] == "failed"


def test_origin_failed_case_passes_under_v2_adr0005():
    verdict = judge_pilot_case({"terminal": origin_failed_terminal(),
                                "origin_evidence": origin_connection_lost()})
    assert verdict["verdict"] == "passed"
    assert verdict["criteria"] == {"identity_conflicts": True,
                                   "progressed_beyond_reference": None,
                                   "origin_failure_typed": True}


def test_origin_failed_identity_conflict_still_fails_under_v2():
    verdict = judge_pilot_case({"terminal": origin_failed_terminal(
        identity_conflict_count=1), "origin_evidence": origin_connection_lost()})
    assert verdict["verdict"] == "failed"
    assert verdict["criteria"]["identity_conflicts"] is False


def test_blocked_stays_blocked_even_with_origin_failure_evidence():
    verdict = judge_pilot_case({"terminal": origin_failed_terminal(
        terminal_class="blocked", blocked_reason="acp_environment_unavailable",
        blocked_needs=["PES2TS_ACP_ROOT ..."]),
        "origin_evidence": origin_connection_lost()})
    assert verdict["verdict"] == "blocked"
    assert verdict["needs"] == ["PES2TS_ACP_ROOT ..."]


def test_reached_continuation_keeps_three_criteria_under_v2():
    verdict = judge_pilot_case({"terminal": terminal(last_accepted_lambda=0.5),
                                "reference": {"last_accepted_lambda": 0.5}})
    assert verdict["verdict"] == "failed"
    assert verdict["criteria"]["progressed_beyond_reference"] is False
    assert "criteria_note" not in verdict  # the v2 note is scoped to origin-failed cases


def test_blocked_environment_is_blocked_with_needs_never_failed():
    verdict = judge_pilot_case({"terminal": terminal(
        terminal_class="blocked", blocked_reason="acp_environment_unavailable",
        blocked_needs=["PES2TS_ACP_ROOT ..."], n_accepted_frames_beyond_origin=0,
        last_accepted_lambda=None)})
    assert verdict["verdict"] == "blocked"
    assert verdict["needs"] == ["PES2TS_ACP_ROOT ..."]
    report = judge_pilot({"RXN_0000017762": {
        "terminal": terminal(terminal_class="blocked",
                             blocked_reason="acp_environment_unavailable",
                             blocked_needs=["PES2TS_ACP_ROOT ..."])}})
    assert report["counts"] == {"passed": 0, "failed": 0, "blocked": 1}
    assert report["all_passed"] is False


def test_aggregate_report_counts_and_freezes_criteria():
    report = judge_pilot({
        "RXN_0000079731": {"terminal": terminal(), "reference": {"last_accepted_lambda": 0.2}},
        "RXN_0000091050": {"terminal": terminal(identity_conflict_count=0),
                           "reference": {"last_accepted_lambda": 0.6}},
        "RXN_0000017762": {"terminal": origin_failed_terminal(),
                           "origin_evidence": origin_connection_lost()},
    })
    assert report["schema_version"] == PILOT_REPLAY_SCHEMA == \
        "pes2ts_pilot_replay_report_v2"
    assert report["criteria_revision"] == "v2_adr0005"
    assert report["counts"] == {"passed": 3, "failed": 0, "blocked": 0}
    assert report["all_passed"] is True
    assert report["identity_conflicts_total"] == 0
    assert report["criteria_frozen"] is True
    with pytest.raises(ValueError, match="EMPTY_PILOT"):
        judge_pilot({})
    with pytest.raises(ValueError, match="INVALID_PILOT_TERMINAL"):
        judge_pilot_case({"terminal": {"schema_version": "other"}})


# -- branch calibration -----------------------------------------------------

SMALL_GRID = {
    "branch_rmsd_radius": (.25,),
    "branch_atom_radius": (.50,),
    "cumulative_rmsd_radius": (.25,),
    "bias_kappa": (.05, .50),
    "release_iterations": (4,),
    "cumulative_rmsd_limit": (1.0,),
    "max_curvature_probes": (1,),
}


def test_chosen_constants_match_a_fresh_scan():
    report = scan_branch_policy_grid(SMALL_GRID)
    assert report["schema_version"] == CALIBRATION_SCHEMA
    assert report["chosen_cell"]["bias_kappa"] == .50  # weak anchor fails branch
    assert asdict_subset(report["chosen_cell"], CHOSEN_BRANCH_POLICY)
    assert report["chosen_cell"]["cumulative_rmsd_limit"] == \
        CHOSEN_CONTINUATION_KNOBS["cumulative_rmsd_limit"]
    assert report["chosen_cell"]["max_curvature_probes"] == \
        CHOSEN_CONTINUATION_KNOBS["max_curvature_probes"]
    assert report["holdout_touched"] is False
    assert report["calibration_set_sha256"]


def asdict_subset(cell: dict, policy: BranchPolicy) -> bool:
    import dataclasses
    frozen = dataclasses.asdict(policy)
    return all(cell[key] == value for key, value in frozen.items())


def test_fixture_criteria_discriminate():
    weak = BranchPolicy(branch_rmsd_radius=.25, branch_atom_radius=.5,
                        cumulative_rmsd_radius=.25, bias_kappa=.05,
                        release_iterations=4)
    strong = BranchPolicy(branch_rmsd_radius=.25, branch_atom_radius=.5,
                          cumulative_rmsd_radius=.25, bias_kappa=.5,
                          release_iterations=4)
    assert branch_selection_case(weak)["branch_selection_ok"] is False
    assert branch_selection_case(strong)["branch_selection_ok"] is True
    assert biased_release_case(strong)["biased_only_honest"] is True
    assert biased_release_case(strong)["convergence_mode"] == "biased_only"
    strict = cumulative_gate_case(1.0)
    loose = cumulative_gate_case(2.0)
    assert strict["cumulative_gate_discriminates"] is True
    assert loose["cumulative_gate_discriminates"] is False
    assert probe_budget_case(1) == {"probe_budget_respected": True,
                                    "probes_fired": 1, "probe_calls": 1,
                                    "budget": 1}


def test_calibration_report_lists_rejected_cells():
    report = scan_branch_policy_grid(SMALL_GRID)
    assert report["n_cells"] == 2
    assert report["n_passing"] == 1
    assert report["rejected_example_cells"]
    assert report["chosen_is_most_conservative_passing"] is True
