"""G2.T campaign orchestration tests (G2-AB2 WP-1, offline V0).

Drives the full case pipeline (real snapshot parsing → origin preparation →
plan build → continuation → candidates → costs → typed terminal) with a
synthetic exact-projection corrector on the real Demo24 fixture records, and
unit-locks the manifest/idempotence/budget/terminal-class disciplines.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pes2ts_core.generation.campaign import (
    BLOCKED_CAMPAIGN_BUDGET,
    BLOCKED_HUMAN_REVIEW,
    CAMPAIGN_MANIFEST_SCHEMA,
    CampaignBackends,
    CampaignError,
    TERMINAL_STATE_SCHEMA,
    attach_validation_result,
    build_campaign_manifest,
    campaign_status,
    load_campaign_manifest,
    run_campaign,
    validate_campaign_manifest,
)
from pes2ts_core.generation.planning.continuation import (
    ContinuationPolicy,
    project_geometry,
)
from pes2ts_core.generation.planning.origin_preparation import prepare_origin

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "p0_demo24" / "records"
PILOT = ["RXN_0000017762", "RXN_0000079731", "RXN_0000091050", "RXN_0000138453"]


def synthetic_policy(**overrides) -> ContinuationPolicy:
    base = dict(max_frames=14, max_attempts=80, max_seconds=600.,
                cumulative_rmsd_limit=10.0)
    base.update(overrides)
    return ContinuationPolicy(**base)


def identity_origin(geometry, elements, bond_indices, charge, multiplicity, case_root):
    def evaluate_free(x, evaluation_id):
        return {"success": True, "failure_class": None,
                "coordinates": np.asarray(x, float).tolist(), "energy": -1.0,
                "duration_seconds": 0.01, "acp_receipt_ref": "synthetic:origin"}
    return prepare_origin(geometry, elements, bond_indices, evaluate_free,
                          method_context={"method": "synthetic"})


class SyntheticCorrector:
    def __init__(self, plan):
        self.plan = plan
        self.coordinates = plan["drivers"] + plan.get("guards", [])

    def set_branch_reference(self, geometry, frame_id):
        pass

    def on_accept(self, result):
        pass

    def __call__(self, guess, targets, attempt_id):
        x, _info = project_geometry(np.asarray(guess, float), self.coordinates,
                                    targets, self.plan["masses"], ContinuationPolicy())
        if x is None:
            return {"success": False, "converged": False, "failure_class": "OPT_FAILED",
                    "duration_seconds": 0.001, "n_gradient_evaluations": 1,
                    "acp": {"reused": False}}
        return {"success": True, "converged": True, "coordinates": x.tolist(),
                "energy": 1e-6*float(np.sum(x**2)), "duration_seconds": 0.002,
                "physical_gradient_status": "not_collected",
                "n_gradient_evaluations": 2,
                "evaluations": [{"evaluation_id": attempt_id + "/e0"}],
                "acp": {"reused": False, "wall_seconds": 0.002}}


def synthetic_backends():
    return CampaignBackends(
        prepare_origin=identity_origin,
        make_corrector=lambda plan, case_root: SyntheticCorrector(plan))


def tiny_total_budget_manifest(manifest_path: Path, tmp_path: Path) -> Path:
    """A copy of the module manifest with the campaign gradient ceiling at 1."""
    manifest = json.loads(manifest_path.read_text())
    manifest["budget"]["max_total_gradient_calls"] = 1
    del manifest["content_sha256"]
    from pes2ts_core.generation.planning.synchronized_path import digest
    manifest["content_sha256"] = digest(manifest)
    path = tmp_path / "tiny_total_budget_manifest.json"
    path.write_text(json.dumps(manifest))
    return path


@pytest.fixture(scope="module")
def manifest_path(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("manifest")
    # Shrink to 2 cases for a fast suite: one strict pilot case + one more.
    snapshots = [FIXTURES / "RXN_0000017762.json", FIXTURES / "RXN_0000079731.json"]
    path = directory / "manifest.json"
    build_campaign_manifest(
        snapshots, output_path=path, pilot_cases=["RXN_0000017762"],
        continuation_policy=synthetic_policy(),
        budget={"max_frames_per_path": synthetic_policy().max_frames,
                "max_attempts_per_path": synthetic_policy().max_attempts,
                "max_wall_seconds_per_path": synthetic_policy().max_seconds},
        reference_failures={"RXN_0000017762": {"source": "round2",
                                               "last_accepted_lambda": 0.5,
                                               "known_failure": "B1-2 branch drift"}})
    return path


def test_manifest_freezes_snapshot_hashes_and_validates(manifest_path):
    manifest = load_campaign_manifest(manifest_path)
    assert manifest["schema_version"] == CAMPAIGN_MANIFEST_SCHEMA
    assert [case["reaction_id"] for case in manifest["cases"]] == \
        ["RXN_0000017762", "RXN_0000079731"]
    for case in manifest["cases"]:
        assert len(case["snapshot_sha256"]) == 64
    assert manifest["review_status"]["RXN_0000017762"] == "needs_review"
    assert manifest["reference_failures"]["RXN_0000017762"]["known_failure"] == "B1-2 branch drift"
    with pytest.raises(CampaignError, match="MANIFEST_CONTENT_HASH_MISMATCH"):
        tampered = json.loads(manifest_path.read_text())
        tampered["review_status"]["RXN_0000017762"] = "accepted"
        validate_campaign_manifest(tampered)


def test_budget_must_match_frozen_policy_single_truth():
    with pytest.raises(CampaignError, match="BUDGET_POLICY_MISMATCH"):
        build_campaign_manifest(
            [FIXTURES / "RXN_0000017762.json"],
            continuation_policy=synthetic_policy(),
            budget={"max_frames_per_path": synthetic_policy().max_frames + 1,
                    "max_attempts_per_path": synthetic_policy().max_attempts,
                    "max_wall_seconds_per_path": synthetic_policy().max_seconds})


def test_run_campaign_produces_typed_terminals_and_costs(manifest_path, tmp_path):
    root = tmp_path / "campaign"
    summary = run_campaign(manifest_path, root, backends=synthetic_backends())
    assert summary.n_executed == 2
    assert set(summary.terminal_counts) <= {"partial_prefix", "complete_path",
                                            "rejected_typed"}
    for reaction_id in ("RXN_0000017762", "RXN_0000079731"):
        terminal = json.loads((root / reaction_id / "terminal_state.json").read_text())
        assert terminal["schema_version"] == TERMINAL_STATE_SCHEMA
        assert terminal["terminal_class"] in {"partial_prefix", "complete_path",
                                              "rejected_typed"}
        assert terminal["validation"]["status"] == BLOCKED_HUMAN_REVIEW
        assert terminal["validation"]["needs"]
        costs = json.loads((root / reaction_id / "costs.json").read_text())
        assert costs["aggregate"]["N_gradient"] > 0
        for artifact in ("origin_evidence.json", "plan.json",
                         "continuation_result.json", "candidates.json", "costs.json"):
            assert (root / reaction_id / artifact).is_file()
    status = campaign_status(root)
    assert status["n_cases_with_terminal"] == 2
    denominators = status["four_class_denominators"]
    assert sum(denominators.values()) == 2
    # No success phrasing: rejected/censored are counted, never folded in.
    assert "rejected_typed" in denominators or denominators["partial_prefix"] + \
        denominators["complete_path"] + denominators["validated"] == 2


def test_resume_is_idempotent_and_blocked_cases_retry(manifest_path, tmp_path):
    root = tmp_path / "campaign"
    first = run_campaign(manifest_path, root, backends=synthetic_backends())
    before = {p.name: p.read_bytes() for p in sorted(root.rglob("terminal_state.json"))}
    second = run_campaign(manifest_path, root, backends=synthetic_backends())
    assert second.n_executed == 0
    assert second.n_skipped_idempotent == 2
    after = {p.name: p.read_bytes() for p in sorted(root.rglob("terminal_state.json"))}
    assert set(before) == set(after)
    for name in before:
        assert json.loads(before[name])["terminal_class"] == \
            json.loads(after[name])["terminal_class"]
    assert first.terminal_counts == second.terminal_counts


def test_identity_conflict_is_a_typed_rejected_terminal(manifest_path, tmp_path):
    def conflicting_origin(geometry, elements, bond_indices, charge, multiplicity,
                           case_root):
        return identity_origin(geometry, elements, bond_indices, charge,
                               multiplicity, case_root)

    class ConflictingCorrector(SyntheticCorrector):
        def __call__(self, guess, targets, attempt_id):
            raise ValueError(
                "IDENTITY_CONFLICT:CACHED_GRADIENT_INPUT_MISMATCH:"
                f"evaluation_id={attempt_id}:stored=aa:binding=bb")

    backends = CampaignBackends(
        prepare_origin=conflicting_origin,
        make_corrector=lambda plan, case_root: ConflictingCorrector(plan))
    root = tmp_path / "conflict"
    summary = run_campaign(manifest_path, root, backends=backends)
    assert summary.terminal_counts["rejected_typed"] == 2
    terminal = json.loads((root / "RXN_0000017762" / "terminal_state.json").read_text())
    assert terminal["failure_code"] == "IDENTITY_CONFLICT"
    assert terminal["identity_conflict_count"] == 1
    assert terminal["label_state"] == "execution_failed"
    assert (root / "RXN_0000017762" / "origin_evidence.json").is_file()
    assert (root / "RXN_0000017762" / "costs.json").is_file()


def test_origin_failure_maps_to_rejected_typed_with_or_code(manifest_path, tmp_path):
    def breaking_origin(geometry, elements, bond_indices, charge, multiplicity,
                        case_root):
        broken = np.asarray(geometry, float).copy()
        broken[1] += np.array([2.9, 0., 0.])

        def evaluate_free(x, evaluation_id):
            return {"success": True, "failure_class": None,
                    "coordinates": broken.tolist(), "energy": -1.1,
                    "duration_seconds": 0.01}
        return prepare_origin(geometry, elements, bond_indices, evaluate_free)

    backends = CampaignBackends(
        prepare_origin=breaking_origin,
        make_corrector=lambda plan, case_root: SyntheticCorrector(plan))
    root = tmp_path / "origin_failed"
    summary = run_campaign(manifest_path, root, backends=backends)
    assert summary.terminal_counts["rejected_typed"] == 2
    terminal = json.loads((root / "RXN_0000017762" / "terminal_state.json").read_text())
    assert terminal["origin_failure_code"] in {"ORIGIN_CONNECTION_LOST",
                                               "ORIGIN_PREPARED_COLLISION"}
    assert terminal["stage_reached"] == "origin_preparation"
    costs = json.loads((root / "RXN_0000017762" / "costs.json").read_text())
    assert costs["aggregate"]["N_gradient"] == 1  # the failed free opt is billed


def test_environment_block_is_typed_and_retried_when_available(manifest_path, tmp_path):
    from pes2ts_core.generation.campaign import OriginBackendUnavailable
    state = {"available": False}

    def probing_origin(geometry, elements, bond_indices, charge, multiplicity,
                       case_root):
        if not state["available"]:
            raise OriginBackendUnavailable(
                ["PES2TS_ACP_ROOT must point at an ACP checkout",
                 "ACP BatchOptimize non-TS profile capability probe failed"])
        return identity_origin(geometry, elements, bond_indices, charge,
                               multiplicity, case_root)

    backends = CampaignBackends(
        prepare_origin=probing_origin,
        make_corrector=lambda plan, case_root: SyntheticCorrector(plan))
    root = tmp_path / "blocked"
    first = run_campaign(manifest_path, root, backends=backends)
    assert first.n_blocked == 2
    terminal = json.loads((root / "RXN_0000017762" / "terminal_state.json").read_text())
    assert terminal["terminal_class"] == "blocked"
    assert terminal["blocked_reason"] == "acp_environment_unavailable"
    assert terminal["failure_code"] == "ORIGIN_BACKEND_UNAVAILABLE"
    assert terminal["blocked_needs"]
    assert campaign_status(root)["four_class_denominators"]["blocked"] == 2
    state["available"] = True
    second = run_campaign(manifest_path, root, backends=backends)
    assert second.n_blocked == 0
    assert second.n_executed == 2  # blocked cases were re-attempted, not skipped
    assert second.n_skipped_idempotent == 0


def test_per_path_gradient_budget_censors_terminal(manifest_path, tmp_path):
    class ExpensiveCorrector(SyntheticCorrector):
        def __call__(self, guess, targets, attempt_id):
            record = super().__call__(guess, targets, attempt_id)
            record["n_gradient_evaluations"] = 5000
            return record

    backends = CampaignBackends(
        prepare_origin=identity_origin,
        make_corrector=lambda plan, case_root: ExpensiveCorrector(plan))
    manifest = json.loads(manifest_path.read_text())
    manifest["budget"]["max_gradient_calls_per_path"] = 100
    manifest["budget"]["max_total_gradient_calls"] = 10_000_000
    del manifest["content_sha256"]
    from pes2ts_core.generation.planning.synchronized_path import digest
    manifest["content_sha256"] = digest(manifest)
    shrink = tmp_path / "censored_manifest.json"
    shrink.write_text(json.dumps(manifest))
    root = tmp_path / "censored_root"
    summary = run_campaign(shrink, root, backends=backends)
    assert summary.terminal_counts["censored_budget"] == 2
    terminal = json.loads((root / "RXN_0000017762" / "terminal_state.json").read_text())
    assert terminal["failure_code"] == "PATH_GRADIENT_BUDGET_EXCEEDED"
    assert terminal["label_state"] == "censored_budget"
    assert (root / "RXN_0000017762" / "continuation_result.json").is_file()


def test_campaign_total_budget_blocks_remaining_cases(manifest_path, tmp_path):
    manifest = json.loads(manifest_path.read_text())
    manifest["budget"]["max_total_gradient_calls"] = 1
    del manifest["content_sha256"]
    from pes2ts_core.generation.planning.synchronized_path import digest
    manifest["content_sha256"] = digest(manifest)
    path = tmp_path / "tiny_budget_manifest.json"
    path.write_text(json.dumps(manifest))
    root = tmp_path / "tiny"
    summary = run_campaign(path, root, backends=synthetic_backends())
    # First case consumes the budget; remaining cases are typed blocked.
    assert summary.n_blocked >= 1
    blocked = [json.loads(p.read_text()) for p in sorted(root.rglob("terminal_state.json"))
               if json.loads(p.read_text())["terminal_class"] == "blocked"]
    assert all(doc["blocked_reason"] == BLOCKED_CAMPAIGN_BUDGET for doc in blocked)
    assert all(doc["blocked_needs"] for doc in blocked)


def test_budget_exhausted_resume_preserves_completed_case(manifest_path, tmp_path):
    """Regression: resume after budget exhaustion must never re-block finished work."""
    path = tiny_total_budget_manifest(manifest_path, tmp_path)
    root = tmp_path / "exhausted_resume"
    first = run_campaign(path, root, case_filter=["RXN_0000017762"],
                         backends=synthetic_backends())
    assert first.n_executed == 1
    case_a = root / "RXN_0000017762"
    terminal_bytes = (case_a / "terminal_state.json").read_bytes()
    costs_bytes = (case_a / "costs.json").read_bytes()
    assert json.loads(terminal_bytes)["terminal_class"] != "blocked"
    assert json.loads(costs_bytes)["aggregate"]["N_gradient"] >= 1
    # Case A alone has consumed the frozen total budget; the resume must still
    # skip it idempotently instead of overwriting it with a blocked terminal.
    resumed = run_campaign(path, root, backends=synthetic_backends())
    assert resumed.n_executed == 0
    assert resumed.n_skipped_idempotent == 1
    assert resumed.n_blocked == 1  # the never-run case, not the completed one
    assert (case_a / "terminal_state.json").read_bytes() == terminal_bytes
    assert (case_a / "costs.json").read_bytes() == costs_bytes


def test_budget_exhausted_never_run_case_gets_blocked_needs(manifest_path, tmp_path):
    """A case with no terminal still gets the blocked campaign-budget terminal."""
    path = tiny_total_budget_manifest(manifest_path, tmp_path)
    root = tmp_path / "exhausted_block"
    first = run_campaign(path, root, backends=synthetic_backends())
    assert first.n_executed == 1
    assert first.n_blocked == 1
    case_b = root / "RXN_0000079731"
    blocked_bytes = (case_b / "terminal_state.json").read_bytes()
    blocked = json.loads(blocked_bytes)
    assert blocked["terminal_class"] == "blocked"
    assert blocked["blocked_reason"] == BLOCKED_CAMPAIGN_BUDGET
    assert blocked["blocked_needs"][0].startswith(
        "campaign budget exhausted: N_gradient=")
    assert blocked["blocked_needs"][1] == \
        "split the remaining cases into a NEW campaign manifest"
    costs = json.loads((case_b / "costs.json").read_text())
    assert costs["ledgers"] == []
    assert costs["aggregate"] is None
    # The resume keeps the completed case skipped and the blocked case untouched.
    resumed = run_campaign(path, root, backends=synthetic_backends())
    assert resumed.n_executed == 0
    assert resumed.n_skipped_idempotent == 1
    assert resumed.n_blocked == 1
    assert (case_b / "terminal_state.json").read_bytes() == blocked_bytes


def test_manifest_edit_in_same_root_is_refused(manifest_path, tmp_path):
    root = tmp_path / "campaign"
    run_campaign(manifest_path, root, backends=synthetic_backends())
    manifest = json.loads(manifest_path.read_text())
    manifest["frozen_parameters"]["candidate_top_k"] = 1
    del manifest["content_sha256"]
    from pes2ts_core.generation.planning.synchronized_path import digest
    manifest["content_sha256"] = digest(manifest)
    edited = tmp_path / "edited_manifest.json"
    edited.write_text(json.dumps(manifest))
    with pytest.raises(CampaignError, match="CAMPAIGN_ROOT_IDENTITY_MISMATCH"):
        run_campaign(edited, root, backends=synthetic_backends())


def test_case_filter_and_unknown_case(manifest_path, tmp_path):
    root = tmp_path / "single"
    summary = run_campaign(manifest_path, root, case_filter=["RXN_0000017762"],
                           backends=synthetic_backends())
    assert summary.n_cases == 1
    assert (root / "RXN_0000017762" / "terminal_state.json").is_file()
    assert not (root / "RXN_0000079731").exists()
    with pytest.raises(CampaignError, match="CASE_NOT_IN_CAMPAIGN"):
        run_campaign(manifest_path, root, case_filter=["RXN_0009999999"],
                     backends=synthetic_backends())


def test_attach_validation_result_keeps_chemistry_and_engineering_separate():
    terminal = {"schema_version": TERMINAL_STATE_SCHEMA, "terminal_class": "partial_prefix",
                "label_state": "unattempted", "validation": {"status": BLOCKED_HUMAN_REVIEW}}
    failed = attach_validation_result(terminal, {"status": "failed", "object_id": "v:1"})
    assert failed["terminal_class"] == "validated_failed"
    assert failed["label_state"] == "verified_non_target"
    assert failed["validation"]["preparation_layer_status"] == "absent"
    passed = attach_validation_result(
        terminal, {"status": "passed", "object_id": "v:2",
                   "optts": {"attempt_id": "a1"}, "frequency": {"attempt_id": "a1"},
                   "irc_forward": {"attempt_id": "a2"},
                   "irc_reverse": {"attempt_id": "a3"}})
    assert passed["label_state"] == "verified_target"
    assert passed["validation"]["evidence_refs"]["irc_forward"] == "a2"
    with pytest.raises(CampaignError, match="INVALID_VALIDATION_STATUS"):
        attach_validation_result(terminal, {"status": "exploded"})


def test_interrupted_run_resumes_without_recomputing_completed_cases(manifest_path,
                                                                     tmp_path):
    root = tmp_path / "resumed"
    run_campaign(manifest_path, root, case_filter=["RXN_0000017762"],
                 backends=synthetic_backends())
    calls = []

    def counting_origin(geometry, elements, bond_indices, charge, multiplicity,
                        case_root):
        calls.append(str(case_root))
        return identity_origin(geometry, elements, bond_indices, charge,
                               multiplicity, case_root)

    backends = CampaignBackends(
        prepare_origin=counting_origin,
        make_corrector=lambda plan, case_root: SyntheticCorrector(plan))
    summary = run_campaign(manifest_path, root, backends=backends)
    assert summary.n_skipped_idempotent == 1
    assert summary.n_executed == 1
    assert calls == [str(root / "RXN_0000079731")]  # completed case never re-ran


def test_full_pilot_manifest_builds_from_all_24_fixtures(tmp_path):
    snapshots = sorted(FIXTURES.glob("RXN_*.json"))
    assert len(snapshots) == 24
    manifest = build_campaign_manifest(snapshots, pilot_cases=PILOT,
                                       continuation_policy=synthetic_policy(),
                                       budget={"max_frames_per_path":
                                               synthetic_policy().max_frames,
                                               "max_attempts_per_path":
                                               synthetic_policy().max_attempts,
                                               "max_wall_seconds_per_path":
                                               synthetic_policy().max_seconds})
    assert len(manifest["cases"]) == 24
    assert manifest["pilot_cases"] == PILOT
    assert set(manifest["review_status"].values()) == {"needs_review"}
