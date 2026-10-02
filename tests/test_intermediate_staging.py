"""Offline tests for intermediate-evidence staging (plan todo 27).

Locks the todo-27 acceptance surface (design §8.2/§10.1):

* G1 pre-freezes a typed intermediate exploration/adjudication **rule set**
  (deterministic digest; four rule families; each rule a typed record);
* stable intermediate evidence → a **new plan version** carrying the
  intermediate hash plus adjacent **segments** (multi-peak paths never
  merge into one TS);
* insufficient evidence (no stable structure / no topology confirmation) →
  **typed refusal**, no segmentation;
* "coordinate temporarily unchanged" alone → typed refusal
  (``COORDINATE_UNCHANGED_NOT_MECHANISM``) — a schedule is not a mechanism;
* the **original plan is immutable** (hash + dict unchanged) while the
  derived superseded copy completes a contracts_v2-valid ``supersedes``
  chain;
* the failure-tree vocabulary and budget denominator **replay across
  versions** (todo-23 ``FAILURE_TREE_CODE_ORDER`` / ``BudgetLedger`` linkage);
* G2 never self-segments: staging runs structurally on the G1 side and
  copies every candidate driver verbatim.

Fixtures reuse the todo-23/26 synthetic plans (scan + path); no test reads
the real ``data/`` tree.
"""

from __future__ import annotations

import copy
import inspect
from typing import Any

import pytest

from test_generation_plan_freeze import _frozen_plan  # noqa: E402
from test_path_request import SHA_A, _request  # noqa: E402

from pes2ts_core.contracts import FORBIDDEN_TRUTH_KEYS
from pes2ts_core.g1.v2_verify import FORBIDDEN_EXPORT_KEYS
from pes2ts_core.generation.planning.contracts_v2 import (
    CANDIDATE_KIND_PATH,
    validate_v2_document,
)
from pes2ts_core.generation.planning.intermediate_staging import (
    CODE_COORDINATE_UNCHANGED_NOT_MECHANISM,
    CODE_ENERGY_LOCAL_MINIMUM_MISSING,
    CODE_HESSIAN_CONTRADICTS_STABILITY,
    CODE_INSUFFICIENT_STABILITY_EVIDENCE,
    CODE_INSUFFICIENT_TOPOLOGY_EVIDENCE,
    CODE_INTERMEDIATE_GEOMETRY_MISSING,
    CODE_NO_STABLE_INTERMEDIATE,
    CODE_PLAN_NOT_FROZEN,
    FORMAL_CONFIRMED,
    FORMAL_CONTRADICTED,
    FORMAL_PENDING_EVIDENCE,
    FROZEN_INTERMEDIATE_RULES,
    HESSIAN_COMPLETE,
    HESSIAN_PENDING,
    REF_PRODUCT,
    REF_REACTANT,
    RULE_CLASS_CANDIDATE_FRAME,
    RULE_CLASS_CONSTRAINT_RELEASE,
    RULE_CLASS_HESSIAN_FREQUENCY,
    RULE_CLASS_TOPOLOGY_CHARGE_ELECTRONIC,
    RULE_CLASSES,
    STAGING_REFUSAL_CODES,
    STAGING_RULES_VERSION,
    STAGING_RUNS_ON,
    STAGING_STAGE_ID,
    ConfirmationEvidence,
    HessianEvidence,
    IntermediateEvidence,
    IntermediateStagingError,
    OptimizationEvidence,
    adjudicate_intermediate,
    build_adjacent_segments,
    confirmed_intermediate_sha256,
    frozen_intermediate_rules,
    parse_intermediate_evidence,
    stage_new_plan_version,
)
from pes2ts_core.generation.planning.path_request import (
    INTERMEDIATE_NEXT_STAGE,
    flag_intermediate_minimum,
)
from pes2ts_core.generation.planning.plan_freeze import (
    BUDGET_ACCOUNTING_CATEGORIES,
    FAILURE_TREE_CODE_ORDER,
    FAILURE_TREE_VERSION,
    BudgetLedger,
    FailureState,
    active_failure_codes,
    freeze_path_candidate_plan,
    verify_generation_plan,
)

FORBIDDEN: frozenset[str] = frozenset(
    {key.lower() for key in FORBIDDEN_TRUTH_KEYS}
    | {key.lower() for key in FORBIDDEN_EXPORT_KEYS}
)


# ---------------------------------------------------------------------------
# Evidence fixtures.
# ---------------------------------------------------------------------------
def _stable_evidence(frame_index: int = 2, **overrides: Any) -> IntermediateEvidence:
    fields: dict[str, Any] = {
        "frame_index": frame_index,
        "energy_local_minimum": True,
        "optimization": OptimizationEvidence(
            released_from_constraints=True,
            converged=True,
            stable_structure=True,
        ),
        "confirmation": ConfirmationEvidence(
            topology_confirmed=True,
            charge_confirmed=True,
            electronic_state_confirmed=True,
        ),
        "hessian": HessianEvidence(status=HESSIAN_PENDING),
        "coordinate_unchanged_claim": False,
        "depth": 0.8,
        "deep_intermediate": True,
        "source": "neb",
    }
    fields.update(overrides)
    return IntermediateEvidence(**fields)


def _path_plan() -> dict[str, Any]:
    request = _request(reaction_id="RXN_0000000001")
    candidate = request.to_path_candidate(candidate_id="cand-path-27")
    return freeze_path_candidate_plan(
        candidate,
        reaction_id="RXN_0000000001",
        case_id="case-0001",
        split="train",
        source_case_sha256=SHA_A,
        endpoint_graph_sha256=request.bundle_content_sha256,
    )


def _geom4(shift: float = 0.0) -> list[list[float]]:
    """4-row intermediate geometry (SINGLE_BOND n_atoms=4).

    Shifts keep intermediates distinct from both endpoint blocks (R atoms
    3/4 at x=6.0/7.54; P atoms 3/4 at x=3.08/4.62).
    """
    return [
        [0.0, 0.0, 0.0],
        [1.54, 0.0, 0.0],
        [4.0 + shift, 0.0, 0.0],
        [5.54 + shift, 0.0, 0.0],
    ]


def _walk_forbidden(value: Any, path: str = "$") -> list[str]:
    hits: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN:
                hits.append(f"{path}.{key}")
            hits.extend(_walk_forbidden(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            hits.extend(_walk_forbidden(child, f"{path}[{index}]"))
    return hits


# ---------------------------------------------------------------------------
# Frozen rule set (G1 pre-freezes before any G2 execution).
# ---------------------------------------------------------------------------
def test_frozen_rule_set_is_deterministic_across_calls() -> None:
    first = frozen_intermediate_rules()
    second = frozen_intermediate_rules()
    assert first.rules_version == STAGING_RULES_VERSION
    assert first.content_sha256 == second.content_sha256
    assert first.content_sha256 == FROZEN_INTERMEDIATE_RULES.content_sha256
    assert first.to_doc() == second.to_doc()


def test_rule_set_covers_design_step_vocabulary_as_typed_records() -> None:
    rules = frozen_intermediate_rules()
    classes = {rule.rule_class for rule in rules.rules}
    assert classes == set(RULE_CLASSES)
    assert RULE_CLASS_CANDIDATE_FRAME in classes
    assert RULE_CLASS_CONSTRAINT_RELEASE in classes
    assert RULE_CLASS_TOPOLOGY_CHARGE_ELECTRONIC in classes
    assert RULE_CLASS_HESSIAN_FREQUENCY in classes
    for rule in rules.rules:
        assert rule.rule_id
        assert rule.description
        assert isinstance(rule.required_for_segmentation, bool)
    # Three rules gate segmentation; Hessian/frequency is the formal channel.
    required = {
        rule.rule_id for rule in rules.rules if rule.required_for_segmentation
    }
    assert "R_INTERMEDIATE_HESSIAN_FREQUENCY_FORMAL" not in required
    assert len(required) == 3
    assert rules.rules_by_class(RULE_CLASS_HESSIAN_FREQUENCY)[0].required_for_segmentation is False


def test_rule_set_documents_are_deterministic_and_pure() -> None:
    doc = frozen_intermediate_rules().to_doc()
    assert _walk_forbidden(doc) == []
    assert doc["content_sha256"] == frozen_intermediate_rules().content_sha256
    again = frozen_intermediate_rules().to_doc()
    assert stable_dumps_equal(doc, again)


def stable_dumps_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    from pes2ts_core.utils.hashing import stable_json_dumps

    return stable_json_dumps(a) == stable_json_dumps(b)


# ---------------------------------------------------------------------------
# Cross-module handoff (todo-26 seam).
# ---------------------------------------------------------------------------
def test_path_request_flag_next_stage_points_at_staging() -> None:
    assert STAGING_STAGE_ID == INTERMEDIATE_NEXT_STAGE == "intermediate_evidence_staging"
    flag = flag_intermediate_minimum([0.0, -1.0, 0.2, 1.5], deep_threshold=0.5)
    assert flag.next_stage == STAGING_STAGE_ID
    assert flag.deep_intermediate is True


# ---------------------------------------------------------------------------
# Adjudication rules (typed verdicts + typed refusals).
# ---------------------------------------------------------------------------
def test_adjudication_confirms_stable_evidence() -> None:
    adjudication = adjudicate_intermediate(_stable_evidence(frame_index=3))
    assert adjudication.confirmed is True
    assert adjudication.verdict == "confirmed_stable"
    assert adjudication.refusal_code is None
    assert adjudication.formal_conclusion == FORMAL_PENDING_EVIDENCE
    assert adjudication.frame_index == 3
    assert adjudication.evidence_sha256 == _stable_evidence(frame_index=3).evidence_sha256()


def test_adjudication_hessian_pending_stays_pending_evidence() -> None:
    evidence = _stable_evidence(hessian=None)
    adjudication = adjudicate_intermediate(evidence)
    assert adjudication.confirmed is True
    assert adjudication.formal_conclusion == FORMAL_PENDING_EVIDENCE


def test_adjudication_hessian_zero_imaginary_confirms_formally() -> None:
    evidence = _stable_evidence(
        hessian=HessianEvidence(status=HESSIAN_COMPLETE, n_imaginary=0, reference="freq-1")
    )
    adjudication = adjudicate_intermediate(evidence)
    assert adjudication.confirmed is True
    assert adjudication.formal_conclusion == FORMAL_CONFIRMED


def test_adjudication_hessian_imaginary_modes_contradict_and_refuse() -> None:
    evidence = _stable_evidence(
        hessian=HessianEvidence(status=HESSIAN_COMPLETE, n_imaginary=1)
    )
    adjudication = adjudicate_intermediate(evidence)
    assert adjudication.confirmed is False
    assert adjudication.refusal_code == CODE_HESSIAN_CONTRADICTS_STABILITY
    assert adjudication.formal_conclusion == FORMAL_CONTRADICTED


def test_adjudication_insufficient_stability_refuses() -> None:
    evidence = _stable_evidence(
        optimization=OptimizationEvidence(
            released_from_constraints=True, converged=False, stable_structure=False
        )
    )
    adjudication = adjudicate_intermediate(evidence)
    assert adjudication.refusal_code == CODE_INSUFFICIENT_STABILITY_EVIDENCE


def test_adjudication_insufficient_topology_refuses() -> None:
    evidence = _stable_evidence(
        confirmation=ConfirmationEvidence(
            topology_confirmed=False, charge_confirmed=True, electronic_state_confirmed=True
        )
    )
    adjudication = adjudicate_intermediate(evidence)
    assert adjudication.refusal_code == CODE_INSUFFICIENT_TOPOLOGY_EVIDENCE


def test_adjudication_coordinate_unchanged_alone_refuses() -> None:
    evidence = _stable_evidence(
        energy_local_minimum=False,
        coordinate_unchanged_claim=True,
        optimization=OptimizationEvidence(
            released_from_constraints=False, converged=False, stable_structure=False
        ),
        confirmation=ConfirmationEvidence(
            topology_confirmed=False, charge_confirmed=False, electronic_state_confirmed=False
        ),
    )
    adjudication = adjudicate_intermediate(evidence)
    assert adjudication.confirmed is False
    assert adjudication.refusal_code == CODE_COORDINATE_UNCHANGED_NOT_MECHANISM


def test_adjudication_energy_minimum_missing_refuses_without_claim() -> None:
    evidence = _stable_evidence(energy_local_minimum=False)
    adjudication = adjudicate_intermediate(evidence)
    assert adjudication.refusal_code == CODE_ENERGY_LOCAL_MINIMUM_MISSING


def test_adjudication_coordinate_unchanged_with_full_stability_still_confirms() -> None:
    # The unchanged claim never *drives* segmentation — stability evidence does.
    evidence = _stable_evidence(coordinate_unchanged_claim=True)
    adjudication = adjudicate_intermediate(evidence)
    assert adjudication.confirmed is True
    assert adjudication.refusal_code is None


def test_parse_intermediate_evidence_accepts_mapping_and_dataclass() -> None:
    evidence = _stable_evidence(frame_index=4)
    revived = parse_intermediate_evidence(evidence.to_doc())
    assert revived == evidence
    assert parse_intermediate_evidence(evidence) is evidence
    with pytest.raises(IntermediateStagingError) as excinfo:
        parse_intermediate_evidence({"frame_index": -1, "energy_local_minimum": True})
    assert excinfo.value.code  # typed


# ---------------------------------------------------------------------------
# Segments: adjacent pairs, never merged into one TS.
# ---------------------------------------------------------------------------
def test_build_adjacent_segments_single_intermediate() -> None:
    adjudications = [adjudicate_intermediate(_stable_evidence(frame_index=2))]
    segments = build_adjacent_segments(adjudications)
    assert len(segments) == 2
    assert segments[0].from_ref == REF_REACTANT
    assert segments[0].to_ref == "I:2"
    assert segments[1].from_ref == "I:2"
    assert segments[1].to_ref == REF_PRODUCT
    assert all(segment.adjacent for segment in segments)


def test_build_adjacent_segments_multi_peak_never_merged() -> None:
    adjudications = [
        adjudicate_intermediate(_stable_evidence(frame_index=2)),
        adjudicate_intermediate(_stable_evidence(frame_index=5, depth=1.2)),
    ]
    segments = build_adjacent_segments(adjudications)
    assert len(segments) == 3
    refs = [(s.from_ref, s.to_ref) for s in segments]
    assert refs == [(REF_REACTANT, "I:2"), ("I:2", "I:5"), ("I:5", REF_PRODUCT)]
    # NEVER merged into a single R→P segment.
    assert (REF_REACTANT, REF_PRODUCT) not in refs
    assert all(segment.adjacent for segment in segments)


def test_confirmed_intermediate_sha256_changes_with_evidence() -> None:
    a = [adjudicate_intermediate(_stable_evidence(frame_index=2))]
    b = [adjudicate_intermediate(_stable_evidence(frame_index=2, depth=9.9))]
    c = [adjudicate_intermediate(_stable_evidence(frame_index=2))]
    assert confirmed_intermediate_sha256(a) == confirmed_intermediate_sha256(c)
    assert confirmed_intermediate_sha256(a) != confirmed_intermediate_sha256(b)


# ---------------------------------------------------------------------------
# stage_new_plan_version: scan plan happy path.
# ---------------------------------------------------------------------------
def test_stable_evidence_stages_new_scan_plan_version_with_hash_and_segments() -> None:
    original = _frozen_plan()
    original_hash = str(original["content_sha256"])
    result = stage_new_plan_version(original, _stable_evidence(frame_index=2))

    new_plan = result.new_plan
    assert new_plan["plan_version"] == original["plan_version"] + 1
    assert new_plan["plan_id"] == f"{original['reaction_id']}:gp-v{new_plan['plan_version']}"
    assert new_plan["supersedes"] == original["plan_id"]
    assert new_plan["status"] == "frozen"
    assert validate_v2_document(new_plan) == []
    assert verify_generation_plan(new_plan) == []

    staging = new_plan["extensions"]["staging"]
    assert staging["intermediate_sha256"] == result.intermediate_sha256
    assert staging["intermediate_sha256"] == confirmed_intermediate_sha256(
        [adjudicate_intermediate(_stable_evidence(frame_index=2))]
    )
    assert staging["n_segments"] == 2
    assert [s["from_ref"] for s in staging["segments"]] == [REF_REACTANT, "I:2"]
    assert [s["to_ref"] for s in staging["segments"]] == ["I:2", REF_PRODUCT]
    assert staging["rules_version"] == STAGING_RULES_VERSION
    assert staging["rules_sha256"] == FROZEN_INTERMEDIATE_RULES.content_sha256
    assert staging["confirmed_intermediates"][0]["frame_index"] == 2
    assert staging["confirmed_intermediates"][0]["formal_conclusion"] == FORMAL_PENDING_EVIDENCE
    assert result.confirmed_frame_indices == (2,)
    assert original_hash == original["content_sha256"]


def test_staged_scan_segment_candidates_keep_drivers_verbatim() -> None:
    original = _frozen_plan()
    result = stage_new_plan_version(original, _stable_evidence())
    original_drivers = original["candidates"][0]["drivers"]
    segment_candidates = [
        c for c in result.new_plan["candidates"] if ":seg-" in str(c["candidate_id"])
    ]
    assert len(segment_candidates) == 2
    for candidate in segment_candidates:
        assert candidate["drivers"] == original_drivers
        assert candidate["mode"] == original["candidates"][0]["mode"]
        assert candidate["lambda_values"] == original["candidates"][0]["lambda_values"]
        assert candidate["failure_reasons"] == []
        assert "staging_segment" in candidate["extensions"]


# ---------------------------------------------------------------------------
# stage_new_plan_version: path plan + multi-peak.
# ---------------------------------------------------------------------------
def test_path_plan_multi_peak_stages_adjacent_segments_with_geometry() -> None:
    original = _path_plan()
    assert original["candidates"][0]["candidate_kind"] == CANDIDATE_KIND_PATH
    evidence = [
        _stable_evidence(
            frame_index=2,
            optimized_geometry=tuple(tuple(row) for row in _geom4(0.0)),
        ),
        _stable_evidence(
            frame_index=5,
            depth=1.2,
            optimized_geometry=tuple(tuple(row) for row in _geom4(1.5)),
        ),
    ]
    result = stage_new_plan_version(original, evidence)
    new_plan = result.new_plan
    assert validate_v2_document(new_plan) == []
    assert verify_generation_plan(new_plan) == []

    staging = new_plan["extensions"]["staging"]
    assert staging["n_segments"] == 3
    assert staging["never_merged_into_single_ts"] is True
    assert len(staging["confirmed_intermediates"]) == 2
    assert result.confirmed_frame_indices == (2, 5)

    segment_candidates = [
        c for c in new_plan["candidates"] if ":seg-" in str(c["candidate_id"])
    ]
    assert len(segment_candidates) == 3
    # Segment-local endpoint blocks: R→I1, I1→I2, I2→P (never full R→P).
    first = segment_candidates[0]
    assert first["endpoint_geometries"]["reactant"] == [
        list(row) for row in original["candidates"][0]["endpoint_geometries"]["reactant"]
    ]
    assert first["endpoint_geometries"]["product"] == _geom4(0.0)
    last = segment_candidates[-1]
    assert last["endpoint_geometries"]["reactant"] == _geom4(1.5)
    assert last["endpoint_geometries"]["product"] == [
        list(row) for row in original["candidates"][0]["endpoint_geometries"]["product"]
    ]
    # No segment candidate carries the un-split full path endpoints.
    for candidate in segment_candidates:
        endpoints = candidate["endpoint_geometries"]
        full_r = original["candidates"][0]["endpoint_geometries"]["reactant"]
        full_p = original["candidates"][0]["endpoint_geometries"]["product"]
        assert not (
            endpoints["reactant"] == full_r and endpoints["product"] == full_p
        )
    # PathCandidateV1 carries no drivers; frozen execution dimensions stay verbatim.
    original_primary = original["candidates"][0]
    for candidate in segment_candidates:
        assert candidate["image_chain"] == original_primary["image_chain"]
        assert candidate["method_kind"] == original_primary["method_kind"]
        assert candidate["n_atoms"] == original_primary["n_atoms"]
        assert candidate["anchor_reason"] == original_primary["anchor_reason"]
        assert candidate["direction"] == original_primary["direction"]


def test_path_plan_without_intermediate_geometry_refuses() -> None:
    original = _path_plan()
    with pytest.raises(IntermediateStagingError) as excinfo:
        stage_new_plan_version(original, _stable_evidence(frame_index=2))
    assert excinfo.value.code == CODE_INTERMEDIATE_GEOMETRY_MISSING


# ---------------------------------------------------------------------------
# Typed refusals: no segmentation without stable-intermediate evidence.
# ---------------------------------------------------------------------------
def test_insufficient_stability_refuses_no_segmentation() -> None:
    original = _frozen_plan()
    evidence = _stable_evidence(
        optimization=OptimizationEvidence(
            released_from_constraints=True, converged=True, stable_structure=False
        )
    )
    with pytest.raises(IntermediateStagingError) as excinfo:
        stage_new_plan_version(original, evidence)
    assert excinfo.value.code == CODE_INSUFFICIENT_STABILITY_EVIDENCE


def test_insufficient_topology_refuses_no_segmentation() -> None:
    original = _frozen_plan()
    evidence = _stable_evidence(
        confirmation=ConfirmationEvidence(
            topology_confirmed=False, charge_confirmed=True, electronic_state_confirmed=True
        )
    )
    with pytest.raises(IntermediateStagingError) as excinfo:
        stage_new_plan_version(original, evidence)
    assert excinfo.value.code == CODE_INSUFFICIENT_TOPOLOGY_EVIDENCE


def test_coordinate_unchanged_alone_refuses_staging() -> None:
    original = _frozen_plan()
    evidence = _stable_evidence(
        energy_local_minimum=False,
        coordinate_unchanged_claim=True,
        optimization=OptimizationEvidence(
            released_from_constraints=False, converged=False, stable_structure=False
        ),
    )
    with pytest.raises(IntermediateStagingError) as excinfo:
        stage_new_plan_version(original, evidence)
    assert excinfo.value.code == CODE_COORDINATE_UNCHANGED_NOT_MECHANISM


def test_empty_evidence_list_refuses_no_stable_intermediate() -> None:
    original = _frozen_plan()
    with pytest.raises(IntermediateStagingError) as excinfo:
        stage_new_plan_version(original, [])
    assert excinfo.value.code == CODE_NO_STABLE_INTERMEDIATE


def test_all_refused_evidence_raises_first_refusal_code() -> None:
    original = _frozen_plan()
    evidence = [
        _stable_evidence(energy_local_minimum=False),
        _stable_evidence(
            optimization=OptimizationEvidence(
                released_from_constraints=False, converged=False, stable_structure=False
            )
        ),
    ]
    with pytest.raises(IntermediateStagingError) as excinfo:
        stage_new_plan_version(original, evidence)
    assert excinfo.value.code == CODE_ENERGY_LOCAL_MINIMUM_MISSING


def test_non_frozen_original_refuses() -> None:
    original = _frozen_plan()
    superseded_like = copy.deepcopy(original)
    superseded_like["status"] = "superseded"
    superseded_like["supersedes"] = "RXN_0000000001:gp-v99"
    from pes2ts_core.contracts import seal_document

    superseded_like = seal_document(superseded_like)
    with pytest.raises(IntermediateStagingError) as excinfo:
        stage_new_plan_version(superseded_like, _stable_evidence())
    assert excinfo.value.code == CODE_PLAN_NOT_FROZEN


def test_unknown_refusal_code_is_rejected_at_construction() -> None:
    with pytest.raises(IntermediateStagingError) as excinfo:
        IntermediateStagingError("NOT_A_REAL_CODE")
    assert excinfo.value.code == "STAGING_INPUT_INVALID"
    assert "NOT_A_REAL_CODE" in str(excinfo.value)
    # The typed vocabulary is closed.
    assert CODE_PLAN_NOT_FROZEN in STAGING_REFUSAL_CODES
    assert CODE_COORDINATE_UNCHANGED_NOT_MECHANISM in STAGING_REFUSAL_CODES


# ---------------------------------------------------------------------------
# Original plan immutable + supersede chain valid via contracts_v2.
# ---------------------------------------------------------------------------
def test_original_plan_immutable_and_supersede_chain_valid() -> None:
    original = _frozen_plan()
    snapshot = copy.deepcopy(original)
    result = stage_new_plan_version(original, _stable_evidence(frame_index=2))

    # Original dict never mutated; hash unchanged.
    assert original == snapshot
    assert original["content_sha256"] == snapshot["content_sha256"]
    assert original["status"] == "frozen"
    assert "supersedes" not in original

    # Derived superseded copy completes a contracts_v2-valid chain.
    superseded = result.superseded_original
    assert superseded["status"] == "superseded"
    assert superseded["supersedes"] == result.new_plan["plan_id"]
    assert superseded["plan_id"] == original["plan_id"]
    assert validate_v2_document(superseded) == []
    # Chain direction: new frozen plan references the plan it replaces.
    assert result.new_plan["supersedes"] == original["plan_id"]
    assert result.new_plan["status"] == "frozen"


def test_superseded_original_without_reference_would_fail_contract() -> None:
    # Sanity: the chain field is what contracts_v2 requires for superseded.
    original = _frozen_plan()
    result = stage_new_plan_version(original, _stable_evidence())
    broken = copy.deepcopy(result.superseded_original)
    broken.pop("supersedes", None)
    from pes2ts_core.contracts import seal_document

    broken = seal_document(broken)
    problems = validate_v2_document(broken)
    assert any("supersedes" in problem for problem in problems)


# ---------------------------------------------------------------------------
# Failure tree + budget replay across versions (todo-23 vocabulary link).
# ---------------------------------------------------------------------------
def test_failure_tree_and_budget_replay_across_versions() -> None:
    original = _frozen_plan()
    result = stage_new_plan_version(original, _stable_evidence())
    new_plan = result.new_plan

    freeze = new_plan["extensions"]["freeze"]
    assert freeze["failure_tree"]["version"] == FAILURE_TREE_VERSION
    assert freeze["failure_tree"]["codes"] == list(FAILURE_TREE_CODE_ORDER)

    staging = new_plan["extensions"]["staging"]
    assert staging["failure_tree_replay"]["codes"] == list(FAILURE_TREE_CODE_ORDER)
    assert staging["budget_replay"]["from_plan_id"] == original["plan_id"]
    assert staging["budget_replay"]["from_plan_content_sha256"] == original["content_sha256"]
    assert staging["budget_replay"]["budget"] == original["budget"]
    assert staging["budget_replay"]["budget_accounting"] == list(BUDGET_ACCOUNTING_CATEGORIES)
    # Budget denominator carried verbatim into the new version.
    assert new_plan["budget"] == original["budget"]

    # Failure predicates replay against the same closed vocabulary.
    assert active_failure_codes(FailureState()) == ()
    assert "INTERMEDIATE_CANDIDATE" in FAILURE_TREE_CODE_ORDER
    ledger = BudgetLedger(max_attempts=int(original["budget"]["max_attempts"]))
    exhausted = ledger.record(attempts=int(original["budget"]["max_attempts"]))
    codes = active_failure_codes(FailureState(budget=exhausted, intermediate_candidate=True))
    assert "BUDGET_EXHAUSTED" in codes
    assert "INTERMEDIATE_CANDIDATE" in codes
    # Every replayed code is inside the vocabulary the staged plan froze.
    for code in codes:
        assert code in freeze["failure_tree"]["codes"]


# ---------------------------------------------------------------------------
# G2 structural guarantee: staging never self-segments / never changes drivers.
# ---------------------------------------------------------------------------
def test_staging_surface_is_g1_side_only_no_driver_mutation() -> None:
    signature = inspect.signature(stage_new_plan_version)
    parameter_names = set(signature.parameters)
    assert parameter_names == {"original_plan", "intermediate_evidence", "rules"}
    assert "driver" not in "".join(parameter_names).lower()
    assert STAGING_RUNS_ON == "G1_side_only"

    original = _frozen_plan()
    result = stage_new_plan_version(original, _stable_evidence())
    assert result.staging_runs_on == STAGING_RUNS_ON
    staging = result.new_plan["extensions"]["staging"]
    assert staging["g2_self_segmentation"] is False
    assert staging["drivers_changed_by_staging"] is False
    # Original candidates carried into the new version keep their drivers too.
    for carried in result.new_plan["candidates"]:
        if ":seg-" not in str(carried.get("candidate_id")):
            match = next(
                c for c in original["candidates"] if c["candidate_id"] == carried["candidate_id"]
            )
            assert carried["drivers"] == match["drivers"]


# ---------------------------------------------------------------------------
# Determinism + purity.
# ---------------------------------------------------------------------------
def test_staging_is_deterministic_over_same_inputs() -> None:
    original = _frozen_plan()
    first = stage_new_plan_version(original, _stable_evidence(frame_index=2))
    second = stage_new_plan_version(copy.deepcopy(original), _stable_evidence(frame_index=2))
    assert first.new_plan["content_sha256"] == second.new_plan["content_sha256"]
    assert first.intermediate_sha256 == second.intermediate_sha256
    assert first.segments == second.segments


def test_staged_documents_carry_no_forbidden_truth_keys() -> None:
    original = _frozen_plan()
    result = stage_new_plan_version(original, _stable_evidence())
    for document in (result.new_plan, result.superseded_original, result.to_doc()):
        assert _walk_forbidden(document) == []


def test_evidence_documents_carry_no_forbidden_truth_keys() -> None:
    evidence = _stable_evidence(optimized_geometry=tuple(tuple(r) for r in _geom4()))
    assert _walk_forbidden(evidence.to_doc()) == []
    adjudication = adjudicate_intermediate(evidence)
    assert _walk_forbidden(adjudication.to_doc()) == []


# ---------------------------------------------------------------------------
# Formal conclusion recorded as pending-evidence on the staged version.
# ---------------------------------------------------------------------------
def test_hessian_pending_recorded_as_pending_evidence_on_plan() -> None:
    original = _frozen_plan()
    result = stage_new_plan_version(
        original, _stable_evidence(hessian=HessianEvidence(status=HESSIAN_PENDING))
    )
    staging = result.new_plan["extensions"]["staging"]
    assert staging["confirmed_intermediates"][0]["formal_conclusion"] == FORMAL_PENDING_EVIDENCE
    assert result.formal_conclusions == (FORMAL_PENDING_EVIDENCE,)
    # A formal Hessian confirmation flows through as confirmed.
    confirmed = stage_new_plan_version(
        original,
        _stable_evidence(
            hessian=HessianEvidence(status=HESSIAN_COMPLETE, n_imaginary=0, reference="f1")
        ),
    )
    assert confirmed.formal_conclusions == (FORMAL_CONFIRMED,)


def test_parse_mapping_evidence_round_trip_through_staging() -> None:
    original = _frozen_plan()
    payload = _stable_evidence(frame_index=2).to_doc()
    result = stage_new_plan_version(original, payload)
    assert result.confirmed_frame_indices == (2,)
    assert validate_v2_document(result.new_plan) == []
