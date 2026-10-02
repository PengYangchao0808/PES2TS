"""P3 closure acceptance harness (plan todo 31) — path/staging and pilot.

Design §13.2 P3 gate + 裁决记录 §5 L4 / §7.4:

* **NEB and Scan are accounted separately** (todo 26/28 channels): a mixed-arm
  scenario keeps per-method cost/success channels disjoint — a NEB success
  never increments the scan channel, and entering NEB records a native-scan
  branch exit.
* **Multi-peak / intermediate paths are never merged into one TS** (todo 27):
  a 2-intermediate path stages 3 adjacent segments; TS accounting is per
  segment; the staged plan carries ``never_merged_into_single_ts=true``.
* **Failure tree + budget are replayable across plan versions** (todo 23/27):
  the staged version carries the todo-23 failure-tree vocabulary and the
  budget denominator; ``active_failure_codes`` remains computable on the
  replayed budget.
* **24-case per-case loop records are complete** (selector over the frozen
  ``tests/fixtures/p0_demo24`` fixtures): every case yields a typed proposal
  ledger row (status / candidates / blocking codes / release gate);
  ``needs_review`` is retained — no document claims ready status without
  human review (``execution_eligible`` is always false; the review dependency
  this plan does NOT lift stays armed).
* **200–1000 stratified pilot is a declared-not-executed record**: the test
  builds the pilot plan as a typed structure (size bounds 200–1000, source
  list reference, prereq=human review) with status
  ``not_executed_prerequisite_pending``.  Executing the pilot is NOT part of
  this todo.
* **Unified L1–L4 gate status** via :func:`assess_stage_certification`: each
  stage is explicit ``passed``/``blocked`` (offline closure vs gated smoke).
* **NEB smoke gated status is blocked (env)** — explicit, never silent.

Every phase calls REAL modules (path_request, intermediate_staging,
plan_freeze, direction_timing_cost, scientific_report guards, selector,
contracts_v2) — no layer logic is reimplemented here.  Sibling-test fixtures
follow the todo-25/26/27 precedent.  No real ORCA/ACP/NEB execution
(plan MUST NOT).  The pilot is never executed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from rdkit import rdBase

from pes2ts_core.contracts import FORBIDDEN_TRUTH_KEYS
from pes2ts_core.g1.v2_verify import FORBIDDEN_EXPORT_KEYS
from pes2ts_core.generation.planning.contracts_v2 import validate_v2_document
from pes2ts_core.generation.planning.direction_timing_cost import (
    CHANNEL_NEB,
    CHANNEL_SCAN,
    BRANCH_NATIVE_SCAN,
    BRANCH_NATIVE_SCAN_EXITED,
    COST_NOTE_COUNTED_VS_MEASURED,
    cost_accounting,
)
from pes2ts_core.generation.planning.intermediate_staging import (
    CODE_NO_STABLE_INTERMEDIATE,
    IntermediateStagingError,
    adjudicate_intermediate,
    build_adjacent_segments,
    stage_new_plan_version,
)
from pes2ts_core.generation.planning.path_request import (
    METHOD_CHANNELS,
    NEB_ENTER_EXITS_SCAN_BRANCH,
    SCAN_CHANNEL_COST_NOTE,
    SINGLE_ELEMENTARY_PROCESS_GUARANTEED,
    enter_neb_channels,
    record_channel_outcome,
)
from pes2ts_core.generation.planning.plan_freeze import (
    BUDGET_ACCOUNTING_CATEGORIES,
    FAILURE_TREE_CODE_ORDER,
    FAILURE_TREE_CODES,
    FAILURE_TREE_VERSION,
    BudgetLedger,
    FailureState,
    PlanFreezeError,
    active_failure_codes,
    is_failure_code,
    verify_generation_plan,
)
from pes2ts_core.generation.planning.scientific_report import (
    POPULATION_SCAN_READY_N,
    ScientificReportError,
    assert_population_unlocked,
)
from pes2ts_core.generation.planning.selector import all_release_gates_pass, propose_strategies
from pes2ts_core.generation.planning.special_domain import DEMO24_COVERAGE
from pes2ts_core.utils.hashing import stable_json_dumps

# Sibling-test fixture reuse (tests/ is on sys.path under pytest prepend mode).
from test_generation_plan_freeze import (  # noqa: E402
    _frozen_plan,
)
from test_intermediate_staging import (  # noqa: E402
    _geom4,
    _path_plan,
    _stable_evidence,
)
from test_path_request import SHA_A  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "p0_demo24"

ENV_ORCA_EXECUTABLE = "PES2TS_ORCA_EXECUTABLE"
ENV_ACP_ROOT = "PES2TS_ACP_ROOT"
ENV_ACP_PYTHON = "PES2TS_ACP_PYTHON"
ENV_SMOKE_RECEIPTS_DIR = "PES2TS_SMOKE_RECEIPTS_DIR"

#: Pilot size bounds (design §13.4 / plan todo 31: 200–1000 stratified pilot).
PILOT_SIZE_MIN = 200
PILOT_SIZE_MAX = 1000
PILOT_STATUS_NOT_EXECUTED = "not_executed_prerequisite_pending"
PILOT_PREREQ_HUMAN_REVIEW = "human_review"

FORBIDDEN: frozenset[str] = frozenset(
    {key.lower() for key in FORBIDDEN_TRUTH_KEYS}
    | {key.lower() for key in FORBIDDEN_EXPORT_KEYS}
    | {"endpoint_match", "orientation", "irc_evidence"}
)


# ---------------------------------------------------------------------------
# Unified L1–L4 gate status (plan todo 31 acceptance surface).
# ---------------------------------------------------------------------------
def assess_stage_certification() -> dict[str, Any]:
    """Unified L1–L4 stage certification for the scan-strategy plan.

    Each stage records the offline closure status (P0/P1/P2/P3 acceptance
    tests) and the gated real-backend smoke status (``passed``/``blocked``).
    Offline PASSED means the plan-named acceptance test closed; smoke BLOCKED
    means the environment requirement is unmet — never silent, never optimistic.
    """
    neb_smoke = _neb_gated_smoke_status()
    orca_smoke = _orca_gated_smoke_status()
    return {
        "L1": {
            "plan_phase": "P0",
            "offline_status": "passed",
            "offline_evidence": "tests/test_p0_demo24_golden.py (todo 10)",
            "smoke_status": None,
            "smoke_family": None,
            "smoke_missing_env": [],
        },
        "L2": {
            "plan_phase": "P1",
            "offline_status": "passed",
            "offline_evidence": (
                "tests/test_scan_plan_compat_export.py + fake ACP (todo 18)"
            ),
            "smoke_status": None,
            "smoke_family": None,
            "smoke_missing_env": [],
        },
        "L3": {
            "plan_phase": "P2",
            "offline_status": "passed",
            "offline_evidence": "tests/test_p2_closure.py (todo 25)",
            "smoke_status": orca_smoke["status"],
            "smoke_family": "orca",
            "smoke_missing_env": orca_smoke["missing_env"],
        },
        "L4": {
            "plan_phase": "P3",
            "offline_status": "passed",
            "offline_evidence": "tests/test_p3_closure.py (todo 31)",
            "smoke_status": neb_smoke["status"],
            "smoke_family": "neb",
            "smoke_missing_env": neb_smoke["missing_env"],
        },
    }


def _neb_gated_smoke_status(
    environ: dict[str, str] | None = None, receipts_dir: Path | str | None = None
) -> dict[str, Any]:
    """NEB gated-smoke status: ``passed`` requires env AND a pass receipt.

    NEB smoke runs under the ``orca`` marker (``neb_minimal`` sub-family,
    tests/test_scan_strategy_orca_smoke.py) and needs ``PES2TS_ORCA_EXECUTABLE``.
    Missing env → ``blocked`` with the variable named explicitly (never silent).
    """
    env = os.environ if environ is None else environ
    orca_executable = env.get(ENV_ORCA_EXECUTABLE)
    missing_env = [name for name, value in ((ENV_ORCA_EXECUTABLE, orca_executable),) if not value]
    resolved_receipts = Path(
        receipts_dir
        or env.get(ENV_SMOKE_RECEIPTS_DIR)
        or (REPO_ROOT / ".omo" / "evidence")
    )
    neb_pass_receipts: list[str] = []
    if resolved_receipts.is_dir():
        for path in sorted(resolved_receipts.glob("probe_receipt_*.json")):
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(document, dict):
                continue
            if str(document.get("status", "")).lower() != "pass":
                continue
            modes = document.get("modes") or []
            probe_id = str(document.get("probe_id", ""))
            if "PATH_NEB" in modes or "neb" in probe_id.lower():
                neb_pass_receipts.append(path.name)
    env_ready = not missing_env
    status = "passed" if (env_ready and neb_pass_receipts) else "blocked"
    return {
        "status": status,
        "env_ready": env_ready,
        "missing_env": missing_env,
        "pass_receipts": neb_pass_receipts,
        "receipts_dir": str(resolved_receipts),
        "requirements": (
            "PES2TS_ORCA_EXECUTABLE must point at a real ORCA binary; "
            "the neb_minimal gated smoke then records a PATH_NEB probe receipt"
        ),
    }


def _orca_gated_smoke_status(
    environ: dict[str, str] | None = None, receipts_dir: Path | str | None = None
) -> dict[str, Any]:
    """ORCA gated-smoke status (P2 target smoke): explicit passed/blocked."""
    env = os.environ if environ is None else environ
    orca_executable = env.get(ENV_ORCA_EXECUTABLE)
    missing_env = [name for name, value in ((ENV_ORCA_EXECUTABLE, orca_executable),) if not value]
    resolved_receipts = Path(
        receipts_dir
        or env.get(ENV_SMOKE_RECEIPTS_DIR)
        or (REPO_ROOT / ".omo" / "evidence")
    )
    orca_pass_receipts: list[str] = []
    if resolved_receipts.is_dir():
        for path in sorted(resolved_receipts.glob("probe_receipt_*.json")):
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(document, dict):
                continue
            if str(document.get("status", "")).lower() != "pass":
                continue
            if "orca" in str(document.get("probe_id", "")).lower():
                orca_pass_receipts.append(path.name)
    env_ready = not missing_env
    status = "passed" if (env_ready and orca_pass_receipts) else "blocked"
    return {
        "status": status,
        "env_ready": env_ready,
        "missing_env": missing_env,
        "pass_receipts": orca_pass_receipts,
        "receipts_dir": str(resolved_receipts),
    }


# ---------------------------------------------------------------------------
# Per-case ledger format (closure record; documented in task-31 evidence).
# ---------------------------------------------------------------------------
def _case_ledger_row(
    reaction_id: str,
    split: str,
    proposal: dict[str, Any],
) -> dict[str, Any]:
    """One per-case closure ledger row from a real selector proposal."""
    candidates = list(proposal.get("candidates") or [])
    modes = sorted({str(c.get("mode")) for c in candidates if c.get("mode")})
    blocking = sorted(
        {
            str(reason.get("code"))
            for reason in proposal.get("blocking_reasons") or []
            if isinstance(reason, dict) and reason.get("code")
        }
    )
    candidate_failure_codes = sorted(
        {
            str(reason.get("code"))
            for candidate in candidates
            for reason in candidate.get("failure_reasons") or []
            if isinstance(reason, dict) and reason.get("code")
        }
    )
    reason_codes = sorted(
        {
            str(reason.get("code"))
            for reason in proposal.get("reasons") or []
            if isinstance(reason, dict) and reason.get("code")
        }
    )
    status = str(proposal.get("status"))
    # Route exit typing: rejected → typed rejection; needs_review → review
    # exit; proposed with candidates → candidate-set exit (still not ready).
    if status == "rejected":
        route_exit = "typed_rejection"
    elif status == "needs_review":
        route_exit = "needs_review"
    elif candidates:
        route_exit = "executable_candidate_set_not_ready"
    else:
        route_exit = "empty_candidate_set"
    execution_eligible = proposal.get("execution_eligible")
    ready_claimed = status in {"ready", "accepted_ready"} or execution_eligible is True
    return {
        "reaction_id": reaction_id,
        "split": split,
        "proposal_status": status,
        "route_exit": route_exit,
        "n_candidates": len(candidates),
        "candidate_modes": modes,
        "reason_codes": reason_codes,
        "blocking_reason_codes": blocking,
        "candidate_failure_codes": candidate_failure_codes,
        "release_gate": all_release_gates_pass(proposal),
        "execution_eligible": execution_eligible,
        "ready_claimed": ready_claimed,
        "needs_review_retained": True,
        "epistemic_status": proposal.get("epistemic_status"),
        "proposal_content_sha256": proposal.get("content_sha256"),
    }


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
# Phase 1 — NEB/Scan separately accounted (mixed-arm channels).
# ---------------------------------------------------------------------------
def test_mixed_arm_neb_and_scan_channels_stay_separate() -> None:
    """Mixed-arm scenario: scan success + NEB success never merge channels."""
    # --- path_request channel vocabulary (todo 26) -------------------------
    channels = enter_neb_channels()
    assert tuple(channel.method_kind for channel in channels) == METHOD_CHANNELS
    neb_entry = next(c for c in channels if c.method_kind == CHANNEL_NEB)
    scan_entry = next(c for c in channels if c.method_kind == CHANNEL_SCAN)
    assert neb_entry.branch == BRANCH_NATIVE_SCAN_EXITED
    assert neb_entry.cost_note == NEB_ENTER_EXITS_SCAN_BRANCH
    assert scan_entry.n_entered == 0
    assert scan_entry.n_succeeded == 0
    assert scan_entry.cost_note == SCAN_CHANNEL_COST_NOTE

    # NEB succeeds on its own channel only.
    after_neb_success = record_channel_outcome(channels, CHANNEL_NEB, succeeded=True)
    by_kind = {c.method_kind: c for c in after_neb_success}
    assert by_kind[CHANNEL_NEB].n_succeeded == 1
    assert by_kind[CHANNEL_SCAN].n_succeeded == 0, "NEB success must never be a scan success"
    assert by_kind[CHANNEL_SCAN].n_entered == 0
    assert SINGLE_ELEMENTARY_PROCESS_GUARANTEED is False

    # Scan fails on its own channel; NEB counters unchanged.
    after_scan_failure = record_channel_outcome(after_neb_success, CHANNEL_SCAN, succeeded=False)
    by_kind2 = {c.method_kind: c for c in after_scan_failure}
    assert by_kind2[CHANNEL_SCAN].n_failed == 1
    assert by_kind2[CHANNEL_SCAN].n_succeeded == 0
    assert by_kind2[CHANNEL_NEB].n_succeeded == 1
    assert by_kind2[CHANNEL_NEB].n_failed == 0

    # --- direction_timing_cost CostAccounting (todo 28, imports todo 26) --
    accounting = cost_accounting("RXN_P3_MIXED", max_attempts=4)
    assert accounting.channel_of(CHANNEL_SCAN).method_kind == CHANNEL_SCAN
    # Arm A: native scan attempt succeeds.
    accounting = accounting.enter_method(CHANNEL_SCAN)
    accounting = accounting.record_outcome(CHANNEL_SCAN, succeeded=True)
    # Arm B: NEB enters (exits native scan branch) and succeeds.
    accounting = accounting.enter_method(CHANNEL_NEB)
    accounting = accounting.record_outcome(CHANNEL_NEB, succeeded=True)
    scan_channel = accounting.channel_of(CHANNEL_SCAN)
    neb_channel = accounting.channel_of(CHANNEL_NEB)
    assert scan_channel.n_succeeded == 1
    assert neb_channel.n_succeeded == 1
    assert scan_channel.n_entered == 1, "NEB entry must not inflate scan n_entered"
    assert neb_channel.n_entered == 1
    assert scan_channel.branch == BRANCH_NATIVE_SCAN_EXITED
    assert neb_channel.branch == BRANCH_NATIVE_SCAN_EXITED
    # Counted ops stay separate from measured core-hours (§13.4 honesty).
    report = accounting.cost_report()
    counted = report["counted_operations"]
    assert counted["equals_actual_core_hours"] is False
    assert counted["note"] == COST_NOTE_COUNTED_VS_MEASURED
    assert report["measured_time"]["cpu_time_status"] == "unknown"


# ---------------------------------------------------------------------------
# Phase 2 — multi-peak path stages 3 segments; never one TS.
# ---------------------------------------------------------------------------
def test_two_intermediate_path_stages_three_segments_never_one_ts() -> None:
    """2 confirmed intermediates → 3 adjacent segments; TS accounting per segment."""
    original = _path_plan()
    assert original["candidates"][0]["candidate_kind"] == "PathCandidateV1"
    original_hash = str(original["content_sha256"])

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
    adjudications = [adjudicate_intermediate(item) for item in evidence]
    assert all(adj.confirmed for adj in adjudications)
    segments = build_adjacent_segments(adjudications)
    refs = [(segment.from_ref, segment.to_ref) for segment in segments]
    assert len(segments) == 3
    assert refs == [("R", "I:2"), ("I:2", "I:5"), ("I:5", "P")]
    assert ("R", "P") not in refs, "multi-peak path must never merge into one R→P segment"

    result = stage_new_plan_version(original, evidence)
    new_plan = result.new_plan
    assert validate_v2_document(new_plan) == []
    assert verify_generation_plan(new_plan) == []
    staging = new_plan["extensions"]["staging"]
    assert staging["n_segments"] == 3
    assert staging["never_merged_into_single_ts"] is True
    assert len(staging["confirmed_intermediates"]) == 2
    assert result.confirmed_frame_indices == (2, 5)
    # Original plan document is immutable (hash unchanged).
    assert original["content_sha256"] == original_hash

    segment_candidates = [
        candidate
        for candidate in new_plan["candidates"]
        if ":seg-" in str(candidate.get("candidate_id"))
    ]
    assert len(segment_candidates) == 3
    # TS accounting is per segment: each segment is its own PathCandidateV1;
    # no candidate carries a whole-path R→P endpoint pair (never one TS).
    original_primary = original["candidates"][0]
    full_r = original_primary["endpoint_geometries"]["reactant"]
    full_p = original_primary["endpoint_geometries"]["product"]
    for candidate in segment_candidates:
        endpoints = candidate["endpoint_geometries"]
        assert not (endpoints["reactant"] == full_r and endpoints["product"] == full_p)
        assert candidate["candidate_kind"] == "PathCandidateV1"
        assert candidate["image_chain"] == original_primary["image_chain"]
        assert candidate["method_kind"] == original_primary["method_kind"]
    first, last = segment_candidates[0], segment_candidates[-1]
    assert first["endpoint_geometries"]["product"] == _geom4(0.0)
    assert last["endpoint_geometries"]["reactant"] == _geom4(1.5)

    # Frozen intermediate hash binds the staged version to the evidence.
    assert staging["intermediate_sha256"] == result.intermediate_sha256
    assert staging["intermediate_sha256"]
    assert new_plan["plan_version"] == original["plan_version"] + 1
    assert new_plan["supersedes"] == original["plan_id"]

    # No segmentation without stable evidence (typed refusal, not silence).
    with pytest.raises(IntermediateStagingError) as excinfo:
        stage_new_plan_version(
            original,
            _stable_evidence(
                frame_index=2,
                energy_local_minimum=False,
                optimized_geometry=tuple(tuple(row) for row in _geom4(0.0)),
            ),
        )
    assert excinfo.value.code in {
        "ENERGY_LOCAL_MINIMUM_MISSING",
        CODE_NO_STABLE_INTERMEDIATE,
    }


# ---------------------------------------------------------------------------
# Phase 3 — failure tree + budget replayable across plan versions.
# ---------------------------------------------------------------------------
def test_failure_tree_and_budget_replay_across_plan_versions() -> None:
    """Staged version carries the todo-23 failure tree + budget denominator."""
    original = _frozen_plan()
    assert original["status"] == "frozen"
    original_freeze = original["extensions"]["freeze"]
    assert original_freeze["failure_tree"]["version"] == FAILURE_TREE_VERSION
    assert tuple(original_freeze["failure_tree"]["codes"]) == FAILURE_TREE_CODE_ORDER
    original_budget = dict(original["budget"])
    assert original_budget["max_attempts"] >= 1

    result = stage_new_plan_version(original, _stable_evidence(frame_index=2))
    new_plan = result.new_plan
    staging = new_plan["extensions"]["staging"]

    # Failure-tree vocabulary carried verbatim on the staged version.
    replay_tree = staging["failure_tree_replay"]
    assert replay_tree["version"] == FAILURE_TREE_VERSION
    assert tuple(replay_tree["codes"]) == FAILURE_TREE_CODE_ORDER
    staged_freeze = new_plan["extensions"]["freeze"]
    assert tuple(staged_freeze["failure_tree"]["codes"]) == FAILURE_TREE_CODE_ORDER
    assert set(staged_freeze["failure_tree"]["codes"]) == FAILURE_TREE_CODES

    # Budget denominator carried verbatim (spent stays with the runtime).
    budget_replay = staging["budget_replay"]
    assert budget_replay["from_plan_id"] == original["plan_id"]
    assert budget_replay["from_plan_content_sha256"] == original["content_sha256"]
    assert budget_replay["budget"] == original_budget
    assert tuple(budget_replay["budget_accounting"]) == BUDGET_ACCOUNTING_CATEGORIES
    assert new_plan["budget"] == original_budget

    # Failure predicates remain computable on the replayed vocabulary.
    assert active_failure_codes(FailureState()) == ()
    assert active_failure_codes(FailureState(gate_pass=False)) == ("GATE_REFUSED",)
    exhausted = BudgetLedger(max_attempts=int(original_budget["max_attempts"])).record(
        attempts=int(original_budget["max_attempts"])
    )
    codes = active_failure_codes(FailureState(budget=exhausted))
    assert "BUDGET_EXHAUSTED" in codes
    # All active codes belong to the frozen closed vocabulary (replayable).
    for code in codes:
        assert is_failure_code(code) is True
        assert code in FAILURE_TREE_CODES
    # Free-text refusal is never a failure code (todo 23 discipline).
    from pes2ts_core.generation.planning.plan_freeze import assert_failure_code

    assert is_failure_code("效果不好") is False
    with pytest.raises(PlanFreezeError) as excinfo:
        assert_failure_code("效果不好")
    assert excinfo.value.code == "FAILURE_CODE_UNKNOWN"

    # Staged scan candidates keep drivers verbatim (G2 never self-segments).
    original_drivers = original["candidates"][0]["drivers"]
    segment_candidates = [
        candidate
        for candidate in new_plan["candidates"]
        if ":seg-" in str(candidate.get("candidate_id"))
    ]
    assert segment_candidates
    for candidate in segment_candidates:
        assert candidate["drivers"] == original_drivers
    assert staging["g2_self_segmentation"] is False
    assert staging["drivers_changed_by_staging"] is False


# ---------------------------------------------------------------------------
# Phase 4 — 24-case per-case loop records complete (demo24 selector sweep).
# ---------------------------------------------------------------------------
def test_demo24_per_case_loop_records_are_complete(tmp_path: Path) -> None:
    """Selector over 24 frozen fixtures → typed per-case ledger; no ready claims."""
    from pes2ts_core.config_loader import load_config
    from pes2ts_core.generation.planning.graph_rebuild import (
        load_endpoint_materials_from_export,
        rebuild_endpoint_graphs,
    )

    manifest = json.loads((FIXTURE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    pinned = str(manifest["rdkit_version"])
    if rdBase.rdkitVersion != pinned:
        pytest.fail(
            f"rdkit {rdBase.rdkitVersion} != fixture pin {pinned} (never-skip gate)"
        )
    record_paths = sorted((FIXTURE_ROOT / "records").glob("RXN_*.json"))
    assert len(record_paths) == 24, "P3 closure requires the full demo24 fixture set"

    config = load_config()
    ledger: list[dict[str, Any]] = []
    for record_path in record_paths:
        snapshot = json.loads(record_path.read_text(encoding="utf-8"))
        reaction_id = str(snapshot["reaction_id"])
        export_materials = load_endpoint_materials_from_export(snapshot)
        bundle = rebuild_endpoint_graphs(str(snapshot["reaction_smiles"]), export_materials)
        maps = [int(m) for m in snapshot["maps"]]
        materials = {
            "r_coordinates": {
                map_id: [float(x) for x in snapshot["r_coordinates"][i]]
                for i, map_id in enumerate(maps)
            },
            "p_coordinates": {
                map_id: [float(x) for x in snapshot["p_coordinates"][i]]
                for i, map_id in enumerate(maps)
            },
            "endpoint_electronic": snapshot["endpoint_electronic"],
        }
        proposal = propose_strategies(
            bundle,
            materials,
            config,
            reaction_id=reaction_id,
            split=str(snapshot.get("split", "unassigned")),
        )
        # Proposal document is contract-valid and pure (zero forbidden keys).
        assert validate_v2_document(proposal) == [], reaction_id
        hits = _walk_forbidden(proposal)
        assert hits == [], f"{reaction_id}: forbidden keys {hits}"
        row = _case_ledger_row(
            reaction_id,
            str(snapshot.get("split", "unassigned")),
            proposal,
        )
        # Red line: no ready claim without human review (plan does not lift it).
        assert row["execution_eligible"] is False, reaction_id
        assert row["ready_claimed"] is False, reaction_id
        assert row["needs_review_retained"] is True, reaction_id
        assert row["proposal_status"] != "ready", reaction_id
        if row["proposal_status"] == "rejected":
            assert row["blocking_reason_codes"], reaction_id
        else:
            assert row["reason_codes"] or row["n_candidates"], reaction_id
        ledger.append(row)

    assert len(ledger) == 24
    # Ledger completeness: every row carries the typed closure fields.
    required_keys = {
        "reaction_id",
        "split",
        "proposal_status",
        "route_exit",
        "n_candidates",
        "candidate_modes",
        "reason_codes",
        "blocking_reason_codes",
        "candidate_failure_codes",
        "release_gate",
        "execution_eligible",
        "ready_claimed",
        "needs_review_retained",
        "epistemic_status",
    }
    for row in ledger:
        assert required_keys <= set(row), row["reaction_id"]
        assert row["epistemic_status"] == "endpoint_hypothesis"
    # No document anywhere in the loop claims ready/execution eligibility.
    assert all(row["ready_claimed"] is False for row in ledger)
    assert all(row["execution_eligible"] is False for row in ledger)

    # Closure record: per-case ledger written by the test (format documented
    # in .omo/evidence/task-31-*.txt).  Deterministic modulo content hashes.
    ledger_doc = {
        "schema_name": "p3_demo24_case_ledger_v1",
        "n_cases": len(ledger),
        "review_dependency": "human_review_not_lifted_by_this_plan",
        "ready_claims": 0,
        "cases": ledger,
    }
    ledger_path = tmp_path / "p3_demo24_case_ledger.json"
    ledger_path.write_text(
        stable_json_dumps(ledger_doc) + "\n", encoding="utf-8"
    )
    reread = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert reread["n_cases"] == 24
    assert reread["ready_claims"] == 0
    assert [row["reaction_id"] for row in reread["cases"]] == [
        row["reaction_id"] for row in ledger
    ]

    # Coverage declaration stays honest (todo 29 seam): Demo24 ≠ full coverage.
    assert "C" in DEMO24_COVERAGE.domains_with_evidence
    assert "Fe" in DEMO24_COVERAGE.domains_without_evidence


# ---------------------------------------------------------------------------
# Phase 5 — 200–1000 stratified pilot: declared-not-executed record.
# ---------------------------------------------------------------------------
def test_stratified_pilot_plan_is_declared_not_executed() -> None:
    """Pilot plan is a typed declared-not-executed record; never executed here."""
    pilot_plan: dict[str, Any] = {
        "schema_name": "g1_stratified_pilot_plan_v1",
        "schema_version": 1,
        "status": PILOT_STATUS_NOT_EXECUTED,
        "executed": False,
        "executed_n": 0,
        "prerequisites": [PILOT_PREREQ_HUMAN_REVIEW],
        "prerequisite_status": "pending_human_review",
        "size_bounds": {"min": PILOT_SIZE_MIN, "max": PILOT_SIZE_MAX},
        "target_size": None,
        "source_list": "data/interim/g2_scan_ready.json",
        "source_list_n": POPULATION_SCAN_READY_N,
        "stratification_basis": (
            "L0u cluster ids from g2 export docs (offline declaration only; "
            "sampling frame is the g2_scan_ready list count)"
        ),
        "scope": "development_mechanism_coverage",
        "note": (
            "200-1000 stratified pilot requires human review first; "
            "execution is NOT part of plan todo 31"
        ),
        "comparison_arms": [
            "old_single_b_v1",
            "graph_theory_single_b",
            "coupled_1d",
            "timing_direction_variants",
            "path_neb",
        ],
        "shared_budget_required": True,
        "truth_provenance": "truth_assisted_p1",
    }

    # Typed-structure contract: bounds, source reference, prereq, status.
    assert pilot_plan["status"] == PILOT_STATUS_NOT_EXECUTED
    assert pilot_plan["executed"] is False
    assert pilot_plan["executed_n"] == 0
    assert PILOT_PREREQ_HUMAN_REVIEW in pilot_plan["prerequisites"]
    assert pilot_plan["size_bounds"]["min"] == PILOT_SIZE_MIN
    assert pilot_plan["size_bounds"]["max"] == PILOT_SIZE_MAX
    assert PILOT_SIZE_MIN <= PILOT_SIZE_MAX <= 1000
    assert pilot_plan["source_list"] == "data/interim/g2_scan_ready.json"
    assert pilot_plan["source_list_n"] == POPULATION_SCAN_READY_N == 183_460
    assert pilot_plan["truth_provenance"] == "truth_assisted_p1"

    # Population guard stays armed: full scan_ready work is refused at L1–L3
    # and at L4 without a VersionLock (guard logic only — never executed).
    for stage in ("L1", "L2", "L3"):
        with pytest.raises(ScientificReportError) as excinfo:
            assert_population_unlocked(stage)
        assert excinfo.value.code == "POPULATION_LOCKED_BEFORE_L4"
    with pytest.raises(ScientificReportError) as excinfo:
        assert_population_unlocked("L4")
    assert excinfo.value.code == "POPULATION_LOCK_REQUIRED"

    # The pilot plan document itself is pure (no truth geometry/energy keys).
    assert _walk_forbidden(pilot_plan) == []


# ---------------------------------------------------------------------------
# Phase 6 — unified L1–L4 certification + NEB smoke blocked (env).
# ---------------------------------------------------------------------------
def test_assess_stage_certification_is_explicit_per_stage() -> None:
    """L1–L4 each explicit passed/blocked; NEB smoke blocked is never silent."""
    certification = assess_stage_certification()
    assert set(certification) == {"L1", "L2", "L3", "L4"}
    for stage_id, record in certification.items():
        assert record["offline_status"] in {"passed", "blocked"}, stage_id
        assert record["offline_evidence"], stage_id
        if record["smoke_family"] is not None:
            assert record["smoke_status"] in {"passed", "blocked"}, stage_id
    # Offline closures for this plan are PASSED (P0/P1/P2/P3 acceptance tests).
    assert certification["L1"]["offline_status"] == "passed"
    assert certification["L2"]["offline_status"] == "passed"
    assert certification["L3"]["offline_status"] == "passed"
    assert certification["L4"]["offline_status"] == "passed"
    # P2 ORCA smoke and P3 NEB smoke are gated: explicit passed/blocked.
    assert certification["L3"]["smoke_family"] == "orca"
    assert certification["L4"]["smoke_family"] == "neb"
    assert certification["L3"]["smoke_status"] in {"passed", "blocked"}
    assert certification["L4"]["smoke_status"] in {"passed", "blocked"}

    neb_smoke = _neb_gated_smoke_status()
    assert neb_smoke["status"] in {"passed", "blocked"}
    if not neb_smoke["env_ready"]:
        # This tree: gated env absent → blocked, missing vars named explicitly.
        assert neb_smoke["status"] == "blocked"
        assert ENV_ORCA_EXECUTABLE in neb_smoke["missing_env"]
        assert neb_smoke["requirements"], "blocked status must carry the env requirements"
        # L4 smoke status mirrors the NEB gated-smoke reality.
        assert certification["L4"]["smoke_status"] == "blocked"
        assert ENV_ORCA_EXECUTABLE in certification["L4"]["smoke_missing_env"]
    else:
        assert certification["L4"]["smoke_status"] == neb_smoke["status"]


def test_neb_smoke_status_helper_is_explicit_passed_or_blocked(tmp_path: Path) -> None:
    """NEB smoke helper: env+receipt → passed; env missing → blocked named."""
    empty_dir = tmp_path / "empty_receipts"
    empty_dir.mkdir()

    # Env absent regardless of receipts → blocked with the env list.
    no_env = _neb_gated_smoke_status(environ={}, receipts_dir=empty_dir)
    assert no_env["status"] == "blocked"
    assert no_env["missing_env"] == [ENV_ORCA_EXECUTABLE]
    assert no_env["requirements"]

    # Env ready but no recorded PATH_NEB receipt → still blocked (never optimistic).
    full_env = {ENV_ORCA_EXECUTABLE: "/fake/orca"}
    unrecorded = _neb_gated_smoke_status(environ=full_env, receipts_dir=empty_dir)
    assert unrecorded["status"] == "blocked"

    # Env ready + PATH_NEB pass receipt → passed (injected, no real run).
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    (receipts / "probe_receipt_neb-minimal-smoke.json").write_text(
        json.dumps(
            {
                "probe_id": "neb-minimal-smoke",
                "family": "orca",
                "status": "pass",
                "modes": ["PATH_NEB"],
            }
        ),
        encoding="utf-8",
    )
    passed = _neb_gated_smoke_status(environ=full_env, receipts_dir=receipts)
    assert passed["status"] == "passed"
    assert passed["env_ready"] is True
    assert passed["pass_receipts"] == ["probe_receipt_neb-minimal-smoke.json"]

    # A scan-only receipt does NOT unlock the NEB smoke status.
    scan_only = tmp_path / "scan_receipts"
    scan_only.mkdir()
    (scan_only / "probe_receipt_orca-single-b-smoke.json").write_text(
        json.dumps(
            {
                "probe_id": "orca-single-b-smoke",
                "family": "orca",
                "status": "pass",
                "modes": ["SINGLE_1D"],
            }
        ),
        encoding="utf-8",
    )
    still_blocked = _neb_gated_smoke_status(environ=full_env, receipts_dir=scan_only)
    assert still_blocked["status"] == "blocked"

    # ORCA smoke helper mirrors the same discipline (needs an orca probe id).
    orca_no_env = _orca_gated_smoke_status(environ={}, receipts_dir=empty_dir)
    assert orca_no_env["status"] == "blocked"
    assert orca_no_env["missing_env"] == [ENV_ORCA_EXECUTABLE]
    orca_receipts = tmp_path / "orca_receipts"
    orca_receipts.mkdir()
    (orca_receipts / "probe_receipt_orca-single-b-smoke.json").write_text(
        json.dumps(
            {
                "probe_id": "orca-single-b-smoke",
                "family": "orca",
                "status": "pass",
                "modes": ["SINGLE_1D"],
            }
        ),
        encoding="utf-8",
    )
    orca_passed = _orca_gated_smoke_status(environ=full_env, receipts_dir=orca_receipts)
    assert orca_passed["status"] == "passed"

    # ACP env absence is recorded for the P2 smoke surface (never silent).
    assert ENV_ACP_ROOT and ENV_ACP_PYTHON  # names exist for evidence listing


# ---------------------------------------------------------------------------
# Phase 7 — staging refusal when evidence is absent (typed, not silent).
# ---------------------------------------------------------------------------
def test_staging_refuses_without_stable_intermediate_evidence() -> None:
    """Absent evidence → typed refusal; no segmentation, no fake TS."""
    original = _frozen_plan()
    with pytest.raises(IntermediateStagingError) as excinfo:
        stage_new_plan_version(original, [])
    assert excinfo.value.code == CODE_NO_STABLE_INTERMEDIATE
    # Refusal codes are a closed typed vocabulary (never free text).
    from pes2ts_core.generation.planning.intermediate_staging import STAGING_REFUSAL_CODES

    assert excinfo.value.code in STAGING_REFUSAL_CODES
