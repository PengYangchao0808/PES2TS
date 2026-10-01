"""P2 closure acceptance harness (plan todo 25) — multi-coordinate loop.

Design §13.2 P2 gate: COUPLED/SCHEDULED compile, per-frame all-driver
recovery, target-path quality tiers, plan freeze, explicit capability
refusals, no first-driver projection anywhere, and an explicit gated-smoke
status surface (``passed`` / ``blocked``).

Every phase calls REAL modules (selector, compile_orca, multicoord adapter,
frame_recovery, target_path, plan_freeze, capabilities) — no layer logic is
reimplemented here.  Synthetic candidates follow the selector-output shape
established by ``tests/test_compile_orca.py`` / ``tests/test_selector_core.py``
and are reused across phases via sibling-test imports (precedent:
``tests/test_g1_strata.py``).  No real ORCA/ACP execution (plan MUST NOT).
"""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Any

import pytest

from pes2ts_core.integration.acp.adapter import ACPMappingError
from pes2ts_core.integration.acp.frame_recovery import (
    PATH_BUNDLE_EXTENSION_KEY,
    recover_frames,
    to_path_bundle_extension,
)
from pes2ts_core.integration.acp.multicoord import (
    CODE_BACKEND_CAPABILITY_MISSING as MC_CODE_BACKEND_CAPABILITY_MISSING,
    driver_id_for,
    multicoord_request_payload,
    parse_multicoord_result,
)
from pes2ts_core.scan_strategy import capabilities as caps
from pes2ts_core.scan_strategy import compile_orca as co
from pes2ts_core.scan_strategy.contracts_v2 import (
    make_strategy_proposal,
    validate_v2_document,
)
from pes2ts_core.scan_strategy.plan_freeze import (
    FAILURE_PREDICATES,
    FAILURE_TREE_CODES,
    BudgetLedger,
    FailureState,
    PlanFreezeError,
    active_failure_codes,
    assert_failure_code,
    freeze_generation_plan,
    is_failure_code,
    verify_generation_plan,
)
from pes2ts_core.scan_strategy.selector import propose_strategies
from pes2ts_core.scan_strategy.target_path import (
    CODE_TARGET_INCOMPLETE,
    EVIDENCE_FIRST_ORDER_IMAGINARY,
    EVIDENCE_IRC_FORWARD,
    EVIDENCE_IRC_REVERSE,
    EVIDENCE_OPTTS,
    LABEL_ACHIEVED,
    LABEL_NOT_ACHIEVED,
    TIERS,
    TIER_EXECUTION_COMPLETE,
    TIER_NUMERICALLY_USABLE,
    TIER_TARGET_PATH_COMPATIBLE,
    TIER_VALIDATED_TS,
    assess_target_path,
)

# Sibling-test fixture reuse (tests/ is on sys.path under pytest prepend mode).
from test_acp_frame_all_drivers import _raw_frames_2b  # noqa: E402
from test_acp_multicoord_adapter import (  # noqa: E402
    ATOM_ROWS,
    _ad_geometry,
    _b2_geometry,
    _compile_2b as _mc_compile_2b,
    _compile_scheduled_nonuniform,
    _default_capability,
    _materials as _mc_materials,
)
from test_compile_orca import (  # noqa: E402
    _driver,
    _lambdas,
    _scan_candidate,
    _smoke,
)
from test_generation_plan_freeze import (  # noqa: E402
    _config as _freeze_config,
    _proposal as _freeze_proposal_fixture,
)
from test_selector_core import (  # noqa: E402
    EXCHANGE_P,
    EXCHANGE_R,
    EXCHANGE_SMILES,
    _bundle as _selector_bundle,
    _config as _selector_config,
    _materials as _selector_materials,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER_01 = driver_id_for(0)
DRIVER_02 = driver_id_for(1)
EXCHANGE_ATOM_ROWS = (1, 2, 3, 4)
EXCHANGE_ELEMENTS = ["C", "C", "C", "H"]

ENV_ACP_ROOT = "PES2TS_ACP_ROOT"
ENV_ACP_PYTHON = "PES2TS_ACP_PYTHON"
ENV_ORCA_EXECUTABLE = "PES2TS_ORCA_EXECUTABLE"
ENV_SMOKE_RECEIPTS_DIR = "PES2TS_SMOKE_RECEIPTS_DIR"

#: Gate-state codes the freeze projection may clear — each represents a gate
#: that passes in the post-review / post-smoke state freeze consumes.  Budget
#: prunes and capability refusals are NEVER cleared (asserted below).
_PROJECTION_CLEARABLE_CODES = frozenset({
    "DIRECTION_NOT_READY",  # human review gate (plan: review not lifted here)
    "SCHEDULE_INTEGRITY",   # selector records passing integrity notes as rows
    "BACKEND_SMOKE_REQUIRED",  # multi-coordinate smoke gate (blocked in this tree)
})


def _freeze_ready_proposal_from_selector(
    selector_proposal: dict[str, Any], selected: dict[str, Any]
) -> dict[str, Any]:
    """Project a selector candidate into post-review/post-smoke freeze-ready form.

    The selector honestly records gate-state codes on every multi-coordinate
    candidate (review incomplete; backend smoke blocked in this tree).  Freeze
    consumes the post-gate state; this projection clears ONLY gate-state codes
    and refuses any candidate carrying a budget prune or technical failure.
    The raw-candidate freeze refusal is asserted separately in the e2e test —
    the red line "un-smoked plans never become ready" stays armed.
    """
    candidate = copy.deepcopy(selected)
    codes = {str(reason.get("code")) for reason in candidate.get("failure_reasons") or []}
    assert "PRUNED_BY_BUDGET" not in codes, "budget prunes are never projected away"
    uncleared = codes - _PROJECTION_CLEARABLE_CODES
    assert not uncleared, f"technical failures block freeze projection: {uncleared}"
    candidate["failure_reasons"] = []
    candidate["capability_check"] = {"status": "pass", "missing": []}
    fields = {
        key: selector_proposal[key]
        for key in (
            "reaction_id",
            "case_id",
            "split",
            "source_case_sha256",
            "graph_input",
            "graph_features",
            "family",
            "motif_tags",
            "rule_trace",
            "epistemic_status",
            "reasons",
            "blocking_reasons",
        )
    }
    assert not fields["blocking_reasons"], "whole-case blocking reasons must be empty to freeze"
    fields["candidates"] = [candidate]
    return make_strategy_proposal(fields["reaction_id"], "proposed", **fields)


# ---------------------------------------------------------------------------
# Shared builders (selector-output candidate shape; todo-19/20 vocabulary).
# ---------------------------------------------------------------------------
def _exchange_geometry() -> dict[str, dict[int, tuple[float, float, float]]]:
    return {"R": dict(EXCHANGE_R), "P": dict(EXCHANGE_P)}


def _exchange_materials() -> dict[str, Any]:
    return {
        "atom_map_ids": list(EXCHANGE_ATOM_ROWS),
        "elements": list(EXCHANGE_ELEMENTS),
        "r_coordinates": [list(EXCHANGE_R[m]) for m in EXCHANGE_ATOM_ROWS],
        "p_coordinates": [list(EXCHANGE_P[m]) for m in EXCHANGE_ATOM_ROWS],
        "charge": 0,
        "multiplicity": 1,
    }


def _exchange_selector_outputs(
    tmp_path: Path,
    capability: caps.EffectiveCapability,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Real selector on the synthetic exchange reaction (2-driver COUPLED route)."""
    materials = _selector_materials(EXCHANGE_R, EXCHANGE_P)
    bundle = _selector_bundle(EXCHANGE_SMILES, materials)
    return propose_strategies(
        bundle,
        materials,
        config if config is not None else _selector_config(),
        reaction_id="RXN_P2_CLOSURE",
        split="unassigned",
        capability=capability,
    )


def _pick_coupled_candidate(proposal: dict[str, Any]) -> dict[str, Any]:
    coupled = [
        candidate
        for candidate in proposal["candidates"]
        if candidate.get("mode") == "COUPLED_1D" and len(candidate.get("drivers") or ()) == 2
    ]
    assert coupled, "exchange route must propose a 2-driver COUPLED_1D candidate"
    active = [
        candidate
        for candidate in coupled
        if not any(
            reason.get("code") == "PRUNED_BY_BUDGET"
            for reason in candidate.get("failure_reasons") or []
        )
    ]
    return (active or coupled)[0]


def _as_scan_candidate(payload: dict[str, Any]) -> dict[str, Any]:
    """Selector payload → frozen-plan candidate shape expected by compile_orca."""
    return {**payload, "candidate_kind": "ScanCandidateV2"}


def _frames_following_schedule(
    compiled: Any,
    *,
    residual: float = 0.01,
    drop_actual_for: str | None = None,
    drop_at: int | None = None,
) -> list[dict[str, Any]]:
    """Synthetic ACP-like frames whose actuals track the compiled schedule."""
    driver_ids = [driver_id_for(index) for index in range(len(compiled.coordinates))]
    frames: list[dict[str, Any]] = []
    for index in range(compiled.total_points):
        targets = {
            driver_ids[pos]: compiled.coordinates[pos].values[index]
            for pos in range(len(driver_ids))
        }
        actuals = {driver_id: value + residual for driver_id, value in targets.items()}
        if drop_actual_for is not None and index == drop_at:
            actuals.pop(drop_actual_for, None)
        frames.append(
            {
                "index": index,
                "optimization_converged": True,
                "target_coordinates": dict(targets),
                "actual_coordinates": actuals,
                "constraint_residuals": {
                    driver_id: -residual for driver_id in targets if driver_id in actuals
                },
                "scan_energy_hartree": -40.0 - index / 10,
                "single_point_energy_hartree": None,
                "geometry_path": f"frames/f{index}.xyz",
                "retry_history": [],
            }
        )
    return frames


def _parse(compiled: Any, raw_frames: list[dict[str, Any]]) -> Any:
    return parse_multicoord_result(
        compiled=compiled,
        execution_id="exec-p25",
        acp_task_id="job-p25",
        raw_frames=raw_frames,
    )


def _exchange_budget_config() -> dict[str, Any]:
    """Selector config with headroom so multi-coordinate rows survive budget."""
    return _selector_config(
        max_total_candidates={"scan": 12, "path_neb": 2}, schedule_budget=4
    )


def _unfinished_formed_target(*, driver_id: str = DRIVER_01) -> dict[str, Any]:
    """C–C formed bond that never reaches the bonded range at the end frame."""
    return {
        "target_id": "t-formed-unfinished",
        "atom_maps": [1, 2],
        "edit_kind": "formed",
        "element_a": "C",
        "element_b": "C",
        "r_distance_angstrom": 3.0,
        "p_distance_angstrom": 1.54,
        "driver_id": driver_id,
    }


def _p2_candidate() -> dict[str, Any]:
    """Freeze-ready 2-driver COUPLED candidate (todo-23 fixture shape)."""
    return {
        "candidate_id": "cand-p2-0000",
        "mode": "COUPLED_1D",
        "start_endpoint": "R",
        "direction": "R_to_P",
        "assembly_id": "asm-p2-c1",
        "anchor_reason": "layered_direction_comparison_v1",
        "drivers": [_driver("B", [1, 2]), _driver("B", [3, 4])],
        "monitors": [],
        "guards": [],
        "event_coverage": [{"event_id": "evt-0001", "coverage": "direct"}],
        "lambda_values": [0.0, 0.5, 1.0],
        "schedule_id": "sched-linear-p2",
        "schedule_kind": "linear",
        "required_capabilities": ["COUPLED_1D"],
        "capability_check": {"status": "pass", "missing": []},
        "budget": {"max_attempts": 1, "max_cpu_hours": 0.0, "max_wall_seconds": 0},
        "failure_reasons": [],
        "fallback_ids": ["LOCAL_CONNECTIVITY"],
    }


def _bond_targets_from_geometry(compiled: Any) -> list[dict[str, Any]]:
    """Derive per-driver B-bond target definitions from endpoint geometry."""
    geometry = _exchange_geometry()
    definitions: list[dict[str, Any]] = []
    for position, row in enumerate(compiled.coordinates):
        if row.kind != "B" or len(row.maps) != 2:
            continue
        map_a, map_b = row.maps
        d_r = math.dist(geometry["R"][map_a], geometry["R"][map_b])
        d_p = math.dist(geometry["P"][map_a], geometry["P"][map_b])
        definitions.append(
            {
                "target_id": f"t-bond-{position}",
                "atom_maps": [map_a, map_b],
                "edit_kind": "formed" if d_p < d_r else "broken",
                "element_a": "C",
                "element_b": "C",
                "r_distance_angstrom": round(d_r, 6),
                "p_distance_angstrom": round(d_p, 6),
                "driver_id": driver_id_for(position),
            }
        )
    return definitions


# ---------------------------------------------------------------------------
# P2 gated-smoke status surface (explicit passed/blocked; todo-24 evidence).
# ---------------------------------------------------------------------------
def _p2_gated_smoke_status(
    environ: dict[str, str] | None = None, receipts_dir: Path | str | None = None
) -> dict[str, Any]:
    """Report acp/orca gated-smoke status as ``passed`` or ``blocked``.

    ``passed`` requires the gated env vars AND recorded pass probe receipts
    for both families (never optimistic — mirrors todo-16/24 discipline).
    Anything else is ``blocked`` with the missing env var names listed.
    """
    env = os.environ if environ is None else environ
    acp_root = env.get(ENV_ACP_ROOT)
    acp_python = env.get(ENV_ACP_PYTHON)
    orca_executable = env.get(ENV_ORCA_EXECUTABLE)
    missing_env = [
        name
        for name, value in (
            (ENV_ACP_ROOT, acp_root),
            (ENV_ACP_PYTHON, acp_python),
            (ENV_ORCA_EXECUTABLE, orca_executable),
        )
        if not value
    ]
    resolved_receipts = Path(
        receipts_dir
        or env.get(ENV_SMOKE_RECEIPTS_DIR)
        or (REPO_ROOT / ".omo" / "evidence")
    )
    pass_receipts: list[str] = []
    if resolved_receipts.is_dir():
        for path in sorted(resolved_receipts.glob("probe_receipt_*.json")):
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(document, dict) and str(document.get("status", "")).lower() == "pass":
                pass_receipts.append(path.name)
    families = {
        "acp": any("acp" in name for name in pass_receipts),
        "orca": any("orca" in name for name in pass_receipts),
    }
    env_ready = not missing_env
    status = "passed" if (env_ready and families["acp"] and families["orca"]) else "blocked"
    return {
        "status": status,
        "env_ready": env_ready,
        "missing_env": missing_env,
        "families": families,
        "pass_receipts": pass_receipts,
        "receipts_dir": str(resolved_receipts),
    }


# ---------------------------------------------------------------------------
# Phase 1 — COUPLED/SCHEDULED compile from selector outputs.
# ---------------------------------------------------------------------------
def test_coupled_and_scheduled_compile_from_selector_outputs(tmp_path) -> None:
    """Selector 2-driver outputs → compile_orca: both drivers, aligned λ, hashes."""
    capability = _smoke(tmp_path)  # injected passing Simul_Scan probe receipts

    # --- COUPLED: real selector output compiled with the same capability ----
    proposal = _exchange_selector_outputs(tmp_path, capability, _exchange_budget_config())
    selected = _pick_coupled_candidate(proposal)
    assert selected["capability_check"]["status"] == "pass"
    compiled = co.compile_orca(
        _as_scan_candidate(selected),
        EXCHANGE_ATOM_ROWS,
        capability,
        geometry=_exchange_geometry(),
    )
    assert compiled.mode == "COUPLED_1D"
    assert len(compiled.coordinates) == 2  # both drivers recovered, never projected
    compiled_maps = {frozenset(row.maps) for row in compiled.coordinates}
    assert compiled_maps == {frozenset((1, 2)), frozenset((2, 3))}
    assert compiled.simultaneous is True
    assert any("Simul_Scan true" in fragment for fragment in compiled.geom_fragments)
    # Aligned λ: shared grid, per-point values, no implicit grid.
    assert list(compiled.lambda_values) == list(selected["lambda_values"])
    assert compiled.total_points == len(compiled.lambda_values)
    for row in compiled.coordinates:
        assert len(row.values) == compiled.total_points
        # compile rounds schedule values to 6 dp (byte-stable serialization).
        assert list(row.schedule_values) == pytest.approx(
            list(compiled.lambda_values), abs=1e-6
        )
        for index, lam in enumerate(compiled.lambda_values):
            expected = row.start_value + lam * (row.end_value - row.start_value)
            assert row.values[index] == pytest.approx(expected, abs=1e-5)
    assert compiled.point_input_sha256, "native scan records an input hash"

    # --- SCHEDULED: synthetic selector-output-shaped candidate -------------
    scheduled_candidate = _scan_candidate(
        "SCHEDULED_1D",
        [
            _driver("B", [1, 2], schedule_values=[0.0, 0.25, 0.6, 1.0]),
            _driver("B", [3, 4], schedule_values=[0.0, 0.5, 0.75, 1.0]),
        ],
        _lambdas(4),
        schedule_kind="event_A_early",
        candidate_id="cand-p2-sched",
    )
    scheduled = co.compile_orca(
        scheduled_candidate, ATOM_ROWS, capability, geometry=_b2_geometry()
    )
    assert scheduled.compiled_kind == "recipe"
    assert len(scheduled.coordinates) == 2
    assert scheduled.total_points == len(scheduled.lambda_values) == 4
    assert len(scheduled.point_input_sha256) == scheduled.total_points  # per-point hashes
    assert len(set(scheduled.point_input_sha256)) == scheduled.total_points
    for fragment in scheduled.geom_fragments:
        assert fragment.count("{ B ") == 2  # both drivers in every point block
    for row in scheduled.coordinates:
        assert len(row.values) == scheduled.total_points
        assert list(row.schedule_values) != list(scheduled.lambda_values)  # non-uniform kept


# ---------------------------------------------------------------------------
# Phase 2 — per-frame all-driver recovery with one incomplete frame.
# ---------------------------------------------------------------------------
def test_per_frame_all_driver_recovery_marks_missing_driver_frame_incomplete(tmp_path) -> None:
    """2 drivers × N frames, one frame missing driver-02 actual → named incomplete."""
    compiled = _mc_compile_2b(tmp_path)  # COUPLED 2-B, 5 λ points
    raw = _raw_frames_2b(drop_driver02_actual_at=1)
    result = _parse(compiled, raw)
    report = recover_frames(result, compiled)

    assert report.driver_ids == (DRIVER_01, DRIVER_02)
    assert report.n_frames == 3
    assert report.complete is False
    assert report.incomplete_frame_indices == (1,)
    # Complete per-frame traces on the healthy frames — every driver, every frame.
    for record in report.frames:
        assert set(record.targets_by_driver) == {DRIVER_01, DRIVER_02}
        assert set(record.actuals_by_driver) == {DRIVER_01, DRIVER_02}
        assert set(record.residuals_by_driver) == {DRIVER_01, DRIVER_02}
    frame0 = next(record for record in report.frames if record.frame_index == 0)
    frame2 = next(record for record in report.frames if record.frame_index == 2)
    for record in (frame0, frame2):
        assert record.incomplete is False
        assert record.missing_driver_ids == ()
        for driver_id in report.driver_ids:
            assert record.targets_by_driver[driver_id] is not None
            assert record.actuals_by_driver[driver_id] is not None
            assert record.residuals_by_driver[driver_id] is not None
    # The one defective frame is named, not silently passed.
    frame1 = next(record for record in report.frames if record.frame_index == 1)
    assert frame1.incomplete is True
    assert frame1.missing_driver_ids == (DRIVER_02,)
    assert frame1.missing_residual_driver_ids == (DRIVER_02,)
    assert frame1.residuals_by_driver[DRIVER_02] is None
    assert frame1.residuals_by_driver[DRIVER_01] is not None


# ---------------------------------------------------------------------------
# Phase 3 — target quality layer on recovered frames.
# ---------------------------------------------------------------------------
def test_target_quality_layer_records_tiers_and_unfinished_target(tmp_path) -> None:
    """Recovered frames → assess_target_path: tiers recorded; unfinished → false."""
    compiled = _mc_compile_2b(tmp_path)
    raw = _frames_following_schedule(compiled)
    report = recover_frames(_parse(compiled, raw), compiled)
    assert report.complete is True

    assessment = assess_target_path(
        report,
        {"bonds": [_unfinished_formed_target()], "regions": [], "stereo": []},
        {},
    )
    # All four tiers recorded independently — never collapsed.
    assert set(assessment.tiers) == set(TIERS)
    for tier_name in TIERS:
        assert assessment.tiers[tier_name].status in {
            LABEL_ACHIEVED,
            LABEL_NOT_ACHIEVED,
            "not_evaluable",
        }
    assert assessment.tiers[TIER_EXECUTION_COMPLETE].status == LABEL_ACHIEVED
    # Technical usability ≠ chemical completion (design §11.2 plan failure).
    assert assessment.tiers[TIER_NUMERICALLY_USABLE].status == LABEL_ACHIEVED
    assert assessment.tiers[TIER_TARGET_PATH_COMPATIBLE].status == LABEL_NOT_ACHIEVED
    assert (
        CODE_TARGET_INCOMPLETE
        in assessment.tiers[TIER_TARGET_PATH_COMPATIBLE].reason_codes
    )
    # validated_ts needs explicit evidence refs — absent here, honestly.
    ts_tier = assessment.tiers[TIER_VALIDATED_TS]
    assert ts_tier.status == LABEL_NOT_ACHIEVED
    assert set(ts_tier.missing_evidence) == {
        EVIDENCE_OPTTS,
        EVIDENCE_FIRST_ORDER_IMAGINARY,
        EVIDENCE_IRC_FORWARD,
        EVIDENCE_IRC_REVERSE,
    }


# ---------------------------------------------------------------------------
# Phase 4 — plan freeze + replayable failure tree.
# ---------------------------------------------------------------------------
def test_plan_freeze_valid_document_and_failure_tree_replayable(tmp_path) -> None:
    """Freeze-ready proposal + real CompiledRequest → valid frozen plan."""
    capability = _smoke(tmp_path)
    candidate_payload = _p2_candidate()
    compiled = co.compile_orca(
        _as_scan_candidate(candidate_payload),
        ATOM_ROWS,
        capability,
        geometry=_b2_geometry(),
    )
    proposal = _freeze_proposal_fixture(candidate=candidate_payload)
    plan = freeze_generation_plan(
        proposal, candidate_payload, compiled, _freeze_config()
    )
    assert validate_v2_document(plan) == []
    assert plan["schema_version"] == "g1_generation_plan_v2"
    assert plan["status"] == "frozen"
    assert verify_generation_plan(plan) == []
    # Multi-coordinate freeze: both drivers survive into the frozen candidate
    # and into the sealed compiled-request binding.
    frozen_candidate = plan["candidates"][0]
    assert frozen_candidate["candidate_kind"] == "ScanCandidateV2"
    assert len(frozen_candidate["drivers"]) == 2
    assert {frozenset(d["maps"]) for d in frozen_candidate["drivers"]} == {
        frozenset((1, 2)),
        frozenset((3, 4)),
    }
    compiled_binding = plan["extensions"]["freeze"]["compiled_request"]
    assert len(compiled_binding["coordinates"]) == 2

    # Failure tree: closed vocabulary + computable, replayable predicates.
    assert set(FAILURE_PREDICATES) == FAILURE_TREE_CODES
    assert active_failure_codes(FailureState()) == ()
    assert active_failure_codes(FailureState(gate_pass=False)) == ("GATE_REFUSED",)
    ledger = BudgetLedger(max_attempts=1).record(attempts=1)
    assert active_failure_codes(FailureState(budget=ledger)) == ("BUDGET_EXHAUSTED",)
    multi = FailureState(opt_failed=True, constraint_residual=True)
    assert active_failure_codes(multi) == ("OPT_FAILED", "CONSTRAINT_RESIDUAL")
    assert is_failure_code("BUDGET_EXHAUSTED") is True
    assert is_failure_code("效果不好") is False
    with pytest.raises(PlanFreezeError) as excinfo:
        assert_failure_code("效果不好")
    assert excinfo.value.code == "FAILURE_CODE_UNKNOWN"


# ---------------------------------------------------------------------------
# Phase 5 — explicit refusals + no implicit grids.
# ---------------------------------------------------------------------------
def test_explicit_refusals_capability_missing_unsupported_and_no_implicit_grids(
    tmp_path,
) -> None:
    """Default registry refuses COUPLED/A-D loudly; grids stay explicit."""
    default_capability = _default_capability()  # shipped registry, unprobed

    # Inject test: request COUPLED on the default registry → compile refuses.
    coupled_candidate = _scan_candidate(
        "COUPLED_1D", [_driver("B", [1, 2]), _driver("B", [3, 4])], _lambdas(5)
    )
    with pytest.raises(co.OrcaCompileError) as excinfo:
        co.compile_orca(
            coupled_candidate, ATOM_ROWS, default_capability, geometry=_b2_geometry()
        )
    assert excinfo.value.code == co.CODE_BACKEND_CAPABILITY_MISSING

    # Adapter refuses the same request with the ACP-mappable typed code.
    probed = _smoke(tmp_path)
    compiled = co.compile_orca(
        coupled_candidate, ATOM_ROWS, probed, geometry=_b2_geometry()
    )
    with pytest.raises(Exception) as adapter_excinfo:
        multicoord_request_payload(
            compiled, _mc_materials(_b2_geometry()), {"capability": default_capability}
        )
    adapter_error = adapter_excinfo.value
    assert isinstance(adapter_error, ACPMappingError)
    assert getattr(adapter_error, "code", None) == MC_CODE_BACKEND_CAPABILITY_MISSING

    # Unsupported A/D on the current adapter (shipped registry: B only).
    ad_candidate = _scan_candidate(
        "COUPLED_1D", [_driver("A", [2, 1, 3]), _driver("D", [2, 1, 3, 4])], _lambdas(5)
    )
    with pytest.raises(co.OrcaCompileError) as ad_excinfo:
        co.compile_orca(
            ad_candidate, ATOM_ROWS, default_capability, geometry=_ad_geometry()
        )
    assert ad_excinfo.value.code == co.CODE_BACKEND_CAPABILITY_MISSING
    compiled_ad = co.compile_orca(
        ad_candidate, ATOM_ROWS, probed, geometry=_ad_geometry()
    )
    with pytest.raises(Exception) as ad_payload_excinfo:
        multicoord_request_payload(
            compiled_ad,
            {
                "atom_map_ids": [1, 2, 3, 4],
                "elements": ["C", "C", "H", "H"],
                "r_coordinates": [list(_ad_geometry()["R"][m]) for m in (1, 2, 3, 4)],
                "p_coordinates": [list(_ad_geometry()["P"][m]) for m in (1, 2, 3, 4)],
                "charge": 0,
                "multiplicity": 1,
            },
            {"capability": default_capability},
        )
    assert isinstance(ad_payload_excinfo.value, ACPMappingError)
    assert getattr(ad_payload_excinfo.value, "code", None) == (
        MC_CODE_BACKEND_CAPABILITY_MISSING
    )

    # No implicit grids: compiled totals equal the schedule grid, everywhere.
    assert compiled.total_points == len(compiled.lambda_values)
    assert len(compiled.coordinates[0].values) == compiled.total_points
    scheduled = _compile_scheduled_nonuniform(tmp_path)
    assert scheduled.total_points == len(scheduled.lambda_values)
    assert len(scheduled.point_input_sha256) == scheduled.total_points
    assert len(scheduled.geom_fragments) == scheduled.total_points


# ---------------------------------------------------------------------------
# Phase 6 — no first-driver projection anywhere down the chain.
# ---------------------------------------------------------------------------
def test_no_first_driver_projection_across_every_chain_artifact(tmp_path) -> None:
    """Both drivers appear in compile, payload, recovery, extension, freeze."""
    capability = _smoke(tmp_path)
    candidate_payload = _p2_candidate()
    compiled = co.compile_orca(
        _as_scan_candidate(candidate_payload),
        ATOM_ROWS,
        capability,
        geometry=_b2_geometry(),
    )
    payload = multicoord_request_payload(
        compiled, _mc_materials(_b2_geometry()), {"capability": capability}
    )
    raw = _frames_following_schedule(compiled)
    report = recover_frames(_parse(compiled, raw), compiled)
    extension = to_path_bundle_extension(report)
    proposal = _freeze_proposal_fixture(candidate=candidate_payload)
    plan = freeze_generation_plan(proposal, candidate_payload, compiled, _freeze_config())

    driver_maps = [frozenset(row.maps) for row in compiled.coordinates]
    assert driver_maps == [frozenset((1, 2)), frozenset((3, 4))]

    assert payload["metadata"]["n_drivers"] == 2
    assert payload["metadata"]["driver_ids"] == [DRIVER_01, DRIVER_02]
    payload_maps = [frozenset(row["atom_map_ids"]) for row in payload["coordinates"]]
    assert payload_maps == driver_maps

    for record in report.frames:
        assert set(record.targets_by_driver) == {DRIVER_01, DRIVER_02}
        assert set(record.actuals_by_driver) == {DRIVER_01, DRIVER_02}
        assert set(record.residuals_by_driver) == {DRIVER_01, DRIVER_02}

    extension_frames = extension["frames"]
    assert len(extension_frames) == report.n_frames
    for frame in extension_frames:
        assert set(frame["all_driver_targets"]) == {DRIVER_01, DRIVER_02}
        assert set(frame["all_driver_actuals"]) == {DRIVER_01, DRIVER_02}
        assert set(frame["all_driver_residuals"]) == {DRIVER_01, DRIVER_02}
    assert PATH_BUNDLE_EXTENSION_KEY in extension or "semantics" in extension

    frozen_candidate = plan["candidates"][0]
    frozen_maps = [frozenset(d["maps"]) for d in frozen_candidate["drivers"]]
    assert frozen_maps == driver_maps
    binding_maps = [
        frozenset(row["maps"])
        for row in plan["extensions"]["freeze"]["compiled_request"]["coordinates"]
    ]
    assert binding_maps == driver_maps


# ---------------------------------------------------------------------------
# Phase 7 — explicit P2 gated-smoke status (passed/blocked vs env reality).
# ---------------------------------------------------------------------------
def test_p2_gated_smoke_status_is_explicit_passed_or_blocked(tmp_path) -> None:
    """Status function returns passed|blocked and matches env reality."""
    report = _p2_gated_smoke_status()
    assert report["status"] in {"passed", "blocked"}

    env_present = all(
        os.environ.get(name)
        for name in (ENV_ACP_ROOT, ENV_ACP_PYTHON, ENV_ORCA_EXECUTABLE)
    )
    if not env_present:
        # This tree: gated env absent → blocked, missing vars named explicitly.
        assert report["status"] == "blocked"
        assert report["missing_env"], "blocked status must list the missing env vars"
        for name in (ENV_ACP_ROOT, ENV_ACP_PYTHON, ENV_ORCA_EXECUTABLE):
            if not os.environ.get(name):
                assert name in report["missing_env"]

    # Helper contract: env ready + pass receipts → passed (injected, no real run).
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    for probe_id, family in (
        ("acp-single-b-smoke", "acp"),
        ("orca-single-b-smoke", "orca"),
    ):
        (receipts / f"probe_receipt_{probe_id}.json").write_text(
            json.dumps(
                {
                    "probe_id": probe_id,
                    "family": family,
                    "status": "pass",
                    "modes": ["SINGLE_1D"],
                }
            ),
            encoding="utf-8",
        )
    full_env = {
        ENV_ACP_ROOT: "/fake/acp",
        ENV_ACP_PYTHON: "/fake/python",
        ENV_ORCA_EXECUTABLE: "/fake/orca",
    }
    passed = _p2_gated_smoke_status(environ=full_env, receipts_dir=receipts)
    assert passed["status"] == "passed"
    assert passed["env_ready"] is True
    assert passed["families"] == {"acp": True, "orca": True}

    # Env ready but no recorded receipts → still blocked (never optimistic).
    empty_dir = tmp_path / "empty_receipts"
    empty_dir.mkdir()
    unrecorded = _p2_gated_smoke_status(environ=full_env, receipts_dir=empty_dir)
    assert unrecorded["status"] == "blocked"

    # Env absent regardless of receipts → blocked with the env list.
    no_env = _p2_gated_smoke_status(environ={}, receipts_dir=receipts)
    assert no_env["status"] == "blocked"
    assert set(no_env["missing_env"]) == {
        ENV_ACP_ROOT,
        ENV_ACP_PYTHON,
        ENV_ORCA_EXECUTABLE,
    }


# ---------------------------------------------------------------------------
# End-to-end happy path — the whole P2 multi-coordinate loop.
# ---------------------------------------------------------------------------
def test_end_to_end_multi_coordinate_loop_happy_path(tmp_path) -> None:
    """Selector → compile → adapter → recovery → quality → freeze, all real."""
    capability = _smoke(tmp_path)  # injected passing Simul_Scan probe receipts

    # 1. Real selector produces a 2-driver COUPLED candidate.
    proposal = _exchange_selector_outputs(tmp_path, capability, _exchange_budget_config())
    selected = _pick_coupled_candidate(proposal)
    candidate_id = str(selected["candidate_id"])
    assert "PRUNED_BY_BUDGET" not in {
        str(reason.get("code")) for reason in selected.get("failure_reasons") or ()
    }

    # 2. compile_orca on the selector output (both drivers, aligned λ).
    compiled = co.compile_orca(
        _as_scan_candidate(selected),
        EXCHANGE_ATOM_ROWS,
        capability,
        geometry=_exchange_geometry(),
    )
    assert compiled.candidate_id == candidate_id
    assert len(compiled.coordinates) == 2
    assert compiled.total_points == len(compiled.lambda_values)

    # 3. ACP multicoord payload carries every driver.
    payload = multicoord_request_payload(
        compiled, _exchange_materials(), {"capability": capability}
    )
    assert payload["metadata"]["n_drivers"] == 2
    assert payload["metadata"]["driver_ids"] == [DRIVER_01, DRIVER_02]
    assert len(payload["coordinates"]) == 2
    assert payload["lambda_values"] == list(compiled.lambda_values)

    # 4. Synthetic attempt → per-frame all-driver recovery (complete).
    raw = _frames_following_schedule(compiled)
    result = _parse(compiled, raw)
    assert result.complete is True
    report = recover_frames(result, compiled)
    assert report.complete is True
    assert report.n_frames == compiled.total_points
    for record in report.frames:
        assert set(record.targets_by_driver) == {DRIVER_01, DRIVER_02}
        assert set(record.actuals_by_driver) == {DRIVER_01, DRIVER_02}
        assert set(record.residuals_by_driver) == {DRIVER_01, DRIVER_02}

    # 5. Target quality on the recovered frames (targets derived, not hardcoded).
    bond_targets = _bond_targets_from_geometry(compiled)
    assert len(bond_targets) == 2
    assessment = assess_target_path(
        report, {"bonds": bond_targets, "regions": [], "stereo": []}, {}
    )
    assert set(assessment.tiers) == set(TIERS)
    assert assessment.tiers[TIER_EXECUTION_COMPLETE].status == LABEL_ACHIEVED
    assert assessment.tiers[TIER_NUMERICALLY_USABLE].status == LABEL_ACHIEVED
    # Actuals follow the schedule to the P endpoint → both edits realized.
    assert assessment.tiers[TIER_TARGET_PATH_COMPATIBLE].status == LABEL_ACHIEVED
    # No TS evidence refs → validated_ts honestly absent.
    assert assessment.tiers[TIER_VALIDATED_TS].status == LABEL_NOT_ACHIEVED

    # 6. Freeze: raw gate-carrying candidate is refused (red line armed);
    #    the post-gate projection with the real compiled binding freezes.
    with pytest.raises(PlanFreezeError) as raw_excinfo:
        freeze_generation_plan(proposal, selected, compiled, _freeze_config())
    assert raw_excinfo.value.code == "CANDIDATE_NOT_CLEAN"

    freeze_proposal = _freeze_ready_proposal_from_selector(proposal, selected)
    plan = freeze_generation_plan(
        freeze_proposal, freeze_proposal["candidates"][0], compiled, _freeze_config()
    )
    assert validate_v2_document(plan) == []
    assert verify_generation_plan(plan) == []
    assert plan["candidates"][0]["candidate_id"] == candidate_id
    assert len(plan["candidates"][0]["drivers"]) == 2

    # 7. No first-driver projection anywhere down the chain.
    assert len(payload["coordinates"]) == len(compiled.coordinates) == 2
    binding = plan["extensions"]["freeze"]["compiled_request"]
    assert len(binding["coordinates"]) == 2
    for frame in to_path_bundle_extension(report)["frames"]:
        assert set(frame["all_driver_targets"]) == {DRIVER_01, DRIVER_02}

    # 8. P2 gated-smoke status is explicit in this tree.
    smoke = _p2_gated_smoke_status()
    assert smoke["status"] in {"passed", "blocked"}
    if not smoke["env_ready"]:
        assert smoke["status"] == "blocked"
