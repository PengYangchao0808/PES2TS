"""Offline tests for the scan-strategy selector core (plan todo 17).

Every fixture is synthetic and in-memory: a mapped reaction SMILES plus
map-keyed endpoint materials, or the frozen ``tests/fixtures/p0_demo24``
snapshots.  No test reads the real ``data/`` tree.  The suite locks:

* the ``propose_strategies`` pipeline (P0 chain → registry → coordinate pool
  → geometry feasibility → schedules → direction/assembly → capabilities →
  rank/dedup/budget → sealed ``g1_strategy_proposal_v1``);
* per-candidate ``failure_reasons`` separate from whole-case
  ``blocking_reasons``; ``execution_eligible`` always false;
* the six-layer ranking order (§8.3) plus the stable residual tie-break;
* ``PRUNED_BY_BUDGET`` traceability and single-candidate failure isolation;
* ``all_release_gates_pass`` (empty runnable subgraph → false; every fallback
  reference must exist in the strategy registry);
* determinism (two runs byte-identical minus ``created_at``);
* the Demo24 sweep: every frozen record yields a valid proposal or a typed
  exit, ``validate_v2_document`` passes, zero forbidden truth keys.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from rdkit import rdBase

from pes2ts_core.contracts import FORBIDDEN_TRUTH_KEYS
from pes2ts_core.g1.v2_verify import FORBIDDEN_EXPORT_KEYS
from pes2ts_core.scan_strategy.graph_rebuild import (
    load_endpoint_materials_from_export,
    rebuild_endpoint_graphs,
)
from pes2ts_core.scan_strategy.selector import (
    RankInputs,
    all_release_gates_pass,
    propose_strategies,
    rank_rank_inputs,
)
from pes2ts_core.scan_strategy.contracts_v2 import validate_v2_document
from pes2ts_core.utils.hashing import stable_json_dumps

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "p0_demo24"
FORBIDDEN: frozenset[str] = frozenset(
    {key.lower() for key in FORBIDDEN_TRUTH_KEYS}
    | {key.lower() for key in FORBIDDEN_EXPORT_KEYS}
    | {"endpoint_match", "orientation", "irc_evidence"}
)

# ---------------------------------------------------------------------------
# Synthetic mapped-SMILES fixtures + materials (whitelisted P0 inputs).
# ---------------------------------------------------------------------------
#: Single bond formation: fragments 1-2 and 3-4 associate into a chain.
SINGLE_BOND_SMILES = (
    "[CH3:1][CH3:2].[CH3:3][CH3:4]>>[CH3:1][CH2:2][CH2:3][CH3:4]"
)
SINGLE_BOND_R = {1: (0.0, 0.0, 0.0), 2: (1.54, 0.0, 0.0),
                 3: (6.0, 0.0, 0.0), 4: (7.54, 0.0, 0.0)}
SINGLE_BOND_P = {1: (0.0, 0.0, 0.0), 2: (1.54, 0.0, 0.0),
                 3: (3.08, 0.0, 0.0), 4: (4.62, 0.0, 0.0)}

#: Pure H transfer: H4 migrates atom2 → atom1 (spectator C3 apart).
H_TRANSFER_SMILES = (
    "[CH3:1][CH2:2][H:4].[CH3:3]>>[CH2:1]([H:4])[CH2:2].[CH3:3]"
)
H_TRANSFER_R = {1: (0.0, 0.0, 0.0), 2: (1.54, 0.0, 0.0),
                4: (2.05, 0.95, 0.35), 3: (8.0, 0.0, 0.0)}
H_TRANSFER_P = {1: (0.0, 0.0, 0.0), 2: (1.54, 0.0, 0.0),
                4: (-0.55, 0.95, 0.25), 3: (8.0, 0.0, 0.0)}
#: Collinear D–H–A at both endpoints: A drivers degenerate, B drivers fine.
H_TRANSFER_COLLINEAR_R = {1: (0.0, 0.0, 0.0), 2: (1.54, 0.0, 0.0),
                          4: (2.63, 0.0, 0.0), 3: (8.0, 0.0, 0.0)}
H_TRANSFER_COLLINEAR_P = {1: (0.0, 0.0, 0.0), 2: (1.54, 0.0, 0.0),
                          4: (-1.09, 0.0, 0.0), 3: (8.0, 0.0, 0.0)}

#: Coupled two-bond connectivity exchange: formed (2,3) + broken (1,2), shared center 2.
EXCHANGE_SMILES = (
    "[CH3:1][CH2:2][H:4].[CH3:3]>>[CH3:3][CH2:2][H:4].[CH3:1]"
)
EXCHANGE_R = {2: (0.0, 0.0, 0.0), 1: (1.54, 0.0, 0.0),
              4: (0.35, 1.05, 0.0), 3: (6.0, 6.0, 0.0)}
EXCHANGE_P = {2: (0.0, 0.0, 0.0), 3: (1.54, 0.0, 0.0),
              4: (0.35, 1.05, 0.0), 1: (6.0, 0.0, 0.0)}

#: R ≡ P (no reaction change) → typed rejection, still a proposal document.
NO_EDIT_SMILES = "[CH3:1][CH2:2][CH3:3]>>[CH3:1][CH2:2][CH3:3]"
NO_EDIT_XYZ = {1: (0.0, 0.0, 0.0), 2: (1.54, 0.0, 0.0), 3: (3.08, 0.0, 0.0)}

#: Metal input → SPECIAL_DOMAIN typed exit (watershed, design §5.1 step 1).
METAL_SMILES = "[Fe:1][CH3:2]>>[Fe:1][CH3:2]"
METAL_XYZ = {1: (0.0, 0.0, 0.0), 2: (1.9, 0.0, 0.0)}


def _materials(
    r_coordinates: Mapping[int, tuple[float, float, float]],
    p_coordinates: Mapping[int, tuple[float, float, float]],
    *,
    electronic: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, Any]:
    materials: dict[str, Any] = {
        "r_coordinates": {int(k): list(v) for k, v in r_coordinates.items()},
        "p_coordinates": {int(k): list(v) for k, v in p_coordinates.items()},
    }
    if electronic is not None:
        materials["endpoint_electronic"] = {
            "reactant": dict(electronic["reactant"]),
            "product": dict(electronic["product"]),
        }
    return materials


def _config(**scan_overrides: Any) -> dict[str, Any]:
    """Minimal config mapping mirroring config/defaults.yaml scan_strategy."""
    scan: dict[str, Any] = {
        "context_radius": 2,
        "max_scan_coordinates": 3,
        "point_limits": {"baseline": 9, "max": 101},
        "schedule_budget": 3,
        "direction_budget": 2,
        "assembly_candidate_budget": 2,
        "max_total_candidates": {"scan": 6, "path_neb": 1},
        "max_step_by_kind": {"distance": 0.2, "angle": 10.0, "dihedral": 10.0},
        "jacobian_condition_number_max": 1.0e6,
        "element_pair_bond_thresholds": {"tolerance": 0.45},
        "special_domain_policy": "unsupported",
    }
    scan.update(scan_overrides)
    return {
        "scan_strategy": scan,
        "g2": {"validity": {"collision_min_distance": 0.8}},
    }


def _bundle(smiles: str, materials: Mapping[str, Any]):
    from pes2ts_core.g1.parse import parse_reaction

    reactants, products = parse_reaction(smiles)
    export_like: dict[str, dict[int, dict[str, Any]]] = {"r": {}, "p": {}}
    for side_key, side in (("r", reactants), ("p", products)):
        source = materials["r_coordinates"] if side_key == "r" else materials["p_coordinates"]
        for mol, map_list in zip(side.mols, side.map_lists, strict=True):
            for atom, map_number in zip(mol.GetAtoms(), map_list, strict=True):
                map_id = int(map_number)
                export_like[side_key][map_id] = {
                    "element": atom.GetSymbol(),
                    "coordinates": list(source[map_id]),
                }
    return rebuild_endpoint_graphs(smiles, export_like)


def _propose(
    smiles: str,
    r_coordinates: Mapping[int, tuple[float, float, float]],
    p_coordinates: Mapping[int, tuple[float, float, float]],
    *,
    config: Mapping[str, Any] | None = None,
    reaction_id: str = "RXN_SYNTH_SELECTOR",
    electronic: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, Any]:
    materials = _materials(r_coordinates, p_coordinates, electronic=electronic)
    bundle = _bundle(smiles, materials)
    return propose_strategies(
        bundle,
        materials,
        config if config is not None else _config(),
        reaction_id=reaction_id,
        split="unassigned",
    )


def _failure_codes(candidate: Mapping[str, Any]) -> list[str]:
    reasons = candidate.get("failure_reasons") or []
    return [str(reason.get("code")) for reason in reasons if isinstance(reason, Mapping)]


def _collect_keys(value: Any, found: set[str]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            found.add(str(key))
            _collect_keys(child, found)
    elif isinstance(value, list):
        for item in value:
            _collect_keys(item, found)


def _strip_created_at(document: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in document.items() if k != "created_at"}


def _load_manifest() -> dict[str, Any]:
    return json.loads((FIXTURE_ROOT / "manifest.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Ranking: six design-§8.3 layers + stable residual tie-break.
# ---------------------------------------------------------------------------
def _rank_pair(**layer_diff: Any) -> list[RankInputs]:
    """Two candidates identical except for the named ranking layer(s)."""
    base_kwargs: dict[str, Any] = dict(
        has_uncovered_event=False,
        geometry_ok=True,
        assembly_ready=True,
        capability_status="pass",
        constraint_rank=1,
        n_follow_along=0,
        cost=9,
        driver_maps_key=((2, 3),),
        kinds_key=("B",),
    )
    worse_kwargs = dict(base_kwargs)
    better_kwargs = dict(base_kwargs)
    worse_kwargs.update(layer_diff.get("worse", {}))
    better_kwargs.update(layer_diff.get("better", {}))
    worse = RankInputs(**worse_kwargs)  # type: ignore[arg-type]
    better = RankInputs(**better_kwargs)  # type: ignore[arg-type]
    return rank_rank_inputs([worse, better])


def test_rank_layer_a_uncovered_events_sort_last() -> None:
    # Given two candidates differing only in uncovered-event status
    ranked = _rank_pair(worse={"has_uncovered_event": True})

    # Then the covered candidate ranks first
    assert ranked[0].has_uncovered_event is False
    assert ranked[1].has_uncovered_event is True


def test_rank_layer_b_geometry_qualified_first() -> None:
    ranked = _rank_pair(worse={"geometry_ok": False})

    assert ranked[0].geometry_ok is True
    assert ranked[1].geometry_ok is False


def test_rank_layer_c_assembly_ready_first() -> None:
    ranked = _rank_pair(worse={"assembly_ready": False})

    assert ranked[0].assembly_ready is True
    assert ranked[1].assembly_ready is False


def test_rank_layer_c_capability_pass_before_unknown_before_fail() -> None:
    ordered = rank_rank_inputs(
        [
            RankInputs(False, True, True, "fail", 1, 0, 9, ((2, 3),), ("B",)),
            RankInputs(False, True, True, "unknown", 1, 0, 9, ((2, 3),), ("B",)),
            RankInputs(False, True, True, "pass", 1, 0, 9, ((2, 3),), ("B",)),
        ]
    )
    assert [view.capability_status for view in ordered] == ["pass", "unknown", "fail"]


def test_rank_layer_d_fewer_constraints_first() -> None:
    ranked = _rank_pair(worse={"constraint_rank": 2})

    assert ranked[0].constraint_rank == 1
    assert ranked[1].constraint_rank == 2


def test_rank_layer_e_fewer_follow_along_assumptions_first() -> None:
    ranked = _rank_pair(worse={"n_follow_along": 2})

    assert ranked[0].n_follow_along == 0
    assert ranked[1].n_follow_along == 2


def test_rank_layer_f_lower_cost_first() -> None:
    ranked = _rank_pair(worse={"cost": 27})

    assert ranked[0].cost == 9
    assert ranked[1].cost == 27


def test_rank_residual_tie_breaks_lexicographically_on_driver_maps_then_kinds() -> None:
    # Given two otherwise identical candidates, only the driver identity differs
    high = RankInputs(False, True, True, "pass", 1, 0, 9, ((3, 4),), ("B",))
    low = RankInputs(False, True, True, "pass", 1, 0, 9, ((1, 2),), ("B",))
    same_maps_worse_kind = RankInputs(False, True, True, "pass", 1, 0, 9, ((1, 2),), ("D",))

    # Then map-tuple order decides first, kind order second
    ranked = rank_rank_inputs([high, low, same_maps_worse_kind])
    assert ranked[0].driver_maps_key == ((1, 2),)
    assert ranked[0].kinds_key == ("B",)
    assert ranked[1].driver_maps_key == ((1, 2),)
    assert ranked[1].kinds_key == ("D",)
    assert ranked[2].driver_maps_key == ((3, 4),)


def test_rank_inputs_ranking_key_order_is_total_and_deterministic() -> None:
    views = [
        RankInputs(True, False, False, "fail", 3, 4, 100, ((9, 9),), ("D",)),
        RankInputs(False, True, True, "pass", 1, 0, 9, ((1, 2),), ("B",)),
    ]
    first = [v.ranking_key() for v in rank_rank_inputs(views)]
    second = [v.ranking_key() for v in rank_rank_inputs(list(reversed(views)))]
    assert first == second
    assert first[0] < first[1]


# ---------------------------------------------------------------------------
# Synthetic integration: each reaction yields a sealed, valid proposal.
# ---------------------------------------------------------------------------
def test_single_bond_reaction_yields_valid_proposal() -> None:
    # Given a single-bond formation with sane endpoint geometry
    doc = _propose(SINGLE_BOND_SMILES, SINGLE_BOND_R, SINGLE_BOND_P)

    # Then a sealed proposal validates and stays execution-ineligible
    assert validate_v2_document(doc) == []
    assert doc["execution_eligible"] is False
    assert doc["schema_name"] == "StrategyProposal"
    assert doc["schema_version"] == "g1_strategy_proposal_v1"
    assert doc["epistemic_status"] == "endpoint_hypothesis"
    assert doc["status"] in {"proposed", "needs_review"}
    assert doc["candidates"], "single-bond reaction must yield candidates"
    for candidate in doc["candidates"]:
        assert candidate["mode"] in {"SINGLE_1D", "COUPLED_1D", "SCHEDULED_1D"}
        assert candidate["drivers"], "scan candidates require drivers"
        assert candidate["lambda_values"], "lambda grid must be non-empty"
        assert candidate["capability_check"]["status"] in {"pass", "fail", "unknown"}
    assert any(
        reason.get("code") == "PROPOSED_FROM_ENDPOINT_HYPOTHESIS"
        for reason in doc["reasons"]
    )


def test_h_transfer_reaction_yields_valid_proposal() -> None:
    doc = _propose(H_TRANSFER_SMILES, H_TRANSFER_R, H_TRANSFER_P)

    assert validate_v2_document(doc) == []
    assert doc["execution_eligible"] is False
    assert doc["candidates"], "H-transfer reaction must yield candidates"
    assert "H_TRANSFER" in doc["motif_tags"]
    modes = {candidate["mode"] for candidate in doc["candidates"]}
    assert modes & {"SINGLE_1D", "COUPLED_1D", "SCHEDULED_1D"}


def test_coupled_two_bond_exchange_yields_valid_proposal() -> None:
    doc = _propose(EXCHANGE_SMILES, EXCHANGE_R, EXCHANGE_P)

    assert validate_v2_document(doc) == []
    assert doc["execution_eligible"] is False
    assert doc["candidates"], "connectivity-exchange reaction must yield candidates"
    assert "CONNECTIVITY_EXCHANGE" in doc["motif_tags"]
    # The two-bond coupled driver set must appear with 2..3 drivers under a coupled/scheduled mode
    coupled = [
        c for c in doc["candidates"] if c["mode"] in {"COUPLED_1D", "SCHEDULED_1D"}
    ]
    assert coupled, "exchange route must propose multi-coordinate candidates"
    assert all(2 <= len(c["drivers"]) <= 3 for c in coupled)


def test_no_edit_reaction_still_yields_proposal_with_reasons_and_gate_false() -> None:
    # Given R ≡ P (no reaction change)
    doc = _propose(NO_EDIT_SMILES, NO_EDIT_XYZ, NO_EDIT_XYZ)

    # Then a rejected proposal retains typed reasons and the release gate fails
    assert validate_v2_document(doc) == []
    assert doc["status"] == "rejected"
    assert doc["execution_eligible"] is False
    assert doc["candidates"] == []
    assert doc["blocking_reasons"], "rejected proposal must explain refusal"
    assert any(
        reason.get("code") == "NO_REACTION_CHANGE"
        for reason in doc["blocking_reasons"]
    )
    assert all_release_gates_pass(doc) is False


def test_metal_input_special_domain_exit_is_typed_rejection() -> None:
    doc = _propose(METAL_SMILES, METAL_XYZ, METAL_XYZ)

    assert validate_v2_document(doc) == []
    assert doc["status"] == "rejected"
    assert doc["execution_eligible"] is False
    assert any(
        reason.get("code") == "SPECIAL_DOMAIN"
        for reason in doc["blocking_reasons"]
    )
    assert all_release_gates_pass(doc) is False


# ---------------------------------------------------------------------------
# Failure isolation + budget traceability.
# ---------------------------------------------------------------------------
def test_single_candidate_failure_does_not_mask_other_candidates() -> None:
    # Given an H-transfer reaction whose D–H–A atoms are collinear: A-driver
    # candidates fail geometry while B-driver candidates stay feasible.
    doc = _propose(
        H_TRANSFER_SMILES,
        H_TRANSFER_COLLINEAR_R,
        H_TRANSFER_COLLINEAR_P,
        reaction_id="RXN_SYNTH_COLLINEAR",
    )

    assert validate_v2_document(doc) == []
    assert doc["candidates"], "feasible B-driver candidates must survive"
    codes_per_candidate = [_failure_codes(c) for c in doc["candidates"]]
    has_angle_failure = any(
        "ANGLE_DEGENERATE" in codes or "JACOBIAN_RANK_DEFICIENT" in codes
        for codes in codes_per_candidate
    )
    has_clean_candidate = any(
        "ANGLE_DEGENERATE" not in codes and "JACOBIAN_RANK_DEFICIENT" not in codes
        for codes in codes_per_candidate
    )
    assert has_angle_failure, "degenerate A-driver candidates must be recorded"
    assert has_clean_candidate, "one failing candidate must not mask the others"
    # Whole-case blocking reasons stay reserved for route-level refusals
    assert not any(
        reason.get("code") == "ANGLE_DEGENERATE" for reason in doc["blocking_reasons"]
    )


def test_pruned_by_budget_is_traceable_in_candidate_failure_reasons() -> None:
    # Given the default scan budget of 6 against a reaction that proposes more
    doc = _propose(H_TRANSFER_SMILES, H_TRANSFER_R, H_TRANSFER_P)

    assert validate_v2_document(doc) == []
    pruned = [
        c
        for c in doc["candidates"]
        if "PRUNED_BY_BUDGET" in _failure_codes(c)
    ]
    assert pruned, "over-budget candidates must be recorded with PRUNED_BY_BUDGET"
    for candidate in pruned:
        details = [
            str(reason.get("detail", ""))
            for reason in candidate["failure_reasons"]
            if isinstance(reason, Mapping) and reason.get("code") == "PRUNED_BY_BUDGET"
        ]
        assert details, "PRUNED_BY_BUDGET reasons must carry a detail"
    runnable = [c for c in doc["candidates"] if "PRUNED_BY_BUDGET" not in _failure_codes(c)]
    assert runnable, "budget pruning must keep the top-ranked candidates"


def test_schedule_budget_prune_merges_pruned_by_budget_code() -> None:
    # Given schedule_budget=1: the single-bond candidate keeps linear only and
    # the named smoothstep schedule is pruned — recorded, never silent.
    config = _config(schedule_budget=1)
    doc = _propose(SINGLE_BOND_SMILES, SINGLE_BOND_R, SINGLE_BOND_P, config=config)

    assert validate_v2_document(doc) == []
    codes = [code for c in doc["candidates"] for code in _failure_codes(c)]
    assert "PRUNED_BY_BUDGET" in codes


def test_budget_prune_keeps_top_ranked_candidates_first() -> None:
    # Given scan budget 4 (schedules survive todo-14's n_active×direction
    # projection for this reaction) but far fewer than the assembled views
    config = _config(max_total_candidates={"scan": 4, "path_neb": 1})
    doc = _propose(H_TRANSFER_SMILES, H_TRANSFER_R, H_TRANSFER_P, config=config)

    assert validate_v2_document(doc) == []
    assert len(doc["candidates"]) > 4
    active = [c for c in doc["candidates"] if "PRUNED_BY_BUDGET" not in _failure_codes(c)]
    pruned = [c for c in doc["candidates"] if "PRUNED_BY_BUDGET" in _failure_codes(c)]
    assert len(active) == 4
    assert pruned, "beyond-budget candidates stay in the record"
    assert all_release_gates_pass(doc) is True


# ---------------------------------------------------------------------------
# Capability honesty: unprobed modes recorded, never silently enabled.
# ---------------------------------------------------------------------------
def test_unprobed_modes_recorded_unknown_and_never_enabled() -> None:
    doc = _propose(SINGLE_BOND_SMILES, SINGLE_BOND_R, SINGLE_BOND_P)

    statuses = {
        (c["mode"], c["capability_check"]["status"]) for c in doc["candidates"]
    }
    # Shipped registry: SINGLE_1D declared but unprobed → unknown, never pass
    assert ("SINGLE_1D", "unknown") in statuses or (
        "SINGLE_1D" in {c["mode"] for c in doc["candidates"]}
        and all(
            c["capability_check"]["status"] != "pass"
            for c in doc["candidates"]
            if c["mode"] == "SINGLE_1D"
        )
    )
    for candidate in doc["candidates"]:
        check = candidate["capability_check"]
        assert check["status"] in {"pass", "fail", "unknown"}
        if check["status"] == "unknown":
            assert check["missing"], "unknown status must list what is missing"


def test_coupled_mode_beyond_adapter_capability_fails_loudly() -> None:
    doc = _propose(EXCHANGE_SMILES, EXCHANGE_R, EXCHANGE_P)

    coupled = [c for c in doc["candidates"] if c["mode"] == "COUPLED_1D"]
    assert coupled, "exchange route proposes COUPLED_1D candidates"
    for candidate in coupled:
        assert candidate["capability_check"]["status"] == "fail"
        assert candidate["capability_check"]["missing"]
    # Failed-capability candidates remain in the record with typed reasons
    assert any(
        "BACKEND_CAPABILITY_MISSING" in _failure_codes(c) or
        "CAPABILITY" in "".join(_failure_codes(c))
        for c in coupled
    )


# ---------------------------------------------------------------------------
# all_release_gates_pass semantics.
# ---------------------------------------------------------------------------
def _minimal_candidate(
    candidate_id: str = "cand-0000",
    *,
    failure_codes: tuple[str, ...] = (),
    fallback_ids: tuple[str, ...] = ("NETWORK_PATH",),
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "mode": "SINGLE_1D",
        "start_endpoint": "R",
        "direction": "R_to_P",
        "assembly_id": "asm-R-000",
        "anchor_reason": "LOCAL_CONNECTIVITY",
        "drivers": [{"kind": "B", "maps": [2, 3], "unit": "angstrom"}],
        "monitors": [],
        "guards": [],
        "event_coverage": [],
        "lambda_values": [0.0, 1.0],
        "schedule_id": "sched:linear",
        "required_capabilities": ["SINGLE_1D"],
        "capability_check": {"status": "unknown", "missing": []},
        "budget": {"max_attempts": 1, "max_cpu_hours": 0.0, "max_wall_seconds": 0},
        "failure_reasons": [{"code": code} for code in failure_codes],
        "fallback_ids": list(fallback_ids),
    }


def test_release_gate_false_when_runnable_subgraph_empty() -> None:
    # Given a proposal whose candidates are all pruned by budget
    proposal = {"candidates": [_minimal_candidate(failure_codes=("PRUNED_BY_BUDGET",))]}

    # Then the release gate fails
    assert all_release_gates_pass(proposal) is False


def test_release_gate_false_when_no_candidates() -> None:
    assert all_release_gates_pass({"candidates": []}) is False
    assert all_release_gates_pass({}) is False


def test_release_gate_true_with_runnable_candidate_and_defined_fallbacks() -> None:
    proposal = {"candidates": [_minimal_candidate()]}

    assert all_release_gates_pass(proposal) is True


def test_release_gate_false_when_fallback_reference_undefined_in_registry() -> None:
    # Given a runnable candidate whose fallback names a non-registry strategy
    proposal = {
        "candidates": [
            _minimal_candidate(fallback_ids=("NOT_A_REGISTRY_STRATEGY",))
        ]
    }

    # Then the release gate fails even though a runnable candidate exists
    assert all_release_gates_pass(proposal) is False


# ---------------------------------------------------------------------------
# Determinism.
# ---------------------------------------------------------------------------
def test_two_runs_produce_byte_identical_docs_minus_created_at() -> None:
    first = _propose(SINGLE_BOND_SMILES, SINGLE_BOND_R, SINGLE_BOND_P)
    second = _propose(SINGLE_BOND_SMILES, SINGLE_BOND_R, SINGLE_BOND_P)

    assert first["content_sha256"] == second["content_sha256"]
    assert stable_json_dumps(_strip_created_at(first)) == stable_json_dumps(
        _strip_created_at(second)
    )


# ---------------------------------------------------------------------------
# Demo24 sweep over the frozen fixture (offline, git-tracked inputs).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "record_path",
    sorted((FIXTURE_ROOT / "records").glob("RXN_*.json")),
    ids=lambda p: p.stem,
)
def test_demo24_every_record_yields_valid_proposal_or_typed_exit(
    record_path: Path,
) -> None:
    manifest = _load_manifest()
    pinned = str(manifest["rdkit_version"])
    if rdBase.rdkitVersion != pinned:
        pytest.fail(
            f"rdkit {rdBase.rdkitVersion} != fixture pin {pinned} (never-skip gate)"
        )
    from pes2ts_core.config_loader import load_config

    snapshot = json.loads(record_path.read_text(encoding="utf-8"))
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
    config = load_config()
    doc = propose_strategies(
        bundle,
        materials,
        config,
        reaction_id=str(snapshot["reaction_id"]),
        split=str(snapshot.get("split", "unassigned")),
    )

    # Then every record yields a sealed proposal or typed exit — never a crash
    assert isinstance(doc, dict)
    issues = validate_v2_document(doc)
    assert issues == [], f"{record_path.name}: {issues[:5]}"
    assert doc["execution_eligible"] is False
    found: set[str] = set()
    _collect_keys(doc, found)
    hits = sorted(found & FORBIDDEN)
    assert hits == [], f"{record_path.name}: forbidden keys {hits}"
    if doc["status"] == "rejected":
        assert doc["blocking_reasons"], record_path.name
    else:
        assert doc["reasons"], record_path.name
    gate = all_release_gates_pass(doc)
    assert isinstance(gate, bool)


def test_demo24_fixture_has_exactly_twentyfour_records() -> None:
    paths = sorted((FIXTURE_ROOT / "records").glob("RXN_*.json"))
    assert len(paths) == 24
