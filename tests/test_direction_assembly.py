"""Tests for todo 15 — direction & multi-component assembly policy (design §7)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from pes2ts_core.contracts import FORBIDDEN_TRUTH_KEYS
from pes2ts_core.g2.endpoints import ComponentGeometry, place_single
from pes2ts_core.g1.v2_verify import FORBIDDEN_EXPORT_KEYS
from pes2ts_core.scan_strategy.direction_assembly import (
    ANCHOR_DRIVER_BONDED_AT_START,
    ANCHOR_LAYER1_REVIEW_PRIORITY,
    ANCHOR_TIE_BIDIRECTIONAL,
    CODE_ASSEMBLY_COLLISION,
    CODE_ASSEMBLY_FRAME_MISMATCH,
    CODE_ASSEMBLY_READY,
    CODE_NO_USABLE_ENDPOINT,
    CODE_SPECTATOR_PRESERVED,
    HYBRID_RULE_FEWER_BOND_EVALUATED,
    HYBRID_RULE_IDS,
    HYBRID_RULE_MINIMAL_CONSTRAINTS,
    HYBRID_RULE_MORE_BOND_DEFAULT,
    HYBRID_RULE_REPRESENTATIVE_OR_PATH,
    HYBRID_RULE_TIE_GEOMETRY,
    LAYER_1,
    LAYER_2,
    LAYER_4,
    PRIOR_FB_NONE,
    PRIOR_FB_STRETCH_FROM_ANCHOR,
    SCHEMA_DIRECTION_ASSEMBLY,
    SELECTION_MECHANISM,
    SPLICE_POLICY,
    ApproachConfiguration,
    AnchorReason,
    ComponentMaterial,
    DirectionAssemblyInputs,
    DirectionCandidate,
    DriverBondSpec,
    DriverSetSpec,
    EndpointProfile,
    FBCounts,
    SideMaterial,
    assemble_start_side,
    compare_directions,
    fb_counts_from_edit_graph,
    hybrid_rule_ids_for,
    policy_from_config,
    resolve_direction_assembly,
)
from pes2ts_core.scan_strategy.registry import (
    STRATEGY_CONNECTIVITY_EXCHANGE,
    STRATEGY_H_TRANSFER,
    STRATEGY_LOCAL_CONNECTIVITY,
)

FORBIDDEN = {key.lower() for key in FORBIDDEN_TRUTH_KEYS} | {
    key.lower() for key in FORBIDDEN_EXPORT_KEYS
} | {"endpoint_match", "orientation", "irc_evidence"}


@dataclass(frozen=True)
class _FakeEditGraph:
    edit_counts: Mapping[str, int]


# ---------------------------------------------------------------------------
# Fixtures / helpers.
# ---------------------------------------------------------------------------
def _rot_z_90(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Rigid motion: +90° about z, then translate (5, -2, 1.5)."""
    return (-y + 5.0, x - 2.0, z + 1.5)


def _rigid_side(side: SideMaterial, transform) -> SideMaterial:
    return SideMaterial(
        endpoint=side.endpoint,
        components=tuple(
            ComponentMaterial(
                tag=c.tag,
                coordinates={m: transform(*xyz) for m, xyz in c.coordinates.items()},
                elements=dict(c.elements),
                contains_edit_atoms=c.contains_edit_atoms,
            )
            for c in side.components
        ),
    )


def _bimolecular_inputs(**overrides: Any) -> DirectionAssemblyInputs:
    """R = two fragments A{1,2} + B{3,4}; P = bonded AB; driver bond (3,1)."""
    base: dict[str, Any] = dict(
        reaction_id="RXN_TEST_15",
        fb_counts=FBCounts(formed=1, broken=0, order_changed=0),
        r_profile=EndpointProfile("R"),
        p_profile=EndpointProfile("P"),
        driver_sets=(
            DriverSetSpec(
                driver_set_id="ds-000",
                route_strategy_id=STRATEGY_LOCAL_CONNECTIVITY,
                driver_bonds=(DriverBondSpec(atom_maps=(3, 1)),),
            ),
        ),
        r_side=SideMaterial(
            "R",
            (
                ComponentMaterial(
                    "A", {1: (0.0, 0.0, 0.0), 2: (1.5, 0.0, 0.0)},
                    {1: "C", 2: "H"}, True,
                ),
                ComponentMaterial(
                    "B", {3: (5.0, 0.0, 0.0), 4: (6.5, 0.0, 0.0)},
                    {3: "C", 4: "H"}, True,
                ),
            ),
        ),
        p_side=SideMaterial(
            "P",
            (
                ComponentMaterial(
                    "AB",
                    {1: (0.0, 0.0, 0.0), 2: (1.5, 0.0, 0.0),
                     3: (2.9, 0.0, 0.0), 4: (4.4, 0.0, 0.0)},
                    {1: "C", 2: "H", 3: "C", 4: "H"}, True,
                ),
            ),
        ),
        bonded_pairs=frozenset({(1, 2), (3, 4)}),
        elements={1: "C", 2: "H", 3: "C", 4: "H"},
    )
    base.update(overrides)
    return DirectionAssemblyInputs(**base)


def _candidate_codes(candidate: DirectionCandidate) -> set[str]:
    return {reason.code for reason in candidate.anchor_reason}


def _strip_labels(record: dict[str, Any]) -> dict[str, Any]:
    """Remove direction labels (ids, endpoints, label-bearing prose) for R/P swap comparison."""
    import copy

    stripped = copy.deepcopy(record)
    stripped.pop("comparison", None)
    stripped.pop("fb_counts", None)
    candidates = stripped.get("direction_candidates", [])
    for candidate in candidates:
        for key in ("start_endpoint", "direction", "assembly_id", "bidirectional"):
            candidate.pop(key, None)
        for reason in candidate.get("anchor_reason", []):
            reason.pop("endpoint", None)
            reason.pop("detail", None)
        for asm_key in ("assembly", "target_assembly"):
            assembly = candidate.get(asm_key)
            if isinstance(assembly, dict):
                assembly.pop("endpoint", None)
                assembly.pop("direction", None)
                assembly.pop("assembly_id", None)
                assembly.pop("anchor_component_tag", None)
                for placement in assembly.get("placements", []):
                    placement.pop("component_tag", None)
                    placement.pop("onto_tag", None)
                for config in assembly.get("approach_configurations", []):
                    config.pop("configuration_id", None)
                    config.pop("moved_component_tag", None)
                    config.pop("fixed_component_tag", None)
                for mismatch in assembly.get("frame_mismatches", []):
                    mismatch.pop("component_tag", None)
                    mismatch.pop("detail", None)
                assembly["spectator_tags"] = sorted(
                    t.split(":", 1)[-1] for t in assembly.get("spectator_tags", [])
                )
    return stripped


# ---------------------------------------------------------------------------
# Layered comparison order.
# ---------------------------------------------------------------------------
def test_layer2_beats_layer3() -> None:
    """Layer-2 driver-bonded signal outranks richer layer-3 flags (§7 order)."""
    r_profile = EndpointProfile("R", driver_bonded_at_start=True)
    p_profile = EndpointProfile(
        "P", configuration_ok=True, orientation_ok=True, undriven_event_coverage_ok=True
    )
    comparison = compare_directions(r_profile, p_profile)
    assert comparison.preferred == "R"
    assert comparison.decided_layer == 2
    assert not comparison.bidirectional
    assert any(
        reason.code == ANCHOR_DRIVER_BONDED_AT_START and reason.endpoint == "R"
        for reason in comparison.reasons
    )


def test_layer3_decides_when_layer2_tied() -> None:
    r_profile = EndpointProfile("R", configuration_ok=True)
    p_profile = EndpointProfile("P", configuration_ok=True, orientation_ok=True)
    comparison = compare_directions(r_profile, p_profile)
    assert comparison.preferred == "P"
    assert comparison.decided_layer == 3


def test_tie_keeps_bidirectional() -> None:
    comparison = compare_directions(EndpointProfile("R"), EndpointProfile("P"))
    assert comparison.preferred is None
    assert comparison.bidirectional is True
    assert comparison.decided_layer == 4
    assert any(reason.code == ANCHOR_TIE_BIDIRECTIONAL for reason in comparison.reasons)


def test_review_completion_is_layer1_preference() -> None:
    """Completed electronic/geometry review on one side wins layer 1."""
    r_profile = EndpointProfile("R", electronic_review_complete=False)
    p_profile = EndpointProfile("P")
    comparison = compare_directions(r_profile, p_profile)
    assert comparison.preferred == "P"
    assert comparison.decided_layer == 1
    assert any(
        reason.code == ANCHOR_LAYER1_REVIEW_PRIORITY and reason.endpoint == "P"
        for reason in comparison.reasons
    )


def test_target_definition_hard_gate_blocks_both_sides() -> None:
    """Design §7: both sides must carry a usable target definition."""
    r_profile = EndpointProfile("R")
    p_profile = EndpointProfile("P", has_usable_target_definition=False)
    comparison = compare_directions(r_profile, p_profile)
    assert comparison.preferred is None
    assert comparison.layer1_eligible == ()
    assert comparison.reasons[0].code == CODE_NO_USABLE_ENDPOINT
    assert comparison.reasons[0].layer == LAYER_1


def test_compare_directions_rejects_swapped_profile_labels() -> None:
    with pytest.raises(ValueError, match="ENDPOINT_INVALID"):
        compare_directions(EndpointProfile("P"), EndpointProfile("R"))


# ---------------------------------------------------------------------------
# Demoted five-rule anchors + guardrail.
# ---------------------------------------------------------------------------
def test_hybrid_rule_ids_follow_fb_branches() -> None:
    assert hybrid_rule_ids_for(FBCounts(2, 2, 0)) == (
        HYBRID_RULE_TIE_GEOMETRY, HYBRID_RULE_MINIMAL_CONSTRAINTS,
    )
    assert hybrid_rule_ids_for(FBCounts(2, 1, 0)) == (
        HYBRID_RULE_MORE_BOND_DEFAULT, HYBRID_RULE_MINIMAL_CONSTRAINTS,
    )
    assert hybrid_rule_ids_for(FBCounts(5, 1, 0)) == (
        HYBRID_RULE_FEWER_BOND_EVALUATED, HYBRID_RULE_MINIMAL_CONSTRAINTS,
    )
    assert hybrid_rule_ids_for(FBCounts(5, 4, 0)) == (
        HYBRID_RULE_REPRESENTATIVE_OR_PATH, HYBRID_RULE_MINIMAL_CONSTRAINTS,
    )
    assert set(HYBRID_RULE_IDS) == {
        HYBRID_RULE_TIE_GEOMETRY,
        HYBRID_RULE_MORE_BOND_DEFAULT,
        HYBRID_RULE_FEWER_BOND_EVALUATED,
        HYBRID_RULE_REPRESENTATIVE_OR_PATH,
        HYBRID_RULE_MINIMAL_CONSTRAINTS,
    }


def test_rule_trace_records_demoted_five_rule_anchors() -> None:
    """LOCAL_CONNECTIVITY candidates carry the demoted anchors in rule_trace."""
    inputs = _bimolecular_inputs()
    result = resolve_direction_assembly(inputs)
    assert result.schema_version == SCHEMA_DIRECTION_ASSEMBLY
    assert len(result.direction_candidates) == 2  # tie → bidirectional
    for candidate in result.direction_candidates:
        assert candidate.route_strategy_id == STRATEGY_LOCAL_CONNECTIVITY
        for rule_id in hybrid_rule_ids_for(inputs.fb_counts):
            assert rule_id in candidate.rule_trace
        assert SELECTION_MECHANISM in candidate.rule_trace
        assert SPLICE_POLICY in candidate.rule_trace
        hybrid_layers = [
            reason for reason in candidate.anchor_reason if reason.layer == "hybrid_anchor"
        ]
        assert len(hybrid_layers) == len(hybrid_rule_ids_for(inputs.fb_counts))


def test_connectivity_exchange_also_gets_anchors_h_transfer_does_not() -> None:
    inputs = _bimolecular_inputs(
        driver_sets=(
            DriverSetSpec(
                "ds-000",
                STRATEGY_CONNECTIVITY_EXCHANGE,
                (DriverBondSpec((3, 1)),),
            ),
        ),
    )
    result = resolve_direction_assembly(inputs)
    assert all(
        HYBRID_RULE_MORE_BOND_DEFAULT in c.rule_trace
        for c in result.direction_candidates
    )

    inputs_h = _bimolecular_inputs(
        driver_sets=(
            DriverSetSpec(
                "ds-000", STRATEGY_H_TRANSFER, (DriverBondSpec((3, 1)),)
            ),
        ),
    )
    result_h = resolve_direction_assembly(inputs_h)
    for candidate in result_h.direction_candidates:
        assert not any(
            rule_id in candidate.rule_trace for rule_id in HYBRID_RULE_IDS
        )
        assert "hybrid_anchors_not_applicable_to_strategy" in candidate.rule_trace


def test_hybrid_anchors_do_not_override_layered_selection() -> None:
    """Guardrail: five-rule anchors are trace only; §7 layers select.

    F>B gives the cheap prior start=P, but layer 2 favours R → only the R
    candidate is emitted, with the prior recorded but not applied.
    """
    inputs = _bimolecular_inputs(
        fb_counts=FBCounts(formed=2, broken=1, order_changed=0),
        r_profile=EndpointProfile("R", driver_bonded_at_start=True),
        p_profile=EndpointProfile("P", configuration_ok=True),
        driver_sets=(
            DriverSetSpec(
                "ds-000", STRATEGY_LOCAL_CONNECTIVITY, (DriverBondSpec((3, 1)),)
            ),
        ),
    )
    result = resolve_direction_assembly(inputs)
    assert len(result.direction_candidates) == 1
    candidate = result.direction_candidates[0]
    assert candidate.start_endpoint == "R"
    assert candidate.direction == "R_to_P"
    # prior recorded (endpoint P) but selection stayed with layer-2 winner R
    priors = [
        reason for reason in candidate.anchor_reason if reason.layer == "prior"
    ]
    assert priors and priors[0].code == PRIOR_FB_STRETCH_FROM_ANCHOR
    assert priors[0].endpoint == "P"
    assert HYBRID_RULE_MORE_BOND_DEFAULT in candidate.rule_trace
    assert result.comparison.preferred == "R"
    assert result.comparison.decided_layer == 2


def test_fb_prior_none_recorded_on_tie() -> None:
    inputs = _bimolecular_inputs(fb_counts=FBCounts(1, 1, 0))
    result = resolve_direction_assembly(inputs)
    assert len(result.direction_candidates) == 2
    for candidate in result.direction_candidates:
        priors = [r for r in candidate.anchor_reason if r.layer == "prior"]
        assert priors and priors[0].code == PRIOR_FB_NONE
        assert candidate.bidirectional is True


# ---------------------------------------------------------------------------
# Equivalence properties.
# ---------------------------------------------------------------------------
def test_rigid_motion_invariance() -> None:
    """Rigid endpoint motions must not change equivalent strategy output."""
    inputs = _bimolecular_inputs()
    assert inputs.r_side is not None and inputs.p_side is not None
    moved = DirectionAssemblyInputs(
        reaction_id=inputs.reaction_id,
        fb_counts=inputs.fb_counts,
        r_profile=inputs.r_profile,
        p_profile=inputs.p_profile,
        driver_sets=inputs.driver_sets,
        r_side=_rigid_side(inputs.r_side, _rot_z_90),
        p_side=_rigid_side(inputs.p_side, _rot_z_90),
        bonded_pairs=inputs.bonded_pairs,
        elements=inputs.elements,
    )
    original = resolve_direction_assembly(inputs)
    transformed = resolve_direction_assembly(moved)
    assert original.comparison.decided_layer == transformed.comparison.decided_layer
    assert original.comparison.bidirectional == transformed.comparison.bidirectional
    assert len(original.direction_candidates) == len(transformed.direction_candidates)
    for left, right in zip(
        original.direction_candidates, transformed.direction_candidates, strict=True
    ):
        assert left.start_endpoint == right.start_endpoint
        assert left.ready == right.ready
        assert left.readiness_blockers == right.readiness_blockers
        assert left.rule_trace == right.rule_trace
        assert left.assembly is not None and right.assembly is not None
        assert left.assembly.ready == right.assembly.ready
        assert left.assembly.failure_codes == right.assembly.failure_codes
        assert left.assembly.collision_ok == right.assembly.collision_ok
        assert left.assembly.min_nonbonded_distance == pytest.approx(
            right.assembly.min_nonbonded_distance
        )
        assert len(left.assembly.approach_configurations) == len(
            right.assembly.approach_configurations
        )
        for lc, rc in zip(
            left.assembly.approach_configurations,
            right.assembly.approach_configurations,
            strict=True,
        ):
            assert lc.target_distance == rc.target_distance
            assert lc.shift == pytest.approx(rc.shift)
            assert lc.collision_ok == rc.collision_ok
            assert lc.moved_maps == rc.moved_maps


def test_rp_swap_equivalence_modulo_direction_labels() -> None:
    """Template R/P swap yields the same candidates modulo direction labels."""
    original = _bimolecular_inputs(
        fb_counts=FBCounts(formed=2, broken=1, order_changed=0),
        r_profile=EndpointProfile("R", driver_bonded_at_start=True),
        p_profile=EndpointProfile("P", configuration_ok=True),
        driver_sets=(
            DriverSetSpec(
                "ds-000", STRATEGY_LOCAL_CONNECTIVITY, (DriverBondSpec((3, 1)),)
            ),
        ),
    )
    assert original.r_side is not None and original.p_side is not None
    swapped = DirectionAssemblyInputs(
        reaction_id=original.reaction_id,
        fb_counts=FBCounts(
            formed=original.fb_counts.broken,
            broken=original.fb_counts.formed,
            order_changed=original.fb_counts.order_changed,
        ),
        r_profile=EndpointProfile(
            "R",
            electronic_review_complete=original.p_profile.electronic_review_complete,
            geometry_review_complete=original.p_profile.geometry_review_complete,
            has_usable_target_definition=original.p_profile.has_usable_target_definition,
            driver_bonded_at_start=original.p_profile.driver_bonded_at_start,
            reduces_fragment_dof=original.p_profile.reduces_fragment_dof,
            configuration_ok=original.p_profile.configuration_ok,
            orientation_ok=original.p_profile.orientation_ok,
            undriven_event_coverage_ok=original.p_profile.undriven_event_coverage_ok,
        ),
        p_profile=EndpointProfile(
            "P",
            electronic_review_complete=original.r_profile.electronic_review_complete,
            geometry_review_complete=original.r_profile.geometry_review_complete,
            has_usable_target_definition=original.r_profile.has_usable_target_definition,
            driver_bonded_at_start=original.r_profile.driver_bonded_at_start,
            reduces_fragment_dof=original.r_profile.reduces_fragment_dof,
            configuration_ok=original.r_profile.configuration_ok,
            orientation_ok=original.r_profile.orientation_ok,
            undriven_event_coverage_ok=original.r_profile.undriven_event_coverage_ok,
        ),
        driver_sets=original.driver_sets,
        r_side=SideMaterial("R", original.p_side.components),
        p_side=SideMaterial("P", original.r_side.components),
        bonded_pairs=original.bonded_pairs,
        elements=original.elements,
    )
    res_original = resolve_direction_assembly(original)
    res_swapped = resolve_direction_assembly(swapped)

    assert res_original.comparison.decided_layer == res_swapped.comparison.decided_layer
    assert res_original.comparison.preferred == "R"
    assert res_swapped.comparison.preferred == "P"
    assert len(res_original.direction_candidates) == len(res_swapped.direction_candidates) == 1

    left = res_original.direction_candidates[0]
    right = res_swapped.direction_candidates[0]
    assert left.start_endpoint == "R" and right.start_endpoint == "P"
    assert left.direction == "R_to_P" and right.direction == "P_to_R"
    assert left.rule_trace == right.rule_trace
    assert left.ready == right.ready
    assert left.readiness_blockers == right.readiness_blockers
    assert _candidate_codes(left) == _candidate_codes(right)
    # winner assembly geometry is the same materials under a new label
    assert left.assembly is not None and right.assembly is not None
    assert left.assembly.ready == right.assembly.ready
    assert (
        len(left.assembly.approach_configurations)
        == len(right.assembly.approach_configurations)
    )
    assert left.assembly.min_nonbonded_distance == pytest.approx(
        right.assembly.min_nonbonded_distance
    )
    stripped_left = _strip_labels(res_original.to_doc())
    stripped_right = _strip_labels(res_swapped.to_doc())
    assert stripped_left["direction_candidates"] == stripped_right["direction_candidates"]


# ---------------------------------------------------------------------------
# Kabsch correctness vs hand case.
# ---------------------------------------------------------------------------
def test_kabsch_place_single_hand_case() -> None:
    """Known 90° rotation + translation is recovered with RMSD ≈ 0."""
    onto = ComponentGeometry(
        "A",
        {1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0), 3: (0.0, 1.0, 0.0)},
    )
    # moving = R_z(90°) @ onto + (5, -2, 1)
    moving = ComponentGeometry(
        "B",
        {1: (5.0, -2.0, 1.0), 2: (5.0, -1.0, 1.0), 3: (4.0, -2.0, 1.0)},
    )
    placed, placement = place_single(moving, onto, anchors=[1, 2, 3])
    assert placement.basis == "anchor_maps"
    assert placement.rmsd_shared == pytest.approx(0.0, abs=1e-9)
    for map_, coord in onto.coords.items():
        assert placed[map_][0] == pytest.approx(coord[0], abs=1e-9)
        assert placed[map_][1] == pytest.approx(coord[1], abs=1e-9)
        assert placed[map_][2] == pytest.approx(coord[2], abs=1e-9)
    # right-handed: Kabsch result keeps det(rotation)=+1 via reflection correction
    from pes2ts_core.g2.endpoints import kabsch_transform

    moving_arr = np.array([moving.coords[m] for m in (1, 2, 3)], dtype=float)
    onto_arr = np.array([onto.coords[m] for m in (1, 2, 3)], dtype=float)
    result = kabsch_transform(moving_arr, onto_arr)
    assert float(np.linalg.det(result.rotation)) == pytest.approx(1.0, abs=1e-9)
    assert result.rmsd == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Assembly: frame mismatch, spectators, collision, approach configs.
# ---------------------------------------------------------------------------
def _single_broken_component_side() -> SideMaterial:
    """One 'component' whose bonded pair sits 6 A apart (independent frames)."""
    return SideMaterial(
        "R",
        (
            ComponentMaterial(
                "A",
                {1: (0.0, 0.0, 0.0), 2: (6.0, 0.0, 0.0)},
                {1: "C", 2: "H"},
                True,
            ),
        ),
    )


def test_frame_mismatch_detected_never_silently_accepted() -> None:
    side = _single_broken_component_side()
    assembly = assemble_start_side(
        side,
        assembly_id="asm-R-000",
        direction="R_to_P",
        bonded_pairs=frozenset({(1, 2)}),
        elements={1: "C", 2: "H"},
    )
    assert assembly.ready is False
    assert CODE_ASSEMBLY_FRAME_MISMATCH in assembly.failure_codes
    mismatches = [
        m for m in assembly.frame_mismatches if m.code == CODE_ASSEMBLY_FRAME_MISMATCH
    ]
    assert mismatches
    assert any("independent physical frames" in m.detail for m in mismatches)
    assert all(not m.spectator_preserved for m in mismatches)


def test_order_changed_pair_far_apart_also_flagged() -> None:
    """Union bonded pairs (order changes bonded on both sides) are checked."""
    assembly = assemble_start_side(
        _single_broken_component_side(),
        assembly_id="asm-R-000",
        direction="R_to_P",
        bonded_pairs=frozenset({(1, 2)}),  # order_changed ⇒ bonded in R and P
        elements={1: "C", 2: "H"},
    )
    assert assembly.ready is False
    assert any(
        m.code == CODE_ASSEMBLY_FRAME_MISMATCH for m in assembly.frame_mismatches
    )


def test_spectator_preserved_and_flagged() -> None:
    """Disconnected spectator keeps stored placement, flagged, not blocking."""
    side = SideMaterial(
        "R",
        (
            ComponentMaterial(
                "A", {1: (0.0, 0.0, 0.0), 2: (1.5, 0.0, 0.0)},
                {1: "C", 2: "H"}, True,
            ),
            ComponentMaterial(
                "S", {9: (20.0, 0.0, 0.0), 10: (21.5, 0.0, 0.0)},
                {9: "C", 10: "H"}, False,
            ),
        ),
    )
    assembly = assemble_start_side(
        side,
        assembly_id="asm-R-000",
        direction="R_to_P",
        bonded_pairs=frozenset({(1, 2), (9, 10)}),
        elements={1: "C", 2: "H", 9: "C", 10: "H"},
    )
    assert assembly.ready is True
    assert assembly.failure_codes == (CODE_ASSEMBLY_READY,)
    assert assembly.spectator_tags == ("R:S",)
    spectator_mismatches = [
        m for m in assembly.frame_mismatches if m.spectator_preserved
    ]
    assert spectator_mismatches
    assert spectator_mismatches[0].component_tag == "R:S"
    # stored placement preserved exactly
    assert assembly.coordinates[9] == (20.0, 0.0, 0.0)
    assert assembly.coordinates[10] == (21.5, 0.0, 0.0)


def test_spectator_overlap_collision_blocks_ready() -> None:
    """A spectator overlapping the anchor frame collides → not ready."""
    side = SideMaterial(
        "R",
        (
            ComponentMaterial(
                "A", {1: (0.0, 0.0, 0.0), 2: (1.5, 0.0, 0.0)},
                {1: "C", 2: "H"}, True,
            ),
            ComponentMaterial(
                "S", {9: (0.4, 0.0, 0.0), 10: (1.9, 0.0, 0.0)},
                {9: "C", 10: "H"}, False,
            ),
        ),
    )
    assembly = assemble_start_side(
        side,
        assembly_id="asm-R-000",
        direction="R_to_P",
        bonded_pairs=frozenset({(1, 2), (9, 10)}),
        elements={1: "C", 2: "H", 9: "C", 10: "H"},
    )
    assert assembly.ready is False
    assert CODE_ASSEMBLY_COLLISION in assembly.failure_codes
    assert assembly.collision_ok is False


def test_intra_component_collision_not_ready() -> None:
    side = SideMaterial(
        "R",
        (
            ComponentMaterial(
                "A",
                {1: (0.0, 0.0, 0.0), 2: (1.1, 0.0, 0.0), 3: (1.5, 0.0, 0.0)},
                {1: "C", 2: "H", 3: "C"},
                True,
            ),
        ),
    )
    assembly = assemble_start_side(
        side,
        assembly_id="asm-R-000",
        direction="R_to_P",
        bonded_pairs=frozenset({(1, 2)}),  # (2,3) nonbonded at 0.4 A
        elements={1: "C", 2: "H", 3: "C"},
    )
    assert assembly.ready is False
    assert CODE_ASSEMBLY_COLLISION in assembly.failure_codes
    assert assembly.n_severe_contacts >= 1


def test_approach_configurations_deterministic_enumeration() -> None:
    """Cross-component drivers get finite, ordered approach configurations."""
    inputs = _bimolecular_inputs(
        driver_sets=(
            DriverSetSpec(
                "ds-000",
                STRATEGY_LOCAL_CONNECTIVITY,
                (DriverBondSpec((3, 1), driver_id="d1"),),
            ),
        ),
    )
    first = resolve_direction_assembly(inputs)
    second = resolve_direction_assembly(inputs)
    assert first.to_json() == second.to_json()

    r_candidate = next(
        c for c in first.direction_candidates if c.start_endpoint == "R"
    )
    assert r_candidate.assembly is not None
    configs = r_candidate.assembly.approach_configurations
    assert len(configs) == 3  # default targets 2.5 / 3.0 / 3.5
    assert [c.target_distance for c in configs] == [2.5, 3.0, 3.5]
    assert [c.configuration_id for c in configs] == [
        "asm-R-000-cfg-000",
        "asm-R-000-cfg-001",
        "asm-R-000-cfg-002",
    ]
    assert all(isinstance(c, ApproachConfiguration) for c in configs)
    assert all(c.collision_ok for c in configs)
    # P side is single-component: no cross-component approach configs
    p_candidate = next(
        c for c in first.direction_candidates if c.start_endpoint == "P"
    )
    assert p_candidate.assembly is not None
    assert p_candidate.assembly.approach_configurations == ()


def test_assembly_failure_marks_candidate_not_ready() -> None:
    """A broken start side AND a broken target side both block readiness."""
    valid_p_side = SideMaterial(
        "P",
        (
            ComponentMaterial(
                "AB",
                {1: (0.0, 0.0, 0.0), 2: (1.5, 0.0, 0.0)},
                {1: "C", 2: "H"},
                True,
            ),
        ),
    )
    inputs = _bimolecular_inputs(
        r_side=_single_broken_component_side(),
        p_side=valid_p_side,
        bonded_pairs=frozenset({(1, 2)}),
        elements={1: "C", 2: "H"},
        driver_sets=(),
    )
    result = resolve_direction_assembly(inputs)
    for candidate in result.direction_candidates:
        assert candidate.ready is False
        assert CODE_ASSEMBLY_FRAME_MISMATCH in candidate.readiness_blockers
    r_candidate = next(c for c in result.direction_candidates if c.start_endpoint == "R")
    p_candidate = next(c for c in result.direction_candidates if c.start_endpoint == "P")
    assert r_candidate.assembly is not None and r_candidate.assembly.ready is False
    # P-start's own assembly passes, but its target (broken R) blocks it
    assert p_candidate.assembly is not None and p_candidate.assembly.ready is True
    assert p_candidate.target_assembly is not None
    assert p_candidate.target_assembly.ready is False


def test_no_side_material_assembly_not_evaluated_trace() -> None:
    inputs = _bimolecular_inputs(r_side=None, p_side=None)
    result = resolve_direction_assembly(inputs)
    for candidate in result.direction_candidates:
        assert candidate.assembly is None
        assert candidate.ready is True  # layer-1 only; release gates are todo 17
        codes = _candidate_codes(candidate)
        assert "ASSEMBLY_NOT_EVALUATED" in codes


def test_target_definition_block_candidate_not_ready() -> None:
    inputs = _bimolecular_inputs(
        p_profile=EndpointProfile("P", has_usable_target_definition=False),
    )
    result = resolve_direction_assembly(inputs)
    assert result.comparison.preferred is None
    assert len(result.direction_candidates) == 2  # both emitted, both blocked
    for candidate in result.direction_candidates:
        assert candidate.ready is False
        assert CODE_NO_USABLE_ENDPOINT in candidate.readiness_blockers


# ---------------------------------------------------------------------------
# Contracts: direction consistency, determinism, purity.
# ---------------------------------------------------------------------------
def test_direction_candidate_rejects_inconsistent_direction() -> None:
    with pytest.raises(ValueError, match="DIRECTION_MISMATCH"):
        DirectionCandidate(
            start_endpoint="R",
            direction="P_to_R",
            assembly_id=None,
            anchor_reason=(),
            route_strategy_id=None,
            driver_set_id=None,
            rule_trace=(),
            bidirectional=False,
            assembly=None,
            target_assembly=None,
            ready=True,
            readiness_blockers=(),
        )


def test_to_doc_deterministic_and_truth_pure() -> None:
    inputs = _bimolecular_inputs()
    first = resolve_direction_assembly(inputs)
    second = resolve_direction_assembly(inputs)
    assert first.to_json() == second.to_json()
    doc = first.to_doc()
    assert doc["splice_policy"] == SPLICE_POLICY
    assert doc["selection_mechanism"] == SELECTION_MECHANISM

    def walk(value: Any, hits: list[str]) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).lower() in FORBIDDEN:
                    hits.append(str(key))
                walk(item, hits)
        elif isinstance(value, list):
            for item in value:
                walk(item, hits)

    hits: list[str] = []
    walk(doc, hits)
    assert hits == []
    for candidate in first.direction_candidates:
        record = candidate.to_record()
        assert record["equivalence_claimed"] is False
        assert record["splice_policy"] == SPLICE_POLICY
        assert candidate.anchor_reason
        for reason in candidate.anchor_reason:
            assert reason.code
            assert reason.layer


def test_fb_counts_from_edit_graph() -> None:
    graph = _FakeEditGraph(
        edit_counts={"formed": 2, "broken": 1, "order_changed": 3}
    )
    counts = fb_counts_from_edit_graph(graph)
    assert counts == FBCounts(formed=2, broken=1, order_changed=3)
    assert counts.hi == 2 and counts.lo == 1
    assert not counts.is_tie


def test_anchor_reason_typed_complete_for_every_candidate() -> None:
    inputs = _bimolecular_inputs()
    result = resolve_direction_assembly(inputs)
    for candidate in result.direction_candidates:
        assert candidate.anchor_reason
        layers = {reason.layer for reason in candidate.anchor_reason}
        assert "prior" in layers
        assert "hybrid_anchor" in layers
        assert "assembly" in layers or candidate.assembly is None


def test_policy_from_config_yaml_numeric_string_quirk() -> None:
    config = {
        "scan_strategy": {
            "direction_assembly": {
                "frame_mismatch_tolerance": "2.0",
                "approach_target_distances": [2.0, 3.0],
            }
        },
        "g2": {"validity": {"collision_min_distance": "0.9"}},
    }
    policy = policy_from_config(config)
    assert policy.frame_mismatch_tolerance == 2.0
    assert policy.approach_target_distances == (2.0, 3.0)
    assert policy.collision_min_distance == 0.9
    default_policy = policy_from_config(None)
    assert default_policy.collision_min_distance == pytest.approx(0.8)
    with pytest.raises(ValueError, match="POLICY_INVALID"):
        policy_from_config({"scan_strategy": {"direction_assembly": {
            "approach_target_distances": [],
        }}})


def test_bidirectional_order_is_deterministic_r_then_p() -> None:
    inputs = _bimolecular_inputs(fb_counts=FBCounts(1, 1, 0))
    result = resolve_direction_assembly(inputs)
    starts = [c.start_endpoint for c in result.direction_candidates]
    assert starts == ["R", "P"]
    directions = [c.direction for c in result.direction_candidates]
    assert directions == ["R_to_P", "P_to_R"]
