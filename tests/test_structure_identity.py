"""G2-AB1 WP-5: strict landing-identity tiering.

``target_fb_pattern`` is a distance mode only and is never a chemical success
marker.  A correct active-F/B distance pattern with a broken spectator bond or
a flipped specified stereocentre must be a typed rejection, and
``full_identity_verified`` must be unreachable without ACP validation
evidence.
"""
from __future__ import annotations

import numpy as np
import pytest

from pes2ts_core.generation.planning.structure_checks import (
    LANDING_IDENTITY_TIERS, endpoint_geometry_identity, landing_identity, structure_issues,
)

PLAN = {"elements": ["H", "H", "H", "H"], "atom_map_order": [1, 2, 3, 4],
        "drivers": [{"id": "d", "kind": "distance", "atoms": [0, 1], "maps": [1, 2],
                     "edit_kind": "formed", "lambda_values": [0., 1.], "values": [2., .8]}]}

# H2-like: 0-1 forming (~.8 Å), spectator H 0-2/0-3 at .95 Å (below the
# 1.07 Å H-H bonded cutoff, while every cross pair stays above it).
GOOD = np.array([[0., 0., 0.], [.8, 0., 0.], [0., .95, 0.], [0., 0., .95]])
SPECTATOR_BONDS = [[0, 2], [0, 3]]


def test_correct_active_distance_gives_target_fb_pattern_not_chemical_success():
    result = landing_identity(GOOD, PLAN)
    assert result["landing_identity_status"] == "target_fb_pattern"
    assert result["landing_identity_scope"] == "active_fb_distance_pattern"
    assert result["target_fb_pattern_is_chemical_success"] is False
    assert result["complete_chemical_identity_verified"] is False


def test_wrong_active_distance_is_unknown_never_an_auto_pass():
    wrong = GOOD.copy()
    wrong[1] = [2.6, 0., 0.]
    result = landing_identity(wrong, PLAN)
    assert result["landing_identity_status"] == "unknown"
    assert result["driver_pattern"][0]["matched"] is False


def test_full_adjacency_screen_upgrades_scope_only_with_target_bonds():
    bonds = [[0, 1]] + SPECTATOR_BONDS
    result = landing_identity(GOOD, PLAN, target_bond_indices=bonds)
    assert result["landing_identity_status"] == "adjacency_screen_only"
    assert result["landing_identity_scope"] == "full_covalent_adjacency_screen"
    assert result["complete_chemical_identity_verified"] is False
    # A spectator bond broken in the geometry fails the adjacency screen.
    broken = GOOD.copy()
    broken[2] = [0., 2.9, 0.]
    result = landing_identity(broken, PLAN, target_bond_indices=bonds)
    assert result["landing_identity_status"] == "unknown"
    assert result["adjacency_screen"]["adjacency_screen_passed"] is False
    typed = structure_issues(broken, {"common_bond_checks": [
        {"atoms": [0, 2], "maps": [1, 3], "minimum_distance": .4, "maximum_distance": 1.2}]})
    assert typed[0]["reason"] == "SPECTATOR_BOND_GEOMETRY"


def test_specified_stereocentre_flip_is_a_typed_rejection():
    plan = {"spectator_stereo_checks": [{"atoms": [0, 1, 2, 3], "maps": [1, 2, 3, 4],
                                         "reference_sign": 1., "minimum_volume": .01}]}
    flipped = GOOD.copy()
    flipped[3] = [0., 0., -.95]
    assert structure_issues(flipped, plan)[0]["reason"] == "SPECTATOR_STEREO_CHANGED"
    assert not structure_issues(GOOD, plan)


def test_full_identity_verified_requires_complete_acp_evidence():
    bonds = [[0, 1]] + SPECTATOR_BONDS
    full = {"optts": "converged", "frequency": "passed",
            "irc_forward": "matched", "irc_reverse": "matched"}
    result = landing_identity(GOOD, PLAN, target_bond_indices=bonds,
                              acp_validation_evidence=full)
    assert result["landing_identity_status"] == "full_identity_verified"
    assert result["complete_chemical_identity_verified"] is True
    # Partial evidence (no IRC) must not verify.
    partial = {k: v for k, v in full.items() if k != "irc_reverse"}
    result = landing_identity(GOOD, PLAN, target_bond_indices=bonds,
                              acp_validation_evidence=partial)
    assert result["landing_identity_status"] == "adjacency_screen_only"
    # No ACP evidence at all: unreachable by construction.
    result = landing_identity(GOOD, PLAN, target_bond_indices=bonds)
    assert result["landing_identity_status"] != "full_identity_verified"
    geometry_only = endpoint_geometry_identity(GOOD, PLAN["elements"], bonds)
    assert geometry_only["complete_chemical_identity_verified"] is False


def test_tier_enum_is_the_only_vocabulary():
    assert LANDING_IDENTITY_TIERS == ("target_fb_pattern", "adjacency_screen_only",
                                      "full_identity_verified", "unknown")


def test_continuation_frames_record_landing_identity_scope_and_status():
    from test_connectivity_continuation import example, physical_backend
    from pes2ts_core.generation.planning.continuation import run_continuation
    plan, initial = example(max_frames=4)
    result = run_continuation(plan, initial, physical_backend)
    for frame in result["frames"]:
        assert frame["landing_identity_status"] in LANDING_IDENTITY_TIERS
        assert isinstance(frame["landing_identity_scope"], str)
    # The plan's driver is "broken": at the stretched final frame the distance
    # pattern matches the target, at the compressed origin it does not.
    assert result["frames"][-1]["landing_identity_status"] == "target_fb_pattern"
    assert result["frames"][0]["landing_identity_status"] == "unknown"
