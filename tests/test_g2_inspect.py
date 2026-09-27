"""Unit tests for g2.inspect: per-frame metrics and validity verdicts.

Synthetic fixtures only: Frame objects are constructed directly (the parser
already guarantees decimal energies, so NaN/inf energies can only be exercised
through direct construction).
"""

from __future__ import annotations

import dataclasses
import json
import math

import pytest

from pes2ts_core.g0.rejections import RejectionCode
from pes2ts_core.g2.inspect import Verdict, evaluate_validity, frame_metrics
from pes2ts_core.g2.xtb_output import Frame

MAP_ORDER = (1, 2, 3)
ELEMENTS = {1: "C", 2: "O", 3: "H"}
R_PAIRS = frozenset({(2, 3)})
P_PAIRS = frozenset({(1, 2), (2, 3)})
EVENTS_FORMED = {
    "formed": [{"atoms": [1, 2], "order_r": None, "order_p": 1.0}],
    "broken": [],
    "order_changed": [],
    "hydrogen_migration": [],
}
# C travels x=5.0 -> ~1.4 in 8 frames; O fixed at origin; H bonded to O.
LINEAR_XS = [5.0 - 3.6 * i / 7 for i in range(8)]
# Exact-4.0 single-atom step at frame 3; formed 5.0 -> 1.4 (drift-free).
JUMP_XS = [5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.4]
# Constant bonded C-O distance (topology drift for both directions).
CONSTANT_XS = [0.0] * 8


def _frames(xs, *, h=(0.0, 0.96, 0.0), energies=None):
    return tuple(
        Frame(
            ("C", "O", "H"),
            ((x, 0.0, 0.0), (0.0, 0.0, 0.0), tuple(h)),
            float(i) if energies is None else energies[i],
        )
        for i, x in enumerate(xs)
    )


def _keyed(frame, shift=(0.0, 0.0, 0.0)):
    return {
        m: (
            frame.coordinates[i][0] + shift[0],
            frame.coordinates[i][1] + shift[1],
            frame.coordinates[i][2] + shift[2],
        )
        for i, m in enumerate(MAP_ORDER)
    }


def _inspect(
    frames,
    *,
    r_pairs=R_PAIRS,
    p_pairs=P_PAIRS,
    events=EVENTS_FORMED,
    reactant_shift=(0.0, 0.0, 0.0),
    product_shift=(0.0, 0.0, 0.0),
    elements=ELEMENTS,
    xtb_failure=None,
    config=None,
):
    return evaluate_validity(
        frames,
        reaction_id="RXN_TEST",
        reactant_coords=_keyed(frames[0], reactant_shift),
        product_coords=_keyed(frames[-1], product_shift),
        elements=elements,
        r_pairs=r_pairs,
        p_pairs=p_pairs,
        events=events,
        xtb_failure=xtb_failure,
        config=config,
    )


def _metrics(frames, **kwargs):
    return frame_metrics(
        frames,
        reaction_id="RXN_TEST",
        reactant_coords=kwargs.get("reactant_coords", _keyed(frames[0])),
        product_coords=kwargs.get("product_coords", _keyed(frames[-1])),
        elements=kwargs.get("elements", ELEMENTS),
        r_pairs=kwargs.get("r_pairs", R_PAIRS),
        p_pairs=kwargs.get("p_pairs", P_PAIRS),
        events=kwargs.get("events", EVENTS_FORMED),
        config=kwargs.get("config"),
    )


def test_evaluate_validity_returns_valid_when_every_predicate_holds():
    verdict = _inspect(_frames(LINEAR_XS))
    assert verdict.status == "valid"
    assert verdict.failure_code is None


def test_evaluate_validity_reports_xtb_failure_when_xtb_failed():
    verdict = _inspect(_frames(CONSTANT_XS), xtb_failure="xtb exited 3 (timeout)")
    assert verdict.status == "failed"
    assert verdict.failure_code == RejectionCode.G2_XTB_FAILED
    assert verdict.detail == "xtb exited 3 (timeout)"


def test_evaluate_validity_reports_endpoint_not_reached_when_only_first_vs_r_exceeds():
    verdict = _inspect(_frames(LINEAR_XS), reactant_shift=(0.5 + 1e-9, 0.0, 0.0))
    assert verdict.failure_code == RejectionCode.G2_ENDPOINT_NOT_REACHED
    assert "first_vs_R" in (verdict.detail or "")


def test_evaluate_validity_passes_when_endpoint_rmsd_exactly_at_threshold():
    # every atom displaced by exactly (0.5, 0, 0): rmsd == 0.5 == threshold.
    verdict = _inspect(_frames(LINEAR_XS), reactant_shift=(0.5, 0.0, 0.0))
    assert verdict.status == "valid"


def test_evaluate_validity_reports_topology_drift_when_formed_pair_already_bonded():
    verdict = _inspect(_frames(CONSTANT_XS))
    assert verdict.failure_code == RejectionCode.G2_TOPOLOGY_DRIFT
    assert "formed:1-2" in (verdict.detail or "")


def test_evaluate_validity_reports_topology_drift_when_broken_pair_still_bonded():
    r_pairs = frozenset({(1, 2), (2, 3)})
    p_pairs = frozenset({(2, 3)})
    events = {
        "formed": [],
        "broken": [{"atoms": [1, 2], "order_r": 1.0, "order_p": None}],
        "order_changed": [],
        "hydrogen_migration": [],
    }
    verdict = _inspect(_frames(CONSTANT_XS), r_pairs=r_pairs, p_pairs=p_pairs, events=events)
    assert verdict.failure_code == RejectionCode.G2_TOPOLOGY_DRIFT
    assert "broken:1-2" in (verdict.detail or "")


def test_evaluate_validity_treats_null_h_migration_partner_as_vacuous():
    events = {
        "formed": EVENTS_FORMED["formed"],
        "broken": [],
        "order_changed": [],
        "hydrogen_migration": [{"h": 3, "from": 2, "to": None}],
    }
    verdict = _inspect(_frames(LINEAR_XS), events=events)
    assert verdict.status == "valid"


def test_evaluate_validity_reports_energy_incomplete_when_energy_is_none():
    energies = [float(i) for i in range(8)]
    energies[3] = None
    verdict = _inspect(_frames(LINEAR_XS, energies=energies))
    assert verdict.failure_code == RejectionCode.G2_ENERGY_INCOMPLETE


def test_evaluate_validity_reports_energy_incomplete_when_energy_is_nan():
    energies = [float(i) for i in range(8)]
    energies[5] = float("nan")
    verdict = _inspect(_frames(LINEAR_XS, energies=energies))
    assert verdict.failure_code == RejectionCode.G2_ENERGY_INCOMPLETE


def test_evaluate_validity_reports_energy_incomplete_when_energy_is_infinite():
    energies = [float(i) for i in range(8)]
    energies[5] = float("inf")
    verdict = _inspect(_frames(LINEAR_XS, energies=energies))
    assert verdict.failure_code == RejectionCode.G2_ENERGY_INCOMPLETE


def test_evaluate_validity_reports_path_discontinuous_when_step_exceeds_max():
    verdict = _inspect(_frames([5.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.4]))
    assert verdict.failure_code == RejectionCode.G2_PATH_DISCONTINUOUS


def test_evaluate_validity_passes_when_step_exactly_at_max():
    verdict = _inspect(_frames(JUMP_XS))
    assert verdict.status == "valid"


def test_evaluate_validity_reports_path_discontinuous_when_frame_count_below_min():
    short = _frames(JUMP_XS[:7])
    verdict = _inspect(short)
    assert verdict.failure_code == RejectionCode.G2_PATH_DISCONTINUOUS
    assert "n_frames=7" in (verdict.detail or "")


def test_evaluate_validity_passes_when_frame_count_exactly_min():
    assert _inspect(_frames(JUMP_XS)).status == "valid"


def test_evaluate_validity_reports_collision_when_nonbonded_pair_below_min():
    verdict = _inspect(_frames(JUMP_XS, h=(1.0, 0.79, 0.0)))
    assert verdict.failure_code == RejectionCode.G2_COLLISION


def test_evaluate_validity_passes_when_collision_exactly_at_min():
    # C sits at (1, 0, 0) for frames 3-6 -> d(C, H) == 0.8 exactly.
    verdict = _inspect(_frames(JUMP_XS, h=(1.0, 0.8, 0.0)))
    assert verdict.status == "valid"


def test_evaluate_validity_xtb_failure_beats_all_other_violations():
    verdict = _inspect(_frames(CONSTANT_XS), xtb_failure="boom")
    assert verdict.failure_code == RejectionCode.G2_XTB_FAILED


def test_evaluate_validity_endpoint_beats_topology_drift():
    verdict = _inspect(_frames(CONSTANT_XS), reactant_shift=(0.6, 0.0, 0.0))
    assert verdict.failure_code == RejectionCode.G2_ENDPOINT_NOT_REACHED


def test_evaluate_validity_topology_beats_energy_incomplete():
    energies = [float(i) for i in range(8)]
    energies[3] = None
    verdict = _inspect(_frames(CONSTANT_XS, energies=energies))
    assert verdict.failure_code == RejectionCode.G2_TOPOLOGY_DRIFT


def test_evaluate_validity_energy_beats_path_discontinuous():
    energies = [float(i) for i in range(7)]
    energies[3] = None
    verdict = _inspect(_frames(JUMP_XS[:7], energies=energies))
    assert verdict.failure_code == RejectionCode.G2_ENERGY_INCOMPLETE


def test_evaluate_validity_energy_beats_collision():
    energies = [float(i) for i in range(8)]
    energies[3] = None
    verdict = _inspect(_frames(JUMP_XS, h=(1.0, 0.79, 0.0), energies=energies))
    assert verdict.failure_code == RejectionCode.G2_ENERGY_INCOMPLETE


def test_evaluate_validity_honors_config_overrides():
    config = {"g2": {"validity": {"endpoint_rmsd_max": 0.1}}}
    verdict = _inspect(_frames(LINEAR_XS), reactant_shift=(0.5, 0.0, 0.0), config=config)
    assert verdict.failure_code == RejectionCode.G2_ENDPOINT_NOT_REACHED


def test_verdict_is_frozen():
    verdict = _inspect(_frames(LINEAR_XS))
    with pytest.raises(dataclasses.FrozenInstanceError):
        verdict.status = "failed"  # pyright: ignore[reportAttributeAccessIssue]


def test_frame_metrics_column_contract():
    metrics = _metrics(_frames(LINEAR_XS))
    required = {
        "reaction_id",
        "frame_index",
        "energy_rel_kcal",
        "rmsd_to_start",
        "rmsd_to_end",
        "step_max",
        "step_rmsd",
        "min_nonbonded_distance",
        "event_distances",
    }
    assert required <= set(metrics[0])
    assert len(metrics) == 8
    assert [m["frame_index"] for m in metrics] == list(range(8))
    assert all(m["reaction_id"] == "RXN_TEST" for m in metrics)
    assert [m["energy_rel_kcal"] for m in metrics] == [float(i) for i in range(8)]


def test_frame_metrics_rmsd_and_step_values():
    metrics = _metrics(_frames(LINEAR_XS))
    assert metrics[0]["rmsd_to_start"] == 0.0
    assert metrics[-1]["rmsd_to_end"] == 0.0
    assert metrics[0]["step_max"] == 0.0
    assert metrics[0]["step_rmsd"] == 0.0
    # only C moves: rmsd over 3 atoms = displacement / sqrt(3).
    step = LINEAR_XS[0] - LINEAR_XS[1]
    assert metrics[1]["step_max"] == pytest.approx(step)
    assert metrics[1]["step_rmsd"] == pytest.approx(math.sqrt(step**2 / 3))
    assert metrics[0]["rmsd_to_end"] == pytest.approx(math.sqrt((5.0 - LINEAR_XS[-1]) ** 2 / 3))


def test_frame_metrics_min_nonbonded_distance():
    metrics = _metrics(_frames(LINEAR_XS))
    for record, x in zip(metrics, LINEAR_XS):
        # the only neither-R-nor-P bonded pair is (C, H).
        assert record["min_nonbonded_distance"] == pytest.approx(math.sqrt(x**2 + 0.96**2))


def test_frame_metrics_event_distances_key_scheme():
    frames = _frames(LINEAR_XS)
    events = {
        "formed": EVENTS_FORMED["formed"],
        "broken": [{"atoms": [2, 1], "order_r": None, "order_p": None}],
        "order_changed": [{"atoms": [1, 2], "order_r": 1.0, "order_p": 2.0}],
        "hydrogen_migration": [{"h": 3, "from": 2, "to": 1}],
    }
    metrics = _metrics(frames, events=events)
    assert list(metrics[0]["event_distances"]) == [
        "formed:1-2",
        "broken:1-2",
        "order_changed:1-2",
        "h_migration:3:from=2",
        "h_migration:3:to=1",
    ]
    assert metrics[0]["event_distances"]["formed:1-2"] == pytest.approx(5.0)
    assert metrics[0]["event_distances"]["h_migration:3:from=2"] == pytest.approx(0.96)
    assert metrics[0]["event_distances"]["h_migration:3:to=1"] == pytest.approx(
        math.sqrt(5.0**2 + 0.96**2)
    )
    # from=None leaves only the "to" half.
    events_null = {
        "formed": [],
        "broken": [],
        "order_changed": [],
        "hydrogen_migration": [{"h": 3, "from": None, "to": 1}],
    }
    null_metrics = _metrics(frames, events=events_null)
    assert list(null_metrics[0]["event_distances"]) == ["h_migration:3:to=1"]


def test_frame_metrics_is_deterministic_across_calls():
    frames = _frames(LINEAR_XS)
    first = json.dumps(_metrics(frames), sort_keys=True)
    second = json.dumps(_metrics(frames), sort_keys=True)
    assert first == second
