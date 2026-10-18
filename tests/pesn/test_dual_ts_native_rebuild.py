"""V4.2 native-reconnect rebuild tests — SYNTHETIC fixtures only.

Covers: inserted-state removal + cross-insertion reconnection, long-step
supervision preservation (the reconnected native step is the FULL original
step, not the halved sub-step), arc-length recompute identity (native
remaining arc length == original subdivided value, midpoints collinear),
unchanged-step label preservation, and the 395↔399 multi-round discrepancy
explanation.
"""
from __future__ import annotations

import numpy as np
import pytest

from pes2ts_core.pesn.offline.data.direct_preprocess import masses_for
from pes2ts_core.pesn.offline.data.dual_ts_preprocess import (
    ARC_LENGTH_METRIC_V1_DIAG_MASSES,
    build_dual_ts_reaction,
    endpoint_graph_from_bond_list,
)
from pes2ts_core.pesn.offline.data.dual_ts_native_rebuild import (
    NativeRebuildError,
    inserted_state_ids,
    native_supervision_mask,
    rebuild_native_reaction,
)

Z = (8, 1, 6, 1)  # O, H, C, H — all in the v1 mass table
R_BONDS = [(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)]
P_BONDS = [(1, 2, 1.0), (2, 4, 2.0), (3, 4, 1.0)]


def _graphs():
    z = np.asarray(Z, dtype=int)
    return (endpoint_graph_from_bond_list(z, R_BONDS),
            endpoint_graph_from_bond_list(z, P_BONDS))


def _path_with_exceedance() -> np.ndarray:
    """5-frame ordered path (ts_position=2) with a clear >0.1 Å step on the
    R branch (frame 0 -> 1) so midpoint insertion triggers."""
    path = np.array([
        [[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [1.8, 0.0, 0.3], [2.7, 0.1, 0.0]],
        [[0.1, 0.0, 0.0], [1.0, 0.1, 0.1], [2.05, 0.1, 0.2], [2.6, 0.0, 0.1]],
        [[0.2, 0.1, 0.0], [1.3, 0.2, 0.0], [1.5, 0.0, 0.1], [2.4, 0.1, 0.2]],
        [[0.1, 0.0, 0.2], [1.6, 0.1, 0.0], [1.4, 0.1, 0.2], [2.5, 0.0, 0.0]],
        [[0.0, 0.1, 0.1], [1.9, 0.0, 0.1], [1.2, 0.2, 0.0], [2.6, 0.1, 0.1]],
    ], dtype=float)
    # inject a large per-atom step on atom 2 of the R branch (frame 0 -> 1).
    path[1, 2, 0] += 0.5
    return path


def _build_inserted_teacher():
    graph_r, graph_p = _graphs()
    z = np.asarray(Z, dtype=int)
    outcome = build_dual_ts_reaction(
        reaction_id="RXN_SYNTH0001",
        reaction_group_id="GRP_SYNTH0001",
        partition="fit",
        atomic_numbers=Z,
        ordered_path=_path_with_exceedance(),
        authority_ts_position=2,
        energies=np.array([0.0, 0.2, 1.0, 0.3, 0.05]),
        graph_r=graph_r,
        graph_p=graph_p,
        method_id="synthetic_method",
        source="synthetic fixture",
        insert_on_exceedance=True,
        ell_masses=masses_for(z),
        arc_length_metric_version=ARC_LENGTH_METRIC_V1_DIAG_MASSES,
    )
    assert outcome.status == "ok"
    return outcome


def test_inserted_state_ids_from_gaps():
    outcome = _build_inserted_teacher()
    ids = inserted_state_ids(outcome.gaps)
    # every inserted state id must be present in the teacher states
    teacher_ids = {s.state_id for s in outcome.teacher.states}
    assert ids <= teacher_ids
    # and the registry must actually contain insertions (our injected step)
    assert len(ids) >= 1


def test_removal_and_reconnection():
    outcome = _build_inserted_teacher()
    teacher = outcome.teacher
    rebuilt = rebuild_native_reaction(teacher, outcome.gaps)
    audit = rebuilt["audit"]

    # every native state is NOT inserted, and inserted states are gone.
    inserted = inserted_state_ids(outcome.gaps)
    native_ids = {s.state_id for s in rebuilt["native_states"]}
    assert native_ids.isdisjoint(inserted)
    # the teacher endpoints (TS frames) are preserved exactly.
    endpoints = {s.state_id for s in teacher.states
                 if s.boundary_type == "teacher_endpoint"}
    assert endpoints <= native_ids
    assert audit["n_teacher_endpoints"] == 2
    # reconnected count equals the exceedance count.
    assert audit["n_reconnected_steps"] == audit["a_max_exceedances_count"]
    assert audit["n_reconnected_steps"] >= 1


def test_long_step_supervision_not_lost():
    outcome = _build_inserted_teacher()
    rebuilt = rebuild_native_reaction(outcome.teacher, outcome.gaps)
    # a reconnected native step must exceed a_max (the full original step).
    by_id = {s.state_id: s for s in rebuilt["native_states"]}
    labels = {lab.state_id: lab for lab in rebuilt["native_labels"]}
    saw_long = False
    for lab in labels.values():
        if lab.action_delta is None:
            continue
        nxt = by_id[lab.state_id].next_state_id
        assert nxt is not None
        a = np.asarray(lab.action_delta)
        if float(np.linalg.norm(a, axis=1).max()) > 0.1:
            saw_long = True
    assert saw_long


def test_arc_length_recompute_identity():
    outcome = _build_inserted_teacher()
    rebuilt = rebuild_native_reaction(outcome.teacher, outcome.gaps)
    original_labels = {lab.state_id: lab for lab in outcome.teacher.labels}
    for lab in rebuilt["native_labels"]:
        orig = original_labels[lab.state_id]
        assert abs(lab.remaining_arc_length
                   - orig.remaining_arc_length) <= 1e-9


def test_unchanged_step_label_preserved():
    outcome = _build_inserted_teacher()
    rebuilt = rebuild_native_reaction(outcome.teacher, outcome.gaps)
    # unchanged steps keep the exact original action (float64 identity).
    assert rebuilt["audit"]["unchanged_step_max_abs_err"] <= 1e-10


def test_label_identity_x_next_minus_x():
    outcome = _build_inserted_teacher()
    rebuilt = rebuild_native_reaction(outcome.teacher, outcome.gaps)
    by_id = {s.state_id: s for s in rebuilt["native_states"]}
    for lab in rebuilt["native_labels"]:
        if lab.action_delta is None:
            continue
        nxt = by_id[lab.state_id].next_state_id
        cur = by_id[lab.state_id]
        direct = np.asarray(by_id[nxt].coordinates) - \
            np.asarray(cur.coordinates)
        assert np.max(np.abs(np.asarray(lab.action_delta) - direct)) <= 1e-10


def test_native_supervision_mask_rule():
    n = 3
    # terminal -> all-True
    assert native_supervision_mask(None, True, n).all()
    # non-terminal with action -> per-atom near-zero mask
    a = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.005, 0.0, 0.0]])
    mask = native_supervision_mask(a, False, n)
    assert mask.tolist() == [True, False, True]
    with pytest.raises(NativeRebuildError):
        native_supervision_mask(None, False, n)


def test_multi_round_discrepancy_explained():
    outcome = _build_inserted_teacher()
    rebuilt = rebuild_native_reaction(outcome.teacher, outcome.gaps)
    disc = rebuilt["audit"]["discrepancy"]
    # explained always True; difference is accounted by multi-round steps.
    assert disc["explained"] is True
    assert disc["difference"] == disc["extra_from_multi_round"]
