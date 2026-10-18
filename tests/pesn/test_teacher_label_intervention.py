"""v4.1 WP1 acceptance test (a): teacher-label intervention.

Fixes one observation (geometry, gradient, ΔE, charge, multiplicity,
topology) and arbitrarily varies the TEACHER-label-derived supervision
(``action_delta`` including magnitudes crossing the 0.01 Å threshold, and
``is_teacher_endpoint``).  ALL network-input fields produced by the
observation builder (``node_scalars`` and the dual-topology blocks) must be
bit-identical — teacher labels may never enter the state inputs (red line
R1 / F1).

Second half: on a built shard, mutating the stored teacher npz keys
(``near_zero_action_mask``, ``is_teacher_endpoint``) leaves the collated
model batch inputs unchanged — those keys are supervision metadata, not
network input.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.contracts.dual_ts import (  # noqa: E402
    BranchRecord,
    DualEndpointTask,
    DualTSTeacher,
    EndpointBond,
    EndpointGraph,
    EpisodeRecord,
    TeacherLabel,
    TeacherState,
)
from pes2ts_core.pesn.features.dual_ts_graph import (  # noqa: E402
    build_state_features,
    build_teacher_supervision,
    build_topology_features,
)
from pes2ts_core.pesn.features.dual_ts_spec import (  # noqa: E402
    BOND_SOURCE_COVALENT_FALLBACK,
    NEAR_ZERO_ACTION_THRESHOLD_ANGSTROM,
)
from pes2ts_core.pesn.models import require_torch  # noqa: E402
from pes2ts_core.pesn.models.dual_egnn import (  # noqa: E402
    collate_dual_states,
)

Z3 = np.asarray([6, 6, 1], dtype=int)  # C, C, H


def _endpoint(orders: list[tuple[int, int, float]]) -> EndpointGraph:
    return EndpointGraph(
        atomic_numbers=Z3.copy(),
        atom_map_ids=tuple(f"m{k + 1:04d}" for k in range(Z3.size)),
        bonds=tuple(EndpointBond(i, j, o) for i, j, o in orders),
    )


def _task() -> DualEndpointTask:
    return DualEndpointTask(
        reaction_id="RXN_T", reaction_group_id="RXN_TG",
        endpoints=(_endpoint([(0, 1, 1.0), (1, 2, 1.0)]),
                   _endpoint([(0, 1, 2.0), (0, 2, 1.0)])),
        method_id="M",
        eligibility={"bond_authority": "geometry_verified"},
    )


def _disputed_task() -> DualEndpointTask:
    """Same endpoints, but bond_authority == "disputed" → the covalent
    fallback in ``consumed_endpoint_graphs`` READS the teacher episode-start
    geometry (the one remaining teacher-taking topology builder)."""
    return DualEndpointTask(
        reaction_id="RXN_TD", reaction_group_id="RXN_TDG",
        endpoints=(_endpoint([(0, 1, 1.0), (1, 2, 1.0)]),
                   _endpoint([(0, 1, 2.0), (0, 2, 1.0)])),
        method_id="M",
        eligibility={"bond_authority": "disputed"},
    )


def _teacher_for(task: DualEndpointTask) -> DualTSTeacher:
    rid = task.reaction_id
    z = np.asarray(task.endpoints[0].atomic_numbers, dtype=int)
    maps = tuple(task.endpoints[0].atom_map_ids)
    n = int(z.size)
    branches, episodes, states, labels = [], [], [], []
    for label in ("A", "B"):
        bid, eid = f"{rid}#br{label}", f"{rid}#ep{label}0"
        branches.append(BranchRecord(branch_id=bid, endpoint_label=label,
                                     reaction_id=rid, episode_ids=(eid,)))
        episodes.append(EpisodeRecord(episode_id=eid, branch_id=bid,
                                      reaction_id=rid, start_state_id=f"{rid}|{label}|0000",
                                      budget_version="b", environment_version="e",
                                      reward_version="r"))
        for step in range(2):
            terminal = step == 1
            sid = f"{rid}|{label}|{step:04d}"
            states.append(TeacherState(
                state_id=sid, episode_id=eid, branch_id=bid, step_index=step,
                atomic_numbers=z.copy(), coordinates=np.zeros((n, 3)),
                atom_map_ids=maps, method_id="M", observation=None,
                observation_missing_reason=("teacher_frame_isolated" if terminal
                                            else "observation_not_exported"),
                boundary_type=("teacher_endpoint" if terminal else None),
                next_state_id=None if terminal else f"{rid}|{label}|{step + 1:04d}"))
            labels.append(TeacherLabel(
                state_id=sid, ts_displacement=np.zeros((n, 3)),
                remaining_arc_length=0.0, teacher_endpoint=terminal,
                action_delta=None if terminal else np.zeros((n, 3))))
    teacher = DualTSTeacher(
        reaction_id=rid, reaction_group_id=task.reaction_group_id,
        branches=tuple(branches), episodes=tuple(episodes),
        states=tuple(states), labels=tuple(labels))
    teacher.validate()
    return teacher


def _mutate_labels(teacher: DualTSTeacher) -> DualTSTeacher:
    """Identical branches/episodes/states; labels MUTATED so the record no
    longer satisfies the teacher cross-record algebra (intentionally
    schema-INVALID — ``validate()`` would refuse it).  This is exactly the
    point: the topology/observation builders never READ the labels, so they
    never notice."""
    mutated = []
    for lab in teacher.labels:
        mutated.append(TeacherLabel(
            state_id=lab.state_id,
            ts_displacement=np.asarray(lab.ts_displacement) + 0.37,
            remaining_arc_length=float(lab.remaining_arc_length) + 1.7,
            teacher_endpoint=not bool(lab.teacher_endpoint),
            action_delta=(None if lab.action_delta is None
                          else np.asarray(lab.action_delta) + 0.21),
        ))
    return DualTSTeacher(
        reaction_id=teacher.reaction_id,
        reaction_group_id=teacher.reaction_group_id,
        branches=teacher.branches, episodes=teacher.episodes,
        states=teacher.states, labels=tuple(mutated))


def test_teacher_labels_do_not_change_observation_inputs():
    """Vary action_delta / is_teacher_endpoint; node_scalars + topology must
    be bit-identical (the observation builder no longer accepts them)."""
    task = _task()
    teacher = _teacher_for(task)
    topo = build_topology_features(task=task, teacher=teacher)

    x = np.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [2.1, 0.1, 0.0]])
    g = np.asarray([[0.02, -0.01, 0.005], [0.01, 0.03, -0.02],
                    [-0.005, 0.0, 0.015]])
    baseline = build_state_features(
        state_id="S", atomic_numbers=Z3.copy(), coordinates=x, gradient=g,
        delta_e_hartree=0.037, s_g=0.05, s_delta_e=1.2, charge=0,
        multiplicity=1)

    thr = NEAR_ZERO_ACTION_THRESHOLD_ANGSTROM
    interventions = [
        (None, True),                                    # terminal
        (np.zeros((3, 3)), False),
        (np.full((3, 3), thr / 2), False),               # all near-zero
        (np.full((3, 3), 2 * thr), False),               # all above threshold
        (np.asarray([[thr / 2, 0, 0], [2 * thr, 0, 0], [thr, 0, 0]]), False),
    ]
    # Supervision MUST actually change across interventions (proves the
    # "intervention" moved the label), while observation stays fixed.
    mask_signatures = set()
    for action_delta, is_teacher_endpoint in interventions:
        sup = build_teacher_supervision(
            action_delta=action_delta, is_teacher_endpoint=is_teacher_endpoint,
            n_atoms=3)
        mask_signatures.add(tuple(sup.near_zero_action_mask.tolist()))
        # Rebuild observation inputs under the SAME fixed observation.
        feats = build_state_features(
            state_id="S", atomic_numbers=Z3.copy(), coordinates=x, gradient=g,
            delta_e_hartree=0.037, s_g=0.05, s_delta_e=1.2, charge=0,
            multiplicity=1)
        np.testing.assert_array_equal(feats.node_scalars,
                                      baseline.node_scalars)
        np.testing.assert_array_equal(feats.x_centered, baseline.x_centered)
        np.testing.assert_array_equal(feats.g_scaled, baseline.g_scaled)
    # Topology blocks are teacher-label-independent by construction.
    assert topo.n_edges_undirected > 0
    assert len(mask_signatures) >= 2  # the labels genuinely differed
    # The observation builder rejects any attempt to pass teacher labels.
    from pes2ts_core.pesn.features.dual_ts_graph import build_state_features \
        as _bsf
    with pytest.raises(TypeError):
        _bsf(state_id="S", atomic_numbers=Z3, coordinates=x, gradient=g,
             delta_e_hartree=0.0, s_g=0.05, s_delta_e=1.2, charge=0,
             multiplicity=1, action_delta=np.zeros((3, 3)))
    with pytest.raises(TypeError):
        _bsf(state_id="S", atomic_numbers=Z3, coordinates=x, gradient=g,
             delta_e_hartree=0.0, s_g=0.05, s_delta_e=1.2, charge=0,
             multiplicity=1, is_teacher_endpoint=False)


def test_topology_independent_of_teacher_labels_disputed_path():
    """Finding 1 / F4(a) literal coverage: the disputed-5 covalent fallback
    is the one remaining builder that reads the teacher record (episode-start
    coordinates).  Two teachers with IDENTICAL episodes/states (same
    episode-start geometry) but MUTATED labels must yield bit-identical
    topology arrays AND the same observation features — labels never enter
    those inputs."""
    task = _disputed_task()
    base = _teacher_for(task)
    mutated = _mutate_labels(base)

    # The intervention genuinely changed the label content.
    assert not np.array_equal(base.labels[0].ts_displacement,
                              mutated.labels[0].ts_displacement)
    assert base.labels[0].teacher_endpoint != mutated.labels[0].teacher_endpoint
    assert base.labels[0].remaining_arc_length != \
        mutated.labels[0].remaining_arc_length

    topo_base = build_topology_features(task=task, teacher=base)
    topo_mut = build_topology_features(task=task, teacher=mutated)

    # The disputed path really did run through the teacher-reading fallback.
    assert topo_base.bond_source_a == BOND_SOURCE_COVALENT_FALLBACK
    assert topo_base.bond_source_b == BOND_SOURCE_COVALENT_FALLBACK
    assert topo_base.n_edges_undirected > 0

    # Every topology array is bit-identical under the label mutation.
    np.testing.assert_array_equal(topo_base.atomic_numbers,
                                  topo_mut.atomic_numbers)
    np.testing.assert_array_equal(topo_base.edge_pairs, topo_mut.edge_pairs)
    np.testing.assert_array_equal(topo_base.edge_index, topo_mut.edge_index)
    np.testing.assert_array_equal(topo_base.atom_feats_a, topo_mut.atom_feats_a)
    np.testing.assert_array_equal(topo_base.atom_feats_b, topo_mut.atom_feats_b)
    np.testing.assert_array_equal(topo_base.edge_feats_a, topo_mut.edge_feats_a)
    np.testing.assert_array_equal(topo_base.edge_feats_b, topo_mut.edge_feats_b)
    assert topo_base.bond_source_a == topo_mut.bond_source_a
    assert topo_base.bond_source_b == topo_mut.bond_source_b

    # The observation feature set is teacher-independent by construction
    # (build_state_features accepts NO teacher), so it is trivially
    # bit-identical across the two teachers: re-assert the frozen layout.
    x = np.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [2.1, 0.1, 0.0]])
    g = np.asarray([[0.02, -0.01, 0.005], [0.01, 0.03, -0.02],
                    [-0.005, 0.0, 0.015]])
    feats = build_state_features(
        state_id="S", atomic_numbers=Z3.copy(), coordinates=x, gradient=g,
        delta_e_hartree=0.037, s_g=0.05, s_delta_e=1.2, charge=0,
        multiplicity=1)
    assert feats.node_scalars.shape == (3, 5)


def test_shard_supervision_keys_do_not_change_collated_batch():
    """Mutating stored teacher npz keys leaves the collated model batch
    inputs unchanged (supervision is metadata, never a network input)."""
    torch = require_torch()
    rng = np.random.default_rng(0)
    n, pairs = 3, np.array([[0, 1], [0, 2]])
    directed = np.concatenate([pairs, pairs[:, ::-1]], axis=0).T
    state = {
        "atomic_numbers": Z3.copy(),
        "node_scalars": rng.normal(size=(n, 5)),
        "x_angstrom": rng.normal(size=(n, 3)),
        "g_scaled": rng.normal(size=(n, 3)),
        "atom_feats_a": rng.random((n, 4)),
        "atom_feats_b": rng.random((n, 4)),
        "edge_feats_a": rng.random((2 * len(pairs), 4)),
        "edge_feats_b": rng.random((2 * len(pairs), 4)),
        "edge_index": directed,
        # supervision metadata (top-level npz keys; NOT in REQUIRED_STATE_KEYS)
        "near_zero_action_mask": np.asarray([True, False, False]),
        "is_teacher_endpoint": np.asarray([False]),
    }
    baseline = collate_dual_states(torch, [state])
    # flip the supervision keys; the model-input batch must be unchanged
    mutated = dict(state)
    mutated["near_zero_action_mask"] = np.asarray([False, True, True])
    mutated["is_teacher_endpoint"] = np.asarray([True])
    after = collate_dual_states(torch, [mutated])
    assert set(baseline.keys()) == set(after.keys())
    for key, val in baseline.items():
        if torch.is_tensor(val):
            assert torch.equal(val, after[key]), \
                f"supervision mutation changed model input {key}"
