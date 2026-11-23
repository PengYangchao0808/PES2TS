"""Dual-TS teacher contract tests (WP0a): swap invariance, label algebra,
boundary rules, zero-fill bans, digest determinism. Synthetic data only —
no truth data, no ACP, no network."""
from __future__ import annotations

import numpy as np
import pytest

from pes2ts_core.pesn.contracts.dual_ts import (
    ACTION_SCHEMA_VERSION,
    BOUNDARY_TEACHER_ENDPOINT,
    BUNDLE_SCHEMA_VERSION,
    FEATURES_SCHEMA_VERSION,
    FEATURES_SCHEMA_VERSION_V2,
    REWARD_SCHEMA_VERSION,
    SCHEMAS,
    TASK_SCHEMA_VERSION,
    TEACHER_SCHEMA_VERSION,
    TRANSITION_SCHEMA_VERSION,
    BranchRecord,
    DualEndpointTask,
    DualTSAction,
    DualTSActionKind,
    DualTSBundle,
    DualTSContractError,
    DualTSFeaturesSpec,
    DualTSTeacher,
    DualTSTransition,
    DualTSReward,
    EndpointBond,
    EndpointGraph,
    EpisodeRecord,
    ObservationRef,
    TeacherLabel,
    TeacherState,
    canonical_endpoint_order,
    content_digest_of,
    partition_by_reaction_group,
    radial_tanh_transform,
    validate_record,
)

N_ATOMS = 3
ATOMS = np.array([8, 1, 1])                      # O + 2H toy H-transfer
MAP_IDS = ("O", "H_mig", "H_spect")
STEP = 0.14
X_STAR = np.array([[0.0, 0.0, 0.0],
                   [0.96 + 3 * STEP, 0.0, 0.0],
                   [1.75, 0.0, 0.0]])


def coords_at(t: int) -> np.ndarray:
    return np.array([[0.0, 0.0, 0.0],
                     [0.96 + t * STEP, 0.0, 0.0],
                     [1.75, 0.0, 0.0]])


def make_bond(i: int, j: int, order: float, **over) -> EndpointBond:
    return EndpointBond(i=i, j=j, bond_order=order, **over)


def make_endpoint(bond_specs, stereo=None) -> EndpointGraph:
    bonds = [make_bond(i, j, o) for i, j, o in bond_specs]
    if stereo is not None:
        bonds[0] = EndpointBond(i=bonds[0].i, j=bonds[0].j,
                                bond_order=bonds[0].bond_order,
                                stereo=stereo, stereo_known=True)
    return EndpointGraph(atomic_numbers=ATOMS.copy(), atom_map_ids=MAP_IDS,
                         bonds=tuple(bonds))


A_ORDERS = {0: [(0, 1, 1.0)], 1: [(0, 1, 1.0), (1, 2, 0.5)]}
B_ORDERS = {0: [(1, 2, 1.0)], 1: [(1, 2, 1.0)]}


def build_task(ri: int = 0, group: str = "GRP_DUAL_T0", **over) -> DualEndpointTask:
    base = dict(
        reaction_id=f"RXN_DUAL_{ri}",
        reaction_group_id=group,
        endpoints=(make_endpoint(A_ORDERS[ri]), make_endpoint(B_ORDERS[ri])),
        charge=0, multiplicity=1,
        constraints={"plan": "v4", "role": "dual_endpoint"},
        source="synthetic_wp0a_fixture",
        method_id="graph_identity_v0",
        eligibility={"dual_sided": True},
    )
    base.update(over)
    return DualEndpointTask(**base)


def relabel(label: TeacherLabel, **over) -> TeacherLabel:
    base = dict(
        state_id=label.state_id,
        ts_displacement=label.ts_displacement,
        remaining_arc_length=label.remaining_arc_length,
        teacher_endpoint=label.teacher_endpoint,
        action_delta=label.action_delta,
    )
    base.update(over)
    return TeacherLabel(**base)


def restage(state: TeacherState, **over) -> TeacherState:
    base = dict(
        state_id=state.state_id, episode_id=state.episode_id,
        branch_id=state.branch_id, step_index=state.step_index,
        atomic_numbers=state.atomic_numbers, coordinates=state.coordinates,
        atom_map_ids=state.atom_map_ids, method_id=state.method_id,
        observation=state.observation,
        observation_missing_reason=state.observation_missing_reason,
        boundary_type=state.boundary_type, next_state_id=state.next_state_id,
    )
    base.update(over)
    return TeacherState(**base)


def rebuild_teacher(teacher: DualTSTeacher, **over) -> DualTSTeacher:
    base = dict(
        reaction_id=teacher.reaction_id,
        reaction_group_id=teacher.reaction_group_id,
        branches=teacher.branches, episodes=teacher.episodes,
        states=teacher.states, labels=teacher.labels,
    )
    base.update(over)
    return DualTSTeacher(**base)


def build_teacher(ri: int = 0, group: str = "GRP_DUAL_T0",
                  missing_obs_state: str | None = None,
                  **over) -> DualTSTeacher:
    branches, episodes, states, labels = [], [], [], []
    for lab in ("A", "B"):
        bid = f"r{ri}-b{lab}"
        eid = f"r{ri}-e{lab}"
        sids = [f"r{ri}-b{lab}-s{t}" for t in range(4)]
        branches.append(BranchRecord(
            branch_id=bid, endpoint_label=lab,
            reaction_id=f"RXN_DUAL_{ri}", episode_ids=(eid,)))
        episodes.append(EpisodeRecord(
            episode_id=eid, branch_id=bid, reaction_id=f"RXN_DUAL_{ri}",
            start_state_id=sids[0], budget_version="budget_v0",
            environment_version="ENV_dual_teacher_v0",
            reward_version="REW_dual_v0"))
        for t in range(4):
            missing = missing_obs_state == sids[t]
            states.append(TeacherState(
                state_id=sids[t], episode_id=eid, branch_id=bid,
                step_index=t,
                atomic_numbers=ATOMS.copy(), coordinates=coords_at(t),
                atom_map_ids=MAP_IDS, method_id="teacher_DFT_v0",
                observation=(None if (t == 3 or missing) else ObservationRef(
                    status="ok",
                    energy_hartree=-0.9 - 0.05 * t,
                    gradient_hartree_per_angstrom=np.full((N_ATOMS, 3), 0.01),
                    method_id="teacher_DFT_v0", software="toy",
                    settings_digest="a" * 64)),
                observation_missing_reason=(
                    "teacher_frame_isolated" if t == 3
                    else ("observation_not_exported" if missing else None)),
                boundary_type=(BOUNDARY_TEACHER_ENDPOINT if t == 3 else None),
                next_state_id=(None if t == 3 else sids[t + 1]),
            ))
            labels.append(TeacherLabel(
                state_id=sids[t],
                ts_displacement=X_STAR - coords_at(t),
                remaining_arc_length=(3 - t) * STEP,
                teacher_endpoint=(t == 3),
                action_delta=(None if t == 3
                              else coords_at(t + 1) - coords_at(t))))
    base = dict(
        reaction_id=f"RXN_DUAL_{ri}", reaction_group_id=group,
        branches=tuple(branches), episodes=tuple(episodes),
        states=tuple(states), labels=tuple(labels),
    )
    base.update(over)
    teacher = DualTSTeacher(**base)
    teacher.validate()
    return teacher


def make_action(**over) -> DualTSAction:
    u = np.zeros((N_ATOMS, 3))
    u[:, 0] = 0.05
    z = radial_tanh_transform(u)
    base = dict(
        kind=DualTSActionKind.MOVE, a_max_angstrom=0.1,
        latent_u=u, normalized_z=z, action_angstrom=0.1 * z,
        transform_version="radial_tanh_v0", density_version="diag_gauss_v0",
    )
    base.update(over)
    return DualTSAction(**base)


def make_transition(**over) -> DualTSTransition:
    base = dict(
        transition_id="tr0", state_id="r0-bA-s0",
        next_state_id="r0-bA-s1", episode_id="r0-eA", branch_id="r0-bA",
        proposed_action=make_action(),
        next_observation_digest="b" * 64,
        logical_cost=1,
        head_eligibility={
            "Q": {"eligible": True, "reason": "successor_and_reward_known"},
            "V": {"eligible": True, "reason": "compatible_successor"},
            "actor": {"eligible": True, "reason": "known_physical_action"},
        },
    )
    base.update(over)
    return DualTSTransition(**base)


def make_reward(**over) -> DualTSReward:
    base = dict(
        transition_id="tr0", logical_cost=0.01,
        outcome_protocol="strict_stop_protocol_v0",
        outcome_result=None, outcome_missing_reason="protocol_not_concluded",
        reward=None, reward_missing_reason="outcome_unknown",
        episode_return=None, return_missing_reason="protocol_not_concluded",
    )
    base.update(over)
    return DualTSReward(**base)


def make_bundle(**over) -> DualTSBundle:
    base = dict(
        run_id="run_dual_v4_smoke",
        bc_parent_checkpoint_digest="c" * 64,
        seeds={"python": 0, "numpy": 1, "cuda": 2},
        hyperparams={"tau": 0.7, "T_A": 3.0},
        dataset_version="dual_synth_v0", norm_version="fit_only_v0",
        split_version="split_dual_v0", action_version=ACTION_SCHEMA_VERSION,
        environment_version="ENV_dual_teacher_v0",
        reward_version="REW_dual_v0",
    )
    base.update(over)
    return DualTSBundle(**base)


def make_features(**over) -> DualTSFeaturesSpec:
    base = dict(
        frozen_scales={"x": 1.0, "g": 3.0},
        masks={"endpoint_pair": "canonical"},
        support_domain={"a_max": 0.1},
        declared_layout={"node_scalar": 5, "equiv_vec": 10},
        input_dim=15,
    )
    base.update(over)
    return DualTSFeaturesSpec(**base)


def rejects(fn, code: str) -> None:
    with pytest.raises(DualTSContractError) as err:
        fn()
    assert err.value.code == code, \
        f"expected {code}, got {err.value.code}: {err.value}"


# ---------------------------------------------------------------------------
# (d) A/B swap invariance + reaction_group partitioning
# ---------------------------------------------------------------------------

def test_swap_invariance_digest_and_partition():
    task = build_task()
    task.validate()
    swapped = build_task(endpoints=(task.endpoints[1], task.endpoints[0]))
    assert task.content_digest() == swapped.content_digest()
    first = canonical_endpoint_order(*task.endpoints)
    second = canonical_endpoint_order(*swapped.endpoints)
    assert first[0] is second[0] and first[1] is second[1]
    assert task.partition_key() == swapped.partition_key() == "GRP_DUAL_T0"
    # duplicate reactions of the same physical reaction share the group key
    r0 = build_task(0, group="GRP_DUAL_T0")
    r1 = build_task(1, group="GRP_DUAL_T0")
    other = build_task(0, group="GRP_DUAL_T1")
    groups = partition_by_reaction_group([r0, r1, other])
    assert set(groups) == {"GRP_DUAL_T0", "GRP_DUAL_T1"}
    assert {t.reaction_id for t in groups["GRP_DUAL_T0"]} == {
        "RXN_DUAL_0", "RXN_DUAL_1"}
    # teacher partition key matches the task partition
    assert build_teacher().reaction_group_id == task.partition_key()


def test_canonical_order_handles_identical_endpoints():
    pair1 = canonical_endpoint_order(make_endpoint(A_ORDERS[0]),
                                     make_endpoint(A_ORDERS[0]))
    pair2 = canonical_endpoint_order(pair1[1], pair1[0])
    assert pair1[0].content_digest() == pair1[1].content_digest()
    assert ([p.content_digest() for p in pair1]
            == [p.content_digest() for p in pair2])


# ---------------------------------------------------------------------------
# endpoint unit integrity, incl. (e) merged bond-order rejection
# ---------------------------------------------------------------------------

def test_task_rejections():
    rejects(lambda: build_task(multiplicity=0).validate(),
            "MULTIPLICITY_INVALID")
    bad_map = make_endpoint(A_ORDERS[0])
    other = EndpointGraph(atomic_numbers=ATOMS.copy(),
                          atom_map_ids=("O", "H_x", "H_spect"),
                          bonds=make_endpoint(B_ORDERS[0]).bonds)
    rejects(lambda: build_task(endpoints=(bad_map, other)).validate(),
            "ENDPOINT_MAP_MISMATCH")
    rejects(lambda: make_endpoint([(0, 5, 1.0)]).validate(),
            "BOND_INDEX_INVALID")
    rejects(lambda: make_bond(0, 1, 1.0, stereo="E",
                              stereo_known=False).validate(3),
            "STEREO_UNMASKED")
    rejects(lambda: make_bond(0, 1, 0.0).validate(3), "BOND_ORDER_INVALID")
    rejects(lambda: build_task(reaction_group_id="").validate(), "ID_MISSING")
    rejects(lambda: build_task(method_id="").validate(), "ID_MISSING")
    # stereo with its known-mask set passes
    make_endpoint(A_ORDERS[0], stereo="trans").validate()


def test_merged_bond_orders_rejected_and_association_kept():
    # (e) negative: the classic flattening — per-edge bond orders sorted into
    # a detached list — must be rejected by the canonical form.
    payload = build_task(ri=1).canonical_payload()
    broken = {
        "schema_version": payload["schema_version"],
        "reaction_id": payload["reaction_id"],
        "reaction_group_id": payload["reaction_group_id"],
        "endpoints": [
            {**payload["endpoints"][0],
             "bonds": [{**b, "bond_orders": [0.5, 1.0]}
                       for b in payload["endpoints"][0]["bonds"]]},
            payload["endpoints"][1],
        ],
        "charge": payload["charge"],
        "multiplicity": payload["multiplicity"],
        "constraints": payload["constraints"],
        "source": payload["source"],
        "method_id": payload["method_id"],
        "eligibility": payload["eligibility"],
    }
    rejects(lambda: validate_record(TASK_SCHEMA_VERSION, broken),
            "BOND_ORDER_FORMAT_INVALID")
    # canonical form keeps orders bound to their edges: permuting the two
    # orders of endpoint A's two edges changes its digest (and the task
    # digest) while B stays untouched — no order-bag shortcuts.
    task = build_task(ri=1)
    ga, gb = task.endpoints
    ga_swapped = make_endpoint([(0, 1, 0.5), (1, 2, 1.0)])
    assert ga.content_digest() != ga_swapped.content_digest()
    assert gb.content_digest() == make_endpoint(B_ORDERS[1]).content_digest()
    assert (build_task(ri=1, endpoints=(ga_swapped, gb)).content_digest()
            != task.content_digest())
    # ... and swap invariance still holds after the modification
    assert (build_task(ri=1, endpoints=(gb, ga_swapped)).content_digest()
            == build_task(ri=1, endpoints=(ga_swapped, gb)).content_digest())


# ---------------------------------------------------------------------------
# (a) label algebra: X+a=X_next and d_t − d_{t+1} = a_t
# ---------------------------------------------------------------------------

def test_teacher_valid_and_label_algebra():
    # fixture coverage: 2 tiny reactions x 2 branches x 4 states each
    # (including one TS terminal state per branch)
    teachers = [build_teacher(0), build_teacher(1)]
    for teacher in teachers:
        teacher.validate()
        assert len(teacher.states) == 8 and len(teacher.labels) == 8
        assert sum(s.boundary_type == BOUNDARY_TEACHER_ENDPOINT
                   for s in teacher.states) == 2
    assert sum(len(t.states) for t in teachers) == 16
    teacher = teachers[0]
    teacher.content_digest()

    def with_action_perturbation(eps):
        labels = list(teacher.labels)
        labels[0] = relabel(labels[0],
                            action_delta=labels[0].action_delta
                            + np.array([[0.0, 0.0, 0.0],
                                        [eps, 0.0, 0.0],
                                        [0.0, 0.0, 0.0]]))
        rebuild_teacher(teacher, labels=tuple(labels)).validate()

    rejects(lambda: with_action_perturbation(2e-6), "DISPLACEMENT_MISMATCH")
    with_action_perturbation(5e-7)  # within 1e-6 tolerance stays valid

    # d_{t+1} perturbed -> d_t - d_{t+1} != a_t
    labels = list(teacher.labels)
    labels[1] = relabel(labels[1],
                        ts_displacement=labels[1].ts_displacement
                        - np.array([[0.0, 0.0, 0.0],
                                    [2e-6, 0.0, 0.0],
                                    [0.0, 0.0, 0.0]]))
    rejects(lambda: rebuild_teacher(teacher, labels=tuple(labels)).validate(),
            "LABEL_MISMATCH")

    # non-terminal state without an action carrier
    labels = list(teacher.labels)
    labels[0] = relabel(labels[0], action_delta=None)
    rejects(lambda: rebuild_teacher(teacher, labels=tuple(labels)).validate(),
            "ACTION_DELTA_MISSING")

    # remaining arc length must be non-increasing along a branch
    labels = list(teacher.labels)
    labels[0] = relabel(labels[0], remaining_arc_length=0.1)
    rejects(lambda: rebuild_teacher(teacher, labels=tuple(labels)).validate(),
            "ARC_LENGTH_INCREASING")

    # label coverage: dropping one label is a typed rejection
    rejects(lambda: rebuild_teacher(teacher, labels=tuple(teacher.labels[:1])
                                    + tuple(teacher.labels[2:])).validate(),
            "LABEL_COVERAGE_INVALID")


# ---------------------------------------------------------------------------
# (b) cross-branch successor / branch integrity
# ---------------------------------------------------------------------------

def test_cross_branch_successor_rejected():
    teacher = build_teacher()
    states = list(teacher.states)
    idx = next(i for i, s in enumerate(states) if s.state_id == "r0-bA-s1")
    states[idx] = restage(states[idx], next_state_id="r0-bB-s3")
    rejects(lambda: rebuild_teacher(teacher, states=tuple(states)).validate(),
            "CROSS_BRANCH_SUCCESSOR")
    # unknown successor id
    states[idx] = restage(states[idx], next_state_id="missing-state")
    rejects(lambda: rebuild_teacher(teacher, states=tuple(states)).validate(),
            "SUCCESSOR_UNKNOWN")
    # step continuity inside one episode
    states[idx] = restage(states[idx], next_state_id="r0-bA-s3")
    rejects(lambda: rebuild_teacher(teacher, states=tuple(states)).validate(),
            "STEP_CONTINUITY_INVALID")


# ---------------------------------------------------------------------------
# (c) successor after teacher_endpoint rejected
# ---------------------------------------------------------------------------

def test_successor_after_teacher_endpoint_rejected():
    ts_state = TeacherState(
        state_id="s_ts", episode_id="e0", branch_id="b0", step_index=3,
        atomic_numbers=ATOMS.copy(), coordinates=X_STAR.copy(),
        atom_map_ids=MAP_IDS, method_id="teacher_DFT_v0",
        observation=None, observation_missing_reason="teacher_frame_isolated",
        boundary_type=BOUNDARY_TEACHER_ENDPOINT, next_state_id="s_next")
    rejects(lambda: ts_state.validate(), "TEACHER_ENDPOINT_WITH_SUCCESSOR")
    rejects(lambda: restage(ts_state, boundary_type="magic",
                            next_state_id=None).validate(),
            "BOUNDARY_UNKNOWN")

    teacher = build_teacher()
    # teacher-endpoint label with an action carrier (even zero) is rejected
    labels = list(teacher.labels)
    labels[3] = relabel(labels[3], teacher_endpoint=True,
                        action_delta=np.zeros((N_ATOMS, 3)))
    rejects(lambda: rebuild_teacher(teacher, labels=tuple(labels)).validate(),
            "TEACHER_ENDPOINT_WITH_ACTION")
    # TS state retaining nonzero remaining displacement
    d = teacher.labels[3].ts_displacement.copy()
    d[1, 0] += 5e-3
    labels = list(teacher.labels)
    labels[3] = relabel(labels[3], ts_displacement=d)
    rejects(lambda: rebuild_teacher(teacher, labels=tuple(labels)).validate(),
            "TEACHER_ENDPOINT_DISPLACED")
    # flag/boundary disagreement
    labels = list(teacher.labels)
    labels[3] = relabel(labels[3], teacher_endpoint=False)
    rejects(lambda: rebuild_teacher(teacher, labels=tuple(labels)).validate(),
            "TEACHER_ENDPOINT_FLAG_MISMATCH")


# ---------------------------------------------------------------------------
# (g) missing observation: null + reason ok, zero-fill rejected
# ---------------------------------------------------------------------------

def test_missing_observation_null_plus_reason():
    teacher = build_teacher(missing_obs_state="r0-bA-s1")
    teacher.validate()  # null + registered reason accepted
    assert teacher.states[1].observation is None
    bare = dict(
        state_id="s", episode_id="e", branch_id="b", step_index=0,
        atomic_numbers=ATOMS.copy(), coordinates=coords_at(0),
        atom_map_ids=MAP_IDS, method_id="m",
    )
    rejects(lambda: TeacherState(**bare, observation=None,
                                 observation_missing_reason=None).validate(),
            "OBS_REASON_UNKNOWN")
    rejects(lambda: TeacherState(**bare, observation=None,
                                 observation_missing_reason="pretend_zero"
                                 ).validate(),
            "OBS_REASON_UNKNOWN")
    # zero-fill: exactly-zero energy + exactly-zero gradient
    rejects(lambda: ObservationRef(
        status="ok", energy_hartree=0.0,
        gradient_hartree_per_angstrom=np.zeros((N_ATOMS, 3))).validate(),
        "OBS_ZERO_FILLED")
    # missing status carrying zero arrays instead of nulls
    rejects(lambda: ObservationRef(
        status="missing", missing_reason="computation_missing",
        energy_hartree=0.0,
        gradient_hartree_per_angstrom=np.zeros((N_ATOMS, 3))).validate(),
        "OBS_ZERO_FILLED")
    # present observation with a stale missing_reason is a conflict
    rejects(lambda: ObservationRef(
        status="ok", energy_hartree=-0.9,
        gradient_hartree_per_angstrom=np.full((N_ATOMS, 3), 0.01),
        missing_reason="computation_missing").validate(),
        "OBS_REASON_CONFLICT")
    # a real (nonzero) observation passes
    ObservationRef(status="ok", energy_hartree=-0.9,
                   gradient_hartree_per_angstrom=np.full((N_ATOMS, 3), 0.01)
                   ).validate()


# ---------------------------------------------------------------------------
# (f) STOP_AND_REFINE is never a zero-displacement MOVE
# ---------------------------------------------------------------------------

def test_stop_and_refine_distinct_from_zero_move():
    stop = make_action(kind=DualTSActionKind.STOP_AND_REFINE,
                       latent_u=None, normalized_z=None, action_angstrom=None,
                       reason="strict_stop_protocol_v0")
    stop.validate()
    zero_u = np.zeros((N_ATOMS, 3))
    zero_z = radial_tanh_transform(zero_u)
    zero_move = make_action(latent_u=zero_u, normalized_z=zero_z,
                            action_angstrom=0.1 * zero_z)
    zero_move.validate()  # zero MOVE is legal — as a MOVE
    assert stop.kind is not zero_move.kind
    assert stop.content_digest() != zero_move.content_digest()
    # STOP carrying (zero) displacement carriers is rejected
    rejects(lambda: make_action(
        kind=DualTSActionKind.STOP_AND_REFINE,
        latent_u=zero_u, normalized_z=zero_u, action_angstrom=zero_u,
        reason="strict_stop_protocol_v0").validate(),
        "STOP_WITH_DISPLACEMENT")
    rejects(lambda: make_action(
        kind=DualTSActionKind.STOP_AND_REFINE, latent_u=None,
        normalized_z=None, action_angstrom=None, reason=None).validate(),
        "REASON_REQUIRED")
    # MOVE carrier algebra enforced
    z = radial_tanh_transform(make_action().latent_u)
    rejects(lambda: make_action(normalized_z=z * 1.001,
                                action_angstrom=0.1 * z).validate(),
            "Z_TRANSFORM_MISMATCH")
    rejects(lambda: make_action(action_angstrom=0.2 * z).validate(),
            "ACTION_SCALE_MISMATCH")
    rejects(lambda: make_action(a_max_angstrom=0.0).validate(),
            "ACTION_MAX_INVALID")
    rejects(lambda: make_action(
        kind=DualTSActionKind.ABORT, latent_u=None, normalized_z=None,
        action_angstrom=None, reason=None).validate(), "REASON_REQUIRED")
    # accepted fallback records the executed physical action, not samples
    fallback = make_action(kind=DualTSActionKind.ACCEPTED_FALLBACK,
                           latent_u=None, normalized_z=None,
                           reason="host_proposed_accepted")
    fallback.validate()
    rejects(lambda: make_action(
        kind=DualTSActionKind.ACCEPTED_FALLBACK, latent_u=None,
        normalized_z=None, action_angstrom=None, reason="host").validate(),
        "FALLBACK_ACTION_MISSING")
    rejects(lambda: make_action(
        kind=DualTSActionKind.ACCEPTED_FALLBACK, latent_u=None,
        normalized_z=None).validate(), "REASON_REQUIRED")
    rejects(lambda: make_action(
        kind=DualTSActionKind.ACCEPTED_FALLBACK,
        reason="host_proposed_accepted").validate(),
        "FALLBACK_NOT_SAMPLED")


def test_radial_transform_continuous_limit():
    assert not radial_tanh_transform(np.zeros((2, 3))).any()
    tiny = np.full((2, 3), 1e-12)
    assert float(np.max(np.abs(radial_tanh_transform(tiny)))) < 1e-11
    big = np.full((2, 3), 50.0)
    z = radial_tanh_transform(big)
    norms = np.linalg.norm(big, axis=-1)
    assert float(np.max(np.abs(np.linalg.norm(z, axis=-1) - np.tanh(norms)))) < 1e-12


# ---------------------------------------------------------------------------
# (h) digest determinism
# ---------------------------------------------------------------------------

def test_digest_determinism_and_order_independence():
    t1, t2 = build_task(), build_task()
    assert t1.content_digest() == t2.content_digest()
    payload = t1.canonical_payload()
    reordered = dict(reversed(list(payload.items())))
    assert content_digest_of(payload) == content_digest_of(reordered)
    perturbed = build_task(constraints={"plan": "v4", "role": "other"})
    assert perturbed.content_digest() != t1.content_digest()

    s1, s2 = build_teacher(), build_teacher()
    assert s1.content_digest() == s2.content_digest()
    shifted = build_teacher(arc_length_metric_version="other_metric")
    assert shifted.content_digest() != s1.content_digest()
    # perturbing a single state field changes the digest of the payload
    states = list(s1.states)
    states[5] = restage(states[5],
                        coordinates=states[5].coordinates + 1e-3)
    assert (DualTSTeacher(
        reaction_id=s1.reaction_id, reaction_group_id=s1.reaction_group_id,
        branches=s1.branches, episodes=s1.episodes,
        states=tuple(states), labels=s1.labels).to_payload()
        != s1.to_payload())
    # (the same perturbation breaks label algebra and is a typed rejection)
    rejects(lambda: rebuild_teacher(s1, states=tuple(states)).validate(),
            "DISPLACEMENT_MISMATCH")


# ---------------------------------------------------------------------------
# (i) file-slice boundary is not terminal
# ---------------------------------------------------------------------------

def test_file_slice_not_terminal():
    slice_end = make_transition(boundary="data_boundary",
                                boundary_reason="export_slice_end")
    slice_end.validate()  # terminal=False, real successor referenced
    assert slice_end.next_state_id is not None
    assert not slice_end.terminal and not slice_end.censored
    rejects(lambda: make_transition(
        boundary="data_boundary", terminal=True,
        boundary_reason="export_slice_end").validate(), "SLICE_NOT_TERMINAL")
    rejects(lambda: make_transition(
        boundary="data_boundary", censored=True,
        boundary_reason="export_slice_end").validate(), "SLICE_NOT_TERMINAL")
    rejects(lambda: make_transition(
        boundary="data_boundary", next_state_id=None,
        next_observation_digest=None,
        next_observation_missing_reason="observation_not_exported",
        ).validate(), "SLICE_WITHOUT_SUCCESSOR")
    # real terminals: teacher endpoint and budget exhaustion
    make_transition(boundary=BOUNDARY_TEACHER_ENDPOINT, terminal=True,
                    next_state_id=None, next_observation_digest=None,
                    next_observation_missing_reason="teacher_frame_isolated",
                    ).validate()
    make_transition(boundary="budget_exhausted", terminal=True,
                    next_state_id=None, next_observation_digest=None,
                    next_observation_missing_reason="computation_missing",
                    ).validate()
    rejects(lambda: make_transition(
        boundary="none", terminal=True, next_state_id=None,
        next_observation_digest=None,
        next_observation_missing_reason="observation_not_exported",
        ).validate(), "TERMINAL_BOUNDARY_INVALID")
    rejects(lambda: make_transition(
        terminal=False, next_state_id=None).validate(), "NEXT_STATE_MISSING")
    # censoring: registered censored boundary + reason, never terminal
    make_transition(boundary="external_censored", censored=True,
                    boundary_reason="host_timeout",
                    next_observation_digest=None,
                    next_observation_missing_reason="censored_missing",
                    ).validate()
    rejects(lambda: make_transition(
        censored=True, boundary_reason=None).validate(),
        "CENSOR_BOUNDARY_INVALID")
    rejects(lambda: make_transition(
        boundary="external_censored", censored=True, terminal=True,
        boundary_reason="host_timeout", next_state_id=None,
        next_observation_digest=None,
        next_observation_missing_reason="censored_missing").validate(),
        "CENSOR_NOT_TERMINAL")
    # per-head eligibility: registered heads, boolean eligible, real reason
    rejects(lambda: make_transition(
        head_eligibility={"policy": {"eligible": True, "reason": "x"}}
    ).validate(), "HEAD_UNKNOWN")
    rejects(lambda: make_transition(
        head_eligibility={"Q": {"eligible": True}}).validate(),
        "ELIGIBILITY_MALFORMED")
    rejects(lambda: make_transition(
        next_observation_digest="nothex").validate(), "DIGEST_MALFORMED")


# ---------------------------------------------------------------------------
# (j) unknown reward/outcome/return: None + reason, never 0
# ---------------------------------------------------------------------------

def test_reward_unknown_not_zero():
    make_reward().validate()
    # a present value together with an "unknown" reason is a conflict —
    # unknown is encoded as None + reason, never as a fabricated 0
    rejects(lambda: make_reward(reward=0.0,
                                reward_missing_reason="outcome_unknown"
                                ).validate(), "REWARD_CONFLICT")
    rejects(lambda: make_reward(reward_missing_reason=None).validate(),
            "REASON_UNKNOWN")
    rejects(lambda: make_reward(reward_missing_reason="made_up").validate(),
            "REASON_UNKNOWN")
    rejects(lambda: make_reward(outcome_result="ts_confirmed",
                                outcome_protocol="",
                                outcome_missing_reason=None).validate(),
            "PROTOCOL_REQUIRED")
    rejects(lambda: make_reward(outcome_missing_reason=None).validate(),
            "REASON_UNKNOWN")
    make_reward(outcome_result="ts_confirmed",
                outcome_protocol="strict_stop_protocol_v0",
                outcome_missing_reason=None,
                reward=1.5, reward_missing_reason=None).validate()
    rejects(lambda: make_reward(episode_return=0.0,
                                return_missing_reason="not_evaluated"
                                ).validate(), "RETURN_CONFLICT")
    rejects(lambda: make_reward(logical_cost=-0.1).validate(),
            "LOGICAL_COST_INVALID")


# ---------------------------------------------------------------------------
# feature spec / bundle validation
# ---------------------------------------------------------------------------

def test_features_spec():
    make_features().validate()
    rejects(lambda: make_features(input_dim=14).validate(),
            "INPUT_DIM_MISMATCH")
    rejects(lambda: make_features(frozen_scales={"x": 0.0, "g": 1.0}
                                  ).validate(), "SCALE_INVALID")
    rejects(lambda: make_features(
        delta_energy_definition="E - E_final").validate(),
        "DELTA_E_DEFINITION_FROZEN")
    rejects(lambda: make_features(equivariant_vectors=("a",)).validate(),
            "EQUIVARIANT_CHANNEL_INVALID")
    rejects(lambda: make_features(encoding="plain_mlp").validate(),
            "ENCODING_INVALID")
    assert make_features().content_digest() == make_features().content_digest()


def test_features_layout_sum_rules_v2_and_v3():
    """The frozen spec layout sum rule holds for BOTH the new v3 (5-channel
    node scalars) and the legacy v2 (7-channel) schemas — v2 stays registered
    so old products remain readable."""
    from pes2ts_core.pesn.features.dual_ts_spec import (
        build_features_spec,
        declared_layout_blocks,
    )
    layout = declared_layout_blocks()
    assert layout["node_scalar"] == 5
    assert sum(layout.values()) == 15
    _spec, v3 = build_features_spec(
        s_g=0.05, s_delta_e=0.4, element_vocab=[1, 6, 7, 8],
        population={"wave": "fit"}, n_atoms_range=(3, 3))
    assert v3["schema_version"] == FEATURES_SCHEMA_VERSION
    assert v3["input_dim"] == 15
    assert v3["declared_layout"]["node_scalar"] == 5
    validate_record(FEATURES_SCHEMA_VERSION, v3)
    # legacy v2: 7-channel node scalars, input_dim 17; still validates.
    v2 = dict(v3)
    v2["schema_version"] = FEATURES_SCHEMA_VERSION_V2
    v2["declared_layout"] = dict(v3["declared_layout"], node_scalar=7)
    v2["input_dim"] = 17
    validate_record(FEATURES_SCHEMA_VERSION_V2, v2)


def test_features_spec_schema_node_scalar_coupling():
    """v4.1 WP1 hardening: schema NAME ↔ node-scalar width must agree.

    A synthetic v3 payload declaring 7 channels, or a v2 payload declaring
    5, is rejected with a typed code (oracle G1 accepted-risk: the digest
    binds the real layout, this makes the coupling explicit in ``validate``).
    """
    rejects(lambda: make_features(
        declared_layout={"node_scalar": 7, "equiv_vec": 8},
        input_dim=15).validate(), "NODE_SCALAR_SCHEMA_MISMATCH")
    rejects(lambda: make_features(
        schema_version=FEATURES_SCHEMA_VERSION_V2,
        declared_layout={"node_scalar": 5, "equiv_vec": 12},
        input_dim=17).validate(), "NODE_SCALAR_SCHEMA_MISMATCH")


def test_bundle_validation():
    make_bundle().validate()
    rejects(lambda: make_bundle(bc_parent_checkpoint_digest="zz"
                                ).validate(), "DIGEST_MALFORMED")
    rejects(lambda: make_bundle(dataset_version="").validate(),
            "VERSION_BINDING_MISSING")
    rejects(lambda: make_bundle(seeds={}).validate(), "SEEDS_MISSING")
    rejects(lambda: make_bundle(seeds={"python": "zero"}).validate(),
            "SEED_INVALID")


# ---------------------------------------------------------------------------
# registry + roundtrips
# ---------------------------------------------------------------------------

def test_registry_dispatch_and_names():
    assert set(SCHEMAS) == {
        TASK_SCHEMA_VERSION, TEACHER_SCHEMA_VERSION, FEATURES_SCHEMA_VERSION,
        FEATURES_SCHEMA_VERSION_V2,
        ACTION_SCHEMA_VERSION, TRANSITION_SCHEMA_VERSION,
        REWARD_SCHEMA_VERSION, BUNDLE_SCHEMA_VERSION,
    }
    for name, info in SCHEMAS.items():
        assert info.name == name and callable(info.validator)
    payloads = {
        TASK_SCHEMA_VERSION: build_task().canonical_payload(),
        TEACHER_SCHEMA_VERSION: build_teacher().to_payload(),
        FEATURES_SCHEMA_VERSION: make_features().to_payload(),
        FEATURES_SCHEMA_VERSION_V2: dict(
            make_features().to_payload(),
            schema_version=FEATURES_SCHEMA_VERSION_V2,
            declared_layout={"node_scalar": 7, "equiv_vec": 10},
            input_dim=17),
        ACTION_SCHEMA_VERSION: make_action().to_payload(),
        TRANSITION_SCHEMA_VERSION: make_transition().to_payload(),
        REWARD_SCHEMA_VERSION: make_reward().to_payload(),
        BUNDLE_SCHEMA_VERSION: make_bundle().to_payload(),
    }
    for name, payload in payloads.items():
        validate_record(name, payload)  # must not raise
    with pytest.raises(DualTSContractError) as err:
        validate_record("pesn_dual_endpoint_task_v9",
                        payloads[TASK_SCHEMA_VERSION])
    assert err.value.code == "SCHEMA_UNKNOWN"
    with pytest.raises(DualTSContractError) as err:
        validate_record(TEACHER_SCHEMA_VERSION, {"schema_version": "x"})
    assert err.value.code == "PAYLOAD_MALFORMED"
    # schema_version mismatch inside a payload is typed, not silent
    wrong = dict(payloads[ACTION_SCHEMA_VERSION])
    wrong["schema_version"] = "pesn_dual_ts_action_v1"
    rejects(lambda: validate_record(ACTION_SCHEMA_VERSION, wrong),
            "SCHEMA_UNKNOWN")


def test_payload_roundtrips_preserve_digests():
    task = build_task(ri=1)
    assert DualEndpointTask.from_payload(
        task.canonical_payload()).content_digest() == task.content_digest()
    teacher = build_teacher()
    assert DualTSTeacher.from_payload(
        teacher.to_payload()).content_digest() == teacher.content_digest()
    for record in (make_action(), make_transition(), make_reward(),
                   make_bundle(), make_features()):
        clone = type(record).from_payload(record.to_payload())
        assert clone.content_digest() == record.content_digest()
