"""WP3-A dual-TS feature tests (synthetic only — no real data touched).

Covers the frozen ``pesn_dual_ts_features_v2`` decisions:

* swap invariance of the exchange-symmetric fused features (two DIFFERENT
  toy endpoint graphs; shared encoder, sum + elementwise-max fusion);
* translation invariance + rotation equivariance of the vector channels
  and the assembled state inputs;
* permutation equivariance (atom relabeling with edge relabeling);
* changed-bond channel correctness (formed/broken AND order-changed);
* disputed-5 rule application (covalent fallback iff flagged, authority
  otherwise, applied symmetrically);
* s_g / s_ΔE estimator math on hand-computed toy populations;
* spec digest determinism + validate_record round-trip;
* near-zero action mask correctness (threshold + teacher endpoint).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.contracts.dual_ts import (  # noqa: E402
    DualEndpointTask,
    DualTSTeacher,
    EndpointBond,
    EndpointGraph,
)
from pes2ts_core.pesn.features.dual_ts_graph import (  # noqa: E402
    DualTSEncoderConfig,
    DualTSFusionEncoder,
    build_state_features,
    build_teacher_supervision,
    build_topology_features,
    consumed_endpoint_graphs,
    endpoint_atom_features,
    endpoint_edge_features,
    fuse_endpoint_pair,
    topology_to_tensors,
    union_edge_pairs,
)
from pes2ts_core.pesn.features.dual_ts_spec import (  # noqa: E402
    NEAR_ZERO_ACTION_THRESHOLD_ANGSTROM,
    build_features_spec,
    declared_layout_blocks,
    estimate_fit_scales_from_records,
    load_spec_payload,
    save_spec_payload,
)
from pes2ts_core.pesn.offline.data.dual_ts_preprocess import (  # noqa: E402
    A_MAX_ANGSTROM,
    atom_map_ids_for,
    build_dual_ts_reaction,
    endpoint_graph_from_bond_list,
)

# ---------------------------------------------------------------------------
# Synthetic fixtures: two DIFFERENT toy endpoint graphs over a shared map
# ---------------------------------------------------------------------------

Z3 = np.asarray([6, 6, 1], dtype=int)  # C, C, H


def graph_a_orders() -> EndpointGraph:
    return EndpointGraph(
        atomic_numbers=Z3.copy(),
        atom_map_ids=atom_map_ids_for(3),
        bonds=(EndpointBond(0, 1, 1.0), EndpointBond(1, 2, 1.0)),
    )


def graph_b_orders() -> EndpointGraph:
    # DIFFERENT topology: C-C becomes double, C-H broken, C0-H2 formed.
    return EndpointGraph(
        atomic_numbers=Z3.copy(),
        atom_map_ids=atom_map_ids_for(3),
        bonds=(EndpointBond(0, 1, 2.0), EndpointBond(0, 2, 1.0)),
    )


def toy_task(rid: str = "RXN_T", a: EndpointGraph | None = None,
             b: EndpointGraph | None = None,
             eligibility: dict | None = None) -> DualEndpointTask:
    return DualEndpointTask(
        reaction_id=rid,
        reaction_group_id=rid + "G",
        endpoints=(a or graph_a_orders(), b or graph_b_orders()),
        method_id="M",
        eligibility=eligibility or {"bond_authority": "geometry_verified"},
    )


# ---------------------------------------------------------------------------
# (1) swap invariance of the fused features
# ---------------------------------------------------------------------------


def test_swap_invariance_fused_features_two_different_graphs():
    torch.manual_seed(3)
    enc = DualTSFusionEncoder(DualTSEncoderConfig(d_model=8, d_element=4))
    task = toy_task()
    topo = build_topology_features(task=task, teacher=_stub_teacher_for(task))
    topo_swapped = build_topology_features(
        task=toy_task(a=task.endpoints[1], b=task.endpoints[0]),
        teacher=_stub_teacher_for(task))
    t1 = topology_to_tensors(topo)
    t2 = topology_to_tensors(topo_swapped)
    with torch.no_grad():
        out1 = enc(**t1)
        out2 = enc(**t2)
    # Shared weights + symmetric fusion: the fused tensors are IDENTICAL,
    # and the (non-fused) endpoint embeddings merely swap roles.
    assert torch.allclose(out1["fused_atom"], out2["fused_atom"], atol=1e-6)
    assert torch.allclose(out1["fused_edge"], out2["fused_edge"], atol=1e-6)
    assert torch.allclose(out1["h_atom_a"], out2["h_atom_b"], atol=1e-6)
    assert torch.allclose(out1["h_atom_b"], out2["h_atom_a"], atol=1e-6)
    # The fusion helper itself is exactly symmetric for random inputs.
    ha = torch.randn(5, 7)
    hb = torch.randn(5, 7)
    assert torch.allclose(fuse_endpoint_pair(ha, hb),
                          fuse_endpoint_pair(hb, ha), atol=0.0)


# ---------------------------------------------------------------------------
# (2) translation invariance + rotation equivariance of vector channels
# ---------------------------------------------------------------------------

SG, SDE = 0.05, 1.2


def _state(x, g):
    return build_state_features(
        state_id="S", atomic_numbers=Z3.copy(), coordinates=x, gradient=g,
        delta_e_hartree=0.6, s_g=SG, s_delta_e=SDE, charge=0, multiplicity=1)


def test_translation_invariance_rotation_equivariance():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(3, 3))
    g = rng.normal(scale=0.04, size=(3, 3))
    base = _state(x, g)

    # Translation: centroid removal makes X invariant; everything else is
    # translation-blind by construction.
    shifted = _state(x + np.asarray([10.0, -3.0, 0.5]), g)
    assert np.allclose(base.x_centered, shifted.x_centered, atol=1e-12)
    assert np.array_equal(base.node_scalars, shifted.node_scalars)
    assert np.array_equal(base.g_scaled, shifted.g_scaled)

    # Rotation (det=+1) applied to coords AND gradient: vector channels
    # rotate, scalar block is invariant.
    theta = 0.9
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0.0, 0.0, 1.0]])
    rotated = _state(x @ rot.T, g @ rot.T)
    assert np.allclose(base.x_centered @ rot.T, rotated.x_centered, atol=1e-12)
    assert np.allclose(base.g_scaled @ rot.T, rotated.g_scaled, atol=1e-12)
    assert np.allclose(base.node_scalars, rotated.node_scalars, atol=1e-12)
    # Per-atom gradient norms (the scaled scalar channel) are invariant.
    assert np.allclose(
        base.node_scalars[:, 1], rotated.node_scalars[:, 1], atol=1e-12)


# ---------------------------------------------------------------------------
# (3) permutation equivariance (atom relabeling + edge relabeling)
# ---------------------------------------------------------------------------


def test_permutation_equivariance_atom_and_edge_relabeling():
    perm = np.asarray([2, 0, 1])
    inv = np.argsort(perm)
    task = toy_task()
    topo = build_topology_features(task=task, teacher=_stub_teacher_for(task))

    def relabel_graph(g: EndpointGraph) -> EndpointGraph:
        bonds = []
        for b in g.bonds:
            i, j = int(inv[b.i]), int(inv[b.j])
            bonds.append(EndpointBond(min(i, j), max(i, j), b.bond_order))
        return EndpointGraph(
            atomic_numbers=g.atomic_numbers[perm],
            atom_map_ids=tuple(g.atom_map_ids[p] for p in perm),
            bonds=tuple(sorted(bonds, key=lambda b: (b.i, b.j))),
        )

    task_perm = toy_task(a=relabel_graph(task.endpoints[0]),
                         b=relabel_graph(task.endpoints[1]))
    topo_perm = build_topology_features(task=task_perm,
                                        teacher=_stub_teacher_for(task_perm))

    # Node/atom blocks permute; edge sets remap consistently.
    assert np.array_equal(topo_perm.atomic_numbers, topo.atomic_numbers[perm])
    assert np.allclose(topo_perm.atom_feats_a, topo.atom_feats_a[perm])
    assert np.allclose(topo_perm.atom_feats_b, topo.atom_feats_b[perm])
    old_pairs = [(int(p[0]), int(p[1])) for p in topo.edge_pairs]
    new_pairs = {tuple(sorted((int(inv[i]), int(inv[j]))))
                 for i, j in old_pairs}
    assert {(int(p[0]), int(p[1])) for p in topo_perm.edge_pairs} == new_pairs

    # State assembly: relabeled build equals permuted build.
    rng = np.random.default_rng(7)
    x = rng.normal(size=(3, 3))
    g = rng.normal(scale=0.04, size=(3, 3))
    base = _state(x, g)
    relabeled = build_state_features(
        state_id="S", atomic_numbers=Z3[perm], coordinates=x[perm],
        gradient=g[perm], delta_e_hartree=0.6, s_g=SG, s_delta_e=SDE,
        charge=0, multiplicity=1)
    assert np.allclose(base.node_scalars[perm], relabeled.node_scalars)
    assert np.allclose(base.x_centered[perm], relabeled.x_centered)
    assert np.allclose(base.g_scaled[perm], relabeled.g_scaled)


# ---------------------------------------------------------------------------
# (4) changed-bond channel correctness
# ---------------------------------------------------------------------------


def test_changed_bond_channel_formed_broken_and_order_changed():
    bonds_a = {(0, 1): 1.0, (1, 2): 1.0}   # (1,2) will change ORDER
    bonds_b = {(0, 1): 2.0}                # (1,2) broken; (0,1) order-changed
    pairs = union_edge_pairs(bonds_a, bonds_b)
    assert {tuple(p) for p in pairs} == {(0, 1), (1, 2)}
    ea = endpoint_edge_features(bonds_a, bonds_b, pairs)
    eb = endpoint_edge_features(bonds_b, bonds_a, pairs)
    # Union edge (0,1): present both, order differs -> changed in BOTH blocks.
    assert ea[0, 0] == 1.0 and ea[0, 1] == 1.0 and ea[0, 2] == 1.0
    assert eb[0, 0] == 1.0 and eb[0, 1] == 2.0 and eb[0, 2] == 1.0
    # Union edge (1,2): present in A only -> broken; changed in both.
    assert ea[1, 0] == 1.0 and ea[1, 1] == 1.0 and ea[1, 2] == 1.0
    assert eb[1, 0] == 0.0 and eb[1, 1] == 0.0 and eb[1, 2] == 1.0
    # Unchanged spectator bond: present both, same order -> changed = 0.
    bonds_c = {(0, 1): 1.0, (2, 3): 1.0}
    bonds_d = {(0, 1): 1.0}
    pairs2 = union_edge_pairs(bonds_c, bonds_d)
    e2 = endpoint_edge_features(bonds_c, bonds_d, pairs2)
    assert e2[0, 2] == 0.0 and e2[1, 2] == 1.0
    # Per-atom changed-bond degree follows the incident changed edges:
    # only atoms 2,3 touch the broken (2,3); atoms 0,1 touch only the
    # unchanged (0,1).
    atom_c = endpoint_atom_features(4, bonds_c, bonds_d, pairs2)
    assert atom_c[0, 2] == 0.0 and atom_c[1, 2] == 0.0
    assert atom_c[2, 2] == 1.0 and atom_c[3, 2] == 1.0
    # Bond orders never merge across endpoints: block A keeps its own order.
    assert ea[0, 1] != eb[0, 1]


# ---------------------------------------------------------------------------
# (5) disputed-5 rule application (fallback bonds iff flagged)
# ---------------------------------------------------------------------------


def _disputed_probe_reaction() -> tuple[DualEndpointTask, DualTSTeacher]:
    """C-H pair: authority says bonded on BOTH sides; geometry disagrees
    (3.0 Å apart at the A endpoint, 0.7 Å at the B endpoint)."""
    z = np.asarray([6, 1], dtype=int)
    path = np.asarray([
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],   # A endpoint: C-H far
        [[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]],   # "TS"
        [[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]],   # B endpoint: C-H close
    ])
    authority = endpoint_graph_from_bond_list(z, [[1, 2, 1.0]])
    outcome = build_dual_ts_reaction(
        reaction_id="RXN_DISP",
        reaction_group_id="RXN_DISPG",
        partition="fit",
        atomic_numbers=z,
        ordered_path=path,
        authority_ts_position=1,
        energies=None,
        graph_r=authority,
        graph_p=authority,
    )
    assert outcome.task is not None and outcome.teacher is not None
    disputed_task = DualEndpointTask(
        reaction_id=outcome.task.reaction_id,
        reaction_group_id=outcome.task.reaction_group_id,
        endpoints=outcome.task.endpoints,
        charge=0, multiplicity=1,
        constraints=outcome.task.constraints,
        source=outcome.task.source,
        method_id=outcome.task.method_id,
        eligibility=dict(outcome.task.eligibility,
                         bond_authority="disputed"),
    )
    return disputed_task, outcome.teacher


def test_disputed_rule_fallback_used_iff_flagged():
    task, teacher = _disputed_probe_reaction()
    # Flagged disputed -> BOTH sides consume geometry perception (same
    # radii/tolerance as the dual_ts_preprocess fallback), applied
    # symmetrically to A and B.
    ga, gb, sa, sb = consumed_endpoint_graphs(task, teacher)
    assert sa == "covalent_fallback" and sb == "covalent_fallback"
    # Exactly the endpoint frame with the SHORT C-H (0.7 Å <= 0.76+0.31+
    # 0.45) is perceived as bonded; the 3.0 Å side is not.  (The authority
    # graph was bonded on BOTH sides — perception overrides it.)
    assert sorted(len(g.bonds) for g in (ga, gb)) == [0, 1]
    bonded = ga if len(ga.bonds) == 1 else gb
    assert {(b.i, b.j) for b in bonded.bonds} == {(0, 1)}
    # Unflagged -> authority bonds consumed unchanged on both sides.
    ga2, gb2, sa2, sb2 = consumed_endpoint_graphs(
        toy_task(a=task.endpoints[0], b=task.endpoints[1],
                 eligibility={"bond_authority": "geometry_verified"}),
        teacher)
    assert sa2 == "authority" and sb2 == "authority"
    assert len(ga2.bonds) == 1 and len(gb2.bonds) == 1
    # Topology features carry the flag and the fallback union edges.
    topo = build_topology_features(task=task, teacher=teacher)
    assert topo.bond_source_a == "covalent_fallback"
    assert topo.bond_source_b == "covalent_fallback"
    assert topo.n_edges_undirected == 1  # only B's perceived bond survives


# ---------------------------------------------------------------------------
# (6) s_g / s_ΔE estimator math on hand-computed toy populations
# ---------------------------------------------------------------------------


def test_scale_estimators_hand_computed_toy():
    # Per-atom gradient norms over the WHOLE population:
    # [1, 2] + [3, 4] + [1, 1] -> sorted [1,1,1,2,3,4]; s_g = median = 1.5.
    recs = [
        {"gradient": [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], "delta_e": 0.0},
        {"gradient": [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]], "delta_e": 2.0},
        {"gradient": [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], "delta_e": -3.0},
    ]
    scales = estimate_fit_scales_from_records(recs)
    assert scales["s_g"] == pytest.approx(1.5)
    # |ΔE| = [0, 2, 3]; p95 with numpy linear interpolation:
    # index 0.95*(3-1)=1.9 -> 2 + 0.9*(3-2) = 2.9.
    assert scales["s_delta_e"] == pytest.approx(np.percentile([0, 2, 3], 95))
    assert scales["s_delta_e"] == pytest.approx(2.9)
    # One shared scalar for ALL axes: per-axis scales never appear.
    with pytest.raises(ValueError):
        estimate_fit_scales_from_records([])
    with pytest.raises(ValueError):
        estimate_fit_scales_from_records(
            [{"gradient": [[0.0, 0.0, 0.0]], "delta_e": 0.0}])  # s_g = 0
    # Applied scaling preserves direction and relative magnitude.
    feats = build_state_features(
        state_id="S", atomic_numbers=Z3,
        coordinates=np.zeros((3, 3)),
        gradient=np.asarray([[2.5, 0.0, 0.0], [0.0, 0.0, 0.0],
                             [0.0, 5.0, 0.0]]),
        delta_e_hartree=2.85, s_g=2.5, s_delta_e=2.85, charge=0,
        multiplicity=1)
    assert feats.g_scaled[0, 0] == pytest.approx(1.0)
    assert feats.g_scaled[2, 1] == pytest.approx(2.0)
    assert feats.delta_e_scaled == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# (7) spec digest determinism + validate_record round-trip
# ---------------------------------------------------------------------------


def test_spec_digest_determinism_and_validate_record_roundtrip(tmp_path):
    kwargs = dict(
        s_g=0.0511, s_delta_e=0.42, element_vocab=[1, 6, 7, 8],
        population={"wave": "fit", "reactions": 1, "branches": 2,
                    "states": 10},
        n_atoms_range=(3, 3),
    )
    spec1, payload1 = build_features_spec(**kwargs)
    spec2, payload2 = build_features_spec(**kwargs)
    assert spec1.content_digest() == spec2.content_digest()
    assert payload1 == payload2
    # Layout contract: per-block dims sum exactly to input_dim.
    layout = declared_layout_blocks()
    assert sum(layout.values()) == payload1["input_dim"]
    assert payload1["declared_layout"] == layout
    # Registry validation + from_payload round-trip keep the digest stable.
    from pes2ts_core.pesn.contracts.dual_ts import (
        FEATURES_SCHEMA_VERSION,
        DualTSFeaturesSpec,
        validate_record,
    )
    validate_record(FEATURES_SCHEMA_VERSION, payload1)
    rt = DualTSFeaturesSpec.from_payload(payload1)
    assert rt.content_digest() == spec1.content_digest()
    # Save/load with digest verification; tampering is caught.
    path = tmp_path / "spec.json"
    assert save_spec_payload(payload1, path) == spec1.content_digest()
    loaded, loaded_payload = load_spec_payload(path)
    assert loaded.content_digest() == spec1.content_digest()
    assert loaded_payload == payload1
    bad = dict(payload1)
    bad["frozen_scales"] = dict(payload1["frozen_scales"], s_g=9.9)
    (tmp_path / "bad.json").write_text(
        json.dumps({"spec": bad,
                    "spec_digest": spec1.content_digest()}),
        encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_spec_payload(tmp_path / "bad.json")
    # Contract rejections still apply to our frozen payload shape.
    with pytest.raises(Exception):
        DualTSFeaturesSpec.from_payload(
            dict(payload1, frozen_scales={"s_g": 0.0}))


# ---------------------------------------------------------------------------
# (8) teacher supervision builder (near-zero mask + teacher endpoint)
# ---------------------------------------------------------------------------


def test_near_zero_mask_threshold_and_teacher_endpoint():
    thr = NEAR_ZERO_ACTION_THRESHOLD_ANGSTROM
    # Action norms: [thr/2 (masked), 2*thr (kept), exactly thr (kept: strict
    # <), 0 (masked)].
    a = np.asarray([
        [thr / 2, 0.0, 0.0],
        [2 * thr, 0.0, 0.0],
        [thr, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    sup = build_teacher_supervision(
        action_delta=a, is_teacher_endpoint=False, n_atoms=4)
    assert sup.near_zero_action_mask.tolist() == [True, False, False, True]
    assert sup.is_teacher_endpoint is False
    # Teacher endpoint states carry no action: EVERY atom masked out.
    term = build_teacher_supervision(
        action_delta=None, is_teacher_endpoint=True, n_atoms=4)
    assert term.near_zero_action_mask.all()
    assert term.is_teacher_endpoint is True
    # A non-terminal state without an action is a typed refusal.
    from pes2ts_core.pesn.features.dual_ts_graph import DualTSGraphError
    with pytest.raises(DualTSGraphError):
        build_teacher_supervision(
            action_delta=None, is_teacher_endpoint=False, n_atoms=4)


# ---------------------------------------------------------------------------
# Drift guard: runtime perception == offline preprocess fallback
# ---------------------------------------------------------------------------


def test_covalent_perception_matches_dual_ts_preprocess_fallback():
    """The runtime fallback must stay byte-compatible with the offline
    ``dual_ts_preprocess`` fallback (same radii, tolerance, rule)."""
    from pes2ts_core.pesn.features.dual_ts_graph import (
        COVALENT_BOND_TOL_ANGSTROM,
        covalent_radius_endpoint_graph,
    )
    from pes2ts_core.pesn.features.geometry import COVALENT_RADII
    from pes2ts_core.pesn.offline.data.dual_ts_preprocess import (
        COVALENT_BOND_TOL as OFFLINE_TOL,
        COVALENT_RADII as OFFLINE_RADII,
        endpoint_graph_from_geometry,
    )
    assert COVALENT_BOND_TOL_ANGSTROM == OFFLINE_TOL
    for z, r in OFFLINE_RADII.items():
        assert COVALENT_RADII[int(z)] == pytest.approx(r), f"Z={z} drifted"
    # Identical perception outcome on a toy geometry (C-H-C chain with one
    # bonded pair and one non-bonded pair).
    rng = np.random.default_rng(11)
    z = np.asarray([6, 1, 6, 1], dtype=int)
    coords = np.asarray([[0, 0, 0], [1.1, 0, 0], [4.0, 0, 0],
                         [4.0 + 0.9, 0.2, 0.1]], dtype=float)
    coords = coords + rng.normal(scale=1e-3, size=coords.shape)
    runtime = covalent_radius_endpoint_graph(z, coords)
    offline = endpoint_graph_from_geometry(z, coords)
    assert [(b.i, b.j, b.bond_order) for b in runtime.bonds] == \
        [(b.i, b.j, b.bond_order) for b in offline.bonds]


# ---------------------------------------------------------------------------
# Shared stub teacher: only the fields the topology builder reads
# ---------------------------------------------------------------------------


def _stub_teacher_for(task: DualEndpointTask) -> DualTSTeacher:
    """Minimal valid teacher whose consumed graph resolution returns the
    authority graphs (non-disputed eligibility is what the swap/permutation
    tests exercise)."""
    from pes2ts_core.pesn.contracts.dual_ts import (
        BranchRecord,
        EpisodeRecord,
        TeacherLabel,
        TeacherState,
    )
    rid = task.reaction_id
    z = np.asarray(task.endpoints[0].atomic_numbers, dtype=int)
    maps = tuple(task.endpoints[0].atom_map_ids)
    n = int(z.size)
    states = []
    labels = []
    episodes = []
    branches = []
    for label in ("A", "B"):
        branch_id = f"{rid}#br{label}"
        episode_id = f"{rid}#ep{label}0"
        start_state_id = f"{rid}|{label}|0000"
        branches.append(BranchRecord(
            branch_id=branch_id, endpoint_label=label, reaction_id=rid,
            episode_ids=(episode_id,)))
        episodes.append(EpisodeRecord(
            episode_id=episode_id, branch_id=branch_id, reaction_id=rid,
            start_state_id=start_state_id, budget_version="b",
            environment_version="e", reward_version="r"))
        coords = np.zeros((2, n, 3))
        for step in range(2):
            is_terminal = step == 1
            sid = f"{rid}|{label}|{step:04d}"
            states.append(TeacherState(
                state_id=sid, episode_id=episode_id, branch_id=branch_id,
                step_index=step, atomic_numbers=z.copy(),
                coordinates=coords[step], atom_map_ids=maps, method_id="M",
                observation=None,
                observation_missing_reason=(
                    "teacher_frame_isolated" if is_terminal
                    else "observation_not_exported"),
                boundary_type=("teacher_endpoint" if is_terminal else None),
                next_state_id=None if is_terminal
                else f"{rid}|{label}|{step + 1:04d}"))
            labels.append(TeacherLabel(
                state_id=sid,
                ts_displacement=np.zeros((n, 3)),
                remaining_arc_length=0.0,
                teacher_endpoint=is_terminal,
                action_delta=None if is_terminal else np.zeros((n, 3))))
    teacher = DualTSTeacher(
        reaction_id=rid, reaction_group_id=task.reaction_group_id,
        branches=tuple(branches), episodes=tuple(episodes),
        states=tuple(states), labels=tuple(labels))
    teacher.validate()
    return teacher
