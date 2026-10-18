"""WP2-A fit50 build tests — SYNTHETIC ONLY (no truth h5, no ACP/xTB).

Covers the frozen WP2 decisions through the core functions: the C5
diagonal-mass arc metric (hand-computed, masses from the v1 ``masses_for``
table), the metric-version binding gate, step-5 midpoint insertion with C1
discipline (renumbering, re-differencing, re-asserted identities, inserted
flags + gap entries), the weak-match bond cross-check policy (changed-bond
dispute flags the task but keeps the reaction; spectator mismatch notes
only), TS dedupe by computation identity (one node, two state refs),
coverage identity digests, honest cache matching, and the cost basis.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from pes2ts_core.pesn.contracts.dual_ts import (
    BOUNDARY_TEACHER_ENDPOINT,
    TASK_SCHEMA_VERSION,
    TEACHER_SCHEMA_VERSION,
    validate_record,
)
from pes2ts_core.pesn.offline.data.direct_preprocess import masses_for
from pes2ts_core.pesn.offline.data.dual_ts_fit50 import (
    ASSUMED_PER_QC_SECONDS,
    METHOD_GFN2_XTB,
    OPERATION_EG_SINGLE_POINT,
    atom_order_hash,
    build_cache_index,
    computation_identity,
    estimate_cost,
    geometry_content_digest,
    node_records_for_reaction,
)
from pes2ts_core.pesn.offline.data.dual_ts_preprocess import (
    A_MAX_ANGSTROM,
    ARC_LENGTH_METRIC_V1_DIAG_MASSES,
    ARC_LENGTH_METRIC_VERSION,  # WP1 placeholder (must stay untouched)
    DualTSPreprocessError,
    bond_cross_check,
    build_dual_ts_reaction,
    compute_branch_labels,
    endpoint_graph_from_bond_list,
    export_round,
    insert_midpoints_for_exceedance,
)

Z = (8, 1, 6, 1)  # O, H, C, H — all in the v1 mass table

R_BONDS = [(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)]  # map space, 1-based
P_BONDS = [(1, 2, 1.0), (2, 4, 2.0), (3, 4, 1.0)]


def graphs():
    z = np.asarray(Z, dtype=int)
    return (endpoint_graph_from_bond_list(z, R_BONDS),
            endpoint_graph_from_bond_list(z, P_BONDS))


def synthetic_ordered_path(step_scale: float = 0.2) -> np.ndarray:
    base = np.array([
        [[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [1.8, 0.0, 0.3], [2.7, 0.1, 0.0]],
        [[0.1, 0.0, 0.0], [1.0, 0.1, 0.1], [1.7, 0.1, 0.2], [2.6, 0.0, 0.1]],
        [[0.2, 0.1, 0.0], [1.3, 0.2, 0.0], [1.5, 0.0, 0.1], [2.4, 0.1, 0.2]],
        [[0.1, 0.0, 0.2], [1.6, 0.1, 0.0], [1.4, 0.1, 0.2], [2.5, 0.0, 0.0]],
        [[0.0, 0.1, 0.1], [1.9, 0.0, 0.1], [1.2, 0.2, 0.0], [2.6, 0.1, 0.1]],
    ], dtype=float)
    return base * step_scale


def build_outcome(path=None, ts_position: int | None = 2, **overrides):
    graph_r, graph_p = graphs()
    kwargs = dict(
        reaction_id="RXN_SYNTH0001",
        reaction_group_id="GRP_SYNTH0001",
        partition="fit",
        atomic_numbers=Z,
        ordered_path=synthetic_ordered_path() if path is None else path,
        authority_ts_position=ts_position,
        energies=np.array([0.0, 0.2, 1.0, 0.3, 0.05]),
        graph_r=graph_r,
        graph_p=graph_p,
        bond_perception="g1_p1_mapping_endpoint_bonds_v1",
        method_id=METHOD_GFN2_XTB,
        source="synthetic fixture",
    )
    kwargs.update(overrides)
    return build_dual_ts_reaction(**kwargs)


def wp2_kwargs(**extra):
    """Frozen WP2 build flags: C5 mass metric + insertion + cross-check."""
    kwargs = dict(
        arc_length_metric_version=ARC_LENGTH_METRIC_V1_DIAG_MASSES,
        ell_masses=masses_for(np.asarray(Z, dtype=int)),
        insert_on_exceedance=True,
        crosscheck_endpoint_bonds=True,
    )
    kwargs.update(extra)
    return kwargs


def states_of_branch(teacher, label: str):
    return [s for s in teacher.states if s.branch_id.endswith("br" + label)]


# ---------------------------------------------------------------------------
# C5 metric: diagonal-mass arc length (frozen formula, v1 mass table)
# ---------------------------------------------------------------------------


def test_c5_diag_mass_metric_hand_computed_two_atom_case():
    # masses from the SAME v1 table register_path uses
    m = masses_for(np.array([8, 1]))          # [15.999, 1.008]
    assert m.tolist() == [15.999, 1.008]
    coords = np.array([
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.3, 0.0, 0.0], [0.0, 0.4, 0.0]],
    ])
    actions, d, ell = compute_branch_labels(coords, masses=m)
    # hand value: sqrt(m_O * 0.3^2 + m_H * 0.4^2) = sqrt(1.60119)
    expected = math.sqrt(15.999 * 0.09 + 1.008 * 0.16)
    assert ell.tolist() == pytest.approx([expected, 0.0], rel=1e-12)
    # unit-mass differs correctly: plain norm would be 0.5
    _, _, ell_unit = compute_branch_labels(coords)
    assert ell_unit[0] == pytest.approx(0.5, rel=1e-12)
    assert abs(float(ell[0]) - 0.5) > 0.7
    # identity checks still hold under the mass metric
    assert float(np.max(np.abs(d[0] - coords[0] * -1.0))) >= 0.0
    assert ell[0] > ell[1] == 0.0


def test_c5_metric_version_binding_gate():
    masses = masses_for(np.asarray(Z, dtype=int))
    # masses without the frozen version string -> typed refusal
    with pytest.raises(DualTSPreprocessError) as excinfo:
        build_outcome(ell_masses=masses)
    assert excinfo.value.code == "METRIC_VERSION_MISMATCH"
    # frozen version string without masses -> typed refusal
    with pytest.raises(DualTSPreprocessError) as excinfo:
        build_outcome(arc_length_metric_version=ARC_LENGTH_METRIC_V1_DIAG_MASSES)
    assert excinfo.value.code == "METRIC_VERSION_MISMATCH"
    # WP1 default (placeholder + unit mass) must keep working untouched
    outcome = build_outcome()
    assert outcome.teacher.arc_length_metric_version == \
        ARC_LENGTH_METRIC_VERSION
    assert ARC_LENGTH_METRIC_VERSION.endswith("PLACEHOLDER")


def test_c5_mass_metric_ell_recomputed_from_exported_payload_both_branches():
    outcome = build_outcome(**wp2_kwargs())
    assert outcome.status == "ok"
    teacher = outcome.teacher
    assert teacher.arc_length_metric_version == \
        ARC_LENGTH_METRIC_V1_DIAG_MASSES
    m = masses_for(np.asarray(Z, dtype=int))
    payload = outcome.teacher_payload
    labels = {lab["state_id"]: lab for lab in payload["labels"]}
    for label in ("A", "B"):
        states = sorted(states_of_branch(teacher, label),
                        key=lambda s: s.step_index)
        coords = np.asarray([s.coordinates for s in states])  # exported round-10
        steps = coords[1:] - coords[:-1]
        per_step = np.sqrt((m[None, :, None] * steps * steps).sum(axis=(1, 2)))
        ell_expect = np.concatenate([np.cumsum(per_step[::-1])[::-1], [0.0]])
        ell_stored = [labels[s.state_id]["remaining_arc_length"]
                      for s in states]
        assert np.allclose(ell_stored, ell_expect, atol=1e-9)
        # monotone non-increase, zero at the shared TS terminal
        assert all(ell_stored[i] >= ell_stored[i + 1] - 1e-12
                   for i in range(len(ell_stored) - 1))
        assert ell_stored[-1] == 0.0


# ---------------------------------------------------------------------------
# Step-5 insertion (C1 discipline)
# ---------------------------------------------------------------------------


def test_insertion_pure_function_single_round_exact_midpoints():
    coords = np.array([
        [[0.0, 0.0, 0.0]],
        [[0.15, 0.0, 0.0]],
        [[0.30, 0.0, 0.0]],
    ])  # both per-atom steps 0.15 > a_max=0.1, halves <= 0.1
    new, origins, records = insert_midpoints_for_exceedance(
        coords, A_MAX_ANGSTROM)
    assert new.shape == (5, 1, 3)
    assert np.allclose(new[:, 0, 0],
                       [0.0, 0.075, 0.15, 0.225, 0.30], atol=1e-15)
    assert [o["kind"] for o in origins] == [
        "original", "inserted", "original", "inserted", "original"]
    assert [r["inserted_node_index"] for r in records] == [1, 3]
    assert [r["between_original_steps"] for r in records] == [[0, 1], [1, 2]]
    # no step exceeds any more
    assert float(np.linalg.norm(new[1:] - new[:-1], axis=2).max()) \
        <= A_MAX_ANGSTROM


def test_insertion_pure_function_multi_round_halving():
    coords = np.array([
        [[0.0, 0.0, 0.0]],
        [[0.30, 0.0, 0.0]],
        [[0.60, 0.0, 0.0]],
    ])  # 0.3 Å steps need repeated halving (no clipping, ever)
    new, origins, records = insert_midpoints_for_exceedance(
        coords, A_MAX_ANGSTROM)
    assert float(np.linalg.norm(new[1:] - new[:-1], axis=2).max()) \
        <= A_MAX_ANGSTROM
    assert len(new) == 3 + 6  # 0.3 Å segments need dyadic refinement
    assert len(records) == 6
    assert [r["between_original_steps"] for r in records] == \
        [[0, 1]] * 3 + [[1, 2]] * 3
    # endpoints preserved exactly (successors never dropped)
    assert np.array_equal(new[0], coords[0]) and np.array_equal(new[-1],
                                                                coords[-1])


def test_build_insertion_on_exceedance_full_discipline():
    path = synthetic_ordered_path()
    path = path.copy()
    path[1, 2, 0] += 0.12  # one non-rigid spike; registration smears it
    outcome = build_outcome(path=path, **wp2_kwargs())
    assert outcome.status == "ok"
    teacher = outcome.teacher
    items = outcome.gaps["items"]

    ins_a = len(items["inserted_nodes"]["A"])
    ins_b = len(items["inserted_nodes"]["B"])
    assert ins_a + ins_b >= 2  # both smeared steps got midpoints
    counts = items["branch_step_counts"]
    assert counts["A"] == 3 + ins_a and counts["B"] == 3 + ins_b

    # inserted:true flags + per-insert entries with resolvable state ids
    for side, ins in (("A", ins_a), ("B", ins_b)):
        exceed = items["a_max_exceedances"][side]
        if ins:
            assert exceed["inserted"] is True
            assert exceed["insertion_count"] == ins
            assert exceed["post_insertion_count"] == 0
        else:
            assert exceed["inserted"] is False
    all_ids = [e["inserted_state_id"] for side in ("A", "B")
               for e in items["inserted_nodes"][side]]
    state_ids = {s.state_id for s in teacher.states}
    assert set(all_ids) <= state_ids
    assert len(all_ids) == ins_a + ins_b
    assert sorted(outcome.diagnostics["inserted_state_ids"]["A"] +
                  outcome.diagnostics["inserted_state_ids"]["B"]) == \
        sorted(all_ids)

    # renumbered step_index + re-differenced labels re-asserted at 1e-8
    labels = {lab.state_id: lab for lab in teacher.labels}
    for label in ("A", "B"):
        states = states_of_branch(teacher, label)
        assert [s.step_index for s in states] == list(range(len(states)))
        for state, nxt in zip(states[:-1], states[1:]):
            a = np.asarray(labels[state.state_id].action_delta)
            err_a = float(np.max(np.abs(
                np.asarray(state.coordinates) + a
                - np.asarray(nxt.coordinates))))
            err_d = float(np.max(np.abs(
                np.asarray(labels[state.state_id].ts_displacement)
                - np.asarray(labels[nxt.state_id].ts_displacement) - a)))
            assert err_a <= 1e-8 and err_d <= 1e-8
            assert state.next_state_id == nxt.state_id
        # the per-atom a_max bound holds everywhere after insertion
        coords = np.asarray([s.coordinates for s in states])
        per_atom = np.linalg.norm(coords[1:] - coords[:-1], axis=2)
        assert float(per_atom.max()) <= A_MAX_ANGSTROM + 1e-9

    # TS is still the shared bit-identical terminal of both branches
    ts_a = states_of_branch(teacher, "A")[-1]
    ts_b = states_of_branch(teacher, "B")[-1]
    assert np.array_equal(ts_a.coordinates, ts_b.coordinates)
    assert ts_a.boundary_type == ts_b.boundary_type == \
        BOUNDARY_TEACHER_ENDPOINT

    validate_record(TASK_SCHEMA_VERSION, outcome.task_payload)
    validate_record(TEACHER_SCHEMA_VERSION, outcome.teacher_payload)


# ---------------------------------------------------------------------------
# Weak-match bond cross-check policy
# ---------------------------------------------------------------------------


def cc_pair_graphs(r_bonds, p_bonds):
    z = np.array([6, 6])
    return (z, endpoint_graph_from_bond_list(z, r_bonds),
            endpoint_graph_from_bond_list(z, p_bonds))


def cc_path(r_distance: float, p_distance: float) -> np.ndarray:
    return np.array([
        [[0.0, 0.0, 0.0], [r_distance, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.5 * (r_distance + p_distance), 0.05, 0.0]],
        [[0.0, 0.0, 0.0], [p_distance, 0.0, 0.0]],
    ])


def test_bond_crosscheck_changed_bond_dispute_flags_but_keeps_reaction():
    # authority says the C-C bond exists at R and is gone at P (changed
    # bond), but the R endpoint frame is 2.5 Å apart -> perception disagrees
    # on the CHANGED bond -> disputed; the reaction STAYS in the build.
    z, gr, gp = cc_pair_graphs([(1, 2, 1.0)], [])
    outcome = build_dual_ts_reaction(
        reaction_id="RXN_SYNTHDISP", reaction_group_id="GRP_SYNTHDISP",
        partition="fit", atomic_numbers=z, ordered_path=cc_path(2.5, 3.0),
        authority_ts_position=1, energies=np.array([0.0, 1.0, 0.1]),
        graph_r=gr, graph_p=gp, method_id=METHOD_GFN2_XTB,
        **wp2_kwargs())
    # reaction remains in the denominator: full honest build, status ok
    assert outcome.status == "ok"
    assert outcome.task.eligibility["bond_authority"] == "disputed"
    cross = outcome.gaps["items"]["bond_cross_check"]
    assert cross["outcome"] == "disputed"
    assert cross["disputed_bonds"] == [{"side": "R", "i": 0, "j": 1}]
    assert outcome.gaps["items"]["mapping_problems"]
    assert any("bond_authority_disputed" in p
               for p in outcome.gaps["items"]["mapping_problems"])
    assert outcome.diagnostics["bond_cross_check_outcome"] == "disputed"
    validate_record(TASK_SCHEMA_VERSION, outcome.task_payload)
    validate_record(TEACHER_SCHEMA_VERSION, outcome.teacher_payload)


def test_bond_crosscheck_spectator_mismatch_is_note_only():
    # the C-C bond is present at BOTH endpoints per authority (spectator,
    # never changed), but geometry perceives no bond at either endpoint ->
    # spectator-only disagreement -> accept-with-note, no mapping problem.
    z, gr, gp = cc_pair_graphs([(1, 2, 1.0)], [(1, 2, 1.0)])
    outcome = build_dual_ts_reaction(
        reaction_id="RXN_SYNTHSPEC", reaction_group_id="GRP_SYNTHSPEC",
        partition="fit", atomic_numbers=z, ordered_path=cc_path(2.5, 2.6),
        authority_ts_position=1, energies=np.array([0.0, 1.0, 0.1]),
        graph_r=gr, graph_p=gp, method_id=METHOD_GFN2_XTB,
        **wp2_kwargs())
    assert outcome.status == "ok"
    assert outcome.task.eligibility["bond_authority"] == "accepted_with_note"
    cross = outcome.gaps["items"]["bond_cross_check"]
    assert cross["outcome"] == "accept_with_note"
    assert cross["disputed_bonds"] == []
    assert len(cross["spectator_disagreements"]) == 2  # R and P endpoints
    assert outcome.gaps["items"]["mapping_problems"] == []
    assert any("spectator" in note for note in cross["notes"])


def test_bond_crosscheck_geometry_verified():
    z, gr, gp = cc_pair_graphs([(1, 2, 1.0)], [(1, 2, 1.0)])
    cross = bond_cross_check(z, gr, gp,
                             cc_path(1.5, 1.5)[0], cc_path(1.5, 1.5)[-1])
    assert cross["outcome"] == "geometry_verified"
    assert cross["disputed_bonds"] == []
    assert cross["spectator_disagreements"] == []
    assert cross["tolerance_angstrom"] == 0.45


def test_bond_crosscheck_unavailable_radius_is_recorded():
    z = np.array([6, 95])  # Z=95 has no Cordero radius registered
    gr = endpoint_graph_from_bond_list(z, [(1, 2, 1.0)])
    gp = endpoint_graph_from_bond_list(z, [(1, 2, 1.0)])
    coords = np.array([[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]])
    cross = bond_cross_check(z, gr, gp, coords[0], coords[0])
    assert cross["outcome"] == "crosscheck_unavailable"
    assert cross["notes"]


# ---------------------------------------------------------------------------
# TS dedupe + coverage identity
# ---------------------------------------------------------------------------


def test_ts_dedupe_one_node_two_state_refs():
    outcome = build_outcome(**wp2_kwargs())
    nodes = node_records_for_reaction(outcome, method=METHOD_GFN2_XTB,
                                      charge=0, multiplicity=1)
    teacher = outcome.teacher
    assert len(nodes) == len(teacher.states) - 1  # TS merged into one node
    ts_nodes = [n for n in nodes if n["is_ts"]]
    assert len(ts_nodes) == 1
    ts_node = ts_nodes[0]
    assert len(ts_node["state_refs"]) == 2
    terminals = {s.state_id for s in teacher.states
                 if s.boundary_type == BOUNDARY_TEACHER_ENDPOINT}
    assert {r["state_id"] for r in ts_node["state_refs"]} == terminals
    assert {r["branch_id"] for r in ts_node["state_refs"]} == \
        {b.branch_id for b in teacher.branches}
    # every other node carries exactly one state ref; identities unique
    assert all(len(n["state_refs"]) == 1 for n in nodes if not n["is_ts"])
    assert len({n["computation_identity"] for n in nodes}) == len(nodes)
    assert all(n["cache"]["status"] == "to_compute" for n in nodes)


def test_coverage_identity_digests():
    coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    d1 = geometry_content_digest(coords)
    assert d1 == geometry_content_digest(coords.copy())
    # sub-rounding perturbation is absorbed by the declared canonicalization
    bumped = coords.copy()
    bumped[1, 0] += 1e-11
    assert d1 == geometry_content_digest(bumped)
    # real geometry change -> different digest
    moved = coords.copy()
    moved[1, 0] += 1e-6
    assert d1 != geometry_content_digest(moved)
    # atom order hash: mapping + Z sequence (canonical default == G1 m0001..)
    h1 = atom_order_hash(np.array([6, 6]))
    assert h1 == atom_order_hash(np.array([6, 6]))
    assert h1 == atom_order_hash(np.array([6, 6]), ("m0001", "m0002"))
    assert h1 != atom_order_hash(np.array([6, 6]), ("m0002", "m0001"))
    assert h1 != atom_order_hash(np.array([7, 6]))
    # full identity: same inputs -> same digest; charge change -> different
    base = computation_identity(d1, h1, METHOD_GFN2_XTB, 0, 1,
                               OPERATION_EG_SINGLE_POINT)
    same = computation_identity(d1, h1, METHOD_GFN2_XTB, 0, 1,
                                OPERATION_EG_SINGLE_POINT)
    charged = computation_identity(d1, h1, METHOD_GFN2_XTB, 1, 1,
                                   OPERATION_EG_SINGLE_POINT)
    other_op = computation_identity(d1, h1, "GFN1-xTB", 0, 1,
                                    OPERATION_EG_SINGLE_POINT)
    assert base["digest"] == same["digest"] and base["digest"] != charged[
        "digest"] and base["digest"] != other_op["digest"]


# ---------------------------------------------------------------------------
# Cache-match honesty (synthetic v1-style receipts, no network)
# ---------------------------------------------------------------------------


def _write_eval(base, eval_id, request, result=None):
    d = base / "RXN_TEST0001" / "direct-RXN_TEST0001" / f"eval-{eval_id}"
    d.mkdir(parents=True)
    (d / "request.json").write_text(json.dumps(request))
    if result is not None:
        (d / "result.json").write_text(json.dumps(result))
    return d


def test_cache_match_honesty_and_hit(tmp_path):
    outcome = build_outcome(**wp2_kwargs())
    states = sorted(outcome.teacher.states,
                    key=lambda s: (s.branch_id, s.step_index))
    elements = ["O", "H", "C", "H"]

    def request_for(coords, method=METHOD_GFN2_XTB, with_geometry=True):
        req = {
            "method": method, "charge": 0, "multiplicity": 1,
            "elements": elements,
        }
        if with_geometry:
            req["geometry"] = np.asarray(coords, dtype=float).tolist()
        return req

    result_ok = {"success": True, "duration_seconds": 4.2}
    # hit: geometry content bound + successful receipt
    _write_eval(tmp_path, "aaa",
                request_for(states[0].coordinates), result_ok)
    # honest miss #1: no geometry content (paths only) -> never matched
    _write_eval(tmp_path, "bbb",
                request_for(states[1].coordinates, with_geometry=False),
                result_ok)
    # honest miss #2: geometry bound but no result -> not a receipt
    _write_eval(tmp_path, "ccc", request_for(states[2].coordinates))
    # honest miss #3: failed computation -> not a receipt
    _write_eval(tmp_path, "ddd", request_for(states[3].coordinates),
                {"success": False, "failure_class": "driver"})
    # honest miss #4: different method -> different computation identity
    _write_eval(tmp_path, "eee",
                request_for(states[4].coordinates, method="GFN1-xTB"),
                result_ok)

    index, audit = build_cache_index(tmp_path)
    assert audit["requests_total"] == 5
    assert audit["requests_without_geometry_binding"] == 1
    assert audit["requests_without_result"] == 1
    assert audit["unsuccessful_results"] == 1
    assert audit["successful_results"] == 2  # aaa (GFN2) + eee (GFN1)
    assert audit["methods_seen"] == {"GFN2-xTB": 3, "GFN1-xTB": 1}
    assert audit["durations_seconds"] == [4.2, 4.2]
    assert len(index) == 2  # two distinct identities indexed

    nodes = node_records_for_reaction(outcome, method=METHOD_GFN2_XTB,
                                      charge=0, multiplicity=1,
                                      cache_index=index)
    by_state = {}
    for node in nodes:
        for ref in node["state_refs"]:
            by_state[ref["state_id"]] = node
    hit = by_state[states[0].state_id]
    assert hit["cache"]["status"] == "cache_hit"
    assert hit["cache"]["receipt_path"].endswith("eval-aaa")
    assert hit["cache"]["duration_seconds"] == 4.2
    # states 1..4 stay honestly to_compute (unresolvable or mismatched)
    for i in (1, 2, 3, 4):
        node = by_state[states[i].state_id]
        assert node["cache"]["status"] == "to_compute"
        assert node["cache"]["receipt_path"] is None
    # a node record built with the WRONG charge must not hit either
    nodes_q1 = node_records_for_reaction(outcome, method=METHOD_GFN2_XTB,
                                         charge=1, multiplicity=1,
                                         cache_index=index)
    assert all(n["cache"]["status"] == "to_compute" for n in nodes_q1)


def test_cost_estimate_basis_measured_vs_assumed():
    measured = estimate_cost(10, [2.0, 4.0, 9.0])
    assert measured["per_qc_seconds"] == 4.0  # median
    assert measured["per_qc_is_assumed"] is False
    assert "measured" in measured["per_qc_basis"]
    assert measured["qc_seconds_total"] == pytest.approx(40.0)
    assert measured["estimated_wall_seconds"] == pytest.approx(20.0)  # 2 workers
    assumed = estimate_cost(10, [])
    assert assumed["per_qc_seconds"] == ASSUMED_PER_QC_SECONDS
    assert assumed["per_qc_is_assumed"] is True
    assert "assumed" in assumed["per_qc_basis"]
    assert assumed["estimated_wall_hours"] == pytest.approx(
        10 * ASSUMED_PER_QC_SECONDS / 2 / 3600.0, abs=1e-3)


# ---------------------------------------------------------------------------
# WP4-A2 additive --groups resolver (monitor-wave teacher build)
# ---------------------------------------------------------------------------

def _lineage_for_groups(tmp_path: Path):
    from tests.pesn.test_dual_ts_split import build_v1_tree
    from pes2ts_core.pesn.offline.splits.dual_ts_split import (
        build_dual_ts_split_lineage,
        DualTSSplitLineage,
    )
    base = build_v1_tree(tmp_path, n_fit=6, n_monitor=3, n_dev=2,
                         n_sealed=1, n_isolated=1, duplicate_pair=False)
    return DualTSSplitLineage.from_payload(
        build_dual_ts_split_lineage(base))


def test_groups_resolver_default_is_full_fit_wave(tmp_path):
    from scripts.prepare_dual_ts_fit50 import resolve_group_queue
    lineage = _lineage_for_groups(tmp_path)
    queue, mode = resolve_group_queue(lineage, None)
    assert mode == "fit_wave"
    assert queue == lineage.wave_groups("fit")
    assert all(lineage.partition_of(g) == "fit" for g in queue)


def test_groups_resolver_accepts_only_the_complete_monitor_wave(tmp_path):
    from scripts.prepare_dual_ts_fit50 import resolve_group_queue
    lineage = _lineage_for_groups(tmp_path)
    monitor = lineage.wave_groups("monitor")
    queue, mode = resolve_group_queue(lineage, ",".join(monitor))
    assert mode == "monitor_wave"
    assert queue == sorted(monitor)
    assert all(lineage.partition_of(g) == "monitor" for g in queue)
    # fit ∩ monitor = ∅ asserted by construction of the resolver
    assert not (set(queue) & set(lineage.wave_groups("fit")))


def test_groups_resolver_refuses_subset_and_unknown_and_fit_members(tmp_path):
    from scripts.prepare_dual_ts_fit50 import resolve_group_queue
    lineage = _lineage_for_groups(tmp_path)
    monitor = lineage.wave_groups("monitor")
    # subset of the monitor wave → refuse (must be the COMPLETE wave)
    with pytest.raises(SystemExit):
        resolve_group_queue(lineage, ",".join(monitor[:2]))
    # unknown / non-monitor member → refuse
    with pytest.raises(SystemExit):
        resolve_group_queue(lineage, ",".join(monitor) + ",RXN_9999999999")
    # a fit-wave member → refuse (fit ∩ monitor must stay empty)
    fit_member = lineage.wave_groups("fit")[0]
    with pytest.raises(SystemExit):
        resolve_group_queue(lineage, ",".join(monitor) + f",{fit_member}")
    # duplicates → refuse
    with pytest.raises(SystemExit):
        resolve_group_queue(lineage, ",".join(monitor) + f",{monitor[0]}")
    # empty → refuse
    with pytest.raises(SystemExit):
        resolve_group_queue(lineage, " , ")
