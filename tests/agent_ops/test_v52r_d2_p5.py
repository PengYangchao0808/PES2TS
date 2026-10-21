"""Unit tests for the V5.2-R D2 phase 2 (P5 basin supervision) — SYNTHETIC.

Covers the basin window + edit-gate semantics (pinned three-state rule), the
D0.1 T-row → negative assertion, B_target membership (symmetry images in,
mirrors out), conformer candidate construction (rotatable-bond rules,
same-seed determinism, rigid kicks, clash filter), the xtb budget/abort logic
with a MOCKED runner (real-xtb path tagged ``xtb`` and deselected by
pytest.ini), seed-flip accounting, intersection-population pairing, and the
p5_spec pre-registration hash stability — no dataset loads, no ``outputs/``
reads (except nothing), no network.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pes2ts_core.pesn.offline.data.basin_labels_v1 import (  # noqa: E402
    BasinLabelTable,
    BasinSampleLabel,
    BasinTarget,
    CONFORMER_SEED_FLIP,
    CONFORMER_SEED_PRIMARY,
    EDIT_TOLERANCE_ANGSTROM,
    as_endpoint_stop_labels,
    basin_label_from_distance,
    changing_bond_atoms,
    check_t_rows,
    cv_pair_deviations,
    d_basin,
    edit_gate_complete,
    flip_stats,
    generate_conformer,
    has_clash,
    population_counts,
    restrict_to,
    rotatable_bonds,
    with_conformers,
)
from pes2ts_core.pesn.offline.data.graph_symmetry import (  # noqa: E402
    MolecularGraph,
)
from pes2ts_core.pesn.offline.data.key_atom_endpoint_labels_v1 import (  # noqa: E402
    CurveEndpointStopSummary,
    EndpointStopLabelTable,
    EndpointStopSampleLabel,
)
from scripts.agent_ops.data_gate.v52r_d2_p5_probe import (  # noqa: E402
    ENERGY_WINDOW_HARTREE,
    MAX_XTB_CALLS,
    MAX_XTB_WALL_S,
    ConformerJob,
    XtbBudget,
    _parse_energy,
    build_p5_spec,
    enrich_conformers,
    paired_curve_bootstrap_two_pops,
    run_reference_sps,
    spec_sha256,
    unit_deltas,
    validity_energy_window,
    validity_positive_rate,
    validity_reference_in_basin,
    validity_seed_flip,
)
from scripts.agent_ops.data_gate.v52r_d2_probe import (  # noqa: E402
    paired_cluster_bootstrap_auprc,
)


# --------------------------------------------------------------- fixtures


def _basin_sample(sid: str = "s1", label: int = 0, d: float = 0.05,
                  complete: bool = False, degenerate: bool = False,
                  curve: str = "C1") -> BasinSampleLabel:
    return BasinSampleLabel(
        sample_id=sid, curve_id=curve, node_index=1, intent_id="i1",
        direction="forward", target_endpoint="B", target_endpoint_node=5,
        d_basin=d, edit_complete=complete, basin_label=label,
        degenerate_flag=degenerate, move_status="ok")


def _chain_job(curve_id: str = "CH4x") -> ConformerJob:
    """4-carbon chain: bonds (0,1),(1,2),(2,3) all rotatable."""
    graph = MolecularGraph.from_intent_payload(
        atoms=(6, 6, 6, 6),
        bonds=((0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)))
    ref = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0],
                    [3.0, 0.5, 0.0], [4.5, 0.0, 0.5]])
    return ConformerJob(curve_id=curve_id, ref_frame=ref,
                        atomic=(6, 6, 6, 6), k_rows=(0, 3),
                        rot_bonds=rotatable_bonds(graph, ()),
                        graph=graph, symbols=("C", "C", "C", "C"))


def _mock_runner(budget: XtbBudget, *, wall_s: float = 0.5,
                 energy: float = -100.001, ok: bool = True):
    """Fake xtb runner honoring the budget protocol (no subprocess)."""
    def runner(coords, *, symbols, freeze_atoms=(), mode="sp", tag=None):
        assert tag is not None and "purpose" in tag
        if not budget.can_call():
            return {"ok": False, "converged": False, "energy_ha": None,
                    "wall_seconds": 0.0, "exit_code": None, "coords": None,
                    "error": "budget exhausted",
                    "budget_exhausted": budget.exhausted_reason() or "exhausted"}
        res = {"ok": ok, "converged": ok,
               "energy_ha": (energy if mode == "sp" else energy),
               "wall_seconds": wall_s, "exit_code": 0 if ok else 1,
               "coords": (None if not ok
                          else np.asarray(coords, dtype=np.float64)),
               "error": None if ok else "mock failure"}
        entry = budget.record(tag, res)
        res["log_entry"] = entry
        res["budget_exhausted"] = False
        return res
    return runner


# ------------------------------------------------------------------ tests


class TestBasinWindowSemantics:
    def test_three_state_truth_table(self):
        # complete + close → positive
        assert basin_label_from_distance(0.05, True) == 1
        assert basin_label_from_distance(0.10, True) == 1          # inclusive
        # incomplete + close → NEGATIVE (T-rows land here)
        assert basin_label_from_distance(0.05, False) == 0
        assert basin_label_from_distance(0.10, False) == 0
        # middle band → mask regardless of completeness (pinned literal rule)
        assert basin_label_from_distance(0.10 + 1e-9, True) == -1
        assert basin_label_from_distance(0.30, True) == -1
        assert basin_label_from_distance(0.20, False) == -1
        # far → negative
        assert basin_label_from_distance(0.30 + 1e-9, False) == 0
        assert basin_label_from_distance(1.2, True) == 0

    def test_edit_gate_boundary(self):
        frame = np.array([[0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]])
        target = frame.copy()
        assert edit_gate_complete(frame, target, [(0, 1)]) is True
        # atom 2 shifted along x by 0.75 → d(0,2): 1.00 → 1.25 (exactly 0.25)
        target2 = frame.copy()
        target2[2, 0] += 0.75
        assert cv_pair_deviations(frame, target2, [(0, 2)])[0] == \
            pytest.approx(EDIT_TOLERANCE_ANGSTROM)
        assert edit_gate_complete(frame, target2, [(0, 2)]) is True
        target3 = frame.copy()
        target3[2, 0] += 0.751
        assert edit_gate_complete(frame, target3, [(0, 2)]) is False
        # one deviating pair closes the gate even if others are fine
        assert edit_gate_complete(frame, target3, [(0, 1), (0, 2)]) is False
        # vacuous completeness without CV pairs
        assert edit_gate_complete(frame, target3, []) is True

    def test_cv_pair_deviations_values(self):
        frame = np.array([[0.0, 0, 0], [2.0, 0, 0], [0, 3.0, 0]])
        target = frame.copy()
        target[1, 0] = 1.5
        devs = cv_pair_deviations(frame, target, [(0, 1), (0, 2)])
        assert devs == pytest.approx([0.5, 0.0])


class TestTRowAssertion:
    def test_t_row_like_row_is_negative(self):
        # D0.1 T-row: d_end ≤ 0.10 with an incomplete edit → basin 0
        lab = _basin_sample(label=basin_label_from_distance(0.093, False),
                            d=0.093, complete=False)
        assert lab.basin_label == 0
        table = BasinLabelTable(by_sample={lab.sample_id: lab}, curves={},
                                positive_window=0.1, negative_window=0.3)
        out = check_t_rows([lab.sample_id], table)
        assert out["pass"] is True
        assert out["n_negative"] == 1 and out["n_positive"] == 0

    def test_positive_t_row_fails_assertion(self):
        bad = _basin_sample(sid="bad", label=1, d=0.05, complete=True)
        table = BasinLabelTable(by_sample={bad.sample_id: bad}, curves={},
                                positive_window=0.1, negative_window=0.3)
        out = check_t_rows([bad.sample_id], table)
        assert out["pass"] is False and out["n_positive"] == 1

    def test_missing_row_fails_assertion(self):
        table = BasinLabelTable(by_sample={}, curves={},
                                positive_window=0.1, negative_window=0.3)
        out = check_t_rows(["ghost"], table)
        assert out["pass"] is False and out["missing"] == ["ghost"]


class TestBasinTargetMembership:
    @staticmethod
    def _chiral_target() -> tuple[BasinTarget, np.ndarray]:
        # four DISTINCT elements → automorphism group trivial (identity only)
        ref = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0],
                        [0.7, 1.3, 0.0], [0.6, 0.5, 1.2]])
        tgt = BasinTarget(curve_id="CX", direction="forward",
                          target_endpoint_node=0, k_rows=(0, 1, 2, 3),
                          ref_frame=ref, frames=(ref,),
                          provenance=("reference",),
                          perms=((0, 1, 2, 3),))
        return tgt, ref

    def test_reference_in_basin_zero(self):
        tgt, ref = self._chiral_target()
        assert d_basin(ref, tgt) == pytest.approx(0.0, abs=1e-12)

    def test_rotation_invariant_zero(self):
        tgt, ref = self._chiral_target()
        theta = 0.7
        rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1.0]])
        rotated = (rot @ ref.T).T + np.array([3.0, -2.0, 1.0])
        assert d_basin(rotated, tgt) == pytest.approx(0.0, abs=1e-9)

    def test_mirror_outside_basin(self):
        tgt, ref = self._chiral_target()
        mirrored = ref * np.array([-1.0, 1.0, 1.0])
        assert d_basin(mirrored, tgt) > 0.3

    def test_conformer_joins_basin(self):
        tgt, ref = self._chiral_target()
        conf = ref.copy()
        conf[3] = conf[3] + np.array([1.6, 0.0, 0.0])   # genuine new geometry
        tgt2 = with_conformers(tgt, [conf])
        assert tgt2.n_conformers == 1
        assert tgt2.provenance == ("reference", "conformer_1")
        assert d_basin(conf, tgt2) == pytest.approx(0.0, abs=1e-12)
        # the conformer is NOT in the reference-only basin
        assert d_basin(conf, tgt) > 0.3


class TestConformerConstruction:
    def test_rotatable_bond_rules(self):
        graph = MolecularGraph.from_intent_payload(
            atoms=(6, 6, 8, 6, 6),
            bonds=((0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 2.0)))
        # single bonds with BOTH endpoints degree ≥ 2 survive; double bonds
        # and terminal-atom bonds never rotate
        assert rotatable_bonds(graph, ()) == ((1, 2), (2, 3))
        assert rotatable_bonds(graph, (1,)) == ((2, 3),)
        assert rotatable_bonds(graph, (2,)) == ()        # both touch atom 2
        # terminal atom bonds (degree 1) never rotatable
        graph2 = MolecularGraph.from_intent_payload(
            atoms=(6, 1, 6, 6),
            bonds=((0, 1, 1.0), (0, 2, 1.0), (2, 3, 1.0)))
        assert rotatable_bonds(graph2, ()) == ((0, 2),)

    def test_changing_bond_atoms(self):
        class _Intent:
            def pair_orders(self):
                return {(0, 1): (1.0, 1.0), (1, 2): (1.0, 0.0),
                        (2, 3): (2.0, 2.0)}
        assert changing_bond_atoms(_Intent()) == frozenset({1, 2})

    def test_kick_determinism_and_rigidity(self):
        job = _chain_job()
        rng_a = np.random.RandomState(20261021)
        rng_b = np.random.RandomState(20261021)
        c1 = generate_conformer(job.ref_frame, job.rot_bonds, job.graph,
                                job.atomic, rng_a)
        c2 = generate_conformer(job.ref_frame, job.rot_bonds, job.graph,
                                job.atomic, rng_b)
        assert np.array_equal(c1, c2)                     # same seed
        rng_c = np.random.RandomState(7)
        c3 = generate_conformer(job.ref_frame, job.rot_bonds, job.graph,
                                job.atomic, rng_c)
        assert not np.allclose(c1, c3)                    # different seed
        # rigid rotations preserve every pairwise distance within each side
        d_ref = np.linalg.norm(job.ref_frame[:, None] - job.ref_frame[None, :],
                               axis=-1)
        d_new = np.linalg.norm(c1[:, None] - c1[None, :], axis=-1)
        assert d_new[0, 1] == pytest.approx(d_ref[0, 1], abs=1e-9)  # cut bond
        assert d_new[2, 3] == pytest.approx(d_ref[2, 3], abs=1e-9)  # side rig.
        assert d_new[0, 3] != pytest.approx(d_ref[0, 3], abs=1e-3)  # moved

    def test_h_jitter_only_moves_h(self):
        ref = np.array([[0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]])
        atomic = (6, 1, 1)
        graph = MolecularGraph.from_intent_payload(
            atoms=atomic, bonds=((0, 1, 1.0), (0, 2, 1.0)))
        rng = np.random.RandomState(3)
        out = generate_conformer(ref, (), graph, atomic, rng)
        assert np.allclose(out[0], ref[0])                # heavy atom fixed
        assert not np.allclose(out[1], ref[1])            # H jittered

    def test_clash_filter(self):
        near = np.array([[0.0, 0, 0], [0.5, 0, 0]])
        far = np.array([[0.0, 0, 0], [1.5, 0, 0]])
        assert has_clash(near) is True
        assert has_clash(far) is False


class TestXtbBudget:
    def test_caps_and_reasons(self):
        b = XtbBudget(max_calls=2, max_seconds=60.0)
        assert b.can_call() and b.exhausted_reason() is None
        b.record({"purpose": "t"}, {"wall_seconds": 1.0})
        b.record({"purpose": "t"}, {"wall_seconds": 1.0})
        assert not b.can_call()
        assert b.exhausted_reason() == "call_cap"
        b2 = XtbBudget(max_calls=100, max_seconds=10.0)
        b2.record({"purpose": "t"}, {"wall_seconds": 9.8})
        assert not b2.can_call()                        # < headroom left
        assert b2.exhausted_reason() == "wall_cap"

    def test_defaults_are_pre_registered(self):
        assert MAX_XTB_CALLS == 50 and MAX_XTB_WALL_S == 600.0
        assert ENERGY_WINDOW_HARTREE == pytest.approx(0.00797)
        assert (CONFORMER_SEED_PRIMARY, CONFORMER_SEED_FLIP) == (20261021, 20261105)

    def test_parse_energy_last_match_wins(self):
        out = (" cycle        total energy    energy error\n"
               " TOTAL ENERGY     -76.123456789012 Eh\n"
               " TOTAL ENERGY     -76.234567890123 Eh\n")
        assert _parse_energy(out) == pytest.approx(-76.234567890123)
        assert _parse_energy("no energy here") is None


class TestEnrichmentWithMockRunner:
    def test_budget_cap_aborts_and_keeps_accepted(self):
        budget = XtbBudget(max_calls=4, max_seconds=1e6)
        runner = _mock_runner(budget)
        jobs = [_chain_job("CA"), _chain_job("CB")]
        accepted, abort = enrich_conformers(
            jobs, seed=1, runner=runner, sp_energies={"CA": -100.0,
                                                      "CB": -100.0})
        assert abort == "call_cap"
        assert budget.calls == 4
        assert len(accepted["CA"]) == 2 and len(accepted["CB"]) == 2
        assert all(e["purpose"] == "conformer_opt" for e in budget.log)
        assert all(e["accepted"] for e in budget.log)

    def test_energy_window_rejects(self):
        budget = XtbBudget(max_calls=30, max_seconds=1e6)
        # conformers 0.05 Ha away from the reference SP → all rejected
        runner = _mock_runner(budget, energy=-100.05)
        accepted, abort = enrich_conformers(
            [_chain_job("CA")], seed=1, runner=runner,
            sp_energies={"CA": -100.0})
        assert abort is None
        assert accepted["CA"] == []
        # attempts capped at 5, opts at 3 → exactly 3 xtb calls
        assert budget.calls == 3
        assert all(e["within_window"] is False for e in budget.log)

    def test_wall_cap_aborts(self):
        budget = XtbBudget(max_calls=100, max_seconds=10.0)
        runner = _mock_runner(budget, wall_s=4.0)
        accepted, abort = enrich_conformers(
            [_chain_job("CA")], seed=1, runner=runner,
            sp_energies={"CA": -100.0})
        assert abort == "wall_cap"
        assert budget.calls == 2 and budget.wall >= 8.0
        assert len(accepted["CA"]) == 2                  # kept what was built

    def test_reference_sp_phase_and_failure(self):
        budget = XtbBudget(max_calls=10, max_seconds=1e6)
        runner = _mock_runner(budget, energy=-50.0)
        jobs = [_chain_job("CA"), _chain_job("CB")]
        energies, abort = run_reference_sps(jobs, runner)
        assert abort is None and energies == {"CA": -50.0, "CB": -50.0}
        budget2 = XtbBudget(max_calls=10, max_seconds=1e6)
        runner2 = _mock_runner(budget2, ok=False)
        energies2, abort2 = run_reference_sps(jobs, runner2)
        assert energies2 == {} and abort2.startswith("reference_sp_failed")

    def test_determinism_per_seed(self):
        def _run(seed):
            budget = XtbBudget(max_calls=30, max_seconds=1e6)
            accepted, _ = enrich_conformers(
                [_chain_job("CA")], seed=seed, runner=_mock_runner(budget),
                sp_energies={"CA": -100.0})
            return [c["coords"].tolist() for c in accepted["CA"]]
        assert _run(11) == _run(11)
        assert _run(11) != _run(12)


class TestFlipAccounting:
    def test_flip_stats(self):
        a = {"s1": 1, "s2": 0, "s3": -1, "s4": 1}
        b = {"s1": 1, "s2": 1, "s3": -1, "s4": 0}
        out = flip_stats(a, b, ["s1", "s2", "s3", "s4"])
        assert out["n_rows"] == 4 and out["n_flips"] == 2
        assert out["flip_rate"] == pytest.approx(0.5)
        assert out["transitions"] == {"0->1": 1, "1->0": 1}

    def test_validity_gate_uses_limit(self):
        ok = validity_seed_flip({"n_rows": 100, "n_flips": 2,
                                 "flip_rate": 0.02, "transitions": {}})
        assert ok["pass"] is True
        bad = validity_seed_flip({"n_rows": 100, "n_flips": 7,
                                  "flip_rate": 0.07, "transitions": {}})
        assert bad["pass"] is False


class TestIntersectionPairing:
    @staticmethod
    def _stop_table():
        def row(sid, cid, label, deg=False):
            return EndpointStopSampleLabel(
                sample_id=sid, curve_id=cid, node_index=1, intent_id="i",
                direction="forward", target_endpoint="B",
                target_endpoint_node=5, d_end_angstrom=0.05,
                endpoint_stop_label=label, degenerate_flag=deg,
                move_status="ok")
        by_sample = {
            "r1": row("r1", "C1", 1), "r2": row("r2", "C1", 0),
            "r3": row("r3", "C2", -1), "r4": row("r4", "C2", 1, deg=True),
        }
        def summ(cid):
            return CurveEndpointStopSummary(
                curve_id=cid, endpoint_a_node=0, endpoint_b_node=5,
                ts_index=3, n_rows=2, n_pos=1, n_neg=1, n_mask=0,
                n_degenerate=0, d_end_min=0.0, d_end_median=0.05,
                d_end_max=0.1)
        return EndpointStopLabelTable(
            by_sample=by_sample, curves={"C1": summ("C1"), "C2": summ("C2")},
            positive_window=0.1, negative_window=0.3)

    def test_restrict_to_masks_outside_keep(self):
        table = self._stop_table()
        out = restrict_to(table, ["r1", "r2"])
        assert out.by_sample["r1"].endpoint_stop_label == 1
        assert out.by_sample["r2"].endpoint_stop_label == 0
        assert out.by_sample["r3"].endpoint_stop_label == -1
        assert out.by_sample["r4"].endpoint_stop_label == -1  # degenerate
        # usable population == keep set
        usable = [sid for sid, lab in out.by_sample.items()
                  if lab.endpoint_stop_label >= 0 and not lab.degenerate_flag]
        assert sorted(usable) == ["r1", "r2"]

    def test_as_endpoint_stop_labels_maps_basin(self):
        table = BasinLabelTable(
            by_sample={"s1": _basin_sample("s1", label=1, d=0.05,
                                           complete=True)},
            curves={}, positive_window=0.1, negative_window=0.3)
        out = as_endpoint_stop_labels(table)
        lab = out.by_sample["s1"]
        assert lab.endpoint_stop_label == 1
        assert lab.d_end_angstrom == pytest.approx(0.05)

    def test_unit_deltas_pairs_only_matched(self):
        ua = [
            {"fold": 0, "seed": 17, "probes": {"P5/hgb": {"auprc": 0.60}}},
            {"fold": 1, "seed": 17, "probes": {"P5/hgb": {"auprc": 0.40}}},
        ]
        ub = [
            {"fold": 0, "seed": 17, "probes": {"P0/hgb": {"auprc": 0.30}}},
            {"fold": 2, "seed": 17, "probes": {"P0/hgb": {"auprc": 0.99}}},
        ]
        out = unit_deltas(ua, "P5/hgb", ub, "P0/hgb")
        assert out["n_units"] == 1                    # only (0, 17) matched
        assert out["median"] == pytest.approx(0.30)

    def test_population_counts(self):
        table = BasinLabelTable(by_sample={
            "a": _basin_sample("a", label=1),
            "b": _basin_sample("b", label=0),
            "c": _basin_sample("c", label=-1),
            "d": _basin_sample("d", label=1, degenerate=True),
        }, curves={}, positive_window=0.1, negative_window=0.3)
        pop = population_counts(table, ["a", "b", "c", "d"])
        assert (pop["positive"], pop["negative"], pop["mask"]) == (2, 1, 1)
        assert pop["usable_rows"] == 2                  # degenerate d excluded
        assert pop["usable_positive_rate"] == pytest.approx(0.5)


class TestCrossSchemeBootstrap:
    def test_cross_scheme_deterministic(self):
        rng = np.random.RandomState(5)
        n = 300
        curves = np.array([f"C{i % 15}" for i in range(n)], dtype=object)
        y_a = rng.randint(0, 2, size=n)
        y_b = rng.randint(0, 2, size=n)
        s_a = y_a * 0.5 + rng.uniform(0, 0.4, size=n)
        s_b = y_b * 0.2 + rng.uniform(0, 0.4, size=n)
        out1 = paired_cluster_bootstrap_auprc(
            s_a, s_b, y_a, curves, labels_b=y_b, n_resamples=100, seed=0)
        out2 = paired_cluster_bootstrap_auprc(
            s_a, s_b, y_a, curves, labels_b=y_b, n_resamples=100, seed=0)
        assert out1 == out2
        assert out1["cross_scheme"] is True
        assert out1["point"] == pytest.approx(
            float(average_precision_score(y_a, s_a)
                  - average_precision_score(y_b, s_b)), abs=1e-6)
        assert out1["ci95"]["lower"] is not None

    def test_two_population_bootstrap(self):
        rng = np.random.RandomState(9)
        n = 200
        curves = np.array([f"C{i % 10}" for i in range(n)], dtype=object)
        y_a = rng.randint(0, 2, size=n)
        y_b = rng.randint(0, 2, size=n)
        s_a = y_a * 0.5 + rng.uniform(0, 0.4, size=n)
        s_b = y_b * 0.5 + rng.uniform(0, 0.4, size=n)
        out = paired_curve_bootstrap_two_pops(
            s_a, y_a, curves, s_b, y_b, curves, n_resamples=100, seed=0)
        assert out["two_populations"] is True
        assert out["ci95"]["lower"] <= out["point"] <= out["ci95"]["upper"]


class TestValidityHelpers:
    def test_reference_in_basin_all_zero(self):
        ref = np.array([[0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]])
        tgt = BasinTarget(curve_id="C", direction="forward",
                          target_endpoint_node=1, k_rows=(0, 1, 2),
                          ref_frame=ref, frames=(ref,),
                          provenance=("reference",),
                          perms=((0, 1, 2),))
        out = validity_reference_in_basin({("C", "forward"): tgt})
        assert out["pass"] is True
        assert out["max_deviation"] == pytest.approx(0.0, abs=1e-9)

    def test_energy_window_validity(self):
        sp = {"CA": -100.0, "CB": -100.0}
        acc = {"CA": [{"energy_ha": -100.001}], "CB": []}
        out = validity_energy_window(sp, acc, ["CA", "CB"])
        assert out["pass"] is True and out["n_accepted"] == 1
        acc2 = {"CA": [{"energy_ha": -100.5}], "CB": []}
        out2 = validity_energy_window(sp, acc2, ["CA", "CB"])
        assert out2["pass"] is False and out2["n_outside_window"] == 1
        out3 = validity_energy_window({}, {}, ["CA", "CB"])
        assert out3["pass"] is False                   # missing SPs

    def test_positive_rate_band(self):
        assert validity_positive_rate(
            {"usable_positive": 20, "n_rows": 100,
             "usable_positive_rate": 0.2})["pass"] is True
        assert validity_positive_rate(
            {"usable_positive": 5, "n_rows": 100,
             "usable_positive_rate": 0.05})["pass"] is False
        assert validity_positive_rate(
            {"usable_positive": 40, "n_rows": 100,
             "usable_positive_rate": 0.40})["pass"] is False


class TestP5SpecPreRegistration:
    _KW = dict(hashes={"d01_audit_sha256": "abc", "dataset_sha256": "d"},
               n_bootstrap=1000, subset_curves=["RXN_1", "RXN_2"],
               n_dev_rows=7010, n_all_rows=12034)

    def test_stable_and_sensitive(self):
        a = spec_sha256(build_p5_spec(**self._KW))
        b = spec_sha256(build_p5_spec(**self._KW))
        assert a == b
        other = dict(self._KW)
        other["hashes"] = {"d01_audit_sha256": "CHANGED", "dataset_sha256": "d"}
        assert spec_sha256(build_p5_spec(**other)) != a
        other2 = dict(self._KW)
        other2["n_bootstrap"] = 500
        assert spec_sha256(build_p5_spec(**other2)) != a

    def test_pins_present(self):
        spec = build_p5_spec(**self._KW)
        text = str(spec)
        for token in ("20261021", "20261105", "0.25", "0.00797",
                      "0.10", "0.30", "50", "600", "forward-intent",
                      "INTERSECTION", "class-T", "subprocess"):
            assert token in text, token
        assert len(spec["validity_checks"]) == 6
        assert spec["conformer_construction"]["hard_caps_global"][
            "max_xtb_calls"] == 50


class TestRealXtb:
    @pytest.mark.xtb
    def test_water_sp_and_constrained_opt(self):
        xtb = Path("/opt/software/xtb/bin/xtb")
        if not xtb.is_file():
            pytest.skip("real xtb binary not installed")
        from scripts.agent_ops.data_gate.v52r_d2_p5_probe import xtb_invoke
        water = np.array([[0.0, 0.0, 0.119],
                          [0.0, 0.763, -0.477],
                          [0.0, -0.763, -0.477]])
        sp = xtb_invoke(water, ("O", "H", "H"), mode="sp", timeout_s=120)
        assert sp["ok"] and sp["energy_ha"] is not None
        # xtb is valence-only: GFN2 water ≈ −5.07 Eh (NOT the ~−76 DFT scale)
        assert -6.5 < sp["energy_ha"] < -4.0
        opt = xtb_invoke(water, ("O", "H", "H"), freeze_atoms=(0, 1, 2),
                         mode="opt", timeout_s=300)
        assert opt["ok"] and opt["converged"]
        assert opt["coords"].shape == (3, 3)
        # $constrain restraints hold every frozen atom close to its input
        assert opt["k_drift_max"] is not None and opt["k_drift_max"] < 0.01
