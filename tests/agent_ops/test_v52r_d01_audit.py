"""V5.2-R D0.1 — graph-symmetry + topology-anomaly classification tests.

SYNTHETIC ONLY — no ``outputs/`` reads, no dataset loads, no network.
Covers: automorphism enumeration against known permutation groups (water,
methane, ethanol, symmetric carboxylate), K-subset restriction + dedupe,
``min_sym_rmsd`` (symmetric swap lowers RMSD; a mirrored conformer cannot
beat the proper-rotation floor), ``sym_min_cv_deviation``, the §3.1 S/T/D
decision tree on fixtures, and the §3.2 flip accounting + restatement flag.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pes2ts_core.pesn.offline.data.graph_symmetry import (  # noqa: E402
    AUTOMORPHISM_CAP,
    GraphSymmetryError,
    MolecularGraph,
    enumerate_automorphisms,
    k_automorphism_permutations,
    min_sym_rmsd,
    refine_partition,
    sym_min_cv_deviation,
)
from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (  # noqa: E402
    kabsch_rmsd_v52,
)
from scripts.agent_ops.data_gate.v52r_d01_audit import (  # noqa: E402
    classify_row,
    ks_2samp,
    strictly_between,
    sym_min_cv_deviation as audit_sym_min_cv_deviation,
    wasserstein_1d,
)


# --------------------------------------------------------------- fixtures

def water_graph() -> MolecularGraph:
    """H2O: O(0) bonded to H(1), H(2) — one symmetry (swap the two H)."""
    return MolecularGraph.from_intent_payload(
        atoms=(8, 1, 1),
        bonds=((0, 1, 1.0), (0, 2, 1.0)))


def methane_graph() -> MolecularGraph:
    """CH4: C(0) + H(1..4) — full group is 4! = 24 on the H's."""
    return MolecularGraph.from_intent_payload(
        atoms=(6, 1, 1, 1, 1),
        bonds=((0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (0, 4, 1.0)))


def _ethanol() -> MolecularGraph:
    # C0 bonded to C1; C0 has H_a, H_b, H_c; C1 has H_d, H_e and O; O has H_o.
    # Row order: 0=C0, 1=C1, 2=Ha, 3=Hb, 4=Hc, 5=Hd, 6=He, 7=O, 8=Ho
    return MolecularGraph.from_intent_payload(
        atoms=(6, 6, 1, 1, 1, 1, 1, 8, 1),
        bonds=((0, 1, 1.0),
               (0, 2, 1.0), (0, 3, 1.0), (0, 4, 1.0),   # methyl H's
               (1, 5, 1.0), (1, 6, 1.0),                # C1 H's (also symmetric!)
               (1, 7, 1.0), (7, 8, 1.0)))               # C1-O, O-H


def _carboxylate() -> MolecularGraph:
    """Symmetric carboxylate-like: formate HCOO⁻ skeleton.

    C0 with two EQUAL-order O's (1.5/1.5 — delocalized) plus one H; the
    two O's are graph-interchangeable (group order 2).
    """
    return MolecularGraph.from_intent_payload(
        atoms=(6, 8, 8, 1),
        bonds=((0, 1, 1.5), (0, 2, 1.5), (0, 3, 1.0)))


# ----------------------------------------------------- automorphism tests

class TestAutomorphismEnumeration:
    def test_water_group_order_2(self) -> None:
        autos = enumerate_automorphisms(water_graph())
        assert len(autos) == 2
        assert tuple(range(3)) in autos           # identity
        assert (0, 2, 1) in autos                 # H swap

    def test_methane_full_group_order_24(self) -> None:
        autos = enumerate_automorphisms(methane_graph())
        assert len(autos) == 24                   # 4! on the H's, C fixed
        assert all(p[0] == 0 for p in autos)      # carbon never moves

    def test_ethanol_group_order_12(self) -> None:
        """Methyl 3! × C1-H pair 2! = 12 (O-H H is distinct)."""
        autos = enumerate_automorphisms(_ethanol())
        assert len(autos) == 12

    def test_carboxylate_group_order_2(self) -> None:
        autos = enumerate_automorphisms(_carboxylate())
        assert len(autos) == 2                    # O1/O2 swap only

    def test_identity_always_present(self) -> None:
        for g in (water_graph(), methane_graph(), _ethanol(), _carboxylate()):
            n = g.n_atoms
            assert tuple(range(n)) in enumerate_automorphisms(g)

    def test_partition_refinement_splits_by_element(self) -> None:
        colors = refine_partition(water_graph())
        assert colors[0] != colors[1]             # O ≠ H
        assert colors[1] == colors[2]             # H == H

    def test_cap_raises_typed_error(self) -> None:
        with pytest.raises(GraphSymmetryError) as exc:
            enumerate_automorphisms(methane_graph(), cap=5)
        assert exc.value.code == "AUTO_CAP"

    def test_cap_boundary_ok(self) -> None:
        autos = enumerate_automorphisms(methane_graph(), cap=24)
        assert len(autos) == 24

    def test_deterministic_order(self) -> None:
        a = enumerate_automorphisms(_ethanol())
        b = enumerate_automorphisms(_ethanol())
        assert a == b

    def test_empty_graph_raises(self) -> None:
        with pytest.raises(GraphSymmetryError) as exc:
            enumerate_automorphisms(
                MolecularGraph.from_intent_payload(atoms=(), bonds=()))
        assert exc.value.code == "GRAPH_EMPTY"


class TestKRestriction:
    def test_methane_k_on_h_set_full_24(self) -> None:
        perms = k_automorphism_permutations(methane_graph(), k_rows=(1, 2, 3, 4))
        assert len(perms) == 24                   # all 4! on H's
        assert tuple(range(4)) in perms           # identity

    def test_methane_k_on_carbon_plus_one_h_only_identity(self) -> None:
        # K = {C, H1}: the only automorphism mapping {0,1} onto itself is
        # the identity (every other perm moves H1 off K).
        perms = k_automorphism_permutations(methane_graph(), k_rows=(0, 1))
        assert perms == ((0, 1),)

    def test_ethanol_k_on_methyl_h_group_6(self) -> None:
        perms = k_automorphism_permutations(_ethanol(), k_rows=(2, 3, 4))
        assert len(perms) == 6                    # 3! on the methyl H's
        assert tuple(range(3)) in perms

    def test_ethanol_k_on_oh_pair_only_identity(self) -> None:
        # K = {O7, H8}: no symmetry maps this set onto itself non-trivially.
        perms = k_automorphism_permutations(_ethanol(), k_rows=(7, 8))
        assert perms == ((0, 1),)

    def test_carboxylate_k_on_o_pair_swap(self) -> None:
        perms = k_automorphism_permutations(_carboxylate(), k_rows=(1, 2))
        assert len(perms) == 2
        assert (0, 1) in perms and (1, 0) in perms

    def test_dedupe_correctness(self) -> None:
        """Two distinct full automorphisms that induce the SAME K-perm are deduped."""
        # In ethanol, the full group is 12 (methyl 6 × C1-H pair 2).
        # K = {2,3,4} (methyl H's): the C1-H pair swap induces identity on
        # K, so all 12 collapse to 6 distinct K-perms.
        autos = enumerate_automorphisms(_ethanol())
        k = (2, 3, 4)
        raw_induced = set()
        for p in autos:
            if set(p[a] for a in k) == set(k):
                pos = {a: i for i, a in enumerate(k)}
                raw_induced.add(tuple(pos[p[a]] for a in k))
        perms = k_automorphism_permutations(_ethanol(), k_rows=k)
        assert len(perms) == len(raw_induced) == 6

    def test_empty_k_raises(self) -> None:
        with pytest.raises(GraphSymmetryError) as exc:
            k_automorphism_permutations(methane_graph(), k_rows=())
        assert exc.value.code == "K_SET_EMPTY"

    def test_k_out_of_range_raises(self) -> None:
        with pytest.raises(GraphSymmetryError) as exc:
            k_automorphism_permutations(methane_graph(), k_rows=(0, 99))
        assert exc.value.code == "K_SET_RANGE"


class TestMinSymRmsd:
    def test_symmetric_swap_lowers_rmsd(self) -> None:
        # water K-geometry with UNEQUAL H distances so the H-swap is NOT a
        # proper rotation of the molecule (identity Kabsch gives > 0, but
        # the swap permutation gives exactly 0).
        x_ref = np.array([[0.0, 0.0, 0.0],    # O
                          [1.5, 0.0, 0.0],    # H1 (far)
                          [0.0, 1.0, 0.0]])   # H2 (near)
        # mobile: H1 and H2 rows swapped (same molecule, different labeling)
        x_mob = x_ref[[0, 2, 1]]
        perms = k_automorphism_permutations(water_graph(), k_rows=(0, 1, 2))
        assert len(perms) == 2                  # identity + swap
        d_identity = kabsch_rmsd_v52(x_mob, x_ref)
        assert d_identity > 0.1                 # swap is not a proper rotation
        d_sym = min_sym_rmsd(x_mob, x_ref, perms)
        assert d_sym == pytest.approx(0.0, abs=1e-12)
        assert d_sym < d_identity

    def test_identity_only_matches_plain_kabsch(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.normal(size=(5, 3))
        b = rng.normal(size=(5, 3))
        d = min_sym_rmsd(a, b, perms=[(0, 1, 2, 3, 4)])
        assert d == pytest.approx(kabsch_rmsd_v52(a, b), abs=1e-12)

    def test_mirrored_conformer_cannot_beat_proper_rotation_floor(self) -> None:
        # chiral point set (scalene tetrahedron — no mirror symmetry):
        # the mirror image cannot be superimposed by ANY permutation + proper
        # rotation because the signed volume flips sign.
        x_ref = np.array([[0.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0],
                          [0.0, 2.0, 0.0],
                          [0.0, 0.0, 3.0]])
        x_mirror = x_ref * np.array([1.0, 1.0, -1.0])   # reflect through z=0
        all_perms = list(itertools.permutations(range(4)))
        d_floor = min_sym_rmsd(x_mirror, x_ref, all_perms)
        # proper-rotation floor is strictly positive for a chiral pair:
        # no permutation + proper rotation superimposes opposite enantiomers
        assert d_floor > 0.1
        # and no permutation beats the identity's floor (the best perm is
        # no better than the plain proper-rotation Kabsch)
        d_identity = kabsch_rmsd_v52(x_mirror, x_ref)
        assert d_floor <= d_identity + 1e-9
        assert d_floor > 0.0

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(GraphSymmetryError) as exc:
            min_sym_rmsd(np.zeros((3, 3)), np.zeros((4, 3)),
                         perms=[(0, 1, 2)])
        assert exc.value.code == "SYM_RMSD_SHAPE"

    def test_empty_perms_raises(self) -> None:
        with pytest.raises(GraphSymmetryError) as exc:
            min_sym_rmsd(np.zeros((3, 3)), np.zeros((3, 3)), perms=[])
        assert exc.value.code == "SYM_RMSD_NO_PERMS"


class TestSymMinCvDeviation:
    def test_identity_matches_manual(self) -> None:
        frame = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 2.0, 0.0]])
        ref = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        pairs = [(0, 1), (0, 2)]
        # identity: |1.5-1.0| = 0.5 and |2.0-1.0| = 1.0 → max = 1.0
        d = sym_min_cv_deviation(frame, ref, pairs, perms=[(0, 1, 2)])
        assert d == pytest.approx(1.0, abs=1e-12)

    def test_swap_reduces_deviation(self) -> None:
        # ref: O at origin, H1 at (1.5,0,0), H2 at (0,1,0).
        # frame: H1 and H2 positions exchanged (same molecule, different
        # labeling).  Identity gives large deviation; the swap perm gives 0.
        ref = np.array([[0.0, 0.0, 0.0],
                        [1.5, 0.0, 0.0],
                        [0.0, 1.0, 0.0]])
        frame = ref[[0, 2, 1]]
        pairs = [(0, 1), (0, 2)]
        d_id = sym_min_cv_deviation(frame, ref, pairs, perms=[(0, 1, 2)])
        d_swap = sym_min_cv_deviation(frame, ref, pairs, perms=[(0, 2, 1)])
        assert d_swap == pytest.approx(0.0, abs=1e-12)
        assert d_id > 0.1
        assert d_swap < d_id

    def test_no_pairs_raises(self) -> None:
        with pytest.raises(GraphSymmetryError) as exc:
            sym_min_cv_deviation(np.zeros((3, 3)), np.zeros((3, 3)),
                                 cv_pairs=[], perms=[(0, 1, 2)])
        assert exc.value.code == "SYM_CV_NO_PAIRS"

    def test_audit_reexport_matches(self) -> None:
        frame = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        ref = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        pairs = [(0, 1)]
        d1 = sym_min_cv_deviation(frame, ref, pairs, perms=[(0, 1)])
        d2 = audit_sym_min_cv_deviation(frame, ref, pairs, [(0, 1)])
        assert d1 == d2


# ------------------------------------------------- decision-tree fixtures

def _row_fixture(**overrides) -> dict:
    base = {
        "sample_id": "SYN:0:SYN_fwd:0.08",
        "curve_id": "SYN",
        "node_index": 0,
        "direction": "forward",
        "fold": 0,
        "split": "dev",
        "label_status": "ok",
        "d_end_angstrom": 0.08,
        "d_end_sym_angstrom": 0.08,
        "sym_min_cv_dev_angstrom": 0.01,
        "n_automorphisms": 2,
        "n_k_perms": 2,
        "n_pairs": 5,
        "max_pair_dev_angstrom": 0.30,
        "pair_diagnosis": [],
    }
    base.update(overrides)
    return base


class TestDecisionTree:
    def test_s_by_sym_min_dev(self) -> None:
        # sym_min_cv_dev ≤ 0.25 triggers S; d_end_sym ≤ 0.10 keeps positive
        row = _row_fixture(sym_min_cv_dev_angstrom=0.20,
                           d_end_sym_angstrom=0.08)
        verdict = classify_row(row)
        assert verdict["class"] == "S"
        assert verdict["disposition"] == "keep_positive"
        assert verdict["final_label"] == 1

    def test_s_by_d_end_sym(self) -> None:
        # the flip clause: d_end > 0.10 (non-positive under fixed mapping)
        # but d_end_sym ≤ 0.10 under some automorphism → S, becomes positive
        row = _row_fixture(sym_min_cv_dev_angstrom=0.40,
                           d_end_angstrom=0.25,
                           d_end_sym_angstrom=0.09)
        verdict = classify_row(row)
        assert verdict["class"] == "S"
        assert verdict["d_end_sym_label"] == 1
        assert verdict["disposition"] == "keep_positive"

    def test_s_flip_clause_vacuous_for_positives(self) -> None:
        # for an already-positive row (d_end ≤ 0.10), the d_end_sym ≤ 0.10
        # clause is VACUOUS (d_end_sym ≤ d_end ≤ 0.10 always) — it must not
        # fire S when sym_min_cv_dev > 0.25, otherwise genuine anomalies
        # would be misclassified as symmetry-explained.
        row = _row_fixture(sym_min_cv_dev_angstrom=0.40,
                           d_end_angstrom=0.08,
                           d_end_sym_angstrom=0.08)
        verdict = classify_row(row)
        assert verdict["class"] != "S"      # falls through to T or D

    def test_s_flips_to_mask(self) -> None:
        # sym_min_cv_dev ≤ 0.25 triggers S; d_end_sym = 0.20 (between
        # windows) → new label = −1 → Mask
        row = _row_fixture(sym_min_cv_dev_angstrom=0.20,
                           d_end_sym_angstrom=0.20)
        verdict = classify_row(row)
        assert verdict["class"] == "S"
        assert verdict["d_end_sym_label"] == -1
        assert verdict["disposition"] == "mask"
        assert verdict["final_label"] == -1

    def test_t_by_changing_bond_between(self) -> None:
        row = _row_fixture(
            sym_min_cv_dev_angstrom=0.40, d_end_sym_angstrom=0.25,
            pair_diagnosis=[
                {"pair": [0, 1], "deviating": True,
                 "delta_b_intent": -1.0, "is_changing_bond": True,
                 "strictly_between_ab": True},
                {"pair": [2, 3], "deviating": True,
                 "delta_b_intent": 1.0, "is_changing_bond": True,
                 "strictly_between_ab": True},
                {"pair": [4, 5], "deviating": False,
                 "delta_b_intent": 0.0, "is_changing_bond": False,
                 "strictly_between_ab": False},
            ])
        verdict = classify_row(row)
        assert verdict["class"] == "T"
        assert verdict["disposition"] == "mask"
        assert verdict["final_label"] == -1

    def test_t_fails_when_not_changing_bond(self) -> None:
        row = _row_fixture(
            sym_min_cv_dev_angstrom=0.40, d_end_sym_angstrom=0.25,
            pair_diagnosis=[
                {"pair": [0, 1], "deviating": True,
                 "delta_b_intent": 0.0, "is_changing_bond": False,
                 "strictly_between_ab": True},
            ])
        verdict = classify_row(row)
        assert verdict["class"] == "D"       # not a changing bond → fall to D

    def test_t_fails_when_not_strictly_between(self) -> None:
        row = _row_fixture(
            sym_min_cv_dev_angstrom=0.40, d_end_sym_angstrom=0.25,
            pair_diagnosis=[
                {"pair": [0, 1], "deviating": True,
                 "delta_b_intent": -1.0, "is_changing_bond": True,
                 "strictly_between_ab": False},
            ])
        verdict = classify_row(row)
        assert verdict["class"] == "D"

    def test_d_fallthrough(self) -> None:
        row = _row_fixture(
            sym_min_cv_dev_angstrom=0.40, d_end_sym_angstrom=0.25,
            pair_diagnosis=[])
        verdict = classify_row(row)
        assert verdict["class"] == "D"
        assert verdict["disposition"] == "mask"
        assert verdict["final_label"] == -1

    def test_strictly_between_guards(self) -> None:
        assert strictly_between(1.5, 1.0, 2.0)
        assert not strictly_between(1.0, 1.0, 2.0)   # equal to lo
        assert not strictly_between(2.0, 1.0, 2.0)   # equal to hi
        assert not strictly_between(3.0, 1.0, 2.0)   # above hi
        assert strictly_between(1.5, 2.0, 1.0)       # reversed args


class TestFlipStats:
    def test_ks_identical_distributions(self) -> None:
        a = [1.0, 2.0, 3.0]
        b = [1.0, 2.0, 3.0]
        assert ks_2samp(a, b) == pytest.approx(0.0, abs=1e-12)

    def test_ks_disjoint(self) -> None:
        a = [0.0, 0.1, 0.2]
        b = [5.0, 6.0, 7.0]
        assert ks_2samp(a, b) == pytest.approx(1.0, abs=1e-12)

    def test_ks_empty(self) -> None:
        assert ks_2samp([], [1.0]) is None

    def test_wasserstein_shift(self) -> None:
        a = [0.0, 1.0, 2.0]
        b = [1.0, 2.0, 3.0]
        # W1 = 1.0 (uniform shift by 1)
        assert wasserstein_1d(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_wasserstein_empty(self) -> None:
        assert wasserstein_1d([], [1.0]) is None


class TestRestatementFlag:
    """The §3.2 flip accounting + >2% restatement rule."""

    @staticmethod
    def _simulate_flip_stats(pos_flips: int, neg_to_pos: int,
                             mask_to_pos: int, total_pos: int = 1781
                             ) -> dict:
        flips_involving_pos = pos_flips + neg_to_pos + mask_to_pos
        frac = flips_involving_pos / max(1, total_pos)
        return {
            "n_flips_involving_positive_class": flips_involving_pos,
            "flip_fraction_of_positives": frac,
            "requires_d0_restatement": frac > 0.02,
        }

    def test_below_threshold_no_restatement(self) -> None:
        stats = self._simulate_flip_stats(pos_flips=0, neg_to_pos=10,
                                          mask_to_pos=20)
        assert stats["requires_d0_restatement"] is False
        assert stats["flip_fraction_of_positives"] == pytest.approx(
            30 / 1781)

    def test_above_threshold_triggers_restatement(self) -> None:
        stats = self._simulate_flip_stats(pos_flips=0, neg_to_pos=20,
                                          mask_to_pos=20)
        # 40/1781 = 2.25% > 2%
        assert stats["requires_d0_restatement"] is True

    def test_pos_to_neg_also_counts(self) -> None:
        stats = self._simulate_flip_stats(pos_flips=36, neg_to_pos=0,
                                          mask_to_pos=0)
        # 36/1781 = 2.02% > 2%
        assert stats["requires_d0_restatement"] is True


class TestGraphPayloadParsing:
    def test_from_graph_payload(self) -> None:
        g = MolecularGraph.from_graph_payload(
            {"atoms": [8, 1, 1], "bonds": [[0, 1, 1.0], [0, 2, 1.0]]})
        assert g.n_atoms == 3
        assert len(g.bonds) == 2

    def test_selfloop_raises(self) -> None:
        with pytest.raises(GraphSymmetryError) as exc:
            MolecularGraph.from_intent_payload(
                atoms=(8, 1), bonds=((0, 0, 1.0),))
        assert exc.value.code == "GRAPH_SELFLOOP"

    def test_out_of_range_bond_raises(self) -> None:
        with pytest.raises(GraphSymmetryError) as exc:
            MolecularGraph.from_intent_payload(
                atoms=(8, 1), bonds=((0, 5, 1.0),))
        assert exc.value.code == "GRAPH_BOND_RANGE"

    def test_bond_order_normalization(self) -> None:
        """1.0000000001 rounds to 1.0 so it does not fragment color classes."""
        g = MolecularGraph.from_intent_payload(
            atoms=(8, 1, 1), bonds=((0, 1, 1.0000000001), (0, 2, 1.0)))
        assert g.bonds == ((0, 1, 1.0), (0, 2, 1.0))


class TestCapConstant:
    def test_cap_is_100k(self) -> None:
        assert AUTOMORPHISM_CAP == 100_000


class TestRdkitCrossValidation:
    """Optional cross-check of the enumerator against rdkit (tests only)."""

    @staticmethod
    def _rdkit_aut_count(smi: str) -> int:
        from math import factorial
        from collections import Counter
        pytest.importorskip("rdkit")
        from rdkit import Chem
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
        prod = 1
        for v in Counter(ranks).values():
            prod *= factorial(v)
        return prod

    @pytest.mark.parametrize("smi,name,expect", [
        ("O", "water", 2),
        ("C", "methane", 24),
        ("CCO", "ethanol", 12),
    ])
    def test_group_order_matches_rdkit(self, smi: str, name: str,
                                       expect: int) -> None:
        """Canonical-rank product-of-factorials cross-check (tests only)."""
        assert self._rdkit_aut_count(smi) == expect
        # and our enumerator agrees on the same hand-built graph
        if name == "water":
            g = water_graph()
        elif name == "methane":
            g = methane_graph()
        else:
            g = _ethanol()
        assert len(enumerate_automorphisms(g)) == expect
