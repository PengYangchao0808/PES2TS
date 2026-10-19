"""Unit tests for the V5.1a diagnostics suite (synthetic, CPU, no real data).

Covers the pure helper surfaces of the four ``diagnose_v51_*`` scripts:

* D-baseline: alpha ratio with the ||y||>0.02 gate, dead-zone fraction,
  short-id suffix resolution;
* D2 (G5.3 buckets): bucket-edge correctness at exactly 0.005/0.01/0.02 A,
  abstention-equals-incorrect sign accounting, SRE frame eligibility +
  per-bucket contribution, the plan §3.1 D2 verdict rule (both branches);
* D3 (hard clusters): gauge-fallback detection, teacher-shard row
  summarization on mock dicts;
* D4 (gauge): unresolved-row counting, 2-hop candidate counting and
  scatter-gate feasibility on a tiny synthetic molecule, distribution
  summarization.

No GPU, no network, no real V5 outputs required.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from diagnose_v51_baseline_oof import (  # noqa: E402
    DEAD_ZONE_MAG,
    EFFECTIVE_ATOM_TAU,
    atom_alphas,
    bucket_of,
    bucket_verdict,
    dead_zone_fraction,
    resolve_cluster_suffix,
)
from diagnose_v51_g53_buckets import (  # noqa: E402
    SIGN_THRESHOLD,
    bucketize,
    make_verdict,
    pair_rows,
    sre_frame_contribution,
)
from diagnose_v51_hard_clusters import (  # noqa: E402
    uses_gauge_fallback,
)
from diagnose_v51_gauge import (  # noqa: E402
    fallback_feasibility,
    is_unresolved,
    summarize_distribution,
    two_hop_candidates,
)


# ---------------------------------------------------------------------------
# 1. bucket edges (D2)
# ---------------------------------------------------------------------------

def test_bucket_edge_boundaries() -> None:
    # strictly below an edge -> lower bucket
    assert bucket_of(0.0) == 0
    assert bucket_of(0.0049999) == 0
    # EXACTLY on the edge -> UPPER (active/stronger) bucket
    assert bucket_of(0.005) == 1
    assert bucket_of(0.0099999) == 1
    assert bucket_of(0.01) == 2
    assert bucket_of(0.0199999) == 2
    assert bucket_of(0.02) == 3
    assert bucket_of(1.234) == 3


# ---------------------------------------------------------------------------
# 2. alpha with the ||y|| gate (D-baseline)
# ---------------------------------------------------------------------------

def test_atom_alphas_effective_atom_gate() -> None:
    y = np.array([[0.03, 0.0, 0.0],    # effective (0.03 > 0.02)
                  [0.01, 0.0, 0.0],    # NOT effective (0.01 <= 0.02)
                  [0.0, 0.0, 0.0]])    # zero teacher -> excluded, never div0
    u = np.array([[0.015, 0.0, 0.0],   # alpha = 0.5
                  [9.9, 9.9, 9.9],     # huge pred on non-effective atom:
                                       # must NOT leak into alpha
                  [1.0, 1.0, 1.0]])    # zero-teacher atom: excluded
    alphas, eff = atom_alphas(u, y, tau=EFFECTIVE_ATOM_TAU)
    assert eff.tolist() == [True, False, False]
    assert alphas.shape == (1,)
    assert alphas[0] == pytest.approx(0.5)
    # y-gate is exclusive at exactly tau
    y2 = np.array([[EFFECTIVE_ATOM_TAU, 0.0, 0.0]])
    u2 = np.array([[0.01, 0.0, 0.0]])
    a2, eff2 = atom_alphas(u2, y2, tau=EFFECTIVE_ATOM_TAU)
    assert not eff2.any() and a2.size == 0


# ---------------------------------------------------------------------------
# 3. dead-zone fraction (D-baseline)
# ---------------------------------------------------------------------------

def test_dead_zone_fraction_definition() -> None:
    y = np.array([[0.05, 0.0, 0.0],
                  [0.04, 0.0, 0.0],
                  [0.01, 0.0, 0.0]])   # non-effective: not in denominator
    u = np.array([[0.001, 0.0, 0.0],   # dead (||u||=0.001 < 0.005)
                  [0.05, 0.0, 0.0],    # alive
                  [0.0, 0.0, 0.0]])
    dz = dead_zone_fraction(u, y, tau=EFFECTIVE_ATOM_TAU, dead=DEAD_ZONE_MAG)
    assert dz == pytest.approx(0.5)
    # no effective atoms -> None, never coerced to 0
    y0 = np.zeros((2, 3))
    u0 = np.ones((2, 3))
    assert dead_zone_fraction(u0, y0) is None


# ---------------------------------------------------------------------------
# 4. verdict logic (D2, plan §3.1 decision rule)
# ---------------------------------------------------------------------------

def test_verdict_dead_zone_branch() -> None:
    assert bucket_verdict(0.95, 0.70) == (
        "dead-zone/amplitude problem, pair margin loss first")
    # strong exactly 0.9 is NOT ">0.9" -> mixed
    assert bucket_verdict(0.9, 0.70) == "mixed/inconclusive"
    # weak exactly 0.8 is NOT "<0.8" -> mixed
    assert bucket_verdict(0.95, 0.8) == "mixed/inconclusive"


def test_verdict_representation_branch() -> None:
    assert bucket_verdict(0.79, 0.99) == (
        "representation/feature/label problem — loss alone insufficient")
    # strong < 0.8 wins even when weak is also bad
    assert bucket_verdict(0.5, 0.3) == (
        "representation/feature/label problem — loss alone insufficient")
    assert bucket_verdict(None, 0.5) == "insufficient_data"
    assert bucket_verdict(0.95, None) == "insufficient_data"


def test_make_verdict_reads_bucket_table() -> None:
    buckets = {
        "0": {"sign_accuracy": 0.5, "abstention_rate": 0.8,
              "sign_accuracy_where_predicted": 0.6},
        "1": {"sign_accuracy": 0.7},
        "2": {"sign_accuracy": 0.85},
        "3": {"sign_accuracy": 0.95},
    }
    v = make_verdict(buckets)
    assert v["verdict"].startswith("dead-zone")
    assert v["dead_zone_abstention_rate"] == 0.8
    buckets["3"]["sign_accuracy"] = 0.6
    assert make_verdict(buckets)["verdict"].startswith("representation")


# ---------------------------------------------------------------------------
# 5. D2 pair accounting: abstention is never free
# ---------------------------------------------------------------------------

def _pair_row(dd_true, dd_pred, edit=1.0, gid="RXN_T", frame=0):
    return {"group_id": gid, "fold": 0, "branch": 0, "pair": (0, 1),
            "pair_edit": float(edit), "dd_true": float(dd_true),
            "dd_pred": float(dd_pred), "sre_frame_id": frame}


def test_bucketize_abstention_counts_incorrect() -> None:
    # active pair, prediction abstains (|dd_pred| < 0.005) -> INCORRECT
    rows = [_pair_row(0.03, 0.001)]
    out = bucketize(rows)
    strong = out["buckets"]["3"]
    assert strong["n_pairs"] == 1
    assert strong["n_sign_correct"] == 0
    assert strong["abstention_rate"] == 1.0
    assert strong["sign_accuracy_where_predicted"] is None


def test_bucketize_correct_and_flip_accounting() -> None:
    rows = [
        _pair_row(0.03, 0.04, edit=1.0),    # forming, pred longer -> flip
        _pair_row(0.03, -0.04, edit=1.0),   # forming, pred shorter -> ok dir
        _pair_row(-0.03, 0.04, edit=-1.0),  # breaking, pred longer -> ok dir
        _pair_row(-0.03, -0.04, edit=-1.0), # breaking, pred shorter -> flip
    ]
    out = bucketize(rows)
    strong = out["buckets"]["3"]
    assert strong["n_forming"] == 2 and strong["n_breaking"] == 2
    assert strong["sign_flips_forming"] == 1
    assert strong["sign_flips_breaking"] == 1
    # main denominator: sign(dd_pred)==sign(dd_true) AND |pred|>=0.005
    assert strong["n_sign_correct"] == 2
    assert strong["sign_accuracy"] == pytest.approx(0.5)


def test_bucketize_dead_zone_bucket() -> None:
    rows = [_pair_row(0.002, 0.001), _pair_row(0.006, 0.02)]
    out = bucketize(rows)
    dead = out["buckets"]["0"]
    assert dead["n_pairs"] == 1
    assert dead["abstention_rate"] == 1.0
    active = out["buckets"]["1"]
    assert active["n_pairs"] == 1
    assert active["sign_accuracy"] == 1.0


def test_pair_rows_uses_exact_delta_d() -> None:
    pos = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    rec = {"group_id": "RXN_T", "fold": 0, "branch": 0,
           "pos": pos, "key_rows": [0, 1], "pairs": [(0, 1)],
           "pair_edit": [1.0],
           "u": np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
           "y": np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]])}
    rows = pair_rows([rec])
    assert len(rows) == 1
    assert rows[0]["dd_true"] == pytest.approx(0.05, abs=1e-12)
    assert rows[0]["dd_pred"] == pytest.approx(0.1, abs=1e-12)


# ---------------------------------------------------------------------------
# 6. SRE eligibility + per-bucket contribution
# ---------------------------------------------------------------------------

def test_sre_frame_requires_two_pairs_and_total_change() -> None:
    # single-pair frame -> excluded
    assert sre_frame_contribution(np.array([0.05]), np.array([0.05]),
                                  np.array([3])) is None
    # two pairs but tiny total -> excluded
    small = np.array([0.003, 0.002])
    assert sre_frame_contribution(small, small, np.array([0, 0])) is None
    # qualifying frame: proportional prediction -> SRE terms 0
    ddt = np.array([0.03, 0.01])
    out = sre_frame_contribution(ddt, ddt * 2.0, np.array([3, 2]))
    assert out is not None
    assert sum(out.values()) == pytest.approx(0.0, abs=1e-12)
    # fully-zero prediction -> SRE 1 under the dominant bucket
    zero = sre_frame_contribution(ddt, np.zeros(2), np.array([3, 2]))
    assert zero is not None and sum(zero.values()) == pytest.approx(1.0)


def test_sre_bucket_contribution_sums_to_frame_sre() -> None:
    ddt = np.array([0.06, 0.02])
    ddp = np.array([0.03, 0.03])
    buckets = np.array([3, 1])
    out = sre_frame_contribution(ddt, ddp, buckets)
    p = np.abs(ddt) / np.abs(ddt).sum()
    p_hat = np.abs(ddp) / np.abs(ddp).sum()
    expect = 0.5 * np.abs(p_hat - p).sum()
    assert sum(out.values()) == pytest.approx(expect, abs=1e-12)
    assert out[3] == pytest.approx(0.5 * abs(p_hat[0] - p[0]))
    assert out[1] == pytest.approx(0.5 * abs(p_hat[1] - p[1]))


# ---------------------------------------------------------------------------
# 7. hard-cluster suffix matching (D3)
# ---------------------------------------------------------------------------

KNOWN = ["RXN_0000001675", "RXN_0000002309", "RXN_0000000238",
         "RXN_0000001484", "RXN_0000001672", "RXN_0000011484"]


def test_suffix_resolution() -> None:
    res = resolve_cluster_suffix("RXN_1675", KNOWN)
    assert res["status"] == "resolved"
    assert res["matches"] == ["RXN_0000001675"]
    # exact full id also resolves
    res = resolve_cluster_suffix("RXN_0000000238", KNOWN)
    assert res["status"] == "resolved"
    assert res["matches"] == ["RXN_0000000238"]
    # "2309" is unique
    res = resolve_cluster_suffix("2309", KNOWN)
    assert res["status"] == "resolved"
    assert res["matches"] == ["RXN_0000002309"]


def test_suffix_resolution_absent_and_ambiguous() -> None:
    # absent: no cluster ends with 0167
    res = resolve_cluster_suffix("RXN_0000000167", KNOWN)
    assert res["status"] == "absent"
    assert res["matches"] == []
    # ambiguous: BOTH 0000001484 and 0000011484 end with "1484"
    res = resolve_cluster_suffix("1484", KNOWN)
    assert res["status"] == "ambiguous"
    assert set(res["matches"]) == {"RXN_0000001484", "RXN_0000011484"}
    # no digits at all -> absent
    assert resolve_cluster_suffix("RXN_ABC", KNOWN)["status"] == "absent"


# ---------------------------------------------------------------------------
# 8. gauge fallback detection (D3) on mock sidecars
# ---------------------------------------------------------------------------

def test_uses_gauge_fallback_detection() -> None:
    ok = {"0": {"ref_tier": "rigid_1_2hop", "gate_ok": True},
          "1": {"ref_tier": "rigid_1_2hop", "gate_ok": True}}
    assert uses_gauge_fallback(ok) is False
    tiered = {"0": {"ref_tier": "any_1_2hop", "gate_ok": True},
              "1": {"ref_tier": "rigid_1_2hop", "gate_ok": True}}
    assert uses_gauge_fallback(tiered) is True
    failed = {"0": {"ref_tier": "all_nonkey_heavy", "gate_ok": False},
              "1": {"status": "missing"}}
    assert uses_gauge_fallback(failed) is True


# ---------------------------------------------------------------------------
# 9. gauge counting logic (D4) on mock shard arrays
# ---------------------------------------------------------------------------

def test_is_unresolved_mask() -> None:
    reason = np.array([0, 3, 3, 0, 1], dtype=np.uint8)
    mask = is_unresolved(reason)
    assert mask.tolist() == [False, True, True, False, False]
    assert int(mask.sum()) == 2


def test_two_hop_candidates_counts_heavy_neighbors() -> None:
    # H2C=CH2-like: keys 0,1 (double bond); heavy 2 on key0; H 3 on key0;
    # heavy 4 on heavy2 (2 hops from key0); heavy 5 far away (3+ hops via 4)
    z = [6, 6, 8, 1, 6, 6]
    bonds = [(0, 1, 2.0), (0, 2, 1.0), (0, 3, 1.0), (2, 4, 1.0), (4, 5, 1.0)]
    key_rows = [0, 1]
    pool = two_hop_candidates(z, bonds, key_rows)
    # heavy non-key within 2 hops: 2 (hop1), 4 (hop2); NOT 3 (H), NOT 5 (hop3)
    assert pool.tolist() == [2, 4]


def test_fallback_feasibility_gate_pass_and_fail() -> None:
    # spread heavy atoms, no rotors -> 2-hop pool == rigid tier; needs >=3
    # non-collinear candidates for the scatter gate (lambda2/lambda1)
    pos_good = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0],
                         [0.75, 1.3, 0.0], [0.75, -1.3, 0.0],
                         [0.75, 0.0, 1.4]])
    z = [6, 6, 8, 8, 8]
    bonds = [(0, 1, 2.0), (0, 2, 1.0), (1, 3, 1.0), (1, 4, 1.0)]
    out = fallback_feasibility(z, bonds, [0, 1], pos_good)
    assert out["has_any_2hop_candidate"] is True
    assert out["n_two_hop_candidates"] == 3
    assert out["n_candidates_outside_rigid_tier"] == 0
    assert out["adds_beyond_rigid_tier"] is False
    assert out["two_hop_pool_gate_ok"] is True

    # rotor-cut chain: key0-A(deg2)-B(deg2)-C; bond (A,B) is a free rotor,
    # so B is a 2-hop candidate OUTSIDE the rigid tier
    z2 = [6, 6, 8, 8]
    bonds2 = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
    pos_line = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0],
                         [3.0, 0.0, 0.0], [4.5, 0.0, 0.0]])
    out2 = fallback_feasibility(z2, bonds2, [0], pos_line)
    assert out2["has_any_2hop_candidate"] is True
    assert out2["n_two_hop_candidates"] == 2   # atoms 1 (hop1), 2 (hop2)
    assert out2["n_candidates_outside_rigid_tier"] == 1  # atom 2
    assert out2["adds_beyond_rigid_tier"] is True
    # collinear geometry -> 2-hop pool fails lambda2/lambda1
    assert out2["two_hop_pool_gate_ok"] is False
    eigs = out2["two_hop_pool_eigenvalues_angstrom2"]
    assert eigs[0] > 0.01
    assert eigs[1] / eigs[0] <= 1e-3


def test_summarize_distribution_counts() -> None:
    rows = [
        {"d0": 0.05, "family": "famA", "fold": 0, "reason_id": 3,
         "n_atoms": 10, "n_keys": 2, "reaction_id": "RXN_1", "branch": 0,
         "n_unresolved_rows": 5,
         "hop_counts": {"hop0": 0, "hop1": 3, "hop2": 1, "hop3": 0,
                        "hop_gt3": 2}},
        {"d0": 0.05, "family": "famA", "fold": 0, "reason_id": 3,
         "n_atoms": 12, "n_keys": 2, "reaction_id": "RXN_1", "branch": 1,
         "n_unresolved_rows": 3,
         "hop_counts": {"hop0": 0, "hop1": 2, "hop2": 2, "hop3": 1,
                        "hop_gt3": 0}},
        {"d0": 0.08, "family": "famB", "fold": 2, "reason_id": 3,
         "n_atoms": 8, "n_keys": 1, "reaction_id": "RXN_2", "branch": 0,
         "n_unresolved_rows": 2,
         "hop_counts": {"hop0": 0, "hop1": 1, "hop2": 0, "hop3": 0,
                        "hop_gt3": 1}},
    ]
    dist = summarize_distribution(rows)
    # n_rows counts unresolved SAMPLE rows (weighted), n_shards the entries
    assert dist["n_rows"] == 10
    assert dist["n_shards"] == 3
    assert dist["n_branches"] == 3  # RXN_1.branch0/1 + RXN_2.branch0
    assert dist["by_d0"] == {"0.05": 8, "0.08": 2}
    assert dist["by_d0_shards"] == {"0.05": 2, "0.08": 1}
    assert dist["by_family"] == {"famA": 8, "famB": 2}
    assert dist["by_fold"] == {"0": 8, "2": 2}
    assert dist["n_atoms"] == {"min": 8, "median": 10.0, "max": 12}
    assert dist["n_keys"] == {"min": 1, "median": 2.0, "max": 2}
    assert dist["hop_histogram_nonkey_heavy"]["hop1"] == 6
    assert dist["hop_histogram_nonkey_heavy"]["hop2"] == 3
    assert dist["hop_histogram_nonkey_heavy"]["hop_gt3"] == 3


def test_gauge_reason_id_constant() -> None:
    from pes2ts_core.pesn.contracts.key_atom import (
        REASON_GAUGE_UNRESOLVED, REASON_IDS)
    assert REASON_GAUGE_UNRESOLVED == 3
    assert REASON_IDS[3] == "gauge_unresolved"


def test_math_sanity_on_sign_threshold_edge() -> None:
    # a prediction EXACTLY at |dd_pred| = 0.005 is NOT an abstention
    rows = [_pair_row(0.03, SIGN_THRESHOLD)]
    out = bucketize(rows)
    assert out["buckets"]["3"]["n_abstained"] == 0
    assert out["buckets"]["3"]["n_sign_correct"] == 1
    assert math.isclose(out["buckets"]["3"]["sign_accuracy"], 1.0)
