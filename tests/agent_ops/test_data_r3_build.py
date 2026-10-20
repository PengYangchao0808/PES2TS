"""E1 data-build tests (lane W-DATA): builder + real-lineage invariants.

These tests exercise ``scripts/agent_ops/data_gate/r3_build_tables.py``
against the real R3 fit-wave lineage (wp2_fit50 export + G1 P1 mappings +
frozen V5 split/audit artifacts).  The lineage lives in the main working
tree; when the test file itself is executed from a worktree that does not
carry the (gitignored) data artifacts, the lineage repo is resolved via
``PES2TS_R3_LINEAGE_REPO`` or the campaign main-tree path.

Covered:
* real build (subset) passes every Data Gate test T0.1a..T0.5;
* deterministic rebuild (byte-identical manifests);
* PD-9 edge encoding round-trip (float64-LE sha, offsets);
* intent pairing / reverse-of involution on the built tables;
* per-(curve, d0) curve_weight sum == 1.0 invariant;
* teacher round-trip: a stored sample target equals a fresh
  ``solve_curve_direction`` recomputation;
* prior denominators reproduction (5568/5088/480 rows; 137@0.05 and
  282@0.08 terminal) — the original denominators are pinned;
* cohort dispositions: 1248 fully masked, RXN_0000001174 quarantined,
  RXN_0000002022 rechecked at d0=0.08 (terminal rows exist);
* split inherits the frozen V5 fold map with singleton components.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

import pandas as pd
import pytest

WORKTREE_ROOT = Path(__file__).resolve().parents[2]
MAIN_TREE_CANDIDATE = Path("/mnt/e/Calculations/AI4S_ML_Studys/PES2TS")
LINEAGE_MARKER = Path("outputs/pesn/dual_ts_v2_20261015/wp2_fit50")


def _resolve_lineage_repo() -> Path:
    import os
    env = os.environ.get("PES2TS_R3_LINEAGE_REPO")
    if env and (Path(env) / LINEAGE_MARKER).is_dir():
        return Path(env)
    if (WORKTREE_ROOT / LINEAGE_MARKER).is_dir():
        return WORKTREE_ROOT
    if (MAIN_TREE_CANDIDATE / LINEAGE_MARKER).is_dir():
        return MAIN_TREE_CANDIDATE
    pytest.skip("R3 fit-wave lineage not found (worktree without data "
                "artifacts and no PES2TS_R3_LINEAGE_REPO)")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def lineage_repo() -> Path:
    return _resolve_lineage_repo()


@pytest.fixture(scope="module")
def builder(lineage_repo):
    return _load_module(
        "r3_build_tables_under_test",
        WORKTREE_ROOT / "scripts/agent_ops/data_gate/r3_build_tables.py")


@pytest.fixture(scope="module")
def validator():
    return _load_module(
        "validate_r3_tables_under_test",
        WORKTREE_ROOT / "scripts/agent_ops/data_gate/validate_r3_tables.py")


@pytest.fixture(scope="module")
def smoke_build(builder, lineage_repo, tmp_path_factory):
    out = tmp_path_factory.mktemp("r3_smoke")
    summary = builder.build(lineage_repo, out, limit=3)
    return out, summary


@pytest.fixture(scope="module")
def full_build(builder, lineage_repo, tmp_path_factory):
    out = tmp_path_factory.mktemp("r3_full")
    summary = builder.build(lineage_repo, out)
    return out, summary


# ---------------------------------------------------------------------------
# Data Gate on real builds
# ---------------------------------------------------------------------------

def test_smoke_build_passes_all_gate_tests(smoke_build, validator, tmp_path):
    out, _ = smoke_build
    report = validator.run_validation(
        out, tmp_path / "gate_smoke.json",
        truth_log=out / "truth_access_log.jsonl")
    statuses = {t["id"]: t["status"] for t in report["tests"]}
    assert all(s == "pass" for s in statuses.values()), statuses
    assert report["exit_code"] == 0


def test_full_build_passes_all_gate_tests(full_build, validator, tmp_path):
    out, _ = full_build
    report = validator.run_validation(
        out, tmp_path / "gate_full.json",
        truth_log=out / "truth_access_log.jsonl")
    statuses = {t["id"]: t["status"] for t in report["tests"]}
    assert all(s == "pass" for s in statuses.values()), statuses
    assert report["exit_code"] == 0


def test_full_build_coverage_matches_lineage(full_build, builder,
                                             lineage_repo):
    out, summary = full_build
    coverage = summary["coverage"]
    assert coverage["n_curves"] == 50
    # node count equals the wp2 coverage-audit unique node count
    cov = json.loads((lineage_repo / builder.COVERAGE_REL).read_text())
    assert coverage["n_nodes"] == cov["branches"]["states_total"] - \
        cov["branches"]["count"] + 50  # drop 1 duplicate TS per reaction
    assert coverage["n_intents"] == 100
    assert coverage["n_samples"] == 2 * coverage["n_nodes"]
    # every curve has both intents; 1248 fully masked
    sv = pd.read_csv(out / "sample_view.csv")
    r1248 = sv[sv["curve_id"] == "RXN_0000001248"]
    assert len(r1248) > 0
    assert (r1248["label_status"] == "masked").all()
    assert (r1248["gauge_ref"] == "unresolved_masked").all()
    assert (r1248["mask"].astype(str).str.lower() == "true").all()


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------

def test_deterministic_rebuild(builder, lineage_repo, tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    builder.build(lineage_repo, a, limit=3)
    builder.build(lineage_repo, b, limit=3)
    for name in ("data_manifest.json", "teacher_manifest.json",
                 "split_manifest_r3.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    ma = json.loads((a / "data_manifest.json").read_text())
    mb = json.loads((b / "data_manifest.json").read_text())
    assert ma["dataset_sha256"] == mb["dataset_sha256"]


# ---------------------------------------------------------------------------
# table-level invariants on the full real build
# ---------------------------------------------------------------------------

def test_curve_weight_sums_to_one_per_curve_d0(full_build):
    out, _ = full_build
    sv = pd.read_csv(out / "sample_view.csv")
    grouped = sv.groupby(["curve_id", "d0"])["curve_weight"].apply(
        lambda s: float(sum(map(float, s))))
    assert ((grouped - 1.0).abs() <= 1e-9).all()


def test_intent_pairing_involution(full_build):
    out, _ = full_build
    ri = pd.read_csv(out / "reaction_intent.csv")
    assert len(ri) == 100
    by_curve = dict(list(ri.groupby("curve_id")))
    assert len(by_curve) == 50
    for _cid, grp in by_curve.items():
        assert len(grp) == 2
        assert set(grp["direction"]) == {"forward", "reverse"}
        fwd = grp[grp["direction"] == "forward"].iloc[0]
        rev = grp[grp["direction"] == "reverse"].iloc[0]
        assert fwd["reverse_of_intent_id"] == rev["intent_id"]
        assert rev["reverse_of_intent_id"] == fwd["intent_id"]
        assert fwd["intent_pair_id"] == rev["intent_pair_id"]
        assert not bool(fwd["intent_underdetermined"])


def test_pd9_edge_encoding_roundtrip(full_build):
    out, _ = full_build
    edges = pd.read_csv(out / "irc_edge.csv")
    assert (edges["units"] == "angstrom").all()
    offsets_prev_end = -1
    for pos, row in edges.head(200).iterrows():
        flat = json.loads(row["delta_xyz_flat"])
        start, end = json.loads(row["delta_xyz_offsets"])
        assert end - start == len(flat)
        if pos == 0:
            assert start == 0
        else:
            assert start == offsets_prev_end
        offsets_prev_end = end
        packed = struct.pack(f"<{len(flat)}d", *flat)
        assert hashlib.sha256(packed).hexdigest() == \
            str(row["delta_xyz_sha256"]).lower()


def test_reverse_edges_are_exact_negations(full_build):
    out, _ = full_build
    edges = pd.read_csv(out / "irc_edge.csv")
    cid = "RXN_0000000072"
    fwd = edges[(edges["curve_id"] == cid) &
                (edges["intent_id"] == f"{cid}_fwd")].set_index("edge_index")
    rev = edges[(edges["curve_id"] == cid) &
                (edges["intent_id"] == f"{cid}_rev")].set_index("edge_index")
    assert len(fwd) == len(rev) > 0
    for ei, frow in fwd.iterrows():
        rrow = rev.loc[ei]
        assert int(rrow["src_node_index"]) == int(frow["dst_node_index"])
        assert int(rrow["dst_node_index"]) == int(frow["src_node_index"])
        fflat = json.loads(frow["delta_xyz_flat"])
        rflat = json.loads(rrow["delta_xyz_flat"])
        worst = max(abs(a + b) for a, b in zip(fflat, rflat))
        assert worst <= 1e-6


def test_teacher_roundtrip_one_sample(full_build, builder, lineage_repo):
    out, _ = full_build
    sv = pd.read_csv(out / "sample_view.csv")
    row = sv[(sv["curve_id"] == "RXN_0000000139") &
             (sv["node_index"] == 10) &
             (sv["intent_id"] == "RXN_0000000139_fwd")].iloc[0]
    assert row["label_status"] == "ok"
    bundles = builder.load_reactions(lineage_repo, limit=None)
    bundle = next(b for b in bundles if b.rid == "RXN_0000000139")
    curve = builder.Curve(bundle)
    sol = builder.solve_curve_direction(curve, "forward")[10]
    stored_u = json.loads(row["target_action_u"])
    assert stored_u == [float(v) for v in sol["u_raw"].ravel()]
    sol_row = {
        "label_status": "ok", "mask": False,
        "target_node_index": sol["target_node_index"],
        "target_alpha": sol["target_alpha"], "u_raw": sol["u_raw"]}
    assert builder.sha256_json(builder.label_payload(
        sol_row, builder.MAIN_D0)) == str(row["label_sha"]).lower()
    # and the target is an interpolated first contact at d0
    u = sol["u_raw"]
    assert u.shape == (len(curve.key_rows), 3)
    rms = (u ** 2).sum(axis=1).mean() ** 0.5
    assert 0.0 < rms <= builder.MAIN_D0 + 1e-9


# ---------------------------------------------------------------------------
# prior denominators + cohort dispositions (honesty pins)
# ---------------------------------------------------------------------------

def test_prior_denominators_reproduced(builder, lineage_repo):
    prior = builder.prior_teacher_counts(lineage_repo)
    t05 = prior["0.05"]["totals"]
    t08 = prior["0.08"]["totals"]
    assert t05["rows"] == 5568
    assert t05["eligible"] == 5088
    assert t05["gauge_unresolved"] == 480
    assert t05["terminal"] == 137
    assert t08["gauge_unresolved"] == 480
    assert t08["terminal"] == 282


def test_prior_gauge_branches_seven_reactions_eight_branches(
        builder, lineage_repo):
    prior = builder.prior_branch_dispositions(lineage_repo)
    unres = [(rid, b, info["side"])
             for rid, info in sorted(prior.items())
             for b, info in sorted(info.items()) if info["unresolved"]]
    assert len(unres) == 8
    assert len({rid for rid, _b, _s in unres}) == 7
    assert "RXN_0000001248" in {rid for rid, _b, _s in unres}


def test_cohort_dispositions_in_ledger(full_build):
    out, _ = full_build
    ledger = json.loads((out / "build_ledger_r3.json").read_text())
    cohorts = ledger["cohorts"]
    assert cohorts["RXN_0000001248"]["disposition"] == \
        "excluded_gauge_unresolved"
    ov = {row["reaction_id"]: row["disposition"]
          for row in cohorts["mapping_overflow_5_pairs"]
          ["disposition_table"]}
    assert set(ov) == {"RXN_0000000238", "RXN_0000001089",
                       "RXN_0000001174", "RXN_0000001504",
                       "RXN_0000002226"}
    assert ov["RXN_0000001174"] == "quarantine_mapping_overflow"
    assert all(v == "consistent_not_quarantined"
               for k, v in ov.items() if k != "RXN_0000001174")
    r22 = cohorts["RXN_0000002022"]["r3_samples_by_status_at_d0_0_08"]
    assert r22.get("terminal", 0) > 0  # recheck flips the 0.05 conclusion
    assert cohorts["historical_342_conflict_pairs"][
        "pair_list_artifact"].startswith("NOT PRESENT")


def test_split_inherits_frozen_v5(full_build, lineage_repo):
    out, _ = full_build
    split = json.loads((out / "split_manifest_r3.json").read_text())
    frozen = json.loads(
        (lineage_repo / "data/manifests/v5/split_manifest.json").read_text())
    got = {a["curve_id"]: a["fold"] for a in split["assignments"]}
    assert got == {k: int(v) for k, v in frozen["group_fold"].items()}
    # near-dup components are singletons in fit50 -> one curve per group
    assert all(len(v) == 1 for v in split["near_dup_groups"].values())
    # every curve's intents/nodes/d0 share one fold (same curve => same row
    # group by construction of the manifest)


def test_no_ground_truth_access_in_build(full_build):
    out, _ = full_build
    log = out / "truth_access_log.jsonl"
    assert log.is_file()
    assert log.read_text(encoding="utf-8").strip() == ""
