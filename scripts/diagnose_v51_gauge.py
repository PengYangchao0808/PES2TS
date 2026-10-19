#!/usr/bin/env python3
"""V5.1a D4: where does the 8.6% gauge_unresolved live, and would a
second-order topological-neighbor fallback even have candidates?
(plan §3.1 D4; analysis only — NO fallback is implemented here).

Reads the V5 teacher shards (all three d0 values, primarily 0.05), selects
rows with ``reason_id == 3`` (``REASON_GAUGE_UNRESOLVED``), and reports the
distribution by reaction family, atom count, key-atom count, hop-structure
proxies (per-atom hop-from-key one-hot counts available in the feature
shards' ``h_node_inv``) and outer fold.

Fallback feasibility: for every unresolved (reaction, branch), re-derives
the tiered reference-candidate sets from the START topology via the
frozen teacher helpers (``compute_reference_block`` /
``reference_gate``) and counts how many branches have at least one
non-key heavy atom within 2 hops of ANY key atom that is NOT already
covered by the four existing tiers (rigid_1_2hop -> any_1_2hop ->
any_3hop -> all_nonkey_heavy).  Because the existing tiers are already
nested supersets ending at ALL non-key heavy atoms, a pure "2-hop
neighbors" tier adds candidates only relative to the strict rigid tier —
we report both readings:
  * ``n_branches_with_any_2hop_candidate`` — a 2-hop topological pool is
    non-empty at all;
  * ``n_branches_2hop_gate_ok`` — that pool PASSES the frozen scatter gate
    (lambda1 > 0.01 A^2, lambda2/lambda1 > 1e-3), i.e. a fallback tier
    would actually be USABLE, not merely non-empty.

ADR-0026 constraint (recorded verbatim in the output): global-axis /
principal-axis fallback is FORBIDDEN; any new fallback requires ADR-0027
+ G5.1 fixtures + dual-denominator reporting (common eligible subset vs
expanded subset).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Final, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.contracts.key_atom import (  # noqa: E402
    REASON_GAUGE_UNRESOLVED, REASON_IDS)
from pes2ts_core.pesn.offline.data.key_atom_teacher import (  # noqa: E402
    compute_reference_block, reference_gate)

SCHEMA_VERSION: Final[str] = "pesn_v51_gauge_analysis_v1"

DEFAULT_TEACHER_ROOTS: Final[tuple[str, ...]] = (
    "outputs/pesn/v5/teacher/d0_0.05",
    "outputs/pesn/v5/teacher/d0_0.08",
    "outputs/pesn/v5/teacher/d0_0.12",
)
DEFAULT_FEATURE_ROOT: Final[str] = "outputs/pesn/v5/features/d0_0.05"
DEFAULT_SPLIT: Final[str] = "data/manifests/v5/split_manifest.json"
DEFAULT_OUT = (PROJECT_ROOT / "outputs/pesn/v5_1/diagnostics"
               / "gauge_analysis.json")

ADR_0026_CONSTRAINT: Final[str] = (
    "ADR-0026 §2.2: gauge_unresolved branches never fall back to a global "
    "axis or unverified full-atom registration. Any new fallback tier "
    "requires ADR-0027 + G5.1 fixtures + dual-denominator reporting "
    "(common eligible subset 5,088 vs expanded subset; the expanded "
    "result may never replace the common-subset primary conclusion).")

#: Hop structure proxy buckets (feature h_node_inv slots 8..12).
HOP_BINS: Final[tuple[str, ...]] = ("hop0", "hop1", "hop2", "hop3", "hop_gt3")


# ---------------------------------------------------------------------------
# pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def is_unresolved(reason_id) -> np.ndarray:
    """Boolean mask of gauge_unresolved rows (reason_id == 3)."""
    return np.asarray(reason_id, dtype=np.int64) == REASON_GAUGE_UNRESOLVED


def tier_candidate_sets(z: Sequence[int], start_bonds: Sequence,
                        key_rows: Sequence[int]) -> dict[str, np.ndarray]:
    """Non-key heavy atom rows per reference tier (frozen teacher rule)."""
    ref = compute_reference_block(z, start_bonds, key_rows)
    out = {"rigid_1_2hop": np.asarray(ref["tiers"][0]["rows"], dtype=np.int64)}
    for tier in ref["tiers"][1:]:
        out[str(tier["tier"])] = np.asarray(tier["rows"], dtype=np.int64)
    return out


def two_hop_candidates(z: Sequence[int], start_bonds: Sequence,
                       key_rows: Sequence[int]) -> np.ndarray:
    """Non-key HEAVY atoms within <=2 covalent hops of ANY key atom.

    This is the candidate pool a "second-order topological neighbor"
    fallback tier would draw from (topology only, no gate applied).
    """
    ref = compute_reference_block(z, start_bonds, key_rows)
    key = {int(r) for r in key_rows}
    z_arr = np.asarray(z, dtype=np.int64)
    n = len(z_arr)
    adj: list[list[int]] = [[] for _ in range(n)]
    for i, j, _order in start_bonds:
        adj[int(i)].append(int(j))
        adj[int(j)].append(int(i))
    hops = {r: 0 for r in key}
    frontier = sorted(key)
    while frontier:
        nxt = []
        for node in frontier:
            for nb in adj[node]:
                if nb not in hops:
                    hops[nb] = hops[node] + 1
                    nxt.append(nb)
        frontier = nxt
    rows = [i for i in range(n)
            if i not in key and z_arr[i] > 1 and hops.get(i, 99) <= 2]
    _ = ref  # tier sets intentionally not used: candidates are TOPLOGICAL
    return np.asarray(sorted(rows), dtype=np.int64)


def fallback_feasibility(z: Sequence[int], start_bonds: Sequence,
                         key_rows: Sequence[int],
                         pos_start: np.ndarray) -> dict:
    """Would a 2-hop topological fallback tier have USABLE candidates?

    Counts candidates outside the STRICT rigid tier (the tier the frozen
    pipeline tries first) and re-runs the frozen scatter gate on the
    2-hop pool.  Analysis only — nothing here changes labels.
    """
    tiers = tier_candidate_sets(z, start_bonds, key_rows)
    pool = two_hop_candidates(z, start_bonds, key_rows)
    rigid = tiers.get("rigid_1_2hop", np.zeros(0, dtype=np.int64))
    outside_rigid = np.asarray(
        [int(r) for r in pool if int(r) not in set(rigid.tolist())],
        dtype=np.int64)
    weights = np.zeros(len(z), dtype=np.float64)
    key = {int(r) for r in key_rows}
    # inverse-square hop weights identical to the teacher's rule
    adj: list[list[int]] = [[] for _ in range(len(z))]
    for i, j, _order in start_bonds:
        adj[int(i)].append(int(j))
        adj[int(j)].append(int(i))
    hops = {r: 0 for r in key}
    frontier = sorted(key)
    while frontier:
        nxt = []
        for node in frontier:
            for nb in adj[node]:
                if nb not in hops:
                    hops[nb] = hops[node] + 1
                    nxt.append(nb)
        frontier = nxt
    for r in pool:
        h = hops.get(int(r))
        if h is not None and h >= 1:
            weights[int(r)] = 1.0 / (1.0 + h) ** 2
    gate_ok, eigenvalues = reference_gate(
        np.asarray(pos_start, dtype=np.float64), pool, weights)
    rigid_set = set(int(r) for r in rigid)
    rigid_weights = np.asarray(
        [weights[i] if i in rigid_set else 0.0 for i in range(len(z))],
        dtype=np.float64)
    rigid_gate_ok, rigid_eigs = reference_gate(
        np.asarray(pos_start, dtype=np.float64), rigid, rigid_weights)
    return {
        "n_two_hop_candidates": int(pool.size),
        "n_candidates_outside_rigid_tier": int(outside_rigid.size),
        "has_any_2hop_candidate": bool(pool.size > 0),
        "adds_beyond_rigid_tier": bool(outside_rigid.size > 0),
        "two_hop_pool_gate_ok": bool(gate_ok),
        "two_hop_pool_eigenvalues_angstrom2": [float(v) for v in eigenvalues],
        "rigid_tier_gate_ok": bool(rigid_gate_ok),
        "rigid_tier_eigenvalues_angstrom2": [float(v) for v in rigid_eigs],
        "tier_row_counts": {k: int(v.size) for k, v in tiers.items()},
    }


def count_by(counter_like, key) -> int:
    """Key-safe counter increment helper (unit-tested)."""
    return int(counter_like.get(key, 0))


def summarize_distribution(rows: Sequence[dict]) -> dict:
    """Distribution of unresolved rows across the recorded dimensions.

    ``by_d0`` / ``by_fold`` / ``by_family`` count UNRESOLVED SAMPLE ROWS
    (weighted by ``n_unresolved_rows``); ``*_branches`` variants count
    distinct (reaction, branch) shards.
    """
    by_family = Counter()
    by_fold = Counter()
    by_d0 = Counter()
    by_family_branches = Counter()
    by_fold_branches = Counter()
    by_d0_branches = Counter()
    by_reason = Counter()
    for r in rows:
        n = int(r.get("n_unresolved_rows", 1))
        by_family[r["family"]] += n
        by_fold[str(r["fold"])] += n
        by_d0[f"{r['d0']:.2f}"] += n
        by_d0_branches[f"{r['d0']:.2f}"] += 1
        by_fold_branches[str(r["fold"])] += 1
        by_family_branches[r["family"]] += 1
        by_reason[str(r["reason_id"])] += n
    n_atoms = [r["n_atoms"] for r in rows if r["n_atoms"] is not None]
    n_keys = [r["n_keys"] for r in rows if r["n_keys"] is not None]
    branch_counts = Counter(f"{r['reaction_id']}.branch{r['branch']}"
                            for r in rows)
    return {
        "n_rows": sum(int(r.get("n_unresolved_rows", 1)) for r in rows),
        "n_shards": len(rows),
        "n_branches": len(branch_counts),
        "by_d0": dict(sorted(by_d0.items())),
        "by_d0_shards": dict(sorted(by_d0_branches.items())),
        "by_reason_id": dict(sorted(by_reason.items())),
        "by_fold": dict(sorted(by_fold.items())),
        "by_fold_shards": dict(sorted(by_fold_branches.items())),
        "by_family": dict(sorted(by_family.items())),
        "by_family_shards": dict(sorted(by_family_branches.items())),
        "n_atoms": {
            "min": (min(n_atoms) if n_atoms else None),
            "median": (float(np.median(n_atoms)) if n_atoms else None),
            "max": (max(n_atoms) if n_atoms else None)},
        "n_keys": {
            "min": (min(n_keys) if n_keys else None),
            "median": (float(np.median(n_keys)) if n_keys else None),
            "max": (max(n_keys) if n_keys else None)},
        "hop_histogram_nonkey_heavy": _hop_histogram(rows),
    }


def _hop_histogram(rows: Sequence[dict]) -> dict:
    totals = Counter()
    for r in rows:
        for k, v in (r.get("hop_counts") or {}).items():
            totals[k] += int(v)
    return {k: int(totals.get(k, 0)) for k in HOP_BINS}


# ---------------------------------------------------------------------------
# shard readers
# ---------------------------------------------------------------------------

def _load_task(task_root: Path, reaction_id: str):
    from pes2ts_core.pesn.contracts.key_atom import load_key_atom_task
    return load_key_atom_task(task_root / f"{reaction_id}.task.json")


def _hop_counts_for_sample(z: np.ndarray, key_rows: Sequence[int],
                           edge_index: np.ndarray,
                           edge_attr: np.ndarray) -> dict[str, int]:
    """Histogram of non-key heavy atoms by hop-from-key bin.

    Uses the feature shard's covalent edge flags (slots 21/22) for the
    adjacency; mirrors the model's BFS hop definition.
    """
    from pes2ts_core.pesn.contracts.key_atom import EDGE_SLOTS
    cov_s = EDGE_SLOTS["covalent_start"]
    cov_o = EDGE_SLOTS["covalent_other"]
    key = {int(r) for r in key_rows}
    n = len(z)
    adj: list[list[int]] = [[] for _ in range(n)]
    if edge_index.size:
        cov = (edge_attr[:, cov_s] > 0.5) | (edge_attr[:, cov_o] > 0.5)
        for e in np.nonzero(cov)[0]:
            a = int(edge_index[0, e])
            b = int(edge_index[1, e])
            adj[a].append(b)
            adj[b].append(a)
    hops = {r: 0 for r in key}
    frontier = sorted(key)
    while frontier:
        nxt = []
        for node in frontier:
            for nb in adj[node]:
                if nb not in hops:
                    hops[nb] = hops[node] + 1
                    nxt.append(nb)
        frontier = nxt
    counts = {k: 0 for k in HOP_BINS}
    for i in range(n):
        if i in key or int(z[i]) <= 1:
            continue
        h = hops.get(i)
        if h is None:
            counts["hop_gt3"] += 1
        elif h == 0:
            counts["hop0"] += 1
        elif h == 1:
            counts["hop1"] += 1
        elif h == 2:
            counts["hop2"] += 1
        elif h == 3:
            counts["hop3"] += 1
        else:
            counts["hop_gt3"] += 1
    return counts


def collect_unresolved_rows(teacher_roots: Sequence[Path],
                            feature_root: Path, task_root: Path,
                            families: dict[str, str],
                            folds: dict[str, int]) -> list[dict]:
    """One row per unresolved (reaction, branch, d0) with structure proxies."""
    rows: list[dict] = []
    for root in teacher_roots:
        d0 = float(root.name.split("d0_")[1])
        for path in sorted(root.glob("*.branch*.npz")):
            reaction_id = path.name.split(".branch")[0]
            branch = int(path.name.split(".branch")[1].split(".")[0])
            with np.load(path, allow_pickle=False) as d:
                reason = np.asarray(d["reason_id"], dtype=np.int64)
            mask = is_unresolved(reason)
            if not bool(mask.any()):
                continue
            group_id = reaction_id  # fit50: one reaction per group
            row = {
                "d0": d0,
                "reaction_id": reaction_id,
                "branch": branch,
                "family": families.get(group_id, "unknown"),
                "fold": folds.get(group_id),
                "reason_id": REASON_GAUGE_UNRESOLVED,
                "reason_name": REASON_IDS.get(REASON_GAUGE_UNRESOLVED),
                "n_unresolved_rows": int(mask.sum()),
                "n_rows": int(reason.size),
            }
            task = None
            try:
                task = _load_task(task_root, reaction_id)
            except Exception:
                task = None
            fpath = feature_root / path.name
            if fpath.is_file():
                with np.load(fpath, allow_pickle=False) as f:
                    sp = np.asarray(f["sample_ptr"], dtype=np.int64)
                    kp = np.asarray(f["key_ptr"], dtype=np.int64)
                    s0 = int(np.nonzero(mask)[0][0])
                    n_nodes = int(sp[s0 + 1] - sp[s0])
                    n_keys = int(kp[s0 + 1] - kp[s0])
                    key_rows = [int(v) - int(sp[s0])
                                for v in f["key_index"][kp[s0]:kp[s0 + n_keys]]]
                    e0 = int(f["edge_ptr"][s0])
                    e1 = int(f["edge_ptr"][s0 + 1])
                    edge_index = np.asarray(f["edge_index"][:, e0:e1],
                                            dtype=np.int64)
                    edge_attr = np.asarray(f["edge_attr"][e0:e1],
                                           dtype=np.float32)
                    z = np.asarray(f["z_atomic"][int(sp[s0]):int(sp[s0 + 1])],
                                   dtype=np.int64)
                    pos0 = np.asarray(f["pos0"][int(sp[s0]):int(sp[s0 + 1])],
                                      dtype=np.float64)
                row["n_atoms"] = n_nodes
                row["n_keys"] = n_keys
                row["hop_counts"] = _hop_counts_for_sample(z, key_rows,
                                                           edge_index,
                                                           edge_attr)
                if task is not None:
                    bonds = task.start_bonds(branch)
                    row["fallback"] = fallback_feasibility(
                        task.atomic_numbers, bonds, task.key_atom_rows, pos0)
            else:
                row["n_atoms"] = task.n_atoms if task is not None else None
                row["n_keys"] = (len(task.key_atom_rows)
                                 if task is not None else None)
            rows.append(row)
    return rows


def feasibility_summary(rows: Sequence[dict]) -> dict:
    """Aggregate fallback-feasibility counts over unresolved branches.

    The reference gate is a property of the START topology and is
    d0-INDEPENDENT, so the per-d0 duplicates are collapsed to one entry
    per (reaction, branch); disagreement across d0 (which would indicate a
    shard bug) is recorded rather than hidden.
    """
    by_branch: dict[tuple[str, int], list[dict]] = {}
    for r in rows:
        if "fallback" in r:
            by_branch.setdefault((r["reaction_id"], int(r["branch"])),
                                 []).append(r)
    per_branch = []
    n_disagree = 0
    for (rid, branch), group in sorted(by_branch.items()):
        first = group[0]["fallback"]
        for other in group[1:]:
            if (other["fallback"]["two_hop_pool_gate_ok"]
                    != first["two_hop_pool_gate_ok"]
                    or other["fallback"]["n_two_hop_candidates"]
                    != first["n_two_hop_candidates"]):
                n_disagree += 1
                break
        per_branch.append({
            "reaction_id": rid, "branch": branch,
            "family": group[0]["family"], "fold": group[0]["fold"],
            "d0_values": sorted({f"{g['d0']:.2f}" for g in group}),
            **first})
    return {
        "n_unresolved_branches_analyzed": len(per_branch),
        "n_unresolved_branch_entries_all_d0": sum(
            1 for r in rows if "fallback" in r),
        "n_branches_missing_structure": len(rows)
        - sum(1 for r in rows if "fallback" in r),
        "n_branches_cross_d0_disagree": n_disagree,
        "n_branches_with_any_2hop_candidate": sum(
            1 for b in per_branch if b["has_any_2hop_candidate"]),
        "n_branches_2hop_adds_beyond_rigid_tier": sum(
            1 for b in per_branch if b["adds_beyond_rigid_tier"]),
        "n_branches_2hop_pool_gate_ok": sum(
            1 for b in per_branch if b["two_hop_pool_gate_ok"]),
        "n_branches_rigid_tier_gate_ok": sum(
            1 for b in per_branch if b["rigid_tier_gate_ok"]),
        "per_branch": per_branch,
        "interpretation": (
            "a 2-hop topological tier is only USABLE when its candidate "
            "pool passes the frozen scatter gate (lambda1>0.01 A^2 and "
            "lambda2/lambda1>1e-3); non-empty pools that fail the gate "
            "cannot rescue the branch without relaxing the gate, which "
            "would itself need ADR-0027"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--teacher-roots", nargs="*",
                        default=list(DEFAULT_TEACHER_ROOTS))
    parser.add_argument("--feature-root", default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--task-root", default="outputs/pesn/v5/tasks")
    parser.add_argument("--split-manifest", default=DEFAULT_SPLIT)
    args = parser.parse_args(argv)

    split_path = PROJECT_ROOT / args.split_manifest
    split_doc = json.loads(split_path.read_text(encoding="utf-8"))
    families: dict[str, str] = {}
    for comp in split_doc.get("components") or []:
        fam = str(comp.get("family", "unknown"))
        for gid in comp.get("group_ids") or []:
            families[str(gid)] = fam
    folds = {str(g): int(f) for g, f in
             (split_doc.get("group_fold") or {}).items()}

    teacher_roots = [PROJECT_ROOT / p for p in args.teacher_roots]
    feature_root = PROJECT_ROOT / args.feature_root
    task_root = PROJECT_ROOT / args.task_root

    rows = collect_unresolved_rows(teacher_roots, feature_root, task_root,
                                   families, folds)
    distribution = summarize_distribution(rows)
    feasibility = feasibility_summary(rows)

    # per-fold unresolved row counts relative to the d0=0.05 denominator
    primary = [r for r in rows if abs(r["d0"] - 0.05) < 1e-9]
    per_fold_primary = Counter()
    for r in primary:
        per_fold_primary[str(r["fold"])] += int(
            r.get("n_unresolved_rows", 1))
    denominator_rows = 0
    for path in sorted((teacher_roots[0]).glob("*.branch*.npz")):
        with np.load(path, allow_pickle=False) as d:
            denominator_rows += int(np.asarray(d["eligible"]).size)

    doc = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "reason_id_analyzed": REASON_GAUGE_UNRESOLVED,
        "reason_name": REASON_IDS.get(REASON_GAUGE_UNRESOLVED),
        "primary_d0": 0.05,
        "d0_0.05_unresolved_rows": distribution["by_d0"].get("0.05"),
        "d0_0.05_total_rows": denominator_rows,
        "d0_0.05_unresolved_fraction": (
            distribution["by_d0"].get("0.05", 0) / denominator_rows
            if denominator_rows else None),
        "distribution": distribution,
        "per_fold_primary_d0": dict(sorted(per_fold_primary.items())),
        "fallback_feasibility": feasibility,
        "adr_0026_constraint": ADR_0026_CONSTRAINT,
        "implementation_note": (
            "analysis only: no fallback tier is implemented or applied by "
            "this script; any candidate design must go through ADR-0027 "
            "with G5.1 fixtures and dual-denominator reporting"),
        "provenance": {
            "teacher_roots": [str(p) for p in teacher_roots],
            "feature_root": str(feature_root),
            "split_manifest": str(split_path),
            "split_manifest_sha256": _sha256_file(split_path),
        },
    }
    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"[gauge] wrote {out}")
    print(f"  d0=0.05 unresolved: {doc['d0_0.05_unresolved_rows']}/"
          f"{denominator_rows} rows "
          f"({doc['d0_0.05_unresolved_fraction']:.4f})")
    print(f"  branches analyzed: {feasibility['n_unresolved_branches_analyzed']}"
          f"  with 2-hop candidates: "
          f"{feasibility['n_branches_with_any_2hop_candidate']}"
          f"  2-hop pool gate-ok: "
          f"{feasibility['n_branches_2hop_pool_gate_ok']}")
    return 0


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":
    raise SystemExit(main())
