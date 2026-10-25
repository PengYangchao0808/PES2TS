#!/usr/bin/env python3
"""V5.3-MP S0 finalize: pool freeze + split manifest + shards + label/resource
reports + the S0 gate summary.

Runs AFTER the S0-B build (``r3_v53mp_build_tables_h5.py all``).  Steps:

1. **pool freeze** — qualified pool = pool candidates whose curve built with
   ``curve_quality.quality_status == "ok"``; the frozen pool = the first
   ``--pool-target`` qualified groups in cohort rank order (default 20,000,
   the frozen 20k build; a 50k finalize passes 50,000); tiers are nested
   prefixes (2k ⊂ 5k ⊂ 20k ⊂ 50k — the 50k entry is emitted only when the
   pool reaches 50,000, so the default run still writes exactly the
   {2k, 5k, 20k} layout).  ``split_manifest_v53mp.json`` (schema
   ``v53mp_split_v1``) carries ``subsets.pool.tiers`` + ``subsets.dev``
   (SHARED dev pool — identical across every tier, so the 50k tier reuses
   the frozen dev set + dev pack) +
   ``subsets.confirm`` (ids only — confirm reactions were never read/built)
   + ``curve_folds`` for every built curve (pool/dev fold = position % 3;
   confirm listed at fold 4 — never present in the tables).
2. **curve_family.csv** — per-curve family labels (from S0-A).
3. **geometry shards** — :func:`write_geometry_shards` (S0-D fast path) +
   data_manifest regeneration over the full file set.
4. **S0-C/E label report** — ``endpoint_label_spec_v1`` over the full build
   (ShardedKeyAtomDatasetV3, all statuses): TS rows 0% positive; start rows
   0% positive; terminal rows positive iff d_end<=w_pos + edits; per-family
   coverage; degenerate curves.
5. **S0-D loader resource report** — startup time / peak RSS / per-curve
   random-access time / shard-vs-CSV equivalence on a seeded subset,
   against pre-registered limits (startup <= 900 s, RSS <= 24 GiB,
   equivalence exact on 200 sampled curves).
6. **s0_gate_report.json** — machine summary of S0-A/B/C/D/E/F.

Usage::

    python3 scripts/agent_ops/data_gate/r3_v53mp_finalize.py \
        --data outputs/pesn/v53mp/data \
        --cohort outputs/pesn/v53mp/cohort/v53mp_cohort_v1.json \
        --s0a outputs/pesn/v53mp/s0a
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA = "v53mp_split_v1"
POOL_TARGET = 20_000
#: nested-prefix tier sizes over the frozen pool in cohort rank order.
#: The 50k entry is EMITTED ONLY when the frozen pool actually reaches
#: 50,000 curves (``size <= len(frozen_pool)``), so the default 20k
#: finalize keeps emitting exactly {2k, 5k, 20k} — byte-identical split
#: manifests vs the pre-50k code.  A 50k finalize passes
#: ``--pool-target 50000`` and emits all four tiers; because the ranking
#: is per-group deterministic and the 65k cohort's ranked list is a
#: prefix extension of the 26k one, the nested-prefix property
#: (2k ⊂ 5k ⊂ 20k ⊂ 50k) holds ACROSS builds by construction.
TIER_SIZES = {"2k": 2_000, "5k": 5_000, "20k": 20_000, "50k": 50_000}

# pre-registered S0-D limits (plan §4.4 hard gate)
LIMIT_STARTUP_S = 900.0
LIMIT_RSS_GIB = 24.0
LIMIT_EQUIVALENCE_CURVES = 200


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. pool freeze + split manifest
# ---------------------------------------------------------------------------


def freeze_pool(repo: Path, data_dir: Path, cohort_path: Path,
                s0a_dir: Path, pool_target: int = POOL_TARGET,
                tier_sizes: Optional[Dict[str, int]] = None) -> dict:
    """Freeze the qualified pool + nested-prefix tiers into the split manifest.

    ``pool_target`` (default 20,000 — the frozen 20k build) sizes the
    frozen pool; a 50k finalize passes 50,000.  ``tier_sizes`` (default
    :data:`TIER_SIZES`) decides which tier entries are emitted — a tier is
    written only when its size fits the frozen pool, so the default run
    emits exactly {2k, 5k, 20k} (byte-identical to the pre-50k manifest
    layout, modulo timestamps).  ``pool_target`` must reach the 2k floor.
    """
    pool_target = int(pool_target)
    if pool_target < min((tier_sizes or TIER_SIZES).values()):
        raise ValueError(
            "pool_target must be >= {} (the smallest tier), got {}".format(
                min((tier_sizes or TIER_SIZES).values()), pool_target))
    sizes = dict(TIER_SIZES if tier_sizes is None else tier_sizes)
    repo = repo.resolve()
    cohort_path = cohort_path.resolve()
    s0a_dir = s0a_dir.resolve()
    cohort = _read_json(cohort_path)
    pool_candidates = [str(c) for c in cohort["pool_candidates"]]
    dev_candidates = [str(c) for c in cohort["dev"]]
    confirm_ids = [str(c) for c in cohort["confirm"]]

    # which pool candidates built with quality_status == ok?
    quality: Dict[str, str] = {}
    with open(data_dir / "curve_quality.csv", "r", encoding="utf-8",
              newline="") as fh:
        for row in csv.DictReader(fh):
            quality[row["curve_id"]] = row["quality_status"]
    dev_ids = [c for c in dev_candidates if c in quality]
    qualified = [c for c in pool_candidates if quality.get(c) == "ok"]
    frozen_pool = qualified[:pool_target]
    n_short = len(qualified) - len(frozen_pool)

    folds: Dict[str, int] = {}
    for i, cid in enumerate([c for c in pool_candidates if c in quality]):
        folds[cid] = i % 3
    for i, cid in enumerate(dev_ids):
        folds[cid] = i % 3
    for cid in confirm_ids:
        folds[cid] = 4          # confirm seal fold (never in the tables)

    tiers = {
        name: frozen_pool[:size] for name, size in sizes.items()
        if size <= len(frozen_pool)
    }
    tier_chain = " ⊂ ".join(sorted(tiers, key=lambda n: sizes[n]))
    split_doc = {
        "schema": SCHEMA,
        "schema_version": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed_key": cohort["seed_key"],
        "parents": {
            "cohort": {"path": str(cohort_path.relative_to(repo)),
                       "sha256": sha256_file(cohort_path)},
            "g3p1_split": cohort["parent_split"],
            "s0a_qualification": cohort["s0a_qualification"],
        },
        "subsets": {
            "pool": {name: list(ids) for name, ids in tiers.items()},
            "dev": dev_ids,
            "confirm": confirm_ids,
        },
        "curve_folds": folds,
        "counts": {
            "pool_qualified": len(qualified),
            # pool_20k kept verbatim for reader compat: it is the FROZEN
            # POOL (20,000 at the default target; 50,000 for a 50k
            # finalize).  pool_frozen carries the same number under the
            # target-neutral name.
            "pool_20k": len(frozen_pool),
            "pool_frozen": len(frozen_pool),
            "pool_overflow_unqualified_or_margin": (
                len(pool_candidates) - len(frozen_pool)),
            "dev": len(dev_ids),
            "confirm_listed_never_built": len(confirm_ids),
        },
        "rules": [
            "tiers are nested prefixes of the frozen pool in cohort rank "
            "order ({})".format(tier_chain),
            "confirm ids are listed only; confirm reactions were never read "
            "or built in this campaign stage",
            "fold: built curves carry position % 3; confirm carries 4",
        ],
        "frozen": True,
    }
    (data_dir / "split_manifest_v53mp.json").write_text(
        json.dumps(split_doc, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    # curve_family.csv
    import pyarrow.parquet as pq
    fam_rows = pq.read_table(
        s0a_dir / "eligible_reactions.parquet",
        columns=["reaction_id", "family_labels", "edit_family_id"]).to_pylist()
    fam_of = {str(r["reaction_id"]): r for r in fam_rows}
    with open(data_dir / "curve_family.csv", "w", encoding="utf-8",
              newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["curve_id", "family_labels", "edit_family_id"])
        for cid in sorted(quality):
            fam = fam_of.get(cid) or {}
            w.writerow([cid, (fam or {}).get("family_labels", ""),
                        (fam or {}).get("edit_family_id", "")])
    return {
        "split_doc": split_doc,
        "n_qualified": len(qualified),
        "n_pool": len(frozen_pool),
        "n_dev": len(dev_ids),
        "shortfall_dropped": n_short,
        "quality_counts": dict(_count(quality)),
    }


def _count(quality: Dict[str, str]):
    from collections import Counter
    return Counter(quality.values()).items()


# ---------------------------------------------------------------------------
# 2. shards + manifest regeneration
# ---------------------------------------------------------------------------


def build_shards_and_manifest(data_dir: Path) -> dict:
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (
        write_geometry_shards,
    )
    index_path = data_dir / "shard_index_v53mp.json"
    shard_files = sorted(p.name for p in data_dir.glob("shard_*.npz"))
    if index_path.is_file() and shard_files:
        # idempotent resume: verify the recorded shard shas, never rebuild
        index = _read_json(index_path)
        recorded = {str(s["path"]): str(s["sha256"])
                    for s in index.get("shards", [])}
        missing = [name for name in recorded
                   if not (data_dir / name).is_file()]
        bad = [name for name in recorded
               if (data_dir / name).is_file()
               and sha256_file(data_dir / name) != recorded[name]]
        if missing or bad:
            raise SystemExit(
                f"shard_index_v53mp.json drift: missing={missing[:3]} "
                f"sha-mismatch={bad[:3]} — rebuild shards explicitly")
        shard_wall = 0.0
    else:
        t0 = time.time()
        write_geometry_shards(data_dir)
        shard_wall = time.time() - t0
        shard_files = sorted(p.name for p in data_dir.glob("shard_*.npz"))
    # regenerate the data manifest over the full file set
    names = [
        "irc_curve.csv", "irc_node.csv", "irc_edge.csv", "reaction_intent.csv",
        "curve_quality.csv", "sample_view.csv", "branch_diagnostics.csv",
        "curve_family.csv", "shard_index_v53mp.json",
    ]
    names += shard_files
    files = [{"path": name, "sha256": sha256_file(data_dir / name)}
             for name in names]
    import hashlib
    dataset_sha = hashlib.sha256(json.dumps(
        [[f["path"], f["sha256"]] for f in files],
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    old = _read_json(data_dir / "data_manifest.json")
    manifest = dict(old)
    manifest["files"] = files
    manifest["dataset_sha256"] = dataset_sha
    manifest["regenerated_at_finalize"] = (
        datetime.now(timezone.utc).isoformat())
    manifest["note"] = ("regenerated at S0 finalize: adds curve_family.csv + "
                        "geometry shards (shard_*.npz + shard_index) to the "
                        "assembled six tables")
    (data_dir / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n",
        encoding="utf-8")
    return {
        "shard_wall_clock_s": round(shard_wall, 1),
        "n_shard_files": len(shard_files),
        "dataset_sha256": dataset_sha,
    }


# ---------------------------------------------------------------------------
# 3. S0-C/E label report
# ---------------------------------------------------------------------------


def label_report(data_dir: Path) -> dict:
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (
        LABEL_STATUSES,
        ShardedKeyAtomDatasetV3,
    )
    from pes2ts_core.pesn.offline.data.key_atom_endpoint_labels_v1 import (
        build_endpoint_labels_spec_v1,
    )
    from pes2ts_core.pesn.offline.data.key_atom_label_cache_v53mp import (
        save_label_cache,
    )
    fam_of: Dict[str, str] = {}
    with open(data_dir / "curve_family.csv", "r", encoding="utf-8",
              newline="") as fh:
        for row in csv.DictReader(fh):
            fam_of[row["curve_id"]] = row["family_labels"]
    t0 = time.time()
    dataset = ShardedKeyAtomDatasetV3(
        data_dir, label_statuses=LABEL_STATUSES,
        split_manifest=data_dir / "split_manifest_v53mp.json")
    labels = dataset.v52_labels()
    table = build_endpoint_labels_spec_v1(dataset, labels,
                                          curve_families=fam_of)
    report = table.report()
    # write the deterministic label cache (bound to the CURRENT data
    # manifest) so the training runs never recompute labels at 20k scale
    save_label_cache(data_dir, labels, table,
                     dataset_manifest_sha256=dataset.manifest_sha256)
    report["wall_clock_s"] = round(time.time() - t0, 1)
    report["hard_gates"] = {
        # plan §4.5: TS 帧及 TS 邻域 0% 正例（degenerate/conflict 的 TS 行
        # 按 spec 为 Mask — 正例为 0 才是硬门，Mask 不计违规）
        "ts_rows_zero_positive": not any(
            lab.endpoint_label == 1
            for lab in table.by_sample.values() if lab.ts_row),
        "start_rows_zero_positive": not any(
            lab.endpoint_label == 1
            for lab in table.by_sample.values() if lab.start_row),
        "effective_positive_coverage": report.get(
            "effective_positive_coverage"),
    }
    (data_dir / "endpoint_label_report_v53mp.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# 4. S0-D loader resource report
# ---------------------------------------------------------------------------


def loader_resource_report(data_dir: Path) -> dict:
    import resource
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (
        LABEL_STATUSES,
        KeyAtomDatasetV3,
        ShardedKeyAtomDatasetV3,
    )
    t0 = time.time()
    dataset = ShardedKeyAtomDatasetV3(
        data_dir, label_statuses=LABEL_STATUSES,
        split_manifest=data_dir / "split_manifest_v53mp.json")
    startup_s = time.time() - t0
    rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2

    # random-access timing + shard-vs-CSV equivalence on a seeded subset;
    # the CSV side is a TARGETED stream over irc_node rows of the probed
    # curves only (a full in-memory KeyAtomDatasetV3 load at 26k scale would
    # not fit RAM — that is precisely why the sharded loader exists).
    curve_ids = sorted(dataset._curves)
    rng = random.Random(20261021)
    probe = rng.sample(curve_ids, min(LIMIT_EQUIVALENCE_CURVES,
                                      len(curve_ids)))
    t0 = time.time()
    for cid in probe:
        nodes = dataset._node_index[cid]
        for node in sorted(nodes):
            dataset.frame_xyz(cid, node)
    ra_s = (time.time() - t0) / max(1, len(probe))

    # targeted CSV reference: parse irc_node rows of the probed curves only
    probe_set = set(probe)
    csv_frames: Dict[str, Dict[int, np.ndarray]] = {c: {} for c in probe}
    path = data_dir / "irc_node.csv"
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cid = row["curve_id"]
            if cid not in probe_set:
                continue
            csv_frames[cid][int(row["node_index"])] = (
                KeyAtomDatasetV3._parse_frame_xyz(row))
    eq_ok = True
    for cid in probe:
        for node, arr in csv_frames[cid].items():
            a = np.asarray(dataset.frame_xyz(cid, node), dtype=np.float64)
            if a.shape != arr.shape or not np.array_equal(a, arr):
                eq_ok = False
                break
        if not eq_ok:
            break
    report = {
        "startup_s": round(startup_s, 1),
        "peak_rss_gib": round(rss_gib, 2),
        "random_access_s_per_curve": round(ra_s, 3),
        "n_curves_probed": len(probe),
        "shard_csv_equivalence_exact": eq_ok,
        "limits": {"startup_s": LIMIT_STARTUP_S,
                   "rss_gib": LIMIT_RSS_GIB,
                   "equivalence_curves": LIMIT_EQUIVALENCE_CURVES},
        "passed": bool(startup_s <= LIMIT_STARTUP_S
                       and rss_gib <= LIMIT_RSS_GIB and eq_ok),
    }
    (data_dir / "loader_resource_report_v53mp.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="r3_v53mp_finalize")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--data", default="outputs/pesn/v53mp/data")
    parser.add_argument("--cohort",
                        default="outputs/pesn/v53mp/cohort/v53mp_cohort_v1.json")
    parser.add_argument("--s0a", default="outputs/pesn/v53mp/s0a")
    parser.add_argument("--pool-target", type=int, default=POOL_TARGET,
                        help="frozen pool size (default {} = the 20k build; "
                             "a 50k finalize passes 50000 and the split "
                             "manifest then also emits the 50k tier "
                             "entry)".format(POOL_TARGET))
    parser.add_argument("--skip-labels", action="store_true")
    parser.add_argument("--skip-resource", action="store_true")
    args = parser.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    data_dir = Path(args.data).resolve()

    summary: Dict[str, Any] = {}
    t0 = time.time()
    try:
        pool = freeze_pool(repo, data_dir, Path(args.cohort), Path(args.s0a),
                           pool_target=args.pool_target)
    except ValueError as exc:
        print(f"FATAL: {exc}", flush=True)
        return 2
    summary["pool_freeze"] = {
        "n_qualified": pool["n_qualified"],
        "n_pool_20k": pool["n_pool"],
        "pool_target": int(args.pool_target),
        "n_dev": pool["n_dev"],
        "quality_counts": dict(pool["quality_counts"]),
    }
    print(json.dumps(summary["pool_freeze"], sort_keys=True), flush=True)
    if pool["n_pool"] < int(args.pool_target):
        print(f"FATAL: qualified pool {pool['n_qualified']} < "
              f"{int(args.pool_target)}", flush=True)
        return 2

    summary["shards"] = build_shards_and_manifest(data_dir)
    print(json.dumps(summary["shards"], sort_keys=True), flush=True)

    if not args.skip_labels:
        report = label_report(data_dir)
        summary["labels"] = {
            "n_rows": report.get("n_rows"),
            "n_pos": report.get("n_pos"),
            "n_neg": report.get("n_neg"),
            "n_mask": report.get("n_mask"),
            "effective_positive_coverage": report.get(
                "effective_positive_coverage"),
            "hard_gates": report.get("hard_gates"),
        }
        print(json.dumps(summary["labels"], sort_keys=True), flush=True)

    if not args.skip_resource:
        summary["loader_resource"] = loader_resource_report(data_dir)
        print(json.dumps(summary["loader_resource"], sort_keys=True),
              flush=True)

    summary["wall_clock_s"] = round(time.time() - t0, 1)
    (data_dir / "s0_finalize_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
