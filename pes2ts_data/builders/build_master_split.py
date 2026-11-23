#!/usr/bin/env python3
"""Build the single master split table for the PES2TS datastore.

Outputs (write scope: <out-dir>, default /pes2ts_datastore/01_master_index/):
  - master_split.parquet          1 row per curve_id (built curves) + 1 row per
                                  sealed confirm_2000 reaction id (no curve artifact).
  - master_split_manifest.json    source sha256, counts, decisions, validations,
                                  gaps and open_questions.

Inputs are READ-ONLY:
  repo (Windows/WSL repo):
    data/manifests/split_manifest.json            L0 G0 official split manifest
    data/manifests/split_ids_g0_v1.csv            L0 (reaction_id,split)
    data/manifests/pesn_irc_split_ids_v1.json     L1 parent split (subsets)
    data/manifests/v53mp_split_50k_ids_v1.json    L3 50k split (pool tiers/dev/confirm/curve_folds)
    data/manifests/v53mp_cohort_50k_ids_v1.json   L3 cohort 50k ledger
    data/manifests/v53mp_cohort_20k_ids_v1.json   L3 cohort 20k ledger
    data/manifests/split_registry_v1.json         sha/count baseline
    data/interim/split_assignment.parquet         L0 source artifact
    .slim/deepwork/v55-v56/idx_stats.json         pre-computed stats (cross-check only)
  WSL data tree:
    /pes2ts_data/v53mp_50k_data/idx_v2/curves.json
    /pes2ts_data/v53mp_50k_data/curve_family.csv
    /pes2ts_data/v53mp_50k_data/idx_v2/intents/{curve_idx,direction}.npy
    /pes2ts_data/v53mp_50k_data/staging_chunks/*.npz   (enumerated; 12 sampled)
    /pes2ts_data/v53mp_50k_data/split_manifest_v53mp.json (verbatim copy check)
    /pes2ts_data/v53mp_50k_data/data_manifest.json
    /pes2ts_data/v53mp_data/idx_v2/curves.json    (legacy 20k tree, custody check)

NOTHING under /pes2ts_data is modified. No git actions. No docs/ edits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = "pes2ts_master_split_v1"
MANIFEST_SCHEMA = "pes2ts_master_split_manifest_v1"

# ---- 口径 constants (recorded verbatim in the manifest) --------------------
FAMILY_OOD_K = 4
FAMILY_MIN_COUNT = 200
FAMILY_EXCLUDE = {"other"}
FAMILY_OOD_SEED = "pes2ts:family_ood:v1:"
SIZE_OOD_RULE = "strict_gt_q3"  # size_ood = n_atoms > Q3(n_atoms over built curves)
L0_LABELS = ["train", "valid", "test"]
L1_LABELS = ["train_fit", "train_dev", "train_cal", "family_holdout",
             "isolated_conflict", "none"]
V5_ROLES = ["pool", "dev_589", "confirm_2000", "unused"]
SCALE_TIERS = ["2k", "5k", "20k", "50k", "none"]
EVAL_SUITES = ["id_dev", "family_ood", "scaffold_ood", "size_ood", "none"]
ACCESS_POLICIES = ["open", "one_shot_dev", "sealed_never_read"]
ROW_KINDS = ["curve", "sealed_confirm_id"]

LOG = logging.getLogger("build_master_split")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def file_meta(path: Path, layer: str | None = None, rows: int | None = None,
              role: str | None = None) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False, "layer": layer, "role": role}
    m = {"path": str(path), "exists": True, "bytes": path.stat().st_size,
         "sha256": sha256_file(path)}
    if layer:
        m["layer"] = layer
    if rows is not None:
        m["rows"] = rows
    if role:
        m["role"] = role
    return m


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dict_array(values, labels=None, index_type=pa.int32()):
    """Build a deterministic arrow DictionaryArray (pandas reads as category).

    index_type is int32 to match what parquet round-trips (parquet dictionary
    encoding does not preserve a narrower arrow index type).
    """
    labels = list(labels) if labels is not None else sorted(set(values))
    pos = {lab: i for i, lab in enumerate(labels)}
    missing = [v for v in set(values) if v not in pos]
    if missing:
        raise RuntimeError(f"dictionary values outside declared domain: {missing}")
    idx = np.fromiter((pos[v] for v in values), dtype=np.int8, count=len(values))
    return pa.DictionaryArray.from_arrays(
        pa.array(idx, type=index_type), pa.array(labels, type=pa.string()))


def read_l0(path: Path) -> tuple[dict, dict]:
    m: dict[str, str] = {}
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        if rd.fieldnames != ["reaction_id", "split"]:
            raise RuntimeError(f"unexpected L0 header {rd.fieldnames}")
        for row in rd:
            m[row["reaction_id"]] = row["split"]
    counts: dict[str, int] = {}
    for v in m.values():
        counts[v] = counts.get(v, 0) + 1
    return m, counts


def read_l1(path: Path) -> tuple[dict, dict]:
    q = load_json(path)
    mapping: dict[str, str] = {}
    for key, label in (("train_fit", "train_fit"), ("train_dev", "train_dev"),
                       ("train_calibration", "train_cal")):
        for rid in q["subsets"][key]:
            mapping[rid] = label
    for rid in q["family_holdout"]["reaction_ids"]:
        mapping.setdefault(rid, "family_holdout")
    # isolated conflicts overwrite (they are known cross-split groups)
    for item in q["isolated_conflicts"]:
        for rid in item["members"]:
            mapping[rid] = "isolated_conflict"
    meta = {
        "subset_sizes": {k: len(v) for k, v in q["subsets"].items()},
        "family_holdout_groups": len(q["family_holdout"]["group_ids"]),
        "family_holdout_reactions": len(q["family_holdout"]["reaction_ids"]),
        "isolated_conflict_groups": len(q["isolated_conflicts"]),
        "isolated_conflict_reactions": len(
            {m for it in q["isolated_conflicts"] for m in it["members"]}),
        "development_exposure": q.get("development_exposure"),
        "seed": q.get("seed"),
        "schema_version": q.get("schema_version"),
    }
    return mapping, meta


def dict_values(seq) -> list:
    return list(dict.fromkeys(seq))


# ---------------------------------------------------------------------------
# main build
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", default=None,
                    help="repo root (default: two levels above this script)")
    ap.add_argument("--data-root", default="/pes2ts_data",
                    help="WSL native data root (read-only)")
    ap.add_argument("--out-dir", default="/pes2ts_datastore/01_master_index",
                    help="output dir (default: /pes2ts_datastore/01_master_index)")
    ap.add_argument("--log-path", default="/tmp/pes2ts_build_master_split.log")
    ap.add_argument("--strict-curve-only", action="store_true",
                    help="only built curves -> 64,195 rows (drops 2,000 sealed "
                         "confirm id placeholders, i.e. sealed rows=0)")
    ap.add_argument("--size-ood-rule", choices=["strict_gt_q3", "ge_q3"],
                    default=SIZE_OOD_RULE)
    ap.add_argument("--family-ood-k", type=int, default=FAMILY_OOD_K)
    ap.add_argument("--skip-staging-scan", action="store_true",
                    help="skip enumerating staging_chunks (debug/fast path)")
    args = ap.parse_args(argv)

    started = time.time()
    started_at = utc_now()

    if args.repo_root:
        repo = Path(args.repo_root).resolve()
    else:
        repo = Path(__file__).resolve().parents[2]
    data_root = Path(args.data_root)
    d50 = data_root / "v53mp_50k_data"
    d20 = data_root / "v53mp_data"
    out_dir = Path(args.out_dir)
    parquet_path = out_dir / "master_split.parquet"
    manifest_path = out_dir / "master_split_manifest.json"

    # write-scope guard: never write outside <out-dir> / /tmp
    out_resolved = out_dir.resolve()
    parquet_resolved = parquet_path.resolve()
    manifest_resolved = manifest_path.resolve()
    allowed = [out_resolved, Path("/tmp").resolve()]
    for target in (parquet_resolved, manifest_resolved):
        if not any(target.is_relative_to(a) for a in allowed):
            raise SystemExit(f"refusing to write outside allowed scope: {target}")
    if out_resolved.is_relative_to(data_root.resolve()):
        raise SystemExit("refusing to write inside /pes2ts_data (read-only input tree)")

    # logging
    log_path = Path(args.log_path)
    if not log_path.resolve().is_relative_to(Path("/tmp").resolve()):
        raise SystemExit(f"log path must be under /tmp: {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    LOG.addHandler(fh)
    LOG.addHandler(sh)
    LOG.info("=" * 78)
    LOG.info("build_master_split start repo=%s data_root=%s out_dir=%s",
             repo, data_root, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------- load inputs (read-only) -------------
    man = repo / "data" / "manifests"
    p_l0_csv = man / "split_ids_g0_v1.csv"
    p_l0_manifest = man / "split_manifest.json"
    p_l1 = man / "pesn_irc_split_ids_v1.json"
    p_split50 = man / "v53mp_split_50k_ids_v1.json"
    p_cohort50 = man / "v53mp_cohort_50k_ids_v1.json"
    p_cohort20 = man / "v53mp_cohort_20k_ids_v1.json"
    p_registry = man / "split_registry_v1.json"
    p_l0_parquet = repo / "data" / "interim" / "split_assignment.parquet"
    p_idx_stats = repo / ".slim" / "deepwork" / "v55-v56" / "idx_stats.json"
    p_curves50 = d50 / "idx_v2" / "curves.json"
    p_family = d50 / "curve_family.csv"
    p_intent_curve = d50 / "idx_v2" / "intents" / "curve_idx.npy"
    p_intent_dir = d50 / "idx_v2" / "intents" / "direction.npy"
    p_split_manifest_wsl = d50 / "split_manifest_v53mp.json"
    p_data_manifest_wsl = d50 / "data_manifest.json"
    p_curves20 = d20 / "idx_v2" / "curves.json"

    for p in (p_l0_csv, p_l1, p_split50, p_registry, p_curves50, p_family):
        if not p.exists():
            raise SystemExit(f"required input missing: {p}")

    l0, l0_counts = read_l0(p_l0_csv)
    l1_map, l1_meta = read_l1(p_l1)
    split50 = load_json(p_split50)
    cohort50 = load_json(p_cohort50)
    cohort20 = load_json(p_cohort20)
    registry = load_json(p_registry)
    idx_stats = load_json(p_idx_stats)
    curves50 = load_json(p_curves50)
    LOG.info("loaded inputs: L0=%d, L1=%d ids, split50 curve_folds=%d",
             len(l0), len(l1_map), len(split50["curve_folds"]))

    # curves + family
    built_ids = list(curves50["curve_ids"])
    if len(set(built_ids)) != len(built_ids):
        raise RuntimeError("duplicate curve_id in curves.json")
    curve_rec = {cid: cu for cid, cu in zip(curves50["curve_ids"], curves50["curves"])}
    n_atoms = {cid: len(cu["atomic_numbers"]) for cid, cu in curve_rec.items()}

    fam_label: dict[str, str] = {}
    family_rows = 0
    with open(p_family, newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            family_rows += 1
            fam_label[row["curve_id"]] = row["family_labels"]

    # intents: direction availability per curve
    ci = np.load(p_intent_curve)
    cdir = np.load(p_intent_dir)
    intents_per_curve = np.bincount(ci)
    dir_sets: dict[int, set] = {}
    for cu, dv in zip(ci, cdir):
        dir_sets.setdefault(int(cu), set()).add(int(dv))

    # L3 / L0/L1 sets
    built_set = set(built_ids)
    cf = split50["curve_folds"]
    subsets = split50["subsets"]
    tier_sets = {k: set(v) for k, v in subsets["pool"].items()}
    tier_order = ["2k", "5k", "20k", "50k"]
    dev589 = set(subsets["dev"])
    confirm = set(subsets["confirm"])
    confirm_from_folds = {cid for cid, fold in cf.items() if fold == 4}
    cohort_pool = set(cohort50["pool_candidates"])
    cohort_dev = set(cohort50["dev"])
    cohort_confirm = set(cohort50["confirm"])

    # 20k legacy tree custody check
    curves20_count = None
    curves20_subset = None
    if p_curves20.exists():
        c20 = load_json(p_curves20)
        ids20 = set(c20["curve_ids"])
        curves20_count = len(ids20)
        curves20_subset = len(ids20 - built_set) == 0

    # ------------- 口径 assignments -------------
    def entry_tier(cid: str) -> str:
        for t in tier_order:
            if cid in tier_sets[t]:
                return t
        return "none"

    def l1_of(rid: str) -> str:
        return l1_map.get(rid, "none")

    # family OOD selection (deterministic hash over eligible families)
    fam_counts: dict[str, int] = {}
    for cid in built_ids:
        lab = fam_label.get(cid, "unassigned")
        fam_counts[lab] = fam_counts.get(lab, 0) + 1
    eligible = sorted(f for f, n in fam_counts.items()
                      if f not in FAMILY_EXCLUDE and f != "unassigned"
                      and n >= FAMILY_MIN_COUNT)
    fam_hash = {f: hashlib.sha256((FAMILY_OOD_SEED + f).encode()).hexdigest()
                for f in eligible}
    family_ood_families = sorted(eligible, key=lambda f: (fam_hash[f], f))[:args.family_ood_k]
    family_ood_set = {cid for cid in built_ids if fam_label.get(cid) in family_ood_families}

    # size OOD threshold
    n_atoms_arr = np.array([n_atoms[c] for c in built_ids])
    q3 = float(np.percentile(n_atoms_arr, 75))
    if args.size_ood_rule == "strict_gt_q3":
        size_cand = {c for c, n in zip(built_ids, n_atoms_arr) if n > q3}
    else:
        size_cand = {c for c, n in zip(built_ids, n_atoms_arr) if n >= q3}
    q3_ge_count = int((n_atoms_arr >= q3).sum())

    # suites with priority: id_dev > family_ood > size_ood > none
    id_dev_set = built_set & dev589
    family_ood_set -= id_dev_set
    size_ood_set = size_cand - id_dev_set - family_ood_set
    scaffold_ood_set: set[str] = set()  # no scaffold labels available (gap)

    def eval_of(cid: str) -> str:
        if cid in id_dev_set:
            return "id_dev"
        if cid in family_ood_set:
            return "family_ood"
        if cid in scaffold_ood_set:
            return "scaffold_ood"
        if cid in size_ood_set:
            return "size_ood"
        return "none"

    def access_of(row_kind: str, cid: str, l0v: str, ev: str) -> str:
        if row_kind == "sealed_confirm_id" or l0v == "test":
            return "sealed_never_read"
        if ev != "none":
            return "one_shot_dev"
        return "open"

    pos = {cid: i for i, cid in enumerate(built_ids)}
    rows = []
    # built curves, sorted by curve_id (deterministic)
    for cid in sorted(built_ids):
        i = pos[cid]
        l0v = l0.get(cid, "missing")
        ds = dir_sets.get(i, set())
        if ds == {0, 1}:
            direction = "forward+reverse"
        elif ds == {0}:
            direction = "forward"
        elif ds == {1}:
            direction = "reverse"
        else:
            direction = "unknown"
        ev = eval_of(cid)
        rows.append({
            "curve_id": cid,
            "reaction_id": cid,
            "row_kind": "curve",
            "curve_built": True,
            "direction_available": direction,
            "n_intents": int(intents_per_curve[i]),
            "n_atoms": int(n_atoms[cid]),
            "n_irc_nodes": int(curve_rec[cid]["n_nodes"]),
            "ts_index": int(curve_rec[cid]["ts_index"]),
            "family_label": fam_label.get(cid, "unassigned"),
            "l0_official": l0v,
            "l1_parent": l1_of(cid),
            "v5_fold": int(cf.get(cid, -1)),
            "v5_role": ("pool" if cid in tier_sets["50k"] else
                        "dev_589" if cid in dev589 else "unused"),
            "scale_tier": entry_tier(cid),
            "in_pool_2k": cid in tier_sets["2k"],
            "in_pool_5k": cid in tier_sets["5k"],
            "in_pool_20k": cid in tier_sets["20k"],
            "in_pool_50k": cid in tier_sets["50k"],
            "eval_suite": ev,
            "access_policy": access_of("curve", cid, l0v, ev),
            "has_xtb_coverage": False,
            "build_source": "v53mp_50k_data/idx_v2/curves.json",
        })
    # sealed confirm placeholders (id-only, never built), sorted
    if not args.strict_curve_only:
        for cid in sorted(confirm):
            l0v = l0.get(cid, "missing")
            rows.append({
                "curve_id": cid,
                "reaction_id": cid,
                "row_kind": "sealed_confirm_id",
                "curve_built": False,
                "direction_available": "unknown",
                "n_intents": 0,
                "n_atoms": -1,
                "n_irc_nodes": -1,
                "ts_index": -1,
                "family_label": "unassigned",
                "l0_official": l0v,
                "l1_parent": l1_of(cid),
                "v5_fold": int(cf.get(cid, -1)),
                "v5_role": "confirm_2000",
                "scale_tier": "none",
                "in_pool_2k": False,
                "in_pool_5k": False,
                "in_pool_20k": False,
                "in_pool_50k": False,
                "eval_suite": "none",
                "access_policy": "sealed_never_read",
                "has_xtb_coverage": False,
                "build_source": "v53mp_split_50k_ids_v1.json#subsets.confirm",
            })

    n_rows = len(rows)
    LOG.info("rows built: %d (built curves=%d, sealed confirm ids=%d)",
             n_rows, len(built_ids), 0 if args.strict_curve_only else len(confirm))

    # ------------- arrow table -------------
    cols = {k: [r[k] for r in rows] for k in rows[0]}
    schema = pa.schema([
        pa.field("curve_id", pa.string(), nullable=False),
        pa.field("reaction_id", pa.string(), nullable=False),
        pa.field("row_kind", pa.dictionary(pa.int32(), pa.string()), nullable=False),
        pa.field("curve_built", pa.bool_(), nullable=False),
        pa.field("direction_available", pa.dictionary(pa.int32(), pa.string()), nullable=False),
        pa.field("n_intents", pa.int16(), nullable=False),
        pa.field("n_atoms", pa.int16(), nullable=False),
        pa.field("n_irc_nodes", pa.int16(), nullable=False),
        pa.field("ts_index", pa.int16(), nullable=False),
        pa.field("family_label", pa.dictionary(pa.int32(), pa.string()), nullable=False),
        pa.field("l0_official", pa.dictionary(pa.int32(), pa.string()), nullable=False),
        pa.field("l1_parent", pa.dictionary(pa.int32(), pa.string()), nullable=False),
        pa.field("v5_fold", pa.int8(), nullable=False),
        pa.field("v5_role", pa.dictionary(pa.int32(), pa.string()), nullable=False),
        pa.field("scale_tier", pa.dictionary(pa.int32(), pa.string()), nullable=False),
        pa.field("in_pool_2k", pa.bool_(), nullable=False),
        pa.field("in_pool_5k", pa.bool_(), nullable=False),
        pa.field("in_pool_20k", pa.bool_(), nullable=False),
        pa.field("in_pool_50k", pa.bool_(), nullable=False),
        pa.field("eval_suite", pa.dictionary(pa.int32(), pa.string()), nullable=False),
        pa.field("access_policy", pa.dictionary(pa.int32(), pa.string()), nullable=False),
        pa.field("has_xtb_coverage", pa.bool_(), nullable=False),
        pa.field("build_source", pa.string(), nullable=False),
    ])
    arrays = [
        pa.array(cols["curve_id"], type=pa.string()),
        pa.array(cols["reaction_id"], type=pa.string()),
        dict_array(cols["row_kind"], ROW_KINDS),
        pa.array(cols["curve_built"], type=pa.bool_()),
        dict_array(cols["direction_available"], ["forward", "forward+reverse", "reverse", "unknown"]),
        pa.array(cols["n_intents"], type=pa.int16()),
        pa.array(cols["n_atoms"], type=pa.int16()),
        pa.array(cols["n_irc_nodes"], type=pa.int16()),
        pa.array(cols["ts_index"], type=pa.int16()),
        dict_array(cols["family_label"]),
        dict_array(cols["l0_official"], ["missing", "test", "train", "valid"]),
        dict_array(cols["l1_parent"], L1_LABELS),
        pa.array(cols["v5_fold"], type=pa.int8()),
        dict_array(cols["v5_role"], V5_ROLES),
        dict_array(cols["scale_tier"], SCALE_TIERS),
        pa.array(cols["in_pool_2k"], type=pa.bool_()),
        pa.array(cols["in_pool_5k"], type=pa.bool_()),
        pa.array(cols["in_pool_20k"], type=pa.bool_()),
        pa.array(cols["in_pool_50k"], type=pa.bool_()),
        dict_array(cols["eval_suite"], EVAL_SUITES),
        dict_array(cols["access_policy"], ACCESS_POLICIES),
        pa.array(cols["has_xtb_coverage"], type=pa.bool_()),
        pa.array(cols["build_source"], type=pa.string()),
    ]
    table = pa.Table.from_arrays(arrays, schema=schema)

    tmp_parquet = out_dir / ".master_split.parquet.tmp"
    pq.write_table(table, tmp_parquet, compression="zstd", compression_level=9,
                   use_dictionary=True, write_statistics=True, version="2.6",
                   data_page_size=1 << 20, row_group_size=65536)
    os.replace(tmp_parquet, parquet_path)
    parquet_bytes = parquet_path.stat().st_size
    parquet_sha = sha256_file(parquet_path)
    LOG.info("wrote %s rows=%d bytes=%d sha256=%s",
             parquet_path, n_rows, parquet_bytes, parquet_sha)

    # ------------- validation: read back -------------
    t2 = pq.read_table(parquet_path)
    read_rows = t2.num_rows
    # parquet 不保留 arrow dictionary 的索引位宽（写 int32 读回 int32，已用 int32 建表）；
    # equals 忽略 field metadata，roundtrip 应完全一致。
    read_schema_ok = t2.schema.remove_metadata().equals(schema.remove_metadata())
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    dtype_map = {c: str(df[c].dtype) for c in df.columns}
    category_cols = [c for c in df.columns if str(df[c].dtype) == "category"]
    string_cols = [c for c in df.columns if str(df[c].dtype) in ("str", "string", "object")]

    # recompute sets from the *read-back* table (proves the written artifact)
    def col_vals(name):
        return t2.column(name).to_pylist()

    d_curve_id = col_vals("curve_id")
    d_l0 = col_vals("l0_official")
    d_v5role = col_vals("v5_role")
    d_eval = col_vals("eval_suite")
    d_access = col_vals("access_policy")
    d_scale = col_vals("scale_tier")
    d_l1 = col_vals("l1_parent")
    d_kind = col_vals("row_kind")
    d_built = col_vals("curve_built")

    def props(vals):
        out = {}
        for v in vals:
            out[v] = out.get(v, 0) + 1
        return out

    counts = {
        "total_rows": read_rows,
        "row_kind": props(d_kind),
        "curve_built_true": int(sum(d_built)),
        "l0_official": props(d_l0),
        "l1_parent": props(d_l1),
        "v5_role": props(d_v5role),
        "scale_tier": props(d_scale),
        "eval_suite": props(d_eval),
        "access_policy": props(d_access),
    }

    # sets for intersection proofs (read-back based)
    rs = {
        "id_dev": {c for c, e in zip(d_curve_id, d_eval) if e == "id_dev"},
        "family_ood": {c for c, e in zip(d_curve_id, d_eval) if e == "family_ood"},
        "scaffold_ood": {c for c, e in zip(d_curve_id, d_eval) if e == "scaffold_ood"},
        "size_ood": {c for c, e in zip(d_curve_id, d_eval) if e == "size_ood"},
        "sealed": {c for c, a in zip(d_curve_id, d_access) if a == "sealed_never_read"},
        "trainable_open": {c for c, a in zip(d_curve_id, d_access) if a == "open"},
        "pool": {c for c, r in zip(d_curve_id, d_v5role) if r == "pool"},
        "confirm_rows": {c for c, k in zip(d_curve_id, d_kind) if k == "sealed_confirm_id"},
        "l0_train": {c for c, x in zip(d_curve_id, d_l0) if x == "train"},
        "l0_test": {c for c, x in zip(d_curve_id, d_l0) if x == "test"},
        "l0_valid": {c for c, x in zip(d_curve_id, d_l0) if x == "valid"},
    }
    eval_names = ["id_dev", "family_ood", "scaffold_ood", "size_ood"]
    suite_sizes = {k: len(rs[k]) for k in eval_names}
    pairwise = {}
    for i in range(len(eval_names)):
        for j in range(i + 1, len(eval_names)):
            a, b = eval_names[i], eval_names[j]
            pairwise[f"{a}&{b}"] = len(rs[a] & rs[b])
    sealed_vs = {
        "sealed & trainable(access_policy==open)": len(rs["sealed"] & rs["trainable_open"]),
        "sealed & v5_role==pool": len(rs["sealed"] & rs["pool"]),
        "sealed & any_eval_suite": len(rs["sealed"] & (rs["id_dev"] | rs["family_ood"] | rs["scaffold_ood"] | rs["size_ood"])),
        "sealed & l0_official==train": len(rs["sealed"] & rs["l0_train"]),
        "sealed & l0_official==test": len(rs["sealed"] & rs["l0_test"]),
        "sealed & l0_official==valid": len(rs["sealed"] & rs["l0_valid"]),
    }
    validation = {
        "rows_read_back": read_rows,
        "rows_expected": n_rows,
        "rows_match": read_rows == n_rows,
        "unique_curve_id": len(set(d_curve_id)),
        "unique_reaction_id": len(set(col_vals("reaction_id"))),
        "schema_roundtrip_identical": bool(read_schema_ok),
        "pandas_dtypes": dtype_map,
        "category_columns": category_cols,
        "string_columns": string_cols,
        "dtype_note": ("pyarrow: string / dictionary<int32,string> / bool / int8 / int16；"
                       "pandas %s 读回：dictionary->category，arrow string->'str'(StringDtype)。" % pd.__version__),
        "four_suite_sizes": suite_sizes,
        "four_suite_pairwise_intersections": pairwise,
        "four_suite_pairwise_disjoint": all(v == 0 for v in pairwise.values()),
        "sealed_vs_train_sets": sealed_vs,
        "sealed_rows": len(rs["sealed"]),
        "sealed_equals_confirm_rows": len(rs["sealed"]) == len(rs["confirm_rows"]),
        "l0_test_rows": len(rs["l0_test"]),
        "l0_valid_rows": len(rs["l0_valid"]),
        "l0_missing_rows": int(sum(1 for x in d_l0 if x == "missing")),
        "curve_rows_all_have_intents": all(intents_per_curve[pos[c]] == 2 for c in built_ids),
        "datasets_mutually_exclusive_note": (
            "四个测试集互斥按单值 eval_suite + 优先级(id_dev>family_ood>scaffold_ood>size_ood)构造；"
            "sealed 与可训练集(access_policy==open / v5_role==pool)交集为 0；"
            "sealed 与 l0_official==train 的交集为 2,000 是结构性事实(战役曲线全部取自 L0 train)，由密封策略覆盖。"
        ),
    }

    # ------------- reconciliation -------------
    def find_registry_artifact(aid):
        for a in registry.get("artifacts", []):
            if a.get("id") == aid:
                return a
        return None

    reg_g0 = find_registry_artifact("g0_official_split") or {}
    reg_l1 = find_registry_artifact("l1_pesn_irc_split") or {}
    reg_c20 = find_registry_artifact("l3_v53mp_cohort_20k") or {}
    reg_c50 = find_registry_artifact("l3_v53mp_cohort_50k") or {}
    reg_s50 = find_registry_artifact("l3_v53mp_split_50k") or {}
    idx_split = idx_stats.get("split_50k", {})
    idx_c50 = idx_stats.get("cohort_50k", {})

    built_l0 = {c: l0.get(c, "missing") for c in built_ids}
    built_l0_counts = props(built_l0.values())
    l1_counts = props(l1_of(c) for c in built_ids)  # built rows only, not sealed placeholders

    checks = []

    def add_check(name, expected, actual, note=""):
        ok = expected == actual
        checks.append({"check": name, "expected": expected, "actual": actual,
                       "delta": (actual - expected) if isinstance(actual, int) and isinstance(expected, int) else None,
                       "match": bool(ok), "note": note})

    # L0 csv vs registry
    add_check("L0 csv rows vs registry", 199168, len(l0))
    for k in ("train", "valid", "test"):
        add_check(f"L0 csv {k} vs registry", reg_g0["counts"][k], l0_counts.get(k, 0))
    add_check("L0 csv rows vs split_manifest covered", load_json(p_l0_manifest)["covered"], len(l0))
    # L3 split counts
    add_check("pool 50k", 50000, len(tier_sets["50k"]))
    for t in ("2k", "5k", "20k", "50k"):
        add_check(f"pool tier {t} vs registry", reg_s50["counts"]["pool_tiers"][t], len(tier_sets[t]))
    add_check("dev_589", 589, len(dev589))
    add_check("confirm_2000", 2000, len(confirm))
    add_check("confirm == curve_folds fold==4", len(confirm), len(confirm_from_folds))
    add_check("curve_folds entries (built+confirm)", len(built_ids) + len(confirm), len(cf))
    add_check("curve_folds built only", len(built_ids), sum(1 for v in cf.values() if v != 4))
    add_check("registry curve_folds", reg_s50["counts"]["curve_folds"], len(cf))
    add_check("idx_stats curve_folds_n", idx_split.get("curve_folds_n"), len(cf))
    for fold, cnt in idx_split.get("curve_folds_values", {}).items():
        add_check(f"idx_stats fold {fold}", cnt, sum(1 for v in cf.values() if v == int(fold)))
    add_check("cohort50 pool_candidates", reg_c50["counts"]["pool_candidates"], len(cohort_pool))
    add_check("cohort50 dev", reg_c50["counts"]["dev"], len(cohort_dev))
    add_check("cohort50 confirm", reg_c50["counts"]["confirm"], len(cohort_confirm))
    add_check("cohort20 pool_candidates", reg_c20["counts"]["pool_candidates"], len(cohort20["pool_candidates"]))
    add_check("idx_stats cohort50 n_pool", idx_c50.get("n_pool"), len(cohort_pool))
    # table counts
    add_check("table rows (built+sealed policy)", n_rows,
              len(built_ids) + (0 if args.strict_curve_only else len(confirm)),
              "曲线口径=64,195；含 2,000 条 confirm 密封 id 行")
    add_check("table built curve rows", 64195, len(built_ids))
    add_check("table sealed rows", 2000 if not args.strict_curve_only else 0, counts["access_policy"].get("sealed_never_read", 0))
    add_check("table v5_role pool", 50000, counts["v5_role"].get("pool", 0))
    add_check("table v5_role dev_589", 589, counts["v5_role"].get("dev_589", 0))
    add_check("table v5_role confirm_2000", 2000 if not args.strict_curve_only else 0,
              counts["v5_role"].get("confirm_2000", 0))
    add_check("table v5_role unused", len(built_ids) - 50000 - 589, counts["v5_role"].get("unused", 0))
    add_check("table l0 train rows", len(built_ids) + (0 if args.strict_curve_only else len(confirm)),
              counts["l0_official"].get("train", 0), "所有 built curve 均取自 L0 train")

    # ------------- cross-file custody checks -------------
    custody = []
    repo_split50_sha = sha256_file(p_split50)
    registry_claims = {a.get("id"): a.get("sha256") for a in registry.get("artifacts", [])}
    custody.append({
        "check": "repo split_ids_g0_v1.csv sha256 vs registry",
        "path": str(p_l0_csv), "actual": sha256_file(p_l0_csv),
        "expected": registry_claims.get("g0_official_split"),
        "match": sha256_file(p_l0_csv) == registry_claims.get("g0_official_split"),
    })
    custody.append({
        "check": "repo pesn_irc_split_ids_v1.json sha256 vs registry",
        "path": str(p_l1), "actual": sha256_file(p_l1),
        "expected": registry_claims.get("l1_pesn_irc_split"),
        "match": sha256_file(p_l1) == registry_claims.get("l1_pesn_irc_split"),
    })
    custody.append({
        "check": "repo v53mp_split_50k_ids_v1.json sha256 vs registry",
        "path": str(p_split50), "actual": repo_split50_sha,
        "expected": registry_claims.get("l3_v53mp_split_50k"),
        "match": repo_split50_sha == registry_claims.get("l3_v53mp_split_50k"),
    })
    for aid, path in (("l3_v53mp_cohort_50k", p_cohort50), ("l3_v53mp_cohort_20k", p_cohort20)):
        custody.append({"check": f"repo {path.name} sha256 vs registry", "path": str(path),
                        "actual": sha256_file(path), "expected": registry_claims.get(aid),
                        "match": sha256_file(path) == registry_claims.get(aid)})
    # WSL copy verbatim
    if p_split_manifest_wsl.exists():
        wsl_sha = sha256_file(p_split_manifest_wsl)
        custody.append({"check": "WSL split_manifest_v53mp.json sha256 == repo split50 sha256",
                        "path": str(p_split_manifest_wsl), "actual": wsl_sha,
                        "expected": repo_split50_sha, "match": wsl_sha == repo_split50_sha})
        custody.append({"check": "WSL split_manifest_v53mp.json counts internal",
                        "expected": split50["counts"], "actual": load_json(p_split_manifest_wsl).get("counts"),
                        "match": load_json(p_split_manifest_wsl).get("counts") == split50["counts"]})
    if p_data_manifest_wsl.exists():
        dm_sha = sha256_file(p_data_manifest_wsl)
        bind = curves50.get("binding", {})
        custody.append({"check": "WSL data_manifest.json sha256 == curves.json binding.data_manifest_sha256",
                        "path": str(p_data_manifest_wsl), "actual": dm_sha,
                        "expected": bind.get("data_manifest_sha256"),
                        "match": dm_sha == bind.get("data_manifest_sha256")})
    bind = curves50.get("binding", {})
    custody.append({"check": "curves.json binding.split_manifest_sha256 == repo split50 sha256",
                    "expected": repo_split50_sha, "actual": bind.get("split_manifest_sha256"),
                    "match": bind.get("split_manifest_sha256") == repo_split50_sha})
    custody.append({"check": "curves.json binding n_curves == curve_ids length",
                    "expected": len(curves50["curve_ids"]), "actual": bind.get("n_curves"),
                    "match": bind.get("n_curves") == len(curves50["curve_ids"])})
    custody.append({"check": "curves.json binding n_intents == 2*curves",
                    "expected": 2 * len(curves50["curve_ids"]), "actual": bind.get("n_intents"),
                    "match": bind.get("n_intents") == 2 * len(curves50["curve_ids"])})
    custody.append({"check": "intents npy length == binding n_intents",
                    "expected": bind.get("n_intents"), "actual": int(len(ci)),
                    "match": len(ci) == bind.get("n_intents")})
    custody.append({"check": "every built curve has exactly 2 intents",
                    "expected": len(built_ids),
                    "actual": int(sum(1 for c in built_ids if intents_per_curve[pos[c]] == 2)),
                    "match": all(intents_per_curve[pos[c]] == 2 for c in built_ids)})
    custody.append({"check": "every built curve has both directions (0 and 1)",
                    "expected": len(built_ids),
                    "actual": int(sum(1 for c in built_ids if dir_sets.get(pos[c]) == {0, 1})),
                    "match": all(dir_sets.get(pos[c]) == {0, 1} for c in built_ids)})
    custody.append({"check": "curve_family.csv rows", "expected": len(built_ids),
                    "actual": family_rows, "match": family_rows == len(built_ids)})
    custody.append({"check": "curve_family.csv curve_id set == built curve set",
                    "expected": len(built_set), "actual": len(set(fam_label) & built_set),
                    "match": set(fam_label) == built_set})
    custody.append({"check": "all built curve_ids present in L0 csv",
                    "expected": len(built_ids), "actual": int(sum(1 for c in built_ids if c in l0)),
                    "match": all(c in l0 for c in built_ids)})
    custody.append({"check": "built curve_ids in cohort50.pool_candidates",
                    "expected": len(built_set & cohort_pool), "actual": len(built_set & cohort_pool),
                    "match": len(built_set - cohort_pool - dev589) == 0,
                    "note": f"built = cohort candidates {len(built_set & cohort_pool)} + dev_589 {len(built_set & dev589)}"})
    custody.append({"check": "confirm disjoint from built curves", "expected": 0,
                    "actual": len(confirm & built_set), "match": len(confirm & built_set) == 0})
    custody.append({"check": "dev_589 disjoint from pool & confirm", "expected": 0,
                    "actual": len(dev589 & tier_sets["50k"]) + len(dev589 & confirm),
                    "match": len(dev589 & tier_sets["50k"]) == 0 and len(dev589 & confirm) == 0})
    custody.append({"check": "pool tiers nested prefixes 2k<=5k<=20k<=50k", "expected": True,
                    "actual": (tier_sets["2k"] <= tier_sets["5k"] <= tier_sets["20k"] <= tier_sets["50k"]),
                    "match": (tier_sets["2k"] <= tier_sets["5k"] <= tier_sets["20k"] <= tier_sets["50k"])})
    custody.append({"check": "split pool_50k subset of cohort50 pool_candidates",
                    "expected": True, "actual": tier_sets["50k"] <= cohort_pool,
                    "match": tier_sets["50k"] <= cohort_pool,
                    "note": "cohort pool_candidates 65,000 是候选集，split 的 pool 50k 是冻结后集合；不要求相等"})
    custody.append({"check": "cohort20 pool_candidates prefix subset of cohort50 pool_candidates",
                    "expected": True, "actual": set(cohort20["pool_candidates"]) <= cohort_pool,
                    "match": set(cohort20["pool_candidates"]) <= cohort_pool})
    if curves20_count is not None:
        custody.append({"check": "legacy 20k tree curves are subset of 50k built curves",
                        "expected": True, "actual": bool(curves20_subset),
                        "match": bool(curves20_subset),
                        "note": f"20k tree n={curves20_count}；50k tree 是累积树(20k ∪ 50k 增量)"})

    # staging chunks: enumerate + sample shape check
    staging = {"dir": str(d50 / "staging_chunks")}
    chunks_dir = d50 / "staging_chunks"
    chunk_files = sorted(chunks_dir.glob("*.npz")) if chunks_dir.exists() else []
    staging["n_chunk_files"] = len(chunk_files)
    if chunk_files and not args.skip_staging_scan:
        covered: set[str] = set()
        for cf_path in chunk_files:
            with np.load(cf_path) as z:
                for k in z.files:
                    if "::" in k:
                        covered.add(k.split("::", 1)[0])
        staging["n_curves_listed_in_chunks"] = len(covered)
        staging["curves_not_in_chunks"] = sorted(built_set - covered)[:20]
        staging["n_curves_not_in_chunks"] = len(built_set - covered)
        staging["chunks_cover_all_built_curves"] = built_set <= covered
        staging["n_extra_ids_in_chunks_not_built"] = len(covered - built_set)
        staging["extra_ids_in_chunks_sample"] = sorted(covered - built_set)[:20]
        staging["extra_ids_note"] = (
            "chunks 里还列了 1,405 个非 built 的 reaction id（= cohort 候选未构建缺口，见 gaps G5）；"
            "它们没有 curves.json 记录，不在本表。")
        # deterministic sample of 12 chunks, first 3 curves each
        sample_idx = sorted({int(x) for x in np.linspace(0, len(chunk_files) - 1, 12)})
        checked = 0
        mism = []
        frame_rule_ok = 0
        for si in sample_idx:
            cf_path = chunk_files[si]
            with np.load(cf_path) as z:
                keys = sorted(k for k in z.files if k.endswith("::atomic_numbers"))
                for k in keys[:3]:
                    cid = k.split("::", 1)[0]
                    if cid not in built_set:
                        continue
                    an = z[k]
                    coords = z[f"{cid}::coordinates"]
                    energies = z[f"{cid}::energies"]
                    n_frames_expected = int(curve_rec[cid]["n_nodes"]) + 1  # 口径: chunk 帧数 = n_nodes + 1
                    checked += 1
                    ok = (len(an) == n_atoms[cid]
                          and coords.shape[1] == n_atoms[cid]
                          and coords.shape[0] == n_frames_expected
                          and energies.shape[0] == n_frames_expected)
                    if ok:
                        frame_rule_ok += 1
                    else:
                        mism.append(cid)
        staging["sample_chunks"] = [str(chunk_files[i].name) for i in sample_idx]
        staging["sample_curves_checked"] = checked
        staging["sample_curves_ok"] = frame_rule_ok
        staging["sample_mismatches"] = mism[:20]
        staging["sample_shape_check_ok"] = not mism
        staging["frame_count_rule"] = (
            "实测 staging chunk 的 coordinates/energies 帧数 = curves.json.n_nodes + 1（抽样 %d/%d 一致）；"
            "label_curves.lab_n == n_nodes。本表 n_irc_nodes 保留 curves.json 原值，不重算。" % (frame_rule_ok, checked))
        # 交叉验证：chunk 里列出的非 built id 应恰为 cohort 候选/ dev 中未构建的 id
        unbuilt_expected = (cohort_pool - built_set) | (cohort_dev - built_set)
        extra_chunk_ids = covered - built_set
        staging["unbuilt_expected_ids"] = len(unbuilt_expected)
        staging["chunk_extra_ids_match_unbuilt"] = extra_chunk_ids == unbuilt_expected
        custody.append({
            "check": "chunk-listed non-built ids == cohort unbuilt candidates",
            "expected": len(unbuilt_expected), "actual": len(extra_chunk_ids),
            "match": extra_chunk_ids == unbuilt_expected,
            "note": f"未构建 = cohort50 候选未构建 {len(cohort_pool - built_set)} + cohort50.dev 未构建 {len(cohort_dev - built_set)}",
        })
    elif chunk_files:
        staging["skipped"] = True

    # ------------- source files (sha256) -------------
    source_files = [
        file_meta(p_l0_manifest, layer="L0", role="G0 official split manifest"),
        file_meta(p_l0_csv, layer="L0", rows=len(l0), role="L0 (reaction_id,split) ids"),
        file_meta(p_l0_parquet, layer="L0", rows=None, role="L0 source artifact (split_assignment.parquet)"),
        file_meta(p_l1, layer="L1", role="L1 parent split ids (subsets + family_holdout + isolated_conflicts)"),
        file_meta(p_split50, layer="L3", role="L3 50k split (tiers/dev/confirm/curve_folds)"),
        file_meta(p_cohort50, layer="L3", role="L3 cohort 50k ledger"),
        file_meta(p_cohort20, layer="L3", role="L3 cohort 20k ledger (legacy campaign)"),
        file_meta(p_registry, layer="all", role="sha/count baseline"),
        file_meta(p_idx_stats, layer="stats", role="pre-computed stats cross-check"),
        file_meta(p_curves50, layer="data50k", role="50k tree curves.json (curve_ids/atomic_numbers/n_nodes)"),
        file_meta(p_family, layer="data50k", rows=family_rows, role="curve family labels (20 families)"),
        file_meta(p_intent_curve, layer="data50k", role="intents curve_idx (N intents)"),
        file_meta(p_intent_dir, layer="data50k", role="intents direction (N intents)"),
        file_meta(p_split_manifest_wsl, layer="data50k", role="WSL split_manifest_v53mp.json (verbatim copy)"),
        file_meta(p_data_manifest_wsl, layer="data50k", role="WSL data_manifest.json (curves binding target)"),
        file_meta(p_curves20, layer="data20k", role="legacy 20k tree curves.json (custody check)"),
    ]
    if p_l0_parquet.exists():
        try:
            meta = pq.ParquetFile(p_l0_parquet).metadata
            source_files[2]["rows"] = meta.num_rows
        except Exception:  # pragma: no cover
            pass

    # ------------- reconciliation details -------------
    reconciliation = {
        "reference_files": {
            "split_registry_v1.json": str(p_registry),
            "idx_stats.json": str(p_idx_stats),
        },
        "checks": checks,
        "all_checks_pass": all(c["match"] for c in checks),
        "differences": [c for c in checks if not c["match"]],
        "notes": [
            "split_registry_v1.json 的 l3_v53mp_split_50k.counts.curve_folds=66,195 是 built 64,195 + confirm 2,000 的 ids 总数；"
            "curves.json 实际 built 曲线 64,195，差值 2,000 即 confirm(从未构建)。",
            "idx_stats.json 的 split_50k.curve_folds_values 与源文件一致：0:21,399 1:21,398 2:21,398 4(confirm):2,000。",
            "idx_stats.json 的 split_50k.subsets_sizes.pool=4 指四档 tier 的键数，不是 pool 行数；pool=50,000 见 counts.pool_frozen。",
            "pool_overflow_unqualified_or_margin=15,000 = 已构建 overflow(unused) 13,606 + 未构建候选 1,394；"
            "cohort50.dev=600 = 已建 dev 589 + 未构建 11；两者合计 1,405 条无曲线 reaction，与 staging_chunks 多列的 1,405 个 id 完全一致。",
        ],
    }

    # ------------- roll-up numbers for the reply / manifest -------------
    key_counts = {
        "pool_50k": len(tier_sets["50k"]),
        "dev_589": len(dev589),
        "confirm_2000": len(confirm),
        "sealed_rows": counts["access_policy"].get("sealed_never_read", 0),
        "family_ood_rows": suite_sizes["family_ood"],
        "size_ood_rows": suite_sizes["size_ood"],
        "id_dev_rows": suite_sizes["id_dev"],
        "scaffold_ood_rows": suite_sizes["scaffold_ood"],
        "eval_none_rows": counts["eval_suite"].get("none", 0),
        "open_rows": counts["access_policy"].get("open", 0),
        "one_shot_dev_rows": counts["access_policy"].get("one_shot_dev", 0),
        "unused_rows": counts["v5_role"].get("unused", 0),
        "has_xtb_coverage_true": 0,
        "l0_train_rows": counts["l0_official"].get("train", 0),
        "l0_test_rows": counts["l0_official"].get("test", 0),
    }
    eval_rules = {
        "priority": "id_dev > family_ood > scaffold_ood > size_ood > none (单值、互斥，后面的规则只在前面未占用的行上生效)",
        "id_dev": {"rule": "v5_role == dev_589 (L3 冻结 dev, 589 条)", "rows": suite_sizes["id_dev"]},
        "family_ood": {
            "rule": ("在 built curves 上统计 family_labels 计数；候选族 = 计数>=%d 且 !='other' 且 !='unassigned'；"
                     "按 sha256('%s'+family) 升序取前 %d 个族；family_ood = 属于这些族且未被 id_dev 占用的行"
                     % (FAMILY_MIN_COUNT, FAMILY_OOD_SEED, args.family_ood_k)),
            "families": [{"family": f, "curves_in_family": fam_counts.get(f, 0),
                          "sha256_score": fam_hash[f]} for f in family_ood_families],
            "rows": suite_sizes["family_ood"],
            "rows_per_family": {f: sum(1 for c in family_ood_set if fam_label.get(c) == f)
                                for f in family_ood_families},
        },
        "scaffold_ood": {
            "rule": "无 scaffold 标签源 -> 置 none(空集)",
            "rows": 0,
            "gap": "仓库 data/manifests、config、scripts、data 下未发现 scaffold 标签文件；需先产 scaffold 指派(见 open_questions)",
        },
        "size_ood": {
            "rule": ("n_atoms = len(curves.json[].atomic_numbers)；Q3 = numpy.percentile(n_atoms,75) = %.3f；"
                     "size_ood 候选 = n_atoms %s Q3；再排除 id_dev/family_ood 已占用的行" %
                     (q3, ">" if args.size_ood_rule == "strict_gt_q3" else ">=")),
            "q3_threshold": q3,
            "n_atoms_ge_q3": q3_ge_count,
            "n_atoms_eq_q3": int((n_atoms_arr == q3).sum()),
            "size_ood_candidates": len(size_cand),
            "rows": suite_sizes["size_ood"],
            "n_atoms_range_in_suite": [int(min(n_atoms[c] for c in size_ood_set)),
                                       int(max(n_atoms[c] for c in size_ood_set))] if size_ood_set else [],
            "tie_note": ("n_atoms 是离散值，19 处有 5,023 条并列；用 >=Q3 会把 32.5% 的行算进来，"
                         "因此采用严格 >Q3(约 24.7%)；切换 --size-ood-rule ge_q3 可复现另一种口径。"),
        },
    }
    access_rules = {
        "sealed_never_read": "row_kind=='sealed_confirm_id' 或 l0_official=='test'（本期 2,000 条 confirm；L0 test 在 built 集中为 0 条）",
        "one_shot_dev": "eval_suite != 'none' 的所有行（id_dev/family_ood/scaffold_ood/size_ood）→ 仅供评测读取一次，训练 loader 必须过滤 eval_suite=='none'",
        "open": "其余行（v5_role=pool/unused 且 eval_suite==none）",
        "training_loader_contract": "训练可用行 = (access_policy == 'open')；评测行 = (eval_suite != 'none')；禁止读取 access_policy == 'sealed_never_read'",
    }

    gaps = [
        {"id": "G1", "gap": "has_xtb_coverage 全列为 false（占位）",
         "detail": "本期不计算 XTB 覆盖；列已预留。", "next_step": "由 frame_crosswalk 回填后再更新本表/新增覆盖列。"},
        {"id": "G2", "gap": "scaffold_ood 为空集：无 scaffold 标签源",
         "detail": "仓库 manifests/config/scripts/data 与 WSL 数据树均未发现 scaffold 指派文件。",
         "next_step": "从 L0 原始 SMILES(B3LYP-RXN_*.csv / 数据集 inventory) 计算 Murcko/Bemis-Murcko scaffold，"
                      "再按 scaffold 分组留出；新增列 scaffold_id 并重生成。"},
        {"id": "G3", "gap": "built 曲线全部来自 L0 train；L0 valid(19,904)/test(19,909) 在本表 0 行",
         "detail": "64,195 条 built 曲线在 split_ids_g0_v1.csv 中 l0_official 全为 train；confirm 2,000 也是 train。"
                   "官方 valid/test 需要另行建表(本树未构建)。",
         "next_step": "若需官方 valid/test 评测，从 L0 原始 CSV 另建测试表；不要把 official test 行导入训练 pool。"},
        {"id": "G4", "gap": "confirm_2000 无曲线(从未构建/从未读取)",
         "detail": "表内 2,000 行为 id-only 占位(row_kind=sealed_confirm_id)，n_atoms/n_irc_nodes=-1。",
         "next_step": "评测前需按 v53mp 构建流程单独构建 confirm 表并保持 seal；构建后回填本表 has_curve/n_atoms 等列。"},
        {"id": "G5", "gap": "cohort 候选未构建缺口",
         "detail": "cohort50.pool_candidates 65,000，其中 built=63,606（另 dev_589 不在候选集内）；"
                   "未构建候选=1,394；cohort50.dev=600 但 built dev=589（11 条未构建）；合计 1,405 条 reaction 无曲线。",
         "next_step": "如需全覆盖，检查这 1,405 条的 s0a 资格/构建失败日志；否则在数据账本标注为预期缺口。"},
        {"id": "G6", "gap": "staging_chunks 只做抽样形状核对，未逐曲线解压全量坐标",
         "detail": "枚举了全部 chunk 的曲线 id 覆盖；随机 12 个 chunk 抽样核对 atomic_numbers/坐标/能量形状。",
         "next_step": "全量坐标一致性交给 frame_crosswalk/帧审计任务。"},
        {"id": "G7", "gap": "L1 父划分的并集含 960 条 L0-valid / 945 条 L0-test 反应",
         "detail": "按 reaction_ids 统计；对 built 集无影响(全为 L0 train)，但说明 L1 grouping 与 L0 分区存在边缘交叉"
                   "(manifests 记录 known_group_cross_split_conflicts=0，仅 2 组 isolated_conflicts)。",
         "next_step": "与 L1 生成脚本 owner 确认：这些反应是 group 成员还是分组口径差异；必要时在 L1 manifest 增加解释。"},
    ]

    open_questions = [
        {"id": "Q1", "question": "总表行数口径：是否保留 2,000 条 confirm id-only 行？",
         "current_decision": "保留 → 66,195 行 = 64,195 built curves + 2,000 sealed confirm ids；这样 sealed 行数=2,000 "
                             "且能在一张表里表达 'confirm 永远不可读'。",
         "alternative": "严格 1 行=1 built curve_id → 64,195 行，sealed 行=0（confirm 只在 manifest 账本里）。",
         "next_step": "如要 64,195 行口径：python pes2ts_data/builders/build_master_split.py --strict-curve-only（一行命令可复现）。"},
        {"id": "Q2", "question": "eval_suite 单值 + 优先级的互斥口径是否符合下游用法？",
         "current_decision": "单值互斥，优先级 id_dev > family_ood > scaffold_ood > size_ood；未占用的行才归下一档。",
         "alternative": "多标签（每套一列 bool）以便同一行同时是 family_ood 和 size_ood。",
         "next_step": "若下游需要多标签：把 eval_suite 之外再加 4 个 bool 列（family_ood/size_ood/...）即可，行集合不变。"},
        {"id": "Q3", "question": "family_ood 的 4 个族选择规则是否认可？",
         "current_decision": "候选=计数>=200 且非 'other' 的族，按 sha256('pes2ts:family_ood:v1:'+family) 升序取前 4："
                             "h_transfer|ring_closure, ring_opening, rearrangement|ring_closure, h_transfer|ring_opening（6,519 行）。",
         "alternative": "业务指定族；或用 L1 family_holdout（但其与 built 集交集=0，无法直接用于本表）。",
         "next_step": "如需改族：修改脚本常量 FAMILY_OOD_SEED / FAMILY_OOD_K 或传入 --family-ood-k；族名单与行数会写入 manifest。"},
        {"id": "Q4", "question": "size_ood 阈值用 >Q3 还是 >=Q3？",
         "current_decision": ">Q3（Q3=19，19 处并列 5,023 条；>= 会把 32.5% 算进 OOD，> 为 24.7%）。",
         "alternative": ">=Q3 → 候选 20,879 行（--size-ood-rule ge_q3）。",
         "next_step": "下游若按高原子数连续分布评测，可用 ge_q3 重生成；manifest 两个数都已记录。"},
        {"id": "Q5", "question": "eval 套件与冻结 50k pool 有重叠（family_ood 5,095 行、size_ood 部分行在 pool 内），是否接受？",
         "current_decision": "接受；这些行 access_policy=one_shot_dev，训练 loader 必须按 eval_suite=='none' 过滤。"
                             "pool 是构建/候选概念，不自动等于训练集。",
         "alternative": "eval 套件严格取自 v5_role=unused（13,606 行），从而与 pool 完全无交集；但族/OOD 规模会大幅缩小。",
         "next_step": "若要 eval∩pool=0：在脚本里把候选限制到 v5_role=='unused' 后重生成，并重新核对四套件规模。"},
        {"id": "Q6", "question": "confirm_2000 属于 L0 train 且 L1 parent=train_cal，与 'sealed 不可训练' 是否冲突？",
         "current_decision": "不冲突：L0/L1 是历史标签层；access_policy=sealed_never_read 覆盖它们。但审计上 sealed∩L0-train=2,000，"
                             "sealed∩可训练集(open/pool)=0。",
         "alternative": "若审计要求 '任何 train 标签集合都不含 sealed'，需在 L0/L1 标签层把 confirm 剔除/改写（会影响上游 manifest，不宜在本表做）。",
         "next_step": "由 Lead 决定是否在 L0/L1 层标注 confirm 为 sealed 子集；本表已提供 row_kind/access_policy 可 join。"},
        {"id": "Q7", "question": "2k/5k/20k 名单的权威源是 v53mp_split_50k_ids_v1.json 的 subsets.pool 前缀，确认？",
         "current_decision": "是；三档均为 50k 冻结 pool 的嵌套前缀(2k⊂5k⊂20k⊂50k)。若文件缺某 tier，则按 rules 的"
                             "'nested prefixes in cohort rank order' 用前 N 条候选重构。",
         "alternative": "用 v53mp_cohort_20k_ids_v1.json 作 20k 名单（它与 50k split 的 20k tier 不是同一集合）。",
         "next_step": "下游按 scale_tier 取预算时用 in_pool_* 布尔列（同时给出）避免歧义；若上游改 tier 定义，重跑脚本即可。"},
        {"id": "Q8", "question": "reaction_id 派生规则确认",
         "current_definition": "curves.json 无独立 reaction 字段；64,195 个 curve_id 全部形如 RXN_* 且在 split_ids_g0_v1.csv 中"
                               "都能命中，故 reaction_id := curve_id（1 curve = 1 reaction，双向 intent 各 1 条）。"
                               "confirm 占位行 reaction_id := confirm id 本身。",
         "next_step": "如后续数据树引入 1 reaction 多 curve，需要新的 curve↔reaction crosswalk 列并重生成。"},
    ]

    # ------------- manifest payload -------------
    payload = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": 1,
        "purpose": ("PES2TS 数据仓唯一总表：把 L0/L1/L3 划分合并为 1 行=1 curve_id 的分区索引，"
                    "并给出评测套件/访问策略/缺口与口径裁决。"),
        "created_by": "pes2ts_data/builders/build_master_split.py",
        "script": {"path": str(Path(__file__).resolve()),
                   "sha256": sha256_file(Path(__file__).resolve())},
        "outputs": {
            "table": {
                "path": str(parquet_path), "sha256": parquet_sha, "bytes": parquet_bytes,
                "rows": n_rows,
                "columns": [f.name for f in schema],
                "arrow_schema": str(schema),
                "write_options": {"format": "parquet", "compression": "zstd",
                                  "compression_level": 9, "version": "2.6",
                                  "use_dictionary": True, "write_statistics": True,
                                  "data_page_size": 1 << 20, "row_group_size": 65536},
            },
            "manifest": {
                "path": str(manifest_path),
                "content_sha256": None,  # 写入 manifest 时回填 = sha256(canonical payload, 本字段视作 null)
                "note": "content_sha256 = sha256(canonical payload，本字段按 null 计算)，覆盖除 run/idempotency 外的全部稳定内容；脚本 stdout 会打印 manifest 文件自身 sha256。",
            },
        },
        "row_universe": {
            "definition": "1 row = 1 curve_id（built）+ 1 row = 1 冻结 confirm reaction id（id-only 密封占位）",
            "total_rows": n_rows,
            "built_curve_rows": len(built_ids),
            "sealed_confirm_id_rows": 0 if args.strict_curve_only else len(confirm),
            "row_count_explanation": (
                "曲线口径 64,195 行（curves.json 全部 built 曲线）；为让 'confirm_2000/L0 test 必须 sealed' "
                "在表内可表达并满足 'sealed 行数=2,000' 验收，额外包含 2,000 条 confirm id-only 行（无曲线，"
                "row_kind=sealed_confirm_id, curve_built=false），合计 66,195 行。"
                "L0 test 在 built 集中为 0 行，故 sealed 行数恰为 2,000。"
                "如需严格 64,195 行口径：--strict-curve-only。"),
            "direction_note": "每 curve 有 forward+reverse 两个 intent(128,390=2×64,195)；direction_available 记录该曲线实际存在的方向。",
            "reaction_id_rule": "curves.json 无独立 reaction 字段；curve_id 均为 RXN_* 且逐一命中 L0 csv，故 reaction_id := curve_id。",
        },
        "counts": key_counts,
        "counts_by_column": counts,
        "column_contract": [
            {"name": "curve_id", "arrow": "string", "pandas": "string",
             "meaning": "曲线 id（RXN_*）；= reaction_id。sealed 占位行为 confirm reaction id，用 row_kind/curve_built 区分。"},
            {"name": "reaction_id", "arrow": "string", "pandas": "string",
             "meaning": "反应 id；规则见 row_universe.reaction_id_rule。"},
            {"name": "row_kind", "arrow": "dictionary<int8,string>", "pandas": "category",
             "values": ROW_KINDS, "meaning": "curve=有真实曲线；sealed_confirm_id=confirm 冻结 id 占位(从未构建/读取)。"},
            {"name": "curve_built", "arrow": "bool", "pandas": "bool", "meaning": "该行是否有真实曲线产物。"},
            {"name": "direction_available", "arrow": "dictionary<int8,string>", "pandas": "category",
             "values": ["forward", "forward+reverse", "reverse", "unknown"],
             "meaning": "该曲线在 intents 中实际存在的方向；unknown=无曲线(confirm 占位)。"},
            {"name": "n_intents", "arrow": "int16", "meaning": "该曲线 intent 数（built 恒为 2；sealed 占位 0）。"},
            {"name": "n_atoms", "arrow": "int16", "meaning": "原子数 len(curves.json.atomic_numbers)；sealed 占位 -1。size_ood 依据列。"},
            {"name": "n_irc_nodes", "arrow": "int16", "meaning": "curves.json.n_nodes（IRC 帧数）；sealed 占位 -1。"},
            {"name": "ts_index", "arrow": "int16", "meaning": "curves.json.ts_index；sealed 占位 -1。"},
            {"name": "family_label", "arrow": "dictionary<int8,string>", "pandas": "category",
             "meaning": "curve_family.csv family_labels（20 族）；sealed 占位 unassigned。family_ood 依据列。"},
            {"name": "l0_official", "arrow": "dictionary<int8,string>", "pandas": "category",
             "values": ["train", "valid", "test", "missing"],
             "meaning": "L0 官方划分（split_ids_g0_v1.csv）；本表 built+confirm 全为 train。"},
            {"name": "l1_parent", "arrow": "dictionary<int8,string>", "pandas": "category",
             "values": L1_LABELS,
             "meaning": "L1 父划分（pesn_irc_split_ids_v1.json）；isolated_conflict 优先于其它子集。"},
            {"name": "v5_fold", "arrow": "int8", "meaning": "L3 curve_folds（built:0/1/2；confirm:4）；-1=缺失。"},
            {"name": "v5_role", "arrow": "dictionary<int8,string>", "pandas": "category", "values": V5_ROLES,
             "meaning": "L3 角色：pool=冻结 50k pool；dev_589=冻结 dev；confirm_2000=confirm 占位；unused=其余 built 曲线。"},
            {"name": "scale_tier", "arrow": "dictionary<int8,string>", "pandas": "category", "values": SCALE_TIERS,
             "meaning": ("嵌套前缀的进入档位：最小(最内层)包含该曲线的 tier。2k/5k/20k/50k 名单取自 "
                         "v53mp_split_50k_ids_v1.json subsets.pool（nested prefixes, 2k⊂5k⊂20k⊂50k）；"
                         "不在 pool 内= none。若要判断'是否属于某预算集'请用 in_pool_* 列。")},
            {"name": "in_pool_2k", "arrow": "bool", "meaning": "是否属于冻结 pool 的 2k 前缀。"},
            {"name": "in_pool_5k", "arrow": "bool", "meaning": "是否属于冻结 pool 的 5k 前缀。"},
            {"name": "in_pool_20k", "arrow": "bool", "meaning": "是否属于冻结 pool 的 20k 前缀。"},
            {"name": "in_pool_50k", "arrow": "bool", "meaning": "是否属于冻结 50k pool。"},
            {"name": "eval_suite", "arrow": "dictionary<int8,string>", "pandas": "category",
             "values": EVAL_SUITES, "meaning": "建议评测套件（单值互斥，构造规则见 eval_suite_rules）。"},
            {"name": "access_policy", "arrow": "dictionary<int8,string>", "pandas": "category",
             "values": ACCESS_POLICIES, "meaning": "访问策略（见 access_policy_rules）。"},
            {"name": "has_xtb_coverage", "arrow": "bool", "meaning": "本表暂全 false；由 frame_crosswalk 回填。"},
            {"name": "build_source", "arrow": "string", "meaning": "该行数据来源（curves.json / confirm 名单）。"},
        ],
        "eval_suite_rules": eval_rules,
        "access_policy_rules": access_rules,
        "validation": validation,
        "reconciliation": reconciliation,
        "custody_checks": custody,
        "staging_chunks": staging,
        "source_files": source_files,
        "counts_vs_reference": {
            "registry_g0_counts": reg_g0.get("counts"),
            "registry_l1_counts": reg_l1.get("counts"),
            "registry_l3_split_counts": reg_s50.get("counts"),
            "registry_l3_cohort20_counts": reg_c20.get("counts"),
            "registry_l3_cohort50_counts": reg_c50.get("counts"),
            "idx_stats_split_50k_counts": idx_split.get("counts"),
            "idx_stats_split_50k_curve_folds_values": idx_split.get("curve_folds_values"),
            "built_l0_counts": built_l0_counts,
            "built_l1_counts": l1_counts,
            "confirm_l0_counts": props(l0.get(c, "missing") for c in confirm),
            "confirm_l1_counts": props(l1_of(c) for c in confirm),
        },
        "gaps": gaps,
        "open_questions": open_questions,
        "determinism": {
            "parquet_sha256": parquet_sha,
            "parquet_bytes": parquet_bytes,
            "rows": n_rows,
            "expected_stable_across_runs": True,
            "why": "行序按 curve_id 排序；dictionary 标签按固定顺序；无时间戳/浮点；parquet zstd 写入确定性。",
            "idempotency_proof": "manifest.idempotency.runs 会累积每次运行的 parquet_sha256；全部相同即证明幂等。",
        },
    }
    payload_json = canonical_json(payload)
    payload_sha = sha256_text(payload_json)

    # ------------- idempotency history -------------
    prev_manifest = None
    prev_manifest_file_sha = None
    if manifest_path.exists():
        try:
            prev_manifest_file_sha = sha256_file(manifest_path)
            prev_manifest = load_json(manifest_path)
        except Exception as exc:  # pragma: no cover
            LOG.warning("could not read previous manifest: %s", exc)
    prior_runs = []
    if isinstance(prev_manifest, dict):
        prior_runs = prev_manifest.get("idempotency", {}).get("runs", []) or []
    run_id = len(prior_runs) + 1
    finished_at = utc_now()
    duration_s = round(time.time() - started, 3)
    run_entry = {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": duration_s,
        "parquet_sha256": parquet_sha,
        "parquet_bytes": parquet_bytes,
        "rows": n_rows,
        "manifest_content_sha256": payload_sha,
        "previous_manifest_file_sha256": prev_manifest_file_sha,
        "script_sha256": payload["script"]["sha256"],
    }
    runs = (prior_runs + [run_entry])[-5:]
    sha_values = {r.get("parquet_sha256") for r in runs}
    idempotency = {
        "runs": runs,
        "n_runs_recorded": len(runs),
        "parquet_sha256_stable": len(sha_values) == 1,
        "distinct_parquet_sha256": sorted(sha_values),
        "manifest_content_sha256_stable": len({r.get("manifest_content_sha256") for r in runs}) == 1,
        "note": ("manifest 文件自身 sha256 因 run/idempotency 时间块每次不同；确定性内容由 manifest_content_sha256 覆盖。"
                 "脚本 stdout 末尾打印本次 manifest 文件 sha256。"),
    }

    run_block = {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": duration_s,
        "command": " ".join(sys.argv),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "library_versions": {"pyarrow": pa.__version__, "pandas": pd.__version__,
                             "numpy": np.__version__},
        "cgroup_memory_max": os.environ.get("MEMORY_MAX", None),
        "log_path": str(log_path),
    }

    manifest = dict(payload)
    manifest["outputs"]["manifest"]["content_sha256"] = payload_sha
    manifest["run"] = run_block
    manifest["idempotency"] = idempotency

    tmp_manifest = out_dir / ".master_split_manifest.json.tmp"
    with open(tmp_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_manifest, manifest_path)
    manifest_file_sha = sha256_file(manifest_path)
    manifest_bytes = manifest_path.stat().st_size

    LOG.info("wrote %s bytes=%d", manifest_path, manifest_bytes)
    LOG.info("parquet sha256=%s rows=%d bytes=%d", parquet_sha, n_rows, parquet_bytes)
    LOG.info("manifest content_sha256=%s", payload_sha)
    LOG.info("MANIFEST_SHA256=%s", manifest_file_sha)
    LOG.info("run_id=%d parquet_sha256_stable=%s", run_id, idempotency["parquet_sha256_stable"])
    LOG.info("counts: %s", json.dumps(key_counts, ensure_ascii=False, sort_keys=True))
    LOG.info("validation: four_suite_disjoint=%s sealed_rows=%d",
             validation["four_suite_pairwise_disjoint"], validation["sealed_rows"])
    LOG.info("reconciliation all_checks_pass=%s", reconciliation["all_checks_pass"])
    LOG.info("build_master_split done in %.2fs", duration_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
