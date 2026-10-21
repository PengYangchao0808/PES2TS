#!/usr/bin/env python3
"""V5.3-MP S0-A: inventory & metadata qualification (NO raw IRC frame reads).

Implements the S0-A half of the V5.3-MP data gate (plan §4.1): qualify the
~200k-reaction inventory for the 20k MOVE+PROCESS build using ONLY metadata
sources — the shape-only IRC index, the official split assignment, the
duplicate ledger, the G1 P1 summary parquets, the G1 class summary parquet,
and the frozen G3-P1 predictor split.  No IRC frame/TS geometry is read here
(the single audited accessor call is :func:`truth_reader.load_irc_index`,
which returns shape-only rows and appends ONE audit line; S0-B is the stage
that reads frames, with per-reaction audit).

Outputs ``metadata_qualification.json`` (plan §4.1): total / eligible /
excluded(reason) / dedup-group accounting / family distribution / exposure
sets with provenance.  Also writes ``eligible_reactions.parquet``
(reaction_id + canonical group + subset membership + family) consumed by the
cohort freeze step — qualification is pure bookkeeping and touches no
ground-truth geometry.

Development-exposure policy (V5.3-MP campaign level; the plan's S0 rule
"最终确认集必须保留未参与任何设计/标签 QC 的新反应与新反应族"): the exposure
set = fit50 selection ∪ V5 sealed folds(3/4) ∪ demo24 candidates ∪
irc_dev_250 ∪ G3-P1 teacher-dataset members, expanded to duplicate-ledger
groups.  Exposure groups are excluded from ALL V5.3-MP subsets
(pool/dev/confirm) at cohort construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pes2ts_core.g0.truth import truth_reader  # noqa: E402

SCHEMA = "v53mp_s0a_qualification_v1"
CALLER = "scripts.agent_ops.data_gate.r3_v53mp_s0a_qualify"

#: inputs (repo-relative)
SPLIT_ASSIGNMENT = "data/interim/split_assignment.parquet"
DUP_LEDGER = "data/manifests/duplicate_ledger.json"
P1_SUMMARY = "data/interim/g1_p1_summary.parquet"
CLASS_SUMMARY = "data/interim/g1_reaction_class_summary.parquet"
G3P1_SPLIT = "outputs/pesn/g3_p1/splits/pesn_irc_split.json"
MANIFESTS_DIR = "data/manifests"

#: development-exposure sources (campaign-level; see module docstring)
EXPOSURE_FIT50 = ("fit50_selection",
                  "outputs/pesn/direct_egnn_v1_20261015/splits/selection_direct_fit50.json")
EXPOSURE_V5_SPLIT = ("v5_split_sealed_folds34",
                     "data/manifests/v5/split_manifest.json")
EXPOSURE_DEMO24 = ("demo24_candidates",
                   "data/manifests/demo24_reaction_case_manifest_v1.json")
EXPOSURE_IRC_DEV250 = ("irc_dev_250",
                       "outputs/pesn/scale20k_rl2000_v1_20261008/cohorts/irc_fit_cohorts.json")
EXPOSURE_G3P1_TEACHER = ("g3p1_teacher_dataset",
                         "outputs/pesn/g3_p1/teacher_dataset/dataset_manifest.json")

#: eligibility thresholds (metadata-only)
MIN_N_FRAMES = 8
P1_STATUS_OK = frozenset({"resolved_unique", "resolved_symmetry_collapsed"})
LAYOUT_OK = "ts_first_dual_branch"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _exposure_ids(repo: Path) -> Dict[str, Dict[str, Any]]:
    """Named exposure sets with provenance (ids per source)."""
    out: Dict[str, Dict[str, Any]] = {}
    name, rel = EXPOSURE_FIT50
    doc = _read_json(repo / rel)
    ids = sorted(str(r) for r in (doc.get("reaction_ids") or []))
    out[name] = {"path": rel, "sha256": sha256_file(repo / rel),
                 "reaction_ids": ids, "n": len(ids)}
    name, rel = EXPOSURE_V5_SPLIT
    doc = _read_json(repo / rel)
    sealed = sorted(str(g) for g, f in (doc.get("group_fold") or {}).items()
                    if int(f) in (3, 4))
    out[name] = {"path": rel, "sha256": sha256_file(repo / rel),
                 "reaction_ids": sealed, "n": len(sealed),
                 "note": "V5 frozen folds 3/4 (sealed 20 of the 50-curve "
                         "campaign cohort)"}
    name, rel = EXPOSURE_DEMO24
    ids: list[str] = []
    path = repo / rel
    if path.is_file():
        doc = _read_json(path)
        # the manifest carries reaction ids in one of several shapes
        if isinstance(doc, dict):
            for key in ("reaction_ids", "candidates", "cases"):
                vals = doc.get(key)
                if isinstance(vals, list):
                    for v in vals:
                        if isinstance(v, str):
                            ids.append(v)
                        elif isinstance(v, dict) and "reaction_id" in v:
                            ids.append(str(v["reaction_id"]))
        ids = sorted(set(ids))
    out[name] = {"path": rel, "sha256": sha256_file(path) if path.is_file() else None,
                 "reaction_ids": ids, "n": len(ids)}
    name, rel = EXPOSURE_IRC_DEV250
    doc = _read_json(repo / rel)
    ids = sorted(str(r) for r in (doc.get("irc_dev_250") or []))
    out[name] = {"path": rel, "sha256": sha256_file(repo / rel),
                 "reaction_ids": ids, "n": len(ids)}
    name, rel = EXPOSURE_G3P1_TEACHER
    ids = []
    path = repo / rel
    if path.is_file():
        doc = _read_json(path)
        members = doc.get("subset_members") or {}
        if isinstance(members, dict):
            for subset, vals in members.items():
                if isinstance(vals, list):
                    ids.extend(str(v) for v in vals)
        ids = sorted(set(ids))
    out[name] = {"path": rel, "sha256": sha256_file(path) if path.is_file() else None,
                 "reaction_ids": ids, "n": len(ids),
                 "note": "G3-P1 teacher-dataset members (2250 reactions used "
                         "for the pesn-irc teacher build 2026-10-07)"}
    return out


def qualify(repo: Path, out_dir: Path, *, limit: Optional[int] = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- audited shape-only index read (single audit line) ------------------
    index_rows = truth_reader.load_irc_index(
        allow_truth=True, manifests_dir=repo / MANIFESTS_DIR)
    index = {str(r["reaction_id"]): r for r in index_rows}

    # --- free (non-quarantined) metadata sources ----------------------------
    assignment = {
        str(r["reaction_id"]): str(r["split"])
        for r in pq.read_table(repo / SPLIT_ASSIGNMENT).to_pylist()
    }
    dup_doc = _read_json(repo / DUP_LEDGER)
    canonical_of: Dict[str, str] = {}
    for grp in dup_doc.get("groups") or []:
        canon = str(grp["canonical_reaction_id"])
        for member in grp.get("member_reaction_ids") or []:
            canonical_of[str(member)] = canon
    p1_rows = pq.read_table(repo / P1_SUMMARY).to_pylist()
    p1 = {str(r["reaction_id"]): r for r in p1_rows}
    cls_rows = pq.read_table(
        repo / CLASS_SUMMARY,
        columns=["reaction_id", "family_labels", "edit_family_id", "status"],
    ).to_pylist()
    cls = {str(r["reaction_id"]): r for r in cls_rows}
    g3p1 = _read_json(repo / G3P1_SPLIT)
    subsets = {k: sorted(str(v) for v in vals)
               for k, vals in (g3p1.get("subsets") or {}).items()}
    subset_of: Dict[str, str] = {}
    for name, members in subsets.items():
        for rid in members:
            subset_of[rid] = name
    exposure = _exposure_ids(repo)
    exposure_ids: set[str] = set()
    for entry in exposure.values():
        exposure_ids.update(entry["reaction_ids"])
    # expand exposure to duplicate-ledger groups
    exposure_groups = {canonical_of.get(rid, rid) for rid in exposure_ids}

    counts = Counter()
    family_counter: Counter = Counter()
    records = []
    all_ids = sorted(index)
    if limit is not None:
        all_ids = all_ids[:limit]
    for rid in all_ids:
        row = index[rid]
        reason = None
        n_frames = int(row["n_frames"])
        p1_row = p1.get(rid)
        cls_row = cls.get(rid)
        split = assignment.get(rid)
        canon = canonical_of.get(rid, rid)
        if split != "train":
            reason = "parent_split_not_train"
        elif rid not in subset_of:
            reason = "not_in_g3p1_predictor_split"
        elif n_frames < MIN_N_FRAMES:
            reason = "n_frames_below_8"
        elif p1_row is None:
            reason = "p1_missing"
        elif str(p1_row.get("status")) not in P1_STATUS_OK:
            reason = "p1_status_{}".format(p1_row.get("status"))
        elif not bool(p1_row.get("g2_eligible")):
            reason = "p1_not_g2_eligible"
        elif str(p1_row.get("layout")) != LAYOUT_OK:
            reason = "layout_{}".format(p1_row.get("layout"))
        elif not (
            int(p1_row.get("n_broken") or 0)
            + int(p1_row.get("n_formed") or 0)
            + int(p1_row.get("n_order_changed") or 0)
            + int(p1_row.get("n_h_migration") or 0) > 0
        ):
            reason = "no_bond_events"
        elif cls_row is None or not cls_row.get("family_labels"):
            reason = "family_missing"
        elif canon in exposure_groups or rid in exposure_ids:
            reason = "development_exposure"
        counts["total"] += 1
        if reason is not None:
            counts[f"excluded.{reason}"] += 1
            continue
        counts["eligible"] += 1
        fams = sorted(str(cls_row["family_labels"]).split("|"))
        for fam in fams:
            family_counter[fam] += 1
        records.append({
            "reaction_id": rid,
            "canonical_group": canon,
            "subset": subset_of[rid],
            "n_frames": n_frames,
            "n_atoms": int(row["n_atoms"]),
            "family_labels": "|".join(fams),
            "edit_family_id": str(cls_row.get("edit_family_id") or ""),
            "p1_orientation": str(p1_row.get("orientation") or ""),
            "exposure": False,
        })

    # dedup accounting over the eligible rows
    groups = {r["canonical_group"] for r in records}
    by_subset = Counter(r["subset"] for r in records)

    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage": "S0-A (metadata qualification; no raw IRC frame reads)",
        "truth_access": {
            "accessor": "truth_reader.load_irc_index (shape-only rows)",
            "n_audit_lines": 1,
            "note": "no trajectory/TS geometry read at this stage; the "
                    "single audited index read is recorded in "
                    "data/manifests/truth_access_log.jsonl",
        },
        "inputs": {
            "irc_index": {"via": "truth_reader.load_irc_index",
                          "n_rows": len(index)},
            "split_assignment": {"path": SPLIT_ASSIGNMENT,
                                 "sha256": sha256_file(repo / SPLIT_ASSIGNMENT)},
            "duplicate_ledger": {"path": DUP_LEDGER,
                                 "sha256": sha256_file(repo / DUP_LEDGER)},
            "g1_p1_summary": {"path": P1_SUMMARY,
                              "sha256": sha256_file(repo / P1_SUMMARY)},
            "g1_class_summary": {"path": CLASS_SUMMARY,
                                 "sha256": sha256_file(repo / CLASS_SUMMARY)},
            "g3p1_split": {"path": G3P1_SPLIT,
                           "sha256": sha256_file(repo / G3P1_SPLIT)},
        },
        "exposure_sets": {k: {kk: vv for kk, vv in v.items()
                              if kk != "reaction_ids"}
                          for k, v in sorted(exposure.items())},
        "exposure_total_ids": len(exposure_ids),
        "exposure_total_groups": len(exposure_groups),
        "counts": dict(counts),
        "eligible_by_subset": dict(by_subset),
        "eligible_canonical_groups": len(groups),
        "family_distribution": dict(sorted(family_counter.items())),
        "eligibility_rule": {
            "parent_split": "train",
            "g3p1_subset": "any of train_fit/train_dev/train_calibration",
            "min_n_frames": MIN_N_FRAMES,
            "p1_status": sorted(P1_STATUS_OK),
            "g2_eligible": True,
            "layout": LAYOUT_OK,
            "bond_events": ">= 1",
            "family": "present",
            "exposure": "excluded (campaign development-exposure set)",
        },
    }
    (out_dir / "metadata_qualification.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    import pyarrow as pa
    table = pa.Table.from_pylist(records)
    pq.write_table(table, out_dir / "eligible_reactions.parquet")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="r3_v53mp_s0a_qualify")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--out", default="outputs/pesn/v53mp/s0a")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    report = qualify(repo, Path(args.out), limit=args.limit)
    print(json.dumps({
        "counts": report["counts"],
        "eligible_canonical_groups": report["eligible_canonical_groups"],
        "eligible_by_subset": report["eligible_by_subset"],
        "exposure_total_ids": report["exposure_total_ids"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
