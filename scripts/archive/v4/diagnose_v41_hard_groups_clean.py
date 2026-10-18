#!/usr/bin/env python3
"""WP3 v4.1 item 9 — recompute the hard-group diversity diagnostic with CLEAN
5-channel features.

The historical ``diagnose_wp4_hard_groups.py`` reported a "diversity_gap" but
its distance was computed on the OLD 7-channel node_scalars (its metric
docstring said "5-channel" — the G1 finding-5 wording mismatch), which
included the teacher-label-derived ``near_zero_action_mask`` channel (P01).
This script does NOT modify or rerun the historical script: it recomputes the
same group-profile L2 metric on the OBSERVATION-ONLY 5-channel node_scalars
(v41 shards) and compares the unresponsive/responsive classification against
the frozen ``wp4_bc/g5_diagnosis_slice.json``.

Metric (own declaration): "observation-only 5-channel node_scalars L2,
state/atom-averaged" — per state, mean the 5-channel node_scalars over atoms;
a group's profile is the mean of those per-state atom-mean vectors over all
its states; distance between two groups = L2 between profiles; nearest fit
group = argmin over the 50 fit groups.  Zero new QC.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from diagnose_v41_common import (
    G5_SLICE_PATH, NEW_FIT_FEATURES, NEW_MON_FEATURES, OUT_DIR, write_json,
)

UNRESPONSIVE = ["RXN_0000004669", "RXN_0000006957", "RXN_0000010293"]
RESPONSIVE = ["RXN_0000000798", "RXN_0000001178", "RXN_0000002770",
              "RXN_0000006274", "RXN_0000009014"]


def group_ids_of(features_dir: Path):
    return sorted(p.stem.removeprefix("features_")
                  for p in features_dir.glob("features_*.npz"))


def profile(features_dir: Path, gid: str):
    z = np.load(features_dir / f"features_{gid}.npz", allow_pickle=False)
    ns = np.asarray(z["node_scalars"], dtype=np.float64)  # (S,N,5)
    per_state = ns.mean(axis=1)                           # (S,5)
    return per_state.mean(axis=0), per_state


def classify(dead_vals, resp_vals):
    dead_med = float(np.median(dead_vals))
    resp_med = float(np.median(resp_vals))
    ratio = dead_med / resp_med if resp_med > 0 else float("inf")
    if min(dead_vals) > max(resp_vals) or ratio >= 2.0:
        verdict = "diversity_gap"
    elif max(dead_vals) <= max(resp_vals) and ratio <= 1.5:
        verdict = "architecture_or_objective"
    else:
        verdict = "inconclusive_mixed"
    return {"dead_median": dead_med, "responsive_median": resp_med,
            "ratio": ratio, "verdict": verdict}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DIR / "v41_hard_groups_clean.json"))
    ap.add_argument("--md", default=str(OUT_DIR / "v41_hard_groups_clean.md"))
    args = ap.parse_args(argv)

    fit_profiles = {g: profile(NEW_FIT_FEATURES, g)[0]
                    for g in group_ids_of(NEW_FIT_FEATURES)}
    mon = {}
    for g in group_ids_of(NEW_MON_FEATURES):
        prof, per_state = profile(NEW_MON_FEATURES, g)
        dists = {n: float(np.linalg.norm(prof - fit_profiles[n]))
                 for n in fit_profiles}
        nearest = min(dists, key=lambda n: (dists[n], n))
        mon[g] = {
            "nearest_fit_group": nearest,
            "profile_l2_to_nearest_fit": dists[nearest],
            "mean_per_state_l2_to_nearest_fit_profile": float(
                np.linalg.norm(per_state - fit_profiles[nearest], axis=1).mean()),
        }

    dead_vals = [mon[g]["profile_l2_to_nearest_fit"] for g in UNRESPONSIVE]
    resp_vals = [mon[g]["profile_l2_to_nearest_fit"] for g in RESPONSIVE]
    clean = classify(dead_vals, resp_vals)

    old = json.loads(G5_SLICE_PATH.read_text())
    old_verdict = old["verdict"]["evidence_favors"]
    old_dead_med = old["verdict"]["dead_median"]
    old_resp_med = old["verdict"]["responsive_median"]
    old_ratio = old["verdict"]["dead_over_responsive_median_ratio"]

    result = {
        "schema_version": "pesn_v41_diag_hard_groups_clean_v1",
        "title": "WP3 v4.1 item 9: hard-group diversity diagnostic on CLEAN 5-ch",
        "metric_definition": "observation-only 5-channel node_scalars L2, "
                             "state/atom-averaged (group profile = mean over "
                             "states of per-state atom-mean of 5-ch "
                             "node_scalars; distance = L2 between profiles; "
                             "nearest = min over the 50 fit groups)",
        "回答什么": [
            "不含标签衍生通道（5ch）的覆盖距离重算及 unresponsive/responsive 分类",
            "与 v4 含标签通道版本（g5_diagnosis_slice.json）的分类对比",
        ],
        "不能声称什么": [
            "该距离只是 5 个 node scalar 的原子平均 L2，不是完整拓扑/流形相似度",
            "2.0 分类界限未预注册，diversity_gap 结论不得提升为因果根因",
        ],
        "clean_5ch": {
            "per_monitor_group": mon,
            "dead_profile_l2_values": dead_vals,
            "responsive_profile_l2_values": resp_vals,
            "classification": clean,
        },
        "historical_7ch_g5_slice": {
            "dead_median": old_dead_med,
            "responsive_median": old_resp_med,
            "ratio": old_ratio,
            "verdict": old_verdict,
        },
        "comparison": {
            "clean_verdict": clean["verdict"],
            "historical_verdict": old_verdict,
            "clean_ratio": clean["ratio"],
            "historical_ratio": old_ratio,
            "verdict_changed": clean["verdict"] != old_verdict,
        },
    }
    write_json(Path(args.out), result)

    md = ["# WP3 v4.1 条目9：干净 5 通道困难组多样性诊断", "",
          "| 回答什么 | 不能声称什么 |",
          "|---|---|",
          "| 5ch 覆盖距离重算与分类 | 距离≠完整拓扑/流形相似度 |",
          "| 与 v4 含标签通道版本对比 | 分类界限未预注册，不得提升为因果 |",
          "", "| monitor 组 | 最近 fit 组 | profile L2 |",
          "|---|---|---|"]
    for g, v in mon.items():
        tag = "UNRESP" if g in UNRESPONSIVE else "resp"
        md.append(f"| {g} [{tag}] | {v['nearest_fit_group']} | "
                  f"{v['profile_l2_to_nearest_fit']:.6f} |")
    md += ["", f"- **clean 5-ch 判定**: {clean['verdict']} "
               f"(dead median {clean['dead_median']:.4f}, "
               f"responsive median {clean['responsive_median']:.4f}, "
               f"ratio {clean['ratio']:.3f})",
           f"- **历史 7-ch 判定**: {old_verdict} "
               f"(ratio {old_ratio:.3f})",
           f"- 判定是否改变：{result['comparison']['verdict_changed']}"]
    Path(args.md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md).write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[item9] clean-5ch verdict = {clean['verdict']} "
          f"(ratio {clean['ratio']:.3f}) vs historical {old_verdict} "
          f"(ratio {old_ratio:.3f}); changed={result['comparison']['verdict_changed']}")
    print(f"[item9] wrote {args.out} and {args.md}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover
        print(f"DIAGNOSIS REFUSAL {exc!r}", file=sys.stderr)
        raise SystemExit(2)
