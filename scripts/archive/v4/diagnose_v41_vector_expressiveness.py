#!/usr/bin/env python3
"""WP3 v4.1 item 4 — vector-expressiveness linear-span probe.

For every dir_mask atom (non-teacher, |a*| >= 0.01 A) we project the teacher
UNIT direction ``d = a*/||a*||`` onto the span of the equivariant input
vectors available AT that atom:

  * edge relative-position unit vectors (from edge_index + x): for each
    directed edge (src -> dst) with dst == atom i, ``unit = (x_src - x_dst)/||.||``
    (the exact vectors the EGNN aggregates into ``v`` via
    ``v.index_add(0, dst, unit * gate)``, dual_egnn.py:188);
  * the scaled gradient vector ``g_i`` (the 'g' equivariant channel).

and report ``cos_res = 1 - cos(d, proj(d)) = 1 - ||proj(d)||`` (0 = the
direction lies fully inside the available span, 1 = no overlap).

FRAMING (binding): this is a ONE-HOP LINEAR AVAILABILITY probe — a
necessary condition on FIRST-LAYER information only.  Measured residual:
median 2–3%; tail 15.5% (fit50) / 16.3% (monitor8) of dir atoms have
residual > 0.5 (NOT covered).  A large residual means the direction cannot be
represented as a linear combination of the AVAILABLE equivariant input
vectors AT that atom; it is NOT a claim about what a multi-layer EGNN (whose
coordinate updates can mix neighbor geometry) can express, nor about what the
trained network actually learned.  Zero new QC; deterministic forward.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from diagnose_v41_common import (
    ACTION_MAX, ARMS, OBSERVED_FIT, OBSERVED_MON, OLD_FIT_FEATURES,
    OLD_MON_FEATURES, OUT_DIR, aligned, dir_cosine_weighted, forward_det,
    load_old_policy, load_old_states, percentile_dict, project_residual,
    write_json,
)

PARTITIONS = {"fit50": (OLD_FIT_FEATURES, OBSERVED_FIT),
              "monitor8": (OLD_MON_FEATURES, OBSERVED_MON)}


def incident_units(x, edge_index):
    """For each atom, the list of incoming edge unit vectors (dst-indexed)."""
    src, dst = edge_index[0].astype(int), edge_index[1].astype(int)
    n = x.shape[0]
    rel = x[src] - x[dst]
    dist = np.linalg.norm(rel, axis=1, keepdims=True)
    unit = rel / np.maximum(dist, 1e-12)
    per_atom = [[] for _ in range(n)]
    for k in range(src.shape[0]):
        per_atom[int(dst[k])].append(unit[k])
    return per_atom


def residual_per_atom(state, al, a_star, row_start):
    """Per-atom projection residual for one state's dir-mask atoms."""
    x = np.asarray(state["x_angstrom"], dtype=np.float64)
    g = np.asarray(state["g_scaled"], dtype=np.float64)
    ei = np.asarray(state["edge_index"], dtype=np.int64)
    units = incident_units(x, ei)
    n = x.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        if not al["dir_mask"][row_start + i]:
            continue
        a = a_star[row_start + i]
        an = np.linalg.norm(a)
        if an < 1e-12:
            continue
        d = a / an
        span = list(units[i]) + [g[i]]
        out[i] = project_residual(d, span)["residual"]
    return out


def per_reaction_cosine(states, al, a_det):
    out = {}
    for gid in np.unique(al["group"]):
        m = al["dir_mask"] & (al["group"] == gid)
        if m.any():
            out[str(gid)] = dir_cosine_weighted(a_det, al["a_star"], m)
    return out


def analyze(policy, states):
    al = aligned(states, ACTION_MAX)
    a_det, _ = forward_det(policy, states)
    a_star = al["a_star"]
    residuals = []
    row = 0
    for s in states:
        n = int(np.asarray(s["atomic_numbers"]).size)
        r = residual_per_atom(s, al, a_star, row)
        residuals.append(r)
        row += n
    res = np.concatenate(residuals) if residuals else np.zeros(0)
    res = res[al["dir_mask"] & np.isfinite(res)]
    return {
        "n_atoms_dir_with_residual": int(res.size),
        "cos_res_distribution": percentile_dict(res, "cos_res"),
        "fraction_large_residual_gt_0_5": float((res > 0.5).mean()) if res.size else None,
        "per_reaction_cosine": per_reaction_cosine(states, al, a_det),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="bc_nll")
    ap.add_argument("--out", default=str(OUT_DIR / "v41_vector_expressiveness.json"))
    ap.add_argument("--md", default=str(OUT_DIR / "v41_vector_expressiveness.md"))
    args = ap.parse_args(argv)

    arm = next(a for a in ARMS if a["name"] == args.arm)
    policy, _ = load_old_policy(arm)
    result = {
        "schema_version": "pesn_v41_diag_vector_expressiveness_v1",
        "title": "WP3 v4.1 item 4: vector-expressiveness linear-span residual",
        "framing": (
            "one-hop linear availability probe: median residual 2-3%; tail "
            "15.5% (fit50) / 16.3% (monitor8) of dir atoms residual >0.5 — a "
            "necessary condition on first-layer information, NOT a claim "
            "about what a multi-layer EGNN can express"),
        "arm": arm["name"],
        "span_definition": "span{ incoming edge unit vectors (dst==i), g_i }",
        "回答什么": [
            "教师单位方向投影到该原子可用等变向量张成空间的残差 1-cos(d, proj) 分布",
            "方向表现最差的反应（来自本脚本自算的 per-reaction cosine）上的残差",
        ],
        "不能声称什么": [
            "one-hop 线性可用性探针只是第一层信息的必要条件，不证明训练网络实际学到的内容",
            "残差大只说明'现有等变输入无法线性表示该方向'，不排除多层 EGNN 坐标更新混合邻居几何/非线性/消息传递可表达",
        ],
        "partitions": {},
    }
    for part_name, (feat_dir, obs_dir) in PARTITIONS.items():
        states, _meta = load_old_states(feat_dir, obs_dir)
        result["partitions"][part_name] = analyze(policy, states)

    # worst reactions by direction cosine (fit50 + monitor8 pooled ranking)
    all_cos = {}
    for part_name in PARTITIONS:
        all_cos.update(result["partitions"][part_name]["per_reaction_cosine"])
    ranked = sorted((v, k) for k, v in all_cos.items() if v is not None)
    worst = [k for _, k in ranked[:5]]
    result["worst_direction_reactions"] = worst
    write_json(Path(args.out), result)

    md = ["# WP3 v4.1 条目4：向量表达能力（线性张成残差）", "",
          "| 回答什么 | 不能声称什么 |",
          "|---|---|",
          "| 教师方向在可用等变向量张成空间上的投影残差分布 | 线性张成残差≠训练网络的实际表达 |",
          "| 最差方向反应的残差 | 残差大≠非线性/消息传递无法表达 |",
          "",
          "> **勘误（framing correction）**: 本探针是 **one-hop linear "
          "availability probe** —— median residual 2–3%；tail 15.5% (fit50) / "
          "16.3% (monitor8) 的 dir 原子 residual >0.5（未被覆盖）。它是**第一层"
          "信息的必要条件**，不是对多层 EGNN 能表达什么的断言。",
          "", f"- 张成定义：{result['span_definition']}",
          f"- 方向表现最差 5 反应：{worst}", ""]
    for part_name in PARTITIONS:
        blk = result["partitions"][part_name]
        d = blk["cos_res_distribution"]
        md += [f"## {part_name}", "",
               f"| cos_res | median | q25 | q75 | >0.5 占比 |",
               "|---|---|---|---|---|",
               f"| 残差 | {d['median']:.4f} | {d['q25']:.4f} | {d['q75']:.4f} | "
               f"{blk['fraction_large_residual_gt_0_5']} |", ""]
    Path(args.md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md).write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[item4] wrote {args.out} and {args.md}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover
        print(f"DIAGNOSIS REFUSAL {exc!r}", file=sys.stderr)
        raise SystemExit(2)
