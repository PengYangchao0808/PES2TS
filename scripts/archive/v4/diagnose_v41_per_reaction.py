#!/usr/bin/env python3
"""WP3 v4.1 item 5 — per-reaction (fit50 group) error profile.

For each of the 50 fit groups, under the OLD bc_nll arm (deterministic
actions): train-partition MSE ratio vs the zero-action baseline AND absolute
MSE, |a|-weighted direction cosine, effective-action fraction
(# dir_mask atoms / # valid atoms), and the near-TS fraction (frozen rule:
remaining_arc <= 0.2 * branch max).  Groups are sorted and bottom-quartile
groups are flagged — this is where an aggregate mean hides complete failure.

Zero new QC; reads OLD 7-ch shards + bc_nll checkpoint.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from diagnose_v41_common import (
    ACTION_MAX, ARMS, OBSERVED_FIT, OLD_FIT_FEATURES, OUT_DIR, aligned,
    atom_mse, dir_cosine_weighted, forward_det, load_old_policy,
    load_old_states, mse_ratio_vs_zero, write_json,
)

NEAR_TS_ARC_FRACTION = 0.2


def near_ts_mask(al):
    """Frozen per-branch near-TS stratum rule (remaining_arc <= 0.2 * branch max)."""
    nt = np.zeros(al["ral"].shape, dtype=bool)
    for br in np.unique(al["branch"]):
        m = al["branch"] == br
        ral = al["ral"][m]
        nt[m] = ral <= NEAR_TS_ARC_FRACTION * float(ral.max())
    return nt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="bc_nll")
    ap.add_argument("--out", default=str(OUT_DIR / "v41_per_reaction.json"))
    ap.add_argument("--md", default=str(OUT_DIR / "v41_per_reaction.md"))
    args = ap.parse_args(argv)

    arm = next(a for a in ARMS if a["name"] == args.arm)
    states, _meta = load_old_states(OLD_FIT_FEATURES, OBSERVED_FIT)
    policy, _ = load_old_policy(arm)
    al = aligned(states, ACTION_MAX)
    a_det, _ = forward_det(policy, states)
    nt = near_ts_mask(al)

    rows = {}
    for gid in np.unique(al["group"]):
        valid = al["valid"] & (al["group"] == gid)
        dmask = al["dir_mask"] & (al["group"] == gid)
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue
        mse_abs = float(atom_mse(a_det, al["a_star"])[valid].mean())
        mse_ratio = mse_ratio_vs_zero(a_det, al["a_star"], valid)
        cos = dir_cosine_weighted(a_det, al["a_star"], dmask)
        eff_frac = float(dmask.sum()) / float(n_valid)
        near_ts_frac = float((valid & nt).sum()) / float(n_valid)
        rows[str(gid)] = {
            "group_id": str(gid),
            "mse_ratio_vs_zero": mse_ratio,
            "mse_abs_A2": mse_abs,
            "direction_cosine_absa_weighted": cos,
            "effective_action_fraction": eff_frac,
            "near_ts_fraction": near_ts_frac,
            "n_valid_atoms": n_valid,
            "n_dir_atoms": int(dmask.sum()),
        }

    ranked = sorted(rows.values(), key=lambda r: (
        r["direction_cosine_absa_weighted"]
        if r["direction_cosine_absa_weighted"] is not None else -1.0))
    n = len(ranked)
    bottom_quartile = [r["group_id"] for r in ranked[: max(1, n // 4)]]

    result = {
        "schema_version": "pesn_v41_diag_per_reaction_v1",
        "title": "WP3 v4.1 item 5: per-reaction (fit50) error profile",
        "arm": arm["name"],
        "回答什么": [
            "50 个 fit 组各自的 MSE 比值+绝对值、方向余弦、有效动作比例、近 TS 占比",
            "排序表并标记下四分位组（聚合均值掩盖完全失败的组）",
        ],
        "不能声称什么": [
            "这是训练分区（fit）自诊断，不是 monitor 门结果，不用于选型",
            "组级失败不等于'该反应不可学'；样本量/路径差异未控制",
        ],
        "bottom_quartile_group_ids": bottom_quartile,
        "n_groups": n,
        "per_group_sorted": ranked,
    }
    write_json(Path(args.out), result)

    md = ["# WP3 v4.1 条目5：逐反应（fit50）误差画像", "",
          "| 回答什么 | 不能声称什么 |",
          "|---|---|",
          "| 逐组 MSE比值/绝对值、余弦、有效动作比、近TS占比 | fit 自诊断，非 monitor 门，不选型 |",
          "| 下四分位组标记 | 组级失败≠不可学 |",
          "", f"- 下四分位（方向余弦最低）组：{bottom_quartile}", "",
          "| group | cos (|a|加权) | MSE比值 | MSE绝对(Å²) | 有效动作比 | 近TS占比 |",
          "|---|---|---|---|---|---|"]
    for r in ranked:
        flag = " ⚑" if r["group_id"] in bottom_quartile else ""
        md.append(f"| {r['group_id']}{flag} | {r['direction_cosine_absa_weighted']} "
                  f"| {r['mse_ratio_vs_zero']:.4f} | {r['mse_abs_A2']:.5f} "
                  f"| {r['effective_action_fraction']:.3f} | "
                  f"{r['near_ts_fraction']:.3f} |")
    Path(args.md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md).write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[item5] wrote {args.out} and {args.md}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover
        print(f"DIAGNOSIS REFUSAL {exc!r}", file=sys.stderr)
        raise SystemExit(2)
