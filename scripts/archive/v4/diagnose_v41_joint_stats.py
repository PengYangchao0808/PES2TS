#!/usr/bin/env python3
"""WP3 v4.1 item 1 — SAME-SAMPLE joint direction+magnitude statistics.

Replaces the INVALID group-level back-inference (F9).  For every OLD arm
(5 checkpoints) x {fit50, monitor8} we measure, on each ``dir_mask`` atom
(non-teacher AND |a*| >= 0.01 A):

  * alpha = ||a_det|| / ||a*||      (JOINT, measured directly per atom)
  * cos   = cos(a_det, a*)          (per atom)
  * mse   = ||a_det - a*||^2        (per atom)

and verify the PER-SAMPLE identity

    ||a_det - a*||^2 / ||a*||^2  ==  1 + alpha^2 - 2*alpha*cos

holds to floating accuracy per atom, while showing that substituting the
GROUP-LEVEL MSE ratio and the GROUP-LEVEL median cosine into the same
identity does NOT reproduce the directly-measured group median alpha (the
"16-23% magnitude contraction" back-inference is invalid).  We also report
the alpha-vs-cos joint distribution (alpha quartiles within cos quartile
bins) to answer whether any magnitude contraction concentrates on
direction-uncertain samples.

Deterministic (mean/mu) actions, torch.no_grad.  Reads OLD 7-channel
shards + OLD checkpoints.  Zero new QC.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

from diagnose_v41_common import (
    ACTION_MAX, ARMS, OBSERVED_FIT, OBSERVED_MON, OLD_FIT_FEATURES,
    OLD_MON_FEATURES, OUT_DIR, aligned, atom_alpha, atom_cos, atom_mse,
    forward_det, load_old_policy, load_old_states, med_iqr,
    percentile_dict, per_sample_identity, write_json,
)

PARTITIONS = {
    "fit50": (OLD_FIT_FEATURES, OBSERVED_FIT),
    "monitor8": (OLD_MON_FEATURES, OBSERVED_MON),
}


def _aggregate_per_state(al, alpha, cos, mse):
    """Per-state medians of alpha/cos and mean mse over dir_mask atoms."""
    state_idx = al["state_idx"]
    out = []
    for si in np.unique(state_idx):
        m = al["dir_mask"] & (state_idx == si)
        if not m.any():
            continue
        out.append({
            "state_idx": int(si),
            "alpha_median": float(np.median(alpha[m])),
            "cos_median": float(np.median(cos[m])),
            "mse_mean": float(mse[m].mean()),
        })
    return out


def _aggregate_per_reaction(al, alpha, cos, mse):
    """Per-reaction medians + MSE ratio (dir-mask atoms only)."""
    out = {}
    for gid in np.unique(al["group"]):
        m = al["dir_mask"] & (al["group"] == gid)
        if not m.any():
            continue
        mse_model = float(mse[m].mean())
        mse_zero = float((al["a_star"][m] ** 2).sum(-1).mean())
        out[str(gid)] = {
            "alpha_median": float(np.median(alpha[m])),
            "cos_median": float(np.median(cos[m])),
            "mse_ratio_vs_zero": mse_model / mse_zero if mse_zero > 0 else None,
            "mean_mse_over_mean_norm": mse_model,
            "n_dir_atoms": int(m.sum()),
        }
    return out


def _alpha_by_cos_quartile(al, alpha, cos):
    """alpha quartiles within 4 equal-count cos-quartile bins."""
    m = al["dir_mask"]
    a, c = alpha[m], cos[m]
    if a.size == 0:
        return []
    order = np.argsort(c, kind="stable")
    a, c = a[order], c[order]
    qbins = np.array_split(np.arange(a.size), 4)
    out = []
    for k, idx in enumerate(qbins):
        out.append({
            "cos_quartile": k + 1,
            "cos_range": [float(c[idx].min()), float(c[idx].max())],
            "n": int(idx.size),
            "alpha_median": float(np.median(a[idx])),
            "alpha_q25": float(np.percentile(a[idx], 25)),
            "alpha_q75": float(np.percentile(a[idx], 75)),
        })
    return out


def _back_inference_invalid(al, alpha, cos, mse):
    """Demonstrate group-level back-inference disagrees with direct alpha."""
    deltas, n_noreal, n_reactions = [], 0, 0
    for gid in np.unique(al["group"]):
        m = al["dir_mask"] & (al["group"] == gid)
        if not m.any():
            continue
        n_reactions += 1
        r = float(mse[m].mean()) / max(float((al["a_star"][m] ** 2).sum(-1).mean()), 1e-30)
        c = float(np.median(cos[m]))
        alpha_true = float(np.median(alpha[m]))
        disc = c ** 2 + r - 1.0
        if disc < 0:
            n_noreal += 1
            continue
        alpha_hat = c + math.sqrt(disc)
        deltas.append(alpha_hat - alpha_true)
    return {
        "n_reactions": n_reactions,
        "n_no_real_root": n_noreal,
        "alpha_hat_minus_alpha_true": percentile_dict(deltas, "delta"),
        "median_abs_delta": float(np.median(np.abs(np.asarray(deltas))))
        if deltas else None,
    }


def analyze(policy, states, action_max):
    al = aligned(states, action_max)
    a_det, _sigma = forward_det(policy, states, action_max=action_max)
    dmask = al["dir_mask"]
    mse = atom_mse(a_det, al["a_star"])
    alpha = atom_alpha(a_det, al["a_star"])
    cos = atom_cos(a_det, al["a_star"])
    resid, alpha_dm, cos_dm = per_sample_identity(a_det, al["a_star"], dmask)
    per_state = _aggregate_per_state(al, alpha, cos, mse)
    per_rxn = _aggregate_per_reaction(al, alpha, cos, mse)
    return {
        "population": {
            "n_states": len(states),
            "n_atoms_dir": int(dmask.sum()),
            "n_atoms_valid": int(al["valid"].sum()),
            "n_reactions": int(len(np.unique(al["group"]))),
        },
        "per_atom_dir_mask": {
            "alpha": percentile_dict(alpha[dmask], "alpha"),
            "cosine": percentile_dict(cos[dmask], "cosine"),
            "mse_A2": percentile_dict(mse[dmask], "mse"),
        },
        "per_sample_identity": {
            "residual_max_abs": float(np.abs(resid).max()) if resid.size else None,
            "residual_median_abs": float(np.median(np.abs(resid))) if resid.size else None,
            "note": "||a_det-a*||^2/||a*||^2 - (1+alpha^2-2*alpha*cos) per atom; "
                    "~0 demonstrates the identity holds PER-SAMPLE only.",
        },
        "per_state": {
            "alpha_median": med_iqr([s["alpha_median"] for s in per_state]),
            "cos_median": med_iqr([s["cos_median"] for s in per_state]),
            "mse_mean": med_iqr([s["mse_mean"] for s in per_state]),
        },
        "per_reaction": {
            "alpha_median": med_iqr([v["alpha_median"] for v in per_rxn.values()]),
            "cos_median": med_iqr([v["cos_median"] for v in per_rxn.values()]),
            "mse_ratio_vs_zero": med_iqr(
                [v["mse_ratio_vs_zero"] for v in per_rxn.values()]),
            "n_dir_atoms_per_reaction": med_iqr(
                [v["n_dir_atoms"] for v in per_rxn.values()]),
        },
        "alpha_by_cos_quartile": _alpha_by_cos_quartile(al, alpha, cos),
        "group_back_inference_invalid": _back_inference_invalid(al, alpha, cos, mse),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DIR / "v41_joint_stats.json"))
    ap.add_argument("--md", default=str(OUT_DIR / "v41_joint_stats.md"))
    ap.add_argument("--arms", default=None,
                    help="comma list of arm names (default all 5)")
    args = ap.parse_args(argv)

    arm_names = args.arms.split(",") if args.arms else [a["name"] for a in ARMS]
    arms = [a for a in ARMS if a["name"] in arm_names]

    result = {
        "schema_version": "pesn_v41_diag_joint_stats_v1",
        "title": "WP3 v4.1 item 1: same-sample joint direction+magnitude stats",
        "回答什么": [
            "同一样本上直接测量的幅度比 alpha=||a_det||/||a*|| 与方向余弦 cos 的联合分布（逐原子/逐状态/逐反应聚合）",
            "逐样本恒等式 ||a_det-a*||^2/||a*||^2 == 1+alpha^2-2alpha*cos 的逐原子残差（应 ~1e-12）",
            "用组级 MSE 比值 + 组级余弦中位数反推 alpha 与直接测量的组级 alpha 中位数的偏差（证明反推无效）",
            "幅度收缩（alpha<1）是否集中在方向不确定（低 cos）样本上",
        ],
        "不能声称什么": [
            "该恒等式不可跨样本/跨权重代入；组级 MSE 比值 x 余弦中位数不能反推幅度",
            "alpha 的组级数值不能解释为'路线学会了收缩'或'数据缺陷已排除'",
            "方向余弦低本身不构成根因证据；低梯度-动作相关也不能单独证明任何机制",
        ],
        "a_max_angstrom": ACTION_MAX,
        "arms": {},
    }
    for arm in arms:
        policy, _ckpt = load_old_policy(arm)
        arm_block = {}
        for part_name, (feat_dir, obs_dir) in PARTITIONS.items():
            states, _meta = load_old_states(feat_dir, obs_dir)
            arm_block[part_name] = analyze(policy, states, ACTION_MAX)
        result["arms"][arm["name"]] = arm_block

    write_json(Path(args.out), result)

    # ---- Markdown ----------------------------------------------------------
    md = ["# WP3 v4.1 条目1：同一样本联合方向+幅度统计",
          "",
          "## 回答什么 / 不能声称什么",
          "",
          "| 回答什么 | 不能声称什么 |",
          "|---|---|",
          "| 逐原子直接测量 alpha=‖â‖/‖a*‖ 与 cos 的联合分布 | 恒等式不可跨样本/跨权重代入 |",
          "| 逐样本恒等式逐原子残差（应≈1e-12） | 组级 MSE比值×余弦中位数不能反推幅度 |",
          "| 反推 alpha 与直接测量 alpha 的偏差 | alpha 数值不等于'学会收缩'或'数据缺陷已排除' |",
          "| 幅度收缩是否集中在低 cos 样本 | 低余弦本身不构成根因/机制证明 |",
          ""]
    for arm in arms:
        md.append(f"## arm {arm['name']}")
        for part_name in ("fit50", "monitor8"):
            blk = result["arms"][arm["name"]][part_name]
            pa = blk["per_atom_dir_mask"]
            md.append(f"### {part_name}  (n_atoms_dir={blk['population']['n_atoms_dir']})")
            md.append("")
            md.append("| 量 | median | q25 | q75 | mean |")
            md.append("|---|---|---|---|---|")
            for label, key in (("alpha=‖â‖/‖a*‖", "alpha"),
                               ("cosine", "cosine"), ("mse (Å²)", "mse_A2")):
                d = pa[key]
                md.append(f"| {label} | {d['median']:.4f} | {d['q25']:.4f} | "
                          f"{d['q75']:.4f} | {d['mean']:.4f} |")
            ident = blk["per_sample_identity"]
            md.append("")
            md.append(f"- 逐样本恒等式残差 max|·| = {ident['residual_max_abs']:.3e} "
                      f"(median {ident['residual_median_abs']:.3e}) — 仅在逐样本成立")
            bbi = blk["group_back_inference_invalid"]
            md.append(f"- 组级反推 alpha 与直接测量 alpha 中位数之差的 median|Δ| = "
                      f"{bbi['median_abs_delta']} "
                      f"({bbi['n_no_real_root']}/{bbi['n_reactions']} 组无实数解)")
            md.append("")
            md.append("**alpha 随 cos 四分位**：")
            for row in blk["alpha_by_cos_quartile"]:
                md.append(f"- cos Q{row['cos_quartile']} "
                          f"({row['cos_range'][0]:.3f}..{row['cos_range'][1]:.3f}): "
                          f"alpha median {row['alpha_median']:.4f} "
                          f"[{row['alpha_q25']:.4f},{row['alpha_q75']:.4f}] n={row['n']}")
            md.append("")
    Path(args.md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md).write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[item1] wrote {args.out} and {args.md}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover
        print(f"DIAGNOSIS REFUSAL {exc!r}", file=sys.stderr)
        raise SystemExit(2)
