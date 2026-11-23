#!/usr/bin/env python3
"""WP3 v4.1 item 3 — original teacher nodes vs inserted interpolation midpoints.

Classification provenance (DOCUMENTED rule): WP2 subdivided over-cap steps
(|a| > 0.1 A) by LINEAR midpoint insertion.  The insertion provenance is NOT
carried in the teacher.json payload (states are renumbered 0..n-1); it lives
in the WP2 gaps registry ``items.inserted_nodes.{A,B}`` (per-insert record
with ``inserted_state_id``, ``between_original_steps``, ``trigger magnitude``)
and in ``coverage_nodes.json`` (``inserted`` flag).  A fit50 state is an
*inserted midpoint* iff its state_id appears in the gaps-registry inserted
lists; everything else is an *original teacher node*.  This script documents
and applies that rule (no supervision-weight change, output only).

Per class (original vs inserted), for the selected bc_nll arm, over dir-mask
atoms: direction cosine (|a|-weighted), alpha = ||a_det||/||a*|| median,
MSE ratio vs zero baseline, and the parameter-gradient contribution share
(same recipe-loss ||dL/dtheta|| methodology as item 2, batches split by
original vs inserted).  Zero new QC.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from diagnose_v41_common import (
    ACTION_MAX, ARMS, OBSERVED_FIT, OLD_FIT_FEATURES, OUT_DIR, aligned,
    atom_alpha, atom_cos, atom_mse, device_of, dir_cosine_weighted,
    forward_det, grad_norm, load_inserted_state_ids, load_old_policy,
    load_old_states, med_iqr, mse_ratio_vs_zero, recipe_combine,
    recipe_loss_terms, write_json,
)

BATCH_SIZE = 64
N_BATCHES = 8
GRAD_SEED = 20261017


def classify(states, inserted_ids):
    """Return (inserted_idx, original_idx) lists over ACTION states."""
    ins, orig = [], []
    for i, s in enumerate(states):
        if s["is_teacher"]:
            continue
        (ins if s["state_id"] in inserted_ids else orig).append(i)
    return ins, orig


def class_block(policy, states, class_idx, al, a_det):
    """Direction/alpha/mse metrics over one class of action states."""
    mask = np.zeros(len(states), dtype=bool)
    mask[np.asarray(class_idx, dtype=int)] = True
    atom_mask = np.repeat(mask, al["n_atoms"])
    dmask = al["dir_mask"] & atom_mask
    valid_c = al["valid"] & atom_mask
    alpha = atom_alpha(a_det, al["a_star"])
    cos = atom_cos(a_det, al["a_star"])
    mse = atom_mse(a_det, al["a_star"])
    return {
        "n_states": int(len(class_idx)),
        "n_atoms_dir": int(dmask.sum()),
        "n_atoms_valid": int(valid_c.sum()),
        "direction_cosine_absa_weighted": dir_cosine_weighted(
            a_det, al["a_star"], dmask),
        "alpha_median": float(np.median(alpha[dmask])) if dmask.any() else None,
        "mse_ratio_vs_zero": mse_ratio_vs_zero(a_det, al["a_star"], valid_c),
        "cos_median": float(np.median(cos[dmask])) if dmask.any() else None,
    }


def grad_share(policy, states, class_idx, recipe):
    device = device_of()
    policy.train_mode(True)
    norms = []
    rng = np.random.default_rng(GRAD_SEED)
    idx = np.asarray(class_idx, dtype=int)
    need = N_BATCHES * BATCH_SIZE
    if len(idx) >= need:
        idx = idx[rng.permutation(len(idx))[:need]]
    else:
        idx = idx[rng.integers(0, len(idx), size=need)]
    for i in range(0, need, BATCH_SIZE):
        chunk = [states[int(j)] for j in idx[i:i + BATCH_SIZE]]
        policy.module.zero_grad(set_to_none=True)
        nll_loss, mse_loss, dir_loss = recipe_loss_terms(
            policy, chunk, ACTION_MAX, device)
        loss = recipe_combine(nll_loss, mse_loss, dir_loss, recipe)
        norms.append(grad_norm(loss, policy))
    return float(np.mean(np.asarray(norms, dtype=np.float64)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="bc_nll")
    ap.add_argument("--out", default=str(OUT_DIR / "v41_raw_vs_interp.json"))
    ap.add_argument("--md", default=str(OUT_DIR / "v41_raw_vs_interp.md"))
    args = ap.parse_args(argv)

    arm = next(a for a in ARMS if a["name"] == args.arm)
    inserted_ids = load_inserted_state_ids()
    states, _meta = load_old_states(OLD_FIT_FEATURES, OBSERVED_FIT)
    ins_idx, orig_idx = classify(states, inserted_ids)
    policy, _ = load_old_policy(arm)
    al = aligned(states, ACTION_MAX)
    a_det, _ = forward_det(policy, states)

    ins_block = class_block(policy, states, ins_idx, al, a_det)
    orig_block = class_block(policy, states, orig_idx, al, a_det)
    ins_grad = grad_share(policy, states, ins_idx, arm["recipe"])
    orig_grad = grad_share(policy, states, orig_idx, arm["recipe"])

    result = {
        "schema_version": "pesn_v41_diag_raw_vs_interp_v1",
        "title": "WP3 v4.1 item 3: original vs inserted-midpoint states",
        "arm": arm["name"],
        "insertion_provenance_rule": {
            "source": "wp2_fit50/gaps_registry.json items.inserted_nodes.{A,B}",
            "marker": "inserted_state_id (states renumbered; no marker in "
                      "teacher.json itself)",
            "rule": "state is inserted-midpoint iff its state_id is in the "
                    "gaps-registry inserted lists (linear midpoint, WP2 "
                    "a_max=0.1 A exceedance subdivision)",
            "n_inserted_states": int(len(ins_idx)),
            "n_original_states": int(len(orig_idx)),
        },
        "回答什么": [
            "原始教师节点 vs 插值中点两类状态的方向余弦/alpha/MSE 差异（逐类）",
            "两类状态对参数梯度的贡献份额（同一 bc_nll 配方损失 ||dL/dtheta||）",
            "插值中点的分类来源（gaps registry 规则）",
        ],
        "不能声称什么": [
            "只报贡献，不改变监督权重；不构成'插值点有害/有益'的因果结论",
            "插值点数量少（399/6067），类间统计不等价于插值策略评估",
        ],
        "per_class": {
            "inserted_midpoints": ins_block,
            "original_nodes": orig_block,
        },
        "gradient_contribution": {
            "inserted_mean_grad_norm": ins_grad,
            "original_mean_grad_norm": orig_grad,
            "inserted_over_original_ratio": ins_grad / orig_grad if orig_grad > 0 else None,
        },
    }
    write_json(Path(args.out), result)

    md = ["# WP3 v4.1 条目3：原始节点 vs 插值中点", "",
          "| 回答什么 | 不能声称什么 |",
          "|---|---|",
          "| 两类状态方向/幅度/MSE 差异 | 只报贡献，不调监督权重 |",
          "| 两类梯度贡献份额 | 不等价于插值策略评估 |",
          "", f"- 插值分类规则：{result['insertion_provenance_rule']['rule']}",
          f"- 插值状态数 {len(ins_idx)}，原始状态数 {len(orig_idx)}",
          "", "| 类 | n_states | cos (|a|加权) | alpha median | MSE比值 |",
          "|---|---|---|---|---|"]
    for label, b in (("插值中点", ins_block), ("原始节点", orig_block)):
        md.append(f"| {label} | {b['n_states']} | {b['direction_cosine_absa_weighted']} "
                  f"| {b['alpha_median']} | {b['mse_ratio_vs_zero']} |")
    md += ["", f"- 梯度范数：插值均值 {ins_grad:.3e}，原始均值 {orig_grad:.3e}，"
               f"比值 {result['gradient_contribution']['inserted_over_original_ratio']}"]
    Path(args.md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md).write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[item3] wrote {args.out} and {args.md}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover
        print(f"DIAGNOSIS REFUSAL {exc!r}", file=sys.stderr)
        raise SystemExit(2)
