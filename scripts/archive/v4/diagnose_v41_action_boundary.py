#!/usr/bin/env python3
"""WP3 v4.1 item 2 — teacher action boundary + gradient contribution.

Read-only.  Four measured facts:

  1. Teacher action norm distribution (per atom, per reaction), distance to
     the action cap a_max = 0.1 A (``DEFAULT_A_MAX_ANGSTROM``,
     ``pes2ts_core/pesn/models/dual_ts_policy.py:36``; radial tanh mapping
     ``R(u) = tanh(||u||)u/||u||`` in ``direct_policy.py:51-55``
     ``radial_transform_stable``), fraction of atoms within 10%/1% of the cap,
     and the inverse-transform latent norm ||R^-1(a*)|| = ||u*||
     (``dual_ts_bc.latent_targets``).

  2. Loss-contribution share of top-decile-||a*|| states (fraction of total
     action-MSE mass living in the 10% of states with the largest per-state
     max atom norm) — teacher side ||a*||^2 AND model side ||a_det-a*||^2.

  3. PARAMETER-GRADIENT contribution for the 3 main arms (bc_nll, bc_a,
     bc_d): ||dL/dtheta|| of the recipe loss measured on top-decile vs
     remaining batches (>=5 batches each, same batch size, same update count
     = the loaded 20k-update checkpoint).  Reported as % of gradient norm
     from boundary-approaching samples.

  FRAMING (binding): the radial-tanh Jacobian term ``log sech^2(r) +
  2 log(tanh(r)/r)`` is evaluated AT the FIXED teacher target u*, so it is a
  CONSTANT with respect to parameters (``d/dtheta log_det(u*) = 0``).  We do
  NOT claim it "pushes" the network.  We only report the inverse-transform
  scale effects (||u*|| -> huge near the cap) and the measured gradient
  magnitudes, exactly as computed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from diagnose_v41_common import (
    ACTION_MAX, ARMS, OBSERVED_FIT, OLD_FIT_FEATURES, OUT_DIR, aligned,
    atom_mse, device_of, forward_det, grad_norm, load_old_policy,
    load_old_states, percentile_dict, recipe_combine, recipe_loss_terms,
    write_json,
)

#: Per-state max-atom-norm decile for the boundary / gradient split.
TOP_DECILE = 0.10
N_BATCHES = 8
BATCH_SIZE = 64
GRAD_SEED = 20261017


def teacher_boundary(states):
    """Teacher-side boundary facts (no model): norms, cap distance, ||u*||."""
    al = aligned(states, ACTION_MAX)
    a_star = al["a_star"]
    valid = al["valid"]
    norms = np.linalg.norm(a_star, axis=1)  # (N,) all atoms (incl. terminal zeros)
    vn = norms[valid]
    dist_to_cap = ACTION_MAX - vn
    within_10pct = float((vn >= 0.90 * ACTION_MAX).mean()) if vn.size else None
    within_1pct = float((vn >= 0.99 * ACTION_MAX).mean()) if vn.size else None
    # inverse-transform latent norm ||u*|| = arctanh(||a*||/a_max)
    u_star_norm = np.linalg.norm(al["u_star"], axis=1)[valid]

    # per-reaction max atom norm
    per_rxn = {}
    for gid in np.unique(al["group"]):
        m = valid & (al["group"] == gid)
        if m.any():
            per_rxn[str(gid)] = float(norms[m].max())

    # per-state max atom norm (ranking key for the top-decile split)
    state_max = np.asarray([
        float(norms[al["state_idx"] == si].max()) if (al["state_idx"] == si).any() else 0.0
        for si in range(len(states))
    ], dtype=np.float64)
    state_max = state_max[~np.asarray([s["is_teacher"] for s in states], dtype=bool)]
    return {
        "a_max_angstrom": ACTION_MAX,
        "radial_map_cite": "direct_policy.py:51-55 radial_transform_stable; "
                           "dual_ts_policy.py:36 DEFAULT_A_MAX_ANGSTROM=0.1",
        "teacher_action_norm_A": percentile_dict(vn, "norm"),
        "distance_to_cap_A": percentile_dict(dist_to_cap, "dist"),
        "fraction_within_10pct_of_cap": within_10pct,
        "fraction_within_1pct_of_cap": within_1pct,
        "inverse_latent_norm_abs_u_star": percentile_dict(u_star_norm, "u*"),
        "per_reaction_max_atom_norm_A": {
            "min": min(per_rxn.values()), "median": float(np.median(list(per_rxn.values()))),
            "max": max(per_rxn.values()), "reactions": per_rxn,
        },
        "state_max_norm_top_decile_threshold": float(np.quantile(
            state_max, 1.0 - TOP_DECILE)) if state_max.size else None,
    }


def top_decile_split(states):
    """Action states split into (top_decile, remaining) by per-state max ||a*||."""
    drawable = [s for s in states if not s["is_teacher"]]
    state_max = []
    for s in drawable:
        a = np.asarray(s["action"], dtype=np.float64)
        state_max.append(float(np.linalg.norm(a, axis=1).max()))
    order = np.argsort(np.asarray(state_max, dtype=np.float64))[::-1]
    n_top = max(1, int(np.ceil(len(drawable) * TOP_DECILE)))
    top = [drawable[i] for i in order[:n_top]]
    rest = [drawable[i] for i in order[n_top:]]
    return top, rest


def _sample_batches(states, n_batches, batch_size, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(states))
    need = n_batches * batch_size
    if len(states) >= need:
        idx = idx[:need]
    else:
        # with replacement if the stratum is smaller than the sample budget
        idx = rng.integers(0, len(states), size=need)
    return [list(np.asarray(states)[idx[i:i + batch_size]])
            for i in range(0, need, batch_size)]


def grad_contribution(policy, states, recipe):
    """||dL/dtheta|| on top-decile vs remaining batches (3-term recipe loss)."""
    top, rest = top_decile_split(states)
    device = device_of()
    policy.train_mode(True)
    norms = {"top_decile": [], "remaining": []}
    for label, group in (("top_decile", top), ("remaining", rest)):
        n_batches = min(N_BATCHES, max(1, len(group) // BATCH_SIZE))
        batches = _sample_batches(group, n_batches, BATCH_SIZE, GRAD_SEED)
        for chunk in batches:
            policy.module.zero_grad(set_to_none=True)
            nll_loss, mse_loss, dir_loss = recipe_loss_terms(
                policy, chunk, ACTION_MAX, device)
            loss = recipe_combine(nll_loss, mse_loss, dir_loss, recipe)
            norms[label].append(grad_norm(loss, policy))
    top_arr = np.asarray(norms["top_decile"], dtype=np.float64)
    rest_arr = np.asarray(norms["remaining"], dtype=np.float64)
    n_top, n_rest = len(top), len(rest)
    top_total = float(n_top) * float(top_arr.mean())
    rest_total = float(n_rest) * float(rest_arr.mean())
    pop_weighted = top_total / (top_total + rest_total) if (top_total + rest_total) > 0 else None
    return {
        "n_batches_each": int(len(top_arr)),
        "n_top_states": n_top,
        "n_remaining_states": n_rest,
        "top_decile_mean_grad_norm": float(top_arr.mean()),
        "remaining_mean_grad_norm": float(rest_arr.mean()),
        "per_sample_gradient_ratio_top_over_remaining": (
            float(top_arr.mean()) / float(rest_arr.mean())
            if float(rest_arr.mean()) > 0 else None),
        "top_decile_share_equal_batch": (
            float(top_arr.mean()) / float(top_arr.mean() + rest_arr.mean())
            if float(top_arr.mean() + rest_arr.mean()) > 0 else None),
        "top_decile_share_population_weighted": pop_weighted,
        "top_decile_grad_norms": [float(v) for v in top_arr],
        "remaining_grad_norms": [float(v) for v in rest_arr],
        "recipe": recipe,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DIR / "v41_action_boundary.json"))
    ap.add_argument("--md", default=str(OUT_DIR / "v41_action_boundary.md"))
    args = ap.parse_args(argv)

    states, _meta = load_old_states(OLD_FIT_FEATURES, OBSERVED_FIT)
    tb = teacher_boundary(states)

    # model-side loss share (bc_nll arm, deterministic)
    policy_nll, _ = load_old_policy(ARMS[0])
    al = aligned(states, ACTION_MAX)
    a_det, _ = forward_det(policy_nll, states)
    mse = atom_mse(a_det, al["a_star"])
    # rank by state-level max ||a*|| over DRAWABLE states
    drawable_mask = ~al["is_teacher"]
    state_max = np.zeros(len(states), dtype=np.float64)
    for si in range(len(states)):
        m = al["state_idx"] == si
        if m.any():
            state_max[si] = float(np.linalg.norm(al["a_star"][m], axis=1).max())
    draw_idx = [i for i, s in enumerate(states) if not s["is_teacher"]]
    sm = state_max[draw_idx]
    order = np.argsort(sm)[::-1]
    n_top = max(1, int(np.ceil(len(draw_idx) * TOP_DECILE)))
    top_set = set(int(i) for i in order[:n_top])
    top_mask = np.zeros(len(states), dtype=bool)
    for k, si in enumerate(draw_idx):
        top_mask[si] = k in top_set
    atom_top = np.repeat(top_mask, al["n_atoms"]) & al["valid"]
    atom_rest = ~np.repeat(top_mask, al["n_atoms"]) & al["valid"]
    teacher_share = float((al["a_star"][atom_top] ** 2).sum(-1).sum()) / max(
        float((al["a_star"][atom_top | atom_rest] ** 2).sum(-1).sum()), 1e-30)
    model_share = float(mse[atom_top].sum()) / max(float(mse[atom_top | atom_rest].sum()), 1e-30)

    grad_by_arm = {}
    for arm in ARMS:
        if arm["recipe"] not in ("bc_nll", "bc_a", "bc_d"):
            continue
        policy, _ = load_old_policy(arm)
        grad_by_arm[arm["name"]] = grad_contribution(policy, states, arm["recipe"])

    result = {
        "schema_version": "pesn_v41_diag_action_boundary_v1",
        "title": "WP3 v4.1 item 2: teacher action boundary + gradient contribution",
        "回答什么": [
            "教师动作范数分布、距 a_max=0.1 A 的距离、10%/1% 近界占比、逆变换潜范数 ||u*||",
            "top-decile ||a*|| 状态对总动作 MSE 的贡献份额（教师侧 ||a*||^2 与模型侧 ||a_det-a*||^2）",
            "3 个主臂在 top-decile vs 其余批次的 ||dL/dtheta|| 均值及近界样本的梯度范数占比",
        ],
        "不能声称什么": [
            "固定教师目标下 Jacobian 项对参数梯度为零，不得声称其'推动'网络；只报告逆变换尺度效应",
            "梯度范数占比不是'边界样本导致了失败'的因果证明",
            "教师动作近界不等于模型动作饱和（模型确定性动作受同一 a_max 径向 tanh 约束）",
        ],
        "teacher_boundary": tb,
        "loss_share_top_decile": {
            "top_decile_fraction": TOP_DECILE,
            "teacher_action_mse_share": teacher_share,
            "model_action_mse_share": model_share,
            "note": "state ranked by per-state max atom ||a*||; share over valid atoms",
        },
        "parameter_gradient_contribution": grad_by_arm,
    }
    write_json(Path(args.out), result)

    md = ["# WP3 v4.1 条目2：动作边界与参数梯度贡献", "",
          "| 回答什么 | 不能声称什么 |",
          "|---|---|",
          "| 教师动作范数/距界/10%·1%近界占比/‖u*‖ | Jacobian 项对参数梯度为零，不得声称其推动网络 |",
          "| top-decile 状态对动作 MSE 的贡献份额 | 份额≠因果根因 |",
          "| 3臂 ‖∂L/∂θ‖ top-decile vs 其余 | 近界≠模型动作饱和 |",
          "", "## 教师侧边界", "",
          f"- a_max = {tb['a_max_angstrom']} A（{tb['radial_map_cite']}）", "",
          "| 量 | median | q25 | q75 | max |", "|---|---|---|---|---|"]
    for label, key in (("教师动作范数 (Å)", "teacher_action_norm_A"),
                       ("距界距离 (Å)", "distance_to_cap_A"),
                       ("逆变换潜范数 ‖u*‖", "inverse_latent_norm_abs_u_star")):
        d = tb[key]
        md.append(f"| {label} | {d['median']:.4f} | {d['q25']:.4f} | "
                  f"{d['q75']:.4f} | {d['max']:.4f} |")
    md += ["", f"- 10% 近界占比 = {tb['fraction_within_10pct_of_cap']}；"
               f"1% 近界占比 = {tb['fraction_within_1pct_of_cap']}",
           f"- top-decile 教师动作 MSE 份额 = {teacher_share:.3f}；"
           f"模型动作 MSE 份额 = {model_share:.3f}",
           "", "## 参数梯度贡献（3 主臂，20k-update checkpoint）", ""]
    for name, g in grad_by_arm.items():
        md.append(f"- **{name}** (recipe {g['recipe']}): top-decile 梯度范数均值 "
                  f"{g['top_decile_mean_grad_norm']:.3e}，其余 "
                  f"{g['remaining_mean_grad_norm']:.3e}；每样本梯度比 "
                  f"{g['per_sample_gradient_ratio_top_over_remaining']}；"
                  f"按总体加权近界占比 {g['top_decile_share_population_weighted']}")
    Path(args.md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md).write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[item2] wrote {args.out} and {args.md}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover
        print(f"DIAGNOSIS REFUSAL {exc!r}", file=sys.stderr)
        raise SystemExit(2)
