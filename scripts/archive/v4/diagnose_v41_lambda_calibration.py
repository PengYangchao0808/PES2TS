#!/usr/bin/env python3
"""WP3 v4.1 item 8 — B-arm lambda_dir calibration from training-side scales.

Builds a FRESH seeded model under the NEW v41 5-channel spec (existing
builders: ``DualTSPolicy`` + ``DualTSModelConfig(light=True)`` against
``wp3_features_v41/features_spec.json``) and measures, over >=20 fit50
batches, the per-batch PARAMETER gradient norms of the two bc_d loss terms
exactly as implemented in ``dual_ts_bc.py``:

  * L_NLL = -log pi_z(u*|s)   (``per_atom_nll``, dual_ts_bc.py:371-374;
    teacher-latent NLL; its Jacobian log-det term is evaluated AT the fixed
    u*, so it contributes NO parameter gradient by itself).
  * L_dir = ||a*||·(1 - cos(a_det, a*))   (``per_atom_dir_loss``,
    dual_ts_bc.py:382-388; near-zero atoms masked out upstream).

Recommended lambda_dir = median over batches of (||dL_nll/dtheta|| /
||dL_dir/dtheta||), chosen so the direction-term gradient contribution
approximately matches the NLL term; sensitivity range [0.5x, 2x].

This feeds protocol slot arm_b_lambda (frozen by the orchestrator, NOT by
monitor8).  Zero new QC; training-side (fit) data only.

Two modes:
  * FRESH model (default, back-compat): builds a fresh seeded model and
    reports the spec-mandated fresh-model ratio (init-time dominated).
  * ``--checkpoint <path>``: loads a TRAINED BC checkpoint (v1
    ``checkpoint_last.pt`` or v2 ``checkpoint_v2.pt``) via the existing
    loader — matches the frozen r1 calibration procedure (load A's final
    short-2000 checkpoint, >=20 fit50 batches, seeded 20261017, median
    ratio), giving the post-training ratio the arm B slot should freeze.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from diagnose_v41_common import (
    ACTION_MAX, NEW_FIT_FEATURES, OBSERVED_FIT, OUT_DIR, device_of,
    grad_norm, load_new_states, recipe_loss_terms, write_json,
)

import torch  # noqa: E402

from pes2ts_core.pesn.models.dual_egnn import DualTSModelConfig  # noqa: E402
from pes2ts_core.pesn.models.dual_ts_policy import DualTSPolicy  # noqa: E402
from scripts.train_dual_ts_bc import load_policy_from_checkpoint  # noqa: E402

N_BATCHES = 24
BATCH_SIZE = 64
SEED = 20261017


def nll_and_dir_grad_norms(policy, states, device):
    """(||dL_nll/dtheta||, ||dL_dir/dtheta||) for one batch (one forward)."""
    nll_loss, _mse_loss, dir_loss = recipe_loss_terms(
        policy, states, ACTION_MAX, device)
    params = list(policy.parameters())
    gn_nll = grad_norm(nll_loss, policy, retain_graph=True)
    policy.module.zero_grad(set_to_none=True)
    gn_dir = grad_norm(dir_loss, policy)
    return gn_nll, gn_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DIR / "v41_lambda_calibration.json"))
    ap.add_argument("--md", default=str(OUT_DIR / "v41_lambda_calibration.md"))
    ap.add_argument("--n-batches", type=int, default=N_BATCHES)
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="load a TRAINED BC checkpoint (v1 checkpoint_last.pt "
                         "or v2 checkpoint_v2.pt, both read via the existing "
                         "loader) instead of a fresh model — matches the "
                         "frozen r1 calibration procedure (load A's final "
                         "short-2000 checkpoint, >=20 fit50 batches, seeded "
                         "20261017, median ratio)")
    args = ap.parse_args(argv)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    states, meta = load_new_states(NEW_FIT_FEATURES, OBSERVED_FIT)
    drawable = [s for s in states if not s["is_teacher"]]

    if args.checkpoint is not None:
        policy = load_policy_from_checkpoint(torch, args.checkpoint)
        policy.to(torch.float32)
        policy.to(device_of())
        policy.train_mode(True)
        model_source = {
            "mode": "trained_checkpoint",
            "checkpoint": str(args.checkpoint),
        }
    else:
        policy = DualTSPolicy(
            config=DualTSModelConfig(light=True,
                                     signed_vector_messages=False),
            action_max=ACTION_MAX, sigma_range=(0.02, 1.0),
            signed_mu_gate=False)
        policy.to(torch.float32)
        policy.to(device_of())
        policy.train_mode(True)
        model_source = {
            "mode": "fresh_model",
            "model": "DualTSPolicy(DualTSModelConfig(light=True)) under v41 "
                     "5-ch spec, seed 20261017",
        }
    device = device_of()

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(drawable))[: args.n_batches * BATCH_SIZE]
    nll_norms, dir_norms = [], []
    for i in range(0, len(idx), BATCH_SIZE):
        chunk = [drawable[int(j)] for j in idx[i:i + BATCH_SIZE]]
        gn_nll, gn_dir = nll_and_dir_grad_norms(policy, chunk, device)
        nll_norms.append(gn_nll)
        dir_norms.append(gn_dir)

    nll_a = np.asarray(nll_norms, dtype=np.float64)
    dir_a = np.asarray(dir_norms, dtype=np.float64)
    ratios = nll_a / np.maximum(dir_a, 1e-30)
    lambda_dir = float(np.median(ratios))

    result = {
        "schema_version": "pesn_v41_diag_lambda_calibration_v1",
        "title": "WP3 v4.1 item 8: lambda_dir calibration (training-side)",
        "features_spec_digest": meta["spec_digest"],
        "loss_citations": {
            "L_NLL": "per_atom_nll (dual_ts_bc.py:371-374), teacher-latent NLL; "
                     "Jacobian log-det evaluated at fixed u* -> no d/dtheta",
            "L_dir": "per_atom_dir_loss (dual_ts_bc.py:382-388), "
                     "||a*||(1-cos(a_det,a*)), near-zero masked upstream",
        },
        "model_source": model_source,
        "method": {
            "n_batches": int(len(nll_a)),
            "batch_states": BATCH_SIZE,
            "seed": SEED,
            "metric": "per-batch ||dL_nll/dtheta|| and ||dL_dir/dtheta||; "
                      "lambda_dir = median(||dL_nll||/||dL_dir||)",
        },
        "grad_norms": {
            "nll_median": float(np.median(nll_a)),
            "nll_q25": float(np.percentile(nll_a, 25)),
            "nll_q75": float(np.percentile(nll_a, 75)),
            "dir_median": float(np.median(dir_a)),
            "dir_q25": float(np.percentile(dir_a, 25)),
            "dir_q75": float(np.percentile(dir_a, 75)),
        },
        "ratio_stats": {
            "median": lambda_dir,
            "q25": float(np.percentile(ratios, 25)),
            "q75": float(np.percentile(ratios, 75)),
        },
        "recommended_lambda_dir": lambda_dir,
        "sensitivity_range": [0.5 * lambda_dir, 2.0 * lambda_dir],
        "回答什么": [
            "训练侧（fit 数据）梯度尺度诊断给出的 lambda_dir 校准建议值",
            "目标：方向项梯度贡献 ≈ NLL 项梯度贡献",
        ],
        "不能声称什么": [
            "不使用 monitor8 调权；该值仅供 orchestrator 冻结 prereg 槽位 arm_b_lambda",
            "梯度尺度相等不保证方向门通过；不替代 WP4 训练验证",
        ],
        "note": "fresh-init mode reports the spec-mandated fresh-model ratio "
                "(dominated by init-time NLL Gaussian residual); --checkpoint "
                "mode reports the post-training ratio matching the frozen r1 "
                "calibration procedure.",
    }
    write_json(Path(args.out), result)

    md = ["# WP3 v4.1 条目8：lambda_dir 校准（训练侧梯度尺度）", "",
          "| 回答什么 | 不能声称什么 |",
          "|---|---|",
          "| 训练侧梯度尺度诊断 → lambda_dir 建议值 | 不用 monitor8 调权 |",
          "| 目标：方向项梯度≈NLL项梯度 | 梯度相等≠方向门通过 |",
          "", f"- features spec digest: {meta['spec_digest']}",
          f"- median ‖∂L_nll/∂θ‖ = {result['grad_norms']['nll_median']:.3e}；"
          f"median ‖∂L_dir/∂θ‖ = {result['grad_norms']['dir_median']:.3e}",
          f"- **recommended λ_dir = {lambda_dir:.4f}** "
          f"(sensitivity [{0.5*lambda_dir:.4f}, {2.0*lambda_dir:.4f}])",
          "",
          f"- ⚠ 该值为全新初始化下测得，被初始化 NLL 残差主导（median ‖∂L_nll‖="
          f"{result['grad_norms']['nll_median']:.2e} vs ‖∂L_dir‖="
          f"{result['grad_norms']['dir_median']:.2e}）；是方向项相对 NLL 早期梯度的上界"]
    Path(args.md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md).write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[item8] recommended lambda_dir = {lambda_dir:.4f} "
          f"(median ||dL_nll||/||dL_dir|| over {len(nll_a)} batches)")
    print(f"[item8] wrote {args.out} and {args.md}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover
        print(f"DIAGNOSIS REFUSAL {exc!r}", file=sys.stderr)
        raise SystemExit(2)
