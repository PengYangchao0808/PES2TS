#!/usr/bin/env python3
"""WP4a staging — fit50 short-run health checks (F7 step 2).

Zero new QC.  Reads the short run's ``run_record.json`` and its final
checkpoint, and reports:

  (a) loss-curve summary: initial/final NLL + MSE, monotonicity, NaN/inf scan;
  (b) gradient health: ONE backward pass on 3 seeded batches AT THE END state —
      median ||dL/dtheta|| per loss term (NLL / MSE / direction), with
      explosion (>1e4) and vanishing (<1e-8) flags;
  (c) exposure/sampling coverage from the run record: per-group exposure
      uniformity (coverage / repeat rate / near-TS fraction spread), distinct-
      state coverage, repeat rate, near-TS fraction vs the population near-TS
      fraction;
  (d) training-curve epoch progression (curve entries + exposure snapshots).

Read-only; writes short_run_health.{json,md} under wp41_diagnostics.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from diagnose_v41_common import (
    ACTION_MAX, OUT_DIR, device_of, grad_norm, load_new_states,
    recipe_loss_terms, set_new_layout, write_json,
)
from pes2ts_core.pesn.offline.training.dual_ts_bc import (
    NEAR_TS_ARC_FRACTION, load_dual_ts_states,
)

GRAD_BATCHES = 3
GRAD_BATCH_STATES = 64
GRAD_SEED = 20261017


def _pct(a, p):
    a = np.asarray(a, dtype=np.float64)
    return float(np.percentile(a, p))


def loss_curve_summary(rec: dict) -> dict:
    h = rec.get("history", [])
    if not h:
        return {"n_entries": 0}
    nll = [e["nll_loss"] for e in h]
    mse = [e["mse_loss"] for e in h]
    dirs = [e["dir_loss"] for e in h]
    finite_ok = all(
        math.isfinite(float(v))
        for e in h for v in (e["nll_loss"], e["mse_loss"], e["dir_loss"]))
    monotonic = all(nll[i] >= nll[i + 1] for i in range(len(nll) - 1))
    return {
        "n_entries": len(h),
        "nll_initial": float(h[0]["nll_loss"]),
        "nll_final": float(h[-1]["nll_loss"]),
        "mse_initial": float(h[0]["mse_loss"]),
        "mse_final": float(h[-1]["mse_loss"]),
        "dir_initial": float(h[0]["dir_loss"]),
        "dir_final": float(h[-1]["dir_loss"]),
        "nll_per_atom_initial": float(h[0]["nll_per_atom_running"]),
        "nll_per_atom_final": float(h[-1]["nll_per_atom_running"]),
        "nll_monotonic_decreasing": monotonic,
        "mse_improved": float(h[-1]["mse_loss"]) < float(h[0]["mse_loss"]),
        "all_finite": finite_ok,
        "any_nan_inf": not finite_ok,
    }


def exposure_summary(rec: dict, population_near_ts_fraction: float) -> dict:
    ex = rec.get("exposure", {})
    pg = ex.get("per_group", {})
    covs = [v["coverage"] for v in pg.values()]
    reps = [v["repeat_rate"] for v in pg.values()]
    ntf = [v["near_ts_fraction"] for v in pg.values()]
    return {
        "drawable_states": ex.get("drawable_states"),
        "distinct_states": ex.get("distinct_states"),
        "coverage": ex.get("coverage"),
        "repeat_rate": ex.get("repeat_rate"),
        "near_ts_fraction_draws": ex.get("near_ts_fraction"),
        "population_near_ts_fraction": population_near_ts_fraction,
        "n_groups": ex.get("n_groups"),
        "per_group_coverage": {
            "min": min(covs), "median": _pct(covs, 50), "max": max(covs),
        },
        "per_group_repeat_rate": {
            "min": min(reps), "median": _pct(reps, 50), "max": max(reps),
        },
        "per_group_near_ts_fraction": {
            "min": min(ntf), "median": _pct(ntf, 50), "max": max(ntf),
        },
    }


def epoch_progression(rec: dict) -> dict:
    h = rec.get("history", [])
    exp_hist = (rec.get("exposure", {}) or {}).get("history", [])
    curve = [(e["update"], e["epoch"]) for e in h]
    snaps = [(e["update"], e["epoch"], e["kind"]) for e in exp_hist]
    return {
        "curve_entries": curve,
        "final_epoch": h[-1]["epoch"] if h else None,
        "n_exposure_snapshots": len(snaps),
        "exposure_snapshot_kinds": [s[2] for s in snaps],
    }


def gradient_health(torch, run_dir: Path, states, drawable) -> dict:
    set_new_layout()
    from scripts.train_dual_ts_bc import load_policy_from_checkpoint
    policy = load_policy_from_checkpoint(torch, run_dir / "checkpoint_v2.pt")
    policy.to(torch.float32)
    policy.to(device_of())
    policy.train_mode(True)
    device = device_of()

    rng = np.random.default_rng(GRAD_SEED)
    idx = rng.permutation(len(drawable))[: GRAD_BATCHES * GRAD_BATCH_STATES]
    nll_norms, mse_norms, dir_norms = [], [], []
    for i in range(0, len(idx), GRAD_BATCH_STATES):
        chunk = [drawable[int(j)] for j in idx[i:i + GRAD_BATCH_STATES]]
        nll_loss, mse_loss, dir_loss = recipe_loss_terms(
            policy, chunk, ACTION_MAX, device)
        gn_nll = grad_norm(nll_loss, policy, retain_graph=True)
        policy.module.zero_grad(set_to_none=True)
        gn_mse = grad_norm(mse_loss, policy, retain_graph=True)
        policy.module.zero_grad(set_to_none=True)
        gn_dir = grad_norm(dir_loss, policy)
        policy.module.zero_grad(set_to_none=True)
        nll_norms.append(gn_nll)
        mse_norms.append(gn_mse)
        dir_norms.append(gn_dir)

    def summarize(arr):
        a = np.asarray(arr, dtype=np.float64)
        return {
            "median": float(np.median(a)),
            "min": float(a.min()), "max": float(a.max()),
            "explosion_gt_1e4": bool(float(np.median(a)) > 1e4),
            "vanishing_lt_1e-8": bool(float(np.median(a)) < 1e-8),
        }

    return {
        "n_batches": len(nll_norms),
        "batch_states": GRAD_BATCH_STATES,
        "seed": GRAD_SEED,
        "nll_term_grad_norm": summarize(nll_norms),
        "mse_term_grad_norm": summarize(mse_norms),
        "dir_term_grad_norm": summarize(dir_norms),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True,
                    help="fit50 short run directory")
    ap.add_argument("--out", default=str(OUT_DIR / "short_run_health.json"))
    ap.add_argument("--md", default=str(OUT_DIR / "short_run_health.md"))
    args = ap.parse_args(argv)

    import torch  # noqa: E402
    run_dir = Path(args.run_dir)
    rec = json.loads((run_dir / "run_record.json").read_text())
    data = rec["data"]

    # population near-TS fraction computed over the SAME fit50 drawable states
    set_new_layout()
    states, meta = load_dual_ts_states(
        Path(data["features_dir"]), Path(data["observed_dir"]))
    drawable = [s for s in states if not s["is_teacher"]]
    branch_max: dict[tuple, float] = {}
    for s in drawable:
        key = (s["reaction_group_id"], s["branch_id"])
        branch_max[key] = max(branch_max.get(key, 0.0),
                              float(s["remaining_arc_length"]))
    n_near = sum(
        1 for s in drawable
        if float(s["remaining_arc_length"])
        <= NEAR_TS_ARC_FRACTION
        * branch_max[(s["reaction_group_id"], s["branch_id"])])
    pop_near_ts_frac = float(n_near) / float(len(drawable))

    curve = loss_curve_summary(rec)
    exposure = exposure_summary(rec, pop_near_ts_frac)
    epoch = epoch_progression(rec)
    grad = gradient_health(torch, run_dir, states, drawable)

    result = {
        "schema_version": "pesn_v41_diag_short_health_v1",
        "title": "WP4a staging: fit50 short-run health (F7 step 2)",
        "run_dir": run_dir.name,
        "settings": {
            "recipe": rec["settings"]["recipe"],
            "stage": rec["settings"]["stage"],
            "seed": rec["settings"]["seed"],
            "budget_updates": rec["settings"]["budget_updates"],
            "lr": rec["settings"]["lr"],
            "batch_states": rec["settings"]["batch_states"],
            "diagnostic_run": rec["settings"]["diagnostic_run"],
            "protocol_sha256": rec["settings"]["protocol_sha256"],
        },
        "accounting": rec["accounting"],
        "fit_diagnostics_final": rec.get("fit_diagnostics_final"),
        "loss_curve": curve,
        "gradient_health": grad,
        "exposure": exposure,
        "epoch_progression": epoch,
        "回答什么": [
            "数值/梯度/采样覆盖/训练曲线四类健康检查（不构成选型）",
        ],
        "不能声称什么": [
            "健康检查通过不保证方向门通过；不用于 monitor8 选型",
        ],
    }
    write_json(Path(args.out), result)

    g = grad
    md = [
        "# WP4a staging：fit50 短程健康检查（F7 step 2）",
        "",
        "| 回答什么 | 不能声称什么 |",
        "|---|---|",
        "| 数值/梯度/采样覆盖/训练曲线 | 健康≠方向门通过，不选型 |",
        "",
        f"## {run_dir.name}",
        f"- stage={rec['settings']['stage']} recipe={rec['settings']['recipe']} "
        f"seed={rec['settings']['seed']} budget={rec['settings']['budget_updates']} "
        f"lr={rec['settings']['lr']} batch={rec['settings']['batch_states']}",
        f"- protocol_sha256={rec['settings']['protocol_sha256'][:16]}… "
        f"diagnostic={rec['settings']['diagnostic_run']}",
        "",
        "### (a) 损失曲线",
        f"- NLL: {curve['nll_initial']:.4f} → {curve['nll_final']:.4f} "
        f"(单调递减={curve['nll_monotonic_decreasing']})",
        f"- MSE: {curve['mse_initial']:.6f} → {curve['mse_final']:.6f} "
        f"(改善={curve['mse_improved']})",
        f"- NLL/atom: {curve['nll_per_atom_initial']:.4f} → "
        f"{curve['nll_per_atom_final']:.4f}",
        f"- 全部有限={curve['all_finite']}；NaN/inf={curve['any_nan_inf']}",
        "",
        "### (b) 梯度健康（末态 3 批，中位 ‖∂L/∂θ‖）",
        f"- ‖∂L_nll/∂θ‖ median={g['nll_term_grad_norm']['median']:.4e} "
        f"[{g['nll_term_grad_norm']['min']:.3e},{g['nll_term_grad_norm']['max']:.3e}] "
        f"爆炸(>1e4)={g['nll_term_grad_norm']['explosion_gt_1e4']} "
        f"消失(<1e-8)={g['nll_term_grad_norm']['vanishing_lt_1e-8']}",
        f"- ‖∂L_mse/∂θ‖ median={g['mse_term_grad_norm']['median']:.4e} "
        f"爆炸={g['mse_term_grad_norm']['explosion_gt_1e4']} "
        f"消失={g['mse_term_grad_norm']['vanishing_lt_1e-8']}",
        f"- ‖∂L_dir/∂θ‖ median={g['dir_term_grad_norm']['median']:.4e} "
        f"爆炸={g['dir_term_grad_norm']['explosion_gt_1e4']} "
        f"消失={g['dir_term_grad_norm']['vanishing_lt_1e-8']}",
        "",
        "### (c) 采样覆盖",
        f"- distinct 覆盖={exposure['coverage']}（{exposure['distinct_states']}/"
        f"{exposure['drawable_states']}），repeat={exposure['repeat_rate']:.4f}",
        f"- 逐组覆盖 uniform: min={exposure['per_group_coverage']['min']:.3f} "
        f"median={exposure['per_group_coverage']['median']:.3f} "
        f"max={exposure['per_group_coverage']['max']:.3f}",
        f"- 逐组 repeat: min={exposure['per_group_repeat_rate']['min']:.3f} "
        f"median={exposure['per_group_repeat_rate']['median']:.3f} "
        f"max={exposure['per_group_repeat_rate']['max']:.3f}",
        f"- near-TS 抽取占比={exposure['near_ts_fraction_draws']:.4f} vs "
        f"总体 near-TS 占比={exposure['population_near_ts_fraction']:.4f}",
        "",
        "### (d) 训练曲线 epoch 推进",
        f"- 末 epoch={epoch['final_epoch']}，曲线条目={len(epoch['curve_entries'])}，"
        f"曝光快照={epoch['n_exposure_snapshots']}",
        f"- curve (update, epoch): {epoch['curve_entries']}",
        "",
    ]
    Path(args.md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md).write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[short-health] wrote {args.out} and {args.md}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover
        print(f"SHORT HEALTH REFUSAL {exc!r}", file=sys.stderr)
        raise SystemExit(2)
