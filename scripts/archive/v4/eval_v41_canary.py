#!/usr/bin/env python3
"""WP4a staging — v4.1 canary (fit1/fit5) IN-SAMPLE fitability evaluation.

Zero new QC.  Reads the v41 feature shards + each canary run's checkpoint and
evaluates each canary ON ITS OWN TRAINING REACTIONS (in-sample fitability, NOT
generalization).  This is the F7 "覆盖普通动作和近 TS 动作" canary check:

  per reaction group, on its training states:
    (a) MSE ratio vs zero-action baseline (same reduction as the monitor:
        mean ||a_det-a*||^2 / mean ||a*||^2 over valid action atoms);
    (b) |a|-weighted direction cosine over dir_mask atoms (monitor definition);
    (c) near-TS stratum cosine (ral <= 0.2 * branch max) vs rest — covers
        ordinary AND near-TS actions;
    (d) same-sample magnitude ratio alpha = ||a_det|| / ||a*|| median + IQR
        (WP3 joint-stats method: per-atom over dir_mask atoms only);
    (e) per-reaction effective-action fraction (# dir_mask atoms / # valid).

Nothing here is consumed by selection (§I.9: monitor8-only).  Deterministic,
read-only.  Outputs canary_v41.{json,md} under wp41_diagnostics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from diagnose_v41_common import (
    ACTION_MAX, OUT_DIR, aligned, atom_alpha, device_of, dir_cosine_weighted,
    forward_det, med_iqr, mse_ratio_vs_zero, set_new_layout, write_json,
)
from pes2ts_core.pesn.offline.training.dual_ts_bc import (
    load_dual_ts_states,
)

NEAR_TS_ARC_FRACTION = 0.2


def near_ts_mask(al: dict) -> np.ndarray:
    """Frozen per-branch near-TS stratum (ral <= 0.2 * branch max)."""
    nt = np.zeros(al["ral"].shape, dtype=bool)
    for br in np.unique(al["branch"]):
        m = al["branch"] == br
        ral = al["ral"][m]
        nt[m] = ral <= NEAR_TS_ARC_FRACTION * float(ral.max())
    return nt


def evaluate_run(torch, run_dir: Path) -> dict:
    rec = json.loads((run_dir / "run_record.json").read_text())
    settings = rec["settings"]
    data = rec["data"]
    group_ids = [str(g) for g in data["group_ids"]]
    features_dir = Path(data["features_dir"])
    observed_dir = Path(data["observed_dir"])

    set_new_layout()
    states, meta = load_dual_ts_states(
        features_dir, observed_dir, group_filter=group_ids)

    # import here to avoid a module-level scripts-package dependency cycle
    from scripts.train_dual_ts_bc import load_policy_from_checkpoint
    policy = load_policy_from_checkpoint(torch, run_dir / "checkpoint_v2.pt")
    policy.to(torch.float32)
    policy.to(device_of())
    policy.train_mode(False)

    al = aligned(states, ACTION_MAX)
    a_det, _ = forward_det(policy, states, action_max=ACTION_MAX)
    nt = near_ts_mask(al)
    alpha = atom_alpha(a_det, al["a_star"])

    rows = {}
    for gid in np.unique(al["group"]):
        valid = al["valid"] & (al["group"] == gid)
        dmask = al["dir_mask"] & (al["group"] == gid)
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue
        dmask_nt = dmask & nt
        dmask_rest = dmask & (~nt)
        rows[str(gid)] = {
            "group_id": str(gid),
            "mse_ratio_vs_zero": mse_ratio_vs_zero(a_det, al["a_star"], valid),
            "direction_cosine_absa_weighted": dir_cosine_weighted(
                a_det, al["a_star"], dmask),
            "near_ts_cosine": dir_cosine_weighted(a_det, al["a_star"], dmask_nt),
            "rest_cosine": dir_cosine_weighted(a_det, al["a_star"], dmask_rest),
            "alpha": med_iqr(alpha[dmask]),
            "effective_action_fraction": float(dmask.sum()) / float(n_valid),
            "near_ts_fraction": float((valid & nt).sum()) / float(n_valid),
            "n_valid_atoms": n_valid,
            "n_dir_atoms": int(dmask.sum()),
            "n_near_ts_dir_atoms": int(dmask_nt.sum()),
            "n_rest_dir_atoms": int(dmask_rest.sum()),
        }

    fit = rec.get("fit_diagnostics_final", {})
    return {
        "run_dir": run_dir.name,
        "stage": settings["stage"],
        "recipe": settings["recipe"],
        "seed": settings["seed"],
        "budget_updates": settings["budget_updates"],
        "protocol_sha256": settings["protocol_sha256"],
        "spec_digest": data["spec_digest"],
        "group_ids": group_ids,
        "n_states": meta["n_states"],
        "n_groups": meta["n_groups"],
        "checkpoint": "checkpoint_v2.pt",
        "fit_side_recorded": {
            "mse_over_baseline": fit.get("mse_over_baseline"),
            "direction_cosine_absa_weighted":
                fit.get("direction_cosine_absa_weighted"),
            "nll_per_atom": fit.get("nll_per_atom"),
        },
        "per_reaction": rows,
        "framing": "IN-SAMPLE fitability on the run's OWN training reactions; "
                   "NOT generalization, NOT a monitor/selection metric",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dirs", nargs="+", required=True,
                    help="one or more canary run directories")
    ap.add_argument("--out", default=str(OUT_DIR / "canary_v41.json"))
    ap.add_argument("--md", default=str(OUT_DIR / "canary_v41.md"))
    args = ap.parse_args(argv)

    import torch  # noqa: E402
    runs = [evaluate_run(torch, Path(d)) for d in args.run_dirs]

    result = {
        "schema_version": "pesn_v41_diag_canary_fitability_v1",
        "title": "WP4a staging: fit1/fit5 canary IN-SAMPLE fitability",
        "回答什么": [
            "各 canary 在其自身训练反应上的逐反应 MSE 比值、方向余弦、"
            "近 TS vs 普通动作余弦、同一样本幅度比 α 中位数+IQR、有效动作比例",
            "覆盖普通动作与近 TS 动作（F7 可拟合性 canary 检查）",
        ],
        "不能声称什么": [
            "这是 in-sample（训练分区）可拟合性，不是泛化，不是 monitor 门，"
            "不用于选型（§I.9 monitor8-only）",
        ],
        "runs": runs,
    }
    write_json(Path(args.out), result)

    lines = [
        "# WP4a staging：fit1/fit5 canary 可拟合性（in-sample）",
        "",
        "| 回答什么 | 不能声称什么 |",
        "|---|---|",
        "| 逐反应 MSE比值、余弦、近TS vs 普通、α、有效动作比 | in-sample，"
        "非泛化，非 monitor 门，不选型 |",
        "| 覆盖普通动作与近 TS 动作 | — |",
        "",
    ]
    for run in runs:
        lines += [
            f"## {run['run_dir']}",
            f"- stage={run['stage']} recipe={run['recipe']} "
            f"seed={run['seed']} budget={run['budget_updates']} "
            f"groups={run['n_groups']} states={run['n_states']}",
            f"- protocol_sha256={run['protocol_sha256'][:16]}…",
            f"- fit-side recorded: MSE/baseline="
            f"{run['fit_side_recorded']['mse_over_baseline']:.3f} "
            f"cos={run['fit_side_recorded']['direction_cosine_absa_weighted']} "
            f"NLL/atom={run['fit_side_recorded']['nll_per_atom']:.3f}",
            "",
            "| group | MSE比值 | cos(|a|加权) | 近TS cos | 普通 cos | "
            "α中位[IQR] | 有效动作比 | 近TS占比 | n_dir |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for gid in sorted(run["per_reaction"]):
            r = run["per_reaction"][gid]
            a = r["alpha"]
            lines.append(
                f"| {gid} | {r['mse_ratio_vs_zero']:.4f} | "
                f"{r['direction_cosine_absa_weighted']:.4f} | "
                f"{r['near_ts_cosine']:.4f} | {r['rest_cosine']:.4f} | "
                f"{a['median']:.3f} "
                f"[{a['q25']:.3f},{a['q75']:.3f}] | "
                f"{r['effective_action_fraction']:.3f} | "
                f"{r['near_ts_fraction']:.3f} | {r['n_dir_atoms']} |")
        lines.append("")
    Path(args.md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[canary-eval] wrote {args.out} and {args.md}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover
        print(f"CANARY EVAL REFUSAL {exc!r}", file=sys.stderr)
        raise SystemExit(2)
