#!/usr/bin/env python3
"""WP3 v4.1 item 6 — old-model label-channel sensitivity (ch5/ch6).

The OLD 7-channel node-scalar layout is (frozen v2 spec):
  [z_id, grad_norm_scaled, delta_e_scaled, charge, multiplicity,
   is_teacher_endpoint (ch5), near_zero_action_mask (ch6)].

We hold the OLD bc_nll checkpoint fixed and intervene ONLY on ch5/ch6:

  zero                      -> set channel to 0
  one                       -> set channel to 1
  flip                      -> 1 - channel
  shuffle-within-batch      -> permute the channel values across the atoms of
                               each forward batch (breaks per-atom alignment)
  arriving-action-rule      -> online-consistent rule: ch6 = ||arriving||<0.01
                               (the action that GENERATED this state; zero at
                               episode start => all-True), ch5 = 0 (no future
                               endpoint label available online).

and measure per-reaction ΔMSE (ratio vs zero), Δcos (|a|-weighted), Δ||a_det||
on fit50 + monitor8.

FRAMING (binding): sensitivity quantifies RELIANCE on the channel.  It does
NOT prove the channel caused the v4 failure, and does NOT predict post-removal
performance (that requires retraining — WP4).  Zero new QC.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from diagnose_v41_common import (
    ACTION_MAX, ARMS, OBSERVED_FIT, OBSERVED_MON, OLD_FIT_FEATURES,
    OLD_MON_FEATURES, OUT_DIR, aligned, atom_mse, device_of,
    dir_cosine_weighted, forward_det, load_old_policy, load_old_states,
    mse_ratio_vs_zero, write_json,
)

from pes2ts_core.pesn.models.dual_egnn import collate_dual_states  # noqa: E402

CH5, CH6 = 5, 6
PARTITIONS = {"fit50": (OLD_FIT_FEATURES, OBSERVED_FIT),
              "monitor8": (OLD_MON_FEATURES, OBSERVED_MON)}


def arriving_near_zero(states):
    """For each state: the near-zero mask of the ARRIVING action.

    arriving action = action_delta of the previous state in the same branch
    (the displacement that produced the current geometry); zero at step 0.
    Returns a dict state_id -> (n_atoms,) bool mask.
    """
    by_branch: dict[str, list[dict]] = {}
    for s in states:
        by_branch.setdefault(s["branch_id"], []).append(s)
    out: dict[str, np.ndarray] = {}
    for branch, sts in by_branch.items():
        sts = sorted(sts, key=lambda s: int(s["step_index"]))
        prev_action = None
        for s in sts:
            n = int(np.asarray(s["atomic_numbers"]).size)
            if prev_action is None:
                mask = np.ones(n, dtype=bool)  # episode start: zero arriving
            else:
                mask = np.linalg.norm(prev_action, axis=1) < 0.01
            out[str(s["state_id"])] = mask
            prev_action = (None if s["action"] is None
                           else np.asarray(s["action"], dtype=np.float64))
    return out


def build_interventions(states):
    """Per-state modified node_scalars for each intervention (except shuffle)."""
    arr_nz = arriving_near_zero(states)
    base = {str(s["state_id"]): np.asarray(s["node_scalars"], dtype=np.float64).copy()
            for s in states}
    out = {}
    for ch, chname in ((CH5, "ch5_is_teacher_endpoint"), (CH6, "ch6_near_zero")):
        for mode in ("zero", "one", "flip"):
            key = f"{chname}_{mode}"
            out[key] = {}
            for s in states:
                a = base[str(s["state_id"])].copy()
                if mode == "zero":
                    a[:, ch] = 0.0
                elif mode == "one":
                    a[:, ch] = 1.0
                else:
                    a[:, ch] = 1.0 - a[:, ch]
                out[key][str(s["state_id"])] = a
    # arriving-action rule: ch5 -> 0, ch6 -> arriving mask
    out["arriving_rule"] = {}
    for s in states:
        a = base[str(s["state_id"])].copy()
        a[:, CH5] = 0.0
        a[:, CH6] = arr_nz[str(s["state_id"])].astype(np.float64)
        out["arriving_rule"][str(s["state_id"])] = a
    return out


def _modified_states(states, ns_map):
    out = []
    for s in states:
        c = dict(s)
        c["node_scalars"] = ns_map[str(s["state_id"])]
        out.append(c)
    return out


def forward_shuffle(policy, states, ch, batch_size=256):
    """Forward with per-batch channel shuffle (breaks atom alignment)."""
    torch_mod = policy.torch
    device = policy.module.encoder.node_proj.weight.device
    rng = np.random.default_rng(20261017)
    mus = []
    with torch_mod.no_grad():
        for i in range(0, len(states), batch_size):
            chunk = list(states[i:i + batch_size])
            batch = collate_dual_states(torch_mod, chunk)
            for k, v in list(batch.items()):
                if torch_mod.is_tensor(v) and v.dtype in (torch.float64, torch.float32):
                    batch[k] = v.to(dtype=torch.float32)
                if torch_mod.is_tensor(v):
                    batch[k] = v.to(device)
            ns = batch["node_scalars"].clone()
            perm = torch_mod.from_numpy(
                rng.permutation(ns.shape[0])).to(ns.device)
            ns[:, ch] = ns[perm, ch]
            batch["node_scalars"] = ns
            out = policy(batch)
            mus.append(policy.deterministic(out["mu"]).detach().cpu().numpy())
    return np.concatenate(mus, axis=0) if mus else np.zeros((0, 3))


def per_reaction_metrics(al, a_det):
    out = {}
    for gid in np.unique(al["group"]):
        valid = al["valid"] & (al["group"] == gid)
        dmask = al["dir_mask"] & (al["group"] == gid)
        out[str(gid)] = {
            "mse_ratio": mse_ratio_vs_zero(a_det, al["a_star"], valid),
            "cos": dir_cosine_weighted(a_det, al["a_star"], dmask),
            "med_norm": float(np.median(np.linalg.norm(a_det, axis=1)[valid]))
            if valid.any() else None,
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="bc_nll")
    ap.add_argument("--out", default=str(OUT_DIR / "v41_label_channel_sensitivity.json"))
    ap.add_argument("--md", default=str(OUT_DIR / "v41_label_channel_sensitivity.md"))
    args = ap.parse_args(argv)

    arm = next(a for a in ARMS if a["name"] == args.arm)
    policy, _ = load_old_policy(arm)

    result = {
        "schema_version": "pesn_v41_diag_label_channel_sensitivity_v1",
        "title": "WP3 v4.1 item 6: old-model label-channel sensitivity (ch5/ch6)",
        "arm": arm["name"],
        "channel_layout": {
            "ch5": "is_teacher_endpoint", "ch6": "near_zero_action_mask",
            "full_7ch": ["z_id", "grad_norm_scaled", "delta_e_scaled",
                         "charge", "multiplicity", "is_teacher_endpoint",
                         "near_zero_action_mask"],
        },
        "回答什么": [
            "固定 bc_nll checkpoint，仅扰动 ch5/ch6 时输出 (ΔMSE, Δcos, Δ‖â‖) 的逐反应变化",
            "模型对教师标签衍生通道的依赖（敏感性）量级",
        ],
        "不能声称什么": [
            "敏感性量化依赖，不证明通道导致了 v4 失败",
            "不预测移除通道后的表现（需重训，WP4）",
        ],
        "partitions": {},
    }

    for part_name, (feat_dir, obs_dir) in PARTITIONS.items():
        states, _meta = load_old_states(feat_dir, obs_dir)
        al = aligned(states, ACTION_MAX)
        base_det, _ = forward_det(policy, states)
        base_metrics = per_reaction_metrics(al, base_det)
        intervs = build_interventions(states)
        part_block = {"baseline": base_metrics, "interventions": {}}
        for key, ns_map in intervs.items():
            if key.endswith("_shuffle"):
                ch = CH5 if key.startswith("ch5") else CH6
                a_det = forward_shuffle(policy, states, ch)
            else:
                a_det, _ = forward_det(policy, _modified_states(states, ns_map))
            m = per_reaction_metrics(al, a_det)
            # per-reaction deltas
            deltas = []
            for gid in base_metrics:
                b = base_metrics[gid]
                iv = m[gid]
                deltas.append({
                    "group_id": gid,
                    "delta_mse_ratio": (iv["mse_ratio"] - b["mse_ratio"])
                    if iv["mse_ratio"] is not None and b["mse_ratio"] is not None else None,
                    "delta_cos": (iv["cos"] - b["cos"])
                    if iv["cos"] is not None and b["cos"] is not None else None,
                    "delta_med_norm": (iv["med_norm"] - b["med_norm"])
                    if iv["med_norm"] is not None and b["med_norm"] is not None else None,
                })
            part_block["interventions"][key] = {
                "per_reaction_delta": deltas,
                "overall_median_delta": {
                    "mse_ratio": float(np.median([
                        d["delta_mse_ratio"] for d in deltas
                        if d["delta_mse_ratio"] is not None])),
                    "cos": float(np.median([
                        d["delta_cos"] for d in deltas
                        if d["delta_cos"] is not None])),
                    "med_norm": float(np.median([
                        d["delta_med_norm"] for d in deltas
                        if d["delta_med_norm"] is not None])),
                },
            }
        result["partitions"][part_name] = part_block

    write_json(Path(args.out), result)

    md = ["# WP3 v4.1 条目6：旧模型标签通道敏感性（ch5/ch6）", "",
          "| 回答什么 | 不能声称什么 |",
          "|---|---|",
          "| 扰动 ch5/ch6 后 ΔMSE/Δcos/Δ‖â‖（逐反应+总体中位） | 敏感性≠因果根因 |",
          "| 模型对标签衍生通道的依赖量级 | 不预测移除后表现（需重训） |", ""]
    for part_name in PARTITIONS:
        pb = result["partitions"][part_name]
        md += [f"## {part_name}", "",
               "| 干预 | ΔMSE比值(中位) | Δcos(中位) | Δ‖â‖(中位) |",
               "|---|---|---|---|"]
        for key, iv in pb["interventions"].items():
            od = iv["overall_median_delta"]
            md.append(f"| {key} | {od['mse_ratio']:+.4f} | {od['cos']:+.4f} | "
                      f"{od['med_norm']:+.5f} |")
        md.append("")
    Path(args.md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md).write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[item6] wrote {args.out} and {args.md}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover
        print(f"DIAGNOSIS REFUSAL {exc!r}", file=sys.stderr)
        raise SystemExit(2)
