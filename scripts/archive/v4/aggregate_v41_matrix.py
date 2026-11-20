#!/usr/bin/env python3
"""WP4b v4.1 experiment-matrix aggregation (ZERO QC — mechanical, reads run dirs).

Reads the A/B/C/D full-stage run dirs under the ``wp41_bc`` namespace and the
frozen monitor8 ``monitor_eval.json`` each run produced, then emits:

* ``matrix_summary.{json,md}``     — per-arm × per-seed gate table (the four
  frozen monitor-gate metrics), arm-level MEDIAN across seeds, per-seed gate
  status (all four criteria met = PASS), arm status (>=2/3 seeds PASS), and the
  D-rule comparison table (B/C effective vs A on 3-seed medians).
* ``matrix_monitor_per_group.{json,md}`` — per-reaction monitor8 tables for
  every arm (per-seed + arm-level median across seeds), full 8-group denominator.

The four frozen gate metrics (from ``monitor_thresholds_v1.json`` gate, bound
by sha256 in the protocol) are read directly from each ``monitor_eval.json``
``summary``:
  1. median per-group MSE ratio vs zero-action      -> summary.mse_ratio_median
  2. #groups with ratio < 1.0 (of 8)                -> summary.n_groups_ratio_below_one
  3. median |a|-weighted cosine                     -> summary.direction_cosine_median
  4. median near-TS cosine                          -> summary.strata.near_ts.direction_cosine_median

No QC is performed and no source files are touched.  This script only reads
run artifacts and writes aggregation outputs.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Frozen r2 protocol freeze sha (the operative formal protocol).
R2_SHA = "26a3f789ddbe53752f666d824c6601cd30c4e72836149008e30cc20cc8348a33"

# Frozen monitor8 gate thresholds (monitor_thresholds_v1.json).
GATE = {
    "median_mse_ratio_max": 1.0,
    "n_groups_ratio_below_one_min": 6,
    "median_cosine_min": 0.3,
    "near_ts_cosine_min": 0.4,
}

# Arm -> (recipe, lambda_dir, signed_vector_messages) tuple.
ARM_DEFS = {
    "A": ("bc_nll", 1.0, False),
    "B": ("bc_d", 559.37, False),
    "C": ("bc_nll", 1.0, True),
    "D": ("bc_d", 559.37, True),
}

ARM_LABEL = {
    "A": "A (clean baseline)",
    "B": "B (scale-calibrated direction loss, lambda_dir=559.37)",
    "C": "C (signed vector messages)",
    "D": "D (B + C)",
}


def _median(values) -> float | None:
    vals = [float(v) for v in values
            if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    vals.sort()
    n = len(vals)
    mid = n // 2
    return vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2.0


def _median_per_group(lists) -> list | None:
    lists = [p for p in lists if p is not None]
    if not lists:
        return None
    n = len(lists[0])
    out = []
    for i in range(n):
        vals = [float(p[i]) for p in lists
                if p[i] is not None and math.isfinite(float(p[i]))]
        out.append(_median(vals))
    return out


def classify_arm(settings: dict) -> str | None:
    recipe = str(settings.get("recipe", ""))
    lambda_dir = float(settings.get("lambda_dir", 1.0))
    signed = bool(settings.get("signed_vector_messages", False))
    for arm, (r, ld, sv) in ARM_DEFS.items():
        if recipe == r and abs(lambda_dir - ld) < 1e-6 and signed == sv:
            return arm
    return None


def four_metrics(summary: dict) -> dict:
    strata = summary.get("strata") or {}
    near_ts = strata.get("near_ts") or {}
    return {
        "mse_ratio_median": summary.get("mse_ratio_median"),
        "n_groups_ratio_below_one": summary.get("n_groups_ratio_below_one"),
        "direction_cosine_median": summary.get("direction_cosine_median"),
        "near_ts_cosine_median": near_ts.get("direction_cosine_median"),
    }


def gate_criteria(metrics: dict) -> dict:
    mse = metrics["mse_ratio_median"]
    n = metrics["n_groups_ratio_below_one"]
    cos = metrics["direction_cosine_median"]
    nts = metrics["near_ts_cosine_median"]
    c1 = bool(mse is not None and math.isfinite(float(mse))
              and float(mse) < float(GATE["median_mse_ratio_max"]))
    c2 = bool(n is not None and int(n) >= int(GATE["n_groups_ratio_below_one_min"]))
    c3 = bool(cos is not None and math.isfinite(float(cos))
              and float(cos) >= float(GATE["median_cosine_min"]))
    c4 = bool(nts is not None and math.isfinite(float(nts))
              and float(nts) >= float(GATE["near_ts_cosine_min"]))
    return {
        "mse_ratio_median_below_max": c1,
        "n_groups_ratio_below_one_min": c2,
        "direction_cosine_median_above_min": c3,
        "near_ts_cosine_above_min": c4,
        "pass": c1 and c2 and c3 and c4,
    }


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "PASS" if v else "FAIL"
    return f"{v:.4f}"


def _fmt_n(v) -> str:
    if v is None:
        return "—"
    return str(int(round(float(v))))


def _fmt_table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def build_per_group_tables(arms_by_seed: dict) -> dict:
    """Per-arm per-group monitor8 tables (full 8-group denominator)."""
    out = {}
    for arm in sorted(arms_by_seed):
        seeds = arms_by_seed[arm]
        group_ids = None
        per_seed = {}
        for seed in sorted(seeds):
            pg = seeds[seed]["per_group"]  # list of 8 dicts
            if group_ids is None:
                group_ids = [str(g["group_id"]) for g in pg]
            per_seed[seed] = {
                "group_ids": [str(g["group_id"]) for g in pg],
                "mse_ratio_vs_zero": [
                    (g.get("all") or {}).get("mse_ratio_vs_zero") for g in pg],
                "direction_cosine": [
                    (g.get("all") or {}).get("direction_cosine_absa_weighted")
                    for g in pg],
                "near_ts_cosine": [
                    ((g.get("strata") or {}).get("near_ts") or {}).get(
                        "direction_cosine_absa_weighted") for g in pg],
            }
        seed_list = sorted(seeds)
        mse_med = _median_per_group(
            [per_seed[s]["mse_ratio_vs_zero"] for s in seed_list])
        cos_med = _median_per_group(
            [per_seed[s]["direction_cosine"] for s in seed_list])
        nts_med = _median_per_group(
            [per_seed[s]["near_ts_cosine"] for s in seed_list])
        out[arm] = {
            "group_ids": group_ids,
            "per_seed": {str(s): per_seed[s] for s in seed_list},
            "median_across_seeds": {
                "mse_ratio_vs_zero": mse_med,
                "direction_cosine": cos_med,
                "near_ts_cosine": nts_med,
            },
        }
    return out


def d_rule_effective(x_med: dict, a_med: dict) -> dict:
    """B/C EFFECTIVE iff strictly better than A on BOTH cosine metrics AND
    not worse on MSE ratio AND not worse on groups<1.0 count (3-seed medians)."""
    x = x_med
    a = a_med

    def strict_better_cos(xv, av):
        return (xv is not None and av is not None
                and float(xv) > float(av))

    def not_worse_mse(xv, av):
        return (xv is not None and av is not None and float(xv) <= float(av))

    def not_worse_groups(xv, av):
        return (xv is not None and av is not None and int(xv) >= int(av))

    c1 = strict_better_cos(x["direction_cosine_median"],
                           a["direction_cosine_median"])
    c2 = strict_better_cos(x["near_ts_cosine_median"],
                           a["near_ts_cosine_median"])
    c3 = not_worse_mse(x["mse_ratio_median"], a["mse_ratio_median"])
    c4 = not_worse_groups(x["n_groups_ratio_below_one"],
                          a["n_groups_ratio_below_one"])
    return {
        "strictly_better_cosine": c1,
        "strictly_better_near_ts_cosine": c2,
        "not_worse_mse_ratio": c3,
        "not_worse_groups_below_one": c4,
        "effective": c1 and c2 and c3 and c4,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path,
                    default=PROJECT_ROOT / "outputs/pesn/dual_ts_v2_20261015"
                            / "wp41_bc")
    ap.add_argument("--diagnostics-dir", type=Path,
                    default=PROJECT_ROOT / "outputs/pesn/dual_ts_v2_20261015"
                            / "wp41_diagnostics")
    ap.add_argument("--protocol-sha", default=R2_SHA)
    args = ap.parse_args(argv)

    out_dir = args.out_dir
    diag_dir = args.diagnostics_dir
    diag_dir.mkdir(parents=True, exist_ok=True)

    # -- discover full-stage formal runs ------------------------------------
    runs: dict[str, dict] = {}
    for run_dir in sorted(out_dir.glob("run_bc_*")):
        rec_path = run_dir / "run_record.json"
        if not rec_path.is_file():
            continue
        rec = json.loads(rec_path.read_text())
        if rec.get("schema_version") != "pesn_dual_ts_bc_run_v2":
            continue
        settings = rec.get("settings") or {}
        if str(settings.get("stage", "")) != "full":
            continue
        if bool(settings.get("diagnostic_run", False)):
            continue
        if str(settings.get("protocol_sha256", "")) != args.protocol_sha:
            continue
        eval_path = run_dir / "monitor_eval.json"
        if not eval_path.is_file():
            continue
        arm = classify_arm(settings)
        if arm is None:
            continue
        seed = int(settings.get("seed", 0))
        evaluation = json.loads(eval_path.read_text())
        summary = evaluation["summary"]
        metrics = four_metrics(summary)
        criteria = gate_criteria(metrics)
        per_group = evaluation["per_group"]
        runs.setdefault(arm, {})[seed] = {
            "run_dir": run_dir.name,
            "metrics": metrics,
            "criteria": criteria,
            "gate": "PASS" if criteria["pass"] else "FAIL",
            "per_group": per_group,
            "c3_trigger": (evaluation.get("c3_trigger") or {}).get("status"),
        }

    # -- per-arm aggregation ------------------------------------------------
    arms_by_seed: dict[str, dict] = {}
    arm_summary: dict[str, dict] = {}
    for arm in ("A", "B", "C", "D"):
        if arm not in runs:
            continue
        seeds = runs[arm]
        arms_by_seed[arm] = seeds
        med_metrics = {
            k: _median([seeds[s]["metrics"][k] for s in sorted(seeds)])
            for k in ("mse_ratio_median", "n_groups_ratio_below_one",
                      "direction_cosine_median", "near_ts_cosine_median")
        }
        med_metrics["n_groups_ratio_below_one"] = (
            None if med_metrics["n_groups_ratio_below_one"] is None
            else int(round(med_metrics["n_groups_ratio_below_one"])))
        n_pass = sum(1 for s in seeds if seeds[s]["gate"] == "PASS")
        n = len(seeds)
        arm_summary[arm] = {
            "n_seeds": n,
            "seeds": sorted(seeds),
            "n_seeds_passing": n_pass,
            "arm_status": "PASS" if (n and n_pass * 3 >= 2 * n) else "BELOW_GATE",
            "median_metrics": med_metrics,
            "median_gate_criteria": gate_criteria(med_metrics),
            "per_seed": {
                str(s): {
                    "metrics": seeds[s]["metrics"],
                    "gate": seeds[s]["gate"],
                    "criteria": seeds[s]["criteria"],
                    "run_dir": seeds[s]["run_dir"],
                    "c3_trigger": seeds[s]["c3_trigger"],
                } for s in sorted(seeds)
            },
        }

    # -- D-rule comparison (on 3-seed medians) ------------------------------
    d_rule = {}
    a_med = arm_summary.get("A", {}).get("median_metrics")
    if a_med is not None:
        for arm in ("B", "C"):
            x_med = arm_summary.get(arm, {}).get("median_metrics")
            if x_med is not None:
                d_rule[arm] = d_rule_effective(x_med, a_med)
    run_d = bool(d_rule.get("B", {}).get("effective")
                 and d_rule.get("C", {}).get("effective"))

    payload = {
        "schema_version": "pesn_wp41_matrix_summary_v1",
        "protocol_sha256": args.protocol_sha,
        "gate": GATE,
        "run_d_decision": "RUN_D" if run_d else "SKIP_D",
        "d_rule": d_rule,
        "arms": arm_summary,
    }
    json_path = diag_dir / "matrix_summary.json"
    json_path.write_text(json.dumps(payload, indent=1, sort_keys=True,
                                    allow_nan=False) + "\n")

    # -- markdown -----------------------------------------------------------
    md: list[str] = []
    md.append("# WP4b v4.1 matrix summary\n")
    md.append(f"protocol_sha256 = `{args.protocol_sha[:16]}…`\n")

    gate_headers = ["arm", "seed", "MSE ratio med", "#<1.0", "cos med",
                    "near-TS cos", "gate"]
    for arm in ("A", "B", "C", "D"):
        if arm not in arm_summary:
            continue
        md.append(f"\n## {ARM_LABEL[arm]}\n")
        rows = []
        for seed in arm_summary[arm]["seeds"]:
            m = arm_summary[arm]["per_seed"][str(seed)]["metrics"]
            g = arm_summary[arm]["per_seed"][str(seed)]["gate"]
            rows.append([arm, seed, _fmt(m["mse_ratio_median"]),
                         _fmt_n(m["n_groups_ratio_below_one"]),
                         _fmt(m["direction_cosine_median"]),
                         _fmt(m["near_ts_cosine_median"]), g])
        mm = arm_summary[arm]["median_metrics"]
        rows.append(["**MEDIAN**", "—", _fmt(mm["mse_ratio_median"]),
                     _fmt_n(mm["n_groups_ratio_below_one"]),
                     _fmt(mm["direction_cosine_median"]),
                     _fmt(mm["near_ts_cosine_median"]),
                     arm_summary[arm]["arm_status"]])
        md.append(_fmt_table(gate_headers, rows))

    # D-rule table
    if d_rule:
        md.append("\n## D-rule comparison (3-seed medians)\n")
        md.append("B (resp. C) EFFECTIVE iff strictly better than A on BOTH "
                  "cosine metrics AND not worse on MSE ratio AND not worse on "
                  "groups<1.0 count.\n")
        headers = ["arm", "cos strictly >A", "near-TS cos strictly >A",
                   "MSE ratio not worse", "#<1.0 not worse", "EFFECTIVE"]
        rows = []
        for arm in ("B", "C"):
            if arm not in d_rule:
                continue
            d = d_rule[arm]
            rows.append([arm,
                         _fmt(d["strictly_better_cosine"]),
                         _fmt(d["strictly_better_near_ts_cosine"]),
                         _fmt(d["not_worse_mse_ratio"]),
                         _fmt(d["not_worse_groups_below_one"]),
                         "YES" if d["effective"] else "NO"])
        md.append(_fmt_table(headers, rows))
        md.append(f"\n**D decision: {'RUN D' if run_d else 'SKIP D'}**\n")

    (diag_dir / "matrix_summary.md").write_text("\n".join(md) + "\n")

    # -- per-group tables ---------------------------------------------------
    pg = build_per_group_tables(arms_by_seed)
    (diag_dir / "matrix_monitor_per_group.json").write_text(
        json.dumps({"schema_version": "pesn_wp41_matrix_per_group_v1",
                    "protocol_sha256": args.protocol_sha,
                    "arms": pg}, indent=1, sort_keys=True,
                   allow_nan=False) + "\n")

    pg_md: list[str] = ["# WP4b matrix — per-reaction monitor8 tables\n"]
    for arm in sorted(pg):
        if arm not in arms_by_seed:
            continue
        pg_md.append(f"\n## {ARM_LABEL[arm]}\n")
        group_ids = pg[arm]["group_ids"]
        seed_list = sorted(arms_by_seed[arm])
        headers = ["group"] + [f"s{str(s)[-2:]}" for s in seed_list] + ["MED"]
        rows_mse, rows_cos, rows_nts = [], [], []
        for i, gid in enumerate(group_ids):
            row_mse = [gid]
            row_cos = [gid]
            row_nts = [gid]
            for s in seed_list:
                row_mse.append(_fmt(pg[arm]["per_seed"][str(s)]
                                    ["mse_ratio_vs_zero"][i]))
                row_cos.append(_fmt(pg[arm]["per_seed"][str(s)]
                                    ["direction_cosine"][i]))
                row_nts.append(_fmt(pg[arm]["per_seed"][str(s)]
                                    ["near_ts_cosine"][i]))
            row_mse.append(_fmt(pg[arm]["median_across_seeds"]
                                ["mse_ratio_vs_zero"][i]))
            row_cos.append(_fmt(pg[arm]["median_across_seeds"]
                                ["direction_cosine"][i]))
            row_nts.append(_fmt(pg[arm]["median_across_seeds"]
                                ["near_ts_cosine"][i]))
            rows_mse.append(row_mse)
            rows_cos.append(row_cos)
            rows_nts.append(row_nts)
        pg_md.append("**MSE ratio vs zero-action**\n")
        pg_md.append(_fmt_table(headers, rows_mse))
        pg_md.append("\n**|a|-weighted cosine**\n")
        pg_md.append(_fmt_table(headers, rows_cos))
        pg_md.append("\n**near-TS cosine**\n")
        pg_md.append(_fmt_table(headers, rows_nts))
    (diag_dir / "matrix_monitor_per_group.md").write_text("\n".join(pg_md) + "\n")

    print(f"matrix_summary -> {json_path}")
    print(f"run_d_decision = {'RUN_D' if run_d else 'SKIP_D'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
