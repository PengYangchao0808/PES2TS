#!/usr/bin/env python3
"""WP4b A45-vs-A50 sensitivity contrast (F11) — mechanical, reads two run dirs.

Compares the A-config full 20k run on the FULL fit50 population (A50, seed
20261017, ``run_bc_bc_nll_seed20261017_stage-full``) against the A-config
diagnostic run on fit50 MINUS the five disputed reactions (A45, seed
20261017, ``run_bc_bc_nll_seed20261017_stage-full_diag``), SAME seed, SAME
evaluation set (the frozen monitor8 partition), FULL 8-group denominator.

Emits ``a45_vs_a50.{json,md}`` under the diagnostics dir: per-group tables for
both runs, the four gate metrics for both, and a per-group "flip" flag for each
metric threshold crossing (MSE ratio vs 1.0, |a|-w cosine vs 0.0, near-TS
cosine vs 0.4).

No QC; no source changes; reads run artifacts only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WP41_BC = PROJECT_ROOT / "outputs/pesn/dual_ts_v2_20261015/wp41_bc"
DIAG = PROJECT_ROOT / "outputs/pesn/dual_ts_v2_20261015/wp41_diagnostics"

A50_DIR = "run_bc_bc_nll_seed20261017_stage-full"
A45_DIR = "run_bc_bc_nll_seed20261017_stage-full_diag"


def load_eval(run_dir: Path) -> dict:
    return json.loads((run_dir / "monitor_eval.json").read_text())


def _fmt(v) -> str:
    if v is None:
        return "—"
    return f"{v:.4f}"


def _fmt_n(v) -> str:
    if v is None:
        return "—"
    return str(int(round(float(v))))


def _table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a50-dir", type=Path, default=WP41_BC / A50_DIR)
    ap.add_argument("--a45-dir", type=Path, default=WP41_BC / A45_DIR)
    ap.add_argument("--out-dir", type=Path, default=DIAG)
    args = ap.parse_args(argv)

    e50 = load_eval(args.a50_dir)
    e45 = load_eval(args.a45_dir)

    def pg_map(e):
        return {str(g["group_id"]): g for g in e["per_group"]}

    g50 = pg_map(e50)
    g45 = pg_map(e45)
    group_ids = [str(g["group_id"]) for g in e50["per_group"]]

    def all_metrics(g):
        a = g["all"]
        nts = (g.get("strata") or {}).get("near_ts") or {}
        return {
            "mse_ratio_vs_zero": a["mse_ratio_vs_zero"],
            "direction_cosine": a["direction_cosine_absa_weighted"],
            "near_ts_cosine": nts.get("direction_cosine_absa_weighted"),
        }

    per_group = {}
    flips = {"mse_ratio": [], "cosine_sign": [], "near_ts": []}
    for gid in group_ids:
        m50 = all_metrics(g50[gid])
        m45 = all_metrics(g45[gid])
        # flip: threshold crossing between the two runs
        mse_flip = (m50["mse_ratio_vs_zero"] is not None
                    and m45["mse_ratio_vs_zero"] is not None
                    and (m50["mse_ratio_vs_zero"] < 1.0)
                    != (m45["mse_ratio_vs_zero"] < 1.0))
        cos_flip = (m50["direction_cosine"] is not None
                    and m45["direction_cosine"] is not None
                    and (m50["direction_cosine"] > 0.0)
                    != (m45["direction_cosine"] > 0.0))
        nts_flip = (m50["near_ts_cosine"] is not None
                    and m45["near_ts_cosine"] is not None
                    and (m50["near_ts_cosine"] >= 0.4)
                    != (m45["near_ts_cosine"] >= 0.4))
        per_group[gid] = {
            "a50": m50, "a45": m45,
            "flip": {"mse_ratio": mse_flip, "cosine_sign": cos_flip,
                     "near_ts": nts_flip},
        }
        if mse_flip:
            flips["mse_ratio"].append(gid)
        if cos_flip:
            flips["cosine_sign"].append(gid)
        if nts_flip:
            flips["near_ts"].append(gid)

    def summary4(e):
        s = e["summary"]
        return {
            "mse_ratio_median": s["mse_ratio_median"],
            "n_groups_ratio_below_one": s["n_groups_ratio_below_one"],
            "direction_cosine_median": s["direction_cosine_median"],
            "near_ts_cosine_median":
                (s.get("strata") or {}).get("near_ts", {}).get(
                    "direction_cosine_median"),
        }

    payload = {
        "schema_version": "pesn_wp41_a45_vs_a50_v1",
        "a50_dir": args.a50_dir.name,
        "a45_dir": args.a45_dir.name,
        "a50_summary": summary4(e50),
        "a45_summary": summary4(e45),
        "flipped_groups": flips,
        "per_group": per_group,
    }
    out_json = args.out_dir / "a45_vs_a50.json"
    out_json.write_text(json.dumps(payload, indent=1, sort_keys=True,
                                   allow_nan=False) + "\n")

    md: list[str] = ["# A45 vs A50 sensitivity contrast (F11)\n"]
    md.append(f"- A50 dir: `{args.a50_dir.name}` (full fit50, seed 20261017)\n")
    md.append(f"- A45 dir: `{args.a45_dir.name}` (fit50 − disputed-5, seed 20261017)\n")
    md.append("\n## Four frozen gate metrics\n")
    headers = ["metric", "A50", "A45"]
    rows = [
        ["MSE ratio median", _fmt(payload["a50_summary"]["mse_ratio_median"]),
         _fmt(payload["a45_summary"]["mse_ratio_median"])],
        ["#groups <1.0", _fmt_n(payload["a50_summary"]["n_groups_ratio_below_one"]),
         _fmt_n(payload["a45_summary"]["n_groups_ratio_below_one"])],
        ["cos median", _fmt(payload["a50_summary"]["direction_cosine_median"]),
         _fmt(payload["a45_summary"]["direction_cosine_median"])],
        ["near-TS cos", _fmt(payload["a50_summary"]["near_ts_cosine_median"]),
         _fmt(payload["a45_summary"]["near_ts_cosine_median"])],
    ]
    md.append(_table(headers, rows))

    md.append("\n## Groups with status flips\n")
    md.append("- MSE ratio crossing 1.0: "
              f"{', '.join(flips['mse_ratio']) or 'none'}\n")
    md.append("- cosine sign flip: "
              f"{', '.join(flips['cosine_sign']) or 'none'}\n")
    md.append("- near-TS cosine crossing 0.4: "
              f"{', '.join(flips['near_ts']) or 'none'}\n")

    md.append("\n## Per-group tables (full 8-group denominator)\n")
    hdr = ["group", "A50 mse", "A45 mse", "A50 cos", "A45 cos",
           "A50 nTS", "A45 nTS", "flips"]
    rows = []
    for gid in group_ids:
        p = per_group[gid]
        fl = p["flip"]
        fl_str = "".join([
            "M" if fl["mse_ratio"] else "",
            "C" if fl["cosine_sign"] else "",
            "T" if fl["near_ts"] else ""]) or "—"
        rows.append([gid,
                     _fmt(p["a50"]["mse_ratio_vs_zero"]),
                     _fmt(p["a45"]["mse_ratio_vs_zero"]),
                     _fmt(p["a50"]["direction_cosine"]),
                     _fmt(p["a45"]["direction_cosine"]),
                     _fmt(p["a50"]["near_ts_cosine"]),
                     _fmt(p["a45"]["near_ts_cosine"]),
                     fl_str])
    md.append(_table(hdr, rows))
    (args.out_dir / "a45_vs_a50.md").write_text("\n".join(md) + "\n")

    print(f"a45_vs_a50 -> {out_json}")
    print(f"flips: {flips}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
