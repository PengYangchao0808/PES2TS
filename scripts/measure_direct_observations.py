#!/usr/bin/env python3
"""Authoritative fit50 QC measurement from ACP receipts (WP1 §11.1 pricing).

Every observation receipt (``result.json``) carries the real ACP execution
wall time in ``duration_seconds`` and the cache flag; replaying identical
geometry returns the stored receipt without touching this number. This
script aggregates receipts per reaction so the pricing uses *real* QC
execution time, not preparation-side replay wall time.

Reads only the observation receipts (audit artifacts); no truth, no QC.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(p * (len(ordered) - 1)))))
    return ordered[idx]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    per_reaction = {}
    all_durations: list[float] = []
    total = 0
    failures = 0
    for reaction_dir in sorted(p for p in args.observations_dir.iterdir() if p.is_dir()):
        durations = []
        n_ok = 0
        n_fail = 0
        for receipt in reaction_dir.rglob("result.json"):
            try:
                doc = json.loads(receipt.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            total += 1
            if doc.get("success"):
                n_ok += 1
                duration = doc.get("duration_seconds")
                if isinstance(duration, (int, float)) and duration >= 0:
                    durations.append(float(duration))
                    all_durations.append(float(duration))
            else:
                n_fail += 1
        per_reaction[reaction_dir.name] = {
            "n_receipts": n_ok + n_fail, "n_success": n_ok, "n_failed": n_fail,
            "exec_seconds_total": round(sum(durations), 2),
            "exec_seconds_p50": round(percentile(durations, 0.5), 3),
        }
        failures += n_fail

    report = {
        "observations_dir": str(args.observations_dir),
        "n_reactions": len(per_reaction),
        "n_receipts_total": total,
        "n_failed_receipts": failures,
        "qc_wall_seconds_total": round(sum(all_durations), 1),
        "qc_wall_hours_total": round(sum(all_durations) / 3600, 2),
        "seconds_per_call": {
            "p50": round(percentile(all_durations, 0.5), 3),
            "p90": round(percentile(all_durations, 0.9), 3),
            "p95": round(percentile(all_durations, 0.95), 3),
            "max": round(max(all_durations), 1) if all_durations else None,
        },
        "per_reaction": per_reaction,
        "pricing_inputs_500": None,
    }
    if all_durations:
        per_call_p90 = percentile(all_durations, 0.9)
        calls_per_reaction_p90 = percentile(
            [v["n_receipts"] for v in per_reaction.values()], 0.9)
        calls_500 = calls_per_reaction_p90 * 500
        seconds_500 = calls_500 * per_call_p90
        report["pricing_inputs_500"] = {
            "projected_qc_calls_500": int(calls_500),
            "projected_hours_serial_p90": round(seconds_500 / 3600, 1),
            "projected_hours_2way_p90": round(seconds_500 / 3600 / 2, 1),
            "projected_hours_4way_p90": round(seconds_500 / 3600 / 4, 1),
        }
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(payload)
    summary = {k: v for k, v in report.items() if k != "per_reaction"}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
