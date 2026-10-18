#!/usr/bin/env python3
"""Summarize fit50 preparation measurements from prep checkpoints (WP1).

Reads only the per-reaction checkpoints under ``audit/prep/<selection>/``
(no truth, no QC) and reports statuses, node counts, QC call counts, cache
hits, wall times, and the p50/p90/p95 statistics the plan requires before
releasing 500/2k batches (plan §11.1: unpriced batches are never released).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(p * (len(ordered) - 1)))))
    return ordered[idx]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    docs = [json.loads(p.read_text()) for p in sorted(args.prep_dir.glob("*.json"))]
    statuses: dict[str, int] = {}
    reasons: dict[str, int] = {}
    nodes, qc_calls, cache_hits, walls, per_node_seconds = [], [], [], [], []
    completed_walls = []
    for doc in docs:
        statuses[doc["status"]] = statuses.get(doc["status"], 0) + 1
        reasons[doc["reason"]] = reasons.get(doc["reason"], 0) + 1
        diag = doc.get("diagnostics", {})
        n_nodes = float(diag.get("n_nodes", 0))
        n_qc = float(diag.get("qc_calls", 0))
        wall = float(diag.get("wall_seconds_total", 0.0))
        if n_nodes > 0:
            nodes.append(n_nodes)
            qc_calls.append(n_qc)
            cache_hits.append(float(diag.get("qc_cache_hits", 0)))
            walls.append(wall)
            if n_qc > 0:
                per_node_seconds.append(wall / n_qc)
            if doc["status"] == "completed":
                completed_walls.append(wall)

    report = {
        "selection_dir": str(args.prep_dir),
        "n_reactions": len(docs),
        "statuses": statuses,
        "termination_reasons": reasons,
        "nodes": {
            "p50": percentile(nodes, 0.50), "p90": percentile(nodes, 0.90),
            "p95": percentile(nodes, 0.95),
            "max": max(nodes) if nodes else None,
        },
        "qc_calls_total": int(sum(qc_calls)),
        "qc_cache_hits_total": int(sum(cache_hits)),
        "wall_seconds_total": round(sum(walls), 1),
        "seconds_per_qc_call": {
            "p50": round(percentile(per_node_seconds, 0.50), 2),
            "p90": round(percentile(per_node_seconds, 0.90), 2),
            "p95": round(percentile(per_node_seconds, 0.95), 2),
        },
        "wall_per_completed_reaction": {
            "p50": round(percentile(completed_walls, 0.50), 1),
            "p90": round(percentile(completed_walls, 0.90), 1),
        },
        "pricing_inputs_500": None,
        "note": (
            "wall seconds reflect the LAST preparation pass; after a refresh "
            "replay they are cache-hit timings, NOT original QC execution. "
            "Authoritative QC time lives in the observation receipts "
            "(scripts/measure_direct_observations.py); pricing must use that"
        ),
    }
    if nodes:
        p90_nodes = percentile(nodes, 0.90)
        p90_qc_seconds = percentile(per_node_seconds, 0.90)
        report["pricing_inputs_500"] = {
            "projected_qc_calls_500": round(p90_nodes * 500),
            "projected_seconds_serial_p90": round(p90_nodes * 500 * p90_qc_seconds),
            "projected_hours_serial_p90": round(p90_nodes * 500 * p90_qc_seconds / 3600, 1),
            "projected_hours_2way_p90": round(p90_nodes * 500 * p90_qc_seconds / 3600 / 2, 1),
        }
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(payload)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
