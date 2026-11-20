#!/usr/bin/env python3
"""a_max calibration projection (offline; zero QC).

Replays the stored fit50 expert paths through the resampler at alternative
``a_max`` values (0.05/0.1/0.2) to project node counts and the share of
paths that fit the frozen 100-step logical budget. The path geometry comes
from prep checkpoints (reset + sequential X_next); no ACP calls and no
truth reads happen here. This informs the pre-registered a_max calibration
decision (plan §6.1); it does not replace the requirement to rebuild the
action/budget data at the chosen value.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pes2ts_core.pesn.contracts.direct import DirectTransition  # noqa: E402
from pes2ts_core.pesn.offline.data.direct_preprocess import (  # noqa: E402
    masses_for,
    resample_path,
)


def reconstruct_path(doc: dict) -> np.ndarray | None:
    transitions = [DirectTransition.from_payload(t) for t in doc.get("transitions", [])]
    if not transitions:
        return None
    nodes = [np.asarray(transitions[0].state.coordinates, dtype=float)]
    for tr in transitions:
        if tr.next_state is not None and tr.next_state.observation_mask:
            nodes.append(np.asarray(tr.next_state.coordinates, dtype=float))
    # dedupe consecutive
    keep = [nodes[0]]
    for node in nodes[1:]:
        if np.abs(node - keep[-1]).max() > 1e-9:
            keep.append(node)
    return np.asarray(keep, dtype=float) if len(keep) >= 2 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    # Full resampled node counts come from diagnostics (pre-truncation).
    # Reconstructing from transitions would only see the truncated prefix
    # (that mistake overstates coverage), so scaling uses n_nodes.
    full_nodes = []
    for path in sorted(args.prep_dir.glob("*.json")):
        doc = json.loads(path.read_text())
        n_nodes = doc.get("diagnostics", {}).get("n_nodes")
        if n_nodes:
            full_nodes.append(float(n_nodes))
    projections: dict[str, dict] = {}
    if full_nodes:
        base_max = 0.1  # the value the stored runs used
        for a_max in (0.05, 0.1, 0.2):
            # nodes scale ~ inversely with the per-atom step bound
            scale = base_max / a_max
            steps = [(n - 1.0) * scale for n in full_nodes]
            fits = sum(1 for s in steps if s <= args.budget)
            projections[f"a_max_{a_max}"] = {
                "n_paths": len(steps),
                "steps_p50": round(float(np.median(steps)), 1),
                "steps_p90": round(float(np.percentile(steps, 90)), 1),
                "steps_max": round(float(max(steps)), 1),
                "paths_within_budget": fits,
                "coverage_frac_geometric": round(fits / len(steps), 3),
                "note": "geometric reachability only; no-progress losses "
                        "subtract independently and require a rebuild to measure",
            }
    payload = {
        "prep_dir": str(args.prep_dir),
        "budget_steps": args.budget,
        "method": ("full pre-truncation node counts from diagnostics, scaled "
                   "by the per-atom step bound; no ACP calls"),
        "projections": projections,
        "observed_terminals_note": (
            "the stored runs show the true goal/budget/no-progress split at "
            "a_max=0.1; coverage at other a_max values is projected only"
        ),
    }
    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
