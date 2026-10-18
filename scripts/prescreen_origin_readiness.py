"""Offline origin-preparability pre-screen over a cohort file.

Runs the exact offline prefix of the origin-preparation chain (endpoint
graph rebuild + choose_origin — no QC) over the given reactions and reports
the typed funnel.  Used to re-freeze cohorts whose members pass the
input-screen, and to measure the pool pass rate before mass preparation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from prepare_rl_origins import build_control_snapshot, export_path  # noqa: E402
from pes2ts_core.generation.planning.connectivity_plan import choose_origin  # noqa: E402
from pes2ts_core.generation.planning.graph_rebuild import (  # noqa: E402
    load_endpoint_materials_from_export,
    rebuild_endpoint_graphs,
)


def prescreen(reaction_id: str, snapshot: dict) -> tuple[str, dict | None]:
    try:
        bundle = rebuild_endpoint_graphs(
            snapshot["reaction_smiles"],
            load_endpoint_materials_from_export(snapshot),
        )
    except Exception as exc:  # noqa: BLE001
        return f"graph_rebuild:{type(exc).__name__}", None
    choice = choose_origin(snapshot, bundle)
    status = str(choice.get("status"))
    if status == "ready":
        return "ready", choice
    return status, choice


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reactions", required=True, help="JSON list file")
    parser.add_argument("--key", default=None)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    payload = json.loads(Path(args.reactions).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload[args.key] if args.key else next(
            v for v in payload.values() if isinstance(v, list)
        )
    reaction_ids = [str(rid) for rid in payload]
    if args.sample:
        stride = max(1, len(reaction_ids) // args.sample)
        reaction_ids = reaction_ids[::stride][: args.sample]

    import pyarrow.parquet as pq

    smiles = {
        str(row["reaction_id"]): str(row["reaction_smiles"])
        for row in pq.read_table(
            REPO_ROOT / "data/interim/inventory.parquet",
            columns=["reaction_id", "reaction_smiles"],
        ).to_pylist()
    }

    funnel: Counter[str] = Counter()
    ready: list[str] = []
    rows: list[dict] = []
    for reaction_id in reaction_ids:
        source = export_path(reaction_id)
        if not source.is_file():
            funnel["export_missing"] += 1
            rows.append({"reaction_id": reaction_id, "status": "export_missing"})
            continue
        export_doc = json.loads(source.read_text(encoding="utf-8"))
        snapshot = build_control_snapshot(export_doc, smiles.get(reaction_id, ""))
        status, _choice = prescreen(reaction_id, snapshot)
        funnel[status] += 1
        rows.append({"reaction_id": reaction_id, "status": status})
        if status == "ready":
            ready.append(reaction_id)
    report = {
        "n_screened": len(reaction_ids),
        "funnel": dict(sorted(funnel.items())),
        "ready_fraction": (
            round(funnel["ready"] / len(reaction_ids), 4) if reaction_ids else 0.0
        ),
    }
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {**report, "ready_ids": ready, "rows": rows},
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
