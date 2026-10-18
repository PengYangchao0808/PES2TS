"""Prepare repaired startpoint plans for the two historical Demo24 cases.

Runs the SAME production input pipeline the historical campaign used
(``rebuild_endpoint_graphs -> choose_origin -> prepare_origin (ACP
BatchOptimize free-opt) -> build_plan -> add_online_checks``) for:

* ``RXN_0000155302`` — endpoint-collision case; consumes the repaired
  snapshot built by ``build_demo24_control_inputs.py``;
* ``RXN_0000171675`` — origin-budget case; the original snapshot is fine,
  only the origin free-optimization re-runs under the NEW frozen recipe
  (registered in ``origin_repair_recipe_v1.json`` BEFORE this execution).

Each output plan lands under ``<output-root>/repaired_plans/<rid>/plan.json``
(the ``--repaired-root`` overlay of the control round) with its origin
evidence and a preparation cost ledger entry.  The control round then reads
these plans exactly like the 22 historical legacy plans.  This script starts
REAL ACP calculations; its cost is billed separately from the scan envelope.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pes2ts_core.generation.planning.connectivity_plan import build_plan, choose_origin  # noqa: E402
from pes2ts_core.generation.planning.graph_rebuild import (  # noqa: E402
    load_endpoint_materials_from_export,
    rebuild_endpoint_graphs,
)
from pes2ts_core.generation.planning.origin_preparation import prepare_origin  # noqa: E402
from pes2ts_core.generation.planning.structure_checks import add_online_checks  # noqa: E402
from pes2ts_core.generation.campaign import load_campaign_manifest  # noqa: E402
from pes2ts_core.integration.acp.origin_backend import ORCAOriginPreparation  # noqa: E402

DEFAULT_CASES = ("RXN_0000155302", "RXN_0000171675")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def _load_snapshot(manifest: dict, reaction_id: str) -> dict:
    entry = next(c for c in manifest["cases"] if c["reaction_id"] == reaction_id)
    path = Path(entry["snapshot_path"])
    if not path.is_absolute():
        path = REPO_ROOT / path
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path,
        default=REPO_ROOT / "outputs/pesn/demo24_rl_v1_20261007",
    )
    parser.add_argument("--cases", nargs="*", default=list(DEFAULT_CASES))
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = load_campaign_manifest(args.manifest)
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    repaired_root = args.output_root / "repaired_plans"
    repaired_root.mkdir(parents=True, exist_ok=True)
    ledger_path = args.output_root / "prep_cost_ledger.jsonl"

    for reaction_id in args.cases:
        plan_path = repaired_root / reaction_id / "plan.json"
        if plan_path.is_file() and not args.force:
            print(f"{reaction_id}: repaired plan already present, skipping")
            continue
        case_dir = repaired_root / reaction_id
        case_dir.mkdir(parents=True, exist_ok=True)
        snapshot = _load_snapshot(manifest, reaction_id)
        bundle = rebuild_endpoint_graphs(
            snapshot["reaction_smiles"], load_endpoint_materials_from_export(snapshot)
        )
        choice = choose_origin(snapshot, bundle)
        if choice.get("status") != "ready":
            print(
                f"{reaction_id}: choose_origin -> {choice.get('status')}; "
                "refusing to prepare a plan from an unqualified input"
            )
            return 3
        side_geometry = snapshot[
            "r_coordinates" if choice["side"] == "R" else "p_coordinates"
        ]
        backend = ORCAOriginPreparation(
            elements=list(snapshot["elements"]),
            charge=choice["charge"],
            multiplicity=choice["multiplicity"],
            folder=case_dir / "origin_backend",
            method=recipe["method"],
            timeout_seconds=float(recipe["timeout_seconds"]),
            nproc=int(recipe["nproc"]),
            acp_wiring=dict(manifest["acp_wiring"]),
            capability_probe=bool(recipe.get("capability_probe", True)),
        )
        started = time.monotonic()
        evidence = prepare_origin(
            side_geometry,
            snapshot["elements"],
            choice["bond_indices"],
            backend.evaluate_free,
            method_context={
                "method": backend.method,
                "charge": choice["charge"],
                "multiplicity": choice["multiplicity"],
                "elements": list(snapshot["elements"]),
            },
            budget_seconds=float(recipe["timeout_seconds"]),
        )
        _write_json(case_dir / "origin_evidence.json", evidence)
        if evidence.get("status") != "prepared":
            print(
                f"{reaction_id}: origin preparation failed: "
                f"{evidence.get('failure_code')} {evidence.get('failure_detail')}"
            )
            return 4
        plan = build_plan(
            snapshot,
            bundle,
            choice,
            __import__("numpy").asarray(evidence["coordinates"], float),
        )
        plan = add_online_checks(plan, snapshot, bundle)
        plan["startpoint_repair_recipe"] = {
            "recipe_path": str(args.recipe),
            "historical_failure": (
                "INPUT_GEOMETRY_INVALID" if reaction_id == "RXN_0000155302"
                else "ORIGIN_BUDGET_EXHAUSTED"
            ),
        }
        _write_json(plan_path, plan)
        with ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "reaction_id": reaction_id,
                "phase": "startpoint_preparation",
                "status": "prepared",
                "duration_seconds": round(time.monotonic() - started, 3),
                "origin_evidence_duration": evidence.get("duration_seconds"),
                "recipe": recipe,
                "ts": __import__("datetime").datetime.now(
                    __import__("datetime").UTC
                ).isoformat(timespec="seconds"),
            }, sort_keys=True) + "\n")
        print(
            f"{reaction_id}: plan prepared (origin free-opt "
            f"{evidence.get('duration_seconds', 0):.1f}s, side {choice['side']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
