"""Origin preparation for new RL control-round reactions (plan §12.3).

For each cohort reaction this runs the SAME production chain the G2.T
campaign uses up to (and including) plan.json — endpoint graph rebuild,
typed origin choice, real ACP origin free-optimization, plan build, online
checks — and stages the g1_v2 export snapshot beside it.  The continuation
and candidate stages are deliberately NOT run: the control round consumes
plan.json + snapshot only.  Every reaction ends with a typed row in the
funnel (selected / precheck_rejected / origin_failed / plan_built); nothing
is dropped silently.  Real QC: the origin step launches ACP calculations
under the frozen wiring; per-case wall time and request counts are recorded.

Resume: a case with an existing plan.json is skipped and counted as
`resumed` unless --force.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from pes2ts_core.generation.campaign import production_backends  # noqa: E402
from pes2ts_core.generation.planning.connectivity_plan import (  # noqa: E402
    build_plan,
    choose_origin,
)
from pes2ts_core.generation.planning.graph_rebuild import (  # noqa: E402
    load_endpoint_materials_from_export,
    rebuild_endpoint_graphs,
)
from pes2ts_core.generation.planning.structure_checks import (  # noqa: E402
    add_online_checks,
)
from pes2ts_core.utils.jsonio import write_json  # noqa: E402

G1V2_EXPORT_DIR = REPO_ROOT / "data/interim/g1_v2/export"
ORCA_LIB_DIR = Path("/opt/software/orca")


def _ensure_orca_runtime() -> None:
    """ACP's ORCA binary needs its own lib dir on LD_LIBRARY_PATH."""
    if not (ORCA_LIB_DIR / "liborca_tools_5_0_4.so.5").is_file():
        return
    existing = __import__("os").environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in existing.split(":") if p]
    if str(ORCA_LIB_DIR) in parts:
        return
    __import__("os").environ["LD_LIBRARY_PATH"] = ":".join(
        [str(ORCA_LIB_DIR), *parts]
    )


def _inventory_smiles() -> dict[str, str]:
    import pyarrow.parquet as pq

    table = pq.read_table(
        REPO_ROOT / "data/interim/inventory.parquet",
        columns=["reaction_id", "reaction_smiles"],
    )
    return {
        str(row["reaction_id"]): str(row["reaction_smiles"])
        for row in table.to_pylist()
    }


def build_control_snapshot(export_doc: dict, smiles: str) -> dict:
    """Augment a g1_v2 export with the control-round snapshot fields.

    Mirrors the Demo24 staging snapshot contract: adds the inventory
    reaction SMILES and the endpoint electronic state derived from the
    export's own charge totals (both sides share multiplicity_max by
    construction, so choose_origin's equal-state gate is honest).
    """
    snapshot = dict(export_doc)
    snapshot["reaction_smiles"] = smiles
    charge_r = int(export_doc["charge_total_reactants"])
    charge_p = int(export_doc["charge_total_products"])
    multiplicity = int(export_doc["multiplicity_max"])
    snapshot["endpoint_electronic"] = {
        "reactant": {"charge": charge_r, "multiplicity": multiplicity},
        "product": {"charge": charge_p, "multiplicity": multiplicity},
    }
    snapshot["endpoint_electronic_source"] = "g1_v2_export_totals"
    snapshot["split"] = "train"
    snapshot["stratum"] = "scale_rl_fit_pool"
    return snapshot


def export_path(reaction_id: str) -> Path:
    shard = f"{int(reaction_id.split('_')[1]) // 1000:05d}"
    return G1V2_EXPORT_DIR / shard / f"{reaction_id}.json"


def g2t_policies():
    """Continuation policy exactly as the frozen g2t_real campaign config."""
    import yaml
    from dataclasses import asdict

    from pes2ts_core.generation.planning.branch_calibration import BranchPolicy
    from pes2ts_core.generation.planning.continuation import ContinuationPolicy
    from pes2ts_core.generation.planning.local_corrector import (
        LocalCorrectorPolicy,
    )

    config = yaml.safe_load((REPO_ROOT / "config/defaults.yaml").read_text())
    g2 = config.get("g2", {})
    continuation = dict(g2.get("continuation") or {})
    branch = continuation.pop("branch_policy", {}) or {}
    policy = ContinuationPolicy(
        **{
            **asdict(ContinuationPolicy()),
            **{
                k: continuation[k]
                for k in (
                    "cumulative_rmsd_limit",
                    "max_curvature_probes",
                    "max_seconds",
                )
                if k in continuation
            },
        }
    )
    local = dict(g2.get("local_corrector") or {})
    local_policy = LocalCorrectorPolicy(
        **{
            **asdict(LocalCorrectorPolicy()),
            **{k: local[k] for k in ("max_seconds",) if k in local},
        }
    )
    branch_policy = BranchPolicy(**{**asdict(BranchPolicy()), **branch})
    return policy, local_policy, branch_policy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reactions", required=True, metavar="RID_FILE|RID,RID,...",
                        help="JSON list file or comma-separated reaction ids")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--acp-root", default=os.environ.get("PES2TS_ACP_ROOT"))
    parser.add_argument("--acp-python", default=os.environ.get("PES2TS_ACP_PYTHON"))
    parser.add_argument("--acp-config", default=None,
                        help="ACP config YAML (defaults to <acp-root>/config/defaults.yaml)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.acp_root or not args.acp_python:
        parser.error(
            "set PES2TS_ACP_ROOT and PES2TS_ACP_PYTHON, or pass "
            "--acp-root/--acp-python"
        )

    raw = args.reactions
    _ensure_orca_runtime()
    if raw.endswith(".json"):
        payload = json.loads(Path(raw).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("reaction_ids", "rl_smoke_5_not_counted",
                        "rl_fit_batch1_50", "rl_dev_50_fixed_checkpoint_subset"):
                if key in payload:
                    payload = payload[key]
                    break
        reaction_ids = [str(rid) for rid in payload]
    else:
        reaction_ids = [rid.strip() for rid in raw.split(",") if rid.strip()]
    if args.limit is not None:
        reaction_ids = reaction_ids[: args.limit]

    acp_config = args.acp_config or str(
        Path(args.acp_root) / "config/defaults.yaml"
    )
    wiring = {
        "root": str(Path(args.acp_root)),
        "python": args.acp_python,
        "config_path": acp_config,
        "resolve_from_environment": True,
    }

    import yaml

    defaults = yaml.safe_load((REPO_ROOT / "config/defaults.yaml").read_text())
    origin_config = dict(defaults.get("g2", {}).get("origin_preparation") or {})
    policy, local_policy, branch_policy = g2t_policies()
    backends = production_backends(
        acp_wiring=wiring, origin_config=origin_config,
        local_policy=local_policy, branch_policy=branch_policy,
    )

    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    staging = args.output_root / "snapshots"
    staging.mkdir(exist_ok=True)
    smiles_by_reaction = _inventory_smiles()
    rows: list[dict] = []
    for reaction_id in reaction_ids:
        case_root = args.output_root / reaction_id
        row: dict = {
            "reaction_id": reaction_id,
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        plan_path = case_root / "plan.json"
        if plan_path.is_file() and not args.force:
            row["status"] = "resumed"
            row["plan_sha256"] = hashlib.sha256(
                plan_path.read_bytes()
            ).hexdigest()
            rows.append(row)
            print(f"{reaction_id}: resumed (plan exists)", flush=True)
            continue
        source = export_path(reaction_id)
        if not source.is_file():
            row["status"] = "precheck_rejected"
            row["failure_code"] = "G1V2_EXPORT_MISSING"
            row["failure_detail"] = str(source)
            rows.append(row)
            print(f"{reaction_id}: PRECHECK_REJECTED export missing", flush=True)
            continue
        snapshot_target = staging / f"{reaction_id}.json"
        if not snapshot_target.is_file():
            export_doc = json.loads(source.read_text(encoding="utf-8"))
            smiles = smiles_by_reaction.get(reaction_id)
            if not smiles:
                row["status"] = "precheck_rejected"
                row["failure_code"] = "INVENTORY_SMILES_MISSING"
                rows.append(row)
                print(f"{reaction_id}: PRECHECK_REJECTED smiles missing", flush=True)
                continue
            snapshot_target.write_text(
                json.dumps(
                    build_control_snapshot(export_doc, smiles),
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                ),
                encoding="utf-8",
            )
        snapshot = json.loads(snapshot_target.read_text(encoding="utf-8"))
        row["snapshot_sha256"] = hashlib.sha256(
            snapshot_target.read_bytes()
        ).hexdigest()
        started = time.monotonic()
        try:
            bundle = rebuild_endpoint_graphs(
                snapshot["reaction_smiles"],
                load_endpoint_materials_from_export(snapshot),
            )
            choice = choose_origin(snapshot, bundle)
            if choice.get("status") != "ready":
                row["status"] = "precheck_rejected"
                row["failure_code"] = f"choose_origin:{choice.get('status')}"
                rows.append(row)
                print(f"{reaction_id}: PRECHECK_REJECTED {choice.get('status')}", flush=True)
                continue
            case_root.mkdir(parents=True, exist_ok=True)
            origin_evidence = backends.prepare_origin(
                np.asarray(
                    snapshot[
                        "r_coordinates" if choice["side"] == "R" else "p_coordinates"
                    ],
                    float,
                ),
                snapshot["elements"],
                choice["bond_indices"],
                choice["charge"],
                choice["multiplicity"],
                case_root,
            )
            write_json(case_root / "origin_evidence.json", origin_evidence)
            row["origin_status"] = origin_evidence.get("status")
            if origin_evidence.get("status") != "prepared":
                row["status"] = "origin_failed"
                row["failure_code"] = origin_evidence.get("failure_code")
                rows.append(row)
                print(f"{reaction_id}: ORIGIN_FAILED {row['failure_code']}", flush=True)
                continue
            plan = build_plan(
                snapshot,
                bundle,
                choice,
                np.asarray(origin_evidence["coordinates"], float),
                policy=policy,
            )
            plan = add_online_checks(plan, snapshot, bundle)
            write_json(plan_path, plan)
            row["status"] = "plan_built"
            row["plan_sha256"] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        except Exception as exc:  # noqa: BLE001 - typed row, batch continues
            row["status"] = "origin_failed"
            row["failure_code"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        row["wall_seconds"] = round(time.monotonic() - started, 2)
        rows.append(row)
        print(
            f"{reaction_id}: {row['status']} ({row.get('failure_code') or 'ok'}) "
            f"{row['wall_seconds']}s",
            flush=True,
        )

    funnel: dict[str, int] = {}
    for row in rows:
        funnel[row["status"]] = funnel.get(row["status"], 0) + 1
    summary = {
        "schema_version": "pes2ts_scale_origin_prep_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "acp_wiring": {
            "root": wiring["root"],
            "python": wiring["python"],
            "config_path": wiring["config_path"],
        },
        "origin_config": origin_config,
        "n_selected": len(reaction_ids),
        "funnel": funnel,
        "total_wall_seconds": round(sum(r.get("wall_seconds", 0.0) for r in rows), 2),
        "cases": rows,
    }
    write_json(args.output_root / "origin_prep_summary.json", summary)
    print(json.dumps(funnel, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
