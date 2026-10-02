"""Rerun frozen Demo24 through F/B-only, single-ended continuation.

Run in the existing Linux ORCA environment. No service or ACP source changes.
Outputs and every rejected attempt are separate from the frozen first round.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

# Existing WSL ORCA installation runs under the root-owned compute account.
os.environ.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
os.environ.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
os.environ["PATH"] = "/opt/openmpi418/bin:/opt/orca_6_1_1:" + os.environ.get("PATH", "")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, os.environ.get("ACP_SOURCE", "/mnt/e/Calculations/Common_Script/Auto_Calc_Platform/ACP_V1_20260811/src"))

from pes2ts_core.integration.acp.continuation_backend import ORCAContinuationBackend, write_json
from pes2ts_core.g1.reaction_edit_graph import build_reaction_edit_graph
from pes2ts_core.generation.planning.connectivity_plan import build_plan, choose_origin, input_screen
from pes2ts_core.generation.planning.continuation import ContinuationPolicy, run_continuation
from pes2ts_core.generation.planning.graph_rebuild import load_endpoint_materials_from_export, rebuild_endpoint_graphs
from pes2ts_core.generation.planning.synchronized_path import align, digest

OUT = ROOT / "outputs/PES2TS_Demo24_continuation_round2_20261002"
OLD = ROOT / "outputs/PES2TS_Demo24_synchronized_reference_20261002"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def xyz(x, elements, comment):
    return "\n".join([str(len(elements)), comment] + [e+" "+" ".join(f"{v:.12f}" for v in row) for e, row in zip(elements, x)])+"\n"


def rank_candidates(result, plan):
    frames = result["frames"]
    peaks = []
    for i in range(1, len(frames)-1):
        left, centre, right = (f["energy_hartree"] for f in frames[i-1:i+2])
        if centre > left and centre >= right:
            peaks.append({"frame_index": i, "lambda": frames[i]["lambda"], "energy_hartree": centre,
                          "bracket": [frames[i-1]["lambda"], frames[i+1]["lambda"]],
                          "evidence": "sampled_constrained_energy_maximum", "ts_verified": False})
    peaks.sort(key=lambda r: r["energy_hartree"], reverse=True)
    return {"candidates": peaks[:3], "completed_interval": result["completed_interval"],
            "ranking_status": "candidate_only" if peaks else "no_interior_energy_peak",
            "physical_gradient_status": "not_collected", "frequency_verified": False, "irc_verified": False}


def run_case(path, arm):
    s = read(path); rid = s["reaction_id"]
    folder = OUT / rid / arm
    folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / "result.json"
    if receipt.exists():
        existing = read(receipt)
        if existing["source_sha256"] != digest(s):
            raise ValueError("SOURCE_CHANGED")
        return existing
    b = rebuild_endpoint_graphs(s["reaction_smiles"], load_endpoint_materials_from_export(s))
    graph = build_reaction_edit_graph(b)
    origin = choose_origin(s, b)
    if arm == "reverse_origin" and origin["status"] == "ready":
        other = next(r for r in origin["sides"] if r["side"] != origin["side"])
        if other["failures"]:
            origin = {**origin, "status": "ALTERNATE_ORIGIN_INVALID"}
        else:
            origin = {**origin, "side": other["side"], "bond_indices": other["bond_indices"]}
    record = {"reaction_id": rid, "source_sha256": digest(s), "arm": arm, "origin": origin,
              "active_counts": dict(Counter(e.edit_kind for e in graph.edits if e.edit_kind in {"formed", "broken"})),
              "archived_order_changes": sum(e.edit_kind == "order_changed" for e in graph.edits),
              "completed_interval": False, "qualified_complete_path": False}
    try:
        if origin["status"] != "ready":
            record["status"] = origin["status"]
        else:
            prep_folder = "alternate_origin_preparation" if arm == "reverse_origin" else "origin_preparation"
            prep = ORCAContinuationBackend(charge=origin["charge"], multiplicity=origin["multiplicity"], elements=s["elements"], folder=OUT/rid/prep_folder)
            start = np.asarray(s["r_coordinates" if origin["side"] == "R" else "p_coordinates"])
            initial = prep.call(start, [], "free_opt", free=True)
            record["origin_preparation"] = {k: v for k, v in initial.items() if k != "coordinates"}
            if not initial["success"]:
                record["status"] = "ORIGIN_PREPARATION_FAILED:"+str(initial["failure_class"])
            else:
                issues = input_screen(initial["coordinates"], s["elements"], origin["bond_indices"])
                if issues:
                    raise ValueError("PREPARED_ORIGIN_GEOMETRY_INVALID")
                record["origin_preparation_rmsd_angstrom"] = float(np.sqrt(np.mean(np.sum((align(initial["coordinates"], start)-start)**2, axis=1))))
                reference = None
                if arm == "reference_control":
                    refpath = OLD/rid/"PathPlan_v1.json"
                    if not refpath.exists(): refpath = OLD/rid/"PathPlan.json"
                    reference = read(refpath)
                policy = ContinuationPolicy(predictor_enabled=False, adaptive_enabled=False, max_step=.04) if arm == "fixed_control" else ContinuationPolicy()
                phase = None
                if arm in {"phase_early", "phase_late"}:
                    n = sum(e.edit_kind in {"formed", "broken"} for e in graph.edits)
                    phase = ([.4]+[.6]*(n-1)) if arm == "phase_early" else ([.6]+[.4]*(n-1))
                plan = build_plan(s, b, origin, initial["coordinates"], reference_plan=reference, policy=policy, phase_profile=phase)
                write_json(folder/"PathPlan.json", plan)
                backend = ORCAContinuationBackend(charge=plan["charge"], multiplicity=plan["multiplicity"], elements=plan["elements"], folder=folder/"attempts", coordinates=plan["drivers"]+plan["guards"])
                result = run_continuation(plan, initial, backend, lambda doc: write_json(folder/"checkpoint.json", doc))
                record.update(result)
                frames = result["frames"]
                record["n_frames"] = len(frames)
                record["n_attempts"] = len(result["attempts"])
                record["rejection_counts"] = dict(Counter(a["reason"] for a in result["attempts"] if not a["accepted"]))
                record["maximum_rmsd_angstrom"] = max(f["quality"]["rmsd_angstrom"] for f in frames)
                record["maximum_atom_step_angstrom"] = max(f["quality"]["maximum_atom_step_angstrom"] for f in frames)
                record["qualified_complete_path"] = bool(result["completed_interval"] and len(frames)>1 and record["maximum_rmsd_angstrom"]<=.3 and record["maximum_atom_step_angstrom"]<=.6)
                (folder/"trajectory.xyz").write_text("".join(xyz(f["geometry"], s["elements"], f"lambda={f['lambda']:.10f} energy={f['energy_hartree']:.12f}") for f in frames), encoding="utf-8")
                ranking = rank_candidates(result, plan)
                write_json(folder/"ranking.json", ranking)
                for j, candidate in enumerate(ranking["candidates"]):
                    f = frames[candidate["frame_index"]]
                    (folder/f"ts_candidate_{j+1}.xyz").write_text(xyz(f["geometry"], s["elements"], "Unverified constrained-path candidate"), encoding="utf-8")
                record["n_energy_peak_candidates"] = len(ranking["candidates"])
                if result["completed_interval"]:
                    landing = backend.call(np.asarray(frames[-1]["geometry"]), [], "landing_free_opt", free=True)
                    write_json(folder/"landing.json", landing)
                    record["landing_optimization_converged"] = landing["success"]
                    record["landing_minimum_frequency_verified"] = False
                    if landing["success"]:
                        (folder/"landing.xyz").write_text(xyz(landing["coordinates"], s["elements"], "Free optimization; minimum not frequency verified"), encoding="utf-8")
    except Exception as exc:
        record["status"] = "ERROR:"+str(exc)
        record["traceback"] = traceback.format_exc()
    write_json(receipt, record)
    print(rid, arm, record["status"], "frames", record.get("n_frames", 0), "lambda", round(record.get("last_lambda", 0), 5), flush=True)
    return record


def summarize(arm):
    paths = sorted((ROOT/"tests/fixtures/p0_demo24/records").glob("RXN_*.json"))
    records = [read(OUT/p.stem/arm/"result.json") for p in paths if (OUT/p.stem/arm/"result.json").exists()]
    baseline = read(OLD/"completed_batch_summary.json")
    manifest = read(OUT/f"{arm}_manifest.json") if (OUT/f"{arm}_manifest.json").exists() else {"reaction_ids": [p.stem for p in paths]}
    summary = {"arm": arm, "n_requested": len(manifest["reaction_ids"]), "n_results": len(records),
               "status_counts": dict(Counter(r["status"] for r in records)),
               "n_completed_interval": sum(r["completed_interval"] for r in records),
               "n_qualified_complete_path": sum(r["qualified_complete_path"] for r in records),
               "n_frames": sum(r.get("n_frames", 0) for r in records),
               "n_attempts": sum(r.get("n_attempts", 0) for r in records),
               "baseline_original_global_continuity": baseline["n_continuity_ok"],
               "baseline_common_global_and_atom_continuity": sum(r["continuity_ok"] and r["maximum_atom_step_angstrom"]<=.6 for r in baseline["records"]),
               "baseline_comparability": "Changed chemistry scope, prepared origin and target schedule; full R/P boundary preservation is not evaluated by this single-ended run.",
               "records": records}
    write_json(OUT/f"{arm}_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["endpoint", "reference_control", "fixed_control", "reverse_origin", "phase_early", "phase_late"], default="endpoint")
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        paths = sorted((ROOT/"tests/fixtures/p0_demo24/records").glob("RXN_*.json"))
        if args.ids: paths = [p for p in paths if p.stem in args.ids]
        policy = ContinuationPolicy(predictor_enabled=False, adaptive_enabled=False, max_step=.04) if args.arm == "fixed_control" else ContinuationPolicy()
        write_json(OUT/f"{args.arm}_manifest.json", {"reaction_ids": [p.stem for p in paths], "arm": args.arm, "policy": asdict(policy), "reference_geometry_seed": False})
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(run_case, p, args.arm) for p in paths]):
                future.result()
    summary = summarize(args.arm)
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
