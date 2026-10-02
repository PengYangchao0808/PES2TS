"""Round-three failure replays and matched real-QC continuation experiments."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, os.environ.get("ACP_SOURCE", "/mnt/e/Calculations/Common_Script/Auto_Calc_Platform/ACP_V1_20260811/src"))
os.environ["PATH"] = "/opt/openmpi418/bin:/opt/orca_6_1_1:"+os.environ.get("PATH", "")
os.environ.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
os.environ.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")

from pes2ts_core.integration.acp.continuation_backend import ORCAContinuationBackend, write_json
from pes2ts_core.integration.acp.gradient_backend import ORCALocalCorrector
from pes2ts_core.generation.planning.candidate_selection import rank_continuation_candidates
from pes2ts_core.generation.planning.connectivity_plan import input_screen
from pes2ts_core.generation.planning.continuation import ContinuationPolicy, geometry_quality, run_continuation, target_values
from pes2ts_core.generation.planning.continuation_diagnostics import segment_diagnostics
from pes2ts_core.generation.planning.graph_rebuild import load_endpoint_materials_from_export, rebuild_endpoint_graphs
from pes2ts_core.generation.planning.local_corrector import LocalCorrectorPolicy
from pes2ts_core.generation.planning.structure_checks import add_online_checks
from pes2ts_core.generation.planning.synchronized_path import digest

OLD = ROOT/"outputs/PES2TS_Demo24_continuation_round2_20261002"
OUT = ROOT/"outputs/PES2TS_Demo24_round3_20261002"
FIXTURES = ROOT/"tests/fixtures/p0_demo24/records"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def material(rid):
    snapshot = read(FIXTURES/f"{rid}.json")
    source = read(OLD/rid/"endpoint/result.json")
    planpath = OLD/rid/"endpoint/PathPlan.json"
    if not planpath.exists():
        return snapshot, source, None
    plan = read(planpath)
    bundle = rebuild_endpoint_graphs(snapshot["reaction_smiles"], load_endpoint_materials_from_export(snapshot))
    return snapshot, source, add_online_checks(plan, snapshot, bundle)


def make_backend(plan, folder, arm):
    if arm in {"local", "local_warm", "local_warm_wide"}:
        return ORCALocalCorrector(plan, folder, policy=local_policy_for_arm(arm), warm_start=arm!="local")
    if arm not in {"original","internal_step","orca_tight"}:
        raise ValueError("UNKNOWN_ROUND3_ARM:"+arm)
    controls = {"trust_radius": -.1, "max_step": .05} if arm == "internal_step" else {}
    return ORCAContinuationBackend(charge=plan["charge"], multiplicity=plan["multiplicity"],
        elements=plan["elements"], folder=folder, coordinates=plan["drivers"]+plan.get("guards", []),
        opt_level="tight" if arm == "orca_tight" else "normal", **controls)


def local_policy_for_arm(arm):
    return LocalCorrectorPolicy(rmsd_radius=.25,atom_radius=.55) if arm=="local_warm_wide" else LocalCorrectorPolicy()


def diagnose(rid):
    snapshot, source, plan = material(rid)
    if plan is None or not source.get("frames"):
        return {"reaction_id": rid, "status": source["status"], "origin": source["origin"]}
    rows = []
    frames = {f["frame_id"]: f for f in source["frames"]}
    rejected = [a for a in source["attempts"] if not a["accepted"]]
    for attempt in rejected[-3:]:
        folder = OLD/rid/"endpoint/attempts"/attempt["attempt_id"]
        backend = read(folder/"result.json")
        request = read(folder/"request.json")
        if backend.get("coordinates") is None:
            continue
        previous = frames[attempt["parent_frame_id"]]
        rows.append({"attempt_id": attempt["attempt_id"], "lambda": attempt["lambda"],
                     "actual_step": attempt["actual_step"], "reason": attempt["reason"],
                     **segment_diagnostics(previous["geometry"], request["geometry"], backend["coordinates"],
                        plan, attempt["targets"], source["origin"]["bond_indices"])})
    row = {"reaction_id": rid, "source_status": source["status"], "n_rejected": len(rejected),
           "source_result_sha256": digest(source), "failed_segments": rows,
           "physical_bifurcation_confirmed": False}
    write_json(OUT/rid/"diagnostics.json", row)
    return row


def replay(rid, arm):
    snapshot, source, plan = material(rid)
    if plan is None or not source.get("frames"):
        return {"reaction_id": rid, "arm": arm, "status": "NO_ACCEPTED_PREFIX"}
    frames = source["frames"]
    last = frames[-1]
    failed = [a for a in source["attempts"] if not a["accepted"]]
    basearm = arm.removesuffix("_zero")
    if arm.endswith("_zero") or not failed:
        # A completed-path control replays an internal point, not its endpoint.
        previous = frames[len(frames)//2] if source["completed_interval"] else last
        guess = previous["geometry"]
        targets = previous["targets"]
        lam = previous["lambda"]
    else:
        attempt = failed[-1]
        previous = next(f for f in frames if f["frame_id"] == attempt["parent_frame_id"])
        request = read(OLD/rid/"endpoint/attempts"/attempt["attempt_id"]/"request.json")
        guess, targets, lam = request["geometry"], attempt["targets"], attempt["lambda"]
    folder = OUT/rid/"replays"/arm
    backend = make_backend(plan, folder/"attempts", basearm)
    started = time.monotonic()
    result = backend(np.asarray(guess), targets, "replay")
    quality = None
    if result.get("coordinates") is not None:
        quality = geometry_quality(np.asarray(previous["geometry"]), np.asarray(result["coordinates"]),
            plan["drivers"]+plan.get("guards", []), targets, ContinuationPolicy())
    row = {"reaction_id": rid, "arm": arm, "lambda": lam, "source_frame_id": previous["frame_id"],
           "source_result_sha256": digest(source), "backend_success": result["success"],
           "failure_class": result.get("failure_class"), "quality": quality,
           "continuous_corrected_frame": bool(result["success"] and quality and quality["reason"] is None),
           "duration_seconds": time.monotonic()-started, "n_gradient_evaluations": result.get("n_gradient_evaluations"),
           "local_quality": result.get("quality"), "physical_bifurcation_confirmed": False}
    write_json(folder/"summary.json", row)
    print(rid, arm, "success", row["continuous_corrected_frame"], "reason", row["failure_class"],
          "seconds", round(row["duration_seconds"], 2), flush=True)
    return row


def run_case(rid, arm):
    # Separate launches may request the same immutable experiment. Serialize
    # them before inspecting caches or writing any request/gradient receipt.
    import fcntl
    directory = OUT/rid/"paths"/arm
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/"execution.lock").open("a") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        return _run_case(rid,arm)


def _run_case(rid, arm):
    snapshot, source, plan = material(rid)
    folder = OUT/rid/"paths"/arm
    receipt = folder/"result.json"
    policy = ContinuationPolicy(require_physical_gradient=True, relative_constraint_tolerance=.2,
        free_gradient_rms_tolerance=.0001, free_gradient_max_tolerance=.0003,
        distance_tolerance=.001, max_seconds=600., max_frames=80, max_attempts=160)
    binding = digest({"source_result_sha256": digest(source), "arm": arm, "policy": asdict(policy),
                      "prepared_plan_sha256": plan["content_sha256"] if plan else None,
                      "local_policy": asdict(local_policy_for_arm(arm)), "round": 3})
    if receipt.exists():
        cached = read(receipt)
        if cached["experiment_request_sha256"] != binding:
            raise ValueError("ROUND3_EXPERIMENT_INPUT_MISMATCH")
        return cached
    record = {"reaction_id": rid, "arm": arm, "round": 3, "experiment_request_sha256": binding,
              "source_round2_result_sha256": digest(source), "completed_interval": False,
              "qualified_complete_path": False, "origin_preparation_reused": True,
              "reference_seed_used": False, "stationary_point_claimed": False, "frames": [], "attempts": []}
    if plan is None:
        record.update(status=source["status"], origin=source["origin"], origin_preparation=source.get("origin_preparation"))
        prep = OLD/rid/"origin_preparation/free_opt/result.json"
        if prep.exists():
            origin_result = read(prep)
            record["origin_geometry_issues"] = input_screen(origin_result["coordinates"], snapshot["elements"], source["origin"]["bond_indices"]) if origin_result.get("coordinates") else []
        write_json(receipt, record)
        return record
    plan["policy"] = asdict(policy)
    plan["round3_source_plan_sha256"] = read(OLD/rid/"endpoint/PathPlan.json")["content_sha256"]
    plan["content_sha256"] = digest({k:v for k,v in plan.items() if k != "content_sha256"})
    write_json(folder/"PathPlan.json", plan)
    initial = {"success": True, "converged": True, "coordinates": source["frames"][0]["geometry"],
               "energy": source["frames"][0]["energy_hartree"], "duration_seconds": 0.}
    backend = make_backend(plan, folder/"attempts", arm)
    started = time.monotonic()
    try:
        result = run_continuation(plan, initial, backend, lambda state: write_json(folder/"checkpoint.json", state))
        record.update(result)
        record["qualified_complete_path"] = result["completed_interval"]
        record["measured_generation_wall_seconds"] = time.monotonic()-started
        record["n_gradient_evaluations"] = sum(a["backend_evidence"].get("n_gradient_evaluations", 0)
            for a in result["attempts"] if "backend_evidence" in a)
        record["n_frames"] = len(result["frames"])
        record["n_attempts"] = len(result["attempts"])
    except (ValueError, RuntimeError, OSError, np.linalg.LinAlgError) as exc:
        record.update(status="ERROR:"+str(exc), traceback=traceback.format_exc())
    write_json(receipt, record)
    ranking = rank_continuation_candidates(record, plan)
    write_json(folder/"ranking_with_gradient.json", ranking)
    xyzrows = []
    for frame in record["frames"]:
        xyzrows.extend([str(len(plan["elements"])), f"lambda={frame['lambda']} E={frame['energy_hartree']}"])
        xyzrows.extend(e+" "+" ".join(f"{v:.12f}" for v in row) for e,row in zip(plan["elements"],frame["geometry"]))
    (folder/"trajectory.xyz").write_text("\n".join(xyzrows)+"\n", encoding="utf-8")
    print(rid, arm, record["status"], "frames", record.get("n_frames",0),
          "lambda", round(record.get("last_lambda",0),4), "seconds", round(record.get("duration_seconds",0),1), flush=True)
    return record


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["diagnose", "replay", "paths"])
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--label", default="strict")
    args = parser.parse_args()
    if Path(args.label).name != args.label or args.label in {".", ".."}:
        parser.error("label must be a single directory name")
    OUT = OUT/args.label
    ids = args.ids or [p.stem for p in sorted(FIXTURES.glob("RXN_*.json"))]
    if args.mode == "diagnose":
        rows = [diagnose(rid) for rid in ids]
        write_json(OUT/"diagnostics_summary.json", {"records": rows, "n_cases": len(rows)})
        return
    arms = args.arms or (["original_zero", "internal_step", "local", "local_zero"] if args.mode == "replay" else ["orca_tight", "local"])
    jobs = [(rid, arm) for rid in ids for arm in arms]
    function = replay if args.mode == "replay" else run_case
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(function, *job): job for job in jobs}
        for future in as_completed(futures):
            rows.append(future.result())
            write_json(OUT/f"{args.mode}_progress.json", {"records": rows, "n_finished": len(rows), "n_expected": len(jobs)})
    write_json(OUT/f"{args.mode}_summary.json", {"records": rows, "n_finished": len(rows), "n_expected": len(jobs)})


if __name__ == "__main__":
    main()
