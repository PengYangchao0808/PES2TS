"""One evidence-selected local guard, followed by mandatory physical release."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import copy
import numpy as np
from run_demo24_round3 import OLD,OUT,material,read,write_json
from pes2ts_core.integration.acp.gradient_backend import ORCALocalCorrector
from pes2ts_core.generation.planning.continuation import ContinuationPolicy,geometry_quality,project_geometry,validate_plan
from pes2ts_core.generation.planning.continuation_diagnostics import segment_diagnostics
from pes2ts_core.generation.planning.synchronized_path import digest,value


def run(rid):
    _,source,plan=material(rid)
    attempt=[a for a in source["attempts"] if not a["accepted"]][-1]
    previous=next(f for f in source["frames"] if f["frame_id"]==attempt["parent_frame_id"])
    source_folder=OLD/rid/"endpoint/attempts"/attempt["attempt_id"]
    guess=np.asarray(read(source_folder/"request.json")["geometry"])
    output=read(source_folder/"result.json")["coordinates"]
    evidence=segment_diagnostics(previous["geometry"],guess,output,plan,attempt["targets"],source["origin"]["bond_indices"])
    selected=None
    support=set(plan["local_support_maps"])
    centres={m for d in plan["drivers"] for m in d["maps"]}
    for coordinate in evidence["local_coordinate_changes"]:
        if not set(coordinate["maps"])<=support or not set(coordinate["maps"])&centres:
            continue
        target=value(previous["geometry"],coordinate["kind"],coordinate["atoms"])
        guard={"id":"diagnostic_local_guard","kind":coordinate["kind"],"maps":coordinate["maps"],
               "atoms":coordinate["atoms"],"values":[target,target],"lambda_values":[0.,1.],
               "source":"largest_observed_rejected_coordinate_motion","release_required":True}
        candidate=copy.deepcopy(plan)
        candidate["start_geometry"]=previous["geometry"]
        candidate["guards"]=[guard]
        try:
            validate_plan(candidate)
        except ValueError:
            continue
        selected=candidate,guard
        break
    folder=OUT/"guard_ablation"/rid
    if selected is None:
        row={"reaction_id":rid,"status":"NO_INDEPENDENT_LOCAL_GUARD"}
    else:
        guarded,guard=selected
        guarded["content_sha256"]=digest({k:v for k,v in guarded.items() if k!="content_sha256"})
        write_json(folder/"guarded_plan.json",guarded)
        targets=attempt["targets"]+[guard["values"][0]]
        seed,projection=project_geometry(guess,guarded["drivers"]+guarded["guards"],targets,guarded["masses"],ContinuationPolicy())
        if seed is None:
            row={"reaction_id":rid,"status":"GUARD_PROJECTION_FAILED","projection":projection}
        else:
            first=ORCALocalCorrector(guarded,folder/"guarded",warm_start=True)(seed,targets,"corrector")
            released=None
            if first["success"]:
                released=ORCALocalCorrector(plan,folder/"released",warm_start=True)(
                    np.asarray(first["coordinates"]),attempt["targets"],"corrector")
            quality=geometry_quality(np.asarray(previous["geometry"]),np.asarray(released["coordinates"]),
                plan["drivers"],attempt["targets"],ContinuationPolicy()) if released and released.get("coordinates") else None
            success=bool(released and released["success"] and quality and quality["reason"] is None)
            row={"reaction_id":rid,"guard":guard,"guarded_success":first["success"],
                 "guarded_failure":first.get("failure_class"),"released_success":bool(released and released["success"]),
                 "released_failure":released.get("failure_class") if released else "NOT_RUN",
                 "quality_after_release":quality,"physical_continuous_frame":success,
                 "status":"RELEASED_CONTINUOUS_FRAME" if success else "GUARD_DID_NOT_PRODUCE_RELEASED_CONTINUOUS_FRAME",
                 "reference_geometry_used":False,"physical_bifurcation_confirmed":False}
    write_json(folder/"summary.json",row)
    print(rid,row["status"],flush=True)
    return row


if __name__=="__main__":
    with ThreadPoolExecutor(max_workers=2) as pool:
        records=list(pool.map(run,["RXN_0000017762","RXN_0000077619","RXN_0000161724"]))
    write_json(OUT/"guard_ablation/summary.json",{"records":records})
