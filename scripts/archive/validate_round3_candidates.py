"""Validate gradient-ranked complete/partial seeds, with separate stage budgets."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_demo24_round3 import ROOT,OLD,OUT,read,write_json,material
from seed_validation import ValidationBudgets,validate_seed
from pes2ts_core.generation.planning.candidate_selection import rank_continuation_candidates
from pes2ts_core.generation.planning.synchronized_path import digest


def validate(rid, *, source="round2", arm="local_warm", method="HF-3c", rank=1, calibration=False,
             label="final", irc_seconds=1800.):
    snapshot, old, original_plan = material(rid)
    if calibration:
        reference = read(ROOT/"outputs/PES2TS_Demo24_synchronized_reference_20261002"/rid/"PathPlan.json")
        plan = {**original_plan,"reference_seed_used":True}
        plan["content_sha256"]=digest({k:v for k,v in plan.items() if k!="content_sha256"})
        result={"completed_interval":False,"frames":[{"frame_id":"reference-calibration","geometry":reference["known_ts_geometry"]}]}
        candidate={"frame_index":0,"selection_source":"reference_guided_validation_diagnostic"}
    elif source=="round2":
        plan=read(OLD/rid/"endpoint/PathPlan.json")
        result=old
        evidence=read(OLD/rid/"endpoint/physical_evidence.json")
        ranking=rank_continuation_candidates(result,plan,physical_evidence=evidence)
        write_json(OUT/label/rid/"round2_seed_proposals.json",ranking)
        if len(ranking["candidates"]) < rank:
            return {"reaction_id":rid,"status":"NO_BOUND_CANDIDATE"}
        candidate=ranking["candidates"][rank-1]
    else:
        folder=OUT/label/rid/"paths"/arm
        result,plan=read(folder/"result.json"),read(folder/"PathPlan.json")
        ranking=read(folder/"ranking_with_gradient.json")
        if len(ranking["candidates"]) < rank:
            return {"reaction_id":rid,"status":"NO_BOUND_CANDIDATE"}
        candidate=ranking["candidates"][rank-1]
    budgets=ValidationBudgets(irc_seconds=irc_seconds)
    identity=digest({"candidate":candidate,"method":method,"calibration":calibration,"budgets":asdict(budgets)})[:16]
    folder=OUT/label/rid/"validation"/("calibration" if calibration else source)/(method.replace("/","_"))/identity
    answer=validate_seed(plan,snapshot,result,candidate,folder,method=method,budgets=budgets)
    print(rid,source,method,"calibration",calibration,answer["status"],flush=True)
    return answer


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("ids",nargs="+")
    parser.add_argument("--source",choices=["round2","round3"],default="round2")
    parser.add_argument("--arm",default="local_warm")
    parser.add_argument("--method",choices=["HF-3c","GFN2-xTB"],default="HF-3c")
    parser.add_argument("--rank",type=int,default=1)
    parser.add_argument("--calibration",action="store_true")
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--label",default="final")
    parser.add_argument("--irc-seconds",type=float,default=1800.)
    args=parser.parse_args()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(lambda rid:validate(rid,source=args.source,arm=args.arm,method=args.method,
            rank=args.rank,calibration=args.calibration,label=args.label,irc_seconds=args.irc_seconds),args.ids))
    tag="calibration" if args.calibration else f"{args.source}_{args.method}_{args.arm}_rank{args.rank}"
    write_json(OUT/args.label/f"validation_summary_{tag}.json",{"records":rows,"reference_seed_used":args.calibration})


if __name__=="__main__":
    main()
