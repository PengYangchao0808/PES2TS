"""Repair midpoint tables by measuring every coordinate on one common geometry."""
import copy,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from run_demo24_synchronized import OUT,BATCH,PILOTS,read
from pes2ts_core.generation.planning.synchronized_path import value,digest,validate_plan,coordinate_rank
from pes2ts_core.integration.acp.scheduler_backend import ACPWorkbench,save_receipt

api=ACPWorkbench();receipt=read(OUT/'submission_receipt.json')
for rid in PILOTS:
    folder=OUT/rid
    if not (folder/'PathPlan_v2.json').exists() or (folder/'PathPlan_v3.json').exists():continue
    plan=read(folder/'PathPlan_v2.json');job=api.request('/api/v1/jobs/'+receipt['jobs'][rid]['job_id'])
    if job['status'] not in ('failed','completed'):raise ValueError('cannot edit active attempt')
    seed=np.asarray(plan['reference_geometries'])
    for d in plan['drivers']:d['values']=[value(x,d['kind'],d['atoms']) for x in seed]
    # All targets now have an explicit common Cartesian witness.
    plan['plan_revision']=3;plan['parent_plan_sha256']=digest(read(folder/'PathPlan_v2.json'))
    plan['repair_policy']='coherent geometric midpoint targets; rank-increasing scaffold controls; shared grid'
    plan['rank_checks_per_point']=[coordinate_rank(x,plan['drivers']) for x in seed]
    validate_plan(plan)
    coords=[{'kind':d['kind'],'atoms':d['atoms'],'unit':d['unit'],'start':d['values'][0],
             'end':d['values'][-1],'n_points':len(seed),'values':d['values']} for d in plan['drivers']]
    payload=copy.deepcopy(job['spec']);inp=payload['input'];inp['coordinate']=coords[0];inp['coordinates']=coords
    inp['protocol']['coordinate']=coords[0];inp['selection']['pes2ts']['plan_sha256']=digest(plan)
    inp['scan_request'].update(coordinate=coords[0],coordinates=coords,protocol=inp['protocol'],selection=inp['selection'])
    save_receipt(folder/'PathPlan_v3.json',plan);save_receipt(folder/'PathPlan.json',plan)
    save_receipt(folder/'ACPJobRequest_v3.json',payload)
    payload.update(mode='in_place',request_id=BATCH+'_coherent_midpoint_v3_'+rid)
    result=api.request('/api/v1/jobs/'+receipt['jobs'][rid]['job_id']+'/edit-recalculate',payload)
    save_receipt(folder/'CoherentRetargetReceipt.json',result);print(rid,result['status'],flush=True)
