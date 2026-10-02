"""Fail-safe energy acceptance and SCC recovery for the diagnosed node."""
from pathlib import Path
import shutil,copy
from run_demo24_synchronized import ROOT,OUT,BATCH,read
from pes2ts_core.generation.planning.synchronized_path import digest,validate_plan
from pes2ts_core.integration.acp.scheduler_backend import ACPWorkbench,save_receipt
acp=Path('/mnt/e/Calculations/Common_Script/Auto_Calc_Platform/ACP_V1_20260811')
p=acp/'src/cccp/qc/interfaces/orca.py';text=p.read_text()
backup=OUT/'orca_before_energy_guard.py'
if not backup.exists():shutil.copyfile(p,backup)
a='                if result.success and result.coordinates is not None:'
b='                if result.success and result.coordinates is not None and result.energy is not None and np.isfinite(result.energy):'
if b not in text:
    if a not in text:raise ValueError('energy acceptance anchor missing')
    text=text.replace(a,b,1)
a='            if result is None or not result.success or result.coordinates is None:'
b='            if result is None or not result.success or result.coordinates is None or result.energy is None or not np.isfinite(result.energy):'
if b not in text:
    if a not in text:raise ValueError('energy failure anchor missing')
    text=text.replace(a,b,1)
p.write_text(text)
rid='RXN_0000110869';folder=OUT/rid;api=ACPWorkbench();receipt=read(OUT/'submission_receipt.json')
jid=receipt['jobs'][rid]['job_id'];job=api.request('/api/v1/jobs/'+jid)
if job['status']!='completed':raise ValueError('only diagnosed terminal attempt may be recovered')
save_receipt(folder/'MissingEnergyAttempt.json',{'job':job,'profile':read(folder/'ACPProfile.json'),'quality':read(folder/'GeometryQuality.json')})
old=read(folder/'PathPlan.json');plan=copy.deepcopy(old);plan['plan_revision']=2;plan['parent_plan_sha256']=digest(old)
plan['solver_parameters']={'xtb_scc_max_iterations':2000,'method_changed':False}
validate_plan(plan)
payload=copy.deepcopy(job['spec']);inp=payload['input'];inp['selection']['path_plan']['xtb_scc_max_iterations']=2000
inp['selection']['pes2ts']['plan_sha256']=digest(plan);inp['scan_request']['selection']=inp['selection']
save_receipt(folder/'PathPlan_v1.json',old);save_receipt(folder/'PathPlan_v2.json',plan);save_receipt(folder/'PathPlan.json',plan)
save_receipt(folder/'ACPJobRequest_v2.json',payload)
payload.update(mode='in_place',request_id=BATCH+'_missing_energy_scc2000_'+rid)
result=api.request('/api/v1/jobs/'+jid+'/edit-recalculate',payload);save_receipt(folder/'MissingEnergyRecovery.json',result)
print(rid,result['status'],flush=True)
