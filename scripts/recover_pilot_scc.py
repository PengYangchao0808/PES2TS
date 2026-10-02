"""Version SCC solver limit while preserving the physical method and endpoints."""
from pathlib import Path
import copy,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from run_demo24_synchronized import OUT,BATCH,read
from pes2ts_core.generation.planning.synchronized_path import digest
from pes2ts_core.integration.acp.scheduler_backend import ACPWorkbench,save_receipt
rid='RXN_0000155302';folder=OUT/rid;receipt=read(OUT/'submission_receipt.json')
api=ACPWorkbench();jid=receipt['jobs'][rid]['job_id'];job=api.request('/api/v1/jobs/'+jid)
if job['status']!='failed':raise ValueError('SCC recovery requires failed pilot')
old=read(folder/'PathPlan.json');plan=copy.deepcopy(old);plan['plan_revision']=2
plan['parent_plan_sha256']=digest(old);plan['solver_parameters']={'xtb_scc_max_iterations':2000,'method_changed':False}
save_receipt(folder/'PathPlan_v1.json',old);save_receipt(folder/'PathPlan_v2.json',plan);save_receipt(folder/'PathPlan.json',plan)
payload=copy.deepcopy(job['spec']);inp=payload['input'];inp['selection']['path_plan']['xtb_scc_max_iterations']=2000
inp['selection']['pes2ts']['plan_sha256']=digest(plan);inp['scan_request']['selection']=inp['selection']
save_receipt(folder/'ACPJobRequest_v2.json',payload)
payload.update(mode='in_place',request_id=BATCH+'_scc2000_'+rid)
result=api.request('/api/v1/jobs/'+jid+'/edit-recalculate',payload);save_receipt(folder/'SCCRecoveryReceipt.json',result)
print(rid,result['status'])
