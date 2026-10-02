"""Recover the failed pilot using all required edits and proven SCC settings."""
import copy
from run_demo24_synchronized import OUT,BATCH,read
from pes2ts_core.generation.planning.synchronized_path import digest,validate_plan
from pes2ts_core.integration.acp.scheduler_backend import ACPWorkbench,save_receipt
rid='RXN_0000155302';folder=OUT/rid;api=ACPWorkbench();receipt=read(OUT/'submission_receipt.json')
jid=receipt['jobs'][rid]['job_id'];job=api.request('/api/v1/jobs/'+jid)
if job['status']!='failed':raise ValueError('recovery requires failed terminal task')
save_receipt(folder/'FailedShapeV5BeforeRecovery.json',{'job':job,'logs':api.request('/api/v1/jobs/'+jid+'/logs?lines=120')})
current=read(folder/'PathPlan.json');plan=copy.deepcopy(read(folder/'PathPlan_v2.json'))
plan['plan_revision']=8;plan['parent_plan_sha256']=digest(current)
plan['repair_policy']='retain all required edit coordinates; remove optional failed shape controls; proven SCC limit 2000; immutable original boundaries'
validate_plan(plan)
payload=copy.deepcopy(read(folder/'ACPJobRequest_v2.json'))
payload['input']['selection']['pes2ts']['plan_sha256']=digest(plan)
payload['input']['scan_request']['selection']=payload['input']['selection']
save_receipt(folder/'PathPlan_v8.json',plan);save_receipt(folder/'PathPlan.json',plan)
save_receipt(folder/'ACPJobRequest_v8.json',payload)
payload.update(mode='in_place',request_id=BATCH+'_required_only_scc_recovery_v8_'+rid)
result=api.request('/api/v1/jobs/'+jid+'/edit-recalculate',payload)
save_receipt(folder/'RequiredOnlyRecovery_v8.json',result);print(rid,result['status'])
