"""Recalculate the two adapter-affected jobs without changing frozen plans."""
from run_demo24_synchronized import OUT,BATCH,read
from pes2ts_core.integration.acp.scheduler_backend import ACPWorkbench,save_receipt
api=ACPWorkbench();receipt=read(OUT/'submission_receipt.json')
for rid in ('RXN_0000026256','RXN_0000077619'):
    entry=receipt['jobs'][rid];jid=entry['job_id'];job=api.request('/api/v1/jobs/'+jid)
    if job['status'] not in ('completed','failed'):raise ValueError('affected job still active')
    folder=OUT/rid
    save_receipt(folder/'IncorrectNativeAdapterAttempt.json',{'job':job,'profile':api.request('/api/v1/jobs/'+jid+'/s2/profile')})
    if (folder/'GeometryQuality.json').exists():save_receipt(folder/'IncorrectNativeAdapterQuality.json',read(folder/'GeometryQuality.json'))
    payload=dict(job['spec']);payload.update(mode='in_place',request_id=BATCH+'_single_explicit_adapter_fix_'+rid)
    result=api.request('/api/v1/jobs/'+jid+'/edit-recalculate',payload)
    save_receipt(folder/'ExplicitAdapterRecovery.json',result);print(rid,result['status'],flush=True)
