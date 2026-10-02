"""Wait for the existing batch; never submit jobs or change scientific status."""
from collections import Counter
import json,time
from run_demo24_synchronized import OUT,read
from pes2ts_core.integration.acp.scheduler_backend import ACPWorkbench,save_receipt
api=ACPWorkbench();receipt=read(OUT/'submission_receipt.json');previous=None
for _ in range(180):
    records=[]
    for rid,e in receipt['jobs'].items():
        j=api.request('/api/v1/jobs/'+e['job_id'])
        records.append({'reaction_id':rid,'job_id':e['job_id'],'status':j['status'],'error':j.get('error')})
    counts=dict(Counter(r['status'] for r in records))
    save_receipt(OUT/'batch_status.json',{'records':records,'status_counts':counts})
    if counts!=previous:print(json.dumps(counts),flush=True);previous=counts
    if all(r['status'] in ('completed','failed','cancelled') for r in records):break
    time.sleep(20)
else:raise SystemExit('Batch still active; inspect saved status')
