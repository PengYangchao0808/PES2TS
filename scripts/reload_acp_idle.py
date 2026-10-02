"""Reload the existing idle local ACP service, keeping its exact launch environment."""
import json,os,signal,subprocess,time,urllib.request,sys
from pathlib import Path
with urllib.request.urlopen('http://127.0.0.1:8765/api/v1/jobs?limit=1000') as response:
    jobs=json.load(response)['jobs']
active=[j['job_id'] for j in jobs if j['status'] in ('running','queued','paused')]
if active: raise SystemExit('Refusing service reload: active jobs exist')
pid=int(sys.argv[1]) if len(sys.argv)>1 else 141310
proc=Path('/proc')/str(pid)
cmd=[v.decode() for v in (proc/'cmdline').read_bytes().split(b'\0') if v]
if 'acp.api.server:app' not in cmd: raise SystemExit('Unexpected service process')
env=dict(v.decode().split('=',1) for v in (proc/'environ').read_bytes().split(b'\0') if v)
cwd=os.readlink(proc/'cwd')
os.kill(pid,signal.SIGTERM)
for _ in range(100):
    if not proc.exists():break
    time.sleep(.1)
else:raise SystemExit('Existing server did not stop cleanly')
log=Path('/mnt/e/Calculations/AI4S_ML_Studys/PES2TS/outputs/acp_synchronized_patch_20261002/server.log')
with log.open('ab') as output:
    child=subprocess.Popen(cmd,cwd=cwd,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
print('ACP reloaded; PID',child.pid)
