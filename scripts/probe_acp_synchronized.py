"""Real ORCA evidence for nonlinear B/A/D and >4 simultaneous constraints."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
ACP=Path('/mnt/e/Calculations/Common_Script/Auto_Calc_Platform/ACP_V1_20260811')
sys.path.insert(0,str(ACP/'src'))
from cccp.config import load_config
from cccp.qc.interfaces.orca import ORCAInterface
from cccp.qc.interfaces.constraints import CoordinateSpec,ReactionCoordinatePlan
import importlib.util
module_spec=importlib.util.spec_from_file_location('synchronized_path',ROOT/'pes2ts_core/generation/planning/synchronized_path.py')
module=importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
value,residual=module.value,module.residual

s=json.loads((ROOT/'tests/fixtures/p0_demo24/records/RXN_0000007104.json').read_text())
x=np.asarray(s['r_coordinates']); elements=s['elements']
out=ROOT/'outputs/acp_synchronized_capability_20261002'; out.mkdir(exist_ok=True)
cfg=load_config(); cfg['executables']['orca']['nproc']=2; cfg['resources']['nproc']=2
interface=ORCAInterface(cfg)
rows=[]
for name, definitions in [('mixed_BAD',[('distance',(0,10)),('angle',(0,1,2)),('dihedral',(0,1,2,3))]),
                          ('five_distances',[('distance',p) for p in [(0,10),(2,10),(0,1),(1,2),(2,3)]])]:
    specs=[]
    for i,(kind,atoms) in enumerate(definitions):
        q=value(x,kind,atoms); change=.02 if kind=='distance' else .5
        specs.append(CoordinateSpec(str(i),kind,atoms,start=q,end=q+change,values=(q,q+change*.8,q+change)))
    plan=ReactionCoordinatePlan(tuple(specs),points=3,lambda_values=(0,.2,1))
    result=interface.relaxed_scan(x,elements,plan=plan,charge=0,multiplicity=1,
                                 output_dir=out/name,method='GFN2-xTB',geom_maxiter=250)
    errors=[]
    for point in result.points:
        if point.coordinates is not None:
            errors.append([abs(residual(value(point.coordinates,d.kind,d.atoms),d.values[point.frame_index],d.kind)) for d in specs])
    ok=result.success and len(errors)==3 and all(e<(.01 if d.kind=='distance' else .5) for row in errors for e,d in zip(row,specs))
    rows.append({'probe':name,'success':result.success,'accepted':ok,'message':result.message,'errors':errors,
                 'n_drivers':len(specs),'lambda_values':[0,.2,1], 'targets':[list(d.values) for d in specs]})
    print(json.dumps(rows[-1]),flush=True)
(out/'receipts.json').write_text(json.dumps({'engine':'ORCA 6.1.1','method':'GFN2-xTB','records':rows},indent=2))
if not all(r['accepted'] for r in rows): raise SystemExit(1)
