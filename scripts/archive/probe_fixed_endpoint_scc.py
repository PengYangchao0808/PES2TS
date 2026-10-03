"""Test more SCC iterations without moving the failing immutable product."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,'/mnt/e/Calculations/Common_Script/Auto_Calc_Platform/ACP_V1_20260811/src')
from cccp.config import load_config
from cccp.qc.interfaces.orca import ORCAInterface
s=json.loads((ROOT/'tests/fixtures/p0_demo24/records/RXN_0000155302.json').read_text())
cfg=load_config();cfg['executables']['orca']['nproc']=2;cfg['resources']['nproc']=2
interface=ORCAInterface(cfg); x=np.asarray(s['p_coordinates'])
out=ROOT/'outputs/acp_synchronized_capability_20261002/fixed_endpoint_scc'
result=interface.single_point(x,s['elements'],charge=s['endpoint_electronic']['product']['charge'],
    multiplicity=s['endpoint_electronic']['product']['multiplicity'],output_dir=out,method='GFN2-xTB',basis='',
    extra_blocks=['%xtb\n XTBINPUTSTRING "--iterations 2000"\nend'])
dist=np.linalg.norm(x[:,None]-x[None,:],axis=2);np.fill_diagonal(dist,np.inf)
i,j=np.unravel_index(np.argmin(dist),dist.shape)
record={'success':result.success,'energy_hartree':result.energy,'message':result.error_message,
        'geometry_changed':False,'scc_max_iterations':2000,'minimum_pair_distance_angstrom':float(dist[i,j]),
        'minimum_pair_maps':[s['maps'][i],s['maps'][j]],'minimum_pair_elements':[s['elements'][i],s['elements'][j]]}
(out/'receipt.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
