"""Versioned pilot repair: independent scaffold controls and shared-grid refinement."""
from __future__ import annotations
import copy,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pes2ts_core.generation.planning.synchronized_path import align,value,digest,validate_plan,coordinate_rank
from pes2ts_core.integration.acp.scheduler_backend import ACPWorkbench,save_receipt
from run_demo24_synchronized import OUT,BATCH,PILOTS,read

REVISION=int(sys.argv[1]) if len(sys.argv)>1 else 2
api=ACPWorkbench(); receipt=read(OUT/'submission_receipt.json')
for rid in PILOTS:
    folder=OUT/rid; old=read(folder/'PathPlan.json'); job=api.request('/api/v1/jobs/'+receipt['jobs'][rid]['job_id'])
    if job['status']!='completed' or read(folder/'GeometryQuality.json')['qualified']:continue
    if (folder/f'PathPlan_v{REVISION}.json').exists():continue
    seed=np.asarray(old['reference_geometries']); frames=[]
    for p in sorted((folder/'frames').glob('frame_*.xyz')):
        lines=p.read_text().splitlines(); frames.append([[float(v) for v in r.split()[1:4]] for r in lines[2:]])
    frames=np.asarray(frames); controls=[]
    graph=read(ROOT/f'outputs/demo24_acp_geometry_audit_20261001/{rid}_graph.json')
    from rdkit import Chem
    source=read(ROOT/f'tests/fixtures/p0_demo24/records/{rid}.json')
    parser=Chem.SmilesParserParams();parser.removeHs=False
    mols=[Chem.MolFromSmiles(s,parser) for s in source['reaction_smiles'].split('>>')]
    bonds=[]
    for mol in mols:
        bonds.append({tuple(sorted((b.GetBeginAtom().GetAtomMapNum(),b.GetEndAtom().GetAtomMapNum()))) for b in mol.GetBonds()})
    persistent=(bonds[0]|bonds[1]) if REVISION>=5 else (bonds[0]&bonds[1])
    maprows={m:i for i,m in enumerate(old['atom_map_order'])}
    adjacent={m:set() for m in maprows}
    for a,b in persistent:adjacent[a].add(b);adjacent[b].add(a)
    existing={tuple(sorted(d['maps'])) for d in old['drivers']}
    candidates=[('distance',tuple(maprows[m] for m in pair)) for pair in sorted(persistent-existing)]
    if REVISION>=6:
        # Geometric shape links are not chemical bond events. They control
        # fragment placement and avoid singular angles at nearly linear sites.
        maps=old['atom_map_order']
        candidates.extend(('distance',(i,j)) for i in range(len(maps)) for j in range(i+1,len(maps))
                          if tuple(sorted((maps[i],maps[j]))) not in existing|persistent)
    for center,neighbors in adjacent.items():
        for a in sorted(neighbors):
            for b in sorted(neighbors):
                if a<b:candidates.append(('angle',(maprows[a],maprows[center],maprows[b])))
    for a,b in sorted(persistent):
        for left in adjacent[a]-{b}:
            for right in adjacent[b]-{a,left}:
                candidates.append(('dihedral',tuple(maprows[m] for m in (left,a,b,right))))
    for kind,atoms in candidates:
        q=np.asarray([value(f,kind,atoms) for f in seed]); actual=np.asarray([value(f,kind,atoms) for f in frames])
        if not np.isfinite(q).all():continue
        if kind=='angle' and (q.min()<5 or q.max()>175):continue
        scale=.1 if kind=='distance' else 10
        score=float(np.max(np.abs((actual-q+180)%360-180) if kind=='dihedral' else np.abs(actual-q)))/scale
        controls.append((score,{'id':'shape_'+kind+'_'+'_'.join(map(str,atoms)),'kind':kind,'atoms':list(atoms),
                                'maps':[old['atom_map_order'][i] for i in atoms],
                                'unit':'angstrom' if kind=='distance' else 'degree','role':'drive',
                                'purpose':'independent_scaffold_shape_control','values':q.tolist()}))
    plan=copy.deepcopy(old)
    if REVISION>=7:
        plan['drivers']=[d for d in plan['drivers'] if d.get('purpose')!='independent_scaffold_shape_control']
        controls=[row for row in controls if row[1]['kind']=='distance']
    rank=coordinate_rank(seed[len(seed)//2],plan['drivers'])['rank']
    fraction=.95 if REVISION>=7 else (.9 if REVISION>=4 else .7)
    maxrank=min(64,max(rank,int((3*len(seed[0])-6)*fraction)))
    for _,d in sorted(controls,key=lambda row:-row[0]):
        if rank>=maxrank:break
        result=coordinate_rank(seed[len(seed)//2],plan['drivers']+[d])
        if result['rank']>rank:
            plan['drivers'].append(d);rank=result['rank']
    # Split the biggest offending intervals within the ACP point budget.
    quality=read(folder/'GeometryQuality.json'); splits={i for i,s in enumerate(quality['step_rmsd_angstrom']) if s>.3}
    splits=set(sorted(splits,key=lambda i:-quality['step_rmsd_angstrom'][i])[:40-len(seed)])
    grid=[]; newseed=[]
    for i in range(len(seed)):
        grid.append(old['lambda_values'][i]);newseed.append(seed[i])
        if i in splits:
            grid.append((old['lambda_values'][i]+old['lambda_values'][i+1])/2)
            newseed.append((seed[i]+align(seed[i+1],seed[i]))/2)
    for d in plan['drivers']:
        # One common Cartesian witness prevents inconsistent independent interpolation.
        d['values']=[value(x,d['kind'],d['atoms']) for x in newseed]
    plan['lambda_values']=grid;plan['reference_geometries']=np.asarray(newseed).tolist()
    plan['plan_revision']=REVISION;plan['parent_plan_sha256']=digest(old)
    plan['repair_policy']='rank-increasing scaffold controls plus shared midpoint grid; all original events retained'
    # Signed dihedral helper and ORCA must share a convention: covered by live probe.
    validate_plan(plan)
    plan['rank_checks']={k:coordinate_rank(plan[k],plan['drivers']) for k in ('start_geometry','target_geometry')}
    coords=[{'kind':d['kind'],'atoms':d['atoms'],'unit':d['unit'],'start':d['values'][0],
             'end':d['values'][-1],'n_points':len(grid),'values':d['values']} for d in plan['drivers']]
    payload=copy.deepcopy(job['spec']); inp=payload['input']; inp['coordinate']=coords[0];inp['coordinates']=coords
    inp['protocol']['coordinate']=coords[0]
    inp['selection']['pes2ts']['plan_sha256']=digest(plan)
    inp['selection']['path_plan']={'lambda_values':grid,'reference_geometries':plan['reference_geometries'],'fixed_endpoints':True}
    inp['scan_request'].update(coordinate=coords[0],coordinates=coords,protocol=inp['protocol'],selection=inp['selection'])
    original_version=old.get('plan_revision',1)
    if not (folder/f'PathPlan_v{original_version}.json').exists():save_receipt(folder/f'PathPlan_v{original_version}.json',old)
    save_receipt(folder/f'PathPlan_v{REVISION}.json',plan)
    save_receipt(folder/'PathPlan.json',plan);save_receipt(folder/f'ACPJobRequest_v{REVISION}.json',payload)
    payload.update(mode='in_place',request_id=BATCH+f'_shape_grid_repair_v{REVISION}_'+rid)
    result=api.request('/api/v1/jobs/'+receipt['jobs'][rid]['job_id']+'/edit-recalculate',payload)
    save_receipt(folder/f'ShapeRepairReceipt_v{REVISION}.json',result)
    print(rid,len(plan['drivers']),'drivers',len(grid),'points',result['status'],flush=True)
