"""Explicitly authorized reference-guided Demo24 experiment, separate from blind planning."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
from rdkit import Chem

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pes2ts_core.g0.truth.truth_reader import load_ts_geometry, load_irc_frames
from pes2ts_core.g1.truth_alignment import analyze_irc_layout
from pes2ts_core.generation.planning.synchronized_path import align,value,digest,validate_plan,coordinate_rank,audit_path
from pes2ts_core.integration.acp.scheduler_backend import ACPWorkbench,save_receipt

BATCH='PES2TS_Demo24_synchronized_reference_20261002'
OUT=ROOT/'outputs'/BATCH
PILOTS=['RXN_0000007104','RXN_0000017762','RXN_0000047010','RXN_0000109608','RXN_0000155302','RXN_0000187964']
CALLER='user_authorized_demo24_reference_guided_path_20261002'

def read(p): return json.loads(p.read_text(encoding='utf-8'))

def xyztext(x,e,title):
    return '\n'.join([str(len(e)),title]+[a+' '+' '.join(f'{v:.12f}' for v in row) for a,row in zip(e,x)])+'\n'

def teacher(snapshot):
    rid=snapshot['reaction_id']; shard=f'{int(rid.split("_")[1])//1000:05d}'
    mapping=read(ROOT/f'data/interim/g1_truth/p1_mapping/{shard}/{rid}.json')
    ts=load_ts_geometry(rid,True,manifests_dir=ROOT/'data/manifests',caller=CALLER)
    irc=load_irc_frames(rid,True,manifests_dir=ROOT/'data/manifests',caller=CALLER)
    rows={int(a['map']):int(a['ts_irc_index']) for a in mapping['mapping']['map_to_atoms']}
    order=[rows[m] for m in snapshot['maps']]
    if [ts['atomic_numbers'][i] for i in order] != [Chem.GetPeriodicTable().GetAtomicNumber(e) for e in snapshot['elements']]:
        raise ValueError('teacher atom identity mismatch')
    raw=np.asarray(irc['coordinates']); layout=analyze_irc_layout(raw,np.asarray(ts['coordinates']))
    if layout.ts_frame_rmsd>.03 or not layout.branch_b_indices: raise ValueError('IRC has no verified double branch')
    a,b=layout.branch_a_indices,layout.branch_b_indices
    if mapping['irc_validation']['orientation']=='P_first': a,b=b,a
    elif mapping['irc_validation']['orientation']!='R_first': raise ValueError('unresolved teacher direction')
    tx=np.asarray(ts['coordinates'])[order]
    # Preserve the TS explicitly and sample each continuous branch separately.
    near_r=raw[list(reversed(a))][:,order]; near_p=raw[list(b)][:,order]
    chain=[]
    for branch in (near_r,tx[None,:,:],near_p):
        indices=np.unique(np.linspace(0,len(branch)-1,min(9,len(branch))).round().astype(int))
        for x in branch[indices]:
            aligned=align(x,chain[-1] if chain else snapshot['r_coordinates'])
            if not chain or np.linalg.norm(aligned-chain[-1])>1e-8: chain.append(aligned)
    # Bridges are trial seeds, not certified paths. Their relaxed frames must pass gates.
    r=np.asarray(snapshot['r_coordinates']); p=np.asarray(snapshot['p_coordinates'])
    p_aligned=align(p,chain[-1]); seeds=[r]
    for t in np.linspace(0,1,5)[1:]: seeds.append(r*(1-t)+chain[0]*t)
    seeds.extend(chain[1:])
    for t in np.linspace(0,1,5)[1:]: seeds.append(chain[-1]*(1-t)+p_aligned*t)
    # Restore canonical complete end geometry, a common rigid frame change only.
    # Audit aligned adjacency and exact endpoint identity independently.
    seeds[-1]=p.copy()
    return np.asarray(seeds),tx,{'caller':CALLER,'source':'known_TS_IRC','orientation':mapping['irc_validation']['orientation'],
                                'ts_sha256':digest(tx.tolist()),'irc_sha256':digest(raw.tolist()),
                                'endpoint_bridges':'linear trial seeds; require post-optimization validation'}

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'batch_manifest.json').exists(): print('Existing frozen batch retained'); return
    records=[]
    for path in sorted((ROOT/'tests/fixtures/p0_demo24/records').glob('RXN_*.json')):
        s=read(path); rid=s['reaction_id']; x,ts,provenance=teacher(s)
        g=read(ROOT/f'outputs/demo24_acp_geometry_audit_20261001/{rid}_graph.json')
        edits=g['delta_graph']['edits']; rows={m:i for i,m in enumerate(s['maps'])}
        drivers=[]; coverage=[]
        for edit in edits:
            pair=edit['pair']; atoms=[rows[m] for m in pair]; cid='B_'+'_'.join(map(str,pair))
            drivers.append({'id':cid,'kind':'distance','maps':pair,'atoms':atoms,'unit':'angstrom','role':'drive',
                            'values':[value(f,'distance',atoms) for f in x]})
            coverage.append({'maps':pair,'edit_kind':edit['edit_kind'],'driver_ids':[cid]})
        electronic=s['endpoint_electronic']; er,ep=electronic['reactant'],electronic['product']
        if (er['charge'],er['multiplicity'])!=(ep['charge'],ep['multiplicity']): raise ValueError('different endpoint electronic states')
        plan={'schema_version':'pes2ts_synchronized_reference_path_v1','batch_id':BATCH,'reaction_id':rid,
              'logical_primary_id':rid+':primary','parameter_dimension':1,'direction':'R_to_P',
              'atom_map_order':s['maps'],'elements':s['elements'],'start_geometry':s['r_coordinates'],
              'target_geometry':s['p_coordinates'],'charge':er['charge'],'multiplicity':er['multiplicity'],
              'endpoint_hashes':{k:digest(s[k]) for k in ('r_coordinates','p_coordinates')},
              'lambda_values':np.linspace(0,1,len(x)).tolist(),'drivers':drivers,'event_coverage':coverage,
              'schedule_source':'reference_guided','teacher_provenance':provenance,'reference_geometries':x.tolist(),
              'method':'GFN2-xTB','boundary_policy':'fixed_full_geometry_single_point',
              'quality_policy':{'coordinate_tolerances':{'distance':.01,'angle':.5,'dihedral':.5},
                                'max_step_rmsd_angstrom':.3,'threshold_status':'provisional_frozen_before_execution'},
              'blind_evaluation':False,'known_ts_geometry':ts.tolist()}
        validate_plan(plan)
        plan['rank_checks']={k:coordinate_rank(plan[k],drivers) for k in ('start_geometry','target_geometry')}
        ph=digest(plan)
        coords=[{'kind':d['kind'],'atoms':d['atoms'],'unit':d['unit'],'start':d['values'][0],
                 'end':d['values'][-1],'n_points':len(x),'values':d['values']} for d in drivers]
        protocol={'name':'PES2TS_synchronized_reference_v1','scan_type':'bond_length','coordinate':coords[0],
                  'scan_driver':{'software':'orca','mode':'relaxed_scan','full_scan':True,'use_scants':False,
                                 'max_iterations':250,'failure_policy':'abort','retry_count':0},
                  'scan_optimizer':{'method':'GFN2-xTB','convergence':'normal','max_iterations':250,'retry_count':0},
                  'single_point':{'enabled':False,'software':'orca','method':'GFN2-xTB','charge':er['charge'],'multiplicity':er['multiplicity']}}
        payload={'workflow':'PESsearch','name':BATCH+'_'+rid,'molecule_name':rid,'task_name':'Synchronized_reference_PES',
                 'remark':'Reference-guided trial; complete boundaries fixed; qualification requires independent geometry audit.',
                 'execution_mode':'local','resources':{'nproc':2,'mem':'2000MB','batch_id':BATCH,'parallelism':1},
                 'tags':['PES2TS','Demo24','synchronized-reference-guided'],
                 'method':{'mode':'bond_length_scan','method':'GFN2-xTB'},
                 'input':{'source':{'source_type':'xyz_text','xyz_text':xyztext(x[0],s['elements'],rid),
                                   'charge':er['charge'],'multiplicity':er['multiplicity']},
                          'coordinate':coords[0],'coordinates':coords,'protocol':protocol,
                          'selection':{'pes2ts':{'batch_id':BATCH,'reaction_id':rid,'plan_sha256':ph,'primary_count':1,
                                                'parameter_dimension':1,'reference_guided':True},
                                       'path_plan':{'lambda_values':plan['lambda_values'],'reference_geometries':x.tolist(),
                                                    'fixed_endpoints':True}}}}
        folder=OUT/rid; folder.mkdir(exist_ok=True)
        save_receipt(folder/'PathPlan.json',plan); save_receipt(folder/'ACPJobRequest.json',payload)
        save_receipt(folder/'SeedAudit.json',audit_path(plan,x,ts))
        records.append({'reaction_id':rid,'plan_sha256':ph,'request_sha256':digest(payload),'n_points':len(x),'n_drivers':len(drivers)})
        print(rid,len(drivers),'drivers',len(x),'points',flush=True)
    if len(records)!=24: raise ValueError('Demo24 identity count mismatch')
    save_receipt(OUT/'batch_manifest.json',{'batch_id':BATCH,'records':records,'n_primary':24,'parameter_dimension':1})

def submit(pilot=False, remaining_experiment=False):
    api=ACPWorkbench()
    if not pilot and not remaining_experiment:
        if not (OUT/'quality_summary.json').exists():raise ValueError('Pilot quality audit required before formal expansion')
        checked={r['reaction_id']:r for r in read(OUT/'quality_summary.json')['records']}
        failed=[rid for rid in PILOTS if not checked.get(rid,{}).get('qualified')]
        if failed:raise ValueError('Formal expansion gated by unqualified pilots: '+', '.join(failed))
    if (OUT/'batch_manifest.json').exists():
        manifest=read(OUT/'batch_manifest.json')
    elif pilot:
        manifest={'records':[{'reaction_id':rid,'plan_sha256':digest(read(OUT/rid/'PathPlan.json')),
                            'request_sha256':digest(read(OUT/rid/'ACPJobRequest.json'))}
                            for rid in PILOTS if (OUT/rid/'ACPJobRequest.json').exists()]}
    else:
        raise ValueError('Full frozen manifest required for whole-batch submission')
    project=api.ensure_project('PES2TS Demo24 — synchronized reference PES','One shared lambda, all edited bonds, fixed full endpoints. Reference-guided trials; audited qualification separate.')
    receipt=read(OUT/'submission_receipt.json') if (OUT/'submission_receipt.json').exists() else {'project_id':project,'jobs':{}}
    if remaining_experiment:
        save_receipt(OUT/'remaining_experiment_authorization.json',{
            'source':'explicit_user_instruction_20261002_complete_remaining_calculations',
            'scope':'remaining_18_frozen_reference_guided_calculations',
            'formal_quality_gate_passed':False,'qualification_requires_postrun_audit':True})
    for row in manifest['records']:
        rid=row['reaction_id']
        if pilot and rid not in PILOTS: continue
        if remaining_experiment and rid in PILOTS: continue
        payload=read(OUT/rid/'ACPJobRequest.json')
        if digest(payload)!=row['request_sha256']: raise ValueError('frozen request changed')
        payload['project_id']=project; result=api.submit_once(payload)
        receipt['jobs'][rid]={'job_id':result.get('job_id') or result.get('id'),'plan_sha256':row['plan_sha256']}
        save_receipt(OUT/'submission_receipt.json',receipt); print(rid,result,flush=True)

def status():
    api=ACPWorkbench(); receipt=read(OUT/'submission_receipt.json'); records=[]
    for rid,entry in receipt['jobs'].items():
        job=api.request('/api/v1/jobs/'+entry['job_id']); job=job.get('job',job)
        save_receipt(OUT/rid/'ACPJobStatus.json',job)
        row={'reaction_id':rid,'job_id':entry['job_id'],'status':job['status'],'error':job.get('error_message') or job.get('error')}
        if job['status']=='completed':
            profile=api.request('/api/v1/jobs/'+entry['job_id']+'/s2/profile')
            save_receipt(OUT/rid/'ACPProfile.json',profile); row['n_frames']=len(profile.get('frames',[]))
        records.append(row)
    save_receipt(OUT/'batch_status.json',{'records':records,'status_counts':dict(Counter(r['status'] for r in records))})
    print(json.dumps(records,ensure_ascii=False,indent=2))

def retry_failed():
    api=ACPWorkbench(); receipt=read(OUT/'submission_receipt.json')
    for rid,entry in receipt['jobs'].items():
        job=api.request('/api/v1/jobs/'+entry['job_id'])
        if job['status']!='failed': continue
        logs=api.request('/api/v1/jobs/'+entry['job_id']+'/logs?lines=100')
        save_receipt(OUT/rid/'FailedAttemptBeforeContractFix.json',{'job':job,'logs':logs})
        payload=dict(job['spec']); payload.update(mode='in_place',request_id=BATCH+'_generic_multicoordinate_selector_fix_'+rid)
        result=api.request('/api/v1/jobs/'+entry['job_id']+'/edit-recalculate',payload)
        save_receipt(OUT/rid/'ContractFixRecalculation.json',result)
        print(rid,result,flush=True)

def audit():
    api=ACPWorkbench(); receipt=read(OUT/'submission_receipt.json'); records=[]
    for rid,entry in receipt['jobs'].items():
        base='/api/v1/jobs/'+entry['job_id']; job=api.request(base)
        if job['status']!='completed':
            records.append({'reaction_id':rid,'status':job['status'],'qualified':False}); continue
        plan=read(OUT/rid/'PathPlan.json'); profile=api.request(base+'/s2/profile')
        graph=api.request(base+'/energy-graph'); save_receipt(OUT/rid/'ACPProfile.json',profile)
        frames=[]; folder=OUT/rid/'frames'; folder.mkdir(exist_ok=True)
        for i in range(len(profile['frames'])):
            data=api.request(base+'/s2/frame/'+str(i)); lines=data['xyz'].splitlines()
            rows=[line.split() for line in lines[2:2+int(lines[0])]]
            if [r[0] for r in rows]!=plan['elements']:raise ValueError('frame atom identity changed')
            frames.append([[float(v) for v in r[1:4]] for r in rows])
            (folder/f'frame_{i:03d}.xyz').write_text(data['xyz'],encoding='utf-8')
        result=audit_path(plan,frames,np.asarray(plan['known_ts_geometry']))
        result.update(reaction_id=rid,status='completed',job_id=entry['job_id'],n_frames=len(frames),
                      finite_energy_ok=all(f.get('scan_energy_hartree') is not None and np.isfinite(f['scan_energy_hartree']) for f in profile['frames']),
                      frontend_nodes=len(graph['nodes']),frame_roles=[f.get('frame_role','missing') for f in profile['frames']])
        result['qualified'] &= result['finite_energy_ok'] and len(graph['nodes'])==len(frames)
        if (OUT/'endpoint_bond_audit.json').exists():
            endpoint_record=next(r for r in read(OUT/'endpoint_bond_audit.json')['records'] if r['reaction_id']==rid)
            result['endpoint_geometry_ok']=endpoint_record['both_complete_endpoints_geometry_ok']
            result['qualified'] &= result['endpoint_geometry_ok']
        save_receipt(OUT/rid/'GeometryQuality.json',result); records.append(result)
        save_receipt(OUT/rid/f'GeometryQuality_v{plan.get("plan_revision",1)}.json',result)
        print(rid,'boundary',result['boundary_ok'],'continuity',result['continuity_ok'],
              'residual',result['max_normalized_constraint_error'],'TS',result['ts_near_ok'],flush=True)
    save_receipt(OUT/'quality_summary.json',{'records':records,'n_qualified':sum(r['qualified'] for r in records),
                                           'project_id':receipt['project_id']})

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['prepare','pilot','submit','remaining-experiment','status','retry-failed','audit'])
    args=parser.parse_args()
    if args.action=='prepare': prepare()
    elif args.action=='status': status()
    elif args.action=='retry-failed': retry_failed()
    elif args.action=='audit': audit()
    elif args.action=='remaining-experiment': submit(remaining_experiment=True)
    else: submit(args.action=='pilot')
