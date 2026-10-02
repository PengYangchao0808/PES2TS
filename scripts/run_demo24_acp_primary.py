"""Freeze, submit and inspect one ACP PESsearch primary task per Demo24 reaction.

This explicitly authorized development batch uses the prior truth-assisted
anchor audit, endpoint starting structures, and endpoint-only range rules.
It does not forge dual-review decisions or feed TS geometries to ACP.
"""
from __future__ import annotations

import argparse
import math
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from pes2ts_core.integration.acp.scheduler_backend import ACPWorkbench, save_receipt
from pes2ts_core.generation.planning.coordinate_pool import EndpointMaterials
from pes2ts_core.generation.planning.graph_rebuild import load_endpoint_materials_from_export,rebuild_endpoint_graphs
from pes2ts_core.generation.planning.primary import primary_stretch

BATCH='PES2TS_Demo24_primary_20261001'
OUT=ROOT/'outputs'/BATCH
AUDIT=ROOT/'outputs/demo24_truth_algorithm_audit_20261001'
FIX=ROOT/'tests/fixtures/p0_demo24/records'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    frozen=OUT/'batch_manifest.json'
    if frozen.exists():
        print('Frozen batch already exists; use status/submit to resume.',flush=True)
        return
    source=AUDIT/'one_primary_per_reaction_diagnostic.json'
    anchors=json.loads(source.read_text(encoding='utf-8'))['records']
    assert len(anchors)==len({a['reaction_id'] for a in anchors})==24
    rows=[]
    for row in anchors:
        rid=row['reaction_id']
        path=FIX/f'{rid}.json'
        s=json.loads(path.read_text(encoding='utf-8'))
        b=rebuild_endpoint_graphs(s['reaction_smiles'],load_endpoint_materials_from_export(s))
        xyz={key:{m:s[key][i] for i,m in enumerate(s['maps'])} for key in ('r_coordinates','p_coordinates')}
        plan=primary_stretch(b,EndpointMaterials(**xyz),anchor=row['selection'])
        if not plan['primary']:
            raise RuntimeError(f'{rid}: {plan["reason"]}')
        p=plan['primary']
        # Verification against the already-consulted truth; no truth targets
        # or geometries are injected into the endpoint-only range calculation.
        ts_length=row['selection']['q_TS']
        if not p['start']<ts_length<p['end']:
            raise RuntimeError(f'{rid}: primary endpoint-only range misses audited TS')
        electronic=s['endpoint_electronic']['reactant' if p['start_endpoint']=='R' else 'product']
        charge=int(electronic['charge']); mult=int(electronic['multiplicity'])
        atom_rows={m:i for i,m in enumerate(s['maps'])}
        coordinate={'kind':'distance','atoms':[atom_rows[m] for m in p['maps']],
                    'unit':'angstrom','start':p['start'],'end':p['end'],'n_points':p['n_points']}
        xyztext='\n'.join([str(len(s['maps'])),f'{BATCH} {rid} endpoint={p["start_endpoint"]} charge={charge} multiplicity={mult}']+
                          [s['elements'][i]+' '+' '.join(f'{x:.10f}' for x in p['geometry'][str(m)]) for i,m in enumerate(s['maps'])])+'\n'
        auth={'scope':'first24_development_PES_calculations','source':'explicit_user_instruction_in_chat',
              'dual_chemistry_review_complete':False,'truth_assisted_development':True,'blind_evaluation':False}
        frozen_plan={'schema_version':'pes2ts_primary_pes_plan_v1','batch_id':BATCH,'reaction_id':rid,
                     'split':s['split'],'primary':p,'authorization':auth,
                     'source_snapshot_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                     'anchor_audit_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                     'maps':s['maps'],'elements':s['elements'],'charge':charge,'multiplicity':mult}
        planhash=digest(frozen_plan)
        protocol={'name':'PES2TS_single_primary_GFN2xTB_v1','scan_type':'bond_length','coordinate':coordinate,
                  'scan_driver':{'software':'orca','mode':'relaxed_scan','reuse_previous_geometry':True,
                                 'full_scan':True,'use_scants':False,'max_iterations':250,
                                 'failure_policy':'abort','retry_count':0},
                  'scan_optimizer':{'method':'GFN2-xTB','convergence':'normal','max_iterations':250,'retry_count':0},
                  'single_point':{'enabled':False,'software':'orca','method':'B97-3c','charge':charge,'multiplicity':mult}}
        payload={'workflow':'PESsearch','name':BATCH+'_'+rid,'molecule_name':rid,
                 'task_name':f'Primary_PES_{p["start_endpoint"]}_B{p["maps"][0]}-{p["maps"][1]}',
                 'remark':'PES2TS truth-assisted development batch; one primary stretch; chemistry dual-review pending; no TS claim.',
                 'execution_mode':'local','resources':{'nproc':2,'mem':'2000MB','batch_id':BATCH,'parallelism':1},
                 'tags':['PES2TS','Demo24','primary-only','truth-assisted-development',s['split']],
                 'method':{'mode':'bond_length_scan','method':'GFN2-xTB'},
                 'input':{'source':{'source_type':'xyz_text','xyz_text':xyztext,'charge':charge,'multiplicity':mult},
                          'coordinate':coordinate,'protocol':protocol,
                          'selection':{'pes2ts':{'batch_id':BATCH,'reaction_id':rid,'plan_sha256':planhash,
                                                'primary_count':1,'authorization':auth}}}}
        folder=OUT/rid
        folder.mkdir(exist_ok=True)
        save_receipt(folder/'PrimaryPlan.json',frozen_plan)
        save_receipt(folder/'ACPJobRequest.json',payload)
        (folder/'start.xyz').write_text(xyztext,encoding='utf-8')
        rows.append({'reaction_id':rid,'plan_sha256':planhash,'request_sha256':digest(payload),'request':str(folder/'ACPJobRequest.json'),
                     'n_points':p['n_points'],'primary_maps':p['maps'],'start_endpoint':p['start_endpoint']})
    save_receipt(frozen,{'schema_version':'pes2ts_acp_primary_batch_v1','batch_id':BATCH,'n_reactions':24,'n_primary':24,
                         'max_primary_attempts_per_reaction':1,'truth_assisted_development':True,'records':rows})
    print(f'Frozen 24 unique primary plans, {sum(r["n_points"] for r in rows)} scan points.',flush=True)


def submit(limit):
    api=ACPWorkbench()
    health=api.request('/api/status')
    if health['status']!='ok':
        raise RuntimeError('ACP unavailable')
    project=api.ensure_project('PES2TS Demo24 — primary PES','24 unique primary stretches; explicitly authorized truth-assisted development batch.')
    manifest=json.loads((OUT/'batch_manifest.json').read_text(encoding='utf-8'))
    receiptpath=OUT/'submission_receipt.json'
    receipt=json.loads(receiptpath.read_text(encoding='utf-8')) if receiptpath.exists() else {'batch_id':BATCH,'project_id':project,'jobs':{}}
    for r in manifest['records'][:limit]:
        payload=json.loads(Path(r['request']).read_text(encoding='utf-8'))
        if digest(payload)!=r['request_sha256']:
            raise RuntimeError('ACP request changed after freeze')
        payload['project_id']=project
        result=api.submit_once(payload)
        job_id=result.get('job_id') or result.get('id')
        if not job_id:
            raise RuntimeError(f'unrecognized ACP job receipt: {result}')
        receipt['jobs'][r['reaction_id']]={'job_id':job_id,'plan_sha256':r['plan_sha256']}
        save_receipt(receiptpath,receipt)
        print(r['reaction_id'],job_id,'recovered' if result.get('recovered') else 'submitted',flush=True)


def status():
    api=ACPWorkbench()
    receipt=json.loads((OUT/'submission_receipt.json').read_text(encoding='utf-8'))
    rows=[]
    for rid,item in receipt['jobs'].items():
        result=api.request('/api/v1/jobs/'+item['job_id'])
        job=result.get('job',result)
        folder=OUT/rid
        save_receipt(folder/'ACPJobStatus.json',job)
        row={'reaction_id':rid,'job_id':item['job_id'],'status':job['status'],
             'current_stage':job.get('current_stage'),'error':job.get('error_message') or job.get('error')}
        if job['status']=='completed':
            profile=api.request('/api/v1/jobs/'+item['job_id']+'/s2/profile')
            save_receipt(folder/'ACPProfile.json',profile)
            row['n_frames']=len(profile.get('frames',[]))
            row['finite_scan_energies']=sum(f.get('scan_energy_hartree') is not None for f in profile.get('frames',[]))
        rows.append(row)
        print(json.dumps(row,ensure_ascii=False),flush=True)
    summary={'batch_id':BATCH,'project_id':receipt['project_id'],'n_submitted':len(rows),
             'status_counts':dict(Counter(r['status'] for r in rows)),'records':rows}
    save_receipt(OUT/'batch_status.json',summary)
    print(json.dumps(summary['status_counts']),flush=True)


def repair_resources():
    """Version the resource correction and rerun the same failed pilot identity."""
    api=ACPWorkbench()
    manifest_path=OUT/'batch_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('resource_revision') == 2:
        print('Resource correction already recorded; inspect status before further action.')
        return
    receipt=json.loads((OUT/'submission_receipt.json').read_text(encoding='utf-8'))
    if len(receipt['jobs']) != 1:
        raise RuntimeError('Resource repair expects exactly one failed pilot')
    rid,item=next(iter(receipt['jobs'].items()))
    job=api.request('/api/v1/jobs/'+item['job_id'])
    job=job.get('job',job)
    if job['status'] != 'failed':
        raise RuntimeError('Pilot is not failed; do not restart active work')
    save_receipt(OUT/'batch_manifest_resource_v1.json',manifest)
    save_receipt(OUT/rid/'ACPFailedAttempt1.json',job)
    save_receipt(OUT/rid/'ACPFailedAttempt1Logs.json',api.request('/api/v1/jobs/'+item['job_id']+'/logs?tail=100'))
    for row in manifest['records']:
        path=Path(row['request'])
        payload=json.loads(path.read_text(encoding='utf-8'))
        if digest(payload) != row['request_sha256']:
            raise RuntimeError('Frozen request changed before resource repair')
        save_receipt(path.with_name('ACPJobRequest_resource_v1.json'),payload)
        payload['resources']['mem']='2000MB'
        save_receipt(path,payload)
        row['request_sha256']=digest(payload)
    manifest['resource_revision']=2
    manifest['repair_note']='Explicit MB units; ACP ORCA helper thread pinning; failed pilot preserved and recalculated in place, no new primary candidate.'
    save_receipt(manifest_path,manifest)
    # ACP materializes scan_request and source snapshots at submission; preserve
    # that canonical input for an in-place resource-only recalculation.
    payload=dict(job['spec'])
    payload['resources']=dict(payload['resources'],mem='2000MB')
    payload.update(mode='in_place',request_id=BATCH+'_resource_repair_v2',project_id=receipt['project_id'])
    result=api.request('/api/v1/jobs/'+item['job_id']+'/edit-recalculate',payload)
    save_receipt(OUT/rid/'ACPResourceRepairReceipt.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


def resume_capacity():
    """Requeue cancelled capacity waiters with a persisted cohort slot limit."""
    api=ACPWorkbench()
    receipt=json.loads((OUT/'submission_receipt.json').read_text(encoding='utf-8'))
    for rid,item in receipt['jobs'].items():
        job=api.request('/api/v1/jobs/'+item['job_id'])
        job=job.get('job',job)
        if job['status'] != 'cancelled':
            continue
        payload=dict(job['spec'])
        payload['resources']=dict(payload['resources'],batch_id=BATCH,parallelism=1)
        payload.update(mode='in_place',request_id=BATCH+'_capacity_repair_'+rid)
        result=api.request('/api/v1/jobs/'+item['job_id']+'/edit-recalculate',payload)
        save_receipt(OUT/rid/'ACPCapacityRecoveryReceipt.json',result)
        print(rid,result['status'],flush=True)


def validate_results():
    """Check real ACP frames and the frontend's energy/geometry contracts."""
    api=ACPWorkbench()
    receipt=json.loads((OUT/'submission_receipt.json').read_text(encoding='utf-8'))
    manifest=json.loads((OUT/'batch_manifest.json').read_text(encoding='utf-8'))
    if len(receipt['jobs']) != 24 or len({i['job_id'] for i in receipt['jobs'].values()}) != 24:
        raise RuntimeError('Final cohort must have exactly 24 unique task identities')
    project_jobs=[j for j in api.request('/api/v1/jobs?limit=1000')['jobs']
                  if j.get('project_id')==receipt['project_id']]
    if len(project_jobs) != 24:
        raise RuntimeError('ACP final project does not contain exactly 24 tasks')
    rows=[]
    for rid,item in receipt['jobs'].items():
        base='/api/v1/jobs/'+item['job_id']
        job=api.request(base)
        if job['status'] != 'completed':
            raise RuntimeError(f'{rid}: real calculation is {job["status"]}')
        folder=OUT/rid
        plan=json.loads((folder/'PrimaryPlan.json').read_text(encoding='utf-8'))
        profile=api.request(base+'/s2/profile')
        graph=api.request(base+'/energy-graph')
        frames=profile['frames']
        expected=plan['primary']['n_points']
        if len(frames)!=expected or len(graph['nodes'])!=expected or not graph['complete']:
            raise RuntimeError(f'{rid}: missing scan frames/frontend nodes')
        energies=[f['scan_energy_hartree'] for f in frames]
        if not all(e is not None and math.isfinite(e) for e in energies):
            raise RuntimeError(f'{rid}: missing/nonfinite real energies')
        monitors={str(pair):[] for pair in plan['primary']['monitored_edits']}
        coordinate_errors=[]
        structures=folder/'frames'
        structures.mkdir(exist_ok=True)
        for index,f in enumerate(frames):
            data=api.request(base+'/s2/frame/'+str(index))
            xyztext=data['xyz']
            lines=xyztext.splitlines()
            atoms=[line.split() for line in lines[2:2+int(lines[0])]]
            if [a[0] for a in atoms] != plan['elements']:
                raise RuntimeError(f'{rid}: ACP frame atom identity changed')
            coords={m:tuple(map(float,a[1:4])) for m,a in zip(plan['maps'],atoms)}
            if not all(math.isfinite(v) for xyz in coords.values() for v in xyz):
                raise RuntimeError(f'{rid}: nonfinite geometry')
            pair=plan['primary']['maps']
            coordinate_errors.append(abs(math.dist(coords[pair[0]],coords[pair[1]])-f['target_coordinate']))
            for pair in plan['primary']['monitored_edits']:
                monitors[str(pair)].append(math.dist(coords[pair[0]],coords[pair[1]]))
            (structures/f'frame_{index:03d}.xyz').write_text(xyztext,encoding='utf-8')
        if max(coordinate_errors)>0.01:
            raise RuntimeError(f'{rid}: relaxed scan constraint not satisfied')
        relative=[(e-energies[0])*627.509474 for e in energies]
        peaks=[i for i in range(1,len(energies)-1)
               if energies[i]>energies[i-1]+1e-6 and energies[i]>energies[i+1]+1e-6]
        row={'reaction_id':rid,'job_id':item['job_id'],'n_frames':len(frames),
             'n_converged':sum(f['optimization_converged'] is True for f in frames),
             'max_constraint_error_angstrom':max(coordinate_errors),
             'interior_peak_indices':peaks,'max_relative_energy_kcal_mol':max(relative),
             'monitored_edit_distances_angstrom':monitors,
             'stationary_point_claimed':False,'frontend_energy_and_frame_access_verified':True}
        save_receipt(folder/'ACPProfile.json',profile)
        save_receipt(folder/'ACPResultValidation.json',row)
        rows.append(row)
    summary={'batch_id':BATCH,'project_id':receipt['project_id'],'n_completed':24,
             'n_frames':sum(r['n_frames'] for r in rows),
             'n_converged':sum(r['n_converged'] for r in rows),
             'max_constraint_error_angstrom':max(r['max_constraint_error_angstrom'] for r in rows),
             'n_scans_with_interior_peak':sum(bool(r['interior_peak_indices']) for r in rows),
             'truth_assisted_development':True,'stationary_point_claimed':False,'records':rows}
    save_receipt(OUT/'result_validation.json',summary)
    report=ROOT/'docs/reports/PES2TS_Demo24_ACP首批计算_20261001.md'
    lines=[
        '# Demo24 首批 ACP PES 计算结果（2026-10-01）', '',
        '24 条反应各有一个主拉伸任务，正式 ACP 项目内恰好 24 个任务，全部完成。', '',
        '- ACP 前端：http://127.0.0.1:8765/；项目：`PES2TS Demo24 — primary PES`。',
        f'- 项目 ID：`{receipt["project_id"]}`。',
        f'- 真实扫描结构与有限能量：{summary["n_frames"]} 帧；报告优化收敛：{summary["n_converged"]} 帧。',
        f'- 最大实际拉伸坐标偏差：{summary["max_constraint_error_angstrom"]:.9f} Å（验收阈值 0.01 Å）。',
        '- 所有任务的 ACP 能量图节点和逐帧 XYZ 接口均已实际读取验证。', '',
        '## 实施内容', '',
        '增加每反应唯一主拉伸选择层，要求起点存在有效成键、正向增加键长；跨组分独立坐标不作为终点距离。原候选保留为审计备选，执行入口只消费冻结的 24 条主计划。', '',
        '修复装配坐标传播、碰撞修复坐标未实际落地、监测目标量纲与原子身份绑定、成功调度说明误报失败、单扭转类型误拒绝、ORCA 约束块闭合及 CLI 显式输入清单。直线插值碰撞降为路径假设警告；起点真实碰撞和结构检查仍有阻断意义。', '',
        '## 计算与科学解释', '',
        '使用 ACP 原生 PESsearch / bond_length_scan，ORCA 调用 GFN2-xTB 进行逐点约束优化，单点 DFT 补算关闭；每条 16–19 点，共 394 点。其他反应键变化作为监测量，不强行施加共享线性多坐标驱动。', '',
        '本批采用已查阅 TS/IRC 的开发锚点，扫描起点为端点结构，范围按端点规则生成，并核查覆盖已知 TS 的主坐标。它属于 truth-assisted development，不能作为盲测成功率。没有伪造双人化学复核；用户执行授权单独记录。', '',
        f'{summary["n_scans_with_interior_peak"]} 条曲线具有内部离散局部峰，{24-summary["n_scans_with_interior_peak"]} 条没有。局部峰可能来自目标反应、竞争路径、跳支或数值波动；仅凭峰不能声称找到 TS。未执行 TS 优化、频率或新 IRC。所有计划和结果均记录 stationary_point_claimed=false。', '',
        '## 后端修复与审计', '',
        '首个试算因 ORCA 外部 xTB 线程过量而失败。修复 ACP `src/cccp/qc/interfaces/orca.py`，把 OMP/MKL/OpenBLAS 线程限制为任务 nproc；补齐空环境兼容。提交脚本内存由含糊的 2000 改为明确的 2000MB。首个试算在原任务中重算成功，失败状态和日志保留。', '',
        'ACP 自动重新生成任务显示名称，原按显示名称去重的方式漏掉试算。已改用 batch_id + reaction_id + project_id 识别。额外试算移入 `PES2TS Demo24 pilot audit` 项目保留；正式项目恰好 24 条，receipt 指向正式任务。', '',
        '并发提交暴露 ACP STARTING 等待者被计为资源占用、互相阻塞的问题。取消未开始计算的 20 个等待者，再沿用原任务身份通过 edit-recalculate 重新排队，采用持久化 batch_id/parallelism=1；未创建额外反应候选，未修改 ACP 调度器、未重启用户服务。任务恢复回执均保留。', '',
        '## 验证', '',
        '- PES2TS 完整回归：1360 passed，10 deselected；随后新增任务身份去重回归：3 passed。',
        '- ACP 环境及扫描回归：50 passed、1 failed；失败位于未修改的 MPI 探测测试 test_orca_runtime_env_no_mpi_and_no_ld_inherits，当前机器探测到实际 MPI 环境导致“返回 None”的假设不成立，该项未修复。',
        '- ORCA 线程修复定向验证：2 种环境通过；真实 24 条 / 394 帧计算及前端接口通过。', '',
        '## 每反应结果', '',
        '| 反应 | 起点 / 主键（map） | 帧 / 收敛 | 内部局部峰数 | 最大相对扫描能量 kcal/mol | ACP 任务 |',
        '|---|---|---:|---:|---:|---|',
    ]
    for row in rows:
        plan=json.loads((OUT/row['reaction_id']/'PrimaryPlan.json').read_text(encoding='utf-8'))['primary']
        lines.append(f'| {row["reaction_id"]} | {plan["start_endpoint"]} / {plan["maps"][0]}–{plan["maps"][1]} | {row["n_frames"]} / {row["n_converged"]} | {len(row["interior_peak_indices"])} | {row["max_relative_energy_kcal_mol"]:.2f} | `{row["job_id"]}` |')
    lines += ['', '## 可复核文件', '',
              f'- 批次目录：`outputs/{BATCH}/`。',
              '- batch_manifest.json / submission_receipt.json：24 条冻结计划和 ACP 身份。',
              '- result_validation.json / batch_status.json：汇总验收与最终状态。',
              '- 各 RXN 目录：PrimaryPlan.json、ACPJobRequest.json、ACPProfile.json、ACPResultValidation.json，以及 frames/*.xyz。',
              '- ACPResultValidation.json 含其他反应键的实际距离轨迹，便于判断单键拉伸是否伴随目标变化。',
              '- pilot_audit_receipt.json、capacity_recovery_cancel.json、各恢复回执：试算与排队恢复记录。',
              '- 代码修改尚未提交到 Git；预先存在的 learnings.md 修改保留。', '']
    report.write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='records'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','submit','status','repair-resources','resume-capacity','validate'])
    parser.add_argument('--limit',type=int,default=24)
    args=parser.parse_args()
    {'prepare':prepare,'submit':lambda:submit(args.limit),'status':status,'repair-resources':repair_resources,
     'resume-capacity':resume_capacity,'validate':validate_results}[args.action]()
