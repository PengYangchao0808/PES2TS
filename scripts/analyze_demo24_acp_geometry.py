"""User-authorized post-computation TS/IRC audit; never feeds planning inputs."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pes2ts_core.g0.truth.truth_reader import load_ts_geometry, load_irc_frames
from pes2ts_core.g1.truth_alignment import kabsch_rmsd, analyze_irc_layout
from pes2ts_core.integration.acp.scheduler_backend import save_receipt

BATCH=ROOT/'outputs/PES2TS_Demo24_primary_20261001'
OUT=ROOT/'outputs/demo24_acp_geometry_audit_20261001'
CALLER='demo24_acp_postrun_endpoint_and_ts_audit_20261001'


def geometry(path, elements):
    lines=path.read_text(encoding='utf-8').splitlines()
    rows=[line.split() for line in lines[2:2+int(lines[0])]]
    assert [r[0] for r in rows]==elements
    xyz=np.array([[float(v) for v in r[1:4]] for r in rows])
    assert np.isfinite(xyz).all()
    return xyz


def lengths(xyz, pairs, row):
    return np.array([np.linalg.norm(xyz[row[a]]-xyz[row[b]]) for a,b in pairs])


def main():
    OUT.mkdir(exist_ok=True)
    manifest=json.loads((BATCH/'batch_manifest.json').read_text(encoding='utf-8'))
    cases=[]
    for entry in manifest['records']:
        rid=entry['reaction_id']
        folder=BATCH/rid
        snapshot=ROOT/f'tests/fixtures/p0_demo24/records/{rid}.json'
        s=json.loads(snapshot.read_text(encoding='utf-8'))
        plan=json.loads((folder/'PrimaryPlan.json').read_text(encoding='utf-8'))
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==plan['source_snapshot_sha256']
        p=plan['primary']
        maps=s['maps']; row={m:i for i,m in enumerate(maps)}
        heavy=[i for i,e in enumerate(s['elements']) if e!='H']
        original={'R':np.asarray(s['r_coordinates']), 'P':np.asarray(s['p_coordinates'])}
        inp=geometry(folder/'start.xyz',s['elements'])
        frames=np.array([geometry(path,s['elements']) for path in sorted((folder/'frames').glob('frame_*.xyz'))])
        assert len(frames)==p['n_points']
        shard=f'{int(rid.split("_")[1])//1000:05d}'
        mapping=json.loads((ROOT/f'data/interim/g1_truth/p1_mapping/{shard}/{rid}.json').read_text(encoding='utf-8'))
        truth=load_ts_geometry(rid,True,manifests_dir=ROOT/'data/manifests',caller=CALLER)
        irc=load_irc_frames(rid,True,manifests_dir=ROOT/'data/manifests',caller=CALLER)
        trows={int(x['map']):int(x['ts_irc_index']) for x in mapping['mapping']['map_to_atoms']}
        assert [truth['atomic_numbers'][trows[m]] for m in maps]==[Chem.GetPeriodicTable().GetAtomicNumber(e) for e in s['elements']]
        ts=np.asarray(truth['coordinates'])[np.array([trows[m] for m in maps])]
        ircraw=np.asarray(irc['coordinates'])
        layout=analyze_irc_layout(ircraw,np.asarray(truth['coordinates']))
        assert layout.ts_frame_rmsd<0.03 and layout.branch_b_indices
        orientation=mapping['irc_validation']['orientation']
        assert orientation in ('R_first','P_first')
        sides=('R','P') if orientation=='R_first' else ('P','R')
        endpoint_indices={sides[0]:layout.branch_a_indices[-1],sides[1]:layout.branch_b_indices[-1]}
        ircends={side:ircraw[index][np.array([trows[m] for m in maps])] for side,index in endpoint_indices.items()}
        audit=json.loads((ROOT/f'outputs/demo24_truth_algorithm_audit_20261001/{rid}.json').read_text(encoding='utf-8'))
        edits=[x['maps'] for x in audit['changed_bonds']]
        reactive=sorted({row[m] for pair in edits for m in pair})
        qts=lengths(ts,edits,row)
        rmsds=np.array([kabsch_rmsd(f,ts) for f in frames])
        heavyrmsds=np.array([kabsch_rmsd(f[heavy],ts[heavy]) for f in frames])
        reactrmsds=np.array([kabsch_rmsd(f[reactive],ts[reactive]) for f in frames])
        errors=np.array([np.abs(lengths(f,edits,row)-qts) for f in frames])
        maxerrors=errors.max(axis=1)
        ibest=int(np.argmin(rmsds))
        ijoint=int(np.argmin(np.maximum(rmsds/0.2,maxerrors/0.2)))
        driver_ts=float(lengths(ts,[p['maps']],row)[0])
        driver_values=np.array([lengths(f,[p['maps']],row)[0] for f in frames])
        start=p['start_endpoint']; end='P' if start=='R' else 'R'
        ends={}
        for label,frame,side in [('input',inp,start),('scan_first',frames[0],start),('scan_last',frames[-1],end)]:
            ends[label]={'reference_endpoint':side,
                         'fixture_all_atom_rmsd':kabsch_rmsd(frame,original[side]),
                         'fixture_heavy_atom_rmsd':kabsch_rmsd(frame[heavy],original[side][heavy]),
                         'irc_endpoint_all_atom_rmsd':kabsch_rmsd(frame,ircends[side]),
                         'max_edit_distance_error_to_fixture':float(np.abs(lengths(frame,edits,row)-lengths(original[side],edits,row)).max()),
                         'max_edit_distance_error_to_irc_endpoint':float(np.abs(lengths(frame,edits,row)-lengths(ircends[side],edits,row)).max())}
        metrics=[]
        for i in range(len(frames)):
            metrics.append({'index_0based':i,'frame_number_1based':i+1,'all_atom_rmsd_to_ts':float(rmsds[i]),
                            'heavy_atom_rmsd_to_ts':float(heavyrmsds[i]),'reactive_atom_rmsd_to_ts':float(reactrmsds[i]),
                            'max_edit_distance_error_to_ts':float(maxerrors[i]),
                            'driver_distance':float(driver_values[i]),
                            'edit_distance_errors_to_ts':[{ 'maps':pair,'error_angstrom':float(err)} for pair,err in zip(edits,errors[i])]})
        case={'reaction_id':rid,'n_frames':len(frames),'start_endpoint':start,'target_endpoint':end,
              'driver_maps':p['maps'],'component_placements':p['component_placements'],
              'input_max_absolute_coordinate_difference_to_fixture':float(np.abs(inp-original[start]).max()),
              'endpoints':ends,'ts_driver_distance':driver_ts,
              'minimum_driver_distance_error_to_ts':float(np.abs(driver_values-driver_ts).min()),
              'minimum_all_atom_rmsd_to_ts':float(rmsds.min()),
              'minimum_heavy_atom_rmsd_to_ts':float(heavyrmsds.min()),
              'minimum_edit_max_error_to_ts':float(maxerrors.min()),
              'best_rmsd_frame':metrics[ibest],'best_joint_frame':metrics[ijoint],
              'passes_joint_020':bool(np.any((rmsds<=0.2)&(maxerrors<=0.2))),
              'passes_joint_030':bool(np.any((rmsds<=0.3)&(maxerrors<=0.3))),
              'passes_joint_050':bool(np.any((rmsds<=0.5)&(maxerrors<=0.5))),
              'frame_metrics':metrics}
        save_receipt(OUT/f'{rid}.json',case)
        cases.append(case)
        print(rid, 'first',round(ends['scan_first']['fixture_all_atom_rmsd'],3),
              'last',round(ends['scan_last']['fixture_all_atom_rmsd'],3),
              'TS',round(rmsds.min(),3),'edit',round(maxerrors.min(),3), 'near',case['passes_joint_020'],flush=True)
    summary={'n_reactions':len(cases),'n_frames':sum(c['n_frames'] for c in cases),
             'n_input_same_as_fixture_1e_6':sum(c['input_max_absolute_coordinate_difference_to_fixture']<=1e-6 for c in cases),
             'n_first_frame_fixture_rmsd_le_020':sum(c['endpoints']['scan_first']['fixture_all_atom_rmsd']<=0.2 for c in cases),
             'n_last_frame_fixture_rmsd_le_020':sum(c['endpoints']['scan_last']['fixture_all_atom_rmsd']<=0.2 for c in cases),
             'n_last_frame_joint_endpoint_le_020':sum(c['endpoints']['scan_last']['fixture_all_atom_rmsd']<=0.2 and c['endpoints']['scan_last']['max_edit_distance_error_to_fixture']<=0.2 for c in cases),
             'n_first_frame_irc_rmsd_le_020':sum(c['endpoints']['scan_first']['irc_endpoint_all_atom_rmsd']<=0.2 for c in cases),
             'n_last_frame_irc_rmsd_le_020':sum(c['endpoints']['scan_last']['irc_endpoint_all_atom_rmsd']<=0.2 for c in cases),
             'n_ts_driver_distance_error_le_010':sum(c['minimum_driver_distance_error_to_ts']<=0.1 for c in cases),
             'n_min_ts_all_atom_rmsd_le_020':sum(c['minimum_all_atom_rmsd_to_ts']<=0.2 for c in cases),
             'n_joint_ts_020':sum(c['passes_joint_020'] for c in cases),
             'n_joint_ts_030':sum(c['passes_joint_030'] for c in cases),
             'n_joint_ts_050':sum(c['passes_joint_050'] for c in cases),
             'criteria':'Map-preserving proper Kabsch all-atom RMSD AND max absolute distance error over all changed reaction bonds, same frame. Thresholds are explicit audit conventions, not TS certification.',
             'ts_truth_used_only_for_postrun_audit':True,'records':cases}
    save_receipt(OUT/'summary.json',summary)
    report=['# Demo24 扫描首尾与 TS 结构核查（2026-10-01）','',
            '比较实际 ACP 的 394 帧与原始 R/P 输入、IRC 两端及已知 TS。按已验证 atom map 对应，先移除整体平移和旋转，禁止镜像对齐；另核对全部反应变化键距离。编号均为从 1 开始的帧号。','',
            '## 判据与结论','',
            f'输入几何与选定原始端点一致（坐标误差 ≤1e-6 Å）：{summary["n_input_same_as_fixture_1e_6"]}/24。',
            f'首个实际优化帧相对原始起始端点 RMSD ≤0.2 Å：{summary["n_first_frame_fixture_rmsd_le_020"]}/24；末帧相对原始目标端点：{summary["n_last_frame_fixture_rmsd_le_020"]}/24。',
            f'末帧同时满足原始目标端点 RMSD ≤0.2 Å、全部变化键距离偏差 ≤0.2 Å：{summary["n_last_frame_joint_endpoint_le_020"]}/24。',
            f'对照原始 IRC 端点，首帧/末帧 RMSD ≤0.2 Å：{summary["n_first_frame_irc_rmsd_le_020"]}/24 和 {summary["n_last_frame_irc_rmsd_le_020"]}/24。','',
            f'仅主键距离接近 TS（最小误差 ≤0.1 Å）：{summary["n_ts_driver_distance_error_le_010"]}/24；至少一帧整体 TS RMSD ≤0.2 Å：{summary["n_min_ts_all_atom_rmsd_le_020"]}/24。',
            f'同一帧同时满足整体 RMSD 和全部变化键最大距离误差 ≤0.2 Å：{summary["n_joint_ts_020"]}/24；放宽到 0.3 Å：{summary["n_joint_ts_030"]}/24；0.5 Å：{summary["n_joint_ts_050"]}/24。','',
            '0.2/0.3/0.5 Å 是本次明确的几何筛查尺度，不是通用 TS 认证标准。距离峰、单个主键长度和局部能量极大均不足以确认目标 TS。Hydrogen atom maps 保持原对应，重原子及反应中心 RMSD 同时保存在 JSON 中供排查对称氢/构象影响。','',
            '## 为什么不能保证首尾与目标 TS','',
            '当前起点输入来自所选已成键一端，实际首帧重新进行了 GFN2-xTB 约束优化，因此会偏离原始计算级别端点。R→P 与 P→R 均可作为拉伸方向。终点是延长主键后的约束优化结果，没有固定为另一原始端点，也没有使用目标端点整体几何作为边界条件。范围覆盖 TS 主键距离只证明采样到该距离，其他反应键、角度和二面角仍可走向竞争路径。','',
            '## 逐条几何结果','',
            '| 反应 | 方向 | 首帧/原起点 RMSD Å | 末帧/原目标 RMSD Å | 最小 TS RMSD Å | 最佳 RMSD 帧 | 该帧变化键最大偏差 Å | 联合 0.2 Å 接近 TS |',
            '|---|---|---:|---:|---:|---:|---:|---|']
    for c in cases:
        report.append(f'| {c["reaction_id"]} | {c["start_endpoint"]}→{c["target_endpoint"]} | {c["endpoints"]["scan_first"]["fixture_all_atom_rmsd"]:.3f} | {c["endpoints"]["scan_last"]["fixture_all_atom_rmsd"]:.3f} | {c["minimum_all_atom_rmsd_to_ts"]:.3f} | {c["best_rmsd_frame"]["frame_number_1based"]} | {c["best_rmsd_frame"]["max_edit_distance_error_to_ts"]:.3f} | {"是" if c["passes_joint_020"] else "否"} |')
    report+=['','逐帧指标：`outputs/demo24_acp_geometry_audit_20261001/RXN_*.json`；汇总：`summary.json`。已通过 truth_reader 记录全部 TS/IRC 访问，仅用于本次事后核查，未改变计算结果或规划输入。','']
    (ROOT/'docs/reports/PES2TS_Demo24_端点与TS几何核查_20261001.md').write_text('\n'.join(report),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='records'},ensure_ascii=False),flush=True)


def enrich_summary():
    """Cheap additional checks; reuse the audited per-frame measurements."""
    path=OUT/'summary.json'
    summary=json.loads(path.read_text(encoding='utf-8'))
    cases=summary['records']
    differences=[]
    component_checks=[]
    for c in cases:
        rid=c['reaction_id']
        s=json.loads((ROOT/f'tests/fixtures/p0_demo24/records/{rid}.json').read_text(encoding='utf-8'))
        plan=json.loads((BATCH/rid/'PrimaryPlan.json').read_text(encoding='utf-8'))['primary']
        side=plan['start_endpoint']; opposite='P' if side=='R' else 'R'
        coords=np.asarray(s['r_coordinates'] if opposite=='R' else s['p_coordinates'])
        row={m:i for i,m in enumerate(s['maps'])}
        original_target=float(lengths(coords,[plan['maps']],row)[0])
        differences.append({'reaction_id':rid,'planned_end':plan['end'],
                            'original_opposite_endpoint_driver_distance':original_target,
                            'difference':plan['end']-original_target})
        original=np.asarray(s['r_coordinates'] if side=='R' else s['p_coordinates'])
        inp=geometry(BATCH/rid/'start.xyz',s['elements'])
        params=Chem.SmilesParserParams(); params.removeHs=False
        mol=Chem.MolFromSmiles(s['reaction_smiles'].split('>>')[0 if side=='R' else 1],params)
        component_rmsds=[]
        for group in Chem.GetMolFrags(mol):
            indices=[row[mol.GetAtomWithIdx(i).GetAtomMapNum()] for i in group]
            component_rmsds.append(kabsch_rmsd(inp[indices],original[indices]))
        component_checks.append({'reaction_id':rid,'input_component_rmsds':component_rmsds})
    summary['n_input_components_preserved_1e_6']=sum(max(c['input_component_rmsds'])<=1e-6 for c in component_checks)
    summary['n_scans_reactive_joint_ts_020']=sum(any(f['reactive_atom_rmsd_to_ts']<=0.2 and f['max_edit_distance_error_to_ts']<=0.2 for f in c['frame_metrics']) for c in cases)
    summary['n_scans_heavy_joint_ts_020']=sum(any(f['heavy_atom_rmsd_to_ts']<=0.2 and f['max_edit_distance_error_to_ts']<=0.2 for f in c['frame_metrics']) for c in cases)
    summary['n_planned_end_driver_matches_original_opposite_001']=sum(abs(d['difference'])<=0.01 for d in differences)
    summary['end_driver_differences']=differences
    summary['input_component_checks']=component_checks
    save_receipt(path,summary)
    report=ROOT/'docs/reports/PES2TS_Demo24_端点与TS几何核查_20261001.md'
    text=report.read_text(encoding='utf-8')
    marker='## 补充核查'
    text=text.split(marker)[0].rstrip()
    text+='\n\n'+marker+'\n\n'
    text+=f'- 输入各独立组分内部几何保持一致：{summary["n_input_components_preserved_1e_6"]}/24。23 条整体坐标直接保留；RXN_0000155302 有一个组分沿 x 平移 8.413883924 Å，改变组分相对摆放，未改变组分内部几何。\n'
    text+=f'- 计划终点主键距离与原始另一端点相同（误差 ≤0.01 Å）：{summary["n_planned_end_driver_matches_original_opposite_001"]}/24。原始跨组分坐标未装配时，跨组分距离本身不宜解释为物理终点；IRC 与整体几何核查仍显示终点没有被固定为目标结构。\n'
    text+=f'- 放宽为仅反应中心 RMSD 与变化键距离误差均 ≤0.2 Å：{summary["n_scans_reactive_joint_ts_020"]}/24；重原子整体 RMSD 与变化键距离误差均 ≤0.2 Å：{summary["n_scans_heavy_joint_ts_020"]}/24。局部接近不能代替整体 TS 相似。\n'
    report.write_text(text,encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if not isinstance(v,list)},ensure_ascii=False),flush=True)


if __name__=='__main__':
    if '--enrich-only' not in sys.argv:
        main()
    enrich_summary()
