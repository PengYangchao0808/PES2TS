"""Rebuild design evidence, not executable ScanPlans or an automatic selector.

Only the frozen Demo24 endpoint CSV, sanitized exports, and review cases are read.
PROPOSALS are explicit design hypotheses; measured graph features are recomputed.
Run from any directory with: python <absolute path to this file>
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter, deque
from pathlib import Path

from rdkit import Chem, rdBase

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from pes2ts_core.contracts import loads_document

# rid, family, mode, driver map pairs, direction, interpretation, next candidate, risks
PROPOSALS = [
 (7104, 'H_TRANSFER', 'SINGLE_1D', [(3,11)], 'P_to_R', 'N1–H11→N3；1–2 键级变化是伴随电子重排。', '双 B：1–11、3–11；再比较反向。', '邻接芳香骨架；不能把 1–2 的键级变化硬编码为独立成键。'),
 (161724, 'H_TRANSFER', 'SINGLE_1D', [(5,13)], 'P_to_R', 'O4–H13→N5，O+/N− 形式电荷消失。', '双 B：4–13、5–13；R→P 复核。', '总电荷不变；形式电荷变化不能直接判断电子转移或自旋。'),
 (91050, 'H_TRANSFER', 'SINGLE_1D', [(7,9)], 'P_to_R', 'C1–H9→N7，产物形成 C−/N+。', '双 B：1–9、7–9；R→P 复核。', '电荷分离、产物 N7 立体结构和供受体取向需复核。'),
 (77619, 'LOCAL_CONNECTIVITY', 'SINGLE_1D', [(2,5)], 'P_to_R', 'C2–C5 成键闭环；从产物已成键几何拉伸。', 'R→P，必要时经同骨架 D 预组织后重建计划。', '环张力和新立体中心；保持目标构型。'),
 (26256, 'LOCAL_CONNECTIVITY', 'SINGLE_1D', [(3,7)], 'R_to_P', 'C3–C7 断键开环。', 'P→R；若重闭环失败转 NEB。', '小环开裂电子态和非目标骨架断裂。'),
 (79731, 'LOCAL_CONNECTIVITY', 'SINGLE_1D', [(6,8)], 'R_to_P', 'N6–N8 断裂，N6–N7 单键变双键。', 'P→R；必要时 NEB。', 'resolved_symmetry_collapsed；等价映射必须对驱动键及立体目标不产生歧义。'),
 (53132, 'CONNECTIVITY_EXCHANGE', 'COUPLED_1D', [(5,6),(6,8)], 'P_to_R', 'O6 从 N8 转接 C5；经 C5–C7–N8 的键级重排耦合。', 'R→P；O6 转接偏早/偏晚时间表；NEB。', '共有原子是迁移的 O6；不能仅凭 F/B 共享原子命名为 SN2。'),
 (155302, 'CONNECTIVITY_EXCHANGE', 'COUPLED_1D', [(1,2),(1,5)], 'P_to_R', 'C1 从 C5 转接 C2；C2–N4–C5 键级重排；HCl 为图上不变组分。', '装配合格后 R→P；两种转接时间表；NEB。', '2→2 组分；HCl 可能参与相互作用，不能擅自删除或永久冻结。'),
 (171675, 'CONNECTIVITY_EXCHANGE', 'COUPLED_1D', [(1,2),(1,5)], 'P_to_R', 'C1 从 C5 转接 C2，另有两处 C–N 键级改变。', 'R→P；转接偏早/偏晚；NEB。', '电荷分离和含 N 骨架耦合；共享 C1 不等于已知取代机理。'),
 (100071, 'RING_COUPLED', 'COUPLED_1D', [(1,3),(2,4)], 'P_to_R', '两根 O–C 成键与 C3–O4 断裂、O1–C2 降键级耦合。', '若 3–4 不回成键，三 B 加入 3–4；再反向/NEB。', '多环变化；第三根键作为监测量有待试点证明。'),
 (132222, 'H_TRANSFER_COUPLED', 'COUPLED_1D', [(3,4),(5,16)], 'P_to_R', 'O3–C4 闭环并伴 H16 从 O3 转移到 O5。', '三 B 加入 3–16；H 转移偏早/偏晚；NEB。', 'H 与重原子事件的时序未知；不能因同一编辑环就断言同步。'),
 (102998, 'RING_COUPLED', 'COUPLED_1D', [(2,7),(4,8)], 'P_to_R', 'C2–C7、C4–O8 成键；C2–C4 断裂；C7–O8 降键级。', '三 B 加入 2–4；再反向/NEB。', '环应变及未驱动 2–4 的完成情况。'),
 (59834, 'MULTI_EVENT_CONNECTED', 'COUPLED_1D', [(1,10),(2,6),(3,6)], 'P_to_R', 'Cl10 转接 C1，C6 同时与 C2/C3 成键；多事件共中心 C6。', 'NEB 优先回退；只在预冻结预算内比较分组时间表。', '5 条连接编辑；3 个驱动仅是降维假设，两个断键都需监测。'),
 (94438, 'MULTI_EVENT_CONNECTED', 'COUPLED_1D', [(1,8),(2,7),(3,4)], 'P_to_R', 'H8 转移与 N2–O7、C3–N4 两处闭环耦合。', 'NEB；若发现稳定中间体再提交分阶段新版计划。', '完整 H 双距离再加两成键需 4 B，不能塞入原生 Simul_Scan。'),
 (187964, 'MULTI_EVENT_CONNECTED', 'COUPLED_1D', [(2,9),(3,4),(7,8)], 'P_to_R', 'F9 转接、含氧环重组与 C2–O4 降键级形成相连编辑网络。', 'NEB；或有几何依据的有限时序变体。', '5 条连接编辑；三个成键距离不能证明两个断键会随动。'),
 (100736, 'H2_EVENT', 'COUPLED_1D', [(5,12),(5,13),(12,13)], 'P_to_R', 'H2 加到同一 C5，同时 N1–C5 开环及 N1–C2 升键级。', 'NEB；只有证实环事件可随动才保留三 B 候选。', 'R 为 2 组分，先装配；H–H 事件去重；三距离须满足三角不等式。'),
 (149368, 'H2_EVENT', 'COUPLED_1D', [(1,10),(3,13),(10,13)], 'R_to_P', 'C1–H10、C3–H13 断裂生成 H2，另伴 N4–C2 开环。', '重新装配 P 后再生成数值时间表；NEB。', 'P 中已断开的 C3–H13 只有约 0.754 Å，禁止直接插值；现有跨组分摆放需修复。'),
 (17762, 'H2_EVENT', 'COUPLED_1D', [(2,9),(5,12),(9,12)], 'P_to_R', 'H2 消耗到 C2/C5，另形成 O7–O8 并改变两个羰基键级。', '若 O7–O8 不随动，转 NEB；不静默加第四个 Scan。', 'R 为 2 组分；H2 朝向、O–O 成键及多中心时序均需检查。'),
 (138453, 'AROMATIC_COUPLED', 'COUPLED_1D', [(2,4),(2,6),(5,7)], 'R_to_P', 'C2 转接 N4、C5–C7 开裂伴五元环芳香化。', 'P→R；NEB。', '5 个芳香键级变化按区域监测；不是 5 根扫描键，也不全是表示噪声。'),
 (110869, 'AROMATIC_COUPLED', 'COUPLED_1D', [(5,8),(8,9)], 'P_to_R', 'Cl8 从 N5 转到 C9，六元区域芳香化。', 'R→P；Cl 转接早/晚；NEB。', '6 项环键级变化压成一个区域观察对象，保留各键证据。'),
 (47010, 'AROMATIC_COUPLED', 'COUPLED_1D', [(2,3),(5,6)], 'P_to_R', 'C2–N3 断裂、O5–N6 成键使环骨架扩展并芳香化。', 'R→P；NEB。', '5–6 是真实新增连接（产物键级 1.5），不能被芳香归一化删除。'),
 (194484, 'NETWORK_PATH', 'PATH_REQUIRED', [], 'R_to_P', 'H10 转移、三根重原子成键、三个键级变化形成密集环重组。', '双端 NEB；稳定中间体确认后分阶段。', 'H 双距离加三根重原子新键需 5 B；选任意 3 根没有充分事件覆盖依据。'),
 (74997, 'NETWORK_PATH', 'PATH_REQUIRED', [], 'R_to_P', '四根断键、一根成键与三个升键级分布在相连多环骨架。', '双端 NEB；有中间体证据后拆为多个基元段。', '不能因 F=1 就只扫描 3–8；1–6 原始距离变化达约 4.36 Å，构象变化显著。'),
 (109608, 'NETWORK_PATH', 'PATH_REQUIRED', [], 'P_to_R', '两组三键片段形成四条交叉连接，得到紧密四原子网络。', '重建 R 侧组分装配后 NEB；有可靠中间体再分段。', 'R 中未来成键 3–4、3–5 已比 P 更短；原始跨组分距离不可作形成窗口。'),
]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def components(vertices, edges):
    adj = {v:set() for v in vertices}
    for a,b in edges:
        adj[a].add(b); adj[b].add(a)
    result=[]; unseen=set(vertices)
    while unseen:
        seed=min(unseen); seen={seed}; stack=[seed]; unseen.remove(seed)
        while stack:
            for v in sorted(adj[stack.pop()] & unseen):
                unseen.remove(v); seen.add(v); stack.append(v)
        result.append(sorted(seen))
    return result

def shortest_path(bonds, a, b, exclude=None):
    adj={}
    for pair in bonds:
        if pair == exclude: continue
        i,j=pair; adj.setdefault(i,[]).append(j); adj.setdefault(j,[]).append(i)
    queue=deque([[a]]); visited={a}
    while queue:
        path=queue.popleft()
        if path[-1]==b: return path
        for nxt in sorted(adj.get(path[-1],[])):
            if nxt not in visited: visited.add(nxt); queue.append(path+[nxt])
    return None

def graph(mol):
    return {tuple(sorted((b.GetBeginAtom().GetAtomMapNum(),b.GetEndAtom().GetAtomMapNum()))):
            1.5 if b.GetIsAromatic() else float(b.GetBondTypeAsDouble()) for b in mol.GetBonds()}

def pairtext(pair): return '–'.join(map(str,pair))

def main():
    source=ROOT/'data/manifests/pes2ts_demo24_candidates_v1.csv'
    manifest_path=ROOT/'data/manifests/demo24_reaction_case_manifest_v1.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    frozen={r['reaction_id']:r for r in manifest['records']}
    rows=list(csv.DictReader(source.open(encoding='utf-8-sig',newline='')))
    specs={f'RXN_{s[0]:010d}':s for s in PROPOSALS}
    assert len(specs)==len(rows)==24 and set(specs)=={r['reaction_id'] for r in rows}==set(frozen)
    records=[]
    for row in rows:
        rid=row['reaction_id']; spec=specs[rid]
        export_path=ROOT/'data/interim/g1_v2/export_contracts_v1'/f'{int(rid[4:])//1000:05d}'/(rid+'.json')
        case_path=ROOT/'data/interim/g1_v2/reaction_cases_review_v1'/row['split']/(rid+'.json')
        export=json.loads(export_path.read_text(encoding='utf-8'))
        case=loads_document(case_path.read_text(encoding='utf-8'))
        assert sha(export_path)==frozen[rid]['source_export_sha256']
        assert case['content_sha256']==frozen[rid]['content_sha256']
        assert case['status']=='needs_review'
        parser=Chem.SmilesParserParams(); parser.removeHs=False
        mols=[Chem.MolFromSmiles(s,parser) for s in row['reaction_smiles'].split('>>')]
        assert len(mols)==2 and all(m is not None for m in mols)
        graphs=[graph(m) for m in mols]
        diffs={p:(graphs[0].get(p),graphs[1].get(p)) for p in graphs[0].keys()|graphs[1].keys() if graphs[0].get(p)!=graphs[1].get(p)}
        assert diffs=={tuple(e['pair']):(e['r_bond_order'],e['p_bond_order']) for e in export['edits']}
        maps=export['maps']; index={m:i for i,m in enumerate(maps)}
        assert maps==[a['atom_map_id'] for a in case['atoms']]
        assert export['elements']==[a['element'] for a in case['atoms']]
        assert case['reactant']['geometry']==export['r_coordinates']
        assert case['product']['geometry']==export['p_coordinates']
        for mol in mols: assert sorted(a.GetAtomMapNum() for a in mol.GetAtoms())==sorted(maps)
        edits=[]
        for e in export['edits']:
            item=dict(e); a,b=e['pair']; i,j=index[a],index[b]
            item['distance_R_A']=round(math.dist(export['r_coordinates'][i],export['r_coordinates'][j]),6)
            item['distance_P_A']=round(math.dist(export['p_coordinates'][i],export['p_coordinates'][j]),6)
            item['cross_component_R']=export['atom_rows'][i]['r_component']!=export['atom_rows'][j]['r_component']
            item['cross_component_P']=export['atom_rows'][i]['p_component']!=export['atom_rows'][j]['p_component']
            if e['edit_kind']=='formed': item['support_path_R']=shortest_path(graphs[0],a,b)
            if e['edit_kind']=='broken': item['support_path_P']=shortest_path(graphs[1],a,b)
            edits.append(item)
        pairs={tuple(e['pair']) for e in edits}; drivers=[tuple(sorted(p)) for p in spec[3]]
        assert set(drivers)<=pairs and len(drivers)==len(set(drivers))<=3
        observers=sorted(pairs-set(drivers))
        assert set(drivers).isdisjoint(observers) and set(drivers)|set(observers)==pairs
        vertices=sorted({a for p in pairs for a in p}); cc=components(vertices,pairs)
        fb={tuple(e['pair']) for e in edits if e['edit_kind']!='order_changed'}
        attrs=[]
        atom_tables=[{a.GetAtomMapNum():a for a in m.GetAtoms()} for m in mols]
        for amap in maps:
            a,b=[t[amap] for t in atom_tables]
            before={'formal_charge':a.GetFormalCharge(),'radical_electrons':a.GetNumRadicalElectrons(),'aromatic':a.GetIsAromatic(),'chiral_tag':str(a.GetChiralTag())}
            after={'formal_charge':b.GetFormalCharge(),'radical_electrons':b.GetNumRadicalElectrons(),'aromatic':b.GetIsAromatic(),'chiral_tag':str(b.GetChiralTag())}
            if before!=after: attrs.append({'atom_map_id':amap,'R':before,'P':after})
        endpoints={side:{'n_components':len(Chem.GetMolFrags(mol)), 'cycle_rank':mol.GetNumBonds()-mol.GetNumAtoms()+len(Chem.GetMolFrags(mol)), 'charge':case[side]['charge'],'multiplicity':case[side]['multiplicity']} for side,mol in zip(('reactant','product'),mols)}
        for side,mol in zip(('reactant','product'),mols):
            endpoints[side]['rdkit_formal_charge_sum']=sum(a.GetFormalCharge() for a in mol.GetAtoms())
            endpoints[side]['rdkit_charged_atom_count']=sum(a.GetFormalCharge()!=0 for a in mol.GetAtoms())
            endpoints[side]['rdkit_radical_electron_count']=sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
            assert endpoints[side]['rdkit_formal_charge_sum']==endpoints[side]['charge']
        assert all(endpoints[s]['n_components']==int(row[k]) for s,k in [('reactant','R_components'),('product','P_components')])
        rec={'reaction_id':rid,'stratum':row['stratum'],'split':row['split'],'proposal_status':'design_hypothesis','execution_eligible':False,
             'case_status':case['status'],'case_sha256':case['content_sha256'],'export_sha256':sha(export_path),'case_path':case_path.relative_to(ROOT).as_posix(),
             'mapping_status':row['mapping_status'],'mapping_provenance':export['mapping_provenance'],
             'features':{'edit_components':cc,'edit_cycle_rank':len(pairs)-len(vertices)+len(cc),'connectivity_edit_components':components(sorted({a for p in fb for a in p}),fb),
                         'edit_counts':dict(Counter(e['edit_kind'] for e in edits)),'endpoints':endpoints,'atom_attribute_changes':attrs,'hydrogen_events':export['hydrogen_partner_changes'],'aromatic_regions':export['aromatic_regions']},
             'edits':edits,'proposal':{'family':spec[1],'mode':spec[2],'drivers_map':[list(p) for p in drivers],'observers_map':[list(p) for p in observers],
                                      'direction':spec[4],'interpretation':spec[5],'fallback':spec[6],'risk':spec[7],
                                      'auxiliary_monitor_policy':'H: donor-acceptor distance and D-H-A angle; rings/aromatics: regional geometry, stereo and non-target contacts',
                                      'required_before_execution':['manual_chemistry_review','endpoint_geometry_and_assembly_check','backend_capability_check','frozen_schedule_and_budget']}}
        records.append(rec)
    payload={'schema_name':'Demo24ScanStrategyDesignEvidence','schema_version':1,'status':'design_only','date':'2026-09-30','rdkit_version':rdBase.rdkitVersion,
             'source_csv_sha256':sha(source),'source_manifest_sha256':sha(manifest_path),
             'builder_sha256':sha(Path(__file__)),
             'information_boundary':'Endpoint CSV, sanitized R/P exports and review cases only; no TS/IRC/energies read. Existing mapping provenance is truth_assisted_p1.',
             'proposal_provenance':'Manually specified design hypotheses in PROPOSALS; this script does not implement strategy selection.',
             'counts':{'records':len(records),'case_status':dict(Counter(r['case_status'] for r in records)), 'modes':dict(Counter(r['proposal']['mode'] for r in records)), 'splits':dict(Counter(r['split'] for r in records))},'records':records}
    (OUT/'demo24_strategy_evidence.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# Demo24：逐条图特征与扫描策略建议','', '> 2026-09-30；设计假设，24 条均仍为 needs_review，执行资格均为 false。', '',
           '本表由白名单端点材料重算图特征和距离；策略建议为设计阶段人工明确列出的候选，不是选择器运行结果，也不是化学复核通过记录。原子号全部是 atom map；距离单位 Å。跨组分原始距离只用于诊断，装配通过后才能定扫描范围。', '',
           '## 总览','', '| 反应 | 分层 / split | 主策略 | 驱动 B(map) | 建议首方向 |','| --- | --- | --- | --- | --- |']
    for r in records:
        p=r['proposal']; lines.append(f"| {r['reaction_id']} | {r['stratum']} / {r['split']} | {p['family']} · {p['mode']} | {', '.join(pairtext(x) for x in p['drivers_map']) or '双端路径，无固定 B'} | {p['direction']} |")
    lines.extend(['','模式建议：'+json.dumps(payload['counts']['modes'],ensure_ascii=False)+'。这些数字表示待验证候选数量。','',
                  '所有条目通用监测：全目标编辑、非目标短接触/断裂、构型、逐点收敛与约束残差。H 转移另监测 D–A 距离及 D–H–A 角；环/芳香变化另监测对应区域的几何与电子指标。下列观察键是与驱动键不重复的全部编辑；驱动键本身也参与结果核验。', '',
                  'edit_cc 为含 F/B/O 的编辑图分量；fb_cc 只含连接变化。cycle_rank=E−V+C 为图不变量，不代表协同机理。端点芳香归一化采用当前 RDKit 默认解析，并逐键与冻结导出核对。chiral_tag 变化仅作复核提示；不能替代映射后的几何手性检验。'])
    for r in records:
        p=r['proposal']; f=r['features']; ep=f['endpoints']
        lines.extend(['',f"## {r['reaction_id']} · {r['stratum']} · {r['split']}",'',
                      f"- **编辑解释**：{p['interpretation']}",
                      f"- **图特征**：F/B/O={f['edit_counts'].get('formed',0)}/{f['edit_counts'].get('broken',0)}/{f['edit_counts'].get('order_changed',0)}；edit_cc={f['edit_components']}；fb_cc={f['connectivity_edit_components']}；编辑图 cycle_rank={f['edit_cycle_rank']}；端点 cycle_rank={ep['reactant']['cycle_rank']}→{ep['product']['cycle_rank']}；组分={ep['reactant']['n_components']}→{ep['product']['n_components']}。",
                      f"- **主策略**：{p['family']} / {p['mode']}；首方向 {p['direction']}。驱动：{', '.join(pairtext(x) for x in p['drivers_map']) or '双端全几何，无固定距离驱动'}。",
                      f"- **观察键**：{', '.join(pairtext(x) for x in p['observers_map']) or '无额外编辑键'}。",
                      f"- **回退**：{p['fallback']}",f"- **拒绝风险**：{p['risk']}",
                      f"- **账目**：映射状态 {r['mapping_status']}；R/P 总电荷 {ep['reactant']['charge']}/{ep['product']['charge']}，来源多重度 {ep['reactant']['multiplicity']}/{ep['product']['multiplicity']}；人工化学复核未完成。",'',
                      f"RDKit 端点表示中的带电原子数：{ep['reactant']['rdkit_charged_atom_count']}→{ep['product']['rdkit_charged_atom_count']}；自由基电子数：{ep['reactant']['rdkit_radical_electron_count']}→{ep['product']['rdkit_radical_electron_count']}。这不是量化波函数的电子态诊断。",'',
                      '| 编辑 | map 对 | 键级 R→P | d(R) | d(P) | 跨组分侧 |','| --- | --- | --- | ---: | ---: | --- |'])
        for e in r['edits']:
            cross='/'.join(s for s in ['R','P'] if e[f'cross_component_{s}']) or '—'
            lines.append(f"| {e['edit_kind']} | {pairtext(e['pair'])} | {e['r_bond_order']}→{e['p_bond_order']} | {e['distance_R_A']:.3f} | {e['distance_P_A']:.3f} | {cross} |")
        if f['atom_attribute_changes']:
            desc=[]
            for a in f['atom_attribute_changes']:
                changes=', '.join(f"{k}: {a['R'][k]}→{a['P'][k]}" for k in a['R'] if a['R'][k]!=a['P'][k])
                desc.append(f"map {a['atom_map_id']} ({changes})")
            lines.extend(['','原子属性提示：'+'；'.join(desc)+'。'])
    lines.extend(['','## 溯源与复算','',f"- 输入候选 CSV SHA256：`{sha(source)}`。",f"- ReactionCase manifest SHA256：`{sha(manifest_path)}`。",f"- RDKit：`{rdBase.rdkitVersion}`。",'- 逐条导出哈希、案例哈希、支持路径、完整数值见同目录 `demo24_strategy_evidence.json`。','- `build_evidence.py` 可复算；只写本目录两个设计证据文件，不修改正式案例、不提交计算。',''])
    (OUT/'Demo24_逐条扫描策略建议.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(payload['counts'],ensure_ascii=False))
    print('Verified 24 source hashes, case digests, mapped graph diffs and driver/observer coverage partitions.')

if __name__=='__main__': main()
