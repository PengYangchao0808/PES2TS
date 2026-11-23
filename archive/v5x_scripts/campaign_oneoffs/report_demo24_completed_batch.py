"""Summarize completed execution separately from scientific qualification."""
from datetime import datetime
from collections import Counter
from pathlib import Path
import json
from run_demo24_synchronized import OUT,ROOT,PILOTS,read
from pes2ts_core.generation.planning.synchronized_path import digest
from pes2ts_core.integration.acp.scheduler_backend import ACPWorkbench,save_receipt
api=ACPWorkbench();receipt=read(OUT/'submission_receipt.json');quality=read(OUT/'quality_summary.json')
qrows={r['reaction_id']:r for r in quality['records']};records=[]
for rid,entry in receipt['jobs'].items():
    job=api.request('/api/v1/jobs/'+entry['job_id']);row=dict(qrows[rid]);row['job_id']=entry['job_id']
    row['started_at']=job.get('started_at');row['completed_at']=job.get('completed_at')
    row['execution_seconds']=(datetime.fromisoformat(row['completed_at'])-datetime.fromisoformat(row['started_at'])).total_seconds() if row['started_at'] and row['completed_at'] else None
    row['error']=job.get('error')
    plan=read(OUT/rid/'PathPlan.json');row['plan_revision']=plan.get('plan_revision',1)
    row['plan_sha256']=digest(plan)
    if row['plan_sha256']!=job['spec']['input']['selection']['pes2ts']['plan_sha256']:
        raise ValueError('Scientific identity mismatch: '+rid)
    if job['status']=='failed':
        logs=api.request('/api/v1/jobs/'+entry['job_id']+'/logs?lines=120')
        save_receipt(OUT/rid/'FinalFailedLogs.json',logs)
        row['failure_log_tail']=logs.get('stdout','').splitlines()[-4:]
    records.append(row)
remaining=[r for r in records if r['reaction_id'] not in PILOTS]
completed=[r for r in records if r['status']=='completed']
if any(not r.get('finite_energy_ok') for r in completed):
    raise ValueError('Completed batch still contains missing or non-finite physical energies')
summary={'batch_id':OUT.name,'project_id':receipt['project_id'],'n_unique_reactions':len(records),
         'status_counts':dict(Counter(r['status'] for r in records)),'remaining18_status_counts':dict(Counter(r['status'] for r in remaining)),
         'n_frames':sum(r.get('n_frames',0) for r in completed),'n_basic_geometry_qualified':sum(r['qualified'] for r in records),
         'remaining18_execution_seconds_sum':sum(r['execution_seconds'] or 0 for r in remaining),
         'n_exact_full_boundaries':sum(r.get('boundary_ok',False) for r in completed),
         'n_all_constraints_ok':sum(r.get('constraints_ok',False) for r in completed),
         'n_continuity_ok':sum(r.get('continuity_ok',False) for r in completed),
         'n_ts_near_ok':sum(r.get('ts_near_ok',False) for r in completed),'records':records,
         'reference_guided':True,'blind_prediction_claimed':False,'stationary_point_claimed':False}
save_receipt(OUT/'completed_batch_summary.json',summary)
lines=['# Demo24 多坐标同步：全批计算与验收结果（2026-10-02）','',
       '**计算执行状态与路径合格状态分别统计。** 本轮按用户继续计算的明确授权补算剩余 18 例，未以代表例质量门已通过的名义扩批。完整原始双端保留；每反应仅一个逻辑主任务。', '',
       f'- 全批 24 个身份：{summary["status_counts"]}；剩余 18 例：{summary["remaining18_status_counts"]}。',
       f'- 完成任务可读取真实结构、有限能量和前端节点，共 {summary["n_frames"]} 帧。',
       f'- 完成结果中：完整边界通过 {summary["n_exact_full_boundaries"]}；全部驱动残差通过 {summary["n_all_constraints_ok"]}；整体连续性通过 {summary["n_continuity_ok"]}；TS 联合接近度通过 {summary["n_ts_near_ok"]}。',
       f'- 同时满足本轮基础几何门槛：**{summary["n_basic_geometry_qualified"]}/24**。这是参考引导、GFN2-xTB 工程结果，不代表盲反应发现或高层级 TS 验证。',
       f'- 剩余 18 个任务最新执行版本时长之和：{summary["remaining18_execution_seconds_sum"]:.1f} 秒；含任务启动/文件处理，不含排队，也不包含单驱动适配修复前的错误尝试或先前六例多版修复成本；每任务申请 2 核，批内串行。',
       '- [ACP 前端](http://127.0.0.1:8765/)：选择项目 **PES2TS Demo24 — synchronized reference PES**。','',
       '门槛：完整首尾逐分量误差 ≤10^-6 Å；距离残差 ≤0.01 Å、角/扭转 ≤0.5°；整体相邻 proper-Kabsch RMSD ≤0.3 Å；同一帧整体和反应中心 TS RMSD、最大距离驱动误差均 ≤0.2 Å。未放宽先前冻结阈值。','',
       '| 反应 | 驱动/计划帧 | 执行状态 | 首尾 | 残差 | 连续性 | TS 邻近 | 基础几何验收 | 执行秒 |',
       '|---|---:|---|---|---|---|---|---|---:|']
for r in sorted(records,key=lambda x:x['reaction_id']):
    p=read(OUT/r['reaction_id']/'PathPlan.json')
    flags=['通过' if r.get(k) else ('未通过' if r['status']=='completed' else '未验收') for k in ('boundary_ok','constraints_ok','continuity_ok','ts_near_ok')]
    lines.append(f'| {r["reaction_id"]} | {len(p["drivers"])}/{len(p["lambda_values"])} | {r["status"]} | '+ ' | '.join(flags)+f' | {"通过" if r["qualified"] else "未通过"} | {r["execution_seconds"] or 0:.1f} |')
lines+=['','完成计算但未通过连续性或 TS 联合指标的路径，仍可用于 ACP 查看及算法诊断，不能计为目标路径成功。完整双向分支跟踪、固定双端联合路径修复、同方法线性多坐标对照与跨反应学习仍未完成。','',
        'RXN_0000155302 的原始 R/P 两个独立组分存在 C/Cl 重叠，距离为 0.534526/0.660185 Å。原始文件和完整边界未修改。本轮以 v8 恢复全部四个必要编辑驱动及已证实的 SCC 2000 迭代设置，移除失败版本的可选形状控制；计算状态见上表。无论计算是否收敛，该反应完整端点质量仍未通过，不能计为科学合格。', '',
        '本轮验收发现 ACP 的单驱动适配层未传递完整路径合同，导致 RXN_0000026256、RXN_0000077619 的第一次执行退回原生均匀扫描。已修复并在原身份中重算；旧错误结果单独归档。', '',
        'RXN_0000110869 初次运行第 24 号节点 SCC 失败，旧优化收敛标志却令任务报告 completed，留下缺失能量。已增加逐点有限能量成功门，并以相同完整边界、全部目标及 GFN2-xTB 方法增加 SCC 上限至 2000 后重算；最终全部 678 帧均有有限物理能量。新增合同回归 8 项通过，包括缺失能量不得判成功的负例。', '',
        f'- [完整指标与实际状态]({(OUT/"completed_batch_summary.json").as_posix()})',
        f'- [逐反应几何质量]({(OUT/"quality_summary.json").as_posix()})',
        f'- [原始双端检查]({(OUT/"endpoint_bond_audit.json").as_posix()})',
        f'- [仅供审阅的 RXN_0000155302 边界修正候选]({(OUT/"RXN_0000155302/boundary_proposal/proposal.json").as_posix()})','']
report=ROOT/'docs/reports/PES2TS_Demo24_多坐标全批计算结果_20261002.md'
report.write_text('\n'.join(lines).replace('/mnt/e/','E:/'),encoding='utf-8')
ledger_path=OUT/'execution_ledger.json'
if ledger_path.exists() and not (OUT/'execution_ledger_round1.json').exists():save_receipt(OUT/'execution_ledger_round1.json',read(ledger_path))
save_receipt(ledger_path,{'batch_id':OUT.name,'n_primary':24,'records':[
    {k:r[k] for k in ('reaction_id','job_id','plan_revision','plan_sha256','status','qualified')} for r in records]})
print(json.dumps({k:v for k,v in summary.items() if k!='records'},ensure_ascii=False,indent=2))
print(report)
