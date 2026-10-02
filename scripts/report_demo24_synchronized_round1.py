"""Produce an evidence-backed first-iteration report, including unfinished gates."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/PES2TS_Demo24_synchronized_reference_20261002'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
quality=read(OUT/'quality_summary.json');status=read(OUT/'batch_status.json');manifest=read(OUT/'batch_manifest.json')
states={r['reaction_id']:r for r in status['records']}
endpoint=read(OUT/'endpoint_bond_audit.json')
lines=['# PES2TS 多坐标同步路径：第一轮改造与实测报告（2026-10-02）','',
'## 结论','',
'完成了第一轮工程改造、24 份初始冻结计划、真实 ORCA 能力探针与六例 ACP 试算，并进行了多版修复。**尚未完成实施方案 S0–S6 的全部退出条件，也未执行新版 24 条正式全批。** 本报告不将测试通过、任务完成或已知 TS 附近的参考引导结构称为盲预测成功。','',
'最新 ACP 状态：5 个代表任务 completed，1 个 failed；178 帧完成结果可读取。完成的 5 条均保持完整原始首尾并通过全部驱动残差检查、存在同时满足整体/反应中心/编辑键 TS 接近度门槛的点；其中只有 3 条通过本轮冻结的整体相邻 RMSD 门槛。其余 18 个反应尚未提交。','',
'ACP：[打开现有前端](http://127.0.0.1:8765/)；项目 **PES2TS Demo24 — synchronized reference PES**。',
f'项目 ID：`{quality["project_id"]}`。旧单键项目和 394 帧结果仍保留。','',
'## 实际改动','',
'- 新路径合同：每反应一个 logical_primary_id、parameter_dimension=1；完整 R/P、原子顺序、电子态、逐点 lambda/values、事件覆盖、教师来源及哈希。初始 24 个反应包含全部 115 个图编辑（成断键 71、键级变化 44），每份初始计划有 1–8 个距离驱动。一个驱动的少数反应仍然只有一个必要编辑，未人为增加化学事件。',
'- 将旧 primary_stretch 标记为 legacy；新批次使用独立同步入口。主候选选择允许多驱动，并检查成断键和键级变化覆盖。主路径数量与驱动数量分开。',
'- ACP ScanCoordinate/CoordinateSpec 支持显式 values；逐点 ORCA Constraints 消费全部目标，支持非均匀公共 lambda、B/A/D 和超过 4 个驱动。保留旧均匀单坐标接口；新接口保留 64 个驱动、40 帧的工程预算上限，这不是 ORCA 原生 Simul_Scan 的上限。',
'- 修复原生平均最小步长误拒绝非线性/近等双端值，以及将任意多坐标误路由到“双键、四原子”选择器的问题。结果恢复优先使用显式 values；扭转残差周期处理。',
'- 首尾采用同方法、完整固定几何单点；内部为多约束松弛优化。frame_role 区分 fixed_boundary_single_point 与 constrained_optimization；端点 optimization_converged=false，成功单点不再被误报为优化失败。',
'- 参考读取在明确授权、allow_truth/caller 审计入口，核对 TS/IRC 原子对应和方向。盲规划数值模块不读取真值。原始 R/P 不被 IRC 端点替换。',
'- 增加约束 Jacobian 秩、周期残差、共同几何见证、完整边界、整体/重原子/反应中心连续性、TS 联合接近度的检查。提供质量加权弧长及最小范数 Jacobian 预测器并单测。',
'- 修复目标表独立插值导致的联合不可实现性：所有插入节点的目标从一个共同完整几何测得。新增控制明确标为构象/组分姿态控制，不冒充成断键事件。保留 rank-increasing 选择与有限松弛自由度。',
'- 代表例修复用实际跳跃定位公共节点加密、R/P 联合图角/扭转控制，以及近线性位点的几何距离控制。所有修复在原六个任务身份中执行，科学计划保存 v1–v7 中实际存在的版本，不生成新增主反应。','',
'## 六例最新结果','',
'门槛：完整端点最大逐分量误差 ≤10^-6 Å；全部距离残差 ≤0.01 Å，角/扭转 ≤0.5°；整体相邻 proper-Kabsch RMSD ≤0.3 Å；同一帧整体 TS RMSD、反应中心 TS RMSD、最大编辑/控制距离误差均 ≤0.2 Å。它们是本轮工程门槛，尚非充分的机理正确性认证；没有事后放宽失败阈值。','',
'| 反应 | 最新版本 | 驱动/帧 | ACP 状态 | 首尾最大误差 Å | 最大相邻整体 RMSD Å | 最大归一化残差 | TS 联合接近 | 基础几何门 |',
'|---|---:|---:|---|---:|---:|---:|---|---|']
ledger=[]
for r in quality['records']:
    rid=r['reaction_id'];p=read(OUT/rid/'PathPlan.json');s=states[rid]
    ledger.append({'reaction_id':rid,'job_id':s['job_id'],'initial_plan_sha256':next(x['plan_sha256'] for x in manifest['records'] if x['reaction_id']==rid),
                   'current_plan_sha256':hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
                   'plan_revision':p.get('plan_revision',1),'status':s['status'],'qualified':r['qualified']})
    if r['status']=='completed':
        boundary=max(r['boundary_errors_angstrom']);step=max(r['step_rmsd_angstrom']);err=r['max_normalized_constraint_error']
        lines.append(f'| {rid} | {p.get("plan_revision",1)} | {len(p["drivers"])}/{r["n_frames"]} | completed | {boundary:.2e} | {step:.4f} | {err:.5f} | 是 | {"通过" if r["qualified"] else "未通过"} |')
    else:
        lines.append(f'| {rid} | {p.get("plan_revision",1)} | {len(p["drivers"])}/{len(p["lambda_values"])} 计划 | {s["status"]} | — | — | — | 未验收 | 未通过 |')
lines += ['',
'前三个基础几何通过反应为 RXN_0000007104、RXN_0000047010、RXN_0000109608。RXN_0000017762 与 RXN_0000187964 最新虽有正确首尾、全部目标残差和 TS 邻近点，仍存在局部跳跃。距离控制与接近满秩的形状控制并不能排除反射/近线性分支及剩余松弛自由度的不连续；不能用更多约束数量代替路径连接证明。','',
'RXN_0000155302 曾在增加 SCC 迭代后完整完成 27 帧，但连续性未通过；最新增加构象约束的 v5 失败，当前 ACP 状态为 failed。不得把此前已完成的旧版本当作最新版本合格结果。','',
'## 原始边界的数据问题','',
'对全部 24 例采用保留显式映射氢的图解析：24/24 两端组分内成键距离通过宽松 sanity screen；23/24 完整边界通过组分间近接触检查。RXN_0000155302 的两个独立组分重叠，非成键 C(map 5)/Cl(map 6) 间距在 R/P 为 0.534526/0.660185 Å。此前“组分内几何保持”不能证明完整边界的物理有效性。',
'原始 P 的 GFN2-xTB 单点 SCC 默认迭代失败；2000 次迭代探针在完全不移动几何的情况下收敛，能量 −29.35423133486 Eh。增加迭代不会消除异常近接触。迭代参数依据 [ORCA xTB 接口文档](https://www.faccts.de/docs/orca/6.1/manual/contents/modelchemistries/semiempirical.html) 的 XTBINPUTSTRING 传递，计算方法未改变。','',
'另行生成 **仅供审阅、未用于计算** 的 R/P 组分摆放候选：每个组分整体对齐至已验证 IRC 端点参考，组分内所有两两距离最大变化 <5×10^-15 Å，组分间最小距离 R/P 为 2.518048/2.893590 Å。这会改变整体边界：相对原始全结构 proper-Kabsch RMSD 为 1.138836/1.206983 Å。因此未擅自替换冻结边界，不能称为“原始完整首尾完全一致”。','',
'## 验证证据','',
'- PES2TS 默认全量：1375 passed、1 skipped、10 deselected（PES 环境缺 ACP 依赖导致 ACP 适配测试整模块跳过，随后在 ACP 自身环境独立执行）。',
'- ACP 新合同定向测试：6 passed，覆盖五驱动逐点目标、混合 B/A/D、非单调近等双端值、无效目标表、固定边界 API 语义和任意多坐标路由。',
'- 最新核心/选择/任务身份定向：19 passed，包含错误末帧拼接、漏控必要键、NaN/重复 ID、周期扭转、Jacobian 同步预测、共同弧长和不相容几何见证负例。',
'- ACP 约束门、实时轨迹和 ORCA 输入控制回归：67 passed。',
'- 真实 ORCA 6.1.1 / GFN2-xTB：混合 B/A/D 与五距离两种非线性公共 lambda 探针均通过；距离最大误差 <10^-6 Å、角/扭转最大误差 <5×10^-5°。这是能力执行证据，不能代表任意体系所有组合均可解。',
'- 已实际读取五个 completed 任务的 energy-graph、s2/profile 和全部逐帧 XYZ：当前 178 帧均有有限能量，节点数量匹配。首末帧类型通过 API 读取；failed 任务保留日志。','',
'## 与实施方案的对应：仍需完成什么','',
'S0–S3 的首轮工程路线与能力探针已落地；S2 的独立坐标选择仍属参考引导试验，不是已训练的机理推荐模型。S4 尚未完成：当前真实执行为参考几何种子 + 逐点硬约束优化；数值 Jacobian 预测器和质量加权弧长已单测，但尚未贯通为完整双向预测—校正执行。未实现固定双端联合路径修复、邻点分支跟踪或物理能量/梯度的完整连续性认证。不能把现有逐点松弛解释成它们。','',
'S5 完成六例真实试算与修复诊断，尚未满足六例全部退出条件，也未完成同方法、同多坐标的线性目标对照。旧单键基线无法代替这个对照。S6 正式扩展到 24 被程序质量门阻断；不得在这些条件未过时将试算宣布为正式合格全批。S7 跨反应学习与 S8 GSM/NEB 对照未开展。','',
'下一实现重点是稳定的双向分支跟踪与固定双端联合修复，处理近线性/反射歧义及局部松弛与边界连续性的冲突；物理能量与额外路径正则项分开保存。与此同时，需要明确 RXN_0000155302 的正确完整边界来源。当前原始整体边界有组分重叠，新的合理组分摆放候选尚待确认，原始文件保持不变。','',
'## 可复核文件','',
f'- [初始 24 份批次清单]({OUT.as_posix()}/batch_manifest.json)',
f'- [六例质量表]({OUT.as_posix()}/quality_summary.json)',
f'- [全部双端几何审计]({OUT.as_posix()}/endpoint_bond_audit.json)',
f'- [当前科学计划版本与任务身份]({OUT.as_posix()}/execution_ledger.json)',
f'- [边界修正候选说明]({OUT.as_posix()}/RXN_0000155302/boundary_proposal/proposal.json)',
f'- [R 候选 XYZ]({OUT.as_posix()}/RXN_0000155302/boundary_proposal/R_proposed.xyz)，[P 候选 XYZ]({OUT.as_posix()}/RXN_0000155302/boundary_proposal/P_proposed.xyz)',
f'- [真实后端能力探针]({ROOT.as_posix()}/outputs/acp_synchronized_capability_20261002/receipts.json)',
f'- [同步合同与质量门]({ROOT.as_posix()}/pes2ts_core/generation/planning/synchronized_path.py)',
f'- [新批次入口]({ROOT.as_posix()}/scripts/run_demo24_synchronized.py)',
f'- [ACP 修改原文件备份]({ROOT.as_posix()}/outputs/acp_synchronized_patch_20261002/manifest.json)',
'',
'ACP 源码改动位于外部 ACP checkout；逐文件备份在 PES2TS 输出目录，保留之前的 ORCA 线程修复。未提交 Git commit/PR，未改动已有 .omo 笔记，未覆写原始 fixture 或旧单键输出。']
(OUT/'execution_ledger.json').write_text(json.dumps({'batch_id':manifest['batch_id'],'initial_n_primary':24,'submitted_logical_primaries':6,'records':ledger},indent=2))
report=ROOT/'docs/reports/PES2TS_多坐标同步第一轮实现与实测_20261002.md'
report.write_text(('\n'.join(lines)+'\n').replace('/mnt/e/','E:/'),encoding='utf-8')
print(report)
