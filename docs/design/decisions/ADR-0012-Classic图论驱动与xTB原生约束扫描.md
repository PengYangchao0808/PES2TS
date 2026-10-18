# ADR-0012：Classic 图论驱动与 xTB 原生约束扫描

> 状态：Accepted（用户指定的设计方向；代码、正式schema、真实能力及默认切换待验收）
> 层级：G1/G2；G2-AB3；工作流 C/R/X
> 取代关系：细化 ADR-0006/0010/0011 的控制/原生优化分工；本批优先 ACP/xTB 原生扫描，ORCA方案保留为独立适配方向；不改历史配方与验收
> 上游依赖：[开发宪法](../../PES2TS_开发宪法_v1.md)、[ADR-0002](ADR-0002-计算后端统一经ACP执行.md)、[ADR-0010](ADR-0010-PES-N反应状态价值与STOP-MOVE统一决策.md)、[ADR-0011](ADR-0011-三模块产品架构与统一可视化.md)
> 下游消费者：[Classic设计](../PES2TS_Classic图论驱动与xTB原生约束扫描设计_v1_20261006.md)、[AB3实施方案](../../plans/PES2TS_G2_AB3功能整合与原生扫描改造方案_20261006.md)、ACP配套、规则/执行/数据集成
> 当前版本：v1 @ 2026-10-06

## 背景与变更声明

用户希望 Classic 按“化学图论约束→每次只移动关键成键原子→xTB受约束扫描→重复覆盖规定范围”运行。AB2 当前实现使用全Cartesian投影/BFGS、反复ACP/ORCA梯度；24例中仅3条完整路径、19条前缀、2条拒绝，没有严格验证成功。

本决策触及新算法/默认方法的未来迁移、执行与候选观测接口（P1）、能力/费用表达（R3/R5）、共享数据合同。当前只接受设计；不改变红线、真值隔离、模块保护目录、阶段出口或已冻结运行。规范引用宪法§2.5/§5.2–5.4/§9.6；能力证据仍分implemented/fixture_passed/smoke_passed。

## 决策

1. 以薄门面整合拓扑约束、控制器、ACP计算器接缝，保留现有模块目录；独立化学规则前检/后检不归入引擎或模型。
2. 默认设计仅由PES2TS主动修改F/B driver涉及的K原子作为初猜，非K初猜保持原坐标；xTB原生优化松弛全部未固定自由度。全冻结环境是单独模式，不以势约束假装精确固定。
3. Classic推低维target，ACP/xTB负责真实局部优化；首版逐点调用公开ACP操作，化学接受后才推进父结构。现有ACP批量回调不具有接受握手，不能直接替代此循环。
4. xTB软约束允许实际值偏离目标，接受条件包含收敛、坐标残差与化学质量。结构文件存在不足以证明优化成功；内部优化梯度也不证明无约束驻点。
5. 能量分physical/effective/restraint；无法可靠分离时用同几何无额外势ACP单点获取physical能量。规则候选支持缺梯度观测，不零填；TS由既有严格链验证。
6. 扫描完全仅定义为有限坐标区间/规定加密点覆盖；1D协同扫描不宣称完整高维PES/MEP。旧全原子BFGS显式保留对照，原生模式无隐式fallback。
7. 缓存/恢复/费用以hash绑定的ACP收据为依据，失败/补点/额外观测全部留数。内部评估计数缺失为null+reason，每点一次任务不写成一次梯度。
8. 原生新分支先离线→真实接缝→train/calibration四例冻结→24例与严格验证；生产默认切换单独依据证据裁决。未来PES-N复用相同规则/执行接口，不在本批加入模型。

## 接口草案与兼容

登记待实施名称：`pes2ts_chemical_control_spec_v1`、`pes2ts_native_scan_plan_v1`、`pes2ts_native_constraint_request_v1`、`pes2ts_native_constraint_outcome_v1`、`pes2ts_native_scan_trace_v1`、`pes2ts_native_scan_seed_proposals_v1`。字段要求见设计§7。它们细化既有ChemicalControlSpec、Action/Transition/DecisionTrace、ExecutionBackend及候选合同族；尚无正式发布schema或实现证据。

历史GenerationPlanV2、PathBundle/SeedProposal/ValidationResult主链、`g2_path_v1`、AB2 manifest及报告保持可读和原义。新方法和旁车显式版本化、绑定父hash；实施时补注册validator/fixture。旧报告的审计缺陷先发新版本修正，不能原地改写以改善对照。

## 证据、同步与回退

设计证据：当前PES2TS与ACP源码定位、AB2冻结产物及既有2026-10-06审核、xTB官方扫描/优化/固定/能量文档。ACP底层能力存在不等于本接缝smoke_passed；公开单点入口、收敛证据、能量与计量仍待补齐和真实验收。

同步设计、AB3方案、宪法接口草案指针、AGENT当前进度指针、根README、docs索引和修订台账。宪法规范内容保持v2.1；没有新增实际schema、改生产代码、迁移目录、启动QC或修改默认。

未达到能力/质量/成本门时原生路线保持显式实验资格；旧配方继续按原证据使用。显式fallback另有attempt及费用，新旧运行根分离，不覆盖历史产物。
