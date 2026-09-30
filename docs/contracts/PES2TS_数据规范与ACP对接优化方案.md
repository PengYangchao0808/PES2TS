# PES2TS 数据规范与 ACP 对接优化方案

日期：2026-09-29。状态：设计建议，尚未实施。

后续统一实施顺序与阶段验收见 [PES2TS 统一开发顺序与阶段验收](../plans/PES2TS_统一开发顺序与阶段验收.md)；本文保留为数据合同和 ACP 映射的详细依据。

## 1. 目标与依据

PES2TS 保留 PES generation 和 PES ranking 两个独立项目，通过版本化科学数据合同连接；ACP 承担计算任务、资源、结果展示和人工操作。推荐采用“PES2TS 科学数据包 + ACP 执行与展示适配层”。

本方案核对了 PES2TS 的双项目主方案、G1 计划方案及 v2_gate.py，并读取了本地 ACP 项目 `E:/Calculations/Common_Script/Auto_Calc_Platform/ACP_V1_20260811` 中以下实现：

- `src/acp/storage/manifest.py`、`layout.py`：v2 结果产品清单与 WORK/RESULT 布局。
- `src/acp/calculations/pes/contracts.py`、`outputs.py`：扫描请求、逐帧结果、能量曲线及推荐结果。
- `src/acp/results/frames.py`、`frame_candidates.py`、`energy_graph.py`：可视化帧与候选保存接口。
- `src/acp/scheduler/jobs.py`：计算任务状态。
- `AGENTS.md` 与 `docs/ACP_Energy_Trajectory_Viewer_DevDoc.md`：远程读取、任务组织与查看器约定。

这里确认的是本地源码的能力边界，尚未通过运行中的 ACP 服务验证。

### 已有能力与需补齐内容

| 内容 | ACP 现状 | PES2TS 对接方式 |
| --- | --- | --- |
| 结果入口 | `RESULT/result_manifest.json`，version=2 | 复用产品注册，科学合同放独立文件 |
| 路径帧 | ScanFrame 已含几何、扫描/SP 能量、收敛、约束残差、重试历史 | 转换为统一 PathBundle，增加反应、计划和稳定帧身份 |
| 展示帧 | TrajectoryFrame / TrajectoryAnnotation | 从科学数据投影，复用能量与结构联动 |
| 候选 | 通用帧保存与 PESsearch 专用复核是两条现有流程 | 按来源调用对应服务，保存 ACP 与 PES2TS ID 映射 |
| 自动推荐 | PESsearch 自动推荐作为报告；人工确认后才注册可复用结构 | 自动排序与人工确认分别记账；自动验证另建明确的评估适配入口 |
| NEB | 查看器注册占位，尚无数据投影 | 单独开发投影和几何解析，不能仅改 view_type 即宣称支持 |
| 远程结果 | 统一结果读取根、目录预取和几何懒加载 | 新数据包需登记小文件预取与几何解析；不能依赖浏览器访问磁盘路径 |

## 2. 七种核心数据对象

在主方案的五种合同上补充实验清单和执行记录。每类都有 schema、示例、验证器及版本迁移规则。

| 对象 | 主要内容 | 权威生产者 |
| --- | --- | --- |
| ExperimentManifest | 样本、split、候选计划、规则、预算、验证协议及冻结哈希 | PES2TS 实验管理 |
| ReactionCase | 原始 R/P 身份、原子映射、端点结构、组分、电子态、反应编辑与来源 | G0/G1 |
| ScanPlan | 候选树、扫描方向、驱动/观察坐标、完整点列、计算方法、回退与预算 | G1 |
| ExecutionRecord | 执行请求、ACP job/task ID、尝试、实际参数、状态、耗时、日志、错误 | ACP 适配器 |
| PathBundle | 路径元信息、逐帧结构引用、能量通道、路径质量和成本引用 | generation |
| SeedProposal | 路径可用性、拒绝原因、Top-k 帧、排序版本、分数与解释 | ranking |
| ValidationResult | OptTS、频率、目标模态、双向 IRC、端点匹配与成本 | 隔离评估器 |

附属对象：ArtifactRef（文件引用）、MethodSpec（计算方法）、ReviewRecord（人工复核）。人工修改形成独立版本，保留原始模型建议。

### 公共字段

每个科学对象必备 `schema_name`、`schema_version`、`object_id`、`created_at`、`producer`、`input_refs`、`content_sha256`。关联反应的对象还须有 `dataset_version`、`reaction_id`、`case_id`、`split`。

- producer 记录程序版本、提交版本；工作区有未提交修改时记录 dirty 标记和代码快照摘要。
- 时间统一 UTC ISO 8601；前端按用户时区显示。
- 核心字段使用严格 schema；扩展字段放命名空间化的 extensions，输入白名单同样覆盖扩展字段。
- 缺失数值使用 null 并给出原因；禁止 NaN/Infinity、用 0 代替缺失、用字符串填充数值列。
- 科学内容摘要排除自身摘要、生成时间和机器绝对路径；采用明确且带版本的规范序列化规则。文件 SHA256 单独计算，覆盖真实文件字节。
- 不支持的主版本拒绝读取；旧数据通过显式迁移器转为新版本，记录输入/输出摘要。

## 3. 身份、结构和能量规范

### 3.1 身份链

`experiment_id → case_id → plan_id / candidate_id → execution_id / attempt_id → path_id → frame_id → proposal_id → validation_id`

这是关联图：一个实验包含多个反应，一个计划有多个候选和尝试，同一路径可被多个排序器消费，同一帧可被多个 proposal 引用。

- reaction_id 沿用原数据集 ID，以 dataset_version 限定命名空间；反向扫描不改写原始 R/P 身份或 split。
- case_id 标识冻结的端点、映射、电子态和装配版本；同反应不同构象/装配可有不同 case。
- plan_id 标识完整冻结候选树，candidate_id 标识其中一个候选；修改计划生成新版本。
- path_id 标识一次已发布路径结果；补算或重算生成新版本，并用 supersedes 引用旧版本。
- frame_id 在路径版本内稳定。frame_index 固定为从 0 开始的存储顺序，界面另显示从 1 开始的 frame_number。
- 缺失或失败采样点保留 sample_id 和原因，不能删除后重排身份；同一点的多次优化尝试单独保留。
- ACP task_id、job_id 和尝试序号保存在绑定记录中，不能代替科学 ID，也不能通过任务名称反推身份。

### 3.2 结构与坐标

- atom_map_ids 为显式数组，不假定 map 连续；atom_index 从 0 开始，必须有 map→index 映射表。
- XYZ 的元素与原子顺序必须与 atom_map_ids 一致；XYZ 注释不能作为唯一的映射、电子态或来源记录。
- 长度使用 angstrom，角度使用 degree；原始格式换算记录来源单位。
- 扫描计划同时保存目标坐标与执行帧的实际坐标、约束残差；观察键与驱动键分开。
- 同步多键扫描使用无量纲进度 s，并保存每根驱动坐标的 s→目标值点列；实际键长不能替代 s。
- 原始反应方向、锚点侧、实际计算方向和展示方向分别记录；图形反转只能改变展示顺序。
- 分别记录体系和组分的 charge、multiplicity；必要时记录 spin_mode、初始猜测和波函数来源。不能把现有 multiplicity_max 当作整个计算体系的多重度。

### 3.3 能量

推荐将能量存为明确的通道：`scan_electronic`、`refined_electronic` 等；每个通道引用 MethodSpec。

MethodSpec 至少包含引擎及版本、方法、基组、溶剂、色散、电子态、数值设置与参数摘要。每个能量点关联实际几何哈希、通道、数值、单位和计算状态。

- 原始绝对电子能统一 Hartree；界面相对能可用 kcal/mol，必须记录 reference_frame_id、reference_energy、转换规则。
- ACP EnergyProfile 支持 mixed；PES2TS 不将不同层级的能量拼成可排序曲线。SP 缺失时保留空值，或按预先冻结策略切换到完整扫描能量通道，并记录切换。
- 跨路径比较需要满足相同组成、电子态、方法和能量定义，且采用可比较的参考。各路径独立归零后的峰高不能直接比较。
- 收敛未知不能默认为成功；导入 ACP 旧帧时显式检查，避免继承缺省 optimization_converged=True。

## 4. 文件布局与 ACP 产品注册

科学包使用 JSON + XYZ：JSON 保存结构化元信息和小规模逐帧记录，XYZ 保存几何。实验汇总使用可重建的 Parquet 索引；首轮 Demo 不引入额外的大型轨迹存储系统。

ACP 任务内建议新增以下产物布局，路径由适配器通过 ACP 布局接口生成：

```text
<ACP 分配的任务目录>/
  task.json                         # ACP 管理
  WORK/
    00_RUNTIME/                     # ACP 调度、日志和事件
    07_PATH/                        # generation 计算中间文件
    08_ANALYSIS/                    # ranking / 报告处理中间文件
  RESULT/
    result_manifest.json            # ACP v2 产品清单
    pes2ts/
      reaction_case.json
      scan_plan.json
      execution_record.json
      paths/<path_id>/
        path_bundle.json
        frames.json
        geometries/<frame_id>.xyz
      proposals/<proposal_id>.json
      reviews/<review_id>.json
    structures/                     # 经候选服务发布的可复用结构
    reports/                        # 展示报告与派生摘要
```

上图是产品布局集合；每项任务只发布自己拥有的对象。独立 ranking 任务引用冻结 PathBundle，可打包导入或通过解析器获取，无须复制 generation 工作目录。验证任务的 ValidationResult 存放在独立评估输出中。

ArtifactRef 使用 `artifact_id、relative_path、sha256、media_type、size_bytes`，包间引用额外记录 package_id。科学包引用相对包根；ACP Product.path 按现有规则相对 RESULT；适配器负责转换。禁止把 Windows 盘符路径当成跨机器数据合同。

| PES2TS 产物 | ACP Product.kind 建议 | 说明 |
| --- | --- | --- |
| 合同与索引 | file | 登记 schema 和摘要 metadata |
| 路径包 | trajectory | 新解析器完成后提供帧联动；单独登记 kind 不会自动产生可视化 |
| 排序与拒绝结果 | report | 保留算法来源和版本 |
| 已保存的候选 XYZ | structure | 通过对应候选服务发布，记录 source_frame_id |
| 人工复核和案例摘要 | report | 引用原始数据，不覆盖算法输出 |

不将通用 PathBundle 冒充为现有 PESsearch 的 pes_profile_v2。只有通过完整语义映射的受限扫描结果才可生成该格式；NEB 和其他路径方法走专用投影。

发布顺序：先写到临时位置 → 校验 JSON、几何、引用和摘要 → 发布不可变科学包 → 通过单一写入者更新产品清单。失败时不登记完整可用产物。运行中展示采用带 revision 的快照；ranking 只消费冻结版本。

## 5. ACP 任务管理与状态

建议 ACP 中设两个项目：PES generation 和 PES ranking；共用 experiment_id、reaction_id 关联。验证任务关联 proposal_id，评估权限单独管理。一个任务对应一个可独立重试的计划候选或排序运行；帧作为任务产物，避免为每帧创建顶层任务。

ACP 继续管理 queued/running/paused/completed/failed 等执行状态。科学结论使用独立字段：

| 维度 | 建议状态 |
| --- | --- |
| plan_status | ready / needs_review / rejected |
| path_quality | unchecked / usable / unusable / needs_review |
| ranking_decision | accepted / rejected / needs_review |
| validation_outcome | not_run / incomplete / passed / failed |
| artifact_availability | local / remote / pending_fetch / missing |

完成计算但路径不可用应同时呈现“任务完成、路径不可用”；远程等待拉取不能记为化学失败。验证的每个阶段还要分别记录未运行、通过、失败和原因。

ExecutionRecord 记录 requested/effective 参数、ACP ID、执行节点、尝试、重试来源、开始/结束时间、核数、壁钟秒数、实际或估计核时及计量来源。核时覆盖失败和回退；共享验证的真实成本只计一次，比较实验另给分摊规则。

提交使用 request_id 和输入/计划/程序版本摘要防止重复派单；网络超时后先查询已有任务。重试产生新 attempt；修改输入或预算策略产生新计划/运行。优先复制为新 ACP 任务；原地重算前必须把旧科学包和执行证据完整归档。

规划层检查 ACP 实际能力。当前 PES 合同声明扫描点数 3–101、同步坐标最多 4；这只是该接口的边界，不是所有后端通用能力。超出能力应提前报错或选择已冻结回退。

## 6. 可视化最小范围

### 第一批：直接帮助开发和复核

1. 反应概览：R/P、编辑键、组分/电子态、split、映射来源、计划方向。
2. 路径详情：能量—s 曲线、逐帧三维结构、驱动/观察键距离、收敛与约束残差；点击曲线点定位同一 frame_id。
3. 排序对照：同一路径上的规则推荐帧、Top-k、分数与理由；分数与置信度分开，没有校准就不显示为成功概率。
4. 实验列表：各反应停在哪一步、拒绝/失败原因、累计成本；按机制、方法、split、映射来源筛选。

### 第二批：评估侧展示

OptTS 优化轨迹、频率及目标虚频振型、IRC 双分支、端点匹配证据，以及完整成功漏斗。图中的“TS 初猜”和“已验证 TS”使用不同标签。

ACP 投影必须通过 TrajectoryFrame.to_node() / TrajectoryAnnotation.to_annotation()；保留 native_frame_id↔PES2TS frame_id 映射。几何通过统一解析器读取，远程走现有结果缓存，页面先获取摘要、按需加载 XYZ。

能量查看器只负责查看与保存候选；创建计算任务仍走 ACP 统一新建任务入口。PESsearch 候选沿用专用复核服务。若需要无人值守 Top-k 验证，应新增评估适配流程，记录 selection_source=ranking，不伪装成人工确认。

## 7. 数据隔离、迁移和验收

### 数据隔离

- 首先修复 G1 导出中 status_summary.endpoint_match 和 orientation 等 TS/IRC 派生字段，同时更新验证器。
- 生产端使用正向白名单；隔离评估文件、索引与 API 权限，不能只靠前端隐藏字段。
- truth_assisted_p1 的映射来源必须保留在审计记录中；与端点独立映射分组评价。筛选群体受 IRC 证据影响的情况也需记录，删字段不能消除样本选择偏差。
- ranking 推理输入只来自部署时可获得的字段；训练标签只通过显式训练集导出产生，valid/test 结果不回写推理数据包。

### 实施顺序

| 阶段 | 工作 | 验收条件 |
| --- | --- | --- |
| P0 合同冻结 | 七类对象、字段/单位/ID/状态字典、schema 与合成示例 | 不运行 QC 即可验证；错误索引、缺能量、未知版本和真值字段被识别 |
| P1 现有数据迁移 | G1 白名单修复，旧导出转换，迁移清单和独立读回检查 | 反应数与 split 对账；旧文件保留；未知值显式记录 |
| P2 ACP 单反应对接 | 一条已复核 A/B 反应的计划提交、结果收集、PathBundle 发布 | 本地/远程引用可解析，任务和科学状态分开，重试不覆盖旧证据 |
| P3 ranking 与展示 | 独立读取冻结包，规则排序，能量/结构/选点联动 | generation 无须重算；相同 frame_id 的坐标/哈希贯穿全链 |
| P4 24 条 Demo | 批量管理、失败报告、同预算验证和成本对账 | 每条反应有终态；覆盖拒绝、部分结果、超时与缓存重复消费 |

优先测试：原子映射错位、能量混用、损坏或越界引用、部分路径、反向扫描、重复提交、旧结果被重算覆盖、远程目录缺失、人工复核版本冲突、同帧验证重复计费及真值泄漏。

最终验收采用“一条反应全程可追溯”：从 ACP 展示的任一候选帧回到原始端点、冻结计划、具体计算尝试、原始日志、能量方法、排序版本和验证结论。汇总索引及界面数据必须可由这些权威记录重建。

## 8. 第一轮开发范围建议

第一轮交付：七类合同及校验器、一个 ACP 适配模块、单反应可回放示例、路径与候选联动页面。以此通过接口验收后，再扩到 24 条 Demo。现阶段不必同时拆仓库、训练模型或重做 ACP 调度与三维查看器。

建议代码落点为 PES2TS 的 contracts/、integration/acp/、evaluation/；ACP 侧补充产品解析、视图投影、结果缓存目录登记和必要的任务来源元数据。若注册新工作流，还需一并补齐 catalog、CLI、参数编辑覆盖、远程同步和测试。
