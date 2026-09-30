# PES2TS contracts v1 与 ACP v2 字段映射

本地 ACP 源码快照核对自 `ACP_V1_20260811` commit `4870bfb22da93eefa49756c7a56addd15bfa1f4c`；PES2TS 适配器输出已通过该版本 `PesScanRequest.from_dict` 和 `validate_scan_coordinates` 静态校验。2026-09-29 运行时探测到 ACP Workbench v0.1.3 在线，ORCA 6.1.1 和 xTB 6.7.1 的扫描能力可用；首个 9 点计划生成的 `pes_scan` method levels 也通过运行时 `/api/v1/validate-method` 校验（无错误/警告）。尚未提交任务，24 条端点化学复核仍是 S3 计算前置条件。

本次原生轨迹投影复核的当前 checkout 为 `E:/Calculations/Common_Script/Auto_Calc_Platform/ACP_V1_20260811`，branch `feat/task-organization`，HEAD `d8781de7310213058921752f1d56936a4146b9fe`。它不同于上述 9 月 29 日静态校验快照；每份 `acp_trajectory_graph.json` provenance 另记录加载到的 `src/acp/results/frames.py` SHA256，避免仅凭目录名假定接口版本。

## 身份及目录

| PES2TS | ACP | 规则 |
| --- | --- | --- |
| `experiment_id`, `reaction_id`, `case_id` | task metadata / project metadata | 显式保存，不从任务标题反推 |
| `plan_id`, `candidate_id`, `request_id` | task metadata / idempotency key | 一个候选尝试一项可重试任务；重算创建新 attempt |
| `execution_id`, `candidate_id`, `request_id`, `acp_task_id` | execution record ↔ task id | 双向留存；每次尝试的状态、失败码和 CPU 秒数分别记账；ACP ID 不能替代科学 ID |
| `path_id`, `candidate_id`, `frame_id` | product metadata / native frame id | PathBundle 记录实际执行的 candidate_id；保留 `native_frame_id` 与 PES2TS `frame_id` 对照 |
| ArtifactRef 相对包路径 | `Product.path` 相对 `RESULT` | 发布写 `RESULT`；原始输入、日志及临时计算留在 `WORK` |
| 产品清单 | `RESULT/result_manifest.json`, `version=2` | 使用 ACP 单一写入者注册，失败/未完成产物不得宣称可消费 |

## ScanPlan → ACP scan request

| ScanPlan | ACP request projection |
| --- | --- |
| `candidate.method.engine/method/basis` | scan driver / optimizer method / basis；同时完整映射支持的色散、溶剂模型/溶剂、grid、SCF 和 RI 字段 |
| 端点 `charge`, `multiplicity` | 整体计算电子态；明确使用扫描起始端点 |
| 按 `atom_map_ids` 排序的 elements/geometry | `PesScanRequest.source` 的 `xyz_text`、charge、multiplicity；map IDs 留在并行的 PES2TS task metadata |
| 单距离坐标 `atom_indices` | `ScanCoordinate(kind=distance, atoms=[i,j])` 的 ACP 0 基下标 |
| 均匀 `points` + `unit=angstrom` | `ScanCoordinate.start/end/n_points`；非均匀列表和多维结构不静默折损，提交前拒绝 |
| 扫描方法/引擎 | `protocol.scan_optimizer.method` 与 `protocol.scan_driver.software`；本计划为 GFN2-xTB 优化、ORCA 扫描执行 |
| 坐标、重试和逐点优化设置 | 并列投影到 ACP `pes_scan` 的 `method_levels`：`scan_coordinate`、`scan_driver`、`scan_optimizer`；retry count/reuse 不使用隐式默认值 |
| request/candidate/plan IDs 及 budget | ACP task metadata/幂等键及资源预算；当前投影把它们列为 metadata，调用方需交给 ACP task creator |

`scan_plan_to_acp_job_payload` 根据该预检组装 ACP V1 `PESsearch` 的 job-create body；它只生成请求，不提交任务。调用方必须显式传入 `resources`。反应/计划/候选/请求身份及 CPU-hour/wall-time 预算写入 ACP tags 与 remark；job-create resources schema 未暴露等价的 CPU-hour/wall-time 硬上限，因此这两项当前留在 PES2TS 执行记录中，后续需另行实现 ACP 侧限额控制。`output_dir` 不由适配器指定，ACP 继续管理任务目录和 `WORK/RESULT`。

按现有扫描接口规划的能力上限是每坐标 3–101 点、最多 4 个同步坐标；本首版投影仅支持一维均匀距离坐标。请求对象通过本地 `PesScanRequest` 静态校验，`pes_scan` method levels 已经 ACP `/api/v1/validate-method` 运行时校验；提交前仍需重新核对 capability 与预算。

`acp_execution_to_record` 接收 ACP reader 归一化后的 task/attempt 状态，不负责轮询或提交。它保留 attempt 失败码和成本；成本缺失时 attempt 与总成本写 null、`cost_complete=false`。未知任务状态、候选身份错配和越界 WORK/RESULT 引用会被拒绝。ACP 原生状态字段的采集/轮询器和实际 job submit 仍属于运行时适配工作。

## ACP result → PathBundle / viewer

| ACP result | PES2TS |
| --- | --- |
| 扫描帧原生 ID | `source_acp_frame_id`；无稳定 ACP ID 时按执行 ID 与零基帧序生成 `frame_id` |
| 帧原子顺序、元素与 XYZ | `atom_map_ids`, `geometry`, 原子/映射一致性检查 |
| 扫描能量及实际方法 | `scan_electronic`, Hartree, `method_id` |
| 精化单点能及实际方法 | 独立 `refined_electronic` 通道；缺失时为 null/缺项，不填 0 |
| 收敛状态 | `converged`; 未知保留 null |
| 完成/失败及 ACP task ID | `ExecutionRecord`，与 `PathBundle.path_quality` 分开 |
| attempt 状态/失败码/CPU 秒数 | 逐尝试保留；总 CPU 秒为可用尝试成本之和。任一成本缺失时记录 null 并将 `cost_complete=false`，不得按 0 计费 |
| `TrajectoryFrame` / annotation | 由 `frame_id` 投影；排序解释作为 annotation，候选结构哈希对应选中帧 |

能量曲线一个 channel 内必须使用同一方法和单位。扫描能量与精化能量不可拼成一条可排序曲线。实际扫描方法从 ACP 帧的 `optimizer_level` 与 `optimizer_engine` 读取，不用计划方法回填；缺少实际方法时保留能量值但 `method_id=null`，质量检查不会将该能量通道判为可比。适配器把计划方法写入 provenance；非空但未映射的方法参数会在请求导出时拒绝。`WORK` 原始文件通过校验后，科学包等可消费文件放 `RESULT` 并由 ACP v2 清单登记；跨机器引用只用包相对路径及文件 SHA256。

`verify_acp_result_manifest_files(manifest, result_dir)` 是只读消费端核对器：要求每个 Product 带 `metadata.sha256`，检查相对路径不越界、文件存在、摘要一致及可选 `size_bytes`，返回不含本机路径的核验摘要。它不写入或替代 ACP 的 `result_manifest.json` 单一写入者。

`assess_scan_path_quality` 把 ExecutionRecord、冻结 ScanPlan 与 PathBundle 的身份关联后，判断任务状态、帧数量、目标坐标、实际几何距离、约束残差、原子重叠、逐帧收敛和扫描能量可比性。只有完整覆盖冻结坐标、实际距离满足容差、逐点收敛且同方法/单位的扫描能量才标记 `usable`；缺帧或数据不完整为 `needs_review`，错位/违反约束/原子碰撞的帧或失败且无帧为 `unusable`，活动任务为 `unchecked`。报告保留三份输入摘要；`apply_scan_path_quality` 将该技术状态写入重新封存的 PathBundle，并同时登记输出路径包摘要。这不代表 TS/IRC 的物理验证。

## 状态与来源

执行状态 (`ExecutionRecord.status`)、路径可用性 (`PathBundle.status`)、排序决定 (`SeedProposal.status`) 和物理验证 (`ValidationResult.status`) 各自独立。SeedProposal 回显 `path_status`；只有 `usable` 路径的规则建议可标为 `accepted`，待审路径保留帧建议但 proposal 为 `needs_review`，`unusable` 路径不得产生候选帧。`selection_source` 必须为 `ranking`、`human` 或 `validation` 之一；自动推荐、人工确认和验证结果不得混作一种来源。合成示例的验证状态为 `not_run`。

ACP `ScanFrame` 的 frame index、target/actual coordinate、Hartree energies、convergence、retry history 可投影回 PathBundle。XYZ 几何通过注入的 ACP 几何加载器读取，所以远程缓存策略仍由 ACP 侧提供。`ResultManifest` 输出采用其 `version/task_id/workflow/status/products[{id,label,path,kind,metadata}]` 原生形状；`Product.path` 必须相对 RESULT。真实调度提交、产物清单落盘及 ACP 原生查看器联动仍需 S3 运行时冒烟验收。

`pes2ts_core.integration.acp.trajectory.project_path_bundle_to_acp_graph` 使用给定 ACP checkout 中真实的 `TrajectoryFrame.to_node()`、`TrajectoryAnnotation.to_annotation()` 和 `view_spec("scan")` 生成响应形状的图投影；`path_bundle_to_acp_pes_profile` 生成可由 ACP 正式 PESsearch builder 消费的 `pes_profile_v2`。`pes2ts project-acp` 可从 PathBundle、ScanPlan 与一个或多个 SeedProposal JSON 同时导出图 JSON 和可选 profile JSON。节点身份映射保留 PES2TS `frame_id`，provenance 留有 0 基 frame index→frame ID 映射和 ACP frame-contract SHA256。扫描能、相对扫描能、精化单点能、相对精化单点能是独立 series；缺能量保留 null。候选建议核验 PathBundle 内容摘要和帧几何摘要后投影。profile 只有携带安全的 RESULT 几何引用时才提供 `geometry_path`，不把 WORK 路径提升成正式结果。

生成的图字段已通过 ACP `EnergyGraphResponse.model_validate`。其中 `geometries` 是 PES2TS 侧的几何交接表，不是 ACP HTTP response 字段；只有先将 XYZ 写入 ACP 任务的 `RESULT` 并登记正式产品，再为图节点设置 `RESULT/...` `geometry_ref`，Workbench 才能通过其解析器/远程缓存显示这些结构。此材料级验证不等同于调用活跃任务的 `/jobs/{job_id}/energy-graph` 路由。

`pes_profile_v2.json` 通过 ACP `build_pes_energy_graph` 生成 6 节点/6 标注，并通过 `EnergyGraphResponse` 响应模型验证。要走真实 HTTP 路由，ACP 仍须先存在关联的 PESsearch scheduler job；调用方需由 ACP 结果写入层把 profile 安置并登记为 `RESULT/pes_search/pes_profile.json`，PES2TS 不直接写 ACP 任务目录或替代 ACP manifest writer。

结果回收适配器 `acp_s2_profile_to_path_bundle` 接受 ACP `GET /jobs/{job_id}/s2/profile` 的响应，以及逐帧 `GET /jobs/{job_id}/s2/frame/{frame_index}` 经 ACP 几何解析器得到的坐标数组。它核对 ACP job ID、帧序号连续性和帧/几何索引一一对应，保留独立扫描能/单点能、收敛标志、坐标和安全 RESULT 引用，并默认将路径标为 `needs_review`。它不会把 ACP job 完成或 `stationary_point_claimed` 字段提升为路径可用或 TS 验证通过。

`pes2ts acp-request` 可从 ready `ReactionCase`、ready `ScanPlan` 和显式资源 JSON 导出 ACP V1 `/jobs` 请求体。`scan_driver.software` 由冻结的候选方法引擎映射；当前 ACP 一维扫描映射只接受 `orca` 与 `xtb`，避免引擎和优化方法错配。传入 `--acp-root` 时还会调用该 checkout 的 `V1JobCreateRequest`、`BondLengthScanJobInput`、`PesScanRequest` 和 `validate_scan_protocol` 原生校验；该命令只写请求 JSON，不提交任务。当前合成演示的 `acp_job_request.json` 已由 ACP checkout 原生校验通过。复现命令：

```powershell
python bin/pes2ts acp-request `
  --case data/interim/pes2ts_synthetic_demo_v1/ReactionCase.json `
  --plan data/interim/pes2ts_synthetic_demo_v1/ScanPlan.json `
  --resources data/interim/pes2ts_synthetic_demo_v1/acp_resources_example.json `
  --acp-root E:/Calculations/Common_Script/Auto_Calc_Platform/ACP_V1_20260811 `
  --output data/interim/pes2ts_synthetic_demo_v1/acp_job_request.json
```

真实 ACP checkout 的可复现合成冒烟命令（输出仍是合成路径，不提交计算）：

```powershell
python bin/pes2ts project-acp `
  --path data/interim/pes2ts_synthetic_demo_v1/PathBundle.json `
  --plan data/interim/pes2ts_synthetic_demo_v1/ScanPlan.json `
  --acp-root E:/Calculations/Common_Script/Auto_Calc_Platform/ACP_V1_20260811 `
  --proposal data/interim/pes2ts_synthetic_demo_v1/SeedProposal.json `
  --proposal data/interim/pes2ts_synthetic_demo_v1/SeedProposal.internal_scan_peak.json `
  --output data/interim/pes2ts_synthetic_demo_v1/acp_trajectory_graph.json `
  --profile-output data/interim/pes2ts_synthetic_demo_v1/pes_profile_v2.json
```

本地合成 `PathBundle` 可用离线查看器展示扫描能量和三维结构：

```powershell
python bin/pes2ts view-path --path examples/contracts_v1/PathBundle.json --output examples/contracts_v1/path_viewer.html
```

查看器把缺失能量显示为空点；不同排序规则只重排现有帧；曲线选中的 `frame_id` 同步控制结构。结构连线只是距离提示，不代表键级。页面明确标记“未做 TS 验证”。ACP frame/annotation 原生投影现已用目标 checkout 冒烟验证；图 JSON 的 Workbench/API 接入、RESULT 几何物化与远程缓存解析仍待 ACP 运行时验收。
