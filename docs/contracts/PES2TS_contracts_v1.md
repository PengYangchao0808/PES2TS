# PES2TS 科学数据合同 v1

由 `pes2ts_core.contracts` 实施。七类顶层对象共用严格的 `schema_name`、`schema_version=pes2ts_contracts_v1`、稳定 `object_id`、UTC `created_at`、`producer`、`input_refs`、`status` 和 `content_sha256`。不认识的顶层字段拒绝；有扩展需要时放进命名空间化 `extensions`。非有限数值、真值字段混入生产 `ReactionCase`/`ScanPlan`、旧版本、摘要错误均拒绝。

## 对象与关键身份字段

| 对象 | 必需身份及内容 | 独立状态 |
| --- | --- | --- |
| `ExperimentManifest` | experiment ID、数据版本、split 反应清单、策略版本、预算、验证协议 | `draft` / `frozen` |
| `ReactionCase` | reaction/case ID、dataset version、split、显式 atom map、R/P 几何、电荷、多重度、互斥编辑、来源 | `ready` / `needs_review` / `rejected` |
| `ScanPlan` | plan ID/version、R→P 原身份、原子映射、候选方向/方法/点列/预算、回退 | `ready` / `needs_review` / `rejected` |
| `ExecutionRecord` | execution/ACP task/candidate/request ID、尝试与重试、逐尝试 CPU 成本及 complete 标志、WORK/RESULT 引用 | `queued` / `running` / `completed` / `failed` / `cancelled` |
| `PathBundle` | path ID、计划/候选/执行/ACP 任务绑定、稳定 frame ID/index、结构和独立能量通道 | `unchecked` / `usable` / `unusable` / `needs_review` |
| `SeedProposal` | proposal ID、path/frame 引用及源路径状态和 SHA256、ranker 版本、规则、分数、理由与选择来源 | `accepted` / `rejected` / `needs_review` |
| `ValidationResult` | validation ID、候选引用、OptTS、频率/振型、双向 IRC、端点匹配；命名扩展记录各阶段重试与 CPU 成本 | `not_run` / `incomplete` / `passed` / `failed` |
| `ReviewRecord`（附属合同） | 源案例/工作簿摘要、两位独立复核者、各维度结论、扫描可行性和最终裁定 | `accepted` / `rejected` / `needs_more_info` / `replacement_required` |

合同验证器不把 ExecutionRecord 完成等同路径可用，不把 SeedProposal 接受等同物理验证通过。

`ValidationResult.status=passed` 还有独立的物理证据门槛：OptTS 必须收敛并标记为一级鞍点，关联来源初猜 `source_frame_id`；频率分析必须恰有一个虚频；双向 IRC 都必须匹配且分别到达反应物与产物端点，正反符号可以互换。通过结果还必须绑定 `path_content_sha256`、`proposal_content_sha256`、`source_frame_id` 和 `source_geometry_sha256`；指标入口会重新比对当前路径、提案和源帧几何，旧内容版本的验证记录不能计入新候选。OptTS 必须记录 `optimized_geometry_sha256` 和 `protocol_sha256`；频率与双向 IRC 分别记录 `source_ts_geometry_sha256`、`protocol_sha256`，三个后续阶段的源 TS 摘要必须与 OptTS 输出结构一致。每个成功阶段都要记录其 ACP v2 `result_manifest_sha256`。调度器模式记录 `acp_task_id`；纯 CLI 模式允许 `acp_task_id=null`，但必须记录 `execution_id`、不可变 `attempt_id` 和完成产物摘要。缺任一项不得构造通过合同。`build_validation_result` 从阶段状态自动推导 `not_run`/`incomplete`/`failed`/`passed`，并在 `extensions.pes2ts.validation_costs.v1` 留存每个阶段的重试、失败码、逐次 CPU 秒和完整性标志。该结构门槛不替代对 ACP 原始输出、振型和 IRC 几何的独立科学复核。

`integration.acp.stage_results` 提供只读 ACP 产物提取：从完成的 BatchOptimize v2 清单校验 TS 优化结构、normal_modes 产品与 `geometry_product_id` 绑定，重算有限频率、模式数和虚频标记；从 IRC 清单核实两方向端点文件、报告来源证明与 TS XYZ 文件哈希，并按 ReactionCase 原子序通过 proper-rotation Kabsch RMSD 比对 R/P 几何。该实现目前只在合成 v2 产物上回归；它尚未启动 BatchOptimize/IRC、读取真实 ACP 任务记录，亦不能代替真实计算中的方法/自旋/端点科学审核。IRC 几何比对默认阈值为 0.35 Å，调用方应按冻结的验证协议选择并保留该值。

`build_scan_plan_with_rejection` 会将策略不支持的有效 `ReactionCase` 转成 `status=rejected`、空 candidates 和显式 `reject_reasons` 的 `ScanPlan`，以便计算覆盖率时保留拒绝分母。

`ReactionCase.status=needs_review` 可显式保留未知端点多重度（`multiplicity: null`），但必须附带 `review_reasons`；这种案例不能进入 ScanPlan。标为 `ready` 时，R/P 两端都必须有明确的正整数多重度。不得从 G1 `multiplicity_max` 推断体系自旋。

若 ReactionCase 的 `source.spin_provenance` 存在，校验器会将其视为可验证证据而非自由备注：要求列出 R/P 两侧来源组分，校验组件 ID、原子数覆盖、组件电荷和端点总电荷，并依据各组件多重度重新推导端点多重度及解析状态。未知组件自旋或多个非单重态组件必须保持 unresolved；记录的解析结论与 ReactionCase 多重度不一致时拒绝。可选 `source.g1_export_sha256` 与 `source.g1_payload_sha256` 必须为小写 64 位 SHA256。此校验不把自动解析升级为人工化学接受，案例状态仍须由复核流程决定。

当前规则 ranker 只对 `usable` PathBundle 发出 `accepted` proposal；`unchecked` 或 `needs_review` 路径只产生待审建议，`unusable` 路径不产生候选帧。`SeedProposal.path_status` 回显源状态，`review_note` 解释临时排序状态。

新的 ranker proposal 必须携带 `path_content_sha256`，以便物理验证拒绝由旧路径内容生成的候选；proposal ID 由路径内容摘要、规则、ranker 版本和 top-k 共同确定。为读取既有 v1 样例，该字段在基础加载器中暂为可选；`build_validation_result` 对来源摘要执行强制匹配。

## 附属数据对象

- `ArtifactRef`: `artifact_id`, 包相对 `relative_path`, 文件字节 SHA256, `media_type`, `size_bytes`；POSIX 与 Windows 分隔符统一检查，拒绝盘符/UNC 绝对路径与 `..` 越界。
- `MethodSpec`: 由路径能量引用，说明计算引擎/版本、方法/基组、溶剂、电子态、数值设置和参数摘要。
- `ReviewRecord`: 复核人、输入版本/摘要、复核清单、结论、理由和时间；新复核版本追加而不覆盖旧建议。

## 编号、单位和完整性

- `reaction_id` 来自数据集；`case_id`, `plan_id`, `execution_id`/`attempt_id`, `path_id`, `frame_id`, `proposal_id`, `validation_id` 各自独立。ACP task/job ID 单独绑定。
- `atom_map_ids` 为显式唯一正整数；`atom_indices` 与 `frame_index` 从 0 开始。展示 frame number 从 1 开始。帧缺失能量不改变后续 frame ID/index。
- 长度 Å，扫描角度 degree，原始电子能 Hartree；精化单点与扫描能量为不同通道。一个通道跨帧只能引用同一方法。相对能必须另存参考帧/能量。
- 结构 JSON 摘要用规范化 UTF-8 JSON，键排序、紧凑分隔。科学内容摘要排除自身摘要、创建时间和 producer；真实文件另算字节 SHA256。未知量用 null，禁止 NaN/Infinity 或用 0 填缺失。
- ACP 目录由 ACP 分配：原始输入、日志、scratch 在 `WORK`；已校验可消费产物在 `RESULT`，通过 ACP `result_manifest.json` v2 登记。Product.path 相对 RESULT，科学包内部引用相对包根。

PathBundle 合同逐帧检查原子映射顺序与顶层 `atom_map_ids` 一致、元素顺序一致、geometry 为有限数值 N×3；能量值也必须是有限数值。扫描路径质量层会从几何重算驱动距离，对比目标坐标、ACP actual coordinate 和 constraint residual，并拒绝低于 0.45 Å 的原子重叠。该阈值用于拦截显然不可能的几何，不构成键级、价态或过渡态的化学验证。

当前最小计划限定单根成键/断键或 H 转移距离坐标，3–101 个点，最多 4 个同步坐标；超出范围提前拒绝。该计划构造器暂不替代完整 G1 选边、几何复核或 ACP 后端 capability check。
