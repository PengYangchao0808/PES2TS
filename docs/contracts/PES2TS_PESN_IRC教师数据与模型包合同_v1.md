# PES-N IRC 教师数据与模型包合同 v1.1（草案内容版本）

> **2026-10-21 V5.2-R 增补（ADR-0030）**：V5.2+ 的任务语义以 [PES2TS_FullPathNavigationContract_v1](PES2TS_FullPathNavigationContract_v1.md) 为准——完整 canonical IRC 教师（非 endpoint→TS branch）、Move/Stop eligibility 分离（terminal=no_future_contact 行进入 Endpoint Stop 数据）、Stop 为目标端点到达（d_end 0.10/0.30 Å）。本合同的旧 TS 教师字段仅作历史草案保留。

> 状态：已登记设计草案；字段语义供 G3-P1 实现，正式 schema/validator/fixture 尚未发布
> 层级：G3.P；数据/特征/标签/run/bundle 接口；规范效力以宪法 §9.6 登记及实现验收为准
> 取代关系：新增隔离教师合同；细化 PESNModelBundle 家族，不重定义旧 ReactiveState/Action/Transition/label_state
> 上游依赖：[开发宪法](../PES2TS_开发宪法_v1.md)、[ADR-0013](../design/decisions/ADR-0013-G3并行IRC预训练与模型版本体系.md)、[G3-P1 方案](../plans/PES2TS_G3_P1_IRC教师预训练与模型版本计划_v1_20261006.md)
> 下游消费者：离线构建器、dataset loader、特征/训练/隔离评测与模型包校验
> 当前版本：草案内容 v1.1 @ 2026-10-07（新增研究预测及产品依赖边界）；原payload schema v1仍待实现

## 1. 身份关系

`source → grouping/split → dataset/labels → feature_spec/train_normalization → training_run → model_bundle → sealed_evaluation`。

每层有 schema_version、内容发布版本或不可变 ID、内容 sha256 和父引用。时间戳不参与内容哈希；序列化/数组精度/行序/排除清单的规范化算法在正式实现中冻结。来源哈希不是模型哈希，同版本号不同内容为冲突，不能自动覆盖。

## 2. Dataset / TeacherState

合同 `pesn_irc_teacher_dataset_v1`；首个数据发布目标 `0.1.0`。数据身份包括 source manifest/hash、方法/单位/电子态范围、父 inventory/split/map/graph 版本、构建器版本、LabelSpec/hash、分片及逐 shard hash、join/rejection/exposure 汇总、各任务有效计数。

| TeacherState 字段族 | 语义与检查 |
|---|---|
| sample_id / reaction_group_id / reaction_id | 不可变样本与防泄漏组；组身份不由路径帧编号决定 |
| source_ref / source_method / electronic_state | 原来源及物理方法，电荷/多重度/必要环境；未知记 null+reason |
| atom_order / atomic_numbers / atom_map_hash | 与当前 X、图、坐标支持集严格一致；重复/缺失映射 typed 拒绝 |
| task_graph / task_source | 端点图/目标编辑/原子角色及 provenance；不含参考 TS 派生特征 |
| current_geometry / units / geometry_hash | 隔离 IRC 当前帧训练观测；全结构、有限数值及长度单位 |
| coordinate_spec_ref / q / observation_mask | 图定义坐标版本及可用性；q 从当前 X 计算而不是标签回填 |
| optional_E_g / observable_provenance | 后置消融，绑定同几何/方法/单位；物理与有效约束量区别 |

帧索引、分支真值、TS/未来引用存储于隔离 provenance，不作为输入张量。TeacherState 是离线训练专用对象；生产 ReactiveState 不能由宿主从教师文件加载，在线 FeatureSpec 只用当前真实观测。

## 3. TeacherLabel

合同 `pesn_irc_teacher_labels_v1`；样本 join 键 sample_id + geometry/map/coordinate hashes。每项值及 mask/reason、来源引用、构建算法版本共同存储。

| 字段 | 语义 |
|---|---|
| signed_progress / branch_arc_length / branch_total_length | 分支去刚体、质量加权弧长及端点资格；不是精修成功 |
| path_proximity | 本分支归一化 `abs(s)`；不作跨反应绝对距离或概率 |
| teacher_direction_q / direction_mask | 尺度化内部坐标朝 TS 的单位局部方向；TS/零方向/退化显式缺失 |
| coordinate_activity / activity_mask | 冻结局部变化定义的协同标签；不改 driver/guard 角色 |
| teacher_delta_q / lookahead_spec（可选） | 固定弧长 lookahead 的有限教师目标；不是 command_delta_q |
| classic_reference（可选） | Classic/动作/配置版本与名义命令引用，只供离线教师差值实验 |
| reference_TS / neighbor_refs | 仅隔离侧使用；不进入 features/bundle 运行输入 |
| endpoint_status / quality / provenance | 完整/单侧/截短/定向/数值资格，来源及排除原因 |

LabelSpec 冻结质量/配准/角 unwrap/局部差分/尺度/阈值/反向语义/TS 方向屏蔽。有效任务覆盖分别统计，不将无标签写成 0。`label_state` 七态继续用于真实精修，不给教师样本伪造 verified_target。

## 4. Split / FeatureSpec

合同 `pesn_irc_split_v1`；记录父 split/hash、canonical grouping 版本/成员哈希、train_fit/dev/calibration、sealed valid/test/family、开发暴露组、历史排除/unassigned、冲突处理、冻结时间和访问权限。新跨 split 冲突隔离而非偷移回 train。

合同 `pesn_feature_spec_v1`；记录训练与线上共同允许字段、图/坐标定义、单位/尺度/missing mask、不变性、输入支持域和 normalizer/hash。normalizer 仅拟合 train_fit；训练选参在 train 内完成，sealed 集合参与训练/标准化的样本数为 0。

schema validator 未来验收包含：组跨 split、方法/原子/图不一致、未来/参考标签字段进入 input、normalizer 对 sealed 集合拟合、时间戳以外未绑定内容变化。输入不合法返回 typed rejection，不猜测支持域。

## 5. TrainingRun

合同 `pesn_training_run_v1`；包含 run_id、attempt/resume 事件、status、software commit 或 dirty-tree 内容快照/hash、Python/库/硬件、seed、配置/数据/split/特征/标签哈希、模型参数量、任务有效覆盖、loss/epoch/checkpoint、train 内选择理由、输出 hash、所有失败/资源费用。

预算包括 reaction/frame/epoch 上限、模型选择配置数、GPU/CPU 时间与实测峰值内存；缺失 null+reason。预训练若没有发起 QC，新增 QC 调用计数可记 0；历史数据生成费用未知不能记 0，应另列 source_cost_unknown。训练费用和生产 QC 成本分别保留。

重启恢复同 run 不意味着覆盖日志；数据/特征/训练配置 hash 改变时开始新 run。resume 的 checkpoint 来源与优化器/随机状态兼容性受校验。

## 6. ModelBundle

合同 `pesn_model_bundle_v1`；目标模型 ID `pesn-irc-v0.1.0`。bundle 绑定权重/架构/FeatureSpec/Normalizer/LabelSpec、训练 run、数据/split、支持范围、依赖版本、输入输出 shape/单位、capabilities、model card、评测和 checksums。

| capability | 本期允许的范围 |
|---|---|
| encoder / progress / path_proximity | 对应教师任务经独立评测后按头开放 |
| teacher_direction / coordinate_activity | 方向/协同门和覆盖通过后才开放，否则 unsupported |
| p_seed / C_full / Q_STOP / Q_MOVE | disabled，未训练/未验收；不输出占位值 |
| production_control | false；没有自动接入宿主或切换默认的权限 |

bundle 包含教师算法语义与来源引用，但不打包逐样本参考 TS/未来帧/原始教师真值。研究推理输出沿用教师头名称，不别名成 DecisionEstimate 成功概率或动作价值。未知 feature/schema、权重摘要不匹配、能力不存在或输入域外时 typed unsupported/rejection。

独立评测报告绑定 frozen model/hash、sealed split/hash、基线/预算/门及每族结果。不能通过重新挑 seed 或回调训练消除 sealed 失败；模型失败保留结果，后续版本需新的预注册评价设计。

## 7. 实现验收与兼容

正式实现再交付 JSON schema/array schema、validators、fixture、guard scoped caller、读写与哈希工具。合同族在宪法登记仅表示设计存在，不能称正式接口已落地；本次没有给 truth guard 开新权限。

旧主数据链、历史 0.1.0 软件、旧 split/TS/IRC 索引/生成轨迹与报告保持原内容。本合同是新增隔离数据与模型包，不以同名字段给旧 Action/Transition 偷换动力学语义。

## 8. 预测器运行输入与研究输出（ADR-0014）

PESNInput以pesn_feature_spec_v1为约束，含当前结构/元素/原子映射、普通任务图/来源、坐标描述、电子态/方法/单位和观测资格。不含G1/G2可执行编译器对象、参考TS/未来帧/标签或源读取回调；图/坐标定义由拓扑authority给出，预测器只转换和数值求值。

新增登记草案 `pesn_teacher_prediction_v1`：input/model/feature hash、研究progress/path_proximity/score/direction/activity、逐头availability/mask/reason、支持域与研究能力标记。它与正式DecisionEstimate、Action/Transition、ValidationResult分开；不输出精修成功概率、已执行动作或科学成功标签。单点/批量请求保持对应ID/顺序，无效样本typed拒绝而非静默丢行。

本期产品代码归pes2ts_core/pesn，source/标签/训练/模型评测/导出位于offline，运行可达集位于api/contracts/features/models/inference/bundle。离线依赖运行集，反向不可；唯一拟真值桥为offline/data/truth_source，仍调用G0唯一访问器并保留allow_truth/审计/split-scope，未在本次扩guard权限。

导出包只包含运行白名单代码和模型资产/公开元数据，不含offline/source/HDF5依赖、教师样本答案或训练optimizer状态。包初始化不触发ML训练/真值/ACP调用；正式依赖锁与API/CLI验收见[预测器模块设计](../design/PES2TS_预测器产品与G3代码模块划分_v1_20261007.md)。
