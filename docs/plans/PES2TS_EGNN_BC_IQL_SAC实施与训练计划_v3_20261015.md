# PES2TS 本轮实施与训练计划：EGNN → IRC-BC → IQL → 真实 SAC

> 状态：前版记录；当前下一步训练排期已由统一方案v4取代（原状态见正文）
> 层级：G3.P/G3.2 数据与模仿、G4.0 离线价值、G4.1 真实交互、G2.T/X 严格验证配套
> 取代关系：取代[完整方案 v2](PES2TS_EGNN_BC_IQL_SAC下一步完整训练方案_v2_20261015.md)的下一步实施安排；旧模型、执行配置、数据和实验报告保持原义
> 上游依赖：[开发宪法](../PES2TS_开发宪法_v1.md)、[阶段定义](PES2TS_阶段定义_G0-G3.md)、[ADR-0020](../design/decisions/ADR-0020-EGNN直接位移与BC-IQL-SAC研究主线.md)、冻结来源与 ACP
> 下游消费者：预测器、离线构建/训练、direct 宿主、ACP 接缝、严格验证和实验报告
> 当前版本：v3.4 @ 2026-10-15（v3.3 基础上追加 §11.11 批次 1–6 执行记录：direct 合同/推进器/数据/模型/BC/IQL 实现与真实 fit50 执行；a_max 校准信号登记）；这是实施设计，不是现有 CLI 或真实能力声明

## 2026-10-15 当前取代与进度说明

当前训练任务与排期见[统一方案v4](PES2TS_双端拓扑条件训练与项目进度统一方案_v4_20261015.md)及ADR-0021：无序双端完整拓扑→完整A→TS/B→TS教师→逐点xTB→规范化EGNN→BC→相容IQL，SAC后置。本文作为前版研究设计/实施证据保留；有向端点、跨TS教师、旧产物势/终态、100步截断、优先扩量/SAC及旧费用安排已被取代。旧WP编号不等于新WP已完成。

旧direct fit50与BC/IQL表征证据不撤销，但不作为新双侧任务验收；新版本实现/数据/训练待执行。本次只完成文档整合与规范适用登记，不启动计算。下文“当前/下一步/未运行”等均按原版本日期和范围解读。

## 1. 本轮范围、目标和交付

本轮回答三个问题：全原子等变位移是否能离开固定扫描轨迹并生成可用候选；IQL 相比同数据 BC 是否有贡献；相同真实交互预算下，IQL→SAC 是否优于 BC→SAC。任务输入是实际反应物几何、已知目标产物图/机理和电子态。第一轮聚焦已登记的中性单重态基本反应；未知产物、多步总反应、电子态变化和未登记事件不混入主比较。

**主训练规模为 2,000 个独立 fit 组，先完成 50/500 的数据与闭环小试。** 5,000 组为条件扩量，20k/全库属于下一轮。SAC 首包是 12 个 fit 组上的有限真实交互，绝不将“2k 离线反应”写成“2k 真实 SAC 反应”。

| 目标 | 本轮完成证据 | 不由该证据推出的结论 |
|---|---|---|
| 数据目标 | 新用途合格 split、真实 E/g、完整状态推进、终态与预算一致的 direct 转移 | 旧教师样本已经是 RL 数据 |
| 工程目标 | 可回放环境、概率正确的 EGNN、BC/IQL/SAC 独立训练与导出 | 测试通过即有化学收益 |
| 闭环目标 | 实际部署起点自由 rollout；至少一个自生成候选完成严格目标 TS 链 | 一个成功即证明泛化 |
| 科学目标 | 同环境、同候选/验证预算比较严格成功率与全费用 | 连接率、非零动作或高熵即有效策略 |

科学改善的预登记候选目标：相对 EGNN-BC，严格成功率提升至少 10 个百分点；或成功率非劣界为 −5 个百分点时，全输入总成本下降至少 20%。最终结论同时报告 95% 区间；区间不足以支持门槛时写“证据不足”。这些是研究目标，不是保证；在 fit 校准结束、查看新开发结果之前冻结，不能按 sealed 结果调整。

交付包括：用途/来源清单、direct 合同与环境、2k 三 seed BC/IQL 包、受控 SAC 包、同预算对照报告、严格验证与成本账本。所有模型初始能力为 `research_only`。

## 2. 已核实资产与本次数据选择

### 2.1 来源和现有数量

源数据由 `data/manifests/source_manifest.json` 标识为 `zenodo-18551029-rev1`，IRC 源方法为 B3LYP-D3/TZVP。源 IRC 文件登记 SHA256 为 `2a55b17c4ddbc7c394825fd0143d131ad13dc796654ac4c486b2cbee6fd00ef5`；本次只核对清单，未重新散列约 9.95 GB 原始文件，也未读取真值几何。正式读取仍走现有审计访问器。

以下是 2026-10-15 对已有 JSON 清单和分类元数据的实查，不是新训练结果：

| 资产 | 路径/键（相对仓库根） | 实查数量与本轮用途 |
|---|---|---|
| 历史教师包 | `outputs/pesn/scale20k_rl2000_v1_20261008/irc_5k/teacher_dataset/dataset_manifest.json` | 5,250 反应、605,490 教师样本、27 shards；其中 fit 5k、monitor 250；只复用来源/身份，不复用旧动作标签 |
| 旧 fit5k | 同根 `cohorts/irc_fit_cohorts.json:irc_fit_5000` | 5,000 reaction IDs，实际 4,994 canonical groups |
| 旧 fit2k | 同文件 `irc_fit_2000_verified` | 2,000 IDs，实际 1,995 groups；不等于新 2k 独立组 |
| 监控 | 同文件 `irc_dev_250` | 250 IDs/250 groups；作为本轮 fit-monitor 候选，允许早停/选择 |
| 开发排除 | `cohorts/rl_cohorts.json:rl_dev_200` 联合 `origin_gap_replacements.json` 的开发使用记录 | 217 IDs/217 groups；整组禁止进入新拟合 |
| 开发主名单 | `rl_cohorts.json:rl_dev_50_fixed_checkpoint_subset` | 原始 50 组；保留起点失败，不改用成功替补集作为新分母 |
| 封存候选 | `cohorts/sealed_cohorts.json:sealed_iid_100/sealed_family_100` | 各 100 IDs/100 groups；family100 含 14 个既有 L0u 分类簇，历史暴露/完整族闭合仍待验收 |
| 旧真实交互 | 同根 `reports/first_batch_report_v1.json:main_rl50` | 47 episodes、605 转移；动作/环境不相容，仅作历史诊断 |
| 旧开发结果 | 同报告 `dev_round` | 46 组，规则21完整、BC19完整、Q21完整；不当新严格成功基线 |

原库存清单记录 199,217 行、146,731 个当时合格 train ID；它们不是本轮已合格独立反应数。旧模型卡说明教师映射含 `truth_assisted_p1`，新报告继承这一来源声明；独立部署来源另做端点映射/无真值筛选验证。

### 2.2 本次审计发现与可审阅名单

[元数据审计 JSON](fixtures/pes2ts_direct_training_input_audit_20261015.json)保存来源文件哈希、交集、排除 ID、候选顺序和待验事项。复核工具为 `scripts/audit_direct_training_inputs.py --check`；该工具只读名单/分类元数据，不读取 IRC/TS 几何、不提交 QC、不赋予新数据用途。

1. 旧 fit5k 与历史开发暴露联合集有 **17 个 ID/17 个组交叉**，来源于开发替补。它们全部从新 fit 候选排除。历史报告保持原样，新报告注明旧开发集合并非全部与旧教师 fit 独立。
2. 排除后为 **4,983 个 ID/4,977 个独立候选组**。各组取可用 ID 中字典序最小者，再按 `sha256("42:"+canonical_group_id)` 排序，完整代表 ID 清单存于 JSON 的 `candidate_pool.reaction_ids_ranked`。
3. 所查名单及其重复组成员中，发现 **14 个组具有多个 L0u 标签，3 个成员缺有效 L0u 标签**。这不是已证明的封存泄漏，但足以禁止把旧多数票分类直接视为整族保证。当前审计不是全库族闭合或完整历史暴露审计。
4. 旧 fit5k 与现有两个 sealed100 清单在 ID/已知 canonical group 层面交集为零；不能由此推导 family 全库闭合或历史所有模型均未暴露。

当前候选前四个 ID 为 `RXN_0000153579`、`RXN_0000017131`、`RXN_0000100200`、`RXN_0000009781`。它们只是可审查的小试候选，不代表新的电子态、QC 或 direct 用途已通过。

### 2.3 本轮集合与冻结规则

| 集合 | 目标独立组数 | 选择与用途 |
|---|---:|---|
| `direct_fit50` | 50 | 合格候选固定排序前50；映射、重采样、共享推进器和 ACP 校准 |
| `direct_fit500` | 500 | 包含 fit50；单 seed 冒烟及自由 rollout，形成数据/资源证据 |
| `direct_fit2000` | 2,000 | 包含 fit500；本轮主 BC/IQL，三训练 seed |
| `direct_fit_expansion` | 当前上限4,977，目标5,000 | 先使用已审计候选；补足至少23组还需新资格漏斗，不能把重复 ID 凑成5k |
| `direct_monitor250` | 250 | 旧 `irc_dev_250` 的组级复核集合；monitor8 为组哈希排序前8，用于小试/选择 |
| `direct_dev50` | 50 | 原始固定 checkpoint50；dev8 为原序前8，只作已冻结版本诊断 |
| `direct_sac_fit12` | 12 | 合格 fit50 固定排序前12；BC→SAC 与 IQL→SAC 共用，失败保留 |
| `sealed_iid100/family100` | 各100候选 | 本轮准备冻结身份；最终验证前不用于选模型；不足如实留缺口 |

JSON 中 50/500/2k 是当前候选顺序的半开区间前缀。WP0 完成组/族、用途、电子态、活动事件和来源审核后，产生新的正式 selection 文件及哈希；当前候选 JSON 标记 `metadata_audited_not_training_qualified`，不是执行许可。首次标签/QC前允许依公开资格规则排除并按既定顺序补位；冻结后 QC/准备失败保留在该批分母，补充批另建 manifest，不能悄悄换题。

新 2k 与新5k按**独立组**嵌套，不追求与旧2k的ID列表完全相同。扩量不足时保留真实规模。逆反应与多路径可以增加训练 episode，但不增加独立组数。

### 2.4 组/族和开发暴露的硬验收

沿 `duplicate_groups.json` 建立组级闭合；对族归属不一致的组，先核对分类来源。不能解决时将该组隔离，或对“组—族”二部图取连通分量作为不可拆分单位。所采用规则版本写入新 split。缺族标签不默认为普通训练样本。

封存整族从全部 fit、monitor、开发选择/阈值拟合中排除，检查全库族成员，不只检查抽中的100个ID。历史所有教师训练、超参选择、RL采集和开发替补清单做组级联合审计。L0u 是事件计数归一化分类；如论文声明“未见机理族”，还需冻结有化学意义的机理族 taxonomy，不能把 L0u 簇数直接解释为机理族数。

最终 `split_audit.json` 记录组交集=0、族交集=0、缺失/冲突隔离清单、历史暴露来源和 source hash。检查失败则不能签发 Gate-D。

## 3. 统一环境：逻辑预算与真实费用分开

### 3.1 第一版 MDP 的确定选择

环境 `E_direct_xtb_v1`：`a=ΔX`，`X_proposed=X+a`，通过 ACP 获取该原位几何的 GFN2-xTB E/g。成功观测时 `X_next=X_proposed`，无逐步完整优化、约束松弛或动作改写。非反应核心原子可协同运动。

本版将有限任务定义为 **100 次逻辑动作预算**，每次采样提交（包括拒绝）消耗1；每个合法节点最多一次标准 E/g 操作，无策略内重试。缓存命中仍消耗1逻辑动作，费用账本按真实命中计量。起点制备另分配并完整计费，同方法比较共用起点。

与 v2 的关键修正：**900秒是累计执行保障上限，不进入本版可学习状态，也不是MDP失败终态。** 到达该上限记 `censored_operational_limit`，保留完整前缀和已发生费用；不虚构失败回报或剩余时间。恢复继承逻辑预算和累计墙钟，上限扩大需新预算版本。这样离线样本不必伪造部署墙钟；若未来优化硬实时任务，另建含时间转移观测的新环境，不混入本版 replay。

### 3.2 状态与快照

`s=(Z, X, H_direction, atom_map, charge, multiplicity, method_id, ΔE, g, observation_mask, steps_remaining, event_state)`。

- `H_direction` 含完整映射目标图、F/B及必要键级/立体约束、方向和可执行范围；目标几何不是主协议输入。
- `ΔE=E(X)-E(reset)`，参考值与来源写快照。梯度单位统一为 Hartree/Å；输入转换和原始 Hartree/bohr 收据同时保存。
- `event_state` 首版仅包含目标连续命中计数、无进展计数、已达到的最佳部署目标势值；停止规则所需变量全部进入观察。
- 首版不设依赖隐含完整历史的循环终止；循环只诊断。若增加访问哈希等停止规则，相关状态一起版本化。
- 快照另外保存 episode_id、策略版本、随机数状态、请求收据、缓存绑定和累计费用；恢复信息不能暗中改变相同可见状态下的动作/奖励语义。

状态充分性测试不止“同快照重放”：还构造不同历史但相同可见状态，在相同动作/物理观测下比较后继、奖励、终止一致性。该测试不能证明实际QC绝对确定性，但能排除应用层隐含状态。

### 3.3 reset、目标与终止

训练教师 reset 是经同方法单点评价的 R 侧首帧，标记 `teacher_origin`；部署 reset 来自当前输入和经 ACP 的起点准备，标记 `deployment_origin`。两者同动力学但起点分布不同；自由 rollout 的主结果只用 deployment_origin。源端点优化后若需要连接到教师轨迹，不拼造连接；连接动作必须实际评价并计费。

`goal_connected_proxy` 首版采用完整目标图/可判定立体身份匹配、电子态/原子身份一致、E/g有效、无灾难性重叠，并连续两次命中。首次完成两次命中即终止；不以参考IRC终帧、序号或未来TS定位作为依据。无目标改变的 reset 记 `invalid_task`。这不宣称极小点；第一版不隐式执行端点优化来修改后继。

| 事件 | 任务终止/TD处理 | 账本处理 |
|---|---|---|
| 首次代理目标命中 | `terminated=true, bootstrap=0` | 单列 proxy success |
| 非法数值/身份/灾难性重叠动作 | 拒绝执行，失败终态；记录原始采样动作 | 没有QC也扣1逻辑步；真实QC费用0须有未提交证据 |
| 几何相关、可确定的QC失败 | 失败终态；吸收状态，无伪造梯度 | 保存失败请求、原因、费用 |
| 100步耗尽/冻结的无进展终止 | 失败终态；bootstrap=0 | 失败进入分母 |
| 取消、调度故障、900秒外部上限 | `censored`；不追加人工失败reward | 有有效next_state的完整转移可bootstrap；未解析后继的动作不进TD，待收据对账 |
| 教师数据切片/缺节点 | 不是目标终态；保留可证明的完整前缀 | 不跨缺口，不为填满batch生成假转移 |

非有限动作属于模型数值故障：保留原始诊断与该次尝试，但设`td_eligible=false`，不把NaN/Inf送入Q或伪造有限logprob；停止该采集版本并修复。有限、可计算密度但被几何门拒绝的动作才按上述失败终态进入TD。终态吸收状态由mask跳过下一策略/Q计算，不能用`0×NaN`实现bootstrap屏蔽。

同一步事件优先级固定为：执行前拒绝→物理/基础设施结果分类→有效观察上的goal→预算耗尽→无进展→继续。有效第100步首次命中goal算成功，不同时添加失败惩罚；已完成请求遇外部取消先对账，避免丢失已取得的真实后继。

无进展候选为连续20步部署目标势未改善超过 `1e-3`；这一阈值、图判定阈值和灾难性重叠半径在 fit50 校准后冻结。反应外键响应不一律硬拒绝；最终完整目标身份负责区分 wrong_target。具体 A(H) 适用在WP0登记。

## 4. 数据制备与同一转移推进器

离线适配器与在线环境共同调用 `advance_direct_state(state, action, observation, event, spec)`。函数由宿主拥有，无QC副作用，输出下一状态、奖励、终止、bootstrap和逻辑费用。离线代码可导入该函数；预测器runtime不导入宿主。

工序：冻结用途/组 → 审计窄桥读取 → R→P定向并拼接TS两侧 → 一致整体坐标框架 → 质量加权弧长重采样 → 动作范围验证 → ACP补算每个新节点E/g → 从reset顺序推进 → 按首次目标/任务终态截断 → shard及漏斗。

整体配准只去除整体系刚体漂移，不对各碎片独立对齐、不进行反射；记录变换并正确旋转梯度。质量度规用于采样，不声称保护化学键。重采样控制所有原子动作严格落在策略支持域内部，建议 `max_i||a_i|| <= 0.95 a_max`；零位移重复节点去重。超限插入节点后重新补算E/g，不裁剪动作后保留原后继。

每条合格TD记录至少保存：源反应/组/split、父轨迹与节点hash、源/目标方法、完整s/a/s'、采样空间/执行动作、E/g收据、validity、终止原因、bootstrap、reward各分量、逻辑成本与真实cost、environment/reducer/reward版本。专家动作没有行为策略logprob，填 `null + expert_not_probabilistic`；不能伪造随机采样概率。

**资格分开**：有明确动作对但预算/后继上下文不完整时，仅可列入另行声明的BC辅助集；本轮主BC/IQL比较使用相同的完整direct转移集合，避免把数据覆盖差异当算法收益。教师序列在100步内未成功，按实际任务成为失败轨迹；超出终态后的帧不进主集合。不得按参考剩余路径长度给中间帧分配预算，防止未来信息泄漏。

50组验收逐条核对 `X+a=X_next`、E/g几何绑定、step递减、首次goal截断、失败吸收、censored mask，以及离线/在线共享推进结果。DFT-IRC经xTB评价只是该环境中的示范路径，不自动成为xTB-IRC或最优专家。

## 5. 奖励细化：固定失败回报，明确代价排序

初始候选 `reward_direct_proxy_v1` 采用 `H=100, c=0.01, B=2, F=1, η=0.1, γ=1`。每次动作先扣c；代理成功加B；任务失败额外扣 `F + c×steps_remaining_after_action`。基础回报因此为：

- T步成功：`G_base=B-cT`，范围 `[1,1.99]`。
- 任意时刻失败：`G_base=−F−cH=−2`；立即失败与耗尽预算失败没有“节省负奖励”的差别。
- 外部删失：不人为添加上述失败项，不生成完整MC回报。

这消除了本配方下提前失败相对晚失败的基础收益优势；并不保证随机环境中的任意微小成功率差异都优先于成本。成功率约束仍通过独立严格评测验收，不把标量奖励当科学目标的数学等价物。

部署势函数候选 `Φ=−L_goal ∈[−1,0]`：由目标图成/断键的平滑距离罚函数及必要身份项组成，距离尺度来自元素半径/fit校准，不读取目标几何或参考路径。reward附加 `η(Φ(s')−Φ(s))`，任务终态Φ=0；删失不是终态。完整episode shaping和为 `−ηΦ(s0)`，只改善学习信号，不改变同起点终局排序。

原始物理费用不并入c：c是逻辑查询代理，非法动作/缓存也消耗逻辑预算，真实费用另记。若研究实际CPU费用奖励，新建reward版本和真实费用标签，不与此版本TD混训。

fit50阶段先用手算/可回放反例验收成功、立即失败、延迟失败、循环、预算耗尽和删失回报，再冻结系数。候选范围只允许 `B∈{2,4}`、`η∈{0,0.1}` 的有限fit比较；失败补偿公式保持。严格验证奖励留在未来新版本，不以unknown补0/1。

## 6. EGNN、概率与价值网络

主模型候选 hidden128/4层/径向32/cutoff5Å；轻量对照64/3/32。同一层同时接收当前空间边和固定任务条件边，边种类与归一化固定。目标任务边不要求中间几何拓扑恒定。

梯度通过等变向量通道，以及 `g_i·r_ij`、范数等不变组合接入；坐标更新仅在网络内计算，最终只提交一次动作。节点标量包含Z/角色/状态；图级含目标方向、电子态、ΔE、剩余逻辑预算和事件状态。数据归一化只拟合fit，向量用共同标量尺度，不逐轴独立标准化。

### 6.1 无量纲径向策略

取无量纲潜变量 `u_i~N(μ_i,σ_i²I3)`，`r_i=||u_i||`，`z_i=tanh(r_i)u_i/r_i`，物理动作 `a_i=a_max z_i`。`μ`是潜变量均值；确定性推理用T(μ)，不声称它等于变换后动作的期望。初值 `a_max=0.1Å`；fit比较候选0.05/0.1/0.2Å时同步重建动作/预算数据，并保持正式对照同一冻结步长。

每原子径向/切向Jacobian特征值为 `sech²(r)` 与 `tanh(r)/r`（后者重数2）。

`logπ_z = logp(u) − Σ_i[log(sech² r_i)+2log(tanh(r_i)/r_i)]`；零点使用连续极限与稳定展开。逆变换为 `u=atanh(||z||)z/||z||`；边界外、非有限值和错误mask显式拒绝。

`logπ_a = logπ_z − 3N log(a_max)`（以冻结的Å数值坐标为准）。训练NLL、IQL、SAC均明确使用 `logπ_z`；两种密度可同时记录但不混用。专家接近支持边界时重采样，不能靠clip概率掩盖无限逆变换。

### 6.2 熵与变原子数

SAC首版使用按有效维数归一化的熵：`ell=logπ_z/(3N)`，目标平均熵初值 `h_target=−1 nat/dim`；这是明确的每维熵SAC变体。actor与soft-Q target统一用 `α ell`，不是只在温度更新中除N。α初值0.01，固定α对照与自动α一次fit选择；α上限1只作告警/保护，触顶不视为收敛。

自动温度候选损失 `L_logα=−logα·stop_gradient(ell+h_target)`。σ初始范围建议 `[0.02,1.0]`，所有方向共享同一σ_i，fit校准后冻结；实测熵可达性和α趋势。若采用总熵标准SAC，则另版本并使用一致的`3N`目标和系数，不能两种公式混搭。

以Å直接对半径0.1球算熵时，上限仅 `−5.475N`，不能照搬 `−3N` 目标。无量纲化解决坐标单位依赖，但熵仍会影响变长episode的终止偏好；另测零进展、延迟终止和高噪声策略。学习与部署目标区别写入模型卡。

初版不对随机样本做刚体投影；可对潜变量均值去整体平移。真实动作的刚体分量单独测量，变换后不保证严格零质心。改内部3N−6子空间需要新的基、密度和线性分子处理，暂不实施。

### 6.3 Q/V与损失归约

actor、V、Q1、Q2各自拥有独立参数的等变编码器；Q同时编码当前几何和候选位移，使用范数、相对向量点积、梯度点积及候选后几何等不变特征，避免只看位移长度而丢掉方向。V/Q输出图级标量。

BC/IQL的每样本logprob在有效原子维求和；先在每个反应组内平均，再跨组平均。变N梯度贡献另报告；若改每维NLL，命名独立损失变体。SAC每维熵按§6.2执行，Q奖励不随原子数放大。

概率验收包括：float64逆变换/解析Jacobian/数值Jacobian、小r稳定、重参数化梯度；旋转/平移/排列一致；变N mask不计padding。建议float64相对误差1e−6、float32等变相对误差1e−4（近零输出另报绝对误差）；阈值在测试规格冻结，不用科学结果调整。

## 7. BC、IQL与SAC训练配方

### 7.1 BC

`L_BC=−E_group[logπ_z(a_expert|s)]`。初轮不叠加方向辅助损失，先记录方向余弦、质量加权位移/步长误差、关键/非关键原子运动和刚体分量；需要辅助损失时另建配方版本。

AdamW lr3e−4、weight_decay1e−4、clip1；8反应×16转移/批，按总原子/边数动态预算。定义一epoch为每个fit组被抽到一次、每组16转移；不把“50 epochs”误解为遍历全部节点50次。最多50 epochs，monitor NLL patience8；每组固定校验节点，选最好checkpoint。

50/500冒烟先seed42；2k主实验seed17/42/2026。2k→5k主规模对照从相同随机初始化分别训练；旧2k继续训练5k仅另记warm-start实验。旧标量教师权重不加载至EGNN。

### 7.2 IQL

`q=min(Qbar1,Qbar2)`；`δ=q−V`；`L_V=E[|τ−I(δ<0)|δ²]`；`L_Q=E[(Q−(r+bV(s')))²]`；`A=stop_gradient(q−V)`；`w=exp(min(A/T_A,log w_max))`；`L_actor=−E[w logπ_z]`。

τ0.7、T_A3、w_max100、lr3e−4、Polyak0.005；目标/优势均停止梯度。actor继承对应seed BC；新Q/V独立初始化。先2,000次Q/V更新、再最多20,000次联合更新，batch128（按边预算可拆微批累计）；每1,000更新评估固定monitor集。有限比较τ={0.7,0.9}、T_A={1,3}，先在seed42/fit-monitor选择配方，另两seed复现该配方。

记录优势/ESS、clip比例、双Q差、完整非删失轨迹的MC校准、actor对BC偏移和自由rollout。只在数据内动作上形成IQL目标。若权重近常数或价值无法校准，允许结论“IQL无额外贡献”；不强制改大网络证明胜出。

补数据优先级：同动作域内不同步长且重新推进的转移、多路径/正确逆向条件、fit真实偏离及失败。跳帧仍是一动作时只扣一次逻辑成本，重新运行目标/终止规则；不能合并已发生多步费用后假装一步。

### 7.3 SAC

两条分支 `BC→SAC` 与 `IQL→SAC` 共用fit12、环境、回报、预算和轮次。每轮每组2 episodes；单episode/采集轮冻结参数，轮末更新并发布新的collector版本。采集用随机动作；主冻结评测用T(μ)，随机评测另列。

首包只跑seed42、一轮：12×2×2初始化=48 episodes。通过Gate-SAC后，最多3轮、3训练seed：12×2×2×3×3=432 episodes（包含首包），不得把episode数算作432个独立反应。

IQL普通Q不直接当soft-Q。两分支均先冻结actor，用首轮真实fit转移进行最多1,000次critic warm-up，再联合更新。IQL分支“加载后warm-up”与“Q重置”在fit上记录校准；不能只给其中一组额外在线数据。

`a'~π`；`y=r+b[min(Qbar1(s',a'),Qbar2(s',a'))−α ell(s',a')]`；`L_Q=E[(Q−y)²]`；`L_actor=E[α ell(s,a)−min(Q1,Q2)]`。γ=1且时间预算已显式有限。`ell`定义见§6.2。

初始expert:online=50:50，专家仅来自同一fit2k direct集合；online覆盖充分后25:75作为后续消融，不在首包混改。UTD=1次联合更新/新增完整真实转移，warm-up次数另外列账；replay最多200,000转移、在线组均衡、丢弃策略FIFO按完整episode可追溯。对invalid终态保留动作/回报；未解析QC后继只进失败/待对账记录，不伪造TD。

每次更新重新计算当前策略的logprob，采集时旧logprob仅用于行为审计。SAC+BC/KL锚定另标变体；首包不同时搜索锚定、共享编码器、相关噪声和多种环境。

## 8. direct路径到严格验证

新增direct候选适配，不伪造旧扫描lambda或沿用约束乘子斜率。路径参数使用实际累计质量加权弧长；候选依据实际物理能量的局部夹峰、附近梯度与局部路径切向。保存被选帧及邻帧hash；RMSD只用于候选去重/诊断，不作为成功标签。

每反应候选上限K=3，先夹峰邻域后明确标记的未夹峰边缘候选；固定规则、去重阈值及排序在fit小试冻结。部分路径也可提交候选，未到代理目标不等于没有TS种子。结果映射到既有 `PathBundle→SeedProposal→ValidationResult`，新扩展携带direct provenance。

严格协议版本包括：目标DFT方法/基组/溶剂/电子态、xTB→DFT preparation、OptTS、频率目标模态、双向IRC、端点优化与完整R/P身份。源B3LYP-D3/TZVP仅是候选验证方法，真实可用协议须在fit冒烟冻结。按宪法§8保留DFT preparation；现有代码接缝不等于该完整能力已可用。

每反应共用K及验证总预算，不按算法成功候选数额外加预算。每个候选最多一个冻结协议的完整验证尝试，已登记恢复另计且所有方法一致；验证资源上限需fit实测，未登记前不启动批量严格链。人工ReviewRecord与ACP能力资格状态进入前置清单，不能把`blocked/not_run`算为物理失败。

首次闭环小试使用monitor8×{direct几何基线,EGNN-BC}=16 episodes，最多48候选链。所有已尝试输入留分母；至少一个BC自生成候选严格成功，且协议/费用链完整，才批准常规2k/5k扩量。未达标时保留500规模诊断，区分生成、候选选择、preparation与后端失败，不直接扩大训练集。

在2k阶段对相同固定样本同时报告proxy连接和strict结果的交叉表、候选可验证率及高应变失败。若proxy上升而strict持续下降，停止扩大SAC，修改后使用新reward/env版本重新验收。

## 9. 文件级代码调整与所有权

以下文件名为拟实施落点，不表示已经存在。旧标量/残差路径保持兼容；新CLI/config显式分流。

| 工作包/所属产品 | 拟新增或修改的文件 | 具体职责与验收 |
|---|---|---|
| WP0 规范/合同 | 宪法§5.2/6.5/7.3/9.6、阶段研究入口、ADR-0020；`pesn/contracts/direct.py` | 直接动作/教师转移用途适用条款；state/action/transition/bundle validator；拒绝旧schema混入 |
| WP0 数据分裂 | `pesn/offline/splits/direct_freeze.py`、复用`grouping.py` | 历史暴露联合、组/族连通闭合、候选与正式用途分开；不原地修改旧split |
| WP5a 控制器（前置） | `generation/planning/direct_mdp.py` | 无副作用共享推进器：事件、逻辑预算、奖励、terminal/TD mask；离线在线一份实现 |
| WP5a 控制器 | `generation/planning/direct_environment.py` | reset/step、独立合法性门、请求和真实几何后继、删失/恢复；无隐式重采样 |
| WP1 ACP接缝 | 复用`integration/acp/gradient_backend.py`及`orca_gradient_request/transport.py`；必要时新增`direct_observation.py` | 无约束原位E/g、方法/几何绑定、单位、超时/缓存/原始工作量收据；不增加直启引擎路径 |
| WP1 离线数据 | `pesn/offline/data/direct_preprocess.py`、`direct_dataset.py` | 审计读取、R→P/对齐/重采样；调用共享推进器；shard/lazy loader、BC/TD资格和漏斗 |
| WP2 预测器 | `pesn/features/direct_graph.py`、`models/egnn.py` | 任务/空间边、向量梯度、坐标等变与数值稳定；不替换旧`models/encoder.py` |
| WP2 预测器 | `models/direct_policy.py`、`models/direct_critics.py` | 径向分布/逆变换/logprob、Q/V动作条件化、独立编码器、变N mask |
| WP3 离线训练 | `pesn/offline/training/direct_bc.py` | 组均衡训练、monitor早停、三seed、checkpoint/run记录 |
| WP4 离线训练 | `pesn/offline/training/direct_iql.py` | expectile、AWR、目标网络、MC/ESS诊断、有效后继mask |
| WP5b 真实采集 | `generation/direct_round.py`、`generation/planning/direct_snapshot.py` | 轮内冻结、episode身份、恢复、累计成本/版本；ACP负责作业调度 |
| WP6 离线RL | `pesn/offline/training/direct_sac.py`、`direct_replay.py` | soft-Q warm-up、每维熵/α、混合replay、UTD与轮间更新 |
| WP7a 严格链（前置） | `generation/planning/direct_candidates.py`、`direct_path_bundle.py` | 自由路径候选、K预算、frame hash与现有验证合同适配；旧lambda排序器不强行复用 |
| WP7b 评测 | `pesn/offline/evaluation/direct_benchmark.py`，复用宿主严格验证/成本模块 | teacher-forced/free-rollout分开、组级配对区间、proxy/strict交叉表；不在预测器复制QC验证器 |
| 导出/推理 | `pesn/offline/publishing/direct_export.py`、`pesn/inference/direct_predictor.py`、bundle新判别分支 | 纯runtime依赖、特征/单位/模型/环境hash、能力表；旧包可读 |
| CLI/config | `pesn/__main__.py`、offline CLI显式分支；`config/pesn/direct_{data,bc,iql,sac,eval}_v1.yaml` | 新配置严格校验未知键、默认不指向新策略；新增命令分别验收 |
| 测试 | `tests/pesn/test_direct_{split,data,policy,iql,sac,bundle}.py`；`tests/test_direct_{mdp,environment,candidates,replay}.py` | 隔离、数值、终止、失败、恢复、TD mask与真实收据合同；测试替身不冒充QC |

表中`pesn/`等路径均位于`pes2ts_core/`下，config/tests另在仓库根。离线包可调用宿主纯推进逻辑，预测器runtime及公共初始化不能导入offline/generation/ACP。独立测试验证该依赖方向。

WP0具体修订草案：为研究版`A_direct(H,s)⊂R^(3N)`登记成员检查和独立Shield；限定离线IRC只在经过实际同方法E/g、共享推进及用途审计后形成`expert_direct`；现有“IRC邻帧不自动具有生产转移资格”保持。新用途只开放明确构建入口，不给包级truth_guard豁免；当前文档不修改这些规范或现行guard。

## 10. 合同、配置与运行产物

拟议schema族：`pesn_direct_state_v1`、`pesn_direct_action_v1`、`pes2ts_direct_transition_v1`、`pes2ts_direct_reward_v1`、`pesn_direct_bundle_v1`；配套split/selection/run/replay也单独版本化。动作记录同时区分`latent_u / normalized_z / proposed_angstrom / executed_angstrom`，未执行填null和拒绝原因。状态特征不包含source标签、参考进度、未来TS距离或文件路径等身份捷径。

每个正式配置在启动前绑定：source/split/selection/feature/normalization/env/reducer/reward/goal/ACP/strict protocol hash、软件工作树快照hash、模型seed、运行seed和预算。软件版本与模型版本分开，保存脏工作树差异摘要；不能仅记录HEAD来声称完全复现。

拟运行目录 `outputs/pesn/direct_egnn_v1_20261015/`：`audit/`、`splits/`、`observations/`、`transitions/`、`bc/`、`iql/`、`sac/`、`evaluation/`、`costs/`。正式模型独立导出`pesn-egnn-direct-{bc,iql,sac}-v0.1.0-rc.1`，每个训练seed具有独立run和checkpoint。

计划中的命令职责分别为“冻结用途→制备观察→构建转移→训练BC/IQL→采集轮→训练SAC→导出→冻结评测”。当前这些direct命令尚未实现；不能把旧`g3_p1_scale5k.yaml`或`control_trainer.py`换路径当作新训练入口。

## 11. 实验矩阵、预算和资源门

### 11.1 最小顺序与预算上限

| 批次 | 规模/对照 | 已明确的上限或公式 |
|---|---|---|
| 元数据/接缝 | fit50候选，monitor8候选 | 先有限节点，再完整序列；新观察数以实际重采样清单计 |
| BC冒烟 | fit500、seed42；4fit环境回放 | 最多50epochs；QC只在独立制备/rollout执行 |
| 严格链小试 | monitor8×2方法 | 16episodes，≤1,600逻辑动作，执行上限和≤4小时；≤48候选链另分预算 |
| 主离线训练 | fit2k、monitor250，BC/IQL各3seed | BC≤50epochs；IQL每seed≤22,000更新；不能给出未测GPU小时 |
| SAC首包 | fit12×2episodes×2初始化×seed42×1轮 | 48episodes，≤4,800逻辑动作，执行上限和≤12小时；准备/验证另列 |
| SAC条件确认 | 上述集合，最多3seed×3轮 | 总计≤432episodes/43,200逻辑动作/108小时执行上限和，含首包 |
| 开发诊断 | dev8×4学习分支×2运行seed，固定训练seed42 | 64episodes/6,400逻辑动作/16小时执行上限和 |
| 开发确认 | dev50×4分支×3训练seed，加几何基线/Classic各50 | 700episodes/学习+direct最多65,000逻辑动作；旧环境Classic请求单列；执行900秒共同上限和175小时 |
| 最终sealed | iid100+family100；主BC和冻结候选各3训练seed，几何/Classic各1 | 1,600episodes；900秒上限和400小时；是后续验收包，不随首包自动提交 |

“执行上限和”是每episode保障上限的加总，不是预计实际时长、并行历时或CPU核时。删失/未运行不伪造费用。Classic不能把一次含多个内部QC的宏步当一个direct节点比较，比较共同实际总费用与严格链预算。

数据制备量 `N_obs=Σ_reactions N_unique_resampled_nodes`，包含fit与monitor，reset观察可按内容缓存共享并保留父关系。测量50组p50/p90/p95延时、失败、重试原始工作量、磁盘bytes/节点；以p90和真实节点数外推，再登记硬总预算后才能放行500/2k。没有测量时预算状态为`unpriced`，不自动批准大批QC。

历史2026-10-08资源probe仅说明当时WSL 32逻辑线程/59GB RAM、CPU版torch、无可用CUDA、磁盘余量约38.5GB，不能当作今天可用资源。2026-10-15 Windows只读核查及新增infra计划见§11.3–11.8；训练实际解释器、WSL、磁盘余量和ACP全局资源配置仍待运行环境验收。

### 11.2 对照及选择纪律

首要四分支为EGNN-BC、EGNN-IQL、BC→SAC、IQL→SAC。另有同direct动作域几何基线，以及旧环境Classic系统基线。几何基线根据部署目标距离误差给推进方向，可加归一化真实梯度的横向纠偏；系数只在fit小试确定，每步同样一次E/g，无参考端点几何或教师切向。

编码器对照先在500/2k seed42比较SchNet+等变头与PaiNN，固定数据、输出分布、预算；是否进入三seed确认在fit-monitor冻结，不做编码器×算法×所有规模的全组合。

dev用于冻结版本的比较诊断，不作连续超参搜索。若dev反馈引发新设计，该版本的dev属于已暴露开发证据，最终仍依赖未接触sealed。sealed前冻结所有算法、阈值、候选/验证协议及发布版本，不挑最好seed只报一份。

### 11.3 本机实查与瓶颈判断（2026-10-15）

本次属于G3.P/G4训练基础设施规划与G2.T/X执行配套；证据见[资源核查记录](fixtures/pes2ts_infra_probe_20261015.json)。未安装依赖、修改系统资源、终止进程或启动训练/QC。

| 项目 | 本次观测 | 对计划的影响 |
|---|---|---|
| CPU | AMD Ryzen 9 7950X，16物理核/32逻辑线程；1秒全机利用率16.4% | 线程数不等于物理核；并行容量先按16物理核留余量，SMT收益实测 |
| 内存 | 总计约63.15GiB，采样可用约16.67GiB（占用73.6%） | 不能按64GB全空闲开worker；初期采用低内存profile，启动时重新测量 |
| GPU | RTX 4060 Ti，8188MiB；驱动591.86，驱动显示CUDA13.1 | 单卡8GB，单训练进程；驱动能力不等于PyTorch CUDA构建已安装 |
| GPU瞬时负载 | GPU 3%，占用2565MiB，约余5623MiB | 仅桌面瞬时样本，不能据此判断历史训练或归属具体训练任务 |
| Windows解释器 | `E:/Anaconda/python.exe`及`E:/Anaconda/envs/YY/python.exe`均未发现torch | 不能在这两个环境直接运行现有torch训练；实际历史训练解释器尚待绑定 |
| WSL/磁盘 | WSL枚举返回E_ACCESSDENIED；当前磁盘查询失败 | 标为unknown；不能用旧38.5GB当当前可用量，也不能断言WSL无CUDA |
| 现有实现 | 教师trainer未配置device；encoder组批与control trainer多处创建CPU tensor；未见CUDA模型迁移；direct EGNN/IQL/SAC尚未实现 | 现有路径没有展示充分使用GPU的实现证据；新增device参数还需完整迁移所有张量 |
| 既有QC并行 | `prepare_rl_origins_parallel.py`默认3个worker，注释按每任务10线程/“32-core”估算；`control_round.py`按反应循环 | 3×10是30个请求线程，并非30物理核；ACP实际线程/并发另核；旧采集尚无已验收的多环境并发证据 |

可见多个Python进程，但未核实其工作内容、环境或CPU归属，不停止或重新配置。上述证据支持“目前缺资源利用保障和GPU训练接线”，不能证明所有慢运行均由CPU串行导致。实际瓶颈可能为QC等待、图构建/读取、单样本推理、CPU训练或频繁同步，按下一节测量后定案。

### 11.4 资源分工、并行层次与初始profile

采用“单GPU learner/推理服务＋CPU数据worker＋多条独立episode＋ACP统一QC资源”的设计。CPU准备/读取与GPU训练可以流水，但首版SAC保持**整轮冻结采集→轮间学习→新版本发布**；不让采集worker持有可变actor，不在单episode内部并行有因果依赖的step，不改变环境、UTD、样本数或预算换取表面加速。

ACP负责QC作业的队列、并发、线程/内存配额、取消和幂等；PES2TS仅管理化学episode状态、请求及收据关联。所有数据制备、rollout和严格验证共用ACP全局容量，不能各自开满。未来ACP字段以公开接口验收，本节配置为设计键，不冒充现有ACP参数。

| profile | CPU数据worker | GPU训练 | 独立episode在途上限 | ACP同时QC/每任务线程 | 初始用途 |
|---|---:|---|---:|---|---|
| `safe_start` | 0→2 | 1进程，FP32 | 2 | 2×2，总4线程 | 当前可用内存约16.7GiB，先测吞吐/峰值 |
| `offline_gpu` | 2，比较0/2/4 | 1进程，3个训练seed依次运行 | 0 | 0 | BC/IQL/SAC离线更新；不同时启动第二seed占GPU |
| `rollout_cpu` | 0 | 冻结actor推理，比较CPU与GPU小批 | 4 | 4×2，总8线程 | SAC采集起点；同一轮各episode独立推进 |
| `qc_throughput`（条件） | 0–2 | 无训练或冻结推理 | ≤8 | 比较8×1、4×2、3×4；总线程≤12 | 内存/吞吐门通过才选，短任务优先任务并行 |
| `strict_validation` | 0 | 无训练 | 1–2 | 起点1×8，对照2×4；全局≤12线程 | 每任务内存按方法/体系实测；不把ORCA内存参数误作整个作业上限 |

以上为候选起点，未证明最优。按16物理核初期预留4核对应的容量给桌面、数据和控制；不凭32逻辑线程把QC×内部BLAS嵌套到32或更多。启动放行条件使用 `Σ(QC任务峰值RSS)+loader/learner/缓存峰值+系统预留≤实时可用内存`；已有进程占用不再重复计入。初期系统预留8GiB、整个新任务CPU内存增量上限8GiB，因此本次采样不放行高并发profile；若峰值无法容纳则减并发而不是换页硬跑。严格验证另冻结预算。

worker内部OMP/MKL/OpenBLAS各1线程、torch intra-op/inter-op各1作为起点；GPU learner CPU线程起点2/1，CPU-only learner比较1/4/8。这些限制按进程设置，ACP/QC任务单独传线程配额，不用全局OMP=1误伤QC。记录实际生效线程与ACP收据，线程请求不当实际CPU core-hours。

### 11.5 单GPU训练与数据通路

1. 建立独立、锁版本的训练环境，记录解释器绝对路径、torch版本/构建、CUDA运行时、驱动及依赖锁hash。先在实际训练环境验证CUDA可用、一个代表性图批次前向/反向与优化器更新均在GPU上；要求CUDA的profile缺CUDA时明确失败，不静默退CPU。选择Windows原生或WSL2中**一个**经ACP接线验收的环境；WSL当前访问失败不作为系统迁移许可。
2. `device`贯穿模型、actor/双critic/target/V/温度参数、图index/mask、奖励/TD后继、optimizer状态及恢复路径；CPU加载checkpoint后统一映射。避免循环内CPU建tensor、逐样本`.item()`/`.cpu()`同步和重复图构建。离线图/拓扑与静态特征按内容hash缓存，位置/梯度/空间边随状态正确刷新，不缓存错误的动态图。
3. 使用拼接图批处理、原子/边数分桶与预算上限，保留组均衡采样和既定有效batch。先测microbatch 8/16/32图，另冻结节点/边上限；大图单独分桶。OOM只重试当前尚未提交的batch、降低microbatch并用梯度累积维持有效batch，记录重试/耗时；不漏样本或修改更新次数。BC/IQL/SAC的原配方有效batch分别保持各自定义。
4. DataLoader起点2 worker、prefetch=2、persistent worker开启；只有worker>0才传prefetch/persistent选项。worker返回CPU tensor，主进程pin-memory并异步传GPU；Windows使用spawn、主入口保护、可序列化的顶层collate/worker函数，为每个worker派生seed。lazy/mmap读取shard，禁止每个spawn worker复制整套教师数据/605k Python对象。CPU回退时禁用无效pin配置。
5. FP32 eager为正确性基线。AMP单独试验：可支持时BF16优先、FP16配GradScaler；密度/Jacobian/径向逆变换/熵、奖励、Bellman目标与敏感归约保留FP32。对同批次比较损失、梯度、等变/概率测试与NaN；不改变算法阈值来通过性能门。TF32、torch.compile逐项比较后再冻结，首轮均关闭；动态小图或Windows编译不支持时保留eager，记录编译启动成本和重编译次数。
6. 8GB显存以**总占用**≤约7GiB作为候选软门，保留约1GiB余量；当前桌面已占约2.5GiB，新任务只能从余量预算。记录allocated/reserved峰值及桌面占用变化；IQL/SAC按全部网络和optimizer计算，不能用仅actor推理显存外推。吞吐较低但epoch很短时选择成本更小的profile，不以GPU利用率100%为硬目标。

### 11.6 采集并行、数据一致性与恢复

fit50节点制备可跨组/独立节点排队；单episode的第t+1步等待第t步真实收据。采用每episode独立目录/快照，单一汇总写入者，避免现有多worker共用输出根造成summary/manifest覆盖。请求ID、episode/step/attempt ID、geometry/方法/状态hash绑定，ACP原子幂等去重；worker重启不重复收费、不遗失失败尝试。取消后先核对ACP在途任务状态，再恢复，避免新旧请求重复在跑。

SAC fit12首包仍为48episodes，2→4→8并行只缩短排队与独立任务历时；首包两初始化以固定交错次序或配对批次执行，记录缓存/时间影响及全部成本。推理按当前已就绪状态做短等待小批，初始最多8状态/等待≤5ms；小批GPU比CPU慢时选CPU actor，离线learner仍可用GPU。轮内actor hash一致，每episode RNG由run seed、组ID、episode编号派生，不能随worker完成顺序分配随机数。

replay只在整轮合并后由单写入者原子提交；去重键、transition数、轮hash一致后放行学习。保存Python/NumPy/torch CPU/CUDA RNG、采样器位置、optimizer/target/温度和AMP scaler状态。物理返回顺序可能变化，报告并行数和请求序；不宣称QC逐位复现。900秒执行保障额外记录queue/执行/端到端时间；并行压测中排队导致的删失按§3处理，不重写MDP。

### 11.7 性能测量、加速验收与预算回填

先固定合格fit内的同一图批次、split、seed、模型/软件hash及QC方法。性能测量不用dev/sealed调参；QC压测也先登记fit50内的节点与请求预算，包含在正式观察预算中，缓存命中和真实执行分开报告。

| 测量包 | 对照与计量 | 决策 |
|---|---|---|
| GPU/loader | CPU FP32→GPU FP32→worker0/2/4→AMP；每项预热20步、计时至少200更新，3次重复 | 报graph/s、node/s、update/s、p50/p95 step、读/组批/传输/前反向/保存时间及峰值；CUDA计时正确同步 |
| CPU/QC | 同一冻结任务集，串行1×1基线及2×2、4×2、8×1、3×4候选 | 报观测/小时、episode/小时、p95延时、失败/超时率、实际/分配核时、总费用；冷/暖缓存分表 |
| 持续资源 | 1–2秒采样覆盖至少10分钟代表性训练/采集；按PID/进程树归属 | CPU每核/总负载、RSS/可用内存、GPU负载/显存/功耗、磁盘吞吐与队列、ACP在途/排队数；无法读取填null+reason |
| 恢复与数值 | 小包中断恢复、同批CPU/GPU及FP32/AMP对照，长短图/mask/失败/删失反例 | 数值容差在fit冒烟前登记；无漏样/重复提交/费用丢失，必要合同测试通过 |

GPU纯更新时利用率长期<50%触发profiling，不自动增加batch；QC主导采集的GPU低利用可以合理。数据等待占比>20%才优先加loader/预取；QC占端到端>70%时优先改善ACP吞吐；CPU大量上下文切换/内存下降则减worker/内层线程。阈值是诊断候选，非收益证据。

新增 **Gate-INFRA**：实际环境GPU前反向/恢复通过；资源时间线与配置hash齐全；同数据/同算法数值及隔离检查通过；无OOM/换页持续抖动；所选并行profile较串行吞吐有可重复提升且失败/超时无系统恶化。候选目标离线≥1.5倍、QC吞吐≥2倍；未达到时记录实测、选择最好稳定profile，不承诺固定倍数或以虚构收益放行。至少覆盖BC与IQL/SAC代表更新，不能只测一个MLP。

`speedup=T_serial/T_parallel`仅对同任务包计算；记录预热/编译/加载和端到端两套时间。`T_total=T_prepare+T_BC+T_IQL+Σ(T_collect_round+T_update_round)+T_strict`，流水重叠另记实际端到端，不把各项相加当历时；QC预算不除以并发后当核时。48×900秒=12小时是保障上限和，4并行理想均衡时约3小时仅作排程示例，仍受轮次屏障、长尾、排队和准备/严格验证影响。fit50得到节点数量、bytes/节点、延迟分布与核时后，回填500/2k预算和磁盘高水位（预计新增产物×1.5＋checkpoint/重试空间），当前磁盘unknown时不放行批量。

### 11.8 infra配置草案与实现依赖

以下内容保存于计划，**尚非可执行配置或现有CLI参数**。未来落点`config/pesn/direct_infra_v1.yaml`，由新loader严格校验与训练配置共同hash；ACP资源部分仅为对外需求，实际映射到公开接口后验收。

```yaml
schema: pes2ts_direct_infra_plan_v1
status: planned
runtime:
  python_path: null  # 启动前冻结已验收解释器绝对路径
  environment_lock_sha256: null
learner:
  device: cuda:0
  require_cuda: true
  processes: 1
  seeds_concurrent: 1
  precision: fp32
  compile: false
  tf32: false
  cpu_intra_threads: 2
  cpu_inter_threads: 1
  microbatch_graphs: 8
  effective_batch_source: algorithm_config
  max_nodes_per_batch: null  # fit50实测后填写，未填不放行批量
  max_edges_per_batch: null
loader:
  workers: 2
  start_method: spawn
  threads_per_worker: 1
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2
  lazy_shards: true
collection:
  policy_update: between_rounds_only
  max_inflight_episodes: 2
  inference_batch_max: 8
  inference_wait_ms: 5
acp_resource_request:  # 设计需求，不是现行ACP配置键
  global_qc_jobs: 2
  threads_per_job: 2
  global_qc_thread_budget: 12
  job_peak_memory_gib: null
limits:
  new_host_memory_gib: 8
  host_available_reserve_gib: 8
  gpu_total_used_soft_limit_gib: 7
  disk_free_min_gib: null
telemetry:
  interval_seconds: 2
  unavailable_value: null
  retain_failed_attempts: true
```

| 追加工作包 | 文件/所有权 | 交付及依赖 |
|---|---|---|
| WP-INFRA0（前置） | `scripts/probe_direct_training_env.py`、`config/pesn/direct_infra_v1.yaml`（拟新增） | 只读运行环境/依赖/资源探针；GPU代表批次为显式子命令；锁解释器与版本；WP0前可做只读探针 |
| WP-INFRA1 | `pesn/offline/training/device.py`、`direct_loader.py`、`performance.py`（拟新增） | device/seed/AMP/分桶/CPU worker与时间线；WP2模型可运行后测；WP3/4/6共用，runtime不导入offline |
| WP-INFRA2 | `generation/direct_round.py`、`direct_snapshot.py`配套与ACP侧资源任务 | PES2TS episode隔离/汇总/轮屏障；ACP队列/配额/取消/收据单一所有权；依赖WP5a/5b与ACP公开能力 |
| WP-INFRA3 | `scripts/benchmark_direct_infra.py`、`tests/pesn/test_direct_device_loader.py`、`tests/test_direct_parallel_round.py`（拟新增） | 同包计时、进程spawn、数值/恢复/无重复/配额测试；真实吞吐报告另交付，不将fake ACP并发当QC速度 |

更新依赖：**WP0＋WP-INFRA0 → WP5a → WP1 fit50（safe_start/测量）→ WP2＋WP-INFRA1 → Gate-INFRA离线小包 → WP3 fit500 → WP7a → 2k BC/IQL → WP5b＋WP-INFRA2/3 → Gate-INFRA采集小包 → WP6 SAC首包**。Gate-INFRA不代替数据、概率、严格链或科学验收门。性能调优先只改变执行布局；若改变有效batch、UTD、精度导致数值语义变化、采样分布或奖励，登记新实验变体，重新核对原比较资格。

### 11.9 WSL专项实查与GPU落地顺序（v3.2）

用户明确主要任务在WSL中运行。本轮通过隔离外获准的只读WSL调用核查Ubuntu及历史训练解释器，补齐§11.3中WSL/磁盘unknown；Windows瞬时内存与负载仍保留其采样语义。原始结果见[WSL探针JSON](fixtures/pes2ts_wsl_gpu_probe_20261015.json)，可复用` scripts/probe_wsl_gpu.py `在目标解释器重测；脚本仅查询环境、磁盘和CUDA驱动初始化，不读真值或执行QC/训练。

| 现状 | 实查证据 | 判断 |
|---|---|---|
| WSL | Ubuntu 24.04.2 LTS，WSL2，内核5.15.167.4；32逻辑线程 | 本轮GPU实施目标明确为WSL，不再以Windows默认Python判断训练环境 |
| 历史训练解释器 | `/root/miniforge3/envs/pes2ts/bin/python`，Python3.12.14；与旧资源报告路径一致 | 启动脚本绑定绝对路径，避免shell激活漂移；本次未发现活动PES2TS训练进程，不断言其他未核环境状态 |
| PyTorch | `2.14.1+cpu`；`torch.version.cuda=null`，`cuda.is_available=false`，构建`USE_CUDA=0` | **该解释器当前不能执行GPU模型训练**；只修改环境变量或多开worker无效 |
| 驱动层 | WSL可见4060Ti 8188MiB；`libcuda.so.1`存在，`cuInit=0`、device_count=1、driver API=13010 | WSL驱动入口初始化成功；无需先重装驱动，尚未验证CUDA torch前反向 |
| CPU线程 | torch intra=16、inter-op=32；OMP/MKL/OpenBLAS未显式限制 | 多进程叠加时存在超额线程风险；小图CPU训练还需比较1/4/8线程而非默认开满 |
| WSL内存 | `.wslconfig memory=56GB`、swap=0、autoMemoryReclaim=gradual；探针MemAvailable约52.16GiB | 这是guest视图，不能当Windows当前物理可用量；32线程已可见，无需先加processors |
| 存储 | 仓库为`/mnt/e/.../PES2TS`，9p/dirsync挂载；E盘仅约25.7GiB空闲 | 跨文件系统读取/小文件写入是待测瓶颈；优先lazy shard和减少频繁checkpoint，而非盲目加loader |
| WSL虚拟盘 | 注册位置`E:/WSL`，`ext4.vhdx`显示约195GiB；guest `/`报告约786GiB空闲 | 同一E盘承载VHD增长，guest空闲并非宿主物理余量；把数据复制到`/root`不会凭空获得786GiB空间 |
| 其他资源 | D盘本次可用约3.24TiB，C盘约30GiB；WSL pip缓存约11GiB | D盘可作为迁移候选，磁盘类型/吞吐待测；不自动迁移发行版、删除缓存或修改系统配置 |
| ACP | 活动服务来自`/root/miniforge3/bin/python3.12`；既有PES2TS配置还登记`envs/rph4/bin/python` | GPU训练环境与ACP执行环境分离，逐入口核对，不把ACP服务整体换成ML环境 |

**P0：先固定空间与环境变更边界。** 保留`pes2ts` CPU环境及原run，导出依赖清单后计划另建`pes2ts-gpu`（不使用system-site-packages），从官方Linux CUDA wheel安装torch，并锁定实际可解析的Python/torch/CUDA依赖及hash。GPU变体优先保持相同torch核心版本；若官方没有相应CUDA构建，明确登记版本变更并重跑兼容测试，不假定本地`2.14.1+cpu`必有同版本GPU wheel。最终安装指令以官方selector/包索引的实际可用构建为准，不按驱动13.1字样强行装同号toolkit，也不采用未验证的第三方wheel。仅需项目用到的torch，不默认安装torchvision/torchaudio。

E盘空间门先计入新环境、CUDA wheels/依赖解包、pip临时区、checkpoint、QC临时目录和VHD增长，保留至少10GiB宿主余量；估算不能容纳则先安排存储。优先评估将WSL虚拟盘迁到D盘充足空间（核验盘速与迁移前备份），迁移涉及停机和系统路径，另行安排，当前不执行；迁移前可做不复制全库的小包。pip缓存11GiB只登记可清理候选，清理是否回收宿主空间还受VHD块回收影响，不能按11GiB承诺E盘立即等额增加。安装时可用`--no-cache-dir`降低新增下载缓存，仍计解包与安装空间。

**P1：先验CUDA执行，再接业务。** 新环境验证`torch.version.cuda != null`、`cuda.is_available()`、设备名/计算能力；用小张量matmul/autograd/optimizer smoke检查真实CUDA执行与同步结果，随后测试一个代表性变长图batch。确认BF16支持后才开展AMP对照。NVIDIA官方要求WSL复用Windows驱动，WSL内不安装Linux NVIDIA显卡驱动；预编译torch wheel路径与自行编译CUDA扩展分开，只有后者另核toolkit/编译器。现在驱动API已成功，不把更新WSL/驱动作为首要性能动作。

**P2：完成device全链路，解决“装好CUDA仍CPU跑”。** 原实现具体修改清单：`offline/training/trainer.py`模型/优化器创建与恢复，`models/encoder.py`组批tensor，`offline/training/losses.py`target/valid/pair index，`offline/control_trainer.py`context/action/reward/bootstrap/target，以及`models/control_heads.py`counts/graph_id缓冲；全部与对应模型输出设备一致。仅给model调用`.to(cuda)`会在这些CPU张量处出错。新direct实现依§11.5/11.8共用device入口；旧标量模型不伪装为direct EGNN。导出保持CPU可读，checkpoint恢复包含optimizer与随机状态映射。验收关注一次完整更新与resume，不能只看`cuda.is_available=true`。

**P3：单卡稳态优化。** 初始learner=1、seed串行、FP32 eager、CPU线程2/1、loader=2；数据worker使用spawn/forkserver中经验证的显式方式，避免主进程CUDA初始化后fork。比较loader0/2/4、原子/边分桶、microbatch8/16/32＋梯度累积；有效batch和更新次数保持原配方。初始显存以当前桌面约2.4GiB占用另留余量，按全部actor/critic/target/optimizer实测，不同时开多个GPU训练seed。随后逐项试BF16、FP16+scaler、TF32、compile；敏感概率/Bellman归约保留FP32，compile只在包含编译开销的端到端受益后开启。

**P4：减少9p热路径。** 首轮保持仓库/source authority与split不变，先测原有shard读取/图构建时间，再在有空间的ext4目录建立**内容hash绑定的派生缓存**，只放已获用途资格的训练shard/静态特征及临时checkpoint，原始TS/IRC真值仍经唯一审计访问器；不借缓存绕过truth_guard。代码、依赖与频繁小文件的Linux原生路径通常更合适，但当前VHD同在E盘，先解决宿主容量再大规模缓存。若只能用D盘Windows目录作过渡，减少小文件并测吞吐，不称其等价于ext4；不在本次规划中复制数据或移动仓库。

**P5：CPU/QC和GPU分别提速。** 离线BC/IQL/SAC learner用GPU；xTB/ORCA是否GPU加速取决于ACP方法能力，不能因torch启用CUDA就声称QC变快。真实采集仍按§11.4由ACP控制2×2→4×2/8×1对照，冻结actor推理可比较CPU与GPU批量。若计时表显示大部分时间等待QC，优先QC任务并行与缓存正确性，不给GPU训练加速倍数套到整个RL墙钟。

WSL资源配置先保持现状，不自动执行`wsl --shutdown`。候选长期配置为memory=48GB、processors=32、swap=4GB，但需结合Windows+WSL同期物理内存、运行峰值和可用盘实测再决定；当前swap=0有内存不足直接失败风险，swap只作容错不作训练内存扩容。swap不得继续挤占已紧张E盘；如后续设置到D盘，先核盘速/容量与目标路径。跨guest/host监测比单看`free -h`更可信，不凭52GiB guest available放行48GiB新任务。

实施顺序为：**空间预算/CPU依赖冻结→独立GPU环境→CUDA小张量与代表图更新→业务device全链路→FP32吞吐基线→loader/AMP优化→ACP并发对照→fit500/2k及SAC验收**。本轮已完成环境实查和驱动入口初始化；新GPU环境、业务GPU更新、GPU加速倍数仍`planned/unverified`。未安装/替换依赖、迁移磁盘、修改`.wslconfig`、停止进程或运行训练/QC。**（v3.3更新：本段为v3.2时点状态；前三步已于同日执行完毕，见§11.10。）**

参考：[NVIDIA WSL CUDA指南](https://docs.nvidia.com/cuda/wsl-user-guide/)、[PyTorch官方安装选择](https://docs.pytorch.org/get-started/locally/)、[Microsoft跨文件系统性能](https://learn.microsoft.com/en-us/windows/wsl/filesystems)、[WSL虚拟磁盘空间](https://learn.microsoft.com/en-us/windows/wsl/disk-space)。

### 11.10 P0/P1 执行记录（v3.3 @ 2026-10-15）

按 §11.9 落地顺序执行前三步，证据见[GPU环境与磁盘预算执行记录](../reports/PES2TS_GPU环境与磁盘预算执行记录_20261015.md)：

| 步骤 | 结果 | 状态 |
|---|---|---|
| 空间预算 | E: 26G 可用；新环境稳态 ~7.5G＋瞬时 ~3.5G → 峰值余 ~15G/稳态 ~18.5G，均 ≥10G 保留线；**GO** | 完成 |
| CPU 依赖冻结 | `pes2ts` 环境原样保留，freeze+SHA256 落盘 [fixtures](fixtures/pes2ts_env_freeze_cpu_20261015.txt) | 完成 |
| 独立 GPU 环境 | `pes2ts-gpu`（miniforge，Python 3.12，torch **2.14.1+cu130** 官方 wheel，与 CPU 环境同核心版本）；`--no-cache-dir`；numpy/pandas/pyarrow/pyyaml 对齐 CPU 版本 | implemented |
| CUDA 执行验证 | `scripts/probe_direct_training_env.py --gpu-representative-batch`：小张量 matmul/autograd/optimizer 与合成变长图批次前反向/优化步在 cuda:0 通过，梯度有限，BF16 支持确认；缺 CUDA 时显式失败不回退 CPU | implemented |

WP-INFRA0 交付物落盘：探针脚本、`config/pesn/direct_infra_v1.yaml`（python_path/锁 hash 已回填，严格 loader 属 WP-INFRA1，当前无消费者）、双环境 freeze。另核实 VHD 195G vs guest 已用 170G，存在 ~25G 已分配块可复用，实际余量优于保守预算。

**边界**：未迁移磁盘/清理缓存/改 `.wslconfig`/动 CPU 环境与历史 run；未改业务训练代码（P2 依赖 WP5a→WP1→WP2）；未运行训练/QC；GPU 加速倍数、loader/AMP/ACP 并发对照未测量，`planned/unverified` 保持。

### 11.11 批次 1–6 执行记录（v3.4 @ 2026-10-15）

真实 RL（SAC 首包）前的业务代码批次已实现并部分真实执行，证据见[Direct 批次 1–6 实施记录](../reports/PES2TS_Direct批次1-6实施记录_20261015.md)：

| 批次 | 交付 | 状态 |
|---|---|---|
| WP0 合同/分裂 | `pesn/contracts/direct.py`、`offline/splits/direct_freeze.py`；真实 split 冻结（fit 4,964 组、隔离账本、6 selection、audit） | implemented；Gate-D 证据待评审 |
| WP5a 推进器/环境 | `generation/planning/direct_mdp.py`、`direct_environment.py`；奖励手算表 6 类 + 状态充分性测试 | fixture_passed |
| WP1 fit50 数据 | `acp/direct_observation.py`、`offline/data/direct_preprocess.py`、`direct_dataset.py`、`prepare_direct_data.py`；真实 ACP GFN2-xTB 制备（双 slice 可恢复） | **50/50 完成**：3,612 转移、3,662 QC（0 失败、3.51h 真实执行）、数据验收 pass；校准信号：goal 12/50，a_max=0.2 投影几何可达 54% |
| WP2+INFRA1 | `features/direct_graph.py`、`models/egnn.py`、`direct_policy.py`、`direct_critics.py`、`device.py`、`direct_loader.py`、`performance.py` | fixture_passed（概率合同 9 项：float64 Jacobian 1e-6 等） |
| WP3 BC | `offline/training/direct_bc.py`、`train_direct_bc.py`、`direct_bc_v1.yaml`；真实 fit50 数据 GPU 冒烟（482k 参数，cuda:0） | smoke_passed；表征：修复 σ 钳位饱和缺陷后 150-epoch NLL −78.4（每原子 +4.99 vs 先验 +2.72、oracle +7.10）；方向余弦 0.163 → 辅助方向损失版本有登记依据（[表征分析](../reports/PES2TS_Direct_fit50训练表征分析_20261015.md)） |
| WP7a 候选桥 | `generation/planning/direct_candidates.py`（frame hash/K 预算/PathBundle 适配） | fixture_passed |
| WP4 IQL | `offline/training/direct_iql.py`、`train_direct_iql.py`、`direct_iql_v1.yaml` | smoke_passed（合成 CPU）；真实 fit50 表征 3,000 更新：损失收敛、Q/V 有限、AWR ESS≈均匀（重加权无信号）、特征缓存修复 ~7× |

全仓 2051 测试通过；真值守卫全绿（修复动态执行风险，未扩 allowlist）。**未完成**：fit500/2k 制备（拟合定价后放行，当前 unpriced）、三 seed 正式训练、严格链执行、WP5b/WP-INFRA2/3 与 SAC。

## 12. 验收门与停止条件

| 门 | 量化或可核对条件 | 未通过后的动作 |
|---|---|---|
| Gate-D | 全部拟用组交集/族交集=0；来源/用途/电子态/映射明确；共享推进核对100%完整转移；所有拒绝留数 | 修split/接口，保持50小包 |
| Gate-A | 密度/逆变换/Jacobian/重参数化/等变/mask检查通过；随机动作实际执行与所记密度一致 | 修策略，禁止SAC |
| Gate-BC | 独立monitor NLL相对零均值/fit拟合尺度的同径向概率基线改善；monitor8自由rollout前10步有效观测比例候选≥90%；来源不含参考下一帧 | 保持500，诊断协同/误差累积；目标阈值fit校准后冻结 |
| Gate-LINK | monitor8两方法完整处理，至少1个BC自生成严格目标TS；preparation/候选/验证/费用链可追溯 | 区分生成与验证失败，不启动常规扩量 |
| Gate-IQL | Q/V有限、失败/成功MC分层校准可报告；ESS/clip/actor偏移可解释；同数据BC对照齐全 | 可以报告无增益，不强行通过价值收益门 |
| Gate-SAC | 首包真实收据/动作/策略版本一致，α无持续触顶、无删失假失败TD、两初始化预算相同；严格指标无已识别系统性恶化 | 保留首包结果，停止加轮/加seed，修复后新版本 |
| Gate-EVAL | 模型/环境/奖励/候选/strict/split已冻结；全部预选输入计入覆盖表；按组配对区间和全费用完整 | 不宣称独立泛化或生产提升 |

上述90%等经验值是待fit50校准的预登记候选，不是现有测量结果。校准变化在新数据/训练前写入冻结规格；不能在dev/sealed失败后回改门槛。训练loss下降、某个checkpoint生成出来和严格科学收益分别验收。

主指标分母使用冻结输入组，准备失败/未运行/删失分列，确认严格成功数除以全部预选输入作为确认覆盖率；物理失败率另给，不能把unknown当负标签训练。每组多seed/多episode不重复增加独立样本数。CI按组配对重采样，训练seed单列层级，family评测另报族数与族级敏感性。成功为0时单位成功成本未定义。

## 13. 实施顺序、规范修改与交接

基础设施前置与分阶段资源门同时按§11.8执行；下列为业务主干顺序。

依赖顺序为：**WP0 → WP5a最小环境/推进器 → WP1的50组 → WP2 → WP3的500 BC → WP7a严格链小试 → WP3/4的2k → WP5b/6 SAC首包 → 条件5k与多seed确认 → WP7b封存评测**。

WP0交付可审查的具体规范diff、direct schema/fixture、data-purpose和A(H)适用，不通过“research_only”标签自动绕过现有教师/Bellman边界。文档修订与实际schema发布分开；当前新增的是计划和元数据核查工具，宪法/guard/业务执行默认尚未改变。

首个可实施代码批次只包含新合同、split闭合、无副作用推进器、fake-ACP环境和概率测试；第二批接ACP50组及500BC；后续按上表验收。本轮不顺带重构旧残差控制器、不迁移保护目录、不修改历史权重或报告。

交接时逐项填写`planned / implemented / fixture_passed / smoke_passed`，附源码/配置hash、名单与排除账本、真实收据和未解决风险。新增direct数据/模型的生产能力默认关闭；数据规模不足、IQL无收益或SAC退化均是可接受的诚实研究结果。

## 14. 原始资料与适用边界

- [EGNN](https://proceedings.mlr.press/v139/satorras21a.html)：等变结构，不证明物理路径合理或优于其他编码器。
- [IQL原论文](https://arxiv.org/abs/2110.06169)与[作者实现](https://github.com/ikostrikov/implicit_q_learning)：expectile/优势加权策略提取；本项目预算、动作和奖励仍需自证相容。
- [SAC](https://arxiv.org/html/1812.05905v2)：最大熵目标、温度和有界变换；本计划的径向球映射、每维熵和有限逻辑预算为明确实现选择，不照搬原文性能结论。
- [PaiNN](https://arxiv.org/abs/2102.03150)、[SchNet](https://arxiv.org/abs/1706.08566)：表达能力/成本对照；实测参数、内存、吞吐后再选择。

基础设施参考：[PyTorch性能调优](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide)、[DataLoader与Windows spawn](https://docs.pytorch.org/docs/stable/data.html)、[CUDA语义与传输](https://docs.pytorch.org/docs/stable/notes/cuda.html)。多worker/pin-memory与AMP为有条件优化，最终选择以本机同包测量为准。
