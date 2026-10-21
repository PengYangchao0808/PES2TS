# PES2TS 双端 TS 教师与相容转移合同 v2（设计草案）

> **HISTORICAL / LEGACY（2026-10-21）**：「branch 不跨 TS 串接」的双分支教师语义不适用于 V5.2+。V5.2-R 使用完整 canonical IRC 与 Endpoint Stop，见 [ADR-0030](../design/decisions/ADR-0030-V5.2全路径导航与端点停止.md) 与 [FullPathNavigationContract v1](PES2TS_FullPathNavigationContract_v1.md)。本文件保留为 V5.1a/V4 历史证据。

> 状态：设计登记；未发布可执行schema/validator；实现资格待WP0–WP6
> 层级：G0/G1增量接口、G3.P教师、G4.0研究转移
> 取代关系：新任务另版本；不覆盖旧teacher/direct/control v1/v2产物
> 上游依赖：开发宪法§6.7/§9.6、ADR-0021、统一方案v4
> 下游消费者：offline构建/训练、runtime特征/bundle、宿主与ACP接缝
> 当前版本：内容v2.0 @ 2026-10-15；具体序列化/验证器随实现冻结

## 1. 拟议接口登记与隔离

拟议名称：`pesn_dual_endpoint_task_v2`、`pesn_dual_ts_teacher_v2`、`pesn_dual_ts_features_v2`、`pesn_dual_ts_action_v2`、`pes2ts_dual_ts_transition_v2`、`pes2ts_dual_ts_reward_v2`、`pesn_dual_ts_bundle_v2`。名称中的v2表示本研究语义迭代，不表示旧schema原地兼容。

教师表与RL表独立导出。共享几何/观测使用内容身份引用，不把参考标签字段直接透传runtime；导出采用输入字段白名单。数据、split、方法、特征/归一化、采样/归约、环境、奖励、模型、run分别版本化并绑定真实内容摘要及父摘要。历史split的组/族/暴露归属不因新任务重置。

## 2. 字段族与约束

| 记录 | 内容与校验 |
|---|---|
| task | 无序完整G_A/G_B（边、键级、立体及已知mask）、元素/原子映射、charge/multiplicity、task constraints、reaction_group_id/reaction_id、来源/方法/资格 |
| branch/episode | branch_id仅追溯；episode_id唯一、真实起点、budget/environment/reward版本；允许一组多episode；分支不跨TS串接 |
| teacher state | state_id、X Å、完整原子/电子态/方法身份、节点观测引用；边界类型与缺失原因；当前教师帧仅隔离训练观测 |
| offline label | action a=X_next−X、TS剩余位移d、剩余质量弧长ell、teacher_endpoint；参考TS/未来帧/路径序号不进入输入 |
| observation | E Hartree、g Hartree/Å、实际几何/映射/电子态/方法/软件/设置摘要、原始单位与转换、ACP收据/完成状态、缓存与实际费用；缺失null+reason |
| feature spec | 完整双图交换对称编码、X与g等变向量、ΔE=E−E_episode_start；fit-only冻结尺度与mask/支持域；输入维数精确匹配 |
| action | MOVE/STOP_AND_REFINE/ABORT/已验收fallback；MOVE分别存u、无量纲z、物理a Å与a_max、变换/密度版本；STOP不是零动作 |
| transition | transition_id、state_id、next_state_id显式关联、episode_id、proposed/executed动作、真实后继/观测、逻辑成本、receipt、boundary/terminal/censored及reason、逐头eligibility与reason |
| reward | 逻辑成本与真实终局协议/结果；旧产物势停用，oracle诊断独立；未知reward/outcome/return显式缺失，不能补零 |
| bundle/run | 输入/输出资格、BC父checkpoint真实摘要、seed、超参、norm/split/dataset/action/env/reward身份；研究能力表默认不开放生产决策 |

无序身份的规范序列化/原子映射与重复归组算法在WP0冻结并验证；不只对每条边的两个键级排序而丢掉完整双图关系。A/B交换保持反应身份、分区及相同物理状态下的预测。缺失一侧不补造，纳入覆盖/资格漏斗；双侧主实验与单侧子集分列。

## 3. 资格、终止与bootstrap

teacher_endpoint只表示参考路径边界；不自动设成功或bootstrap=0。文件切片也不设终态；继续引用真实后继。预算耗尽按冻结环境为失败终态；外部超时/取消/计算缺失为删失。真实STOP结果依严格协议形成标签；若实际协议未结束，保持unknown。

Q样本需已知物理动作、相容后继及必要真实观测、已定义奖励与bootstrap语义；V及actor使用的来源/资格各自记录。没有相容后继或奖励/终局证据时保留教师资格，不伪造TD/MC。非终端样本可以在相容后继与奖励齐备时单独验收TD资格；完整MC资格另列。对没有可用后继的教师边界不强行bootstrap。

## 4. 动作、归约与可恢复性

径向映射 z=tanh(||u||)u/||u||，a=a_max z，零点用连续极限；logπ的空间/单位显式登记。Q接物理a，再用冻结尺度构造与原子/任务边联合的几何特征；不直接将a与带单位g相加。新投影/后处理重新核验密度，记录实际执行动作。

采样为组均衡→分支均衡→冻结分支内规则；主logprob有效原子维度求和，分支内平均、反应内分支平均、跨组平均。每原子NLL独立报告；epoch为每组抽到一次，组内数量、更新次数/曝光量同时记录。BC与IQL matched subset保留相同资格/采样/归约对照。

checkpoint保存actor/Q/V/目标网络、优化器、阶段/计数、采样器、CPU/CUDA RNG、norm与全部身份。先加载匹配BC，actor冻结完成2,000次Q/V预热，再最多20,000联合更新；冒烟缩短需另记阶段长度。

IQL目标候选沿用原计划的明确公式：q=min(Qbar1,Qbar2)，V拟合q的expectile（τ=0.7）；Q目标为r+γ·bootstrap_mask·V(next)，目标停止梯度；A=stop_gradient(q−V)，w=min(exp(A/T_A),100)，T_A=3（分母温度，不解释为乘数3）。目标网络更新、γ及缺失掩码随新环境在实现时冻结；有限逻辑预算原候选γ=1不自动授予未知边界bootstrap资格。BC与IQL使用同一主logprob归约；仅fit-monitor作有限选择，不为降低ESS放大权重差异。

## 5. 验收与兼容

WP0合同/validator，WP1双侧配准，WP2完整观测，WP3/4对称/等变/BC，WP5/6单位/恢复/环境转移分别留证据。至少检查X+a=X_next、d−d_next=a、无跨分支后继/分区泄漏、切片非终态、Q方向敏感、预热冻结及恢复一致。

旧3,612转移与终态只属旧环境；收据按完整计算身份条件复用，不复制旧reward/terminal。DFT教师与xTB观测双方法身份保留，参考TS不视为xTB鞍点。具体字段类型、序列化、hash规范及测试fixture实现前仍待冻结；本文不等于运行包发布。
