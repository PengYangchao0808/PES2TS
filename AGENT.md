# AGENT.md — PES2TS 宪法执行投影（Enforcement Projection）

> 状态：权威（执行投影，**非独立规范**）
> 层级：全仓全局；所有开发动作的前置阅读
> 规范根（SSOT）：[`docs/PES2TS_开发宪法_v1.md`](docs/PES2TS_开发宪法_v1.md) —— **任何规范冲突以宪法为准**
> 权威来源（阶段/边界）：[`docs/plans/PES2TS_阶段定义_G0-G3.md`](docs/plans/PES2TS_阶段定义_G0-G3.md)（文件保留历史名，内容 G0–G4）
> 流程总纲：[`docs/plans/PES2TS_完整开发流方案_整合版_v1.md`](docs/plans/PES2TS_完整开发流方案_整合版_v1.md)
> 当前版本：v3.16 @ 2026-10-17（无序双端TS训练与项目进度v4；宪法v2.7适用登记；旧能力证据保持原版本）

**本文件不是规范。规范在宪法。** 本文件只做三件事：(1) 强制你读宪法；(2) 把红线 **R1–R6** 以 ID + 一句话列出以便自检；(3) 存放 PES2TS 专属的操作性规则。**本文件不得新增任何规范条文**；任何分歧以宪法为准（见 §11）。

---

## 1. 项目身份（摘要，详见宪法 §1）

PES2TS 是化学约束、真实物理闭环、成本敏感的TS搜索决策工作流；PES-N是共享的反应状态价值内核。PES-G/PES-R分别使用同一内核导航/评价STOP，Classic保留为基线与已验收backup。见宪法§1–§2。
运行时产品视图为拓扑约束、控制器、ACP计算器接缝及横向可视化层；不同完整度输入与评价资格见宪法§2.5/§5.5及[产品架构设计](docs/design/PES2TS_三模块与可视化产品架构_v1_20261005.md)。这是逻辑整合，具体模块/schema/UI另验收，受保护目录与阶段出口保持原定义。
预测器作为独立代码责任提供估计，控制器负责显式选择与执行请求；本期G3-P1新增业务代码统一归`pes2ts_core/pesn/`，内部runtime与offline单向隔离。文件职责、接口及配套文件范围见宪法§2.6和[模块划分](docs/design/PES2TS_预测器产品与G3代码模块划分_v1_20261007.md)。
严格产出：OptTS + 频率（目标模态一阶鞍点）+ 双向 IRC + 完整 R/P 身份。
核心句：**PES2TS 不学势能面，它学的是在化学允许的动作下如何导航真实量子力学表面。**

---

## 2. 红线自检（派生投影，规范见宪法 §3）

> 以下是 **ID + 一句话摘要**，非完整规范。触线即架构变更（§7）。

- **R1 真值隔离** —— TS/IRC 真值只经审计访问器；不得进入生成输入 / 状态 `s_t` / 在线推理。
- **R2 物理标签唯一** —— 成功 := OptTS+Freq+双向 IRC+身份；RMSD 只作诊断。
- **R3 ACP 唯一生产执行后端** —— 不实现引擎、不直启 xTB/ORCA；CI 测试替身除外。
- **R4 ML 控制搜索 / QM 判定** —— 动作必须落在 `A(H)` 内，越界由构造拒绝。
- **R5 全量尝试与成本留存** —— 禁止成功分子独占分母；结果须带成本账本。
- **R6 评测完整性** —— 强制 reaction-family holdout；分裂防火墙；禁止对 holdout 调参。

工程政策自检（可豁免）：**P1** 接口版本化 / **P2** typed rejection / **P3** 决策层可回放 / **P4** 单一归属 / **P5** 学习默认先击败基线；研究原型可并行，受控试点及正式接入门分别见宪法§4/§6.5。

---

## 3. 阶段流（唯一主干，详见阶段定义）

| 阶段 | 目标 | 出口 |
|---|---|---|
| **G0** 数据底座 | 数据/真值隔离，新增状态与转移资格 | 历史基础已完成；增量另验收 |
| **G1** 化学先验编译 | 图审计、原子/坐标角色与合法动作 | 边界合同及24复核，新模板验收 |
| **G2** Classic物理闭环 | Shield、原生QC优化、严格验证和费用 | 24可审计终态＋≥1非参考严格目标TS |
| **G3** 教师预训练、STOP监督与残差 | G3.P与G2并行；正式1000漏斗、单点p/C/Q_STOP、校准、模仿/残差 | 教师研究包与真实决策门分别验收；非生产默认 |
| **G4** 离线Q及闭环 | 保守离线Q→冻结模型rollout→跨来源/整族评测 | 成功率门下全量成本收益与回退/部署证据 |

**规则**：新工作必须先声明属于 `G#` 或 `C/R/X`，否则不立项。禁用旧字母阶梯；旧编号出现时必须同时给出对应 G#/工作流。`demo` = **G2.T**。
科学里程碑 `M0–M5` 只在评测/论文语境使用，是 G 阶段的声明投影（宪法 §10.2）。

G3.P 的入口/四个子任务见[阶段定义](docs/plans/PES2TS_阶段定义_G0-G3.md)与[G3-P1 训练方案](docs/plans/PES2TS_G3_P1_IRC教师预训练与模型版本计划_v1_20261006.md)：数据/坐标/隔离资格后可先训练 `pesn-irc-v0.1.0`。软件、模型、数据/split/run 版本分开；教师方向/路径分数不当 STOP/MOVE-Q，IRC 帧差不当命令/生产转移（宪法§6.5/§7.3）。G2、正式G3.0–G4门继续保留。

---

## 4. 数据流与职责边界

```
R/P与H → 化学控制模板 → ReactiveState → Classic/残差候选
  → PES-N评价STOP/MOVE/FALLBACK → Shield与显式选择
  → ACP/ORCA真实优化 → post-quality → 新观测
  → STOP: 准备/OptTS/Freq/IRC/端点身份 → 隔离标签/费用
  → train离线训练 → 冻结新版本用于新运行
```


**职责分工**：`ACP 回答"计算发生了什么"；PES2TS 判断"是否为所请求的反应"`。RPH-Agent 与 PES2TS 共用 ACP 产物。在线化学质量逻辑只用端点与当前计算证据，**不导入离线真值**。
**实现要求**：PES2TS 不得内化 HPC/调度（ACP 职责）或文献 LLM 推理（RPH-Agent）；`cccp` 是 ACP 仓库内的低层 QC 包，PES2TS 不得 import。

---

## 5. 受保护配置与模块布局

- **真值守卫**：`pes2ts_core/g0/truth/truth_reader.py`（唯一访问器）、`utils/truth_guard.py` 的 allowlist 与 `SUBPROCESS_IMPORT_ALLOWLIST` 为受保护配置；扩展 allowlist 走审查。
- **模块布局（受保护）**：`generation/{planning,assembly,execution,quality}/`、`integration/acp/`、`ranking/`、`evaluation/`、`g0/`、`g1/`。迁移目录须走审查并同步文档。预测器落点`pes2ts_core/pesn/`由宪法§2.3/§2.6及ADR-0014细化；教师构建/训练/模型评测归`pesn/offline/`，不另建`evaluation/pesn_irc`。runtime及公共初始化不能依赖offline或QC/生成链；既有模块仅经离线适配器复用，具体实现与隔离边界以实施证据为准。
- **数据链身份**：`ExecutionBackend → TrajectoryRecord → PathBundle → SeedProposal → ValidationResult`。缓存/idempotency 键绑定 `geometry_sha256 + atom_order/map_hash + method/state/solvent + convergence + operation`。同键不同摘要 = 显式冲突。

---

## 6. 数学工具的适用条件（证据边界纪律）

任何引入数学工具（质量度规、测地信赖域、平行移动、谱门、图同构投影、irrep 投影、CNPI/Weyl chamber、VRI 诊断、曲率探针等）的改动，**必须在文档中同时写明其成立条件与不能声称什么**。已确立边界（见 [PES-G 收紧方案](docs/plans/PES2TS_G2_PESG收紧算法迭代方案_20261003.md) §2）：

- 质量范数**不**等同化学保护；需独立逐原子/H 伙伴门。
- 测地距离只限制运动大小，**不**编码能垒、**不**证明不跨拓扑。
- 平行移动作用于**切向量**，点经 retraction 推进。
- 加 εI 的逆是**阻尼算子**，非严格正交投影；精确投影用尺度化 SVD/伪逆。
- 连续键级权重**不是**离散拓扑不变量；谱门只作辅助。
- **无**通用、唯一、可微的"最近图同构投影"；不宣称"锁死滑移/消除分岔/保证目标 TS"。
- irrep 投影默认**不**硬置零其他模式；有限势垒构象运动保留物理轨迹。
- 路径峰 ≠ TS；`dE/dλ=0` 可由乘子抵消产生，不推出 `g=0`。
- PES-N额外数学边界见宪法§6：局部松弛分支、乘子符号/Schur补条件、g_perp非距离、失败惩罚与有限预算Q、相容转移、可校准不确定性；不得将这些近似称为无条件保证。

---

## 7. 架构变更审查协议

> 触线定义：**宪法 R1–R6**，或阶段边界（§3）、数据链（§4）、数学适用条件（§6）、任一接口（宪法 §9.6）。

### 7.1 什么算架构变更（非穷举）
- 增删/重命名阶段，或改其定义/出口判据。
- 改任一红线（R1–R6）。
- 改数据合同字段或 schema 语义（`ExecutionBackend`/`TrajectoryRecord`/`PathBundle`/`SeedProposal`/`ValidationResult`/`g2_path_v1`/`g1_v2_export_v1`/`g1_g2_boundary_v2` 等）。
- 改执行后端恒等式（ACP 唯一）、真值 allowlist、模块目录布局、活动事件范围、三类路径方法资格门、PES-N内核、宿主及模式边界。
- 引入会改变默认方法/预算/协议的新算法。

### 7.2 变更流程（缺一不可）
1. **声明**：改动内容、触及哪条（R/P/接口）、为什么必需、影响面。
2. **决策记录**：写/改 `docs/design/decisions/ADR-XXXX-*.md`（不原地改写既有结论）。
3. **版本化**：新 schema 用**新版本号**；旧产物不覆盖；旁车绑定父 hash。
4. **证据**：可审计证据（fixture/回放/真实冒烟/审计 JSON）；不得以"测试全绿"替代真实能力。
5. **同步**：宪法版本、本文件、修订台账、`docs/README.md`。
6. **不得**：静默改默认方法、静默提高预算、删除摘要检查以"修复报错"、把未冒烟能力标为 `smoke_passed`。

### 7.3 审查清单（合并前自检）
- [ ] 改动是否触及 R1–R6？如是，ADR 已写。
- [ ] 数据合同是否用了新 schema 版本？旧产物是否仍可读？
- [ ] 是否新增了绕过 ACP 的执行路径？（必须是 No）
- [ ] 是否引入真值回流推理？（必须是 No）
- [ ] 新算法是否声明了适用条件与不能声称的结论？
- [ ] 宪法 / 阶段定义 / 本文件 / 台账 / README 是否同步？
- [ ] 是否提供了真实能力证据而非仅测试绿？

### 7.4 完成状态表达的层级纪律
`unknown / implemented / fixture_passed / smoke_passed` 四层状态必须分别表达。**只有通过真实冒烟与科学证据的能力才能升级状态**。不得因"写入计划"而改变任务或能力状态。

### 7.5 豁免
仅 **P1–P5** 可申请**时限豁免**（R1–R6 不可豁免），流程见宪法 §12.3。

---

## 8. 新工作立项流程

1. 读**宪法** + 本文件 + 阶段定义 + 相关阶段方案。
2. 声明归属：`G#` 或 `C/R/X`；若不是，先登记主干。
3. 判定是否架构变更（§7.1）；如是，先走 §7.2。
4. 立项：功能归于明确的模块目录（§5）。
5. 实现与验证：小步、可审计、留证据；失败也留数。
6. 收尾：更新文档 + 台账 + README；进度用四层状态词。

---

## 9. 文档治理

1. **单一主干编号**、**一文档一职责**、**状态头强制**、**取代登记**、**报告只读**。
2. 每份 plan/design 顶部须有状态块：`状态 / 层级 / 取代关系 / 上游依赖 / 下游消费者 / 当前版本`。
3. 新增或修改计划文档后，必须同步 [`docs/plans/PES2TS_文档修订台账.md`](docs/plans/PES2TS_文档修订台账.md) 与 [`docs/README.md`](docs/README.md)。
4. 报告永不改内容；仅可加"当前阶段/作废"指针。
5. 规范只写在**宪法**；其余文档引用，不重复。

---

## 10. 当前进度指针

**2026-10-29 V5.4.1 执行完成（E*=E0 留任；registry executed）**：[V5.4.1 实施与训练报告](docs/reports/PES2TS_V541_Embedding_50K_实施与训练报告_20261029.md)——WP0–WP7 全部完成。T0-R matched E0×3（seeds 17/29/43）+ E1/E2/E3 seed17 均 37,900 updates done；**E1/E2 `scientific_failure`（H2 sign guard）、E3 `scientific_failure`（H2 sign+dead）**——三者虽 `provisional_H3=true`（ΔAUPRC +0.05~+0.46），H2 硬门不被 PROCESS 增益豁免；**E4 `gate_skipped`（core-time ratio 3.6455>2.0 成本门，非资源失败）**；E5（父门 E4）/T3（`n_selected=0`，0 补 seed、`N_core` 未膨胀）/T4（`no_new_embedding_gain`；容量/深度混杂未闭合已声明）均 `gate_skipped`。**E\*=E0 留任**：dev（589 冻结全集，三 seed mean）AUPRC 0.290702 / q_corr 0.397494 / RMSE-zero 0.910039；dev veto 不适用、无调参、tau 未重算；候选三 seed CI 合法缺席（不编造 p 值）。rollout 1,152 episodes no-physics 双口径分开（endpoint arrival 均 0.005208、TS candidate 0.055556、QC=0、非关键原子位移 0、生产 No-Move 不开）；E0 冠军 bundle 8 文件白名单+SHA+strict-load/read-only 通过（`outputs/pesn/v541_embed_50k/exports/embedding_champion_bundle/`）。**confirm 从未读取、`new_QC_calls=0`、无 verified TS/部署资格声明**。留账的失败/修复：loader WP5-CERT 单次 `EVIDENCE_MISSING` → E0×3 重训裁决（sampler 不等价）；rollout attempt-1 v1 loader SIGTERM 资源失败 + loader v2 实现修复重跑。方案转 v1.2 Executed-Frozen（§19 执行指针）；ADR-0032 增 §6 执行附录；全仓 pytest 4,954 passed/1 skipped/11 deselected（2026-10-29）；`plan_revision: 2026-10-29.1`。**V5.4.2 `joint-finetune-E*` 须另立预注册实验。**

**2026-10-23 V5.3-MP S3 完成（9/9，门 V53MP-S3-formal 通过）**：[S3 20k 主结果报告](docs/reports/PES2TS_V53MP_S3_20k主结果报告_20261023.md)——Move guard 全过且全为负退化（−9.9%~−12.0% vs R3 zero-shot）、零 NaN、attempt 15/15 恰好用尽（4 次外部中断逐次记账提额：commit 6951683/3133f99/9d19b19/a098f17）、GPU 12,699s。科学结果：ordering cross-TS 0.80（5k 时 0.57）、q corr 0.21–0.25、C 臂 endpoint AUROC 0.59–0.60 / AUPRC 0.24（prevalence 0.175）、联合监督对 Move 同评估集代价 +1.9%（endpoint 项仅 +0.13%）；C_s29 endpoint 头误停 0.63（seed 不稳定，"训练资格≠部署资格"再实证）。闭环门 V53MP-S3-closed-loop 通过（[闭环评估报告](docs/reports/PES2TS_V53MP_S3_闭环评估报告_20261023.md)：双预算分开报、864 episode、C 在线停 seed 分裂 s29 82/96、No-Move 控制权维持未授予）；C_s43 已导出 `pes2ts_move_process_bundle_v1`（`outputs/pesn/v53mp/exports/v53mp_20k_C_s43_bundle/`，白名单 7 文件 + 账本绑定）；S4（50k–100k）另议。

**2026-10-22 V5.3-MP S1 完成 / S2 进行中**（历史）：[S1 冒烟报告](docs/reports/PES2TS_V53MP_S1_2k冒烟报告_20261022.md)（三臂 ok、Move guard 全过、GPU 1,829s≤上限）+ [S0 报告](docs/reports/PES2TS_V53MP_S0_数据建库与资格报告_20261022.md)（全部门通过：20k 冻结池 + dev 589 + confirm 2k 封存）。λ 探针 verdicts：endpoint calibrated / q target_unreachable / rank scale_unreachable（上限运行并记录）。当前 S2（5k，三臂×三 seed，density-preserving joint 3,750 更新）在跑；S3（20k，joint 15,000 更新）待 S2 通过后授权。任务合同 [MoveProcessContract v1](docs/contracts/PES2TS_MoveProcessContract_v1.md)；文档门 `--strict` 训练前强制。

**2026-10-22 V5.3-MP S0 完成（全部门通过，S1 待 λ 探针 + 授权）**：[S0 数据建库与资格报告](docs/reports/PES2TS_V53MP_S0_数据建库与资格报告_20261022.md)——20,000 合格反应冻结池（2k⊂5k⊂20k 嵌套）+ dev 589 + confirm 2,000（仅列名封存）；S0-B 经审计 truth bridge 建库 26,038 曲线/5.96M 样本；endpoint_label_spec_v1 硬门全过（TS/起点 0% 正例，覆盖 99.62%）；S0-F 就绪 7/7 + 表校验 10/10。当前任务合同 [MoveProcessContract v1](docs/contracts/PES2TS_MoveProcessContract_v1.md)；实施层全部落地（heads/loss/trainer/driver/loader/labels/export + λ 探针）。S1 2k 冒烟：λ gradient-share 探针回填 config → `s1_authorized` 翻转 → plan-init → 3 臂 × seed17。GPU 在 S0 通过前保持 0；本轮 S0 完成后方启动 S1。

**2026-10-21 V5.3-MP MOVE–PROCESS（ADR-0031，当前）**：V5.2 / V5.2-R 已关闭（D2-G STOP，T1 不授权，负结果归档）。当前唯一任务合同为 [MoveProcessContract v1](docs/contracts/PES2TS_MoveProcessContract_v1.md)：一个共享 EGNN，两个头——MOVE（关键原子位移，跨越 TS）+ PROCESS（有向进度 q、端点完成置信度 p_end、ordering）；No-Move = p_end ≥ τ_end 且目标化学状态一致；q 仅辅助/诊断，TS 永远继续 MOVE。输入边界不变（当前单帧 + 合法有向拓扑；禁止参考端点 XYZ/历史/真值）。执行计划 [V5.3-MP](docs/plans/PES2TS_V5.3-MP_大规模MOVE-PROCESS联合训练计划_v1_20261021.md)：S0 数据资格/建库（CPU）→ S1 2k 冒烟 → S2 5k → S3 20k 正式。实测：当前六表仅 50 曲线/12,034 行；~200k 反应库尚未转 V5 六表，S0 是真实建库而非配置开关。S0 通过前 GPU=0；S1 需单独授权。文档门 `python scripts/agent_ops/docs_consistency.py --strict` 训练前强制。

**2026-10-21 V5.2-R Full-Path Navigation 任务纠偏（ADR-0030，历史，已关闭）**：审计确认 V5.2 的 Stop 标签是 TS 邻域且终点行被 terminal 排除（见 docs/reports/PES2TS_V52_任务纠偏审计_20261021.md）。该线以 D2-G STOP 结束；[FullPathNavigationContract v1](docs/contracts/PES2TS_FullPathNavigationContract_v1.md) 已由 ADR-0031 / MoveProcessContract v1 取代。旧 V5.2 负结果仅对旧任务成立。

**2026-10-20 V5.1-R3 Reaction-Intent revision**：V5.1a 五折门被拒，V5.1a-R1 无候选准入；**V5.1b 继续关闭**。当前活跃合同为 [ReactionIntent Contract v1](docs/contracts/PES2TS_ReactionIntent_Contract_v1.md)：`π(X_t,G_rxn) → (ΔX_K, q̂, m̂)`，`G_rxn` 是有方向反应意图，不是 branch；Relabel invariant、Reverse sensitive；不添加单独 direction token。当前阶段计划为 [V5–V7 v3.0](docs/plans/PES2TS_V5-V7_反应意图条件化与低维生长方案_v3.0_20261020.md)：V5=Reaction-Conditioned Geometric Policy、V6=Physics-Corrected Reaction Growth、V7=Cost-Aware Autonomous PES Exploration。立即执行 [V5.1-R3 v3](docs/plans/PES2TS_V5.1-R3_反应意图条件化与端点标签不变重训方案_v3_20261020.md)；v2.0/R2 仅为历史。产品设计见 [单端点 IRC 控制器 v3](docs/plans/PES2TS_单端点IRC生成控制器_产品设计_v3_20261020.md)。本轮未实现脚本、未启动训练/QC、未解除封存。

**以下为旧版本证据及历史指针（2026-10-17 及更早）**：

**2026-10-17 v4.1 收口**：[v4.1 阶段最终报告与下一阶段决策包](docs/reports/PES2TS_v4.1阶段训练_修复输入语义与最小实验矩阵_最终报告_20261017.md)——输入语义修复（观测-only 5 通道/ADR-0022）+ A/B/C×3 种子矩阵收口；注册门 REGISTER_ABLATION，B NOT EFFECTIVE、D 跳过；附包归因/扩数据/架构增强三选项决策包。报告只读，不改写冻结件。

**2026-10-15 当前训练入口**：[双端训练与项目进度统一方案v4](docs/plans/PES2TS_双端拓扑条件训练与项目进度统一方案_v4_20261015.md)、[ADR-0021](docs/design/decisions/ADR-0021-无序双端拓扑与完整双侧TS教师主线.md)、[合同v2草案](docs/contracts/PES2TS_双端TS教师与相容转移合同_v2.md)。当前采用无序双端完整拓扑→完整A→TS/B→TS教师→逐点xTB→规范化EGNN→BC→相容IQL。WP0–WP7映射G0/G1增量、G3.P/G3.2研究及G4.0，G2/X严格链并行，SAC后置；研究适用见宪法v2.7 §6.7。首轮0.1Å；旧100步/产物势不截断教师；500/2k依新门和完整路径定价。新版本尚无实现/数据/训练验收，WP0本次仅文档登记完成；既有能力不因规划升级。

**以下为旧版本证据及历史指针**：与上段排期冲突时按v4；旧fit50、σ修复及真实运行证据仍有效于旧任务，旧合同/终态/权重不转写为新任务。旧记录“未训练/未实现”和“当前下一阶段”等表述仅限其当时版本，不作本日总进度。


WSL专项/训练链路：实施计划v3.4 §11.9–11.11。GPU环境与P0/P1已完成；批次1–6 direct链路已实现并真实执行：合同/split冻结（fit 4,964组）、共享推进器、ACP E/g接缝、fit50 50/50真实制备（3,612转移、3,662 QC、0失败、验收pass）、EGNN/径向策略/概率合同、BC/IQL训练器。fit50训练表征（[分析](docs/reports/PES2TS_Direct_fit50训练表征分析_20261015.md)）：BC修复σ钳位饱和缺陷后150-epoch NLL −78.4（每原子+4.99，oracle +7.10，仍上升）；方向余弦0.163（NLL方向信号0.2%→辅助方向损失版本有据）；IQL 3,000更新ESS≈均匀（重加权无信号）。全仓2052测试通过。待决：a_max校准决策（0.2候选投影几何可达54%）、fit500/2k定价放行（500≈70.2h串行/35.1h二路）、三seed正式训练、严格链与SAC。

基础设施增补：实施计划现为v3.1（§11.3–11.8），含本机资源核查、单GPU训练/多环境采集、ACP全局QC配额、内存/显存保护、WP-INFRA0–3与Gate-INFRA。起点2×2 QC线程；旧训练CUDA接线与实际运行环境待验收，未启动训练/QC。


- 最新用户确认研究方向：[EGNN→IRC-BC→IQL→真实SAC实施与训练计划v3](docs/plans/PES2TS_EGNN_BC_IQL_SAC实施与训练计划_v3_20261015.md)/[ADR-0020草案v1.3](docs/design/decisions/ADR-0020-EGNN直接位移与BC-IQL-SAC研究主线.md)。本轮2k独立fit主训练、50/500前置、fit12真实SAC，文件级代码/预算/目标明确；共享推进器和严格链适配先于扩量。元数据审计已交付，旧5k去重/开发排除后4,977候选，正式族/用途资格待验收。新训练业务WP仍planned；WP0先登记动作/教师用途及规范适用，不扩guard、不改生产默认。旧证据保持原身份。

- 当前下一阶段：[IRC20k与RL2000详细方案v1.0](docs/plans/PES2TS_IRC20k与RL2000下一阶段详细方案_v1_20261008.md)/[ADR-0019规划](docs/design/decisions/ADR-0019-IRC20k与RL2000分阶段扩量.md)。IRC按20k独立fit反应、RL按新2k反应主episode计数；先补四项审核缺陷、多episode/回放与跨反应支持，再分批50/200/500/1000/2000。新任务均planned；模型、能力和收益另验收，主实验QC包络982小时15分，训练/制备/严格验证另列。
- 上一轮纠错：[方案v1.1](docs/plans/PES2TS_RL下一轮代码修复训练与调整方案_v1_20261008.md)/[ADR-0018草案](docs/design/decisions/ADR-0018-RL纠错与动作支持研究计划.md)。第一批BC模式修复与纠错权重证据成立，CQL复现旧权重；独立审核27通过/1跳过，未重跑声称的全量1966。context匹配、TD白名单、registry贯穿和审计缺源四项未关闭；WP-2不能全验收，真实配对/新采集/收益尚未成立。round0与纠错包/报告保留，实施前正式接口登记继续适用。

- 已完成：G0；G1 图/审计/统计/v2 导出；G2.0 装配；ACP 接缝 + xTB PATH ACP 化 + `--register`。
- 进行中：G1 边界 v2；G2.2–2.3 分支稳定与严格验证；G2.4 执行统一收尾；G2.T 24 条。
- 设计已登记：G3.P/G3-P1 并行 IRC 教师预训练、模型目标 `pesn-irc-v0.1.0` 与合同草案；见[ADR-0013](docs/design/decisions/ADR-0013-G3并行IRC预训练与模型版本体系.md)。
- 本期代码任务按[G3-P1 v1.2](docs/plans/PES2TS_G3_P1_IRC教师预训练与模型版本计划_v1_20261006.md)和[模块划分 B0–B7](docs/design/PES2TS_预测器产品与G3代码模块划分_v1_20261007.md)推进；ADR-0014仅调整产品归属，真值窄桥登记在B2实现时另验收，本次规划未改guard。
- G3-P1：用户确认实施进行中，pesn/config/tests目录已存在；继续原B0–B7，具体数据/权重/硬件/评测以该任务证据更新，本次未重新验收。正式G3价值头及G4策略收益未由本计划证明。
- 当前未来开发计划：[PES-N v2](docs/plans/PES2TS_PES-N未来开发计划_v2_20261005.md)。
- 当前 Classic：[G2-AB3 功能整合与原生扫描](docs/plans/PES2TS_G2_AB3功能整合与原生扫描改造方案_20261006.md)已实现并有24例真实产物（13 complete/9 partial/2 rejected，validated=0）；`g2t-native-run`显式分支，未切生产默认。算法见[设计](docs/design/PES2TS_Classic图论驱动与xTB原生约束扫描设计_v1_20261006.md)/[ADR-0012](docs/design/decisions/ADR-0012-Classic图论驱动与xTB原生约束扫描.md)。
- 统一基线：[统一实施方案 v1.1](docs/plans/PES2TS_统一残差决策与G2有限回退实施方案_v1_20261007.md)/[ADR-0017](docs/design/decisions/ADR-0017-统一残差决策与G2有限回退.md)：包 A/B/C 的合同、G2共享单步/宏步、残差/Q、训练器及CLI已有fixture证据；真实round0及训练模式/动作支持缺陷见当前下一轮指针。AB4包D独立门、正式阶段与默认门仍按宪法；新修复/注意力/扩量须另验收。
- 当前小批设计：[G2-AB4 ORCA驱动xTB试点](docs/plans/PES2TS_G2_AB4_ORCA驱动xTB约束优化试点方案_20261007.md)/[ADR-0016](docs/design/decisions/ADR-0016-ORCA驱动xTB硬约束优化试点.md)：3失败点+1正常点已冻结源产物hash；先公开工作流与4点对照，再考虑共享预算的有限回退。设计已完成，接口/回退/真实试点未实施，不修改G3或默认。

---

## 11. 与宪法的关系（对齐律）

- 本文件是宪法在代理执行面的**派生投影**，只含引用与摘要，**不新增规范**。
- 若本文件任何语句与 [`docs/PES2TS_开发宪法_v1.md`](docs/PES2TS_开发宪法_v1.md) 分歧，**以宪法为准**，并须立即修正本文件。
- CI 文档权威 lint 须校验本文件 §2 与宪法 §3 的一致性（宪法 §12.4）。
- 任何与本文件冲突的临时做法，一律以**宪法**为准；如需突破，先改宪法并走 §7 审查。
