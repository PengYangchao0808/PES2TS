# PES2TS FullPathNavigationContract v1（规范 / 设计冻结）

> 状态：**SUPERSEDED（2026-10-21，ADR-0031 / MoveProcessContract v1）**。本文件保留为 ADR-0030 历史证据；新的 MOVE/PROCESS/No-Move 语义以 [MoveProcessContract v1](PES2TS_MoveProcessContract_v1.md) 为唯一真相。本文中的 Endpoint Stop 单一 `d_end` 语义已降级为诊断量；任务纠偏审计（ADR-0030）结论继续有效。
>
> 原状态（历史）：**Active-Normative**（2026-10-21 起为本仓库 Move/Stop/TS/q 语义的唯一任务合同）
> 决策记录：[ADR-0030](../design/decisions/ADR-0030-V5.2全路径导航与端点停止.md)
> 上游依赖：[开发宪法](../PES2TS_开发宪法_v1.md)（v2.8 增量）、[ReactionIntent Contract v2](PES2TS_ReactionIntent_Contract_v2.md)、[GeometryPolicyContract v2](PES2TS_GeometryPolicyContract_v2.md)
> 取代：GeometryPolicyContract v1 §4.2/§5.2、ReactionIntent Contract v1 §4、30h 计划 §5.3/§5.6 的 Stop 语义（详见 ADR-0030 §5）
> 证据：[V5.2 任务纠偏审计 20261021](../reports/PES2TS_V52_任务纠偏审计_20261021.md)

**单一真相规则**：本文件 + 宪法相关增量 + ADR-0030 是 Move/Stop/TS/q 语义的唯一规范来源。其他文档只能引用本文，不得重新定义；代码命名、标签 schema、评估指标必须与本文一致。任何变更必须走"新 ADR + 新合同版本 + `scripts/agent_ops/docs_consistency.py --strict` 通过"，禁止静默修改。

## 1. 任务定义

**PES2TS — Reaction-Intent-Guided Full-Path Navigation**

\[
(X_t, G_{\rm rxn}, H)\ \longrightarrow\ (\Delta X_{K,t},\ \hat q_t,\ \hat m_t)
\]

- **Forward intent**：R → TS → P；
- **Reverse intent**：P → TS → R；
- **Move（m=1）**：沿 intent 继续推进完整反应路径；允许且必须跨越 TS；
- **No-Move（m=0）/ Endpoint Stop**：模型判断当前关键原子几何已进入**目标端点**的几何邻域，不再执行关键原子位移；
- **TS**：完整路径上的一阶鞍点途经点；是沿途重要计算/诊断目标，**不触发 No-Move**；
- **Progress q**：当前构型在有向曲线上的位置，内部辅助量；
- 输入合法性不变：当前单帧几何 + 合法有向反应拓扑/约束；历史帧、TS 坐标、真实 progress、未来帧、能量/梯度一律禁止进入 forward；
- 历史轨迹可供外部执行器做事后路径确认，但不作为 V5 模型输入；
- **Model Stop ≠ Verified Reaction Path**：端点到达判断后，路径确认仍需独立的拓扑核对、物理松弛与 TS/IRC 验证流程。

## 2. 规范术语表（含弃用词）

| 术语 | 规范定义 | 状态 |
|---|---|---|
| `Move` | 沿 intent 继续完整 IRC，允许跨 TS | **规范** |
| `TS` | 一阶鞍点 / 路径途经点；只作诊断与验证事件 | **规范** |
| `No-Move` / `Endpoint Stop` | 目标端点到达判断；选中坐标 = 当前 \(X_t\) | **规范** |
| `EndpointStopHead` | 端点停止头（logit 输出） | **规范** |
| `directed ordering` | 同 curve + 同 intent 内沿 intent 方向的 q 单调排序监督 | **规范** |
| `terminal` / `no_future_contact` | Move 教师在 intent 方向无未来 contact，仅表示 Move 不可用 | **规范**（`terminal` 为历史列名，语义见左） |
| `verified TS` | OptTS + 频率（目标模态一阶鞍点）+ 双向 IRC + 端点身份 | **规范**（与模型输出严格分离） |
| `TSStop` / `TS Stop Head` | 旧 V5.2 TS 邻域停止头 | **弃用**（`LEGACY_TS_STOP`，仅存于旧 checkpoint 加载） |
| `TS-neighborhood stop` / `d_key_TS≤0.04` 作为 Stop 定义 | 旧标签 | **弃用**（`d_key_TS` 降级为诊断量） |
| `No-Move candidate`（TS 窗口语义） | GeometryPolicyContract v1 旧三档标签 | **弃用** |
| `approach-side filter` / `TS-directed Move` | 2026-10-20 错误修复草案 | **撤回**，禁止出现在任何 ACTIVE 文档 |
| `terminal = endpoint` | 把 no_future_contact 当反应终点 | **弃用** |
| `q=0 触发 Stop` | 把 TS 位置当停止条件 | **弃用** |
| branch `A→TS` / `B→TS`（V5.2+ 训练分支） | 历史 V5.1a/V4 教师隔离语义 | **历史** |

## 3. 输入 / 输出

### 3.1 输入（唯一合法前向输入）

- 当前完整构型 \(X_t\)（全部原子）；
- 有向反应图 \(G_{\rm rxn}\)（G_src/G_tgt、signed ΔB_intent、K 集合）与反应约束 \(H\)；
- 禁止：历史帧、参考 TS 坐标、真实 q、剩余弧长、future 帧、branch/split/family、能量/梯度。

### 3.2 输出

| 字段 | 形状 | 语义 |
|---|---|---|
| `u` / `ΔX_K` | [ΣK,3] | 关键原子位移（非关键原子不输出动作） |
| `q_pred` | [G] | 有向进度辅助量（tanh，A=−1/TS=0/B=+1 方向） |
| `stop_logit` / `endpoint_logit` | [G] | Endpoint Stop logit（BCEWithLogits，禁止训练时双重 sigmoid） |
| `move_flag` | bool | 在线二值决策 Move/No-Move |
| `selected_coords` | 当前 \(X_t\) | m=0 时选中坐标即当前构型，不输出新坐标 |

在线解释：`ΔX_K` 与 `Move/No-Move` 是对外接口；`q_pred`、`stop_score` 是辅助诊断，可供电外部控制器调节步长，但 **q 不得反向作为模型输入，也不得作为 Stop 判据**。

## 4. Move 标签（保留 R3 Full-Traversal）

- 教师：`intent_directed_first_contact_v1`（d0=0.08 Å，per-atom cap 0.20 Å，native_verified_v1 帧）；
- 行集合：`move_status = ok`，两个 intent、全部节点，包括穿越 TS 之后的行；
- 禁止：approach-side 过滤、TS 附近裁剪/屏蔽、按"远离 TS"剔除；
- 合法性 mask：`no_future_contact`（terminal）、gauge unresolved、conflict——只影响 Move，不影响 Stop eligibility；
- Move 质量相对匹配的 R3 Full-Traversal 基线退化 ≤2%（RMSE/zero；sign/dead 在容差内）为**硬门**。

## 5. Endpoint Stop 标签

### 5.1 定义

\[
d_{\rm end}(X,H)=\operatorname{RMSD}_K\bigl(X,\ X_{\text{target endpoint}}(H)\bigr)
\]

- Kabsch 合法刚体对齐（proper rotation），K = intent 关键原子集；
- forward 目标端点 = canonical 路径末端 B（P）；reverse = 起点 A（R）；
- 参考端点坐标只用于离线标签，禁止进入推理输入。

### 5.2 三态标签

\[
y_{\rm end}=
\begin{cases}
1,& d_{\rm end}\le w_{\rm pos}\\
0,& d_{\rm end}\ge w_{\rm neg}\\
-1\ (\text{Mask}),& \text{otherwise}
\end{cases}
\]

- 主候选窗口：**w_pos = 0.10 Å，w_neg = 0.30 Å**；
- D1 阶段可比较 0.08/0.10/0.12/0.15 的敏感性，但 T1 前必须按预注册标准（端点 recall @ precision ≥0.8，起点/TS 误正例 ≤2%，非 AUPRC 最大化）冻结唯一窗口；
- 实测分布（12034 行，50 曲线）：0.10/0.30 → 正例 1781（14.8%）/负例 8399/Mask 1854；每曲线正例 min/med/max = 9/32/90；起点端点误正例 0；TS 误正例 2/100。

### 5.3 必守规则

1. **eligibility 独立**：`stop_status` 只由几何有效性与退化 flag 决定；`terminal`（no_future_contact）行**必须**进入 Stop 数据；
2. **intent 条件对照**：同一端点几何 \((X_P,H_{R\to P})\to\) Stop、\((X_P,H_{P\to R})\to\) Move 是核心监督，不得去重；
3. **TS 默认负例**：禁止用 `d_key_TS`、q 阈值或"接近 TS"定义 Stop 正例；
4. **几何退化**：TS 与端点几何近简并的样本（审计已发现 RXN_0000001174 d_end(TS)=0.066、RXN_0000001675 d_end(TS)=0.019）单独 flag/mask，报告含/排两种口径；
5. **拓扑一致性**：D0 用 target bond-edit completion 审计"几何近但拓扑未完成"的样本，不一致者进 Mask；
6. **不混入其他语义**：不确定、碰撞风险、预算耗尽、局部位移过小都不是 Stop 正例；碰撞/预算属于外部控制器安全逻辑，单独记录。

## 6. Progress（q）

### 6.1 标签

\[
q_i=
\begin{cases}
-\dfrac{s_{\rm TS}-s_i}{s_{\rm TS}-s_A},& i\le i_{\rm TS}\\[6pt]
+\dfrac{s_i-s_{\rm TS}}{s_B-s_{\rm TS}},& i>i_{\rm TS}
\end{cases}
\qquad
q^{\rm reverse}=-q^{\rm forward}
\]

- 质量加权弧长分段归一化；A=−1、TS=0、B=+1；零弧段 mask（现行 `progress_q_from_arcs_v52`，不变）；
- 本定义取代 ReactionIntent Contract v1 §4.1 的 \(q=\tau-\tau_{\rm TS}\) 表述。

### 6.2 训练用途

- 主目标：**directed pairwise ordering**——同 curve + 同 intent 内，沿 intent 更靠后 → q 更大；允许跨 TS 构对；
- q 回归（Huber）仅为可选辅助损失；
- 禁止：q=0 作为 Stop 判据；q_pred 输入 Stop/Move head；以 q MAE 作为唯一验收门槛；
- 反向一致性检验：\(q(X,\mathrm{Reverse}(G))\approx -q(X,G)\)。

## 7. 标签 schema：`pes2ts_fullpath_nav_labels_v1`

| 字段 | 值域 | 说明 |
|---|---|---|
| `move_status` | `ok` / `no_future_contact` / `masked` / `conflict` | Move eligibility（继承 R3） |
| `move_target_u` | [K,3] 或 null | R3 first-contact 目标 |
| `stop_status` | `endpoint_pos` / `endpoint_neg` / `endpoint_ambiguous` / `invalid` | Stop eligibility（含 terminal 行） |
| `endpoint_stop_label` | 1 / 0 / −1 | 三态端点标签 |
| `target_endpoint` | `A` / `B` | 由 intent 方向决定 |
| `d_end_angstrom` | float | 到目标端点的 K-RMSD |
| `q_signed` | [−1,1] 或 null | 有向进度 |
| `q_status` | `ok` / `zero_arc_masked` / `invalid` | q eligibility |
| `ts_stop_legacy` | 1 / 0 / −1 | 旧 TS 邻域标签，仅 ablation |

旧 `pes2ts_e4_labels_v1` 保留只读，用于解释已执行的 V5.2 负结果。

## 8. 模型与损失

- 背骨：R3 winner（F2-bidir）EGNN 初始化；D1 前禁止架构变更；
- 头：`ProgressHeadV3`（tanh）+ 新 `EndpointStopHead`（logit）；旧 `TSStopHeadV3`（LEGACY_TS_STOP，弃用）仅用于加载旧 checkpoint；
- EndpointStopHead 输入由 D1 冻结编码器探针决定（z vs key-pooled h），这是 head 输入选择，不是架构扩容；
- 损失：

\[
L=L_{\rm move}+\lambda_{\rm end}L_{\rm end}+\lambda_{\rm ord}L_{\rm ord}+\lambda_qL_q
\]

- 统一 batch + `move_mask`/`end_mask`/`q_mask`：terminal 行进入 Endpoint/Ordering，不进入 Move；
- λ 用 encoder 梯度占比校准（2–10% 探索范围），不做大网格；
- 分阶段：冻结编码器 head warmup → Endpoint/Ordering 独立 → 联合加入 Move → 可选 q；
- Move guard（≤2%）为硬门：破防 run 一律 invalid，不得以 Endpoint 指标交换。

## 9. 评估与验收

| 维度 | 指标 | 门槛（预注册候选） |
|---|---|---|
| 起点 | 起点误停止率 | ≤2%，理想 0 |
| TS | TS 附近误停止率 | ≤5%，单列 2 个退化样本两种口径 |
| 跨越 | TS 跨越完成率 | 相对 teacher-oracle 不显著低于 R3 |
| 终点 | 目标端点到达率 | 预算内相对 R3 提升 |
| Stop 位置 | 首次 Stop 的 d_end | 中位 ≤0.10 Å，报分布 |
| Move | RMSE/zero、sign、dead | 相对 R3 退化 ≤2% |
| Progress | 全路径 / 跨 TS ordering | mean >0.65 / 跨 TS >0.6 |
| Stop 分类 | AUROC + AUPRC（同时报 prevalence） | AUPRC ≥2× prevalence，CI 下界 > prevalence |

- 统计：cluster bootstrap（by curve，1000 次）+ 3 seeds（17/29/43）；单折/单 seed 不得晋级；
- Stop 阈值在 inner-val 按目标 precision 校准后冻结；
- free rollout 与 physics-relaxed rollout 分开报告；
- 模型输出最多称 `TS_candidate` / `endpoint_reached`，不得称 verified TS/path。

## 10. Rollout 预算

\[
B=\max\left(40,\ \left\lceil 1.5\cdot L_{\rm key}/0.08\right\rceil\right)
\]

- 依据：关键原子弧长中位 1.74 Å，d0=0.08 下估算步数中位 22、71% 曲线 >20 步；
- 20 步预算仅作历史对照；报告 `budget_limited` 比例；
- TS 跨越判定：将生成帧投影回参考 IRC（K-RMSD 最近节点），弧坐标越过 s_TS 即记跨越（诊断用，不声称生成路径等于 IRC）。

## 11. 阶段与门

| 阶段 | 内容 | 通过条件 |
|---|---|---|
| **D0** | 标签/provenance/文档审计 | 终点正例进入 Stop；起点/TS 误正例可控；commit+hash 齐全；docs gate 通过 |
| **D1** | 冻结 R3 编码器，独立 Endpoint Head + directed ordering | 端点判别优于基线；起点/TS 误停低；跨 TS ordering 有效 |
| **T1** | Full-Traversal Move + Endpoint + Ordering 联合 | Move ≤2% 退化；Endpoint 不明显低于 D1；TS 不误停 |
| **R** | 全路径 rollout（长度感知预算） | TS 跨越、端点到达、Stop 位置、Move 质量达标 |

dev folds 0–2 only；folds 3/4 继续封存；V5.1b/OOD/Fit150+ 边界不变。

## 12. 禁止项（红线）

1. 禁止任何 ACTIVE 文档重新定义或另述 Move/Stop/TS/q 语义（docs gate 强制）；
2. 禁止用 `d_key_TS`、q 阈值、TS 邻域构造 Stop 正例；
3. 禁止 approach-side 过滤 / TS-directed Move / TS 附近裁剪；
4. 禁止把 `terminal`（no_future_contact）当反应终点，或借其排除端点 Stop 行；
5. 禁止把历史帧、TS 坐标、真实 q、未来帧、能量/梯度喂入 forward；
6. 禁止以 Endpoint 指标交换 Move 退化；
7. 禁止把 Model Stop / q≈0 / RMSD 接近当作 verified TS；
8. 禁止静默修改本合同——必须新 ADR + 新版本 + docs gate。
