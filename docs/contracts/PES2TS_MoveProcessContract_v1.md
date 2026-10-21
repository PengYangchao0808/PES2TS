# PES2TS MoveProcessContract v1（规范 / 设计冻结）

> 状态：**Active-Normative**（2026-10-21 起为 V5.3-MP 的唯一 Move/Process/Stop 任务合同）
> 决策记录：[ADR-0031](design/decisions/ADR-0031-MOVE-PROCESS统一学习合同.md)
> 上游依赖：[开发宪法](PES2TS_开发宪法_v1.md)（v2.9 增量）、[ReactionIntent Contract v2](PES2TS_ReactionIntent_Contract_v2.md)、[GeometryPolicyContract v2](PES2TS_GeometryPolicyContract_v2.md)
> 取代：[FullPathNavigationContract v1](PES2TS_FullPathNavigationContract_v1.md) 的任务语义（该文件保留只读，作为 ADR-0030 历史证据）
> 证据：[V5.2 任务纠偏审计](reports/PES2TS_V52_任务纠偏审计_20261021.md)、[V5.2-R 输入充分性决策记录](reports/PES2TS_V52R_输入充分性决策记录_20261021.md)

**单一真相规则**：本文件 + 宪法 v2.9 增量 + ADR-0031 是 MOVE/PROCESS/No-Move/TS 语义的唯一规范来源。其他文档只能引用本文，不得重新定义；代码命名、标签 schema、评估指标必须与本文一致。任何变更必须走“新 ADR + 新合同版本 + `scripts/agent_ops/docs_consistency.py --strict` 通过”。

## 1. 任务定义

**PES2TS — Reaction-Intent-Guided MOVE + PROCESS**

\[
(X_t,G_{\rm rxn},H)\longrightarrow(\Delta X_{K,t},\mathcal P_t),
\qquad
\mathcal P_t=(\hat q_t,\hat p_{{\rm end},t},s_t)
\]

- **Forward intent**：R → TS → P；**Reverse intent**：P → TS → R。
- **MOVE**：沿 intent 继续完整反应路径；允许且必须跨越 TS。
- **No-Move**：目标端点完成判断成立；选中坐标 = 当前 \(X_t\)。
- **TS**：完整路径上的一阶鞍点途经点；是诊断/验证事件，**不触发 No-Move**。
- **PROCESS**：当前几何在有向路径中的位置（\(\hat q\)）、目标端点完成置信度（\(\hat p_{\rm end}\)）、排序分数（\(s\)）。
- **\(\hat q\)**：A=−1 / TS=0 / B=+1 的辅助量；reverse 取 −q；**不得作为 Stop 判据**。
- 输入合法性不变：当前完整构型 + 合法有向反应拓扑/约束；历史帧、参考 TS 坐标、参考端点 XYZ、真实 progress、未来帧、能量/梯度一律禁止进入 forward。
- **Model No-Move ≠ Verified Reaction Path**：端点到达判断后，路径确认仍需独立拓扑核对、物理松弛与 TS/IRC 验证。

## 2. 规范术语表（含弃用词）

| 术语 | 规范定义 | 状态 |
|---|---|---|
| `MOVE` | 沿 intent 继续完整路径，允许跨 TS | **规范** |
| `PROCESS` | Reaction Process State：位置 + 阶段 + 端点完成状态 | **规范** |
| `No-Move` | 目标端点完成判断；选中坐标 = 当前 \(X_t\) | **规范** |
| `ReactionProcessHead` | 共享 PROCESS 头的实现名（q / endpoint / ordering） | **规范** |
| `directed ordering` | 同 curve + 同 intent 内沿 intent 方向的单调排序监督 | **规范** |
| `endpoint completion` | 目标端点完成监督；待验证学习目标 | **规范** |
| `terminal` / `no_future_contact` | Move 教师在 intent 方向无未来 contact；仅表示 Move 不可用 | **规范**（历史列名） |
| `verified TS` | OptTS + 频率（目标模态一阶鞍点）+ 双向 IRC + 端点身份 | **规范**（与模型输出严格分离） |
| 旧 TS 邻域停止头 / `LEGACY_TS_STOP` | V5.2 旧语义 | **弃用**（禁止新训练使用，仅旧 checkpoint 加载） |
| 单一参考构象 `d_end ≤ 0.10 Å` 作为唯一 Stop 定义 | 旧 endpoint 分类语义 | **弃用**（降级为诊断量） |
| `q > 0.95` 触发 Stop | 把进度当停止条件 | **丢弃** |
| `terminal = endpoint` | 把 no_future_contact 当反应终点 | **丢弃** |
| `approach-side filter` / `TS-directed MOVE` | 2026-10-20 错误修复草案 | **撤回**，禁止出现在任何 ACTIVE 文档 |

## 3. 输入 / 输出

### 3.1 输入（唯一合法前向输入）

- 当前完整构型 \(X_t\)（全部原子）；
- 有向反应图 \(G_{\rm rxn}\)（G_src/G_tgt、signed ΔB_intent、K 集合）与反应约束 \(H\)；
- 禁止：历史帧、参考 TS 坐标、参考端点 XYZ、真实 q、剩余弧长、未来帧、branch/split/family、能量/梯度。

### 3.2 输出

| 字段 | 形状 | 语义 |
|---|---|---|
| `u` / `ΔX_K` | [ΣK,3] | 关键原子位移（非关键原子不输出动作） |
| `q_pred` | [G] | 有向进度辅助量（tanh，A=−1/TS=0/B=+1 方向） |
| `endpoint_logit` | [G] | 端点完成 logit（BCEWithLogits；训练时禁止双重 sigmoid） |
| `order_score` | [G] | directed ordering 分数（可与 q 共享或独立） |
| `move_flag` | bool | 在线二值决策 MOVE / No-Move |
| `selected_coords` | 当前 \(X_t\) | No-Move 时选中坐标即当前构型 |

在线解释：`ΔX_K` 与 MOVE/No-Move 是对外接口；`q_pred`、`endpoint_score` 是辅助诊断，可供电外部控制器调节步长，但 **q 不得反向作为模型输入，也不得作为 Stop 判据**。

## 4. MOVE 标签

- 教师：`intent_directed_first_contact_v1`（d0=0.08 Å，per-atom cap 0.20 Å，native_verified_v1 帧）；
- 行集合：`move_status = ok`，两个 intent、全部节点，包括穿越 TS 之后的行；
- 禁止：approach-side 过滤、TS 附近裁剪/屏蔽、按“远离 TS”剔除；
- 合法性 mask：`no_future_contact`（terminal）、gauge unresolved、conflict——只影响 MOVE，不影响 PROCESS eligibility；
- MOVE 质量相对匹配的 R3 Full-Traversal 基线退化 ≤2%（RMSE/zero；sign/dead 在容差内）为**硬门**。

## 5. PROCESS 标签

### 5.1 Progress q

\[
q_i=
\begin{cases}
-\dfrac{s_{\rm TS}-s_i}{s_{\rm TS}-s_A},& i\le i_{\rm TS}\\[4pt]
+\dfrac{s_i-s_{\rm TS}}{s_B-s_{\rm TS}},& i>i_{\rm TS}
\end{cases}
\]

reverse 取 \(-q\)；q 只作辅助/诊断，不作为 No-Move 触发条件。

### 5.2 Directed ordering

- 同 curve + 同 intent 内采样 \((X_i,X_j)\)；若 \(X_j\) 沿 intent 晚于 \(X_i\)，要求 \(\hat q(X_j)>\hat q(X_i)\)；
- 允许跨 TS 构对，不限制同侧；
- 排序为主监督；q 回归为辅助。

### 5.3 Endpoint completion

- **正例**：有独立端点身份依据、目标键编辑完成、几何合法的目标端点构型；只有通过化学有效性审计的邻域构型才可扩展为正例。
- **负例**：起点、TS 附近及明确尚未到达目标端点的构型。
- **Mask**：终点邻域归属不清、映射冲突、截断或化学判定存在歧义的构型。
- 旧的单构象 `d_end` 与 P5 basin **仅作诊断量**，不得自动升级为可靠化学终点真值。
- **eligibility 独立**：`stop_status` 只由几何有效性与退化 flag 决定；`terminal` 行**必须**进入 PROCESS 数据。
- **intent 条件对照**：同一端点几何 \((X_P,H_{R\to P})\to\) No-Move、\((X_P,H_{P\to R})\to\) MOVE 是核心监督，不得去重。
- **TS 默认负例**：禁止用 `d_key_TS`、q 阈值或“接近 TS”定义 endpoint 正例。
- **化学状态一致**：No-Move 还必须通过目标键编辑完成、价键与几何合法性检查；检查只使用当前几何 + 合法拓扑，不得使用参考端点 XYZ。

## 6. 模型与头

| 组件 | 规范 |
|---|---|
| 主干 | `StarEGNNV3`（55,441 参数），S1–S2 不扩网络 |
| MOVE | 现有 `u` 通路，保持 R3 Full-Traversal 语义 |
| PROCESS | `ReactionProcessHead`，输入共享 `z [G,64]`（可选 key-pooled `h [G,48]`），输出 q / endpoint / ordering |
| 旧进度头 | 可作为 q 子头复用，不得改变输入/输出语义 |
| 旧停止头 | `LEGACY_TS_STOP`，只读旧 checkpoint；禁止新训练/新推理 |
| 版本 | `pesn_key_atom_v53mp_heads_v1` |
| 导出 | `pes2ts_move_process_bundle_v1`：u + q/endpoint 诊断 + 冻结 τ_end + provenance |

**Attachability**：构造新模型并前向不得改变主干 `u` 的逐比特输出（同 `KeyAtomV52Model` 测试口径）。

## 7. 损失

\[
L_{\rm total}=L_{\rm Move}+L_{\rm Process},
\qquad
L_{\rm Process}=\lambda_rL_{\rm ranking}+\lambda_qL_{\rm progress}+\lambda_eL_{\rm endpoint}
\]

- \(L_{\rm Move}\)：R3 已验证组合（atom + pair + cosine + strong + dead）；
- \(L_{\rm ranking}\)：主监督；同 curve/intent 的 pairwise margin；
- \(L_{\rm progress}\)：辅助 Huber；低权重起步；
- \(L_{\rm endpoint}\)：BCEWithLogits + 类别权重 + 独立 Mask；
- 每项按独立有效样本 Mask 归一化；λ 用 encoder 梯度占比校准（2–10%），不做大网格；
- MOVE guard ≤2% 为硬门；训练有效但破防 = `scientific_failure`，不得丢弃、不得标 `invalid`；`invalid_run` 仅限泄漏/协议违规/哈希不符/运行时错误。

## 8. 评估与验收

| 维度 | 指标 | 门槛（预注册候选） |
|---|---|---|
| 起点 | 起点误 No-Move 率 | ≤2%，理想 0 |
| TS | TS 附近误 No-Move 率 | ≤5%，单列退化样本两种口径 |
| 跨越 | TS 跨越完成率 | 相对 teacher-oracle 不显著低于 R3 |
| 终点 | 目标端点到达率 | 预算内相对 R3 提升 |
| 首次 No-Move | 首次 No-Move 的 endpoint 距离 | 报完整分布；不以单点阈值代替 |
| MOVE | RMSE/zero、sign、dead | 相对 R3 退化 ≤2%（硬门） |
| Progress | 全路径 / 跨 TS ordering | mean >0.65 / 跨 TS >0.6 |
| Endpoint | AUROC + AUPRC（同报 prevalence） | AUPRC ≥2× prevalence，CI 下界 > prevalence |

- 统计：cluster bootstrap（by curve，1000 次）+ 3 seeds（17/29/43）；单折/单 seed 不得晋级；
- No-Move 阈值在 inner-val 按目标 precision 校准后冻结；
- free rollout 与 physics-relaxed rollout 分开报告；
- 模型输出最多称 `endpoint_reached` / `TS_candidate`，不得称 verified TS/path。

## 9. 在线决策

\[
\mathrm{No\!-\!Move}=[\hat p_{\rm end}\ge\tau_{\rm end}]\ \land\ [\text{目标化学状态一致}]
\]

- \(\tau_{\rm end}\) 内校准并冻结；
- \(\hat q\) 不得单独触发 No-Move；
- 碰撞、输入非法、模型不确定、预算耗尽由外部控制器记录为独立事件，不得伪装成模型 No-Move。

## 10. 声明边界

- 本合同的训练目标是 **MOVE + PROCESS**，不是已验证的化学反应路径；
- endpoint completion 是**待验证学习目标**；D2-G 的负结果对旧 T1 继续有效；
- 任何“找到过渡态/验证 IRC”的主张必须走 OptTS + 单虚频 + 双向 IRC + 端点身份确认。

## 11. 版本与取代

| 版本 | 日期 | 内容 |
|---|---|---|
| v1.0 | 2026-10-21 | 初版：MOVE + PROCESS；取代 FullPathNavigationContract v1 任务语义；旧 endpoint 标签降级为诊断 |
