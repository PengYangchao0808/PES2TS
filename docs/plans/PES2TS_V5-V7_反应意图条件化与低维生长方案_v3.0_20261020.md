# PES2TS / PES-N V5–V7 反应意图条件化与低维生长方案 v3.0

> **v3.3 MOVE–PROCESS 修订（2026-10-21）**：V5.2/V5.2-R 已关闭（D2-G STOP）；V5.3-MP 以 [ADR-0031](../design/decisions/ADR-0031-MOVE-PROCESS统一学习合同.md) / [MoveProcessContract v1](../contracts/PES2TS_MoveProcessContract_v1.md) 为唯一任务合同：一个共享 EGNN，MOVE + PROCESS 两头；No-Move = p_end ≥ τ_end 且目标化学状态一致；q 仅辅助/诊断；TS 永远继续 MOVE。旧的 FullPathNavigationContract v1（Endpoint Stop 单一 d_end）仅为历史。V5.3-MP 的 S0–S4 与三臂 A/B/C 见 [V5.3-MP 计划](PES2TS_V5.3-MP_大规模MOVE-PROCESS联合训练计划_v1_20261021.md)。
>
> **v3.2 任务纠偏修订（2026-10-21，历史）**：V5.2 的 Move/Stop/q 语义按 [ADR-0030](../design/decisions/ADR-0030-V5.2全路径导航与端点停止.md) 与 [FullPathNavigationContract v1](../contracts/PES2TS_FullPathNavigationContract_v1.md) 修订——Full-Path Navigation（R→TS→P / P→TS→R）；Move 保留完整曲线教师并跨越 TS；No-Move=目标端点到达（Endpoint Stop，d_end 0.10/0.30 Å）；TS 是途经点不触发停止；q 为分段归一化辅助量 + 全路径 directed ordering。
>
> **v3.1 监督修订（2026-10-20）**：V5.1-R3 固定为 `intent_directed_irc_step`：一条完整 IRC、两种有向 intent、沿完整曲线 first-contact 单步监督；TS 不是 R3 停止目标。Stop/Progress 仍在 V5.2。
>
> 状态：**Proposed v3.2**（取代 v2.0；不启动训练/QC）
> 日期：2026-10-20
> 核心合同：[ReactionIntent Contract v1](../contracts/PES2TS_ReactionIntent_Contract_v1.md)
> 关系：V5.1-R3 是 V5 的局部步进基线；V5.2 起引入 Progress/Stop；V5.3 起进入单端生长；V6 物理修正；V7 自主探索
> 不变：V5.1b、sealed、Confirm/OOD、Fit150+ 继续关闭；G0–G4.2 阶段边界不变

---

## 0. 一页结论

PES2TS 的统一形式：

\[
\boxed{
\pi_\theta(X_t,G_{\rm rxn})
\rightarrow
(\Delta X_{K,t},\hat q_t,\hat m_t)
}
\]

- \(G_{\rm rxn}\)：有方向的反应意图，不是无序端点集合，也不是 branch；
- \(\Delta X_{K,t}\)：关键原子下一步位移；
- \(\hat q_t\)：当前 IRC 进度（TS=0，有符号）；
- \(\hat m_t\)：是否停止（目标端点到达）并进入后续独立验证。

三阶段重新定义：

| 阶段 | 新定位 | 核心问题 |
|---|---|---|
| **V5** | **Reaction-Conditioned Geometric Policy** | 给定低维反应坐标控制意图，能否预测关键原子三维位移、进度和停止位置？ |
| **V6** | **Physics-Corrected Reaction Growth** | 当前 xTB 局部响应能否修正动作、协调多 CV、提高生长稳定性？ |
| **V7** | **Cost-Aware Autonomous PES Exploration** | 如何在动作/验证策略中选择下一步，提高 TS 找到率并降低总成本？ |

**方向语义：**

- 端点重命名（Relabel）：输出不变；
- 真实反应反转（Reverse）：输出可改变；
- 不添加单独 direction token；
- branch 不再存在。

---

## 1. 与 v2.0 的差异

| 主题 | v2.0 | v3.0 |
|---|---|---|
| 方向来源 | 关键原子 Move/No-Move | **反应意图 \(G_{\rm rxn}\)** |
| 端点 | 双端/无序 | **Reaction Intent + endpoint-label invariance** |
| 动作 | 直接位移 | **低维 CV 推进 + Jacobian 提升 + ML 修正** |
| Progress | 未定义 | **有符号 q，TS=0** |
| Stop | No-Move | **Endpoint Stop（目标端点到达）；full-curve 中 TS 是途经点，不触发停止** |
| V5 | Geometry Policy | **Reaction-Conditioned Geometric Policy** |
| V6 | Physics-Corrected Policy | **Physics-Corrected Reaction Growth** |
| V7 | Cost-aware RL | **Cost-Aware Autonomous PES Exploration** |
| 数据 | branch/双半路径 | **曲线中心，无 branch** |

---

## 2. 共同合同

完整定义见 [ReactionIntent Contract v1](../contracts/PES2TS_ReactionIntent_Contract_v1.md)。要点：

### 2.1 Reaction Intent

\[
G_{\rm rxn}=(G_{\rm src},G_{\rm tgt},\mathcal K,\mathcal C,\mathcal R)
\]

\[
\Delta B^{\rm intent}_{ij}=B^{\rm tgt}_{ij}-B^{\rm src}_{ij}
\]

- source/target 由反应任务定义；
- 不由历史 branch 定义；
- \(\Delta B>0\) 目标键级增大，\(\Delta B<0\) 减小。

### 2.2 反应坐标

\[
\boldsymbol\xi(X;G_{\rm rxn})\in\mathbb R^m
\]

\[
J_K=\frac{\partial\boldsymbol\xi}{\partial X_K}
\]

\[
\Delta X_K^{(0)}
=
J_K^\top(J_KJ_K^\top+\epsilon I)^{-1}\Delta\xi
\]

Hybrid：

\[
\Delta X_K=\Delta X_K^{(0)}+\Delta X_K^{\rm ML}
\]

### 2.3 合法输入

forward 只能使用：

```text
X_t
G_t
G_src, G_tgt
K, CV, rules
当前 CV 值
```

禁止：

```text
历史/未来 IRC 帧
真实 TS 坐标
端点三维几何
真实 progress / near_ts / d_eff / terminal / capped
branch ID
split / family / conflict class
```

---

## 3. V5：Reaction-Conditioned Geometric Policy

### 3.1 V5.1-R3：intent-directed 全曲线单步基线

监督任务：

\[
\boxed{
\texttt{supervision\_task = intent\_directed\_irc\_step}
}
\]

不是 `ts_directed_step`。

目标：

- 端点标签不变（Relabel invariant）；
- 反应意图条件化（intent-conditioned）；
- 一条完整 IRC、两种有向 intent（A→B、B→A）；
- 沿完整曲线做 first-contact 单步位移；
- TS 只是曲线节点，不是 R3 停止目标；
- 不做 progress/stop；
- 不做 CV/Jacobian；
- 重训 B0-F，重新校准 strong/dead。

数据：

- canonical IRC 曲线；
- 无 branch；
- `sample_id=(curve_id,node_index,intent_id,d0)`；
- forward/reverse 共用同一曲线、同一 fold；
- 反向样本不是独立 IRC；
- `reaction_intent` 表；
- 1248 排除；5 个 mapping overflow quarantine；
- d0=0.08（canonical main）。

特征：

- Chemical Context：元素、当前几何、无序拓扑；
- Reaction Intent：\(\Delta B^{\rm intent}\)、目标 CV 增量；
- Geometric State：当前键长/角/关键原子位置；
- Teacher Diagnostics：禁止。

实验：

- **主实验**：forward-only，全曲线单步；
- **双向增强**：A→B + B→A 联合；相同曲线预算/优化步数/模型容量；
- **消融**：屏蔽 signed intent，只保留无序拓扑；
- 反向样本与正向共享同一 IRC、同一 fold，不能当独立 curve 计权；
- 如果双向只在“更多数据/更多更新”时更好，不算双向监督收益。

Gate：

- Relabel invariance；
- Reverse sensitivity；
- rotation/translation/permutation；
- λ=0 与 B0-F 逐比特一致；
- 梯度 2%/10% 或 unreachable。

### 3.2 V5.2：Intent + CV/Jacobian + Progress/Stop

目标：

- 加入 Reaction Intent Encoder；
- 加入 Classical CV / Direct EGNN / Hybrid 三臂；
- 加入 Progress Head（q）；
- 加入 EndpointStopHead（m，目标端点到达判断；TS 不触发）；
- Known-B，不做 full rollout。

Gate：

- q MAE / monotonicity / TS sign accuracy；
- stop precision/recall；
- 起点误停止率 / TS 附近误停止率 / TS 跨越完成率 / 目标端点到达率 / 首次 Stop 的 d_end；
- Hybrid vs Direct vs Classical 的受控对照；
- 仍不打开 sealed/OOD。

### 3.3 V5.3：Known-B 单端生长

目标：

- PREDICT → ACTION → RELAX → UPDATE；
- 从 A 出发，经过 TS，到 B；
- 路径记忆/切线/trust region；
- 反应坐标推进受 intent 保护；
- 独立 TS/IRC 验证。

Gate：

- endpoint success；
- TS verified；
- IRC connected；
- QC calls；
- 反向一致性。

### 3.4 V5.4：From Any Point

目标：

- 从曲线任意节点出发；
- q 预测决定行进方向；
- 经过 TS，到另一侧端点；
- q≈0 起点返回两侧/要求目标；
- 仍 Known-B/known intent。

Gate：

- any-point success；
- reverse consistency；
- path RMSD；
- 不承诺唯一产物。

---

## 4. V6：Physics-Corrected Reaction Growth

### V6.1：单点 xTB 响应修正

- 当前点 xTB E/g；
- MLP 残差修正 \(\Delta X_K\) 或 \(\Delta\xi\)；
- 不改 V5 主策略；
- Gate：相对 V5 有物理修正收益。

### V6.2：局部曲线 + 多 CV 耦合

- 短程受限曲线 CNN；
- 多 CV 协调；
- trust region / 信赖域；
- Gate：同物理输入下优于 MLP，或同成本净收益。

### V6.3：受约束生长 + OptTS 验证

- 真实受限拉伸；
- 反应坐标推进 + 横向松弛；
- OptTS/频率/IRC 验证；
- Gate：正确 TS 候选质量和成本优于简单扫描。

---

## 5. V7：Cost-Aware Autonomous PES Exploration

### V7.1：环境/动作/验证/成本账本

- 状态：当前几何、intent、路径记忆、预算；
- 动作：推进、松弛、回滚、提交 TS、停止；
- 成本：QC 调用、墙钟、失败；
- 未知 B：Endpoint Head 提议图编辑候选。

### V7.2：离线 RL

- BC 初始化；
- IQL/AWR；
- 只用真实转移/成本；
- Gate：独立执行优于 BC/固定控制器。

### V7.3：受控在线 RL + Unknown-B Discovery

- 在线探索；
- Top-k 产物候选；
- 严格验证；
- 成本感知；
- Gate：同预算下 TS 找到率/成本优于对照；
- **Unknown-B 只在这里做，不在 V5 做。**

---

## 6. 数据与标签

### 6.1 曲线表

```text
irc_curve
  curve_id, canonical_irc_id
  G_src, G_tgt, K, CV spec, rules
  ts_index, progress[]
  nodes[]

irc_node
  curve_id, node_index
  xyz, graph, energy, gradient
  progress, is_ts, is_endpoint
  quality

sample_view
  sample_id=(curve_id,node_index,d0)
  intent_id
  target_next_node
  target_progress q
  stop_label
  disposition
```

### 6.2 Split

- 按 canonical IRC / near-dup component；
- 同一曲线所有节点/方向/d0 同折；
- 不按帧随机；
- 不按 branch；
- 训练/评估端点身份隔离；
- 失败/替代路径单独标记。

### 6.3 数据质量

- 曲线连续；
- gauge 解析；
- 单 TS；
- 端点身份可信；
- progress 单调；
- 能量/梯度有限；
- 失败/截断显式标记。

---

## 7. 训练与损失

### 7.1 V5

\[
L_{\rm V5}
=
L_{\rm step}
+
\lambda_q L_q
+
\lambda_m L_m
+
\lambda_{\rm path}L_{\rm path}
\]

- `L_step`：Huber + cosine；
- `L_q`：progress Huber（可选辅助）+ 全路径 directed ordering（跨 TS）；
- `L_m`：BCE；
- `L_path`：endpoint/TS/barrier/cost（V5.3+）。

### 7.2 一致性

- Relabel invariance；
- Reverse sensitivity；
- \(\hat u(x,+)\approx-\hat u(x,-)\)（小步长诊断，不强制）；
- rotation/translation/permutation；
- optional cycle consistency。

### 7.3 V6

- 物理残差监督；
- 局部曲线/峰形；
- 多 CV 耦合；
- trust region。

### 7.4 V7

\[
r_t=
\beta I_{\rm success}
-\lambda_c C_t
+\eta \Delta\Phi
-\lambda_l L_t
-\lambda_f F_t
\]

---

## 8. 评估与门

| Gate | 条件 |
|---|---|
| V5.1-R3 | Relabel invariant；Reverse sensitive；λ=0 回归；梯度可达 |
| V5.2 | q MAE/monotonicity/stop；Hybrid 对照 |
| V5.3 | endpoint/TS/IRC/cost |
| V5.4 | any-point/reverse consistency |
| V6.1 | 物理修正收益 |
| V6.2 | CNN/多 CV 净收益 |
| V6.3 | OptTS/IRC 质量与成本 |
| V7.1 | 环境/成本可复算 |
| V7.2 | 离线 RL 优于 BC |
| V7.3 | 在线 RL/Unknown-B 同预算收益 |

三层成功：

1. 生长成功；
2. TS 成功；
3. IRC 成功。

---

## 9. 预算与顺序

```text
V5.1-R3: 数据/特征/损失，66 run 量级
V5.2: intent + CV + progress/stop，受控小矩阵
V5.3: known-B rollout，先 fold0/fold1
V5.4: any-point/reverse，先诊断
V6.1–V6.3: 物理修正，逐级净收益
V7.1–V7.3: 环境/离线/在线，严格成本
```

不跳过 V5.2/V5.3 直接做 V7.3。

---

## 10. 禁止事项

- 把 branch 加回输入；
- 添加单独 direction token；
- 把 reversal 当成 relabel；
- 使用真实端点几何/TS 坐标/progress 作为 forward；
- 把低维扫描路径称为 IRC；
- 未验证就称 TS/端点成功；
- 在 V5 做 Unknown-B；
- 打开 sealed/OOD/Fit150+；
- 用旧 v2.0 的 V5.2/V5.3 语义冒充 v3.0。

---

## 11. 术语迁移

| v2.0 | v3.0 |
|---|---|
| Geometry Policy | Reaction-Conditioned Geometric Policy |
| Physics-Corrected Policy | Physics-Corrected Reaction Growth |
| Cost-aware RL | Cost-Aware Autonomous PES Exploration |
| branch | 删除 |
| endpoint-symmetric | endpoint-label-invariant |
| direction token | 删除；方向由 intent 规定 |
| Move/No-Move | EndpointStopHead + 独立 TS/端点验证 |

---

## 12. 文档关系

- 合同：[ReactionIntent Contract v1](../contracts/PES2TS_ReactionIntent_Contract_v1.md)
- R3：[V5.1-R3 v3](PES2TS_V5.1-R3_反应意图条件化与端点标签不变重训方案_v3_20261020.md)
- 产品：[单端点 IRC 控制器 v3](PES2TS_单端点IRC生成控制器_产品设计_v3_20261020.md)
- 旧 v2.0：Superseded，仅历史。
