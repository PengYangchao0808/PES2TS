# PES2TS Reaction-Intent Contract v1

> **SUPERSEDED（2026-10-21）**：§4（q 与 Stop 语义）与 §6.2 q 公式已由 [ReactionIntent Contract v2](PES2TS_ReactionIntent_Contract_v2.md) 与 [ADR-0030](../design/decisions/ADR-0030-V5.2全路径导航与端点停止.md) 取代（q 改为分段归一化 A=−1/TS=0/B=+1；Stop 改为目标端点到达）。其余 intent/relabel/reverse 语义仍有效。本文件保留为历史。

> 状态：**Proposed v1**（语义合同；先于 V5.1-R3 训练冻结）
> 日期：2026-10-20
> 作用：统一 `G_rxn`、reaction intent、endpoint-label invariance、reaction reversal、progress、stop、CV/Jacobian lifting 的定义
> 取代：旧文档中的 `branch` 方向语义、`endpoint-symmetric` 术语、单独 `direction token` 语义

---

## 0. 一句话

\[
\boxed{
\pi_\theta(X_t,G_{\rm rxn})
\rightarrow
(\Delta X_{K,t},\hat q_t,\hat m_t)
}
\]

- \(X_t\)：当前唯一分子三维构型；
- \(G_{\rm rxn}\)：**有方向的反应意图**（不是无序端点集合，也不是 branch）；
- \(\Delta X_{K,t}\)：关键原子下一步位移；
- \(\hat q_t\)：当前构型在指定反应路径上的有符号进度；
- \(\hat m_t\)：是否停止局部推进并进入 TS/端点验证。

**没有 branch 输入，没有单独 direction token。**

---

## 1. 术语

| 术语 | 定义 | 是否进模型 |
|---|---|---|
| `reaction_id` | 原始反应记录身份 | 否 |
| `canonical_irc_id` | 一条实际 IRC 曲线的规范身份 | 仅 split/溯源 |
| `G_src` | 反应任务定义的源端图 | 是（Reaction Intent） |
| `G_tgt` | 反应任务定义的目标端图 | 是（Reaction Intent） |
| `G_rxn` | `(G_src, G_tgt, K, CV, rules)` | 是 |
| `K` | 关键原子集合 | 是（mask/index） |
| `CV` | 低维反应坐标/集体变量 | 是（当前值 + 目标增量） |
| `q` | 有符号 IRC 进度，TS=0 | 仅监督/输出 |
| `m` | Move/No-Move / stop | 仅监督/输出 |
| `branch` | 旧数据组织槽位 | **否** |
| `travel_direction` | 采样/推理时的行进方向 | 否（由 intent 或 q 决定） |

---

## 2. Reaction Intent

\[
G_{\rm rxn}=(G_{\rm src},G_{\rm tgt},\mathcal K,\mathcal C,\mathcal R)
\]

- \(G_{\rm src}\)：源端键图；
- \(G_{\rm tgt}\)：目标端键图；
- \(\mathcal K\)：关键原子；
- \(\mathcal C\)：低维反应坐标/约束（键长、角、二面角、配位数等）；
- \(\mathcal R\)：推进规则（哪些键目标增大/减小、步长、允许松弛范围）。

### 2.1 有向图编辑

\[
\Delta B^{\rm intent}_{ij}
=
B^{\rm tgt}_{ij}-B^{\rm src}_{ij}
\]

- \(\Delta B>0\)：目标键级增大（形成/加强）；
- \(\Delta B<0\)：目标键级减小（断裂/减弱）；
- \(\Delta B=0\)：不变。

**source/target 由反应任务定义，不由历史 IRC branch 定义。**

### 2.2 反应坐标

\[
\boldsymbol\xi(X;G_{\rm rxn})
=
(\xi_1,\ldots,\xi_m)
\in\mathbb R^m
\]

例如：

```text
C–Br 距离
C–O 距离
O–H 距离
N–H 距离
```

推进目标：

\[
\Delta\boldsymbol\xi
\]

由 \(\mathcal R\) 给出，例如 C–Br 距离 +0.02 Å。

### 2.3 Jacobian 几何提升

\[
J_K=\frac{\partial\boldsymbol\xi}{\partial X_K}
\]

最小范数提升：

\[
\Delta X_K^{(0)}
=
J_K^\top
\left(
J_KJ_K^\top+\epsilon I
\right)^{-1}
\Delta\boldsymbol\xi
\]

这是经典 CV 基线；实际需处理质量权重、奇异性和多约束冲突。

### 2.4 Hybrid

\[
\Delta X_K
=
\Delta X_K^{(0)}
+
\Delta X_K^{\rm ML}
\]

或 ML 直接预测，再由约束求解器检查/修正。

三种算法臂：

| 臂 | 核心 | 用途 |
|---|---|---|
| Classical CV | Jacobian 提升 | 可解释基线 |
| Direct EGNN | 当前几何 + intent → 位移 | R3 主路线 |
| Hybrid | CV 提升 + ML 修正 | V5.2+ |

---

## 3. 方向语义

### 3.1 不需要单独 direction token

方向已经由 \(G_{\rm rxn}\) 的目标变化规定：

```text
C–Br 断裂 -> Δr_CBr > 0
C–O 形成  -> Δr_CO < 0
```

再加 direction token 是重复条件，可能冲突。

### 3.2 两种操作必须区分

#### 操作 A：仅重命名端点标签

\[
\pi(X,\mathrm{Relabel}(G_{\rm rxn}))
=
\pi(X,G_{\rm rxn})
\]

只是把 A/B 名字交换，物理任务不变，输出必须不变。

#### 操作 B：真正反转反应目标

\[
\pi(X,\mathrm{Reverse}(G_{\rm rxn}))
\ne
\pi(X,G_{\rm rxn})
\]

例如：

```text
C–Br -> C + Br      vs      C + Br -> C–Br
```

小步长下切向可能近似相反，但不要求有限步长严格互为负数。

**禁止把操作 A 和操作 B 混为一谈。**

---

## 4. Progress 与 Stop

### 4.1 有符号进度

\[
q=\tau-\tau_{\rm TS}
\]

- \(q<0\)：源侧；
- \(q=0\)：TS；
- \(q>0\)：目标侧。

\(q\) 的符号由 \(G_{\rm rxn}\) 的 source→target 方向定义。

- 交换端点名字（Relabel）：q 不变；
- 反转反应目标（Reverse）：q 符号反转。

### 4.2 q 是输出，不是输入

- 真实 IRC progress 只能作为监督；
- 推理时由模型预测；
- 不得把 TS 距离、IRC 位置、未来帧喂入 forward。

### 4.3 Stop

\(m=1\) 表示：

- TS 候选；或
- 端点到达；或
- 应停止局部推进进入验证。

TS 在 full-curve 模式是途经点，不是必然终点；在 TS-only 模式才是终点。

**V5.1-R3 不训练 Stop Head。** R3 只做 `intent_directed_irc_step` 的单步位移；Stop/Progress 留给 V5.2。

### 4.4 监督任务：intent_directed_irc_step

R3 的监督任务固定为：

\[
\boxed{
\texttt{supervision\_task = intent\_directed\_irc\_step}
}
\]

不是 `ts_directed_step`。

对一条完整曲线：

\[
A=X_0\rightarrow\cdots\rightarrow X_{\rm TS}\rightarrow\cdots\rightarrow X_N=B
\]

#### Forward intent

\[
G_{A\to B}
\]

对每个节点 \(i\)，沿曲线正方向寻找首次达到 \(d_0\) 的目标：

\[
j^+=\mathrm{first\_contact\_along\_curve}(i,+1,d_0)
\]

\[
u_i^+=X_{j^+,K}-X_{i,K}
\]

#### Reverse intent

\[
G_{B\to A}=\mathrm{Reverse}(G_{A\to B})
\]

\[
j^-=\mathrm{first\_contact\_along\_curve}(i,-1,d_0)
\]

\[
u_i^-=X_{j^-,K}-X_{i,K}
\]

- 目标不一定相邻帧；可能是插值点；
- 两个方向共享同一条 IRC；
- 同一 \(X_i\) 在 forward/reverse 下是不同的完整输入（intent 不同），因此不是冲突标签；
- R3 不需要 branch，也不需要为 TS 两侧分别设计特征。

### 4.5 样本身份与采样

\[
\boxed{
\texttt{sample\_id} =
(canonical\_irc\_id,\ node\_index,\ intent\_id,\ d_0)
}
\]

规则：

- 同一曲线的 forward/reverse 样本必须在同一 fold；
- 反向样本不是独立 IRC 样本；
- 曲线级采样权重固定：一条曲线总权重为 1，按 intent/node 分摊；
- forward-only 与 bidirectional 必须使用相同曲线级预算、优化步数、模型容量；
- 不能把“重复使用反向数据”误判为双向监督收益。

### 4.6 进度

\[
\tau_i\in[0,1],\quad q_i=\tau_i-\tau_{\rm TS}
\]

- forward：\(q(A)<0,q(TS)=0,q(B)>0\)；
- reverse：\(q_i^{\rm reverse}=-q_i^{\rm forward}\)；
- R3 只存储 \(q\) 作为元数据/未来 V5.2 监督，不训练 Progress Head；
- V5.2 才训练 Progress/Stop。

### 4.7 路径歧义

方向身份由 intent 解决，但路径歧义不一定消除：

- 同一 source/target 图可能有多个 TS 构象/通道；
- 纯构象变化可能 \(\Delta B_{\rm intent}=0\)，仅靠键图无法定义方向；
- 这时必须用 `CV_spec`、构象约束或其他任务条件补充。

R3 数据审计问题简化为：

> 给定 \(X_t\) 与完整 \(G_{\rm rxn}\) 后，教师动作是否基本成为单值、可学习的映射？

如果不是，标记 ambiguous，不强行当硬标签。

---

## 5. 推理合法性

\[
\boxed{
\text{forward 只能使用 }X_t\text{ 与合法的 }G_{\rm rxn}
}
\]

允许：

- 当前几何；
- 当前拓扑；
- \(G_{\rm src}/G_{\rm tgt}\) 图；
- \(\mathcal K,\mathcal C,\mathcal R\)；
- 当前 CV 值。

禁止：

- 历史/未来 IRC 帧；
- 真实 TS 坐标；
- 端点三维几何 \(X_A/X_B\)；
- 由端点几何计算的 \(d_A/d_B/s\)；
- 真实 progress、`near_ts`、`d_eff`、`terminal`、`capped`；
- branch ID；
- split/family/conflict class。

---

## 6. 特征通道

| 通道 | 内容 | 对称性 |
|---|---|---|
| **Chemical Context** | 元素、当前几何、无序拓扑、变化键集合 | 端点重命名不变 |
| **Reaction Intent** | \(\Delta B^{\rm intent}\)、目标 CV 增量、约束规则 | 随真实目标反转而改变 |
| **Geometric State** | 当前键长、角度、关键原子相对位置 | 只依赖当前构型 |
| **Teacher Diagnostics** | IRC 位置、历史 branch、TS、未来帧 | 禁止进入 forward |

Reaction Intent 是**合法的有符号信息**；禁止的是来自历史 branch 的方向编码。

---

## 7. 架构

```text
X_t, G_t, G_rxn
      │
      ├─ Geometry Encoder (SE(3)-equivariant EGNN)
      ├─ Intent Encoder (signed graph edits + CV spec)
      └─ Current CV Encoder
      │
      ├─ CV Head / Jacobian Lift -> ΔX_K^(0)
      ├─ ML Correction -> ΔX_K^ML
      ├─ Progress Head -> q̂
      └─ Stop Head -> m̂
      │
Constrained Relaxation (protect CV advance)
      │
Path Memory / Controller
```

### 7.1 Classical CV baseline

- 由 \(\Delta\xi\) 和 \(J_K\) 算 \(\Delta X_K^{(0)}\)；
- 不做学习；
- 用于对照。

### 7.2 Direct EGNN

- \(X_t,G_{\rm rxn}\to\Delta X_K\)；
- R3 主路线。

### 7.3 Hybrid

- \(\Delta X_K^{(0)}+\Delta X_K^{\rm ML}\)；
- V5.2+ 研究。

---

## 8. PREDICT → ACTION → RELAX → UPDATE

```text
PREDICT:
  ΔX_K, q̂, m̂ = policy(X_t, G_rxn)

ACTION:
  ΔX_K = trust_region_clip(ΔX_K, q̂)
  X_prop = X_t + ΔX_K

RELAX:
  X_next = relax(X_prop, G_rxn, protect CV advance)

UPDATE:
  update path memory, q̂, TS/endpoint checks
```

关键：

- 反应坐标推进由 intent 主导；
- RELAX 不能直接撤销 CV 推进；
- 物理验证决定接受/回滚；
- TS/端点由独立验证器确认。

---

## 9. 数据合同

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
  sample_id = (curve_id, node_index, intent_id, d0)
  state: xyz, graph
  target_first_contact (node/alpha)
  target_action u
  target_progress q
  stop_label
```

规则：

- 没有 branch；
- source/target 由 intent 定义；
- 同一曲线的所有节点、两个方向、所有 d0 同折；
- split 按 canonical IRC / near-dup component。

---

## 10. 评估

### 10.1 语义测试

- **Relabel invariance**：交换端点名字，输出不变；
- **Reverse sensitivity**：反转 intent，输出改变；
- 旋转/平移等变；
- 原子重编号等变。

### 10.2 任务

- step RMSE/cos/SRE/sign/dead；
- progress MAE/monotonicity；
- stop precision/recall；
- TS pass / IRC connected / endpoint success；
- cost。

### 10.3 基线

- Classical CV；
- Direct EGNN；
- Hybrid；
- 线性插值/GSM/NEB。

---

## 11. 术语迁移

| 旧术语 | 新术语 |
|---|---|
| branch0/branch1 | 删除；曲线节点 + intent |
| endpoint-symmetric | endpoint-label-invariant |
| dual-endpoint | reaction-intent-conditioned |
| direction token | 删除；方向由 intent 规定 |
| Geometry Policy | Reaction-Conditioned Geometric Policy |
| Physics-Corrected Policy | Physics-Corrected Reaction Growth |
| Cost-aware RL | Cost-Aware Autonomous PES Exploration |

---

## 12. 一句话

**PES2TS 学习的是：在给定有向反应意图 \(G_{\rm rxn}\) 下，如何把低维反应坐标推进转化为关键原子三维位移，并预测当前 IRC 进度与停止位置。端点重命名不变，真实反应反转可改变输出；branch 和单独 direction token 都不存在。**
