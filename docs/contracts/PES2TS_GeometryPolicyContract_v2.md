# PES2TS GeometryPolicyContract v2

> 状态：**Active-Normative**（2026-10-21；No-Move/Stop 条款由 ADR-0031 取代）
> 日期：2026-10-21（v1：2026-10-20）
> 层级：V5 Geometry Policy（G3.P.0–.3）；V6/V7 继承同一状态/动作语义
> 后续决策：[ADR-0031](../design/decisions/ADR-0031-MOVE-PROCESS统一学习合同.md)（MOVE–PROCESS；2026-10-21）
> 当前任务合同：[MoveProcessContract v1](PES2TS_MoveProcessContract_v1.md)（MOVE/PROCESS/No-Move 唯一真相）
> 历史任务合同：[PES2TS_FullPathNavigationContract_v1](PES2TS_FullPathNavigationContract_v1.md)（**SUPERSEDED by ADR-0031**）
> 取代关系：取代 [v1](PES2TS_GeometryPolicyContract_v1.md) §4.2（No-Move 语义）与 §5.2（停止标签）；v1 其余条款沿袭
> 上游依赖：[V5–V7 v2.0 方案](../plans/PES2TS_V5-V7_三阶段研究与架构修订方案_v2.0_20261020.md)、[ADR-0028](../design/decisions/ADR-0028-关键原子几何策略合同与V5-V7任务拆分.md)（§2.1 No-Move 语义已由 ADR-0030 部分取代）
> 下游消费者：V5.2 标签/特征/模型、V5.3 训练、V5.4 rollout、V6 物理修正、V7 RL

---

## 1. 合同目标

冻结一个**单点、无历史、仅关键原子输出**的几何策略接口：

\[
\boxed{
\pi(X_t,G)\longrightarrow\left(\Delta X_{K,t},\,m_t\right)
}
\]

该合同只规定在线推理的输入/输出语义、禁止项与验证点，不规定网络结构、损失函数或训练算法。

---

## 2. 输入

### 2.1 必需输入

| 字段 | 类型/形状 | 说明 |
|---|---|---|
| `X_t` | `[N, 3]` float | 当前唯一一个完整分子三维构型；必须包含全部原子 |
| `atom_types` | `[N]` int | 元素类型 |
| `masses` | `[N]` float | 原子质量（用于几何/教师弧长，不要求在线网络使用） |
| `formal_charges` | `[N]` int | 形式电荷 |
| `bond_graph` | 边表 | 当前拓扑；键级、成断键意图 |
| `reaction_center` | 掩码/索引 | 合法反应中心 |
| `stereo_constraints` | 可选 | 允许的立体化学约束 |
| `K` | `[K]` int | 关键原子索引；由合法拓扑确定 |
| `G` | 结构体 | 合法拓扑/反应约束的集合；不得包含未来 IRC/TS 信息 |

### 2.2 允许的派生特征

- 原子类型、质量、形式电荷；
- 当前三维坐标；
- 键级、反应中心、成断键意图；
- 允许的立体化学约束；
- 与 `K` 相关的局部几何/拓扑描述符。

### 2.3 禁止输入

- 未来 IRC 帧、参考 TS 坐标、剩余路径长度、路径序号；
- 教师真实位移大小/方向；
- 上一轮推理的隐状态、RNN 记忆、跨调用历史；
- 任何 sealed/Confirm/OOD 真值；
- 非关键原子的输出位移（它们只参与编码）。

---

## 3. 关键原子集合 K

### 3.1 主定义

\[
K = \text{成断键事件涉及的直接原子集合}
\]

- 包括 forming/broken/order-changed 键的两端原子；
- 同一原子参与多个事件时去重；
- K 由合法拓扑约束确定，**不能依据教师真实位移大小临时筛选**。

### 3.2 扩展（单独消融）

- 一层共价邻域可作为独立动作空间消融；
- 扩展 K 必须重新预注册、重新计算标签与分母；
- 不得在主合同训练中静默切换 K。

### 3.3 非关键原子

- 参与 EGNN 消息传递/几何编码；
- 不输出位移；
- 不作为动作损失的分母；
- 在受限拉伸执行器中按冻结规则松弛。

---

## 4. 输出

### 4.1 Move

| 字段 | 类型/形状 | 说明 |
|---|---|---|
| `delta_X_K` | `[K, 3]` float | 关键原子三维位移；与输入同一坐标系 |
| `move_mask` | `[K]` bool | 合法可动原子掩码；被 mask 的原子位移必须为 0 |
| `m` | `0/1` | `1` = Move |
| `stop_score` | scalar | 可选；校准前的停止分数，不直接作为概率 |

约束：

- 只允许 `K` 行输出；非关键原子不得出现在动作张量中；
- 平移不变：对 `X_t` 整体平移，`delta_X_K` 不变；
- 旋转等变：对 `X_t` 旋转 `R`，`delta_X_K` 应旋转 `R`；
- 输出坐标系 = 输入 `X_t` 坐标系，不做隐式全局对齐；
- 位移幅度必须经过执行器/安全层的冻结 cap 处理，但网络输出本身不被静默裁剪。

### 4.2 No-Move（Endpoint Stop）

| 字段 | 值 | 说明 |
|---|---|---|
| `m` | `0` | No-Move = 目标端点到达判断（Endpoint Stop） |
| `selected_coords` | `X_t[K]` | 当前构型即停止位置；不得输出新坐标 |
| `stop_score` | scalar | 可选；用于校准/排序 |

**语义**：

- No-Move = **目标端点到达判断（Endpoint Stop）**：当前 K 几何进入该 intent 目标端点的几何邻域；不是普通不确定性拒绝；
- 选中的三维坐标就是当前 \(X_t\)；
- 它**不是** OptTS 候选初猜，也不是“已找到正确 TS”或已验证反应路径的声明；
- TS 是完整路径的途经点，**不触发 No-Move**；
- 端点到达后的路径确认必须由独立验证器完成（拓扑核对、物理松弛、OptTS/Freq/IRC 等）；
- 不得用 No-Move 绕过验证、封存或评估分母。

### 4.3 Mask / 不可辨识

- 训练标签可有 `Mask`，但**在线输出只有 Move / No-Move**；
- 对不可辨识/冲突样本，训练时进入 Mask 或低权重，不强迫硬二分类；
- 在线推理遇到合同外输入（缺失拓扑、非法 K、gauge 不可解）应返回 typed invalid，而不是伪造动作。

---

## 5. 教师标签合同

### 5.1 教师类型

| 教师 | 来源 | 用途 |
|---|---|---|
| T0 | 当前 IRC 相邻帧 | 基准关键原子位移 |
| T1 | 固定质量加权弧长上的后继 | 多尺度方向/幅度 |
| T2 | IRC 平滑曲线局部切线 | 局部运动方向 |

所有教师的在线学生输入只能是 `X_t, G`。教师标签只存在于离线训练/评估侧。

### 5.2 停止标签（Endpoint Stop）

停止对象是**目标端点**，不是 TS。定义

\[
d_{\rm end}(X,H)=\operatorname{RMSD}_K\bigl(X,\ X_{\text{target endpoint}}(H)\bigr)
\]

（Kabsch 合法刚体对齐；forward 目标端点 = B（P），reverse = A（R））。

| 标签 | 条件 | 用途 |
|---|---|---|
| `Stop`（No-Move 正例） | \(d_{\rm end}\le w_{\rm pos}=0.10\ \text{Å}\) | 硬 Stop 监督 |
| `Move`（负例） | \(d_{\rm end}\ge w_{\rm neg}=0.30\ \text{Å}\) | 硬 Move 监督 |
| `Mask` | \(0.10\ \text{Å} < d_{\rm end} < 0.30\ \text{Å}\)，或边界不确定、截断、冲突、不可辨识 | 不用于硬二分类；可作诊断 |

- 起点端点为负例；**TS 默认负例**；禁止用 `d_key_TS` 或 q 阈值定义 Stop；
- stop eligibility 独立于 move eligibility：`terminal`（no_future_contact）行**必须**进入 Stop 数据；
- 主候选窗口 0.10/0.30 Å；D1 可比较敏感性（0.08/0.10/0.12/0.15），但 T1 前必须按预注册标准冻结，禁止按 AUPRC 事后挑窗；
- 不得把“剩余 IRC 弧长 = 0”当作唯一规则；
- 不得把所有接近 TS 的帧标成 No-Move；
- 冲突样本必须保留记录，不用更大模型强行拟合。

### 5.3 多尺度跨度

- 按 IRC 质量加权弧长定义，不机械选择第 2/4/8 帧；
- 跨度值在 V5.2 协议中预注册；
- 投影到 K 个关键原子后，必须验证其在受限执行器中能产生有意义后继。

---

## 6. 运行时合同

| 项目 | 规定 |
|---|---|
| 调用次数 | 每次决策一次前向；V5 不调用 xTB |
| 历史 | 不需要；不得要求 RNN/跨调用缓存 |
| 批处理 | 允许同批多反应，但每个样本的 K 可不同；用 ptr/mask |
| 设备 | 输入/输出同设备；float32/float64 在合同中显式 |
| 失败 | typed invalid（缺失拓扑、非法 K、gauge 不可解、NaN） |
| 安全层 | cap/碰撞/价键/执行器能力由独立 Safety 层处理；网络不静默改写物理动作 |

---

## 7. 验证点（实现前必须通过）

1. **形状/掩码**：只输出 K 行；非关键原子位移恒为 0；`move_mask` 外恒为 0。
2. **平移不变**：整体平移输入，输出不变。
3. **旋转等变**：旋转输入，输出同步旋转（数值容差预注册）。
4. **无历史泄漏**：同一 `X_t,G` 在不同调用顺序下输出一致；不得依赖缓存。
5. **无教师泄漏**：输入特征中不出现未来帧、TS、剩余路径、教师位移。
6. **No-Move（Endpoint Stop）语义**：`m=0` 时 `selected_coords == X_t[K]`；不得输出新坐标；Stop 判据只看 `d_end` 端点窗口（0.10/0.30 Å），TS 途经点与 q 不触发 m=0。
7. **无效输入**：缺失/非法 K、gauge 不可解、NaN 返回 typed invalid，不返回伪造动作。
8. **评估分母**：Move/No-Move 指标使用预注册分母；Mask 样本不计入硬二分类分子，但保留在报告中；stop eligibility 独立于 move eligibility，terminal（no_future_contact）行必须计入 Stop 数据分母。

---

## 8. 版本与兼容

| 合同 | 版本 | 说明 |
|---|---|---|
| `pesn_geometry_policy_contract` | `v2` | 本文件；v1 §4.2/§5.2 已被取代，其余沿袭 |
| `pesn_geometry_policy_contract` | `v1` | 历史（见 [v1](PES2TS_GeometryPolicyContract_v1.md)，带 SUPERSEDED 标注） |
| `pesn_key_atom_move` | `v1` | K×3 位移输出 |
| `pesn_stop_decision` | `v1` | Move/No-Move + stop_score（v2 起 Stop = Endpoint Stop） |

- 旧 V5.1a/R1 的 `KeyAtomTask`/`key_atom_bc` 合同保留为历史；V5.2 不原地覆写。
- V6/V7 继承本合同的输入/输出语义；V6 可在一次调用内部增加 xTB 探测，但不得改变 Actor 的在线输入边界。
- 任何字段/语义变更必须新版本 + ADR + fixture；不得静默兼容。
