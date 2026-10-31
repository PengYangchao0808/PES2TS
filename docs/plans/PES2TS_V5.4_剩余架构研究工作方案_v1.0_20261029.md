# PES2TS V5.4 剩余工作方案（修订版）

> 建议状态：**Active-Proposed**。本方案用于后续预注册，不构成训练授权。  
> V5.4.1 已冻结报告不得回改；V5.4.2–V5.4.4 均须另立实验登记。

## 1. 冻结起点与仓库现状

V5.4 后续必须从以下事实出发：

- **V5.4.1 冠军仍是 E0 + 三层 StarEGNNV3**，不得将 E3 或其他 PROCESS 增益候选升格为基线。
- E0 三 seed dev 表征：
  - Endpoint AUPRC：**0.290702**
  - q correlation：**0.397494**
  - MOVE RMSE/zero：**0.910039**
- E0 no-physics autonomous rollout 共 **576 episodes**：
  - 目标端点到达率：**0.52%**
  - TS candidate 跨越率：**5.56%**
  - 发生碰撞的 episode：**565/576 = 98.1%**
- 当前主线只有 `StarEGNNV3` 完整实现；GeoEGNN、Reaction Query、Graph Transformer、GraphGRU、Flow Matching 尚未进入该主线。
- 仓库中的 PaiNN 属于旧 dual-ts 轨道，**不能直接视为当前 key-atom 单帧主干的现成实现**。
- 当前 MOVE readout 内嵌于 StarEGNNV3，尚无不依赖具体 Backbone 的统一 Decoder 接口。
- 完整 IRC 的授权流式读取接口已经存在，但教师—学生蒸馏协议、训练目标和评估流程尚未实现。

因此，V5.4 剩余阶段应研究**空间表示、动作生成和离线路径知识**，而不是继续增加 Embedding 分支。

---

## 2. 不可变的研究合同

### 2.1 在线 Student

每次前向只能使用：

\[
(X_t,G_{\rm rxn},H)
\]

即当前完整三维构型、有向反应拓扑和合法反应约束。

禁止输入：

- 历史帧、未来帧或完整 IRC；
- 真实 q、剩余弧长；
- 参考 TS 或参考端点 XYZ；
- 教师隐藏状态、路径位置或序列编码；
- 能量、梯度及其他 V6 物理输入。

输出仍为一个 MOVE 动作和 PROCESS 辅助量。非关键原子不得输出位移。

### 2.2 离线 Teacher

Teacher 只能在授权训练曲线上读取完整 IRC，包括前后帧和路径顺序。其知识必须通过训练标签、损失或蒸馏目标转移给 Student，**不得进入 Student 的推理接口**。

### 2.3 No-Move 与 TS

\[
\mathrm{No\!-\!Move}
=
[\hat p_{\rm end}\ge\tau_{\rm end}]
\land
[\text{目标化学状态一致}]
\]

- q 不得单独触发停止；
- TS 是途经点，始终 MOVE；
- V5.4 的离线 rollout 不自动开放生产 No-Move；
- 只声明 `endpoint_reached` / `TS_candidate`，不声明 verified TS。

---

## 3. 总体路线

| 阶段 | 核心科学问题 | 实验对象 | 出口 |
|---|---|---|---|
| V5.4.1 已完成 | 哪种前置特征增强有效？ | E0–E5 | E0 留任，负结果冻结 |
| **V5.4.2** | 哪种单帧空间主干最有效？ | EGNN、PaiNN、Reaction Query、Geometry Transformer | 最佳合格 Backbone，允许仍为 A0 |
| **V5.4.3** | 怎样把空间特征转化为更合理的动作？ | Direct、Tangent-aware、Conditional Flow | 最佳合格 MOVE Decoder |
| **V5.4.4** | 完整 IRC 能否提供单帧模型可学习的额外知识？ | TCN、BiGRU、Path Transformer Teacher | 蒸馏后的单帧 Student 或负结果 |
| **V5.4 收尾** | 单步收益能否转化为多步稳定推进？ | 基线、各阶段冠军、最终 Student | 统一闭环评估与架构冻结 |
| V6 | 纯几何无法解决的问题能否由物理反馈改善？ | xTB、Physics-MLP、1D CNN | 另行规划 |

不执行 Backbone × Decoder × Teacher 全组合。

---

# 4. V5.4.2：单帧空间 Backbone

## 4.1 候选矩阵

| 编号 | 架构 | 回答的问题 | 安排 |
|---|---|---|---|
| **A0** | E0 + 3-layer StarEGNNV3 + 现有 Direct Head | 匹配基线 | 必须 |
| **A1** | E0 + 2-layer GeoEGNN | 轻量连续距离消息传递是否足够 | 高优先级 |
| **A2** | E0 + 单帧 PaiNN-lite | 标量—向量跨层传播是否改善方向 | 高优先级 |
| **A3** | E0 + EGNN + Reaction Query | 反应中心与全局意图聚合是否有效 | 高优先级 |
| **A4** | E0 + EGNN + Geometry-aware Graph Transformer | 远距离 Attention 是否提供额外收益 | 中优先级 |
| **A5** | E0 + GraphGRU | 同图迭代循环更新是否有效 | 条件实验 |

约束：

- A2 必须新建当前 key-atom 单帧实现，不直接复用旧 dual-ts PaiNN。
- A3 的 Query 只能由当前状态和合法反应图产生。
- A4 必须使用合法几何特征和对称性设计，不能搬入普通文本 Transformer。
- A5 的循环只发生在当前图的消息更新中，每次推理重新初始化，不保存跨帧状态。
- 不把普通 CNN 强行加入 XYZ 回归矩阵；径向直方图 CNN 会同时改变输入表示，需另立表示学习实验。

**归因限制：**A1 同时改变了层数和 EGNN 算子。若 A1 晋级，应激活一个预注册的深度/算子对照；在对照完成前，只能声称“轻量 GeoEGNN 配方有效”，不能单独归因于连续距离或层数。

## 4.2 前置工程门

正式候选训练前必须完成：

1. 定义统一的 `BackboneOutput + MOVE Decoder + PROCESS Head` 接口。
2. 用新接口重建 A0，并完成 strict-load、前向 parity 和训练可比性审计。
3. 保持 E0、Direct Head、损失、采样器和数据视图不变。
4. 对所有候选通过：
   - 输入白名单与无泄漏检查；
   - E(3) 等变性；
   - 原子置换一致性；
   - reverse/relabel 检查；
   - 非关键原子零动作；
   - restore/resume 和 typed-invalid 测试。

这一接口工作同时是 V5.4.3 的前置条件，避免把“更换 Backbone”和“更换 readout”混在一起。

## 4.3 晋级式训练顺序

1. **WP0：预注册与预算签认**
   - 冻结候选规格、参数/容量记录方式、诊断剂量、评价指标和停止规则。
2. **200-update smoke**
   - 仅检查学习启动、梯度、资源、恢复和合法性，不参与科学排名。
3. **同环境 profile**
   - 测量 core-time、wall-time、显存、host RSS 和前向成本。
4. **seed 17 小剂量诊断**
   - 只淘汰不学习、破硬门或明显超预算的候选，不据此宣布冠军。
5. **seed 17 全剂量筛选**
   - 合格候选运行 400 warmup + 37,500 joint updates。
6. **前两名补 seed 29/43**
   - 使用 inner-val 配对比较和 curve-level bootstrap。
7. **冻结 Backbone**
   - 无候选可靠超过 A0 时，A0 留任并保留负结果。

现有 E0×3 只有在训练代码、初始化、sampler、schedule、评估器和 checkpoint 身份均通过可比性审计后才能复用；否则重训 matched A0×3。

## 4.4 晋级门

建议继承并重新签认：

- **H0 合法性**：全部合同和对称性检查通过。
- **H1 资源**：单路、无 swap/OOM，符合冻结预算。
- **H2 MOVE 非劣**：
  - RMSE/zero 相对 R3 参照及 matched A0 退化不超过 2%；
  - sign 绝对下降不超过 0.01；
  - dead fraction 绝对增加不超过 0.01；
  - 固定覆盖率方向指标相对退化不超过 2%。
- **H3 任务收益**：
  - PROCESS 路线可沿用 ΔAUPRC ≥ +0.02 或 Δq corr ≥ +0.03；
  - Backbone 也应允许通过 MOVE/方向改善晋级，但其最小效应量和统计规则必须在 WP0 另行冻结；
  - 至少 2/3 seeds 同向，其他主指标非劣。
- **成本门**：建议继续以 `core_time_ratio ≤ 2.0` 作为预注册候选值，签认后不得事后调整。

任何 PROCESS 或 rollout 收益都不能豁免 H2。

---

# 5. V5.4.3：动作 Decoder

固定 V5.4.2 选出的 Backbone 和 E0，不再同时修改 Embedding、Encoder、损失权重或采样器。

| 编号 | 方法 | 目的 |
|---|---|---|
| **D0** | 现有 signed-vector Direct Head | 匹配基线 |
| **D1** | Tangent-aware / CV-aware Head | 改善沿反应路径的方向一致性 |
| **D2** | Conditional Flow Matching | 研究更连续的 conditional displacement field |

### D1 边界

真实 IRC 切线只能作为**训练监督**。推理时只能根据当前构型预测方向；CV 也必须能由当前几何和合法拓扑计算，不能输入真实路径进度。

### D2 启动条件

先完成 D0/D1。只有结果表明动作表达仍有明确改进空间，且预算允许，才另立 D2 预注册。D2 必须事先定义：

- 条件变量和插值/时间变量；
- 训练目标及等变性；
- 采样或 ODE 求解方式；
- 推理 NFE、步数和确定性规则；
- 单次动作输出方式及额外前向成本。

D2 可以内部多次前向，但对外仍是**单帧输入、一个动作输出**；不得使用未来几何、真实 q 或完整 IRC。Flow Matching 不是默认必做项，也不被视为碰撞问题的当然解法。

---

# 6. V5.4.4：完整 IRC 离线教师

这是新增独立子战役，不是 V5.4.1 的延续训练。

## 6.1 统一 Student

- 固定为 V5.4.2/V5.4.3 选出的 Backbone + Decoder。
- 所有教师使用相同 Student 架构、训练剂量、seed 和基础损失。
- 除蒸馏目标外，不同时改变其他训练条件。

## 6.2 教师顺序

| 编号 | 教师 | 作用 |
|---|---|---|
| **T0** | 直接 IRC 标签；必要时增加非学习式平滑/切线对照 | 区分蒸馏收益与普通标签正则化 |
| **T1** | 1D CNN / TCN | 提取相邻 IRC 片段的局部变化和平滑模式 |
| **T2** | 双向 GRU | 提取完整路径的前后顺序和累积关系 |
| **T3** | Path Transformer | 提取远距离帧关联，仅在 T1/T2 显示 Student 收益后启动 |

### 数据边界

- Teacher 只能在授权的 **train 曲线**上学习和生成蒸馏目标。
- inner-val 只用于选择和评价，不用于构建 Student 蒸馏标签。
- dev/confirm 不得用于教师训练、目标生成或隐藏状态提取。
- Student 推理不得读取 Teacher、完整路径、序列位置或历史隐藏状态。

### 主评价

Teacher 自身的 IRC 拟合误差只作诊断。晋级取决于 Student 的：

1. MOVE/PROCESS 单步指标；
2. 方向、dead 和端点预测；
3. autonomous rollout 稳定性；
4. 相对额外训练成本是否值得。

若 T1/T2 都没有带来可靠 Student 收益，T3 记为条件跳过，保留负结果。

---

# 7. 统一评价与计算治理

## 7.1 数据使用

| 数据 | 用途 |
|---|---|
| 50K 冻结训练池 | 所有正式训练及 Teacher 目标生成 |
| inner-val | 小剂量诊断、候选排序、超参和 τ 校准、晋级 |
| 589 dev | 只对最终冻结的少量模型做确认性评估 |
| 2K confirm | 继续封存 |

V5.4.2–V5.4.4 不应反复使用 dev 排名，否则 dev 会变成架构调参集。已有 E0 dev 数值只作基线描述。

## 7.2 两种评价必须分开

**Teacher-forced 单步评价：**

- MOVE RMSE/zero、sign、dead、方向；
- Endpoint AUPRC/AUROC、q、ordering；
- 不同路径区段和反应簇分层结果。

**Autonomous 多步评价：**

- 起点、pre-TS、post-TS、端点邻域出发；
- 双向推进和跨 TS candidate；
- endpoint arrival；
- 碰撞、非有限值、非法输出；
- 错误 No-Move、预算耗尽；
- 非关键原子位移；
- 模型前向次数、NFE、wall/GPU 成本。

`reference_normalized` 与 `autonomous` 分开报告。No-physics rollout 是重要压力测试，但其失败不能全部归因于网络，因为非关键原子冻结、缺少物理松弛和 off-IRC 分布偏移同样可能导致失败。

## 7.3 两种成本口径

同时报告：

1. **Fixed-dose**：相同 37,900 updates 下的科学效果；
2. **Fixed-GPU-time**：相同计算预算下的效果和部署经济性。

不得只凭 AUPRC、q correlation 或 Teacher 拟合误差选冠军。

---

# 8. V5.4 收尾与停止规则

最终至少保留以下可比链条：

\[
\mathrm{A0/D0}
\rightarrow
\mathrm{Backbone^*}
\rightarrow
\mathrm{Decoder^*}
\rightarrow
\mathrm{Student^*}
\]

这样才能分别归因 Backbone、Decoder 和蒸馏的增量收益。

### 停止规则

- 无 Backbone 合格胜出：A0 留任，但可继续研究 Decoder。
- D0/D1 已足够或剩余问题不属于动作表达：不启动 D2。
- 轻量 Teacher 无 Student 收益：不启动 Path Transformer。
- 只有单步拟合改善、闭环无改善：不得称为策略成功。
- 所有纯几何方案仍普遍碰撞或无法稳定推进：保留负结果，停止扩张 V5 网络，转入 V6 物理反馈研究。
- 不得为改善 rollout 而临时引入 xTB、历史状态或真实路径信息。

---

## 9. 下一步只做 V5.4.2 正式预注册

下一项工作应限定为：

1. 新增架构职责与实验治理 ADR；
2. 新建 V5.4.2 详细方案、campaign config、schema 和 registry 条目；
3. 冻结 A0–A5 的规格、条件激活规则和预算上限；
4. 先实现统一 Backbone/Decoder 接口及 A0 parity；
5. 完成签认后，才授权 200-update smoke 和后续训练。

**本轮不启动 CNN/RNN/Transformer 全矩阵，不修改 V5.4.1 冻结文件，也不改变现有 V5.4 proposed 条目。**
