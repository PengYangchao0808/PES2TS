# PES2TS：30 小时 R3 → V5.2 → V5.3 分层训练与验证战役 v1.1

> **EXECUTED-FROZEN / Stop 语义已取代（2026-10-21）**：本计划的 §5.3/§5.6/T3.2 把 Stop 定义为 TS 邻域（d_key_TS 0.04/0.12 Å）——该定义已由 [ADR-0030](../design/decisions/ADR-0030-V5.2全路径导航与端点停止.md) 取代为 Endpoint Stop（目标端点到达），并进一步由 [ADR-0031](../design/decisions/ADR-0031-MOVE-PROCESS统一学习合同.md) / [MoveProcessContract v1](../contracts/PES2TS_MoveProcessContract_v1.md) 取代为 MOVE + PROCESS。本文件保留为已执行战役的历史依据；旧 V5.3 rollout 记为 **V5.3-ENG**，新战役为 **V5.3-MP**（[计划](PES2TS_V5.3-MP_大规模MOVE-PROCESS联合训练计划_v1_20261021.md)）。R3 Move 语义仍有效。

> 日期：2026-10-20 · 状态：**待冻结的执行计划**，非已运行结果  
> 适用前提：**一条 canonical IRC、正反两种 Reaction Intent、整条曲线有向单帧监督**；R3 新标签与 R1 旧实验不直接可比。  
> 版本关系：R3 是 `intent_directed_irc_step` 位移基线；V5.2 完成 Progress/Stop；V5.3 是有约束 rollout 的技术试运行。不得从本战役自动升级 V5.1b、正式 OOF、sealed/Confirm/OOD 或 Fit150+。  
> 当前合同：[ReactionIntent Contract v1](../contracts/PES2TS_ReactionIntent_Contract_v1.md)；当前阶段计划：[V5–V7 v3.1](PES2TS_V5-V7_反应意图条件化与低维生长方案_v3.0_20261020.md)；当前 R3：[V5.1-R3 v3.1](PES2TS_V5.1-R3_反应意图条件化与端点标签不变重训方案_v3_20261020.md)。  
> 预算修订：R3 从 63 调整为 **66 runs**（新增 bidirectional 等预算臂）；R3 66 + V5.2 33 = **99 runs**。  
> 资料边界：本计划已合并用户 30h 草案，并统一到 ReactionIntent v3.1；旧 v2.0/GeometryPolicyContract 仅作历史引用。

## 0. 30 小时必须交付什么

**最低成功交付（必须尝试完成）：**
1. R3 的曲线级数据库、双向 intent、合法特征与教师标签完成审计；**66-run 为上限**，阶段 gate 失败应提前止损。
2. 经冻结的 R3 位移基线：相同输入 `X_t,G_rxn`，仅输出关键原子 `[K,3]` 位移；可选 strong/dead 通过校准而非凭 λ 猜测。
3. V5.2 同一 EGNN 背骨上的 Progress Head 与 TS Stop Head；训练出可复现实验以及发展/确认折的结果。
4. V5.3 至少完成 **无松弛 5/10/20 步 rollout** 和受约束执行器的少量端到端试运行（如果物理执行器已有可靠实现）；报告与扫描基线对比，不冒称验证过的 IRC/TS。
5. 全部记录协议、训练 checkpoint、split 与数据 hash、失败样本和成本。若 Data/Feature 门被阻断，以可信诊断结果为本次成功交付，不强行跳到 V5.2。

**不得保证：**30 小时内获得跨反应泛化可靠的 Stop Head、真正通过 TS 的自治生长、verified TS 或发表级方法学结论。约 49 个独立有效反应簇，证据仍属小样本开发验证。

## 1. 两套概念先固定

- **物理对象** `CanonicalIRC`：一条完整 A—TS—B 曲线，只存一次，具有节点、映射、TS、端点、质量/能量血缘。它不是两个独立 branch。
- **有向任务对象** `ReactionIntent`：`G_src,G_tgt, ΔB_intent, K, CV_spec, rules`；正向 A→B，反向 B→A。只交换显示名称而不交换物理任务时，特征严格不变；真正反转任务时，signed intent 变化。若端点键图完全相同，必须有可合法获取的 CV/构象控制目标，否则标记 `intent_underdetermined`。
- **训练样本**：`(curve_id,node_index,intent_id,d0)`，每个当前几何单帧配对指定方向的未来目标；这里的“未来”只能作为教师标签，不得进 `forward`。
- **内部进度** q 是由 canonical IRC 标签生成的目标预测量，**绝不是在线输入**；TS 候选 Stop 与真正 verified TS 不等价。

### 1.1 固定核心推理合同

```text
Input:
  xyz_current             float32 [N,3]   # 当前完整分子，唯一的三维构型
  atomic_numbers          int64   [N]
  charge, multiplicity    scalar
  graph_current / graph_union               # 当前/无序合法拓扑
  reaction_intent         G_src,G_tgt, signed edits, CV_spec
  key_index               int64   [K]     # 由合法反应事件确定
Output (policy):
  delta_xyz_key           float32 [K,3]   # 仅关键原子
  q_pred                  float32 []      # V5.2 起的辅助预测
  stop_logit              float32 []      # V5.2 起的辅助预测
Output (product controller):
  move_flag               bool            # Move/No-Move
  selected_coords         current xyz at No-Move
  event / typed_reason    diagnostics only
```

V5 绝不把历史帧、真实 TS 坐标、真实 progress、上一轮隐藏状态、未来帧、能量梯度作为 policy 输入。非关键原子参与编码但模型不输出其动作。V5.3 受约束执行器可调用物理软件；它的返回数据只影响本次外部执行/诊断，不馈入 V5 policy；V6 才探索物理响应修正 head。

**Stop 的语义：** `stop_logit` 预测“当前位置适宜提交 TS 初猜”的信号。在 TS-only 模式中超过冻结阈值则 `Move=False`、选中当前几何；在 Full-Curve 模式中 controller 可以记录 TS 事件并按有向 intent 继续，但不能把“预测已过 TS”直接当作真实验证。

### 1.2 固定 R3 教师

- 将 IRC 全路径规范化排为 `A → TS → B`，反向任务读取逆序，不新增物理曲线。
- 主 `d0=0.08 Å`，仅一个训练主任务；其他 d0 是独立敏感性、不计入主训练样本统计。
- 从当前节点沿 intent 指定方向使用 **first-contact interpolation** 寻找目标，允许跨过 TS；非必然等于下一离散帧。
- 教师坐标必须在同一可比较的局部参考系，关键原子目标位移与当前坐标的 gauge 一致；不可解时 mask，绝不静默用实验室固定轴。
- 原始 `irc_edge.delta_xyz` 与派生的 `teacher_action_key` 分开保存及哈希。
- 到达曲线末端没有未来 contact 则标注 terminal/mask；近 TS 的位移标签仍可非零，Stop Head 与 Move Head 是不同任务。
- 每条 curve 的正/反样本权重和为固定目标，不因为双向增强而把它当成 2 个独立化学反应簇。

## 2. 30h 机器时间表

| 墙钟 | 阶段 | 最终应看到的文件/现象 | 截止门 |
|---|---|---|---|
| H00–H02（2h） | E0 冻结/测速 | git commit/tag、`protocol_30h.json`、基准速度测量、环境清单 | 合同签认 |
| H02–H05（3h） | E1 Curve/Intent/Data | 新 6 张数据 CSV+分片、教师与质量报告、split 泄漏报告 | Data Gate |
| H05–H07（2h） | E2 Feature/Loss Tests | F3 固定槽位、Relabel/Reverse/等变/λ=0 测试 | Feature/Loss Gate |
| H07–H12（5h） | E3 R3 Training | F/B/L/V/R/C 实验，最多 66 runs、B0-F 与校准报告 | R3 dev & confirm |
| H12–H15（3h） | E4 V5.2 Labels/Heads | 有符号 q、Move/Stop/Mask 标签、正例覆盖率与 head fixture | V5.2 label gate |
| H15–H20（5h） | E5 V5.2 Training | q-only、stop-only、joint，开发折及条件确认折；checkpoint | V5.2 supervision gate |
| H20–H26（6h） | E6 V5.3 Rollout | 5/10/20 步无松弛；受限松弛小样本，经典扫描对照 | Engineering gate |
| H26–H28（2h） | E7 统计复核 | cluster 置信区间、错误簇卡片、成本 ledger、门报告 | Report Gate |
| H28–H30（2h） | E8 缓冲/归档 | 失败补测、哈希验收、最终 decision memo | **硬停止 H30** |

H00–H02 可以先以旧环境做 100–300 updates 的硬件/吞吐 smoke；R3 数据及模型通过 E2 后，再以 3 个正式 3000-update F 阶段 run 测量新环境平均耗时 `t_run`，重算余下阶段预算。旧 4060 Ti 配置的 ~2.5 min/run 只作参考。R3 66 + V5.2 最多 33 新 runs，即 99 run：`2.5 min` 约 4.1h，`3.0 min` 约 5.0h，`4 min` 约 6.6h；尚未包含数据、评测和 rollout。各阶段设置独立截止时间，不能用继续盲目跑屏幕实验挤占 V5.2 数据构造。

**硬停规则：** H05 Data Gate 不过，停止训练；H07 Feature Gate 不过，停止训练；H12 R3 缺可信基线，V5.2 只能做标签/头的合约测试，不宣布验证结果；H20 V5.2 未通过，则 V5.3 仅测试经典/教师上限控制器；H30 无论结果如何停止并生成报告。

### 2.1 任务依赖 DAG

```text
E0 freeze/protocol
  -> E1 curve/intent/teacher
  -> G0 Data Gate
  -> E2 feature/loss tests
  -> G1 Math/Feature Gate
  -> E3 R3 training
  -> G2 Move Gate
  -> E4 q/stop labels
  -> G3a label/positive-coverage Gate
  -> E5 V5.2 Stage A/B (Q, T, QT)
  -> G3 Progress/Stop Gate
  -> E6 V5.3 rollout (no-physics -> constrained RELAX)
  -> G4 Rollout Engineering Gate
  -> E7 statistics/cost/report
  -> E8 final decision memo
```

**阻断规则：**

- G0 失败 → 不得启动 E2/E3；
- G1 失败 → 不得启动 R3；
- G2 失败 → V5.2 只能做合约/标签测试，不能宣称验证；
- G3a 正例不足 → Stop 标记 `insufficient_stop_positives`，只报告探索性；
- G3 失败 → 不解锁 V5.3 候选；
- G4 只代表工程闭环，不代表 verified TS。

## 3. E0–E2：先构建可验证的数据与模型接口

### 3.1 数据表

`irc_curve.csv`、`irc_node.csv`、`irc_edge.csv`、`reaction_intent.csv`、`curve_quality.csv`、`sample_view.csv`（共六张，不是五张），另存 `split_manifest_r3.json`、`data_manifest.json`、`teacher_manifest.json`。

- `curve_id` 代表完整物理 IRC；`d0` 不进入曲线身份，而进入教师协议与 sample 主键。
- 原始 TS 哈希只溯源，去重还参考映射、相似结构及 IRC 轨迹。
- 每个 `curve_id` 两个有效相反的 intent，具体数量在报告中由可辨识性决定；不强行把退化/同构无方向的图编辑当成有效反向任务。
- `G_src/G_tgt/ΔB_intent` 从合法反应任务获得，不从真实 IRC 所处区段恢复；原子 `intent_role` 使用可同时 forming/breaking 的多通道表示。
- 没有任何 branch ID 进入网络、训练目标标识和 split 主键；允许保留历史 segment/branch quality 供溯源。
- split 按 near-duplicate 连通分量，确保同曲线/多 intent/多 d0 同折。若旧 split 已确认泄漏，冻结修订 split 并重新计算所有 R3 基线，不沿用旧确认结果。

### 3.2 Data Gate（需要全部通过）

1. 曲线、节点、边、样本的 FK 完整；TS 唯一；TS 两侧节点顺序合法；无非有限坐标。
2. 同 curve 的正/反转换做 **label involution**：`reverse(reverse(intent)) == intent`，反向单步教师与逆序路径相符；标签在相同参考架中计算。
3. `d0=0.08` 所有主标签；2022 按 0.08 复核；1248 gauge、5 mapping overflow、342 冲突与 fit50 交集有明确处置，并保留训练覆盖率及原始分母。
4. 对相同 `(X,G_intent,d0)` 的近似输入重新执行冲突审计；若仍多解，说明由拓扑/CV 不足或映射/几何数值问题造成，必须 mask、降权或阻断。
5. 无确认的跨折同源/near-dup 泄漏；所有 `P0` 有审计结论，不允许隐瞒。

### 3.3 Feature Gate

固定 F3 superset 输入形状，用 mask 控制 F0/F1/F2/F3，禁止不同特征臂偷偷增加网络参数。F2 包括当前 pos、元素、K、degK、无序端点上下文及**目标方向明确的 `B_src/B_tgt/ΔB_intent`**；F3 额外包含当前距离 RBF(12)。Intent-off 必须同时 mask 所有有序端点、signed role、directional CV 描述；保留无序拓扑，才是有意义的消融。

测试项目：
- 同一化学任务只重新命名端点：特征逐比特一致，eval mode 输出数值一致（期望 max absolute Δu ≤1e-6 Å，无法满足需给出 fp32 浮点误差证据）。
- 真正 Reverse：签名反转、label 对应、角色变化；**训练前只校验语义，不要求随机初始化网络输出必须不同**。
- 平移不变、旋转 SO(3) 等变、原子重编号等变，FP32 误差容差建议 1e-5 相对 +1e-6 绝对；记录 median/P99/max。
- 两次 deterministic shard build 内容哈希相同；不含教师 TS、未来帧、progress、branch、split、family、conflict、energy/gradient、near_ts/capped。
- 合法 K 由成断键/明确 CV 涉及原子得到，非法映射有 typed error；非关键原子不输出动作。

## 4. E3：R3 训练——先证明位移能学，再调强弱损失

### 4.1 网络和固定训练超参数

基于现有 StarEGNN（旧版 hidden 48、3 layers，59k 左右参数），新版允许固定 F3 superset 输入槽位与少量头扩展。建议用训练协议锁定：AdamW；`lr=1e-3`、`weight_decay=1e-4`、warmup 250、cosine；3000 updates；batch32=8 curves×4 frames；每 curve 内平衡正反 intent；梯度裁剪1；fp32；seeds `[17,29,43]`。Inner-train/inner-val 划分只从 outer-train 簇中做；选 checkpoint 只能看 inner-val。不可用 fold3/4 指标回头调整模型。

固定核心基线：

\[
L_{\rm base}=L_{\rm atom}+0.5L_{\rm pair}+\lambda_{\cos}(t)L_{\cos},\quad \lambda_{\cos}^{\rm max}=0.02.
\]

\[
L_{{\rm atom},b}=\sum_{i\in K_b}w_{b,i}\operatorname{Huber}_{0.25}\left(\frac{\|\hat u_{b,i}-u^*_{b,i}\|}{0.08\,\mathrm{\AA}}\right),\quad
w_{b,i}=\frac{1/\max(1,\deg K_i)}{\sum_{j\in K_b}1/\max(1,\deg K_j)}.
\]

`L_pair` 使用旧 `legacy_j`（当前构型定义 pair 轴的相对位移投影 Huber），`L_cos` 只对 `||u*||>0.02 Å` 原子生效；其 ramp 默认为 R0：250 更新后开始、250 更新渐至0.02，**明确将 R0 设成正式主日程**，另两种 ramp 只是诊断。采样已 curve-balanced 时，不再叠乘等效 inverse-curve 权重；只在 loss 层施加 label-quality 权重。

strong：变键 `|Δd*|>=0.02 Å`，每 frame 桶均值然后加权 frame 均值；
\[
\ell_s=\operatorname{Huber}_{0.25}\left(\frac{\widehat{\Delta d}-\Delta d^*}{\max(|\Delta d^*|,0.01\,\mathrm{\AA})}\right).
\]
在 strong 桶中 `|Δd*|>=0.02 Å`，因此 0.01 Å 下限实际不生效，不额外做无意义的 tau_s 敏感性。dead：变键 `|Δd*|<0.005 Å`，
\[
\ell_d=\left[\frac{\max(0,|\widehat{\Delta d}|-0.005\,\mathrm{\AA})}{0.005\,\mathrm{\AA}}\right]^2.
\]

只训练 `S2/S10/D2/D10` 四种单独辅助臂，先不混合。`λ_s`/`λ_d` 用 **新 R3 B0-F checkpoint** 在20个真实 inner-train batch 上校准 encoder 对 `L_base` 的梯度占比2%、10%，λ≤0.5，head≤0.3；不足1%为该臂 `scale_unreachable`。训练期间每100 updates 只监控，不随意在线调λ。λ=0 要求回归到同一新版 B0-F 路径逐比特一致。

### 4.2 R3 最多66-run（各阶段有 stop rule）

| 子阶段 | 实验 | 新 runs |
|---|---|---:|
| F-main | F0/F1/F2/F3，forward-only，fold0×3 | 12 |
| F-bidir | F2，A→B + B→A，等曲线预算，fold0×3 | 3 |
| F-ablate | F2，真正 intent-off（mask signed intent/role/CV 方向），fold0×3 | 3 |
| B | 获胜特征 baseline 补 folds1–4×3；fold0 复用 | 12 |
| L | S2/S10/D2/D10，fold0×3 | 12 |
| V | Top-2 folds1–2×3 | 12 |
| R | R1/R2 ramp诊断，fold0×3，R0复用 | 6 |
| C | dev-positive Top-1，folds3–4×3 | 6 |
| **合计** | | **66** |

F 阶段的 F0/F1/F2-intent-off 属于科学负对照，**能够晋级到 V5.2 的主表示只能从 F2-intent-on 与 F3（包含完整有向 intent）中选取**，否则同一几何下反向 q 无法区分；若它们的 Move 表现明显劣于无 intent 对照，必须先报告矛盾而不是偷偷采用 F0 后声称能预测有向 q。F 阶段 fold0 只是筛选，不能单折宣布泛化。B 阶段 folds3/4 的评估指标封存，不看结果选型；必要时将其 baseline 指标在 C 阶段一次性解封配对。Top-2 的 RMSE/SRE、sign/dead 容差冻结使用最近 R3 协议值；确认折只能一次性检验不能回调。若 R3 数据/标签相较当前用户方案重新变更，就必须重新冻结 R3 protocol 与 run manifest，不得继续累计为同一 66-run。

**双向增强纪律**：F-bidir 与 F-main 使用同一批 curve、同一 fold、同一曲线级采样权重、同一优化步数与模型容量；反向样本不是独立 curve。若 bidirectional 仅在“更多数据/更多更新”时更好，不算双向监督收益。主实验仍是 forward-only；bidirectional 是增强臂。

### 4.3 R3 指标和门

- **主指标**：`key_RMSE/zero`（越低越好）和冻结版本的 `SRE`（越低越好），以 **独立 curve/cluster** 聚合，而不是把几千帧当几千个独立化学反应。
- **防退化**：方向 cosine、关键变键 Δd 的 sign accuracy、strong 幅度比、dead 假变化率；按预注册规则 `RMSE/zero` 不得较匹配 B0 劣化>2%、sign下降>0.01、dead-zone上升>0.01（这些是开发保护阈值，确认使用更严格预注册门）。
- **机制证据**：`ρ_encoder` 在目标 ±30% 内，`ρ_head<=0.3`，bucket 的 active frames/curves 数和覆盖率完整。
- **基线**：zero step、可执行的经典 CV/Jacobian step、同特征 minimal MLP（若有）和新版 EGNN。**旧 OOF=0.824 不能混到新任务表里比较。**
- **决策**：Data/Feature 失败先停止；R3 终局仍采用 `data_blocked,feature_blocked,scale_unreachable,no_gain,scale_controlled`；`scale_controlled` 需要预注册 confirm 通过；不开放旧 V5.1b。

## 5. E4–E5：V5.2 的完整 Head 设计与训练

### 5.1 强制保留相同在线输入

模型共享 R3 的 EGNN 编码器，最小变化是增加两个辅助 head：

```text
(X_current, G_reaction) → Encoder → H_node,H_graph
                                  ├→ MoveHead  : u_K [K,3]
                                  ├→ Progress  : q in [-1,1]
                                  └→ TSStop    : stop_logit scalar
```

实现可采用 `z_graph[64] → Linear(64,32)→SiLU→Linear(32,1)`，progress 输出 `tanh`，Stop 输出 logit（使用 `BCEWithLogitsLoss`，不要训练时预先 sigmoid 两次）。head 输入仍来自同一个单帧，`q` 不能反向作为模型的历史状态输入。在线对外按 **ReactionIntent Contract v1** 解释为 `delta_X_K, Move/No-Move`；`q_pred` 和 `stop_score` 是辅助诊断，可供外部控制器控制步长。

### 5.2 离线 q 标签：明确数学定义

对每一条已规范定向的曲线，先用相邻帧对齐后的**质量加权弧长**积累 `s_i`，TS 位于 `s_TS`，两端 `s_A,s_B`。令

\[
q_i=\begin{cases}
-\frac{s_{TS}-s_i}{s_{TS}-s_A},& i\le i_{TS},\\
\frac{s_i-s_{TS}}{s_B-s_{TS}},& i>i_{TS}.
\end{cases}
\]

A=-1、TS=0、B=+1；同结构、反向 intent 则 `q_reverse=-q_forward`。若某一段弧长几乎为零，mask 而不是除以零；质量加权对齐的具体原子集合、质量单位和曲线插值规则必须冻结。对于任务内预测 q，它是依赖有向 intent 的几何进度估计，不是真实的未来已知路径长度。

**不要在 V5.2 同时开启未知的 progress 定义、多个 normalization 与多尺度教师**。主指标用上述 q，其他版本另做后续研究。

### 5.3 Stop 三态离线标签，二态在线决策

`stop_teacher∈{Move,NoMoveCandidate,Mask}`。由训练曲线的可信 TS 参考，采用合法刚体对齐后当前 K 几何与 TS 的差距、曲线弧长邻域和 TS 质量标记联合生成。先预注册候选阈值，比如 `d_key_TS<=0.04 Å` 作为正例几何要求、`d_key_TS>=0.12 Å` 作为明确负例几何要求，中间 Mask；同时增加训练集内可观察到的局部质量/参考 TS 验证条件，以免只用弧长或随机近邻当正例。这些数值**是待训练集内预检并冻结的工程建议，非已证实的化学最佳窗口**。若 0.04 Å 使每折只有极少正例，则不要偷偷扩窗；记录`insufficient_stop_positives`，使用训练内预注册的备选窗口，或者 Stop 只报告探索性结果。

在线根据开发折拟合、在确认折之前冻结的 `θ_stop`：`p_stop=sigmoid(logit)`，`p_stop≥θ_stop` 则 Mode=TS-only 时返回 `Move=False` 和当前坐标；Full-Curve 由控制器记录 TS 候选事件并独立决定是否继续。端点停止与预算失败属于 Controller，不作为 TSStop 的正例；Mask 不计入硬二分类指标，但进入数据覆盖率报告。

### 5.4 固定 V5.2 loss（从 R3 基线扩展）

\[
\boxed{L_{5.2}=L_{\mathrm{move}}+\lambda_qL_q+\lambda_mL_{\mathrm{TSStop}}.}
\]

`L_move` 等于在 R3 中事先冻结、并通过门的完整位移损失（包含的 S/D 若有也被冻结）。

- `L_q = weighted_mean(Huber_{0.1}(q_pred-q_true))`，仅使用 quality 有效的进度标签；curve-balanced 聚合。
- `L_TSStop = weighted_mean(BCEWithLogits(stop_logit,y_stop))`，Mask 不参与分母；Stop 正负样本权重只在 inner-train 估计并冻结，禁止从 dev/confirm 统计类别比例后回灌训练。
- `λ_q,λ_m` 不做大网格；先测 shared encoder 中相对 Move loss 的梯度占比，优先各 2–10% 作为**探索性范围**；大幅破坏 Move 就降低联合共享程度，或保留冻结编码器。head 自身必须有可学梯度，不套用 R3 的辅助 action-head≤0.3 门；对 action head 强制检测来自 q/stop 的非预期梯度路径。
- 本战役 `λ_mono=0`、`λ_path=0`，先把监督任务研究清楚。单调性、正反一致性只作诊断，后续另立消融。
- 为防止 head 随机初始化破坏动作网络，先 **冻结 R3 编码器，训练新增 head 400 updates**，随后在合格臂 **联合微调 1500 updates**，`lr_head=3e-4`、`lr_encoder=1e-4` 为建议初值，在协议冻结前用 1 次 smoke 检查学习稳定性；冻结后不得因确认结果改变。优化器 AdamW，grad clip1，fp32。

### 5.5 V5.2 训练矩阵（最多33个新 runs）

R3 的 Match Baseline 已存在且哈希一致时复用，无需再次训练：

| 臂 | 训练内容 | 开发折 folds0–2 | 确认折 folds3–4 |
|---|---|---:|---:|
| M0 | Move-only R3 冻结基线 | 复用 R3 | 复用 R3 |
| Q | Move+Progress | 3fold×3seed=9 | 只作诊断，不必须重跑 |
| T | Move+TSStop | 9 | 只作诊断，不必须重跑 |
| QT | Move+Progress+TSStop | 9 | 若获预注册 dev-positive 则 2fold×3seed=6 |
| **最多新 runs** | | **27** | **6** |

Q/T/QT 使用完全相同的曲线数据和训练预算，仅通过 head loss 开关对照；每个 run 保存 `heads_only.pt` 与 `joint_best.pt`，冻结/微调两个阶段可以诊断多任务学习是否导致动作遗忘。若 Q/T 分别有用而 QT 没有，则不能宣称“三头协同”；需要记录多任务梯度冲突，仍允许保存表现最可靠的单头研究结果。确认资格要求 QT 在开发折的 Move 保护、q、stop 各项达标，且冻结所有阈值。

### 5.6 V5.2 模型测试（不止 loss）

**必须报告：**
1. `Move`: 每曲线 RMSE/zero、cos、SRE、strong/dead/sign；相对匹配 R3 checkpoint 是否退化。
2. `Progress`: MAE(q)、按 curve Spearman、TS 两侧符号正确率（排除 |q| 近0灰区）、反向 `q(X,Reverse(G))≈-q(X,G)` 的误差。`q=0`、简单 CV 代理是匹配基线。
3. `Stop`: PR-AUC（同时报告阳性率作为随机基线）、precision/recall、F1、误报率，固定 threshold；**沿每条曲线第一个 Stop 触发位置**与 reference TS 的距离，且记录“从未停止/过早停止/重复停止”。
4. `Generalization`: 只按反应簇 bootstrap（建议 2000 resamples），报告 3 seed 中位与方差、家族/规模分类；同一 curve 正反方向仍只有 1 个统计单位。
5. `Negative controls`: 将教师的 q 在不同曲线之间打乱应显著降低 q 性能；将有向 intent 关闭应破坏有向进度/动作；保证没有无意泄漏。
6. `Regression tests`: Stop 不应迫使 TS 附近的 MoveHead 本身输出0。R3 全曲线教师在 TS 可有非零下一步，网络能同时提出动作并报告 TS 事件。

**建议的开发准入（阈值须预先从训练内校准，不能在确认后修订）**：Move 相对 R3 RMSE/zero 恶化≤2%，sign/dead 不越过 R3 容差；q MAE 优于 q=0 及固定 CV 代理、其方向符号明显优于随机基线；Stop PR-AUC 大于其阳性率且沿曲线 first-stop 位置优于固定阈值代理，覆盖率与可用正例充足；≥2/3 seeds 多开发折稳定。是否满足统计确定性仍受约49独立簇限制，应标明 exploratory。

## 6. E6：V5.3 闭环，不把近 TS 等同于真实 TS

### 6.1 先跑无物理的 autoregressive rollout

每个 episode 从实际 IRC 的第一个端点或独立留出的中间帧开始，选择固定的 Reaction Intent，只给当前 `X,G` 调用模型。依次 `Predict u` → `trust_region_clip` → `geometry/valence/safety` → `update X`，执行 5/10/20 步；这里非关键原子不由网络移动，几何积累可能失真，故仅是 **策略稳定性压力测试**，不代表真实受控反应路径。

- 汇报几何合法率、键长方向完成率、步长塌缩、碰撞、弧长/进度震荡、重新进入原位置、偏离参考轨迹的趋势。
- 每一帧使用新模型单点前向，不读取过去状态作为网络输入；controller 可以记历史做防环和回滚。
- Stop 触发时在 TS-only 中保存当前构型；full-curve 记录 TS 候选事件但不是终止整个 episode。

### 6.2 再跑有约束 RELAX 的小样本

只在 Development 区间挑选预注册的 **8–12 个代表性曲线×固定初始端点**（小体系优先、含两个 hard clusters）；一张卡片同时跑以下三臂：

- Classic：固定 CV 增量/Jacobian 或简单键扫描；
- R3：EGNN Move + 同一约束修正器；
- V5.2：EGNN Move + q/stop 调节步长与候选提交。

同一反应、同一初始结构、同样总量化预算、同一受控 Relax 设置。若暂时没有稳定物理后端，则使用已实现的 dry-run/mock 做集成测试，并把真正的物理对照标记 `not_run`，不可伪装为通过。

**重要**：不能用普通全自由度几何最小化代替沿 CV 的受约束松弛；否则可能退回同一势阱。物理执行器应明确“约束哪些关键 CV、允许哪些非关键原子松弛、成功/拒绝如何判断”。使用 xTB 做执行器不代表 V5 Policy 获得 xTB 输入；V6 是**网络修正**层而不是所有物理后端的首次出现。

### 6.3 Rollout Gate 与真实验证分开

- 工程门：5/10/20步是否可运行，95%/80% 等几何有效率若设置为门槛应在实验前冻结；模型是否确实比 Classic 减少了无效步骤或改善了目标 CV 轨迹。
- 科学门：是否找到正确 TS，必须有 OptTS、单虚频、IRC 连接与端点确认；没有这些结果只可报告 `TS_candidate`，不能报告 `verified TS`。
- 成本账本：每个 episode 的 xTB/DFT 调用次数、core-hours、失败尝试、重试与墙钟时间全部入分母。

本次 30h 的默认 V5.3 目标是完成 **end-to-end 可执行闭环及机制比较**，不把它当成已经通过 V6 的正式物理搜索门。

## 7. 机器可执行门与报告格式

### 7.1 一次性冻结文件

```text
config/campaigns/pes2ts_30h_20261020.yaml
outputs/pesn/30h_campaign/protocol_30h.json
outputs/pesn/30h_campaign/data_manifest.json
outputs/pesn/30h_campaign/feature_spec.json
outputs/pesn/30h_campaign/teacher_manifest.json
outputs/pesn/30h_campaign/split_manifest.json
outputs/pesn/30h_campaign/run_plan.jsonl
```

协议须写：版本/commit/data_hash/teacher_hash/feature_hash、d0、曲线 canonical 映射、K 规则、intent reverse、loss 配置、optimizer、seed、fold、预注册 checkpoint/筛选/止损规则、可见/不可见 partition、wall-clock 截止、计算执行器版本与成本上限。

### 7.2 每个实验 `run_record.json`

```json
{
  "campaign_id": "PES2TS_30H_R3_V52_V53",
  "phase": "V5.2",
  "arm": "QT",
  "fold": 0,
  "seed": 17,
  "n_updates": 1900,
  "data_sha256": "REQUIRED_AT_RUNTIME",
  "teacher_sha256": "REQUIRED_AT_RUNTIME",
  "features_sha256": "REQUIRED_AT_RUNTIME",
  "split_sha256": "REQUIRED_AT_RUNTIME",
  "config_sha256": "REQUIRED_AT_RUNTIME",
  "checkpoint_update": null,
  "checkpoint_sha256": null,
  "status": "planned",
  "metrics_path": null
}
```

哈希必须由运行程序填入真实值，不能把以上 placeholder 当成有效报告。`run_plan` 和 `run_records` 分开保存，避免规划中的 run 被误计为已完成。

### 7.3 门状态与终局

各阶段独立状态：`pass | fail | blocked | partial | not_run`；具体失败原因例如 `data_leakage`,`gauge_unresolved`,`intent_underdetermined`,`invalid_target`,`gradient_unreachable`,`no_gain`,`stop_positive_insufficient`,`rollout_unstable`,`physics_not_available`。R3 自己的 campaign-level outcome 保留且只保留其已冻结的五种终局；V5.2/V5.3 另出各自 report，不把多个阶段混成一个 `scale_controlled`。

`approved_for_v5_1b=false` 始终不变。旧 R1、R2 的 53-run/旧标签不纳入 R3 66-run 的新样本结果。fold3/4 的内部确认可用一次；不得打开**正式 sealed/Confirm/OOD** 或 Fit150+。

### 7.4 全部测试清单（最小集）

| ID | 测试 | 通过条件 | 失败动作 |
|---|---|---|---|
| T0.1 | curve/node/edge/intent FK | 计数自洽、无孤儿、TS 唯一 | data_blocked |
| T0.2 | 正反教师 involution | `reverse(reverse(intent))==intent`；反向单步与逆序路径一致 | data_blocked |
| T0.3 | d0/gauge/P0 | d0=0.08；1248/5 overflow/342/2022 有处置 | data_blocked |
| T0.4 | split 泄漏 | 同 curve/intent/d0 同折；near-dup 不跨折 | data_blocked |
| T1.1 | Relabel | 特征逐比特相同；`max|Δu|≤1e-6` | feature_blocked |
| T1.2 | Reverse | signed intent 反转；输出改变；不要求 exact negative | feature_blocked |
| T1.3 | 旋转/平移/置换 | FP32 容差内等变 | feature_blocked |
| T1.4 | 输入合法性 | 无 TS/progress/future/endpoint geometry/branch | feature_blocked |
| T1.5 | λ=0 回归 | 与 B0-F 逐比特一致 | loss_blocked |
| T1.6 | deterministic shard | 两次构建 hash 相同 | feature_blocked |
| T2.1 | Move 指标 | RMSE/SRE/cos/sign/dead 达到预注册门 | no_gain/partial |
| T2.2 | strong/dead 梯度 | encoder 2%/10% 或 unreachable；head≤0.3 | scale_unreachable |
| T2.3 | 种子/簇稳定性 | ≥2/3 seeds，cluster 聚合 | no_gain |
| T3.1 | Progress | q MAE、单调性、TS 两侧符号优于基线 | progress_fail |
| T3.2 | Stop | PR-AUC > 阳性率；first-stop error 达标 | stop_fail |
| T3.3 | 类别权重 | 仅 inner-train 估计；Mask 不进分母 | stop_fail |
| T3.4 | Move 非退化 | 相对 R3 RMSE/zero 恶化≤2%，sign/dead 容差内 | joint_fail |
| T4.1 | 无松弛 rollout | 5/10/20 步可运行；几何有效率达标 | rollout_unstable |
| T4.2 | 受约束 RELAX | 执行器接受动作；CV 推进；不退回原势阱 | physics_fail |
| T4.3 | Classic 对照 | 同预算同约束 | comparison_fail |
| T4.4 | 成本账本 | QC calls/core-hours/失败/重试全部记录 | report_fail |
| T4.5 | 真实 TS | OptTS + 单虚频 + IRC + 端点；否则 `not_verified` | 不得称 verified |

每项测试必须输出 `pass | fail | blocked | partial | not_run` 和证据路径；不能只写“看起来通过”。

## 8. 意外情况时的止损与降级交付

| 故障 | 当前行动 | 当日仍应产出 |
|---|---|---|
| 数据门无法通过 | 不训练、不把坏样本暗中删去 | S1 分类、mask、覆盖率和 data_blocked |
| 两向 intent 无法区分 | 修复 CV_spec 或标记不可辨识，不加 branch token | 有效 intent 清单与负对照 |
| R3 指标反复退化 | 不继续网格搜 λ/重做 feature | 稳定 Move baseline + no_gain 报告 |
| Stop 正例太少 | 只完成 q head；Stop `insufficient_data` | 训练内阳性分布、无监督结论 |
| 联合 Q+Stop 破坏 Move | 冻结 backbone 或拆分 head 诊断，不回调确认折 | 单头结果与梯度冲突诊断 |
| 物理执行器准备不足 | 仅作纯几何 rollout 与 mock 集成 | V5.3 工程演示、`physics_not_run` |
| 提前触及30h | 结束正在运行的可安全 checkpoint 步骤、导出部分数据 | partial report + 确切已完成 run 数 |

## 9. 给新手的解释

- **一条 IRC 是一本书**：正着读和倒着读还是同一本书，不算两本独立样本。
- **Reaction Intent 是阅读方向**：它不等于把书拆成两个 branch。它告诉模型这一步应该按哪个化学目标推进。
- **训练样本是书中的一页**：给模型当前分子形状，让它猜下一步，而答案只在计算 loss 时对照。
- **Encoder 是共同的化学几何理解器**：把元素、键和三维距离变成隐藏表示；三个 Head 分别回答“往哪动”“现在在曲线哪里”“现在是不是适合停下提交 TS”。
- **Loss 是老师批改作业的扣分规则**：数值越小不意味着真实化学 TS 就越可信，还要看独立簇、方向、Stop 和物理验证。
- **Gradient ρ 是每个老师意见的音量**：辅助 loss 音量大过主任务会把学习目标带跑，太小则根本听不见；2%/10% 是事先规定的实验档，不是普适最优值。
- **Fold 是换一批完全没见过的反应来测试**：同一 IRC 的几千帧不是几千个不同反应；正反同折、重复同折。
- **Rollout 是闭卷模拟真实行动**：每一步只用自己产生的当前结构，不给下一帧答案；rollout 才能暴露微小错误不断积累的问题。
- **验证 TS 是独立化学实验**：Stop Head 说“可能到了”，不等于量化计算已经证明“确实到了”。

## 10. 最终建议

按 **E0–E2 先建正确数据与单帧输入合同 → E3 R3 有向完整 IRC 动作监督 → E4/E5 V5.2 q/TSStop 独立头与联合头 → E6 V5.3 受约束单端闭环** 执行。机器时间够支撑这套开发型战役，但不够保证论文级泛化/真实 TS 成功率；宁可让上游失败阻断下游，也不要以 run 数量代替科学证据。
