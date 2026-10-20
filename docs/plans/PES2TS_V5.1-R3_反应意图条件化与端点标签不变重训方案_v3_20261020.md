# PES2TS V5.1-R3：反应意图条件化与端点标签不变重训方案 v3

> **v3.1 监督修订（2026-10-20）**：R3 监督任务从“朝 TS 的局部步”改为 **`intent_directed_irc_step`**：一条完整 IRC、两种有向 Reaction Intent（A→B、B→A）、沿完整曲线做 first-contact 单步监督；TS 只是曲线节点，不是 R3 的停止目标。Stop/Progress 留给 V5.2。旧的 `ts_directed_step` 标签与结果不能作严格可比对照。
>
> **v3.2 数据表修订（2026-10-20，PD-1 裁定）**：R3 数据表结构以 [30h 计划 v1.1](PES2TS_30h_R3_V52_V53_分层训练与验收计划_v1.1_20261020.md) §3.1 **六表**为准（irc_curve / irc_node / irc_edge / reaction_intent / curve_quality / sample_view）；本节原五表清单作废。`irc_edge.delta_xyz` 为 flat 数值数组（shape/offset 可验证，绑定原子映射/单位/SHA；侧合法性 eps=1e-6；TS 权威身份由 ts_index 确定——PD-9）。
>
> 状态：**Proposed v3.2**（取代 v1/v2/v3.1；先于任何 R3 训练冻结）
> 日期：2026-10-20
> 核心合同：[ReactionIntent Contract v1](../contracts/PES2TS_ReactionIntent_Contract_v1.md)
> 上游：R3 v2 的数据合同、d0 决策、split gate、60-run 预算
> 关系：R3 做完整曲线的 `intent_directed_irc_step` 单步监督；CV/Jacobian、Progress/Stop、闭环生长属于 V5.2+

---

## 0. 一页结论

R3 的唯一原则性修订：

> **把 `endpoint-symmetric` 改为 `reaction-intent-conditioned, endpoint-label-invariant`。**

- 不再要求所有端点特征做无符号绝对值；
- 允许合法的有向反应编辑 \(\Delta B^{\rm intent}=B_{\rm tgt}-B_{\rm src}\)；
- 但禁止来自历史 branch 的方向编码；
- 仅交换 A/B 名字 → 输出不变；
- 真正反转反应目标 → 输出可改变；
- 当前几何仍是唯一三维输入；
- 历史 IRC/TS/未来帧仍只用于离线监督。

R3 的监督任务：

> **`intent_directed_irc_step`**：一条 canonical IRC、两种有向 intent（A→B、B→A）、沿完整曲线做 first-contact 单步位移监督。TS 只是曲线节点，不是 R3 的停止目标。

R3 不实现：
- CV/Jacobian 几何提升；
- Progress Head；
- Stop Head；
- 闭环曲线生长。

这些属于 V5.2+。

---

## 1. 方向语义

### 1.1 允许：Reaction Intent

\[
G_{\rm rxn}=(G_{\rm src},G_{\rm tgt},\mathcal K,\mathcal C,\mathcal R)
\]

\[
\Delta B^{\rm intent}_{ij}=B^{\rm tgt}_{ij}-B^{\rm src}_{ij}
\]

- \(\Delta B>0\)：目标键级增大；
- \(\Delta B<0\)：目标键级减小；
- source/target 由反应任务定义，不由 branch 定义。

### 1.2 禁止：branch 方向

- 不使用 `branch0/branch1`；
- 不使用由 branch 决定的 `b_start/b_other/db`；
- 不使用单侧 `anchor_vec`；
- 不使用历史 IRC 起点作为方向。

### 1.3 两种测试

#### Relabel invariance

\[
\pi(X,\mathrm{Relabel}(G_{\rm rxn}))
=
\pi(X,G_{\rm rxn})
\]

只是交换端点名字，输出不变。

#### Reverse sensitivity

\[
\pi(X,\mathrm{Reverse}(G_{\rm rxn}))
\ne
\pi(X,G_{\rm rxn})
\]

真正反转反应目标，输出可改变；小步长下切向可能近似相反，但不要求有限步长严格互为负数。

---

## 2. 数据合同

**（v3.2 修订：PD-1）六表结构为准**，落 `outputs/pesn/30h_campaign/data/`（战役根规则，见 [agent_runs/parallel_config](../agent_runs/state/parallel_config.json)）：

```text
outputs/pesn/30h_campaign/data/
  irc_curve.csv               # 一条完整 A—TS—B 物理曲线一行；d0 不进曲线身份
  irc_node.csv                # 节点/坐标/有限性；FK→curve
  irc_edge.csv                # 相邻帧有向边；delta_xyz=flat数组+shape/offset+映射/单位/SHA（PD-9）
  reaction_intent.csv         # 有向任务 G_src,G_tgt,ΔB_intent,K,CV_spec,rules
  curve_quality.csv           # 曲线级质量/血缘
  sample_view.csv             # (curve_id,node_index,intent_id,d0) 训练样本视图
  branch_diagnostics.csv      # 仅诊断/溯源（历史 segment/branch quality），非六表成员、不入模型
  split_manifest_r3.json / data_manifest.json / teacher_manifest.json
```

### reaction_intent.csv

```text
intent_id
curve_id / canonical_irc_id
G_src, G_tgt
K
CV spec
rules
delta_B_intent
source/target metadata
intent_version
```

### sample_view.csv

```text
sample_id = (curve_id, node_index, intent_id, d0)
state: xyz, graph
target_first_contact (node/alpha)
target_action u
target_progress q
stop_label
label_status / disposition
label_weight / curve_weight / sample_weight
feature_sha / label_sha
```

规则：

- 没有 `branch_id` 进入模型；
- `branch_diagnostics` 只做数据质量；
- d0=0.08（canonical main）；
- forward/reverse 共用同一条曲线、同一 fold；
- 反向样本不是独立 IRC；
- 曲线级采样：一条曲线总权重固定，按 intent/node 分摊；
- forward-only 与 bidirectional 使用相同曲线预算/优化步数/模型容量；
- split 按 canonical IRC / near-dup component；
- 1248 排除；5 个 mapping overflow quarantine；
- 342 对交集闭环；
- 2022 按 d0=0.08 复核。

### 2.10 R3 数据审计问题

不再问“属于哪条 branch”，而是问：

> **给定 \(X_t\) 与完整 \(G_{\rm rxn}\) 后，教师动作是否基本成为单值、可学习的映射？**

需要标记的歧义：

- 同一 source/target 图对应多个 TS 构象/通道；
- 纯构象变化，\(\Delta B_{\rm intent}=0\)，仅靠键图无法定义方向；
- 近重复曲线/不同 IRC 在局部几何上不可区分；
- gauge/映射导致同一输入对应不同标签。

处理：

- 能用 `CV_spec`/构象约束区分的，补 intent；
- 不能区分的，标 `ambiguous`，不进硬标签训练；
- 不强行把歧义样本压成单值。

---

## 3. 特征合同

### 3.1 四个通道

| 通道 | 内容 | 对称性 |
|---|---|---|
| **Chemical Context** | 元素、当前几何、无序拓扑、变化键集合 | 端点重命名不变 |
| **Reaction Intent** | \(\Delta B^{\rm intent}\)、目标 CV 增量、约束规则 | 随真实目标反转而改变 |
| **Geometric State** | 当前键长、角度、关键原子相对位置 | 只依赖当前构型 |
| **Teacher Diagnostics** | IRC 位置、历史 branch、TS、未来帧 | 禁止进入 forward |

### 3.2 正式 F2 特征

#### Node

```text
z_atomic        nn.Embedding(119,16)
is_key          1
degK            1
intent_role     3   # breaking / forming / spectator，由 incident delta_B_intent 推导
```

#### Edge

```text
is_spatial          1
B_src               1
B_tgt               1
delta_B_intent      1   # B_tgt - B_src
abs_delta_B         1
is_changed          1   # 1[abs_delta_B > 0]
```

#### Vector

```text
pos
```

#### Task

```text
无（d0 固定 0.08）
```

### 3.3 与 v2 的差异

- v2 把所有端点变化做成无符号 `B_min/B_max`；
- v3 保留有符号 `delta_B_intent`，因为它是反应目标，不是 branch；
- `intent_role` 允许模型知道“这条边目标是要断还是要成”；
- 但 source/target 必须来自 `G_rxn`，不能来自历史 branch。

### 3.4 特征阶梯

| 级别 | 开放特征 |
|---|---|
| **F0** | 元素 + `pos` + 基础拓扑 |
| **F1** | + `is_key` + `degK` |
| **F2** | + `B_src/B_tgt/delta_B_intent/abs_delta/is_changed/intent_role` |
| **F3** | + RBF(12) |

固定 F3 superset 形状 + feature mask；同一 seed 相同网络形状/初始化/采样顺序。

增加 **Intent Ablation**：

```text
F2-intent-on
F2-intent-off（把 delta_B_intent/intent_role 置零）
```

证明 signed intent 是否真的有价值。

---

## 4. 对称性测试

### 4.1 Relabel

- 交换端点名字；
- `G_src/G_tgt` 不变；
- 特征张量应逐比特相同；
- 输出不变。

### 4.2 Reverse

- 交换 `G_src/G_tgt`；
- `delta_B_intent` 变号；
- 输出应改变；
- 小步长切向近似相反，只作诊断，不作硬约束。

### 4.3 其他

- 旋转/平移等变；
- 原子重编号等变；
- 镜像行为说明；
- median/P99/max 误差；
- near-zero 输出单独统计。

门禁：

```text
Relabel: max |Δu| < 1e-6（输入应逐比特相同）
Reverse: output changed, 但不要求 exact negative
Rotation: equivariance error < 1e-4
Permutation: equivariance error < 1e-4
```

---

## 5. 损失

R3 不引入 progress/stop/CV loss，只保留 `intent_directed_irc_step` 的位移监督。

监督目标 \(u^*\) 是沿当前 intent 方向 first-contact 的目标位移，不一定是相邻帧：

\[
u_i^\pm=X_{j^\pm,K}-X_{i,K}
\]

其中 \(j^\pm\) 由 `first_contact_along_curve(i, ±1, d0=0.08)` 给出。

### 5.1 B0-F

\[
L_{\rm base}=L_{\rm atom}+0.5L_{\rm pair}+0.02L_{\rm cos}
\]

### 5.2 Strong / Dead

\[
L_{\rm strong}
=
\frac{\sum_b w_b \mathbf1[|S_b|>0]L_{{\rm strong},b}}
{\sum_b w_b \mathbf1[|S_b|>0]}
\]

\[
L_{{\rm strong},b}
=
\frac{1}{|S_b|}
\sum_{p\in S_b}
\mathrm{Huber}_{0.25}
\left(
\frac{\hat{\Delta d}_p-\Delta d_p^*}
{\max(|\Delta d_p^*|,0.01)}
\right)
\]

\[
L_{\rm dead}
=
\frac{\sum_b w_b \mathbf1[|D_b|>0]L_{{\rm dead},b}}
{\sum_b w_b \mathbf1[|D_b|>0]}
\]

\[
L_{{\rm dead},b}
=
\frac{1}{|D_b|}
\sum_{p\in D_b}
\left[
\frac{\max(0,|\hat{\Delta d}_p|-0.005)}{0.005}
\right]^2
\]

- \(w_b=\) sample_weight；
- 所有损失项使用同一 \(w_b\)；
- 权重只乘一次；
- 空桶为 0。

### 5.3 总损失

\[
L=L_{\rm base}+\lambda_sL_{\rm strong}+\lambda_dL_{\rm dead}
\]

第一轮不同时开 strong/dead。

### 5.4 梯度

\[
\rho_{k,g}
=
\frac{
\|\nabla_{\theta_g}(\lambda_kL_k)\|
}{
\|\nabla_{\theta_g}L_{\rm base}\|+\epsilon
}
\]

- shared encoder 主指标；
- head 同时报告；
- 20 batch median；
- 2%/10%；
- λ 上限 0.5；
- `scale_unreachable` / `target_unreachable` / `head_overweighted`。

---

## 6. 实验矩阵

| 阶段 | 内容 | 新 run |
|---|---|---:|
| **F-main** | F0/F1/F2/F3，forward-only，fold0 × 3 | 12 |
| **F-bidir** | F2，A→B + B→A 联合，等曲线预算，fold0 × 3 | 3 |
| **F-ablate** | F2，屏蔽 signed intent，fold0 × 3 | 3 |
| **B** | 选中 F 的 B0-F，补 folds1–4 × 3 | 12 |
| **L** | S2/S10/D2/D10，fold0 × 3 | 12 |
| **V** | Top-2，folds1–2 × 3 | 12 |
| **R** | 最佳 loss × R1/R2，fold0 × 3；R0 复用 | 6 |
| **C** | Top-1，folds3–4 × 3 | 6 |
| **合计** | | **66** |

规则：

- **主实验 = forward-only**；
- bidirectional 是增强臂，不假设更好；必须与 forward-only 使用相同曲线级采样预算、优化步数、模型容量；
- intent-off 用于验证 signed intent 的必要性；
- 反向样本与正向共享同一 IRC、同一 fold，不能当独立 curve 计权；
- 如果 bidirectional 只在“看得更多数据/更多更新”时更好，不算双向监督收益。

如预算严格，先做：

```text
B0-F forward fold0 × 3
+ intent ablation × 3
+ bidirectional F2 × 3
+ S2/S10/D2/D10 fold0 × 3
```

约 21 run。

---

## 7. 代码修改

| 文件 | 修改 |
|---|---|
| `contracts/key_atom_v3.py` | intent 字段、v3 槽位 |
| `features/key_atom_spec_v3.py` | v3 spec |
| `features/key_atom_graph_v3.py` | Reaction Intent 特征 |
| `offline/data/key_atom_dataset_v3.py` | manifest-driven，intent_id |
| `models/key_atom_egnn_v3.py` | 固定 F3 shape + mask |
| `offline/training/key_atom_bc_v3.py` | v3 loss、weights |
| `scripts/campaigns/r3_build_data.py` | 三表 + intent |
| `scripts/campaigns/r3_build_features.py` | v3 shard |
| `scripts/campaigns/run_v5_1_r3.py` | 阶段驱动 |
| `scripts/campaigns/r3_gradient_probe.py` | ρ |
| `scripts/campaigns/r3_report.py` | gate |
| `tests/pesn/test_r3_relabel_invariance.py` | Relabel 门 |
| `tests/pesn/test_r3_reverse_sensitivity.py` | Reverse 诊断 |
| `tests/pesn/test_r3_input_legality.py` | 输入白名单 |
| `tests/pesn/test_key_atom_losses_r3.py` | λ=0/空桶/权重 |

v1/v2 模块只读。

---

## 8. 验收

| Gate | 条件 |
|---|---|
| Data | canonical/intent/branch/sample 计数自洽；d0=0.08；无阻断 P0 |
| Feature | Relabel invariant；Reverse sensitive；旋转/平移/置换通过；无诊断泄漏 |
| Baseline | B0-F 独立完成；folds3–4 隔离 |
| Loss | λ=0 与 B0-F 逐比特一致；权重只一次；空桶正确 |
| Gradient | 2%/10% 或 unreachable；head ≤0.3 |
| Dev | Top-2 预注册 |
| Confirm | 仅 dev-positive；两折/指标/bootstrap 合格 |

五终局：

1. `data_blocked`
2. `feature_blocked`
3. `scale_unreachable`
4. `no_gain`
5. `scale_controlled`（必须 confirm 通过）

`frozen_candidate_r3.json`：

```json
{"approved_for_v5_1b": false}
```

---

## 9. 一句话

**R3 不是“所有端点特征都取绝对值”；R3 是“反应意图条件化、端点标签不变”。合法的有向 graph edit 必须保留，来自 branch 的方向编码必须删除。**
