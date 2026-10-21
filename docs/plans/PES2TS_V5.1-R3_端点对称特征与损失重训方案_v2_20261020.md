# PES2TS / PES-N V5.1-R3：端点对称特征与损失重训方案 v2

> **Superseded（2026-10-20）**：本文件的 `endpoint-symmetric` 语义已被 [R3 v3](PES2TS_V5.1-R3_反应意图条件化与端点标签不变重训方案_v3_20261020.md) 修正为 `reaction-intent-conditioned, endpoint-label-invariant`；请勿再按“所有端点特征取无符号绝对值”执行。
>
> 状态：**Proposed v2 / Superseded by R3 v3**（修订 v1；先于任何 R3 训练冻结）
> 日期：2026-10-20
> 关系：**取代 v1**；R2 的 strong/dead 公式保留，但 baseline、λ、checkpoint、梯度探针全部重来
> 上游证据：`outputs/pesn/v5_1_r1/final_report.md`、`outputs/pesn/v5_1_r1/protocol_r1.json`
> 下游：V5.2 新 Key-Atom 合同、V5.3 纯几何 EGNN、V6.1 物理 MLP

---

## 0. 一页结论

V5.1-R3 要证明三个连续假设：

- **H1（Endpoint-Symmetric Representation）**：不需要 branch 身份和历史 anchor，仅用当前单帧几何 + 无序端点拓扑，能否学习具有合理准确度和几何等变性的 key-atom 位移？
- **H2（Consistent IRC Supervision）**：在 canonical IRC、gauge、near-duplicate、教师冲突处置后，剩余标签是否足以支持可靠的单帧映射？
- **H3（Controlled Chemical Auxiliary Supervision）**：在端点对称、数据可信的 B0-F 上，梯度受控的 strong/dead 能否改善反应坐标预测，同时保持基础几何质量？

H1 不成立 → 不进入 λ；H2 不成立 → 预测错误可能来自不可辨识教师；只有 H1/H2 成立，H3 才有意义。

本轮定义：

> **V5.1-R3 = 合法单帧输入下的 key-atom 位移监督基线，不是完整 Geometry Policy，不包含 No-Move/Stop Head。**

R3 不打开 sealed/Confirm/OOD，不启动 Fit150+；V5.1b 继续关闭。

---

## 1. 范围与非目标

### 1.1 范围

- 固定主步长 `d0=0.08 Å`（canonical main；见 §3.1）；
- 修复 canonical IRC、branch、sample 三层数据合同；
- 特征改为严格 endpoint-swap invariant；
- 重训 B0-F；
- 在 B0-F 上校准无量纲 strong/dead；
- 给出五种终局之一。

### 1.2 非目标

- 不实现 V5.2 新网络、Stop Head、V6 xTB、V7 RL；
- 不做正式 OOF；
- 不打开 sealed/Confirm/OOD；
- 不启动 Fit150+；
- 不把 branch 加回输入；
- 不使用真实端点几何作为正式推理特征；
- 不在 R3 主矩阵同时改 split、特征、损失；
- 不使用 confirm 折回头调参。

---

## 2. 与 V5.1-R2 的关系

| 项目 | V5.1-R2 | V5.1-R3 |
|---|---|---|
| 特征 | 保持 v1 | 端点对称 v2，去掉 branch |
| 主 d0 | 0.05（与 V5.1 配置一致） | **0.08（canonical main）** |
| baseline | 复用旧 B0 | 从零训练 B0-F |
| 旧 53-run | 可直接对比 | 仅历史参考，不作 gate |
| strong/dead 公式 | 无量纲 | 保留同一公式 |
| λ | 旧 B0 校准 | B0-F 重新校准 |
| 梯度分母 | L_atom | **完整 L_base** |
| 结论 | 旧任务损失修复 | 特征 + 损失联合重训 |

**R3 不是 R2 的补跑，而是新数据/特征合同下的新 campaign。**

---

## 3. 数据合同

### 3.1 d0 语义：先解决 P0 不一致

仓库当前存在明确冲突：

```text
pes2ts_core/pesn/offline/data/key_atom_teacher.py: MAIN_D0 = 0.08
config/campaigns/v5_1_r1.yaml:                  d0_angstrom: 0.05
outputs/pesn/v5/features/d0_0.05:                V5.1/R1 实际使用
```

R3 必须先冻结唯一主步长：

| 方案 | 规则 | 适用 |
|---|---|---|
| **A（推荐）** | R3 主矩阵固定 `d0=0.08`，`0.05/0.12` 仅敏感性 | 对齐 canonical plan 与 `MAIN_D0` |
| B | R3 固定 `d0=0.05` | 与 R1 证据/当前特征一致，但不是 canonical main |

**推荐 A：`d0=0.08`。** 理由：

- `key_atom_teacher.py` 的 canonical main 是 0.08；
- V5 计划写明主 d0=0.08；
- R3 已经因特征变化无法复用旧 baseline，不如同时修正 d0 语义；
- 0.05/0.12 仍可作敏感性，不进入主矩阵。

强制规则：

```text
sample_id 必须包含 d0：
(canonical_irc_id, branch_id, start_row, d0)
```

即使 R3 固定单一步长，也要把 `d0` 写进主键，防止未来多步长碰撞。

**2022 必须按 d0 重新复核。** 0.05 下 last-step 0.067 > 0.05；0.08 下 0.067 < 0.08，contact/terminal 结论可能不同。不能用 0.05 的结论替代 0.08。

### 3.2 三层数据表

```text
outputs/pesn/v5_1_r3/data/
  canonical_irc.csv
  branch_diagnostics.csv
  sample_view.csv
  split_manifest_r3.json
  data_manifest.json
```

#### canonical_irc.csv

| 字段 | 定义 |
|---|---|
| `canonical_irc_id` | 规范图 + 元素标记 key 距离谱 + charge/mult |
| `source_irc_hash` | 原始 IRC 文件哈希（血缘） |
| `canonical_mapping_sha` | 规范 atom mapping 哈希 |
| `ts_geometry_sha` | TS 几何规范化哈希 |
| `ts_raw_hash` | 原始坐标哈希（仅溯源，不作唯一去重） |
| `reaction_ids` | 映射到该 IRC 的原始 RXN 列表 |
| `split_group_id` | 同一 canonical IRC 的 split 单元 |
| `n_branches` | 2 |
| `n_frames` / `n_eligible` / `n_gauge` | 计数 |
| `endpoint_match_mode` | `pair_set` / `pair_order_signature` / `graph_ambiguous_fallback` |
| `mapping_status` | `ok` / `overflow` |
| `near_dup_component_id` | 经验近重复连通分量 |
| `family_id` | 仅分层，不合并身份 |
| `data_quality_status` | `ok` / `suspect` / `quarantine` |

**身份规则：**

- `reaction_id`、`canonical_irc_id`、`near_dup_component_id` 是三种不同身份，不能合并；
- 一个 `reaction_id` 对应多个 IRC **不自动判错**；先标记 P0 待分类：不同 TS 构象/通道/立体选择性/重复计算；
- 只有身份混淆、来源不一致、输入-标签不可辨识，才是阻断性 P0；
- `ts_raw_hash` 不能作为唯一去重依据；同一 TS 可能对应不同轨迹采样；
- `EXACT_IRC_DUPLICATE` 表示整条 IRC 在规范映射和容差下等价，不只是 TS 坐标相同。

#### branch_diagnostics.csv

```text
branch_id, irc_id, side,
n_frames, n_teacher_rows, n_eligible,
n_contact, n_terminal, n_capped, n_other,
max_step_key_rmsd, last_step_key_rmsd, cum_path_key_len,
frame_layer, gate_ok, endpoint_match_mode,
teacher_direction, branch_endpoint_hash,
label_generation_version,
short_branch, coarse_step, branch_weight
```

硬不变量：

```text
n_teacher_rows = n_frames - 1
n_contact + n_terminal + n_other = n_teacher_rows
n_capped 是独立诊断标记，不计入互斥 outcome
branch0.TS == branch1.TS
branch0.key_rows == branch1.key_rows
teacher_direction = "toward_TS"
```

**Teacher 方向统一：** 两条原始 IRC 半路径若都从 TS 向外积分，则必须反向构造监督样本，使教师位移均指向 TS。TS 帧不能同时以两个相反的“向外动作”成为普通 MOVE 样本。

#### sample_view.csv

```text
sample_id, irc_id, branch_id, start_row, d0, fold,
target_irc_id, source_row_sha,
label_status, disposition,
teacher_valid, sampling_group_id, loss_eligible,
label_weight, branch_weight, sample_weight,
feature_sha, label_sha
```

`label_status`：

```text
ok_contact / ok_terminal / gauge_unresolved /
no_future / reference_invalid / conflict / suspect
```

`disposition`：

```text
train_hard / train_soft / eval_only / exclude
```

规则：

- 每个 sample 有且只有一个 status/disposition；
- gauge 不进训练，保留在评估 coverage；
- 不删除困难样本；
- 1248 明确 `excluded_gauge_unresolved`；
- 5 个 mapping overflow 先进 `quarantine`。

### 3.3 P0 处置

| 问题 | 处置 |
|---|---|
| `RXN_0000001248` 全 gauge | `excluded_gauge_unresolved`；fold0 按 9 簇报告 |
| 2022 全 contact | 按 `d0=0.08` 重新复核；用 `teacher_outcome` 判，不用 0.05 结论 |
| 5 个 mapping overflow | 核查 `pair_edit` 与 R/P 键变化；不一致则 `quarantine` |
| 342 对历史冲突 | 与 fit50 求交集；空则记 `R3_scope_not_affected`，不声称历史问题已解决 |
| 3 个缺失 family label | 标记 `MISSING_FAMILY`，只影响分层 |

### 3.4 Split Gate

R3 主实验暂不改 split，但必须过两层门。

#### Gate A：确定交叉污染

- `fold(branch_A) == fold(branch_B)`；
- 同一 `canonical_irc_id` 的所有 `d0`、所有帧、所有别名 reaction ID 在同一 split group；
- 同一 `near_dup_component_id` 的样本不能跨训练/评估折；
- 发现实际泄漏 → 不能因为旧 split 冻结就继续作为可信泛化评估。

#### Gate B：Shadow split 敏感性

- 按 `near_dup_component_id` 重分组，生成 `split_manifest_v5_1_shadow`；
- 不以“20% 簇换折”作为泄漏判据；
- 以**实际跨训练/评估边**作为泄漏诊断；
- 以 cluster-level 性能变化作为 split 敏感性诊断；
- 保留原 split 供 R3 内部受控比较；
- 如果旧 split 确认泄漏，阻止其作为独立确认结论的依据。

`key-TS RMSD <0.25 Å` 只作 near-duplicate 候选筛选，不能单独认定两条完整路径等价；还要核对周围原子环境、atom mapping、反应身份。

---

## 4. 特征合同 v2

### 4.1 最高优先级：推理合法性

\[
\boxed{
\text{所有 forward 特征必须能够仅由 }(X_t,H)\text{ 重建}
}
\]

其中 \(H\) 只包含推理时可获得的合法化学约束。

允许：

- 当前帧 \(X_t\)；
- 当前元素/拓扑；
- 已知反应约束 \(G_A,G_B\)；
- 由当前几何计算的量。

禁止作为正式输入：

- 真实端点三维坐标 \(X_A,X_B\)；
- 由真实端点坐标计算的 \(d_A/d_B/s\)；
- IRC 历史/未来帧；
- TS 坐标；
- `d_eff/terminal/capped/near_ts/key_progress/reason_id`；
- 教师标签、split、family、conflict class。

**由代码白名单、离线/推理一致性测试、数据溯源检查共同保证，不能依赖自觉。**

### 4.2 端点交换不变原则

A–TS–B 没有内在正向。交换 A/B：

```text
B_A <-> B_B
ΔB -> -ΔB
```

目标 \(u\) 不变。

因此正式特征必须满足：

\[
f(X,G_A,G_B)=f(X,G_B,G_A)
\]

### 4.3 正式 F2 最小特征

#### Node

```text
z_atomic    nn.Embedding(119,16)
is_key      1
degK        1
```

维度：18。

**不放 `s`。** 因为 `s` 交换 A/B 会变号；正式 F2 不使用任何端点几何派生变量。

#### Edge

```text
is_union_covalent   1   # 1[B_A>0 or B_B>0]
is_covalent_both    1   # 1[B_A>0 and B_B>0]
B_min               1   # min(B_A,B_B)
B_max               1   # max(B_A,B_B)
abs_dB              1   # B_max - B_min
is_spatial          1
```

维度：6 + RBF。

说明：

- 这些量交换 A/B 后逐比特不变；
- `changed = 1[abs_dB>0]` 可由 `abs_dB` 推导，正式特征不重复放；
- `is_union_covalent` / `is_covalent_both` 替代有序的 `is_covalent_A/B`；
- `B_min/B_max/abs_dB` 替代有序的 `B_A/B_B`；
- 如果证明需要方向符号，只能通过法律代理或 oracle 臂，不能直接用 `B_A-B_B`。

#### Vector

```text
pos
```

正式 R3 不放 `X_A/X_B`。

#### Task

```text
无（d0 固定 0.08）
```

如果未来多步长，`d0` 进入 sample key 和 task，但 R3 主矩阵固定。

### 4.4 可选 oracle / 诊断特征（不进正式 F2）

如果确实需要“当前结构更接近哪一端”的进度代理，只能使用**当前几何 + 端点拓扑**构造，不能使用真实端点坐标。

定义当前键存在度：

\[
\tilde b_{ij}(X_t)=\phi(r_{ij};Z_i,Z_j)
\]

其中 \(\phi\) 只使用当前距离和冻结的元素键长先验。

端点拓扑不匹配度：

\[
D_A(X_t)=\sum_{(i,j)\in K} w_{ij}
\left|\tilde b_{ij}(X_t)-\mathbf1[B_{A,ij}>0]\right|
\]

\[
D_B(X_t)=\sum_{(i,j)\in K} w_{ij}
\left|\tilde b_{ij}(X_t)-\mathbf1[B_{B,ij}>0]\right|
\]

平滑方向因子：

\[
s_{\rm dir}=\tanh\left(\frac{D_B-D_A}{0.05}\right)
\]

对称进度：

\[
progress_{\rm sym}=\frac{|D_B-D_A|}{D_A+D_B+\epsilon}
\]

可选 signed change：

\[
s\_dB = s_{\rm dir}\cdot(B_{\max}-B_{\min})\ \text{的有符号版本}
\]

注意：

- `s_dir` 用 `tanh` 防止 TS 附近归零；
- 但这些量仍然只是**当前几何相对两个拓扑端点的匹配度代理**，不是 IRC progress，也不是真实 TS 距离；
- R3 主 F2 **先不用**；只有 H1 完成后，才做独立消融；
- 任何 oracle 臂必须与可部署模型分开报告，不能混作候选。

### 4.5 特征阶梯与固定网络形状

为避免“输入维度变化”导致模型容量/初始化不同，采用 **F3 superset schema + feature mask**：

| 级别 | 开放特征 | 研究目的 |
|---|---|---|
| **F0** | 元素 + `pos` + 基础交换不变边图 | 最小几何基线 |
| **F1** | F0 + `is_key` + `degK` | 反应中心信息价值 |
| **F2** | F1 + `B_min/B_max/abs_dB/is_union_covalent/is_covalent_both` | 反应拓扑条件价值 |
| **F3** | F2 + 当前距离 RBF(12) | 连续距离展开价值 |

- 所有级别使用**同一 F3 superset 张量形状**；
- 未开放槽位置零；
- 同一 seed 使用相同网络形状、相同初始权重、相同采样顺序；
- 只改 feature mask；
- 原 F4（端点几何）只作离线 oracle 上限，不进正式梯度矩阵；
- 原 F5（hop/charge/aromatic）在 R3 之后按失败模式再加；
- hop 必须在端点拓扑并图上计算，不能悄悄使用某一侧起点键图。

### 4.6 对称性测试

至少四类：

\[
f(X,G_A,G_B)=f(X,G_B,G_A)
\]

\[
f(RX+t,G)=Rf(X,G)
\]

\[
f(PX,PG)=Pf(X,G)
\]

以及镜像行为说明（普通极性向量 EGNN 对镜像的行为需明确）。

报告：

- median 和 P99 相对 swap 误差；
- 最大绝对位移差（Å）；
- near-zero 输出单独统计；
- 随机刚体旋转/平移；
- 合法原子重编号；
- 输入张量规范化后应逐比特相同，再验证网络输出；
- 不依赖训练让网络“学会”对称性。

门禁：

```text
median C_swap < 1e-3
P99 C_swap < 1e-2
max |Δu| < 1e-3 Å（非 near-zero）
near-zero 单独报告
```

### 4.7 feature_manifest.json

```json
{
  "schema": "pesn_v5_r3_feature_manifest_v1",
  "feature_spec": "v2",
  "feature_spec_sha": "...",
  "data_manifest_sha": "...",
  "d0": 0.08,
  "f0_level": "F2",
  "rbf_dim": 12,
  "n_samples": 0,
  "n_train": 0,
  "n_eval_eligible": 0,
  "shards": {"...": "sha256..."}
}
```

训练 run 必须记录 `feature_manifest_sha256`。

---

## 5. 损失函数

### 5.1 B0-F 基线

\[
L_{\rm base}=L_{\rm atom}+0.5L_{\rm pair}+0.02L_{\rm cos}
\]

- `L_atom`：主位移损失；
- `L_pair`：`legacy_j`，只对 `abs_dB>0` 的 pair；
- `L_cos`：教师幅度门控余弦；
- `lambda_s=lambda_d=0`；
- 无 magnitude head，无 legacy sign/dead。

要求：λ=0 时与 B0-F 逐比特一致。

### 5.2 主损失

\[
L_{\rm atom}
=
\frac{\sum_b w_b L_{{\rm atom},b}}
{\sum_b w_b}
\]

\[
L_{{\rm atom},b}
=
\sum_i w_{b,i}
\mathrm{Huber}_{0.25}
\left(
\frac{\|\hat u_{b,i}-u^*_{b,i}\|}{0.08}
\right)
\]

\[
w_{b,i}
=
\frac{1/\max(1,degK_{b,i})}
{\sum_j 1/\max(1,degK_{b,j})}
\]

`w_b = sample_weight_b`。

### 5.3 Strong / Dead：按 frame 归一，再按 frame 加权

先在每个 frame 的 bucket 内取均值：

\[
L_{k,b}
=
\frac{1}{|K_{k,b}|}
\sum_{p\in K_{k,b}}\ell_{k,p}
\]

再在有效 frame 之间按 `w_b` 聚合：

\[
L_k
=
\frac{\sum_b w_b \mathbf1[|K_{k,b}|>0]L_{k,b}}
{\sum_b w_b \mathbf1[|K_{k,b}|>0]}
\]

空 bucket 为 0；整个 batch 无该类型 pair 时严格返回 0。

**不要用全 batch pair mean**，否则 pair 多的复杂反应会主导梯度。

#### Strong

\[
\ell_{{\rm strong},p}
=
\mathrm{Huber}_{0.25}
\left(
\frac{\hat{\Delta d}_p-\Delta d_p^*}
{\max(|\Delta d_p^*|,0.01)}
\right)
\]

\[
K_{{\rm strong},b}=\{p:|\Delta d_p^*|\ge0.02\}
\]

#### Dead

\[
\ell_{{\rm dead},p}
=
\left[
\frac{\max(0,|\hat{\Delta d}_p|-0.005)}{0.005}
\right]^2
\]

\[
K_{{\rm dead},b}=\{p:|\Delta d_p^*|<0.005\}
\]

### 5.4 总损失

\[
L_{\rm R3}=L_{\rm base}+\lambda_s(t)L_{\rm strong}+\lambda_d(t)L_{\rm dead}
\]

第一轮：

```text
lambda_mag = 0
lambda_sign = 0
lambda_dead_legacy = 0
pair_loss = legacy_j
```

不同时开 strong 和 dead。

### 5.5 权重只施加一次

```text
sample_weight = label_weight × branch_weight
```

- 所有损失项都使用同一个 `w_b`；
- 如果 sampler 已按 q/reaction 加权，则 loss 不再重复乘同等权重；
- 记录每个 sample 的实际采样概率；
- 以 canonical IRC 作为主要均衡单位，不以 sample row 为唯一单位；
- 两种方案二选一，不能混用不记录。

### 5.6 梯度校准

分母必须是完整 B0-F：

\[
\rho_{k,g}
=
\frac{
\|\nabla_{\theta_g}(\lambda_k L_k)\|
}{
\|\nabla_{\theta_g}L_{\rm base}\|+\epsilon
}
\]

- \(g\)：shared encoder / action head；
- 主校准指标固定 **shared encoder**；
- 同时保留相对 `L_atom` 的比值作诊断；
- 20 个真实 batch，取 **median**；
- \(\lambda_{\rm cos}\) 使用 checkpoint 实际生效值；
- 2%/10% 两档；
- λ 上限 0.5；
- 每 100 步复测只监测，不自动调 λ。

### 5.7 状态

- \(\rho<1\%\) at λ cap → `scale_unreachable`；
- \(\rho>1\%\) 但不在目标 ±30% → `target_unreachable`；
- head \(\rho>0.3\) → `head_overweighted`；
- 达标且指标正向 → 进入 dev。

---

## 6. 代码修改

### 6.1 新增模块（v1 只读）

| 文件 | 作用 |
|---|---|
| `contracts/key_atom_v2.py` | v2 槽位、维度、batch 字段 |
| `features/key_atom_spec_v2.py` | v2 spec、RBF、尺度、manifest |
| `features/key_atom_graph_v2.py` | 端点对称特征构建 |
| `offline/data/key_atom_dataset_v2.py` | manifest-driven dataset |
| `models/key_atom_egnn_v2.py` | 固定 F3 shape、feature mask |
| `offline/training/key_atom_bc_v2.py` | v2 loss、masks、weights |
| `scripts/campaigns/r3_build_data.py` | 三张主表 |
| `scripts/campaigns/r3_build_features.py` | v2 shard |
| `scripts/campaigns/run_v5_1_r3.py` | 阶段驱动 |
| `scripts/campaigns/r3_gradient_probe.py` | ρ / λ |
| `scripts/campaigns/r3_report.py` | gate report |

### 6.2 测试

```text
tests/pesn/test_key_atom_features_r3.py
tests/pesn/test_key_atom_losses_r3.py
tests/pesn/test_r3_data_contract.py
tests/pesn/test_r3_driver_plan.py
tests/pesn/test_r3_swap_invariance.py
tests/pesn/test_r3_input_legality.py
```

### 6.3 代码原则

- v1 模块不改；
- v2 新建文件，不原地改 schema；
- 训练器拒绝 `branch`、`X_A/X_B`、`d_eff/terminal/near_ts/capped/reason_id` 进入 forward；
- 训练器拒绝 `feature_manifest_sha` 不匹配；
- 训练器拒绝 `d0` 与 manifest 不一致；
- shard 写入前做指针/形状/finite 检查；
- 所有随机性有 seed；
- feature mask 不改变模型形状。

---

## 7. 实验矩阵

| 阶段 | 内容 | 新 run |
|---|---|---:|
| **F** | F0/F1/F2/F3，fold0 × 3 | 12 |
| **B** | 选中 F 的 B0-F，补 folds1–4 × 3 | 12 |
| **L** | S2/S10/D2/D10，fold0 × 3 | 12 |
| **V** | Top-2，folds1–2 × 3 | 12 |
| **R** | 最佳辅助损失 × R1/R2，fold0 × 3；R0 复用 | 6 |
| **C** | Top-1，folds3–4 × 3 | 6 |
| **合计** | | **60** |

- F 阶段胜出的 fold0 三个 run 直接复用为 B0-F fold0 baseline；
- F0–F3 使用同一 F3 superset 形状与 feature mask；
- 不把 F4 端点几何放进主矩阵；
- GPU 约 150 min；总预算 ≤900 min；
- dry-run 后重新核算实际耗时。

### Ramp 规则

- **R0 是主实验固定日程**；
- R1/R2 只提供诊断证据，不改变本次 confirm 候选；
- 如果 R1/R2 没有 folds1–2 复现，不得声称跨折最佳 ramp；
- ramp 分三层：fold0 时序诊断、folds1–2 开发确认、folds3–4 最终 confirm。

### Confirm 隔离

- B 阶段可以生成 folds3–4 baseline，但 C 阶段解锁前不得查看/使用其指标；
- R3 内部 confirm 与项目长期 sealed/Confirm/OOD 严格区分；
- 后者仍然不得打开。

---

## 8. WBS

| WP | 任务 | 产出 |
|---|---|---|
| **R3-0** | 冻结旧证据 | tag、53-run manifest、protocol |
| **R3-1** | 数据身份与 S1 | canonical/branch/sample、d0、split 硬门 |
| **R3-2** | 单帧/交换不变特征 | F0–F3 schema、白名单、对称性测试 |
| **R3-3** | Loss v2 与训练器 | 权重归一、pair mask、λ=0 回归 |
| **R3-4** | 特征消融 + B0-F | F 阶梯、五折基线 |
| **R3-5** | 梯度校准 | encoder/head ρ |
| **R3-6** | Loss screen + dev | 四辅助臂、Top-2 |
| **R3-7** | Ramp 诊断 + Confirm | 预注册准入 |
| **R3-8** | Gate + 冻结报告 | 五终局、approved=false |

R3-2 与 R3-3 可在协议冻结后并行编码；正式训练必须先过 R3-1。

---

## 9. 验收

| Gate | 条件 |
|---|---|
| **Data** | canonical/branch/sample 计数自洽；d0 语义明确；无阻断性 P0；无确认跨折泄漏 |
| **Feature** | 全部 forward 字段单帧合法；交换不变；旋转/平移/置换通过；无诊断泄漏 |
| **Baseline** | B0-F 独立完成；特征配置与 checkpoint hash 对应；folds3–4 隔离 |
| **Loss** | λ=0 与 B0-F 精确一致；权重只施加一次；空 bucket/非有限值正确 |
| **Gradient** | 2%/10% 达到 encoder 容差；head ≤0.3；λ 不越界 |
| **Dev** | Top-2 预注册筛选，指标和保护条件同时满足 |
| **Confirm** | 仅 dev-positive；两折改善、RMSE、sign/dead、bootstrap 全部合格 |
| **Final** | 五终局明确；V5.1b 不自动批准 |

### 五种终局优先级

1. `data_blocked`：数据 P0 或 split 泄漏；
2. `feature_blocked`：特征不合法/不对称/泄漏；
3. `scale_unreachable`：λ 上限下 ρ<1%；
4. `no_gain`：ρ 达标但指标无正向；
5. `scale_controlled`：ρ 达标 + dev 正向 + **confirm 通过**。

`scale_controlled` 不能仅凭 dev-positive 宣告。

`frozen_candidate_r3.json`：

```json
{"approved_for_v5_1b": false}
```

---

## 10. 禁止事项

- 把 branch 加回输入；
- 使用真实端点几何 \(X_A/X_B\) 或由其计算的 \(d_A/d_B/s\)；
- 使用 `B_A-B_B` 作为无符号补偿的正式特征；
- 使用 `d_eff/terminal/capped/near_ts/key_progress` 作为输入；
- 同时改 split、特征、损失；
- 复用旧 checkpoint；
- 用 confirm 折调 λ/ramp/checkpoint；
- 未做消融就加 hop/charge/aromatic/angle；
- 打开 sealed/OOD/Fit150+；
- 用旧 53-run 作为 R3 gate。

---

## 11. 交付物

```text
outputs/pesn/v5_1_r3/
  protocol_r3.json
  data/canonical_irc.csv
  data/branch_diagnostics.csv
  data/sample_view.csv
  data/split_manifest_r3.json
  data/data_manifest.json
  features/feature_spec_v2/
  features/feature_manifest.json
  gradients/gradient_probe_r3.json
  screen/screen_summary.csv
  dev/dev_summary.csv
  ramp/ramp_summary.csv
  confirm/confirm_summary.csv
  reports/gate_report_r3.md
  reports/gate_report_r3.json
  reports/decision_memo.md
  frozen_candidate_r3.json
```

---

## 12. 目标命令序列

```bash
# 0) 冻结旧证据
git tag v5.1a-r1 <commit>
git checkout -b v5.1-r3

# 1) 数据合同（只读建表）
python scripts/campaigns/r3_build_data.py \
  --d0 0.08 \
  --out outputs/pesn/v5_1_r3/data

# 2) 特征 v2
python scripts/campaigns/r3_build_features.py \
  --data outputs/pesn/v5_1_r3/data/data_manifest.json \
  --feature-level F2 \
  --d0 0.08 \
  --out outputs/pesn/v5_1_r3/features/feature_spec_v2

# 3) B0-F fold0
python scripts/campaigns/run_v5_1_r3.py \
  --stage baseline --folds 0 --seeds 17 29 43 --dry-run

# 4) 梯度探针
python scripts/campaigns/r3_gradient_probe.py \
  --checkpoint outputs/pesn/v5_1_r3/runs/B0F_f0_s17/checkpoints/checkpoint_best.pt \
  --batches 20 --targets 0.02 0.10 \
  --out outputs/pesn/v5_1_r3/gradients/gradient_probe_r3.json

# 5) screen
python scripts/campaigns/run_v5_1_r3.py --stage screen \
  --plan outputs/pesn/v5_1_r3/plans/screen.json

# 6) dev + ramp
python scripts/campaigns/run_v5_1_r3.py --stage dev \
  --plan outputs/pesn/v5_1_r3/plans/dev.json
python scripts/campaigns/run_v5_1_r3.py --stage ramp \
  --plan outputs/pesn/v5_1_r3/plans/ramp.json

# 7) confirm（仅 dev-positive）
python scripts/campaigns/run_v5_1_r3.py --stage confirm \
  --arm <top1> --plan outputs/pesn/v5_1_r3/plans/confirm.json

# 8) 报告
python scripts/campaigns/r3_report.py \
  --root outputs/pesn/v5_1_r3 \
  --out outputs/pesn/v5_1_r3/reports
```

---

## 13. 一句话

**R3 的第一目标不是降 SRE，而是证明：IRC 教师动作可以在不依赖 branch、不依赖历史/端点几何、仅用当前单帧 + 无序拓扑的条件下被学习；只有 H1/H2 成立，strong/dead 的 H3 才有意义。**
