# PES2TS V5.1-R3：数据库 + 特征 + 损失三位一体修复方案 v1

> **v1.1 监督修订（2026-10-20）**：R3 固定为 `intent_directed_irc_step`；一条完整 IRC、两种有向 intent、first-contact 单步；TS 不是 R3 停止目标。样本主键加入 `intent_id`；forward-only 为主实验，bidirectional 为等预算增强臂。
>
> 状态：**Proposed v1.1**（V5.1 阶段执行方案；先于任何 R3 训练冻结）
> 日期：2026-10-20
> 核心合同：[ReactionIntent Contract v1](../contracts/PES2TS_ReactionIntent_Contract_v1.md)
> 阶段计划：[V5–V7 v3.0](PES2TS_V5-V7_反应意图条件化与低维生长方案_v3.0_20261020.md)
> R3 方案：[V5.1-R3 v3](PES2TS_V5.1-R3_反应意图条件化与端点标签不变重训方案_v3_20261020.md)

---

## 0. 一页结论

V5.1-R3 只做三件事：

1. **数据层重构**：把 branch 组织改成曲线中心；建立 `irc_curve / irc_node / reaction_intent / sample_view`；解决 d0、gauge、冲突、split。
2. **特征层重构**：删除 branch；改成 **reaction-intent-conditioned, endpoint-label-invariant**；允许合法 signed `ΔB_intent`；建立 F0–F3 最小充分集。
3. **损失层重构**：以 B0-F 为基线；加入无量纲 strong/dead；统一 sample weight/reduction；用完整 B0-F 梯度做 λ 校准。

R3 不实现：

- CV/Jacobian 几何提升；
- Progress Head；
- Stop Head；
- 完整曲线生长。

这些是 V5.2+。

---

## 1. 范围与非目标

### 1.1 范围

- 固定主步长 `d0=0.08 Å`；
- 50 个 fit 反应，49 个有效簇；
- canonical IRC 曲线；
- Reaction Intent（A→B / B→A 两种有向 intent）；
- `intent_directed_irc_step`：沿完整曲线 first-contact 的单步 key-atom 位移；
- B0-F 重训；
- strong/dead 重新校准；
- 66-run 矩阵；
- 五终局。

### 1.2 非目标

- 不实现 V5.2 新架构/Stop Head；
- 不做正式 OOF；
- 不打开 sealed/Confirm/OOD；
- 不启动 Fit150+；
- 不把 branch 加回输入；
- 不使用端点几何/TS 坐标/progress 作为 forward；
- 不在 R3 主矩阵同时改 split、特征、损失；
- 不用 confirm 调参。

---

## 2. 当前基线

| 项目 | 数量/状态 |
|---|---:|
| reactions | 50 |
| branches（旧组织） | 100 |
| teacher rows | 5568 |
| eligible | 5088 |
| gauge_unresolved | 480 |
| terminal | 137 |
| contact | 4951 |
| capped | 0 |
| effective clusters | 49 |
| RXN_0000001248 | 0 eligible，排除 |
| mapping overflow | 5 对：238/1089/1174/1504/2226 |
| 历史 342 对冲突 | 需与 fit50 求交集 |
| d0 | 仓库冲突：MAIN_D0=0.08，V5.1 配置/特征=0.05 |

R3 先解决 d0 冲突，再谈模型。

---

## 3. 数据层重构

### 3.1 目标目录

```text
outputs/pesn/v5_1_r3/
  data/
    irc_curve.csv
    irc_node.csv
    irc_edge.csv
    reaction_intent.csv
    curve_quality.csv
    sample_view.csv
    split_manifest_r3.json
    data_manifest.json
```

### 3.2 表结构

#### irc_curve

```text
curve_id
canonical_irc_id
source_irc_hash
canonical_mapping_sha
ts_geometry_sha
ts_raw_hash
method
charge
multiplicity
d0
n_nodes
ts_index
quality_status
```

约束：

- `canonical_irc_id` 唯一；
- `ts_raw_hash` 仅溯源，不作为唯一去重；
- 同一曲线所有节点共享 `curve_id`；
- 不允许 branch 字段进入模型。

#### irc_node

```text
curve_id
node_index
xyz
graph
energy
gradient
progress
is_ts
is_endpoint
gauge_status
quality_status
```

主键：`(curve_id, node_index)`。

#### irc_edge

```text
curve_id
i
j
delta_xyz
delta_progress
direction
quality
```

主键：`(curve_id, i, j, direction)`。

#### reaction_intent

```text
intent_id
curve_id
G_src
G_tgt
K
CV_spec
rules
delta_B_intent
source_metadata
intent_version
```

规则：

- `G_src/G_tgt` 由反应任务定义；
- `delta_B_intent = B_tgt - B_src`；
- 不由 branch 定义；
- Reverse 时 source/target 交换；
- Relabel 时不变。

#### curve_quality

```text
curve_id
n_contact
n_terminal
n_capped
n_gauge
n_other
max_step_key_rmsd
last_step_key_rmsd
cum_path_key_len
frame_layer
gate_ok
short_curve
coarse_step
```

`n_capped` 是独立诊断，不计入互斥 outcome。

#### sample_view

```text
sample_id = (curve_id, node_index, intent_id, d0)
fold
label_status
disposition
teacher_valid
target_first_contact (node/alpha)
target_action u
target_progress q
sampling_group_id
loss_eligible
label_weight
curve_weight
sample_weight
feature_sha
label_sha
```

规则：

- forward/reverse 共享同一 curve、同一 fold；
- 反向样本不是独立 IRC；
- 一条曲线总采样权重固定，按 intent/node 分摊；
- forward-only 与 bidirectional 使用相同曲线预算/优化步数/模型容量。

`label_status`：

```text
ok_contact / ok_terminal / gauge_unresolved /
no_future / reference_invalid / conflict / suspect
```

`disposition`：

```text
train_hard / train_soft / eval_only / exclude
```

### 3.2.1 Teacher 版本：intent-directed first contact

R3 的 teacher 不是相邻帧位移，而是沿当前 intent 方向 first-contact：

\[
j^\pm=\mathrm{first\_contact\_along\_curve}(i,\pm1,d_0=0.08)
\]

\[
u_i^\pm=X_{j^\pm,K}-X_{i,K}
\]

- 目标可能是插值点，不一定是离散相邻帧；
- forward A→B 与 reverse B→A 都从同一条曲线重新计算；
- 无 target 的端点方向标 `terminal/no_target`，不伪造动作；
- `teacher_version = intent_directed_first_contact_v1`；
- 旧 0.05/ts-directed 标签不可混用；
- protocol_r3、教师版本、数据哈希必须在训练前重新冻结。

### 3.3 d0 决策

R3 推荐：

```text
d0 = 0.08 Å（canonical main）
0.05 / 0.12 仅敏感性
```

理由：

- `key_atom_teacher.py` 的 `MAIN_D0=0.08`；
- V5 计划主 d0=0.08；
- R3 已因特征变化无法复用旧 baseline，不如同时修正 d0 语义。

如果团队选择 0.05：

- 必须在协议中写明“R3 不是 canonical main”；
- 不能与 0.08 结果混比；
- 2022 结论按对应 d0 重新复核。

### 3.4 P0 处置

| 问题 | 处置 |
|---|---|
| RXN_0000001248 全 gauge | `excluded_gauge_unresolved`；49 个有效簇 |
| 2022 全 contact | 按 d0=0.08 复核；用 `teacher_outcome`，不用 0.05 结论 |
| 5 个 mapping overflow | 核查 `pair_edit` 与 R/P 键变化；不一致则 quarantine |
| 342 对历史冲突 | 与 fit50 求交集；空则 `R3_scope_not_affected`，不声称历史问题已解决 |
| 3 个缺失 family label | `MISSING_FAMILY`，只影响分层 |
| 480 gauge | 训练排除；评估双分母 + coverage |

### 3.5 split

主实验暂不改 split：

- 使用冻结 `split_manifest_v5`；
- 另建 `split_manifest_v5_1_shadow` 做泄漏敏感性；
- 同一 `curve_id` 所有节点/d0/方向同折；
- 同一 near-dup component 不跨训练/评估；
- 发现确认泄漏 → `data_blocked`；
- 如果切换 split，所有 baseline 重跑。

### 3.6 数据门

- canonical/curve/node/intent/sample 计数自洽；
- 无未关闭 P0 >5%；
- 1248 明确排除；
- 5 个 mapping overflow 有处置；
- 342 交集闭环；
- d0 语义唯一；
- 无 branch 进入模型；
- 所有文件有 sha256。

---

## 4. 特征层重构

### 4.1 合法输入

\[
\boxed{
\text{forward 只能使用 }X_t\text{ 与合法的 }G_{\rm rxn}
}
\]

允许：

- 当前几何；
- 当前拓扑；
- `G_src/G_tgt` 图；
- `K`；
- CV spec/rules；
- 当前 CV 值。

禁止：

- 历史/未来 IRC 帧；
- TS 坐标；
- 端点三维几何；
- 由端点几何计算的 d/s；
- 真实 progress / near_ts / d_eff / terminal / capped；
- branch ID；
- split/family/conflict class。

### 4.2 四个通道

| 通道 | 内容 | 对称性 |
|---|---|---|
| Chemical Context | 元素、当前几何、无序拓扑 | 端点重命名不变 |
| Reaction Intent | signed `ΔB_intent`、目标 CV 增量 | 随真实反转改变 |
| Geometric State | 当前键长/角/关键原子位置 | 只依赖当前构型 |
| Teacher Diagnostics | IRC 位置、TS、未来帧 | 禁止进入 forward |

### 4.3 正式 F2 特征

#### Node

```text
z_atomic        nn.Embedding(119,16)
is_key          1
degK            1
intent_role     3   # breaking / forming / spectator
```

#### Edge

```text
is_spatial          1
B_src               1
B_tgt               1
delta_B_intent      1
abs_delta_B         1
is_changed          1
```

#### Vector

```text
pos
```

#### Task

```text
无（d0 固定 0.08）
```

### 4.4 特征阶梯

| 级别 | 开放特征 |
|---|---|
| F0 | 元素 + `pos` + 基础拓扑 |
| F1 | + `is_key` + `degK` |
| F2 | + `B_src/B_tgt/delta_B_intent/abs_delta/is_changed/intent_role` |
| F3 | + RBF(12) |

固定 F3 superset 形状 + feature mask；同一 seed 相同网络形状/初始化/采样顺序。

Intent Ablation：

```text
F2-intent-on
F2-intent-off（delta_B_intent/intent_role 置零）
```

### 4.5 对称性测试

#### Relabel

交换端点名字，`G_src/G_tgt` 不变：

\[
\pi(X,\mathrm{Relabel}(G_{\rm rxn}))
=
\pi(X,G_{\rm rxn})
\]

要求特征逐比特相同，`max|Δu| < 1e-6`。

#### Reverse

交换 source/target：

\[
\pi(X,\mathrm{Reverse}(G_{\rm rxn}))
\ne
\pi(X,G_{\rm rxn})
\]

`delta_B_intent` 变号；输出应改变；小步长切向近似相反，不作硬约束。

#### 其他

- 旋转/平移等变；
- 原子重编号等变；
- 镜像行为说明；
- median/P99/max；
- near-zero 单独统计。

### 4.6 feature_manifest

```json
{
  "schema": "pesn_v5_r3_feature_manifest_v1",
  "feature_spec": "v3",
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

训练 run 记录 `feature_manifest_sha256`。

### 4.7 特征门

- Relabel invariant；
- Reverse sensitive；
- 旋转/平移/置换通过；
- 无诊断字段泄漏；
- 指针/形状/finite；
- 重建两次 hash 一致；
- intent ablation 完成。

---

## 5. 损失层重构

### 5.1 B0-F

\[
L_{\rm base}=L_{\rm atom}+0.5L_{\rm pair}+0.02L_{\rm cos}
\]

- `L_atom`：主位移；
- `L_pair`：`legacy_j`；
- `L_cos`：教师幅度门控；
- `lambda_s=lambda_d=0`；
- 无 magnitude head，无 legacy sign/dead。

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

### 5.3 Strong / Dead

先在 frame 内 bucket 均值：

\[
L_{k,b}
=
\frac{1}{|K_{k,b}|}
\sum_{p\in K_{k,b}}\ell_{k,p}
\]

再按 `w_b` 聚合：

\[
L_k
=
\frac{\sum_b w_b \mathbf1[|K_{k,b}|>0]L_{k,b}}
{\sum_b w_b \mathbf1[|K_{k,b}|>0]}
\]

Strong：

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

Dead：

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
L=L_{\rm base}+\lambda_sL_{\rm strong}+\lambda_dL_{\rm dead}
\]

第一轮不同时开 strong/dead。

### 5.5 权重

```text
sample_weight = label_weight × curve_weight
```

所有损失项使用同一 `w_b`；只乘一次；不重复乘 q。

### 5.6 梯度

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

### 5.7 损失门

- λ=0 与 B0-F 逐比特一致；
- 空桶/边界/非有限值；
- 权重只一次；
- 梯度 2%/10% 或 unreachable；
- head ≤0.3。

---

## 6. 代码改动

### 6.1 新增

```text
pes2ts_core/pesn/contracts/reaction_intent.py
pes2ts_core/pesn/features/key_atom_spec_v3.py
pes2ts_core/pesn/features/key_atom_graph_v3.py
pes2ts_core/pesn/offline/data/key_atom_dataset_v3.py
pes2ts_core/pesn/models/key_atom_egnn_v3.py
pes2ts_core/pesn/offline/training/key_atom_bc_v3.py
scripts/campaigns/r3_build_data.py
scripts/campaigns/r3_build_features.py
scripts/campaigns/run_v5_1_r3.py
scripts/campaigns/r3_gradient_probe.py
scripts/campaigns/r3_report.py
```

### 6.2 测试

```text
tests/pesn/test_r3_data_contract.py
tests/pesn/test_key_atom_features_r3.py
tests/pesn/test_r3_relabel_invariance.py
tests/pesn/test_r3_reverse_sensitivity.py
tests/pesn/test_r3_input_legality.py
tests/pesn/test_key_atom_losses_r3.py
tests/pesn/test_r3_driver_plan.py
```

### 6.3 原则

- v1/v2 模块只读；
- v3 新建文件，不原地改 schema；
- 训练器拒绝 branch/端点几何/TS/progress 进入 forward；
- 训练器拒绝 feature manifest 不匹配；
- shard 写入前检查指针/形状/finite；
- 所有随机性有 seed。

---

## 7. 实验矩阵

| 阶段 | 内容 | 新 run |
|---|---|---:|
| F-main | F0/F1/F2/F3，forward-only，fold0 × 3 | 12 |
| F-bidir | F2，A→B + B→A，等曲线预算，fold0 × 3 | 3 |
| F-ablate | F2，屏蔽 signed intent，fold0 × 3 | 3 |
| B | 选中 F 的 B0-F，补 folds1–4 × 3 | 12 |
| L | S2/S10/D2/D10，fold0 × 3 | 12 |
| V | Top-2，folds1–2 × 3 | 12 |
| R | 最佳 loss × R1/R2，fold0 × 3；R0 复用 | 6 |
| C | Top-1，folds3–4 × 3 | 6 |
| 合计 | | **66** |

规则：

- 主实验 = forward-only；
- bidirectional 是增强臂，不假设更好；
- intent-off 用于验证 signed intent 必要性；
- 反向样本与正向共享同一 IRC、同一 fold，不能当独立 curve 计权。

预算：

```text
GPU ~160 min
总预算 ≤900 min
```

先做 fold0 门：

```text
B0-F fold0 × 3
+ intent ablation × 3
+ S2/S10/D2/D10 fold0 × 3
```

约 18 run。

---

## 8. WBS

| WP | 任务 | 产出 |
|---|---|---|
| R3-0 | 冻结旧证据 | tag、manifest |
| R3-1 | 数据层 | 五表 + split |
| R3-2 | 特征 v3 | spec + shard + manifest |
| R3-3 | 特征阶梯 | F0–F3 + intent ablation |
| R3-4 | B0-F | 五折基线 |
| R3-5 | 梯度 | probe + λ |
| R3-6 | loss screen + dev | screen/dev |
| R3-7 | ramp + confirm | ramp/confirm |
| R3-8 | 报告 | gate/frozen |

---

## 9. 验收

| Gate | 条件 |
|---|---|
| Data | 计数自洽；d0=0.08；无 P0；无跨折泄漏 |
| Feature | Relabel invariant；Reverse sensitive；合法输入；无泄漏 |
| Baseline | B0-F 独立完成；folds3–4 隔离 |
| Loss | λ=0 逐比特；权重一次；空桶正确 |
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

## 10. 禁止事项

- branch 回输入；
- 单独 direction token；
- 端点几何/TS/progress 进 forward；
- 把 Reverse 当 Relabel；
- 同时改 split/特征/损失；
- 复用旧 checkpoint；
- 用 confirm 调参；
- 未验证就称 TS/IRC；
- 打开 sealed/OOD/Fit150+。

---

## 11. 交付物

```text
outputs/pesn/v5_1_r3/
  protocol_r3.json
  data/irc_curve.csv
  data/irc_node.csv
  data/irc_edge.csv
  data/reaction_intent.csv
  data/curve_quality.csv
  data/sample_view.csv
  data/split_manifest_r3.json
  data/data_manifest.json
  features/feature_spec_v3/
  features/feature_manifest.json
  gradients/gradient_probe_r3.json
  screen/screen_summary.csv
  dev/dev_summary.csv
  ramp/ramp_summary.csv
  confirm/confirm_summary.csv
  reports/gate_report_r3.md/json
  reports/decision_memo.md
  frozen_candidate_r3.json
```

---

## 12. 一句话

**V5.1-R3 的修复顺序是：先把数据变成“曲线 + 反应意图 + 样本状态”，再把特征变成“合法输入 + signed intent + 最小充分集”，最后把损失变成“B0-F + 无量纲 strong/dead + 权重/梯度纪律”。三者必须按顺序过门，不能同时改。**
