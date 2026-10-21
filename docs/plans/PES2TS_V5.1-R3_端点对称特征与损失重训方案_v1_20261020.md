# PES2TS / PES-N V5.1-R3：端点对称特征与损失重训方案 v1

> **Superseded（2026-10-20）**：已被 [R3 v3](PES2TS_V5.1-R3_反应意图条件化与端点标签不变重训方案_v3_20261020.md) 取代；请勿按本文件的 endpoint-symmetric 语义执行。
>
> 状态：**Proposed v1 / Superseded by R3 v3**（可执行研究方案；先于任何 R3 训练冻结）
> 日期：2026-10-20
> 层级：G3.P.2（训练/损失/评测），支持 G3.P.0/.1/.3
> 关系：**取代 V5.1-R2 的“特征不变”假设**；R2 的 strong/dead 公式保留，但 baseline、λ、checkpoint、梯度探针全部重来
> 上游证据：`outputs/pesn/v5_1_r1/final_report.md`、`outputs/pesn/v5_1_r1/protocol_r1.json`
> 下游：V5.2 新 Key-Atom 合同、V5.3 纯几何 EGNN、V6.1 物理 MLP

---

## 0. 一页结论

V5.1-R2 的前提是“特征不变，只修损失尺度”。但当前数据审计发现：

1. `branch0/branch1` 是**文件槽位**，不是化学方向：50 个 task 中 25 个 `['P','R']`、25 个 `['R','P']`；
2. `b_start/b_other/db`、`anchor_vec`、`endpoints[branch]` 都随 branch 翻转，同一个物理帧会得到不同符号的输入，而目标位移不变；
3. 20 个 node 槽中约 9 个是死特征或强冗余；29 个 edge 槽中约 6 个是死特征或确定性重复；19 个 task 槽中 17 个在当前数据上是常数；
4. K 边只占 5.94%，mean aggregation 会稀释关键键变化信号。

因此 R3 的目标不是“再调一次 loss”，而是：

> **先把特征改成 endpoint-swap invariant 的最小充分集，重训 B0-F；再在 B0-F 上校准无量纲 strong/dead 损失。**

R3 仍然不打开 sealed/Confirm/OOD，不启动 Fit150+；V5.1b 继续关闭。

---

## 1. 范围与非目标

### 1.1 范围

- 保留 V5.1 的任务、数据口径、5-fold split、key-atom 标签定义；
- 修复数据身份与样本状态：canonical IRC、branch diagnostics、sample view；
- 重写特征：去掉 branch，改成端点对称；
- 重写损失：保留主损失，加入无量纲 strong/dead；
- 重训 baseline：B0-F 从零开始，旧 checkpoint 不可用；
- 重新校准 λ、ramp、checkpoint；
- 给出五种终局之一。

### 1.2 非目标

- 不实现 V5.2 新网络、Stop Head、V6 xTB、V7 RL；
- 不做正式 OOF，不打开 sealed/Confirm/OOD，不启动 Fit150+；
- 不在 R3 主实验里同时改 split、特征、损失三个变量；
- 不继续旧 `{0.03, 0.1, 0.3}` λ 网格；
- 不使用 confirm 折回头调参；
- 不把 branch 重新加回任何输入。

---

## 2. 与 V5.1-R2 的关系

| 项目 | V5.1-R2 | V5.1-R3 |
|---|---|---|
| 特征 | 保持 v1 | 端点对称 v2，去掉 branch |
| baseline | 复用旧 B0 | 从零训练 B0-F |
| 旧 53-run | 可直接对比 | 仅历史参考，不作 gate |
| strong/dead 公式 | 无量纲 | 保留同一公式 |
| λ | 在旧 B0 上校准 | 在 B0-F 上重新校准 |
| checkpoint | 旧规则 | 同一规则，但新 baseline |
| 结论 | 旧任务损失修复 | 特征 + 损失联合重训 |

**关键：R3 不是 R2 的补跑，而是新数据/特征合同下的新 campaign。**

---

## 3. 数据合同规范化

### 3.1 目标结构

```text
outputs/pesn/v5_1_r3/
  data/
    canonical_irc.csv
    branch_diagnostics.csv
    sample_view.csv
    split_manifest_r3.json
    data_manifest.json
  features/
    feature_spec_v2/
      ...
    feature_manifest.json
  runs/
  protocol_r3.json
  reports/
```

### 3.2 canonical_irc.csv

一行一条 canonical IRC：

| 字段 | 定义 |
|---|---|
| `canonical_irc_id` | `sha256(canon_graph_A + canon_graph_B + sorted_element_labeled_key_distance_spectrum + charge + multiplicity)` |
| `snapshot_ids` | 所有来源文件 sha256 |
| `reaction_ids` | 映射到该 IRC 的原始 RXN 列表 |
| `ts_raw_hash` | TS 坐标快照哈希（仅溯源） |
| `ts_branch_max_diff` | branch0/branch1 最后帧最大差；当前应为 0 |
| `key_rows` | 关键原子行 |
| `n_branches` | 2 |
| `n_frames` | 两分支总帧数 |
| `n_eligible` | 可用帧 |
| `n_gauge` | `reason_id=3` 帧 |
| `endpoint_match_mode` | `pair_set` / `pair_order_signature` / `graph_ambiguous_fallback` |
| `mapping_status` | `ok` / `overflow` |
| `near_dup_component_id` | 经验近重复连通分量 |
| `family_id` | 仅分层，不合并身份 |
| `fold` | 当前 split 的 fold |

**规则：**

- `canonical_irc_id` 唯一；
- `reaction_ids` 长度 >1 → `EXACT_IRC_DUPLICATE`；
- 同一个 `reaction_id` 出现多个 `canonical_irc_id` → 一反应多 IRC，标记 P0；
- `branch0/branch1` 共享 TS，不合并两半。

### 3.3 branch_diagnostics.csv

一行一个 branch：

```text
branch_id, irc_id, side, n_frames, n_teacher_rows,
n_eligible, n_gauge, n_terminal, n_contact, n_capped,
max_step_key_rmsd, last_step_key_rmsd, cum_path_key_len,
frame_layer, gate_ok, endpoint_match_mode,
short_branch, coarse_step, branch_weight
```

硬不变量：

```text
n_teacher_rows = n_frames - 1
n_eligible + n_gauge + n_other = n_teacher_rows
n_contact + n_terminal + n_capped = n_eligible
branch0.TS == branch1.TS
branch0.key_rows == branch1.key_rows
```

当前账本：

```text
50 reactions, 100 branches, 5568 teacher rows,
5088 eligible, 480 gauge, 137 terminal, 4951 contact, 0 capped
49 effective clusters; RXN_0000001248 has 0 eligible
3 short branches (<20 frames): 1675.b0=8, 2309.b0=15, 2309.b1=18
17 all-contact branches; 2022 last step 0.067 Å > d0=0.05
```

### 3.4 sample_view.csv

一行一个 `(branch, start_row)`：

```text
sample_id, irc_id, branch_id, start_row, fold,
label_status, disposition, label_weight, branch_weight,
sample_weight, feature_sha, label_sha
```

`label_status`：

```text
ok_contact
ok_terminal
gauge_unresolved
no_future
reference_invalid
conflict
suspect
```

`disposition`：

```text
train_hard
train_soft
eval_only
exclude
```

规则：

- 每个 sample 有且只有一个 status 和一个 disposition；
- gauge 不进训练，保留在评估 coverage；
- 不删除困难样本；
- 1248 明确 `excluded_gauge_unresolved`；
- 5 个 mapping overflow 先进 `quarantine`，不进第一轮 baseline。

### 3.5 P0 处置

| 问题 | 处置 |
|---|---|
| RXN_0000001248 全 gauge | `excluded_gauge_unresolved`；fold0 按 9 簇报告 |
| 2022 全 contact | `capped=0` + last-step 0.067 > 0.05 → `valid_contact`，解除 suspect |
| 5 个 mapping overflow（238/1089/1174/1504/2226） | 核查 `pair_edit` 与 R/P 键变化；不一致则 `quarantine` |
| 342 对历史冲突 | 与 fit50 求交集；空则写 `historical_342_overlap_count: 0`；非空则 342/342 分类 |
| 3 个缺失 family label | 标记 `MISSING_FAMILY`，只影响分层，不合并身份 |

### 3.6 split 策略

R3 主实验**暂不改 split**：

- 使用冻结的 `split_manifest_v5`，保证 R3 内部各臂可比；
- 另建 `split_manifest_v5_1_shadow`，按 `near_dup_component_id` 重分折，仅作敏感性分析；
- 如果 shadow split 显示 >20% 簇换折，或存在同 key 元素集合、同键事件、key-TS RMSD <0.25 Å 的跨折边 → `data_blocked`，先决定是否切换；
- 如果切换 split，则 R3 所有 baseline 必须重跑，旧 53-run 全部失效。

**注意：当前快速审计在 key 元素可匹配的反应对中未发现 key-TS RMSD <0.35 Å 的对；13 个 family 跨折，但 family 内 key 元素组成不同。family 不是 identity，不能按 family 合并。**

---

## 4. 特征合同 v2

### 4.1 端点对称原则

A–TS–B 没有内在正向。交换 A/B：

```text
端点图交换
B_A <-> B_B
ΔB -> -ΔB
s -> -s
s·ΔB -> 不变
```

目标位移 \(u\) 不变。

因此：

- 不使用 `branch0/branch1`；
- 不使用 `b_start/b_other/db`；
- 不使用单侧 `anchor_vec`；
- 所有端点特征必须 endpoint-swap invariant。

### 4.2 最小特征集

#### Node

```text
z_atomic    16 维 element embedding
is_key      1
degK        1
s           1（graph-level 广播，可选）
```

维度：18–19。

#### Edge

```text
is_spatial        1
is_covalent_A     1
is_covalent_B     1
B_A               1
B_B               1
changed           1
abs_dB            1
s_dB              1
RBF(distance)     12
```

维度：20。

#### Vector

```text
pos           当前坐标
X_A, X_B      可选，双端点几何，对称使用
```

#### Task

```text
d0            1（只有 d0 变化时才需要）
```

#### 几何侧变量

\[
d_A = \mathrm{RMSD}_{key}(X_t, X_A), \quad
d_B = \mathrm{RMSD}_{key}(X_t, X_B)
\]

\[
s = \frac{d_B - d_A}{d_A + d_B + \epsilon}
\]

\[
\Delta B = B_B - B_A
\]

\[
\text{signed change} = s \cdot \Delta B
\]

### 4.3 图结构

```text
graph = G_A ∪ G_B ∪ K ∪ {当前距离 < 4.5 Å 的空间邻居}
```

- 共价边和 K 边永远保留；
- 空间边最多 32 邻居；
- 每条无向边展开为双方向；
- K 边必须两端都是 key 原子。

### 4.4 特征阶梯

| 级别 | 特征 | 目的 |
|---|---|---|
| **F0** | 元素 + `pos` + `edge_index` + `is_spatial/is_covalent_A/is_covalent_B` | 纯 EGNN 核 |
| **F1** | + `is_key` + `degK` | 输出支撑 |
| **F2** | + `B_A/B_B/changed/abs_dB/s_dB` + `s` | 图论约束 + 几何侧 |
| **F3** | + RBF(12) | 距离编码 |
| **F4** | + `X_A/X_B` 或 `d_A/d_B`（可选） | 当前侧判断 |
| **F5** | + hop 标量/charge/aromatic（按需） | 只在有证据时加 |

每加一级，fold0 × 3 seeds，用 B0 loss；没有增益就停。

### 4.5 交换测试

对每个验证样本构造 swap 视图：

```text
A <-> B
B_A <-> B_B
s -> -s
s_dB -> -s_dB
```

模型输出应不变：

\[
C_{\rm swap}
=
\mathrm{median}
\frac{\|\hat u(x)-\hat u(\mathrm{swap}(x))\|}
{\|\hat u(x)\|+\epsilon}
< 10^{-3}
\]

不通过 → `feature_blocked`。

### 4.6 feature_manifest.json

```json
{
  "schema": "pesn_v5_r3_feature_manifest_v1",
  "feature_spec": "v2",
  "feature_spec_sha": "...",
  "data_manifest_sha": "...",
  "f0_level": "F2",
  "rbf_dim": 12,
  "use_endpoint_geometry": false,
  "n_samples": 5568,
  "n_train": 0,
  "n_eval_eligible": 5088,
  "shards": {"...": "sha256..."}
}
```

训练 run 必须记录 `feature_manifest_sha256`。

---

## 5. 损失函数

### 5.1 B0-F：特征基线

```text
L = L_atom + 0.5 L_pair + 0.02 L_cos
```

- `L_atom`：主位移损失，保持旧定义；
- `L_pair`：`legacy_j`，只对 `pair_edit != 0` 计算；
- `L_cos`：教师幅度门控的余弦损失；
- `lambda_s = lambda_d = 0`；
- 无 magnitude head，无 legacy sign/dead。

要求：λ=0 时与旧 B0 的**同特征版本**逐比特一致；不是与旧 v1 B0 一致。

### 5.2 主损失

\[
L_{\rm atom}
=
\frac{1}{B}\sum_b
\sum_i
w_{b,i}
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

### 5.3 Strong 项

\[
L_{\rm strong}
=
\frac{1}{|S|}
\sum_{p\in S}
\mathrm{Huber}_{0.25}
\left(
\frac{\hat{\Delta d}_p-\Delta d_p^*}
{\max(|\Delta d_p^*|,0.01)}
\right)
\]

- \(S = \{p: |\Delta d_p^*| \ge 0.02\}\)；
- \(\tau_s=0.01\)；
- bucket-mean；空桶为 0。

### 5.4 Dead 项

\[
L_{\rm dead}
=
\frac{1}{|D|}
\sum_{p\in D}
\left[
\frac{\max(0,|\hat{\Delta d}_p|-0.005)}{0.005}
\right]^2
\]

- \(D = \{p: |\Delta d_p^*| < 0.005\}\)；
- bucket-mean；空桶为 0。

### 5.5 总损失

\[
L =
L_{\rm atom}
+ 0.5 L_{\rm pair}
+ 0.02 L_{\rm cos}
+ \lambda_s L_{\rm strong}
+ \lambda_d L_{\rm dead}
\]

第一轮：

```text
lambda_mag = 0
lambda_sign = 0
lambda_dead_legacy = 0
pair_loss = legacy_j
```

不同时开 strong 和 dead；先单独筛选，再按预注册规则组合。

### 5.6 masks 与 weights

```text
frame_mask = disposition in {train_hard, train_soft}
             and label_status != gauge_unresolved
pair_mask  = pair_edit != 0
sample_weight = label_weight × branch_weight
```

`sample_weight` 只在 sampler 或 loss 中乘一次，不得重复乘 `q_weight`。

### 5.7 λ 校准

测量点：

```text
B0F_f0_s17/checkpoint_best
20 个真实 inner-train batch
```

定义：

\[
\rho_k =
\frac{\|\nabla_\theta(\lambda_k L_k)\|}
{\|\nabla_\theta L_{\rm atom}\|}
\]

反解：

\[
\lambda_k^{\rm target}
=
\rho_{\rm target}
\frac{\|\nabla L_{\rm atom}\|}
{\|\nabla L_k\|}
\]

约束：

- 2% 与 10% 两档；
- λ 上限 0.5；
- 上限下 ρ<1% → `scale_unreachable`；
- head ρ ≤0.3；
- 每 100 步复测一次。

### 5.8 不需要 swap loss

如果特征按第 4 节做成 endpoint-symmetric，swap 后输入本身不变；`C_swap` 只作为门禁指标，不作为损失项。不要用大权重 swap loss 掩盖特征不对称。

---

## 6. 代码修改

### 6.1 新增模块（保持 v1 只读）

| 文件 | 作用 |
|---|---|
| `pes2ts_core/pesn/contracts/key_atom_v2.py` | v2 槽位、维度、batch 字段 |
| `pes2ts_core/pesn/features/key_atom_spec_v2.py` | v2 特征 spec、RBF、尺度、feature manifest |
| `pes2ts_core/pesn/features/key_atom_graph_v2.py` | 端点对称特征构建 |
| `pes2ts_core/pesn/offline/data/key_atom_dataset_v2.py` | manifest-driven dataset、v2 shard 读取 |
| `pes2ts_core/pesn/models/key_atom_egnn_v2.py` | 可配置输入维度的 StarEGNN v2 |
| `pes2ts_core/pesn/offline/training/key_atom_bc_v2.py` | v2 loss、masks、sample weights |
| `scripts/campaigns/r3_build_data.py` | canonical IRC / branch diagnostics / sample view |
| `scripts/campaigns/r3_build_features.py` | 构建 v2 特征 shard + manifest |
| `scripts/campaigns/run_v5_1_r3.py` | 阶段驱动：F/B/L/V/R/C |
| `scripts/campaigns/r3_gradient_probe.py` | encoder/head ρ 与 λ 校准 |
| `scripts/campaigns/r3_report.py` | gate report / decision memo |
| `tests/pesn/test_key_atom_features_r3.py` | 特征槽位、端点对称、图结构 |
| `tests/pesn/test_key_atom_losses_r3.py` | λ=0 bit-identical、空桶、边界、非有限值 |
| `tests/pesn/test_r3_data_contract.py` | 主表计数、fold、gauge、1248 |
| `tests/pesn/test_r3_driver_plan.py` | 阶段/折/seed 白名单 |
| `tests/pesn/test_r3_swap_invariance.py` | `C_swap` 门禁 |

### 6.2 修改文件

| 文件 | 修改 |
|---|---|
| `pes2ts_core/pesn/contracts/key_atom.py` | 只加 v2 常量/别名，不改 v1 槽位 |
| `pes2ts_core/pesn/features/key_atom_graph.py` | 不改；v2 独立实现 |
| `pes2ts_core/pesn/models/key_atom_egnn.py` | 不改；v1 只读 |
| `pes2ts_core/pesn/offline/training/key_atom_bc.py` | 不改旧损失；v2 独立 |
| `config/campaigns/v5_1_r1.yaml` | 只读 |
| `AGENT.md` / `docs/README.md` | 增加 R3 入口 |
| `config/registry/g_v_map.yaml` | 登记 V5.1-R3 |

### 6.3 代码原则

- v1 模块冻结，保证 53-run 可复现；
- v2 通过新文件实现，不原地改 schema；
- 新旧模型/特征/损失通过 `feature_spec` / `loss_schema` 显式区分；
- 训练器拒绝 `feature_manifest_sha` 不匹配；
- 训练器拒绝 `branch` 字段进入 forward；
- 训练器拒绝 `d_eff/terminal/near_ts/capped/reason_id` 进入 forward；
- 所有随机性有 seed；
- 所有 shard 写入前做指针/形状/finite 检查。

---

## 7. 实验矩阵

### 7.1 阶段 F：特征阶梯

| 臂 | 特征 | 损失 | folds × seeds | 新 run |
|---|---|---|---|---:|
| B0-V1 | 旧 v1 | 旧 B0 | fold0 × 3 | 0（可复用） |
| B0-F2 | F2 | 旧 B0 | fold0 × 3 | 3 |
| B0-F3 | F3 | 旧 B0 | fold0 × 3 | 3 |
| B0-F4 | F4 | 旧 B0 | fold0 × 3 | 3 |

选最小不劣于 B0-V1 的 F。

### 7.2 阶段 B：B0-F baseline

| 臂 | folds × seeds | 新 run |
|---|---|---:|
| B0-F | folds0–4 × 3 | 15 |

先跑 fold0 × 3；通过后再补 folds1–4。

### 7.3 阶段 L：损失筛选

| 臂 | 损失 | folds × seeds | 新 run |
|---|---|---|---:|
| R0 | B0-F | fold0 × 3 | 复用 |
| S2 | + strong 2% | fold0 × 3 | 3 |
| S10 | + strong 10% | fold0 × 3 | 3 |
| D2 | + dead 2% | fold0 × 3 | 3 |
| D10 | + dead 10% | fold0 × 3 | 3 |

共 12 个新 run。

### 7.4 阶段 V：dev 复现

Top-2 × folds1–2 × 3 seeds = 12 个新 run。

### 7.5 阶段 R：ramp

最佳 loss × {R0,R1,R2} × fold0 × 3 = 9 cells，6 个新 run（R0 复用）。

### 7.6 阶段 C：confirm

Top-1 × folds3–4 × 3 = 6 个新 run。

### 7.7 预算

```text
F: 9
B: 15
L: 12
V: 12
R: 6
C: 6
-----
约 60 个新 run
```

单 run ~2.5 min → GPU 约 2.5 h；加特征构建、探针、分析，总预算 ≤900 min。

如果预算严格 ≤600 min，先做：

```text
B0-F fold0 × 3
+ S2/S10/D2/D10 fold0 × 3
```

共 15 个 run，约 40 min，先决定特征是否可用、损失是否可达。

---

## 8. WBS 与日程

| WP | 任务 | 产出 | 验收 |
|---|---|---|---|
| **R3-0** | 冻结旧 v1/53-run | tag、evidence manifest | 只读 |
| **R3-1** | 数据合同 | canonical_irc.csv、branch_diagnostics.csv、sample_view.csv | 计数/状态全对 |
| **R3-2** | 特征 v2 | feature_spec_v2、shards、manifest | swap 测试通过 |
| **R3-3** | 特征阶梯 | F2/F3/F4 fold0 × 3 | 选最小 F |
| **R3-4** | B0-F baseline | B0-F runs | fold0 可复现 |
| **R3-5** | 梯度探针 | gradient_probe_r3.json | 2%/10% λ 或 unreachable |
| **R3-6** | 损失筛选 | screen_summary.csv | 12 run 完整 |
| **R3-7** | dev + ramp | dev_summary.csv、ramp_summary.csv | 18 run 完整 |
| **R3-8** | confirm + 报告 | confirm_summary.csv、gate_report_r3 | 五终局之一 |

建议日程：

- Day 1：R3-0/1，数据合同 + feature spec；
- Day 2：R3-2/3，特征构建 + 阶梯；
- Day 3：R3-4/5，B0-F + 探针；
- Day 4：R3-6/7，screen/dev/ramp；
- Day 5：R3-8，confirm + 报告。

---

## 9. 验收与终局

### 9.1 数据门

- `canonical_irc.csv` 50 行，`canonical_irc_id` 唯一；
- `branch_diagnostics.csv` 100 行，`n_teacher_rows = n_frames - 1`；
- `sample_view.csv` 5568 行，每行有 status/disposition；
- 1248 明确排除；49 个有效簇；
- 5 个 mapping overflow 有处置；
- 342 对交集闭环；
- 2022 有复核结论；
- 无未关闭 P0 >5%。

### 9.2 特征门

- 无 branch/b_start/b_other/db/单侧 anchor；
- `C_swap < 1e-3`；
- 无诊断字段进入 forward；
- shard 指针/形状/finite 全通过；
- 重建两次 hash 一致。

### 9.3 损失门

- λ=0 与 B0-F 逐比特一致；
- strong/dead 空桶/边界/非有限值测试通过；
- 梯度探针 encoder/head 两组 ρ；
- 2%/10% λ 或 `scale_unreachable`；
- head ρ ≤0.3。

### 9.4 训练门

- B0-F 三 seed 可复现；
- screen 12 run 完整；
- dev 12 run 完整；
- ramp 6 run 完整；
- confirm 仅 dev-positive；
- 无止损触发。

### 9.5 五种终局

| 终局 | 条件 |
|---|---|
| **scale_controlled** | ρ 达标，dev 正向，`C_swap<1e-3`，无 sign/dead 退化 |
| **no_gain** | ρ 达标但指标无正向 |
| **scale_unreachable** | λ 上限下 ρ<1% |
| **data_blocked** | S1 未关闭 P0 或 split 敏感性失败 |
| **feature_blocked** | 特征不对称/泄漏/最小化门失败 |

`frozen_candidate_r3.json` 必须：

```json
{"approved_for_v5_1b": false}
```

---

## 10. 风险与禁止事项

### 禁止

- 把 branch 加回输入；
- 使用 `b_start/b_other/db` 或单侧 `anchor_vec`；
- 同时改 split、特征、损失；
- 复用旧 checkpoint 初始化 B0-F；
- 用 confirm 折调 λ/ramp/checkpoint；
- 未做消融就加 hop/charge/aromatic/angle；
- 打开 sealed/OOD/Fit150+；
- 用旧 53-run 作为 R3 gate。

### 风险

| 风险 | 处理 |
|---|---|
| 新特征太少，欠拟合 | F2→F3→F4 阶梯，按 fold0 证据加 |
| 端点几何不可得 | 只用 `s` + 端点图；F4 可选 |
| split 泄漏 | shadow split 审计；>20% 换折则 data_blocked |
| gauge 影响分母 | 双分母 + coverage |
| λ 不可达 | `scale_unreachable` |
| 短分支过采样 | `branch_weight` 或 sampler q，只乘一次 |
| 新 baseline 成本 | fold0-first gate，再扩 5 折 |

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
  --out outputs/pesn/v5_1_r3/data

# 2) 特征 v2
python scripts/campaigns/r3_build_features.py \
  --data outputs/pesn/v5_1_r3/data/data_manifest.json \
  --feature-level F2 \
  --out outputs/pesn/v5_1_r3/features/feature_spec_v2

# 3) B0-F fold0
python scripts/campaigns/run_v5_1_r3.py \
  --stage baseline --folds 0 --seeds 17 29 43 \
  --dry-run

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

**先修数据身份，再把特征改成端点对称的最小充分集，重训 B0-F；最后在 B0-F 上校准无量纲 strong/dead。旧 branch、旧 λ、旧 checkpoint、旧 53-run 都不再是 R3 的起点。**
