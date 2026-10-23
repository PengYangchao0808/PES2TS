# PES2TS V5.3-MP：大规模 MOVE–PROCESS 联合训练计划 v1.1

> 状态：**Proposed v1.1**（2026-10-21 用户评审后修订；v1 为有条件通过）
> 执行状态（2026-10-23）：S0/S1/S2/S3 训练全部完成——门 V53MP-S0 系列、S1-smoke、S2-scale、S3-formal 均通过（[S0](../reports/PES2TS_V53MP_S0_数据建库与资格报告_20261022.md) / [S1](../reports/PES2TS_V53MP_S1_2k冒烟报告_20261022.md) / [S2](../reports/PES2TS_V53MP_S2_5k规模报告_20261022.md) / [S3](../reports/PES2TS_V53MP_S3_20k主结果报告_20261023.md) 报告）；闭环门 V53MP-S3-closed-loop 单独报告；S4 另议
> 日期：2026-10-21 · 版本：v1.1
> 决策记录：[ADR-0031](../design/decisions/ADR-0031-MOVE-PROCESS统一学习合同.md)
> 唯一任务合同：[MoveProcessContract v1](../contracts/PES2TS_MoveProcessContract_v1.md)
> 上游：V5.2 已关闭（[D2-G 决策记录](../reports/PES2TS_V52R_输入充分性决策记录_20261021.md)）；V5.2-R D0/D0.1/D1/D2 证据保留
> 命名：**V5.3-MP** = 本计划；**V5.3-ENG** = 已执行的 30h rollout/relax 工程闭环（历史）
> v1.1 修订：真值访问审计、20k 可扩展加载、Endpoint 标签算法、三臂评估、S2→S3 晋级、闭环预算六项

---

## 0. 裁定摘要

| 事项 | 状态 |
|---|---|
| V5.2 / V5.2-R | **关闭**：D2-G STOP，T1 不授权，负结果归档；旧门控继续有效 |
| V5.3-MP | **有条件通过**：先完成 v1.1 修订与 S0；S0 通过前不启动 GPU |
| S0 目标 | **证明 ≥20k 独立反应具有合格监督数据**；2k/5k 是同一冻结池的嵌套工程子集 |
| S0-A | 库存/元数据资格检查（不读原始 IRC 真值）——可立即启动 |
| S0-B | 经授权的完整 IRC 提取与教师建库——逐次访问记录，硬门是“无未授权访问” |
| S1 2k 冒烟 | S0 + 实施就绪门通过后；≤1 GPU-h 为**资源上限**，不是完成保证 |
| S2 5k | 稳定性/资源检查；**不因统计不显著阻断 S3** |
| S3 20k | 主要科学结果；3 seeds；数据规模为唯一主变量 |
| S4 50k–100k | 仅 S3 后条件扩量；此时才讨论架构扩容 |

**一句话**：先把 20 万级库存中的合格反应转成 20k 可训练数据资产，再用 2k/5k 做工程与资源检查，20k 回答科学问题。

---

## 1. V5.2 完成确认

| 证据 | 结论 |
|---|---|
| D0 PASS | 新 endpoint 标签 1781 正例；terminal 行纳入；起点 FP=0 |
| D1 FAIL | 冻结 R3 表征无 endpoint/ordering 信号（AUROC≈0.47–0.51） |
| D0.1 PASS | 22/22 拓扑异常 = T（键编辑未完成）→ Mask；`d_end_sym` 翻转 0/1781 |
| D2 | P1 PASS（目标信息 +0.678）；P4 FAIL（−0.079）；P5 FAIL（−0.085，有效性 5/6） |
| D2-G | **STOP**：T1 不授权；P5 标签不冻结；GPU 消耗 0 |
| 代码/文档 | 155 项 V5.2-R 测试通过；docs gate 134 文件 CLEAN；`master` 与 origin 同步 |

**结论**：V5.2 作为负结果归档；编辑性修正已写入 D2-G Addendum（“不可学”→“未获可学证据”；P5 TS 为 Mask 带；P3 退化触发；sealed QC 例外；sklearn 钉版）。

---

## 2. 数据现状（实测）

| 资产 | 数量 | 是否可直接训练 |
|---|---:|---|
| 当前 V5 六表训练集 | **50 曲线 / 6,017 节点 / 12,034 样本** | 是（仅 50 曲线；dev 30 / sealed 20） |
| `data/ground_truth/irc_index.parquet` | **199,890 反应 / 22,983,028 帧** | **否**：未转六表、未过 G0/G1、未冻结 split |
| `data/interim/split_assignment.parquet` | 199,168（train 159,355 / valid 19,904 / test 19,909） | **否**：旧 split 语义，非 V5 split |
| `irc_fit_2000_verified` | 2,000 IDs / **1,995 canonical groups** | 否：只有 ID |
| `irc_fit_5000` | 5,000 IDs / **4,994 canonical groups** | 否：只有 ID |
| 直接 fit 冻结集 | 4,964 groups；fit50 已制备 | 部分；需重建 V5 六表 |

- **计数口径**：按去重后的 canonical reaction group 计数，不按原始 ID 数量；2k ≈1,995、5k ≈4,994。
- **旧 sealed 20 曲线**：不因新建 split 自动转为训练数据；此前 D0.1 的 14 条 sealed 标签 QC 访问须登记；最终确认集必须保留**未参与任何设计/标签 QC** 的新反应与新反应族。

---

## 3. 目标架构与首轮冻结配置

```
X_t + G_rxn + H
        │
 StarEGNNV3 (hidden 48, 3 layers, 55,441 params, shared)
        │
   ┌────┴─────┐
 MOVE       PROCESS
 ΔX_K       q̂, p̂_end, s
   └────┬─────┘
        │
 No-Move = [p̂_end ≥ τ_end] ∧ [chemical state consistent]
```

- **首轮不扩网络**：`StarEGNNV3(hidden=48, n_layers=3)` 已冻结；不换教师、不改 Move 损失、不改输入特征。
- **训练资格 ≠ 部署资格**：PROCESS 可以训练 endpoint 完成监督；但 No-Move 的在线控制权只在独立验证（校准、误停率、闭环）通过后启用。Endpoint 指标失败时保留完整训练结果，不删除第二个头、不改写失败记录。
- **唯一主变量 = 数据规模**：第一轮规模曲线只改变训练反应数；架构、教师、损失、优化器、评估集保持冻结。

---

## 4. S0 — 20k 数据资格审计与建库（硬门）

### 4.1 S0-A：库存与元数据资格（不读原始 IRC 真值）

- 输入：库存索引、元数据、已有 split/cohort 清单；**不读取** `data/ground_truth` 的原始 IRC 帧。
- 检查：canonical 身份、R/TS/P 端点齐全、原子映射、反应族、去重、退化、split 资格、历史暴露记录。
- 输出：`metadata_qualification.json`——total / eligible / excluded(reason) / dedup group / family 分布。
- 真值访问：允许为空日志；本阶段不访问原始真值。

### 4.2 S0-B：经授权的完整 IRC 提取与教师建库

- 必须通过现有 **truth accessor / truth_guard**；**禁止**绕过 guard 或直接读 `ground_truth` 原始文件。
- 每次访问记录：source、purpose、reaction group、帧数、字节数、SHA256、approver、timestamp。
- **硬门 = 无未授权真值访问**，不是“日志为空”。合法访问必须完整记录；空日志只适用于 S0-A。
- 输出：`truth_access_log.jsonl` + `extraction_manifest.json`（逐组哈希）。

### 4.3 S0-C：Endpoint Completion 标签算法（`endpoint_label_spec_v1`）

> 这是候选监督定义，必须通过 S0-E 的硬负例与覆盖率检查；不是已验证的化学终点真值。

**正例（高置信）**

1. **真实 IRC 端点帧**（forward→B，reverse→A）：端点身份已验证、键编辑完成、几何合法 → 正例。
2. **端点邻域帧**：`d_end ≤ w_pos`（候选 0.10 Å）**且**目标键编辑完成（所有 required bonds 在容差内）**且**无碰撞/价键错误 → 正例。
3. **同一产物的不同构象**：目标拓扑正确、局部极小、通过能量窗（如 GFN-2 相对参考端点 ≤5 kcal/mol）→ 可作正例；未通过能量窗或构象归属不清 → Mask。
4. **terminal / no_future_contact 行**：仅当 `d_end ≤ w_pos` 且键编辑完成 → 正例；否则 **Mask**（terminal ≠ endpoint）。

**负例**

1. 起点端点帧（给定 intent 的相反端点）。
2. TS 帧及 TS 邻域帧（`d_key_TS ≤ 0.04 Å`）——**永远负例**。
3. 中段帧：目标键编辑未完成且 `d_end > w_neg`。
4. 产物侧但未到端点：目标拓扑正确但 `d_end > w_neg`。

**Mask**

1. `w_pos < d_end ≤ w_neg`（如 0.10–0.30 Å）且构象有效性无法判定。
2. 键编辑完成但几何未松弛/高应变（碰撞、价键异常、能量超窗）。
3. 映射冲突、截断路径、退化曲线（TS 与端点近简并）。
4. 不属于上述正/负的 terminal 行、构象归属不清行。

**计数与平衡**

- 按 reaction group 与 reaction family 报告：正例数、负例数、Mask 数、**有效正例覆盖率**（有 ≥N 个正例的组占比）。
- 类别平衡只在 loss 权重/采样中做；**禁止**通过重复端点帧虚增独立反应数。
- 端点邻域扩展必须逐条通过键编辑 + 几何/能量检查；不得仅按 RMSD 扩正例。

### 4.4 S0-D：分片数据集 + 惰性索引（20k 加载改造）

- 现状：`KeyAtomDatasetV3` 读取六张 CSV 并把全量行/帧几何放入 Python 对象；50 曲线可行，20k 不可直接使用。
- 目标：**分片数据集 + 轻量反应索引 + 按反应组惰性读取 + 有界缓存**。
  - 索引：`curve_id / family / fold / n_nodes / shard_path / hashes`，常驻内存；
  - 分片：按 reaction group/family 切分，单 shard 目标 50–200 MB；
  - 惰性加载：只加载当前 batch 涉及的曲线/帧；LRU 缓存上限可配置；
  - 复用 `pes2ts_core/pesn/offline/training/loader.py` 的 `LazyTeacherProvider` 设计，不另起炉灶；
  - 资源小试：记录 2k/5k/20k 的启动时间、峰值 RSS、每 epoch 读取时间；未测不得声称可扩展。
- 硬门：20k 配置下启动时间、峰值内存、随机访问正确性、shard 等价性（与全量加载逐样本一致）通过预注册上限。

### 4.5 S0-E：标签覆盖率与硬负例验证

| 检查 | 判据 |
|---|---|
| 端点正例 | 真实端点帧 100% 正例；邻域正例逐条通过键编辑 + 几何/能量检查 |
| TS 负例 | TS 帧 0% 正例；TS 邻域 0% 正例 |
| 起点负例 | 起点帧 0% 正例 |
| terminal | 仅 `d_end ≤ w_pos` 且键编辑完成者正例；其余 Mask |
| 构象 | 同一产物多构象：能量窗内可正例，窗外 Mask；报告数量 |
| 覆盖率 | 每 family 有效正例覆盖率、每曲线正例 min/med/max；低于预注册下限的 family 单列 |
| 退化 | 退化曲线单独 flag；含/排两种口径报告 |

### 4.6 S0-F：实施就绪门（文档 ≠ 代码）

> 当前仓库只有规范层；以下文件**尚不存在**（2026-10-21 检查）。S1 前必须全部落地并通过测试。

| 文件 | 状态 | 测试 |
|---|---|---|
| `config/campaigns/pes2ts_v53mp_20261021.yaml` | ❌ 缺失 | 配置/哈希/授权位测试 |
| `pes2ts_core/pesn/models/key_atom_loss_v53mp_v3.py` | ❌ 缺失 | loss/mask/梯度测试 |
| `pes2ts_core/pesn/offline/training/key_atom_v53mp_train_v3.py` | ❌ 缺失 | 分阶段/恢复/Move guard 测试 |
| `scripts/campaigns/r3_v53mp_run.py` | ❌ 缺失 | runplan/ledger/GPU lock 测试 |
| 分片加载器（`KeyAtomDatasetV3` 扩展或新类） | ❌ 未实现 | shard 等价性/内存小试 |
| Endpoint 标签生成器（`endpoint_label_spec_v1`） | ❌ 未实现 | 正/负/Mask/覆盖率测试 |
| 导出 `pes2ts_move_process_bundle_v1` | ❌ 未实现 | bundle schema/τ_end 测试 |

**S0 总门**：20k 合格组 + 无未授权真值访问 + split 冻结 + 标签覆盖率达标 + 加载器资源小试通过 + 实施就绪 7/7。**不过不进 S1。**

---

## 5. S1–S4 训练阶段

| 阶段 | 规模 | 目的 | 预算（上限） | 进入/晋级 |
|---|---:|---|---|---|
| **S1 冒烟** | 2k（20k 池的嵌套子集） | 两头可训、显存/梯度、吞吐、恢复 | ≤1 GPU-h（资源上限） | S0 全通过 |
| **S2 稳定** | 5k（同一池） | 固定结构下的学习曲线、资源/稳定性 | S1 实测外推，需授权 | S1 健康检查通过 |
| **S3 正式** | 20k（同一池） | 主要科学结果 | S1 实测后冻结，需授权 | S2 数据/稳定性/资源/合法性通过；**不要求统计显著** |
| **S4 条件** | 50k–100k | 饱和判断/架构优化 | 另议 | S3 通过且曲线未饱和 |

- **S2→S3 规则**：只要数据质量、训练稳定性、GPU 资源、任务合法性达标，且没有明确的大幅泛化退化，就优先完成 20k 主训练。2k/5k 是工程与资源检查，不是科学否决权。
- **停止条件**（任一）：数据泄漏、教师错误、未授权真值访问、数值发散、灾难性退化（预注册阈值）、资源不可承受。
- **预算删失**：S1 的 ≤1 GPU-h 是资源上限，不是完成保证；若三臂无法完成基本健康检查，记录 `budget_censored` 并申请调整，**不得**为赶时间改变有效比较条件。

---

## 6. 损失、采样与训练

\[
L_{\rm total}=L_{\rm Move}+L_{\rm Process},
\qquad
L_{\rm Process}=\lambda_rL_{\rm ranking}+\lambda_qL_{\rm progress}+\lambda_eL_{\rm endpoint}
\]

- \(L_{\rm Move}\)：R3 已验证组合（atom + pair + cosine + strong + dead）；首轮不改。
- \(L_{\rm ranking}\)：主监督；同 curve/intent 的 pairwise margin，允许跨 TS。
- \(L_{\rm progress}\)：辅助 Huber，低权重起步。
- \(L_{\rm endpoint}\)：BCEWithLogits + 类别权重 + 独立 Mask。
- 采样：按 reaction group 均衡；按起点/TS 前/TS 后/终点分层；固定 seed；有效 batch 记录。
- 梯度：记录 Move/Process 梯度范数与余弦、encoder 梯度占比；λ 用 2–10% 目标校准；不做大网格。
- 模型选择：只用 dev；最终确认集隔离；单折/单 seed 不晋级。
- **Move guard ≤2%**：破防 = `scientific_failure`，保留 run、分析、不丢弃；`invalid_run` 仅限泄漏/协议/哈希/运行时错误。

---

## 7. 评估：三臂、基线与闭环

### 7.1 三臂主评估

| 臂 | 主指标 | 科学问题 |
|---|---|---|
| **A：Move-only** | Move RMSE/zero、cosine、sign、dead | 数据规模能否改善基本运动能力？ |
| **B：Move + Progress** | A 的全部指标 + directed ordering（mean、跨 TS） | Progress 是否帮助 MOVE 学习？ |
| **C：Move + PROCESS** | B 的全部指标 + Endpoint AUPRC、起点/TS 误停率 | Endpoint 监督是否带来额外价值？ |
| **C vs A/B** | 同评估集 Move 退化、完整 rollout | 联合学习是否负迁移？ |

- **B 不与 A 比 Progress**：A 没有 Progress 输出；B 的 Progress 只与常数/拓扑基线/自身消融比。
- **C 的 Endpoint 基线**：常数 prevalence、拓扑规则、相同数据条件下的独立分类器（如 key-distance HGB/冻结表征探针）；**不得**用不产生 endpoint 分数的 B 作为 AUPRC 基线。
- **冻结评估集**：S1/S2/S3 共用同一独立 dev/confirm 反应集；训练子集是嵌套的，评估集不随规模更换。否则 2k→5k→20k 的差异无法归因。
- 指标分开报告（Move 误差、ordering、Endpoint AUPRC、闭环成功率），不合成单一分数。

### 7.2 闭环预算：参考归一化 vs 自主

| 方式 | 预算来源 | 用途 |
|---|---|---|
| **Reference-normalized rollout** | 参考 IRC 长度 \(L_{\rm key}\)，\(B=\max(40,\lceil1.5L_{\rm key}/0.08\rceil)\) | 离线教师诊断、与既往实验比较 |
| **Autonomous rollout** | 固定预算，或仅用当前合法信息计算的预算 | 正式完整路径成功率评价 |

- 两种结果分开报告；**不得**把使用参考 IRC 长度的成功率当作真实自主导航能力。
- 闭环记录：端点到达率、TS 跨越率、首次 No-Move 距离分布、碰撞/非法事件、预算耗尽比例。
- TS 候选交独立物理验证（OptTS + 单虚频 + 双向 IRC + 端点身份）；此前只称 `TS_candidate`。

---

## 8. 代码实施清单与就绪门

### 8.1 新增/修改文件

| 文件 | 任务 |
|---|---|
| `pes2ts_core/pesn/models/key_atom_heads_v3.py` | 新增 `ReactionProcessHead` + `KeyAtomV53MPModel`；旧停止头标 LEGACY，禁止新训练 |
| `pes2ts_core/pesn/models/key_atom_loss_v53mp_v3.py` | `L_Move + L_Process`，独立 mask，ranking/progress/endpoint |
| `pes2ts_core/pesn/offline/training/key_atom_v53mp_train_v3.py` | 分阶段 trainer；A/B/C；Move guard；梯度监控；resume |
| `scripts/campaigns/r3_v53mp_run.py` | 训练驱动；campaign root / runplan / ledger / GPU lock / monitor |
| `pes2ts_core/pesn/offline/data/key_atom_dataset_v3.py` | 全 status 加载；MOVE/PROCESS 独立 eligibility；分片/惰性接口 |
| `pes2ts_core/pesn/offline/data/key_atom_endpoint_labels_v1.py` | 保留诊断；新增 `endpoint_label_spec_v1` 生成器 |
| `scripts/agent_ops/data_gate/r3_build_tables.py` | 扩展 20k cohort 六表构建；S0-A/B/C 输出 |
| `pes2ts_core/pesn/offline/publishing/export.py` | `pes2ts_move_process_bundle_v1`（u + q/endpoint + τ_end + provenance） |
| `config/campaigns/pes2ts_v53mp_20261021.yaml` | S0–S4、arms、seed、授权位、freeze hashes、预算上限 |
| `config/registry/g_v_map.yaml` | V5.3-MP gates 增补 S0-A/B、20k、readiness |

### 8.2 测试

- heads：形状、主干 bitwise 不变、LEGACY 禁用；
- labels：正/负/Mask、terminal、TS 负例、构象、覆盖率；
- loader：shard 等价性、惰性加载、内存/启动小试；
- trainer：分阶段、Move guard、失败分类、resume；
- runplan：预算/阶段/授权/哈希绑定；
- export：bundle schema、τ_end 冻结。

### 8.3 实施就绪门

`S0-F` 的 7 项全部存在且测试通过，才允许 S1；**文档一致性测试通过 ≠ 代码已实现**。

---

## 9. 文档同步

- 本 v1.1 替换 v1；[文档同步清单](PES2TS_V5.3-MP_文档同步清单_v1_20261021.md) 增补：S0-A/B、20k、分片加载、Endpoint 标签算法、三臂评估、闭环预算、实施就绪。
- 合同 [MoveProcessContract v1](../contracts/PES2TS_MoveProcessContract_v1.md) 增补：合法真值访问记录、训练/部署资格分离、`endpoint_label_spec_v1` 引用。
- registry `V5.3-MP.gate_note` 增补 S0-A/B 与 readiness；docs gate 必须 CLEAN。

---

## 10. 预算与时间线（估算，S0 实测后冻结）

| 阶段 | 时间（估算） | GPU | 备注 |
|---|---|---|---|
| S0-A 元数据资格 | 1–2 天 | 0 | 不读原始真值 |
| S0-B/C 20k 提取+建库 | 5–10 天（视并行） | 0 | 分批；逐组哈希 |
| S0-D/E 分片+标签验证 | 2–4 天 | 0 | 资源小试 |
| S0-F 就绪门 | 1–2 天 | 0 | 7 项检查 |
| S1 2k 冒烟 | 0.5–1 天 | ≤1 GPU-h | 资源上限 |
| S2 5k | 1–2 天 | S1 外推后冻结 | 需授权 |
| S3 20k | 3–7 天 | 单独授权 | 主要结果 |
| S4 50k–100k | 待定 | 另议 | 仅 S3 后 |

- 禁止先承诺 20k 全量 GPU 预算；S1 实测 updates/s、显存、读取时间后再冻结 S2/S3 上限。

---

## 11. 风险与止损

| 风险 | 缓解/止损 |
|---|---|
| 真值访问违规 | S0-B 逐次日志；硬门=无未授权访问；违规即停 |
| 20k 构建不足 | S0-A 先证明合格组数；不足则缩规模或扩库存，不降级硬跑 |
| 加载器内存/启动爆炸 | S0-D 分片/惰性/有界缓存；资源小试不过不进 S1 |
| Endpoint 正例过少 | S0-C/E 覆盖率报告；低覆盖 family 单列；不重复端点帧 |
| Endpoint 误正（TS/产物侧） | TS 硬负例；键编辑+几何/能量双检查；Mask 灰色带 |
| PROCESS 损伤 MOVE | Move guard ≤2%；梯度监控；破防记 scientific_failure |
| S2 不显著 | 不阻断 S3；只有泄漏/教师错误/发散/灾难退化才停 |
| 部署越权 | 训练资格≠部署资格；No-Move 仅在独立验证后启用 |
| 预算不足 | S1 记录 budget_censored 并申请调整；不改比较条件 |
| sealed 污染 | 旧 sealed 不自动转训练；登记历史 QC；保留未触碰 family 作最终确认 |

---

## 12. 请求授权

1. **S0-A 立即启动**（元数据资格，CPU，0 GPU，不读原始真值）；
2. **S0-B 真值提取授权**（经 truth_guard，逐次记录；硬门=无未授权访问）；
3. **S0 完成后 S1 2k 冒烟**（≤1 GPU-h 资源上限；需 S0-F 7/7）；
4. **S2/S3 逐阶段授权**（S3 预算在 S1 实测后冻结）。

---

## 13. 变更记录

| 版本 | 日期 | 内容 |
|---|---|---|
| v1.1 | 2026-10-21 | 按用户评审修订：S0-A/B 真值访问、20k 目标与去重计数、分片惰性加载、Endpoint 标签算法、三臂评估与冻结评估集、S2→S3 放宽、闭环双预算、实施就绪门、训练/部署资格分离 |
| v1.0 | 2026-10-21 | 初版：V5.2 关闭确认；MOVE–PROCESS 架构；S0–S4；三臂 A/B/C；数据/代码/文档清单 |
