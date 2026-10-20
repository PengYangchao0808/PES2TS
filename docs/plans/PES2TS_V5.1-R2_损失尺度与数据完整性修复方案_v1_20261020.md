# PES2TS / PES-N V5.1-R2：损失尺度与数据完整性修复方案 v1

> 状态：**Proposed v1**（可执行研究方案；先于任何 R2 训练冻结；不启动正式 OOF/封存/扩量）
> 日期：2026-10-20
> 层级：G3.P.2（训练/损失/评测），支持 G3.P.0/.1/.3、G0/G1；V5.1-R2 是旧 V5.1 任务的**最后一轮机制修复**，不是新架构
> 上游证据：[V5.1a-R1 最终报告](../../outputs/pesn/v5_1_r1/final_report.md)、[R1 协议](../../outputs/pesn/v5_1_r1/protocol_r1.json)、[校准报告](../../outputs/pesn/v5_1_r1/calibration/calibration_report.md)、[Pair 归因](../../outputs/pesn/v5_1_r1/pair_errors/pair_error_report.md)
> 下游消费者：V5.2 新 Key-Atom 合同、V5.3 纯几何 EGNN、V6.1 物理 MLP
> 配套文档：[V5–V7 v2.0 架构修订方案](PES2TS_V5-V7_三阶段研究与架构修订方案_v2.0_20261020.md)、[ADR-0027](../design/decisions/ADR-0027-V5.1参考架fallback与双分母评估.md)、[ADR-0028](../design/decisions/ADR-0028-关键原子几何策略合同与V5-V7任务拆分.md)

---

## 0. 一页结论

V5.1a/R1 已经证明：**继续调 λ 或增加网络模块，不是当前的主要矛盾**。当前矛盾是：

1. 辅助损失在 Å 量级残差上构造，梯度占比不可控：exact_trio 超配 4–6 倍，桶均值项只有 0.42%；
2. Dead/Strong 误差方向相反，全局尺度或单一幅度头无法同时修复；
3. checkpoint 与 ramp 强交互，单折选型会把偶然波动当成配方；
4. 标签/gauge/近重复冲突仍在污染训练与评估。

V5.1-R2 的目标不是“必须找到一个可晋级配方”，而是：

> **在旧 V5.1 任务上，建立可信的无量纲损失、梯度占比控制、ramp/checkpoint 纪律，并完成 S1 数据质量审计；给出可复算的“有效/无效/不可达”结论。**

V5.1b 继续关闭；R2 不产生正式 OOF，不打开 sealed，不启动 Fit150+。R2 的任何正结果最多只能成为 V5.2 新任务的基线候选，不能自我批准 V5.1b。

---

## 1. 范围与非目标

### 1.1 范围

- 保留 V5.1 的任务、数据口径、5-fold split 和旧 key-atom 标签，以便与 B0/A2/A3 做可比对照。
- 只修改训练机制：pair loss 的无量纲化、梯度占比控制、ramp/checkpoint 规则。
- 并行完成 S1 数据质量审计：2022、342 对冲突、8.6% gauge、2309/1715/1156、1675/238/1484。
- 所有阈值、损失形式、选型规则在 R2 训练前写入 `protocol_r2.json`；训练后不得放宽。

### 1.2 非目标

- 不实现新的 Key-Atom-only 网络、Stop Head、V6 xTB 模块或 V7 RL；这些从 V5.2 开始。
- 不改变 V5.1a/R1 的冻结报告、阈值、数据划分或结论。
- 不做正式 outer OOF（15-run/3-seed）；不打开 Confirm/OOD/sealed。
- 不删除困难样本；不把 2022 直接从分母中抹掉。
- 不继续旧 `{0.03, 0.1, 0.3}` λ 网格；不重试 B3；不做无归因的结构加法。

---

## 2. R1 证据与 R2 假设

### 2.1 直接证据

| 观察 | 数值 | 对 R2 的含义 |
|---|---|---|
| A2 梯度占比 | 4.6% | 已有可观梯度，但 SRE 未稳定改善 → 问题不只是权重 |
| A3 梯度占比 | 0.42% | 监督太弱，不能据此否定 dead suppression 思想 |
| exact_trio 梯度 | 主目标 4–6 倍 | 旧 pair 项形式过强，必须无量纲化/重标定 |
| 全局校准 s* | 中位 1.008 | 全局尺度被证伪，R2 不做 post-hoc scalar |
| Strong 欠猜 | 平方误差 46.2% | 需要针对 strong 的幅度监督，但必须避免放大 dead |
| Dead 虚假 | 76.4% >0.005 Å | 需要尺度无关的 dead 抑制，不是简单 hinge |
| SRE 集中度 | strong+dead ≈82% | R2 只聚焦这两类，不扩散到所有 pair |
| checkpoint | best 多在 400–950 | ramp 与选型必须一起修 |

### 2.2 R2 假设

| 编号 | 假设 | 验证 | 失败处理 |
|---|---|---|---|
| **H-R2-1** | 无量纲 strong Huber 能在不放大 dead 的前提下提供有效梯度 | 梯度占比 + strong/dead 分桶 | 若 dead 恶化 >2%，停止该形式 |
| **H-R2-2** | `(slack/τ_d)²` dead 抑制能在 λ 较小时进入有效窗口 | 梯度占比 ≥1% 且 dead 幅度比下降 | 若仍 <1%，记录 `scale_unreachable`，不再调 λ |
| **H-R2-3** | 从训练开始或短 warmup 加入辅助项，比旧 250/250 ramp 更有效 | ramp 三臂对照 | 若旧 ramp 不劣，保留旧 ramp 作为控制 |
| **H-R2-4** | 多折/多种子 checkpoint 选择能避免 B3 式单折偶然晋级 | 3-seed × 3-fold 中位 + 预注册选型 | 若仍不稳定，退回固定 update 预算报告 |

---

## 3. 损失修复

### 3.1 Strong pair：无量纲 Huber

对 strong 桶（\(|\Delta d^*|\ge 0.02\) Å）：

\[
L_{\mathrm{strong}}=
\frac{1}{|S|}\sum_{p\in S}
\operatorname{Huber}_{\delta}
\left(
\frac{\hat{\Delta d}_{p}-\Delta d_{p}^{*}}
{\max(|\Delta d_{p}^{*}|,\tau_s)}
\right)
\]

- \(\delta=0.25\)（与旧归一化 Huber 一致，作用在无量纲误差上）。
- \(\tau_s\) **主值 0.01 Å**（strong 桶下界 0.02 Å 的一半）；预注册敏感性仅 `{0.005, 0.02}`，只在 fold0/seed17 诊断，不用于选型。
- 分母使用 `max(|Δd*|, τ_s)`，避免真值接近 0 时误差爆炸。
- 该形式不再直接依赖 Å 量级的小残差，梯度对 `|Δd*|` 尺度不敏感。

### 3.2 Dead pair：τ 归一化平方 hinge

对 dead 桶（\(|\Delta d^*|<0.005\) Å）：

\[
L_{\mathrm{dead}}=
\frac{1}{|D|}\sum_{p\in D}
\left[
\frac{\max(0,|\hat{\Delta d}_{p}|-\tau_d)}{\tau_d}
\right]^2
\]

- \(\tau_d=0.005\) Å，沿用 R1 dead 分桶边界。
- 除以 \(\tau_d^2\) 后，0.005 Å 量级的残差变成 O(1)，不会再被主损失淹没。
- 仍使用 bucket-mean（除以 dead pair 数），避免 dead 样本多的反应主导梯度。
- 监控输出塌缩：strong 幅度比不得相对 B0 下降 >10%；dead-zone/弃权不得异常上升。

### 3.3 组合

\[
L = L_{\mathrm{primary}}
+ \lambda_s L_{\mathrm{strong}}
+ \lambda_d L_{\mathrm{dead}}
\]

- 第一轮不同时开两个新项；先单独筛选，再按预注册规则组合。
- 组合臂只在 strong 与 dead 各自出现“正向且不互补冲突”时解锁；不得因为“想一次修完”而强行组合。

---

## 4. 梯度占比控制

### 4.1 定义

\[
\rho_k=
\frac{\|\nabla_\theta(\lambda_k L_k)\|}
{\|\nabla_\theta L_{\mathrm{primary}}\|}
\]

必须在**共享 Encoder** 和**动作头**分别测量，不能只看总 loss 或最后层梯度。

### 4.2 测量协议

- 测量点：沿用 R1 的 `B0_f0_s17_v51/checkpoint_best`（实际 update 550；名义 post-ramp 600），20 个真实 inner-train batch；记录实际 checkpoint update/sha256。
- 统计：中位 + IQR；同时记录 encoder/head 两个参数组的 ρ。
- 目标档：**2%** 与 **10%**（候选实验设置，不是已验证最佳区间）。
- 求解 λ：不用网格。先测 \(\|\nabla L_k\|\) 在 λ=0 时的值，再按
  \[
  \lambda_k^{target}=\rho_{target}\frac{\|\nabla L_{\mathrm{primary}}\|}{\|\nabla L_k\|}
  \]
  做一维校准；训练中每 100 步复测一次，记录漂移。
- 若在 λ 上限（预注册 0.5）下 ρ 仍 <1%，判定该损失形式在当前任务上 `scale_unreachable`，停止该臂并记录机制，不继续加 λ。

### 4.3 重要约束

A2 已经在 4.6% 梯度占比下没有稳定改善。因此：

- **不能假设 10% 一定优于 2%**；两档都必须作为独立假设检验。
- 若 2% 与 10% 都没有正向，结论是“该监督目标/形式不足”，不是“再调一个中间值”。
- 梯度占比达标但指标不改善，同样判失败；ρ 是必要条件，不是充分条件。

---

## 5. Ramp 与 checkpoint

### 5.1 Ramp 三臂

| 臂 | 辅助项开始 | ramp | 用途 |
|---|---|---|---|
| **R0** | update 250 | 250/250 | 旧 V5.1a 控制 |
| **R1** | update 0 | 0/0 | 从训练开始就加入规范化辅助项 |
| **R2** | update 50 | 50/50 | 很短 warmup |

- 三臂使用相同损失形式、相同 λ（按梯度占比校准）、相同预算 3000 updates。
- 先做 fold0/3 seeds 筛选；只有 R1/R2 中至少一个在 2/3 fold 方向正向，才进入 fold1–2 复现。

### 5.2 Checkpoint 选择

旧规则（B3 失败的一个原因）是 fold0 单折 + RMSE 容差内选 SRE。R2 改为：

1. 每 50 updates 记录 inner-val 的 RMSE/zero、SRE、sign、dead-zone、nearTS cos。
2. 候选 checkpoint 必须满足 `RMSE/zero ≤ best_RMSE × 1.02`。
3. 在候选内按 **3-seed 中位 SRE** 选择；同时要求 sign 相对 B0 下降 ≤0.01、dead-zone 上升 ≤0.01。
4. 选型使用 **folds 0–2 的 3-seed 中位**，不得用单折/单 seed；fold3–4 只用于最终确认，不参与选型。
5. 预注册评估时点：250/500/750/1000/1500/2000/3000；不得在报告阶段新增时点。

---

## 6. S1 数据质量审计

S1 与 R2 训练并行，但**任何影响 >5% 训练 pair 的未关闭 P0 都必须先报告**；是否暂停训练由预注册规则决定。

| 编号 | 问题 | 工作 | 验收 |
|---|---|---|---|
| **DQ1** | RXN_2022 恒定 `d_eff == d0`（120 行） | 核查原始记录、标签生成函数、封顶逻辑；判定 `suspect-label` / `valid-cap` / `unknown` | 书面结论 + 证据路径；不删除、不悄悄改分母 |
| **DQ2** | 342 对近似输入/冲突教师 | 分类：真实多解 / 拓扑条件不足 / 映射/gauge / 数值问题 | 342/342 分类表；每类计数与示例 |
| **DQ3** | 8.6% gauge unresolved（480 行） | ADR-0027 研究；二跳 fallback 已证 0/8，不重复；双分母报告 | S1 门结论：可实现/不可实现/需新 ADR；主分母不变 |
| **DQ4** | 2309/1715/1156 gauge + 近重复冲突 | 分离模型误差与标签冲突；per-cluster 诊断 | 每簇结论卡；不把标签冲突算作模型失败 |
| **DQ5** | 1675/238/1484 真实困难簇 | 固定困难样本诊断报告；幅度过冲/远 TS 方向/dead 虚假 | 固定报告 + 后续 V5.2 复测入口 |
| **DQ6** | 342 对与 fit50 内冲突 | 与 DQ2 合并；保留 context_resolved / mapping_overflow / true conflict | 无未关闭 P0；冲突权重规则冻结 |

**S1 通过条件**：

- DQ2 342/342 分类完成；
- DQ1 有明确处置（suspect 样本不进入硬标签训练，但保留在评估分母/报告中）；
- DQ3 有 ADR-0027 结论，且主分母不变；
- DQ4/DQ5 有固定报告；
- 没有未关闭 P0；所有例外有书面理由。

---

## 7. 实验矩阵与预算

### 7.1 阶段化矩阵（避免一次改多个因素）

| 阶段 | 目的 | 臂 | folds × seeds | 新 run 数 |
|---|---|---|---|---|
| **R2-0** | 复核 B0 基线 | R0 | 0–2 × 3 | 0（复用/补齐） |
| **R2-1** | 损失形式筛选 | R0, S2, S10, D2, D10 | fold0 × 3 | 15 cells（R0 可复用则为 12 新 run） |
| **R2-2** | 复现 Top-2 | Top-2 | folds 1–2 × 3 | 12 |
| **R2-3** | ramp 对照 | 最佳损失 × {R0,R1,R2} | fold0 × 3 | 6 |
| **R2-4** | 确认（仅 dev-positive） | Top-1 | folds 3–4 × 3 | 6 |

- `S2/S10`：strong 损失，目标梯度占比 2%/10%；`D2/D10`：dead 损失，目标 2%/10%。
- R2-0 的 A0 folds0–2 × 3 seeds 共 9 cells 已由 V5.1a/R1 baseline_fill 交付；只做 hash/status 复核，不重跑（hash 不匹配才补齐）。
- **Top-2 选型规则（预注册）**：按 fold0 三 seed 中位 SRE 排序；入选项须满足 RMSE/zero ≤ B0×1.02、sign 下降 ≤0.01、dead-zone 上升 ≤0.01；并列时取 RMSE 更低者。选型后不得因 folds1–2 结果回头改规则。
- 组合臂仅在 S 与 D 各自正向时解锁；不预先塞进矩阵。
- 总计新 run ≤36；单 run ~2.5 min（RTX 4060 Ti 8GB），训练 GPU 约 1.5–2 h；加数据审计/分析，总预算 ≤600 min。
- 所有 run 保存 `checkpoint_best/final/last`、`eval.jsonl`、`metrics.json`、`run_record.json`、`console.log`、config/data hash。

### 7.2 止损规则

- 某臂在 ≥2/3 dev folds 的 3-seed 中位 SRE 退化 >0.03，或 RMSE/zero 退化 >5% → 停止该臂。
- 某臂在 λ 上限 0.5 下 ρ 仍 <1% → `scale_unreachable`，停止该臂。
- S1 发现影响 >5% 训练 pair 的未关闭 P0 → 暂停训练，先出数据质量报告。
- 任何臂出现输出塌缩（strong 幅度比下降 >10% 或 dead-zone 异常上升）→ 停止。

### 7.3 Confirm 准入（仅 dev-positive 时）

Top-1 在 folds3–4 × 3 seeds 上必须同时满足：

- 两折的 3-seed 中位 SRE 改善 ≥0.01（\(\Delta SRE \le -0.01\)）；
- RMSE/zero 相对 B0 退化 ≤1%；
- strong sign 下降 ≤0.01；
- dead-zone 上升 ≤0.01；
- cluster-level paired bootstrap 的 95% CI 不支持“仅由单一大簇驱动改善”。

未通过则不得称 `scale_controlled`；只能记 `no_gain`，并禁止用 folds3–4 回头调参。

---

## 8. 验收标准

R2 的验收是**可靠结论**，不是必须晋级。四种允许的终局：

| 终局 | 条件 | 后续 |
|---|---|---|
| **scale_controlled** | 至少一个臂的共享 Encoder ρ 达到目标 ±30%，动作头 ρ 有报告且 ≤0.3（不被辅助项淹没），且 dev 指标方向正向、无 sign/dead 退化 | 可作为 V5.2 基线候选；仍不开放 V5.1b |
| **no_gain** | ρ 达标但指标无正向；或正向只在单折 | 记录监督目标不足；进入 V5.2 新合同，不再调旧 λ |
| **scale_unreachable** | 在 λ 上限下 ρ 仍 <1% | 记录形式不可达；下一版换相对误差/τ 形式 |
| **data_blocked** | S1 发现未关闭 P0 影响训练/评估 | 先修数据；R2 训练暂停/降级 |

**必须交付**：

- `protocol_r2.json`（训练前冻结）；
- `s1/data_quality_s1.md` + `.json`；
- `gate_report_r2.md` + `.json`；
- `run_records.jsonl`（含 config/data hash、seed、fold、best_update、ρ_encoder、ρ_head）；
- `frozen_candidate_r2.json`（默认 `approved_for_v5_1b=false`，R2 无权自我批准）；
- `decision_memo.md`（四种终局之一 + 证据 + 下一步）；
- 代码/配置 commit 与 tag。

**明确不通过**：

- 只在单折/单 seed 正向；
- ρ 未达标却宣称“损失尺度已修复”；
- 用确认折结果回头调 λ/ramp/checkpoint；
- 用 R2 结果打开 V5.1b、sealed 或 Fit150。

---

## 9. 执行顺序

1. **冻结 R1**：提交/打 tag；确认 `outputs/pesn/v5_1_r1/` 只读。
2. **写协议**：`protocol_r2.json` 冻结损失形式、τ_s、λ 校准规则、ramp 臂、checkpoint 规则、S1 验收、止损。
3. **S1 审计启动**：DQ1–DQ6 并行；先出 P0 清单。
4. **R2-0 基线复核**：确认 B0 可复用/补齐。
5. **R2-1 fold0 筛选**：S2/S10/D2/D10；测量 ρ。
6. **R2-2 folds 1–2 复现**：Top-2。
7. **R2-3 ramp 对照**：最佳损失 × 三 ramp。
8. **R2-4 确认**：仅 dev-positive 的 Top-1，folds 3–4。
9. **报告与决策**：四种终局之一；更新 V5.2 输入清单。

---

## 10. 与 V5.2 的接口

R2 的输出只进入 V5.2 的以下位置：

- 无量纲损失形式与梯度占比纪律 → V5.2 的 Move/Stop 辅助损失；
- ramp/checkpoint 规则 → V5.2 训练协议；
- S1 数据质量结论 → V5.2 的 K 掩码、标签权重、gauge 分母；
- Hard cluster 报告 → V5.2 的固定困难样本复测集。

**不进入**：旧 V5.1b 的 15-run OOF；旧 sealed；旧 V6 progress-decoder；旧 V7-A/B/C。

V5.2 起使用新的 `GeometryPolicyContract_v1` 和新的评估分母；R2 的旧任务分数不得直接作为 V5.2 的门槛或初始化权重。

---

## 11. `protocol_r2.json` 最小字段（草案）

```json
{
  "schema_version": "pesn_v5_1_r2_protocol_v1",
  "campaign_id": "V5.1-R2",
  "frozen_before_training": true,
  "parent": "V5_1A_R1",
  "task_scope": "old_v5_1_key_atom_task",
  "sealed_policy": "r2_never_unseals",
  "formal_oof": "none",
  "losses": {
    "strong": {
      "form": "huber((dd_pred-dd_true)/max(abs(dd_true),tau_s))",
      "delta": 0.25,
      "tau_s_angstrom": 0.01,
      "tau_s_sensitivity": [0.005, 0.02]
    },
    "dead": {
      "form": "(max(0,abs(dd_pred)-tau_d)/tau_d)^2",
      "tau_d_angstrom": 0.005
    }
  },
  "gradient_control": {
    "targets": [0.02, 0.10],
    "measure_on": ["shared_encoder", "action_head"],
    "measure_update_nominal": 600,
    "measure_checkpoint": "B0_f0_s17_v51/checkpoint_best (actual update 550)",
    "batches": 20,
    "lambda_cap": 0.5,
    "unreachable_if_below": 0.01
  },
  "ramp_arms": {
    "R0": {"start": 250, "ramp": 250},
    "R1": {"start": 0, "ramp": 0},
    "R2": {"start": 50, "ramp": 50}
  },
  "checkpoint": {
    "eval_every": 50,
    "eval_points": [250, 500, 750, 1000, 1500, 2000, 3000],
    "rmse_tolerance": 1.02,
    "select_by": "3-seed median SRE over dev folds 0-2",
    "sign_drop_max": 0.01,
    "dead_zone_increase_max": 0.01
  },
  "s1_gates": ["DQ1", "DQ2", "DQ3", "DQ4", "DQ5", "DQ6"],
  "futility": {
    "sre_regression_stop": 0.03,
    "rmse_regression_stop": 0.05,
    "collapse_strong_amplitude_drop": 0.10
  },
  "outputs_root": "outputs/pesn/v5_1_r2"
}
```

该 JSON 是**草案字段清单**；实现时由 `protocol_r2.json` 的实际 schema/validator 冻结，训练后不得新增/放宽字段。

---

## 12. 执行 WBS 与日程（下一阶段 V5.1-R2）

### 12.1 关键路径

```text
WP0 冻结 R1/tag
  → WP1 protocol + config 冻结
  → WP2 归一化 loss 实现 + 单测
  → WP3 梯度探针 + λ 校准
  → WP5 baseline + screen（fold0）
  → WP6 dev 复现（folds1–2）+ ramp 对照
  → WP7 confirm（仅 dev-positive，folds3–4）
  → WP8 报告 + 终局决策
        ↑
WP4 S1 数据审计（与 WP2/WP3 并行；硬门）
```

**总预算**：≤600 min（R2 训练 + 分析）；S1 CPU 审计与训练并行。单 run ~2.5 min，R2 新增 ≤36 run，GPU 约 1.5–2 h。

### 12.2 工作包

| WP | 任务 | 关键文件/产物 | 验收 | 预计 |
|---|---|---|---|---|
| **WP0** | 冻结 R1：提交/打 tag、生成 53-run 证据清单；创建 R2 分支 | `outputs/pesn/v5_1_r1/`、`r1_evidence_manifest.json`、git tag | R1 产物只读；无未提交 R1 代码；tag 可复现 | 0.5 h |
| **WP1** | 写 `protocol_r2.json` 与 `v5_1_r2.yaml`；冻结 τ_s/τ_d、梯度目标、ramp、checkpoint、S1 验收、止损 | `outputs/pesn/v5_1_r2/protocol_r2.json`、`config/campaigns/v5_1_r2.yaml` | protocol/config/split hash 落盘；训练前写入 | 1 h |
| **WP2** | 实现 R2 归一化 loss：strong 无量纲 Huber、dead τ 归一化 hinge；保持 λ=0 时 bit-identical；加 schema v4 | `pes2ts_core/pesn/offline/training/key_atom_bc.py`、`tests/pesn/test_key_atom_losses_r2.py` | 旧 665 测试 + 新单测全绿；空桶/边界/非有限值有测试 | 3 h |
| **WP3** | 梯度探针：固定 B0 checkpoint、20 真实 batch、encoder/head 分组梯度范数；解 2%/10% λ | `scripts/campaigns/r2_gradient_probe.py`、`gradient_probe_r2.json` | ρ 可复算；λ 写入 protocol.executed；不可达则 `scale_unreachable` | 1.5 h |
| **WP4** | S1 数据审计：DQ1–DQ6；342 分类、2022 处置、gauge ADR-0027、hard cluster 卡 | `scripts/campaigns/r2_s1_audit.py`、`s1/data_quality_s1.md/json`、`conflict_classification.csv` | 342/342 分类；无未关闭 P0；主分母不变 | 3 h（并行） |
| **WP5** | R2-0 基线复核 + R2-1 fold0 screen（S2/S10/D2/D10） | `run_v5_1_r2.py`、`screen_summary.csv`、run dirs | A0 9 cells 复用/补齐；12 新 run 完整 checkpoint/log | 2 h |
| **WP6** | R2-2 Top-2 复现（folds1–2）+ R2-3 ramp 三臂 | `dev_summary.csv`、`ramp_summary.csv` | 12 + 6 新 run；无止损触发 | 2 h |
| **WP7** | R2-4 confirm（仅 dev-positive；folds3–4） | `confirm_summary.csv` | 6 新 run；预注册准入判定 | 1 h |
| **WP8** | 报告与终局：gate report、decision memo、frozen candidate | `gate_report_r2.md/json`、`decision_memo.md`、`frozen_candidate_r2.json` | 四终局之一；`approved_for_v5_1b=false`；证据链完整 | 1 h |

### 12.3 日程（建议 2.5–3 天，关键路径 ≤600 min）

| 时间 | 工作 | 产出 |
|---|---|---|
| **Day 1 上午** | WP0 + WP1：冻结 R1、写 protocol/config、dry-run | tag、protocol_r2、v5_1_r2.yaml、dry-run 通过 |
| **Day 1 下午** | WP2：loss 实现 + 单测；WP4 启动 | R2 loss schema v4、单测绿、S1 P0 清单 |
| **Day 2 上午** | WP3：梯度探针 + λ 校准；WP4 完成 | gradient_probe_r2.json、λ、S1 报告 |
| **Day 2 下午** | WP5 + WP6：screen、dev、ramp | screen/dev/ramp 表 |
| **Day 3 上午** | WP7 + WP8：confirm（若解锁）、报告 | gate_report、decision_memo、frozen_candidate |

### 12.4 运行顺序与命令（目标接口）

> 以下入口在 WP2/WP5 实现；当前仓库只有 R1 驱动，不能直接用于 R2。

```bash
# 0) 冻结 R1（提交后）
git tag v5.1a-r1
git checkout -b v5.1-r2

# 1) 训练前冻结
python scripts/campaigns/run_v5_1_r2.py --stage baseline --dry-run

# 2) 梯度探针（不训练、不更新权重）
python scripts/campaigns/r2_gradient_probe.py \
  --checkpoint outputs/pesn/v5_1/runs/B0_f0_s17_v51/checkpoints/checkpoint_best.pt \
  --batches 20 --targets 0.02 0.10 \
  --out outputs/pesn/v5_1_r2/gradient_probe_r2.json

# 3) fold0 screen
python scripts/campaigns/run_v5_1_r2.py --stage screen \
  --plan outputs/pesn/v5_1_r2/plans/screen.json

# 4) dev 复现 + ramp
python scripts/campaigns/run_v5_1_r2.py --stage dev \
  --plan outputs/pesn/v5_1_r2/plans/dev.json
python scripts/campaigns/run_v5_1_r2.py --stage ramp \
  --plan outputs/pesn/v5_1_r2/plans/ramp.json

# 5) confirm（仅 dev-positive）
python scripts/campaigns/run_v5_1_r2.py --stage confirm \
  --arm <top1> --plan outputs/pesn/v5_1_r2/plans/confirm.json

# 6) 报告
python scripts/campaigns/r2_report.py \
  --root outputs/pesn/v5_1_r2 --out outputs/pesn/v5_1_r2
```

### 12.5 计划 JSON 示例

```json
// screen.json
{"arms": ["S2", "S10", "D2", "D10"],
 "folds": [0], "seeds": [17, 29, 43],
 "updates": 3000, "mode": "v51"}

// dev.json
{"arms": ["<top1>", "<top2>"],
 "folds": [1, 2], "seeds": [17, 29, 43],
 "updates": 3000, "mode": "v51"}

// ramp.json
{"arms": ["<best_loss>"], "ramp_arms": ["R1", "R2"],
 "folds": [0], "seeds": [17, 29, 43],
 "updates": 3000, "mode": "v51"}

// confirm.json
{"arms": ["<top1>"], "folds": [3, 4],
 "seeds": [17, 29, 43], "updates": 3000, "mode": "v51"}
```

### 12.6 完成定义（DoD）

- [ ] R1 已提交/打 tag；R1 产物只读；`outputs/pesn/v5_1_r1/` 无新增写入。
- [ ] `protocol_r2.json` 在首次 R2 训练前写入；hash 与 config/split hash 一并记录。
- [ ] R2 loss schema v4 单测通过；λ=0 时与旧实现 bit-identical。
- [ ] 梯度探针输出 encoder/head 两组 ρ；λ 已按 2%/10% 校准；不可达有明确记录。
- [ ] S1 342/342 分类完成；2022 有处置；gauge 有 ADR-0027 结论；无未关闭 P0。
- [ ] screen/dev/ramp/confirm 的 run 均有 checkpoint、eval.jsonl、metrics.json、run_record.json、console.log。
- [ ] `gate_report_r2` 给出四终局之一；`frozen_candidate_r2` 明确 `approved_for_v5_1b=false`。
- [ ] 未打开 sealed/Confirm/OOD；未启动 Fit150+；未在确认折后调参。
- [ ] 文档台账、G–V 注册、V5.2 输入清单同步。

### 12.7 交接给 V5.2

R2 结束后，V5.2 只接收以下四项，不接收旧 λ/旧分数：

1. 已验证的无量纲损失形式与梯度占比纪律；
2. ramp/checkpoint 规则（含失败模式）；
3. S1 数据质量结论与 K 掩码/标签权重规则；
4. 固定 hard-cluster 复测集与旧 V5.1 对照基线。

V5.2 不得用 R2 的旧任务分数作为新任务门槛，也不得用旧 V5.1a/R1 checkpoint 初始化新 Key-Atom 模型。


