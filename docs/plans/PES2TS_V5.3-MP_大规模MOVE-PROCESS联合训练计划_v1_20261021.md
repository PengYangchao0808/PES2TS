# PES2TS V5.3-MP：大规模 MOVE–PROCESS 联合训练计划 v1

> 状态：**Proposed**（S0 数据资格审计可立即启动；S1–S4 需逐阶段显式授权）
> 日期：2026-10-21 · 版本：v1.0
> 决策记录：[ADR-0031](../design/decisions/ADR-0031-MOVE-PROCESS统一学习合同.md)
> 唯一任务合同：[MoveProcessContract v1](../contracts/PES2TS_MoveProcessContract_v1.md)
> 上游：V5.2 已关闭（[D2-G 决策记录](../reports/PES2TS_V52R_输入充分性决策记录_20261021.md)）；V5.2-R D0/D0.1/D1/D2 证据保留
> 命名：**V5.3-MP** = 本计划；**V5.3-ENG** = 已执行的 30h rollout/relax 工程闭环（历史）

---

## 0. 裁定摘要

| 事项 | 状态 |
|---|---|
| V5.2 / V5.2-R | **关闭**：D2-G STOP，T1 不授权，负结果归档；旧门控继续有效 |
| 进入 V5.3-MP | **批准启动 S0**（CPU 数据资格审计/建库） |
| S1 2k 冒烟 | 待 S0 通过后授权；≤1 GPU-h |
| S2 5k / S3 20k | 逐阶段授权；S3 预算在 S1 实测吞吐后冻结 |
| 输入合同 | 不变：单帧几何 + 有向拓扑；无参考端点 XYZ / 历史 / 真值 |
| 旧 endpoint 标签 | 降级为诊断量；不作为新训练的唯一真值 |
| GPU | 当前 0；S0 全程 CPU |

**一句话**：V5.3-MP 不是把旧 Stop Head 改名，而是用**数万条独立反应**检验“共享 EGNN 能否同时学会怎么走（MOVE）和走到哪/何时完成（PROCESS）”。

---

## 1. V5.2 完成确认

| 证据 | 结论 |
|---|---|
| D0 PASS | 新 endpoint 标签 1781 正例；terminal 行纳入；起点 FP=0 |
| D1 FAIL | 冻结 R3 表征无 endpoint/ordering 信号（AUROC≈0.47–0.51） |
| D0.1 PASS | 22/22 拓扑异常 = T（键编辑未完成）→ Mask；`d_end_sym` 翻转 0/1781 |
| D2 P1 PASS / P4 FAIL / P5 FAIL | 目标几何信息价值巨大（+0.678），合法单帧 no-goal 特征族未获可学证据 |
| D2-G | **STOP**：T1 不授权；P5 标签不冻结；GPU 消耗 0 |
| 代码/文档 | 155 项 V5.2-R 测试通过；docs gate 130 文件 CLEAN；`master` 与 origin 同步 |

**结论**：V5.2 的**执行与证据链已完成**，作为负结果归档。两处编辑性修正随本计划一并落地（不改变裁定）：
1. “task-chem 不可学”改为“在 n=30 dev 曲线与被测 no-goal 特征族下未获可学证据”；
2. P5 的 `endpoints_outside` 失败明确为“TS 落入 0.30 Å basin 的 Mask 带，无正例污染”；P3 的 LEAK 标为 `DEGENERATE_TRIGGER`。

---

## 2. 数据现状（实测，2026-10-21）

| 资产 | 数量 | 是否可直接训练 |
|---|---:|---|
| 当前 V5 六表训练集 | **50 曲线 / 6,017 节点 / 12,034 样本** | 是（仅 50 曲线，dev 30 / sealed 20） |
| `data/ground_truth/irc_index.parquet` | **199,890 反应 / 22,983,028 帧** | **否**：未转六表、未过 G0/G1、未冻结 split |
| `data/interim/split_assignment.parquet` | 199,168（train 159,355 / valid 19,904 / test 19,909） | **否**：旧 split 语义，非 V5 split |
| `irc_fit_2000_verified` cohort | 2,000 IDs（1,995 canonical groups） | **否**：只有 ID |
| `irc_fit_5000` cohort | 5,000 IDs（4,994 canonical groups） | **否**：只有 ID |
| fit 直接冻结集 | 4,964 groups；fit50 已制备 | 部分；需重建 V5 六表 |

**关键结论**：2k/5k/20k 不是配置开关，而是**三次数据构建**。S0 必须先回答“有多少条完整 IRC 能通过资格审计”，再谈 GPU。

---

## 3. 目标架构

```
X_t (all atoms) + G_rxn (G_src, G_tgt, K, CV, rules) + H
                    │
             StarEGNNV3 (55,441 params, shared)
                    │
        ┌───────────┴────────────┐
        │                        │
   MOVE head                 PROCESS head
   ΔX_K [ΣK,3]        q̂ [G], p̂_end [G], s [G]
        │                        │
        └──────────┬─────────────┘
                   │
        MOVE / No-Move:
        [p̂_end ≥ τ_end] ∧ [chemical state consistent]
```

- 主干不扩；PROCESS 头控制在 ~10^3–10^4 参数；
- q 只作辅助/诊断；No-Move 只由 endpoint 完成 + 化学一致性决定；
- TS 永远继续 MOVE。

---

## 4. S0 — 数据资格审计与建库（CPU，硬门）

### 4.1 资格审计（先做，不写数据）

1. 从 199,890 反应中筛出：完整 IRC、canonical 化、R/TS/P 身份齐全、无映射冲突、无几何退化；
2. 复用/扩展 G0/G1 审计：原子序、键图、端点身份、split 资格、truth quarantine；
3. 输出分层计数：total / eligible / excluded(reason) / 可用于 MOVE / 可用于 PROCESS；
4. **禁止**把旧 `split_assignment.parquet` 直接当 V5 split；必须新建 curve-level、reaction-family-aware 的冻结 split；
5. 目标 cohort：2k（S1）→ 5k（S2）→ 20k（S3）；每级都必须有独立 manifest SHA。

### 4.2 建库（六表）

- 扩展 `scripts/agent_ops/data_gate/r3_build_tables.py` 到 cohort；
- 生成 `irc_curve` / `irc_node` / `irc_edge` / `reaction_intent` / `curve_quality` / `sample_view`；
- 每个样本三套标签：
  - `move_target_u`（R3 full-traversal 教师）；
  - `q_signed` + ordering pairs；
  - `endpoint_completion_label`（正/负/Mask，附键编辑完成与几何合法性）；
- **关键修复**：数据集加载全部 status（`ok/terminal/masked/conflict`），MOVE 与 PROCESS 使用独立 mask；禁止 `label_statuses=("ok",)` 决定 PROCESS 可见性。
- 输出：`data_manifest.json`（dataset_sha256）、split manifest、label spec SHA、leak audit、truth log（空）。

### 4.3 S0 门（全部通过才进 S1）

| 检查 | 判据 |
|---|---|
| 资格审计 | 2k cohort ≥1,800 条完整 IRC；排除原因逐条可复现 |
| split | curve-level 单一折；reaction-family 不跨折；dev/sealed 冻结 |
| 泄漏 | truth 访问日志为空；未来帧/参考端点未进入 forward |
| 标签 | endpoint 正/负/Mask 统计合理；硬负例（起点/TS）不误正 |
| 结构保护 | 数据集拒绝构建“缺 terminal 行”的 PROCESS 表 |
| 文档 | docs gate CLEAN；本计划与合同已登记 |

**S0 不过 → 不进 S1，不买 GPU。**

---

## 5. S1–S4 训练阶段

| 阶段 | 规模 | 目的 | 预算（上限） | 晋级条件 |
|---|---:|---|---|---|
| **S1 冒烟** | 2k | 两头可训、显存/梯度正常、测吞吐 | ≤1 GPU-h | loss 有限；两头都有梯度；Move 不崩；吞吐可外推 |
| **S2 初步** | 5k | 数据扩量收益 | 由 S1 实测外推，需单独授权 | S2 相对 S1 的 val 指标有可解释提升 |
| **S3 正式** | 20k | 主要科学结果 | S1 实测后冻结，需单独授权 | 预注册门（§7） |
| **S4 条件扩量** | 50k–100k | 学习曲线饱和判断 | 另议 | S3 通过且曲线未饱和 |

- 每阶段三个固定臂：**A = Move-only**、**B = Move + Progress（ranking+q）**、**C = Move + PROCESS（B + endpoint）**；
- S1：1 seed、短日程、只做工程判定；S2：2 seeds；S3：3 seeds（17/29/43）；
- 所有阶段 dev/sealed 边界沿用 S0 冻结 split；sealed 只在最终评估使用；
- 任何阶段不得以“run 数量”代替统计证据。

---

## 6. 损失与优化

\[
L_{\rm total}=L_{\rm Move}+L_{\rm Process}
\]

- \(L_{\rm Move}\)：R3 已验证组合（atom + pair + cosine + strong + dead）；S1 不更换；
- \(L_{\rm Process}=\lambda_rL_{\rm ranking}+\lambda_qL_{\rm progress}+\lambda_eL_{\rm endpoint}\)；
- ranking 主监督；progress 辅助（Huber，低权重）；endpoint BCEWithLogits + 类别权重 + 独立 Mask；
- λ 用 encoder 梯度占比校准（2–10%），每阶段记录；
- **Move guard ≤2%**：破防 = `scientific_failure`，保留 run、分析、不得丢弃；`invalid_run` 仅限泄漏/协议/哈希/运行时错误；
- 梯度冲突监控：Move/Process 梯度余弦、encoder 梯度占比、任务间 loss 权重敏感性。

---

## 7. 预注册与门

### 7.1 主比较

| 问题 | 比较 | 门槛 |
|---|---|---|
| PROCESS 是否损伤 MOVE | C vs A | RMSE/zero 退化 ≤2%（硬门） |
| Progress 是否有价值 | B vs A | ordering mean >0.65、跨 TS >0.6；ΔAUPRC/ RMSE 配对 CI 下界 >0 |
| Endpoint 是否有增量 | C vs B | AUPRC ≥2× prevalence 且 CI 下界 > prevalence；起点 FP ≤2%、TS FP ≤2% |
| 数据扩量是否有收益 | S2/S3 vs S1 | val 指标 paired Δ 的 CI 下界 >0，且不是单折/单 seed |

### 7.2 闭环评估（S3）

- 从未参与训练的曲线端点出发，按长度感知预算 \(B=\max(40,\lceil1.5L_{\rm key}/0.08\rceil)\) 执行 MOVE；
- PROCESS 触发 No-Move 或预算耗尽停止；
- 记录：端点到达率、TS 跨越率、首次 No-Move 距离分布、碰撞/非法事件；
- TS 候选交独立物理验证（OptTS + 单虚频 + 双向 IRC + 端点身份）；此前只称 `TS_candidate`。

### 7.3 声明边界

- 只可称 `endpoint_reached` / `TS_candidate`；
- 不得称 verified TS/path；
- 旧 D2-G 负结果对旧 T1 继续有效；V5.3-MP 是独立预注册战役，不得靠重命名绕开原门控。

---

## 8. 代码修改清单（file-level）

| 文件 | 修改 |
|---|---|
| `pes2ts_core/pesn/models/key_atom_heads_v3.py` | 保留 `ProgressHeadV3`；新增 `ReactionProcessHead`；`TSStopHeadV3` 标 LEGACY；新增 `KeyAtomV53MPModel` 或扩展 wrapper |
| `pes2ts_core/pesn/models/key_atom_loss_v53mp_v3.py`（新） | `L_Move + L_Process`，独立 mask，ranking/progress/endpoint |
| `pes2ts_core/pesn/offline/training/key_atom_v53mp_train_v3.py`（新） | 分阶段 trainer；3 arms；Move guard；梯度监控；resume |
| `pes2ts_core/pesn/offline/data/key_atom_dataset_v3.py` | 默认 status 改为全量或显式传参；新增 `v53mp_labels()`；MOVE/PROCESS 独立 eligibility |
| `pes2ts_core/pesn/offline/data/key_atom_endpoint_labels_v1.py` | 保留为诊断标签；新增 high-confidence endpoint anchors + hard negatives + Mask |
| `scripts/campaigns/r3_v53mp_run.py`（新） | 训练驱动；ledger/runplan/gpu lock/monitor；禁止绕过 `assert_campaign_root` |
| `scripts/agent_ops/data_gate/r3_build_tables.py` | 扩展 cohort 建库；输出三套标签与新 manifest |
| `scripts/agent_ops/runplan/build_r3_runplan.py` | 新 campaign run plan；预算/阶段绑定 |
| `pes2ts_core/pesn/offline/publishing/export.py` | 新增 `pes2ts_move_process_bundle_v1`（当前 key-atom 导出缺失） |
| `config/campaigns/pes2ts_v53mp_20261021.yaml`（新） | S0–S4、arms A/B/C、授权位、freeze hashes、预算上限 |
| `config/registry/g_v_map.yaml` | V5.3 → V5.3-ENG（executed）；新增 V5.3-MP（proposed）；gates/code_scope/config |
| `tests/pesn/test_v53mp_heads.py`（新） | 头输出形状、主干 bitwise 不变、LEGACY 禁用 |
| `tests/pesn/test_v53mp_labels.py`（新） | 独立 eligibility、terminal 进入 PROCESS、TS 必须 MOVE、无泄漏 |
| `tests/pesn/test_v53mp_trainer.py`（新） | 分阶段、Move guard、失败分类、resume |
| `tests/agent_ops/test_v53mp_runplan.py`（新） | 预算/阶段/授权/哈希绑定 |

**必须先修的两处旧代码**：
1. `scripts/campaigns/r3_v52_run.py:158-160` 与 `run_v5_1_r3.py:109` 的 `label_statuses=("ok",)`；
2. `key_atom_dataset_v3.py:182` 默认值 + `:411-413` 的 status 丢弃逻辑（改为任务级 mask，而不是加载级丢弃）。

---

## 9. 文档同步清单（详见独立清单）

本计划配套 `PES2TS_V5.3-MP_文档同步清单_v1_20261021.md`，同步顺序：

1. 宪法 v2.9 → 2. ADR-0031 / MoveProcessContract v1 → 3. 旧合同/ADR 的 SUPERSEDED 横幅 → 4. README/AGENT → 5. docs README/台账 → 6. registry + G–V map 重渲染 → 7. CODE MAP/阶段定义 → 8. 旧计划指针 → 9. docs_consistency 配置 → 10. `--strict` 门。

**SSOT 链**：宪法 v2.9 → ADR-0031 → MoveProcessContract v1 → 本计划。其他文档只引用，不重定义。

---

## 10. 预算与时间线

| 阶段 | 时间 | GPU | 备注 |
|---|---|---|---|
| S0 资格审计 + 2k 建库 | 2–4 天（CPU） | 0 | 先跑资格审计，再决定建库规模 |
| S1 2k 冒烟 | 0.5–1 天 | ≤1 GPU-h | 测吞吐；不追求科学结论 |
| S2 5k | 1–2 天 | S1 外推后冻结 | 需单独授权 |
| S3 20k | 3–5 天 | 单独授权 | 主要科学结果 |
| S4 50k–100k | 待定 | 另议 | 仅当 S3 曲线未饱和 |

**预算原则**：S1 实测 updates/s、显存、每 epoch 成本后，才允许冻结 S2/S3 上限；禁止先承诺 20k 全量。

---

## 11. 风险与止损

| 风险 | 缓解/止损 |
|---|---|
| 2k cohort 资格不足 | S0 直接停；不降级到 50 曲线硬跑 |
| 建库泄漏/端点身份错误 | G0/G1 审计 + truth log + 独立复核；不过不进 S1 |
| PROCESS 损伤 MOVE | Move guard ≤2% 硬门；C vs A 为主比较；破防记 scientific_failure |
| Endpoint 标签仍不可学 | S2 报告端点门；未过则保留 B 臂（Move+Progress），不强行宣称 No-Move |
| 数据扩量无收益 | S2 学习曲线；无提升则停 S3，写负结果 |
| 命名歧义 | V5.3-ENG / V5.3-MP 强制区分；注册表与文档一致 |
| GPU 预算失控 | 每阶段硬上限；S1 实测后才冻结后续 |

---

## 12. 请求授权

请确认以下四项后启动：

1. **S0 立即启动**（CPU，2–4 天，0 GPU）；
2. **S1 2k 冒烟上限 ≤1 GPU-h**（S0 通过后执行）；
3. **命名**：新战役 `V5.3-MP`，旧 rollout 记 `V5.3-ENG`；
4. **文档同步**：按 §9 清单一次性执行，宪法升 v2.9，旧合同/ADR 加 SUPERSEDED 横幅，docs gate 必须 CLEAN。

S2/S3/S4 不在本次授权内，逐阶段另行裁定。

---

## 13. 变更记录

| 版本 | 日期 | 内容 |
|---|---|---|
| v1.0 | 2026-10-21 | 初版：V5.2 关闭确认；MOVE–PROCESS 架构；S0–S4；三臂 A/B/C；数据/代码/文档清单 |
