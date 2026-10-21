# ADR-0031：MOVE–PROCESS 统一学习合同

> 状态：**Accepted**（2026-10-21 用户裁定：V5.2 关闭，正式进入 V5.3-MP 大规模联合训练）
> 日期：2026-10-21
> 层级：G3.P.1–G3.P.3（V5.3-MP）；取代 [FullPathNavigationContract v1](../../contracts/PES2TS_FullPathNavigationContract_v1.md) 的任务语义
> 上游证据：[V5.2 任务纠偏审计](../../reports/PES2TS_V52_任务纠偏审计_20261021.md)、[V5.2-R 输入充分性决策记录](../../reports/PES2TS_V52R_输入充分性决策记录_20261021.md)
> 下游消费者：V5.3-MP（S0–S4）、V6/V7、全部开发文档与代码命名
> 唯一任务合同：[PES2TS_MoveProcessContract_v1](../../contracts/PES2TS_MoveProcessContract_v1.md)

---

## 1. 背景

1. **V5.2 已关闭**：旧 TS 邻域 Stop 任务是任务定义错误（终点行被 `label_statuses=("ok",)` 排除），G3 = FAIL(E-NO-GAIN)；该负结果对旧任务成立，保留只读。
2. **V5.2-R 已关闭**：D0 PASS、D1 FAIL、D0.1 PASS、D2 P1 PASS / P4 FAIL / P5 FAIL；D2-G 裁定 **STOP**，T1 不授权。旧门控继续对旧 T1 有效。
3. **第二任务不应被简化为“单参考构象 0.10 Å 分类”**：D2 证明目标几何信息价值巨大（P1 AUPRC 0.982），但合法单帧输入下的旧 endpoint 特征族未获可学证据（P0 0.336、P4 0.252、P5 0.239）。问题不在“是否要第二任务”，而在第二任务的**定义、监督与数据规模**。
4. 因此把第二任务从 **STOP Head（TS 邻域停止）** 重新定位为 **PROCESS Head（反应路径进程与完成状态）**，与 MOVE 共享同一个 EGNN。

## 2. 决策

采用 **PES2TS V5 Unified MOVE–PROCESS**：

\[
\pi_\theta(X_t,G_{\rm rxn},H)\rightarrow(\Delta X_{K,t},\mathcal P_t),
\qquad
\mathcal P_t=(\hat q_t,\hat p_{{\rm end},t},s_t)
\]

- **MOVE**：关键原子下一步三维位移 \(\Delta X_{K,t}\)，沿指定 Reaction Intent 推进；
- **PROCESS**：当前构型在有向完整路径中的位置（\(\hat q\)）、目标端点完成置信度（\(\hat p_{\rm end}\)）、排序分数（\(s\)）；
- **Move/No-Move**：由 \(\hat p_{\rm end}\) 与目标化学状态一致性共同决定，不由 \(\hat q\) 阈值决定；
- **TS 是途经点**，永远不触发 No-Move；
- 一个共享 EGNN、两个功能头，**不是三个独立研究任务**。

## 3. 任务定义

| 概念 | 定义 |
|---|---|
| Forward intent | R → TS → P |
| Reverse intent | P → TS → R |
| MOVE | 沿 intent 继续完整 IRC，允许且必须跨越 TS |
| No-Move | 目标端点完成判断成立，选中坐标 = 当前 \(X_t\) |
| PROCESS | Reaction Process State：位置 + 阶段 + 端点完成状态 |
| \(\hat q\) | 有向分段进度，A=−1 / TS=0 / B=+1；reverse 取 −q；**仅辅助/诊断** |
| \(\hat p_{\rm end}\) | 目标端点完成置信度；**唯一 Stop 触发来源** |
| 化学状态一致 | 只用当前几何 + 合法有向拓扑可计算的键编辑完成/价键/几何合法性检查 |

**输入边界不变**：当前完整构型 + 合法有向反应图/约束；禁止历史帧、参考 TS 坐标、参考端点 XYZ、真实 q、未来帧、能量/梯度。

## 4. 模型

| 组件 | 决策 |
|---|---|
| 主干 | `StarEGNNV3`（55,441 参数）保留，S1–S2 不扩网络 |
| MOVE | 现有 `u` 通路，语义 = R3 Full-Traversal |
| PROCESS | 新增 `ReactionProcessHead`，输入共享 `z [G,64]`（可选 key-pooled `h [G,48]`），输出 `q_hat [G]`、`endpoint_logit [G]`、`order_score [G]` |
| 旧 `ProgressHeadV3` | 可作为 q 子头复用；不得改变其输入/输出语义 |
| 旧 `TSStopHeadV3` | **LEGACY_TS_STOP**：只读旧 checkpoint，禁止新训练/新推理使用 |
| 版本号 | `pesn_key_atom_v53mp_heads_v1` |
| 导出 | 新增 `pes2ts_move_process_bundle_v1`（u + q/endpoint 诊断 + 冻结 τ_end + provenance）；现有 key-atom 导出缺失，必须补齐 |

## 5. 标签与 eligibility

- **Move eligibility 与 PROCESS eligibility 完全独立**。
- 数据集必须加载全部 status（`ok`/`terminal`/`masked`/`conflict`），按任务使用不同 mask；**禁止**再用 `label_statuses=("ok",)` 决定 PROCESS 可见样本。
- **Move**：R3 full-traversal first-contact 教师（d0=0.08 Å），全部 ok 行、两个 intent、全部节点，允许跨 TS。
- **Progress**：分段 signed q + 同 curve/intent 的 directed pairwise ordering（允许跨 TS 构对）。
- **Endpoint completion**：高置信度端点锚点 + 明确负例（起点、TS、未完成中段）+ 歧义 Mask；必须附键编辑完成与几何合法性检查。旧的单构象 `d_end` 与 P5 basin **仅作诊断量**，不得自动升级为化学终点真值。
- 端点完成监督是**待验证的学习目标**，不是 D2-G 已证明的能力；新战役是在检验“更多独立反应 + 共享路径表征”能否解决它。

## 6. 损失

\[
L_{\rm total}=L_{\rm Move}+L_{\rm Process}
\]

- \(L_{\rm Move}\)：沿用 R3 已验证组合（atom + pair + cosine + strong + dead），不首次扩量时更换网络/教师/损失。
- \(L_{\rm Process}=\lambda_r L_{\rm ranking}+\lambda_q L_{\rm progress}+\lambda_e L_{\rm endpoint}\)。
- **ranking 为主监督**；progress 为辅助校准（Huber，低权重起步）；endpoint 为独立 BCEWithLogits。
- 每项独立 mask 归一化；λ 用 encoder 梯度占比校准（2–10% 探索范围），不做大网格。
- **Move guard ≤2%** 为硬门；破防记为 `scientific_failure`，不得丢弃、不得标 `invalid`。`invalid_run` 只保留给泄漏/协议违规/哈希不符/运行时错误。

## 7. 在线决策

\[
\mathrm{No\!-\!Move}=[\hat p_{\rm end}\ge\tau_{\rm end}]\ \land\ [\text{目标化学状态一致}]
\]

- \(\tau_{\rm end}\) 在 inner validation 校准后冻结；
- \(\hat q\) 用于解释位置、监测单调性与诊断，**不得**用 `q>0.95` 直接停止；
- 碰撞、输入非法、模型不确定、预算耗尽由外部控制器记录为独立事件，**不得**伪装成模型 No-Move。

## 8. 与旧合同的关系

| 文档 | 处置 |
|---|---|
| FullPathNavigationContract v1 | **Superseded by ADR-0031 / MoveProcessContract v1**；保留只读与 ADR-0030 证据链 |
| ADR-0030 | **Partially superseded**：Endpoint Stop 的单一 `d_end` 语义由 PROCESS 取代；任务纠偏审计结论继续有效 |
| D2-G 决策记录 | **继续有效，仅对旧 T1**；新 V5.3-MP 是独立预注册战役，不得靠重命名绕开原门控 |
| V5.2-R 计划 | **Closed**；D0/D1/D0.1/D2 证据保留，T1 不授权 |
| 30h 计划 V5.3 rollout | 记为 **V5.3-ENG**（工程闭环，历史）；新战役记为 **V5.3-MP** |

## 9. 数据规模与阶段

| 阶段 | 独立反应规模 | 任务 | 目标 |
|---|---|---|---|
| S0 | 资格审计/建库 | 教师、标签、划分、manifest | 生成合格训练池（硬门） |
| S1 | 2,000 | MOVE + PROCESS 冒烟 | 两头可训、显存/梯度正常、测吞吐 |
| S2 | 5,000 | 初步训练 | 判断数据扩量收益 |
| S3 | 20,000 | 正式联合训练 | 主要科学结果 |
| S4 | 50k–100k | 条件扩量 | 依学习曲线与资格决定 |

- **实测事实**：当前六表训练集只有 50 曲线/12,034 行；`data/ground_truth/irc_index.parquet` 有 199,890 反应/22.98M 帧，但**尚未转成 V5 六表数据**。2k/5k cohort 只有 ID 清单，教师数据未制备。
- 因此 S0 是**新数据构建**，不是配置开关；S1 前必须完成 G0/G1 门、split freeze、leak audit、dataset SHA。

## 10. 门与失败分类

- Move：RMSE/zero、sign、dead 相对匹配 R3 基线；退化 ≤2%（硬门）。
- PROCESS ranking：mean >0.65、跨 TS >0.6（cluster bootstrap）。
- Endpoint：AUPRC ≥2× prevalence 且 CI 下界 > prevalence；起点 FP ≤2%、TS FP ≤2%（冻结 operating point）。
- 闭环：端点到达率、TS 跨越率、TS 候选提取；独立物理验证后才可称 `TS_candidate`，不得称 verified TS。
- 声明边界：只可称 `endpoint_reached` / `TS_candidate`；OptTS + 单虚频 + 双向 IRC + 端点身份确认前，禁止 verified TS/path。

## 11. 代码与文档同步后果

- 新增 `ReactionProcessHead`、`key_atom_v53mp_train_v3.py`、`r3_v53mp_run.py`、导出 bundle；
- 修复 `KeyAtomDatasetV3` 默认 `label_statuses=("ok",)` 与训练入口的 ok-only 过滤；
- 注册表 `config/registry/g_v_map.yaml` 新增 V5.3-MP，V5.3 旧条目改名 V5.3-ENG；
- 宪法升 v2.9；README/AGENT/docs README/台账/G–V map/CODE MAP/阶段定义同步；
- `docs_consistency.py --strict` 训练前强制通过。

## 12. 风险与未决

- **端点标签仍是待验证目标**：S0 必须给出标签有效性审计；未通过不得进入 S3。
- **数据规模是主变量**：30 曲线下的宽 CI 不能外推到 20k；S2 必须报告学习曲线。
- **任务负迁移**：PROCESS 梯度可能损伤 MOVE；Move guard 与梯度占比监控为硬约束。
- **命名冲突**：旧 V5.3 rollout 已占用 V5.3；本 ADR 以 V5.3-MP / V5.3-ENG 区分，注册表与文档必须一致。
- **算力预算未定**：S1 测吞吐后再冻结 S2–S3 上限，不得先承诺 20k 全量。

## 13. 变更记录

| 版本 | 日期 | 内容 |
|---|---|---|
| v1.0 | 2026-10-21 | 初版：V5.2 关闭；MOVE–PROCESS 统一；V5.3-MP 阶段与门；旧合同取代关系 |
