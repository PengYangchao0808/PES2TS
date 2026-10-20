# V5.2 Progress/Stop 报告 — PES2TS_30H_R3_V52_V53

> 生成：2026-10-20T21:25Z · 编制：supervisor · G3-V52Supervision = **FAIL（E-NO-GAIN，诚实科学结论）**
> 证据：`outputs/pesn/30h_campaign/v52_runs/g3_evidence_pack.json` + 9 个有效 run 的 metrics.json（已抽查复核）

## 1. 执行摘要

V5.2 在冻结配置下完成实现、校准与训练，结果是**明确的负结果（no-gain）**：Progress 头不可学习，Stop 头处于随机水平，联合训练（QT）反而损伤 Move 能力（护栏 ≤2% 被 4/6 run 破防，中位 +5.5%）。按用户指令，此结论作为真实研究结论保存，未做任何调参消除尝试。

## 2. 训练配置（全部冻结）

- backbone：R3 winner（F2-bidir）checkpoint 逐比特初始化（fold+seed 匹配，strict load）
- 日程：冻结 encoder 400 updates → 联合 1500 updates；lr head 3e-4 / encoder 1e-4（一次 yaml 许可的稳定性 smoke 后固定）
- λ 校准（plan §5.4 形式体系，encoder 梯度占比 2% 目标）：**λq=0.4210，λm=0.0132**（唯一 λ 来源，无猜测）
- stop 类权重：仅 inner-train 估计并冻结（16.9–20.3，与 G3a 报告逐项断言一致）
- Mask 语义：不进损失分母（机器检查）

## 3. 预算与事故（全披露）

- **32/33 attempts**：16 done（**9 有效 + 7 无效**）+ 1 interrupted + 6 QT-confirm blocked + 1 held（未花）
- 事故 1：LAMBDA_RANGE 校验 bug（off-term λ=0 被错误拒绝）在训练开始前烧掉 15 attempts → 修复 + 回归测试
- 事故 2：v1 split 在 `fold≠F` 上训练，将确认折 3/4 泄入 8 个 done runs → **结果观察前自判无效**（invalid_runs.json），修复为 E4 一致的 sealed split（train_folds [1,2]/[0,2]，SPLIT_SEAL 断言）
- salvage 集（结果观察前预注册）：QT fold0–1×3 + T fold0×3 = 9 runs，全部封存验证通过

## 4. 结果（9 有效 runs）

### Progress 头（q）
| 指标 | 值 | 结论 |
|---|---|---|
| q MAE（曲线级均值） | 0.507 | **劣于常数基线 0.476** |
| q 相关（曲线级中位） | ≈ −0.01（均值 −0.11） | 信号缺失 |
| learnable_above_constant | false（全部） | **不可学习** |

### Stop 头
| 指标 | T-only | QT |
|---|---|---|
| AUC | 0.491（随机） | 中位 0.527（5/6 勉强 >0.5） |
| recall@0.5 | — | **0.0**（无可用工作点） |

### Move 非退化护栏（vs R3 winner 同折同种子）
- QT 6 runs：4 破防；RMSE 退化中位 **+5.5%**、最大 **+9.3%**（护栏 ≤2%）
- T-only：Move 未损伤（单头训练不干扰编码器）

## 5. 科学解读（假设，非结论）

1. **q 标签可能本质上不可从 (X, G_intent) 预测**：q 由 canonical IRC 弧长分段定义，局部几何+意图也许不携带足够信息（尤其远离 TS 的平坦区）。q-teacher 信息量审计（互信息/可达性）是明天的关键诊断。
2. **λq=0.421 偏强 + 1900 updates 联合训练损伤 encoder**：Move +5.5% 退化与该假设一致。λq 消融（0.421→0.1→0.02）在 fold0 上 ~6 个 run 即可回答——但这属超参修订，需你授权。
3. Stop AUC≈0.5 与"d_key_TS≤0.04Å 窗口在 9655 样本中仅 ~5%"的小样本结构一致：正例太少且几何上难以区分。



## 5.1 战役后追加诊断（supervisor，analyze-completed-experiments 授权内，无新训练）

两个只读探针（`outputs/pesn/30h_campaign/stats/q_informativeness_probe.json`、`q_doublevaluedness_check.json`）：

1. **线性探针**（5 折 curve 分组 ridge，仅合法几何：key-pair 距离 + intent 符号投影）：MAE 0.4722 vs 常数基线 0.4724——**改进 0.0%，pearson 0.018**。
2. **双值性检查**：within-(curve,intent) |corr(q, d̄_key)| 中位 **0.743**（q25 0.449 / q75 0.944）；全局池化 corr **−0.045**。

**结论：H1 从 leaning 升级为强证据——弧长 q 在 (X_t, G_intent) 单帧合法输入下不可辨识**。同一 CV 距离剖面在 TS 两侧对应相反 q、跨反应对应不同 q；预测 q 需要路径身份/历史——恰是宪法禁止的输入。V5.2 progress 头的失败是**任务定义层面的不可辨识**，非训练配置错误（H2 降为次要；λq/日程问题仅解释 Move 退化部分）。架构无关论证：任何单帧预测器面对同一非单值映射。

**对 V6 的含义**：进度表征需重设计——如以 d_key_TS（局部可观测）+ 由 u 预测隐含的方向估计构造"相对进度"，或放弃单帧 q 改用序列级进度（需修改输入合同，需你裁定）。

## 6. H20 后果

V5.3 按 H20 以**上限控制器模式**执行（classic + R3 winner），v52 臂全程 diagnostic-only 标注（`v52_supervision_failed_no_claim`，机器强制）。详见 V53_ROLLOUT_REPORT.md。

## 7. 交付状态分类

- 实现：**implemented**（trainer/driver/probe/evidence 全套 + 10 项新测试）
- 测试：fixture_passed（471 agent_ops 全绿）+ 真实数据训练执行（9 valid runs）
- 科学结论：**scientifically_validated NEGATIVE result**（冻结配置下 V5.2 无增益；样本 9 runs × 2 折，种子 17/29/43）
