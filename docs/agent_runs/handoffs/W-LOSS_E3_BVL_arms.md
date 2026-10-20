# W-LOSS · E3-CONT — λ 校准探针 + B/V/L 臂执行 + G2 dev-fold 证据包

> 状态：**IMPLEMENTED + EXECUTED** · 2026-10-20T19:10Z · 分支 `ops/e3-cont` · lane W-LOSS wave 3
> 授权：`formal_training_authorized=true`（至 2026-10-21T13:45Z 硬停）；confirm/sealed/OOD/fit150+ 恒 false
> 预注册选择（supervisor 排名，fold0 screening，eligible set）：**Top-1 = F2-bidir，Top-2 = F3**
> 账本：本波后 **61/66** R3 launch attempts（见 §5 事件披露：6 次 sealed-fold 收尾重试）；C×6 + R×6 blocked 未发射

---

## 1. λ 梯度校准探针（`scripts/campaigns/r3_gradient_probe.py`，plan §5.4 / yaml aux_calibration）

**参考模型**：B0-F = 预注册 Top-1 winner F2-bidir fold0 checkpoint_best，逐 seed 匹配（s17=5372d9b1…、s29=1e01499b…、s43=9e2ef3ef…）。**20 个真实 inner-train 批**（与 B0-F 训练同管线：F2 特征、bidirectional 采样、同 inner 切分与种子）。梯度隔离只用冻结 loss：`∇L_k = ∇total(λ=1) − ∇total(λ=0)`（梯度线性；无任何 loss 重实现）。λ_cos 固定 R0 终值 0.02。

| seed | strong ρ_enc（λ=1 中位） | strong ρ_head | dead ρ_enc | dead ρ_head |
|---|---|---|---|---|
| 17 | 1.0495 | 1.0480 | 867.2 | 1013.4 |
| 29 | 1.2623 | 1.0269 | 948.4 | 1328.0 |
| 43 | 0.8875 | 1.0358 | 1227.5 | 1778.0 |

**校准结果**（campaign λ = 跨 seed 中位比例 → λ = target/ρ；约束 λ≤0.5、head share≤0.3、floor 1%）：

| arm | bucket | target | **λ（校准）** | 达成 encoder share | verdict | head shares @λ | head flag |
|---|---|---|---|---|---|---|---|
| S2 | strong | 2% | **0.019056400705052445** | 0.020 | calibrated | 0.020/0.020/0.020 | ok |
| S10 | strong | 10% | **0.09528200352526223** | 0.100 | calibrated | 0.100/0.098/0.099 | ok |
| D2 | dead | 2% | **2.1087830388997296e-05** | 0.020 | calibrated | 0.021/0.028/0.038 | ok |
| D10 | dead | 10% | **0.00010543915194498648** | 0.100 | calibrated | 0.107/0.140/0.188 | ok |

- **无 scale_unreachable / target_unreachable 臂**（dead 桶原始梯度比 ~900-1200× 基线 → λ 极小但达到目标；λ≤0.5 上限内）。
- λ=0 回归：loss 级逐位一致（单线程；base vs aux_arm@λ=0 损失值+全部梯度 `torch.equal`）+ `aux_arm_lambdas_v3` 桶放置验证 —— 见测试矩阵 #3。
- 校准文件：`outputs/pesn/30h_campaign/r3_runs/lambda_calibration.json`（逐 seed 比例 + 逐臂 λ/verdict/head share）。
- **解释记录**：supervisor 任务书写 "F0 forward fold0 … (= R3 B0-F reference)" 存在歧义；按等式右侧（B0-F 定义 = winner 特征配置的 B0 基线，已预注册为 F2-bidir）采用 **F2-bidir fold0 逐 seed checkpoint** 作为探针参考——λ 须在与 L 臂同配置（F2-bidir）的批上测才可迁移。若 supervisor 本意为 arm-F0，探针 ~3 分钟可重跑。

## 2. B/V/L 臂执行日志（36/36 done，GPU flock 逐 run 持有）

配置：B = F2-bidir 基损 folds1-4×3（fold0 复用 F-bidir，未重发）；V-top1 = F2-bidir folds1-2×3；V-top2 = F3 folds1-2×3；L = S2/S10/D2/D10 fold0×3（F2-bidir 配置 + 校准 λ；与探针参考同配置）。全部 `monitor.attempt_launch(stage=g0)` 发射。

| run_id | arm | f | s | sel_upd | rmse/zero | sign | SRE | wall s |
|---|---|---|---|---|---|---|---|---|
| cc9340b86592b099 | B | 1 | 17 | 500 | 0.927 | 0.858 | 0.524 | 53.0 |
| 0629b5979df6a2df | B | 1 | 29 | 250 | 0.983 | 0.569 | 0.578 | 55.4 |
| c09e9cd601bafb88 | B | 1 | 43 | 250 | 0.967 | 0.836 | 0.547 | 54.0 |
| 0ebd4c92c85f5601 | B | 2 | 17 | 250 | 1.019 | 0.891 | 0.602 | 51.6 |
| dedd62d588f3b6cd | B | 2 | 29 | 250 | 0.946 | 0.632 | 0.435 | 53.3 |
| b3941e53f44b962b | B | 2 | 43 | 250 | 0.999 | 0.930 | 0.571 | 45.1 |
| 75a847a631219a28 | B **sealed** | 3 | 17 | — | SEALED | SEALED | SEALED | (eval-only finalize) |
| f97e3ed222040521 | B **sealed** | 3 | 29 | — | SEALED | SEALED | SEALED | (eval-only finalize) |
| 8369e58777dddb81 | B **sealed** | 3 | 43 | — | SEALED | SEALED | SEALED | (eval-only finalize) |
| dcd75e4478682b1c | B **sealed** | 4 | 17 | — | SEALED | SEALED | SEALED | (eval-only finalize) |
| 5d4ef03d65185ac1 | B **sealed** | 4 | 29 | — | SEALED | SEALED | SEALED | (eval-only finalize) |
| 9c308d177407bda9 | B **sealed** | 4 | 43 | — | SEALED | SEALED | SEALED | (eval-only finalize) |
| 782c59289443bf03 | V-top1 | 1 | 17 | 500 | 0.927 | 0.858 | 0.524 | 41.6 |
| b50cfc678d2ed634 | V-top1 | 1 | 29 | 250 | 0.983 | 0.569 | 0.578 | 46.4 |
| 9694946ef08ef01a | V-top1 | 1 | 43 | 250 | 0.967 | 0.836 | 0.547 | 56.7 |
| 2bf26953c60d23bf | V-top1 | 2 | 17 | 250 | 1.019 | 0.891 | 0.602 | 50.0 |
| 030045c21327acab | V-top1 | 2 | 29 | 250 | 0.946 | 0.632 | 0.435 | 53.2 |
| 0d84196e1350cb9b | V-top1 | 2 | 43 | 250 | 0.999 | 0.930 | 0.571 | 55.5 |
| 2a6557bcab64352f | V-top2 | 1 | 17 | 250 | 1.010 | 0.735 | 0.600 | 53.6 |
| 795c3a10cd0b23c4 | V-top2 | 1 | 29 | 250 | 1.095 | 0.514 | 0.610 | 49.0 |
| 2f98956c0360804a | V-top2 | 1 | 43 | 250 | 1.050 | 0.589 | 0.413 | 54.5 |
| 6e8852f1a9dc82d0 | V-top2 | 2 | 17 | 250 | 0.980 | 0.891 | 0.583 | 56.1 |
| 029fadcfc6ee850e | V-top2 | 2 | 29 | 250 | 1.055 | 0.660 | 0.537 | 55.9 |
| 6056dc52d01b5eb6 | V-top2 | 2 | 43 | 250 | 1.045 | 0.502 | 0.563 | 60.5 |
| 2c60e697c9c402f8 | S2 | 0 | 17 | 250 | 0.882 | 0.764 | 0.412 | 65.2 |
| 212b3df39b0dfcbc | S2 | 0 | 29 | 250 | 0.896 | 0.888 | 0.453 | 60.1 |
| 7d32f49043ce4cd9 | S2 | 0 | 43 | 750 | 0.946 | 1.000 | 0.313 | 65.9 |
| 32f2c4debd61117a | S10 | 0 | 17 | 250 | 0.874 | 0.764 | 0.413 | 61.9 |
| 26acd7d0b86542b3 | S10 | 0 | 29 | 250 | 0.900 | 0.888 | 0.476 | 68.5 |
| 25c8bd6b67d381b6 | S10 | 0 | 43 | 750 | 0.960 | 1.000 | 0.322 | 60.0 |
| 4a4c56d30eaa6909 | D2 | 0 | 17 | 250 | 0.888 | 0.764 | 0.412 | 66.9 |
| 94d69176c06aaf5a | D2 | 0 | 29 | 250 | 0.896 | 0.888 | 0.449 | 62.4 |
| 424dd9afd5f569a4 | D2 | 0 | 43 | 750 | 0.951 | 1.000 | 0.328 | 63.9 |
| 21ca0877f61a9720 | D10 | 0 | 17 | 250 | 0.900 | 0.764 | 0.411 | 63.0 |
| 223a27271adb6a28 | D10 | 0 | 29 | 250 | 0.899 | 0.888 | 0.443 | 66.7 |
| 3ee78aeff43e99da | D10 | 0 | 43 | 750 | 0.961 | 1.000 | 0.337 | 58.5 |

**V-top1 一致性**：同配置同 seed 的独立重训在上报精度上与 B folds1-2 **逐值一致**（checkpoint 字节不同=序列化非确定性；训练在 float 容差内确定）。Top-1 在 folds1-2 上保持 <1.0（均值 0.974）—— fold0 筛选信号在未见 fold 上成立。V-top2（F3）folds1-2 均值 1.039，与预注册排名一致（Top-1 < Top-2）。L 臂（校准 λ）与 matched B0-F 全部通过 guards（见 §4）。

## 3. 封存确认（plan §4.2 seals — B folds3/4）

- 6 个 sealed run（B f3/f4 ×3 seeds）**metrics 文件已计算并落盘**（`r3_runs/<run_id>/metrics.json`）但**被排除于一切选择/报告聚合**：`r3_g2_evidence.py` 的 `split_entries` 是唯一聚合闸口，sealed 条目只贡献 run_id 审计清单、数值永不进入；证据包 `n_sealed_runs_excluded=6`、`sealed_run_ids` 列出。
- ledger 侧：6 条 done 记录携带 `metrics_seal` 注记（"sealed: B-phase folds3/4 metrics excluded from selection/report aggregation"）+ `metric_unseal` 消费凭据（token `state/gates/metric_unseal.json`，batch=plan 的 B folds3/4 六 run 集，count=1，批内幂等——WP-L3 接线首次真实消费）。
- 选择（G2 winner 判定、任何聚合）只用 dev folds 0-2 —— 证据包 `dev_folds=[0,1,2]`。

## 4. G2 dev-fold 证据包（`r3_g2_evidence.py` → `r3_runs/g2_evidence_pack.json`；48 dev runs，folds3/4 sealed）

| arm | n | folds | mean rmse/zero | median sign | median SRE | sel updates |
|---|---|---|---|---|---|---|
| **B (F2-bidir 基损)** | 6 | 1,2 | **0.974** | 0.847 | 0.559 | [250,500] |
| **V-top1** | 6 | 1,2 | **0.974** | 0.847 | 0.559 | [250,500] |
| V-top2 (F3) | 6 | 1,2 | 1.039 | 0.624 | 0.573 | [250] |
| S2 | 3 | 0 | 0.908 | 0.888 | 0.412 | [250,750] |
| S10 | 3 | 0 | 0.911 | 0.888 | 0.413 | [250,750] |
| D2 | 3 | 0 | 0.912 | 0.888 | 0.412 | [250,750] |
| D10 | 3 | 0 | 0.920 | 0.888 | 0.411 | [250,750] |
| F2-bidir（fold0 筛选） | 3 | 0 | 0.910 | 0.888 | 0.412 | [250,750] |
| F3 | 3 | 0 | 1.107 | 0.417 | 0.501 | [250] |
| F2 | 3 | 0 | 1.154 | 0.462 | 0.577 | [250] |
| F2-intent-off | 3 | 0 | 1.142 | 0.312 | 0.496 | [250] |
| F0 | 3 | 0 | 1.204 | 0.312 | 0.484 | [250,1250] |
| F1 | 3 | 0 | 1.225 | 0.518 | 0.566 | [250,1000] |

**Guards vs matched B0-F**（同 fold+seed；阈值 rmse 退化≤2%、sign drop≤0.01、deadzone rise≤0.01；记录性，不闸门 dev run）：

- **L 臂全部 12/12 通过三项 guard**（rmse 退化 −1.14%…+1.82%，sign drop = 0，deadzone rise ≤ 0）—— 校准 λ 不损伤 winner；S2/S10 有微小改进（负退化）。
- V-top2 vs B：folds1-2 六行中仅 f2-s17 三绿（−3.79%）；其余 rmse 退化 4.5-11.6% —— Top-2 在多数 dev 行上劣于 Top-1（与预注册排名一致，recorded）。
- F 各臂 vs B0-F 全部超 guard（16-50% 退化）—— 预期内（它们是被 winner 击败的筛选臂；recorded，不闸门）。
- 基线表（证据包内）：zero-step（rmse/zero≡1、sign≡0）、classic CV/Jacobian（逐臂 classic sign 中位 0.0-0.125；公式 min-norm 一阶对步）、EGNN=新模型（上表）。
- **诚实 early-overfit 注记**：dev 48 run 中 43 个 inner-val 选择停在 ≤250 updates（7 曲线内层切分的早期过拟合模式；冻结超参，未调参）。

## 5. blocked 臂清单（未发射，无 run 消耗）

| 臂 | runs | blocked_with_reason |
|---|---|---|
| R1/R2 | 6（7071dd9b…/9f5398cf…/695859e3…/4f182fa3…/42a8d106…/bd5a5a8d…） | `spec_gap_ramp_undefined_in_frozen_protocol: R1/R2 ramp parameters are not defined in any frozen doc (yaml r3_training.lambda_cos_ramp=R0 only); supervisor ruling: no invented hyperparameters` |
| C/QT-confirm | 12（wave-1 已 blocked） | `confirm_awaiting_dev_positive` |

无 L 臂 scale_unreachable（探针 verdict 全部 calibrated）。

## 6. 测试矩阵（新增 5 项；全套 `tests/agent_ops/` **449 passed**）

| # | path | 类型 | 结果 |
|---|---|---|---|
| 1 | 探针数学：bucket_ratios（含 ε 除零）、median、calibrate_lambda 五类 verdict（calibrated×多目标 / target_unreachable / scale_unreachable×2 / λ=0 无梯度）、head_flag | positive+rejection | PASS |
| 2 | **λ=0 回归逐位一致**：单线程下 base vs aux_arm@λ=0 同批损失值 `==` + 全梯度 `torch.equal`；`aux_arm_lambdas_v3` 桶放置；随附校准 JSON schema/λ 范围/target 校验 | positive | PASS |
| 3 | **封存语义**：folds3/4 条目被 `split_entries` 剔除（sealed 优秀数值不泄漏进 B 聚合——n_runs/mean/min/folds 断言）、sealed id 审计清单、guards 用 matched B0-F 基线、early-overfit 计数排除 sealed | positive+rejection | PASS |
| 4 | 驱动接线：ARM_MAP B/V-top1/S/D bidirectional 继承、SEALED_FOLDS=(3,4)、**AUX_LAMBDA_REQUIRED 拒绝**（无 --lambda-arm 的 S 臂拒跑）、λ/level/bidirectional 流入 TrainerConfigV3 | rejection+positive | PASS |
| 5 | 探针校准文件（若存在）schema/verdict/λ∈(0,0.5]/target 断言 | positive | PASS |
| 6 | 回归全套 tests/agent_ops | positive | **449 passed** |

## 7. 流程事件披露（透明）

1. **metric_unseal 首次真实消费**：B folds3/4 的 done 转移触发 WP-L3 接线（`METRIC_UNSEAL_NOT_CONFIGURED` 拒绝）——驱动当时未接 token 路径 → 6 个 sealed run 训练完成后 done 被拒（各烧 1 次 launch attempt，留 running→interrupted）。修复：驱动新增 `--metric-unseal`（默认 `state/gates/metric_unseal.json`）；收尾重试（resume 仅重评估 ~40s）完成 done，token 首次消费（count=1，批内 6 run 幂等），凭据入账本。**账本代价：61/66**（supervisor 预算 55 + 6 次 unseal 失败尝试 + 0——队列 tee 掩盖退出码使 RETRY-ONCE 未触发，反而避免了双烧）。
2. 队列脚本 `rc=$?` 取到 tee 的退出码（无 pipefail）——重试逻辑对驱动失败不生效；本次无损害（失败=确定性 unseal 拒绝，重试同样失败），教训入规程：后续队列须 `set -o pipefail`。
3. 探针参考歧义（任务书 "F0 forward" vs "B0-F reference"）：按 B0-F 定义采用 F2-bidir fold0 逐 seed；探针可 ~3 分钟重跑（见 §1 尾注）。

## 8. 下游依赖

- **G2 绑定**（supervisor）：`g2_evidence_pack.json` 为证据包；Top-1=F2-bidir 在 folds1-2 复证（0.974<1.0）；L 臂全绿 guards；sealed 6 run 已隔离。R3 收官待 supervisor G2 判定（66 上限内余 5 次 attempt）。
- **V5.2（E5）**：V5.2 33-run cap 未动；M0 基线复用 R3 winner checkpoint（`reuses` 语义）；QT-confirm 6 仍 blocked 待 `confirm_authorized`。
- **λ 值冻结**：S2=0.019056400705052445 / S10=0.09528200352526223 / D2=2.1087830388997296e-05 / D10=0.00010543915194498648（来源 lambda_calibration.json；任何复用须引用该文件，不得再猜）。
