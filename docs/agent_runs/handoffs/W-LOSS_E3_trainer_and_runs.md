# W-LOSS · E3 — R3 训练循环 + 正式 dev-fold 队列执行

> 状态：**IMPLEMENTED + EXECUTED (fixture_passed → real runs done)** · 2026-10-20T17:12Z · 分支 `ops/e3-trainer` · lane W-LOSS wave 2
> 授权：`formal_training_authorized=true`（2026-10-20T16:55Z 记录，硬停 2026-10-21T13:45Z）；confirm/sealed/OOD/fit150+ 恒 false
> 战役：PES2TS_30H_R3_V52_V53 · plan pin `a8972a19` · GPU RTX 4060 Ti（flock `/tmp/opencode/pes2ts_gpu.lock` 全程持有）

---

## 1. 交付物

| 组件 | 文件 | 说明 |
|---|---|---|
| 训练循环 | `pes2ts_core/pesn/offline/training/key_atom_bc_v3.py` | `TrainerConfigV3`（冻结超参：AdamW 1e-3/wd 1e-4/warmup 250/cosine/clip 1.0/fp32/3000 upd/batch 32=8曲线×4帧）；`train_run`（内层切分、曲线均衡采样、checkpoint by inner-val、resume-by-run-id、metrics.json）；loss **只读消费** `key_atom_loss_v3`（λ_cos=R0 `cos_lambda_r0`，frame_weight=label_weight 在 loss 内恰好施加一次）；模型输入恰为 `forward_view_v3(batch)`；指标用冻结 `evaluate_g52`（key_RMSE_over_zero）+ `evaluate_g53`（SRE/sign/deadzone），按独立 curve/cluster 聚合；zero-step 与 classic CV/Jacobian（min-norm 一阶对投影步，公式记录在 metrics）基线同管线计算，guards 只记录不闸门 |
| 驱动 | `scripts/campaigns/run_v5_1_r3.py` | 子命令 `check`（+历史 `--check` 旗标）/`smoke`（非 ledger 吞吐探针，持锁）/`run`（正式：plan 查行 → 注册 planned → `monitor.attempt_launch(stage=g0)`（pin+五sha+budget，**绝不裸 append running**）→ 持锁训练 → done 转移带 `checkpoint_sha256`+`metrics_path`）/`block-confirm`（12 条 confirm 臂注册+`mark_blocked`）。冻结防呆：正式 updates 恒 3000 不可覆写；S/D 臂无显式 λ 拒跑；R1/R2 拒跑（冻结配置未定义 ramp）；B/L/V 臂需显式 `--level`（G2 选择后） |
| 测试 | `tests/agent_ops/test_gpu_r3_trainer.py` | 9 项（见 §3 测试矩阵） |

## 2. 吞吐 smoke（GPU，持锁 run_id=throughput-smoke-300）

| 项 | 值 |
|---|---|
| updates/s | **43.25**（300 upd / 6.94s 训练壁钟） |
| **t_run(3000) 估计** | **≈69s 训练 + 缓存/评估 ≈ 40-55s → 单 run 总壁钟 ≈ 45-55s**（18 个正式 run 实测均值 ~47s） |
| phase 预算含义 | R3 剩余臂（B 12 + L 12 + V 12 + R 6 = 42 run）≈ 35-40 min GPU；33 个 V5.2 run ≈ 30 min |

## 3. TEST MATRIX（path × 类型 × 结果）

| # | 路径 | 类型 | 结果 |
|---|---|---|---|
| 1 | 配置拒收：batch 组成 32≠8×4 / aux λ>0.5 / 无臂带 λ | rejection | PASS（`test_config_refusals`） |
| 2 | 内层切分：仅 outer-train 簇、train/val 曲线不相交、同 seed 确定、异 seed 等大 | positive | PASS |
| 3 | 采样器：批恒 32 帧（fold0 仅 7 条内层曲线→曲线槽有放回补足，帧数与均衡不变）、forward-only 臂纯 forward、bidir 臂 2+2 曲线内均衡 | positive | PASS |
| 4 | mini-run（30 upd, CPU）：训练 loss 下降、checkpoint 往返加载、best_update=内层 val argmin、val 曲线⊆outer-train、metrics schema 全键、ckpt sha 与 metrics 自洽 | positive | PASS |
| 5 | resume：续训到更长预算 update 正确推进；**异 seed resume 拒**（`RESUME_IDENTITY_MISMATCH`） | positive+rejection | PASS |
| 6 | 确定性：同 seed 两次运行 loss/选择一致（float 容差；CPU 线程非逐位——**如实记录，不宣称 bit-exact**） | positive | PASS |
| 7 | label_weight 恰好一次：spy 捕获 loss 调用，frame_weight 即 batch 的 [G] 权重（不平方不缩放） | positive | PASS |
| 8 | classic Jacobian 基线代数：J u = Δd* 可解子空间精确复现（1e-8） | positive | PASS |
| 9 | S2 臂接线：λ_strong=0.05 λ_dead=0 传入冻结 loss | positive | PASS |
| 10 | 全回归 `tests/agent_ops/` | positive | **433 passed** |
| 11 | pesn 冒烟 `test_r3_model/test_r3_losses`（E2 冻结件未动） | positive | 36 passed |
| 12 | 驱动 CLI 兼容（`--check` 历史旗标 + sealed 拒收） | positive+rejection | PASS（16 driver 测试） |

## 4. 正式 run 日志（18/18 done，19 次 launch attempt（含 F0-s17 一次驱动 bug 重试），R3 cap 66）

| # | run_id | substage | arm | f | s | sel_upd | rmse/zero | sign | SRE | wall s | ckpt sha256[:16] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 358e8a6875ceb3ac | F-main | F0 | 0 | 17 | 250 | 1.030 | 0.125 | 0.647 | 48.5 | ce7fe81e7602ced7 |
| 2 | 4ec075bf61bc0c36 | F-main | F0 | 0 | 29 | 1250 | 1.341 | 0.688 | 0.484 | 42.6 | 26ba6ad632807de8 |
| 3 | 8f2314bf141e6351 | F-main | F0 | 0 | 43 | 250 | 1.243 | 0.312 | 0.327 | 39.2 | ab3a3832db885aa9 |
| 4 | b9fcede32901724f | F-main | F1 | 0 | 17 | 250 | 1.181 | 0.236 | 0.629 | 41.4 | 3997b0718e8c7c53 |
| 5 | d79a2ef0c9cee8ab | F-main | F1 | 0 | 29 | 1000 | 1.292 | 0.518 | 0.566 | 50.3 | 34540e8435b34894 |
| 6 | c70ff14571d6320f | F-main | F1 | 0 | 43 | 250 | 1.203 | 0.625 | 0.404 | 47.5 | 90a37f17a1354ac9 |
| 7 | 67413c8609026299 | F-main | F2 | 0 | 17 | 250 | 1.134 | 0.375 | 0.588 | 39.4 | 919c2ca5d11db556 |
| 8 | 9f698ee3a5835b75 | F-main | F2 | 0 | 29 | 250 | 1.118 | 0.462 | 0.577 | 51.0 | fc656fb971cc560f |
| 9 | 64b509704b8119a2 | F-main | F2 | 0 | 43 | 250 | 1.209 | 0.542 | 0.388 | 51.1 | d120e546d6101de5 |
| 10 | c832d07e1868b886 | F-main | F3 | 0 | 17 | 250 | 1.076 | 0.375 | 0.524 | 52.3 | 991d2c9e57342e14 |
| 11 | 40a7b1a0a927dfc4 | F-main | F3 | 0 | 29 | 250 | 1.075 | 0.707 | 0.501 | 51.0 | 7f2292f9f59a3bc3 |
| 12 | dfa64c2ac03cfcb2 | F-main | F3 | 0 | 43 | 250 | 1.170 | 0.417 | 0.477 | 49.6 | e447e46c62846eaa |
| 13 | 5372d9b12807c3f0 | F-bidir | F2-bidir | 0 | 17 | 250 | **0.884** | 0.764 | 0.412 | 45.8 | a2e46b6ad65cc189 |
| 14 | 1e01499b933d4ad1 | F-bidir | F2-bidir | 0 | 29 | 250 | **0.896** | 0.888 | 0.453 | 45.1 | 357466bf172cff28 |
| 15 | 9e2ef3ef468db0af | F-bidir | F2-bidir | 0 | 43 | 750 | **0.949** | 1.000 | 0.318 | 43.4 | 06b3cbcbbbaf1690 |
| 16 | fec990766056385b | F-ablate | F2-intent-off | 0 | 17 | 250 | 1.127 | 0.236 | 0.517 | 41.1 | 256d4b1d212b9e51 |
| 17 | db1f79764840826b | F-ablate | F2-intent-off | 0 | 29 | 250 | 1.192 | 0.529 | 0.496 | 46.2 | f12f626fcfbf3bcb |
| 18 | 6494413e4dc8ee2c | F-ablate | F2-intent-off | 0 | 43 | 250 | 1.106 | 0.312 | 0.419 | 50.1 | e7bbf7432b972acf |

全部 run：checkpoint + `metrics.json` + `train_log.jsonl` 落盘 `outputs/pesn/30h_campaign/r3_runs/<run_id>/`；账本 `run_records.jsonl` 每 run planned→running→done（done 携 checkpoint_sha256 + metrics_path）；12 条 confirm 臂（C/QT-confirm folds3/4）已注册并 `blocked_with_reason=confirm_awaiting_dev_positive`（从未 launch，未触碰 unseal token）。

## 5. 指标小结 vs 基线（inner-val，fold0 screening；**fold0 只作筛选，不得声称泛化**）

| arm | best rmse/zero（seed） | 三 seed 范围 | sign_median 范围 | 结论（record-only，未闸门） |
|---|---|---|---|---|
| F0 | 1.030 (s17) | 1.03-1.34 | 0.13-0.69 | 全部 >1：最小特征在 fold0 内层 val 不敌 zero-step |
| F1 | 1.181 (s17) | 1.18-1.29 | 0.24-0.63 | 同上 |
| F2 | 1.118 (s29) | 1.12-1.21 | 0.38-0.54 | 同上 |
| F3 | 1.075 (s29) | 1.07-1.17 | 0.38-0.71 | forward 臂中最稳，仍 >1 |
| **F2-bidir** | **0.884 (s17)** | **0.88-0.95** | **0.76-1.00** | **唯一低于 zero-step 的臂**；与 forward-F2 同曲线预算同更新数（30h §4.2 双向纪律满足） |
| F2-intent-off | 1.106 (s43) | 1.11-1.19 | 0.24-0.53 | ablation 不敌 bidir；intent-off 在 sign 上部分种子尚可 |

- guards 全部记录于各 metrics.json：forward 臂 `rmse_degradation_vs_zero_step_pct` 3-34%（>2% 记录性超标，dev 不闸门）；bidir 三臂为负退化（改进）；`deadzone_rise_vs_zero_step` 全为负（模型 dead-zone 远低于 zero-step 的 1.0）。
- 训练动力学：多数 run inner-val loss 在 250 upd 后单调上升（7 条内层曲线的早期过拟合）→ 选择器正确多在 250-1250 截止；train loss 终值 0.06-0.10。**不调参**（冻结超参）；此为筛选证据。
- classic CV/Jacobian 基线：sign_median 0.0-0.125、SRE 0.38-0.57（公式在 metrics.json `baselines.classic_cv_jacobian.formula`，G2 复核时可接受/替换）。

## 6. 流程事件（透明披露）

1. F0-s17 首次 launch 因驱动 `Path(None)` bug 在 attempt_launch 后崩溃 → run 留 running → 下次运行 `ledger.reconcile()`（dead-pid→interrupted）→ 重试为 attempt 2（计数；诚实入账）。修复后无再发。
2. 确定性如实降级：CPU torch 多线程导致同 seed 重跑 loss 差 ~1e-8，测试改为容差断言；**不宣称 bit-exact**。
3. 工作树 outputs/ 曾为 E0 期旧 plan 副本（d2a63599）——supervisor 已同步 a8972a19；本 lane 执行全程 cwd=主仓，未受影响；同步的 plan/protocol 工作树字节**未纳入本 lane 提交**（supervisor-owned）。

## 7. 下游依赖

- **G2-Move 评估**（supervisor/E7）：18 个 done run 的 metrics.json 为证据包；F2-bidir 的领先信号供 B 臂 winner 选择（B 臂需 `--level` 显式指定 winner 特征级；本 lane 未做选择——选择权在 G2 评估）。
- **L 臂（S2/S10/D2/D10）**：需 WP-5 `r3_gradient_probe` 校准 λ（driver 已拒无 λ 跑）；V/R 臂需 top-2/最佳 loss 选择 + R1/R2 ramp 规格（冻结配置未定义，driver 拒跑）。
- **confirm 臂**：等待 dev-positive 判定 + 用户 `confirm_authorized`；unseal token 未触碰。
- 账本余量：R3 19/66 attempts 已用；V5.2 0/33。
