# PES2TS-30H 24h 执行时间线（living document — supervisor 维护）

- 战役: PES2TS_30H_R3_V52_V53 · protocol v1.0.0（E0 条件签认）
- 授权窗口: 2026-10-20T13:45Z → 2026-10-21T13:45Z（24h，用户显式授权）
- 授权状态: formal_training_authorized=false（前置: WP-L3 + G0 + E2/G1 + 实值 SHA 绑定 + 训练前审计）；sealed/confirm/ood/fit150plus 恒 false
- E0 核验（supervisor 实测）: yaml=3081e05479007eaf ✓ run_plan=d2a63599727804e9 ✓ allowlist=9a3e27e12e15c50d ✓ ORA-3=LAUNCH_BLOCKERS_NONE ✓ → protocol 1.0.0-e0→1.0.0，commit 2630d6c

## 事件日志

| UTC | 事件 | 证据 |
|---|---|---|
| 13:44Z | E0 证据核验（4 SHA + ORA-3 + 两段式 PENDING 声明一致） | 本文件 §E0 |
| 13:45Z | E0 条件签认生效；campaign clock 启动；phase deadlines 写入 task_state | 2630d6c |
| 13:47Z | 协调裁定: E1 builder 落 scripts/agent_ops/data_gate/（单写者规则）；key_atom_dataset_v3.py 授予 W-FEAT；loss 函数(models/key_atom_losses_v3.py)=W-FEAT、训练循环(key_atom_bc_v3.py)=W-LOSS wave-2 | ad4b879 |
| 13:50Z | 三 lane 并行派发: E1-DATA(wt-a/ops/e1-data, ses_ea66b6551ffejO6LMIT7xKsffQ)、WP-L3(wt-c/ops/wp-l3, ses_ea66b654fffe2REQ8iiWIEqW4p)、E2-FEAT(wt-b/ops/e2-feat, ses_ea66b654fffdIshaQ5CgZsFuO4) | 本文件 |
| 15:11Z | E1 数据集落地: 六表+3 manifests, validator T0.1-T0.5 10/10 pass | data_gate_report_r3.json |
| 15:30Z | E1-DATA 完成 (d5f4cb6): 50 曲线/100 intents/12034 样本; 覆盖率 ok 9655/terminal 1123/conflict 68/masked 1188; 四 cohort 全处置; 确定性重建 ×3 | W-DATA_E1_sixtables.md |
| 15:35Z | **supervisor 验收通过并合并 (90f7844)**; 独立复跑: validator 10/10、pytest 14 passed、三 SHA 复算一致、run_plan 守卫 d2a63599 未动 | /tmp/opencode/g0_supervisor_rerun.json |
| 15:36Z | **G0-Data Gate = PASS**; protocol 绑定实值 SHA (data c21ff84b/teacher bc02a2bd/split 7cc1f98e; features 仍 PENDING_E2) | gates/G0-Data.json, 97dda4e |
| 15:55Z | E2-FEAT 完成 (b789fe7, rebase 于 97dda4e): v3 全栈 7 模块 + 136 数学测试; StarEGNNV3 实测 55,441 参数; λ=0 与 B0-F 逐比特一致 | W-FEAT_E2_v3_stack.md |
| 16:00Z | supervisor 验收: 136 battery + 1165 suite 复跑一致 (venv); scope 19 文件干净; 合并 badb141 | 本文件 |
| 16:05Z | **G1-Feature Gate = PASS（真实数据）**: E1 表抽样 7 样本(5 曲线/5 折/正反 intent), T1.1-T1.4c 6/6, yaml 绑定验证; feature_spec.json 绑定 sha=0e7de362458e2b0f | gates/G1-Feature.json, /tmp/opencode/g1_real_report.json, 5d92ebd |
| 16:15Z | WP-L3 完成 (0c2e806) 并验收合并 (45eb3b5): 计划 pin 每次 append 强制校验(Oracle 三篡改路径全拒)、metric_unseal 消费接线、monitor PlanNotFrozen 捕获、driver root 断言、stage_bind 助手; 405 agent_ops 复跑一致 | W-LOSS_WP-L3_ledger_hardening.md |
| 16:20Z | **stage-g0 两段式重绑**: run_plan 五实值 SHA (E0 版归档 freeze_archive/), 结构 diff=NONE(身份序列逐行一致), verify stage=g0 PASS 99 run_ids; 四方 pin 同步 (task_state/protocol/代码常量/测试 stage-aware 化); 篡改拒绝复验 PLAN_SHA_MISMATCH | run_plan_pinned_history |
| 16:55Z | **训练前审计通过 → formal_training_authorized = TRUE**: agent_ops 424 + pesn 819 全绿, GPU 5631MiB 空闲; sealed/confirm/ood/fit150+ 恒 false; 确认折实验自动 BLOCKED | task_state.formal_training_authorization_record, 6553d50 |
| 17:00Z | wave-2 双 lane 派发: E3-TRAINER(wt-c 续会话, 训练器+GPU 队列+F0), E4-V52HEADS(wt-b 新会话, V5.2 heads+标签, CPU) | 本文件 |
| 17:05Z | E4-V52HEADS 完成 (5da5686): heads 2178 参数(backbone 逐比特不变), q 标签 9655 有效/0 掩码, reverse q=-q 精确(6017 对), stop 每折正例 164-171, 类权重冻结; 发现 wt 内过期 E0 计划副本环境问题 → 三 worktree 战役工件全同步 + 通知训练 lane | e4_label_report.json |
| 17:35Z | E4 验收合并 (1351931): 61 新测试 + 全量 1304 全绿(主树); **G3a 证据完备, 正式绑定按 DAG 顺延至 G2 后** | gates/G3a-Label.json (evidence_complete) |
| ~18:30Z | **E3-TRAINER 完成 (ff3c773 合并)**: 18/18 dev-fold runs (19 attempts, R3 19/66), 12 confirm 臂 BLOCKED; 吞吐 43.25 upd/s → ~45-55s/run; key_atom_bc_v3 + 驱动全接 ledger/GPU-flock | W-LOSS_E3_trainer_and_runs.md |
| ~18:35Z | supervisor 从 metrics.json 独立复算确认: **F2-bidir 唯一优于 zero-step** (0.884/0.896/0.949, sign med 0.76-1.00); forward 臂全部 >1 (1.03-1.34, 早期过fit sel@250); 意图信息只在双向监督下有效 — 诚实筛选结论, 不调参 | r3_runs/*/metrics.json |
| ~18:45Z | 修正 model.eval()→train(False) (truth_guard 静态拦截误报, 遵仓库 v1 惯例, guard 未动); 全量 1312+14 全绿 | b4bffbc |
| ~18:50Z | wave-3 派发: E3-CONT(λ 校准探针 + B12/V12/L12 臂, folds3/4 指标封存, R6 臂=spec gap BLOCKED), E6-V53PREP(rollout harness + ACP 桥预检 + 曲线预注册, CPU) | 本文件 |
| ~19:55Z | **E6-V53PREP 完成 (dc30176 合并)**: rollout harness(classic/r3/v52 臂, TS_candidate_only), 12 曲线冻结预注册, CPU smoke(真实 F2-bidir ckpt, 有限/单调, sign_agree 0.75 诚实记录); **supervisor 修正: xtb 实际可用**(/opt/software/xtb/bin/xtb, lane 探测漏检) → 受约束松弛在 G4 阶段可执行(待许可) | relax_preflight.json 修正, 2cb4696 |
| ~20:10Z | E7-STATSPREP 派发 (W-DATA, CPU): cluster-级 bootstrap CI/错误簇卡片/成本账本工具 + F 阶段首份统计报告 | 本文件 |
| ~20:40Z | **E7 工具完成 (77f5418 合并 bc8758a)**: 12 测试过, 双跑字节一致; 报告显示 **B/V/L 臂已全部落地** (54 metrics = F18+B12+V12+L12, 6 个 B f3/4 封存正确排除); 成本 ~0.79 GPU-h | stats/r3_stats_report_v1.md |
| ~20:45Z | 关键跨臂结果 (dev folds 0-2): **F2-bidir 0.910 (6/6 簇优于零步)** + λ 校准辅助臂 S2/S10/D2/D10 0.908-0.920 (6/6); **B(=F2-bidir 跨折 f1-2) 0.9735 [CI 0.949-0.998], 10/12 簇优于零步, 2/2 折 ≥2/3 种子 <1**; V-top2(F3) 1.039 平价偏弱 → Top-1 选择获跨折支持; forward-only 全部 >1 | stats 报告 |
| ~21:25Z | **E3-CONT 完成 (7315e00 合并)**: λ 探针 (S2=0.019/S10=0.095/D2=2.1e-5/D10=1.05e-4, 全 calibrated, λ=0 逐比特) + 36/36 B/V/L runs (61/66 attempts); **folds3/4 封存端到端执行** (metric_unseal 一次性消费, 6 收据, 值不入任何聚合); R1/R2 spec-gap BLOCKED; G2 证据包 | W-LOSS_E3_BVL_arms.md |
| ~21:30Z | **G2-Move = PASS + G3a-Label = PASS** (3341452): G2 以 dev 折证据绑定, 诚实附注 (泛化衰减 0.91→0.974、早期过拟合 43/48 选择≤250upd、fold2 s17 单点 1.019、confirm 封存→本战役 R3 终局=dev-positive-confirm-blocked, scale_controlled 不可主张); 探针参照裁定=F2-bidir (辅助臂承袭获胜者配置) | gates/G2-Move.json, G3a-Label.json |
| ~21:35Z | **E5-V52 派发** (W-LOSS wave 4): Q9/T9/QT9 训练 (backbone 初始化自 R3 winner ckpt, freeze 400+joint 1500), QT-confirm 6 保持 blocked, Move 非退化护栏, G3 证据包 | 本文件 |
| ~21:00Z | **E5-V52 完成 (3cc44c2 合并)**: V5.2 全栈 + 9-run 预注册 salvage (λq=0.421/λm=0.0132 探针校准); **两次事故全披露**: λ 校验 bug 烧 15 attempts(修复+回归测试), v1 split 确认折泄漏 8 runs 自判无效(结果观察前声明, SPLIT_SEAL 修复) | W-LOSS_E5_v52.md, invalid_runs.json |
| ~21:15Z | **G3-V52Supervision = FAIL (E-NO-GAIN 诚实结论, 2029264)**: q 头不可学习 (MAE 0.507>常数基线 0.476, corr≈0), stop AUC≈0.5 (recall@0.5=0), **Move 护栏 4/6 QT 破防 (中位 +5.5%)**; 裁定: 不扩容 (32/33 关闭), λ/schedule 修订越权→明日决策备忘; **H20 激活: V5.3 仅上限控制器** | gates/G3-V52Supervision.json |
| ~21:20Z | **E6-V53EXEC 派发** (W-FEAT wave 4, GPU 空闲转交): 12 冻结曲线 × {5,10,20} 步 × {classic, r3 上限, v52-diagnostic} + xtb 受约束松弛 (同预算同约束, G4 阶段放行) | 本文件 |

## Phase 状态

| Phase | 目标截止 | 状态 |
|---|---|---|
| E1+WP-L3+E2 并行 | 17:45Z | ✅ 全部完成 |
| G0 Data Gate | 18:30Z | ✅ **PASS** (15:36Z) |
| G1 Feature Gate | 19:15Z | ✅ **PASS** (16:05Z) |
| E3 R3 训练 (≤66 runs) + G2 | 00:30Z | ✅ **61/66, G2 PASS** (21:30Z) |
| E4 labels + G3a | 02:00Z | ✅ **G3a PASS** (21:30Z) |
| E5 V5.2 训练 (≤33 runs) + G3 | 07:30Z | ✅ **32/33 关闭, G3 FAIL (E-NO-GAIN 诚实结论)** (21:15Z) |
| E6 V5.3 rollout + G4 | 11:00Z | 🔄 执行中 (harness+xtb 就绪, H20 约束) |
| E7 统计复核 | 12:30Z | 🔄 工具 ✅; 报告待 V5.3 后终版 |
| E8 冻结/归档 | 13:45Z 硬停 | ⏸ |

## 训练 run 计数（所有 launch attempt 计数）

- R3: 61/66 关闭 (54 done; C6+R6 BLOCKED) · V5.2: 32/33 关闭 (9 valid + 7 invalid done, 1 interrupted, 6 confirm blocked, 1 held) · V5.3: rollout/relax 执行中 (非 run_plan 计数体系)
| G1 Feature Gate | 19:15Z | ⏸ 等 E2+G0 |
| E3 R3 训练 (≤66 runs) + G2 | 00:30Z | ⏸ |
| E4 labels + G3a | 02:00Z | ⏸ |
| E5 V5.2 训练 (≤33 runs) + G3 | 07:30Z | ⏸ |
| E6 V5.3 rollout + G4 | 11:00Z | ⏸ |
| E7 统计复核 | 12:30Z | ⏸ |
| E8 冻结/归档 | 13:45Z 硬停 | ⏸ |


| ~21:15Z | **G3-V52Supervision = FAIL (E-NO-GAIN)**: q 不可学习/stop 随机/Move +5.5% 破防; 裁定不扩容 32/33 关闭; H20 激活 | gates/G3-V52Supervision.json |
| ~21:20Z | **E6-V53EXEC 完成 (e0db8f7 合并) + G4-Rollout = PASS (工程闭环)**: 216 episodes + 36 真实 xtb (5 收敛/15 未收敛/16 物理失败诚实归因); TS_candidate_only 零违规 | g4_evidence_pack.json |
| ~21:25Z | **六门全部终态**: G0✅ G1✅ G2✅ G3a✅ G3❌(E-NO-GAIN) G4✅(工程); E7 终版统计; 授权撤销; 七份交付报告产出 | 24H_FINAL_REPORT.md |

## 战役终态（21:30Z，开始后 7.75h）

- 科学执行全部完成（低于 24h 预算的 1/3）；GPU 训练净耗 0.93 GPU-h
- R3: 61/66 attempts 关闭 · V5.2: 32/33 关闭 · V5.3: 216+36 episodes 工程闭环
- BLOCKED (授权/spec): C6 + QTc6 + R6 + V5.2 余量 14+1
- 全量测试终态: 1369 passed / 0 failed
