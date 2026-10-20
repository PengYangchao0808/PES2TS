# E0 冻结签认报告 — PES2TS-30H

> 状态：**ASSEMBLED — 待人工签认** · 日期：2026-10-20 · 编制：supervisor（orchestrator）
> 签认对象：本报告 + `outputs/pesn/30h_campaign/protocol_30h.json`（v1.0.0-e0）
> **授权状态（签认前后均不变，直至用户另行显式授权）**：
> `formal_training_authorized = false` · `sealed_authorized = false` · `confirm_authorized = false` · `ood_authorized = false` · `fit150plus_authorized = false`

---

## 1. 签认范围

| 冻结件 | 路径 | SHA-256（前 16） |
|---|---|---|
| 战役协议 YAML | `config/campaigns/pes2ts_30h_20261020.yaml` | `3081e05479007eaf` |
| 运行计划（99 唯一 run_id） | `outputs/pesn/30h_campaign/run_plan.jsonl` | `d2a63599727804e9` |
| 写入 allowlist（23 patterns，lane 分域） | `docs/agent_runs/state/e0_write_allowlist.json` | `9a3e27e12e15c50d` |
| 冻结协议包 | `outputs/pesn/30h_campaign/protocol_30h.json` | 见提交后 task_state |
| Gate 初始记录 | `docs/agent_runs/state/gates/G0..G4.json` | 全部 `not_run` |

**两段式数据绑定（诚实声明）**：`data/teacher/split` SHA = `PENDING_E1`、`features` = `PENDING_E2`——E1/E2 产物尚不存在（本阶段未获授权也未执行数据构建）。G0 前以实值写入并以 `--stage g0` 重验；**在此之前任何 run 无法 launch**（ledger 实测 `PLAN_NOT_FROZEN` 拒绝）。此非缺陷，是 fail-closed 设计。

## 2. PD-1..9 落实状态（全部闭环）

| PD | 落实 | 证据 |
|---|---|---|
| 1 六表 | R3 v3.2 修订 + schema `r3_6table_v2` + 10 fixture 校验 | `29f0e25`/`279967b` |
| 2 分层容差 | 规范预设绑定 yaml；1e-4 → record-only 不可作通过门；无预设=拒绝 | WP-F2 `7fa6aba`（Oracle 实测 tamper→`PRESET_YAML_MISMATCH`） |
| 3 66-run | 66+33=99 唯一 run_id；PD-3 note 内嵌；额外 3-run=F-bidir（F2-forward fold0×3 在 F-main 内作配对对照）；预算计所有 attempt | run_plan verify PASS（stage=e0） |
| 4 21-run 降级 | `--mode partial` 仅 21 run 探索集，ledger 强制 `not_admissible_exploratory`，拒 scale_controlled/V5.1b | WP-L2 r2 `b1907c6` |
| 5 campaign_id | 全链固定 `PES2TS_30H_R3_V52_V53`；stage/arm/run_id 唯一 | ledger/run_plan |
| 6 intent-off 全 mask | 五类全 mask + 无向上下文保留 = T1.4c 机器检查 | WP-F2（实测 partial mask→fail 并列泄漏类） |
| 7 agent_runs | 目录 + 台账 + docs/README 索引 | `a5fde4b`/台账 §24 |
| 8 GPU 诊断 | 无 CUDA 进程；2652MiB=Windows 显示栈；不清理；~5.5GB 可用 | §4 |
| 9 delta_xyz flat | flat+shape/offset+映射/单位/SHA（float64-LE）；eps=1e-6；ts_index 权威 | schema v2 + T0.1c/T0.2b |

## 3. 审查链（三轮 Oracle）

- **ORA-1**（bootstrap）：3 阻塞 → 编排层修复 + WP-L2。
- **ORA-2**（WP-L2 r1 后）：5 组阻塞 → WP-D2（数据门 v2）/WP-F2（特征门 v2）/WP-L2 r2（账本/计划/守卫 v2）全部关闭。
- **ORA-3**（终审，@b1907c6）：**LAUNCH BLOCKERS: NONE**。五组逐项 CLOSED，含 live 实测（fail-closed 探针 11 项全拒绝路径生效）。

## 4. 测试与运行记录

- 主树全量 pytest：**2997 passed / 1 skipped / exit=0**（186s，@b1907c6；原 2649 + agent_ops 349）。
- fail-closed 探针记录（protocol `test_record`）：launch-stage 拒 PENDING（396 problems）、`PLAN_NOT_FROZEN`、`RUN_ID_MISMATCH`、`UNSEAL_OUTSIDE_BATCH/BATCH_MISMATCH`、`AUTHORIZATION_REQUIRED`（授权=false 时 formal 拒绝——**当前即为该状态，正确**）、`GPU_PRECHECK_REFUSED`、`SEALED_ROOT_REFUSED`。
- GPU：RTX 4060 Ti 8GB；CUDA 可用；占用属 Windows 显示栈（PD-8）；可用 ~5.5GB（StarEGNN 59k 参数 batch32 充足，E0 吞吐 smoke 未运行——属战役 H00–H02 内容，随正式授权开启）。

## 5. 已知残余（不阻塞 E0 签认，均有归属与期限）

| 项 | 归属 | 期限 |
|---|---|---|
| ledger.append 校验钉定 run_plan sha+mode（Oracle 探明的残余路径：改 frozen plan 的 confirm flag / partial 改 full 头仍可 launch——verify 能抓但 append 未调用） | **WP-L3**（W-LOSS 新任务） | **E3 首个训练 run 前**（E1/E2 期间完成即可，正式训练授权前置条件之一） |
| `metric_unseal` token 消费接线（B folds3/4 指标揭示） | WP-L3 | 同上 |
| `assert_campaign_root` 集成进复用 driver | E3 lane 启动时 | E3 |
| G1 报告消费用 `state/passed` 而非 `gate_admissible`（yaml 不匹配时后者不复位） | WP-2 接口注意项 | E2 |
| 测试解释器：agent_ops 需 torch 环境（venv 或 pes2ts-gpu）；系统 python3 下 feature 模块 skip | CI/运行手册 | E1 起执行 |
| 两 lane 曾报告的 master 侧问题（config 尾逗号、yaml L96） | 已修复（`1c56fc6`/`710a3f5`） | ✅ |

## 6. 流程偏差记录（透明披露）

1. WP-D2 合并先于 check-merge（master 缺 allowlist + diff 基准误用）→ 事后按 lane 自有提交复核 100% scope 内；规程已修正（lane-authored commits 为基准；allowlist 随 E0 入库——本次已入库）。
2. 一次配置编辑引入 JSON 尾逗号（W-D2 lane 发现）→ 已修；教训：状态/配置文件变更后即跑 JSON 校验（现已入流程）。
3. W-LOSS r2 中断一次（provider 错误）→ 按规程复活原会话续作，无重复派发。

## 7. 签认后解锁序列（均需你逐一显式批准）

1. **E0 签认**（本报告）→ 冻结生效（protocol 升 1.0.0）。
2. WP-L3 + E1 数据构建授权 → 六表+教师+split 实值 SHA → G0（Data Gate 串行验收）。
3. E2 → G1 → E3（R3 66-run，GPU 队列）→ … 按 DAG。
4. 正式训练还需：`formal_training_authorized=true`（独立显式授权；不解封任何 sealed/Confirm/OOD/Fit150+，恒 false）。

**签字栏**：□ 批准 E0 冻结　□ 需修改（请列项）　□ 拒绝
