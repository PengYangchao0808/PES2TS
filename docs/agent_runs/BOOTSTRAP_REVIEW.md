# PES2TS-30H BOOTSTRAP REVIEW — 启动前评审

> 状态：**Review（提交用户决策；未授权训练）** · 日期：2026-10-20 · 评审人：supervisor（orchestrator）
> 上游：[PARALLEL_EXECUTION_PLAN.md](PARALLEL_EXECUTION_PLAN.md) v0.2 · 依据：宪法 R1–R6、AGENT.md、30h 计划 v1.1、ADR-0028/0029、ReactionIntent Contract v1、Oracle ORA-1 预审
> 结论预览：**并行执行系统 dry-run 通过；正式启动被 4 项宪法阻塞决策（PD-1/2/4/6）+ WP-L2 硬化 + E0 冻结签认阻挡。**

---

## 1. 环境与平台核查（全过）

| 项 | 结果 |
|---|---|
| OpenCode / OMO-Slim | v2.0.29 / v3.0.4（preset `zai-mix`；orchestrator→`zai-coding-plan/glm-5.3`；explorer/fixer 走轻量链） |
| GLM-5.3 | `zai-coding-plan/glm-5.3`（active，low/high/max variants，订阅制 cost 0 → 无计量配额，仅并发窗） |
| 服务商并发 | **3 并发实测无 429/限速**（§3 dry-run 期间） |
| 后台子任务能力 | 已实证：5 个后台 session（2 explorer + 3 fixer）+ 1 oracle，task 追踪/完成通知/续话均正常 |
| GPU | RTX 4060 Ti 8GB；⚠ 2.6GB 已占用（归属未查，PD-8，禁盲杀）；CUDA 可用 |
| 训练解释器 | 正式训练=`/root/miniforge3/envs/pes2ts-gpu/bin/python`（torch 2.14.1+cu130，driver 走 `PES2TS_GPU_PYTHON`）；工具/测试=系统 `python3`；⚠ lane 间解释器发现不一致（wt-b 自建 /tmp/opencode/venv）→ E0 统一登记 |
| 依赖/测试/磁盘 | numpy2.2.6/pandas2.2.3/rdkit2026.03.6/h5py3.16/pyarrow18.1/pyyaml6.0.3；2649+70=2719 项可收集；/mnt/e 余 120G；仓库 tracked 23M（worktree 廉价） |
| Marketplace | live，无需外部包 |

## 2. 文档与代码侦察结论（EXP-1/EXP-2）

- EXP-1：V5-V7 v3.1 / V5.1-R3 v3.1 / ReactionIntent v1 摘要 + **7 项文档冲突**（已并入 PD 清单）；R3 待建脚本/模块/测试清单（`scripts/campaigns/r3_*.py`、`pesn/**/*_v3.py`、`tests/pesn/test_r3_*.py` 均不存在）。
- EXP-2：`pesn/` 代码地图 + **11 项 gaps**（要点：§7.1 六个冻结件全缺；F0–F3 superset/mask 无实现且 NODE20/EDGE29/TASK19 无 intent 槽；`intent_directed_irc_step`/involution 无实现；S10/D 档与 `scale_unreachable` 缺；V5.2 三头/三态 Stop、V5.3 rollout、d0=0.08 重标全待建；旧 d0=0.05）。可复用：损失族、first-contact 教师求解、RunRecord/checkpoint config-hash 守卫、崩溃安全队列、G5.x 门逻辑、split 工具。
- 关键勘误：真实 53-run 封存目录为 `outputs/pesn/v5_1/**`(21)+`v5_1_r1/**`(32)（初版状态机笔误 v5_1a 已修正）。

## 3. 并行 dry-run 结果（3 独立任务，全部 supervisor 复核）

| Lane | 交付 | commit | 测试 |
|---|---|---|---|
| A W-DATA | R3 六表 schema/FK/泄漏校验器（T0.1a/b/c、T0.4a/b）+ 4 组 fixture | `2eda543` | **18 passed**（复跑）；CLI 语义正确（clean=0/违规=1/缺目录=blocked） |
| B W-FEAT | 等变/Relabel/Reverse/合法输入/NaN-typed-invalid harness（T1.1–T1.4）+ 4 bug 模式 dummy | `121fbce` | **41 passed**（复跑）；4 bug 模式均恰好捕获；容差双预设显式警告（不静默默认） |
| C W-LOSS | run 账本（run_id 幂等/占位符拒绝/retry≤2+override/审计事件）+ GPU flock 互斥 + 陈旧锁检测 | `e2133cf`+`efe5bce` | **11 passed**（复跑）；supervisor 修正默认路径→`outputs/pesn/30h_campaign` |

**验证结论**：① 三 lane 并发无文件冲突（write_scopes 交集为空；合并零冲突，scope 预检三支全 OK）；② 3 并发 provider 无限速；③ 结构化 7 字段报告全部合规、无原始日志；④ supervisor 对每条 lane 独立复跑测试并审读代码后才采信。

## 4. Oracle ORA-1 合规预审（已处置）

| 发现 | 级别 | 处置 |
|---|---|---|
| `data/**`、`outputs/pesn/**` 读域过宽（含 ground_truth/sealed）；复用驱动默认根写 sealed | 红线级 | ✅ 已改 config（never_write/排除 ground_truth/driver_campaign_root_rule）；runner 断言=WP-L2 |
| sealed/Confirm/OOD/Fit150+ 与训练授权绑同一开关 | 红线级 | ✅ 已拆独立 token（恒 false） |
| gate `not_admissible` 纯声明；ledger 无 gate_id/admissibility | 高 | ⚠ WP-L2（代码级强制：取 GPU 锁前校验 gate JSON+protocol hash） |
| WP-4' λ 校准依赖错误（G1→应为 WP-5 F0+G2） | 高 | ✅ 已改 DAG |
| run_plan/run_records 未分离；run_id 键不含 hash；66/33 上限可被 retry 绕过 | 高 | ✅ 规则入 config；实现=WP-L2 |
| `remaining_wall_clock_h` 静态、无绝对 deadline | 中 | ✅ time_control 规则入状态机；E0 写死；实现=WP-L2 |
| WP-6 标签可基于未过 G0 数据生成；WP-10 可早发 | 中 | ✅ 已改（标签绑定 data_manifest_sha+gate_id；WP-10 需 WP-9 终态） |
| PD-1/2/4/6 为宪法阻塞；PD-7 需台账登记 | 决策 | ✅ 已标记；本文档即登记 |

## 5. 遗留阻塞（正式启动前必须清空）

1. **用户裁定 PD-1/2/4/6（宪法阻塞）** + PD-3/5/7/8/9；
2. **WP-L2 预启动硬化**（scope 见 task_state.prelaunch_hardening）+ Oracle 复审 pass；
3. **E0 冻结签认**：六表 schema、F0–F3 槽位、容差预设、per-file allowlist+冻结 hash、campaign 绝起止时间、`protocol_30h.json` 等 §7.1 七件；
4. GPU 2.6GB 占用查属清理（PD-8）；主树全量 pytest 绿（进行中，结果回填本节）；
5. 用户显式 `formal_training_authorized=true`（不连带任何解封）。

**主树全量 pytest**：2719 项（2649 旧 + 70 新）——结果：`见 git log 后续 commit / 状态机记录`。

## 6. 合规声明

本 bootstrap 未启动任何训练；未触碰 sealed 目录、`data/ground_truth`、旧 53-run 证据；未修改任何科研指标/阈值/损失/数据语义；所有新增均为编排/测试脚手架（`docs/agent_runs/**`、`scripts/agent_ops/**`、`tests/agent_ops/**`），主树用户未提交改动未被纳入我的 commit。
