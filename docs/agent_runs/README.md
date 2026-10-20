# docs/agent_runs/ — PES2TS-30H 执行套件（agent 运行区）

> 状态：**Active（并行 dry-run 已过；正式启动待用户授权）** · 层级：执行编排 · 版本 v0.2 @ 2026-10-20
> 规范根：宪法（`docs/PES2TS_开发宪法_v1.md`）；本目录不新增规范，仅承载编排/状态/交接。

> 用途：30h 战役（PES2TS_30H_R3_V52_V53）的编排/状态/交接层。**本目录不承载科研规范**；规范见宪法与 docs/contracts、docs/plans。

## 目录

| 路径 | 职责 | 写入负责人 |
|---|---|---|
| `PARALLEL_EXECUTION_PLAN.md` | 并行执行模式方案（并发模型、DAG、WP、gate 串行规则、合规映射、待决策清单） | supervisor（orchestrator） |
| `BOOTSTRAP_REVIEW.md` | 启动前评审：环境核查、dry-run 结果、启动检查单 | supervisor |
| `state/parallel_config.json` | 并发配置（lane/worktree/write_scopes/修复限额/启动闸） | supervisor |
| `state/task_state.json` | 持久化任务状态机（阶段/gate/task/预算/封存边界） | supervisor |
| `state/runs.jsonl` | 运行账本（run_id 幂等，append-only） | supervisor（经 run_ledger） |
| `state/heartbeat.jsonl` | lane 心跳快照 | supervisor |
| `state/gates/` | G0–G4 gate 评审 JSON（串行验收记录） | supervisor |
| `state/lane_reports/` | lane 结构化报告（7 字段，禁原始日志） | supervisor 收存 |
| `handoffs/` | 阶段交接文件（低上下文机制：每阶段一份） | 完成该阶段的 lane + supervisor 定稿 |

## 阶段交接文件模板（handoffs/PHASE_<E#>_<名称>.md）

```markdown
# PHASE_<E#> 交接
> 完成时间 / lane / commit
1. 目标与结果（≤10 行）
2. 产物与证据路径（表：文件 → 用途 → 测试状态）
3. 与计划的偏差（无则写"无"；偏差不回改计划，只登记）
4. 未决问题（→ task_state.pending_user_decisions 编号）
5. 下游依赖（哪个 WP/门 等待本阶段哪些产物）
```

## 硬规则

1. 正式训练与解封是**两个独立 token**：`formal_training_authorized==true` 仅授权 30h 正式训练；sealed/Confirm/OOD 解封与 Fit150+ 属独立审批，**本战役恒为 false**（ADR-0029 §2.4、30h §7.3）。
2. 一切 CUDA 进程（含 smoke/CUDA 测试）持 `/tmp/opencode/pes2ts_gpu.lock`；正式 run 须先过 gate 强制校验（WP-L2）。
3. 科研不可变项（指标/阈值/损失/数据身份/标签语义）任何 agent 不得修改；`data/ground_truth/**` 仅经 truth_reader（R1）。
4. 每文件唯一写入负责人；冲突以 `parallel_config.write_scopes` 为准；合并前 supervisor 校验 changed_files ⊆ write_scopes。
5. 中断恢复：先查进程与账本，run_id 幂等，禁止重复提交；66/33 上限计所有 launch attempt。
6. 旧证据目录（`state.sealed_readonly`：`outputs/pesn/v5_1/**`、`v5_1_r1/**`）只读。
