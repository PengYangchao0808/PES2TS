# PES2TS 30h 并行研究执行模式方案（PARALLEL_EXECUTION_PLAN）

> 状态：**Draft（并行 dry-run 进行中；未授权启动正式训练）**
> 层级：执行编排（ops）；非规范，规范见[开发宪法](../PES2TS_开发宪法_v1.md)与[AGENT.md](../../AGENT.md)
> 取代关系：取代 bootstrap 单 lane 模式（无显式并发策略）；不改写任何科研计划/合同
> 上游依赖：[30h R3→V5.2→V5.3 计划 v1.1](../plans/PES2TS_30h_R3_V52_V53_分层训练与验收计划_v1.1_20261020.md)、[ReactionIntent Contract v1](../contracts/PES2TS_ReactionIntent_Contract_v1.md)、[ADR-0028](../design/decisions/ADR-0028-关键原子几何策略合同与V5-V7任务拆分.md)、[ADR-0029](../design/decisions/ADR-0029-V5.2+双门评估与封存边界.md)、[V5.1-R3 v3](../plans/PES2TS_V5.1-R3_反应意图条件化与端点标签不变重训方案_v3_20261020.md)
> 下游消费者：PES2TS-30H 战役全部执行 lane、supervisor（orchestrator）、BOOTSTRAP_REVIEW.md
> 当前版本：v0.1 @ 2026-10-20 · 并发配置：[`state/parallel_config.json`](state/parallel_config.json)

---

## 1. 目的与红线

用模块级并行缩短 R3→V5.2→V5.3 墙钟时间，**同时**保持：化学合同（GeometryPolicyContract / ReactionIntent v1）、数据隔离（R1 真值隔离、R6 评测完整性）、科研验收规则（ADR-0029 双门、封存边界）全部不变。

**本模式不改变任何科学定义。** 指标、阈值、损失、数据身份、split 语义冻结于 E0 协议；Agent 无权修改（§9）。冲突科学定义进入待决策清单（§13），不猜测。

启动前置：并行 dry-run（§14）通过 + 用户正式确认。**在此之前不启动任何正式 GPU 训练。**

## 2. 环境核查结论（2026-10-20）

| 项 | 结果 | 备注 |
|---|---|---|
| OpenCode | v2.0.29 | 后台 subagent、worktree、task 追踪可用 |
| OMO-Slim | v3.0.4（preset `zai-mix`） | orchestrator 含 `zai-coding-plan/glm-5.3`；explorer/fixer 走轻量模型链 |
| GLM-5.3 模型 ID | `zai-coding-plan/glm-5.3`（active，variants low/high/max） | 订阅制（cost 0）；无计量配额，仅并发/速率限制 |
| 服务商并发限制 | 无公开文档；**以 3-并行 dry-run 实测为准** | 结果记入 BOOTSTRAP_REVIEW.md |
| GPU | RTX 4060 Ti 8GB，CUDA 可用 | ⚠ 当前已有 2.6GB 占用——正式训练前须清空并核实 |
| 训练环境 | **正式训练 python：`/root/miniforge3/envs/pes2ts-gpu/bin/python`（torch 2.14.1+cu130）**；driver 经 `PES2TS_GPU_PYTHON` 覆盖（`scripts/campaigns/run_v5_sprint.py:76`）；系统 `python3` 另有 torch 2.10.0+cu128（仅测试/工具用） | E0 测速与正式 run 必须用 pes2ts-gpu env |
| 环境探针 | `scripts/local/probe_wsl_gpu.py`、`probe_direct_training_env.py` | E0 测速起点 |
| 依赖 | numpy 2.2.6 / pandas 2.2.3 / rdkit 2026.03.6 / h5py 3.16 / pyarrow 18.1 / pyyaml 6.0.3 | 与 requirements 一致 |
| 测试 | 2649 项可收集（12.5s），10 项默认排除（realdata/xtb/acp/orca） | pytest.ini 已隔离重型标记 |
| 磁盘 | /mnt/e 余 120G；仓库 tracked 23M | worktree 开销极低 |
| Marketplace | live，无需安装包 | — |

## 3. 并发模型与配置差异

- **并发上限 3 lane**；三工作流固定：**W-DATA**（Data/Teacher）、**W-FEAT**（Feature/Model）、**W-LOSS**（Loss/Training）。Oracle 只在有空闲槽时进入（只读复核，永不第 4 并发）。
- 速率退避：provider 429/5xx → 暂停新派发 120s，重试一次，仍失败则该 lane 降级串行。
- **配置差异**（vs bootstrap 单 lane）：新增 [`state/parallel_config.json`](state/parallel_config.json)——含 `max_parallel_lanes=3`、worktree/branch 映射、每 lane `write_scopes`/`forbidden`、串行 gate 清单、GPU 互斥、修复限额、报告 schema、恢复策略、启动闸（`formal_training_authorized=false`）。Bootstrap 模式无此文件、隐式单 lane、无 GPU 互斥与 run 账本。
- 心跳与状态：supervisor 周期快照 `task_status` 写入 `state/heartbeat.jsonl`；lane 报告落入 `state/lane_reports/`。

## 4. 任务依赖 DAG（并行标注）

科学执行严格串行于 gate；**代码脚手架在接口冻结后可跨 gate 并行**（其结果标记 `not_admissible_until_gate_pass`）。

```text
E0 协议冻结(supervisor, 人工签认)
  ├─→ [W-DATA]  WP-1 E1 数据/教师构建 ────────────→ G0 Data Gate（串行验收）
  ├─→ [W-FEAT]  WP-2 特征规格+T1测试脚手架(合成fixture) ⫽ 与WP-1并行
  ├─→ [W-LOSS]  WP-4 训练栈构建(runner/ledger/gpu锁) ⫽ 与WP-1并行
  │        └ WP-2' 真实特征构建(需G0数据) ──→ G1 Feature Gate（串行验收）
  │        └ WP-4' λ校准探针(需 WP-5 F0 完成且 G2 pass；30h §4.1 要求新 B0-F ckpt+20真实inner-train batch；此前仅 synthetic fixture)
  ├─→ [GPU队列] WP-5 E3 R3 66-run 矩阵(需G1) ─────→ G2 Move Gate（串行验收）
  │        ⫽ 训练期间并行：[W-FEAT] WP-6 V5.2头/标签开发(冻结接口后)
  │                    [W-LOSS] WP-8 V5.3 rollout执行器开发
  ├─→ WP-7 E5 V5.2 33-run 矩阵(需G2+G3a) ─────────→ G3 Supervision Gate（串行验收）
  ├─→ WP-9 E6 V5.3 rollout(需G3) ────────────────→ G4 Rollout Gate（串行验收）
  ├─→ WP-10 E7 统计复核(可与E6尾部并行, 只读)
  └─→ WP-11 E8 缓冲/归档/decision memo ──────────→ H30 硬停
```

（⫽ = 并行；→ 带依赖。G0 失败阻断 E2/E3；G1 失败阻断 R3；G2 失败则 V5.2 仅合约测试；G3a 正例不足 Stop 降级探索性；G3 失败不解锁 V5.3 候选；G4 仅工程门。）

## 5. Work Packages

| WP | 阶段 | Lane | 交付物（关键文件） | 写入归属（唯一） | 验收测试 | 时间预算 | 依赖 |
|---|---|---|---|---|---|---|---|
| WP-0 | E0 | supervisor | `config/campaigns/pes2ts_30h_20261020.yaml`、`outputs/pesn/30h_campaign/protocol_30h.json` 等 §7.1 冻结件、吞吐 smoke（旧环境 100–300 updates，**需用户授权后**） | supervisor + 人工签认 | 协议 hash 齐全、合同签认 | H00–H02 | — |
| WP-1 | E1 | W-DATA | `scripts/campaigns/r3_build_data.py`、六表 CSV+分片、`teacher_manifest.json`、`split_manifest_r3.json`、`data_manifest.json` | worktree-a → 门审后并入主树 | T0.1–T0.4 全 pass（用 WP-D1 校验器） | H02–H05 | WP-0 |
| WP-D1 | 脚手架 | W-DATA | `scripts/agent_ops/data_gate/`（r3_table_schema + validate CLI）+ 合成 fixture 测试 | worktree-a（dry-run 已建） | 30h §3.2 五项全 pass | 0.5h | WP-0 接口冻结 |
| WP-2 | E2前 | W-FEAT | `pes2ts_core/pesn/features/key_atom_spec_v3.py` + `key_atom_graph_v3.py`（F0–F3 superset+mask）、T1.1–T1.6 测试脚手架（`tests/pesn/test_r3_*.py`） | worktree-b | 等变性 harness（WP-F1）+ 合成 fixture | H02–H07（与WP-1并行） | WP-0 接口冻结 |
| WP-F1 | 脚手架 | W-FEAT | `scripts/agent_ops/feature_gate/equivariance_harness.py`（relabel/reverse/等变/合法输入/typed-invalid，双容差预设） | worktree-b（dry-run 已建） | harness 自测 + bug 模式可捕获 | 0.5h | WP-0 |
| WP-4 | E3前 | W-LOSS | `pes2ts_core/pesn/models/key_atom_egnn_v3.py`、`offline/training/key_atom_bc_v3.py`（含 S2/S10/D2/D10、λ=0 回归）、`scripts/campaigns/run_v5_1_r3.py`（阶段驱动+账本+GPU锁）、`contracts/key_atom_v3.py` | worktree-c | T1.5 λ=0 逐比特一致；G5.1 fixture；账本/锁测试 | H02–H07（并行） | WP-0 接口冻结 |
| WP-L1 | 脚手架 | W-LOSS | `scripts/agent_ops/ledger/run_ledger.py` + `gpu_queue/gpu_lock.py`（run_id 幂等、retry≤2、flock 互斥、陈旧锁检测） | worktree-c（dry-run 已建） | 账本/锁 pytest 全 pass | 0.5h | WP-0 |
| WP-5 | E3 | GPU队列 | R3 66-run 矩阵（F12/F-bidir3/F-ablate3/B12/L12/V12/R6/C6）、B0-F 校准报告、`run_record.json`×N | 主树（仅 supervisor 执行） | T2.1–T2.3 + G2 门 | H07–H12 | G1 |
| WP-6 | E4/E5前 | W-FEAT | q 弧长标签、Stop 三态标签、`MoveHead/Progress/TSStop` 头、V5.2 loss、T3 测试（冻结接口后可与 E3 并行开发；**q/Stop 标签产物绑定 data_manifest_sha+gate_id，G0 pass 前只产 fixture**；30h §2 硬停规则：G2 未过仅"标签/头合约测试"） | worktree-b | T3.1–T3.4 脚手架 + head fixture | H12–H15（开发可提前） | 接口冻结；标签生成需 G0 |
| WP-L2 | 预启动硬化 | W-LOSS | gate 强制化（gpu_lock 取锁前校验 gates/G#.json==pass + protocol hash）、ledger 加 gate_id/admissibility/evidence_sha、run_plan/run_records 分离、run_id 幂等键扩展、E0 per-file allowlist 生成器、phase deadline 强制、fold3/4 解封计数器 | worktree-c | Oracle 复审 pass | 启动前必须 | Oracle ORA-1 |
| WP-7 | E5 | GPU队列 | V5.2 33-run（Q9/T9/QT9+confirm6）、`heads_only.pt`/`joint_best.pt` | 主树（supervisor） | G3 + T3.x 全量 | H15–H20 | G2+G3a |
| WP-8 | E6前 | W-LOSS | 无物理 rollout 执行器（trust_region_clip/几何合法性/防环）、受约束 RELAX 集成、Classic 对照卡 | worktree-c | mock 集成测试（物理缺则 `physics_not_run`） | 与E3/E5并行 | WP-0 |
| WP-9 | E6 | GPU队列+CPU | 5/10/20 步 rollout、8–12 曲线×3 臂对照、成本 ledger | 主树（supervisor） | T4.1–T4.5、G4 | H20–H26 | G3 |
| WP-10 | E7 | CPU 只读 | cluster bootstrap CI、错误簇卡片、门报告 | 主树 | 报告完整性；**发布条件：WP-9 全部 run_record 终态（否则标 partial 且禁用于 gate/decision memo）** | H26–H28 | WP-5/7/9 |
| WP-11 | E8 | supervisor | 失败补测、哈希验收、`final decision memo` | 主树 | §0 最低交付清单 | H28–H30 | 全部 |

**接口冻结顺序**：WP-0 冻结（a）六表 schema（以 30h §3.1 为准，待决策#1 确认）、（b）特征字典 F0–F3 槽位、（c）`run_record.json`+账本 API、（d）**per-file 写入 allowlist**（仅 `pes2ts_core/pesn/**/*_v3.py`、`tests/pesn/test_r3_*.py`、`scripts/campaigns/{r3_*,run_v5_1_r3}.py`）+ 对冻结文件（`key_atom_protocols.py` 阈值、`key_atom_bc.py` 损失、`key_atom_teacher.py` 标签）记录 sha256 denylist——翻转 `pes2ts_core` 禁令前 allowlist+hash 先行（Oracle ORA-1a）。冻结前 W-FEAT/W-LOSS 只写 `scripts/agent_ops/**` 与 `tests/agent_ops/**`，不写 `pes2ts_core/`。

### 5.1 现有代码复用 vs 待建（EXP-2 侦察结论，2026-10-20）

**可复用（旧车道，改造点明确）**：
- 损失族：`pes2ts_core/pesn/offline/training/key_atom_bc.py`（radial_huber δ=0.25、pair legacy_j、L_cos ramp、L_strong、L_dead2 全在）
- 教师求解：`offline/data/key_atom_teacher.py` `solve_teacher`（first-contact 解析求根、shard 写出、conflict_audit、gauge 分层）——**需加 intent 维度与 reverse involution**
- run 记账：`offline/experiments/runs.py` RunRecord + `checkpoints.py`（config-hash 拒改续跑）+ `run_v5_1_r1.py` 崩溃安全顺序队列模式
- 门逻辑：`offline/evaluation/key_atom_protocols.py`（G5.2/5.3/5.4、cluster bootstrap）；split 工具 `offline/splits/*`
- 现有测试锚点：等变（test_key_atom_model.py:203+）、λ=0 逐位（test_key_atom_trainer_r1.py:507）、双建同 hash（test_dual_ts_dev_split.py:112）、split 防火墙（tests/test_split_*.py）

**待建（EXP-2 gaps 摘要，全量 11 项见 BOOTSTRAP_REVIEW）**：E0 协议冻结件与写入器（§7.1 六文件全缺）；E1 六表构建器与 `r3_build_data.py`；F0–F3 superset/mask 与 `key_atom_egnn_v2` 输入维（现 NODE20/EDGE29/TASK19 无 intent 槽）；`intent_directed_irc_step` 教师；T1.1/T1.2/T1.6 测试；S10/D 档与 `scale_unreachable` 状态机；V5.2 三头与三态 Stop 标签；V5.3 rollout（StarEGNN 侧为 0）；d0=0.08 全量重标（现役 0.05）；`runs.jsonl`+`scripts/agent_ops/*`（dry-run 正在补）。

## 6. 文件写入所有权与 worktree 策略

- 每文件唯一写入负责人：lane 内文件集在 `parallel_config.json` `write_scopes` 声明；supervisor 派发前做冲突检查（两 lane scope 交集必须为空）。
- lane 开发在独立 worktree（`/tmp/opencode/pes2ts-wt-{a,b,c}`，Linux FS，pytest 快）；**gate 审核通过后由 supervisor 唯一执行合并到主树**，合并前逐 lane 校验 `git diff --name-only ⊆ write_scopes`（`scripts/agent_ops/__init__.py`、`tests/agent_ops/__init__.py` 两个 supervisor 属共享占位除外），merge commit sha 写入对应 gate/phase 记录；主树 = 集成与正式运行区，训练期间对已冻结版本只读。
- 共享产物（`outputs/pesn/30h_campaign/**`）只有 supervisor/GPU 队列可写；lane 报告只引用路径。

## 7. Gate 串行验收（不可越级）

G0→G1→G2(+G3a)→G3→G4 严格按 DAG 顺序由 supervisor 串行评估；每个 gate 产出 `state/gates/G#.json`（pass|fail|blocked|partial|not_run + 每测试证据路径）。**下游代码完成不构成越级启动正式训练的理由**；脚手架/fixture 级结果一律标 `not_admissible_until_gate_pass`。

## 8. GPU 队列与训练期间并行规则

- 单 GPU 训练队列：正式 run 持 `/tmp/opencode/pes2ts_gpu.lock`（flock + holder 元数据）；同刻仅 1 个正式 run。
- 训练期间允许并行 CPU 工作：数据分析、测试设计、V5.2 head 开发（冻结接口后）、V5.3 执行器开发；**禁止**修改该 run `config_sha256` 集 hash 锁定的任何文件（账本 pre-write 检查强制）。

## 9. 失败分类与修复限额

| 类 | 例 | 处置 |
|---|---|---|
| E-INFRA | 环境/IO/锁/网络 | lane 自愈或 supervisor 重派 |
| E-CODE | 实现缺陷 | **唯一负责人，最多 2 轮修复**（账本 retry_count 强制）；每轮必须带回归测试 |
| E-DATA-SCI | 数据质量/覆盖不足 | 不自动修；诊断报告 → 人工 |
| E-SCI-DEF | 科学定义歧义（§13） | 停止相关任务 → 待决策清单 |
| E-GATE-FAIL / E-NO-GAIN | 门不过/无增益 | 按止损表降级交付；**禁止调阈值或重定义任务** |
| E-LEAK-SUSPECT | 疑似真值泄漏 | 立即停 lane；Oracle 只读复核 → 用户 |

跨模块问题：Oracle 先只读复核定位归属，再派唯一 owner lane。科研不可变项：指标定义、门阈值、损失公式、数据身份/split 语义、标签语义。

## 10. Agent 结构化报告协议

每 lane 完成仅返回 JSON 化 7 字段：`task_id, lane, summary(≤40行), changed_files, test_status, artifact_paths, commit, downstream_deps`（+open_questions）。禁止原始日志入报告；证据走文件路径。报告存 `state/lane_reports/<task_id>.json`。

## 11. Supervisor 职责（orchestrator 承担）

1. 心跳：周期 `task_status` 快照 → `state/heartbeat.jsonl`；lane 卡死（idle 超阈值 + 无产出）→ 一次 steer，再无响应 → cancel + 重派（保留 session 供审计）。
2. 速率/配额：429/5xx 退避（§3）；API 用量以 subscription 并发窗为准。
3. GPU 互斥、checkpoint（账本 + 每 run `checkpoint_update/sha256`）、剩余时间预算（对照 §5 表每 30min 校验一次，超支触发止损表）。
4. 中断恢复：先 `pgrep` 查存活进程 → 校验产物与账本 → reconcile（`run_ledger reconcile` 标 interrupted）→ 再补派；**run_id 幂等，禁止重复提交**。
5. H30 硬停：结束可安全 checkpoint 的步骤，导出部分数据，生成 partial report。

## 12. 宪法合规映射

| 规则 | 落实 |
|---|---|
| R1 真值隔离 | 特征合法输入检查（T1.4）；lane 禁读 sealed/真值目录 |
| R2 物理标签唯一 | TS 验证语义不改；V5.3 只报 `TS_candidate` |
| R3 ACP 唯一后端 | 本阶段不启 QC；rollout 物理臂走 ACP 缝，mock 标 `physics_not_run` |
| R4 动作合法域 | K 由合法拓扑确定；非法输入 typed invalid |
| R5 全量成本留存 | 成本 ledger 强制字段（T4.4） |
| R6 评测完整性 | split 泄漏检查（T0.4）；fold3/4 一次性解封；确认折禁调参 |
| ADR-0028 §2.4 | 旧 53-run 证据只读：`outputs/pesn/v5_1/**`（21 runs）+ `outputs/pesn/v5_1_r1/**`（32 runs）（parallel_config `sealed_boundaries`，路径已按 EXP-2 实勘修正） |
| ADR-0029 | V5.1b/sealed/Confirm/OOD/Fit150+ 保持关闭；双门不互替 |

## 13. 待决策清单（不猜测，等用户）

1. **六表 vs 五表**：30h 计划六表（irc_curve/irc_node/irc_edge/reaction_intent/curve_quality/sample_view）vs V5.1-R3 v3 §2 五表（canonical_irc…）。建议以 30h 六表为准并回改 R3 v3，需确认。
2. **等变容差**：30h（rel 1e-5 + abs 1e-6，"建议"）vs R3 v3（abs 1e-4）。harness 双预设已建，gate 用哪个须冻结。
3. **run 数溯源口径**：v2"60-run"/30h"63→66"。执行值 66 已一致，叙述需统一。
4. **21-run 降级集**是否可在 `PES2TS_30H_R3_V52_V53` 下计为合法 R3 终局（涉及重新冻结规则）。
5. **campaign_id 写入**：`run_v5_1_r3.py` 是否固定 `PES2TS_30H_R3_V52_V53`。
6. **F-ablate mask 范围**：以 30h 全 mask（有序端点/signed role/CV 方向）为准？
7. `docs/agent_runs/` 原请求称"读取"，实际不存在——已按"新建执行套件"处理，请确认。
8. GPU 现有 2.6GB 占用是否可在启动前清理（正式训练需近全量 8GB）。

## 14. 并行 dry-run 结果（2026-10-20，全部完成并经 supervisor 复核）

| Lane | 任务 | commit | 测试（supervisor 复跑） | 结论 |
|---|---|---|---|---|
| W-DATA (wt-a) | R3 六表 schema/FK/泄漏校验器 + 4 组合成 fixture | `2eda543` | **18 passed**；CLI clean=0 / orphan=1 / dup_ts=1 / nan+leak=1 / 空目录=blocked | ✅ T0.1a/b/c、T0.4a/b 机器可查 |
| W-FEAT (wt-b) | 等变性/Relabel/Reverse/合法输入/NaN typed-invalid harness + 4 bug 模式 dummy | `121fbce` | **41 passed**；4 bug 模式均被恰好捕获；容差双预设显式警告 | ✅ T1.1–T1.4 可执行 |
| W-LOSS (wt-c) | run 账本（幂等/占位符拒绝/retry≤2+override）+ GPU flock 互斥 + 陈旧锁检测 | `e2133cf`+`efe5bce` | **11 passed**（supervisor 修正默认路径→`outputs/pesn/30h_campaign`） | ✅ supervisor 核心可用 |

**并发实测**：3 lane 同时运行无文件冲突（write_scopes 交集为空，仅预期共享空 `__init__.py`）；provider（zai-coding-plan 订阅）3 并发无 429/限速现象；Oracle 占用释放槽位（4 agent 瞬时仅发生于只读收尾与 lane 重叠窗口，已记为策略备注）。结构化 7 字段报告全部合规、无原始日志。
**遗留**：PD-9（delta_xyz 存储格式）；StarEGNN v3 forward 确定性（无 forward 内 dropout）列入 WP-2 验收；lane 间解释器不一致（系统 py3 有/无 torch、wt-b 用 /tmp/opencode/venv）→ 统一为：工具/测试=系统 python3，训练=PES2TS_GPU_PYTHON（miniforge pes2ts-gpu），E0 记录。

## 15. 启动前检查单

- [ ] §13 待决策清单用户裁定（**PD-1/2/4/6 为宪法阻塞**，E0 无法合法冻结，必须先裁；PD-9 E0 前）
- [x] dry-run 三 lane pass + supervisor 复核（§14）
- [ ] 三分支合并主树 + 主树全量 pytest（§6 合并前校验）
- [ ] **WP-L2 预启动硬化完成 + Oracle 复审 pass**（gate 强制化/账本字段/run_plan 分离/allowlist/deadline）
- [ ] E0 协议冻结件 hash 齐全 + 人工签认 + campaign 起止绝对时间写死
- [ ] GPU 占用查明归属并清理（PD-8）、磁盘 ≥100G、账本初始化
- [ ] 用户显式授权 `formal_training_authorized=true`（**不连带解封** sealed/Confirm/OOD/Fit150+，恒 false）
