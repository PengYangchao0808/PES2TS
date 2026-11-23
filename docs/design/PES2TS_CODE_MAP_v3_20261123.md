# PES2TS 代码地图 v3（五功能区 / 单一训练引擎）

> **状态**：Active-Reference（现行代码地图；非规范，规范以[开发宪法](../PES2TS_开发宪法_v1.md)与 [ADR-0036](decisions/ADR-0036-功能分区与单一训练引擎.md) 为准）
> **层级**：L1 导航 / 工程架构（C 工作流；不涉及 G0–G4 能力、V 战役语义与科学结论）
> **取代关系**：取代 [CODE_MAP v2](PES2TS_CODE_MAP_v2_20261018.md)（v2 保留为历史，不再维护）；机制规范见 ADR-0036
> **上游依赖**：[开发宪法 v2.9](../PES2TS_开发宪法_v1.md)、[AGENT.md](../../AGENT.md) §5.1、[ADR-0025](decisions/ADR-0025-G-V双轴映射与战役注册.md)、[ADR-0036](decisions/ADR-0036-功能分区与单一训练引擎.md)、[G–V 映射](PES2TS_G_V_MAPPING_v1_20261018.md)、[阶段定义](../plans/PES2TS_阶段定义_G0-G3.md)
> **下游消费者**：新会话/AI、代码任务拆分、导入边界审查、CI 门禁排查、归档判断
> **当前版本**：v3 @ 2026-11-23（依据迁移提交 `ccd31f2` 与 2026-11-24 未提交接线；执行证据见[执行记录](../reports/PES2TS_代码分区重构执行记录_v1_20261123.md)）

本图回答一个问题：**当前有效的代码在哪里、负责什么、由哪份文档拥有**。历史代码一律在 `archive/`（Zone 5，只读）或 `pes2ts_core/pesn/`（冻结模型区），不再在此图中逐个罗列。

## 1. 五功能区总览（ADR-0036 §1.1）

| Zone | 目录 | 职责 | 读写 | 拥有文档 |
|---|---|---|---|---|
| 1 内核 | `pes2ts_core/` | `utils/`、`g0/`、`g1/`、`generation/`、`integration/acp/`、`ranking/`、`evaluation/`；`pesn/` 为 V4 + V5.1–V5.4 冻结模型区 | 保护；`pesn/` **FROZEN 只读** | [开发宪法](../PES2TS_开发宪法_v1.md)、[阶段定义](../plans/PES2TS_阶段定义_G0-G3.md)、[MoveProcessContract v1](../contracts/PES2TS_MoveProcessContract_v1.md) |
| 2 数据 | `pes2ts_data/` | `contracts.py`（`StandardBatch`）、`splits/`（R6 防火墙）、`ingest/`（退役账本）、`registry.py`（数据仓版本锁）、`builders/`（构建器） | 活跃 | [本地数据资产与数据工程方案 v1](../plans/PES2TS_本地数据资产整合与数据工程方案_v1_20261123.md)、本文 §3.2 |
| 3 引擎 | `pes2ts_v5/` | **唯一**训练/评估引擎：`configs/`、`data/`、`models/`、`losses/`、`training/`、`evaluation/`、`runner.py`、`screening.py` | 活跃 | [V5 深度模型架构 Benchmark 方案 v2.1](../plans/PES2TS_V5_深度模型架构Benchmark方案_v2.1_20261122.md)、[V5 后续阶段训练大纲 v1](../plans/PES2TS_V5后续阶段训练大纲_v1_20261123.md)、ADR-0036 §1.3 |
| 4 基础设施 | `infra/` | `runtime/`（WSL/内存/线程/数据迁移/日志轮转）、`governance/`（文档门禁、链接检查、G–V 渲染、`arch_guard`）、`dev/` | 活跃（工具面） | [WSL 稳定运行与内存治理方案 v1](../infra/PES2TS_WSL稳定运行与内存治理方案_v1_20261122.md)、ADR-0036 §2 |
| 5 归档 | `archive/` | `v4/`、`v5x_scripts/{agent_ops,campaigns,campaign_oneoffs}/`、`v5x_configs/{campaigns,pesn}/` | **只读**（只进不出） | [归档方案 v2.0](../infra/PES2TS_开发文档规范化归档方案_v2_20261121.md)、[ADR-0035](decisions/ADR-0035-文档生命周期与退役政策.md)、[档案树](../archive_tree/README.md) |

支撑层（不单独成区，但受同一布局门禁约束）：`tests/{kernel,data,v5,legacy}/` 为四层测试；`config/{experiments,data,registry}/` 为活跃配置；`docs/` 为文档导航；`bin/pes2ts` 为 Classic CLI 入口。`scripts/` 根已清空，**不得复活**（`tests/kernel/test_repo_layout.py` 锁定）。

## 2. 依赖方向与五道硬门

依赖只允许自上而下（ADR-0036 §1.2）：

```text
pes2ts_v5  ->  pes2ts_data  ->  pes2ts_core(utils/g0/g1/...)
     \             |                  ^
      \            v                  |
       +-------> infra (工具面)       |
                                       |
archive  --只允许-->  infra.runtime    |   （文档化 sink 例外）
archive  --历史冻结调用-->  pes2ts_core   （grandfathered，不再扩展）
```

| 门 | 规则 | 实现 | CI/本地入口 |
|---|---|---|---|
| G1 单一训练循环 | 仅 `pes2ts_v5/training/**` 可出现 `.backward()` / `optimizer.step()` | `infra/governance/arch_guard.py` | `pytest -q tests/kernel` |
| G2 无战役版本代码文件 | 活跃区 `.py` stem 不得匹配 `^v(5[4-9]\|[6-9][0-9]*)_` | 同上 | 同上 |
| G3 单向 import | 上述依赖矩阵（AST 级；字符串/注释不算 import） | 同上 | 同上 |
| G4 冻结区只读 | `git diff --name-only HEAD` 不得触碰 `archive/**`、`pes2ts_core/pesn/**` | 同上（默认 advisory，`--strict-frozen` 强制） | `python infra/governance/arch_guard.py --json` |
| G5 活跃区 lint | `ruff check pes2ts_v5 pes2ts_data infra`，既有项须登记在 `arch_guard_baseline.json` | 同上 | 同上 |

日常完整校验：`python infra/governance/arch_guard.py --json`；快速 AST/结构校验：`python infra/governance/arch_guard.py --no-ruff`。`.rgignore` 已将 `archive/`、`tests/legacy/`、`pes2ts_core/pesn/models/v54*`、`pes2ts_core/pesn/**/*dual_ts*` 移出检索面。

## 3. 活跃模块地图

### 3.1 Zone 1 内核 `pes2ts_core/`（365 文件 / 约 16.2 万行；`pesn/` 冻结）

| 子包 | 代表模块 | 职责 | 状态 |
|---|---|---|---|
| `g0/`（33 文件 / 7,438 行） | `truth/truth_reader.py`、`split_manifest.py`、`rejections.py`、`inventory` | 数据底座、真值隔离（R1）、划分冻结 | 保护/活跃 |
| `g1/`（36 / 12,324） | `build`、`event_coupling`、`v2_*` | G1 化学先验编译/审计/导出 | 保护/活跃 |
| `generation/`（69 / 38,390） | `planning/`、`assembly/`、`execution/`、`quality/`、`control_round.py` | G2 Classic 控制器与 G4 控制回合接缝 | 保护（2026-10-18 后未改） |
| `integration/`（22 / 8,137） | `acp/`（唯一执行后端接缝，R3）、`validation/` | ACP 计算后端接缝 | 保护/活跃 |
| `pesn/`（185 / 91,073） | `contracts/`、`features/`、`models/`（含 `v54x/`、`dual_*`）、`offline/`、`bundle/`、`inference/` | **V4 + V5.1–V5.4 冻结模型与训练记录**；只读证据，不再演进 | **FROZEN** |
| `utils/`（5 / 610） | `hashing.py`、`jsonio.py`、`parquet_io.py`、`truth_guard.py` | 哈希/原子写/Parquet/真值静态守卫（fan-in 最高） | 活跃 |
| `ranking/`、`evaluation/` | `__init__.py` 薄封装 | 预留命名空间 | 保留 |
| 顶层 13 文件 | `cli.py`（约 2,000 行、45 handler）、`contracts.py`、`config_loader.py`、`gv_registry.py`、`flywheel.py`、`viewer.py` 等 | Classic CLI 与跨模块合同 | 活跃（拆 `cli/parser.py` 为债务，未实现） |

入口：`bin/pes2ts` → `pes2ts_core.cli`（G0/G1/G2/G2.T/ACP 命令）。

### 3.2 Zone 2 数据 `pes2ts_data/`（v1.0.0，14 文件 / 9,201 行）

| 模块 | 内容 | 职责 |
|---|---|---|
| `contracts.py` | `StandardBatch`、`FIELD_SPECS`、`validate_batch`、`ContractViolation`、`to_standard/from_standard` | **唯一数据批次合同**：dtype/shape/provenance、R1 真值字段拒绝、R5 成本权重、legacy V5.6 compat 桥 |
| `splits/policy.py` + `config/data/splits.yaml` | `SplitPolicy`、`load_split_policy`、`assert_no_family_leak`、`assert_sealed`、`check_split_frame` | R6 划分防火墙（L0–L4 链、四测试集、2,000 封印 confirm） |
| `ingest/ledger.py` | `load_campaign_index`、`load_ledger`、`verify_entry`、`summarize` | 退役战役 tombstone/成本账本读取（R5；只读、类型化拒绝） |
| `registry.py` | `DatastoreRegistry`、`load_registry`、`validate_registry`、`RegistryError` | 数据仓版本锁合同（`01_master_index/datastore_registry.json`，schema `pes2ts_datastore_registry_v1`） |
| `builders/`（6 构建器 + `__init__`） | `build_master_split.py`、`build_frame_crosswalk.py`、`build_channels.py`、`build_g1_summary.py`、`archive_campaigns.py`、`register_datastore.py` | 数据仓构建脚本（原 `scripts/datastore/`）；只被人工/流水线调用，包导入时不加载 |

### 3.3 Zone 3 引擎 `pes2ts_v5/`（55 文件 / 7,654 行）

| 模块 | 内容 | 职责 |
|---|---|---|
| `runner.py`（559 行） | `train` / `eval` / `sweep` / `smoke` 四个子命令，JSON 输出 | **唯一 CLI 入口**（`python -m pes2ts_v5.runner`；`__main__.py` 转发） |
| `screening.py` | 历史 screen 入口别名，转调 `runner sweep` | 兼容旧 `scripts/v5/run_screening.py` 参数 |
| `configs/schema.py` | `ExperimentConfig`、`load_experiment_config`、`config_hash` | **唯一实验配置合同**（`config/experiments/*.yaml`；未知键拒绝、NaN 拒绝、canonical sha256） |
| `data/` | `manifest.py`（`V55DatasetSpec`）、`trajectories.py`、`idx_v2.py`、`split.py`、`labels.py`、`contracts.py`（StandardBatch 懒桥） | V5.5 数据/标签接口与确定性 synthetic fixture（缺数据时诚实报 `V55ContractNotAvailable`） |
| `models/` | `registry.py`（14 cores + 2 encoders + heads 注册表）、`v5_model.py`、`heads.py`、`cores/m01…m14`、`encoders/` | 架构插件注册；`stregnn_v2` 缺输入合同时以 `ENCODER_INPUT_CONTRACT_MISMATCH` 拒绝，不伪造数据 |
| `losses/` | `registry.py`（loss component 插件）、`components.py` | 加权损失组件；未实现组件注册为 `implemented=False` 并拒绝训练 |
| `training/` | `trainer.py`（`V5Trainer`，唯一训练循环）、`callbacks.py`（checkpoint/cost ledger/OOM guard/早停）、`sweep.py`、`_util.py` | 统一 trainer + callbacks |
| `evaluation/` | `single_step.py`、`rollout_q.py`、`rollout_kinematic.py`、`irregular_sampling.py`、`ood.py`、`cpu_benchmark.py`、`_common.py` | 单步/rollout/OOD/CPU 基准 |
| `reports/` | `experiment_registry.csv`、`leaderboard.csv`、`failure_taxonomy.csv`、`model_selection.md` | 实验登记与选择规则产物 |

### 3.4 Zone 4 基础设施 `infra/`（23 文件 / 15 py / 5,676 行）

| 模块 | 内容 | 职责 |
|---|---|---|
| `runtime/`（12 文件） | `wsl_guarded_run.sh`、`cgroup_scope.sh`、`wsl_reclaim.sh`、`wsl_stability_check.ps1`、`host_commit_monitor.ps1`、`infra_bench.py`、`concurrency_drill.py`、`thread_env_ab.py`、`migrate_data_to_ext4.py`、`sha_sidecar.py`、`log_rotate.py`（2026-11-23 新增） | WSL/内存治理、受限运行、数据迁移、sha256 sidecar、审计日志轮转 |
| `governance/`（5 文件） | `docs_consistency.py`（+ config）、`check_docs_links.py`、`render_gv_map.py`、`arch_guard.py`（+ baseline） | 文档一致性门、链接门、G–V 渲染、五道架构硬门 |
| `dev/` | `_capture_r1_golden.py` | 本地环境辅助 |

### 3.5 Zone 5 归档 `archive/`（240 文件 / 111,728 行；只读）

| 子区 | 内容 | 数量（2026-11-23 实测） |
|---|---|---|
| `v4/` | V4 dual-TS 脚本 + `campaign_oneoffs/`（14 个 root 一次性脚本） | 38 文件 / 37 py / 5,895 行 |
| `v5x_scripts/agent_ops/` | V5.4.x 战役工具（config schema/stats/rollout/ledger/data_gate/feature_gate） | 65 文件 / 65 py / 60,406 行 |
| `v5x_scripts/campaigns/` | V5.0–V5.4.3 战役驱动 | 32 文件 / 31 py / 26,896 行 |
| `v5x_scripts/campaign_oneoffs/` | 66 个 root 一次性脚本 | 67 文件 / 67 py / 18,529 行 |
| `v5x_configs/campaigns/` | 10 战役 yaml + 13 预注册模板 | 23 文件 |
| `v5x_configs/pesn/` | PES-N 历史配置 | 11 yaml |

### 3.6 测试与配置布局

| 层 | 路径 | 内容 | 默认执行 |
|---|---|---|---|
| kernel | `tests/kernel/` | 文档一致性门测试、`arch_guard` 五门测试、仓库布局锁定（`test_repo_layout.py`） | `pytest -q tests/kernel`（CI base，无 torch） |
| data | `tests/data/` | `StandardBatch`、split firewall、ledger、registry 合同测试 | `pytest -q tests/data` |
| v5 | `tests/v5/` | 引擎合同：config schema、model registry、runner smoke、single train loop、trainer contract | CI `pesn` 全量层 |
| legacy | `tests/legacy/{pesn,agent_ops,campaigns,root}/` | V4/V5.1–V5.4 历史回归（315 文件 / 215 py） | 收集时自动打 `legacy`，默认跳过；CI 全量层执行 |
| root | `tests/*.py` | 活跃 G0–G2/A/ACP 回归（约 128 个 tracked 模块） | 默认层 |
| config | `config/experiments/*.yaml`（5）、`config/data/splits.yaml`、`config/registry/g_v_map.yaml` | 活跃实验/划分/G–V 注册表 | — |

pytest 默认门槛：`pytest.ini` 的 `addopts = -m "not realdata and not xtb and not acp and not orca and not legacy"`。

## 4. 任务 → 区 → 入口文件

| 任务 | Zone | 入口/主文件 | 备注 |
|---|---|---|---|
| 训练一个模型 | 3 | `python -m pes2ts_v5.runner train --config config/experiments/<name>.yaml` | 唯一训练循环 `pes2ts_v5/training/trainer.py` |
| 评估一个 run | 3 | `python -m pes2ts_v5.runner eval --run-dir outputs/v5/runs/<run>` | 输出 JSON |
| 架构扫描/对比 | 3 | `python -m pes2ts_v5.runner sweep --config config/experiments/v56_arch_matrix.yaml --arch M01 --arch M04` | 或兼容入口 `python -m pes2ts_v5.screening` |
| 快速 smoke | 3 | `python -m pes2ts_v5.runner smoke --config config/experiments/v56_baseline_gru.yaml` | 只做构建 + 单次前反向 |
| 新增/校验数据批次 | 2 | `pes2ts_data/contracts.py`（`StandardBatch`、`validate_batch`） | legacy V5.6 batch 用 `to_standard` |
| 划分策略/封印集 | 2 | `pes2ts_data/splits/policy.py` + `config/data/splits.yaml` | 先 `load_split_policy`，再 `assert_no_family_leak` / `assert_sealed` |
| 登记数据仓 | 2 | `pes2ts_data/builders/register_datastore.py` + `pes2ts_data/registry.py` | 写 `01_master_index/datastore_registry.json` |
| 构建数据仓产物 | 2 | `pes2ts_data/builders/{build_master_split,build_frame_crosswalk,build_channels,build_g1_summary}.py` | 原 `scripts/datastore/` |
| 查询退役成本账本 | 2 | `pes2ts_data/ingest/ledger.py` | 只读；缺件报 `MISSING` 不抛异常 |
| 文档一致性门 | 4 | `python infra/governance/docs_consistency.py --strict` | 禁用语/historical 横幅/required files |
| 文档链接门 | 4 | `python infra/governance/check_docs_links.py` | 0 error 为通过 |
| G–V 映射渲染/校验 | 4 | `python infra/governance/render_gv_map.py --check`（写用 `--write`） | 生成物禁止手改 |
| 架构五门 | 4 | `python infra/governance/arch_guard.py --json`；`pytest -q tests/kernel` | ADR-0036 §2 |
| WSL 受限运行 | 4 | `infra/runtime/wsl_guarded_run.sh --mem <SIZE> --name <tag> -- <cmd>` | 6G/16G 内存硬上限 + cgroup 采样 |
| 审计日志轮转 | 4 | `python -m infra.runtime.log_rotate --path data/manifests/truth_access_log.jsonl --max-bytes 64MiB --dry-run` | 定时任务调用；见模块 docstring |
| 查历史战役证据 | 5 | `archive/v4/`、`archive/v5x_scripts/`、`archive/v5x_configs/` | 只读；不可被活跃区 import |
| Classic CLI（G0–G2/ACP） | 1 | `bin/pes2ts` → `pes2ts_core/cli.py` | 不含 torch/引擎依赖 |

## 5. v2 中「planned 但不存在」的 `.py`：NOT_IMPLEMENTED 清单

**方法**（2026-11-23 于工作树复核）：以[代码探查报告 v1](../reports/PES2TS_代码探查报告_v1_20261123.md) 的 991 文件/443,792 行清点为基线，从 [CODE_MAP v2](PES2TS_CODE_MAP_v2_20261018.md) 提取全部 `.py` 字面引用 → 去重 78 条 → 与全仓 `.py` 做路径/后缀匹配 → **39 条可解析、39 条不可解析**；不可解析引用合并重复写法（同一文件的全路径与裸文件名/短后缀）后为 **30 个不同目标**。其中 **28 个属未实现的规划模块/重构项**，2 个是 v2 明文记录为“已归档旧名”的历史文件名（`v6_progress.py`、`v6_decoder.py`）。

| 组 | 目标（v2 写法） | v3 状态 | 说明 |
|---|---|---|---|
| V5.2-R | `pes2ts_core/pesn/contracts/reaction_intent.py` | NOT_IMPLEMENTED | v2 标 planned |
| V5.2-R | `pes2ts_core/pesn/features/reaction_coords.py` | NOT_IMPLEMENTED | planned |
| V5.2-R | `pes2ts_core/pesn/models/key_atom_heads_v52r.py` | NOT_IMPLEMENTED | planned（含裸名重复引用） |
| V5.2-R | `pes2ts_core/pesn/models/key_atom_loss_v52r_v3.py` | NOT_IMPLEMENTED | planned |
| V5.2-R | `archive/v5x_scripts/campaigns/r3_v52r_run.py` | NOT_IMPLEMENTED | planned；战役已 Closed |
| V5.3/V5.4 | `archive/v5x_scripts/campaigns/run_v5_2_progress_stop.py`、`run_v5_3_growth.py`、`run_v5_4_any_point.py` | NOT_IMPLEMENTED | planned |
| V6 | `pes2ts_core/pesn/models/v6_physics_mlp.py`、`v6_physics_cnn.py` | NOT_IMPLEMENTED | planned |
| V6 | `pes2ts_core/pesn/offline/data/xtb_probe.py`、`offline/training/v6_train.py`、`offline/evaluation/physics_protocols.py` | NOT_IMPLEMENTED | planned |
| V6 | `pes2ts_core/integration/acp/xtb_freeze_request.py`、`xtb_freeze_transport.py`、`integration/validation/xtb_ts_verification.py` | NOT_IMPLEMENTED | v2 无 planned 标记但文件不存在 |
| V6 | `archive/v5x_scripts/campaigns/run_v6_1_mlp.py`、`run_v6_2_cnn.py`、`run_v6_3_verify.py` | NOT_IMPLEMENTED | planned |
| V7 | `pes2ts_core/pesn/contracts/qc_transition.py` | NOT_IMPLEMENTED | 无 planned 标记但不存在 |
| V7 | `archive/v5x_scripts/campaigns/run_v7_1_env.py`、`run_v7_2_offline_rl.py`、`run_v7_3_online_rl.py` | NOT_IMPLEMENTED | planned |
| 债务 | `pes2ts_core/g1/key_atom_audit.py`、`generation/planning/key_atom_writer.py` | NOT_IMPLEMENTED | v2 标“新增”，实际不存在 |
| 债务 | `tests/test_stage_boundaries.py` | NOT_IMPLEMENTED | v2 §4 声明由该测试强制，实际不存在（现行边界门为 `tests/kernel/test_arch_boundaries.py`） |
| 债务 | `pes2ts_core/cli/parser.py` | NOT_IMPLEMENTED | v2 §7 债务，`cli.py` 未拆分 |
| 历史名 | `v6_progress.py`、`v6_decoder.py` | HISTORICAL_NAME_ONLY | v2 明文记录 2026-10-20 已归档，代码不在树内 |

> 结论：v2 的 39 条不可解析引用**不是丢失代码**；v3 起以本表替代 v2 §3 的 planned 区块，`archive/` 中的历史实现仍是唯一可检索证据。

## 6. 冻结区与历史落点

- **`pes2ts_core/pesn/**`（FROZEN，ADR-0036 §1.4）**：V4 双端线与 V5.1–V5.4 模型/训练/评测实现的只读记录；允许历史调用（grandfathered），不得新增/改语义。真实修复需 ADR + `[unlock-legacy]` 提交说明（ADR-0036 §5）。
- **`archive/**`（FROZEN，只读）**：V4 全部脚本（`archive/v4/`，含 14 个 root 一次性脚本）与 V5.0–V5.4.3 战役脚本/配置。只允许 import `infra.runtime`（文档化 sink 例外），活跃区不得反向 import。
- V4/V5.x 的结论仍以报告为准，代码位置不改变证据效力：如 [V5.4.3 双 Decoder 实施与训练报告](../reports/PES2TS_V543_双Decoder_50K_实施与训练报告_20261106.md)、[V5.3-MP 规模训练收益报告](../reports/PES2TS_V53MP_规模训练收益报告_2k_to_50k_20261027.md)。

## 7. 存储分级（2026-11-23 现状）

| 类别 | 路径 | Git |
|---|---|---|
| 活跃源码/测试/config | `pes2ts_core/`、`pes2ts_data/`、`pes2ts_v5/`、`infra/`、`tests/{kernel,data,v5}` + `tests/*.py`、`config/{experiments,data,registry}` | tracked |
| 历史源码/配置 | `archive/**`、`tests/legacy/**` | tracked（只读） |
| 小 manifest/审计 | `data/manifests/*.json`（含 `retired_scripts_local_20261123.json`） | tracked（按需） |
| 大 manifest/运行产物 | `data/interim/**`、`outputs/**`、`artifacts/**`、`.slim/**` | ignored |
| 统一数据仓 | WSL `/pes2ts_datastore`（registry：`01_master_index/datastore_registry.json`） | 仓外（WSL ext4） |
| 审计日志 | `data/manifests/truth_access_log.jsonl`（约 326,877,347 B） | ignored；由 `infra/runtime/log_rotate.py` 轮转 |

## 8. 已知债务（不在本次重构中解决）

1. `pes2ts_core/cli.py` 约 2,000 行/45 handler，`cli/parser.py` 拆分未实现（v2 债务延续）。
2. `generation/` ↔ `integration.acp` 双向耦合；`generation/control_round.py` ↔ `pesn.offline` 循环（V4 线冻结，不扩展）。
3. `infra/governance/arch_guard_baseline.json` 现有 **5 条 G3 历史容忍（17 个 finding）**：`render_gv_map` 读 G–V registry、`infra/runtime` 三个旧战役 benchmark 脚本 import 历史 campaign 代码等（G5 lint 债务已清零，baseline 收敛后 `arch_guard --json` 为 clean）。
4. `pes2ts_v5/evaluation/` 尚未覆盖 `tests/v5` 的全部架构（14 cores 中仅部分列入矩阵 smoke）。
5. `config/registry/g_v_map.yaml` 仍有 11 个 `config:` 指向不存在的 YAML（均属 `proposed`/`archived` 条目，validator 不强制；属 owner 待决，见[执行记录](../reports/PES2TS_代码分区重构执行记录_v1_20261123.md) §6）。

## 9. 变更记录

| 版本 | 日期 | 变更 |
|---|---|---|
| v2.2 | 2026-10-20 | V5–V7 reaction-intent 修订（历史） |
| v3 | 2026-11-23 | 全面改写为五功能区（ADR-0036）：新增 Zone 2/3/4/5 与 tests/config 布局；标记 `pes2ts_core/pesn/**` FROZEN；v2 的 39 条不可解析 `.py` 引用收敛为 28 个 NOT_IMPLEMENTED + 2 个历史名；新增任务→入口表与五道硬门 |
