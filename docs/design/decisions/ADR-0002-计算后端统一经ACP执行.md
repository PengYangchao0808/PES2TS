# ADR-0002 计算后端统一经 ACP 执行（部分取代 ADR-0001 的本地封装条款）

> 状态：已接受（Accepted）
> 层级：工作流 X（执行收敛），服务 G2.4
> 日期：2026-10-03
> 决策者：PES2TS 开发负责人
> 取代关系：**部分取代** ADR-0001 的 Decision（执行位置：XTB_PATH 从"本地封装"改为"经 ACP 执行"）与 X1/X2 的退出条件（"逐字节一致的本地封装证书"）；**保留** ADR-0001 的执行接缝（`ExecutionBackend` 协议）、计划（`GenerationPlanV2`）、结果记录（`TrajectoryRecord`）与 method 词表
> 下游：X1′–X5′ 实现、`pes2ts-generation-v1` 发布
> 上游依赖：ADR-0001、`PES2TS_整体开发与发布方案_v2`、`PES2TS_阶段定义_G0-G3`

> 实施快照补注（2026-10-03）：下文“背景”保留决策形成时的状态；后续只读核查已看到 ACP 工作区的 `XTBBackend.path_search` 与新 `workflows/xtb_path.py`，尚未完成全部注册与联合冒烟。当前能力缺口、梯度/计量/验证补齐及 ACP calculations 基元落点以 [联合开发计划](../../plans/PES2TS_G2_ACP能力补齐与联合验收计划_20261003.md) 为实施细化。本 ADR 的 ACP 唯一后端、配方/判定归属、旧产物只读和等价证书决策保持有效。

---

## 背景（Context）

ADR-0001 裁决"冻结执行接缝、推迟合并实验代码"，并把 `XTB_PATH` 定义为一个**本地** `ExecutionBackend`——包住现有 G2 本地 `subprocess` 管线（`generation/execution/xtb_path/runner.py`），用"逐字节一致等价证书"证明零行为改变。

现状与新的项目指令：

1. **项目边界早已声明**："ACP 负责任务执行/预算/日志/溯源；PES2TS 不把执行基础设施内化"。但 G2 的 xTB PATH 至今仍是本地 `subprocess`（`runner.py:run_xtb_path` 起 `xtb` 二进制），**实现与声明边界相悖**。
2. **新指令**：PES2TS 不再内化任何独立计算引擎；**所有计算化学后端统一经 ACP 执行**，以便可视化与溯源。
3. **ACP 侧事实**（已核实）：`cccp/qc/interfaces/xtb_path.py::XTBPathInterface.path_search` **已完整实现** `xtb --path` 元动力学（写 `path.inp`、切帧到 `path_frames/*.xyz`、解析逐帧能量、RPH 兼容），但**是孤儿**——未被任何 workflow/catalog/CLI/scheduler/manifest 调用；ACP 的平台件（workflow catalog、`acp.cli`、`JobRunner`、`ResultManifest v2`、WORK/RESULT 布局、`/s2/profile` 可视化 API）均存在且可复用。
4. **PES2TS 侧仍有 in-process cccp 调用**：`integration/acp/continuation_backend.py`（`ORCAInterface`）、`gradient_backend.py`（`ORCAGradientBackend`/`ORCALocalCorrector`）、`seed_validation.py`（`cccp.qc.interfaces.orca_ts`），同样违反"不内化引擎"。
5. **已存在 191,148 条本地 G2 产物**；ADR-0001 明确"不中途迁移 19.1 万条"。

本 ADR 与 ADR-0001 的接缝决策**不冲突**：ACP 只是"执行位置"的改变，`ExecutionBackend` 协议仍是唯一的执行接口，只是其后端从"本地进程"变为"ACP 编排"。

---

## 决策（Decision）

1. **PES2TS 不实现任何独立计算引擎。** 所有计算化学后端（`XTB_PATH` / `ORCA_SCAN` / `ORCA_CONSTRAINTS` / `ORCA_NEB` / `CONTINUATION`）统一经 ACP 执行。
2. **传输通道 = ACP CLI**（`python -m acp.cli run <workflow> ...`），沿用现有 `ACPCLIBackend` / `stage_cli` 的 receipt / claim / log / manifest-v2 / timeout 机制。
3. **删除本地 runner**：`generation/execution/xtb_path/runner.py` 整体移除（含 `run_xtb_path` / `resolve_executable` / `write_path_inp` / `_probe_seed_flag` / `XtbRunResult` / `XtbNotFoundError`）。纯派生层 `xtb_output.py`（解析）与 `inspect.py`（逐帧指标与有效性判定）**保留**，因为它们是 truth-free 的解析/判定，不是引擎。
4. **ORCA 也迁入 ACP**：`continuation_backend.py`、`gradient_backend.py`、`seed_validation.py` 的 in-process cccp 调用改为 ACP workflow 请求，随后移除这些模块的 cccp 依赖。
5. **旧产物不动**：191k 本地产物保持只读；`g2_path_v1` schema 不变，旧/新人群保持可比。本地 runner 删除后，旧产物 + 保留的解析/判定层共同充当**回放 oracle**。
6. **配方（`path.inp` 与执行参数）由 PES2TS 拥有**：ACP 是忠实执行者，只校验不补默认值。这是"配方等价"可成立、且物理不被 ACP 默认值漂移污染的前提。
7. **判定权留在 PES2TS**：`g2_path_v1` 的有效性判定仍由 `inspect.evaluate_validity` 产生；反向重试策略仍由 PES2TS 决定（换端点再发一次 ACP attempt），ACP 不自行重试。
8. **可视化需要 ACP 任务身份**：CLI 模式不产生 `task_id`，Workbench 的 `/s2/profile` 无法解析；因此设"可视化桥"（登记 CLI 尝试为 ACP 任务，见下）。

---

## 等价性证书（取代"逐字节一致"）

端到端"原始输出逐字节一致"**不可达**：xTB PATH 是元动力学（不可复现），且 ACP `path_search` 的默认值（`npoint=25`/`alp=1.2`、gfn==2 时不传 `--gfn`、`--uhf` 仅 >0 时传、无 seed 探针）与 PES2TS 冻结配方（`npoint=50`/`alp=0.5`、恒传 `--gfn 2`/`--uhf`）**本就不同**。改用两段式证书：

| 证书 | 定义（大白话） | 能证明什么 | 确定性 |
|---|---|---|---|
| **RecipeEquivalence（配方等价）** | 把"发给 ACP 的 `path.inp` 文本 + argv（除可执行路径）+ `--gfn`/`--uhf`/电荷/多重度/线程/超时/`OMP_NUM_THREADS`"与"旧版冻结配置生成的同一组内容"逐字节比对 | 没有意外改变物理旋钮 | 是（字节级） |
| **ReplayParity（回放等价）** | 把同一份**历史存档原始轨迹**喂进新的解析+判定+投影管线，产出 `frames.parquet` 与 `reaction_path.json` 必须与当年本地产物**逐字节一致** | 换了执行器，但"解析/判定/投影"三步没变 | 是（给定相同输入字节） |
| 化学分布等价 | N 条反应上 validity 通过率/帧数/势垒分布无显著差异 | 群体行为一致 | 否（**验收指标，非证明**） |

**规则**：永不比较两次真实运行的原始输出；只比较"输入配方"与"给定固定轨迹的派生物"。证书产物命名为 `equivalence_report.json`，随对照子集冻结。

---

## 确定性与溯源契约

把确定性重述为 **deterministic(inputs + 冻结配方 + 原始轨迹 digest)**：

- **冻结（确定性）**：`path_config.json`、`path.inp` 字节、argv、电荷/多重度、线程/超时、config digest、可执行 `sha256` + 版本行、seed 支持。
- **确定性派生**：`reaction_path.json`、`frames.parquet`、summary/manifest/coverage —— 给定相同原始轨迹字节则逐字节一致；把 `raw_trajectory_sha256` 登记为输入。
- **非确定、仅记哈希**：原始轨迹、wall time、ACP `task_id`/`attempt_id`/receipt/scheduler 状态/时间戳。

**hashes-not-state 规则**：可变调度元数据放入 `TrajectoryRecord.provenance["acp"]`，**排除出内容哈希与字节比较**（等同 `VOLATILE_KEYS`）；只把其 **digest**（`request_sha256` / `manifest_sha256` / `raw_trajectory_sha256`）纳入哈希。`plan_sha256` 继续绑定冻结计划。

---

## 仓库边界（两侧最小改动集）

### ACP 侧（平台接线，非造引擎）
1. `src/cccp/qc/interfaces/xtb_path.py`：接口保真——接受 `path_inp_text` 全量透传、`extra_args`、**无条件传 `--gfn`/`--uhf`**、seed 处理。
2. `src/acp/backends/xtb.py`：新增 `XTBBackend.path_search(...)` 委托接口。
3. 新增 `src/acp/workflows/xtb_path.py`：`run_xtb_path_search(...)` → 落 start/end → 调接口 → `build_xtb_path_profile` → 持久化 → 注册产品。
4. `acp/catalog.py` + `acp/workflows/registry.py`：新增 `XtbPathSearch`（active、`default_backend="xtb"`、`requires_binaries=["xtb"]`）。
5. `acp/cli.py` + `acp/scheduler/jobs.py`：`run XtbPathSearch --path-config <req.json>`（照 `--scan-config` 模式，新增 `PATH_CONFIG_FILENAME="path_config.json"`）。
6. （ORCA 迁移用）`BatchOptimize`/`irc`/constraint 已有；新增 gradient/continuation 所需的 workflow 或复用现有 SP/constraint workflow。
7. （可视化桥，可选）CLI `--register` / `--job-id` 把 CLI 尝试登记进 jobs 库。

### PES2TS 侧
1. 新增 `integration/acp/xtb_path_request.py`：`build_path_request(plan, materials) -> dict`（start/end xyz_text、电荷、多重度、字面 `path.inp`、线程/超时、provenance 标记）+ `request_sha256`。
2. 新增 `generation/execution/xtb_path/acp_backend.py`：`XtbPathACPBackend`（`method="XTB_PATH"`，注入 `ACPTransport`），把 ACP RESULT 投影为 `TrajectoryRecord` 与既有 `g2_path_v1`。
3. `integration/acp/cpc_transport.py`（或复用 `stage_cli._run_stage`）：通用 ACP CLI 传输，workflow 参数化。
4. `pipeline.py`：`_run_attempt` 由"调 `run_xtb_path`"改为"调 ACP 传输"；`_xtb_fingerprint` 由"本地解析可执行"改为"读 ACP provenance"；移除 runner 相关 import/常量。
5. `cli.py`：`XtbNotFoundError` 的 import 与处理改为 ACP 侧错误类型。
6. 能力登记 `orca_capabilities_v1.json`：`xtb-native-path`/`g2-path-adapter` 的 adapter 由 `g2-path-runner` 改为 `acp`；新增 ACP path 探针收据；`PATH_METHOD_KINDS` 扩入 `XTB_PATH`。
7. `utils/truth_guard.py`：从 `DYNAMIC_EXEC_ALLOWLIST` 移除 runner.py；新起进程模块加入 `SUBPROCESS_IMPORT_ALLOWLIST`。
8. 清理：删 `runner.py`；重写/删除 `tests/test_g2_runner.py`、`tests/test_g2_xtb_smoke.py`；归档 `scripts/` 一次性脚本。

---

## 可视化契约与桥接

- ACP workflow 产出 `RESULT/result_manifest.json`（v2），产品：`pes_profile`（`pes_profile_v2`，`source="xtb_peb"`）+ `trajectory`（多帧 xyz）+ 逐帧 `structure`。
- 复用 `/api/v1/jobs/{id}/s2/profile`、`/s2/frame/{i}`（先确认 `normalize_pes_profile` 接受 `source="xtb_peb"`；否则在持久化阶段做映射）。
- **CLI 模式不产生 `task_id`**，因此：
  - 文件级查看：CLI 已产出 ACP 标准 WORK/RESULT + profile，可直接离线查看；
  - Workbench 图查看：需"可视化桥"——追加一次 `登记 CLI 尝试为 ACP 任务（拿 job_id）`，或 ACP CLI 增 `--register`。**这是达到"尽可能可视化"目标的必要子项。**

---

## 迁移阶段（X 工作流重定义）

> 说明：ADR-0001 的 X0（协议 + `TrajectoryRecord` + method 词表）已完成。以下 X1′–X5′ 重定义 X1–X4，并新增 X5′。

| 阶段 | 对应旧 X | 交付 | 退出条件 |
|---|---|---|---|
| X1′ | 旧 X1（ACP 侧） | ACP xTB PATH 执行器：接口保真 + `path_search` 后端 + `XtbPathSearch` workflow + catalog/CLI + manifest/profile | ACP fixture：`path.inp` 字节相等；产出可归一化 `pes_profile_v2`；ACP 单测绿 |
| X2′ | 旧 X1（PES2TS 侧） | PES2TS ACP 后端（CLI 传输）+ 请求投影 + `g2_path_v1` 投影 + **删除 runner** | RecipeEquivalence 通过；truth_guard 干净；套件绿 |
| X3′ | 旧 X2/X3 | ReplayParity 证书；`TrajectoryRecord` → `PathBundle` 投影；统一门 | ReplayParity 逐字节通过；191k 人群执行零变化 |
| X4′ | 旧 X4 | ORCA/NEB/continuation 全量迁 ACP；移除 in-process cccp | 未冒烟能力诚实拒绝；无 cccp import |
| X5′ | 新增 | 可视化桥 + `g2.xtb.backend=acp` 默认 + 文档同步 + 旧代码清理收尾 | Workbench 可渲染；`g2 verify` 绿；旧产物不动 |

---

## 风险（按严重度）

1. **物理漂移**（关键）：ACP `path_search` 默认参数/缺 `--gfn`/`--uhf` 会产出**不同科学结果** → 配方 PES2TS 拥有 + 接口保真；RecipeEquivalence 拦住。
2. **能力未探针被拒**：`probe_receipts: []`、`PATH_METHOD_KINDS` → X1′/X2′ 写入收据后再启用。
3. **可视化断裂**：CLI 无 `task_id` → 必须显式做可视化桥，否则达不到目标。
4. **truth_guard 白名单**：删 runner + 加新起进程模块，两张表同步，否则守卫报错或误放行。
5. **跨仓库版本漂移**：CLI vs Workbench、ACP 版本演进 → 单一请求 schema + 单一结果归一化器；provenance 钉 ACP commit + `adapter_version`。
6. **反向重试语义**：留在 PES2TS，ACP 不重试。
7. **CPU 时间**：ACP CLI 只给 wall time → `cpu_seconds: null` 显式未知。
8. **测试覆盖断层**：删 `test_g2_runner.py`/`test_g2_xtb_smoke.py` 后需以 ACP 对照测试补齐，避免"删测试=通过"。

---

## 非目标（Non-goals）

- 不在 PES2TS 内保留任何计算引擎或 in-process cccp 计算调用。
- 不让 ACP 拥有配方默认值（配方归 PES2TS）。
- 不修改 `ACPCLIBackend`/`ACPValidationCLIBackend` 既有 `PESsearch` 契约（新增 xtb-path 走通用 stage 传输）。
- 不迁移 / 不强制重跑 191k 旧产物。
- 不建插件/注册框架（协议 + 按 method 分派 dict）。
- 不重封已冻结计划（新增 method 会改新计划 `content_sha256`，旧计划不动）。

---

## 后果（Consequences）

**正面**：一个执行体系（全部经 ACP）；执行基础设施外置，PES2TS 回归"计划 + 判定 + 投影"；ACP 统一日志/产物/manifest 便于可视化与溯源；旧产物仍可比。

**负面 / 成本**：需 ACP 侧接线（跨仓库）；需接口保真变更；需维护 ACP 传输层与能力探针；删除本地 runner 后需以证书+对照测试重建信心；可视化需额外登记桥。

**必须遵守**：真值守卫两张白名单同步；能力 `unknown` 必须拒绝；`g2_path_v1` schema 不变以免旧人群失联。

---

## 状态

已接受。实施排入 G2.4（开发序列 P2→P4）。第一根桩为 X1′+X2′ 的 RecipeEquivalence 证书与本地 runner 删除。
