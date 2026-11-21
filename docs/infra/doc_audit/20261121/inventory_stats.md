# PES2TS 开发文档盘点统计报告 (inventory_stats.md)

- 采集时点 (快照): 2026-11-21 22:32:48 (本清单为**该时点**仓库状态; 此后新增/修改的文档不在本清单内)
- 仓库根: `E:/Calculations/AI4S_ML_Studys/PES2TS`
- 采集脚本: `collect_inventory.py` (同目录, 可复跑: `python collect_inventory.py`)
- 机器可读清单: `doc_inventory.csv` / `doc_inventory.json`
- **记录总数: 253** 条; 总体积 **4,972,194 B** (4.742 MB); 总行数 55,452 行

## 0. 采集口径 (CALIBER)

**扫描范围 (白名单)**:

| 位置 | 规则 | 命中 | 说明 |
|---|---|---:|---|
| 仓库根 | `*.md` + `*.txt` + `LICENSE` | 15 | 项目入口/宪法投影/战役 memo/依赖清单/许可证 |
| `docs/` | 递归 `*.md` + `*.html` | 223 | 不含 data/state/fixture 类文件(见下) |
| `examples/` | 递归 `*.md` + `*.html` | 3 | 示例说明 + 可视化 html |
| `artifacts/` | 递归 `*.md` | 8 | 仅 8 份 `model_card.md` |
| `scripts/` | 递归 `*.md` | 2 | 脚本目录 README |
| `data/` + `outputs/` | 仅顶层 `*.md` | 2 | 只取两个 README,不递归(12GB/1.3GB 大数据目录) |
| `config/` | 递归 `*.md`/`*.txt`/`*.rst` | 0 | 实测 config 下无说明文档,只有 yaml/ini |

**排除项**: `.git/`、`.slim/`(工作区)、`data/` 数据文件、`outputs/` 大文件、`__pycache__`/`.pytest_cache`/`.ruff_cache`/`node_modules`/`.venv` 缓存目录、以及 `docs/` 下非文档类文件。

`docs/` 下按口径排除的非文档文件共 **69** 个: `.txt` 35、`.json` 30、`.csv` 2、`.lock` 1、`.py` 1(状态 JSON、归档快照 `.snapshot.txt` 字节副本、`.lock` 等;属数据/状态/字节快照,不是文档)。

**计划分母对照 (任务预期 ~245 条)**: 实际 **253** 条 = `.md` 245 + `.html` 4 + `.txt` 3 + `LICENSE` 1。静态预期 245 的构成: docs 216 `.md` + 4 `.html`(3 docs + 1 examples) + 根 11 `.md` + 3 `requirements*.txt` + `LICENSE` + 8 artifacts `model_card.md` + 2 examples `.md` + 2 scripts `.md` + 2 data/outputs README。差额来自: ① html/txt/LICENSE 是否计为文档的口径(本报告计入, 见上表); ② 盘点期间团队仍在写入新文档(`docs/infra/**`, 见 §6.1 未跟踪清单), 清单以采集时点为准。

**doc_type 判定**: 按路径与文件名规则判定(不用目录名硬猜内容),规则链依次为: `docs/archive/**`→快照; `ADR-*`→ADR; 宪法/AGENT→治理规范; `docs/contracts/**`→合同; `docs/design/**`→设计; `docs/plans/**`→阶段计划(含“台账”转为运营记录、含“报告/复盘”转为报告); `docs/reports/**`→报告; `docs/infra/**`→运营记录; `docs/demo/**`→设计(含“审核/解读”转为报告); `docs/superseded/**`→阶段计划/设计; `docs/agent_runs/**`→运营记录(含 REPORT/REVIEW 转报告、PLAN 转阶段计划); 根 README→产品文档、CONTRIBUTING/LICENSE→外部说明、requirements→其它、其余 memo/报告按名判定; examples/artifacts→产品文档; README 类索引→索引; 其余→其它。

**declared_status**: 按优先级解析文档顶部 20 行: ① 行首英文状态横幅 (`> **SUPERSEDED（...）**`/`Executed-Frozen`/`Historical`/`Active-Proposed`/`Frozen-Preregistered`/`Evidence`); ② 显式 `状态:`/`Status:` 字段(含 `> **状态**：` 与 `日期：… · 状态：…` 行内形态); ③ 行首中文状态词(`参考`/`历史`/`权威`/`现行`); ④ 表格行 `| 状态 | … |`。取到后在括号深度 0 处按 `。`/`：` 截断、去 markdown 标记, ≤120 字符。取不到一律留空,**不做推断**。

**declared_version**: 顶部 20 行内 `当前版本:`/`版本:` 等标签后的 `vX.Y[.Z]`; 无标签时回退取标题中的 `vX.Y`。**declared_date**: 顶部 20 行内 `日期:`/`生成:` 等标签或 `@ YYYY-MM-DD` 形态,统一归一为 `YYYYMMDD`。

**ref_count**: 以文档基本名为单位,在全库 `.md/.py/.json/.yaml/.yml` 文本中出现的次数; 语料 1310 个文件 / 21,599,029 字符; 排除 `.git/`、`.slim/`、`data/`、`outputs/`、缓存目录与 `docs/archive/` 归档快照。**注意**: 同名文件(如多个 `README.md`、8 份 `model_card.md`)共享同一基本名计数,故该字段对同名通用文件名只表示“该名字被引用次数”,不区分具体实例; 计数含文档自身出现。

**on_disk_flag**: `1` = 已被 git 跟踪(`git ls-files`); `0` = 磁盘存在但未跟踪/被忽略。时间戳: `mtime` 为本地时区 `YYYY-MM-DDTHH:MM:SS`; `git_*_date` 为 git 提交日期 `YYYY-MM-DD`(`git log --follow`,首次=最早提交,末次=最新提交)。

## 1. 按 zone 计数

| zone | 数量 | 占比 |
|---|---:|---:|
| `docs-plans` | 58 | 22.9% |
| `docs-archive` | 36 | 14.2% |
| `docs-reports` | 36 | 14.2% |
| `docs-design-decisions` | 34 | 13.4% |
| `docs-design` | 16 | 6.3% |
| `root` | 15 | 5.9% |
| `docs-agent-runs` | 14 | 5.5% |
| `docs-contracts` | 13 | 5.1% |
| `artifacts` | 8 | 3.2% |
| `docs-infra` | 6 | 2.4% |
| `docs-superseded` | 5 | 2.0% |
| `docs-demo` | 3 | 1.2% |
| `examples` | 3 | 1.2% |
| `data-outputs` | 2 | 0.8% |
| `docs-root` | 2 | 0.8% |
| `scripts` | 2 | 0.8% |
| **合计** | **253** | 100% |

## 2. doc_type 分布

| doc_type | 数量 | 占比 |
|---|---:|---:|
| 阶段计划 | 60 | 23.7% |
| 报告 | 43 | 17.0% |
| 快照 | 36 | 14.2% |
| ADR | 34 | 13.4% |
| 设计 | 20 | 7.9% |
| 运营记录 | 20 | 7.9% |
| 合同 | 13 | 5.1% |
| 产品文档 | 12 | 4.7% |
| 索引 | 8 | 3.2% |
| 其它 | 3 | 1.2% |
| 外部说明 | 2 | 0.8% |
| 治理规范 | 2 | 0.8% |

交叉表 (zone × doc_type, 只列非零):

| zone | 治理规范 | 阶段计划 | 设计 | ADR | 合同 | 报告 | 索引 | 运营记录 | 快照 | 产品文档 | 外部说明 | 其它 | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `docs-plans` |  | 57 |  |  |  |  |  | 1 |  |  |  |  | 58 |
| `docs-archive` |  |  |  |  |  |  |  |  | 36 |  |  |  | 36 |
| `docs-reports` |  |  |  |  |  | 36 |  |  |  |  |  |  | 36 |
| `docs-design-decisions` |  |  |  | 34 |  |  |  |  |  |  |  |  | 34 |
| `docs-design` |  |  | 16 |  |  |  |  |  |  |  |  |  | 16 |
| `root` | 1 |  |  |  |  | 4 |  | 4 |  | 1 | 2 | 3 | 15 |
| `docs-agent-runs` |  | 1 |  |  |  | 2 | 1 | 10 |  |  |  |  | 14 |
| `docs-contracts` |  |  |  |  | 13 |  |  |  |  |  |  |  | 13 |
| `artifacts` |  |  |  |  |  |  |  |  |  | 8 |  |  | 8 |
| `docs-infra` |  |  |  |  |  |  | 1 | 5 |  |  |  |  | 6 |
| `docs-superseded` |  | 2 | 2 |  |  |  | 1 |  |  |  |  |  | 5 |
| `docs-demo` |  |  | 2 |  |  | 1 |  |  |  |  |  |  | 3 |
| `examples` |  |  |  |  |  |  |  |  |  | 3 |  |  | 3 |
| `data-outputs` |  |  |  |  |  |  | 2 |  |  |  |  |  | 2 |
| `docs-root` | 1 |  |  |  |  |  | 1 |  |  |  |  |  | 2 |
| `scripts` |  |  |  |  |  |  | 2 |  |  |  |  |  | 2 |

## 3. 按 declared_status 计数

- 有状态块声明: **177** / 253 (70.0%)
- **无状态块: 76** 条 (详见 §7)

### 3.1 归一化状态类别

| 状态类别 | 数量 |
|---|---:|
| 权威/SSOT | 21 |
| Accepted/已接受 | 22 |
| Superseded/已取代 | 17 |
| Historical/历史快照 | 18 |
| Executed-Frozen | 7 |
| Frozen-Preregistered | 2 |
| Proposed/Active-Proposed | 18 |
| 冻结/Frozen | 1 |
| 已实现 | 5 |
| 执行中/待实施 | 7 |
| 设计稿 | 3 |
| 其它 | 56 |
| (空) | 76 |

### 3.2 原始声明值 (Top 30)

| 原始 declared_status | 数量 |
|---|---:|
| (空) | 76 |
| 历史提案，当前方向已由 ADR-0010/PES-N 计划取代 | 6 |
| 权威（执行投影，非独立规范） | 5 |
| 前版记录 | 4 |
| 只读历史快照 | 4 |
| 权威（唯一规范来源 / SSOT） | 4 |
| 权威（文档治理） | 4 |
| Superseded （2026-10-20） | 3 |
| 已接受（Accepted） | 3 |
| 当前增量计划（用户授权简化路线；本文完成设计，策略接入与 RL 收益待验收） | 3 |
| 权威（阶段与边界定义的唯一来源） | 3 |
| Accepted（实现批次合同；能力状态按四层纪律另记） | 2 |
| Proposed | 2 |
| SUPERSEDED （2026-10-21） | 2 |
| Superseded （2026-10-20；2026-10-21 后续） | 2 |
| 参考（已并入 R 工作流 R0–R4） | 2 |
| 只读报告（证据留存；后续以新报告或指针追加） | 2 |
| 当前小批优化方案（设计完成；工作流、回退和真实试点尚未实施） | 2 |
| 当前未来开发计划 | 2 |
| 权威整合稿（流程总入口；阶段定义与规范不在本文重复立法） | 2 |
| 权威（项目级开发与发布；能力状态按独立证据验收） | 2 |
| 阶段执行总结与问题复核 | 2 |
| ASSEMBLED — 待人工签认 | 1 |
| Accepted | 1 |
| Accepted（2026-10-21 用户裁定：V5.2 关闭，正式进入 V5.3-MP 大规模联合训练） | 1 |
| Accepted（2026-10-21；部分被 ADR-0031 取代） | 1 |
| Accepted（2026-10-27 用户指令：「完成5.4.1的代码生成和训练工作」→ 按既有方案开展 implementation + formal training 的授权；同时签认方案 §11.2 建议阈值为 … | 1 |
| Accepted（2026-10-29 用户指令：「请完成5.4.2阶段代码修改与训练工作」→ 按既有方案开展 implementation + gated formal training 的授权；同时签认 Oracle… | 1 |
| Accepted（实现交付中；能力状态按四层纪律单列，见 §7） | 1 |
| Accepted（用户下达的 v4.1 P01 修复；可执行接口、数据重建与验收测试另验收） | 1 |

(另 108 个长尾状态值, 见 doc_inventory.json `stats.by_declared_status_raw`)

## 4. 文档总量与最大 20 个文档

- 总体积: **4,972,194 B** = 4.742 MB
- 总行数: 55,452 行
- 平均: 19,653 B/篇, 219 行/篇

| # | path | size (B) | 行数 | zone |
|---:|---|---:|---:|---|
| 1 | `docs/reports/PES2TS_V53MP_训练过程可视化报告_20261023.html` | 379,567 | 1,660 | `docs-reports` |
| 2 | `docs/plans/PES2TS_文档修订台账.md` | 169,816 | 894 | `docs-plans` |
| 3 | `README.md` | 104,309 | 1,312 | `root` |
| 4 | `docs/archive/20261007_pre_orca_xtb_pilot/files/README.md` | 94,658 | 1,279 | `docs-archive` |
| 5 | `docs/archive/20261007_pre_light_policy/source_README.md` | 93,875 | 1,277 | `docs-archive` |
| 6 | `docs/plans/PES2TS_V5.4.3_双Decoder_50K_全面开发与训练方案_v1.0_20261102.md` | 93,311 | 1,670 | `docs-plans` |
| 7 | `docs/README.md` | 81,513 | 355 | `docs-root` |
| 8 | `docs/plans/PES2TS_V5.4.1_Embedding_50K_详细阶段实施方案_v1.0_20261027.md` | 73,875 | 916 | `docs-plans` |
| 9 | `docs/plans/PES2TS_EGNN_BC_IQL_SAC实施与训练计划_v3_20261015.md` | 68,272 | 548 | `docs-plans` |
| 10 | `docs/infra/PES2TS_开发文档规范化归档方案_v2_20261121.md` | 65,318 | 800 | `docs-infra` |
| 11 | `docs/plans/PES2TS_PES-N_V5-V7架构调整与实施计划_v1.2_20261018.md` | 61,440 | 928 | `docs-plans` |
| 12 | `docs/design/decisions/ADR-0034-V5.4.3-双Decoder与DecoderContextV1.md` | 60,763 | 671 | `docs-design-decisions` |
| 13 | `docs/infra/doc_audit/20261121/GOVERNANCE_CONSTRAINTS.md` | 59,197 | 535 | `docs-infra` |
| 14 | `docs/archive/20261008_pre_scale20k_rl2k/docs/plans/PES2TS_文档修订台账.md` | 55,488 | 460 | `docs-archive` |
| 15 | `docs/reports/PES2TS_V4.2-V4.7_过夜战役汇总_20261018.md` | 54,556 | 383 | `docs-reports` |
| 16 | `docs/PES2TS_开发宪法_v1.md` | 50,706 | 480 | `docs-root` |
| 17 | `docs/archive/20261007_pre_unified_residual_rl/docs__plans__PES2TS_文档修订台账.md` | 49,002 | 424 | `docs-archive` |
| 18 | `docs/archive/20261007_pre_orca_xtb_pilot/files/docs/plans/PES2TS_文档修订台账.md` | 46,354 | 403 | `docs-archive` |
| 19 | `docs/plans/PES2TS_V5.0-V5.2_十小时冲刺训练与验证执行方案_v1_20261018.md` | 45,685 | 396 | `docs-plans` |
| 20 | `docs/plans/PES2TS_训练测试集划分标准_v1_20261120.md` | 45,080 | 452 | `docs-plans` |

## 5. ref_count 最高的 20 个文档

| # | path | ref_count |
|---:|---|---:|
| 1 | `README.md` | 237 |
| 2 | `data/README.md` | 237 |
| 3 | `docs/README.md` | 237 |
| 4 | `docs/agent_runs/README.md` | 237 |
| 5 | `docs/archive/20261005_pre_pesn/README.md` | 237 |
| 6 | `docs/archive/20261006_pre_g3p/README.md` | 237 |
| 7 | `docs/archive/20261007_pre_light_policy/README.md` | 237 |
| 8 | `docs/archive/20261007_pre_light_policy/docs/README.md` | 237 |
| 9 | `docs/archive/20261007_pre_orca_xtb_pilot/README.md` | 237 |
| 10 | `docs/archive/20261007_pre_orca_xtb_pilot/files/README.md` | 237 |
| 11 | `docs/archive/20261007_pre_orca_xtb_pilot/files/docs/README.md` | 237 |
| 12 | `docs/archive/20261007_pre_predictor_scope/README.md` | 237 |
| 13 | `docs/archive/20261007_pre_unified_residual_rl/README.md` | 237 |
| 14 | `docs/archive/20261008_pre_scale20k_rl2k/README.md` | 237 |
| 15 | `docs/archive/20261008_pre_scale20k_rl2k/docs/README.md` | 237 |
| 16 | `docs/archive/README.md` | 237 |
| 17 | `docs/infra/doc_audit/20261121/README.md` | 237 |
| 18 | `docs/superseded/README.md` | 237 |
| 19 | `examples/contracts_v1/README.md` | 237 |
| 20 | `examples/m1_first_plan/README.md` | 237 |

## 6. git 跟踪与时间戳

- 已被 git 跟踪: **249** 条
- **未被 git 跟踪: 4** 条
- git 首次提交日期范围: `2026-09-25` → `2026-11-20`
- git 最新提交日期(全清单最大值): `2026-11-20`
- 取不到 git 日期的记录: 4 条

### 6.1 未被 git 跟踪的文档

- `docs/infra/PES2TS_开发文档规范化归档方案_v2_20261121.md`
- `docs/infra/doc_audit/20261121/GOVERNANCE_CONSTRAINTS.md`
- `docs/infra/doc_audit/20261121/MIGRATION_LOG.md`
- `docs/infra/doc_audit/20261121/README.md`

### 6.2 取不到 git 日期的文档

- `docs/infra/PES2TS_开发文档规范化归档方案_v2_20261121.md`
- `docs/infra/doc_audit/20261121/GOVERNANCE_CONSTRAINTS.md`
- `docs/infra/doc_audit/20261121/MIGRATION_LOG.md`
- `docs/infra/doc_audit/20261121/README.md`

## 7. 无状态块文档清单

共 **76** 条 (定义: 顶部 20 行内既无 `状态:`/`Status:` 字段, 也无 Superseded/Historical/Executed-Frozen/Active-Proposed/Frozen-Preregistered/Evidence/参考/历史/权威 行)。

### (root)/ (12 条)

- `24H_EXECUTION_TIMELINE.md`
- `24H_FINAL_REPORT.md`
- `CONTRIBUTING.md`
- `FAILURE_AND_REMAINING_WORK.md`
- `LICENSE`
- `NEXT_DAY_DECISION_MEMO.md`
- `README.md`
- `V52_PROGRESS_STOP_REPORT.md`
- `V53MP_RESUME_20261022.md`
- `requirements-pesn-runtime.txt`
- `requirements-pesn-training.txt`
- `requirements.txt`

### artifacts/ (8 条)

- `artifacts/pesn/pesn-control-demo24-bc-v0.1.0/model_card.md`
- `artifacts/pesn/pesn-control-demo24-bc-v0.1.1-rlrepair1/model_card.md`
- `artifacts/pesn/pesn-control-demo24-cql-v0.1.1-rlrepair1/model_card.md`
- `artifacts/pesn/pesn-control-demo24-v0.1.0/model_card.md`
- `artifacts/pesn/pesn-control-scale-bc-v0.1.0-rl50/model_card.md`
- `artifacts/pesn/pesn-control-scale-cql-v0.1.0-rl50/model_card.md`
- `artifacts/pesn/pesn-irc-scale5k-v0.1.0-rc.1/model_card.md`
- `artifacts/pesn/pesn-irc-v0.1.0-rc.1/model_card.md`

### data/ (1 条)

- `data/README.md`

### docs/ (49 条)

- `docs/README.md`
- `docs/agent_runs/handoffs/W-DATA_E1_sixtables.md`
- `docs/agent_runs/handoffs/W-DATA_E7_stats.md`
- `docs/agent_runs/handoffs/W-FEAT_E4_v52labels.md`
- `docs/agent_runs/handoffs/W-FEAT_E6_v53exec.md`
- `docs/agent_runs/handoffs/W-FEAT_E6_v53prep.md`
- `docs/archive/20261007_pre_light_policy/docs/README.md`
- `docs/archive/20261007_pre_light_policy/source_README.md`
- `docs/archive/20261007_pre_orca_xtb_pilot/files/README.md`
- `docs/archive/20261007_pre_orca_xtb_pilot/files/docs/README.md`
- `docs/archive/20261007_pre_unified_residual_rl/docs__README.md`
- `docs/archive/20261008_pre_scale20k_rl2k/docs/README.md`
- `docs/archive/README.md`
- `docs/contracts/PES2TS_ACP字段映射_v1.md`
- `docs/contracts/PES2TS_contracts_v1.md`
- `docs/demo/PES2TS_Demo实现审核与架构解读_20260930.md`
- `docs/demo/PES2TS_Demo评估指标_v1.md`
- `docs/demo/PES2TS_demo24_反应挑选与复核方案.md`
- `docs/design/G1_v2_ORCA扫描模式识别与计划.md`
- `docs/design/G1_v2_一维优先与xTB二维成本策略.md`
- `docs/design/G1_v2_产物成键逆向扫描优先策略.md`
- `docs/design/G1_v2_多成键与少成键方向对比.md`
- `docs/design/G1_v2_成断键扫描方向选择可视化.html`
- `docs/design/G1_v2_方向无关成键锚点与双向扫描策略.md`
- `docs/design/G1_v2_补全实施总方案.md`
- `docs/design/PES_GENERATION_图论扫描策略选择器设计_v1.md`
- `docs/plans/PES2TS_V5.1_修复冲刺与扩量准入方案_v1_20261019.md`
- `docs/plans/PES2TS_双项目开发与Demo方案.md`
- `docs/plans/PES2TS_实施路线图.html`
- `docs/plans/PES2TS_文档修订台账.md`
- `docs/reports/PES2TS_Demo24_ACP首批计算_20261001.md`
- `docs/reports/PES2TS_Demo24_TS_IRC算法诊断_20261001.md`
- `docs/reports/PES2TS_Demo24_多坐标全批计算结果_20261002.md`
- `docs/reports/PES2TS_Demo24_端点与TS几何核查_20261001.md`
- `docs/reports/PES2TS_Demo24_第二轮修改与实测_20261002.md`
- `docs/reports/PES2TS_V52_任务纠偏审计_20261021.md`
- `docs/reports/PES2TS_V53MP_S4_20k_vs_50k_配对统计与C臂导出_20261027.md`
- `docs/reports/PES2TS_V53MP_S4_50k_单路正式训练报告_20261027.md`
- `docs/reports/PES2TS_V53MP_规模训练收益报告_2k_to_50k_20261027.md`
- `docs/reports/PES2TS_V53MP_训练过程可视化报告_20261023.html`
- `docs/reports/PES2TS_V5第一阶段冲刺执行报告_20261018.md`
- `docs/reports/PES2TS_generation关键问题与根因分析_20261002.md`
- `docs/reports/PES2TS_多坐标PES逐步实施方案_20261001.md`
- `docs/reports/PES2TS_多坐标同步第一轮实现与实测_20261002.md`
- `docs/reports/PES2TS_多键同步一维扫描全面问题报告_20261001.md`
- `docs/reports/PES2TS_对话与现有任务完整总结报告_20261002.md`
- `docs/reports/PES2TS_开发阶段性报告_20260930.md`
- `docs/reports/PES_GENERATION_单端预测校正延续可行性分析_20261002.md`
- `docs/superseded/README.md`

### examples/ (3 条)

- `examples/contracts_v1/README.md`
- `examples/contracts_v1/path_viewer.html`
- `examples/m1_first_plan/README.md`

### outputs/ (1 条)

- `outputs/README.md`

### scripts/ (2 条)

- `scripts/README.md`
- `scripts/archive/README.md`

## 8. mtime 与 git_last_commit 相差 > 7 天的文档

共 **2** 条。含义: 最后一次提交之后工作区仍被本地修改过(或 checkout 时间不同), 内容可能与 git 记录不一致, 需人工确认。

| # | path | mtime | 最后提交 | 差(天) |
|---:|---|---|---|---:|
| 1 | `requirements-pesn-runtime.txt` | 2026-10-30T14:44:12 | 2026-10-18 | 12 |
| 2 | `docs/reports/PES2TS_V53MP_训练过程可视化报告_20261023.html` | 2026-10-23T19:02:19 | 2026-10-31 | 8 |

## 9. 文件名含日期但内容无版本信息的文档

共 **48** 条 (文件名含 `YYYYMMDD`/`YYYY-MM-DD`, 但顶部 20 行与标题均未声明版本号)。

- `V53MP_RESUME_20261022.md`
- `docs/demo/PES2TS_Demo实现审核与架构解读_20260930.md`
- `docs/design/PES2TS_G_V_MAPPING_v1_20261018.md`
- `docs/plans/PES2TS_ACP执行统一与旧代码清理方案_20261003.md`
- `docs/plans/PES2TS_G1_G2边界审计与补齐任务_20261003.md`
- `docs/plans/PES2TS_V5.1-R2_损失尺度与数据完整性修复方案_v1_20261020.md`
- `docs/plans/PES2TS_V5.1-R3_反应意图条件化与端点标签不变重训方案_v3_20261020.md`
- `docs/plans/PES2TS_V5.1-R3_数据库特征损失三位一体修复方案_v1_20261020.md`
- `docs/plans/PES2TS_V5.1_修复冲刺与扩量准入方案_v1_20261019.md`
- `docs/plans/PES2TS_V5.3-MP_文档同步清单_v1_20261021.md`
- `docs/plans/PES2TS_V5.4_剩余架构研究工作方案_v1.0_20261029.md`
- `docs/plans/PES2TS_单端点IRC生成控制器_产品设计_v3_20261020.md`
- `docs/plans/PES2TS_图论立体先验与单端协同延续优化方案_v2_20261002.md`
- `docs/plans/PES2TS_扫描规划谱系合并方案_20260930.md`
- `docs/plans/PES2TS_第二轮修改最终方案_仅完全成断键_20261002.md`
- `docs/reports/PES2TS_Demo24_ACP首批计算_20261001.md`
- `docs/reports/PES2TS_Demo24_TS_IRC算法诊断_20261001.md`
- `docs/reports/PES2TS_Demo24_多坐标全批计算结果_20261002.md`
- `docs/reports/PES2TS_Demo24_端点与TS几何核查_20261001.md`
- `docs/reports/PES2TS_Demo24_第二轮修改与实测_20261002.md`
- `docs/reports/PES2TS_V4.2-V4.7_过夜战役汇总_20261018.md`
- `docs/reports/PES2TS_V52R_输入充分性决策记录_20261021.md`
- `docs/reports/PES2TS_V52_任务纠偏审计_20261021.md`
- `docs/reports/PES2TS_V53MP_S0_数据建库与资格报告_20261022.md`
- `docs/reports/PES2TS_V53MP_S1_2k冒烟报告_20261022.md`
- `docs/reports/PES2TS_V53MP_S2_5k规模报告_20261022.md`
- `docs/reports/PES2TS_V53MP_S3_20k主结果报告_20261023.md`
- `docs/reports/PES2TS_V53MP_S3_闭环评估报告_20261023.md`
- `docs/reports/PES2TS_V53MP_S4_20k_vs_50k_配对统计与C臂导出_20261027.md`
- `docs/reports/PES2TS_V53MP_S4_50k_单路正式训练报告_20261027.md`
- `docs/reports/PES2TS_V53MP_规模训练收益报告_2k_to_50k_20261027.md`
- `docs/reports/PES2TS_V53MP_训练过程可视化报告_20261023.html`
- `docs/reports/PES2TS_V541_Embedding_50K_实施与训练报告_20261029.md`
- `docs/reports/PES2TS_V542_Backbone_50K_实施与训练报告_20261031.md`
- `docs/reports/PES2TS_V5第一阶段冲刺执行报告_20261018.md`
- `docs/reports/PES2TS_generation关键问题与根因分析_20261002.md`
- `docs/reports/PES2TS_多坐标PES逐步实施方案_20261001.md`
- `docs/reports/PES2TS_多坐标同步第一轮实现与实测_20261002.md`
- `docs/reports/PES2TS_多键同步一维扫描全面问题报告_20261001.md`
- `docs/reports/PES2TS_实现报告_20261018.md`
- `docs/reports/PES2TS_对话与现有任务完整总结报告_20261002.md`
- `docs/reports/PES2TS_开发阶段性报告_20260930.md`
- `docs/reports/PES_GENERATION_单端预测校正延续可行性分析_20261002.md`
- `docs/reports/PES_generation_多坐标约束学习_ORCA能力与数学方案_20261001.md`
- `docs/superseded/PES2TS_V5.1-R3_端点对称特征与损失重训方案_v1_20261020.md`
- `docs/superseded/PES2TS_V5.1-R3_端点对称特征与损失重训方案_v2_20261020.md`
- `docs/superseded/PES2TS_单端点IRC生成控制器_产品设计_v1_20261020.md`
- `docs/superseded/PES2TS_单端点IRC生成控制器_产品设计_v2_20261020.md`

## 10. declared_version 分布 (非空)

| declared_version | 数量 |
|---|---:|
| `v1` | 43 |
| `v1.0` | 19 |
| `v1.1` | 12 |
| `v0.1.0` | 6 |
| `v2` | 6 |
| `v1.2` | 4 |
| `v2.0` | 4 |
| `v2.4` | 4 |
| `v2.5` | 3 |
| `v0.1.1` | 2 |
| `v2.2` | 2 |
| `v2.3` | 2 |
| `v3.0` | 2 |
| `v3.3` | 2 |
| `v3.4` | 2 |
| `v3.5` | 2 |
| `v0.1` | 1 |
| `v0.2` | 1 |
| `v1.3` | 1 |
| `v2.6` | 1 |
| `v2.9` | 1 |
| `v3.16` | 1 |
| `v3.7` | 1 |
| `v4.0` | 1 |
| `v4.2` | 1 |
| `v4.3` | 1 |
| `v4.8` | 1 |
| (空) | 127 |

## 11. 附: 全量清单 (按 zone, path 排序)

| # | path | ext | size(B) | 行 | zone | doc_type | declared_status | ver | date | ref | tracked |
|---:|---|---|---:|---:|---|---|---|---|---|---:|:-:|
| 1 | `artifacts/pesn/pesn-control-demo24-bc-v0.1.0/model_card.md` | .md | 1,771 | 30 | `artifacts` | 产品文档 |  | v0.1.0 | - | 75 | 1 |
| 2 | `artifacts/pesn/pesn-control-demo24-bc-v0.1.1-rlrepair1/model_card.md` | .md | 1,966 | 30 | `artifacts` | 产品文档 |  | v0.1.1 | - | 75 | 1 |
| 3 | `artifacts/pesn/pesn-control-demo24-cql-v0.1.1-rlrepair1/model_card.md` | .md | 1,964 | 30 | `artifacts` | 产品文档 |  | v0.1.1 | - | 75 | 1 |
| 4 | `artifacts/pesn/pesn-control-demo24-v0.1.0/model_card.md` | .md | 1,765 | 30 | `artifacts` | 产品文档 |  | v0.1.0 | - | 75 | 1 |
| 5 | `artifacts/pesn/pesn-control-scale-bc-v0.1.0-rl50/model_card.md` | .md | 1,961 | 30 | `artifacts` | 产品文档 |  | v0.1.0 | - | 75 | 1 |
| 6 | `artifacts/pesn/pesn-control-scale-cql-v0.1.0-rl50/model_card.md` | .md | 1,959 | 30 | `artifacts` | 产品文档 |  | v0.1.0 | - | 75 | 1 |
| 7 | `artifacts/pesn/pesn-irc-scale5k-v0.1.0-rc.1/model_card.md` | .md | 2,787 | 25 | `artifacts` | 产品文档 |  | v0.1.0 | - | 75 | 1 |
| 8 | `artifacts/pesn/pesn-irc-v0.1.0-rc.1/model_card.md` | .md | 2,777 | 25 | `artifacts` | 产品文档 |  | v0.1.0 | - | 75 | 1 |
| 9 | `data/README.md` | .md | 3,255 | 68 | `data-outputs` | 索引 |  | - | - | 237 | 1 |
| 10 | `outputs/README.md` | .md | 1,401 | 32 | `data-outputs` | 索引 |  | - | - | 237 | 1 |
| 11 | `docs/agent_runs/BOOTSTRAP_REVIEW.md` | .md | 5,999 | 63 | `docs-agent-runs` | 报告 | Review（提交用户决策；未授权训练） | - | 20261020 | 9 | 1 |
| 12 | `docs/agent_runs/E0_SIGNOFF_REPORT.md` | .md | 5,883 | 72 | `docs-agent-runs` | 报告 | ASSEMBLED — 待人工签认 | - | 20261020 | 0 | 1 |
| 13 | `docs/agent_runs/PARALLEL_EXECUTION_PLAN.md` | .md | 20,117 | 182 | `docs-agent-runs` | 阶段计划 | Draft（并行 dry-run 进行中；未授权启动正式训练） | v0.1 | 20261020 | 8 | 1 |
| 14 | `docs/agent_runs/README.md` | .md | 2,882 | 41 | `docs-agent-runs` | 索引 | Active（并行 dry-run 已过；正式启动待用户授权） | v0.2 | 20261020 | 237 | 1 |
| 15 | `docs/agent_runs/handoffs/W-DATA_E1_sixtables.md` | .md | 5,351 | 40 | `docs-agent-runs` | 运营记录 |  | - | - | 1 | 1 |
| 16 | `docs/agent_runs/handoffs/W-DATA_E7_stats.md` | .md | 7,473 | 136 | `docs-agent-runs` | 运营记录 |  | - | - | 0 | 1 |
| 17 | `docs/agent_runs/handoffs/W-FEAT_E2_v3_stack.md` | .md | 12,654 | 65 | `docs-agent-runs` | 运营记录 | fixture_passed — 全部结果 not_admissible_until_gate_pass，需 G0+G1… | - | - | 2 | 1 |
| 18 | `docs/agent_runs/handoffs/W-FEAT_E4_v52labels.md` | .md | 8,840 | 73 | `docs-agent-runs` | 运营记录 |  | - | - | 0 | 1 |
| 19 | `docs/agent_runs/handoffs/W-FEAT_E6_v53exec.md` | .md | 6,957 | 73 | `docs-agent-runs` | 运营记录 |  | - | - | 0 | 1 |
| 20 | `docs/agent_runs/handoffs/W-FEAT_E6_v53prep.md` | .md | 7,744 | 144 | `docs-agent-runs` | 运营记录 |  | - | - | 0 | 1 |
| 21 | `docs/agent_runs/handoffs/W-LOSS_E3_BVL_arms.md` | .md | 13,001 | 141 | `docs-agent-runs` | 运营记录 | IMPLEMENTED + EXECUTED · 2026-10-20T19:10Z · 分支 ops/e3-cont … | - | - | 2 | 1 |
| 22 | `docs/agent_runs/handoffs/W-LOSS_E3_trainer_and_runs.md` | .md | 9,537 | 93 | `docs-agent-runs` | 运营记录 | IMPLEMENTED + EXECUTED (fixture_passed → real runs done) · 2… | - | - | 2 | 1 |
| 23 | `docs/agent_runs/handoffs/W-LOSS_E5_v52.md` | .md | 10,712 | 89 | `docs-agent-runs` | 运营记录 | IMPLEMENTED + PARTIALLY EXECUTED（预注册抢救集完成） · 2026-10-20T20:4… | - | - | 1 | 1 |
| 24 | `docs/agent_runs/handoffs/W-LOSS_WP-L3_ledger_hardening.md` | .md | 13,870 | 96 | `docs-agent-runs` | 运营记录 | IMPLEMENTED + FIXTURE_PASSED | - | 20261020 | 1 | 1 |
| 25 | `docs/archive/20261005_pre_pesn/README.md` | .md | 1,119 | 14 | `docs-archive` | 快照 | 只读历史快照 | v1 | 20261005 | 237 | 1 |
| 26 | `docs/archive/20261006_pre_g3p/README.md` | .md | 1,058 | 14 | `docs-archive` | 快照 | 只读历史快照，非当前规范或计划 | v1 | 20261006 | 237 | 1 |
| 27 | `docs/archive/20261007_pre_light_policy/AGENT.md` | .md | 13,471 | 170 | `docs-archive` | 快照 | 权威（执行投影，非独立规范） | v3.3 | 20261007 | 141 | 1 |
| 28 | `docs/archive/20261007_pre_light_policy/README.md` | .md | 826 | 10 | `docs-archive` | 快照 | 只读历史快照 | v1 | 20261007 | 237 | 1 |
| 29 | `docs/archive/20261007_pre_light_policy/docs/PES2TS_开发宪法_v1.md` | .md | 38,831 | 430 | `docs-archive` | 快照 | 权威（唯一规范来源 / SSOT） | v2.3 | 20261007 | 128 | 1 |
| 30 | `docs/archive/20261007_pre_light_policy/docs/README.md` | .md | 22,582 | 132 | `docs-archive` | 快照 |  | v1 | - | 237 | 1 |
| 31 | `docs/archive/20261007_pre_light_policy/docs/plans/PES2TS_G3_P1_IRC教师预训练与模型版本计划_v1_20261006.md` | .md | 24,847 | 234 | `docs-archive` | 快照 | 当前第一期训练实施方案（设计完成，数据构建/训练/权重/硬件验收未启动） | v1.1 | 20261007 | 28 | 1 |
| 32 | `docs/archive/20261007_pre_light_policy/docs/plans/PES2TS_PES-N未来开发计划_v2_20261005.md` | .md | 31,780 | 308 | `docs-archive` | 快照 | 当前未来开发计划 | v2.3 | 20261007 | 32 | 1 |
| 33 | `docs/archive/20261007_pre_light_policy/docs/plans/PES2TS_完整开发流方案_整合版_v1.md` | .md | 11,426 | 121 | `docs-archive` | 快照 | 权威整合稿（流程总入口；阶段定义与规范不在本文重复立法） | v2.2 | 20261007 | 17 | 1 |
| 34 | `docs/archive/20261007_pre_light_policy/docs/plans/PES2TS_整体开发与发布方案_v2.md` | .md | 14,326 | 114 | `docs-archive` | 快照 | 权威（项目级开发与发布；能力状态按独立证据验收） | v3.3 | 20261007 | 21 | 1 |
| 35 | `docs/archive/20261007_pre_light_policy/docs/plans/PES2TS_文档修订台账.md` | .md | 43,765 | 383 | `docs-archive` | 快照 | 权威（文档治理） | v2.4 | 20261007 | 56 | 1 |
| 36 | `docs/archive/20261007_pre_light_policy/docs/plans/PES2TS_阶段定义_G0-G3.md` | .md | 18,290 | 178 | `docs-archive` | 快照 | 权威（阶段与边界定义的唯一来源） | v4.2 | 20261007 | 95 | 1 |
| 37 | `docs/archive/20261007_pre_light_policy/source_README.md` | .md | 93,875 | 1,277 | `docs-archive` | 快照 |  | v1 | - | 0 | 1 |
| 38 | `docs/archive/20261007_pre_orca_xtb_pilot/README.md` | .md | 547 | 10 | `docs-archive` | 快照 | 只读历史快照 | v1 | 20261007 | 237 | 1 |
| 39 | `docs/archive/20261007_pre_orca_xtb_pilot/files/AGENT.md` | .md | 14,071 | 171 | `docs-archive` | 快照 | 权威（执行投影，非独立规范） | v3.4 | 20261007 | 141 | 1 |
| 40 | `docs/archive/20261007_pre_orca_xtb_pilot/files/README.md` | .md | 94,658 | 1,279 | `docs-archive` | 快照 |  | v1 | - | 237 | 1 |
| 41 | `docs/archive/20261007_pre_orca_xtb_pilot/files/docs/PES2TS_开发宪法_v1.md` | .md | 41,230 | 439 | `docs-archive` | 快照 | 权威（唯一规范来源 / SSOT） | v2.4 | 20261007 | 128 | 1 |
| 42 | `docs/archive/20261007_pre_orca_xtb_pilot/files/docs/README.md` | .md | 23,101 | 134 | `docs-archive` | 快照 |  | v1 | - | 237 | 1 |
| 43 | `docs/archive/20261007_pre_orca_xtb_pilot/files/docs/plans/PES2TS_文档修订台账.md` | .md | 46,354 | 403 | `docs-archive` | 快照 | 权威（文档治理） | v2.5 | 20261007 | 56 | 1 |
| 44 | `docs/archive/20261007_pre_orca_xtb_pilot/files/docs/plans/PES2TS_轻量控制与RL衔接计划_v1_20261007.md` | .md | 10,636 | 86 | `docs-archive` | 快照 | 当前增量计划（用户授权简化路线；本文完成设计，策略接入与 RL 收益待验收） | v1 | 20261007 | 18 | 1 |
| 45 | `docs/archive/20261007_pre_predictor_scope/README.md` | .md | 911 | 12 | `docs-archive` | 快照 | 只读历史快照，非当前规范 | v1 | 20261007 | 237 | 1 |
| 46 | `docs/archive/20261007_pre_unified_residual_rl/AGENT.md` | .md | 14,468 | 172 | `docs-archive` | 快照 | 权威（执行投影，非独立规范） | v3.5 | 20261007 | 141 | 1 |
| 47 | `docs/archive/20261007_pre_unified_residual_rl/README.md` | .md | 1,159 | 12 | `docs-archive` | 快照 | 只读历史快照与本次规划验证记录 | v1 | 20261007 | 237 | 1 |
| 48 | `docs/archive/20261007_pre_unified_residual_rl/docs__PES2TS_开发宪法_v1.md` | .md | 42,381 | 443 | `docs-archive` | 快照 | 权威（唯一规范来源 / SSOT） | v2.5 | 20261007 | 0 | 1 |
| 49 | `docs/archive/20261007_pre_unified_residual_rl/docs__README.md` | .md | 23,563 | 136 | `docs-archive` | 快照 |  | v1 | - | 0 | 1 |
| 50 | `docs/archive/20261007_pre_unified_residual_rl/docs__plans__PES2TS_G2_AB4_ORCA驱动xTB约束优化试点方案_20261007.md` | .md | 11,164 | 90 | `docs-archive` | 快照 | 当前小批优化方案（设计完成；工作流、回退和真实试点尚未实施） | v1 | 20261007 | 0 | 1 |
| 51 | `docs/archive/20261007_pre_unified_residual_rl/docs__plans__PES2TS_PES-N未来开发计划_v2_20261005.md` | .md | 32,558 | 311 | `docs-archive` | 快照 | 当前未来开发计划 | v2.4 | 20261007 | 0 | 1 |
| 52 | `docs/archive/20261007_pre_unified_residual_rl/docs__plans__PES2TS_文档修订台账.md` | .md | 49,002 | 424 | `docs-archive` | 快照 | 权威（文档治理） | v2.6 | 20261007 | 0 | 1 |
| 53 | `docs/archive/20261007_pre_unified_residual_rl/docs__plans__PES2TS_轻量控制与RL衔接计划_v1_20261007.md` | .md | 11,205 | 88 | `docs-archive` | 快照 | 当前增量计划（用户授权简化路线；本文完成设计，策略接入与 RL 收益待验收） | v1 | 20261007 | 0 | 1 |
| 54 | `docs/archive/20261007_pre_unified_residual_rl/docs__plans__PES2TS_阶段定义_G0-G3.md` | .md | 19,498 | 181 | `docs-archive` | 快照 | 权威（阶段与边界定义的唯一来源） | v4.3 | 20261007 | 0 | 1 |
| 55 | `docs/archive/20261008_pre_scale20k_rl2k/AGENT.md` | .md | 15,220 | 174 | `docs-archive` | 快照 | 权威（执行投影，非独立规范） | v3.7 | 20261008 | 141 | 1 |
| 56 | `docs/archive/20261008_pre_scale20k_rl2k/README.md` | .md | 637 | 12 | `docs-archive` | 快照 | 只读历史快照 | v1.0 | 20261008 | 237 | 1 |
| 57 | `docs/archive/20261008_pre_scale20k_rl2k/docs/README.md` | .md | 25,430 | 142 | `docs-archive` | 快照 |  | - | - | 237 | 1 |
| 58 | `docs/archive/20261008_pre_scale20k_rl2k/docs/plans/PES2TS_RL下一轮代码修复训练与调整方案_v1_20261008.md` | .md | 25,908 | 248 | `docs-archive` | 快照 | 规划已完成 | v1.0 | 20261008 | 8 | 1 |
| 59 | `docs/archive/20261008_pre_scale20k_rl2k/docs/plans/PES2TS_文档修订台账.md` | .md | 55,488 | 460 | `docs-archive` | 快照 | 权威（文档治理） | v3.0 | 20261008 | 56 | 1 |
| 60 | `docs/archive/README.md` | .md | 1,733 | 20 | `docs-archive` | 快照 |  | - | - | 237 | 1 |
| 61 | `docs/contracts/PES2TS_ACP字段映射_v1.md` | .md | 13,306 | 102 | `docs-contracts` | 合同 |  | - | - | 7 | 1 |
| 62 | `docs/contracts/PES2TS_FullPathNavigationContract_v1.md` | .md | 13,101 | 218 | `docs-contracts` | 合同 | SUPERSEDED（2026-10-21，ADR-0031 / MoveProcessContract v1） | - | - | 34 | 1 |
| 63 | `docs/contracts/PES2TS_G2T_战役合同字段字典_v1.md` | .md | 6,329 | 61 | `docs-contracts` | 合同 | 离线 V0 已验证（默认测试套件） | - | - | 4 | 1 |
| 64 | `docs/contracts/PES2TS_GeometryPolicyContract_v1.md` | .md | 8,156 | 200 | `docs-contracts` | 合同 | SUPERSEDED （2026-10-21） | - | 20261020 | 16 | 1 |
| 65 | `docs/contracts/PES2TS_GeometryPolicyContract_v2.md` | .md | 9,583 | 214 | `docs-contracts` | 合同 | Active-Normative（2026-10-21；No-Move/Stop 条款由 ADR-0031 取代） | - | 20261021 | 18 | 1 |
| 66 | `docs/contracts/PES2TS_MoveProcessContract_v1.md` | .md | 11,788 | 183 | `docs-contracts` | 合同 | Active-Normative（2026-10-21 起为 V5.3-MP 的唯一 Move/Process/Stop… | - | - | 45 | 1 |
| 67 | `docs/contracts/PES2TS_PESN_IRC教师数据与模型包合同_v1.md` | .md | 10,195 | 98 | `docs-contracts` | 合同 | 已登记设计草案 | v1 | 20261007 | 12 | 1 |
| 68 | `docs/contracts/PES2TS_ReactionIntent_Contract_v1.md` | .md | 11,787 | 500 | `docs-contracts` | 合同 | SUPERSEDED （2026-10-21） | - | 20261020 | 26 | 1 |
| 69 | `docs/contracts/PES2TS_ReactionIntent_Contract_v2.md` | .md | 13,660 | 515 | `docs-contracts` | 合同 | Active-Normative（2026-10-21；Stop/q 条款由 ADR-0031 取代） | - | 20261021 | 20 | 1 |
| 70 | `docs/contracts/PES2TS_V4.2-V4.7_实施合同与战役冻结_v1.md` | .md | 16,581 | 149 | `docs-contracts` | 合同 | Frozen（实现批次合同；启动正式训练前冻结，训练开始后不得修改） | v1 | 20261017 | 3 | 1 |
| 71 | `docs/contracts/PES2TS_contracts_v1.md` | .md | 8,124 | 50 | `docs-contracts` | 合同 |  | - | - | 7 | 1 |
| 72 | `docs/contracts/PES2TS_双端TS教师与相容转移合同_v2.md` | .md | 6,546 | 55 | `docs-contracts` | 合同 | HISTORICAL / LEGACY（2026-10-21） | v2.0 | 20261015 | 13 | 1 |
| 73 | `docs/contracts/PES2TS_数据规范与ACP对接优化方案.md` | .md | 16,338 | 208 | `docs-contracts` | 合同 | 设计建议，尚未实施 | - | 20260929 | 7 | 1 |
| 74 | `docs/demo/PES2TS_Demo实现审核与架构解读_20260930.md` | .md | 31,251 | 280 | `docs-demo` | 报告 |  | - | 20260930 | 5 | 1 |
| 75 | `docs/demo/PES2TS_Demo评估指标_v1.md` | .md | 5,586 | 62 | `docs-demo` | 设计 |  | - | - | 5 | 1 |
| 76 | `docs/demo/PES2TS_demo24_反应挑选与复核方案.md` | .md | 6,312 | 61 | `docs-demo` | 设计 |  | - | - | 10 | 1 |
| 77 | `docs/design/G1_TS_IRC_映射与反应归簇实现方案.md` | .md | 24,263 | 386 | `docs-design` | 设计 | 已实现（Batches A–E 全部落地并通过 355 项测试；Batch F 全量运行经 | - | - | 4 | 1 |
| 78 | `docs/design/G1_v2_L0到G2扫描策略设计.md` | .md | 12,277 | 107 | `docs-design` | 设计 | 设计稿，2026-09-27 | - | - | 5 | 1 |
| 79 | `docs/design/G1_v2_ORCA扫描模式识别与计划.md` | .md | 15,787 | 123 | `docs-design` | 设计 |  | - | - | 8 | 1 |
| 80 | `docs/design/G1_v2_一维优先与xTB二维成本策略.md` | .md | 14,710 | 114 | `docs-design` | 设计 |  | - | - | 6 | 1 |
| 81 | `docs/design/G1_v2_产物成键逆向扫描优先策略.md` | .md | 9,952 | 105 | `docs-design` | 设计 |  | - | - | 5 | 1 |
| 82 | `docs/design/G1_v2_多成键与少成键方向对比.md` | .md | 7,002 | 72 | `docs-design` | 设计 |  | - | - | 11 | 1 |
| 83 | `docs/design/G1_v2_成断键扫描方向选择可视化.html` | .html | 30,773 | 310 | `docs-design` | 设计 |  | - | - | 6 | 1 |
| 84 | `docs/design/G1_v2_方向无关成键锚点与双向扫描策略.md` | .md | 9,303 | 107 | `docs-design` | 设计 |  | - | - | 8 | 1 |
| 85 | `docs/design/G1_v2_补全实施总方案.md` | .md | 16,085 | 163 | `docs-design` | 设计 |  | - | - | 11 | 1 |
| 86 | `docs/design/PES2TS_CODE_MAP_v2_20261018.md` | .md | 11,248 | 180 | `docs-design` | 设计 | 参考（非规范） | v2.2 | 20261018 | 6 | 1 |
| 87 | `docs/design/PES2TS_Classic图论驱动与xTB原生约束扫描设计_v1_20261006.md` | .md | 25,575 | 219 | `docs-design` | 设计 | 设计完成 | v1 | 20261006 | 10 | 1 |
| 88 | `docs/design/PES2TS_G_V_MAPPING_v1_20261018.md` | .md | 10,739 | 103 | `docs-design` | 设计 | 权威（机器可读源：config/registry/g_v_map.yaml） | - | - | 27 | 1 |
| 89 | `docs/design/PES2TS_PES-C到PES-G架构与迁移方案_v1_20261004.md` | .md | 34,094 | 437 | `docs-design` | 设计 | 设计提案（已汇总本轮用户选择；尚未实施、未改变生产默认或能力状态） | v1 | 20261004 | 8 | 1 |
| 90 | `docs/design/PES2TS_三模块与可视化产品架构_v1_20261005.md` | .md | 17,318 | 160 | `docs-design` | 设计 | 当前产品架构设计 | v1.1 | 20261007 | 12 | 1 |
| 91 | `docs/design/PES2TS_预测器产品与G3代码模块划分_v1_20261007.md` | .md | 18,529 | 182 | `docs-design` | 设计 | 当前代码范围与实施设计 | v1 | 20261007 | 18 | 1 |
| 92 | `docs/design/PES_GENERATION_图论扫描策略选择器设计_v1.md` | .md | 44,399 | 500 | `docs-design` | 设计 |  | - | - | 32 | 1 |
| 93 | `docs/design/decisions/ADR-0001-统一执行后端接缝.md` | .md | 7,170 | 122 | `docs-design-decisions` | ADR | 已接受（Accepted） | - | 20261002 | 5 | 1 |
| 94 | `docs/design/decisions/ADR-0002-计算后端统一经ACP执行.md` | .md | 13,817 | 153 | `docs-design-decisions` | ADR | 已接受（Accepted） | - | 20261003 | 19 | 1 |
| 95 | `docs/design/decisions/ADR-0003-G2-AB1接口登记.md` | .md | 5,117 | 38 | `docs-design-decisions` | ADR | 已接受（Accepted） | - | 20261004 | 7 | 1 |
| 96 | `docs/design/decisions/ADR-0004-G2-AB2接口登记.md` | .md | 7,421 | 41 | `docs-design-decisions` | ADR | 已接受（Accepted，离线 V0 已验证；gated 真实引擎项见状态列） | - | 20261004 | 7 | 1 |
| 97 | `docs/design/decisions/ADR-0005-G2-AB2执行前判定器与恢复修补.md` | .md | 8,876 | 64 | `docs-design-decisions` | ADR | 已接受（Accepted，2026-10-04；判定器 v2 与三项恢复/适配修补均在 S2 真实执行之前登记并锁定） | - | 20261004 | 7 | 1 |
| 98 | `docs/design/decisions/ADR-0006-ORCA原生约束优化与控制策略分离.md` | .md | 4,201 | 53 | `docs-design-decisions` | ADR | Proposed（用户已提出并支持设计方向；本记录不宣布生产切换或能力验收） | v1 | 20261004 | 9 | 1 |
| 99 | `docs/design/decisions/ADR-0007-统一价值模型与搜索内生选点.md` | .md | 5,093 | 53 | `docs-design-decisions` | ADR | 历史提案，当前方向已由 ADR-0010/PES-N 计划取代 | v1 | 20261004 | 10 | 1 |
| 100 | `docs/design/decisions/ADR-0008-拓扑约束下的距离与位移联合预测.md` | .md | 4,632 | 43 | `docs-design-decisions` | ADR | 历史提案，当前方向已由 ADR-0010/PES-N 计划取代 | v1 | 20261004 | 8 | 1 |
| 101 | `docs/design/decisions/ADR-0009-G阶段下的统一模型开发框架.md` | .md | 4,505 | 43 | `docs-design-decisions` | ADR | 历史提案，当前方向已由 ADR-0010/PES-N 计划取代 | v1 | 20261005 | 7 | 1 |
| 102 | `docs/design/decisions/ADR-0010-PES-N反应状态价值与STOP-MOVE统一决策.md` | .md | 4,332 | 40 | `docs-design-decisions` | ADR | Accepted（用户明确要求按最新开发哲学修改未来计划；开发方向及文档同步生效，代码/模型/生产切换另按证据验收） | v1 | 20261005 | 30 | 1 |
| 103 | `docs/design/decisions/ADR-0011-三模块产品架构与统一可视化.md` | .md | 3,721 | 36 | `docs-design-decisions` | ADR | Accepted（用户要求的产品架构与开发方向；统一代码模块、接口schema、UI和真实能力另验收） | v1 | 20261005 | 15 | 1 |
| 104 | `docs/design/decisions/ADR-0012-Classic图论驱动与xTB原生约束扫描.md` | .md | 4,723 | 39 | `docs-design-decisions` | ADR | Accepted（用户指定的设计方向；代码、正式schema、真实能力及默认切换待验收） | v1 | 20261006 | 14 | 1 |
| 105 | `docs/design/decisions/ADR-0013-G3并行IRC预训练与模型版本体系.md` | .md | 6,382 | 49 | `docs-design-decisions` | ADR | Accepted（用户授权调整阶段规划；设计生效，训练实现与权重未交付） | v1 | 20261006 | 16 | 1 |
| 106 | `docs/design/decisions/ADR-0014-预测器产品归属与G3代码范围.md` | .md | 5,648 | 42 | `docs-design-decisions` | ADR | Accepted（用户授权按产品细分开发范围；设计生效，代码/包/权限未实施） | v1 | 20261007 | 16 | 1 |
| 107 | `docs/design/decisions/ADR-0015-G2稳定底座与轻量RL研究入口.md` | .md | 3,606 | 31 | `docs-design-decisions` | ADR | Accepted（用户授权简化近期路线；规划生效，策略/RL 能力待实现验收） | v1 | 20261007 | 14 | 1 |
| 108 | `docs/design/decisions/ADR-0016-ORCA驱动xTB硬约束优化试点.md` | .md | 3,143 | 29 | `docs-design-decisions` | ADR | Accepted（用户授权尝试并要求优化方案；设计生效，实现/真实能力待验收） | v1 | 20261007 | 10 | 1 |
| 109 | `docs/design/decisions/ADR-0017-统一残差决策与G2有限回退.md` | .md | 4,815 | 35 | `docs-design-decisions` | ADR | Accepted（用户授权规划；实现与真实能力待验收） | v1 | 20261007 | 17 | 1 |
| 110 | `docs/design/decisions/ADR-0018-RL纠错与动作支持研究计划.md` | .md | 3,247 | 35 | `docs-design-decisions` | ADR | Proposed（用户要求的下一轮方案已登记；接口/算法实施与能力验收待完成） | v1.0 | 20261008 | 9 | 1 |
| 111 | `docs/design/decisions/ADR-0019-IRC20k与RL2000分阶段扩量.md` | .md | 3,391 | 35 | `docs-design-decisions` | ADR | Proposed（详细规划已完成；接口实施/训练/真实计算未启动） | v1.0 | 20261008 | 8 | 1 |
| 112 | `docs/design/decisions/ADR-0020-EGNN直接位移与BC-IQL-SAC研究主线.md` | .md | 5,927 | 58 | `docs-design-decisions` | ADR | 前版记录 | v1.3 | 20261015 | 8 | 1 |
| 113 | `docs/design/decisions/ADR-0021-无序双端拓扑与完整双侧TS教师主线.md` | .md | 3,261 | 33 | `docs-design-decisions` | ADR | Superseded for V5.2+（2026-10-21） | v1.0 | 20261015 | 11 | 1 |
| 114 | `docs/design/decisions/ADR-0022-v4.1因果输入合同与特征v3.md` | .md | 5,766 | 31 | `docs-design-decisions` | ADR | Accepted（用户下达的 v4.1 P01 修复；可执行接口、数据重建与验收测试另验收） | v1.0 | 20261017 | 3 | 1 |
| 115 | `docs/design/decisions/ADR-0023-V4.2原生重连数据与内部开发划分.md` | .md | 2,673 | 24 | `docs-design-decisions` | ADR | Accepted（实现批次合同；能力状态按四层纪律另记） | - | - | 4 | 1 |
| 116 | `docs/design/decisions/ADR-0024-V4.2无界位移模型与战役训练协议.md` | .md | 3,049 | 26 | `docs-design-decisions` | ADR | Accepted（实现批次合同；能力状态按四层纪律另记） | - | - | 2 | 1 |
| 117 | `docs/design/decisions/ADR-0025-G-V双轴映射与战役注册.md` | .md | 5,780 | 115 | `docs-design-decisions` | ADR | Accepted | v1.0 | 20261018 | 6 | 1 |
| 118 | `docs/design/decisions/ADR-0026-V5冲刺key-atom位移合同与OOF语义.md` | .md | 9,978 | 102 | `docs-design-decisions` | ADR | Accepted（实现交付中；能力状态按四层纪律单列，见 §7） | v1 | - | 11 | 1 |
| 119 | `docs/design/decisions/ADR-0027-V5.1参考架fallback与双分母评估.md` | .md | 2,789 | 48 | `docs-design-decisions` | ADR | Proposed（研究门；不实现新 fallback） | - | 20261020 | 8 | 1 |
| 120 | `docs/design/decisions/ADR-0028-关键原子几何策略合同与V5-V7任务拆分.md` | .md | 5,087 | 82 | `docs-design-decisions` | ADR | Proposed（设计冻结；实现前需签认） | - | 20261020 | 15 | 1 |
| 121 | `docs/design/decisions/ADR-0029-V5.2+双门评估与封存边界.md` | .md | 5,719 | 109 | `docs-design-decisions` | ADR | Proposed（评估设计冻结；阈值待 V5.2 数据校准后预注册） | - | 20261020 | 10 | 1 |
| 122 | `docs/design/decisions/ADR-0030-V5.2全路径导航与端点停止.md` | .md | 9,918 | 109 | `docs-design-decisions` | ADR | Accepted（2026-10-21；部分被 ADR-0031 取代） | - | 20261021 | 31 | 1 |
| 123 | `docs/design/decisions/ADR-0031-MOVE-PROCESS统一学习合同.md` | .md | 9,326 | 144 | `docs-design-decisions` | ADR | Accepted（2026-10-21 用户裁定：V5.2 关闭，正式进入 V5.3-MP 大规模联合训练） | - | 20261021 | 28 | 1 |
| 124 | `docs/design/decisions/ADR-0032-V5.4.1-Embedding接口扩展与预注册治理.md` | .md | 14,486 | 122 | `docs-design-decisions` | ADR | Accepted（2026-10-27 用户指令：「完成5.4.1的代码生成和训练工作」→ 按既有方案开展 implem… | - | 20261027 | 11 | 1 |
| 125 | `docs/design/decisions/ADR-0033-V5.4.2-Backbone接口与预注册治理.md` | .md | 44,939 | 145 | `docs-design-decisions` | ADR | Accepted（2026-10-29 用户指令：「请完成5.4.2阶段代码修改与训练工作」→ 按既有方案开展 impl… | - | 20261029 | 9 | 1 |
| 126 | `docs/design/decisions/ADR-0034-V5.4.3-双Decoder与DecoderContextV1.md` | .md | 60,763 | 671 | `docs-design-decisions` | ADR | Proposed（D1–D8 已于 2026-11-02 由仓库所有者经 DSH 编排会话明确签认；本 ADR 只完成 … | - | 20261102 | 14 | 1 |
| 127 | `docs/infra/PES2TS_仓库归档与清理方案_v1_20261120.md` | .md | 24,927 | 362 | `docs-infra` | 运营记录 | 操作性方案（Operational，非规范） | v1 | 20261120 | 7 | 1 |
| 128 | `docs/infra/PES2TS_开发文档规范化归档方案_v2_20261121.md` | .md | 65,318 | 800 | `docs-infra` | 运营记录 | Active-Proposed（规范性方案；经仓库所有者签认后转 Canonical 并纳入文档门禁 G5） | v2.0 | 20261121 | 6 | 0 |
| 129 | `docs/infra/doc_audit/20261121/GOVERNANCE_CONSTRAINTS.md` | .md | 59,197 | 535 | `docs-infra` | 运营记录 | 审计基线（只读取证，不构成新规范；规范冲突一律以宪法为准） | - | - | 14 | 0 |
| 130 | `docs/infra/doc_audit/20261121/MIGRATION_LOG.md` | .md | 1,630 | 30 | `docs-infra` | 运营记录 | Evidence-ReadOnly（执行流水模板；本文件只允许追加，不允许改写已发生的记录） | v1 | 20261121 | 4 | 0 |
| 131 | `docs/infra/doc_audit/20261121/README.md` | .md | 7,184 | 84 | `docs-infra` | 索引 | Evidence-ReadOnly（只读证据；结论与规则以 开发文档规范化归档方案 v2.0 为准，本目录只提供可复跑的… | v1 | 20261121 | 237 | 0 |
| 132 | `docs/infra/wsl_memory_governance.md` | .md | 14,567 | 203 | `docs-infra` | 运营记录 | 随 S4 infra 工具链落地（scripts/infra/） | - | - | 1 | 1 |
| 133 | `docs/plans/PES2TS_30h_R3_V52_V53_分层训练与验收计划_v1.1_20261020.md` | .md | 33,941 | 406 | `docs-plans` | 阶段计划 | EXECUTED-FROZEN / Stop 语义已取代（2026-10-21） | v1.1 | 20261020 | 22 | 1 |
| 134 | `docs/plans/PES2TS_ACP执行统一与旧代码清理方案_20261003.md` | .md | 20,699 | 351 | `docs-plans` | 阶段计划 | 执行方案（待实施与验收；ACP 唯一后端方向已确定） | - | 20261003 | 14 | 1 |
| 135 | `docs/plans/PES2TS_Demo24_真实RL闭环与对照实施方案_v1_20261007.md` | .md | 18,036 | 162 | `docs-plans` | 阶段计划 | 已实施（2026-10-07 round0 完成一轮真实闭环） | v1.0 | 20261007 | 6 | 1 |
| 136 | `docs/plans/PES2TS_EGNN_BC_IQL_SAC下一步完整训练方案_v2_20261015.md` | .md | 22,299 | 233 | `docs-plans` | 阶段计划 | 前版记录 | v2.0 | 20261015 | 3 | 1 |
| 137 | `docs/plans/PES2TS_EGNN_BC_IQL_SAC实施与训练计划_v3_20261015.md` | .md | 68,272 | 548 | `docs-plans` | 阶段计划 | 前版记录 | v3.4 | 20261015 | 8 | 1 |
| 138 | `docs/plans/PES2TS_G0-G4整体开发框架修改方案_v1_20261005.md` | .md | 33,336 | 327 | `docs-plans` | 阶段计划 | 历史提案，当前方向已由 ADR-0010/PES-N 计划取代 | v1 | 20261005 | 9 | 1 |
| 139 | `docs/plans/PES2TS_G1_G2边界审计与补齐任务_20261003.md` | .md | 11,269 | 130 | `docs-plans` | 阶段计划 | 只读抽样审计 + 待实施合同设计 | - | 20261003 | 11 | 1 |
| 140 | `docs/plans/PES2TS_G2T真实战役执行方案_20261004.md` | .md | 10,120 | 114 | `docs-plans` | 阶段计划 | 执行方案（Wave 0 四项执行前修补已实施并通过离线回归门 2026-10-04，证据 outputs/g2t_cam… | v1 | 20261004 | 6 | 1 |
| 141 | `docs/plans/PES2TS_G2_AB3功能整合与原生扫描改造方案_20261006.md` | .md | 20,246 | 158 | `docs-plans` | 阶段计划 | 方案设计完成 | v1.1 | 20261007 | 13 | 1 |
| 142 | `docs/plans/PES2TS_G2_AB4_ORCA驱动xTB约束优化试点方案_20261007.md` | .md | 12,292 | 94 | `docs-plans` | 阶段计划 | 当前小批优化方案（设计完成；工作流、回退和真实试点尚未实施） | v1.1 | 20261007 | 10 | 1 |
| 143 | `docs/plans/PES2TS_G2_ACP能力补齐与联合验收计划_20261003.md` | .md | 23,299 | 179 | `docs-plans` | 阶段计划 | 配套开发设计，待实现与真实冒烟 | v2 | 20261005 | 14 | 1 |
| 144 | `docs/plans/PES2TS_G2_PESG收紧算法迭代方案_20261003.md` | .md | 34,323 | 405 | `docs-plans` | 阶段计划 | 修订设计 v1 | v2 | 20261003 | 12 | 1 |
| 145 | `docs/plans/PES2TS_G2_下一批实现方案_真引擎接通与验收_20261004.md` | .md | 26,956 | 294 | `docs-plans` | 阶段计划 | 批次方案（S1 离线实施完成 2026-10-04，V0 门通过——S1 基建/WP-2-A 校准网格/WP-3 假 A… | v2 | 20261004 | 9 | 1 |
| 146 | `docs/plans/PES2TS_G2_下一批算法调整与测试方案_20261003.md` | .md | 28,960 | 302 | `docs-plans` | 阶段计划 | 批次方案（设计层；未实施；未启动 QC；不改变任何能力完成层级） | v2 | 20261004 | 10 | 1 |
| 147 | `docs/plans/PES2TS_G2后续开发方案_Classic到Rank到Gen_20261003.md` | .md | 29,854 | 268 | `docs-plans` | 阶段计划 | 当前开发方案 | v2 | 20261003 | 12 | 1 |
| 148 | `docs/plans/PES2TS_G3_P1_IRC教师预训练与模型版本计划_v1_20261006.md` | .md | 25,516 | 238 | `docs-plans` | 阶段计划 | 当前第一期训练实施方案（用户确认实施进行中；数据/权重/硬件和各能力以执行证据验收） | v1.2 | 20261007 | 28 | 1 |
| 149 | `docs/plans/PES2TS_IRC20k与RL2000下一阶段详细方案_v1_20261008.md` | .md | 30,493 | 285 | `docs-plans` | 阶段计划 | 规划完成 | v1 | 20261008 | 10 | 1 |
| 150 | `docs/plans/PES2TS_IRC_BC_IQL_SAC直接位移研究方案_v1_20261015.md` | .md | 16,397 | 160 | `docs-plans` | 阶段计划 | 前版记录 | v1.0 | 20261015 | 2 | 1 |
| 151 | `docs/plans/PES2TS_PES-N_V5-V7架构调整与实施计划_v1.2_20261018.md` | .md | 61,440 | 928 | `docs-plans` | 阶段计划 | SUPERSEDED （2026-10-20） | v1.2 | 20261018 | 22 | 1 |
| 152 | `docs/plans/PES2TS_PES-N未来开发计划_v2_20261005.md` | .md | 34,328 | 321 | `docs-plans` | 阶段计划 | Superseded / Historical（作为“当前未来开发计划”的地位已由 V5–V7 v1.2 取代；PES-… | v2.5 | 20261007 | 32 | 1 |
| 153 | `docs/plans/PES2TS_RL下一轮代码修复训练与调整方案_v1_20261008.md` | .md | 26,889 | 254 | `docs-plans` | 阶段计划 | 第一批部分实施并纠错重训 | v1.1 | 20261008 | 8 | 1 |
| 154 | `docs/plans/PES2TS_RL自适应路径生长与IRC验证研究方案_v1_20261009.md` | .md | 18,316 | 193 | `docs-plans` | 阶段计划 | Proposed | v1.0 | 20261009 | 2 | 1 |
| 155 | `docs/plans/PES2TS_V4.2-V4.7_十小时训练实验方案_20261017.md` | .md | 37,004 | 310 | `docs-plans` | 阶段计划 | Executed-Frozen（2026-10-18 已执行 23 个正式 run；V4.7 FINAL 三项种子四项门… | v1.1 | 20261017 | 8 | 1 |
| 156 | `docs/plans/PES2TS_V5-V7_三阶段研究与架构修订方案_v2.0_20261020.md` | .md | 27,355 | 528 | `docs-plans` | 阶段计划 | Superseded （2026-10-20） | v2.0 | 20261020 | 18 | 1 |
| 157 | `docs/plans/PES2TS_V5-V7_反应意图条件化与低维生长方案_v3.0_20261020.md` | .md | 12,621 | 465 | `docs-plans` | 阶段计划 | Proposed v3.2（取代 v2.0；不启动训练/QC） | v3.0 | 20261020 | 19 | 1 |
| 158 | `docs/plans/PES2TS_V5.0-V5.2_十小时冲刺训练与验证执行方案_v1_20261018.md` | .md | 45,685 | 396 | `docs-plans` | 阶段计划 | Proposed | v1.1 | - | 18 | 1 |
| 159 | `docs/plans/PES2TS_V5.1-R2_损失尺度与数据完整性修复方案_v1_20261020.md` | .md | 23,727 | 492 | `docs-plans` | 阶段计划 | Proposed v1（可执行研究方案；先于任何 R2 训练冻结；不启动正式 OOF/封存/扩量） | - | 20261020 | 16 | 1 |
| 160 | `docs/plans/PES2TS_V5.1-R3_反应意图条件化与端点标签不变重训方案_v3_20261020.md` | .md | 13,058 | 457 | `docs-plans` | 阶段计划 | Proposed v3.2（取代 v1/v2/v3.1；先于任何 R3 训练冻结） | - | 20261020 | 19 | 1 |
| 161 | `docs/plans/PES2TS_V5.1-R3_数据库特征损失三位一体修复方案_v1_20261020.md` | .md | 15,996 | 773 | `docs-plans` | 阶段计划 | Proposed v1.1（V5.1 阶段执行方案；先于任何 R3 训练冻结） | - | 20261020 | 6 | 1 |
| 162 | `docs/plans/PES2TS_V5.1_修复冲刺与扩量准入方案_v1_20261019.md` | .md | 16,974 | 243 | `docs-plans` | 阶段计划 |  | - | 20261019 | 6 | 1 |
| 163 | `docs/plans/PES2TS_V5.3-MP_大规模MOVE-PROCESS联合训练计划_v1_20261021.md` | .md | 19,878 | 323 | `docs-plans` | 阶段计划 | Proposed v1.1（2026-10-21 用户评审后修订；v1 为有条件通过） | v1.1 | 20261021 | 28 | 1 |
| 164 | `docs/plans/PES2TS_V5.3-MP_文档同步清单_v1_20261021.md` | .md | 9,427 | 129 | `docs-plans` | 阶段计划 | Active（执行清单；每项完成后勾选） | - | 20261021 | 4 | 1 |
| 165 | `docs/plans/PES2TS_V5.4.1_Embedding_50K_详细阶段实施方案_v1.0_20261027.md` | .md | 73,875 | 916 | `docs-plans` | 阶段计划 | Executed-Frozen（2026-10-29；WP0–WP7 已执行完毕，E*=E0 留任；执行结果与证据见 §… | v1.2 | 20261029 | 13 | 1 |
| 166 | `docs/plans/PES2TS_V5.4.2_Backbone_50K_详细阶段实施方案_v1.0_20261029.md` | .md | 38,669 | 263 | `docs-plans` | 阶段计划 | Active-Proposed（2026-10-29；WP0 治理/预注册文档已落地，WP1–WP7 待实现与执行；本轮… | v1.2 | 20261030 | 10 | 1 |
| 167 | `docs/plans/PES2TS_V5.4.3_WP执行清单_v1.0_20261102.md` | .md | 7,768 | 229 | `docs-plans` | 阶段计划 | Frozen-Preregistered v1.0（2026-11-02；D1–D8 已签认，配套主方案 PES2TS_… | v1.0 | 20261102 | 13 | 1 |
| 168 | `docs/plans/PES2TS_V5.4.3_双Decoder_50K_全面开发与训练方案_v1.0_20261102.md` | .md | 93,311 | 1,670 | `docs-plans` | 阶段计划 | Frozen-Preregistered v1.0（2026-11-02；规划、预注册草案与执行设计，不含代码实现、训练… | v1.0 | 20261102 | 17 | 1 |
| 169 | `docs/plans/PES2TS_V5.4_剩余架构研究工作方案_v1.0_20261029.md` | .md | 12,803 | 314 | `docs-plans` | 阶段计划 | Active-Proposed | - | - | 0 | 1 |
| 170 | `docs/plans/PES2TS_V52R_全路径导航与端点停止修复方案_v1_20261021.md` | .md | 15,974 | 175 | `docs-plans` | 阶段计划 | Closed（D2-G STOP；T1 不授权；由 V5.3-MP / ADR-0031 接续） | v1.1 | 20261021 | 7 | 1 |
| 171 | `docs/plans/PES2TS_V52R_输入充分性与化学终点识别优化方案_v1_20261021.md` | .md | 17,636 | 314 | `docs-plans` | 阶段计划 | Executed / Closed（2026-10-21）——D0.1 PASS（22 异常全 T→Mask） | v1.0 | 20261021 | 7 | 1 |
| 172 | `docs/plans/PES2TS_单端点IRC生成控制器_产品设计_v3_20261020.md` | .md | 6,244 | 248 | `docs-plans` | 阶段计划 | Proposed v3（取代 v1/v2；不启动训练/不改代码） | - | 20261020 | 9 | 1 |
| 173 | `docs/plans/PES2TS_双端拓扑条件训练与项目进度统一方案_v4_20261015.md` | .md | 30,912 | 595 | `docs-plans` | 阶段计划 | Executed-Frozen / Historical（V4.2–V4.7 已执行并判定负结果；本文作为 V4 排期与… | v4.0 | 20261015 | 15 | 1 |
| 174 | `docs/plans/PES2TS_双项目开发与Demo方案.md` | .md | 17,382 | 138 | `docs-plans` | 阶段计划 |  | - | - | 7 | 1 |
| 175 | `docs/plans/PES2TS_图论立体先验与单端协同延续优化方案_v2_20261002.md` | .md | 24,659 | 302 | `docs-plans` | 阶段计划 | 参考（已并入 R 工作流 R0–R4） | - | 20261002 | 4 | 1 |
| 176 | `docs/plans/PES2TS_完整开发流方案_整合版_v1.md` | .md | 13,239 | 133 | `docs-plans` | 阶段计划 | 权威整合稿（流程总入口；阶段定义与规范不在本文重复立法） | v2.4 | 20261015 | 17 | 1 |
| 177 | `docs/plans/PES2TS_实施路线图.html` | .html | 26,753 | 261 | `docs-plans` | 阶段计划 |  | - | - | 3 | 1 |
| 178 | `docs/plans/PES2TS_扫描规划谱系合并方案_20260930.md` | .md | 9,145 | 86 | `docs-plans` | 阶段计划 | 参考（ADR 决策记录，已并入 G2.4 / X 工作流） | - | - | 6 | 1 |
| 179 | `docs/plans/PES2TS_拓扑约束TS接近度与位移联合学习方案_v2_20261004.md` | .md | 18,616 | 191 | `docs-plans` | 阶段计划 | 历史提案，当前方向已由 ADR-0010/PES-N 计划取代 | v2 | 20261004 | 8 | 1 |
| 180 | `docs/plans/PES2TS_整体开发与发布方案_v2.md` | .md | 16,267 | 125 | `docs-plans` | 阶段计划 | 权威（项目级开发与发布；能力状态按独立证据验收） | v3.5 | 20261015 | 21 | 1 |
| 181 | `docs/plans/PES2TS_文档修订台账.md` | .md | 169,816 | 894 | `docs-plans` | 运营记录 |  | - | - | 56 | 1 |
| 182 | `docs/plans/PES2TS_第二轮修改最终方案_仅完全成断键_20261002.md` | .md | 15,387 | 207 | `docs-plans` | 阶段计划 | 参考（已并入 R 工作流 R0–R4） | - | 20261002 | 6 | 1 |
| 183 | `docs/plans/PES2TS_统一价值模型与搜索排序一体化发展方案_v1_20261004.md` | .md | 28,627 | 350 | `docs-plans` | 阶段计划 | 历史提案，当前方向已由 ADR-0010/PES-N 计划取代 | v1 | 20261004 | 8 | 1 |
| 184 | `docs/plans/PES2TS_统一开发顺序与阶段验收.md` | .md | 22,063 | 139 | `docs-plans` | 阶段计划 | 历史（已并入主干 G0–G4） | - | 20260929 | 8 | 1 |
| 185 | `docs/plans/PES2TS_统一残差决策与G2有限回退实施方案_v1_20261007.md` | .md | 23,416 | 178 | `docs-plans` | 阶段计划 | 当前实施方案 | v1 | 20261007 | 17 | 1 |
| 186 | `docs/plans/PES2TS_训练测试集划分标准_v1_20261120.md` | .md | 45,080 | 452 | `docs-plans` | 阶段计划 | 说明性汇总（Descriptive，非规范） | v1 | 20261120 | 4 | 1 |
| 187 | `docs/plans/PES2TS_轻量控制与RL衔接计划_v1_20261007.md` | .md | 12,032 | 90 | `docs-plans` | 阶段计划 | 当前增量计划（用户授权简化路线；本文完成设计，策略接入与 RL 收益待验收） | v1 | 20261007 | 18 | 1 |
| 188 | `docs/plans/PES2TS_逐步实施与验证方案.md` | .md | 12,724 | 119 | `docs-plans` | 阶段计划 | 历史资料 | v1.0 | - | 5 | 1 |
| 189 | `docs/plans/PES2TS_阶段定义_G0-G3.md` | .md | 22,490 | 195 | `docs-plans` | 阶段计划 | 权威（阶段与边界定义的唯一来源） | v4.8 | 20261021 | 95 | 1 |
| 190 | `docs/plans/V541_WP7_frozen_dev_eval_设计方案_v1.0_20261028.md` | .md | 41,456 | 714 | `docs-plans` | 阶段计划 | 设计稿（本步只产出设计文档；实现由 fixer 按本文执行，验证负责人为主代理） | v1.0 | - | 3 | 1 |
| 191 | `docs/reports/PES2TS_Demo24_ACP首批计算_20261001.md` | .md | 7,123 | 76 | `docs-reports` | 报告 |  | - | - | 2 | 1 |
| 192 | `docs/reports/PES2TS_Demo24_TS_IRC算法诊断_20261001.md` | .md | 13,006 | 120 | `docs-reports` | 报告 |  | - | 20261001 | 1 | 1 |
| 193 | `docs/reports/PES2TS_Demo24_多坐标全批计算结果_20261002.md` | .md | 5,722 | 52 | `docs-reports` | 报告 |  | - | - | 3 | 1 |
| 194 | `docs/reports/PES2TS_Demo24_端点与TS几何核查_20261001.md` | .md | 4,496 | 56 | `docs-reports` | 报告 |  | - | - | 3 | 1 |
| 195 | `docs/reports/PES2TS_Demo24_第二轮修改与实测_20261002.md` | .md | 8,747 | 97 | `docs-reports` | 报告 |  | - | - | 2 | 1 |
| 196 | `docs/reports/PES2TS_Direct_fit50训练表征分析_20261015.md` | .md | 6,450 | 90 | `docs-reports` | 报告 | 只读报告（证据留存；后续以新报告或指针追加） | v1.0 | 20261015 | 6 | 1 |
| 197 | `docs/reports/PES2TS_Direct批次1-6实施记录_20261015.md` | .md | 10,149 | 111 | `docs-reports` | 报告 | 只读报告（证据留存；后续以新报告或指针追加） | v1.1 | 20261015 | 4 | 1 |
| 198 | `docs/reports/PES2TS_GPU环境与磁盘预算执行记录_20261015.md` | .md | 8,889 | 124 | `docs-reports` | 报告 | 只读报告（证据留存，内容不改；后续以新报告或指针追加） | v1.0 | 20261015 | 6 | 1 |
| 199 | `docs/reports/PES2TS_PES-C-G架构审核与创新性可行性分析_20261004.md` | .md | 38,919 | 449 | `docs-reports` | 报告 | 审核报告（只读证据与修改建议；不使建议自动生效） | v1 | 20261004 | 6 | 1 |
| 200 | `docs/reports/PES2TS_V4.2-V4.7_过夜战役汇总_20261018.md` | .md | 54,556 | 383 | `docs-reports` | 报告 | 草稿（自动聚合渲染；最终结论待人工终审） | - | - | 4 | 1 |
| 201 | `docs/reports/PES2TS_V52R_输入充分性决策记录_20261021.md` | .md | 7,378 | 79 | `docs-reports` | 报告 | DECIDED（2026-10-21） | - | - | 10 | 1 |
| 202 | `docs/reports/PES2TS_V52_任务纠偏审计_20261021.md` | .md | 8,353 | 143 | `docs-reports` | 报告 |  | - | 20261021 | 17 | 1 |
| 203 | `docs/reports/PES2TS_V53MP_S0_数据建库与资格报告_20261022.md` | .md | 6,572 | 63 | `docs-reports` | 报告 | S0 完成（全部门通过） · 阶段 | - | - | 9 | 1 |
| 204 | `docs/reports/PES2TS_V53MP_S1_2k冒烟报告_20261022.md` | .md | 3,812 | 50 | `docs-reports` | 报告 | S1 完成（三臂 ok，Move guard 全过，GPU 1,829s ≤ 3,600s 上限） · 阶段 | - | - | 8 | 1 |
| 205 | `docs/reports/PES2TS_V53MP_S2_5k规模报告_20261022.md` | .md | 6,152 | 71 | `docs-reports` | 报告 | S2 完成（9/9 done，Move guard 全过，零 NaN，GPU 合计 13,048s） · 阶段 | - | - | 6 | 1 |
| 206 | `docs/reports/PES2TS_V53MP_S3_20k主结果报告_20261023.md` | .md | 10,354 | 102 | `docs-reports` | 报告 | S3 训练完成（9/9 done，Move guard 全过且全为负退化，零 NaN，GPU 合计 12,699s） ·… | - | - | 8 | 1 |
| 207 | `docs/reports/PES2TS_V53MP_S3_闭环评估报告_20261023.md` | .md | 6,287 | 65 | `docs-reports` | 报告 | 门 V53MP-S3-closed-loop —— 通过（执行门：reference-normalized 与 auto… | - | - | 2 | 1 |
| 208 | `docs/reports/PES2TS_V53MP_S4_20k_vs_50k_配对统计与C臂导出_20261027.md` | .md | 9,413 | 139 | `docs-reports` | 报告 |  | - | - | 3 | 1 |
| 209 | `docs/reports/PES2TS_V53MP_S4_50k_单路正式训练报告_20261027.md` | .md | 8,456 | 131 | `docs-reports` | 报告 |  | - | - | 2 | 1 |
| 210 | `docs/reports/PES2TS_V53MP_规模训练收益报告_2k_to_50k_20261027.md` | .md | 11,522 | 194 | `docs-reports` | 报告 |  | - | 20261027 | 1 | 1 |
| 211 | `docs/reports/PES2TS_V53MP_训练过程可视化报告_20261023.html` | .html | 379,567 | 1,660 | `docs-reports` | 报告 |  | - | - | 1 | 1 |
| 212 | `docs/reports/PES2TS_V541_Embedding_50K_实施与训练报告_20261029.md` | .md | 12,928 | 118 | `docs-reports` | 报告 | Executed-Frozen（只读最终报告；报告内容永不改写，后续仅可追加"当前阶段/作废"指针） | - | - | 14 | 1 |
| 213 | `docs/reports/PES2TS_V542_Backbone_50K_实施与训练报告_20261031.md` | .md | 14,222 | 173 | `docs-reports` | 报告 | Executed-Frozen（2026-10-31；owner 签认） | - | - | 14 | 1 |
| 214 | `docs/reports/PES2TS_V543_双Decoder_50K_实施与训练报告_20261106.md` | .md | 8,979 | 92 | `docs-reports` | 报告 | Executed-Frozen——18-run 正式矩阵全部执行完毕、H0–H3 统计合同全部产出、双轨冠军确证、T7 … | v1.0 | - | 4 | 1 |
| 215 | `docs/reports/PES2TS_V5第一阶段冲刺执行报告_20261018.md` | .md | 813 | 24 | `docs-reports` | 报告 |  | - | 20261018 | 1 | 1 |
| 216 | `docs/reports/PES2TS_generation关键问题与根因分析_20261002.md` | .md | 19,462 | 234 | `docs-reports` | 报告 |  | - | 20261002 | 3 | 1 |
| 217 | `docs/reports/PES2TS_v4.1阶段训练_修复输入语义与最小实验矩阵_最终报告_20261017.md` | .md | 23,911 | 238 | `docs-reports` | 报告 | 阶段执行总结与问题复核 | v1.1 | 20261017 | 5 | 1 |
| 218 | `docs/reports/PES2TS_v4阶段训练_最终报告与问题复核_20261017.md` | .md | 25,757 | 228 | `docs-reports` | 报告 | 阶段执行总结与问题复核 | v1.0 | 20261017 | 6 | 1 |
| 219 | `docs/reports/PES2TS_多坐标PES逐步实施方案_20261001.md` | .md | 20,953 | 279 | `docs-reports` | 报告 |  | - | 20261001 | 2 | 1 |
| 220 | `docs/reports/PES2TS_多坐标同步第一轮实现与实测_20261002.md` | .md | 10,796 | 80 | `docs-reports` | 报告 |  | - | - | 2 | 1 |
| 221 | `docs/reports/PES2TS_多键同步一维扫描全面问题报告_20261001.md` | .md | 31,913 | 355 | `docs-reports` | 报告 |  | - | 20261001 | 4 | 1 |
| 222 | `docs/reports/PES2TS_实现报告_20261018.md` | .md | 41,488 | 524 | `docs-reports` | 报告 | 实现与最新训练证据汇总 | - | 20261018 | 12 | 1 |
| 223 | `docs/reports/PES2TS_对话与现有任务完整总结报告_20261002.md` | .md | 24,763 | 253 | `docs-reports` | 报告 |  | - | 20261002 | 0 | 1 |
| 224 | `docs/reports/PES2TS_开发阶段性报告_20260930.md` | .md | 27,928 | 433 | `docs-reports` | 报告 |  | - | - | 3 | 1 |
| 225 | `docs/reports/PES_GENERATION_单端预测校正延续可行性分析_20261002.md` | .md | 20,788 | 235 | `docs-reports` | 报告 |  | - | 20261002 | 1 | 1 |
| 226 | `docs/reports/PES_generation_多坐标约束学习_ORCA能力与数学方案_20261001.md` | .md | 26,812 | 584 | `docs-reports` | 报告 | 设计方案，尚未实现模型、修改生产算法或提交新计算 | - | 20261001 | 3 | 1 |
| 227 | `docs/PES2TS_开发宪法_v1.md` | .md | 50,706 | 480 | `docs-root` | 治理规范 | 权威（唯一规范来源 / SSOT） | v2.9 | 20261021 | 128 | 1 |
| 228 | `docs/README.md` | .md | 81,513 | 355 | `docs-root` | 索引 |  | - | - | 237 | 1 |
| 229 | `docs/superseded/PES2TS_V5.1-R3_端点对称特征与损失重训方案_v1_20261020.md` | .md | 21,896 | 815 | `docs-superseded` | 阶段计划 | Superseded （2026-10-20） | - | 20261020 | 2 | 1 |
| 230 | `docs/superseded/PES2TS_V5.1-R3_端点对称特征与损失重训方案_v2_20261020.md` | .md | 24,296 | 840 | `docs-superseded` | 阶段计划 | Superseded （2026-10-20） | - | 20261020 | 2 | 1 |
| 231 | `docs/superseded/PES2TS_单端点IRC生成控制器_产品设计_v1_20261020.md` | .md | 18,858 | 730 | `docs-superseded` | 设计 | Superseded （2026-10-20；2026-10-21 后续） | - | 20261020 | 2 | 1 |
| 232 | `docs/superseded/PES2TS_单端点IRC生成控制器_产品设计_v2_20261020.md` | .md | 11,851 | 552 | `docs-superseded` | 设计 | Superseded （2026-10-20；2026-10-21 后续） | - | 20261020 | 2 | 1 |
| 233 | `docs/superseded/README.md` | .md | 3,463 | 38 | `docs-superseded` | 索引 |  | - | - | 237 | 1 |
| 234 | `examples/contracts_v1/README.md` | .md | 524 | 3 | `examples` | 产品文档 |  | - | - | 237 | 1 |
| 235 | `examples/contracts_v1/path_viewer.html` | .html | 15,252 | 45 | `examples` | 产品文档 |  | - | - | 6 | 1 |
| 236 | `examples/m1_first_plan/README.md` | .md | 838 | 5 | `examples` | 产品文档 |  | - | - | 237 | 1 |
| 237 | `24H_EXECUTION_TIMELINE.md` | .md | 10,335 | 81 | `root` | 运营记录 |  | - | - | 6 | 1 |
| 238 | `24H_FINAL_REPORT.md` | .md | 7,732 | 94 | `root` | 报告 |  | - | 20261020 | 8 | 1 |
| 239 | `AGENT.md` | .md | 26,011 | 205 | `root` | 治理规范 | 权威（执行投影，非独立规范） | v3.16 | 20261017 | 141 | 1 |
| 240 | `CONTRIBUTING.md` | .md | 2,589 | 80 | `root` | 外部说明 |  | - | - | 3 | 1 |
| 241 | `FAILURE_AND_REMAINING_WORK.md` | .md | 4,579 | 50 | `root` | 运营记录 |  | - | 20261020 | 7 | 1 |
| 242 | `LICENSE` | - | 1,073 | 21 | `root` | 外部说明 |  | - | - | 27 | 1 |
| 243 | `NEXT_DAY_DECISION_MEMO.md` | .md | 4,382 | 47 | `root` | 运营记录 |  | - | 20261020 | 7 | 1 |
| 244 | `R3_RESULT_SUMMARY.md` | .md | 3,636 | 47 | `root` | 报告 | R3 阶段关闭（dev-positive / confirm-blocked） | - | 20261020 | 9 | 1 |
| 245 | `README.md` | .md | 104,309 | 1,312 | `root` | 产品文档 |  | - | - | 237 | 1 |
| 246 | `V52_PROGRESS_STOP_REPORT.md` | .md | 5,640 | 73 | `root` | 报告 |  | - | 20261020 | 6 | 1 |
| 247 | `V53MP_RESUME_20261022.md` | .md | 4,946 | 63 | `root` | 运营记录 |  | - | - | 0 | 1 |
| 248 | `V53_ROLLOUT_REPORT.md` | .md | 3,582 | 45 | `root` | 报告 | 历史命名（V5.3-ENG） | - | 20261020 | 12 | 1 |
| 249 | `requirements-pesn-runtime.txt` | .txt | 365 | 7 | `root` | 其它 |  | - | - | 12 | 1 |
| 250 | `requirements-pesn-training.txt` | .txt | 270 | 8 | `root` | 其它 |  | - | - | 16 | 1 |
| 251 | `requirements.txt` | .txt | 116 | 9 | `root` | 其它 |  | - | - | 22 | 1 |
| 252 | `scripts/README.md` | .md | 1,426 | 32 | `scripts` | 索引 |  | - | - | 237 | 1 |
| 253 | `scripts/archive/README.md` | .md | 2,655 | 38 | `scripts` | 索引 |  | - | - | 237 | 1 |

---
*本报告由 `collect_inventory.py` 自动生成, 记录总数 253。数据以 `doc_inventory.csv` / `doc_inventory.json` 为准。*
