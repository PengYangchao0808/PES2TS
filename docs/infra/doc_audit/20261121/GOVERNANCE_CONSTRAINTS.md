# PES2TS 文档治理现行约束基线（GOVERNANCE_CONSTRAINTS）

> 状态：审计基线（只读取证，不构成新规范；规范冲突一律以宪法为准）
> 用途：供"任何新建档方案/文档重构方案"直接引用的现行规则与门禁事实
> 审计对象：PES2TS 仓库 `master` @ `2f1d3baff574936cf6ee044798cfd006f908c0d5`（2026-11-21，工作树 clean）
> 审计执行：task-2「文档治理现行规则与门禁事实提取」，只读仓库，未修改/移动任何既有文件（含门禁配置与 CI）
> 实测时间：2026-11-21T22:27:31+08:00（门禁基线，见 `baseline_gate_outputs.txt`）
> 配套产物：`baseline_gate_outputs.txt`（4 类门禁原始输出）、`status_vocabulary_stats.txt`（状态词表全量统计）、`pinned_paths.txt`（钉死路径逐条清单）、`index_coverage_stats.txt`（双索引覆盖明细）、`extract_rules.py`（统计脚本）

**一句话结论**：现行机器门禁 4 类当前全绿（`docs_consistency --strict` 158 文件 0 违规；链接检查 182 docs 0 error/172 warning；G–V 映射 current；registry 测试 26 passed），但门禁只覆盖"禁用语 + 历史横幅 + 必选子串 + 链接存在性 + G–V 渲染一致性"，**不覆盖状态头规范、版本一致性、单权威、AGENT↔宪法对齐**；因此存在 **12 项可复现的治理缺口/冲突**（§5），其中 4 类文件名与正文版本不一致、42 份 live 文档无状态块、状态词表实际 17 个归并桶且与官方 5 值图例不收敛。路径钉死方面：`docs_consistency_config.json` 17 historical + 27 required_files + 9 root_scope，加上 `g_v_map.yaml`/CI 间接钉死，共 **52 条硬钉死路径**（31 份 `docs/` 文档 + 9 份根 md + 13 份 config/模板/validator），另有 1 条软登记（`R3_RESULT_SUMMARY.md`）。

---

## 1. 现行规则条文清单（编号 + 原文要点 + 出处）

> 出处行号均已对照当前 HEAD 复核。规则按"效力层级 → 文档治理 → 状态头 → 修订/取代 → 索引同步 → 只读区 → 运行区 → 命名/入库"排列。

### 1.1 规范层级与效力

| 编号 | 规则要点 | 出处 |
|---|---|---|
| G-01 | **单一规范来源（SSOT）**：宪法是唯一规范来源；任何"必须/禁止"语句只允许存在于宪法或宪法明确登记的版本化接口规范；其余文档只做引用，不得重复声明规范。 | `docs/PES2TS_开发宪法_v1.md:23`（§0.1） |
| G-02 | **三层效力**：红线 R1–R6 不可违背（触碰=架构变更走 §12）；工程政策 P1–P5 默认绑定、可经时限豁免；接口 I-* 只以新版本变更。 | `docs/PES2TS_开发宪法_v1.md:26-30`（§0.2） |
| G-03 | **执行投影防漂移**：仓库根 `AGENT.md` 只是宪法的派生投影，不得新增规范；若摘要与宪法分歧**以宪法为准并须立即修正 AGENT.md**。 | `docs/PES2TS_开发宪法_v1.md:33-38`；`AGENT.md:10,201-205` |
| G-04 | **命名澄清**："开发宪法"=宪法文件；旧称"架构守则"=`AGENT.md`（现为执行投影）。 | `docs/PES2TS_开发宪法_v1.md:40-43`（§0.4） |
| G-05 | **每个开发动作前置检查**：①是否触碰 R1–R6；②是否新增/修改 I-*（只以新版本）；③新能力须归属唯一一层并登记台账；④科学声明只用四层状态词并提供成本账本。 | `docs/PES2TS_开发宪法_v1.md:45-49`（§0.5） |
| G-06 | **唯一主干编号**：程序进度只用 G0/G1/G2/G3/G4，工作流只用 C/R/X；禁止再发明新字母阶梯；任何旧阶梯编号出现必须同时给出对应 G#/工作流；禁止静默复活旧阶梯。 | `AGENT.md:49-50`；`docs/plans/PES2TS_文档修订台账.md:418,565-567`；`docs/PES2TS_开发宪法_v1.md:383-384,430-431` |
| G-07 | **一文档一职责**：每份文档只对一个领域权威，其余引用不重复。 | `AGENT.md:145`；`docs/plans/PES2TS_文档修订台账.md:419` |
| G-08 | **规范只写在宪法**：其余文档引用宪法，不重复规范条文。 | `AGENT.md:149`；`docs/PES2TS_开发宪法_v1.md:23` |

### 1.2 文档治理（AGENT.md §9 + 台账 §1）

| 编号 | 规则要点 | 出处 |
|---|---|---|
| G-09 | 文档治理五条：**单一主干编号 / 一文档一职责 / 状态头强制 / 取代登记 / 报告只读**。 | `AGENT.md:143-149` |
| G-10 | **状态头强制**：每份 plan/design 顶部须有状态块，字段=`状态 / 层级 / 取代关系 / 上游依赖 / 下游消费者 / 当前版本`。 | `AGENT.md:146`；`docs/plans/PES2TS_文档修订台账.md:552-561`（§4） |
| G-11 | **取代登记**：被取代文档不删除，加"已并入 X"注记；已降级/取代文档须带顶部指针；宪法 §12.4 进一步要求"已取代文档不得出现在检索/嵌入索引中"。 | `AGENT.md:145,148`；`docs/PES2TS_开发宪法_v1.md:428`；`docs/plans/PES2TS_文档修订台账.md:421` |
| G-12 | **报告只读**：`docs/reports/**` 是历史证据，永不修改内容；仅可加"当前阶段/作废"指针。 | `AGENT.md:148`；`docs/plans/PES2TS_文档修订台账.md:422,544-548` |
| G-13 | **一切新努力先登记主干**：新工作必须声明属于 `G#` 或 `C/R/X`，否则不立项；立项须归于明确模块目录。 | `AGENT.md:49,132-139`；`docs/plans/PES2TS_文档修订台账.md:423` |
| G-14 | 台账 §1 六条治理原则（单一主干编号/一文档一职责/状态头强制/取代登记/报告只读/新努力先登记主干）。 | `docs/plans/PES2TS_文档修订台账.md:416-423` |

### 1.3 状态头与完成状态

| 编号 | 规则要点 | 出处 |
|---|---|---|
| G-15 | **状态头规范模板**（4 值示例）：`> 状态：权威 | 参考 | 已取代 | 历史`，另有层级/取代关系/上游依赖/下游消费者/当前版本字段。 | `docs/plans/PES2TS_文档修订台账.md:552-561`（§4） |
| G-16 | **CI 须校验状态头**：宪法 §12.4 要求 CI 校验"所有 `docs/**/*.md` 带状态头；每主题恰有一份 AUTHORITATIVE 文档；已降级/取代文档带顶部指针；已取代文档不进检索索引"。 | `docs/PES2TS_开发宪法_v1.md:426-428`（§12.4） |
| G-17 | **四层完成状态词**：`unknown / implemented / fixture_passed / smoke_passed` 分别表达；只有真实冒烟与科学评测才能升级状态；不得因"写入计划"升级能力状态。 | `docs/PES2TS_开发宪法_v1.md:402-403`（§11.5）；`AGENT.md:124-125`（§7.4） |
| G-18 | **M0–M5 命名边界**：科学里程碑只在评测/论文语境使用，**不出现在计划状态头中**；G0–G4 才是工程阶段。 | `docs/PES2TS_开发宪法_v1.md:366-367,383-384`（§10.2/§10.4） |
| G-19 | **版本命名纪律**：G0–G4 为工程阶段、G3.P 为 G3 内支线、G3-P1 为批次、C/R/X 为工作流、PES-C/G/R/N 为模式/内核、发布包/模型/数据版本为产物身份；旧 L-G*/L-R* 仅作历史映射。 | `docs/PES2TS_开发宪法_v1.md:383-384`（§10.4） |

### 1.4 修订、豁免、日落

| 编号 | 规则要点 | 出处 |
|---|---|---|
| G-20 | **架构变更五步缺一不可**：声明 → ADR（`docs/design/decisions/ADR-XXXX-*.md`，不原地改写既有结论）→ 版本化（新接口新版本，旧产物不覆盖）→ 证据 → 同步（宪法版本、AGENT.md、台账、`docs/README.md`）。 | `docs/PES2TS_开发宪法_v1.md:409-415`（§12.1）；`AGENT.md:107-113`（§7.2） |
| G-21 | 宪法自身修订须 version bump + 变更记录；重大不变量变更须同时更新 §3 与 `AGENT.md` 摘要。 | `docs/PES2TS_开发宪法_v1.md:417-418`（§12.2） |
| G-22 | **豁免仅 P1–P5 可时限豁免**（R1–R6 不可豁免）；须记录为 ADR（理由/范围/到期条件/回滚计划）；禁止用豁免绕过 R 或用于 holdout 调参。 | `docs/PES2TS_开发宪法_v1.md:420-424`（§12.3）；`AGENT.md:127-128`（§7.5） |
| G-23 | **对齐律 + CI 文档权威 lint（防漂移）**：CI 须校验 AGENT.md 只含红线 ID 与一句话摘要、与宪法 §3 一致；须校验状态头/单权威/顶部指针等。 | `docs/PES2TS_开发宪法_v1.md:426-428`；`AGENT.md:204` |
| G-24 | **Sunset**：被取代的规范/接口/文档须在台账登记取代关系并设定日落。 | `docs/PES2TS_开发宪法_v1.md:430-431`；`docs/plans/PES2TS_文档修订台账.md:565-567`（§5） |
| G-25 | **接口注册表**：I-* 只以新版本变更，且必须登记于 `docs/plans/PES2TS_文档修订台账.md`。 | `docs/PES2TS_开发宪法_v1.md:339-341`（§9.6）；P1 `:172-174` |
| G-26 | **P1 历史产物绝不原地改写**（旧 `g2_path_v1`、191,148/183,460 条历史产物只读，扩展用新版本或 hash 绑定旁车）。 | `docs/PES2TS_开发宪法_v1.md:172-174` |

### 1.5 索引同步与门禁强制点

| 编号 | 规则要点 | 出处 |
|---|---|---|
| G-27 | **新增/修改计划文档后必须同步修订台账 + `docs/README.md`**。 | `AGENT.md:147`（§9.3） |
| G-28 | **训练前文档门强制**：`python scripts/agent_ops/docs_consistency.py --strict` 在训练前强制（当前记录于 V5.3-MP 进度指针）。 | `AGENT.md:159,163`（§10） |
| G-29 | 台账是**规范映射与取代登记**，服从宪法；两者不一致时以宪法为准并修正台账。 | `docs/PES2TS_开发宪法_v1.md:455-456`（§14） |
| G-30 | **门禁工具链本身是受保护输入**：`docs_consistency_config.json`、`check_docs_links.py`、`render_gv_map.py`、`gv_registry.py`、`g_v_map.yaml`、CI 工作流共同定义门禁；本审计期间保持零改动。 | 证据：`scripts/agent_ops/docs_consistency.py:57-68`；`.github/workflows/ci.yml:20-25`；`config/registry/g_v_map.yaml:4-7` |

### 1.6 只读区与归档边界

| 编号 | 规则要点 | 出处 |
|---|---|---|
| G-31 | **`docs/archive/` 是版本快照区**：只读、不修改；**禁止把过时文档搬入**（它不是"过时文档墓地"，移动会破坏宪法/台账引用）；链接检查必须排除本目录；新增快照命名 `YYYYMMDD_pre_<主题>/` 并携带 manifest/哈希。 | `docs/archive/README.md:3,15-20` |
| G-32 | **`docs/superseded/` 是已取代文档存放区**：只收"自身已声明被取代"的文档；**不收** `docs/reports/**`、`docs/archive/**`、以及 `historical` 名单中的文档（移动会断门禁）；移入须同时做三件事（修链接 / 本目录索引登记 / 台账增量登记）；正文不改。 | `docs/superseded/README.md:14-24` |
| G-33 | `docs/superseded/` 索引表是人工登记（4 条已迁入记录），"未收录但常被误认为该归档"的 5 份文档因在 `historical` 名单中必须留在原地。 | `docs/superseded/README.md:26-33,35-38` |

### 1.7 agent_runs 运行区规则（非规范编排层）

| 编号 | 规则要点 | 出处 |
|---|---|---|
| G-34 | `docs/agent_runs/` 只承载编排/状态/交接，**不新增科研规范**；作为运营记录被列入 `token_exempt_prefixes`。 | `docs/agent_runs/README.md:6`；`scripts/agent_ops/docs_consistency_config.json:24-26` |
| G-35 | 运行区硬规则：正式训练/解封两个独立 token；CUDA 进程持 GPU 锁；科研不可变项（指标/阈值/损失/数据身份/标签语义）任何 agent 不得修改；每文件唯一写入负责人；中断恢复按 run_id 幂等；旧证据目录只读。 | `docs/agent_runs/README.md:34-41` |

### 1.8 入库边界（文档/数据/产物）

| 编号 | 规则要点 | 出处 |
|---|---|---|
| G-36 | 只入库小体量人类可读 manifest：`data/manifests/*.json`、`*.csv`、`v5/*.json`；`data/raw|interim|ground_truth`、`*.h5/hdf5/xyz/log` 不入库（测试 fixture 例外）。 | `.gitignore:1-22` |
| G-37 | `outputs/` 整体忽略，仅白名单个别证据目录入库；会话缓存/`/.slim/`/`/.omo/` 等不入库（**本审计产物位于 `.slim/deepwork/`，属本地工作目录，不入库**）。 | `.gitignore:29-52,54-64` |
| G-38 | `docs/archive/** -text`（快照禁止文本归一化）；`data/manifests/*ids* -text`（冻结 split 逐字节，保证 manifest sha256 可复核）。 | `.gitattributes:11,13-16` |

---

## 2. 门禁实现事实（扫描范围 / 检查项 / 输出与退出行为 / CI 触发 / 实测基线）

### 2.1 门禁全景

| # | 门禁 | 命令 | 入口实现 | 是否在 CI | 当前实测 |
|---|---|---|---|---|---|
| 1 | 文档一致性（禁用语 + historical 横幅 + required_files 子串） | `python scripts/agent_ops/docs_consistency.py --strict` | `scripts/agent_ops/docs_consistency.py:251-302` | **否**（训练前人工强制，`AGENT.md:159,163`） | scanned 158 files; 0 violation; **CLEAN**；exit 0 |
| 2 | 相对链接存在性 | `python scripts/check_docs_links.py` | `scripts/check_docs_links.py:46-86` | **是**（`.github/workflows/ci.yml:22-23`） | checked 182 docs; **0 error, 172 warning**；exit 0 |
| 3 | G–V 映射文档时效 | `python scripts/render_gv_map.py --check` | `scripts/render_gv_map.py:104-139` | **是**（`ci.yml:20-21`） | "G–V mapping doc is current"；exit 0 |
| 4 | registry/config 测试 | `python -m pytest -q tests/test_gv_registry.py tests/test_config_env.py` | `tests/test_gv_registry.py`、`tests/test_config_env.py` | **是**（`ci.yml:24-25`） | **26 passed**；exit 0 |
| 5 | 全仓测试档（两档） | CI `base` 档（无 torch）与 `pesn` 档（CPU torch，`pytest -q`） | `ci.yml:26-37,39-53` | 是 | 本次未跑（与文档门禁无直接关系；CI 事实记录） |

**CI 触发条件**：`pull_request`（全部 PR）+ `push` 到 `master` 分支（`.github/workflows/ci.yml:3-6`）。CI 共两 job：`base`（ubuntu + py3.12 + requirements.txt；4 道门/测试）与 `pesn`（额外装 CPU torch 后跑全量 `pytest -q`）。

### 2.2 实测基线（2026-11-21T22:27:31+08:00，master@2f1d3ba，工作树 clean，Python 3.12.7 / Windows）

| # | 完整命令 | 退出码 | 关键输出 | 耗时 |
|---|---|---|---|---|
| 1 | `python scripts/agent_ops/docs_consistency.py --strict` | **0** | `docs_consistency: scanned 158 file(s); 0 violation(s); CLEAN` | 0.2s |
| 2 | `python scripts/check_docs_links.py` | **0** | `checked 182 docs; 0 error(s), 172 warning(s)` | 0.2s |
| 3 | `python scripts/render_gv_map.py --check` | **0** | `G–V mapping doc is current` | 0.2s |
| 4 | `python -m pytest -q tests/test_gv_registry.py tests/test_config_env.py` | **0** | `26 passed in 1.20s` | 2.3s |

原始 stdout/stderr 全文（含 172 条 warning 逐条、命令、时间、退出码）见 `baseline_gate_outputs.txt`。

**warning 构成（门禁 2 的"容忍债务"，不计 error）**：172 条 = `E:/` 绝对路径链接 **101 条** + 本机/`outputs`/`.slim` 等 local-only 链接 **71 条**；涉及 31 份文档，集中在历史报告（如 `docs/reports/PES2TS_多坐标PES逐步实施方案_20261001.md` 21 条、`PES2TS_多键同步一维扫描全面问题报告_20261001.md` 19 条）。这些是历史本机证据链接，设计上只 warning 不 fail（`scripts/check_docs_links.py:54-70`）。

### 2.3 门禁 1：`docs_consistency.py` 实现事实

- **扫描范围**：`docs/**/*.md`，排除 `docs/archive/` 前缀、`historical` 名单、`token_exempt_prefixes`（`docs/agent_runs/`）；另加 `root_scope` 中非 historical 的根文件（`docs_consistency.py:111-137`）。
- **扫描数口径验证**：158 = docs 非 archive 180 − historical 文档 12 − `docs/agent_runs/` md 14 + 非 historical 根文件 4（脚本口径实测；config 中 historical 共 17=12 docs + 5 根 memo）。
- **三类检查项**（`docs_consistency.py:18-27,140-207`）：
  1. **forbidden-token**：6 个禁用语，命中行若无 exemption marker 即 `file:line:token` 违规；
  2. **historical**：17 个 historical 文件必须存在且正文含横幅标记 `ADR-0030`（`historical_banner_marker`）；
  3. **required-files**：27 个文件必须存在且含全部"必须子串"（当前 27/27 存在、子串 0 缺失）。
- **输出与退出行为**：config 缺失/不可读 → `CONFIG ERROR`，exit **2**（`:275-279`）；默认 advisory 模式打印 `VIOLATION ...` 但始终 exit 0；`--strict` 下只要非 clean 即 exit **1**（`:288-302`）。输出固定摘要格式 `docs_consistency: scanned N file(s); M violation(s) [category=count]; CLEAN|VIOLATIONS`。
- **不含**：状态头格式校验、单权威校验、AGENT↔宪法对齐、版本一致性（见 §5 C-09）。

### 2.4 门禁 2：`check_docs_links.py` 实现事实

- **扫描范围**：根 `README.md`、`AGENT.md`、`docs/**/*.md` 排除 `docs/archive/**`（`check_docs_links.py:23-30`）；当前 182 = docs 非 archive 180 + README + AGENT。
- **检查项**：Markdown 相对链接目标存在性；`http(s)/mailto/tel/#` 跳过；`E:/`、`e:/` 绝对路径 → warning（`:54-56`）；含 `outputs`/`.slim`/`.omo` 或 `.log` 的本机链接 → warning（`:60-70`）；相对链接解析失败 → `ERROR ... missing link target`，exit **1**；否则 exit 0（`:71-86`）。
- **不覆盖**：`docs/archive/**`（快照区故意排除）；非 Markdown 文件；锚点存在性（仅去 `#` 片段）。

### 2.5 门禁 3/4：`render_gv_map.py` + `gv_registry.py` + 测试

- **SSOT**：`config/registry/g_v_map.yaml`；渲染目标固定为 `docs/design/PES2TS_G_V_MAPPING_v1_20261018.md`（`render_gv_map.py:29`）。`--check` 将渲染结果与目标文件逐字节比较：目标缺失/不等 → exit 1 并提示 `--write`；YAML 结构/交叉引用错误 → `GVMapError` → exit 2；一致 → `G–V mapping doc is current` → 0（`render_gv_map.py:111-137`）。
- **路径校验分级**（`pes2ts_core/gv_registry.py:239-307`）：仅 `active`/`executed` 战役强制 `code_scope`（glob 命中）、`config`/`data_manifest`/`evidence`（仓库内相对路径必须存在）与 `comparison_only`；`proposed`/`archived` 不强制存在；**仓库外路径**（POSIX 绝对/Windows 盘符/UNC/resolve 逃逸）只作 provenance 记录、不强制存在（`gv_registry.py:218-236,278-302`）。
- **run record 校验**：`gv` 块必填 `map_version`（须等于当前 map_version）/`campaign_id`/`primary_g`/`supporting_g`/`gates`；`*_sha256` 须 64 位小写 hex（`gv_registry.py:310-354`）。
- **registry 测试的额外钉死**：`tests/test_gv_registry.py:168-169` 断言 `V5.4.2.evidence` 必须以 `PES2TS_V542_Backbone_50K_实施与训练报告_20261031.md` 结尾；同文件 `:28,33` 使用 `stage_doc="docs/plans/PES2TS_阶段定义_G0-G3.md"`（fixture 字符串）。`tests/test_config_env.py` 只测 `${ENV}` 配置展开（49 行，与文档无关，但被 CI 一并执行）。

### 2.6 门禁覆盖缺口（实现层）

- `docs_consistency` **未接入 CI**：`ci.yml` 的 4 个步骤中无该命令（`ci.yml:20-25`），它目前只在训练前人工强制执行（`AGENT.md:159,163`）。
- 宪法 §12.4 要求的三项 CI 校验（**状态头 / 单权威 / AGENT↔宪法对齐**）**在 `ci.yml` 与 `docs_consistency.py` 中均未实现**；台账 §6 仍挂着 `[ ] 提交 CI 文档权威 lint ...` 未完成项（`docs/plans/PES2TS_文档修订台账.md:585`）。
- 因此"门禁全绿"**不能**推出"文档治理合规"：§5 的版本漂移（如 `docs/README.md:179` 写宪法 v2.7，实际 v2.9）在当前 CLEAN 状态下长期存在。

---

## 3. 路径钉死清单（"移动即断门禁"逐条）

> 判定原则：某路径被门禁配置/脚本/CI 以字符串硬引用，或为门禁断言的 evidence/渲染目标 → **移动、重命名、删除即断门禁**（下表逐条标注）。所有路径均相对仓库根。

### 3.0 钉死路径汇总

| 类别 | 条数 | 来源 | 断门禁机制 |
|---|---:|---|---|
| `historical` | **17** | `docs_consistency_config.json:27-45` | 文件缺失 → `historical_missing_file`；缺 `ADR-0030` 横幅 → `historical_missing_banner`；`--strict` exit 1 |
| `required_files` | **27**（14 份 docs + 13 份 config/模板/validator） | `docs_consistency_config.json:57-419` | 缺失 → `required_missing_file`；子串缺失 → `required_substring_missing`；`--strict` exit 1 |
| `root_scope` | **9** | `docs_consistency_config.json:46-56` | 纳入禁用语扫描；5 份同时是 historical（缺失即 fail）；`README.md`/`AGENT.md` 另受链接门约束 |
| `token_exempt_prefixes` | **1**（`docs/agent_runs/`） | `docs_consistency_config.json:24-26` | 前缀内文件豁免禁用语扫描与横幅要求（运营记录定位，**特例需保留**） |
| `forbidden_tokens` | **6** | `docs_consistency_config.json:2-9` | 活跃文档命中且无豁免标记 → violation |
| `exemption_markers` | **12** | `docs_consistency_config.json:10-23` | 同行出现任一 marker 即豁免 |
| `historical_banner_marker` | **1**（`ADR-0030`） | `docs_consistency_config.json:420` | 17 historical 必须含该串 |
| `g_v_map` 引用的 docs 路径 | **43 条引用**（去重后 17 条文档路径：stage_doc 1 + evidence 15 + 文本注记 3；含 1 条根 memos） | `config/registry/g_v_map.yaml`（行号见 3.5） | 见 3.5：executed/active 战役 evidence/config 缺失 → `render_gv_map --check` exit 2；渲染目标缺失 → exit 1 |
| `render_gv_map` 渲染目标 | **1** | `scripts/render_gv_map.py:29` | 缺失/内容过期 → exit 1 |
| CI 直接引用 docs 路径 | **0** | `.github/workflows/ci.yml`（无 docs/ 字符串） | CI 经脚本/测试**间接**钉死（见 3.6） |
| **硬钉死路径合计（去重）** | **52** = 31 份 `docs/` 文档 + 9 份根 `.md` + 13 份 config/模板/validator | 3.1–3.6 并集 | 另有 1 条软登记：`R3_RESULT_SUMMARY.md`（仅 root_scope，缺失不 fail，见 3.3） |

### 3.1 `historical` 17 条（逐条；移动/重命名即断 `docs_consistency --strict`）

| # | 路径 | config 行 | 类别 | 当前存在 | ADR-0030 横幅 |
|---:|---|---:|---|---|---|
| 1 | `docs/contracts/PES2TS_ReactionIntent_Contract_v1.md` | L28 | contracts | ✅ | ✅ |
| 2 | `docs/contracts/PES2TS_GeometryPolicyContract_v1.md` | L29 | contracts | ✅ | ✅ |
| 3 | `docs/contracts/PES2TS_双端TS教师与相容转移合同_v2.md` | L30 | contracts | ✅ | ✅ |
| 4 | `docs/design/decisions/ADR-0021-无序双端拓扑与完整双侧TS教师主线.md` | L31 | ADR | ✅ | ✅ |
| 5 | `docs/design/decisions/ADR-0028-关键原子几何策略合同与V5-V7任务拆分.md` | L32 | ADR | ✅ | ✅ |
| 6 | `docs/design/decisions/ADR-0029-V5.2+双门评估与封存边界.md` | L33 | ADR | ✅ | ✅ |
| 7 | `docs/plans/PES2TS_30h_R3_V52_V53_分层训练与验收计划_v1.1_20261020.md` | L34 | plans | ✅ | ✅ |
| 8 | `docs/plans/PES2TS_V5-V7_三阶段研究与架构修订方案_v2.0_20261020.md` | L35 | plans | ✅ | ✅ |
| 9 | `docs/plans/PES2TS_V5.0-V5.2_十小时冲刺训练与验证执行方案_v1_20261018.md` | L36 | plans | ✅ | ✅ |
| 10 | `docs/plans/PES2TS_V5.1-R2_损失尺度与数据完整性修复方案_v1_20261020.md` | L37 | plans | ✅ | ✅ |
| 11 | `docs/plans/PES2TS_PES-N_V5-V7架构调整与实施计划_v1.2_20261018.md` | L38 | plans | ✅ | ✅ |
| 12 | `V52_PROGRESS_STOP_REPORT.md` | L39 | 根 memo | ✅ | ✅ |
| 13 | `NEXT_DAY_DECISION_MEMO.md` | L40 | 根 memo | ✅ | ✅ |
| 14 | `FAILURE_AND_REMAINING_WORK.md` | L41 | 根 memo | ✅ | ✅ |
| 15 | `24H_FINAL_REPORT.md` | L42 | 根 memo | ✅ | ✅ |
| 16 | `24H_EXECUTION_TIMELINE.md` | L43 | 根 memo | ✅ | ✅ |
| 17 | `docs/contracts/PES2TS_FullPathNavigationContract_v1.md` | L44 | contracts | ✅ | ✅ |

**断门禁事实**：`docs_consistency.py:167-186` 对每条检查"存在 + 横幅"；当前 17/17 存在且 17/17 含 `ADR-0030`。台账明确记录这些文档"移动即断门禁 → 留原地"（`docs/plans/PES2TS_文档修订台账.md:35,48`；`docs/superseded/README.md:17-19`）。

### 3.2 `required_files` 27 条（逐条；移动/改名即断 `docs_consistency --strict`）

| # | 路径 | config 行 | 必须子串数 | 类型 |
|---:|---|---:|---:|---|
| 1 | `docs/design/decisions/ADR-0030-V5.2全路径导航与端点停止.md` | L58 | 4 | docs（ADR） |
| 2 | `docs/contracts/PES2TS_FullPathNavigationContract_v1.md` | L64 | 6 | docs（contract，且为 historical） |
| 3 | `docs/reports/PES2TS_V52_任务纠偏审计_20261021.md` | L72 | 4 | docs（report，只读） |
| 4 | `docs/design/decisions/ADR-0031-MOVE-PROCESS统一学习合同.md` | L78 | 4 | docs（ADR） |
| 5 | `docs/contracts/PES2TS_MoveProcessContract_v1.md` | L84 | 5 | docs（contract） |
| 6 | `docs/plans/PES2TS_V5.3-MP_大规模MOVE-PROCESS联合训练计划_v1_20261021.md` | L91 | 6 | docs（plan） |
| 7 | `docs/design/decisions/ADR-0032-V5.4.1-Embedding接口扩展与预注册治理.md` | L99 | 9 | docs（ADR） |
| 8 | `docs/plans/PES2TS_V5.4.1_Embedding_50K_详细阶段实施方案_v1.0_20261027.md` | L110 | 5 | docs（plan） |
| 9 | `config/campaigns/pes2ts_v541_embed_50k.yaml` | L117 | 7 | config |
| 10 | `docs/design/decisions/ADR-0033-V5.4.2-Backbone接口与预注册治理.md` | L126 | 38 | docs（ADR） |
| 11 | `docs/plans/PES2TS_V5.4.2_Backbone_50K_详细阶段实施方案_v1.0_20261029.md` | L166 | 21 | docs（plan） |
| 12 | `config/campaigns/pes2ts_v542_backbone_50k.yaml` | L189 | 44 | config |
| 13 | `config/campaigns/v542_templates/parity_certificate.template.json` | L235 | 11 | config 模板 |
| 14 | `config/campaigns/v542_templates/runplan_pin.template.json` | L248 | 13 | config 模板 |
| 15 | `config/campaigns/v542_templates/preregistration_signoff.template.json` | L263 | 17 | config 模板 |
| 16 | `config/campaigns/v542_templates/a0_reuse_decision.template.json` | L282 | 10 | config 模板 |
| 17 | `docs/reports/PES2TS_V542_Backbone_50K_实施与训练报告_20261031.md` | L294 | 6 | docs（report） |
| 18 | `docs/design/decisions/ADR-0034-V5.4.3-双Decoder与DecoderContextV1.md` | L302 | 5 | docs（ADR） |
| 19 | `docs/plans/PES2TS_V5.4.3_双Decoder_50K_全面开发与训练方案_v1.0_20261102.md` | L309 | 11 | docs（plan） |
| 20 | `docs/plans/PES2TS_V5.4.3_WP执行清单_v1.0_20261102.md` | L322 | 3 | docs（plan） |
| 21 | `config/campaigns/pes2ts_v543_dual_decoder_50k.yaml` | L327 | 20 | config |
| 22 | `scripts/agent_ops/v543_config_schema.py` | L349 | 13 | 脚本（受保护） |
| 23 | `config/campaigns/v543_templates/preregistration_signoff.template.json` | L364 | 8 | config 模板 |
| 24 | `config/campaigns/v543_templates/runplan_pin.template.json` | L374 | 8 | config 模板 |
| 25 | `config/campaigns/v543_templates/budget_policy.template.json` | L384 | 9 | config 模板 |
| 26 | `config/campaigns/v543_templates/parity_certificate.template.json` | L395 | 12 | config 模板 |
| 27 | `config/campaigns/v543_templates/decoder_context.template.json` | L409 | 8 | config 模板 |

**说明**：27 条当前 27/27 存在、全部必须子串 0 缺失。其中 14 份 docs（#1-8、10、11、17-20）移动即断 `--strict`；13 份非 docs（#9、12-16、21-27）虽不是文档，但同样被文档门禁硬钉死，**任何重构方案不得把它们当"可自由移动的文件"**。

### 3.3 `root_scope` 9 条（逐条）

| # | 路径 | config 行 | 同时是 historical | 断门禁机制 |
|---:|---|---:|---|---|
| 1 | `README.md` | L47 | 否 | `check_docs_links` 扫描面 + 大量文档反向链接；移动/改名 → 链接 error |
| 2 | `AGENT.md` | L48 | 否 | 同上；且宪法 §0.3/§12.4 的执行投影地位（`docs/PES2TS_开发宪法_v1.md:33-38`） |
| 3 | `V52_PROGRESS_STOP_REPORT.md` | L49 | ✅ | historical_missing_file → `--strict` exit 1 |
| 4 | `NEXT_DAY_DECISION_MEMO.md` | L50 | ✅ | 同上 |
| 5 | `FAILURE_AND_REMAINING_WORK.md` | L51 | ✅ | 同上 |
| 6 | `24H_FINAL_REPORT.md` | L52 | ✅ | 同上 |
| 7 | `24H_EXECUTION_TIMELINE.md` | L53 | ✅ | 同上 |
| 8 | `R3_RESULT_SUMMARY.md` | L54 | 否 | **软登记**：`_scope_files` 对 root_scope 缺失文件静默跳过（`docs_consistency.py:131-136`），无其他门禁引用 → 移动不 fail，但会退出治理扫描面 |
| 9 | `V53_ROLLOUT_REPORT.md` | L55 | 否 | **硬钉死（g_v_map）**：V5.3-ENG（executed）evidence；缺失 → `render_gv_map --check` exit 2（`gv_registry.py:278-302`） |

### 3.4 禁用词 / 豁免标记 / 豁免前缀 / 横幅标记（逐条）

- **`forbidden_tokens` 6 个**（`docs_consistency_config.json:2-9`）：`TSStop`、`TS Stop`、`TS-neighborhood`、`approach-side`、`TS-directed`、`No-Move candidate`。**（禁用词登记；禁止在活跃文档正文使用，故本行保留原文并带标记豁免）**
- **`exemption_markers` 12 个**（`config:10-23`）：`弃用`、`撤回`、`Superseded`、`superseded`、`取代`、`禁止`、`HISTORICAL`、`Historical`、`historical`、`LEGACY`、`POST-HOC`、`SUPERSEDED`。
- **`token_exempt_prefixes` 1 个**（`config:24-26`）：`docs/agent_runs/`（运营记录，禁用语扫描与横幅要求均豁免）。
- **`historical_banner_marker`**（`config:420`）：`ADR-0030`（17 条 historical 必含）。

### 3.5 `g_v_map.yaml` 引用/钉死的文档路径（逐条；移动即断 `render_gv_map --check`）

`config/registry/g_v_map.yaml` 共 **43 条 docs/文档路径引用**（去重后 17 条文档路径）。`code_scope` 字段内 **docs 路径 0 条**（全部为代码/config/test/scripts 路径；其存在性仅对 active/executed 战役强制）。逐条如下：

**(a) `stage_doc`（13 处引用同 1 条路径；值参与 G–V 文档渲染）**
- `docs/plans/PES2TS_阶段定义_G0-G3.md` ← L12,16,20,24,28,32,36,40,44,48,52,56,60。**移动即断**：`render_gv_map --check` 渲染字符串会变（`--write` 才可重建），且 AGENT/宪法/README 大量链接会报 error。注：`gv_registry.validate_gv_map` **不校验** `stage_doc` 存在性（`gv_registry.py:255-276` 未覆盖），所以必须是"渲染 + 链接"双重断。

**(b) `evidence`（26 处引用，去重 15 条；★=executed/active 战役，缺失直接 exit 2）**

| 行 | 路径 | 战役 | 是否强制存在 |
|---:|---|---|---|
| L79 | `docs/reports/PES2TS_实现报告_20261018.md` | V4.2 ★ | **是** |
| L96 | 同上 | V4.3 ★ | **是** |
| L112 | 同上 | V4.4 ★ | **是** |
| L127 | 同上 | V4.5 ★ | **是** |
| L143 | 同上 | V4.6 ★ | **是** |
| L159 | 同上 | V4.7 ★ | **是** |
| L181 | `docs/plans/PES2TS_V5.0-V5.2_十小时冲刺训练与验证执行方案_v1_20261018.md` | V5.0 | 否（proposed；该文件同时是 historical） |
| L197 | 同上 | V5.1 | 否（archived） |
| L216 | `docs/plans/PES2TS_V5.1-R2_损失尺度与数据完整性修复方案_v1_20261020.md` | V5.1-R2 | 否（archived；同时是 historical） |
| L237 | `docs/plans/PES2TS_V5.1-R3_反应意图条件化与端点标签不变重训方案_v3_20261020.md` | V5.1-R3 | 否（proposed） |
| L262 | `docs/plans/PES2TS_30h_R3_V52_V53_分层训练与验收计划_v1.1_20261020.md` | V5.2 ★ | **是**（同时是 historical） |
| L282 | `docs/reports/PES2TS_V52R_输入充分性决策记录_20261021.md` | V5.2-R | 否（archived） |
| L301 | `V53_ROLLOUT_REPORT.md` | V5.3-ENG ★ | **是**（根 memo，root_scope） |
| L329 | `docs/plans/PES2TS_V5.3-MP_大规模MOVE-PROCESS联合训练计划_v1_20261021.md` | V5.3-MP | 否（proposed；同时是 required_files） |
| L347 | `docs/plans/PES2TS_V5-V7_反应意图条件化与低维生长方案_v3.0_20261020.md` | V5.4 | 否（proposed） |
| L374 | `docs/reports/PES2TS_V541_Embedding_50K_实施与训练报告_20261029.md` | V5.4.1 ★ | **是** |
| L396 | `docs/reports/PES2TS_V542_Backbone_50K_实施与训练报告_20261031.md` | V5.4.2 ★ | **是**（同时是 required_files + registry 测试断言） |
| L418 | `docs/design/decisions/ADR-0034-V5.4.3-双Decoder与DecoderContextV1.md` | V5.4.3 | 否（proposed；同时是 required_files） |
| L444 | `docs/design/decisions/ADR-0026-V5冲刺key-atom位移合同与OOF语义.md` | V5_SPRINT ★ | **是** |
| L471 | `docs/plans/PES2TS_V5.1_修复冲刺与扩量准入方案_v1_20261019.md` | V5_1A_SPRINT ★ | **是** |
| L489/L504/L519/L536/L552/L568 | `docs/plans/PES2TS_PES-N_V5-V7架构调整与实施计划_v1.2_20261018.md` | V6.0/V6.1/V6.2/V7-A/V7-B/V7-C | 否（archived；同时是 historical） |

**(c) `gate_note`/标题等文本内嵌的 docs 引用（4 处）**
- L3 `docs/design/PES2TS_G_V_MAPPING_v1_20261018.md`（map 说明）；L7 `docs/reports/PES2TS_V542_Backbone_50K_实施与训练报告_20261031.md`；L7 `docs/reports/PES2TS_V541_Embedding_50K_实施与训练报告_20261029.md`；L397 V542 报告（gate_note 内）。这些不参与 `validate_gv_map` 的路径存在性检查，但属于"文档间引用事实"，移动会造成 YAML 文本悬空。

**(d) 渲染目标（独立硬钉死）**
- `docs/design/PES2TS_G_V_MAPPING_v1_20261018.md`（`scripts/render_gv_map.py:29`）：删除/移动 → `--check` exit 1；内容与 YAML 渲染不一致 → exit 1。**禁止手改该文件**，只可 `python scripts/render_gv_map.py --write` 重生成。

### 3.6 CI 直接 / 间接引用的 docs 路径（逐条）

- **直接**：`.github/workflows/ci.yml` 中 docs/ 路径 **0 条**（CI 只引用 scripts 与 tests）。
- **间接（4 条通道）**：
  1. `render_gv_map.py --check` → 渲染目标 `docs/design/PES2TS_G_V_MAPPING_v1_20261018.md`（`render_gv_map.py:29`；`ci.yml:20-21`）；
  2. `check_docs_links.py` → 扫描面 = `README.md` + `AGENT.md` + `docs/**/*.md`（排除 archive）（`check_docs_links.py:23-30`；`ci.yml:22-23`）→ **任何被这三类文档链接的文档移动都可能产生 error**；
  3. `pytest tests/test_gv_registry.py tests/test_config_env.py` → `tests/test_gv_registry.py:168-169` 断言 `docs/reports/PES2TS_V542_Backbone_50K_实施与训练报告_20261031.md` 后缀；`:28,33` 的 fixture 字符串钉 `docs/plans/PES2TS_阶段定义_G0-G3.md`；且测试整体重校验 `g_v_map.yaml` 全部路径；
  4. 两档 `pytest`（`ci.yml:26-37,52-53`）→ 全仓测试，可覆盖其他对文档路径的间接断言（本审计未穷举全仓测试中的文档字符串引用）。

### 3.7 其他治理明文"不移动/不改写"区域

- `docs/archive/**`：快照只读、**禁止搬入过时文档**、禁止移动（`docs/archive/README.md:3,15-20`）。
- `docs/reports/**`：只读历史证据，永不改内容（`AGENT.md:148`；`docs/plans/PES2TS_文档修订台账.md:422,544-548`）。
- `docs/superseded/**`：正文不改，仅允许修链接与加状态横幅（`docs/superseded/README.md:10,24`）。
- 根目录 7 份战役 memo：**推荐不移动**（其中 5 份在 historical、7 份在 root_scope；移动须同 PR 改门禁配置）——`docs/README.md:6-9`、`docs/infra/PES2TS_仓库归档与清理方案_v1_20261120.md:166-167`、台账 `2026-11-20` 条目（`docs/plans/PES2TS_文档修订台账.md:46`）。
- 冻结 split 名单：`data/manifests/*ids*` 必须逐字节入库（`.gitattributes:13-16`；`data/manifests/split_registry_v1.json` 索引），改动须同 commit 更新 registry SHA（台账 `:18`）。

### 3.8 "移动即断门禁"路径总数

- **Tier A（名单硬编码，确定性断 fail）**：17 historical ∪ 27 required_files = **43 条**（交集 1 条 = FullPathNavigationContract）。
- **Tier B（g_v_map 断言/渲染）**：+ render target 1 + executed/active evidence 新增 5 条（`PES2TS_实现报告_20261018`、`V53_ROLLOUT_REPORT.md`、`V541` 报告、`ADR-0026`、`V5.1_修复冲刺` 计划；V542 报告与 30h 计划已在 A）→ 累计 **49 条**。
- **Tier C（链接级 + stage_doc）**：+ `docs/plans/PES2TS_阶段定义_G0-G3.md`（stage_doc，且被 AGENT/宪法/README 链接）+ `README.md` + `AGENT.md` → 累计 **52 条**。
- **Tier D（索引链接级，面状）**：`docs/README.md` 索引到的 **133 份** live 文档 + 台账索引到的 **100 份**，移动任意一份被链接文档都会让 `check_docs_links` 从 0 error 变 >0（当前基线 182 docs / 0 error）。
- **软登记 1 条**：`R3_RESULT_SUMMARY.md`（仅 root_scope，缺失静默跳过，见 3.3）。

---

## 4. 状态词表事实（全库实际取值统计）

> 统计脚本：`extract_rules.py --section status`；全量明细见 `status_vocabulary_stats.txt`。统计口径：全库 `docs/**/*.md`（216 份，含 archive 36 / agent_runs 14 / superseded 5 / live 161），取每份文件前 40 行内首个 `状态：/Status:` 行，按"分隔符前首词"归一化。

### 4.1 状态块覆盖率

- **有状态块 167 / 216**：其中前 40 行内 **165 份**，仅在 40 行之外 **2 份**。
- **无任何状态块 49 份**（其中 live 42 份；archive 历史快照本就不强制，故 live 42 是真实缺口）。
- 官方示例词表只有 **5 个值**（`docs/README.md:118`）：`Canonical` / `Active-Proposed` / `Executed-Frozen` / `Superseded` / `Evidence-Historical`；台账 §4 模板给的是 **4 个中文值**（权威/参考/已取代/历史，`docs/plans/PES2TS_文档修订台账.md:552-561`）——三者互不一致。

### 4.2 live 文档状态取值 Top 20（归一化首词）

| # | 状态取值 | 次数 | 示例文件（最多 3） |
|---:|---|---:|---|
| 1 | `Accepted` | 18 | `docs/design/decisions/ADR-0010-PES-N反应状态价值与STOP-MOVE统一决策.md`；`ADR-0011-三模块产品架构与统一可视化.md`；`ADR-0012-Classic图论驱动与xTB原生约束扫描.md` |
| 2 | `Proposed` | 9 | `docs/design/decisions/ADR-0006-ORCA原生约束优化与控制策略分离.md`；`ADR-0018-RL纠错与动作支持研究计划.md`；`ADR-0019-IRC20k与RL2000分阶段扩量.md` |
| 3 | `历史提案` | 6 | `docs/design/decisions/ADR-0007-统一价值模型与搜索内生选点.md`；`ADR-0008-拓扑约束下的距离与位移联合预测.md`；`ADR-0009-G阶段下的统一模型开发框架.md` |
| 4 | `已接受` | 5 | `docs/design/decisions/ADR-0001-统一执行后端接缝.md`；`ADR-0002-计算后端统一经ACP执行.md`；`ADR-0003-G2-AB1接口登记.md` |
| 5 | `权威` | 5 | `docs/design/PES2TS_G_V_MAPPING_v1_20261018.md`；`docs/PES2TS_开发宪法_v1.md`；`docs/plans/PES2TS_整体开发与发布方案_v2.md` |
| 6 | `Executed-Frozen` | 5 | `docs/plans/PES2TS_V4.2-V4.7_十小时训练实验方案_20261017.md`；`docs/plans/PES2TS_V5.4.1_Embedding_50K_详细阶段实施方案_v1.0_20261027.md`；`docs/reports/PES2TS_V541_Embedding_50K_实施与训练报告_20261029.md` |
| 7 | `前版记录` | 4 | `docs/design/decisions/ADR-0020-EGNN直接位移与BC-IQL-SAC研究主线.md` 等 3 份（见统计文件） |
| 8 | `参考` | 4 | `docs/design/PES2TS_CODE_MAP_v2_20261018.md`；`docs/plans/PES2TS_图论立体先验与单端协同延续优化方案_v2_20261002.md`；`PES2TS_扫描规划谱系合并方案_20260930.md` |
| 9 | `Active-Normative` | 3 | `docs/contracts/PES2TS_GeometryPolicyContract_v2.md`；`PES2TS_MoveProcessContract_v1.md`；`PES2TS_ReactionIntent_Contract_v2.md` |
| 10 | `只读报告` | 3 | `docs/reports/PES2TS_Direct_fit50训练表征分析_20261015.md` 等 3 份 |
| 11 | `IMPLEMENTED + EXECUTED` | 2 | `docs/agent_runs/handoffs/W-LOSS_E3_BVL_arms.md`；`W-LOSS_E3_trainer_and_runs.md` |
| 12 | `Active` | 2 | `docs/agent_runs/README.md`；`docs/plans/PES2TS_V5.3-MP_文档同步清单_v1_20261021.md` |
| 13 | `Proposed v1` | 2 | `docs/contracts/PES2TS_ReactionIntent_Contract_v1.md`；`docs/plans/PES2TS_V5.1-R2_损失尺度与数据完整性修复方案_v1_20261020.md` |
| 14 | `设计稿` | 2 | `docs/design/G1_v2_L0到G2扫描策略设计.md`；`docs/plans/V541_WP7_frozen_dev_eval_设计方案_v1.0_20261028.md` |
| 15 | `执行方案` | 2 | `docs/plans/PES2TS_ACP执行统一与旧代码清理方案_20261003.md`；`PES2TS_G2T真实战役执行方案_20261004.md` |
| 16 | `批次方案` | 2 | `docs/plans/PES2TS_G2_下一批实现方案_真引擎接通与验收_20261004.md`；`PES2TS_G2_下一批算法调整与测试方案_20261003.md` |
| 17 | `Proposed v3.2` | 2 | `docs/plans/PES2TS_V5-V7_反应意图条件化与低维生长方案_v3.0_20261020.md`；`PES2TS_V5.1-R3_反应意图条件化与端点标签不变重训方案_v3_20261020.md` |
| 18 | `Proposed v1.1` | 2 | `docs/plans/PES2TS_V5.1-R3_数据库特征损失三位一体修复方案_v1_20261020.md`；`PES2TS_V5.3-MP_大规模MOVE-PROCESS联合训练计划_v1_20261021.md` |
| 19 | `Frozen-Preregistered v1.0` | 2 | `docs/plans/PES2TS_V5.4.3_WP执行清单_v1.0_20261102.md`；`PES2TS_V5.4.3_双Decoder_50K_全面开发与训练方案_v1.0_20261102.md` |
| 20 | `阶段执行总结与问题复核` | 2 | `docs/reports/PES2TS_v4.1阶段训练_修复输入语义与最小实验矩阵_最终报告_20261017.md`；`PES2TS_v4阶段训练_最终报告与问题复核_20261017.md` |

### 4.3 归并桶（live 文档，17 类）

`其他 62` / `Proposed 18`（含已接受等近义）/ `Superseded 7` / `历史/Historical 7` / `权威 7` / `Accepted 6` / `Executed-Frozen 6` / `fixture_passed 5` / `参考 4` / `只读报告 4` / `Active-Normative 3` / `implemented 2` / `Active 2` / `Frozen-Preregistered 2` / `Review 1` / `Draft 1` / `Active-Proposed 1`。可见：**"其他"类占 live 文档近 4 成**，状态词表远未收敛；`Accepted`/`已接受`、`Superseded`/`SUPERSEDED`、`Active`/`Active-Normative` 等近义并存。

### 4.4 状态头书写格式变体（5 种）

| 格式 | 次数 | 示例 |
|---|---:|---|
| `> 状态：…`（普通引用行） | 160 | `docs/agent_runs/BOOTSTRAP_REVIEW.md:3` 等 |
| `> **状态**：…`（粗体只包词） | 2 | `docs/infra/PES2TS_仓库归档与清理方案_v1_20261120.md:3`；`docs/plans/PES2TS_训练测试集划分标准_v1_20261120.md:3` |
| `> **状态：…**`（冒号在粗体内） | 1 | `docs/agent_runs/handoffs/W-FEAT_E2_v3_stack.md:4` |
| `状态：…`（裸行，无引用符） | 1 | `docs/plans/V541_WP7_frozen_dev_eval_设计方案_v1.0_20261028.md:4` |
| 其他变体 | 1 | `docs/reports/PES2TS_V543_双Decoder_50K_实施与训练报告_20261106.md:5` |

### 4.5 无状态块文件（live 42 份，示例）

高影响缺口示例（完整 42 条见 `status_vocabulary_stats.txt` §D）：`docs/contracts/PES2TS_contracts_v1.md`、`docs/contracts/PES2TS_ACP字段映射_v1.md`、`docs/contracts/PES2TS_数据规范与ACP对接优化方案.md`、`docs/demo/PES2TS_demo24_反应挑选与复核方案.md`、`docs/design/G1_v2_补全实施总方案.md`、`docs/design/PES_GENERATION_图论扫描策略选择器设计_v1.md`、`docs/plans/PES2TS_文档修订台账.md`、`docs/reports/PES2TS_V52_任务纠偏审计_20261021.md`、`docs/reports/PES2TS_V53MP_S4_20k_vs_50k_配对统计与C臂导出_20261027.md` 等。

---

## 5. 现状缺口与冲突（逐条，均带文件:行号证据）

> 共 **12 项（C-01…C-12）**。判定标准：能由当前 HEAD 的文件内容 + 上述门禁实测复现；不依赖推测。

### C-01 文件名版本 ≠ 正文版本（4 例，历史命名保留 + 内容多次升级）

| 文件 | 文件名版本 | 正文/标题版本 | 证据 |
|---|---|---|---|
| 开发宪法 | `_v1` | 标题 v2.9；正文"内容 v2.9 @ 2026-10-21" | 文件名 `docs/PES2TS_开发宪法_v1.md`；正文 `:1`（`# PES2TS 开发宪法 v2.9`）、`:8`（"文件名保留 `_v1` 以兼容既有链接"） |
| 阶段定义 | `_G0-G3` | 标题 "G0–G4"；当前版本 v4.8 | `docs/plans/PES2TS_阶段定义_G0-G3.md:1`（`G0–G4`）、`:8`（v4.8 @ 2026-10-21）、`:9`（文件名保留历史名） |
| 整体开发与发布方案 | `_v2` | 标题 v3.5；当前版本 v3.5 | `docs/plans/PES2TS_整体开发与发布方案_v2.md:1,8` |
| 完整开发流方案 | `_v1` | 标题 v2.4；当前版本 v2.4 | `docs/plans/PES2TS_完整开发流方案_整合版_v1.md:1,8` |

> 任务要求核实的"宪法 v1/v2.9、阶段定义 G0-G3/v4.7、整体开发与发布方案 v2/v3.5"：**实际阶段定义已是 v4.8**（2026-10-21 ADR-0031 增量），v4.7 是索引滞后（见 C-03）。

### C-02 状态头版本与文件名/正文三方不一致（2 例）

- **V5–V7 方案**：文件名/标题 **v3.0**（`docs/plans/PES2TS_V5-V7_反应意图条件化与低维生长方案_v3.0_20261020.md:1`）、状态行 **Proposed v3.2**（`:9`）、顶部横幅 **v3.3**（`:3` "v3.3 MOVE–PROCESS 修订（2026-10-21）"）。
- **V5.1-R3 方案**：文件名/标题 **v3**（`..._v3_20261020.md:1`）、状态行 **Proposed v3.2**（`:7`）、横幅 v3.1/v3.2（`:3,:5`）。
- 同一文档内"文件名版本 / 状态版本 / 修订横幅版本"三处并存且互不一致，直接违反"状态头强制 + 当前版本"的单一表达意图（`AGENT.md:146`；台账 `:552-561`）。

### C-03 索引自述版本滞后（5 处；门禁 CLEAN 也检测不到）

| 位置 | 索引写的版本 | 实际版本 | 证据 |
|---|---|---|---|
| `docs/README.md:179` | 宪法"内容 v2.7" | v2.9 | 宪法 `:1,:8` |
| `docs/README.md:187` | 阶段定义"内容 v4.7" | v4.8 | 阶段定义 `:8` |
| `docs/README.md:238` | V5–V7"内容 v3.2（2026-10-21 任务纠偏修订）" | 顶部已是 v3.3 修订 | V5–V7 `:3` |
| `docs/README.md:95` | V5.1-R3"v3.1" | 状态行 v3.2 | V5.1-R3 `:7` |
| `AGENT.md:8` | "当前版本 v3.16 @ 2026-10-17（…宪法v2.7适用登记）" | 宪法 v2.9 @ 2026-10-21 | 宪法 `:1,:8` |

一致性对照：`docs/README.md:188`（整体发布 v3.5）与 `:215`（完整流 v2.4）与正文一致，说明滞后是逐条维护失配而非系统性规则问题。

### C-04 AGENT.md 执行投影未随宪法 v2.9 更新

`AGENT.md:8` 仍写"当前版本：v3.16 @ 2026-10-17（…宪法 v2.7 适用登记）"；宪法已是 v2.9（`docs/PES2TS_开发宪法_v1.md:1,8`），§0.3/§12.2 要求"重大不变量变更须同时更新 §3 与 AGENT.md 摘要、须立即修正"（`:33-38,417-418`）。当前无任何机器门禁能发现该漂移（§2.6）。

### C-05 状态词表未收敛（17 归并桶 vs 官方 5 值、台账 4 值）

- 实际 live 文档状态头 → **17 个归并桶**，"其他"占 62 份（`status_vocabulary_stats.txt` §B）。
- 官方图例 5 值：`docs/README.md:118`；台账 §4 模板 4 中文值：`docs/plans/PES2TS_文档修订台账.md:552-561`；实际大量使用 `Accepted/已接受/历史提案/前版记录/只读报告/Active-Normative/IMPLEMENTED + EXECUTED/设计稿/执行方案/批次方案` 等未登记取值。
- 中英文混用、同义词并存（`Accepted` 18 次 vs `已接受` 5 次；`SUPERSEDED` 与 `Superseded` 均被 exemption marker 覆盖，`config:13-22`）。

### C-06 状态头缺失（42/161 live 文档无任何状态块）

统计：全库 49/216 无状态块，其中 live 42/161（`status_vocabulary_stats.txt` §D）。这与宪法 §12.4"所有 `docs/**/*.md` 带状态头"（`docs/PES2TS_开发宪法_v1.md:428`）直接冲突；但 `docs_consistency.py` 只检查三类规则（`:18-27,140-207`），**不检查状态头**，故门禁 CLEAN。

### C-07 状态头书写格式 5 变体（脚本识别口径风险）

`> 状态：…` 160 份、`> **状态**：…` 2 份、`> **状态：…**` 1 份、裸 `状态：…` 1 份、其他 1 份（`status_vocabulary_stats.txt` §C）。格式差异使任何"状态头自动解析/自动校验"在落地前必须先冻结格式；当前 §1 的模板（台账 `:552-561`）只给字段不给书写格式。

### C-08 双索引覆盖缺口（40 份既不在 README 也不在台账；README-only 40、台账-only 7）

统计（`index_coverage_stats.txt`）：live 文档 180 份；`docs/README.md` 索引 133、台账索引 100、两处都有 93、**两处都没有 40**（reports 19、agent_runs 12、plans 4、superseded 4、infra 1），另 README-only 40、台账-only 7。
- 索引覆盖风险实例：`docs/reports/PES2TS_V542_Backbone_50K_实施与训练报告_20261031.md` 仅在 `docs/README.md` 被索引（台账缺口）；`docs/reports/PES2TS_V543_双Decoder_50K_实施与训练报告_20261106.md` 仅在台账被索引（README 缺口）——关键执行报告的"双索引"依赖单侧维护，任一索引漏更新即出现覆盖缺口。
- 台账 §2 权威文档表自述"2026-10-08 同步"（`docs/plans/PES2TS_文档修订台账.md:427`），其内容已与现状冲突：`PES-N 未来开发计划 v2` 在台账 §2 仍表述为"当前未来路线"（`:434`），而 `docs/README.md:189` 标注为"历史（已被 V5–V7 v2.0 取代）"、`:244` 标注 v2.0 "Superseded by v3.0"。
- 台账 §6 执行状态仍有 5 项未完成（`docs/plans/PES2TS_文档修订台账.md:571-587`）：计数与标定台账（:579）、reports 顶部"当前阶段指针"批量（:581）、**提交 CI 文档权威 lint**（:585）、阶段定义重命名 `_G0-G3`→`_G0-G4`（:586）、多坐标学习方案归位（:587）。

### C-09 门禁实现滞后于宪法 §12.4（"全绿"≠"合规"）

- 宪法要求 CI 校验：AGENT↔§3 一致、所有 docs 带状态头、每主题恰一份 AUTHORITATIVE、已取代文档顶部指针/不进索引（`docs/PES2TS_开发宪法_v1.md:426-428`；`AGENT.md:204` 复述）。
- 实际 CI 只有 4 步：G–V map check、Docs link check、Registry tests、两档 pytest（`.github/workflows/ci.yml:20-37,52-53`），**无任何内容一致性/状态头/单权威校验**；`docs_consistency --strict` 也不在 CI（仅在训练前人工强制，`AGENT.md:159,163`）。
- 后果实证：C-01~C-03、C-05~C-08 全部在 `docs_consistency --strict` = CLEAN、CI 4 步全绿的当前基线下并存（`baseline_gate_outputs.txt`）。

### C-10 `g_v_map` 路径校验分级导致的"看起来钉死、实际不校验"盲区

- `validate_gv_map` 只对 `active/executed` 战役强制 `config/data_manifest/evidence/comparison_only` 存在（`pes2ts_core/gv_registry.py:278-302`）；`proposed/archived` 战役的 evidence（如 V5.3-MP 计划、V5.4.3 ADR-0034、V5.1-R3 计划）**缺失不会 fail**。
- `stage_doc` 完全不参与存在性校验（`gv_registry.py:255-276`），仅参与渲染 → 阶段定义文件移动不会让 `--check` 失败，但会断掉 AGENT/宪法/README 的链接（转由门禁 2 捕获）。
- V5.4.1/V5.4.2 的 `data_manifest` 为仓库外 WSL 路径 `/pes2ts_data/v53mp_50k_data/split_manifest_v53mp.json`（`g_v_map.yaml:394,416`），按 `_is_external_path` 只作 provenance、不强制存在（`gv_registry.py:218-236,296-300`；台账 `:24-27` 记录该 CI 红灯修复）。

### C-11 `check_docs_links` 的 172 warning 是"容忍债务"，非 0 债务

- 101 条 `E:/` 绝对链接 + 71 条 local-only 链接，涉及 31 份文档（`baseline_gate_outputs.txt`；实现 `check_docs_links.py:54-70`）。这些链接在 GitHub/异机不可解析，基线允许其存在；但 `docs/reports/**` 只读（`AGENT.md:148`），结构上不可清零 → 只能冻结为容忍基线：**任何新增 warning 须逐条说明，不得静默增长**。

### C-12 归档/清理执行后遗留的双登记负担

`docs/superseded/` 已迁入 4 份（`docs/superseded/README.md:26-33`；台账 `:31-39`），但索引覆盖显示 superseded 5 份中 4 份既不在 README 也不在台账（`index_coverage_stats.txt`），仅由 superseded README 自身索引；而 `docs/README.md:6-9` 的"根目录 7 份 memo 不移动、只留索引"计划与 `docs/infra/PES2TS_仓库归档与清理方案_v1_20261120.md:166-167` 的 A/B/C 档清单尚未在 `docs/README.md` 形成逐条索引表 → 后续任何移动/改名都需**同时**维护 superseded 索引、台账、README、门禁配置四处登记。

---

## 6. "新方案禁止破坏项"清单（供方案设计直接引用）

> 以下 D-xx 为**硬约束**；任何新建档方案/文档重构方案必须在设计阶段逐条声明"不违反"，并在实施后跑 §6 末尾 4 门复验。

### 6.1 路径与文件（移动/删除/改名）

| 编号 | 禁止项 | 依据 |
|---|---|---|
| D-01 | **禁止**移动/重命名/删除 17 条 `historical`（含 5 份根 memo；清单见 §3.1） | `docs_consistency_config.json:27-45`；`docs_consistency.py:167-186` |
| D-02 | **禁止**移动/重命名/删除 27 条 `required_files`（14 docs + 13 config/模板/validator；清单见 §3.2） | `docs_consistency_config.json:57-419`；`docs_consistency.py:189-207` |
| D-03 | **禁止**删除/移动/手改 `docs/design/PES2TS_G_V_MAPPING_v1_20261018.md`；只能 `python scripts/render_gv_map.py --write` 重生成 | `render_gv_map.py:29,119-137`；`ci.yml:20-21` |
| D-04 | **禁止**移动/改名 executed/active 战役的 evidence 文档（V4.2–V4.7 实现报告、30h 计划、`V53_ROLLOUT_REPORT.md`、V541 报告、V542 报告、ADR-0026、V5.1 修复冲刺计划；§3.5(b)） | `gv_registry.py:278-302`；`g_v_map.yaml:79-471` |
| D-05 | **禁止**在未同步修改 `config/registry/g_v_map.yaml` + 重渲染的情况下移动/改名 `docs/plans/PES2TS_阶段定义_G0-G3.md`（stage_doc）；重命名阶段定义须走台账 §6 未完成项（`:586`）的独立受审改动 | `g_v_map.yaml:12-60`；`render_gv_map.py:42-101` |
| D-06 | **禁止**把 `docs/archive/` 当"过时文档墓地"或搬入/移出文件；新增快照只能新建 `YYYYMMDD_pre_<主题>/` 并带 manifest/哈希 | `docs/archive/README.md:3,15-20` |
| D-07 | **禁止**修改 `docs/reports/**` 正文；只能新增指针或新报告 | `AGENT.md:148`；台账 `:422,544-548` |
| D-08 | **禁止**把 `docs/reports/**`、`docs/archive/**`、`historical` 名单文档迁入 `docs/superseded/`（移动会断门禁） | `docs/superseded/README.md:17-19` |
| D-09 | **禁止**删除/重命名根 `README.md`、`AGENT.md`、`V53_ROLLOUT_REPORT.md`、5 份历史 memo；`R3_RESULT_SUMMARY.md` 若移动须同步门禁配置（当前为软登记） | §3.3；`config:46-56` |
| D-10 | **禁止**把文档写进 `.gitignore` 排除路径（`data/*`、`outputs/*`、`/.slim/`、`/.omo/`、缓存等）并宣称"已入库" | `.gitignore:1-64` |

### 6.2 内容、状态头与索引

| 编号 | 禁止项 | 依据 |
|---|---|---|
| D-11 | **禁止**在活跃文档使用 6 个禁用语且不带同行豁免标记；新增文档不得让 `docs_consistency --strict` 出现任何 violation | `config:2-9,10-23`；`docs_consistency.py:140-164` |
| D-12 | **禁止**新增/移动文档后让 `check_docs_links` 的 error 从 0 变 >0（相对链接必须可解析；**不要**新增 `E:/`/`outputs/` 的 local-only 链接——它们只加 warning 债务） | `check_docs_links.py:54-86`；基线 0 error/172 warning |
| D-13 | **禁止**让新增 plan/design 缺状态头；新文档必须带 §1 G-10 的 6 字段状态块，且状态取值落在受控词表内（不得自创未登记取值） | `AGENT.md:146`；台账 `:552-561`；宪法 `:428` |
| D-14 | **禁止**在状态头/索引中写错误版本：文件名版本、状态行版本、正文当前版本、索引登记版本必须四方一致（本轮 C-01/C-02/C-03 即为反例） | `AGENT.md:146`；宪法 `:417-418` |
| D-15 | **禁止**新建 "plan/design" 却不更新 `docs/README.md` + 修订台账（一处都不能漏）；涉及 V 战役还须更新 `g_v_map.yaml` 并重渲染 | `AGENT.md:147`；`AGENT.md:112`；`render_gv_map.py:95-99` |
| D-16 | **禁止**通过改名/重写来"修复"版本不一致：文件名历史名（宪法 `_v1`、阶段定义 `_G0-G3`、整体发布 `_v2`、完整流 `_v1`）**保留**，版本差异在状态头/正文表达并在索引中同步 | 宪法 `:8`；阶段定义 `:9`；整体发布 `:5`；完整流 `:5`；`AGENT.md:6` |
| D-17 | **禁止**新增"双索引"外的新索引而不登记既有两处；新文档必须能回答"README 行号 + 台账行号"两个位置 | §5 C-08；`AGENT.md:147` |
| D-18 | **禁止**把 G/V 轴混淆：新计划须声明 `G#` 或 `C/R/X`；V 战役登记进 `g_v_map.yaml`；M0–M5 不出现在计划状态头 | 宪法 `:366-367,383-384`；`AGENT.md:49-50` |

### 6.3 门禁与治理机制本身

| 编号 | 禁止项 | 依据 |
|---|---|---|
| D-19 | **禁止**修改门禁配置/脚本/CI 来"让红灯变绿"（`docs_consistency_config.json` 的 6 个列表、`check_docs_links.py`、`render_gv_map.py`、`gv_registry.py`、`ci.yml`）；如需变更须单独受审并与本文审计基线对比 | G-30；§3.0 本轮零改动 |
| D-20 | **禁止**触碰 R1–R6 红线；P1–P5 的例外只能走 §12.3 时限豁免（ADR 记录理由/范围/到期/回滚） | 宪法 `:140-166,420-424`；`AGENT.md:24-35,127-128` |
| D-21 | **禁止**把"写入计划/新增文档"当作能力升级；能力状态只用四层状态词，且不得把 `fixture_passed` 写成 `smoke_passed` | 宪法 `:402-403`；`AGENT.md:124-125` |
| D-22 | **禁止**在未跑 4 门的情况下宣称文档治理通过；交付前必须复跑并保持基线不退化（4 门全 exit 0） | §2.2 基线；`AGENT.md:159,163` |

### 6.4 实施后复验命令（4 门，缺一不可）

```powershell
python scripts/agent_ops/docs_consistency.py --strict     # 期望：scanned 158+; 0 violation(s); CLEAN
python scripts/check_docs_links.py                        # 期望：0 error(s)；warning 不高于基线 172
python scripts/render_gv_map.py --check                   # 期望：G–V mapping doc is current
python -m pytest -q tests/test_gv_registry.py tests/test_config_env.py   # 期望：26 passed
```

> 基线数字随新增文档会变化：`scanned`/`checked` 数量允许增加；**violation/error 必须为 0**；warning 若增加须在方案中逐条说明（不允许静默增长）。

---

## 附录 A：本审计的复现方式

1. 门禁实测：`python .slim/deepwork/20261121_docaudit/rules/extract_rules.py --section gates`
2. 钉死路径：`--section pins`（产物：`pinned_paths.txt`，43 条 g_v_map 引用 + 17/27/9 逐条）
3. 状态词表：`--section status`（产物：`status_vocabulary_stats.txt`）
4. 双索引覆盖：`--section index`（产物：`index_coverage_stats.txt`）
5. 版本一致性：`--section versions`
6. 全部一次跑：`python .slim/deepwork/20261121_docaudit/rules/extract_rules.py`（幂等，重新生成全部产物；门禁原始输出写入 `baseline_gate_outputs.txt`）

## 附录 B：关键事实速查

| 事实 | 数值 |
|---|---|
| `docs_consistency --strict` | scanned **158**；**0** violation；CLEAN；exit 0 |
| `check_docs_links` | checked **182** docs；**0** error；**172** warning（101 E:/ + 71 local-only；31 份文档） |
| `render_gv_map --check` | current；exit 0 |
| registry tests | **26 passed**；exit 0 |
| historical / required_files / root_scope | **17 / 27 / 9**（另 token_exempt 1、forbidden 6、markers 12、banner 1） |
| g_v_map docs 引用 | **43 条**（stage_doc 13 引用/1 路径；evidence 26 引用/15 路径；文本 4 引用/3 路径；code_scope 内 docs 路径 **0**） |
| CI 直接引用 docs 路径 | **0**（间接 4 通道：render 目标、链接扫描面、registry 测试断言、全仓 pytest） |
| 硬钉死路径合计 | **52 条** = 31 docs + 8 根 md + 13 config/模板/validator；另 1 条软登记（`R3_RESULT_SUMMARY.md`，移动前须同步配置） |
| 状态块覆盖 | 216 份中 **167 有 / 49 无**（live 42 无）；前 40 行内 165 份 |
| 状态取值 | live 文档 **17 个归并桶**；Top 值 `Accepted 18`、`Proposed 9`、`历史提案 6`、`已接受 5`、`权威 5`、`Executed-Frozen 5` |
| 索引覆盖 | live 180 份；README 133 / 台账 100 / 双有 93 / **双无 40**；README-only 40 / 台账-only 7 |
| 关键缺口/冲突 | **12 项（C-01…C-12）**；禁止破坏项 **22 条（D-01…D-22）** |
