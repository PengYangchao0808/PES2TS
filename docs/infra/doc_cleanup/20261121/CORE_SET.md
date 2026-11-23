# CORE_SET — PES2TS 核心功能文档集（30–50 份受管单元）

> 任务：task-8；生成日期：2026-11-21；只读仓库（本文件为设计交付，未改任何既有文件）。
> 上位方案：`docs/infra/PES2TS_开发档案树与文档清理方案_v1_20261121.md` §2.2/§4.2（占位符 `FILL:CORE_SET` / `FILL:CORE_SET_OWNERS` 的填充底稿）；
> 兼容基线：`docs/infra/PES2TS_开发文档规范化归档方案_v2_20261121.md`（§5 状态机、§6 元数据头、§10 三件套、§11 SOP）、`AGENT.md` §9、
> `docs/plans/PES2TS_文档修订台账.md` §1/§4、`docs/README.md` 结构、根 `README.md` 首屏。
> 数据来源：`docs/infra/doc_audit/20261121/classification_mapping.csv`、`DISPOSITION.csv`、`RECOMMENDATIONS.md`（253 条口径）。
> 边界：本清单只回答"清理后长期维护哪些文档、谁维护、何时维护"；不裁决删除范围（见 task-7）、不定义档案树分期（见 task-6）。

---

## 0. 口径与计数（先读这一节）

| 项 | 数值 | 说明 |
|---|---:|---|
| 受管单元 | **44** | 一个"受管单元"= 1 份文档，或 1 个必须整体维护的文档集合（ADR 集、模型卡集、分期条目集） |
| 覆盖文件 | **≈98** | 44 单元展开：ADR 34 份、模型卡 8 份、档案树 7 期 17 条目 + 横切层/索引/证据层、`docs/archive`+`superseded` 索引与门禁证据 |
| owner 角色 | **4** | 仓库所有者 / 科学负责人 / 工程负责人 / Agent 会话；**角色可兼任，但每单元必须有且仅有一个 owner 角色** |
| 审查频率档 | **5** | 双周 / 里程碑 / 季度 / 事件驱动（当日）+ 双周抽检 / 只读（季度巡检） |
| 与目标关系 | 一致 | ADR 全留裁决后：`standard` 档清完保留 137 份（含 34 ADR + 8 模型卡 + 8 archive 索引）；核心集 44 单元 ≈98 份 + 关键终报 ≤6 + 门禁证据，其余由档案树 32 份承接 |

**四条纳入原则**（为什么是这些、不是全部）：
1. **被 5 个以上现行文档引用，或本身就是规范/接口/索引** → 必须纳入（删了四处断链，改错全局失真）。
2. **门禁钉死**（`docs_consistency_config.json` 的 `historical`/`required_files`/`root_scope` 或 `g_v_map.yaml` 引用）→ 必须纳入；移动/改名前先改配置（见 v2 方案 §2.3）。
3. **档案树的入口与生成物** → 必须纳入，否则"压缩后怎么读"没有答案。
4. **纯过程记录、单轮报告、已被取代且结论已入档案树者** → **不纳入**（由 `docs/archive_tree/` 条目 + git tag 承接，见 task-6/task-7）。

**不纳入本集但必须长期只读保留的锚点**（不计入 47 单元，避免膨胀）：
- 关键终报 ≤6 份（V5.4.3、V5.3-MP S3、V5.2-R 收尾等，按 task-7 的 `KEEP_REPORT` 清单定稿）：状态 `Evidence-ReadOnly`，owner = Agent 会话，只做季度可用性巡检，**永不改内容**。
- `docs/agent_runs/state/gates/**` 门禁 JSON、`docs/archive/**/manifest|verification|README`：永不删（方案 §0 红线），不设审查频率。

---

## 1. 治理根（11 个受管单元）

> 定义：L0 治理层——规范根、执行投影、阶段/流程总纲、索引与流水、治理方案、机器注册投影。**这批文档出错 = 全库失真**，因此 owner 一律为仓库所有者（索引/生成物可授 Agent 会话）。

| ID | 路径 | 职责（一句话） | owner 角色 | 更新触发条件 | 审查频率 | 状态头要求 |
|---|---|---|---|---|---|---|
| G1 | `docs/PES2TS_开发宪法_v1.md` | 规范根（SSOT）：身份、红线 R1–R6、政策 P1–P5、接口注册、修订程序 | 仓库所有者 | 红线/政策/接口注册表/修订程序变更（须走宪法 §12 修订程序） | 双周抽检 + 每里程碑全读 | `Canonical`；冲突时以本文件为准 |
| G2 | `AGENT.md` | 宪法执行投影：操作性规则、架构变更审查协议、文档治理 §9 | 仓库所有者 | 宪法修订后同步；§7.2/§9 操作性规则变更（Phase 0 将修订 §9.4） | 双周（与 G1 同批） | `Canonical`（执行投影，冲突以宪法为准） |
| G3 | `docs/plans/PES2TS_阶段定义_G0-G3.md` | 阶段与边界唯一来源（G0–G4、C/R/X） | 仓库所有者 + 科学负责人 | 新增/合并阶段、阶段边界变更（须先有 ADR） | 里程碑（每阶段转换必读） | `Canonical`；文件名历史名保留（改名另立受审改动） |
| G4 | `docs/plans/PES2TS_完整开发流方案_整合版_v1.md` | G0–G4 阶段流、双轨/三工作流、依赖图、里程碑与文档地图 | 科学负责人 | 阶段流/里程碑结构变化 | 里程碑 | `Canonical` |
| G5 | `docs/plans/PES2TS_文档修订台账.md` | 追加型流水：每份文档"何时、因何、变成什么状态" | Agent 会话（执行）/ 仓库所有者（复核） | **每一次文档新增/状态转换/退役/档案条目新增的当日**（含 Phase 0 新增"退役"动作类型） | 双周抽检最近 20 条 | `Canonical`；**只追加，不改写已发生记录** |
| G6 | `docs/README.md` | 人工导航投影：读者从哪读起、哪些是权威（Phase 4 收敛为 ≤60 行 4 入口短索引） | Agent 会话 | 每次新增/改状态/退役文档的**同一 PR**；Phase 4 重写 | 双周 | `Canonical`；与 G5、`DOC_MANIFEST.json` 三件套一致 |
| G7 | `docs/infra/PES2TS_开发文档规范化归档方案_v2_20261121.md` | 分类法 L0–L4、11 状态封闭词表、元数据头 v2、归档判定、G5 门禁规格 | 仓库所有者 | 状态词表/目录规范/门禁规格变更（须修订本方案版本） | 里程碑 + 季度 | `Active-Proposed` → 签认后 `Canonical`；8 字段元数据头 |
| G8 | `docs/infra/PES2TS_开发档案树与文档清理方案_v1_20261121.md` | 退役（Retire）政策、档案树结构、Phase 0–5 执行与回滚 | 仓库所有者 | 退役红线/清理强度/档案树结构变更 | 里程碑（Phase 0 前后必读） | `Active-Proposed` → Phase 0 签认后 `Canonical`；8 字段头 |
| G9 | `docs/design/PES2TS_G_V_MAPPING_v1_20261018.md`（生成物；源 `config/registry/g_v_map.yaml`） | 战役注册投影：G（能力）× V（战役）状态、evidence、gate_note | Agent 会话（生成）/ 工程负责人（配置源） | `g_v_map.yaml` 任何变更后**重渲染**（`python infra/governance/render_gv_map.py --write`） | 事件驱动 + 双周核对 `--check` | `Canonical`（生成物，**禁止手改**）；`map_version`/`plan_revision` 必须可见 |
| G10 | `docs/agent_runs/FAQ.md`（新建，FAQ 落点） | 运营问答库：可复用的操作口径/排错知识（**非规范、不参与状态校验**） | Agent 会话 | 双周同步会新增条目；答案被规范取代时改为指针 | 双周 | L4 豁免状态头；每条 FAQ 带编号/依据/失效条件/复审日（见 `DISCUSSION_TEMPLATE.md`） |
| G11 | `docs/infra/doc_cleanup/20261121/RETIRE_LOG.md`（待建，Phase 4 由 `MIGRATION_LOG.md` 改名/升级） | 退役流水：日期/批次/路径/档案条目/恢复 tag | Agent 会话 | 每批删除完成后当日追加 | 里程碑 + 季度对账 | `Evidence-ReadOnly`（追加式）；不得回改历史行 |

---

## 2. 现行合同与接口（6 个受管单元）

> **Lead 最终裁决（2026-11-21）**：C6 `PES2TS_G2T_战役合同字段字典_v1.md`、C7 `PES2TS_PESN_IRC教师数据与模型包合同_v1.md`、D2 `PES2TS_PES-C到PES-G架构与迁移方案_v1_20261004.md` 不纳入核心集——G2.T 已收口、G3.P 草案被 V5 系列承接、PES-C→G 被 PES-N 承接；三份先抽结论入档案树 E04/E05/E03，再按 aggressive 档退役。
>
> 判定标准：**当前仍作为执行依据**的合同/接口文档。同一合同族同一时刻只允许 1 份 `Active-Normative`（v2 方案 §5.1）。
> **旧版不进核心集**：`PES2TS_FullPathNavigationContract_v1.md`、`PES2TS_GeometryPolicyContract_v1.md`、`PES2TS_ReactionIntent_Contract_v1.md`、`PES2TS_双端TS教师与相容转移合同_v2.md`——已由取代者承接，按 task-7 处置（标 `Superseded` 横幅或退役），结论进档案树第 11/13 期。

| ID | 路径 | 职责（一句话） | owner 角色 | 更新触发条件 | 审查频率 | 状态头要求 |
|---|---|---|---|---|---|---|
| C1 | `docs/contracts/PES2TS_MoveProcessContract_v1.md` | **唯一任务合同**：MOVE/PROCESS/No-Move/TS 语义、输入边界、禁读清单 | 科学负责人 | 任务语义/标签定义/输入输出边界变更 → 先 ADR，再发 `v2` 且旧版转 `Superseded` | 双周（每战役前必读） | `Active-Normative`；`required_files` 钉死，**只发新版本不改语义** |
| C2 | `docs/contracts/PES2TS_ReactionIntent_Contract_v2.md` | 反应意图输入合同：有向意图、signed graph edit、Relabel invariant / Reverse sensitive | 科学负责人 | 意图表示/CV-Jacobian lifting/对称性条款变化 | 双周 | `Active-Normative`；必须写明取代 v1 的条款范围 |
| C3 | `docs/contracts/PES2TS_GeometryPolicyContract_v2.md` | 几何策略合同：位移/参考架/双分母与封存边界条款 | 科学负责人 | 几何可观测性、参考架 fallback、评估分母变化 | 双周 | `Active-Normative`；v1 遗留条款须显式标注"未继承" |
| C4 | `docs/contracts/PES2TS_contracts_v1.md` | 数据/接口契约总表：schema 名称、字段身份、版本演进入口 | 工程负责人 | 新 schema 注册、字段级变更（须与 AGENT §7.2 联动） | 双周 | 现行规范（`Active-Normative` 或明确"参考字典+生效条款指针"） |
| C5 | `docs/contracts/PES2TS_ACP字段映射_v1.md` | ACP 请求/响应/收据字段映射（外部执行后端唯一入口） | 工程负责人 | ACP 接口、字段名/单位/失败码变化 | 事件驱动 + 双周 | `Active-Normative`（或注明"映射说明，随 ACP 版本"） |
| C8 | `docs/contracts/PES2TS_数据规范与ACP对接优化方案.md` | 数据规范与 ACP 对接的规范性说明（表结构/真值隔离/对接流程） | 工程负责人 | 数据表结构、真值隔离流程、ACP 对接方式变化 | 季度 | 现行说明性规范；与 C4/C5 交叉引用而不复制 |

---

## 3. 现行设计、计划与 ADR（12 个受管单元：4 设计 + 7 计划 + 1 ADR 集）

> 判定标准：**当前执行或最近一个已执行战役仍然会查**的设计/计划；全部随战役状态在 `Active-Proposed` / `Frozen-Preregistered` / `Executed-Frozen` 间迁移（v2 方案 §5.2）。
> ADR 的取舍见 §3.3。

### 3.1 现行设计（4）

| ID | 路径 | 职责（一句话） | owner 角色 | 更新触发条件 | 审查频率 | 状态头要求 |
|---|---|---|---|---|---|---|
| D1 | `docs/design/PES2TS_CODE_MAP_v2_20261018.md` | 代码地图：模块目录、入口、责任边界（新会话第一个读的代码导航） | 工程负责人 | 新增/迁移模块、入口命令变化 | 事件驱动 + 双周 | 现行；模块清单必须与仓库实况一致 |
| D3 | `docs/design/PES2TS_三模块与可视化产品架构_v1_20261005.md` | 三模块产品架构（控制/执行/可视化）与责任划分 | 工程负责人 | 产品边界、模块职责、可视化接口变化 | 里程碑 | `Active-Proposed`；schema/UI 能力须声明"未验收" |
| D4 | `docs/design/PES2TS_预测器产品与G3代码模块划分_v1_20261007.md` | 预测器产品与 G3 代码的 B0–B7 模块划分/接口/验收 | 工程负责人 | G3 模块拆分、产品归属变化 | 里程碑 | `Active-Proposed`；随 ADR-0014 同步 |
| D5 | `docs/design/PES2TS_Classic图论驱动与xTB原生约束扫描设计_v1_20261006.md` | Classic 基线算法设计（图论驱动 + xTB 原生约束扫描） | 科学负责人 | Classic 默认分支、扫描策略变化 | 季度 | `Active-Proposed`；生产默认门状态须写明 |

### 3.2 现行执行计划（7）

| ID | 路径 | 职责（一句话） | owner 角色 | 更新触发条件 | 审查频率 | 状态头要求 |
|---|---|---|---|---|---|---|
| P1 | `docs/plans/PES2TS_V5-V7_反应意图条件化与低维生长方案_v3.0_20261020.md` | 当前阶段研究计划：V5/V6/V7 三阶段及其验收门 | 科学负责人 | 研究问题/阶段划分/验收门变化（先 ADR） | 双周 | `Active-Proposed`；版本号在元数据头，不进状态 |
| P2 | `docs/plans/PES2TS_V5.3-MP_大规模MOVE-PROCESS联合训练计划_v1_20261021.md` | V5.3-MP 战役计划：S0→S4 数据/训练/授权链 | 科学负责人 | 战役收口（状态转 `Executed-Frozen`）、S4 决策 | 双周（含 P0 裁决 Q2 跟踪） | `Proposed` → `Executed-Frozen`（P0 状态矛盾，`required_files` 钉死，**只改状态不移动**） |
| P3 | `docs/plans/PES2TS_V5.4.2_Backbone_50K_详细阶段实施方案_v1.0_20261029.md` | V5.4.2 Backbone-50K 阶段方案（A0–A5、预注册与门） | 科学负责人 | 战役收口（状态转 `Executed-Frozen`） | 双周（含 P0 裁决 Q1 跟踪） | `Active-Proposed` → `Executed-Frozen`（P0 状态矛盾，`required_files`） |
| P4 | `docs/plans/PES2TS_V5.4.3_双Decoder_50K_全面开发与训练方案_v1.0_20261102.md` | V5.4.3 双 Decoder 50K 方案（M1–M5/P1–P5、18-run、H0–H3） | 科学负责人 | 执行完成 → `Executed-Frozen` + §执行指针（2026-11-06 已执行，状态待裁决） | 双周 | `Frozen-Preregistered` → `Executed-Frozen`（`required_files`）；Σ 正文不改，加执行指针 |
| P5 | `docs/plans/PES2TS_V5.4.3_WP执行清单_v1.0_20261102.md` | V5.4.3 WP0–WP9 验收与 18-run 追踪清单 | 科学负责人 | 同 P4（同步转态） | 双周 | 同 P4（`required_files`） |
| P6 | `docs/plans/PES2TS_整体开发与发布方案_v2.md` | 项目级开发与发布主方案（发布门、交付物、外部可见性） | 仓库所有者 | 发布策略、开源范围、交付物变化 | 里程碑 | `Canonical`（项目级主方案）或按 v2 方案降为 `Active-Proposed` 并说明 |
| P7 | `docs/plans/PES2TS_训练测试集划分标准_v1_20261120.md` | 训练/测试集分层继承链与数据来源（划分、manifest、封存纪律） | 科学负责人 + 工程负责人 | 划分规则、manifest 版本、封存纪律变化 | 季度（数据发布前必读） | 说明性汇总（非规范）；升格规范须走 ADR | 

### 3.3 ADR 集（1 个受管单元 = `docs/design/decisions/` 全部 34 份）

| ID | 路径 | 职责（一句话） | owner 角色 | 更新触发条件 | 审查频率 | 状态头要求 |
|---|---|---|---|---|---|---|
| A1 | `docs/design/decisions/`（ADR-0001 … ADR-0034） | 架构/接口/科学路线的**决策留痕**：决策内容 + 依据 + 被放弃的替代方案 | 科学负责人 | 新裁决 → 新增 `ADR-NNNN-<标题>.md`（编号唯一连续）；状态迁移（Proposed→Accepted→Superseded） | 活跃 6 份双周；其余每季度抽 1/3（约 10 份） | `Accepted`（裁决生效）/`Active-Proposed`（提案）/`Superseded`（须写"被 X 取代"）；**ADR 不改写结论**，语义变化发新 ADR |

**ADR 是否全留：建议全留（34 份全保留在 `docs/design/decisions/`）。** 与上位方案裁决 C3-A 一致。

理由（成本 vs 价值逐项）：
1. **成本极低**：ADR 是只读决策记录，**没有"维护正文"的义务**——维护成本只有状态头正确性与编号连续性（34 份 ≤1 个受管单元，审查摊薄到双周/季度）。全留不增加活跃面维护负担，只增加"存在感"。
2. **价值不可替代**：ADR 记录"为什么否掉别的方案"，是**负结果与决策追溯的唯一载体**；档案树条目 §3 要求"关联 ADR"，砍掉 ADR 会使档案树失去可追溯的决策链。
3. **门禁硬约束**：`ADR-0030`–`ADR-0034` 在 `docs_consistency_config.json.required_files` 名单内，删除/移动必红（v2 方案 §2.3）；`historical` 名单含 ADR-0021/0028/0029，同样不能动。
4. **编号唯一性不可合并**：ADR 的价值在"编号 = 决策指纹"，按主题合并等于发明新编号，违背"不引入第二套身份"（v2 方案 §4.3）；合并还会让现行文档里 100+ 处 `ADR-00xx` 引用全部失锚。
5. **"只留最近 20"不成立**：早期 ADR 仍被现行文档引用为**当前口径来源**——ADR-0010（PES-N 方向）、ADR-0013（G3.P 并行/版本体系）、ADR-0014（产品归属）、ADR-0017（统一残差与 G2 回退）、ADR-0025（G–V 映射）等；删掉它们等于把"现行依据"变成死引用。

**代价控制（让"全留"不变成"全读"）**：
- `docs/README.md` 只列"活跃 ADR 6 份"（见下表）+ 一行"全部 34 份见 `docs/design/decisions/`，按编号检索"，不逐条铺开；
- 档案树 `_INDEX.md` 每期列出该期关联 ADR 编号，读者从档案树反向定位；
- 季度抽审 1/3 只查状态头/取代关系/编号连续性，不重读全文。

**活跃 ADR 子集（双周逐份过状态）**：

| ADR | 主题 | 当前口径 | 双周检查点 |
|---|---|---|---|
| `ADR-0025-G-V双轴映射与战役注册.md` | G–V 双轴映射与战役注册 | G–V 注册制度的依据 | `g_v_map.yaml` 与 G9 文档是否一致 |
| `ADR-0030-V5.2全路径导航与端点停止.md` | 全路径导航与端点停止 | 旧 V5.2 语义已关闭（`required_files` 钉死） | 是否被误当现语义引用；与 C1 的取代关系是否写明 |
| `ADR-0031-MOVE-PROCESS统一学习合同.md` | MOVE–PROCESS 统一学习 | **当前任务合同依据**（`required_files`） | 与 C1 条款是否仍一一对应 |
| `ADR-0032-V5.4.1-Embedding接口扩展与预注册治理.md` | V5.4.1 Embedding 接口/预注册 | 已执行（§6 执行附录），`required_files` | 执行附录与 V5.4.1 报告链接是否有效 |
| `ADR-0033-V5.4.2-Backbone接口与预注册治理.md` | V5.4.2 Backbone 接口/预注册 | 已执行（`no_backbone_winner`），`required_files` | 状态与报告、registry 是否一致 |
| `ADR-0034-V5.4.3-双Decoder与DecoderContextV1.md` | V5.4.3 双 Decoder/DecoderContextV1 | 头标 `Proposed` 但 18-run 已于 2026-11-06 执行完 → 按裁决 Q4-**A** 保留冻结原文，用台账/README 指针表达执行事实 | 台账 2026-11-06 条目与 `docs/README` 是否已补；不改 ADR 正文 |

---

## 4. 档案树入口（7 个受管单元）

> 依赖：task-6（`archive-tree-architect`）的最终文件清单与命名。若其产出与本表命名不同，**以 task-6 为准，本表路径同步替换**，其余字段不变。
> 档案树 = 压缩后"怎么读项目史"的答案；**只追加、定稿条目 Frozen**（条目内 `> 档案状态：Frozen`）。

| ID | 路径 | 职责（一句话） | owner 角色 | 更新触发条件 | 审查频率 | 状态头要求 |
|---|---|---|---|---|---|---|
| T1 | `docs/archive_tree/README.md` | 档案树总纲：怎么读、分期索引、如何新增一期、与 `docs/archive/` 的区别 | 仓库所有者 + Agent 会话 | 一期定稿、档案规则变更 | 里程碑 + 季度 | 治理入口（`Canonical` 或 `Active-Proposed` 至签认） |
| T2 | `docs/archive_tree/_INDEX.md` | 全期索引表：期号/时间/主题/一句话结论/关联 ADR/条目文件 | Agent 会话 | 新期条目或结论变更 | 事件驱动 + 双周 | Frozen（追加式） |
| T3 | `docs/archive_tree/_TEMPLATE.md` | 档案条目统一模板（§1 目标 … §8 本期文档处置） | 仓库所有者 | 模板字段调整（需同步已定稿条目另议） | 季度 | 治理样板；`Active-Proposed` |
| T4 | `docs/archive_tree/CHANGELOG.md` | 档案树自身变更流水（追加式） | Agent 会话 | 每次档案树变更当日 | 双周 | Frozen（追加式）；`Evidence-ReadOnly` |
| T5 | `docs/archive_tree/00_global.md` | 项目全局条目：身份、目标、最终形态、技术栈、当前状态 | 科学负责人（Agent 起草） | 项目定位/最终形态变化（极少） | 季度 | 档案状态 `Frozen`；只追加 |
| T6 | `docs/archive_tree/01_…_14_…`（14 期分期条目，1 个受管单元） | 每期核心思路/技术决策/量化成果/**负结果**/后续影响/证据指针 | 科学负责人（Agent 起草） | 该期战役收口（`Executed-Frozen`）或历史期抽取迁移 | 里程碑（收口当期）+ 季度抽读 3 期 | 档案状态 `Frozen`；§5 负结果必填；指标带口径 |
| T7 | `docs/archive/README.md` + `docs/superseded/README.md`（2 份索引） | 存量快照索引与已取代文档索引；退役后仅保留 manifest/README | Agent 会话 | 快照新增/退役（Phase 3 后可能整目录退役，见 task-7） | 季度 | 只读/Frozen；`docs/archive/**` 链接检查排除 |

---

## 5. 产品与外部说明（8 个受管单元）

> 判定标准：对外可见（GitHub 首页/贡献者/使用者）或作为产品产物一部分的说明。**状态头豁免**（v2 方案 §6.1）：不强制元数据头，但必须被 `docs/README.md` 或宿主目录 README 索引。

| ID | 路径 | 职责（一句话） | owner 角色 | 更新触发条件 | 审查频率 | 状态头要求 |
|---|---|---|---|---|---|---|
| E1 | `README.md`（根，首屏） | 仓库门面：当前方向第一段 + 快速开始 + 关键入口（Phase 4 收敛，首屏 ≤60 行） | 仓库所有者 | 当前方向/主入口/快速开始变化（**每个里程碑必查第一段是否过时**） | 里程碑 + 季度 | 外部说明豁免；**首屏第一段 = 最新已执行事实**，历史段落下移 |
| E2 | `CONTRIBUTING.md` | 贡献流程：分支/PR/门禁/提交规范 | 仓库所有者 | 门禁流程、PR 规范、发布流程变化 | 季度 | 外部说明豁免 |
| E3 | `examples/contracts_v1/README.md` | 合成 demo bundle/viewer 用法 | 工程负责人 | demo 命令、输出结构变化 | 季度 | 外部说明豁免 |
| E4 | `examples/m1_first_plan/README.md` | 首个真实输入预检示例说明 | 工程负责人 | 预检流程/输入格式变化 | 季度 | 外部说明豁免 |
| E5 | `archive/v5x_scripts/README_scripts_convention.md` | 脚本入口与常用命令（挡在"根目录一次性脚本"之前） | 工程负责人 | 入口脚本增删、命令变更 | 季度 | 外部说明豁免 |
| E6 | `data/README.md` | 数据来源、大文件不入库、`data/manifests/*ids*` 例外与 SHA 规则 | 科学负责人 + 工程负责人 | 数据版本/来源/例外规则变化（数据发布前必查） | 里程碑（数据发布）+ 季度 | 外部说明豁免；例外必须逐条写清 |
| E7 | `outputs/README.md` | 产物目录约定（本地/不入库/可重生成边界） | 工程负责人 | 产物结构、导出约定变化 | 季度 | 外部说明豁免 |
| E8 | `artifacts/**/model_card.md`（8 份，1 个受管单元） | 模型卡：模型身份/数据/指标/已知限制（bundle 的对外说明书） | 科学负责人（产物生成会话起草） | **每次新 bundle 发布必须新增/更新模型卡**；指标口径变化 | 季度抽检（8 份轮查） | 固定 `Evidence-ReadOnly`（只读产品证据）；不得与报告数字矛盾 |

---

## 6. 维护节奏总表（与 `MAINTENANCE.md` 对齐）

| 频率 | 对象 | 动作 | 产出 |
|---|---|---|---|
| 双周（隔周五 30 min） | 核心集中"双周"档（约 20 单元）+ `docs/agent_runs/FAQ.md` | 过讨论→落档队列、状态转换、索引一致性 | 台账增量 ≥1 条 + 受影响文档更新 |
| 每里程碑（战役收口/阶段转换） | 治理根全读、计划/设计/合同、档案条目 | 文档冻结评审（状态/版本/结论/负结果） | 状态转换 + 档案条目定稿 |
| 每季度 | 全核心集 + ADR 抽 1/3 + 度量 | 档案树刷新 + 季度审计（复用 `doc_audit/` 模板） | 季度审计报告 + 指标对照 |
| 事件驱动（48h 内） | 五类落点（见 `MAINTENANCE.md` §3） | 讨论/决策 → 判定树 → 落档 | 五类落点之一 + 台账登记 |

## 7. 依赖与待确认

| # | 依赖/待确认 | 影响 | 处理 |
|---|---|---|---|
| 1 | task-6 档案树文件清单与命名（`docs/archive_tree/`） | §4 的 T1–T7 路径 | 以其产出为准，替换路径即可；字段不变 |
| 2 | task-7 清理候选与 `KEEP_REPORT` 终报清单 | §0 "不纳入锚点"的终报数量 | 清理定稿后回填（|锚点| ≤6） |
| 3 | 上位方案 Phase 0 是否签认（ADR-0035 + AGENT §9.4 例外） | G7/G8/退役机制 → `Canonical` | 未签认前，本核心集按"提案"执行；退役动作不启动 |
| 4 | `docs/README.md` 是否按 C7-A 重写为 4 入口短索引 | G6/E1 的形态 | 重写后核心集索引只保留 4 个入口；本表不新增索引 |
| 5 | `doc_manifest.py`（G5）实现与 CI 接入（Q15） | "状态头正确性"能否机器度量 | G5 落地前用 `docs_consistency --strict` + 人工季度抽检替代 |

---

## 附录 A｜核心集速查（47 单元一行式）

治理根 11（G1–G11）；现行合同与接口 8（C1–C8）；现行设计 5（D1–D5）+ 计划 7（P1–P7）+ ADR 集 1（A1，34 份）；档案树入口 7（T1–T7）；产品与外部说明 8（E1–E8）。
owner 角色分布（47 单元按主责计）：仓库所有者 10、科学负责人 18、工程负责人 11、Agent 会话 8。
`DOC_OWNERS.csv` 共 **68 行** = 47 个核心受管单元 + 21 个目录级巡检/watch 单元（W1–W15 及 ADR 活跃子集 A1a–A1f 等），两者合计即全部受管对象。

## 附录 B｜并入上位方案的方式

- §1–§5 的五张表 → 直接替换 `PES2TS_开发档案树与文档清理方案_v1_20261121.md` 的 `<!--FILL:CORE_SET-->`（建议保留本文件 §0 口径说明作为前置段落）。
- owner 角色定义与"角色可兼任"说明 + `DOC_OWNERS.csv` 的摘要表 → 填充 `<!--FILL:CORE_SET_OWNERS-->`（§4.2）。
- `DISCUSSION_TEMPLATE.md` → 填充附录 D（`<!--FILL:DISCUSSION_TEMPLATE-->`）。
- 本文件的"更新触发条件/审查频率/状态头要求"三列是维护契约，必须与本文件 `MAINTENANCE.md`、`DOC_OWNERS.csv` 逐字段一致；改动任一处三步同步。
