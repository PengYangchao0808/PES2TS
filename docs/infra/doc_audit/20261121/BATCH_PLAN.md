# 迁移批次手册（BATCH_PLAN）— 2026-11-21

> **状态**：Evidence-ReadOnly（执行手册；**本文件不授权任何移动/删除**，执行权属仓库所有者）
> **层级**：L4 运营（归档执行）
> **取代关系**：无；是 [开发文档规范化归档方案 v2.0](../../PES2TS_开发文档规范化归档方案_v2_20261121.md) §9 的可执行细化
> **上游依赖**：[DISPOSITION.csv](DISPOSITION.csv)（253 条逐条处置）、[LINK_FIX.csv](LINK_FIX.csv)（条件迁移引用清单）、[GOVERNANCE_CONSTRAINTS.md](GOVERNANCE_CONSTRAINTS.md) §3/§6（52 条钉死路径、D-01…D-22）、[RECOMMENDATIONS.md](RECOMMENDATIONS.md)
> **下游消费者**：执行归档的所有者、复核者
> **当前版本**：v1 @ 2026-11-21（基线 master@2f1d3ba；**本批执行前工作树干净**）

---

## 0. 本批结论（先看这里）

| 批次 | 含动作 | 条数 | 是否移动文件 | 是否需要所有者批准 |
|---|---|---:|---|---|
| **B0** 登记/索引/横幅 | `keep` **109** + `retain-readonly` 109 + `banner` 16 | **234** | **否** | B0-5 之后需（状态/横幅改动） |
| **B1** 零引用已取代 → `superseded/` | `move`（立即满足三条件） | **0** | 否 | — （空集） |
| **B1′** 条件迁移（待批准，**B3 的子集**） | `move`（F19 三份，需先修 11 处链 + 2 处同迁免改） | 3 | 是（批准后） | **是** |
| **B2** 门禁名单内错位迁移 | 无建议项 | 0 | 否 | 不建议做 |
| **B3** 裁决 | 状态矛盾/无状态头/拆分/归属 | **19**（`keep` 16 + `move` 3，含 B1′） | 否（裁决后才可能动） | **是** |

> 批次合计：**B0 234 + B3 19 = 253**；B1′ 不重复计入（DISPOSITION 中 3 条 `move` 的 `batch` 记为 B3）。

**关键判断**：

1. **B1 立即归位为空集**——满足"0 引用 + 自身已声明被取代 + 不在门禁名单"的 4 份文档已于 2026-11-20 迁入 `docs/superseded/`；当前唯一候选家族 F19（EGNN 三代）仍各有 2–7 处引用，**不满足零引用条件**，只能走 B3 裁决后作为 B1′ 执行。
2. **B2 为空且不建议做**——把根目录 7 份 memo 或 `historical` 名单内文档"归位"需要同 PR 改 `docs_consistency_config.json` 与 `g_v_map.yaml` 并修链，**收益低于断门禁风险**（D-01/D-02/D-05）。
3. **本次唯一"动文件"的可能性在 B1′（3 份）**，且必须先批准、先改链、后迁移。

## 1. B0 — 零风险：登记、索引、横幅补全（234 条）

### B0-A 已完成（本次审计自身，可复核）

| # | 动作 | 对象 | 状态 |
|---|---|---|---|
| B0-1 | 新建方案 + 8 字段元数据头 | `docs/infra/PES2TS_开发文档规范化归档方案_v2_20261121.md` | ✅ |
| B0-2 | `docs/README.md` 增加方案与审计证据条目 | `docs/README.md`（顶部新增 2026-11-21 节） | ✅ |
| B0-3 | 修订台账增量登记 | `docs/plans/PES2TS_文档修订台账.md`（顶部 2026-11-21 条） | ✅ |
| B0-4 | 审计证据目录与可复跑脚本 | `docs/infra/doc_audit/20261121/**` | ✅ |

### B0-B 待执行（建议下一个 PR，按风险从低到高）

| # | 动作 | 对象 | 逐条依据 |
|---|---|---|---|
| B0-5 | 补/改 **3 处状态行**（仅改状态，不动位置） | `PES2TS_V5.4.2_…_v1.0_20261029.md`（`Executed-Frozen` 或加§执行指针）、`PES2TS_V5.3-MP_…_v1_20261021.md`（同）、`PES2TS_30h_…_v1.1_20261020.md`（删/改自相矛盾的 L5 旧状态行） | ANOMALIES `A01/A02/A03`（P0/P0/P1） |
| B0-6 | 补 **7 份历史横幅**（不移动） | `ADR-0007`、`ADR-0008`、`ADR-0009`、`ADR-0020`、`ADR-0023`、`ADR-0024`、`PES2TS_PES-N未来开发计划_v2_20261005.md` | `DISPOSITION.csv` 中 `action=banner` 的 16 条（含 B0-7 的 9 条降级） |
| B0-7 | 补 **9 条历史提案状态**（只改状态行，不搬） | `PES2TS_G0-G4整体开发框架修改方案_v1`、`PES2TS_G1_G2边界审计与补齐任务`、`双端拓扑条件训练与项目进度统一方案_v4`、`图论立体先验与单端协同延续优化方案_v2`、`扫描规划谱系合并方案`、`拓扑约束TS接近度与位移联合学习方案_v2`、`统一价值模型与搜索排序一体化发展方案_v1`、`统一开发顺序与阶段验收`、`逐步实施与验证方案` | `RECOMMENDATIONS.md` §四-B |
| B0-8 | 补 **13 份最小状态块**（无状态头；DISPOSITION 未登记中 设计 10 + 阶段计划 3） | `V541_WP7_frozen_dev_eval_设计方案_v1.0`（设计类）、`PES2TS_V5.4_剩余架构研究工作方案_v1.0`、`PES2TS_V5.1_修复冲刺与扩量准入方案_v1`、`G2-ACP能力补齐与联合验收计划`、`G2-下一批算法调整与测试方案`、`ACP执行统一与旧代码清理方案` 等（完整 13 条见 DISPOSITION `canonical_status=未登记` 且 `doc_type∈{设计,阶段计划}`） | ANOMALIES `A13/A14/A15`；§8.2 逐桶动作 |
| B0-9 | 补 `docs/README.md` **2026-11-06 条目**（V5.4.3 已执行） | `docs/README.md` | ANOMALIES `A04`（ADR-0034 不改文，靠索引说明历史时点） |
| B0-10 | 补 **双索引缺口 27 条**（双缺 40 − agent_runs 12 − infra 1 = 27；reports 19 + plans 4 + superseded 4 中可索引者；agent_runs 12 由 `docs/agent_runs/README.md` 统一索引） | `docs/README.md` / `docs/plans/PES2TS_文档修订台账.md` | C-08；`index_coverage_stats.txt` 逐条清单 |
| B0-11 | 修 **索引自述版本 5 处**（宪法 v2.7→v2.9、阶段定义 v4.7→v4.8、V5–V7 v3.2→v3.3、V5.1-R3 v3.1→v3.2、`AGENT.md:8` 的宪法版本登记） | `docs/README.md`、`AGENT.md` | C-03/C-04（改 AGENT.md 属受审改动，单独 PR） |
| B0-12 | 根目录 7+1 份 memo 在 `docs/agent_runs/README.md` 建索引表（不移动） | `docs/agent_runs/README.md` | ANOMALIES `A30`；v1 方案 §4-P2 折中 |
| B0-13 | 冻结本方案为 `Canonical` 并接入 G5（M3/M4） | 方案 §10.3/§10.4、`.github/workflows/ci.yml` | Q15；需所有者批准 |

**B0 验收**：

```powershell
python scripts/check_docs_links.py                        # 0 error；warning 不高于 172
python scripts/agent_ops/docs_consistency.py --strict     # CLEAN
python scripts/render_gv_map.py --check                   # current
python -m pytest -q tests/test_gv_registry.py tests/test_config_env.py
```

> 注意：B0-5/B0-7 改的是 `historical`/`root_scope` 名单内文件的**状态行**——允许（改内容不改路径），但 `historical` 文件必须**保留 `ADR-0030` 横幅**，且 `docs_consistency --strict` 必须仍 CLEAN。

## 2. B1 — 零引用已取代 → `docs/superseded/`：**空集（0 条）**

### 2.1 判定复核（为什么是 0）

2026-11-20 已迁入 4 份（`docs/superseded/README.md` 索引在册）。当前全库复核：

| 候选 | 自身已声明被取代 | 引用数 | 在门禁名单 | 结论 |
|---|---|---:|---|---|
| `PES2TS_EGNN_BC_IQL_SAC下一步完整训练方案_v2_20261015.md` | 是（"前版记录，已由统一方案 v4 取代"） | 3 | 否 | ❌ 不满足零引用 → B1′ |
| `PES2TS_EGNN_BC_IQL_SAC实施与训练计划_v3_20261015.md` | 是 | 7（含 `AGENT.md`、`README.md`、`docs/README.md`） | 否 | ❌ 同上 → B1′ |
| `PES2TS_IRC_BC_IQL_SAC直接位移研究方案_v1_20261015.md` | 是 | 2（`docs/README.md`、台账） | 否 | ❌ 同上 → B1′ |
| 其余声明 Superseded 者（12 份） | 是 | ≥1 或门禁钉死 | 多数是 | ❌ 不满足 |

### 2.2 复核命令（可复跑）

```powershell
# 对每个候选：期望输出为空才可立即迁移
git -c core.quotepath=false grep -l -- "PES2TS_EGNN_BC_IQL_SAC下一步完整训练方案_v2_20261015" -- ':!docs/archive' ':!docs/superseded'
git -c core.quotepath=false grep -l -- "PES2TS_EGNN_BC_IQL_SAC实施与训练计划_v3_20261015" -- ':!docs/archive' ':!docs/superseded'
git -c core.quotepath=false grep -l -- "PES2TS_IRC_BC_IQL_SAC直接位移研究方案_v1_20261015" -- ':!docs/archive' ':!docs/superseded'
```

## 3. B1′ — 条件迁移（F19 三份；**需所有者批准后执行**）

**前置条件（缺一不可）**：① 所有者批准；② 先修 `LINK_FIX.csv` 的 **11 处需改链引用**（另 2 处为"同迁兄弟文件互相引用"，迁移后相对链接不变 → `action=no-change`，**不要**按 new_target 改写，否则断链）；③ `docs/superseded/README.md` 索引表预登记三行；④ 台账登记。

> `LINK_FIX.csv` 有两个路径列：`new_target` 是**仓库根相对路径**（便于脚本/git 校验）；`markdown_link` 是**应写入 markdown 的相对链接**（已按 referrer 所在目录换算，如 `docs/design/decisions/ADR-0020` → `../../superseded/…`、`docs/plans/台账` → `../superseded/…`、`docs/README.md` → `superseded/…`、根 `AGENT.md`/`README.md` → `docs/superseded/…`）。**改链时用 `markdown_link` 列。**

```powershell
# 步骤 1：改链（13 处，见 LINK_FIX.csv；示例）
#   AGENT.md:183、README.md:26、docs/README.md:172、ADR-0020:32、
#   docs/plans/PES2TS_EGNN_BC_IQL_SAC下一步完整训练方案_v2:3、
#   docs/plans/PES2TS_文档修订台账.md:407/413/415 等
#   把指向 docs/plans/<name>.md 的相对链接改为 docs/superseded/<name>.md
python scripts/check_docs_links.py     # 期望 0 error

# 步骤 2：迁移（git mv 保留历史）
git mv "docs/plans/PES2TS_EGNN_BC_IQL_SAC下一步完整训练方案_v2_20261015.md" "docs/superseded/"
git mv "docs/plans/PES2TS_EGNN_BC_IQL_SAC实施与训练计划_v3_20261015.md" "docs/superseded/"
git mv "docs/plans/PES2TS_IRC_BC_IQL_SAC直接位移研究方案_v1_20261015.md" "docs/superseded/"

# 步骤 3：三件套同步
#   docs/superseded/README.md 索引表 +3 行（原路径/取代者/迁入日期）
#   docs/plans/PES2TS_文档修订台账.md 增量登记
#   docs/README.md 对应条目状态列改 Superseded（链接改新路径）

# 步骤 4：门禁
python scripts/check_docs_links.py
python scripts/agent_ops/docs_consistency.py --strict
python scripts/render_gv_map.py --check
python -m pytest -q tests/test_gv_registry.py tests/test_config_env.py
```

**回滚（本批未提交时）**：`git reset --hard HEAD`（一并重置 index，可撤销已 staged 的 `git mv`）+ `git clean -fd docs/superseded`（清本批新增未跟踪文件）→ 复跑门禁。**注意**：`git checkout -- .` 无法撤销 staged 的 `git mv`（2026-11-21 临时仓库实测），不要使用；已提交时用 `git revert <commit>`。

**风险**：`AGENT.md`/`README.md` 为 `root_scope` 门禁输入——只改链接文本不改路径，属安全改动，但须与 `AGENT.md` 的"执行投影"口径一致（D-14/D-16）。

## 4. B2 — 门禁名单内错位迁移：**无建议项（0）**

本方案**不建议**移动任何 `historical`/`required_files`/`root_scope` 路径（D-01/D-02/D-05/D-09）。若所有者仍决定移动根目录 7 份 memo，必须同一 PR 完成下列配置修改（示意，实际以 `docs_consistency_config.json` 当前内容为准）：

```jsonc
// docs_consistency_config.json —— 若移动 V52_PROGRESS_STOP_REPORT.md 等 5 份 historical 根 memo
"historical": [
  // 删除旧路径，改为新路径：
  // "V52_PROGRESS_STOP_REPORT.md"  →  "docs/agent_runs/sessions/20261021_v52_stop/V52_PROGRESS_STOP_REPORT.md"
],
"root_scope": [
  // 同理改路径；R3_RESULT_SUMMARY.md、V53_ROLLOUT_REPORT.md 为软登记，也需同改
]
```

另需：`git mv` + 修全部相对链接 + `g_v_map.yaml`（若涉及 evidence）+ 重渲染 G–V 文档 + 台账登记。**综合风险：中高；收益：低。建议不做。**

## 5. B3 — 需所有者裁决（19 条）

| # | 裁决项 | 选项 | 影响 |
|---|---|---|---|
| Q1 | V5.4.2 方案状态（A01/P0） | A 改 `Executed-Frozen`；B 加§执行指针不改状态 | 决定该方案能否作为引用依据 |
| Q2 | V5.3-MP 计划状态（A02/P0） | A 改 `Executed-Frozen`；B "文本冻结 + 执行状态见正文" | 同上 |
| Q3 | 30h 计划两处状态矛盾（A03/P1，门禁钉死） | A 统一 `Executed-Frozen`；B 只改 L5 | 低（不移动） |
| Q4 | ADR-0034 状态 vs 已执行（A04/P1） | A 保留原文 + README 补条目（推荐）；B 改状态（违反"不改文"惯例） | 中 |
| Q5 | F19 三份迁移（B1′） | A 批准并修 13 处链；B 仅加"已取代"指针不移动 | 中（引用多） |
| Q6 | F03 三位一体 v1 是否并入 R3 v3 家族 | A 并入并在 superseded README 登记关系；B 维持 Active-Proposed | 低 |
| Q7 | 5 份无状态头文档补什么状态（A13/A14/A15） | A `Historical`；B `Active-Proposed`（若仍用） | 中（决定后续引用） |
| Q8 | `docs/plans` 设计类 3 份是否迁 `docs/design/`（A06） | A 不迁（推荐，门禁/索引成本高）；B 迁 + 同 PR 改链 | 中 |
| Q9 | 台账 166 KB 是否拆分（A10） | A 暂不拆；B 拆"年度索引 + 附录"（需改 AGENT §9.3 引用） | 中 |
| Q10 | 根 README 与 `docs/README.md` 双索引治理（A11） | A 根 README 收敛为门面（推荐）；B 维持双维护 | 中高 |
| Q11 | `docs/infra/` 收录规则（A31） | A 按方案 §6.3（推荐）；B 并入 `docs/design/` | 低 |
| Q12 | 9 条清单外/新增文档归属（含本审计目录 5 份） | A 按方案 §6.4 归 `doc_audit/` + `infra/`（推荐）；B 另行裁决 | 低 |
| Q13 | 路线 B（目录编号化重构）是否启动 | A 暂不启动（推荐）；B 规划 v3.0 另行实施 | 高（全量路径映射） |
| Q14 | 根目录 7 份 memo 是否物理移动（B2） | A 不移动 + 加索引（推荐）；B 移动 + 改门禁配置 | 中 |
| Q15 | `doc_manifest.py`（G5）是否实现并接 CI（方案 §10.4 M3/M4） | A 批准（推荐）；B 仅本地 | 中（治理闭环） |
| Q16 | 仓库开源范围（LICENSE 已有 MIT；reports/agent_runs 是否公开） | A 全公开；B 仅代码+README；C 私有 | 高 |
| Q17 | `sync/20261018` 分支是否保留 | A 保留；B 合并后删除 | 低 |
| Q18 | 报告类历史文档是否补"当前阶段指针"批量（台账 §6 未完成项） | A 批量加（只加指针）；B 维持 | 低 |
| Q19 | `docs/README.md` 拆分阈值（≥500 行？） | A 暂不拆；B 按目录拆子 README | 中 |

## 6. 通用门禁与回滚（每批必跑）

```powershell
# 四门（缺一不可）
python scripts/agent_ops/docs_consistency.py --strict
python scripts/check_docs_links.py
python scripts/render_gv_map.py --check
python -m pytest -q tests/test_gv_registry.py tests/test_config_env.py

# 若改动代码/测试，再加：python -m pytest -q --ignore=tests/pesn

# 回滚（本批未提交时；reset --hard 才能撤销已 staged 的 git mv）
git reset --hard HEAD
git clean -fd docs/superseded
git status --short    # 应只剩预期文件
```

**通过判据**：`0 violation`、`0 error`、`current`、`26 passed`；warning 不高于基线 172（新增须逐条说明）。

## 7. 执行记录

执行任何批次后，在 [MIGRATION_LOG.md](MIGRATION_LOG.md) 追加一行（日期/批次/旧路径/新路径/引用修正数/门禁结果/执行人）。**本文件当前无任何执行记录 = 本次审计未移动任何文件。**
