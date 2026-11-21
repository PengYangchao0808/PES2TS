# PES2TS 文档清理方案（删除安全性与分批执行手册）

> 状态：Active-Proposed（本文件是方案，未执行任何删除；仓库只读）
> 层级：操作性方案（Operational，非规范）；规范冲突一律以宪法为准
> 生成：2026-11-21 · task-7 · cleanup-auditor
> 输入：`docs/infra/doc_audit/20261121/`（inventory 253 / DISPOSITION / GOVERNANCE_CONSTRAINTS / RECOMMENDATIONS / FAMILIES / ANOMALIES）、
> `.slim/deepwork/20261121_cleanup/archive_tree/ARCHIVE_MAP.csv`（E01–E17 档案条目）
> 配套交付：`CLEANUP_CANDIDATES.csv`（253 行全覆盖）、`KEEP_SET.csv`、`LINK_FIX.csv`、`check_deletion.py`、`gen_candidates.py`
> 边界：**本方案不执行任何 `git rm`/`git mv`**；所有命令仅供 owner 批准后逐批执行。md 内不用超链接（避免复制进 docs/ 时受链接门禁扫描）。

---

## 0. 结论摘要（三档强度）

| 强度 | 删除数 | 保留数 | 判据 | 前置 |
|---|---:|---:|---|---|
| **safe（D1）** | **67** | 186 | 只删"已被取而代之且无独有结论"的重复件：28 份 archive 载荷副本、4 份 superseded、18 份已被承接的旧计划、4 份历史提案 ADR、3 份 old demo、6 份零引用过程记录、1 份旧清理方案、3 份纯过程报告 | Phase 0 完成（治理条款生效） |
| **standard（D2）** | **129**（+62） | 124 | 再删过程性报告/handoff/一次性计划：18 报告 + 8 运营记录 + 13 计划 + 11 设计 + 9 ADR + 2 合同 + 1 infra；结论先抽入档案树 | D2 = D1 全绿 + 档案树 E01–E17 条目验收 |
| **aggressive（D3）** | **177**（+48） | **76**（核心 68 + archive manifest 8） | 只留核心：治理根 10 + 现行合同/ADR/计划 30 + 证据 13 + 索引/manifest 10 + 外部说明/环境 13；**18 条门禁钉死路径走 GATE_CONFIG_MUST_CHANGE**（同 PR 改配置+重渲染，否则保留） | D3 = D2 全绿 + 配置 diff 双签 + g_v_map 重渲染 |

- 三档集合**严格单调**：safe(67) ⊂ standard(129) ⊂ aggressive(177)，KEEP=76，删除集 177 + 保留 76 = 253；`check_deletion.py` 逐条断言。
- 逐条判定：`KEEP` 76、`DELETE` 60（safe 内无需抽结论）、`ARCHIVE_EXTRACT_THEN_DELETE` 91、`MERGE_INTO` 8、`GATE_CONFIG_MUST_CHANGE` 18。
- **Lead 裁决（2026-11-21）**：5 份 CORE_SET 核心集文档由启发式删除判定改判 KEEP（`docs/contracts/PES2TS_contracts_v1.md`、`PES2TS_ACP字段映射_v1.md`、`PES2TS_数据规范与ACP对接优化方案.md`、`docs/design/PES2TS_CODE_MAP_v2_20261018.md`、`docs/design/PES2TS_Classic图论驱动与xTB原生约束扫描设计_v1_20261006.md`），已用 `OVERRIDE_KEEP` 显式覆盖并重生成全部 CSV。
- **需先抽结论的文档 109 条**（91 抽取 + 18 GCM），全部映射到档案树条目（`target` 列），与 task-6 `ARCHIVE_MAP.csv` 的 E01–E17 对齐。
- 危险级别：**D1 低风险**（重复件/副本，链接修正量小）；**D2 中风险**（109 条抽取、docs/README 与台账索引大面积改写）；**D3 高风险**（18 条门禁钉死 + g_v_map evidence 4 条 + 4 门 CI 共同受影响）。

### 0.1 只读校验结果（`check_deletion.py`，15/15 PASS，exit 0）

执行 `python .slim\deepwork\20261121_cleanup\cleanup\check_deletion.py`（2026-11-21，master@2f1d3ba 工作区）：

| 断言组 | 结果 | 关键数字 |
|---|---|---|
| (a) 待删文件存在 / 可 git 恢复 / 门禁钉死一致性 / 配置 diff 声明 | 5/5 PASS | 253/253 存在；0 未跟踪删除；34 条钉死路径 0 越界；18/18 GCM 均为钉死路径且在本文档逐条出现；config edits: historical=17、required_files=1、root_scope=6；g_v_map 受影响 4 条 |
| (b) 链接预演（check_docs_links 扫描面） | 2/2 PASS | 562 条硬链接全部在 `LINK_FIX.csv` 声明；另有 124 条 inline-code 提及（不产生链接 error） |
| (c) 三档单调 | 4/4 PASS | safe=67 ⊂ standard=129 ⊂ aggressive=177；KEEP ⟺ intensity='-'；GCM ⟹ aggressive |
| (d) KEEP_SET 与候选互补 | 4/4 PASS | 候选 253/253 唯一；KEEP 76 = KEEP_SET 76；76 + 159 + 18 = 253 互斥 |
| **总计** | **15/15 PASS — 全部 PASS** | 候选 253；KEEP 76；删除 159；GCM 18；三档 67/129/177；LINK_FIX 562 |

输出原文留档：`check_deletion_output.txt`。**校验通过的只是"方案自洽性"**；删除合法性仍取决于本文件 §1 的 Phase 0 治理修订与 §2 快照前置。

---

## 1. Phase 0：治理修订（一切删除的合法性前置）

现行规则**禁止删除证据**：`AGENT.md` §9.4 报告只读；`docs/archive/README.md` 规则 1/2（快照只读、禁当墓地）；`docs/superseded/README.md`（正文不改）；`docs/consistency` D-01/D-02/D-06/D-07/D-08。**不先改治理，D1 即违规**。Phase 0 只改规则/建档案树/打快照，**不删任何文件**。

### 1.1 新增 ADR（草案要点）

`docs/design/decisions/ADR-0035-文档生命周期与退役政策.md`（Proposed → Accepted，owner 签认）：

1. **生命周期词表**（6 值，收敛现行 137 种原始值）：`Canonical`（权威现行）/ `Active-Proposed` / `Executed-Frozen` / `Superseded`（被取代）/ `Evidence-Historical`（只读历史证据）/ `Retired`（已退役：证据已档案化 + tag 可恢复，仓库内不再保留正文）。
2. **退役四条件（缺一不可）**：① 结论/负结果已抽入 `docs/archive_tree/` 对应条目；② 退役前存在 `pre-doc-cleanup-<date>` tag 与 `git bundle` 快照；③ 所有活跃引用已修（`check_docs_links` 0 error）且索引/台账登记"原路径→档案条目→tag"；④ owner 对批次清单书面签认（一次批量授权可覆盖整批）。
3. **报告只读例外**：`docs/reports/**` 默认只读不改；满足四条件后可整篇退役，但 **g_v_map evidence 报告与每个战役的最新终报不可退役**（列白名单）。
4. **archive 快照策略**：manifest/verification/README **永久保留**；`payload`（镜像树、副本）在四条件下可退役，README 改为"载荷已退役，经 tag 恢复"。
5. **绝对禁止退役**：宪法、AGENT、阶段定义、完整开发流、台账、docs/README、现行合同、现行 ADR、模型卡、门禁配置文件、archive manifest+verification、`data/manifests` 冻结名单。
6. **门禁配置变更的合法性**：D-19 禁止"为让红灯变绿"改配置；本政策区分"为删除证据链而受审地改配置"——须 ADR 记录 + owner 双签 + 前后门禁数字对照，且**不得**放宽任何检查项。

### 1.2 需修订的条款清单（逐文件）

| # | 文件 | 改动 | 审批点 |
|---|---|---|---|
| G1 | `AGENT.md` §9 | 新增 §9.6「文档退役与档案化流程」（四条件 + 批次签认 + tag 要求）；§9.4 报告只读加"经 ADR-0035 退役例外"一句 | A1 |
| G2 | `docs/PES2TS_开发宪法_v1.md` §12/§14 | 在"修订豁免/台账关系"处登记"文档退役"为受审动作（不新增规范条文外的能力声明） | A1（双签） |
| G3 | `docs/plans/PES2TS_文档修订台账.md` | 新增 §7「文档退役登记」（列：原路径 / 档案条目 / pre tag / 批准人 / 日期 / 退役批次）；D1/D2/D3 每批结束后追加增量 | A1/A4 |
| G4 | `docs/archive/README.md` | 规则 2 改为：快照载荷可退役（manifest 永久保留，附 tag 与恢复命令）；新增 §"已退役载荷清单" | A1 |
| G5 | `docs/superseded/README.md` | 新增第 5 条：本区文档在档案化后可整体退役；索引表增"退役批次/tag"列 | A1 |
| G6 | `docs/README.md` | 新增 §0「文档生命周期入口」：live 索引只列 Canonical/Active/现行证据；退役件只经 `docs/archive_tree/README` 访问 | A3 |
| G7 | `.github/workflows/ci.yml`（可选） | 若接入 G5 门禁 `doc_manifest --check`：仅在 D3 后接入，不改变现有 4 门 | A5 |
| G8 | `docs/design/decisions/ADR-0035-*.md` | 新 ADR（Proposed→Accepted） | A1 |

审批点：**A1** ADR-0035 + G1–G6 文本修订（owner 单次签认）；**A2** 档案树 v1（E01–E17 条目 + 109 条抽取内容）评审通过；**A3** pre tag/bundle/快照就位且 `docs/README` 生命周期入口上线；**A4** 每个删除批次单独 PR + 门禁对照 + 台账登记；**A5** D3 配置 diff 双签（owner + 工程负责人）后执行。

---

## 2. 保护性快照（Phase 0 内完成，D1 之前）

```powershell
# 2.1 工作区必须干净、基线记录
git status --short
git rev-parse HEAD > .slim\deepwork\20261121_cleanup\backup\BASE_COMMIT.txt
python scripts/agent_ops/docs_consistency.py --strict | Tee-Object -FilePath .slim\deepwork\20261121_cleanup\backup\gate_before.txt
python scripts/check_docs_links.py 2>&1 | Tee-Object -Append -FilePath .slim\deepwork\20261121_cleanup\backup\gate_before.txt
python scripts/render_gv_map.py --check | Tee-Object -Append -FilePath .slim\deepwork\20261121_cleanup\backup\gate_before.txt
python -m pytest -q tests/test_gv_registry.py tests/test_config_env.py | Tee-Object -Append -FilePath .slim\deepwork\20261121_cleanup\backup\gate_before.txt

# 2.2 删除前锚点（可一键恢复）
git tag -a pre-doc-cleanup-20261121 -m "PES2TS doc cleanup baseline (253 inventory; safe=67/std=129/agg=177)"
git bundle create .slim\deepwork\20261121_cleanup\backup\pre-doc-cleanup-20261121.bundle --all

# 2.3 可选：被删文件整包快照（脱离 git 也能看原文）
git archive --format=zip -o .slim\deepwork\20261121_cleanup\backup\deleted-docs-D1.zip HEAD -- (Get-Content .slim\deepwork\20261121_cleanup\cleanup\deletion_manifest_D1.txt)
# 若 tar 可用：
# tar -czf .slim\deepwork\20261121_cleanup\backup\docs-full-20261121.tar.gz docs README.md AGENT.md
```

恢复路径（任一）：`git show pre-doc-cleanup-20261121:<path>`（单文件）；`git log --diff-filter=D -- <path>` 找删除 commit；`git reset --hard pre-doc-cleanup-20261121`（整仓回滚）；`git clone pre-doc-cleanup-20261121.bundle`（离线恢复）。

---

## 3. 分批执行（D1 → D2 → D3）

通用前置检查（每批开始前）：

```powershell
# P0 清单一致性 + 只读校验（本方案自带）
python .slim\deepwork\20261121_cleanup\cleanup\check_deletion.py
# P1 基线干净 & tag 存在
git rev-parse pre-doc-cleanup-20261121
# P2 本批每个路径存在且被 git 跟踪
Get-Content .slim\deepwork\20261121_cleanup\cleanup\deletion_manifest_D1.txt | ForEach-Object { git ls-files --error-unmatch -- "$_" }
# P3 抽取覆盖（D2/D3）：本批 target 档案条目在 ARCHIVE_MAP.csv 中均有条目且 extract_needed=Y 的已填内容
```

### 3.0 生成批次清单（精确路径，中文路径逐条引号）

```powershell
$rows = Import-Csv .slim\deepwork\20261121_cleanup\cleanup\CLEANUP_CANDIDATES.csv -Encoding UTF8
$d1 = $rows | Where-Object { $_.intensity -eq 'safe' }       | ForEach-Object { $_.path }
$d2 = $rows | Where-Object { $_.intensity -eq 'standard' }   | ForEach-Object { $_.path }
$d3 = $rows | Where-Object { $_.intensity -eq 'aggressive' } | ForEach-Object { $_.path }
$d1 | Set-Content -Encoding UTF8 .slim\deepwork\20261121_cleanup\cleanup\deletion_manifest_D1.txt
$d2 | Set-Content -Encoding UTF8 .slim\deepwork\20261121_cleanup\cleanup\deletion_manifest_D2.txt
$d3 | Set-Content -Encoding UTF8 .slim\deepwork\20261121_cleanup\cleanup\deletion_manifest_D3.txt
# 单调复核（必须 67 / 129 / 177）
$d1.Count; ($d1 + $d2).Count; ($d1 + $d2 + $d3).Count
```

### 3.1 D1（safe，67 条）——重复件退役，D1 不抽结论

**内容**：archive 载荷 28、superseded 4、旧计划 18、历史 ADR 4、old demo 3、零引用过程记录 6、旧清理方案 1、纯过程报告 3（明细见 CSV `intensity=safe`）。其中 7 条走 `MERGE_INTO`（如 `docs/infra/PES2TS_仓库归档与清理方案_v1_20261120.md` → v2 方案；`PES2TS_实施路线图.html` → V5-V7 v3.0 方案），合并完成后才删。

```powershell
# D1-1 档案化点名（先把可复用结论落到档案树/现行件，再删）
#   MERGE_INTO 的 7 条：见 CSV target 列，逐条并入后由 owner 在 PR 描述勾选
# D1-2 逐条删除（保留 Git 历史；禁止 -f 批量）
Get-Content .slim\deepwork\20261121_cleanup\cleanup\deletion_manifest_D1.txt |
  ForEach-Object { git rm -- "$_" }        # 中文路径由引号保护
# 示例（幂等校验用）：
# git rm -- "docs/archive/20261007_pre_light_policy/AGENT.md"
# git rm -- "docs/superseded/PES2TS_单端点IRC生成控制器_产品设计_v1_20261020.md"
# git rm -- "docs/plans/PES2TS_PES-N未来开发计划_v2_20261005.md"
```

**D1 同 PR 必须做的文本修正**（不允许跨 PR 遗留断链）：

1. `docs/superseded/README.md` 索引 4 行 → 改为"已退役（见 `docs/archive_tree/` E08/E11 + `pre-doc-cleanup-20261121`）"；
2. 7 个 archive 快照 `README.md` 各加一段"载荷已于 2026-11-21 退役，经 tag 恢复：`git show pre-doc-cleanup-20261121:<path>`"，并移除指向被删载荷的相对链接行；
3. `LINK_FIX.csv` 中 `batch=D1` 的行逐条处理（`docs/README.md` 删索引行、`README.md`/`AGENT.md` 改档案树指针；台账加退役登记而非删历史）；
4. 台账 G3 §7 追加 D1 登记（67 行汇总，不逐文档抄）。

**D1 验收**（4 门全绿 + 数字对照）：
`docs_consistency --strict` 0 violation（scanned 减少约 60）；`check_docs_links` 0 error、warning 不高于 172；`render_gv_map --check` current；`pytest tests/test_gv_registry.py tests/test_config_env.py` 26 passed。任一失败 → 回滚见 §3.4。

### 3.2 D2（standard，+62 条 → 累计 129）——过程性报告/handoff/一次性计划

**内容**：报告 18、agent_runs 8、计划 13、设计 11、ADR 9、合同 2、infra 1（明细 `intensity=standard`）。**每条先抽结论**：按 ARCHIVE_MAP `extract_content` 填 `docs/archive_tree/E01–E17` 条目（含负结果，不得美化）。

```powershell
# D2-0 抽取校验：本批 target 条目的 extract_content 非空比例 = 100%
python -c "import csv,io; r=[x for x in csv.DictReader(io.open('.slim/deepwork/20261121_cleanup/archive_tree/ARCHIVE_MAP.csv',encoding='utf-8-sig')) if x['extract_needed']=='Y']; print(len(r))"
# D2-1 删除
Get-Content .slim\deepwork\20261121_cleanup\cleanup\deletion_manifest_D2.txt |
  ForEach-Object { git rm -- "$_" }
```

**D2 同 PR 修正**：`LINK_FIX.csv` 中 `batch=D2` 的行（docs/README 约 100+ 行索引、台账若引用则改"档案化指针"）；`docs/plans/PES2TS_阶段定义_G0-G3.md`、`docs/PES2TS_开发宪法_v1.md`、`README.md` 中指向被删计划的链接改指 `docs/archive_tree/`；`docs/agent_runs/README.md` handoff 清单改"已档案化"表。

### 3.3 D3（aggressive，+48 条 → 累计 177；含 18 条 GCM）——核心收缩 + 门禁配置受审变更

D3 分两个子批次，**G3a 先于 G3b**：

**D3a：非门禁路径 30 条（先做，验证抽取质量）**
```powershell
# 30 条 = intensity=aggressive 且 decision=ARCHIVE_EXTRACT_THEN_DELETE（合同 1、ADR 9、计划 10、报告 10）
git rm -- "docs/design/decisions/ADR-0001-统一执行后端接缝.md"
git rm -- "docs/reports/PES2TS_v4阶段训练_最终报告与问题复核_20261017.md"
# …… 其余按 deletion_manifest_D3a.txt 逐条
```

**D3b：18 条 GATE_CONFIG_MUST_CHANGE（必须同 PR 改 4 处门禁输入并重渲染）**

本节 18 条路径逐条（`check_deletion.py` (a5) 断言本表存在）：

| # | 路径 | config 所属列表 | g_v_map 引用 | 处置 |
|---|---|---|---|---|
| 1 | `24H_EXECUTION_TIMELINE.md` | historical + root_scope | — | 配置移除 |
| 2 | `24H_FINAL_REPORT.md` | historical + root_scope | — | 配置移除 |
| 3 | `FAILURE_AND_REMAINING_WORK.md` | historical + root_scope | — | 配置移除 |
| 4 | `NEXT_DAY_DECISION_MEMO.md` | historical + root_scope | — | 配置移除 |
| 5 | `V52_PROGRESS_STOP_REPORT.md` | historical + root_scope | — | 配置移除 |
| 6 | `R3_RESULT_SUMMARY.md` | root_scope（软登记） | — | 配置移除 |
| 7 | `docs/contracts/PES2TS_FullPathNavigationContract_v1.md` | historical + required_files | — | 配置移除（两处） |
| 8 | `docs/contracts/PES2TS_ReactionIntent_Contract_v1.md` | historical | — | 配置移除 |
| 9 | `docs/contracts/PES2TS_GeometryPolicyContract_v1.md` | historical | — | 配置移除 |
| 10 | `docs/contracts/PES2TS_双端TS教师与相容转移合同_v2.md` | historical | — | 配置移除 |
| 11 | `docs/design/decisions/ADR-0021-无序双端拓扑与完整双侧TS教师主线.md` | historical | — | 配置移除 |
| 12 | `docs/design/decisions/ADR-0028-关键原子几何策略合同与V5-V7任务拆分.md` | historical | — | 配置移除 |
| 13 | `docs/design/decisions/ADR-0029-V5.2+双门评估与封存边界.md` | historical | — | 配置移除 |
| 14 | `docs/plans/PES2TS_30h_R3_V52_V53_分层训练与验收计划_v1.1_20261020.md` | historical | V5.2 evidence（executed，强制存在） | **替换 evidence 后再删** |
| 15 | `docs/plans/PES2TS_PES-N_V5-V7架构调整与实施计划_v1.2_20261018.md` | historical | V6.0–V7-C evidence（archived） | 替换/移除 evidence |
| 16 | `docs/plans/PES2TS_V5-V7_三阶段研究与架构修订方案_v2.0_20261020.md` | historical | — | 配置移除 |
| 17 | `docs/plans/PES2TS_V5.0-V5.2_十小时冲刺训练与验证执行方案_v1_20261018.md` | historical | V5.0/V5.1 evidence（proposed/archived） | 替换/移除 evidence |
| 18 | `docs/plans/PES2TS_V5.1-R2_损失尺度与数据完整性修复方案_v1_20261020.md` | historical | V5.1-R2 evidence（archived） | 替换/移除 evidence |

> **风险提示**：#14–#18 是 g_v_map evidence（`render_gv_map --check` + registry tests 双重断言）。若 owner 不批准"替换 evidence"，**保留这 5 条**（GCM→KEEP），aggressive 删除数降为 172，其余结论不变。**默认建议 D3b 只批准 #1–#13（13 条）**，evidence 5 条保留到下一轮（此时 aggressive 删除数 177→172）。

**D3b 配置 diff（旧 → 新，精确删除行）**

`scripts/agent_ops/docs_consistency_config.json`：

```diff
-  "historical": [
-    "docs/contracts/PES2TS_ReactionIntent_Contract_v1.md",
-    "docs/contracts/PES2TS_GeometryPolicyContract_v1.md",
-    "docs/contracts/PES2TS_双端TS教师与相容转移合同_v2.md",
-    "docs/design/decisions/ADR-0021-无序双端拓扑与完整双侧TS教师主线.md",
-    "docs/design/decisions/ADR-0028-关键原子几何策略合同与V5-V7任务拆分.md",
-    "docs/design/decisions/ADR-0029-V5.2+双门评估与封存边界.md",
-    "docs/plans/PES2TS_30h_R3_V52_V53_分层训练与验收计划_v1.1_20261020.md",
-    "docs/plans/PES2TS_V5-V7_三阶段研究与架构修订方案_v2.0_20261020.md",
-    "docs/plans/PES2TS_V5.0-V5.2_十小时冲刺训练与验证执行方案_v1_20261018.md",
-    "docs/plans/PES2TS_V5.1-R2_损失尺度与数据完整性修复方案_v1_20261020.md",
-    "docs/plans/PES2TS_PES-N_V5-V7架构调整与实施计划_v1.2_20261018.md",
-    "V52_PROGRESS_STOP_REPORT.md",
-    "NEXT_DAY_DECISION_MEMO.md",
-    "FAILURE_AND_REMAINING_WORK.md",
-    "24H_FINAL_REPORT.md",
-    "24H_EXECUTION_TIMELINE.md",
-    "docs/contracts/PES2TS_FullPathNavigationContract_v1.md"
-  ],
+  "historical": [],
   "root_scope": [
     "README.md",
     "AGENT.md",
-    "V52_PROGRESS_STOP_REPORT.md",
-    "NEXT_DAY_DECISION_MEMO.md",
-    "FAILURE_AND_REMAINING_WORK.md",
-    "24H_FINAL_REPORT.md",
-    "24H_EXECUTION_TIMELINE.md",
-    "R3_RESULT_SUMMARY.md",
     "V53_ROLLOUT_REPORT.md"
   ],
-  "required_files": { "docs/contracts/PES2TS_FullPathNavigationContract_v1.md": [ ... ] , ... }
+  "required_files": { "docs/contracts/PES2TS_FullPathNavigationContract_v1.md" 键整体删除，其余 26 键不动 }
```

> 注意：`historical` 在 D3b 后**清空**（17 条全部退役），`historical_missing_banner`（ADR-0030 横幅检查）随之无对象；`required_files` 27→26；`root_scope` 9→3。**不得**顺带删除任何必须子串（D-19 禁止放宽检查；本 diff 只删路径键，不改检查逻辑）。

`config/registry/g_v_map.yaml`（仅当批准 #14–#18）：

```diff
-  stage_doc 不动。
  V5.2  evidence:  docs/plans/PES2TS_30h_R3_V52_V53_分层训练与验收计划_v1.1_20261020.md
+  V5.2  evidence:  docs/reports/PES2TS_V52_任务纠偏审计_20261021.md  # 退役件替代证据（须先确认其覆盖 V5.2 冻结结论）
-  V5.0/V5.1 evidence: docs/plans/PES2TS_V5.0-V5.2_十小时冲刺训练与验证执行方案_v1_20261018.md
+  V5.0/V5.1 evidence: docs/reports/PES2TS_V5第一阶段冲刺执行报告_20261018.md  # 该报告 D2 已抽取，可先保留至 D3 后
-  V5.1-R2 evidence: docs/plans/PES2TS_V5.1-R2_损失尺度与数据完整性修复方案_v1_20261020.md
+  V5.1-R2 evidence: docs/archive_tree/E09-V5.1-R2.md  # 档案树节点
-  V6.0–V7-C evidence（6 处）: docs/plans/PES2TS_PES-N_V5-V7架构调整与实施计划_v1.2_20261018.md
+  V6.0–V7-C evidence（6 处）: docs/archive_tree/E09-PES-N-V5-V7.md
```

随后**必须**：`python scripts/render_gv_map.py --write`（禁止手改 `docs/design/PES2TS_G_V_MAPPING_v1_20261018.md`，D-03），再 `--check`。

**D3 删除命令**：
```powershell
Get-Content .slim\deepwork\20261121_cleanup\cleanup\deletion_manifest_D3b.txt |
  ForEach-Object { git rm -- "$_" }
# 中文路径示例：
# git rm -- "docs/design/decisions/ADR-0028-关键原子几何策略合同与V5-V7任务拆分.md"
# git rm -- "docs/plans/PES2TS_30h_R3_V52_V53_分层训练与验收计划_v1.1_20261020.md"
```

**D3 同 PR 修正**：`LINK_FIX.csv` 中 `batch=D3` 行（含 README/AGENT 里 5 份 memo 链接 → archive_tree E10）；`docs/archive/README.md` 退役载荷清单；`docs/README.md` 生命周期入口；台账 G3 登记；`docs/PES2TS_开发宪法_v1.md`/`AGENT.md`/阶段定义中指向被删历史合同的链接改档案树指针（**只改链接不删规范条文**）。

### 3.4 每批回滚

```powershell
# 情形 1：未 commit（工作区删除后想撤销）
git restore --source=HEAD --staged --worktree -- .
# 情形 2：已 commit 但未 push（回到基线）
git reset --hard pre-doc-cleanup-20261121
# 情形 3：已 push（用 tag 反向恢复，禁止 force push 到共享分支）
git revert --no-commit <delete_commit>            # 反向补丁
git checkout pre-doc-cleanup-20261121 -- <path>   # 单文件恢复
# 情形 4：本地仓库损坏（bundle 重建）
git clone .slim\deepwork\20261121_cleanup\backup\pre-doc-cleanup-20261121.bundle restored-repo
```

---

## 4. 删除后门禁影响与处置（4 门，缺一不可）

| 门禁 | 当前基线（2026-11-21 实测） | D1 后 | D2 后 | D3 后 | 处置 |
|---|---|---|---|---|---|
| `docs_consistency.py --strict` | scanned 169；0 violation；CLEAN | scanned 约 110；0 violation | scanned 约 60；0 violation | scanned 约 30；0 violation | historical 17→0；required_files 27→26；root_scope 9→3；**不得新增违例** |
| `check_docs_links.py` | checked 193 docs；0 error / **172 warning** | 0 error；warning 下降（约 165） | 0 error；warning 显著下降（被删报告含 E:/ local-only 链接） | 0 error；warning 约 20–40 | 每批重跑；**error 必须 0**；warning 逐条对照，不得新增（D-12） |
| `render_gv_map.py --check` | current | current | current | 需 `--write` 重渲染后 current | 只有 D3b 触及 g_v_map；禁止手改渲染件（D-03） |
| `pytest tests/test_gv_registry.py tests/test_config_env.py` | 26 passed | 26 passed | 26 passed | 26 passed（若 D3b 改 evidence，须先更新 fixture 期望并单独评审） | registry fixture 钉 `阶段定义`/V542 报告后缀——这两份均 KEEP，不受影响 |
| 两档全量 pytest（CI） | 全绿 | 绿 | 绿 | 绿 | `ci.yml` 无 docs 字符串直接引用（§3.6），CI 经脚本/测试间接钉死；D3b 后必须本地全量复跑 |

**warning 减少来源**（预期，非豁免依据）：被删 reports 中的 `E:/...` local-only 链接（如 `docs/reports/PES_generation_多坐标约束学习_ORCA能力与数学方案_20261001.md` 的 3 条）；`docs/archive` 排除在链接扫描外，D1 删载荷不改变 error/warning 计数（但须确认 manifest 内相对链接被排除逻辑覆盖）。

**required_files 同步删**：D3b 只删 `PES2TS_FullPathNavigationContract_v1` 1 个 docs 键；其余 13 个非 docs config/模板/validator **一个都不动**（GOVERNANCE_CONSTRAINTS §3.8：Tier A 43 条中 KEEP 的 16 条钉死路径原样保留）。

---

## 5. 风险表（按严重度）

| # | 风险 | 触发场景 | 概率 | 影响 | 缓解 |
|---|---|---|---|---|---|
| R1 | 删除被 g_v_map evidence 钉死的文件 → `render_gv_map --check` exit 2 / registry tests 红 | D3b #14–#18 | 中 | 高（CI 红） | 默认不批 #14–#18；批则先改 evidence+重渲染+全量 pytest |
| R2 | 引用断链：docs/README 索引 133 份 live、台账 100 份 | D1–D3 | 高 | 中（check_docs_links error>0） | `LINK_FIX.csv` 562 条按 batch 逐条修；每批 4 门复跑 |
| R3 | 抽结论质量不足/美化负结果 | D2/D3 | 中 | 高（证据价值损失） | 档案树条目验收（A2）：每条 E 条目必须含"负结果"字段；抽完才删 |
| R4 | 治理修订不充分导致删除违规 | Phase 0 | 低 | 高（返工） | A1 双签；ADR-0035 与 G1–G6 同日合入；删除批次 PR 引用 ADR 编号 |
| R5 | 中文路径/长路径命令失误（GBK 控制台） | 执行期 | 中 | 中 | 批次清单由 CSV 生成（§3.0）；`PYTHONIOENCODING=utf-8`；`git rm -- "$_"` 逐条；先 `git ls-files --error-unmatch` 预检 |
| R6 | 误删未跟踪文件（不可 git 恢复） | 执行期 | 低 | 高 | check_deletion (a2) 断言删除集回收均为 git blob；4 条未跟踪文件（本次审计新增）已全部 KEEP |
| R7 | tag/bundle 丢失或未打 tag 就删 | Phase 0 遗漏 | 低 | 高 | §2 快照为 A3 门禁；bundle 校验 `git bundle verify` |
| R8 | archive 载荷退役后快照语义丢失 | D1 | 中 | 中 | 8 份 manifest 永久保留；README 记录 tag 恢复命令 |
| R9 | 门禁配置被指"放宽"（D-19） | D3b | 中 | 高 | ADR-0035 §1.1(6) 合法性框架 + 前后数字对照 + 双签；只删路径键，不改检查逻辑 |
| R10 | 历史契约/ADR 退役后规范追溯断档 | D3 | 中 | 中 | 4 门复验 + 档案树 E01/E02/E06/E09 条目 + tag；建议 owner 只批 #1–#13 |

---

## 6. 绝不删除清单（任何档位都 KEEP，见 `KEEP_SET.csv`）

1. **宪法/治理根**（10）：`README.md`、`AGENT.md`、`docs/PES2TS_开发宪法_v1.md`、`docs/README.md`、`docs/plans/PES2TS_文档修订台账.md`、`docs/plans/PES2TS_阶段定义_G0-G3.md`、`docs/plans/PES2TS_完整开发流方案_整合版_v1.md`、`docs/plans/PES2TS_整体开发与发布方案_v2.md`、`GOVERNANCE_CONSTRAINTS.md`、`doc_audit/20261121/README.md`。
2. **现行合同与现行 ADR**（15）：MoveProcessContract v1、GeometryPolicyContract v2、ReactionIntent Contract v2、`PES2TS_contracts_v1`（schema 总表）、`ACP字段映射_v1`、`数据规范与ACP对接优化方案`（后 3 份为 Lead 裁决 KEEP/CORE_SET C4/C5/C8）、ADR-0010、ADR-0011、ADR-0025、ADR-0026、ADR-0030/0031/0032/0033/0034。
3. **当前战役计划**（12）：V5.3-MP 计划、V5.4.1/2/3 方案与 WP 清单、V5-V7 v3.0、V5.1-R3 v3、训练测试集划分标准等。
4. **关键证据**（13）：`PES2TS_实现报告_20261018`、V541/V542/V543 终报、`V52_任务纠偏审计`、`V53_ROLLOUT_REPORT.md`、8 份 `artifacts/*/model_card.md`。
5. **archive manifest + verification**（8）：`docs/archive/README.md` + 7 份快照 README（manifest 永久保留）。
6. **门禁输入文件**：`docs_consistency_config.json`、`g_v_map.yaml`、`docs/design/PES2TS_G_V_MAPPING_v1_20261018.md`（只允许 `--write` 重生成）、13 份 config/模板/validator（不在 253 清单内，但受 §3.8 保护）。
7. **环境/外部说明**（13）：requirements×3、LICENSE、CONTRIBUTING、examples/scripts/data/outputs README、`path_viewer.html`、`docs/infra/wsl_memory_governance.md`。

---

## 7. 与 task-6 档案树对齐

- 253 条每条都有 `archive_entry`（E01–E17 / X0）；KEEP 中的治理常青层对应 X0，档案树条目覆盖 E01–E17。
- **109 条需抽结论**（`target` 列 = 档案条目编号）与 `ARCHIVE_MAP.csv` 的 `extract_needed=Y`（67 条）取并集：task-6 的 67 是"需人工写抽取内容"的最小集，本方案的 109 还包含 18 条 GCM 与 24 条"结论已明示可迁移"的候选（合并写入同一 E 条目）。执行时以本方案 `target` 列为准，`extract_content` 由档案树条目作者补写。
- **Lead 裁决维持退役的 3 份**（CORE_SET 将同步剔除）：`docs/contracts/PES2TS_G2T_战役合同字段字典_v1.md`、`docs/contracts/PES2TS_PESN_IRC教师数据与模型包合同_v1.md`、`docs/design/PES2TS_PES-C到PES-G架构与迁移方案_v1_20261004.md` → 均须**先抽结论入档案树（E04 / E05 / E11）**后再删，不得先删后补。
- D2/D3 的删除时序 = 档案树条目验收（A2）之后；**不存在"先删后补"路径**。

## 8. 本方案自校验

```powershell
python .slim\deepwork\20261121_cleanup\cleanup\check_deletion.py
```

输出 4 组断言（a 门禁钉死/可恢复/配置 diff 声明；b 链接预演；c 三档单调；d KEEP 互补）+ 统计；全部 PASS 才允许进入 Phase 0 审批。最近一次结果见 §0.1（15/15 PASS，exit 0）。

## 9. 附录：LINK_FIX.csv 使用说明

- 562 行 = 执行 aggressive 档时，KEEP 集活跃文档中指向被删文件的 markdown 硬链接（check_docs_links 扫描面：README/AGENT/docs/**/*.md 排除 archive）。
- 按 `batch` 列过滤：D1 用 `safe` 行、D2 加 `standard` 行、D3 加 `aggressive` 行；每批 PR 必须清零对应行后才复跑门禁。
- 只修正链接/索引行，**不改正文结论**（报告只读原则在退役前的最后一道约束）。

---

> 最后：本方案的所有判定可复现——`gen_candidates.py` 读 4 份审计输入 + ARCHIVE_MAP 生成 CSV；`check_deletion.py` 只读校验。判据不来自"看着旧"，而来自 DISPOSITION 的 blocker/action、GOVERNANCE_CONSTRAINTS 的 52 条钉死路径与 D-01…D-22、REF 计数、自身状态头与档案条目映射。
