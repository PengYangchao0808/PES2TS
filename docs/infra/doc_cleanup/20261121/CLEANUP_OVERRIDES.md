# CLEANUP_OVERRIDES — Lead 最终裁决覆盖（2026-11-21）

> 本文件是对 `CLEANUP_PLAN.md` 与 `CLEANUP_CANDIDATES.csv` 的**最终裁决覆盖件**：
> 在 task-7 机器判定之上，Lead 追加两项人工裁决并重算三档数字。**以本文件为准**。

## 1. 覆盖项

| # | 裁决 | 对象 | 影响 |
|---|---|---|---|
| O1 | **ADR 全留** | `docs/design/decisions/ADR-0001…0034`（34 份） | 25 份从删除集移入 KEEP；其中 safe 档 4、standard 档 9、aggressive 档 12（含原 GCM 3） |
| O2 | **核心集 5 份改判 KEEP** | `PES2TS_contracts_v1.md`、`PES2TS_ACP字段映射_v1.md`、`PES2TS_数据规范与ACP对接优化方案.md`、`PES2TS_CODE_MAP_v2_20261018.md`、`PES2TS_Classic图论驱动与xTB原生约束扫描设计_v1_20261006.md` | 已在 task-7 `OVERRIDE_KEEP` 落地 |
| O3 | **保持退役 3 份** | G2T 字段字典、PESN_IRC 教师合同、PES-C→PES-G 架构迁移 | 先抽结论入档案树 E04/E05/E11 再退役 |

**裁决理由（ADR 全留）**：① ADR 是“为什么否掉别的方案”的唯一载体，档案树每条目的「关键决策」字段必须引用 ADR 编号；② 成本极低（只读记录，无正文维护义务）；③ 编号=决策指纹，删除/合并会让现行文档 100+ 处 `ADR-00xx` 引用失锚；④ 门禁 `required_files` 含 ADR-0030…0034、`historical` 含 0021/0028/0029。

## 2. 覆盖后最终数字（权威）

| 项 | 覆盖前（task-7 初稿） | **覆盖后（最终）** |
|---|---:|---:|
| KEEP（不删） | 76 | **101** |
| safe 档删除 | 67 | **63** |
| standard 档累计删除 | 129 | **116** |
| aggressive 档累计删除 | 177 | **152** |
| 需先抽结论（ARCHIVE + GCM） | 109 | **88** |
| GATE_CONFIG_MUST_CHANGE | 18 | **15** |

**决策分布**：`ARCHIVE_EXTRACT_THEN_DELETE` 73、`DELETE` 56、`GATE_CONFIG_MUST_CHANGE` 15、`KEEP` 101、`MERGE_INTO` 8

**按 Phase**：D0/KEEP 101；D1/DELETE 56；D1/MERGE_INTO 7；D2/ARCHIVE_EXTRACT_THEN_DELETE 52；D2/MERGE_INTO 1；D3/ARCHIVE_EXTRACT_THEN_DELETE 21；D3/GATE_CONFIG_MUST_CHANGE 15

**保留集构成**（101 份）：核心受管单元 44 组（≈98 份，含 34 ADR + 8 模型卡）+ `docs/archive/**` manifest/README 8 份 + 其余门禁证据/索引。

## 3. 对 CLEANUP_PLAN.md 的修正点（执行时以本表为准）

| 原计划位置 | 原数字 | 修正后 |
|---|---|---|
| §0 / §0.1 三档删除数 | 67 / 129 / 177 | **63 / 116 / 152** |
| §3.1 D1 | 67 条 | **63** 条 |
| §3.2 D2 | +65 → 累计 132 | **+53 → 累计 116** |
| §3.3 D3 | +48 → 累计 182（含 18 GCM） | **+36 → 累计 152（含 15 GCM）** |
| §3.3 D3a/D3b | 32 + 18 | **21 + 15** |
| §6 绝不删除清单 | 76 条 KEEP | **101** 条 KEEP（含 34 ADR） |
| §7 抽取覆盖 | 109 条 | **88** 条 |

## 4. 执行前必须重跑的校验

```powershell
python .slim/deepwork/20261121_cleanup/cleanup/check_deletion.py
# 期望：全部 PASS；三档 = 63 / 116 / 152；KEEP = 101
```

> 备注：`check_deletion.py` 中针对 GCM 的提示文案（“18 条”）为初稿口径，实际以本文件的 15 条为准；断言逻辑按 CSV 动态计算，不受影响。
