# 证据指针层（90_evidence）

> **状态**：Evidence-ReadOnly
> **层级**：L3 证据
> **取代关系**：无
> **上游依赖**：档案条目、原始报告
> **下游消费者**：维护者、审计
> **当前版本**：v1.0 @ 2026-11-21

规则：路径必须存在或可从 `pre-doc-purge-20261121` tag 恢复；报告只读；大文件只存指针。

| 文件 | 内容 |
|---|---|
| `archive_map.csv` | 253 条原文档 → 档案条目映射（含 extract_content 提纲） |
| `negative_results.md` | 跨期负结果总表 N01–N21 |
| `metrics_ledger.csv` | 跨期量化指标账本（表头；数值以条目内指标为准） |
| `decision_index.md` | ADR-0001…0035 × 条目映射 |
