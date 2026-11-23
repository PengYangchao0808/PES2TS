# 迁移记录（MIGRATION_LOG）— 2026-11-21 批次模板

> **状态**：Evidence-ReadOnly（执行流水模板；本文件只允许追加，不允许改写已发生的记录）
> **层级**：L4 运营（归档执行流水）
> **取代关系**：无
> **上游依赖**：[开发文档规范化归档方案 v2.0](../../PES2TS_开发文档规范化归档方案_v2_20261121.md) §9、[BATCH_PLAN.md](BATCH_PLAN.md)
> **下游消费者**：仓库所有者、复核者、下次审计
> **当前版本**：v1 @ 2026-11-21（模板；**当前无任何迁移记录**——本次审计未执行任何 `git mv`）

## 使用规则

1. **只在真正执行迁移（B1/B2）时追加行**；未执行的批次不要预填，保持"零记录"即证明"未移动任何文件"。
2. 每行必须能回答：何时、哪批、从哪到哪、修了几处引用、门禁结果、谁执行的。
3. 记录**不可改写**：若迁移被回滚，追加一行 `reverted`，不删除原行。
4. 与 git 的关系：本表是"人可读的意图与验收记录"，`git log --follow` 是"机器可读的字节历史"，两者必须一致。

## 记录表

| 日期 | 批次 | 旧路径 | 新路径 | 引用修正数 | 门禁结果 | 执行人 | 备注 |
|---|---|---|---:|---|---|---|---|
| — | — | — | — | — | — | — | **2026-11-21 审计未执行任何迁移；B1 候选见 BATCH_PLAN.md** |

## 回滚命令（若某批失败）

```powershell
# 本批未提交时：reset --hard 同时重置 index，才能撤销已 staged 的 git mv
git reset --hard HEAD
git clean -fd docs/superseded   # 仅清理本批新增且未跟踪的文件
# 已提交时：git revert <commit>
python infra/governance/check_docs_links.py
python infra/governance/docs_consistency.py --strict
```
