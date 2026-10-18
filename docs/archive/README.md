# docs/archive/ — 版本快照索引

`docs/archive/` 保存**文档修订时点的原始快照**，不是“已完成文档的墓地”。快照内保留当时的相对链接和文件名；这些链接按快照时点解释，允许在仓库中失效。`docs/README.md` 的 live-doc 链接检查应排除本目录。

| 快照 | 时点 | 范围 | 主要入口 |
|---|---|---|---|
| `20261005_pre_pesn/` | 2026-10-05 | PES-N 产品归属/ADR-0014 前的入口快照 | `*.snapshot.txt`、manifest、revision_check |
| `20261006_pre_g3p/` | 2026-10-06 | G3.P 并行教师预训练立项前 | 9 份入口快照 + manifest + verification |
| `20261007_pre_light_policy/` | 2026-10-07 | 轻量控制/RL 衔接前 | 镜像树（docs/plans、AGENT.md、source_README） |
| `20261007_pre_orca_xtb_pilot/` | 2026-10-07 | AB4 ORCA-xTB 试点前 | `files/` 子树 + README |
| `20261007_pre_predictor_scope/` | 2026-10-07 | 预测器代码范围/模块划分前 | 13 份文档快照 + manifest + verification |
| `20261007_pre_unified_residual_rl/` | 2026-10-07 | 统一残差决策/RL 入口前 | 9 份快照 + manifest + root README |
| `20261008_pre_scale20k_rl2k/` | 2026-10-08 | IRC20k/RL2000 扩量前 | 4 份入口 + source_hashes |

## 规则

1. **只读**：快照内容不修改；需要修订时新增当前文档，并在台账登记取代关系。
2. **不移动**：本目录是版本快照，不是“过时文档整理区”；移动会破坏 `docs/PES2TS_开发宪法_v1.md`、`docs/plans/PES2TS_文档修订台账.md` 等处的引用。
3. **链接检查排除**：`scripts/check_docs_links.py` 必须跳过 `docs/archive/**`。
4. **命名**：新增快照用 `YYYYMMDD_pre_<变更主题>/`，并携带 manifest/哈希。
