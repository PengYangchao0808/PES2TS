# PES2TS 代码地图 v2（按 G 能力 / V 战役）

> 状态：参考（非规范）；规范以[开发宪法](../PES2TS_开发宪法_v1.md)和[阶段定义](../plans/PES2TS_阶段定义_G0-G3.md)为准
> 日期：2026-10-18
> 上游依赖：[ADR-0025](decisions/ADR-0025-G-V双轴映射与战役注册.md)、[G–V 映射](PES2TS_G_V_MAPPING_v1_20261018.md)
> 当前版本：v2.0 / 2026-10-18

## 1. 两条轴

| 轴 | 含义 | 命名落点 |
|---|---|---|
| **G** | 代码/物理能力阶段 | `pes2ts_core/g0`、`g1`、`generation`、`integration`、`pesn` 内的能力名 |
| **V** | 训练/测试战役 | `config/campaigns`、`scripts/campaigns`、`outputs/campaigns`、`docs/reports` 的战役名 |

V5–V7 不是新 G 阶段：V5 → G3.P；V6 → G3.2；V7-A/B → G4.1；V7-C → G4.0→G4.1；M0.5/M6.5 → G2.T。

## 2. 代码根与阶段归属

| 阶段 | 代码根 | 代表模块 | 状态 |
|---|---|---|---|
| G0 | `pes2ts_core/g0/` | fetch/inventory/reader/truth_quarantine/split_freeze | active |
| G1 | `pes2ts_core/g1/` | build/event_coupling/v2_*；新增 `key_atom_audit.py` | active |
| G2 | `pes2ts_core/generation/` | planning/assembly/execution/quality；新增 `planning/key_atom_writer.py` | active |
| G2.T | `pes2ts_core/integration/` | `acp/`；新增 `validation/xtb_ts_verification.py` | active |
| G3.P.0–.3 | `pes2ts_core/pesn/` | contracts/features/models/offline/bundle/inference | active |
| G3.1/G3.2 | `pes2ts_core/pesn/` | inference、`models/control_heads.py`、`offline/control_trainer.py` | research |
| G4.0/G4.1 | `pes2ts_core/generation/control_round.py`、`pes2ts_core/pesn/offline/transitions.py`、`integration/acp/` | control round、transition store、ACP | research |
| G4.2 | 无新增代码根 | 发布证据 + 独立评测 | gate |

## 3. V4 / V5 / V6 / V7 的代码落点

### V4（executed，frozen evidence）

```text
pes2ts_core/pesn/contracts/dual_ts.py
pes2ts_core/pesn/features/dual_ts_graph.py
pes2ts_core/pesn/features/dual_ts_graph_v4.py
pes2ts_core/pesn/models/dual_egnn.py
pes2ts_core/pesn/models/dual_egnn_painn.py
pes2ts_core/pesn/models/dual_ts_policy.py
pes2ts_core/pesn/offline/training/dual_ts_bc.py
pes2ts_core/pesn/offline/training/dual_ts_campaign.py
pes2ts_core/pesn/offline/evaluation/dual_ts_protocols.py
```

规则：不删除、不改语义；只允许状态头/注释与共享 bug 修复（单独 PR + V4 回归）。V4.7 负结果（MSE/零动作 1.132、0/8 组、cos 0.180、near-TS 0.248）保留。

### V5（proposed，G3.P）

```text
pes2ts_core/pesn/contracts/key_atom.py          # ReactionTask/KeyAtomSet/MoveProposal
pes2ts_core/pesn/features/key_atom_spec.py
pes2ts_core/pesn/features/key_atom_graph.py
pes2ts_core/pesn/offline/data/key_atom_teacher.py
pes2ts_core/pesn/offline/data/key_atom_dataset.py
pes2ts_core/pesn/offline/data/pre_relax.py
pes2ts_core/pesn/offline/training/key_atom_bc.py
pes2ts_core/pesn/models/key_atom_mlp.py
pes2ts_core/pesn/models/key_atom_egnn.py
pes2ts_core/pesn/models/key_atom_painn.py
pes2ts_core/pesn/offline/evaluation/key_atom_protocols.py
pes2ts_core/generation/planning/key_atom_writer.py
pes2ts_core/integration/acp/xtb_freeze_request.py
pes2ts_core/integration/acp/xtb_freeze_transport.py
pes2ts_core/integration/validation/xtb_ts_verification.py
```

### V6（proposed，G3.2）

```text
pes2ts_core/pesn/contracts/decision.py
pes2ts_core/pesn/models/v6_progress.py
pes2ts_core/pesn/models/v6_decoder.py
pes2ts_core/pesn/offline/data/decision_dataset.py
pes2ts_core/pesn/offline/splits/decision_split.py
pes2ts_core/pesn/offline/training/v6_train.py
pes2ts_core/pesn/offline/training/decision_train.py
pes2ts_core/pesn/offline/evaluation/decision_protocols.py
```

### V7（proposed，G4.0/G4.1）

```text
pes2ts_core/generation/control_round.py
pes2ts_core/generation/campaign.py
pes2ts_core/integration/acp/
pes2ts_core/pesn/offline/transitions.py
pes2ts_core/pesn/offline/control_trainer.py
pes2ts_core/pesn/contracts/qc_transition.py
scripts/campaigns/run_v7_rollout.py
```

## 4. 导入边界

必须由 `tests/test_gv_registry.py` + `tests/test_stage_boundaries.py`（V5 M0 添加）强制：

1. `g0/g1` 不 import `generation`/`pesn`。
2. `generation.planning` 不 import `pesn`，除非纯 schema（`pes2ts_core.key_atom_contracts` 或 `pesn/contracts/key_atom.py` 的纯数据部分）。
3. `pesn` runtime 不 import `pesn.offline`/`generation`。
4. V5 `key_atom_*` 不 import `dual_ts_*`/`direct_*`。
5. `generation/control_round.py` 是唯一允许的 G4→offline 接缝。
6. V5/V6/V7 主结论不得从 V4 checkpoint 初始化；V4 仅 `comparison_only`。

## 5. 存储分级

| 类别 | 路径 | Git |
|---|---|---|
| 源码/测试/config/schema | `pes2ts_core/`、`tests/`、`config/pesn/`、`config/campaigns/` | tracked |
| 小 manifest/审计 | `data/manifests/*.json`（<1MB）、`data/manifests/v5/`、`v6/`、`v7/` | tracked |
| 大 manifest/派生 | `data/manifests/g1_cluster_report.json`、`data/interim/**` | ignored |
| Teacher shards/parquet | `data/interim/v5/`、`outputs/pesn/v5/` | ignored |
| 权重/bundle | `artifacts/**/weights.pt` | ignored → GitHub Release |
| 模型卡/checksum/gate | `artifacts/pesn/<model>/model_card.md`、`checksums.json`、`docs/reports/` | tracked |
| 运行产物 | `outputs/`（除白名单） | ignored |

## 6. 入口

- `bin/pes2ts` → `pes2ts_core/cli.py`：G0/G1/G2/G2.T/ACP 命令。
- `python -m pes2ts_core.pesn` → PES-N 离线 CLI（inventory/split/build-teacher/train/evaluate/export/predict）。
- V5 新战役入口：`scripts/campaigns/run_v5_sprint.py`（编排），底层训练脚本按 G 能力命名。
- V7 新战役入口：`scripts/campaigns/run_v7_rollout.py`。

## 7. 已知债务（不在本次清理中重构）

1. `pes2ts_core/cli.py` 约 2000 行、45 个 handler；同步后单独拆 `cli/parser.py` + `handlers/`。
2. `generation` ↔ `integration.acp` 双向耦合；`generation/control_round.py` ↔ `pesn.offline` 循环；V5 不新增同类边。
3. `evaluation/`、`generation/quality/` 为空 marker 包；保留命名空间。
4. `utils/truth_guard.py` 只被测试调用；必须进 CI，否则真值隔离没有自动执行点。
5. `direct_*`/`dual_ts_*` 与 V5 并存；删除条件为 V5 过 M2 + 全仓 0 引用 + 旧证据归档。
