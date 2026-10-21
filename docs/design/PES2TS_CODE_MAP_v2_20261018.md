# PES2TS 代码地图 v2（按 G 能力 / V 战役）

> 状态：参考（非规范）；规范以[开发宪法](../PES2TS_开发宪法_v1.md)和[阶段定义](../plans/PES2TS_阶段定义_G0-G3.md)为准
> 日期：2026-10-18（原文）；架构修订：2026-10-20
> 上游依赖：[ADR-0025](decisions/ADR-0025-G-V双轴映射与战役注册.md)、[G–V 映射](PES2TS_G_V_MAPPING_v1_20261018.md)、[V5–V7 v3.0](../plans/PES2TS_V5-V7_反应意图条件化与低维生长方案_v3.0_20261020.md)、[ReactionIntent Contract v1](../contracts/PES2TS_ReactionIntent_Contract_v1.md)
> 当前版本：v2.2 / 2026-10-20（V5/V6/V7 按 v3.0 reaction-intent 修订；v2.0 保留为历史）
> 2026-10-21 v2.1：V5.2+ 任务语义按 ADR-0030/FullPathNavigationContract v1 —— Move 全路径、Endpoint Stop；新增/规划的 v52r 模块见下

## 1. 两条轴

| 轴 | 含义 | 命名落点 |
|---|---|---|
| **G** | 代码/物理能力阶段 | `pes2ts_core/g0`、`g1`、`generation`、`integration`、`pesn` 内的能力名 |
| **V** | 训练/测试战役 | `config/campaigns`、`scripts/campaigns`、`outputs/campaigns`、`docs/reports` 的战役名 |

V5–V7 不是新 G 阶段：V5 → G3.P；V6 → G3.2 + G2.T（xTB 探测/验证）；V7 → G4.0/G4.1；M0.5/M6.5 → G2.T。V5.1-R2 仍属 G3.P.2；V5.2 起新合同仍归 G3.P.0–.3。

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

### V5.1-R3 / V5.2–V5.4（proposed，G3.P；reaction-intent v3.0）

**V5.1-R3（局部步进基线；无 branch；signed intent）**

```text
pes2ts_core/pesn/contracts/reaction_intent.py          # planned: G_rxn / signed graph edit / CV spec
pes2ts_core/pesn/features/key_atom_spec_v3.py          # planned
pes2ts_core/pesn/features/key_atom_graph_v3.py         # planned: intent-conditioned features
pes2ts_core/pesn/models/key_atom_egnn_v3.py            # planned: fixed F3 superset + masks
pes2ts_core/pesn/offline/training/key_atom_bc_v3.py    # planned: strong/dead + sample weights
pes2ts_core/pesn/offline/evaluation/key_atom_protocols.py
scripts/campaigns/run_v5_1_r3.py                       # planned
config/campaigns/v5_1_r3.yaml                          # planned
```

**V5.2–V5.4（reaction intent + CV/Jacobian + progress/stop + growth）**

```text
pes2ts_core/pesn/contracts/reaction_intent.py          # G_rxn, CV, progress/stop semantics
pes2ts_core/pesn/features/reaction_coords.py           # planned: CV / Jacobian lift
pes2ts_core/pesn/models/key_atom_egnn_v3.py            # progress/stop heads (V5.2)
pes2ts_core/pesn/offline/data/key_atom_teacher.py      # T0/T1/T2 + stop labels
pes2ts_core/pesn/offline/data/key_atom_dataset_v3.py
pes2ts_core/pesn/offline/training/key_atom_bc_v3.py
pes2ts_core/pesn/offline/evaluation/key_atom_protocols.py
scripts/campaigns/run_v5_2_progress_stop.py             # planned
scripts/campaigns/run_v5_3_growth.py                    # planned
scripts/campaigns/run_v5_4_any_point.py                 # planned
```

**V5.2-R（planned，G3.P；ADR-0030 全路径导航与端点停止）**

```text
pes2ts_core/pesn/offline/data/key_atom_dataset_v3.py    # existing (endpoint labels feed from here)
pes2ts_core/pesn/offline/data/key_atom_endpoint_labels_v1.py  # planned: d_end 0.10/0.30 endpoint stop labels
pes2ts_core/pesn/models/key_atom_heads_v52r.py          # planned: EndpointStopHead (+ ProgressHeadV3 kept)
pes2ts_core/pesn/models/key_atom_loss_v52r_v3.py        # planned: L_move + λ_end·L_end + λ_ord·L_ord + λ_q·L_q
scripts/campaigns/r3_v52r_run.py                        # planned: D0/D1/T1/R entry
scripts/agent_ops/docs_consistency.py                   # docs gate: python scripts/agent_ops/docs_consistency.py --strict
```

- 遗留符号 `TSStopHeadV3` / `ts_stop_loss_v52` / `stop_label_from_distance_v52` = **LEGACY_TS_STOP**（仅旧 checkpoint 加载/ablation，不用于新训练）；落点 `models/key_atom_heads_v3.py`、`models/key_atom_loss_v52_v3.py`、`offline/data/key_atom_dataset_v3.py`。
- `pes2ts_core/pesn/offline/data/key_atom_teacher.py` docstring 中 "TS-terminal" 为**遗留措辞**（R3 builder 实际传入完整曲线），见 [任务纠偏审计](../reports/PES2TS_V52_任务纠偏审计_20261021.md) §2。

**V5.3-MP（proposed，G3.P；ADR-0031 MOVE–PROCESS 统一学习）**

```text
pes2ts_core/pesn/models/key_atom_heads_v3.py            # existing: ProgressHeadV3 kept; add ReactionProcessHead + KeyAtomV53MPModel
pes2ts_core/pesn/models/key_atom_loss_v53mp_v3.py       # planned: L_Move + L_Process (ranking/progress/endpoint, independent masks)
pes2ts_core/pesn/offline/training/key_atom_v53mp_train_v3.py  # planned: staged trainer, arms A/B/C, Move guard, gradient monitor
pes2ts_core/pesn/offline/data/key_atom_dataset_v3.py    # fix: load all statuses; MOVE/PROCESS independent eligibility
pes2ts_core/pesn/offline/data/key_atom_endpoint_labels_v1.py  # existing: keep as diagnostic + high-confidence endpoint labels
scripts/campaigns/r3_v53mp_run.py                       # planned: S0-S4 driver
scripts/agent_ops/data_gate/r3_build_tables.py          # extend: 2k/5k/20k cohort six-table build
pes2ts_core/pesn/offline/publishing/export.py           # planned: pes2ts_move_process_bundle_v1
config/campaigns/pes2ts_v53mp_20261021.yaml             # planned: S0-S4, arms, authorization, freeze hashes
```

- 命名：`V5.3-ENG` = 30h rollout/relax 工程闭环（historical）；`V5.3-MP` = 本战役。
- 旧 `V5.2-R` 已 **Closed（D2-G STOP）**；其模块 `key_atom_heads_v52r.py` / `r3_v52r_run.py` 为 planned 历史，不进入新训练。
- 旧 `TSStopHeadV3` 继续标 **LEGACY_TS_STOP**；新 PROCESS 头不得复用该名字。

### V6（proposed，G3.2 + G2.T）

```text
pes2ts_core/pesn/models/v6_physics_mlp.py               # planned: 单点 xTB E/g 校正
pes2ts_core/pesn/models/v6_physics_cnn.py               # planned: 3–5 点局部曲线
pes2ts_core/pesn/offline/data/xtb_probe.py              # planned: 受限拉伸探测
pes2ts_core/pesn/offline/training/v6_train.py           # planned
pes2ts_core/pesn/offline/evaluation/physics_protocols.py  # planned
pes2ts_core/integration/acp/xtb_freeze_request.py
pes2ts_core/integration/acp/xtb_freeze_transport.py
pes2ts_core/integration/validation/xtb_ts_verification.py
scripts/campaigns/run_v6_1_mlp.py                       # planned
scripts/campaigns/run_v6_2_cnn.py                       # planned
scripts/campaigns/run_v6_3_verify.py                    # planned
```

### V7（proposed，G4.0/G4.1）

```text
pes2ts_core/generation/control_round.py
pes2ts_core/generation/campaign.py
pes2ts_core/integration/acp/
pes2ts_core/pesn/offline/transitions.py
pes2ts_core/pesn/offline/control_trainer.py
pes2ts_core/pesn/contracts/qc_transition.py
scripts/campaigns/run_v7_1_env.py                       # planned
scripts/campaigns/run_v7_2_offline_rl.py                # planned
scripts/campaigns/run_v7_3_online_rl.py                 # planned
```

> 旧 v1.2 的 `v6_progress.py`、`v6_decoder.py`、`decision_*` 与 V7-A/B/C 路线已于 2026-10-20 归档，不再作为当前代码落点；保留为历史设计。

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
- V5.1-R3 新战役入口：`scripts/campaigns/run_v5_1_r3.py`（planned）。
- V5.2–V5.4 新战役入口：`scripts/campaigns/run_v5_2_progress_stop.py`、`run_v5_3_growth.py`、`run_v5_4_any_point.py`（planned）。
- V6/V7 新战役入口：`scripts/campaigns/run_v6_1_mlp.py`、`run_v6_2_cnn.py`、`run_v6_3_verify.py`、`run_v7_1_env.py`、`run_v7_2_offline_rl.py`、`run_v7_3_online_rl.py`（planned）。

## 7. 已知债务（不在本次清理中重构）

1. `pes2ts_core/cli.py` 约 2000 行、45 个 handler；同步后单独拆 `cli/parser.py` + `handlers/`。
2. `generation` ↔ `integration.acp` 双向耦合；`generation/control_round.py` ↔ `pesn.offline` 循环；V5 不新增同类边。
3. `evaluation/`、`generation/quality/` 为空 marker 包；保留命名空间。
4. `utils/truth_guard.py` 只被测试调用；必须进 CI，否则真值隔离没有自动执行点。
5. `direct_*`/`dual_ts_*` 与 V5 并存；删除条件为 V5 过 M2 + 全仓 0 引用 + 旧证据归档。
