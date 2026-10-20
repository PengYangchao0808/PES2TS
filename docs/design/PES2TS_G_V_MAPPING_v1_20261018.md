# PES2TS G–V 映射（G = 代码/物理能力，V = 训练/测试战役）

> 状态：权威（机器可读源：`config/registry/g_v_map.yaml`）
> 映射版本：`2026-10-18.1`
> 计划修订：`2026-10-20.1`
> 生成命令：`python scripts/render_gv_map.py --write`
> 校验命令：`python scripts/render_gv_map.py --check`

## 0. 读法

- **G** 是代码/物理能力阶段，拥有模块、契约和出口门；不随实验编号改名。
- **V** 是一次训练/测试战役，拥有 config、冻结数据 manifest、run 集合和报告。
- `P` = primary_g，`S` = supporting_g，`D` = data dependency。
- 代码按 G 能力命名（`key_atom_*`、`xtb_freeze_*`），配置/输出/报告按 V 战役命名（`v5_1_*`）。
- 每次 run 的 `gv` 块必须记录 `campaign_id`、`primary_g`、`supporting_g`、`gates`、`map_version`。

## 1. 阶段 × 战役矩阵

| V campaign | G0 | G1 | G2 | G2.T | G3.P.0 | G3.P.1 | G3.P.2 | G3.P.3 | G3.1 | G3.2 | G4.0 | G4.1 | G4.2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V4.2 | S | S |  |  |  |  | P |  |  |  |  |  |  |
| V4.3 | S | S |  |  |  |  | P |  |  |  |  |  |  |
| V4.4 | S | S |  |  |  |  | P |  |  |  |  |  |  |
| V4.5 | S | S |  |  |  |  | P |  |  |  |  |  |  |
| V4.6 | S | S |  |  |  |  | P |  |  |  |  |  |  |
| V4.7 | S | S |  |  |  |  | S | P |  |  |  |  |  |
| V5.0 | S | S |  | S | S | P |  |  |  |  |  |  |  |
| V5.1 | S | S |  |  |  |  | P |  |  |  |  |  |  |
| V5.1-R2 | S | S |  |  | S | S | P | S |  |  |  |  |  |
| V5.2 | S | S |  |  | S | P |  | S |  |  |  |  |  |
| V5.3 | S | S |  |  |  | S | P | S |  |  |  |  |  |
| V5.4 | S | S |  |  |  | S | P | S |  |  |  |  |  |
| V5_SPRINT | S | S |  |  | P | S | S | S |  |  |  |  |  |
| V5_1A_SPRINT | S | S |  |  | S | S | P | S |  |  |  |  |  |
| V6.0 | D | D |  |  |  |  |  | S |  | P |  |  |  |
| V6.1 | D | D |  |  |  |  |  | S |  | P |  |  |  |
| V6.2 | D | D |  |  |  |  |  | S |  | P |  |  |  |
| V7-A | D | D |  | S |  |  |  |  |  | S | S | P |  |
| V7-B | D | D |  | S |  |  |  |  |  | S |  | P |  |
| V7-C | D | D |  | S |  |  |  |  |  | S | P | S |  |

## 2. 战役登记

| V | 状态 | primary G | supporting G | config | data manifest | gates |
|---|---|---|---|---|---|---|
| V4.2 | executed | G3.P.2 | G0, G1 | `config/pesn/dual_ts_observations_16x1_v1.yaml` | `data/manifests/split_manifest.json` | V4-MSE-zero, V4-cos |
| V4.3 | executed | G3.P.2 | G0, G1 | `config/pesn/dual_ts_observations_16x1_v1.yaml` | `data/manifests/split_manifest.json` | V4-MSE-zero, V4-cos |
| V4.4 | executed | G3.P.2 | G0, G1 | `config/pesn/dual_ts_observations_16x1_v1.yaml` | `data/manifests/split_manifest.json` | V4-MSE-zero, V4-cos |
| V4.5 | executed | G3.P.2 | G0, G1 | `config/pesn/dual_ts_observations_16x1_v1.yaml` | `data/manifests/split_manifest.json` | V4-MSE-zero, V4-cos |
| V4.6 | executed | G3.P.2 | G0, G1 | `config/pesn/dual_ts_observations_16x1_v1.yaml` | `data/manifests/split_manifest.json` | V4-MSE-zero, V4-cos |
| V4.7 | executed | G3.P.3 | G3.P.2, G0, G1 | `config/pesn/dual_ts_observations_16x1_v1.yaml` | `data/manifests/split_manifest.json` | V4.7-MSE-zero, V4.7-groups, V4.7-cos, V4.7-nearTS-cos |
| V5.0 | proposed | G3.P.1 | G0, G1, G3.P.0, G2.T | `config/campaigns/v5_0_data_math.yaml` | `data/manifests/v5/split_manifest.json` | V5-G1, M0.5-oracle |
| V5.1 | archived | G3.P.2 | G0, G1 | `config/campaigns/v5_1_architecture.yaml` | `data/manifests/v5/split_manifest.json` | V5-G1, V5-G2, V5-G3, V5-G4 |
| V5.1-R2 | proposed | G3.P.2 | G3.P.0, G3.P.1, G3.P.3, G0, G1 | `config/campaigns/v5_1_r2.yaml` | `data/manifests/v5/split_manifest.json` | V5-G1, V5.1-R2-loss, V5.1-R2-data |
| V5.2 | proposed | G3.P.1 | G3.P.0, G3.P.3, G0, G1 | `config/campaigns/v5_2_key_atom_contract.yaml` | `data/manifests/v5/split_manifest.json` | V5.2-contract, V5.2-labels, V5.2-identifiability |
| V5.3 | proposed | G3.P.2 | G3.P.1, G3.P.3, G0, G1 | `config/campaigns/v5_3_geometry.yaml` | `data/manifests/v5/split_manifest.json` | V5-G1, V5.3-move, V5.3-stop, V5.3-generalization |
| V5.4 | proposed | G3.P.2 | G3.P.1, G3.P.3, G0, G1 | `config/campaigns/v5_4_rollout.yaml` | `data/manifests/v5/split_manifest.json` | V5-G1, V5.4-field, V5.4-rollout, V5.4-stop |
| V5_SPRINT | executed | G3.P.0 | G3.P.1, G3.P.2, G3.P.3, G0, G1 | `config/campaigns/v5_sprint.yaml` | `data/manifests/v5/split_manifest.json` | V5-G1, V5-G2, V5-G3, V5-G4 |
| V5_1A_SPRINT | executed | G3.P.2 | G3.P.0, G3.P.1, G3.P.3, G0, G1 | `config/campaigns/v5_1_sprint.yaml` | `data/manifests/v5/split_manifest.json` | V5-G1, V5-G2, V5-G3, V5-G4 |
| V6.0 | archived | G3.2 | G3.P.3 | `config/campaigns/v6_0_data.yaml` | `data/manifests/v6/split_manifest.json` | V6-OOF, V6-leakage |
| V6.1 | archived | G3.2 | G3.P.3 | `config/campaigns/v6_1_progress.yaml` | `data/manifests/v6/split_manifest.json` | V6-progress, V6-conformal |
| V6.2 | archived | G3.2 | G3.P.3 | `config/campaigns/v6_2_decoder.yaml` | `data/manifests/v6/split_manifest.json` | V6-decoder, V6-stop |
| V7-A | archived | G4.1 | G2.T, G3.2, G4.0 | `config/campaigns/v7a_rollout.yaml` | `data/manifests/v7/rollout_cohort.json` | V7-verified-TS, V7-cost |
| V7-B | archived | G4.1 | G3.2, G2.T | `config/campaigns/v7b_calibration.yaml` | `data/manifests/v7/rollout_cohort.json` | V7-calibration, V7-cost |
| V7-C | archived | G4.0 | G4.1, G2.T, G3.2 | `config/campaigns/v7c_rl.yaml` | `data/manifests/v7/transition_cohort.json` | V7-RL, V7-cost |

## 3. G 阶段登记

| G | 能力 | code globs | 阶段文档 |
|---|---|---|---|
| G0 | Data base, truth quarantine and split freeze | `pes2ts_core/g0/**/*.py` | `docs/plans/PES2TS_阶段定义_G0-G3.md` |
| G1 | Chemical task, graph audit and legal action compilation | `pes2ts_core/g1/**/*.py` | `docs/plans/PES2TS_阶段定义_G0-G3.md` |
| G2 | Classic physical closure, Shield and replayable baseline | `pes2ts_core/generation/**/*.py` | `docs/plans/PES2TS_阶段定义_G0-G3.md` |
| G2.T | G2 24-case test and strict physical verification | `pes2ts_core/integration/**/*.py`<br>`pes2ts_core/generation/**/*.py` | `docs/plans/PES2TS_阶段定义_G0-G3.md` |
| G3.P.0 | Source/identity/split reconciliation for PES-N | `pes2ts_core/pesn/contracts/inputs.py`<br>`pes2ts_core/pesn/offline/splits/**/*.py` | `docs/plans/PES2TS_阶段定义_G0-G3.md` |
| G3.P.1 | Teacher state/label construction and deployable features | `pes2ts_core/pesn/offline/data/**/*.py`<br>`pes2ts_core/pesn/features/**/*.py` | `docs/plans/PES2TS_阶段定义_G0-G3.md` |
| G3.P.2 | Small-graph encoder, multi-task training and scaling ablations | `pes2ts_core/pesn/models/**/*.py`<br>`pes2ts_core/pesn/offline/training/**/*.py` | `docs/plans/PES2TS_阶段定义_G0-G3.md` |
| G3.P.3 | Frozen weights/gates, sealed evaluation and research bundle | `pes2ts_core/pesn/offline/evaluation/**/*.py`<br>`pes2ts_core/pesn/bundle/**/*.py` | `docs/plans/PES2TS_阶段定义_G0-G3.md` |
| G3.1 | Single-point p_seed, refinement cost and Q_STOP supervision | `pes2ts_core/pesn/inference/**/*.py`<br>`pes2ts_core/pesn/models/**/*.py` | `docs/plans/PES2TS_阶段定义_G0-G3.md` |
| G3.2 | Context, action imitation and bounded Classic residual | `pes2ts_core/pesn/contracts/control.py`<br>`pes2ts_core/pesn/models/control_heads.py`<br>`pes2ts_core/pesn/offline/control_trainer.py` | `docs/plans/PES2TS_阶段定义_G0-G3.md` |
| G4.0 | Conservative offline value/Q on compatible transitions | `pes2ts_core/pesn/contracts/control.py`<br>`pes2ts_core/pesn/offline/transitions.py`<br>`pes2ts_core/pesn/offline/control_trainer.py` | `docs/plans/PES2TS_阶段定义_G0-G3.md` |
| G4.1 | Frozen-model real QC closed loop and in-train collection | `pes2ts_core/generation/control_round.py`<br>`pes2ts_core/integration/**/*.py` | `docs/plans/PES2TS_阶段定义_G0-G3.md` |
| G4.2 | Full-system, cross-generator and release evidence | — | `docs/plans/PES2TS_阶段定义_G0-G3.md` |

## 4. 规则

1. `proposed` 允许代码/配置/数据路径尚不存在；`active`/`executed` 必须存在，由 `validate_gv_map` 强制。
2. 同一 V 战役只有一个 `primary_g`；跨阶段工作放 `supporting_g`，不得把 V7 执行代码塞进 `pesn/offline`。
3. V4 `dual_ts`/`direct` 只作 `comparison_only`，不得被 V5/V6/V7 主结论复用为初始化权重。
4. 任何 run record 的 `gv.campaign_id` 必须能在本表解析；`map_version` 必须与当前映射一致。
5. 新增/修改 V 战役先改 YAML，再跑 `--write`，最后提交生成文档；不要手改本文件。
