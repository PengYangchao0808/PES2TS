# PES2TS V5.4.1 Embedding 50K 实施与训练报告（2026-10-29，只读最终报告）

> 状态：**Executed-Frozen**（只读最终报告；报告内容永不改写，后续仅可追加"当前阶段/作废"指针）
> 战役：`V5.4.1` / `PES2TS_V541_EMBED_50K`（[G–V registry](../design/PES2TS_G_V_MAPPING_v1_20261018.md)；schema `pes2ts_v541_campaign_v1`）
> 规划：[V5.4.1 Embedding 50K 详细阶段实施方案 v1.2](../plans/PES2TS_V5.4.1_Embedding_50K_详细阶段实施方案_v1.0_20261027.md)（§3/§6/§7/§9/§11–§17）；治理：[ADR-0032](../design/decisions/ADR-0032-V5.4.1-Embedding接口扩展与预注册治理.md)、ADR-0031、MoveProcessContract v1
> 冻结证据根：`outputs/pesn/v541_embed_50k/`（plan/audits/smoke/profile/runs/reports/exports；run 账本 `run_records.jsonl` 50 行追加写）
> 结论先行：**WP0–WP7 全部完成；E*=E0 留任**。T0-R matched E0×3（seeds 17/29/43）与 E1/E2/E3 seed17 均完成 37,900 updates 正式训练；**E1/E2 因 H2 sign guard、E3 因 H2 sign+dead guard 判 `scientific_failure`**（尽管三者 inner-val `provisional_H3=true`，H2 硬门不可被 PROCESS 增益豁免）；**E4 因 core-time ratio 3.6455 > 2.0 `gate_skipped`**（成本门，非资源失败）；**E5/T3/T4 均 `gate_skipped`**（T4 记 `no_new_embedding_gain`）。无候选补 seed 29/43，无候选三 seed CI。正式训练新增 **6 次**（3+3，`N_core` 按 §9.1 全额 T0-R 分支），单路、无 swap、OOM kill=0、`new_QC_calls=0`、**confirm 从未读取**。

---

## 1. 实施模块（WP0–WP4 交付；代码 SHA 绑定见 `reports/wp7_freeze.json`）

| 责任 | 模块 |
|---|---|
| WP0 治理 | `config/campaigns/pes2ts_v541_embed_50k.yaml`、`config/campaigns/v541_templates/*`、`scripts/agent_ops/v541_config_schema.py`（独立 schema validator，12 项检查含 T3 预注册）、G–V registry `V5.4.1` 条目 |
| WP1 评估器 | `pes2ts_core/pesn/offline/evaluation/pair_ownership_v3.py`（共享 pair 归属/映射工具）；`audits/evaluator_before_after.json`（C×3 inner-val/dev before/after）、`audits/dev_baseline_sealed/`（密封至 WP7）、`audits/verify_report.json`（账本/封存/cgroup 全过） |
| WP2 模型 | `pes2ts_core/pesn/models/v541/{interfaces,embedding_registry,model}.py` + `embeddings/{gin_e_chemical,pna_chemical,schnet_spatial,angle_mpnn,dual_stream,chemical_common,geometric_common}.py` + `feature_providers/{legal_neighbors,angle_triplets,common}.py`；CPU bitwise + GPU 1e-6/1e-5 parity（`audits/v53_parity{,_gpu,_gpu_bitwise_probe}.json`） |
| WP4 训练器 | `pes2ts_core/pesn/offline/training/{v541_train,v541_identity,v541_monitor,v541_strata,v541_degree_stats}.py`；stream 视图 `pes2ts_core/pesn/offline/data/key_atom_stream_view.py`；runner/plan/profiling/审计/评估/rollout/导出 `scripts/campaigns/v541_{run,runplan,parity,legality_audit,baseline_repair,dev_evaluate,rollout,export_bundle,provenance}.py` |
| WP6/WP7 报表 | `reports/{single_seed_screen,three_seed_paired_ci,dev_frozen_evaluation,rollout_comparison,compute_pareto,failure_and_censoring,benchmark_summary,wp7_freeze}.json(.csv)` + `exports/embedding_champion_bundle/`（8 文件） |

冻结计划实例：`plan/runplan.pin.json`（code_commit `1d8b7972…`、common_protocol `2b9ce1ed…`、config `464c1f6f…`）、`plan/preregistration_signoff.json`、`plan/budget_policy.json`。

## 2. 正式 run 与成本（全部 `objective=unified_move_process`，37,900 updates，单路 fp32）

| run_id | embedding | seed | attempt | wall (s) | core (s) | outcome |
|---|---|---|---|---|---|---|
| V541_T0R_E0_S17 | E0 | 17 | 1 | 3,540.33 | 2,124.81（E0 基准，w1 ratio 1.0） | done（best_update 37,000） |
| V541_T0R_E0_S29 | E0 | 29 | 1 | 3,278.81 | 同上 | done（best_update 36,400） |
| V541_T0R_E0_S43 | E0 | 43 | 1 | 3,337.53 | 同上 | done（best_update 37,800） |
| V541_E1_S17 | E1 | 17 | 4 | 3,916.31 | 2,118.06（ratio 0.9914） | scientific_failure（H2 sign） |
| V541_E2_S17 | E2 | 17 | 4 | 4,042.49 | 2,197.65（ratio 1.0319） | scientific_failure（H2 sign） |
| V541_E3_S17 | E3 | 17 | 4 | 9,827.64 | 4,235.71（ratio 1.9967） | scientific_failure（H2 sign+dead） |

- 正式训练合计 GPU 墙钟 **27,943 s ≈ 7.76 h**；E1–E3 的 attempt 1–2 为 200-update smoke、attempt 3 为 `profile_only` 测速（均留账，不计入正式训练）。
- E4 未进入正式训练：`profile_only` core 7,752.43 s → ratio **3.6455 > 2.0**，按 §11.2 hard_ratio 记 `gate_skipped`（成本结果保留、科学上不晋级，**不是** `resource_failure`）。E5 记 `parent_E4_cost_gate_failed`。
- 峰值资源（formal）：GPU max allocated ≈ 207–214 MB；host peak RSS 9.12–9.42 GB；cgroup peak anon 4.67 GB；OOM kill=0、swap=0（`audits/h1_resource.json`）。

## 3. 单 seed 初筛（inner-val only；dev 不参与排序，§9.3）

匹配基线 = 同 seed T0-R E0（seed17：AUPRC 0.27424、q_corr 0.31802、sign 0.83333、dead 0.08801）。

| 候选 | ΔAUPRC | Δq corr | cost ratio | H0/H1 | H2 守卫 | outcome |
|---|---|---|---|---|---|---|
| E1 | +0.07874 | +0.10188 | 0.9914 | ✓/✓ | **sign_drop_abs 0.03333 > 0.01**（dead +0.00410 ✓、RMSE −0.73% ✓、方向 ✓） | scientific_failure |
| E2 | +0.05162 | +0.10544 | 1.0319 | ✓/✓ | **sign_drop_abs 0.03333 > 0.01** | scientific_failure |
| E3 | +0.46004 | +0.45043 | 1.9967 | ✓/✓ | **sign 0.03333 > 0.01 + dead +0.01551 > 0.01** | scientific_failure |

- 三者 `provisional_H3=true`（单 seed 初筛口径），但 §11.1 H2 是硬门：**PROCESS 增益不得豁免 MOVE 坍缩**（§15），候选全部失去晋级资格。
- `n_selected=0` → **T3 gate_skipped**（预注册的 10 个 E1–E5×{29,43} 条件 run 全部未激活，0 次补 seed 训练，`N_core` 未膨胀）；**T4 gate_skipped(`no_new_embedding_gain`)**——无候选存活时容量/深度诊断不能改变裁决，容量/深度混杂**未闭合**已如实声明。
- 无候选三 seed：`reports/three_seed_paired_ci.json` 记 `status=gate_skipped`、`n_selected=0`、`bootstrap.executed=false`、`seed_p_value_claim=false`——**不编造任何三 seed CI 或 p 值**。E0 的三 seed 配对基线（checkpoint SHA 三枚）已冻结供后续使用。

## 4. E0 留任与 frozen dev 评估（§10；dev 只评估冻结 E0×3）

`E_star=E0`。`audits/e0_reuse_decision.json`：`decision=retrain_e0_x3`——loader 训练路径证书单次 `EVIDENCE_MISSING`（probe sid 不在 label cache；§8.1 只允许一轮预注册补证，无重试）+ **sampler 不等价**（legacy curve+direction vs V541 start/pre_TS/post_TS/endpoint 分层 → matched 训练比较不可能）；其余六域（资产身份/数据标签 split/forward parity/参数组/schedule/修复后重评）全过。历史 C×3 降级为 auxiliary reference。

dev（589 curves，冻结全集；`reports/dev_frozen_evaluation.json` + `reports/dev_frozen/V541_T0R_E0_S{17,29,43}_dev.json`）：

| 指标（三 seed mean） | 值 |
|---|---|
| endpoint AUPRC | **0.290702**（per-seed 0.284809 / 0.286216 / 0.301081；prevalence 0.174655，n_pos 19,743 / n_neg 93,297） |
| q corr（curve-mean） | **0.397494** |
| MOVE RMSE/zero（curve median） | **0.910039** |
| dead fraction | 0.095771；sign cluster median 0.799378；cos_eff median 0.616318 |
| ordering | full 0.751152；cross-TS 0.855791 |
| AUROC / q MAE | 0.648604 / 0.509560 |

**dev veto 不适用**（E*=E0，无升级可否决）；`selection_or_tuning_performed=false`、`tau_recomputed=false`、候选 dev 访问 0 次（`audits/dev_access_ledger.jsonl`：WP7 前全部 `purpose=baseline_repair`，WP7 起 `candidate_evaluation` 仅含冻结 E0 与 rollout）。

## 5. no-physics rollout（`reports/rollout_comparison.json`；双口径分开报，§10.4）

1,152 episodes = 96 curves × 2 directions × 3 seeds × 2 budget modes（E0 only；名单 24⊂96⊂dev，curve-ID hash 规则 + SHA 冻结）。

| 口径 | endpoint arrival | TS candidate crossing | collisions | would_stop（错误） |
|---|---|---|---|---|
| autonomous（budget 200） | 0.005208 | 0.055556 | 85,117（565 episodes） | 64（64 全部错误 → 生产 No-Move 不开） |
| reference_normalized（46–186） | 0.005208 | 0.055556 | 25,152（518 episodes） | 10（10 全部错误） |

- 物理成本合计：`N_energy/N_gradient/N_Hessian/N_OptTS_trials/new_QC_calls = 0`；模型前向 166,902 次、墙钟 754.4 s；**非关键原子最大位移 0.0**；nonfinite/refusal/illegal-output/online-stop 均 0；`production_no_move_open=false`；声明仅限 `TS_candidate/endpoint_reached`，**无 verified TS、无部署资格**。
- **rollout attempt-1 资源失败留账**：v1 `ShardedKeyAtomDatasetV3(light_cache=True)` 物化全量 50K light 记录 → peak anon 12.26 GB 触 12G cgroup SIGTERM（exit 143，0 episodes、0 QC）；经授权的实现修复改为已认证 loader v2 `ColumnarKeyAtomDatasetV3`（idx_v2 memmap + 有界 LRU）重跑；名单/checkpoint/tau/预算未变（`rollout_comparison.json.loader.fix_note` + `reports/rollouts_manifest.json`）。

## 6. E0 冠军 bundle（`exports/embedding_champion_bundle/`，§12.3）

8 文件白名单精确匹配（`embedding_weights.pt`、`feature_schema.json`、`embedding_config.yaml`、`provenance.json`、`checksums.json`、`benchmark_summary.json`、`model_card.md`、`downstream_handoff.json`），`verify_embedding_bundle` 白名单+SHA 事后审计通过、无漂移。strict-load/forward/read-only 证据：`downstream_handoff.json` 声明 `frozen-E0` `strict_load=true / read_only=true / requires_grad=false`（strict-load 入口 `V541Model(...).load_state_dict(..., strict=True)`）；三个 seed 的 `model_strict_loaded=true`（`reports/rollout_plan.json`）；dev 冻结评估前向 6,372 batches / 407,778 positions（GPU 131.6 s，peak 86.6 MB）。`joint-finetune-E*` 需 V5.4.2 另立实验；完整模型复现依赖 reference checkpoint（S17 `c1e35610…`）。

## 7. 失败与删失全账（`reports/failure_and_censoring.csv` + `run_records.jsonl`）

| 项 | outcome | 原因 |
|---|---|---|
| V541_E1_S17 / E2_S17 / E3_S17 | scientific_failure（各 37,900 updates） | H2_MOVE_guard_failed（sign；E3 另 dead） |
| E4 | gate_skipped（0 updates） | cost ratio 3.6455 > 2.0 |
| E5 | gate_skipped | parent E4 cost gate failed |
| T3 | gate_skipped | n_selected=0（无候选补 seed） |
| T4 | gate_skipped | no_new_embedding_gain（容量/深度混杂未闭合） |
| WP5_CERT | evidence_missing | probe sid 不在 label cache；单次预注册补证，无重试 |
| formal_initial_refusal | refused_before_any_update | resource 配置 falsy-default 缺陷；修复后重冻结再启动 |
| rollout_attempt1 | resource_failure | v1 light-cache SIGTERM；loader v2 实现修复重跑（0 episodes） |

无失败删除、无换 seed、无删减更新数/任务。

## 8. 证据边界与不声明

1. **confirm 2,000 curves 从未读取**（`confirm_read=false` 于 signoff/screen/rollout/export 全链）；sealed/dev 账本按访问目的分开登记。
2. **`new_QC_calls=0`**：本阶段全部计算为既有数据/模型上的研究计算；无 ACP/xTB/ORCA/DFT、无能量/梯度/Hessian/OptTS。
3. 本轮 E1–E4 为本项目轻量适配实现，**不复现任何原论文完整 GNN**；E3/E4 收益/失败归因于"算子+合法视图"配方，不能单归网络算子。
4. 三候选 H2 失败是**科学负结果**（`scientific_failure`），不是实现 invalid；E1–E3 的强 PROCESS 增益（provisional）不构成升级资格。
5. T4 未执行 → 容量/深度混杂未排除；E0 留任的归因相应收窄（不声称"E0 优于一切容量匹配方案"）。
6. 单 seed 初筛无三 seed 稳定性声明；dev 三 seed 仅是 E0 基线表征，无候选对比 CI。
7. 无生产 No-Move 控制权、无 verified TS/TS_candidate 之外的路径声明、无部署资格；rollout 碰撞率高（autonomous 0.98 episodes-with-collision/episode）如实报告。

## 9. 全仓测试与文档门（2026-10-29）

- 全仓 `pytest`：**4,954 passed / 1 skipped / 11 deselected**（含 V541 模型/特征/训练器/resume/评估/rollout/导出全套新增测试）。
- `python scripts/agent_ops/docs_consistency.py --strict`：CLEAN；`python scripts/render_gv_map.py --check`：current。
- G–V registry：`V5.4.1` 状态 `executed`，evidence 指向本报告，`plan_revision: 2026-10-29.1`；旧 `V5.4` 条目未动。

## 10. 给 V5.4.2 的交接（§16.2）

冻结 E*=E0（空分支统一 schema，`legacy_state + 空 branch_state`）、`frozen-E*`/`joint-finetune-E*` 显式区分、feature recipe 48/22 + 内部 chem/spatial/angle recipe SHA、三 seed checkpoint 与 dev 报告 SHA 均在 bundle/provenance；MOVE/PROCESS 合同、输入白名单与 No-Move 化学语义不变；下游不得静默修改 E*，需修改则另开新 Embedding 版本与实验。
