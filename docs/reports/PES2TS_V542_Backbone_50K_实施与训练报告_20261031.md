# PES2TS V5.4.2 单帧空间 Backbone 50K：实施与训练报告（Executed-Frozen）

> 状态：**Executed-Frozen**（2026-10-31；owner 签认）
> 战役：`V5.4.2`（registry status `executed`，`plan_revision 2026-10-30.3`）
> 依据：[ADR-0033 v1.3.3](../design/decisions/ADR-0033-V5.4.2-Backbone接口与预注册治理.md)、[V5.4.2 详细阶段实施方案 v1.2](../plans/PES2TS_V5.4.2_Backbone_50K_详细阶段实施方案_v1.0_20261029.md)、`config/campaigns/pes2ts_v542_backbone_50k.yaml`（`0.1.2-wp0`）
> 任务合同：MoveProcessContract v1（不变）；objective = `unified_move_process`
> 提交链：`d94ad73`(WP0) → `3bbbdaa`(WP1) → `6939f6a`(WP2) → `dbae3f4`(WP3) → `3acecc4`(WP4 预算) → `844fcad`(WP4 裁决+WP5 runner) → `efae8f4`(WP6 统计) → `9568610`(T3 top2 门) → `647f804`(T2 激活门+stats 修复) → 本报告提交

## 0. 结论（冻结写法）

**战役结论：`no_backbone_winner` —— A0（matched baseline）留任。**

- 三个 seed 完整且逐 seed H2 无 breach 的候选集为空（`H3 eligible = []`），因此 `final_selection_rule` 走 `zero_h3_qualified_candidates_a0_retained` 分支：**不产生 Backbone\***，A0 作为对照基线继续服役。
- 该结论**不得**写成「A0 胜出」——A0 只是 matched 对照；也**不得**暗示 A0 优于 V541 E0（E0 仅 parity/历史参照，未参与本战役对照）。
- 候选侧：A2/A3 的 inner-val 与 PROCESS endpoint 明显优于 A0，但 **seed 43 触发 per-seed H2 breach**（MOVE sign 与 fixed-coverage direction 相对同 seed A0 退化），按冻结规则记 `scientific_failure`，不晋级、不得以 PROCESS 收益豁免；A1 未进 top2、未获得三 seed；A4 被冻结的 5% 学习筛淘汰；A5 完成 full-dose 但 seed17 未过 screen、未获补种。
- `confirm` 从未读取；`new_QC_calls=0`；单路、`swap=0`；dev 未访问；未作生产 No-Move / verified-TS / 部署资格声明。

## 1. 范围与冻结接口

| 项 | 冻结值 |
|---|---|
| Backbone 接口 | `BackboneOutput{z[G,64], edge_msg[E,48]}`（无 `u`、无 `node_h`） |
| MOVE readout | 唯一共享 `DirectMoveDecoder`，仅 `w_v = Linear(48→1)`（49 params），`a_ref = 0.08 Å` 非学习且只读 |
| PROCESS | `ReactionProcessHeadV53MP` 原样复用（2,178 params），只读 `z`；`order_score IS q` |
| A0 | `A0LegacyBackbone(StarEGNNV3)` 子类 + `message_mlp[-1]` hook；`backbone.w_v is move_decoder.w_v`；active 55,392 / joint 57,619 |
| 候选 | A1 41,232 / A1-C 57,264 / A2 54,896 / A3 53,328 / A4 55,840 / A5 31,584 / A5-C 53,712（实测精确，≤70k） |
| 单帧白名单 | 仅 forward-view keys；无历史/未来帧、真实 q、TS/端点 XYZ、能量/梯度；非关键原子零位移 |
| 剂量 | formal 400 W0 + 37,500 W1；`val_every=100`；checkpoint 仅按 inner-val total loss |
| 构造 | `fork_rng` + sha256 seed recipe + R3-only warm start + 确定性 process-head anchor + 构造后 `set_seed` |
| 禁止 | `torch.compile`、dev 参与选择、confirm/sealed/QC、事后调阈值/换 seed/缩剂量 |

## 2. 执行账本（全部实跑）

| 项 | 结果 |
|---|---|
| H0/接口 | parity 证书 6/6 资产（3 V541 E0 + 3 V53MP C）：CPU bitwise 0.0；GPU 硬上限内（最大 9.5367e-7）；frozen evaluator 重算 delta 0.0（63 float + 82 exact leaves）；`q is order_score` |
| smoke 200 | A0–A5 seed 17，6/6 `done`，kill-and-resume 证明，`rank_excluded` |
| profile 40+40 | 8 个 backbone（含 A1-C/A5-C conditional）；A0 1.919 s / A1 1.953 / A2 2.041 / A3 1.840 / A4 1.896 / A5 1.763 / A1-C 1.337 / A5-C 1.303 core-s（80-update window，20 预热剔除） |
| 预算签认 | `budget.per_candidate_core_time` 以实测回填；core 比 0.679–1.064，全部 ≤2.0；`signed_wp4_profile` |
| diagnostic 2000 | A0–A5 全做（400+1600）；A0 = `control_not_screened`；A1/A2/A3/A5 = `advance_full_dose`；A4 = `eliminated` / `gate_skipped(diag_best_inner_val_drop_lt_5pct_vs_initial)`（3.849742% < 5.0%） |
| A0 sanity band | **pass**：seed 均值 A0 0.29670 vs E0 0.28858（AUPRC +0.00812 ≤0.01）；q_corr 0.99682 vs 0.99727（−0.00045 ≤0.01）；RMSE/zero −0.199%（≤2%） |
| T0 | `V542_A0_S17/S29/S43` 全部 `done`，inner-val 0.6806 / 0.6666 / 0.6841；core 1317 / 1322 / 1265 s |
| T1 | `V542_A1_S17` 0.6924、`V542_A2_S17` 0.3275、`V542_A3_S17` 0.6476 全部 `done`；`V542_A4_S17` `gate_skipped`（WP4 binding） |
| seed17 screen | top2 = **[A2, A3]**（A1 未入选；A4 不合格）；freeze 前置（A0_S17 done + band pass）已断言 |
| T3 | `V542_A2_S29` 0.3213、`V542_A2_S43` 0.3209、`V542_A3_S29` 0.6506、`V542_A3_S43` 0.6713 全部 `done`；A1 S29/S43 `gate_skipped(top2_not_selected_inner_val_screen)`；A4 S29/S43 binding skip；A5 S29/S43 `gate_skipped(a5_supplement_parent_gate_unmet)` |
| T2 | `V542_A5_S17` `done`（inner-val 0.7079，core 1222 s）——按冻结条件在 A1–A4 无 H3 合格候选后自动激活 |
| N_core | 3（T0）+ 3（T1 A1–A3）+ 6（diagnostic）+ 1（A5 full-dose）+ 4（top2）+ 0（A5 supplement）+ 0（A1-C）+ 0（A5-C）= **17**（smoke/profile 为技术测量，不进 N_core） |

## 3. 冻结指标表（inner-val 冻结 evaluator；全部预注册指标，含失败）

| run | inner-val | RMSE/zero | direction | dead | AUPRC | q_corr |
|---|---:|---:|---:|---:|---:|---:|
| A0_S17 | 0.6806 | 0.9099 | 0.8000 | 0.1079 | 0.2868 | 0.9988 |
| A0_S29 | 0.6666 | 0.9008 | 0.8000 | 0.1015 | 0.3068 | 0.9999 |
| A0_S43 | 0.6841 | 0.8979 | 0.8333 | 0.0889 | 0.2965 | 0.9940 |
| A1_S17 | 0.6924 | 0.9042 | 0.8333 | 0.0894 | 0.2803 | 0.9948 |
| A2_S17 | 0.3275 | 0.8392 | 0.8000 | 0.0988 | 0.7991 | 1.0000 |
| A2_S29 | 0.3213 | 0.8370 | 0.8056 | 0.0877 | 0.8103 | 1.0000 |
| A2_S43 | 0.3209 | 0.8423 | 0.8111 | 0.0895 | 0.7909 | 1.0000 |
| A3_S17 | 0.6476 | 0.8925 | 0.8333 | 0.0735 | 0.3233 | 1.0000 |
| A3_S29 | 0.6506 | 0.8907 | 0.8333 | 0.0805 | 0.3266 | 0.9995 |
| A3_S43 | 0.6713 | 0.8916 | 0.8000 | 0.0963 | 0.3429 | 0.9995 |
| A5_S17 | 0.7079 | 0.9129 | 0.8000 | 0.0857 | 0.2374 | 0.9934 |
| R3 reference（同 evaluator，seed 均值） | — | 1.0060 | 0.6667 | 0.0937 | 0.1798 | −0.0889 |

`R3` 参照按 seed 重评（同 evaluator/环境），记录于 `plan/r3_reference_metrics.json`（checkpoint SHA 钉死）。

## 4. H2 与 H3 判定

**H2（逐 seed，对照 matched A0 与 R3）**

| run | H2 | 触发 guard |
|---|---|---|
| A1_S17 | pass | — |
| A2_S17 / A2_S29 | pass | — |
| **A2_S43** | **breach** | `matched_a0.sign_abs_drop`、`matched_a0.fixed_coverage_direction_rel_degradation` |
| A3_S17 / A3_S29 | pass | — |
| **A3_S43** | **breach** | 同上 |
| A5_S17 | pass | — |
| A0_S17 / S29 / S43（对照） | S17 对 R3 有 `r3.dead_fraction_abs_increase` 微超 | 基线漂移，非候选门；S29/S43 pass |

- A2/A3 在 seed 43 的 MOVE sign 与 fixed-coverage direction 相对同 seed A0 退化 → 记 `scientific_failure`；**不晋级**，不得用 PROCESS AUPRC（A2 约 0.79–0.81 vs A0 约 0.29–0.31）豁免。
- A0_S17 的 R3 dead-fraction 微超属 warm-start 后基线漂移，已在诊断裁决中记录为「非候选门」。

**H3（冻结统计合同）**

- eligible = **[ ]**：
  - A1 `three_seeds_incomplete`（未进 top2，S29/S43 `gate_skipped`）；
  - A2、A3 `per_seed_h2_breach`（S43）；
  - A5 `three_seeds_incomplete`（seed17 screen 未过，未获补种）；
  - A4 `gate_skipped`（diagnostic 淘汰）。
- 因 eligible 为空，Holm/bootstrap 推断**无对象**（不做空集显著性声明）；点估计与 CI 计算方法已在 `scripts/agent_ops/v542_stats.py` 冻结（10,000 paired cluster bootstrap，`bootstrap_seed_root=20261029`，one-sided p，MOVE 两出口各 α=0.025、PROCESS 单 family α=0.025），本报告不为空集伪造推断。
- `final_selection_rule`：0 个 H3-qualified → `no_backbone_winner`，A0 留任；报告不产生 Backbone\* 选择。

## 5. 激活与删失账

| 行 | 结果 | reason code |
|---|---|---|
| `V542_A4_S17` / S29 / S43 | `gate_skipped` | `diag_best_inner_val_drop_lt_5pct_vs_initial`（WP4 binding） |
| `V542_A1_S29/S43` | `gate_skipped` | `top2_not_selected_inner_val_screen` |
| `V542_A5_S29/S43`（supplement） | `gate_skipped` | `a5_supplement_parent_gate_unmet`（`a5_full_dose_complete=true`，`a5_seed17_screen_pass=false`） |
| `V542_A1C_S17` | `gate_skipped` | `a1_not_in_top2_and_not_selected` |
| `V542_A5C_S17` | `gate_skipped` | `a5_not_selected_backbone` |

- A5 的 seed17 未过 screen（H2 单 seed 虽 pass，但 screen 的 route/point-effect 条件未满足），故无补种、无 H3 资格；A5 smoke/profile/diagnostic 仅为工程证据，不进排名。
- A4 的 5% 学习筛在 warm start 下主要由「fresh 初始化比例」驱动（A0 自身 3.61% 亦不过），已由 Oracle WP4 裁决确认为**预注册设计缺陷**但不得事后改阈值；A4 结论按 binding 执行。

## 6. 成本

| backbone | core-s / 80-update profile window | 相对 A0 |
|---|---:|---:|
| A0 | 1.919 | 1.000 |
| A1 | 1.953 | 1.018 |
| A2 | 2.041 | 1.064 |
| A3 | 1.840 | 0.959 |
| A4 | 1.896 | 0.988 |
| A5 | 1.763 | 0.919 |
| A1-C | 1.337 | 0.697 |
| A5-C | 1.303 | 0.679 |

formal 实跑（core）：T0 1265–1322 s；T1 1167–1354 s；T3 1401–1439 s；T2 1222 s。无候选触发 `gate_skipped_cost_ratio_exceeds_2`。

## 7. 负结果逐项写法（冻结措辞）

- **A1（2L GeoEGNN）**：该轻量配方在本剂量/本合同下未超过 A0（seed17 inner-val 0.6924 vs A0 0.6806，且未进 top2）；深度/距离算子归因**未解决**（A1-C 记 `gate_skipped(a1_not_in_top2_and_not_selected)`，不事后补跑 C2 或未预注册对照）。
- **A2（PaiNN-lite v32）**：PROCESS 显著占优（AUPRC 约 0.79–0.81），MOVE 点估计亦优于 A0，但 seed 43 触发 H2 sign/direction 退化 → `scientific_failure`；不晋级、不用 PROCESS 收益豁免。
- **A3（A1+ReactionQuery）**：与 A2 同型结局（seed 43 H2 breach），PROCESS 小幅占优（AUPRC 约 0.32–0.34），不晋级。
- **A4（sparse transformer）**：被冻结学习筛淘汰（3.849742% < 5.0%），按 binding 不跑 full dose；该筛的初始化代理混杂已记录为设计局限，不事后改阈值。
- **A5（shared GRU T4）**：full-dose 完成但 seed17 未过 screen，补种未激活；容量 caveat（31.6k vs A0 55.4k）未解决，`capacity_caveat_unresolved`。
- **A0 留任**：对照语义，不是「A0 胜出」，也不构成 V541 E0 的优劣比较。

## 8. 边界与纪律

- `confirm` 从未读取（`confirm_read=false`）；`new_QC_calls=0`；`sealed_authorized=false`；单路（`single_lane=true`）；`swap=0`；`oom_kill_max=0`。
- dev 未访问：本阶段未对任何模型做 dev 评估，dev 账本无新增读取；若后续需要 dev final-few，须在 backbone 冻结后按 ADR-0033 §2.6 执行一次并记账。
- 旧冻结文件（V5.4/V5.4.1 registry 条目、`outputs/` 历史记录、旧 ledger 状态机）未改；本战役 outcome 使用六词表，append-only。
- 未作生产 No-Move / verified-TS / 部署资格声明。

## 9. 给 V5.4.3 的交接

1. **冻结接口**：`BackboneOutput{z, edge_msg}` + 唯一 `DirectMoveDecoder`（`w_v` 49、`a_ref` 0.08 只读）+ 原样 `ReactionProcessHeadV53MP`；候选构造协议（`fork_rng` + seed recipe + R3-only warm start + 确定性 anchor）在 `pes2ts_core/pesn/models/v542/candidates/`。
2. **留任模型**：A0（matched baseline）；三 seed checkpoint 位于 `outputs/pesn/v542_backbone_50k/runs/V542_A0_S{17,29,43}/checkpoint_best.pt`，身份块与 canonical digest 在 `run_ledger.jsonl` / `run_manifest.json`。
3. **对照口径**：H2/H3 一律对 matched A0（同 seed）；R3 仅作 warm-start 参照（`plan/r3_reference_metrics.json`，同 evaluator 重评）。
4. **评估身份**：frozen `evaluate_v53mp`；`tau_end=None`；`statistics_contract_sha256` 与 `final_selection_rule` 规则块 SHA 见 `reports/final_selection.json`。
5. **未决归因**：(a) A1 深度/距离算子归因未解决（A1-C 未跑）；(b) A5 容量 caveat 未解决（补种未激活）；(c) diagnostic 5% 学习筛的初始化代理问题（下一战役建议 step-0 baseline 或 control-relative non-inferiority，且须在预注册时冻结）。
6. **可复现**：`v542_formal.py status`、`v542_stats.py {extract,h2,screen,h3,select,activate}` 与全部报告 JSON 均在 `outputs/pesn/v542_backbone_50k/`；配置、validator、模板与测试自带的 625+ 项断言可重放。

## 10. 证据路径

| 证据 | 路径 |
|---|---|
| run ledger（18 行，append-only） | `outputs/pesn/v542_backbone_50k/run_ledger.jsonl` |
| 各 run 目录（checkpoint/manifest/metrics） | `outputs/pesn/v542_backbone_50k/runs/V542_*` |
| A0 sanity band | `outputs/pesn/v542_backbone_50k/plan/a0_sanity_band.json` |
| 诊断 v1（不可变）/ v2（生效）/ current | `diagnostics/diagnostic_report.jsonl` / `.v2.jsonl` / `diagnostic_report_current.json` |
| 协议修订（WP4） | `plan/protocol_amendment_wp4_diagnostic.json` |
| 治理记录实例 | `plan/a0_reuse_decision.json`、`plan/preregistration_signoff.json` |
| 预算事实源 | `profile/profile_report.jsonl` |
| 指标表 / H2 / screen / H3 / 选择 / 激活 | `reports/metrics_table.json`、`h2_report.json`、`seed17_screen_report.json`、`h3_report.json`、`final_selection.json`、`activation_decisions.json` |
| R3 参照 | `plan/r3_reference_metrics.json` |
| parity 证书 | `scripts/agent_ops/v542_a0_parity.py --out …`（6/6，CPU bitwise 0.0） |

## 11. 复核清单

- [x] 六项 parity 资产 path+SHA 核验，CPU bitwise + GPU 上限 + evaluator parity
- [x] A0 迁移不变量（active 55,392 / joint 57,619）与 `w_v` 共享
- [x] 候选精确参数量与 ≤70k 上限
- [x] T0 A0×3 + sanity band 先于任何 T1 formal（冻结顺序）
- [x] top2 freeze（A0_S17 + band pass + seed17 screen）后仅 top2 成员补种
- [x] A4 全程 `gate_skipped`（binding），A5 补充门未满足
- [x] 每 run identity/seed/剂量/outcome/reason code 入账；append-only
- [x] confirm 未读、QC=0、单路、swap=0
- [x] 负结果与删失全账，不补跑、不降阈值、不换 seed
