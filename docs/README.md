# PES2TS 文档索引

## 2026-10-29 V5.4.2 WP0 治理/预注册落地（Active-Proposed）

- **2026-10-30 v1.1（Oracle gate 0 REVISE 有界语义修复）**：top2/A5 supplement 完全拆分（`top2_max_runs=4` 只属 A1–A4 + `a5_supplement_max_runs=2` + `t3_max_runs=6`，N_core 显式公式、smoke/profile 不进 N_core）；`seed17_screen_pass` predicate 冻结（非三 seed H3 声明）；H3 统计合同冻结（`bootstrap_seed_root=20261029`、确定性子 seed 公式、one-sided improvement p 值、Holm family 显式冻结、CI 与 p 同 draws）；初始化冻结（sha256 seed recipe、construction order、per-candidate R3 映射表、A1-C→A1/A5-C→A5 init stream）；A5 full-dose launch_requires 与 A1-C/A5-C profile/cost/budget 前置；A5-C 激活=`a5_selected_backbone`；H2「退化」方向词恢复；`t1_start_requires` canonical gate IDs；identity 增列 source_manifest/bootstrap/anchor/mapping/plan/oracle SHA；A0 决策模板/实例分离（instance 为 T1 前置）；parity 6 checkpoint 全 required 且 path+SHA 钉死（3 V541 E0 + 3 V53MP C，smoke 禁止）；registry `data_manifest` 改实际路径。登记见 [台账](plans/PES2TS_文档修订台账.md)。validator/tests 同步已由 **fix-4/fix-6 机器硬化车道完成（2026-10-30）：426 schema tests + 20 registry tests = 446 total 全绿，prereg checks=26**。
- **2026-10-30 v1.2（fix-5 PASS_WITH_CONDITIONS 条件收口，ADR-0033 v1.2）**：C1 冻结 `final_selection_rule`（H3-qualified only；MOVE-qualified 优先 PROCESS-only → min Holm-adjusted p → max effect margin → min core_time_ratio → min params → min inner-val loss → min candidate_id；不用 dev；A0/diagnostic 不参加）；C2 `route_logic` 括号化（MOVE/PROCESS 两支各自闭合，seed17 screen 复用）；C3 parity 资产分层（V541 E0 `runs/` 37,900 final 主证据 + V53MP C 20k S3 formal 15,400 辅助证据，逐行 `source_tier`/`training_updates`；GPU 1e-6/1e-5 硬上限）；C4 signoff roles=`main_session_owner` + `oracle_reviewer`（reviewer 非签认人）；C5 新 canonical gate `V542-A0-sanity-band`（T0 A0×3+sanity band 先于任何 T1 formal，breach 严格阻断）；C7 stale 计数清理、config `0.1.2-wp0`、smoke `conditional_backbones:[A1-C,A5-C]`、runplan `bootstrap_seed_root`、registry `plan_revision: 2026-10-30.1`。schema/tests 断言增量由 fix-6 负责。
- **[ADR-0033 V5.4.2 Backbone 接口与预注册治理](design/decisions/ADR-0033-V5.4.2-Backbone接口与预注册治理.md)**（**Accepted**；v1.2）：MoveProcessContract v1 任务语义不变，统一单帧空间 Backbone 接口——`BackboneOutput` 只有 `z[G,64]+edge_msg[E,48]` 两个功能字段；唯一共享 `DirectMoveDecoder`（仅 `w_v` 49 params）；PROCESS 原样 `ReactionProcessHeadV53MP`（2,178 params）。A0 = legacy Star 子类 adapter（hook 捕获 + 共享 `w_v` + 显式映射 strict audit + CPU bitwise/GPU 1e-6/1e-5 parity；active 55,392/joint 57,619 迁移不变量；禁 `torch.compile`）。**默认且实际选择重训 matched A0×3**（`retrain_a0_x3`；不实现 shadow-run/reuse 分支；V541 E0×3 仅 parity/历史参照）。
- **[战役配置 pes2ts_v542_backbone_50k.yaml](../config/campaigns/pes2ts_v542_backbone_50k.yaml)**（schema `pes2ts_v542_campaign_v1`，primary_g=G3.P.2；v1.2/0.1.2-wp0 = fix-5 条件收口）：候选 A1 2L GeoEGNN / A2 单帧 PaiNN-lite v32 / A3 A1+ReactionQuery 2×24 / A4 2L sparse geometry transformer 4×12+token / A5 shared GRU T4（hidden48/z64/edge_msg48，active ≤70k，R3 冻结 checkpoint warm-start + per-candidate 语义映射表 + process-head fork_rng anchor + sha256 init recipe（A1-C→A1、A5-C→A5 init stream））；smoke 200 + 同环境 profile（A1-C/A5-C conditional）+ seed17 diagnostic 2,000（400+1,600，A0–A5，只做工程粗筛）；formal 37,900 updates；**top2 只属 A1–A4**（`seed17_screen_pass` predicate，≤2 候选→`top2_max_runs=4`）+ **A5 supplement 独立分支**（条件=`a5_full_dose_complete_and_seed17_screen_pass`，`a5_supplement_max_runs=2`，`t3_max_runs=6`；N_core 公式显式化、smoke/profile 不进 N_core）；A1-C 激活条件 = `a1_in_top2_supplement_set_OR_a1_selected_as_backbone`（**不是仅 top-ranked**），A5-C 激活 = `a5_selected_backbone` 仅此一项；H2 继承（**退化** ≤2% 措辞）、H3 对照 matched A0 + `route_logic` 括号化 + MOVE 两 exit 各 α=0.025 各自对 eligible candidates Holm + PROCESS 单一 family（2 endpoints × candidates）Holm + seed-averaged curve paired bootstrap 10,000 + `bootstrap_seed_root=20261029` + one-sided improvement p 值公式；**`final_selection_rule` 冻结（H3-qualified only、MOVE>PROCESS-only 字典序、不用 dev）**；parity 6 checkpoint 全 required（V541 E0 `runs/` 37,900 final 主证据 + V53MP C 20k S3 formal 15,400 辅助证据，path+SHA 钉死，smoke/非钉死禁止）；成本门 `core_time_ratio≤2.0`（分母 V542 A0 同环境 profile）；budget 在 profile 前保持 null——prereg validator 通过、`--formal` typed block `V542_BUDGET_NULL`。
- **独立 validator**：`scripts/agent_ops/v542_config_schema.py`（prereg/formal 双模式 + CLI + typed rejection（计数见 validator 模块））；测试 `tests/agent_ops/test_v542_config_schema.py`（**fix-6 完成：426 例 schema 断言**；+20 registry tests = **446 total**；v1.2 新增语义的机器断言已同步）。
- **WP0 模板**：`config/campaigns/v542_templates/{preregistration_signoff,budget_policy,runplan_pin,parity_certificate,a0_reuse_decision}.template.json`。
- **注册表**：`config/registry/g_v_map.yaml` 新增 `V5.4.2`（status `proposed`——**未**写 executed/smoke_passed；旧 `V5.4`/`V5.4.1` 条目不动）；G–V 文档已重渲染。
- **边界**：本轮仅 WP0 治理产物；未改模型/训练器/旧冻结文件/`outputs/` 历史记录，未启动训练/QC，未访问 dev/confirm，不宣称训练完成或候选科学收益；正式 run 仍须 H0/H1/smoke/profile/预算/runplan SHA 冻结全过。

## 2026-10-29 V5.4.1 执行完成（Executed-Frozen，E*=E0 留任）

- **[V5.4.1 Embedding 50K 实施与训练报告 20261029](reports/PES2TS_V541_Embedding_50K_实施与训练报告_20261029.md)**（只读最终报告）：WP0–WP7 全部完成。T0-R matched E0×3（17/29/43）+ E1/E2/E3 seed17 均 37,900 updates done；**E1/E2 H2 sign guard 失败、E3 sign+dead 失败 → `scientific_failure`**（`provisional_H3` 不豁免硬门）；**E4 core ratio 3.6455>2.0 → `gate_skipped`**（成本门）；E5/T3/T4 `gate_skipped`（T4 `no_new_embedding_gain`，容量/深度混杂未闭合已声明）；无候选补 seed、无候选三 seed CI（不编造）。**E\*=E0 留任**：dev mean AUPRC 0.290702 / q_corr 0.397494 / RMSE-zero 0.910039；dev veto 不适用，无调参、tau 未重算；rollout 1,152 episodes 双口径分开、endpoint arrival 两口径均 0.005208、TS candidate 0.055556、QC=0、非关键原子位移 0；E0 bundle 8 文件 strict-load/read-only 审计通过。confirm 未读、`new_QC_calls=0`、无生产 No-Move/verified TS/部署资格声明。
- **状态同步**：[实施方案 v1.2](plans/PES2TS_V5.4.1_Embedding_50K_详细阶段实施方案_v1.0_20261027.md) 转 Executed-Frozen（新增 §19 执行指针，§0–§18 规划历史未改写）；[ADR-0032](design/decisions/ADR-0032-V5.4.1-Embedding接口扩展与预注册治理.md) 增补 §6 执行附录（v1.2）；G–V registry `V5.4.1` → `executed`（`plan_revision: 2026-10-29.1`，旧 `V5.4` 不动）。
- **全仓测试（2026-10-29）**：pytest **4,954 passed / 1 skipped / 11 deselected**；`docs_consistency --strict` CLEAN；`render_gv_map --check` current。

## 2026-10-27 V5.4.1 WP0 治理/预注册落地（Active-Proposed）

- **[ADR-0032 V5.4.1 Embedding 接口扩展与预注册治理](design/decisions/ADR-0032-V5.4.1-Embedding接口扩展与预注册治理.md)**（**Accepted**）：MoveProcessContract v1 任务语义不变，只扩展 Embedding 残差接口；新增独立 schema `pes2ts_v541_campaign_v1`（objective 仅 `unified_move_process`，不暴露三臂矩阵）与独立 validator `scripts/agent_ops/v541_config_schema.py`——未知 embedding、非法 forward key、EdgeAdapter/vector 消费、confirm/sealed 授权、缺字段、阈值 null（formal）均 typed rejection。
- **[战役配置 pes2ts_v541_embed_50k.yaml](../config/campaigns/pes2ts_v541_embed_50k.yaml)**：`primary_g=G3.P.2`、supporting G3.P.1/G3.P.3/G0/G1；§11.2 阈值 2026-10-27 按用户指令签认为 frozen（含 hard_ratio `max_core_time_ratio=2.0`、GPU parity 1e-6/1e-5、CI 10,000 次、rollout 96 曲线）；`confirm/sealed/new_QC=false`、单路、swap=0；run outcome 六词表与旧 ledger 状态机分开，旧状态机不改。
- **预注册模板**：`config/campaigns/v541_templates/{preregistration_signoff,budget_policy,runplan_pin}.template.json`（nominal/conditional/diagnostic 三类 run 分离，diagnostic 不计入主赢家）。
- **注册表**：`config/registry/g_v_map.yaml` 新增 `V5.4.1`（`plan_revision: 2026-10-27.1`），旧 `V5.4` from-any-point 条目保持不动；G–V 文档已重渲染。
- **T3 预注册修正（WP0 复核）**：删去含混 `V541_T3_TOP2_S##` 占位，改为预注册全部可被初筛选中的候选 E1–E5 × seeds {29,43} 共 **10 个稳定 `V541_E#_S##` 条件 run**（`condition=activated_only_by_inner_val_top2`、`winner_eligible=true`、`new_training_if_activated=1`、未激活 `gate_skipped` 计 0 新训练）；仅 inner-val top-2 冻结后激活，**满额 4 次**，`N_core` 按 §9.1 公式计数而非 listed runs；E5 行叠加父门。validator 新增 `t3_preregistration` 检查（含 `TOP2` 占位与 null embedding 身份拒绝）。
- **WP5 待裁决事实（已核实，记入 ADR-0032 §4）**：旧 V53MP `_curve_batch_positions` 实际仅按 curve+direction 分池采样，无 config 声明的 `start/pre_TS/post_TS/endpoint` 分层；V541 冻结 config 保留该声明（未改 strata），不得静默等价——runner 必须实现/证明配置所声明的共同 sampler，WP5 确认不成立则触发 E0×3 重训分支。
- **边界**：本轮只完成 WP0 治理与签认，**未启动训练/QC、未授权 confirm、不宣称任何候选科学收益**；实际 T1 仍须 H0/H1/单路 smoke/预算签认/formal SHA 冻结全过（validator 报告 `per_candidate_core_time=null` 为 T1 阻断项）。

## 2026-10-27 V5.4.1 Embedding 50K 详细实施方案（Active-Proposed）

- **[V5.4.1 Embedding 50K 详细阶段实施方案 v1.1.1](plans/PES2TS_V5.4.1_Embedding_50K_详细阶段实施方案_v1.0_20261027.md)**：细化用户提供的单臂双功能头设计冻结候选 v1.1；归属 V5.4 / G3.P.2，冻结评估延伸至 G3.P.3。固定 50K、三层 StarEGNNV3 core、MOVE+PROCESS、37,900 updates；E1–E4 初筛、E5 条件启动、两个候选补三 seed，E0 可留任。明确先修 pair evaluator 并重评，E0 复用或 +3 重训由训练可比性审计决定；confirm 继续封存。内容已补充分子 GNN 文献选型、F2/RBF 派生隔离、算子规格与学习监控（§6.5–6.8/§7.5/§18）；文件名保留 v1.0。v1.1.1 进一步明确本阶段 `new_QC_calls=0`，rollout 为 no-physics/pure-geometry。本轮仅完善规划，未实现 V5.4.1、未启动训练/QC、未授权 confirm。

## 2026-10-22 V5.3-MP S1 2k 冒烟完成（当前）

- **[V5.3-MP S1 2k 冒烟报告 20261022](reports/PES2TS_V53MP_S1_2k冒烟报告_20261022.md)**（**S1 PASS**）：三臂 ok、Move guard 全过、GPU 1,829s≤上限；ordering/endpoint 为冻结段随机水平（预期）；λ 探针 verdicts 记录。S2（5k 三臂×三 seed）进行中。

## 2026-10-22 V5.3-MP S0 数据建库完成

- **[V5.3-MP S0 数据建库与资格报告 20261022](reports/PES2TS_V53MP_S0_数据建库与资格报告_20261022.md)**（**S0 PASS，全部门通过**）：20,000 条合格独立反应冻结池（2k ⊂ 5k ⊂ 20k 嵌套）+ dev 589 + confirm 2,000（仅列名封存）；S0-A 资格 142,221/141,483；S0-B 经审计提取 26,038 曲线 / 5,958,548 样本；endpoint_label_spec_v1 有效正例覆盖 99.62%、TS/起点 0% 正例；S0-D 分片惰性加载过资源门；S0-F 就绪 7/7 + 表校验 10/10。S1 待 λ 探针 + 单独授权。

## 2026-10-21 V5.3-MP MOVE–PROCESS 统一学习

- **[ADR-0031 MOVE–PROCESS 统一学习合同](design/decisions/ADR-0031-MOVE-PROCESS统一学习合同.md)**（**Accepted**）：一个共享 EGNN，两个头——MOVE（关键原子位移，跨越 TS）+ PROCESS（有向进度 q、端点完成置信度 p_end、ordering）；No-Move = p_end ≥ τ_end 且目标化学状态一致；q 仅辅助/诊断；TS 永远继续 MOVE。取代 FullPathNavigationContract v1 的任务语义。
- **[MoveProcessContract v1](contracts/PES2TS_MoveProcessContract_v1.md)**（**Active-Normative**）：MOVE/PROCESS/No-Move/TS 的唯一任务合同；输入仍为当前单帧 + 合法有向拓扑，禁止参考端点 XYZ/历史/真值；旧单构象 `d_end` 与 P5 basin 降级为诊断量。
- **[V5.3-MP 大规模 MOVE–PROCESS 联合训练计划 v1.1](plans/PES2TS_V5.3-MP_大规模MOVE-PROCESS联合训练计划_v1_20261021.md)**（**Proposed v1.1**）：S0 20k 数据资格/建库（S0-A 元数据不读原始真值；S0-B 授权提取逐次记录；S0-D 分片惰性加载；S0-F 实施就绪 7/7）→ S1 2k 冒烟 → S2 5k → S3 20k 正式 → S4 条件扩量；三臂 A=Move-only / B=Move+Progress / C=Move+PROCESS；当前 GPU=0，S0+就绪门通过且 S1 单独授权后才开训。
- **[V5.3-MP 文档同步清单 v1](plans/PES2TS_V5.3-MP_文档同步清单_v1_20261021.md)**：P0–P6 同步项与 docs gate 判据。
- **命名**：`V5.3-ENG` = 已执行的 30h rollout/relax 工程闭环（历史）；`V5.3-MP` = 本战役。

## 2026-10-21 V5.2-R 全路径导航与端点停止（历史，CLOSED at D2-G）

- **[ADR-0030 V5.2 全路径导航与端点停止](design/decisions/ADR-0030-V5.2全路径导航与端点停止.md)**（**Accepted**）：Stop 旧定义为 TS 邻域属任务定义错误；改为 Full-Path Navigation + Endpoint Stop；条款级取代 ADR-0021/0028/0029 相关语义。
- **[FullPathNavigationContract v1](contracts/PES2TS_FullPathNavigationContract_v1.md)**（**SUPERSEDED by ADR-0031 / MoveProcessContract v1**）：保留为 ADR-0030 历史证据；Move/Stop/TS/q 旧语义不再用于新训练。
- **[V5.2-R 全路径导航与端点停止修复方案 v1](plans/PES2TS_V52R_全路径导航与端点停止修复方案_v1_20261021.md)**（**Closed — D2-G STOP**）：D0 PASS / D1 FAIL / D0.1 PASS / D2 P1 PASS + P4/P5 FAIL；T1 不授权；负结果归档，由 V5.3-MP 接续。
- **[V5.2 任务纠偏审计 20261021](reports/PES2TS_V52_任务纠偏审计_20261021.md)**：终点样本被 terminal 排除、旧 Stop 正例全在 TS 附近、20 步预算不足、provenance 缺口等实测证据。
- 合同修订：[ReactionIntent Contract v2](contracts/PES2TS_ReactionIntent_Contract_v2.md)、[GeometryPolicyContract v2](contracts/PES2TS_GeometryPolicyContract_v2.md)（取代 v1 的 Stop/q 条款）。

## 2026-10-20 V5–V7 v3.0 反应意图条件化（当前）

- **[ReactionIntent Contract v1](contracts/PES2TS_ReactionIntent_Contract_v1.md)**（Proposed；**§4 Stop/q 条款已由 [ReactionIntent Contract v2](contracts/PES2TS_ReactionIntent_Contract_v2.md) 取代（ADR-0030）**）：`π(X_t,G_rxn) → (ΔX_K, q̂, m̂)`；`G_rxn` 是有方向反应意图，不是 branch；定义 signed graph edit、CV/Jacobian lifting、Relabel invariant / Reverse sensitive、输入合法性。
- **[V5–V7 反应意图条件化与低维生长方案 v3.0](plans/PES2TS_V5-V7_反应意图条件化与低维生长方案_v3.0_20261020.md)**（Proposed，当前阶段计划；**内容 v3.2（2026-10-21 任务纠偏修订）**）：V5 = Reaction-Conditioned Geometric Policy；V6 = Physics-Corrected Reaction Growth；V7 = Cost-Aware Autonomous PES Exploration。**取代 v2.0 的 Geometry Policy / Physics-Corrected Policy / Cost-aware RL 语义。**
- **[V5.1-R3 反应意图条件化与端点标签不变重训方案 v3.1](plans/PES2TS_V5.1-R3_反应意图条件化与端点标签不变重训方案_v3_20261020.md)**（Proposed，当前 R3）：`intent_directed_irc_step`；一条完整 IRC、两种有向 intent、first-contact 单步；保留合法 signed `ΔB_intent`；删除 branch；Relabel invariant、Reverse sensitive；不做 CV/Jacobian/Progress/Stop（留给 V5.2）。
- **[V5.1-R3 数据库+特征+损失三位一体修复方案 v1](plans/PES2TS_V5.1-R3_数据库特征损失三位一体修复方案_v1_20261020.md)**（Proposed，执行细化）：五张数据表、Reaction Intent 特征、B0-F + strong/dead、梯度纪律、66-run 矩阵与验收门。
- **[单端点 IRC 生成控制器产品设计 v3](plans/PES2TS_单端点IRC生成控制器_产品设计_v3_20261020.md)**（Proposed，产品设计）：曲线中心、无 branch；PREDICT → ACTION → RELAX → UPDATE；Known-B / TS-only / Unknown-B 三模式；TS 在 full-curve 中是途经点。
- **[30h R3 → V5.2 → V5.3 分层训练与验收计划 v1.1](plans/PES2TS_30h_R3_V52_V53_分层训练与验收计划_v1.1_20261020.md)**（**Executed-Frozen / Stop 语义已被 ADR-0030 取代**）：R3 66 runs + V5.2 33 runs；R3 语义仍有效，TSStop/三态 Stop 章节（§5.3/§5.6/T3.2）仅历史；First-stop error、两段训练、V5.3 rollout、G0–G4 门、DAG 与测试清单。
- **关键语义**：端点重命名（Relabel）输出不变；真实反应反转（Reverse）输出可改变；不添加单独 direction token；方向由反应意图规定。V5.1b、sealed、Confirm/OOD、Fit150+ 继续关闭。

### 历史：V5–V7 v2.0 与 V5.1-R2（Superseded / Historical）

- [V5–V7 三阶段研究与架构修订方案 v2.0](plans/PES2TS_V5-V7_三阶段研究与架构修订方案_v2.0_20261020.md)：**Superseded by v3.0**；保留为历史设计。
- [V5.1-R2 损失尺度与数据完整性修复方案](plans/PES2TS_V5.1-R2_损失尺度与数据完整性修复方案_v1_20261020.md)：**Superseded by V5.1-R3**；R2 的 strong/dead 公式与 S1 审计仍被 R3 继承，但“特征不变”假设不再成立。
- [GeometryPolicyContract v1](contracts/PES2TS_GeometryPolicyContract_v1.md)：**Historical-Proposed**；其 `π(X_t,G)→(ΔX_K,m)` 语义被 ReactionIntent Contract v1 扩展/取代。
- [ADR-0027](design/decisions/ADR-0027-V5.1参考架fallback与双分母评估.md)、[ADR-0028](design/decisions/ADR-0028-关键原子几何策略合同与V5-V7任务拆分.md)、[ADR-0029](design/decisions/ADR-0029-V5.2+双门评估与封存边界.md)：历史/边界 ADR；v3.0 继承其数据质量、双分母、封存边界，不继承 branch/旧 V5.2 语义。
- `V5.1-R2` 已归档；`V5.2–V5.4` 已按 v3.0 重新解释（见 G–V map 2026-10-20.2）。旧 branch/旧 V5.2 语义不得作为当前训练依据。

## 0. G–V 双轴映射（2026-10-18；2026-10-20 内容修订）

- **G = 代码/物理能力**（G0–G4.2），**V = 训练/测试战役**（V4.x/V5.x/V6/V7）；V5–V7 不是新 G 阶段。
- 权威入口：[G–V 映射表](design/PES2TS_G_V_MAPPING_v1_20261018.md)、[ADR-0025](design/decisions/ADR-0025-G-V双轴映射与战役注册.md)、[代码地图 v2](design/PES2TS_CODE_MAP_v2_20261018.md)。
- 机器可读源：`config/registry/g_v_map.yaml`；生成/校验：`python scripts/render_gv_map.py --write|--check`。
- 2026-10-20 内容修订：`plan_revision: 2026-10-20.2`；`map_version: 2026-10-18.1` 保持不变（schema 未变）。V5.1-R3 为当前 R3；V5.1-R2/v2.0 归档；V5.2–V5.4 按 ReactionIntent Contract v1 重新解释；旧 V6.0–V6.2/V7-A/B/C 归档。
- 2026-10-21 任务纠偏：`plan_revision: 2026-10-21.1`（`map_version` 仍为 2026-10-18.1）——ADR-0030 全路径导航与端点停止生效；`V5.2` → `executed`（旧 TS-stop 任务负结果、冻结证据），新增 `V5.2-R` proposed（FullPathNavigationContract v1）；V5.3+ gate_note 更新为端点停止与长度感知 rollout 预算。
- 2026-10-27 WP0：`plan_revision: 2026-10-27.1`（`map_version` 仍为 2026-10-18.1）——ADR-0032 新增 `V5.4.1` Embedding-50K 战役（primary G3.P.2；supporting G3.P.1/G3.P.3/G0/G1；schema `pes2ts_v541_campaign_v1`）；旧 `V5.4` 条目不动。
- 2026-10-29 执行收尾：`plan_revision: 2026-10-29.1`——`V5.4.1` 状态 `proposed` → `executed`，evidence 指向[最终报告](reports/PES2TS_V541_Embedding_50K_实施与训练报告_20261029.md)，code_scope 回填实际实现路径（models/v541、training/v541_*、pair_ownership_v3、stream view、scripts/campaigns/v541_*）；旧 `V5.4` 条目不动。
- 状态图例：`Canonical`（规范根）/ `Active-Proposed`（当前提案）/ `Executed-Frozen`（已执行、冻结证据）/ `Superseded`（被取代）/ `Evidence-Historical`（历史报告）。

## 2026-10-18 V5–V7 终审与第一阶段冲刺提案（历史）

- **冲刺方案内容v1.1更新**：[50簇输入/标签快速筛选与粗到细路线](plans/PES2TS_V5.0-V5.2_十小时冲刺训练与验证执行方案_v1_20261018.md#5-600分钟执行预算与熔断v11先筛标签再扩量再精修)：固定小EGNN，anchor on/off×0.05/0.08/0.12 Å共6臂、1500更新短训练、每分支16起点轮换、密集验证、Top-2跨划分三seed复核；OOF按测速条件执行。200/500簇扩量与粗到细配对实验置于后续窗口，600分钟内不叠加旧四架构/30run矩阵。仅更新提案，未启动训练/QC。

- [PES-N V5–V7 架构调整与实施计划 v1.2](plans/PES2TS_PES-N_V5-V7架构调整与实施计划_v1.2_20261018.md)：**Superseded（2026-10-20）**；保留为历史设计，V5/V6/V7 分工与 V6 progress-decoder 已被 v2.0/v3.0 取代。未修改业务代码、未启动训练/QC。

- [V5.0–V5.2 十小时冲刺训练与验证执行方案](plans/PES2TS_V5.0-V5.2_十小时冲刺训练与验证执行方案_v1_20261018.md)：七模块专项方案，含参考架/手性/泄漏终审、精确张量与损失、600分钟条件预算、OOF与四级Gate。**V5.1a 已执行并 `rejected_gate`；V5.1a-R1 无候选准入；本文件保留为历史执行依据**。未实现脚本、未启动训练/QC，不升级能力或生产默认。作为[V5–V7架构计划v1.2](plans/PES2TS_PES-N_V5-V7架构调整与实施计划_v1.2_20261018.md)的专项评审提案，原计划保留；具体冲突裁决见新方案§0，实现前另行正式接口/ADR登记。

## 2026-10-18 V4.2–V4.7 执行结果与实现报告

- [PES2TS 实现报告（截至 2026-10-18）](reports/PES2TS_实现报告_20261018.md)：系统实现总览、V4.2–V4.7 新增数据/模型/训练/评测能力、23 个正式 run 的最新结果与失败归因。结论为负结果：V4.7 FINAL 三种子（fit50、native_full）中位 **MSE/零动作 1.132、0/8 组、整体 cos 0.180、near-TS cos 0.248**，四项研究门全未通过；不建议晋级、部署或扩量。报告同时列出 E2 评测缺口、H1 `prior_not_supported`、H2 λ 失败、V2 裁剪与工程修复记录。
- [V4.2–V4.7 过夜战役汇总（自动草稿）](reports/PES2TS_V4.2-V4.7_过夜战役汇总_20261018.md)：按 run_records/protocol_evals 自动聚合的逐臂表；最终解释以实现报告为准。

## 2026-10-17 V4.2–V4.7 训练实验提案

[V4.2–V4.7 约十小时训练实验方案](plans/PES2TS_V4.2-V4.7_十小时训练实验方案_20261017.md)（v1.1）：基于 V4/V4.1 本地收据和源码复核，保留五组实验与最终三种子定稿；新增原生标签/梯度来源预检、Direct 对 Gradient-residual、5k/10k/20k 曝光与饱和诊断，保留稀疏条件和 PaiNN 式向量传播。600min 封套，最终默认 20k，内部曲线与测速允许才升 40k；fit 内部开发集选配方，monitor8 最后按三种口径复核。状态 Proposed，本次仅修订文档，未实现或启动新训练/QC，不改变历史冻结协议、主任务与能力状态。

## 历史训练与项目进度入口（2026-10-15，保留为历史）

> 以下 v4 排期为 2026-10-15 的历史入口；当前活跃提案是上方 V5–V7 v3.0 与 V5.1-R3。V4.2–V4.7 已执行并判定负结果，保留为冻结证据。

- [双端训练与项目进度统一方案v4](plans/PES2TS_双端拓扑条件训练与项目进度统一方案_v4_20261015.md)：唯一当前训练排期入口；B节证据基线，C节阶段进度，D节WP0–WP7，E节冲突裁决，F节预算，G节验收，I节用户完整规格。
- [ADR-0021](design/decisions/ADR-0021-无序双端拓扑与完整双侧TS教师主线.md)：用户确认的无序双端/完整双侧TS主线，取代旧有向任务近期安排。**Superseded for V5.2+ by ADR-0030**（V5.2+ 语义以 [ADR-0030](design/decisions/ADR-0030-V5.2全路径导航与端点停止.md) 条款级取代为准）。
- [ADR-0022](design/decisions/ADR-0022-v4.1因果输入合同与特征v3.md)：v4.1 WP1 因果输入合同——教师标签派生的 5/6 通道移出网络输入，特征升级 `pesn_dual_ts_features_v3`（观测-only 5 通道）。
- [ADR-0026](design/decisions/ADR-0026-V5冲刺key-atom位移合同与OOF语义.md)：V5 冲刺（V5.0–V5.2 第一阶段）key-atom 位移合同——任务/帧合同/三跨度教师/冲突审计/5-fold split/特征 36-29-19/M2 Star EGNN/损失与 OOF 语义；配套战役 `V5_SPRINT` 与 `config/campaigns/v5_sprint.yaml`。
- [双端教师与相容转移合同v2草案](contracts/PES2TS_双端TS教师与相容转移合同_v2.md)：新任务接口设计登记，尚未发布validator/schema。
- [阶段定义](plans/PES2TS_阶段定义_G0-G3.md)、[整体发布](plans/PES2TS_整体开发与发布方案_v2.md)、[完整流程](plans/PES2TS_完整开发流方案_整合版_v1.md)、[宪法v2.7](PES2TS_开发宪法_v1.md)：G0–G4映射、研究适用与正式发布边界。

当前进度：旧fit50与训练链路有真实证据；新双侧任务WP0仅文档登记完成，代码/完整数据/训练/验收待执行。先双侧数据和动作语义→BC可学与自由运行→相容IQL，再条件500/2k三seed；SAC后置。G2严格链并行，正式STOP及生产能力未因本次整合升级。

## 2026-10-17 阶段结果与问题复核

[PES2TS v4 阶段训练——最终报告](reports/PES2TS_v4阶段训练_最终报告与问题复核_20261017.md)：WP2完整覆盖、五臂BC冻结门全部失败、WP6停靠/WP7冻结；补充源码与全fit50特征只读复核，确认教师动作近零掩码进入模型且离线/自由运行语义不同。原“已排除全部数据/泄漏缺陷”“多样性为已证实主因”“IQL共享采样全闭合”需收窄；详见报告P01/P05/P06。此链接更新事实入口，不改写历史门审/冻结记录，不启动新方案。上文10月15日“仅规划、待实现”状态保留其历史时点，不作10月17日总进度。

[PES2TS v4.1 阶段训练——修复输入语义与最小实验矩阵·最终报告](reports/PES2TS_v4.1阶段训练_修复输入语义与最小实验矩阵_最终报告_20261017.md)：v4.1 修复 P01（观测-only 5 通道/ADR-0022/两条验收测试）并重建可信基建后按冻结协议完成 A/B/C × 3 种子矩阵。注册门结果 REGISTER_ABLATION（0/3 种子×3 臂四项全不满足，非边缘差距）；B 臂 NOT EFFECTIVE、D 跳过；C 臂增益仅中位层面成立；A-vs-v4 包回归成立但归因未定（输入×采样器未分离）；v4.1 checkpoint 实测幅度收缩持续且随臂异；A45 种子散布内无实质变化。附三选项下一阶段决策包（包归因 2×2 → 扩数据 → 架构增强）。报告只读，不改写任何历史门审/冻结件。

## 以下为旧版本记录与专题索引

下文“当前/下一阶段/未启动”及旧费用估算均保留当时版本语境；旧报告和已验收证据不改写。**2026-10-18 注**：早期报告中的 `E:/...` 与 `outputs/...` 链接是当时本机路径，仅作证据文本，不保证在 GitHub 可解析。


WSL专项/训练链路：实施计划v3.4 §11.9–11.11。GPU环境与P0/P1已完成（§11.10）；批次1–6 direct链路已实现并真实执行：合同/split冻结（fit 4,964组）、共享推进器、ACP E/g接缝、**fit50 50/50 真实制备（3,612转移、3,662 QC、0失败、验收pass）**、EGNN/径向策略/概率合同、BC/IQL训练器。**fit50训练表征**：[分析报告](reports/PES2TS_Direct_fit50训练表征分析_20261015.md)——BC发现并修复σ钳位饱和缺陷（修复后150-epoch NLL −78.4、每原子log-density +4.99）；方向余弦0.163→辅助方向损失版本有登记依据；IQL真实数据3,000更新（ESS≈均匀=重加权无信号）、训练器~7×提速。全仓2052测试通过。**待决**：a_max校准（0.2投影几何可达54%）、fit500/2k正式数据（500定价70.2h串行/35.1h二路）、三seed正式训练、严格链与SAC。

2026-10-15 更新。代码与数据说明见仓库根 [README](../README.md)；本目录收集计划、设计、合同与报告。

最新下一步训练入口：[EGNN→IRC-BC→IQL→真实SAC实施与训练计划v3](plans/PES2TS_EGNN_BC_IQL_SAC实施与训练计划_v3_20261015.md)与[ADR-0020草案v1.3](design/decisions/ADR-0020-EGNN直接位移与BC-IQL-SAC研究主线.md)。本轮主目标2k独立fit组、50/500前置、fit12真实SAC首包；含文件级代码矩阵、共享推进器、失败奖励/无量纲熵、严格链前置、预算与验收。元数据实查旧5k为4,994组，17组开发替补交叉，剩4,977候选；[名单审计](plans/fixtures/pes2ts_direct_training_input_audit_20261015.json)与核查工具已交付，正式族/用途资格和新业务实现/训练/QC仍planned。v3取代v2下一步实施安排。

基础设施增补：实施计划现为v3.1（§11.3–11.8），含本机资源核查、单GPU训练/多环境采集、ACP全局QC配额、内存/显存保护、WP-INFRA0–3与Gate-INFRA。起点2×2 QC线程；旧训练CUDA接线与实际运行环境待验收，未启动训练/QC。

研究细化：[IRC → BC → IQL → SAC 直接位移方案](plans/PES2TS_IRC_BC_IQL_SAC直接位移研究方案_v1_20261015.md)（Proposed）：用户确认的新方向；以马尔可夫状态、等变原子位移和同一纯位移/观测环境衔接专家预训练、离线优势学习与真实交互。明确奖励迁移、策略密度、IQL→SAC价值校准及冻结严格评测；新合同/代码/训练/QC尚未实施，默认与历史产物保持。

研究提案：[RL 自适应路径生长与 IRC 验证](plans/PES2TS_RL自适应路径生长与IRC验证研究方案_v1_20261009.md)（Proposed）：承接首批 47 fit/46 dev 实测，研究解除固定路径与同反应探索限制、物理横向松弛和分批 on-policy 学习；含四组消融、新小包预算及严格 IRC 证据边界。本次仅规划，尚未替代当前默认或启动新计算。

> 规范根（SSOT）见 [PES2TS_开发宪法_v1](PES2TS_开发宪法_v1.md)；文档治理见 [PES2TS_文档修订台账](plans/PES2TS_文档修订台账.md)：唯一主干编号为 **G0–G4**，工作流为 **C（清理）/ R（生成研究）/ X（执行收敛）**。执行投影见仓库根 [AGENT.md](../AGENT.md)。

## 权威文档（规范/入口；历史项已标状态）

| 文档 | 内容 |
| --- | --- |
| [PES2TS_开发宪法_v1.md](PES2TS_开发宪法_v1.md) | **规范根（SSOT，内容 v2.7）**：身份、红线、接口、统一残差/MDP研究入口、预测器隔离与正式决策；冲突以本文件为准 |
| [统一残差决策与G2有限回退实施方案](plans/PES2TS_统一残差决策与G2有限回退实施方案_v1_20261007.md) | **统一设计与A/B/C fixture基线**：真实round0证据及对照缺陷见Demo24/下一轮纠错方案；新修复与收益待实施验收。AB4（包D）独立门保留；先RL后扩量、共享编码器/残差/Q及G2单步/质量/费用 |
| [IRC20k与RL2000下一阶段详细方案](plans/PES2TS_IRC20k与RL2000下一阶段详细方案_v1_20261008.md) | **当前执行细化v1.0，规划完成**：四项审核补修、组级用途/隔离、IRC20k与新RL2000计数、嵌套扩量、跨反应支持、迁移/指标/停止门、主实验982小时15分预算与独立评测；新训练/QC未启动 |
| [ADR-0019](design/decisions/ADR-0019-IRC20k与RL2000分阶段扩量.md) | **Proposed规划**：独立反应计数、分批推进、跨任务排除、支持适用与全预算；接口实施另审查 |
| [RL下一轮代码修复、训练与调整方案](plans/PES2TS_RL下一轮代码修复训练与调整方案_v1_20261008.md) | **上一轮纠错算法细化v1.1**：第一批BC修复/纠错训练证据成立，四项审核缺口未关闭；扩量和预算由IRC20k/RL2000方案承接，WP-2及真实收益未全部验收 |
| [ADR-0018](design/decisions/ADR-0018-RL纠错与动作支持研究计划.md) | **Proposed（规划登记）**：动作支持、快照/分支、成对监督及新合同草案；正式schema/能力升级待实施审查 |
| [Demo24真实RL闭环与对照实施方案](plans/PES2TS_Demo24_真实RL闭环与对照实施方案_v1_20261007.md) | **round0真实闭环已运行，2026-10-08发现对照资格缺陷**：258转移/51探索、369次模型非零执行证据保留；BC实际按CQL训练且两包同权重，不能当纯BC/CQL对照；Q标签支持门全回退，收益未成立。旧产物保留，下一轮按纠错方案推进 |
| [ADR-0017](design/decisions/ADR-0017-统一残差决策与G2有限回退.md) | **Accepted（规划）**：局部取代步长分类优先顺序；MDP近似、原生/混合环境冻结、宏步/控制/bundle新版本草案 |
| [PES2TS_阶段定义_G0-G3.md](plans/PES2TS_阶段定义_G0-G3.md) | **阶段与边界定义唯一来源（内容 v4.7）**：G0–G4、G3.P并行、统一研究入口及正式退出条件；V5–V7 为 V 战役，不是新 G 阶段 |
| [PES2TS_整体开发与发布方案_v2.md](plans/PES2TS_整体开发与发布方案_v2.md) | **项目级开发与发布主方案（内容 v3.5）**：双线依赖、预测器在线/离线分发、研究/正式能力分开发布、模型版本与兼容/回滚 |
| [PES2TS_PES-N未来开发计划_v2_20261005.md](plans/PES2TS_PES-N未来开发计划_v2_20261005.md) | **历史（已被 V5–V7 v2.0 取代；内容 v2.5）**：统一研究闭环先于扩量；正式STOP→残差→离线Q→闭环、数据/接口和数学边界保持 |
| [G2-AB4 ORCA驱动xTB试点](plans/PES2TS_G2_AB4_ORCA驱动xTB约束优化试点方案_20261007.md) | **当前小批方案 v1.1，未实施**：原3失败点+1正常点/60秒及回退门保持；统一G2单步和逻辑目标配额；策略不选优化器 |
| [ADR-0016](design/decisions/ADR-0016-ORCA驱动xTB硬约束优化试点.md) | **Accepted（设计）**：外部xTB/ORCA新配方、请求/收据v1及混合结果/trace v2草案；真实能力另验收 |
| [轻量控制与RL衔接计划](plans/PES2TS_轻量控制与RL衔接计划_v1_20261007.md) | **当前增量计划 v1.2**：研究入口保持，近期顺序由统一方案承接；单一模型/逐坐标候选Q，原有限步长为可选对照 |
| [ADR-0015](design/decisions/ADR-0015-G2稳定底座与轻量RL研究入口.md) | **Accepted（规划）**：轻量研究入口可并行，复杂物理后置；正式阶段/默认门与R1–R6保持 |
| [G3-P1 IRC教师预训练与模型版本计划](plans/PES2TS_G3_P1_IRC教师预训练与模型版本计划_v1_20261006.md) | **当前第一期训练方案 v1.2**：G3.P.0–.3；`pesn-irc-v0.1.0`；B0–B7预测器代码范围、来源/分裂、标签/模型、科研门；实施进行中，具体能力依任务证据 |
| [预测器产品与G3代码模块划分](design/PES2TS_预测器产品与G3代码模块划分_v1_20261007.md) | **代码范围实施设计 v1**：现有源码对照、产品责任、目标文件树、接口/配置/依赖、导入/真值/导出检查与验收 |
| [ADR-0014](design/decisions/ADR-0014-预测器产品归属与G3代码范围.md) | **Accepted（规划）**：G3-P1新增业务代码统一pesn；offline归产品内，runtime独立；guard窄桥待实现 |
| [ADR-0013](design/decisions/ADR-0013-G3并行IRC预训练与模型版本体系.md) | **Accepted（规划）**：G3.P不依赖G2调教，保留正式门；软件/模型/数据独立版本及隔离教师合同；未升级能力 |
| [Classic图论驱动与xTB原生约束扫描设计](design/PES2TS_Classic图论驱动与xTB原生约束扫描设计_v1_20261006.md) | **当前Classic改造设计**：功能复用、关键原子初猜、原生逐点优化、收敛/软约束/物理能量、费用与后续兼容；AB3已实现，正式出口另验收 |
| [G2-AB3功能整合与原生扫描改造方案](plans/PES2TS_G2_AB3功能整合与原生扫描改造方案_20261006.md) | **当前G2批次设计+实施（2026-10-07 WP-0…9 已跑通离线门、4/24例真实对照；默认未切）**：审计修补、ACP公开入口/收据、WP-0…10、离线→真实4/24例→严格验证 |
| [ADR-0012](design/decisions/ADR-0012-Classic图论驱动与xTB原生约束扫描.md) | **Accepted（设计方向）**：ACP/xTB原生优化、稀疏初猜与新合同草案；未切生产默认 |
| [ADR-0010-PES-N反应状态价值与STOP-MOVE统一决策.md](design/decisions/ADR-0010-PES-N反应状态价值与STOP-MOVE统一决策.md) | **Accepted（开发方向）**：共享状态/动作价值及规范同步；代码/模型能力仍需验收 |
| [ADR-0011-三模块产品架构与统一可视化.md](design/decisions/ADR-0011-三模块产品架构与统一可视化.md) | **Accepted（产品方向）**：拓扑约束＋控制器＋ACP计算器接缝，横向可视化、多输入及极简整合；schema/UI/真实能力另验收 |
| [ADR-0001-统一执行后端接缝.md](design/decisions/ADR-0001-统一执行后端接缝.md) | 执行层统一决策（Track A/B 收口）；本地封装条款已被 ADR-0002 部分取代 |
| [ADR-0002-计算后端统一经ACP执行.md](design/decisions/ADR-0002-计算后端统一经ACP执行.md) | **计算后端统一经 ACP 执行**（部分取代 ADR-0001；含等价证书、仓库边界、可视化桥） |
| [ADR-0003-G2-AB1接口登记.md](design/decisions/ADR-0003-G2-AB1接口登记.md) | **G2-AB1 批次接口登记**（执行身份/起点证据/梯度候选→验证/成本与防火墙；P1 存在性+校验） |
| [PES2TS_ACP执行统一与旧代码清理方案_20261003.md](plans/PES2TS_ACP执行统一与旧代码清理方案_20261003.md) | **G2.4 执行统一实施计划**（X1′–X5′、文件级改动、验证门） |
| [PES2TS_G2后续开发方案_Classic到Rank到Gen_20261003.md](plans/PES2TS_G2后续开发方案_Classic到Rank到Gen_20261003.md) | **当前 G2 细化方案**：物理闭环、严格验证与标签/成本；G3/G4 学习安排已由 PES-N 计划取代 |
| [PES2TS_G2_ACP能力补齐与联合验收计划_20261003.md](plans/PES2TS_G2_ACP能力补齐与联合验收计划_20261003.md) | **跨仓配套计划**：ACP 能力矩阵、8 个任务包、请求/产物合同、真实冒烟与迁移门 |
| [PES2TS_G2_PESG收紧算法迭代方案_20261003.md](plans/PES2TS_G2_PESG收紧算法迭代方案_20261003.md) | **G2 数学与算法细化**：KKT 分支、度规投影、图保护、对称条件、三类路径对照与 A–I 工作包 |
| [PES2TS_G1_G2边界审计与补齐任务_20261003.md](plans/PES2TS_G1_G2边界审计与补齐任务_20261003.md) | **边界审计与补齐**：资产抽样、辅助映射/IRC 筛选来源、方法证据、全图立体 authority 与六个任务 |
| [PES2TS_文档修订台账.md](plans/PES2TS_文档修订台账.md) | 文档状态与取代关系 |
| [PES2TS_G2_下一批算法调整与测试方案_20261003.md](plans/PES2TS_G2_下一批算法调整与测试方案_20261003.md) | **下一批批次方案（G2-AB1，v2 宪法对齐）**：分支稳定与验证接通的 7 个工作包（实现+测试），引用宪法 R1–R6/P1–P5，范围锁定 PES2TS 侧算法核心；**2026-10-04 离线实施完成（V0 门通过，接口登记见 [ADR-0003](design/decisions/ADR-0003-G2-AB1接口登记.md)）** |
| [PES2TS_G2_下一批实现方案_真引擎接通与验收_20261004.md](plans/PES2TS_G2_下一批实现方案_真引擎接通与验收_20261004.md) | **下一批批次方案（G2-AB2 v2）**：以 G2.T 24 例首轮端到端战役为主轴——试点 4 例（=V2 门）→ 全量 24 例四类终态 + 成本账本 → 验证链实跑（双轨，复核到位后）→ **第一份真实反馈报告**（失败直方图/B1 对照/AB3 改进清单）；不承诺 M0。实施：S1 离线交付全落地（战役编排器 `g2t-*` CLI、校准网格冻结、起点 ACP 适配器 fail-closed、反馈报告器、收口脚本；ADR-0004 + [战役合同字段字典](contracts/PES2TS_G2T_战役合同字段字典_v1.md)）；**S1–S5 已执行（2026-10-05，D1–D5 交付）；S4 gated（人工复核未满足）** |
| [PES2TS_G2T真实战役执行方案_20261004.md](plans/PES2TS_G2T真实战役执行方案_20261004.md) | **G2.T 真实战役执行方案（G2-AB2 S2–S5 执行阶段，v1）**：细化 G2-AB2 v2 的 S2–S5 真实执行；Wave 0 四项执行前修补已实施并通过离线回归门（证据 `outputs/g2t_campaign_logs/T2..T6`，判据修订以 [ADR-0005](design/decisions/ADR-0005-G2-AB2执行前判定器与恢复修补.md) 为准）；七组件阶段表、测试与输出纪律、λ 绑定法、S4 双轨与诚实边界；**S1–S5 已执行（2026-10-05，D1–D5 交付）；S4 gated（人工复核未满足）**；执行期事件 T17b/T18b 已修 |
| [PES2TS_完整开发流方案_整合版_v1.md](plans/PES2TS_完整开发流方案_整合版_v1.md) | **流程总纲（内容 v2.4）**：预测器归属、共享内核、G3.P并行及G3.1初始化交接、G0–G4放行门 |
| [`AGENT.md`](../AGENT.md)（仓库根） | **执行投影**：宪法执行投影（红线 ID 摘要、数学证据边界、架构变更审查协议） |

## 补充索引（2026-10-18）

| 文档 | 状态 | 内容 |
| --- | --- | --- |
| [G–V 映射表](design/PES2TS_G_V_MAPPING_v1_20261018.md) | **权威（生成）** | G 能力 × V 战役矩阵；由 `config/registry/g_v_map.yaml` 生成，`--check` 校验 |
| [ADR-0025](design/decisions/ADR-0025-G-V双轴映射与战役注册.md) | **Accepted** | G = 代码/物理能力，V = 训练/测试战役；RunRecord `gv` 块与命名规则 |
| [代码地图 v2](design/PES2TS_CODE_MAP_v2_20261018.md) | **参考（非规范）** | 阶段代码根、V4/V5/V6/V7 落点、导入边界与存储分级（v2.2，2026-10-20 reaction-intent 修订） |
| [V5–V7 架构计划 v1.2](plans/PES2TS_PES-N_V5-V7架构调整与实施计划_v1.2_20261018.md) | **Superseded（2026-10-20）** | 保留为历史；当前采用 [v3.0](plans/PES2TS_V5-V7_反应意图条件化与低维生长方案_v3.0_20261020.md) |
| [V5.0–V5.2 冲刺方案](plans/PES2TS_V5.0-V5.2_十小时冲刺训练与验证执行方案_v1_20261018.md) | **Executed-Frozen（V5.1a）/ Historical** | 数据/数学、架构/消融、冻结/OOF/封存三阶段；B3 已 rejected；V5.1b 关闭 |
| [V4.2–V4.7 实验方案](plans/PES2TS_V4.2-V4.7_十小时训练实验方案_20261017.md) | **Executed-Frozen** | 23 个正式 run；10-18 判定负结果，保留证据 |
| [V4.2–V4.7 实施合同与战役冻结](contracts/PES2TS_V4.2-V4.7_实施合同与战役冻结_v1.md) | **Executed-Frozen** | 冻结配方/战役合同；不改写为 V5 |
| [ADR-0023](design/decisions/ADR-0023-V4.2原生重连数据与内部开发划分.md) | **Accepted（历史）** | V4.2 原生重连数据与内部开发划分 |
| [ADR-0024](design/decisions/ADR-0024-V4.2无界位移模型与战役训练协议.md) | **Accepted（历史）** | V4.2 无界位移模型与战役训练协议 |
| [10-18 实现报告](reports/PES2TS_实现报告_20261018.md) | **Evidence** | V4.7 负结果与工程修复记录；只读 |
| [Archive 索引](archive/README.md) | **Evidence** | 7 个版本快照目录的日期、范围与链接 |

## plans/ — 项目级计划与验收

| 文档 | 状态 | 内容 |
| --- | --- | --- |
| [V5–V7 反应意图条件化与低维生长方案 v3.0](plans/PES2TS_V5-V7_反应意图条件化与低维生长方案_v3.0_20261020.md) | **Active-Proposed** | V5 Reaction-Conditioned Geometric Policy / V6 Physics-Corrected Reaction Growth / V7 Cost-Aware Autonomous PES Exploration；取代 v2.0；**内容 v3.2（2026-10-21 任务纠偏修订）** |
| [ReactionIntent Contract v1](contracts/PES2TS_ReactionIntent_Contract_v1.md) | **Superseded（§4 Stop/q 条款由 [v2](contracts/PES2TS_ReactionIntent_Contract_v2.md) 取代；ADR-0030）** | `G_rxn`、signed graph edit、CV/Jacobian、Relabel/Reverse、输入合法性 |
| [V5.1-R3 反应意图条件化与端点标签不变重训方案 v3](plans/PES2TS_V5.1-R3_反应意图条件化与端点标签不变重训方案_v3_20261020.md) | **Active-Proposed** | 无 branch、signed intent、B0-F 重训、strong/dead 重校准；不做 CV/Progress/Stop |
| [V5.1-R3 数据库+特征+损失三位一体修复方案 v1](plans/PES2TS_V5.1-R3_数据库特征损失三位一体修复方案_v1_20261020.md) | **Active-Proposed（执行细化）** | 五表数据合同、Reaction Intent 特征、B0-F/strong/dead、梯度与验收门；66-run |
| [单端点 IRC 生成控制器产品设计 v3](plans/PES2TS_单端点IRC生成控制器_产品设计_v3_20261020.md) | **Active-Proposed** | 曲线中心、PREDICT-ACTION-RELAX、Known-B/TS-only/Unknown-B |
| [30h R3 → V5.2 → V5.3 分层训练与验收计划 v1.1](plans/PES2TS_30h_R3_V52_V53_分层训练与验收计划_v1.1_20261020.md) | **Executed-Frozen / Stop 语义已被 ADR-0030 取代** | R3 66 + V5.2 33；R3 语义仍有效，TSStop 章节仅历史；First-stop/两段训练、V5.3 rollout、G0–G4、DAG/测试清单 |
| [V5–V7 三阶段研究与架构修订方案 v2.0](plans/PES2TS_V5-V7_三阶段研究与架构修订方案_v2.0_20261020.md) | **Superseded by v3.0** | 历史：V5 Geometry Policy / V6 xTB / V7 cost-aware RL |
| [V5.1-R2 损失尺度与数据完整性修复方案](plans/PES2TS_V5.1-R2_损失尺度与数据完整性修复方案_v1_20261020.md) | **Superseded by R3** | 历史：无量纲 strong/dead、S1 审计；特征不变假设已废弃 |
| [统一残差决策与G2有限回退实施方案](plans/PES2TS_统一残差决策与G2有限回退实施方案_v1_20261007.md) | **已实施（fixture_passed）；待真实运行** | 六包交付顺序、统一网络/MDP、真实宏步和AB4有限救援，先RL后扩量（§10 实施记录） |
| [PES2TS_G3_P1_IRC教师预训练与模型版本计划_v1_20261006.md](plans/PES2TS_G3_P1_IRC教师预训练与模型版本计划_v1_20261006.md) | **实施进行中，验收依任务证据** | G3-P1版本路线、方法/分裂/污染审计、教师进度/方向/协同、多任务小模型、扩量与研究冻结 |
| [PES2TS_G2_AB3功能整合与原生扫描改造方案_20261006.md](plans/PES2TS_G2_AB3功能整合与原生扫描改造方案_20261006.md) | **已实现并有24例真实产物；未切默认** | 功能整合、原生优化接缝、13/9/2终态与兼容；严格validated=0 |
| [PES2TS_PES-N未来开发计划_v2_20261005.md](plans/PES2TS_PES-N未来开发计划_v2_20261005.md) | **历史（已被 V5–V7 v1.2 取代）** | 共享 PES-N 内核、单点 STOP 监督、残差/离线 Q、真实闭环、跨来源评测与小模型目标 |
| [PES2TS_G0-G4整体开发框架修改方案_v1_20261005.md](plans/PES2TS_G0-G4整体开发框架修改方案_v1_20261005.md) | **历史提案，已被 PES-N 取代** | 保留共享双输出的原先框架；当前学习目标与阶段安排见 PES-N 计划 |
| [PES2TS_拓扑约束TS接近度与位移联合学习方案_v2_20261004.md](plans/PES2TS_拓扑约束TS接近度与位移联合学习方案_v2_20261004.md) | **历史模型提案，已被 PES-N 取代** | 距离/直接位移不再为主接口；有明确标签时可作辅助任务/残差候选 |
| [PES2TS_统一价值模型与搜索排序一体化发展方案_v1_20261004.md](plans/PES2TS_统一价值模型与搜索排序一体化发展方案_v1_20261004.md) | **历史提案，最新由 PES-N 承接** | 保留初版 G/R 共享论证；STOP/MOVE 的当前严格定义及阶段见 ADR-0010 |
| [PES2TS_开发宪法_v1.md](PES2TS_开发宪法_v1.md) | **权威（SSOT）** | 规范根：红线/政策/接口/阶段命名/评测/修订 |
| [PES2TS_阶段定义_G0-G3.md](plans/PES2TS_阶段定义_G0-G3.md) | **权威** | 阶段与边界定义 |
| [PES2TS_整体开发与发布方案_v2.md](plans/PES2TS_整体开发与发布方案_v2.md) | **权威** | 开发与发布主方案 |
| [PES2TS_文档修订台账.md](plans/PES2TS_文档修订台账.md) | **权威** | 文档治理与取代关系 |
| [PES2TS_ACP执行统一与旧代码清理方案_20261003.md](plans/PES2TS_ACP执行统一与旧代码清理方案_20261003.md) | **执行方案** | G2.4 执行统一：全后端经 ACP、删本地引擎、可视化桥、X1′–X5′ |
| [PES2TS_G2后续开发方案_Classic到Rank到Gen_20261003.md](plans/PES2TS_G2后续开发方案_Classic到Rank到Gen_20261003.md) | **G2 细化，后续学习已取代** | G2 规则选点、严格验证、全帧留数与成本保留；G3/G4 见 PES-N |
| [PES2TS_G2_ACP能力补齐与联合验收计划_20261003.md](plans/PES2TS_G2_ACP能力补齐与联合验收计划_20261003.md) | **配套开发方案** | ACP 已有/部分/缺失能力；梯度、约束、验证、计量、身份桥 |
| [PES2TS_G2_PESG收紧算法迭代方案_20261003.md](plans/PES2TS_G2_PESG收紧算法迭代方案_20261003.md) | **当前算法细化设计** | 数学成立条件、可实现迭代、能力依赖、解析反例与消融门 |
| [PES2TS_G1_G2边界审计与补齐任务_20261003.md](plans/PES2TS_G1_G2边界审计与补齐任务_20261003.md) | **抽样审计/合同草案** | 183,460 ID 对账、五样本、provenance 分层、新边界旁车 |
| [PES2TS_G2_下一批算法调整与测试方案_20261003.md](plans/PES2TS_G2_下一批算法调整与测试方案_20261003.md) | **批次方案（G2-AB1，v2；已离线实施）** | WP-1…WP-7：执行身份/分支稳定(A(H))/起点证据/验证接通/landing 身份/数学反例/成本与分裂防火墙；宪法对齐，离线与 gated 验收门；V0 已通过，V2/V3 待真实环境 |
| [PES2TS_G2_下一批实现方案_真引擎接通与验收_20261004.md](plans/PES2TS_G2_下一批实现方案_真引擎接通与验收_20261004.md) | **批次方案（G2-AB2 v2；S1–S5 已执行，D1–D5 交付，S4 gated）** | 主轴 = G2.T 24 例首轮端到端：战役编排器/试点 4 例（V2）/全量 24 例四类终态 + 账本/验证链实跑/第一份反馈报告；接口登记 ADR-0004 |
| [ADR-0004-G2-AB2接口登记.md](design/decisions/ADR-0004-G2-AB2接口登记.md) | **G2-AB2 批次接口登记**（战役合同/typed 终态/起点适配器/校准与反馈报告；P1 存在性+校验） |
| [ADR-0005-G2-AB2执行前判定器与恢复修补.md](design/decisions/ADR-0005-G2-AB2执行前判定器与恢复修补.md) | **G2-AB2 执行前修补登记**（判定器 v2 语义 + 恢复顺序/缓存复用证据/起点 profile 兼容；先修订后执行，非事后放宽） |
| [PES2TS_G2T真实战役执行方案_20261004.md](plans/PES2TS_G2T真实战役执行方案_20261004.md) | **执行方案（G2-AB2 S2–S5 执行阶段；S1–S5 已执行，D1–D5 交付，S4 gated）** | 战役 S2–S5 执行细化：Wave 0 四修补记录（ADR-0005）/七组件阶段表/测试与输出纪律/λ 绑定法/预算外推占位/S4 双轨/诚实边界；执行期事件 T17b/T18b 已修；细化不取代 G2-AB2 v2 |
| [PES2TS_G2T_战役合同字段字典_v1.md](contracts/PES2TS_G2T_战役合同字段字典_v1.md) | G2.T 战役接口存在性与校验要求（manifest/终态/账本/校准/判定/报告/起点适配器） |
| [PES2TS_完整开发流方案_整合版_v1.md](plans/PES2TS_完整开发流方案_整合版_v1.md) | **流程总纲 v2.4** | 预测器归属、PES-N共享内核、G3.P并行/交接、G2关键路径和正式门 |
| [../AGENT.md](../AGENT.md) | **执行投影** | 宪法执行投影：红线 ID 摘要、证据边界纪律、架构变更审查协议 |
| [PES2TS_双项目开发与Demo方案.md](plans/PES2TS_双项目开发与Demo方案.md) | 参考 | 定位（generation / ranking）、合同、指标 |
| [PES2TS_统一开发顺序与阶段验收.md](plans/PES2TS_统一开发顺序与阶段验收.md) | 历史 | S0–S8 阶段表（已并入 G0–G4） |
| [PES2TS_逐步实施与验证方案.md](plans/PES2TS_逐步实施与验证方案.md) | 历史 | 早期分步方案 |
| [PES2TS_实施路线图.html](plans/PES2TS_实施路线图.html) | 参考 | 实施路线图（浏览器打开） |
| [PES2TS_扫描规划谱系合并方案_20260930.md](plans/PES2TS_扫描规划谱系合并方案_20260930.md) | 参考 | 规划两代谱系归一裁决（已并入 G2.4/X） |
| [PES2TS_第二轮修改最终方案_仅完全成断键_20261002.md](plans/PES2TS_第二轮修改最终方案_仅完全成断键_20261002.md) | 参考 | R 工作流范围与验收来源 |
| [PES2TS_图论立体先验与单端协同延续优化方案_v2_20261002.md](plans/PES2TS_图论立体先验与单端协同延续优化方案_v2_20261002.md) | 参考 | R 工作流 M0–M4 来源 |

## design/ — 阶段设计文档

| 文档 | 内容 |
| --- | --- |
| [ADR-0030](design/decisions/ADR-0030-V5.2全路径导航与端点停止.md) | **Accepted（2026-10-21）**：V5.2 TS 邻域 Stop 属任务定义错误；Full-Path Navigation（R→TS→P / P→TS→R）+ Endpoint Stop（d_end 0.10/0.30 Å）+ directed q ordering；条款级取代 ADR-0021/0028/0029 相关语义；唯一任务合同见 [FullPathNavigationContract v1](contracts/PES2TS_FullPathNavigationContract_v1.md) |
| [GeometryPolicyContract v1](contracts/PES2TS_GeometryPolicyContract_v1.md) | **Superseded（§4.2/§5.2，由 [GeometryPolicyContract v2](contracts/PES2TS_GeometryPolicyContract_v2.md) 取代；ADR-0030）**：`π(X_t,G) → (ΔX_K,m)`、K 定义、No-Move 曲线停止语义、在线禁止项与验证点 |
| [ADR-0027](design/decisions/ADR-0027-V5.1参考架fallback与双分母评估.md) | **Proposed**：gauge fallback 研究门；不实现新 fallback；主分母不变 |
| [ADR-0028](design/decisions/ADR-0028-关键原子几何策略合同与V5-V7任务拆分.md) | **Historical-Proposed**：数据边界/旧证据冻结仍有效；branch/旧 V5.2 语义由 ReactionIntent Contract v1 取代；**Partially superseded by ADR-0030**（§2.1 No-Move=端点停止，不称 OptTS 候选） |
| [ADR-0029](design/decisions/ADR-0029-V5.2+双门评估与封存边界.md) | **Proposed**：监督门 + 物理搜索门；G5.2–G5.4 保留；**Partially superseded by ADR-0030**（§2.2 Stop 指标改为端点到达/位置/起点与 TS 误停） |
| [预测器产品与G3代码模块划分 v1](design/PES2TS_预测器产品与G3代码模块划分_v1_20261007.md) | **当前实施范围**：源码对照、产品/文件责任、B0–B7、运行接口、离线训练、配置/依赖及测试/包导出；待实现 |
| [ADR-0014](design/decisions/ADR-0014-预测器产品归属与G3代码范围.md) | **Accepted（规划）**：部分取代ADR-0013目录建议，所有新增业务代码在预测器，在线/离线隔离 |
| [ADR-0013-G3并行IRC预训练与模型版本体系.md](design/decisions/ADR-0013-G3并行IRC预训练与模型版本体系.md) | **Accepted（规划）**：G3.P、G3-P1、首代研究模型/合同版本，真实决策与物理门保留 |
| [PES2TS_Classic图论驱动与xTB原生约束扫描设计_v1_20261006.md](design/PES2TS_Classic图论驱动与xTB原生约束扫描设计_v1_20261006.md) | **当前Classic设计**：图/角色/K、稀疏预测、ACP/xTB逐点松弛、真实收敛/能量/费用、版本与候选；实现待验收 |
| [ADR-0012-Classic图论驱动与xTB原生约束扫描.md](design/decisions/ADR-0012-Classic图论驱动与xTB原生约束扫描.md) | **Accepted（设计方向）**：细化控制/优化分离，登记具体合同草案，不重定义旧产物 |
| [PES2TS_三模块与可视化产品架构_v1_20261005.md](design/PES2TS_三模块与可视化产品架构_v1_20261005.md) | **当前产品架构设计 v1.1**：三模块运行视图、预测器独立代码归属、G1/G2复用、不同完整度输入、ACP展示及分期验收；未实现模块或UI |
| [ADR-0011-三模块产品架构与统一可视化.md](design/decisions/ADR-0011-三模块产品架构与统一可视化.md) | **Accepted（设计方向）**：三模块与一个工作台，沿用ACP边界/合同族/阶段门 |
| [ADR-0010-PES-N反应状态价值与STOP-MOVE统一决策.md](design/decisions/ADR-0010-PES-N反应状态价值与STOP-MOVE统一决策.md) | **Accepted（规划生效）**：PES-N 核心及数学/数据/执行边界；同步规范和阶段，能力不升级 |
| [ADR-0009-G阶段下的统一模型开发框架.md](design/decisions/ADR-0009-G阶段下的统一模型开发框架.md) | **历史 Proposed，已由 ADR-0010 取代**：保留 G0–G4/双输出原论证，当前采用 PES-N 状态/动作价值 |
| [ADR-0008-拓扑约束下的距离与位移联合预测.md](design/decisions/ADR-0008-拓扑约束下的距离与位移联合预测.md) | **历史 Proposed，已由 ADR-0010 取代**：距离/直接位移主接口降为辅助，化学与原生优化边界承接 |
| [ADR-0007-统一价值模型与搜索内生选点.md](design/decisions/ADR-0007-统一价值模型与搜索内生选点.md) | **历史 Proposed，最新见 ADR-0010**：共享核心承接为当前 STOP/MOVE 价值模型 |
| [PES2TS_PES-C到PES-G架构与迁移方案_v1_20261004.md](design/PES2TS_PES-C到PES-G架构与迁移方案_v1_20261004.md) | **原生优化实施参考**：控制/原生优化分工由 ADR-0010 承接；G/R 学习顺序已取代；生产迁移仍待验收 |
| [ADR-0006-ORCA原生约束优化与控制策略分离.md](design/decisions/ADR-0006-ORCA原生约束优化与控制策略分离.md) | **迁移实现待验收**：控制/原生优化方向由 ADR-0010 承接；旧战役和能力证据不改写 |
| [G1_TS_IRC_映射与反应归簇实现方案.md](design/G1_TS_IRC_映射与反应归簇实现方案.md) | G1 真值层（P1 映射 / P2 分类）设计 |
| [G1_v2_补全实施总方案.md](design/G1_v2_补全实施总方案.md) | G1 v2 完成定义、F/B/O 语义、G1-0…G1-7 关卡（外部工作流已并入图论设计 P0–P3） |
| [G1_v2_L0到G2扫描策略设计.md](design/G1_v2_L0到G2扫描策略设计.md) | L0 群体与扫描路由分析 |
| [G1_v2_ORCA扫描模式识别与计划.md](design/G1_v2_ORCA扫描模式识别与计划.md) | ORCA 扫描模式识别与计划 |
| [G1_v2_一维优先与xTB二维成本策略.md](design/G1_v2_一维优先与xTB二维成本策略.md) | 一维优先与 xTB 二维网格成本分析 |
| [G1_v2_产物成键逆向扫描优先策略.md](design/G1_v2_产物成键逆向扫描优先策略.md) | 产物侧逆向扫描与 H 转移捷径分析 |
| [G1_v2_方向无关成键锚点与双向扫描策略.md](design/G1_v2_方向无关成键锚点与双向扫描策略.md) | 方向无关锚点基线 |
| [G1_v2_多成键与少成键方向对比.md](design/G1_v2_多成键与少成键方向对比.md) | 混合选边（多/少成键）对比 |
| [G1_v2_成断键扫描方向选择可视化.html](design/G1_v2_成断键扫描方向选择可视化.html) | 交互式 F/B 选择可视化（浏览器打开） |
| [PES_GENERATION_图论扫描策略选择器设计_v1.md](design/PES_GENERATION_图论扫描策略选择器设计_v1.md) | 图论选择器设计 v1：三层图、11 策略、四角色坐标、P0–P3（实施并轨记录见头部） |

## contracts/ — 数据合同与对接规范

| 文档 | 内容 |
| --- | --- |
| [PES2TS_FullPathNavigationContract_v1.md](contracts/PES2TS_FullPathNavigationContract_v1.md) | **Active-Normative（唯一 Move/Stop/TS/q 任务合同，ADR-0030）**：Full-Path Navigation R→TS→P / P→TS→R；Move=R3 Full-Traversal 跨 TS；Endpoint Stop（d_end 0.10/0.30 Å）；`pes2ts_fullpath_nav_labels_v1`；docs gate |
| [PES2TS_ReactionIntent_Contract_v2.md](contracts/PES2TS_ReactionIntent_Contract_v2.md) | **Active（取代 v1 §4 Stop/q 条款；ADR-0030）**：q 分段归一化 −1/0/+1；Stop 仅目标端点；其余 G_rxn/Relabel/Reverse 语义延续 v1 |
| [PES2TS_GeometryPolicyContract_v2.md](contracts/PES2TS_GeometryPolicyContract_v2.md) | **Active（取代 v1 §4.2/§5.2；ADR-0030）**：No-Move = Endpoint Stop（非 TS 邻域/OptTS 候选）；三态端点标签与必守规则 |
| [PES2TS_PESN_IRC教师数据与模型包合同_v1.md](contracts/PES2TS_PESN_IRC教师数据与模型包合同_v1.md) | **登记草案内容v1.1**：TeacherState/Label/Split/FeatureSpec/Run/Bundle及PESNInput/TeacherPrediction，哈希、隔离与逐头能力；正式schema未发布 |
| [PES2TS_contracts_v1.md](contracts/PES2TS_contracts_v1.md) | 七类核心对象 + 附属对象的字段字典 |
| [PES2TS_ACP字段映射_v1.md](contracts/PES2TS_ACP字段映射_v1.md) | PES2TS 合同 ↔ ACP v2 字段映射 |
| [PES2TS_数据规范与ACP对接优化方案.md](contracts/PES2TS_数据规范与ACP对接优化方案.md) | 数据规范与 ACP 对接优化（设计建议） |

## demo/ — Demo24 流程

| 文档 | 内容 |
| --- | --- |
| [PES2TS_demo24_反应挑选与复核方案.md](demo/PES2TS_demo24_反应挑选与复核方案.md) | 24 条候选筛选硬条件、A–H 分层、双人复核与替补规则 |
| [PES2TS_Demo评估指标_v1.md](demo/PES2TS_Demo评估指标_v1.md) | Demo 指标定义（generation / ranking / 端到端 / 成本） |
| [PES2TS_Demo实现审核与架构解读_20260930.md](demo/PES2TS_Demo实现审核与架构解读_20260930.md) | Demo 实现审核（F1–F6 与阶段判断） |

## reports/ — 阶段性报告

| 文档 | 内容 |
| --- | --- |
| [PES2TS_PES-C-G架构审核与创新性可行性分析_20261004.md](reports/PES2TS_PES-C-G架构审核与创新性可行性分析_20261004.md) | 架构审核：逐步几何反馈、验证/标签接缝、Chemoton NT2/RMSD-PP 对照、创新性边界、v2 修改建议与可行性；未改业务代码或启动计算 |
| [PES2TS_开发阶段性报告_20260930.md](reports/PES2TS_开发阶段性报告_20260930.md) | 全阶段（G0/G1/G1 v2/G2/集成/Demo24）进展、风险与路线图 |

## archive/ — 计划历史快照

[2026-10-05 PES-N 改版前快照](archive/20261005_pre_pesn/README.md)保存六份入口文档的精确字节和 SHA256；仅作追溯，不作为现行规范。

[2026-10-06 G3.P 改版前快照](archive/20261006_pre_g3p/README.md)保存九份当前工作区入口的精确字节（含已有未提交编辑）及源码/配置/数据清单范围核验哈希；当前版本见 ADR-0013/G3-P1。

[2026-10-07 预测器范围改版前快照](archive/20261007_pre_predictor_scope/README.md)保存十三份文档及376个业务/测试/配置/数据清单文件的核验哈希；当前代码范围见ADR-0014和预测器模块划分。

## agent_runs/ — 30h 战役执行套件（非规范编排层）

[PES2TS-30H 并行执行模式](agent_runs/PARALLEL_EXECUTION_PLAN.md)：3-lane 并发、串行 gate、账本与 GPU 互斥；[启动前评审](agent_runs/BOOTSTRAP_REVIEW.md)含环境核查与 dry-run 结果。状态机与并发配置在 [state/](agent_runs/state/)。训练授权恒 false 直至用户显式批准；sealed/Confirm/OOD/Fit150+ 独立 token 恒 false。
