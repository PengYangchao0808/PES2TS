# PES2TS V5.4.3 双 Decoder 50K 全面开发与训练方案

## 固定 EGNN Encoder + 标准化几何潜桥 + MOVE/PROCESS 独立残差专家

> 状态：**Frozen-Preregistered v1.0**（2026-11-02；规划、预注册草案与执行设计，**不含**代码实现、训练执行或科学收益声明）
> 签认记录：主方案 §13.2 的 D1–D8 已由仓库所有者经 DSH 编排会话明确签认（2026-11-02），并附三条运行授权；治理落地见 [ADR-0034](../design/decisions/ADR-0034-V5.4.3-双Decoder与DecoderContextV1.md)。签认后不得按 seed17/dev 结果修改阈值、矩阵、seed 策略、损失配方或参数带。
> 层级：V5.4 子战役；`primary_g=G3.P.2`，冻结评估延伸至 `G3.P.3`；不新增 G 阶段
> 取代关系：不取代开发宪法、MoveProcessContract v1、ADR-0031/0032/0033、V5.4.1/V5.4.2 冻结产物或只读报告；本文件承接 V5.4.2 报告 §9 的交接，并细化用户提供的「V5.4.3 设计候选 v2.0」
> 上游依赖：V5.4.2 冻结报告与 A0×3 checkpoint；50K 冻结数据与 split；V541 共同 sampler；V542 `BackboneOutput`/`DirectMoveDecoder`/`ReactionProcessHeadV53MP`；V542 strict loader/identity/evaluator
> 下游消费者：V5.4.4 离线全 IRC Teacher 蒸馏、V6 物理反馈闭环、V7 成本感知自主探索；本文件不授予在线部署、生产 No-Move 或 verified-TS 资格
> 当前版本：v1.0 @ 2026-11-02（WP0 签认已完成，转 `Frozen-Preregistered`）
> 配套治理（WP0 产出，本文件只给草案）：ADR-0034、`config/campaigns/pes2ts_v543_dual_decoder_50k.yaml`、`scripts/agent_ops/v543_config_schema.py`、五类模板与 registry 条目

---

## 0. 执行摘要

### 0.1 结论先行

1. **架构方向成立，建议立项。** 把 V542 A0 从「可训练联合模型」拆成「一次性 Coarse Encoder + 参数自由 Geometric Latent Bridge + 两条独立残差 Decoder + 外层 Controller」，能够把 V5.4.3 的科学问题从「换 Backbone」收敛为「同一份分子空间表征，哪种 Decoder 最能产生下一步位移与反应状态」。这比继续堆 Backbone 更可归因，也为 V5.4.4/V6 留下稳定接口。
2. **提案需要 8 处工程/科学修正后才能冻结。** 其中 4 处是硬错误风险：`q0_raw` 当前没有公开出口；`pair_index/pair_edit` 是监督字段，不能作为 Bridge 输入；V542 的 `fixed_coverage_direction` 实际就是 sign 的重复叶子；PROCESS 的 `frame_mask` 不能沿用 MOVE 的 `label_status=="ok"`，否则 terminal 行会被误删。其余 4 处是设计缺口：Bridge 缺 key-key 邻接与 token 布局；8 通道基底的「覆盖率」单独使用会虚高；`q_corr` 在当前每曲线点数下接近退化；V542 实际 MOVE 损失没有 `λ_strong/λ_dead`。
3. **首轮正式矩阵建议收敛为 7 个候选：** MOVE `M1 Residual MLP / M2 GVP / M3 Vector-GNN`；PROCESS `P1 Residual MLP / P2 DeepSets / P3 Graph-GNN / P4 Graph Transformer`。M0/P0 为匹配基线；M4 Flow 为条件激活；M5 Diffusion 与 P5 GraphGRU 只保留接口和激活条件，不进入首轮正式预算。
4. **两条 Track 独立训练、独立 checkpoint、独立冠军。** 每条 Track 的正式候选在 36K 参数目标附近做 **±5% 参数匹配**（不是「≤40K」即可），以避免把容量收益误读为结构收益。最终整模 = 57,619 冻结参数 + MOVE Refiner + PROCESS Refiner；按 36K+36K 目标约 129.6K，带内最大约 133.2K，硬上限 150K。
5. **固定剂量 37,900 updates，但不再机械复制 400+37,500 的两阶段语义。** Encoder/D0/P0 全程冻结，400 updates 只作 LR 线性 warmup；之后恒定 3e-4。这样既保留与 V542 总剂量可比的历史锚点，又避免「W0 冻结 Encoder」在 Decoder-only 训练中没有意义。
6. **评估必须换分母、换指标、换基线。** H2/H3 的对照改为「同 seed 的 V543 M0/P0 在 V543 evaluator 下重算的基线」，不再直接引用 V542 归档数字；新增独立的 fixed-coverage direction@k 与 top-k overlap；`q_corr` 降为诊断指标，q MAE、ordering、FPR-constrained recall、Brier/ECE 进入主表。
7. **执行规模可控。** 最多 18 次正式训练（7 seed17 + 7 seed29 + ≤4 seed43），合计 682,200 formal updates；配合 LRU Packet 缓存，单次正式 run 估计 43–67 min，18 次约 13–20 h GPU；加 smoke/diagnostic/profile/缓存构建，总计约 18–26 h。工程日历建议 6–7 周（含 1 周缓冲）。
8. **本文件不是执行结果。** 所有「将/应/需签认」均为拟议；未修改 V542 冻结文件、未启动训练、未调用 QC、未读取 dev/confirm。

### 0.2 与用户提案 v2.0 的总体关系

用户提案的五个基本原则（Encoder 只跑一次、两 Track 独立训练、所有 Decoder 残差预测、先确定性 Benchmark 再条件生成、不改科学接口）**全部保留**。本方案在其上做三类增强：

- **可冻结**：把 `DecoderContextV1` 从字段清单升级为带单位、对称性、构造算法、禁读清单、审计与缓存身份的完整数据合同。
- **可归因**：把「≤40K」升级为「同 Track 参数匹配 ±5%」，把「固定覆盖率方向」升级为独立定义，把「q 很高」升级为「q MAE + ordering + 有效点数」。
- **可执行**：把 WP0–WP9 展开为带前置、放行证据、失败动作、时间估算和运行矩阵的工作包，并给出可直接落地的 campaign YAML 草案。

---

## 1. 事实基线（只读核验，2026-11-02）

### 1.1 仓库与冻结状态

| 项 | 实测值 | 说明 |
|---|---|---|
| 当前 HEAD | `7ceffe3` | `docs: V5.4 remaining-architecture work plan + V5.3-MP training visualisation report` |
| V5.4.2 结论 | `no_backbone_winner`；A0 留任 | 三个 seed 完整且逐 seed H2 无 breach 的候选集为空；**不得写成「A0 胜出」** |
| V5.4.2 报告 | `docs/reports/PES2TS_V542_Backbone_50K_实施与训练报告_20261031.md` | Executed-Frozen；§9 给 V5.4.3 的交接 |
| 当前任务合同 | MoveProcessContract v1 | 输入仅当前单帧 + 合法有向拓扑；TS 必须可穿越；No-Move 只表示目标端点完成 |
| 数据 | `/pes2ts_data/v53mp_50k_data`（WSL ext4，约 128 GB） | 50,000 pool / 589 dev / 2,000 confirm（仅列名，未构建） |
| 训练环境 | WSL2 Ubuntu；RTX 4060 Ti；`pes2ts-gpu` env；torch 2.14.1+cu130；fp32 | Windows 侧无 torch，所有训练/评估必须在 WSL 内执行 |

### 1.2 A0×3 冻结 checkpoint（V543 的唯一 Encoder 来源）

| Run | 路径 | `checkpoint_best.pt` SHA-256 | best update | V542 inner-val total loss |
|---|---|---|---:|---:|
| `V542_A0_S17` | `outputs/pesn/v542_backbone_50k/runs/V542_A0_S17/checkpoint_best.pt` | `cab512b0094788093a9969484db7f9f303802d2fff0d7380bffb8923fc7f89bb` | 37,400 | 0.680595 |
| `V542_A0_S29` | `outputs/pesn/v542_backbone_50k/runs/V542_A0_S29/checkpoint_best.pt` | `c52db93253777a00637ea6cb41eb1366eeeaf37464d38a3d42c04692f496466b` | 37,500 | 0.666618 |
| `V542_A0_S43` | `outputs/pesn/v542_backbone_50k/runs/V542_A0_S43/checkpoint_best.pt` | `dd37c7e997d74c6426db9052918a2400687ce827600fc3dedd69efd01b6be8d4` | 36,300 | 0.684105 |

**加载事实（已核验）**：checkpoint 的 `model` state dict 有 **45 个键** = 37 个 backbone 键 + 2 个 `move_decoder.w_v`（与 `backbone.w_v` 同一 tensor 的 dual-write）+ 6 个 `process_head` 键。`pes2ts_core/pesn/models/v542/loader.py::load_a0_strict` 已实现显式映射 + `strict=True` + 逐键 `torch.equal`，V543 必须复用，不得另写 `strict=False` 加载器。

### 1.3 参数量与接口不变量

| 组件 | 参数量 | 备注 |
|---|---:|---|
| `StarEGNNV3` 全量 | 55,441 | hidden 48 / 3 层；`z[G,64]`、`edge_msg[E,48]` |
| A0 active（扣除共享 `w_v`） | 55,392 | V542 迁移不变量 |
| `DirectMoveDecoder` | 49 | `w_v = Linear(48,1)`；`a_ref = 0.08 Å` 只读 |
| `ReactionProcessHeadV53MP` | 2,178 | `ProgressHeadV3` 2,113 + `EndpointHeadV53MP` 65；只读 `z` |
| V542 joint | 57,619 | 55,392 + 49 + 2,178 |

**关键事实**：

- `ProgressHeadV3.forward` 返回 `tanh(fc2(SiLU(fc1(z))))`，**没有** `q0_raw` 的公开出口；V543 必须用受控 hook 或显式重放 `fc2` 取得 tanh 前标量，并验证 `tanh(q0_raw) == q_V542`。
- `EndpointHeadV53MP.forward` 返回 `fc(z).squeeze(-1)`，即 `endpoint_logit` 本身，无内部 sigmoid；`e0` 可直接取得并逐元素验证。
- `A0LegacyBackbone` 通过 `message_mlp[-1]` 的 forward hook 捕获 `edge_msg`，并断言 hook 恰好触发一次；V543 可复用该 adapter，再增加 `norm_h[-1]` 的 hook 捕获最后一层 `h`，但**不得修改 V542 文件**。
- `StarEGNNV3` 返回的 legacy `u` 在 V542 adapter 中被丢弃；V543 的 `u0` 必须由 `DirectMoveDecoder` 重新计算，以保证与 V542 完全一致。

### 1.4 数据、split 与采样

| 项 | 实测值 |
|---|---|
| 冻结 pool | 50,000 条曲线（2k ⊂ 5k ⊂ 20k ⊂ 50k 嵌套前缀） |
| dev | 589 条曲线；V543 仅在最终集成后一次性访问 |
| confirm | 2,000 条；仅列名封存，**永不在本阶段读取** |
| 样本总量 | 14,685,162 个 sample（idx_v2） |
| 数据集状态 | `LABEL_STATUSES = ("ok", "masked", "terminal", "conflict")` |
| MOVE 教师可用性 | 仅 `label_status == "ok"`；terminal/masked/conflict 的 `y_teacher` 不可用 |
| PROCESS 端点标签 | 必须加载全部状态；`key_atom_endpoint_labels_v1` 明确拒绝「未加载 terminal」的数据集 |
| 采样器 | `v541_curve_stratified_v1`；`spec_sha256 = 0769817e…b65d9e`；8 curves × 4 frames |
| inner split | 每 seed 80/20 curve-level inner-train/inner-val；inner-val cap 8,192 |

**关键事实**：`pair_index`、`pair_edit`、`y_teacher`、`q_true`、`endpoint y`、`frame_weight` 均在 `SUPERVISION_KEYS_V3` 中，**不是** forward-view 输入。`pair_index` 是「两端都在 K 的 changed pair」的监督索引，`pair_edit = B_tgt - B_src`；它不能直接作为 Bridge 的 `intent_pair_index`。

### 1.5 训练与评估事实

| 项 | V542 实测 |
|---|---|
| 剂量 | 400 W0 + 37,500 W1 = 37,900 updates |
| batch | 8 curves × 4 frames = 32 samples/update |
| optimizer | AdamW；core 1e-4 / head 3e-4；wd 1e-4；clip 1.0；fp32 |
| validation | 每 100 updates；checkpoint 只按 inner-val total loss |
| V542 A0 S17 wall | 5,557 s；core 1,317 s；data 2,017 s；eval 1,165 s；checkpoint 490 s |
| V542 A0 S29/S43 wall | 约 5,300–5,500 s（同量级） |
| 评估器 | `evaluate_v53mp`（frozen）；`tau_end=None` |
| 指标叶子 | `move.key_rmse_over_zero.median_rmse_over_zero`、`move.sre_and_sign.sign_accuracy_cluster_median`、`move.sre_and_sign.dead_zone_fraction`、`endpoint_head.auprc`、`q_head.corr_curve_median` |

**关键事实**：V542 的 `direction` 与 `fixed_coverage_direction` 是同一个叶子。`scripts/agent_ops/v542_stats.py` 第 180–190 行明确写道：「the frozen V5.3-MP evaluator has no separate `fixed_coverage_direction` leaf — the frozen evaluator's direction metric IS `sre_and_sign.sign_accuracy_cluster_median`」。因此 V543 必须新增独立指标，不能把旧指标改名后继续当两个指标用。

### 1.6 V542 A0 基线指标（仅作历史锚点，不作 V543 H2/H3 基线）

| Run | RMSE/zero | direction | dead | AUPRC | q_corr |
|---|---:|---:|---:|---:|---:|
| A0_S17 | 0.9099 | 0.8000 | 0.1079 | 0.2868 | 0.9988 |
| A0_S29 | 0.9008 | 0.8000 | 0.1015 | 0.3068 | 0.9999 |
| A0_S43 | 0.8979 | 0.8333 | 0.0889 | 0.2965 | 0.9940 |

**注意**：这些数字来自 V542 的联合训练模型与 V542 evaluator，不能直接作为 V543 的 H2 分母。V543 的 M0/P0 是「同一份 A0 权重 + 零残差 + V543 evaluator」的匹配基线，必须在 S0 重新计算并冻结。

### 1.7 计算与磁盘

| 项 | 实测/估计 |
|---|---|
| E: 可用空间 | 约 177 GB |
| WSL ext4 可用空间 | 约 641 GB |
| 全量 Packet 缓存（若 14.7M samples × ≈4 KB） | 约 55–65 GB；不建议全量预构建 |
| 建议缓存上限 | 30 GB 磁盘 + 1–2 GB RAM LRU；按曲线惰性构建 |
| 单次正式 run 估计（有缓存） | 43–55 min |
| 单次正式 run 估计（无缓存、live encoder） | 60–75 min |
| 18 次正式训练估计 | 13–20 h GPU；加 smoke/diagnostic/profile 约 18–26 h |

---

## 2. 对用户提案的核验结论：8 处必须修正

| # | 提案写法 | 仓库事实 | 风险 | V543 处理 |
|---|---|---|---|---|
| F1 | 「`q0_raw` 可通过 ProgressHeadV3 最后一层 tanh 前输出取得」 | `ProgressHeadV3.forward` 只返回 tanh 后的 `q`；无 raw 出口 | 直接调用 `fc1/act/fc2` 会重复实现，若与 head 前向不一致则 parity 失败 | 在 `CoarseEncoder` 中对 `progress_head.fc2` 装一次性 hook 捕获 raw，和 `process_head(z)["q"]` 逐元素验证 `tanh(raw)==q`；不修改 legacy 文件 |
| F2 | 「`intent_pair_index` 必须由合法 forward 侧有向反应图独立构造」 | forward 侧只有 `b_src/b_tgt/delta_b_intent/abs_delta/is_changed`，**没有**原子级箭头；`pair_index/pair_edit` 是监督键 | 若直接复用 `pair_index` 即监督泄漏；若声称「化学有向」则当前合同下不可识别 | 由 forward edges 重构 changed K-pair 集合；方向采用**规范序**（key-slot 小→大），符号由 `ΔB` 携带；审计时只读比对监督 `pair_index` 的集合/计数，**运行时不读**；在文档中明确「规范有向，不是化学箭头」 |
| F3 | 「新增独立 fixed-coverage direction，但保留旧指标用于历史对照」 | V542 的 `direction` 与 `fixed_coverage_direction` 是同一叶子 | 继续用旧指标会得到「两个指标同涨同跌」的假独立 | 新定义 `fixed_coverage_direction@3`（教师 top-3 abs(Δd) 对上的符号准确率）与 `topk_overlap@3`；旧 `direction` 保留为历史对照；H2 用新指标且基线用 V543 M0 重算 |
| F4 | MOVE 损失写成 `L_atom + 0.5L_pair + λ_cL_cos + λ_sL_strong + λ_dL_dead` | V542 实际 `_loss_on_batch` 只传 `lambda_cos=0.02`；`lambda_strong/lambda_dead` 未传，默认 0 | 若 V543 重新加 strong/dead，等于改损失，不能与 V542 基线比较 | **原样保留 V542 MOVE 损失**：`L_atom + 0.5L_pair + 0.02L_cos`；`λ_s/λ_d` 只作为未来单独预注册实验，不进入 V543 主矩阵 |
| F5 | 「PROCESS 训练只使用 PROCESS 有效标签」 | `_frame_mask_for` 只保留 `label_status=="ok"`；terminal 行会被 MOVE mask 剔除 | 直接复用会把 terminal 端点行排除，违背「terminal 只影响 MOVE 教师可用性」 | PROCESS Track 使用独立 `process_frame_mask`：endpoint 标签表覆盖 `ok/terminal/masked/conflict`；q 仍用 `q_mask`；terminal 行只要 endpoint 标签有效就进入训练 |
| F6 | 「PROCESS q correlation 很高即可晋级」 | `q_head.corr_curve_median` 是每曲线 Pearson 的中位数；每曲线可能只有 2–4 个有效点，2 点 Pearson 恒为 ±1；A0 的 q_corr 为 0.9988，而同一任务 V5.3-MP 50k 的 q MAE 仍约 0.50——相关高不等于误差小 | q_corr 虚高，不能作为主判据 | q MAE（曲线中位数）+ ordering 作为主指标；q_corr 只作诊断，且报告 `n_eval_curves`、每曲线有效点数中位数、≥3 点曲线占比；若 ≥3 点曲线 <70%，q_corr 不进入结论 |
| F7 | `DecoderContextV1` 字段只有 `key_v`、`pair_s`、`intent_pair_index` | M3 的 key-key 消息传递、P3/P4 的 token 图需要邻接与 token 布局；仅靠 pair 索引不足以构成连通图 | M3/P3/P4 无法公平实现，或各自私建邻接导致不可比 | 在 v1 中增加 `key_edge_index[2,Ek]`、`key_edge_attr[Ek,8]`、`token_index[T]`、`token_type[T]`、`token_graph[T]`、`token_ptr[G+1]`；所有 Decoder 共用同一 token 定义 |
| F8 | 「8 通道基底覆盖率必须计算」 | 8 个任意 3D 向量在 R3 中几乎必然张成满秩；单独报告「覆盖率≈100%」会误导 | 误判 Bridge 足够，跳过 Adapter | 覆盖率必须与**条件数**、**有效秩**、**教师系数幅度**、**逐通道留一**一起报告；覆盖率只是必要条件，不是充分条件；真正的 go/no-go 由 S0-B 潜空间 probe 决定 |

> 以上 8 条不是对提案的否定，而是把「设计候选」变成「可冻结合同」的必要修订。修订后的字段与流程见 §5–§7。

---

## 3. 科学问题、假设与否证条件

### 3.1 中心问题

> **同一份冻结的分子空间表征（A0 的 `z`、最后一层 `h`、`edge_msg`、`u0`、`q0/e0`），采用何种 Decoder 结构，能在不损害化学语义与推理成本的前提下，最准确地产生下一步关键原子位移和当前反应状态？**

V5.4.3 不回答「哪个 Backbone 更好」；A0 是固定 Encoder，不参加冠军竞争。V5.4.3 也不回答「Flow/Diffusion 是否流行」；生成模型只有在确定性 Decoder 暴露明确表达瓶颈后才获得预算。

### 3.2 五个可证伪假设

| ID | 假设 | 主要检验 | 否证条件 |
|---|---|---|---|
| H-M1 | 在冻结表征上，简单残差 MLP 已能显著超过 M0 | M1 vs M0：RMSE/zero 或 fixed-coverage direction | 三 seed 均无 H3 效应，或 H2 任一项 breach |
| H-M2 | 标量—向量交互（GVP）比纯标量 MLP 更能修正方向 | M2 vs M1：direction/top-k overlap；控制参数量 | M2 不优于 M1，且方向指标无 ≥3% 相对增益 |
| H-M3 | 多个关键原子的协同动作需要 key-key 消息传递 | M3 vs M1/M2：多关键原子帧的 RMSE/方向 | M3 仅在单关键原子帧占优，或整体不优于 M1/M2 |
| H-P1 | 局部 key/pair token 比仅全局 `z` 更能判断端点与进度 | P2/P3/P4 vs P1：AUPRC、FPR-constrained recall、q MAE | 局部 token 无增益，或仅容量增益（P4-C 同样好） |
| H-P2 | 全局关系建模（Graph Transformer）优于局部 GNN | P4 vs P3：AUPRC/q MAE；P4-C 参数匹配对照 | P4 不优于 P3，或 P4-C 追平 P4（说明是容量而非注意力） |
| H-I1 | 单步增益能转化为闭环路径收益 | B0/BM/BP/BMP rollout | 单步改善但 rollout 无改善；此时只能声明单步 Decoder 改进 |

### 3.3 成功、负结果与「不可宣称」

- **成功**：每条 Track 至少一个候选通过 H2（逐 seed 非劣）与 H3（效应量 + 统计显著），且通过成本门；两条 Track 各自产生冠军；集成门通过。
- **科学负结果**：无候选通过 H3，或只有单 seed 证据，或收益被成本抵消。此时保留 M0/P0，按冻结措辞记录 `no_decoder_winner`，不得用「趋势好」「接近显著」替代。
- **不可宣称**：A0 最优、Flow/Diffusion 必然优于回归、单步指标提升等于路径导航成功、q 高等于 PROCESS 好、无物理松弛 rollout 的失败全部归因于 Decoder。

---

## 4. 架构冻结：五组件与唯一数据流

### 4.1 五个独立组件

| 组件 | 职责 | 是否读原始 XYZ | 训练状态 | 参数量 |
|---|---|---|---|---|
| `CoarseEncoder` | A0 EGNN + D0 + P0；产生 `z`、`h`、`edge_msg`、`u0`、`q0_raw`、`e0` | 是（仅此一处） | **冻结** | 57,619 |
| `LatentBridge` | 从 Encoder 输出与 forward 拓扑构造 `DecoderContextV1` | 只在构造时读一次 `pos` 的差值 | **固定、参数自由** | 0 trainable |
| `MoveRefiner` | `δu` | 否 | 独立训练 | ≤36K 目标 |
| `ProcessRefiner` | `δq_raw`、`δe` | 否 | 独立训练 | ≤36K 目标 |
| `ReactionController` | 化学核验 + MOVE/No-Move | 可读当前合法输入做核验 | 非本阶段学习目标 | 0 |

### 4.2 唯一前向流程

```text
单帧 forward_view（当前 XYZ + 合法有向拓扑 + 约束）
        │
        ▼
CoarseEncoder（只调用一次）
  ├─ StarEGNNV3 → z[G,64], h[N,48], edge_msg[E,48]
  ├─ DirectMoveDecoder → u0[K,3]
  └─ ReactionProcessHeadV53MP → q0_raw[G], e0[G]
        │
        ▼
LatentBridge（参数自由）
  → DecoderContextV1（不含 pos / 不含监督）
        │
        ├───────────────┐
        ▼               ▼
MoveRefiner        ProcessRefiner
  δu[K,3]            δq_raw[G], δe[G]
        │               │
        ▼               ▼
u = u0 + δu        q = tanh(q0_raw + δq)
                   e = e0 + δe; p_end = σ(e)
        │               │
        └───────┬───────┘
                ▼
ReactionController（外层，不在训练前向内）
  → move_flag / selected_coords（仅依据校准端点 + 化学状态）
```

### 4.3 输出语义（冻结）

- `u = u0 + δu`，单位 Å；行序严格等于 `key_index`；非关键原子零位移。
- `q = tanh(q0_raw + δq)`；`order_score IS q`（保持 V542 外部合同）。
- `e = e0 + δe`；`p_end = σ(e)`；训练时只用 logit，禁止双重 sigmoid。
- 外部 forward 仍只返回五键 `{u, z, q, endpoint_logit, order_score}`；诊断量（`u0`、`δu`、`q0_raw`、`δq`）通过独立 `forward_debug` 返回，不进入外部合同。
- **单 Track Benchmark 时，未参与当前 Track 的 Refiner 固定为零残差（M0/P0）**，避免另一 Track 的随机初始化污染当前指标；只有 T7 集成才同时启用两个冠军。
- `No-Move` 只能由「校准后的端点概率 + 目标化学状态一致」触发；`q` 阈值、接近 TS、路径长度均不得触发 No-Move；TS 行必须继续 MOVE。

### 4.4 冻结边界

1. **Encoder 只运行一次**：每次当前构型恰好一次 `CoarseEncoder.forward`；Flow/Diffusion 的 NFE 循环不得重新调用 Encoder。
2. **两条 Track 独立**：MOVE 训练不更新 PROCESS Refiner，PROCESS 训练不更新 MOVE Refiner；两者都不更新 Encoder/D0/P0。
3. **所有 Refiner 残差预测**：最后一层零初始化；初始模型必须逐元素还原 V542 A0。
4. **先确定性、后生成**：M4/M5 只在 S1 诊断显示明确剩余瓶颈时激活；不得因「模型流行」直接进入正式矩阵。
5. **科学接口不变**：单帧输入、合法有向拓扑、TS 可穿越、No-Move 语义不变。

### 4.5 零残差初始化

| 模块 | 零初始化位置 | 初始输出 |
|---|---|---|
| M1 | 最后一层 `Linear(·,8)` 的 weight/bias | 8 个系数 = 0 → `δu = 0` |
| M2 | 最后一个 scalar→vector 的输出投影 + 外层 gate | `δu = 0` |
| M3 | 输出 head 最后一层 | `δu = 0` |
| M4 | 速度输出最后一层 + 外层零 gate | `δu = 0`（不依赖噪声） |
| M5 | 去噪输出最后一层 + 外层零 gate | `δu = 0` |
| P1–P5 | 最后一层 `Linear(·,2)` 的 weight/bias | `δq = 0`、`δe = 0` |

**注意**：仅把 Flow 内部参数初始化为零，不能保证随机采样输出为零；必须在最外层加零初始化残差门控。测试必须验证「随便采样多少次，初始 `δu` 恒为 0」。

### 4.6 Controller 语义（非学习）

```python
def move_flag(p_end, tau_end, chemical_state_ok, is_ts):
    if is_ts:
        return True                    # TS 永远继续 MOVE
    if p_end >= tau_end and chemical_state_ok:
        return False                   # 唯一合法的 No-Move
    return True                        # q 不参与
```

- `tau_end` 只在 inner-val 上校准一次，目标 precision 0.9；dev 不重校准。
- `chemical_state_ok` 由外层合法端点/键级核验给出，不由模型 `q` 给出。
- 该函数不在 `V543Model.forward` 内；训练前向不产生 `move_flag`。

---

## 5. `DecoderContextV1`：V5.4.3 的真正交付物

### 5.1 设计原则

1. **只读、类型化、不可变**：`@dataclass(frozen=True, slots=True)`；Decoder 只能拿到 context，拿不到 `batch`、`pos` 或任何监督键。
2. **单位与对称性显式**：每个字段在表里写清 dtype、shape、单位、旋转/平移/置换行为。
3. **参数自由**：Bridge 不引入可训练参数；固定投影矩阵以 `persistent=False` buffer 存放，不进入 optimizer。
4. **构造一次**：每个当前构型只构造一次；Flow/Diffusion 的 NFE 循环复用同一 context。
5. **可缓存**：缓存键覆盖 Encoder/Bridge/输入/精度/代码身份；缓存内容只有 context，没有标签。
6. **可审计**：任何字段都能追溯到 forward-view 键或 Encoder 输出；禁止字段在运行时不可达。

### 5.2 字段规格（v1 拟议冻结）

| 字段 | Shape | dtype | 单位 | 旋转 | 平移 | 置换 | 来源 |
|---|---|---|---|---|---|---|---|
| `graph_s` | `[G,64]` | fp32 | 无量纲 | 不变 | 不变 | 图置换等变 | `z`（A0 `z_head` 输出，tanh 饱和） |
| `key_s` | `[K,48]` | fp32 | 无量纲 | 不变 | 不变 | 原子置换等变 | 最后一层 `h[key_index]` |
| `key_v` | `[K,8,3]` | fp32 | 无量纲 | **极向量** | 不变 | 原子置换等变 | Bridge 构造（见 §5.3） |
| `u0` | `[K,3]` | fp32 | Å | **极向量** | 不变 | 原子置换等变 | `DirectMoveDecoder` |
| `key_graph` | `[K]` | int64 | — | — | — | 随原子置换重编号 | `graph_index[key_index]` |
| `key_ptr` | `[G+1]` | int64 | — | — | — | 图置换等变 | 每图 key 数前缀和 |
| `key_edge_index` | `[2,Ek]` | int64 | key-local 行号 | — | — | 原子置换等变 | Bridge 构造（见 §5.5） |
| `key_edge_attr` | `[Ek,8]` | fp32 | 混合（距离 Å + 无量纲） | 不变 | 不变 | 边置换等变 | Bridge 构造 |
| `pair_s` | `[P,48]` | fp32 | 无量纲 | 不变 | 不变 | 对置换等变 | Bridge 构造（见 §5.4） |
| `intent_pair_index` | `[2,P]` | int64 | key-local 行号 | — | — | 对置换等变 | Bridge 构造；规范序，非化学箭头 |
| `pair_graph` | `[P]` | int64 | — | — | — | 图置换等变 | 每对所属图 |
| `pair_ptr` | `[G+1]` | int64 | — | — | — | 图置换等变 | 每图 pair 数前缀和 |
| `token_index` | `[T]` | int64 | key/pair 行号 | — | — | token 置换等变 | 布局：每图 `[graph, keys…, pairs…]` |
| `token_type` | `[T]` | int8 | 0=graph,1=key,2=pair | — | — | token 置换等变 | 布局 |
| `token_graph` | `[T]` | int64 | — | — | — | 图置换等变 | 布局 |
| `token_ptr` | `[G+1]` | int64 | — | — | — | 图置换等变 | 每图 token 数前缀和 |
| `q0_raw` | `[G]` | fp32 | tanh 前标量 | 不变 | 不变 | 图置换等变 | `progress_head.fc2` 输出（hook） |
| `endpoint0_logit` | `[G]` | fp32 | logit | 不变 | 不变 | 图置换等变 | `endpoint_head.fc(z)` |
| `schema` | str | — | — | — | — | — | `"decoder_context_v1"` |
| `bridge_spec_sha256` | str | — | — | — | — | — | Bridge 规格 canonical SHA |
| `encoder_ckpt_sha256` | str | — | — | — | — | — | A0 checkpoint SHA |
| `precision_version` | str | — | — | — | — | — | `"fp32_v1"` |

**形状约束**：

- `K = key_ptr[-1]`；`P = pair_ptr[-1]`；`T = token_ptr[-1] = G + K + P`。
- `key_ptr`、`pair_ptr`、`token_ptr` 单调不减，首项 0，末项等于对应总数。
- 每个图至少 1 个 key；允许 `P=0`（无 changed K-pair），此时 `pair_s` 为 `[0,48]`，Decoder 必须能处理空 pair。
- `Ek=0` 允许；M3 必须能退化为无消息传递。

#### 5.2.1 两个只读视图（强制 Track 独立）

`DecoderContextV1` 只在 `V543Model` 内部持有；两个 Refiner 拿到的是互不重叠的只读视图：

```python
@dataclass(frozen=True, slots=True)
class MoveContextV1:
    graph_s: Tensor          # [G,64]
    key_s: Tensor            # [K,48]
    key_v: Tensor            # [K,8,3]
    u0: Tensor               # [K,3] Å
    key_edge_index: Tensor   # [2,Ek]
    key_edge_attr: Tensor    # [Ek,8]
    key_graph: Tensor        # [K]
    key_ptr: Tensor          # [G+1]

@dataclass(frozen=True, slots=True)
class ProcessContextV1:
    graph_s: Tensor          # [G,64]
    key_s: Tensor            # [K,48]
    pair_s: Tensor           # [P,48]
    intent_pair_index: Tensor  # [2,P]
    key_graph: Tensor        # [K]
    pair_graph: Tensor       # [P]
    key_ptr: Tensor          # [G+1]
    pair_ptr: Tensor         # [G+1]
    token_index: Tensor      # [T]
    token_type: Tensor       # [T]
    token_graph: Tensor      # [T]
    token_ptr: Tensor        # [G+1]
    q0_raw: Tensor           # [G]
    endpoint0_logit: Tensor  # [G]
```

**硬约束**：

- `MoveContextV1` **不含** `q0_raw`、`endpoint0_logit`、`pair_s`、`token_*`；MOVE 不能读 PROCESS 状态。
- `ProcessContextV1` **不含** `u0`、`key_v`、`key_edge_*`；PROCESS 不能读 MOVE 初猜或 MOVE 的向量基底。
- 两个视图都不含 `pos`、监督键、绝对坐标、历史帧。
- `V543Model.forward` 只把对应视图传给对应 Refiner；测试 `test_v543_track_view_isolation` 必须断言 `MoveContextV1` 没有 `q0_raw`/`u0` 以外的 PROCESS 字段，`ProcessContextV1` 没有 `u0`/`key_v`。

**为什么强制互斥**：如果 MOVE 能读 `q0_raw`，它可能学会「用 PROCESS 的进度作弊」；如果 PROCESS 能读 `u0`，它可能从 MOVE 的位移反推进度。V543 要回答的是「两条独立专家各自能从同一份 Encoder 表征中学到什么」，不是「两个头互相泄题能到多高」。

### 5.3 `key_v` 的 8 个通道（参数自由、确定性）

记 `x = pos`，`a_ref = 0.08`，`unit_e = (x[src_e] - x[dst_e]) / (||…|| + 0.1)`（与 D0 完全相同的单位向量约定）。对每个 key 原子 `k`：

| 通道 | 定义 | 说明 |
|---|---|---|
| 0 | `u0[k] / a_ref` | 继承 MOVE 初猜；除以 `a_ref` 使其无量纲 |
| 1 | `Σ_{e: dst=k, ΔB_e<0} abs(ΔB_e) · unit_e / Σabs(ΔB_e)` | 断键方向 |
| 2 | `Σ_{e: dst=k, ΔB_e>0} abs(ΔB_e) · unit_e / Σabs(ΔB_e)` | 成键方向 |
| 3 | `Σ_{e: dst=k, ΔB_e=0, covalent} unit_e / count` | 不变共价边方向 |
| 4 | `Σ_{e: dst=k, spatial} κ(d_e) · unit_e / Σκ`，`κ(d)=exp(-((d-3.0)/1.0)^2)` | 空间邻近方向 |
| 5 | `Σ_{p: k∈pair, changed K-pair} sign(ΔB_p) · (x[other]-x[k])/(‖…‖+0.1) / count` | 关键对方向 |
| 6 | `Σ_{e: dst=k} (d_e/4.0) · unit_e / Σ(d_e/4.0)` | 距离加权全边均值 |
| 7 | `(mean_{j∈K, j≠k} x[j] - x[k]) / (‖…‖+0.1)` | 指向关键原子质心的方向 |

**规则**：

- 任一分母为 0 时该通道置零向量；不得产生 NaN。
- 所有通道只依赖 `pos` 的差值、`ΔB`、`is_changed`、`is_covalent`、`is_spatial`、`key_index`；不依赖绝对坐标。
- 通道 0 与 `u0` 线性相关；在 S0-C 中必须同时报告「含通道 0」与「去掉通道 0」的覆盖率，避免用 `u0` 自证。
- 8 个通道的目标不是「一定表达真实动作」，而是给 Decoder 一个可解释、等变、低成本的向量读出基底；是否足够由 S0-B/S0-C 判定。

### 5.4 `pair_s` 的 48 维布局（参数自由）

对每个 changed K-pair `(i,j)`（`i<j` 为 key-slot 规范序）：

| 维度 | 内容 |
|---|---|
| 0–5 | `b_src`, `b_tgt`, `ΔB`, `abs(ΔB)`, `is_changed`, `is_covalent` |
| 6–7 | `is_aromatic`, `is_spatial` |
| 8–19 | `RBF12(d_ij)`（复用 `gaussian_rbf_v3` 的 12 个中心） |
| 20–22 | `d_ij`, `1/(d_ij+1)`, `ΔZ/10` |
| 23 | `abs(ΔZ)/10` |
| 24–27 | 端点 i 的 `(breaking, forming, spectator, is_key)` |
| 28–31 | 端点 j 的同一组角色 |
| 32–39 | 图级广播：`signed_intent`, `‖cv_delta‖`, `cv_delta[0:6]` |
| 40–47 | `W_fixed @ (key_s[i] - key_s[j])`；`W_fixed` 为固定的 `8×48` ±1/√48 矩阵（`persistent=False`，无训练参数） |

**说明**：40–47 是固定的 Johnson–Lindenstrauss 式 sketch，只提供低维标量上下文；它不是可训练层，也不改变 `key_s` 的语义。若 WP2 审计认为它引入不必要的耦合，可替换为 8 个固定索引的 `key_s` 差值；两种实现都必须参数自由，并在 Bridge 规格中冻结。

### 5.5 key-key 邻接与 token 布局

**key 图边集**（去重、无向、双向存储）：

1. 每个 changed K-pair 加一条双向边；
2. 每个 key 原子到最近的最多 4 个 key 原子（距离 < 6.0 Å）加边，排除自身与已存在边；
3. 不做稠密 `K²` 全连接；`Ek ≤ 2·(P + 4K)`。

**`key_edge_attr` 8 维**：`d`、`RBF(d;2,3,4,5)`（4 维）、`|ΔB|`、`is_changed`、`edge_type`（0=covalent/changed，1=spatial）。

**token 布局**：每图 token 连续排列为 `[graph_token, key_tokens…, pair_tokens…]`；`token_index` 对 graph token 填 `-1`，对 key/pair 填对应行号；`token_type` 区分三类。这样 P2/P3/P4 可以共享同一套 token 定义，不需要各自私建索引。

### 5.6 禁止输入与运行时审计

**Bridge 不得读取**：

- `pos` 以外的任何几何量（且 `pos` 只用于差值）；
- `pair_index`、`pair_edit`、`y_teacher`、`q_true`、`endpoint` 标签、`frame_weight`、`d0`；
- 参考 TS/端点 XYZ、未来帧、IRC 历史、能量/梯度；
- `split_family`、`branch_id`、`curve_id`、`sample_id`；
- 任何 `SUPERVISION_KEYS_V3` 成员。

**运行时审计**：

```python
def assert_no_supervision(ctx: DecoderContextV1) -> None:
    for name in SUPERVISION_KEYS_V3:
        assert not hasattr(ctx, name)
    for forbidden in ("pos", "xyz", "pair_edit", "y_teacher", "q_true"):
        assert not hasattr(ctx, forbidden)
```

- `DecoderContextV1` 的 `__slots__` 只含 §5.2 字段；`__getattr__` 不得回退到 batch。
- 测试 `test_v543_decoder_no_xyz`：给 Decoder 传一个「只有 context」的假对象，任何 `ctx.pos` 访问必须 `AttributeError`。
- 测试 `test_v543_no_supervision_leak`：在 batch 中注入哨兵值（如 `pair_edit=12345`、`y_teacher=999`），构造 context 后所有字段与哨兵无关。

**pair 集合来源审计（只读、离线）**：

- 在 10,000 个采样帧上，比较 Bridge 从 forward edges 重构的 pair 集合与监督 `pair_index` 的集合/计数/端点。
- 期望：集合完全一致（因为二者都来自同一 intent 的 changed pairs）；若不一致，说明 forward 视图缺信息或重构规则错，必须阻断。
- 审计代码只读监督字段用于**比对**，不进入 context、不进入模型；审计报告单独存放，SHA 进 run identity。

### 5.7 缓存记录与 Packet 的边界

**缓存内容**：只有 `DecoderContextV1` 的 tensor 字段 + schema/identity；**不含** `y_teacher`、`q_true`、`endpoint`、`frame_weight`、`sample_id`、`curve_id`。

**缓存键**：

```text
encoder_ckpt_sha256
| bridge_spec_sha256
| input_feature_sha256
| source_frame_identity   # curve_id + node_index + intent_id + d0 的 canonical hash
| precision_version
| code_fingerprint
```

**规则**：

- 缓存只在 `frozen_encoder` 模式有效；`joint_finetune` 必须实时构建、保留 autograd，禁止读 detached 缓存。
- 缓存构建必须在 split 之后；只允许 inner-train + inner-val 曲线；dev/confirm 永不入缓存。
- 缓存按曲线分片（建议 256 curves/片），按需加载，RAM LRU 1–2 GB；磁盘上限 30 GB；命中率目标 ≥80%。
- 身份不匹配、文件 SHA 不匹配、schema 版本不匹配 → 拒绝命中并重建；不得静默复用。
- 缓存统计（hit/miss/build time/disk bytes）写入 run records；缓存本身不参与科学选择。

### 5.8 A0 迁移与零残差 parity

**必须证明的等式**（三 seed、≥8 个真实 inner-val batch、CPU 逐位、GPU ≤1e-6）：

```text
u0_V543            == u_V542
z_V543             == z_V542
q0_V543            == q_V542
tanh(q0_raw_V543)  == q0_V543
endpoint0_V543     == endpoint_logit_V542
零残差 V543(u,q,e) == V542(u,q,e)
order_score is q
```

**实现要点**：

- `CoarseEncoder` 复用 `A0LegacyBackbone`（含 `message_mlp[-1]` hook、共享 `w_v`、strict load），不修改 V542 文件。
- 增加 `norm_h[-1]` 的 forward hook 捕获最后一层 `h`；hook 必须在 `finally` 中移除，并断言恰好触发一次。
- `q0_raw` 用 `progress_head.fc2` 的 forward hook 捕获；`q` 仍由 `process_head(z)` 产生；两者逐元素验证。
- `u0` 由 `DirectMoveDecoder` 用与 V542 完全相同的五个张量计算；不得重排 `edge_index` 或改变 `key_index` 行序。
- 所有 hook 不得 detach；frozen 模式下可以在 `torch.no_grad()` 中运行，但 zero-residual parity 必须在同一 dtype/device 下比较。

---

## 6. Decoder 候选、注册表与参数预算

### 6.1 注册表协议

```python
class MoveDecoderProtocol(Protocol):
    def forward(self, ctx: MoveContextV1) -> torch.Tensor:
        """Return δu [K,3] in Å; zero rows for non-key atoms are impossible
        because K rows are key-only."""
        ...

class ProcessDecoderProtocol(Protocol):
    def forward(self, ctx: ProcessContextV1) -> ProcessResidual:
        """Return δq_raw [G] and δe [G]."""
        ...
```

**Registry 每条记录至少包含**：

| 字段 | 说明 |
|---|---|
| `decoder_id` | `M0`…`M5` / `P0`…`P5` |
| `track` | `move` / `process` |
| `model_version` | 语义版本 |
| `contract_version` | `decoder_context_v1` |
| `n_trainable_params` | 构造后精确 `sum(p.numel())` |
| `equivariance` | `invariant` / `polar` / `stochastic_polar` |
| `stochastic` | bool |
| `nfe` | 推理调用次数；确定性模型为 1 |
| `zero_init` | 最后一层/门控是否零初始化 |
| `status` | `proposed/smoke/diagnostic/formal/champion/rejected` |
| `checkpoint_sha256` | 训练后回填 |
| `config_sha256` | 训练配置 SHA |
| `acceptance` | 契约/性能/部署验收结果 |

**禁止**：Decoder 私自实现数据读取、损失定义、checkpoint 选择；这些统一由 `v543_train.py` 负责。

### 6.2 MOVE Track 候选

| ID | 模型 | 结构要点 | 目标参数 | 等变读出 | 执行策略 |
|---|---|---|---:|---|---|
| M0 | Direct | 旧 D0，`δu=0` | 0 | `u0` 本身 | 必须（基线） |
| M1 | Residual MLP | `[graph_s,key_s,u0/a_ref,‖key_v‖]`（123 维）→ 136 → 136 → 8 个系数；`δu=Σ c_j key_v_j`（约 36.6K） | 36K ±5% | 8 通道线性组合 | 正式 |
| M2 | GVP Refiner | 2 个 GVP block；标量 96 / 向量 8 通道；标量—向量交互 | 36K ±5% | 向量通道直接更新 | 正式 |
| M3 | Vector-GNN | key 图上 2 层标量—向量消息传递；hidden 56；边特征 8 维（约 34.5K） | 36K ±5% | 向量通道消息聚合 | 正式 |
| M4 | Conditional Flow | 在残差空间 `r=y-u0` 上做条件速度场；hidden 128；Euler NFE 1/2/4/8；外层零 gate（约 35.2K） | 36K ±5% | 速度用 8 通道基底读出 | 条件 |
| M5 | Residual Diffusion | 条件动作去噪；固定噪声日程；外层零 gate | 36K ±5% | 8 通道基底读出 | 条件（默认关闭） |

**M1 的关键约束**：不能直接让 MLP 输出绝对 `(x,y,z)`；必须输出 8 个标量系数，再与 `key_v` 组合。否则旋转一个分子会得到不旋转的残差，破坏等变性。

**M2 的关键约束**：向量通道禁止 LayerNorm/ReLU；标量可用 SiLU；不使用裸 cross product（若使用，必须证明 O(3) 反射下的 parity 处理，否则只声明 SO(3)）。

**M3 的关键约束**：只在 key 图上做 1–2 层局部交互；不得重新运行原始几何 Encoder，不得对全部原子做稠密 attention。

### 6.3 PROCESS Track 候选

| ID | 模型 | 结构要点 | 目标参数 | 执行策略 |
|---|---|---|---:|---|
| P0 | 原 PROCESS | 旧 q/endpoint head，`δq=δe=0` | 0 | 必须（基线） |
| P1 | Residual MLP | `graph_s[64]` → 160 → 160 → 2（约 36.5K） | 36K ±5% | 正式 |
| P2 | DeepSets | key/pair token 编码 48→88；mean+max 池化；与 `graph_s` 拼接（240）→ 128 → 2（约 35.7K） | 36K ±5% | 正式 |
| P3 | Latent Graph-GNN | 同一 token 图上 2 层消息传递；hidden 56；边特征 8（约 34.5K） | 36K ±5% | 正式 |
| P4 | Graph Transformer | 2 层、d=48、4 heads（head_dim 12）、FFN 80；局部图 bias；token 级 attention（约 37K） | 36K ±5% | 正式 |
| P4-C | 参数匹配非 Attention 对照 | 与 P4 同宽、同 token、同参数 ±2%，MLP/池化替代 attention | 与 P4 匹配 | 诊断（仅 P4 进 top2 时） |
| P5 | GraphGRU | 2 个 GRU cell，hidden 52；同一帧内迭代 T=4；每次 forward 重置状态（约 35.7K） | 36K ±5% | 条件 |

> 表中宽度是设计起点，不是实现前的精确断言；构造时必须按 `sum(p.numel())` 调整宽度，使精确值落入 34,200–37,800，且不得超过 40,000。不得用 dead weights 或无用 bias 填充参数。

**P2/P3/P4 必须使用同一 token 定义**（§5.5）；否则「局部信息有无收益」无法归因。

**P4 的关键约束**：attention 只在 token 级（T≈G+K+P，通常 <40/图），不做原子级稠密 N²；attention logits 不得含绝对坐标；局部图 bias 用 `key_edge_index`/`token` 邻接，不用可学习全局帧。

**P5 的关键约束**：每次推理从当前 Packet 重新初始化隐藏状态，不保留前一 IRC 步的 hidden；测试 `test_v543_graph_gru_reset` 必须证明两次独立 forward 与「先 forward A 再 forward B」的 B 结果一致。

### 6.4 参数匹配与总预算

| 项 | 规则 |
|---|---|
| 每 Track 目标 | 36,000 trainable params |
| 允许带 | ±5%（34,200–37,800） |
| 单候选硬上限 | 40,000 |
| M0/P0 | 0 |
| 最终整模（目标） | 57,619 + 36K + 36K = 129,619 |
| 最终整模（带内最大） | 57,619 + 37,800 + 37,800 = 133,219 |
| 全局硬上限 | 150,000 |
| 断言 | 构造时 `sum(p.numel())` 精确断言；估算值不得作为实现前 exact 断言 |
| 禁止 | 用 dead weights / 无用 bias 凑参数；参数匹配是为了排除容量混杂，不是为了填满预算 |

**为什么收紧到 ±5%**：V542 的 A5 只有 31.6K，A0 55.4K，容量 caveat 至今未闭合。V543 若允许 P1 8K、P4 38K，则「P4 更好」无法区分注意力与容量。±5% 是低成本、可机器检查的最小公平性约束。

### 6.5 条件生成模型的激活条件

**M4 Flow 激活条件（全部满足）**：

1. M1–M3 的 S1 诊断显示残差存在**系统性方向瓶颈**：例如 M1/M2/M3 的 fixed-coverage direction 在 Pre-TS/多关键原子层显著低于 Post-TS，且 basis projection residual 不能解释；
2. 残差分布在同一 `(graph_s,key_s)` 条件下呈现多模态或重尾（用 held-out 残差的核密度/分位数诊断，不用教师动作挑样本）；
3. 成本审计允许 NFE 1/2/4/8 的额外推理；p95 延迟 ≤2× A0；
4. 新签认：M4 不进入首轮 18-run 预算，单独登记 T8。

**M5 Diffusion 激活条件**：M4 已证明分布建模有收益，且采样成本、校准、旋转一致性、NFE 都可接受；默认关闭。

**P5 GraphGRU 激活条件**：P1–P4 已完成且仍存在「同一帧内需要多步潜空间推理」的系统性失败；否则不启动。

**禁止**：用真实教师动作挑选生成样本；从多个采样结果中事后挑误差最小者作为模型成绩；把 Flow 的 NFE 隐藏在「一次 forward」里不报成本。

---

## 7. 训练方案：S0 → S1 → S2

### 7.1 Stage S0：迁移验收与冻结表征

#### S0-A 零残差 parity（阻断门）

对三个 seed 分别加载匹配的 A0 checkpoint：

- `V543_M0_S17` ← `V542_A0_S17`
- `V543_M0_S29` ← `V542_A0_S29`
- `V543_M0_S43` ← `V542_A0_S43`

必须复现：

- `u0`、`z`、`q`、`endpoint_logit` 与 V542 对应 seed 逐元素一致（CPU bitwise；GPU ≤1e-6）；
- `tanh(q0_raw) == q`；
- 零残差 M0/P0 全模型输出与 V542 一致；
- `order_score is q`；
- Encoder forward 每次当前构型恰好 1 次；hook 恰好 1 次；
- 3 seeds × ≥8 个真实 inner-val batch。

**失败处理**：阻断所有训练；按缺陷类别修复（代码/数据/初始化/hook），bump 对应 identity 层，重跑 S0-A。不得用「近似相等」放行。

#### S0-B Latent Sufficiency Probe（信息充分性）

冻结 Encoder，只训练很小的线性/MLP 诊断头：

| Probe | 输入 | 目标 | 目的 |
|---|---|---|---|
| B1 | `graph_s` only | `q`、`endpoint` | 全局标量是否足够 PROCESS |
| B2 | `graph_s + key_s + key_v` | `δu = y - u0` | Bridge 是否包含 MOVE 残差信息 |
| B3 | `key_s + key_v`（无 graph_s） | `δu` | 局部信息是否足够 |
| B4 | `graph_s + key_s`（无 key_v） | `δu` | 向量基底是否有额外贡献 |

**训练/评估纪律**：

- Probe 只在 inner-train 上训练；从 inner-train 再切一个 curve-level probe-val 选 epoch；inner-val 只在最后评估一次。
- Probe 结果只用于「是否继续/是否需要 Adapter」的诊断，不进入冠军选择。
- 报告：残差 MAE、R²、按 stratum/反应类型分层；与 M0 的 `||y-u0||` 基线比较。
- **Go 条件**：B2 的残差 MAE 明显低于 B1/B4（例如相对 B4 改善 ≥10%），说明 `key_v` 有增量信息；否则先改 Bridge，不加深 Decoder。

#### S0-C Vector Basis Coverage（基底能力）

对真实教师残差 `r = y - u0`，逐 key 原子计算：

- `rank(key_v[k])`、最小/最大奇异值、条件数；
- 投影残差 `||r_perp|| / ||r||`（含通道 0 / 去掉通道 0 两版）；
- 教师系数幅度分布；
- 留一通道后的覆盖率变化；
- 按 Start / Pre-TS / Post-TS / Endpoint 分层，按 breaking/forming/both 分层。

**解释纪律**：8 个任意 3D 向量几乎必然张成 R3，所以「覆盖率≈100%」不是好消息也不是坏消息；真正要看的是**条件数**、**系数是否可预测**、**留一通道是否敏感**。若条件数 >100 或投影残差中位数 >0.5，Bridge 需要重新设计；不得靠加深 Decoder 掩盖。

#### S0 出口

- S0-A 全过；
- S0-B 给出「Bridge 是否有信息」的结论；
- S0-C 给出「基底是否可用的」结论；
- 三者写入 `encoder_bridge_parity` 报告；任一失败则 WP2 回炉，不进入 S1。

### 7.2 Stage S1：Decoder-only 训练

#### 固定训练日程

| 项 | 值 |
|---|---|
| Encoder / Bridge / D0 / P0 | 冻结；`requires_grad=False`；frozen 模式可用 `no_grad` |
| 可训练 | 当前 Track 的 Refiner |
| updates | **37,900 flat** |
| warmup | 400 updates 线性 warmup |
| LR | 3e-4（warmup 后恒定） |
| optimizer | AdamW |
| weight decay | 1e-4 |
| grad clip | 1.0 |
| precision | fp32 |
| batch | 8 curves × 4 frames = 32 |
| sampler | `v541_curve_stratified_v1`；确定性 RNG `(seed, update, salt)` |
| validation | 每 100 updates；inner-val cap 8,192 |
| checkpoint | 只按当前 Track 的 inner-val 主损失 |
| seeds | 17 / 29 / 43 |
| 恢复 | 只从匹配 identity 的 `checkpoint_last.pt` 恢复 |
| 禁止 | `torch.compile`、减帧/减层/减 updates、换 seed、按 H3 指标选 checkpoint |

#### 单步训练数据流（每 update）

1. `curve_batch_positions_v541` 按 `(seed, update, salt)` 抽 8 curves × 4 frames，得到 32 个 position。
2. `_batch_from_positions` 从 dataset/LRU 取特征并 collate；`forward_view_v3(batch)` 去掉全部监督键。
3. `CoarseEncoder.forward(forward_view)` **恰好一次**，得到 `z/h/edge_msg/u0/q0_raw/e0`。
4. `LatentBridge.build_context(packet, forward_view)` 得到 `DecoderContextV1`；frozen 模式可走缓存，miss 时实时构建。
5. 当前 Track 的 Refiner 读 context，输出 `δu` 或 `(δq,δe)`；组装 `u/q/e`。
6. **独立标签通道**：从 `labels_v52`、`endpoint_labels`、`batch["pair_index"]/["pair_edit"]/["y_teacher"]/["frame_weight"]` 取监督；这些键永不进入 context。
7. 按 Track 计算损失 → backward → clip → step；记录 loss 分项、有效样本数、grad finite、Encoder call count。

**为什么把标签通道单独写**：V543 的核心风险不是模型结构，而是「Bridge 看起来参数自由，实际上把监督偷偷带进来」。把 context 构造与标签读取写成两条互不引用的代码路径，是比单元测试更可靠的结构性隔离。

**为什么不再用 400+37,500 两阶段**：V542 的 W0 是「冻结 core、训练 head」，W1 是「联合训练」。V543 的 Encoder/D0/P0 全程冻结，W0 没有对应语义；保留 400 只作 LR warmup，既避免无意义阶段，又保留 37,900 总剂量锚点。该选择必须在 WP0 预注册，不能看完结果再改。

#### MOVE Track 损失（原样复用 V542）

```text
L_M = L_atom + 0.5 * L_pair + 0.02 * L_cos
```

- 复用 `key_atom_loss_v53mp` / `key_atom_loss_v3` 的现有函数与 mask；
- `lambda_strong = lambda_dead = 0`（V542 实际配方）；
- 只用 `label_status == "ok"` 的帧（`_frame_mask_for`）；
- `pair_index`/`pair_edit` 只作为损失输入，不进入模型；
- 不重新实现不同版本的损失。

#### PROCESS Track 损失（原样复用 V542 系数）

```text
L_P = 0.5 * L_ranking + 0.5 * L_q + 0.012385049119773344 * L_endpoint
```

- 直接调用 `ranking_loss_v53mp`、`progress_loss_v52`、`endpoint_loss_v53mp`，不调用 MOVE 损失；
- **PROCESS 的 frame mask 不能沿用 `_frame_mask_for`**：必须使用 endpoint 标签表的有效 mask，覆盖 `ok/terminal/masked/conflict`；terminal 行只要 endpoint 标签有效就进入训练；
- q 用 `q_mask`；endpoint 用 `endpoint_mask`；`endpoint_pos_weight` 从 inner-train 估计一次；
- `order_pairs` 用与 V542 相同的 `sample_order_pairs` 与 RNG 协议；
- 两条 Track 的 loss 数值不可直接比较；
- 若某 batch 在当前 Track 无任何有效监督（全 mask），跳过该 update 并记 `skipped_no_grad`；不得反复重采样直到有标签，也不得伪造零标签。

#### 梯度隔离（必须机器证明）

```python
# MOVE training
for p in encoder.parameters(): p.requires_grad_(False)
for p in process_refiner.parameters(): p.requires_grad_(False)
for p in move_refiner.parameters(): p.requires_grad_(True)
# backward 后：
assert all(p.grad is None for p in encoder.parameters())
assert all(p.grad is None for p in process_refiner.parameters())
assert any(p.grad is not None for p in move_refiner.parameters())
```

PROCESS 训练对称。测试 `test_v543_gradient_isolation` 必须同时覆盖两条 Track。

#### 训练日志与 checkpoint

- `run_records.jsonl`：每 100 updates 一行，含 track、decoder_id、seed、inner-val loss、core/wall time、grad finite、encoder call count、cache hit rate。
- `train_log.jsonl`：每 update 一行，含 loss 分项、有效样本数、LR、grad norm。
- `checkpoint_best.pt`：当前 Track inner-val 主损失最优；`checkpoint_last.pt`：每 500 updates + 结束。
- `metrics.json`：参数计数、identity、timing、outcome、reason code。

### 7.3 Packet 缓存（性能路径，不是正确性路径）

**默认策略**：live encoder + RAM LRU；缓存是可选加速。

- RAM LRU：200K packets（约 1 GB）；命中率目标 ≥80%。
- 磁盘缓存：按曲线分片，上限 30 GB；只在需要时构建；LRU 淘汰。
- 全量 14.7M 预构建不推荐（约 55–65 GB，构建/IO 成本高）。
- 缓存构建只允许 inner-train + inner-val；dev/confirm 永不入缓存。
- 缓存键与身份见 §5.7；任何不匹配都拒绝命中。
- `joint_finetune` 模式禁用缓存，实时前向 + 完整 autograd。

**缓存收益估计**：V542 的 data 2,017 s/37,900 updates（53 ms/update）；缓存命中后目标 10–15 ms/update，单 run 可省 25–30 min。

### 7.4 Resume 与 identity

**run identity 至少包含**：

```text
encoder_ckpt_sha256
bridge_spec_sha256
decoder_id / track / model_version
contract_version
seed
config_sha256
source_manifest_sha256
data_manifest_sha256
split_sha256
label_spec_sha256
sampler_spec_sha256
environment_sha256
code_fingerprint
cache_identity (if enabled)
```

**规则**：

- 恢复只从 `checkpoint_last.pt`，且 identity 逐键 + digest 匹配；不匹配 = `invalid_run`，不得静默新建。
- 恢复必须带回 model/optimizer/step/RNG/sampler counters/schedule。
- `test_v543_exact_resume`：连续 100 updates 与「50 + kill + resume 50」的权重、loss、records 逐位一致（CPU）。

### 7.5 Stage S2：条件 Adapter / 联合微调（不进入主排名）

只有 S1 证明「冻结表征不足」时才启动：

- MOVE 私有向量 Adapter；
- PROCESS 私有 token Adapter；
- 部分解冻 Encoder（最后 1 层或 norm）；
- 全联合微调（最后手段）。

**纪律**：

- 单独登记为 `V543-S2`，独立 run identity、独立预算、独立报告；
- 不与 S1 的固定 Encoder 排名混在一起；
- 可训练 Bridge 必须明确算 Encoder 还是 Decoder 预算，并重新做参数匹配；
- S2 的收益不能回溯改写 S1 的冠军。

---

## 8. 运行矩阵、剂量与资源边界

### 8.1 阶段矩阵

| 阶段 | 内容 | 训练次数 | 剂量 | 是否进 N_core |
|---|---|---:|---:|---|
| T0 | M0/P0 三 seed parity、基线锚定 | 0 | 0 | 否 |
| T1 | M1–M3、P1–P4：200-update smoke | 7 | 1,400 | 否 |
| T2 | 同上：2,000-update diagnostic + profile | 7 | 14,000 | 否 |
| T3 | 7 名候选 seed17 全剂量 | 7 | 265,300 | 是 |
| T4 | 7 名候选 seed29 全剂量 | 7 | 265,300 | 是 |
| T5 | MOVE top2 + PROCESS top2 补 seed43 | ≤4 | ≤151,600 | 是 |
| T6 | 各 Track 选择合格冠军 | 0 | 0 | 否 |
| T7 | 冠军组合、校准、自主 rollout | 0 | 0 | 否 |
| T8 | Flow/Diffusion/GraphGRU 条件实验 | 另计 | 另计 | 否 |

**首轮正式上限**：7 + 7 + 4 = **18 次正式训练**；18 × 37,900 = **682,200 updates**。

### 8.2 Seed 策略（比 V542 更稳健，但仍有成本控制）

1. **所有正式候选先完成 seed17、seed29**（T3/T4）。
2. 每条 Track 按预注册 screen score 选 **最多 2 名**，补 seed43（T5）。
3. **两 seed 只用于排名筛选，不得直接宣布科学冠军**；最终 H3 必须有完整三 seed 证据。
4. 若某 Track 没有合格候选，保留 M0/P0；这是正常负结果。
5. 不允许「seed17 好看就提前补 seed43」「seed43 不好就换 seed」；补种名单由冻结规则产生。

**Screen score（预注册）**：

```text
MOVE:   max(rmse_rel_improve / 0.03,
            fixed_coverage_direction_rel / 0.03)
PROCESS: max(delta_auprc / 0.02,
             q_mae_rel_improve / 0.05,
             ordering_abs_improve / 0.02)
```

tie-break：inner-val 主损失 → core time → 参数少 → decoder_id 字典序。

### 8.3 资源边界

| 项 | 建议值 |
|---|---|
| 单候选 MOVE Refiner | ≤40K trainable；目标 36K ±5% |
| 单候选 PROCESS Refiner | ≤40K trainable；目标 36K ±5% |
| 最终整模 | ≤150K；预计 129,619 |
| 推理延迟 p95 | ≤2× A0 同环境 |
| Encoder 执行次数 | 每当前构型恰好 1 次 |
| 训练 | 单路 GPU、无 swap、无 OOM；host ≤12 GB |
| Flow NFE | 固定 1/2/4/8；单独登记 |
| QC 调用 | 本阶段新增 0 次 |
| 缓存磁盘 | ≤30 GB；RAM LRU ≤2 GB |

### 8.4 时间与磁盘估算

| 项 | 估算 |
|---|---:|
| 单次正式 run（缓存命中） | 43–55 min |
| 单次正式 run（live encoder） | 60–75 min |
| 18 次正式训练 | 13–20 h |
| smoke + diagnostic + profile | 3–5 h |
| 缓存构建/预热 | 1–3 h |
| S0 parity + probe + basis | 2–4 h |
| 总计 GPU | 18–26 h |
| 缓存磁盘 | ≤30 GB |
| 报告/分析 CPU | 2–3 天 |

### 8.5 熔断与停止规则

- 单 run 超过 2× 预估 wall（>2.5 h）→ 检查数据/缓存/显存；不得静默延长。
- OOM/swap → `resource_failure`；修复后从 `checkpoint_last` 恢复；不得减 batch/帧数。
- 非有限 loss/grad → `invalid_run`；定位后重跑；不得跳过。
- H0/H1 任一失败 → 阻断该候选正式训练。
- 成本比 >2.0 → `gate_skipped_cost_ratio_exceeds_2`（科学结果保留，不晋级）。
- 预注册剂量用尽即停；不做 dose extension、不换 seed、不降阈值。

---

## 9. 评估、统计与最终选择

### 9.1 指标表（V543 新评估器）

| 维度 | 指标 | 用途 | 备注 |
|---|---|---|---|
| MOVE | `rmse_zero` | 主性能 | RMSE(u,y)/RMSE(0,y)；越低越好 |
| MOVE | `sign_accuracy` | 主性能/历史对照 | 教师显著对上的符号准确率；旧指标保留 |
| MOVE | `dead_fraction` | 硬性非劣保护 | 预测位移小于阈值的比例 |
| MOVE | `fixed_coverage.direction_at_3` | **新独立主指标** | 教师 top-3 abs(Δd) 对上的符号准确率 |
| MOVE | `fixed_coverage.topk_overlap_at_3` | 诊断 | 预测 top-3 与教师 top-3 的重合率 |
| MOVE | `pair_delta_d.mae` | 化学动作诊断 | 关键对距离变化误差 |
| MOVE | `basis_projection_residual` | Bridge 诊断 | 教师残差在 `key_v` 上的投影残差 |
| PROCESS | `endpoint.auprc` | 主判别 | 阈值无关 |
| PROCESS | `endpoint.recall_at_fpr_0p05` | 主操作点 | inner-val 上选阈值使 FPR≤0.05 |
| PROCESS | `endpoint.brier` | 校准 | 概率质量 |
| PROCESS | `endpoint.ece` | 校准 | 10 bins |
| PROCESS | `q.mae_curve_median` | 主回归 | 每曲线 MAE 的中位数 |
| PROCESS | `q.n_eval_curves` | 有效性 | 有效曲线数 |
| PROCESS | `q.median_points_per_curve` | 有效性 | 每曲线有效 q 点数中位数 |
| PROCESS | `q.frac_curves_ge3` | 有效性 | ≥3 点曲线占比 |
| PROCESS | `q.corr_curve_median` | **仅诊断** | 旧指标；受 2 点 Pearson 影响 |
| PROCESS | `ordering.pairwise_accuracy_full` | 主排序 | 同曲线/同 intent 内成对比较 |
| PROCESS | `ordering.pairwise_accuracy_cross_ts` | 主排序 | 跨 TS 成对比较 |
| 闭环 | endpoint arrival / TS crossing / collision / steps / cost | 集成门 | T7 单独报告 |

### 9.2 独立 fixed-coverage direction 的正式定义

对每个含 ≥1 个 changed K-pair 的帧：

```text
dd_y[l] = Δd_l(y) = ||x_i + y_i - (x_j + y_j)|| - ||x_i - x_j||
dd_u[l] = Δd_l(u) = ||x_i + u_i - (x_j + u_j)|| - ||x_i - x_j||
k = min(3, n_pairs)
T_k = 教师 |dd_y| 最大的 k 个 pair（ties 按 pair 索引）
P_k = 预测 |dd_u| 最大的 k 个 pair
fixed_coverage_direction@3 = mean_{l∈T_k} [ sign(dd_u[l]) == sign(dd_y[l]) AND |dd_u[l]| ≥ 0.005 Å ]
topk_overlap@3 = |P_k ∩ T_k| / k
```

**聚合**：帧均值 → 曲线均值 → seed 均值；曲线级 cluster bootstrap。按 Start/Pre-TS/Post-TS/Endpoint 与 breaking/forming/both 分层报告。

**为什么独立**：旧 `sign_accuracy` 的分母是「所有教师显著对」，新指标的分母是「教师最关键的 top-3 对」。两者可以背离：模型可能把大量容易对做对，却漏掉最关键对。H2/H3 用新指标，旧指标只作历史对照。

### 9.3 H0–H3 门

#### H0：合法性与等价性（任一失败 = `invalid_run`）

- schema/forward-view 白名单；
- 无监督泄漏（哨兵测试）；
- E(3)：平移不变、旋转/反射等变、原子置换等变、边置换等变；
- reverse intent 合同；
- 非关键原子零位移；
- typed-invalid 输入拒绝；
- restore/resume；
- 有限梯度、有限 loss；
- Encoder 每次 forward 恰好 1 次；
- 零残差 parity。

#### H1：资源（失败 = `resource_failure` / `budget_censored`）

- 单路、无 swap、无 OOM；
- host ≤12 GB；VRAM 不超线；
- core-time ratio ≤2.0（分母 = 同环境 M0/P0）；
- p95 latency ≤2× A0；
- 参数在预算带内。

#### H2：逐 seed 非劣（失败 = `scientific_failure`，结果保留）

**MOVE（对照同 seed V543 M0）**：

| 指标 | 允许退化 |
|---|---:|
| `rmse_zero` | ≤2.0% 相对 |
| `sign_accuracy` | ≤0.01 绝对 |
| `dead_fraction` | ≤0.01 绝对增加 |
| `fixed_coverage.direction_at_3` | ≤2.0% 相对 |

**PROCESS（对照同 seed V543 P0）**：

| 指标 | 允许退化 |
|---|---:|
| `endpoint.auprc` | ≤0.01 绝对 |
| `endpoint.recall_at_fpr_0p05` | ≤0.02 绝对 |
| `q.mae_curve_median` | ≤5.0% 相对 |
| `ordering.pairwise_accuracy_full` | ≤0.02 绝对 |
| `endpoint.ece` | ≤0.02 绝对增加 |

**规则**：每个正式 seed 都必须无 H2 breach；任一 seed breach → 该候选 `scientific_failure`，不得用另一 Track 的收益豁免。

**零基线规则**：任何相对阈值（RMSE/zero、fixed-coverage direction、q MAE）在对照基线为 0 或接近 0 时退化为绝对阈值：退化 ≤0.01（比例类）或 ≤0.005 Å（距离类）。该规则必须在 WP0 写死，不得事后选择「相对更有利」的解释。

#### H3：效应量与统计（对照同 seed V543 M0/P0）

**MOVE 路线（括号逻辑）**：

```text
(rmse_pass AND direction_non_inferior)
OR
(direction_pass AND rmse_non_inferior)

rmse_pass:                    rmse_zero 相对改善 ≥3.0%
direction_pass:               fixed_coverage.direction_at_3 相对改善 ≥3.0%
                              AND 绝对提升 ≥0.01
rmse_non_inferior:            rmse_zero 退化 ≤2.0%
direction_non_inferior:       fixed_coverage.direction_at_3 退化 ≤2.0%
```

**PROCESS 路线（括号逻辑）**：

```text
(auprc_pass AND q_non_inferior AND recall_non_inferior AND calibration_non_inferior AND ordering_non_inferior)
OR
(q_pass AND auprc_non_inferior AND recall_non_inferior AND calibration_non_inferior)

auprc_pass:                   endpoint.auprc 绝对提升 ≥0.02
q_pass:                       q.mae_curve_median 相对改善 ≥5.0%
                              AND ordering 非劣（退化 ≤0.02）
auprc_non_inferior:           AUPRC 下降 ≤0.01
q_non_inferior:               q MAE 相对退化 ≤5.0%
recall_non_inferior:          recall_at_fpr_0p05 下降 ≤0.02
calibration_non_inferior:     ECE 增加 ≤0.02
ordering_non_inferior:        pairwise_accuracy_full 下降 ≤0.02
```

**统计显著性**：

- 三 seed 中 ≥2 个同向；
- 每个 seed 无 H2 breach；
- 三 seed seed-averaged curve-level paired bootstrap 10,000 draws，95% CI 下界 >0；
- one-sided p = `(1 + #{seed-averaged paired improvement ≤ 0}) / 10001`；
- MOVE 两个 exit 各 α=0.025，各自对 eligible candidates 做 Holm；
- PROCESS 单一 family（2 endpoints × eligible candidates）α=0.025，Holm；
- 报告未校正与校正后结果。

**效应量 margin**（用于 tie-break，不替代显著性）：

```text
move_rmse:        relative_improvement / 0.03
move_direction:   min(relative_improvement / 0.03, absolute_improvement / 0.01)
process_auprc:    delta_auprc / 0.02
process_q:        relative_q_mae_improvement / 0.05
```

### 9.4 统计合同

| 项 | 冻结值 |
|---|---|
| bootstrap unit | curve（reaction cluster） |
| pairing | 同 cluster、同 frame set；候选与基线共享 resample indices |
| draws | 10,000 |
| CI | percentile 95%（2.5/97.5） |
| p 值 | one-sided improvement paired cluster bootstrap |
| seed aggregation | 每 draw 先 seed-average，再比较 |
| 缺失 cluster | 两边同时丢弃；报告丢弃数；不插补 |
| frame-level bootstrap | 禁止 |
| 多重比较 | Holm；family 定义见 §9.3 |
| 阈值冻结 | WP0 签认后不得按 seed17/dev 结果修改 |

**q 有效性规则**：

- 报告 `n_eval_curves`、`median_points_per_curve`、`frac_curves_ge3`、`frac_curves_ge5`。
- 若 `frac_curves_ge3 < 0.70`，`q.corr_curve_median` 只作诊断，不进入结论；q MAE 与 ordering 仍是主指标。
- 禁止用「两点 Pearson=±1」的曲线支持「q 很好」。

### 9.5 基线、dev 与 confirm

- **基线**：S0 在 V543 evaluator 下重算 M0/P0 的 per-seed inner-val 文档；SHA 冻结；H2/H3 只用这份基线，不用 V542 归档数字。
- **dev**：只在 T6 冻结冠军后、T7 集成时一次性访问 589 曲线；不用于选择、不调阈值、不生成缓存特征；访问账本 append-only。
- **confirm**：永封存；`confirm_read=false`；任何读取都是 `invalid_run`。
- **tau**：只在 inner-val 校准一次（target precision 0.9）；dev 不重校准。
- **新 QC**：0 次。

---

## 10. 集成、Controller 与 Rollout

### 10.1 四组合对照（T7）

| 组合 | MOVE | PROCESS | 目的 |
|---|---|---|---|
| B0 | M0 | P0 | 双基线 |
| BM | M* | P0 | MOVE 改进是否带来连续路径收益 |
| BP | M0 | P* | PROCESS 改进是否改善端点判断 |
| BMP | M* | P* | 双专家组合是否出现意外错误停止/几何退化 |

**规则**：

- 单步指标在 inner-val 上比较；rollout 只在 dev 上一次性执行。
- BMP 通过集成门需同时满足：两条 Track 的 H2 不退化；misstop（start/TS）不增加；collision 不增加；p95 延迟 ≤2× A0。
- 若 BMP 不通过，按预注册顺序选择 BM → BP → B0；不得看完 rollout 再改顺序。
- 单步进步但 rollout 无进步 → 只能宣布「单步 Decoder 改进」，不能宣布「路径导航成功」。

### 10.2 Rollout 协议（冻结）

- 初始构型：dev 589 曲线上的冻结起点集；每个起点只跑一次。
- 推进预算：固定步数/固定 wall；双向 intent 分开报告。
- 端点阈值：`tau_end` 来自 inner-val 校准；不在 dev 重算。
- 异常处理：非法动作 → 拒绝并记录；不静默回退到参考路径。
- 物理松弛：**不做**；与 reference-normalized rollout 分开报告。
- 指标：endpoint arrival、TS crossing、collision、steps、path length、success、cost。
- 安全：任何 No-Move 必须记录 `p_end`、`chemical_state_ok`、`is_ts`；`q` 不得出现在 No-Move 决策日志的触发原因中。

### 10.3 Controller 验收

- `test_v543_no_move_contract`：
  - `q=+1`、`p_end` 低 → 必须 MOVE；
  - `q=-1`、`p_end` 高且化学状态一致 → 允许 No-Move；
  - TS 行 → 永远 MOVE；
  - `p_end` 高但化学状态不一致 → 必须 MOVE。
- Controller 不参与训练；不进入 `V543Model.forward`；不在单步 benchmark 中触发。

---

## 11. 代码修改清单

### 11.1 目录树（新增，不改 V542）

```text
pes2ts_core/pesn/models/v543/
├── __init__.py
├── contracts.py              # DecoderContextV1 / MoveContextV1 / ProcessContextV1 / ProcessResidual
├── coarse_encoder.py         # A0 strict load + 三个 hook + u0/q0_raw/e0
├── latent_bridge.py          # key_v / pair_s / key graph / token layout / 审计
├── assembly.py               # 单 Encoder + 双 Refiner 组装
├── registry.py               # Decoder 注册、发现、构造、预算与版本
├── model.py                  # V543Model 唯一 forward
├── move/
│   ├── __init__.py
│   ├── residual_mlp.py       # M1
│   ├── gvp.py                # M2
│   ├── vector_gnn.py         # M3
│   ├── flow.py               # M4（条件）
│   └── diffusion.py          # M5（条件）
└── process/
    ├── __init__.py
    ├── residual_mlp.py       # P1
    ├── deepsets.py           # P2
    ├── graph_gnn.py          # P3
    ├── graph_transformer.py  # P4 + P4-C
    └── graph_gru.py          # P5（条件）

pes2ts_core/pesn/offline/training/
├── v543_train.py             # 单 Track 训练器
└── v543_feature_cache.py     # Packet 缓存

pes2ts_core/pesn/offline/evaluation/
├── v543_eval.py              # MOVE/PROCESS 新评估
└── v543_rollout.py           # 自主路径压力测试

scripts/agent_ops/
├── v543_schema.py            # 配置/签认检查
├── v543_campaign.py          # smoke/profile/diagnostic
├── v543_formal.py            # 正式训练编排
├── v543_stats.py             # 统计、晋级、最终选择
└── v543_bundle.py            # 统一打包与验证

config/campaigns/
├── pes2ts_v543_dual_decoder_50k.yaml
└── v543_templates/
    ├── preregistration_signoff.template.json
    ├── runplan_pin.template.json
    ├── budget_policy.template.json
    ├── parity_certificate.template.json
    └── decoder_context.template.json

tests/pesn/test_v543_*.py
tests/agent_ops/test_v543_*.py
docs/design/decisions/ADR-0034-V5.4.3-双Decoder与DecoderContextV1.md
```

### 11.2 文件职责与关键 API

| 文件 | 关键 API | 职责 |
|---|---|---|
| `contracts.py` | `DecoderContextV1`、`MoveContextV1`、`ProcessContextV1`、`ProcessResidual`、`assert_no_supervision` | 冻结数据合同；只读视图；字段校验 |
| `coarse_encoder.py` | `CoarseEncoder.load(checkpoint, seed)`、`forward(batch) -> CoarsePacket` | A0 strict load；`z/h/edge_msg/u0/q0_raw/e0`；hook 计数 |
| `latent_bridge.py` | `build_context(packet, forward_view) -> DecoderContextV1`、`bridge_spec_sha256()` | 参数自由 Bridge；向量基底；pair/key/token；审计 |
| `assembly.py` | `assemble_v543(encoder, move_refiner, process_refiner)` | 冻结 Encoder + 双 Refiner；参数分组；预算断言 |
| `registry.py` | `register`、`build`、`describe`、`assert_budget` | Decoder 插件协议；版本/能力/NFE/参数 |
| `model.py` | `V543Model.forward(forward_view)`、`forward_debug` | 唯一推理流程；五键外部合同 |
| `v543_train.py` | `V543TrainerConfig`、`train_run_v543` | 单 Track 训练；损失；checkpoint；identity；resume |
| `v543_feature_cache.py` | `PacketCache`、`cache_key`、`assert_cache_identity` | LRU/磁盘缓存；命中统计；身份拒绝 |
| `v543_eval.py` | `evaluate_v543`、`evaluate_baseline`、`paired_curve_bootstrap` | 新旧指标；基线；统计输入 |
| `v543_rollout.py` | `rollout`、`controller_step` | 闭环压力测试；No-Move 语义 |
| `v543_schema.py` | `--prereg` / `--formal` | 配置、阈值、预算、record、SHA |
| `v543_campaign.py` | `smoke` / `diagnostic` / `profile` | 工程筛；资源；参数；hook；缓存 |
| `v543_formal.py` | `run-stage` / `status` / `freeze-runplan` | 阶段编排；ledger；门 |
| `v543_stats.py` | `extract/h2/screen/h3/select/activate` | 指标表；H2/H3；Holm；冠军 |
| `v543_bundle.py` | `build/verify` | 打包；SHA；strict-load；只读 |

### 11.3 复用与禁止修改

**复用（只读导入）**：

- `pes2ts_core/pesn/models/v542/backbone_a0.py`、`decoder.py`、`loader.py`、`interfaces.py`、`identity.py`；
- `pes2ts_core/pesn/models/key_atom_heads_v3.py`、`key_atom_egnn_v3.py`；
- `pes2ts_core/pesn/offline/training/key_atom_v53mp_train_v3.py` 的损失子函数、sampler、label helpers；
- `pes2ts_core/pesn/offline/evaluation/key_atom_protocols.py` 的 `evaluate_g53`/`exact_delta_d`；
- `pes2ts_core/pesn/models/v542/identity.py` 的 `canonical_sha256`；
- `pes2ts_core/pesn/offline/training/v541_identity.py` 的 manifest/RNG/outcome 工具；
- `scripts/agent_ops/v542_identity.py` 的 `build_identity_record`。

**禁止**：

- 修改 V542 冻结文件、V541/V542 报告、旧 run/metrics；
- 修改 `pair_index/pair_edit` 语义；
- 在 V543 包内复制一份 `w_v` 或重新训练 Encoder；
- 用 `strict=False` 加载；
- 在 Decoder 内读取 `batch`、`pos`、监督键；
- 用 `torch.compile`；
- 在 `joint_finetune` 模式读 detached 缓存。

### 11.4 CLI（在 WSL `pes2ts-gpu` 内执行）

```bash
cd /mnt/e/Calculations/AI4S_ML_Studys/PES2TS

# WP0
python scripts/agent_ops/v543_schema.py --prereg \
  --config config/campaigns/pes2ts_v543_dual_decoder_50k.yaml

# S0
python scripts/agent_ops/v543_campaign.py parity --config ...
python scripts/agent_ops/v543_campaign.py probe --config ...
python scripts/agent_ops/v543_campaign.py basis --config ...

# S1 工程筛
python scripts/agent_ops/v543_campaign.py smoke --config ...
python scripts/agent_ops/v543_campaign.py diagnostic --config ...
python scripts/agent_ops/v543_campaign.py profile --config ...

# 正式
python scripts/agent_ops/v543_formal.py freeze-runplan --config ...
python scripts/agent_ops/v543_formal.py run-stage T0 --config ...
python scripts/agent_ops/v543_formal.py run-stage T3 --config ...
python scripts/agent_ops/v543_formal.py status --config ...

# 统计/选择/打包
python scripts/agent_ops/v543_stats.py extract --config ...
python scripts/agent_ops/v543_stats.py h2 --config ...
python scripts/agent_ops/v543_stats.py screen --config ...
python scripts/agent_ops/v543_stats.py h3 --config ...
python scripts/agent_ops/v543_stats.py select --config ...
python scripts/agent_ops/v543_bundle.py build --config ...
python scripts/agent_ops/v543_bundle.py verify --config ...
```

### 11.5 测试矩阵（最低要求）

| 测试 | 必须证明 | 门 |
|---|---|---|
| `test_v543_zero_residual_parity` | 零残差 = V542 A0（三 seed） | S0-A |
| `test_v543_q0_raw_tanh_identity` | `tanh(q0_raw)==q` | S0-A |
| `test_v543_encoder_forward_once` | 每构型恰好 1 次 Encoder | H0 |
| `test_v543_hook_fires_once` | `message_mlp[-1]`/`norm_h[-1]`/`fc2` 各 1 次 | S0-A |
| `test_v543_decoder_no_xyz` | Decoder 无法读 `pos` | H0 |
| `test_v543_no_supervision_leak` | 哨兵监督值不进入 context | H0 |
| `test_v543_track_view_isolation` | MOVE 视图无 q/e；PROCESS 视图无 u0/key_v | H0 |
| `test_v543_rotation_equivariance` | 旋转输入 → 旋转 `key_v/u0/δu` | H0 |
| `test_v543_translation_invariance` | 平移不改变预测 | H0 |
| `test_v543_permutation_equivariance` | 原子/边/对置换一致 | H0 |
| `test_v543_reverse_intent` | 正反 intent 合同 | H0 |
| `test_v543_pair_index_ownership` | 规范序/归属正确；监督比对只读 | H0/WP2 |
| `test_v543_gradient_isolation` | 不更新 Encoder/另一 Track | H0 |
| `test_v543_exact_resume` | 中断恢复等价 | H0 |
| `test_v543_flow_no_reencode` | NFE 不重复 Encoder | M4 |
| `test_v543_flow_zero_gate` | 初始任意采样 `δu=0` | M4 |
| `test_v543_flow_rotation_noise` | 同噪声旋转一致 | M4 |
| `test_v543_no_move_contract` | q/TS 不误触发 No-Move | Controller |
| `test_v543_graph_gru_reset` | 每帧重置隐藏状态 | P5 |
| `test_v543_cache_identity` | 错误 checkpoint/几何身份不命中 | Cache |
| `test_v543_param_matching` | 每 Track 参数在 ±5% 带内 | H1 |
| `test_v543_basis_rank` | 8 通道秩/条件数报告 | S0-C |
| `test_v543_fixed_coverage_metric` | 新指标定义与分母正确 | Eval |
| `test_v543_bundle_strict_load` | bundle 可 strict-load/只读 | WP9 |
| `test_v543_legacy_files_untouched` | V542 文件 SHA 不变 | WP9 |
| `test_v543_docs_tokens` | 文档门 token 齐全 | WP0 |

**目标**：新增 ≥30 个测试、≥150 条断言；全仓 pytest 保持绿色；`docs_consistency --strict` CLEAN。

---

## 12. 工作包、依赖与日历

### 12.1 WP0–WP9

| WP | 任务 | 关键交付 | 完成标准 | 依赖 | 估算 |
|---|---|---|---|---|---:|
| WP0 | 预注册与治理 | ADR-0034、campaign YAML、validator、五模板、registry `proposed`、阈值签认 | `--prereg` 通过；docs gate CLEAN；阈值/预算/候选/矩阵/seed 全部签认；旧冻结文件未动 | 本方案 | 3 d |
| WP1 | A0 迁移 | `coarse_encoder.py`、`assembly.py`、parity 证书 | 三 seed strict-load；`u0/z/q/e0` parity；`tanh(q0_raw)==q`；hook 1 次；CPU bitwise/GPU ≤1e-6 | WP0 | 3 d |
| WP2 | Bridge 工程化 | `contracts.py`、`latent_bridge.py`、schema SHA、审计 | 不读监督；字段/形状/单位合法；E(3)；pair 集合审计；rank/conditioning；cache 身份 | WP1 | 4 d |
| WP3 | 信息充分性诊断 | `probe.py`、`basis_audit.py`、S0-B/C 报告 | 明确「Bridge 是否足够」；go/no-go；不足则回炉 Bridge，不加深 Decoder | WP2 | 3 d |
| WP4 | Decoder Registry 与训练器 | `registry.py`、`v543_train.py`、`v543_feature_cache.py` | 候选可注册/构造/加载；单 Track 训练；gradient isolation；exact resume；缓存身份 | WP2/WP3 | 5 d |
| WP5 | MOVE 模型 M1–M3 | 三个 Decoder + 测试 | 参数带内；smoke/梯度/显存/恢复/等变全过；零残差 parity | WP4 | 5 d |
| WP6 | PROCESS 模型 P1–P4（+P4-C） | 四个 Decoder + 对照 | 同 token 定义；mask/置换/进度/端点测试；参数匹配 | WP4 | 5 d |
| WP7 | 50K 工程筛 | smoke、diagnostic、profile 账本 | H0/H1、学习启动、资源曲线、成本比、预算签认 | WP5/WP6 | 3 d + compute |
| WP8 | 50K Benchmark | T0–T6、H2/H3、两冠军或留任 | 18 run 账本；三 seed 统计；Holm；selection 冻结 | WP7 | 3 d compute + 2 d analysis |
| WP9 | 集成与归档 | T7、bundle、五份报告、registry `executed`、V5.4.4 交接 | 四组合 rollout；bundle strict-load；负结果/删失全账；旧文件未动 | WP8 | 4 d |

### 12.2 依赖与并行

```text
WP0 ─→ WP1 ─→ WP2 ─→ WP3 ─┬─→ WP4 ─┬─→ WP5 ─┐
                            │        └─→ WP6 ─┼─→ WP7 ─→ WP8 ─→ WP9
                            └─ 若 S0-B/C 不足：回炉 WP2，不得进入 WP4
```

- WP5 与 WP6 可以在 WP4 完成后并行；共用 registry/trainer，但各自独立测试。
- WP3 的 probe/basis 可与 WP4 的 registry 骨架并行，但 **WP4 的正式训练必须等 WP3 go**。
- WP8 的 T0 parity 可在 WP7 之前先跑；T1/T2 必须先于 T3。
- WP9 的 bundle 可在 WP8 选择冻结后立即开始，rollout 可与报告并行。

### 12.3 建议日历（2026-11-03 起，含 1 周缓冲）

| 周 | 日期 | 主要工作 | 出口 |
|---|---|---|---|
| W1 | 11/02–11/06 | WP0、WP1 | ADR/config/validator/签认；三 seed parity |
| W2 | 11/09–11/13 | WP2、WP3 | DecoderContextV1 冻结；S0-B/C go/no-go |
| W3 | 11/16–11/20 | WP4 | Registry/trainer/cache/resume 通过 |
| W4 | 11/23–11/27 | WP5、WP6 | M1–M3、P1–P4 单元/等变/参数测试 |
| W5 | 11/30–12/04 | WP7 + T0–T3 | smoke/diagnostic/profile；seed17 正式 |
| W6 | 12/07–12/11 | T4–T6 + T7 | seed29；top2 seed43；两冠军；四组合 |
| W7 | 12/14–12/18 | WP9 + 缓冲 | bundle、五报告、registry executed、交接 |

**日历不是承诺**：若 WP1 parity 失败、WP3 判定 Bridge 不足、或正式 run 出现 `resource_failure`，日历顺延；不得压缩剂量/阈值/seed 来追日期。

---

## 13. 风险、签认事项与交付

### 13.1 风险登记

| # | 风险 | 触发信号 | 影响 | 缓解 | 回退 |
|---|---|---|---|---|---|
| R1 | Bridge 信息不足 | S0-B probe 明显低于 M0；S0-C 条件数差 | Decoder 再深也无用 | 先改 Bridge；必要时 S2 Adapter | 回炉 WP2，不进入 WP4 |
| R2 | 监督泄漏 | 哨兵测试失败；pair 审计不一致 | 结论无效 | 运行时白名单 + 离线审计 + 代码评审 | `invalid_run`，修复重跑 |
| R3 | A0 parity 漂移 | `u0/q/e0` 不逐位一致 | 基线不可比 | 复用 V542 adapter/loader；hook 计数；CPU 逐位 | 阻断所有训练 |
| R4 | PROCESS 指标虚高 | q_corr≈1 但 q MAE 不降；有效点 <3 | 错误晋级 | q MAE/ordering/有效点数同报 | q_corr 只作诊断 |
| R5 | 缓存膨胀/损坏 | 磁盘 >30 GB；命中率 <50%；SHA 不匹配 | 训练中断/结果污染 | LRU、分片、身份校验、live fallback | 关闭缓存，live encoder |
| R6 | seed43 MOVE 不稳定 | 前两 seed 好、seed43 H2 breach | 冠军无效 | 三 seed 必需；H2 逐 seed | `scientific_failure`，保留 M0 |
| R7 | 算力超支 | 单 run >2.5 h；OOM/swap | 预算/日历 | 熔断、恢复、单路 | `resource_failure`/`budget_censored` |
| R8 | 选择过拟合 | 7 候选 × 2 seed 的噪声 | 假冠军 | top2 预注册、Holm、三 seed、dev 一次性 | 无合格则留任 |
| R9 | rollout 失败 | 单步好但闭环差 | 不能宣称路径成功 | 四组合、物理松弛分开、V6 物理反馈 | 只声明单步改进 |
| R10 | 治理漂移 | docs gate 失败；SHA 不一致 | 不可复现 | WP0 冻结；append-only；SHA 记录 | 停止并审计 |

### 13.2 需要用户/负责人签认的 8 个决策

| # | 决策 | 建议 | 不签认的后果 |
|---|---|---|---|
| D1 | 接受 8 处修正（F1–F8） | 全部接受 | 合同不可冻结，WP1/WP2 无法开始 |
| D2 | 新 fixed-coverage 定义（top-3 教师对） | 采用 §9.2 | H2/H3 仍用重复指标，结论不可信 |
| D3 | q MAE/ordering 为主、q_corr 为诊断 | 采用 | PROCESS 可能被 q_corr 虚高误导 |
| D4 | 同 Track 参数匹配 36K ±5% | 采用 | 容量与结构混杂，归因失败 |
| D5 | MOVE 损失沿用 V542（不含 strong/dead） | 采用 | 与 V542 基线不可比 |
| D6 | 37,900 flat + 400 warmup，不设 W0/W1 | 采用 | 训练语义含混 |
| D7 | 18-run 矩阵与两 seed 筛选 + 三 seed 终审 | 采用 | 成本失控或单 seed 假冠军 |
| D8 | T7 rollout 只在 dev 一次性执行，confirm 永封 | 采用 | 选择泄漏/封存破坏 |

### 13.3 交付物

```text
PES2TSBundleV543/
├── encoder/
│   ├── A0 checkpoint（三 seed 或选定 seed）
│   └── identity / SHA
├── bridge/
│   ├── decoder_context_v1.json（schema + SHA）
│   └── feature / parity certificates
├── move_expert/
│   ├── checkpoint_best.pt
│   └── move_benchmark.md
├── process_expert/
│   ├── checkpoint_best.pt
│   └── process_benchmark.md + calibration.json
├── controller/
│   └── legal_endpoint_policy.json
└── provenance/
    ├── data / split / sampler SHA
    ├── training identities
    ├── selection decisions
    └── rollout / resource reports
```

**五份可审计报告**：

1. `encoder_bridge_parity`：三 seed parity、hook、q0_raw、basis/probe；
2. `move_benchmark`：M0–M3（+M4 条件）、H2/H3、成本；
3. `process_benchmark`：P0–P4（+P4-C/P5）、H2/H3、校准、有效点数；
4. `integration_rollout`：B0/BM/BP/BMP、四组合、安全、成本；
5. `final_selection`：两冠军/留任、理由、失败与删失、交接。

**账本**：`outputs/pesn/v543_dual_decoder_50k/` 下 `run_ledger.jsonl`、`run_records.jsonl`、`train_log.jsonl`、`metrics.json`、`h2_report.json`、`h3_report.json`、`final_selection.json`、`activation_decisions.json`；append-only，不因最终选择删除失败记录。

### 13.4 完成定义（DoD）

- S0-A 三 seed parity 全过；
- S0-B/S0-C 有明确 go/no-go 结论；
- 7 个候选完成 smoke/diagnostic/profile；
- 18 次正式 run（或按规则少跑）全部入账，含失败/删失；
- 两 Track 各自产生冠军或明确留任；
- H2/H3、Holm、CI、有效点数、成本同报；
- T7 四组合完成，rollout 与 reference-normalized 分开；
- bundle strict-load/read-only 通过；
- registry `V5.4.3` 从 `proposed` 转 `executed`（仅在有报告后）；
- V542/V541 冻结文件、旧报告、旧 run 未改；
- `docs_consistency --strict` CLEAN，全仓 pytest 绿。

### 13.5 明确不宣称

- 不宣称 A0 最优；A0 是 matched Encoder，不参加冠军。
- 不宣称 Flow/Diffusion 优于回归；除非 T8 有独立证据。
- 不宣称单步指标提升等于路径导航成功。
- 不宣称 q 高等于 PROCESS 好。
- 不宣称无物理松弛 rollout 的失败全部由 Decoder 造成。
- 不宣称 verified TS、生产 No-Move、部署资格或 G4 出口。

### 13.6 建议的下一步 5 个动作

1. **签认 D1–D8**（§13.2），特别是 F1–F8 修正。
2. **WP0**：写 ADR-0034 + campaign YAML + validator 骨架；只做治理，不训练。
3. **WP1**：A0 adapter + 三 seed parity；先证 `u0/q0/e0`，再谈 Bridge。
4. **WP2**：冻结 `DecoderContextV1`，完成 pair/监督/等变审计。
5. **WP3**：跑 S0-B/S0-C；若 Bridge 不足，回炉，不进入 WP4。

---

## 附录 A：Campaign YAML 草案

> 这是 `config/campaigns/pes2ts_v543_dual_decoder_50k.yaml` 的起点；WP0 必须由 `v543_schema.py` 正式校验并冻结 SHA。字段值为拟议，签认后不得按结果修改。

```yaml
schema: pes2ts_v543_dual_decoder_campaign_v1
campaign_id: PES2TS_V543_DUAL_DECODER_50K
version: 0.1.0-wp0
status: proposed

source:
  encoder: V542_A0
  seeds: [17, 29, 43]
  checkpoint_rule: same_seed_v542_a0_best
  checkpoints:
    "17": {path: outputs/pesn/v542_backbone_50k/runs/V542_A0_S17/checkpoint_best.pt,
           sha256: cab512b0094788093a9969484db7f9f303802d2fff0d7380bffb8923fc7f89bb}
    "29": {path: outputs/pesn/v542_backbone_50k/runs/V542_A0_S29/checkpoint_best.pt,
           sha256: c52db93253777a00637ea6cb41eb1366eeeaf37464d38a3d42c04692f496466b}
    "43": {path: outputs/pesn/v542_backbone_50k/runs/V542_A0_S43/checkpoint_best.pt,
           sha256: dd37c7e997d74c6426db9052918a2400687ce827600fc3dedd69efd01b6be8d4}
  freeze_encoder: true
  freeze_coarse_move: true
  freeze_coarse_process: true
  allow_confirm_access: false
  allow_new_qc_calls: false

data:
  tables_dir: /pes2ts_data/v53mp_50k_data
  split_manifest: /pes2ts_data/v53mp_50k_data/split_manifest_v53mp.json
  pool: 50000
  dev: 589
  confirm: 2000
  confirm_policy: sealed_never_read
  label_statuses: [ok, masked, terminal, conflict]
  move_eligible_status: ok
  process_endpoint_statuses: [ok, masked, terminal, conflict]

bridge:
  schema: geometric_latent_bridge_v1
  context_schema: decoder_context_v1
  graph_scalar_dim: 64
  key_scalar_dim: 48
  pair_scalar_dim: 48
  vector_channels: 8
  vector_type: polar
  reference_move_channel: 0
  parameter_free: true
  allow_decoder_raw_xyz: false
  allow_supervision_leak: false
  pair_direction: canonical_key_slot_order_with_signed_delta_b
  pair_source: forward_edges_only
  audit_pair_equivalence_sample: 10000

tracks:
  move:
    baseline: M0
    formal_candidates: [M1, M2, M3]
    conditional_candidates: [M4, M5]
    loss: frozen_v53mp_move_atom_pair_cos
    lambda_pair: 0.5
    lambda_cos: 0.02
    lambda_strong: 0.0
    lambda_dead: 0.0
    max_trainable_params: 40000
    target_params: 36000
    param_band_pct: 5.0
    output: residual_key_displacement
  process:
    baseline: P0
    formal_candidates: [P1, P2, P3, P4]
    matched_control: P4-C
    conditional_candidates: [P5]
    loss: frozen_v53mp_process
    lambda_rank: 0.5
    lambda_q: 0.5
    lambda_endpoint: 0.012385049119773344
    max_trainable_params: 40000
    target_params: 36000
    param_band_pct: 5.0
    output: residual_q_raw_endpoint_logit
    share_order_with_q: true
    process_frame_mask: endpoint_label_mask_all_statuses

training:
  mode: decoder_only
  encoder_frozen: true
  optimizer: AdamW
  learning_rate: 3.0e-4
  warmup_updates: 400
  weight_decay: 1.0e-4
  grad_clip: 1.0
  precision: fp32
  updates: 37900
  val_every: 100
  batch_curves: 8
  frames_per_curve: 4
  sampler: v541_curve_stratified_v1
  sampler_spec_sha256: 0769817ef01dca3abbb1d67f5405e98138b4707e74b690240d3f9b6821b65d9e
  inner_val_max_samples: 8192
  resume_requires_identity_match: true
  torch_compile: false

cache:
  enabled: true
  mode: frozen_encoder_only
  ram_lru_packets: 200000
  disk_max_gb: 30
  shard_curves: 256
  hit_rate_target: 0.80
  joint_finetune_forbidden: true

screening:
  smoke_updates: 200
  diagnostic_updates: 2000
  initial_formal_seeds: [17, 29]
  finalist_seed: 43
  finalists_per_track_max: 2
  diagnostic_h3_selection_forbidden: true
  diagnostic_loss_drop_min_pct: 5.0
  diagnostic_move_dead_margin: 0.20
  diagnostic_move_rmse_rel_max_pct: 10.0
  diagnostic_process_auprc_drop_max: 0.02

evaluation:
  matched_seed_baseline: true
  baseline: V543_M0_P0_recomputed
  cluster_bootstrap: true
  bootstrap_draws: 10000
  bootstrap_seed_root: 20261102
  multiple_testing_correction: Holm
  freeze_thresholds_before_runs: true
  dev_policy: final_frozen_candidates_only_one_shot
  confirm_policy: sealed
  new_metric: fixed_coverage_direction_at_3
  q_corr_role: diagnostic_only
  q_min_points_for_corr: 3
  q_frac_curves_ge3_min: 0.70

thresholds:
  move_rmse_regression_rel_max_pct: 2.0
  move_sign_drop_abs_max: 0.01
  move_dead_increase_abs_max: 0.01
  move_fixed_coverage_regression_rel_max_pct: 2.0
  move_h3_rmse_rel_min_pct: 3.0
  move_h3_direction_rel_min_pct: 3.0
  move_h3_direction_abs_min: 0.01
  process_auprc_drop_abs_max: 0.01
  process_recall_at_fpr_drop_abs_max: 0.02
  process_q_mae_regression_rel_max_pct: 5.0
  process_ordering_drop_abs_max: 0.02
  process_ece_increase_abs_max: 0.02
  process_h3_auprc_abs_min: 0.02
  process_h3_q_mae_rel_min_pct: 5.0
  process_fpr_constraint: 0.05
  move_exit_alpha_each: 0.025
  process_family_alpha: 0.025
  max_core_time_ratio: 2.0
  max_latency_ratio_p95: 2.0

resources:
  total_model_params_max: 150000
  move_refiner_params_max: 40000
  process_refiner_params_max: 40000
  p95_latency_ratio_max: 2.0
  encoder_forward_per_frame: 1
  single_lane: true
  swap_allowed: false
  host_limit_gib: 12

conditional_generation:
  flow_nfe: [1, 2, 4, 8]
  flow_activation: residual_bottleneck_evidence
  diffusion_default_enabled: false
  graph_gru_activation: process_systematic_failure
  sample_selection_uses_teacher: false
  activation_requires_new_signoff: true

run_matrix:
  formal_runs_max: 18
  t3_seed17: [M1, M2, M3, P1, P2, P3, P4]
  t4_seed29: [M1, M2, M3, P1, P2, P3, P4]
  t5_seed43_max: 4
  t6_selection: two_track_champions_or_retain
  t7_integration: [B0, BM, BP, BMP]
  t8_conditional: separate_identity
```

## 附录 B：预注册阈值表（签认版）

| 阈值 | 值 | 用途 |
|---|---:|---|
| `move_rmse_regression_rel_max_pct` | 2.0 | MOVE H2 |
| `move_sign_drop_abs_max` | 0.01 | MOVE H2 |
| `move_dead_increase_abs_max` | 0.01 | MOVE H2 |
| `move_fixed_coverage_regression_rel_max_pct` | 2.0 | MOVE H2 |
| `move_h3_rmse_rel_min_pct` | 3.0 | MOVE H3 |
| `move_h3_direction_rel_min_pct` | 3.0 | MOVE H3 |
| `move_h3_direction_abs_min` | 0.01 | MOVE H3 |
| `process_auprc_drop_abs_max` | 0.01 | PROCESS H2 |
| `process_recall_at_fpr_drop_abs_max` | 0.02 | PROCESS H2 |
| `process_q_mae_regression_rel_max_pct` | 5.0 | PROCESS H2 |
| `process_ordering_drop_abs_max` | 0.02 | PROCESS H2 |
| `process_ece_increase_abs_max` | 0.02 | PROCESS H2 |
| `process_h3_auprc_abs_min` | 0.02 | PROCESS H3 |
| `process_h3_q_mae_rel_min_pct` | 5.0 | PROCESS H3 |
| `process_fpr_constraint` | 0.05 | 操作点 |
| `move_exit_alpha_each` | 0.025 | MOVE 两出口 |
| `process_family_alpha` | 0.025 | PROCESS 单 family |
| `max_core_time_ratio` | 2.0 | 成本门 |
| `max_latency_ratio_p95` | 2.0 | 部署门 |
| `bootstrap_draws` | 10000 | 统计 |
| `bootstrap_seed_root` | 20261102 | 统计 |

## 附录 C：18-run 正式矩阵

| # | Run ID | Track | Decoder | Seed | 阶段 | 剂量 |
|---:|---|---|---|---:|---|---:|
| 1 | `V543_M1_S17` | MOVE | M1 | 17 | T3 | 37,900 |
| 2 | `V543_M2_S17` | MOVE | M2 | 17 | T3 | 37,900 |
| 3 | `V543_M3_S17` | MOVE | M3 | 17 | T3 | 37,900 |
| 4 | `V543_P1_S17` | PROCESS | P1 | 17 | T3 | 37,900 |
| 5 | `V543_P2_S17` | PROCESS | P2 | 17 | T3 | 37,900 |
| 6 | `V543_P3_S17` | PROCESS | P3 | 17 | T3 | 37,900 |
| 7 | `V543_P4_S17` | PROCESS | P4 | 17 | T3 | 37,900 |
| 8 | `V543_M1_S29` | MOVE | M1 | 29 | T4 | 37,900 |
| 9 | `V543_M2_S29` | MOVE | M2 | 29 | T4 | 37,900 |
| 10 | `V543_M3_S29` | MOVE | M3 | 29 | T4 | 37,900 |
| 11 | `V543_P1_S29` | PROCESS | P1 | 29 | T4 | 37,900 |
| 12 | `V543_P2_S29` | PROCESS | P2 | 29 | T4 | 37,900 |
| 13 | `V543_P3_S29` | PROCESS | P3 | 29 | T4 | 37,900 |
| 14 | `V543_P4_S29` | PROCESS | P4 | 29 | T4 | 37,900 |
| 15 | `V543_<MOVE_top1>_S43` | MOVE | top1 | 43 | T5 | 37,900 |
| 16 | `V543_<MOVE_top2>_S43` | MOVE | top2 | 43 | T5 | 37,900 |
| 17 | `V543_<PROC_top1>_S43` | PROCESS | top1 | 43 | T5 | 37,900 |
| 18 | `V543_<PROC_top2>_S43` | PROCESS | top2 | 43 | T5 | 37,900 |

**说明**：T5 的 4 行只有在对应 Track 有 2 名合格候选时才全部激活；不足 2 名则少跑。M0/P0 不训练，只在 S0/T0 做 parity/基线。P4-C、M4、M5、P5 不在 18 行内。

## 附录 D：ADR-0034 建议目录

```text
ADR-0034 V5.4.3 双 Decoder 基准与 DecoderContextV1
1. 背景与问题（V542 交接、A0 留任、Decoder 问题）
2. 决策（固定 Encoder + 参数自由 Bridge + 两 Track 独立残差）
3. 数据合同（DecoderContextV1 字段/单位/对称性/禁读）
4. 训练合同（37,900 flat + warmup；两 Track 独立；损失复用）
5. 评估合同（新 fixed-coverage；q MAE/ordering；校准；有效点数）
6. 预注册与治理（阈值、矩阵、seed、H2/H3、Holm）
7. 与 V542/V541/合同的关系（不修改旧文件；不改变任务语义）
8. 适用条件与不能声称的结论
9. 风险、回退与 S2 条件
10. 变更记录
```

## 附录 E：与用户提案 v2.0 的 10 个关键差异

| # | 提案 | 本方案 | 理由 |
|---|---|---|---|
| 1 | `q0_raw` 从 ProgressHead 取 | 用 `fc2` hook 取，并验证 `tanh(raw)==q` | 当前无公开出口 |
| 2 | `intent_pair_index` 由「有向反应图」构造 | 规范序 + 符号 `ΔB`；明确不是化学箭头 | forward 视图无原子级箭头 |
| 3 | `pair_s` 由「合法反应键」构造 | 48 维固定布局；不读 `pair_index/pair_edit` | 防监督泄漏 |
| 4 | 8 通道基底覆盖率 | 覆盖率 + 条件数 + 有效秩 + 留一 + probe | 8 向量几乎必然满秩，覆盖率会虚高 |
| 5 | 「≤40K」 | 同 Track 36K ±5% 参数匹配 | 排除容量混杂 |
| 6 | 37,900 = 400 + 37,500 | 37,900 flat + 400 warmup | Decoder-only 无 W0 语义 |
| 7 | MOVE 损失含 strong/dead | 沿用 V542 实际损失（无 strong/dead） | 不改损失才能比基线 |
| 8 | fixed-coverage 新指标但沿用旧基线 | 新指标 + V543 M0 重算基线 | 旧基线没有新指标 |
| 9 | q_corr 可作主指标 | q MAE/ordering 主，q_corr 诊断 | 2 点 Pearson 退化 |
| 10 | 先做 Flow 再决定 | 先确定性，Flow 条件激活 | 避免把回归换名 |

---

> **最终提醒**：V5.4.3 最重要的交付物不是某一个 Decoder 权重，而是 **`DecoderContextV1` + Decoder Registry + 可复现的两 Track 训练/评估合同**。有了它们，V5.4.4 可以替换离线 IRC Teacher，V6 可以加入物理校正，而不需要推翻 V5.4.3 的模型组织方式。

