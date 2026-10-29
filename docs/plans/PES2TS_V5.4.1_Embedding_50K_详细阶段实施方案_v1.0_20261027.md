# PES2TS V5.4.1 Embedding 50K Benchmark 详细阶段实施方案（单臂双功能头）

> 状态：**Executed-Frozen**（2026-10-29；WP0–WP7 已执行完毕，E*=E0 留任；执行结果与证据见 §19 执行指针与[最终报告](../reports/PES2TS_V541_Embedding_50K_实施与训练报告_20261029.md)。以下 §0–§18 为规划历史原文，未改写）
> 层级：V5.4 子战役；主归属 G3.P.2，冻结评估与交付延伸至 G3.P.3；不新增 G 阶段
> 取代关系：不取代开发宪法、MoveProcessContract v1、V5.3-MP 历史方案或报告；细化用户提供的 V5.4.1 设计冻结候选 v1.1
> 上游依赖：开发宪法 v2.9、阶段定义 v4.8、ADR-0031、MoveProcessContract v1、V5.3-MP S4 50K 冻结数据/配置/检查点、V5.4.1 Embedding 50K 设计冻结候选 v1.1
> 下游消费者：V5.4.2 Encoder Benchmark、V5.4.3 Decoder Benchmark、后续 V5.4 from-any-point / V6；不直接授予在线部署资格
> 当前版本：v1.2 @ 2026-10-29（执行收尾：新增 §19 执行指针；v1.1.1 分子 GNN 文献与实施细化；保留原文件名以维持引用）
> 编制核查点：仓库 `master@1d8b797`；正式 runplan 应另行 pin 实施后的代码 SHA，不得把本核查点当作执行冻结点

---

## 0. 执行摘要

本阶段只回答一个问题：**在固定 50K 数据、固定 StarEGNNV3 三层 Encoder、固定 MOVE/PROCESS 双功能头、固定联合损失和固定训练剂量下，哪一种局部 Embedding 能在不损害 MOVE 的前提下改善 PROCESS，并且收益值得其实测计算成本？**

执行采用一个联合目标、多个可替换 Embedding：

- 所有新正式 run 均为 `objective=unified_move_process`，同时训练 MOVE 与 `PROCESS(q, order_score, endpoint_logit)`；`order_score=q` 沿用现实现。
- 不重新组织 A/B/C 三任务臂，不做 MOVE-only / PROCESS-only 默认矩阵，不做 100K 扩容。
- 主候选为 E0、E1、E2、E3、E4；E5 条件启动；E6 只保留为另行批准的可选诊断。
- 正式主线为 seed 17 初筛、两个候选补齐 seeds 29/43；单 seed 不作最终架构结论。
- E0 可以最终胜出。没有稳定、足够大的收益时，不强制选择新 GNN。
- train/inner-val 用于训练和选择；候选 dev 只在候选、检查点、阈值和评估协议冻结后评估一次。WP1 仅允许单独登记的 E0 baseline-repair 重评，详细 dev 指标密封至 WP7，不得用于候选设计；confirm 2,000 curves 继续封存。
- 本阶段的模型输出仍只是 MOVE、`endpoint_reached` / `TS_candidate` 等研究量，不等于 verified TS / verified path，也不自动授予 No-Move 在线控制权。

本轮文献修订的结论：保留 E1–E4 主矩阵，以 GINE/PNA 检验化学邻域聚合，以 SchNet-like/Angle-MPNN 检验距离与角度归纳偏置；完整 PaiNN、EGNN、GemNet、NequIP/MACE 留给 Encoder 阶段评估。新增 §6.5–6.8、§7.5 和 §18；所有新增结构建议仍须在 WP0 冻结，不构成已实现能力或训练授权。

### 0.1 本方案对上游 v1.1 的执行裁决

| 编号 | 裁决 | 执行后果 |
|---|---|---|
| D01 | 上游“审计通过可复用 E0”与“必须重跑 E0×3”按作用域统一 | **重评总是先做；重训只在训练可比性审计失败或证据缺失时发生。** 评估器修复本身不触发重训 |
| D02 | 历史 V5.3 50K 完整任务 C×3 只是候选基线资产 | 只有通过 artifact、forward、数据、采样、优化器、调度、loader/resume 等价审计，才升级为 matched E0 |
| D03 | 旧 `sign=0.8546` 不作为正式绝对基线 | 先修 key-row/node-row `pair_index` 缺陷，再对 C×3 同检查点重评并固定新分母和指标 |
| D04 | V5.4.1 是 V5.4 的 Embedding 子战役 | 不等待旧 registry 中 from-any-point rollout 入口；WP0 需登记子战役并保留下游 from-any-point 边界 |
| D05 | E0–E6 仅在 `V541` 命名空间内有效 | run_id 使用 `V541_E1_S17` 等全称，避免与历史 30h 计划中的同名 E0–E6 混用 |
| D06 | 主实验固定 zero-init 标量残差门和 legacy edge 22 通道透传 | 不开 EdgeAdapter，不让固定 Encoder 消费新向量，不把容量/深度诊断组纳入 Embedding 赢家竞争 |
| D07 | 所有审计均 fail-closed | “没有找到证据”“证书 SKIP”“旧日志缺失”不等于等价通过 |
| D08 | dev 不用于反复排名或调参 | WP1 的 E0 baseline-repair 是密封的基线审计；候选 dev 只在 WP7 解冻评估一次。dev 可对已冻结升级作一次确认/否决回退 E0，但不得改选另一个 GNN、改阈值或改结构 |
| D09 | 任何 formal-run 比较需要共同协议冻结 | 正式 T1–T5 使用同一 common-protocol SHA 和兼容代码冻结点；中途科学配置变更应新建 runplan，不得混入原比较 |

### 0.2 当前事实，而非计划假设

1. 仓库目前没有 `v541` 模型、配置、脚本或测试；所有 V5.4.1 CLI 均为待实现接口。
2. 可复用基础是 V5.3-MP 训练栈，不是现成的 V5.4.1 runner：`key_atom_v53mp_train_v3.py`、`r3_v53mp_run.py`、双功能头、损失、sharded/columnar loader、tau 校准、export 和测试模式。
3. 50K 历史配置已经固定 400 warmup + 37,500 joint updates、8×4 stratified batch、AdamW、FP32、inner-val cap 8,192 和 inner-val loss 选检查点。
4. `evaluate_checkpoint` 当前用 `n_keys` 累加的 key-row 窗口筛选 node-row `pair_index`；`collate_v3` 明确对 pair/key index 加 node offset。该缺陷在详细方案启动时仍未修复。
5. S4 报告记录 loader-v2 50K 位级证书因内存问题 SKIP；因此 E0 复用当前只能标为 `unknown`，不能预先写成通过。
6. 50K rollout 与 confirm 均未完成；本方案只新做授权的 dev/inner-val 工作，不把 confirm 列为隐式收尾步骤。

---

## 1. 范围、合同与完成定义

### 1.1 科学范围

**允许改变：** `embedding_id`、该 Embedding 内部算子、内部合法局部图/角特征派生规则、相应参数量和实测资源协变量。

**保持共同：** 50K pool、每 seed 的 curve-level inner-train/inner-val 划分、dev/confirm 边界、F2 原始特征语义、合法有向拓扑、教师与标签、R3 common-core warmstart、三层 StarEGNNV3 core、MOVE readout、ReactionProcessHeadV53MP、联合损失及 λ、采样和训练剂量、inner-val 检查点规则、tau 校准纪律、FP32 和评估协议。

**明确不做：**

- 不修改旧 V5.3 模型代码、历史 run、历史 metrics 或只读报告来“原地升级”结果。
- 不引入历史帧、未来 XYZ、参考 TS/端点 XYZ、真实 q、E/g、curve/family/split 或教师字段进入 forward。
- 不改变 cutoff 4.5 Å、主图 32-neighbor、RBF12、坐标归一化、`d0=0.08 Å`、cap 0.20 Å。
- 不引入 Multi-basis、Cross-Attention、Residual MOVE decoder 或新的 PROCESS 头。
- 不把 SchNet-like、DimeNet-inspired、PaiNN scalar-projected 写成完整原论文模型复现。
- 不用离线指标或离线 rollout 宣称严格 TS 成功、生产部署或 G4 阶段出口。
- **本阶段 `new_QC_calls=0`。** 不调用 ACP/xTB/ORCA/DFT，不新增能量/梯度、OptTS、频率或 IRC 计算；只消费既有 50K 数据、标签、R3/V5.3 权重及本阶段训练的模型。真实物理闭环不是 V5.4.1 出口，未来如需要应另立授权战役。

### 1.2 阶段完成不是“训练跑完”

V5.4.1 只有同时交付以下结果才算关闭：

1. 合法性、等价性、几何、资源、恢复和评估分母证据完整；
2. matched E0 身份明确，且复用/重训分支有机器可审的决策记录；
3. 两个最终候选或提前终止原因均有完整三 seed/删失账本；
4. 预注册 H0/H1/H2/H3、curve-paired CI、失败数量和计算成本同报；
5. dev/rollout 的声明边界清楚，confirm 访问账本保持封存；
6. 产出唯一 `E*`（可以是 E0）及可 strict-load 的 `embedding_bundle_v1`；
7. V5.4.2 能区分 `frozen-E*` 与 `joint-finetune-E*`，不会静默改变本阶段赢家。

---

## 2. 总体工作分解与依赖

### 2.1 工作包 DAG

```text
WP0 治理/预注册冻结
  ├─→ WP1 评估器修复 + 历史 E0×3 重评 ───────────────┐
  └─→ WP2 V541Model / Legacy E0 / strict-load parity │
        ├─→ WP3 合法特征派生 + E1–E4 实现             │
        └─→ WP4-core 统一 runner / manifest / resume  │
WP3 ─→ WP4-candidate 候选接入                         │
WP1 + WP2 + WP4-core ─→ WP5 E0 可比性审计             │
WP3 + WP4 ─→ WP5 单路 smoke + W0/W1 分别 profiling    │
                                                     ↓
                                      WP6 正式 T0/T1/T2/T3/T4
                                                     ↓
                                      WP7 dev / 统计 / rollout / 冻结交付
```

WP3 的化学、空间、角度特征可在同一接口冻结后并行实现；正式 GPU 执行始终单路。任何并行开发不得同时改写同一模块或同一 runplan。

### 2.2 工作包总表

| WP | 主要交付 | 前置 | 放行证据 | 失败时动作 |
|---|---|---|---|---|
| WP0 | ADR/合同指针、registry、campaign schema、预注册 runplan | 本方案 | docs gate、schema validation、阈值/预算签认单 | 不进入实现或正式训练 |
| WP1 | 修复后的共享 pair 评估路径、E0 before/after 重评 | WP0 | 回归 fixture、pair 分母独立参照、C×3 重评报告 | 阻断所有绝对 sign/方向结论 |
| WP2 | `V541Model`、LegacyEmbedding、fixed core/readouts、E0 parity | WP0 | strict-load、u/z/q/endpoint/order parity、白名单一致 | 阻断候选训练 |
| WP3 | feature providers、E1–E4、条件 E5 | WP2 接口 | leak/E(3)/permutation/tie/degenerate/有限值测试 | 仅修复或轻量重设计，不改科学输入 |
| WP4 | objective 包装、参数组、plan-pin、manifest、resume、评估导出 | core 在 WP2 后；候选接入在 WP3 后 | 旧 C 完整任务语义等价、20+20 vs 40 resume、batch schedule hash | `invalid_run`，不得正式训练 |
| WP5 | E0 复用裁决、200-update smoke、W0/W1 成本模型、预算冻结 | WP1–WP4 | H0/H1、学习启动、资源曲线、runplan.pin | E0 重训或候选 resource_failure |
| WP6 | T0–T4 正式训练与诊断 | WP5 | 全量 run ledger、固定 updates、三 seed、H2/H3 | 保留科学失败/删失，不换 seed |
| WP7 | paired stats、dev、rollout、Pareto、bundle、model card | WP6 | 一次性冻结评估、strict-load handoff、完整声明边界 | E0 留任或结果降级，不追调 dev |

---

## 3. WP0：治理、接口与预注册冻结

### 3.1 任务

1. 新增 V5.4.1 ADR/设计登记，说明本阶段只扩展 Embedding 接口，不改变 MoveProcessContract v1 的任务语义。
2. 在机器可读 G–V registry 中登记 V5.4.1：`primary_g=G3.P.2`，`supporting_g` 包含 G3.P.1/G3.P.3/G0/G1；旧 V5.4 from-any-point 保留下游身份。
3. 新增 `config/campaigns/pes2ts_v541_embed_50k.yaml` 及 schema validator；配置未通过 validator 时 runner 拒绝启动。
4. 将 `objective=unified_move_process` 设为唯一外部任务名。过渡实现可内部映射旧 `arm="C"`，但新配置、日志、报表和 run_id 不暴露三臂矩阵。
5. 固定 `embedding_id` 注册表、forward-view 白名单、feature schema、artifact schema、失败词表和 run 状态机。
6. 将上游 v1.1 设计文件纳入执行证据链。编制时源文件 SHA256 为 `c2c1e5e5588ecd7109037095cd2035474722b366f953f1811cfcc0da56cfd24c`；执行时记录仓库内副本或只读来源的 SHA。
7. 签认本方案 §11 的候选阈值、成本接受规则、T4 对照策略、dev 否决规则、rollout 名单生成规则和单路预算规则。

### 3.2 预注册配置分层

为避免“每个模型 config 不同”误伤共同协议，manifest 至少拆成：

- `common_protocol_sha256`：数据、split、teacher、label、core、head、loss、optimizer、sampler、schedule、评估和统计协议；
- `embedding_spec_sha256`：候选内部结构、合法局部图、角度规则、初始化和参数量；
- `feature_recipe_sha256`：实际派生特征及缓存键；
- `code_commit` / `environment_sha256`；
- `runplan_sha256` 与 `runplan.pin.json`；
- `budget_policy_sha256`。

同一 seed 的所有主候选必须共享 `common_protocol_sha256` 和 batch schedule hash；只允许 Embedding 相关 SHA 按计划变化。

### 3.3 WP0 出口

- `python scripts/agent_ops/docs_consistency.py --strict` 通过；
- 新 schema 对缺字段、未知 Embedding、非统一 objective、confirm 授权、非法 forward key、非预期 EdgeAdapter/vector consumption 均 typed rejection；
- runplan 明确 nominal、conditional、diagnostic 三类 run，不把诊断组计入主赢家数；
- 所有待定阈值已签认；未签认项保持 `null` 并阻断 T1，而不是由 runner 填默认值。

---

## 4. WP1：评估器修复与历史 E0 重评

### 4.1 修复范围

主要现状位置：

- `pes2ts_core/pesn/offline/training/key_atom_bc_v3.py:400-469`；
- `scripts/campaigns/r3_v53mp_scale_paired_ci.py` 中镜像的 pair 收集逻辑；
- `pes2ts_core/pesn/features/key_atom_graph_v3.py:321-351` 证明 pair/key index 属于 batch-global node-row 空间。

实施时提取一个共享的 pair 归属/映射工具，避免两个评估入口分别修补后再次漂移。key-row 指针继续用于 `u/y/key_index`；node-row 指针用于 pair 所属图窗口，并同时校验 pair 两端属于同一图。不得用“增加 active pair 数”作为唯一正确性判断，应以独立构造的期望 pair 集为准。

### 4.2 必配回归样本

至少覆盖：

1. 两个图拼接且 `n_nodes != n_keys`；
2. key atom 在 node 行中非连续、非前缀排列；
3. 第二图 pair 的 node offset 大于第一图 key 总数；
4. 一个图无 active pair、一个图多 pair；
5. pair 一端不属于 K 时按合同忽略并记录原因，而非静默丢弃合法 pair；
6. 正/反 intent、terminal/masked 行的 MOVE eligibility 不被评估器修复改变。

每个 fixture 独立列出期望 active-pair 数、有效曲线/位置/pair 分母和 sign 结果；应包含旧实现会失败的样本。

### 4.3 历史基线重评

对 V5.3 50K 历史完整任务 C 的 seeds 17/29/43 做本阶段基线重评，不扩展到 A/B 重跑。另对既有 R3 reference 只重评 H2 所需的 MOVE guard 指标，使 R3 与候选使用同一修复后评估管线；若 R3 资产无法完成该重评，T1 保持阻断，不得静默删除 R3 guard：

1. 冻结三个 C `checkpoint_best.pt`、metrics、run record、config 和 R3 initializer/reference 的 SHA；
2. C×3 在每 seed 固定的 inner-val 8,192-sample 子集和相同 dev curve/position 上生成预测，分别保留旧口径和修复口径的审计结果；inner-val 重评用于初筛配对，dev 重评只作密封独立基线，二者不得混用；
3. 对 inner-val/dev 分别输出 before/after 的 active pair、有效 curve/position/pair 分母、sign、SRE、dead-zone、fixed-coverage direction、RMSE/zero；
4. q、ordering、endpoint 指标做保真检查，确认 pair 修复没有改变无关指标；R3 的 RMSE/zero 等 H2 参照也在同一 inner-val/dev 分层上重评并冻结；
5. 旧报告保持只读。inner-val 结果写入 `audits/evaluator_before_after.json`；详细 dev 结果写入 `audits/dev_baseline_sealed/`，记录访问人、目的、curve/position SHA、代码 SHA、时间和输出 SHA，在 WP7 前只暴露审计 pass/fail 与分母 sanity，不暴露可供候选调参的细粒度指标；
6. 以修复后的同 seed E0 为配对参照，以三 seed 聚合为汇总参照。WP1 的 dev baseline-repair 与 WP7 的 candidate-evaluation 是两种分别登记的访问，不得借“只读一次”省略任一访问记录。

### 4.4 WP1 出口与边界

- pair 分母与独立参照一致，无 silent pair drop；
- 修复前后预测输入、checkpoint、inner-val/dev 位置和标签 SHA 一致；
- 修复后的 E0 指标成为后续唯一正式绝对基线，R3 guard 参照口径同步冻结；
- dev 详细结果保持密封，候选结构、超参、阈值和初筛不得访问或引用；
- 该步骤是 **re-evaluation**，不改变历史权重，不自动意味着训练器等价，也不自动触发 E0 重训。

---

## 5. WP2：V541Model 拆解与 E0 数值等价

### 5.1 模块边界

计划新增：

```text
pes2ts_core/pesn/models/v541/
  interfaces.py
  embedding_registry.py
  model.py
  embeddings/{legacy,gin_e_chemical,pna_chemical,schnet_spatial,angle_mpnn,dual_stream,painn_scalar}.py
  feature_providers/{legal_neighbors,angle_triplets}.py
```

`painn_scalar.py` 不进入默认实现批次；只有 E6 另获批准时才创建。旧 `StarEGNNV3`、旧 heads、旧训练账本保持可读和可复现，不为方便而原地重写旧模型。

### 5.2 统一残差定义

以下为消除实现歧义的本方案默认，WP0 签认后冻结。主实验形式为：

\[
h_{v541}=h_{legacy}+\alpha\,W\,f_\phi(h_{legacy},V_{legal}),\qquad \alpha_0=0.
\]

- 五个主候选的 `f_phi` 节点初值均取 `h_legacy`，不 detach；差异只在候选内部消息算子及其合法局部视图 `V_legal`。
- `W` 非零初始化并投影到 48；`alpha` 是单一可训练标量，零初始化。不得同时把 `W` 零初始化，否则学习无法启动。
- E0 没有新分支，数学上等价于 `delta=0`，不使用“极小非零 alpha”冒充 parity。
- 主实验不用 channel-wise gate；若以后采用，应登记新 `embedding_spec`，不能混入首批。
- 输出给固定 core 的 `edge_scalar [E,22]` 始终是 legacy 原通道及顺序；化学/角度内部视图只影响 `delta_node`。
- `node_vector` 本阶段不被 core 消费；E6 的 scalar invariant 投影也不能声称验证了完整 PaiNN 向量优势。

如果实现评审改用 raw F2 而非 `h_legacy` 作为分支节点初值，应对全部候选统一修改 `embedding_spec` 版本并重新审查，不得逐候选选择不同基底。

### 5.3 E0 strict-load / parity 证书

对三个历史 C checkpoint 及代表性 fixture 执行：

1. 显式 state-dict 名称映射，`strict=True`；missing/unexpected keys 均为空，shape/dtype 全部一致；
2. R3 common-core 权重与历史加载路径逐张量一致；
3. `model.eval()`、相同 batch、相同设备和精度下比较 `u/z/q_pred/endpoint_logit/order_score`；
4. CPU 采用上游约定的 1e-7 量级容差；GPU 采用 §11.2 签认的 `atol/rtol`（本方案建议 1e-6/1e-5），在 smoke 前固定，不得看到差异后放宽；
5. `q_pred == order_score` 的现语义不变，非关键原子没有动作输出；
6. forward-view 白名单与旧模型一致，新 provenance 不得被 forward 读取；
7. E0 新 runner 的 objective→旧 C 完整任务兼容映射只改变名称，不改变 loss、mask、归一化或参数初始化。

### 5.4 初始化顺序

为保持同 seed 的 PROCESS head 与历史一致：

1. 按旧路径 `set_seed(seed)` 后构造 legacy backbone/head；
2. 先完成 legacy core/readout/PROCESS head 的加载或初始化并保存其 hash；
3. 再用独立、可复现的 embedding-init generator 初始化新分支，不得消耗旧 sampler 的随机序列；
4. sampler 继续沿用 `(seed, update, salt)` 纯函数协议，记录全计划 batch ID 流的 SHA；
5. E0 不创建会改变旧参数组或旧 RNG 消耗的空分支。

### 5.5 WP2 出口

E0 parity 全部通过后才允许实现候选进入训练 smoke。parity 失败属于实现阻断，不得通过重新训练掩盖。

---

## 6. WP3：合法特征与候选 Embedding

### 6.1 主候选冻结表

| ID | 首批结构 | 内部合法视图 | 输出与边界 | 特定审计 |
|---|---|---|---|---|
| E0 | 原始 element 16d + 21→48 linear | 原 F2 | legacy h；原 22 edge | strict-load / runner parity |
| E1 | 1 层 GINE-chem，48 channels | source/target 共价并图 + reaction edit 边 | 仅 residual node delta | edge 语义、成/断键方向、环境重复 |
| E2 | 1 层 PNA-chem，48 channels | 同 E1 | 仅 residual node delta | degree histogram 只 fit inner-train，记录 SHA 和成本 |
| E3 | **主配置 1 次** SchNet-like continuous-filter 更新，48 | legacy spatial 子集 + 当前距离重算 RBF12（§6.6） | 仅距离导出的 scalar 权重 | W0/W1 edge 成本、旋转/反射不变量 |
| E4 | 1 层 Angle-MPNN，48 | 预注册有限角三元组 | 只使用旋转/反射不变角度 | triplet 上限、tie、退化几何、CPU/RSS |
| E5 | GINE 24 + Angle 24，融合投影 48 | E1/E4 共用合法输入 | 条件启动；总 delta 48 | 参数/FLOPs/成本，与 E0-wide 对照 |
| E6 | PaiNN scalar-projected，可选诊断 | 原 spatial graph | 不进核心矩阵 | 只能声明 scalar-projected 诊断 |

E3 的第二次连续滤波更新不混入主 run；若需要，应另列 diagnostic，并相应补深度对照。E2 建议冻结为 aggregators=`[mean,max,min,std]`、degree scalers=`[identity,amplification,attenuation]` 的单一组合；WP0 签认后写入 `embedding_spec`，不得在看到训练结果后挑选其他组合。

### 6.2 化学图规则

- E1/E2 的 source/target union 和 reaction edit 组成对所有样本一致，不因模型而变。
- 内部 edge 至少保留原成/断键拓扑、source/target bond order 和 signed delta 的合法语义。
- 所有原子都编码；key 角色是合法条件，监督标签不是输入。
- E2 degree histogram 按每个 seed 的 inner-train 图分布流式计算，不能使用 inner-val/dev/confirm；统计文件绑定 split、graph recipe 和 seed SHA。
- 若主 loader 对同一样本有两 intent，两个方向的训练分布处理应与 sampler 一致并在 spec 中说明。

### 6.3 角度三元组规则

1. 仅从当前 `pos`、合法 edge/topology 和化学规则派生，不读取未来/参考构型。
2. 为消除“同一图”的歧义，建议主配置的角度邻居只取 legacy 共价 source/target union 与 intent pair；spatial-angle 扩展不进入首批，如需启用应另立 diagnostic spec。该默认在 WP0 签认后冻结。
3. 主配置每个中心最多选择 12 个角度邻居，目标约不超过 66 个无序邻居对/center。
4. 邻居排序键不得含 atom ID、curve ID 或输入排列位置。距离边界并列采用 **`reject_boundary_tie_shell`**：若整个同距离 shell 纳入会超过上限，则整壳不纳入；保留 tie 计数。无角中心使用显式 mask/token。
5. 若实现选择“整壳保留”而非拒绝，应在 runplan 前统一改成另一 spec，并用分块保证内存；不得按原子 ID 静默截断。
6. 夹角计算使用安全归一化、`eps`、cos clamp [-1,1]；零距离 typed error；缺角、共线和近共线样本均有测试。
7. 不使用带符号手性/镜像标签；O(3) 测试通过不代表手性敏感性已经解决。
8. triplet 按 batch 惰性生成或有界 LRU 缓存，可分块聚合；缓存键绑定 source data、graph recipe、embedding spec 和代码 SHA。

### 6.4 几何、泄漏和数值测试

每个候选均执行：

- 平移：`u,z,q,endpoint` 不变；
- 旋转和反射：`u(RX)=R u(X)`，scalar 不变；
- 原子重排：输出按同一排列等变，内部 neighbor/triplet 集不变；
- forward/reverse intent：q 和 endpoint 的目标语义正确；`u_rev≈-u_fwd` 只作诊断；
- 相同合法 forward、不同 teacher/metadata 的干预测试：输出完全一致；
- 单原子、双原子、无空间邻居、无角中心、密集邻居、距离 tie、近零键长、极端但合法坐标；
- 所有输出和梯度有限，无 NaN/Inf，非关键原子无直接动作；
- 小 batch 反传后，`alpha` 先获得非零有限更新，随后分支参数在 `alpha!=0` 时能获得梯度。首个 zero-gate step 的分支零梯度是预期，不作为失败。

### 6.5 常用分子 GNN 调研与本阶段取舍

检索日期为 2026-10-27；以下依据原论文和官方实现/算子文档，属于代表性方法梳理，不是完整排行榜。文献的分子属性、能量/力基准不能直接推出 PES2TS 的 MOVE/PROCESS 收益；表中“取舍”是本项目设计判断，来源链接见 §18。

| 模型族 | 主要表示与归纳偏置 | 对 V5.4.1 的价值及取舍 |
|---|---|---|
| GCN / GAT | 归一化邻居聚合 / 邻居注意力；原始版本不是专门的键特征化学模型 | 通用参照；不额外加入主矩阵，避免以模型名单长度替代机制覆盖 [R1–R2] |
| MPNN / D-MPNN | MPNN 是消息—更新—读出的框架；D-MPNN 维护有向键状态并排除即时反向回传 | 反应编辑具有边语义，值得后续比较；本轮已有 GINE 化学支路，新增持久边状态会增加另一实现轴，登记后续候选而非临时 E7 [R3–R4] |
| GIN / GINE | sum 聚合；GINE 将边特征纳入消息 | E1：低成本、可解释的化学邻域基线；不把 GIN 理论表达力结论视为连续几何任务保证 [R5–R6] |
| PNA | 多聚合器 × degree scaler | E2：检验邻居分布统计是否超越单一 sum；需控制图度数统计和参数量 [R7] |
| SchNet | 距离驱动的连续滤波 | E3：低阶几何基线；只移植单次滤波，不移植能量/力头或完整多层 SchNet [R8] |
| DimeNet / DimeNet++ | 有向边消息、距离和角度基函数；++ 调整结构提高效率 | 为 E4 提供角度动机；当前中心节点无序邻居对聚合不等于原论文有向边传播 [R9] |
| PaiNN | 标量与等变向量交互 | E6 仅 scalar-projected 诊断；完整向量跨层传播优先在 V5.4.2 设计 [R10] |
| EGNN | 用不变消息驱动等变坐标/特征更新 | 现有 StarEGNNV3 core 的相关方法背景；再放入完整 EGNN 会混入 Encoder 更新，不作为纯 Embedding 主组 [R11] |
| GemNet | 有向边嵌入与几何消息传播 | 更丰富几何表达的后续候选；具体变体所需角/高阶组合、内存和接口另审 [R12] |
| NequIP / MACE | E(3) 等变张量特征 / 高体阶消息 | 后续 Encoder 研究候选；其势能模型结果不证明导航动作质量，不引入能量监督或预训练势作为本轮默认 [R13–R14] |

**排序是实现优先级，不是预判科学胜负：** E1、E3 先打通化学/距离基线，再实现 E2、E4；正式集合、同剂量和三 seed 规则保持 §9。E5 按父条件启动，E6 不自动晋升。首批不下载外部预训练权重、不增加自监督预训练剂量；今后采用预训练须单独审计数据重叠、元素/电荷/自旋覆盖和迁移成本。

### 6.6 特征可用性与图语义冻结（修复 F2/RBF 歧义）

仓库核查依据：`pes2ts_core/pesn/contracts/key_atom_v3.py` 的 `LEVEL_EDGE_PACK_V3` 仅在 F3 打开 RBF12；`features/key_atom_spec_v3.py` 给出 12 个中心、0.3 Å 宽度；`features/key_atom_graph_v3.py` 的主图为共价并图与当前空间图并集。

1. **F2 保持原样。** E3 从合法当前 `pos` 计算物理 Å 距离，再调用相同数学定义的 Gaussian RBF12（中心均匀覆盖 0–4.5 Å，sigma=0.3 Å）；只存内部视图，不解除传给 core 的 F2 mask。E4 的径向特征同理。禁止从 F2 已置零的 edge 前 12 槽直接读取“距离表示”。坐标若已缩放，必须按冻结转换恢复物理距离；不得把归一化数值当 Å。
2. **区分三个视图。** `G_chem` 是合法 source/target 共价并图和 edit 边去重后的双向图；`G_spatial` 是 legacy 空间邻居选择所得子集；`G_angle` 依 §6.3 使用化学/intent 邻居。E3 不把超 cutoff 的保留共价边误作空间边。空间图的双向化、边界 tie 保留可使实际度数超过 nominal 32，资源估算必须使用实际边数，不能重新按 atom ID 截到 32。
3. 每个有向原子对只产生一条内部边，合并边类型属性；成键/断键的 signed delta 由 reaction intent 决定，**不是**由消息边 i→j / j→i 决定。交换消息方向不翻转 bond delta；反转 intent 时才按合同交换 source/target 并改变相应符号。
4. E1/E2 使用完全一致的化学边特征集合、缩放和缺失规则；不得顺手加入 RDKit 推断的电荷、杂化、额外芳香性或新原子描述符。冻结 `chem_edge_fields`、`chem_edge_dim` 和两种 intent 的手工期望 fixture。
5. 新支路允许的派生几何信息与 E0 不相同，因此本轮比较的是“算子 + 合法视图”的局部 Embedding 配方；不能把 E3/E4 的收益全部归因于网络算子。若需分离信息增量与聚合结构，另立同输入消融，不挤入现有 T4 容量/深度预算。
6. 坐标相关缓存键必须包含当前几何摘要、原子映射、intent/合法拓扑、单位、dtype、recipe/code SHA；仅用 curve ID 或分子 ID 不够。rollout 每步坐标变化都须失效重建，禁止缓存带 autograd 图的张量。

### 6.7 首批算子最小可实现规格（WP0 待签认）

以下是本项目轻量适配的建议值，不冒充原论文默认。公共项：输出 48，1 次消息更新，无 dropout、无 BatchNorm、无全图 pooling、无坐标更新；MLP 默认两层、隐宽 48、ReLU、中间激活而输出线性。GINE 内部消息 ReLU 按其算子定义；所有初始化、偏置和归一化选择写入 spec，不能依赖第三方库易变默认。

- **E1 GINE-chem：** 化学边线性投影到 48；`f_i = MLP((1+epsilon)*h_i + sum_j ReLU(h_j + P_e e_ji))`，`epsilon=0` 固定，self term 显式保留，图不另加 self-loop。孤立节点走 self term。外层统一使用 §5.2 的 W 和 alpha，不在候选外再次偷偷加一层残差。参考 [R6]。
- **E2 PNA-chem：** 消息 `m_ji=MLP([h_i,h_j,P_e e_ji])` 为 48 维；mean/max/min/std 与 identity/amplification/attenuation 拼接后，经含 self feature 的投影/MLP 回到 48。`towers=1`；std 使用总体方差而非无偏估计。零邻居统计全零且保留 self，单邻居 std=0。训练度数参考 `mu_log=mean_inner_train(log(1+d))`；正度数的 amplification 为 `log(1+d)/mu_log`、attenuation 为倒数，d=0 另行显式设 scaler=1；全训练图无边时 typed rejection。数值稳定实现及 eps 必须在 spec 中固定并验证有限梯度，不声称与任一库实现逐位相同 [R7]。
- **E3 SchNet-like：** `m_i=sum_j A(h_j) ⊙ F(RBF12(r_ji))*c(r_ji)`，经 atomwise MLP 得到 f；A 为 48→48 线性，F 为 12→48→48，`c(r)=0.5*(cos(pi*r/4.5)+1)`（r<4.5 Å，否则为 0）。只用距离生成滤波权重；无邻居 m=0。此 cosine envelope 是新分支 spec，不改 legacy core；不能据此声称含离散邻居裁剪的整个模型处处光滑 [R8]。
- **E4 Angle-MPNN：** 不持久保存有向边隐状态；每中心对无序邻居对 {j,k} 使用 `cos(theta_jik)` 的 `[1,c,c²,c³]` 四维低阶基底及两侧 RBF12/合法边属性。令 `a_ij=MLP([h_j,P_e e_ji,RBF12(r_ji)])` 为 48 维，再用 `[h_i,a_ij+a_ik,a_ij⊙a_ik,1,c,c²,c³]` 生成 48 维 triplet 消息并 sum，经输出 MLP 得到 f。这样交换 j/k 严格不变，不能以 atom ID 排序后直接拼接两侧特征。缺角时聚合为零并带显式 valid mask；不学习依赖原子编号的缺角 token。这是 angle-inspired 节点模型，**不是 DimeNet 的球谐/Bessel 有向边模型** [R9]。
- **E5：** 两支路从 h_legacy 分别投影 48→24，使用 E1/E4 的 24 宽版本，拼接后融合 48→48，仍只用一个外部 W 和 alpha。父模型权重不拷贝，分支从独立 generator 初始化；E5 不能额外获得父模型训练剂量。结构在 T1 前一起预注册，运行资格仍由 §9.2 决定。

E2 degree histogram 建议以固定 inner-train 样本索引全集为统计总体，每个计划样本访问一次（含已登记 intent），不用实际 replacement sampler 的重复抽样轨迹；这使归一化可独立复验。若同一拓扑重复出现，按该样本总体计权，记录 unique graph 数和加权总数；只累加真实化学边入度，不加 self-loop、不使用 spatial degree 或标签权重。

### 6.8 需要新增的最小验收样本

| 风险 | 直接验收 |
|---|---|
| F2 RBF 恒零造成伪距离模型 | 两个具有相同 F2 拓扑但不同当前距离的 fixture：legacy F2 edge mask 不变，E3 内部 RBF 改变；用非零测试 gate 检查几何敏感性 |
| 化学重复边 / intent 符号错误 | source/target/edit 重叠边只计一次；双向消息边属性一致，intent 反转遵循合同 |
| PNA 零度 / 单邻居 NaN | d=0/1/多邻居的手算聚合值、有限输出/梯度，与统计文件重算一致 |
| 角度重排泄漏 | 交换 j/k、全原子 permutation、tie shell 与无角 mask；triplet 数及聚合输出一致 |
| zero gate 隐藏几何错误 | alpha=0 做 legacy parity；另在非零 gate 和训练后权重下做完整 E(3)/permutation 测试，不能只测零支路 |
| 批处理或缓存污染 | 同一分子独立/混合 batch 预测一致；同 curve 不同帧及同帧反 intent 均正确失效/复用 |

主模型仍用现有原生 tensor batch。PyG/SchNetPack 是算法参照，不因调研自动成为运行依赖；如采用其算子，应 pin 包版本/源码 commit、scatter 后端、许可证和已知行为，并在项目 fixture 上验证。纯 PyTorch 实现也须记录对应参考和偏离项。

---

## 7. WP4：统一训练器、manifest 与恢复

### 7.1 参数组和 W0/W1

| 参数组 | W0：updates 1–400 | W1：updates 401–37,900 | 学习率 |
|---|---|---|---:|
| legacy 三层 core + legacy MOVE readout + h_legacy | 冻结 | 联合训练 | 1e-4 |
| PROCESS head | 训练 | 联合训练 | 3e-4 |
| 新 Embedding 分支 + W + alpha | 训练；E0 无此组 | 联合训练 | 1e-4 |

这里的 fixed core 指结构和超参固定，不是 W1 继续冻结权重；W1 仍按上表联合更新。

- E0 的 W0 可训练参数、初始化和 loss 路径应与历史完整任务 C 等价。
- 不再用“所有非 backbone 参数都是 head”推断新参数组；应使用显式命名注册，防止新 Embedding 错用 3e-4。
- AdamW、weight decay 1e-4、grad clip 1、FP32 不变。
- `lambda_rank=0.5`、`lambda_q=0.5`、`lambda_endpoint=0.012385049119773344` 和原 MOVE atom/pair/cos/strong/dead 不变。
- 每 100 updates 记录各 loss 项、各参数组梯度范数和 pairwise cosine；主 run 不动态重校准 λ。
- 如确需 embedding-lr probe，只在 train/inner-val 预注册一个统一探针，并将结果应用到所有新分支；不得逐模型调优。E0 复用需另行证明该变化对 E0 是无操作，否则重训 matched E0×3。

### 7.2 数据、采样与选择

- 50K 是 pool，不是每 seed 全部进入梯度更新；沿用每 seed 80%/20% curve-level inner-train/inner-val 和原位置索引。
- inner-val 继续使用固定 seed 的 8,192-sample cap；dev 使用既有冻结全集；confirm 不读。
- batch 固定 8 curves × 4 frames，stratified start/pre-TS/post-TS/endpoint；采样位置由既有 `(seed,update,salt)` 协议决定。
- 同 seed 全候选记录并比较 batch schedule SHA。模型初始化或特征构建不得改变 batch 顺序。
- 每 100 updates 以同一 inner-val total loss 选择 `checkpoint_best`，不得为不同 Embedding 更换选择指标。
- tau 只在 inner-val 以 target precision 0.9 校准并冻结；fallback、support 不足和退化校准应显式标记，不能据此开放 No-Move。

### 7.3 checkpoint / resume 内容

每次 checkpoint 至少包含：model、optimizer、scheduler（若有）、update、best_val/best_update、所有 RNG 状态、sampler 位置或可重建证明、数据/shard 位置、common/embedding/feature/config SHA、git commit、环境和 checkpoint content digest。

恢复验证分两层：

1. fixture 上 20+20 updates 与不中断 40 updates 对比模型、optimizer、batch 顺序和指标；
2. 200-update smoke 中至少做一次 kill/resume，验证从记录 update 继续、无重复/遗漏 batch、manifest 追加而非覆盖。

正式 run 中断后只能从匹配 SHA 的 `checkpoint_last` 恢复；不得用其他 seed、其他 best checkpoint 或人工挑选中间点续跑。

### 7.4 run 状态和失败词表

能力状态沿用 `unknown / implemented / fixture_passed / smoke_passed`。run 结果另分：

- `done`：完成全部 37,900 updates 和规定评估；
- `invalid_run`：泄漏、越界、SHA 不符、非关键原子动作、恢复不一致、运行时错误等协议/实现问题；
- `scientific_failure`：训练合法完成但 H2 等科学门失败；
- `resource_failure`：OOM、swap、host/VRAM 红线；
- `budget_censored`：超过预注册时间/预算；
- `gate_skipped`：父条件未满足，如 E5 未启动。

失败结果保留原始日志和成本；不得删除后换 seed 重试直到得到理想结果。

### 7.5 新分支学习与双任务干扰监控

每 100 updates 的既有日志增加 `alpha`、`RMS(h_legacy)`、`RMS(alpha*W*f)`、二者比值（分母 eps 固定）、分支梯度范数、参数变化范数、实际边/triplet 数和零邻居比例。alpha 单值不能反映残差强度，W/f 的尺度必须同报。W0 结束、W1 初期和 best checkpoint 留存同一 train probe batch 的诊断；probe 不参与验证排名、不读取 dev。

MOVE 与 PROCESS 的梯度范数/余弦在相同共享参数集合、固定 probe 上按各自损失 mask 计算；没有有效监督时记 `null+reason`，不记零冲突。诊断梯度不得写入 optimizer、不重复施加训练更新，成本单列。发生冲突先如实解释，主 run 不临时切 PCGrad、GradNorm、额外 warmup、alpha clamp 或 loss reweighting。

分支启动验收需区分“初步梯度为零”和“没有进入学习路径”：zero gate 首步 f/W 的任务梯度为零是数学预期；alpha 更新后须验证至少一个含有效监督的 probe 上分支获得有限非零任务梯度。仅有 AdamW decay 引起的参数变化不算学习证据。W0→W1 切换应检查 core 已入优化器且梯度可达；冻结参数不等于对 core 前向使用 `no_grad()`，后者会切断新 Embedding 到损失的梯度。

---

## 8. WP5：E0 可比性裁决、技术冒烟与预算

### 8.1 E0 复用审计清单

| 审计域 | 通过条件 | 失败分支 |
|---|---|---|
| 资产身份 | C×3 checkpoint、metrics、run record、R3 init、plan/pin 和六类 provenance SHA 可复验 | 缺失则历史资产不能作 matched E0 |
| forward parity | WP2 对三 checkpoint strict-load 和输出等价通过 | 先修包装；不能进入正式比较 |
| 数据/标签/split | 50K pool、每 seed 位置、dev、teacher、label spec 和 SHA 一致 | 训练合同不可比，重训 E0×3 |
| sampler/RNG | 8×4、分层规则、batch schedule、pair RNG 和初始化顺序等价 | 重训 E0×3 |
| optimizer/参数组 | W0/W1 可训练参数、lr、wd、clip、λ、pos_weight 规则一致 | 重训 E0×3 |
| schedule/checkpoint | 400+37,500、val_every=100、inner-val cap 8,192、best 规则和 resume 语义一致 | 重训 E0×3 |
| loader 训练路径 | sharded/columnar loader 的训练取值、label/cache、prefetch 不改变样本或顺序；所需等价证书闭合 | 证书缺失/SKIP 时复用不通过；先做一轮有界补证，仍不能闭合则重训 E0×3 |
| 修复后重评 | WP1 完成且分母/指标可审 | 只影响评估时必须重评；训练资产是否复用仍由上述项决定 |

**补证优先规则：** 对当前已知的 loader-v2 证书 SKIP，先执行一轮 `WP5-CERT` 流式/分片等价审计，避免同时驻留 v1 dataset 与 label cache。补证沿用 12 GiB host 限制并记录 CPU/GPU/墙钟成本；只允许一轮预注册补证，不反复改变证据定义。补证通过才可复用；发现真实不等价、资源内无法完成或证据仍缺失时，由 `e0_reuse_decision.json` 明确授权 E0×3 重训。补证期间 E0 复用状态保持 `unknown`，不得先按复用启动 T1。

**裁决规则：**

- 全部训练相关项通过：复用历史 C×3 权重和重评指标，T0 新训练数为 0。
- 任一训练相关项失败或证据不足：用 V541 统一 runner 重训 E0 seeds 17/29/43；修复后的历史 C×3 只保留为 auxiliary historical reference。
- 仅评估器改变：不要求重训；所有模型统一走修复后的评估器。
- 不用 100–200 updates 的指标好坏决定复用，也不用 smoke 排名架构。

### 8.2 技术 smoke

每个 E0–E4 和条件 E5 在同一 50K pool 上完成。E0 只需 seed17 的技术 smoke，产物写入 `smoke/V541_E0_S17/`；它是 runner/预算验证，不是 T0-R 正式 run，不计入核心训练次数或正式 attempt，也不进入排名/Pareto。E0 复用与重训分支都应保留这一技术证据：

1. CPU 形状、非法输入、E(3)、permutation、leak 和 fixture 测试；
2. GPU 32 samples forward/backward；
3. 200 个 batch updates，正式 batch 形状 8×4；
4. 一次中断/恢复；
5. 记录 `ms/update`、GPU active/wait、CPU feature/triplet 时间、eval 时间、peak VRAM、peak host anon/RSS、cache hit、shard/page-cache 行为；
6. 检查 alpha/分支学习启动、loss/grad 有限、非关键原子无动作；
7. 12 GiB cgroup、`MemoryHigh=10G`、swap=0、OOM kill=0、单路 concurrency=1。

### 8.3 W0/W1 分别测速，禁止错误外推

200 个 W0 updates 不能直接代表 37,500 个 W1 joint updates。技术 profiler 应在明确标记的 `profile_only` 窗口分别测量 frozen-core W0 和 unfrozen-core W1；profile 不改变正式 run 的 400+37,500 schedule，也不用于科学排名。

预算模型为：

\[
T_{run}=400\bar t_{W0}+37{,}500\bar t_{W1}+T_{startup}+T_{eval}+T_{checkpoint}.
\]

同时报 GPU active、CPU data preparation、host 缓存和 wall-clock，不以单一 `ms/update` 隐藏 CPU 角度构建瓶颈。每候选形成：

- `projected_train_gpu_hours`；
- `projected_wall_hours`；
- `fixed_37900_update_budget_seconds`；
- `fixed_gpu_time_comparison_budget`；
- `peak_vram` / `peak_host_anon`；
- 是否批准进入正式训练。

E0 成本分两类报告，禁止混写：复用时，`historical_accounting_cost` 取历史 C×3 实测训练成本并保留其代码/硬件 provenance，新重评成本另列；`matched_runtime_profile` 则在当前环境用同一 profile_only 协议测 E0 的 W0/W1，供跨候选 core-time tie-break 和成本比计算，明确标为 profile estimate。E0 重训时可用当前正式 run 实测替代 estimate。E0 技术 smoke 本身不进科学排名。

正式训练始终固定 37,900 updates；fixed-GPU-time 只作另报的成本匹配分析，不替代同剂量主比较。

### 8.4 资源失败处理

- 达到红线即停止并记 `resource_failure` / `budget_censored`。
- 不得通过减少 frames/curves、缩短 updates、改变 stratification、删除 PROCESS 损失或更换 seed 来“通过”。
- 可先做数学等价的 triplet 分块、lazy materialization、缓存或更轻候选内部结构；新实现重新过 fixture/smoke 和 SHA 冻结。
- 若轻量重设计改变科学结构，应使用新 `embedding_spec` 并保留原失败记录。

---

## 9. WP6：正式运行矩阵与选择流程

### 9.1 Nominal run matrix

| 阶段 | run | seeds | 新完整训练数 | 启动条件 |
|---|---|---|---:|---|
| T0 | matched E0 历史 C×3 | 17/29/43 | 0 | E0 全部复用审计通过，WP1 重评完成 |
| T0-R | V541 matched E0 | 17/29/43 | +3 | 任一训练可比性失败或证据不足 |
| T1 | E1/E2/E3/E4 | 17 | 4（满额） | H0、smoke、预算和 runplan 全过 |
| T2 | E5 | 17 | 0–1 | E1 与 E4 均完成，H0/H1/H2 通过且资源获批 |
| T3 | 两个冻结候选 | 29/43 | 2×n_selected（满额 4） | T1/T2 inner-val 初筛完成；每个补两 seed |
| T4 | 容量/传播深度诊断 | 预注册，默认 17 | 0–2 | T1 前冻结对照策略/参数匹配规则；实际配置在 T4 前、任何 candidate dev 前 pin |
| T5 | dev/offline/no-physics rollout | 冻结 checkpoints | 0（新增 QC 也为 0） | 候选、tau、阈值、名单和统计代码冻结 |

- 核心新训练数按 `N_core = n_T1 + I(E5) + 2*n_selected + 3*I(E0_retrain)` 计算，其中满额 `n_T1=4`，`n_selected∈{0,1,2}`。满额 nominal（`n_T1=4,n_selected=2`）为 **8–9 次**；E0 重训时满额为 **11–12 次**。T4 的 0–2 次诊断另列，不混入核心数。
- 若 E5 进入最终两个候选，T3 仍总共只补 4 次。
- 候选在 WP5 提前失败时不进入 T1，`n_T1` 按实际获准集合缩减；合格候选不足两个时也不强行凑数，按上式缩减 T3 并记录 `gate_skipped`，E0 保留。已启动 run 的 attempt/资源成本另账保留，不等于完整训练次数。gate-truncated 的实际次数可以低于满额 8–9，不得再写成固定次数。
- 任何候选不得减少任一 MOVE/PROCESS 任务或训练剂量。

### 9.2 E5 启动条件

E5 是条件候选，不是默认必跑：

1. E1、E4 的 seed17 正式 run 均完成；
2. 两者 H0/H1/H2 均通过；
3. 双支路参数/FLOPs/内存经过 profiling，单路预算另获批准；
4. E5 spec 在 T1 前预注册为 24+24→48（§6.7），E5 smoke/正式训练前再核验 SHA，且不得晚于任何 WP7 `candidate_evaluation` dev 访问；WP1 已密封的 baseline-repair 不影响这一时序；
5. 若任一父候选失败，E5 记 `gate_skipped`，不得通过改父模型任务或预算绕过。

### 9.3 单 seed 初筛算法（只看 inner-val）

T1/T2 的 H2、provisional H3、Pareto 和 top-2 选择全部使用 inner-val；dev 不参与这一阶段的门禁或排序。

1. 计算每个候选相对同 seed matched E0 的 `ΔEndpoint AUPRC`、`Δq corr`、q MAE、MOVE guards 和成本。
2. 先过滤 H0/H1/H2 和预注册 core-time 成本比不合格者；成本比超标保留科学结果但不晋级。单 seed 的 H3 只称 `provisional_H3`，不是正式科学通过。
3. 按 `(ΔAUPRC, Δq corr)` 最大化构造 Pareto front；前沿不足两名时按下一前沿递补。
4. 前沿内依次按：满足的 H3 条件数、MOVE 方向 guard 余量、core-time 较低、最后按 embedding_id 字典序，保证选择可重现。
5. 冻结两个候选及其 best-checkpoint 选择状态后才启动 seeds 29/43；不得根据后续 seed 结果替换初筛规则。

### 9.4 T4 容量/深度诊断

默认最多两项。T1 前冻结“选哪两项、如何匹配参数/深度”的规则；inner-val champion 产生后，只按该规则实例化实际配置，并在 T4 启动和任何 candidate dev 访问前 pin，不得根据诊断结果改宽度：

- **E0-wide**：匹配赢家相对 E0-primary 的新增参数量，目标 `abs(delta_params_control-delta_params_champion)/max(delta_params_champion,1)≤5%`；离散宽度无法落入 5% 时取确定性的最近配置并披露实际偏差。此时不得再宣称 V5.3-compatible core。
- **E0-deeper**：精确匹配 Embedding 内部消息传播 + 三层 core 的总深度，检验深度混杂；明确标为 Encoder-depth diagnostic。
- **E0+MLP**：本轮建议优先替换 E0-wide，与 E0-deeper 组成两项对照；使用相同 h_legacy→残差 W/alpha 接口、初始化和参数组，但不聚合邻居。沿用同一新增参数匹配规则，用于区分非线性加宽与邻居消息传递；若未执行，报告必须明确该混杂未闭合。

T4 不参与 Embedding-only 赢家竞争。若诊断组达到相似收益，应把结论降级为“尚不能排除容量/深度解释”，不得把诊断组静默升级为 V5.4.1 赢家。若已确定 `E*=E0` 且没有待解释的新 Embedding 增益，T4 可为 0 并记 `gate_skipped(reason=no_new_embedding_gain)`，不为凑矩阵增加训练。

### 9.5 三 seed 后的 inner-val 赢家

1. 每个 run 都完成规定 updates，且 H0/H1/H2 通过；失败数按 seed 全量列出。
2. 正式 H3 要求三 seed 聚合的至少一个 PROCESS 主指标达到预注册效应门，且该指标改善方向至少 2/3 seeds 一致，另一主指标满足非劣要求。
3. 先看 H3，再按 AUPRC/q corr Pareto、MOVE 方向、core-time 和 WP0 冻结的成本接受规则决定 `inner_val_champion`。
4. 若未通过 H3，或按预注册成本规则不值得升级，直接确定 `E*=E0`；paired CI 用于解释证据强弱，不在看到结果后新增临时门。
5. 只有三个 seeds，不作 seed 级 `p<0.05` 声明。

---

## 10. WP7：冻结评估、统计、rollout 与交付

### 10.1 评估顺序

1. 冻结 matched E0、两个候选的三 seed checkpoints、inner-val champion、tau、H 门、统计代码和 runplan SHA。
2. 核对 dev 访问账本：`baseline_repair` 只能包含 WP1 的 E0/R3 重评，`candidate_evaluation` 只能从本步骤开始。此时才解封 WP1 的 dev 基线详细指标，并核验其输出 SHA 未变。
3. 在 dev 上只评估冻结集合；dev 不再用于挑 checkpoint、改阈值或重排 GNN。
4. dev 可以否决一次升级。WP0 建议冻结的否决条件为：H0/H2 任一失败，或 dev 三 seed 聚合未达到与 inner-val 相同的 H3 效应门，或另一 PROCESS 主指标突破非劣界；触发后记录 `development_not_reproduced`，下游保守回退 E0。
5. dev 否决后不得改选 runner-up；若需继续研究，应另开预注册实验，不在本轮反复读 dev。
6. confirm 继续封存。任何 confirm/OOD 申请都应在本阶段报告和 bundle 冻结后单独审批。

### 10.2 离线指标与分母

| 维度 | 必报内容 |
|---|---|
| MOVE | curve median RMSE/zero、有效 curve/position 分母 |
| MOVE 方向 | 修复后的 key-pair sign、active pair 分母、signed bias、fixed-coverage direction/cosine |
| MOVE 覆盖 | dead fraction、strong-action direction、动作幅度、coverage，防止靠缩小动作骗低 RMSE |
| q | curve-mean MAE、Pearson corr、常数/拓扑基线（若已冻结） |
| ranking | overall/cross-TS ordered-pair accuracy，按 curve 聚合，正/反 intent 分列 |
| endpoint | AUPRC、AUROC、prevalence、Brier/ECE、有效正负例分母 |
| 控制 | inner-val 冻结 tau 后的 start/TS misstop、endpoint recall、support/fallback 状态 |
| 计算 | parameters、FLOPs（可测范围内）、train/eval GPU 与 wall time、CPU feature 时间、VRAM/anon、latency |
| 分层 | TS 前/附近/后/端点、正反 intent、成/断键、K 数、分子/邻居规模、反应 family |

所有比例同时给分子和分母；masked、terminal、conflict、invalid、resource failure 不得从无说明的分母中消失。

### 10.3 配对统计

- 以 curve 为 bootstrap 单位，同一 curve 的所有位置、intent 和相关 seed 结果成组重采样。WP0 同时核验 canonical reaction、逆反应与映射副本的 split 隔离；若多 curve 属于同一 canonical reaction，主 CI 改用该组为 cluster 并成组采样，不能把相关曲线当独立样本。此判断只依据元数据，在结果可见前冻结；分组与跨 seed 支持集交集/缺失规则写入统计 spec。
- 初筛可用 1,000 次；最终 paired CI 使用 10,000 次，bootstrap seed 在 runplan 固定。
- 主对比为同 seed 候选减 matched E0，再报告三 seed 的逐 seed 效应、均值、范围和 percentile 95% CI。
- CI 解释限于当前冻结评估语料的 curve 抽样，不声称解决 seed 抽样或新反应族泛化。
- 失败、删失和资源失败单列，不只对成功 run 做平均。

### 10.4 Rollout

- 本阶段 rollout 固定为 **no-physics / pure-geometry**。`reference_normalized` 与 `autonomous` 都不调用 QC；`autonomous` 只表示固定步数预算下由模型自主展开，不表示 physics-in-the-loop。新增 `N_energy/N_gradient/N_Hessian/N_OptTS_trials` 均为 0；历史数据制备成本只引用 provenance，不重复计作本阶段新增 QC。
- 两个子集均来自冻结 **dev**，不含 50K pool/inner-val/confirm；采用 `24-curve regression ⊂ 96-curve fixed rollout ⊂ dev`。若已有合规名单则复核复用，否则只用 curve-ID hash 和不读标签的预注册分层生成，保存名单、来源 split SHA 和生成规则 SHA。WP7 前不得生成 E1–E6 候选或冻结集合在这些子集上的预测；WP1 已密封的 E0/R3 baseline-repair 预测是唯一例外，且仍不得用于候选设计。
- `reference_normalized` 与 `autonomous` 严格分开，预算、终止和删失定义分别冻结。
- 报端点到达、TS 跨越、碰撞/异常、预算耗尽、错误 No-Move、非关键原子动作和全部尝试成本。
- 只使用当前合法 forward 信息驱动控制；未执行 OptTS/Freq/双向 IRC/端点身份时，只能写 `TS_candidate` / `endpoint_reached`。
- 50K 自主闭环未完成的历史状态不能因离线 rollout 而改写；本阶段不自动开放生产 No-Move。

### 10.5 计算—科学 Pareto

至少给出两张视图：

1. **fixed-dose**：同为 37,900 updates，比较科学指标与实际成本；
2. **fixed-GPU-time**：比较相同计算预算下的学习曲线/效果，仅作部署经济性补充。

若新 Embedding 未通过 H3，或在 WP0 预注册的成本接受规则下不值得升级，E0 留任；不得只按 AUPRC 单指标决定赢家，也不得看到结果后新设成本阈值。

---

## 11. 预注册硬门与签认单

### 11.1 硬门规则（WP0 签认后冻结）

H2 的 2% MOVE guard 承继现有合同；H3 的数值是上游 v1.1 的建议候选。下表在 WP0 签认前不构成训练授权，签认后不得按 seed17/dev 结果修改。

| 门 | 规则 | 结果处理 |
|---|---|---|
| H0 legality | schema/no-leak/confirm 隔离/restore/E(3)/permutation/非关键原子动作全部通过 | 协议或实现问题记 `invalid_run`；科学不适用不得伪装成 invalid |
| H1 resource | 单路、无 swap/OOM、host/VRAM/预算不超线 | `resource_failure` 或 `budget_censored` |
| H2 MOVE | 每个正式 seed 相对修复口径 R3 参照的既有 RMSE/zero ≤2% guard，且相对同 seed matched E0 的 RMSE/zero 退化 ≤2%；sign/dead/fixed-coverage 方向按签认容差执行 | 候选失去晋级资格，结果保留为 `scientific_failure` |
| H3 PROCESS | `ΔAUPRC ≥ +0.02 absolute` 或 `Δq corr ≥ +0.03`，正式结论另需 ≥2/3 seeds 同方向，另一主指标非劣 | 不满足则无新 Embedding 正式赢家 |

H2 的 R3 guard 只使用 WP1 同管线重评后的参照；R3 无法重评时阻断 T1，不得降级为仅 matched-E0 参照。matched-E0 比较采用同 seed 配对，并另报三 seed 聚合；每个 seed 均须满足 hard guard，不得只用均值掩盖单 seed 严重坍缩。

### 11.2 T1 前需签认的建议容差

上游把以下项称为预注册候选但未给完整数值。本方案建议冻结为：

| 参数 | 建议值 | 用途 |
|---|---:|---|
| `sign_drop_abs_max` | 0.01 | matched E0 的 key-pair sign 绝对下降上限 |
| `dead_fraction_increase_abs_max` | 0.01 | dead fraction 绝对增加上限 |
| `fixed_coverage_direction_regression_rel_max_pct` | 2.0% | 固定覆盖率下方向指标相对退化上限 |
| `other_endpoint_auprc_drop_abs_max` | 0.01 | 以 q corr 申请 H3 时，AUPRC 非劣界 |
| `other_q_corr_drop_abs_max` | 0.02 | 以 AUPRC 申请 H3 时，q corr 非劣界 |
| `gpu_parity_atol` / `gpu_parity_rtol` | 1e-6 / 1e-5 | FP32 GPU forward parity 建议容差；失败先查实现，不得事后放宽 |
| `max_core_time_ratio` | 2.0 | 相对 matched E0 的同环境 fixed-dose core-time 升级上限建议值 |
| `final_bootstrap_draws` | 10,000 | 最终 curve-paired CI |
| `rollout_large_subset_curves` | 96 | dev 固定较大 rollout 子集 |

这些数值在 WP0 签认前仍是 **proposed**；未签认不得启动 T1。签认后写入 runplan 和 config SHA，不能根据 seed17 结果调整。

本方案建议默认采用 `hard_ratio` 成本接受规则：`candidate_core_time / matched_E0_core_time ≤ max_core_time_ratio`。复用 E0 时使用 §8.3 的同环境 matched profile 并标注 estimate，重训 E0 时使用当前正式实测；超过比率不是 `resource_failure`，而是科学结果保留但不授予升级、E0 留任。PNA 统计、角度构建、评估和缓存预热的单次成本另列并进入总成本 Pareto，不得通过只报 core-time 隐去。WP0 若不采用 2.0，应在任何 T1 科学结果可见前重新履行签认，改成另一明确值，并同步更新 `preregistration_signoff.json`、runplan/config SHA 与 `budget_policy_sha256`；不得只改文档或单个 run 的配置。

### 11.3 Stretch，不作为失败门

- Endpoint AUPRC ≥0.35；
- q corr ≥0.45；
- Move RMSE/zero 不劣于修复后 matched E0；历史 0.8912 仅作来源描述，不作跨评估器口径的正式阈值；
- dead-zone 不高于 matched E0；
- rollout 稳定性优于 E0。

Stretch 未达不等于研究失败；达到 stretch 也不能豁免 H0/H1/H2/H3。

---

## 12. 产物、账本与可追溯性

### 12.1 目录

```text
outputs/pesn/v541_embed_50k/
  plan/
    runplan.jsonl
    runplan.pin.json
    preregistration_signoff.json
    budget_policy.json
  audits/
    evaluator_before_after.json
    dev_baseline_sealed/       # WP1 密封，WP7 核验 SHA 后解封
    dev_access_ledger.jsonl
    v53_parity.json
    e0_reuse_decision.json
    geometry_invariance.json
    input_leak.json
    resume_equivalence.json
  smoke/
    V541_E0_S17/...            # 技术 smoke；非正式 run/attempt
    V541_E1_S17/{metrics.json,resources.json,train_log.jsonl}
  runs/
    V541_E0_S17/...        # 仅 T0-R 分支
    V541_E1_S17/...
    V541_E2_S17/...
    V541_E3_S17/...
    V541_E4_S17/...
    V541_E5_S17/...        # 条件
  diagnostics/
    V541_E0_WIDE_S17/...
    V541_E0_DEEPER_S17/...
  reports/
    single_seed_screen.json
    three_seed_paired_ci.json
    compute_pareto.csv
    dev_frozen_evaluation.json
    rollout_comparison.json
    failure_and_censoring.csv
  exports/
    embedding_champion_bundle/
      embedding_config.yaml
      embedding_weights.pt
      feature_schema.json
      provenance.json
      checksums.json
      benchmark_summary.json
      model_card.md
      downstream_handoff.json
```

### 12.2 每个 run 的必填字段

`run_id`、`embedding_id`、`objective`、`seed`、`status/attempt`、`git_commit`、Python/Torch/CUDA、data/teacher/split/label SHA、common/embedding/feature/config SHA、R3 init SHA、实际 updates、best update、batch schedule SHA、参数/FLOPs、各阶段计时、cache hit、VRAM/host 峰值、checkpoint SHA、tau 状态、失败/删失原因。

所有 attempt 追加写入总账；重试必须关联原 attempt 和原因，不得覆盖。

### 12.3 赢家 bundle

- `embedding_weights.pt` 包含 LegacyEmbedding 所需部分、候选分支、W 和 alpha；E0 胜出也按同一接口导出。
- `feature_schema.json` 固定节点 48、legacy edge 22、内部 chem/spatial/angle recipe、tie/mask/eps 和缓存 fingerprint。
- `provenance.json` 指向完整 V541 checkpoint 及其 SHA，避免只交付 Embedding 后丢失联合训练 core/head 的对应关系。
- `downstream_handoff.json` 明确 `frozen-E*` 默认只读、`requires_grad=False`、strict-load 入口，以及 `joint-finetune-E*` 必须另立实验条件。
- `model_card.md` 写明 common-core warmstart、三 seed、dev exposure、未读 confirm、资源成本、失败/删失、未闭合容量诊断和“不证明完整原论文模型/严格 TS/部署资格”。

另保留完整 V541 champion checkpoint 或不可变引用，供重现 MOVE/PROCESS 输出；仅有 Embedding 权重不构成完整模型复现包。

---

## 13. 验证证据矩阵

| 需要建立的声明 | 最小直接证据 | 责任角色 | 失效时需重做条件 |
|---|---|---|---|
| E0 与旧模型一致 | 三 checkpoint strict-load + 五输出 parity | 模型负责人 / 独立验证人 | core/readout/wrapper、权重、精度或输入构建变化 |
| pair 评估无 silent drop | 独立 pair fixture + before/after 分母 | 评估负责人 | collate、pair schema、评估代码变化 |
| forward 无泄漏 | metadata/teacher 干预 + AST/schema 白名单 | 数据与模型负责人 | forward keys、provider、batch 构造变化 |
| 几何合同成立 | 平移、旋转、反射、permutation、退化几何测试 | 模型负责人 | 图/角度/坐标或消息算子变化 |
| 新分支确实学习 | alpha 与分支梯度日志、200-update smoke | 训练负责人 | 初始化、参数组、loss 或 schedule 变化 |
| 恢复不改变自己的训练轨迹 | 20+20 vs 40 + kill/resume batch/SHA | 训练负责人 | sampler、checkpoint、loader/prefetch 变化 |
| E0 可复用 | §8.1 全项机器审计 | 训练与数据负责人共同签认 | 任一共同训练合同项变化 |
| 科学改善不是单 seed 偶然 | 三 seed + 同 curve paired bootstrap + 失败全账 | 评估负责人 | checkpoint、评估集、指标或统计代码变化 |
| 成本可接受 | W0/W1 分项 profiling + 正式全量成本 | 资源负责人 | batch/core/feature/硬件或环境变化 |
| 下游可 strict-load | 干净进程加载 bundle + 小 batch 前向 + 只读检查 | 发布负责人 | bundle schema、feature recipe 或权重变化 |

已有测试通过不能替代真实 50K smoke；smoke 通过不能替代正式三 seed 科学证据。

---

## 14. 执行入口：现有与拟实现严格区分

### 14.1 仓库已有入口，可在各自合同和授权范围内使用

```bash
python scripts/agent_ops/docs_consistency.py --strict
python scripts/campaigns/r3_v53mp_run.py --help
python scripts/campaigns/r3_v53mp_scale_paired_ci.py --help
python scripts/agent_ops/feature_gate/v53mp_rollout_exec.py --help
```

这些入口不证明支持 V5.4.1；尤其不得把 V5.3 的 `run --tier 50k` 当作新 Embedding runner。

### 14.2 V5.4.1 拟实现流程，当前不可直接执行

```text
v541 validate-config --config config/campaigns/pes2ts_v541_embed_50k.yaml
v541 gate audit-data-and-manifests
v541 gate fix-and-retest-pair-evaluator
v541 gate legacy-E0-forward-parity
v541 plan create --pool 50k --embeddings E0,E1,E2,E3,E4 --primary-seed 17
v541 smoke run --all --updates 200 --single-lane
v541 profile run --all --measure-w0-w1 --single-lane
v541 baseline audit-and-reuse --source v53mp_50k_C --seeds 17,29,43
# 仅训练可比性失败时：
v541 train E0 --seeds 17,29,43 --full-schedule
v541 train E1,E2,E3,E4 --seed 17 --full-schedule
# 仅父条件满足时：
v541 train E5 --seed 17 --full-schedule
v541 select --source inner-val --topk 2 --policy frozen
v541 train selected --seeds 29,43 --full-schedule
v541 diagnose capacity-depth --policy preregistered
v541 evaluate --frozen-dev --curve-paired-bootstrap --draws 10000
v541 rollout --frozen-subset --no-physics --autonomous-and-reference-separated
v541 freeze --embedding-winner-or-E0 --seal-v542-interface
```

实现后每一步仍需前一门通过和显式授权；文档中的命令名不构成现有能力声明。

---

## 15. 失败分支与停止规则

| 情形 | 动作 | 禁止动作 |
|---|---|---|
| E0 forward parity 失败 | 停止候选训练，修包装/映射 | 用重训掩盖不等价 |
| E0 训练等价证据缺失 | 补证书或重训 E0×3 | 把 SKIP 当 pass |
| pair 分母仍不一致 | 阻断 sign/方向结论 | 继续引用旧 0.8546 |
| alpha/分支不学习 | 在 smoke 修初始化/参数组 | 进入完整 50K 后再发现零分支 |
| OOM/swap/host 红线 | 记资源失败，做等价分块或轻量重设计 | 改 batch、updates、任务或 seed |
| H2 失败 | 保留 `scientific_failure`，候选淘汰 | 用 PROCESS 增益豁免 MOVE 坍缩 |
| E1/E4 父条件不满足 | E5 `gate_skipped` | 无条件下放 E5 预算 |
| 三 seed 无 H3 | `E*=E0`，完成负结果和成本报告 | 强行选“最不坏”的新 GNN |
| dev 未复现关键效应 | 标记未复现并回退 E0 交付 | 在 dev 上改选、调参、调 tau |
| T4 显示容量/深度可解释 | 收窄归因，必要时另开对照 | 把 diagnostic 静默升级为赢家 |
| confirm 被访问或发生 split 污染 | 相关 run/结论 `invalid_run`，启动治理审查 | 删除记录后继续声称封存 |

---

## 16. 最终出口与 V5.4.2 交接清单

### 16.1 V5.4.1 出口

- [ ] WP0–WP5 全部硬证据完成，E0 复用/重训决策无歧义；
- [ ] 主运行矩阵或提前终止原因完整，所有正式 run 为统一双功能目标；
- [ ] 三 seed、curve-paired CI、H0/H1/H2/H3 和成本报告完成；
- [ ] T4 已执行、明确声明未闭合的容量/深度混杂，或因 `no_new_embedding_gain` 合法记为 `gate_skipped`；
- [ ] dev 只按冻结协议使用，rollout 双口径分列，confirm 封存；
- [ ] 唯一 `E*` 决策及理由可机器重放；无可靠改善时为 E0；
- [ ] bundle、完整 checkpoint 引用、feature recipe、checksums 和 model card 可 strict-load；
- [ ] 文档状态按真实证据更新，不把 `implemented` / `fixture_passed` / `smoke_passed` 混为科学成功。

### 16.2 给 V5.4.2 的固定输入

1. `embedding_id`、内部超参、权重、初始化模式、feature recipe 和 SHA；
2. `EmbeddingOutput(node_scalar[N,48], edge_scalar[E,22], node_vector optional)`，首批主通道只冻结 node_scalar；
3. E0+旧 Encoder 与 E*+旧 Encoder 两个必要对照的身份；
4. `frozen-E*` 与 `joint-finetune-E*` 的显式区分；
5. 原 MOVE/PROCESS 合同、输入白名单和 No-Move 化学语义不变；
6. 下游表现不佳时不得悄悄修改 E*；需要修改则另开新 Embedding 版本和实验。

---

## 17. 启动前最终签认清单

只有以下项目全部勾选，T1 才可授权：

- [ ] V5.4.1 ADR/registry/config/schema 与文档门完成；
- [ ] WP5-CERT 有界补证完成或记录不能闭合的原因，matched E0 的复用或 +3 重训分支已由审计 JSON 决定；
- [ ] pair evaluator 修复、C×3 inner-val/dev before/after 和 R3 H2 参照重评完成；dev 详细指标及访问账本保持密封；
- [ ] E1–E5 spec、E3 内部 RBF 重算与 F2 mask 隔离、E4 无序邻居对对称性/角度来源/tie policy、E2 degree 统计与零度行为冻结；
- [ ] 非零 gate 几何测试、当前几何缓存键、W0 梯度可达性与残差强度日志通过；canonical reaction split/统计 cluster 审计完成；
- [ ] H2 方向/dead 容差、H3 非劣界、GPU parity 容差、Pareto/tie-break、成本接受规则和 dev 否决规则签认；
- [ ] T4 的 0–2 项选择策略与参数/深度匹配规则已在 T1 前冻结；实际配置只允许随后按规则实例化并在 T4/dev 前 pin；
- [ ] 24⊂96⊂dev 的 rollout 名单或确定性生成规则、来源 split SHA 已冻结；
- [ ] W0/W1 分项成本模型和每候选预算签认；
- [ ] formal-run code/common-protocol/runplan SHA 冻结；
- [ ] confirm=false、单路、无 swap、失败留账和禁止换 seed 的执行守卫生效。

**授权结论默认是 `not_yet_authorized`；本方案的落盘不等于批准启动 50K 正式训练。**


---

## 18. 文献与实现来源（2026-10-27 核验）

只将以下来源用于方法依据；不把原文数据集成绩作为本项目阈值，不宣称覆盖截至检索日的全部最新模型。线上 `latest` 文档用于本次调研，实施冻结时必须另 pin 版本。

- **R1 GCN**：Kipf & Welling, ICLR 2017，[原论文](https://arxiv.org/abs/1609.02907)。
- **R2 GAT**：Veličković et al., ICLR 2018，[原论文](https://arxiv.org/abs/1710.10903)。
- **R3 MPNN**：Gilmer et al., ICML 2017，[Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212)。
- **R4 D-MPNN**：Yang et al., JCIM 2019，[Analyzing Learned Molecular Representations for Property Prediction](https://arxiv.org/abs/1904.01561)。
- **R5 GIN**：Xu et al., ICLR 2019，[How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826)。
- **R6 GINE**：Hu et al., ICLR 2020，[Strategies for Pre-training Graph Neural Networks](https://arxiv.org/abs/1905.12265)；[PyG GINEConv 算子定义](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html)。
- **R7 PNA**：Corso et al., NeurIPS 2020，[原论文](https://arxiv.org/abs/2004.05718)；[PyG PNAConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.PNAConv.html)。
- **R8 SchNet**：Schütt et al., NeurIPS 2017，[原论文](https://arxiv.org/abs/1706.08566)；[作者团队 SchNetPack](https://github.com/atomistic-machine-learning/schnetpack)。
- **R9 DimeNet / DimeNet++**：Gasteiger et al., ICLR 2020 / NeurIPS Workshop 2020，[DimeNet 原论文](https://arxiv.org/abs/2003.03123)；[作者仓库及两种模型说明](https://github.com/gasteigerjo/dimenet)。
- **R10 PaiNN**：Schütt et al., ICML 2021，[Equivariant message passing for the prediction of tensorial properties and molecular spectra](https://arxiv.org/abs/2102.03150)；实现参照 SchNetPack（R8）。
- **R11 EGNN**：Satorras et al., ICML 2021，[E(n) Equivariant Graph Neural Networks](https://arxiv.org/abs/2102.09844)。
- **R12 GemNet**：Gasteiger et al., NeurIPS 2021，[Universal Directional Graph Neural Networks for Molecules](https://arxiv.org/abs/2106.08903)。
- **R13 NequIP**：Batzner et al., Nature Communications 2022，[原论文预印本](https://arxiv.org/abs/2101.03164)；[作者仓库](https://github.com/mir-group/nequip)。
- **R14 MACE**：Batatia et al., NeurIPS 2022，[Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields](https://arxiv.org/abs/2206.07697)。

### 18.1 本次修订记录

v1.0→v1.1：补充分子 GNN 选型依据、F2/RBF 和图语义、候选最小算子规格、非零 gate 几何验收、学习启动及任务干扰监控；建议 T4 优先 MLP/深度对照，并补 canonical reaction 聚类统计边界。保持 50K、三层 core、双功能头、37,900 updates、E0–E6 身份、核心训练数公式和 confirm 封存。此为设计细化；尚未实现、训练或测得候选收益。

---

## 19. 执行指针（2026-10-29 增量登记；§0–§18 规划正文未改写）

> 执行结果的唯一权威叙述是只读最终报告：[V5.4.1 Embedding 50K 实施与训练报告 20261029](../reports/PES2TS_V541_Embedding_50K_实施与训练报告_20261029.md)。本节只做状态映射与证据指针，不复述、不改写规划条款。

### 19.1 阶段状态映射

| 方案阶段 | 执行状态 | 证据 |
|---|---|---|
| WP0 治理/预注册 | done | ADR-0032、`config/campaigns/pes2ts_v541_embed_50k.yaml`、`plan/preregistration_signoff.json`、`plan/runplan.pin.json` |
| WP1 评估器修复 + 历史 C×3 重评 | done | `audits/evaluator_before_after.json`、`audits/dev_baseline_sealed/`、`audits/verify_report.json` |
| WP2 V541Model + E0 parity | done | `audits/v53_parity{,_gpu,_gpu_bitwise_probe}.json`（CPU bitwise + GPU 1e-6/1e-5） |
| WP3 候选实现 | done（E1–E5 代码落地；E6 未创建，按 §5.1 不进默认批次） | `pes2ts_core/pesn/models/v541/` |
| WP4 统一 runner/manifest/resume | done | `pes2ts_core/pesn/offline/training/v541_*.py`、`audits/resume_equivalence.json` |
| WP5 E0 裁决 + smoke + 成本 | done | `audits/e0_reuse_decision.json`（retrain_e0_x3）、`audits/h0_legality.json`、`audits/h1_resource.json`、`profile/` |
| WP6 正式矩阵 | done（提前终止合法） | T0-R E0×3 + E1/E2/E3 seed17 全部 37,900 updates；E4 成本门 skipped；`reports/failure_and_censoring.csv` |
| WP7 冻结评估/rollout/bundle | done | `reports/{dev_frozen_evaluation,rollout_comparison,wp7_freeze}.json`、`exports/embedding_champion_bundle/` |

### 19.2 §16.1 出口逐项状态

- [x] WP0–WP5 硬证据完成；E0 决策 = **retrain_e0_x3**（loader 证书单次 `EVIDENCE_MISSING` + sampler 不等价；`e0_reuse_decision.json`），无歧义。
- [x] 主运行矩阵完整或提前终止原因完整；所有正式 run 均为 `objective=unified_move_process`。
- [x] 三 seed/CI/H0–H3/成本报告完成——其中候选三 seed CI 合法缺席（`n_selected=0`，`three_seed_paired_ci.json` 记 `gate_skipped`、不编造 p 值）；E0 三 seed 基线冻结。
- [x] T4 因 `no_new_embedding_gain` 合法 `gate_skipped`；容量/深度混杂**未闭合**，已按 §9.4 如实声明。
- [x] dev 只按冻结协议使用（veto 不适用、无调参、tau 未重算）；rollout 双口径分列（1,152 episodes）；confirm 未读。
- [x] 唯一 `E*` 决策 = **E0**（无可靠改善时留任，§9.5/§10.5），机器可重放（`reports/single_seed_screen.json` + `benchmark_summary.json`）。
- [x] bundle 8 文件白名单 + SHA 审计通过，`frozen-E0` strict-load/read-only（`downstream_handoff.json`）。
- [x] 文档状态按真实证据更新；`scientific_failure`/`gate_skipped` 未混同实现失败。

### 19.3 与规划假设的偏离（如实登记，均为门内合法分支）

1. **E0 复用分支未走通** → 按 §8.1 走 `T0-R` 重训 E0×3（非偏离，是预注册分支的实现）。
2. **E1–E3 全部 H2 失败**（sign；E3 另 dead）→ T3/T4 合法跳过；`provisional_H3` 不豁免硬门（§11.1/§15）。
3. **E4 成本比 3.6455 > 2.0** → 按 §11.2 hard_ratio 记 `gate_skipped`（科学结果保留，不晋级；非 `resource_failure`）。
4. **rollout attempt-1 v1 loader SIGTERM** → 资源失败留账 + 已认证 loader v2 实现修复重跑（名单/checkpoint/tau/预算不变）。
5. **WP5-CERT 单次 `EVIDENCE_MISSING`** → 按 §8.1「只允许一轮预注册补证」不重试，直接进入 E0 重训分支。

### 19.4 给 V5.4.2 的固定输入

§16.2 六项已随 `exports/embedding_champion_bundle/` 冻结交付（E*=E0 空分支统一 schema + reference checkpoint SHA）；`joint-finetune-E*` 须另立 V5.4.2 预注册实验。
