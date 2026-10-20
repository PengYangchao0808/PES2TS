# W-FEAT / E2 交接 — R3 v3 特征·模型·损失栈 + 数学测试电池

> 完成时间：2026-10-20 · lane：W-FEAT（E2 特征/模型/数学测试道）· branch：`ops/e2-feat` · worktree：`/tmp/opencode/pes2ts-wt-b`
> **状态：fixture_passed — 全部结果 `not_admissible_until_gate_pass`，需 G0+G1 在真实数据上通过后方可采信（预期状态，非缺陷）。**

## 1. 目标与结果（摘要 ≤40 行）

实现了 R3 v3（ReactionIntent 条件化）的合同/特征/模型/损失栈与完整数学测试电池，全部在冻结的 F3 槽位超集与 PD-2/PD-6 容差/掩码语义下：

- **合同** `contracts/key_atom_v3.py`：F3 固定槽位超集（F0/F1/F2 仅由 mask 获得，臂间禁止偷加参数）、`IntentRecordV3`（G_src/G_tgt/ΔB_intent/CV/规则；`reversed_intent()` 实现真 Reverse 与标签对合）、前视键白名单 + 监督键隔离、PD-6 五类 intent-off 键映射、`FORBIDDEN_INPUT_CLASSES_V3` 十类清单。
- **特征** `features/key_atom_spec_v3.py` + `key_atom_graph_v3.py`：当前几何唯一 3D 输入（无 X0/anchor/history 通道）；边 = G_src∪G_tgt 共价 ∪ 当前半径图（4.5 Å/≤32 邻/1e-6 平票全留）；f2 内容 = [pos, elements, K, degK(原始计数), 无序端点上下文, B_src/B_tgt/ΔB_intent]；f3 = +当前距离 RBF(12)；`intent_role` 三通道多标签（可同时 forming/breaking）；`cv_target_delta` 有向 CV 描述；`signed_intent` = 主 CV 目标增量符号（G_rxn 的确定性投影，非独立 direction token——合同 §3.1 不可冲突性质满足；PD-6 类5载体，intent-off 置零）。
- **模型** `models/key_atom_egnn_v3.py`：StarEGNN v3，hidden 48 / 3 层 / **精确 55,441 参数**（<70k 预算；见 §4 偏差说明）；节点输入 21 维、边输入 22 维、图条件 5 维（CV+符号）；向量头 = signed-tanh × 极向边向量 × a_ref/max(1,degK)，无 anchor 项；z 头 64 维保留（V5.2 head 复用）；fp32；无 dropout/batchnorm → eval 两次前向逐比特一致（测试断言）；NaN/越界 Z → typed invalid；`forward` 拒绝任何非前视键（`FORWARD_VIEW_LEAK`）。
- **损失** `models/key_atom_losses_v3.py`：`L_atom + 0.5·L_pair + λcos(t)·L_cos + λs·L_strong + λd·L_dead`；λcos=R0 斜坡（250 步后起、250 步线性至 0.02）；Huber δ=0.25、d0 归一 0.08 Å、pair_mode=legacy_j、cos 门 0.02 Å；strong 桶 `Huber_0.25((Δd̂−Δd*)/max(|Δd*|,0.01))`（|Δd*|≥0.02）；dead 桶 `[(max(0,|Δd̂|−0.005)/0.005)²]`（|Δd*|≤0.005）；空桶恰为 0；S2/S10/D2/D10 单桶臂注册表（拒绝混臂、λ≤0.5，数值 λ 由 WP 梯度校准提供）；权重恰乘一次（`frame_weight`=label_weight，均匀权重下与 B0-F 逐比特一致）。
- **λ=0 逐比特回归（T1.5）**：λs=λd=0 且权重均匀时，total 及 l_atom/l_pair/l_cos 与冻结基线 `key_atom_bc_loss_vectorized` `torch.equal`（多种子/带 frame_mask/带均匀权重）；保证机制 = 逐字复制的 legacy_j 帧数学 + 0/1 keep×weight 归约（全1 ⇒ w==keep 逐比特）+ 除以 w.sum()（恰等于基线整数帧数）+ λ=0 时桶分支整体跳过。四段辅助函数（radial_huber/soft_cosine/pair_unit_vectors/exact_delta_d_torch）因产品边界（models/** 不得 import offline）逐字复制并由该测试钉死。
- **数据** `offline/data/key_atom_dataset_v3.py`：manifest 驱动、intent_id 键控的六表加载（PD-1；列镜像 E1 `r3_6table_v2` 并有机器钉死测试）；d0=0.08 主任务过滤、fold/split manifest、curve_weight 每 (curve,d0) 和=1 校验、intent 对合校验、manifest sha256 fail-closed；监督字段永不进前视。
- **门 harness 扩展** `scripts/agent_ops/feature_gate/`（v2 机制内增量扩展，未改 v2 语义/版本号）：非空间 [*,3] 通道（intent_role）排除出旋转变换；边对齐键排除出置换节点 gather；PD-6 保留集补 key_index/is_key/edge_attr；新增特征级 `check_feature_relabel_identity`（display-name 交换 ⇒ 输入逐比特一致）与 `check_feature_reverse_change`（真 Reverse ⇒ 有向特征全变、无向上下文保留）；v3 合成化学与六表 fixture（`v3_fixtures.py`）。
- **G1 报告**：`build_gate_report(v3 model, forward-view batch, preset=30h_v1.1_normative)` → state=pass，T1.1–T1.4c 全 pass，yaml sha 绑定；`r3_v3_record`（1e-4）恒 blocked/inadmissible，PD-2 禁止自动放宽（有测试：数值上低于 1e-4 也绝不通过）。

## 2. 产物与证据

| 文件 | 用途 | 测试状态 |
|---|---|---|
| `pes2ts_core/pesn/contracts/key_atom_v3.py` | v3 合同：槽位/意图记录/前视-监督分离/PD-6 类映射 | fixture_passed |
| `pes2ts_core/pesn/features/key_atom_spec_v3.py` | v3 spec：RBF12/空间图参数/level mask/意图-off 数组级实现/spec 文档读写 | fixture_passed |
| `pes2ts_core/pesn/features/key_atom_graph_v3.py` | Reaction Intent 特征构建 + collate + torch 批 | fixture_passed |
| `pes2ts_core/pesn/models/key_atom_egnn_v3.py` | StarEGNN v3（55,441 参数、fp32、确定性、typed invalid） | fixture_passed |
| `pes2ts_core/pesn/models/key_atom_losses_v3.py` | R3 损失函数（λ=0 与 B0-F 逐比特、桶、权重一次、臂注册表） | fixture_passed |
| `pes2ts_core/pesn/offline/data/key_atom_dataset_v3.py` | 六表 manifest 数据集（intent 键控；scope gap 已由 supervisor 授权） | fixture_passed |
| `scripts/agent_ops/feature_gate/equivariance_harness.py` | v3 增量扩展（非空间/边对齐/保留集/特征级检查） | fixture_passed（v2 回归全绿） |
| `scripts/agent_ops/feature_gate/v3_fixtures.py` | v3 合成化学 + 六表 fixture 写出 | fixture_passed |
| `tests/pesn/test_r3_*.py`（8 文件）+ `tests/agent_ops/test_feature_v3_gate.py` | 数学测试电池（见 §3） | 136 passed |

## 3. 测试矩阵（解释器 `/tmp/opencode/venv/bin/python`，CPU，无 CUDA）

| 命令 | 结果 |
|---|---|
| `pytest tests/pesn/test_r3_relabel_invariance.py tests/pesn/test_r3_reverse_sensitivity.py tests/pesn/test_r3_intent_off_mask.py tests/pesn/test_r3_input_legality.py tests/pesn/test_r3_equivariance.py tests/pesn/test_r3_losses.py tests/pesn/test_r3_model.py tests/pesn/test_r3_dataset.py tests/agent_ops/test_feature_v3_gate.py -q` | **136 passed**（≈2s） |
| `pytest tests/pesn tests/agent_ops -q` | **1165 passed, 22 skipped**（无回归；含 WP-F2 harness 全部旧测试） |
| `pytest tests -q --ignore=tests/test_neardup.py` | 3009 passed / 72 failed —— 72 个失败在 `tests/test_{selector_core,scan_plan_compat_export,split_freeze,pipeline,...}.py`，**已在 master 干净树复现（同解释器 36/36 失败）＝预存环境依赖问题，与本 lane 无关**；`test_neardup.py` 缺 `drfp`（预存） |

覆盖要点：relabel 特征逐比特（4 个 level × 3 组名字 + 对合）与输出 ≤1e-6；真 Reverse 有向特征精确交换/取反/角色换位、无向上下文逐比特保留、输出改变（exact negative 明确不要求）；小步切向近似相反诊断（非精确负）；PD-6 五类逐一可检出（每类单独泄漏均 fail）、全 mask pass、nuke fail、上下文保留；十类禁止输入逐一探测命中 + 前视白名单/黑名单双检 + 监督键不可达 + 样本数据类无历史字段；PD-2 三向等变（median/P99/max 记录）+ record-only 永不准入 + yaml 绑定拒绝放宽；λ=0 三种子×三 λcos 逐比特（含 mask/权重）、R0 斜坡数值、strong/dead 桶手算公式精确匹配、空桶恰 0 且总损逐比特不变、未变化 pair 排除、权重均匀重标度为无操作/一次加权与手工一致/二次加权可检出、确定性归约、混臂拒绝、梯度有限；模型参数 55,441==常数 <70k、fp32、eval 两调用逐比特、无随机模块、typed invalid（NaN/Z）、监督键拒绝；数据集六表加载/意图键合/权重不变式/fold 过滤/manifest 篡改拒绝/非主 d0 丢弃/schema 镜像 ⊆ E1 r3_6table_v2。

## 4. 与计划的偏差（登记，不回改计划）

1. **参数量 55,441 vs "~59k"**：v1 的 59,235 含 task19（branch/method/solvent/charge/…）与 X0 anchor 通路，二者在 v3 均为 ReactionIntent v1 §5 禁止输入；v3 增加 CV(4)+signed_intent 图条件后净差 −3.8k（<70k 预算内；yaml "~59k" 为量级描述，精确值以 `MEASURED_PARAMS_V3=55441` 落盘）。
2. **计划文件名 `tests/pesn/test_key_atom_losses_r3.py` → 实现为 `tests/pesn/test_r3_losses.py`**：lane 写入白名单模式为 `tests/pesn/test_r3_*.py`（单写者规则优先）。
3. **cv_spec 无显式 Δξ 时的派生 + 几何载体适配**：E1 交付的 `cv_spec`（kind=`graph_edit_bond_lengths`）只带坐标对，v3 依合同 §3.1 方向规则派生 `Δξ_ij = −ΔB_intent_ij`（断键→距离目标增、成键→减；G_rxn 的确定性投影，relabel 不变、reverse 敏感均成立；显式 `target_deltas` 优先）。几何载体由 E1 的 `irc_node.coords_flat/coords_shape`（PD-9 flat）承载（原计划的 JSON-in-x 备选保留）。两者均已与 G0 真实表实测兼容（§6.0）。
4. 损失模块内联复制了基线四段 helper（产品边界：`models/**` 禁 import `offline/**`；`test_product_boundaries` 强制），以 λ=0 逐比特测试钉死等价。

## 5. 接口冻结说明（WP-2 / 下游消费）

- **`endpoint_names` 合法性**：仅为 relabel 门的 display-name 元数据（allowlist reaction_intent_fields 成员）；任何数值特征禁止读取（有测试）；`model.forward` 不读取它（交换后输出逐比特一致）。WP-2 在线包装须保持该语义：端点名是任务描述的一部分，不参与计算图。
- **`model.forward(batch)` 只接受前视键**（`FORWARD_VIEW_KEYS_V3` 子集），多余键 → `KeyAtomV3ContractError(FORWARD_VIEW_LEAK)`；W-LOSS 训练循环须传 `forward_view_v3(batch)`，监督（y_teacher/pair_index/pair_edit/frame_weight）走损失入参。
- **`degK` 是原始计数**（损失需 1/max(1,degK)）；模型侧内部归一。`frame_weight` = `label_weight`（损失恰乘一次）；curve_weight 属采样侧（和=1 不变式由数据集校验）。
- **`signed_intent`** = sign(主 CV 目标增量)，G_rxn 的确定性投影（非独立 direction token；intent-off 置零；PD-6 类5）。若 WP-2/V5.2 需要更强的方向语义，须走架构审查，不得静默改键义。
- **G1 报告消费**：使用 `state` / `passed`（`gate_admissible` 仅为预设可采信标志，单独不构成门结论）。
- 容差预设机制沿用 WP-F2：`30h_v1.1_normative`（rel 1e-5 + abs 1e-6；relabel ≤1e-6 且逐比特）；`r3_v3_record`（1e-4）仅记录，永不放宽。

## 6. 未决问题 / 下游依赖

0. **与 E1 G0 交付的真实六表已实测兼容（只读探针，2026-10-20）**：`KeyAtomDatasetV3` 直接加载 `outputs/pesn/30h_campaign/data/`（G0 PASS 件）→ 9655 个 ok 样本、100 个 intent、folds 0–4；特征构建与 StarEGNN v3 前向在真实样本上运行通过；Reverse 时 `cv_target_delta`/`signed_intent` 正确翻转。为此数据集/合同已适配 E1 编码：`g_src/g_tgt` dict（`atoms`+`bonds`）、`irc_node.coords_flat/coords_shape`（PD-9 flat）、扁平 `target_action_u`、assignments 型 split manifest；`cv_spec` 无数值 Δξ 时按合同 §3.1 方向规则派生 `Δξ=−ΔB`（kind=`graph_edit_bond_lengths`；显式 target_deltas 优先）；curve_weight 不变式在**过滤前**的全 (curve,d0) 组上校验（masked 行也计入）。以上均有 fixture 级测试（tests/pesn/test_r3_dataset.py E1-encoding 组）。
1. **E1（W-DATA）**：~~几何载体~~（已由 `coords_flat` 解决）；若后续在 `cv_spec` 落显式 `target_deltas`，派生逻辑自动让位（向后兼容）；CV 原子已可经 `coordinates` 参与 K 集。
2. **W-LOSS（wave 2）**：`offline/training/key_atom_bc_v3.py` 只读消费 `key_atom_loss_v3`（函数级，无训练循环在此）；S2/S10/D2/D10 的数值 λ 需在新 R3 B0-F checkpoint 上按 20 个真实 inner-train batch 校准（ρ 2%/10%，λ≤0.5，head≤0.3）——`AUX_ARMS_V3` 已给目标份额与校验；训练循环须传 `forward_view_v3(batch)` 给模型、监督走损失入参。
3. **G0/G1 门**：本批全部 fixture 级结果 `not_admissible_until_gate_pass`；G1 消费 `build_gate_report` 输出时以真实数据重建（真实数据探针通过 ≠ G1 门通过）。
4. 预存环境问题（非本 lane）：本 venv 下 `tests/test_neardup.py`（缺 drfp）与 72 个 selector/split/pipeline 测试失败在干净 master 复现，需基础设施道处理。
