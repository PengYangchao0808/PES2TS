# PES2TS V4.2–V4.7 实施合同与战役冻结 v1

> 状态：Frozen（实现批次合同；启动正式训练前冻结，训练开始后不得修改）
> 层级：G3.P / 工作流 R；上游：[V4.2–V4.7 方案 v1.1](../plans/PES2TS_V4.2-V4.7_十小时训练实验方案_20261017.md)、[ADR-0023](../design/decisions/ADR-0023-V4.2原生重连数据与内部开发划分.md)、[ADR-0024](../design/decisions/ADR-0024-V4.2无界位移模型与战役训练协议.md)
> 下游消费者：本实现批次全部泳道（数据/模型/训练驱动/评测）、overnight 驱动器、终审报告
> 当前版本：v1 @ 2026-10-17；只读引用下列既有资产，不改动任何历史冻结件

## 0. 冻结资产清单（输入，全部只读）

| 资产 | 路径 | 身份 |
|---|---|---|
| fit 特征 v3（旧细分数据） | `outputs/pesn/dual_ts_v2_20261015/wp3_features_v41/` | spec `pesn_dual_ts_features_v3@f678f8e5…`，50 npz |
| monitor8 特征 v3 | `outputs/pesn/dual_ts_v2_20261015/wp4_monitor/features_v41/` | 同 spec digest（reused_frozen） |
| fit50 teacher（含观测） | `outputs/pesn/dual_ts_v2_20261015/wp2_fit50_observed/` | `pes2ts_dual_ts_teacher_v2` + `gaps_registry_observed.json` |
| monitor8 teacher（含观测） | `outputs/pesn/dual_ts_v2_20261015/wp4_monitor/teachers_observed/` | 同上（monitor 版） |
| E/g 收据缓存 | `wp2_observations/`、`wp4_monitor/observations/` | `orca_gradient_product_v1`；GFN2-xTB，dE/dX，Ha/Å |
| 划分谱系 | `splits/dual_ts_split_lineage_v1.json` | fit 波 50 组名单 + 身份绑定审计 |
| v4.1 训练收据 | `wp41_bc/run_bc_*/run_record.json` | `pesn_dual_ts_bc_run_v2`，13 run |
| 采样器/评测/收据核验代码 | `pes2ts_core/pesn/offline/training/dual_ts_sampling.py`、`dual_ts_monitor.py`、`dual_ts_bc.py`、`pes2ts_core/pesn/offline/data/dual_ts_assemble.py` | 复用，不改动语义 |

禁止读取：`wave.test`（direct_dev8）、confirm8、family holdout 名单之外的任何用途、`wp3_features/`（旧 v2 泄漏特征）。`wp2_observations` 全量遍历禁止（9p 慢盘）；审计只允许按 computation_identity 定点抽查。

## 1. 新版本与身份登记

| 对象 | schema/版本 id | 父 hash 绑定 |
|---|---|---|
| 原生重连数据（fit+monitor） | `pesn_dual_ts_features_native_v1` | v3 spec digest + 对应 `gaps_registry_observed.json` sha256 + teacher digest |
| 原生特征 manifest 工具 | `pes2ts_dual_ts_native_rebuild_v1` | — |
| 内部开发划分 | `pesn_dual_ts_dev_split_v42_v1` | splits lineage sha256 + fit 波组名单 hash |
| 物理先验 spec | `pesn_dual_ts_physical_prior_v1` | 划分 hash + 质量表版本（`direct_preprocess.masses_for`） |
| 无界位移头/动作版本 | `dual_ts_unbounded_displacement_v1` | — |
| E1 特征 v4（7ch 边） | `pesn_dual_ts_features_v4` | v3 spec digest（node_scalars 不变） |
| V1/V2 向量网络 | `dual_egnn_painn_v1` | — |
| 战役协议 | `pesn_v42_campaign_protocol_v1` | 本合同 sha256 |
| run 记录 | `pesn_dual_ts_bc_run_v3` | 向后兼容 v2 字段，新增 §6.4 扩展键 |
| V4.7 定稿 spec | `pesn_dual_ts_physical_prior_v1@fit50` + native_v1@fit50 | 全 fit50 重拟合，新 hash |

s_g/s_delta_e **沿用 v3 冻结值**（同一批已配准几何与收据，逐态 g 不因重连改变；避免把缩放重拟合混入 D2↔D3 归因）。新增统计量（a_s、Δs_fit、eps_g、小梯度阈值、λ）一律只在 fit45_train（V4.7 为完整 fit50）上按本合同规则估计并冻结。

## 2. 原生重连数据规格（Lane A 实现）

**输入**：`wp2_fit50_observed/RXN_*.teacher.json`（states 链 + labels + observation.gradient）+ `gaps_registry_observed.json`（fit 版与 monitor 版各一）+ `wp3_features_v41/*.npz` / `wp4_monitor/features_v41/*.npz`（逐态特征原样继承）。

**重建规则**：
1. 逐分支读 teacher states，按 `gaps_registry` 的 `inserted_state_id` 全集（fit: 399；monitor: 以其 registry 实数为准）剔除插入态，得原生节点序列（含教师端点）。
2. 原生后继 = 序列中下一原生节点；产出新 `next_state_id` 与 `action_delta = X_next − X`（沿用已配准坐标，**不得重新配准**）；`remaining_arc_length` 在原生链上重算（同 `mass_weighted_arc_v1_diag_masses` 度规）；`ts_displacement` 端点量不变。
3. 逐态特征数组（node_scalars、x_angstrom、g_scaled、edge_*、masks 等）按 state_id 从现有 npz **原样拷贝**；只替换链结构与标签侧字段。
4. 受影响子转移解释（R7）：`a_max_exceedances.count=395` vs `inserted_nodes=399` 的差额必须从 registry 逐条解释（同一步多轮/多重插入），写入审计；解释不了即 hard fail。
5. 产出目录：`wp42_campaign/native_features/`（fit 50 组）与 `wp42_campaign/native_monitor_features/`（8 组），每目录含 npz + `features_manifest.json`（含 population 计数、逐反应原生步数、parent hash、审计摘要）。

**审计硬门（native_reconnect_audit.json，全部通过才允许训练）**：
- 计数：fit 侧原生动作态 = 5,568（5,967−399），总态 5,668，教师端点 100；monitor 侧按其实 registry 重算并登记。
- 标签恒等：`action_delta` 与 `X_next − X` 逐态逐原子 |差| ≤ 1e-10 Å（float64 精确重算）。
- 每个非终态恰一合法后继；双侧朝 TS 排序不变；原子 map/单位断言沿用 teacher 链。
- 原生最大原子动作 P50/P90/P95/P99/max（Å）。
- E/g 抽查：随机 20 个原生态，定点读收据 `gradient_hartree_per_angstrom`，与 teacher `observation.gradient` 数值一致（≤1e-10），并登记 method=GFN2-xTB/basis=""(xTB 无基组)/charge=0/multiplicity=1/solvent=未记录（显式标注，不默认与教师 DFT PES 同源）。
- npz 与 teacher.json 的 `x_angstrom` 抽查一致（20 态）。

## 3. 内部开发划分（Lane A 实现，首跑前冻结）

1. 单元 = 身份簇：从 splits lineage 的 fit 波 50 组出发，审计内 identity 绑定的 duplicate-group 伙伴聚为同簇（簇内不得跨 train/dev）。
2. 分层元数据（每簇）：原生动作数、是否含插入（重连前）、原子数 max、disputed-5 成员。
3. 确定性规则：簇按 `(min_group_id)` 字典序排列；dev 簇取排列后第 5/15/25/35/45 位（1 起）；若 dev 集合"含插入组数"为 0 或为 5，则自第 46 位起按字典序找最近的未用簇中首个满足补齐性质的簇替换（一次替换，规则唯一）。
4. 产物 `wp42_campaign/dev_split.json`：两名单、每簇元数据、规则字符串、`split_sha256`、生成时间、lineage 父 hash。与既有 `wp3_features_v41_a45/` 的组集合差异只记录不采纳。
5. D0–D2 训练态 = 旧细分数据中 fit45_train 组的态；D3+ 训练态 = native_features 中 fit45_train 组的态；dev 评测 = 各标签版本下 fit5_dev 组的态。

## 4. 物理先验 spec（Lane A 实现估计，Lane B/C 消费）

- `b_i = (g_i/m_i) / sqrt(Σ_j ‖g_j‖²/m_j + eps_g²)`，g 取 teacher `observation.gradient`（Ha/Å，dE/dX；若源为力按 `g=−F` 转换——当前收据已声明 dE/dX，直接用），m 用 `direct_preprocess.masses_for`（绑定版本，未知 Z typed refusal）。
- `eps_g = 1e-3 × median_fit45_train( sqrt(Σ_j ‖g_j‖²/m_j) )`；小梯度阈值 `tau_g = 1e-6 × 同 median`；低于 tau_g 的状态 a_phys 全零并单列覆盖率。
- `Δs_fit = max(0, Σ w b·a* / Σ w ‖b‖²)`，权重 w 与 L_xyz 归约一致（原子→态→分支→组等权）；分母 ≤1e-30 显式报错。
- 附带输出：fit 原生质量加权步长中位数（审计用）。
- 产物 `wp42_campaign/physical_prior_spec.json`（V4.7 时按同规则在完整 fit50 重估为 `…@fit50` 变体）。

## 5. 模型、头与损失（Lane B 实现）

- **无界位移头** `dual_ts_unbounded_displacement_v1`：`a_hat = readout(h,v)`，直接输出 Å，无 tanh/无上限；readout 复用现有 mu 路径但去 radial transform。H1 变体：`a_hat = a_phys + readout(h,v)`，a_phys 由 (g,m,Δs_fit,eps_g,tau_g) 逐态预计算、不参与梯度；同时记录纯 a_phys/残差/合成三套指标与 ‖residual‖/‖skip‖（零分母单列）。
- **L_xyz** = `‖a_hat−a*‖²/a_s²`，a_s = fit45_train 非零原子 `‖a*‖` 中位数（冻结正尺度，非上限）；归约：原子均值→态→分支→组（沿用 `reduce_step_sums` 固定归约）。
- **L_dir**：仅 `‖a*_i‖≥0.01 Å` 原子，`1−cos`，有效分母显式记录（复用 per_atom_dir_loss）。**L_mag**：同一集合 `Huber_δ=1( log((‖a_hat‖+1e-8)/(‖a*‖+1e-8)) )`。
- **λ 校准 canary**：fit45_train 固定 32 batch（共享层参数），各损失梯度范数中位数，`λ_dir = 0.3·med‖∇L_xyz‖/med‖∇L_dir‖`、`λ_mag = 0.1·…`；无效分母/零梯度显式退出；校准后冻结入 run 配置。
- **E1**（features_v4）：边特征 4→7，追加 `rhat_ij·ghat_i、rhat_ij·ghat_j、ghat_i·ghat_j`；ghat=g/‖g‖（g_scaled 归一即可，尺度不变），零范数原子置零并 mask；node_scalars 仍 5ch；新 spec digest。
- **E2**：关键对白名单 = 任一端点 bond 表 changed_flag=1 的并集边（部署可得）；pair 条件记录 = [dist RBF32, 两端 bond 状态编码, driver(对本身 changed)/guard(邻接 changed) 角色, 来源/缺失标记]；独立 pair encoder → 节点广播求和 + FiLM 调制 h；参数计入报表。骨架消息图不变。
- **V1** `dual_egnn_painn_v1`：16 向量通道；邻居向量传递、径向向量消息、标量/向量内积耦合；输入仅当前 X/g/合法拓扑；空间轴不做独立 LayerNorm；无单叉积；参数/每步耗时上报。
- **V2**：V1 + 半径 5Å 接触边（逐态构建，平滑 envelope 仅作用于新增接触分支；化学并集边永不消减；确定性截断：每原子至多 16 个最近接触邻居，按 (dist, atom_idx) 排序冻结）。
- **宽度**：hidden ∈ {64,96,128}，n_layers=3，`light=False` 显式传参；融合端点编码宽度固定并记录；参数量从实例统计。

## 6. 训练协议与 run 记录（Lane C 实现）

- 新 preset `campaign_v1`：AdamW(lr 峰值 3e-4, wd 1e-5)、warmup 5%（20k→1000 步；40k→2000）、余弦至 1e-5、clip 1.0、batch 64 states、EMA decay 0.999（warmup 后启动，raw+EMA 都存）。preset `legacy_v41`：Adam lr 1e-3、旧设置原样（D0/D1 专用；batch 64 与旧 run 一致）。
- 检查点：`checkpoints/u{5000,10000,20000,…}` 按 update 独立落盘（修复 v1 name-被忽略缺陷只在新 trainer，不动旧文件/旧 run 语义）；同时保存 raw 与 EMA 身份。
- dev 评估（5k/10k/20k 各一次）：fit5_dev 上组中位 MSE/零动作比值、整体 cos、near-TS cos、near-zero 原子绝对 RMSE、有效分母、逐组值；fit45_train 固定诊断集同点留存。raw-vs-EMA 读取规则：首个 D2 种子完成后按 dev（组中位 MSE→cos）一次性选定并全夜冻结，写入 stage_decisions。
- 采样器：`group_balanced_v41`（现有 DualTSGroupBranchSampler）与 `epoch_traversal_v41`（现有 epoch_state_order 语义封装成同接口）。D2/D3 用 D0/D1 在 dev 上的 §9.1 优胜者。
- 曝光：扩 ExposureTracker——逐反应/分支 draws P10/P50/P90/max、唯一态覆盖率、近零/长步(P90)/近 TS 桶。
- 恢复：网络/优化器/scheduler/EMA/采样器 RNG/曝光账本/更新计数全量可恢复；断点恢复时间计入该臂封套；非有限损失/梯度立即停臂留 typed 记录。
- run_record v3：继承 v2 全部键，新增 `arm_id、preset、checkpoints[]、dev_evals{u5000,u10000,u20000}、ema、timing{steps_per_sec,peak_mem_mb,eval_seconds}、resume_events[]、budget_state`。
- 时序标定：每新架构 warmup 100 步弃置 + 500 步计时 + 1 次 dev 评估，×1.25 余量估全程；记录设备/精度/batch/原子边数/峰值显存。

## 7. 评测三口径与晋级（Lane D 实现）

- `legacy_full`：现 monitor8 v3 特征 + 原组名单/掩码 → 原四项门（复用现有 monitor 机器与阈值 digest）。
- `native_full`：native_monitor_features → 新任务四项研究门（阈值 1.0/6-of-8/0.30/0.40 预注册沿用，明示为新协议）。
- `common_unchanged`：legacy∩native 中"当前帧未被细分、下一帧/动作/掩码一致"的转移交集；交集名单+hash 在看任何预测前冻结；缺失分母显式登记。
- §9.1 晋级规则按方案原文机械化（两种子成对比较；cos +0.01/near-TS −0.01 上限、组中位 MSE 比值 +0.01、pooled +0.02、near-zero RMSE +10%、组数不减；排序：组中位 MSE→cos→参数量）。§9.2 40k 门与 V4.7 重拟合同样机械化。
- 终审报告字段按方案 §10 清单；旧 checkpoint 仅在新字段/动作语义兼容时参与比较，否则显式跳过。

## 8. 战役臂登记（预注册上限；实跑由时序标定裁剪）

种子：筛选 20261017/20261018；V4.7 20261017/18/19。全部 20k（40k 仅 §9.2）。

| 臂 | 配置元组（数据,头,采样器,preset,增量） | 新 run 数 |
|---|---|---|
| D0 | 旧细分@fit45_train, bc_nll(radial-tanh), group_balanced, legacy_v41 | 2 |
| D1 | 同 D0 但 epoch_traversal | 2 |
| D2 | 旧细分@fit45_train, displacement+L_xyz, D0/D1 dev 优胜采样器, campaign_v1 | 2 |
| D3=H0 | native@fit45_train, 同 D2 | 2 |
| H1 | D3 + a_phys skip | 2 |
| H2 | §9.1(H0,H1) 优胜 + λ_dir L_dir + λ_mag L_mag（canary 校准） | 2 |
| E0 | V4.3 优胜（复用 run） | 0 |
| E1 | E0 + features_v4 边特征 | 2 |
| E2 | E1 + 关键对条件 | 2 |
| V0 | V4.4 优胜（复用） | 0 |
| V1 | V0 + dual_egnn_painn_v1 | 2 |
| V2 | V1 + 接触边【可裁】 | 2 |
| W64 | V4.5 优胜（复用） | 0 |
| W96 | 优胜结构 hidden=96 | 2 |
| W128 | hidden=128【可裁】 | 2 |
| FINAL | 唯一配方 @ 完整 fit50 native（spec/prior 重估）×3 种子 | 3 |

裁剪顺序（超时按此删完整双种子臂）：W128 → V2 → E2。V4.7 三种子与末段 50 min 缓冲优先保障。预算删失记 `budget_censored`，非性能失败。

阶段边界（自正式窗口 T0 起）：预检 0:00–0:35；V4.2 0:35–2:00；V4.3 2:00–3:25；V4.4 3:25–4:50；V4.5 4:50–6:25；V4.6 6:25–7:35；V4.7 7:35–9:10；终审 9:10–10:00。

## 9. 写域（并行泳道唯一允许改动范围）

| 泳道 | 新增文件 | 允许触碰的既有文件 |
|---|---|---|
| A 数据 | `pes2ts_core/pesn/offline/data/dual_ts_native_rebuild.py`、`dual_ts_dev_split.py`、`dual_ts_physical_prior.py`；`scripts/audit_native_reconnect.py`、`scripts/audit_gradient_baseline.py`、`scripts/build_native_features.py`、`scripts/freeze_v42_splits.py`；`tests/pesn/test_dual_ts_native_rebuild.py`、`test_dual_ts_dev_split.py`、`test_dual_ts_physical_prior.py` | 无（全部只读消费） |
| B 模型 | `pes2ts_core/pesn/models/dual_ts_displacement.py`、`dual_egnn_painn.py`；`pes2ts_core/pesn/features/dual_ts_graph_v4.py`；`tests/pesn/test_dual_ts_displacement.py`、`test_dual_egnn_painn.py`、`test_dual_ts_features_v4.py` | 无 |
| C 训练驱动 | `pes2ts_core/pesn/offline/training/dual_ts_campaign.py`、`dual_ts_optim.py`；`scripts/run_v42_campaign.py`；`tests/pesn/test_dual_ts_campaign.py`、`test_dual_ts_optim.py` | 无（不修改 dual_ts_bc.py；legacy preset 在新 trainer 内等价重建并加对照测试） |
| D 评测 | `pes2ts_core/pesn/offline/evaluation/dual_ts_protocols.py`、`dual_ts_promotion.py`；`scripts/eval_v42_protocols.py`、`scripts/render_v42_summary.py`；`tests/pesn/test_dual_ts_protocols.py`、`test_dual_ts_promotion.py` | 无 |

产物根：`outputs/pesn/dual_ts_v2_20261015/wp42_campaign/`。任何泳道不得越界改他人文件；对既有模块的必要 import 以只读方式使用。

## 10. 验收门（实现批次 → 预检 → 正式窗口）

1. 泳道单测全绿（各自跑 `pytest tests/pesn/test_<本泳道>` 于 `pes2ts-gpu` 环境）+ 全仓 pesn 套件回归通过。
2. 真实数据冒烟：native rebuild 全 50+8 组实跑出审计四硬门；displacement/残差/EMA/检查点在 200 步真实 batch 上收敛冒烟；三口径评测在 v4.1 旧 checkpoint 上跑通 legacy_full 对照（兼容时）。
3. 预检（T+00:00–00:35 等价）：§2 审计硬门 + §4 基线（零/±g/±M⁻¹g/固定物理基线 + 可辨识性检索）+ 全架构时序标定；硬门失败即停正式训练。
4. 四层状态纪律：各新能力以 implemented/fixture_passed/smoke_passed 分层登记，训练收益另以运行收据表达。
