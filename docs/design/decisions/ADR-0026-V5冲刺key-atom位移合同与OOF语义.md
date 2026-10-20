# ADR-0026：V5 冲刺 key-atom 位移合同与 OOF 语义

> **2026-10-20 后续注记**：V5.1a 已 `rejected_gate`；V5.1a-R1 无候选准入；V5.1b 继续关闭。本 ADR 的旧 key-atom 合同保留用于 V5.1-R2 的可比对照；**V5.2 起采用 [GeometryPolicyContract v1](../../contracts/PES2TS_GeometryPolicyContract_v1.md) 与 [ADR-0028](ADR-0028-关键原子几何策略合同与V5-V7任务拆分.md) 的新任务语义**。本 ADR 不被删除或改写，历史证据继续有效。
>
> 状态：Accepted（实现交付中；能力状态按四层纪律单列，见 §7）
> 层级：G3.P.0–G3.P.3（V5.0 数据/数学、V5.1 输入/标签筛选、V5.2 条件 OOF）、G0/G1 仅数据资格增量
> 取代关系：不取代任何 ADR；落实 [V5 冲刺方案 v1.1](../../plans/PES2TS_V5.0-V5.2_十小时冲刺训练与验证执行方案_v1_20261018.md) §0 的"实现前版本化"要求
> 上游依赖：ADR-0021（双端拓扑）、ADR-0022（因果输入合同）、ADR-0024（无界位移）、ADR-0025（G–V 注册）
> 下游消费者：V5 冲刺战役（`config/campaigns/v5_sprint.yaml`）、V6 OOF 消费者（未开放）
> 当前版本：v1 / 2026-10-19

## 1. 决策

按冲刺方案 v1.1 实现 V5 第一阶段（10 小时窗口）所需的新动作/特征/OOF 语义，正式版本化如下。**本 ADR 只覆盖冲刺必需子集**；g_v_map 中 V5.0/V5.1/V5.2 条目登记的 `pre_relax.py`、`xtb_freeze_*.py`、`key_atom_painn.py` 等仍为 proposed，留给后续窗口。

## 2. 新合同（schema 版本均新建，不覆写旧物）

### 2.1 KeyAtomTask（`pes2ts_core/pesn/contracts/key_atom.py`，schema `pesn_key_atom_task_v1`）

- 输入原子行序 = IRC `ts_irc_index` 行序；`key_atom_rows` = bond_events（broken/formed/order_changed 的 atoms ∪ hydrogen_migration 的 from/h/to）涉及 map ID 按 map 号排序后的唯一行。
- 每分支：起点端点 = 该分支部署起点（branch b 指明用 endpoints[b] 作起点），另一端点图取 endpoints[1−b]。ΔB = B_other − B_start。
- 来源：`data/interim/g1_truth/p1_mapping/<shard>/<RXN>.json`（map/图/事件）+ `outputs/pesn/dual_ts_v2_20261015/wp2_fit50/<RXN>.task.json`（端点图）。写 source 路径与 sha256。

### 2.2 帧合同（`frame_contract.json`，schema `pesn_v5_frame_contract_v1`）

- 输入帧 = 因果当前帧 X_t（原生，不经任何未来对齐）；anchor X0 = 分支起点帧（row 0）经**固定 S/W 参考**的加权 Kabsch 对齐入 X_t 坐标系（只用 X0 与 X_t）。
- 标签帧 = X_t 坐标系；未来帧 Y[j] 仅在标签侧按同一固定 S/W 对齐后分段线性插值。分段含首段 [X_t, Y_{t+1}]。
- 参考集 S：起点拓扑中 A 的 1–2 共价 hop 内非关键重原子，取无可旋转单键连通的刚性块；权重 w_i=[i∈S]/(1+hop(i,A))²；参考散布门（λ1>0.01 Å²，λ2/λ1>1e-3）与秩不足时 `gauge_unresolved`，不退回全局轴。
- Kabsch 行向量约定：C=(Y−μY)^T W (X−μX)=UΣV^T，R=U diag(1,1,det(UV^T)) V^T，float64。

### 2.3 教师标签（`offline/data/key_atom_teacher.py`，schema `pesn_v5_teacher_v1`）

- 每分支原生帧存 `outputs/pesn/v5/native/<RXN>.branch{k}.npz`（pos float64、关键路径弧长、尾巴标记、q_frame）。
- 每 d0∈{0.05,0.08,0.12} 存 `outputs/pesn/v5/teacher/d0_<值>/<RXN>.branch{k}.npz`：`u_raw[S,m,3] float32`、`start_row/eligible/reason_id/terminal/capped/d_eff/eta/near_ts`。不将 d_eff/terminal/near_ts/采样索引写入特征。
- 首触点求解：每段 mean-square 距离 D²(α)=mean_i||y_i+αΔ_i−x_i||²（i 遍历关键原子，即 RMS_A²）为二次式，解 RMS_A=d0 取 [0,1] 最早正根（float64，根容差 1e-8）；cap=0.20 Å 为逐原子范数事件，取路径上首次触点（先到者），记录 d_eff/capped；先到 TS 取剩余尾步标 terminal。无未来非零步 → 无 MOVE 标签（reason `no_future_step`）。
- 尾巴标记（§1.1 vq/f 公式）：起点端向内连续 f<0.05 的 ≥5 间隔且前缀 max_l range(q_l)<0.02 Å → `flat_tail_candidate`，训练 q=0.25；裁剪上限 20% 关键路径长；非尾巴区连续低 f → `pair_metric_uninformative` 保留独立切片。

### 2.4 Split（`offline/splits/v5_fold.py`，`data/manifests/v5/split_manifest.json`，schema `pesn_v5_split_v1`）

- fit50 组级 5-fold（每组 ~10 簇，两侧全部帧同折），family/尺寸分层，seed=20261018；union-find 合并反向/重复/映射副本。
- Confirm（扩展池 ≥30 簇）与 OOD（family_holdout ≥3 族 × ≥10 簇）为**封存候选**，在查看任何 V5 标签前冻结登记；本轮默认不开封。
- 冻结前断言：fold 组集两两不交、组唯一折、含 sha256 与历史开发排除名单哈希。

### 2.5 冲突审计（`data/manifests/v5/dataset_audit.json`，schema `pesn_v5_conflict_audit_v1`）

- 范围 = fit50 内部（50 簇、两分支）近邻检索；历史 342 对未在本轮重扫（方案证据边界已声明）。
- 判据：完整分子 RMSD<0.03 Å 且关键 RMSD<0.02 Å 为候选；再要求 X0 对齐 RMSD<0.03 Å 且同 b/ρ/图编辑；标签 RMS>0.02 Å 且 C<0 才算输入不可辨识冲突。等价置换枚举上限 128（局部自同构群，按元素/度/环不变量分类），超限 `mapping_group_overflow` 不取随机前 128。
- 冲突连通分量无法依质量规则定坏端 → 训练权重 0；C∈[0,0.25) 近冲突权重 0.25；`context_resolved` 保留。

### 2.6 特征（`features/key_atom_spec.py` + `key_atom_graph.py`，schema `pesn_v5_feature_spec_v1`）

- 字段/维度严格按方案 §2 表：node36 / edge29 / task19 / 向量基 (r_ij, r0_ij, a_i) / pair_index+pair_edit / key_index+key_ptr / batch+ptr。
- **偏差（登记）**：仓库无 torch_geometric/TensorDict 依赖，用纯 torch 张量 + ptr 偏移实现**同语义同形状**合同；不引入新依赖。
- anchor off = 固定槽位清零（pos0 派生 r0 槽、Δd 槽、节点 X0 摘要 3 槽、夹角槽、anchor_valid、anchor 向量），张量形状不变；起点拓扑/电子态任务字段保留。特征只生成 anchor-on 一份，anchor-off 由装载期固定槽位掩码实现，保证六臂同一预处理几何。
- 空间图 ≤32 邻居、并列 1e-6 Å 全留；共价/K 边强制保留不受 cutoff 影响；图 = G_start∪G_other∪K∪当前 r<4.5 Å。

### 2.7 模型（`models/key_atom_egnn.py`，版本 `pesn_key_atom_star_egnn_v1`）

- M2 Star EGNN：hidden48、3 层、消息子图 = A 的 2 共价 hop ∪ 距 A<4.5 Å 节点，所选节点间共价/K 边 + 以 A 为中心星形空间边；message(48+48+29)→48→48，node residual(48+48+task19)→48→48；向量头 signed tanh 系数 + anchor 极向量 signed 通道，a_ref=0.08 Å，仅 A 输出；z=图/关键池化 64 维。
- 构造期断言 `sum(p.numel()) ≤ 90_000`，实测落盘。O(3) 等变由构造保证（向量通路只用极向量输入×标量系数），G5.1 fixture 数值复测。
- M0 Classic（0 参数解析基线）入 `models/key_atom_classic.py`；M1/M3/M4 本轮不实现（后续窗口）。

### 2.8 损失与训练（`offline/training/key_atom_bc.py`，版本 `pesn_key_atom_bc_v1`）

- 径向 Huber（δ=0.25，e=||u−y||/0.08）+ 0.5·pair + 0.02·cos（250 更新后 250 线性 ramp）；w_i=1/max(1,degK_i) 帧内归一；反应均衡 sampler 8 簇×4 帧、分支 2/2、q 采样；已采样实现权重则 loss 不重复乘。
- AdamW lr1e-3/wd1e-4、warmup250、cosine→1e-4、batch32、clip1.0、fp32、按更新数训练；正式外折固定预算不早停；开发筛选 early_stop 按 inner_val 反应簇中位 RMSE/zero、patience 8、min_delta 0.005。
- 每臂配对初始化与样本顺序（同 seed 同序）；run record 携带 ADR-0025 `gv` 块（campaign_id `V5_SPRINT`）。

### 2.9 OOF 语义（方案 §1.4/§5.5）

- 正式 crossfit 从零训练、外折不选 checkpoint、固定 3000 更新、标 selection-exposed；每 (样本,模型,种子) 恰一行，缺失明示 censored；`move_proposal_v5_draft1.parquet` 仅在 Tier3 完整后导出。
- z 隐空间仅存档不开放；主接口只有 U/几何摘要/不变量。

## 3. 冲刺战役注册

g_v_map 新增 campaign `V5_SPRINT`（status active，code_scope = 本 ADR 实现文件集，config `config/campaigns/v5_sprint.yaml`，data_manifest `data/manifests/v5/split_manifest.json`）；V5.0/V5.1/V5.2 保持 proposed（其 code_scope 含未实现文件，active 会触发路径存在性校验失败）。渲染：`python scripts/render_gv_map.py --write`。

## 4. 明确不做（本轮）

- 不启动任何 QC/ACP；Oracle 仅静态 teacher ceiling（G5.4 同 strict-copy 协议）。
- 不实现 M1/M3/M4、粗到细课程、扩量（Fit200/500）、封存集开启；Confirm/OOD 只登记不消费。
- 不改现行生产默认、guard、已冻结数据；历史 342 冲突对不重扫。

## 5. 验证义务（证据，非"tests green"）

1. G5.1 数学门 fixture 全绿（≥100 含共线/共面/对称 fixture × 20 旋转 × 10 置换，误差 <1e-5）。
2. 教师合成解析 fixture：线性路径首触点、cap、terminal、尾巴标记精确复现。
3. Stage0 `dataset_audit.json`/`split_manifest.json`/`frame_contract.json` 落盘且 sha256 链完整。
4. Tier1–3 run_records + `timing_probe.json` + `gate_report.md/json` 实测落盘；状态只用方案 §5.7 词汇。

## 6. 风险与裁决记录

- PyG 缺失 → 纯 torch 同语义实现（§2.6 偏差）。
- fit50 冲突检索覆盖 ≠ 全库覆盖 → `dataset_audit.json` 声明检索范围，"零冲突"仅限设定距离/群/覆盖。
- 600 分钟硬截止 → CLI 强制 wall-minute 包络与统一降档（1500→1000→500），censored 状态如实记录。

## 7. 能力状态（四层）

- 代码：implemented（本 ADR 文件集）
- 数学合同：fixture_passed（G5.1 合成 fixture）
- 真实数据：smoke_passed 于 Stage0 完成后升级
- 封存/泛化：unknown（未开启）
