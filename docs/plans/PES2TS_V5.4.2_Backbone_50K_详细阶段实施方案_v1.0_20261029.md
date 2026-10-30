# PES2TS V5.4.2 Backbone 50K 详细阶段实施方案（单帧空间 Backbone 基准）

> 状态：**Active-Proposed**（2026-10-29；WP0 治理/预注册文档已落地，WP1–WP7 待实现与执行；本轮不宣称训练完成或候选科学收益）
> 层级：V5.4 子战役；primary_g=G3.P.2，冻结评估延伸至 G3.P.3；不新增 G 阶段
> 取代关系：不取代开发宪法、MoveProcessContract v1、ADR-0032/V5.4.1 冻结产物或报告；细化 Oracle WP0 架构建议（`.slim/deepwork/v542-backbone-20261029/oracle-architecture.md`）
> 上游依赖：开发宪法、阶段定义、ADR-0031/ADR-0032、MoveProcessContract v1、V5.3-MP S4 50K 冻结数据、R3 冻结 checkpoint（`5372d9b12807c3f0`，SHA `a2e46b6ad65cc18996d6c1660c5ae24c723397ca22526e52fa34e430afd0417f`）、V541 共同 sampler/评估身份；Oracle 设计源 `.slim/deepwork/v542-backbone-20261029/oracle-architecture.md` SHA256 `4f13bfbd26d4acf17572641e30453000d65243599519c269473f07d6678ccccb`（WP0 冻结进 run identity）
> 下游消费者：V5.4.3 Decoder 基准、后续 backbone 冻结交接；不直接授予在线部署资格
> 当前版本：v1.2 @ 2026-10-30（fix-5 WP0 条件收口：final_selection_rule / route_logic / parity 分层 / signoff 角色 / A0 sanity band 硬前置；v1.0 文件名保留；v1.1 修复见 git 历史与台账）
> 配套治理：[ADR-0033 V5.4.2 Backbone 接口与预注册治理](../design/decisions/ADR-0033-V5.4.2-Backbone接口与预注册治理.md)；config `config/campaigns/pes2ts_v542_backbone_50k.yaml`（fix-2 车道产出）；validator `scripts/agent_ops/v542_config_schema.py`

---

## 0. 执行摘要

本阶段只回答一个问题：**在固定 50K 数据、固定 MOVE/PROCESS 任务合同、固定损失/采样/训练剂量下，哪一种单帧空间 Backbone（A1–A5）能在不损害 MOVE 的前提下超过 matched A0（legacy Star 三层 + 统一 Decoder），并且收益值得其实测计算成本？**

- 所有正式 run 均为 `objective=unified_move_process`；MOVE 由统一 `DirectMoveDecoder`（仅共享 `w_v`，49 params）产生；PROCESS 原样 `ReactionProcessHeadV53MP`（2,178 params）。
- **A0×3 重训是默认且实际选择**（`a0_reuse_decision=retrain_a0_x3`）：V541 E0×3 只作 parity/历史参照，H2/H3 对照固定为 matched A0×3。
- 候选 A1–A5 统一 hidden 48 / z 64 / edge_msg 48 / node 输入 21，backbone active ≤70,000；从冻结 R3 checkpoint 显式语义映射 warm-start（R3-only，禁止 V541 E0 final）。
- 运行阶梯：smoke 200（A0–A5；A1-C/A5-C 激活前 conditional smoke）→ 同环境 profile（A0–A5；A1-C/A5-C 激活前 conditional smoke+profile）→ seed17 diagnostic 2,000=400+1,600（A0–A5，仅工程淘汰）→ formal 400+37,500 共 37,900 updates，**顺序冻结：T0 A0×3 formal + `V542-A0-sanity-band` pass 最先 → T1 A1–A4 seed17（经诊断/成本门）→ top2 补 seeds 29/43（**top2 只属 A1–A4**，预注册 A1–A4×{29,43}=8 条件 run，激活 ≤4）→ A5 full-dose 条件激活 → A5 supplement（S29/S43 ≤2 runs，条件=`a5_full_dose_complete_and_seed17_screen_pass`）→ 条件归因诊断（A1-C/A5-C）**（smoke/profile/diagnostic 可在 T0 前先行，属技术测量）。
- H2 继承 V541 硬门（每 seed 都过；RMSE/zero 为**退化** ≤2% 非劣语义）；H3 对照 matched A0，路线判定用冻结括号逻辑（route_logic）：PROCESS `(auprc_pass AND q_non_inferior) OR (q_pass AND auprc_non_inferior)`（ΔAUPRC ≥+0.02 / Δq ≥+0.03），MOVE `(direction_pass AND rmse_non_inferior) OR (rmse_pass AND direction_non_inferior)`（direction 相对 ≥3% 且绝对 +0.01 / RMSE/zero 相对 ≥3%）；≥2/3 seed 同向、seed-averaged curve-level paired bootstrap 10,000 draws（`bootstrap_seed_root=20261029`，确定性子 seed 公式见 §9）、one-sided improvement p 值、Holm family 冻结（MOVE 两 exit 各 α=0.025 各自对 eligible candidates Holm；PROCESS 单一 family 对 2 endpoints × eligible candidates Holm，α=0.025）；**最终 Backbone\* 由冻结 `final_selection_rule` 在 H3-qualified 集内按字典序选定（不用 dev；A0/diagnostic 不参加；无人合格 → A0 留任）**。
- 成本门 `core_time_ratio ≤ 2.0`（分母 = V542 A0 同环境 profile）；A5 full-dose 仅在 A1–A4 已完成筛选及必要 top2 补种且无任何 H3 合格候选时自动激活。
- 无候选合格则 A0 留任（负结果按 §12 冻结写法如实登记）；不做 seed shopping、不做 dose extension、不按 dev 调参；confirm 永封；`new_QC_calls=0`；无生产 No-Move / verified TS 声明。

## 1. 范围、合同与完成定义

### 1.1 科学范围

**允许改变：** backbone 内部算子（A0–A5 whitelist）、相应参数量与实测资源协变量。

**保持共同：** 50K pool 与 split manifest、FORWARD_VIEW_KEYS_V3 输入视图（node 21 维 / edge 22 维）、合法有向拓扑、教师与标签、MOVE/PROCESS 任务语义（MoveProcessContract v1）、统一 `DirectMoveDecoder` 与 `ReactionProcessHeadV53MP`、联合损失与 λ、V541 共同 sampler、400+37,500 训练剂量、inner-val 选择与 checkpoint 规则、tau 校准纪律、FP32、评估协议。

**明确不做：**

- 不修改 legacy 模型源文件（`key_atom_egnn_v3.py`/`key_atom_heads_v3.py` 等）、旧 V5.4/V5.4.1 冻结文件、历史 run/metrics/只读报告。
- 不引入历史帧、未来 XYZ、参考 TS/端点 XYZ、真实 q、能量/梯度、curve/family/split 进 forward；非关键位移恒为零。
- 不改 RBF/cutoff/d0/cap；不引入向量通道 LayerNorm/ReLU、cross product、绝对坐标 attention logits、可学习全局帧、稠密 N² attention、adaptive halting、坐标更新 backbone。
- 不用 `torch.compile`。
- 不用离线指标或离线评估宣称 verified TS、生产部署或 G4 出口；**无生产 No-Move 声明**。
- **本阶段 `new_QC_calls=0`**：不调用 ACP/xTB/ORCA/DFT。

### 1.2 阶段完成定义

同时交付以下结果才算关闭：(1) 统一接口与 A0 strict 迁移 parity 证据完整；(2) 候选 H0 合法性/等价/有限梯度证据完整；(3) smoke/profile/diagnostic 账本完整；(4) 两个最终候选（或 A0 留任）有完整三 seed/删失账本；(5) 预注册 H0–H3、curve-paired CI、失败数量与计算成本同报；(6) dev/声明边界清楚，confirm 访问账本保持封存；(7) 产出唯一 Backbone\*（可以是 A0）及可 strict-load 的交接件。

## 2. 总体工作分解与依赖（WP0–WP7）

```text
WP0 治理/预注册冻结
  └─→ WP1 统一接口 + A0 strict 迁移 + parity 证书 ──┐
WP0 ─→ WP2 候选 A1–A5 实现 + H0 合法性              │
        └─→ WP3 V542 runner/identity/resume ────────┤
WP1 + WP2 + WP3 ─→ WP4 smoke + profile + diagnostic + 预算签认
                          ↓
                    WP5 formal T0/T1/T2 + 归因诊断 (A1-C/A5-C 条件)
                          ↓
                    WP6 top2 补 seed + H2/H3 + 配对统计
                          ↓
                    WP7 最终冻结 + dev final-few(若合格) + 报告/交接
```

| WP | 主要交付 | 前置 | 放行证据 | 失败时动作 |
|---|---|---|---|---|
| WP0 | ADR-0033、本方案、冻结 config、validator、五模板、registry 登记、阈值签认 | 本方案 | docs gate CLEAN、schema prereg 通过、阈值 frozen 签认 | 不进入实现或正式训练 |
| WP1 | `BackboneOutput`、`DirectMoveDecoder`、PROCESS 接线、A0 adapter + 显式映射 + strict load、6-checkpoint parity | WP0 | CPU bitwise + GPU 1e-6/1e-5 + 真实数据 + evaluator parity + `q==order_score` | 阻断候选训练（`invalid_run`） |
| WP2 | A1–A5 实现、E(3)/置换/reverse/typed-invalid/有限梯度测试 | WP1 接口 | common fixtures 全过、参数断言 ≤70k | 仅修复或轻量重设计，不改科学输入 |
| WP3 | V542 runner（by-id param group、fork_rng 构造、anchor、resume、identity）、profile 支持 | WP1 | resume identity、batch schedule hash、W0/W1 分项测速可执行 | `invalid_run`，不得正式训练 |
| WP4 | smoke 200 ×6、同环境 profile ×6、diagnostic 2,000 ×6（A0–A5）、A0 sanity band、预算回填签认 | WP2+WP3 | H0/H1、学习启动、资源曲线、budget 签认 | 候选 `resource_failure`/`budget_censored`；A0×3 重训先行 |
| WP5 | formal T0（A0×3 + sanity band）、T1（A1–A4 seed17）、T2（A5 条件）、T-ATTR（A1-C/A5-C 条件）；**顺序冻结：T0 先于任何 T1** | WP4 | run ledger、固定 updates、H0–H2 逐 seed 判定、`V542-A0-sanity-band` pass | 保留科学失败/删失；不换 seed、不降阈值；sanity breach 严格阻断 T1 |
| WP6 | T3 top2 seeds 29/43、H2/H3 判定、seed-averaged paired bootstrap、Holm 校正 | WP5 | 全部预注册指标含失败项同报、校正后 CI | 无人合格 → A0 留任（§12） |
| WP7 | Backbone\* 冻结（或 A0 留任）、dev final-few（若合格且解封获批）、registry/report/bundle 交接 | WP6 | 只读最终报告、声明边界、strict-load 交接件 | 结果降级登记；不追调 dev |

## 3. WP0：治理、接口与预注册冻结

1. ADR-0033 + 本方案 + 冻结 config（`pes2ts_v542_campaign_v1`，由 fix-2 车道产出）。
2. validator `v542_config_schema.py`：prereg 模式必须通过；`--formal` 在 budget 未回填时 typed block（`V542_BUDGET_NULL`）；阈值 null / signoff 非 frozen / gv 不镜像 / run matrix 越界均 typed rejection。
3. 五模板：`preregistration_signoff` / `budget_policy` / `runplan_pin` / `parity_certificate` / `a0_reuse_decision`。
4. registry：`V5.4.2` 登记（status `proposed`，**不得写 executed/smoke_passed**）+ `render_gv_map.py --write` 重渲染（由主会话统一执行共享登记）。
5. 阈值签认：Oracle 建议值 → frozen（`user_directive_20261029`）。

**WP0 出口：** docs gate `--strict` CLEAN；`v542_config_schema.py`（prereg）通过；registry tests 通过；主会话（owner）签认完成。**正式 T1 仍须** §14 清单全过。

## 4. WP1：统一接口与 A0 迁移

### 4.1 `BackboneOutput`（冻结）

```python
@dataclass(frozen=True, slots=True)
class BackboneOutput:
    z: torch.Tensor        # [G, 64], fp32, tanh 饱和, 与 graph_index 图序一致
    edge_msg: torch.Tensor # [E, 48], fp32, 与 edge_index 列对齐, 最后层, 带 autograd
```

- 只有这两个功能字段；不放 `u`（由统一 Decoder 产生）、不放 `node_h`（诊断走单独 hook）。
- 形状/有限性检查放 `V542Model.forward`（wrapper），不在 dataclass `__post_init__` 做全量 `isfinite`；输入合法性由 wrapper 在调 backbone 前统一执行 `check_forward_view_keys_v3`。
- `edge_msg` 必须是最后表示层输出、不 detach、平移/旋转不变标量；候选内部可重排计算但返回前必须 scatter 回输入边顺序。

### 4.2 `DirectMoveDecoder`（唯一共享 MOVE readout）

- 唯一参数 `w_v = nn.Linear(48,1)`（49 params，与 backbone 共享同一 module 对象）；`a_ref=0.08 Å` 非学习常量。
- 逐字复刻 legacy：`coeff=tanh(w_v(edge_msg))`；`unit = rel/(‖rel‖+0.1)`；`index_add` 到 dst；`u = (a_ref/max(1,degK[key_index]))·u`。
- 入参 5 张量：`edge_msg, pos, edge_index, key_index, degK`（结构上禁止读取监督键）；输出行序 = `key_index` 序；非关键原子结构性零动作；形状断言齐全。

### 4.3 PROCESS 接线

- `ReactionProcessHeadV53MP` 原样（2,178 = progress_head 2,113 + endpoint_head 65），只读 `z`；外部 5 键 `{u,z,q,endpoint_logit,order_score}` 不变；`forward_move` 2 键 `{u,z}` 不变；`q == order_score` 断言。

### 4.4 A0 adapter 与 `w_v` 迁移

- `A0LegacyBackbone(StarEGNNV3)`：子类化 last-message hook 捕获 `message_mlp[-1]` 输出恰好一次（`finally` 移除、不 detach、禁 compile）；核心循环 `super().forward` 原样；legacy `u` compute-and-discard；**不改 legacy 源文件**。
- 共享 `w_v`：`backbone.w_v = decoder.w_v`（同一对象）；state_dict 双键 `torch.equal` 断言；param-group 按 `id`（`core=backbone∪move_decoder` lr 1e-4；`head=process_head` lr 3e-4；W0 冻结 core、W1 joint）。
- 显式 `w_v` 映射 + strict audit：`missing/unexpected/shape/dtype` 全空 → `strict=True` → 逐键 `torch.equal` → 审计 JSON（source SHA、映射表 SHA、目标键集）；禁止 `strict=False` 与运行时 shape 启发式。
- R3 初始化：`payload["model"]` 是 R3 trainer 的裸 backbone 键（37 键）；V542 A0 目标键 `backbone.*(37, 含共享 w_v) + process_head.*(6) + move_decoder.w_v.*(2)`；R3 → backbone identity 映射（**37 个 backbone 键 identity，含声明的 `w_v` 双写例外 `backbone.w_v` ↔ `move_decoder.w_v` 同一 tensor**），process_head 由 anchor 初始化。**映射键数精确表述（gate-0 澄清）**：R3 warm-start 的 backbone-only 映射 = 37 键；联合 checkpoint（V541 E0 / V53MP C）的 A0 strict-audit 映射 = 37 backbone + 6 process head = **43 joint 键** identity + `w_v` 双写 —— 与 `pes2ts_core/pesn/models/v542/loader.py` 实际行为一致（该 loader 对联合 checkpoint 映射全部 43 键）。

### 4.5 前向 parity 证书（与训练可比性分开）

6 checkpoint **全 required**（3 V541 E0 S17/S29/S43 **主证据**（`runs/` 下 37,900-update run 的 **best inner-val checkpoint**，`source_tier=v541_e0_runs_final_37900`，`run_updates=37900`，`checkpoint_update=37000/36400/37800`）+ 3 V53MP C S17/S29/S43 **辅助证据**（`outputs/pesn/v53mp/` S3 formal 20k-tier 15,400-update run 的 best inner-val checkpoint，`source_tier=v53mp_20k_s3_formal_15400_auxiliary`，`run_updates=15400`，`checkpoint_update=15400/15300/15400`；`training_updates` 仅为 `run_updates` 的单义别名，永远不是 checkpoint update，`checkpoint_update <= run_updates` 由 schema 强制（gate-0 A6）；非 smoke、非 37,900、不在 `runs/`；path+SHA-256 已于 2026-10-30 由只读 lane 核实并钉入 config `backbone_contract.a0_migration.parity.assets` 与 parity 模板；`outputs/pesn/v541_embed_50k/smoke/**` 与非钉死路径**禁止**）：V541 E0 S17=`c1e35610…ca56d` / S29=`a214d6e9…29e49d` / S43=`92412a1a…6f00c`（`outputs/pesn/v541_embed_50k/runs/V541_T0R_E0_S##/checkpoint_best.pt`）；V53MP C S17=`871596e6…c5db3` / S29=`d456050e…06884` / S43=`d6b11a20…715b`（`outputs/pesn/v53mp/v53mp_20k_C_s##/checkpoint_best.pt`）。流程：strict-audit load → CPU bitwise（5 输出 + `forward_move`）→ ≥8 真实 inner-val batch（覆盖 start/pre_TS/post_TS/endpoint、forward/reverse、大小图、孤立原子）→ GPU tolerance atol 1e-6 / rtol 1e-5 **硬上限**（关 TF32；run-to-run 地板仅作诊断，floor > cap ⇒ GPU parity inconclusive，CPU bitwise 仍是决定证据，不放宽容差；记录环境 SHA）→ `q==order_score` → evaluator parity（V542 evaluator 重算 vs V541 存档，bitwise 或 ≤1e-9）。**该证书只证明前向/评估迁移正确，不作为 E0 复用或训练可比性证据。**

### 4.6 A0 迁移风险与防护（WP1 pre-flight checklist）

| 风险 | 防护 |
|---|---|
| `w_v` 双归属/双更新/进错组 | 共享同一 module；identity 去重；按 id 分组；W0 冻结含 `w_v`；断言 unique 49 |
| 构造 RNG 漂移 | 整体 `fork_rng`；head 用 anchor；构造后 `set_seed(run_seed)` |
| hook 多次/残留/被 compile 跳过 | `finally` 移除；恰好一次；禁 compile；局部 list |
| `edge_msg` 与输入边错位 | 输入边 shuffle + 同步 shuffle → `u` 不变；真实数据 parity |
| `z` 被重算/cast | 原样返回；dtype/device 断言；evaluator parity |
| 白名单只在 legacy 内执行 | wrapper 统一执行；运行时键访问 hook；静态审计 |
| 把前向 parity 当训练可比 | 双证书分离；重训 A0×3；sanity band |

## 5. WP2：候选 A1–A5（共同约束先行）

### 5.1 共同约束

- 宽度接口冻结（hidden 48 / z 64 / edge_msg 48 / node 输入 21）；输入视图不变（backbone 内可用**当前帧确定性**几何量 `d/d²/r̂`——A2/A4/A5 明确允许——但不得新学习式 featurization、不得用绝对坐标、不更新坐标、不回写坐标）；backbone position-preserving（不更新坐标）；z head 固定 `tanh(Linear(96→64)([graph_mean(h), key_mean(h)]))` 同名 `z_head`。
- graph cond 注入：A1/A3 由同形 node MLP 注入（不额外加 `W_g`）；A2/A4/A5/A5-C 加 `h_0 = node_proj(...) + W_g(graph_cond)[graph_index]`（`W_g: 5→48`，零初始化）。**bias 按候选冻结并显式记录（2026-10-30 gate-2 Ruling 2；取代旧「288」泛写）**：A2/A4 = biased `nn.Linear(5,48)`（288 params）；A5/A5-C = `nn.Linear(5,48,bias=False)`（240 params/cell）。同一精确总量算术：A5 的唯一 `GRUCell(48,48)` 带 bias（14,112）；A5-C 第一 cell 同 A5（14,112）、第二 `GRUCell(48,48,bias=False)` 无 bias（13,824）。冻结总量（A2 54,896 / A4 55,840 / A5 31,584 / A5-C 53,712）不变且被精确复现；`W_g` 零初始化 ⇒ warm start 两变体一致，差异仅为每 cell 48 params 的容量选择。config `candidates` 行以 `w_g_bias`/`gru_cell_bias` 显式记录（schema exact-match 强制，缺失/漂移均 typed rejection）。
- attention logits 只能来自不变量；无向量通道 LayerNorm/ReLU/dropout；无 cross product。
- warm-start 一律 R3-only + 显式语义映射 + 逐键审计；候选 fresh 参数 seeded `fork_rng`；构造期断言参数上限（backbone active ≤70,000）；`n_parameters` 报 active backbone（排除共享 `w_v`），joint 另报。

### 5.2 候选规格（Oracle §2 配方；设计估算，构造后回填精确值）

| ID | 结构要点 | backbone active (est.) | joint (est.) |
|---|---|---:|---:|
| A0 | 3L legacy Star + adapter（迁移不变量） | **55,392** | **57,619** |
| A1 | 2L Star 同形消息 + 零初始化 `W_d(12→48)`（rbf12 取 edge_attr 前 12 槽） | 41,232 | 43,459 |
| A1-C | A1 + 第 3 层 fresh（**`init_source_id=A1`**——第 3 层用 A1 init stream，深度归因不引入初始化 nuisance），`W_d` 第 3 层零初始化 | 57,264 | 59,491 |
| A2 | 2L 标量 48 + 向量 32 PaiNN-lite；`edge_msg=φ_s(s^L)` 重算；`‖V‖` 加 eps | 54,896 | 57,123 |
| A3 | A1 + 最后层 query attention 2×24（`W_o` 零初始化；空入边安全） | 53,328 | 55,555 |
| A4 | 2L sparse transformer 4×12 + reaction token + Star 同形 edge readout | 55,840 | 58,067 |
| A5 | 1 shared GRUCell T=4 + `W_g`；`edge_msg=m^T` | 31,584 | 33,811 |
| A5-C | 2 个交替 GRUCell T=4 hidden48（容量对照；**第二 cell 用 A5 init stream**（`init_source_id=A5`）；full-dose 归因诊断非 winner） | 53,712 | 55,939 |

归因规则：A1 vs A0 = 深度+距离项；A1-C 分离深度；A3−A1 隔离 query；A5 是机制检验非容量匹配；A2 的向量通道对 MOVE 的作用依赖 `edge_msg` 重算语义（须在 registry/报告明示）。

**A2 初始化（WP0 冻结）**：`V_0 = 0`；`W_s/W_v` 默认 seeded；`φ_u/φ_g` 最终输出零初始化；`W_g` 零初始化。A1 `W_d` 零初始化；A3 `W_o` 零初始化；A4 attention 输出投影零初始化；A5 `W_g` 零初始化。

**per-candidate R3 语义映射表（WP0 冻结，Oracle §3.2）**：A0 = 全部 **37 个 backbone 键 identity**（`w_v` 双写例外 `backbone.w_v` ↔ `move_decoder.w_v` 同一 tensor；联合 checkpoint 的 A0 strict-audit 另含 6 个 process-head 键 identity = **43 joint 键**，与 `pes2ts_core/pesn/models/v542/loader.py` 一致，backbone-only 映射保持 37 键）；A1/A3 = `elem_emb, node_proj, z_head, message_mlp.0/1, node_mlp.0/1`；A1-C = A1 映射集 + 第 3 层 fresh（A1 init stream）；A2 = common 映射集 + 两层同形 `φ_s`；A4 = common 映射集 + `edge_readout = R3 message_mlp.2`（attention/token fresh、输出投影零）；A5 = common 映射集 + shared msg MLP = R3 `message_mlp.0`（GRU/W_g fresh、W_g 零）；A5-C = A5 映射集 + 第二 cell fresh（A5 init stream）。所有 `w_v` 来自 R3；process head 来自 deterministic anchor。

### 5.3 H0 合法性与 E(3) 清单（每候选）

平移/旋转/反射（O(3)，含反射排除轴向量）/边置换/节点置换/非关键零动作/reverse-relabel harness/typed-invalid（NaN/Inf、Z=0/119、缺键、多监督键、形状错）/restore-resume/有限梯度（A2 V=0 不 NaN、A4 空入边不 NaN、A5 四步反传不 Inf）。明确禁止项见 §1.1。

## 6. WP3：runner、身份与恢复

- **param-group 按 id**：`core = backbone ∪ move_decoder`（lr 1e-4）、`head = process_head`（lr 3e-4）；W0 冻结 core、W1 joint；断言每 unique param 恰好一组、无重复 id。
- **构造 RNG 隔离**：整体构造在 `torch.random.fork_rng(devices=[])`；候选 fresh 参数 `seed_init = int(sha256("v542-init|{run_seed}|{init_source_id}").digest()[:8], "big") % 2147483647`（默认 `init_source_id=candidate_id`；**A1-C→A1、A5-C→A5**）；构造顺序冻结：fresh 模块（seed_init）→ R3 映射覆盖 → 特殊零初始化 → 共享 `w_v` decoder → anchor 加载 process head → `set_seed(run_seed)`；process head 由 deterministic anchor 初始化（seed 17/29/43，`fork_rng` 重放 legacy 构造序，算法+SHA 记入 run identity）；记录 seed map。
- **身份**：source tree fingerprint + common-protocol SHA（排除候选特有字段）+ environment SHA + init_mapping/process_head_anchor/statistics_contract SHA + `bootstrap_seed_root` + plan/oracle 设计源 SHA；`source_manifest_sha256` 必填、`code_commit` 可选；runplan/budget/config SHA 冻结后才 formal。
- **resume**：state_dict/optimizer/RNG/sampler counter round-trip；只从匹配 SHA 的 `checkpoint_last` 恢复；崩了的 run 记 invalid，不得换 seed 重来。

## 7. WP4：smoke、profile、diagnostic 与预算签认

- **smoke 200**：A0–A5，seed 17，单路，kill-resume 一次；rank_excluded。A1-C/A5-C 激活前需 conditional smoke 200（非主 smoke 矩阵的一部分，单独记录）。
- **profile 同环境**：40 W0 + 40 W1（前 20 update 预热）；报告 core/wall/VRAM/host RSS/每边 FLOPs；分母 = V542 A0 实测；**禁止用 V541 存档成本表**。A1-C/A5-C 为 full-dose 诊断，激活前须各自 40+40 profile + cost gate + 预算记录（`profile.conditional_backbones`）。
- **diagnostic 2,000（400+1,600）**：A0–A5 全做；A0@2000（`V542_DIAG_A0_S17`）是同剂量对照。**仅工程淘汰**：H0/H1 任一失败；loss/梯度非有限；best inner-val total loss 相对初始诊断评估下降 <5%；dead fraction > A0@2000+0.20；RMSE/zero 相对 A0@2000 退化 >10%。其余一律进 full-dose；**不得套用 H3 的 3% 门**；诊断指标不进 winner 选择。
- **A0 sanity band（`V542-A0-sanity-band` 硬门，fix-5 C5）**：T0 A0×3 formal 完成后，以 **A0×3 三 seed 均值 vs V541 E0×3 三 seed 均值**（seed-mean vs seed-mean，非单 seed 检查点）判定：RMSE/zero ≤2% 相对、AUPRC/q ≤0.01 绝对。**T0 A0×3 + band pass 必须在任何 T1 A1–A4 formal 前完成**（smoke/profile/diagnostic 可先行）；breach 时：(1) 严格阻断全部 T1 formal 候选启动（不得带 breach 启动候选）；(2) 逐项迁移审计 parity 证书 / process-head anchor / R3 映射 / sampler / evaluator / code / data / initialization；(3) 若发现缺陷 → 修复后**按缺陷类别 bump 对应身份层**（gate-0 A8：代码缺陷 → bump `source_tree_fingerprint`；数据缺陷 → bump `data_manifest_sha256`；初始化缺陷 → bump `init_mapping_sha256`，anchor 牵连时 bump `process_head_anchor_sha256`）并重跑 A0×3 formal（旧 A0×3 记 `invalid_run`，不混比）；(4) 清洁 A0×3 上 band pass 后才恢复 T1；(5) **不得签 sanity exception 带 breach 继续**。
- **预算签认**：W0/W1 分项测速后写 `budget.per_candidate_core_time` 并重哈希；`--formal` 此前 typed block。

## 8. WP5：formal T0/T1/T2 与归因诊断

**执行顺序冻结（fix-5 C5）**：T0 `V542_A0_S17/S29/S43` formal + `V542-A0-sanity-band` pass → T1 A1–A4 seed17 formal → T2/T3（WP6）→ T-ATTR；smoke/profile/diagnostic 可在 T0 前先行（技术测量）。

### 8.1 运行矩阵（预注册 26 行）

| 组 | runs | 说明 |
|---|---|---|
| T0 nominal | `V542_A0_S17/S29/S43` | matched A0×3 重训（必跑；`winner_eligible=false`，基线） |
| T1 nominal | `V542_A1..A4_S17` | 经 diagnostic + cost gate 后 full dose（400+37,500）；`winner_eligible=true` |
| T2 conditional | `V542_A5_S17` | 仅当 A1–A4 top2 三 seed 最终 H3 无人合格（且 A5 自身 diagnostic/cost/H0/H1 全过）自动激活；否则 `gate_skipped` |
| T3 conditional (top2) | `V542_A1..A4_S29/S43`（8 行，WP6 执行） | **top2 只属 A1–A4**；满足 `seed17_screen_pass` predicate 者按冻结 screen score 取最多 2 个（≤4 runs）；A0_S17 complete **且 A0×3 sanity band pass（`V542-A0-sanity-band`）**是 freeze 前置 |
| T3 conditional (A5 supplement) | `V542_A5_S29/S43`（2 行） | **与 top2 完全拆分**；条件=`a5_full_dose_complete_and_seed17_screen_pass`（A5 full-dose 完成 + A5 seed17_screen_pass）；≤2 runs |
| T-DIAG diagnostic | `V542_DIAG_A0..A5_S17`（6 行，WP4 执行） | 2,000 步粗筛；不排名 |
| T-ATTR diagnostic | `V542_A1C_S17`、`V542_A5C_S17` | 归因对照（选择结果确定后、WP7 冻结前执行）；激活前需 H0 fixture + conditional smoke 200 + 40+40 profile + cost gate + 预算记录；A1-C 激活条件 = A1 进 top2 集**或**被选 Backbone\*（skip=`a1_not_in_top2_and_not_selected`）；A5-C 激活条件 = **仅** A5 被选 Backbone\*（skip=`a5_not_selected_backbone`；若 A5 H3 合格但未选，记 capacity caveat unresolved，不跑）；均非 winner-eligible |

**`seed17_screen_pass` predicate（冻结筛选谓词，非三 seed H3 声明）**：run done 且全部预注册指标有限；H0/H1/diagnostic/cost≤2 全过；matched A0_S17 已完成；seed17 H2 全 guard 无 breach；point effect 至少一路，**显式复用 route_logic 括号逻辑**——PROCESS `(auprc_pass AND q_non_inferior) OR (q_pass AND auprc_non_inferior)`（auprc_pass = ΔAUPRC≥0.02；q_pass = Δq≥0.03；auprc_non_inferior = AUPRC 下降 ≤0.01；q_non_inferior = q 下降 ≤0.02），或 MOVE `(direction_pass AND rmse_non_inferior) OR (rmse_pass AND direction_non_inferior)`（direction_pass = direction 相对 ≥3% 且绝对 ≥0.01；rmse_pass = RMSE/zero 相对改善 ≥3%；rmse_non_inferior = 退化 ≤2%；direction_non_inferior = 退化 ≤2%）——非劣条件只约束所在支，**不得悬空**。满足者按冻结 `screen score` 排序（tie = inner-val total loss 再 core-time），最多 2 个 A1–A4 候选；可少于 2；不替补。

`N_core = 3 + n_T1_seed17(≤4) + I(A5_full_dose) + n_diag2000(=6) + 2·n_top2_selected(≤2 candidates) + 2·I(A5_supplement) + I(A1C_activated) + I(A5C_activated)`；这是**新训练 run 计数**，smoke/profile 是技术测量**不进 N_core**；**按公式计数，不按 listed runs 求和**。T3 上限：`top2_max_runs=4` + `a5_supplement_max_runs=2` → `t3_max_runs=6`。

### 8.2 checkpoint 与选择（只看 inner-val）

- checkpoint 始终仅按 inner-val total loss（`val_every=100`）；禁止按 H3 指标在多个 checkpoint 中挑选。
- `seed17_screen_pass` 候选（A1–A4）按冻结 `screen score = max(direction_rel/0.03, rmse_rel_improve/0.03, delta_auprc/0.02, delta_q/0.03)` 排序，tie = inner-val total loss 再 core-time；最多 2 个补 seeds 29/43；**top2 freeze requires A0_S17 complete + A0×3 sanity band pass（`V542-A0-sanity-band`）**。screen score 只用于选择，不进报告主表。

## 9. WP6：top2 补 seed、H2/H3 与配对统计

- **H2（继承 V541，每正式 seed 都过）**：RMSE/zero **退化** ≤2% 相对 R3 且**退化** ≤2% 相对同 seed matched A0；sign 绝对下降 ≤0.01；dead 绝对增加 ≤0.01；fixed-coverage direction 相对退化 ≤2%；R3 参照必须同 evaluator/环境可重评，否则 T1 阻断（不得降级为仅 matched-A0 参照）。任何收益不豁免 H2。
- **H3（对照 matched A0，非 R3）**：
  - PROCESS 路线（括号逻辑）：`(auprc_pass AND q_non_inferior) OR (q_pass AND auprc_non_inferior)`——auprc_pass = ΔAUPRC ≥+0.02；q_pass = Δq ≥+0.03；auprc_non_inferior = AUPRC 下降 ≤0.01；q_non_inferior = q 下降 ≤0.02。
  - MOVE 路线（括号逻辑）：`(direction_pass AND rmse_non_inferior) OR (rmse_pass AND direction_non_inferior)`——direction_pass = fixed-coverage direction 相对改善 ≥3% **且**绝对 +0.01；rmse_pass = RMSE/zero 相对改善 ≥3%；rmse_non_inferior = RMSE/zero 退化 ≤2%；direction_non_inferior = direction 退化 ≤2%。非劣条件只约束所在支，两支各自括号闭合（消除「或」后条件悬空；结构化 block = config `evaluation_contract.route_logic`）。
  - ≥2/3 seeds 同向（方向 guard，`{min_seeds_same_direction: 2, total_seeds: 3}`）；每 seed 无 H2 breach；三 seed **seed-averaged curve-level paired bootstrap 10,000 draws**，95% CI 下界 >0；3% 阈值用 seed 平均 point estimate。
  - **bootstrap 冻结**：`bootstrap_seed_root = 20261029`；每 endpoint/seed 子 seed = `int(sha256(f"v542|{root}|{seed}|{endpoint}").hexdigest()[:16], 16) % 2147483647`；final draws = 10,000；**CI 与 p 用同一组 paired cluster bootstrap draws**；禁止 frame-level 重采样。
  - **p 值（one-sided improvement）**：improvement = direction/AUPRC/q 的 candidate−A0、RMSE 的 A0−candidate；`p = (1 + #{seed-averaged paired bootstrap improvement ≤ 0}) / (10001)`。
  - **Holm family（冻结）**：MOVE direction exit 与 MOVE RMSE exit 各 α=0.025（出口间 Bonferroni），各自对 eligible candidate 集内做 Holm；PROCESS 单一 family，对 2 endpoints × eligible candidates 的 p 做 Holm，family α=0.025。eligible = 完成三 seed 且每 seed H2 无 breach 的 winner-eligible candidate（A5 仅当三 seed 完成）；diagnostic variants（A1-C/A5-C/A0）不入。
  - qualification：冠军候选须 adjusted p < 对应 α 且效应/非劣条件成立；报告同时给未校正与校正后结果。
  - **最终 Backbone\* 选择（`final_selection_rule`，fix-5 C1 冻结）**：候选资格 = H3-qualified only（A0/diagnostic 不参加）；恰 1 个 → 选它；0 个 → A0 留任（§12）；多个 → 冻结字典序：(1) route priority：MOVE-qualified 优先于 PROCESS-only；(2) qualifying 确认性 endpoint 的 Holm-adjusted p 最小值较小者；(3) qualifying endpoint 的 effect margin 最大值较大者（MOVE direction = min(relative_effect/0.03, absolute_effect/0.01)；MOVE RMSE = relative_improvement/0.03；PROCESS AUPRC = delta/0.02；PROCESS q = delta/0.03）；(4) `core_time_ratio` 较小；(5) active backbone params 较小；(6) inner-val total loss 较小；(7) `candidate_id` 字典序较小。**不使用 dev**；报告所有 H3-qualified 候选；完整 rule block 进 statistics/final-selection SHA 与 run identity；A5-C activation 仅用被选 candidate（`a5_selected_backbone`）。
  - **H3 final requires matched A0 对应 seed（S29/S43）complete**。
  - PROCESS-only 可晋级但报告必须明确 MOVE 未改善。
- bootstrap 细节：curve/reaction-cluster 配对（同 cluster 同帧集合、两模型同一组重采样索引）；未定义 cluster 两模型同时剔除并报告剔除数、不插补。

## 10. WP7：最终冻结、dev final-few 与交接

- Backbone\*（或 A0 留任）由 §9 校正后门 + `final_selection_rule` 冻结字典序决定（不用 dev；报告所有 H3-qualified 候选）；三 seed 内部账本、校正后 CI、成本 Pareto、失败/删失全账同报。
- dev 仅对最终冻结的少数模型在 backbone 冻结后评估一次（访问账本）；任何 dev 选择/调参使 confirmatory claim 作废；confirm 永封；tau 不在 dev 重校。
- registry 状态推进与只读最终报告由 owner 签认；交接件可 strict-load。
- 给 V5.4.3：冻结的 backbone 接口、Backbone\* checkpoint 身份、matched A0 对照口径、tau/评估身份、未决归因项（如 A1 未晋级）。

## 11. 产物与验证矩阵

| WP | 产物（路径/形态） | 验证证据 | 验证方式 |
|---|---|---|---|
| WP0 | ADR-0033、本方案、config `pes2ts_v542_backbone_50k.yaml`、`v542_config_schema.py`、五模板、registry 条目 | docs gate CLEAN；schema prereg 通过；阈值 signoff frozen | `docs_consistency.py --strict`；`v542_config_schema.py`（prereg/formal）；registry tests |
| WP1 | `pes2ts_core/pesn/models/v542/`（BackboneOutput/Decoder/A0 adapter）、映射审计 JSON、parity 证书 JSON | 6-checkpoint CPU bitwise + GPU 1e-6/1e-5 + 真实 batch + evaluator parity + `q==order_score` | parity harness + 证书模板实例化 |
| WP2 | A1–A5 模块、common fixtures、H0 报告 | E(3)/置换/reverse/typed-invalid/有限梯度全过；参数 ≤70k 断言 | fixture 测试套件 |
| WP3 | V542 runner、identity/resume 模块、seed map | resume identity、batch schedule hash、param-group 审计 | round-trip 测试 + 审计 JSON |
| WP4 | smoke ×6、profile ×6、diagnostic ×6 账本；budget 实例 | H0/H1、学习启动、资源曲线、A0 sanity band、成本分母实测 | run ledger + budget_policy 实例 SHA 冻结 |
| WP5 | T0/T1/T2/T-ATTR run 记录 | 固定 updates、逐 seed H0–H2、outcome 六词 | runplan.pin 实例 + ledger |
| WP6 | T3 run 记录、H2/H3 判定、bootstrap/Holm 输出 | 校正后 CI、全部预注册指标（含失败）同报 | 统计脚本调用冻结 evaluator |
| WP7 | 只读最终报告、Backbone\* 交接件、registry 状态推进 | 声明边界、strict-load、confirm 未读账本 | owner 签认 + bundle 审计 |

## 12. 负结果表达（冻结写法）

- **无 H3 合格候选**：outcome `scientific_failure`（候选）/ 战役结论 `no_backbone_winner`；结论句为「A0（matched baseline）留任」；不得写成「A0 胜出」超出对照语义，也不得暗示 A0 优于 V541 E0（E0 仅历史/parity 参照，未参与对照）。
- **A1 未晋级**：负结果写「该轻量 GeoEGNN 配方未在本剂量/本合同下超过 A0；深度/距离算子归因未解决」；A1-C 记 `gate_skipped(a1_not_in_top2_and_not_selected)`；**不事后补跑 C2 或任何未预注册对照**。
- **A5 未激活**：记 `gate_skipped(no_A1_A4_h3_qualified_candidate_after_top2_seeds)`；A5 smoke/profile/diagnostic 结果只作工程证据，不进排名、不作科学结论。
- **A5 未被选**：若 A5 H3 合格但未被选为 Backbone\*，A5-C 记 `gate_skipped(a5_not_selected_backbone)` 并登记 `capacity_caveat_unresolved`；不事后补跑。
- **成本超限**：记 `gate_skipped_cost_ratio_exceeds_2`（科学结果保留、不晋级）；**不是** `resource_failure`；不得减层/减帧/缩 update「修」成本。
- **H2 breach**：记 `scientific_failure`；即使 H3 point estimate 为正也不得晋级、不得用 PROCESS 收益豁免。
- **PROCESS-only 晋级**：可晋级，但报告结论必须同时包含「MOVE 未改善」的明示句。
- 通用：outcome 只用六词表；每个 `gate_skipped` 必须带 reason code；报告列出全部预注册指标（含失败项），不得只报显著者；负结果不补跑、不降阈值、不换 seed、不延 dose；决策账本 append-only。

## 13. 反事后调参纪律

WP0 先冻结后看结果（改动 = 新 campaign + 重签 + 受影响 run 重跑）；不 swapping 主指标；不按 H3 选 checkpoint；不做 seed shopping；不做 dose extension；不混用 comparator（H2/H3 对照固定 matched A0×3）；不按 dev 调架构/阈值；tau 只在 inner-val 校准一次；不做 subgroup rescue；多重比较必须处理；非劣界冻结；评估器/环境冻结进 run identity；决策账本 append-only；代码改动即 bump fingerprint、旧 run 不混比。

## 14. 启动前最终签认清单（formal T1 前）

1. `v542_config_schema.py --formal` 通过（budget 已回填、阈值 frozen、gv 镜像）。
2. H0 legality 全过（A0–A5 fixtures + smoke）。
3. H1 resource 全过（单路、无 OOM、host/VRAM 预算内）。
4. smoke 200 ×6 通过（含 kill-resume）；A1-C/A5-C 激活前 conditional smoke 200 通过。
5. 同环境 profile ×6 完成（+ A1-C/A5-C conditional profile），A0 分母实测，cost gate 判定完成。
6. parity 证书（6 checkpoint 全 required，V541 E0 `runs/` 37,900-update run 的 best inner-val checkpoint 主证据（checkpoint update 37000/36400/37800）+ V53MP C 20k S3 formal 15,400-update run 的 best inner-val checkpoint 辅助证据（checkpoint update 15400/15300/15400），path+SHA 已钉死；smoke/非钉死路径禁止；GPU 1e-6/1e-5 硬上限）签发；**T0 A0×3 formal 完成且 `V542-A0-sanity-band` pass（A0×3 seed 均值 vs V541 E0×3 seed 均值；breach 严格阻断 T1、迁移审计、按缺陷类别 bump 对应身份层（代码→`source_tree_fingerprint`、数据→`data_manifest_sha256`、初始化→`init_mapping_sha256`/anchor 牵连→`process_head_anchor_sha256`）后重跑 A0×3、禁止 exception）**。
7. runplan/budget/config/common-protocol/**source_manifest** SHA 冻结（`code_commit` 可选）；**`a0_reuse_decision.json` 实例**（`outputs/pesn/v542_backbone_50k/plan/a0_reuse_decision.json`）存在且 = `retrain_a0_x3`（模板不满足此条；WP0 时点实例按设计缺席，T1 前必须生成签认）。
8. 主会话（owner）签认（含 WP0 治理/统计语义作者记录 + owner re-affirm；statistics/evaluation signoff = `main_session_owner`，`oracle_gate0_reviewer` 仅为 reviewer 非签认人）；confirm/sealed/QC 位保持 false/0。
