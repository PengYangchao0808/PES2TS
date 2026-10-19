# PES2TS / PES-N V5.1a 修复冲刺与扩量准入方案 v1

> 日期：2026-10-19
> 层级：G3.P（V5 线）；G–V 映射：`config/registry/g_v_map.yaml` @ 2026-10-18.1
> 前置：`docs/plans/PES2TS_V5.0-V5.2_十小时冲刺训练与验证执行方案_v1_20261018.md`（V5 sprint phase-1 已执行，状态 `complete_oof_1seed`）
> 决策记录：ADR-0026（key-atom 位移合同与 OOF 语义）；ADR-0027（幅度解耦与参考架 fallback，条件启用）

## 0. 对上一版 V5.1 草案的三处方案级纠正（本方案的前提）

### 0.1 V5.1 与 V5.2 不得混装

按既有 G–V 映射：V5.1 = 架构/消融/修复；V5.2 = 冻结配方 + 3-seed × 5-fold OOF + 封存评测。上一版草案把 15 runs 的 3-seed OOF 放进了 V5.1，属治理混乱。本方案拆分为：

| 阶段 | 定位 | 主要产出 |
|---|---|---|
| **V5.1a**（本方案主体） | 修复冲刺 | 诊断、2×2 消融、5 折确认；**不做正式 OOF** |
| **V5.1b / V5.2** | 正式验证 | 15 runs、3-seed OOF、完整 Gate、封存决策 |

### 0.2 不得靠降低门槛"通过"

V5.1 全程保持 G5.1–G5.4 原阈值不变（见 §5.1）；另设更高的 stretch target。禁止把"没达到原门"改写成"达到了新门"。原门定义以 V5.0–V5.2 方案 §6 为准。

### 0.3 先诊断，再决定是否改架构

V5 幅度比 α≈0.52 至少有三个可能来源，未经诊断不得直接上双头架构：
1. 3000 步 endpoint 过训练（最佳 checkpoint 在 500–1250 步，但未保存）；
2. 标签本身的 d_eff/cap/terminal 分布（部分样本真实步长小于 d0）；
3. 模型确实在学条件均值（方向对、幅度缩）。

诊断决策树（D1，见 §3.1）：
- `α_best ≥ 0.85` → 主要问题是 checkpoint 选择，先不加 magnitude head；
- `0.70 ≤ α_best < 0.85` → 先做 post-hoc scale calibration，再决定是否加头；
- `α_best < 0.70` → magnitude head / log-mag loss 有明确必要性。

## 1. 四个病根的简单 ML 解释

| 病根 | 简单 ML 解释 | 对策 |
|---|---|---|
| 幅度缩到 0.52 | MSE/L2 的最优解是条件期望；方向不确定时模型输出"折中且偏短"的平均向量 | 方向与幅度解耦；log-magnitude loss；先做 calibration 诊断 |
| G5.3 符号/SRE 失败 | 模型只被训练拟合绝对坐标，没有被直接考核"键长变长/缩短"的派生量 | exact Δd pair regression + sign margin + dead-zone 惩罚 |
| 3000 步退化 | 40 训练簇、59k 参数，反复刷几十遍后开始背具体分子 | 32/8 inner-val，每 50–100 步验证，保存 best checkpoint |
| 8.6% gauge_unresolved | 共线/对称结构构不出唯一参考架，系统直接丢样本 | ADR 级 fallback 设计；不得用全局轴绕过合同 |

一句话：方向已经学会了；现在要教它"走够远"和"键往哪边变"。

## 2. V5.1a 定位与总原则

**定位：修复冲刺，不是扩量冲刺，也不是正式 OOF 冲刺。**

目标优先级：
1. 解决 3000 步 endpoint 退化；
2. 幅度比从 ~0.52 拉到 0.85–1.05；
3. G5.3 符号/SRE 拉入可用区间；
4. 真实幅度下重新验证 G5.4；
5. 为 V5.2 的 15-run 3-seed OOF 冻结配方。

总原则（硬约束）：
- 不改 5 折 split（split_manifest hash 不变）；
- 不引入新反应簇；
- 不用 outer fold 调参或早停；
- 不打开 Confirm/OOD（sealed_eval 保持 REFUSED）；
- 不把 Fit200/500 混进本轮；
- 所有新损失、新头、新 fallback 必须先过 G5.1 fixture；
- 所有阈值在训练前冻结（`outputs/pesn/v5_1/protocol_v51.json`，先于任何 V5.1 训练写入）。

## 3. V5.1a 训练内容

### 3.1 诊断阶段（先行，回答四个问题）

**D1：best checkpoint 的 α 是多少？**
- 重跑 A05 fold0 seed17，最多 3000 步；
- 每 50 步在 inner-val 上评估；
- 保存 `checkpoint_best.pt`；
- 在 best checkpoint 上计算 α = ‖u_pred‖/‖y_teacher‖；
- 分层：d_eff≈d0、terminal、capped、nearTS、per-cluster；
- 按上文决策树分流。

**D2：G5.3 的失败在哪里？**
对 OOF 预测按 |Δd_true| 分桶：<0.005（死区/噪声）、0.005–0.01（弱信号）、0.01–0.02（中等）、>0.02（强信号）。分别统计符号准确率、|Δd_pred|<0.005 弃权比例、SRE、成键/断键方向翻转。
- 强信号区符号准确率 >0.9 而弱信号区差 → 死区/幅度问题，pair margin loss 优先；
- 强信号区 <0.8 → 表示/特征/标签问题，不能只加 loss。

**D3：hard clusters 是"难"还是"脏"？**
对 RXN_1675、RXN_2309、RXN_238、RXN_1484 建诊断卡：d_eff/terminal/capped、α/cos/sign/SRE、branch/family/reference gauge 状态、近邻冲突、family 系统性。不直接加权；先判标签问题 vs 模型问题。

**D4：8.6% gauge_unresolved 分布？**
按 family、原子数、线性/平面对称性、关键原子 hop 数统计；评估二阶拓扑邻居 fallback 可行性。主惯性轴 fallback 需新 ADR（ADR-0026 明确禁止 global axis fallback），不能直接写进 V5.1 代码。

### 3.2 模型：Direction–Magnitude 解耦（条件启用，仅当 D1 判定 α_best<0.70）

方向头：复用 Star EGNN 等变向量通道；`v_hat = v/sqrt(‖v‖²+eps²)`，eps=1e-4 Å；零向量样本返回零向量 + degenerate flag，不伪造方向。
幅度头：不变标量特征经 MLP+Softplus，每关键原子一个 s_i>0；初始化 s_i≈0.05 Å；`u_i = s_i·v_hat_i`。幅度头为不变量、方向头为等变量，G5.1 分别测旋转/反射/置换。
备选（先简后繁）：per-atom 幅度头不稳则退回 graph-level scalar S，`u_i = S·v_hat_i`。

### 3.3 损失：三项 pair 损失，不是单一 hinge

`L_pair = L_pair_reg + λ_sign·L_sign + λ_dead·L_dead`：
- 回归项：`Huber((Δd_pred − Δd_true)/a_ref)`，保留幅度信息；
- 符号项：仅对 |Δd_true|≥0.005 Å 的 active pair，`L_sign = softplus(−sign(Δd_true)·Δd_pred/τ_sign)`，τ_sign=0.005 Å；
- 死区项：active pair 上 `L_dead = softplus((0.005−|Δd_pred|)/τ_dead)`，防预测缩进弃权区。

总损失：`L = L_atom + 0.5·L_pair_reg + λ_cos·L_cos + λ_mag·L_mag + λ_sign·L_sign + λ_dead·L_dead`。
起点超参（非已验证最优）：λ_cos=0.02（沿用 V5）、λ_mag=0.5、λ_sign=0.2、λ_dead=0.1。
前 250 步只开 `L_atom + L_pair_reg`，之后 250 步线性 ramp 辅助项。
约束：pair 项用精确 Δd（不用一阶 J）；不对非关键 pair 施加强 sign loss；每项做 zero-case / finite-gradient 测试并纳入 G5.1。

### 3.4 训练控制：32/8 inner-val + best checkpoint

- 每 outer fold 的 40 训练簇 → 32 inner-train / 8 inner-val，family/size 分层，seed 固定，记录 split hash；
- 每 50 步评估 inner-val；主指标 inner-val cluster-median RMSE/zero；
- checkpoint 选择：在 RMSE/zero ≤ best×1.02 的 checkpoint 中选 SRE 最低，再 tie-break nearTS cos 更高；
- patience 8 次检查，min_delta=0.005；
- 保存 `checkpoint_best.pt` 与 `checkpoint_last.pt`；外折 OOF 只用 best；
- 禁用 outer dev 选 checkpoint；不得以 3000 步 endpoint 作为正式交付。

### 3.5 正式验证（V5.1b/V5.2，不在 V5.1a 内）

V5.1a 冻结配方后：seeds 17/29/43 × folds 0–4 = 15 runs，从零训练，输出 `move_proposal_v5_1_3seed.parquet`。Ensemble：默认同 sample/fold 对 3 seed 的 u_raw 算术平均（spherical Fréchet mean 须预注册）；ensemble 不用于 checkpoint 选择；至少 2/3 seed 过点估计门，ensemble 也要过。

### 3.6 Gauge fallback：条件工作包，不阻塞模型修复

二阶拓扑邻居 fallback 先做可行性与 fixture；主惯性轴 fallback 需新 ADR；新 feature version `v5_1_features_v1`；同一 split；双分母报告（common eligible subset 5,088 vs expanded subset），expanded 更好也不得替换 common 主结论。ADR 未落地则 V5.1a 用原 5,088 样本，但扩量 gate 不通过。

## 4. 实验矩阵与预算

| 阶段 | 内容 | runs | 预计 |
|---|---|---|---|
| 诊断 + B0 | D1 best-ckpt 重跑 | 3–6 | 20–30 min |
| 2×2 消融 | fold0 × 3 seeds × {B0,B1,B2,B3} | 12 | 40–50 min |
| 5 折确认 | 最优配置 × 5 folds × seed17 | 5 | 20 min |
| 正式 15 runs（V5.1b/V5.2） | 3 seeds × 5 folds | 15 | 50–75 min |
| 评测/报告 | — | — | 30–60 min |

合计 35–38 runs，约 3–4 h GPU；代码、单测、ADR、诊断卡另计。

2×2 消融（fold0、3 seeds）：

| 配置 | magnitude head | pair sign/dead loss | 目的 |
|---|---|---|---|
| B0 | off | off | 只加 inner-val，测 checkpoint 效应 |
| B1 | on | off | 测幅度解耦 |
| B2 | off | on | 测化学 pair 项 |
| B3 | on | on | 候选组合 |

5 折确认不满足"所有 fold 不退化"则回 2×2，不直接上 15 runs。

## 5. 量化验收目标（原门不变）

### 5.1 Gate 表

| Gate | 指标 | V5 基线 | V5.1 必须通过 | Stretch |
|---|---|---|---|---|
| G5.1 | fixture | 75 passed | 100% pass，含新头/新损失/fallback | 更全覆盖 |
| G5.2 | 全 OOF 簇级 RMSE/zero 中位 | 0.824 | ≤0.80 | ≤0.75 |
| G5.2 | RMSE/Classic 中位 | 0.707 | ≤0.90，bootstrap 下界>0 | ≤0.75 |
| G5.2 | 优于 zero 簇比例 | 43/49 | ≥0.70 | ≥0.85 |
| G5.2 | cos / nearTS | 0.780 / 0.785 | ≥0.60 / ≥0.50 | ≥0.80 / ≥0.75 |
| G5.3 | 符号中位 / Q25 | 0.750 / 0.724 | ≥0.90 / ≥0.80 | ≥0.92 / ≥0.85 |
| G5.3 | SRE 中位 / Q75 | 0.293 / 0.315 | ≤0.15 / ≤0.25 | ≤0.12 / ≤0.20 |
| G5.3 | 死区弃权率 | 未报 | <10% | <5% |
| G5.4 | 新增碰撞 / 保护边 | 0 / 0（幅度 52%） | 原阈值，恢复幅度下复测 | 全 0 |
| M1 | 幅度 α 中位 | 0.52 | 0.85–1.05 | 0.90–1.05 |
| M1 | α Q25 / Q75 | 未报 | ≥0.70 / ≤1.30 | ≥0.80 / ≤1.20 |
| S1 | 3-seed 稳定性 | 1 seed | ≥2/3 seed + ensemble 过门 | 3/3 seed |

### 5.2 G5.2 全 OOF 口径（G5.2-v51 协议，训练前冻结）

49 簇 OOF 是更强证据，但 V5 正式门是 9 簇，不得事后替换。V5.1 做法：
1. 训练前新建 G5.2-v51 协议，明确使用全 OOF 簇级聚合；
2. 保留 V5 9 簇 gate 为历史/secondary；
3. 全 OOF 要求：49 簇 cluster-median、2000 次 cluster bootstrap、Classic 改善下界>0、至少 4/5 folds 点估计过门、任一 fold RMSE/zero 中位不得 >1.0。

### 5.3 G5.4 真实幅度复测

用 u_raw 直接 patch（不依赖 bounded）；全部 OOF 样本；原阈值：新增碰撞 ≤0.5%、新增保护边异常 ≤1%、bootstrap 上界 ≤2%、无任一簇 >5%、拒答 ≤5%。幅度恢复后 G5.4 失败 → V5 的"静态安全"是幅度小的假象，必须回去修，不得放宽。

## 6. 交付物

```
config/campaigns/v5_1_sprint.yaml
docs/plans/PES2TS_V5.1_修复冲刺与扩量准入方案_v1_20261019.md   (本文件)
docs/design/decisions/ADR-0027-V5.1幅度解耦与参考架fallback.md   # 仅当做 fallback
outputs/pesn/v5_1/diagnostics/
outputs/pesn/v5_1/runs/
outputs/pesn/v5_1/move_proposal_v5_1_3seed.parquet              # V5.1b/V5.2
outputs/pesn/v5_1/gate_report_v51.md / .json
outputs/pesn/v5_1/protocol_v51.json
outputs/pesn/v5_1/run_records.jsonl
outputs/pesn/v5_1/scale_readiness_v51.json
```

run record 至少包含：arm、fold、seed、updates、best_update、inner_val_score、config/split/feature/checkpoint SHA256、alpha_median、dead_zone_fraction、selection_exposed。

## 7. 扩量准入：四阶门 + 路线图

### 7.1 为什么现在不能扩量
幅度/符号是结构性问题；8.6% gauge_unresolved 随量线性放大；checkpoint 协议未解决；历史 342 对冲突未重扫。

### 7.2 四阶放行准则

- **S0 模型修复门**：G5.1–G5.4 原阈值通过；M1 幅度 0.85–1.05；G5.3 过原门；≥2/3 seed + ensemble 过；G5.4 真实幅度下通过。
- **S1 数据质量门**：gauge_unresolved 经 ADR 级 fallback 降至 <2%；342 对冲突全量分类（context_resolved / mapping_overflow / 真实冲突）；fit50 标签审计；无未关闭 P0。
- **S2 动态微闭环门**：OOF 选 3–5 个代表反应，5–10 步自回归 rollout + 每步 xTB/半经验单点：SCF 不发散、无新增碰撞、无拓扑穿越、能量有限、无明显环畸变。
- **S3 Fit150 学习曲线门**：嵌套 Fit50 ⊂ Fit150，同配方 3 seeds；匹配 updates 与匹配 wall-clock 双口径；≥2/3 seed 相对 Fit50 改善 ≥5%（工程决策值）；G5.3/G5.4 不退化。不满足就停在 Fit50。
- **S4 Confirm/OOD 一次性解封**：S0–S3 全过 + hash 冻结；Confirm 40 簇、OOD 3 族 × 10 簇；只开一次，不迭代 peeking；通过后才考虑 Fit500。

### 7.3 扩量节奏
当前 Fit50/49 簇 → V5.1a（2×2 + 5 折确认）→ V5.1b/V5.2（15-run OOF + 全 Gate）→ Fit150 → Confirm/OOD → Fit500+。
禁止 Fit50 直接跳 Fit500；禁止 V5.1 未过门时开 Confirm/OOD。

## 8. 三天执行排期

| Day | 工作 | 产出 |
|---|---|---|
| 1 上午 | 协议冻结、G5.1 新 fixture、timing probe 重校 | protocol_v51.json、测试通过 |
| 1 下午 | D1–D3 诊断 | diagnostics_v51.json |
| 2 上午 | 2×2 消融（12 runs） | 消融表 |
| 2 下午 | 选配置 + 5 折确认（5 runs） | 冻结配置 |
| 3 上午 | 正式 15 runs（V5.1b/V5.2） | move_proposal_v5_1_3seed.parquet |
| 3 下午 | G5.1–G5.4 + M1 + 3-seed 统计 | gate_report_v51.md/json |
| 3 晚 | 扩量 readiness 评审 | scale_readiness_v51.json |

gauge fallback 需新 ADR 时单列 1–2 天；不阻塞 V5.1a 模型修复，但阻塞 S1/S3。

## 9. 结论

V5.1a 的核心不是"再加一个头"，而是四颗螺丝：inner-val best checkpoint（排除 endpoint 伪影）、幅度校准（先 calibration 诊断再决定 magnitude head）、pair 化学损失（exact Δd regression + sign margin + dead-zone）、gauge fallback（先 ADR、再 fixture、再双分母）。
扩量唯一通行证：Fit50 上幅度 0.85–1.05、G5.3 过原门、3-seed 全 OOF 稳定、真实幅度下 G5.4 仍安全。然后 Fit150 学习曲线 → Confirm/OOD → Fit500。
**V5.1 未过门之前，任何 Fit200/500 的启动都是在用算力放大一个尚未修好的归纳偏置。**

## 附录 A. 诊断前置结果（2026-10-19，训练前冻结记录）

D2–D4 与基线重推已在任何 V5.1 训练前执行（`outputs/pesn/v5_1/diagnostics/`）：

1. **基线重推**：49 簇 OOF 全部复现方案引用值——RMSE/zero 中位 0.8239（引用 0.824）、43/49 优于零、RMSE/Classic 0.707、cos 0.7799/0.7850、Classic 改善 bootstrap 下界 +0.246。
2. **α 口径对账（重要）**：方案引用的 0.52 在四种候选口径下均不可复现（全原子 0.8104、rms 比 0.7208、帧均范数比 0.6765、有效原子中位 0.6757）。V5.1 冻结口径为：有效原子（‖y‖>0.02 Å）逐原子 ‖u‖/‖y‖ → 簇内中位 → 簇间中位；该口径下 V5 基线 α=0.6757。决策树阈值（0.85/0.70）不改变，按此口径执行。
3. **D2 判定**：强信号桶（|Δd_true|≥0.02 Å）符号准确率 0.929（>0.9），弱信号桶（0.005–0.01）0.500（<0.8）→ **死区/幅度问题，pair margin loss 优先**。逐桶幅度比 7.73/2.60/1.69/0.81（死/弱/中/强）——典型条件均值回归签名。
4. **D3 判定**：RXN_1675/238/1484 为 hard（模型容量问题），RXN_2309 为 conflicted（同族近重复 + gate-failed branch）；引用的 RXN_167 在 fit50 不存在（已记录，以最差 4 簇替补）。所有 49 簇 d_eff 中位 == d0，无近退化标签证据 → 标签侧基本干净。
5. **D4 判定**：8.62% gauge_unresolved = 480 行、8 个 branch-shard、6 个反应（两族占 720 行）；**8/8 的二阶拓扑邻居候选池全部不过现有 scatter gate（λ2/λ1 共线）** → 纯二跳 fallback 在不放宽 gate 的前提下救不了任何 branch，属 ADR-0027 议题。V5.1a 按原 eligible 样本集继续；S1 扩量门维持封锁。
