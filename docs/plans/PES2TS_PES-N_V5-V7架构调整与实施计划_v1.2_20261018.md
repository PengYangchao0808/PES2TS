# PES2TS / PES-N V5–V7 架构调整与实施计划

> **SUPERSEDED（2026-10-20）**：本 v1.2 的 V5/V6/V7 分工与 V6 progress-decoder 路线已被 [V5–V7 三阶段研究与架构修订方案 v2.0](PES2TS_V5-V7_三阶段研究与架构修订方案_v2.0_20261020.md) 取代。保留为历史设计，不得据此启动 V6.0–V6.2 / V7-A/B/C。关键变化：V5 增加 Move/No-Move 曲线停止；V6 改为 xTB 物理响应 + MLP/CNN 残差修正；V7 改为 cost-aware RL；V5.1-R2 + S1 先行；V5.1b、sealed、Fit150+ 继续关闭。
> 2026-10-21：其 Move/No-Move 曲线停止语义亦由 [ADR-0030](../design/decisions/ADR-0030-V5.2全路径导航与端点停止.md) 的 Endpoint Stop 取代。
>
> 状态：Proposed v1.2（历史；规划与合同草案；不修改业务代码、不启动训练/QC/扩量）
> 日期：2026-10-18
> 层级：G3.P.0–.3（V5 离线学习）、G3.2（V6 决策）、G4.0/G4.1（V7-B/C 闭环与离线 RL）、G2.T（M0.5/M6.5 执行验证）；G4.2 发布门不变
> G–V 映射：`config/registry/g_v_map.yaml` @ `2026-10-18.1`；V5.0–V5.2 / V6.0–V6.2 / V7-A/B/C 的 primary/supporting G 与代码范围见 [G–V 映射表](../design/PES2TS_G_V_MAPPING_v1_20261018.md)
> 输入：用户提供的《PES2TS / PES-N V5–V7：统一控制策略、数学规范与训练实施计划》v1.1、其评审意见、`docs/reports/PES2TS_实现报告_20261018.md`、V4.2–V4.7 合同与 ADR-0020/0021/0022/0023/0024
> 输出：V5/V6/V7 的目标架构、数学合同、数据/模型/训练/执行/评测工作包、门控与迁移方案
> 版本说明：v1.2 在 v1.1 规范与评审意见基础上，吸收了一次独立 red-team 复核；所有对 v1.1 的破坏性改动集中在 §17 的 delta 表，需用户签认后才能冻结。
> 证据边界：V4.7 的负结果只证明“当前全原子位移 + 单帧输入 + 45 组训练”未达门；不证明关键原子动作、`$fix` 执行或 V7 闭环已经可行。本计划中的所有阈值均为待预注册的工程假设。

---

## 0. 一页结论

### 0.1 核心决策

1. **动作空间从全原子 3N 位移收敛为去重关键原子 m×3 位移**：`A = ⋃_{(i,j)∈K}{i,j}`，`U_A = {Δx_i}_{i∈A}`。`K` 只用于定义反应中心、派生键长/键角指标和评价，不再作为执行层的原子对向量。
2. **教师坐标系用“固定分支帧”，不用逐帧 Kabsch**：主标签是已配准 IRC 分支帧内的原始 `Δx_i`；可选“分支起点锚定”版本；逐帧加权 Kabsch 只作诊断，不作为训练标签。原因：逐帧对齐依赖下一帧，且会让关键原子运动参与定义 gauge，造成标签不可复现/衰减。
3. **执行合同改为 Patch Key XYZ → xTB `$fix` → Relax Others**：只覆盖关键原子坐标，非关键原子在种子中保持原值；xTB 用 `$fix` 精确冻结关键原子，优化其余自由度。MVP 不使用 `$constrain atoms` 冒充硬冻结。
4. **V5/V6 严格分离**：V5 只做几何→关键原子位移；V6 只做剩余路程/步长/STOP 决策，是唯一对外决策出口；V7 才引入真实 xTB 执行、OptTS/Freq/IRC 验证和成本学习。
5. **V5 默认模型不再是“全分子 EGNN”**：主候选是**稀疏关键原子方向头 EGNN/MPNN**，但必须带**秩完备向量基**（共价 + 4.5 Å 半径邻居；平面/线性中心补 per-atom 向量通道或局部法向）。Classic、局部框架 MLP、Deep Sets 是必跑基线；PaiNN 只作对照。
6. **V6 输出绝对步长 ℓ（关键原子 RMS，Å），不是无界 α**：`α=ℓ/d0` 只作报告。这样 d0、逐原子 cap、V6 训练目标三者一致，避免“学到 α>1 但执行时被裁剪”的标签/执行错位。
7. **先验证执行合同，再训练模型**：新增 **M0.5 Oracle Teacher Rollout**。用 100% 正确的关键原子教师位移，经 Patch + `$fix` + relax 逐步推进；必须在**同一执行 PES** 和**显式 verified-TS 定义**下评估，不能用“比 Classic 好”这种弱门。
8. **V6 训练分布要对齐在线执行**：V6 主数据使用“固定关键原子、xTB 预弛豫非关键原子”的帧；V5 OOF 提案在同一帧上重算；再加 **M3.5 离线 aggregation/DAgger** 状态，避免只在干净 IRC 上训练。
9. **V7-B 先于 V7-C**：先用真实 rollout 做 STOP 阈值、σ、ℓ 的校准与 filtered BC/AWR；只有积累 ≥2,000 条完整转移且动作覆盖足够后，才启动保守离线 RL。新增 **M8.5** 专门负责这段采集。
10. **数据扩量不是修复可辨识性的手段**：500/2k 独立反应扩量必须在 WP0 身份绑定、family 审计、144 个 identity partner 补绑定等前置条件闭合后启动。

### 0.2 与 V4 的关系

- V4 的 `dual_ts` 全原子位移模型、旧 k×3/全原子 checkpoint、旧 bundle 保留为历史对照，不删除、不冒充 V5 模型。
- V5 使用新的动作/特征/标签/模型版本；旧 checkpoint 不能无审计地加载为 m×3 Actor。
- V4.7 的 negative result 是本计划的出发点，不是被覆盖的结论；V5/V6 的任何收益必须用新协议、新划分、新门控重新证明。

---

## 1. 证据基础与设计修正

### 1.1 V4.7 事实

- 50 个 fit 反应身份簇、100 条分支、5,668 个原生态、5,568 个原生动作；monitor8 是历史开发回归集，不是封存测试集。
- FINAL 三种子 `native_full` 中位：MSE/零动作 1.132、优于零动作组数 0/8、整体 cos 0.180、near-TS cos 0.248；四项门全部失败。
- fit MSE/零动作 0.52–0.78，dev 1.26–1.49；W128 380,181 参数、PaiNN 120,833 参数均未改善 dev；容量不是瓶颈。
- 342 对“当前输入接近、教师方向冲突”的样本；单帧 5 通道输入缺少端点几何、曲率、历史和分支身份。
- 42.3% 的原子动作 <0.01 Å；Top-3 原子承载约 66.5% 的平方位移；MSE 门被近零原子主导，模型可用收缩动作换低 MSE。
- H1 xTB 梯度先验 `Δs_fit=0`，dev cos 为负；xTB 与 DFT IRC 不同源，不能把 xTB 梯度当精确切线。
- 20k 终点不是最优：多个臂在 5k–10k 的 dev 最好；固定 20k 使 encoder 排名不可信。
- **没有跑过真正的非 EGNN 位移基线**：`ENCODERS = ("egnn","painn")`；`baselines.py` 的 MLP/rule 只服务旧的 progress/ranking lane。

### 1.2 本计划的四个新假设

| 编号 | 假设 | 验证方式 | 失败后的处理 |
|---|---|---|---|
| H1 | 少数关键原子的逐原子位移比全原子位移更可执行、更可学 | V5 在未见反应族上的向量误差/键长变化优于 Classic/MLP | 回到全原子或 pair-Δd 动作重新定义 |
| H2 | 完整分子几何对关键原子运动有额外信息 | 全几何 vs 关键子图 vs 仅局部描述符的消融 | 缩小图/特征，不加大模型 |
| H3 | xTB E/g 的**关键原子投影**能为 V6 提供几何之外的增益 | Geometry-only vs +E/g 的 V6 消融 | 去掉 E/g，只保留几何/历史 |
| H4 | 真实 `$fix` + relax 反馈能带来可测的核时收益 | V7-A 同成功率下的 intention-to-treat core-hours/verified TS | 停止 V7-C，回到执行/动作设计 |

---

## 2. 目标架构

```text
                    ┌─────────────────────────────────────────────┐
                    │  ReactionTask (G_R, G_P, M, ΔG, b, K, A, ρ) │
                    └──────────────────────┬──────────────────────┘
                                           │
              ┌────────────────────────────▼────────────────────────────┐
              │ V5  MOVE Encoder  f_θ                                  │
              │  full geometry + key-atom graph + start context         │
              │  output: z_t, U_A^0 ∈ R^{m×3}                          │
              └────────────────────────────┬────────────────────────────┘
                                           │
              ┌────────────────────────────▼────────────────────────────┐
              │ V6  STOP / Decision Decoder  g_φ                       │
              │  z_t, U_A^0, X_t, start, E_t-E_0, g_A, history, ρ      │
              │  output: r, σ_r, ℓ, (p,L diagnostic), (δU_A)           │
              │  rule: STOP if r+ ≤ r_stop; else MOVE(ℓ·unit(U_A^0))    │
              └────────────────────────────┬────────────────────────────┘
                                           │
              ┌────────────────────────────▼────────────────────────────┐
              │ Final Action + Safety Layer                            │
              │  MOVE / STOP / FALLBACK / ABORT                        │
              │  step cap, collision, valence, §fix capability, budget │
              └────────────────────────────┬────────────────────────────┘
                                           │
              ┌────────────────────────────▼────────────────────────────┐
              │ V7  Real QC loop                                       │
              │  patch key XYZ → xTB $fix → relax non-key → E/g        │
              │  STOP → OptTS/Freq/bidirectional IRC → verified TS     │
              │  replay (S,a,S',c,status) → calibration → conservative │
              │  offline RL only after coverage gate                   │
              └─────────────────────────────────────────────────────────┘
```

### 2.1 组件职责

| 组件 | 输入 | 输出 | 不做什么 |
|---|---|---|---|
| V5 | 当前完整几何、R/P 图、A/K、分支/起点、方法身份 | `z_t`、`U_A^0 (m×3)` | 不读 E/g、不输出 STOP、不预测非关键原子 |
| V6 | V5 的 `z_t/U_A^0`、当前几何、xTB 关键原子 E/g、起点、历史、ρ | `r, σ_r, ℓ, (p,L), (δU_A)`；规则化 MOVE/STOP | 不做第二次全分子几何重建；不读未来 IRC/TS/真实 L |
| Safety | V6 动作、当前几何、执行能力 | MOVE/STOP/FALLBACK/ABORT + typed reason | 不静默换物理动作；不放行越界步长/碰撞/不支持的后端 |
| V7 | 真实状态、动作、xTB/验证结果、成本 | replay transition、校准模型、离线 RL | 不用模型自报 p 冒充真实成功；不只记成功尝试 |

---

## 3. 数学合同

### 3.1 状态与动作

\[
\mathcal H=(G_R,G_P,\mathcal M,\Delta G,b,K,A,\rho)
\]

- `K = {(i_l,j_l)}`：成键、断键、键级变化等有向关键对，仅用于定义反应中心与派生指标。
- `A = ⋃_{(i,j)∈K}{i,j}`：去重关键原子，`m=|A|`。
- 动作：

\[
\boxed{U_A=\{\Delta x_i\}_{i\in A}\in\mathbb R^{m\times3}}
\]

- 派生量：

\[
u_{ij}=\Delta x_j-\Delta x_i,\quad (i,j)\in K
\]

`u_ij` 不能唯一决定 `Δx_i, Δx_j`（整体平移规范），因此**不能训练 pair 向量、执行时却当作唯一原子更新**。直接预测每个唯一关键原子的位移，保证同一原子参与多个 K 对时动作一致。

### 3.2 教师坐标系：固定分支帧，逐帧 Kabsch 只作诊断

> 本节对 v1.1 §1.2 的“稳定非反应中心参考骨架”作出修订。v1.1 要求用 `F(X)` 定义局部坐标；但小分子、全反应中心、柔性取代基上 `F(X)` 容易退化。评审建议的逐帧加权 Kabsch 也有两个问题：(a) 对齐依赖下一帧，标签不可由当前帧复现；(b) 关键原子即使低权重，仍参与定义 gauge，可能衰减教师位移。因此 v1.2 改用固定分支帧。

**主标签（v1.2）**：

- 使用仓库已有的“链式质量加权 Kabsch”预处理结果；每条 IRC 分支在一个固定分支帧内。
- 教师位移直接取：

\[
\boxed{\mathbf u_{i,t}^*=\mathbf x_{i,t+\delta}^{\text{branch frame}}-\mathbf x_{i,t}^{\text{branch frame}},\quad i\in A}
\]

- 不做逐帧重新配准；不把下一帧对齐到当前帧。
- 若分支起点 `X_0` 可用，可另存 `branch_start_anchored_v1` 版本：在分支内用固定非关键重原子骨架把各帧对齐到 `X_0` 一次；该旋转对整个分支固定，不逐帧变化。
- `frame_contract_id` 记录：`branch_frame_v1`（主）、`branch_start_anchored_v1`（可选）、`local_frame_mlp_v1`（仅非等变基线）、`diagnostic_weights_kabsch_v1`（只诊断）。

**诊断（不训练）**：

- 逐帧加权 Kabsch 仍可计算，用于报告“刚体修正量/位移量”比例、旋转角、关键原子权重敏感性。
- 若诊断显示 raw 标签中刚体旋转占比很大，先检查 IRC 预处理/分支配准，而不是自动改用逐帧 Kabsch。
- 若必须用逐帧 Kabsch（例如某些轨迹未配准），权重层次固定为：
  1. 只用非关键重原子（key weight 0, H 0）；
  2. 若秩不足，用全部重原子但 key weight 0；
  3. 若仍退化，用局部框架；
  4. 不允许“全部原子均匀权重”作为回退（那会让关键原子权重变大，方向相反）。
- 秩用加权惯性张量判定，不用“原子数 ≥3”判定；记录 `rank`、`rotation_angle`、`correction_norm/displacement_norm`。

**模型侧**：

- **等变模型（EGNN/MPNN/PaiNN）**：输入当前全局坐标，输出天然附着在当前坐标系；不需要显式 `F(X)`。训练标签用固定分支帧的 `u_i*`；推理输出直接执行。
- **非等变 MLP 基线**：用反应中心局部框架 `F_t∈SO(3)`（关键原子主轴 + 次轴 Gram-Schmidt）预测局部分量，再映射回全局；框架退化时回退到全局坐标 + 旋转增强。局部框架只属于该基线，不进入主合同。
- 手性/反射单独测试：不得用普通叉积破坏 O(3)/E(3) 等变；若使用轴向量，必须证明反射变换正确，并登记 SO(3) vs O(3) 语义。

**为什么不用逐帧 Kabsch 作为主标签**：

- 对齐旋转是下一帧的函数；同一当前帧在不同路径/不同下一帧下会得到不同标签。
- 关键原子运动被部分吸收进对齐旋转，教师位移被衰减；在近 TS 的软模方向上尤其危险。
- 执行时没有“下一帧对齐”这一步；标签必须是在当前分支帧里可直接执行的向量。

### 3.3 V5 教师步距：关键原子 RMS 步长 + cap-aware

> 本节对 v1.1 §2.1 的“统一全分子质量加权 Δs0”作出修订。全分子弧长会被旁观原子/取代基运动主导，导致同一 Δs0 下关键原子位移随分子大小剧烈变化；同时 d0、逐原子 cap、V6 的 α 上限必须一致。

定义关键原子 RMS 位移：

\[
d_A(X,X')=\left(\frac{1}{|A|}\sum_{i\in A}\|x_i'-x_i\|^2\right)^{1/2}
\]

V5 教师采样：

\[
\boxed{\delta s_t=\arg\min_{\delta>0}\left|d_A(X_t,\Gamma(s_t+\delta))-d_0\right|}
\]

- 初始 `d0 = 0.06 Å`（关键原子 RMS）；M1 审计后可调，但必须在看 test 前冻结。
- 若到 TS 端点的剩余关键原子 RMS 位移 < d0，取端点。
- 用单调三次插值/线性插值沿 IRC 路径取帧；插值方法、端点规则、单位写入 manifest。
- **cap-aware 教师**：对每个教师步计算 `max_i ||u_i*||`；若超过逐原子 cap（初始 0.12 Å），把整个教师步按 `cap/max` 缩放，记录 `teacher_step_clipped=true` 与 scale。V6 的 `ℓ*` 用缩放后的关键原子 RMS，保证训练标签是执行层可执行的。
- `d0` 与 V4 的全分子质量加权步距不是同一个量；不得混用旧标签。

全分子质量加权弧长仍用于 V6 的诊断/辅助输出：

\[
\Delta s_t=\left[\sum_{i=1}^N m_i\|\Delta x_{i,t}\|^2\right]^{1/2},\quad
s_t=\sum_{j<t}\Delta s_j,\quad L_{r,b}=s_{\rm TS}
\]

\[
r_t^{\rm mw}=L_{r,b}-s_t,\quad p_t^{\rm mw}=s_t/L_{r,b}
\]

- `p^mw/r^mw` 是“从指定分支起点沿参考 IRC 的质量加权弧长完成百分比/剩余量”，不是实验转化率、绝对 TS 接近度或 OptTS 成功概率。
- 部署时未知 `L`，输入必须包含分支起点 `(X_0,q_0,E_0)` 或其摘要；否则 60%/80% 没有唯一含义。
- V5 的 `d0` 与 V6 的 `r^mw` 是两个独立合同；V6 的主输出改用关键原子剩余路程（见 §5.2），`p^mw/L^mw` 只作辅助/诊断。

### 3.4 唯一执行合同

对外动作：

\[
a_t\in\{\mathrm{MOVE}(A,U_A),\mathrm{STOP},\mathrm{FALLBACK},\mathrm{ABORT}\}
\]

第一步：只覆盖关键原子，构造种子 XYZ：

\[
\boxed{
x_{i,t}^{\rm seed}=
\begin{cases}
x_{i,t}+\Delta x_{i,t}^{\rm final},&i\in A\\
x_{i,t},&i\notin A
\end{cases}}
\]

第二步：`$fix` 关键原子，优化非关键自由度：

\[
X_{\bar A,t+1}=\arg\min_{X_{\bar A}}E_{\rm xTB}(X_{A,t}^{\rm seed},X_{\bar A};\rho),\qquad
X_{A,t+1}=X_{A,t}^{\rm seed}
\]

xTB xcontrol 语义：

```text
$fix
   atoms: 3,7,12
$end
```

- 原子索引按 xTB XYZ 的 1-based 行号；由 atom mapping 转换。
- `$fix` 是精确笛卡尔冻结；`$constrain atoms` 是原子间相对距离/角度的势约束，**不得混用**。
- 实际 xTB 版本必须通过小分子冻结回归：固定原子坐标零漂移、非关键原子可弛豫、SCF/优化收敛、日志有 `$fix` 生效证据。历史版本曾有 `$fix` 不生效的回归（grimme-lab/xtb #1073），必须以 ACP 实际二进制测试，不能只看文档。

### 3.5 执行安全层

MVP 默认严格复制非关键坐标；以下为必须在 M0/M6 实现并在 M0.5 做 A/B 的安全措施：

| 措施 | 默认 | 说明 |
|---|---|---|
| 单原子最大位移 | `0.12 Å` | 超过则全体关键原子按比例缩放；记录 `step_clipped` |
| 单步关键原子 RMS 位移 | `≤0.06 Å` | 与 V5 教师 d0 对齐；超限缩放 |
| 键长变化检查 | 涉及关键原子的键长变化 >20% 或 <0.7 Å 触发缩步 | 防止拉断未指定键 |
| 碰撞检查 | 任意原子对 <0.75×(r_cov_i+r_cov_j) 触发缩步 50% 或 FALLBACK | 只做几何检查，不做化学猜测 |
| 价态/电子态检查 | 形式电荷、多重度、原子序数、键级变化 | typed refusal |
| 终端氢随动 | **默认关闭，A/B 实验** | 仅对“只与单一关键重原子成键、自身不在 A”的终端 H，在种子中随母体平移，但不冻结；用于消除 C–H/N–H/O–H 初始撕裂 |
| 后端能力 | `$fix` 不支持/版本不符 → FALLBACK | 不静默改用 `$constrain` |
| 执行后检查 | 固定原子零漂移、非关键原子收敛、连接性/价态/手性、能量突变 | 任一失败 → typed reject，不进入下一状态 |

> 终端氢随动会改变“非关键坐标逐值不变”的严格语义，因此不默认启用。M0.5 的 Oracle rollout 同时跑 strict-copy 与 follower 两个分支；只有 strict-copy 出现可归因于初始键长撕裂的 SCF/优化失败，且 follower 显著改善，才把 follower 升级为默认，并登记为 v1.2 合同变更。

---

## 4. 数据与特征

### 4.1 数据集

| 数据集 | 内容 | 禁止进入输入 |
|---|---|---|
| `IRCGeometryDataset` (V5) | reaction/family/branch/atom-map/K/A/charge/spin/method/trajectory_id；当前完整 XYZ；下一教师帧；固定分支帧元数据；逐关键原子 `U_A*`；`d_A`；cap/scale；split/manifest/checksum | 下一帧/TS 几何、真实剩余弧长、E/g（V5 不读） |
| `IRCDecisionDataset` (V6) | 同一身份；当前几何（主：预弛豫帧；对照：原始 DFT 帧）；起点锚点；当前 q；V5 OOF `z_t/U_A^0`（在对应帧上重算）；关键原子 xTB E/g；历史；真实 `r,σ,ℓ`；`p^mw/L^mw` 辅助；mask；xTB 来源/单位 | 未来 xTB E、TS 坐标、真实 L、未来 IRC 帧 |
| `QCAggregationDataset` (M3.5) | V5 OOF 提案 → patch → xTB relax → 最近 IRC 投影/恢复动作标签；失败/拒绝也保留 | 未执行动作的 outcome 必须 missing |
| `QCReplayDataset` (V7) | 执行前 state、V5 提案、V6 final、patched XYZ、冻结原子索引、真实 xTB next state、execution/attempt id、成本、时长、方法、typed failure、OptTS/Freq/IRC/端点/验证、预算删失 | 未执行动作的 outcome（必须 missing，不能假设失败） |

### 4.2 特征

**V5 节点特征**（关键原子 + 邻域）：

- 原子序数 embedding、形式电荷、杂化/键级统计；
- `is_key_atom`、参与的形成键数/断裂键数、`degree_K`；
- 相对起点锚点的关键键长累计变化；
- 当前局部几何（键长/键角/二面角）与起点差；
- 可选的全局上下文（分子大小、总电荷、方法身份）。

**V5 边特征**：

- 当前距离 RBF；
- 反应图键级变化 `ΔB_ij = B_ij^target - B_ij^start ∈ {-2,-1,0,+1,+2}`；
- 当前距离与起点距离差 `d_ij(X_t)-d_ij(X_0)`；
- Pauling 键级估计 `exp(-(d_ij-d_ij^0)/0.3)`；
- 是否为 K 边、是否为半径边、是否 changed_flag。

**V6 特征**（只使用可观测、与在线分布一致的量）：

- V5 `z_t`（关键原子 token + 全局池化）；
- `U_A^0` 及其范数/方向；
- 关键原子 xTB 梯度 `g_A` 的标量投影：
  - `P_proj = <g_A, U_A^0> / (||g_A||·||U_A^0|| + ε)`；
  - `||g_A||`、每个 K 对的内部张力 `<g_j-g_i, e_ij>`；
  - 不做全分子 N×3 梯度的 GNN 编码；
- `E_t-E_0`、`δE_t`、`δP_proj`；
- 起点锚点的关键键长/能量/电荷/自旋；
- 历史：最近 4–8 步的 ℓ、r_hat、E、P_proj 的汇总统计（先不用 RNN/TCN；数据足够后再加）；
- 方法/电荷/自旋/溶剂 embedding（用于条件化，不做物理外推）。

> **关键分布修正**：离线 DFT IRC 帧在 xTB 下并非非关键原子极小点，`g_nonkey` 很大；在线 rollout 每步刚做完 `$fix`+relax，`g_nonkey≈0`，只有冻结关键原子上的约束反力。因此 V6 不读非关键原子梯度；V6 训练数据优先使用“固定关键原子、xTB 预弛豫非关键原子”的帧（`IRC-PreRelaxed`），原始 DFT 帧单点作为消融。预弛豫每帧约 0.1–0.3 s，5,568 帧量级可接受，但必须计入成本账本。

> **V5/V6 帧一致性**：V6 的当前帧、V5 OOF 提案、教师 `U_A*/r*/ℓ*` 必须在**同一帧版本**上计算。不能“V5 用原始 DFT 帧、V6 用预弛豫帧”混搭；否则是 gauge/标签错位，不是未来泄漏。主版本用 `pre_relaxed_v1`；对照版本用 `raw_dft_v1`，两者分别建 manifest。V5 训练也采用 **raw IRC + pre-relaxed 增广**（同一反应、同一教师路径，标签按投影后的教师帧重算），OOF 模型在同一混合分布上训练；若预算只允许一种，优先 pre-relaxed，因为 V6/V7 的在线状态就在这个流形上。

### 4.3 数据卫生

1. **平缓区裁剪**：IRC 两端关键键距变化率 <5% 且关键原子几乎静止的尾部帧，降权/剔除；保留反应物/产物覆盖，不能只留近 TS。
2. **对称等价原子 canonical matching**：对 `-CF3/-CH3/-Ph` 等对称基团，枚举保持元素/键型的自同构，选关键原子 RMSD 最小的置换；记录 `permutation_id`。**必须在标签冻结前完成**，不能先造标签再匹配。
3. **冲突样本审计**：在关键原子描述符上检索 `RMSD<0.05 Å` 且教师方向 `cos<0` 的对；区分“不同分支/多路径”与“同分支标签噪声”。同分支冲突降权/剔除；多路径样本单独标记，供 V6 多假设/拒绝学习。
4. **单位/坐标/方法身份**：Å、Ha、Ha/Å、amu、电荷、自旋、溶剂、xTB 版本、ORCA 版本全部入 manifest；坐标配准规则与梯度旋转绑定。
5. **旧 342 对不是新标签的结论**：必须用 V5 的 m×3 标签重新审计；不能直接沿用旧全原子冲突统计。
6. **rank 审计**：逐关键原子计算邻居方向矩阵的秩；平面/线性中心单独标记，供 §5.1 的向量基选择与切片评价。

### 4.4 划分与 OOF

- **禁止**按帧、按分支随机划分；同一反应的 R→TS 与 P→TS 共享 TS 几何，必须同簇。
- 反应家族簇：反应中心 SMARTS + DRFP Tanimoto（阈值 0.6）连通分量；再加入 identity partner/duplicate group 绑定。
- 数据分层：
  - `monitor8`：只做冒烟/代码回归，不参与超参、早停、结论；
  - Dev/CV pool ≈70% 家族簇：5-fold Group-K-Fold；
  - In-distribution confirm ≈15%：同大类、全新独立簇；
  - OOD family holdout ≈15%：整族封存（未见机理/子族）。若 OOD 家族 <3 个，只能作描述性报告，不宣称 OOD 泛化。
- V5 OOF：5 折 × 2 seeds；对每个反应，用未见过该反应的模型输出 `U_A^0`；V6 训练只用 OOF。
- 部署：5 折 × 2 seeds 的 10 个 V5 模型做 ensemble；V6 训练时对 OOF 提案注入 `N(0, 0.01²)` 或使用 ensemble 输出，保持训练/部署分布一致；记录 `proposal_distribution_audit`。
- V6 训练/评测绝不能读取 V5 训练反应的真实下一帧；V5 的 OOF 提案是唯一接口。
- **V6 自己的 CV/校准**：V6 也按家族 Group-K-Fold；conformal 校准用独立 calibration split，不能与 early-stopping/dev 共用。否则覆盖率会被污染。

---

## 5. 模型

### 5.1 V5：Geometry MOVE Encoder

**输出**：`z_t` 与 `U_A^0 ∈ R^{m×3}`；只输出关键原子。

**候选与优先级**：

| 优先级 | 模型 | 作用 | 备注 |
|---|---|---|---|
| P0 | Classic key-atom baseline | 由 K 的键长变化率按质量/度分配两端位移；或最小范数 K-bond 解 | 检验 GNN 是否值得存在的“生死线”；必须审计 |
| P0 | 局部框架 MLP | 完整分子几何描述符 + 关键原子 readout；非等变 | 必须与 EGNN 同数据、同划分、同标签 |
| P1 | 稀疏关键原子 EGNN/MPNN（主候选） | A + 2-hop 共价邻域 + 4.5 Å 半径邻居；3–4 层；hidden 64–128；signed 边方向系数 + 秩完备向量通道 | 直接修 V4 的“正锥/单射线”限制 |
| P1 | Deep Sets / 轻量全分子图网络 + 稀疏 key-atom 向量头 | 变长 m；全局上下文 | 与 EGNN 对照 |
| P2 | PaiNN + key-atom vector head | 4–16 向量通道对照 | 只作对照/秩完备补充，不因参数多而默认晋级 |
| P2 | 径向派生动作 vs 完整 3D 位移 | 检验侧向/协同信息 | 输出 m×3，但约束在 K 轴方向 |

**主候选的关键实现约束**：

- 向量输出头：

\[
\boxed{\hat u_i=\sum_{j\in\mathcal N(i)}\phi_{ij}(h_i,h_j,e_{ij})\cdot\frac{x_i-x_j}{\|x_i-x_j\|}+\sum_{c=1}^{C}\psi_{i,c}(h_i)\,v_{i,c}}
\]

  - 第一项是 signed 边方向系数；`φ_ij` 必须是**有符号线性层或带缩放 tanh**，禁止 sigmoid/softplus 非负门控（V4 的正锥问题）。
  - `N(i)` 包含共价邻居 + 4.5 Å 半径邻居（至少对关键原子）；否则方向基不完整。
  - 第二项是秩完备向量通道 `v_{i,c}`（PaiNN 风格或局部法向），仅在 `rank(D_i)<3` 的平面/线性中心启用；`ψ` 为 signed 标量门。
  - **rank 审计**：逐关键原子计算邻居方向矩阵 `D_i` 的秩；若 rank<3 且没有向量通道，则该原子的面外/法向运动不可表示。平面 sp2 的 pyramidalization、线性中心的弯曲都是常见 TS 坐标，必须覆盖。
  - 若用普通叉积构造法向：它是轴向量，反射下不变号；必须明确 SO(3) 语义、做 parity 测试，不能当作 O(3) 极向量随意相加。
- 图规模：关键原子及其 2-hop 共价邻域 + 4.5 Å 空间近邻；远端原子聚合为虚拟基团节点或全局标量上下文；必须做 full-graph vs subgraph 消融，防止环张力/远程立体控制被截断。
- 输出只对 `i∈A` 计算损失；非关键原子不产生监督。
- 参数量控制：初始 3 层、hidden 96；数据扩到 500 反应前不超过 ~150k 参数；不以 W128/PaiNN 作为起点。

**V5 损失**：

\[
\mathcal L_5=\mathcal L_{\text{atom\_vec}}+\lambda_{\text{pair}}\mathcal L_{\text{pair\_dist}}+\lambda_{\text{cos}}\mathcal L_{\text{gated\_cos}}
\]

\[
\mathcal L_{\text{atom\_vec}}=\frac{1}{|A_r|}\sum_{i\in A_r}w_i\,\mathrm{Huber}_\delta\left(\frac{\|\hat u_i-u_i^*\|}{a_{\rm ref}}\right)
\]

\[
\mathcal L_{\text{pair\_dist}}=\frac{1}{|K_r|}\sum_{(i,j)\in K_r}\mathrm{Huber}_\delta\left(\frac{\hat{\Delta d}_{ij}-\Delta d_{ij}^*}{a_{\rm ref}}\right),
\quad
\hat{\Delta d}_{ij}=\langle \hat u_j-\hat u_i,e_{ij}\rangle
\]

\[
\mathcal L_{\text{gated\_cos}}=\frac{1}{|A_r|}\sum_{i\in A_r}\mathbb I[\|u_i^*\|>\tau_{\rm norm}]\left(1-\frac{\langle\hat u_i,u_i^*\rangle}{\|\hat u_i\|\|u_i^*\|+\epsilon}\right)
\]

- `a_ref = d0 = 0.06 Å`；`τ_norm = 0.02 Å`；`λ_pair=0.5`；`λ_cos=0.10–0.15`。
- 不用 V4 H2 的大权重（λ_dir=16311.87、λ_mag=56.20 已证明有害）。
- 按反应/家族聚合，不按帧平均；报告 α=|û|/|u*| 中位，防止收缩。
- 键长符号准确率只在 `|Δd*|>0.01 Å` 上统计，避免近零噪声。

### 5.2 V6：Progress-Aware Action Decoder

**定位**：轻量 set decoder，不是新的全分子 GNN。参数量目标 <50k（不含冻结 V5）。

**输出（v1.2 修订）**：

\[
\boxed{(\hat r_t,\hat s_{\sigma,t},\hat\ell_t,\hat p_t,\hat L_t,\delta U_{A,t})=g_\phi(Z_t)}
\]

\[
\hat r_t=\mathrm{softplus}(h_r)+\epsilon_r,\quad
\hat\sigma_{r,t}=e^{\hat s_{\sigma,t}},\quad
\hat\ell_t=\ell_{\max}\sigma(h_\ell)
\]

- **主输出 `r`**：从当前关键原子构型沿参考 IRC 到 TS 的**剩余关键原子路程**（Å，按 V5 的 `d_A` 逐步累加），直接对应 STOP 与动作步长。
- **辅助输出 `p^mw, L^mw`**：全分子质量加权弧长完成比例/总长度，只作可解释诊断；不再让 `p,L,r` 三者同时独立监督。
- **步长 `ℓ`**：关键原子 RMS 步长（Å），`ℓ≤ℓ_max=0.06 Å`；`α=ℓ/d0` 仅作报告。
- `δU_A` 默认关闭；若开启，必须使用与 V5 相同的向量基（边方向 + 秩完备通道），不能用纯标量 MLP 输出 3-vector。
- `U_{A,t}^{\rm final}`：

\[
\boxed{U_{A,t}^{\rm final}=\hat\ell_t\cdot\frac{U_{A,t}^0}{\mathrm{RMS}(U_{A,t}^0)}+\delta U_{A,t}}
\]

  然后 Safety 层按逐原子 cap/键长/碰撞缩放；训练标签用同样的 cap-aware 教师。

- V6 不使用全分子 N×3 梯度；只使用 §4.2 的关键原子标量投影。
- V6 输出 MOVE/STOP；FALLBACK/ABORT 由 Safety 层返回。对外语义仍是“V6 是唯一决策出口 + Safety 覆盖”。

**V6 损失**：

\[
\mathcal L_{r}=\mathrm{asym}(\hat r,r^*)\,\mathrm{Huber}\left(\frac{\hat r-r^*}{s_{\rm ref}}\right)
\]

\[
\mathrm{asym}(\hat r,r^*)=
\begin{cases}
2.0,&\hat r<r^*\quad(\text{低估剩余距离，倾向早 STOP})\\
1.0,&\hat r\ge r^*
\end{cases}
\]

\[
\mathcal L_\sigma=\frac12e^{-2\hat s_\sigma}\left(\frac{\hat r-r^*}{s_{\rm ref}}\right)^2+\hat s_\sigma
\quad(\text{对 }\hat r\text{ detach，只校准误差包络})
\]

\[
\bar u_i^0=\frac{u_i^0}{\mathrm{RMS}(U_A^0)},\qquad
\mathcal L_{\rm action}=\frac{1}{|A|}\sum_{i\in A}w_i\,\mathrm{Huber}\left(\frac{\|\hat\ell_t\,\bar u_i^0-u_i^*\|}{a_{\rm ref}}\right)
\]

\[
\mathcal L_{\rm aux}=\lambda_p\mathrm{Huber}(\hat p-p^{mw})+\lambda_L\mathrm{Huber}(\log\hat L-\log L^{mw})
\]

- `s_ref` 由 fit 预注册（建议 0.08 Å，按关键原子路程）；`ℓ_max=0.06 Å`。
- `L_6 = L_r + λ_σ L_σ + λ_a L_action + λ_aux L_aux`；先单独训练 r/σ，再小步联合 ℓ/action。
- `ℓ*` 教师 = cap-aware 教师步的关键原子 RMS；若 `U_A^0` 方向与 `U_A*` 的 cos <0.25 或 `RMS(U_A^0)<τ_min`，该样本不入 `L_action`，并作为 V6 FALLBACK/异常负样本。
- STOP 规则：

\[
r_t^+=\hat r_t+\kappa_\sigma\hat\sigma_{r,t}
\]

\[
a_t=
\begin{cases}
\mathrm{STOP},&r_t^+\le r_{\rm stop}\ \land\ \text{confidence valid}\ \land\ \text{safety checks}\\
\mathrm{MOVE}(A,U_{A,t}^{\rm final}),&\text{其余合法状态}\\
\mathrm{FALLBACK},&\text{不确定/越界/后端不可执行}
\end{cases}
\]

- `r_stop` 初值 0.06–0.10 Å（关键原子路程），必须在 dev 上校准到“关键键距残差 <0.08–0.12 Å”的等效范围；不得看 test 后定。
- `κ_σ` 由独立 calibration split 的 group conformal 校准；目标 85–90% 单侧覆盖。
- **early/late STOP 定义**：用真实 `r*` 与 `r_stop` 比较；`r_hat+≤r_stop` 且 `r*>r_stop+margin` 为 early STOP；`r*≤r_stop` 但模型继续 MOVE 为 late STOP。两者都报告，不能只压 early 而放任 late。

### 5.3 必跑基线

1. Zero action（全零位移）。
2. Classic key-atom（K 键长变化按质量/度分配，或最小范数 K-bond 解）。
3. 局部框架 MLP（完整几何描述符）。
4. 稀疏关键原子 EGNN/MPNN（主候选，含秩完备向量通道）。
5. Deep Sets / 全分子 GNN + key-atom head。
6. PaiNN + key-atom head（对照）。
7. 径向派生动作（只沿 K 轴）vs 完整 3D 原子位移。

所有模型：同数据、同 `d0`、同 m×3 标签、同家族划分、同 early-stopping 规则、同 checkpoint 选择；不得用旧全原子 MSE 推断新任务。

---

## 6. 执行接口（M0/M6/M6.5）

### 6.1 合同对象

```python
@dataclass(frozen=True)
class ReactionTask:
    reaction_id: str
    branch: str
    atom_map: tuple
    pair_ids: tuple[tuple[int, int], ...]  # chemistry metadata
    key_atom_ids: tuple[int, ...]         # unique atoms actually moved/frozen
    start_anchor_id: str
    method_id: str

@dataclass
class MoveProposal:
    key_atom_displacements: Array        # (M, 3), current branch frame
    geometry_features: Array             # full-molecule encoded context
    uncertainty: Array | None
    frame_contract_id: str
    model_id: str

@dataclass
class FinalDecision:
    action: Literal['MOVE','STOP','FALLBACK','ABORT']
    corrected_key_atom_displacements: Array | None  # (M, 3), if MOVE
    remaining_key_atom_path: float       # r, Å
    remaining_mw_arc: float | None       # auxiliary
    progress_fraction_mw: float | None   # auxiliary
    step_length_key_rms: float | None    # ell, Å
    remaining_upper: float
    decision_reason: str
    decision_model_id: str

@dataclass
class QCTransition:
    state_id: str
    proposal_id: str
    action_id: str
    freeze_receipt_id: str
    next_state_id: str | None
    outcome_code: str
    measured_cost_core_h: float
    ts_validation_id: str | None
```

### 6.2 `KeyAtomCoordinateWriter`

职责：

- 复制完整 XYZ，逐行写入；只覆盖 `A` 的行；其余行 bit-for-bit 保留。
- 将 key atom id 转为 XYZ 1-based 行号；校验元素顺序、原子数、mapping。
- 施加步长/键长/碰撞/价态检查；返回 `patched_xyz`、`applied_displacements`、`scale_factor`、`checks`。
- 输出内容摘要；不修改父文件。

可复用 `pes2ts_core/generation/planning/active_predictor.py` 的“只写 K 行、非 K 不变”不变量与测试思路，但不复用其 driver Jacobian 求解器（新动作是直接位移，不是目标键长反解）。

### 6.3 `XTBFreezeRunner`

职责：

- 生成 xcontrol：`$fix` + atoms 列表 + 方法/溶剂/电荷/自旋/优化级别。
- 通过 ACP 的 argv-only、claim/receipt/log/timeout 纪律执行；不直启 xTB 绕过 ACP。
- 解析 evidence：固定原子实际坐标、零漂移、非关键原子优化收敛、能量、梯度、SCC 状态、typed failure、成本。
- 版本能力探测：在 M6 前期用合法小分子做 `$fix` 回归；失败则返回 `BACKEND_CAPABILITY_MISSING`，不静默改用 `$constrain`。
- 记录 `freeze_receipt_id`，与 `QCTransition` 绑定。

### 6.4 `M6.5` 验证适配器

- 现有 ACP 严格验证链是 ORCA/OptTS/Freq/IRC 风格；V7 需要能对 xTB 执行后的结构做同方法或声明方法的 OptTS/Freq/双向 IRC。
- 若 xTB 原生 TS 验证不支持，必须显式定义：xTB 只负责 MOVE/relax；STOP 后交给 ORCA/DFT 验证；记录方法切换与成本，不能把“xTB 收敛”当 verified TS。
- M6.5 负责：`verified_target_ts` 判定、虚频模式、双侧 IRC 端点、原子映射/身份、失败类型、成本。
- 没有 M6.5，M0.5/V7 的“成功率”不可评估。

### 6.5 执行顺序

1. V6 产出 `FinalDecision`。
2. Safety 层检查步长、碰撞、价态、后端能力、预算。
3. Writer 生成 patched XYZ。
4. FreezeRunner 执行 `$fix` + relax。
5. 读取 next state/E/g；失败 typed 记录。
6. 若 STOP：M6.5 验证 OptTS → Freq → 双向 IRC → 端点身份；只有全部通过才 `verified_target_ts`。
7. 记录 cost（xTB + OptTS + Freq + IRC + 其他），不把失败尝试漏账。

---

## 7. V7：真实 QC 反馈与保守 RL

### 7.1 V7-A：冻结模型，真实转移

- 冻结 V5/V6；从真实反应端点循环。
- 对照臂：Classic、V5-only、V5+V6 frozen；同体系、同方法、同预算、同验证标准。
- 每个 episode 记录 `(S_t,a_t,X_{t+1},E_{t+1},g_{t+1},c_t,status_t)`；失败、超时、预算耗尽全部保留。
- STOP 后验证：M6.5 的 OptTS 收敛 + Freq 正确虚频 + 双向 IRC 连接目标端点；缺一不算成功。
- 首批建议 5–10 个反应 × 3–5 episodes，先证明执行链和成本账本，不追求统计显著。

### 7.2 V7-B：监督校准（优先）

- 用真实 outcome 校准 `r_stop`、`κ_σ`、`ℓ` 区间。
- 成功轨迹做 filtered BC/AWR；失败轨迹做负样本/拒绝学习。
- 目标：在 50–200 反应规模下拿到大部分真实收益；不急于 Q 学习。
- 进入 V7-C 的门：**M8.5 采集 ≥2,000 条完整转移**，覆盖多种动作/结果；否则继续 V7-B。

### 7.3 V7-C：保守离线 RL

- 优先 IQL/CQL/AWR 类受限离线更新；不做未执行动作的乐观外推。
- 冻结 V5，只优化 V6 的 STOP 阈值、ℓ、可选 δU；真实增益确认后才小学习率微调 V5。
- 保留 `L_5/L_6` 监督锚点或 KL/行为支持约束；不对 xTB/ORCA 计算图反传。
- 目标：

\[
\min_\pi \mathbb E_\pi[C_{\rm total}]
\quad \mathrm{s.t.}\quad
P_\pi(Y_{\rm targetTS}=1)\ge p_{\min}
\]

- 报告成功率、失败/删失成本、每 verified TS 的总 core-hours；不得用降低成功率换“省核时”。
- **成本口径**：intention-to-treat；分子含所有 attempts（含失败、超时、预算耗尽），分母含所有启动 episode，不只成功轨迹。

---

## 8. 工作包、依赖与门控

> v1.2 修正了 v1.1 工作包的三个循环依赖：(a) M0.5 需要教师标签，因此先做 M1-lite；(b) canonical matching/裁剪必须在标签冻结前；(c) 2,000 转移门需要独立的 M8.5 采集包。

| 工作包 | 阶段 | 关键交付 | 通过条件 | 阻断/回退 |
|---|---|---|---|---|
| **M0 动作合同** | 前置 | `ReactionTask`、`MoveProposal`、`FinalDecision`、`QCTransition`；`KeyAtomCoordinateWriter`；步长/碰撞/价态规则 | 单位、原子顺序、关键原子去重、非关键坐标 bitwise 不变、手性/反射测试通过 | 语义不清 → 不进入 M1 |
| **M1-lite 教师试点** | 前置 | 5 个分层反应（含氢转移/环体系/平面中心/长步/多 K 对）的 A/K、固定分支帧标签、rank 审计、cap-aware 步长 | 标签/单位/坐标/映射审计通过；无未解释冲突 | 失败 → 修 M0/M1-lite，不进入 M0.5 |
| **M0.5 Oracle Teacher Rollout** | 前置（最高优先） | 用真实教师 `U_A*` + Patch + `$fix` + relax 逐步推进；strict-copy vs terminal-H follower；**same-PES xTB 臂（先用 xTB 生成/验证参考路径）** + **cross-PES DFT 臂（DFT IRC 教师在 xTB 执行）**；M6.5 验证 | 在**同一执行 PES** 下，≥4/5 试点达到**绝对 verified TS**（或明确记录不可达原因）；固定原子零漂移；无系统性 SCF/优化崩溃 | Oracle 失败 → 停 V5/V6，修动作/执行/后端；不得用模型训练掩盖。same-PES 臂不能拿 DFT IRC 标签冒充；必须另有 xTB 参考路径 |
| **M1 完整教师构造** | V5.0 | 全量 key atom m×3 标签；固定分支帧；`d0` 采样；cap-aware；canonical matching；平缓区裁剪；manifest/hash | 标签恒等、单位、坐标、方法身份、双侧排序、起点歧义、冲突审计通过 | 标签冲突/退化 → 回退/剔除并记录，不带病入训 |
| **M1.5 数据卫生与划分** | V5.0 | family cluster、5-fold OOF 名单、confirm/OOD holdout、rank 切片 | 同反应不跨 split；OOF 无泄漏；monitor8 只作 smoke；OOD≥3 族或降级描述 | 家族不足 → 记录实际簇数，不伪造 45/5 |
| **M2 V5 benchmark** | V5.1–V5.2 | Classic、MLP、Deep Sets/GNN、稀疏 EGNN、PaiNN；同协议 | 未见家族上绝对指标优于 Classic/MLP；无严重族级回退；ℓ 不塌缩 | 未过 → 不进入 V6；先修标签/输入/动作 |
| **M3 OOF + V6 数据** | V6.0 | 5 折×2 seeds OOF 提案（在预弛豫帧上重算）；`IRC-PreRelaxed`；`r,σ,ℓ` 标签；`p^mw/L^mw` 辅助 | V5 训练反应不参与该折预测；输入无未来真值；V5/V6 帧版本一致；分布审计通过 | 泄漏/分布不符 → 重建 |
| **M3.5 离线 aggregation** | V6.0 | V5 OOF 提案 → patch → xTB relax → 按关键原子 RMSD 投影到参考 IRC；投影距离超阈值标 OOD/拒绝；否则取投影点切向/下一教师帧作为恢复动作标签；失败/拒绝保留 | 状态分布覆盖在线 rollout 的主要失败模式；预弛豫失败率、投影距离分布有账 | 覆盖不足/投影大多超阈值 → 增加采集或收窄动作分布，不直接开 V7 |
| **M4 V6 Progress** | V6.1 | r/σ 模型；独立 calibration split；conformal | 未见反应 r 误差、early/late STOP、覆盖率达标 | 不达标 → 不加 RL，先修特征/标签 |
| **M5 V6 Decoder** | V6.2–V6.3 | ℓ、可选 δU、FinalDecision、Safety 接口 | 动作误差优于 V5；E/g 消融有益；STOP 规则安全 | 无增益 → 冻结 V6 为规则版，不进入 V7-C |
| **M6 执行集成** | 与 M1 并行 | `XTBFreezeRunner`、xcontrol、版本回归、ACP 接缝 | 固定原子零漂移；非关键可弛豫；typed failure；成本账本 | 版本不支持 → V7 阻断 |
| **M6.5 验证适配器** | 与 M6 并行 | `verified_target_ts` 判定、Freq/IRC/端点/方法切换 | 明确成功/失败/不可验证；成本入账 | 未定义 → M0.5/V7 不可评估 |
| **M7 V7-A rollout** | V7-A | Classic vs V5-only vs V5+V6 真实 episode | 真实转移、失败、成本、验证完整 | 无成本收益/成功率下降 → 停 |
| **M8 V7-B 校准** | V7-B | STOP/σ/ℓ 校准、filtered BC/AWR | 同成功率下成本下降；early/late STOP 改善 | 不达标 → 继续采数据，不开 RL |
| **M8.5 转移采集** | V7-B→C | 按覆盖目标采集到 ≥2,000 条完整 transition；预算/样本量计划 | 动作/结果覆盖足够；失败/删失入账 | 不足 → 不开 M9 |
| **M9 V7-C 保守 RL** | V7-C | replay、critic、V6 修正、可选 V5 微调 | ≥2,000 转移；同成功率下 core-hours/verified TS 下降；可重复 | 不开 RL 或回退 V7-B |

**顺序要点**：

- M0 → M1-lite → M0.5；M0.5 通过后才做 M1 全量。
- M6/M6.5 与 M1 并行，不能等到 M5 之后。
- M1.5 的 canonical matching/裁剪在 M1 标签冻结前完成。
- M2 未过不得进入 M3；M4/M5 未过不得进入 M7。
- 数据扩量（500/2k）在 M9 之前只做 WP0 前置闭合，不作为 M1–M5 的替代。

---

## 9. 训练协议与实验矩阵

### 9.1 V5 训练

- 优化器：AdamW，lr 3e-4，wd 1e-3，dropout 0.1，grad clip 1.0。
- Schedule：warmup 5% + cosine；early stopping 按家族 dev，patience 15 epochs。
- Batch：按关键原子/图规模动态，64–256；不按帧随机打乱到跨家族。
- 精度：BF16（若硬件支持）；记录峰值显存、steps/s、图规模。
- Checkpoint：每个 epoch 的 dev 指标、最佳权重、EMA；选择规则预注册。
- 损失：§5.1；报告 `L_atom_vec/L_pair/L_cos` 分项。
- 诊断：α 中位、方向 cos（gated）、键长变化 MAE、符号准确率（`|Δd*|>0.01 Å`）、按重原子/H/长步/族/rank 切片。

### 9.2 V6 训练

- 先单独训练 r/σ（冻结 V5），再小步联合 ℓ/action。
- lr 1e-4–3e-4；batch 64–256 状态；early stopping 按 dev 家族。
- 不确定性：Gaussian NLL + 独立 calibration split 的 conformal；报告 80/90/95% 覆盖率。
- 消融：
  - Geometry only；
  - Geometry + V5 proposal；
  - Geometry + E/g；
  - Geometry + V5 + E/g；
  - + history；
  - rule step vs learned ℓ；
  - ℓ only vs ℓ + δU；
  - raw DFT 帧 vs pre-relaxed 帧。
- 评价：r MAE（Å）、`p^mw` MAE（百分点）、early STOP 率、late STOP 率、动作误差、校准覆盖率、按族/rank 切片。

### 9.3 预注册门（建议值，待 M1/M2 分布校准后冻结）

| 门 | 指标 | 建议阈值 | 说明 |
|---|---|---|---|
| V5-G1 | 未见家族关键原子向量 MAE | 比 Classic 低 ≥15%，且绝对 MAE ≤0.04 Å | 同数据/同标签；不能用弱 Classic 刷相对值 |
| V5-G2 | 方向 cos（gated） | ≥0.45 | 低于此不进入 V6 |
| V5-G3 | 键长变化符号准确率 | ≥80%（仅 `|Δd*|>0.01 Å`） | 成/断键方向 |
| V5-G4 | ℓ 比值中位 | 0.75–1.25 | 防止收缩/过冲 |
| V5-G5 | 族级回退 | 最差族不劣于 Classic 10% | 不允许“平均好、单族崩”；OOD<3 族只作描述 |
| V5-G6 | rank<3 中心的面外/法向误差 | 不劣于 rank=3 中心 1.5× | 检查 pyramidalization/线性弯曲 |
| V6-G1 | r MAE | ≤0.03 Å（待分布校准） | 关键原子剩余路程 |
| V6-G2 | early STOP 率 | ≤5% | 安全优先 |
| V6-G2b | late STOP/额外步数 | 不高于 Classic/规则版 | 防止“总是晚停”刷 early |
| V6-G3 | 动作误差较 V5 | 改善 ≥10%（绝对） | ℓ/δU 的净增益 |
| V6-G4 | 90% 区间覆盖率 | 85–95% | conformal |
| V7-G1 | verified TS 成功率 | ≥Classic（同预算，绝对验证） | 不降低 |
| V7-G2 | core-hours/verified TS | 比 Classic 低 ≥20%（intention-to-treat） | 主收益指标；含失败/删失 |
| V7-G3 | 误判 STOP/约束违例 | 0 严重违例 | 安全门 |

> 这些数值是工程筛选起点，不是文献保证；必须在看 test 前预注册。若 M1/M2 的分布显示阈值不合理，只能改 dev 规则并记录，不能看 test 后调。

---

## 10. 实验矩阵

### 10.1 V5（先 3 seeds 筛选，重要结论扩到 5 seeds）

| 臂 | 模型 | 输入 | 目的 |
|---|---|---|---|
| V5-0 | Zero | — | 下界 |
| V5-1 | Classic key-atom | K + 几何 | 物理基线 |
| V5-2 | Local-frame MLP | 完整几何描述符 | 非等变基线 |
| V5-3 | Deep Sets / 全分子 GNN | 关键 token + 全局 | 集合/图对照 |
| V5-4 | 稀疏 key-atom EGNN | A + 2-hop + 4.5 Å；signed 方向头 | 主候选 |
| V5-5 | 稀疏 key-atom EGNN + 秩完备向量通道 | 同上 + 法向/PaiNN 通道 | 平面/线性中心 |
| V5-6 | 稀疏 key-atom EGNN + 全图上下文 | 同上 + 全局池化 | 检验远端效应 |
| V5-7 | PaiNN + key-atom head | 4–16 向量通道 | 表达力对照 |
| V5-8 | 径向派生动作 | 只沿 K 轴 | 检验侧向/协同信息 |

### 10.2 V6（冻结 V5 最优）

| 臂 | 输入 | 目的 |
|---|---|---|
| V6-0 | Geometry only | 无 V5/无 E/g 下界 |
| V6-1 | + V5 proposal | V5 的增益 |
| V6-2 | + E/g（关键原子投影） | 物理观测增益 |
| V6-3 | + V5 + E/g | 主候选 |
| V6-4 | + history | 时序增益 |
| V6-5 | r + rule step | 规则步长 |
| V6-6 | r + learned ℓ | 学习步长 |
| V6-7 | r + ℓ + δU | 残差增益（可选） |
| V6-8 | raw DFT vs pre-relaxed | 分布对齐消融 |

### 10.3 V7

| 臂 | 策略 | 目的 |
|---|---|---|
| V7-0 | Classic | 基线 |
| V7-1 | V5-only | 几何动作的独立价值 |
| V7-2 | V5+V6 frozen | 决策层的独立价值 |
| V7-3 | V7-B calibrated | 真实校准的收益 |
| V7-4 | V7-C RL | 是否超越 V7-B |

同体系、同 xTB/DFT 方法、同预算、同验证、同成本口径；报告成功率与成本双轴。

---

## 11. 迁移与兼容

### 11.1 必须迁移

| 旧资产 | 新资产 | 规则 |
|---|---|---|
| `dual_ts_v2` 全原子动作/checkpoint | `pesn_v5_key_atom_*` | 旧权重不能直接当 m×3 Actor；只能作历史对照 |
| `dual_ts_features_v3/v4` | `pesn_v5_key_atom_features_v1` | 旧 node/edge 特征可复用部分，但 A/K/标签/坐标系全新 |
| `dual_ts` bundle v1/v2 | `pesn_v5_bundle_v1`、`pesn_v6_bundle_v1` | 新 schema/validator；旧 bundle 不改义 |
| `dual_ts_bc/iql` | `v5_bc.py`、`v6_train.py` | 旧训练器保留 legacy；新训练器独立 |
| 旧 monitor8 | 只作 smoke | confirm/test/family holdout 遵守冻结规则 |
| `XtbConstrainedOpt` | `XtbFreezeRunner` | 新能力；`$fix` 与 `$constrain` 分开登记 |

### 11.2 代码落点（建议）

```text
pes2ts_core/pesn/contracts/key_atom.py          # ReactionTask/MoveProposal/FinalDecision/QCTransition
pes2ts_core/pesn/features/key_atom_spec.py      # 特征/坐标系/单位/manifest
pes2ts_core/pesn/features/key_atom_graph.py     # A + 2-hop + 半径边
pes2ts_core/pesn/offline/data/key_atom_teacher.py   # 固定分支帧、d0 采样、弧长、标签
pes2ts_core/pesn/offline/data/key_atom_dataset.py
pes2ts_core/pesn/offline/data/pre_relax.py      # 固定关键原子、xTB 预弛豫非关键原子
pes2ts_core/pesn/models/key_atom_mlp.py
pes2ts_core/pesn/models/key_atom_egnn.py
pes2ts_core/pesn/models/key_atom_painn.py       # 对照
pes2ts_core/pesn/models/v6_decoder.py
pes2ts_core/pesn/offline/training/key_atom_bc.py
pes2ts_core/pesn/offline/training/decision_train.py
pes2ts_core/pesn/offline/evaluation/key_atom_protocols.py
pes2ts_core/pesn/offline/evaluation/decision_protocols.py
pes2ts_core/generation/planning/key_atom_writer.py
pes2ts_core/integration/acp/xtb_freeze_request.py
pes2ts_core/integration/acp/xtb_freeze_transport.py
pes2ts_core/integration/validation/xtb_ts_verification.py  # M6.5
scripts/prepare_key_atom_teacher.py
scripts/prepare_v6_pre_relaxed.py
scripts/campaigns/train_v5.py
scripts/campaigns/train_v6.py
scripts/campaigns/run_v7_rollout.py
tests/pesn/test_key_atom_contract.py
tests/pesn/test_key_atom_teacher.py
tests/pesn/test_key_atom_models.py
tests/pesn/test_v6_decoder.py
tests/test_xtb_freeze_transport.py
```

> 命名规则（ADR-0025）：以上训练/评测模块按 **G 能力**命名；V5/V6 战役编号只出现在 `config/campaigns/`、`scripts/campaigns/`、`outputs/campaigns/` 与报告中。

### 11.3 禁止事项

- 不得把旧 k×3 或全原子 checkpoint 无审计地加载为 m×3 Actor。
- 不得用 `$constrain atoms` 冒充 `$fix`。
- 不得把未来 IRC/TS/真实 L 作为 V6 输入。
- 不得用 monitor8 做超参/早停/结论。
- 不得用模型自报 `p` 作为 V7 成功标签。
- 不得用降低成功率换“省核时”。
- 不得在看 test 后改阈值/划分/门控。
- 不得在没有 M0.5 Oracle 证据前启动 V5/V6 大规模训练。
- 不得逐帧 Kabsch 造标签、执行时却当作固定分支帧动作。
- 不得用“比 Classic 好”替代绝对 verified-TS 门；Classic 当前正式 validated=0，必须同时报告绝对成功率。

---

## 12. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| 教师帧/gauge 不可复现 | 标签/执行错位 | 固定分支帧；逐帧 Kabsch 只诊断；Oracle 用执行 PES 验证 |
| 关键原子集合 A 定义歧义 | 标签不一致 | K 规范化、去重、per-reaction audit；多 K 集单独版本 |
| `$fix` 版本不生效 | 执行结果不可信 | M6 早期小分子回归、零漂移检查、typed refusal |
| 终端键撕裂/SCF 崩溃 | Oracle 失败 | 步长上限、碰撞/键长检查、strict vs follower A/B |
| 平面/线性中心向量基缺失 | 面外运动不可表示 | rank 审计；法向/PaiNN 向量通道；rank 切片评价 |
| d0/ℓ/cap 不一致 | 学到的动作执行时被裁剪 | 绝对 ℓ；cap-aware 教师；训练标签用缩放后动作 |
| V6 读到非关键梯度/未来信息 | 分布偏移/泄漏 | 只读 g_A、输入白名单、泄漏测试、预弛豫数据 |
| 离线 IRC 与在线状态分布不同 | V6 过拟合干净路径 | 预弛豫 + M3.5 aggregation/DAgger；分布审计 |
| 342 冲突/多路径 | 监督矛盾 | 平缓区裁剪、canonical matching、多路径标记/剔除 |
| 45 组反应过少 | 过拟合 | 家族划分、OOF、参数上限、early stopping、数据扩量后置 |
| STOP 过早/过晚 | 昂贵 OptTS 浪费/错失 | 非对称损失、r+ 上界、conformal、early/late 双门 |
| 成本账本不全 | 假收益 | intention-to-treat；所有 attempt/失败/超时入账 |
| 旧资产混用 | 归因错误 | 新版本/新 bundle/新协议；旧权重只作对照 |

---

## 13. 开放决策（实现前必须冻结）

1. **帧合同**：主标签用 `branch_frame_v1` 还是 `branch_start_anchored_v1`？逐帧 Kabsch 是否只保留诊断？
2. **A/K 规范化**：一个反应存在多个 K 集/多条分支时，A 如何合并？是否允许同一原子在不同分支有不同位移？
3. **d0**：0.06 Å 是否适合所有反应族？小环/大环/氢转移是否需要分族 d0？cap-aware 缩放是否会改变标签分布？
4. **终端氢随动**：strict-copy 与 follower 的 Oracle A/B 谁胜出？是否升级为 v1.2 合同？
5. **V6 预弛豫**：预弛豫成本/覆盖是否可接受？原始 DFT 帧单点是否只作消融？V5 OOF 是否在预弛豫帧上重算？
6. **V6 主输出**：r 用关键原子路程（Å）是否可接受？`p^mw/L^mw` 是否只作辅助？
7. **V6 参数量**：<50k 是否足够？若不足，优先加特征还是加参数？
8. **OOF 集成**：5 折×2 seeds 的 ensemble 是否纳入首轮预算？还是先单模型 OOF？
9. **V7 首批反应**：选哪些 family？必须覆盖成功/失败/长步/氢转移/环体系/平面中心。
10. **STOP 阈值**：r_stop 的初值与关键键距残差的映射如何预注册？
11. **验证方法**：STOP 后用 xTB 还是 ORCA/DFT 做 OptTS/Freq/IRC？方法切换的成本/成功定义如何登记？
12. **数据扩量**：500/2k 的 WP0 前置何时闭合？谁负责 identity partner 补绑定？

---

## 14. 评审意见处置表

| 评审意见 | 处置 | 说明 |
|---|---|---|
| 局部参考架 F(X) 易退化，改用加权 Kabsch | **部分采纳，进一步修正** | 主标签改为固定分支帧；逐帧加权 Kabsch 只作诊断；回退层次为非关键重原子→全重原子(key=0)→局部框架，禁止均匀权重回退 |
| Patch + hard `$fix` 会撕裂终端键 | **部分采纳** | 默认 strict-copy；终端 H follower 作 A/B，不静默改合同；步长/碰撞/键长检查 |
| 全分子 Δs0 导致关键原子步长随分子大小变化 | **采纳** | V5 用关键原子 RMS d0=0.06 Å；cap-aware；V6 主输出关键原子剩余路程 |
| Huber 对方向钝化，加 pair-distance + gated cosine | **采纳** | §5.1；λ_pair=0.5，λ_cos=0.10–0.15，不复制 H2 大权重 |
| V6 缺 σ 损失 | **采纳** | Gaussian NLL + 独立 calibration split 的 conformal；非对称 r 惩罚 |
| α* 需方向/范数掩码 | **采纳并改为 ℓ** | V6 输出绝对 ℓ；方向 cos<0.25/范数过小不入 action loss |
| V6 不应是新的全分子 GNN | **采纳** | 轻量 set decoder；只用关键原子标量投影 |
| 离线 DFT 帧与在线 relax 帧的 g_nonkey 分布不同 | **采纳** | 只读 g_A；V6 训练优先预弛豫帧；V5 OOF 同帧重算 |
| 342 冲突需平缓区裁剪/canonical matching | **采纳** | §4.3；用新 m×3 标签重审；匹配在标签冻结前 |
| 家族划分/5-fold OOF/holdout | **采纳** | §4.4；monitor8 只 smoke；OOD<3 族只作描述 |
| 先做 Oracle Replay 物理上限测试 | **采纳并提前** | M1-lite → M0.5；same-PES + cross-PES；绝对 verified-TS 定义 |
| M6 应与 M1 并行提前 | **采纳** | §8；`$fix` 回归不能等 M5；新增 M6.5 验证适配器 |
| V7-B 优先，V7-C 需 ≥2,000 转移 | **采纳** | §7；新增 M8.5 采集包与预算 |
| V5 主候选用 star-graph/k-hop EGNN | **采纳但限定** | 主候选；必须与 Classic/MLP/Deep Sets 同台；PaiNN 只对照；补 rank 完备向量基 |
| 数据扩至 500/2k 前先闭合 WP0 | **采纳** | §0/§8；不用扩量修可辨识性 |
| **Red-team：逐帧 Kabsch 标签依赖下一帧** | **采纳** | 固定分支帧为主；逐帧 Kabsch 只诊断 |
| **Red-team：d0/α/cap 不一致** | **采纳** | 绝对 ℓ + cap-aware 教师；训练/执行同尺度 |
| **Red-team：预弛豫不解决 covariate shift** | **采纳** | 新增 M3.5 aggregation/DAgger；V6 独立 CV/校准 |
| **Red-team：平面/线性中心向量基缺失** | **采纳** | rank 审计 + 法向/PaiNN 向量通道；rank 切片门 |
| **Red-team：工作包循环/门不可评估** | **采纳** | M1-lite→M0.5；M6.5；M8.5；OOD≥3 族 |
| **Red-team：指标可被刷** | **采纳** | 绝对 verified-TS；intention-to-treat；early+late 双门；符号阈值 |

---

## 15. 最终前向策略（v1.2）

\[
\boxed{(z_t,U_{A,t}^0)=f_\theta(G,X_t,\mathcal H,A)}
\]

\[
\boxed{(\hat r_t,\hat s_{\sigma,t},\hat\ell_t,\hat p_t,\hat L_t,\delta U_{A,t})=g_\phi(z_t,U_{A,t}^0,E_t^{\rm xTB},g_{A,t}^{\rm xTB},\ldots)}
\]

\[
\boxed{\pi_{\theta,\phi}(S_t)=
\begin{cases}
\mathrm{STOP},&\hat r_t+\kappa_\sigma\hat\sigma_{r,t}\le r_{\rm stop}\ \land\ \text{valid}\\
\mathrm{MOVE}\left(A,\hat\ell_t\frac{U_{A,t}^0}{\mathrm{RMS}(U_{A,t}^0)}+\delta U_{A,t}\right),&\text{合法且未 STOP}\\
\mathrm{FALLBACK/ABORT},&\text{不确定/能力不兼容}
\end{cases}}
\]

三条不可跨越的证据边界：

1. 参考 IRC 的 `r` 不是 xTB 扫描中可直接观测的“真实剩余路程”；V6 的距离/百分比是可解释的代理监督。
2. V6 的 IRC 动作修正不是经验上真实最优的 constrained optimization 动作；V7 必须纠偏。
3. 预测近 TS、OptTS 几何收敛、verified target TS 是三种不同标准；只有最后一种才是科学终点。

---

## 16. 参考

- 用户提供的 V5–V7 规范 v1.1 与评审意见（2026-10-18）。
- `docs/reports/PES2TS_实现报告_20261018.md`、`docs/reports/PES2TS_v4阶段训练_最终报告与问题复核_20261017.md`、V4.2–V4.7 合同。
- Maeda et al., *Intrinsic reaction coordinate: Calculation, bifurcation, and automated search*, Int. J. Quantum Chem. 2015, DOI: 10.1002/qua.24757.
- Satorras et al., *E(n) Equivariant Graph Neural Networks*, ICML 2021.
- Grimme-lab xcontrol manual: `$fix` vs `$constrain atoms`.
- grimme-lab/xtb issue #1073（`$fix` 历史回归风险）.
- Ross et al., *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*, AISTATS 2011.
- Kostrikov et al., *Offline Reinforcement Learning with Implicit Q-Learning*, arXiv:2110.06169.

---

## 17. v1.1 → v1.2 合同 delta（需签认）

| # | 项目 | v1.1 | v1.2 本计划 | 原因 |
|---|---|---|---|---|
| 1 | 教师坐标系 | 稳定参考骨架局部框架 F(X) | 固定分支帧 raw Δx 为主；start-anchored 可选；逐帧 Kabsch 仅诊断 | F(X) 易退化；逐帧 Kabsch 依赖下一帧并衰减关键运动 |
| 2 | V5 教师步距 | 全分子质量加权 Δs0 | 关键原子 RMS d0=0.06 Å；cap-aware | 避免步长随分子大小/旁观原子漂移；与执行 cap 一致 |
| 3 | V6 动作倍率 | α∈[α_min,α_max]，可 >1 | 绝对步长 ℓ∈(0,0.06] Å；α=ℓ/d0 仅报告 | 避免“学 α>1、执行被 cap 裁剪”的标签/执行错位 |
| 4 | V6 进度 | p, L, r 三者独立监督 | 主输出 r（关键原子剩余路程）；p^mw/L^mw 辅助 | p/L/r 冗余、L 难辨识、误差累积 |
| 5 | 执行非关键坐标 | 严格原样复制 | 默认 strict-copy；终端 H follower 仅 A/B，不默认 | 撕裂风险 vs 合同语义；先用 Oracle 证据决定 |
| 6 | V6 训练观测 | DFT IRC 帧的 xTB 单点 | 主：固定关键原子、xTB 预弛豫非关键原子；raw 仅消融 | 对齐在线 `$fix`+relax 分布；g_nonkey 不读 |
| 7 | V5 向量头 | 边方向 signed 和 | 边方向 + rank<3 的 per-atom 向量通道/法向 | 平面 sp2/线性中心的面外运动不可表示 |
| 8 | V6 架构 | 未限定 | 轻量 set decoder；关键原子标量投影；<50k | 避免第二个全分子 GNN 过拟合 |
| 9 | 工作包 | M0–M9 | 增加 M1-lite、M0.5、M3.5、M6.5、M8.5 | 解决循环依赖、验证缺口、2,000 转移采集 |
| 10 | 成功门 | 相对/四项门 | 绝对 verified-TS + intention-to-treat 成本 + early/late STOP | 防止弱 Classic/删失失败刷指标 |
| 11 | OOD 评价 | family holdout | ≥3 OOD 家族才宣称泛化；否则描述性 | 样本量不足 |
| 12 | V6 CV | 未明确 | 独立 Group-K-Fold + calibration split | 防止 early stopping 与 conformal 污染 |

> 以上 12 项是 v1.2 的破坏性变更。若用户不接受其中任何一项，应在实现前明确回退到 v1.1 语义，并重新评估相应门控；不能在同一实验中混用两套语义。
