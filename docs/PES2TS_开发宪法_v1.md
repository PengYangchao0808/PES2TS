# PES2TS 开发宪法 v2（PES-N 反应状态决策）

> 状态：权威（**唯一规范来源 / SSOT**）
> 层级：全仓全局，凌驾于所有计划/设计/方案文档之上（报告为只读历史，不受改写，仅受指针约束）
> 取代关系：保留唯一 SSOT；本版取代旧 G/R 独立学习与几何距离/位移主接口路线，采用 PES-N；原 v1 精确快照已归档
> 上游依赖：无（本文件是规范根）
> 下游消费者：`AGENT.md`、`docs/README.md`、`docs/plans/PES2TS_文档修订台账.md`、所有 plan/design/contracts、实验与发布流程
> 当前版本：内容 v2 @ 2026-10-05；文件名保留 `_v1` 以兼容既有链接
> 决策记录：[ADR-0010](design/decisions/ADR-0010-PES-N反应状态价值与STOP-MOVE统一决策.md)；计划：[PES-N 未来开发计划](plans/PES2TS_PES-N未来开发计划_v2_20261005.md)
> 生效边界：开发规范与规划方向生效；实现、模型、真实能力和生产默认不因本次文档变更升级
> 效力：本文件约束**后续所有开发行为**（代码、文档、实验、评测、发布）。与任何其他文档冲突时，一律以本文件为准。

---

## 0. 地位、效力与执行方式

### 0.1 单一规范来源（SSOT）
本文件是 PES2TS 的**唯一规范来源**。任何规范性的"必须/禁止"语句，只允许存在于本文件或本文件明确登记的版本化接口规范中。其余文档**只做引用**，不得重复声明规范。

### 0.2 三层效力
| 层 | 名称 | 可违背性 | 例 |
|---|---|---|---|
| **红线** | R1–R6 | **不可违背**；任何触碰 = 架构变更，须走 §12 | 真值隔离、物理标签唯一 |
| **工程政策** | P1–P5 | **默认绑定**；可经**时限豁免**（§12.3）例外 | 接口版本化、typed rejection |
| **接口规范** | I-* | 可演进，但**只以新版本**变更 | H schema、`label_state`、成本字段 |

### 0.3 执行投影（防漂移）
仓库根 `AGENT.md` 是本宪法在"AI/人类协作者自动加载面"上的**派生投影**，只允许包含：
- 指向本文件的强制阅读指令；
- 红线 **R1–R6 的 ID 与一句话摘要**（不得展开完整规范）；
- PES2TS 专属的操作性规则（真值守卫路径、模块布局、阶段流表、变更审查清单）。

**对齐律**：`AGENT.md` 不得引入任何新规范。若其摘要与本文件分歧，**以本文件为准**，并须立即修正 `AGENT.md`。CI 文档权威 lint 须校验两者的一致性（见 §12.4）。

### 0.4 命名澄清
- "开发宪法" = 本文件。
- 旧称"架构守则" = `AGENT.md`（现为执行投影）。
- 本文件不描述算法实现细节；细节在各自的版本化方案文档中，受本文件约束。

### 0.5 强制前置检查（每个开发动作前）
1. 本动作是否触碰 R1–R6？→ 是则先走 §12。
2. 是否新增/修改接口（I-*）？→ 是则只以新版本变更并登记 §9.6。
3. 是否引入新能力？→ 必须归属**唯一一层**（§2.3），并登记台账。
4. 是否产生科学声明？→ 只用四层状态词（§11.5），并提供成本账本（R5）。

---

## 1. 身份与定位

### 1.1 一句话定义（核心）
> **PES2TS does not learn the potential-energy surface; it learns how to navigate a real quantum-mechanical surface under chemically constrained actions.**

### 1.2 正式定义
> **Chemistry-constrained, physics-in-the-loop, cost-aware transition-state search architecture.**

PES2TS 是连接化学规划、低成本真实 QC 探索和严格高层级 TS 精修的决策工作流。核心 PES-N 是 **chemistry-constrained, physics-in-the-loop, reactive-state decision model**：学习 seedability、精修成本与状态/动作价值。PES2TS uses PES-N；PES-N 不限定唯一宿主。

### 1.3 科学流程链
```
Chemical Planner → PES-N评价STOP/MOVE/FALLBACK → Chemical Shield与显式选择
  → ACP/真实QC → 质量验收与新观测 → 重新决策
  → STOP: preparation/OptTS/Freq/IRC/端点身份 → 隔离标签与成本
```

### 1.4 严格产出（不可退让的终点）
目标反应由 **OptTS 收敛 + 频率确认目标模态的一阶鞍点 + 双向 IRC 连回目标 R/P + 完整化学身份** 确认。一条路径、一个高能帧、几何接近参考 TS、单虚频**都只是中间证据**。

### 1.5 明确不是（Non-goals）
PES2TS **不是**：
- 直接预测最终 TS 几何的 generator 或独立 saddle-point optimizer；
- 不带目标化学的全空间穷举发现器；
- 仅以 sequential decision-making 或 RL 名称作为创新的通用搜索器；
- 用神经网络泛化势能面（learned PES surrogate）的模型；
- HPC / 任务管理平台（那是 ACP）；
- 文献 LLM 推理器（那是 RPH-Agent）。

ML 不定义 TS。一级鞍点的负曲率计数在去除刚体自由度的相关振动空间判断，并由目标模式及双向 IRC/端点身份完成严格验证（R2）。

---

## 2. 系统栈与职责边界

### 2.1 栈图
```
RPH thinks  →  PES2TS navigates  →  ACP operates  →  CCCP computes
```
- **RPH-Agent**：读论文、理解实验背景、提出竞争机理 H₁,H₂,…，依据证据修正机理认识。
- **PES2TS**：把机理假设编译成可执行路径并生成物理证据（本宪法主体）。
- **PES-N**：可嵌入宿主的纯状态/动作评价内核，不执行 QC、不调度任务、不自行改变目标化学。
- **ACP**：调度、缓存、资源、日志、provenance、错误恢复、结果管理。
- **CCCP**：xTB / ORCA / DFT / gradient / optimization / Freq / IRC 等计算接口层。

### 2.2 所有权表
| PES2TS 拥有 | ACP 拥有 | RPH 拥有 |
|---|---|---|
| 机理编译、A(H)、反应坐标角色、路径接受/分支策略、候选排序、目标反应判定、化学身份 | 二进制/集群、QC 基元、**进程/资源/取消/任务调度**、WORK/RESULT、manifest、**任务/帧展示、缓存/成本基础设施** | 机理假设生成、文献推理、证据编排与展示 |

### 2.3 不吸收原则
PES2TS **不得**内化 HPC/任务管理/调度（ACP 职责）或文献 LLM 推理（RPH 职责）。任何新能力必须归属**恰好一层**；跨层能力必须拆分为"PES2TS 侧化学策略 + ACP 侧执行基元"，两侧不得各自实现同一套化学逻辑。
> 注：`docs/plans/PES2TS_G2_ACP能力补齐与联合验收计划_20261003.md` §1 表中的"方向/调度"专指**扫描方向 / 路径调度**（化学层），不指作业调度；后者归 ACP。措辞须以此为准。

PES-N 的图/特征/模型/推理可独立打包；宿主适配器负责观测获取、执行与验证。PES2TS 内所有生产 QC 仍适用 R3，外部宿主适配不构成直启 ORCA/xTB 或导入 cccp 的例外。新增内核目录 `pes2ts_core/pesn/` 作为已登记的未来单一落点，具体 schema 与实现另验收，在线部分不导入真值构建器。

### 2.4 命名空间（防同词异义）
| 术语 | 唯一含义 |
|---|---|
| **PES-N** | 共享反应状态决策内核：p_seed、成本、Q及不确定性；输出按训练资格分期开放 |
| **PES-G** | 兼容使用模式：在合法动作上比较 STOP/MOVE/FALLBACK 的 navigation |
| **PES-G controller** | 宿主控制器：PES-N评价＋显式选择＋Shield＋Classic backup |
| **PES-R** | 兼容使用模式：评价已有候选的同协议 STOP 价值，不是独立模型 |
| **PES-C / Classic** | 规则基线、数据生产与已验收 backup controller |
| **ReactionProfileHunter** | 外部参考实现（xTB PATH 命令形状来源），缩写只允许写全名或 `RPH-code` |
| **RPH-Agent** | 文献/机理推理层（栈顶），不得与 `RPH-code` 混用 |
| **CCCP** | 计算接口层（xTB/ORCA/DFT 等）；不是 PES2TS 模块，也不得被 PES2TS 直接 import |
| **学习级 (learning rung)** | 算法内部代际（§6.5 / §7.4），**不是**工程阶段 |

---

## 3. 红线 R1–R6（不可违背）

> 任何触碰 R1–R6 的改动 = **架构变更**，必须走 §12 程序。以下为完整规范；`AGENT.md` 只允许引用 ID 与一句话摘要。

### R1 · 真值物理隔离
DFT TS/IRC 真值只经 `pes2ts_core.g0.truth.truth_reader` 的 allow-list 访问器读取；`allow_truth` 强制；每次读取记 `truth_access_log.jsonl`；静态 AST guard 的 allowlist 与 `SUBPROCESS_IMPORT_ALLOWLIST` 是受保护配置。
**禁止**：任何 TS/IRC 几何/能量/由其直接导出的字段进入生成输入、controller 状态 `s_t`、或在线推理。教师数据（如 OptTS 结果）只能在**离线隔离构建器**中用于 train 标签，之后模型冻结再评测。

### R2 · 物理标签唯一
成功标签只来自物理验证：`OptTS 收敛 → 一阶鞍点频率/模态 → 双向 IRC → 端点极小值 + 完整 R/P 化学身份`。
**禁止**：用参考 TS 的 RMSD、能量峰、单虚频作为成功标签。RMSD 只能作诊断量。

### R3 · ACP 是唯一生产执行后端（测试替身例外）
所有**正式/生产** QC 计算（能量/梯度/Hessian/优化/频率/IRC/NEB）必须经 ACP；PES2TS 不实现任何独立计算引擎，不直接 import `cccp`，不直接启动 xTB/ORCA/Gaussian。
**例外**：CI/离线测试允许使用**同接口的测试替身（fake ACP）**；测试替身不得进入生产路径，且必须在报告中标注。缺能力时新增/扩展 ACP 能力，**不得**在 PES2TS 内造第二条执行路径或长期保留本地引擎。

### R4 · ML 控制搜索，QM 判定；A(H) 硬约束
机器学习的权限仅在**化学允许的动作流形** `A(H)` 内。任何越出 `A(H)` 的动作必须**由构造拒绝**（成员测试 + typed rejection），不得靠约定。不得以学习势能面替代真实 QM/xTB 评估。

### R5 · 全量尝试与成本留存
所有（配置范围内的）尝试、失败、回退与成本必须归档；**禁止只对成功案例计算分母/平均成本**。任何结果必须报告：`N_energy / N_gradient / N_Hessian / N_OptTS_trials`、实测 CPU core-hours 与分配核时（缺失记 `null + reason`，不得写 0）。

### R6 · 评测完整性（分裂防火墙）
- 必须执行 **reaction-family holdout**：整类反应（如 cycloaddition / rearrangement / radical）从训练中整族移除，用于泛化声明。
- 逆反应、同反应不同方向、邻帧、映射副本、对称副本**不得跨 train/valid/test**。
- holdout 与 valid/test 在**查看真值前冻结**；**禁止**对 holdout 调参或反复用于模型选择。
- 训练标签只经隔离构建器导出；`ValidationResult` 不得回流当前推理或用于 valid/test 调参。

---

## 4. 工程政策 P1–P5（默认绑定，可豁免）

### P1 · 接口版本化
H / A(H) / 状态 / 动作 / 拒绝 / 标签等接口只以**新版本**变更；历史产物绝不原地改写（旧 `g2_path_v1`、191,148 / 183,460 条历史产物只读，扩展用新版本或有哈希绑定的旁车）。
> 接口「字段仍在演进」——因此宪法只固定其**存在性与校验要求**，不冻结字段。见 §9.6 接口注册表。

### P2 · Typed honest rejection
任何非成功结果必须发出结构化拒绝记录 `{stage, code, detail, source}`；不得静默丢弃、不得把"未运行"当"失败"、不得把能力未冒烟标为可用。

### P3 · 可回放决策层（非逐位要求）
给定记录的**后端版本 + seed + config hash + 输入 hash**，**决策层**必须可复现。QM/QM 接口的逐位可复现**不作要求**；但每次物理结果必须携带足够 provenance 以回放决策。`generated_at` 等为唯一允许的易变键。

### P4 · 单一归属边界
每项能力归属恰好一层；PES2TS 不吸收 ACP/RPH 能力。跨层能力须拆分（§2.3）。

### P5 · 基线门
任一学习级（learning rung）**只有在预先注册的 benchmark 上击败上一级基线后**，才允许成为默认。按 §6.5 先监督/模仿再离线 Q；真实学习控制实验在对应阶段的模型、Shield、uncertainty 与 backup 接入门通过后进行。禁止直接在线 RL 抢先，禁止把收益自动归因于网络规模或单一组件。

---

## 5. 输入与化学编译

### 5.1 机理假设 H（版本化接口）
第一代接口仍可只接受 `R, P, ΔG`，但接口设计须**从现在起**允许未来输入：
```
H = (R, P, ΔG, E, O, S, Q, M)
```
其中：ΔG 必须发生的成/断键；E 基本事件；O 允许的先后/同步关系；S 必要立体约束；Q/M 电荷与自旋。
> 字段为版本化接口（P1），本宪法只固定其存在性与校验，不冻结具体字段。

### 5.2 化学编译与合法动作集合 A(H)
```
H → ReactionEditGraph G → A_MOVE(H,s) ⊂ R^k  (k ≪ 3N)
A(H,s) = {合法 MOVE(Δq), STOP_OPTTS, 已验收 FALLBACK, ABORT}
```
连续动作限制于图定义的低维坐标；离散 STOP/FALLBACK/ABORT 与位移分开。允许集合可含不等式、离散相位和退化点，不无条件称光滑流形。图定义目标及允许回退范围，模型不能改变目标事件；成员测试与越界拒绝适用 R4。

### 5.3 反应坐标角色
沿用四角色：`driver / monitor / guard / target_test`。任一坐标的角色与启用范围必须显式冻结。

化学角色（反应核心/几何支持/环境）与实际硬约束支持集分别记录。非驱动原子参与完整结构评价并自由松弛；反应原子未受约束的自由度亦由 QM 优化。图先验不冻结所有键、不保证目标存在或先验完备。

### 5.4 Shield、质量门与 Classic
独立 Chemical Shield 前置检查合法性、尺度、坐标退化、可实现性初检及电子态，后置质量门检查真实松弛的碰撞、身份/拓扑、位移、连续性与收敛。未知结果不能宣称已保证安全。风险阈值/拒答规则可使用经校准模型信号，但最终执行规则独立版本化。

Classic 原动作及已验收的回退始终保留；fallback 同样接受 Shield 和预算约束。NEB/GSM 等按能力放行；历史 local corrector 不绕过 R3。原生约束优化承担实际几何响应，模型不得冒充优化器内部迭代。

---

## 6. PES-N：搜索闭合模型与控制

### 6.1 定义与输出
PES-N 学习给定化学任务、当前低成本观测、下游协议下的 seedability、追加成本和动作价值；不学习完整 PES，不替代真实 E/g。共享编码器与任务头按数据证据分期开放：先 p_seed/C_full/Q_STOP 及校准不确定性，后 J^π/Q_MOVE；D_perp/D_parallel 及位移头为有监督时的可选辅助。

### 6.2 状态与物理转移
状态包括完整 X、绑定 E/g、q、G/H、方法/电子态、质量与可用性、必要历史、剩余预算和候选档案。MOVE 输出坐标目标，经 ACP/ORCA 从上一接受结构完成原生约束优化，实测结果再经质量门更新；拒绝保留父帧并记录真实费用/历史。有限历史只构成近似信息状态，不自动保证 Markov 性。

### 6.3 成功域、代价与 Bellman 边界
p_seed 绑定固定的 preparation/OptTS/Freq/IRC/端点协议、预算、目标 TS 集合及显式扰动分布；确定性单次结果不冒充已测稳健概率。优化目标在成功率/预算门下减少全量费用，首版可学习代理为：

```
J_Lambda^pi = E[C_total/C_ref + Lambda*(1-Y)]
Q_STOP = E[C_full/C_ref] + Lambda*(1-p_seed)
Q_MOVE = E[c_move/C_ref + J_Lambda(s_next, remaining_budget)]
J_Lambda = min(Q over allowed actions)
```

有限预算及最大决策/重试步数下首版 γ=1；STOP 终端包含完整费用且不重复加终端成本，失败/ABORT/预算或步数耗尽保留惩罚；Λ、成本单位与成功率门在 train 内冻结。该代理不等同于费用/成功数比率的精确最优解。只用成功轨迹或行为轨迹剩余费用不构成 J*/Q* 标签。

主评测分母是全部冻结输入。有限教师成功仅为可行见证，失败不证明动作空间无解；撤销旧版仅在 Oracle 阳性输入上比较的口径。生成/精修/验证资格分列，不设一个无严格证据的 basin-entry 成功标签。

### 6.4 数学适用条件
约束松弛面 U_G,l 按局部构象分支定义。独立光滑约束、收敛 KKT 和稳定松弛块条件下，采用 L=V+lambda^T(phi-q) 有 grad_q U=-lambda；负梯度力为 +lambda。约化曲率使用有效坐标中的 Schur 补 Hqq-Hqy Hyy^-1 Hyq，包含非线性坐标或 Lagrangian 二阶项，退化/分支切换时不无条件使用。

g_perp 仅为对指定切线的梯度残差，不是到未知流形的距离；约束自由梯度和轨迹法向梯度分开定义。曲率、乘子重建、距离代理注明来源/单位/适用条件，无支持时输出缺失。完整 MEP 保真不是主目标。

### 6.5 学习顺序与残差控制
开发顺序为 STOP 监督学习 → 行为模仿/有界残差 → 保守离线 Q → 冻结模型真实闭环。残差仅作用于连续 MOVE，Classic 动作保留，候选经 Shield 后由 Q 显式选择；离散 STOP/FALLBACK 不与向量相加。没有 MOVE 数据时只交付 STOP，不虚填 Q_MOVE。

所有实际动作/前瞻由真实 QC 反馈并计费。一步重新决策为首版；1–3 步短时域需真实可查询前瞻及单独验收，不承诺稳定或误差不累积。不学习动力学不自动等于 model-based RL。在线采集限 train 实验；更新在离线完成、冻结新版本后用于新运行，严格结果不回流当前生成（R1/R6）。

### 6.6 小模型与风险决策
先规则与低维小模型，参数目标约 10^5–10^6，少量百万级扩容需证据；包括 ensemble 总体体积及计算。完整决策开销/低成本 QC 工作流 <1% 是测量目标而非保证，图/特征、推理、Shield/选择、冷暖启动和 CPU/GPU 分列。

费用型 Q 可用 mu+beta*sigma 作经校准风险规则；sigma 不自动是可靠置信界。风险增大时按冻结规则缩步、付费 probe、Classic 或拒答；所有 fallback 均过门。输出值、规则放行、真实验收及科学成功分别留证据。

---

## 7. Ranking 与 Navigation：同一内核的使用方式

### 7.1 STOP 即 Ranking
PES-R 比较同目标、同精修协议/预算和单位下的 Q_STOP。PES-G 比较 STOP/MOVE/FALLBACK；内部档案可通过 STOP(i) 提交历史合格帧，既有规则后排序保留为基线。多候选顺序中的相关性/预算使静态排序不自动成为全局最优策略。

### 7.2 单点优先、上下文可选
最小可用性包括 X/E/g、目标图、元素/映射、方法/电子态、单位及质量；仅有 XYZ 不保证能够评价目标成功率。上下文只增强，缺失显式处理。静态全轨迹邻域与在线因果前缀分别标注，禁止未来信息进入当前决策。

### 7.3 标签与跨来源资格
STOP 成功标签仅来自 R2 完整链，费用含失败/超时。任意来源的合格帧可用于同协议 STOP 标注；MOVE-Q 仅接受已知动作及相容状态转移合同。MD 速度/温控、NEB 全链、其他优化器状态与约束松弛不同，不能把相邻坐标差冒充命令动作。

### 7.4 共享与泛化
PES-N 保持一份核心/特征/模型版本，任务头与校准可以不同；独立模型只作必要对照或有证据的回退。外部适配保留来源/方法/偏置 provenance；跨生成器留出与 reaction-family holdout 同时报告，统一格式不等于已证泛化。

---

## 8. 精修链路（Preparation Layer）

xTB → DFT 之间**必须**插入准备层；禁止 `xTB 帧 → DFT OptTS` 直连：
```
X_i^{xTB} → constrained DFT preparation（短暂固定 q_reactive，令其他 DOF 在 DFT PES 上重新适应）
         → OptTS → Freq → IRC(+/-) → endpoint opt → graph identity
```
理由：xTB 只负责导航；DFT preparation 让候选在目标方法下局部适应，不能保证消除方法偏差；OptTS 只负责局部 saddle refinement。
> 现状边界：先前审核的链路为 xTB 帧 → 规则排序 → 直接 DFT OptTS；本轮未验收上述完整准备协议。应复核现有实现并补齐真实证据，不能把其他起点准备模块或本规范的存在当成该能力已具备。

---

## 9. 数据、溯源、成本与接口注册表

### 9.1 数据飞轮（Classic 提供真实尝试，严格验证提供成功真值）
必须保留**全部候选**（不只 winner）：
- PES-G 最珍贵：`(s_t, a_t, s_{t+1})`；
- PES-R 最珍贵：`(X_i, OptTS outcome, C_i)`。
Classic 长期保留为规则基线和已验收 backup；其轨迹本身不是 TS 真值。状态、已知动作转移、严格精修结果分别具有数据资格。

### 9.2 `label_state` 七态（版本化接口）
`verified_target / verified_non_target / optimizer_failed / validation_incomplete / execution_failed / unattempted / censored_budget`。
`1000×20=20000` 是**帧数**，不是标签数；未运行邻帧不得称 hard negative。

### 9.3 Provenance 四层
`geometry / mapping / cohort_selection / development_exposure`。"字段隔离 ≠ 来源独立"；当前 G1 映射为 `truth_assisted_p1`，不得描述为端到端无真值独立。

### 9.4 成本账本标准
见 R5。缺失值 `null + reason`；数值 Hessian 的内部梯度只计一次；缓存命中本次成本为零但保留来源与共享关系；基准另报冷启动/分摊成本。

### 9.5 长周期留存层
Reaction / ReactiveState / Action / Environment / Transition / ShieldDecision / Path / STOP评价 / Refinement / TS / IRC / Label / Cost；保留原子、方法、来源、观测掩码、终端/删失及模型/规则版本。

### 9.6 接口注册表（I-*，版本化）
以下接口只以新版本变更，且必须登记于 `docs/plans/PES2TS_文档修订台账.md`：
`H`、`A(H)`、`state s_t`、`action a_t`、`rejection {stage,code,detail,source}`、`label_state`、`ValidationResult`、`ExecutionBackend`、`TrajectoryRecord`、`PathBundle`、`SeedProposal`、`calculation_cost_v1`、`energy_gradient`、`bond_order_evidence_v1`、`g1_g2_boundary_v2`、`endpoint_method_evidence_v1`（G2-AB1 WP-3 拟定：只固定存在性与校验，不冻结字段）。

ADR-0010 登记待实施接口存在性：ChemicalControlSpec、ReactiveState、PESNTrajectory、Action/Transition、SeedProtocol/SeedOutcome、DecisionEstimate、ShieldDecision/DecisionTrace、PESNModelBundle。具体新 schema/fixture 随实现登记；旧主数据链不改义，扩展可用 hash 绑定旁车。冻结模型推理得到的估计与隔离真值标签分别存储，不将真实未来结果作为输入。

---

## 10. 阶段、里程碑与命名

### 10.1 唯一交付轴：G0–G4
工程阶段仍为 G0–G4，工作流仍为 C/R/X。G0 数据/真值隔离；G1 化学任务与动作编译；G2 Classic/Shield/真实优化/严格验证；G3 规模化与 STOP 监督、模仿/残差；G4 离线 Q、冻结模型闭环与泛化。详见阶段定义，不新增 PES-N 专属工程阶梯。

### 10.2 科学里程碑 M0–M5
仅用于评测/论文声明，不出现在计划状态头中；既有名称的后续口径如下，历史结论不重标。

| 里程碑 | 阶段 | 新声明所需证据 |
|---|---|---|
| M0 Classic Closure | G2 | 24 可审计终态及≥1非参考严格目标 TS，辅助来源如实披露 |
| M1 Strong Baselines | G2/G3.0 | 相同先验/QC/验证预算的规则与路径方法基线 |
| M2 STOP Value | G3.1 | 单点 seedability/费用/STOP 在固定候选上改善严格选点 |
| M3 Residual and Offline Q | G3.2/G4.0 | 残差/保守 Q 相对于行为/Classic 的独立接入证据 |
| M4 Closed-loop Navigation | G4.1 | 冻结模型真实闭环在目标成功率门下改善全量成本 |
| M5 Cross-generator System | G4.2 | 单点/上下文、跨来源与整族泛化、完整系统费用与可复现发布 |

### 10.3 开发与论文叙述
开发为 Classic → STOP supervision → imitation/residual → offline Q → frozen closed loop。论文解释同一内核如何评价 STOP 与选择计算，不预设 G→独立R 两模型串行。静态排序可先交付，不代表另建 PES-R 学习产品。

### 10.4 命名纪律
G0–G4 为工程阶段；C/R/X 为工作流；PES-C 是 Classic，PES-G/PES-R 是兼容使用模式，PES-N 是共享决策内核；M0–M5 是上述科学声明；发布包版本为产品身份。历史 L-G*/L-R* 仅作旧算法学习级映射，不设新的并行编号体系。

---

## 11. 评测与泛化

### 11.1 分层评测
固定候选池评价 STOP；固定公共执行/Shield/验证协议比较动作策略；共享权重或交接层同时变更时报告联合系统收益。保持生成、静态选择和完整系统评测可分辨，模型精度不代替严格物理结果。

### 11.2 双维泛化
R6 的整反应族留出保持强制，同时安排 cross-generator holdout 和二者交叉实验。正逆方向、邻帧、映射/对称副本同组；同一反应换 generator 不等于化学泛化。单点/上下文、输入资格、缺失/拒答、方法与偏置差异分别报告。

### 11.3 全量成本与成功率门
主指标为全输入严格目标成功率/覆盖—成本和总费用/成功数，在预注册的成功率非劣界限下判断效率。默认每个反应目标最多计一次成功，不同 TS 数另以冻结身份去重。零成功时单位成功成本未定义，报告总费用与零覆盖。全部失败/恢复/教师/前瞻/验证费用留存，训练/运行/摊销情景分列；同单位才能聚合。

### 11.4 基线与反过宣称
比较规则峰/事件峰/经典排序、Classic、残差、显式 Q、共享/独立模型、Shield与不确定性策略、环境/QM反馈以及付费短前瞻。固定共同验收，不能通过移除验收制造优势。24/24 终态不是 24/24 化学成功，≥1 TS 为工程门；跨来源通用、全局收敛、安全与概率校准均需独立证据。

### 11.5 完成状态
`unknown / implemented / fixture_passed / smoke_passed` 分别表达；真实冒烟与科学评测才能支撑相应能力声明。规划、规范、文档或模型文件的存在不升级能力。调参、阈值与风险权重按 R6 在 train 内冻结。

---

## 12. 修订、豁免与生效

### 12.1 修订（架构变更）
凡触碰 R1–R6、§5–§9 接口、§10 阶段边界者，均为架构变更，须：
1. **声明**：改动内容、触及条目、理由、影响面。
2. **ADR**：写/改 `docs/design/decisions/ADR-XXXX-*.md`（不原地改写既有结论）。
3. **版本化**：新接口用新版本号；旧产物不覆盖。
4. **证据**：fixture / 回放 / 真实冒烟 / 审计 JSON；不得以"测试全绿"替代真实能力。
5. **同步**：本文件版本、`AGENT.md`、台账、`docs/README.md`。

### 12.2 本宪法自身的修订
version bump + 变更记录；重大不变量变更须同时更新 §3 与 `AGENT.md` 摘要。

### 12.3 豁免（Waiver）
允许对 **P1–P5** 申请**时限豁免**（R1–R6 不可豁免）：
- 记录为 ADR，含理由、范围、到期条件、回滚计划；
- 豁免期结束必须收回或转为正式修订；
- **禁止**用豁免绕过 R 或用于在 holdout 上调参。

### 12.4 对齐律与 CI 文档权威 lint（防漂移）
- `AGENT.md` 只准引用红线 ID 与一句话摘要；CI 须校验 `AGENT.md` 与 §3 一致。
- CI 须校验：所有 `docs/**/*.md` 带状态头；每主题恰有一份 `AUTHORITATIVE` 文档；已降级/取代文档带顶部指针；已取代文档不得出现在检索/嵌入索引中。

### 12.5 Sunset（日落）
被取代的规范/接口与文档，须在台账登记取代关系并设定日落；禁止静默复活旧阶梯（`S0–S8`、`L0–L4`、`M1–M3` 等）。

---

## 13. 术语表

| 术语 | 含义 |
|---|---|
| H / G | 机理条件合同 / 反应任务图，不是在线参考 TS |
| A_MOVE / A(H,s) | 连续坐标动作子集 / 含 STOP、FALLBACK、ABORT 的合法动作集合 |
| PES-N | graph-conditioned search closure model，共享决策内核 |
| p_seed | 给定精修/扰动/预算协议的严格目标成功预测概率 |
| Q_STOP / Q_MOVE | 同单位和失败处理下的精修/推进动作价值；首代不声称 Q* |
| Chemical Shield | 执行前确定性资格与规则层，另有执行后真实质量验收 |
| ReactiveState / PESNTrajectory | 跨来源物理状态 / 带来源及转移资格的状态容器 |
| Preparation Layer | 低成本帧切换 DFT 精修前的受限准备 |
| SSOT | 本文件；阶段/计划/投影均服从本规范 |

---

## 14. 与文档权威台账的关系
本文件是规范根；`docs/plans/PES2TS_文档修订台账.md` 是**规范映射与取代登记**（哪份文档属于哪一主题、何状态、被谁取代）。台账服从本文件；两者不一致时，以本文件为准并修正台账。

## 15. v2 变更记录与历史

2026-10-05 按 ADR-0010 将 PES-N 状态/动作价值、STOP/MOVE、Residual-first、Offline-first、跨来源状态及轻量独立内核写入未来开发规范。保留 R1–R6 原文，更新 §1–§2、P5、§5–§7、§9–§11/术语，纠正有限 Oracle 主分母与物理距离过强解释。

原规范与入口精确快照见 [归档说明](archive/20261005_pre_pesn/README.md)。本版规范方向生效，旧运行/模型/接口结果不重标，生产默认不改变。

