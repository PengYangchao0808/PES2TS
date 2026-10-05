# PES generation：多坐标约束学习、ORCA 能力与数学方案

> **宪法指针（2026-10-04）**：本设计把 PES-G 定义为"学习条件化多坐标目标曲线 `c_θ(λ)`"并以联合平滑路径为目标。**该路线非当前 PES-G 定义**。现行为：PES-G = graph-conditioned learned predictor + QM corrector，目标是**进入 TS basin 而非重建 MEP**（见 [`PES2TS_开发宪法_v1.md`](../PES2TS_开发宪法_v1.md) §6）。本文降级为 **G-Oracle / teacher 数据生成器参考**；正文保留以备历史核对，不改写。

日期：2026-10-01。状态：设计方案，尚未实现模型、修改生产算法或提交新计算。依据：当前 PES2TS 与本机 ACP 源码、上一轮 Demo24 几何审计，以及 ORCA 6.1/6.1.1 官方文档。以下优化、学习与边界算法是本项目建议及数学推导，不是对 ORCA 内部源代码实现的逐行描述。

> 路径注记（2026-10-03）：本文引用的旧模块路径已随 G-R 布局迁移：`pes2ts_core/scan_strategy/` → `pes2ts_core/generation/planning/`，`pes2ts_core/g2/` → `pes2ts_core/generation/execution/xtb_path/`（`endpoints` → `pes2ts_core/generation/assembly/`），`ranking.py` → `pes2ts_core/ranking/`；正文历史引用保留，不改写。

## 1. 对目标的准确表述

本项目需要学习一个条件化的多坐标目标曲线：

\[
\mathcal C_\theta:\ (\mathcal R,\lambda)\longmapsto
\mathbf c_\theta(\lambda\mid\mathcal R),\quad 0\leq\lambda\leq1.
\]

反应条件 \(\mathcal R\) 包含已映射的 R/P 图和几何、多键编辑图、事件耦合图、局部化学环境、构象与组分关系、总电荷、自旋多重度、计算方法及溶剂条件；机理已知时作为条件，未知时由模型预测并保留不确定性。

每条反应最终只有一份主计划和一条共享 \(\lambda\) 的扫描曲线。每个点有 \(K\) 条必要坐标共同受控；K 由机理、几何和独立性决定，不由“一维”这个词决定。

需要区分三类数值：

| 对象 | 符号 | 意义 | 建议 |
|---|---|---|---|
| 约束目标值 | \(c_j(\lambda)\) | 这一帧的键长、键角、二面角应为多少 | 第一阶段的学习主体 |
| 同步变化规律 | \(f_j(\lambda),c'_j(\lambda)\) | 各事件何时发生、变化多快、如何配合 | 与目标值一起学习 |
| 软约束强度 | \(k_j\) 或矩阵 \(K_{\rm soft}\) | 偏离目标时增加多少惩罚能量 | 可选扩展；ORCA 硬约束没有这个参数 |

已知完整两端提供的边界不是待预测量：

\[
\mathbf c(0)=\mathbf q(X_R),\quad
\mathbf c(1)=\mathbf q(X_P),\quad
X(0)=X_R,\quad X(1)=X_P.
\]

学习重点是内点的目标值和曲线形状，以及实现目标反应所需的坐标集合。若未来端点未知，应将端点生成明确列为另一项任务。

## 2. ORCA 能满足哪些要求

ORCA 支持 B/A/D 和原子 Cartesian 约束，并以冗余内坐标的投影方式限制允许的优化方向；原生扫描可用 Simul_Scan 同时改变至多三条扫描坐标，也有自定义距离点列表。6.1 还提供带参数的 bond bias，可用于软引导。它们均需要调用者提供数值；这些接口没有按反应机理训练推荐值的功能。[约束与 bias 官方说明](https://orca-manual.mpi-muelheim.mpg.de/contents/structurereactivity/optimizations.html#constrained-optimization)、[扫描官方说明](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/optimizations_scans.html#multidimensional-scans)。

| 要求 | 判断 | 责任归属 |
|---|---|---|
| 同一帧约束多个距离/角度/二面角 | ORCA 原理上支持，具体组合需实跑 | ORCA + ACP |
| 多坐标共用一个进度参数 | 可以执行；需确保同步而非笛卡尔网格 | PES2TS 编译 + ACP |
| 任意学习所得非线性逐点目标值 | 可以外部逐点编排 Constraints | PES2TS/ACP 需要接通 |
| 原生 Scan 超过三条扫描坐标 | 官方扫描语法不能满足 | 改用逐点 Constraints；不是删除必要坐标 |
| 自动推荐机理相关目标值 | 现有功能不包含 | PES generation 学习层 |
| 自动保证完整已知末端 | 局部多坐标优化不能单独保证 | 固定边界合同 + 路径求解与验收 |
| 保证找到目标 TS 或全局正确机理 | 不保证 | 学习、执行闭环与科学验证 |
| 任意耦合软势/通用矩阵刚度 | 不能假设由现有 B/A/D 硬约束实现 | 需额外能量/梯度或优化器集成 |

官方“最多三条”指原生 Scan/Simul_Scan，不应套到普通逐点 Constraints 的全部坐标上。后者仍受坐标独立性、几何可行性及部署验证限制，不应宣传为无限可用。

本机 ACP 目前 MAX_SYNC_COORDINATES=4，坐标合同只有 start/end/n_points，ReactionCoordinatePlan 根据帧索引线性重算目标值。它虽然有多坐标逐点优化入口，却尚不能忠实表达本方案的任意 \(c_j(\lambda_i)\)。PES2TS 编译层已经有 values/schedule_values，和 ACP 的生产合同需要统一。

证据：[ACP 合同](E:/Calculations/Common_Script/Auto_Calc_Platform/ACP_V1_20260811/src/acp/calculations/pes/contracts.py:45)、[ACP 线性目标计算](E:/Calculations/Common_Script/Auto_Calc_Platform/ACP_V1_20260811/src/cccp/qc/interfaces/constraints.py:99)、[PES2TS 多坐标投影](E:/Calculations/AI4S_ML_Studys/PES2TS/pes2ts_core/integration/acp/multicoord.py:405)、[编译层逐点值](E:/Calculations/AI4S_ML_Studys/PES2TS/pes2ts_core/scan_strategy/compile_orca.py:799)。

## 3. 一维、多坐标和松弛自由度的数学关系

令 \(X\in\mathbb R^d\) 为去除全体系整体平移和旋转后的几何。一般非线性体系 \(d=3N-6\)；线性体系需另行处理。组分之间的相对平移和转动仍是物理自由度，不能一并删除。

选出 \(K\) 个反应相关内坐标：

\[
\mathbf q(X)=(q_1(X),\ldots,q_K(X))^\top.
\]

模型给出参数曲线 \(\mathbf c_\theta(\lambda)\in\mathbb R^K\)。若 \(K>1\)，曲线位于多维坐标空间，但它的参数维数依然是 1。

在固定 \(\lambda\) 时，约束集合为

\[
\mathcal M_\lambda=\{X:\mathbf q(X)=\mathbf c_\theta(\lambda)\}.
\]

若约束 Jacobian 秩为 K，局部可行流形维数是 \(d-K\)，这些自由度由能量松弛决定。模型的曲线维数 1、驱动坐标数 K、每点剩余自由度 d-K 是三个不同概念。

当前错误的单键实现保留了太多未受控的反应自由度。反过来，机械固定所有可枚举的内坐标也会产生冗余或不相容约束。合理目标是保留能控制所需事件和分支的必要独立坐标。

## 4. 多坐标硬约束优化的推导

设同一电子态、计算方法及环境下的无偏置势能为 \(E(X)\)。一个内点的理想问题是

\[
\min_X E(X),\qquad
\mathbf q(X)=\mathbf c(\lambda).
\tag{1}
\]

这通常是局部约束极小化，不保证得到全局最小分支。

### 4.1 Lagrange 乘子与一阶条件

引入 \(\boldsymbol\nu\in\mathbb R^K\)：

\[
\mathcal L(X,\boldsymbol\nu;\lambda)
=E(X)+\boldsymbol\nu^\top[\mathbf q(X)-\mathbf c(\lambda)].
\]

记 \(g=\nabla E(X)\)、\(J=\partial\mathbf q/\partial X\)，则驻点满足

\[
g+J^\top\boldsymbol\nu=0,\qquad
\mathbf q(X)-\mathbf c(\lambda)=0.
\tag{2}
\]

乘子描述保持坐标目标所需的广义约束力。它是求解结果，不能当成输入的弹簧常数。

### 4.2 为什么要投影梯度

在已满足约束的点上，小位移需要满足

\[
J\,\delta X=0.
\]

在欧氏坐标下，允许位移空间的正交投影为

\[
P_T=I-J^\top(JJ^\top)^+J,
\tag{3}
\]

其中 + 为 Moore–Penrose 伪逆。若 J 满行秩，可用普通逆。约束驻点条件等价于

\[
P_Tg=0.
\tag{4}
\]

式 (3)–(4) 是便于解释的 Cartesian 推导；ORCA 实际采用其冗余内坐标中的投影与步长算法，不能把本式宣称为其内部代码原样实现。

若使用质量加权或其他度量，需先变换坐标并相应变换 J。距离、角度和二面角也要按尺度归一化；不能将 Å 与度数混在一个未经尺度处理的矩阵里比较条件数。

### 4.3 二阶步与稳定性

Lagrangian Hessian 为

\[
H_L=\nabla_X^2E+\sum_j\nu_j\nabla_X^2q_j.
\]

对式 (2) 作 Newton 线性化：

\[
\begin{bmatrix}
H_L&J^\top\\
J&0
\end{bmatrix}
\begin{bmatrix}\delta X\\\delta\boldsymbol\nu\end{bmatrix}
=-
\begin{bmatrix}
g+J^\top\boldsymbol\nu\\
\mathbf q(X)-\mathbf c
\end{bmatrix}.
\tag{5}
\]

若 Z 的列张成 \(\ker J\)，严格局部约束极小值还需要

\[
Z^\top H_L Z\succ0
\tag{6}
\]

（已排除刚体零模等例外）。因此，数值目标可实现并不意味着该分支对松弛方向稳定。

在真正 TS 附近，若允许位移空间中仍存在负曲率方向，普通约束最小化可能离开 TS。即使模型给出了 IRC 上的正确键长，也不能断言 ORCA 会复原该帧全部原子坐标；必须同时检查参考帧的 \(P_Tg\) 和受限曲率。

### 4.4 反应曲线能量与约束力

若局部解 \(X^*(\lambda)\) 光滑：

\[
J\frac{dX^*}{d\lambda}=\mathbf c'(\lambda).
\]

由式 (2) 得

\[
\frac{dE(X^*(\lambda))}{d\lambda}
=g^\top\frac{dX^*}{d\lambda}
=-\boldsymbol\nu^\top\mathbf c'(\lambda).
\tag{7}
\]

曲线峰满足一个标量条件 \(\boldsymbol\nu^\top\mathbf c'=0\)，多个乘子项可以互相抵消。峰不意味着 \(\boldsymbol\nu=0\)，更不意味着全空间 \(g=0\)。严格 TS 仍须独立检查驻点、负模与连接关系；ORCA 提供 TS 搜索与 IRC 功能。[TS 搜索](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/optimizations_TS.html)、[IRC](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/irc.html)。

### 4.5 学习参数如何影响实际结构

固定 \(\lambda\)，设目标由模型参数 \(\theta\) 决定，且式 (5) 的 bordered 矩阵可逆。对式 (2) 求导：

\[
\begin{bmatrix}H_L&J^\top\\J&0\end{bmatrix}
\begin{bmatrix}
\partial X^*/\partial\theta\\
\partial\boldsymbol\nu/\partial\theta
\end{bmatrix}
=
\begin{bmatrix}0\\\partial\mathbf c_\theta/\partial\theta\end{bmatrix}.
\tag{8}
\]

式 (8) 给出约束值学习与实际几何误差之间的联系。可以构造双层训练：

\[
\min_\theta\mathcal J(X^*(\mathbf c_\theta),\theta),
\quad X^*\text{ 满足式 (2)}.
\]

它需要一致梯度、Hessian/近似以及稳定分支，ORCA/ACP 当前未直接提供端到端自动微分合同。第一阶段先做监督拟合和少量闭环实验；未来可采用有限差分或另行实现隐式求导。边界切换、秩变化、分支跳跃时式 (8) 不适用。

## 5. 保持端点的目标值学习

### 5.1 单调坐标

对沿机理单调变化且两端差异非零的坐标：

\[
c_j(\lambda)=q_{j,R}+\Delta q_j f_j(\lambda;\mathcal R),
\quad \Delta q_j=q_{j,P}-q_{j,R}.
\tag{9}
\]

选择一个严格正密度

\[
p_j(t)=\operatorname{softplus}(h_{\theta,j}(t,\mathcal R))+\epsilon,\quad\epsilon>0,
\]

并定义

\[
f_j(\lambda)=
\frac{\int_0^\lambda p_j(t)\,dt}{\int_0^1p_j(t)\,dt}.
\tag{10}
\]

从积分直接得到

\[
f_j(0)=0,\quad f_j(1)=1,\quad
f'_j(\lambda)=\frac{p_j(\lambda)}{\int_0^1p_j(t)\,dt}>0.
\]

所以起止值由结构保证；成键 \(\Delta q_j<0\) 自动缩短，断键 \(\Delta q_j>0\) 自动延长。不同坐标可以有不同 \(p_j\)，但调用同一个 \(\lambda\)。实际实现可用非负样条密度和规范化积分，并在数值序列中显式写入精确起止值。

归一化使密度的整体乘法尺度不可辨识，应固定其积分或正则化参数。只要求需要单调的驱动满足此形式；不能强迫所有 IRC 坐标单调。

### 5.2 非单调或两端近似相等的坐标

有些键角先弯曲再恢复，或某一坐标在两端几乎相同而中途明显变化。此时不能除以 \(\Delta q_j\)，可用

\[
c_j(\lambda)
=(1-\lambda)q_{j,R}+\lambda q_{j,P}
+s_j\lambda(1-\lambda)g_{\theta,j}(\lambda,\mathcal R).
\tag{11}
\]

s_j 是按训练集/坐标类型确定的单位尺度。因 \(\lambda(1-\lambda)\) 在两端为零，式 (11) 保留端点；内点由 g 学习。需要额外检查距离正值、角度范围、几何可实现性及曲线曲率。

二面角需基于连续展开后的角度拟合，损失用周期距离，例如 \(1-\cos(\phi-\phi^*)\)。最短旋转只是基线，路径可能有指定旋向或额外绕转；不能在每一帧 wrap 后用普通差分制造跳变。

### 5.3 Demo24 的具体解析例子

上一轮审计中 RXN_0000007104 的已知 TS 相对原始 R/P 的归一化进度为：

\[
s_{N1-H11}^{TS}=0.257407846,\qquad
s_{N3-H11}^{TS}=0.810668860.
\]

严格线性同步 \(f_1=f_2=\lambda\) 无法经过这个二维 TS 坐标点。作为函数形状示例，人为将公共参数的 TS 位置设为 \(\lambda_{TS}=1/2\)；这是重参数化演示，不是声称该反应的 IRC 质量加权弧长中点恰好是 TS。

令

\[
f_\beta(\lambda)=
\frac{e^{\beta\lambda}-1}{e^\beta-1},\quad
f_0(\lambda)=\lambda.
\tag{12}
\]

在中点：

\[
f_\beta(1/2)=\frac{1}{1+e^{\beta/2}},
\quad
\beta=2\log\frac{1-s^{TS}}{s^{TS}}.
\]

因此

\[
\beta_1=2.118970399,\qquad
\beta_2=-2.908724229.
\]

| 公共 λ | 第一条坐标进度 | 第二条坐标进度 |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0.25 | 0.095389 | 0.546542 |
| 0.50 | 0.257408 | 0.810669 |
| 0.75 | 0.532595 | 0.938313 |
| 1 | 1 | 1 |

两条函数同处一条一维曲线，目标值体现异步程度与先后关系，末值仍固定。这里仅解析匹配两条 TS 坐标，并未执行新的多约束计算，也未证明完整几何、其他坐标或目标方法下的 TS 被复原。

式 (12) 是低参数初始模型，可按机理学习 \(\beta_j\)；大量数据后再升级为式 (10) 的样条函数。

## 6. 从 TS/IRC 获得训练标签

### 6.1 参考一致性与公共参数

每个训练反应需要已映射 R/P、经验证 TS、两侧 IRC 轨迹及方法/电荷/自旋/环境。IRC 可用来确认 TS 与两侧中间体的直接连接。[ORCA IRC 说明](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/irc.html)。

整理两侧轨迹方向，去重 TS，将全体系正确对齐后按质量加权弧长定义

\[
\ell_k=\sum_{m=1}^k
\sqrt{(X_m-X_{m-1})^\top M(X_m-X_{m-1})},
\qquad
\lambda_k=\ell_k/\ell_{\rm end}.
\tag{13}
\]

所有坐标使用同一 \(\lambda_k\)，得到标签

\[
y_{rkj}=q_j(X_{rk}^{IRC}).
\]

若原始 R/P 与 IRC 末端不同，不能直接把原始结构贴到 IRC 两端称为连续教师轨迹。必须验证从所选原始边界到 IRC 的连接段，或将该样本标记为边界不一致并暂不用于完整路径训练。允许参考经过的共同刚体对齐，不允许独立移动组分而不记录。

弧长是参数规范，必须在训练和评估时一致。另一种规范是统一 TS 位置后分段重参数化，但不能与弧长标签混用。

### 6.2 模型结构与数据规模

第一阶段建议采用“机理共享参数 + 局部环境修正”的低参数模型，例如

\[
\beta_{rj}=b_{m(r),t(j)}
+w_{t(j)}^\top z_{rj},
\tag{14}
\]

其中 m 是机理类别，t 是坐标/事件角色，z 包含局部化学和几何描述。用正则化、分层共享和不确定性估计避免小样本过拟合。对于数据足够的机理再增加样条系数。24 条反应适合作为接口和方法验证集，不足以证明对不同机理普遍泛化；394 帧高度相关，也不是 394 个独立反应样本。

长期模型可用映射 R/P 双图与事件图编码，按原子/事件角色输出坐标函数及低秩联合不确定性。未知机理应保留多分支分布并用可行性、目标编辑及概率选择最终主计划；不能平均两条不同机理的坐标曲线生成无意义的中间机理。

### 6.3 监督损失

距离等非周期坐标可采用

\[
\mathcal L_{\rm fit}
=\sum_{r,k,j}w_{rkj}\,
\rho\!\left(\frac{c_{\theta,rj}(\lambda_{rk})-y_{rkj}}{s_j}\right),
\tag{15}
\]

其中 \(\rho\) 可为 Huber 损失，s_j 处理量纲和尺度。采样权重应避免长 IRC 轨迹支配训练。TS 区域可增加权重，并添加

\[
\mathcal L_{\rm TS}
=\sum_{r,j}w_{rj}^{TS}
\rho\!\left(
\frac{c_{\theta,rj}(\lambda_r^{TS})-q_j(X_r^{TS})}{s_j}
\right).
\]

曲率正则化为

\[
\mathcal L_{\rm smooth}
=\sum_{r,j}\int_0^1
\left|\frac{d^2(c_{\theta,rj}/s_j)}{d\lambda^2}\right|^2d\lambda.
\]

最终加入几何可行性、坐标独立性和经实际执行得到的路径/分支误差。训练目标不能只最小化能垒：模型可能选择另一个更低能但错误的机理。

已知 TS/IRC 可以作为训练标签，也可用于当前 24 条的参考引导重建；这不等于新反应的预测。测试反应的 TS/IRC 不得进入其预测输入。拆分以反应/骨架/机理为单位，禁止同一 IRC 的不同帧跨训练与测试。报告参考引导恢复、同机理预测和未见机理外推三种成绩。

## 7. 几何可行性和坐标选择

图编辑是必要线索，不是完整几何算法。F/B 变化通常对应距离，O 键级变化可需要共轭区距离/角度/二面角；离散键级不能直接作为 ORCA B/A/D 数值硬约束。氢转移至少要考虑供体/受体关系，环重组要考虑环闭合，分子间反应要保留相对位置和进攻方向。

先对 J 的行和几何变量尺度处理，再做秩揭示 QR/SVD。报告有效秩、最小非零奇异值与条件数；不能以“后端最多三条/四条”为理由删除化学上必要的独立变化。

在线性近似下，要实现目标增量 \(\delta\mathbf c\)，需解

\[
J\,\delta X=\delta\mathbf c.
\]

最小欧氏位移是

\[
\delta X=J^+\delta\mathbf c.
\tag{16}
\]

若 \(\delta\mathbf c\notin {\rm range}(J)\)，伪逆只给最小二乘近似，剩余残差不可忽略。小奇异值会放大位移，提示减小公共步长或修正函数/坐标集。式 (16) 只是一阶预测，必须再做非线性校正。

在一般正定度量 W 下，最小 \(\frac12\delta X^\top W\delta X\) 的可行预测为

\[
\delta X=W^{-1}J^\top
(JW^{-1}J^\top)^+\delta\mathbf c.
\]

每个 λ 点可先做几何投影检查

\[
\Phi(\mathbf c)=\min_X
\tfrac12\|S^{-1}[\mathbf q(X)-\mathbf c]\|^2
\]

并记录实际残差、碰撞、手性、组分装配和必要的拓扑条件。局部求解失败不能严格证明全局不可行；应记录诊断及多初值证据，不能悄悄修改目标值。

训练得到的正确 IRC 坐标值仍不足以唯一还原整帧。还要检验 \(P_Tg\)、分支和受限稳定性；不满足时扩大必要独立坐标或增加明确的构象控制。不能用全部原子冻结掩盖该问题并称其为内点松弛扫描。

## 8. 完整固定边界与路径连续性

必要条件：固定两端完整 XYZ、原子映射、电子态和几何哈希。两端按统一方法做单点能量，不再优化并覆盖它们。内部 λ 点做多约束松弛。

但 K 个目标坐标通常不足以确定 d 个自由度。因此

\[
\mathbf q(X_P')=\mathbf q(X_P)
\quad\not\Rightarrow\quad X_P'=X_P.
\]

仅靠单端独立极小化无法保证到达完整 X_P。建议执行层采用双端初值和受约束预测—校正，检查两条延续分支的相遇，并逐步加密跳变区。高不确定性、投影失败、约束残差、构象跳支及目标端点无法连接均触发明确失败。

若上述局部延续不能连接，可以在外部求解固定边界的联合路径问题。示例：

\[
\min_{X_1,\ldots,X_{n-1}}
\sum_{i=1}^{n-1}w_iE(X_i)
+\frac{\gamma}{2}\sum_{i=1}^{n-1}
\|X_{i+1}-2X_i+X_{i-1}\|_W^2,
\tag{17}
\]

\[
X_0=X_R,\quad X_n=X_P,\quad
\mathbf q(X_i)=\mathbf c(\lambda_i).
\]

对于非均匀 λ，第二差分须换为与步长一致的离散导数。实际还可把经尺度定义的相邻步长和路径分支作为限制。这里首尾是硬边界，γ 控制构造过程平滑性，ORCA 提供原始能量和梯度，外部协调器处理邻点耦合。

式 (17) 是建议的新算法，不是现有 ORCA Scan 功能，也不是每点完全独立的最低能松弛。它会影响内点几何；需要降低/比较 γ、保留参数与误差，并明确结果是受约束平滑路径。展示的物理 PES 能量始终是 E(X_i)，不能把平滑项当成量化能量。此方案仍保留一个 λ 和多坐标同步，没有要求把任务换成 NEB。

两端若不是所用方法下的相应驻点，或者属于不相连的电子态/分支，完整固定边界与“所有内点都是独立约束局部极小值”可能无法同时满足。应如实暴露冲突。数学合同保证成功结果的端点相同；求解器不能保证对任意体系必然产生合格连续路径。

## 9. 如果还要学习软约束强度

通用软约束可以定义为

\[
E_{\rm soft}(X)=E(X)
+\frac12[\mathbf q(X)-\mathbf c]^\top
K_{\rm soft}[\mathbf q(X)-\mathbf c],
\quad K_{\rm soft}\succeq0.
\tag{18}
\]

其梯度：

\[
\nabla_XE_{\rm soft}
=g+J^\top K_{\rm soft}(\mathbf q-\mathbf c).
\]

硬约束不会因受力而任意偏离目标；软约束会允许偏离。若局部软解接近硬约束解，由式 (2) 有

\[
K_{\rm soft}(\mathbf q-\mathbf c)\approx\boldsymbol\nu,
\quad \mathbf q-\mathbf c\approx K_{\rm soft}^{-1}\boldsymbol\nu.
\tag{19}
\]

这解释了约束越强误差通常越小，但式 (19) 是局部近似，不是任意非线性体系的保证。若乘子未知，可以在原始无偏置梯度可用时求

\[
\widehat{\boldsymbol\nu}=-(JJ^\top)^+Jg
\]

并同时记录 \(P_Tg\)；不能假设 ACP 当前已经导出准确乘子。

一个严格的一维二次例子更直观：

\[
E(q)=\tfrac12\kappa(q-a)^2,\quad
V(q)=\tfrac12k(q-c)^2,\quad\kappa>0.
\]

驻点给出

\[
q^*=\frac{\kappa a+kc}{\kappa+k},\qquad
|q^*-c|=\frac{\kappa|a-c|}{\kappa+k}.
\]

若希望误差不超过 ε，可选

\[
k\geq\max\{0,\ \kappa|a-c|/\epsilon-\kappa\}.
\tag{20}
\]

软强度由局部物理刚度、目标偏移及可接受误差共同决定。TS 附近存在负曲率时，正 κ 的推导不适用，需重新检查整体稳定性。过大 k 还会使 Hessian 条件数恶化。

若以后学习 K，应使用正定参数化，例如 \(K=LL^\top+\epsilon I\)，先标准化各坐标，再控制特征值与容差，避免不同单位造成假耦合。不确定性应触发验证/补样，不能自动转化为任意软化核心反应约束。

ORCA 6.1 的 bond bias 是特定形式的软势，不能把其参数直接当作式 (18) 中任意 B/A/D 耦合矩阵。通用实现需要外部优化器调用 ORCA 能量/梯度，或专门实现和验证额外势能/梯度集成。第一阶段建议核心坐标采用硬约束，软势仅用于经验证的辅助引导。

## 10. 执行架构与数据合同

~~~mermaid
flowchart LR
    A[映射双端与多键事件图] --> B[选择必要独立坐标]
    B --> C[机理条件模型预测各坐标函数]
    C --> D[固定边界并生成共享 λ 目标表]
    D --> E[几何可行性与身份核查]
    E --> F[ACP 逐点多约束 / 外部边界协调]
    F --> G[ORCA 能量与几何求解]
    G --> H[全驱动残差 / 连续性 / 目标事件验收]
    H --> I[一条正式 PES 与 ACP 展示]
    H --> J[不确定性补样及模型改进]
    J --> C
~~~

合同应至少包含：

~~~text
parameter_dimension = 1
lambda_values = [λ0, ..., λn]
drivers = [{id, kind, atom_map_ids, atom_indices, unit, values[0:n+1]}]
start_geometry, target_geometry, atom_order
endpoint_geometry_hashes, method, charge, multiplicity, environment
mechanism_id_or_distribution, model_version, uncertainty
execution_mode, boundary_policy, continuity_policy
~~~

每点记录 target/actual/residual 的全部驱动、结构、物理能量、收敛和失败原因；端点帧标明固定几何单点，内点标明约束松弛或联合路径修复。曲线横轴为共享 λ，前端可同时显示多条坐标变化。

ACP 必须消费冻结 values，不得悄悄根据 start/end 线性重算；返回逐项验证其目标表。超过部署数量上限或不支持的坐标组合应明确拒绝，不能降为单键。扩展坐标数量与自定义列表前须实跑代表性小体系并形成能力记录。

## 11. 建议分阶段实施

1. 修复执行定义：保持每反应一份计划、多条驱动、共享 λ、完整不可变双端和全驱动验收，接通 ACP 任意逐点 values。
2. 建立参考教师：整理 Demo24 TS/IRC，检查参考端点及方法差异，抽取目标函数；先比较线性、单参数指数函数与小规模样条，不训练大型通用网络。
3. 验证几何恢复：同一驱动集合下比较线性与参考目标函数的实际 ORCA 结果，检验是否改善目标事件、端点连接和 TS 几何，不能只比较目标数值拟合。
4. 建立跨反应学习：扩充各机理数据，按反应/骨架/机理划分；从分层低参数模型逐步扩展，校准不确定性。
5. 闭环改进：在失败、事件切换、高不确定性和 TS 邻域补算；学习型非线性函数、必要坐标与边界控制各自做消融。
6. 可选软势：在硬约束主链达标后，再研究辅助势和强度学习，始终保留无偏置物理能量。

正式交付仍为 24 条经过独立验收的主路径。开发中的对照试验、重试和模型训练记录分别计数，不能把它们展示为新增正式反应路径。

## 12. 验收与结论

| 层 | 最少需要证明 |
|---|---|
| 函数 | 端点数值严格保持；需单调的坐标单调；周期坐标正确；函数可行 |
| 请求 | 全部必要驱动保留；所有点共享 λ；ACP 实际采用学习目标值 |
| 数值 | 全驱动残差和自由梯度满足尺度一致的容差；所有帧可用 |
| 边界 | 完整首尾与唯一参考一致；不是优化后近似一致 |
| 路径 | 内点到固定边界连续；目标事件和立体/组分关系正确 |
| 几何 | 目标函数拟合与实际 TS/IRC 结构恢复分别报告 |
| 科学 | 需要报告严格 TS 时，另做 TS/频率/IRC 连接验证 |
| 泛化 | 测试反应 TS/IRC 不进入预测输入；分组拆分与不确定性校准 |
| 展示 | 执行完成、边界保持、路径合格、TS 近似和严格 TS 状态分开 |

ORCA 可以作为执行学习所得多坐标目标值的物理后端。PES generation 应学习机理与体系条件化的目标函数、事件时序及必要坐标；完整双端由硬边界规定。学习曲线、几何可行性和路径闭环三者都成立，才可能满足本项目要求。本方案没有声称现有 ACP 已经实现这些能力，也没有把分析中的解析例子当成新的计算成功结果。

相关：[上一轮完整问题审计](E:/Calculations/AI4S_ML_Studys/PES2TS/docs/reports/PES2TS_多键同步一维扫描全面问题报告_20261001.md)、[RXN_0000007104 原始审计记录](E:/Calculations/AI4S_ML_Studys/PES2TS/outputs/demo24_acp_geometry_audit_20261001/RXN_0000007104.json)。
