# Demo24：逐条图特征与扫描策略建议

> 2026-09-30；设计假设，24 条均仍为 needs_review，执行资格均为 false。

本表由白名单端点材料重算图特征和距离；策略建议为设计阶段人工明确列出的候选，不是选择器运行结果，也不是化学复核通过记录。原子号全部是 atom map；距离单位 Å。跨组分原始距离只用于诊断，装配通过后才能定扫描范围。

## 总览

| 反应 | 分层 / split | 主策略 | 驱动 B(map) | 建议首方向 |
| --- | --- | --- | --- | --- |
| RXN_0000007104 | A_H_transfer / train | H_TRANSFER · SINGLE_1D | 3–11 | P_to_R |
| RXN_0000161724 | A_H_transfer / train | H_TRANSFER · SINGLE_1D | 5–13 | P_to_R |
| RXN_0000091050 | A_H_transfer / valid | H_TRANSFER · SINGLE_1D | 7–9 | P_to_R |
| RXN_0000077619 | B_one_bond / train | LOCAL_CONNECTIVITY · SINGLE_1D | 2–5 | P_to_R |
| RXN_0000026256 | B_one_bond / train | LOCAL_CONNECTIVITY · SINGLE_1D | 3–7 | R_to_P |
| RXN_0000079731 | B_one_bond / valid | LOCAL_CONNECTIVITY · SINGLE_1D | 6–8 | R_to_P |
| RXN_0000053132 | C_substitution / train | CONNECTIVITY_EXCHANGE · COUPLED_1D | 5–6, 6–8 | P_to_R |
| RXN_0000155302 | C_substitution / train | CONNECTIVITY_EXCHANGE · COUPLED_1D | 1–2, 1–5 | P_to_R |
| RXN_0000171675 | C_substitution / valid | CONNECTIVITY_EXCHANGE · COUPLED_1D | 1–2, 1–5 | P_to_R |
| RXN_0000100071 | D_two_bond_coupled / train | RING_COUPLED · COUPLED_1D | 1–3, 2–4 | P_to_R |
| RXN_0000132222 | D_two_bond_coupled / train | H_TRANSFER_COUPLED · COUPLED_1D | 3–4, 5–16 | P_to_R |
| RXN_0000102998 | D_two_bond_coupled / valid | RING_COUPLED · COUPLED_1D | 2–7, 4–8 | P_to_R |
| RXN_0000059834 | E_three_bond_coupled / train | MULTI_EVENT_CONNECTED · COUPLED_1D | 1–10, 2–6, 3–6 | P_to_R |
| RXN_0000094438 | E_three_bond_coupled / train | MULTI_EVENT_CONNECTED · COUPLED_1D | 1–8, 2–7, 3–4 | P_to_R |
| RXN_0000187964 | E_three_bond_coupled / valid | MULTI_EVENT_CONNECTED · COUPLED_1D | 2–9, 3–4, 7–8 | P_to_R |
| RXN_0000100736 | F_H2_event / train | H2_EVENT · COUPLED_1D | 5–12, 5–13, 12–13 | P_to_R |
| RXN_0000149368 | F_H2_event / train | H2_EVENT · COUPLED_1D | 1–10, 3–13, 10–13 | R_to_P |
| RXN_0000017762 | F_H2_event / valid | H2_EVENT · COUPLED_1D | 2–9, 5–12, 9–12 | P_to_R |
| RXN_0000138453 | G_aromatic_order / train | AROMATIC_COUPLED · COUPLED_1D | 2–4, 2–6, 5–7 | R_to_P |
| RXN_0000110869 | G_aromatic_order / train | AROMATIC_COUPLED · COUPLED_1D | 5–8, 8–9 | P_to_R |
| RXN_0000047010 | G_aromatic_order / valid | AROMATIC_COUPLED · COUPLED_1D | 2–3, 5–6 | P_to_R |
| RXN_0000194484 | H_high_edit_fallback / train | NETWORK_PATH · PATH_REQUIRED | 双端路径，无固定 B | R_to_P |
| RXN_0000074997 | H_high_edit_fallback / train | NETWORK_PATH · PATH_REQUIRED | 双端路径，无固定 B | R_to_P |
| RXN_0000109608 | H_high_edit_fallback / valid | NETWORK_PATH · PATH_REQUIRED | 双端路径，无固定 B | P_to_R |

模式建议：{"SINGLE_1D": 6, "COUPLED_1D": 15, "PATH_REQUIRED": 3}。这些数字表示待验证候选数量。

所有条目通用监测：全目标编辑、非目标短接触/断裂、构型、逐点收敛与约束残差。H 转移另监测 D–A 距离及 D–H–A 角；环/芳香变化另监测对应区域的几何与电子指标。下列观察键是与驱动键不重复的全部编辑；驱动键本身也参与结果核验。

edit_cc 为含 F/B/O 的编辑图分量；fb_cc 只含连接变化。cycle_rank=E−V+C 为图不变量，不代表协同机理。端点芳香归一化采用当前 RDKit 默认解析，并逐键与冻结导出核对。chiral_tag 变化仅作复核提示；不能替代映射后的几何手性检验。

## RXN_0000007104 · A_H_transfer · train

- **编辑解释**：N1–H11→N3；1–2 键级变化是伴随电子重排。
- **图特征**：F/B/O=1/1/1；edit_cc=[[1, 2, 3, 11]]；fb_cc=[[1, 3, 11]]；编辑图 cycle_rank=0；端点 cycle_rank=1→1；组分=1→1。
- **主策略**：H_TRANSFER / SINGLE_1D；首方向 P_to_R。驱动：3–11。
- **观察键**：1–2, 1–11。
- **回退**：双 B：1–11、3–11；再比较反向。
- **拒绝风险**：邻接芳香骨架；不能把 1–2 的键级变化硬编码为独立成键。
- **账目**：映射状态 resolved_unique；R/P 总电荷 0/0，来源多重度 1/1；人工化学复核未完成。

RDKit 端点表示中的带电原子数：0→0；自由基电子数：0→0。这不是量化波函数的电子态诊断。

| 编辑 | map 对 | 键级 R→P | d(R) | d(P) | 跨组分侧 |
| --- | --- | --- | ---: | ---: | --- |
| order_changed | 1–2 | 1.0→2.0 | 1.345 | 1.255 | — |
| broken | 1–11 | 1.0→None | 1.006 | 2.719 | — |
| formed | 3–11 | None→1.0 | 2.627 | 1.006 | — |

## RXN_0000161724 · A_H_transfer · train

- **编辑解释**：O4–H13→N5，O+/N− 形式电荷消失。
- **图特征**：F/B/O=1/1/0；edit_cc=[[4, 5, 13]]；fb_cc=[[4, 5, 13]]；编辑图 cycle_rank=0；端点 cycle_rank=0→0；组分=1→1。
- **主策略**：H_TRANSFER / SINGLE_1D；首方向 P_to_R。驱动：5–13。
- **观察键**：4–13。
- **回退**：双 B：4–13、5–13；R→P 复核。
- **拒绝风险**：总电荷不变；形式电荷变化不能直接判断电子转移或自旋。
- **账目**：映射状态 resolved_unique；R/P 总电荷 0/0，来源多重度 1/1；人工化学复核未完成。

RDKit 端点表示中的带电原子数：2→0；自由基电子数：0→0。这不是量化波函数的电子态诊断。

| 编辑 | map 对 | 键级 R→P | d(R) | d(P) | 跨组分侧 |
| --- | --- | --- | ---: | ---: | --- |
| broken | 4–13 | 1.0→None | 0.966 | 1.957 | — |
| formed | 5–13 | None→1.0 | 2.011 | 1.017 | — |

原子属性提示：map 4 (formal_charge: 1→0)；map 5 (formal_charge: -1→0)。

## RXN_0000091050 · A_H_transfer · valid

- **编辑解释**：C1–H9→N7，产物形成 C−/N+。
- **图特征**：F/B/O=1/1/0；edit_cc=[[1, 7, 9]]；fb_cc=[[1, 7, 9]]；编辑图 cycle_rank=0；端点 cycle_rank=1→1；组分=1→1。
- **主策略**：H_TRANSFER / SINGLE_1D；首方向 P_to_R。驱动：7–9。
- **观察键**：1–9。
- **回退**：双 B：1–9、7–9；R→P 复核。
- **拒绝风险**：电荷分离、产物 N7 立体结构和供受体取向需复核。
- **账目**：映射状态 resolved_unique；R/P 总电荷 0/0，来源多重度 1/1；人工化学复核未完成。

RDKit 端点表示中的带电原子数：0→2；自由基电子数：0→0。这不是量化波函数的电子态诊断。

| 编辑 | map 对 | 键级 R→P | d(R) | d(P) | 跨组分侧 |
| --- | --- | --- | ---: | ---: | --- |
| broken | 1–9 | 1.0→None | 1.081 | 2.787 | — |
| formed | 7–9 | None→1.0 | 2.815 | 1.021 | — |

原子属性提示：map 1 (formal_charge: 0→-1)；map 7 (formal_charge: 0→1, chiral_tag: CHI_UNSPECIFIED→CHI_TETRAHEDRAL_CCW)。

## RXN_0000077619 · B_one_bond · train

- **编辑解释**：C2–C5 成键闭环；从产物已成键几何拉伸。
- **图特征**：F/B/O=1/0/0；edit_cc=[[2, 5]]；fb_cc=[[2, 5]]；编辑图 cycle_rank=0；端点 cycle_rank=0→1；组分=1→1。
- **主策略**：LOCAL_CONNECTIVITY / SINGLE_1D；首方向 P_to_R。驱动：2–5。
- **观察键**：无额外编辑键。
- **回退**：R→P，必要时经同骨架 D 预组织后重建计划。
- **拒绝风险**：环张力和新立体中心；保持目标构型。
- **账目**：映射状态 resolved_unique；R/P 总电荷 0/0，来源多重度 1/1；人工化学复核未完成。

RDKit 端点表示中的带电原子数：0→0；自由基电子数：2→0。这不是量化波函数的电子态诊断。

| 编辑 | map 对 | 键级 R→P | d(R) | d(P) | 跨组分侧 |
| --- | --- | --- | ---: | ---: | --- |
| formed | 2–5 | None→1.0 | 3.053 | 1.541 | — |

原子属性提示：map 2 (radical_electrons: 1→0, chiral_tag: CHI_UNSPECIFIED→CHI_TETRAHEDRAL_CW)；map 5 (radical_electrons: 1→0, chiral_tag: CHI_UNSPECIFIED→CHI_TETRAHEDRAL_CW)。

## RXN_0000026256 · B_one_bond · train

- **编辑解释**：C3–C7 断键开环。
- **图特征**：F/B/O=0/1/0；edit_cc=[[3, 7]]；fb_cc=[[3, 7]]；编辑图 cycle_rank=0；端点 cycle_rank=2→1；组分=1→1。
- **主策略**：LOCAL_CONNECTIVITY / SINGLE_1D；首方向 R_to_P。驱动：3–7。
- **观察键**：无额外编辑键。
- **回退**：P→R；若重闭环失败转 NEB。
- **拒绝风险**：小环开裂电子态和非目标骨架断裂。
- **账目**：映射状态 resolved_unique；R/P 总电荷 0/0，来源多重度 1/1；人工化学复核未完成。

RDKit 端点表示中的带电原子数：0→0；自由基电子数：0→2。这不是量化波函数的电子态诊断。

| 编辑 | map 对 | 键级 R→P | d(R) | d(P) | 跨组分侧 |
| --- | --- | --- | ---: | ---: | --- |
| broken | 3–7 | 1.0→None | 1.481 | 2.573 | — |

原子属性提示：map 3 (radical_electrons: 0→1, chiral_tag: CHI_TETRAHEDRAL_CW→CHI_UNSPECIFIED)；map 7 (radical_electrons: 0→1, chiral_tag: CHI_TETRAHEDRAL_CW→CHI_UNSPECIFIED)。

## RXN_0000079731 · B_one_bond · valid

- **编辑解释**：N6–N8 断裂，N6–N7 单键变双键。
- **图特征**：F/B/O=0/1/1；edit_cc=[[6, 7, 8]]；fb_cc=[[6, 8]]；编辑图 cycle_rank=0；端点 cycle_rank=2→1；组分=1→1。
- **主策略**：LOCAL_CONNECTIVITY / SINGLE_1D；首方向 R_to_P。驱动：6–8。
- **观察键**：6–7。
- **回退**：P→R；必要时 NEB。
- **拒绝风险**：resolved_symmetry_collapsed；等价映射必须对驱动键及立体目标不产生歧义。
- **账目**：映射状态 resolved_symmetry_collapsed；R/P 总电荷 0/0，来源多重度 1/1；人工化学复核未完成。

RDKit 端点表示中的带电原子数：2→0；自由基电子数：0→0。这不是量化波函数的电子态诊断。

| 编辑 | map 对 | 键级 R→P | d(R) | d(P) | 跨组分侧 |
| --- | --- | --- | ---: | ---: | --- |
| order_changed | 6–7 | 1.0→2.0 | 1.473 | 1.244 | — |
| broken | 6–8 | 1.0→None | 1.453 | 2.172 | — |

原子属性提示：map 7 (formal_charge: -1→0)；map 8 (formal_charge: 1→0)。

## RXN_0000053132 · C_substitution · train

- **编辑解释**：O6 从 N8 转接 C5；经 C5–C7–N8 的键级重排耦合。
- **图特征**：F/B/O=1/1/2；edit_cc=[[5, 6, 7, 8]]；fb_cc=[[5, 6, 8]]；编辑图 cycle_rank=1；端点 cycle_rank=0→0；组分=1→1。
- **主策略**：CONNECTIVITY_EXCHANGE / COUPLED_1D；首方向 P_to_R。驱动：5–6, 6–8。
- **观察键**：5–7, 7–8。
- **回退**：R→P；O6 转接偏早/偏晚时间表；NEB。
- **拒绝风险**：共有原子是迁移的 O6；不能仅凭 F/B 共享原子命名为 SN2。
- **账目**：映射状态 resolved_unique；R/P 总电荷 0/0，来源多重度 1/1；人工化学复核未完成。

RDKit 端点表示中的带电原子数：0→0；自由基电子数：0→0。这不是量化波函数的电子态诊断。

| 编辑 | map 对 | 键级 R→P | d(R) | d(P) | 跨组分侧 |
| --- | --- | --- | ---: | ---: | --- |
| formed | 5–6 | None→1.0 | 3.431 | 1.412 | — |
| order_changed | 5–7 | 2.0→1.0 | 1.304 | 1.480 | — |
| broken | 6–8 | 1.0→None | 1.431 | 3.395 | — |
| order_changed | 7–8 | 2.0→3.0 | 1.229 | 1.151 | — |

原子属性提示：map 5 (chiral_tag: CHI_UNSPECIFIED→CHI_TETRAHEDRAL_CCW)。

## RXN_0000155302 · C_substitution · train

- **编辑解释**：C1 从 C5 转接 C2；C2–N4–C5 键级重排；HCl 为图上不变组分。
- **图特征**：F/B/O=1/1/2；edit_cc=[[1, 2, 4, 5]]；fb_cc=[[1, 2, 5]]；编辑图 cycle_rank=1；端点 cycle_rank=0→0；组分=2→2。
- **主策略**：CONNECTIVITY_EXCHANGE / COUPLED_1D；首方向 P_to_R。驱动：1–2, 1–5。
- **观察键**：2–4, 4–5。
- **回退**：装配合格后 R→P；两种转接时间表；NEB。
- **拒绝风险**：2→2 组分；HCl 可能参与相互作用，不能擅自删除或永久冻结。
- **账目**：映射状态 resolved_unique；R/P 总电荷 0/0，来源多重度 1/1；人工化学复核未完成。

RDKit 端点表示中的带电原子数：0→0；自由基电子数：0→0。这不是量化波函数的电子态诊断。

| 编辑 | map 对 | 键级 R→P | d(R) | d(P) | 跨组分侧 |
| --- | --- | --- | ---: | ---: | --- |
| formed | 1–2 | None→1.0 | 3.611 | 1.505 | — |
| broken | 1–5 | 1.0→None | 1.513 | 3.307 | — |
| order_changed | 2–4 | 2.0→1.0 | 1.209 | 1.422 | — |
| order_changed | 4–5 | 1.0→2.0 | 1.402 | 1.235 | — |

## RXN_0000171675 · C_substitution · valid

- **编辑解释**：C1 从 C5 转接 C2，另有两处 C–N 键级改变。
- **图特征**：F/B/O=1/1/2；edit_cc=[[1, 2, 4, 5, 7]]；fb_cc=[[1, 2, 5]]；编辑图 cycle_rank=0；端点 cycle_rank=0→0；组分=1→1。
- **主策略**：CONNECTIVITY_EXCHANGE / COUPLED_1D；首方向 P_to_R。驱动：1–2, 1–5。
- **观察键**：2–7, 4–5。
- **回退**：R→P；转接偏早/偏晚；NEB。
- **拒绝风险**：电荷分离和含 N 骨架耦合；共享 C1 不等于已知取代机理。
- **账目**：映射状态 resolved_unique；R/P 总电荷 0/0，来源多重度 1/1；人工化学复核未完成。

RDKit 端点表示中的带电原子数：0→2；自由基电子数：0→0。这不是量化波函数的电子态诊断。

| 编辑 | map 对 | 键级 R→P | d(R) | d(P) | 跨组分侧 |
| --- | --- | --- | ---: | ---: | --- |
| formed | 1–2 | None→1.0 | 3.441 | 1.536 | — |
| broken | 1–5 | 1.0→None | 1.551 | 3.913 | — |
| order_changed | 2–7 | 2.0→1.0 | 1.264 | 1.362 | — |
| order_changed | 4–5 | 1.0→2.0 | 1.446 | 1.311 | — |

原子属性提示：map 4 (formal_charge: 0→1)；map 7 (formal_charge: 0→-1)。

## RXN_0000100071 · D_two_bond_coupled · train

- **编辑解释**：两根 O–C 成键与 C3–O4 断裂、O1–C2 降键级耦合。
- **图特征**：F/B/O=2/1/1；edit_cc=[[1, 2, 3, 4]]；fb_cc=[[1, 2, 3, 4]]；编辑图 cycle_rank=1；端点 cycle_rank=1→2；组分=1→1。
- **主策略**：RING_COUPLED / COUPLED_1D；首方向 P_to_R。驱动：1–3, 2–4。
- **观察键**：1–2, 3–4。
- **回退**：若 3–4 不回成键，三 B 加入 3–4；再反向/NEB。
- **拒绝风险**：多环变化；第三根键作为监测量有待试点证明。
- **账目**：映射状态 resolved_unique；R/P 总电荷 0/0，来源多重度 1/1；人工化学复核未完成。

RDKit 端点表示中的带电原子数：0→0；自由基电子数：0→0。这不是量化波函数的电子态诊断。

| 编辑 | map 对 | 键级 R→P | d(R) | d(P) | 跨组分侧 |
| --- | --- | --- | ---: | ---: | --- |
| order_changed | 1–2 | 2.0→1.0 | 1.219 | 1.424 | — |
| formed | 1–3 | None→1.0 | 2.383 | 1.467 | — |
| formed | 2–4 | None→1.0 | 2.504 | 1.390 | — |
| broken | 3–4 | 1.0→None | 1.433 | 2.450 | — |

原子属性提示：map 2 (chiral_tag: CHI_UNSPECIFIED→CHI_TETRAHEDRAL_CW)。

## RXN_0000132222 · D_two_bond_coupled · train

- **编辑解释**：O3–C4 闭环并伴 H16 从 O3 转移到 O5。
- **图特征**：F/B/O=2/1/1；edit_cc=[[3, 4, 5, 16]]；fb_cc=[[3, 4, 5, 16]]；编辑图 cycle_rank=1；端点 cycle_rank=1→2；组分=1→1。
- **主策略**：H_TRANSFER_COUPLED / COUPLED_1D；首方向 P_to_R。驱动：3–4, 5–16。
- **观察键**：3–16, 4–5。
- **回退**：三 B 加入 3–16；H 转移偏早/偏晚；NEB。
- **拒绝风险**：H 与重原子事件的时序未知；不能因同一编辑环就断言同步。
- **账目**：映射状态 resolved_unique；R/P 总电荷 0/0，来源多重度 1/1；人工化学复核未完成。

RDKit 端点表示中的带电原子数：0→0；自由基电子数：0→0。这不是量化波函数的电子态诊断。

| 编辑 | map 对 | 键级 R→P | d(R) | d(P) | 跨组分侧 |
| --- | --- | --- | ---: | ---: | --- |
| formed | 3–4 | None→1.0 | 3.354 | 1.422 | — |
| broken | 3–16 | 1.0→None | 0.970 | 2.324 | — |
| order_changed | 4–5 | 2.0→1.0 | 1.209 | 1.392 | — |
| formed | 5–16 | None→1.0 | 1.942 | 0.966 | — |

## RXN_0000102998 · D_two_bond_coupled · valid

- **编辑解释**：C2–C7、C4–O8 成键；C2–C4 断裂；C7–O8 降键级。
- **图特征**：F/B/O=2/1/1；edit_cc=[[2, 4, 7, 8]]；fb_cc=[[2, 4, 7, 8]]；编辑图 cycle_rank=1；端点 cycle_rank=0→1；组分=1→1。
- **主策略**：RING_COUPLED / COUPLED_1D；首方向 P_to_R。驱动：2–7, 4–8。
- **观察键**：2–4, 7–8。
- **回退**：三 B 加入 2–4；再反向/NEB。
- **拒绝风险**：环应变及未驱动 2–4 的完成情况。
- **账目**：映射状态 resolved_unique；R/P 总电荷 0/0，来源多重度 1/1；人工化学复核未完成。

RDKit 端点表示中的带电原子数：0→0；自由基电子数：0→0。这不是量化波函数的电子态诊断。

| 编辑 | map 对 | 键级 R→P | d(R) | d(P) | 跨组分侧 |
| --- | --- | --- | ---: | ---: | --- |
| broken | 2–4 | 1.0→None | 1.549 | 2.601 | — |
| formed | 2–7 | None→1.0 | 2.537 | 1.512 | — |
| formed | 4–8 | None→1.0 | 2.409 | 1.358 | — |
| order_changed | 7–8 | 2.0→1.0 | 1.196 | 1.495 | — |

原子属性提示：map 7 (chiral_tag: CHI_UNSPECIFIED→CHI_TETRAHEDRAL_CCW)。

## RXN_0000059834 · E_three_bond_coupled · train

- **编辑解释**：Cl10 转接 C1，C6 同时与 C2/C3 成键；多事件共中心 C6。
- **图特征**：F/B/O=3/2/1；edit_cc=[[1, 2, 3, 6, 10]]；fb_cc=[[1, 2, 3, 6, 10]]；编辑图 cycle_rank=2；端点 cycle_rank=1→2；组分=1→1。
- **主策略**：MULTI_EVENT_CONNECTED / COUPLED_1D；首方向 P_to_R。驱动：1–10, 2–6, 3–6。
- **观察键**：1–6, 2–3, 6–10。
- **回退**：NEB 优先回退；只在预冻结预算内比较分组时间表。
- **拒绝风险**：5 条连接编辑；3 个驱动仅是降维假设，两个断键都需监测。
- **账目**：映射状态 resolved_unique；R/P 总电荷 0/0，来源多重度 1/1；人工化学复核未完成。

RDKit 端点表示中的带电原子数：0→0；自由基电子数：2→2。这不是量化波函数的电子态诊断。

| 编辑 | map 对 | 键级 R→P | d(R) | d(P) | 跨组分侧 |
| --- | --- | --- | ---: | ---: | --- |
| broken | 1–6 | 1.0→None | 1.537 | 2.640 | — |
| formed | 1–10 | None→1.0 | 2.763 | 1.856 | — |
| order_changed | 2–3 | 2.0→1.0 | 1.343 | 1.514 | — |
| formed | 2–6 | None→1.0 | 2.505 | 1.501 | — |
| formed | 3–6 | None→1.0 | 2.789 | 1.526 | — |
| broken | 6–10 | 1.0→None | 1.836 | 3.493 | — |

原子属性提示：map 1 (chiral_tag: CHI_TETRAHEDRAL_CCW→CHI_TETRAHEDRAL_CW)；map 2 (chiral_tag: CHI_UNSPECIFIED→CHI_TETRAHEDRAL_CW)；map 3 (chiral_tag: CHI_UNSPECIFIED→CHI_TETRAHEDRAL_CW)；map 6 (chiral_tag: CHI_TETRAHEDRAL_CCW→CHI_TETRAHEDRAL_CW)。

## RXN_0000094438 · E_three_bond_coupled · train

- **编辑解释**：H8 转移与 N2–O7、C3–N4 两处闭环耦合。
- **图特征**：F/B/O=3/1/2；edit_cc=[[1, 2, 3, 4, 7, 8]]；fb_cc=[[1, 2, 7, 8], [3, 4]]；编辑图 cycle_rank=1；端点 cycle_rank=0→2；组分=1→1。
- **主策略**：MULTI_EVENT_CONNECTED / COUPLED_1D；首方向 P_to_R。驱动：1–8, 2–7, 3–4。
- **观察键**：1–4, 2–8, 3–7。
- **回退**：NEB；若发现稳定中间体再提交分阶段新版计划。
- **拒绝风险**：完整 H 双距离再加两成键需 4 B，不能塞入原生 Simul_Scan。
- **账目**：映射状态 resolved_unique；R/P 总电荷 0/0，来源多重度 1/1；人工化学复核未完成。

RDKit 端点表示中的带电原子数：0→0；自由基电子数：0→0。这不是量化波函数的电子态诊断。

| 编辑 | map 对 | 键级 R→P | d(R) | d(P) | 跨组分侧 |
| --- | --- | --- | ---: | ---: | --- |
| order_changed | 1–4 | 2.0→1.0 | 1.267 | 1.470 | — |
| formed | 1–8 | None→1.0 | 2.035 | 1.094 | — |
| formed | 2–7 | None→1.0 | 2.301 | 1.508 | — |
| broken | 2–8 | 1.0→None | 1.013 | 2.139 | — |
| formed | 3–4 | None→1.0 | 3.574 | 1.440 | — |
| order_changed | 3–7 | 2.0→1.0 | 1.209 | 1.395 | — |

## RXN_0000187964 · E_three_bond_coupled · valid

- **编辑解释**：F9 转接、含氧环重组与 C2–O4 降键级形成相连编辑网络。
- **图特征**：F/B/O=3/2/1；edit_cc=[[2, 3, 4, 7, 8, 9]]；fb_cc=[[2, 3, 4, 7, 8, 9]]；编辑图 cycle_rank=1；端点 cycle_rank=1→2；组分=1→1。
- **主策略**：MULTI_EVENT_CONNECTED / COUPLED_1D；首方向 P_to_R。驱动：2–9, 3–4, 7–8。
- **观察键**：2–4, 3–7, 8–9。
- **回退**：NEB；或有几何依据的有限时序变体。
- **拒绝风险**：5 条连接编辑；三个成键距离不能证明两个断键会随动。
- **账目**：映射状态 resolved_unique；R/P 总电荷 0/0，来源多重度 1/1；人工化学复核未完成。

RDKit 端点表示中的带电原子数：0→0；自由基电子数：0→0。这不是量化波函数的电子态诊断。

| 编辑 | map 对 | 键级 R→P | d(R) | d(P) | 跨组分侧 |
| --- | --- | --- | ---: | ---: | --- |
| order_changed | 2–4 | 2.0→1.0 | 1.212 | 1.398 | — |
| formed | 2–9 | None→1.0 | 3.257 | 1.367 | — |
| formed | 3–4 | None→1.0 | 2.369 | 1.455 | — |
| broken | 3–7 | 1.0→None | 1.445 | 2.437 | — |
| formed | 7–8 | None→1.0 | 2.352 | 1.504 | — |
| broken | 8–9 | 1.0→None | 1.438 | 3.847 | — |

原子属性提示：map 2 (chiral_tag: CHI_UNSPECIFIED→CHI_TETRAHEDRAL_CCW)；map 3 (chiral_tag: CHI_UNSPECIFIED→CHI_TETRAHEDRAL_CCW)；map 6 (chiral_tag: CHI_UNSPECIFIED→CHI_TETRAHEDRAL_CW)。

## RXN_0000100736 · F_H2_event · train

- **编辑解释**：H2 加到同一 C5，同时 N1–C5 开环及 N1–C2 升键级。
- **图特征**：F/B/O=2/2/1；edit_cc=[[1, 2, 5, 12, 13]]；fb_cc=[[1, 5, 12, 13]]；编辑图 cycle_rank=1；端点 cycle_rank=1→0；组分=2→1。
- **主策略**：H2_EVENT / COUPLED_1D；首方向 P_to_R。驱动：5–12, 5–13, 12–13。
- **观察键**：1–2, 1–5。
- **回退**：NEB；只有证实环事件可随动才保留三 B 候选。
- **拒绝风险**：R 为 2 组分，先装配；H–H 事件去重；三距离须满足三角不等式。
- **账目**：映射状态 resolved_unique；R/P 总电荷 0/0，来源多重度 1/1；人工化学复核未完成。

RDKit 端点表示中的带电原子数：0→0；自由基电子数：2→0。这不是量化波函数的电子态诊断。

| 编辑 | map 对 | 键级 R→P | d(R) | d(P) | 跨组分侧 |
| --- | --- | --- | ---: | ---: | --- |
| order_changed | 1–2 | 1.0→2.0 | 1.326 | 1.278 | — |
| broken | 1–5 | 1.0→None | 1.500 | 2.645 | — |
| formed | 5–12 | None→1.0 | 1.829 | 1.088 | R |
| formed | 5–13 | None→1.0 | 1.909 | 1.089 | R |
| broken | 12–13 | 1.0→None | 0.744 | 1.781 | — |

原子属性提示：map 2 (radical_electrons: 1→0)；map 5 (radical_electrons: 1→0)。

## RXN_0000149368 · F_H2_event · train

- **编辑解释**：C1–H10、C3–H13 断裂生成 H2，另伴 N4–C2 开环。
- **图特征**：F/B/O=1/3/2；edit_cc=[[1, 2, 3, 4, 10, 13]]；fb_cc=[[1, 3, 10, 13], [2, 4]]；编辑图 cycle_rank=1；端点 cycle_rank=1→0；组分=1→2。
- **主策略**：H2_EVENT / COUPLED_1D；首方向 R_to_P。驱动：1–10, 3–13, 10–13。
- **观察键**：1–2, 2–4, 3–4。
- **回退**：重新装配 P 后再生成数值时间表；NEB。
- **拒绝风险**：P 中已断开的 C3–H13 只有约 0.754 Å，禁止直接插值；现有跨组分摆放需修复。
- **账目**：映射状态 resolved_unique；R/P 总电荷 0/0，来源多重度 1/1；人工化学复核未完成。

RDKit 端点表示中的带电原子数：0→0；自由基电子数：0→0。这不是量化波函数的电子态诊断。

| 编辑 | map 对 | 键级 R→P | d(R) | d(P) | 跨组分侧 |
| --- | --- | --- | ---: | ---: | --- |
| order_changed | 1–2 | 1.0→2.0 | 1.506 | 1.333 | — |
| broken | 1–10 | 1.0→None | 1.088 | 2.496 | P |
| broken | 2–4 | 1.0→None | 1.471 | 2.426 | — |
| order_changed | 3–4 | 1.0→2.0 | 1.481 | 1.290 | — |
| broken | 3–13 | 1.0→None | 1.084 | 0.754 | P |
| formed | 10–13 | None→1.0 | 2.480 | 0.744 | — |

原子属性提示：map 2 (chiral_tag: CHI_TETRAHEDRAL_CCW→CHI_UNSPECIFIED)；map 3 (chiral_tag: CHI_TETRAHEDRAL_CCW→CHI_UNSPECIFIED)。

## RXN_0000017762 · F_H2_event · valid

- **编辑解释**：H2 消耗到 C2/C5，另形成 O7–O8 并改变两个羰基键级。
- **图特征**：F/B/O=3/1/2；edit_cc=[[2, 5, 7, 8, 9, 12]]；fb_cc=[[2, 5, 9, 12], [7, 8]]；编辑图 cycle_rank=1；端点 cycle_rank=0→1；组分=2→1。
- **主策略**：H2_EVENT / COUPLED_1D；首方向 P_to_R。驱动：2–9, 5–12, 9–12。
- **观察键**：2–8, 5–7, 7–8。
- **回退**：若 O7–O8 不随动，转 NEB；不静默加第四个 Scan。
- **拒绝风险**：R 为 2 组分；H2 朝向、O–O 成键及多中心时序均需检查。
- **账目**：映射状态 resolved_unique；R/P 总电荷 0/0，来源多重度 1/1；人工化学复核未完成。

RDKit 端点表示中的带电原子数：0→0；自由基电子数：0→0。这不是量化波函数的电子态诊断。

| 编辑 | map 对 | 键级 R→P | d(R) | d(P) | 跨组分侧 |
| --- | --- | --- | ---: | ---: | --- |
| order_changed | 2–8 | 2.0→1.0 | 1.186 | 1.392 | — |
| formed | 2–9 | None→1.0 | 1.499 | 1.087 | R |
| order_changed | 5–7 | 2.0→1.0 | 1.177 | 1.428 | — |
| formed | 5–12 | None→1.0 | 1.498 | 1.090 | R |
| formed | 7–8 | None→1.0 | 3.248 | 1.466 | — |
| broken | 9–12 | 1.0→None | 0.744 | 3.825 | — |

## RXN_0000138453 · G_aromatic_order · train

- **编辑解释**：C2 转接 N4、C5–C7 开裂伴五元环芳香化。
- **图特征**：F/B/O=1/2/5；edit_cc=[[2, 4, 5, 6, 7, 8]]；fb_cc=[[2, 4, 6], [5, 7]]；编辑图 cycle_rank=3；端点 cycle_rank=2→1；组分=1→1。
- **主策略**：AROMATIC_COUPLED / COUPLED_1D；首方向 R_to_P。驱动：2–4, 2–6, 5–7。
- **观察键**：4–5, 4–8, 5–6, 6–7, 7–8。
- **回退**：P→R；NEB。
- **拒绝风险**：5 个芳香键级变化按区域监测；不是 5 根扫描键，也不全是表示噪声。
- **账目**：映射状态 resolved_unique；R/P 总电荷 0/0，来源多重度 1/1；人工化学复核未完成。

RDKit 端点表示中的带电原子数：0→0；自由基电子数：0→0。这不是量化波函数的电子态诊断。

| 编辑 | map 对 | 键级 R→P | d(R) | d(P) | 跨组分侧 |
| --- | --- | --- | ---: | ---: | --- |
| formed | 2–4 | None→1.0 | 2.958 | 1.395 | — |
| broken | 2–6 | 1.0→None | 1.484 | 3.545 | — |
| order_changed | 4–5 | 1.0→1.5 | 1.505 | 1.375 | — |
| order_changed | 4–8 | 2.0→1.5 | 1.257 | 1.365 | — |
| order_changed | 5–6 | 1.0→1.5 | 1.504 | 1.365 | — |
| broken | 5–7 | 1.0→None | 1.502 | 2.210 | — |
| order_changed | 6–7 | 1.0→1.5 | 1.518 | 1.421 | — |
| order_changed | 7–8 | 1.0→1.5 | 1.499 | 1.317 | — |

原子属性提示：map 4 (aromatic: False→True)；map 5 (aromatic: False→True, chiral_tag: CHI_TETRAHEDRAL_CW→CHI_UNSPECIFIED)；map 6 (aromatic: False→True)；map 7 (aromatic: False→True, chiral_tag: CHI_TETRAHEDRAL_CW→CHI_UNSPECIFIED)；map 8 (aromatic: False→True)。

## RXN_0000110869 · G_aromatic_order · train

- **编辑解释**：Cl8 从 N5 转到 C9，六元区域芳香化。
- **图特征**：F/B/O=1/1/6；edit_cc=[[2, 3, 4, 5, 6, 8, 9]]；fb_cc=[[5, 8, 9]]；编辑图 cycle_rank=2；端点 cycle_rank=1→1；组分=1→1。
- **主策略**：AROMATIC_COUPLED / COUPLED_1D；首方向 P_to_R。驱动：5–8, 8–9。
- **观察键**：2–3, 2–9, 3–4, 4–5, 5–6, 6–9。
- **回退**：R→P；Cl 转接早/晚；NEB。
- **拒绝风险**：6 项环键级变化压成一个区域观察对象，保留各键证据。
- **账目**：映射状态 resolved_unique；R/P 总电荷 0/0，来源多重度 1/1；人工化学复核未完成。

RDKit 端点表示中的带电原子数：0→0；自由基电子数：0→0。这不是量化波函数的电子态诊断。

| 编辑 | map 对 | 键级 R→P | d(R) | d(P) | 跨组分侧 |
| --- | --- | --- | ---: | ---: | --- |
| order_changed | 2–3 | 1.0→1.5 | 1.396 | 1.382 | — |
| order_changed | 2–9 | 2.0→1.5 | 1.375 | 1.390 | — |
| order_changed | 3–4 | 2.0→1.5 | 1.374 | 1.388 | — |
| order_changed | 4–5 | 1.0→1.5 | 1.350 | 1.331 | — |
| order_changed | 5–6 | 1.0→1.5 | 1.377 | 1.336 | — |
| broken | 5–8 | 1.0→None | 1.831 | 4.008 | — |
| order_changed | 6–9 | 2.0→1.5 | 1.372 | 1.402 | — |
| formed | 8–9 | None→1.0 | 4.068 | 1.747 | — |

原子属性提示：map 2 (aromatic: False→True)；map 3 (aromatic: False→True)；map 4 (aromatic: False→True)；map 5 (aromatic: False→True)；map 6 (aromatic: False→True)；map 9 (aromatic: False→True)。

## RXN_0000047010 · G_aromatic_order · valid

- **编辑解释**：C2–N3 断裂、O5–N6 成键使环骨架扩展并芳香化。
- **图特征**：F/B/O=1/1/5；edit_cc=[[2, 3, 4, 5, 6, 7]]；fb_cc=[[2, 3], [5, 6]]；编辑图 cycle_rank=2；端点 cycle_rank=1→1；组分=1→1。
- **主策略**：AROMATIC_COUPLED / COUPLED_1D；首方向 P_to_R。驱动：2–3, 5–6。
- **观察键**：2–4, 2–7, 3–4, 4–5, 6–7。
- **回退**：R→P；NEB。
- **拒绝风险**：5–6 是真实新增连接（产物键级 1.5），不能被芳香归一化删除。
- **账目**：映射状态 resolved_unique；R/P 总电荷 0/0，来源多重度 1/1；人工化学复核未完成。

RDKit 端点表示中的带电原子数：0→0；自由基电子数：0→0。这不是量化波函数的电子态诊断。

| 编辑 | map 对 | 键级 R→P | d(R) | d(P) | 跨组分侧 |
| --- | --- | --- | ---: | ---: | --- |
| broken | 2–3 | 1.0→None | 1.510 | 2.453 | — |
| order_changed | 2–4 | 1.0→1.5 | 1.507 | 1.458 | — |
| order_changed | 2–7 | 1.0→1.5 | 1.490 | 1.343 | — |
| order_changed | 3–4 | 1.0→2.0 | 1.345 | 1.262 | — |
| order_changed | 4–5 | 2.0→1.5 | 1.190 | 1.400 | — |
| formed | 5–6 | None→1.5 | 3.831 | 1.430 | — |
| order_changed | 6–7 | 2.0→1.5 | 1.266 | 1.386 | — |

原子属性提示：map 2 (aromatic: False→True, chiral_tag: CHI_TETRAHEDRAL_CW→CHI_UNSPECIFIED)；map 4 (aromatic: False→True)；map 5 (aromatic: False→True)；map 6 (aromatic: False→True)；map 7 (aromatic: False→True)。

## RXN_0000194484 · H_high_edit_fallback · train

- **编辑解释**：H10 转移、三根重原子成键、三个键级变化形成密集环重组。
- **图特征**：F/B/O=4/1/3；edit_cc=[[1, 2, 3, 4, 8, 9, 10]]；fb_cc=[[1, 2, 9, 10], [3, 4, 8]]；编辑图 cycle_rank=2；端点 cycle_rank=0→3；组分=1→1。
- **主策略**：NETWORK_PATH / PATH_REQUIRED；首方向 R_to_P。驱动：双端全几何，无固定距离驱动。
- **观察键**：1–2, 1–9, 1–10, 2–10, 3–4, 3–8, 4–8, 8–9。
- **回退**：双端 NEB；稳定中间体确认后分阶段。
- **拒绝风险**：H 双距离加三根重原子新键需 5 B；选任意 3 根没有充分事件覆盖依据。
- **账目**：映射状态 resolved_unique；R/P 总电荷 0/0，来源多重度 1/1；人工化学复核未完成。

RDKit 端点表示中的带电原子数：0→0；自由基电子数：0→0。这不是量化波函数的电子态诊断。

| 编辑 | map 对 | 键级 R→P | d(R) | d(P) | 跨组分侧 |
| --- | --- | --- | ---: | ---: | --- |
| order_changed | 1–2 | 2.0→1.0 | 1.336 | 1.531 | — |
| formed | 1–9 | None→2.0 | 3.899 | 1.273 | — |
| broken | 1–10 | 1.0→None | 1.083 | 2.176 | — |
| formed | 2–10 | None→1.0 | 2.132 | 1.094 | — |
| order_changed | 3–4 | 2.0→1.0 | 1.344 | 1.518 | — |
| formed | 3–8 | None→1.0 | 2.497 | 1.591 | — |
| formed | 4–8 | None→1.0 | 3.433 | 1.498 | — |
| order_changed | 8–9 | 3.0→1.0 | 1.151 | 1.442 | — |

原子属性提示：map 3 (chiral_tag: CHI_UNSPECIFIED→CHI_TETRAHEDRAL_CW)；map 4 (chiral_tag: CHI_UNSPECIFIED→CHI_TETRAHEDRAL_CW)；map 8 (chiral_tag: CHI_UNSPECIFIED→CHI_TETRAHEDRAL_CCW)。

## RXN_0000074997 · H_high_edit_fallback · train

- **编辑解释**：四根断键、一根成键与三个升键级分布在相连多环骨架。
- **图特征**：F/B/O=1/4/3；edit_cc=[[1, 2, 3, 5, 6, 8, 9, 10]]；fb_cc=[[1, 6], [2, 10], [3, 5, 8, 9]]；编辑图 cycle_rank=1；端点 cycle_rank=4→1；组分=1→1。
- **主策略**：NETWORK_PATH / PATH_REQUIRED；首方向 R_to_P。驱动：双端全几何，无固定距离驱动。
- **观察键**：1–2, 1–6, 2–10, 3–8, 3–9, 5–6, 5–8, 9–10。
- **回退**：双端 NEB；有中间体证据后拆为多个基元段。
- **拒绝风险**：不能因 F=1 就只扫描 3–8；1–6 原始距离变化达约 4.36 Å，构象变化显著。
- **账目**：映射状态 resolved_unique；R/P 总电荷 0/0，来源多重度 1/1；人工化学复核未完成。

RDKit 端点表示中的带电原子数：0→0；自由基电子数：0→0。这不是量化波函数的电子态诊断。

| 编辑 | map 对 | 键级 R→P | d(R) | d(P) | 跨组分侧 |
| --- | --- | --- | ---: | ---: | --- |
| order_changed | 1–2 | 2.0→3.0 | 1.288 | 1.160 | — |
| broken | 1–6 | 1.0→None | 1.507 | 5.862 | — |
| broken | 2–10 | 1.0→None | 1.418 | 4.287 | — |
| formed | 3–8 | None→1.0 | 2.376 | 1.480 | — |
| broken | 3–9 | 1.0→None | 1.502 | 2.476 | — |
| order_changed | 5–6 | 1.0→2.0 | 1.410 | 1.199 | — |
| broken | 5–8 | 1.0→None | 1.525 | 2.427 | — |
| order_changed | 9–10 | 1.0→2.0 | 1.482 | 1.202 | — |

原子属性提示：map 5 (chiral_tag: CHI_TETRAHEDRAL_CW→CHI_UNSPECIFIED)；map 8 (chiral_tag: CHI_TETRAHEDRAL_CW→CHI_TETRAHEDRAL_CCW)；map 9 (chiral_tag: CHI_TETRAHEDRAL_CCW→CHI_UNSPECIFIED)。

## RXN_0000109608 · H_high_edit_fallback · valid

- **编辑解释**：两组三键片段形成四条交叉连接，得到紧密四原子网络。
- **图特征**：F/B/O=4/0/2；edit_cc=[[3, 4, 5, 6]]；fb_cc=[[3, 4, 5, 6]]；编辑图 cycle_rank=3；端点 cycle_rank=0→3；组分=2→1。
- **主策略**：NETWORK_PATH / PATH_REQUIRED；首方向 P_to_R。驱动：双端全几何，无固定距离驱动。
- **观察键**：3–4, 3–5, 3–6, 4–5, 4–6, 5–6。
- **回退**：重建 R 侧组分装配后 NEB；有可靠中间体再分段。
- **拒绝风险**：R 中未来成键 3–4、3–5 已比 P 更短；原始跨组分距离不可作形成窗口。
- **账目**：映射状态 resolved_unique；R/P 总电荷 0/0，来源多重度 1/1；人工化学复核未完成。

RDKit 端点表示中的带电原子数：0→0；自由基电子数：0→0。这不是量化波函数的电子态诊断。

| 编辑 | map 对 | 键级 R→P | d(R) | d(P) | 跨组分侧 |
| --- | --- | --- | ---: | ---: | --- |
| formed | 3–4 | None→1.0 | 1.424 | 1.493 | R |
| formed | 3–5 | None→1.0 | 1.361 | 1.439 | R |
| order_changed | 3–6 | 3.0→1.0 | 1.200 | 1.439 | — |
| order_changed | 4–5 | 3.0→1.0 | 1.148 | 1.488 | — |
| formed | 4–6 | None→1.0 | 2.444 | 1.488 | R |
| formed | 5–6 | None→1.0 | 2.408 | 1.448 | R |

原子属性提示：map 5 (chiral_tag: CHI_UNSPECIFIED→CHI_TETRAHEDRAL_CW)；map 6 (chiral_tag: CHI_UNSPECIFIED→CHI_TETRAHEDRAL_CCW)。

## 溯源与复算

- 输入候选 CSV SHA256：`63ac74ee55e3367f8f937c897620a5cf9993275df22b89e4ac9c2d48644e489c`。
- ReactionCase manifest SHA256：`81922e28ab5c1748c203d467e07e0adfb3653eed4088c77044951e81596d9fd3`。
- RDKit：`2026.03.3`。
- 逐条导出哈希、案例哈希、支持路径、完整数值见同目录 `demo24_strategy_evidence.json`。
- `build_evidence.py` 可复算；只写本目录两个设计证据文件，不修改正式案例、不提交计算。
