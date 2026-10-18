# PES2TS / PES-N V5–V7：终审与 V5.0–V5.2 第一阶段 10 小时冲刺执行方案

> 状态：Proposed；设计交付，未实现本方案脚本、未运行训练或 QC
> 层级：G3.P.0–.3（V5.0–V5.2 主体）、G3.2（V6 数据/决策）、G4.0/G4.1（V7-B/C）、G2.T（M0.5 Oracle / M6.5）；G0/G1 仅数据资格增量
> G–V 映射：`config/registry/g_v_map.yaml` @ `2026-10-18.1`；本文件对应 V5.0 / V5.1 / V5.2，见 [G–V 映射表](../design/PES2TS_G_V_MAPPING_v1_20261018.md)
> 命名约定：模型别名建议用 `B0–B4`（本文件 §3 的 `M0–M4` 仅指模型别名）；工作包保留 `M0–M9`，避免撞名
> 与 v1.2 的差异：`d0=0.08 Å` 覆盖 v1.2 的 `0.06 Å`；`λ_cos=0.02` 为 V5.0–V5.2 专项值（v1.2 的 0.10–0.15 不适用于本冲刺）；`cap=0.20 Å`，legacy `0.12 Å` 仅作诊断
> 取代关系：作为 V5–V7 v1.2 的专项终审与第一阶段实施提案，不覆盖原计划或历史报告；冲突裁决列于 §0
> 上游依赖：[V5–V7 v1.2](PES2TS_PES-N_V5-V7架构调整与实施计划_v1.2_20261018.md)、用户本轮七模块要求、[开发宪法](../PES2TS_开发宪法_v1.md)、[阶段定义](PES2TS_阶段定义_G0-G3.md)
> 下游消费者：数据构建、模型、训练与隔离评测工程；V6 OOF 消费者
> 当前版本：v1.0 / 2026-10-18
> 证据边界：本文数值是建议预注册配置，不是实测收益或已验证物理阈值。历史 50 簇、100 分支、5,568 动作及 342 冲突对来自给定背景和现有计划，本轮未重新扫描数据确认。

## 0. 终审裁决与启动条件

**结论：可以推进 V5 离线研究，但 v1.2 尚不能直接冻结为执行合同。** 本轮只交付可实施提案；V5.0=数据/数学资格，V5.1=架构/消融，V5.2=冻结配方/OOF/封存评测，均属 G3.P，不新增工程阶段。

| 风险 | 终审裁决 | 本轮采用 |
|---|---|---|
| “标签用下一帧配准就是泄漏” | 不成立。监督标签本来依赖未来；泄漏发生在未来影响输入、选样来源未披露或评测拟合 | 当前输入只因果处理；未来仅参与标签配准，保存旋转与权重 |
| “链式 Kabsch 后固定分支帧” | 共同坐标表示不等于未进行逐帧旋转；一个固定旋转也不能消除随时间变化的刚体漂移 | 显式区分原始帧、因果当前帧、标签对齐帧；不复用身份不明的旧缓存 |
| 非关键重原子不足/柔性转子 | 原子数不保证秩，远端柔性原子可支配配准；“全部重原子但 key=0”并未增加参考信息 | 参考权重固定在起点；无可靠刚性参考时隔离，不以任意全局轴补架 |
| 平面中心用叉积补极向量 | 叉积是轴向量；直接作为位移破坏反射语义。PaiNN 在完全对称输入上也不能创造缺失的对称破缺信息 | 保持 O(3) 极向量合同；不可辨识样本隔离/声明覆盖，禁止凭空补法向 |
| 剪短教师后仍称固定 d0 | 线性缩放跨多个 IRC 线段可能离开原路径 | 沿路径寻找 cap 首次触点，记录实际步距与 terminal/cap mask |
| follower 偷改执行对象 | 改动非关键 H 即违反 strict-copy | 主评测关闭；独立反事实静态诊断，不进入 Gate 主分母 |
| Fold-0 选型后称完整 CV 无偏 | Fold-0 已暴露；选型也会影响其他 fold 训练流程 | OOF 仅保证样本未参与对应权重拟合；CV 标记 selection-exposed，独立 confirm 判定泛化 |
| 外折 early stopping | 用外折标签选 checkpoint 污染供 V6 的 OOF | 正式外折固定 epoch；选 epoch 仅用外折训练部分的内部分组切分 |
| 600 分钟强保证 | 未知分子规模、GPU/驱动、数据可用性时不成立 | 600 分钟硬截止、结果可完整或 censored；30 run 完成另设可行性门 |
| Oracle 在 90 分钟证明真实 TS | 静态 oracle 不能证明 relax/OptTS 可达；DFT→xTB 有跨 PES 偏差 | 本轮静态 teacher ceiling；真实 oracle rollout 是 V7 前置独立 ACP 验证，不挤占并假称完成 |

适用范围：输入完整原子映射、端点拓扑和部署可得起点 X0；固定电子态、非外场环境。分支 b 表示“哪一个端点作为起点”，不能由 IRC 文件名/未来 TS 推断。交换 R/P 名称时同时交换 b、端点图及 ΔB，物理预测保持不变。X0 有助于辨识，但不保证 Markov 性或路径唯一。

启动前具备：受审计真值访问器、可运行 CUDA 训练环境、下面设计的脚本与 fixture、原始 IRC/map/单位/端点来源、冻结 split manifest、独立 confirm/OOD 候选。开发这些能力的时间不包含在训练的 600 分钟内。缺少实现时返回 `not_ready`，不能把文档中的命令当作已存在工具。

本提案涉及新动作/特征/OOF 语义；实现前将下述草案通过新 ADR、接口登记与宪法适用登记正式版本化。此处不改现行生产默认、guard 或已冻结数据。原 v1.1 全文未随本轮附件提供，审查范围为附件共识和仓库 v1.2，不声称完成逐条原文核对。

## 1. 数据卫生、教师与切分合同

### 1.1 构建顺序与平缓尾巴

离线构建顺序：先按身份冻结 split → 核验 units/map/来源 → 起点因果对称匹配/配准 → 标签配准 → 尾巴标记 → 路径重采样 → 新标签冲突检索 → 按簇导出。原始分支保留，所有删除写 reason 和原样本 ID。能量/梯度若用于标签审计，绝不写入 V5 feature 文件。

设从起点到 TS 的原生帧为 X[j]；关键键距 q_l[j]=||x_jl−x_il||。用原生帧自身相邻的关键原子路径长 ΔsA[j]=RMS_A(X[j+1]−X[j])，中心差分（边界单边）定义：

    vq[j] = RMS_l(q_l[j+1]−q_l[j−1]) / max(sA[j+1]−sA[j−1], 1e−6 Å)
    f[j] = vq[j] / max(quantile_0.9(vq over this branch), 1e−3)

仅从反应物/产物**起点端**向内识别连续 f<0.05 的 ≥5 个原生间隔，且该前缀 `max_l range(q_l)<0.02 Å`，标为 `flat_tail_candidate`。至首次连续 3 点 f≥0.05 为止；最多移除分支关键路径长度的 20%。TS 端不依据低梯度裁剪：TS 本来 g≈0。若分母过小、整支无显著键距变化，或非尾巴区还有连续低 f，标 `pair_metric_uninformative`，保留并独立切片，不武断判为尾巴（弯曲/扭转反应可少变键距）。

默认尾巴只降低训练采样权重至 0.25，不抹掉部署起点；Tier 2 比较 include(weight=1) 与 exclude(weight=0)，主报告始终含未经尾裁的合法完整路径及 tail-only。标签定义的尾裁只在训练侧用于实验，不能用于筛掉测试难例。X0 永远是原始部署起点，不改为裁剪后的第一帧。若尾裁超过 20% 则整支待审，不自动扩阈值。

### 1.2 局部自同构与 342 对冲突重审

群 G 是**同时**保持 G_R/G_P、元素/同位素/形式电荷、端点立体约束、K 的编辑类型与任务身份的图自同构；不是按元素任意排列。可对 CH3/CF3 末端等价原子枚举 S3，对 Ph 用保留取代位点和立体语义的环自同构；旋转单键不是图置换，不能被当作“无代价等价”。

1. 在每个局部等价分量枚举允许置换，联合搜索上限 128；超限返回 `mapping_group_overflow`，不取随机前 128 个。
2. 起点由端点映射固定身份；第 j 帧只对 j−1 的已处理前缀求 `argmin_(π∈G,R∈SO3,t) RMS_W(π(X[j])R+t−X[j−1])`。
3. 若多个极小值 RMS 相差 ≤1e−4 Å，优先延续上一置换；仍不能区分，保留所有等价轨道/标 ambiguity，不能用下一帧方向挑当前帧排列。
4. 将 π 同步作用于所有原子属性、端点图、K、A、标签与 map sidecar；预测导出通过 π逆恢复任务原子身份。不得逐帧只换 H 坐标。
5. 跨样本比较时，在同一合法群下对几何和向量应用同一置换及 proper rotation；不允许反射配准“消除”真实对映体差异。

重算原 342 对，并在同 split、同任务语义的全部样本中补做近邻搜索。候选条件：完整分子最小 RMSD<0.03 Å，关键 RMSD<0.02 Å；最终条件再要求 X0 对齐 RMSD<0.03 Å、相同 b/ρ/图编辑。定义向量展平余弦 C；只有两标签 RMS>0.02 Å 且 C<0 才算**输入不可辨识冲突**。几何相似但 X0/b 不同的对标 `context_resolved`，不能为了“清零”删除。

以冲突边构连通分量；无法依预先质量规则确定坏映射的一整分量训练权重=0，保留评测/拒答覆盖记录，不挑更好拟合的一端。C∈[0,0.25) 的近冲突权重=0.25。阈值固定后可应用于测试诊断，但不据此清洗测试主分母。训练可用集残余冲突率=仍有冲突邻居的样本数/可用样本数，目标 0，>1% 熔断；隔离率>5% 或任一簇损失>20% 则停训审计。此处“零冲突”仅限设定距离/群/检索覆盖，并非证明函数唯一。

### 1.3 加权 Kabsch 与关键原子教师

参考集合 S 由**起点拓扑**确定：优先 A 周围 1–2 共价 hop 内的非关键重原子、同一刚性环/无可旋转单键连通块。候选块按距 A、尺寸选择，等价并列不可用 atom row ID 破坏对称；并列分别求解，结果不一致则待审。不以整条轨迹方差选择参考，因为这会把未来用于输入预处理。

    w_i = [i∈S] / (1+hop(i,A))² ; W=diag(w/sum(w))

质心 μY、μX 按 W；行向量约定 C=(Y−μY)^T W (X−μX)=UΣV^T：

    R = U diag(1,1,det(UV^T)) V^T
    align(Y→X) = (Y−μY)R + μX

参考散布最大特征值>0.01 Å²、第二/第一>1e−3 才可使用；非共线平面点足够确定 proper rotation，**不要求第三特征值非零**。参考块配准残差>0.03 Å、相邻旋转>15° 或不同合格块产生标签 RMS 差>0.02 Å，标 gauge-sensitive 待审。只有秩不足且原始计算器提供经验证统一 Cartesian frame 时才可用 `native_verified_v1` 独立分层；否则 `gauge_unresolved`，不退回全局轴或未经验证全原子配准。覆盖不足导致数据门失败，而不是伪造可辨识性。

当前 X_t 的预处理只用 X0/过去；对未来 Y 的配准只在标签侧做。每个起始 t，把后续原生帧 Y[j] 分别按同一个固定 S/W 对齐到当前 X_t，然后在对齐后的相邻节点间**分段线性插值**。不插旋转矩阵、不用 Cartesian cubic spline 过冲，也不用 SLERP 插原子坐标。

在每段 Y(α)=Y_j+α(Y_j+1−Y_j) 中，从 α∈[0,1] 找满足 `RMS_A(Y(α)−X_t)=d0` 的最早正根（平方后是二次方程，float64，根容差 1e−8）；沿路径顺序找首次触点，不假定 chord distance 单调。若任何关键原子范数先达 cap=0.20 Å，取该**路径上的**首次触点，记录实际 d_eff；若先到 TS，取剩余尾步。无未来非零步则没有 MOVE 标签，不制造 STOP。

主 d0=0.08 Å；敏感性 0.05/0.12。此值按用户本轮要求替代 v1.2 的建议 0.06；cap=0.20 是为三档实验统一比较的离线候选包络，**不是**改变现行执行上限。G5.4 同时报 cap=0.12 的旧安全包络诊断，V7 最终上限另定。标签 U*=Y(α)_A−X_t,A；有效步距、终端、cap、W/hash/R、native-frame IDs 写标签表。不得将 d_eff、terminal、采样索引或到 TS 剩余长度输入 V5。

### 1.4 三层 split 与 OOF

身份键 `reaction_group_hash=SHA256(canonical unordered endpoint graph pair + isotope/stereo + mapped edit equivalence class + charge/spin identity)`；反向、重复来源、映射/对称副本用 union-find 合并。源 ID/方法/trajectory hash 另存，不以方法差异把同反应拆成不同 split。保存规范化算法版本与完整对应关系，hash 不能代替去重算法。

* fit50：5-fold、每折目标 10 簇，两侧全部帧同折；按 family/尺寸分层后用 seed=20261018 冻结，组大小不等时允许折数不等。开发验证只从 fit 内产生。
* Confirm：从扩展池选至少 30 个未开发暴露身份簇，family 分布尽量匹配 fit；独立去重，不能用 monitor8 充数。
* OOD：先按元数据冻结至少 3 个完全未进入 fit 的 family，每族目标≥10 簇；若目标族已在 fit50，则整族从 fit 移出，并重新申报实际训练簇数/预算，不能仍写 fit50。
* 不足上述规模时 `insufficient_evidence`，可交付探索性结果，不发泛化晋级。monitor8 只 smoke；任何既往看过的 confirm 身份转开发暴露，不“重新封存”。

冻结前断言 group 集两两无交集，所有反向/partner/映射重复连通分量同 split，同 group 的 fold 唯一；记录 manifest SHA256 与历史开发排除名单 SHA256。只依元数据切分，然后独立评测器读取封存标签。

`move_proposal_v5_draft1.parquet` 每行一个 (sample,model,seed)，包含 group/family/branch/sample ID、fold、model/config/checkpoint/split/training_group hashes、atom_map_hash、input_geometry_hash、X0_hash、frame_contract、d0、key_atom_ids:list<int64>、u_raw:list<fixed3 float32>、u_bounded、z:list<float32>[64]、eligible/reason。teacher/future/TS 存另一隔离表，默认 OOF 消费器不可读。每个模型种子每个合格样本恰有一行，缺行明确 censored。

正式 OOF 外折完全不参与 checkpoint/标准化/阈值拟合。内折可用外折训练的 40 簇做 32/8 切分确定 epoch，再用40簇固定 epoch 重训；若时间不足，使用 Tier 1–2 冻结的统一40 epoch，标 selection-exposed（不是严格嵌套 CV）。三个种子保留独立预测及均值，模型选择过程写 `selection_scope_hash`。供 V6 时按 reaction 分组；不同 fold 的 z 隐空间不能直接视作同一坐标：主接口仅用 U、固定几何摘要和不变量，z 仅存档，待训练内拟合跨编码器对齐/统一冻结 encoder 后再开放。OOF 并不自动消除 V6 的评测泄漏或部署分布漂移。

## 2. PyG / TensorDict 输入输出

长度单位 Å；浮点默认 float32，配准 float64；atom IDs 和索引 int64。以下是明确的建议 schema，不读取 xTB E/g。

| 字段 | Shape | 内容 |
|---|---|---|
| pos, pos0 | [N,3] | 当前及因果对齐起点；同一坐标系，中心化，原值独立保留用于 patch |
| h_node | [N,36] | 下表标量，绝不把 xyz 当标量特征 |
| edge_index / edge_attr | [2,E] / [E,29] | 双向有向边；同一无向对仅出现一次再双向展开 |
| task | [B,19] | 分支2维；方法 embedding8；溶剂 embedding4；charge/4、(multiplicity−1)/4、d0/0.08、ρ-known、solvent-known 各1 |
| key_index / key_ptr | [Σm] / [B+1] | batch 全局关键节点索引 / 分子边界 |
| batch / ptr | [N] / [B+1] | PyG 分子归属 |
| pair_index / pair_edit | [2,Σk] / [Σk,1] | 派生损失索引、起点→另一端的有符号键级变化 |
| output.u / output.z | [Σm,3] / [B,64] | 仅关键原子向量；图/关键 pooling 后64维标量摘要 |

节点36维：元素 embedding16（Z=1…118，未知拒绝）；起点/另一端形式电荷各1除4；两端共价度各1除6；两端芳香标志各1；is_key 1；关键对度 min(degK,4)/4 1；距 A 的 hop one-hot 0/1/2/3/>3 共5；||x_i−c_A||/5、||x0_i−c0_A||/5、||(x_i−c_A)−(x0_i−c0_A)||/1 共3；同一分支当前/起点径向向量夹角1（零范数置0并用下一维标缺失）；anchor_valid 1；is_H 1；同位素质量差/4 1。计数16+2+2+2+1+1+5+3+1+1+1+1=36。连续标量除上述固定尺度后截到[-5,5]；不对向量坐标分量独立 clip。X0 全三维记忆还通过下述几何向量进入头，不仅保留范数。

图是 G_R∪G_P 共价边、K、当前 r<4.5 Å 空间图的并集；保留所有共价/K 边，即使超半径。空间边目标上限32邻居，边界并列距离1e−6 Å 内全留，不能按 atom ID 截断导致置换破坏；最坏边数纳入测速。cutoff 余弦包络，强制拓扑边不套空间截断。起点图/两端键级来自可部署任务，不从未来帧重新感知。

边29维：当前距离16个 Gaussian RBF，中心均匀[0,4.5] Å、σ=0.3 Å；r0/4.5 1；clip((r−r0)/1,−3,3) 1；B_start/3、B_other/3、ΔB/3 共3；covalent_start/covalent_other/is_K/is_radius 共4；两端芳香2；Pauling proxy及有效mask 2。总16+1+1+3+4+2+2=29。proxy `exp(clip((r_eq−r)/0.30,−5,ln3))/3`，r_eq 用冻结元素对单键参考表；缺表置0/mask0。它是几何描述符，不是量化计算键级或拓扑真值，消融可关闭。

向量基包括当前 r_ij、对齐起点 r0_ij、a_i=(x_i−c_A)−(x0_i−c0_A)，除固定1 Å尺度；它们作为极向量，不拼进标量 MLP。任务方法编码仅 train 建词表，未知映射 UNK，拟合和 missing mask 分开。

关键原子集合按 `ReactionTask.key_atom_ids` 的唯一映射 ID 顺序，lookup 到当前节点行，batch 时加 ptr；网络只对 key_index gather 后 readout，不生成非关键位移张量。patch 使用原几何 clone+index_add（唯一索引），非关键逐位一致。内部 hidden vector 的存在不是物理坐标更新；禁止把 latent EGNN 坐标写回 XYZ。

## 3. 五模型与架构护栏

本节 M0–M4 仅用户指定的模型别名，非项目科学里程碑。共享相同样本、特征来源、任务语义、输出顺序与 cap。参数量为预算，工程用 `sum(p.numel())` 实测落盘并断言上限，不虚报精确计数。

### M0：Classic Key-Atom Pair-Driver（0参数）

令 e_l=(x_j−x_i)/d_ij，s_l=sign(B_other−B_start)，b>起点方向已反映在 ΔB。每个 pair 累加 `v_i += s_l e_l`、`v_j -= s_l e_l`（成键相向，断键相背）；各原子除 max(1,degK_i)。令 U=d0·v/max(RMS_A(v),1e−8)，再全体同因子缩至 max_atom≤0.20 Å。若 v 范数<1e−8 则输出零和 typed degeneracy。它是几何解析基线，不能保证每条实际 IRC 的局部键距都单调；保留局部教师符号与任务编辑符号两种统计。不得以 key pair 两端分别覆盖同一原子的更新。

### M1：Local-Frame MLP（≤50k）

共享节点/边 invariant encoder：node36→32；edge29→16；邻居消息(32+16+distance)→32→32 mean pooling；与node32、task19、局部anchor3拼成86维，86→64→32→3，SiLU，标量 dropout0.05。输出三个局部分量再乘 F^T；z 由图/关键平均池化投影64。

对每个关键原子构建候选有序邻居对 (j,k)：v=x_j−x_i，w=x_k−x_i；||v||>1e−4 Å 且 ||w−(w·e1)e1||>1e−3 Å 且 sin(angle)>0.1：e1=v/||v||，e2=normalize(w−(w·e1)e1)，e3=e1×e2，F=[e1,e2,e3]∈SO3。优先 K/共价邻居，再空间/起点向量。对全部符合优先级的并列候选架做共享 MLP 并平均全局输出；不按节点序号挑一架。超过64个候选或不存在合格架，回退到共享 invariant scalar coefficient×r_ij 的 signed 向量和头；无非零向量返回零+degenerate flag。

手性修复：普通 SO3 框架 MLP 不自动 O3 等变。用固定反射 Q=diag(−1,1,1) 做 `f_O3(X,X0)=0.5[f_SO3(X,X0)+f_SO3(XQ,X0Q)Q^T]`，同时镜像立体几何约束；对向量坐标全变换。这个群平均依赖基础 f_SO3 真正旋转等变，不能救任意全局 MLP。训练与推理都做，不只数据增强。若测试失败淘汰 M1，不能以“只是 baseline”豁免。

### M2：2-hop / Star EGNN（≤90k）

输入仍完整几何；仅消息计算子图为 A 的2共价hop并上距 A<4.5 Å的节点，保留所选节点间共价/K和以 A 为中心的星形空间边；远端贡献限固定 invariant pooling，明确这是上下文截断消融。hidden48，3层，message(48+48+29)→48→48，node residual(48+48+task19)→48→48，LayerNorm仅标量。固定物理 pos 不修改。

    δv_i = a_ref / max(1,deg(i)) · Σ_j tanh(MLP_v(m_ij)) · r_ij/(||r_ij||+0.1 Å)

每层向量系数保留正负，不能 sigmoid/softplus-only；3层系数合成并加入 anchor 极向量 signed 通道，最终仅 A 输出。a_ref=0.08 Å，输出不强制归一化成d0。图截断漏掉的非局部信息由 M3 对照测量。

### M3：全分子 Sparse-Head EGNN（≤110k）

hidden48、3层、相同标量消息网络遍历全图；向量 head 仅 A 的 incident edges与anchor，节点不做真实坐标更新。预算内共享层结构，不增加“看似稀疏、实际全原子3维监督”。沿用 signed tanh 和 mean normalization；不把非关键运动塞进 auxiliary loss。

### M4：Edge-Gated Sparse PaiNN（≤120k）

scalar/vector channels32、interaction blocks3；vector state [N,32,3]，点积/范数产生 scalar，scalar线性混合vector；不能对每个 xyz 分量做 LayerNorm/ReLU/dropout。距离过滤29→32→96；edge gate可sigmoid控制权重，但被门控的向量生成系数用 signed tanh，不能只输出正的径向系数。

每通道向量按范数统一 rescale 到上限2（内部无量纲），不逐分量 clip；记录饱和率，>10%连续3epoch 停止该臂排查尺度。只在 A readout；width 禁止临时升64/128，train–inner-val RMSE差>30%且val连续8epoch恶化停止。此前 PaiNN 负结果不能单靠换头宣称解决，胜负由同一 Group 验证决定。

所有模型报告 rank([当前/起点/anchor向量]) 的0/1/2/3切片。完全共面且起点也共面时，O3极向量模型不能确定唯一面外方向；若教师要求该方向，归不可辨识/需额外可部署立体条件，不用叉积伪造答案。SO3 与 O3 不可混写。

## 4. 损失与超参数

令 u、y 单位Å，e_i=||u_i−y_i||/a_ref；旋转不变的**径向 Huber** hδ(e)=e²/2（e≤δ），否则 δ(e−δ/2)。逐xyz分量 Huber 不具有严格旋转不变性，不采用。w_i=1/max(1,degK_i)，每帧归一至Σw_i=1。

    L_atom = Σ_i w_i h_0.25(||u_i−y_i||/0.08)
    J_l(u) = e_l · (u_j−u_i)
    L_pair = mean_l h_0.25((J_l(u)−J_l(y))/0.08)
    g_i = 1[||y_i||>0.02 Å]
    c_i = (u_i·y_i)/(sqrt(||u_i||²+(1e−4 Å)²) sqrt(||y_i||²+(1e−4 Å)²))
    L_cos = Σ_i w_i g_i(1−c_i) / max(Σ_i w_i g_i, 1e−8)
    L5 = (1/R) Σ_r (1/B_r) Σ_b [Σ_t q_t(L_atom+0.5 L_pair+0.02 L_cos)/Σ_t q_t]

q_t=质量权重×尾巴权重；无合格样本分支不进入均值，缺失率另报。cos只用教师模长门，不依预测模长gate防止模型通过收缩逃罚；soft norm限制零附近梯度。λcos前5epoch=0，随后5epoch线性升到0.02，禁止升到V4 H2大权重。pair损失拟合教师**一阶**变化，不把精确有限步距离与线性式混作同一标签；评测另外算精确距离变化。

反应均衡 sampler：每个batch采8个不同反应簇，每簇4帧，两分支尽量各2；先簇均匀、分支均匀、帧按q采样。已通过采样实现权重时 loss 不再乘 1/n_r 或 q，避免双重降权；上式是目标分布的定义。epoch=ceil(N_train/32)个batch，有放回，记录实际曝光量。

```yaml
schema: pesn_v5_sprint_draft1
d0_angstrom: 0.08
d0_ablation: [0.05, 0.08, 0.12]
a_ref_angstrom: 0.08
teacher_atom_cap_angstrom: 0.20
legacy_cap_diagnostic_angstrom: 0.12
huber_delta_normalized: 0.25  # 对应误差0.02 Å
lambda_pair: 0.5
lambda_cos: 0.02
cos_start_epoch: 5
cos_ramp_epochs: 5
tau_norm_angstrom: 0.02
cos_soft_norm_angstrom: 0.0001
optimizer: AdamW
lr: 0.001
weight_decay: 0.0001
warmup_epochs: 5
scheduler: cosine
min_lr: 0.0001
batch_size: 32
reactions_per_batch: 8
frames_per_reaction: 4
grad_clip_norm: 1.0
precision: fp32
max_epochs_screen: 60
max_epochs_ablation: 40
formal_epochs: 40
early_stop_metric: inner_val_reaction_median_key_rmse_over_zero
early_stop_patience: 8
early_stop_min_delta: 0.005
seeds: [17, 29, 43]
folds: 5
radius_angstrom: 4.5
spatial_neighbors_target: 32
wall_minutes: 600
```

训练raw输出保留；u_bounded以统一因子限制单原子≤0.20 Å和RMS≤d0，所有指标raw/bounded双报，主Gate用raw精度与bounded可执行性；不能靠仅展示截断版掩盖发散。

## 5. 600分钟执行预算与熔断

**所有命令是待实现的 CLI 设计，当前不声称可直接执行。** 入口建议统一 `python scripts/campaigns/run_v5_sprint.py --config config/campaigns/v5_sprint.yaml --stage ...`；该脚本仅编排离线工作，不启动QC。业务归 pesn/offline，runtime特征/模型不导入offline。

| Tier | 窗口/分钟 | stage与工作 | 熔断/降级 |
|---|---|---|---|
| 0 | 00:00–01:30 /90 | `--stage prepare-audit`：0–15来源/split；15–45配准/对称/尾标；45–65重采样/冲突；65–80不变量与oracle/classic；80–90缓存与GPU测速 | 残余冲突>1%、隔离>5%、来源泄漏或G5.1失败：停止GPU计划，输出失败账本；不消耗余时空训 |
| 1 | 01:30–03:30 /120 | `--stage screen --fold 0 --seed 17`：M1–M4顺序，每臂≤25分钟/60epoch，余20分钟比较与检查 | 20epoch后内验证RMSE≥Classic且最近5epoch改善<2%：停该臂；全部失败则保留Classic研究结论、不进入正式30run |
| 2 | 03:30–06:00 /150 | `--stage ablate --fold 0 --seed 17`：最好架构固定40epoch，6臂×≤20分钟，30分钟复核/冻结 | 总耗时预计超窗只做冻结优先级前缀，记录 skipped；某臂NaN直接停止，不能临时换学习率后继续同run |
| 3 | 06:00–08:30 /150 | `--stage crossfit --top 2 --folds 5 --seeds 17,29,43`：30 run，每run固定40epoch，128分钟训练+12分钟OOF+10分钟缓冲 | 用实测P90估算超预算则预注册降级：Top1×5×3；仍不足则Top1×5×1；不伪称完成Top2三seed，不能通过挑好折补齐 |
| 4 | 08:30–10:00 /90 | `--stage sealed-report`：0–15冻结检查/OOF；15–45 confirm/OOD；45–60 strict-copy/follower静态诊断；60–80组级bootstrap/Gate；80–90原子写入报告/成本 | 缺封存或coverage不足=insufficient；任Gate失败=reject；T+590不启动新工作，T+600取消并保存censored报告 |

Tier 2 六臂：共同基准（d0=.08，pair=.5，anchor on，tail downweight）；pair=0；d0=.05；d0=.12；tail include；tail exclude。另 `anchor off` 是必做第七比较，可复用Tier1的相同配方基准节省一臂：实际新增6训练=pair0、.05、.12、include、exclude、anchor-off。若Tier1基准配方不完全相同，则基准重跑占一个槽，最后优先级为 anchor-off、pair0、tail两臂、.05、.12；删减明确标partial。anchor-off同时关闭pos0、r0、Δd、节点anchor摘要与向量，不只清一个标志。各d0测试在相同原生起始帧、同fold/曝光预算对照，终端短步单列；不得用重采样后不同帧数的原始MSE直接排名。

吞吐估算：约5,568动作、外折训练80%=4,454，batch32→140步/epoch。假定N中位≤60、P95≤120、空间度约32且图缓存CPU/GPU可用：M1预计0.5–1.5s/epoch、M2 1.5–4s、M3 2.5–6s、M4 3–8s，**均为待测预算假设**。每epoch含验证/I/O需另计，不能拿纯forward速度代替。Tier0做30个warmup+200个完整优化step、3次验证，记录P50/P90 step和加载时间。

    t_run = 40*(140*t_step_P90 + t_epoch_overhead) + t_start_save
    t_tier3 = 15*t_run_top1 + 15*t_run_top2 + 22*60 seconds

若每epoch 4s，start/save15s，30run≈87.5分钟+22=109.5分钟；若6s则127.5+22=149.5分钟，已无余量；8s则167.5+22=189.5分钟，**不可完成150分钟目标**。准入建议预测≤135分钟（留15分钟风险），否则按上表降级。GPU仅一个训练进程，CPU最多8 workers预处理，缓存上限RAM50%，显存80%；禁止因OOM静默改变模型/batch但复用同run ID。

锁定配方后，最终封存预测使用选定架构的5fold×3seed模型按同seed跨fold平均再跨seed平均；报告ensemble15成员与单成员差异，不冒充单模型推理成本。V6 OOF每反应只有该外折3seed成员可用，区别于封存15成员ensemble；该差异单列分布审计。没有时间重训全fit单模型就不发布该模型。

状态输出 `complete_full_matrix / complete_reduced_matrix / censored_budget / rejected_data / rejected_gate / insufficient_evidence`，每run均有耗时、epoch、样本/边曝光、跳过原因；分钟上限可保证结束，不能保证正结果或矩阵完整。

## 6. 四级硬门与晋级定义

全部阈值在封存前冻结；下列为建议工程门，不是文献常数。聚合次序：帧内关键原子→分支平均→反应两侧等权→反应簇中位/IQR；同时报告总数、拒答、尾巴和rank切片。bootstrap以簇为单位2000次，seed=20261018；不按高度相关帧计算置信区间。缺样本/未执行不可判pass。

### G5.1 数学门（训练前fixtures与导出模型复测，100%通过）

每种架构至少100个包含共线/共面/对称/多组分的fixture，每例20个随机proper旋转+平移（±10 Å）+10个atom permutations。float64配准/几何核、float32模型下，`||f(XQ+t)−f(X)Q||F/max(||f(X)||F,0.08√m)<1e−5`，绝对最大误差<1e−6 Å；graph cutoff/tie构造也包含在测试中。训练dropout关闭。

反射detQ=−1时同样检验极向量变换；立体有向体积χ=det[r1,r2,r3]在非退化|χ|>1e−5 Å³样本上变号，镜像输入的预测seed也遵守对应关系。不能要求原分子每个反应中心χ永远不变（真实反应可能反转/过平面）。节点输出是**置换等变**，图z才是置换不变。branch名称交换也测试。strict-copy patch后非关键坐标 `torch.equal`，输入对象本身未修改。

### G5.2 关键向量门

每簇 `MAE=mean||u−y||`、`RMSE=sqrt(mean||u−y||²)`，分别对零动作和Classic求比；分母<1e−6 Å标undefined，单列而不当0。门：中位 MAE、RMSE/zero均≤0.80，对Classic均≤0.90；RMSE/zero 的Q75≤1.00，≥70%簇优于零；相对Classic的簇中位改善bootstrap95%下界>0。有效原子||y||>0.02 Å的cos中位全场≥0.60、nearTS≥0.50。nearTS=剩余**关键路径弧长**/总长≤0.20，仅隔离评测器使用；不是输入特征。至少20个有效反应簇且nearTS至少10簇，否则insufficient。

三seed分别报告，至少2/3独立seed达到精度阈值，另ensemble达门；不只报最好seed。Fold0和选择暴露的CV只开发证据，Confirm为主确认；OOD逐族指标及coverage单列，不能以Confirm优秀抵消OOD失败。

### G5.3 派生化学门

精确 `Δd_l(u)=||r_l+u_j−u_i||−||r_l||`，仅|Δd_l(y)|≥0.005 Å的pair计符号；预测|Δd_l(u)|<0.005视为不正确/弃权并计主分母。教师符号准确率反应中位≥0.90、Q25≥0.80；对任务成断方向另报，教师本身不符合任务单调性不自动判脏数据。

多键≥2且有效总变化≥0.01 Å时 `p_l=|Δd_l|/Σ|Δd|`，同步误差 SRE=0.5Σ|p_hat−p_teacher|，完全预测零则SRE=1。反应中位≤0.15、Q75≤0.25；另报一阶J与精确Δd偏差，防止短键/大步非线性被损失掩盖。

### G5.4 静态可执行门

直接patch，仅修改A，不运行xTB。碰撞=任意异原子对距离<0.75(r_cov_i+r_cov_j)，r_cov表版本固定、缺元素拒答；此检查包含成键对，合法短键例外只能预注册，不能看test后豁免。非反应共价保护边为 `(G_start bonds)\K`，包括关键–非关键的X–H；若只查两端都非关键的键，strict-copy下永远不变，会制造虚假安全。

保护边长度>1.35 d_eq 判拉伸；d_eq取部署起点合格键长并保存，原起点异常另报，不能从未来最低能帧选。补充过短<0.70 d_eq。报告seed总违规率和相对输入**新增**违规率，所有输入违规仍留分母；各簇先算违规帧比例再聚合。门：新增碰撞率≤0.5%，新增保护边异常≤1%，各自单侧95%簇bootstrap上界≤2%，无任一簇>5%；拒答/未知≤5%。Oracle teacher在同一strict-copy协议也通过这些阈值，才允许声称动作表示具有静态可行性；Oracle不通过先改步长/合同，不能归咎网络。

follower反事实仅平移“与唯一关键重原子成键且自身不在A”的末端H，排除桥H/转移H/有歧义连接；同时检查新增H碰撞，报告与strict-copy配对差。该分支标 `contract_counterfactual`，不可计入G5.4 pass，不能默认增加冻结集。

G5.1–4在Confirm通过且OOD至少3族、每族≥10簇并分别满足同类门，才发 `eligible_for_V6_offline_research`；小样本置信区间不足则继续研究、不降低阈值。G5.4不证明SCF稳定、松弛可达或真实TS成功。V7前仍需ACP `$fix`真实回归、同PES Oracle rollout及完整OptTS/Freq/双向IRC证据。10小时结果不发生产晋级。

OOF一致性附表：原生帧/OOF/封存预测的范数、cap率、零输出、cos、输入支持域按fold/seed/family报告；不同fold输出范数中位比>2或零输出率差>10百分点触发排查，不强行把不同family分布归一到一致。全fit部署模型、预弛豫帧上的OOF需另构建；不能把DFT静态OOF称作V6在线分布已对齐。

## 7. Top-5 致命实现错误与断言

以下为实现者应加入的伪代码断言；变量代表上述schema，不是本轮已执行测试。

1. **跨batch索引/重复关键原子与覆盖写入。**
   ```python
   assert key_ids.numel() == torch.unique(key_ids).numel()  # 每分子
   assert torch.equal(batch[key_index], key_batch)
   assert torch.equal(map_ids[key_index], task_key_ids_flat)
   assert torch.equal(seed[~key_mask], original[~key_mask])
   ```
2. **Kabsch左右乘错、允许反射或误拒平面参考。**
   ```python
   assert torch.allclose(R.T @ R, eye3, atol=1e-10)
   assert abs(torch.det(R).item() - 1) < 1e-10
   assert eigvals[-2] / eigvals[-1] > 1e-3  # 升序；不是最小值
   assert rms(align(rotated_fixture), original_fixture) < 1e-8
   ```
3. **未来字段/折外归一化/外折early stopping混进训练。**
   ```python
   assert set(features) <= FEATURE_ALLOWLIST
   assert train_groups.isdisjoint(outer_groups)
   assert scaler_fit_groups <= train_groups
   assert checkpoint_selection_groups.isdisjoint(outer_groups)
   assert anchor_source_frame <= current_source_frame
   ```
   另用“替换未来帧后当前输入hash不变”fixture，字段白名单本身无法发现间接泄漏。
4. **图截断或对称匹配破坏原子置换/反射；把轴向量当位移。**
   ```python
   assert edges_unique_and_bidirectional(edge_index)
   assert all_required_covalent_and_K_edges_present()
   assert equivariance_error(model, permutation_fixture) < 1e-5
   assert equivariance_error(model, reflection_fixture) < 1e-5
   ```
   测试覆盖距离并列、所有frame fallback和半径边界，不能只测随机非退化几何。
5. **空掩码/零余弦NaN、重复权重、pair方向相反。**
   ```python
   assert torch.isfinite(loss)
   assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
   assert abs(weights.sum().item() - 1) < 1e-6
   assert cosine_loss(zero_teacher, zero_prediction) == 0  # 空gate
   assert finite_difference_pair_derivative_error() < 1e-5
   ```
   finite-difference用步长1e−5 Å核验 e·(u_j−u_i)；zero prediction/nonzero teacher也检查有限梯度。采样器加权后损失不再重复乘样本频次倒数。

## 8. 交付清单与外部依据

工程产物：`split_manifest.json`、`dataset_audit.json`（342对逐项去向+新增冲突）、`frame_contract.json`、`feature_spec.json`、`timing_probe.json`、`run_records.jsonl`、`ablation_table.parquet`、`move_proposal_v5_draft1.parquet`、`sealed_eval.json`、`gate_report.md/json`。每件带schema/config/source/parent hash、四层能力状态；报告区分planned、实测、未执行、删失。没有QC时写 `QC not scheduled`，不把未知历史费用填0。

外部原始资料仅支持方法语义，不支持本方案工程阈值：

* [EGNN 原论文](https://proceedings.mlr.press/v139/satorras21a.html)：旋转、平移、反射与置换等变结构；本文signed head/预算为本项目提案。
* [PaiNN 原论文](https://arxiv.org/abs/2102.03150)：标量与等变向量消息；本文32通道/范数截断是待验证配置。
* [xTB xcontrol 官方文档](https://xtb-docs.readthedocs.io/en/latest/xcontrol.html)：`$fix`的Cartesian精确固定语义与`$constrain`的区别；实际二进制仍由ACP真实冒烟确认。
