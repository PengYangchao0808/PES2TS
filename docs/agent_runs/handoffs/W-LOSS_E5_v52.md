# W-LOSS · E5 — V5.2 Q/T/QT 执行 + G3 证据包（含两次事故的完整披露）

> 状态：**IMPLEMENTED + PARTIALLY EXECUTED（预注册抢救集完成）** · 2026-10-20T20:45Z · 分支 `ops/e5-v52` · lane W-LOSS wave 4
> 授权：`formal_training_authorized=true`（至 2026-10-21T13:45Z）；confirm 恒 false
> **V5.2 账本：32/33 launch attempts**（见 §6 事故披露——15 次烧于 λ 校验 bug，8 个 run 因封存违规作废）；有效完成 **9/27**（QT6 + T3，fold0-1）；Q 臂全损；QT×9 中6 未发（含6 QT-confirm blocked）

---

## 1. 交付物

| 组件 | 文件 | 说明 |
|---|---|---|
| V5.2 训练器 | `pes2ts_core/pesn/offline/training/key_atom_v52_train_v3.py` | `V52TrainerConfig`（arm Q/T/QT；off-term λ≡0 强制，active λ∈(0,0.5]）；E4-一致切分 `v52_curve_split`（**训练=dev folds≠F；fold F=outer-val 仅评估；folds3/4 永不进训练**（v2 修复，带 SPLIT_SEAL 断言））；R3 winner backbone 逐字加载（`load_backbone_from_r3`，strict state_dict）+ 种子化新头；冻结调度 400（encoder requires_grad=False，仅头 lr 3e-4）→ 联合 1500（encoder lr 1e-4）；冻结 stop 类权重（20.31/16.89/19.87，编译常量==G3a 报告双向断言）；λcos=0.02（R0 终值）；指标：q 曲线 MAE/Pearson vs constant-q 基线、stop P/R/AUC（Mask 排除，手写 Mann-Whitney）、Move 退化 vs 同 fold+seed R3 winner 同曲线集（冻结 g52/g53，≤2% guard） |
| λq/λm 探针 | `scripts/campaigns/r3_v52_lambda_probe.py` | yaml 未冻结 λq/λm（E4 交棒 §7.2）→ 用 plan §5.4 同形式校准：B0-F 参考（F2-bidir fold0 逐 seed）+ fold0 V5.2 训练切片 20 真实批 + 冻结 v52 loss 梯度线性隔离（∇L_term=∇total(1)−∇total(0) 逐参数）；目标 2%（§5.4 带下锚，记录性选择）→ **λq=0.42104980687872556、λm=0.0132012267552582（均 calibrated，无 scale_unreachable）**；输出 `v52_runs/v52_lambda_calibration.json`——E5 唯一 λ 来源 |
| 驱动 | `scripts/campaigns/r3_v52_run.py` | run（plan 查行→注册→monitor.attempt_launch(stage g0)→GPU flock→训练→done 带 ckpt sha+metrics 路径于 `v52_runs/<run_id>/`；拒绝 confirm 行/未校准 λ/scale_unreachable）；`block-confirm-update`（QT-confirm×6 重注记 `confirm_awaiting_user_authorization`——dev-positive 已由 G2 给出，余下仅用户授权） |
| G3 证据包 | `scripts/campaigns/r3_g3_evidence.py` | 按 yaml G3 判据聚合：q learnable（vs constant-q）、stop AUC>0.5+coverage、Move 退化≤2%；逐 run 失败清单（永不丢弃）；**作废清单排除**（`v52_runs/invalid_runs.json`）；blocked 臂清单 → `v52_runs/g3_evidence_pack.json` |
| 测试 | `tests/agent_ops/test_gpu_v52_trainer.py` | 10 项（见 §5）；全套 tests/agent_ops **471 passed** |

**lr smoke（yaml lr_note 一次性检查）**：Q fold0 s17 400 upd（冻结相）：total 0.322→0.304（有噪下行）、l_move 0.306→0.285、全程有限无 NaN、val 稳定 0.292 → **按建议初值固定** head 3e-4 / encoder 1e-4。t_run(1900+双模型评估) ≈ **40-45s/run**（GPU 4060 Ti）。

## 2. 预注册抢救分配（在任何新结果观察前写入）

λ 校验 bug 烧掉 15/33 次后（§6），剩余 10 次无法完成 27 run。**预注册**（19:55Z，`invalid_runs.json` 同时落盘）：QT fold0×3 + QT fold1×3 + T fold0×3 = 9 次（1 次余量）；理由：QT 联合臂是 V5.2 本体且是唯一携带 Move 非退化 guard 的臂，QT 同时携带 q 证据；Q 单臂信息量最低（8 个已作废 + 2 未发，Q 臂证据=0 有效）。偏离 plan 序（Q→T→QT）为**显式预注册偏差**，待 supervisor 追认/扩帽。

## 3. 有效 run 日志（9/9 done；train_folds 已验证 [1,2] / [0,2]，无封存泄漏）

| run_id | arm | f | s | sel | q learnable | q corr | stop AUC | move deg% | guard | wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| 9ef757ff0e04a770 | QT | 0 | 17 | 1200 | False | -0.009 | 0.507 | 5.89 | ✗ | 44.6 |
| b8aa710013acac0e | QT | 0 | 29 | 500 | True | 0.016 | 0.467 | 0.33 | ✓ | 42.0 |
| a9d648cca12fb0ef | QT | 0 | 43 | 700 | False | 0.007 | 0.523 | 9.33 | ✗ | 39.0 |
| 8698fb1b748e0cb9 | QT | 1 | 17 | 600 | False | — | 0.531 | 6.42 | ✗ | ~45 |
| ed73178dbc24a9ab | QT | 1 | 29 | 1000 | False | — | 0.563 | 5.12 | ✗ | ~45 |
| b401ad97d61acddd | QT | 1 | 43 | 700 | False | — | 0.564 | 1.62 | ✓ | ~45 |
| d3889b36c0b6da22 | T | 0 | 17 | 700 | False | — | 0.491 | 5.88 | ✗ | ~45 |
| e7ff836da24371ea | T | 0 | 29 | 600 | False | — | 0.467 | 0.32 | ✓ | ~45 |
| 51533d3c4f3bd2dd | T | 0 | 43 | 500 | False | — | 0.511 | 9.22 | ✗ | ~45 |

## 4. G3 证据包小结（`g3_evidence_pack.json`；作废 8 run 已排除并记账）

| arm | n | folds | q 曲线 MAE 中位 | q learnable 率 | stop AUC 中位 | AUC>0.5 率 | move 退化中位 | guard 内率 | 判据 q/stop/move |
|---|---|---|---|---|---|---|---|---|---|
| QT | 6 | 0,1 | 0.481 | 1/6 | **0.527** | 5/6 | **+5.5%** | 2/6 | ✓(弱)/✓(弱)/**✗** |
| T | 3 | 0 | 0.491 | 0/3 | 0.491 | 1/3 | +5.88% | 1/3 | ✗/✗/✗ |

**诚实结论（recorded，不调参）**：当前 λ（2% 梯度份额校准）+ 冻结 400/1500 调度下——
- **q 头基本不可学**：曲线 MAE≈constant-q 基线（~0.48），相关≈0，仅 1/6 run 勉强过基线；
- **stop 头有微弱信号**：QT AUC 中位 0.527（5/6 >chance；0.53-0.56 的三个 fold1 run 是最一致信号），T 单臂 0.491≈chance——联合训练似乎帮助 stop 头；
- **Move 非退化 guard 大多失守**：QT 中位退化 +5.5%（>2% 阈值），6 个仅 2 个在 guard 内；T 臂同型。λq=0.42 近上限，联合梯度扰动 backbone 是最可能机理（假设，未验证）。
- 选择分布 500-1200（比 R3 的 250 模式晚，联合任务仍在学习）。
- 判据臂级旗标：QT q/stop 弱过、move 不过；T 全不过。**G3-V52Supervision 的裁定材料已齐**（supervisor 判：可否以 λ 上限降低/冻结 encoder 全程等修订重跑——那需要新授权与新预算，本 lane 不自作）。

## 5. 测试矩阵（10 新项；全套 tests/agent_ops **471 passed**）

| # | path | 类型 | 结果 |
|---|---|---|---|
| 1 | λ 配置拒收（**烧队列 bug 回归**：off-term λ≠0→LAMBDA_OFF_TERM；active λ=0→LAMBDA_RANGE；>0.5→LAMBDA_RANGE；confirm fold 拒收） | rejection | PASS |
| 2 | E4-一致切分：train folds={0,1,2}\{F}、eval=F、inner-train/val 曲线不相交、确定性、逐 fold 验证、**folds3/4 永不出现**（SPLIT_SEAL 断言） | positive | PASS |
| 3 | 冻结相真实冻结：encoder requires_grad=False 且 backward 后 backbone grads 全 None、头 grads 非 None；联合相解冻 | positive | PASS |
| 4 | backbone 初始化逐字：`torch.equal` 对 R3 ckpt 全部 55441 参数；头参数计数 55441/2113/65 | positive | PASS |
| 5 | 冻结 stop 类权重：fold0=20.3109756097561(164/3331) 与 G3a 报告逐项相等（漂移即拒）；fold4 无冻结权重拒收 | positive+rejection | PASS |
| 6 | 标签映射：stop=-1→Mask 不进分母、y=0；q=None→q_mask False | positive | PASS |
| 7 | AUC 数学：完美/反转/全平局(0.5)/退化(None)；Pearson 常量输入 None | positive+rejection | PASS |
| 8 | fixture mini-run（4 upd，freeze2→joint2）：metrics schema 全键、train_folds=[1,2]、eval_fold=0、schedule 记录、ckpt sha 自洽、日志相位切换（freeze 段 encoder_trainable 全 False、joint 段全 True）、损失有限 | positive | PASS |
| 9 | G3 判据助手：q/stop/move pass 语义 + 聚合失败清单（q_not_learnable/stop_at_or_below_chance/move_guard_breach 逐 run 列名） | positive | PASS |
| 10 | λq/λm 值域与 off-term 对称拒收 | rejection | PASS |

## 6. 事故披露（两次，均本 lane 责任，全部账本可见）

1. **LAMBDA_RANGE 配置校验 bug（15 次 attempt 烧毁）**：V52TrainerConfig 初版要求两 λ 均 ∈(0,0.5]，但单term 臂（Q/T）合法地带 off-term λ=0 → 全部 Q 臂 run 在训练前被拒（含 queue 的 RETRY-ONCE 重烧）。8 个 Q run = 15 attempts（d94af267 retry 未触发即被停）。修复：active/off-term 分开校验（回归测试 #1）。**账本后果：33−15=18 次剩余，27 任务不可达** → 触发 §2 预注册抢救。
2. **封存违规（8 个 done run 作废）**：`v52_curve_split` v1 用 `fold != F` 构造训练池——把 confirm folds3/4 混入训练（违反 plan §4.2 封存）。队列2 的 7 个 Q done + 1 interrupted 全部污染。修复：v2 限 dev folds{0,1,2}+SPLIT_SEAL 断言（测试 #2）；**8 run 声明作废**（`v52_runs/invalid_runs.json`，done 为终态不可撤——作废经证据包排除清单+本披露+supervisor 复核）；队列中止后所有抢救 run 验证 train_folds=[1,2]/[0,2] 无泄漏。**账本后果：done 计 16（9 有效+7 作废）；QT×9 与 Q×2、T×6 未发**。
3. 一次 pkill 自匹配：`pkill -f` 模式命中调用方 shell（模式含于自身 cmdline）→ 自杀；改用 PID 精确 kill。等待循环 pgrep -f 自匹配浪费 ~30 min 墙钟（无 GPU 占用损害）。教训：**队列/等待脚本的进程匹配必须用不含自串的模式或 PID 文件**。

## 7. blocked / 未发清单

| 臂 | runs | 状态 | blocked_with_reason / 备注 |
|---|---|---|---|
| QT-confirm | 6 | planned+blocked | `confirm_awaiting_user_authorization`（本波已按 G2 PASS 重注记） |
| Q | 8 done + d94af267 interrupted + e3fe8a13 planned | **作废/未发** | 封存违规作废（8）；预算不足未发（2） |
| T | 6（folds1-2×3） | planned | 预算不足未发（32/33 已用；余 1 次未用——留给 supervisor 裁定） |
| QT | 6（folds1-2×3） | planned | 预算不足未发 |

**待 supervisor 裁定**：(a) 是否扩 V5.2 帽（作废 8 次 + bug 烧 15 次是否计入裁定）；(b) G3-V52Supervision 基于 QT6+T3 的弱/负结果如何判（λ 修订重跑需新授权；本 lane 不自作调参）；(c) 余 1 次 attempt 的用途。

## 8. 下游依赖

- **G3-V52Supervision（supervisor）**：`v52_runs/g3_evidence_pack.json`（9 有效 run + 判据 + 失败清单 + 作废记账）；本 lane 结论：**当前配置下 V5.2 无增益且伤 Move**（诚实负结果候选；E-NO-GAIN 属 no_auto_fix 类，只能 supervisor 裁定）。
- **V5.3（E6）**：若 G3 不过，v53 rollout 仅上限控制器（yaml hard_stops H20 语义）；rollout loader 的 `load_v52_checkpoint_v3` 已可加载本次 QT checkpoint（schema pesn_key_atom_v52_train_v1）。
- λ 来源冻结：λq=0.42104980687872556 / λm=0.0132012267552582（v52_lambda_calibration.json）。
