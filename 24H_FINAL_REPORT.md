# 24H 最终报告 — PES2TS-30H 战役（R3 → V5.2 → V5.3）

> **POST-HOC 纠偏（2026-10-21）**：V5.2 的「q 不可学习 / stop 随机」是对旧 TS 邻域 Stop 任务的有效负结果；任务定义已由 [ADR-0030](docs/design/decisions/ADR-0030-V5.2全路径导航与端点停止.md) 修订为全路径导航 + 端点停止，后续以 V5.2-R 执行。正文不改写。

> 生成：2026-10-20T21:30Z（战役开始后 7.75h；科学执行已全部完成，进入归档）
> 编制：supervisor（GLM-5.3 orchestrator，OMO-Slim 三 lane 并行架构）
> 授权依据：用户 2026-10-20 24h 自主执行授权；窗口 13:45Z→2026-10-21T13:45Z
> 实际执行：**11 个工作包全部终态，六门全部终态记录**；全程 fail-closed 纪律无越权

---

## 1. 阶段真实完成度（implemented / fixture_passed / smoke_passed / scientifically_validated 四级区分）

| 阶段 | 状态 | 分类 | 说明 |
|---|---|---|---|
| E0 冻结+签认 | ✅ 完成 | scientifically_validated | 4 SHA 核验一致、ORA-3 无阻塞、条件签认生效、protocol v1.0.0 |
| E1 六表数据 | ✅ 完成 | **scientifically_validated** | 真实数据 10/10 门测试、确定性重建×3、四 cohort 全处置、G0 PASS |
| WP-L3 账本强化 | ✅ 完成 | **scientifically_validated** | Oracle 三篡改路径实拒、metric_unseal 真实消费一次（6 收据）、全程生产使用 |
| E2 特征/模型/损失 | ✅ 完成 | fixture_passed + 真实数据 G1 PASS | 136 数学测试；StarEGNNV3 55,441 参数；λ=0 与 B0-F 逐比特一致 |
| E3 R3 训练 | ✅ 完成 | **scientifically_validated (dev folds)** | 61/66 attempts、54 done、G2 PASS；**结论：双向 intent 唯一有效（0.910/6/6 簇）** |
| E4 V5.2 标签/heads | ✅ 完成 | scientifically_validated（标签层） | reverse q=−q 精确、覆盖审计完整、G3a PASS |
| E5 V5.2 训练 | ✅ 完成 | **scientifically_validated NEGATIVE** | 9 有效 runs：q 不可学习、stop 随机、Move +5.5% 破防；G3 FAIL（E-NO-GAIN 诚实保存） |
| E6 V5.3 rollout | ✅ 完成 | engineering-validated only | 216 episodes + 36 次真实 xtb；G4 PASS（工程闭环）；**无任何 TS 科学主张** |
| E7 统计 | ✅ 完成 | scientifically_validated | cluster 级 CI、封存排除、成本账本（工具+终版报告） |
| E8 归档 | ✅ 完成 | — | 七份交付文件 + gate 终态 + 授权撤销 |

**未完成/阻塞（按授权，非缺陷）**：R3 C 臂 6、V5.2 QT-confirm 6（confirm=false）；R1/R2 6（冻结 spec 空缺）；V5.2 未发射 14+1 held（no-gain 后关闭）。

## 2. 训练 runs 与计算成本（全 attempt 计数，含失败）

| 项 | 数值 |
|---|---|
| R3 launch attempts | **61/66**（54 done；1 retry 披露；6 封存终态化经 unseal） |
| V5.2 launch attempts | **32/33**（16 done = 9 有效 + 7 无效自判；1 interrupted；15 次 λ-bug 烧毁 attempts 计入） |
| 总 launch attempts | 93（占 99-run 计划的 93.9%；6 个从未 launch 的为 BLOCKED 臂） |
| GPU 训练时间 | **0.93 GPU-h**（70 runs × ~48s，metrics 实录加总；RTX 4060 Ti 8GB） |
| 物理(xtb) 成本 | 49 s（36 次受约束松弛）+ rollout CPU 8 s |
| 吞吐 | 43.25 updates/s（3000 updates/run）；t_run ≈ 45–55 s |
| 墙钟 | 科学执行 7.75h（13:45Z–21:20Z），远低于 24h 预算 |

## 3. 各模型相对基线的真实指标（dev folds 0–2，cluster 级；sealed folds3/4 未揭示）

| 模型/臂 | RMSE/zero [CI95] | sign med | 优于零步 | 状态 |
|---|---|---|---|---|
| zero-step 基线 | 1.000 | — | — | 参照 |
| classic CV/Jacobian | 记录性对照（rollout 中崩塌 sign 0.24） | 0.10 | 否 | 不可用作对照 |
| F0/F1/F2/F2-intent-off/F3 forward | 1.107–1.225 | 0.31–0.52 | ≈0/6 | 负结果 |
| **F2-bidir（R3 winner, fold0）** | **0.910 [0.884–0.949]** | 0.888 | **6/6** | 三种子一致 |
| B = F2-bidir 跨折 f1–2 | 0.9735 [0.949–0.998] | 0.847 | 10/12 | 泛化衰减 ~6.4pp |
| S2/S10/D2/D10（λ 校准辅助） | 0.908–0.920 | 0.888 | 各 6/6 | 护栏 12/12 绿 |
| V52 progress 头 | MAE 0.507 vs 常数 0.476（corr≈0） | — | 劣于常数 | **不可学习** |
| V52 stop 头 | AUC 0.491–0.527，recall@0.5=0 | — | 随机水平 | 无可用工作点 |
| V52 Move 退化 | +5.5% 中位（4/6 破防 ≤2% 护栏） | — | — | 联合训练损伤 |
| V5.3 rollout（r3 臂） | sign-agree 0.88 / 弧长饱和 0.62 / contact 0.08 | — | — | 上限控制器 |

种子/折分布：R3 每臂种子 17/29/43；F2-bidir fold0 三种子 0.884/0.896/0.949；B 折 f1 {0.927,0.983,0.967} f2 {1.019,0.946,0.999}。

## 4. 方向错误 / 幅度不足 / 过早停止 / rollout 漂移（用户要求四项显式回答）

1. **方向错误**：无证据。r3 sign-agree 0.88（rollout）与 sign median 0.888（离线）一致——模型方向基本正确。
2. **幅度不足**：是。rollout 弧长饱和 0.62（20×d0 预算到不了 TS 邻域）；离线 RMSE 仍距零步基线仅 ~9% 增益。
3. **过早停止**：训练侧是——43/48 R3 run 的 inner-val 选择 ≤250/3000 updates（冻结日程过长的证据，非 bug）；V5.2 stop 头 recall@0.5=0 是另一形态的"从不触发"。
4. **rollout 漂移**：classic 臂严重漂移崩塌（215 碰撞步）；r3 臂无碰撞但进展停滞；16/36 物理失败中 15 例溯源于碰撞种子几何（drift→物理失败的因果链完整记录）。

## 5. 失败修复 vs 仍然阻塞

- **已修复 8 项**（含 V5.2 λ-bug 15 attempts 烧毁、确认折泄漏 8 runs 自判无效、truth_guard 误报、worktree 工件漂移等，全表见 FAILURE_AND_REMAINING_WORK.md §1）
- **按设计阻塞 4 类**：confirm 臂（授权）、R1/R2（spec gap）、V5.2 余量（no-gain 后关闭）、folds3/4 封存
- **科学负结果 6 项**保存（不可调参消除，详见 §3 与 FAILURE §3）

## 6. 授权合规声明

- `formal_training_authorized`：前置全过（WP-L3+G0+G1+实值 SHA 绑定+审计）后开启；**收尾时已撤销**
- 恒 false 未动：`sealed_authorized` / `confirm_authorized` / `ood_authorized` / `fit150plus_authorized`
- 越权执行：0 次（24 个 BLOCKED 注解全在 ledger）；为凑 99 runs 强行执行：0 次
- 冻结件保护：run_plan E0 版归档、重绑结构 diff=NONE、campaign yaml/allowlist 全程未改

## 7. 产物索引

- 门记录：`docs/agent_runs/state/gates/G0..G4.json`（六门终态）+ `metric_unseal.json`
- 数据：`outputs/pesn/30h_campaign/data/`（六表+manifests，SHA 绑定）
- 训练：`r3_runs/`（54）、`v52_runs/`（16+探针+证据包+无效清单）
- V5.3：`v53/`（216 episodes、36 xtb runs、G4 证据包）
- 统计：`stats/r3_stats_report_final.{json,md}`
- 代码：master 提交 2630d6c→（本提交）；测试终态 **1369 passed / 0 failed**（agent_ops+pesn）
- 交付文件：本报告 + TIMELINE + R3_SUMMARY + V52 + V53 + FAILURE + NEXT_DAY_MEMO（仓库根目录）

## 8. 明日三大裁定问题

见 `NEXT_DAY_DECISION_MEMO.md`：① R3 确认折开启（泛化衰减真伪，6 runs/6min）；② V5.2 no-gain 根因诊断序列（H1 标签不可预测 vs H2 训练配置 vs H3 正例稀缺）；③ rollout 饱和对策（自适应步长 vs 进度感知修复）+ R1/R2 参数裁定。

---

**一句话总结**：数据/特征/账本基础设施全链路科学验证通过；R3 得到明确正结果（双向 intent 监督是唯一有效信号，dev 折确立）；V5.2 得到明确负结果（进度/停止监督在冻结配置下无增益且损伤 Move）；V5.3 完成工程闭环（含 36 次真实物理执行）但无 TS 科学主张；全部负结果按科研结论保存，等待你的三项裁定。
