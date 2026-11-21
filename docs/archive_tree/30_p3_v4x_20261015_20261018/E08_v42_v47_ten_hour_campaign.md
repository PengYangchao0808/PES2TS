# E08 V4.2–V4.7 十小时战役（Executed-Frozen 负结果）

> **状态**：Archived（档案条目，冻结后只可追加“后续指针”）
> **层级**：L3 证据 / 开发档案树 · P3
> **取代关系**：无（档案条目；原文已被退役或作为指针保留）
> **上游依赖**：ADR-0035、`docs/infra/doc_cleanup/20261121/ARCHIVE_TREE.md`
> **下游消费者**：维护者、季度审计、AI 会话
> **当前版本**：v1.0 @ 2026-11-21
> **条目编号**：E08　**所属期**：P3（2026-10-17→10-18）　**关联 ADR**：ADR-0022/23/24　**负结果**：N08,N09

---
> 条目标号：E08　时间范围：2026-10-17 → 2026-10-18　状态：Archived（Executed-Frozen）
> 关联 ADR：ADR-0022（v4.1 因果输入合同）、ADR-0023（V4.2 原生重连数据）、ADR-0024（V4.2 无界位移模型与战役训练协议）；合同：`docs/contracts/PES2TS_V4.2-V4.7_实施合同与战役冻结_v1.md`（Frozen）
> 数据/代码/模型版本指针：`campaign_manifest.json`、`native_reconnect_audit.json`、`gradient_baseline_audit.json`、`protocol_freeze.json`、`run_records/*/run_record.json`、`protocol_evals/*.json`（均在 `outputs/`，见报告证据边界）
> 证据根：`docs/plans/PES2TS_V4.2-V4.7_十小时训练实验方案_20261017.md`（v1.1）、`docs/reports/PES2TS_V4.2-V4.7_过夜战役汇总_20261018.md`、`docs/reports/PES2TS_实现报告_20261018.md`

## 目标
在一个单 GPU 10 小时窗口内，用 V4.2–V4.6 五组对照 + V4.7 三种子定稿，判断"原生重连数据 + 无界 Cartesian 位移 + campaign_v1 协议"能否把双端 IRC 教师学习推到四门之上。

## 核心思路
- 先审计（原生标签/坐标、解析梯度基线、测速），再训练；每项收益预设判据。
- 数据版本升级：剔除 399 个插入节点，重建 50 组/100 分支/5,568 个原生动作。
- 模型侧：取消 `0.1·tanh` 动作上限，直接回归 Å 位移；旧 radial-tanh 头保留为 D0/D1 对照。
- 评测侧：三口径协议（legacy_full / native_full / common_unchanged）在看预测前冻结；四门原样保留。
- 工程侧：可恢复战役驱动器（预检、测速、阶段晋级、预注册裁剪、预算删失、失效 run 归档）。

## 关键决策
- ADR-0023/0024（V4.2 数据与模型协议）；实施合同 Frozen（训练开始后不得修改）。
- V4.7 配方先冻结（最终配置、步数、选权重规则），再在完整 fit50 从头训三种子（20261017/18/19）；筛选与定稿分开命名与统计。

## 主要成果与量化指标
- **23 个正式 run 全部 completed**（20000 updates/run；峰值显存 65–103 MB；吞吐约 28–49 steps/s）。臂：D0/D1（legacy_v41 头）、D2/D3/E1/E2（campaign_v1）、FINAL×3、H2、V1、W96、W128。
- 5 项新工程能力：原生重连数据、无界位移头、`campaign_v1`（AdamW + warmup/cosine + EMA + u5k/u10k/u20k 检查点 + 曝光账本 + 断点恢复）、三口径冻结评测、可恢复战役驱动器。
- **V4.7 FINAL 三种子中位（native_full 主口径）：MSE/zero 1.132、优于零步 0/8、整体 cos 0.180、near-TS cos 0.248**；legacy_full 与 common_unchanged 同方向结论。

## 关键负结果
- **V4.7 判定负结果**：三项种子四项门全部未通过；不晋级、不部署、不扩量，保留为冻结证据（方案状态头原文）。
- 全部筛选臂（D0–D3、H2、E1/E2、V1、W96/W128）均 `BELOW_GATE`，没有任何一条通过四门。
- H1 因物理先验 `Δs_fit=0` 被预注册门判 `prior_not_supported`（不是训练失败，是假设不成立）。
- V2 因接触边实现仅 **0.43 steps/s** 按预注册顺序裁剪；40k 延长门判 `STAY_20K`；E2 出现 `MODEL_LOAD_FAILED`（三口径行跳过）。
- 诊断结论：fit 侧 MSE 下降（E1 0.706、W128 0.681、V1 0.516）而 dev 侧 MSE 升到 1.26–1.49、优于零步组数 0 → 当前 50 组数据 + 单帧 5 通道 + 全原子位移目标 + 20k 预算的组合存在**严重跨反应泛化缺口**；fit45 内 **342 对近似相同输入但教师方向冲突** → 优先解决输入可辨识性/训练时长/数据覆盖，而不是继续堆容量。

## 限制与未解问题
- 10 小时窗口只覆盖训练与离线评估；真实 rollout、新 QC、500/2k 扩量、IQL/SAC 不计入本夜。
- monitor8 是历史回归集不是封存测试集；confirm8/test/family_holdout 未触碰（不能声称泛化）。
- 三口径协议虽冻结，但协议间差异较大（见过夜汇总表），单一口径不能代表全貌。

## 证据指针
- `docs/plans/PES2TS_V4.2-V4.7_十小时训练实验方案_20261017.md`（状态头 `Executed-Frozen`；§2 基线表）
- `docs/reports/PES2TS_V4.2-V4.7_过夜战役汇总_20261018.md`（§1 run 表、§2 三口径评测）
- `docs/reports/PES2TS_实现报告_20261018.md`（摘要：负结果与 342 冲突对）
- `docs/contracts/PES2TS_V4.2-V4.7_实施合同与战役冻结_v1.md`

## 后续影响
- 负结果 + "输入可辨识性/泛化缺口"判断直接促成 E09 的 V5.0–V5.2 冲刺方案与 E11 的任务定义纠偏讨论。

---

### 第 4 期 · V5 冲刺与 24H 战役期（2026-10-18 → 2026-10-21）

一句话目标：在 V4 负结果之后，用两个短周期战役（V5.0–V5.2 十小时冲刺、24H 自主战役 R3→V5.2→V5.3）回答"任务定义是不是错了"——答案是：R3 方向成立而 V5.2 的 Stop 任务定义错了，于是 10-21 当天完成两次任务语义修订。

核心思路：①把"路径导航"拆成 Move / Progress / Stop 三个可分别证伪的假设；②训练前冻结门与预注册，失败不调参复活；③多 lane 并行（DATA/FEAT/LOSS）+ fail-closed 授权；④负结果按"已修复/负结果/按设计阻塞"三类分账；⑤任务定义错误要写进 ADR 撤回，而不是悄悄改标签。

关键技术决策：ADR-0026（V5 冲刺 key-atom 位移合同与 OOF 语义，Accepted）、ADR-0027（V5.1 参考架 fallback 与双分母评估，Proposed）、ADR-0028（关键原子几何策略合同与 V5–V7 任务拆分，Proposed）、ADR-0029（V5.2+ 双门评估与封存边界，Proposed）、ADR-0030（全路径导航与端点停止，Accepted，10-21）、ADR-0031（MOVE–PROCESS 统一学习合同，Accepted，10-21）。

主要成果与指标：V5 冲刺 `complete_oof_1seed`（prepare-audit / label-screen / recipe-recheck / crossfit 完成，report not_executed，封存评测 REFUSED，复核稳定配方 A05）。24H 战役：11 个工作包全部终态、六门终态；93 次 launch attempts（R3 61/66、V5.2 32/33）；GPU 训练 0.93 GPU-h（70 runs × ~48s）；物理 xtb 成本 49s（36 次受约束松弛）；墙钟 7.75h。R3 冠军 **F2-bidir：RMSE/zero 0.910 [0.884–0.949]、sign 0.888、6/6 优于零步、三种子一致**；跨折泛化 0.9735 [0.949–0.998]、10/12、衰减约 6.4pp。V5.3-ENG rollout：216 episodes + 36 次真实 xtb，工程闭环 PASS。

关键负结果：V5.1a 在 5-fold 门被 rejected，V5.1a-R1 无合格候选，V5.1b 保持关闭；V5.2 **q 不可学习**（MAE 0.507 vs 常数 0.476，corr≈0）、**stop 头随机**（AUC 0.491–0.527、recall@0.5=0）、**Move 退化 +5.5% 中位（4/6 破 ≤2% 护栏）**；classic 臂严重漂移（215 碰撞步），16/36 物理失败中 15 例溯源于碰撞种子几何；R3 43/48 run 的 inner-val 选择 ≤250/3000 updates（冻结日程过长的证据，非 bug）；V5.2 的 15 次 λ-bug 烧毁 attempts 计入总账（失败修复记录）。任务层根因（E11 审计）：V5.2 的 Stop 任务本身定义错——"应该 Stop"的终点样本被 `terminal` 排除在训练外。

数据/代码/模型指针：`docs/agent_runs/state/gates/*.json`（门禁证据，不删）；7 份根 memo（`24H_EXECUTION_TIMELINE.md`、`24H_FINAL_REPORT.md`、`FAILURE_AND_REMAINING_WORK.md`、`NEXT_DAY_DECISION_MEMO.md`、`R3_RESULT_SUMMARY.md`、`V52_PROGRESS_STOP_REPORT.md`、`V53_ROLLOUT_REPORT.md`）；`V53MP_RESUME_20261022.md`（转 E12）。当期原始文档 git 范围：10-18…10-21 共 111 次提交（17+14+43+37）。
