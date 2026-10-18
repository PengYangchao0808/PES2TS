# PES2TS v4.1 阶段训练——修复输入语义与最小实验矩阵·最终报告

> 状态：阶段执行总结与问题复核；保留原门审、模型和冻结记录，不替代原始实验记录
> 日期：2026-10-17
> 范围：v4.1 WP1–WP5（输入语义修复 → 统一采样/协议冻结 → 零 QC 诊断 → A/B/C 最小实验矩阵 → 收口与决策包）
> 版本：报告 v1.1（含 G4 终门修正）；报告只读，不启动新训练、新 QC、扩量或方案切换
> 证据边界：数值全部取自已落盘工件（`matrix_summary.json`、`alpha_formal.{json,md}`、各 `run_record.json`、`a45_vs_a50.md`、`selection_record_v41.json`、协议 r0/r1/r2、冻结阈值/prereg、WP1–WP4b 报告）。本报告不重算、不重训、不重跑 monitor8，也不改变任何冻结件。

---

## 一、阶段结论

v4.1 完成两件事：**（1）修复 P01 输入语义缺陷**（观测-only 5 通道合同、ADR-0022、两条验收测试），**（2）重建可信实验基建**（统一采样器 60/60 真实等价、协议链 r0→r1→r2 内容寻址冻结、曝光账目、多种子门控、arm 元组绑定）。随后按冻结协议执行 A/B/C × 3 种子矩阵，得到注册门结果：

**注册门结果：`REGISTER_ABLATION`——没有任何臂通过 monitor8 冻结门。** 0/3 种子 × 3 臂，四项判据全不满足，且非边缘差距：最高 |a|-加权 cos 0.193 vs 阈值 0.30、最高近 TS cos 0.316 vs 0.40、最多「优于零动作」组数 3 vs 6。九个逐种子门判定均为 `FAIL`；三个臂均为 `BELOW_GATE`（见 `matrix_summary.json` 与重定稿 `selection_record_v41.json` 的 `gate_outcome=REGISTER_ABLATION`）。最近单项指标：A s18 MSE 比值 1.0183（高于阈值 1.0 约 1.8%），但同种子其余三项远低于阈值——没有任何臂接近完整过门。

四条关键科学线索（均带限定词，详见后续各节）：

1. **输入已修复但包回归未归因**：v4.1 A（干净 5-ch + 统一采样）在 monitor8 上 3 种子中位 4/4 差于 v4 bc_nll（7-ch + 全遍历采样）；这不能归因于「输入清理本身」，因为两因素（输入、采样器）未分离。
2. **训练目标 B 臂未有效**：方向损失（冻结 λ=559.37）在 fit 侧 MSE 与 cos 双升，monitor8 cos 也升，但 monitor8 MSE 与组数更差 → 中位规则判 NOT EFFECTIVE。
3. **方向表达 C 臂增益仅在特定口径下成立**：signed vector messages 中位层面四项全优，但逐种子仅 |a|-加权 cos 一致；atom-weighted pooled MSE 敏感性下优势反转。
4. **幅度收缩持续且随臂而异**：v4.1 checkpoint 实测 α 仍 ≪1，B 收缩最弱、A 最强；收缩仍集中低 cos 样本。

本矩阵**不能区分「数据不足 vs 表达不足」的最终归因**，因此下一阶段决策包给出三选项并明确推荐顺序（见「下一阶段决策包」）。

---

## 二、输入实现问题（已修复，归因未定）

**P01 关闭证据。** WP1 将教师标签派生的 `is_teacher_endpoint`/`near_zero_action_mask` 移出网络输入（node_scalar 7→5 观测-only），教师标签仅保留在监督目标、损失掩码与离线评测；特征合同升级 `pesn_dual_ts_features_v3`，shard 重建 bit-match，schema v3、宽度守卫、ADR-0022 落地，两条验收测试（教师标签干预非干扰；离线/在线一致性）跑通（见 `wp1-features-report.md`、`wp1-remediation-report.md`、`docs/design/decisions/ADR-0022-v4.1因果输入合同与特征v3.md`）。

**A-vs-v4 包比较表（3 种子中位 4/4 更差）。** 依 r1/r2 的 `package_comparison` 框架：v4.1 A 与 v4 bc_nll 之间差两个因素——干净 5-ch 观测-only 输入 **和** 统一有放回分组采样器（v4 为 7-ch 标签派生输入 + 全遍历采样）。因此 Δ 不得归因于输入清理本身。

| 指标 | v4 bc_nll（7-ch 全遍历，seed 20261017） | v4.1 A（3 种子中位） | Δ（v4.1 A − v4） |
|---|---|---|---|
| MSE 比值中位数 | 0.9443 | 1.0817 | +0.137（更差） |
| 优于零动作组数 | 5 | 2 | −3（更差） |
| 中位 \|a\|-加权 cos | 0.1578 | 0.1186 | −0.039（更差） |
| 中位近 TS cos | 0.2614 | 0.2393 | −0.022（更差） |

**同种子核对（s17）。** v4.1 A-s1（seed 20261017）MSE 1.0817 / 组数 2 / cos 0.1182 / 近 TS 0.2666，其中**近 TS 反而更好 +0.005**（0.2666 vs 0.2614），其余三项更差 → 同种子 3/4 更差；s18（MSE 1.0183 / 组数 3 / cos 0.1489 / 近 TS 0.2054）与 s19（1.0848 / 1 / 0.1186 / 0.2393）为 4/4 更差。**包回归结论成立**（3 种子中位 4/4、2/3 同种子 4/4），但**不能归因于输入清理本身**。注：v4 为单种子 run（20261017），v4.1 A 为 3 种子中位，比较不对称已注明。

来源：`wp4b-matrix-report.md` §「A-vs-v4 PACKAGE comparison」+ 追加 Erratum（finding 2），`matrix_summary.json`，`g3-oracle-review.md` §2(b)。v4 比较器使用标签派生 7-ch 输入（P01），**不得把 v4 bc_nll 当作可部署基线**。此未归因缺口直接引出决策包选项一（包归因 2×2）。

---

## 三、训练目标问题（B 臂）

B 臂为「尺度校准方向损失」`bc_d`，λ_dir=559.37（冻结程序校准：短程 A checkpoint，24 个 fit50 batch，seed 20261017；fit5 交叉 15687 越界被正确判废）。**无 monitor8 调参**：只跑了冻结 λ，无第二 λ、无 r3（F8 满足）。

**fit 侧 vs monitor8 并置：**

| 侧 | 指标 | B | A | 方向 |
|---|---|---|---|---|
| fit 侧 | MSE/base | 0.813–0.818 | 0.828–0.830 | 双升（更好） |
| fit 侧 | cos | 0.365–0.367 | 0.196–0.241 | 双升（更好） |
| monitor8 | cos 中位数 | 0.1910 | 0.1186 | 相对 +61% |
| monitor8 | MSE 比值中位 | 1.0984 | 1.0817 | 更差 |
| monitor8 | 优于零动作组数 | 1 | 2 | 更差 |

**结论：NOT EFFECTIVE（中位规则）。** B 在两个 cos 判据上严格优于 A，但 MSE 比值更差（1.098 > 1.082）且组数更差（1 < 2）→ 不满足 D 规则的「MSE 不更差、组数不更差」。逐种子读取同样不通过（MSE 更优 1/3、组数从不更优、近 TS 更优 2/3）。

**逐种子矩阵（A/B/C × 3 种子 × 四项门指标，取自 `matrix_summary.json` 的 `per_seed`）：**

| arm | seed | MSE 比值 | 优于零动作组数 | 中位 \|a\|-加权 cos | 中位近 TS cos |
|---|---|---|---|---|---|
| A | 20261017 | 1.0817 | 2 | 0.1182 | 0.2666 |
| A | 20261018 | 1.0183 | 3 | 0.1489 | 0.2054 |
| A | 20261019 | 1.0848 | 1 | 0.1186 | 0.2393 |
| B | 20261017 | 1.1248 | 1 | 0.1910 | 0.2192 |
| B | 20261018 | 1.0984 | 2 | 0.1769 | 0.2713 |
| B | 20261019 | 1.0547 | 1 | 0.1934 | 0.2847 |
| C | 20261017 | 1.0837 | 3 | 0.1396 | 0.3050 |
| C | 20261018 | 1.0555 | 2 | 0.1705 | 0.3158 |
| C | 20261019 | 1.0550 | 3 | 0.1466 | 0.2138 |

**逐种子 B−A 配对差（B−A per metric per seed）：**

| seed | Δ MSE 比值 | Δ 组数 | Δ \|a\|-加权 cos | Δ 近 TS cos |
|---|---|---|---|---|
| 20261017 | +0.0432（更差） | −1 | +0.0727 | −0.0474 |
| 20261018 | +0.0801（更差） | −1 | +0.0280 | +0.0659 |
| 20261019 | −0.0301（更好） | 0 | +0.0748 | +0.0454 |

逐种子 B−A：MSE 更优 1/3、组数从不更优、近 TS 更优 2/3（与上文一致）。最近单项指标：A s18 MSE 比值 1.0183（高于阈值 1.0 约 1.8%），同种子其余三项远低于阈值——没有任何臂接近完整过门。来源：`matrix_summary.json`（`per_seed`）。

**方向可推动、幅度权衡未测 α 前不能下结论。** B 的 fit 侧 MSE 与 cos 双升说明方向损失确实推动了方向（fit 侧），但 monitor8 MSE 更差不能反推为「幅度↓」——**α 已实测（见 §5）**：B 的 α 是三臂中最高的（0.217/0.217），即 B 收缩最弱，方向损失在保留幅度上确有作用，但该幅度保留未能转化为 monitor8 门通过。因此「目标函数是根因」未被证明，也未被排除。

来源：`wp4b-matrix-report.md` §D-rule + Erratum（finding 10），`matrix_summary.json`，`g3-oracle-review.md` §3，`alpha_formal.{json,md}`，`wp4a-staging-report.md` §λ calibration。

---

## 四、方向表达问题（C 臂）

C 臂为「有符号向量消息」`bc_nll + signed vector messages`。**3 种子中位层面四项全优**（D 规则判 EFFECTIVE），但逐种子读取必须带全部限定词：

| 指标（C vs A） | 逐种子 | 3 种子中位 |
|---|---|---|
| \|a\|-加权 cos | 3/3 更优（+0.0214 / +0.0217 / +0.0281） | 更优（0.1466 vs 0.1186） |
| MSE 比值 | 1/3 更优（s17 +0.0021、s18 +0.0372 更差，s19 −0.0298 更优） | 更优（1.0555 vs 1.0817） |
| 近 TS cos | 2/3 更优（+0.0384 / +0.1104 / −0.0255） | 更优（0.3050 vs 0.2393） |
| 组数 | 2/3 更优（+1 / −1 / +2） | 更优（3 vs 2） |

**聚合敏感性反转。** 注册指标是 8 组比值的逐组中位数（prereg 禁止「仅 pooled」声明），但若改用 atom-weighted pooled MSE 比值，C 的 3 种子中位**反而差于 A**（C 1.0840 vs A 1.0789；逐种子 1/3），这会翻转 D 规则的「MSE 不更差」条件、使 C 不 EFFECTIVE。门通过/失败不受影响（所有臂四项全失败）。**因此「signed messages 有效」必须带全部限定词**，不得写作「signed messages 严格帮助」。

**WP3 表达探针（旧模型 5 臂，方向信息可用性）。** 教师方向在可用等变向量张成空间上的投影残差中位数仅 2–3%（信息在中位意义上存在、可线性表达），但尾部 15.5%（fit50）/16.3%（monitor8）的 dir 原子残差 >0.5，未被一跳线性覆盖（one-hop linear availability probe）。这是一个**必要条件**框架：中位意义上几何方向信息在输入里，但尾部子集「信息缺失」成立——方向失败是幅度收缩 + 方向欠拟合的联合现象。

**D 跳过（稳健）。** D 需 B 与 C 各自有效；B 失败（无论中位或逐种子）→ D 正确 SKIP，且该 SKIP 与 C 的「EFFECTIVE」标签敏感性无关。没有启动 `bc_d --lambda-dir 559.37 --signed-vector-messages` run。

来源：`wp4b-matrix-report.md` §D-rule + Erratum（finding 1/8），`matrix_summary.json`，`g3-oracle-review.md` §2(a)，`wp3-diagnostics-report.md` §4，`monitor_prereg_v1_erratum_v41.md` §2。

---

## 五、幅度收缩（v4.1 checkpoint 实测）

WP3 的 α（0.05–0.24）原测自 **v4.1 之前的旧模型 5 臂**；v4.1 正式 checkpoint 的 α 在 close-out 阶段前向重测（零 QC，确定性 μ 动作，与 WP3 同方法），结论：**收缩持续且随臂而异**。

| arm | fit50 α 中位 | monitor8 α 中位 |
|---|---|---|
| A（干净 NLL，bc_nll） | 0.074 | 0.154 |
| B（方向损失，bc_d λ=559.37） | 0.217 | 0.217 |
| C（signed messages，bc_nll） | 0.144 | 0.181 |

对照 v4 旧模型 bc_nll：α fit 0.051 / monitor 0.107（WP3 §1 表）。全部 α ≪ 1，即模型确定性动作仍系统性收缩到教师幅度的约 5–22%。**B（方向损失臂）收缩最弱（α≈0.22）、A（干净基线）收缩最强（α≈0.05–0.15）、C 居中**。A 的 monitor8 α（0.154）与 WP3 旧值（0.107）量级一致，说明收缩不是 v4.1 输入/采样器改动引入的伪影，而是在正式 checkpoint 上重测仍成立。**α 由 0.107 升至 0.154（收缩略减）**（monitor8 α 0.107→0.154）。v4 α 为 pre-v4.1 包值（7ch 标签输入 + 旧采样）；收缩持续这一结论立足于 v4.1 正式 checkpoint 实测本身，v4 差值不可归因于输入或采样单因素。

**收缩仍集中低 cos 样本**：逐样本按 cos 四分位分解见 `alpha_formal.json`（例：A s17 fit50，cos Q1 α 中位 0.0039 → Q4 0.935；monitor8 同向）。逐样本恒等式 `‖â−a*‖²/‖a*‖² = 1+α²−2αcos` 残差 ≤~1e-14（F9 方法保留）；组级反推仍无效。

来源：`alpha_formal.{json,md}`（`wp41_diagnostics/`），`g3-closeout-report.md` §Headline，`wp3-diagnostics-report.md` §1。

---

## 六、跨反应覆盖（A45 敏感性对照）

A45（排除争议五组 `0160/0756/0778/1076/1174`，shard/spec 位同、全 8 组分母）为单种子（s17）诊断 run：

| 指标 | A50 (s17) | A45 (s17) | Δ |
|---|---|---|---|
| MSE 比值中位 | 1.0817 | 1.0401 | −0.042（更好） |
| 优于零动作组数 | 2 | 3 | +1（更好） |
| 中位 \|a\|-加权 cos | 0.1182 | 0.1183 | ≈0 |
| 中位近 TS cos | 0.2666 | 0.2858 | +0.019（更好） |

**结论：种子散布内无实质变化、门不变。** A50 三种子 MSE 范围 1.0183–1.0848（0.0665）vs A45 Δ −0.0416；近 TS 范围 0.2054–0.2666（0.0612）vs Δ +0.0192 → 单种子、Δ 落在 A50 种子散布内。A45 仍四项全失败。

干净 5-ch 困难组多样性重算：dead/responsive 距离比 **2.809 ≈ 2.808**（历史 7-ch），即移除标签通道后多样性差距几乎不变（距离由 grad_norm_scaled/delta_e_scaled/z_id 等观测通道主导）。**「覆盖不足为主因」既未被证实也未排除**：A45 无感 + 多样性比稳定，说明移除争议五组不改变门结论，但不能反向证明「数据已足够」。

来源：`wp4b-matrix-report.md` §A45 + Erratum（finding 7），`a45_vs_a50.md`，`wp3-diagnostics-report.md` §9，`g3-oracle-review.md` §4。

---

## 七、可拟合性 / 健康度

- **fit1/fit5 canary 样本内可拟合**：6 反应行 MSE 比值均 <1、|a|-加权 cos 为正、近 TS cos ≥ 普通 cos 于 6 行中 5 行（RXN_0000000623 例外：0.107 vs 0.165）；近 TS 不是主要失败模式（样本内可拟合结论不变）；含 160 强拟合（MSE 0.27、cos 0.78、α 0.85）与 623/733 弱尾（cos ~0.14、α ~0.06–0.07）。
- **9+1 run 全部 20k 完成、有限、无拒绝**：`updates_run=20000`、`samples_exposed=1280000`、9 个正式 run 各 214 epochs；A45 236 epochs（45 组/5491 态）、200 条有限 history（A45 亦 200）、无 sigma 病理、无 NaN/非有限指标（正式门接受为隐式——CLI 无 `ACCEPTED` 字段，由记录设置与无 `REFUSED` 佐证）。
- **fit50 短程健康**：NLL −17.44→−56.80 单调、梯度 73.1/3.36e-3/0.1095 无爆炸（>1e4）或消失（<1e-8）、覆盖 1.0（5967/5967）、21 epochs、近 TS 抽取占比 0.1744≈总体 0.1773（无偏）。

来源：`wp4a-staging-report.md`（canary 表、short-run health），`wp4b-matrix-report.md` §Anomalies 6，`g3-oracle-review.md` §6.10（staging 证据引 r1 记录）。

---

## 八、协议与治理

- **协议链内容寻址冻结**：r0 `dfc5a177…` → r1 `d3163614…` → r2 `26a3f789…`，无 r3；r2 冻结时刻早于首个正式 run。
- **prereg 勘误 annex（不改冻结件）**：`monitor_prereg_v1_erratum_v41.md` 登记 v2→v3 输入-only 管线替换（监督逐位、阈值/指标定义未动；prereg sha `24f99378…` 保留）；同 annex 登记 D 规则操作化。
- **selection 记录重定稿**：`selection_record_v41.json` → `gate_outcome=REGISTER_ABLATION`、`relative_best_recipe=bc_nll`（相对标签，非门通过）；`runs`/`per_recipe`/`seed_aggregation`/`evidence.monitor_evals` 均 9 项（含 C×3）；`evidence.fit_features_manifest` 指向协议 fit 目录（50 组/6067 态，spec `f678f8e5…`）。
- **r2 selection 记录 screening/canary_record 为空**：fit1/fit5/short 登记于 r1（`d3163614…`），被活动协议 sha（r2 `26a3f789…`）过滤，故 r2 的 `screening={}`、`canary_record={}`；staging 证据 = r1 的 fit1/fit5/short run 记录 + `wp4a-staging-report.md`。
- **阈值未降**：`monitor_thresholds_v1.json` sha `8ee45b65…`，门 1.0/6-of-8/0.30/0.40 未动，r2 副本与聚合脚本一致。
- **arm 元组绑定**：A `(bc_nll,1.0,1.0,False)`、B `(bc_d,1.0,559.37,False)`、C `(bc_nll,1.0,1.0,True)`；`bc_d` 非标 λ 被收、越界拒绝。

来源：`g3-closeout-report.md`（findings 1–8 修复），`selection_record_v41.json`，`monitor_prereg_v1_erratum_v41.md`，`g3-oracle-review.md` §Verification。

---

## 九、未决与边界

- **无外部验证**：confirm8 未触碰（无 confirm8 特征/run/脚本存在），锁定集（~1k xTB 点位估价）保留给未来 gate-PASS 候选；本阶段无 gate-PASS 模型可确认，故测量既不能改变门决定、又会消耗锁定外部集。若用户希望量取基线可行，但不建议在无候选时执行。
- **WP6 BC→IQL bundle 契约阻塞仍开**（G2 finding 1, High，属 WP4 范围外）：bundle 绑 v1 文件 sha + freerun action_version，与 IQL 需 v2 content digest + ACTION_SCHEMA_VERSION 冲突；解停前需 bundle 契约决定。
- **IQL 停靠依据** = 门未过 + 基建条件部分就绪（统一采样 60/60 已成立，但 bundle 契约未决），**不是「IQL 不可能受益」**。
- **「8/8 无中止」与 oracle「弧长」措辞**沿用 v4 修正（P07）：非「化学安全」，oracle 为配准后到参考 TS 的质量加权直线距离、非剩余质量弧长，仅作事后诊断。
- **v4.1 结果限 monitor8**（历史回归集，曾用于调参/模型选择/诊断），非封存测试集上的独立确认。
- **r2 冻结件 `selection_key` 残留**：r2 中 `selection_key=["recipe","seed"]` 为 close-out 前的描述性字段；权威记录键为 `arm/seed/stage`；未来 r3 应逐字登记（本报告不修改 r2 本体）。

来源：`g3-oracle-review.md` §5/§6.10，`progress.md` open-items，`g3-closeout-report.md` §Residuals。

---

## 十、成本

- **零新增 QC**：全部复用 v4 观测（WP3 诊断、A45、α 重测均零 QC）。
- **GPU 时长**：9×20k 正式 run + 短程 2000 + canary（fit1/fit5）+ A45 诊断 ≈ **2h 量级**（RTX 4060 Ti）。
- **无 confirm8 消耗**：confirm8 未运行。

---

# 下一阶段决策包

本矩阵不能区分「数据不足 vs 表达不足」的最终归因（A45 无感、多样性比稳定 → 不支持「数据主因」；C 的 cos 一致增益 + 表达探针中位覆盖 → 提示表达可提升，但未证明「表达是瓶颈」；B 的 fit 侧双升 + monitor8 更差 → 目标可推动方向但幅度权衡未闭合）。此外 v4.1-vs-v4 的包回归（输入 × 采样器）尚未归因。据此给出三选项，**均不得过度声称**（G3 #11）。

### 选项一（最小、最优先）—— 包归因 2×2

**内容**：2×2 四格——`input ∈ {干净 5-ch, v4 7-ch 标签通道} × sampler ∈ {旧全遍历 epoch, 新统一组平衡}`（或等效最小分解），回答「包回归来自哪里」。

- **证据**：v4.1 A（干净 + 新采样）4/4 差于 v4 bc_nll（7-ch + 旧全遍历），两因素未分离（§2）。
- **代价**：零新增 QC；每格 20k × ≥2 种子（推荐 3，沿用中位与 ≥2/3 种子规则）≈ 30 min GPU/格。
- **风险/预注册**：低；需预注册 4 格配置与聚合口径（沿用 monitor8 四项 + 中位规则），不得用 monitor8 调参。**解释规则**：因子效应（输入或采样主效应）以跨格均值/中位差判断，且须与格内种子散布比较（效应大于散布才可信）；本研究为归因探针，不改变冻结门。
- **分流**：
  - 若回归主要来自**采样** → 可恢复旧采样器 + 干净输入组合重跑矩阵，把输入修复与采样器独立验证。
  - 若回归主要来自**输入清理** → 说明旧模型确实依赖标签通道，与 WP3 ch6 敏感性（扰动近零通道 ΔMSE 比值 +0.02–0.14）定量衔接。

### 选项二 —— 扩数据（500/2k）

- **前提**：选项一先解释包回归，且 fit 侧可拟合性证据支持「泛化缺口」假说。
- **证据**：当前证据**不支持「数据是主因」**——A45 无感（§6）、干净 5ch 多样性比 2.809≈2.808（§6）、fit 侧方向拟合至多中等（A 0.196–0.241、B 0.365–0.367）。
- **前置**：按 v4 报告 WP0 前置（144 identity partner 补绑定、family 审计、重定价）逐项关闭后才放行。
- **预注册**：比较口径 = 扩数据后矩阵在 monitor8 四项门指标的逐种子中位 vs 冻结阈值；种子 ≥2（推荐 3），沿用中位与 ≥2/3 种子规则；决定规则 = 若 ≥2/3 种子且臂级趋势确立（消除「覆盖不足」主因疑点或逼近门通过）才继续，否则停止。
- **代价/风险**：GPU/QC 封套 = 每格 20k × ≥2 种子、零新增 QC；失败/回退 = 无增益则回退 v4.1 结论、不降阈值；数据/工程风险 = 需重新制备与重定价（WP0 前置），工程成本高、收益证据弱，当前不应直接启动。

### 选项三 —— 架构增强

- **证据**：C 的 |a|-加权 cos 一致增益（3/3）+ 表达探针中位覆盖（残差 2–3%）。
- **但「架构是瓶颈」未被证明**：方向可被目标改动推动（B 的 fit 侧 cos 双升 0.365–0.367，§3）。
- **若做**：从「signed messages 组合 B 目标」开始（D 需 B、C 各自有效——B 当前失败，应在选项一/λ fit 侧敏感性之后再评估 D 前提）。
- **预注册**：比较口径 = 新臂「signed messages + B 目标」（bc_d λ=559.37 + signed vector messages）vs A/C 在 monitor8 四项的逐种子中位；种子 ≥2（推荐 3）；决定规则 = 若新臂四项判据达 ≥2/3 种子通过则视为新候选，否则停止。
- **代价/风险**：GPU/QC 封套 = 每格 20k × ≥2 种子、零新增 QC；失败/回退 = 若 B 自身仍无效则停止、不降阈值；数据/工程风险 = 新组合是新注册臂，需新协议（r3）逐字登记，工程与协议成本中。**signed messages + B 目标组合是一个新注册臂，不得复用已失败的 D 前提（D 需 B、C 各自有效；B 当前失败）。**
- **边界（F10）**：叉积/对端构象仍属新输入合同，需独立论证，不随本选项附带放行。

**推荐顺序：选项一 → 视结果定二/三（二的前置审计可并行准备）。** 明确：本矩阵不能区分「数据不足 vs 表达不足」的最终归因；未经选项一不得把包回归归因于任何单因素，也不得把「架构增强」或「扩数据」当作已被本矩阵证明的方向。

---

## 证据索引

- 完整日志：[progress.md](../../.slim/deepwork/pes2ts-dualts-v41/progress.md)；G3 门审十一项 must-have：[g3-oracle-review.md](../../.slim/deepwork/pes2ts-dualts-v41/g3-oracle-review.md)；G3 close-out：[g3-closeout-report.md](../../.slim/deepwork/pes2ts-dualts-v41/g3-closeout-report.md)。
- 矩阵数值：[matrix_summary.json](../../outputs/pesn/dual_ts_v2_20261015/wp41_diagnostics/matrix_summary.json)、[matrix_summary.md](../../outputs/pesn/dual_ts_v2_20261015/wp41_diagnostics/matrix_summary.md)；A45：[a45_vs_a50.md](../../outputs/pesn/dual_ts_v2_20261015/wp41_diagnostics/a45_vs_a50.md)；α 前向实测：[alpha_formal.json](../../outputs/pesn/dual_ts_v2_20261015/wp41_diagnostics/alpha_formal.json)、[alpha_formal.md](../../outputs/pesn/dual_ts_v2_20261015/wp41_diagnostics/alpha_formal.md)。
- WP4b：[wp4b-matrix-report.md](../../.slim/deepwork/pes2ts-dualts-v41/wp4b-matrix-report.md)（含追加 Erratum）；WP4a：[wp4a-staging-report.md](../../.slim/deepwork/pes2ts-dualts-v41/wp4a-staging-report.md)；WP3：[wp3-diagnostics-report.md](../../.slim/deepwork/pes2ts-dualts-v41/wp3-diagnostics-report.md)。
- 治理件：[selection_record_v41.json](../../outputs/pesn/dual_ts_v2_20261015/wp41_bc/selection_record_v41.json)、[monitor_prereg_v1_erratum_v41.md](../../outputs/pesn/dual_ts_v2_20261015/wp4_bc/monitor_prereg_v1_erratum_v41.md)、[monitor_thresholds_v1.json](../../outputs/pesn/dual_ts_v2_20261015/wp4_bc/monitor_thresholds_v1.json)、[protocol_v41_r2.json](../../outputs/pesn/dual_ts_v2_20261015/wp4_bc/protocol_v41_r2.json)。

报告登记的是事实与待验证问题；未修改任何历史门审、模型、数据、阈值、prereg、协议或科研结论原件。
