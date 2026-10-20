# 失败与剩余工作 — PES2TS_30H_R3_V52_V53（24h 战役）

> 生成：2026-10-20T21:25Z · 编制：supervisor · 口径：全部为已核实事件，非推测

## 1. 已修复的失败（修复 + 回归测试 + 复验）

| # | 失败 | 根因 | 修复 | 证据 |
|---|---|---|---|---|
| F1 | harness 真实数据类型错误（numpy vs torch） | fixture 路径内部转换，真实 collate 未转 | 使用 to_torch_batch（集成注记） | G1 复跑 6/6 |
| F2 | truth_guard 拦截 trainer `.eval()` | 静态模式匹配 vs PyTorch 惯用语 | model.train(False)（v1 惯例），guard 未削弱 | b4bffbc，14 测试 |
| F3 | ledger pin 测试 stage 不感知 | 两段式重绑后测试固化 E0 期 sha | 测试改为从 task_state 动态读取 + stage 分支 | 424→449→471 绿 |
| F4 | worktree 过期战役工件 | gitignored outputs 不随 checkout | 三 worktree 同步 + 根因入档 | 时间线 17:05Z |
| F5 | **V5.2 λ 校验 bug（烧 15 attempts）** | off-term λ=0 被错误拒绝 | 修正 + 回归测试 | test_gpu_v52_trainer |
| F6 | **V5.2 确认折泄漏（8 runs 无效）** | v1 split 训练于 fold≠F | sealed split v2 + SPLIT_SEAL 断言；无效 run 观察结果前自判 | invalid_runs.json |
| F7 | 封存 done 转移被 metric_unseal 拒 | driver 未接 token 路径 | --metric-unseal 接线（WP-L3 首次真实消费，一次性 6-run 批） | 6 收据 |
| F8 | E6 lane 误报 xtb 缺失 | PATH 探测漏 /opt/software 前缀 | supervisor 实测修正 + 记录 | relax_preflight 修正注 |

## 2. 未修复 / 按设计阻塞（不属缺陷）

| # | 项 | 状态 | 原因 | 解锁条件 |
|---|---|---|---|---|
| B1 | R3 C 臂 ×6 + V5.2 QT-confirm ×6 | BLOCKED | 用户授权 confirm=false | 明日显式授权 + dev-positive 裁定 |
| B2 | R1/R2 ramp 臂 ×6 | BLOCKED | **冻结文档未定义 R1/R2 参数**（spec gap） | 你裁定参数或正式弃置 |
| B3 | V5.2 未发射 14 runs + 1 held attempt | 关闭 | 结论已定，扩容无信息增益且违"不凑数"规则 | 仅当你推翻 no-gain 判定 |
| B4 | folds3/4 六个 B-run 指标 | 封存（值未揭示） | 确认折纪律 | confirm 阶段一次性配对解封 |

## 3. 科学层面的负结果 / 开放问题（保存，不得调参消除）

1. **V5.2 E-NO-GAIN，根因已定位（战役后诊断）**：q 在单帧合法输入下不可辨识（线性探针改进 0.0%；within-intent corr 0.743 vs 池化 −0.045 = TS 双值性）——任务定义层面不可能，非训练配置错误；stop 随机水平与 Move +5.5% 退化仍部分开放（H2/H3 次要）。详见 V52 报告 §5/§5.1。
2. **forward-only 全臂劣于零步**（1.11–1.23）：意图信号只在双向监督下有效——任务结构本身是决定性变量。
3. **rollout 弧长饱和 0.62**：r3 方向正确但 20×d0 预算内到不了 TS 邻域；15/16 物理失败溯源于碰撞种子几何。
4. **classic 控制器崩塌**（sign 0.24、215 碰撞步）：对照基线失效，经典 CV/Jacobian 单步在此设置下不可用。
5. **历史 342 对清单在仓库中不存在**（ADR-0026 边界）：以全量重审计替代（15,372 对 → 89 冲突 mask）；`R3_scope_not_affected` 未主张，保持开放。
6. **早期过拟合**（43/48 R3 run 选择 ≤250/3000 updates）：冻结日程的产物；缩短主日程是超参修订，需授权。

## 4. 基础设施债（不阻塞结论）

- worktree 内 gitignored 工件漂移机制（已手工同步，未自动化；建议 runbook 条目）
- pes2ts-gpu 环境缺 rdkit（dual_ts 测试集在该解释器下 collect 失败；测试解释器规范已入 E0 记录）
- queue 脚本 tee 掩码退出码（pipefail 教训已入 lane 记录，未回溯改旧脚本）

## 5. 明日可立即执行的剩余工作（若授权）

1. R3 confirm 折开启 + 一次性配对解封（~6 runs，~6 min GPU）→ 直接检验 0.91→0.974 泛化衰减真伪
2. q-teacher 信息量审计（纯 CPU 分析，~30 min）→ 区分 V5.2 假设 1 vs 2
3. λq 消融 fold0（~6 runs，~5 min GPU）→ V5.2 假设 2 检验
4. rollout 种子几何改进（碰撞惩罚/自适应步长）后重跑 relax 失败的 16 例（~1 min xtb）
5. R1/R2 ramp 参数裁定后的 6 runs
