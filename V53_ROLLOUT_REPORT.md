# V5.3 Rollout 报告 — PES2TS_30H_R3_V52_V53

> 生成：2026-10-20T21:25Z · 编制：supervisor · G4-Rollout = **PASS（仅工程闭环，非科学验证）**
> 模式：**H20**（V5.2 监督失败 → 上限控制器 only；v52 臂 diagnostic-only，机器强制标注）
> 证据：`outputs/pesn/30h_campaign/v53/g4_evidence_pack.json`（engineering_closure_ok=true）

## 1. 执行内容

- **无物理 rollout（216 episodes）**：12 条预注册冻结曲线 × {forward, reverse} × {5, 10, 20} 步 × 三臂 {classic, r3=F2-bidir, v52-diagnostic}；信任域 0.20Å；逐步重嵌入只用移动后几何（禁入输入机器检查）；逐比特确定性（双 episode 复跑验证）；预算奇偶性 72 组机器校验
- **xtb 受约束松弛（36 次真实物理执行）**：xtb 6.7.1（`/opt/software/xtb/bin/xtb`），每曲线 20 步 rollout 终点为种子；跨臂**同一冻结配方**（GFN-2, k=50, normal, maxcycle 500, 4 线程, 300s）+ 逐比特相同的 CV 约束目标（奇偶性机器校验）；全部 stdout/stderr/几何入档
- 成本：rollout 8s + relax 49s（wall）；GPU 未使用（模型 55k 参数，CPU 快于锁开销）

## 2. Rollout 结果（诚实诊断）

| 臂 | sign-agree | contact-made | 弧长进展 | 崩溃形态 |
|---|---|---|---|---|
| classic（CV/Jacobian） | **0.24** | 0.04 | tsRMSD 0.68→1.02Å 恶化 | 20 步 215 次碰撞步 |
| **r3（学习上限）** | **0.88** | 0.08 | **饱和于 ~0.62** | 无碰撞但进展停滞 |
| v52（diagnostic-only） | 表面最光滑 | 0.08 | — | 20 步 40 次 stop 事件（随机水平，与 G3 一致） |

**结论：没有任何臂展示 TS 逼近能力。** r3 方向正确（sign 0.88）但在 20×d0=0.08Å 预算内弧长进展饱和于 62%；classic 臂漂移崩塌，不适合作对照基线（已记录）。

## 3. 受约束松弛结果（真实物理）

| 状态 | 数量 | 说明 |
|---|---|---|
| converged | 5/36 | CV 目标偏差 ≤6e-4Å |
| not_converged | 15/36 | 500 cycles 内未达目标 |
| physics_failed | 16/36 | 15/16 归因于**种子几何碰撞**（最小原子间距过近，\|grad\|>500 优化器中止；seed_min_interatomic 逐 run 记录） |

失败的因果链清晰：rollout 20 步终点几何质量差 → 碰撞种子 → xtb 中止。这把改进方向指回 rollout 几何质量（步长/信任域/碰撞惩罚），而非执行器本身。

## 4. 声明范围（审计）

- **TS_candidate_only**：relax 产物 `xtbopt.xyz` 均为**未验证 TS 候选种子**；未尝试虚频验证（OptTS+单虚频+IRC 连接属后续工作）
- claim-scope 审计：0 处肯定性 verified-TS 表述（16 处提及全部为否定/禁止语境，机器审计）
- v52 全部数值携带 `v52_supervision_failed_no_claim` 标注

## 5. 交付状态分类

- 实现：**implemented**（rollout 执行器 + relax 执行器 + G4 证据包，+8 测试）
- 测试：fixture_passed（29 rollout 测试）+ 真实执行 smoke_passed→executed（216+36 episodes，确定性验证）
- 科学结论：**engineering-validated only**——闭环可运行、成本可记录、声明纪律受控；**不构成任何 TS/IRC 科学验证**
