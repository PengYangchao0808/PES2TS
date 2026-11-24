# AGENT.md — PES2TS 宪法执行投影（Enforcement Projection）

> 状态：权威（执行投影，**非独立规范**）
> 规范根（SSOT）：[`docs/PES2TS_开发宪法_v1.md`](docs/PES2TS_开发宪法_v1.md)（内容 **v2.10** @ 2026-11-24）
> 机制规范：[ADR-0036](docs/design/decisions/ADR-0036-功能分区与单一训练引擎.md)（三板块/单向依赖/五道硬门）
> 当前版本：v4.0 @ 2026-11-24（三板块清场后重写；≤150 行）

**本文件不是规范。规范在宪法；任何分歧以宪法为准，并立即修正本文件。** 本文件只做四件事：强制读宪法、列出红线、给出唯一命令入口、标明 No-Read/No-Write 边界。

## 1. 布局（三板块 + 薄基础设施）

| 位置 | 一句话职责 | 硬约束 |
|---|---|---|
| `pes2ts_core/` | 化学/物理内核：`g0/`、`g1/`、`generation/`、`integration/`、`utils/`、`contracts/` | **禁止 torch**；不得 import `pes2ts_v5`/`pes2ts_data` |
| `pes2ts_data/` | 数据与真值层：`contracts/`、`readers/`、`splits/`、`ingest/`、`builders/` + **唯一 R1 桥 `pes2ts_data/truth_bridge.py`** | 只读数据；不得 import torch 训练栈 |
| `pes2ts_v5/` | **唯一**训练/评测引擎：`configs/`、`data/`、`models/`、`losses/`、`training/`、`evaluation/`、`runner.py` | 唯一 `.backward()`/`optimizer.step()` 落点 |
| `infra/` | 门禁与运维：`governance/`（docs/links/G–V/arch_guard）、`runtime/` | 不 import 业务代码 |
| `config/`、`tests/{kernel,data,v5}` | 实验 YAML、划分/G–V registry；三层活跃测试 | 按 `pytest.ini` 标记 |

历史 `archive/`、`tests/legacy/`、`pes2ts_core/pesn/` 已移出 master，只存在于只读分支 `frozen/v4-v54-history`（tag：`archive/pre-v55-legacy-20261123`、`archive/v5x-reorg-20261123`）。

## 2. 唯一命令入口

```text
python -m pes2ts_v5.runner {train,eval,sweep,smoke} --config config/experiments/<name>.yaml
```

- 训练：`python -m pes2ts_v5.runner train --config config/experiments/v56_baseline_gru.yaml`
- 评估：`python -m pes2ts_v5.runner eval --run-dir outputs/v5/runs/<run>`
- 扫描：`python -m pes2ts_v5.runner sweep --config config/experiments/v56_arch_matrix.yaml --arch M01`
- 预检：`python -m pes2ts_v5.runner smoke --config config/experiments/smoke.yaml`

门禁（缺一不可）：

```text
python infra/governance/docs_consistency.py --strict
python infra/governance/check_docs_links.py
python infra/governance/render_gv_map.py --check
python infra/governance/arch_guard.py --json
python -m pytest -q tests/kernel
python -m pytest -q tests/data
python -m pytest -q tests/v5
```

## 3. 四条工程红线（触线 = 架构变更，走宪法 §12）

1. **禁止第二训练器/第二反向站点** —— 全仓只有 `pes2ts_v5/training/` 可出现 `.backward()` / `optimizer.step()`。
2. **禁止战役版本号 `.py`** —— 活跃区不得新增 `v54_*.py`、`v6_*.py` 等；新架构 = 1 个 core 文件 + 1 份 `config/experiments/*.yaml`。
3. **活跃文件 ≤350 行** —— 超限先拆分，不堆长文件。
4. **禁止自造框架/脚本** —— 不新增 `scripts/` 脚本堆、不引入自研训练框架/调度器；用现有 runner、registry、callback 与 gates。

## 4. 红线自检（宪法 §3；只列 ID + 一句话，非完整规范）

- **R1 真值隔离** —— TS/IRC 真值只经 `pes2ts_data/truth_bridge.py`（唯一 R1 桥）；不得进入生成输入/`s_t`/在线推理。
- **R2 物理标签唯一** —— 成功 := OptTS+Freq+双向 IRC+身份；RMSD 只作诊断。
- **R3 ACP 唯一生产执行后端** —— 不实现引擎、不直启 xTB/ORCA；CI 测试替身除外。
- **R4 动作硬约束** —— ML 动作必须落在 `A(H)` 内，越界由构造拒绝。
- **R5 全量尝试与成本留存** —— 禁止成功分子独占分母；结果须带成本账本。
- **R6 评测完整性** —— 强制 reaction-family holdout；分裂防火墙；禁止对 holdout 调参。

工程政策：P1 接口版本化 / P2 typed rejection / P3 可回放 / P4 单一归属 / P5 先击败基线。

## 5. No-Read / No-Write Zone（只在冻结分支，不在工作树）

- **工作树内不存在**：`archive/**`、`tests/legacy/**`、`pes2ts_core/pesn/**`；不得重建、不得以软链/副本复活。
- **只读访问**：用 `git show frozen/v4-v54-history:<path>`、`git worktree add <tmp> frozen/v4-v54-history` 或 tag 取历史；历史只读、不改写。
- **活跃代码不得 import 历史模块**（`pes2ts_core.pesn...`、`archive...`、`tests.legacy...`）；`arch_guard` G3 强制。
- **冻结区修复**（历史文件确需改动）须所有者批准 + 提交信息含 `[unlock-legacy]`（ADR-0036 §5），且只在冻结分支操作。

## 6. 文档纪律（不新增规范）

1. 规范只写在宪法；其余文档引用，不重复。
2. plan/design 顶部状态块：`状态 / 层级 / 取代关系 / 上游依赖 / 下游消费者 / 当前版本`。
3. 新增/修改计划文档后同步 `docs/plans/PES2TS_文档修订台账.md` 与 `docs/README.md`。
4. 报告只读；历史链接由 `check_docs_links.py` 的 frozen-history WARN 策略容忍（不产生 error）。

## 7. 当前指针

- 活地图：实际目录树 + [`infra/governance/arch_guard.py`](infra/governance/arch_guard.py)；`docs/design/PES2TS_CODE_MAP_v3_20261123.md` 已标 **Historical**（停更）。
- 清场记录：[`docs/reports/PES2TS_代码分区重构执行记录_v1_20261123.md`](docs/reports/PES2TS_代码分区重构执行记录_v1_20261123.md) §12。
- 宪法增量：v2.10（三板块所有权/历史冻结/`pesn` 条款对新开发失效；R1–R6 不变）。
