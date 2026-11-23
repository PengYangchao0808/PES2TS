# archive/ — Zone 5 历史冻结归档区（只读）

> 状态：只读证据区（ADR-0036 §历史归档）。**任何活跃区（Zone 1–4）不得 import 本目录**；
> 本目录内部保持自洽，允许依赖 `infra.runtime`（无业务逻辑的运维 sink），其余方向禁止。
> 镜像标签：`archive/v5x-reorg-20261123`（本次重组后的只读冻结点）。

## 目录

| 路径 | 内容 | 来源（2026-11-23 重组） |
|---|---|---|
| `v4/` | V4 双端 TS 线：V4 模型诊断/评估脚本、round2/round3 探针 | 原 `scripts/archive/v4/`（35 文件）+ `scripts/archive/*.py`（14） |
| `v5x_scripts/agent_ops/` | V5.1–V5.4 战役运维工具（ledger / gpu_queue / runplan / supervisor / data_gate / feature_gate） | 原 `scripts/agent_ops/`（66 文件，`docs_consistency` 已提升到 `infra/governance/`） |
| `v5x_scripts/campaigns/` | V5.0–V5.4 战役驱动脚本 | 原 `scripts/campaigns/`（31 文件） |
| `v5x_scripts/campaign_oneoffs/` | 一次性战役脚本（demo24 / dual-ts / v41 / v42 / v51 诊断与构建） | 原 `scripts/` 根目录 64 文件 |
| `v5x_scripts/README_scripts_convention.md` | 原 `scripts/` 目录约定（历史） | 原 `scripts/README.md` |
| `v5x_configs/campaigns/` | 历史战役 YAML 与 13 个模板 | 原 `config/campaigns/` |
| `v5x_configs/pesn/` | 历史 PES-N 运行配置 | 原 `config/pesn/` |

## 只读纪律

1. **不新增**：本目录只接收"已冻结"的资产；新代码一律进五个活跃区。
2. **不修改**：确需修改时必须走宪法 §12 waiver + 提交信息带 `[unlock-legacy]`（`tests/kernel/test_arch_boundaries.py` 会拦截）。
3. **不 import**：`G3` 静态门禁禁止 `pes2ts_v5/**`、`pes2ts_data/**`、`infra/**`、非 legacy `tests/**` import `archive.*`。历史测试在 `tests/legacy/`，默认 `pytest -q` 跳过。
4. **证据不变**：归档内容保持移动时的字节不变（git rename 记录保留 blame 历史）；路径引用已在同一提交内机械重写。

## 复现

归档区内的历史脚本原样保留其 `ROOT = Path(__file__).resolve().parents[N]` 深度口径（随目录深度同步修正），
在仓库根执行即可复现当时行为，但**不再作为当前结论的证据入口**——当前证据入口见
`docs/reports/PES2TS_代码分区重构执行记录_v1_20261123.md`。
