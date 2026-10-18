# scripts/ 约定

`scripts/` 只放**可复现入口**。代码能力在 `pes2ts_core/`；V 战役在 `config/campaigns/` 与 `scripts/campaigns/`。

| 目录 | 内容 | 是否入库 |
|---|---|---|
| `scripts/`（根） | 当前活跃入口：G0/G1/G2 数据、训练、评测、冻结 | 是 |
| `scripts/archive/` | 已收口的一次性脚本（历史证据） | 是 |
| `scripts/archive/v4/` | V4 诊断/审计脚本（V4.2–V4.7 已执行，负结果） | 是，只读 |
| `scripts/campaigns/` | V 战役编排入口（`run_v5_sprint.py`、`run_v7_rollout.py`） | 是 |
| `scripts/local/` | 会改写外部 ACP checkout、绑定本机路径的维护/探测脚本 | **否**（`.gitignore`） |

## 命名

- 能力脚本按 G 能力命名：`prepare_key_atom_teacher.py`、`train_key_atom.py`。
- 战役编排按 V 命名：`campaigns/run_v5_sprint.py`、`campaigns/run_v7_rollout.py`。
- 禁止把 `v5_`/`v6_` 前缀加到 `pes2ts_core/` 的模块名上；模块用能力名。

## 本机路径

脚本不得硬编码 `/mnt/e/...`、`/root/miniforge3/...`。需要外部路径时读环境变量：

```text
PES2TS_ACP_ROOT
PES2TS_ACP_PYTHON
PES2TS_ACP_CONFIG
PES2TS_GPU_PYTHON
PES2TS_CPU_PYTHON
PES2TS_DISK_ROOT
```

本地临时脚本放 `scripts/local/`，不要提交。历史脚本中的绝对路径只作证据，不保证在新机器可运行。
