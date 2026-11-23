"""Report schemas + read/write helpers (CSV is the only ranking source).

Plan v2.1 §7.1–§7.2: ``leaderboard.csv`` / ``experiment_registry.csv`` /
``failure_taxonomy.csv`` / ``model_selection.md``.  All experiment metadata
required for reproducibility is recorded in the registry: dataset version,
data hash, code/config hash, seed, parameter counts, training budget, weights
checksum.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

REGISTRY_COLUMNS = [
    "experiment_id", "stage", "arch", "seed",
    "dataset_version", "dataset_hash", "config_hash", "code_hash",
    "input_mode", "n_curves", "n_trajectories", "n_frames", "n_decisions",
    "encoder", "encoder_params", "core_params", "head_params", "total_params",
    "param_budget", "core_hidden_dim", "core_layers",
    "epochs_run", "best_epoch", "train_seconds", "device",
    "best_val_loss", "move_rmse", "move_cosine", "process_acc", "process_auc",
    "progress_mae", "trust_auc", "cpu_step_ms", "cpu_p95_ms",
    "weights_sha256", "checkpoint_path", "status", "notes", "timestamp",
]

LEADERBOARD_COLUMNS = [
    "rank", "experiment_id", "arch", "seed", "status",
    "core_params", "total_params", "move_rmse", "move_cosine",
    "process_acc", "process_auc", "progress_mae", "trust_auc",
    "cpu_step_ms", "notes",
]

FAILURE_COLUMNS = [
    "experiment_id", "arch", "stage", "failure_class", "error_message", "timestamp",
]

MODEL_SELECTION_TEMPLATE = """# V5 架构选型报告（模板）

> 状态：DRAFT（由 `pes2ts_v5.training.sweep` 生成骨架；人工补写结论后才可 Frozen）
> 数据：`{dataset_version}` / hash `{dataset_hash}`
> 配置：`{config_path}` / hash `{config_hash}`
> 生成时间：{timestamp}

## 1. 目的与范围
（V5 离线架构 Benchmark；不产生 QC 成功/物理可靠性声明）

## 2. 输入表
- leaderboard：`reports/leaderboard.csv`（唯一排名依据，禁止手工改写）
- 全部运行：`reports/experiment_registry.csv`
- 失败归因：`reports/failure_taxonomy.csv`

## 3. 工程门（初筛）
| 门 | 判据 | 结果 |
|---|---|---|
| 可收敛 | 1 epoch 后 loss 有限 | 待填 |
| 变长轨迹 | 支持长度 2..max | 待填 |
| 状态保存/恢复 | stateful 架构可 carry | 待填 |
| CPU batch=1 | 单步可推理 | 待填 |
| 无额外依赖 | 仅 torch/numpy/pandas/yaml | 待填 |

## 4. 结果与排序
（从 leaderboard.csv 生成；初筛结果不得作为正式架构排名）

## 5. 负结果与限制
（如实填写：未收敛、NaN、慢、简化实现未验证等）

## 6. 选型结论
- [ ] 进入 V5.6-B 的 6–8 个架构：
- [ ] 进入 V5.7 的 4 个架构：
- [ ] 待裁决项：

## 7. 复现命令
```bash
/root/miniforge3/envs/pes2ts-gpu/bin/python scripts/v5/run_screening.py --config config/v5/screening.yaml
```

## 8. 签署
- 生成者 / 日期：
- 复核者 / 日期：
"""


def ensure_reports_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    for name, columns in (
        ("experiment_registry.csv", REGISTRY_COLUMNS),
        ("leaderboard.csv", LEADERBOARD_COLUMNS),
        ("failure_taxonomy.csv", FAILURE_COLUMNS),
    ):
        file = out / name
        if not file.exists():
            write_csv(file, columns, [])
    return out


def write_csv(path: str | Path, columns: list[str], rows: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})
    os.replace(tmp, path)
    return path


def append_rows(path: str | Path, columns: list[str], rows: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})
    return path


def write_model_selection_template(path: str | Path, **fields) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(MODEL_SELECTION_TEMPLATE.format(**fields), encoding="utf-8")
    return path
