# V5 架构选型报告（模板）

> 状态：DRAFT（由 `pes2ts_v5.training.sweep` 生成骨架；人工补写结论后才可 Frozen）
> 数据：`DATASET_VERSION` / hash `DATASET_HASH`
> 配置：`CONFIG_PATH` / hash `CONFIG_HASH`
> 生成时间：`TIMESTAMP`

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
