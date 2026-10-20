# R3 结果总结 — PES2TS_30H_R3_V52_V53

> 生成：2026-10-20（战役内）· 编制：supervisor · 状态：R3 阶段关闭（dev-positive / confirm-blocked）
> 口径：全部指标为 dev folds 0–2 独立 curve/cluster 聚合；folds3/4 封存（6 runs，值未入任何聚合）；confirm 臂按用户授权 BLOCKED

## 1. 执行账目（诚实口径）

- **launch attempts: 61/66**（上限计所有 attempt 含 retry）：54 done = F18 + B12 + V12 + L12；C6 + R6 BLOCKED；5 attempts 未使用（预留关闭）
- 1 次 driver bug retry（F0-s17，已披露）；1 次封存终态化经 eval-only resume（计入 attempts）
- GPU 成本：~0.9 GPU-h（~45–55 s/run，3000 updates，StarEGNNV3 55,441 参数 fp32）
- 吞吐：43.25 updates/s（smoke 实测）

## 2. 数据与基线

- 数据：50 canonical IRC / 100 双向 intent / 12,034 样本（ok 9,655 含 15 near-conflict 降权 0.25；terminal 1,123；conflict 68；masked 1,188）；可用曲线 48
- 基线（同表可比）：zero-step、classic CV/Jacobian min-norm（公式入档）、EGNN=StarEGNNV3；旧 OOF 0.824 按协议排除

## 3. 跨臂主结果（RMSE/zero，<1 优于零步基线；cluster 级 CI95）

| 臂 | n | RMSE/zero [CI95] | sign med | 优于零步簇 |
|---|---|---|---|---|
| **F2-bidir（fold0）** | 3 | **0.910 [0.884–0.949]** | 0.888 | **6/6** |
| S2 / S10 / D2 / D10（λ 校准辅助臂, fold0） | 各3 | 0.908–0.920 | 0.888 | 各 6/6 |
| **B（F2-bidir 跨折 f1–2）** | 6 | **0.9735 [0.949–0.998]** | 0.847 | 10/12；2/2 折 ≥2/3 种子 <1 |
| V-top1（=B 复现） | 6 | 0.9735 [0.949–0.998] | 0.847 | 10/12 |
| V-top2（F3 跨折） | 6 | 1.039 [1.008–1.068] | 0.624 | 3/12 |
| F0/F1/F2/F2-intent-off/F3（forward-only, fold0） | 各3 | 1.107–1.225 | 0.31–0.52 | ≈0/6 |

λ 校准（encoder 梯度占比探针，λ=0 与基线逐比特一致）：S2=0.01906 / S10=0.09528 / D2=2.11e-05 / D10=1.05e-04；全部 verdict=calibrated，无 scale_unreachable；L 臂护栏 12/12 绿（RMSE −1.1%..+1.8%、sign 下降 0、dead-zone 上升 ≤0）。

## 4. 科学结论（按证据强度排序）

1. **双向 intent 监督是本任务唯一有效的学习信号**：F2-bidir 三种子一致优于零步且跨折保持；全部 forward-only 臂（含完整 F3 槽位）均劣于零步。意图信息不是免费先验——只有在正反双向任务结构下才转化为位移预测收益。
2. **泛化衰减真实存在**：fold0 0.910 → folds1-2 0.974（CI 上界 0.998）；fold2 seed17 单点 1.0188。49 簇小样本下 5–9% 的跨折衰减需在确认折检验——本战役按授权未开确认折。
3. **早期过拟合**：43/48 run 的 inner-val 最优选择落在 ≤250/3000 updates。冻结日程未调参；下一战役可考虑缩短主日程（属超参修订，需新授权）。
4. **预注册选择规则得到跨折支持**：Top-1=F2-bidir、Top-2=F3（V-top2 1.039 平价偏弱，排序成立）。
5. **R3 终局状态：dev-positive / confirm-blocked**——scale_controlled 不可主张（确认折未开、指标封存未解）。

## 5. 封存与合规

- folds3/4 六个 B runs：指标文件存在但被 id 级排除于所有聚合；metric_unseal 一次性消费（count=1，6 收据）仅用于终态化，未揭示数值
- C6 + QT-confirm 6：BLOCKED（confirm_awaiting_*）；R1/R2 六臂：BLOCKED（spec-gap：冻结文档未定义 ramp 参数）
- 真值访问：全程零直接读取（truth_access_log 空，T0.5 pass）

## 6. 明日决策接口

见 NEXT_DAY_DECISION_MEMO（R1/R2 spec、confirm 折授权、日程缩短、342 对清单缺失等）。
