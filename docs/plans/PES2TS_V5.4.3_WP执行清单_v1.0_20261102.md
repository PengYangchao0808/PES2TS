# PES2TS V5.4.3 WP 执行清单

> 状态：**Frozen-Preregistered v1.0**（2026-11-02；D1–D8 已签认，配套主方案 `PES2TS_V5.4.3_双Decoder_50K_全面开发与训练方案_v1.0_20261102.md`）
> 用途：每日执行/周会核对；不替代主方案，不新增规范
> 规则：上一 WP 未过验收，不得进入依赖它的正式实验；失败/跳过/资源终止一律入账，不删除

---

## 0. 签认（WP0 前必须完成）

- [x] D1 接受 F1–F8 八处修正
- [x] D2 新 fixed-coverage direction@3 定义
- [x] D3 q MAE/ordering 为主、q_corr 为诊断
- [x] D4 同 Track 36K ±5% 参数匹配
- [x] D5 MOVE 损失沿用 V542（不含 strong/dead）
- [x] D6 37,900 flat + 400 warmup
- [x] D7 18-run 矩阵与两 seed 筛选 + 三 seed 终审
- [x] D8 T7 仅 dev 一次性、confirm 永封

签认人：仓库所有者 / DSH 编排会话  日期：2026-11-02  版本：v1.0

> 签认效力：D1–D8 全部采用建议值；三条运行授权同日给出（① WP3 go/no-go 等治理门由编排 agent 内部核验后自主通过，重大异常或需签认事项暂停上报；② 每条 Track 短程 smoke 通过后直接排队启动正式 run；③ confirm 永封、QC=0、dev 仅 T7 一次性访问、`strict=False` 禁用、禁 `torch.compile`）。治理落地见 [ADR-0034](../design/decisions/ADR-0034-V5.4.3-双Decoder与DecoderContextV1.md)，主方案状态转 `Frozen-Preregistered`。

---

## 1. WP 执行表

| WP | 任务 | 状态 | 开始 | 完成 | 验收证据 | 失败动作 |
|---|---|---|---|---|---|---|
| WP0 | 预注册/治理 | ☐ | | | `--prereg` 绿、docs gate CLEAN、registry proposed、signoff | 不进入实现 |
| WP1 | A0 迁移/parity | ☐ | | | 三 seed parity 证书、`tanh(q0_raw)==q`、hook=1 | 阻断全部训练 |
| WP2 | Bridge/ContextV1 | ☐ | | | schema SHA、无泄漏、pair 审计、E(3)、rank | 回炉，不加深 Decoder |
| WP3 | Probe/Basis | ☐ | | | S0-B/C 报告、go/no-go | 回炉 Bridge 或 S2 |
| WP4 | Registry/Trainer/Cache | ☐ | | | 注册/构造/训练/resume/gradient isolation | `invalid_run` |
| WP5 | M1–M3 | ☐ | | | 参数带、smoke、梯度、显存、等变、零残差 | 修复/重设计 |
| WP6 | P1–P4（+P4-C） | ☐ | | | 同 token、mask、置换、进度/端点、参数匹配 | 修复/重设计 |
| WP7 | smoke/profile/diagnostic | ☐ | | | 7 候选账本、H0/H1、成本比、预算签认 | `gate_skipped`/`resource_failure` |
| WP8 | 正式 T0–T6 | ☐ | | | 18 run 账本、H2/H3、Holm、两冠军/留任 | 负结果如实保留 |
| WP9 | 集成/归档 | ☐ | | | T7、bundle、五报告、registry executed | 阻断交付 |

---

## 2. 每周检查点

### W1（WP0–WP1）

- [ ] ADR-0034 草案完成
- [ ] campaign YAML + validator 骨架
- [ ] 五模板 + registry `proposed`
- [ ] A0 strict load 三 seed 成功
- [ ] `u0/z/q/e0` parity（CPU bitwise）
- [ ] `tanh(q0_raw)==q`
- [ ] hook 计数 = 1
- [ ] `test_v543_zero_residual_parity` 绿

### W2（WP2–WP3）

- [ ] `DecoderContextV1` 字段/单位/对称性冻结
- [ ] `key_v` 8 通道 + `pair_s` 48 维 + key graph + token layout
- [ ] 无监督泄漏哨兵测试
- [ ] pair 集合审计（10K 帧）
- [ ] E(3)/置换/reverse 测试
- [ ] S0-B probe 报告
- [ ] S0-C basis 报告
- [ ] go/no-go 记录

### W3（WP4）

- [ ] Registry 可注册/构造/描述
- [ ] 单 Track trainer 可跑 200 updates
- [ ] gradient isolation（两 Track）
- [ ] exact resume（50+50 vs 100）
- [ ] cache identity/拒绝逻辑
- [ ] 参数精确断言

### W4（WP5–WP6）

- [ ] M1/M2/M3 构造 + 参数带
- [ ] P1/P2/P3/P4/P4-C 构造 + 参数带
- [ ] 零残差 parity
- [ ] smoke 200 ×7
- [ ] 等变/置换/有限梯度
- [ ] 无 NaN/Inf

### W5（WP7 + T0–T3）

- [ ] profile 40+40 ×7
- [ ] diagnostic 2,000 ×7
- [ ] H0/H1 全过
- [ ] 成本比 ≤2.0
- [ ] T0 M0/P0 基线文档冻结
- [ ] T3 seed17 7 runs done
- [ ] run ledger append-only

### W6（T4–T6 + T7）

- [ ] T4 seed29 7 runs done
- [ ] screen top2 per Track
- [ ] T5 seed43 ≤4 runs
- [ ] H2 逐 seed
- [ ] H3/Holm/CI
- [ ] 两冠军或留任
- [ ] T7 B0/BM/BP/BMP
- [ ] dev 访问账本

### W7（WP9 + 缓冲）

- [ ] bundle 构建
- [ ] strict-load/read-only 验证
- [ ] 五份报告
- [ ] registry executed
- [ ] V5.4.4 交接
- [ ] 全仓 pytest / docs gate

---

## 3. 18-run 追踪表

| # | Run ID | Track | Decoder | Seed | 状态 | inner-val | H2 | 备注 |
|---:|---|---|---|---:|---|---:|---|---|
| 1 | V543_M1_S17 | MOVE | M1 | 17 | ☐ | | | |
| 2 | V543_M2_S17 | MOVE | M2 | 17 | ☐ | | | |
| 3 | V543_M3_S17 | MOVE | M3 | 17 | ☐ | | | |
| 4 | V543_P1_S17 | PROCESS | P1 | 17 | ☐ | | | |
| 5 | V543_P2_S17 | PROCESS | P2 | 17 | ☐ | | | |
| 6 | V543_P3_S17 | PROCESS | P3 | 17 | ☐ | | | |
| 7 | V543_P4_S17 | PROCESS | P4 | 17 | ☐ | | | |
| 8 | V543_M1_S29 | MOVE | M1 | 29 | ☐ | | | |
| 9 | V543_M2_S29 | MOVE | M2 | 29 | ☐ | | | |
| 10 | V543_M3_S29 | MOVE | M3 | 29 | ☐ | | | |
| 11 | V543_P1_S29 | PROCESS | P1 | 29 | ☐ | | | |
| 12 | V543_P2_S29 | PROCESS | P2 | 29 | ☐ | | | |
| 13 | V543_P3_S29 | PROCESS | P3 | 29 | ☐ | | | |
| 14 | V543_P4_S29 | PROCESS | P4 | 29 | ☐ | | | |
| 15 | V543_T5_M1_S43 | MOVE | top1 | 43 | ☐ | | | |
| 16 | V543_T5_M2_S43 | MOVE | top2 | 43 | ☐ | | | |
| 17 | V543_T5_P1_S43 | PROCESS | top1 | 43 | ☐ | | | |
| 18 | V543_T5_P2_S43 | PROCESS | top2 | 43 | ☐ | | | |

状态词：`pending / running / done / invalid_run / scientific_failure / resource_failure / budget_censored / gate_skipped`。

---

## 4. 门检查清单

### H0（合法性）

- [ ] forward-view 白名单
- [ ] 无监督泄漏
- [ ] 平移不变
- [ ] 旋转/反射等变
- [ ] 原子/边/对置换等变
- [ ] reverse intent
- [ ] 非关键原子零位移
- [ ] typed-invalid 拒绝
- [ ] restore/resume
- [ ] 有限梯度/loss
- [ ] Encoder forward = 1
- [ ] 零残差 parity

### H1（资源）

- [ ] 单路
- [ ] 无 swap
- [ ] 无 OOM
- [ ] host ≤12 GB
- [ ] VRAM 不超线
- [ ] core-time ratio ≤2.0
- [ ] p95 latency ≤2× A0
- [ ] 参数在 34.2–37.8K

### H2（逐 seed 非劣）

MOVE：

- [ ] RMSE/zero 退化 ≤2%
- [ ] sign 下降 ≤0.01
- [ ] dead 增加 ≤0.01
- [ ] fixed-coverage 退化 ≤2%

PROCESS：

- [ ] AUPRC 下降 ≤0.01
- [ ] recall@FPR0.05 下降 ≤0.02
- [ ] q MAE 退化 ≤5%
- [ ] ordering 下降 ≤0.02
- [ ] ECE 增加 ≤0.02

### H3（效应 + 统计）

- [ ] MOVE：RMSE ≥3% 或 direction ≥3%+0.01（另一项非劣）
- [ ] PROCESS：AUPRC +0.02 或 q MAE ≥5%+ordering 非劣（AUPRC/recall/calibration 非劣）
- [ ] 三 seed 中 ≥2 同向
- [ ] 每 seed H2 无 breach
- [ ] paired cluster bootstrap 10,000；CI 下界 >0
- [ ] Holm 校正
- [ ] 未校正 + 校正后同报

---

## 5. 冻结/禁止清单

- [ ] 未修改 V542/V541 源文件与报告
- [ ] 未修改旧 run/metrics
- [ ] 未读取 confirm
- [ ] 未调用 QC（`new_QC_calls=0`）
- [ ] 未用 dev 选择/调参/调阈值
- [ ] 未换 seed/减剂量/减帧/减层
- [ ] 未用 `torch.compile`
- [ ] 未在 Decoder 内读 `pos`/监督
- [ ] 未用 `strict=False`
- [ ] 缓存未含标签/dev/confirm
- [ ] No-Move 未用 q 触发
- [ ] 未宣称 verified TS/生产 No-Move/部署

---

## 6. 交付物清单

- [ ] `PES2TSBundleV543`（encoder/bridge/move/process/controller/provenance）
- [ ] `encoder_bridge_parity` 报告
- [ ] `move_benchmark` 报告
- [ ] `process_benchmark` 报告
- [ ] `integration_rollout` 报告
- [ ] `final_selection` 报告
- [ ] `run_ledger.jsonl` / `run_records.jsonl` / `metrics.json`
- [ ] `h2_report.json` / `h3_report.json` / `activation_decisions.json`
- [ ] registry `V5.4.3` → `executed`
- [ ] V5.4.4 交接（DecoderContextV1 + Registry + 失败/删失清单）
