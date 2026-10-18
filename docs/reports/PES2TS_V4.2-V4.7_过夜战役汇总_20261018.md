# PES2TS V4.2–V4.7 过夜战役汇总（草稿）

> 状态：草稿（自动聚合渲染；最终结论待人工终审）
> 生成时间：2026-10-18T08:51:02.061646+00:00
> 报告只读纪律：本文件为新建文件，未改动任何历史报告。收益未预填，负结果如实记录。

## 1. 训练臂与预算

| arm | preset | seed | updates | steps·s⁻¹ | peak_mem_mb | wall_s | status |
|---|---|---|---|---|---|---|---|
| D0 | legacy_v41 | 20261017 | 20000 | 36.52 | 66 | 547.6 | completed |
| D0 | legacy_v41 | 20261018 | 20000 | 35.99 | 66 | 555.7 | completed |
| D1 | legacy_v41 | 20261017 | 20000 | 36.86 | 66 | 542.6 | completed |
| D1 | legacy_v41 | 20261018 | 20000 | 34.69 | 66 | 576.5 | completed |
| D2 | campaign_v1 | 20261017 | 20000 | 48.98 | 65 | 408.3 | completed |
| D2 | campaign_v1 | 20261018 | 20000 | 43.85 | 65 | 456.1 | completed |
| D3 | campaign_v1 | 20261017 | 20000 | 39.24 | 65 | 509.7 | completed |
| D3 | campaign_v1 | 20261018 | 20000 | 42.25 | 65 | 473.4 | completed |
| E1 | campaign_v1 | 20261017 | 20000 | 40.72 | 65 | 491.2 | completed |
| E1 | campaign_v1 | 20261018 | 20000 | 40.50 | 65 | 493.9 | completed |
| E2 | campaign_v1 | 20261017 | 20000 | 34.37 | 66 | 581.9 | completed |
| E2 | campaign_v1 | 20261018 | 20000 | 35.56 | 66 | 562.5 | completed |
| FINAL | campaign_v1 | 20261017 | 20000 | 41.17 | 65 | 485.8 | completed |
| FINAL | campaign_v1 | 20261018 | 20000 | 37.97 | 65 | 526.8 | completed |
| FINAL | campaign_v1 | 20261019 | 20000 | 39.68 | 65 | 504.1 | completed |
| H2 | campaign_v1 | 20261017 | 20000 | 35.53 | 65 | 562.8 | completed |
| H2 | campaign_v1 | 20261018 | 20000 | 34.22 | 65 | 584.5 | completed |
| V1 | campaign_v1 | 20261017 | 20000 | 27.90 | 76 | 717.0 | completed |
| V1 | campaign_v1 | 20261018 | 20000 | 28.88 | 76 | 692.6 | completed |
| W128 | campaign_v1 | 20261017 | 20000 | 39.87 | 103 | 501.6 | completed |
| W128 | campaign_v1 | 20261018 | 20000 | 40.82 | 103 | 490.0 | completed |
| W96 | campaign_v1 | 20261017 | 20000 | 38.08 | 83 | 525.2 | completed |
| W96 | campaign_v1 | 20261018 | 20000 | 38.46 | 83 | 520.0 | completed |

## 2. 三口径协议评测

| arm | protocol | 组中位 MSE 比值 | atom-pooled MSE | 整体 cos | near-TS cos | 四门 |
|---|---|---|---|---|---|---|
| D0_seed20261017 | common_unchanged | 1.038 | 0.001 | 0.133 | 0.299 | NOT_GATED |
| D0_seed20261017 | legacy_full | 1.052 | 0.001 | 0.115 | 0.300 | BELOW_GATE |
| D0_seed20261017 | native_full | 1.031 | 0.001 | 0.108 | 0.303 | BELOW_GATE |
| D0_seed20261018 | common_unchanged | 1.022 | 0.001 | 0.149 | 0.280 | NOT_GATED |
| D0_seed20261018 | legacy_full | 1.034 | 0.001 | 0.147 | 0.282 | BELOW_GATE |
| D0_seed20261018 | native_full | 1.013 | 0.001 | 0.143 | 0.284 | BELOW_GATE |
| D1_seed20261017 | common_unchanged | 0.982 | 0.001 | 0.169 | 0.274 | NOT_GATED |
| D1_seed20261017 | legacy_full | 0.988 | 0.001 | 0.153 | 0.274 | BELOW_GATE |
| D1_seed20261017 | native_full | 0.977 | 0.001 | 0.151 | 0.277 | BELOW_GATE |
| D1_seed20261018 | common_unchanged | 0.976 | 0.001 | 0.144 | 0.236 | NOT_GATED |
| D1_seed20261018 | legacy_full | 0.986 | 0.001 | 0.127 | 0.237 | BELOW_GATE |
| D1_seed20261018 | native_full | 0.978 | 0.001 | 0.121 | 0.241 | BELOW_GATE |
| D2_seed20261017 | common_unchanged | 1.120 | 0.001 | 0.196 | 0.205 | NOT_GATED |
| D2_seed20261017 | legacy_full | 1.136 | 0.001 | 0.184 | 0.208 | BELOW_GATE |
| D2_seed20261017 | native_full | 1.101 | 0.001 | 0.184 | 0.212 | BELOW_GATE |
| D2_seed20261018 | common_unchanged | 1.168 | 0.001 | 0.165 | 0.173 | NOT_GATED |
| D2_seed20261018 | legacy_full | 1.229 | 0.001 | 0.148 | 0.209 | BELOW_GATE |
| D2_seed20261018 | native_full | 1.105 | 0.001 | 0.154 | 0.209 | BELOW_GATE |
| D3_seed20261017 | common_unchanged | 1.191 | 0.001 | 0.191 | 0.185 | NOT_GATED |
| D3_seed20261017 | legacy_full | 1.207 | 0.001 | 0.181 | 0.262 | BELOW_GATE |
| D3_seed20261017 | native_full | 1.152 | 0.001 | 0.179 | 0.262 | BELOW_GATE |
| D3_seed20261018 | common_unchanged | 1.163 | 0.001 | 0.138 | 0.306 | NOT_GATED |
| D3_seed20261018 | legacy_full | 1.188 | 0.001 | 0.176 | 0.332 | BELOW_GATE |
| D3_seed20261018 | native_full | 1.082 | 0.001 | 0.178 | 0.332 | BELOW_GATE |
| E1_seed20261017 | common_unchanged | 1.128 | 0.001 | 0.228 | 0.204 | NOT_GATED |
| E1_seed20261017 | legacy_full | 1.183 | 0.001 | 0.224 | 0.204 | BELOW_GATE |
| E1_seed20261017 | native_full | 1.109 | 0.001 | 0.224 | 0.204 | BELOW_GATE |
| E1_seed20261018 | common_unchanged | 1.179 | 0.001 | 0.247 | 0.249 | NOT_GATED |
| E1_seed20261018 | legacy_full | 1.208 | 0.001 | 0.251 | 0.281 | BELOW_GATE |
| E1_seed20261018 | native_full | 1.077 | 0.001 | 0.256 | 0.323 | BELOW_GATE |
| E2_seed20261017 | common_unchanged | — | — | — | — | 跳过(MODEL_LOAD_FAILED) |
| E2_seed20261017 | legacy_full | — | — | — | — | 跳过(MODEL_LOAD_FAILED) |
| E2_seed20261017 | native_full | — | — | — | — | 跳过(MODEL_LOAD_FAILED) |
| E2_seed20261018 | common_unchanged | — | — | — | — | 跳过(MODEL_LOAD_FAILED) |
| E2_seed20261018 | legacy_full | — | — | — | — | 跳过(MODEL_LOAD_FAILED) |
| E2_seed20261018 | native_full | — | — | — | — | 跳过(MODEL_LOAD_FAILED) |
| FINAL_seed20261017 | common_unchanged | 1.162 | 0.001 | 0.228 | 0.181 | NOT_GATED |
| FINAL_seed20261017 | legacy_full | 1.177 | 0.001 | 0.222 | 0.211 | BELOW_GATE |
| FINAL_seed20261017 | native_full | 1.109 | 0.001 | 0.221 | 0.212 | BELOW_GATE |
| FINAL_seed20261018 | common_unchanged | 1.173 | 0.001 | 0.183 | 0.276 | NOT_GATED |
| FINAL_seed20261018 | legacy_full | 1.240 | 0.001 | 0.167 | 0.347 | BELOW_GATE |
| FINAL_seed20261018 | native_full | 1.146 | 0.001 | 0.169 | 0.337 | BELOW_GATE |
| FINAL_seed20261019 | common_unchanged | 1.141 | 0.001 | 0.188 | 0.248 | NOT_GATED |
| FINAL_seed20261019 | legacy_full | 1.149 | 0.001 | 0.177 | 0.248 | BELOW_GATE |
| FINAL_seed20261019 | native_full | 1.132 | 0.001 | 0.180 | 0.248 | BELOW_GATE |
| H2_seed20261017 | common_unchanged | 1.998 | 0.001 | 0.198 | 0.257 | NOT_GATED |
| H2_seed20261017 | legacy_full | 2.110 | 0.001 | 0.184 | 0.254 | BELOW_GATE |
| H2_seed20261017 | native_full | 1.720 | 0.001 | 0.183 | 0.256 | BELOW_GATE |
| H2_seed20261018 | common_unchanged | 1.738 | 0.001 | 0.185 | 0.244 | NOT_GATED |
| H2_seed20261018 | legacy_full | 1.899 | 0.001 | 0.179 | 0.243 | BELOW_GATE |
| H2_seed20261018 | native_full | 1.651 | 0.001 | 0.180 | 0.247 | BELOW_GATE |
| V1_seed20261017 | common_unchanged | 1.516 | 0.001 | 0.255 | 0.239 | NOT_GATED |
| V1_seed20261017 | legacy_full | 1.679 | 0.001 | 0.242 | 0.247 | BELOW_GATE |
| V1_seed20261017 | native_full | 1.420 | 0.001 | 0.242 | 0.249 | BELOW_GATE |
| V1_seed20261018 | common_unchanged | 1.640 | 0.001 | 0.220 | 0.174 | NOT_GATED |
| V1_seed20261018 | legacy_full | 1.660 | 0.001 | 0.204 | 0.184 | BELOW_GATE |
| V1_seed20261018 | native_full | 1.576 | 0.001 | 0.206 | 0.183 | BELOW_GATE |
| W128_seed20261017 | common_unchanged | 1.485 | 0.001 | 0.165 | 0.123 | NOT_GATED |
| W128_seed20261017 | legacy_full | 1.491 | 0.001 | 0.157 | 0.167 | BELOW_GATE |
| W128_seed20261017 | native_full | 1.366 | 0.001 | 0.158 | 0.167 | BELOW_GATE |
| W128_seed20261018 | common_unchanged | 1.416 | 0.001 | 0.115 | 0.146 | NOT_GATED |
| W128_seed20261018 | legacy_full | 1.350 | 0.001 | 0.116 | 0.152 | BELOW_GATE |
| W128_seed20261018 | native_full | 1.274 | 0.001 | 0.116 | 0.159 | BELOW_GATE |
| W96_seed20261017 | common_unchanged | 1.271 | 0.001 | 0.167 | 0.105 | NOT_GATED |
| W96_seed20261017 | legacy_full | 1.363 | 0.001 | 0.160 | 0.156 | BELOW_GATE |
| W96_seed20261017 | native_full | 1.182 | 0.001 | 0.158 | 0.149 | BELOW_GATE |
| W96_seed20261018 | common_unchanged | 1.354 | 0.001 | 0.179 | 0.293 | NOT_GATED |
| W96_seed20261018 | legacy_full | 1.432 | 0.001 | 0.172 | 0.291 | BELOW_GATE |
| W96_seed20261018 | native_full | 1.276 | 0.001 | 0.168 | 0.291 | BELOW_GATE |

## 3. 四项门逐条

门为 legacy_full 原四门（median ratio < 1.0、≥6/8 组、整体 cos ≥ 0.30、near-TS cos ≥ 0.40）；native_full 同式但 `new_protocol=True`；common_unchanged 不套门。

- **D0_seed20261017 / legacy_full**: BELOW_GATE
  - median_mse_ratio: 1.052 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 3.000 vs 6.000 → 未通过
  - median_cosine: 0.115 vs 0.300 → 未通过
  - near_ts_cosine: 0.300 vs 0.400 → 未通过
- **D0_seed20261017 / native_full**: BELOW_GATE
  - median_mse_ratio: 1.031 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 3.000 vs 6.000 → 未通过
  - median_cosine: 0.108 vs 0.300 → 未通过
  - near_ts_cosine: 0.303 vs 0.400 → 未通过
- **D0_seed20261018 / legacy_full**: BELOW_GATE
  - median_mse_ratio: 1.034 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 2.000 vs 6.000 → 未通过
  - median_cosine: 0.147 vs 0.300 → 未通过
  - near_ts_cosine: 0.282 vs 0.400 → 未通过
- **D0_seed20261018 / native_full**: BELOW_GATE
  - median_mse_ratio: 1.013 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 2.000 vs 6.000 → 未通过
  - median_cosine: 0.143 vs 0.300 → 未通过
  - near_ts_cosine: 0.284 vs 0.400 → 未通过
- **D1_seed20261017 / legacy_full**: BELOW_GATE
  - median_mse_ratio: 0.988 vs 1.000 → 通过
  - n_groups_ratio_below_one: 4.000 vs 6.000 → 未通过
  - median_cosine: 0.153 vs 0.300 → 未通过
  - near_ts_cosine: 0.274 vs 0.400 → 未通过
- **D1_seed20261017 / native_full**: BELOW_GATE
  - median_mse_ratio: 0.977 vs 1.000 → 通过
  - n_groups_ratio_below_one: 5.000 vs 6.000 → 未通过
  - median_cosine: 0.151 vs 0.300 → 未通过
  - near_ts_cosine: 0.277 vs 0.400 → 未通过
- **D1_seed20261018 / legacy_full**: BELOW_GATE
  - median_mse_ratio: 0.986 vs 1.000 → 通过
  - n_groups_ratio_below_one: 4.000 vs 6.000 → 未通过
  - median_cosine: 0.127 vs 0.300 → 未通过
  - near_ts_cosine: 0.237 vs 0.400 → 未通过
- **D1_seed20261018 / native_full**: BELOW_GATE
  - median_mse_ratio: 0.978 vs 1.000 → 通过
  - n_groups_ratio_below_one: 4.000 vs 6.000 → 未通过
  - median_cosine: 0.121 vs 0.300 → 未通过
  - near_ts_cosine: 0.241 vs 0.400 → 未通过
- **D2_seed20261017 / legacy_full**: BELOW_GATE
  - median_mse_ratio: 1.136 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 1.000 vs 6.000 → 未通过
  - median_cosine: 0.184 vs 0.300 → 未通过
  - near_ts_cosine: 0.208 vs 0.400 → 未通过
- **D2_seed20261017 / native_full**: BELOW_GATE
  - median_mse_ratio: 1.101 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 1.000 vs 6.000 → 未通过
  - median_cosine: 0.184 vs 0.300 → 未通过
  - near_ts_cosine: 0.212 vs 0.400 → 未通过
- **D2_seed20261018 / legacy_full**: BELOW_GATE
  - median_mse_ratio: 1.229 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 0.000 vs 6.000 → 未通过
  - median_cosine: 0.148 vs 0.300 → 未通过
  - near_ts_cosine: 0.209 vs 0.400 → 未通过
- **D2_seed20261018 / native_full**: BELOW_GATE
  - median_mse_ratio: 1.105 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 0.000 vs 6.000 → 未通过
  - median_cosine: 0.154 vs 0.300 → 未通过
  - near_ts_cosine: 0.209 vs 0.400 → 未通过
- **D3_seed20261017 / legacy_full**: BELOW_GATE
  - median_mse_ratio: 1.207 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 1.000 vs 6.000 → 未通过
  - median_cosine: 0.181 vs 0.300 → 未通过
  - near_ts_cosine: 0.262 vs 0.400 → 未通过
- **D3_seed20261017 / native_full**: BELOW_GATE
  - median_mse_ratio: 1.152 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 1.000 vs 6.000 → 未通过
  - median_cosine: 0.179 vs 0.300 → 未通过
  - near_ts_cosine: 0.262 vs 0.400 → 未通过
- **D3_seed20261018 / legacy_full**: BELOW_GATE
  - median_mse_ratio: 1.188 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 2.000 vs 6.000 → 未通过
  - median_cosine: 0.176 vs 0.300 → 未通过
  - near_ts_cosine: 0.332 vs 0.400 → 未通过
- **D3_seed20261018 / native_full**: BELOW_GATE
  - median_mse_ratio: 1.082 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 2.000 vs 6.000 → 未通过
  - median_cosine: 0.178 vs 0.300 → 未通过
  - near_ts_cosine: 0.332 vs 0.400 → 未通过
- **E1_seed20261017 / legacy_full**: BELOW_GATE
  - median_mse_ratio: 1.183 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 1.000 vs 6.000 → 未通过
  - median_cosine: 0.224 vs 0.300 → 未通过
  - near_ts_cosine: 0.204 vs 0.400 → 未通过
- **E1_seed20261017 / native_full**: BELOW_GATE
  - median_mse_ratio: 1.109 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 2.000 vs 6.000 → 未通过
  - median_cosine: 0.224 vs 0.300 → 未通过
  - near_ts_cosine: 0.204 vs 0.400 → 未通过
- **E1_seed20261018 / legacy_full**: BELOW_GATE
  - median_mse_ratio: 1.208 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 2.000 vs 6.000 → 未通过
  - median_cosine: 0.251 vs 0.300 → 未通过
  - near_ts_cosine: 0.281 vs 0.400 → 未通过
- **E1_seed20261018 / native_full**: BELOW_GATE
  - median_mse_ratio: 1.077 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 2.000 vs 6.000 → 未通过
  - median_cosine: 0.256 vs 0.300 → 未通过
  - near_ts_cosine: 0.323 vs 0.400 → 未通过
- **FINAL_seed20261017 / legacy_full**: BELOW_GATE
  - median_mse_ratio: 1.177 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 1.000 vs 6.000 → 未通过
  - median_cosine: 0.222 vs 0.300 → 未通过
  - near_ts_cosine: 0.211 vs 0.400 → 未通过
- **FINAL_seed20261017 / native_full**: BELOW_GATE
  - median_mse_ratio: 1.109 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 2.000 vs 6.000 → 未通过
  - median_cosine: 0.221 vs 0.300 → 未通过
  - near_ts_cosine: 0.212 vs 0.400 → 未通过
- **FINAL_seed20261018 / legacy_full**: BELOW_GATE
  - median_mse_ratio: 1.240 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 0.000 vs 6.000 → 未通过
  - median_cosine: 0.167 vs 0.300 → 未通过
  - near_ts_cosine: 0.347 vs 0.400 → 未通过
- **FINAL_seed20261018 / native_full**: BELOW_GATE
  - median_mse_ratio: 1.146 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 0.000 vs 6.000 → 未通过
  - median_cosine: 0.169 vs 0.300 → 未通过
  - near_ts_cosine: 0.337 vs 0.400 → 未通过
- **FINAL_seed20261019 / legacy_full**: BELOW_GATE
  - median_mse_ratio: 1.149 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 0.000 vs 6.000 → 未通过
  - median_cosine: 0.177 vs 0.300 → 未通过
  - near_ts_cosine: 0.248 vs 0.400 → 未通过
- **FINAL_seed20261019 / native_full**: BELOW_GATE
  - median_mse_ratio: 1.132 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 0.000 vs 6.000 → 未通过
  - median_cosine: 0.180 vs 0.300 → 未通过
  - near_ts_cosine: 0.248 vs 0.400 → 未通过
- **H2_seed20261017 / legacy_full**: BELOW_GATE
  - median_mse_ratio: 2.110 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 0.000 vs 6.000 → 未通过
  - median_cosine: 0.184 vs 0.300 → 未通过
  - near_ts_cosine: 0.254 vs 0.400 → 未通过
- **H2_seed20261017 / native_full**: BELOW_GATE
  - median_mse_ratio: 1.720 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 0.000 vs 6.000 → 未通过
  - median_cosine: 0.183 vs 0.300 → 未通过
  - near_ts_cosine: 0.256 vs 0.400 → 未通过
- **H2_seed20261018 / legacy_full**: BELOW_GATE
  - median_mse_ratio: 1.899 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 0.000 vs 6.000 → 未通过
  - median_cosine: 0.179 vs 0.300 → 未通过
  - near_ts_cosine: 0.243 vs 0.400 → 未通过
- **H2_seed20261018 / native_full**: BELOW_GATE
  - median_mse_ratio: 1.651 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 0.000 vs 6.000 → 未通过
  - median_cosine: 0.180 vs 0.300 → 未通过
  - near_ts_cosine: 0.247 vs 0.400 → 未通过
- **V1_seed20261017 / legacy_full**: BELOW_GATE
  - median_mse_ratio: 1.679 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 0.000 vs 6.000 → 未通过
  - median_cosine: 0.242 vs 0.300 → 未通过
  - near_ts_cosine: 0.247 vs 0.400 → 未通过
- **V1_seed20261017 / native_full**: BELOW_GATE
  - median_mse_ratio: 1.420 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 0.000 vs 6.000 → 未通过
  - median_cosine: 0.242 vs 0.300 → 未通过
  - near_ts_cosine: 0.249 vs 0.400 → 未通过
- **V1_seed20261018 / legacy_full**: BELOW_GATE
  - median_mse_ratio: 1.660 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 1.000 vs 6.000 → 未通过
  - median_cosine: 0.204 vs 0.300 → 未通过
  - near_ts_cosine: 0.184 vs 0.400 → 未通过
- **V1_seed20261018 / native_full**: BELOW_GATE
  - median_mse_ratio: 1.576 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 1.000 vs 6.000 → 未通过
  - median_cosine: 0.206 vs 0.300 → 未通过
  - near_ts_cosine: 0.183 vs 0.400 → 未通过
- **W128_seed20261017 / legacy_full**: BELOW_GATE
  - median_mse_ratio: 1.491 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 0.000 vs 6.000 → 未通过
  - median_cosine: 0.157 vs 0.300 → 未通过
  - near_ts_cosine: 0.167 vs 0.400 → 未通过
- **W128_seed20261017 / native_full**: BELOW_GATE
  - median_mse_ratio: 1.366 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 0.000 vs 6.000 → 未通过
  - median_cosine: 0.158 vs 0.300 → 未通过
  - near_ts_cosine: 0.167 vs 0.400 → 未通过
- **W128_seed20261018 / legacy_full**: BELOW_GATE
  - median_mse_ratio: 1.350 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 0.000 vs 6.000 → 未通过
  - median_cosine: 0.116 vs 0.300 → 未通过
  - near_ts_cosine: 0.152 vs 0.400 → 未通过
- **W128_seed20261018 / native_full**: BELOW_GATE
  - median_mse_ratio: 1.274 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 0.000 vs 6.000 → 未通过
  - median_cosine: 0.116 vs 0.300 → 未通过
  - near_ts_cosine: 0.159 vs 0.400 → 未通过
- **W96_seed20261017 / legacy_full**: BELOW_GATE
  - median_mse_ratio: 1.363 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 0.000 vs 6.000 → 未通过
  - median_cosine: 0.160 vs 0.300 → 未通过
  - near_ts_cosine: 0.156 vs 0.400 → 未通过
- **W96_seed20261017 / native_full**: BELOW_GATE
  - median_mse_ratio: 1.182 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 0.000 vs 6.000 → 未通过
  - median_cosine: 0.158 vs 0.300 → 未通过
  - near_ts_cosine: 0.149 vs 0.400 → 未通过
- **W96_seed20261018 / legacy_full**: BELOW_GATE
  - median_mse_ratio: 1.432 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 0.000 vs 6.000 → 未通过
  - median_cosine: 0.172 vs 0.300 → 未通过
  - near_ts_cosine: 0.291 vs 0.400 → 未通过
- **W96_seed20261018 / native_full**: BELOW_GATE
  - median_mse_ratio: 1.276 vs 1.000 → 未通过
  - n_groups_ratio_below_one: 0.000 vs 6.000 → 未通过
  - median_cosine: 0.168 vs 0.300 → 未通过
  - near_ts_cosine: 0.291 vs 0.400 → 未通过

## 4. 切片指标（native_full 为主口径）

| arm | 有效方向原子 | near-zero RMSE | H | 重原子 | 长步 P90 | 长步 P99 | 分支A | 分支B | 关键对 | 旁观 |
|---|---|---|---|---|---|---|---|---|---|---|
| D0_seed20261017 | 8534 | 0.010 | 0.001 | 0.000 | 0.005 | 0.017 | 0.001 | 0.001 | 0.017 | 0.001 |
| D0_seed20261018 | 8534 | 0.010 | 0.001 | 0.000 | 0.005 | 0.017 | 0.001 | 0.001 | 0.015 | 0.001 |
| D1_seed20261017 | 8534 | 0.008 | 0.001 | 0.000 | 0.005 | 0.017 | 0.001 | 0.001 | 0.015 | 0.001 |
| D1_seed20261018 | 8534 | 0.008 | 0.001 | 0.000 | 0.005 | 0.017 | 0.001 | 0.001 | 0.015 | 0.001 |
| D2_seed20261017 | 8534 | 0.010 | 0.001 | 0.000 | 0.005 | 0.018 | 0.001 | 0.001 | 0.017 | 0.001 |
| D2_seed20261018 | 8534 | 0.009 | 0.002 | 0.000 | 0.005 | 0.018 | 0.001 | 0.001 | 0.017 | 0.001 |
| D3_seed20261017 | 8534 | 0.012 | 0.001 | 0.000 | 0.005 | 0.016 | 0.001 | 0.001 | 0.019 | 0.001 |
| D3_seed20261018 | 8534 | 0.010 | 0.002 | 0.000 | 0.005 | 0.019 | 0.001 | 0.001 | 0.015 | 0.001 |
| E1_seed20261017 | 8534 | 0.011 | 0.001 | 0.000 | 0.005 | 0.017 | 0.001 | 0.001 | 0.015 | 0.001 |
| E1_seed20261018 | 8534 | 0.010 | 0.001 | 0.000 | 0.005 | 0.017 | 0.001 | 0.001 | 0.015 | 0.001 |
| FINAL_seed20261017 | 8534 | 0.010 | 0.001 | 0.000 | 0.005 | 0.017 | 0.001 | 0.001 | 0.016 | 0.001 |
| FINAL_seed20261018 | 8534 | 0.012 | 0.001 | 0.000 | 0.005 | 0.017 | 0.001 | 0.001 | 0.016 | 0.001 |
| FINAL_seed20261019 | 8534 | 0.010 | 0.003 | 0.000 | 0.008 | 0.043 | 0.002 | 0.001 | 0.020 | 0.001 |
| H2_seed20261017 | 8534 | 0.026 | 0.002 | 0.001 | 0.005 | 0.017 | 0.001 | 0.001 | 0.024 | 0.001 |
| H2_seed20261018 | 8534 | 0.022 | 0.002 | 0.001 | 0.005 | 0.017 | 0.001 | 0.001 | 0.023 | 0.001 |
| V1_seed20261017 | 8534 | 0.014 | 0.002 | 0.000 | 0.006 | 0.021 | 0.001 | 0.001 | 0.017 | 0.001 |
| V1_seed20261018 | 8534 | 0.012 | 0.002 | 0.000 | 0.007 | 0.027 | 0.001 | 0.001 | 0.020 | 0.001 |
| W128_seed20261017 | 8534 | 0.015 | 0.002 | 0.000 | 0.006 | 0.016 | 0.001 | 0.001 | 0.017 | 0.001 |
| W128_seed20261018 | 8534 | 0.013 | 0.002 | 0.000 | 0.005 | 0.017 | 0.001 | 0.001 | 0.017 | 0.001 |
| W96_seed20261017 | 8534 | 0.014 | 0.002 | 0.000 | 0.005 | 0.017 | 0.001 | 0.001 | 0.018 | 0.001 |
| W96_seed20261018 | 8534 | 0.012 | 0.002 | 0.000 | 0.005 | 0.018 | 0.001 | 0.001 | 0.017 | 0.001 |

α 分布：近零真值显式未定义，不以小分母造极值。

## 5. 阶段晋级决策

- {"configs": {"D0": {"contact_edges": false, "dataset_kind": "legacy", "ema_read": "raw", "encoder": "egnn", "features_version": "v3", "final_fit50": false, "head": "bc_nll_legacy", "hidden": 64, "keypair_condition": false, "lambda_dir": null, "lambda_mag": null, "n_layers": 3, "preset": "legacy_v41", "sampler": "group_balanced_v41", "updates": 20000, "use_prior_skip": false, "vector_channels": 16}}}
- {"configs": {"D1": {"contact_edges": false, "dataset_kind": "legacy", "ema_read": "raw", "encoder": "egnn", "features_version": "v3", "final_fit50": false, "head": "bc_nll_legacy", "hidden": 64, "keypair_condition": false, "lambda_dir": null, "lambda_mag": null, "n_layers": 3, "preset": "legacy_v41", "sampler": "epoch_traversal_v41", "updates": 20000, "use_prior_skip": false, "vector_channels": 16}}}
- {"sampler_promotion": {"decision": "RETAIN_PARENT", "metrics_used": ["overall_cos", "near_ts_cos", "group_median_ratio", "better_than_zero_groups", "pooled_ratio", "near_zero_rmse_A"], "per_clause": {}, "per_seed": {"20261017": {"per_clause": {"better_than_zero_groups": {"child": 3, "parent": 2, "pass": true}, "group_median_ratio_increase": {"child": 0.9863320384014654, "delta": -0.04579034503702906, "parent": 1.0321223834384945, "pass": true}, "near_ts_cos_drop": {"child": 0.03358673513924148, "delta": 0.026412634397559972, "parent": 0.007174100741681507, "pass": true}, "near_zero_rmse": {"child": 0.007874499334697678, "parent": 0.008697978108103578, "pass": true}, "overall_cos_gain": {"child": 0.08401856895347456, "delta": 0.05092091594602879, "parent": 0.03309765300744578, "pass": true}, "pooled_ratio_increase": {"child": 1.011153901829979, "delta": -0.021658251311349686, "parent": 1.0328121531413288, "pass": true}}, "reasons": []}, "20261018": {"per_clause": {"better_than_zero_groups": {"child": 3, "parent": 0, "pass": true}, "group_median_ratio_increase": {"child": 0.9923692528335987, "delta": -0.0973438723666008, "parent": 1.0897131252001995, "pass": true}, "near_ts_cos_drop": {"child": 0.09611177158393468, "delta": -0.05322279234863396, "parent": 0.14933456393256864, "pass": false}, "near_zero_rmse": {"child": 0.007682494665137677, "parent": 0.010650624533251829, "pass": true}, "overall_cos_gain": {"child": 0.08980477562090329, "delta": -0.028507147018082088, "parent": 0.11831192263898538, "pass": false}, "pooled_ratio_increase": {"child": 1.0068737488333053, "delta": -0.08800024527155537, "parent": 1.0948739941048606, "pass": true}}, "reasons": ["overall_cos_gain: {'parent': 0.11831192263898538, 'child': 0.08980477562090329, 'delta': -0.028507147018082088, 'pass': False}", "near_ts_cos_drop: {'parent': 0.14933456393256864, 'child': 0.09611177158393468, 'delta': -0.05322279234863396, 'pass': False}"]}}, "promote": false, "reasons": ["seed 20261018 :: overall_cos_gain: {'parent': 0.11831192263898538, 'child': 0.08980477562090329, 'delta': -0.028507147018082088, 'pass': False}", "seed 20261018 :: near_ts_cos_drop: {'parent': 0.14933456393256864, 'child': 0.09611177158393468, 'delta': -0.05322279234863396, 'pass': False}"], "rule_note": "§9.1 engineering screening rule; thresholds are NOT significance bounds; multiple passing candidates are ordered by rank_promotion_candidates (group-median MSE ↓ → cos ↑ → params ↓)"}, "sampler_winner": "group_balanced_v41"}
- {"configs": {"D2": {"contact_edges": false, "dataset_kind": "legacy", "ema_read": "raw", "encoder": "egnn", "features_version": "v3", "final_fit50": false, "head": "displacement", "hidden": 64, "keypair_condition": false, "lambda_dir": null, "lambda_mag": null, "n_layers": 3, "preset": "campaign_v1", "sampler": "group_balanced_v41", "updates": 20000, "use_prior_skip": false, "vector_channels": 16}}}
- {"sampler_promotion": {"decision": "RETAIN_PARENT", "metrics_used": ["overall_cos", "near_ts_cos", "group_median_ratio", "better_than_zero_groups", "pooled_ratio", "near_zero_rmse_A"], "per_clause": {}, "per_seed": {"20261017": {"per_clause": {"better_than_zero_groups": {"child": 3, "parent": 2, "pass": true}, "group_median_ratio_increase": {"child": 0.9863320384014654, "delta": -0.04579034503702906, "parent": 1.0321223834384945, "pass": true}, "near_ts_cos_drop": {"child": 0.03358673513924148, "delta": 0.026412634397559972, "parent": 0.007174100741681507, "pass": true}, "near_zero_rmse": {"child": 0.007874499334697678, "parent": 0.008697978108103578, "pass": true}, "overall_cos_gain": {"child": 0.08401856895347456, "delta": 0.05092091594602879, "parent": 0.03309765300744578, "pass": true}, "pooled_ratio_increase": {"child": 1.011153901829979, "delta": -0.021658251311349686, "parent": 1.0328121531413288, "pass": true}}, "reasons": []}, "20261018": {"per_clause": {"better_than_zero_groups": {"child": 3, "parent": 0, "pass": true}, "group_median_ratio_increase": {"child": 0.9923692528335987, "delta": -0.0973438723666008, "parent": 1.0897131252001995, "pass": true}, "near_ts_cos_drop": {"child": 0.09611177158393468, "delta": -0.05322279234863396, "parent": 0.14933456393256864, "pass": false}, "near_zero_rmse": {"child": 0.007682494665137677, "parent": 0.010650624533251829, "pass": true}, "overall_cos_gain": {"child": 0.08980477562090329, "delta": -0.028507147018082088, "parent": 0.11831192263898538, "pass": false}, "pooled_ratio_increase": {"child": 1.0068737488333053, "delta": -0.08800024527155537, "parent": 1.0948739941048606, "pass": true}}, "reasons": ["overall_cos_gain: {'parent': 0.11831192263898538, 'child': 0.08980477562090329, 'delta': -0.028507147018082088, 'pass': False}", "near_ts_cos_drop: {'parent': 0.14933456393256864, 'child': 0.09611177158393468, 'delta': -0.05322279234863396, 'pass': False}"]}}, "promote": false, "reasons": ["seed 20261018 :: overall_cos_gain: {'parent': 0.11831192263898538, 'child': 0.08980477562090329, 'delta': -0.028507147018082088, 'pass': False}", "seed 20261018 :: near_ts_cos_drop: {'parent': 0.14933456393256864, 'child': 0.09611177158393468, 'delta': -0.05322279234863396, 'pass': False}"], "rule_note": "§9.1 engineering screening rule; thresholds are NOT significance bounds; multiple passing candidates are ordered by rank_promotion_candidates (group-median MSE ↓ → cos ↑ → params ↓)"}, "sampler_winner": "group_balanced_v41"}
- {"configs": {"D3": {"contact_edges": false, "dataset_kind": "native", "ema_read": "raw", "encoder": "egnn", "features_version": "v3", "final_fit50": false, "head": "displacement", "hidden": 64, "keypair_condition": false, "lambda_dir": null, "lambda_mag": null, "n_layers": 3, "preset": "campaign_v1", "sampler": "group_balanced_v41", "updates": 20000, "use_prior_skip": false, "vector_channels": 16}}}
- {"arm": "H1", "decision": "prior_not_supported", "source": "b65eb2b904964a63794406e0e49b134eb60b330f31253d24945161fd5186d5aa"}
- {"sampler_promotion": {"decision": "RETAIN_PARENT", "metrics_used": ["overall_cos", "near_ts_cos", "group_median_ratio", "better_than_zero_groups", "pooled_ratio", "near_zero_rmse_A"], "per_clause": {}, "per_seed": {"20261017": {"per_clause": {"better_than_zero_groups": {"child": 3, "parent": 2, "pass": true}, "group_median_ratio_increase": {"child": 0.9863320384014654, "delta": -0.04579034503702906, "parent": 1.0321223834384945, "pass": true}, "near_ts_cos_drop": {"child": 0.03358673513924148, "delta": 0.026412634397559972, "parent": 0.007174100741681507, "pass": true}, "near_zero_rmse": {"child": 0.007874499334697678, "parent": 0.008697978108103578, "pass": true}, "overall_cos_gain": {"child": 0.08401856895347456, "delta": 0.05092091594602879, "parent": 0.03309765300744578, "pass": true}, "pooled_ratio_increase": {"child": 1.011153901829979, "delta": -0.021658251311349686, "parent": 1.0328121531413288, "pass": true}}, "reasons": []}, "20261018": {"per_clause": {"better_than_zero_groups": {"child": 3, "parent": 0, "pass": true}, "group_median_ratio_increase": {"child": 0.9923692528335987, "delta": -0.0973438723666008, "parent": 1.0897131252001995, "pass": true}, "near_ts_cos_drop": {"child": 0.09611177158393468, "delta": -0.05322279234863396, "parent": 0.14933456393256864, "pass": false}, "near_zero_rmse": {"child": 0.007682494665137677, "parent": 0.010650624533251829, "pass": true}, "overall_cos_gain": {"child": 0.08980477562090329, "delta": -0.028507147018082088, "parent": 0.11831192263898538, "pass": false}, "pooled_ratio_increase": {"child": 1.0068737488333053, "delta": -0.08800024527155537, "parent": 1.0948739941048606, "pass": true}}, "reasons": ["overall_cos_gain: {'parent': 0.11831192263898538, 'child': 0.08980477562090329, 'delta': -0.028507147018082088, 'pass': False}", "near_ts_cos_drop: {'parent': 0.14933456393256864, 'child': 0.09611177158393468, 'delta': -0.05322279234863396, 'pass': False}"]}}, "promote": false, "reasons": ["seed 20261018 :: overall_cos_gain: {'parent': 0.11831192263898538, 'child': 0.08980477562090329, 'delta': -0.028507147018082088, 'pass': False}", "seed 20261018 :: near_ts_cos_drop: {'parent': 0.14933456393256864, 'child': 0.09611177158393468, 'delta': -0.05322279234863396, 'pass': False}"], "rule_note": "§9.1 engineering screening rule; thresholds are NOT significance bounds; multiple passing candidates are ordered by rank_promotion_candidates (group-median MSE ↓ → cos ↑ → params ↓)"}, "sampler_winner": "group_balanced_v41"}
- {"arm": "H1", "decision": "prior_not_supported", "source": "b65eb2b904964a63794406e0e49b134eb60b330f31253d24945161fd5186d5aa"}
- {"configs": {"H0": {"contact_edges": false, "dataset_kind": "native", "ema_read": "raw", "encoder": "egnn", "features_version": "v3", "final_fit50": false, "head": "displacement", "hidden": 64, "keypair_condition": false, "lambda_dir": null, "lambda_mag": null, "n_layers": 3, "preset": "campaign_v1", "sampler": "group_balanced_v41", "updates": 20000, "use_prior_skip": false, "vector_channels": 16}}, "promotion": {"decision": "RETAIN_PARENT", "metrics_used": ["overall_cos", "near_ts_cos", "group_median_ratio", "better_than_zero_groups", "pooled_ratio", "near_zero_rmse_A"], "per_clause": {}, "per_seed": {"20261017": {"per_clause": {"better_than_zero_groups": {"child": 0, "parent": 2, "pass": false}, "group_median_ratio_increase": {"child": 1.7498553846626543, "delta": 0.7235636193748884, "parent": 1.026291765287766, "pass": false}, "near_ts_cos_drop": {"child": 0.15748327153830693, "delta": -0.039334357602738124, "parent": 0.19681762914104506, "pass": false}, "near_zero_rmse": {"child": 0.019443800568523042, "parent": 0.009229259819429398, "pass": false}, "overall_cos_gain": {"child": 0.13571909295238466, "delta": -0.07066667860120987, "parent": 0.20638577155359453, "pass": false}, "pooled_ratio_increase": {"child": 1.7693663771342016, "delta": 0.74120589491723, "parent": 1.0281604822169716, "pass": false}}, "reasons": ["overall_cos_gain: {'parent': 0.20638577155359453, 'child': 0.13571909295238466, 'delta': -0.07066667860120987, 'pass': False}", "near_ts_cos_drop: {'parent': 0.19681762914104506, 'child': 0.15748327153830693, 'delta': -0.039334357602738124, 'pass': False}", "group_median_ratio_increase: {'parent': 1.026291765287766, 'child': 1.7498553846626543, 'delta': 0.7235636193748884, 'pass': False}", "better_than_zero_groups: {'parent': 2, 'child': 0, 'pass': False}", "pooled_ratio_increase: {'parent': 1.0281604822169716, 'child': 1.7693663771342016, 'delta': 0.74120589491723, 'pass': False}", "near_zero_rmse: {'parent': 0.009229259819429398, 'child': 0.019443800568523042, 'pass': False}"]}, "20261018": {"per_clause": {"better_than_zero_groups": {"child": 0, "parent": 4, "pass": false}, "group_median_ratio_increase": {"child": 2.3209965607627576, "delta": 1.3746223090959653, "parent": 0.9463742516667925, "pass": false}, "near_ts_cos_drop": {"child": 0.02651090783138822, "delta": -0.330345031784573, "parent": 0.3568559396159612, "pass": false}, "near_zero_rmse": {"child": 0.032621213917101134, "parent": 0.007922443934609344, "pass": false}, "overall_cos_gain": {"child": 0.032682136318364446, "delta": -0.2329959228801284, "parent": 0.26567805919849286, "pass": false}, "pooled_ratio_increase": {"child": 2.4804126872679313, "delta": 1.4863583402967466, "parent": 0.9940543469711847, "pass": false}}, "reasons": ["overall_cos_gain: {'parent': 0.26567805919849286, 'child': 0.032682136318364446, 'delta': -0.2329959228801284, 'pass': False}", "near_ts_cos_drop: {'parent': 0.3568559396159612, 'child': 0.02651090783138822, 'delta': -0.330345031784573, 'pass': False}", "group_median_ratio_increase: {'parent': 0.9463742516667925, 'child': 2.3209965607627576, 'delta': 1.3746223090959653, 'pass': False}", "better_than_zero_groups: {'parent': 4, 'child': 0, 'pass': False}", "pooled_ratio_increase: {'parent': 0.9940543469711847, 'child': 2.4804126872679313, 'delta': 1.4863583402967466, 'pass': False}", "near_zero_rmse: {'parent': 0.007922443934609344, 'child': 0.032621213917101134, 'pass': False}"]}}, "promote": false, "reasons": ["seed 20261017 :: overall_cos_gain: {'parent': 0.20638577155359453, 'child': 0.13571909295238466, 'delta': -0.07066667860120987, 'pass': False}", "seed 20261017 :: near_ts_cos_drop: {'parent': 0.19681762914104506, 'child': 0.15748327153830693, 'delta': -0.039334357602738124, 'pass': False}", "seed 20261017 :: group_median_ratio_increase: {'parent': 1.026291765287766, 'child': 1.7498553846626543, 'delta': 0.7235636193748884, 'pass': False}", "seed 20261017 :: better_than_zero_groups: {'parent': 2, 'child': 0, 'pass': False}", "seed 20261017 :: pooled_ratio_increase: {'parent': 1.0281604822169716, 'child': 1.7693663771342016, 'delta': 0.74120589491723, 'pass': False}", "seed 20261017 :: near_zero_rmse: {'parent': 0.009229259819429398, 'child': 0.019443800568523042, 'pass': False}", "seed 20261018 :: overall_cos_gain: {'parent': 0.26567805919849286, 'child': 0.032682136318364446, 'delta': -0.2329959228801284, 'pass': False}", "seed 20261018 :: near_ts_cos_drop: {'parent': 0.3568559396159612, 'child': 0.02651090783138822, 'delta': -0.330345031784573, 'pass': False}", "seed 20261018 :: group_median_ratio_increase: {'parent': 0.9463742516667925, 'child': 2.3209965607627576, 'delta': 1.3746223090959653, 'pass': False}", "seed 20261018 :: better_than_zero_groups: {'parent': 4, 'child': 0, 'pass': False}", "seed 20261018 :: pooled_ratio_increase: {'parent': 0.9940543469711847, 'child': 2.4804126872679313, 'delta': 1.4863583402967466, 'pass': False}", "seed 20261018 :: near_zero_rmse: {'parent': 0.007922443934609344, 'child': 0.032621213917101134, 'pass': False}"], "rule_note": "§9.1 engineering screening rule; thresholds are NOT significance bounds; multiple passing candidates are ordered by rank_promotion_candidates (group-median MSE ↓ → cos ↑ → params ↓)"}, "winner_of": {"E1": "H0"}}
- {"stage": "V4.3", "stage_winner": true, "winner": "H0", "winner_config": {"contact_edges": false, "dataset_kind": "native", "ema_read": "raw", "encoder": "egnn", "features_version": "v3", "final_fit50": false, "head": "displacement", "hidden": 64, "keypair_condition": false, "lambda_dir": null, "lambda_mag": null, "n_layers": 3, "preset": "campaign_v1", "sampler": "group_balanced_v41", "updates": 20000, "use_prior_skip": false, "vector_channels": 16}}
- {"configs": {"E1": {"contact_edges": false, "dataset_kind": "native", "ema_read": "raw", "encoder": "egnn", "features_version": "v4", "final_fit50": false, "head": "displacement", "hidden": 64, "keypair_condition": false, "lambda_dir": null, "lambda_mag": null, "n_layers": 3, "preset": "campaign_v1", "sampler": "group_balanced_v41", "updates": 20000, "use_prior_skip": false, "vector_channels": 16}}}
- {"configs": {"E2": {"contact_edges": false, "dataset_kind": "native", "ema_read": "raw", "encoder": "egnn", "features_version": "v4", "final_fit50": false, "head": "displacement", "hidden": 64, "keypair_condition": true, "lambda_dir": null, "lambda_mag": null, "n_layers": 3, "preset": "campaign_v1", "sampler": "group_balanced_v41", "updates": 20000, "use_prior_skip": false, "vector_channels": 16}}}
- {"configs": {"E0": {"contact_edges": false, "dataset_kind": "native", "ema_read": "raw", "encoder": "egnn", "features_version": "v3", "final_fit50": false, "head": "displacement", "hidden": 64, "keypair_condition": false, "lambda_dir": null, "lambda_mag": null, "n_layers": 3, "preset": "campaign_v1", "sampler": "group_balanced_v41", "updates": 20000, "use_prior_skip": false, "vector_channels": 16}}, "promotion": {"decision": "RETAIN_PARENT", "metrics_used": ["overall_cos", "near_ts_cos", "group_median_ratio", "better_than_zero_groups", "pooled_ratio", "near_zero_rmse_A"], "per_clause": {}, "per_seed": {"20261017": {"per_clause": {"better_than_zero_groups": {"child": 3, "parent": 2, "pass": true}, "group_median_ratio_increase": {"child": 0.9855332757516039, "delta": -0.04075848953616201, "parent": 1.026291765287766, "pass": true}, "near_ts_cos_drop": {"child": 0.17607790979001398, "delta": -0.020739719351031083, "parent": 0.19681762914104506, "pass": false}, "near_zero_rmse": {"child": 0.008790236170511488, "parent": 0.009229259819429398, "pass": true}, "overall_cos_gain": {"child": 0.22403041613009056, "delta": 0.017644644576496032, "parent": 0.20638577155359453, "pass": true}, "pooled_ratio_increase": {"child": 1.0100881245871174, "delta": -0.018072357629854263, "parent": 1.0281604822169716, "pass": true}}, "reasons": ["near_ts_cos_drop: {'parent': 0.19681762914104506, 'child': 0.17607790979001398, 'delta': -0.020739719351031083, 'pass': False}"]}, "20261018": {"per_clause": {"better_than_zero_groups": {"child": 3, "parent": 4, "pass": false}, "group_median_ratio_increase": {"child": 0.9315052087798015, "delta": -0.01486904288699098, "parent": 0.9463742516667925, "pass": true}, "near_ts_cos_drop": {"child": 0.3544340266331027, "delta": -0.0024219129828584873, "parent": 0.3568559396159612, "pass": true}, "near_zero_rmse": {"child": 0.007602111475099823, "parent": 0.007922443934609344, "pass": true}, "overall_cos_gain": {"child": 0.2640103330200622, "delta": -0.0016677261784306396, "parent": 0.26567805919849286, "pass": false}, "pooled_ratio_increase": {"child": 0.955306502943176, "delta": -0.03874784402800868, "parent": 0.9940543469711847, "pass": true}}, "reasons": ["overall_cos_gain: {'parent': 0.26567805919849286, 'child': 0.2640103330200622, 'delta': -0.0016677261784306396, 'pass': False}", "better_than_zero_groups: {'parent': 4, 'child': 3, 'pass': False}"]}}, "promote": false, "reasons": ["seed 20261017 :: near_ts_cos_drop: {'parent': 0.19681762914104506, 'child': 0.17607790979001398, 'delta': -0.020739719351031083, 'pass': False}", "seed 20261018 :: overall_cos_gain: {'parent': 0.26567805919849286, 'child': 0.2640103330200622, 'delta': -0.0016677261784306396, 'pass': False}", "seed 20261018 :: better_than_zero_groups: {'parent': 4, 'child': 3, 'pass': False}"], "rule_note": "§9.1 engineering screening rule; thresholds are NOT significance bounds; multiple passing candidates are ordered by rank_promotion_candidates (group-median MSE ↓ → cos ↑ → params ↓)"}, "winner_of": {"V1": "E0"}}
- {"stage": "V4.4", "stage_winner": true, "winner": "E0", "winner_config": {"contact_edges": false, "dataset_kind": "native", "ema_read": "raw", "encoder": "egnn", "features_version": "v3", "final_fit50": false, "head": "displacement", "hidden": 64, "keypair_condition": false, "lambda_dir": null, "lambda_mag": null, "n_layers": 3, "preset": "campaign_v1", "sampler": "group_balanced_v41", "updates": 20000, "use_prior_skip": false, "vector_channels": 16}}
- {"configs": {"V1": {"contact_edges": false, "dataset_kind": "native", "ema_read": "raw", "encoder": "painn", "features_version": "v3", "final_fit50": false, "head": "displacement", "hidden": 64, "keypair_condition": false, "lambda_dir": null, "lambda_mag": null, "n_layers": 3, "preset": "campaign_v1", "sampler": "group_balanced_v41", "updates": 20000, "use_prior_skip": false, "vector_channels": 16}}}
- {"configs": {"V2": {"contact_edges": true, "dataset_kind": "native", "ema_read": "raw", "encoder": "painn", "features_version": "v3", "final_fit50": false, "head": "displacement", "hidden": 64, "keypair_condition": false, "lambda_dir": null, "lambda_mag": null, "n_layers": 3, "preset": "campaign_v1", "sampler": "group_balanced_v41", "updates": 20000, "use_prior_skip": false, "vector_channels": 16}}}
- {"arm": "V2", "decision": "pruned_per_prereg_order", "prune_order": ["W128", "V2", "E2"], "source": "pruned_arms.json"}
- {"configs": {"E0": {"contact_edges": false, "dataset_kind": "native", "ema_read": "raw", "encoder": "egnn", "features_version": "v3", "final_fit50": false, "head": "displacement", "hidden": 64, "keypair_condition": false, "lambda_dir": null, "lambda_mag": null, "n_layers": 3, "preset": "campaign_v1", "sampler": "group_balanced_v41", "updates": 20000, "use_prior_skip": false, "vector_channels": 16}}, "promotion": {"decision": "RETAIN_PARENT", "metrics_used": ["overall_cos", "near_ts_cos", "group_median_ratio", "better_than_zero_groups", "pooled_ratio", "near_zero_rmse_A"], "per_clause": {}, "per_seed": {"20261017": {"per_clause": {"better_than_zero_groups": {"child": 2, "parent": 2, "pass": true}, "group_median_ratio_increase": {"child": 1.0259273466173222, "delta": -0.00036441867044367804, "parent": 1.026291765287766, "pass": true}, "near_ts_cos_drop": {"child": 0.18784562678304612, "delta": -0.008972002357998943, "parent": 0.19681762914104506, "pass": true}, "near_zero_rmse": {"child": 0.00976897162486963, "parent": 0.009229259819429398, "pass": true}, "overall_cos_gain": {"child": 0.18902278230598696, "delta": -0.01736298924760757, "parent": 0.20638577155359453, "pass": false}, "pooled_ratio_increase": {"child": 1.0487646088853495, "delta": 0.020604126668377853, "parent": 1.0281604822169716, "pass": false}}, "reasons": ["overall_cos_gain: {'parent': 0.20638577155359453, 'child': 0.18902278230598696, 'delta': -0.01736298924760757, 'pass': False}", "pooled_ratio_increase: {'parent': 1.0281604822169716, 'child': 1.0487646088853495, 'delta': 0.020604126668377853, 'pass': False}"]}, "20261018": {"per_clause": {"better_than_zero_groups": {"child": 1, "parent": 4, "pass": false}, "group_median_ratio_increase": {"child": 1.310731599704413, "delta": 0.36435734803762065, "parent": 0.9463742516667925, "pass": false}, "near_ts_cos_drop": {"child": 0.2627382032627487, "delta": -0.09411773635321247, "parent": 0.3568559396159612, "pass": false}, "near_zero_rmse": {"child": 0.018930093948996453, "parent": 0.007922443934609344, "pass": false}, "overall_cos_gain": {"child": 0.2377660997080624, "delta": -0.02791195949043046, "parent": 0.26567805919849286, "pass": false}, "pooled_ratio_increase": {"child": 1.3362494638805322, "delta": 0.34219511690934756, "parent": 0.9940543469711847, "pass": false}}, "reasons": ["overall_cos_gain: {'parent': 0.26567805919849286, 'child': 0.2377660997080624, 'delta': -0.02791195949043046, 'pass': False}", "near_ts_cos_drop: {'parent': 0.3568559396159612, 'child': 0.2627382032627487, 'delta': -0.09411773635321247, 'pass': False}", "group_median_ratio_increase: {'parent': 0.9463742516667925, 'child': 1.310731599704413, 'delta': 0.36435734803762065, 'pass': False}", "better_than_zero_groups: {'parent': 4, 'child': 1, 'pass': False}", "pooled_ratio_increase: {'parent': 0.9940543469711847, 'child': 1.3362494638805322, 'delta': 0.34219511690934756, 'pass': False}", "near_zero_rmse: {'parent': 0.007922443934609344, 'child': 0.018930093948996453, 'pass': False}"]}}, "promote": false, "reasons": ["seed 20261017 :: overall_cos_gain: {'parent': 0.20638577155359453, 'child': 0.18902278230598696, 'delta': -0.01736298924760757, 'pass': False}", "seed 20261017 :: pooled_ratio_increase: {'parent': 1.0281604822169716, 'child': 1.0487646088853495, 'delta': 0.020604126668377853, 'pass': False}", "seed 20261018 :: overall_cos_gain: {'parent': 0.26567805919849286, 'child': 0.2377660997080624, 'delta': -0.02791195949043046, 'pass': False}", "seed 20261018 :: near_ts_cos_drop: {'parent': 0.3568559396159612, 'child': 0.2627382032627487, 'delta': -0.09411773635321247, 'pass': False}", "seed 20261018 :: group_median_ratio_increase: {'parent': 0.9463742516667925, 'child': 1.310731599704413, 'delta': 0.36435734803762065, 'pass': False}", "seed 20261018 :: better_than_zero_groups: {'parent': 4, 'child': 1, 'pass': False}", "seed 20261018 :: pooled_ratio_increase: {'parent': 0.9940543469711847, 'child': 1.3362494638805322, 'delta': 0.34219511690934756, 'pass': False}", "seed 20261018 :: near_zero_rmse: {'parent': 0.007922443934609344, 'child': 0.018930093948996453, 'pass': False}"], "rule_note": "§9.1 engineering screening rule; thresholds are NOT significance bounds; multiple passing candidates are ordered by rank_promotion_candidates (group-median MSE ↓ → cos ↑ → params ↓)"}, "winner_of": {"W96": "E0"}}
- {"stage": "V4.5", "stage_winner": true, "winner": "E0", "winner_config": {"contact_edges": false, "dataset_kind": "native", "ema_read": "raw", "encoder": "egnn", "features_version": "v3", "final_fit50": false, "head": "displacement", "hidden": 64, "keypair_condition": false, "lambda_dir": null, "lambda_mag": null, "n_layers": 3, "preset": "campaign_v1", "sampler": "group_balanced_v41", "updates": 20000, "use_prior_skip": false, "vector_channels": 16}}
- {"configs": {"W96": {"contact_edges": false, "dataset_kind": "native", "ema_read": "raw", "encoder": "egnn", "features_version": "v3", "final_fit50": false, "head": "displacement", "hidden": 96, "keypair_condition": false, "lambda_dir": null, "lambda_mag": null, "n_layers": 3, "preset": "campaign_v1", "sampler": "group_balanced_v41", "updates": 20000, "use_prior_skip": false, "vector_channels": 16}}}
- {"configs": {"W128": {"contact_edges": false, "dataset_kind": "native", "ema_read": "raw", "encoder": "egnn", "features_version": "v3", "final_fit50": false, "head": "displacement", "hidden": 128, "keypair_condition": false, "lambda_dir": null, "lambda_mag": null, "n_layers": 3, "preset": "campaign_v1", "sampler": "group_balanced_v41", "updates": 20000, "use_prior_skip": false, "vector_channels": 16}}}
- {"configs": {"W64": {"contact_edges": false, "dataset_kind": "native", "ema_read": "raw", "encoder": "egnn", "features_version": "v3", "final_fit50": false, "head": "displacement", "hidden": 64, "keypair_condition": false, "lambda_dir": null, "lambda_mag": null, "n_layers": 3, "preset": "campaign_v1", "sampler": "group_balanced_v41", "updates": 20000, "use_prior_skip": false, "vector_channels": 16}}, "promotion": {"decision": "RETAIN_PARENT", "metrics_used": ["overall_cos", "near_ts_cos", "group_median_ratio", "better_than_zero_groups", "pooled_ratio", "near_zero_rmse_A"], "per_clause": {}, "per_seed": {"20261017": {"per_clause": {"better_than_zero_groups": {"child": 2, "parent": 2, "pass": true}, "group_median_ratio_increase": {"child": 1.00290794108471, "delta": -0.02338382420305596, "parent": 1.026291765287766, "pass": true}, "near_ts_cos_drop": {"child": 0.3130927032653017, "delta": 0.11627507412425664, "parent": 0.19681762914104506, "pass": true}, "near_zero_rmse": {"child": 0.008673220836813668, "parent": 0.009229259819429398, "pass": true}, "overall_cos_gain": {"child": 0.19392092375906558, "delta": -0.012464847794528944, "parent": 0.20638577155359453, "pass": false}, "pooled_ratio_increase": {"child": 1.025055976584927, "delta": -0.003104505632044674, "parent": 1.0281604822169716, "pass": true}}, "reasons": ["overall_cos_gain: {'parent': 0.20638577155359453, 'child': 0.19392092375906558, 'delta': -0.012464847794528944, 'pass': False}"]}, "20261018": {"per_clause": {"better_than_zero_groups": {"child": 2, "parent": 4, "pass": false}, "group_median_ratio_increase": {"child": 1.0739057704235901, "delta": 0.1275315187567977, "parent": 0.9463742516667925, "pass": false}, "near_ts_cos_drop": {"child": 0.32233025525606707, "delta": -0.034525684359894115, "parent": 0.3568559396159612, "pass": false}, "near_zero_rmse": {"child": 0.00897528726137371, "parent": 0.007922443934609344, "pass": false}, "overall_cos_gain": {"child": 0.21033037404683783, "delta": -0.05534768515165503, "parent": 0.26567805919849286, "pass": false}, "pooled_ratio_increase": {"child": 1.0445128160472503, "delta": 0.05045846907606566, "parent": 0.9940543469711847, "pass": false}}, "reasons": ["overall_cos_gain: {'parent': 0.26567805919849286, 'child': 0.21033037404683783, 'delta': -0.05534768515165503, 'pass': False}", "near_ts_cos_drop: {'parent': 0.3568559396159612, 'child': 0.32233025525606707, 'delta': -0.034525684359894115, 'pass': False}", "group_median_ratio_increase: {'parent': 0.9463742516667925, 'child': 1.0739057704235901, 'delta': 0.1275315187567977, 'pass': False}", "better_than_zero_groups: {'parent': 4, 'child': 2, 'pass': False}", "pooled_ratio_increase: {'parent': 0.9940543469711847, 'child': 1.0445128160472503, 'delta': 0.05045846907606566, 'pass': False}", "near_zero_rmse: {'parent': 0.007922443934609344, 'child': 0.00897528726137371, 'pass': False}"]}}, "promote": false, "reasons": ["seed 20261017 :: overall_cos_gain: {'parent': 0.20638577155359453, 'child': 0.19392092375906558, 'delta': -0.012464847794528944, 'pass': False}", "seed 20261018 :: overall_cos_gain: {'parent': 0.26567805919849286, 'child': 0.21033037404683783, 'delta': -0.05534768515165503, 'pass': False}", "seed 20261018 :: near_ts_cos_drop: {'parent': 0.3568559396159612, 'child': 0.32233025525606707, 'delta': -0.034525684359894115, 'pass': False}", "seed 20261018 :: group_median_ratio_increase: {'parent': 0.9463742516667925, 'child': 1.0739057704235901, 'delta': 0.1275315187567977, 'pass': False}", "seed 20261018 :: better_than_zero_groups: {'parent': 4, 'child': 2, 'pass': False}", "seed 20261018 :: pooled_ratio_increase: {'parent': 0.9940543469711847, 'child': 1.0445128160472503, 'delta': 0.05045846907606566, 'pass': False}", "seed 20261018 :: near_zero_rmse: {'parent': 0.007922443934609344, 'child': 0.00897528726137371, 'pass': False}"], "rule_note": "§9.1 engineering screening rule; thresholds are NOT significance bounds; multiple passing candidates are ordered by rank_promotion_candidates (group-median MSE ↓ → cos ↑ → params ↓)"}, "winner_of": {"FINAL": "W64"}}
- {"stage": "V4.6", "stage_winner": true, "winner": "W64", "winner_config": {"contact_edges": false, "dataset_kind": "native", "ema_read": "raw", "encoder": "egnn", "features_version": "v3", "final_fit50": false, "head": "displacement", "hidden": 64, "keypair_condition": false, "lambda_dir": null, "lambda_mag": null, "n_layers": 3, "preset": "campaign_v1", "sampler": "group_balanced_v41", "updates": 20000, "use_prior_skip": false, "vector_channels": 16}}
- {"a_s_fit50": 0.01198590928329299, "arm": "FINAL", "decision": "STAY_20K", "forty_k_gate": {"allow_40k": false, "decision": "STAY_20K", "per_seed": {"20261017": {"group_median_ratio_drop": {"delta": 0.2601949643769017, "pass": false, "u10000": 1.3018057224347286, "u20000": 1.5620006868116303}, "near_ts_cos_drop": {"delta": -0.060937597918864495, "pass": false, "u10000": 0.34207537733185867, "u20000": 0.28113777941299417}, "near_zero_rmse": {"pass": false, "u10000": 0.015210132154027415, "u20000": 0.016995168083860264}, "overall_cos_drop": {"delta": -0.028195099774301013, "pass": false, "u10000": 0.23427332641598855, "u20000": 0.20607822664168754}}, "20261018": {"group_median_ratio_drop": {"delta": 0.32864371732180064, "pass": false, "u10000": 1.0899958486042467, "u20000": 1.4186395659260473}, "near_ts_cos_drop": {"delta": 0.03431759160869635, "pass": true, "u10000": 0.2959586196410482, "u20000": 0.33027621124974454}, "near_zero_rmse": {"pass": false, "u10000": 0.01224980265683107, "u20000": 0.017446492146839757}, "overall_cos_drop": {"delta": -0.013835094879598442, "pass": false, "u10000": 0.20745803008280067, "u20000": 0.19362293520320223}}}, "reasons": ["seed 20261017 :: group_median_ratio_drop: {'u10000': 1.3018057224347286, 'u20000': 1.5620006868116303, 'delta': 0.2601949643769017, 'pass': False}", "seed 20261017 :: overall_cos_drop: {'u10000': 0.23427332641598855, 'u20000': 0.20607822664168754, 'delta': -0.028195099774301013, 'pass': False}", "seed 20261017 :: near_ts_cos_drop: {'u10000': 0.34207537733185867, 'u20000': 0.28113777941299417, 'delta': -0.060937597918864495, 'pass': False}", "seed 20261017 :: near_zero_rmse: {'u10000': 0.015210132154027415, 'u20000': 0.016995168083860264, 'pass': False}", "seed 20261018 :: group_median_ratio_drop: {'u10000': 1.0899958486042467, 'u20000': 1.4186395659260473, 'delta': 0.32864371732180064, 'pass': False}", "seed 20261018 :: overall_cos_drop: {'u10000': 0.20745803008280067, 'u20000': 0.19362293520320223, 'delta': -0.013835094879598442, 'pass': False}", "seed 20261018 :: near_zero_rmse: {'u10000': 0.01224980265683107, 'u20000': 0.017446492146839757, 'pass': False}"], "rule_note": "§9.2 40k gate is engineering evidence for extending training, NOT a guarantee that 40k beats 20k; on budget failure fall back to 20k uniformly (plan §9.2)", "timing": {"budget_minutes": 95.0, "budget_seconds": 5700.0, "n_seeds": 3, "pass": true, "seed_40k_seconds": 816.8903036392294, "total_40k_seconds": 2450.670910917688}}}

## 6. 状态与结论

_待人工终审填写；本文件不预填收益，负结果如实。_

