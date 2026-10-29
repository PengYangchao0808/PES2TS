# V5.4.1 WP7 冻结 dev 评估（仅 matched E0 × 3 seeds）——架构 / 接口 / 协议设计

版本：v1.0（2026-10-28）
状态：设计稿（本步只产出设计文档；实现由 fixer 按本文执行，验证负责人为主代理）
上游依据：`docs/plans/PES2TS_V5.4.1_Embedding_50K_详细阶段实施方案_v1.0_20261027.md` §10（WP7）、§4.3/§9.3（WP1b 密封）、§12（产物与账本）

---

## 0. 裁决前提（不可变输入）

`outputs/pesn/v541_embed_50k/reports/single_seed_screen.json`（SHA256 `1bae9a0f0b37392fd86c5b9d5a68fa9a4f89a504d6c9cb195479e7511820a17b`）已裁决：

| 候选 | 结论 | WP7 是否可评估 |
|---|---|---|
| E1 / E2 / E3 | `scientific_failure`（H2 MOVE guard 失败） | **禁止** |
| E4 | `gate_skipped`（cost_ratio 3.6455 > 2.0） | **禁止** |
| E5 / T3 / T4 | `gate_skipped`（父门失败 / 无晋级者） | **禁止** |
| E\* | **E0**（T0-R matched retrain × seeds 17/29/43） | **允许，本次唯一评估对象** |

dev 政策（screen 原文）：`WP7 evaluates frozen matched E0 x3 only; no candidate dev access and confirm remains sealed`。
因此 WP7 本次 dev 访问的性质是 **candidate_evaluation（对冻结集合 E0×3 的一次性评估）**，不是 baseline_repair，也不是 confirm。三 seed 之外无任何候选可评，dev 不产生"改选 / 调参 / 重算 tau"的合法路径。

---

## 1. 目标与非目标

### 1.1 目标

1. **先冻结、后解封**：写 `reports/wp7_freeze.json` 绑定全部输入 SHA（screen、E0 checkpoints、inner-val tau、runplan、代码、统计代码），核验 WP1 密封 manifest 与其全部输出文件 SHA 未变之后，才允许任何 dev 读取。
2. **一次性评估**：用 `V541Model` E0 strict load，对每个 seed 在**该 seed 的 R3 reference 同一 dev 位置集**上计算完整 MOVE / PROCESS / tau / 分母指标；`train_q_mean` 只来自该 seed 的 inner-train，`tau_end` 只用 checkpoint 内冻结的 inner-val 值。
3. **可追溯**：每次 dev 访问向 `audits/dev_access_ledger.jsonl` 追加一条 `purpose=candidate_evaluation` 记录（actor、curve/position SHA、code SHA、时间、输出 SHA）；WP1 的 `baseline_repair` 记录一个字节都不改。
4. **可审计产物**：`reports/dev_frozen_evaluation.json`（WP7 允许详细）+ 每 seed 明细文件；三 seed 只做汇总（均值/范围），不做任何选择或阈值动作。

### 1.2 非目标（本步不做，后续 WP7 子步另开）

- paired CI / bootstrap（无候选对比，主对比"候选减 matched E0"不存在；统计代码 SHA 只做冻结绑定）；
- rollout（24/96-curve 子集）、compute-Pareto、export bundle / model card；
- confirm / OOD 访问（继续封存，见 §9 失败语义中的硬拒）。

### 1.3 硬边界

- 不改 `v541_run.py`、`v541_baseline_repair.py`、model、config、既有 docs —— **只 import，不修改**（避免破坏 freeze 与 WP1 资格链）；
- 不训练、不重算 checkpoint、`new_QC_calls = 0`、不触碰 confirm/pool split；
- 不评估 E1–E6 任何预测（不加载其 checkpoint、不前向）；
- dev 只读；输出只写 `reports/`（新文件）与 `audits/dev_access_ledger.jsonl`（追加）。

---

## 2. 模块边界

### 2.1 新增（本 WP7 唯一新代码面）

```
scripts/campaigns/v541_dev_evaluate.py     # WP7 冻结 dev 评估 runner（全新文件）
tests/campaigns/test_v541_dev_evaluate.py  # fixture 测试（全新文件）
outputs/pesn/v541_embed_50k/reports/wp7_freeze.json            # 冻结记录（先写）
outputs/pesn/v541_embed_50k/reports/dev_frozen_evaluation.json # 汇总报告
outputs/pesn/v541_embed_50k/reports/dev_frozen/                # per-seed 明细
    V541_T0R_E0_S17_dev.json
    V541_T0R_E0_S29_dev.json
    V541_T0R_E0_S43_dev.json
```

### 2.2 复用（import-only，全部既有实现）

| 复用对象 | 来源 | 用途 |
|---|---|---|
| `_predict_once` / `_PosPred` | `scripts.campaigns.v541_baseline_repair` | 单次前向（chunk 64，与历史 q/endpoint pass 同构成） |
| `_build_move_variants` / `_move_metrics` | 同上 | 修复后 pair 工具（WP1a `assign_batch_pairs`）的 before/after MOVE |
| `_process_metrics` | 同上 | q / ordering / endpoint / tau misstop（与 `evaluate_v53mp` 同分组同 helper） |
| `_move_sanity` / `_h2_move_guard` / `_compare` / `_pass_all` / `_gate_flags` | 同上 | 分母 sanity、H2 MOVE guard 公式、门比较 |
| `_Cost` / `_GpuLock` / `_free_cuda` | 同上 | 成本计量、GPU 单路锁、显存释放 |
| `_sha256_file` / `_sha256_text` / `_canonical_positions_sha` / `_canonical_curves_sha` / `_jsonable` / `_write_json` / `_read_json` / `_actor` / `_now` | 同上 | SHA、JSON 原子写、actor/时间 |
| `_build_tier` / `_trainer_config` / `_ok_mask` / `TIER_DEFAULTS` | `scripts.campaigns.r3_v53mp_scale_paired_ci` | dev 语料 + 冻结 dev feature pack + ok 掩码 |
| `_subset_curve_ids` / `_tier_pool_curve_ids` | `scripts.campaigns.r3_v53mp_run` | dev 曲线集、pool 曲线集 |
| `load_model_v541` / `V541TrainerConfig` | `pes2ts_core.pesn.offline.training.v541_train` | E0 模型构造（R3 strict warmstart） |
| `v53mp_pool_split` / `_curve_id_set_of` / `_train_q_mean` / `evaluate_g52`/`evaluate_g53`（经由上述封装） | `pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3` | inner-train 切分、常数 q 基线 |
| `positions_sha256` / `features_sha256_of` / `DevFeaturePackView` | `pes2ts_core.pesn.offline.data.key_atom_dev_pack_v53mp` | dev pack SHA 与只读视图 |

> 复用函数**一律不改签名不改实现**；若 WP7 需要 WP1 没有的行为（如常数 q 基线后处理），在 `v541_dev_evaluate.py` 内部做**纯增量包装**，不回写被复用模块。

### 2.3 禁改清单（冻结保护）

`scripts/campaigns/v541_run.py`、`v541_baseline_repair.py`、`v541_runplan.py`、`v541_provenance.py`、`v541_parity.py`、`v541_legality_audit.py`、`r3_v53mp_*.py`、`pes2ts_core/pesn/**`（models/training/evaluation/data）、`config/campaigns/pes2ts_v541_embed_50k.yaml`、`plan/*`、`audits/dev_baseline_sealed/**`、`audits/evaluator_before_after.json`、`reports/single_seed_screen.json`。
执行前用 `git status --porcelain` + 关键文件 SHA 快照证明确未改动（写入 freeze 的 `provenance`）。

### 2.4 `v541_dev_evaluate.py` 内部分层

```
┌────────────────────────────────────────────────────────┐
│ cli（freeze / run / verify 三阶段，fail-closed）         │
├────────────────────────────────────────────────────────┤
│ bindings   §5 冻结绑定构造与核验（SHA 矩阵）              │
│ gate       screen 裁决门 / WP1 密封门 / confirm 硬拒     │
│ loader     V541Model E0 strict load（checkpoint→config）│
│ evalctx    dev corpus + pack + per-seed inner-train 定位 │
│ metrics    per-seed MOVE/PROCESS/tau/分母 组装            │
│ aggregate  三 seed 描述性汇总（无选择、无调参）            │
│ ledger     candidate_evaluation 追加写（fsync）           │
└────────────────────────────────────────────────────────┘
```

---

## 3. 阶段与数据流

```
[stage freeze]  (无 dev 访问)
  screen SHA ─┐
  ckpt SHA ×3 ┼─► wp7_freeze.json ──► (绑定成立)
  tau ×3     ─┤
  runplan/code/stats SHA ─┘

[stage run]  (cgroup 12G/10G 单路, GPU lock)
  ① 核验 WP1 密封：manifest["files"] 逐文件 SHA == 当前文件
                   manifest.json SHA == ledger 最后一条 seal_manifest 记录
                   evaluator_before_after.status == "ok" 且 freeze 未漂移
  ② 核验 freeze 自身：所有绑定 SHA 复算一致；screen E*==E0 且 selected==[]
  ③ 打开 dev 语料：_build_tier("50k", …, dev_pack="auto") → pack 必须 used
       dev_pos / ok_mask / curves；两套 position SHA 复算并与 WP1 密封值 exact 比对
  ④ per-seed ∈ {17,29,43}：
       load_model_v541(cfg_from_ckpt, r3_ckpt=R3) → load_state_dict(strict=True)
       tau   = ckpt["tau_end_calibration"]["tau_end"]        # 只读，不重算
       tqm   = _train_q_mean(samples, labels, train_pos(seed))  # 只来自 inner-train
       _eval_split("dev", dev_pos, …, tau_end=tau)            # ONE pass
       + 常数 q 基线后处理（evaluate_v53mp 口径）
       + 分母 sanity / 与 WP1 密封 R3 reference 的 H2 guard 描述块
       → 写 reports/dev_frozen/V541_T0R_E0_S{seed}_dev.json + ledger 追加
  ⑤ 三 seed 汇总 → reports/dev_frozen_evaluation.json + ledger 追加
  ⑥ confirm：全程 0 次接触（split 常量锁 "dev"）

[stage verify]  (无 dev 访问)
  复算产物 SHA、ledger 尾部一致性、WP1 段未被覆写（字节前缀不变）
```

阶段间失败即停：`run` 未成功不得进 `verify`；`freeze` 未成功任何 dev 读取都被 `gate` 拒绝。

---

## 4. 接口定义

### 4.1 CLI

```
python scripts/campaigns/v541_dev_evaluate.py freeze  [--root outputs/pesn/v541_embed_50k]
python scripts/campaigns/v541_dev_evaluate.py run     [--seeds 17,29,43] [--device cuda]
                                                      [--limit-positions N   # 仅 fixture/测试]
python scripts/campaigns/v541_dev_evaluate.py verify
```

- `freeze`：幂等；若 `wp7_freeze.json` 已存在且绑定 SHA 全部仍一致 → 原样通过；任一漂移 → typed refusal（不覆盖旧文件，写 `failure_*.json` 后退出非零）。
- `run`：必须已有通过核验的 freeze；`--limit-positions` 只在 `PES2TS_WP7_FIXTURE=1` 时合法（防止真实运行截断分母）。
- `verify`：只读复核（产物 SHA、ledger、WP1 前缀不变），不打印科学数值以外的新计算。

### 4.2 关键内部函数（实现契约）

```python
def build_freeze(root: Path) -> Dict[str, Any]                    # §5.1 schema
def verify_wp1_seal(root: Path) -> Dict[str, Any]                 # 抛 WP7Error("WP1_SEAL_*")
def verify_freeze(root: Path) -> Dict[str, Any]                   # 抛 WP7Error("FREEZE_DRIFT_*")
def load_e0_model(seed: int, device: str) -> Tuple[Module, Mapping]
    # cfg = V541TrainerConfig(**filter(ckpt["config"]))  ← 与 legality_audit 同法
    # model = load_model_v541(cfg, r3_ckpt=R3_INIT, device=device)
    # model.load_state_dict(ckpt["model"], strict=True)   # 必须 strict
def seed_inner_train_positions(corpus, seed) -> List[int]         # v53mp_pool_split 的 train 侧
def eval_seed_dev(seed, corpus, device, cost) -> Dict[str, Any]   # §5.2 schema
def constant_q_block(q_pred, q_true, q_mask, train_q_mean) -> Dict
    # 镜像 evaluate_v53mp: const_mae / const_value / learnable_above_constant
def aggregate_three_seed(per_seed: Mapping[str, Mapping]) -> Dict # §5.3 schema
def append_ledger_candidate_evaluation(...) -> Dict[str, Any]     # §5.4 schema
```

错误类型：`class WP7Error(RuntimeError): code, detail`（对齐 `RepairError` 风格），全部走 fail-closed。

---

## 5. JSON Schema

### 5.1 `reports/wp7_freeze.json`（先写；dev 访问的“门票”）

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "v541_wp7_freeze_v1",
  "type": "object",
  "required": ["schema", "created_at", "wp", "dev_access", "screen", "frozen_set",
               "inner_val_tau", "plan_bindings", "code_bindings", "wp1_seal",
               "confirm", "provenance"],
  "properties": {
    "schema":   {"const": "v541_wp7_freeze_v1"},
    "created_at": {"type": "string"},
    "wp":       {"const": "WP7"},
    "dev_access": {
      "type": "object",
      "required": ["type", "purpose", "split", "policy"],
      "properties": {
        "type":   {"const": "candidate_evaluation"},
        "purpose":{"const": "frozen_dev_evaluation_matched_e0_only"},
        "split":  {"const": "dev"},
        "policy": {"type": "string"},
        "n_planned_evaluations": {"const": 3}
      }
    },
    "screen": {
      "type": "object",
      "required": ["path", "sha256", "e_star", "selected", "adjudication"],
      "properties": {
        "path": {"const": "outputs/pesn/v541_embed_50k/reports/single_seed_screen.json"},
        "sha256": {"const": "1bae9a0f0b37392fd86c5b9d5a68fa9a4f89a504d6c9cb195479e7511820a17b"},
        "e_star": {"const": "E0"},
        "selected": {"type": "array", "maxItems": 0},
        "adjudication": {
          "E1": "scientific_failure", "E2": "scientific_failure",
          "E3": "scientific_failure", "E4": "gate_skipped", "E5": "gate_skipped",
          "T3": "gate_skipped", "T4": "gate_skipped"
        },
        "forbidden_candidates": {"const": ["E1","E2","E3","E4","E5"]}
      }
    },
    "frozen_set": {
      "type": "object",
      "description": "E0 T0-R matched retrain × 3 seeds；WP7 唯一允许前向的模型集合",
      "required": ["seeds", "per_seed"],
      "properties": {
        "seeds": {"const": [17, 29, 43]},
        "per_seed": {
          "type": "object",
          "required": ["17", "29", "43"],
          "patternProperties": {
            "^(17|29|43)$": {
              "type": "object",
              "required": ["run_id", "checkpoint_best_sha256", "config_full_sha256",
                           "r3_init_sha256", "schema", "run_outcome", "is_formal_run"],
              "properties": {
                "run_id": {"type": "string"},
                "checkpoint_path": {"type": "string"},
                "checkpoint_best_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "metrics_sha256": {"type": "string"},
                "config_full_sha256": {"type": "string"},
                "r3_init_sha256": {"const": "a2e46b6ad65cc18996d6c1660c5ae24c723397ca22526e52fa34e430afd0417f"},
                "schema": {"const": "pesn_v541_train_v1"},
                "run_outcome": {"const": "done"},
                "is_formal_run": {"const": true},
                "embedding_id": {"const": "E0"}
              }
            }
          }
        }
      }
    },
    "inner_val_tau": {
      "type": "object",
      "description": "tau 只来自 checkpoint 内冻结的 inner-val 校准；WP7 不重算",
      "required": ["calibrated_on", "per_seed"],
      "properties": {
        "calibrated_on": {"const": "inner_val"},
        "per_seed": {
          "patternProperties": {
            "^(17|29|43)$": {
              "type": "object",
              "required": ["tau_end", "target_precision", "n_inner_val_unmasked", "source"],
              "properties": {
                "tau_end": {"type": "number"},
                "target_precision": {"type": "number"},
                "achieved_precision": {"type": "number"},
                "achieved_recall": {"type": "number"},
                "prevalence": {"type": "number"},
                "n_inner_val_unmasked": {"type": "integer"},
                "fallback": {},
                "source": {"const": "checkpoint_best.pt#tau_end_calibration"}
              }
            }
          }
        }
      }
    },
    "plan_bindings": {
      "type": "object",
      "required": ["runplan_sha256", "code_sha256", "config_sha256", "budget_sha256",
                   "source_manifest_sha256", "common_protocol_sha256", "environment_sha256"],
      "properties": {
        "runplan_sha256": {"const": "befdf845d946e8e4e6608319f80fbcd5176453d41a2c86490e3338663063cb7f"},
        "code_sha256": {"const": "dad4d1fe1d1912cab5bed13608f881ab78ac24528ad60410681ad709edd9253e"},
        "config_sha256": {"const": "464c1f6fa02ffb6bd7770861eb5690136384b6f22e4fd3ba9ce9380293321c62"},
        "budget_sha256": {"const": "10fecf9df1cfff456d17ca981fe87cdb531899a327582d42e14ce3dc26bea505"},
        "source_manifest_sha256": {"const": "e25854e3a96b5bf4880cd851d85fa5031c54df85ba984acc796a8f493c972b5b"},
        "runplan_pin_path": {"type": "string"},
        "seeds": {"const": [17, 29, 43]}
      }
    },
    "code_bindings": {
      "type": "object",
      "description": "eval-path + 统计代码的逐文件 SHA（identity 口径，同 WP1）",
      "required": ["eval_path_files", "stats_code_files", "evaluator_code_sha256", "wp7_script_sha256"],
      "properties": {
        "eval_path_files": {"type": "object", "additionalProperties": {"type": "string"}},
        "stats_code_files": {
          "type": "object",
          "properties": {
            "scripts/campaigns/r3_v53mp_scale_paired_ci.py":
              {"const": "39550bdc386659f48f35933d06b3fc2edf9e6403f132f8130572f6dd55d9ae38"},
            "scripts/campaigns/r3_v53mp_scale_paired_ci.py@paired_ci_tooling": {"type": "string"}
          }
        },
        "evaluator_code_sha256": {"type": "string"},
        "wp7_script_sha256": {"type": "string", "description": "v541_dev_evaluate.py 自身 SHA"},
        "dev_pack_tooling_sha256": {"type": "string"}
      }
    },
    "wp1_seal": {
      "type": "object",
      "description": "解封前置条件：WP1 密封完好（详见 §6 WP1 矩阵）",
      "required": ["sealed_manifest_path", "sealed_manifest_sha256", "files", "ledger_prefix_sha256",
                   "evaluator_report_sha256", "status"],
      "properties": {
        "sealed_manifest_path": {"type": "string"},
        "sealed_manifest_sha256": {"const": "a1d29b74116923ad0094c46e99edb68e1b2690291204969d4e3a4f42d2db742b"},
        "files": {
          "type": "object",
          "description": "manifest.files 的逐文件 SHA（当前磁盘复算必须相等）",
          "examples": [{"v53mp_50k_C_s17_dev.json": "<sha>", "r3_reference_dev.json": "<sha>"}]
        },
        "ledger_path": {"type": "string"},
        "ledger_prefix_sha256": {"type": "string",
          "description": "WP1 段 ledger 字节前缀的 SHA（WP7 追加后 verify 阶段必须仍匹配）"},
        "ledger_entries_wp1": {"type": "integer"},
        "evaluator_report_path": {"type": "string"},
        "evaluator_report_sha256": {"const": "4ca61fdb7e642ab38ba92dae1235535f2ca00fbaf65621414e2df76fdf130514"},
        "status": {"const": "ok"},
        "dev_sealed_pass": {"const": true},
        "unseal_rule": {"type": "string"}
      }
    },
    "dev_reference_binding": {
      "type": "object",
      "description": "WP1 密封的 R3 reference dev 指标（H2 描述性 guard 的参照，复用不重算）",
      "properties": {
        "r3_reference_dev_sha256": {"type": "string"},
        "n_dev_positions": {"const": 135926},
        "n_move_ok_positions": {"const": 116527},
        "n_dev_curves": {"const": 589},
        "canonical_positions_sha256": {"const": "3063dcf778a7dea6c13e6cb458234618c677dd6d3c5bf708f780decbd5fe2908"},
        "devpack_positions_sha256": {"const": "4f6c1e950f23dd49d21b36566e38cbd9a310c69393798c8202c2d2caf7b111f5"},
        "curves_sha256": {"const": "6377ea9687f11493aa02d139b8acb0072a34f1fd900c14f3edc05240d7d7151d"}
      }
    },
    "confirm": {
      "type": "object",
      "required": ["status", "new_QC_calls", "confirm_read"],
      "properties": {
        "status": {"const": "sealed"},
        "confirm_read": {"const": false},
        "new_QC_calls": {"const": 0}
      }
    },
    "provenance": {
      "type": "object",
      "properties": {
        "actor": {"type": "string"},
        "git_head": {"type": "string"},
        "git_diff_sha256": {"type": "string"},
        "git_porcelain_sha256": {"type": "string"}
      }
    }
  }
}
```

实际值（2026-10-28 复算，写入时以复算为准）：S17 ckpt `c1e35610…0ca56d`、S29 `a214d6e9…29e49d`、S43 `92412a1a…a6f00c`；tau S17 `0.9046342048457465`、S29 `0.7806053768606613`、S43 `0.9439198254589221`。

### 5.2 `reports/dev_frozen/V541_T0R_E0_S{seed}_dev.json`（per-seed 明细，WP7 允许详细）

```json
{
  "schema": "v541_wp7_dev_seed_v1",
  "split": "dev",
  "seed": 17,
  "model": "V541_E0_T0R",
  "run_id": "V541_T0R_E0_S17",
  "checkpoint_sha256": "…",
  "strict_load": {"mode": "strict", "ok": true, "n_missing": 0, "n_unexpected": 0},
  "tau": {
    "tau_end": 0.9046342048457465,
    "source": "checkpoint_best.pt#tau_end_calibration",
    "calibrated_on": "inner_val",
    "recomputed": false
  },
  "train_q_mean": {
    "value": 0.0,
    "source": "inner-train rows of v53mp pool split (seed=17)",
    "n_rows_used": 0, "n_q_masked": 0,
    "derived_from_dev": false
  },
  "denominators": {
    "n_positions": 135926, "n_positions_ok": 116527,
    "n_curves": 589, "n_curves_move": 557,
    "positions_sha256": "3063dcf7…", "curves_sha256": "6377ea96…",
    "devpack_positions_sha256": "4f6c1e95…",
    "n_forward_batches": 2124,
    "n_pairs_input": 0, "n_pairs_collected_before": 0, "n_pairs_collected_after": 0
  },
  "denominator_gates": [ {"name": "n_dev_positions_exact", "pass": true, "…": "…" } ],
  "sanity_gates": [ {"name": "s1_repaired_no_silent_pair_drop", "pass": true} ],
  "move": {
    "after_repaired": {"key_rmse_over_zero": {}, "sre_and_sign": {}, "fixed_coverage_direction": {}},
    "before_legacy":  {"sre_and_sign": {}},
    "pair_denominators": {}
  },
  "process": {
    "q_head": {"mae_curve_mean": 0, "corr_curve_mean": 0, "n_eval_curves": 0,
               "constant_q_baseline_mae": 0, "constant_q_value": 0,
               "learnable_above_constant": true},
    "ordering": {"pairwise_accuracy_full": 0, "n_pairs_full": 0,
                 "pairwise_accuracy_cross_ts": 0, "n_pairs_cross_ts": 0},
    "endpoint": {"auprc": 0, "auroc": 0, "prevalence": 0,
                 "n_unmasked": 0, "n_pos": 0, "n_neg": 0,
                 "tau_end": 0, "start_misstop_rate_at_tau": 0,
                 "ts_misstop_rate_at_tau": 0, "n_start_rows": 0, "n_ts_rows": 0}
  },
  "h2_reference_guard_dev": {
    "reference": "WP1 sealed r3_reference_dev.json (reused, SHA-verified)",
    "rmse_degradation_pct": 0, "within_rmse_guard_2pct": true,
    "note": "descriptive control only — E0 is the baseline; no candidate gating"
  },
  "cost": {"device": "cuda", "forward_wall_s": 0, "wall_s": 0, "new_QC_calls": 0},
  "seed_pass": true,
  "created_at": "…",
  "ledger_entry_id": "…"
}
```

约束：`denominator_gates` 至少含 `n_dev_positions_exact / n_move_ok_exact / n_curves_exact / canonical_positions_sha_exact / devpack_positions_sha_exact / curves_sha_exact` 六项，全部对 WP1 密封值 **exact** 比对；任一失败 → 该 seed `WP7_DEV_DENOMINATOR_MISMATCH`，fail-closed。

### 5.3 `reports/dev_frozen_evaluation.json`（汇总）

```json
{
  "schema": "v541_wp7_dev_frozen_evaluation_v1",
  "created_at": "…",
  "wp": "WP7",
  "dev_access": {"type": "candidate_evaluation", "split": "dev", "n_evaluations": 3},
  "freeze": {"path": "…/wp7_freeze.json", "sha256": "…", "verified": true},
  "wp1_seal_verified": true,
  "per_seed_files": {"17": {"path": "…", "sha256": "…"}, "29": { }, "43": { }},
  "aggregate": {
    "unit": "seed (descriptive only)",
    "metrics": {
      "endpoint_auprc":        {"per_seed": {}, "mean": 0, "min": 0, "max": 0},
      "q_corr_curve_mean":     {"per_seed": {}, "mean": 0, "min": 0, "max": 0},
      "q_mae_curve_mean":      {"per_seed": {}, "mean": 0, "min": 0, "max": 0},
      "median_rmse_over_zero": {"per_seed": {}, "mean": 0, "min": 0, "max": 0},
      "cos_eff_median":        {"per_seed": {}, "mean": 0, "min": 0, "max": 0},
      "dead_fraction":         {"per_seed": {}, "mean": 0, "min": 0, "max": 0},
      "sign_accuracy_cluster_median": {"per_seed": {}, "mean": 0, "min": 0, "max": 0},
      "ordering_accuracy_full":        {"per_seed": {}, "mean": 0, "min": 0, "max": 0},
      "endpoint_start_misstop_rate_at_tau": {"per_seed": {}, "mean": 0, "min": 0, "max": 0},
      "endpoint_ts_misstop_rate_at_tau":    {"per_seed": {}, "mean": 0, "min": 0, "max": 0}
    }
  },
  "denominator_summary": {"n_dev_positions": 135926, "n_move_ok": 116527, "n_dev_curves": 589,
                          "per_seed_pairs_after": {}, "n_invalid": 0},
  "adjudication_context": {
    "e_star": "E0", "candidates_evaluated_on_dev": 0,
    "forbidden_candidates": ["E1","E2","E3","E4","E5"],
    "dev_veto_applicable": false,
    "reason": "E*=E0 with zero surviving candidates; plan §10.1.4 veto path has no upgrade to veto",
    "selection_or_tuning_performed": false,
    "tau_recomputed": false,
    "runner_up_considered": false
  },
  "controls": {"confirm_read": false, "new_QC_calls": 0, "qc_calls_breakdown": {},
               "training_runs_started": 0},
  "status": "ok",
  "cost": {"wall_s": 0, "peak_rss_mib": 0, "device": "cuda"},
  "ledger_entries_this_run": []
}
```

### 5.4 ledger 追加条目（`audits/dev_access_ledger.jsonl`）

沿用 WP1 的 `v541_dev_access_ledger_v1` 行 schema（同一文件、同一格式），字段取值：

```json
{
  "schema": "v541_dev_access_ledger_v1",
  "entry_id": "dev_<model>_<ms>",
  "actor": "<user@host>#wp7_dev_frozen_eval",
  "purpose": "candidate_evaluation",
  "wp": "WP7",
  "campaign": "V5.4.1",
  "split": "dev",
  "model": "V541_E0_T0R_S17" | "…_S29" | "…_S43" | "aggregate",
  "ts": "<UTC ISO8601>",
  "curve_set_sha256": "6377ea96…",
  "positions_sha256": "3063dcf7…",
  "devpack_positions_sha256": "4f6c1e95…",
  "n_positions": 135926, "n_curves": 589,
  "code_sha256": "<WP7 eval-path identity sha>",
  "output_path": "…/reports/dev_frozen/V541_T0R_E0_S17_dev.json",
  "output_sha256": "<sha of that file>",
  "freeze_sha256": "<wp7_freeze.json sha>",
  "sealed_reference_manifest_sha256": "a1d29b74…",
  "new_QC_calls": 0,
  "confirm_read": false,
  "cost": {"device": "cuda", "wall_s": 0, "forward_batches": 2124},
  "status": "done"
}
```

每个输出文件一条（3 per-seed + 1 aggregate = 4 条新行）；写入用 `open("a") + flush + fsync`，与 WP1 `_append_ledger` 相同字节格式（`json.dumps(sort_keys=True, ensure_ascii=False)` + `\n`）。

---

## 6. 冻结绑定与一致性校验（SHA 矩阵）

### 6.1 WP7 freeze 绑定（`freeze` 阶段写入、`run` 阶段复算）

| 绑定 | 期望值 | 复算来源 |
|---|---|---|
| screen | `1bae9a0f…20a17b` | 文件字节 SHA |
| E0 ckpt S17/29/43 | `c1e35610…` / `a214d6e9…` / `92412a1a…` | checkpoint_best.pt 字节 SHA；且 == metrics.json `checkpoint_best_sha256` |
| metrics.json ×3 | 运行时复算 | 与 checkpoint 交叉一致 |
| tau ×3 | `0.90463…` / `0.78060…` / `0.94391…` | checkpoint `tau_end_calibration.tau_end` == metrics 同字段 |
| r3_init | `a2e46b6a…` | R3 checkpoint 字节 SHA == 三 seed provenance `r3_init_sha256` |
| runplan | `befdf845…` | `plan/runplan.pin.json` `runplan_sha256` 字段 == 文件内自洽 |
| code | `dad4d1fe…` | pin `code_sha256`；eval-path 文件逐个 SHA 记录 |
| 统计代码 | `39550bdc…`（scale_paired_ci）| 文件字节 SHA |
| config / budget / source_manifest / common_protocol / environment | `464c1f6f…` / `10fecf9d…` / `e25854e3…` / `2b9ce1ed…` / `3b6ec3bd…` | pin + provenance 交叉 |

### 6.2 WP1 密封核验（解封前置；`run` 阶段第一步，顺序不可换）

1. `audits/dev_baseline_sealed/manifest.json` SHA == `a1d29b74…db742b`，且 == ledger 最后一条 `seal_manifest` 的 `output_sha256`；
2. manifest `files` 中 5 个文件逐字节 SHA 复算 == 记录值（`v53mp_50k_C_s{17,29,43}_dev.json`、`r3_reference_dev.json`、`h2_move_guard_dev.json`）；
3. `audits/evaluator_before_after.json` SHA == `4ca61fdb…30514`，`status=="ok"`，`dev_sealed_status.pass==true`；
4. ledger 中 WP1 段（前 18 行）字节前缀 SHA 记入 freeze 的 `ledger_prefix_sha256`；WP7 追加后 `verify` 阶段复算前缀必须不变；
5. 任一不符 → `WP1_SEAL_DRIFT`，**拒绝一切 dev 读取**，不写任何 per-seed 产物。

> 语义：满足以上条件后，WP1 密封的 dev 详细指标在 WP7 语境下“解封”（本 WP7 报告可以引用其 R3 reference / 分母 sanity），但解封引用 ≠ 重新访问；引用只读已密封文件的 SHA 绑定内容。

### 6.3 screen 裁决门（`run` 阶段第二步）

- `E_star == "E0"` 且 `selected == []` 且 `n_selected == 0`；
- `candidates[*].outcome ∈ {scientific_failure, gate_skipped}`；
- 任何越界（例如 selected 非空、E_star≠E0）→ `SCREEN_ADJUDICATION_UNEXPECTED` fail-closed。

### 6.4 dev 语料门（打开语料后、前向之前）

- `pack_info.used == true`（dev feature pack 必须命中；否则 `DEV_PACK_UNAVAILABLE` fail-closed——与 WP1 同规则）；
- `len(dev_pos)==135926`、ok==`116527`、curves==`589`；
- `_canonical_positions_sha(dev_pos) == 3063dcf7…`、`devpack.positions_sha256(dev_pos) == 4f6c1e95…`、`_canonical_curves_sha(sorted(dev_curves)) == 6377ea96…`（全部 exact）。

---

## 7. 评估协议

### 7.1 模型加载（strict）

```python
state   = torch.load(ckpt, map_location="cpu", weights_only=False)
names   = {f.name for f in fields(V541TrainerConfig)}
cfg     = V541TrainerConfig(**{k: v for k, v in state["config"].items() if k in names})
model   = load_model_v541(cfg, r3_ckpt=R3_INIT_PATH, device=device)   # E0 → 无分支参数
model.load_state_dict(state["model"], strict=True)                    # 必须 strict
assert model.is_e0 and cfg.embedding_id == "E0"                       # 双保险
```

- `backbone_init=="r3"` 路径在 `load_model_v541` 内已做 R3 strict warmstart；随后整体 `strict=True` 保证 checkpoint 与构造完全一致；
- E0 无 degree stats（`degree_stats=null`），`_load_degree_stats_into` 自动跳过；
- 加载后立刻校验 `state["identity"]["seed"] == seed` 与 `identity["embedding_id"]=="E0"`，否则 `CKPT_IDENTITY_MISMATCH`。

### 7.2 tau（冻结、不重算）

- 唯一来源：`checkpoint_best.pt["tau_end_calibration"]["tau_end"]`（inner-val 校准，`tau_calibrated_on=="inner_val"`）；
- 与 `metrics.json["tau_end_calibration"]` 逐字段一致才放行；
- `_process_metrics(..., tau_end=tau)` 产生 `start_misstop_rate_at_tau` / `ts_misstop_rate_at_tau`；
- **禁止**在 dev 上重新 `_calibrate_tau_on`（代码路径根本不 import 该函数用于 dev）。

### 7.3 train_q_mean（只来自该 seed inner-train）

```python
pool_curves = _tier_pool_curve_ids(split_doc, "50k")
train_pos, _iv = v53mp_pool_split(samples, pool_curve_ids=pool_curves,
                                  seed=seed, inner_val_fraction=0.2)   # 训练同款切分
train_q_mean = _train_q_mean(samples, labels, train_pos)               # 常数 q 基线
```

- 该计算只读 inner-train 行的 q 标签（训练合法面），**不得**用 dev 行替代；
- 写入 per-seed 文件时同时记录 `n_rows_used`、`n_q_masked`、`derived_from_dev: false`；
- 后处理把 `constant_q_baseline_mae / constant_q_value / learnable_above_constant` 合入 `process.q_head`，公式逐字镜像 `evaluate_v53mp`（`key_atom_v53mp_train_v3.py:1660-1674`）：`const_mae = mean(|train_q_mean - q_true|)` 仅在 dev 的 q-masked 行上计算，`learnable_above_constant = mae_curve_mean < const_mae`。参照的 R3 MOVE 块直接引用 WP1 密封 `r3_reference_dev.json`（= `r3_eval_ckpt` zero-shot 于同一 dev 位置），**不重跑 R3 前向**。

### 7.4 单次前向 + 指标

- `_eval_split("dev", dev_pos, corpus, cache, device, model, collect_process=True, tau_end=tau, cost=cost)`：chunk 64 全 dev 位置 ONE pass（与历史 q/endpoint pass 同构成），随后 ok-only 64-chunk 构建 before/after g53；
- MOVE：`key_rmse_over_zero`（curve median RMSE/zero、cos_eff、coverage）、`sre_and_sign`（修复后 key-pair sign、active pair 分母、dead fraction）、`fixed_coverage_direction`、`pair_denominators`（input/before/after + ownership audit）；
- PROCESS：`q_head`（curve MAE/corr + 常数基线）、`ordering`（full/cross-TS 及分母）、`endpoint`（AUPRC/AUROC/prevalence/Brier 需要的 n_pos/n_neg、misstop@tau、n_start/n_ts）；
- 分母：`n_positions / n_positions_ok / n_curves / n_curves_move / positions_sha256 / curves_sha256 / n_forward_batches`；
- 结构 sanity：`_move_sanity` 七项；
- H2 描述块：`_h2_move_guard(e0_move, sealed_r3_move, split="dev", variant="after_repaired", seed=seed)` —— 参照是 WP1 密封的 `r3_reference_dev.json`（同 dev 位置、SHA 已核验），**复用不重跑 R3**；该 guard 是 control 记录，不是门（E0 是基线本身）。

### 7.5 三 seed 汇总规则

- 只报 per-seed + mean/min/max（描述性）；`new CI` 不计算（无候选对比对象）；
- 不做任何 seed 间选择、不设阈值、不回写 tau/配置；
- `adjudication_context.dev_veto_applicable=false`，注明 plan §10.1.4 的否决路径无升级对象；E0 留任为程序性结论（由 screen 已裁决，本步不产生新裁决）。

---

## 8. 失败语义（fail-closed 错误表）

| 错误码 | 触发 | 处理 |
|---|---|---|
| `FREEZE_MISSING` / `FREEZE_DRIFT_*` | freeze 不存在或任一绑定 SHA 不符 | 拒绝 run；不触 dev |
| `SCREEN_ADJUDICATION_UNEXPECTED` | E\*≠E0 / selected 非空 / 候选 outcome 越界 | 拒绝 run |
| `WP1_SEAL_*`（manifest/file/ledger 前缀/evaluator 漂移） | §6.2 任一失败 | 拒绝 run，不读 dev |
| `DEV_PACK_UNAVAILABLE` | dev pack 未命中 | 拒绝评估（与 WP1 同规则） |
| `DEV_DENOMINATOR_MISMATCH` | 位置/曲线 SHA 或计数与密封值不符 | 拒绝评估 |
| `CKPT_MISSING` / `CKPT_IDENTITY_MISMATCH` / `STRICT_LOAD_FAILED` | checkpoint 缺失/身份错/strict 失败 | 该 seed 失败 → 整体 status=failed（不留半成品 per-seed 文件） |
| `INNER_TRAIN_SPLIT_FAILED` | pool/train 切分不可重建 | 拒绝（train_q_mean 无法合法获得） |
| `CONFIRM_ACCESS_REFUSED` | 任何 split!="dev" 请求 | 结构性不可达；防御断言 + 测试覆盖 |
| `FORBIDDEN_CANDIDATE` | 任何 E1–E6 加载/前向企图 | 拒绝 |
| GPU OOM / cgroup OOM（exit 137/91） | 资源 | `resource_failure` 记录 + 停止；不降级 batch、不换设备重试读 dev |

失败时的产物规则：

- per-seed 文件**只在该 seed 全部指标计算成功后**原子写（临时文件 + rename）；失败不留 per-seed 文件；
- 汇总报告在任一 seed 失败时写 `status:"failed"` + `failure:{code, detail, seed}`（不含科学数值），并保留已成功 seed 的文件与 ledger 记录（append-only，不删除）；
- 一次 `run` 失败后**不自动重试**；重试必须由主代理显式发起并在 ledger 留下新 attempt 记录（`entry_id` 含毫秒时间戳天然区分）；
- 全程 `confirm_read=false`、`new_QC_calls=0`、`training_runs_started=0` 写入每份产物。

---

## 9. 并发与资源语义

- **单路**：整个 `run` 在 `scripts/infra/cgroup_scope.sh 12G 10G <outdir>` 内执行（MemoryMax=12G、MemoryHigh=10G、MemorySwapMax=0）；WSL2 无 systemd 时用 `--fallback`；
- **GPU 锁**：进程内先取 `/tmp/opencode/pes2ts_gpu.lock`（复用 `_GpuLock`），防止与其他会话抢 8G 卡；seed 间串行，`_free_cuda()` 逐个释放；
- **ledger 追加锁**：`fcntl.flock`（EX）包住 appender，防止并发会话交错写行；行格式与 WP1 完全一致；
- **内存路径**：columnar + lazy labels + dev pack（`DevFeaturePackView`），禁止 v1 全量缓存；per-seed 评估后立即 `del model/state` 释放；
- **只读面**：所有输入（checkpoints、sealed、split、config、plan）只读；写面仅 `reports/wp7_freeze.json`、`reports/dev_frozen/**`、`reports/dev_frozen_evaluation.json`、`audits/dev_access_ledger.jsonl`（追加）、`audits/cgroup/wp7_*/`（采样器输出）。

---

## 10. 与 v541_run / WP1 / freeze 的一致性约束

1. **零侵入**：`v541_dev_evaluate.py` 对 WP1/runner 的关系是“同口径消费者”——import 复用函数，不改其源文件；`git diff` 在实现后必须显示只新增本脚本与测试（及 reports 产物）。
2. **同口径评估**：预测构成（chunk 64 全位置）、g53 修复工具（`assign_batch_pairs`）、`_process_metrics` 分组、H2 公式、sanity 七项——全部与 WP1 dev stage 逐字节同源；这保证 WP7 与密封基线可直接对照。
3. **同分母**：dev 位置/曲线 SHA 与 WP1 密封 exact 相等（§6.4），使 per-seed 指标与密封 R3 reference / 历史 C 基线在同一样本集上可比。
4. **runplan 不变**：不新增 run class；WP7 评估不是训练 run，不写 `run_records.jsonl`、不占 `N_core` 计数；`plan/runplan.pin.json` 与 `runplan.jsonl` 字节不变。
5. **seed 语义**：`seed` 只决定 inner-train/inner-val 切分与 checkpoint 身份；dev 切分与 seed 无关（同一 135,926 位置），三 seed 是同一 dev 上的三次模型评估。
6. **账本连续性**：`dev_access_ledger.jsonl` 是 WP1+WP7 共享的 append-only 事实源；WP7 行以 `purpose:"candidate_evaluation"` 与 WP1 行区分，前缀不变性在 `verify` 阶段强制。
7. **时间序**：`freeze`（无 dev）→ 核验 → 每 seed 访问 → 汇总；`wp7_freeze.json.created_at` 必须早于任何 candidate_evaluation ledger 行的 `ts`。

---

## 11. 测试设计（`tests/campaigns/test_v541_dev_evaluate.py`，全部 fixture，无 GPU、无真实 dev）

| # | 用例 | 断言 |
|---|---|---|
| T1 | freeze schema 完整性 | 必填键齐全；E0×3、tau×3、runplan/code/stats SHA 均在；`dev_access.type=="candidate_evaluation"`；`confirm.status=="sealed"` |
| T2 | screen 裁决门 | E\*=E0 + selected=[] 通过；篡改 E\* 或 selected 非空 → `SCREEN_ADJUDICATION_UNEXPECTED` |
| T3 | WP1 密封核验通过路径 | fixture manifest + 文件 + ledger 前缀 → 通过 |
| T4 | WP1 密封漂移 | 改任一 sealed 字节 / ledger 插行 → `WP1_SEAL_*`，且断言未发生任何 `_build_tier`/前向调用（monkeypatch 计数） |
| T5 | E0-only 强制 | `load_e0_model` 遇 embedding_id≠E0 → 拒绝；`FORBIDDEN_CANDIDATE` 路径覆盖 E1–E6 |
| T6 | strict load | tiny V541Model checkpoint 往返：strict 成功；删一个 key → `STRICT_LOAD_FAILED` |
| T7 | tau 来源 | checkpoint tau 传入 `_process_metrics` 后 misstop 块使用该值；代码路径无 `_calibrate_tau_on` 调用（grep 断言） |
| T8 | train_q_mean 来源 | 合成 samples/labels：tqm 等于 inner-train 行均值且 ≠ dev 行均值；`derived_from_dev is False` |
| T9 | 常数 q 基线块 | 合成 q 数组验证 `constant_q_baseline_mae` / `learnable_above_constant` 公式 |
| T10 | 分母门 exact | fixture 分母 SHA 不符 → `DEV_DENOMINATOR_MISMATCH` |
| T11 | ledger 行 schema | purpose/wp/curve+position SHA/code SHA/输出 SHA/时间齐全；`json.dumps(sort_keys=True)` 字节格式可被 WP1 解析器回读 |
| T12 | ledger 前缀不变 | append 后前 N 行字节 == 原文件；新增行 purpose 全为 candidate_evaluation |
| T13 | 汇总无选择语义 | aggregate 只含 per-seed/mean/min/max；`selection_or_tuning_performed is False`；`dev_veto_applicable is False` |
| T14 | confirm 硬拒 | 构造 split="confirm" 请求 → 结构性拒绝；所有产物 `confirm_read=False` |
| T15 | 失败不留半成品 | seed 中途抛错 → 无该 seed 的 per-seed 文件；汇总 status=failed |
| T16 | CLI 门 | `run` 无 freeze → 拒；`--limit-positions` 无环境旗标 → 拒 |

运行：`python -m pytest tests/campaigns/test_v541_dev_evaluate.py -q`。

---

## 12. 执行与成本预估

```bash
# 1) 冻结（无 dev 访问）
python scripts/campaigns/v541_dev_evaluate.py freeze

# 2) 评估（单路 cgroup；WSL2 用 --fallback）
scripts/infra/cgroup_scope.sh [--fallback] 12G 10G outputs/pesn/v541_embed_50k/audits/cgroup/wp7_dev_eval \
  python scripts/campaigns/v541_dev_evaluate.py run --seeds 17,29,43 --device cuda

# 3) 只读复核
python scripts/campaigns/v541_dev_evaluate.py verify
```

成本量级（依据 WP1 dev stage 实测：4 次模型 pass 共 8496 batches / 181.6 s GPU wall / 峰值 RSS 6.2 GiB）：

| 项 | 估计 |
|---|---|
| 前向 | 3 × 2124 batches ≈ 6372 batches，GPU wall ≈ 140 s |
| g53/G52 + q/ordering/endpoint 指标 | 每 seed 数十秒（ordering 为 dev 规模主要 CPU 项） |
| 合计 wall | ≈ 8–12 min（单路，含加载与 pack 打开） |
| 峰值 RSS | ≈ 6 GiB（< 10G high / 12G max） |
| GPU | RTX 4060 Ti 8G 单卡，占用 ≈ 1–2 GiB（WP1 实测 gpu_peak ≈ 87 MB + 框架开销） |
| new_QC_calls | 0 |

---

## 13. 产物与验收清单

- [ ] `reports/wp7_freeze.json` 存在且 schema/SHA 矩阵全部复算一致（先于一切 dev 访问）；
- [ ] WP1 密封 5 文件 + manifest + evaluator 报告 SHA 未变；ledger 前 18 行前缀不变；
- [ ] 3 个 per-seed 明细：strict load ok、tau 来源标注、train_q_mean 来源标注、MOVE/PROCESS/分母齐全、分母六门 exact；
- [ ] `reports/dev_frozen_evaluation.json`：status=ok（或 failed 如实）、aggregate 仅描述性、adjudication_context 声明无选择/无调参/无 veto 对象；
- [ ] ledger 新增 4 条 candidate_evaluation 行（3 seed + aggregate），WP1 行零改动；
- [ ] `confirm_read=false`、`new_QC_calls=0`、无 E1–E6 前向、无训练；
- [ ] `git status`：新增仅 `v541_dev_evaluate.py`、`test_v541_dev_evaluate.py`、reports 产物（及 cgroup 采样）；禁改清单文件零 diff。
