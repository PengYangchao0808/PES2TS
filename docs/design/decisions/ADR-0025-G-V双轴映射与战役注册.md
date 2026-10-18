# ADR-0025：G–V 双轴映射与战役注册

> 状态：Accepted
> 日期：2026-10-18
> 层级：治理（G0–G4 能力轴 / V 系列战役轴）
> 取代关系：不取代任何既有 ADR；把散落在 V4/V5 文档中的“归属 G3.P”显式化为可校验映射
> 上游依赖：[开发宪法](../../PES2TS_开发宪法_v1.md)、[阶段定义 G0–G4](../../plans/PES2TS_阶段定义_G0-G3.md)
> 下游消费者：V5–V7 实施、RunRecord、model card、gate report、文档索引
> 当前版本：v1.0 / 2026-10-18

## 1. 背景

仓库同时使用两套编号：

- **G0–G4** 描述代码/物理能力的成熟度（含 G2.T、G3.P.0–.3、G3.1、G3.2、G4.0–G4.2）。
- **V4.2–V4.7、V5.0–V5.2、V6、V7-A/B/C** 描述一次训练/测试战役。

此前 V4/V5 文档只在正文里写“归属 G3.P”，代码中既没有 `g_stage`，也没有 `campaign_id` 与 G 阶段的绑定；RunRecord 只绑定 config/dataset/split hash。结果是：

1. V5/V6/V7 容易被误读为新的 G 阶段（或 G5）；
2. V5 的 `v5_bc.py`/`v6_train.py` 把战役编号写进代码名，下一轮 V5.2 复用同一能力时产生分叉；
3. V7 执行/闭环代码可能被放进 `pesn/offline/`，破坏 G2/G4 归属；
4. 报告无法回答“这次 run 到底验证了哪个 G 能力、属于哪个 V 战役”。

## 2. 决策

### 2.1 两条正交轴

- **G = 代码/物理能力**：拥有模块、契约、出口门；不随实验编号改名。
- **V = 训练/测试战役**：拥有 config、冻结数据 manifest、run 集合、gate report；不拥有新的代码根。

V5–V7 不是新阶段：V5 主体属 **G3.P**，V6 属 **G3.2**，V7-A/B 属 **G4.1**，V7-C 属 **G4.0→G4.1**；M0.5/M6.5 属 **G2.T**；G4.2 发布门不变。完整表见 [G–V 映射](../PES2TS_G_V_MAPPING_v1_20261018.md)。

### 2.2 单一映射源

`config/registry/g_v_map.yaml` 是唯一机器可读源；`docs/design/PES2TS_G_V_MAPPING_v1_20261018.md` 由 `scripts/render_gv_map.py` 生成。禁止手改生成文档。

每条 V 战役声明：

```yaml
primary_g: G3.P.2          # 唯一主归属
supporting_g: [G0, G1]     # 数据/执行/验证支撑
data_dependencies: [G0, G1]
code_scope: [...]          # 该战役允许改动的代码路径
config: config/campaigns/v5_1_architecture.yaml
data_manifest: data/manifests/v5/split_manifest.json
gates: [V5-G1, V5-G2, V5-G3, V5-G4]
status: proposed|active|executed|archived
```

### 2.3 代码/配置命名

- `pes2ts_core/**` 按 G 能力命名：`key_atom_egnn.py`、`xtb_freeze_transport.py`、`key_atom_bc.py`。
- `config/campaigns/**`、`scripts/campaigns/**`、`outputs/campaigns/**`、`docs/reports/**` 按 V 战役命名：`v5_1_architecture.yaml`、`run_v5_sprint.py`。
- 禁止在 `pes2ts_core/**` 新建 `v5/`、`v6/`、`g3p/` 顶层包；阶段是标签，不是目录。
- V4 `dual_ts_*`/`direct_*` 保留为 `legacy-v4`/`comparison_only`；V5 主结论不得用其 checkpoint 初始化。

### 2.4 RunRecord 必须携带 G/V 身份

新 V5+ run 的 `run.json` 必须包含：

```json
"gv": {
  "map_version": "2026-10-18.1",
  "map_sha256": "...",
  "campaign_id": "V5.1",
  "primary_g": "G3.P.2",
  "supporting_g": ["G0", "G1"],
  "code_scope": ["pes2ts_core/pesn/models/key_atom_*.py"],
  "gates": ["V5-G1", "V5-G2", "V5-G3", "V5-G4"],
  "code_commit": "...",
  "config_sha256": "...",
  "data_manifest_sha256": "..."
}
```

`pes2ts_core.gv_registry.validate_gv_run_record` 校验 campaign、primary、supporting、gates、map version 与 digest。V4 旧记录可省略该块；V5 起必须提供。

### 2.5 CI 门

- `python scripts/render_gv_map.py --check`：生成文档必须与 YAML 一致。
- `pytest tests/test_gv_registry.py`：阶段引用、路径存在性、run record 校验。
- `status: active/executed` 的 `code_scope`、`config`、`data_manifest`、`evidence` 必须存在；`proposed` 允许规划中的路径暂缺。
- 每个 V 战役只有一个 `primary_g`；`supporting_g` 必须已登记为 G 阶段。

## 3. 后果

**正面**

- V5/V6/V7 的代码落点、配置、报告和 run 记录可追溯到唯一 G 能力；
- 避免 V5 成为第四个平行家族；V7 执行代码回到 `generation/`/`integration/`；
- 文档索引、台账、ADR 与代码命名使用同一套 ID；
- 队友在 review 时可以只检查 `code_scope` 和 `gates`，不需要读完整计划。

**代价**

- 新增一个约 200 行的纯标准库注册器和一个 YAML 维护点；
- 每次新增 V 战役或改变 G 归属，需要先改 YAML、再跑 `--write`；
- V4 旧文档的“当前”措辞需要一次性标注为 `executed`/`frozen`。

## 4. 不采用 alternatives

- **把 V 编号写进代码包名**：会让同一能力在 V5.1/V5.2/V6 之间复制实现，拒绝。
- **只写文档、不做机器校验**：映射会随文档漂移，无法阻止 V7 代码落入 `pesn/offline`，拒绝。
- **把 V5–V7 当作新 G 阶段（G5）**：违反阶段定义“PES-N 不是新增 G5”，且会绕过 G3.2/G4.0/G4.1 的出口门，拒绝。

## 5. 关联

- [G–V 映射表](../PES2TS_G_V_MAPPING_v1_20261018.md)
- [G–V 映射 YAML](../../../config/registry/g_v_map.yaml)
- [V5–V7 架构计划 v1.2](../../plans/PES2TS_PES-N_V5-V7架构调整与实施计划_v1.2_20261018.md)
- [V5.0–V5.2 冲刺方案](../../plans/PES2TS_V5.0-V5.2_十小时冲刺训练与验证执行方案_v1_20261018.md)
- [阶段定义 G0–G4](../../plans/PES2TS_阶段定义_G0-G3.md)
