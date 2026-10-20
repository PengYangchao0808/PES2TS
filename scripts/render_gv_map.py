#!/usr/bin/env python
"""Render the human-readable G–V mapping from ``config/registry/g_v_map.yaml``.

Usage:
    python scripts/render_gv_map.py            # print Markdown
    python scripts/render_gv_map.py --write    # write the canonical doc
    python scripts/render_gv_map.py --check    # fail if the doc is stale

The YAML file is the single source of truth; this script only renders it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.gv_registry import (  # noqa: E402
    GVMap,
    GVMapError,
    load_gv_map,
    validate_gv_map,
)

TARGET = PROJECT_ROOT / "docs" / "design" / "PES2TS_G_V_MAPPING_v1_20261018.md"


def _cell(campaign, stage_id: str) -> str:
    if campaign.primary_g == stage_id:
        return "P"
    if stage_id in campaign.supporting_g:
        return "S"
    if stage_id in campaign.data_dependencies:
        return "D"
    return ""


def render_markdown(gv_map: GVMap) -> str:
    lines: list[str] = []
    lines.append("# PES2TS G–V 映射（G = 代码/物理能力，V = 训练/测试战役）")
    lines.append("")
    lines.append("> 状态：权威（机器可读源：`config/registry/g_v_map.yaml`）")
    lines.append(f"> 映射版本：`{gv_map.map_version}`")
    if gv_map.plan_revision:
        lines.append(f"> 计划修订：`{gv_map.plan_revision}`")
    lines.append("> 生成命令：`python scripts/render_gv_map.py --write`")
    lines.append("> 校验命令：`python scripts/render_gv_map.py --check`")
    lines.append("")
    lines.append("## 0. 读法")
    lines.append("")
    lines.append("- **G** 是代码/物理能力阶段，拥有模块、契约和出口门；不随实验编号改名。")
    lines.append("- **V** 是一次训练/测试战役，拥有 config、冻结数据 manifest、run 集合和报告。")
    lines.append("- `P` = primary_g，`S` = supporting_g，`D` = data dependency。")
    lines.append("- 代码按 G 能力命名（`key_atom_*`、`xtb_freeze_*`），配置/输出/报告按 V 战役命名（`v5_1_*`）。")
    lines.append("- 每次 run 的 `gv` 块必须记录 `campaign_id`、`primary_g`、`supporting_g`、`gates`、`map_version`。")
    lines.append("")
    lines.append("## 1. 阶段 × 战役矩阵")
    lines.append("")
    stage_ids = list(gv_map.stages)
    lines.append("| V campaign | " + " | ".join(stage_ids) + " |")
    lines.append("|---|" + "|".join("---" for _ in stage_ids) + "|")
    for campaign in gv_map.campaigns.values():
        cells = [_cell(campaign, stage_id) for stage_id in stage_ids]
        lines.append(f"| {campaign.campaign_id} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## 2. 战役登记")
    lines.append("")
    lines.append("| V | 状态 | primary G | supporting G | config | data manifest | gates |")
    lines.append("|---|---|---|---|---|---|---|")
    for campaign in gv_map.campaigns.values():
        supporting = ", ".join(campaign.supporting_g) or "—"
        config = f"`{campaign.config}`" if campaign.config else "—"
        manifest = f"`{campaign.data_manifest}`" if campaign.data_manifest else "—"
        gates = ", ".join(campaign.gates)
        lines.append(
            f"| {campaign.campaign_id} | {campaign.status} | {campaign.primary_g} | "
            f"{supporting} | {config} | {manifest} | {gates} |"
        )
    lines.append("")
    lines.append("## 3. G 阶段登记")
    lines.append("")
    lines.append("| G | 能力 | code globs | 阶段文档 |")
    lines.append("|---|---|---|---|")
    for stage in gv_map.stages.values():
        globs = "<br>".join(f"`{item}`" for item in stage.code_globs) or "—"
        doc = f"`{stage.stage_doc}`" if stage.stage_doc else "—"
        lines.append(f"| {stage.stage_id} | {stage.title} | {globs} | {doc} |")
    lines.append("")
    lines.append("## 4. 规则")
    lines.append("")
    lines.append("1. `proposed` 允许代码/配置/数据路径尚不存在；`active`/`executed` 必须存在，由 `validate_gv_map` 强制。")
    lines.append("2. 同一 V 战役只有一个 `primary_g`；跨阶段工作放 `supporting_g`，不得把 V7 执行代码塞进 `pesn/offline`。")
    lines.append("3. V4 `dual_ts`/`direct` 只作 `comparison_only`，不得被 V5/V6/V7 主结论复用为初始化权重。")
    lines.append("4. 任何 run record 的 `gv.campaign_id` 必须能在本表解析；`map_version` 必须与当前映射一致。")
    lines.append("5. 新增/修改 V 战役先改 YAML，再跑 `--write`，最后提交生成文档；不要手改本文件。")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", action="store_true", help="write the canonical doc")
    group.add_argument("--check", action="store_true", help="fail if the doc is stale")
    args = parser.parse_args(argv)

    try:
        gv_map = load_gv_map()
        validate_gv_map(gv_map)
    except GVMapError as exc:
        print(f"G–V map error: {exc}", file=sys.stderr)
        return 2

    rendered = render_markdown(gv_map)
    if args.write:
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        TARGET.write_text(rendered, encoding="utf-8")
        print(f"wrote {TARGET.relative_to(PROJECT_ROOT)}")
        return 0
    if args.check:
        if not TARGET.is_file():
            print(f"missing generated doc: {TARGET}", file=sys.stderr)
            return 1
        current = TARGET.read_text(encoding="utf-8")
        if current != rendered:
            print(
                "G–V mapping doc is stale; run "
                "`python scripts/render_gv_map.py --write`",
                file=sys.stderr,
            )
            return 1
        print("G–V mapping doc is current")
        return 0
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
