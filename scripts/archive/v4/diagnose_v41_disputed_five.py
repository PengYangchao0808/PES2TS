#!/usr/bin/env python3
"""WP3 v4.1 item 7 — derive the 5 endpoint-bond-disputed reaction ids.

Derivation rule: a reaction is in the disputed-5 iff its feature shard
carries ``bond_source_a == "covalent_fallback"`` OR ``bond_source_b ==
"covalent_fallback"``.  Those flags are written by
``scripts/build_dual_ts_features.py`` (``build_reaction_shard`` -> meta
``bond_source_a/b`` + ``disputed`` bool) from ``build_topology_features`` /
``consumed_endpoint_graphs`` in ``pes2ts_core/pesn/features/dual_ts_graph.py``:
a task with ``eligibility.bond_authority == "disputed"`` consumes the
deterministic covalent-radius geometry perception on BOTH endpoint frames
(same Cordero 2008 single-bond radii + 0.45 A tolerance as the
``dual_ts_preprocess`` fallback); the frozen rule string is
``covalent_radius_fallback_on_disputed_authority_v0``
(``pes2ts_core/pesn/features/dual_ts_spec.py``).

This script writes ``disputed_five_v41.json`` (ids + evidence flags + rule)
consumed by the WP4 50-vs-45 contrast.  Zero new QC; reads shards only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from diagnose_v41_common import (
    DISPUTED_IDS, NEW_FIT_FEATURES, OUT_DIR, write_json,
)


def disputed_from_shards(features_dir: Path) -> list[str]:
    ids = []
    for p in sorted(features_dir.glob("features_RXN_*.npz")):
        z = np.load(p, allow_pickle=False)
        sa = str(z["bond_source_a"])
        sb = str(z["bond_source_b"])
        if sa == "covalent_fallback" or sb == "covalent_fallback":
            ids.append(str(z["reaction_id"]))
    return sorted(ids)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", default=str(NEW_FIT_FEATURES))
    ap.add_argument("--out", default=str(OUT_DIR / "disputed_five_v41.json"))
    ap.add_argument("--md", default=str(OUT_DIR / "disputed_five_v41.md"))
    args = ap.parse_args(argv)

    from_shards = disputed_from_shards(Path(args.features_dir))
    result = {
        "schema_version": "pesn_v41_disputed_five_v1",
        "title": "WP3 v4.1 item 7: the 5 endpoint-bond-disputed reactions",
        "rule": {
            "flag": "bond_source_a/b == 'covalent_fallback' in the feature shard",
            "authority_flag": "task eligibility.bond_authority == 'disputed'",
            "consumption": "covalent_radius_single_link_v0 on BOTH endpoint "
                           "frames; Cordero 2008 single-bond radii; 0.45 A "
                           "tolerance (same as dual_ts_preprocess fallback)",
            "frozen_rule_string": "covalent_radius_fallback_on_disputed_authority_v0",
            "cite": [
                "scripts/build_dual_ts_features.py (build_reaction_shard, "
                "bond_source_a/b + disputed meta)",
                "pes2ts_core/pesn/features/dual_ts_graph.py "
                "(consumed_endpoint_graphs, DISPUTED_BOND_AUTHORITY_FLAG)",
                "pes2ts_core/pesn/features/dual_ts_spec.py (DISPUTED_BOND_RULE)",
            ],
        },
        "回答什么": [
            "从 shard bond_source 回退旗标派生的 5 个映射争议反应 id + 证据旗标 + 规则",
            "供 WP4 50-vs-45 对照使用的完整分母与原始数据保留",
        ],
        "不能声称什么": [
            "回退是'可追溯处置'，不是原映射/键级已正确的证明",
            "身份争议未因回退而被化学证明消除",
        ],
        "disputed_reaction_ids": from_shards,
        "evidence_flags": {
            "from_shard_bond_source_fallback": from_shards,
            "frozen_contract_ids": list(DISPUTED_IDS),
        },
        "consistency": {
            "shard_derived_matches_frozen": from_shards == sorted(DISPUTED_IDS),
        },
        "wp4_contrast_usage": "train A config with all 50 groups vs with the 5 "
                              "disputed groups excluded (45), SAME evaluation "
                              "set, full denominator preserved.",
    }
    write_json(Path(args.out), result)

    md = ["# WP3 v4.1 条目7：5 个映射争议反应", "",
          "| 回答什么 | 不能声称什么 |",
          "|---|---|",
          "| shard bond_source 回退旗标派生的 5 个争议反应 id + 规则 | 回退≠原映射/键级已正确 |",
          "| 供 WP4 50-vs-45 对照的完整分母 | 身份争议未被化学证明消除 |",
          "",
          f"- disputed-5 = {from_shards}",
          f"- 与冻结合同 id 一致：{result['consistency']['shard_derived_matches_frozen']}",
          f"- 规则：{result['rule']['frozen_rule_string']}"]
    Path(args.md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md).write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[item7] disputed-5 = {from_shards}")
    print(f"[item7] wrote {args.out} and {args.md}")
    if not result["consistency"]["shard_derived_matches_frozen"]:
        print("[item7] WARNING: shard-derived ids differ from frozen contract "
              "ids", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover
        print(f"DIAGNOSIS REFUSAL {exc!r}", file=sys.stderr)
        raise SystemExit(2)
