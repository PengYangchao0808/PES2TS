#!/usr/bin/env python3
"""Inherit the v1 direct split into the dual-TS v2 lineage (WP0b, plan v4 §E).

Runs the real inheritance from the frozen v1 split files (READ-ONLY on them)
and writes the new lineage artifact
``pesn_dual_ts_split_lineage_v1``.  Historical group/family/exposure
membership is inherited without re-randomization; groups absent from the old
split land in the explicit unassigned bucket.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pes2ts_core.pesn.offline.splits.dual_ts_split import (  # noqa: E402
    DUAL_TS_SPLIT_LINEAGE_SCHEMA,
    DualTSSplitLineageError,
    build_dual_ts_split_lineage,
    validate_lineage_payload,
    write_lineage_artifact,
)

DEFAULT_V1_SPLITS_DIR = "outputs/pesn/direct_egnn_v1_20261015/splits"
DEFAULT_OUT = ("outputs/pesn/dual_ts_v2_20261015/splits/"
               "dual_ts_split_lineage_v1.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-splits-dir", default=DEFAULT_V1_SPLITS_DIR,
                        help="directory with pesn_irc_split.json and the "
                             "frozen selection_*.json files (read-only)")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="output lineage artifact path")
    parser.add_argument("--groups-source", default=None,
                        help="optional JSON listing new-task reaction group "
                             "ids (checked against v1; absent groups are "
                             "recorded unassigned, never silently assigned)")
    args = parser.parse_args(argv)

    try:
        payload = build_dual_ts_split_lineage(
            args.v1_splits_dir, args.groups_source)
        out_path = write_lineage_artifact(payload, args.out)
        # round-trip: re-read the written file and validate against sources
        written = json.loads(Path(out_path).read_text(encoding="utf-8"))
        validate_lineage_payload(written,
                                 verify_sources_in=Path(args.v1_splits_dir))
    except DualTSSplitLineageError as exc:
        print(f"REFUSED {exc}", file=sys.stderr)
        return 2

    partitions = payload["partitions"]
    counts = payload["audit"]["counts"]
    print(f"schema: {payload['schema_version']}  "
          f"tool: {payload['tool_version']}")
    print(f"sources consumed: {len(payload['sources'])} "
          f"(all sha256-bound)")
    print("inherited partitions (reaction_group_id keyed, no re-randomization):")
    for role in ("fit", "monitor", "test", "family_holdout"):
        block = partitions[role]
        print(f"  {role:15s} groups={block['n_groups']:5d} "
              f"branch_potential={block['branch_potential']:5d} "
              f"(v1 source: {block['v1_source']})")
    wave = payload["dual_ts_wave_selections"]["wave"]
    print("dual-TS first-wave selections:")
    for role in ("fit", "monitor", "test"):
        block = wave[role]
        print(f"  {role:8s} <- {block['selection']}: "
              f"n_groups={block['n_groups']} "
              f"n_reaction_ids={block['n_reaction_ids']}")
    unassigned = payload["unassigned"]
    reasons: dict[str, int] = {}
    for entry in unassigned:
        reasons[entry["reason"]] = reasons.get(entry["reason"], 0) + 1
    print(f"unassigned groups: {len(unassigned)} "
          f"(reasons: {reasons if reasons else '{}'})")
    print(f"binding: {counts['n_reactions_bound']} reactions "
          f"({counts['n_selection_evidenced_reactions']} selection-evidenced, "
          f"{counts['n_identity_bound_reactions']} identity-bound)")
    print(f"isolated-but-partitioned (inherited, flagged): "
          f"{len(payload['audit']['isolated_but_partitioned_reaction_ids'])}")
    print(f"artifact: {out_path}")
    print(f"artifact_sha256: {payload['artifact_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
