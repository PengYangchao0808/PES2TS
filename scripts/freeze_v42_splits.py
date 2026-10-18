#!/usr/bin/env python3
"""V4.2 internal dev split freeze (Lane A, contract §3).

Calls ``dual_ts_dev_split.build_dev_split`` on the splits lineage + the native
rebuild manifest, and writes ``wp42_campaign/dev_split.json``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.offline.data.dual_ts_dev_split import (  # noqa: E402
    build_dev_split,
)

DEFAULT_ROOT: str = "outputs/pesn/dual_ts_v2_20261015"
DEFAULT_LINEAGE: str = f"{DEFAULT_ROOT}/splits/dual_ts_split_lineage_v1.json"
DEFAULT_NATIVE: str = f"{DEFAULT_ROOT}/wp42_campaign/native_features/features_manifest.json"
DEFAULT_OUT: str = f"{DEFAULT_ROOT}/wp42_campaign/dev_split.json"
DEFAULT_DUP_GROUPS: str = "data/interim/duplicate_groups.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineage", default=DEFAULT_LINEAGE)
    ap.add_argument("--native-manifest", default=DEFAULT_NATIVE)
    ap.add_argument("--duplicate-groups", default=DEFAULT_DUP_GROUPS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    payload = build_dev_split(
        args.lineage, args.native_manifest, args.out,
        duplicate_groups_path=args.duplicate_groups,
    )
    print(f"[freeze_v42_splits] wrote {args.out}")
    print(f"[freeze_v42_splits] n_clusters={payload['n_clusters']} "
          f"dev={payload['dev_group_ids']} "
          f"dev_insertion_count={payload['dev_insertion_count']} "
          f"replacement={payload['replacement']}")
    print(f"[freeze_v42_splits] split_sha256={payload['split_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
