"""Stage static case sidecars into a round dir that only ran execute phases.

The registry support gate resolves per-reaction coordinate specs from
``<round_dir>/cases/<rid>.json`` sidecars, which the collect phase writes.
An execute-only round (dev checkpoints) never collected, so the sidecars
must be staged from the same frozen manifest + legacy plan root the
executor loads cases from — pure static data, zero QC.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pes2ts_core.generation.campaign import load_campaign_manifest  # noqa: E402
from pes2ts_core.generation.control_round import (  # noqa: E402
    _case_sidecar_payload,
    _load_round_case,
    build_scan_policy,
    load_control_round_config,
)
from pes2ts_core.pesn.offline.transitions import write_case_sidecar  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--round-config", type=Path, default=REPO_ROOT / "config/pesn/scale_rl_r1.yaml")
    args = parser.parse_args()

    config = load_control_round_config(args.round_config)
    scan_policy = build_scan_policy(config)
    manifest = load_campaign_manifest(args.manifest)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    written = 0
    for reaction_id in selection["reaction_ids"]:
        case = _load_round_case(
            reaction_id=reaction_id,
            manifest=manifest,
            manifest_path=args.manifest,
            legacy_root=args.legacy_root,
            config=config,
            scan_policy=scan_policy,
        )
        write_case_sidecar(args.round_dir, _case_sidecar_payload(case))
        written += 1
    print(f"staged {written} case sidecars into {args.round_dir / 'cases'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
