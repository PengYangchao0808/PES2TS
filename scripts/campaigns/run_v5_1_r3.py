"""30h-campaign R3 driver entrypoint (WP-L3 residual #4).

Reused V5.1 drivers (``run_v5_1_sprint`` / ``run_v5_1_r1`` / ``r1_*``) default
their output root to the SEALED directories (``outputs/pesn/v5_1/**``,
``outputs/pesn/v5_1_r1/**``).  This R3 entrypoint does the opposite by
construction (``parallel_config.driver_campaign_root_rule``):

* the DEFAULT output root is ``outputs/pesn/30h_campaign/r3_runs``;
* ``assert_campaign_root`` runs BEFORE anything else -- sealed prefixes and
  anything outside the campaign root refuse fail-closed (exit 1);
* the heavy R3 training wiring (W-LOSS wave 2, key_atom_bc_v3) is imported
  lazily only after the root assertion passes, so a misconfigured root can
  never open an output file.

Usage::

    python scripts/campaigns/run_v5_1_r3.py --check                 # assert default root
    python scripts/campaigns/run_v5_1_r3.py --output-root <root>    # assert explicit root
    python scripts/campaigns/run_v5_1_r3.py -- <driver args...>     # forward to the trainer
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agent_ops.supervisor.campaign_root import (  # noqa: E402
    CAMPAIGN_ROOT,
    CampaignRootError,
    assert_campaign_root,
)

#: WP-L3: R3 driver output root -- inside the campaign, NEVER the sealed dirs.
DEFAULT_OUTPUT_ROOT = CAMPAIGN_ROOT + "/r3_runs"


def resolve_output_root(value: Optional[str]) -> str:
    """Assert the output root is inside the campaign; return the normalized key."""
    return assert_campaign_root(value if value is not None else DEFAULT_OUTPUT_ROOT)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_v5_1_r3",
        description="30h R3 driver entrypoint: output root defaults to "
        "{}/r3_runs and sealed prefixes (v5_1/v5_1_r1) refuse fail-closed".format(
            CAMPAIGN_ROOT
        ),
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="output root (default: {}; must stay inside the campaign)".format(
            DEFAULT_OUTPUT_ROOT
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="only assert the output root and print it (no training)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    # split off forwarded driver args at "--"
    forwarded: List[str] = []
    if argv is None:
        argv = sys.argv[1:]
    if "--" in argv:
        idx = argv.index("--")
        argv, forwarded = argv[:idx], argv[idx + 1 :]
    args = _build_parser().parse_args(argv)
    try:
        root = resolve_output_root(args.output_root)
    except CampaignRootError as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 1
    if args.check or not forwarded:
        print(
            json.dumps(
                {"ok": True, "output_root": root, "forwarded_args": forwarded},
                sort_keys=True,
            )
        )
        return 0
    # Training wiring lands with W-LOSS wave 2 (key_atom_bc_v3); import only
    # AFTER the root assertion so a bad root can never touch an output file.
    sys.stderr.write(
        "error: R3 training driver not yet wired (W-LOSS wave 2); "
        "output root asserted OK: {}\n".format(root)
    )
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
