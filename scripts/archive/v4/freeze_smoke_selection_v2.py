"""Freeze the smoke-round selection (prepared-plan subset of the smoke cohort)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pes2ts_core.generation.control_round import build_selection_manifest  # noqa: E402

CAMPAIGN = (
    REPO_ROOT
    / "outputs/pesn/scale20k_rl2000_v1_20261008/repair2/origin_prep_smoke/campaign_manifest.json"
)


def main() -> int:
    prepared = sorted(
        path.parent.name for path in CAMPAIGN.parent.glob("*/plan.json")
    )
    print(f"prepared plans: {len(prepared)}")
    build_selection_manifest(
        reaction_ids=prepared,
        campaign_manifest_path=CAMPAIGN,
        output_path=CAMPAIGN.parent / "selection_smoke.json",
        purpose="smoke_not_counted",
        note=(
            "1+4 engineering smoke episodes (plan §4 S1): prepared-plan "
            "subset of the frozen smoke cohort; origin-preparation failures "
            "stay in the funnel ledger; episodes are never counted toward "
            "the RL2000 denominator and never used for BC/Q fitting"
        ),
    )
    print(f"smoke selection frozen: {len(prepared)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
