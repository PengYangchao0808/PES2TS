"""Freeze the smoke-round and main-round selection manifests (S1/S2 inputs)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pes2ts_core.generation.control_round import build_selection_manifest  # noqa: E402

CAMPAIGN = (
    REPO_ROOT
    / "outputs/pesn/scale20k_rl2000_v1_20261008/rl_0050/origin_prep_smoke/campaign_manifest.json"
)
COHORTS = (
    REPO_ROOT / "outputs/pesn/scale20k_rl2000_v1_20261008/cohorts/rl_cohorts.json"
)


def main() -> int:
    cohorts = json.loads(COHORTS.read_text(encoding="utf-8"))
    smoke = list(cohorts["rl_smoke_5_not_counted"])
    fit50 = list(cohorts["rl_fit_batch1_50"])
    build_selection_manifest(
        reaction_ids=smoke,
        campaign_manifest_path=CAMPAIGN,
        output_path=CAMPAIGN.parent / "selection_smoke5.json",
        purpose="smoke_not_counted",
        note=(
            "1+4 engineering smoke episodes (plan §4 S1): verify the repaired "
            "chain end to end; never counted toward the RL2000 main-episode "
            "denominator and never used for BC/Q fitting"
        ),
    )
    print(f"smoke5 selection frozen: {len(smoke)} cases")
    dev50 = list(cohorts["rl_dev_50_fixed_checkpoint_subset"])
    print(
        f"main fit50/dev50 selections are frozen by the dev/fit cohort files; "
        f"dev50 head: {dev50[:2]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
