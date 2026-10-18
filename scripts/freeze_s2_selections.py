"""Freeze the S2 main/dev campaign inputs from the PREPARED plans only.

Builds one campaign manifest over every prepared origin in the main origin
tree (fit50 + dev50 + replacements), plus two hash-bound selections:

  - selection_fit50 : prepared members of the frozen rl_fit_batch1_50 plus
    any ranked replacements recorded by reconcile_origin_gaps.py
  - selection_dev50 : the prepared dev-checkpoint-50 equivalent

Origin-preparation failures stay OUT of the selections and are reported as
funnel gaps with the frozen-cohort ledger; they are never silently padded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pes2ts_core.generation.control_round import build_selection_manifest  # noqa: E402

CAMPAIGN_ROOT = (
    REPO_ROOT / "outputs/pesn/scale20k_rl2000_v1_20261008/rl_0050/origin_prep_main"
)
COHORTS = REPO_ROOT / "outputs/pesn/scale20k_rl2000_v1_20261008/cohorts/rl_cohorts.json"
RECONCILED = (
    REPO_ROOT
    / "outputs/pesn/scale20k_rl2000_v1_20261008/cohorts/origin_gap_replacements.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/g2t_scale.yaml")
    args = parser.parse_args()

    cohorts = json.loads(COHORTS.read_text(encoding="utf-8"))
    fit50_frozen = list(cohorts["rl_fit_batch1_50"])
    dev50_frozen = list(cohorts["rl_dev_50_fixed_checkpoint_subset"])
    basis = "frozen"

    if RECONCILED.is_file():
        reconciled = json.loads(RECONCILED.read_text(encoding="utf-8"))
        fit50_frozen = list(reconciled["effective_fit50"])
        dev50_frozen = list(reconciled["effective_dev50"])
        basis = "reconciled_with_replacements"

    prepared = {p.parent.name for p in CAMPAIGN_ROOT.glob("*/plan.json")}
    fit_ready = [rid for rid in fit50_frozen if rid in prepared]
    dev_ready = [rid for rid in dev50_frozen if rid in prepared]
    fit_gap = [rid for rid in fit50_frozen if rid not in prepared]
    dev_gap = [rid for rid in dev50_frozen if rid not in prepared]

    import subprocess

    manifest_path = CAMPAIGN_ROOT / "campaign_manifest.json"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "bin/pes2ts"),
            "g2t-manifest",
            "--snapshots",
            str(CAMPAIGN_ROOT / "snapshots"),
            "--output",
            str(manifest_path),
            "--config",
            args.config,
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )

    if fit_ready:
        build_selection_manifest(
            reaction_ids=fit_ready,
            campaign_manifest_path=manifest_path,
            output_path=CAMPAIGN_ROOT / "selection_fit50.json",
            purpose="train_collect",
            note=(
                "S2 first-batch RL main collection: prepared subset of the "
                "frozen rl_fit_batch1_50 (origin-prep failures reported as "
                "gaps, replaced only per the frozen-cohort ledger rules)"
            ),
        )
    if dev_ready:
        build_selection_manifest(
            reaction_ids=dev_ready,
            campaign_manifest_path=manifest_path,
            output_path=CAMPAIGN_ROOT / "selection_dev50.json",
            purpose="dev50_checkpoint",
            note=(
                "S2 dev checkpoint-50 execution list; frozen rule set, never "
                "a training permission"
            ),
        )
    print(
        json.dumps(
            {
                "fit_ready": len(fit_ready),
                "fit_gap": fit_gap,
                "dev_ready": len(dev_ready),
                "dev_gap": dev_gap,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
