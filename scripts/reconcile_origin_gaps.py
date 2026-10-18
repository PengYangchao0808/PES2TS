"""Reconcile origin-preparation gaps with ranked replacements (plan §3.2).

Origin-preparation failures are pre-compute eligibility gaps: those groups
never reach a main episode, so the plan sanctions replacing them from
STRICTLY LATER ranks of the frozen ranking (never re-sorting, never
silently padding).  This script:

  1. re-derives the exact cohort-freeze ranking pool from the frozen
     artifact files (qualified fit pool minus final rl_dev/sealed/demo24),
  2. finds frozen fit50/dev50 members without a prepared plan,
  3. walks the ranking past the last prescreen-rejected rank, keeping only
     prescreen-READY candidates not already used or rejected,
  4. writes the effective fit50/dev50 lists (prepared members + ranked
     replacements) plus a full typed ledger to
     cohorts/origin_gap_replacements.json.

The replacement ids still need origin preparation afterwards (the printed
id file feeds prepare_rl_origins.py directly).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from freeze_scale_cohorts_s0 import (  # noqa: E402
    PRESCREEN_CACHE,
    SEED,
    _prescreen_status,
    _rank_order,
)

CAMPAIGN_ROOT = REPO_ROOT / "outputs/pesn/scale20k_rl2000_v1_20261008"
SPLIT = REPO_ROOT / "outputs/pesn/g3_p1/splits/pesn_irc_split.json"


def _ranking_pool(
    rl_dev_200: set[str], sealed_iid: set[str], demo24: set[str]
) -> list[str]:
    """Reproduce freeze_scale_cohorts_s0's rl ranking pool exactly."""
    from pes2ts_core.pesn.offline.data.inventory import load_source_inventory

    inventory = load_source_inventory(REPO_ROOT / "data/interim")
    qualified_all = set(inventory.qualified_train_reactions())
    split_doc = json.loads(SPLIT.read_text(encoding="utf-8"))
    train_fit = set(split_doc["subsets"]["train_fit"])
    qualified_fit = qualified_all & train_fit
    return sorted(qualified_fit - rl_dev_200 - sealed_iid - demo24)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--origin-root", type=Path,
        default=CAMPAIGN_ROOT / "rl_0050/origin_prep_main",
    )
    parser.add_argument(
        "--cohort-file", type=Path,
        default=CAMPAIGN_ROOT / "cohorts/rl_cohorts.json",
    )
    args = parser.parse_args()

    cohorts = json.loads(args.cohort_file.read_text(encoding="utf-8"))
    sealed = json.loads(
        (args.cohort_file.parent / "sealed_cohorts.json").read_text(encoding="utf-8")
    )
    exclusions = json.loads(
        (args.cohort_file.parent / "cross_task_exclusions.json").read_text(
            encoding="utf-8"
        )
    )
    fit_frozen = list(cohorts["rl_fit_batch1_50"])
    dev_frozen = list(cohorts["rl_dev_50_fixed_checkpoint_subset"])
    fit_ledger = cohorts.get("rl_fit_prescreen", {}).get("rejection_ledger", [])

    prepared = {p.parent.name for p in args.origin_root.glob("*/plan.json")}
    fit_gaps = [rid for rid in fit_frozen if rid not in prepared]
    dev_gaps = [rid for rid in dev_frozen if rid not in prepared]

    cache: dict = {}
    if PRESCREEN_CACHE.is_file():
        cache = json.loads(PRESCREEN_CACHE.read_text(encoding="utf-8"))

    ranking_pool = _ranking_pool(
        rl_dev_200=set(cohorts["rl_dev_200"]),
        sealed_iid=set(sealed["sealed_iid_100"]),
        demo24=set(exclusions["demo24_dev_exposure"]),
    )
    pool_set = set(ranking_pool)
    frozen_members = set(fit_frozen) | set(cohorts.get("rl_smoke_5_not_counted", []))
    missing_from_pool = sorted(frozen_members - pool_set)
    if missing_from_pool:
        raise SystemExit(
            f"ranking pool drifted: {len(missing_from_pool)} frozen members "
            f"absent ({missing_from_pool[:3]}...); refusing to reconcile"
        )

    ranked = _rank_order(ranking_pool, SEED)
    prior_output = args.cohort_file.parent / "origin_gap_replacements.json"
    prior_fit_ids: list[str] = []
    prior_dev_ids: list[str] = []
    if prior_output.is_file():
        prior = json.loads(prior_output.read_text(encoding="utf-8"))
        for section, sink in (("fit", prior_fit_ids), ("dev", prior_dev_ids)):
            for row in prior.get("replacements", {}).get(section, []):
                sink.append(row["reaction_id"])
    used = (
        set(fit_frozen)
        | set(dev_frozen)
        | set(cohorts.get("rl_dev_200", []))
        | set(cohorts.get("rl_smoke_5_not_counted", []))
        | {row["reaction_id"] for row in fit_ledger}
        | set(prior_fit_ids)
        | set(prior_dev_ids)
    )
    last_used_rank = max((row["rank"] for row in fit_ledger), default=-1)
    if prior_output.is_file():
        prior_ranks = [
            row["rank"]
            for section in ("fit", "dev")
            for row in prior.get("replacements", {}).get(section, [])
        ]
        last_used_rank = max([last_used_rank, *prior_ranks], default=-1)

    prior_fit_success = [rid for rid in prior_fit_ids if rid in prepared]
    prior_dev_success = [rid for rid in prior_dev_ids if rid in prepared]
    fit_needed = max(0, len(fit_gaps) - len(prior_fit_success))
    dev_needed = max(0, len(dev_gaps) - len(prior_dev_success))
    n_needed = fit_needed + dev_needed

    replacements: list[dict] = []
    scan_ledger: list[dict] = []
    for rank, reaction_id in enumerate(ranked):
        if len(replacements) >= n_needed:
            break
        if rank <= last_used_rank or reaction_id in used:
            continue
        status = _prescreen_status(reaction_id, cache)
        if status == "ready":
            replacements.append({"rank": rank, "reaction_id": reaction_id})
        else:
            scan_ledger.append(
                {"rank": rank, "reaction_id": reaction_id, "status": status}
            )

    fit_replacements = replacements[:fit_needed]
    dev_replacements = replacements[fit_needed : fit_needed + dev_needed]
    effective_fit = (
        [rid for rid in fit_frozen if rid in prepared]
        + prior_fit_success
        + [row["reaction_id"] for row in fit_replacements]
    )
    effective_dev = (
        [rid for rid in dev_frozen if rid in prepared]
        + prior_dev_success
        + [row["reaction_id"] for row in dev_replacements]
    )
    shortfall = max(0, n_needed - len(replacements))

    output = {
        "schema_version": "pes2ts_scale_origin_gap_reconciliation_v1",
        "ranking_pool_sha256": hashlib.sha256(
            "\n".join(ranking_pool).encode("utf-8")
        ).hexdigest(),
        "fit_failures": [
            {"reaction_id": rid, "code": "ORIGIN_PREP_FAILED"} for rid in fit_gaps
        ],
        "dev_failures": [
            {"reaction_id": rid, "code": "ORIGIN_PREP_FAILED"} for rid in dev_gaps
        ],
        "replacements": {
            "fit": fit_replacements,
            "dev": dev_replacements,
            "scan_rejections": scan_ledger,
            "pool_exhausted_shortfall": shortfall,
        },
        "effective_fit50": sorted(effective_fit),
        "effective_dev50": sorted(effective_dev),
        "note": (
            "Failures are pre-compute eligibility gaps (no main episode was "
            "attempted); replacements come from strictly later ranks of the "
            "frozen ordering; the original selection denominator stays on "
            "record in rl_cohorts.json and is never rewritten"
        ),
    }
    target = args.cohort_file.parent / "origin_gap_replacements.json"
    target.write_text(
        json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    replacement_ids = [row["reaction_id"] for row in replacements]
    (args.cohort_file.parent / ".replacement_ids.json").write_text(
        json.dumps(replacement_ids), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "fit_gaps": len(fit_gaps),
                "dev_gaps": len(dev_gaps),
                "replacements_found": len(replacements),
                "shortfall": shortfall,
                "effective_fit50": len(effective_fit),
                "effective_dev50": len(effective_dev),
            },
            indent=2,
        )
    )
    print(f"replacement ids: {args.cohort_file.parent / '.replacement_ids.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
