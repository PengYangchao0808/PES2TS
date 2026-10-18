"""S0 cohort freezing for the scale20k_rl2000 campaign (plan §3).

Freezes the global group-level usage lists BEFORE any new teacher truth read
or real collection: nested IRC fit cohorts (2k verified -> 5k), fixed IRC
dev 250 re-verification, RL dev 200 (fixed 50 checkpoint subset), RL fit
first batch 50, sealed IID 100 + family 100, and the cross-task exclusion
ledger (Demo24, sealed, RL dev).  Selection is unbiased seed-hash ranking
(deterministic_selection); no reference-TS information is consulted.  Every
cohort file records pool digests so later stages can prove the pools did not
change.

Read-only over data/interim + existing manifests; writes only under the
campaign cohorts/ root.  No truth access, no QC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import hashlib  # noqa: E402

from pes2ts_core.pesn.offline.data.dataset import deterministic_selection  # noqa: E402
from pes2ts_core.pesn.offline.data.inventory import (  # noqa: E402
    RESOLVED_P1_STATUSES,
    SUPPORTED_LAYOUTS,
    SUPPORTED_ORIENTATIONS,
    load_source_inventory,
)

SEED = 42

PRESCREEN_CACHE = REPO_ROOT / (
    "outputs/pesn/scale20k_rl2000_v1_20261008/cohorts/.prescreen_cache.json"
)


def _rank_order(reaction_ids, seed: int) -> list[str]:
    def rank_key(rid: str) -> tuple[int, str]:
        digest = hashlib.sha256(f"{seed}:{rid}".encode("utf-8")).hexdigest()
        return (int(digest[:16], 16), rid)

    return sorted(reaction_ids, key=rank_key)


def _prescreen_status(reaction_id: str, cache: dict) -> str:
    if reaction_id in cache:
        return str(cache[reaction_id])
    from prescreen_origin_readiness import prescreen

    import pyarrow.parquet as pq

    if not hasattr(_prescreen_status, "_smiles"):
        _prescreen_status._smiles = {  # type: ignore[attr-defined]
            str(row["reaction_id"]): str(row["reaction_smiles"])
            for row in pq.read_table(
                REPO_ROOT / "data/interim/inventory.parquet",
                columns=["reaction_id", "reaction_smiles"],
            ).to_pylist()
        }
    from prepare_rl_origins import build_control_snapshot, export_path

    source = export_path(reaction_id)
    if not source.is_file():
        status = "export_missing"
    else:
        export_doc = json.loads(source.read_text(encoding="utf-8"))
        snapshot = build_control_snapshot(
            export_doc, _prescreen_status._smiles.get(reaction_id, "")  # type: ignore[attr-defined]
        )
        status, _ = prescreen(reaction_id, snapshot)
    cache[reaction_id] = status
    PRESCREEN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    PRESCREEN_CACHE.write_text(
        json.dumps(cache, sort_keys=True, indent=1), encoding="utf-8"
    )
    return status


def _select_origin_ready(
    ranking_pool: list[str], *, fit_quota: int, smoke_quota: int
) -> tuple[list[str], list[str], list[dict]]:
    """Walk the frozen ranking; take the first origin-READY groups.

    Pre-screen failures are资格缺口 (plan §3.2): the ranking order is never
    re-sorted, every rejection is recorded with its rank and typed status,
    and replacements come from strictly later ranks.
    """
    cache: dict = {}
    if PRESCREEN_CACHE.is_file():
        cache = json.loads(PRESCREEN_CACHE.read_text(encoding="utf-8"))
    ranked = _rank_order(ranking_pool, SEED)
    ready: list[str] = []
    ledger: list[dict] = []
    for rank, reaction_id in enumerate(ranked):
        if len(ready) >= fit_quota + smoke_quota:
            break
        status = _prescreen_status(reaction_id, cache)
        if status == "ready":
            ready.append(reaction_id)
        else:
            ledger.append(
                {"rank": rank, "reaction_id": reaction_id, "status": status}
            )
    return (
        ready[:fit_quota],
        ready[fit_quota : fit_quota + smoke_quota],
        ledger,
    )


def _sha256_json(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


def _atom_bucket(n_atoms: int) -> str:
    if n_atoms <= 8:
        return "small"
    if n_atoms <= 16:
        return "medium"
    if n_atoms <= 28:
        return "large"
    return "xl"


def _strata_report(inventory, reaction_ids: list[str]) -> dict:
    families: dict[str, int] = {}
    buckets: dict[str, int] = {}
    layouts: dict[str, int] = {}
    for rid in reaction_ids:
        row = inventory.inventory_rows.get(rid) or {}
        summary = inventory.p1_summary.get(rid) or {}
        n_atoms = int(row.get("total_atoms_reactants") or 0)
        buckets[_atom_bucket(n_atoms)] = buckets.get(_atom_bucket(n_atoms), 0) + 1
        family = inventory.family_by_reaction.get(rid)
        if family:
            families[family] = families.get(family, 0) + 1
        layout = summary.get("layout")
        layouts[str(layout)] = layouts.get(str(layout), 0) + 1
    return {
        "n": len(reaction_ids),
        "atom_buckets": dict(sorted(buckets.items())),
        "layouts": dict(sorted(layouts.items())),
        "n_families": len(families),
        "top_families": sorted(families.items(), key=lambda kv: -kv[1])[:10],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs/pesn/scale20k_rl2000_v1_20261008/cohorts",
    )
    parser.add_argument("--interim-dir", type=Path, default=REPO_ROOT / "data/interim")
    parser.add_argument(
        "--split", type=Path,
        default=REPO_ROOT / "outputs/pesn/g3_p1/splits/pesn_irc_split.json",
    )
    parser.add_argument(
        "--teacher-manifest", type=Path,
        default=REPO_ROOT / "outputs/pesn/g3_p1/teacher_dataset/dataset_manifest.json",
    )
    parser.add_argument(
        "--demo24-manifest", type=Path,
        default=REPO_ROOT / "outputs/pesn/demo24_rl_v1_20261007/inputs/campaign_manifest.json",
    )
    args = parser.parse_args()

    inventory = load_source_inventory(args.interim_dir)
    qualified_all = set(inventory.qualified_train_reactions())

    split_doc = json.loads(args.split.read_text(encoding="utf-8"))
    train_fit = set(split_doc["subsets"]["train_fit"])
    train_dev = set(split_doc["subsets"]["train_dev"])
    family_holdout = set(split_doc["family_holdout"]["reaction_ids"])
    qualified_fit = qualified_all & train_fit

    teacher = json.loads(args.teacher_manifest.read_text(encoding="utf-8"))
    irc_2k = set(teacher["subset_members"]["train_fit"])
    irc_250 = set(teacher["subset_members"]["train_dev"])

    demo24 = {
        str(case["reaction_id"])
        for case in json.loads(args.demo24_manifest.read_text(encoding="utf-8"))["cases"]
    }

    problems: list[str] = []
    if not irc_2k <= qualified_fit:
        problems.append(
            f"irc_2k not fully qualified in train_fit: {len(irc_2k - qualified_fit)} drifted"
        )
    if not irc_250 <= (qualified_all & train_dev):
        problems.append(
            f"irc_250 not fully qualified in train_dev: "
            f"{len(irc_250 - (qualified_all & train_dev))} drifted"
        )

    family_pool = sorted(
        rid
        for rid in family_holdout
        if rid in inventory.p1_summary
        and str(inventory.p1_summary[rid].get("status")) in RESOLVED_P1_STATUSES
        and rid in inventory.scan_ready
        and str(inventory.p1_summary[rid].get("layout")) in SUPPORTED_LAYOUTS
        and str(inventory.p1_summary[rid].get("orientation")) in SUPPORTED_ORIENTATIONS
    )

    irc_used = irc_2k | irc_250
    sealed_iid = deterministic_selection(
        sorted(qualified_fit - irc_used - demo24), seed=SEED, limit=100
    )
    rl_dev_pool = sorted(qualified_fit - irc_used - demo24 - set(sealed_iid))
    rl_dev_200 = deterministic_selection(rl_dev_pool, seed=SEED, limit=200)
    rl_dev_50 = deterministic_selection(rl_dev_pool, seed=SEED, limit=50)
    if not set(rl_dev_50) <= set(rl_dev_200):
        problems.append("rl_dev_50 is not nested inside rl_dev_200")

    rl_dev_200_ready, _dev_smoke, rl_dev_prescreen_ledger = _select_origin_ready(
        rl_dev_pool, fit_quota=200, smoke_quota=0
    )
    n_dev_replaced = len(set(rl_dev_200) - set(rl_dev_200_ready))
    rl_dev_200 = rl_dev_200_ready
    rl_dev_50 = rl_dev_200[:50]

    irc_5k_pool = sorted(qualified_fit - set(rl_dev_200) - set(sealed_iid))
    irc_5k = deterministic_selection(irc_5k_pool, seed=SEED, limit=5000)
    if not irc_2k <= set(irc_5k):
        problems.append(
            f"irc_2k not nested in irc_5k: {len(irc_2k - set(irc_5k))} missing"
        )

    rl_ranking_pool = sorted(
        qualified_fit - set(rl_dev_200) - set(sealed_iid) - demo24
    )
    rl_fit_50, rl_smoke_5, rl_prescreen_ledger = _select_origin_ready(
        rl_ranking_pool, fit_quota=50, smoke_quota=5
    )
    sealed_family_100 = deterministic_selection(family_pool, seed=SEED, limit=100)

    if set(rl_dev_200) & set(irc_5k):
        problems.append("rl_dev overlaps irc_5k")
    if set(rl_fit_50) & set(rl_dev_200):
        problems.append("rl_fit overlaps rl_dev")
    if set(rl_fit_50) & demo24:
        problems.append("rl_fit overlaps demo24")
    sealed_all = set(sealed_iid) | set(sealed_family_100)
    for name, cohort in (
        ("irc_5k", irc_5k),
        ("rl_dev_200", rl_dev_200),
        ("rl_fit_50", rl_fit_50),
    ):
        if sealed_all & set(cohort):
            problems.append(f"sealed overlaps {name}")

    if problems:
        for line in problems:
            print(f"S0 COHORT PROBLEM: {line}")
        raise SystemExit(3)

    pools = {
        "qualified_all_n": len(qualified_all),
        "qualified_fit_n": len(qualified_fit),
        "train_fit_n": len(train_fit),
        "family_holdout_n": len(family_holdout),
        "family_qualified_n": len(family_pool),
        "irc_2k_n": len(irc_2k),
        "irc_250_n": len(irc_250),
        "demo24": sorted(demo24),
    }
    pool_digest = _sha256_json(
        {
            "sources": inventory.source_hashes,
            "split_sha256": hashlib.sha256(args.split.read_bytes()).hexdigest(),
            "teacher_manifest_sha256": hashlib.sha256(
                args.teacher_manifest.read_bytes()
            ).hexdigest(),
            "qualified_fit_sha256": _sha256_json(sorted(qualified_fit)),
        }
    )

    irc_cohorts = {
        "schema_version": "pes2ts_scale_cohort_irc_v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "pool_digest": pool_digest,
        "irc_fit_2000_verified": sorted(irc_2k),
        "irc_fit_5000": irc_5k,
        "irc_dev_250": sorted(irc_250),
        "pool": {
            **pools,
            "irc_5k_pool_exclusions": {
                "rl_dev_200": len(rl_dev_200),
                "sealed_iid": len(sealed_iid),
            },
            "ranking_pool_sha256": _sha256_json(irc_5k_pool),
        },
        "strata": {
            "irc_5000": _strata_report(inventory, irc_5k),
            "irc_dev_250": _strata_report(inventory, sorted(irc_250)),
        },
    }
    rl_cohorts = {
        "schema_version": "pes2ts_scale_cohort_rl_v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "pool_digest": pool_digest,
        "rl_dev_200": rl_dev_200,
        "rl_dev_prescreen": {
            "n_replaced_by_readiness": n_dev_replaced,
            "rejection_ledger": rl_dev_prescreen_ledger,
            "note": (
                "dev groups must be origin-preparable (3-way execution needs "
                "episodes); the cohort is the first 200 READY groups in the "
                "frozen ranking order, replacements from strictly later ranks"
            ),
        },
        "rl_dev_50_fixed_checkpoint_subset": rl_dev_50,
        "rl_fit_batch1_50": rl_fit_50,
        "rl_smoke_5_not_counted": rl_smoke_5,
        "rl_fit_prescreen": {
            "rejection_ledger": rl_prescreen_ledger,
            "note": (
                "fit batch 1 = first 50 origin-READY groups in the frozen "
                "ranking order; pre-screen failures are typed资格缺口 with "
                "rank and status, replaced from strictly later ranks"
            ),
        },
        "smoke_note": (
            "1+4 smoke episodes for engineering verification only; never "
            "counted toward the RL2000 main-episode denominator; groups may "
            "later contribute their own main episodes"
        ),
        "future_nested_batches_note": (
            "cumulative 50/200/500/1000/2000 stay prefix-nested by re-running "
            "deterministic_selection with larger limits over the same frozen "
            "pool (rl_dev/sealed excluded); never re-rank after seeing results"
        ),
        "strata": {
            "rl_dev_200": _strata_report(inventory, rl_dev_200),
            "rl_fit_50": _strata_report(inventory, rl_fit_50),
        },
    }
    sealed = {
        "schema_version": "pes2ts_scale_cohort_sealed_v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "pool_digest": pool_digest,
        "sealed_iid_100": sealed_iid,
        "sealed_family_100": sealed_family_100,
        "excluded_from": [
            "irc_fit_all_scales",
            "rl_fit_all_batches",
            "rl_dev",
            "registry_and_advantage_fits",
            "calibration",
        ],
        "sample_power_confirmed": False,
        "note": (
            "sealed is read exactly once at final acceptance; sample size is a "
            "budget starting point, not a power guarantee"
        ),
    }
    exclusions = {
        "schema_version": "pes2ts_scale_cross_task_exclusions_v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pool_digest": pool_digest,
        "demo24_dev_exposure": sorted(demo24),
        "demo24_status": "development regression only; never sealed evidence",
        "rl_dev_200_excluded_from": ["irc_fit", "rl_fit", "registry", "advantage_fits"],
        "sealed_all": sorted(sealed_all),
        "development_exposure": {
            "demo24_in_irc_5k": sorted(set(irc_5k) & demo24),
            "rl_fit_may_overlap_irc_fit": sorted(set(rl_fit_50) & set(irc_5k)),
        },
    }
    audit = {
        "schema_version": "pes2ts_scale_s0_audit_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "funnel": {
            "inventory_rows": len(inventory.inventory_rows),
            "p1_summary_rows": len(inventory.p1_summary),
            "qualified_train_all": len(qualified_all),
            "train_fit_split": len(train_fit),
            "qualified_fit_pool": len(qualified_fit),
            "qualified_dev_pool": len(qualified_all & train_dev),
            "family_holdout_qualified": len(family_pool),
            "selected": {
                "irc_5k": len(irc_5k),
                "irc_dev_250": len(irc_250),
                "rl_dev_200": len(rl_dev_200),
                "rl_dev_50": len(rl_dev_50),
                "rl_fit_50": len(rl_fit_50),
                "sealed_iid": len(sealed_iid),
                "sealed_family": len(sealed_family_100),
            },
        },
        "source_hashes": inventory.source_hashes,
        "split_sha256": hashlib.sha256(args.split.read_bytes()).hexdigest(),
        "teacher_manifest_sha256": hashlib.sha256(
            args.teacher_manifest.read_bytes()
        ).hexdigest(),
        "pool_digest": pool_digest,
        "checks_passed": problems == [],
    }

    _write(args.output_root / "irc_fit_cohorts.json", irc_cohorts)
    _write(args.output_root / "rl_cohorts.json", rl_cohorts)
    _write(args.output_root / "sealed_cohorts.json", sealed)
    _write(args.output_root / "cross_task_exclusions.json", exclusions)
    _write(args.output_root / "s0_audit.json", audit)
    print(json.dumps(audit["funnel"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
