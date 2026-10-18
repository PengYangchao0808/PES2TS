#!/usr/bin/env python3
"""Freeze the direct split and prepare direct expert transitions (WP1).

Reads the audited candidate fixture, freezes the direct split (group/family
closure, isolation ledger), then prepares expert transitions for a selection
prefix through the audited truth bridge and ACP GFN2-xTB E/g. Every reaction
is checkpointed as JSON under ``audit/prep/`` so the run is resumable; a
completed reaction is never recomputed and a failed one stays in the batch
denominator (plan §2.3).

No truth geometry is read outside the bridge; no QC runs outside ACP.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pes2ts_core.generation.planning.direct_mdp import _reward_spec_sha256  # noqa: E402
from pes2ts_core.integration.acp.direct_observation import DirectGradientObservation  # noqa: E402
from pes2ts_core.pesn.contracts.direct import DEFAULT_REWARD_SPEC  # noqa: E402
from pes2ts_core.pesn.offline.data.direct_dataset import write_direct_shards  # noqa: E402
from pes2ts_core.pesn.offline.data.direct_preprocess import (  # noqa: E402
    DirectPrepResult,
    prepare_direct_reaction,
)
from pes2ts_core.pesn.offline.splits.direct_freeze import (  # noqa: E402
    freeze_direct_split,
    load_selection,
)


def observation_factory(receipt_root: Path, acp: dict):
    def factory(atomic_numbers: np.ndarray, reaction_id: str):
        return DirectGradientObservation(
            atomic_numbers=atomic_numbers,
            receipt_folder=receipt_root / reaction_id,
            trial_prefix=f"direct-{reaction_id}",
            charge=0, multiplicity=1,
            method=str(acp.get("method", "GFN2-xTB")),
            timeout_seconds=float(acp.get("timeout_seconds", 120.0)),
            nproc=int(acp.get("nproc", 2)),
            acp_wiring={"root": acp.get("root"), "python": acp.get("python"),
                        "config_path": acp.get("config_path")},
        )
    return factory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "config/pesn/direct_data_v1.yaml")
    parser.add_argument("--selection", default=None,
                        help="selection name (default from config)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard", default=None,
                        help="i/n to process the i-th of n interleaved slices")
    parser.add_argument("--action-max", type=float, default=None,
                        help="override mdp action_max via CLI (e.g. 0.2)")
    parser.add_argument("--variant", default=None,
                        help="variant tag for parallel calibration runs")
    parser.add_argument("--refresh", action="store_true",
                        help="re-run even when a checkpoint exists (identical "
                             "geometry replays hit the QC cache; no new QC)")
    parser.add_argument("--only-existing", action="store_true",
                        help="with --refresh: replay only reactions that already "
                             "have a checkpoint (safe partial refresh)")
    parser.add_argument("--freeze-split", action="store_true")
    parser.add_argument("--stage-only", action="store_true",
                        help="freeze split and exit (no truth/QC access)")
    args = parser.parse_args()

    from pes2ts_core.config_loader import load_config_mapping

    config = load_config_mapping(args.config)
    paths = config["paths"]
    interim = ROOT / paths["interim"]
    manifests = ROOT / paths["manifests"]
    split_dir = ROOT / paths["split_dir"]
    output_root = ROOT / paths["output_root"]
    selection = args.selection or config["selection"]

    if args.freeze_split or not (split_dir / "pesn_irc_split.json").is_file():
        cohorts = ROOT / config.get(
            "cohorts_dir", "outputs/pesn/scale20k_rl2000_v1_20261008/cohorts")
        written = freeze_direct_split(
            interim, manifests, ROOT / paths["audit_fixture"], split_dir,
            cohorts_dir=cohorts, seed=int(config.get("seed", 42)))
        print(f"[split] frozen -> {split_dir} ({len(written)} files)", flush=True)
    if args.stage_only:
        return 0

    reaction_ids = load_selection(split_dir, selection)
    group_of = {rid: rid for rid in reaction_ids}
    selection_doc = json.loads((split_dir / f"selection_{selection}.json").read_text())
    group_of = dict(zip(selection_doc["reaction_ids"], selection_doc["group_ids"]))
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        reaction_ids = [rid for k, rid in enumerate(reaction_ids) if k % n == i]
    if args.only_existing:
        reaction_ids = [rid for rid in reaction_ids
                        if (prep_dir_peek := output_root / "audit" / "prep"
                            / (selection if (args.variant or "base") == "base"
                               else f"{selection}__{args.variant or 'base'}")
                            ).joinpath(f"{rid}.json").is_file()]
    if args.limit:
        reaction_ids = reaction_ids[:args.limit]
    print(f"[prep] {selection}: {len(reaction_ids)} reactions in this slice", flush=True)

    variant = args.variant or "base"
    prep_dir = output_root / "audit" / "prep" / (
        selection if variant == "base" else f"{selection}__{variant}")
    prep_dir.mkdir(parents=True, exist_ok=True)
    (output_root / "transitions").mkdir(parents=True, exist_ok=True)
    receipt_root = ROOT / paths["observations_dir"]
    if variant != "base":
        receipt_root = receipt_root / variant
    factory = observation_factory(receipt_root, config["acp"])
    spec = DEFAULT_REWARD_SPEC
    mdp = config.get("mdp", {})
    action_max = float(args.action_max if args.action_max is not None
                       else mdp.get("action_max_angstrom", 0.1))
    print(f"[prep] variant={variant} action_max={action_max}", flush=True)

    results: list[DirectPrepResult] = []
    for rid in reaction_ids:
        checkpoint = prep_dir / f"{rid}.json"
        if checkpoint.is_file() and not args.refresh:
            doc = json.loads(checkpoint.read_text())
            print(f"[prep] {rid}: cached ({doc['status']}/{doc['reason']})", flush=True)
            from pes2ts_core.pesn.contracts.direct import DirectTransition
            transitions = tuple(
                DirectTransition.from_payload(t) for t in doc.get("transitions", []))
            results.append(DirectPrepResult(
                reaction_id=rid, group_id=doc["group_id"], subset=selection,
                status=doc["status"], reason=doc["reason"],
                transitions=transitions,
                task_graph=doc.get("task_graph", {}),
                atomic_numbers=(np.asarray(doc["atomic_numbers"], dtype=int)
                                if doc.get("atomic_numbers") is not None else None),
                atom_map_ids=tuple(doc.get("atom_map_ids", ())),
                diagnostics=doc.get("diagnostics", {}),
            ))
            continue
        t0 = time.perf_counter()
        try:
            result = prepare_direct_reaction(
                rid,
                interim_dir=interim, manifests_dir=manifests, split_dir=split_dir,
                observation_factory=factory,
                reward_spec=spec, subset=selection,
                group_id=group_of.get(rid, rid),
                action_max=action_max,
                shard_size=int(config.get("authority_shard_size", 1000)),
            )
        except Exception as exc:  # honest failure ledger, never silent drop
            result = DirectPrepResult(
                reaction_id=rid, group_id=group_of.get(rid, rid), subset=selection,
                status="excluded", reason=f"{type(exc).__name__}: {exc}",
            )
        elapsed = time.perf_counter() - t0
        diag = dict(result.diagnostics)
        diag["wall_seconds_total"] = round(elapsed, 3)
        payload = {
            "reaction_id": rid, "group_id": result.group_id, "status": result.status,
            "reason": result.reason, "diagnostics": diag,
            "task_graph": dict(result.task_graph),
            "atomic_numbers": (None if result.atomic_numbers is None
                               else result.atomic_numbers.astype(int).tolist()),
            "atom_map_ids": list(result.atom_map_ids),
            "n_transitions": len(result.transitions),
            "transitions": [t.to_payload() for t in result.transitions],
        }
        checkpoint.write_text(json.dumps(payload, ensure_ascii=False))
        print(f"[prep] {rid}: {result.status}/{result.reason} "
              f"nodes={int(diag.get('n_nodes', 0))} qc={int(diag.get('qc_calls', 0))} "
              f"({elapsed:.1f}s)", flush=True)
        results.append(result)

    if args.shard is None:
        transitions_dir = output_root / "transitions" if variant == "base" \
            else output_root / "transitions" / variant
        manifest = write_direct_shards(
            results, transitions_dir,
            reward_spec_sha256=_reward_spec_sha256(spec))
        (transitions_dir / "dataset_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2))
        n_ok = sum(1 for r in results if r.status == "completed")
        print(f"[done] reactions={len(results)} goal_completed={n_ok} "
              f"excluded={manifest['n_excluded']} "
              f"transitions={manifest['n_transitions']}", flush=True)
    else:
        print(f"[done] slice {args.shard}: {len(results)} reactions checkpointed "
              f"(shard manifest written only by slice 0/n without --shard)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
