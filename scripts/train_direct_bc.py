#!/usr/bin/env python3
"""Train the direct BC policy (WP3) on prepared direct transitions.

Accepts either a written dataset directory (``transitions/`` from
``prepare_direct_data.py``) or rebuilds transitions in memory from the
per-reaction prep checkpoints. The device chain is the WP-INFRA1 entry:
``cuda:0`` by default, loud failure when CUDA is missing (no CPU fallback
in production profiles; ``--cpu`` is an explicit diagnostic mode).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pes2ts_core.generation.planning.direct_mdp import _reward_spec_sha256  # noqa: E402
from pes2ts_core.pesn.offline.data.direct_dataset import (  # noqa: E402
    DirectTransitionDataset,
    write_direct_shards,
)
from pes2ts_core.pesn.offline.data.direct_preprocess import DirectPrepResult  # noqa: E402
from pes2ts_core.pesn.offline.training.direct_bc import (  # noqa: E402
    DirectBCSettings,
    DirectBCTrainer,
)


def results_from_checkpoints(prep_dir: Path, selection: str) -> list[DirectPrepResult]:
    """Rebuild prep results (transitions included) from checkpoint JSONs."""
    from pes2ts_core.pesn.contracts.direct import DirectTransition
    results = []
    for path in sorted(prep_dir.glob("*.json")):
        doc = json.loads(path.read_text())
        transitions = tuple(DirectTransition.from_payload(t)
                            for t in doc.get("transitions", []))
        results.append(DirectPrepResult(
            reaction_id=doc["reaction_id"], group_id=doc["group_id"],
            subset=selection, status=doc["status"], reason=doc["reason"],
            transitions=transitions, task_graph=doc.get("task_graph", {}),
            atomic_numbers=(None if doc.get("atomic_numbers") is None
                            else np.asarray(doc["atomic_numbers"], dtype=int)),
            atom_map_ids=tuple(doc.get("atom_map_ids", ())),
            diagnostics=doc.get("diagnostics", {}),
        ))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "config/pesn/direct_bc_v1.yaml")
    parser.add_argument("--dataset-dir", type=Path, default=None,
                        help="transitions/ directory with dataset_manifest.json")
    parser.add_argument("--selection", default="direct_fit50")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cpu", action="store_true",
                        help="explicit diagnostic CPU mode")
    parser.add_argument("--light", action="store_true")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    output_root = ROOT / config["paths"]["output_root"]

    if args.dataset_dir is not None:
        dataset_dir = args.dataset_dir
    else:
        prep_dir = output_root / "audit" / "prep" / args.selection
        results = results_from_checkpoints(prep_dir, args.selection)
        usable = [r for r in results if r.transitions]
        if not usable:
            print(f"[bc] no usable transitions under {prep_dir}", file=sys.stderr)
            return 1
        dataset_dir = output_root / "transitions"
        manifest = write_direct_shards(
            usable, dataset_dir, reward_spec_sha256=_reward_spec_sha256(
                __import__("pes2ts_core.pesn.contracts.direct", fromlist=["x"]).DEFAULT_REWARD_SPEC))
        print(f"[bc] rebuilt dataset: {manifest['n_reactions']} reactions, "
              f"{manifest['n_transitions']} transitions", flush=True)

    train = DirectTransitionDataset(dataset_dir, allowed_subsets=[args.selection])
    print(f"[bc] train reactions: {len(train.reaction_ids)}", flush=True)
    if not train.reaction_ids:
        print("[bc] empty dataset for selection", file=sys.stderr)
        return 1

    settings_kwargs = dict(config.get("train", {}))
    if args.epochs is not None:
        settings_kwargs["max_epochs"] = args.epochs
    if args.seed is not None:
        settings_kwargs["seed"] = args.seed
    if args.light:
        settings_kwargs["light"] = True
    if args.cpu:
        settings_kwargs.update(device="cpu", require_cuda=False)
    settings = DirectBCSettings(**settings_kwargs)
    run_name = args.run_name or f"bc_{args.selection}_seed{settings.seed}"
    run_dir = output_root / "bc" / run_name

    trainer = DirectBCTrainer(
        settings=settings, train_dataset=train, monitor_dataset=None,
        run_dir=run_dir, config_sha256=str(args.config),
        dataset_manifest_sha256=str(dataset_dir / "dataset_manifest.json"))
    print(f"[bc] device={settings.device} params="
          f"{trainer.policy.n_parameters}", flush=True)
    result = trainer.train()
    trainer.write_run_record()
    history = result["history"]
    print(f"[bc] done: epochs={len(history)} "
          f"final_train_group_nll={history[-1]['train_group_nll']:.4f} "
          f"params={result['model_summary']['n_parameters']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
