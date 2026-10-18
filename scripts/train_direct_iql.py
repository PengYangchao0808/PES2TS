#!/usr/bin/env python3
"""Train the direct IQL policy/value (WP4) on prepared direct transitions.

Same dataset handling as ``train_direct_bc.py`` (dataset dir or rebuild from
prep checkpoints). The ``--updates`` override exists for characterization
runs; the formal recipe uses the config value (22,000). Calibration metrics
(Q/V finiteness, MC-return vs V, AWR ESS) are written to the run record.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pes2ts_core.pesn.offline.data.direct_dataset import DirectTransitionDataset  # noqa: E402
from pes2ts_core.pesn.offline.training.direct_iql import (  # noqa: E402
    DirectIQLSettings,
    DirectIQLTrainer,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "config/pesn/direct_iql_v1.yaml")
    parser.add_argument("--dataset-dir", type=Path,
                        default=ROOT / "outputs/pesn/direct_egnn_v1_20261015/transitions")
    parser.add_argument("--selection", default="direct_fit50")
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--light", action="store_true")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    output_root = ROOT / config["paths"]["output_root"]
    dataset = DirectTransitionDataset(args.dataset_dir,
                                      allowed_subsets=[args.selection])
    if not dataset.reaction_ids:
        print("[iql] empty dataset", file=sys.stderr)
        return 1
    print(f"[iql] reactions: {len(dataset.reaction_ids)}", flush=True)

    kwargs = dict(config.get("train", {}))
    if args.updates is not None:
        kwargs["total_updates"] = args.updates
    if args.seed is not None:
        kwargs["seed"] = args.seed
    if args.light:
        kwargs["light"] = True
    if args.cpu:
        kwargs.update(device="cpu", require_cuda=False)
    settings = DirectIQLSettings(**kwargs)
    run_name = args.run_name or f"iql_{args.selection}_seed{settings.seed}"
    run_dir = output_root / "iql" / run_name

    trainer = DirectIQLTrainer(
        settings=settings, train_dataset=dataset, run_dir=run_dir,
        config_sha256=str(args.config),
        dataset_manifest_sha256=str(args.dataset_dir / "dataset_manifest.json"))
    print(f"[iql] device={settings.device} updates={settings.total_updates} "
          f"batch={settings.batch_transitions}", flush=True)
    result = trainer.train()
    trainer.write_run_record()
    history = result["history"]
    last = history[-1] if history else {}
    print(f"[iql] done: updates={settings.total_updates} "
          f"loss_v={last.get('loss_v')} loss_q={last.get('loss_q')} "
          f"loss_pi={last.get('loss_pi')} ess={last.get('awr_ess')}", flush=True)
    calib = trainer.calibration_report()
    print(f"[iql] calibration: {json.dumps(calib)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
