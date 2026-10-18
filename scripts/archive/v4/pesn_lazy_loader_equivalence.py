#!/usr/bin/env python
"""Numerical equivalence probe: legacy eager vs lazy teacher training.

Trains ``max_epochs`` (default 2) on an exported teacher dataset twice —
once via the legacy ``load_teacher_reactions`` full-materialization path and
once via ``LazyTeacherProvider`` — then compares best dev MAE, per-epoch dev
MAE history, and a canonical sha256 over the final model weights. Exit code
0 means bit-identical weights and MAE; 1 means a mismatch (reported, never
forced). Peak RSS (``resource.getrusage``) is printed per run; probe the
lazy path alone with ``--path lazy`` under ``/usr/bin/time -v`` for a clean
process-level max RSS figure.

Seeding note: ``Trainer.__init__`` calls ``torch.manual_seed`` *after*
``build_teacher_model`` (pre-existing behavior, left unchanged), so model
weight init draws from the process-global torch generator. This script seeds
torch explicitly before constructing each Trainer so both paths initialize
from the same state and the comparison isolates the loader path only.

Determinism note: under default multi-threaded torch CPU settings the
training step is not bit-reproducible even legacy-vs-legacy (control runs on
this machine diverge after epoch 0); the loader path itself is bit-identical
at the data level (same reaction ids, frame indices, labels, pairs) and the
verdict below is therefore produced under single-thread deterministic torch
settings (default: ``--torch-threads 1 --deterministic``). Pass
``--torch-threads 0 --no-deterministic`` to probe the default backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def weights_sha256(model) -> str:
    """Canonical sha256 over every parameter tensor of ``model.state_dicts()``."""
    digest = hashlib.sha256()

    def feed(prefix: str, value) -> None:
        if isinstance(value, dict):
            for key in sorted(value):
                feed(f"{prefix}/{key}", value[key])
            return
        if hasattr(value, "detach"):
            array = value.detach().cpu().contiguous().numpy()
        else:
            array = np.ascontiguousarray(value)
        digest.update(prefix.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())

    feed("model", model.state_dicts())
    return digest.hexdigest()


def run_training(*, mode: str, dataset_dir: Path, config: dict, work_root: Path) -> dict:
    from pes2ts_core.pesn.offline.experiments.runs import (
        RunRecord,
        config_content_sha256,
        make_run_id,
    )
    from pes2ts_core.pesn.offline.splits.freeze import TRAIN_DEV, TRAIN_FIT
    from pes2ts_core.pesn.offline.training.loader import (
        LazyReactionSequence,
        LazyTeacherProvider,
        load_teacher_reactions,
    )
    from pes2ts_core.pesn.offline.training.trainer import Trainer, TrainSettings
    from pes2ts_core.utils.hashing import sha256_file
    from pes2ts_core.utils.jsonio import read_json

    settings = TrainSettings.from_config(config)
    config_sha = config_content_sha256(config)
    dataset_sha = sha256_file(dataset_dir / "dataset_manifest.json")
    if mode == "legacy":
        reactions = load_teacher_reactions(dataset_dir)
        dataset_manifest = read_json(dataset_dir / "dataset_manifest.json")
        subsets = dataset_manifest["subset_members"]
        by_subset = {
            TRAIN_FIT: set(subsets[TRAIN_FIT]),
            TRAIN_DEV: set(subsets[TRAIN_DEV]),
        }
        train_reactions = [
            r
            for r in reactions
            if r.reaction_id in by_subset[TRAIN_FIT] and r.s_mask.any()
        ]
        dev_reactions = [
            r
            for r in reactions
            if r.reaction_id in by_subset[TRAIN_DEV] and r.s_mask.any()
        ]
    else:
        provider = LazyTeacherProvider(dataset_dir)
        dataset_manifest = provider.manifest
        train_reactions = LazyReactionSequence(
            provider, provider.valid_label_reaction_ids(subset=TRAIN_FIT)
        )
        dev_reactions = LazyReactionSequence(
            provider, provider.valid_label_reaction_ids(subset=TRAIN_DEV)
        )

    runs_root = work_root / mode
    runs_root.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id(config_sha256=config_sha, seed=settings.seed, attempt=1)
    import torch

    torch.manual_seed(settings.seed)
    run_record = RunRecord.start(
        runs_root=runs_root,
        run_id=run_id,
        config=config,
        config_sha256=config_sha,
        dataset_manifest_sha256=dataset_sha,
        split_manifest_sha256=str(dataset_manifest.get("parent_split_manifest_sha256", "")),
        feature_spec_version="pesn_feature_spec_v1.0",
        label_spec_sha256=str(dataset_manifest.get("label_spec_sha256", "")),
        seed=settings.seed,
        n_parameters=0,
        backend_versions={},
    )
    trainer = Trainer(
        settings=settings,
        run_record=run_record,
        config_sha256=config_sha,
        dataset_manifest_sha256=dataset_sha,
    )
    summary = trainer.train(train_reactions, dev_reactions)
    best_path = run_record.run_dir / "checkpoint_best.pt"
    epochs = [
        {
            "epoch": e["epoch"],
            "dev_mae_s": e.get("dev_mae_s"),
            "n_samples": e.get("n_samples"),
        }
        for e in run_record.document.get("epochs", [])
    ]
    return {
        "mode": mode,
        "n_train": len(train_reactions),
        "n_dev": len(dev_reactions),
        "best_epoch": summary["best_epoch"],
        "best_dev_mae_s": summary["best_dev_mae_s"],
        "epochs": epochs,
        "weights_sha256": weights_sha256(trainer.model),
        "checkpoint_exists": best_path.is_file(),
        "checkpoint_file_sha256": sha256_file(best_path) if best_path.is_file() else None,
        "maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "outputs/pesn/g3_p1/teacher_dataset",
    )
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "config/pesn/g3_p1.yaml"
    )
    parser.add_argument("--path", choices=("legacy", "lazy", "both"), default="both")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="torch.set_num_threads value (0 = leave default)",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="torch.use_deterministic_algorithms(True) for bit-stable CPU training",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Run-record root (default: fresh temp dir)",
    )
    args = parser.parse_args()

    import torch

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)

    from pes2ts_core.pesn.offline.cli import load_pesn_config

    config = load_pesn_config(args.config)
    train_cfg = dict(config.get("train") or {})
    train_cfg["max_epochs"] = args.epochs
    train_cfg["patience"] = max(args.epochs, int(train_cfg.get("patience", 8)))
    config = {**config, "train": train_cfg}

    modes = ["legacy", "lazy"] if args.path == "both" else [args.path]
    work_root = args.workdir or Path(tempfile.mkdtemp(prefix="pesn_lazy_equiv_"))
    results = {}
    for mode in modes:
        results[mode] = run_training(
            mode=mode, dataset_dir=args.dataset, config=config, work_root=work_root
        )
        print(json.dumps(results[mode], sort_keys=True))

    if args.path != "both":
        return 0
    legacy, lazy = results["legacy"], results["lazy"]
    weights_match = legacy["weights_sha256"] == lazy["weights_sha256"]
    mae_match = legacy["best_dev_mae_s"] == lazy["best_dev_mae_s"]
    epochs_match = legacy["epochs"] == lazy["epochs"]
    verdict = {
        "weights_match": weights_match,
        "best_dev_mae_match": mae_match,
        "epoch_history_match": epochs_match,
        "overall_match": weights_match and mae_match and epochs_match,
    }
    print(json.dumps(verdict, sort_keys=True))
    return 0 if verdict["overall_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
