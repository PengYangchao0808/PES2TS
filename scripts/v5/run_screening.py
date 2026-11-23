"""V5.6 screening entry point.

Examples (run from the repository root inside WSL pes2ts-gpu env)::

    python scripts/v5/run_screening.py --config config/v5/smoke.yaml
    python scripts/v5/run_screening.py --config config/v5/screening.yaml \
        --arch M01 --arch M04 --max-epochs 1 --max-train-batches 3
    python scripts/v5/run_screening.py --config config/v5/screening.yaml \
        --generate-synthetic-only

The script is a thin wrapper around ``pes2ts_v5.training.sweep.run_sweep``;
the same entry is available as ``python -m pes2ts_v5.training.sweep`` when the
repository root is on ``PYTHONPATH``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pes2ts_v5.data.trajectories import build_synthetic_dataset  # noqa: E402
from pes2ts_v5.training.sweep import build_dataset, load_config, run_sweep  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arch", action="append", default=None,
                        help="restrict to architecture(s), e.g. --arch M01")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--generate-synthetic-only", action="store_true",
                        help="only build the synthetic fixture and exit")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.generate_synthetic_only:
        ds = build_dataset(cfg)
        print(json.dumps({
            "dataset_version": ds.dataset_version,
            "dataset_hash": ds.dataset_hash,
            "counts": getattr(ds, "counts", {}),
        }, ensure_ascii=False, indent=2))
        return 0
    if not cfg.get("archs") and args.generate_synthetic_only:
        build_synthetic_dataset(ROOT / "data/interim/v5_synthetic")
    summary = run_sweep(
        args.config, archs=args.arch, limit=args.limit, out_dir=args.out_dir,
        device=args.device, max_epochs=args.max_epochs,
        max_train_batches=args.max_train_batches)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
