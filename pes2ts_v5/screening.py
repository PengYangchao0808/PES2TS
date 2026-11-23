"""V5.6 screening CLI — thin alias of ``python -m pes2ts_v5.runner sweep``.

Compatibility entry point for the moved ``scripts/v5/run_screening.py``.  It
contains **no training loop of its own**: dataset building, model assembly and
training all happen in the single V5.6 engine (``pes2ts_v5.runner`` /
``pes2ts_v5.training``).  Prefer the runner directly::

    python -m pes2ts_v5.runner sweep --config config/experiments/screening.yaml \
        --arch M01 --arch M04 --max-epochs 1

This alias keeps the historical flag spellings (``--generate-synthetic-only``,
``--limit``) working and defaults to all 14 architectures when ``--arch`` is not
given::

    python -m pes2ts_v5.screening --config config/experiments/screening.yaml
    python -m pes2ts_v5.screening --config config/experiments/smoke.yaml \
        --arch M01 --arch M04 --max-epochs 1 --max-train-batches 3
    python -m pes2ts_v5.screening --config config/experiments/smoke.yaml \
        --generate-synthetic-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repository root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pes2ts_v5.configs.schema import ConfigError, load_experiment_config  # noqa: E402
from pes2ts_v5.models.registry import list_cores  # noqa: E402
from pes2ts_v5.runner import resolve_dataset, run_sweep  # noqa: E402
from pes2ts_v5.training._util import json_safe  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="experiment YAML under config/experiments/")
    parser.add_argument("--arch", action="append", default=None,
                        help="restrict to architecture(s), e.g. --arch M01 "
                             "(ids or algorithm aliases such as gru)")
    parser.add_argument("--limit", type=int, default=None,
                        help="train only the first N architectures")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--generate-synthetic-only", action="store_true",
                        help="only build/load the dataset fixture and exit")
    args = parser.parse_args(argv)

    try:
        cfg = load_experiment_config(args.config)
        if args.generate_synthetic_only:
            dataset, data_info = resolve_dataset(cfg)
            payload = {
                "status": "ok",
                "mode": "generate-synthetic-only",
                "data": {
                    **data_info,
                    "dataset_version": dataset.dataset_version,
                    "dataset_hash": dataset.dataset_hash,
                    "counts": getattr(dataset, "counts", {}),
                },
            }
            print(json.dumps(json_safe(payload), ensure_ascii=False))
            return 0
        archs = args.arch or list_cores()
        if args.limit is not None:
            archs = archs[: max(int(args.limit), 0)]
        summary = run_sweep(
            cfg, archs=archs, out_dir=args.out_dir, device=args.device,
            max_epochs=args.max_epochs,
            max_train_batches=args.max_train_batches)
    except (ConfigError, RuntimeError, ValueError, NotImplementedError) as exc:
        print(json.dumps(json_safe({
            "status": "error",
            "mode": "sweep",
            "error_type": type(exc).__name__,
            "code": getattr(exc, "code", None),
            "error": str(exc)[:2000],
        }), ensure_ascii=False))
        return 1
    print(json.dumps(json_safe(summary), ensure_ascii=False))
    return 0 if summary.get("n_failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
