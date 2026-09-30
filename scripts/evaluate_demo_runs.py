"""Evaluate a path-referenced Demo run bundle and write a metrics summary."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pes2ts_core.contracts import ContractError
from pes2ts_core.demo_metrics import evaluate_demo_runs, load_demo_run_bundle
from pes2ts_core.utils.jsonio import write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="pes2ts_demo_run_bundle_v1 JSON file")
    parser.add_argument("--output", type=Path, required=True, help="metrics JSON output path")
    parser.add_argument("--ranking-rule", default="highest_scan_energy",
                        choices=("highest_scan_energy", "internal_scan_peak"))
    parser.add_argument("--evaluation-split", default="valid", choices=("train", "valid", "test"))
    args = parser.parse_args()
    try:
        runs, labels, cohort_cases = load_demo_run_bundle(args.bundle)
        report = evaluate_demo_runs(runs=runs, cohort_cases=cohort_cases, ranking_labels=labels,
                                    ranking_rule=args.ranking_rule,
                                    evaluation_split=args.evaluation_split)
        write_json(args.output, report)
    except (ContractError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Demo metrics written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
