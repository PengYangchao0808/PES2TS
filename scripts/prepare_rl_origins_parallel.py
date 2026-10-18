"""Parallel origin-preparation driver for the RL main/dev cohorts.

Splits a cohort list into N worker slices and launches one
``prepare_rl_origins.py`` process per slice so the ACP/ORCA origin
free-optimizations (nproc=10 each) saturate the 32-core box without
oversubscription.  Each worker writes typed rows into its own summary;
``origin_prep_summary.json`` files are merged by the caller later — this
driver only orchestrates processes and reports their exit codes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reactions", required=True, help="JSON list file")
    parser.add_argument("--key", default=None, help="dict key to extract")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--acp-config", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    payload = json.loads(Path(args.reactions).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if args.key:
            payload = payload[args.key]
        else:
            payload = next(v for v in payload.values() if isinstance(v, list))
    reaction_ids = [str(rid) for rid in payload]
    if args.limit:
        reaction_ids = reaction_ids[: args.limit]

    args.output_root.mkdir(parents=True, exist_ok=True)
    n = max(1, args.workers)
    slices = [reaction_ids[i::n] for i in range(n)]
    slice_files: list[Path] = []
    for index, slice_ids in enumerate(slices):
        path = args.output_root / f".worker_{index}_ids.json"
        path.write_text(json.dumps(slice_ids), encoding="utf-8")
        slice_files.append(path)

    processes: list[tuple[int, subprocess.Popen]] = []
    started = time.monotonic()
    for index, path in enumerate(slice_files):
        log = open(  # noqa: SIM115 - closed in the wait loop below
            args.output_root / f".worker_{index}.log", "w", encoding="utf-8"
        )
        process = subprocess.Popen(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/prepare_rl_origins.py"),
                "--reactions",
                str(path),
                "--output-root",
                str(args.output_root),
                "--acp-config",
                args.acp_config,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
        )
        processes.append((index, process))
        print(f"worker {index}: {len(slices[index])} reactions (pid {process.pid})")

    failures = 0
    for index, process in processes:
        code = process.wait()
        (args.output_root / f".worker_{index}.log").open("a").close()
        status = "ok" if code == 0 else f"exit {code}"
        print(f"worker {index}: finished ({status})")
        failures += code != 0
    print(
        f"all workers done in {time.monotonic() - started:.0f}s "
        f"({failures} failed)"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
