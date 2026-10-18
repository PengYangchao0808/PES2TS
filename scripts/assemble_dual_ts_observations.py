"""WP2-C CLI: bind the completed xTB observations into the dual-TS teacher
exports and produce the WP2 dataset manifest + C2 coverage audit.

Reads (never modified):
    outputs/pesn/dual_ts_v2_20261015/wp2_fit50/        50 task + 50 teacher
        payloads (states carry observation=None + placeholder reason) +
        coverage_nodes.json / coverage_summary.json / gaps_registry.json
    outputs/pesn/dual_ts_v2_20261015/wp2_observations/  content-addressed
        receipts <digest[:2]>/<digest>/{request.json,result.json} (v1
        cache hits are read from their recorded receipt paths)

Writes (NEW export dir; WP2-A originals are never overwritten):
    outputs/pesn/dual_ts_v2_20261015/wp2_fit50_observed/
        50x task.json (byte copies) + 50x teacher.json (observations bound)
        + coverage_nodes/coverage_summary/gaps_registry (byte copies)
        + gaps_registry_observed.json + observation_index.json
        + wp2_manifest.json

Hard gates in-loop: full coverage 6017/6017 (typed CoverageIncompleteError
otherwise), validate_record x100, A/B swap-digest identity x50,
assert_task_teacher_identity x50, reload-from-disk revalidation x50.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pes2ts_core.pesn.offline.data.dual_ts_assemble import (  # noqa: E402
    CoverageIncompleteError,
    assemble,
)

DEFAULT_FIT50_DIR = ROOT / "outputs/pesn/dual_ts_v2_20261015/wp2_fit50"
DEFAULT_OBSERVATIONS = ROOT / "outputs/pesn/dual_ts_v2_20261015/wp2_observations"
DEFAULT_OUT = ROOT / "outputs/pesn/dual_ts_v2_20261015/wp2_fit50_observed"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WP2-C dual-TS observation assembly "
                    "(fit50 -> wp2_fit50_observed + manifest)")
    parser.add_argument(
        "--fit50-dir", default=str(DEFAULT_FIT50_DIR),
        help="WP2-A fit50 export dir (read-only)")
    parser.add_argument(
        "--observations-dir", default=str(DEFAULT_OBSERVATIONS),
        help="WP2-B content-addressed receipt root (read-only)")
    parser.add_argument(
        "--out", default=str(DEFAULT_OUT),
        help="NEW export dir for the observed teacher table + manifest")
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="export even with missing receipts (states keep null+reason, "
             "misses registered in the gaps addendum); default is the "
             "strict full-coverage gate (typed error, nothing written)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.time()
    strict = not args.allow_partial
    print(f"[wp2c] fit50={args.fit50_dir}")
    print(f"[wp2c] observations={args.observations_dir}")
    print(f"[wp2c] out={args.out} strict_full_coverage={strict}")
    try:
        manifest = assemble(
            args.fit50_dir, args.observations_dir, args.out,
            require_full_coverage=strict)
    except CoverageIncompleteError as exc:
        print(f"[wp2c] FULL-COVERAGE GATE FAILED: {exc}")
        for miss in exc.missing[:20]:
            print(f"[wp2c]   MISS {miss['node_id']} "
                  f"digest={miss['node_digest']} status={miss['receipt_status']}")
        if len(exc.missing) > 20:
            print(f"[wp2c]   ... and {len(exc.missing) - 20} more")
        return 2
    audit = manifest["coverage_audit"]
    denom = manifest["denominators"]
    stats = manifest["observation_stats"]
    print(f"[done] coverage={audit['coverage']} ({audit['coverage_fraction']*100:.2f}%) "
          f"missing={audit['missing']} failures={audit['failures']} "
          f"per_reaction_complete={audit['per_reaction_complete']} "
          f"wall={time.time() - t0:.1f}s")
    print(f"[done] attribution={audit['attribution_counts']} "
          f"nproc_provenance={audit['nproc_provenance']}")
    print(f"[done] settings_digest_counts={audit['settings_digest_counts']}")
    print(f"[done] states bound={denom['states_bound_to_observations']}/"
          f"{denom['states']} (dual_sided_complete="
          f"{denom['dual_sided_complete']})")
    print(f"[done] E(hartree) min/median/max="
          f"{stats['energy_hartree']['min']}/"
          f"{stats['energy_hartree']['median']}/"
          f"{stats['energy_hartree']['max']}  "
          f"|g|max min/median/max={stats['gradient_max_abs']['min']}/"
          f"{stats['gradient_max_abs']['median']}/"
          f"{stats['gradient_max_abs']['max']}")
    disputed = [d["reaction_id"]
                for d in manifest["carry_forwards"]["disputed_reactions"]]
    print(f"[done] carry-forwards: disputed={disputed} "
          f"insertions={manifest['carry_forwards']['insertions']['nodes_inserted_total']}")
    print(f"[done] manifest -> {Path(args.out) / 'wp2_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
