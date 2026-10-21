#!/usr/bin/env python3
"""V5.3-MP cohort freeze: ranked candidate lists for the 20k build.

Consumes the S0-A qualification (``eligible_reactions.parquet``) and the
frozen G3-P1 predictor split (parent; sha-pinned) and freezes the V5.3-MP
cohort: ranked pool candidates (train_fit side), a dev-eval candidate set
(train_dev side), and the confirm candidate list (train_calibration side —
ids only; confirm reactions are NEVER built/touched in this campaign until
the final, separately-authorized confirmation stage).

Ranking is deterministic: ``sha256(f"{SEED_KEY}:{canonical_group}")``.

Output: ``outputs/pesn/v53mp/cohort/v53mp_cohort_v1.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA = "v53mp_cohort_v1"
SEED_KEY = "v53mp:cohort:20261021"
POOL_CANDIDATE_TARGET = 26_000   # margin over 20,000 for build failures
DEV_TARGET = 600
CONFIRM_TARGET = 2_000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rank(key: str) -> str:
    return hashlib.sha256(f"{SEED_KEY}:{key}".encode("utf-8")).hexdigest()


def freeze_cohort(repo: Path, s0a_dir: Path, out_dir: Path) -> dict:
    repo = repo.resolve()
    s0a_dir = s0a_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(s0a_dir / "eligible_reactions.parquet").to_pylist()
    split_path = repo / "outputs/pesn/g3_p1/splits/pesn_irc_split.json"

    # one build row per canonical group: build the ELIGIBLE MEMBER (prefer
    # the member whose id == canonical id); the scope check applies to the
    # member reaction id (the built curve carries that id)
    by_group: dict[str, dict] = {}
    for row in rows:
        gid = str(row["canonical_group"])
        cur = by_group.get(gid)
        if cur is None or (cur["reaction_id"] != gid
                           and row["reaction_id"] == gid):
            by_group[gid] = row

    def ranked(subset: str) -> list[dict]:
        members = [r for r in by_group.values() if r["subset"] == subset]
        return sorted(members, key=lambda r: (_rank(r["canonical_group"]),
                                              r["canonical_group"]))

    pool = ranked("train_fit")[:POOL_CANDIDATE_TARGET]
    dev = ranked("train_dev")[:DEV_TARGET]
    confirm = ranked("train_calibration")[:CONFIRM_TARGET]

    cohort = {
        "schema": SCHEMA,
        "seed_key": SEED_KEY,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_split": {
            "path": str(split_path.relative_to(repo)),
            "sha256": sha256_file(split_path),
            "schema_version": "pesn_irc_split_v1",
        },
        "s0a_qualification": {
            "path": str((s0a_dir / "metadata_qualification.json")
                        .relative_to(repo)),
            "sha256": sha256_file(s0a_dir / "metadata_qualification.json"),
        },
        "targets": {"pool_candidates": POOL_CANDIDATE_TARGET,
                    "dev": DEV_TARGET, "confirm_listed": CONFIRM_TARGET},
        "pool_candidates": [r["reaction_id"] for r in pool],
        "dev": [r["reaction_id"] for r in dev],
        "confirm": [r["reaction_id"] for r in confirm],
        "canonical_groups": {
            "pool": [r["canonical_group"] for r in pool],
            "dev": [r["canonical_group"] for r in dev],
            "confirm": [r["canonical_group"] for r in confirm],
        },
        "counts": {
            "pool_candidates": len(pool),
            "dev": len(dev),
            "confirm": len(confirm),
        },
        "notes": [
            "ranking = sha256(seed_key:canonical_group) ascending",
            "pool candidates carry build margin over the 20,000 target; the "
            "frozen 20k pool is fixed AFTER the build (first 20,000 qualified "
            "groups in this rank order)",
            "confirm ids are listed ONLY; confirm reactions are not read or "
            "built in this campaign until the separately-authorized final "
            "confirmation stage",
        ],
    }
    out = out_dir / "v53mp_cohort_v1.json"
    out.write_text(json.dumps(cohort, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    cohort["cohort_sha256"] = sha256_file(out)
    return cohort


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="r3_v53mp_freeze_cohort")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--s0a", default="outputs/pesn/v53mp/s0a")
    parser.add_argument("--out", default="outputs/pesn/v53mp/cohort")
    args = parser.parse_args(argv)
    cohort = freeze_cohort(
        Path(args.repo_root).resolve(), Path(args.s0a), Path(args.out))
    print(json.dumps({
        "counts": cohort["counts"],
        "cohort_sha256": cohort["cohort_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
