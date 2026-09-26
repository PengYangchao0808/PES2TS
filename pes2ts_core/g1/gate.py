"""G2 eligibility gate: the explicit release decision after P1 and P2.

:func:`run_gate` reads the P1 and P2 summaries plus both verification
outcomes and writes the release manifest: which reaction ids may enter the
endpoint-only G2 stage, the full denominator accounting (every inventory
reaction is either eligible or carries an exclusion reason), and the
fraction the plan demands be reported alongside the absolute count.  The
eligible id list is exported for G2 consumption as pure reaction metadata
(no truth-derived fields).  The gate never mutates the frozen G0 split and
never invents a verdict: missing or failed verifications refuse with an
error instead of gating on partial evidence.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from pes2ts_core.g1.p1_verify import verify_p1
from pes2ts_core.g1.p2_verify import verify_p2
from pes2ts_core.g1.truth_schema import (
    G2_ELIGIBLE_FILENAME,
    G2_ELIGIBLE_STATUSES,
    GATE_MANIFEST_FILENAME,
    GATE_SCHEMA_VERSION,
    P1_SUMMARY_FILENAME,
    P2_SUMMARY_FILENAME,
    STATUS_CLASSIFIED,
)
from pes2ts_core.utils.hashing import JSONValue
from pes2ts_core.utils.jsonio import write_json
from pes2ts_core.utils.parquet_io import read_parquet

#: Actionable hint for missing prerequisites.
GATE_HINT: Final[str] = "run `g1 resolve-map --allow-truth`, `g1 classify`, then `g1 verify --stage p1`/`--stage p2`"


@dataclass(frozen=True, slots=True)
class GateResult:
    """Outcome of one gate run."""

    n_denominator: int
    n_eligible: int
    eligible_fraction: float
    manifest_path: Path
    eligible_path: Path
    refusals: tuple[str, ...]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def run_gate(config: Mapping[str, Any], *, assume_verified: bool = False) -> GateResult:
    """Write the G2 gate manifest and the eligible id list.

    With ``assume_verified=False`` (the default) both verifications run
    first and any problem refuses the gate.  ``assume_verified=True``
    (used by ``g1 verify --stage all`` flows that already ran them) skips
    the re-verification, not the accounting.
    """
    interim_dir = Path(config["paths"]["interim"])
    manifests_dir = Path(config["paths"]["manifests"])
    p1_summary_path = interim_dir / P1_SUMMARY_FILENAME
    p2_summary_path = interim_dir / P2_SUMMARY_FILENAME
    for path in (p1_summary_path, p2_summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; {GATE_HINT}")
    refusals: list[str] = []
    if not assume_verified:
        p1_check = verify_p1(config)
        refusals.extend(f"p1: {problem}" for problem in p1_check.problems)
        p2_check = verify_p2(config)
        refusals.extend(f"p2: {problem}" for problem in p2_check.problems)
    p1_rows = {
        str(row["reaction_id"]): row
        for row in read_parquet(p1_summary_path).to_pylist()
    }
    p2_rows = {
        str(row["reaction_id"]): row
        for row in read_parquet(p2_summary_path).to_pylist()
    }
    eligible_ids: list[str] = []
    exclusions: Counter[str] = Counter()
    for reaction_id in sorted(p1_rows):
        p1_status = str(p1_rows[reaction_id]["status"])
        p2_status = str(p2_rows.get(reaction_id, {}).get("status") or "missing_p2")
        if p1_status in G2_ELIGIBLE_STATUSES and p2_status == STATUS_CLASSIFIED:
            eligible_ids.append(reaction_id)
        elif p1_status not in G2_ELIGIBLE_STATUSES:
            exclusions[f"p1:{p1_status}"] += 1
        else:
            exclusions[f"p2:{p2_status}"] += 1
    n_denominator = len(p1_rows)
    fraction = round(len(eligible_ids) / n_denominator, 6) if n_denominator else 0.0
    manifest: dict[str, JSONValue] = {
        "schema_version": GATE_SCHEMA_VERSION,
        "n_denominator": n_denominator,
        "n_eligible": len(eligible_ids),
        "eligible_fraction": fraction,
        "exclusions": dict(sorted(exclusions.items())),
        "refusals": refusals,
        "gate_pass": not refusals,
        "generated_at": _now(),
    }
    manifest_path = manifests_dir / GATE_MANIFEST_FILENAME
    write_json(manifest_path, manifest)
    eligible_path = interim_dir / G2_ELIGIBLE_FILENAME
    write_json(eligible_path, {
        "schema_version": GATE_SCHEMA_VERSION,
        "reaction_ids": eligible_ids,
        "n_eligible": len(eligible_ids),
        "note": (
            "reaction ids eligible for endpoint-only G2; endpoint metadata "
            "only, no TS/IRC-derived fields"
        ),
        "generated_at": _now(),
    })
    return GateResult(
        n_denominator=n_denominator,
        n_eligible=len(eligible_ids),
        eligible_fraction=fraction,
        manifest_path=manifest_path,
        eligible_path=eligible_path,
        refusals=tuple(refusals),
    )


__all__ = [
    "GATE_HINT",
    "GateResult",
    "run_gate",
]
