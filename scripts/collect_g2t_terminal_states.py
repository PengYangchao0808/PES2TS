#!/usr/bin/env python
"""G2 退出门前置项① collector: 24/24 typed terminal states + complete ledgers.

G2-AB2 WP-7. Verifies, read-only, that a campaign root carries one
``pes2ts_g2t_terminal_state_v1`` per manifest case, that every terminal
references a complete ``costs.json`` ledger (R5: no ledger, no report), and
that the four-class denominators account for every case. Missing items are
NAMED, never auto-rerun (never-skip); the script exits non-zero while any
item is missing so it can gate the G2 exit checklist.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pes2ts_core.generation.campaign import (  # noqa: E402
    TERMINAL_STATE_SCHEMA,
    load_campaign_manifest,
)

EXIT_MISSING = 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="campaign output root")
    parser.add_argument("--manifest", required=True,
                        help="frozen campaign manifest (pes2ts_campaign_manifest_v1)")
    args = parser.parse_args(argv)

    root = Path(args.root)
    manifest = load_campaign_manifest(args.manifest)
    problems: list[str] = []
    rows: list[dict] = []
    for case in manifest["cases"]:
        rid = case["reaction_id"]
        terminal_path = root / rid / "terminal_state.json"
        if not terminal_path.is_file():
            problems.append(f"missing terminal_state: {rid}")
            continue
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        if terminal.get("schema_version") != TERMINAL_STATE_SCHEMA:
            problems.append(f"invalid terminal schema: {rid}")
            continue
        if terminal.get("campaign_sha256") != manifest["content_sha256"]:
            problems.append(f"terminal from another campaign: {rid}")
            continue
        if terminal.get("case_input_sha256") != case["snapshot_sha256"]:
            problems.append(f"input snapshot hash changed: {rid}")
            continue
        costs_path = root / rid / "costs.json"
        if not costs_path.is_file():
            problems.append(f"missing cost ledger: {rid}")
            continue
        costs = json.loads(costs_path.read_text(encoding="utf-8"))
        aggregate = costs.get("aggregate")
        if not isinstance(aggregate, dict):
            # A blocked case ran zero attempts: an empty-ledger bundle is the
            # auditable record; anything else without an aggregate is broken.
            blocked = terminal.get("terminal_class") == "blocked"
            if not (blocked and costs.get("ledgers") == []):
                problems.append(f"cost ledger without aggregate: {rid}")
                continue
            aggregate = {"N_gradient": 0, "cpu_seconds": None}
        rows.append({"reaction_id": rid,
                     "terminal_class": terminal.get("terminal_class"),
                     "label_state": terminal.get("label_state"),
                     "stage_reached": terminal.get("stage_reached"),
                     "failure_code": terminal.get("failure_code"),
                     "blocked_reason": terminal.get("blocked_reason"),
                     "n_gradient": aggregate.get("N_gradient"),
                     "cpu_seconds": aggregate.get("cpu_seconds")})
    four_class = {
        "rejected_typed": sum(1 for r in rows if r["terminal_class"] in
                              ("rejected_typed", "censored_budget")),
        "partial_prefix": sum(1 for r in rows if r["terminal_class"] == "partial_prefix"),
        "complete_path": sum(1 for r in rows if r["terminal_class"] == "complete_path"),
        "validated": sum(1 for r in rows if str(r["terminal_class"]).startswith("validated_")),
        "blocked": sum(1 for r in rows if r["terminal_class"] == "blocked"),
    }
    summary = {"campaign_sha256": manifest["content_sha256"],
               "n_cases": len(manifest["cases"]),
               "n_terminals": len(rows),
               "four_class_denominators": four_class,
               "cases": rows,
               "problems": problems}
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if problems:
        print(f"collect_g2t_terminal_states: {len(problems)} problem(s); "
              "missing items are listed above and are NOT auto-rerun",
              file=sys.stderr)
        return EXIT_MISSING
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
