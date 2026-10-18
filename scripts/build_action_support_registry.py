"""Build an action-support registry artifact for one control round (WP-2).

Derives ``pesn_action_support_registry_v1`` from the round's FIT transitions
only (the trainer's deterministic reaction-group split: seed +
dev_fraction), writes it beside the run artifacts, and prints the frozen
calibration.  Read-only with respect to recorded transitions; never triggers
QC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pes2ts_core.generation.control_round import (  # noqa: E402
    build_reward_spec,
    load_control_round_config,
)
from pes2ts_core.pesn.inference.action_support import (  # noqa: E402
    ActionSupportRegistry,
)
from pes2ts_core.pesn.offline.action_support import (  # noqa: E402
    build_action_support_registry,
)
from pes2ts_core.pesn.offline.control_trainer import (  # noqa: E402
    ControlTrainSettings,
    split_fit_dev_reactions,
)
from pes2ts_core.pesn.offline.transitions import load_transition_set  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-dir", required=True)
    parser.add_argument("--round-config", required=True)
    parser.add_argument("--selection", required=True, help="train selection manifest")
    parser.add_argument("--output", default=None, help="output JSON path")
    parser.add_argument("--mode", default="cql", choices=["cql", "bc_only"],
                        help="mode tag selects the effective config for the split")
    args = parser.parse_args()

    round_dir = Path(args.round_dir)
    config = load_control_round_config(args.round_config)
    from pes2ts_core.generation.control_round import effective_training_config

    effective = effective_training_config(config, args.mode)
    settings = ControlTrainSettings.from_config(effective)

    freeze_doc = json.loads(
        (round_dir / "environment_freeze.json").read_text(encoding="utf-8")
    )
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    load = load_transition_set(
        round_dir,
        environment_hash=str(freeze_doc["environment_hash"]),
        allowed_reactions=list(selection["reaction_ids"]),
    )
    if not load.qualified:
        print(f"no qualified transitions in {round_dir}", file=sys.stderr)
        return 2
    fit_reactions, dev_reactions = split_fit_dev_reactions(
        load.qualified,
        dev_fraction=settings.dev_fraction,
        seed=settings.seed,
    )
    fit_set = set(fit_reactions)
    fit_transitions = [t for t in load.qualified if t.reaction_id in fit_set]
    reward_spec = build_reward_spec(effective)
    doc = build_action_support_registry(
        fit_transitions,
        environment_hash=str(freeze_doc["environment_hash"]),
        reward_spec_sha256=reward_spec.content_hash(),
    )
    doc["built_from"]["fit_reactions"] = sorted(fit_set)
    doc["built_from"]["dev_reactions_excluded"] = sorted(set(dev_reactions))
    doc["built_from"]["generated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    registry = ActionSupportRegistry(doc)

    output = Path(args.output) if args.output else (
        round_dir / "action_support_registry_fit.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(doc, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    n_failed = sum(
        1 for group in doc["groups"] for entry in group["entries"]
        if not entry["accepted"]
    )
    print(
        f"registry: {output}\n"
        f"  groups={len(doc['groups'])} fit_entries={doc['built_from']['n_fit_transitions']} "
        f"failed_entries={n_failed}\n"
        f"  calibration={json.dumps(doc['calibration'], sort_keys=True)}\n"
        f"  sha256={digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
