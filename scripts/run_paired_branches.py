"""Paired action-branch collection from saved decision snapshots (plan §8.2).

For each selected fit/dev decision state (saved as
``pes2ts_control_state_snapshot_v1`` during the main/dev collection rounds)
this driver executes THREE diagnostic branches with the shared macro-step
seam: the zero residual plus two deterministic legal non-zero probes.
Selected states additionally repeat the zero branch to calibrate compute
noise (8 fit / 4 dev).  Branches inherit the snapshot's remaining budget,
write isolated branch documents, never touch the round transition store,
and record per-branch cost against the 13h/3120-request paired envelope.

State selection uses only recorded observations (decision index spread,
rejection/budget pressure in the context vector); reference TS material is
never consulted.  State-count shortfalls are reported, never padded.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pes2ts_core.generation.control_round import (  # noqa: E402
    _load_round_case,
    build_reward_spec,
    build_scan_policy,
    load_control_round_config,
)
from pes2ts_core.generation.planning.control_replay import (  # noqa: E402
    run_replay_branch,
)
from pes2ts_core.generation.planning.control_snapshot import (  # noqa: E402
    snapshot_content_sha256,
)
from pes2ts_core.pesn.offline.transitions import (  # noqa: E402
    context_vector_from_doc,
    load_transition_set,
)
from pes2ts_core.utils.jsonio import write_json  # noqa: E402

FIT_STATES = 32
DEV_STATES = 16
FIT_ZERO_REPEATS = 8
DEV_ZERO_REPEATS = 4

ENVELOPE_WALL_SECONDS = 46_800.0
ENVELOPE_HOST_REQUESTS = 3_120


def _transport_factory(manifest_path: Path):
    from pes2ts_core.generation.campaign import load_campaign_manifest
    from pes2ts_core.integration.acp.xtb_constraints_transport import (
        run_xtb_constopt_attempt,
    )

    manifest = load_campaign_manifest(manifest_path)
    acp_wiring = manifest["acp_wiring"]

    def transport(request, attempt_dir, execution_id, attempt_id):
        return run_xtb_constopt_attempt(
            acp_root=acp_wiring["root"],
            python_executable=acp_wiring.get("python"),
            acp_config_path=acp_wiring.get("config_path"),
            request=request,
            output_root=attempt_dir,
            execution_id=execution_id,
            attempt_id=attempt_id,
            timeout_seconds=7200.0,
        )

    return transport


def _select_states(round_dir: Path, *, wanted: int) -> list[dict]:
    """Deterministic spread of decision snapshots across fit reactions.

    Accepts both round layouts: collect rounds keep snapshots under
    ``cases/<rid>/control/snapshots``, execute-group rounds under
    ``execute/<group>/<rid>/control/snapshots``.
    """
    paths = list(round_dir.glob("cases/*/control/snapshots/d*.json"))
    paths += list(round_dir.glob("execute/*/*/control/snapshots/d*.json"))
    candidates: list[dict] = []
    for snapshot_path in sorted(paths):
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        context = _snapshot_context(snapshot)
        candidates.append(
            {
                "path": str(snapshot_path),
                "reaction_id": snapshot["episode"]["reaction_id"],
                "decision_index": int(snapshot["episode"]["decision_index"]),
                "context": context,
                "n_drivers": None,
                "snapshot_sha256": snapshot_content_sha256(snapshot),
            }
        )
    by_reaction: dict[str, list[dict]] = {}
    for entry in sorted(
        candidates, key=lambda item: (item["reaction_id"], item["decision_index"])
    ):
        by_reaction.setdefault(entry["reaction_id"], []).append(entry)
    selected: list[dict] = []
    order = sorted(by_reaction)
    offset = 0
    while len(selected) < wanted and any(by_reaction.values()):
        advanced = False
        for reaction_id in order:
            states = by_reaction[reaction_id]
            if offset < len(states):
                selected.append(states[offset])
                advanced = True
                if len(selected) >= wanted:
                    break
        if not advanced:
            break
        offset += 1
    return selected


def _snapshot_context(snapshot: dict) -> list[float]:
    counters = snapshot.get("counters", {})
    budget = snapshot.get("budget", {})
    pending = snapshot.get("physical_state", {}).get("pending", [])
    return [
        float(counters.get("recent_rejections", 0)),
        float(budget.get("request_count", 0)),
        float(len(pending)),
    ]


def _legal_probes(spec, snapshot: dict) -> list[list[float]]:
    """Two deterministic legal non-zero probes bounded by the frozen spec."""
    epsilon = [float(e) for e in spec.epsilon]
    n = len(epsilon)
    probes: list[list[float]] = []
    if n >= 2:
        probes.append([0.5 * epsilon[0]] + [0.0] * (n - 1))
        probes.append([0.0, 0.5 * epsilon[1]] + [0.0] * (n - 2))
    else:
        probes.append([0.5 * epsilon[0]])
        probes.append([1.0 * epsilon[0]])
    return probes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-round-dir", type=Path, required=True)
    parser.add_argument("--dev-round-dir", type=Path, required=True,
                        help="round dir holding the dev rule_zero episodes")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--round-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fit-states", type=int, default=FIT_STATES)
    parser.add_argument("--dev-states", type=int, default=DEV_STATES)
    args = parser.parse_args()

    config = load_control_round_config(args.round_config)
    scan_policy = build_scan_policy(config)
    reward_spec = build_reward_spec(config)
    transport = _transport_factory(args.manifest)

    fit_states = _select_states(args.fit_round_dir, wanted=args.fit_states)
    dev_states = _select_states(args.dev_round_dir, wanted=args.dev_states)
    shortfall = {
        "fit_states_wanted": args.fit_states,
        "fit_states_found": len(fit_states),
        "dev_states_wanted": args.dev_states,
        "dev_states_found": len(dev_states),
    }

    case_cache: dict[str, object] = {}

    def case_for(reaction_id: str):
        if reaction_id not in case_cache:
            case_cache[reaction_id] = _load_round_case(
                reaction_id=reaction_id,
                manifest=json.loads(args.manifest.read_text(encoding="utf-8")),
                manifest_path=args.manifest,
                legacy_root=args.legacy_root,
                config=config,
                scan_policy=scan_policy,
            )
        return case_cache[reaction_id]

    rows: list[dict] = []
    funnel: Counter[str] = Counter()
    total_wall = 0.0
    total_requests = 0

    def run_branch(state: dict, candidate_u, source: str, label: str, subdir: str):
        nonlocal total_wall, total_requests
        reaction_id = state["reaction_id"]
        snapshot = json.loads(Path(state["path"]).read_text(encoding="utf-8"))
        case = case_for(reaction_id)
        branch_dir = (
            args.output_root / subdir / reaction_id / f"d{state['decision_index']:03d}" / label
        )
        started = time.monotonic()
        try:
            branch = run_replay_branch(
                control=case.control,
                snapshot=snapshot,
                candidate_u=candidate_u,
                candidate_source=source,
                transport=transport,
                save_dir=branch_dir,
                reward_spec=reward_spec,
                spec=case.spec,
                policy=scan_policy,
                branch_label=label,
            )
            cost = branch.get("cost", {})
            total_wall += float(cost.get("wall_seconds", 0.0))
            total_requests += int(cost.get("host_requests", 0))
            funnel[str(branch["outcome"]["status"])] += 1
            rows.append(
                {
                    "subset": subdir,
                    "reaction_id": reaction_id,
                    "decision_index": state["decision_index"],
                    "branch": label,
                    "candidate_u": list(candidate_u),
                    "source": source,
                    "outcome": branch["outcome"],
                    "cost": cost,
                    "snapshot_sha256": state["snapshot_sha256"],
                    "branch_dir": str(branch_dir),
                }
            )
        except Exception as exc:  # noqa: BLE001 - typed row, batch continues
            funnel[f"error:{type(exc).__name__}"] += 1
            rows.append(
                {
                    "subset": subdir,
                    "reaction_id": reaction_id,
                    "decision_index": state["decision_index"],
                    "branch": label,
                    "candidate_u": list(candidate_u),
                    "source": source,
                    "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                    "snapshot_sha256": state["snapshot_sha256"],
                    "wall_seconds": round(time.monotonic() - started, 2),
                }
            )
        if total_wall > ENVELOPE_WALL_SECONDS or total_requests > ENVELOPE_HOST_REQUESTS:
            raise SystemExit(
                f"paired budget envelope reached ({total_wall:.0f}s / "
                f"{total_requests} requests); stopping with terminal state"
            )

    zero = None
    for index, state in enumerate(fit_states):
        case = case_for(state["reaction_id"])
        zero = [0.0] * len(case.spec.epsilon)
        probes = _legal_probes(case.spec, state)
        run_branch(state, zero, "rule_zero", "b00_zero", "paired_fit")
        run_branch(state, probes[0], "paired_probe", "b01_probe_a", "paired_fit")
        run_branch(state, probes[1], "paired_probe", "b02_probe_b", "paired_fit")
        if index < FIT_ZERO_REPEATS:
            run_branch(state, zero, "rule_zero", "b03_zero_repeat", "paired_fit")
    for index, state in enumerate(dev_states):
        case = case_for(state["reaction_id"])
        zero = [0.0] * len(case.spec.epsilon)
        probes = _legal_probes(case.spec, state)
        run_branch(state, zero, "rule_zero", "b00_zero", "paired_dev")
        run_branch(state, probes[0], "paired_probe", "b01_probe_a", "paired_dev")
        run_branch(state, probes[1], "paired_probe", "b02_probe_b", "paired_dev")
        if index < DEV_ZERO_REPEATS:
            run_branch(state, zero, "rule_zero", "b03_zero_repeat", "paired_dev")

    summary = {
        "schema_version": "pes2ts_scale_paired_branches_v1",
        "shortfall": shortfall,
        "funnel": dict(sorted(funnel.items())),
        "cost_totals": {
            "wall_seconds": round(total_wall, 2),
            "host_requests": total_requests,
            "envelope_wall_seconds": ENVELOPE_WALL_SECONDS,
            "envelope_host_requests": ENVELOPE_HOST_REQUESTS,
        },
        "n_states": {"fit": len(fit_states), "dev": len(dev_states)},
        "branches": rows,
    }
    write_json(args.output_root / "paired_branches_summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("shortfall", "funnel", "cost_totals")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
