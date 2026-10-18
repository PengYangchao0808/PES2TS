"""First-batch validation report generator (scale20k_rl2000 campaign).

Aggregates every evidence artifact of the S0/S1/S2 first batch into one
honest report: cohort funnels, origin-preparation ledgers, smoke results,
main RL50 collection, dev 3-way comparison, paired branches, BC/Q training
evidence, IRC 5k teacher metrics vs the 2k baseline and the pre-registered
targets, hard-gate checks, and budget consumption vs the frozen envelopes.

Missing stages are reported as ``not_run`` — the generator never fabricates
numbers and never relaxes a gate to make a section look better.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ROOT = REPO_ROOT / "outputs/pesn/scale20k_rl2000_v1_20261008"

TARGETS = {
    "direction_cosine": ("gte", 0.55),
    "activity_f1": ("gte", 0.65),
    "progress_mae": ("lte", 0.14),
    "pair_concordance": ("gte", 0.88),
}

BASELINE_2K = {
    "direction_cosine": 0.39726,
    "activity_f1": 0.55371,
    "progress_mae": 0.16315,
    "pair_concordance": 0.88241,
}

ENVELOPES = {
    "smoke": {"wall_seconds": 4500, "host_requests": 600},
    "main_collect_rl50": {"wall_seconds": 45000, "host_requests": 6000},
    "dev_3way": {"wall_seconds": 135000, "host_requests": 18000},
    "paired": {"wall_seconds": 46800, "host_requests": 3120},
}


def _load(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _ledger_totals(path: Path) -> dict:
    if not path.is_file():
        return {"wall_seconds": 0.0, "host_requests": 0, "rows": 0}
    wall = 0.0
    requests = 0
    rows = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows += 1
        wall += float(row.get("wall_seconds") or 0.0)
        requests += int(row.get("host_requests") or 0)
    return {"wall_seconds": round(wall, 1), "host_requests": requests, "rows": rows}


def _s0_section() -> dict:
    audit = _load(ROOT / "cohorts/s0_audit.json") or {}
    irc = _load(ROOT / "cohorts/irc_fit_cohorts.json") or {}
    rl = _load(ROOT / "cohorts/rl_cohorts.json") or {}
    return {
        "funnel": audit.get("funnel"),
        "pool_digest": audit.get("pool_digest"),
        "checks_passed": audit.get("checks_passed"),
        "nested_2k_in_5k": (
            set(irc.get("irc_fit_2000_verified", []))
            <= set(irc.get("irc_fit_5000", []))
            if irc
            else None
        ),
        "prescreen_ledgers": {
            "fit_rejections": len(rl.get("rl_fit_prescreen", {}).get("rejection_ledger", [])),
            "dev_replaced": rl.get("rl_dev_prescreen", {}).get("n_replaced_by_readiness"),
        },
    }


def _origin_prep_section() -> dict:
    sections = {}
    for name, path in (
        ("smoke", ROOT / "repair2/origin_prep_smoke/origin_prep_summary.json"),
        ("main", ROOT / "rl_0050/origin_prep_main/origin_prep_summary.json"),
    ):
        doc = _load(path)
        if doc is None:
            prepared = len(list((path.parent).glob("*/plan.json"))) if path.parent.is_dir() else 0
            sections[name] = {"status": "in_progress_or_missing", "plans_present": prepared}
        else:
            sections[name] = {
                "status": "complete",
                "funnel": doc.get("funnel"),
                "n_selected": doc.get("n_selected"),
                "total_wall_seconds": doc.get("total_wall_seconds"),
            }
    return sections


def _round_section(round_dir: Path) -> dict:
    collect = _load(round_dir / "collect_summary.json")
    episode_index = _load(round_dir / "transitions/episode_index.json")
    ledger = _ledger_totals(round_dir / "cost_ledger.jsonl")
    store = None
    transitions = round_dir / "transitions/control_transitions.jsonl"
    if transitions.is_file():
        sources: dict[str, int] = {}
        n = 0
        for line in transitions.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            n += 1
            source = str(row.get("action", {}).get("source"))
            sources[source] = sources.get(source, 0) + 1
        store = {"n_transitions": n, "executed_action_sources": sources}
    snapshots = len(list(round_dir.glob("cases/*/control/snapshots/d*.json")))
    snapshots += len(list(round_dir.glob("execute/*/*/control/snapshots/d*.json")))
    executes = {}
    for summary_path in sorted(round_dir.glob("execute/*/execute_summary.json")):
        doc = _load(summary_path)
        if doc:
            cases = doc.get("cases", [])
            completes = sum(
                1 for c in cases if str(c.get("status")) == "complete_interval"
            )
            executes[summary_path.parent.name] = {
                "n_denominator": doc.get("n_denominator"),
                "n_cases": doc.get("n_cases"),
                "complete_interval": completes,
                "support_gate": doc.get("support_gate"),
                "q_support_stats": doc.get("q_support_stats"),
            }
    return {
        "collect": (
            {
                "n_cases": collect.get("n_cases"),
                "n_executed": collect.get("n_executed"),
                "n_resumed": collect.get("n_resumed"),
                "append_stats": collect.get("transition_append_stats"),
                "environment_hash": collect.get("environment_hash", "")[:12],
            }
            if collect
            else None
        ),
        "episodes": (
            {
                "n_episodes": len(episode_index.get("episodes", [])),
                "terminals": sorted(
                    {
                        str(e.get("terminal_status"))
                        for e in episode_index.get("episodes", [])
                    }
                ),
            }
            if episode_index
            else None
        ),
        "transition_store": store,
        "snapshots": snapshots,
        "cost": ledger,
        "executes": executes,
    }


def _irc_section() -> dict:
    run_dirs = sorted((ROOT / "irc_5k").glob("g3p1-*"))
    if not run_dirs:
        return {"status": "not_run"}
    run = run_dirs[-1]
    run_doc = _load(run / "run.json") or {}
    metrics = _load(run / "metrics.json")
    dataset = _load(ROOT / "irc_5k/teacher_dataset/dataset_manifest.json") or {}
    section: dict = {
        "status": run_doc.get("status", "unknown"),
        "n_train_reactions": 5000,
        "dataset_samples": dataset.get("n_samples"),
        "dataset_matches_frozen_cohort": dataset.get("n_reactions") == 5250,
    }
    if metrics:
        model_metrics = metrics.get("model") or {}
        keys = {
            "direction_cosine": model_metrics.get("direction_cosine_mean"),
            "activity_f1": model_metrics.get("activity_f1"),
            "progress_mae": model_metrics.get("mae_s"),
            "pair_concordance": model_metrics.get("pair_concordance"),
        }
        verdicts = {}
        for name, (op, target) in TARGETS.items():
            value = keys.get(name)
            if value is None:
                verdicts[name] = {"value": None, "target": target, "verdict": "missing"}
                continue
            ok = value >= target if op == "gte" else value <= target
            verdicts[name] = {
                "value": round(value, 5),
                "baseline_2k": BASELINE_2K[name],
                "target": target,
                "verdict": "pass" if ok else "below_target",
            }
        section["metrics_vs_targets"] = verdicts
    else:
        section["metrics_vs_targets"] = "metrics_not_written_yet"
    return section


def _paired_section() -> dict:
    doc = _load(ROOT / "paired/paired_branches_summary.json")
    if not doc:
        return {"status": "not_run"}
    zeros = [
        b for b in doc.get("branches", [])
        if b.get("branch") == "b00_zero" and b.get("cost")
    ]
    repeats = [
        b for b in doc.get("branches", [])
        if b.get("branch") == "b03_zero_repeat" and b.get("cost")
    ]
    zero_walls = [b["cost"].get("wall_seconds", 0.0) for b in zeros + repeats]
    return {
        "status": "complete",
        "shortfall": doc.get("shortfall"),
        "funnel": doc.get("funnel"),
        "cost_totals": doc.get("cost_totals"),
        "n_zero_branches": len(zero_walls),
        "zero_wall_spread": {
            "min": round(min(zero_walls), 2) if zero_walls else None,
            "max": round(max(zero_walls), 2) if zero_walls else None,
        },
    }


def _training_section() -> dict:
    round_dir = ROOT / "rl_0050/round_main"
    section = {"status": "not_run"}
    for mode in ("bc", "cql"):
        summary = _load(round_dir / f"train_summary_{mode}.json")
        freeze = _load(round_dir / f"freeze_summary_{mode}.json")
        if summary:
            section[f"{mode}_train"] = {
                "run_id": summary.get("run_id"),
                "best_epoch": summary.get("best_epoch"),
                "q_params_changed": summary.get("q_params_changed"),
                "encoder_params_changed": summary.get("encoder_params_changed"),
                "td_registry_sha256": (summary.get("td_support_registry_sha256") or "")[:12],
                "registry_consistent": summary.get("support_registry_td_export_consistent"),
                "n_imitation_distinct": summary.get("n_imitation_distinct_samples"),
            }
        if freeze:
            section[f"{mode}_freeze"] = {
                "bundle": freeze.get("bundle_path"),
                "registry_bound": (freeze.get("support_registry") or {}).get("bound"),
            }
    if any(k for k in section if k != "status"):
        section["status"] = "partial" if not section.get("cql_freeze") else "complete"
    return section


def _hard_gates_section() -> dict:
    """Explicit hard-gate checks (plan §5/§10): identity, support, registry.

    Every check reads actual artifacts; a missing artifact is reported as
    ``not_run``, never as a pass.
    """
    gates: dict[str, object] = {}
    env_prefix = "cfeea21a73fc"
    rounds = {
        "smoke": ROOT / "repair2/smoke_round",
        "main": ROOT / "rl_0050/round_main",
        "dev": ROOT / "dev/dev50_round",
    }
    env_checks = {}
    for name, round_dir in rounds.items():
        collect = _load(round_dir / "collect_summary.json")
        if collect:
            env_checks[name] = str(collect.get("environment_hash", "")).startswith(env_prefix)
        for group_summary in sorted(round_dir.glob("execute/*/execute_summary.json")):
            doc = _load(group_summary)
            if doc:
                env_checks[f"{name}:{group_summary.parent.name}"] = str(
                    doc.get("environment_hash", "")
                ).startswith(env_prefix)
    gates["environment_hash_consistency"] = {
        "checked": env_checks,
        "pass": bool(env_checks) and all(env_checks.values()),
    }

    registry_checks = {}
    for mode in ("bc", "cql"):
        train = _load(ROOT / f"rl_0050/round_main/train_summary_{mode}.json")
        freeze = _load(ROOT / f"rl_0050/round_main/freeze_summary_{mode}.json")
        if train:
            registry_checks[f"{mode}_td_export_consistent"] = bool(
                train.get("support_registry_td_export_consistent")
            )
        if freeze:
            registry_checks[f"{mode}_bundle_registry_bound"] = bool(
                (freeze.get("support_registry") or {}).get("bound")
            )
    gates["registry_binding"] = {
        "checked": registry_checks,
        "pass": bool(registry_checks) and all(registry_checks.values()),
    }

    support_gates = {}
    for group in ("bc_residual", "residual_q"):
        doc = _load(ROOT / f"dev/dev50_round/execute/{group}/execute_summary.json")
        if doc:
            gate = doc.get("support_gate") or {}
            support_gates[group] = {
                "kind": gate.get("kind"),
                "legacy_support": gate.get("legacy_support"),
                "registry_sha256": (gate.get("registry_sha256") or "")[:12],
            }
    gates["execution_support_gate"] = {
        "checked": support_gates,
        "residual_q_uses_bound_registry": (
            support_gates.get("residual_q", {}).get("kind") == "registry_v1_bound"
            if support_gates.get("residual_q")
            else "not_run"
        ),
    }

    episode_index = _load(ROOT / "rl_0050/round_main/transitions/episode_index.json")
    if episode_index:
        episodes = episode_index.get("episodes", [])
        ids = [e.get("episode_id") for e in episodes]
        broken = [
            e.get("episode_id")
            for e in episodes
            if len(e.get("compatible_links", [])) != max(0, e.get("n_decisions", 1) - 1)
        ]
        gates["episode_identity"] = {
            "n_episodes": len(episodes),
            "duplicate_ids": len(ids) - len(set(ids)),
            "episodes_with_broken_links": broken,
            "pass": len(ids) == len(set(ids)) and not broken,
        }
    else:
        gates["episode_identity"] = "not_run"
    return gates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "reports/first_batch_report_v1.json",
    )
    args = parser.parse_args()

    smoke_round = ROOT / "repair2/smoke_round"
    main_round = ROOT / "rl_0050/round_main"
    dev_round = ROOT / "dev/dev50_round"

    report = {
        "schema_version": "pes2ts_scale_first_batch_report_v1",
        "campaign_root": str(ROOT),
        "s0": _s0_section(),
        "origin_prep": _origin_prep_section(),
        "smoke": _round_section(smoke_round),
        "main_rl50": _round_section(main_round),
        "dev_round": _round_section(dev_round),
        "paired": _paired_section(),
        "training": _training_section(),
        "irc_5k": _irc_section(),
        "hard_gates": _hard_gates_section(),
        "budget_envelopes": ENVELOPES,
        "notes": (
            "Counts follow the plan units: IRC = canonical train_fit groups; "
            "RL = new groups with one real main episode; failures count as "
            "attempts; paired branches are diagnostic and never main "
            "episodes. Metric targets are pre-registered research goals, "
            "not effect claims."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    compact = {
        k: report[k]
        for k in ("s0", "paired", "training", "irc_5k")
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2, default=str)[:4000])
    print(f"\nfull report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
