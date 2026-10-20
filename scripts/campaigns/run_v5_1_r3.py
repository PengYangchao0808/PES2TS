"""30h-campaign R3 driver (WP-L3 root contract + E3 training execution).

Subcommands
-----------
``check``
    Assert the output root is inside ``outputs/pesn/30h_campaign/**``
    (sealed v5_1/v5_1_r1 prefixes refuse fail-closed) and print it.
``smoke``
    Dev-throughput probe (100-300 updates, GPU lock held with metadata,
    NO ledger registration; recorded in the handoff for phase budgeting).
``run``
    Formal planned run: plan lookup -> register (planned) ->
    ``monitor.attempt_launch(stage="g0")`` (plan pin + five-sha binding +
    budget caps; NEVER a raw ledger launch) -> GPU lock with metadata ->
    train -> checkpoint + metrics under ``r3_runs/<run_id>/`` -> ``done``
    transition carrying ``checkpoint_sha256`` + ``metrics_path``.
``block-confirm``
    Register + mark every confirm arm (C / QT-confirm, folds3/4) as
    ``blocked_with_reason=confirm_awaiting_dev_positive`` (never launched).

Frozen-contract guards in ``run``: formal updates are exactly 3000 (no
override); aux arms refuse without an explicit calibrated λ; R1/R2 refuse
(ramp spec not defined in the frozen config); feature level comes from the
plan arm mapping (or an explicit ``--level`` for post-G2 B/L/V arms).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agent_ops.gpu_queue.gpu_lock import acquire as gpu_acquire  # noqa: E402
from scripts.agent_ops.ledger.run_ledger import (  # noqa: E402
    LedgerError,
    RunLedger,
    compute_run_id,
    read_run_plan,
)
from scripts.agent_ops.supervisor.campaign_root import (  # noqa: E402
    CAMPAIGN_ROOT,
    CampaignRootError,
    assert_campaign_root,
)
from scripts.agent_ops.supervisor.monitor import attempt_launch  # noqa: E402
from scripts.agent_ops.supervisor.stage_bind import (  # noqa: E402
    DEFAULT_TASK_STATE_PATH,
    assert_stage_binding,
)

#: WP-L3: R3 driver output root -- inside the campaign, NEVER the sealed dirs.
DEFAULT_OUTPUT_ROOT = CAMPAIGN_ROOT + "/r3_runs"
DEFAULT_PLAN = CAMPAIGN_ROOT + "/run_plan.jsonl"
DEFAULT_LEDGER = CAMPAIGN_ROOT + "/run_records.jsonl"
DEFAULT_DATA_DIR = CAMPAIGN_ROOT + "/data"
DEFAULT_SPLIT = CAMPAIGN_ROOT + "/data/split_manifest_r3.json"
DEFAULT_METRIC_UNSEAL = "docs/agent_runs/state/gates/metric_unseal.json"
GPU_LOCK = "/tmp/opencode/pes2ts_gpu.lock"
FORMAL_UPDATES = 3000
CONFIRM_BLOCK_REASON = "confirm_awaiting_dev_positive"

#: plan arm -> (feature level, trainer switches)
ARM_MAP: Dict[str, Dict[str, Any]] = {
    "F0": {"level": "F0"},
    "F1": {"level": "F1"},
    "F2": {"level": "F2"},
    "F3": {"level": "F3"},
    "F2-bidir": {"level": "F2", "bidirectional": True},
    "F2-intent-off": {"level": "F2", "intent_off": True},
    # B0-F = base loss on the pre-registered Top-1 winner config (F2-bidir);
    # B fold0 reuses the F-bidir fold0 runs (never relaunched)
    "B": {"level": None, "bidirectional": True},
    "S2": {"level": None, "aux_arm": "S2", "bidirectional": True},
    "S10": {"level": None, "aux_arm": "S10", "bidirectional": True},
    "D2": {"level": None, "aux_arm": "D2", "bidirectional": True},
    "D10": {"level": None, "aux_arm": "D10", "bidirectional": True},
    # pre-registered Top-2 (supervisor ranking): Top-1=F2-bidir, Top-2=F3
    "V-top1": {"level": None, "bidirectional": True},
    "V-top2": {"level": None},
    "R1": {"undefined": "ramp spec R1 not defined in the frozen config"},
    "R2": {"undefined": "ramp spec R2 not defined in the frozen config"},
    "C": {"confirm": True},
    "QT": {"v52": True},
}

#: plan 4.2 sealing: B-phase folds3/4 metrics are computed+stored but never
#: enter selection / report aggregation (dev folds 0-2 only)
SEALED_FOLDS = (3, 4)


def resolve_output_root(value: Optional[str]) -> str:
    """Assert the output root is inside the campaign; return the normalized key."""
    return assert_campaign_root(value if value is not None else DEFAULT_OUTPUT_ROOT)


def _load_dataset(tables_dir: str, fold: int, split_manifest: str):
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import KeyAtomDatasetV3

    return KeyAtomDatasetV3(
        tables_dir, fold=fold, split_manifest=split_manifest,
        label_statuses=("ok",))


def _find_plan_line(plan: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    line = plan["by_run_id"].get(run_id)
    if line is None:
        raise LedgerError(
            "RUN_NOT_IN_PLAN: {} is not a planned run in the bound plan".format(run_id))
    return line


def _record_from_plan(line: Mapping[str, Any], *, attempt: int = 1,
                      parent: Optional[str] = None) -> Dict[str, Any]:
    """Ledger record whose hashed run_id EQUALS the plan line's run_id."""
    rec = {
        "campaign_id": line["campaign_id"],
        "phase": line["phase"],
        "arm": line["arm"],
        "fold": int(line["fold"]),
        "seed": int(line["seed"]),
        "variant": line.get("variant"),
        "n_updates": FORMAL_UPDATES,
        "data_sha256": line["data_sha256"],
        "teacher_sha256": line["teacher_sha256"],
        "features_sha256": line["features_sha256"],
        "split_sha256": line["split_sha256"],
        "config_sha256": line["config_sha256"],
        "checkpoint_update": None,
        "checkpoint_sha256": None,
        "status": "planned",
        "metrics_path": None,
        "gate_id": "G2-Move",
        "admissibility": "not_admissible_until_gate_pass",
        "evidence_sha": None,
        "attempt": int(attempt),
        "parent_run_id": parent,
        "confirm": bool(line.get("confirm", False)),
    }
    rid = compute_run_id(rec, "auto")
    if rid != line["run_id"]:
        raise LedgerError(
            "PLAN_RECORD_MISMATCH: record for plan line {} recomputes to {} "
            "(identity fields drifted)".format(line["run_id"], rid))
    return rec


def _trainer_config_from_line(line: Mapping[str, Any], args: argparse.Namespace):
    from pes2ts_core.pesn.offline.training.key_atom_bc_v3 import (
        TrainerConfigV3,
        TrainerError,
    )

    arm = str(line["arm"])
    spec = ARM_MAP.get(arm)
    if spec is None:
        raise TrainerError("UNKNOWN_ARM", "no trainer mapping for arm {!r}".format(arm))
    if spec.get("undefined"):
        raise TrainerError("ARM_SPEC_UNDEFINED", spec["undefined"])
    if spec.get("confirm") or spec.get("v52"):
        raise TrainerError(
            "CONFIRM_ARM_REFUSED",
            "arm {!r} is a confirm/V5.2 arm; R3 dev queue never launches it".format(arm))
    level = args.level or spec.get("level")
    if level is None:
        raise TrainerError(
            "LEVEL_REQUIRED",
            "arm {!r} needs an explicit --level (G2-selected F winner; not "
            "decidable in this lane)".format(arm))
    if spec.get("aux_arm") and args.lambda_arm is None:
        raise TrainerError(
            "AUX_LAMBDA_REQUIRED",
            "aux arm {!r} requires --lambda-arm from the calibrated probe "
            "(outputs/pesn/30h_campaign/r3_runs/lambda_calibration.json); "
            "guessing λ is forbidden".format(arm))
    return TrainerConfigV3(
        arm=arm,
        fold=int(line["fold"]),
        seed=int(line["seed"]),
        level=level,
        intent_off=bool(spec.get("intent_off", False)),
        bidirectional=bool(spec.get("bidirectional", False)),
        aux_arm=spec.get("aux_arm"),
        lambda_arm=float(args.lambda_arm or 0.0),
        device=args.device,
        smoke_updates=None,          # formal runs: frozen 3000 updates
        updates=FORMAL_UPDATES,
    )


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------


def cmd_smoke(args: argparse.Namespace) -> int:
    """Throughput probe: NO ledger; GPU lock held with metadata."""
    root = resolve_output_root(args.output_root)
    handle = gpu_acquire(
        GPU_LOCK, lane="W-LOSS-E3",
        run_id="throughput-smoke-{}".format(args.smoke_updates), cuda=True)
    if handle is None:
        sys.stderr.write("error: GPU lock busy; refusing to smoke\n")
        return 1
    try:
        from pes2ts_core.pesn.offline.training.key_atom_bc_v3 import (
            TrainerConfigV3,
            train_run,
        )

        ds = _load_dataset(args.data_dir, args.fold, args.split)
        cfg = TrainerConfigV3(
            arm="smoke", fold=args.fold, seed=args.seed, level=args.level,
            updates=args.smoke_updates, val_every=max(args.smoke_updates // 3, 1),
            device=args.device, smoke_updates=args.smoke_updates)
        out_dir = Path(root) / "_smoke" / "fold{}_{}".format(
            args.fold, args.seed)
        t0 = time.time()
        metrics = train_run(ds, cfg, out_dir, resume=False)
        ups = args.smoke_updates / max(metrics["wall_clock_s"], 1e-9)
        print(json.dumps({
            "kind": "throughput_smoke",
            "updates": args.smoke_updates,
            "train_wall_s": metrics["wall_clock_s"],
            "total_wall_s": round(time.time() - t0, 2),
            "updates_per_s": round(ups, 2),
            "t_run_3000_est_s": round(FORMAL_UPDATES / ups, 1),
            "t_run_3000_est_min": round(FORMAL_UPDATES / ups / 60.0, 2),
            "last_train_loss": metrics["last_train_loss"],
            "out_dir": str(out_dir),
        }, sort_keys=True))
        return 0
    finally:
        handle.release()


def cmd_run(args: argparse.Namespace) -> int:
    root = resolve_output_root(args.output_root)
    plan = read_run_plan(args.plan)
    assert_stage_binding(args.plan, stage="g0")   # whole-plan five-sha gate
    line = _find_plan_line(plan, args.run_id)
    if line.get("confirm"):
        sys.stderr.write(
            "error: CONFIRM_ARM_REFUSED: {} is a confirm arm; use "
            "block-confirm\n".format(args.run_id))
        return 1
    ledger = RunLedger(
        args.ledger, plan_path=args.plan,  # campaign pin default
        metric_unseal_path=args.metric_unseal,
    )
    ledger.reconcile()  # dead-pid running entries -> interrupted (retryable)
    entry = ledger.status(args.run_id)
    if entry is None or entry.get("status") == "planned":
        attempt, parent = 1, None
    else:  # a prior launch happened: this is a retry (ledger retry_count is
        # authoritative; the attempt field mirrors it honestly)
        attempt = 2 + int(entry.get("retry_count", 0) or 0)
        parent = args.run_id
    record = _record_from_plan(line, attempt=attempt, parent=parent)
    if entry is None:
        ledger.append(record)   # register (planned) exactly once
    elif entry.get("status") == "done":
        print(json.dumps({"run_id": args.run_id, "action": "already_done",
                          "checkpoint_sha256": entry.get("checkpoint_sha256")},
                         sort_keys=True))
        return 0

    # GPU lock BEFORE the running transition (never running without the mutex)
    handle = gpu_acquire(GPU_LOCK, lane="W-LOSS-E3", run_id=args.run_id, cuda=True)
    if handle is None:
        sys.stderr.write("error: GPU lock busy; run stays planned\n")
        return 1
    try:
        report = attempt_launch(ledger, record, stage="g0", task_state=args.task_state)
        if report["state"] != "running":
            print(json.dumps(report, sort_keys=True))
            return 3

        config = _trainer_config_from_line(line, args)
        out_dir = Path(root) / args.run_id
        try:
            from pes2ts_core.pesn.offline.training.key_atom_bc_v3 import train_run

            ds = _load_dataset(args.data_dir, int(line["fold"]), args.split)
            metrics = train_run(ds, config, out_dir, resume=True)
        except BaseException as exc:  # diagnose -> record failed; retry = new attempt
            fail = dict(record)
            fail.update({
                "status": "failed",
                "metrics_path": str(out_dir / "metrics.json")
                if (out_dir / "metrics.json").is_file() else None,
            })
            try:
                ledger.append(fail)
            except LedgerError as ledger_exc:
                sys.stderr.write(
                    "note: failed-record append refused: {}\n".format(ledger_exc))
            sys.stderr.write("error: run {} failed: {}\n".format(args.run_id, exc))
            return 1
    finally:
        handle.release()

    done = dict(record)
    done.update({
        "status": "done",
        "checkpoint_update": int(metrics["selected_update"]),
        "checkpoint_sha256": metrics["checkpoint_best_sha256"],
        "metrics_path": "{}/{}/metrics.json".format(root, args.run_id),
    })
    if int(line["fold"]) in SEALED_FOLDS:
        # plan 4.2: folds3/4 metrics are SEALED -- stored, never aggregated
        done["metrics_seal"] = (
            "sealed: B-phase folds3/4 metrics excluded from selection/"
            "report aggregation (dev folds 0-2 only; plan 4.2)")
    outcome = ledger.append(done)
    print(json.dumps({
        "run_id": args.run_id,
        "action": outcome.action,
        "status": outcome.entry["status"],
        "arm": line["arm"], "fold": line["fold"], "seed": line["seed"],
        "wall_clock_s": metrics["wall_clock_s"],
        "last_train_loss": metrics["last_train_loss"],
        "best_inner_val_loss": metrics["best_inner_val_loss"],
        "selected_update": metrics["selected_update"],
        "key_rmse_over_zero": metrics["key_rmse_over_zero"].get(
            "median_rmse_over_zero"),
        "sign_median": metrics["sre_and_sign"].get(
            "sign_accuracy_cluster_median"),
        "sre_median": metrics["sre_and_sign"].get("sre_cluster_median"),
        "guards": metrics["guards"],
        "checkpoint_sha256": metrics["checkpoint_best_sha256"],
        "metrics_path": done["metrics_path"],
    }, sort_keys=True))
    return 0


def cmd_block_confirm(args: argparse.Namespace) -> int:
    """Register + block every confirm arm (never launched this campaign)."""
    plan = read_run_plan(args.plan)
    ledger = RunLedger(args.ledger, plan_path=args.plan)
    results = []
    for run_id in sorted(plan["by_run_id"]):
        line = plan["by_run_id"][run_id]
        if not (line.get("confirm") and line.get("substage") in
                ("C", "QT-confirm")):
            continue
        record = _record_from_plan(line)
        ledger.append(record)  # registration only
        outcome = ledger.mark_blocked(run_id, CONFIRM_BLOCK_REASON)
        results.append({"run_id": run_id, "action": outcome.action,
                        "substage": line["substage"], "fold": line["fold"]})
    print(json.dumps({"blocked_confirm_arms": results,
                      "count": len(results)}, sort_keys=True))
    return 0


def cmd_block(args: argparse.Namespace) -> int:
    """Register + block named runs with an honest reason (never launched).

    Used for: R arms (R1/R2 ramp spec undefined in any frozen doc — no
    invented hyperparameters) and L arms whose probe verdict is
    ``scale_unreachable`` (evidence recorded; run not burned).
    """
    plan = read_run_plan(args.plan)
    ledger = RunLedger(args.ledger, plan_path=args.plan)
    results = []
    for run_id in args.run_ids:
        line = _find_plan_line(plan, run_id)
        if line.get("confirm"):
            sys.stderr.write(
                "error: {} is a confirm arm; use block-confirm\n".format(run_id))
            return 2
        record = _record_from_plan(line)
        ledger.append(record)  # registration only
        outcome = ledger.mark_blocked(run_id, args.reason)
        results.append({"run_id": run_id, "action": outcome.action,
                        "arm": line["arm"], "fold": line["fold"],
                        "seed": line["seed"], "reason": args.reason})
    print(json.dumps({"blocked_runs": results, "count": len(results)},
                     sort_keys=True))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_v5_1_r3",
        description="30h R3 driver: campaign-root assert + supervised launch "
        "+ frozen-contract training loop",
    )
    parser.add_argument("--output-root", default=None,
                        help="output root (default: {}; must stay inside the "
                        "campaign)".format(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--check", action="store_true",
                        help="historical flag: assert the output root and exit")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("check", help="assert the output root and print it")

    smoke = sub.add_parser("smoke", help="throughput probe (no ledger)")
    smoke.add_argument("--smoke-updates", type=int, default=300)
    smoke.add_argument("--fold", type=int, default=0)
    smoke.add_argument("--seed", type=int, default=17)
    smoke.add_argument("--level", default="F0")
    smoke.add_argument("--device", default="cuda")
    smoke.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    smoke.add_argument("--split", default=DEFAULT_SPLIT)

    run = sub.add_parser("run", help="formal planned run (ledger-bound)")
    run.add_argument("--run-id", required=True)
    run.add_argument("--plan", default=DEFAULT_PLAN)
    run.add_argument("--ledger", default=DEFAULT_LEDGER)
    run.add_argument("--task-state", default=DEFAULT_TASK_STATE_PATH)
    run.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    run.add_argument("--split", default=DEFAULT_SPLIT)
    run.add_argument(
        "--metric-unseal", default=DEFAULT_METRIC_UNSEAL,
        help="one-shot B folds3/4 metric-reveal token (plan 4.2; consumed "
        "idempotently at the first sealed-fold done transition; the token "
        "receipt is recorded on the run record and the metrics stay EXCLUDED "
        "from all selection/aggregation)")
    run.add_argument("--device", default="cuda")
    run.add_argument("--level", default=None,
                     help="feature level override (B/L/V arms after G2)")
    run.add_argument("--lambda-arm", default=None,
                     help="calibrated aux λ (S/D arms; required for them)")

    block = sub.add_parser("block-confirm",
                           help="mark all confirm arms blocked (never run)")
    block.add_argument("--plan", default=DEFAULT_PLAN)
    block.add_argument("--ledger", default=DEFAULT_LEDGER)

    blk = sub.add_parser("block",
                         help="register + block named runs with a reason")
    blk.add_argument("--plan", default=DEFAULT_PLAN)
    blk.add_argument("--ledger", default=DEFAULT_LEDGER)
    blk.add_argument("--reason", required=True,
                     help="blocked_with_reason (honest, evidence-backed)")
    blk.add_argument("run_ids", nargs="+", help="planned run_ids to block")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = _build_parser().parse_args(argv)
    try:
        if args.cmd is None:
            if args.check:
                root = resolve_output_root(args.output_root)
                print(json.dumps({"ok": True, "output_root": root}, sort_keys=True))
                return 0
            _build_parser().print_help()
            return 2
        if args.cmd == "check":
            root = resolve_output_root(args.output_root)
            print(json.dumps({"ok": True, "output_root": root}, sort_keys=True))
            return 0
        if args.cmd == "smoke":
            return cmd_smoke(args)
        if args.cmd == "run":
            return cmd_run(args)
        if args.cmd == "block-confirm":
            return cmd_block_confirm(args)
        if args.cmd == "block":
            return cmd_block(args)
        return 2
    except CampaignRootError as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 1
    except (LedgerError, OSError) as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
