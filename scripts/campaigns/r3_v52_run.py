"""V5.2 run driver (campaign plan §5.5; W-LOSS E5).

One planned V5.2 run per invocation:

1. campaign-root assertion (sealed prefixes refuse fail-closed);
2. plan lookup (exact run_id; Q/T/QT dev arms, folds0-2);
3. register (planned) + ``monitor.attempt_launch(stage="g0")`` (plan pin +
   five-sha binding + budget caps — NEVER a raw ledger launch);
4. GPU flock with metadata;
5. train via ``train_run_v52`` (frozen schedule; λq/λm from the calibrated
   probe output — the driver REFUSES to run without them);
6. ``done`` transition with ``checkpoint_sha256`` + ``metrics_path`` under
   ``outputs/pesn/30h_campaign/v52_runs/<run_id>/``.

Backbone init is the matched R3 winner checkpoint (fold+seed; the F2-bidir
winner config).  QT-confirm rows are never launched here (blocked;
``block-confirm-update`` re-annotates them with the post-G2 reason).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agent_ops.gpu_queue.gpu_lock import acquire as gpu_acquire  # noqa: E402
from scripts.agent_ops.ledger.run_ledger import (  # noqa: E402
    LedgerError,
    RunLedger,
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
from scripts.campaigns.run_v5_1_r3 import (  # noqa: E402
    DEFAULT_METRIC_UNSEAL,
    DEFAULT_PLAN,
    _record_from_plan,
)

DEFAULT_V52_ROOT = CAMPAIGN_ROOT + "/v52_runs"
DEFAULT_LEDGER = CAMPAIGN_ROOT + "/run_records.jsonl"
DEFAULT_DATA_DIR = CAMPAIGN_ROOT + "/data"
DEFAULT_SPLIT = CAMPAIGN_ROOT + "/data/split_manifest_r3.json"
DEFAULT_CALIBRATION = DEFAULT_V52_ROOT + "/v52_lambda_calibration.json"
GPU_LOCK = "/tmp/opencode/pes2ts_gpu.lock"
V52_UPDATES = 1900
CONFIRM_REASON = "confirm_awaiting_user_authorization"

#: matched R3 winner init (F2-bidir config) by (fold, seed) — E3 wave-1/2 results
R3_WINNER_INIT = {
    (0, 17): "5372d9b12807c3f0", (0, 29): "1e01499b933d4ad1",
    (0, 43): "9e2ef3ef468db0af",
    (1, 17): "cc9340b86592b099", (1, 29): "0629b5979df6a2df",
    (1, 43): "c09e9cd601bafb88",
    (2, 17): "0ebd4c92c85f5601", (2, 29): "dedd62d588f3b6cd",
    (2, 43): "b3941e53f44b962b",
}


def cmd_run(args: argparse.Namespace) -> int:
    root = assert_campaign_root(args.output_root)
    plan = read_run_plan(args.plan)
    assert_stage_binding(args.plan, stage="g0")
    line = plan["by_run_id"].get(args.run_id)
    if line is None:
        sys.stderr.write("error: RUN_NOT_IN_PLAN: {}\n".format(args.run_id))
        return 2
    if line.get("confirm") or line.get("variant") == "confirm":
        sys.stderr.write(
            "error: CONFIRM_ARM_REFUSED: {} is a QT-confirm row; use "
            "block-confirm-update\n".format(args.run_id))
        return 1
    arm = str(line["arm"])
    if arm not in ("Q", "T", "QT"):
        sys.stderr.write(
            "error: NOT_A_V52_ARM: {} is arm {!r}\n".format(args.run_id, arm))
        return 2
    fold, seed = int(line["fold"]), int(line["seed"])
    cal = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    term = {"Q": "q", "T": "stop", "QT": None}[arm]
    if term is None:
        lambda_q = float(cal["terms"]["q"]["lambda"])
        lambda_m = float(cal["terms"]["stop"]["lambda"])
        for t in ("q", "stop"):
            if cal["terms"][t]["verdict"] == "scale_unreachable":
                sys.stderr.write(
                    "error: SCALE_UNREACHABLE: {} (probe verdict; arm must be "
                    "blocked, not run)\n".format(t))
                return 1
    else:
        verdict = cal["terms"][term]["verdict"]
        if verdict == "scale_unreachable":
            sys.stderr.write(
                "error: SCALE_UNREACHABLE: {} (probe verdict)\n".format(term))
            return 1
        lambda_q = float(cal["terms"]["q"]["lambda"]) if term == "q" else 0.0
        lambda_m = float(cal["terms"]["stop"]["lambda"]) if term == "stop" else 0.0
    init_run = R3_WINNER_INIT[(fold, seed)]
    r3_ckpt = Path(args.r3_runs) / init_run / "checkpoint_best.pt"
    if not r3_ckpt.is_file():
        sys.stderr.write("error: R3_INIT_MISSING: {}\n".format(r3_ckpt))
        return 2

    ledger = RunLedger(
        args.ledger, plan_path=args.plan,
        metric_unseal_path=args.metric_unseal,
    )
    ledger.reconcile()
    entry = ledger.status(args.run_id)
    if entry is not None and entry.get("status") == "done":
        print(json.dumps({"run_id": args.run_id, "action": "already_done"},
                         sort_keys=True))
        return 0
    attempt = 1 if (entry is None or entry.get("status") == "planned") else \
        2 + int(entry.get("retry_count", 0) or 0)
    record = _record_from_plan(line, attempt=attempt,
                               parent=args.run_id if attempt > 1 else None)
    record["n_updates"] = V52_UPDATES   # frozen v52 schedule (not in run_id)
    if entry is None:
        ledger.append(record)

    handle = gpu_acquire(GPU_LOCK, lane="W-LOSS-E5", run_id=args.run_id,
                         cuda=True)
    if handle is None:
        sys.stderr.write("error: GPU lock busy; run stays planned\n")
        return 1
    try:
        report = attempt_launch(ledger, record, stage="g0",
                                task_state=args.task_state)
        if report["state"] != "running":
            print(json.dumps(report, sort_keys=True))
            return 3
        out_dir = Path(root) / args.run_id
        try:
            from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (
                KeyAtomDatasetV3,
            )
            from pes2ts_core.pesn.offline.training.key_atom_v52_train_v3 import (
                V52TrainerConfig,
                train_run_v52,
            )

            dataset = KeyAtomDatasetV3(
                args.data_dir, split_manifest=args.split,
                label_statuses=("ok",))
            labels = dataset.v52_labels()
            config = V52TrainerConfig(
                arm=arm, fold=fold, seed=seed,
                lambda_q=lambda_q, lambda_m=lambda_m,
                backbone_run_id=init_run, device=args.device)
            metrics = train_run_v52(
                dataset, labels, config, out_dir,
                r3_ckpt=r3_ckpt, r3_eval_ckpt=r3_ckpt, resume=True)
        except BaseException as exc:
            fail = dict(record)
            fail["status"] = "failed"
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
    outcome = ledger.append(done)
    print(json.dumps({
        "run_id": args.run_id, "action": outcome.action, "arm": arm,
        "fold": fold, "seed": seed,
        "backbone_run_id": init_run,
        "lambda_q": lambda_q, "lambda_m": lambda_m,
        "wall_clock_s": metrics["wall_clock_s"],
        "selected_update": metrics["selected_update"],
        "q_mae_curve_median": metrics["q_head"]["mae_curve_median"],
        "q_corr_curve_median": metrics["q_head"]["corr_curve_median"],
        "q_const_baseline_mae": metrics["q_head"]["constant_q_baseline_mae"],
        "q_learnable": metrics["q_head"]["learnable_above_constant"],
        "stop_auc": metrics["stop_head"]["auc"],
        "stop_precision": metrics["stop_head"]["precision_at_0.5"],
        "stop_recall": metrics["stop_head"]["recall_at_0.5"],
        "stop_coverage": metrics["stop_head"]["n_eval_unmasked"],
        "move_rmse_degradation_pct": metrics["move_guard_vs_r3_winner"].get(
            "rmse_degradation_pct"),
        "move_within_guard": metrics["move_guard_vs_r3_winner"].get(
            "within_rmse_guard"),
        "checkpoint_sha256": metrics["checkpoint_best_sha256"],
        "metrics_path": done["metrics_path"],
    }, sort_keys=True))
    return 0


def cmd_block_confirm_update(args: argparse.Namespace) -> int:
    """Re-annotate the 6 QT-confirm rows: dev-positive now shown (G2 PASS);
    the remaining gate is the user's confirm_authorized token."""
    plan = read_run_plan(args.plan)
    ledger = RunLedger(args.ledger, plan_path=args.plan)
    results = []
    for run_id in sorted(plan["by_run_id"]):
        line = plan["by_run_id"][run_id]
        if not (line.get("confirm") and line.get("variant") == "confirm"):
            continue
        outcome = ledger.mark_blocked(run_id, CONFIRM_REASON)
        results.append({"run_id": run_id, "action": outcome.action,
                        "fold": line["fold"], "seed": line["seed"]})
    print(json.dumps({"updated": results, "count": len(results),
                      "reason": CONFIRM_REASON}, sort_keys=True))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="r3_v52_run", description="V5.2 Q/T/QT supervised run driver")
    parser.add_argument("--output-root", default=DEFAULT_V52_ROOT)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="one planned V5.2 run")
    run.add_argument("--run-id", required=True)
    run.add_argument("--plan", default=DEFAULT_PLAN)
    run.add_argument("--ledger", default=DEFAULT_LEDGER)
    run.add_argument("--task-state", default=DEFAULT_TASK_STATE_PATH)
    run.add_argument("--metric-unseal", default=DEFAULT_METRIC_UNSEAL)
    run.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    run.add_argument("--split", default=DEFAULT_SPLIT)
    run.add_argument("--r3-runs", default=CAMPAIGN_ROOT + "/r3_runs")
    run.add_argument("--calibration", default=DEFAULT_CALIBRATION)
    run.add_argument("--device", default="cuda")
    blk = sub.add_parser("block-confirm-update",
                         help="re-annotate QT-confirm rows (post-G2 reason)")
    blk.add_argument("--plan", default=DEFAULT_PLAN)
    blk.add_argument("--ledger", default=DEFAULT_LEDGER)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.cmd == "run":
            return cmd_run(args)
        if args.cmd == "block-confirm-update":
            return cmd_block_confirm_update(args)
        return 2
    except CampaignRootError as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 1
    except (LedgerError, OSError) as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
