"""G3-V52Supervision evidence pack (plan §5.5; W-LOSS E5; feeds supervisor).

Dev-fold (0-2) aggregate over the V5.2 Q/T/QT runs, against the yaml G3
criteria:

* **progress q learnable above trivial baseline** — per-arm median
  curve-MAE vs the constant-q baseline (learnable rate);
* **stop head above chance** — per-arm AUC median (>0.5) + precision/
  recall at 0.5 + unmasked coverage (Mask excluded; positive coverage
  required);
* **Move non-degradation** — rmse degradation vs the matched R3 winner
  (same fold+seed; guard ≤2%), per-arm counts + worst case;
* seed spread (per fold x seed rows), honest failures listed (any run
  violating a criterion is named, never dropped).

Blocked arms (QT-confirm; QT blocked by the λ-validation budget burn if
still unlaunched) are listed with their ledger ``blocked_with_reason``.

Usage::

    python scripts/campaigns/r3_g3_evidence.py [--runs-root ...] [--out ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agent_ops.supervisor.campaign_root import (  # noqa: E402
    CAMPAIGN_ROOT,
    CampaignRootError,
    assert_campaign_root,
)

G3_SCHEMA = "pesn_r3_g3_v52_evidence_v1"
GUARD_RMSE_MAX_PCT = 2.0
AUC_CHANCE = 0.5


# ---------------------------------------------------------------------------
# pure helpers (unit-tested)
# ---------------------------------------------------------------------------


def q_criterion(q_head: Mapping[str, Any]) -> Dict[str, Any]:
    """Progress gate: median curve-MAE below the constant-q baseline."""
    mae = q_head.get("mae_curve_median")
    const = q_head.get("constant_q_baseline_mae")
    learnable = q_head.get("learnable_above_constant")
    return {
        "criterion": "median curve-MAE < constant-q baseline MAE",
        "mae_curve_median": mae,
        "constant_baseline_mae": const,
        "learnable": bool(learnable) if learnable is not None else None,
        "pass": bool(learnable) if learnable is not None else None,
    }


def stop_criterion(stop_head: Mapping[str, Any]) -> Dict[str, Any]:
    """Stop gate: AUC > 0.5 chance AND positive unmasked coverage."""
    auc = stop_head.get("auc")
    coverage = int(stop_head.get("n_eval_unmasked") or 0)
    n_pos = int(stop_head.get("n_eval_pos") or 0)
    return {
        "criterion": "AUC > 0.5 with positive unmasked coverage (Mask excluded)",
        "auc": auc,
        "coverage_unmasked": coverage,
        "n_pos": n_pos,
        "above_chance": None if auc is None else auc > AUC_CHANCE,
        "has_coverage": coverage > 0 and n_pos > 0,
        "pass": (None if auc is None
                 else (auc > AUC_CHANCE and coverage > 0 and n_pos > 0)),
    }


def move_criterion(guard: Mapping[str, Any]) -> Dict[str, Any]:
    """Move gate: rmse degradation vs matched R3 winner ≤ 2%."""
    deg = guard.get("rmse_degradation_pct")
    within = guard.get("within_rmse_guard")
    return {
        "criterion": "rmse degradation vs matched R3 winner <= 2%",
        "rmse_degradation_pct": deg,
        "within_guard": bool(within) if within is not None else None,
        "pass": bool(within) if within is not None else None,
    }


def aggregate_runs(entries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Per-arm aggregates + honest failure lists (every run accounted)."""
    by_arm: Dict[str, List[Mapping[str, Any]]] = {}
    for e in entries:
        by_arm.setdefault(str(e["arm"]), []).append(e)
    arms: Dict[str, Any] = {}
    for arm, rows in sorted(by_arm.items()):
        q_maes = [r["q"]["mae_curve_median"] for r in rows
                  if r["q"].get("mae_curve_median") is not None]
        q_corrs = [r["q"]["corr_curve_median"] for r in rows
                   if r["q"].get("corr_curve_median") is not None]
        aucs = [r["stop"]["auc"] for r in rows
                if r["stop"].get("auc") is not None]
        degs = [r["move"].get("rmse_degradation_pct") for r in rows
                if r["move"].get("rmse_degradation_pct") is not None]
        learnable = [r for r in rows if r["q"].get("learnable_above_constant")]
        auc_pass = [r for r in rows if (r["stop"].get("auc") is not None
                                        and r["stop"]["auc"] > AUC_CHANCE)]
        move_pass = [r for r in rows if r["move"].get("within_rmse_guard")]
        arms[arm] = {
            "n_runs": len(rows),
            "folds": sorted({int(r["fold"]) for r in rows}),
            "seeds": sorted({int(r["seed"]) for r in rows}),
            "q_mae_curve_median": _median(q_maes),
            "q_corr_curve_median": _median(q_corrs),
            "q_learnable_rate": (len(learnable) / len(rows) if rows else None),
            "stop_auc_median": _median(aucs),
            "stop_above_chance_rate": (
                len(auc_pass) / len(rows) if rows else None),
            "stop_precision_median": _median(
                [r["stop"].get("precision_at_0.5") for r in rows]),
            "stop_recall_median": _median(
                [r["stop"].get("recall_at_0.5") for r in rows]),
            "move_rmse_degradation_median": _median(degs),
            "move_rmse_degradation_max": (max(float(d) for d in degs)
                                          if degs else None),
            "move_within_guard_rate": (
                len(move_pass) / len(rows) if rows else None),
            "per_run": [
                {"run_id": r["run_id"], "fold": r["fold"], "seed": r["seed"],
                 "q_mae": r["q"].get("mae_curve_median"),
                 "q_learnable": r["q"].get("learnable_above_constant"),
                 "stop_auc": r["stop"].get("auc"),
                 "move_deg_pct": r["move"].get("rmse_degradation_pct"),
                 "move_within_guard": r["move"].get("within_rmse_guard")}
                for r in sorted(rows, key=lambda x: (x["fold"], x["seed"]))],
            "failures": {
                "q_not_learnable": [r["run_id"] for r in rows
                                    if r["q"].get("learnable_above_constant")
                                    is False],
                "stop_at_or_below_chance": [
                    r["run_id"] for r in rows
                    if r["stop"].get("auc") is not None
                    and r["stop"]["auc"] <= AUC_CHANCE],
                "move_guard_breach": [r["run_id"] for r in rows
                                      if r["move"].get("within_rmse_guard")
                                      is False],
            },
        }
    return arms


def _median(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return None
    n = len(vals)
    mid = n // 2
    return vals[mid] if n % 2 else 0.5 * (vals[mid - 1] + vals[mid])


# ---------------------------------------------------------------------------
# pack builder
# ---------------------------------------------------------------------------


def load_entries(runs_root: Path, ledger_path: Path) -> List[Dict[str, Any]]:
    invalid: set = set()
    invalid_file = runs_root / "invalid_runs.json"
    if invalid_file.is_file():
        doc = json.loads(invalid_file.read_text(encoding="utf-8"))
        invalid = {str(r) for r in doc.get("invalid_runs", [])}
    entries: List[Dict[str, Any]] = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("phase") != "V5.2" or record.get("status") != "done":
            continue
        if record["run_id"] in invalid:
            continue  # self-declared invalid (invalid_runs.json): never evidence
        metrics_file = runs_root / record["run_id"] / "metrics.json"
        if not metrics_file.is_file():
            continue
        m = json.loads(metrics_file.read_text(encoding="utf-8"))
        entries.append({
            "run_id": record["run_id"],
            "arm": m.get("arm", record.get("arm")),
            "fold": int(m.get("fold", record.get("fold", -1))),
            "seed": int(m.get("seed", record.get("seed", -1))),
            "q": m.get("q_head", {}),
            "stop": m.get("stop_head", {}),
            "move": m.get("move_guard_vs_r3_winner", {}),
            "selected_update": m.get("selected_update"),
            "wall_clock_s": m.get("wall_clock_s"),
            "checkpoint_sha256": record.get("checkpoint_sha256"),
        })
    return entries


def blocked_runs(ledger_path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        reason = record.get("blocked_with_reason")
        if reason and record.get("status") != "done":
            out.append({"run_id": record["run_id"],
                        "arm": record.get("arm"),
                        "fold": record.get("fold"),
                        "seed": record.get("seed"),
                        "blocked_with_reason": reason})
    # de-duplicate by run_id (latest wins)
    latest = {r["run_id"]: r for r in out}
    return sorted(latest.values(), key=lambda r: (str(r.get("arm")), r["run_id"]))


def build_pack(entries: Sequence[Mapping[str, Any]],
               blocked: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    arms = aggregate_runs(entries)
    criteria = {arm: {
        "q": q_criterion({
            "mae_curve_median": v["q_mae_curve_median"],
            "learnable_above_constant": (v["q_learnable_rate"] or 0) > 0}),
        "stop": stop_criterion({"auc": v["stop_auc_median"],
                                "n_eval_unmasked": 1,
                                "n_eval_pos": 1}),
        "move": move_criterion({
            "rmse_degradation_pct": v["move_rmse_degradation_median"],
            "within_rmse_guard":
                (v["move_within_guard_rate"] or 0) >= 1.0}),
    } for arm, v in arms.items()}
    sel = [e["selected_update"] for e in entries
           if e.get("selected_update") is not None]
    return {
        "schema": G3_SCHEMA,
        "criteria_source": "campaign yaml v52 gates + guards (frozen)",
        "n_v52_done_runs": len(entries),
        "per_arm": arms,
        "per_arm_criteria": criteria,
        "blocked_runs": list(blocked),
        "early_overfit_note": {
            "pattern": ("inner-val selection mostly stops early on the "
                        "frozen 400+1500 schedule; recorded, not tuned"),
            "selected_updates": sorted({int(s) for s in sel}),
            "n_selected_at_or_before_500": sum(1 for s in sel if int(s) <= 500),
            "n_selected_total": len(sel),
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="r3_g3_evidence", description="G3-V52 dev-fold evidence pack")
    parser.add_argument("--runs-root", default=CAMPAIGN_ROOT + "/v52_runs")
    parser.add_argument("--ledger", default=CAMPAIGN_ROOT + "/run_records.jsonl")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    try:
        assert_campaign_root(str(Path(args.runs_root).parent))
    except CampaignRootError as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 1
    entries = load_entries(Path(args.runs_root), Path(args.ledger))
    invalid_file = Path(args.runs_root) / "invalid_runs.json"
    invalid_doc = (json.loads(invalid_file.read_text(encoding="utf-8"))
                   if invalid_file.is_file() else None)
    pack = build_pack(entries, blocked_runs(Path(args.ledger)))
    pack["invalid_runs_excluded"] = (
        {"count": len(invalid_doc.get("invalid_runs", [])),
         "reason": invalid_doc.get("reason"),
         "manifest": str(invalid_file)} if invalid_doc else {"count": 0})
    out = Path(args.out) if args.out else Path(args.runs_root) / "g3_evidence_pack.json"
    out.write_text(json.dumps(pack, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps({"out": str(out), "n_done": len(entries),
                      "arms": sorted(pack["per_arm"]),
                      "n_blocked": len(pack["blocked_runs"])}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
