"""R3 dev-fold (0-2) G2 evidence pack (plan §4.2; feeds supervisor G2 binding).

Aggregates the campaign run metrics for the G2-Move evidence pack:

* **SEAL ENFORCEMENT (plan §4.2)**: B-phase folds3/4 metrics are SEALED --
  they may exist on disk but MUST NOT enter any selection or report
  aggregation.  :func:`dev_fold_entries` is the single filter every
  aggregation here goes through; sealed entries are listed separately
  (count + run_ids only, never values) so the seal is auditable.
* per-arm aggregate over dev folds 0-2: mean/median rmse_over_zero, sign
  median, SRE median, seed spread (folds present per arm);
* guards vs matched B0-F baseline (same fold+seed): RMSE degradation %,
  sign drop, deadzone rise (thresholds from yaml metrics_frozen.guards);
* baseline table: zero-step (rmse_over_zero = 1 by definition), classic
  CV/Jacobian (from each run's metrics baselines block), EGNN = the model;
* honest early-overfit note: selected-update distribution.

Usage::

    python scripts/campaigns/r3_g2_evidence.py [--runs-root ...] [--out ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agent_ops.supervisor.campaign_root import (  # noqa: E402
    CAMPAIGN_ROOT,
    CampaignRootError,
    assert_campaign_root,
)

EVIDENCE_SCHEMA = "pesn_r3_g2_evidence_v1"
SEALED_FOLDS = (3, 4)
DEV_FOLDS = (0, 1, 2)
GUARD_THRESHOLDS = {
    "rmse_degradation_max_pct": 2.0,
    "sign_drop_max": 0.01,
    "deadzone_rise_max": 0.01,
}
#: the pre-registered Top-1 winner config = B0-F reference for guards
B0F_ARMS = ("F2-bidir", "B", "V-top1")


# ---------------------------------------------------------------------------
# pure helpers (unit-tested)
# ---------------------------------------------------------------------------


def is_sealed_fold(fold: Any) -> bool:
    """Plan §4.2: B-phase folds3/4 metrics are sealed."""
    return int(fold) in SEALED_FOLDS


def split_entries(entries: Sequence[Mapping[str, Any]]
                  ) -> Tuple[List[Mapping[str, Any]], List[str]]:
    """Split (run, fold) entries into (dev-fold entries, sealed run_ids).

    The ONLY aggregation gate: sealed entries contribute their run_id to
    the audit list, never their metric values.
    """
    dev: List[Mapping[str, Any]] = []
    sealed_ids: List[str] = []
    for entry in entries:
        if is_sealed_fold(entry["fold"]):
            sealed_ids.append(str(entry["run_id"]))
        else:
            dev.append(entry)
    return dev, sealed_ids


def _mean(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _median(values: Sequence[float]) -> Optional[float]:
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return None
    n = len(vals)
    mid = n // 2
    return vals[mid] if n % 2 else 0.5 * (vals[mid - 1] + vals[mid])


def aggregate_by_arm(dev_entries: Sequence[Mapping[str, Any]]
                     ) -> Dict[str, Dict[str, Any]]:
    """Per-arm aggregate over DEV-FOLD entries only."""
    by_arm: Dict[str, List[Mapping[str, Any]]] = {}
    for entry in dev_entries:
        by_arm.setdefault(str(entry["arm"]), []).append(entry)
    out: Dict[str, Dict[str, Any]] = {}
    for arm, rows in sorted(by_arm.items()):
        rmse = [r["rmse_over_zero"] for r in rows]
        sign = [r["sign_median"] for r in rows]
        sre = [r["sre_median"] for r in rows]
        out[arm] = {
            "n_runs": len(rows),
            "folds": sorted({int(r["fold"]) for r in rows}),
            "seeds": sorted({int(r["seed"]) for r in rows}),
            "mean_rmse_over_zero": _mean(rmse),
            "median_rmse_over_zero": _median(rmse),
            "min_rmse_over_zero": min(float(v) for v in rmse if v is not None),
            "max_rmse_over_zero": max(float(v) for v in rmse if v is not None),
            "median_sign": _median(sign),
            "median_sre": _median(sre),
            "selected_updates": sorted({int(r["selected_update"])
                                        for r in rows
                                        if r.get("selected_update") is not None}),
        }
    return out


def guards_vs_baseline(
    dev_entries: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Guard comparisons of each run vs its matched B0-F baseline.

    Match rule: same fold + same seed, baseline arm in :data:`B0F_ARMS`
    with the F2-bidir configuration (B / V-top1 / F2-bidir).  Degradation
    uses absolute ``median_rmse`` (recorded, never gating a dev run).
    """
    baseline_index: Dict[Tuple[int, int], Mapping[str, Any]] = {}
    for entry in dev_entries:
        if str(entry["arm"]) in B0F_ARMS and entry.get("median_rmse") is not None:
            key = (int(entry["fold"]), int(entry["seed"]))
            # prefer the B arm (canonical B0-F) when several match
            if key not in baseline_index or str(entry["arm"]) == "B":
                baseline_index[key] = entry
    rows: List[Dict[str, Any]] = []
    for entry in dev_entries:
        if str(entry["arm"]) in B0F_ARMS:
            continue
        base = baseline_index.get((int(entry["fold"]), int(entry["seed"])))
        if base is None or entry.get("median_rmse") is None:
            continue
        rmse_deg = 100.0 * (float(entry["median_rmse"])
                            - float(base["median_rmse"])) / float(base["median_rmse"])
        sign_drop = ((base.get("sign_median") or 0.0)
                     - (entry.get("sign_median") or 0.0))
        dead_rise = ((entry.get("dead_zone_fraction") or 0.0)
                     - (base.get("dead_zone_fraction") or 0.0))
        rows.append({
            "run_id": entry["run_id"], "arm": entry["arm"],
            "fold": entry["fold"], "seed": entry["seed"],
            "baseline_run_id": base["run_id"], "baseline_arm": base["arm"],
            "rmse_degradation_pct": rmse_deg,
            "sign_drop": sign_drop,
            "deadzone_rise": dead_rise,
            "within_rmse_guard": rmse_deg <= GUARD_THRESHOLDS[
                "rmse_degradation_max_pct"],
            "within_sign_guard": sign_drop <= GUARD_THRESHOLDS["sign_drop_max"],
            "within_deadzone_guard": dead_rise <= GUARD_THRESHOLDS[
                "deadzone_rise_max"],
        })
    return rows


# ---------------------------------------------------------------------------
# pack builder
# ---------------------------------------------------------------------------


def load_entries(runs_root: Path, ledger_path: Path) -> List[Dict[str, Any]]:
    """Join ledger done-records with their metrics.json files."""
    entries: List[Dict[str, Any]] = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") != "done" or not record.get("metrics_path"):
            continue
        metrics_file = runs_root / record["run_id"] / "metrics.json"
        if not metrics_file.is_file():
            continue
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        g52 = metrics.get("key_rmse_over_zero", {})
        g53 = metrics.get("sre_and_sign", {})
        classic = metrics.get("baselines", {}).get(
            "classic_cv_jacobian", {}).get("sre_and_sign", {})
        entries.append({
            "run_id": record["run_id"],
            "arm": metrics.get("arm", record.get("arm")),
            "fold": int(metrics.get("fold", record.get("fold", -1))),
            "seed": int(metrics.get("seed", record.get("seed", -1))),
            "rmse_over_zero": g52.get("median_rmse_over_zero"),
            "median_rmse": g52.get("median_rmse"),
            "sign_median": g53.get("sign_accuracy_cluster_median"),
            "sre_median": g53.get("sre_cluster_median"),
            "dead_zone_fraction": g53.get("dead_zone_fraction"),
            "classic_sign_median": classic.get("sign_accuracy_cluster_median"),
            "classic_sre_median": classic.get("sre_cluster_median"),
            "selected_update": metrics.get("selected_update"),
            "last_train_loss": metrics.get("last_train_loss"),
            "checkpoint_sha256": record.get("checkpoint_sha256"),
            "sealed_annotation": record.get("metrics_seal"),
        })
    return entries


def build_evidence(entries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    dev, sealed_ids = split_entries(entries)
    per_arm = aggregate_by_arm(dev)
    guards = guards_vs_baseline(dev)
    sel = [e["selected_update"] for e in dev
           if e.get("selected_update") is not None]
    early = sum(1 for s in sel if int(s) <= 250)
    return {
        "schema": EVIDENCE_SCHEMA,
        "seal_policy": ("B-phase folds3/4 metrics SEALED (plan 4.2): stored "
                        "on disk, excluded from every aggregate below; sealed "
                        "run_ids listed for audit"),
        "dev_folds": list(DEV_FOLDS),
        "n_dev_runs": len(dev),
        "n_sealed_runs_excluded": len(sealed_ids),
        "sealed_run_ids": sorted(sealed_ids),
        "per_arm": per_arm,
        "guards_vs_matched_b0f": guards,
        "guard_thresholds": GUARD_THRESHOLDS,
        "baselines": {
            "zero_step": "rmse_over_zero = 1.0 by definition; sign median 0 "
                         "(abstention counted incorrect)",
            "classic_cv_jacobian": {
                "formula": ("min-norm first-order pair step u = Jᵀ(JJᵀ)⁺Δd*, "
                            "Δd* = one d0-scale move per CV pair"),
                "per_arm_classic_sign_median": {
                    arm: _median([e["classic_sign_median"] for e in dev
                                  if e["arm"] == arm])
                    for arm in sorted({str(e["arm"]) for e in dev})},
            },
            "egnn": "the trained model (per-arm rows above)",
        },
        "early_overfit_note": {
            "pattern": ("inner-val loss rises after ~250 updates on 7-curve "
                        "inner-train splits; checkpoint selection (inner-val "
                        "argmin) mostly stops at update 250"),
            "n_selected_at_or_before_250": early,
            "n_selected_total": len(sel),
            "policy": "frozen hyperparameters; recorded honestly, no tuning",
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="r3_g2_evidence",
        description="Dev-fold (0-2) G2 evidence pack; folds3/4 sealed")
    parser.add_argument("--runs-root", default=CAMPAIGN_ROOT + "/r3_runs")
    parser.add_argument("--ledger", default=CAMPAIGN_ROOT + "/run_records.jsonl")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    try:
        assert_campaign_root(str(Path(args.runs_root).parent))
    except CampaignRootError as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 1
    entries = load_entries(Path(args.runs_root), Path(args.ledger))
    pack = build_evidence(entries)
    out = (Path(args.out) if args.out
           else Path(args.runs_root) / "g2_evidence_pack.json")
    out.write_text(json.dumps(pack, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "n_dev_runs": pack["n_dev_runs"],
        "n_sealed_excluded": pack["n_sealed_runs_excluded"],
        "arms": sorted(pack["per_arm"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
