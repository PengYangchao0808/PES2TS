"""E7 statistics toolkit tests (lane W-DATA): ``r3_stats.py``.

All tests run on synthetic fixtures in a tmp campaign root — no dependence
on live run artifacts.  Covered:

* aggregation math (mean/std/median, pooled cluster accounting);
* bootstrap determinism (same seed => identical CI; different seed allowed
  to differ; single-value CI degenerates);
* sealed-fold discipline (folds 3/4 excluded from every aggregate + flagged);
* cost accounting arithmetic (attempts, retries, wall-clock split,
  GPU-hours, blocked list);
* missing/incremental-run tolerance (unreadable metrics, missing metrics,
  pending ledger runs, malformed ledger lines);
* end-to-end determinism of the written JSON+MD (byte-identical reruns).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

WORKTREE_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = WORKTREE_ROOT / "scripts" / "agent_ops" / "data_gate" / "r3_stats.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("r3_stats_under_test", MODULE_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


r3_stats = _load_module()


# --------------------------------------------------------------- fixtures


def _metrics(
    run_id: str,
    arm: str,
    fold: int,
    seed: int,
    rmse_over_zero: float = 1.0,
    sign: float = 0.5,
    sre: float = 0.5,
    n_clusters: int = 2,
    val_curves: list[str] | None = None,
    wall_clock_s: float = 10.0,
    device: str = "cuda",
    ci95: list[float] | None = None,
) -> dict:
    return {
        "run_id": run_id,
        "arm": arm,
        "fold": fold,
        "seed": seed,
        "level": "F2",
        "device": device,
        "wall_clock_s": wall_clock_s,
        "inner_val_curves": val_curves or [f"RXN_{fold}001", f"RXN_{fold}002"],
        "key_rmse_over_zero": {
            "median_rmse_over_zero": rmse_over_zero,
            "median_mae_over_zero": rmse_over_zero,
            "cos_eff_median": 0.2,
            "n_clusters": n_clusters,
            "n_better_than_zero": n_clusters if rmse_over_zero < 1.0 else 0,
            "n_clusters_rmse_over_zero_defined": n_clusters,
        },
        "sre_and_sign": {
            "sign_accuracy_cluster_median": sign,
            "sre_cluster_median": sre,
            "dead_zone_fraction": 0.0,
        },
        "key_rmse_over_zero_bootstrap": {
            "median_rmse_over_zero_ci95": ci95 or [rmse_over_zero - 0.05, rmse_over_zero + 0.05],
            "unit": "reaction_cluster",
            "draws": 200,
            "seed": 20261018,
        },
        "guards": {"rmse_degradation_vs_zero_step_pct": 5.0},
        "_metrics_sha256": "0" * 64,
    }


def _write_campaign(tmp: Path, runs: list[dict], ledger_lines: list[dict] | None = None) -> Path:
    root = tmp / "campaign"
    root.mkdir(parents=True, exist_ok=True)
    for r in runs:
        d = root / "r3_runs" / r["run_id"]
        d.mkdir(parents=True, exist_ok=True)
        payload = {k: v for k, v in r.items() if k != "run_id"}
        (d / "metrics.json").write_text(json.dumps(payload))
    if ledger_lines is not None:
        with open(root / "run_records.jsonl", "w") as f:
            for line in ledger_lines:
                f.write(json.dumps(line) + "\n")
    return root


def _ledger_line(run_id, arm, phase="R3", fold=0, seed=17, status="done",
                 attempt=1, retry_count=0, blocked=None, created="2026-10-20T00:00:00+00:00",
                 updated="2026-10-20T00:10:00+00:00"):
    rec = {
        "run_id": run_id,
        "arm": arm,
        "phase": phase,
        "fold": fold,
        "seed": seed,
        "status": status,
        "attempt": attempt,
        "retry_count": retry_count,
        "created_at": created,
        "updated_at": updated,
    }
    if blocked:
        rec["blocked_with_reason"] = blocked
        rec["blocked_at"] = updated
        rec["blocked_source"] = "test"
    return rec


# ------------------------------------------------------ aggregation math


def test_aggregation_mean_spread_and_cluster_accounting():
    runs = [
        _metrics("r1", "A", 0, 17, rmse_over_zero=0.8, n_clusters=2),
        _metrics("r2", "A", 0, 29, rmse_over_zero=1.0, n_clusters=2),
        _metrics("r3", "A", 1, 43, rmse_over_zero=1.2, n_clusters=3),
    ]
    import random

    rng = random.Random(1)
    arms = r3_stats.aggregate_arms(runs, rng, draws=200)
    a = arms["A"]
    assert a["n_runs"] == 3
    assert a["cluster_evaluations_total"] == 7  # 2+2+3
    m = a["metrics"]["rmse_over_zero"]
    assert m["n"] == 3
    assert m["mean"] == pytest.approx(1.0)
    assert m["median"] == pytest.approx(1.0)
    assert m["min"] == pytest.approx(0.8) and m["max"] == pytest.approx(1.2)
    assert m["std"] == pytest.approx(0.2, abs=1e-9)  # sample std of {0.8,1.0,1.2}
    frac = a["metrics"]["fraction_better_than_zero"]
    assert frac == {"n_better": 2, "n_defined": 7, "fraction": pytest.approx(2 / 7)}
    # unique val curves pooled across runs
    assert a["n_unique_val_curves"] == 4  # RXN_0001, RXN_0002 (f0) + RXN_1001, RXN_1002 (f1)
    # per-run cluster CI reused
    assert a["per_run_cluster_bootstrap_ci95"]["r1"]["unit"] == "reaction_cluster"
    assert a["per_run_cluster_bootstrap_ci95"]["r1"]["ci95"] == pytest.approx([0.75, 0.85])


def test_seed_stability_accounting():
    runs = [
        _metrics("r1", "A", 0, 17, rmse_over_zero=0.9),
        _metrics("r2", "A", 0, 29, rmse_over_zero=0.95),
        _metrics("r3", "A", 0, 43, rmse_over_zero=1.1),
        _metrics("r4", "A", 1, 17, rmse_over_zero=1.1),
        _metrics("r5", "A", 1, 29, rmse_over_zero=1.2),
        _metrics("r6", "A", 1, 43, rmse_over_zero=0.5),
    ]
    import random

    arms = r3_stats.aggregate_arms(runs, random.Random(1), draws=100)
    st = arms["A"]["seed_stability"]
    assert st["n_folds_with_2of3_seeds_below_1"] == 1  # only fold 0 has 2/3 below 1.0
    assert st["rmse_over_zero_by_fold_seed"]["0"]["17"] == pytest.approx(0.9)


# ------------------------------------------------- bootstrap determinism


def test_bootstrap_determinism_same_seed():
    import random

    values = [0.8, 0.9, 1.0, 1.1, 1.2, 0.95, 1.05]
    ci1 = r3_stats.bootstrap_ci(values, random.Random(7), draws=500)
    ci2 = r3_stats.bootstrap_ci(values, random.Random(7), draws=500)
    assert ci1 == ci2
    assert ci1 is not None and ci1[0] <= sum(values) / len(values) <= ci1[1]


def test_bootstrap_edge_cases():
    import random

    assert r3_stats.bootstrap_ci([], random.Random(1), draws=10) is None
    ci1 = r3_stats.bootstrap_ci([2.5], random.Random(1), draws=10)
    assert ci1 == [2.5, 2.5]
    # non-finite values filtered before bootstrap
    ci2 = r3_stats.bootstrap_ci([1.0, float("nan"), 3.0], random.Random(2), draws=50)
    assert ci2 is not None


def test_report_determinism_end_to_end(tmp_path):
    runs = [
        _metrics("r1", "A", 0, 17, rmse_over_zero=0.8),
        _metrics("r2", "A", 0, 29, rmse_over_zero=1.0),
        _metrics("r3", "B", 1, 17, rmse_over_zero=1.3),
        _metrics("r4", "B", 3, 17, rmse_over_zero=0.1),  # sealed: must not leak
    ]
    root = _write_campaign(tmp_path, runs, [_ledger_line("r1", "A"), _ledger_line("r2", "A")])
    gates = tmp_path / "gates"
    gates.mkdir()
    (gates / "G0-Data.json").write_text(json.dumps({"gate_id": "G0-Data", "status": "pass", "evaluated_at": "t0"}))

    rep1 = r3_stats.build_report(root, gates, bootstrap_seed=11, bootstrap_draws=100, worst_n=2)
    rep2 = r3_stats.build_report(root, gates, bootstrap_seed=11, bootstrap_draws=100, worst_n=2)
    md1 = r3_stats.render_markdown(rep1)
    md2 = r3_stats.render_markdown(rep2)
    p1 = r3_stats.write_outputs(tmp_path / "out1", "v1", rep1, md1)
    p2 = r3_stats.write_outputs(tmp_path / "out2", "v1", rep2, md2)
    for a, b in zip(p1, p2):
        assert a.read_bytes() == b.read_bytes(), f"non-deterministic output: {a.name}"


# ------------------------------------------------------- sealed-fold


def test_sealed_fold_exclusion_and_flagging(tmp_path):
    runs = [
        _metrics("d1", "A", 0, 17, rmse_over_zero=0.8),
        _metrics("d2", "A", 2, 17, rmse_over_zero=0.9),
        _metrics("s1", "A", 3, 17, rmse_over_zero=0.05),  # sealed, absurdly good
        _metrics("s2", "A", 4, 17, rmse_over_zero=0.05),  # sealed
    ]
    root = _write_campaign(tmp_path, runs, [])
    rep = r3_stats.build_report(root, tmp_path / "no_gates", 1, 50, 5)

    a = rep["arms"]["A"]
    assert a["n_runs"] == 2  # sealed excluded
    assert a["metrics"]["rmse_over_zero"]["mean"] == pytest.approx(0.85)
    assert set(a["run_ids"]) == {"d1", "d2"}
    flagged = rep["sealed_runs_flagged"]
    assert {f["run_id"] for f in flagged} == {"s1", "s2"}
    assert all(f["action"] == "excluded_from_dev_aggregates" for f in flagged)
    assert rep["scan"]["n_sealed_runs"] == 2 and rep["scan"]["n_dev_runs"] == 2
    # sealed never appears in error cards
    for cards in rep["error_cluster_cards"]["cards_by_arm"].values():
        assert all(c["run_id"] in {"d1", "d2"} for c in cards)


def test_other_fold_runs_not_in_dev_aggregates(tmp_path):
    runs = [_metrics("x1", "A", 7, 17, rmse_over_zero=0.5)]  # fold 7: neither dev nor sealed
    root = _write_campaign(tmp_path, runs, [])
    rep = r3_stats.build_report(root, tmp_path / "no_gates", 1, 50, 5)
    assert "A" not in rep["arms"]
    assert rep["scan"]["n_other_fold_runs"] == 1


# ------------------------------------------------------- cost accounting


def test_cost_accounting_arithmetic(tmp_path):
    runs = [
        _metrics("r1", "A", 0, 17, wall_clock_s=100.0, device="cuda"),
        _metrics("r2", "A", 0, 29, wall_clock_s=200.0, device="cpu"),
        # r3: interrupted with no metrics -> ledger span fallback
    ]
    ledger = [
        _ledger_line("r1", "A", status="done", attempt=1, retry_count=0),
        _ledger_line("r2", "A", status="done", attempt=2, retry_count=1),
        _ledger_line("r3", "A", status="interrupted", created="2026-10-20T00:00:00+00:00",
                     updated="2026-10-20T00:05:00+00:00"),
        _ledger_line("r4", "B", status="planned", blocked="confirm_awaiting_dev_positive",
                     created="2026-10-20T00:00:00+00:00", updated="2026-10-20T00:00:01+00:00"),
    ]
    root = _write_campaign(tmp_path, runs, ledger)
    final, raw = r3_stats.load_ledger(root / "run_records.jsonl")
    cost = r3_stats.cost_accounting(final, raw, runs)

    assert cost["totals"]["n_unique_runs"] == 4
    assert cost["totals"]["status_counts"] == {"done": 2, "interrupted": 1, "planned": 1}
    (phase_a, arm_a) = ("R3", "A")
    row = next(r for r in cost["by_phase_arm"] if r["phase"] == phase_a and r["arm"] == arm_a)
    assert row["n_runs"] == 3
    assert row["total_attempts"] == 1 + 2 + 1  # r1=1, r2=2, r3=1
    assert row["n_retried"] == 1  # only r2
    assert row["wall_clock_s_metrics"] == pytest.approx(300.0)
    assert row["wall_clock_s_ledger_span"] == pytest.approx(300.0)  # r3: 5 min
    assert row["gpu_wall_clock_s"] == pytest.approx(100.0)
    assert row["cpu_wall_clock_s"] == pytest.approx(200.0)
    assert row["gpu_hours_estimate"] == pytest.approx(100.0 / 3600.0)
    assert len(cost["blocked"]) == 1
    assert cost["blocked"][0]["reason"] == "confirm_awaiting_dev_positive"
    assert len(cost["retried"]) == 1 and cost["retried"][0]["run_id"] == "r2"


def test_ledger_latest_record_wins(tmp_path):
    ledger = [
        _ledger_line("r1", "A", status="planned", updated="2026-10-20T00:00:00+00:00"),
        _ledger_line("r1", "A", status="running", updated="2026-10-20T00:01:00+00:00"),
        _ledger_line("r1", "A", status="done", updated="2026-10-20T00:02:00+00:00"),
    ]
    root = _write_campaign(tmp_path, [], ledger)
    final, raw = r3_stats.load_ledger(root / "run_records.jsonl")
    assert len(raw) == 3 and len(final) == 1
    assert final[0]["status"] == "done"


# ------------------------------------------- missing/incremental tolerance


def test_missing_and_incremental_runs_tolerance(tmp_path):
    root = tmp_path / "campaign"
    # r_ok: full; r_broken: corrupt json; r_nomemetrics: dir without metrics.json
    (root / "r3_runs" / "r_ok").mkdir(parents=True)
    (root / "r3_runs" / "r_ok" / "metrics.json").write_text(
        json.dumps({k: v for k, v in _metrics("r_ok", "A", 0, 17).items() if k != "run_id"})
    )
    (root / "r3_runs" / "r_broken").mkdir(parents=True)
    (root / "r3_runs" / "r_broken" / "metrics.json").write_text("{not json")
    (root / "r3_runs" / "r_nomemetrics").mkdir(parents=True)
    # ledger: r_ok done, r_pending planned (incremental arrival), malformed line
    with open(root / "run_records.jsonl", "w") as f:
        f.write(json.dumps(_ledger_line("r_ok", "A")) + "\n")
        f.write("{broken json line\n")
        f.write(json.dumps(_ledger_line("r_pending", "A", status="running")) + "\n")

    rep = r3_stats.build_report(root, tmp_path / "no_gates", 3, 50, 5)
    assert rep["scan"]["n_metrics_files"] == 1
    assert rep["scan"]["n_dev_runs"] == 1
    skipped = rep["scan"]["skipped_metrics"]
    assert any(s["run_id"] == "r_broken" and "unreadable" in s["reason"] for s in skipped)
    assert "r_pending" in rep["scan"]["pending_ledger_runs_without_metrics"]
    assert rep["arms"]["A"]["n_runs"] == 1  # only r_ok aggregated
    # cost still computed from ledger despite corrupt line + pending run
    assert rep["cost"]["totals"]["n_unique_runs"] == 2


def test_missing_ledger_and_gates_dirs(tmp_path):
    root = tmp_path / "campaign"
    (root / "r3_runs" / "r1").mkdir(parents=True)
    (root / "r3_runs" / "r1" / "metrics.json").write_text(
        json.dumps({k: v for k, v in _metrics("r1", "A", 0, 17).items() if k != "run_id"})
    )
    rep = r3_stats.build_report(root, tmp_path / "no_gates", 1, 20, 3)
    assert rep["cost"]["totals"]["n_ledger_lines"] == 0
    assert rep["gates"]["summary_table"] == []
    assert rep["arms"]["A"]["n_runs"] == 1


def test_write_only_stats_dir(tmp_path):
    """write_outputs creates exactly the two report files under out_dir."""
    root = _write_campaign(tmp_path, [_metrics("r1", "A", 0, 17)], [])
    rep = r3_stats.build_report(root, tmp_path / "no_gates", 1, 20, 3)
    out = tmp_path / "stats"
    paths = r3_stats.write_outputs(out, "v1", rep, r3_stats.render_markdown(rep))
    assert {p.name for p in paths} == {"r3_stats_report_v1.json", "r3_stats_report_v1.md"}
    assert {p.name for p in out.iterdir()} == {"r3_stats_report_v1.json", "r3_stats_report_v1.md"}
