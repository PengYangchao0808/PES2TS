"""V5.3-MP scale-ladder paired-CI tool: synthetic acceptance tests.

Nothing here touches the real multi-GB tier tables or checkpoints — the
``predict`` half is exercised on the real corpus by the run itself, and its
faithfulness gate is validated there.  These tests pin the pieces that decide
whether a published CI is trustworthy:

* the paired bootstrap returns a delta equal to the point estimate and a CI
  that contains it;
* ``_assert_matches`` refuses any drift above the per-metric tolerance
  (fail-closed: a bad re-eval must never be written as a per-curve artifact);
* ``cmd_stats`` refuses a corpus mismatch between the two tiers;
* the markdown renderer emits one table per arm without crashing on int seeds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.campaigns import r3_v53mp_scale_paired_ci as cli


# --------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------


def test_paired_median_delta_is_the_point_estimate() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = a - 0.5
    out = cli._paired_ci_diff_of_medians(a, b, draws=200, seed=7)
    assert out["delta"] == pytest.approx(-0.5)
    lo, hi = out["ci95"]
    assert lo <= out["delta"] <= hi
    assert out["unit"] == "dev_curve"
    assert out["draws"] == 200


def test_paired_median_delta_is_zero_for_identical_inputs() -> None:
    a = np.linspace(0.8, 1.2, 50)
    out = cli._paired_ci_diff_of_medians(a, a.copy(), draws=100, seed=1)
    assert out["delta"] == pytest.approx(0.0)
    assert out["ci95"] == [0.0, 0.0]


def test_paired_mean_delta_uses_the_paired_vectors() -> None:
    a = np.array([0.0, 1.0, 2.0])
    b = np.array([0.5, 1.5, 2.5])
    out = cli._paired_ci_mean_diff(a, b, draws=50, seed=3)
    assert out["delta"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# faithfulness gate
# --------------------------------------------------------------------------


def test_assert_matches_accepts_float32_noise() -> None:
    cli._assert_matches("rmse", 0.8921877275, 0.8921877298, 1e-6)


def test_assert_matches_refuses_drift_above_tolerance() -> None:
    with pytest.raises(cli.PairedCIError) as exc:
        cli._assert_matches("sign", 0.8333, 0.8475, 1e-9)
    assert "re-eval" in str(exc.value)


def test_assert_matches_refuses_missing_values() -> None:
    with pytest.raises(cli.PairedCIError):
        cli._assert_matches("auprc", None, 0.3, 1e-3)


# --------------------------------------------------------------------------
# stats over synthetic per-curve artifacts
# --------------------------------------------------------------------------


def _write_run(root: Path, tier: str, arm: str, seed: int, *,
               curves: list[str], shift: float) -> None:
    payload = {
        "schema": "v53mp_scale_paired_per_curve_v1",
        "tier": tier, "arm": arm, "seed": seed,
        "run_id": "v53mp_{}_{}_s{}".format(tier, arm, seed),
        "n_dev_positions": 10,
        "n_dev_curves": len(curves),
        "recomputed": {},
        "gates_vs_metrics_json": {},
        "per_curve_rmse_over_zero": {c: 0.9 + shift for c in curves},
        "per_curve_sign": {c: 0.85 + shift for c in curves},
        "per_curve_q_mae": {c: 0.50 - shift for c in curves},
        "per_curve_q_corr": {c: 0.4 for c in curves},
    }
    (root / "per_curve_{}_{}_s{}.json".format(tier, arm, seed)).write_text(
        json.dumps(payload), encoding="utf-8")


def _write_endpoint(root: Path, tier: str, arm: str, seed: int,
                    curves: list[str], *, positive: bool) -> None:
    rows = len(curves) * 2
    scores = np.linspace(0.1, 0.9, rows, dtype=np.float32)
    y = np.zeros(rows, dtype=np.int8)
    if positive:
        y[::3] = 1
    np.savez_compressed(
        root / "endpoint_rows_{}_{}_s{}.npz".format(tier, arm, seed),
        scores=scores, y=y,
        curves=np.asarray([c for c in curves for _ in (0, 1)], dtype="U32"))


def test_stats_writes_json_and_markdown(tmp_path: Path,
                                        capsys: pytest.CaptureFixture[str]) -> None:
    curves = ["RXN_{:04d}".format(i) for i in range(120)]
    for tier, shift in (("20k", 0.0), ("50k", -0.01)):
        for seed in (17, 29, 43):
            _write_run(tmp_path, tier, "C", seed, curves=curves, shift=shift)
            _write_endpoint(tmp_path, tier, "C", seed, curves,
                            positive=(tier == "50k"))
    rc = cli.main(["stats", "--dir", str(tmp_path), "--a", "20k", "--b", "50k",
                   "--draws", "50", "--endpoint-draws", "20"])
    assert rc == 0
    doc = json.loads((tmp_path / "scale_paired_ci_20k_vs_50k.json").read_text())
    assert doc["corpus"]["identical"] is True
    assert doc["arms"]["C"]["per_seed"]["17"]["move_rmse_over_zero"]["delta"] \
        == pytest.approx(-0.01)
    md = (tmp_path / "scale_paired_ci_20k_vs_50k.md").read_text()
    assert "## Arm C" in md and "Δ AUPRC" in md


def test_stats_refuses_corpus_mismatch(tmp_path: Path) -> None:
    curves_a = ["RXN_{:04d}".format(i) for i in range(120)]
    curves_b = curves_a[:-1]
    for seed in (17, 29, 43):
        _write_run(tmp_path, "20k", "C", seed, curves=curves_a, shift=0.0)
        _write_run(tmp_path, "50k", "C", seed, curves=curves_b, shift=-0.01)
    with pytest.raises(cli.PairedCIError) as exc:
        cli.cmd_stats(_ns(tmp_path))
    assert "CORPUS_MISMATCH" in str(exc.value)


class _ns:  # tiny argparse.Namespace stand-in
    def __init__(self, root: Path) -> None:
        self.dir = str(root)
        self.a = "20k"
        self.b = "50k"
        self.draws = 20
        self.endpoint_draws = 0
