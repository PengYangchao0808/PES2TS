"""B6 acceptance: reaction-level metrics, frozen benchmark, baselines same-input."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tests.pesn.conftest import build_synthetic_reaction_data  # noqa: E402
from pes2ts_core.pesn.offline.evaluation.benchmark import run_frozen_benchmark  # noqa: E402
from pes2ts_core.pesn.offline.evaluation.metrics import (  # noqa: E402
    activity_metrics,
    bootstrap_ci,
    direction_metrics,
    pair_concordance,
    progress_metrics,
)
from pes2ts_core.pesn.offline.training.trainer import (  # noqa: E402
    TrainSettings,
    build_teacher_model,
)


def _view(rid: str, s: list[float], s_hat: list[float]) -> dict:
    return {
        rid: {
            "s": s,
            "s_hat": s_hat,
            "d": [abs(v) for v in s],
            "scores": [-abs(v) for v in s_hat],
            "cos": [],
            "act_y": [],
            "act_p": [],
        }
    }


def test_localization_top1_better_than_or_equal_top3() -> None:
    view = _view("r1", [-1.0, -0.4, 0.0, 0.6, 1.0], [0.9, -0.2, 0.0, 0.5, -0.8])
    metrics = progress_metrics(view)
    assert metrics["n_reactions_localization"] == 1
    assert metrics["localization_top3_mean_d"] <= metrics["localization_top1_mean_d"] + 1e-12


def test_localization_excludes_exact_ts_anchor() -> None:
    view = _view("r1", [-1.0, 0.0, 1.0], [0.0, 5.0, 0.0])
    metrics = progress_metrics(view)
    assert metrics["mae_s"] > 0  # predictions are garbage
    assert metrics["localization_top1_mean_d"] == pytest.approx(1.0)


def test_perfect_prediction_reaches_anchor_exclusion_floor() -> None:
    # The TS anchor frame is excluded from localization, so even a perfect
    # model's top-1 error floors at the nearest non-anchor |s|.
    view = _view("r1", [-0.8, -0.3, 0.0, 0.25, 0.4], [-0.8, -0.3, 0.0, 0.25, 0.4])
    metrics = progress_metrics(view)
    assert metrics["mae_s"] == pytest.approx(0.0)
    assert metrics["localization_top1_mean_d"] == pytest.approx(0.25)
    assert metrics["localization_top3_mean_d"] == pytest.approx(0.25)


def test_pair_concordance_bounds_and_perfect_case() -> None:
    perfect = _view("r1", [-1.0, -0.5, 0.5, 1.0], [-1.0, -0.5, 0.5, 1.0])
    assert pair_concordance(perfect, 0.05)["pair_concordance"] == pytest.approx(1.0)
    # Sign inversion alone preserves |s| (proximity is invariant under R/P
    # exchange), so the concordance stays perfect; an anti-correlated score
    # ordering is what must break it.
    sign_flipped = _view("r1", [-1.0, -0.5, 0.5, 1.0], [1.0, 0.5, -0.5, -1.0])
    assert pair_concordance(sign_flipped, 0.05)["pair_concordance"] == pytest.approx(1.0)
    anti = {
        "r1": {
            "s": [-1.0, -0.4, 0.5, 0.9],
            "s_hat": [-1.0, -0.4, 0.5, 0.9],
            "d": [1.0, 0.4, 0.5, 0.9],
            "scores": [1.0, 0.4, 0.5, 0.9],
            "cos": [],
            "act_y": [],
            "act_p": [],
        }
    }
    assert pair_concordance(anti, 0.05)["pair_concordance"] == pytest.approx(0.0)


def test_direction_and_activity_metrics_shapes() -> None:
    view = {
        "r1": {
            "s": [-0.5, 0.5],
            "s_hat": [-0.4, 0.4],
            "d": [0.5, 0.5],
            "scores": [0.0, 0.0],
            "cos": [1.0, 0.5],
            "act_y": [1.0, 0.0, 1.0, 0.0],
            "act_p": [0.9, 0.1, 0.8, 0.4],
        }
    }
    direction = direction_metrics(view)
    assert direction["direction_cosine_mean"] == pytest.approx(0.75)
    activity = activity_metrics(view)
    assert 0.0 <= activity["activity_pr_auc"] <= 1.0
    assert activity["n_activity_labels"] == 4


def test_bootstrap_ci_brackets_mean() -> None:
    rng = np.random.default_rng(5)
    values = rng.normal(size=200).tolist()
    interval = bootstrap_ci(values, iterations=200)
    assert interval["ci95_low"] < interval["mean"] < interval["ci95_high"]
    assert interval["n"] == 200


def test_frozen_benchmark_deterministic_and_covers_baselines() -> None:
    model = build_teacher_model(TrainSettings(hidden=16, n_layers=2, n_basis=16))
    reactions = [
        build_synthetic_reaction_data(f"RXN_EV{i:010d}") for i in range(3)
    ]
    first = run_frozen_benchmark(model, reactions, eval_name="t1")
    second = run_frozen_benchmark(model, reactions, eval_name="t1")
    assert first["model"]["mae_s"] == pytest.approx(second["model"]["mae_s"], abs=1e-12)
    assert first["baseline_constant"]["mae_s"] >= 0.0
    assert 0.0 <= first["baseline_rule"]["mae_s"]
    assert first["n_reactions"] == 3
    # a fit call would change weights; none happens here — verified by identity
    assert first["model"]["direction_cosine_mean"] == pytest.approx(
        second["model"]["direction_cosine_mean"], abs=1e-12, nan_ok=True
    )
