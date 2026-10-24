"""V5.3-MP noise-injection diagnostic unit tests (plan §9.5).

Driver under test:
``scripts/agent_ops/feature_gate/v53mp_noise_rollout_diag.py`` — a
ZERO-TRAINING-COST diagnostic (frozen S3 checkpoints, σ=0.02 Å
start-geometry noise, exposure-bias via collision-rate change).

Coverage:

* noise RNG determinism (same ``(noise_seed, curve_id)`` ⇒ same draw;
  σ=0 ⇒ exact zeros ⇒ bitwise-identical episode vs baseline);
* perturbation applied ONCE at t=0 only (engine never re-noises mid
  rollout — verified with a spy on the engine's forward primitive);
* exact (Clopper–Pearson) binomial CI + summary math on synthetic
  episode pairs (rates, stop-step changes, endpoint-deviation deltas,
  the §9.5 decision rule in both directions);
* tiered baseline-anchor comparison (deterministic fields bitwise,
  continuous fields within tolerance, wall_seconds excluded);
* CLI smoke via fake run-dir fixtures (tiny real KeyAtomV53MPModel
  checkpoint_best.pt + metrics.json, synthetic episode contexts —
  fixture style mirrored from tests/test_v53mp_rollout_exec.py).

No existing module is edited; the diagnostic imports the frozen
``v53mp_rollout_exec`` runner read-only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agent_ops.feature_gate import (  # noqa: E402
    v53mp_noise_rollout_diag as diag,
)
from scripts.agent_ops.feature_gate.v53mp_rollout_exec import (  # noqa: E402
    EpisodeContextV53MP,
    l_key_from_arc,
    reference_budget,
    run_episode,
)
from pes2ts_core.pesn.contracts.key_atom_v3 import (  # noqa: E402
    IntentRecordV3,
)


# ---------------------------------------------------------------------------
# synthetic fixtures (minimal copies of the rollout-exec test builders —
# those live in the other test module, not in conftest)
# ---------------------------------------------------------------------------
def _synthetic_intent() -> IntentRecordV3:
    return IntentRecordV3(
        intent_id="SYN_fwd",
        curve_id="SYN",
        direction="forward",
        reverse_of_intent_id="SYN_rev",
        intent_pair_id="SYN_pair",
        g_src_bonds=((0, 1, 1.0),),
        g_tgt_bonds=((1, 2, 1.0),),
        cv_deltas=(1.0,),
        g_atoms=(6, 1, 8, 6),
        cv_coordinates=((0, 1), (1, 2)),
        cv_kind="graph_edit_bond_lengths",
    )


def _synthetic_frames() -> list[np.ndarray]:
    return [
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                  [3.0, 0.0, 0.0], [0.0, 5.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0],
                  [2.5, 0.0, 0.0], [0.0, 5.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0],
                  [2.0, 0.0, 0.0], [0.0, 5.0, 0.0]]),
    ]


def _synthetic_ctx(curve_id: str = "SYN") -> EpisodeContextV53MP:
    frames = _synthetic_frames()
    intent = _synthetic_intent()
    key_rows = (0, 1, 2)
    cv_pairs = [(0, 1), (1, 2)]

    def dkey(f: np.ndarray) -> np.ndarray:
        return np.array([float(np.linalg.norm(f[j] - f[i]))
                         for i, j in cv_pairs])

    key_idx = np.asarray(key_rows, dtype=np.int64)
    l_key = l_key_from_arc([f[key_idx] for f in frames])
    return EpisodeContextV53MP(
        curve_id=curve_id,
        direction="forward",
        intent=intent,
        start_node=0,
        target_node=2,
        ts_index=1,
        atomic_numbers=(6, 1, 8, 6),
        start_xyz=tuple(tuple(float(v) for v in row) for row in frames[0]),
        arc_nodes=(0, 1, 2),
        arc_dkey=(dkey(frames[0]), dkey(frames[1]), dkey(frames[2])),
        target_xyz=tuple(tuple(float(v) for v in row) for row in frames[2]),
        ts_xyz=tuple(tuple(float(v) for v in row) for row in frames[1]),
        l_key=float(l_key),
        reference_budget=reference_budget(l_key),
    )


def _tiny_model(tmp_path: Path):
    """Real KeyAtomV53MPModel round-tripped through a state_dict file."""
    torch = pytest.importorskip("torch")
    from pes2ts_core.pesn.models.key_atom_heads_v3 import KeyAtomV53MPModel

    torch.manual_seed(0)
    model = KeyAtomV53MPModel()
    ckpt = tmp_path / "checkpoint_best.pt"
    torch.save({"model": model.state_dict()}, ckpt)

    from scripts.agent_ops.feature_gate.v53mp_rollout_exec import (
        load_v53mp_checkpoint,
    )
    return load_v53mp_checkpoint(ckpt)


def _run_episode(ctx: EpisodeContextV53MP, model: Any, *,
                 budget: int = 5, arm: str = "C", tau_end: float = 0.5,
                 run_id: str = "v53mp_20k_C_s17") -> dict[str, Any]:
    return run_episode(
        ctx, arm=arm, model=model, budget=budget,
        budget_mode="autonomous", run_id=run_id,
        checkpoint_sha256="c" * 64, seed=17, tau_end=tau_end)


# ---------------------------------------------------------------------------
# noise RNG determinism + σ=0 identity
# ---------------------------------------------------------------------------
def test_noise_rng_deterministic_and_input_order_invariant() -> None:
    a = diag.noise_rng(17, "RXN_0000090455").normal(size=4)
    b = diag.noise_rng(17, "RXN_0000090455").normal(size=4)
    assert np.array_equal(a, b)
    # other draws between the two calls never matter (no shared state)
    diag.noise_rng(17, "RXN_0000073773").normal(size=4)
    c = diag.noise_rng(17, "RXN_0000090455").normal(size=4)
    assert np.array_equal(a, c)
    # distinct curve / seed ⇒ distinct streams
    d = diag.noise_rng(17, "RXN_0000073773").normal(size=4)
    e = diag.noise_rng(29, "RXN_0000090455").normal(size=4)
    assert not np.array_equal(a, d)
    assert not np.array_equal(a, e)
    with pytest.raises(diag.NoiseDiagError):
        diag.noise_rng(-1, "RXN_X")


def test_start_noise_draw_shape_scale_and_zero_sigma() -> None:
    draw = diag.start_noise_draw(17, "RXN_X", 7, 0.02)
    assert draw.shape == (7, 3) and draw.dtype == np.float64
    assert np.isfinite(draw).all()
    # sigma=0 ⇒ exact zeros (bitwise), no RNG path taken
    zero = diag.start_noise_draw(17, "RXN_X", 7, 0.0)
    assert np.array_equal(zero, np.zeros((7, 3)))
    # draw scales linearly with sigma (same underlying standard draw)
    small = diag.start_noise_draw(17, "RXN_X", 7, 0.01)
    assert np.allclose(small * 2.0, draw, rtol=0, atol=1e-15)
    with pytest.raises(diag.NoiseDiagError):
        diag.start_noise_draw(17, "RXN_X", 7, -0.01)
    with pytest.raises(diag.NoiseDiagError):
        diag.start_noise_draw(17, "RXN_X", 0, 0.02)


def test_apply_start_noise_zero_sigma_bitwise_identity() -> None:
    ctx = _synthetic_ctx()
    ctx_noisy, noise = diag.apply_start_noise(
        ctx, sigma=0.0, noise_seed=17)
    assert np.array_equal(noise, np.zeros((4, 3)))
    # start geometry bitwise-identical; every other field carried over
    assert ctx_noisy.start_xyz == ctx.start_xyz
    assert ctx_noisy.arc_dkey == ctx.arc_dkey
    assert ctx_noisy.target_xyz == ctx.target_xyz
    assert ctx_noisy.l_key == ctx.l_key
    assert ctx_noisy.intent is ctx.intent
    # original context never mutated
    assert ctx.start_xyz == _synthetic_ctx().start_xyz


def test_apply_start_noise_sigma_positive_only_start_frame() -> None:
    ctx = _synthetic_ctx()
    ctx_noisy, noise = diag.apply_start_noise(ctx, sigma=0.02,
                                              noise_seed=17)
    start = np.asarray(ctx.start_xyz, dtype=np.float64)
    # stored geometry is exactly start + noise (same float64 op)
    assert np.array_equal(np.asarray(ctx_noisy.start_xyz), start + noise)
    assert np.allclose(np.asarray(ctx_noisy.start_xyz) - start, noise,
                       rtol=0, atol=1e-15)
    assert not np.array_equal(np.asarray(ctx_noisy.start_xyz), start)
    # record-side reference diagnostics stay on the pristine arc
    assert ctx_noisy.arc_dkey == ctx.arc_dkey
    assert ctx_noisy.target_xyz == ctx.target_xyz
    assert ctx_noisy.ts_xyz == ctx.ts_xyz
    assert ctx_noisy.arc_nodes == ctx.arc_nodes
    assert ctx_noisy.l_key == ctx.l_key


def test_sigma_zero_episode_equals_baseline(tmp_path: Path) -> None:
    model = _tiny_model(tmp_path)
    ctx = _synthetic_ctx()
    base = _run_episode(ctx, model)
    ctx_noisy, _ = diag.apply_start_noise(ctx, sigma=0.0, noise_seed=17)
    pert = _run_episode(ctx_noisy, model)
    # everything except wall_seconds must be bitwise-identical
    b = {k: v for k, v in base.items() if k != "wall_seconds"}
    p = {k: v for k, v in pert.items() if k != "wall_seconds"}
    assert json.dumps(b, sort_keys=True) == json.dumps(p, sort_keys=True)


# ---------------------------------------------------------------------------
# perturbation applied ONLY at t=0 (engine forward spied on)
# ---------------------------------------------------------------------------
def test_noise_applied_only_at_t0(tmp_path: Path, monkeypatch) -> None:
    """Every model call inside the episode sees the SAME geometry:
    start + one fixed noise draw — noise is never re-applied per step."""
    model = _tiny_model(tmp_path)
    ctx = _synthetic_ctx()
    ctx_noisy, noise = diag.apply_start_noise(ctx, sigma=0.05,
                                              noise_seed=17)
    start = np.asarray(ctx.start_xyz, dtype=np.float64)

    import scripts.agent_ops.feature_gate.v53mp_rollout_exec as engine
    seen_xyz: list[np.ndarray] = []

    def spy(**kwargs: Any):
        seen_xyz.append(np.asarray(kwargs["xyz"], dtype=np.float64).copy())
        k = len(ctx.intent.key_rows())
        return np.zeros((k, 3), dtype=np.float64), 0.0, -10.0

    monkeypatch.setattr(engine, "_predict_forward_v53mp", spy)
    rec = _run_episode(ctx_noisy, model, budget=6)

    assert rec["n_steps_executed"] == 6
    assert len(seen_xyz) == 6
    expected = start + noise
    for xyz in seen_xyz:                      # constant zero action ⇒
        assert np.array_equal(xyz, expected)  # geometry frozen after t0
    # the observed offset is exactly the single t=0 draw
    assert np.array_equal(seen_xyz[0], start + noise)
    assert np.allclose(seen_xyz[0] - start, noise, rtol=0, atol=1e-15)
    # the perturbed start is visible to the harness record
    assert rec["input_step0_sha256"] != _run_episode(ctx, model)["input_step0_sha256"]


# ---------------------------------------------------------------------------
# exact binomial CI + summary math
# ---------------------------------------------------------------------------
def test_clopper_pearson_known_values_and_scipy_crosscheck() -> None:
    # k=0: upper = 1 - (alpha/2)^(1/n)
    lo, hi = diag.clopper_pearson(0, 10)
    assert lo == 0.0
    assert abs(hi - (1.0 - 0.025 ** (1.0 / 10.0))) < 1e-12
    # k=n: lower = (alpha/2)^(1/n)
    lo, hi = diag.clopper_pearson(10, 10)
    assert hi == 1.0
    assert abs(lo - 0.025 ** (1.0 / 10.0)) < 1e-12
    # interior: cross-check against scipy's beta quantiles
    scipy_stats = pytest.importorskip("scipy.stats")
    for k, n in ((1, 10), (5, 10), (3, 48), (17, 48), (106, 144)):
        lo, hi = diag.clopper_pearson(k, n)
        assert abs(lo - scipy_stats.beta.ppf(0.025, k, n - k + 1)) < 1e-9
        assert abs(hi - scipy_stats.beta.ppf(0.975, k + 1, n - k)) < 1e-9
    with pytest.raises(diag.NoiseDiagError):
        diag.clopper_pearson(11, 10)
    with pytest.raises(diag.NoiseDiagError):
        diag.clopper_pearson(1, 0)


def test_collision_rate_summary_math() -> None:
    s = diag.collision_rate_summary([True, False, True, False])
    assert s["n"] == 4 and s["k"] == 2
    assert abs(s["rate"] - 0.5) < 1e-15
    lo, hi = s["ci"]
    assert 0.0 < lo < 0.5 < hi < 1.0
    assert abs(lo - diag.clopper_pearson(2, 4)[0]) < 1e-15
    empty = diag.collision_rate_summary([])
    assert empty == {"n": 0, "k": 0, "rate": None, "ci": None,
                     "ci_level": 0.95}


# ---------------------------------------------------------------------------
# synthetic episode pairs: comparison + summary + decision rule
# ---------------------------------------------------------------------------
def _fake_pair(run_id: str, curve_id: str, *, seed: int,
               base_coll: int, pert_coll: int,
               base_stop: int | None, pert_stop: int | None,
               base_dev: float, pert_dev: float) -> tuple[dict, dict]:
    def rec(n_coll: int, stop: int | None, dev: float,
            budget: int = 20) -> dict:
        n_steps = stop if stop is not None else budget
        steps = [{
            "step": i + 1,
            "collision": i < n_coll,
            "arc_progress": min(1.0, (i + 1) / budget),
            "endpoint_dev_max": dev + 0.1 * i,
        } for i in range(n_steps)]
        return {
            "schema": "pes2ts_v53mp_rollout_episode_v1",
            "claim_scope": "TS_candidate_only",
            "run_id": run_id, "curve_id": curve_id, "direction": "forward",
            "arm": "C", "seed": seed, "budget_mode": "autonomous",
            "budget": budget,
            "online_stop_step": stop,
            "input_step0_sha256": "f" * 64,
            "steps": steps,
            "summary": {
                "n_collisions": n_coll,
                "n_steps_executed": n_steps,
                "stopped_online": stop is not None,
                "endpoint_reached": False,
                "final_dev_target_max": dev,
                "final_target_rmsd_all": dev / 2.0,
            },
        }
    return rec(base_coll, base_stop, base_dev), \
        rec(pert_coll, pert_stop, pert_dev)


def test_compare_episode_pair_fields() -> None:
    b, p = _fake_pair("r", "c1", seed=17, base_coll=0, pert_coll=3,
                      base_stop=None, pert_stop=None,
                      base_dev=1.0, pert_dev=1.5)
    c = diag.compare_episode_pair(b, p)
    assert c["collision_new"] is True
    assert c["collision_cleared"] is False
    assert c["collision_flag_changed"] is True
    assert c["n_collisions_delta"] == 3
    assert abs(c["final_dev_target_delta"] - 0.5) < 1e-15
    assert abs(c["final_rmsd_delta"] - 0.25) < 1e-15
    assert c["stop_step_changed"] is False
    assert c["start_step0_changed"] is False
    # reverse direction: collision cleared
    c2 = diag.compare_episode_pair(
        *_fake_pair("r", "c2", seed=17, base_coll=2, pert_coll=0,
                    base_stop=None, pert_stop=None, base_dev=1.0,
                    pert_dev=1.0))
    assert c2["collision_cleared"] is True and c2["collision_new"] is False


def test_divergence_profile_min_length() -> None:
    b1, p1 = _fake_pair("r", "c1", seed=17, base_coll=0, pert_coll=0,
                        base_stop=None, pert_stop=None, base_dev=1.0,
                        pert_dev=2.0)
    # perturbed stops early (arm C online stop at step 2)
    b2, p2 = _fake_pair("r", "c2", seed=17, base_coll=0, pert_coll=0,
                        base_stop=None, pert_stop=2, base_dev=1.0,
                        pert_dev=1.0)
    prof1 = diag.divergence_profile(b1, p1)
    assert len(prof1) == 20
    # endpoint_dev differs by 1.0 at every paired step
    assert all(abs(row["abs_d_endpoint_dev"] - 1.0) < 1e-15
               for row in prof1)
    prof2 = diag.divergence_profile(b2, p2)
    assert len(prof2) == 2  # paired only up to the shorter episode
    assert prof2[-1]["step"] == 2


def test_summarize_pairs_rates_ci_and_stop_changes() -> None:
    pairs = [
        _fake_pair("r1", "c1", seed=17, base_coll=0, pert_coll=2,
                   base_stop=None, pert_stop=None, base_dev=1.0,
                   pert_dev=1.2),
        _fake_pair("r1", "c2", seed=17, base_coll=1, pert_coll=1,
                   base_stop=None, pert_stop=None, base_dev=2.0,
                   pert_dev=2.0),
        _fake_pair("r1", "c3", seed=17, base_coll=0, pert_coll=0,
                   base_stop=8, pert_stop=5, base_dev=1.5,
                   pert_dev=1.5),
        _fake_pair("r1", "c4", seed=17, base_coll=0, pert_coll=0,
                   base_stop=None, pert_stop=None, base_dev=3.0,
                   pert_dev=4.0),
    ]
    s = diag.summarize_pairs(pairs, arm="C", budget_mode="autonomous",
                             seed="17")
    assert s["n_pairs"] == 4
    br, pr = s["collision_rate_baseline"], s["collision_rate_perturbed"]
    assert (br["k"], br["n"], br["rate"]) == (1, 4, 0.25)
    assert (pr["k"], pr["n"], pr["rate"]) == (2, 4, 0.5)
    assert abs(pr["ci"][0] - diag.clopper_pearson(2, 4)[0]) < 1e-15
    assert s["collision_flag_changed"] == {"n": 1, "n_new": 1,
                                           "n_cleared": 0}
    assert s["stop_step_changed"]["n_changed"] == 1
    assert abs(s["stop_step_changed"]["fraction"] - 0.25) < 1e-15
    assert s["stop_step_changed"]["baseline_online_halted"] == 1
    assert s["stop_step_changed"]["perturbed_online_halted"] == 1
    # |Δdev| median of [0.2, 0.0, 0.0, 1.0] = 0.1
    assert abs(s["endpoint_dev_abs_delta"]["median"] - 0.1) < 1e-15
    assert abs(s["endpoint_dev_abs_delta"]["max"] - 1.0) < 1e-15
    # profile spans the longest paired prefix (20 steps); n_pairs shrinks
    # where either episode stopped earlier (c3: perturbed stops at 5)
    assert len(s["divergence_profile"]) == 20
    assert s["divergence_profile"][0]["n_pairs"] == 4
    assert s["divergence_profile"][5]["n_pairs"] == 3
    assert s["start_step0_changed_fraction"] == 0.0


def test_decision_rule_both_directions() -> None:
    def group(base_k: int, pert_k: int, n: int) -> dict:
        pairs = [
            _fake_pair("r", f"c{i}", seed=17,
                       base_coll=1 if i < base_k else 0,
                       pert_coll=1 if i < pert_k else 0,
                       base_stop=None, pert_stop=None,
                       base_dev=1.0, pert_dev=1.0)
            for i in range(n)
        ]
        return diag.summarize_pairs(pairs, arm="C",
                                    budget_mode="autonomous",
                                    seed="pooled")

    # significantly worse: perturbed CI lower bound (k=20/24 → ~0.61)
    # exceeds the baseline rate (k=0)
    worse = group(0, 20, 24)
    assert worse["significant_collision_worsening"] is True
    v = diag.decide_verdict([worse], "autonomous")
    assert v["verdict"] == diag.VERDICT_RETRAIN
    assert v["arms_significantly_worse"] == ["C"]

    # unchanged rates ⇒ park
    same = group(12, 12, 24)
    assert same["significant_collision_worsening"] is False
    v = diag.decide_verdict([same], "autonomous")
    assert v["verdict"] == diag.VERDICT_PARK
    assert v["arms_significantly_worse"] == []


# ---------------------------------------------------------------------------
# tiered baseline-anchor comparison
# ---------------------------------------------------------------------------
def _anchor_records() -> tuple[dict, dict]:
    b, p = _fake_pair("r1", "c1", seed=17, base_coll=1, pert_coll=0,
                      base_stop=None, pert_stop=None, base_dev=1.0,
                      pert_dev=1.0)
    stored = dict(b)
    rerun = json.loads(json.dumps(b))  # deep copy, JSON-safe
    return stored, rerun


def test_anchor_comparison_bitwise_and_wall_excluded() -> None:
    stored, rerun = _anchor_records()
    rerun["wall_seconds"] = 123.456  # never compared
    diff, stats = diag.compare_anchor_records(stored, rerun)
    assert diff is None
    assert stats["n_continuous_compared"] > 0
    assert stats["max_continuous_abs_dev"] == 0.0


def test_anchor_comparison_tiered_tolerance() -> None:
    stored, rerun = _anchor_records()
    # 1e-9 FP noise on continuous fields: within tolerance
    rerun["summary"]["final_dev_target_max"] = \
        stored["summary"]["final_dev_target_max"] + 1e-9
    diff, stats = diag.compare_anchor_records(
        stored, rerun, continuous_tol=1e-6)
    assert diff is None
    assert stats["max_continuous_abs_dev"] > 0.0
    assert stats["max_continuous_abs_dev"] < 1e-8
    # 1e-3 deviation: over tolerance ⇒ mismatch
    rerun["summary"]["final_dev_target_max"] = \
        stored["summary"]["final_dev_target_max"] + 1e-3
    diff, _ = diag.compare_anchor_records(
        stored, rerun, continuous_tol=1e-6)
    assert diff is not None and "final_dev_target_max" in diff


def test_anchor_comparison_discrete_fields_must_be_bitwise() -> None:
    stored, rerun = _anchor_records()
    # a discrete flag flip is NEVER tolerated
    rerun["steps"][0]["collision"] = not stored["steps"][0]["collision"]
    diff, _ = diag.compare_anchor_records(stored, rerun)
    assert diff is not None and "collision" in diff
    # exact-float fields are bitwise even at 1-ulp
    stored2, rerun2 = _anchor_records()
    rerun2["summary"]["final_dev_target_max"] = math.nextafter(
        stored2["summary"]["final_dev_target_max"], math.inf)
    diff, _ = diag.compare_anchor_records(
        stored2, rerun2, continuous_tol=1e-6)
    # final_dev_target_max is NOT in the exact-float set → tolerated
    assert diff is None
    stored3, rerun3 = _anchor_records()
    stored3["summary"]["arc_progress_series"] = [0.0, 0.5, 1.0]
    rerun3["summary"]["arc_progress_series"] = [0.0, 0.5,
                                                math.nextafter(1.0, 0.0)]
    diff, _ = diag.compare_anchor_records(
        stored3, rerun3, continuous_tol=1e-6)
    assert diff is not None and "arc_progress_series" in diff


def test_assert_baseline_anchor_missing_and_pass() -> None:
    stored, rerun = _anchor_records()
    anchor = {diag.episode_key(stored): stored}
    summary = diag.assert_baseline_anchor([rerun], anchor,
                                          require_all=True)
    assert summary["n_episodes_checked"] == 1
    assert summary["deterministic_fields_bitwise_exact"] is True
    # unknown key with require_all ⇒ hard failure
    other, _ = _anchor_records()
    other["curve_id"] = "c404"
    with pytest.raises(diag.NoiseDiagError):
        diag.assert_baseline_anchor([other], anchor, require_all=True)
    # same without require_all ⇒ skipped, no error
    summary = diag.assert_baseline_anchor([other], anchor,
                                          require_all=False)
    assert summary["n_episodes_checked"] == 0


# ---------------------------------------------------------------------------
# CLI smoke with fake run dirs (engine-faithful checkpoint loader)
# ---------------------------------------------------------------------------
def _fake_run_dir(runs_root: Path, tier: str, arm: str, seed: int,
                  model: Any) -> Path:
    import torch
    run_dir = runs_root / f"v53mp_{tier}_{arm}_s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "checkpoint_best.pt"
    torch.save({"model": model.state_dict()}, ckpt)
    import hashlib
    sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    (run_dir / "metrics.json").write_text(json.dumps({
        "arm": arm, "seed": seed,
        "checkpoint_best_sha256": sha,
        "endpoint_head": {"tau_end": 0.5},
    }), encoding="utf-8")
    return run_dir


def _cli_args(tmp_path: Path, *, sigma: float) -> argparse.Namespace:
    out_dir = tmp_path / f"out_{sigma}"
    return argparse.Namespace(
        tables_dir=str(tmp_path / "tables"),
        split=str(tmp_path / "split.json"),
        runs_root=str(tmp_path / "runs"),
        arms="C", seeds="17", n_curves=2, autonomous_steps=200,
        sigma=sigma, noise_seed=17,
        out_dir=str(out_dir),
        anchor_dir=str(tmp_path / "anchor"),
        skip_anchor=False,
        report_json=str(out_dir / "report.json"),
        report_md=str(out_dir / "report.md"),
        tier="20k", cache_curves=8, device="cpu", smoke=True,
    )


def test_cli_smoke_end_to_end(tmp_path: Path, monkeypatch) -> None:
    model = _tiny_model(tmp_path)
    _fake_run_dir(tmp_path / "runs", "20k", "C", 17, model)

    # inject synthetic contexts (the only piece needing real tables)
    monkeypatch.setattr(
        diag, "load_dataset_contexts",
        lambda tables_dir, split_path, *, n_curves, directions,
        cache_curves: ([_synthetic_ctx()], {
            "split": {"path": str(split_path), "sha256": "0" * 64,
                      "frozen": True, "subset_used": "dev",
                      "selection_salt": "17", "n_curves": 1,
                      "curve_ids": ["SYN"],
                      "curve_list_sha256": "1" * 64},
            "directions": list(directions)}))
    monkeypatch.setattr(diag, "_assert_v53mp_root", lambda p: p)

    # σ=0 run: perturbed episodes must equal the baseline exactly
    rc = diag.main(["run-noise-diag", *_ns_to_argv(_cli_args(
        tmp_path, sigma=0.0))])
    assert rc == 0
    out0 = tmp_path / "out_0.0"
    base0 = _read_jsonl(out0 / "episodes_baseline.jsonl")
    pert0 = _read_jsonl(out0 / "episodes_perturbed.jsonl")
    assert len(base0) == len(pert0) == 2  # 1 ctx × 2 budget modes
    for b, p in zip(base0, pert0):
        bb = {k: v for k, v in b.items() if k != "wall_seconds"}
        pp = {k: v for k, v in p.items() if k != "wall_seconds"}
        assert json.dumps(bb, sort_keys=True) == json.dumps(pp, sort_keys=True)

    # σ=0.02 run: perturbed trajectories must differ; outputs complete
    rc = diag.main(["run-noise-diag", *_ns_to_argv(_cli_args(
        tmp_path, sigma=0.02))])
    assert rc == 0
    out1 = tmp_path / "out_0.02"
    base1 = _read_jsonl(out1 / "episodes_baseline.jsonl")
    pert1 = _read_jsonl(out1 / "episodes_perturbed.jsonl")
    assert len(base1) == len(pert1) == 2
    for b, p in zip(base1, pert1):
        assert b["input_step0_sha256"] != p["input_step0_sha256"]
        assert b["steps"] != p["steps"]
        assert p["claim_scope"] == "TS_candidate_only"
    manifest = json.loads((out1 / "noise_diag_manifest.json").read_text())
    assert manifest["schema"] == diag.NOISE_MANIFEST_SCHEMA
    assert manifest["sigma_angstrom"] == 0.02
    assert manifest["noise_seed"] == 17
    assert manifest["n_episodes_baseline"] == 2
    assert manifest["n_episodes_perturbed"] == 2
    assert manifest["noise_per_curve"]["SYN"]["applied"] == "once_at_t0"
    # baseline records reproduced identically across the two runs
    for b0, b1 in zip(base0, base1):
        bb0 = {k: v for k, v in b0.items() if k != "wall_seconds"}
        bb1 = {k: v for k, v in b1.items() if k != "wall_seconds"}
        assert json.dumps(bb0, sort_keys=True) == json.dumps(bb1, sort_keys=True)
    # report artifacts written and carry the diagnostic-only claim
    report = json.loads((out1 / "report.json").read_text())
    assert report["schema"] == diag.NOISE_REPORT_SCHEMA
    assert "NOT part of S4.1" in report["diagnostic_only_note"]
    assert report["verdict"]["verdict"] in (diag.VERDICT_RETRAIN,
                                            diag.VERDICT_PARK)
    md = (out1 / "report.md").read_text(encoding="utf-8")
    assert "S4.1" in md and "σ = 0.02" in md


def _ns_to_argv(ns: argparse.Namespace) -> list[str]:
    argv = [
        "--tables-dir", ns.tables_dir, "--split", ns.split,
        "--runs-root", ns.runs_root, "--arms", ns.arms,
        "--seeds", ns.seeds, "--n-curves", str(ns.n_curves),
        "--autonomous-steps", str(ns.autonomous_steps),
        "--sigma", str(ns.sigma), "--noise-seed", str(ns.noise_seed),
        "--out-dir", ns.out_dir, "--anchor-dir", ns.anchor_dir,
        "--report-json", ns.report_json, "--report-md", ns.report_md,
        "--tier", ns.tier, "--cache-curves", str(ns.cache_curves),
        "--device", ns.device,
    ]
    if ns.smoke:
        argv.append("--smoke")
    return argv


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]
