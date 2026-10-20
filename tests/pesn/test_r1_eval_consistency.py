"""R1 evaluator consistency gates (campaign V5_1A_R1_REPAIR, CPU only).

Two layers:

* REAL-DATA consistency gate: rebuild fold-0 inner-val pools with the
  frozen V5.1a config (``config/campaigns/v5_1_sprint.yaml``, inner-val
  seed 20261019 / n_val 8), run ``checkpoint_best.pt`` of
  ``B0_f0_s17_v51`` through :func:`r1_eval.predict_pool`, and assert
  :func:`r1_eval.canonical_metrics` reproduces the landed
  ``metrics.json`` ``raw[update=best_update].aggregate`` within a 2e-3
  RELATIVE tolerance (GPU-trained checkpoint vs CPU recompute — actual
  deltas are printed).  The rebuilt inner-val cluster ids must equal
  ``run_record.inner_val_split.val_ids``.  Skips gracefully when the
  checkpoint or the V5 feature shards are absent (clean checkout
  without campaign outputs).
* UNIT gates: analytic scale solution, linear scale application, the
  frozen pair-bucket edge policy, and seeded bootstrap reproducibility.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pes2ts_core.pesn.models import require_torch
from pes2ts_core.pesn.offline.evaluation import r1_eval

torch = require_torch()

REPO = Path(__file__).resolve().parents[2]
RUN_DIR = REPO / "outputs/pesn/v5_1/runs/B0_f0_s17_v51"
CKPT_PATH = RUN_DIR / "checkpoints/checkpoint_best.pt"
CONFIG_PATH = REPO / "config/campaigns/v5_1_sprint.yaml"
SPLIT_PATH = REPO / "data/manifests/v5/split_manifest.json"
FEATURE_ROOT = REPO / "outputs/pesn/v5/features"
INNER_VAL_SEED = 20261019
INNER_VAL_N = 8
#: GPU-trained checkpoint vs CPU recompute tolerance (relative).
REL_TOL = 2e-3
#: Metrics pinned by the orchestrator spec for the consistency gate.
COMPARED_METRICS = ("rmse_over_zero", "alpha_median", "sre_cluster_median",
                    "cos_eff_near_ts_median")


def _real_data_available() -> bool:
    return (CKPT_PATH.is_file()
            and CONFIG_PATH.is_file()
            and SPLIT_PATH.is_file()
            and FEATURE_ROOT.is_dir())


requires_real_data = pytest.mark.skipif(
    not _real_data_available(),
    reason="V5.1a checkpoint / V5 feature shards / split manifest absent")


# ---------------------------------------------------------------------------
# Real-data consistency gate
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_consistency():
    """One shared prediction + metric pass over the fold-0 inner-val pool.

    Returns ``(recomputed_aggregate, metrics_best_row, val_ids, deltas)``.
    """
    if not _real_data_available():
        pytest.skip("real V5.1a artifacts absent")
    from pes2ts_core.config_loader import load_config_mapping
    from pes2ts_core.pesn.models.key_atom_egnn import StarEGNN
    from pes2ts_core.pesn.offline.training.key_atom_bc import load_lane_b_pools

    config = load_config_mapping(CONFIG_PATH)
    pools = load_lane_b_pools(
        config, fold=0, d0=0.05, split_path=str(SPLIT_PATH),
        inner_val={"n_val": INNER_VAL_N, "seed": INNER_VAL_SEED})
    inner_rows = list(pools["inner_val_dense"])

    model = StarEGNN().to(torch.float32).to(torch.device("cpu"))
    doc = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    assert str(doc.get("schema_version")) == "pesn_key_atom_bc_checkpoint_v1"
    model.load_state_dict(doc["model"])

    records = r1_eval.predict_pool(
        model, inner_rows, torch.device("cpu"),
        config=config, fold=0, d0=0.05, split_path=str(SPLIT_PATH))
    recomputed = r1_eval.canonical_metrics(records)

    run_record = json.loads(
        (RUN_DIR / "run_record.json").read_text(encoding="utf-8"))
    best_update = int(run_record["best_update"])
    metrics = json.loads((RUN_DIR / "metrics.json").read_text(encoding="utf-8"))
    best_rows = [r for r in metrics["raw"]
                 if r.get("split") == "inner_val"
                 and int(r["update"]) == best_update]
    assert best_rows, (f"no inner_val row at best_update={best_update} "
                       "in metrics.json")
    return {
        "records": records,
        "recomputed": recomputed,
        "best_row": best_rows[0],
        "val_ids": list(run_record["inner_val_split"]["val_ids"]),
        "pool_val_ids": list(pools["inner_split"]["val_ids"]),
        "best_update": best_update,
    }


@requires_real_data
def test_inner_val_cluster_ids_match_run_record(real_consistency):
    """Rebuilt inner-val split reproduces the frozen val cluster ids."""
    rc = real_consistency
    assert rc["pool_val_ids"] == rc["val_ids"]
    seen = sorted({str(r["cluster"]) for r in rc["records"]})
    assert seen == sorted(rc["val_ids"])


@requires_real_data
def test_canonical_metrics_match_metrics_json(real_consistency):
    """CPU recompute of inner-val aggregates reproduces the landed
    metrics.json best row (relative tolerance 2e-3; deltas printed)."""
    rc = real_consistency
    landed = rc["best_row"]["aggregate"]
    agg = rc["recomputed"]["aggregate"]
    deltas: dict[str, tuple[float, float, float]] = {}
    for key in COMPARED_METRICS:
        a, b = agg.get(key), landed.get(key)
        assert a is not None and b is not None, f"{key} missing (a={a}, b={b})"
        rel = abs(a - b) / max(abs(b), 1e-12)
        deltas[key] = (float(a), float(b), float(rel))
    # report actual deltas in test output
    lines = ["consistency deltas (recomputed vs landed):"]
    for key, (a, b, rel) in deltas.items():
        lines.append(f"  {key}: {a:.10f} vs {b:.10f}  abs={abs(a-b):.3e}  "
                     f"rel={rel:.3e}")
    print("\n".join(lines))
    for key, (_a, _b, rel) in deltas.items():
        assert rel <= REL_TOL, f"{key}: relative delta {rel:.3e} > {REL_TOL}"


# ---------------------------------------------------------------------------
# Unit gates (synthetic, CPU-fast)
# ---------------------------------------------------------------------------

def _synthetic_records(rng_seed: int = 7, n_clusters: int = 4,
                       n_per_cluster: int = 3) -> list[dict]:
    """Deterministic synthetic records with well-separated pair amplitudes.

    Key atoms sit on an x-axis chain (bond length 1); teacher displaces
    them by ±0.05 Å along x (|Δd_true| = 0.1 Å per pair — strong bucket,
    far from every threshold); the prediction is a 0.6 scaled teacher plus
    small y/z-only noise (so Δd_pred ≈ 0.06 Å ≫ sign threshold and the
    sign always matches — sign metrics stay exact under positive
    rescaling of u).  Branch 0 sample of every cluster is near-TS.
    """
    rng = np.random.default_rng(rng_seed)
    records: list[dict] = []
    for c in range(n_clusters):
        cluster = f"RXN_SYNTH{c:04d}"
        for i in range(n_per_cluster):
            pos = np.concatenate(
                [np.arange(6, dtype=np.float64).reshape(6, 1),
                 np.zeros((6, 2), dtype=np.float64)], axis=1)
            key_rows = [0, 1, 2, 3]
            pairs = [(0, 1), (2, 3)]
            y = np.zeros((4, 3), dtype=np.float64)
            y[0, 0] = -0.05
            y[1, 0] = 0.05
            y[2, 0] = 0.05
            y[3, 0] = -0.05
            noise = np.zeros((4, 3), dtype=np.float64)
            noise[:, 1:] = rng.normal(scale=0.002, size=(4, 2))
            u = 0.6 * y + noise
            records.append({
                "cluster": cluster,
                "reaction_id": cluster,
                "branch": i % 2,
                "near_ts": 0.1 if i == 0 else 0.6 + 0.1 * i,
                "pos": pos,
                "key_rows": key_rows,
                "pairs": pairs,
                "pair_edit": [1.0, -1.0],
                "u": u,
                "y": y,
            })
    return records


def test_scale_solution_matches_closed_form():
    records = _synthetic_records()
    num = den = 0.0
    for r in records:
        num += float(np.sum(r["u"] * r["y"]))
        den += float(np.sum(r["u"] * r["u"]))
    expected = max(num / den, 0.0)
    got = r1_eval.scale_solution(records, weight="all")
    assert got == pytest.approx(expected, rel=1e-12)

    num_e = den_e = 0.0
    for r in records:
        m = np.linalg.norm(r["y"], axis=1) > r1_eval.EFFECTIVE_ATOM_TAU
        num_e += float(np.sum(r["u"][m] * r["y"][m]))
        den_e += float(np.sum(r["u"][m] * r["u"][m]))
    got_e = r1_eval.scale_solution(records, weight="effective")
    assert got_e == pytest.approx(max(num_e / den_e, 0.0), rel=1e-12)

    with pytest.raises(ValueError):
        r1_eval.scale_solution(records, weight="bogus")


def test_scale_solution_clips_negative():
    records = _synthetic_records()
    for r in records:
        r["u"] = -1.5 * r["y"]  # anti-aligned → negative raw s*
    assert r1_eval.scale_solution(records, weight="all") == 0.0


def test_apply_scale_is_linear_and_non_mutating():
    records = _synthetic_records()
    originals = [r["u"].copy() for r in records]
    s = 1.37
    scaled = r1_eval.apply_scale(records, s)
    for rec, orig, sc in zip(records, originals, scaled):
        np.testing.assert_allclose(sc["u"], s * orig, rtol=1e-12)
        np.testing.assert_array_equal(rec["u"], orig)  # inputs untouched
        assert sc is not rec
    # composition: apply(apply(r, a), b) == apply(r, a*b)
    twice = r1_eval.apply_scale(r1_eval.apply_scale(records, 2.0), 3.0)
    once = r1_eval.apply_scale(records, 6.0)
    for t, o in zip(twice, once):
        np.testing.assert_allclose(t["u"], o["u"], rtol=1e-12)


def test_pair_bucket_edge_policy():
    """Frozen protocol edges are half-open [lo, hi): a pair exactly on an
    edge belongs to the STRONGER bucket."""
    assert r1_eval.pair_bucket(0.0) == "dead"
    assert r1_eval.pair_bucket(0.0049999) == "dead"
    assert r1_eval.pair_bucket(0.005) == "weak"      # edge → stronger
    assert r1_eval.pair_bucket(0.0099) == "weak"
    assert r1_eval.pair_bucket(0.01) == "medium"     # edge → stronger
    assert r1_eval.pair_bucket(0.01999) == "medium"
    assert r1_eval.pair_bucket(0.02) == "strong"     # edge → stronger
    assert r1_eval.pair_bucket(0.2) == "strong"


def _pair_record(dd_true: float, dd_pred: float) -> dict:
    """Single-pair record with exact Δd_true / Δd_pred along x.

    pos_i = (0,0,0), pos_j = (1+dd, 0, 0); a displacement u_j = (d, 0, 0)
    yields Δd = ‖(1+d,0,0)‖ − 1 exactly.
    """
    pos = np.array([[0.0, 0.0, 0.0], [1.0 + dd_true, 0.0, 0.0],
                    [5.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
    # teacher displacement realizes dd_true on pair (0,1)
    y = np.array([[0.0, 0.0, 0.0], [dd_true, 0.0, 0.0],
                  [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    # prediction displacement realizes dd_pred
    u = np.array([[0.0, 0.0, 0.0], [dd_pred, 0.0, 0.0],
                  [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    return {
        "cluster": "RXN_SYNTHPAIR", "reaction_id": "RXN_SYNTHPAIR",
        "branch": 0, "near_ts": 0.1,
        "pos": pos, "key_rows": [0, 1, 2, 3],
        "pairs": [(0, 1), (2, 3)],
        "pair_edit": [1.0, 0.0],   # second pair filtered (pair_edit == 0)
        "u": u, "y": y,
    }


def test_strong_pair_eval_abstention_and_strong_bucket():
    # strong pair the model failed to move at all → abstention = incorrect
    rec = _pair_record(dd_true=0.03, dd_pred=0.0)
    out = r1_eval.strong_pair_eval([rec])
    assert out["n_pairs_total"] == 1
    strong = out["per_bucket"]["strong"]
    assert strong["n_pairs"] == 1
    assert strong["sign_accuracy"] == 0.0          # abstention incorrect
    assert strong["sign_accuracy_where_predicted"] is None
    assert strong["abstention_rate"] == 1.0
    # strong pair with correct sign and full amplitude
    rec2 = _pair_record(dd_true=0.03, dd_pred=0.028)
    strong2 = r1_eval.strong_pair_eval([rec2])["per_bucket"]["strong"]
    assert strong2["sign_accuracy"] == 1.0
    assert strong2["sign_accuracy_where_predicted"] == 1.0
    assert strong2["abstention_rate"] == 0.0
    assert strong2["amplitude_ratio_mean"] == pytest.approx(0.028 / 0.03,
                                                            rel=1e-9)
    assert strong2["residual_median"] == pytest.approx(0.002, rel=1e-9)
    # dead bucket assignment for |dd_true| < 0.005
    rec3 = _pair_record(dd_true=0.002, dd_pred=0.02)
    out3 = r1_eval.strong_pair_eval([rec3])
    assert out3["per_bucket"]["dead"]["n_pairs"] == 1
    assert out3["per_bucket"]["strong"]["n_pairs"] == 0


def test_cluster_paired_bootstrap_reproducible():
    rng = np.random.default_rng(11)
    base = _synthetic_records(rng_seed=3, n_clusters=8, n_per_cluster=4)
    # build a "calibrated" set with a known per-cluster shift in u
    cal = []
    for r in base:
        rr = dict(r)
        rr["u"] = r["u"] * (0.9 + 0.02 * int(r["cluster"][-1]))
        cal.append(rr)

    def rmse_metric(records):
        return r1_eval.canonical_metrics(records)["aggregate"]["rmse_over_zero"]

    out1 = r1_eval.cluster_paired_bootstrap(cal, base, rmse_metric,
                                            draws=200, seed=20261018)
    out2 = r1_eval.cluster_paired_bootstrap(cal, base, rmse_metric,
                                            draws=200, seed=20261018)
    assert out1["median_delta"] == out2["median_delta"]
    assert out1["ci_lo_2_5"] == out2["ci_lo_2_5"]
    assert out1["ci_hi_97_5"] == out2["ci_hi_97_5"]
    assert out1["n_clusters"] == 8
    assert sorted(out1["per_cluster_deltas"]) == sorted(
        {r["cluster"] for r in base})
    # point estimate equals the plain median-of-clusters delta computed
    # directly from the per-cluster metric values
    a_vals = [r1_eval.canonical_metrics(
        [r for r in cal if r["cluster"] == c]
    )["aggregate"]["rmse_over_zero"] for c in out1["cluster_ids"]]
    b_vals = [r1_eval.canonical_metrics(
        [r for r in base if r["cluster"] == c]
    )["aggregate"]["rmse_over_zero"] for c in out1["cluster_ids"]]
    assert out1["point_delta"] == pytest.approx(
        float(np.median(a_vals) - np.median(b_vals)), rel=1e-12)
    # CI brackets the median delta for this smooth synthetic shift
    assert out1["ci_lo_2_5"] <= out1["median_delta"] <= out1["ci_hi_97_5"]
    # a different seed must (overwhelmingly) change the replay distribution
    out3 = r1_eval.cluster_paired_bootstrap(cal, base, rmse_metric,
                                            draws=200, seed=999)
    assert out3["median_delta"] != out1["median_delta"] or \
        out3["ci_lo_2_5"] != out1["ci_lo_2_5"] or \
        out3["ci_hi_97_5"] != out1["ci_hi_97_5"]


def test_canonical_metrics_shape_synthetic():
    records = _synthetic_records()
    out = r1_eval.canonical_metrics(records)
    agg = out["aggregate"]
    for key in ("rmse_over_zero", "alpha_median", "alpha_q25", "alpha_q75",
                "cos_eff_median", "cos_eff_near_ts_median",
                "sre_cluster_median", "sign_accuracy_cluster_median",
                "dead_zone_fraction", "n_clusters", "n_sign_pairs",
                "n_active_pairs", "n_dead_pairs", "n_sre_frames"):
        assert key in agg, key
    assert agg["n_clusters"] == 4
    assert sorted(out["per_cluster"]) == sorted(
        {r["cluster"] for r in records})
    # scaling u by s: alpha_median scales by exactly s (norm ratio);
    # SRE is invariant up to the exact-Δd nonlinearity (second-order in
    # the perpendicular noise ~1e-5); sign accuracy is exactly invariant
    # (signs and the |dd| >= threshold status are unchanged for s > 0)
    s = 1.4
    scaled = r1_eval.canonical_metrics(r1_eval.apply_scale(records, s))
    assert scaled["aggregate"]["alpha_median"] == pytest.approx(
        s * agg["alpha_median"], rel=1e-9)
    assert scaled["aggregate"]["sre_cluster_median"] == pytest.approx(
        agg["sre_cluster_median"], abs=2e-4)
    assert scaled["aggregate"]["sign_accuracy_cluster_median"] == pytest.approx(
        agg["sign_accuracy_cluster_median"], rel=1e-9)
