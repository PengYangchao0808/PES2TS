"""Unit tests for the WP3 v4.1 diagnostics pure helpers (synthetic, fast).

Covers the four registered pure-math/bookkeeping surfaces:

  1. projection math       (``project_residual`` / ``orthonormal_basis``)
  2. per-sample identity   (``per_sample_identity`` + group back-inference)
  3. classifier rule       (``load_inserted_state_ids`` provenance parsing)
  4. boundary bookkeeping  (``atom_alpha``/``atom_cos``/``mse_ratio_vs_zero``/
                            ``dir_cosine_weighted``)

No real data, no model loading, no GPU, no QC.  These tests import only the
pure numpy helpers from ``scripts/diagnose_v41_common.py``.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from diagnose_v41_common import (  # noqa: E402
    atom_alpha,
    atom_cos,
    atom_mse,
    dir_cosine_weighted,
    load_inserted_state_ids,
    mse_ratio_vs_zero,
    orthonormal_basis,
    per_sample_identity,
    project_residual,
)


# ---------------------------------------------------------------------------
# 1. Projection math
# ---------------------------------------------------------------------------

def test_orthonormal_basis_orthonormal():
    rng = np.random.default_rng(0)
    for _ in range(200):
        k = int(rng.integers(1, 9))
        vs = rng.normal(size=(k, 3))
        basis = orthonormal_basis(list(vs))
        assert basis.ndim == 2 and basis.shape[1] == 3
        assert basis.shape[0] <= 3
        if basis.size:
            gram = basis @ basis.T
            assert np.allclose(gram, np.eye(basis.shape[0]), atol=1e-10)


def test_project_residual_self_span_is_zero():
    d = np.array([1.0, 2.0, -1.0])
    d = d / np.linalg.norm(d)
    r = project_residual(d, [d])
    assert abs(r["residual"]) < 1e-12
    assert abs(r["proj_norm"] - 1.0) < 1e-12


def test_project_residual_orthogonal_is_one():
    r = project_residual(np.array([0.0, 1.0, 0.0]),
                         [np.array([1.0, 0.0, 0.0])])
    assert abs(r["residual"] - 1.0) < 1e-12
    assert abs(r["proj_norm"]) < 1e-12


def test_project_residual_empty_span_is_one():
    r = project_residual(np.array([1.0, 0.0, 0.0]), [])
    assert r["residual"] == 1.0
    assert r["proj_norm"] == 0.0


def test_project_residual_norm_bounded():
    rng = np.random.default_rng(1)
    for _ in range(5000):
        k = int(rng.integers(1, 9))
        vs = rng.normal(size=(k, 3))
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        r = project_residual(d, list(vs))
        assert 0.0 - 1e-9 <= r["proj_norm"] <= 1.0 + 1e-9
        assert 0.0 - 1e-9 <= r["residual"] <= 1.0 + 1e-9


def test_project_residual_degenerate_vectors():
    # duplicate / near-collinear vectors must not break the projection
    d = np.array([1.0, 0.0, 0.0])
    vs = [np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]),
          np.array([1e-9, 0.0, 0.0])]
    r = project_residual(d, vs)
    assert abs(r["residual"]) < 1e-9


# ---------------------------------------------------------------------------
# 2. Per-sample identity + group back-inference
# ---------------------------------------------------------------------------

def test_per_sample_identity_holds_exactly():
    rng = np.random.default_rng(2)
    a_star = rng.normal(size=(200, 3))
    a_star[np.linalg.norm(a_star, axis=1) < 1e-3] = 0.05  # keep norms > 0
    a_det = rng.normal(size=(200, 3))
    dmask = np.ones(200, dtype=bool)
    resid, alpha, cos = per_sample_identity(a_det, a_star, dmask)
    assert np.abs(resid).max() < 1e-12
    # alpha and cos are finite and alpha >= 0
    assert np.isfinite(alpha).all() and (alpha >= 0).all()


def test_group_back_inference_disagrees_with_direct_alpha():
    """Group-level MSE ratio + median cos cannot recover median alpha.

    Unequal ||a*|| magnitudes make the norm-weighted group MSE ratio fail to
    commute with the UNWEIGHTED median cosine, so the per-sample identity,
    which holds per atom, does NOT hold after aggregation.
    """
    a_star = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    a_det = np.array([[0.5, 0.0, 0.0], [0.0, 0.0, 1.5]])
    alpha = atom_alpha(a_det, a_star)
    cos = atom_cos(a_det, a_star)
    # per-atom identity holds
    r_norm = atom_mse(a_det, a_star) / (np.linalg.norm(a_star, axis=1) ** 2)
    recon = 1.0 + alpha ** 2 - 2.0 * alpha * cos
    assert np.allclose(r_norm, recon, atol=1e-12)

    # group aggregates
    r_group = float(atom_mse(a_det, a_star).mean()) / float(
        (a_star ** 2).sum(-1).mean())
    c_group = float(np.median(cos))
    alpha_true = float(np.median(alpha))
    disc = c_group ** 2 + r_group - 1.0
    assert disc >= 0  # has a real root in this synthetic case
    alpha_hat = c_group + math.sqrt(disc)
    # back-inferred alpha differs from the directly-measured median alpha
    assert abs(alpha_hat - alpha_true) > 0.1


# ---------------------------------------------------------------------------
# 3. Classifier rule (inserted vs original provenance)
# ---------------------------------------------------------------------------

def test_load_inserted_state_ids_rule(tmp_path):
    reg = {
        "schema_version": "pes2ts_dual_ts_gaps_registry_v1",
        "reactions": [
            {"reaction_id": "R", "items": {"inserted_nodes": {
                "A": [{"inserted_state_id": "R|A|0001",
                       "between_original_steps": [0, 1],
                       "trigger_max_abs_per_atom_angstrom": 0.12,
                       "kind": "midpoint_linear", "branch": "A"}],
                "B": []}}},
            {"reaction_id": "S", "items": {"inserted_nodes": {
                "A": [], "B": [{"inserted_state_id": "S|B|0003",
                                "between_original_steps": [2, 3],
                                "trigger_max_abs_per_atom_angstrom": 0.11,
                                "kind": "midpoint_linear", "branch": "B"}]}}},
        ],
    }
    p = tmp_path / "gaps_registry.json"
    p.write_text(json.dumps(reg))
    ids = load_inserted_state_ids(p)
    assert ids == {"R|A|0001", "S|B|0003"}


def test_load_inserted_state_ids_empty(tmp_path):
    reg = {"schema_version": "x", "reactions": [
        {"reaction_id": "R", "items": {"inserted_nodes": {"A": [], "B": []}}}]}
    p = tmp_path / "gaps_registry.json"
    p.write_text(json.dumps(reg))
    assert load_inserted_state_ids(p) == set()


# ---------------------------------------------------------------------------
# 4. Boundary bookkeeping / metric primitives
# ---------------------------------------------------------------------------

def test_atom_metric_primitives():
    a_star = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    a_det = np.array([[0.5, 0.0, 0.0], [0.0, 1.0, 0.0]])
    alpha = atom_alpha(a_det, a_star)
    assert np.allclose(alpha, [0.5, 0.5])
    cos = atom_cos(a_det, a_star)
    assert np.allclose(cos, [1.0, 1.0])
    mse = atom_mse(a_det, a_star)
    assert np.allclose(mse, [0.25, 1.0])


def test_mse_ratio_vs_zero_baseline():
    a_star = np.array([[1.0, 0.0, 0.0]])
    a_det = np.array([[0.5, 0.0, 0.0]])
    mask = np.ones(1, dtype=bool)
    # model MSE 0.25, zero baseline MSE 1.0 -> ratio 0.25
    assert abs(mse_ratio_vs_zero(a_det, a_star, mask) - 0.25) < 1e-12


def test_dir_cosine_weighted():
    a_star = np.array([[1.0, 0.0, 0.0], [0.0, 0.1, 0.0]])
    a_det = np.array([[1.0, 0.0, 0.0], [0.0, -0.1, 0.0]])
    dmask = np.ones(2, dtype=bool)
    # |a|-weighted cosine = (1*1 + 0.1*(-1)) / (1 + 0.1) = 0.9/1.1
    assert abs(dir_cosine_weighted(a_det, a_star, dmask) - 0.9 / 1.1) < 1e-9


def test_dir_cosine_weighted_empty_mask_is_none():
    a_star = np.array([[1.0, 0.0, 0.0]])
    a_det = np.array([[0.5, 0.0, 0.0]])
    dmask = np.zeros(1, dtype=bool)
    assert dir_cosine_weighted(a_det, a_star, dmask) is None
