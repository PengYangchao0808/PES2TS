"""G2-AB1 WP-6: PES-G algebraic contracts and counterexamples as repo tests.

These fixtures freeze the boundary discipline documented in the PES-G
tightening plan (§16 audit): every tool here has explicit standing conditions
and explicit claims it CANNOT make.  Numerical baselines are read from
``outputs/g2_pesg_design_audit_20261003/math_checks.json``; the assertions
recompute everything independently and require agreement.

Standing conditions / non-claims:
- the W-projector identities hold for full-row-rank A with the mass metric W;
  they are NOT a convergence theorem for any real PES;
- the damped (ridge) inverse only APPROXIMATES projection — its constraint
  leakage must be reported, never hidden;
- the normalized two-vertex spectrum shows a normalized graph Laplacian
  cannot report uniform bond weakening — it is not a bond-order measurement;
- the multi-driver counterexample shows a strict scan maximum need NOT be a
  stationary point of the full PES;
- the Lagrangian curvature example shows the physical Hessian alone cannot
  decide constrained curvature sign — a minimum claim needs H_L (and here only
  one direction is ever probed, never a full spectrum).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

BASELINE_PATH = (Path(__file__).resolve().parents[1]
                 / "outputs/g2_pesg_design_audit_20261003/math_checks.json")


@pytest.fixture(scope="module")
def baseline() -> dict:
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def mass_metric_projector(W, A):
    """Oblique projector onto ker(A) in the W metric (full row rank A only)."""
    Wi = np.linalg.inv(W)
    return np.eye(W.shape[0])-Wi@A.T@np.linalg.pinv(A@Wi@A.T)@A


def test_w_projector_idempotent_constraint_null_and_metric_adjoint(baseline):
    W = np.diag([1., 12., 16.])
    A = np.array([[1., -1., 0.]])
    P = mass_metric_projector(W, A)
    idempotency_error = float(np.linalg.norm(P@P-P))
    assert idempotency_error < 1e-12
    assert idempotency_error <= baseline["projector_idempotency_error"] + 1e-15
    assert np.linalg.norm(A@P) < 1e-12
    assert np.linalg.norm(P.T@W-W@P) < 1e-12
    assert baseline["projector_constraint_error"] < 1e-12
    assert baseline["projector_metric_adjoint_error"] < 1e-12


def test_gradient_duality_vector_and_covector_projections_differ(baseline):
    W = np.diag([1., 12., 16.])
    A = np.array([[1., -1., 0.]])
    Wi = np.linalg.inv(W)
    P = mass_metric_projector(W, A)
    g = np.array([2., -1., 3.])
    grad_free = P@Wi@g
    assert np.linalg.norm(P.T@g-W@grad_free) < 1e-12
    assert baseline["gradient_duality_error"] < 1e-12


def test_damped_inverse_leaks_constraints_and_the_leak_is_reported(baseline):
    W = np.diag([1., 12., 16.])
    Wi = np.linalg.inv(W)
    A = np.array([[1., -1., 0.]])
    P_eps = np.eye(3)-Wi@A.T@np.linalg.inv(A@Wi@A.T+.1*np.eye(1))@A
    leak = float(np.linalg.norm(A@P_eps))
    idempotency_error = float(np.linalg.norm(P_eps@P_eps-P_eps))
    assert leak > 1e-3
    assert idempotency_error > 1e-3
    assert baseline["ridge_projector_constraint_leak"] > 1e-3
    assert baseline["ridge_projector_idempotency_error"] > 1e-3


def test_equal_mass_penalty_radius_H_over_C_is_sqrt12(baseline):
    # In the equal-mass penalty metric an H and a C may move by the same
    # norm; with true masses (12:1) the allowed physical displacements ratio
    # is sqrt(12) — an equal-mass penalty is NOT physically neutral.
    assert baseline["equal_mass_penalty_radius_H_over_C"] == pytest.approx(np.sqrt(12.))
    assert np.sqrt(12.) == pytest.approx(2.*np.sqrt(3.))


@pytest.mark.parametrize("weight_index", [0, 1, 2])
def test_normalized_two_vertex_spectrum_is_scale_invariant(baseline, weight_index):
    row = baseline["normalized_laplacian_two_vertex"][weight_index]
    weight = row["weight"]
    adjacency = np.array([[0., weight], [weight, 0.]])
    Dinv = np.diag(1/np.sqrt(adjacency.sum(axis=1)))
    eigenvalues = np.linalg.eigvalsh(np.eye(2)-Dinv@adjacency@Dinv)
    assert eigenvalues.tolist() == pytest.approx(row["eigenvalues"])
    assert eigenvalues[0] == pytest.approx(0., abs=1e-12)
    # The top eigenvalue stays 2 whether the edge weight is 1 or 1e-8: a
    # normalized Laplacian cannot report uniform bond weakening.
    assert eigenvalues[-1] == pytest.approx(2., abs=1e-12)


def test_multi_constraint_scan_peak_is_not_a_stationary_point(baseline):
    def model_energy(x):
        return x[0]-x[1]-(x[0]+x[1])**2+x[2]**2

    step = 1e-4
    model_gradient = np.array([(model_energy(step*e)-model_energy(-step*e))/(2*step)
                               for e in np.eye(3)])
    schedule_tangent = np.array([1., 1., 0.])
    curvature = (model_energy(step*schedule_tangent)
                 + model_energy(-step*schedule_tangent)-2*model_energy(np.zeros(3)))/step**2
    assert model_gradient.tolist() == pytest.approx(baseline["multiple_driver_peak"]["gradient"])
    assert model_gradient@schedule_tangent == pytest.approx(0., abs=1e-12)
    assert curvature == pytest.approx(baseline["multiple_driver_peak"]["energy_second_derivative"])
    assert curvature == pytest.approx(-8., abs=1e-10)
    # Zero slope along the schedule AND a strict maximum there, yet the full
    # gradient is nonzero: the scan peak is NOT a stationary point.
    assert np.linalg.norm(model_gradient) > 0
    assert baseline["multiple_driver_peak"]["is_stationary"] is False


def test_lagrangian_curvature_is_necessary_not_the_physical_hessian(baseline):
    # E(x,y)=x^2/2 on the unit circle at (1,0): the physical Hessian alone
    # gives 0 along the tangent; the constrained second derivative is -1 and
    # is recovered ONLY through H_L = H_E + mu*H_constraint.
    H_E = np.diag([1., 0.])
    H_constraint = 2*np.eye(2)
    mu = -.5
    v = np.array([0., 1.])
    physical = float(v@H_E@v)
    constrained = float(v@(H_E+mu*H_constraint)@v)
    assert physical == pytest.approx(baseline["lagrangian_curvature"]["physical_hessian_only"])
    assert constrained == pytest.approx(baseline["lagrangian_curvature"]["constrained_curvature"])
    assert physical == 0. and constrained == -1.


def test_bordered_kkt_predictor_derivative(baseline):
    # E=.5*(z-2q)^2 under q=lambda: the regular bordered system reproduces
    # dq/dlambda=1, dz/dlambda=2 — the predictor derivative, not a path.
    H = np.array([[4., -2.], [-2., 1.]])
    J = np.array([[1., 0.]])
    K = np.block([[H, J.T], [J, np.zeros((1, 1))]])
    solution = np.linalg.solve(K, [0., 0., 1.])
    assert solution.tolist() == pytest.approx(baseline["kkt_predictor_derivative"])
    assert solution.tolist() == pytest.approx([1., 2., 0.])


def test_baseline_declares_its_own_limits(baseline):
    assert baseline["scope"] == ("Algebraic consistency and counterexamples only; "
                                 "no convergence theorem or chemical benchmark.")
