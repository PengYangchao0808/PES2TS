from pathlib import Path
import numpy as np

from pes2ts_core.generation.planning.gradient_evidence import BOHR_ANGSTROM, force_evidence, read_bound_engrad


def test_gradient_binding_rejects_wrong_energy_and_atom_identity(tmp_path):
    path = tmp_path/"orca.engrad"
    path.write_text("2\n-1.0\n0.1\n0\n0\n-0.1\n0\n0\n1 0 0 0\n1 2 0 0\n", encoding="utf-8")
    x = [[0., 0., 0.], [2*BOHR_ANGSTROM, 0., 0.]]
    assert read_bound_engrad(path, x, ["H", "H"], -1.)["status"] == "bound"
    assert read_bound_engrad(path, x, ["H", "H"], -2.)["reason"] == "GRADIENT_FRAME_MISMATCH"
    assert read_bound_engrad(path, x, ["H", "C"], -1.)["reason"] == "INVALID_GRADIENT_IDENTITY"


def test_projected_gradient_zero_does_not_mean_stationary_point():
    d = {"kind": "distance", "atoms": [0, 1], "lambda_values": [0., 1.], "values": [1., 2.]}
    plan = {"drivers": [d], "guards": []}
    frame = {"geometry": [[0., 0., 0.], [1., 0., 0.]], "lambda": .5}
    bound = {"status": "bound", "gradient_hartree_per_bohr": [[-.1, 0, 0], [.1, 0, 0]]}
    evidence = force_evidence(plan, frame, bound)
    assert evidence["physical_gradient_norm_hartree_per_bohr"]>.1
    assert evidence["perpendicular_gradient_norm_hartree_per_bohr"]<1e-10
    assert evidence["reconstructed_energy_slope_hartree_per_lambda"]>0
    assert not evidence["stationary_point_verified"]
