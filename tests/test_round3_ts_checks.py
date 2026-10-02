import numpy as np
from pes2ts_core.generation.planning.ts_checks import assess_mode, mapped_endpoint_identity
from pes2ts_core.generation.planning.constrained_curvature import directional_curvature


def test_one_imaginary_frequency_without_reaction_motion_is_not_target_ts():
    x = np.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.]])
    modes = {6:np.array([[0.,0.,0.], [0.,0.,0.], [0.,0.,1.]])}
    record = assess_mode(x, [{"kind":"distance","atoms":[0,1]}], {6:-300.,7:150.}, modes)
    assert record["single_meaningful_imaginary"]
    assert not record["target_mode_screen_passed"]


def test_active_mode_is_only_a_screen_not_irc_proof():
    x = np.array([[0.,0.,0.], [1.,0.,0.]])
    record = assess_mode(x,[{"kind":"distance","atoms":[0,1]}], {5:-300.},
                         {5:np.array([[-1.,0.,0.],[1.,0.,0.]])})
    assert record["target_mode_screen_passed"] and not record["irc_connection_verified"]


def test_endpoint_identity_checks_full_graph_and_explicit_atom_mapping():
    snapshot = {"elements":["O","H","H"], "maps":[1,2,3],
                "reaction_smiles":"[O:1]([H:2])[H:3]>>[O:1]([H:2])[H:3]",
                "endpoint_electronic":{"reactant":{"charge":0,"multiplicity":1}}}
    x = [[0.,0.,0.], [.96,0.,0.], [-.24,.93,0.]]
    assert mapped_endpoint_identity(x,snapshot,"R")["matched"]
    x[2] = [5.,5.,5.]
    assert not mapped_endpoint_identity(x,snapshot,"R")["matched"]


def test_curvature_includes_nonlinear_constraint_contribution():
    # Bond energy alone has transverse curvature; on a fixed-distance sphere
    # its Lagrangian curvature vanishes. A third atom supplies a free mode.
    x = np.array([[0.,0.,0.], [1.,0.,0.], [.2,1.,.4]])
    coords = [{"kind":"distance","atoms":[0,1]}]
    def evaluate(y, name):
        v = y[1]-y[0]
        g = np.zeros_like(y); g[1]=v; g[0]=-v
        return {"success":True,"energy":.5*np.sum(v*v),"gradient_hartree_per_angstrom":g}
    direction = np.zeros_like(x); direction[1,1]=1.
    record = directional_curvature(x,coords,evaluate(x,"")["gradient_hartree_per_angstrom"],direction,evaluate)
    assert all(abs(r["curvature_hartree_per_angstrom2"]) < 1e-5 for r in record["measurements"])
    assert not record["positive_definite_hessian_verified"]
