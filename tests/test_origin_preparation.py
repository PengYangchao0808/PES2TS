import numpy as np
from pes2ts_core.generation.planning.origin_preparation import repair_component_overlap,torsion_hypotheses


def test_component_repair_preserves_each_component_internal_coordinates():
    x=np.array([[0.,0.,0.],[1.,0.,0.],[.3,0.,0.]])
    y,evidence=repair_component_overlap(x,["C","H","Cl"],[[0,1],[2]])
    assert evidence["internal_geometry_preserved"]
    assert np.allclose(y[1]-y[0],x[1]-x[0])
    assert np.linalg.norm(y[0]-y[2])>2.


def test_torsion_hypothesis_preserves_bonds_and_does_not_rotate_ring():
    x=np.array([[0.,1.,0.],[0.,0.,0.],[1.,0.,0.],[1.,1.,0.]])
    bonds=[[0,1],[1,2],[2,3]]
    rows=torsion_hypotheses(x,bonds)
    assert len(rows)==2
    for y,record in rows:
        assert all(np.isclose(np.linalg.norm(y[a]-y[b]),np.linalg.norm(x[a]-x[b])) for a,b in bonds)
    assert not torsion_hypotheses(x,bonds+[[0,3]])
