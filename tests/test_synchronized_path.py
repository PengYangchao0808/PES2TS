"""Tests for the scientific failures that a single-coordinate scan missed."""
import copy
import numpy as np
import pytest
from pes2ts_core.generation.planning.synchronized_path import validate_plan,audit_path,residual,coordinate_rank,predict_geometry,value,mass_weighted_lambda


def example():
    r=[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]
    p=[[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]]
    return {'parameter_dimension':1,'elements':['H']*3,'atom_map_order':[1,2,3],
            'start_geometry':r,'target_geometry':p,'lambda_values':[0,.2,1],
            'drivers':[{'id':'a','kind':'distance','atoms':[0,1],'values':[1,1.3,2]},
                       {'id':'b','kind':'distance','atoms':[0,2],'values':[1,1.7,2]}],
            'event_coverage':[{'driver_ids':['a']},{'driver_ids':['b']}],
            'quality_policy':{'coordinate_tolerances':{'distance':.01,'angle':.5,'dihedral':.5},
                              'max_step_rmsd_angstrom':.3}}


def test_independent_drivers_share_one_nonuniform_parameter():
    p=example(); validate_plan(p)
    assert coordinate_rank(p['start_geometry'],p['drivers'])['independent']


def test_missing_second_transfer_bond_rejected():
    p=example(); p['drivers'].pop()
    with pytest.raises(ValueError,match='no active driver'): validate_plan(p)


def test_endpoints_are_whole_geometries_not_only_lengths():
    p=example(); middle=np.asarray(p['start_geometry'])
    frames=np.asarray([p['start_geometry'],middle,p['target_geometry']])
    frames[-1]+=np.array([5,0,0])
    assert not audit_path(p,frames)['boundary_ok']


def test_appending_perfect_endpoint_does_not_pass():
    p=example(); p['target_geometry'][1][0]=9
    p['drivers'][0]['values'][-1]=9
    f=[p['start_geometry'],p['start_geometry'],p['target_geometry']]
    a=audit_path(p,f)
    assert a['boundary_ok'] and not a['continuity_ok'] and not a['qualified']


@pytest.mark.parametrize('values',[[1,float('nan'),2],[1,2],[1,1.5,3]])
def test_invalid_tables_rejected(values):
    p=example(); p['drivers'][0]['values']=values
    with pytest.raises(ValueError): validate_plan(p)


def test_dihedral_error_wraps_across_branch_cut():
    assert residual(-179,179,'dihedral')==2


def test_duplicate_ids_rejected():
    p=example(); p['drivers'][1]['id']='a'
    with pytest.raises(ValueError,match='duplicate'):validate_plan(p)


def test_predictor_changes_both_drivers_and_preserves_translation():
    p=example();r=np.asarray(p['start_geometry']);x=predict_geometry(r,p['drivers'],[1.01,1.02])
    assert abs(value(x,'distance',[0,1])-1.01)<.001
    assert abs(value(x,'distance',[0,2])-1.02)<.001
    assert np.allclose(x.mean(0),r.mean(0))


def test_mass_arclength_is_shared_nonuniform_and_rigid_motion_invariant():
    p=example();r=np.asarray(p['start_geometry']);f=np.asarray([r,1.1*r,2*r])
    lam=mass_weighted_lambda(f,[1,2,3])
    assert lam==pytest.approx([0,.1,1])
    f[1]+=8
    assert mass_weighted_lambda(f,[1,2,3])==pytest.approx(lam)


def test_joint_table_cannot_claim_an_incompatible_geometric_witness():
    p=example();p['reference_geometries']=[p['start_geometry'],p['start_geometry'],p['target_geometry']]
    with pytest.raises(ValueError,match='geometric witness'):validate_plan(p)
