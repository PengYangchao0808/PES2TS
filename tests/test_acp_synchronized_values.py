"""Contract tests against the installed ACP checkout; optional outside that environment."""
import pytest
constraints=pytest.importorskip('cccp.qc.interfaces.constraints')
contracts=pytest.importorskip('acp.calculations.pes.contracts')


def test_arbitrary_values_survive_roundtrip_and_reach_every_constraint():
    coords=[{'kind':'distance','atoms':[i,i+1],'start':1.,'end':2.,'n_points':3,
             'values':[1.,1.2+i*.01,2.]} for i in range(5)]
    parsed=[contracts.ScanCoordinate.from_dict(c) for c in coords]
    contracts.validate_scan_coordinates(parsed)
    assert [c.to_dict()['values'] for c in parsed]==[c['values'] for c in coords]
    specs=tuple(constraints.CoordinateSpec(id=str(i),kind=c.kind,atoms=c.atoms,
              start=c.start,end=c.end,values=c.values) for i,c in enumerate(parsed))
    plan=constraints.ReactionCoordinatePlan(specs,points=3,lambda_values=(0,.2,1))
    assert [c.target for c in plan.frame_constraints(1)]==[c['values'][1] for c in coords]
    assert len(constraints.orca_constraint_block(plan.frame_constraints(1)).splitlines())==7


def test_mixed_b_a_d_targets_not_linearized():
    specs=(constraints.CoordinateSpec('B','distance',(0,1),start=1,end=2,values=(1,1.7,2)),
           constraints.CoordinateSpec('A','angle',(0,1,2),start=90,end=120,values=(90,100,120)),
           constraints.CoordinateSpec('D','dihedral',(0,1,2,3),start=170,end=-170,values=(170,180,-170)))
    plan=constraints.ReactionCoordinatePlan(specs,points=3)
    block=constraints.orca_constraint_block(plan.frame_constraints(1))
    assert '{ B 0 1 1.70000000 C }' in block
    assert '{ A 0 1 2 100.00000000 C }' in block
    assert '{ D 0 1 2 3 180.00000000 C }' in block


def test_invalid_lambda_and_coordinate_lengths_fail():
    c=constraints.CoordinateSpec('B','distance',(0,1),start=1,end=2,values=(1,2))
    with pytest.raises(ValueError):constraints.ReactionCoordinatePlan((c,),points=3)
    c=constraints.CoordinateSpec('B','distance',(0,1),start=1,end=2)
    with pytest.raises(ValueError):constraints.ReactionCoordinatePlan((c,),points=3,lambda_values=(0,.5,.4))


def test_near_equal_endpoint_nonmonotone_coordinate_is_not_rejected_as_zero_scan():
    c=contracts.ScanCoordinate.from_dict({'kind':'distance','atoms':[0,1],
        'start':1.3,'end':1.3,'n_points':3,'values':[1.3,1.301,1.3]})
    contracts.validate_scan_coordinates([c])


def test_fixed_boundary_api_role_never_claims_optimization_convergence():
    from acp.api.v1_schemas import S2FrameModel
    frame=contracts.ScanFrame(index=0,target_coordinate=1,actual_coordinate=1,
        scan_energy_hartree=-1,optimization_converged=False,frame_role='fixed_boundary_single_point',scf_converged=True)
    projected=S2FrameModel(**frame.to_dict())
    assert projected.frame_role=='fixed_boundary_single_point'
    assert not projected.optimization_converged


def test_explicit_multicoordinate_plan_does_not_route_to_two_bond_selector():
    import numpy as np
    from acp.calculations.pes.scan import _validate_functional_selection,_interpolated_coordinate_target
    coords=tuple(contracts.ScanCoordinate.from_dict({'kind':'distance','atoms':[i,i+1],
        'start':1,'end':2,'n_points':3,'values':[1,1.01,2]}) for i in range(5))
    payload={'path_plan':{'fixed_endpoints':True}}
    assert _validate_functional_selection(payload,coords,['C']*6,np.zeros((6,3)))==payload
    assert _interpolated_coordinate_target(coords[0],1,3)==1.01


def test_single_explicit_driver_retains_full_plan_through_backend():
    import numpy as np
    from unittest.mock import MagicMock
    from acp.backends.orca import ORCABackend
    c=constraints.CoordinateSpec('B','distance',(0,1),start=1,end=2,values=(1,1.2,2))
    plan=constraints.ReactionCoordinatePlan((c,),points=3,fixed_endpoints=True,
        reference_geometries=(((0,0,0),(1,0,0)),((0,0,0),(1.2,0,0)),((0,0,0),(2,0,0))))
    backend=object.__new__(ORCABackend);backend._interface=MagicMock()
    backend.relaxed_scan(np.zeros((2,3)),['H','H'],output_dir='unused',plan=plan)
    assert backend._interface.relaxed_scan.call_args.kwargs['plan'] is plan


def test_geometry_without_finite_energy_is_not_a_successful_scan_point(tmp_path):
    import numpy as np
    from unittest.mock import patch
    from cccp.config import load_config
    from cccp.qc.interfaces.orca import ORCAInterface
    from cccp.qc.interfaces.base import QCResult
    interface=ORCAInterface(load_config())
    c=constraints.CoordinateSpec('B','distance',(0,1),start=1,end=2,values=(1,1.2,2))
    plan=constraints.ReactionCoordinatePlan((c,),points=3)
    result_stub=QCResult(success=True,energy=None,coordinates=np.array([[0.,0.,0.],[1.,0.,0.]]),symbols=['H','H'])
    with patch.object(interface,'constrained_optimize',return_value=result_stub):
        result=interface.relaxed_scan(result_stub.coordinates,['H','H'],plan=plan,output_dir=tmp_path,method='GFN2-xTB')
    assert not result.success
    assert not result.points[0].success
