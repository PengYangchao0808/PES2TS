"""Truth-free contracts and quality gates for a full-boundary 1-D path.

Reference geometry acquisition belongs to an explicitly labelled caller, not
this module. A path may have any number of internal-coordinate drivers.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def align(moving, reference):
    """Proper common rigid alignment; never move components separately."""
    a, b = np.asarray(moving, float), np.asarray(reference, float)
    ac, bc = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a-ac).T @ (b-bc))
    correction = np.diag([1., 1., np.linalg.det(u @ vt)])
    return (a-ac) @ u @ correction @ vt + bc


def value(x, kind, atoms):
    x = np.asarray(x, float)[list(atoms)]
    if kind == 'distance':
        return float(np.linalg.norm(x[0]-x[1]))
    if kind == 'angle':
        a, b = x[0]-x[1], x[2]-x[1]
        return float(np.degrees(np.arccos(np.clip(a@b/np.linalg.norm(a)/np.linalg.norm(b), -1, 1))))
    if kind != 'dihedral':
        raise ValueError('unsupported coordinate kind')
    b = x[2]-x[1]; b /= np.linalg.norm(b)
    v, w = x[0]-x[1], x[3]-x[2]
    v -= (v@b)*b; w -= (w@b)*b
    return float(np.degrees(np.arctan2(np.cross(b, v)@w, v@w)))


def residual(actual, target, kind):
    delta = actual-target
    return (delta+180)%360-180 if kind == 'dihedral' else delta


def validate_plan(plan):
    if plan['parameter_dimension'] != 1:
        raise ValueError('a synchronized path has exactly one parameter')
    lam = np.asarray(plan['lambda_values'], float)
    if len(lam)<3 or not np.isfinite(lam).all() or lam[0]!=0 or lam[-1]!=1 or np.any(np.diff(lam)<=0):
        raise ValueError('lambda must strictly increase from zero to one')
    n = len(plan['elements'])
    endpoints = [np.asarray(plan[k], float) for k in ('start_geometry','target_geometry')]
    if any(x.shape!=(n,3) or not np.isfinite(x).all() for x in endpoints):
        raise ValueError('invalid full endpoint geometry')
    if len(set(plan['atom_map_order'])) != n:
        raise ValueError('atom identity must be bijective')
    ids = set()
    for d in plan['drivers']:
        if d['id'] in ids:
            raise ValueError('duplicate coordinate ID')
        ids.add(d['id'])
        if len(d['atoms']) != {'distance':2,'angle':3,'dihedral':4}[d['kind']] or len(set(d['atoms']))!=len(d['atoms']):
            raise ValueError('invalid coordinate atom tuple')
        if any(i<0 or i>=n for i in d['atoms']):
            raise ValueError('coordinate atom index out of bounds')
        q = np.asarray(d['values'], float)
        if q.shape!=lam.shape or not np.isfinite(q).all():
            raise ValueError('each driver needs one finite target per lambda')
        for x, target in zip(endpoints, (q[0],q[-1])):
            if abs(residual(value(x,d['kind'],d['atoms']),target,d['kind']))>1e-7:
                raise ValueError('coordinate targets do not match full boundaries')
    if not ids or any(not set(e['driver_ids'])<=ids or not e['driver_ids'] for e in plan['event_coverage']):
        raise ValueError('a required edit has no active driver')
    if plan.get('reference_geometries'):
        witness=np.asarray(plan['reference_geometries'],float)
        if witness.shape!=(len(lam),n,3) or not np.isfinite(witness).all():
            raise ValueError('invalid common geometric witness')
        for i,x in enumerate(witness):
            if any(abs(residual(value(x,d['kind'],d['atoms']),d['values'][i],d['kind']))>1e-6 for d in plan['drivers']):
                raise ValueError('joint targets have no match to their declared geometric witness')
    return plan


def coordinate_jacobian(geometry, drivers):
    """Finite-difference internal-coordinate Jacobian; angles scaled to radians."""
    x = np.asarray(geometry, float)
    jac = np.empty((len(drivers), x.size)); h=1e-5
    for j in range(x.size):
        a, b = x.copy(), x.copy(); a.flat[j]+=h; b.flat[j]-=h
        for i,d in enumerate(drivers):
            jac[i,j]=residual(value(a,d['kind'],d['atoms']), value(b,d['kind'],d['atoms']),d['kind'])/(2*h)
            if d['kind']!='distance': jac[i,j]*=np.pi/180
    return jac


def coordinate_rank(geometry, drivers):
    singular=np.linalg.svd(coordinate_jacobian(geometry,drivers),compute_uv=False)
    rank=int(np.sum(singular>max(1e-7, singular[0]*1e-6)))
    return {'rank':rank,'n_drivers':len(drivers),'independent':rank==len(drivers),
            'singular_values':singular.tolist()}


def predict_geometry(geometry, drivers, targets, max_cartesian_step=.1):
    """Minimum-norm Jacobian predictor, clipped before the QC corrector.

    This generates a seed, not an optimized frame or an acceptance decision.
    """
    x=np.asarray(geometry,float)
    delta=np.asarray([residual(t,value(x,d['kind'],d['atoms']),d['kind']) *
                      (1 if d['kind']=='distance' else np.pi/180)
                      for d,t in zip(drivers,targets)])
    if len(targets)!=len(drivers):raise ValueError('one target per driver required')
    dx=(np.linalg.pinv(coordinate_jacobian(x,drivers),rcond=1e-6)@delta).reshape(x.shape)
    largest=np.linalg.norm(dx,axis=1).max()
    if largest>max_cartesian_step:dx*=max_cartesian_step/largest
    return x+dx


def mass_weighted_lambda(geometries, masses):
    """Common arc-length progress after proper whole-system alignment."""
    f=np.asarray(geometries,float);mass=np.asarray(masses,float)
    if mass.shape!=(f.shape[1],) or np.any(mass<=0):raise ValueError('invalid atomic masses')
    steps=[np.sqrt(np.sum(mass[:,None]*(align(b,a)-a)**2)) for a,b in zip(f[:-1],f[1:])]
    if not steps or min(steps)<1e-10:raise ValueError('zero-length path interval; remove duplicate nodes')
    arc=np.r_[0,np.cumsum(steps)];return (arc/arc[-1]).tolist()


def audit_path(plan, frames, ts=None):
    """Independent gates: pinned endpoints alone never imply a connected path."""
    validate_plan(plan)
    f=np.asarray(frames,float)
    if f.shape!=(len(plan['lambda_values']),len(plan['elements']),3) or not np.isfinite(f).all():
        raise ValueError('incomplete or non-finite trajectory')
    errors=[float(np.max(np.abs(f[i]-plan[k]))) for i,k in ((0,'start_geometry'),(-1,'target_geometry'))]
    step=[float(np.sqrt(np.mean((align(b,a)-a)**2)*3)) for a,b in zip(f[:-1],f[1:])]
    policy=plan['quality_policy']
    qerrors=[max(abs(residual(value(x,d['kind'],d['atoms']), d['values'][i],d['kind'])) /
                 policy['coordinate_tolerances'][d['kind']] for d in plan['drivers']) for i,x in enumerate(f)]
    result={'boundary_errors_angstrom':errors,'boundary_ok':max(errors)<=1e-6,
            'step_rmsd_angstrom':step,'continuity_ok':max(step)<=policy['max_step_rmsd_angstrom'],
            'max_normalized_constraint_error':max(qerrors),'constraints_ok':max(qerrors)<=1}
    core=sorted({i for d in plan['drivers'] if not d.get('purpose') for i in d['atoms']})
    heavy=[i for i,e in enumerate(plan['elements']) if e!='H'] or list(range(len(plan['elements'])))
    if not core:core=list(range(len(plan['elements'])))
    result['maximum_atom_step_angstrom']=max(float(np.linalg.norm(align(b,a)-a,axis=1).max()) for a,b in zip(f[:-1],f[1:]))
    result['reaction_center_step_rmsd_angstrom']=[float(np.sqrt(np.mean((align(b[core],a[core])-a[core])**2)*3)) for a,b in zip(f[:-1],f[1:])]
    result['heavy_atom_step_rmsd_angstrom']=[float(np.sqrt(np.mean((align(b[heavy],a[heavy])-a[heavy])**2)*3)) for a,b in zip(f[:-1],f[1:])]
    if ts is not None:
        rows=[]
        for i,x in enumerate(f):
            rms=float(np.sqrt(np.mean((align(x,ts)-ts)**2)*3))
            crms=float(np.sqrt(np.mean((align(x[core],np.asarray(ts)[core])-np.asarray(ts)[core])**2)*3))
            qe=max(abs(value(x,d['kind'],d['atoms'])-value(ts,d['kind'],d['atoms'])) for d in plan['drivers'] if d['kind']=='distance')
            rows.append({'index':i,'all_atom_rmsd':rms,'core_rmsd':crms,'max_edit_distance_error':qe})
        result['ts_frames']=rows
        result['ts_near_ok']=any(r['all_atom_rmsd']<=.2 and r['core_rmsd']<=.2 and r['max_edit_distance_error']<=.2 for r in rows)
    result['qualified']=all(result[k] for k in ('boundary_ok','continuity_ok','constraints_ok')) and result.get('ts_near_ok',False)
    return result
