"""Apply a narrow, backed-up ACP extension; run only against the named checkout."""
from pathlib import Path
import hashlib
import json

ROOT=Path('/mnt/e/Calculations/AI4S_ML_Studys/PES2TS')
ACP=Path('/mnt/e/Calculations/Common_Script/Auto_Calc_Platform/ACP_V1_20260811')
BACKUP=ROOT/'outputs/acp_synchronized_patch_20261002'
BACKUP.mkdir(parents=True,exist_ok=True)
changes={}

def edit(rel, replacements):
    p=ACP/rel
    if not p.resolve().is_relative_to(ACP.resolve()): raise ValueError('outside ACP checkout')
    out=BACKUP/rel
    old=out.read_text() if out.exists() else p.read_text(); new=old
    for a,b in replacements:
        if a not in new: raise ValueError(f'{rel}: expected patch anchor absent: {a[:80]}')
        new=new.replace(a,b,1)
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists() and p.read_text() not in (old,new):
        raise ValueError('ACP changed since patch; inspect before reapplying')
    if not out.exists(): out.write_text(old)
    p.write_text(new)
    changes[rel]={'before':hashlib.sha256(old.encode()).hexdigest(),'after':hashlib.sha256(new.encode()).hexdigest()}

edit('src/cccp/qc/interfaces/constraints.py',[
 ('from dataclasses import dataclass','from dataclasses import dataclass\nimport math'),
 ('    force_constant: float | None = None\n\n    def __post_init__', '    force_constant: float | None = None\n    values: tuple[float, ...] = ()\n\n    def __post_init__'),
 ('    def constraint_at(self, progress: float)', '    def constraint_at(self, progress: float, index: int | None = None)'),
 ('        if self.role == "freeze":\n            target = self.start', '        if self.values:\n            if index is None:\n                raise ValueError("explicit coordinate values require a frame index")\n            target = self.values[index]\n        elif self.role == "freeze":\n            target = self.start'),
 ('            force_constant=_opt_float(data.get("force_constant")),','            force_constant=_opt_float(data.get("force_constant")),\n            values=tuple(float(v) for v in data.get("values", ())),'),
 ('    def __post_init__(self) -> None:\n        if self.points < 2:', '    lambda_values: tuple[float, ...] = ()\n    reference_geometries: tuple = ()\n    fixed_endpoints: bool = False\n\n    def __post_init__(self) -> None:\n        if self.lambda_values and (len(self.lambda_values)!=self.points or self.lambda_values[0]!=0 or self.lambda_values[-1]!=1 or any(not math.isfinite(x) for x in self.lambda_values) or any(b<=a for a,b in zip(self.lambda_values,self.lambda_values[1:]))):\n            raise ValueError("lambda_values must increase from zero to one with one value per point")\n        if any(c.values and (len(c.values)!=self.points or any(not math.isfinite(v) for v in c.values) or abs(c.values[0]-c.start)>1e-7 or abs(c.values[-1]-c.end)>1e-7) for c in self.coordinates):\n            raise ValueError("coordinate values must be finite, match points and endpoints")\n        if self.reference_geometries and len(self.reference_geometries)!=self.points:\n            raise ValueError("one reference geometry is required per point")\n        if self.fixed_endpoints and not self.reference_geometries:\n            raise ValueError("fixed boundaries require reference geometries")\n        if self.points < 2:'),
 ('            constraints.append(spec.constraint_at(progress))','            constraints.append(spec.constraint_at(progress, index))'),
 ('            constraint = spec.constraint_at(progress)','            constraint = spec.constraint_at(progress, index)'),
])
edit('src/acp/calculations/pes/contracts.py',[
 ('MAX_SYNC_COORDINATES = 4','MAX_SYNC_COORDINATES = 64'),
 ('    n_points: int = 16\n','    n_points: int = 16\n    values: tuple[float, ...] = ()\n'),
 ('            n_points=n_points,','            n_points=n_points,\n            values=tuple(float(v) for v in payload.get("values", ())),'),
 ('            "n_points": self.n_points,','            "n_points": self.n_points,\n            **({"values": list(self.values)} if self.values else {}),'),
])
edit('src/acp/calculations/pes/scan.py',[
 ('        end=coordinate.end,','        end=coordinate.end,\n        values=coordinate.values,'),
 ('execution_mode = "pointwise" if len(scan_coordinates) > 1 else "native_scan"','execution_mode = "pointwise" if len(scan_coordinates) > 1 or any(c.values for c in scan_coordinates) else "native_scan"'),
 ('            point_callback=snapshot_writer.publish_point,','            path_plan=req.selection.get("path_plan"),\n            point_callback=snapshot_writer.publish_point,'),
 ('    backend_ref = acp.backends.get_backend(protocol.scan_driver.software)','    backend_ref = acp.backends.get_backend(protocol.scan_driver.software)'),
 ('    plan = ReactionCoordinatePlan(coordinates=specs, points=scan_coordinates[0].n_points)','    path_data = path_plan or {}\n    plan = ReactionCoordinatePlan(coordinates=specs, points=scan_coordinates[0].n_points,\n        lambda_values=tuple(path_data.get("lambda_values", ())),\n        reference_geometries=tuple(path_data.get("reference_geometries", ())),\n        fixed_endpoints=bool(path_data.get("fixed_endpoints", False)))'),
])
# Add the keyword at the backend-helper signature only, not unrelated functions.
p=ACP/'src/acp/calculations/pes/scan.py'
t=p.read_text(); start=t.index('def _run_relaxed_scan_backend('); end=t.index(') -> RelaxedScanResult:',start)
t=t[:end]+'    path_plan: dict[str, Any] | None = None,\n'+t[end:]; p.write_text(t)
edit('src/cccp/qc/interfaces/orca.py',[
 ('if plan is not None and len(plan.drive_coordinates()) > 1:', 'if plan is not None and (len(plan.drive_coordinates()) > 1 or any(c.values for c in plan.coordinates) or plan.fixed_endpoints):'),
 ('            progress = index / max(plan.points - 1, 1)','            progress = plan.lambda_values[index] if plan.lambda_values else index / max(plan.points - 1, 1)'),
 ('                result = self.constrained_optimize(\n                    seed,', '                if plan.reference_geometries:\n                    seed = np.asarray(plan.reference_geometries[index], dtype=float)\n                if plan.fixed_endpoints and index in (0, plan.points - 1):\n                    result = self.single_point(seed, symbols, charge=charge, multiplicity=multiplicity,\n                        output_dir=frame_dir, output_name=f"{output_name}_boundary",\n                        method=eff_method, basis=eff_basis, solvent=input_solvent, solvent_model=input_solvent_model)\n                    result.coordinates = seed.copy()\n                    break\n                result = self.constrained_optimize(\n                    seed,'),
 ('                        "scf_converged": True,\n                    },\n                )\n            )\n            _notify_scan_point', '                        "scf_converged": True,\n                        "frame_role": "fixed_boundary_single_point" if plan.fixed_endpoints and index in (0, plan.points-1) else "constrained_optimization",\n                        "reference_guided": bool(plan.reference_geometries),\n                    },\n                )\n            )\n            _notify_scan_point'),
])
(BACKUP/'manifest.json').write_text(json.dumps(changes,indent=2))
print('Patched and backed up',len(changes),'ACP files')
