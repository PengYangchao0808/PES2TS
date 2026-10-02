"""Preserve boundary frame semantics and reject malformed reference geometries."""
from pathlib import Path
import shutil
ACP=Path('/mnt/e/Calculations/Common_Script/Auto_Calc_Platform/ACP_V1_20260811')
BACKUP=Path('/mnt/e/Calculations/AI4S_ML_Studys/PES2TS/outputs/acp_synchronized_patch_20261002/followup')

def edit(rel,a,b):
    p=ACP/rel; text=p.read_text()
    if b in text: return
    if a not in text: raise ValueError(rel+' anchor missing')
    dest=BACKUP/rel
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():shutil.copyfile(p,dest)
    p.write_text(text.replace(a,b,1))

edit('src/cccp/qc/interfaces/constraints.py',
    '        if self.reference_geometries and len(self.reference_geometries)!=self.points:',
    '        if self.reference_geometries:\n            import numpy as np\n            geometry = np.asarray(self.reference_geometries, dtype=float)\n            if geometry.ndim != 3 or geometry.shape[2] != 3 or not np.isfinite(geometry).all():\n                raise ValueError("invalid full reference geometry table")\n        if self.reference_geometries and len(self.reference_geometries)!=self.points:')
edit('src/acp/calculations/pes/contracts.py','    optimization_converged: bool = True\n',
     '    optimization_converged: bool = True\n    frame_role: str = "constrained_optimization"\n')
edit('src/acp/calculations/pes/contracts.py','            "optimization_converged": self.optimization_converged,',
     '            "optimization_converged": self.optimization_converged,\n            "frame_role": self.frame_role,')
edit('src/acp/calculations/pes/scan.py','            residual = float(value - float(target_values[coordinate_id]))',
     '            residual = float(value - float(target_values[coordinate_id]))\n            if coordinate_item.kind == "dihedral":\n                residual = (residual + 180) % 360 - 180')
edit('src/acp/calculations/pes/scan.py','                optimization_converged=bool(point.success),',
     '                optimization_converged=bool(point.success) and point_metadata.get("frame_role") != "fixed_boundary_single_point",\n                frame_role=point_metadata.get("frame_role", "constrained_optimization"),')
print('Boundary roles and periodic residuals updated')
edit('src/acp/calculations/pes/contracts.py',
     '    if math.isclose(float(coordinate.start), float(coordinate.end), abs_tol=1.0e-9):',
     '    if not coordinate.values and math.isclose(float(coordinate.start), float(coordinate.end), abs_tol=1.0e-9):')
edit('src/acp/calculations/pes/contracts.py','    step = coordinate_step(coordinate)\n',
     '    if coordinate.values:\n        if len(coordinate.values) != coordinate.n_points or any(not math.isfinite(v) for v in coordinate.values):\n            raise ValueError("explicit values require one finite value per scan point")\n        if abs(coordinate.values[0]-coordinate.start)>1e-7 or abs(coordinate.values[-1]-coordinate.end)>1e-7:\n            raise ValueError("explicit values must match coordinate endpoints")\n        if coordinate.kind == "distance" and any(v <= 0 for v in coordinate.values):\n            raise ValueError("distance targets must be positive")\n        if coordinate.kind == "angle" and any(v <= 0 or v >= 180 for v in coordinate.values):\n            raise ValueError("angle targets must lie inside zero to 180 degrees")\n        return\n    step = coordinate_step(coordinate)\n')
edit('src/acp/calculations/pes/scan.py','    raw_kind = payload.get("kind")\n',
     '    if payload.get("path_plan"):\n        if any(any(atom < 0 or atom >= len(symbols) for atom in c.atoms) for c in coordinates):\n            raise ValueError("synchronized coordinate index outside structure")\n        if not all(c.values for c in coordinates):\n            raise ValueError("explicit path_plan requires values for ALL drivers")\n        return payload\n    raw_kind = payload.get("kind")\n')
edit('src/acp/calculations/pes/scan.py','    if coordinate.start is None or coordinate.end is None:\n        return float(index / max(total_points - 1, 1))',
     '    if coordinate.values:\n        return coordinate.values[index]\n    if coordinate.start is None or coordinate.end is None:\n        return float(index / max(total_points - 1, 1))')
edit('src/acp/api/v1_schemas.py','    optimization_converged: bool = True\n',
     '    optimization_converged: bool = True\n    frame_role: str = "constrained_optimization"\n')
edit('src/acp/calculations/pes/scan_snapshot.py',
     '"status": "completed" if bool(frame.optimization_converged) else "failed",',
     '"status": "completed" if bool(frame.optimization_converged) or frame.frame_role == "fixed_boundary_single_point" and frame.scf_converged else "failed",\n                    "frame_role": frame.frame_role,')
edit('src/cccp/qc/interfaces/constraints.py','    fixed_endpoints: bool = False\n',
     '    fixed_endpoints: bool = False\n    xtb_scc_max_iterations: int | None = None\n')
edit('src/acp/calculations/pes/scan.py','        fixed_endpoints=bool(path_data.get("fixed_endpoints", False)))',
     '        fixed_endpoints=bool(path_data.get("fixed_endpoints", False)),\n        xtb_scc_max_iterations=path_data.get("xtb_scc_max_iterations"))')
edit('src/cccp/qc/interfaces/orca.py','        result_points: list[RelaxedScanPoint] = []\n        for index in range(plan.points):',
     '        result_points: list[RelaxedScanPoint] = []\n        extra_xtb_blocks = None\n        if plan.xtb_scc_max_iterations is not None:\n            if not _is_orca_gfn_xtb_method(eff_method) or not 1 <= int(plan.xtb_scc_max_iterations) <= 10000:\n                raise ValueError("invalid xTB SCC iteration limit")\n            extra_xtb_blocks = [f\'%xtb\\n XTBINPUTSTRING "--iterations {int(plan.xtb_scc_max_iterations)}"\\nend\']\n        for index in range(plan.points):')
edit('src/cccp/qc/interfaces/orca.py','method=eff_method, basis=eff_basis, solvent=input_solvent, solvent_model=input_solvent_model)',
     'method=eff_method, basis=eff_basis, solvent=input_solvent, solvent_model=input_solvent_model,\n                        extra_blocks=extra_xtb_blocks, route_extras=resolved_route_extras, scf_convergence=scf_convergence,\n                        scf_maxiter=scf_maxiter, grid=grid, dispersion=dispersion, aux_j_basis=aux_j_basis, aux_c_basis=aux_c_basis)')
edit('src/cccp/qc/interfaces/orca.py','                    plan.frame_constraints(index),\n                    charge=charge,',
     '                    plan.frame_constraints(index),\n                    extra_blocks=extra_xtb_blocks,\n                    charge=charge,')
# Fixed SP boundaries are successful energy frames, not non-converged optimizations.
edit('src/acp/calculations/pes/scan.py','    if any(not frame.optimization_converged for frame in frames):\n',
     '    if any(not frame.optimization_converged and frame.frame_role != "fixed_boundary_single_point" for frame in frames):\n')
# The scan has two quality builders; update the second occurrence separately.
p=ACP/'src/acp/calculations/pes/scan.py'
t=p.read_text()
t=t.replace('    if any(not frame.optimization_converged for frame in frames):\n',
            '    if any(not frame.optimization_converged and frame.frame_role != "fixed_boundary_single_point" for frame in frames):\n')
p.write_text(t)
