"""Bind saved ORCA/xTB gradients to a frame before deriving force evidence."""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
from rdkit import Chem

from pes2ts_core.generation.planning.continuation import target_values
from pes2ts_core.generation.planning.synchronized_path import coordinate_jacobian

BOHR_ANGSTROM = .529177210903


def read_bound_engrad(path, geometry, elements, energy):
    path = Path(path)
    if not path.is_file():
        return {"status": "missing"}
    try:
        raw = path.read_text(encoding="utf-8")
        rows = [line.strip() for line in raw.splitlines() if line.strip() and not line.lstrip().startswith("#")]
        n = int(rows[0]); saved_energy = float(rows[1])
        if n != len(elements) or len(rows) != 2+4*n:
            raise ValueError("INVALID_GRADIENT_LENGTH")
        gradient = np.array([float(v) for v in rows[2:2+3*n]]).reshape(n, 3)
        atom_rows = [r.split() for r in rows[2+3*n:]]
        numbers = [int(r[0]) for r in atom_rows]
        expected = [Chem.GetPeriodicTable().GetAtomicNumber(e) for e in elements]
        x = np.array([[float(v) for v in r[1:]] for r in atom_rows])*BOHR_ANGSTROM
        if numbers != expected or x.shape != (n, 3) or not np.isfinite(x).all() or not np.isfinite(gradient).all():
            raise ValueError("INVALID_GRADIENT_IDENTITY")
        error = float(np.linalg.norm(x-np.asarray(geometry), axis=1).max())
        if not np.isfinite(saved_energy) or abs(saved_energy-energy)>1e-6 or error>2e-5:
            raise ValueError("GRADIENT_FRAME_MISMATCH")
        return {"status": "bound", "gradient_hartree_per_bohr": gradient.tolist(),
                "energy_hartree": saved_energy, "maximum_geometry_error_angstrom": error,
                "file": str(path), "sha256": hashlib.sha256(raw.encode()).hexdigest()}
    except (ValueError, IndexError) as exc:
        return {"status": "rejected", "reason": str(exc), "file": str(path)}


def force_evidence(plan, frame, bound):
    if bound["status"] != "bound": return bound
    coordinates = plan["drivers"]+plan.get("guards", [])
    j = coordinate_jacobian(frame["geometry"], coordinates)
    g = np.asarray(bound["gradient_hartree_per_bohr"]).ravel()/BOHR_ANGSTROM
    # L = E + nu*(q-c). This is a reconstructed multiplier, not an ORCA output.
    multipliers = np.linalg.lstsq(j.T, -g, rcond=1e-7)[0]
    perpendicular = g+j.T@multipliers
    lam = frame["lambda"]
    lo, hi = max(0., lam-1e-5), min(1., lam+1e-5)
    dc = (np.asarray(target_values(plan, hi))-target_values(plan, lo))/(hi-lo)
    dc *= [1. if d["kind"] == "distance" else np.pi/180 for d in coordinates]
    return {**bound, "physical_gradient_norm_hartree_per_bohr": float(np.linalg.norm(g)*BOHR_ANGSTROM),
            "perpendicular_gradient_norm_hartree_per_bohr": float(np.linalg.norm(perpendicular)*BOHR_ANGSTROM),
            "reconstructed_multipliers": multipliers.tolist(), "multiplier_source": "least_squares_from_bound_physical_gradient",
            "reconstructed_energy_slope_hartree_per_lambda": float(-multipliers@dc),
            "stationary_point_verified": False}
