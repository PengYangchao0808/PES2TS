"""V4.2 physical prior spec (Lane A estimate, Lane B/C consume) — contract §4.

Mass-weighted reverse-IRC physical skip ``a_phys = Δs_fit · b`` with

    b_i = (g_i / m_i) / sqrt( Σ_j ‖g_j‖² / m_j + eps_g² ),

``g`` = teacher ``observation.gradient`` (Hartree/Å, dE/dX — the receipts are
already declared dE/dX so no sign conversion; force sources would convert
``g = −F`` per §4), ``m`` = ``direct_preprocess.masses_for`` (typed refusal on
unknown Z, never a silent mass 1).

Frozen estimators (all fit-only, contract §4):

* ``eps_g  = 1e-3 × median_fit( sqrt(Σ_j ‖g_j‖²/m_j) )``
* ``tau_g  = 1e-6 × same median``  (below tau_g a state's ``a_phys`` is all-zero,
  coverage recorded separately)
* ``Δs_fit = max(0, Σ w b·a* / Σ w ‖b‖²)`` with ``w`` the SAME reduction as
  ``L_xyz`` (atom SUM → branch mean → reaction mean → group mean); denominator
  ≤ 1e-30 is an explicit typed error (never a silent zero)
* ``a_s`` = median of non-zero per-atom ``‖a*‖`` (frozen positive scale, NOT an
  upper bound)
* audit output: fit native mass-weighted step-length median
  ``median( sqrt(Σ_i m_i ‖a*_i‖²) )``.

``Δs_fit`` has units Å·√amu so that ``a_phys`` lands in Å (b has units
1/√amu).  The artifact ``pesn_dual_ts_physical_prior_v1`` binds the dev-split
hash + the mass-table version.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Final, Mapping, Sequence

import numpy as np

from pes2ts_core.pesn.offline.data.direct_preprocess import masses_for as _masses_for

PHYSICAL_PRIOR_SCHEMA: Final[str] = "pesn_dual_ts_physical_prior_v1"
PHYSICAL_PRIOR_TOOL_VERSION: Final[str] = "pes2ts_dual_ts_physical_prior_tool_v1"
#: Mass-table version bound into every spec (direct_preprocess.MASS_BY_Z).
MASSES_VERSION: Final[str] = "direct_preprocess.masses_for"

EPS_G_FACTOR: Final[float] = 1e-3
TAU_G_FACTOR: Final[float] = 1e-6
#: NNLS denominator floor (≤ this => typed error, contract §4).
DENOMINATOR_FLOOR: Final[float] = 1e-30


class PhysicalPriorError(RuntimeError):
    """Typed refusal for physical-prior estimation violations."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise PhysicalPriorError(code, detail)


@dataclass(frozen=True, slots=True)
class PhysicalPriorSpec:
    """Frozen physical-prior estimate for one split population."""

    delta_s_fit: float
    eps_g: float
    tau_g: float
    a_s: float
    median_mass_weighted_step_length_angstrom: float
    median_gradient_mass_norm: float
    n_states: int = 0
    n_atoms_nonzero: int = 0
    split_hash: str = ""
    masses_version: str = MASSES_VERSION
    schema_version: str = PHYSICAL_PRIOR_SCHEMA
    #: extra context (population counts, rules) kept for traceability only.
    meta: Mapping[str, object] = field(default_factory=dict)

    def to_payload(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "delta_s_fit": float(self.delta_s_fit),
            "eps_g": float(self.eps_g),
            "tau_g": float(self.tau_g),
            "a_s": float(self.a_s),
            "median_mass_weighted_step_length_angstrom":
                float(self.median_mass_weighted_step_length_angstrom),
            "median_gradient_mass_norm": float(self.median_gradient_mass_norm),
            "n_states": int(self.n_states),
            "n_atoms_nonzero": int(self.n_atoms_nonzero),
            "split_hash": str(self.split_hash),
            "masses_version": str(self.masses_version),
            "rules": {
                "b_definition": "b_i = (g_i/m_i)/sqrt(sum_j||g_j||^2/m_j + eps_g^2)",
                "a_phys_definition": "a_phys = delta_s_fit * b; states with "
                                     "sqrt(sum_j||g_j||^2/m_j) < tau_g are zeroed",
                "eps_g_rule": "1e-3 * median_fit(sqrt(sum_j||g_j||^2/m_j))",
                "tau_g_rule": "1e-6 * same median",
                "delta_s_fit_rule": "max(0, sum w b.a* / sum w ||b||^2); w = "
                                    "L_xyz reduction (atom sum -> branch mean "
                                    "-> reaction mean -> group mean)",
                "delta_s_fit_units": "angstrom * sqrt(amu)",
                "a_s_rule": "median of non-zero per-atom ||a*|| (positive "
                            "scale, NOT an upper bound)",
                "gradient_source": "teacher observation.gradient (dE/dX, "
                                   "Ha/angstrom); force sources would use g=-F",
                "mass_source": "direct_preprocess.masses_for (typed refusal "
                               "on unknown Z)",
                "denominator_floor": DENOMINATOR_FLOOR,
            },
            "meta": dict(self.meta),
        }

    @classmethod
    def from_payload(cls, payload: Mapping) -> "PhysicalPriorSpec":
        _require(payload.get("schema_version") == PHYSICAL_PRIOR_SCHEMA,
                 "SCHEMA_UNKNOWN",
                 f"expected {PHYSICAL_PRIOR_SCHEMA}, got "
                 f"{payload.get('schema_version')!r}")
        return cls(
            delta_s_fit=float(payload["delta_s_fit"]),
            eps_g=float(payload["eps_g"]),
            tau_g=float(payload["tau_g"]),
            a_s=float(payload["a_s"]),
            median_mass_weighted_step_length_angstrom=float(
                payload["median_mass_weighted_step_length_angstrom"]),
            median_gradient_mass_norm=float(payload["median_gradient_mass_norm"]),
            n_states=int(payload.get("n_states", 0)),
            n_atoms_nonzero=int(payload.get("n_atoms_nonzero", 0)),
            split_hash=str(payload.get("split_hash", "")),
            masses_version=str(payload.get("masses_version", MASSES_VERSION)),
            schema_version=str(payload.get("schema_version",
                                           PHYSICAL_PRIOR_SCHEMA)),
            meta=dict(payload.get("meta", {})),
        )


def _mass_norm(g: np.ndarray, masses: np.ndarray) -> float:
    """``sqrt(Σ_j ‖g_j‖² / m_j)`` — the per-state gradient mass norm."""
    g = np.asarray(g, dtype=float)
    m = np.asarray(masses, dtype=float)
    if g.ndim != 2 or g.shape[1] != 3:
        raise PhysicalPriorError("GRADIENT_SHAPE",
                                 f"gradient must be (N,3), got {g.shape}")
    _require(m.shape == (g.shape[0],), "MASS_SHAPE_MISMATCH",
             f"masses {m.shape} incompatible with gradient {g.shape}")
    if not np.isfinite(g).all():
        raise PhysicalPriorError("NONFINITE_GRADIENT", "gradient not finite")
    return float(np.sqrt(((g * g).sum(axis=1) / m).sum()))


def estimate_physical_prior(
    records: Sequence[Mapping],
    masses: Callable[[np.ndarray], np.ndarray] = _masses_for,
) -> PhysicalPriorSpec:
    """Estimate the physical-prior spec over the fit45_train population.

    ``records``: one mapping per non-terminal native state with keys
    ``group_id``, ``branch_id``, ``gradient`` (N,3) Ha/Å, ``atomic_numbers``
    (N,) int, and ``action`` (N,3) Å.  Weights follow the frozen ``L_xyz``
    reduction (atom sum → branch mean → reaction mean → group mean).
    """
    records = list(records)
    _require(bool(records), "EMPTY_POPULATION", "no fit records")

    # First pass: resolve per-state gradient mass norms (needed for eps_g) and
    # collect the a_s / step-length / atom-norm statistics.
    per_state_mag: list[float] = []
    step_lengths: list[float] = []
    atom_norms_nonzero: list[float] = []
    prepared: list[dict] = []
    for rec in records:
        g = np.asarray(rec["gradient"], dtype=float)
        z = np.asarray(rec["atomic_numbers"], dtype=int)
        a = np.asarray(rec["action"], dtype=float)
        m = np.asarray(masses(z), dtype=float)
        _require(g.shape == a.shape and g.shape[1] == 3
                 and m.shape == (g.shape[0],),
                 "RECORD_SHAPE_MISMATCH", "gradient/action/masses shape mismatch")
        if not np.isfinite(a).all():
            raise PhysicalPriorError("NONFINITE_ACTION", "action not finite")
        mag = _mass_norm(g, m)
        per_state_mag.append(mag)
        step_lengths.append(float(np.sqrt((m * (a * a).sum(axis=1)).sum())))
        anorms = np.linalg.norm(a, axis=1)
        atom_norms_nonzero.extend(float(v) for v in anorms if v > 0.0)
        prepared.append({"g": g, "a": a, "m": m, "mag": mag,
                         "group_id": str(rec["group_id"]),
                         "branch_id": str(rec["branch_id"])})

    median_mag = float(np.median(np.asarray(per_state_mag, dtype=float)))
    if not np.isfinite(median_mag) or median_mag <= 0.0:
        raise PhysicalPriorError(
            "NNLS_DENOMINATOR_DEGENERATE",
            f"all gradients degenerate (median gradient mass norm "
            f"{median_mag:.3e} <= 0); the NNLS denominator would be zero")
    eps_g = EPS_G_FACTOR * median_mag
    tau_g = TAU_G_FACTOR * median_mag

    # Second pass: build b with the RESOLVED eps_g (frozen, fit-only).
    cell_num: dict[tuple[str, str], float] = {}
    cell_den: dict[tuple[str, str], float] = {}
    cell_count: dict[tuple[str, str], int] = {}
    for rec in prepared:
        g, a, m = rec["g"], rec["a"], rec["m"]
        b = _b_for_state(g, m, eps_g)
        b_dot_a = float((b * a).sum())
        b_norm_sq = float((b * b).sum())
        cell = (rec["group_id"], rec["branch_id"])
        cell_num[cell] = cell_num.get(cell, 0.0) + b_dot_a
        cell_den[cell] = cell_den.get(cell, 0.0) + b_norm_sq
        cell_count[cell] = cell_count.get(cell, 0) + 1

    # L_xyz-consistent reduction: cell mean -> group mean over cells ->
    # group mean over groups (the 1/n_groups factor cancels in the ratio).
    groups = sorted({c[0] for c in cell_num})
    cell_mean_num = {c: cell_num[c] / cell_count[c] for c in cell_num}
    cell_mean_den = {c: cell_den[c] / cell_count[c] for c in cell_den}
    num_total = 0.0
    den_total = 0.0
    for g in groups:
        cells_g = [c for c in cell_num if c[0] == g]
        num_total += float(np.mean([cell_mean_num[c] for c in cells_g]))
        den_total += float(np.mean([cell_mean_den[c] for c in cells_g]))
    num_total /= len(groups)
    den_total /= len(groups)

    _require(den_total > DENOMINATOR_FLOOR, "NNLS_DENOMINATOR_DEGENERATE",
             f"weighted ||b||^2 denominator {den_total:.3e} <= "
             f"{DENOMINATOR_FLOOR:g} (explicit error, not a silent zero)")
    delta_s_fit = max(0.0, num_total / den_total)
    _require(np.isfinite(delta_s_fit), "NNLS_NONFINITE",
             "delta_s_fit is not finite")

    a_s = float(np.median(np.asarray(atom_norms_nonzero, dtype=float))) \
        if atom_norms_nonzero else 0.0
    med_step = float(np.median(np.asarray(step_lengths, dtype=float)))

    return PhysicalPriorSpec(
        delta_s_fit=delta_s_fit,
        eps_g=eps_g,
        tau_g=tau_g,
        a_s=a_s,
        median_mass_weighted_step_length_angstrom=med_step,
        median_gradient_mass_norm=median_mag,
        n_states=len(records),
        n_atoms_nonzero=len(atom_norms_nonzero),
        meta={
            "n_groups": len(groups),
            "eps_g_factor": EPS_G_FACTOR,
            "tau_g_factor": TAU_G_FACTOR,
        },
    )


def _b_for_state(g: np.ndarray, masses: np.ndarray, eps_g: float) -> np.ndarray:
    mag = _mass_norm(g, masses)
    return (g / masses[:, None]) / np.sqrt(mag * mag + eps_g * eps_g)


def a_phys_for_batch(g_angstrom: np.ndarray,
                     atomic_numbers: np.ndarray,
                     spec: PhysicalPriorSpec) -> np.ndarray:
    """Physical skip ``a_phys`` for a batch of gradients.

    ``g_angstrom`` (N, atoms, 3) Hartree/Å; ``atomic_numbers`` (atoms,) int;
    returns (N, atoms, 3) Å.  States with ``sqrt(Σ‖g_j‖²/m_j) < tau_g`` are
    all-zero (coverage is reported by the caller separately).
    """
    g = np.asarray(g_angstrom, dtype=float)
    z = np.asarray(atomic_numbers, dtype=int)
    if g.ndim != 3 or g.shape[2] != 3:
        raise PhysicalPriorError("GRADIENT_SHAPE",
                                 f"expected (N,atoms,3), got {g.shape}")
    m = np.asarray(_masses_for(z), dtype=float)
    _require(m.shape == (g.shape[1],), "MASS_SHAPE_MISMATCH",
             f"masses {m.shape} incompatible with {g.shape[1]} atoms")
    n = g.shape[0]
    out = np.zeros_like(g, dtype=float)
    for i in range(n):
        mag = _mass_norm(g[i], m)
        if mag < float(spec.tau_g):
            continue
        b = _b_for_state(g[i], m, float(spec.eps_g))
        out[i] = float(spec.delta_s_fit) * b
    return out


def save_physical_prior(spec: PhysicalPriorSpec, path: str | Path) -> str:
    """Write the spec payload; returns its content digest."""
    from pes2ts_core.pesn.contracts.dual_ts import content_digest_of

    payload = spec.to_payload()
    digest = content_digest_of(payload)
    record = {"spec": payload, "spec_digest": digest}
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1, sort_keys=True,
                              ensure_ascii=False, allow_nan=False),
                   encoding="utf-8")
    return digest


def load_physical_prior(path: str | Path) -> PhysicalPriorSpec:
    """Load a physical-prior spec, verifying its recorded content digest."""
    from pes2ts_core.pesn.contracts.dual_ts import content_digest_of

    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = dict(doc["spec"])
    digest = content_digest_of(payload)
    if digest != str(doc["spec_digest"]):
        raise PhysicalPriorError(
            "DIGEST_MISMATCH",
            f"recorded {doc['spec_digest']!r} != recomputed {digest!r}")
    return PhysicalPriorSpec.from_payload(payload)


__all__ = [
    "DENOMINATOR_FLOOR",
    "EPS_G_FACTOR",
    "MASSES_VERSION",
    "PHYSICAL_PRIOR_SCHEMA",
    "PHYSICAL_PRIOR_TOOL_VERSION",
    "TAU_G_FACTOR",
    "PhysicalPriorError",
    "PhysicalPriorSpec",
    "a_phys_for_batch",
    "estimate_physical_prior",
    "load_physical_prior",
    "save_physical_prior",
]
