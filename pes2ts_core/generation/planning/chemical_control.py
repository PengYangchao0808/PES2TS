"""Chemical-control compilation for the Classic native scan (G2-AB3 WP-2).

Single facade that compiles a ``ChemicalControl`` document from a frozen
connectivity-continuation plan (schema ``pes2ts_connectivity_continuation_plan_v1``,
already produced per case by the G2.T campaign) plus its staged case
snapshot.  The facade reuses the graph-derived drivers/guards/edits as-is —
it never re-derives chemistry — and adds what the native scan needs:

* the active atom set ``K`` = union of all driver atoms (including every
  transferred hydrogen and both endpoints of each forming/breaking bond);
* the frozen lambda grid and per-driver target interpolation
  ``q_i(lambda)`` (linear between the plan's endpoint values);
* atom-order bindings (elements + map order digests) so index translation to
  ACP/xTB 1-based numbering stays verifiable;
* role separation: only ``role == "driver"`` coordinates may be compiled to
  xTB potentials; guards and monitors stay post-hoc checks.

Validation is typed: a plan whose active edits lack a driver, whose K set
misses an edit endpoint, or whose endpoints do not bracket the grid raises
``ChemicalControlError`` — never a silently narrowed scan.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PLAN_SCHEMA = "pes2ts_connectivity_continuation_plan_v1"
CONTROL_SCHEMA = "pes2ts_chemical_control_spec_v1"

_DRIVER_ROLES = ("driver",)


class ChemicalControlError(ValueError):
    """Raised when a plan/snapshot pair cannot be compiled to controls."""


@dataclass(frozen=True)
class ControlDriver:
    """One driver coordinate with its frozen endpoint schedule."""

    driver_id: str
    kind: str
    atoms: tuple[int, ...]
    maps: tuple[int, ...]
    unit: str
    role: str
    lambda_values: tuple[float, ...]
    values: tuple[float, ...]

    def target_at(self, lam: float) -> float:
        if len(self.lambda_values) == 1:
            if abs(lam - self.lambda_values[0]) > 1e-12:
                raise ChemicalControlError(
                    f"driver {self.driver_id} has a single-value schedule; "
                    f"lambda {lam} is outside it"
                )
            return self.values[0]
        return float(np.interp(lam, self.lambda_values, self.values))


@dataclass(frozen=True)
class CommonBondCheck:
    """A spectator bond common to both endpoints with its distance window."""

    atoms: tuple[int, int]
    maps: tuple[int, ...]
    minimum_distance: float
    maximum_distance: float


@dataclass(frozen=True)
class ChemicalControl:
    """Compiled per-case chemical control for one native scan."""

    reaction_id: str
    elements: tuple[str, ...]
    atom_map_order: tuple[int, ...]
    charge: int
    multiplicity: int
    drivers: tuple[ControlDriver, ...]
    active_atom_indices: tuple[int, ...]
    active_atom_maps: tuple[int, ...]
    guard_atoms: tuple[tuple[int, ...], ...]
    execution_direction: str
    origin_endpoint: str
    lambda_grid: tuple[float, ...]
    plan_sha256: str
    atom_order_sha256: str
    schedule_source: str
    common_bond_checks: tuple[CommonBondCheck, ...] = ()
    endpoint_union_pairs: frozenset[tuple[int, int]] = frozenset()
    local_support_indices: frozenset[int] = frozenset()

    def targets_at(self, lam: float) -> tuple[float, ...]:
        return tuple(driver.target_at(lam) for driver in self.drivers)


def _digest_payload(payload: Any) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compile_chemical_control(
    plan: dict[str, Any],
    *,
    reaction_id: str | None = None,
    n_grid_points: int = 21,
    endpoint_geometry_r: Sequence[Sequence[float]] | None = None,
    endpoint_geometry_p: Sequence[Sequence[float]] | None = None,
) -> ChemicalControl:
    """Compile a frozen plan into :class:`ChemicalControl`.

    Args:
        plan: ``pes2ts_connectivity_continuation_plan_v1`` document.
        reaction_id: Overrides ``plan["reaction_id"]`` when given.
        n_grid_points: Frozen uniform lambda grid size (``>= 2``); the grid
            always includes the exact endpoints 0.0 and 1.0.
        endpoint_geometry_r/p: Optional staged endpoint geometries (frozen
            atom order).  When given, the union of their covalent bond graphs
            defines which atom pairs are topology-known — a pair bonded in
            EITHER endpoint may transit the bonded/non-bonded boundary during
            the scan without being called an unsolicited bond.

    Raises:
        ChemicalControlError: typed compilation failures.
    """
    if not isinstance(plan, dict) or plan.get("schema_version") != PLAN_SCHEMA:
        raise ChemicalControlError(
            f"plan must satisfy {PLAN_SCHEMA!r}; refusing to compile foreign plans"
        )
    if not isinstance(n_grid_points, int) or n_grid_points < 2:
        raise ChemicalControlError("n_grid_points must be an int >= 2")

    elements = plan.get("elements")
    if not isinstance(elements, list) or not elements:
        raise ChemicalControlError("plan.elements must be a non-empty list")
    n_atoms = len(elements)
    map_order = plan.get("atom_map_order")
    if not isinstance(map_order, list) or len(map_order) != n_atoms:
        raise ChemicalControlError("plan.atom_map_order must parallel plan.elements")

    drivers_raw = plan.get("drivers")
    if not isinstance(drivers_raw, list) or not drivers_raw:
        raise ChemicalControlError("plan must carry at least one driver")
    drivers: list[ControlDriver] = []
    for entry in drivers_raw:
        role = entry.get("role")
        if role not in _DRIVER_ROLES:
            raise ChemicalControlError(
                f"driver {entry.get('id')!r} has role {role!r}; only driver-role "
                "coordinates may be compiled to native potentials"
            )
        atoms = entry.get("atoms")
        if not isinstance(atoms, list) or not atoms:
            raise ChemicalControlError(f"driver {entry.get('id')!r} has no atoms")
        for index in atoms:
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < n_atoms:
                raise ChemicalControlError(
                    f"driver {entry.get('id')!r} atom {index!r} is not a valid "
                    f"0-based position in the {n_atoms}-atom order"
                )
        lambda_values = entry.get("lambda_values")
        values = entry.get("values")
        if not isinstance(lambda_values, list) or not lambda_values:
            raise ChemicalControlError(f"driver {entry.get('id')!r} lacks lambda_values")
        if not isinstance(values, list) or len(values) != len(lambda_values):
            raise ChemicalControlError(
                f"driver {entry.get('id')!r} values/lambda_values length mismatch"
            )
        for lam in lambda_values:
            if not isinstance(lam, (int, float)) or not 0.0 <= float(lam) <= 1.0:
                raise ChemicalControlError(
                    f"driver {entry.get('id')!r} lambda values must lie in [0, 1]"
                )
        drivers.append(
            ControlDriver(
                driver_id=str(entry["id"]),
                kind=str(entry.get("kind", "distance")),
                atoms=tuple(int(a) for a in atoms),
                maps=tuple(int(m) for m in entry.get("maps", [])),
                unit=str(entry.get("unit", "angstrom")),
                role=role,
                lambda_values=tuple(float(v) for v in lambda_values),
                values=tuple(float(v) for v in values),
            )
        )

    active_edits = plan.get("active_edits")
    if not isinstance(active_edits, list) or not active_edits:
        raise ChemicalControlError("plan lacks active_edits; the native scan needs them")
    edit_pairs: set[tuple[int, int]] = set()
    for edit in active_edits:
        edit_maps = edit.get("maps")
        if not isinstance(edit_maps, list) or len(edit_maps) != 2:
            raise ChemicalControlError(f"active edit {edit!r} does not name a pair")
        edit_pairs.add((int(edit_maps[0]), int(edit_maps[1])))

    driver_pairs = {tuple(sorted(driver.maps)) for driver in drivers if len(driver.maps) == 2}
    missing = edit_pairs - driver_pairs
    if missing:
        raise ChemicalControlError(
            f"active edits without a driver (scan would silently narrow): {sorted(missing)}"
        )

    active_indices: set[int] = set()
    active_maps: set[int] = set()
    for driver in drivers:
        active_indices.update(driver.atoms)
        active_maps.update(driver.maps)
    for pair in edit_pairs:
        for map_id in pair:
            if map_id not in active_maps:
                raise ChemicalControlError(
                    f"edit endpoint map {map_id} is not covered by any driver; "
                    "K must include every forming/breaking endpoint and transferred H"
                )

    guards_raw = plan.get("guards")
    guard_atoms: list[tuple[int, ...]] = []
    if isinstance(guards_raw, list):
        for guard in guards_raw:
            atoms = guard.get("atoms") if isinstance(guard, dict) else None
            if isinstance(atoms, list):
                guard_atoms.append(tuple(int(a) for a in atoms))

    charge = plan.get("charge")
    multiplicity = plan.get("multiplicity")
    if not isinstance(charge, int) or not isinstance(multiplicity, int) or multiplicity < 1:
        raise ChemicalControlError("plan charge/multiplicity must be present and sane")

    lambda_grid = tuple(float(v) for v in np.linspace(0.0, 1.0, n_grid_points))

    common_checks: list[CommonBondCheck] = []
    raw_common = plan.get("common_bond_checks")
    if isinstance(raw_common, list):
        for entry in raw_common:
            if not isinstance(entry, dict):
                continue
            atoms = entry.get("atoms")
            if not isinstance(atoms, list) or len(atoms) != 2:
                continue
            low = entry.get("minimum_distance")
            high = entry.get("maximum_distance")
            if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
                continue
            common_checks.append(
                CommonBondCheck(
                    atoms=(int(atoms[0]), int(atoms[1])),
                    maps=tuple(int(m) for m in entry.get("maps", [])),
                    minimum_distance=float(low),
                    maximum_distance=float(high),
                )
            )

    endpoint_union: frozenset[tuple[int, int]] = frozenset()
    if endpoint_geometry_r is not None and endpoint_geometry_p is not None:
        union = set(_covalent_pairs(elements, np.asarray(endpoint_geometry_r, float)))
        union |= _covalent_pairs(elements, np.asarray(endpoint_geometry_p, float))
        union |= {check.atoms for check in common_checks}
        endpoint_union = frozenset(union)

    position_by_map = {int(m): i for i, m in enumerate(map_order)}
    support_indices = {
        position_by_map[int(m)]
        for m in (plan.get("local_support_maps") or [])
        if int(m) in position_by_map
    }

    plan_digest = plan.get("content_sha256")
    if not isinstance(plan_digest, str) or not plan_digest:
        plan_digest = _digest_payload(plan)

    return ChemicalControl(
        reaction_id=str(reaction_id or plan.get("reaction_id", "")),
        elements=tuple(str(e) for e in elements),
        atom_map_order=tuple(int(m) for m in map_order),
        charge=int(charge),
        multiplicity=int(multiplicity),
        drivers=tuple(drivers),
        active_atom_indices=tuple(sorted(active_indices)),
        active_atom_maps=tuple(sorted(active_maps)),
        guard_atoms=tuple(guard_atoms),
        execution_direction=str(plan.get("execution_direction", "unknown")),
        origin_endpoint=str(plan.get("origin_endpoint", "unknown")),
        lambda_grid=lambda_grid,
        plan_sha256=plan_digest,
        atom_order_sha256=_digest_payload(
            {"elements": list(elements), "map_order": list(map_order)}
        ),
        schedule_source=str(plan.get("schedule_source", "unknown")),
        common_bond_checks=tuple(common_checks),
        endpoint_union_pairs=endpoint_union,
        local_support_indices=frozenset(support_indices),
    )


_COVALENT_RADII: dict[str, float] = {
    "H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39,
}
_BOND_TOLERANCE = 0.45


def _covalent_pairs(elements: Sequence[str], geometry: np.ndarray) -> set[tuple[int, int]]:
    n = len(elements)
    pairs: set[tuple[int, int]] = set()
    for i in range(n):
        for j in range(i + 1, n):
            ri = _COVALENT_RADII.get(elements[i])
            rj = _COVALENT_RADII.get(elements[j])
            if ri is None or rj is None:
                continue
            if float(np.linalg.norm(geometry[i] - geometry[j])) <= ri + rj + _BOND_TOLERANCE:
                pairs.add((i, j))
    return pairs


def load_plan(plan_path: str | Path) -> dict[str, Any]:
    """Read and structurally check a plan document from disk."""
    try:
        plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ChemicalControlError(f"plan is unreadable: {exc}") from exc
    if not isinstance(plan, dict) or plan.get("schema_version") != PLAN_SCHEMA:
        raise ChemicalControlError(f"plan file does not satisfy {PLAN_SCHEMA!r}")
    return plan


def control_targets_payload(control: ChemicalControl, lam: float) -> list[dict[str, Any]]:
    """Serializable target table for one scan point (driver rows only)."""
    return [
        {
            "driver_id": driver.driver_id,
            "kind": driver.kind,
            "atoms": list(driver.atoms),
            "unit": driver.unit,
            "target": driver.target_at(lam),
        }
        for driver in control.drivers
    ]


__all__ = [
    "ChemicalControl",
    "ChemicalControlError",
    "CONTROL_SCHEMA",
    "ControlDriver",
    "PLAN_SCHEMA",
    "compile_chemical_control",
    "control_targets_payload",
    "load_plan",
]
