"""Frozen request contract for ACP ``XtbConstrainedOpt`` (G2-AB3 WP-4).

Builds and validates ``pes2ts_xtb_constrained_opt_request_v1`` payloads for
one xTB native constrained optimization per scan point.  The contract rules
mirror the Classic design (ADR-0012 §5.3/§7):

* atom indices are **0-based positions in the frozen atom order** — never map
  ids and never ``map_id - 1``; the request carries the full element list and
  geometry so both sides can bind ``atom_order_sha256``;
* every driver target carries its unit (distances in angstrom);
* the ``freeze_environment`` mode is an explicit ``fix_atoms`` list (exact
  Cartesian freeze), never an equality potential pretending to be a freeze;
* the recipe is frozen by the caller (PES2TS owns targets/protocol); nothing
  here defaults a physics decision.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Sequence

XTC_REQUEST_SCHEMA = "pes2ts_xtb_constrained_opt_request_v1"
"""Must match ``acp.workflows.xtb_constrained_opt.XTC_SCHEMA``."""

ACP_WORKFLOW_NAME = "XtbConstrainedOpt"
ACP_OPT_CONFIG_FLAG = "--opt-config"

_SUPPORTED_KINDS = {"distance": ("angstrom", 2), "angle": ("degree", 3), "dihedral": ("degree", 4)}


class ConstraintRequestError(ValueError):
    """Raised when a constraint request cannot be built or is invalid."""


@dataclass(frozen=True)
class DriverTarget:
    """One driver coordinate target for a scan point."""

    driver_id: str
    kind: str  # distance | angle | dihedral
    atoms: tuple[int, ...]  # 0-based positions in the frozen atom order
    target: float
    unit: str
    role: str = "driver"  # drivers only in v1; guards post-hoc


def _sha256_json(payload: Any) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def atom_order_digest(elements: Sequence[str], n_rows: int) -> str:
    """Digest binding the frozen atom order to the request."""
    return _sha256_json({"elements": list(elements), "n_rows": int(n_rows)})


def geometry_digest(geometry: Sequence[Sequence[float]], elements: Sequence[str]) -> str:
    """Digest of the seed geometry actually submitted (6-decimal rounding)."""
    rows = [[round(float(x), 6) for x in row] for row in geometry]
    return _sha256_json({"geometry": rows, "elements": list(elements)})


def build_constraint_request(
    *,
    geometry: Sequence[Sequence[float]],
    elements: Sequence[str],
    charge: int,
    multiplicity: int,
    targets: Sequence[DriverTarget],
    reaction_id: str = "",
    point_id: str = "",
    gfn_level: int = 2,
    force_constant: float | None = 50.0,
    opt_level: str = "normal",
    max_cycle: int | None = 500,
    fix_atoms: Sequence[int] | None = None,
    physical_sp: bool = True,
    threads: int = 4,
    timeout_seconds: int | None = 300,
    plan_sha256: str | None = None,
    parent_frame_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a frozen ``pes2ts_xtb_constrained_opt_request_v1`` payload.

    Raises:
        ConstraintRequestError: on shape/unit/arity violations.  Never
            silently drops a driver target — every entry in ``targets``
            becomes a constraint.
    """
    n_atoms = len(elements)
    if not n_atoms:
        raise ConstraintRequestError("elements must be a non-empty list")
    if len(geometry) != n_atoms:
        raise ConstraintRequestError(
            f"geometry has {len(geometry)} rows but {n_atoms} elements"
        )
    if not targets:
        raise ConstraintRequestError("at least one driver target is required")
    constraints: list[dict[str, Any]] = []
    for target in targets:
        if target.kind not in _SUPPORTED_KINDS:
            raise ConstraintRequestError(f"unsupported coordinate kind {target.kind!r}")
        expected_unit, arity = _SUPPORTED_KINDS[target.kind]
        if target.unit != expected_unit:
            raise ConstraintRequestError(
                f"driver {target.driver_id}: kind {target.kind} requires unit "
                f"{expected_unit!r}, got {target.unit!r}"
            )
        if len(target.atoms) != arity or len(set(target.atoms)) != arity:
            raise ConstraintRequestError(
                f"driver {target.driver_id}: needs exactly {arity} distinct atoms"
            )
        for index in target.atoms:
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < n_atoms:
                raise ConstraintRequestError(
                    f"driver {target.driver_id}: atom index {index!r} is not a valid "
                    f"0-based position in the {n_atoms}-atom order"
                )
        if not (target.target > 0.0):
            raise ConstraintRequestError(
                f"driver {target.driver_id}: target must be positive, got {target.target}"
            )
        if target.role != "driver":
            raise ConstraintRequestError(
                f"driver {target.driver_id}: only role='driver' coordinates may be "
                "compiled to potentials; guards/monitors are post-hoc checks"
            )
        constraints.append(
            {
                "kind": target.kind,
                "atoms": list(target.atoms),
                "target": float(target.target),
                "unit": target.unit,
            }
        )
    if fix_atoms is not None:
        for index in fix_atoms:
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < n_atoms:
                raise ConstraintRequestError(
                    f"fix_atoms entry {index!r} is not a valid 0-based position"
                )
        fix_atoms_payload: list[int] | None = sorted(set(int(i) for i in fix_atoms))
    else:
        fix_atoms_payload = None

    payload: dict[str, Any] = {
        "schema_version": XTC_REQUEST_SCHEMA,
        "reaction_id": str(reaction_id),
        "source": {
            "geometry": [[float(x) for x in row] for row in geometry],
            "elements": [str(e) for e in elements],
            "charge": int(charge),
            "multiplicity": int(multiplicity),
        },
        "recipe": {
            "gfn_level": int(gfn_level),
            "constraints": constraints,
            "force_constant": None if force_constant is None else float(force_constant),
            "opt_level": str(opt_level),
            "max_cycle": None if max_cycle is None else int(max_cycle),
            "fix_atoms": fix_atoms_payload,
            "physical_sp": bool(physical_sp),
            "threads": int(threads),
            "timeout_seconds": None if timeout_seconds is None else int(timeout_seconds),
        },
    }
    provenance: dict[str, Any] = {
        "point_id": str(point_id),
        "atom_order_sha256": atom_order_digest(elements, len(geometry)),
        "seed_geometry_sha256": geometry_digest(geometry, elements),
    }
    if plan_sha256:
        provenance["plan_sha256"] = str(plan_sha256)
    if parent_frame_sha256:
        provenance["parent_frame_sha256"] = str(parent_frame_sha256)
    provenance["request_sha256"] = _sha256_json(payload)
    payload["provenance"] = provenance
    return payload


def request_sha256(request: dict[str, Any]) -> str:
    """Return the provenance-bound request digest (the idempotency key)."""
    try:
        return str(request["provenance"]["request_sha256"])
    except (KeyError, TypeError) as exc:
        raise ConstraintRequestError("request lacks provenance.request_sha256") from exc


def validate_constraint_request(request: Any) -> None:
    """Structural validation mirroring the ACP-side workflow checks.

    Raises:
        ConstraintRequestError: on any violation.
    """
    if not isinstance(request, dict):
        raise ConstraintRequestError("request must be a JSON object")
    if request.get("schema_version") != XTC_REQUEST_SCHEMA:
        raise ConstraintRequestError(
            f"schema_version must be {XTC_REQUEST_SCHEMA!r}"
        )
    source = request.get("source")
    if not isinstance(source, dict):
        raise ConstraintRequestError("source must be an object")
    elements = source.get("elements")
    geometry = source.get("geometry")
    if not isinstance(elements, list) or not elements:
        raise ConstraintRequestError("source.elements must be a non-empty list")
    if not isinstance(geometry, list) or len(geometry) != len(elements):
        raise ConstraintRequestError("source.geometry must match source.elements length")
    for row in geometry:
        if not isinstance(row, list) or len(row) != 3:
            raise ConstraintRequestError("source.geometry rows must be [x, y, z]")
    recipe = request.get("recipe")
    if not isinstance(recipe, dict):
        raise ConstraintRequestError("recipe must be an object")
    constraints = recipe.get("constraints")
    if not isinstance(constraints, list) or not constraints:
        raise ConstraintRequestError("recipe.constraints must be a non-empty list")
    n_atoms = len(elements)
    for entry in constraints:
        if not isinstance(entry, dict):
            raise ConstraintRequestError("recipe.constraints entries must be objects")
        kind = entry.get("kind")
        if kind not in _SUPPORTED_KINDS:
            raise ConstraintRequestError(f"unsupported constraint kind {kind!r}")
        expected_unit, arity = _SUPPORTED_KINDS[kind]
        if entry.get("unit") != expected_unit:
            raise ConstraintRequestError(f"kind {kind} requires unit {expected_unit!r}")
        atoms = entry.get("atoms")
        if not isinstance(atoms, list) or len(atoms) != arity:
            raise ConstraintRequestError(f"kind {kind} needs exactly {arity} atoms")
        for index in atoms:
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < n_atoms:
                raise ConstraintRequestError(f"atom index {index!r} out of range")
        target = entry.get("target")
        if not isinstance(target, (int, float)) or isinstance(target, bool) or target <= 0:
            raise ConstraintRequestError("constraint target must be a positive number")


__all__ = [
    "ACP_OPT_CONFIG_FLAG",
    "ACP_WORKFLOW_NAME",
    "ConstraintRequestError",
    "DriverTarget",
    "XTC_REQUEST_SCHEMA",
    "atom_order_digest",
    "build_constraint_request",
    "geometry_digest",
    "request_sha256",
    "validate_constraint_request",
]
