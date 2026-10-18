"""Runtime input qualification for the PES-N predictor (contract ``pesn_input_v1``).

A :class:`PESNInput` carries only what the predictor is allowed to see at
inference time: the current geometry, elements/atom map, a plain-data endpoint
task graph with its provenance, a coordinate description, and the electronic
state / method / units.  Reference TS geometries, future IRC frames, labels,
and source-read callbacks are structurally absent — inputs are validated to
contain no such fields.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Final, Mapping, Sequence

import numpy as np

from pes2ts_core.pesn.contracts import PESNError
from pes2ts_core.utils.hashing import stable_json_dumps


def _sha256_text(text: str) -> str:
    """Hex sha256 of a UTF-8 string (local; utils.hashing stays untouched)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

INPUT_SCHEMA_VERSION: Final[str] = "pesn_input_v1"
TASK_GRAPH_SCHEMA_VERSION: Final[str] = "pesn_task_graph_v1"
COORDINATE_SPEC_SCHEMA_VERSION: Final[str] = "pesn_coordinate_spec_v1"

LENGTH_UNIT_ANGSTROM: Final[str] = "angstrom"
ANGLE_UNIT_RADIAN: Final[str] = "radian"
_SUPPORTED_LENGTH_UNITS: Final[frozenset[str]] = frozenset({LENGTH_UNIT_ANGSTROM})
_SUPPORTED_ANGLE_UNITS: Final[frozenset[str]] = frozenset({ANGLE_UNIT_RADIAN})
_SUPPORTED_EDIT_KINDS: Final[frozenset[str]] = frozenset(
    {"formed", "broken", "order_changed"}
)
_SUPPORTED_COORD_KINDS: Final[frozenset[str]] = frozenset({"distance", "angle", "dihedral"})
_SUPPORTED_COORD_ROLES: Final[frozenset[str]] = frozenset({"driver", "monitor"})

#: Input fields a runtime input must never carry (leakage guard).
FORBIDDEN_INPUT_FIELDS: Final[tuple[str, ...]] = (
    "reference_ts",
    "neighbor_frames",
    "future_frames",
    "labels",
    "signed_progress",
    "path_proximity",
    "teacher_direction",
    "coordinate_activity",
    "source_reader",
)


class InputRejected(PESNError):
    """Typed rejection of one input; ``code`` is machine-readable."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"input rejected [{code}]: {detail}")


def _require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise InputRejected(code, detail)


@dataclass(frozen=True, slots=True)
class TaskEdit:
    """One declared graph edit between the endpoint graphs (atom-row indices)."""

    kind: str  # formed | broken | order_changed
    i: int
    j: int
    reactant_order: float
    product_order: float

    def validate(self, n_atoms: int) -> None:
        _require(
            self.kind in _SUPPORTED_EDIT_KINDS,
            "TASK_GRAPH_INVALID",
            f"unknown edit kind {self.kind!r}",
        )
        _require(
            0 <= self.i < n_atoms and 0 <= self.j < n_atoms and self.i != self.j,
            "TASK_GRAPH_INVALID",
            f"edit {self.kind}({self.i},{self.j}) out of range or degenerate",
        )
        _require(
            math.isfinite(self.reactant_order) and math.isfinite(self.product_order),
            "TASK_GRAPH_INVALID",
            "edit bond orders must be finite",
        )


@dataclass(frozen=True, slots=True)
class BondTerm:
    """One bond of an endpoint graph (atom-row indices, bond order)."""

    i: int
    j: int
    order: float


@dataclass(frozen=True, slots=True)
class TaskGraph:
    """Plain-data endpoint task graph; produced by the topology authority."""

    schema_version: str
    atomic_numbers: tuple[int, ...]
    atom_map: tuple[int, ...]
    reactant_bonds: tuple[BondTerm, ...]
    product_bonds: tuple[BondTerm, ...]
    edits: tuple[TaskEdit, ...]
    source: str
    provenance: Mapping[str, str] = field(default_factory=dict)

    @property
    def n_atoms(self) -> int:
        return len(self.atomic_numbers)

    def validate(self) -> None:
        _require(
            self.schema_version == TASK_GRAPH_SCHEMA_VERSION,
            "SCHEMA_UNKNOWN",
            f"task graph schema {self.schema_version!r} is not supported",
        )
        _require(
            len(self.atomic_numbers) == len(self.atom_map) and self.n_atoms > 0,
            "TASK_GRAPH_INVALID",
            "atomic_numbers and atom_map must be equal length and non-empty",
        )
        for z in self.atomic_numbers:
            _require(
                isinstance(z, int) and 1 <= z <= 118,
                "ELEMENT_INVALID",
                f"atomic number {z!r} outside 1..118",
            )
        _require(
            sorted(self.atom_map) == list(range(1, self.n_atoms + 1)),
            "MAP_NOT_BIJECTION",
            "atom_map must be a bijection onto 1..n_atoms",
        )
        for bonds, side in ((self.reactant_bonds, "reactant"), (self.product_bonds, "product")):
            for term in bonds:
                _require(
                    0 <= term.i < self.n_atoms
                    and 0 <= term.j < self.n_atoms
                    and term.i != term.j
                    and math.isfinite(term.order),
                    "TASK_GRAPH_INVALID",
                    f"{side} bond ({term.i},{term.j},{term.order}) invalid",
                )
        for edit in self.edits:
            edit.validate(self.n_atoms)
        _require(bool(self.source), "TASK_GRAPH_INVALID", "task graph source is empty")

    def content_hash(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "atomic_numbers": list(self.atomic_numbers),
            "atom_map": list(self.atom_map),
            "reactant_bonds": [[t.i, t.j, t.order] for t in self.reactant_bonds],
            "product_bonds": [[t.i, t.j, t.order] for t in self.product_bonds],
            "edits": [
                [e.kind, e.i, e.j, e.reactant_order, e.product_order] for e in self.edits
            ],
            "source": self.source,
            "provenance": dict(sorted(self.provenance.items())),
        }
        return _sha256_text(stable_json_dumps(payload))


@dataclass(frozen=True, slots=True)
class CoordinateDef:
    """One graph-defined coordinate (drivers and monitors)."""

    kind: str  # distance | angle | dihedral
    atoms: tuple[int, ...]
    units: str
    role: str  # driver | monitor
    scale_a: float
    ref_q: float
    label: str

    def validate(self, n_atoms: int) -> None:
        expected_atoms = {"distance": 2, "angle": 3, "dihedral": 4}[self.kind] if self.kind in _SUPPORTED_COORD_KINDS else -1
        _require(
            self.kind in _SUPPORTED_COORD_KINDS,
            "COORD_SPEC_INVALID",
            f"unknown coordinate kind {self.kind!r}",
        )
        _require(
            len(self.atoms) == expected_atoms,
            "COORD_SPEC_INVALID",
            f"{self.kind} needs {expected_atoms} atoms, got {len(self.atoms)}",
        )
        for idx in self.atoms:
            _require(
                0 <= idx < n_atoms,
                "COORD_SPEC_INVALID",
                f"coordinate atom index {idx} out of range",
            )
        expected_unit = LENGTH_UNIT_ANGSTROM if self.kind == "distance" else ANGLE_UNIT_RADIAN
        _require(
            self.units == expected_unit,
            "UNITS_UNKNOWN",
            f"{self.kind} units must be {expected_unit!r}, got {self.units!r}",
        )
        _require(
            self.role in _SUPPORTED_COORD_ROLES,
            "COORD_SPEC_INVALID",
            f"unknown coordinate role {self.role!r}",
        )
        _require(
            math.isfinite(self.scale_a) and self.scale_a > 0,
            "COORD_SPEC_INVALID",
            f"scale_a must be positive finite, got {self.scale_a}",
        )
        _require(math.isfinite(self.ref_q), "COORD_SPEC_INVALID", "ref_q must be finite")


@dataclass(frozen=True, slots=True)
class CoordinateSpec:
    """Versioned set of graph-defined coordinates for one task graph."""

    schema_version: str
    coords: tuple[CoordinateDef, ...]
    spec_version: str

    def validate(self, n_atoms: int) -> None:
        _require(
            self.schema_version == COORDINATE_SPEC_SCHEMA_VERSION,
            "SCHEMA_UNKNOWN",
            f"coordinate spec schema {self.schema_version!r} is not supported",
        )
        _require(bool(self.coords), "COORD_SPEC_INVALID", "coordinate spec is empty")
        for coord in self.coords:
            coord.validate(n_atoms)
        _require(bool(self.spec_version), "COORD_SPEC_INVALID", "spec_version is empty")

    def content_hash(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "spec_version": self.spec_version,
            "coords": [
                {
                    "kind": c.kind,
                    "atoms": list(c.atoms),
                    "units": c.units,
                    "role": c.role,
                    "scale_a": c.scale_a,
                    "ref_q": c.ref_q,
                    "label": c.label,
                }
                for c in self.coords
            ],
        }
        return _sha256_text(stable_json_dumps(payload))


@dataclass(frozen=True, slots=True)
class PESNInput:
    """Qualified runtime input; validated at construction by :meth:`build`."""

    schema_version: str
    input_id: str
    atomic_numbers: tuple[int, ...]
    coordinates: np.ndarray  # (n_atoms, 3) float64, angstrom
    units: str
    charge: int | None
    multiplicity: int | None
    method: str | None
    task_graph: TaskGraph
    coordinate_spec: CoordinateSpec
    geometry_hash: str

    @property
    def n_atoms(self) -> int:
        return len(self.atomic_numbers)

    @property
    def task_graph_hash(self) -> str:
        return self.task_graph.content_hash()

    @property
    def coordinate_spec_hash(self) -> str:
        return self.coordinate_spec.content_hash()

    @property
    def input_hash(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "input_id": self.input_id,
            "atomic_numbers": list(self.atomic_numbers),
            "geometry_hash": self.geometry_hash,
            "units": self.units,
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "method": self.method,
            "task_graph_hash": self.task_graph_hash,
            "coordinate_spec_hash": self.coordinate_spec_hash,
        }
        return _sha256_text(stable_json_dumps(payload))


def geometry_content_hash(coordinates: np.ndarray) -> str:
    """Stable hash of a geometry (angstrom, rounded to 1e-6 to bound float noise)."""
    rounded = np.round(np.asarray(coordinates, dtype=np.float64), 6)
    payload = [[float(v) for v in row] for row in rounded.tolist()]
    return _sha256_text(stable_json_dumps(payload))


def build_pesn_input(
    *,
    input_id: str,
    task_graph: TaskGraph,
    coordinate_spec: CoordinateSpec,
    coordinates: Sequence[Sequence[float]] | np.ndarray,
    units: str = LENGTH_UNIT_ANGSTROM,
    charge: int | None = None,
    multiplicity: int | None = None,
    method: str | None = None,
) -> PESNInput:
    """Validate and freeze one :class:`PESNInput`.

    Raises :class:`InputRejected` with a machine-readable code on any
    qualification failure; nothing is guessed or defaulted silently.
    """
    _require(bool(input_id), "INPUT_ID_MISSING", "input_id must be non-empty")
    task_graph.validate()
    n_atoms = task_graph.n_atoms
    coordinate_spec.validate(n_atoms)
    _require(
        units in _SUPPORTED_LENGTH_UNITS,
        "UNITS_UNKNOWN",
        f"unsupported length unit {units!r}",
    )
    coords = np.asarray(coordinates, dtype=np.float64)
    _require(
        coords.shape == (n_atoms, 3),
        "ATOM_COUNT_MISMATCH",
        f"coordinates shape {coords.shape} does not match {(n_atoms, 3)}",
    )
    _require(
        bool(np.isfinite(coords).all()),
        "NONFINITE_COORDINATES",
        "coordinates contain non-finite values",
    )
    return PESNInput(
        schema_version=INPUT_SCHEMA_VERSION,
        input_id=input_id,
        atomic_numbers=tuple(task_graph.atomic_numbers),
        coordinates=coords,
        units=units,
        charge=charge,
        multiplicity=multiplicity,
        method=method,
        task_graph=task_graph,
        coordinate_spec=coordinate_spec,
        geometry_hash=geometry_content_hash(coords),
    )


__all__ = [
    "ANGLE_UNIT_RADIAN",
    "COORDINATE_SPEC_SCHEMA_VERSION",
    "CoordinateDef",
    "CoordinateSpec",
    "FORBIDDEN_INPUT_FIELDS",
    "INPUT_SCHEMA_VERSION",
    "InputRejected",
    "LENGTH_UNIT_ANGSTROM",
    "PESNInput",
    "TASK_GRAPH_SCHEMA_VERSION",
    "TaskEdit",
    "TaskGraph",
    "BondTerm",
    "build_pesn_input",
    "geometry_content_hash",
]
