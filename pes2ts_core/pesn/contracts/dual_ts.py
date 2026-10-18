"""Dual-endpoint TS teacher contracts (WP0a, v4 plan; contract doc v2).

状态：WP0a 实现冻结；对应 docs/contracts/PES2TS_双端TS教师与相容转移合同_v2.md
层级：G0/G1增量接口、G3.P教师、G4.0研究转移
邻接：取代关系为"新任务另版本"，不触碰旧 teacher/direct/control v1/v2 产物。

Typed, validated data contracts for the dual-endpoint TS teacher study
(schema names frozen exactly as registered in the contract doc §1):

* ``pesn_dual_endpoint_task_v2``   — unordered complete endpoint graph pair
* ``pesn_dual_ts_teacher_v2``      — branch/episode/state/label records
* ``pesn_dual_ts_features_v3``     — exchange-symmetric feature spec
  (v4.1 WP1 observation-only 5-channel node scalars; legacy v2 kept
  registered for read compatibility)
* ``pesn_dual_ts_action_v2``       — MOVE/STOP_AND_REFINE/ABORT/fallback
* ``pes2ts_dual_ts_transition_v2`` — explicit transition with boundary masks
* ``pes2ts_dual_ts_reward_v2``     — logical cost + real outcome protocol
* ``pesn_dual_ts_bundle_v2``       — run bundle identity bindings

Boundary statements (contract doc §2/§3):

* Each endpoint graph is stored as a COMPLETE unit. Per-edge bond orders are
  never merged or sorted across endpoints; the unordered identity is realized
  by canonically ordering the two endpoint *units* by content digest.
* ``branch_id`` is traceability only — never a network input. Endpoint labels
  ``A``/``B`` are identity numbers without direction semantics.
* Missing observations are ``null`` + registered reason — never zero-filled.
* A file-slice boundary is NOT a terminal: the transition keeps referencing
  the real successor. Only a real terminal carries ``terminal=True``.
* Unknown reward/outcome/return are explicit ``None`` + reason — never 0.

Implementation discipline mirrors ``pes2ts_core.pesn.contracts.direct``:
typed frozen dataclasses, loud typed rejections, no imports from
training/data/backend modules, stdlib + numpy only, content hashes computed
over canonical JSON (sorted keys, compact separators) — never over Python
``repr``.
"""
from __future__ import annotations

import enum
import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final, Mapping, Sequence

import numpy as np

from pes2ts_core.utils.hashing import JSONValue, sha256_bytes, stable_json_dumps

# ---------------------------------------------------------------------------
# Frozen schema names (contract doc §1; exact strings)
# ---------------------------------------------------------------------------

TASK_SCHEMA_VERSION: Final[str] = "pesn_dual_endpoint_task_v2"
TEACHER_SCHEMA_VERSION: Final[str] = "pesn_dual_ts_teacher_v2"
#: v4.1 WP1: observation-only 5-channel node scalars (7 → 5).
FEATURES_SCHEMA_VERSION: Final[str] = "pesn_dual_ts_features_v3"
#: Legacy v2 (7-channel) — kept REGISTERED so old products stay readable.
FEATURES_SCHEMA_VERSION_V2: Final[str] = "pesn_dual_ts_features_v2"
ACTION_SCHEMA_VERSION: Final[str] = "pesn_dual_ts_action_v2"
TRANSITION_SCHEMA_VERSION: Final[str] = "pes2ts_dual_ts_transition_v2"
REWARD_SCHEMA_VERSION: Final[str] = "pes2ts_dual_ts_reward_v2"
BUNDLE_SCHEMA_VERSION: Final[str] = "pesn_dual_ts_bundle_v2"

#: Geometric/label consistency tolerance (contract doc §5 acceptance).
GEOM_TOL: Final[float] = 1e-6

#: Frozen placeholder binding for the remaining mass-weighted arc-length
#: metric ``ell``. The exact mass metric (per-pair reduced masses vs unit
#: masses) is a WP0/WP1 freeze decision; the version string binds whatever
#: metric produced ``ell`` so records remain traceable once frozen.
ARC_LENGTH_METRIC_VERSION: Final[str] = "mass_weighted_arc_v0_PLACEHOLDER"

#: Radial latent transform identity (contract doc §4): z = tanh(||u||)u/||u||.
RADIAL_TRANSFORM_VERSION: Final[str] = "radial_tanh_v0"


class DualTSContractError(ValueError):
    """Typed rejection for dual-TS contract violations."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise DualTSContractError(code, detail)


def _sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def content_digest_of(payload: Mapping[str, JSONValue]) -> str:
    """SHA-256 over canonical JSON (sorted keys, compact, UTF-8 literal).

    Stable across runs and across dict insertion order; hashes actual
    content, never Python ``repr``.
    """
    return _sha256_text(stable_json_dumps(dict(payload)))


def _require_sha256_hex(value: str, code: str, detail: str) -> None:
    _require(
        isinstance(value, str) and len(value) == 64
        and all(c in "0123456789abcdef" for c in value),
        code, detail,
    )


def radial_tanh_transform(u: np.ndarray) -> np.ndarray:
    """Per-atom radial mapping ``z = tanh(||u||) * u / ||u||``.

    The zero vector uses the continuous limit ``z -> 0`` so the transform is
    smooth at the origin (contract doc §4).
    """
    u = np.asarray(u, dtype=float)
    norms = np.linalg.norm(u, axis=-1, keepdims=True)
    # avoid 0/0: where norm==0 the factor tanh(norm)/norm -> 1 (limit)
    factor = np.where(norms > 0.0, np.tanh(norms) / np.where(norms > 0.0, norms, 1.0), 1.0)
    return u * factor


# ---------------------------------------------------------------------------
# Endpoint graph unit: edges + bond orders + stereo with known-masks
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EndpointBond:
    """One edge of ONE endpoint. The bond order is bound to this edge and
    this endpoint only — it is never merged or sorted across endpoints."""

    i: int
    j: int
    bond_order: float
    bond_order_known: bool = True
    stereo: str | None = None          # e.g. "E", "Z", "R", "S"
    stereo_known: bool = False

    def validate(self, n_atoms: int) -> None:
        _require(
            0 <= self.i < n_atoms and 0 <= self.j < n_atoms,
            "BOND_INDEX_INVALID",
            f"bond ({self.i},{self.j}) outside 0..{n_atoms - 1}",
        )
        _require(self.i != self.j, "BOND_INDEX_INVALID", "self-bond not allowed")
        _require(self.i < self.j, "BOND_INDEX_INVALID", "bonds must store i < j")
        _require(
            float(self.bond_order) > 0.0,
            "BOND_ORDER_INVALID", "bond_order must be > 0",
        )
        _require(
            self.stereo is None or self.stereo_known,
            "STEREO_UNMASKED",
            f"stereo {self.stereo!r} set while stereo_known is False",
        )

    def to_payload(self) -> dict[str, JSONValue]:
        return {
            "i": int(self.i),
            "j": int(self.j),
            "bond_order": float(self.bond_order),
            "bond_order_known": bool(self.bond_order_known),
            "stereo": self.stereo,
            "stereo_known": bool(self.stereo_known),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, JSONValue]) -> "EndpointBond":
        # Contract doc §2: bond orders must stay attached to their edge and
        # their endpoint. A detached/sorted per-edge bond-order LIST (the
        # classic flattening that loses the complete dual-graph relation) is
        # rejected loudly instead of being silently re-associated.
        _require(
            "bond_orders" not in payload and "bond_order" in payload,
            "BOND_ORDER_FORMAT_INVALID",
            "per-edge bond-order list (sorted/merged across edges or "
            "endpoints) is not accepted; store one scalar bond_order per "
            "edge inside its complete endpoint unit",
        )
        value = payload["bond_order"]
        _require(
            isinstance(value, (int, float)) and not isinstance(value, bool),
            "BOND_ORDER_FORMAT_INVALID",
            f"bond_order must be a scalar, got {type(value).__name__}",
        )
        return cls(
            i=int(payload["i"]),
            j=int(payload["j"]),
            bond_order=float(value),
            bond_order_known=bool(payload.get("bond_order_known", True)),
            stereo=(None if payload.get("stereo") is None else str(payload["stereo"])),
            stereo_known=bool(payload.get("stereo_known", False)),
        )


@dataclass(frozen=True, slots=True)
class EndpointGraph:
    """One COMPLETE endpoint molecular graph (no coordinates by design: the
    task record fixes graph identity; geometry lives in teacher states).

    Carries no A/B label: endpoint identity is positional only, and the pair
    is made unordered by :func:`canonical_endpoint_order`.
    """

    atomic_numbers: np.ndarray            # (N,) int
    atom_map_ids: tuple[str, ...]         # N labels, shared across endpoints
    bonds: tuple[EndpointBond, ...]

    def validate(self) -> None:
        z = np.asarray(self.atomic_numbers)
        _require(z.ndim == 1 and z.size > 0, "ATOM_COUNT_INVALID",
                 "atomic_numbers must be (N,) with N > 0")
        _require(len(self.atom_map_ids) == z.size, "MAP_MISMATCH",
                 "atom_map_ids length mismatch")
        _require(len(set(self.atom_map_ids)) == z.size, "MAP_DUPLICATE",
                 "atom_map_ids must be unique within an endpoint")
        for bond in self.bonds:
            bond.validate(int(z.size))

    def to_payload(self) -> dict[str, JSONValue]:
        # Bonds are canonically sorted by (i, j) — this never re-assigns a
        # bond order to a different edge because each bond carries its own
        # scalar order inline.
        bonds = sorted(self.bonds, key=lambda b: (b.i, b.j))
        return {
            "atomic_numbers": np.asarray(self.atomic_numbers).astype(int).tolist(),
            "atom_map_ids": list(self.atom_map_ids),
            "bonds": [b.to_payload() for b in bonds],
        }

    def content_digest(self) -> str:
        return content_digest_of(self.to_payload())

    @classmethod
    def from_payload(cls, payload: Mapping[str, JSONValue]) -> "EndpointGraph":
        graph = cls(
            atomic_numbers=np.asarray(payload["atomic_numbers"], dtype=int),
            atom_map_ids=tuple(str(m) for m in payload["atom_map_ids"]),
            bonds=tuple(
                EndpointBond.from_payload(dict(b)) for b in payload["bonds"]
            ),
        )
        graph.validate()
        return graph


def canonical_endpoint_order(
    a: EndpointGraph, b: EndpointGraph,
) -> tuple[EndpointGraph, EndpointGraph]:
    """Canonical, content-derived order of the unordered endpoint pair.

    The pair {G_A, G_B} is ordered by each endpoint unit's content digest
    (ties broken by canonical JSON text, which handles byte-identical
    duplicates deterministically). Swapping the caller's A/B arguments never
    changes the returned order, so reaction identity, partition keys, and the
    task content digest are invariant under the swap — while each endpoint
    graph remains a complete, un-merged unit.
    """
    ka = (a.content_digest(), stable_json_dumps(a.to_payload()))
    kb = (b.content_digest(), stable_json_dumps(b.to_payload()))
    return (a, b) if ka <= kb else (b, a)


def partition_by_reaction_group(
    tasks: Sequence["DualEndpointTask"],
) -> dict[str, list["DualEndpointTask"]]:
    """Group tasks by ``reaction_group_id`` — duplicate reactions of the same
    physical reaction share one group key (contract doc §2)."""
    groups: dict[str, list[DualEndpointTask]] = {}
    for task in tasks:
        groups.setdefault(task.reaction_group_id, []).append(task)
    return groups


# ---------------------------------------------------------------------------
# Task record: unordered complete endpoint pair + reaction identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DualEndpointTask:
    """``pesn_dual_endpoint_task_v2``: the unordered full endpoint graph pair
    {G_A, G_B} plus reaction identity, constraints, and provenance."""

    reaction_id: str
    reaction_group_id: str
    endpoints: tuple[EndpointGraph, EndpointGraph]
    charge: int = 0
    multiplicity: int = 1
    constraints: Mapping[str, JSONValue] = field(default_factory=dict)
    source: str = ""
    method_id: str = ""
    eligibility: Mapping[str, JSONValue] = field(default_factory=dict)
    schema_version: str = TASK_SCHEMA_VERSION

    def validate(self) -> None:
        _require(self.schema_version == TASK_SCHEMA_VERSION, "SCHEMA_UNKNOWN",
                 f"expected {TASK_SCHEMA_VERSION}")
        _require(bool(self.reaction_id), "ID_MISSING", "reaction_id required")
        _require(bool(self.reaction_group_id), "ID_MISSING",
                 "reaction_group_id required")
        _require(len(self.endpoints) == 2, "ENDPOINT_COUNT_INVALID",
                 "exactly two endpoint graphs required")
        for endpoint in self.endpoints:
            endpoint.validate()
        n0 = int(np.asarray(self.endpoints[0].atomic_numbers).size)
        n1 = int(np.asarray(self.endpoints[1].atomic_numbers).size)
        _require(n0 == n1, "ENDPOINT_ATOM_MISMATCH",
                 f"endpoints have different atom counts ({n0} vs {n1})")
        _require(
            sorted(self.endpoints[0].atom_map_ids)
            == sorted(self.endpoints[1].atom_map_ids),
            "ENDPOINT_MAP_MISMATCH",
            "endpoints must share the same atom mapping",
        )
        _require(self.multiplicity >= 1, "MULTIPLICITY_INVALID",
                 "multiplicity must be >= 1")
        _require(bool(self.method_id), "ID_MISSING", "method_id required")
        for key in self.constraints:
            _require(isinstance(key, str), "CONSTRAINT_KEY_INVALID",
                     "constraint keys must be strings")

    def canonical_payload(self) -> dict[str, JSONValue]:
        """Payload with the endpoint pair serialized in canonical (content
        digest) order — the swap-invariant serialization."""
        first, second = canonical_endpoint_order(*self.endpoints)
        return {
            "schema_version": self.schema_version,
            "reaction_id": self.reaction_id,
            "reaction_group_id": self.reaction_group_id,
            "endpoints": [first.to_payload(), second.to_payload()],
            "charge": int(self.charge),
            "multiplicity": int(self.multiplicity),
            "constraints": dict(self.constraints),
            "source": self.source,
            "method_id": self.method_id,
            "eligibility": dict(self.eligibility),
        }

    def content_digest(self) -> str:
        return content_digest_of(self.canonical_payload())

    def partition_key(self) -> str:
        """Duplicate reactions group by ``reaction_group_id``."""
        return self.reaction_group_id

    @classmethod
    def from_payload(cls, payload: Mapping[str, JSONValue]) -> "DualEndpointTask":
        endpoints = payload["endpoints"]
        _require(isinstance(endpoints, list) and len(endpoints) == 2,
                 "ENDPOINT_COUNT_INVALID", "endpoints must be a pair")
        task = cls(
            reaction_id=str(payload["reaction_id"]),
            reaction_group_id=str(payload["reaction_group_id"]),
            endpoints=(
                EndpointGraph.from_payload(dict(endpoints[0])),
                EndpointGraph.from_payload(dict(endpoints[1])),
            ),
            charge=int(payload.get("charge", 0)),
            multiplicity=int(payload.get("multiplicity", 1)),
            constraints=dict(payload.get("constraints", {})),
            source=str(payload.get("source", "")),
            method_id=str(payload.get("method_id", "")),
            eligibility=dict(payload.get("eligibility", {})),
            schema_version=str(payload.get("schema_version", TASK_SCHEMA_VERSION)),
        )
        task.validate()
        return task


# ---------------------------------------------------------------------------
# Teacher records: branches, episodes, states, observations, offline labels
# ---------------------------------------------------------------------------

OBS_STATUS_OK: Final[str] = "ok"
OBS_STATUS_MISSING: Final[str] = "missing"

OBS_MISSING_REASONS: Final[frozenset[str]] = frozenset({
    "teacher_frame_isolated",     # teacher frame excluded from training inputs
    "computation_missing",
    "observation_not_exported",
    "censored_missing",
})

BOUNDARY_TEACHER_ENDPOINT: Final[str] = "teacher_endpoint"
BOUNDARY_TYPES: Final[frozenset[str]] = frozenset({
    BOUNDARY_TEACHER_ENDPOINT,    # reference path boundary — not success
    "data_boundary",              # file-slice boundary — NOT terminal
    "budget_exhausted",
    "external_censored",
    "computation_missing",
})


@dataclass(frozen=True, slots=True)
class ObservationRef:
    """Observation referenced by content identity.

    ``status="missing"`` stores ``None`` + a registered reason — never a
    zero-filled placeholder.
    """

    status: str                              # "ok" | "missing"
    missing_reason: str | None = None
    energy_hartree: float | None = None
    gradient_hartree_per_angstrom: np.ndarray | None = None  # (N,3)
    method_id: str = ""
    software: str = ""
    settings_digest: str = ""                # content digest of settings

    def validate(self) -> None:
        _require(self.status in (OBS_STATUS_OK, OBS_STATUS_MISSING),
                 "OBS_STATUS_UNKNOWN", f"status {self.status!r} not registered")
        if self.status == OBS_STATUS_MISSING:
            _require(self.energy_hartree is None
                     and self.gradient_hartree_per_angstrom is None,
                     "OBS_ZERO_FILLED",
                     "missing observation must carry null values, not zeros")
            _require(self.missing_reason in OBS_MISSING_REASONS,
                     "OBS_REASON_UNKNOWN",
                     f"missing_reason {self.missing_reason!r} not registered")
            return
        _require(self.missing_reason is None, "OBS_REASON_CONFLICT",
                 "present observation carries no missing_reason")
        _require(self.energy_hartree is not None, "OBS_ENERGY_MISSING",
                 "status ok requires energy_hartree")
        _require(self.gradient_hartree_per_angstrom is not None,
                 "OBS_GRADIENT_MISSING", "status ok requires gradient")
        _require(bool(np.isfinite(self.energy_hartree)), "OBS_NONFINITE",
                 "energy must be finite")
        g = np.asarray(self.gradient_hartree_per_angstrom, dtype=float)
        _require(g.ndim == 2 and g.shape[1] == 3, "OBS_SHAPE_INVALID",
                 f"gradient must be (N,3), got {g.shape}")
        _require(bool(np.isfinite(g).all()), "OBS_NONFINITE",
                 "gradient contains NaN/Inf")
        # Frozen zero-fill marker: an exactly-zero energy paired with an
        # exactly-all-zero gradient is the legacy "missing as zeros" pattern,
        # never a real computation — rejected instead of entering training.
        _require(
            not (float(self.energy_hartree) == 0.0 and not g.any()),
            "OBS_ZERO_FILLED",
            "zero-filled observation detected; missing data must be "
            "null + reason, never zeros",
        )

    def to_payload(self) -> dict[str, JSONValue]:
        grad = self.gradient_hartree_per_angstrom
        return {
            "status": self.status,
            "missing_reason": self.missing_reason,
            "energy_hartree": (None if self.energy_hartree is None
                               else float(self.energy_hartree)),
            "gradient": (None if grad is None
                         else np.asarray(grad, dtype=float).round(10).tolist()),
            "method_id": self.method_id,
            "software": self.software,
            "settings_digest": self.settings_digest,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, JSONValue]) -> "ObservationRef":
        grad = payload.get("gradient")
        obs = cls(
            status=str(payload["status"]),
            missing_reason=(None if payload.get("missing_reason") is None
                            else str(payload["missing_reason"])),
            energy_hartree=(None if payload.get("energy_hartree") is None
                            else float(payload["energy_hartree"])),
            gradient_hartree_per_angstrom=(
                None if grad is None else np.asarray(grad, dtype=float)),
            method_id=str(payload.get("method_id", "")),
            software=str(payload.get("software", "")),
            settings_digest=str(payload.get("settings_digest", "")),
        )
        obs.validate()
        return obs


@dataclass(frozen=True, slots=True)
class BranchRecord:
    """Branch of one reaction. ``branch_id`` is traceability only — it is
    never a network input. ``endpoint_label`` is an identity number ("A"/"B")
    without direction semantics."""

    branch_id: str
    endpoint_label: str          # "A" | "B", identity only
    reaction_id: str
    episode_ids: tuple[str, ...] = ()

    def validate(self) -> None:
        _require(bool(self.branch_id), "ID_MISSING", "branch_id required")
        _require(self.endpoint_label in ("A", "B"), "ENDPOINT_LABEL_INVALID",
                 "endpoint_label is the identity 'A' or 'B' only")
        _require(bool(self.reaction_id), "ID_MISSING", "reaction_id required")


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """One episode: real start geometry reference plus frozen environment
    bindings. Several episodes may share one group."""

    episode_id: str
    branch_id: str
    reaction_id: str
    start_state_id: str
    budget_version: str
    environment_version: str
    reward_version: str

    def validate(self) -> None:
        for name in ("episode_id", "branch_id", "reaction_id", "start_state_id",
                     "budget_version", "environment_version", "reward_version"):
            _require(bool(getattr(self, name)), "ID_MISSING",
                     f"{name} required")


@dataclass(frozen=True, slots=True)
class TeacherState:
    """Teacher state: real geometry in Å, full identity, observation
    reference by content identity, boundary type + missing reason."""

    state_id: str
    episode_id: str
    branch_id: str
    step_index: int
    atomic_numbers: np.ndarray            # (N,) int
    coordinates: np.ndarray               # (N,3) float, angstrom
    atom_map_ids: tuple[str, ...]
    method_id: str
    electronic_state: str = "S0"
    multiplicity: int = 1
    observation: ObservationRef | None = None
    observation_missing_reason: str | None = None
    boundary_type: str | None = None      # None | BOUNDARY_TYPES member
    next_state_id: str | None = None

    def validate(self) -> None:
        for name in ("state_id", "episode_id", "branch_id", "method_id"):
            _require(bool(getattr(self, name)), "ID_MISSING", f"{name} required")
        _require(self.step_index >= 0, "STEP_INVALID", "step_index must be >= 0")
        z = np.asarray(self.atomic_numbers)
        x = np.asarray(self.coordinates, dtype=float)
        _require(z.ndim == 1 and z.size > 0, "ATOM_COUNT_INVALID",
                 "atomic_numbers must be (N,)")
        _require(x.shape == (z.size, 3), "ATOM_COUNT_MISMATCH",
                 f"coordinates {x.shape} incompatible with {z.size} atoms")
        _require(bool(np.isfinite(x).all()), "NONFINITE_COORDINATES",
                 "coordinates contain NaN/Inf")
        _require(len(self.atom_map_ids) == z.size, "MAP_MISMATCH",
                 "atom_map_ids length mismatch")
        _require(self.multiplicity >= 1, "MULTIPLICITY_INVALID",
                 "multiplicity must be >= 1")
        if self.boundary_type is not None:
            _require(self.boundary_type in BOUNDARY_TYPES, "BOUNDARY_UNKNOWN",
                     f"boundary_type {self.boundary_type!r} not registered")
        if self.boundary_type == BOUNDARY_TEACHER_ENDPOINT:
            # Teacher endpoint marks the reference path boundary: no
            # successor may exist beyond it (no cross-TS chaining).
            _require(self.next_state_id is None, "TEACHER_ENDPOINT_WITH_SUCCESSOR",
                     "a teacher-endpoint TS state is terminal and carries no "
                     "successor")
        # Missing observation = null + reason (never zeros).
        if self.observation is None:
            _require(self.observation_missing_reason in OBS_MISSING_REASONS,
                     "OBS_REASON_UNKNOWN",
                     "absent observation requires a registered "
                     "observation_missing_reason")
        else:
            _require(self.observation_missing_reason is None,
                     "OBS_REASON_CONFLICT",
                     "present observation must not carry missing_reason")
            self.observation.validate()

    def to_payload(self) -> dict[str, JSONValue]:
        return {
            "state_id": self.state_id,
            "episode_id": self.episode_id,
            "branch_id": self.branch_id,
            "step_index": int(self.step_index),
            "atomic_numbers": np.asarray(self.atomic_numbers).astype(int).tolist(),
            "coordinates": np.asarray(self.coordinates, dtype=float).round(10).tolist(),
            "atom_map_ids": list(self.atom_map_ids),
            "method_id": self.method_id,
            "electronic_state": self.electronic_state,
            "multiplicity": int(self.multiplicity),
            "observation": (None if self.observation is None
                            else self.observation.to_payload()),
            "observation_missing_reason": self.observation_missing_reason,
            "boundary_type": self.boundary_type,
            "next_state_id": self.next_state_id,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, JSONValue]) -> "TeacherState":
        obs = payload.get("observation")
        state = cls(
            state_id=str(payload["state_id"]),
            episode_id=str(payload["episode_id"]),
            branch_id=str(payload["branch_id"]),
            step_index=int(payload["step_index"]),
            atomic_numbers=np.asarray(payload["atomic_numbers"], dtype=int),
            coordinates=np.asarray(payload["coordinates"], dtype=float),
            atom_map_ids=tuple(str(m) for m in payload["atom_map_ids"]),
            method_id=str(payload["method_id"]),
            electronic_state=str(payload.get("electronic_state", "S0")),
            multiplicity=int(payload.get("multiplicity", 1)),
            observation=(None if obs is None
                         else ObservationRef.from_payload(dict(obs))),
            observation_missing_reason=(
                None if payload.get("observation_missing_reason") is None
                else str(payload["observation_missing_reason"])),
            boundary_type=(None if payload.get("boundary_type") is None
                           else str(payload["boundary_type"])),
            next_state_id=(None if payload.get("next_state_id") is None
                           else str(payload["next_state_id"])),
        )
        state.validate()
        return state


@dataclass(frozen=True, slots=True)
class TeacherLabel:
    """Offline label for one teacher state.

    ``action_delta`` a = X_next − X (Å); ``ts_displacement`` d = X* − X (Å);
    ``remaining_arc_length`` ell is the remaining mass-weighted arc length
    bound to :data:`ARC_LENGTH_METRIC_VERSION`. ``action_delta`` is ``None``
    only at the terminal teacher-endpoint state (no successor exists).
    """

    state_id: str
    ts_displacement: np.ndarray            # (N,3) Å
    remaining_arc_length: float
    teacher_endpoint: bool = False
    action_delta: np.ndarray | None = None  # (N,3) Å

    def validate(self) -> None:
        _require(bool(self.state_id), "ID_MISSING", "state_id required")
        d = np.asarray(self.ts_displacement, dtype=float)
        _require(d.ndim == 2 and d.shape[1] == 3, "LABEL_SHAPE_INVALID",
                 f"ts_displacement must be (N,3), got {d.shape}")
        _require(bool(np.isfinite(d).all()), "NONFINITE_LABEL",
                 "ts_displacement contains NaN/Inf")
        _require(
            float(self.remaining_arc_length) >= 0.0
            and bool(np.isfinite(self.remaining_arc_length)),
            "ARC_LENGTH_INVALID", "remaining_arc_length must be finite >= 0",
        )
        if self.action_delta is not None:
            a = np.asarray(self.action_delta, dtype=float)
            _require(a.shape == d.shape, "LABEL_SHAPE_MISMATCH",
                     "action_delta shape must match ts_displacement")
            _require(bool(np.isfinite(a).all()), "NONFINITE_LABEL",
                     "action_delta contains NaN/Inf")


@dataclass(frozen=True, slots=True)
class DualTSTeacher:
    """``pesn_dual_ts_teacher_v2``: the full branch/episode/state/label table
    for one reaction, with all cross-record consistency rules enforced."""

    reaction_id: str
    reaction_group_id: str
    branches: tuple[BranchRecord, ...]
    episodes: tuple[EpisodeRecord, ...]
    states: tuple[TeacherState, ...]
    labels: tuple[TeacherLabel, ...]
    arc_length_metric_version: str = ARC_LENGTH_METRIC_VERSION
    schema_version: str = TEACHER_SCHEMA_VERSION

    # -- structural helpers -------------------------------------------------
    def _state_by_id(self) -> dict[str, TeacherState]:
        return {s.state_id: s for s in self.states}

    def _label_by_state(self) -> dict[str, TeacherLabel]:
        return {lab.state_id: lab for lab in self.labels}

    # -- validation ----------------------------------------------------------
    def validate(self) -> None:
        _require(self.schema_version == TEACHER_SCHEMA_VERSION, "SCHEMA_UNKNOWN",
                 f"expected {TEACHER_SCHEMA_VERSION}")
        _require(bool(self.reaction_id) and bool(self.reaction_group_id),
                 "ID_MISSING", "reaction ids required")
        _require(bool(self.branches), "BRANCH_MISSING", "at least one branch")
        _require(bool(self.episodes), "EPISODE_MISSING", "at least one episode")
        _require(bool(self.states), "STATE_MISSING", "at least one state")
        _require(bool(self.arc_length_metric_version), "ID_MISSING",
                 "arc_length_metric_version binding required")

        branch_ids = {b.branch_id for b in self.branches}
        _require(len(branch_ids) == len(self.branches), "BRANCH_DUPLICATE",
                 "branch_id must be unique within a teacher record")
        for branch in self.branches:
            branch.validate()
            _require(branch.reaction_id == self.reaction_id,
                     "REACTION_MISMATCH",
                     f"branch {branch.branch_id} references another reaction")

        episode_ids = {e.episode_id for e in self.episodes}
        _require(len(episode_ids) == len(self.episodes), "EPISODE_DUPLICATE",
                 "episode_id must be unique")
        episode_branch: dict[str, str] = {}
        for episode in self.episodes:
            episode.validate()
            _require(episode.reaction_id == self.reaction_id, "REACTION_MISMATCH",
                     f"episode {episode.episode_id} references another reaction")
            _require(episode.branch_id in branch_ids, "BRANCH_UNKNOWN",
                     f"episode {episode.episode_id} references unknown branch")
            episode_branch[episode.episode_id] = episode.branch_id

        states = self._state_by_id()
        _require(len(states) == len(self.states), "STATE_DUPLICATE",
                 "state_id must be unique within a teacher record")
        for state in self.states:
            state.validate()
            _require(state.episode_id in episode_ids, "EPISODE_UNKNOWN",
                     f"state {state.state_id} references unknown episode")
            _require(episode_branch[state.episode_id] == state.branch_id,
                     "BRANCH_MISMATCH",
                     f"state {state.state_id} branch disagrees with its episode")

        labels = self._label_by_state()
        _require(len(labels) == len(self.labels), "LABEL_DUPLICATE",
                 "one label per state_id only")
        _require(set(labels) == set(states), "LABEL_COVERAGE_INVALID",
                 "labels must cover exactly the registered states")
        for label in self.labels:
            label.validate()

        # Boundary-flag agreement first, then endpoint-specific label
        # invariants, then the successor chain algebra (order matters: the
        # most specific rejection code must win).
        for state in self.states:
            label = labels[state.state_id]
            _require(
                label.teacher_endpoint
                == (state.boundary_type == BOUNDARY_TEACHER_ENDPOINT),
                "TEACHER_ENDPOINT_FLAG_MISMATCH",
                f"state {state.state_id}: teacher_endpoint flag must agree "
                "with boundary_type",
            )
        # Teacher-endpoint labels: zero remaining displacement and no action.
        for state in self.states:
            if state.boundary_type != BOUNDARY_TEACHER_ENDPOINT:
                continue
            label = labels[state.state_id]
            _require(state.next_state_id is None,
                     "TEACHER_ENDPOINT_WITH_SUCCESSOR",
                     f"teacher-endpoint state {state.state_id} carries a "
                     "successor")
            _require(label.action_delta is None,
                     "TEACHER_ENDPOINT_WITH_ACTION",
                     f"teacher-endpoint state {state.state_id} must not "
                     "carry an action (no successor exists)")
            d_norm = float(np.max(np.abs(label.ts_displacement)))
            _require(d_norm <= GEOM_TOL, "TEACHER_ENDPOINT_DISPLACED",
                     f"teacher-endpoint state {state.state_id} retains "
                     f"remaining displacement {d_norm:.3e}")
        # Cross-record chain consistency (contract doc §2/§5).
        for state in self.states:
            if state.next_state_id is None:
                # A state without successor is terminal only at the teacher
                # endpoint boundary; data boundaries keep referencing the
                # real successor elsewhere (transition layer).
                continue
            label = labels[state.state_id]
            _require(state.next_state_id in states, "SUCCESSOR_UNKNOWN",
                     f"state {state.state_id} references unknown successor")
            nxt = states[state.next_state_id]
            _require(nxt.branch_id == state.branch_id, "CROSS_BRANCH_SUCCESSOR",
                     f"successor of {state.state_id} leaves its branch "
                     "(branches are never chained across TS)")
            if nxt.episode_id == state.episode_id:
                _require(nxt.step_index == state.step_index + 1,
                         "STEP_CONTINUITY_INVALID",
                         f"successor of {state.state_id} skips a step")
            _require(label.action_delta is not None, "ACTION_DELTA_MISSING",
                     f"non-terminal state {state.state_id} requires "
                     "action_delta")
            a = np.asarray(label.action_delta, dtype=float)
            # X + a = X_next within 1e-6.
            x_next_err = float(
                np.max(np.abs(
                    np.asarray(state.coordinates, dtype=float) + a
                    - np.asarray(nxt.coordinates, dtype=float))))
            _require(x_next_err <= GEOM_TOL, "DISPLACEMENT_MISMATCH",
                     f"X + a != X_next at {state.state_id} "
                     f"(max |err| = {x_next_err:.3e} > {GEOM_TOL:g})")
            # d_t − d_{t+1} = a_t within 1e-6.
            nxt_label = labels[state.next_state_id]
            d_err = float(np.max(np.abs(
                np.asarray(label.ts_displacement, dtype=float)
                - np.asarray(nxt_label.ts_displacement, dtype=float) - a)))
            _require(d_err <= GEOM_TOL, "LABEL_MISMATCH",
                     f"d_t - d_(t+1) != a_t at {state.state_id} "
                     f"(max |err| = {d_err:.3e} > {GEOM_TOL:g})")
            # Remaining mass-weighted arc length is non-increasing.
            _require(
                nxt_label.remaining_arc_length
                <= label.remaining_arc_length + GEOM_TOL,
                "ARC_LENGTH_INCREASING",
                f"remaining arc length increases at {state.state_id}",
            )

    # -- serialization -------------------------------------------------------
    def to_payload(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "reaction_id": self.reaction_id,
            "reaction_group_id": self.reaction_group_id,
            "branches": [{
                "branch_id": b.branch_id,
                "endpoint_label": b.endpoint_label,
                "reaction_id": b.reaction_id,
                "episode_ids": list(b.episode_ids),
            } for b in self.branches],
            "episodes": [{
                "episode_id": e.episode_id,
                "branch_id": e.branch_id,
                "reaction_id": e.reaction_id,
                "start_state_id": e.start_state_id,
                "budget_version": e.budget_version,
                "environment_version": e.environment_version,
                "reward_version": e.reward_version,
            } for e in self.episodes],
            "states": [s.to_payload() for s in self.states],
            "labels": [{
                "state_id": lab.state_id,
                "ts_displacement":
                    np.asarray(lab.ts_displacement, dtype=float).round(10).tolist(),
                "remaining_arc_length": float(lab.remaining_arc_length),
                "teacher_endpoint": bool(lab.teacher_endpoint),
                "action_delta": (None if lab.action_delta is None else
                                 np.asarray(lab.action_delta,
                                            dtype=float).round(10).tolist()),
            } for lab in self.labels],
            "arc_length_metric_version": self.arc_length_metric_version,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, JSONValue]) -> "DualTSTeacher":
        branches = tuple(
            BranchRecord(
                branch_id=str(b["branch_id"]),
                endpoint_label=str(b["endpoint_label"]),
                reaction_id=str(b["reaction_id"]),
                episode_ids=tuple(str(e) for e in b.get("episode_ids", ())),
            ) for b in payload["branches"])
        episodes = tuple(
            EpisodeRecord(
                episode_id=str(e["episode_id"]),
                branch_id=str(e["branch_id"]),
                reaction_id=str(e["reaction_id"]),
                start_state_id=str(e["start_state_id"]),
                budget_version=str(e["budget_version"]),
                environment_version=str(e["environment_version"]),
                reward_version=str(e["reward_version"]),
            ) for e in payload["episodes"])
        states = tuple(TeacherState.from_payload(dict(s))
                       for s in payload["states"])
        labels = tuple(
            TeacherLabel(
                state_id=str(lab["state_id"]),
                ts_displacement=np.asarray(lab["ts_displacement"], dtype=float),
                remaining_arc_length=float(lab["remaining_arc_length"]),
                teacher_endpoint=bool(lab["teacher_endpoint"]),
                action_delta=(None if lab.get("action_delta") is None else
                              np.asarray(lab["action_delta"], dtype=float)),
            ) for lab in payload["labels"])
        teacher = cls(
            reaction_id=str(payload["reaction_id"]),
            reaction_group_id=str(payload["reaction_group_id"]),
            branches=branches,
            episodes=episodes,
            states=states,
            labels=labels,
            arc_length_metric_version=str(payload.get(
                "arc_length_metric_version", ARC_LENGTH_METRIC_VERSION)),
            schema_version=str(payload.get("schema_version",
                                           TEACHER_SCHEMA_VERSION)),
        )
        teacher.validate()
        return teacher

    def content_digest(self) -> str:
        return content_digest_of(self.to_payload())


# ---------------------------------------------------------------------------
# Feature spec record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DualTSFeaturesSpec:
    """``pesn_dual_ts_features_v3``: declares the exchange-symmetric dual-graph
    encoding, equivariant channels, the ΔE definition, fit-only frozen scales,
    masks/support domain, and the exact input dimension (observation-only
    5-channel node scalars as of v4.1 WP1)."""

    encoding: str = "dual_graph_exchange_symmetric_v0"
    equivariant_vectors: tuple[str, ...] = ("X", "g")
    delta_energy_definition: str = "E - E_episode_start"
    frozen_scales: Mapping[str, float] = field(default_factory=dict)
    masks: Mapping[str, JSONValue] = field(default_factory=dict)
    support_domain: Mapping[str, JSONValue] = field(default_factory=dict)
    declared_layout: Mapping[str, int] = field(default_factory=dict)
    input_dim: int = 0
    schema_version: str = FEATURES_SCHEMA_VERSION

    def validate(self) -> None:
        _require(self.schema_version in (FEATURES_SCHEMA_VERSION,
                                         FEATURES_SCHEMA_VERSION_V2),
                 "SCHEMA_UNKNOWN",
                 f"expected {FEATURES_SCHEMA_VERSION} (or legacy "
                 f"{FEATURES_SCHEMA_VERSION_V2})")
        _require("exchange_symmetric" in self.encoding, "ENCODING_INVALID",
                 "encoding must declare a dual-graph exchange-symmetric form")
        _require(set(self.equivariant_vectors) <= {"X", "g"},
                 "EQUIVARIANT_CHANNEL_INVALID",
                 "equivariant channels are 'X' and 'g' only")
        _require(self.delta_energy_definition == "E - E_episode_start",
                 "DELTA_E_DEFINITION_FROZEN",
                 "ΔE is frozen to E - E_episode_start")
        for name, scale in self.frozen_scales.items():
            _require(bool(np.isfinite(scale)) and float(scale) > 0.0,
                     "SCALE_INVALID",
                     f"frozen scale {name!r} must be finite > 0 (fit-only)")
        for name, dim in self.declared_layout.items():
            _require(int(dim) >= 0, "LAYOUT_INVALID",
                     f"declared_layout[{name!r}] must be >= 0")
        # v4.1 WP1 accepted-risk hardening: the schema NAME must agree with
        # the declared node-scalar width.  v3 = observation-only 5-channel
        # node scalars; legacy v2 = 7-channel.  The builder and content digest
        # bind the real layout already, but a synthetic payload declaring the
        # wrong width for its schema name must not validate (explicit
        # coupling; the load-time NODE_SCALAR_WIDTH_MISMATCH guard covers the
        # shard side).  This check is purely in ``validate`` and never alters
        # a well-formed payload, so spec digests are unchanged.
        node_scalar_dim = self.declared_layout.get("node_scalar")
        expected_node_scalar = {
            FEATURES_SCHEMA_VERSION: 5,
            FEATURES_SCHEMA_VERSION_V2: 7,
        }[self.schema_version]
        _require(
            node_scalar_dim is not None
            and int(node_scalar_dim) == expected_node_scalar,
            "NODE_SCALAR_SCHEMA_MISMATCH",
            f"schema {self.schema_version!r} requires "
            f"declared_layout['node_scalar'] == {expected_node_scalar}, "
            f"got {node_scalar_dim!r}",
        )
        layout_sum = int(sum(int(v) for v in self.declared_layout.values()))
        _require(layout_sum == int(self.input_dim), "INPUT_DIM_MISMATCH",
                 f"input_dim {self.input_dim} != declared layout sum "
                 f"{layout_sum} (input dim must match exactly)")

    def to_payload(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "encoding": self.encoding,
            "equivariant_vectors": list(self.equivariant_vectors),
            "delta_energy_definition": self.delta_energy_definition,
            "frozen_scales": {k: float(v)
                              for k, v in self.frozen_scales.items()},
            "masks": dict(self.masks),
            "support_domain": dict(self.support_domain),
            "declared_layout": {k: int(v)
                                for k, v in self.declared_layout.items()},
            "input_dim": int(self.input_dim),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, JSONValue]) -> "DualTSFeaturesSpec":
        spec = cls(
            encoding=str(payload["encoding"]),
            equivariant_vectors=tuple(str(v)
                                      for v in payload["equivariant_vectors"]),
            delta_energy_definition=str(payload["delta_energy_definition"]),
            frozen_scales={k: float(v)
                           for k, v in payload["frozen_scales"].items()},
            masks=dict(payload["masks"]),
            support_domain=dict(payload["support_domain"]),
            declared_layout={k: int(v)
                             for k, v in payload["declared_layout"].items()},
            input_dim=int(payload["input_dim"]),
            schema_version=str(payload.get("schema_version",
                                           FEATURES_SCHEMA_VERSION)),
        )
        spec.validate()
        return spec

    def content_digest(self) -> str:
        return content_digest_of(self.to_payload())


# ---------------------------------------------------------------------------
# Action record
# ---------------------------------------------------------------------------


class DualTSActionKind(enum.Enum):
    """Action kinds (contract doc §2/§4). STOP_AND_REFINE is a distinct kind —
    it must never be expressible as a (zero-)displacement MOVE."""

    MOVE = "MOVE"
    STOP_AND_REFINE = "STOP_AND_REFINE"
    ABORT = "ABORT"
    ACCEPTED_FALLBACK = "ACCEPTED_FALLBACK"


@dataclass(frozen=True, slots=True)
class DualTSAction:
    """``pesn_dual_ts_action_v2``.

    MOVE stores the raw latent ``u`` (dimensionless), the radial transform
    ``z = tanh(||u||)u/||u||`` (continuous limit 0 at ||u|| = 0), and the
    physical action ``a = a_max · z`` in Å, plus the transform/density
    version bindings. Non-MOVE kinds carry no displacement carriers at all.
    """

    kind: DualTSActionKind
    a_max_angstrom: float = 0.1
    latent_u: np.ndarray | None = None       # (N,3) raw latent, dimensionless
    normalized_z: np.ndarray | None = None   # (N,3) radial transform of u
    action_angstrom: np.ndarray | None = None  # (N,3) = a_max * z
    transform_version: str = RADIAL_TRANSFORM_VERSION
    density_version: str = ""
    reason: str | None = None
    schema_version: str = ACTION_SCHEMA_VERSION

    def validate(self) -> None:
        _require(self.schema_version == ACTION_SCHEMA_VERSION, "SCHEMA_UNKNOWN",
                 f"expected {ACTION_SCHEMA_VERSION}")
        _require(isinstance(self.kind, DualTSActionKind), "ACTION_KIND_INVALID",
                 "kind must be a DualTSActionKind")
        _require(self.a_max_angstrom > 0.0, "ACTION_MAX_INVALID",
                 "a_max must be positive")
        _require(bool(np.isfinite(self.a_max_angstrom)), "ACTION_MAX_INVALID",
                 "a_max must be finite")
        if self.kind is DualTSActionKind.MOVE:
            _require(
                self.latent_u is not None and self.normalized_z is not None
                and self.action_angstrom is not None,
                "MOVE_CARRIERS_MISSING",
                "MOVE stores u, z, and a together",
            )
            u = np.asarray(self.latent_u, dtype=float)
            z = np.asarray(self.normalized_z, dtype=float)
            a = np.asarray(self.action_angstrom, dtype=float)
            for name, arr in (("latent_u", u), ("normalized_z", z),
                              ("action_angstrom", a)):
                _require(arr.ndim == 2 and arr.shape[1] == 3,
                         "ACTION_SHAPE_INVALID",
                         f"{name} must be (N,3), got {arr.shape}")
                _require(bool(np.isfinite(arr).all()), "NONFINITE_ACTION",
                         f"{name} contains NaN/Inf")
            _require(u.shape == z.shape == a.shape, "ACTION_SHAPE_MISMATCH",
                     "u/z/a must share one (N,3) shape")
            z_expected = radial_tanh_transform(u)
            z_err = float(np.max(np.abs(z - z_expected)))
            _require(z_err <= GEOM_TOL, "Z_TRANSFORM_MISMATCH",
                     f"normalized_z != tanh(||u||)u/||u|| "
                     f"(max |err| = {z_err:.3e})")
            a_err = float(np.max(np.abs(a - self.a_max_angstrom * z)))
            _require(a_err <= GEOM_TOL, "ACTION_SCALE_MISMATCH",
                     f"a != a_max * z (max |err| = {a_err:.3e})")
        elif self.kind is DualTSActionKind.ACCEPTED_FALLBACK:
            # An accepted fallback records the actually executed physical
            # action; it was not sampled, so no latent carriers exist.
            _require(self.action_angstrom is not None, "FALLBACK_ACTION_MISSING",
                     "ACCEPTED_FALLBACK records the executed displacement")
            _require(self.latent_u is None and self.normalized_z is None,
                     "FALLBACK_NOT_SAMPLED",
                     "accepted fallback carries no sampled latent carriers")
            a = np.asarray(self.action_angstrom, dtype=float)
            _require(a.ndim == 2 and a.shape[1] == 3, "ACTION_SHAPE_INVALID",
                     f"action_angstrom must be (N,3), got {a.shape}")
            _require(bool(np.isfinite(a).all()), "NONFINITE_ACTION",
                     "action_angstrom contains NaN/Inf")
            _require(bool(self.reason), "REASON_REQUIRED",
                     "ACCEPTED_FALLBACK requires its acceptance reason")
        else:
            # STOP_AND_REFINE / ABORT: no displacement carriers whatsoever.
            # A STOP with (zero) displacement arrays is rejected — it is a
            # distinct kind, never a zero-displacement MOVE.
            _require(
                self.latent_u is None and self.normalized_z is None
                and self.action_angstrom is None,
                "STOP_WITH_DISPLACEMENT",
                f"{self.kind.value} must never validate as a (zero-)"
                "displacement MOVE; carriers must be None",
            )
            if self.kind is DualTSActionKind.ABORT:
                _require(bool(self.reason), "REASON_REQUIRED",
                         "ABORT requires a reason")
            if self.kind is DualTSActionKind.STOP_AND_REFINE:
                _require(bool(self.reason), "REASON_REQUIRED",
                         "STOP_AND_REFINE requires its refine protocol "
                         "reference in reason")

    def to_payload(self) -> dict[str, JSONValue]:
        def _r(arr: np.ndarray | None) -> list[list[float]] | None:
            return None if arr is None else np.asarray(arr, dtype=float).round(10).tolist()
        return {
            "schema_version": self.schema_version,
            "kind": self.kind.value,
            "a_max_angstrom": float(self.a_max_angstrom),
            "latent_u": _r(self.latent_u),
            "normalized_z": _r(self.normalized_z),
            "action_angstrom": _r(self.action_angstrom),
            "transform_version": self.transform_version,
            "density_version": self.density_version,
            "reason": self.reason,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, JSONValue]) -> "DualTSAction":
        def _arr(name: str) -> np.ndarray | None:
            value = payload.get(name)
            return None if value is None else np.asarray(value, dtype=float)
        try:
            kind = DualTSActionKind(str(payload["kind"]))
        except ValueError as exc:
            raise DualTSContractError(
                "ACTION_KIND_INVALID",
                f"kind {payload.get('kind')!r} not registered") from exc
        action = cls(
            kind=kind,
            a_max_angstrom=float(payload.get("a_max_angstrom", 0.1)),
            latent_u=_arr("latent_u"),
            normalized_z=_arr("normalized_z"),
            action_angstrom=_arr("action_angstrom"),
            transform_version=str(payload.get("transform_version",
                                              RADIAL_TRANSFORM_VERSION)),
            density_version=str(payload.get("density_version", "")),
            reason=(None if payload.get("reason") is None
                    else str(payload["reason"])),
            schema_version=str(payload.get("schema_version",
                                           ACTION_SCHEMA_VERSION)),
        )
        action.validate()
        return action

    def content_digest(self) -> str:
        return content_digest_of(self.to_payload())


# ---------------------------------------------------------------------------
# Transition record
# ---------------------------------------------------------------------------

TRANSITION_BOUNDARY_NONE: Final[str] = "none"
TRANSITION_BOUNDARY_DATA: Final[str] = "data_boundary"
TRANSITION_BOUNDARIES: Final[frozenset[str]] = frozenset({
    TRANSITION_BOUNDARY_NONE,
    BOUNDARY_TEACHER_ENDPOINT,
    TRANSITION_BOUNDARY_DATA,
    "budget_exhausted",
    "external_censored",
    "computation_missing",
})
TRANSITION_TERMINAL_BOUNDARIES: Final[frozenset[str]] = frozenset({
    BOUNDARY_TEACHER_ENDPOINT, "budget_exhausted",
})
TRANSITION_CENSORED_BOUNDARIES: Final[frozenset[str]] = frozenset({
    "external_censored", "computation_missing",
})
HEAD_NAMES: Final[frozenset[str]] = frozenset({"Q", "V", "actor"})


@dataclass(frozen=True, slots=True)
class DualTSTransition:
    """``pes2ts_dual_ts_transition_v2``: explicit state->next link with real
    successor/observation refs, boundary/terminal/censored masks + reasons,
    and per-head (Q/V/actor) eligibility.

    A file-slice end (``data_boundary``) is NOT terminal and NOT censored:
    the record keeps referencing the real successor in the next slice.
    """

    transition_id: str
    state_id: str
    next_state_id: str | None
    episode_id: str
    branch_id: str
    proposed_action: DualTSAction
    executed_action: DualTSAction | None = None
    next_observation_digest: str | None = None
    next_observation_missing_reason: str | None = None
    logical_cost: int = 1
    receipt_ref: str | None = None
    boundary: str = TRANSITION_BOUNDARY_NONE
    boundary_reason: str | None = None
    terminal: bool = False
    censored: bool = False
    head_eligibility: Mapping[str, Mapping[str, JSONValue]] = field(
        default_factory=dict)
    schema_version: str = TRANSITION_SCHEMA_VERSION

    def validate(self) -> None:
        _require(self.schema_version == TRANSITION_SCHEMA_VERSION,
                 "SCHEMA_UNKNOWN", f"expected {TRANSITION_SCHEMA_VERSION}")
        for name in ("transition_id", "state_id", "episode_id", "branch_id"):
            _require(bool(getattr(self, name)), "ID_MISSING", f"{name} required")
        _require(self.boundary in TRANSITION_BOUNDARIES, "BOUNDARY_UNKNOWN",
                 f"boundary {self.boundary!r} not registered")
        _require(self.logical_cost >= 0, "LOGICAL_COST_INVALID",
                 "logical_cost must be >= 0")
        self.proposed_action.validate()
        if self.executed_action is not None:
            self.executed_action.validate()

        # File-slice end first: neither terminal nor censored; successor
        # continues in the next slice (contract doc §3).
        if self.boundary == TRANSITION_BOUNDARY_DATA:
            _require(not self.terminal and not self.censored,
                     "SLICE_NOT_TERMINAL",
                     "a file-slice end is not a terminal or censoring event")
            _require(self.next_state_id is not None,
                     "SLICE_WITHOUT_SUCCESSOR",
                     "data-boundary transitions keep referencing the real "
                     "successor")
        # Censoring semantics: external cancellation / missing computation.
        if self.censored:
            _require(self.boundary in TRANSITION_CENSORED_BOUNDARIES,
                     "CENSOR_BOUNDARY_INVALID",
                     f"boundary {self.boundary!r} is not a censoring event")
            _require(not self.terminal, "CENSOR_NOT_TERMINAL",
                     "censoring is never a terminal")
            _require(bool(self.boundary_reason), "REASON_REQUIRED",
                     "censored transitions carry boundary_reason")
        # Terminal semantics: only real terminal boundaries may terminate.
        if self.terminal:
            _require(self.boundary in TRANSITION_TERMINAL_BOUNDARIES,
                     "TERMINAL_BOUNDARY_INVALID",
                     f"boundary {self.boundary!r} cannot be a terminal")
            _require(self.next_state_id is None, "TERMINAL_WITH_SUCCESSOR",
                     "terminal transitions carry no successor")
        else:
            _require(self.next_state_id is not None, "NEXT_STATE_MISSING",
                     "non-terminal transitions reference their real successor")
        if self.boundary == TRANSITION_BOUNDARY_NONE:
            _require(not self.terminal and not self.censored,
                     "BOUNDARY_FLAG_CONFLICT",
                     "boundary 'none' cannot be terminal or censored")
        # Real successor observation: null + reason when missing.
        if self.next_observation_digest is None:
            _require(self.next_observation_missing_reason in OBS_MISSING_REASONS,
                     "OBS_REASON_UNKNOWN",
                     "missing successor observation requires a registered "
                     "reason")
        else:
            _require(self.next_observation_missing_reason is None,
                     "OBS_REASON_CONFLICT",
                     "present observation digest carries no missing_reason")
            _require_sha256_hex(self.next_observation_digest,
                                "DIGEST_MALFORMED",
                                "next_observation_digest must be sha256 hex")
        for head, entry in self.head_eligibility.items():
            _require(head in HEAD_NAMES, "HEAD_UNKNOWN",
                     f"head {head!r} not in {{Q, V, actor}}")
            _require(isinstance(entry, Mapping), "ELIGIBILITY_MALFORMED",
                     f"head {head!r} eligibility must be a mapping")
            _require(isinstance(entry.get("eligible"), bool),
                     "ELIGIBILITY_MALFORMED",
                     f"head {head!r} needs a boolean 'eligible'")
            _require(isinstance(entry.get("reason"), str)
                     and bool(entry["reason"]),
                     "ELIGIBILITY_MALFORMED",
                     f"head {head!r} needs a non-empty 'reason'")

    def to_payload(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "transition_id": self.transition_id,
            "state_id": self.state_id,
            "next_state_id": self.next_state_id,
            "episode_id": self.episode_id,
            "branch_id": self.branch_id,
            "proposed_action": self.proposed_action.to_payload(),
            "executed_action": (None if self.executed_action is None
                                else self.executed_action.to_payload()),
            "next_observation_digest": self.next_observation_digest,
            "next_observation_missing_reason": self.next_observation_missing_reason,
            "logical_cost": int(self.logical_cost),
            "receipt_ref": self.receipt_ref,
            "boundary": self.boundary,
            "boundary_reason": self.boundary_reason,
            "terminal": bool(self.terminal),
            "censored": bool(self.censored),
            "head_eligibility": {
                head: dict(entry)
                for head, entry in self.head_eligibility.items()
            },
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, JSONValue]) -> "DualTSTransition":
        executed = payload.get("executed_action")
        transition = cls(
            transition_id=str(payload["transition_id"]),
            state_id=str(payload["state_id"]),
            next_state_id=(None if payload.get("next_state_id") is None
                           else str(payload["next_state_id"])),
            episode_id=str(payload["episode_id"]),
            branch_id=str(payload["branch_id"]),
            proposed_action=DualTSAction.from_payload(
                dict(payload["proposed_action"])),
            executed_action=(None if executed is None
                             else DualTSAction.from_payload(dict(executed))),
            next_observation_digest=(
                None if payload.get("next_observation_digest") is None
                else str(payload["next_observation_digest"])),
            next_observation_missing_reason=(
                None if payload.get("next_observation_missing_reason") is None
                else str(payload["next_observation_missing_reason"])),
            logical_cost=int(payload.get("logical_cost", 1)),
            receipt_ref=(None if payload.get("receipt_ref") is None
                         else str(payload["receipt_ref"])),
            boundary=str(payload.get("boundary", TRANSITION_BOUNDARY_NONE)),
            boundary_reason=(None if payload.get("boundary_reason") is None
                             else str(payload["boundary_reason"])),
            terminal=bool(payload.get("terminal", False)),
            censored=bool(payload.get("censored", False)),
            head_eligibility={
                str(head): dict(entry) for head, entry
                in payload.get("head_eligibility", {}).items()
            },
            schema_version=str(payload.get("schema_version",
                                           TRANSITION_SCHEMA_VERSION)),
        )
        transition.validate()
        return transition

    def content_digest(self) -> str:
        return content_digest_of(self.to_payload())


# ---------------------------------------------------------------------------
# Reward record
# ---------------------------------------------------------------------------

REWARD_MISSING_REASONS: Final[frozenset[str]] = frozenset({
    "outcome_unknown",
    "protocol_not_concluded",
    "computation_missing",
    "not_evaluated",
})


@dataclass(frozen=True, slots=True)
class DualTSReward:
    """``pes2ts_dual_ts_reward_v2``: the logical action cost plus the real
    outcome protocol/result.

    Unknown reward/outcome/return are explicit ``None`` + a registered
    reason — never a fabricated 0.
    """

    transition_id: str
    logical_cost: float
    outcome_protocol: str = ""
    outcome_result: str | None = None
    outcome_missing_reason: str | None = None
    reward: float | None = None
    reward_missing_reason: str | None = None
    episode_return: float | None = None
    return_missing_reason: str | None = None
    schema_version: str = REWARD_SCHEMA_VERSION

    def validate(self) -> None:
        _require(self.schema_version == REWARD_SCHEMA_VERSION, "SCHEMA_UNKNOWN",
                 f"expected {REWARD_SCHEMA_VERSION}")
        _require(bool(self.transition_id), "ID_MISSING",
                 "transition_id required")
        _require(bool(np.isfinite(self.logical_cost))
                 and float(self.logical_cost) >= 0.0, "LOGICAL_COST_INVALID",
                 "logical_cost must be finite >= 0")
        # A present value must never coexist with an "unknown" reason; an
        # unknown must be None — 0.0 is never accepted as "unknown".
        if self.reward is None:
            _require(self.reward_missing_reason in REWARD_MISSING_REASONS,
                     "REASON_UNKNOWN",
                     "unknown reward requires a registered reason")
        else:
            _require(self.reward_missing_reason is None, "REWARD_CONFLICT",
                     "reward value present while claiming unknown")
            _require(bool(np.isfinite(self.reward)), "REWARD_NONFINITE",
                     "reward must be finite")
        if self.outcome_result is None:
            _require(self.outcome_missing_reason in REWARD_MISSING_REASONS,
                     "REASON_UNKNOWN",
                     "unknown outcome requires a registered reason")
        else:
            _require(self.outcome_missing_reason is None, "OUTCOME_CONFLICT",
                     "outcome result present while claiming unknown")
            _require(bool(self.outcome_protocol), "PROTOCOL_REQUIRED",
                     "a real outcome requires its protocol identity")
        if self.episode_return is None:
            _require(self.return_missing_reason in REWARD_MISSING_REASONS,
                     "REASON_UNKNOWN",
                     "unknown return requires a registered reason")
        else:
            _require(self.return_missing_reason is None, "RETURN_CONFLICT",
                     "return value present while claiming unknown")
            _require(bool(np.isfinite(self.episode_return)),
                     "RETURN_NONFINITE", "return must be finite")

    def to_payload(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "transition_id": self.transition_id,
            "logical_cost": float(self.logical_cost),
            "outcome_protocol": self.outcome_protocol,
            "outcome_result": self.outcome_result,
            "outcome_missing_reason": self.outcome_missing_reason,
            "reward": (None if self.reward is None else float(self.reward)),
            "reward_missing_reason": self.reward_missing_reason,
            "episode_return": (None if self.episode_return is None
                               else float(self.episode_return)),
            "return_missing_reason": self.return_missing_reason,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, JSONValue]) -> "DualTSReward":
        record = cls(
            transition_id=str(payload["transition_id"]),
            logical_cost=float(payload["logical_cost"]),
            outcome_protocol=str(payload.get("outcome_protocol", "")),
            outcome_result=(None if payload.get("outcome_result") is None
                            else str(payload["outcome_result"])),
            outcome_missing_reason=(
                None if payload.get("outcome_missing_reason") is None
                else str(payload["outcome_missing_reason"])),
            reward=(None if payload.get("reward") is None
                    else float(payload["reward"])),
            reward_missing_reason=(
                None if payload.get("reward_missing_reason") is None
                else str(payload["reward_missing_reason"])),
            episode_return=(None if payload.get("episode_return") is None
                            else float(payload["episode_return"])),
            return_missing_reason=(
                None if payload.get("return_missing_reason") is None
                else str(payload["return_missing_reason"])),
            schema_version=str(payload.get("schema_version",
                                           REWARD_SCHEMA_VERSION)),
        )
        record.validate()
        return record

    def content_digest(self) -> str:
        return content_digest_of(self.to_payload())


# ---------------------------------------------------------------------------
# Bundle / run identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DualTSBundle:
    """``pesn_dual_ts_bundle_v2``: run bundle identity — eligibility, BC parent
    checkpoint content digest, seeds, hyperparameters, and the frozen
    norm/split/dataset/action/env/reward version bindings.

    Research capability tables stay closed to production decisions by
    default (``production_decision_open=False``).
    """

    run_id: str
    bc_parent_checkpoint_digest: str       # real content digest (sha256 hex)
    seeds: Mapping[str, int]
    hyperparams: Mapping[str, JSONValue] = field(default_factory=dict)
    dataset_version: str = ""
    norm_version: str = ""
    split_version: str = ""
    action_version: str = ""
    environment_version: str = ""
    reward_version: str = ""
    input_eligibility: Mapping[str, JSONValue] = field(default_factory=dict)
    output_eligibility: Mapping[str, JSONValue] = field(default_factory=dict)
    production_decision_open: bool = False
    schema_version: str = BUNDLE_SCHEMA_VERSION

    def validate(self) -> None:
        _require(self.schema_version == BUNDLE_SCHEMA_VERSION, "SCHEMA_UNKNOWN",
                 f"expected {BUNDLE_SCHEMA_VERSION}")
        _require(bool(self.run_id), "ID_MISSING", "run_id required")
        _require_sha256_hex(self.bc_parent_checkpoint_digest,
                            "DIGEST_MALFORMED",
                            "bc_parent_checkpoint_digest must be a sha256 "
                            "hex content digest")
        _require(bool(self.seeds), "SEEDS_MISSING",
                 "at least one RNG seed must be bound")
        for name, seed in self.seeds.items():
            _require(isinstance(seed, int) and not isinstance(seed, bool),
                     "SEED_INVALID", f"seed {name!r} must be an int")
        for name in ("dataset_version", "norm_version", "split_version",
                     "action_version", "environment_version",
                     "reward_version"):
            _require(bool(getattr(self, name)), "VERSION_BINDING_MISSING",
                     f"{name} binding required")

    def to_payload(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "bc_parent_checkpoint_digest": self.bc_parent_checkpoint_digest,
            "seeds": {k: int(v) for k, v in self.seeds.items()},
            "hyperparams": dict(self.hyperparams),
            "dataset_version": self.dataset_version,
            "norm_version": self.norm_version,
            "split_version": self.split_version,
            "action_version": self.action_version,
            "environment_version": self.environment_version,
            "reward_version": self.reward_version,
            "input_eligibility": dict(self.input_eligibility),
            "output_eligibility": dict(self.output_eligibility),
            "production_decision_open": bool(self.production_decision_open),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, JSONValue]) -> "DualTSBundle":
        bundle = cls(
            run_id=str(payload["run_id"]),
            bc_parent_checkpoint_digest=str(payload["bc_parent_checkpoint_digest"]),
            seeds={str(k): int(v) for k, v in payload["seeds"].items()},
            hyperparams=dict(payload.get("hyperparams", {})),
            dataset_version=str(payload.get("dataset_version", "")),
            norm_version=str(payload.get("norm_version", "")),
            split_version=str(payload.get("split_version", "")),
            action_version=str(payload.get("action_version", "")),
            environment_version=str(payload.get("environment_version", "")),
            reward_version=str(payload.get("reward_version", "")),
            input_eligibility=dict(payload.get("input_eligibility", {})),
            output_eligibility=dict(payload.get("output_eligibility", {})),
            production_decision_open=bool(
                payload.get("production_decision_open", False)),
            schema_version=str(payload.get("schema_version",
                                           BUNDLE_SCHEMA_VERSION)),
        )
        bundle.validate()
        return bundle

    def content_digest(self) -> str:
        return content_digest_of(self.to_payload())


# ---------------------------------------------------------------------------
# Schema registry + dispatch
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SchemaInfo:
    """Registry entry: frozen name, version, description, payload validator."""

    name: str
    version: str
    description: str
    validator: Callable[[Mapping[str, JSONValue]], None]


def _validate_task_payload(payload: Mapping[str, JSONValue]) -> None:
    DualEndpointTask.from_payload(payload)


def _validate_teacher_payload(payload: Mapping[str, JSONValue]) -> None:
    DualTSTeacher.from_payload(payload)


def _validate_features_payload(payload: Mapping[str, JSONValue]) -> None:
    DualTSFeaturesSpec.from_payload(payload)


def _validate_action_payload(payload: Mapping[str, JSONValue]) -> None:
    DualTSAction.from_payload(payload)


def _validate_transition_payload(payload: Mapping[str, JSONValue]) -> None:
    DualTSTransition.from_payload(payload)


def _validate_reward_payload(payload: Mapping[str, JSONValue]) -> None:
    DualTSReward.from_payload(payload)


def _validate_bundle_payload(payload: Mapping[str, JSONValue]) -> None:
    DualTSBundle.from_payload(payload)


SCHEMAS: Final[dict[str, SchemaInfo]] = {
    TASK_SCHEMA_VERSION: SchemaInfo(
        name=TASK_SCHEMA_VERSION,
        version="v2.0",
        description="Unordered complete endpoint graph pair with reaction "
                    "identity, constraints, and provenance.",
        validator=_validate_task_payload,
    ),
    TEACHER_SCHEMA_VERSION: SchemaInfo(
        name=TEACHER_SCHEMA_VERSION,
        version="v2.0",
        description="Branch/episode/state/label teacher table with "
                    "cross-record geometric and boundary rules.",
        validator=_validate_teacher_payload,
    ),
    FEATURES_SCHEMA_VERSION: SchemaInfo(
        name=FEATURES_SCHEMA_VERSION,
        version="v3.0",
        description="Exchange-symmetric dual-graph feature spec with "
                    "equivariant X/g, ΔE, frozen scales, exact input dim; "
                    "v4.1 WP1 observation-only 5-channel node scalars.",
        validator=_validate_features_payload,
    ),
    FEATURES_SCHEMA_VERSION_V2: SchemaInfo(
        name=FEATURES_SCHEMA_VERSION_V2,
        version="v2.0",
        description="LEGACY 7-channel feature spec (kept registered so old "
                    "products remain readable; not produced by new builds).",
        validator=_validate_features_payload,
    ),
    ACTION_SCHEMA_VERSION: SchemaInfo(
        name=ACTION_SCHEMA_VERSION,
        version="v2.0",
        description="MOVE/STOP_AND_REFINE/ABORT/ACCEPTED_FALLBACK action "
                    "with radial tanh transform carriers.",
        validator=_validate_action_payload,
    ),
    TRANSITION_SCHEMA_VERSION: SchemaInfo(
        name=TRANSITION_SCHEMA_VERSION,
        version="v2.0",
        description="Explicit transition with real successor/observation "
                    "refs, boundary/terminal/censored masks, head eligibility.",
        validator=_validate_transition_payload,
    ),
    REWARD_SCHEMA_VERSION: SchemaInfo(
        name=REWARD_SCHEMA_VERSION,
        version="v2.0",
        description="Logical cost + real outcome protocol; unknown values "
                    "are None + reason, never 0.",
        validator=_validate_reward_payload,
    ),
    BUNDLE_SCHEMA_VERSION: SchemaInfo(
        name=BUNDLE_SCHEMA_VERSION,
        version="v2.0",
        description="Run bundle identity with BC parent digest, seeds, and "
                    "version bindings.",
        validator=_validate_bundle_payload,
    ),
}


def validate_record(schema_name: str, record: Mapping[str, JSONValue]) -> None:
    """Dispatch a record payload to its registered validator.

    Unregistered names raise ``SCHEMA_UNKNOWN``; structurally broken payloads
    raise ``PAYLOAD_MALFORMED``; everything else surfaces the underlying
    typed contract code.
    """
    info = SCHEMAS.get(schema_name)
    if info is None:
        raise DualTSContractError("SCHEMA_UNKNOWN",
                                  f"schema {schema_name!r} not registered")
    if not isinstance(record, Mapping):
        raise DualTSContractError("PAYLOAD_MALFORMED",
                                  "record must be a mapping payload")
    try:
        info.validator(record)
    except DualTSContractError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise DualTSContractError(
            "PAYLOAD_MALFORMED",
            f"{schema_name}: payload rejected ({exc!r})") from exc


__all__ = [
    "ARC_LENGTH_METRIC_VERSION",
    "ACTION_SCHEMA_VERSION",
    "BUNDLE_SCHEMA_VERSION",
    "BOUNDARY_TEACHER_ENDPOINT",
    "BOUNDARY_TYPES",
    "BranchRecord",
    "DualEndpointTask",
    "DualTSAction",
    "DualTSActionKind",
    "DualTSBundle",
    "DualTSContractError",
    "DualTSFeaturesSpec",
    "DualTSTeacher",
    "DualTSTransition",
    "DualTSReward",
    "EpisodeRecord",
    "EndpointBond",
    "EndpointGraph",
    "FEATURES_SCHEMA_VERSION",
    "FEATURES_SCHEMA_VERSION_V2",
    "GEOM_TOL",
    "HEAD_NAMES",
    "OBS_MISSING_REASONS",
    "OBS_STATUS_MISSING",
    "OBS_STATUS_OK",
    "ObservationRef",
    "RADIAL_TRANSFORM_VERSION",
    "REWARD_MISSING_REASONS",
    "REWARD_SCHEMA_VERSION",
    "SCHEMAS",
    "SchemaInfo",
    "TASK_SCHEMA_VERSION",
    "TEACHER_SCHEMA_VERSION",
    "TRANSITION_BOUNDARIES",
    "TRANSITION_BOUNDARY_DATA",
    "TRANSITION_BOUNDARY_NONE",
    "TRANSITION_SCHEMA_VERSION",
    "TeacherLabel",
    "TeacherState",
    "canonical_endpoint_order",
    "content_digest_of",
    "partition_by_reaction_group",
    "radial_tanh_transform",
    "validate_record",
]
