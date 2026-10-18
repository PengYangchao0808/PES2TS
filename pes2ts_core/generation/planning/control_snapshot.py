"""Control-state snapshot contract (WP-3 §6.1 / IRC plan §5 完整回放).

Schema ``pes2ts_control_state_snapshot_v1``: a strict, JSON-serializable
capture of ONE E_native control-scan decision point, sufficient to resume
the scan loop at that point under the same budget/binds with an injected
candidate (paired-branch diagnostics; plan §8.2 — branches never become
main-episode transitions).

Pure data contract: no torch, no QC, no policy imports.  Digests follow the
established canonical form of
:func:`pes2ts_core.generation.planning.synchronized_path.digest`
(``sha256(json.dumps(sort_keys=True, separators=(",", ":")))``), the same
pattern ``freeze_control_environment`` uses for the environment hash.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

SNAPSHOT_SCHEMA_VERSION = "pes2ts_control_state_snapshot_v1"

#: Recorded whenever a policy rng object is (or is not) available.  The
#: macro-step continuation performs no policy sampling, so a replay branch is
#: deterministic given (physical state, injected residual, transport).
DETERMINISM_NOTE = (
    "The macro-step continuation (seed -> constrained-opt request -> "
    "acceptance/refusal -> midpoint subdivision) performs no policy "
    "sampling: given (physical state, injected residual, transport) the "
    "path is deterministic. The exploration policy RNG is unused for the "
    "executed decision on replay."
)
NO_POLICY_NOTE = (
    "decide is a bare callable with no exposed policy identity or rng state; "
    "nothing is fabricated. " + DETERMINISM_NOTE
)

RNG_KIND_GENERATOR = "generator_state"
RNG_KIND_FROZEN_SEED = "frozen_seed_only"
RNG_KIND_NONE = "none"
RNG_KINDS = frozenset({RNG_KIND_GENERATOR, RNG_KIND_FROZEN_SEED, RNG_KIND_NONE})

ALLOWED_TOP = frozenset({
    "schema_version",
    "episode",
    "control_identity",
    "physical_state",
    "budget",
    "counters",
    "history",
    "rng",
    "parent_state_sha256",
})

SECTION_KEYS: dict[str, frozenset[str]] = {
    "episode": frozenset({
        "round_id", "episode_id", "reaction_id", "branch", "decision_index",
    }),
    "control_identity": frozenset({
        "control_digest", "spec_digest", "policy_id", "policy_hash",
        "environment_hash", "reward_spec_sha256", "code_identity",
    }),
    "physical_state": frozenset({
        "parent_geometry", "parent_bonds", "parent_index",
        "parent_request_sha256", "last_accepted_lambda", "target_lambda",
        "pending", "scale", "open_target", "open_transition",
    }),
    "budget": frozenset({
        "max_wall_seconds", "max_host_requests", "wall_elapsed_seconds",
        "request_count",
    }),
    "counters": frozenset({
        "decision_index", "accepted_prefix_frame_indices",
        "recent_rejections", "candidate_already_awarded",
    }),
    "history": frozenset({
        "subdivision", "rejection_memory", "open_refusal_reasons",
    }),
    "rng": frozenset({
        "kind", "exploration_seed", "generator_state", "determinism_note",
    }),
}

_PENDING_KEYS = frozenset({"lambda", "is_grid_point", "rejections"})
_SUBDIVISION_KEYS = frozenset({
    "scale", "min_scale", "max_subdivisions", "max_rejections_per_target",
})
_REJECTION_MEMORY_KEYS = frozenset({"target_lambda", "rejections", "is_grid_point"})


class ControlSnapshotError(ValueError):
    """Typed snapshot contract violation (never a silent default)."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"control snapshot violated [{code}]: {detail}")


def _require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise ControlSnapshotError(code, detail)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def snapshot_content_sha256(snapshot: Mapping[str, Any]) -> str:
    """Canonical content digest of a snapshot document (chain identity)."""
    return hashlib.sha256(_canonical_json(snapshot).encode("utf-8")).hexdigest()


def origin_state_sha256(
    *, reaction_id: str, episode_id: str, origin_geometry: Sequence[Sequence[float]]
) -> str:
    """Chain anchor for the FIRST decision of an episode (origin hash)."""
    payload = {
        "kind": "origin",
        "reaction_id": str(reaction_id),
        "episode_id": str(episode_id),
        "origin_geometry": [[float(x) for x in row] for row in origin_geometry],
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def control_code_identity() -> str:
    """Digest of the frozen E_native execution-code constants.

    Mirrors the canonical-hash pattern of ``freeze_control_environment``:
    the ACP request/evidence schemas plus the frozen engine/candidate
    versions identify the code path a snapshot was captured under.
    """
    from pes2ts_core.integration.acp.native_constraint_results import EVIDENCE_SCHEMA
    from pes2ts_core.integration.acp.xtb_constraints_request import XTC_REQUEST_SCHEMA

    payload = {
        "control_engine_mode": "control_residual_e_native",
        "control_engine_version": 1,
        "candidate_generator_version": "bounded_candidates_v1",
        "acp_workflow": "XtbConstrainedOpt",
        "request_schema": XTC_REQUEST_SCHEMA,
        "evidence_schema": EVIDENCE_SCHEMA,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def control_document_digest(control: Any) -> str:
    """Canonical digest of the compiled ``ChemicalControl`` identity."""
    drivers = []
    for driver in control.drivers:
        drivers.append({
            "driver_id": str(driver.driver_id),
            "kind": str(driver.kind),
            "atoms": [int(a) for a in driver.atoms],
            "maps": [int(m) for m in driver.maps],
            "unit": str(driver.unit),
            "role": str(driver.role),
            "lambda_values": [float(v) for v in driver.lambda_values],
            "values": [float(v) for v in driver.values],
        })
    payload = {
        "reaction_id": str(control.reaction_id),
        "plan_sha256": str(control.plan_sha256),
        "atom_order_sha256": str(control.atom_order_sha256),
        "elements": [str(e) for e in control.elements],
        "atom_map_order": [int(m) for m in control.atom_map_order],
        "charge": int(control.charge),
        "multiplicity": int(control.multiplicity),
        "drivers": drivers,
        "lambda_grid": [float(v) for v in control.lambda_grid],
        "execution_direction": str(control.execution_direction),
        "origin_endpoint": str(control.origin_endpoint),
        "active_atom_indices": [int(i) for i in control.active_atom_indices],
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def policy_identity_from_decide(decide: Any) -> dict[str, Any]:
    """Best-effort policy identity + rng capture from a ``decide`` callable.

    Bound policy methods expose ``policy_id`` / ``policy_hash`` / ``rng``
    (EpsilonExplorePolicy owns a seeded ``numpy.random.Generator``); bare
    fixtures expose none of these and NOTHING is fabricated — the snapshot
    records ``None`` identity and an explicit ``none`` rng kind.
    """
    policy_obj = getattr(decide, "__self__", None)
    policy_id = getattr(policy_obj, "policy_id", None)
    policy_hash = getattr(policy_obj, "policy_hash", None)
    config = getattr(policy_obj, "config", None)
    exploration_seed = getattr(config, "exploration_seed", None) if config is not None else None

    rng_obj = getattr(policy_obj, "rng", None)
    generator_state: Any = None
    if rng_obj is not None:
        bit_gen = getattr(rng_obj, "bit_generator", None)
        state = getattr(bit_gen, "state", None)
        if state is not None:
            try:
                generator_state = json.loads(json.dumps(state))
            except (TypeError, ValueError):
                generator_state = None

    if generator_state is not None:
        kind = RNG_KIND_GENERATOR
        note = DETERMINISM_NOTE
    elif exploration_seed is not None or policy_id is not None:
        kind = RNG_KIND_FROZEN_SEED
        note = DETERMINISM_NOTE
    else:
        kind = RNG_KIND_NONE
        note = NO_POLICY_NOTE

    return {
        "policy_id": None if policy_id is None else str(policy_id),
        "policy_hash": None if policy_hash is None else str(policy_hash),
        "rng": {
            "kind": kind,
            "exploration_seed": (
                None if exploration_seed is None else int(exploration_seed)
            ),
            "generator_state": generator_state,
            "determinism_note": note,
        },
    }


def _jsonable(value: Any) -> Any:
    """Recursively convert tuples/sets/dataclasses-with-to_payload to JSON."""
    to_payload = getattr(value, "to_payload", None)
    if callable(to_payload):
        return _jsonable(to_payload())
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return [_jsonable(v) for v in sorted(value, key=repr)]
    return value


def geometry_to_payload(geometry: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[float(x) for x in row] for row in geometry]


def bonds_to_payload(bonds: Any) -> list[list[int]]:
    if isinstance(bonds, Mapping):
        pairs = list(bonds)
    else:
        pairs = [tuple(pair) for pair in bonds]
    return [[int(a), int(b)] for a, b in sorted(pairs)]


def pending_to_payload(pending: Sequence[Any]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for entry in pending:
        if isinstance(entry, Mapping):
            lam = entry["lambda"]
            is_grid = entry["is_grid_point"]
            rejections = entry["rejections"]
        else:
            lam, is_grid, rejections = entry
        payload.append({
            "lambda": float(lam),
            "is_grid_point": bool(is_grid),
            "rejections": int(rejections),
        })
    return payload


def pending_from_payload(payload: Sequence[Mapping[str, Any]]) -> list[tuple[float, bool, int]]:
    return [
        (float(entry["lambda"]), bool(entry["is_grid_point"]), int(entry["rejections"]))
        for entry in payload
    ]


def build_snapshot_document(
    *,
    # episode identity
    round_id: str | None,
    episode_id: str,
    reaction_id: str,
    branch: str | None,
    decision_index: int,
    # control identity
    control_digest: str,
    spec_digest: str,
    policy_id: str | None,
    policy_hash: str | None,
    environment_hash: str,
    reward_spec_sha256: str,
    code_identity: str,
    # full physical state
    parent_geometry: Sequence[Sequence[float]],
    parent_bonds: Any,
    parent_index: int,
    parent_request_sha256: str | None,
    last_accepted_lambda: float,
    target_lambda: float,
    pending: Sequence[Any],
    scale: float,
    open_target: float | None,
    open_transition: Mapping[str, Any] | None,
    # budget
    max_wall_seconds: float,
    max_host_requests: int,
    wall_elapsed_seconds: float,
    request_count: int,
    # counters
    accepted_prefix_frame_indices: Sequence[int],
    recent_rejections: int,
    candidate_already_awarded: bool,
    # history / subdivision
    subdivision: Mapping[str, Any],
    open_refusal_reasons: Sequence[str] | None = None,
    # rng
    rng: Mapping[str, Any],
    # chain identity
    parent_state_sha256: str | None,
) -> dict[str, Any]:
    """Assemble + validate one ``pes2ts_control_state_snapshot_v1`` document."""
    pending_payload = pending_to_payload(pending)
    subdivision_block = {
        "scale": float(subdivision["scale"]),
        "min_scale": float(subdivision["min_scale"]),
        "max_subdivisions": int(subdivision["max_subdivisions"]),
        "max_rejections_per_target": int(subdivision["max_rejections_per_target"]),
    }
    document: dict[str, Any] = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "episode": {
            "round_id": None if round_id is None else str(round_id),
            "episode_id": str(episode_id),
            "reaction_id": str(reaction_id),
            "branch": None if branch is None else str(branch),
            "decision_index": int(decision_index),
        },
        "control_identity": {
            "control_digest": str(control_digest),
            "spec_digest": str(spec_digest),
            "policy_id": None if policy_id is None else str(policy_id),
            "policy_hash": None if policy_hash is None else str(policy_hash),
            "environment_hash": str(environment_hash),
            "reward_spec_sha256": str(reward_spec_sha256),
            "code_identity": str(code_identity),
        },
        "physical_state": {
            "parent_geometry": geometry_to_payload(parent_geometry),
            "parent_bonds": bonds_to_payload(parent_bonds),
            "parent_index": int(parent_index),
            "parent_request_sha256": (
                None if parent_request_sha256 is None else str(parent_request_sha256)
            ),
            "last_accepted_lambda": float(last_accepted_lambda),
            "target_lambda": float(target_lambda),
            "pending": pending_payload,
            "scale": float(scale),
            "open_target": None if open_target is None else float(open_target),
            "open_transition": (
                None if open_transition is None else _jsonable(open_transition)
            ),
        },
        "budget": {
            "max_wall_seconds": float(max_wall_seconds),
            "max_host_requests": int(max_host_requests),
            "wall_elapsed_seconds": float(wall_elapsed_seconds),
            "request_count": int(request_count),
        },
        "counters": {
            "decision_index": int(decision_index),
            "accepted_prefix_frame_indices": [int(i) for i in accepted_prefix_frame_indices],
            "recent_rejections": int(recent_rejections),
            "candidate_already_awarded": bool(candidate_already_awarded),
        },
        "history": {
            "subdivision": subdivision_block,
            "rejection_memory": [
                {
                    "target_lambda": float(entry["lambda"]),
                    "rejections": int(entry["rejections"]),
                    "is_grid_point": bool(entry["is_grid_point"]),
                }
                for entry in pending_payload
            ],
            "open_refusal_reasons": [
                str(reason) for reason in (open_refusal_reasons or ())
            ],
        },
        "rng": {
            "kind": str(rng["kind"]),
            "exploration_seed": (
                None if rng.get("exploration_seed") is None
                else int(rng["exploration_seed"])
            ),
            "generator_state": rng.get("generator_state"),
            "determinism_note": str(rng["determinism_note"]),
        },
        "parent_state_sha256": (
            None if parent_state_sha256 is None else str(parent_state_sha256)
        ),
    }
    validate_snapshot(document)
    return document


def _check_finite_number(value: Any, code: str, detail: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        code, detail,
    )
    number = float(value)
    _require(number == number and abs(number) != float("inf"), code, detail)
    return number


def _check_optional_str(value: Any, code: str, detail: str) -> None:
    _require(value is None or isinstance(value, str), code, detail)


def _check_sha256(value: Any, code: str, detail: str) -> None:
    _require(isinstance(value, str) and len(value) == 64, code, detail)
    try:
        int(value, 16)
    except ValueError as exc:
        raise ControlSnapshotError(code, detail) from exc


def validate_snapshot(snapshot: Mapping[str, Any]) -> None:
    """Strict validator for ``pes2ts_control_state_snapshot_v1``.

    Unknown top-level or section keys are rejected; every section key is
    required; types and value ranges are enforced.  History/budget leaves
    are also checked for presence so a structurally incomplete document can
    never masquerade as a resumable state.
    """
    if not isinstance(snapshot, Mapping):
        raise ControlSnapshotError("TYPE_INVALID", "snapshot must be a mapping")
    _require(
        snapshot.get("schema_version") == SNAPSHOT_SCHEMA_VERSION,
        "SCHEMA_UNKNOWN",
        f"schema_version {snapshot.get('schema_version')!r} unsupported",
    )
    unknown = set(snapshot) - ALLOWED_TOP
    if unknown:
        raise ControlSnapshotError(
            "UNKNOWN_KEY", f"unknown top-level keys: {sorted(unknown)}"
        )
    for section, allowed in SECTION_KEYS.items():
        _require(section in snapshot, "MISSING_KEY", f"missing section {section!r}")
        block = snapshot[section]
        _require(
            isinstance(block, Mapping),
            "TYPE_INVALID", f"section {section!r} must be a mapping",
        )
        extra = set(block) - allowed
        if extra:
            raise ControlSnapshotError(
                "UNKNOWN_KEY", f"unknown keys in {section!r}: {sorted(extra)}"
            )
        missing = allowed - set(block)
        if missing:
            raise ControlSnapshotError(
                "MISSING_KEY", f"missing keys in {section!r}: {sorted(missing)}"
            )
    _require(
        "parent_state_sha256" in snapshot, "MISSING_KEY", "missing parent_state_sha256"
    )

    episode = snapshot["episode"]
    _require(
        isinstance(episode["episode_id"], str) and bool(episode["episode_id"]),
        "TYPE_INVALID", "episode.episode_id must be a non-empty string",
    )
    _require(
        isinstance(episode["reaction_id"], str) and bool(episode["reaction_id"]),
        "TYPE_INVALID", "episode.reaction_id must be a non-empty string",
    )
    _check_optional_str(episode["round_id"], "TYPE_INVALID", "episode.round_id type")
    _check_optional_str(episode["branch"], "TYPE_INVALID", "episode.branch type")
    _require(
        isinstance(episode["decision_index"], int)
        and not isinstance(episode["decision_index"], bool)
        and episode["decision_index"] >= 0,
        "VALUE_INVALID", "episode.decision_index must be an int >= 0",
    )

    identity = snapshot["control_identity"]
    for key in ("control_digest", "spec_digest", "environment_hash",
                "reward_spec_sha256", "code_identity"):
        _require(
            isinstance(identity[key], str) and bool(identity[key]),
            "TYPE_INVALID", f"control_identity.{key} must be a non-empty string",
        )
    _check_optional_str(identity["policy_id"], "TYPE_INVALID", "control_identity.policy_id type")
    _check_optional_str(identity["policy_hash"], "TYPE_INVALID", "control_identity.policy_hash type")

    physical = snapshot["physical_state"]
    geometry = physical["parent_geometry"]
    _require(
        isinstance(geometry, list) and geometry,
        "VALUE_INVALID", "physical_state.parent_geometry must be a non-empty list",
    )
    for row in geometry:
        _require(
            isinstance(row, list) and len(row) == 3,
            "VALUE_INVALID", "parent_geometry rows must be [x, y, z]",
        )
        for value in row:
            _check_finite_number(value, "VALUE_INVALID", "parent_geometry non-finite")
    bonds = physical["parent_bonds"]
    _require(isinstance(bonds, list), "TYPE_INVALID", "physical_state.parent_bonds type")
    for pair in bonds:
        _require(
            isinstance(pair, list) and len(pair) == 2,
            "VALUE_INVALID", "parent_bonds entries must be [i, j]",
        )
        i, j = pair
        _require(
            isinstance(i, int) and isinstance(j, int)
            and not isinstance(i, bool) and not isinstance(j, bool)
            and 0 <= i < j,
            "VALUE_INVALID", "parent_bonds entries must be int pairs with i < j",
        )
    _require(
        isinstance(physical["parent_index"], int)
        and not isinstance(physical["parent_index"], bool)
        and physical["parent_index"] >= 0,
        "VALUE_INVALID", "physical_state.parent_index must be an int >= 0",
    )
    _check_optional_str(
        physical["parent_request_sha256"], "TYPE_INVALID",
        "physical_state.parent_request_sha256 type",
    )
    for key in ("last_accepted_lambda", "target_lambda", "scale"):
        number = _check_finite_number(
            physical[key], "VALUE_INVALID", f"physical_state.{key} must be a finite number"
        )
        _require(
            -1e-9 <= number <= 1.0 + 1e-9,
            "VALUE_INVALID", f"physical_state.{key} outside [0, 1]",
        )
    _require(
        physical["scale"] > 0.0, "VALUE_INVALID", "physical_state.scale must be > 0"
    )
    pending = physical["pending"]
    _require(
        isinstance(pending, list),
        "TYPE_INVALID", "physical_state.pending must be a list",
    )
    for entry in pending:
        _require(
            isinstance(entry, Mapping) and set(entry) == _PENDING_KEYS,
            "UNKNOWN_KEY", "pending entries must carry exactly {lambda, is_grid_point, rejections}",
        )
        _check_finite_number(
            entry["lambda"], "VALUE_INVALID", "pending lambda must be a finite number"
        )
        _require(
            isinstance(entry["is_grid_point"], bool),
            "TYPE_INVALID", "pending is_grid_point must be a bool",
        )
        _require(
            isinstance(entry["rejections"], int)
            and not isinstance(entry["rejections"], bool)
            and entry["rejections"] >= 0,
            "VALUE_INVALID", "pending rejections must be an int >= 0",
        )
    if physical["open_target"] is not None:
        number = _check_finite_number(
            physical["open_target"], "VALUE_INVALID", "physical_state.open_target type"
        )
        _require(
            -1e-9 <= number <= 1.0 + 1e-9,
            "VALUE_INVALID", "physical_state.open_target outside [0, 1]",
        )
    _require(
        physical["open_transition"] is None
        or isinstance(physical["open_transition"], Mapping),
        "TYPE_INVALID", "physical_state.open_transition must be null or a mapping",
    )

    budget = snapshot["budget"]
    max_wall = _check_finite_number(
        budget["max_wall_seconds"], "VALUE_INVALID", "budget.max_wall_seconds type"
    )
    _require(max_wall > 0.0, "VALUE_INVALID", "budget.max_wall_seconds must be > 0")
    _check_finite_number(
        budget["wall_elapsed_seconds"], "VALUE_INVALID",
        "budget.wall_elapsed_seconds type",
    )
    _require(
        budget["wall_elapsed_seconds"] >= 0.0,
        "VALUE_INVALID", "budget.wall_elapsed_seconds must be >= 0",
    )
    for key in ("max_host_requests", "request_count"):
        _require(
            isinstance(budget[key], int) and not isinstance(budget[key], bool)
            and budget[key] >= 0,
            "VALUE_INVALID", f"budget.{key} must be an int >= 0",
        )

    counters = snapshot["counters"]
    _require(
        counters["decision_index"] == episode["decision_index"],
        "CROSS_CHECK", "counters.decision_index must equal episode.decision_index",
    )
    indices = counters["accepted_prefix_frame_indices"]
    _require(
        isinstance(indices, list),
        "TYPE_INVALID", "counters.accepted_prefix_frame_indices must be a list",
    )
    for index in indices:
        _require(
            isinstance(index, int) and not isinstance(index, bool) and index >= 0,
            "VALUE_INVALID", "accepted_prefix_frame_indices entries must be int >= 0",
        )
    _require(
        isinstance(counters["recent_rejections"], int)
        and not isinstance(counters["recent_rejections"], bool)
        and counters["recent_rejections"] >= 0,
        "VALUE_INVALID", "counters.recent_rejections must be an int >= 0",
    )
    _require(
        isinstance(counters["candidate_already_awarded"], bool),
        "TYPE_INVALID", "counters.candidate_already_awarded must be a bool",
    )

    history = snapshot["history"]
    subdivision = history["subdivision"]
    _require(
        isinstance(subdivision, Mapping) and set(subdivision) == _SUBDIVISION_KEYS,
        "UNKNOWN_KEY", "history.subdivision keys drifted",
    )
    sub_scale = _check_finite_number(
        subdivision["scale"], "VALUE_INVALID", "history.subdivision.scale type"
    )
    min_scale = _check_finite_number(
        subdivision["min_scale"], "VALUE_INVALID", "history.subdivision.min_scale type"
    )
    _require(sub_scale > 0.0 and min_scale > 0.0,
             "VALUE_INVALID", "subdivision scales must be > 0")
    _require(min_scale <= sub_scale + 1e-12,
             "VALUE_INVALID", "history.subdivision.min_scale exceeds scale")
    for key in ("max_subdivisions", "max_rejections_per_target"):
        _require(
            isinstance(subdivision[key], int)
            and not isinstance(subdivision[key], bool)
            and subdivision[key] >= 0,
            "VALUE_INVALID", f"history.subdivision.{key} must be an int >= 0",
        )
    memory = history["rejection_memory"]
    _require(
        isinstance(memory, list),
        "TYPE_INVALID", "history.rejection_memory must be a list",
    )
    for entry in memory:
        _require(
            isinstance(entry, Mapping) and set(entry) == _REJECTION_MEMORY_KEYS,
            "UNKNOWN_KEY", "rejection_memory entries keys drifted",
        )
        _check_finite_number(
            entry["target_lambda"], "VALUE_INVALID", "rejection_memory lambda type"
        )
        _require(
            isinstance(entry["rejections"], int)
            and not isinstance(entry["rejections"], bool)
            and entry["rejections"] >= 0,
            "VALUE_INVALID", "rejection_memory rejections must be an int >= 0",
        )
        _require(
            isinstance(entry["is_grid_point"], bool),
            "TYPE_INVALID", "rejection_memory is_grid_point must be a bool",
        )
    refusals = history["open_refusal_reasons"]
    _require(
        isinstance(refusals, list)
        and all(isinstance(reason, str) for reason in refusals),
        "TYPE_INVALID", "history.open_refusal_reasons must be a list of strings",
    )

    rng = snapshot["rng"]
    _require(
        rng["kind"] in RNG_KINDS,
        "VALUE_INVALID", f"unknown rng kind {rng['kind']!r}",
    )
    if rng["exploration_seed"] is not None:
        _require(
            isinstance(rng["exploration_seed"], int)
            and not isinstance(rng["exploration_seed"], bool),
            "TYPE_INVALID", "rng.exploration_seed must be an int or null",
        )
    _require(
        isinstance(rng["determinism_note"], str) and bool(rng["determinism_note"]),
        "TYPE_INVALID", "rng.determinism_note must be a non-empty string",
    )

    parent_digest = snapshot["parent_state_sha256"]
    _require(
        parent_digest is None or (isinstance(parent_digest, str) and len(parent_digest) == 64),
        "VALUE_INVALID", "parent_state_sha256 must be null or a 64-hex digest",
    )


__all__ = [
    "ALLOWED_TOP",
    "DETERMINISM_NOTE",
    "NO_POLICY_NOTE",
    "RNG_KINDS",
    "SECTION_KEYS",
    "SNAPSHOT_SCHEMA_VERSION",
    "ControlSnapshotError",
    "bonds_to_payload",
    "build_snapshot_document",
    "control_code_identity",
    "control_document_digest",
    "geometry_to_payload",
    "origin_state_sha256",
    "pending_from_payload",
    "pending_to_payload",
    "policy_identity_from_decide",
    "snapshot_content_sha256",
    "validate_snapshot",
]
