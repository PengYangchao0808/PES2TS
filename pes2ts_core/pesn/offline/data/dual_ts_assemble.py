"""WP2-C dual-TS observation assembly: bind xTB receipts into teacher exports.

Closing step of WP2 (plan v4 §D).  Consumes the WP2-A fit50 teacher table
(``wp2_fit50/`` — 50 task + 50 teacher payloads whose states carry
``observation=None`` + a registered placeholder reason) together with the
WP2-B content-addressed xTB receipts (``wp2_observations/`` plus the 50 v1
cache-hit receipt paths recorded in ``coverage_nodes.json``) and produces:

* ``observation_index.json`` — the C2 coverage audit: per computation-identity
  node digest → receipt path, method/software/settings digest, E, |g| stats,
  cache/new attribution, nproc provenance and receipt status.  Coverage is
  counted BY RECEIPTS (method + software + settings digest), never by
  "validator passed" (C2 rule).
* ``wp2_fit50_observed/`` — a NEW export dir (the WP2-A originals are never
  overwritten): byte-copied task payloads + coverage/gaps artifacts, rewritten
  teacher payloads (each bound state's ``observation`` filled with a contract
  :class:`~pes2ts_core.pesn.contracts.dual_ts.ObservationRef` payload; the
  placeholder reason cleared), and the manifest.
* ``wp2_manifest.json`` — the WP2 dataset manifest binding the teacher table,
  observation index, lineage artifact sha, compute ledger totals, gaps
  registry, carry-forwards (insertions, bond cross-check, disputed reactions),
  denominators and tool versions.

Hard gates run in-loop on every rewritten payload pair: ``validate_record``
(task + teacher), A/B swap-digest identity, ``assert_task_teacher_identity``,
and the full-coverage assertion (typed :class:`CoverageIncompleteError` on any
missing node).  A node whose receipt is missing/invalid NEVER becomes a
zero-filled observation: in non-strict mode the state keeps ``null`` + reason
and is registered in the observed-gaps addendum — zero-fill is impossible by
construction and explicitly tested against.

Additive traceability: each bound state additionally carries
``observation_computation_identity`` + ``observation_receipt_path`` sibling
keys (export-level provenance; the contract loader drops unknown keys by
design — ``observation_index.json`` is the durable digest→receipt map).

No ACP/xTB execution, no network; the inputs are read-only.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np

from pes2ts_core.generation.planning.gradient_evidence import BOHR_ANGSTROM
from pes2ts_core.generation.planning.synchronized_path import digest
from pes2ts_core.pesn.contracts.dual_ts import (
    TASK_SCHEMA_VERSION,
    TEACHER_SCHEMA_VERSION,
    DualEndpointTask,
    DualTSTeacher,
    validate_record,
)
from pes2ts_core.pesn.offline.data.dual_ts_fit50 import (
    atom_order_hash,
    computation_identity,
    geometry_content_digest,
)
from pes2ts_core.pesn.offline.data.dual_ts_preprocess import (
    assert_task_teacher_identity,
)

WP2_ASSEMBLE_VERSION: Final[str] = "pes2ts_dual_ts_assemble_wp2_v0"
OBSERVATION_INDEX_SCHEMA: Final[str] = "pes2ts_dual_ts_observation_index_v0"
WP2_MANIFEST_SCHEMA: Final[str] = "pes2ts_dual_ts_wp2_manifest_v0"
GAPS_OBSERVED_SCHEMA: Final[str] = "pes2ts_dual_ts_gaps_registry_observed_v0"

#: ACP workflow recorded in every successful receipt result (new + v1 alike).
OBS_SOFTWARE: Final[str] = "OrcaGradient"

#: Gradient rounding matches the contract's ``ObservationRef.to_payload``.
GRADIENT_DECIMALS: Final[int] = 10

#: bohr→angstrom conversion check tolerance (recorded values are exact
#: double-precision quotients, so 1e-9 relative is generous).
CONVERSION_REL_TOL: Final[float] = 1e-9

OBSERVATION_INDEX_NAME: Final[str] = "observation_index.json"
MANIFEST_NAME: Final[str] = "wp2_manifest.json"
GAPS_OBSERVED_NAME: Final[str] = "gaps_registry_observed.json"

#: Receipt status that counts as bound (C2: receipts, not validators).
RECEIPT_STATUS_OK: Final[str] = "ok"

#: In-memory record key carrying the receipt's gradient rows; stripped
#: before serialization (the index stores stats, never full arrays).
_GRADIENT_KEY: Final[str] = "_gradient_rows"


# ---------------------------------------------------------------------------
# Typed errors
# ---------------------------------------------------------------------------


class AssembleError(ValueError):
    """Typed rejection for assembly/binding violations."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class CoverageIncompleteError(AssembleError):
    """Full-coverage gate: at least one node has no valid receipt."""

    def __init__(self, missing: list[dict]):
        heads = ", ".join(
            f"{m['node_digest'][:12]}({m['receipt_status']})"
            for m in missing[:10])
        super().__init__(
            "COVERAGE_INCOMPLETE",
            f"{len(missing)} node(s) without a valid receipt "
            f"(first: {heads}); full coverage required — nothing exported "
            "under a full-coverage claim",
        )
        self.missing = list(missing)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path) -> str:
    """sha256 over the file's raw bytes."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _load_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssembleError("INPUT_UNREADABLE", f"{path}: {exc!r}") from exc


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(len(values)),
        "min": round(float(arr.min()), 6),
        "median": round(float(np.median(arr)), 6),
        "max": round(float(arr.max()), 6),
        "mean": round(float(arr.mean()), 6),
    }


# ---------------------------------------------------------------------------
# Receipt validation (same binding rule as the WP2-B driver, in-module so the
# core package never imports from scripts/)
# ---------------------------------------------------------------------------


def validate_receipt_pair(request: Any, result: Any,
                          n_atoms: int) -> tuple[bool, str | None]:
    """Validate a stored ``request.json`` + ``result.json`` pair.

    Mirrors the WP2-B driver's ``validate_receipt`` exactly: request binding
    re-digest (excluding the recorded sha fields and additive ``pes2ts_``
    annotations), result⟷request sha match, ``success is True``, finite
    hartree energy, and an (n_atoms, 3) finite gradient.  Values are never
    substituted — a failed check is a reason string, never a zero-fill.
    """
    if not isinstance(request, dict):
        return False, "request_not_a_mapping"
    if not isinstance(result, dict):
        return False, "result_not_a_mapping"
    binding = str(request.get("request_sha256") or "")
    if not binding:
        return False, "missing_request_sha256"
    rebind_payload = {
        k: v for k, v in request.items()
        if k != "request_sha256" and k != "acp_request_sha256"
        and not str(k).startswith("pes2ts_")
    }
    try:
        rebind = digest(rebind_payload)
    except (TypeError, ValueError) as exc:
        return False, f"request_rebind_failed:{exc}"
    if rebind != binding:
        return False, f"request_binding_mismatch:stored={binding}:rebind={rebind}"
    if str(result.get("request_sha256") or "") != binding:
        return False, "result_request_sha256_mismatch"
    if result.get("success") is not True:
        return False, f"success_not_true:{result.get('success')!r}"
    if _finite(result.get("energy")) is None:
        return False, f"energy_not_finite:{result.get('energy')!r}"
    gradient = result.get("gradient_hartree_per_angstrom")
    if not isinstance(gradient, list) or len(gradient) != n_atoms:
        return False, (
            f"gradient_missing_or_wrong_shape:n_atoms={n_atoms}:"
            f"got={len(gradient) if isinstance(gradient, list) else type(gradient).__name__}")
    for i, row in enumerate(gradient):
        if not isinstance(row, list) or len(row) != 3:
            return False, f"gradient_row_{i}_not_3_components"
        for j, v in enumerate(row):
            if _finite(v) is None:
                return False, f"gradient_[{i}][{j}]_not_finite:{v!r}"
    return True, None


def validate_receipt_binding(request: Any, result: Any) -> tuple[bool, str | None]:
    """Binding half of :func:`validate_receipt_pair` (no gradient check):
    mappings, request re-digest, result⟷request sha, success, finite E."""
    if not isinstance(request, dict):
        return False, "request_not_a_mapping"
    if not isinstance(result, dict):
        return False, "result_not_a_mapping"
    binding = str(request.get("request_sha256") or "")
    if not binding:
        return False, "missing_request_sha256"
    rebind_payload = {
        k: v for k, v in request.items()
        if k != "request_sha256" and k != "acp_request_sha256"
        and not str(k).startswith("pes2ts_")
    }
    try:
        rebind = digest(rebind_payload)
    except (TypeError, ValueError) as exc:
        return False, f"request_rebind_failed:{exc}"
    if rebind != binding:
        return False, f"request_binding_mismatch:stored={binding}:rebind={rebind}"
    if str(result.get("request_sha256") or "") != binding:
        return False, "result_request_sha256_mismatch"
    if result.get("success") is not True:
        return False, f"success_not_true:{result.get('success')!r}"
    if _finite(result.get("energy")) is None:
        return False, f"energy_not_finite:{result.get('energy')!r}"
    return True, None


def check_gradient_rows(rows: Any, n_atoms: int) -> tuple[bool, str | None]:
    """Shape/finite gate for final hartree/angstrom gradient rows."""
    if not isinstance(rows, list) or len(rows) != n_atoms:
        return False, (
            f"gradient_missing_or_wrong_shape:n_atoms={n_atoms}:"
            f"got={len(rows) if isinstance(rows, list) else type(rows).__name__}")
    for i, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != 3:
            return False, f"gradient_row_{i}_not_3_components"
        for j, v in enumerate(row):
            if _finite(v) is None:
                return False, f"gradient_[{i}][{j}]_not_finite:{v!r}"
    return True, None


def verify_unit_conversion(result: dict) -> dict:
    """Apply/verify the recorded bohr→angstrom gradient conversion.

    Prefers the receipt's own ``pes2ts_units.BOHR_ANGSTROM`` (WP2-B
    receipts); v1 cache receipts carry no units annotation, so the canonical
    :data:`~pes2ts_core.generation.planning.gradient_evidence.BOHR_ANGSTROM`
    constant is used and its source is reported honestly.  When the converted
    array is absent but the native hartree/bohr product exists, the
    conversion is applied (never left to the consumer).
    """
    evidence = result.get("gradient_evidence")
    native = None
    if isinstance(evidence, dict):
        native = evidence.get("gradient_hartree_per_bohr")
    units = result.get("pes2ts_units")
    if isinstance(units, dict) and _finite(units.get("BOHR_ANGSTROM")):
        bohr = float(units["BOHR_ANGSTROM"])
        source = "receipt_pes2ts_units"
    else:
        bohr = float(BOHR_ANGSTROM)
        source = "canonical_gradient_evidence_constant"
    converted = result.get("gradient_hartree_per_angstrom")
    out: dict = {
        "bohr_angstrom": bohr,
        "source": source,
        "verified": False,
        "applied": False,
        "reason": None,
    }
    if not isinstance(native, list):
        if not isinstance(converted, list):
            out["reason"] = "no_gradient_native_or_converted"
        else:
            out["reason"] = "no_native_gradient_evidence"
        return out
    native_arr = np.asarray(native, dtype=float)
    if converted is None:
        if native_arr.ndim != 2 or native_arr.shape[-1] != 3:
            out["reason"] = "native_gradient_not_(n,3)"
            return out
        out["gradient_hartree_per_angstrom"] = (native_arr / bohr).tolist()
        out["applied"] = True
        out["verified"] = True
        out["source"] = source + ":applied"
        return out
    if not isinstance(converted, list):
        out["reason"] = "converted_gradient_not_a_list"
        return out
    conv_arr = np.asarray(converted, dtype=float)
    if conv_arr.shape != native_arr.shape:
        out["reason"] = (f"conversion_shape_mismatch:"
                         f"native={native_arr.shape}:converted={conv_arr.shape}")
        return out
    ok = bool(np.allclose(native_arr / bohr, conv_arr,
                          rtol=CONVERSION_REL_TOL, atol=0.0))
    if not ok:
        err = float(np.max(np.abs(native_arr / bohr - conv_arr)))
        out["reason"] = f"conversion_mismatch:max_abs_err={err:.3e}"
        return out
    out["verified"] = True
    return out


# ---------------------------------------------------------------------------
# Coverage + receipt binding
# ---------------------------------------------------------------------------


def locate_receipt(node: dict, observations_dir: str | Path) -> Path:
    """Receipt directory for one coverage node.

    New WP2-B receipts are content-addressed under ``<observations_dir>/
    <digest[:2]>/<digest>/``; v1 cache hits use the receipt path recorded in
    the coverage node (fixed by the WP2-A cache audit).
    """
    digest_hex = str(node["computation_identity"])
    cache = node.get("cache") or {}
    if str(cache.get("status")) == "cache_hit":
        recorded = cache.get("receipt_path")
        if not recorded:
            raise AssembleError(
                "CACHE_HIT_WITHOUT_PATH",
                f"{node['node_id']}: cache_hit without recorded receipt_path")
        return Path(str(recorded))
    return Path(observations_dir) / digest_hex[:2] / digest_hex


def load_teacher_states(fit50_dir: str | Path,
                        reaction_id: str) -> dict[str, dict]:
    """state_id → state dict for one reaction's teacher payload."""
    teacher = _load_json(Path(fit50_dir) / f"{reaction_id}.teacher.json")
    return {str(s["state_id"]): s for s in teacher["states"]}


def bind_node_observation(node: dict, states_by_id: dict[str, dict],
                          observations_dir: str | Path) -> dict:
    """Bind one coverage node to its receipt (or an honest miss record).

    The geometry binding gate re-digests the geometry from the teacher
    payload state (the driver's gate already guaranteed identity — it is
    re-asserted here so a stale coverage file can never bind), and every
    state ref of the node must digest identically (shared-TS invariant).
    A mismatch is a data bug and aborts with a typed error; a missing or
    invalid RECEIPT is a coverage miss and is returned honestly.  The
    receipt's gradient rows are attached under the private ``_gradient_rows``
    key (stripped before serialization).
    """
    node_digest = str(node["computation_identity"])
    refs = node.get("state_refs") or []
    if not refs:
        raise AssembleError("NODE_WITHOUT_STATE_REFS", str(node["node_id"]))
    ref_states = []
    for ref in refs:
        state = states_by_id.get(str(ref["state_id"]))
        if state is None:
            raise AssembleError(
                "STATE_REF_UNRESOLVED",
                f"{node['node_id']}: state {ref['state_id']} not in teacher")
        ref_states.append(state)
    coords = np.asarray(ref_states[0]["coordinates"], dtype=float)
    z = np.asarray(ref_states[0]["atomic_numbers"], dtype=int)
    maps = [str(m) for m in ref_states[0]["atom_map_ids"]]
    geom_d = geometry_content_digest(coords)
    if geom_d != str(node["geometry_content_digest"]):
        raise AssembleError(
            "BINDING_MISMATCH",
            f"{node['node_id']}: geometry digest recomputed={geom_d} "
            f"stored={node['geometry_content_digest']}")
    order = atom_order_hash(z, maps)
    if order != str(node["atom_order_hash"]):
        raise AssembleError(
            "BINDING_MISMATCH",
            f"{node['node_id']}: atom order hash recomputed={order} "
            f"stored={node['atom_order_hash']}")
    identity = computation_identity(
        geom_d, order, str(node["method"]), int(node["charge"]),
        int(node["multiplicity"]), str(node["operation"]))
    if identity["digest"] != node_digest:
        raise AssembleError(
            "BINDING_MISMATCH",
            f"{node['node_id']}: computation identity recomputed="
            f"{identity['digest']} stored={node_digest}")
    for state, ref in zip(ref_states[1:], refs[1:]):
        other = geometry_content_digest(
            np.asarray(state["coordinates"], dtype=float))
        if other != geom_d:
            raise AssembleError(
                "BINDING_MISMATCH",
                f"{node['node_id']}: state ref {ref['state_id']} digests "
                f"{other} != node geometry {geom_d} (TS dedupe invariant)")

    n_atoms = int(len(z))
    record: dict = {
        "node_digest": node_digest,
        "node_id": str(node["node_id"]),
        "reaction_id": str(node["reaction_id"]),
        "geometry_content_digest": geom_d,
        "atom_order_hash": order,
        "method": str(node["method"]),
        "operation": str(node["operation"]),
        "is_ts": bool(node.get("is_ts")),
        "inserted": bool(node.get("inserted")),
        "state_ids": [str(r["state_id"]) for r in refs],
        "n_atoms": n_atoms,
        "attribution": ("v1_cache" if str((node.get("cache") or {})
                                          .get("status")) == "cache_hit"
                        else "wp2_computed"),
        "receipt_path": None,
        "receipt_status": "missing",
        "energy_hartree": None,
        "gradient_max_abs": None,
        "gradient_mean_abs": None,
        "gradient_std_abs": None,
        "software": None,
        "settings_digest": None,
        "settings_provenance": None,
        "nproc": None,
        "duration_seconds": None,
        "driver_version": None,
        "conversion": None,
        _GRADIENT_KEY: None,
    }
    receipt_dir = locate_receipt(node, observations_dir)
    request_path = receipt_dir / "request.json"
    result_path = receipt_dir / "result.json"
    if not request_path.is_file() or not result_path.is_file():
        record["receipt_status"] = "missing"
        return record
    try:
        request = _load_json(request_path)
        result = _load_json(result_path)
    except AssembleError as exc:
        record["receipt_status"] = f"invalid:receipt_unreadable:{exc.detail}"
        return record
    ok, reason = validate_receipt_binding(request, result)
    if not ok:
        record["receipt_status"] = f"invalid:{reason}"
        return record
    conversion = verify_unit_conversion(result)
    if not conversion["verified"]:
        record["receipt_status"] = f"invalid:{conversion['reason']}"
        record["conversion"] = {
            k: conversion[k] for k in ("source", "verified", "applied",
                                       "reason")}
        return record
    gradient_rows = result["gradient_hartree_per_angstrom"]
    if conversion.get("applied"):
        gradient_rows = conversion["gradient_hartree_per_angstrom"]
    ok, reason = check_gradient_rows(gradient_rows, n_atoms)
    if not ok:
        record["receipt_status"] = f"invalid:{reason}"
        return record
    g = np.asarray(gradient_rows, dtype=float)
    settings = request.get("pes2ts_settings")
    if isinstance(settings, dict) and settings.get("settings_digest"):
        record["settings_digest"] = str(settings["settings_digest"])
        record["settings_provenance"] = "receipt_pes2ts_settings"
    else:
        # Receipts without the settings annotation (v1 cache hits and any
        # un-annotated pair) get an honest null — never a fabricated digest
        # (C2 counts by method/software/settings).
        record["settings_digest"] = None
        record["settings_provenance"] = "receipt_settings_unrecorded"
    identity_ann = request.get("pes2ts_computation_identity") or {}
    nproc = request.get("nproc")
    record.update({
        "receipt_path": str(receipt_dir),
        "receipt_status": RECEIPT_STATUS_OK,
        "energy_hartree": float(result["energy"]),
        "gradient_max_abs": float(np.max(np.abs(g))),
        "gradient_mean_abs": float(np.mean(np.abs(g))),
        "gradient_std_abs": float(np.std(np.abs(g))),
        "software": str((result.get("acp") or {}).get("workflow")
                        or OBS_SOFTWARE),
        "nproc": (None if nproc is None else int(nproc)),
        "duration_seconds": (None if result.get("duration_seconds") is None
                             else float(result["duration_seconds"])),
        "driver_version": (str(identity_ann["driver_version"])
                           if identity_ann.get("driver_version") else None),
        "conversion": {
            "verified": bool(conversion["verified"]),
            "applied": bool(conversion.get("applied")),
            "source": str(conversion["source"]),
            "bohr_angstrom": float(conversion["bohr_angstrom"]),
        },
        _GRADIENT_KEY: [[float(v) for v in row] for row in gradient_rows],
    })
    return record


def observation_payload(record: dict) -> dict:
    """Contract ``ObservationRef`` payload for a bound node record."""
    rows = record.get(_GRADIENT_KEY)
    if not rows:
        raise AssembleError(
            "BOUND_RECORD_WITHOUT_GRADIENT", str(record["node_digest"]))
    return {
        "status": "ok",
        "missing_reason": None,
        "energy_hartree": float(record["energy_hartree"]),
        "gradient": [
            [round(float(v), GRADIENT_DECIMALS) for v in row]
            for row in rows
        ],
        "method_id": str(record["method"]),
        "software": str(record["software"] or OBS_SOFTWARE),
        "settings_digest": (record["settings_digest"] or ""),
    }


def public_record(record: dict) -> dict:
    """Record without the private in-memory gradient key."""
    return {k: v for k, v in record.items() if k != _GRADIENT_KEY}


# ---------------------------------------------------------------------------
# Teacher payload rewriting
# ---------------------------------------------------------------------------


def bind_teacher_payload(teacher_payload: dict,
                         record_by_state: dict[str, dict]) -> tuple[dict, list]:
    """Fill observations in one loaded teacher payload dict.

    States whose node record is ``receipt_status == "ok"`` get a contract
    observation payload (+ additive traceability keys); all other states keep
    ``observation=None`` + their ORIGINAL registered reason and are returned
    in the misses list (registered in the gaps addendum — never zero-filled).
    The input dict is mutated in place; callers pass a freshly loaded payload
    (the WP2-A files are never touched).
    """
    misses: list[dict] = []
    for state in teacher_payload["states"]:
        sid = str(state["state_id"])
        record = record_by_state.get(sid)
        if record is None:
            raise AssembleError(
                "STATE_WITHOUT_NODE",
                f"{sid}: no coverage node references this state")
        if record["receipt_status"] == RECEIPT_STATUS_OK:
            state["observation"] = observation_payload(record)
            state["observation_missing_reason"] = None
            # Additive export-level traceability (contract loader drops
            # unknown keys; the observation index is the durable map).
            state["observation_computation_identity"] = \
                record["node_digest"]
            state["observation_receipt_path"] = record["receipt_path"]
        else:
            misses.append({
                "state_id": sid,
                "node_digest": record["node_digest"],
                "node_id": record["node_id"],
                "receipt_status": record["receipt_status"],
                "kept_reason": state.get("observation_missing_reason"),
            })
    return teacher_payload, misses


# ---------------------------------------------------------------------------
# Hard gates
# ---------------------------------------------------------------------------


def assert_swap_digest_identity(task_payload: dict) -> None:
    """A/B swap must not change the task content digest (unordered pair)."""
    swapped = dict(task_payload)
    swapped["endpoints"] = [
        dict(task_payload["endpoints"][1]),
        dict(task_payload["endpoints"][0]),
    ]
    d1 = DualEndpointTask.from_payload(dict(task_payload)).content_digest()
    d2 = DualEndpointTask.from_payload(swapped).content_digest()
    if d1 != d2:
        raise AssembleError(
            "SWAP_DIGEST_MISMATCH",
            f"{task_payload.get('reaction_id')}: digest {d1[:12]} != swapped "
            f"{d2[:12]}")


def validate_rewritten_pair(task_payload: dict,
                            teacher_payload: dict) -> None:
    """All per-reaction hard gates on one rewritten payload pair."""
    rid = str(task_payload.get("reaction_id"))
    validate_record(TASK_SCHEMA_VERSION, task_payload)
    validate_record(TEACHER_SCHEMA_VERSION, teacher_payload)
    assert_swap_digest_identity(task_payload)
    task = DualEndpointTask.from_payload(dict(task_payload))
    teacher = DualTSTeacher.from_payload(dict(teacher_payload))
    assert_task_teacher_identity(task, teacher)
    if task.reaction_id != teacher.reaction_id:
        raise AssembleError("REACTION_MISMATCH",
                            f"{rid}: task/teacher reaction ids disagree")


# ---------------------------------------------------------------------------
# Full assembly
# ---------------------------------------------------------------------------


def assemble_observations(fit50_dir: str | Path,
                          observations_dir: str | Path,
                          *,
                          require_full_coverage: bool = True) -> dict:
    """Bind every coverage node to its receipt; return the assembly bundle.

    Bundle keys: ``index`` (the serializable observation index document),
    ``record_by_state`` (state_id → full record incl. private gradient rows),
    ``coverage`` (the parsed coverage_nodes document), ``teachers`` (loaded
    state maps).  Raises :class:`CoverageIncompleteError` when
    ``require_full_coverage`` is set and any node lacks a valid receipt
    (typed, with the miss list).  With the flag off the misses are returned
    inside the index for an honest partial export (states keep null+reason).
    """
    fit50_dir = Path(fit50_dir)
    coverage = _load_json(fit50_dir / "coverage_nodes.json")
    nodes: list[dict] = coverage["nodes"]

    teachers: dict[str, dict[str, dict]] = {}
    records: list[dict] = []
    record_by_state: dict[str, dict] = {}
    for node in nodes:
        rid = str(node["reaction_id"])
        if rid not in teachers:
            teachers[rid] = load_teacher_states(fit50_dir, rid)
        record = bind_node_observation(node, teachers[rid], observations_dir)
        records.append(record)
        for sid in record["state_ids"]:
            if sid in record_by_state:
                raise AssembleError(
                    "STATE_REFERENCED_TWICE",
                    f"{sid}: referenced by nodes "
                    f"{record_by_state[sid]['node_id']} and {record['node_id']}")
            record_by_state[sid] = record

    missing = [r for r in records
               if r["receipt_status"] != RECEIPT_STATUS_OK]
    if require_full_coverage and missing:
        raise CoverageIncompleteError(missing)

    by_reaction: dict[str, list[dict]] = {}
    for record in records:
        by_reaction.setdefault(record["reaction_id"], []).append(record)
    per_reaction = {
        rid: {
            "nodes": len(rs),
            "bound": sum(1 for r in rs
                         if r["receipt_status"] == RECEIPT_STATUS_OK),
            "complete": all(r["receipt_status"] == RECEIPT_STATUS_OK
                            for r in rs),
        }
        for rid, rs in sorted(by_reaction.items())
    }

    bound = [r for r in records if r["receipt_status"] == RECEIPT_STATUS_OK]
    attribution_counts: dict[str, int] = {}
    settings_counts: dict[str, int] = {}
    nproc_counts: dict[str, int] = {}
    for record in bound:
        key = record["attribution"]
        attribution_counts[key] = attribution_counts.get(key, 0) + 1
        skey = record["settings_digest"] or "unrecorded_v1_cache"
        settings_counts[skey] = settings_counts.get(skey, 0) + 1
        nkey = str(record["nproc"]) if record["nproc"] is not None else "unknown"
        nproc_counts[nkey] = nproc_counts.get(nkey, 0) + 1

    total = len(nodes)
    n_bound = len(bound)
    index = {
        "schema_version": OBSERVATION_INDEX_SCHEMA,
        "tool_version": WP2_ASSEMBLE_VERSION,
        "generated_utc": _utc_now(),
        "coverage_nodes_schema": str(coverage.get("schema_version")),
        "observations_dir": str(observations_dir),
        "identity_scheme": coverage.get("identity_scheme"),
        "summary": {
            "total_nodes": total,
            "bound_nodes": n_bound,
            "missing_nodes": len(missing),
            "coverage_fraction": (round(n_bound / total, 6) if total else 0.0),
            "failures": len(missing),
            "attribution_counts": attribution_counts,
            "settings_digest_counts": settings_counts,
            "nproc_provenance": nproc_counts,
            "per_reaction_complete": (
                f"{sum(1 for v in per_reaction.values() if v['complete'])}"
                f"/{len(per_reaction)}"),
            "energy_hartree": _stats(
                [r["energy_hartree"] for r in bound]),
            "gradient_max_abs": _stats(
                [r["gradient_max_abs"] for r in bound]),
            "gradient_mean_abs": _stats(
                [r["gradient_mean_abs"] for r in bound]),
            "gradient_std_abs": _stats(
                [r["gradient_std_abs"] for r in bound]),
        },
        "per_reaction": per_reaction,
        "missing_nodes": [
            {k: m[k] for k in ("node_digest", "node_id", "reaction_id",
                               "receipt_status")}
            for m in missing
        ],
        "nodes": [public_record(r) for r in records],
    }
    return {
        "index": index,
        "record_by_state": record_by_state,
        "coverage": coverage,
        "teachers": teachers,
    }


def _copy_bytes(src: Path, dst: Path) -> None:
    shutil.copyfile(src, dst)


def _resolve_lineage_path(lineage: dict, fit50_dir: Path) -> Path | None:
    """Locate the lineage artifact (absolute / CWD / repo-root relative)."""
    raw = Path(str(lineage.get("path", "")))
    if not str(raw):
        return None
    candidates = [raw] if raw.is_absolute() else [
        Path.cwd() / raw,
        fit50_dir.parents[2] / raw,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def write_observed_export(
    fit50_dir: str | Path,
    observations_dir: str | Path,
    out_dir: str | Path,
    assembled: dict,
) -> dict:
    """Write the new export dir + manifest; return the manifest document.

    Task payloads, coverage_nodes/coverage_summary/gaps_registry are copied
    byte-for-byte; teacher payloads are rewritten with bound observations;
    the gaps addendum registers any (expected-zero) unbound states; the
    manifest is written last so it can digest everything it binds.
    """
    fit50_dir = Path(fit50_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = assembled["index"]
    record_by_state = assembled["record_by_state"]
    teachers = assembled["teachers"]

    # -- rewritten teacher payloads + hard gates (in-loop, per reaction) ----
    total_misses: list[dict] = []
    bound_states = 0
    states_total = 0
    teacher_digests: dict[str, str] = {}
    for reaction_id in sorted(teachers):
        original = _load_json(fit50_dir / f"{reaction_id}.teacher.json")
        states_total += len(original["states"])
        payload, misses = bind_teacher_payload(original, record_by_state)
        total_misses.extend(misses)
        bound_states += sum(1 for s in payload["states"]
                            if s.get("observation") is not None)
        task_payload = _load_json(fit50_dir / f"{reaction_id}.task.json")
        validate_rewritten_pair(task_payload, payload)
        teacher_path = out_dir / f"{reaction_id}.teacher.json"
        teacher_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1),
            encoding="utf-8")
        # reload-from-disk gate: a serialization bug cannot pass silently
        reloaded = _load_json(teacher_path)
        validate_record(TEACHER_SCHEMA_VERSION, reloaded)
        teacher_digests[reaction_id] = sha256_file(teacher_path)
        _copy_bytes(fit50_dir / f"{reaction_id}.task.json",
                    out_dir / f"{reaction_id}.task.json")

    # -- byte-copied coverage + gaps artifacts ------------------------------
    copied: dict[str, str] = {}
    for name in ("coverage_nodes.json", "coverage_summary.json",
                 "gaps_registry.json"):
        _copy_bytes(fit50_dir / name, out_dir / name)
        copied[name] = sha256_file(out_dir / name)

    # -- gaps addendum (additive; expected zero missing states) -------------
    gaps = _load_json(fit50_dir / "gaps_registry.json")
    gaps_observed = dict(gaps)
    gaps_observed["schema_version"] = GAPS_OBSERVED_SCHEMA
    gaps_observed["wp2_assemble"] = {
        "tool_version": WP2_ASSEMBLE_VERSION,
        "observation_binding": {
            "states_total": states_total,
            "states_bound": bound_states,
            "states_still_missing": len(total_misses),
            "missing": total_misses,
            "zero_fill_used": False,
        },
    }
    gaps_observed_path = out_dir / GAPS_OBSERVED_NAME
    gaps_observed_path.write_text(
        json.dumps(gaps_observed, ensure_ascii=False, indent=1),
        encoding="utf-8")

    # -- observation index ---------------------------------------------------
    index_path = out_dir / OBSERVATION_INDEX_NAME
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")

    # -- manifest (last: digests everything it binds) ------------------------
    manifest = build_manifest(
        fit50_dir=fit50_dir,
        observations_dir=Path(observations_dir),
        out_dir=out_dir,
        index=index,
        copied_digests=copied,
        teacher_digests=teacher_digests,
        gaps_observed_path=gaps_observed_path,
        index_path=index_path,
        states_total=states_total,
        bound_states=bound_states,
        states_still_missing=len(total_misses),
    )
    manifest_path = out_dir / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest


def _extract_carry_forwards(gaps_registry: dict,
                            coverage_summary: dict) -> dict:
    """Insertion / bond-cross-check / disputed carry-forwards from artifacts."""
    disputed: list[dict] = []
    verified = note = disputed_n = 0
    for entry in gaps_registry.get("reactions", []):
        rid = str(entry.get("reaction_id"))
        bond = (entry.get("items") or {}).get("bond_cross_check") or {}
        outcome = str(bond.get("outcome"))
        if outcome == "geometry_verified":
            verified += 1
        elif outcome == "accept_with_note":
            note += 1
        elif outcome == "disputed":
            disputed_n += 1
            disputed.append({
                "reaction_id": rid,
                "disputed_bonds": bond.get("disputed_bonds", []),
                "mapping_problems": (entry.get("items") or {})
                .get("mapping_problems", []),
            })
    ins = coverage_summary.get("insertions") or {}
    return {
        "insertions": {
            "reactions_with_insertion": ins.get("reactions_with_insertion"),
            "nodes_inserted_total": ins.get("nodes_inserted_total"),
            "note": "wp2 midpoint insertion (a_max=0.1 Å); inserted nodes "
                    "carry inserted:true and are part of the denominator",
        },
        "bond_cross_check": {
            "geometry_verified": verified,
            "accept_with_note": note,
            "disputed": disputed_n,
            "note": "weak-match policy: disputed reactions stay eligible "
                    "with task.eligibility.bond_authority=disputed; the WP3 "
                    "feature consumer must honor the flag",
        },
        "disputed_reactions": disputed,
    }


def build_manifest(
    *,
    fit50_dir: Path,
    observations_dir: Path,
    out_dir: Path,
    index: dict,
    copied_digests: dict[str, str],
    teacher_digests: dict[str, str],
    gaps_observed_path: Path,
    index_path: Path,
    states_total: int,
    bound_states: int,
    states_still_missing: int,
) -> dict:
    """The WP2 dataset manifest binding every WP2 artifact and number."""
    coverage_summary = _load_json(Path(fit50_dir) / "coverage_summary.json")
    gaps_registry = _load_json(Path(fit50_dir) / "gaps_registry.json")
    ledger_path = Path(observations_dir) / "compute_ledger.json"
    ledger = _load_json(ledger_path)

    runs = ledger.get("runs", [])
    totals = {
        "runs": len(runs),
        "computed": sum(int(r.get("computed", 0)) for r in runs),
        "failed": sum(int(r.get("failed", 0)) for r in runs),
        "skipped_resume_final_run": (int(runs[-1].get("skipped_resume", 0))
                                     if runs else 0),
        "skipped_cache_final_run": (int(runs[-1].get("skipped_cache", 0))
                                    if runs else 0),
        "interrupted_runs": sum(1 for r in runs if r.get("interrupted")),
        "cumulative_wall_seconds": round(
            sum(float(r.get("wall_seconds", 0.0)) for r in runs), 3),
        "settings_digests": sorted({
            str(r["settings_digest"]) for r in runs
            if r.get("settings_digest")}),
        "final_run_id": str(runs[-1].get("run_id")) if runs else None,
    }

    reactions_ok = int(coverage_summary["reactions"]["ok"])
    reactions_total = int(coverage_summary["reactions"]["attempted"])
    summary = index["summary"]

    lineage = coverage_summary["lineage_artifact"]
    lineage_path = _resolve_lineage_path(lineage, Path(fit50_dir))
    lineage_sha = sha256_file(lineage_path) if lineage_path else None

    driver_versions = sorted({
        str(rec["driver_version"]) for rec in index["nodes"]
        if rec["receipt_status"] == RECEIPT_STATUS_OK
        and rec.get("driver_version")
    })

    teacher_table_combined = hashlib.sha256(
        "\n".join(f"{k}:{v}" for k, v in sorted(teacher_digests.items()))
        .encode("utf-8")).hexdigest()

    return {
        "schema_version": WP2_MANIFEST_SCHEMA,
        "tool_version": WP2_ASSEMBLE_VERSION,
        "generated_utc": _utc_now(),
        "study": "dual_ts_v2 WP2 (fit50 full bilateral teacher + xTB "
                 "observations)",
        "fit50_dir": str(fit50_dir),
        "observations_dir": str(observations_dir),
        "teacher_table": {
            "dir": str(out_dir),
            "reactions": len(teacher_digests),
            "task_schema": TASK_SCHEMA_VERSION,
            "teacher_schema": TEACHER_SCHEMA_VERSION,
            "per_reaction_teacher_sha256": teacher_digests,
            "teacher_table_sha256_combined": teacher_table_combined,
            "note": "teacher payloads carry bound observations; task "
                    "payloads + coverage/gaps artifacts are byte-copies of "
                    "the WP2-A originals (never overwritten)",
        },
        "observation_index": {
            "path": str(index_path),
            "sha256": sha256_file(index_path),
            "summary": summary,
        },
        "lineage_artifact": {
            "path": str(lineage.get("path")),
            "sha256_recorded": lineage.get("sha256"),
            "sha256_recomputed": lineage_sha,
            "matches": (lineage_sha == lineage.get("sha256")
                        if lineage_sha is not None else None),
        },
        "compute_ledger": {
            "path": str(ledger_path),
            "sha256": sha256_file(ledger_path),
            "totals": totals,
        },
        "gaps_registry": {
            "path": str(Path(out_dir) / "gaps_registry.json"),
            "sha256": copied_digests["gaps_registry.json"],
            "observed_addendum": {
                "path": str(gaps_observed_path),
                "sha256": sha256_file(gaps_observed_path),
                "states_still_missing": states_still_missing,
            },
        },
        "coverage_audit": {
            "rule": "C2: counted by receipts (method/software/"
                    "settings_digest), never by validator passes",
            "unique_nodes": summary["total_nodes"],
            "bound": summary["bound_nodes"],
            "missing": summary["missing_nodes"],
            "failures": summary["failures"],
            "coverage": f"{summary['bound_nodes']}/{summary['total_nodes']}",
            "coverage_fraction": summary["coverage_fraction"],
            "attribution_counts": summary["attribution_counts"],
            "settings_digest_counts": summary["settings_digest_counts"],
            "nproc_provenance": summary["nproc_provenance"],
            "per_reaction_complete": summary["per_reaction_complete"],
        },
        "carry_forwards": _extract_carry_forwards(
            gaps_registry, coverage_summary),
        "denominators": {
            "reaction_groups": reactions_total,
            "branches": int(coverage_summary["branches"]["count"]),
            "states": states_total,
            "states_from_coverage_summary":
                int(coverage_summary["branches"]["states_total"]),
            "dual_sided_complete": f"{reactions_ok}/{reactions_total}",
            "states_bound_to_observations": bound_states,
            "states_missing_observations": states_still_missing,
        },
        "observation_stats": {
            "energy_hartree": summary["energy_hartree"],
            "gradient_max_abs": summary["gradient_max_abs"],
            "gradient_mean_abs": summary["gradient_mean_abs"],
            "gradient_std_abs": summary["gradient_std_abs"],
            "unit_conversion": "gradient_hartree_per_angstrom = native "
                               "hartree/bohr / BOHR_ANGSTROM (verified per "
                               "receipt; source recorded per node)",
        },
        "tool_versions": {
            "assemble_tool": WP2_ASSEMBLE_VERSION,
            "coverage_tool": str(coverage_summary.get("tool_version")),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "bohr_angstrom": float(BOHR_ANGSTROM),
            "bohr_angstrom_source":
                "pes2ts_core.generation.planning.gradient_evidence",
            "obs_driver_versions": driver_versions,
        },
    }


def assemble(fit50_dir: str | Path,
             observations_dir: str | Path,
             out_dir: str | Path,
             *,
             require_full_coverage: bool = True) -> dict:
    """Full WP2-C assembly; returns the manifest document."""
    assembled = assemble_observations(
        fit50_dir, observations_dir,
        require_full_coverage=require_full_coverage)
    return write_observed_export(
        fit50_dir, observations_dir, out_dir, assembled)


__all__ = [
    "AssembleError",
    "CoverageIncompleteError",
    "CONVERSION_REL_TOL",
    "GAPS_OBSERVED_NAME",
    "GAPS_OBSERVED_SCHEMA",
    "MANIFEST_NAME",
    "OBSERVATION_INDEX_NAME",
    "OBSERVATION_INDEX_SCHEMA",
    "OBS_SOFTWARE",
    "RECEIPT_STATUS_OK",
    "WP2_ASSEMBLE_VERSION",
    "WP2_MANIFEST_SCHEMA",
    "assemble",
    "assemble_observations",
    "assert_swap_digest_identity",
    "bind_node_observation",
    "bind_teacher_payload",
    "build_manifest",
    "locate_receipt",
    "observation_payload",
    "public_record",
    "sha256_file",
    "validate_receipt_pair",
    "validate_receipt_binding",
    "check_gradient_rows",
    "validate_rewritten_pair",
    "verify_unit_conversion",
    "write_observed_export",
]
