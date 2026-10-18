"""WP2-A fit50 FULL dual-TS teacher build: coverage identity + cache audit.

Companion to :mod:`dual_ts_preprocess` (whose builders perform the frozen C5
mass metric, step-5 midpoint insertion, and the weak-match bond cross-check
via opt-in flags).  This module holds the observation-coverage machinery:

* FULL computation identity for every teacher state: geometry content digest
  (sha256 over the canonical round-10 exported-coords payload — the TS frame
  exports byte-identically from both branches, so the shared TS yields one
  identity) + atom order/map hash + method (GFN2-xTB) + charge/multiplicity
  + operation (E/g single point).
* TS dedupe: states are grouped by computation identity, so the shared TS
  frame becomes ONE node attributed to both branch terminal states.
* Cache-match audit against the v1 ACP receipts
  (``outputs/pesn/direct_egnn_v1_20261015/observations/``): a receipt can
  only count as a cache hit when its ``request.json`` carries sufficient
  identity info (full geometry content + elements + method + charge +
  multiplicity) AND a successful ``result.json``.  Anything less is honestly
  reported as ``to_compute`` — matches are never fabricated.
* Cost estimate from measured v1 per-QC durations (median of recorded
  ``duration_seconds``), falling back to an explicitly ASSUMED 3.5 s.

NO ACP/xTB execution happens here (that is the WP2-B lane); no network; the
v1 observations tree is only read.
"""
from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np

from pes2ts_core.pesn.contracts.dual_ts import (
    BOUNDARY_TEACHER_ENDPOINT,
    content_digest_of,
)
from pes2ts_core.pesn.offline.data.dual_ts_preprocess import (
    DualTSPrepOutcome,
    EXPORT_COORD_DECIMALS,
)

WP2_FIT50_VERSION: Final[str] = "pes2ts_dual_ts_fit50_wp2_v0"

#: QC method and operation bound into every computation identity.
METHOD_GFN2_XTB: Final[str] = "GFN2-xTB"
OPERATION_EG_SINGLE_POINT: Final[str] = "eg_single_point"  # E + g, 1 geometry

#: Per-QC wall-time assumption used ONLY when no measured v1 duration exists
#: (the coverage summary must say which basis was used).
ASSUMED_PER_QC_SECONDS: Final[float] = 3.5

COVERAGE_NODES_SCHEMA: Final[str] = "pes2ts_dual_ts_wp2_coverage_nodes_v0"
COVERAGE_SUMMARY_SCHEMA: Final[str] = "pes2ts_dual_ts_wp2_coverage_summary_v0"
GAPS_REGISTRY_SCHEMA_V1: Final[str] = "pes2ts_dual_ts_gaps_registry_v1"
RUN_SUMMARY_SCHEMA: Final[str] = "pes2ts_dual_ts_wp2_run_summary_v0"


# ---------------------------------------------------------------------------
# Computation identity
# ---------------------------------------------------------------------------


def geometry_content_digest(coordinates: np.ndarray) -> str:
    """sha256 over the canonical round-10 exported-coords payload.

    The same canonicalization as the teacher export
    (``EXPORT_COORD_DECIMALS``), so identical exported geometry yields
    identical bytes — the shared TS frame digests identically from both
    branches by construction.
    """
    payload = {
        "coordinates": np.round(np.asarray(coordinates, dtype=float),
                                EXPORT_COORD_DECIMALS).tolist(),
    }
    return content_digest_of(payload)


def atom_order_hash(atomic_numbers: np.ndarray,
                    atom_map_ids=None) -> str:
    """sha256 over the atom order (Z sequence) + shared atom mapping.

    When ``atom_map_ids`` is omitted, the canonical G1 mapping
    ``row i <-> map i+1`` (``m0001...``) is assumed — the same invariant the
    v1 ACP requests implicitly used for their element row order.
    """
    z = np.asarray(atomic_numbers, dtype=int).tolist()
    if atom_map_ids is None:
        maps = [f"m{i + 1:04d}" for i in range(len(z))]
    else:
        maps = [str(m) for m in atom_map_ids]
    return content_digest_of({"atomic_numbers": z, "atom_map_ids": maps})


def computation_identity(
    geometry_digest: str,
    atom_order: str,
    method: str,
    charge: int,
    multiplicity: int,
    operation: str,
) -> dict:
    """FULL computation-identity payload + its content digest."""
    payload = {
        "geometry_content_digest": str(geometry_digest),
        "atom_order_hash": str(atom_order),
        "method": str(method),
        "charge": int(charge),
        "multiplicity": int(multiplicity),
        "operation": str(operation),
    }
    return dict(payload, digest=content_digest_of(payload))


# ---------------------------------------------------------------------------
# v1 ACP receipt cache index + honest matching
# ---------------------------------------------------------------------------


def _elements_to_z(symbols: list) -> list[int] | None:
    from pes2ts_core.pesn.offline.data.endpoint_adapter import _ELEMENT_TO_Z

    out = []
    for symbol in symbols:
        z = _ELEMENT_TO_Z.get(str(symbol))
        if z is None:
            return None
        out.append(int(z))
    return out


def build_cache_index(observations_dir: str | Path) -> tuple[dict, dict]:
    """Index successful v1 E/g receipts by FULL computation identity.

    Walks ``<observations_dir>/<RXN_*>/<trial>/eval-*/``; a receipt enters
    the index only when ``request.json`` carries geometry content (full
    coordinate array + elements), method, charge, multiplicity, AND
    ``result.json`` records a successful gradient product.  Everything else
    is counted in the audit with its honest reason and NEVER matched.
    """
    root = Path(observations_dir)
    audit: dict = {
        "observations_dir": str(root),
        "reaction_dirs": 0,
        "eval_dirs": 0,
        "requests_total": 0,
        "requests_unreadable": 0,
        "requests_without_geometry_binding": 0,
        "unbind_reasons": {},
        "requests_without_result": 0,
        "unsuccessful_results": 0,
        "successful_results": 0,
        "elements_unknown": 0,
        "methods_seen": {},
        "durations_seconds": [],
        "indexed_identities": 0,
        "binding_note": (
            "v1 request.json embeds the full request geometry (float array) "
            "plus elements/method/charge/multiplicity, so content-based "
            "identity matching is possible; the same round-10 canonicalization "
            "as the teacher export is applied to both sides"),
    }
    index: dict[str, list[dict]] = {}

    def _unbind(reason: str) -> None:
        audit["requests_without_geometry_binding"] += 1
        audit["unbind_reasons"][reason] = \
            audit["unbind_reasons"].get(reason, 0) + 1

    if not root.is_dir():
        audit["binding_note"] += " (observations dir missing)"
        return index, audit

    import json

    for rxn_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        audit["reaction_dirs"] += 1
        for trial_dir in sorted(p for p in rxn_dir.iterdir() if p.is_dir()):
            for eval_dir in sorted(trial_dir.glob("eval-*")):
                if not eval_dir.is_dir():
                    continue
                audit["eval_dirs"] += 1
                request_path = eval_dir / "request.json"
                if not request_path.is_file():
                    continue
                audit["requests_total"] += 1
                try:
                    request = json.loads(request_path.read_text())
                except (json.JSONDecodeError, OSError):
                    audit["requests_unreadable"] += 1
                    continue
                geometry = request.get("geometry")
                elements = request.get("elements")
                method = request.get("method")
                charge = request.get("charge")
                multiplicity = request.get("multiplicity")
                if not isinstance(geometry, list) or not geometry:
                    _unbind("request_geometry_missing")
                    continue
                if not isinstance(elements, list) \
                        or len(elements) != len(geometry):
                    _unbind("request_elements_missing_or_length_mismatch")
                    continue
                if not isinstance(method, str) or not method:
                    _unbind("request_method_missing")
                    continue
                if charge is None or multiplicity is None:
                    _unbind("request_charge_or_multiplicity_missing")
                    continue
                zs = _elements_to_z(elements)
                if zs is None:
                    audit["elements_unknown"] += 1
                    _unbind("element_symbol_unknown")
                    continue
                audit["methods_seen"][method] = \
                    audit["methods_seen"].get(method, 0) + 1
                result_path = eval_dir / "result.json"
                if not result_path.is_file():
                    audit["requests_without_result"] += 1
                    continue
                try:
                    result = json.loads(result_path.read_text())
                except (json.JSONDecodeError, OSError):
                    audit["requests_unreadable"] += 1
                    continue
                if not bool(result.get("success")):
                    audit["unsuccessful_results"] += 1
                    continue
                audit["successful_results"] += 1
                duration = result.get("duration_seconds")
                if duration is None:
                    acp = result.get("acp") or {}
                    duration = acp.get("wall_seconds")
                if duration is not None:
                    audit["durations_seconds"].append(float(duration))
                geometry_digest = geometry_content_digest(
                    np.asarray(geometry))
                atom_order = atom_order_hash(np.asarray(zs, dtype=int))
                identity = computation_identity(
                    geometry_digest, atom_order, method,
                    int(charge), int(multiplicity), OPERATION_EG_SINGLE_POINT)
                receipt = {
                    "receipt_dir": str(eval_dir),
                    "reaction_dir": rxn_dir.name,
                    "request_sha256": request.get("request_sha256"),
                    "method": method,
                    "charge": int(charge),
                    "multiplicity": int(multiplicity),
                    "duration_seconds": (None if duration is None
                                         else float(duration)),
                }
                index.setdefault(identity["digest"], []).append(receipt)

    audit["indexed_identities"] = len(index)
    audit["durations_seconds"].sort()
    return index, audit


def match_cache(identity: dict, cache_index: dict | None) -> dict:
    """Honest cache verdict for one computation identity.

    Returns ``cache_hit`` (with the receipt path of the first successful
    v1 receipt) or ``to_compute`` — never a fabricated match.
    """
    if cache_index:
        hits = cache_index.get(str(identity["digest"])) or []
    else:
        hits = []
    if hits:
        first = hits[0]
        return {
            "status": "cache_hit",
            "receipt_path": first["receipt_dir"],
            "request_sha256": first["request_sha256"],
            "duration_seconds": first["duration_seconds"],
            "n_receipts": len(hits),
        }
    return {
        "status": "to_compute",
        "receipt_path": None,
        "request_sha256": None,
        "duration_seconds": None,
        "n_receipts": 0,
    }


# ---------------------------------------------------------------------------
# Coverage node records (TS dedupe by computation identity)
# ---------------------------------------------------------------------------


def node_records_for_reaction(
    outcome: DualTSPrepOutcome,
    *,
    method: str = METHOD_GFN2_XTB,
    charge: int = 0,
    multiplicity: int = 1,
    cache_index: dict | None = None,
) -> list[dict]:
    """One computation-identity node per unique teacher-state geometry.

    The shared TS frame (bit-identical in both branches) yields ONE node
    carrying both terminal state refs.  Inserted nodes are flagged from the
    builder's ``inserted_state_ids`` diagnostics.  Each node carries its
    honest cache verdict.
    """
    diagnostics = outcome.diagnostics or {}
    inserted_ids: set[str] = set()
    for ids in (diagnostics.get("inserted_state_ids") or {}).values():
        inserted_ids.update(str(s) for s in ids)

    by_identity: dict[str, dict] = {}
    for state in outcome.teacher.states:
        identity = computation_identity(
            geometry_content_digest(np.asarray(state.coordinates, dtype=float)),
            atom_order_hash(np.asarray(state.atomic_numbers, dtype=int),
                            state.atom_map_ids),
            method, int(charge), int(multiplicity),
            OPERATION_EG_SINGLE_POINT)
        key = identity["digest"]
        node = by_identity.get(key)
        if node is None:
            node = {
                "node_id": f"{outcome.reaction_id}|node|{key[:12]}",
                "reaction_id": outcome.reaction_id,
                "reaction_group_id": outcome.reaction_group_id,
                "geometry_content_digest": identity["geometry_content_digest"],
                "atom_order_hash": identity["atom_order_hash"],
                "method": identity["method"],
                "charge": identity["charge"],
                "multiplicity": identity["multiplicity"],
                "operation": identity["operation"],
                "computation_identity": key,
                "state_refs": [],
                "is_ts": False,
                "inserted": False,
                "inserted_state_ids": [],
                "cache": match_cache(identity, cache_index),
            }
            by_identity[key] = node
        is_terminal = state.boundary_type == BOUNDARY_TEACHER_ENDPOINT
        node["state_refs"].append({
            "state_id": state.state_id,
            "branch_id": state.branch_id,
            "step_index": int(state.step_index),
            "is_terminal": is_terminal,
            "inserted": state.state_id in inserted_ids,
        })
        node["is_ts"] = node["is_ts"] or is_terminal
        if state.state_id in inserted_ids:
            node["inserted"] = True
            node["inserted_state_ids"].append(state.state_id)

    nodes = sorted(by_identity.values(),
                   key=lambda n: (min((r["branch_id"], r["step_index"])
                                      for r in n["state_refs"]),))
    for node in nodes:
        node["state_refs"].sort(
            key=lambda r: (r["branch_id"], r["step_index"]))
    return nodes


# ---------------------------------------------------------------------------
# Cost estimate
# ---------------------------------------------------------------------------


def estimate_cost(
    nodes_to_compute: int,
    durations: list[float],
    *,
    workers: int = 2,
    threads_per_worker: int = 2,
) -> dict:
    """Per-QC time from measured v1 receipt durations, else ASSUMED 3.5 s.

    The wall-time estimate is ``nodes_to_compute * per_qc / workers``
    (the per-QC duration already embeds the per-job thread allocation,
    v1 ``acp.nproc=2``).
    """
    if durations:
        per_qc = float(np.median(np.asarray(durations, dtype=float)))
        basis = (f"measured: median duration_seconds over {len(durations)} "
                 "successful v1 GFN2-xTB receipt(s) in the v1 observations "
                 "cache")
        assumed = False
    else:
        per_qc = float(ASSUMED_PER_QC_SECONDS)
        basis = (f"assumed {ASSUMED_PER_QC_SECONDS} s per QC single point "
                 "(no measured durations found in v1 receipts)")
        assumed = True
    total = int(nodes_to_compute) * per_qc
    wall = total / float(workers)
    return {
        "nodes_to_compute": int(nodes_to_compute),
        "per_qc_seconds": round(per_qc, 6),
        "per_qc_basis": basis,
        "per_qc_is_assumed": assumed,
        "n_measured_durations": int(len(durations)),
        "qc_seconds_total": round(total, 3),
        "workers": int(workers),
        "threads_per_worker": int(threads_per_worker),
        "estimated_wall_seconds": round(wall, 3),
        "estimated_wall_hours": round(wall / 3600.0, 4),
    }


__all__ = [
    "ASSUMED_PER_QC_SECONDS",
    "COVERAGE_NODES_SCHEMA",
    "COVERAGE_SUMMARY_SCHEMA",
    "GAPS_REGISTRY_SCHEMA_V1",
    "METHOD_GFN2_XTB",
    "OPERATION_EG_SINGLE_POINT",
    "RUN_SUMMARY_SCHEMA",
    "WP2_FIT50_VERSION",
    "atom_order_hash",
    "build_cache_index",
    "computation_identity",
    "estimate_cost",
    "geometry_content_digest",
    "match_cache",
    "node_records_for_reaction",
]
