"""Resumable ACP xTB E/g observation driver for the WP2 dual-TS coverage nodes.

WP2-B part 1 (plan v4 §D WP2).  Consumes the WP2-A computation-identity
coverage list (``coverage_nodes.json``; 6017 nodes = 50 v1 cache hits +
5967 to_compute) and computes an E/g single point for every ``to_compute``
node through the ONLY execution backend (R3): the ACP ``OrcaGradient``
workflow (GFN2-xTB) via :class:`~pes2ts_core.integration.acp.gradient_backend.
ORCAGradientBackend` — the same backend ``DirectGradientObservation`` wraps.

Receipt layout (content-addressed by the FULL computation identity)::

    <out>/<digest[:2]>/<digest>/request.json   geometry + full computation
                                               identity + settings digest
    <out>/<digest[:2]>/<digest>/result.json    E [hartree] and g as returned
                                               by the backend (native ORCA
                                               product hartree/bohr in
                                               ``gradient_evidence``; backend
                                               converts /BOHR_ANGSTROM ->
                                               ``gradient_hartree_per_angstrom``;
                                               the conversion is recorded
                                               explicitly in ``pes2ts_units``)

Resume semantics: a node whose receipt exists under ``--out`` AND passes
validation (success + finite E + (n_atoms, 3) finite g + request binding
re-digest) is skipped.  Failed nodes land in the global ``failures.jsonl``
registry and are never silently retried on resume unless ``--retry-failed``
(which first moves the failed receipt aside, never deleting evidence).

Ledger ``compute_ledger.json`` appends one entry per run (multi-run aware);
``progress.jsonl`` is an append-only event stream so an interrupted run's
state stays inspectable.  Invalid receipts (wrong gradient shape, non-finite
values, missing fields) are NEVER zero-filled — they become registered
failures.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pes2ts_core.generation.planning.gradient_evidence import BOHR_ANGSTROM  # noqa: E402
from pes2ts_core.generation.planning.synchronized_path import digest  # noqa: E402
from pes2ts_core.integration.acp.gradient_backend import (  # noqa: E402
    ORCAGradientBackend,
    write_json,
)
from pes2ts_core.integration.acp.direct_observation import symbols_for  # noqa: E402
from pes2ts_core.pesn.offline.data.dual_ts_fit50 import (  # noqa: E402
    atom_order_hash,
    computation_identity,
    geometry_content_digest,
)

DRIVER_VERSION = "pes2ts_dual_ts_obs_compute_v0"
LEDGER_SCHEMA = "pes2ts_dual_ts_obs_compute_ledger_v0"

DEFAULT_COVERAGE = ROOT / "outputs/pesn/dual_ts_v2_20261015/wp2_fit50/coverage_nodes.json"
DEFAULT_OUT = ROOT / "outputs/pesn/dual_ts_v2_20261015/wp2_observations"
DEFAULT_ACP_CONFIG = ROOT / "config/pesn/direct_data_v1.yaml"

FAILURE_REGISTRY_NAME = "failures.jsonl"
LEDGER_NAME = "compute_ledger.json"
PROGRESS_NAME = "progress.jsonl"
FAILED_ATTEMPTS_DIR = "failed_attempts"


class CoverageBindingError(RuntimeError):
    """A coverage node's stored identity does not bind to its geometry."""


# ---------------------------------------------------------------------------
# Coverage loading + geometry binding
# ---------------------------------------------------------------------------


def _load_teacher_states(coverage_dir: Path, reaction_id: str) -> dict[str, dict]:
    teacher_path = coverage_dir / f"{reaction_id}.teacher.json"
    if not teacher_path.is_file():
        raise CoverageBindingError(f"teacher payload missing: {teacher_path}")
    teacher = json.loads(teacher_path.read_text(encoding="utf-8"))
    states = {}
    for state in teacher["states"]:
        states[str(state["state_id"])] = state
    return states


def build_node_payloads(coverage_path: str | Path) -> list[dict]:
    """Load coverage nodes and bind each to its teacher-state geometry.

    Every node is re-digested from the actual exported coordinates before any
    QC is spent: geometry content digest, atom order hash (with the stored
    atom mapping) and the full computation identity must all reproduce the
    values recorded by WP2-A, and every state ref of a node must digest to
    the same geometry (the shared-TS dedupe invariant).
    """
    coverage = json.loads(Path(coverage_path).read_text(encoding="utf-8"))
    coverage_dir = Path(coverage_path).resolve().parent
    teachers: dict[str, dict[str, dict]] = {}
    payloads: list[dict] = []
    for node in coverage["nodes"]:
        rid = str(node["reaction_id"])
        if rid not in teachers:
            teachers[rid] = _load_teacher_states(coverage_dir, rid)
        states = teachers[rid]
        refs = node["state_refs"]
        if not refs:
            raise CoverageBindingError(f"{node['node_id']}: no state_refs")
        ref_states = []
        for ref in refs:
            state = states.get(str(ref["state_id"]))
            if state is None:
                raise CoverageBindingError(
                    f"{node['node_id']}: state {ref['state_id']} not in teacher")
            ref_states.append(state)
        coords = np.asarray(ref_states[0]["coordinates"], dtype=float)
        z = np.asarray(ref_states[0]["atomic_numbers"], dtype=int)
        maps = [str(m) for m in ref_states[0]["atom_map_ids"]]
        geom_d = geometry_content_digest(coords)
        if geom_d != node["geometry_content_digest"]:
            raise CoverageBindingError(
                f"{node['node_id']}: geometry digest mismatch "
                f"recomputed={geom_d} stored={node['geometry_content_digest']}")
        order = atom_order_hash(z, maps)
        if order != node["atom_order_hash"]:
            raise CoverageBindingError(
                f"{node['node_id']}: atom order hash mismatch "
                f"recomputed={order} stored={node['atom_order_hash']}")
        identity = computation_identity(
            geom_d, order, str(node["method"]), int(node["charge"]),
            int(node["multiplicity"]), str(node["operation"]))
        if identity["digest"] != node["computation_identity"]:
            raise CoverageBindingError(
                f"{node['node_id']}: computation identity mismatch "
                f"recomputed={identity['digest']} "
                f"stored={node['computation_identity']}")
        for state, ref in zip(ref_states[1:], refs[1:]):
            other = geometry_content_digest(
                np.asarray(state["coordinates"], dtype=float))
            if other != geom_d:
                raise CoverageBindingError(
                    f"{node['node_id']}: state ref {ref['state_id']} digests "
                    f"{other} != node geometry {geom_d} (TS dedupe invariant)")
        payloads.append({
            "digest": str(node["computation_identity"]),
            "node_id": str(node["node_id"]),
            "reaction_id": rid,
            "coordinates": coords.tolist(),
            "atomic_numbers": z.tolist(),
            "atom_map_ids": maps,
            "charge": int(node["charge"]),
            "multiplicity": int(node["multiplicity"]),
            "method": str(node["method"]),
            "operation": str(node["operation"]),
            "state_refs": refs,
            "is_ts": bool(node["is_ts"]),
            "inserted": bool(node["inserted"]),
            "cache_status": str((node.get("cache") or {}).get("status")
                                or "to_compute"),
            "n_atoms": int(len(z)),
        })
    return payloads


# ---------------------------------------------------------------------------
# Receipt paths + validation
# ---------------------------------------------------------------------------


def receipt_dir(out_root: str | Path, digest_hex: str) -> Path:
    return Path(out_root) / digest_hex[:2] / digest_hex


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def validate_receipt(request: Any, result: Any, n_atoms: int) -> tuple[bool, str | None]:
    """Validate a stored ``request.json`` + ``result.json`` pair.

    Checks (in order, first failure is the reason):
    1. both documents parse to mappings;
    2. request binding: re-digesting the stored request (minus the recorded
       sha fields and the driver's additive ``pes2ts_`` annotations) must
       reproduce ``request_sha256``;
    3. ``result.request_sha256`` equals the request binding (result belongs
       to this request — never a foreign or stale receipt);
    4. ``success`` is exactly true;
    5. ``energy`` is a finite number (hartree);
    6. ``gradient_hartree_per_angstrom`` is an (n_atoms, 3) array of finite
       floats — missing/None/wrong-shape/non-finite gradients are failures;
       values are never substituted or zero-filled.

    Returns ``(ok, reason)``; ``reason`` is None exactly when ok.
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
        return False, f"success_not_true:{result.get('success')!r}:" \
                      f"{result.get('failure_class')}"
    energy = _finite_float(result.get("energy"))
    if energy is None:
        return False, f"energy_not_finite:{result.get('energy')!r}"
    gradient = result.get("gradient_hartree_per_angstrom")
    if not isinstance(gradient, list) or len(gradient) != n_atoms:
        return False, (f"gradient_missing_or_wrong_shape:"
                       f"n_atoms={n_atoms}:got="
                       f"{len(gradient) if isinstance(gradient, list) else type(gradient).__name__}")
    for i, row in enumerate(gradient):
        if not isinstance(row, list) or len(row) != 3:
            return False, f"gradient_row_{i}_not_3_components"
        for j, v in enumerate(row):
            if _finite_float(v) is None:
                return False, f"gradient_[{i}][{j}]_not_finite:{v!r}"
    return True, None


def load_receipt_pair(rd: Path) -> tuple[Any, Any]:
    request = json.loads((rd / "request.json").read_text(encoding="utf-8"))
    result = json.loads((rd / "result.json").read_text(encoding="utf-8"))
    return request, result


def receipt_state(out_root: str | Path, payload: dict) -> tuple[bool, str | None]:
    """``(valid, reason)`` for an existing receipt; ``(False, None)`` = absent."""
    rd = receipt_dir(out_root, payload["digest"])
    result_path = rd / "result.json"
    if not result_path.is_file():
        return False, None
    try:
        request, result = load_receipt_pair(rd)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"receipt_unreadable:{exc}"
    ok, reason = validate_receipt(request, result, int(payload["n_atoms"]))
    return ok, reason


# ---------------------------------------------------------------------------
# Receipt annotation (additive; never rewrites backend-recorded fields)
# ---------------------------------------------------------------------------


def settings_digest(acp: dict) -> str:
    payload = {
        "workflow": "OrcaGradient",
        "method": str(acp["method"]),
        "basis": "",
        "route_extras": ["EnGrad"],
        "extra_blocks": ['%xtb\n XTBINPUTSTRING "--iterations 2000"\nend'],
        "scf_convergence": "tight",
        "output_name": "grad",
        "timeout_seconds": float(acp["timeout_seconds"]),
        "nproc": int(acp["nproc"]),
        "acp_root": str(acp["root"]),
        "acp_python": str(acp["python"]),
        "acp_config_path": str(acp["config_path"]),
    }
    return digest(payload)


def identity_annotation(payload: dict) -> dict:
    return {
        "scheme": "pes2ts_dual_ts_wp2_computation_identity_v0",
        "computation_identity": payload["digest"],
        "geometry_content_digest": geometry_content_digest(
            np.asarray(payload["coordinates"], dtype=float)),
        "atom_order_hash": atom_order_hash(
            np.asarray(payload["atomic_numbers"], dtype=int),
            payload["atom_map_ids"]),
        "method": payload["method"],
        "operation": payload["operation"],
        "charge": payload["charge"],
        "multiplicity": payload["multiplicity"],
        "node_id": payload["node_id"],
        "reaction_id": payload["reaction_id"],
        "state_refs": payload["state_refs"],
        "driver_version": DRIVER_VERSION,
    }


def units_annotation() -> dict:
    return {
        "scheme": "pes2ts_dual_ts_obs_units_v0",
        "energy": "hartree",
        "gradient_hartree_per_angstrom": "hartree/angstrom",
        "gradient_native_field": "gradient_evidence.gradient_hartree_per_bohr",
        "gradient_native_unit": "hartree/bohr",
        "conversion": ("gradient_hartree_per_angstrom = "
                       "gradient_evidence.gradient_hartree_per_bohr / "
                       "BOHR_ANGSTROM"),
        "BOHR_ANGSTROM": BOHR_ANGSTROM,
        "conversion_source": ("pes2ts_core.generation.planning."
                              "gradient_evidence.BOHR_ANGSTROM"),
        "recorded_by": DRIVER_VERSION,
    }


def annotate_receipt(rd: Path, payload: dict, acp: dict) -> None:
    """Additively record identity + settings/units metadata on the receipt."""
    identity = identity_annotation(payload)
    settings = {
        "scheme": "pes2ts_dual_ts_obs_settings_v0",
        "settings_digest": settings_digest(acp),
        "workflow": "OrcaGradient",
        "method": str(acp["method"]),
        "timeout_seconds": float(acp["timeout_seconds"]),
        "nproc": int(acp["nproc"]),
        "acp_root": str(acp["root"]),
        "acp_python": str(acp["python"]),
        "acp_config_path": str(acp["config_path"]),
    }
    request_path = rd / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    for key, value in (("pes2ts_computation_identity", identity),
                       ("pes2ts_settings", settings)):
        if key not in request:
            request[key] = value
    write_json(request_path, request)
    result_path = rd / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    for key, value in (("pes2ts_computation_identity", identity),
                       ("pes2ts_units", units_annotation())):
        if key not in result:
            result[key] = value
    write_json(result_path, result)


def settle_failed_receipt(out_root: str | Path, digest_hex: str) -> str | None:
    """Move an invalid receipt aside (evidence preserved) before a retry."""
    rd = receipt_dir(out_root, digest_hex)
    if not rd.is_dir():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%f")
    quarantine = Path(out_root) / FAILED_ATTEMPTS_DIR / digest_hex
    quarantine.mkdir(parents=True, exist_ok=True)
    target = quarantine / f"attempt-{stamp}"
    rd.replace(target)
    return str(target)


# ---------------------------------------------------------------------------
# The real compute unit (executed in worker processes)
# ---------------------------------------------------------------------------


def compute_node(payload: dict, acp: dict, out_root: str) -> dict:
    """One node's E/g through ACP with the driver's full retry loop.

    The receipt directory is the content address ``<out>/<d[:2]>/<d>/``; the
    backend writes ``request.json`` before launch and ``result.json`` after
    (success OR failure), and replays an existing valid receipt instead of
    recomputing.  Invalid receipts are moved aside before each fresh attempt.
    """
    digest_hex = payload["digest"]
    started = time.monotonic()
    attempts_allowed = 1 + int(acp.get("retry", 0))
    attempt_logs: list[dict] = []
    result: dict = {"digest": digest_hex, "node_id": payload["node_id"],
                    "status": "failed", "attempts": 0, "seconds": 0.0,
                    "backend_seconds": None, "error": None,
                    "request_sha256": None, "energy": None,
                    "g_abs_max": None, "receipt_dir": None}
    for attempt in range(1, attempts_allowed + 1):
        result["attempts"] = attempt
        rd = receipt_dir(out_root, digest_hex)
        if (rd / "result.json").is_file():
            ok, reason = receipt_state(out_root, payload)
            if ok:
                result.update(status="resume_valid",
                              receipt_dir=str(rd))
                break
            settle_failed_receipt(out_root, digest_hex)
        try:
            backend = ORCAGradientBackend(
                charge=int(payload["charge"]),
                multiplicity=int(payload["multiplicity"]),
                elements=symbols_for(payload["atomic_numbers"]),
                folder=str(out_root),
                method=str(payload["method"]),
                timeout_seconds=float(acp["timeout_seconds"]),
                nproc=int(acp["nproc"]),
                acp_wiring={"root": acp["root"], "python": acp["python"],
                            "config_path": acp["config_path"]},
            )
            record = backend(
                np.asarray(payload["coordinates"], dtype=float),
                f"{digest_hex[:2]}/{digest_hex}")
        except Exception as exc:  # transport/launch faults -> attempt failure
            attempt_logs.append({"attempt": attempt, "ok": False,
                                 "error": f"backend_exception:{type(exc).__name__}:{exc}"})
            result["error"] = attempt_logs[-1]["error"]
            continue
        ok, reason = validate_receipt(
            _request_for_validation(payload, acp, record), record,
            int(payload["n_atoms"]))
        if ok:
            annotate_receipt(rd, payload, acp)
            gradient = np.asarray(record["gradient_hartree_per_angstrom"],
                                  dtype=float)
            result.update(status="computed",
                          backend_seconds=float(record.get("duration_seconds") or 0.0),
                          request_sha256=str(record.get("request_sha256")),
                          energy=float(record["energy"]),
                          g_abs_max=float(np.abs(gradient).max()),
                          receipt_dir=str(rd),
                          error=None)
            attempt_logs.append({"attempt": attempt, "ok": True,
                                 "backend_seconds": result["backend_seconds"]})
            break
        attempt_logs.append({"attempt": attempt, "ok": False,
                             "error": reason,
                             "failure_class": record.get("failure_class")})
        result["error"] = reason
    result["attempt_log"] = attempt_logs
    result["seconds"] = time.monotonic() - started
    return result


def _request_for_validation(payload: dict, acp: dict, record: dict) -> dict:
    """Load the stored request.json for in-process validation.

    ``compute_node`` calls the backend in-process, so the record alone lacks
    the request payload; the freshly written request.json is the source of
    truth and is passed to :func:`validate_receipt` UNMODIFIED — that function
    itself re-digests the request (minus the sha/annotation fields) and
    compares against the stored binding.
    """
    rd = receipt_dir(acp["out_root"], payload["digest"])
    return json.loads((rd / "request.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Run bookkeeping: progress, failures, ledger
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def append_jsonl(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(document, ensure_ascii=False,
                                allow_nan=False) + "\n")


def load_failure_registry(out_root: Path) -> dict[str, dict]:
    registry: dict[str, dict] = {}
    path = out_root / FAILURE_REGISTRY_NAME
    if not path.is_file():
        return registry
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        registry[str(entry["digest"])] = entry
    return registry


def progress_event(handle, run_id: str, payload: dict, status: str, *,
                   seconds: float | None = None, attempt: int | None = None,
                   error: str | None = None) -> None:
    event = {"ts": _utc_now(), "run_id": run_id,
             "node_digest": payload["digest"], "node_id": payload["node_id"],
             "status": status}
    if seconds is not None:
        event["seconds"] = round(float(seconds), 4)
    if attempt is not None:
        event["attempt"] = int(attempt)
    if error:
        event["error"] = str(error)[:500]
    handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def summarize_run_ledger(entry: dict) -> None:
    per_qc = sorted((entry.get("per_receipt_seconds") or {}).values())
    print(f"[wp2obs] run {entry['run_id']}: attempted={entry['attempted']} "
          f"computed={entry['computed']} failed={entry['failed']} "
          f"skipped_resume={entry['skipped_resume']} "
          f"skipped_resume_failed={entry['skipped_resume_failed']} "
          f"skipped_registered_failure={entry['skipped_registered_failure']} "
          f"skipped_cache={entry['skipped_cache']} wall={entry['wall_seconds']:.1f}s")
    if per_qc:
        print(f"[wp2obs] per-QC backend seconds over {len(per_qc)} computed "
              f"receipts: median={_median(per_qc):.3f} min={per_qc[0]:.3f} "
              f"max={per_qc[-1]:.3f} mean={sum(per_qc) / len(per_qc):.3f}")
    if entry["failed"]:
        print(f"[wp2obs] failures registered in "
              f"{Path(entry['out_root']) / FAILURE_REGISTRY_NAME}: "
              f"{entry['failed']} node(s); resume will skip them unless "
              f"--retry-failed")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_driver(args: argparse.Namespace,
               compute_fn: Callable[[dict, dict, str], dict] | None = None,
               ) -> dict:
    """Run the resumable compute loop; returns the ledger entry written.

    ``compute_fn`` is injectable for synthetic tests; the default is the real
    ACP-backed :func:`compute_node`.
    """
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    payloads = build_node_payloads(args.coverage)
    mpi_pin = None
    if compute_fn is None:
        acp = load_acp_wiring(args.acp_config, retry=args.retry)
        acp["out_root"] = str(out_root)
        mpi_pin = _pin_orca_mpi_launcher(acp)
        if mpi_pin:
            print(f"[wp2obs] pinned ACP MPI launcher to {mpi_pin} "
                  f"(CONFSEARCH_ORCA_MPI_PATH; recorded in ledger)")
    else:
        acp = dict(getattr(args, "_acp_test_settings", {}) or {})
        acp.setdefault("retry", args.retry)
        acp.setdefault("out_root", str(out_root))

    run_id = _utc_now()
    registry = load_failure_registry(out_root)
    progress_path = out_root / PROGRESS_NAME

    counts = {"attempted": 0, "computed": 0, "failed": 0,
              "skipped_resume": 0, "skipped_resume_failed": 0,
              "skipped_registered_failure": 0, "skipped_cache": 0}
    per_receipt_seconds: dict[str, float] = {}
    per_receipt_wall: dict[str, float] = {}
    retries_used = 0
    attempts: list[dict] = []

    with progress_path.open("a", encoding="utf-8") as progress:
        for payload in payloads:
            if payload["cache_status"] == "cache_hit":
                counts["skipped_cache"] += 1
                progress_event(progress, run_id, payload, "skipped_cache")
                continue
            ok, reason = receipt_state(out_root, payload)
            if ok:
                counts["skipped_resume"] += 1
                progress_event(progress, run_id, payload, "skipped_resume")
                continue
            if payload["digest"] in registry and not args.retry_failed:
                counts["skipped_registered_failure"] += 1
                progress_event(progress, run_id, payload,
                               "skipped_registered_failure",
                               error=registry[payload["digest"]].get("error"))
                continue
            if reason is not None:
                counts["skipped_resume_failed"] += 1
                progress_event(progress, run_id, payload,
                               "skipped_resume_failed", error=reason)
                if not args.retry_failed:
                    continue
            attempts.append(payload)

    limited = attempts[:args.limit] if args.limit else attempts
    with progress_path.open("a", encoding="utf-8") as progress:
        for payload in limited:
            progress_event(progress, run_id, payload, "started")

        started = time.monotonic()
        interrupted = False
        results: list[dict] = []
        try:
            if args.workers <= 1 or compute_fn is not None:
                for payload in limited:
                    results.append(compute_fn(payload, acp, str(out_root))
                                   if compute_fn is not None
                                   else compute_node(payload, acp, str(out_root)))
                    progress_event(progress, run_id, payload,
                                   results[-1]["status"],
                                   seconds=results[-1]["seconds"],
                                   attempt=results[-1]["attempts"],
                                   error=results[-1]["error"])
            else:
                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    futures = {pool.submit(compute_node, payload, acp,
                                           str(out_root)): payload
                               for payload in limited}
                    for future in as_completed(futures):
                        payload = futures[future]
                        result = future.result()
                        results.append(result)
                        progress_event(progress, run_id, payload,
                                       result["status"],
                                       seconds=result["seconds"],
                                       attempt=result["attempts"],
                                       error=result["error"])
        except (KeyboardInterrupt, Exception) as exc:  # noqa: BLE001
            interrupted = True
            progress_event(progress, run_id,
                           {"digest": "", "node_id": f"INTERRUPTED:{type(exc).__name__}"},
                           "run_interrupted", error=str(exc))
            raise
        finally:
            wall = time.monotonic() - started
            for result in results:
                counts["attempted"] += 1
                if result["status"] in ("computed", "resume_valid"):
                    counts["computed"] += 1
                    if result.get("backend_seconds") is not None:
                        per_receipt_seconds[result["digest"]] = round(
                            float(result["backend_seconds"]), 4)
                    per_receipt_wall[result["digest"]] = round(
                        float(result["seconds"]), 4)
                    retries_used += int(result["attempts"]) - 1
                else:
                    counts["failed"] += 1
                    retries_used += int(result["attempts"]) - 1
                    append_jsonl(out_root / FAILURE_REGISTRY_NAME, {
                        "ts": _utc_now(), "run_id": run_id,
                        "digest": result["digest"],
                        "node_id": result["node_id"],
                        "error": result.get("error"),
                        "attempts": int(result["attempts"]),
                    })
            ledger_path = out_root / LEDGER_NAME
            ledger = json.loads(ledger_path.read_text(encoding="utf-8")) \
                if ledger_path.is_file() else {
                    "schema_version": LEDGER_SCHEMA, "runs": []}
            cumulative = sum(float(r.get("wall_seconds") or 0.0)
                             for r in ledger["runs"])
            entry = {
                "run_id": run_id, "ended_utc": _utc_now(),
                "interrupted": interrupted,
                "coverage": str(args.coverage), "out_root": str(out_root),
                "workers": int(args.workers), "limit": args.limit,
                "retry": int(args.retry), "retry_failed": bool(args.retry_failed),
                "env_mpi_allow_run_as_root": bool(
                    getattr(args, "_mpi_root_override", False)),
                "env_mpi_launcher_pin": mpi_pin,
                "acp": {k: acp[k] for k in
                        ("root", "python", "config_path", "method",
                         "timeout_seconds", "nproc") if k in acp},
                "settings_digest": settings_digest(acp) if "root" in acp else None,
                **counts,
                "retries_used": retries_used,
                "per_receipt_seconds": per_receipt_seconds,
                "per_receipt_wall_seconds": per_receipt_wall,
                "wall_seconds": round(wall, 3),
                "cumulative_wall_seconds": round(cumulative + wall, 3),
            }
            ledger["runs"].append(entry)
            write_json(ledger_path, ledger)
    summarize_run_ledger(entry)
    return entry


def load_acp_wiring(config_path: str | Path, *, retry: int = 1) -> dict:
    from pes2ts_core.config_loader import load_config_mapping

    config = load_config_mapping(config_path)
    acp = config.get("acp") or {}
    if not all(acp.get(k) for k in ("root", "python", "config_path")):
        raise SystemExit(
            f"[wp2obs] {config_path}: acp section must define root, python "
            f"and config_path (found keys: {sorted(acp)})")
    config_path_value = Path(acp["config_path"])
    if not config_path_value.is_absolute():
        config_path_value = ROOT / config_path_value
    return {
        "root": str(acp["root"]),
        "python": str(acp["python"]),
        "config_path": str(config_path_value),
        "method": str(acp.get("method", "GFN2-xTB")),
        "timeout_seconds": float(acp.get("timeout_seconds", 120.0)),
        "nproc": int(acp.get("nproc", 2)),
        "retry": int(retry),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumable ACP xTB E/g computation driver for the WP2 "
                    "dual-TS coverage nodes (R3: ACP is the only backend).")
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE,
                        help="coverage_nodes.json from WP2-A")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="receipt root (content-addressed receipts + "
                             "ledger/progress/failure registry)")
    parser.add_argument("--workers", type=int, default=2,
                        help="concurrent ACP tasks (each uses acp.nproc=2 "
                             "threads; AGENTS.md start profile 2x2)")
    parser.add_argument("--limit", type=int, default=None,
                        help="compute at most N new nodes (smoke runs)")
    parser.add_argument("--retry", type=int, default=1,
                        help="extra attempts per node after the first")
    parser.add_argument("--acp-config", type=Path, default=DEFAULT_ACP_CONFIG,
                        help="PES2TS yaml whose acp: section carries the ACP "
                             "checkout root/python/config_path/method/timeout/"
                             "nproc")
    parser.add_argument("--retry-failed", action="store_true",
                        help="retry nodes present in the failure registry or "
                             "with invalid receipts (failed receipt moved to "
                             f"{FAILED_ATTEMPTS_DIR}/ first; never deleted)")
    return parser.parse_args(argv)


def _ensure_mpi_root_override() -> bool:
    """Allow OpenMPI to run as root for the ACP -> ORCA child processes.

    OpenMPI >= 4 refuses ``mpirun`` under root unless BOTH ``OMPI_ALLOW_RUN_AS_ROOT``
    and ``OMPI_ALLOW_RUN_AS_ROOT_CONFIRM`` are set; this machine's sessions
    run as root (WSL, /root/miniforge3 interpreter), and ORCA 5.0.4 invokes
    ``orca_util_mpi`` through mpirun for GFN2-xTB EnGrad jobs.  The override
    is applied to THIS process environment only (inherited by the worker
    processes and from there by ACP/ORCA), printed, and recorded in the run
    ledger — never silently.
    """
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        return False
    os.environ["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
    os.environ["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"
    return True


def _pin_orca_mpi_launcher(acp: dict) -> str | None:
    """Pin ACP's MPI launcher to the exact file the ACP config intends.

    ``executables.orca.mpi_path`` in the ACP config is a bin DIRECTORY, but
    ACP's resolver (:func:`cccp.software.resolve_mpirun_with_source`)
    expects a launcher FILE and silently falls back to the system OpenMPI
    when given a directory; the system launcher/ABI segfaults ORCA 5.0.4's
    ``orca_util_mpi``.  Setting ``CONFSEARCH_ORCA_MPI_PATH`` (ACP's
    documented override, consulted before PATH) to ``<mpi_path>/mpirun``
    restores the intended pin and makes ACP prepend the matching bin/lib
    directories itself.  Recorded in the run ledger — never silent.
    """
    try:
        config = yaml.safe_load(
            Path(acp["config_path"]).read_text(encoding="utf-8"))
        mpi_dir = str(((config.get("executables") or {}).get("orca")
                       or {}).get("mpi_path") or "")
    except (OSError, yaml.YAMLError):
        return None
    if not mpi_dir:
        return None
    launcher = Path(mpi_dir) / "mpirun"
    if not launcher.is_file():
        return None
    os.environ["CONFSEARCH_ORCA_MPI_PATH"] = str(launcher)
    return str(launcher)


def main() -> int:
    args = parse_args()
    args._mpi_root_override = _ensure_mpi_root_override()
    if args._mpi_root_override:
        print("[wp2obs] running as root: set OMPI_ALLOW_RUN_AS_ROOT{,_CONFIRM}=1 "
              "for the ACP/ORCA mpirun children (recorded in ledger)")
    entry = run_driver(args)
    if entry["failed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
