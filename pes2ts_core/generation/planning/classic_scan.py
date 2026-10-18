"""Classic native per-point scan controller (G2-AB3 WP-5).

Runs the classic execution loop, one scan point at a time::

    compile controls (K, targets, grid)
      -> key-atom-only geometric seed from the last ACCEPTED frame
      -> one ACP ``XtbConstrainedOpt`` native constrained optimization
      -> normalize evidence (real convergence, actual q(X), energy channels)
      -> chemistry acceptance (residual caps, collision, spectator topology)
      -> accepted frame becomes the ONLY parent for the next point
      -> limited midpoint subdivision retries (the original target stays queued)
      -> physical-energy peak candidates at the end

The single-step primitives (seed -> request -> call -> outcome -> acceptance
-> recovery) live in :mod:`pes2ts_core.generation.planning.control_step` and
are shared verbatim with the learned control runs (ADR-0017 §5); the classic
loop drives them with the zero residual, so its behavior and request digests
are unchanged by the extraction.

Discipline implemented here (Classic design §5):

* a rejected/unconverged structure NEVER becomes the parent frame;
* subdivision inserts a midpoint between the same accepted parent and the
  still-queued target; after a midpoint is accepted the ORIGINAL target is
  retried — the planned grid is always visited in order or the scan stops
  with a typed prefix terminal;
* ``completed_interval`` requires every planned grid lambda to have an
  accepted frame whose ACTUAL coordinates met the residual caps — a target
  lambda is never reported as the achieved scan position;
* per-case budgets (wall seconds, host requests, subdivision layers,
  rejections per target) are frozen; exceeding them stops the scan with a
  typed status, never silently raises them;
* the on-disk trace records every request/outcome/decision; per-point
  attempt dirs + the transport's idempotent receipt recovery make
  interrupted runs resume without recomputation.
"""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from pes2ts_core.generation.planning.chemical_control import ChemicalControl
from pes2ts_core.generation.planning.control_step import (
    NativeScanPolicy,
    ScanFrame,
    SEED_AT_TARGET,
    SEED_OK,
    TransportCallable,
    bonded_pairs,
    classify_step_refusal,
    chemistry_check,
    prepare_step_attempt,
    scaled_step_targets,
)

RESULT_SCHEMA = "pes2ts_native_scan_result_v1"
TRACE_SCHEMA = "pes2ts_native_scan_trace_v1"

STATUS_COMPLETE = "complete_interval"
STATUS_PARTIAL_PREFIX = "partial_prefix"
STATUS_REJECTED_INPUT = "rejected_input"
STATUS_BUDGET_EXHAUSTED = "budget_exhausted"


class NativeScanError(ValueError):
    """Raised on misuse or unrecoverable contract breaks (not typed refusals)."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8"
    )


def run_native_scan(
    *,
    control: ChemicalControl,
    origin_geometry,
    transport: TransportCallable,
    save_dir: str | Path,
    policy: NativeScanPolicy | None = None,
    reaction_id: str | None = None,
) -> dict[str, Any]:
    """Run (or resume) one native scan case.

    Args:
        control: Compiled chemical control (drivers, K, grid).
        origin_geometry: Prepared origin frame (frame 0), frozen atom order.
        transport: Callable ``(request, attempt_dir, execution_id, attempt_id)
            -> attempt result`` (e.g. a closure over ``run_xtb_constopt_attempt``).
        save_dir: Case output root; attempts under ``save_dir/attempts/``.
        policy: Frozen execution recipe.

    Returns:
        ``pes2ts_native_scan_result_v1`` document (also carries the event
        trace under ``_events`` for :func:`persist_native_outputs`).
    """
    active_policy = policy or NativeScanPolicy()
    reaction_id = reaction_id or control.reaction_id
    n_atoms = len(control.elements)
    if len(origin_geometry) != n_atoms:
        raise NativeScanError("origin geometry does not match the control atom order")

    save_root = Path(save_dir)
    attempts_root = save_root / "attempts"
    (save_root / "native").mkdir(parents=True, exist_ok=True)

    grid = list(control.lambda_grid)
    frames: list[ScanFrame] = []
    events: list[dict[str, Any]] = []
    request_count = 0
    wall_start = time.monotonic()

    def record(event: dict[str, Any]) -> None:
        event["ts"] = _now()
        events.append(event)

    origin_array = np.asarray(origin_geometry, dtype=float)
    if not np.isfinite(origin_array).all():
        return _terminal_result(
            control, reaction_id, active_policy, frames, events, request_count, 0.0,
            STATUS_REJECTED_INPUT, "INPUT_GEOMETRY_INVALID",
            "origin geometry contains non-finite coordinates",
        )

    driver_pairs = {tuple(sorted(pair)) for pair in
                    (list(driver.atoms) for driver in control.drivers) if len(pair) == 2}

    origin_achieved = {
        driver.driver_id: driver.target_at(0.0) for driver in control.drivers
    }
    frames.append(
        ScanFrame(
            frame_index=0, lambda_value=0.0, is_grid_point=True,
            requested_targets=origin_achieved,
            actual_values=dict(origin_achieved),
            residuals={key: 0.0 for key in origin_achieved},
            geometry=[[float(v) for v in row] for row in origin_array],
            energy_effective_hartree=None, energy_physical_hartree=None,
            parent_frame_index=None, attempt_dir="", request_sha256="",
            n_optimization_cycles=None, max_seed_displacement=None,
        )
    )
    record({"kind": "origin_frame", "lambda": 0.0, "frame_index": 0})

    parent_geometry = origin_array.copy()
    parent_bonds = bonded_pairs(control.elements, origin_array)
    parent_index = 0
    last_accepted_lambda = 0.0

    pending: deque[tuple[float, bool, int]] = deque(
        (lam, True, 0) for lam in grid[1:]
    )
    status = STATUS_COMPLETE
    failure_code: str | None = None
    failure_detail: str | None = None
    min_scale = 0.5 ** active_policy.max_subdivisions

    while pending:
        if time.monotonic() - wall_start > active_policy.max_wall_seconds:
            status, failure_code = STATUS_BUDGET_EXHAUSTED, "WALL_BUDGET_EXHAUSTED"
            failure_detail = (
                f"{time.monotonic() - wall_start:.1f}s exceeded "
                f"{active_policy.max_wall_seconds:g}s after {len(frames) - 1} accepted frames"
            )
            break
        if request_count >= active_policy.max_host_requests:
            status, failure_code = STATUS_BUDGET_EXHAUSTED, "REQUEST_BUDGET_EXHAUSTED"
            failure_detail = (
                f"{request_count} host requests reached the cap "
                f"{active_policy.max_host_requests}"
            )
            break

        target_lambda, is_grid_point, rejections = pending[0]
        span = target_lambda - last_accepted_lambda
        scale = 1.0
        accepted = False
        stop_scan = False

        while not accepted:
            scaled_lambda, scaled_targets = scaled_step_targets(
                control, last_accepted_lambda, target_lambda, scale
            )
            prep = prepare_step_attempt(
                control=control,
                policy=active_policy,
                parent_geometry=parent_geometry,
                scaled_lambda=scaled_lambda,
                command_targets=scaled_targets,
                transport=transport,
                attempts_root=attempts_root,
                reaction_id=reaction_id,
                rejections=rejections,
                parent_request_sha256=frames[-1].request_sha256 or None,
            )
            if prep.seed.status not in (SEED_OK, SEED_AT_TARGET):
                if scale > min_scale:
                    scale *= 0.5
                    record({"kind": "subdivision_after_seed_refusal",
                            "lambda": scaled_lambda, "seed_status": prep.seed.status,
                            "new_scale": scale})
                    continue
                status = STATUS_PARTIAL_PREFIX
                failure_code = f"STEP_LIMIT:SEED_{prep.seed.status.upper()}"
                failure_detail = prep.seed.reason
                record({"kind": "seed_refusal_terminal", "lambda": scaled_lambda,
                        "status": prep.seed.status, "reason": prep.seed.reason})
                stop_scan = True
                break

            request_count += 1
            assert prep.outcome is not None and prep.request is not None
            outcome = prep.outcome
            record({
                "kind": "point_attempt", "lambda": scaled_lambda,
                "is_grid_point": is_grid_point, "attempt_id": prep.attempt_id,
                "attempt_dir": str(prep.attempt_dir),
                "request_sha256": prep.request["provenance"]["request_sha256"],
                "outcome": outcome.status, "converged": outcome.converged_geometry,
                "wall_seconds": outcome.wall_seconds,
                "n_optimization_cycles_logged": outcome.n_optimization_cycles_logged,
                "seed_status": prep.seed.status,
                "max_seed_displacement": prep.seed.max_atom_displacement,
                "failure_reason": outcome.failure_reason,
            })

            refusal_reason = classify_step_refusal(
                outcome, control, active_policy,
                driver_pairs, parent_bonds,
            )
            if refusal_reason is None:
                final_geometry = np.asarray(outcome.geometry, dtype=float)
                frame = ScanFrame(
                    frame_index=len(frames),
                    lambda_value=float(scaled_lambda),
                    is_grid_point=bool(
                        is_grid_point and abs(scaled_lambda - target_lambda) <= 1e-9
                    ),
                    requested_targets={
                        driver.driver_id: float(value)
                        for driver, value in zip(control.drivers, scaled_targets)
                    },
                    actual_values={
                        m.driver_id: float(m.actual) for m in outcome.measurements
                    },
                    residuals={m.driver_id: float(m.residual) for m in outcome.measurements},
                    geometry=[[float(v) for v in row] for row in final_geometry],
                    energy_effective_hartree=outcome.energy_effective_hartree,
                    energy_physical_hartree=outcome.energy_physical_hartree,
                    parent_frame_index=parent_index,
                    attempt_dir=str(prep.attempt_dir),
                    request_sha256=str(prep.request["provenance"]["request_sha256"]),
                    n_optimization_cycles=outcome.n_optimization_cycles_logged,
                    max_seed_displacement=prep.seed.max_atom_displacement,
                )
                frames.append(frame)
                parent_geometry = final_geometry
                parent_bonds = bonded_pairs(control.elements, final_geometry)
                parent_index = frame.frame_index
                last_accepted_lambda = float(scaled_lambda)
                accepted = True
                record({
                    "kind": "accepted", "lambda": float(scaled_lambda),
                    "frame_index": frame.frame_index,
                    "is_grid_point": frame.is_grid_point,
                    "energy_effective_hartree": frame.energy_effective_hartree,
                    "energy_physical_hartree": frame.energy_physical_hartree,
                })
                continue

            record({"kind": "refusal", "lambda": scaled_lambda,
                    "attempt_id": prep.attempt_id, "reason": refusal_reason})
            rejections += 1
            if rejections > active_policy.max_rejections_per_target:
                status = STATUS_PARTIAL_PREFIX
                failure_code = "STEP_LIMIT:REJECTIONS_EXHAUSTED"
                failure_detail = refusal_reason
                stop_scan = True
                break
            if scale > min_scale:
                scale *= 0.5
                record({"kind": "subdivision_after_refusal", "lambda": scaled_lambda,
                        "new_scale": scale})
                continue
            status = STATUS_PARTIAL_PREFIX
            failure_code = "STEP_LIMIT:SUBDIVISIONS_EXHAUSTED"
            failure_detail = refusal_reason
            stop_scan = True
            break

        if stop_scan:
            break

        # A target leaves the queue only when the geometry reached it exactly
        # (scale == 1).  A subdivided midpoint acceptance keeps the original
        # target queued with the accumulated rejection count preserved.
        if accepted:
            if scale >= 1.0 - 1e-12:
                pending.popleft()
            else:
                pending[0] = (target_lambda, is_grid_point, rejections)

    covered = {round(frame.lambda_value, 9) for frame in frames}
    grid_covered = all(round(lam, 9) in covered for lam in grid)
    completed_interval = status == STATUS_COMPLETE and grid_covered and not pending
    if completed_interval:
        failure_code = None
        failure_detail = None
    elif status == STATUS_COMPLETE:
        status = STATUS_PARTIAL_PREFIX
        if failure_code is None:
            failure_code = "STEP_LIMIT:GRID_INCOMPLETE"

    wall_seconds = time.monotonic() - wall_start
    return _terminal_result(
        control, reaction_id, active_policy, frames, events, request_count,
        wall_seconds, status, failure_code, failure_detail,
        completed_interval=completed_interval,
    )


def _terminal_result(
    control: ChemicalControl,
    reaction_id: str,
    policy: NativeScanPolicy,
    frames: list[ScanFrame],
    events: list[dict[str, Any]],
    request_count: int,
    wall_seconds: float,
    status: str,
    failure_code: str | None,
    failure_detail: str | None,
    *,
    completed_interval: bool = False,
) -> dict[str, Any]:
    accepted = frames[1:]
    last_lambda = frames[-1].lambda_value if frames else None
    total_cycles = sum(frame.n_optimization_cycles or 0 for frame in accepted)
    energies_physical = [
        frame.energy_physical_hartree for frame in accepted
        if frame.energy_physical_hartree is not None
    ]
    return {
        "schema_version": RESULT_SCHEMA,
        "engine_mode": "classic_native_xtb_scan",
        "reaction_id": reaction_id,
        "status": status,
        "failure_code": failure_code,
        "failure_detail": failure_detail,
        "completed_interval": bool(completed_interval),
        "last_accepted_lambda": last_lambda,
        "n_accepted_frames_beyond_origin": len(accepted),
        "n_grid_points": len(control.lambda_grid),
        "frames": [frame.to_payload() for frame in frames],
        "energy_channel": {
            "physical_source": "unrestrained_xTB_single_point_same_geometry",
            "effective_source": "restraint_including_TOTAL_ENERGY",
            "physical_available": bool(energies_physical),
            "physical_min_hartree": min(energies_physical) if energies_physical else None,
            "physical_max_hartree": max(energies_physical) if energies_physical else None,
        },
        "active_atom_indices": list(control.active_atom_indices),
        "active_atom_maps": list(control.active_atom_maps),
        "execution_direction": control.execution_direction,
        "origin_endpoint": control.origin_endpoint,
        "policy": policy.to_payload(),
        "plan_sha256": control.plan_sha256,
        "atom_order_sha256": control.atom_order_sha256,
        "cost": {
            "host_requests": request_count,
            "n_accepted_frames": len(accepted),
            "n_optimization_cycles_logged": total_cycles,
            "wall_seconds": wall_seconds,
        },
        "generated_at": _now(),
        "_events": events,
    }


def persist_native_outputs(save_dir: str | Path, result: dict[str, Any]) -> None:
    """Write the result + trace documents under ``<save_dir>/native/``."""
    root = Path(save_dir) / "native"
    root.mkdir(parents=True, exist_ok=True)
    events = result.pop("_events", [])
    trace = {
        "schema_version": TRACE_SCHEMA,
        "engine_mode": result.get("engine_mode"),
        "reaction_id": result.get("reaction_id"),
        "events": events,
        "generated_at": _now(),
    }
    _write_json(root / "native_scan_result.json", result)
    _write_json(root / "native_scan_trace.json", trace)


__all__ = [
    "NativeScanError",
    "NativeScanPolicy",
    "RESULT_SCHEMA",
    "STATUS_BUDGET_EXHAUSTED",
    "STATUS_COMPLETE",
    "STATUS_PARTIAL_PREFIX",
    "STATUS_REJECTED_INPUT",
    "TRACE_SCHEMA",
    "bonded_pairs",
    "chemistry_check",
    "persist_native_outputs",
    "run_native_scan",
]
