"""Native-scan campaign runner over the frozen G2.T manifest (G2-AB3 WP-8/9).

Executes the ``classic_native_xtb_scan`` engine for every case of a frozen
campaign manifest, reusing the legacy campaign's frozen per-case inputs
(plan + prepared origin geometry) so the engine comparison isolates the scan
stage:

* cases the legacy campaign rejected at the input/origin stages keep their
  typed rejection (the native engine does not silently re-run or heal input
  problems); the terminal records ``legacy_stage_reused=true``;
* every other case compiles its frozen plan into chemical controls, runs the
  per-point native scan through the ACP ``XtbConstrainedOpt`` transport, and
  writes versioned artifacts under an independent output root
  (``pes2ts_native_scan_result_v1`` / ``..._trace_v1`` /
  ``pes2ts_native_scan_seed_proposals_v1`` / native terminal + costs);
* idempotent resume: a case with a non-blocked native terminal is skipped
  unless ``resume=False``; per-point attempt receipts make interrupted cases
  resume without recomputation.

The legacy campaign root and its artifacts are read-only here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import time
from typing import Any, Sequence

from pes2ts_core.generation.campaign import load_campaign_manifest
from pes2ts_core.generation.planning.chemical_control import (
    ChemicalControlError,
    compile_chemical_control,
    load_plan,
)
from pes2ts_core.generation.planning.classic_scan import (
    NativeScanPolicy,
    persist_native_outputs,
    run_native_scan,
)
from pes2ts_core.generation.planning.native_candidates import (
    select_native_energy_candidates,
)
from pes2ts_core.integration.acp.xtb_constraints_transport import (
    run_xtb_constopt_attempt,
)

NATIVE_TERMINAL_SCHEMA = "pes2ts_g2t_native_terminal_state_v1"
NATIVE_CAMPAIGN_SCHEMA = "pes2ts_native_campaign_summary_v1"
NATIVE_COSTS_SCHEMA = "pes2ts_native_case_costs_v1"

_REJECTED_STAGE_CODES = {"INPUT_GEOMETRY_INVALID", "ORIGIN_BUDGET_EXHAUSTED"}


class NativeCampaignError(ValueError):
    """Raised on campaign-level contract violations."""


@dataclass
class NativeCampaignSummary:
    n_cases: int
    n_executed: int
    n_skipped_idempotent: int
    n_rejected_reused: int
    terminal_counts: dict[str, int]
    session_wall_seconds: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": NATIVE_CAMPAIGN_SCHEMA,
            "engine_mode": "classic_native_xtb_scan",
            "n_cases": self.n_cases,
            "n_executed": self.n_executed,
            "n_skipped_idempotent": self.n_skipped_idempotent,
            "n_rejected_reused": self.n_rejected_reused,
            "terminal_counts": self.terminal_counts,
            "session_wall_seconds": self.session_wall_seconds,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeCampaignError(f"unreadable JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise NativeCampaignError(f"JSON document is not an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def _terminal_class(scan_status: str) -> str:
    if scan_status == "complete_interval":
        return "complete_path"
    if scan_status == "partial_prefix":
        return "partial_prefix"
    return "rejected_typed"


def _transport_factory(acp_wiring: dict[str, Any]):
    acp_root = acp_wiring["root"]
    python_executable = acp_wiring.get("python")
    config_path = acp_wiring.get("config_path")

    def transport(request: dict[str, Any], attempt_dir: Path,
                  execution_id: str, attempt_id: str):
        return run_xtb_constopt_attempt(
            acp_root=acp_root,
            python_executable=python_executable,
            acp_config_path=config_path,
            request=request,
            output_root=attempt_dir,
            execution_id=execution_id,
            attempt_id=attempt_id,
            timeout_seconds=7200.0,
        )

    return transport


def run_native_campaign(
    *,
    manifest_path: str | Path,
    legacy_root: str | Path,
    output_root: str | Path,
    case_filter: Sequence[str] | None = None,
    resume: bool = True,
    policy: NativeScanPolicy | None = None,
) -> dict[str, Any]:
    """Run (or resume) the native campaign for a frozen manifest.

    Returns:
        The campaign summary payload (also written to
        ``<output_root>/native_campaign_summary.json``).
    """
    manifest = load_campaign_manifest(Path(manifest_path))
    case_entries = {case["reaction_id"]: case for case in manifest["cases"]}
    cases = list(case_entries)
    if case_filter:
        wanted = set(case_filter)
        unknown = wanted - set(cases)
        if unknown:
            raise NativeCampaignError(f"case filter contains unknown ids: {sorted(unknown)}")
        cases = [case for case in cases if case in wanted]

    legacy = Path(legacy_root).expanduser().resolve()
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "native_campaign_identity.json", {
        "engine_mode": "classic_native_xtb_scan",
        "campaign_sha256": manifest["content_sha256"],
        "manifest_path": str(Path(manifest_path)),
        "legacy_root": str(legacy),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    })

    active_policy = policy or NativeScanPolicy()
    transport = _transport_factory(manifest["acp_wiring"])
    counts: dict[str, int] = {}
    n_executed = n_skipped = n_reused = 0
    wall_start = time.monotonic()

    for reaction_id in cases:
        case_root = root / reaction_id
        terminal_path = case_root / "native_terminal_state.json"
        if resume and terminal_path.is_file():
            existing = _read_json(terminal_path)
            if existing.get("terminal_class") is not None:
                counts[existing["terminal_class"]] = (
                    counts.get(existing["terminal_class"], 0) + 1
                )
                n_skipped += 1
                continue

        legacy_case = legacy / reaction_id
        legacy_terminal_path = legacy_case / "terminal_state.json"
        if not legacy_terminal_path.is_file():
            raise NativeCampaignError(
                f"legacy campaign is missing a terminal for {reaction_id}: {legacy_terminal_path}"
            )
        legacy_terminal = _read_json(legacy_terminal_path)
        legacy_plan_path = legacy_case / "plan.json"

        if legacy_terminal.get("terminal_class") == "rejected_typed" and (
            not legacy_plan_path.is_file()
        ):
            failure_code = legacy_terminal.get("failure_code", "LEGACY_REJECTED")
            if failure_code not in _REJECTED_STAGE_CODES:
                raise NativeCampaignError(
                    f"{reaction_id}: unexpected legacy pre-plan failure code {failure_code!r}"
                )
            terminal = {
                "schema_version": NATIVE_TERMINAL_SCHEMA,
                "engine_mode": "classic_native_xtb_scan",
                "reaction_id": reaction_id,
                "terminal_class": "rejected_typed",
                "failure_code": failure_code,
                "failure_detail": (
                    "typed input/origin rejection reused from the frozen legacy "
                    "campaign; the native scan engine was not run for this case"
                ),
                "legacy_stage_reused": True,
                "legacy_terminal_ref": str(legacy_terminal_path),
                "completed_interval": False,
                "last_accepted_lambda": None,
                "n_accepted_frames_beyond_origin": 0,
                "costs": _native_costs_payload(0, 0, 0, 0.0),
                "validation": {
                    "status": "blocked_on_human_review",
                    "needs": [
                        "accepted ReviewRecord for this case "
                        "(two-reviewer workbook + adjudication + import)"
                    ],
                },
                "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            case_root.mkdir(parents=True, exist_ok=True)
            _write_json(terminal_path, terminal)
            _write_json(case_root / "native_costs.json", terminal["costs"])
            counts["rejected_typed"] = counts.get("rejected_typed", 0) + 1
            n_reused += 1
            continue

        plan = load_plan(legacy_plan_path)
        snapshot_ref = Path(case_entries[reaction_id]["snapshot_path"])
        snapshot_path = (
            snapshot_ref if snapshot_ref.is_absolute()
            else Path.cwd() / snapshot_ref
        )
        if not snapshot_path.is_file():
            manifest_anchored = (
                Path(manifest_path).resolve().parent.parent / snapshot_ref
            )
            if manifest_anchored.is_file():
                snapshot_path = manifest_anchored
        endpoint_r = endpoint_p = None
        if snapshot_path.is_file():
            snapshot = _read_json(snapshot_path)
            endpoint_r = snapshot.get("r_coordinates")
            endpoint_p = snapshot.get("p_coordinates")
        control = compile_chemical_control(
            plan, reaction_id=reaction_id,
            n_grid_points=active_policy.n_grid_points,
            endpoint_geometry_r=endpoint_r,
            endpoint_geometry_p=endpoint_p,
        )
        origin_geometry = plan.get("start_geometry")
        if not isinstance(origin_geometry, list) or len(origin_geometry) != len(
            control.elements
        ):
            raise NativeCampaignError(
                f"{reaction_id}: legacy plan lacks a usable start_geometry"
            )

        result = run_native_scan(
            control=control,
            origin_geometry=origin_geometry,
            transport=transport,
            save_dir=case_root,
            policy=active_policy,
            reaction_id=reaction_id,
        )
        persist_native_outputs(case_root, result)
        candidates = select_native_energy_candidates(result)
        _write_json(case_root / "native_candidates.json", candidates)

        terminal_class = _terminal_class(result["status"])
        terminal = {
            "schema_version": NATIVE_TERMINAL_SCHEMA,
            "engine_mode": "classic_native_xtb_scan",
            "reaction_id": reaction_id,
            "terminal_class": terminal_class,
            "failure_code": result.get("failure_code"),
            "failure_detail": result.get("failure_detail"),
            "legacy_stage_reused": False,
            "completed_interval": result.get("completed_interval"),
            "last_accepted_lambda": result.get("last_accepted_lambda"),
            "n_accepted_frames_beyond_origin": result.get("n_accepted_frames_beyond_origin"),
            "n_candidates": candidates.get("n_candidates", 0),
            "candidates_ref": "native_candidates.json",
            "result_ref": "native/native_scan_result.json",
            "trace_ref": "native/native_scan_trace.json",
            "costs": _native_costs_payload(
                result["cost"]["host_requests"],
                result["cost"]["n_accepted_frames"],
                result["cost"]["n_optimization_cycles_logged"],
                result["cost"]["wall_seconds"],
            ),
            "execution_direction": result.get("execution_direction"),
            "origin_endpoint": result.get("origin_endpoint"),
            "validation": {
                "status": "blocked_on_human_review",
                "needs": [
                    "accepted ReviewRecord for this case "
                    "(two-reviewer workbook + adjudication + import)"
                ],
            },
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        _write_json(terminal_path, terminal)
        _write_json(case_root / "native_costs.json", terminal["costs"])
        counts[terminal_class] = counts.get(terminal_class, 0) + 1
        n_executed += 1

    summary = NativeCampaignSummary(
        n_cases=len(cases),
        n_executed=n_executed,
        n_skipped_idempotent=n_skipped,
        n_rejected_reused=n_reused,
        terminal_counts=counts,
        session_wall_seconds=time.monotonic() - wall_start,
    ).to_payload()
    _write_json(root / "native_campaign_summary.json", summary)
    return summary


def _native_costs_payload(host_requests: int, n_accepted: int,
                          cycles_estimate: int, wall_seconds: float) -> dict[str, Any]:
    return {
        "schema_version": NATIVE_COSTS_SCHEMA,
        "engine_mode": "classic_native_xtb_scan",
        "host_requests": int(host_requests),
        "n_accepted_frames": int(n_accepted),
        "n_xtb_optimization_cycles_logged": int(cycles_estimate),
        "n_gradient_note": (
            "per-cycle energy+gradient evaluations counted from the xTB log; "
            "engine-level counters are an estimate column, never mixed into "
            "measured ORCA gradient totals"
        ),
        "wall_seconds": float(wall_seconds),
    }


__all__ = [
    "NATIVE_CAMPAIGN_SCHEMA",
    "NATIVE_COSTS_SCHEMA",
    "NATIVE_TERMINAL_SCHEMA",
    "NativeCampaignError",
    "run_native_campaign",
]
