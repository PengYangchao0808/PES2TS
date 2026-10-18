"""Native-vs-legacy engine comparison report (G2-AB3 WP-9).

Builds ``pes2ts_native_comparison_report_v1`` from a completed native
campaign root and the frozen legacy campaign root, iterating the MANIFEST's
exact case list (never a directory glob — historical crash directories like
``RXN_*.crashed_v1`` must not pollute current statistics, the AB2 audit's
defect P2-2).

Cost comparison discipline: the legacy engine's cost unit is ACP gradient
requests (``N_gradient``) plus wall seconds; the native engine's units are
host optimization requests, per-cycle logged xTB energy/gradient
evaluations (an estimate column), and wall seconds.  These are reported
side-by-side and never summed into one number.  Speedup ratios use wall
seconds per case, with the legacy origin stage cost noted as excluded where
the native run reused the legacy prepared origin.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from pes2ts_core.generation.campaign import load_campaign_manifest

REPORT_SCHEMA = "pes2ts_native_comparison_report_v1"


class NativeReportError(ValueError):
    """Raised when the campaign roots cannot be compared."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeReportError(f"unreadable JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise NativeReportError(f"JSON document is not an object: {path}")
    return payload


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def build_native_comparison_report(
    *,
    native_root: str | Path,
    legacy_root: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Compare a native campaign against its frozen legacy baseline."""
    manifest = load_campaign_manifest(Path(manifest_path))
    campaign_sha = manifest["content_sha256"]
    native = Path(native_root).expanduser().resolve()
    legacy = Path(legacy_root).expanduser().resolve()

    per_case: list[dict[str, Any]] = []
    native_counts: dict[str, int] = {}
    legacy_counts: dict[str, int] = {}
    wall_native: list[float] = []
    wall_legacy_scan: list[float] = []
    requests_native: list[int] = []
    cycles_native: list[int] = []
    gradients_legacy: list[int] = []
    residual_max: list[float] = []
    physical_sp_missing = 0

    for case in manifest["cases"]:
        reaction_id = case["reaction_id"]
        native_terminal_path = native / reaction_id / "native_terminal_state.json"
        legacy_terminal_path = legacy / reaction_id / "terminal_state.json"
        if not native_terminal_path.is_file():
            raise NativeReportError(
                f"native campaign is missing a terminal for {reaction_id}"
            )
        if not legacy_terminal_path.is_file():
            raise NativeReportError(
                f"legacy campaign is missing a terminal for {reaction_id}"
            )
        native_terminal = _read_json(native_terminal_path)
        legacy_terminal = _read_json(legacy_terminal_path)
        if native_terminal.get("schema_version") != "pes2ts_g2t_native_terminal_state_v1":
            raise NativeReportError(
                f"{reaction_id}: native terminal has an unexpected schema"
            )
        legacy_costs = _read_json(legacy / reaction_id / "costs.json")
        legacy_aggregate = legacy_costs.get("aggregate") or {}
        legacy_gradients = int(legacy_aggregate.get("N_gradient") or 0)
        legacy_wall = (legacy_terminal.get("costs") or {}).get("cpu_seconds")
        legacy_scan_wall = _legacy_scan_wall(legacy / reaction_id)

        native_class = native_terminal.get("terminal_class", "unknown")
        legacy_class = legacy_terminal.get("terminal_class", "unknown")
        native_counts[native_class] = native_counts.get(native_class, 0) + 1
        legacy_counts[legacy_class] = legacy_counts.get(legacy_class, 0) + 1

        costs = native_terminal.get("costs") or {}
        wall = float(costs.get("wall_seconds") or 0.0)
        requests = int(costs.get("host_requests") or 0)
        cycles = int(costs.get("n_xtb_optimization_cycles_logged") or 0)
        wall_native.append(wall)
        requests_native.append(requests)
        cycles_native.append(cycles)
        if legacy_gradients:
            gradients_legacy.append(legacy_gradients)
        if legacy_scan_wall is not None:
            wall_legacy_scan.append(legacy_scan_wall)

        result_path = native / reaction_id / "native" / "native_scan_result.json"
        residuals: list[float] = []
        physical_missing_here = False
        if result_path.is_file():
            result = _read_json(result_path)
            for frame in result.get("frames", []):
                residuals.extend(
                    abs(v) for v in (frame.get("residuals") or {}).values()
                    if isinstance(v, (int, float))
                )
                if frame.get("energy_physical_hartree") is None and frame.get(
                    "frame_index", 0
                ) > 0:
                    physical_missing_here = True
        if residuals:
            residual_max.append(max(residuals))
        if physical_missing_here:
            physical_sp_missing += 1

        speedup = (legacy_scan_wall / wall) if (wall > 0 and legacy_scan_wall) else None
        per_case.append({
            "reaction_id": reaction_id,
            "legacy_terminal_class": legacy_class,
            "native_terminal_class": native_class,
            "legacy_failure_code": legacy_terminal.get("failure_code"),
            "native_failure_code": native_terminal.get("failure_code"),
            "legacy_last_accepted_lambda": legacy_terminal.get("last_accepted_lambda"),
            "native_last_accepted_lambda": native_terminal.get("last_accepted_lambda"),
            "legacy_n_accepted_frames": legacy_terminal.get("n_accepted_frames_beyond_origin"),
            "native_n_accepted_frames": native_terminal.get("n_accepted_frames_beyond_origin"),
            "legacy_N_gradient": legacy_gradients,
            "native_host_requests": requests,
            "native_xtb_cycles_logged": cycles,
            "legacy_wall_seconds_total": legacy_wall,
            "legacy_scan_wall_seconds": legacy_scan_wall,
            "native_wall_seconds": wall,
            "scan_wall_speedup": speedup,
            "native_n_candidates": native_terminal.get("n_candidates", 0),
            "legacy_stage_reused": native_terminal.get("legacy_stage_reused", False),
        })

    scan_wall_speedups = [
        entry["scan_wall_speedup"] for entry in per_case
        if entry["scan_wall_speedup"] is not None
        and not entry["legacy_stage_reused"]
    ]
    completed_native = [
        entry for entry in per_case if entry["native_terminal_class"] == "complete_path"
    ]
    completed_legacy = [
        entry for entry in per_case if entry["legacy_terminal_class"] == "complete_path"
    ]
    return {
        "schema_version": REPORT_SCHEMA,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "campaign_sha256": campaign_sha,
        "manifest_path": str(Path(manifest_path)),
        "native_root": str(native),
        "legacy_root": str(legacy),
        "n_cases": len(per_case),
        "terminal_counts": {"native": native_counts, "legacy": legacy_counts},
        "cost_summary": {
            "native_wall_seconds_total": sum(wall_native),
            "native_wall_seconds_median": _median(wall_native),
            "native_host_requests_total": sum(requests_native),
            "native_xtb_cycles_logged_total": sum(cycles_native),
            "native_xtb_cycles_note": (
                "per-cycle energy+gradient evaluations counted from xTB logs; "
                "estimate column, never mixed with legacy ORCA gradient totals"
            ),
            "legacy_N_gradient_total": sum(gradients_legacy),
            "legacy_N_gradient_median_per_case": _median(
                [float(v) for v in gradients_legacy]
            ),
            "legacy_scan_wall_seconds_median": _median(wall_legacy_scan),
            "scan_wall_speedup_median": _median(scan_wall_speedups),
            "units_note": (
                "legacy cost unit = ACP ORCA gradient requests; native unit = "
                "host xTB optimization requests (+ logged internal cycles); "
                "reported side-by-side, never summed"
            ),
        },
        "coverage": {
            "native_complete_path": len(completed_native),
            "legacy_complete_path": len(completed_legacy),
            "native_max_coordinate_residual_angstrom": (
                max(residual_max) if residual_max else None
            ),
            "native_physical_sp_missing_cases": physical_sp_missing,
        },
        "per_case": per_case,
        "claims": {
            "success_rate_claimed": False,
            "ts_validated_claimed": False,
            "note": (
                "candidate extraction is selection evidence only; validation "
                "remains blocked_on_human_review for all cases; speedup is an "
                "engineering observation on 24 cases, not a statistical claim"
            ),
        },
    }


def _legacy_scan_wall(case_root: Path) -> float | None:
    """Legacy continuation wall seconds (scan stage only, origin excluded)."""
    path = case_root / "costs.json"
    if not path.is_file():
        return None
    costs = _read_json(path)
    total = 0.0
    seen = False
    for ledger in costs.get("ledgers", []):
        cost_id = str(ledger.get("cost_id", ""))
        if ":continuation:" in cost_id or ":probe:" in cost_id:
            wall = ledger.get("cpu_seconds")
            if isinstance(wall, (int, float)):
                total += float(wall)
                seen = True
    return total if seen else None


__all__ = [
    "NativeReportError",
    "REPORT_SCHEMA",
    "build_native_comparison_report",
]
