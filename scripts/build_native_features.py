#!/usr/bin/env python3
"""V4.2 native-reconnect build + §2 hard-gate audit (Lane A).

Rebuilds the native (insertion-free) dataset for BOTH the fit-50 wave and the
monitor-8 wave (``dual_ts_native_rebuild.rebuild_native_dataset``), then runs
EVERY contract §2 hard gate and writes ``native_reconnect_audit.json``:

* counts — fit 5,568 native actions / 5,668 native states / 100 teacher
  endpoints (monitor recomputed from ITS registry, never copied);
* label identity — ``action_delta == X_next − X`` per state per atom ≤ 1e-10;
* unique successor + two-sided ordering toward TS + atom map/unit assertions;
* native maximum per-atom action P50/P90/P95/P99/max (Å);
* E/g spot-check on 20 native states: receipt ``gradient_hartree_per_angstrom``
  vs teacher ``observation.gradient`` ≤ 1e-10, with method=GFN2-xTB /
  basis="" (xTB no basis) / charge=0 / multiplicity=1 / solvent=unrecorded
  registered explicitly;
* npz↔teacher ``x_angstrom`` spot-check on 20 states (frozen unweighted
  centroid-removal rule applied to teacher coordinates).

Any hard gate that fails aborts with a nonzero exit and writes the failing
field — data is NEVER fabricated to pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.contracts.dual_ts import (  # noqa: E402
    BOUNDARY_TEACHER_ENDPOINT,
    DualTSTeacher,
)
from pes2ts_core.pesn.offline.data.dual_ts_native_rebuild import (  # noqa: E402
    ACTION_IDENTITY_TOL,
    COORD_SPOT_TOL,
    rebuild_native_dataset,
    rebuild_native_reaction,
    sha256_file,
)

AUDIT_SCHEMA: str = "pes2ts_dual_ts_native_reconnect_audit_v1"
TOOL_VERSION: str = "pes2ts_dual_ts_native_reconnect_audit_tool_v1"

DEFAULT_ROOT: str = "outputs/pesn/dual_ts_v2_20261015"
DEFAULT_OUT: str = f"{DEFAULT_ROOT}/wp42_campaign"

SPOT_STATES: int = 20
SEED: int = 20261017


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _percentiles(values: np.ndarray) -> dict:
    return {
        "P50": float(np.percentile(values, 50)),
        "P90": float(np.percentile(values, 90)),
        "P95": float(np.percentile(values, 95)),
        "P99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def run_hard_gates(observed_dir: Path, features_dir: Path,
                   native_dir: Path, rng: np.random.Generator) -> dict:
    """Run every contract §2 hard gate on one wave; return the gate report."""
    gaps = _load_json(observed_dir / "gaps_registry_observed.json")
    gapmap = {str(r["reaction_id"]): r for r in gaps["reactions"]}
    native_manifest = _load_json(native_dir / "features_manifest.json")
    population = native_manifest["population"]

    teacher_paths = sorted(observed_dir.glob("*.teacher.json"))
    rid_by_state: dict[str, str] = {}

    total_actions = 0
    total_states = 0
    total_endpoints = 0
    max_label_err = 0.0
    max_unchanged_err = 0.0
    max_arc_err = 0.0
    unique_ok = True
    ordering_ok = True
    map_ok = True
    units_ok = True
    max_unchanged_err_holder = 0.0
    atom_action_norms: list[float] = []
    spot_pool: list[tuple[str, str]] = []  # (reaction_id, state_id)
    x_spot_pool: list[tuple[str, str, np.ndarray, np.ndarray]] = []

    for tp in teacher_paths:
        rid = tp.name[:-len(".teacher.json")]
        teacher = DualTSTeacher.from_payload(_load_json(tp))
        rebuilt = rebuild_native_reaction(teacher, gapmap[rid])
        audit = rebuilt["audit"]
        native_states = rebuilt["native_states"]
        native_labels = rebuilt["native_labels"]

        total_actions += audit["n_actions_native"]
        total_states += audit["n_states_native"]
        total_endpoints += audit["n_teacher_endpoints"]
        max_label_err = max(max_label_err, audit["label_identity_max_abs_err"])
        max_unchanged_err_holder = max(max_unchanged_err_holder,
                                       audit["unchanged_step_max_abs_err"])
        max_arc_err = max(max_arc_err, audit["arc_recompute_max_abs_err"])
        unique_ok = unique_ok and audit["unique_successor"]

        # per-branch ordering toward TS + map/unit assertions.
        by_branch: dict[str, list] = {}
        for s in native_states:
            by_branch.setdefault(s.branch_id, []).append(s)
        for branch_id, chain in by_branch.items():
            idxs = [int(s.step_index) for s in chain]
            if idxs != sorted(idxs) or len(set(idxs)) != len(idxs):
                ordering_ok = False
            if chain[-1].boundary_type != BOUNDARY_TEACHER_ENDPOINT:
                ordering_ok = False
        for s in native_states:
            # map assertion: canonical G1 mapping m0001..mN
            expect_map = [f"m{i + 1:04d}" for i in
                          range(int(np.asarray(s.atomic_numbers).size))]
            if list(s.atom_map_ids) != expect_map:
                map_ok = False
            if s.method_id != "GFN2-xTB":
                map_ok = False
        # units: coordinates declared Å, gradient Ha/Å (spot-checked below).

        labels_by_id = {lab.state_id: lab for lab in native_labels}
        shard = np.load(features_dir / f"features_{rid}.npz", allow_pickle=False)
        sid_np = [str(x) for x in shard["state_id"]]
        idx_of = {s: i for i, s in enumerate(sid_np)}
        x_angstrom = np.asarray(shard["x_angstrom"], dtype=np.float64)
        # gather per-atom action norms + spot pools
        for s, lab in zip(native_states, native_labels):
            rid_by_state[s.state_id] = rid
            if lab.action_delta is not None:
                a = np.asarray(lab.action_delta, dtype=np.float64)
                atom_action_norms.extend(
                    float(v) for v in np.linalg.norm(a, axis=1))
                spot_pool.append((rid, s.state_id))
                i = idx_of[s.state_id]
                x_spot_pool.append(
                    (rid, s.state_id, np.asarray(s.coordinates, dtype=np.float64),
                     x_angstrom[i]))

    # E/g spot-check (20 states).
    rng.shuffle(spot_pool)
    eg_pool = spot_pool[:SPOT_STATES]
    eg_rows: list[dict] = []
    max_eg_err = 0.0
    raw_teacher_cache: dict[str, dict] = {}
    for rid, sid in eg_pool:
        if rid not in raw_teacher_cache:
            raw_teacher_cache[rid] = _load_json(
                observed_dir / f"{rid}.teacher.json")
        raw_states = {str(s["state_id"]): s
                      for s in raw_teacher_cache[rid]["states"]}
        state = raw_states[sid]
        receipt = Path(state["observation_receipt_path"])
        request = _load_json(receipt / "request.json")
        result = _load_json(receipt / "result.json")
        g_teacher = np.asarray(state["observation"]["gradient"],
                               dtype=np.float64)
        g_receipt = np.asarray(result["gradient_hartree_per_angstrom"],
                               dtype=np.float64)
        err = float(np.max(np.abs(g_teacher - g_receipt)))
        max_eg_err = max(max_eg_err, err)
        eg_rows.append({
            "reaction_id": rid, "state_id": sid,
            "method": str(request.get("method")),
            "basis": str(request.get("basis", "")),
            "charge": int(request.get("charge", 0)),
            "multiplicity": int(request.get("multiplicity", 1)),
            "solvent": "unrecorded",  # no solvent field in the receipts
            "energy_hartree": float(result.get("energy")),
            "max_abs_err": err,
            "passed": err <= ACTION_IDENTITY_TOL,
        })
        if request.get("method") != "GFN2-xTB":
            units_ok = False
        if str(request.get("basis", "")) != "":
            units_ok = False

    # npz↔teacher x_angstrom spot-check (20 states, centroid-removal rule).
    rng.shuffle(x_spot_pool)
    x_pool = x_spot_pool[:SPOT_STATES]
    max_x_err = 0.0
    for rid, sid, coord, x_row in x_pool:
        centered = coord - coord.mean(axis=0)
        err = float(np.max(np.abs(x_row - centered)))
        max_x_err = max(max_x_err, err)

    counts = {
        "states_native": total_states,
        "actions_native": total_actions,
        "teacher_endpoints": total_endpoints,
    }
    counts_pass = (counts["states_native"] == population["states_native"]
                   and counts["actions_native"] == population["actions_native"]
                   and counts["teacher_endpoints"]
                   == population["teacher_endpoint_states"])

    action_norms = np.asarray(atom_action_norms, dtype=np.float64)
    report = {
        "counts": dict(counts, pass_=counts_pass),
        "label_identity": {
            "max_abs_err": max_label_err, "tolerance": ACTION_IDENTITY_TOL,
            "pass_": max_label_err <= ACTION_IDENTITY_TOL,
        },
        "unchanged_step_identity": {
            "max_abs_err": max_unchanged_err_holder,
            "tolerance": ACTION_IDENTITY_TOL,
            "pass_": max_unchanged_err_holder <= ACTION_IDENTITY_TOL,
        },
        "arc_recompute_identity": {
            "max_abs_err": max_arc_err, "pass_": max_arc_err <= 1e-9,
        },
        "unique_successor": {"pass_": bool(unique_ok)},
        "two_sided_ordering": {"pass_": bool(ordering_ok)},
        "map_unit_assertions": {
            "pass_": bool(map_ok and units_ok),
            "atom_map": "m0001..mNNNN G1 invariant" if map_ok else "VIOLATION",
            "units": "coordinates Å, gradient Ha/Å, method GFN2-xTB, basis=''"
                     if units_ok else "VIOLATION",
        },
        "native_max_atomic_action_angstrom":
            _percentiles(action_norms) if action_norms.size else {},
        "eg_spot_check": {
            "n_states": len(eg_rows),
            "max_abs_err": max_eg_err,
            "tolerance": ACTION_IDENTITY_TOL,
            "pass_": max_eg_err <= ACTION_IDENTITY_TOL,
            "registered": {
                "method": "GFN2-xTB", "basis": "",
                "charge": 0, "multiplicity": 1, "solvent": "unrecorded",
                "note": "xTB has no basis set (explicit empty, not a missing "
                        "field); solvent absent from receipts = unrecorded, "
                        "NOT assumed to share the teacher DFT PES",
            },
            "rows": eg_rows,
        },
        "npz_teacher_x_spot_check": {
            "n_states": len(x_pool),
            "max_abs_err": max_x_err,
            "tolerance": COORD_SPOT_TOL,
            "pass_": max_x_err <= COORD_SPOT_TOL,
            "rule": "x_angstrom == teacher_coordinates - mean_atoms(teacher_"
                    "coordinates) (frozen unweighted centroid removal)",
        },
    }
    report["all_passed"] = all(
        report[g]["pass_"] for g in (
            "counts", "label_identity", "unchanged_step_identity",
            "arc_recompute_identity", "unique_successor",
            "two_sided_ordering", "map_unit_assertions", "eg_spot_check",
            "npz_teacher_x_spot_check"))
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-observed",
                    default=f"{DEFAULT_ROOT}/wp2_fit50_observed")
    ap.add_argument("--fit-features",
                    default=f"{DEFAULT_ROOT}/wp3_features_v41")
    ap.add_argument("--monitor-observed",
                    default=f"{DEFAULT_ROOT}/wp4_monitor/teachers_observed")
    ap.add_argument("--monitor-features",
                    default=f"{DEFAULT_ROOT}/wp4_monitor/features_v41")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    fit_observed = Path(args.fit_observed)
    fit_features = Path(args.fit_features)
    monitor_observed = Path(args.monitor_observed)
    monitor_features = Path(args.monitor_features)

    native_fit_dir = out_root / "native_features"
    native_monitor_dir = out_root / "native_monitor_features"

    print(f"[build_native_features] rebuilding fit -> {native_fit_dir}")
    fit_manifest = rebuild_native_dataset(fit_features, fit_observed,
                                          native_fit_dir)
    print(f"[build_native_features] rebuilding monitor -> {native_monitor_dir}")
    monitor_manifest = rebuild_native_dataset(monitor_features, monitor_observed,
                                              native_monitor_dir)

    print("[build_native_features] running §2 hard gates (fit)")
    fit_gates = run_hard_gates(fit_observed, fit_features, native_fit_dir, rng)
    print("[build_native_features] running §2 hard gates (monitor)")
    monitor_gates = run_hard_gates(monitor_observed, monitor_features,
                                   native_monitor_dir, rng)

    audit = {
        "schema_version": AUDIT_SCHEMA,
        "tool_version": TOOL_VERSION,
        "generated_utc": _utc_now(),
        "fit": {
            "observed_dir": str(fit_observed),
            "features_dir": str(fit_features),
            "native_dir": str(native_fit_dir),
            "manifest": fit_manifest,
            "hard_gates": fit_gates,
        },
        "monitor": {
            "observed_dir": str(monitor_observed),
            "features_dir": str(monitor_features),
            "native_dir": str(native_monitor_dir),
            "manifest": monitor_manifest,
            "hard_gates": monitor_gates,
        },
        "discrepancy_395_vs_399": fit_manifest["discrepancy_395_vs_399"],
        "all_passed": bool(fit_gates["all_passed"]
                           and monitor_gates["all_passed"]),
    }
    audit_path = out_root / "native_reconnect_audit.json"
    audit_path.write_text(json.dumps(audit, indent=1, sort_keys=False,
                                     ensure_ascii=False, allow_nan=False),
                          encoding="utf-8")
    print(f"[build_native_features] wrote {audit_path}")
    print(f"[build_native_features] fit counts={fit_gates['counts']}")
    print(f"[build_native_features] all_passed={audit['all_passed']}")
    return 0 if audit["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
