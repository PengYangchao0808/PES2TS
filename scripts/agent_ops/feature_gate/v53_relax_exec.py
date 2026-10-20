#!/usr/bin/env python3
"""E6 V5.3EXEC: xTB constrained-relax bridge execution (W-FEAT, G4 stage).

Executes the supervisor-cleared (G4 stage, yaml ``v53_rollout``) real-xTB
constrained optimization on the 20-step no-physics rollout ENDPOINT of
every frozen curve × arm:

* one geometry per curve per arm = forward-direction 20-step endpoint
  from ``v53/rollouts/episodes.jsonl``;
* constraints = the intent's K-pair CVs, ALL driven to their target
  distances, where each target is the TRUE IRC TS-node distance for that
  CV pair (diagnostics-tier information — ``ts_index_record_only``; the
  relax is out-of-loop and its result NEVER feeds any policy, plan §5);
* budget/recipe frozen and IDENTICAL across arms
  (:data:`pes2ts_core.pesn.models.rollout_v3.RELAX_RECIPE_V3`): GFN-2,
  force constant 50 eV/Å², normal opt level, maxcycle 500, 4 threads,
  300 s timeout, no fixed atoms — machine-checked by
  ``check_relax_budget_parity_v3`` (per-curve targets must be bit-equal
  across arms);
* physics failures are recorded per curve-arm (``physics_failed`` /
  ``timeout`` / ``not_converged``) and never abort sibling runs;
* claim scope: ``TS_candidate_only`` — single-imaginary-frequency
  verification is NOT attempted (OptTS territory, out of scope).

Request payloads are built with the frozen
``build_constraint_request`` contract (bridge continuity with
``v53_rollout_prep.py preflight-relax``); the executor translates the
same recipe to an xtb ``$constrain`` control file (the ACP CLI transport
is not used here; the request payload is archived per run for
reproducibility).

Usage::

    python3 scripts/agent_ops/feature_gate/v53_relax_exec.py run-relax \\
        --tables-dir outputs/pesn/30h_campaign/data \\
        --curves outputs/pesn/30h_campaign/v53/rollout_curves.json \\
        --episodes outputs/pesn/30h_campaign/v53/rollouts/episodes.jsonl \\
        --out-dir outputs/pesn/30h_campaign/v53/relax
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from pes2ts_core.pesn.models.rollout_v3 import (  # noqa: E402
    CLAIM_SCOPE_V3,
    RELAX_RECIPE_V3,
    V52_DIAGNOSTIC_LABEL,
    PhysicsOutcomeV3,
    check_relax_budget_parity_v3,
    execute_relax_episode_v3,
)
from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (  # noqa: E402
    KeyAtomDatasetV3,
)

XTB_DEFAULT = "/opt/software/xtb/bin/xtb"
RELAX_MANIFEST_SCHEMA = "pes2ts_v53_relax_manifest_v1"

_ELEMENTS = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
    "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr",
    "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr")


def _write_xyz(path: Path, elements: list[str],
               coords: np.ndarray) -> None:
    lines = [str(len(elements)), "pes2ts v53 constrained relax seed"]
    for el, row in zip(elements, coords):
        lines.append(f"{el:2s} {row[0]: .10f} {row[1]: .10f} {row[2]: .10f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _seed_min_interatomic(coords: np.ndarray) -> float:
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    return float(dist.min())


def _xtb_runner_factory(xtb_bin: str, recipe: dict[str, Any]):
    """Runner closure executing one xtb constrained opt in ``workdir``."""

    def runner(request: dict[str, Any], workdir: str) -> PhysicsOutcomeV3:
        wd = Path(workdir)
        wd.mkdir(parents=True, exist_ok=True)
        elements = list(request["source"]["elements"])
        coords = np.asarray(request["source"]["geometry"], dtype=np.float64)
        _write_xyz(wd / "coord.xyz", elements, coords)

        constrain_lines = ["$constrain",
                           f"  force constant={recipe['force_constant']}"]
        for c in request["recipe"]["constraints"]:
            if c["kind"] != "distance":
                raise RuntimeError(
                    f"v1 executor supports distance constraints only, "
                    f"got {c['kind']!r}")
            i, j = (int(a) + 1 for a in c["atoms"])  # xtb is 1-based
            constrain_lines.append(
                f"  distance: {i}, {j}, {c['target']:.6f}")
        constrain_lines.append("$end")
        constrain_lines += ["$opt", f"  maxcycle={recipe['max_cycle']}",
                            "$end"]
        control = "\n".join(constrain_lines) + "\n"
        (wd / "constrain.inp").write_text(control, encoding="utf-8")

        cmd = [xtb_bin, "coord.xyz", "--opt", str(recipe["opt_level"]),
               "--input", "constrain.inp", "--gfn",
               str(recipe["gfn_level"]), "--chrg",
               str(recipe["charge"]), "--parallel",
               str(recipe["threads"])]
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = str(recipe["threads"])
        t0 = time.time()
        timed_out = False
        try:
            proc = subprocess.Popen(
                cmd, cwd=str(wd), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, start_new_session=True)
            try:
                out, err = proc.communicate(
                    timeout=float(recipe["timeout_seconds"]))
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(proc.pid, signal.SIGKILL)
                out, err = proc.communicate()
        except FileNotFoundError as exc:
            return PhysicsOutcomeV3(
                status="physics_failed", returncode=None, wall_seconds=0.0,
                final_xyz=None, error=f"xtb launch failed: {exc}")
        wall = time.time() - t0
        (wd / "xtb_stdout.log").write_text(out or "", encoding="utf-8")
        (wd / "xtb_stderr.log").write_text(err or "", encoding="utf-8")

        converged = "OPTIMIZATION CONVERGED" in (out or "")
        final_xyz = None
        opt_xyz = wd / "xtbopt.xyz"
        if opt_xyz.is_file():
            rows = opt_xyz.read_text(encoding="utf-8").splitlines()
            n = int(rows[0].strip())
            final_xyz = tuple(
                tuple(float(x) for x in line.split()[1:4])
                for line in rows[2:2 + n])
        conv_line = next(
            (ln.strip() for ln in (out or "").splitlines()
             if "OPTIMIZATION CONVERGED" in ln
             or "FAILED CYCLE" in ln or "FAILED TO" in ln), None)
        if timed_out:
            status = "timeout"
        elif converged and final_xyz is not None:
            status = "converged"
        elif proc.returncode != 0:
            status = "physics_failed"
        else:
            status = "not_converged"
        return PhysicsOutcomeV3(
            status=status, returncode=proc.returncode,
            wall_seconds=wall, final_xyz=final_xyz,
            convergence_line=conv_line,
            error=None if status in ("converged", "not_converged")
            else f"status={status} rc={proc.returncode}")

    return runner


def run_relax(args: argparse.Namespace) -> int:
    from pes2ts_core.integration.acp.xtb_constraints_request import (
        DriverTarget,
        build_constraint_request,
    )

    tables_dir = Path(args.tables_dir)
    out_dir = Path(args.out_dir)
    curves_doc = json.loads(Path(args.curves).read_text(encoding="utf-8"))
    if not curves_doc.get("frozen"):
        raise SystemExit("curves file is not a frozen v53 curve list")
    episodes = [
        json.loads(line) for line in
        Path(args.episodes).read_text(encoding="utf-8").splitlines()
        if line.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = set(arms) - {"classic", "r3", "v52"}
    if unknown:
        raise SystemExit(f"unknown arms {sorted(unknown)}")
    mandatory = {"classic", "r3"}
    if not mandatory.issubset(set(arms)):
        raise SystemExit(f"arms must include mandatory {sorted(mandatory)}")

    xtb_bin = args.xtb_bin
    if not Path(xtb_bin).is_file():
        raise SystemExit(f"xtb binary not found: {xtb_bin}")
    recipe = dict(RELAX_RECIPE_V3)
    runner = _xtb_runner_factory(xtb_bin, recipe)

    ds = KeyAtomDatasetV3(tables_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"

    results: list[dict[str, Any]] = []
    t0 = time.time()
    for entry in curves_doc["curves"]:
        cid = entry["curve_id"]
        ts_index = int(entry["ts_index_record_only"])
        forward_intent = ds.intents[(cid, str(entry["forward_intent_id"]))]
        key_set = set(forward_intent.key_rows())
        cv_pairs = [(int(i), int(j)) for i, j in
                    forward_intent.cv_coordinates]
        if not cv_pairs:
            print(f"{cid}: no CV coordinates in intent — physics_failed "
                  f"record for all arms", flush=True)
            for arm in arms:
                results.append({
                    "schema": "pes2ts_v53_relax_result_v1",
                    "curve_id": cid, "arm": arm, "direction": "forward",
                    "status": "physics_failed", "returncode": None,
                    "wall_seconds": 0.0, "convergence_line": None,
                    "error": "intent has no CV coordinates",
                    "driver_pairs": [], "target_distances": [],
                    "achieved_distances": None, "dev_from_target": None,
                    "request_sha256": None, "recipe": recipe,
                    "diagnostic_only": arm == "v52",
                    "diagnostic_label": (
                        V52_DIAGNOSTIC_LABEL if arm == "v52" else None),
                    "claim_scope": CLAIM_SCOPE_V3, "note": ""})
            continue
        ts_xyz = np.asarray(ds.frame_xyz(cid, ts_index), dtype=np.float64)
        # target distances: TS-node d_key per intent CV pair (identical
        # across arms by construction — geometry-independent rule)
        targets = [float(np.linalg.norm(ts_xyz[j] - ts_xyz[i]))
                   for i, j in cv_pairs]
        atomic_numbers = next(
            rec.atomic_numbers for rec in ds.samples
            if rec.curve_id == cid)
        elements = [_ELEMENTS[z - 1] for z in atomic_numbers]

        for arm in arms:
            ep = next(
                (r for r in episodes
                 if r["curve_id"] == cid and r["arm"] == arm
                 and r["direction"] == "forward"
                 and r["n_steps_requested"] == 20), None)
            run_wd = runs_dir / f"{cid}__{arm}"
            if ep is None:
                results.append({
                    "schema": "pes2ts_v53_relax_result_v1",
                    "curve_id": cid, "arm": arm, "direction": "forward",
                    "status": "physics_failed", "returncode": None,
                    "wall_seconds": 0.0, "convergence_line": None,
                    "error": "no 20-step forward episode in episodes.jsonl",
                    "driver_pairs": cv_pairs, "target_distances": targets,
                    "achieved_distances": None, "dev_from_target": None,
                    "request_sha256": None, "recipe": recipe,
                    "diagnostic_only": arm == "v52",
                    "diagnostic_label": (
                        V52_DIAGNOSTIC_LABEL if arm == "v52" else None),
                    "claim_scope": CLAIM_SCOPE_V3, "note": ""})
                continue
            endpoint = np.asarray(ep["final_xyz"], dtype=np.float64)
            request = build_constraint_request(
                geometry=[[float(v) for v in row] for row in endpoint],
                elements=elements, charge=recipe["charge"],
                multiplicity=recipe["multiplicity"],
                targets=[DriverTarget(
                    driver_id=f"cv{i}_{j}", kind="distance",
                    atoms=(i, j), target=t, unit="angstrom")
                    for (i, j), t in zip(cv_pairs, targets)],
                reaction_id=cid,
                point_id=f"{cid}:{arm}:forward:s20",
                gfn_level=recipe["gfn_level"],
                force_constant=recipe["force_constant"],
                opt_level=recipe["opt_level"],
                max_cycle=recipe["max_cycle"],
                threads=recipe["threads"],
                timeout_seconds=recipe["timeout_seconds"])
            run_wd.mkdir(parents=True, exist_ok=True)
            (run_wd / "request.json").write_text(
                json.dumps(request, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8")
            result = execute_relax_episode_v3(
                curve_id=cid, arm=arm, direction="forward",
                driver_pairs=cv_pairs, target_distances=targets,
                request=request, runner=runner, workdir=str(run_wd))
            # honest failure attribution: seed geometry quality of the
            # rollout endpoint this relax started from
            result["seed_min_interatomic"] = _seed_min_interatomic(
                endpoint)
            result["seed_from_episode"] = {
                "direction": "forward", "n_steps_requested": 20,
                "episode_input_step0_sha256": ep["input_step0_sha256"]}
            (run_wd / "result.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8")
            results.append(result)
            dev = result["dev_from_target"]
            dev_s = ("n/a" if dev is None else
                     ",".join(f"{d:+.3f}" for d in dev))
            print(f"{cid} {arm:7s} status={result['status']:13s} "
                  f"wall={result['wall_seconds']:6.1f}s "
                  f"dev=[{dev_s}]", flush=True)

    wall_total = time.time() - t0
    parity = check_relax_budget_parity_v3(results)

    results_path = out_dir / "relax_results.jsonl"
    with results_path.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True)
                     + "\n")

    by_status: dict[str, int] = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    per_arm: dict[str, dict[str, int]] = {}
    for r in results:
        per_arm.setdefault(r["arm"], {}).setdefault(r["status"], 0)
        per_arm[r["arm"]][r["status"]] += 1
    manifest = {
        "schema": RELAX_MANIFEST_SCHEMA,
        "campaign": "PES2TS_30H_R3_V52_V53",
        "stage": "G4",
        "clearance": ("supervisor G4-stage constrained-relax clearance "
                      "under yaml v53_rollout: same budget + same "
                      "constraints across arms"),
        "xtb_binary": str(Path(xtb_bin).resolve()),
        "recipe": recipe,
        "constraint_rule": (
            "all intent CV-pair distances driven to the TRUE IRC "
            "TS-node distances (ts_index_record_only, diagnostics tier; "
            "identical targets across arms by construction)"),
        "arms": arms,
        "n_curves": len(curves_doc["curves"]),
        "n_results": len(results),
        "status_counts": by_status,
        "status_counts_per_arm": per_arm,
        "parity": parity,
        "wall_seconds_total": wall_total,
        "wall_seconds_sum_runs": float(
            sum(r["wall_seconds"] for r in results)),
        "claim_scope": CLAIM_SCOPE_V3,
        "verified_ts_claim": "forbidden",
        "ts_frequency_verification": "not_attempted (OptTS territory)",
        "results_file": str(results_path),
    }
    manifest_path = out_dir / "relax_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(json.dumps(by_status, indent=2))
    print(f"-> {results_path}")
    print(f"-> {manifest_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run-relax")
    p.add_argument("--tables-dir", required=True)
    p.add_argument("--curves", required=True)
    p.add_argument("--episodes", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--xtb-bin", default=XTB_DEFAULT)
    p.add_argument("--arms", default="classic,r3,v52",
                   help="comma-separated; classic+r3 are mandatory")
    p.set_defaults(func=run_relax)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
