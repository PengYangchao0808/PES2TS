#!/usr/bin/env python3
"""E6 V5.3EXEC: full no-physics rollout execution (W-FEAT, H20 mode).

Runs the FROZEN V5.3 rollout matrix per campaign ``v53_rollout``:

* all 12 frozen curves (``v53/rollout_curves.json``) ×
  steps {5, 10, 20} × arms {classic, r3, v52};
* BOTH intent directions per curve (forward starts at endpoint A,
  reverse at endpoint B — the frozen list records both);
* ``r3`` arm = R3 winner F2-bidir fold0 seed 17 checkpoint;
* ``v52`` arm = V5.2 QT checkpoint — DIAGNOSTIC-ONLY everywhere
  (``diagnostic_only=true`` + ``diagnostic_label=
  "v52_supervision_failed_no_claim"``; G3-V52Supervision FAILED, H20:
  V5.3 = upper-bound controllers only, v52 participates for engineering
  comparison only, NOT as a validated controller).

Per-episode metrics (honest diagnostics, TS_candidate_only):

* d_key progression vs true IRC arc: per-step nearest true-arc node
  (arc progress series), contact-made vs the TS node's d_key
  (±0.15 Å/CV), per-step CV sign agreement (vs intent Δξ, already in the
  rollout steps);
* displacement magnitude + clip events (per-step disp_l1_sum,
  clip_factor<1), max key-atom excursion from start;
* final-geometry distance-to-TS-candidate: legal Kabsch RMSD (all atoms
  and K atoms) to the true TS node frame — diagnostics tier only,
  never a policy input;
* divergence / collision / non-finite counts.

Determinism: the whole matrix is seeded (torch/np/manual seed=17 before
each episode); two episodes are re-run and compared BITWISE.

Machine-checked parity: ``check_budget_parity_v3`` over all episodes
(same budget + same step-0 start state across arms) and
``enforce_diagnostic_labeling_v3`` over all episodes.

Usage::

    python3 scripts/agent_ops/feature_gate/v53_rollout_exec.py run-rollouts \
        --tables-dir outputs/pesn/30h_campaign/data \
        --curves outputs/pesn/30h_campaign/v53/rollout_curves.json \
        --r3-checkpoint outputs/pesn/30h_campaign/r3_runs/<run>/checkpoint_best.pt \
        --v52-checkpoint outputs/pesn/30h_campaign/v52_runs/<run>/checkpoint_best.pt \
        --out-dir outputs/pesn/30h_campaign/v53/rollouts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (  # noqa: E402
    KeyAtomDatasetV3,
    kabsch_rmsd_v52,
)

CURVE_LIST_SCHEMA = "pes2ts_v53_rollout_curves_v1"
EPISODES_SCHEMA = "pes2ts_v53_rollout_episodes_v1"
MANIFEST_SCHEMA = "pes2ts_v53_rollout_manifest_v1"
SEED_V53 = 17


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _d_key_of(pos: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    return np.array(
        [float(np.linalg.norm(pos[j] - pos[i])) for i, j in pairs],
        dtype=np.float64)


def _arc_metrics(
    *,
    steps: list[dict[str, Any]],
    arc_nodes: list[int],
    arc_dkey: list[np.ndarray],
    ts_index: int,
    final_xyz: np.ndarray,
    ts_xyz: np.ndarray,
    key_rows: tuple[int, ...],
    contact_tol: float,
) -> dict[str, Any]:
    """d_key progression vs true IRC arc + final distance-to-TS-candidate."""
    ts_pos_in_arc = arc_nodes.index(ts_index)
    ts_dkey = arc_dkey[ts_pos_in_arc]

    progress: list[float] = []
    contact_any = False
    for s in steps:
        d = np.asarray(s["d_key"], dtype=np.float64)
        dists = [float(np.linalg.norm(d - a)) for a in arc_dkey]
        nearest = int(np.argmin(dists))
        frac = nearest / max(1, len(arc_nodes) - 1)
        progress.append(frac)
        if float(np.abs(d - ts_dkey).max()) <= contact_tol:
            contact_any = True

    nondecreasing = sum(
        1 for a, b in zip(progress, progress[1:]) if b >= a)
    final_ts_rmsd_all = kabsch_rmsd_v52(final_xyz, ts_xyz)
    final_ts_rmsd_key = kabsch_rmsd_v52(
        final_xyz[np.asarray(key_rows)], ts_xyz[np.asarray(key_rows)])
    d_final = np.asarray(steps[-1]["d_key"], dtype=np.float64)
    d_start = np.asarray(arc_dkey[0], dtype=np.float64)
    return {
        "arc_progress_series": progress,
        "final_arc_progress": progress[-1],
        "arc_progress_nondecreasing_fraction": (
            nondecreasing / max(1, len(progress) - 1)),
        "contact_made_ts_dkey": contact_any,
        "final_max_cv_dev_from_ts_dkey": float(
            np.abs(d_final - ts_dkey).max()),
        "final_ts_rmsd_all": float(final_ts_rmsd_all),
        "final_ts_rmsd_key": float(final_ts_rmsd_key),
        "start_max_cv_dev_from_ts_dkey": float(
            np.abs(d_start - ts_dkey).max()),
    }


def run_matrix(args: argparse.Namespace) -> int:
    import torch

    from pes2ts_core.pesn.models.rollout_v3 import (
        ARMS_V3,
        CLAIM_SCOPE_V3,
        EPISODE_SCHEMA_V3,
        NO_PHYSICS_STEPS_V3,
        TRUST_REGION_MAX_ATOM_ANGSTROM,
        V52_DIAGNOSTIC_LABEL,
        RolloutStartV3,
        check_budget_parity_v3,
        aggregate_episode_records_v3,
        enforce_diagnostic_labeling_v3,
        load_r3_checkpoint_v3,
        load_v52_checkpoint_v3,
        run_rollout_v3,
    )

    tables_dir = Path(args.tables_dir)
    curves_path = Path(args.curves)
    out_dir = Path(args.out_dir)
    curves_doc = json.loads(curves_path.read_text(encoding="utf-8"))
    if curves_doc.get("schema") != CURVE_LIST_SCHEMA or not curves_doc.get(
            "frozen"):
        raise SystemExit(f"{curves_path} is not a frozen v53 curve list")
    steps_budget = tuple(int(s) for s in curves_doc["no_physics_steps"])
    if steps_budget != NO_PHYSICS_STEPS_V3:
        raise SystemExit(
            f"frozen list steps {steps_budget} != harness budget "
            f"{NO_PHYSICS_STEPS_V3}")
    if tuple(curves_doc["arms"]) != ARMS_V3:
        raise SystemExit(
            f"frozen list arms {curves_doc['arms']} != harness {ARMS_V3}")

    ds = KeyAtomDatasetV3(tables_dir)
    t0 = time.time()
    r3_model = load_r3_checkpoint_v3(args.r3_checkpoint, device="cpu")
    v52_model = load_v52_checkpoint_v3(args.v52_checkpoint, device="cpu")
    ckpt_sha = {
        "r3": _sha256_file(Path(args.r3_checkpoint)),
        "v52": _sha256_file(Path(args.v52_checkpoint)),
    }
    models = {"classic": None, "r3": r3_model, "v52": v52_model}
    load_s = time.time() - t0

    t0 = time.time()
    records: list[dict[str, Any]] = []
    for entry in curves_doc["curves"]:
        cid = entry["curve_id"]
        fwd_start = int(entry["start_node_index"])
        n_nodes = int(entry["n_nodes"])
        ts_index = int(entry["ts_index_record_only"])
        endpoints = ds.curve_endpoints(cid)
        assert fwd_start == int(endpoints[0]), f"{cid}: start != endpoint A"
        b_start = int(endpoints[1])
        atomic_numbers = next(
            rec.atomic_numbers for rec in ds.samples if rec.curve_id == cid)

        for direction in ("forward", "reverse"):
            if direction == "forward":
                intent = ds.intents[(cid, str(entry["forward_intent_id"]))]
                start_node = fwd_start
                arc_nodes = list(range(fwd_start, ts_index + 1))
            else:
                intent = ds.intents[(cid, str(entry["reverse_intent_id"]))]
                start_node = b_start
                arc_nodes = list(range(b_start, ts_index - 1, -1))
            cv_pairs = [(int(i), int(j)) for i, j in intent.cv_coordinates]
            xyz_start = np.asarray(ds.frame_xyz(cid, start_node),
                                   dtype=np.float64)
            ts_xyz = np.asarray(ds.frame_xyz(cid, ts_index),
                                dtype=np.float64)
            arc_dkey = {
                n: _d_key_of(np.asarray(ds.frame_xyz(cid, n),
                                        dtype=np.float64), cv_pairs)
                for n in arc_nodes}
            start = RolloutStartV3(
                curve_id=cid, node_index=start_node, intent=intent,
                atomic_numbers=atomic_numbers,
                xyz=tuple(tuple(float(v) for v in row) for row in xyz_start),
                d0=0.08)
            for arm in ARMS_V3:
                for n_steps in steps_budget:
                    torch.manual_seed(SEED_V53)
                    np.random.seed(SEED_V53)
                    t_ep = time.time()
                    result = run_rollout_v3(
                        arm=arm, start=start, n_steps=n_steps,
                        model=models[arm], level="F2", device="cpu")
                    ep_wall = time.time() - t_ep
                    rec = result.to_dict()
                    rec["rollout_schema"] = rec["schema"]  # v3 harness schema
                    rec["schema"] = EPISODE_SCHEMA_V3
                    rec["fold"] = int(entry["fold"])
                    rec["k_size"] = int(entry["k_size"])
                    rec["start_node_index"] = start_node
                    rec["ts_index_record_only"] = ts_index
                    rec["diagnostic_only"] = arm == "v52"
                    rec["diagnostic_label"] = (
                        V52_DIAGNOSTIC_LABEL if arm == "v52" else None)
                    rec["claim_scope"] = CLAIM_SCOPE_V3
                    rec["checkpoint_sha256"] = ckpt_sha.get(arm)
                    rec["episode_wall_seconds"] = ep_wall
                    rec["trust_region_max_atom"] = (
                        TRUST_REGION_MAX_ATOM_ANGSTROM)

                    # arc/TS diagnostics (d_ts = record-only diagnostics tier)
                    steps_dicts = rec["steps"]
                    arc_dkey_list = [arc_dkey[n] for n in arc_nodes]
                    metrics = _arc_metrics(
                        steps=steps_dicts,
                        arc_nodes=arc_nodes, arc_dkey=arc_dkey_list,
                        ts_index=ts_index,
                        final_xyz=np.asarray(rec["final_xyz"],
                                             dtype=np.float64),
                        ts_xyz=ts_xyz, key_rows=result.key_rows,
                        contact_tol=0.15)
                    rec["summary"] = {**rec["summary"], **metrics}
                    records.append(rec)
                    s = rec["summary"]
                    print(
                        f"{cid} {direction:8s} {arm:7s} s={n_steps:2d} "
                        f"agree={s['cv_sign_agree_fraction']:.3f} "
                        f"contact={int(s['contact_made_ts_dkey'])} "
                        f"prog={s['final_arc_progress']:.2f} "
                        f"tsRMSD={s['final_ts_rmsd_all']:.2f} "
                        f"coll={s['n_collisions']} "
                        f"div={int(s['divergence_flag'])} "
                        f"wall={ep_wall:.1f}s", flush=True)
    rollout_wall = time.time() - t0

    # ---- determinism: two episodes bitwise-rerun ------------------------
    det_targets = [
        (curves_doc["curves"][0]["curve_id"], "forward", "classic", 5),
        (curves_doc["curves"][0]["curve_id"], "forward", "r3", 20),
    ]
    determinism: list[dict[str, Any]] = []
    for cid, direction, arm, n_steps in det_targets:
        idx = next(
            i for i, r in enumerate(records)
            if r["curve_id"] == cid and r["direction"] == direction
            and r["arm"] == arm and r["n_steps_requested"] == n_steps)
        entry = next(e for e in curves_doc["curves"]
                     if e["curve_id"] == cid)
        intent_id = (entry["forward_intent_id"] if direction == "forward"
                     else entry["reverse_intent_id"])
        intent = ds.intents[(cid, intent_id)]
        start_node = (int(entry["start_node_index"]) if direction == "forward"
                      else int(ds.curve_endpoints(cid)[1]))
        atomic_numbers = next(
            rec.atomic_numbers for rec in ds.samples if rec.curve_id == cid)
        xyz_start = ds.frame_xyz(cid, start_node)
        start = RolloutStartV3(
            curve_id=cid, node_index=start_node, intent=intent,
            atomic_numbers=atomic_numbers, xyz=xyz_start, d0=0.08)
        torch.manual_seed(SEED_V53)
        np.random.seed(SEED_V53)
        rerun = run_rollout_v3(
            arm=arm, start=start, n_steps=n_steps, model=models[arm],
            level="F2", device="cpu").to_dict()
        orig = records[idx]
        # keys stamped by the exec layer (not produced by the harness):
        # excluded from the bitwise comparison of harness outputs.
        _stamped = {
            "schema", "rollout_schema", "fold", "k_size",
            "start_node_index", "ts_index_record_only", "diagnostic_only",
            "diagnostic_label", "checkpoint_sha256",
            "episode_wall_seconds", "trust_region_max_atom",
        }
        mismatches = []
        for k in rerun:
            if k in _stamped:
                continue
            if k == "summary":
                for sk in rerun["summary"]:
                    if json.dumps(orig["summary"].get(sk), sort_keys=True) != \
                            json.dumps(rerun["summary"][sk], sort_keys=True):
                        mismatches.append(f"summary.{sk}")
            elif json.dumps(orig.get(k), sort_keys=True) != \
                    json.dumps(rerun[k], sort_keys=True):
                mismatches.append(k)
        bitwise = not mismatches
        determinism.append({
            "curve_id": cid, "direction": direction, "arm": arm,
            "n_steps": n_steps, "bitwise_equal": bool(bitwise)})
        if not bitwise:
            raise SystemExit(
                f"DETERMINISM FAIL: {cid}/{direction}/{arm}/s{n_steps} "
                f"rerun is not bitwise-equal; mismatches: {mismatches}")
        print(f"determinism OK: {cid} {direction} {arm} s{n_steps} "
              f"bitwise-equal", flush=True)

    # ---- machine checks -------------------------------------------------
    enforce_diagnostic_labeling_v3(records)
    parity = check_budget_parity_v3(records)
    aggregate = aggregate_episode_records_v3(records)

    out_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = out_dir / "episodes.jsonl"
    with episodes_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False,
                                sort_keys=True) + "\n")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "campaign": "PES2TS_30H_R3_V52_V53",
        "stage": "G4",
        "hard_stop_mode": "H20 (V5.2 supervision failed → V5.3 = "
                          "upper-bound controllers only; v52 arm is "
                          "diagnostic-only, never a validated controller)",
        "curve_list_sha256": curves_doc.get("curve_list_sha256", ""),
        "n_curves": len(curves_doc["curves"]),
        "directions": ["forward", "reverse"],
        "arms": list(ARMS_V3),
        "steps": list(steps_budget),
        "trust_region_max_atom": TRUST_REGION_MAX_ATOM_ANGSTROM,
        "seed": SEED_V53,
        "device": "cpu (rollout inference < lock-acquisition overhead; "
                  "GPU left free — documented choice)",
        "checkpoints": {
            "r3": {"path": str(Path(args.r3_checkpoint)),
                   "sha256": ckpt_sha["r3"],
                   "arm": "F2-bidir fold0 seed17 (R3 winner)"},
            "v52": {"path": str(Path(args.v52_checkpoint)),
                    "sha256": ckpt_sha["v52"],
                    "arm": "V5.2 QT",
                    "diagnostic_only": True,
                    "diagnostic_label": V52_DIAGNOSTIC_LABEL},
        },
        "n_episodes": len(records),
        "claim_scope": CLAIM_SCOPE_V3,
        "verified_ts_claim": "forbidden",
        "determinism": determinism,
        "budget_parity": parity,
        "wall_seconds": {
            "checkpoint_load": load_s,
            "rollout_matrix": rollout_wall,
            "episodes_total": float(
                sum(r["episode_wall_seconds"] for r in records)),
        },
        "aggregate": aggregate,
        "episodes_file": str(episodes_path),
    }
    manifest_path = out_dir / "rollouts_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"-> {episodes_path} ({len(records)} episodes)")
    print(f"-> {manifest_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run-rollouts")
    p.add_argument("--tables-dir", required=True)
    p.add_argument("--curves", required=True)
    p.add_argument("--r3-checkpoint", required=True)
    p.add_argument("--v52-checkpoint", required=True)
    p.add_argument("--out-dir", required=True)
    p.set_defaults(func=run_matrix)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
