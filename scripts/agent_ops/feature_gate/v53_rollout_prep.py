#!/usr/bin/env python3
"""E6 V5.3 rollout prep tooling (W-FEAT): curve pre-registration + relax
bridge preflight/dry-run adapter.

Two subcommands (campaign ``v53_rollout`` block, FROZEN):

``select-curves``
    Deterministically select 8–12 representative Development curves from
    the 48 usable R3 curves and freeze the list to
    ``outputs/pesn/30h_campaign/v53/rollout_curves.json``.

    Selection rule (documented + machine-checked, plan §6.2
    预注册代表性曲线):

    * pool = curves with ≥1 ``label_status=ok`` sample in the E1 six
      tables AND fold ∈ {0,1,2} (Development region only — confirm
      folds stay sealed);
    * strata = fold × K-quartile, where K = |key_rows| of the curve's
      forward intent and quartiles are rank-based over the pool ordered
      by (K, curve_id) — deterministic, no RNG;
    * per stratum the winner is ``min sha256("v53-rollout|17|<curve_id>")``
      (campaign seed 17);
    * one curve per non-empty stratum, capped at 12; if the cap trims,
      strata are visited in (fold, quartile) order;
    * each entry records forward AND reverse intent ids (arms run both
      directions), the fixed start node (endpoint A of the canonical
      A→B path), K, fold, quartile and n_nodes.

``preflight-relax``
    Constrained-relax bridge availability check + request DRY-RUN only.
    Verifies the ACP executor wiring (source checkout, CLI module,
    interpreter, xtb binary) and, when the tables are available, builds a
    real ``pes2ts_xtb_constrained_opt_request_v1`` payload for one
    pre-registered curve via ``build_constraint_request`` WITHOUT
    executing anything.  Physics execution is forbidden in this task: the
    verdict carries ``physics_missing: physics_not_run`` unless a future
    authorized stage flips it (campaign yaml default).

``smoke-rollout``
    CPU 5-step no-physics rollout on pre-registered curves with a real
    R3 checkpoint (diagnostics only; ``TS_candidate_only`` claim scope).

Usage::

    python3 scripts/agent_ops/feature_gate/v53_rollout_prep.py select-curves \\
        --tables-dir outputs/pesn/30h_campaign/data \\
        --out outputs/pesn/30h_campaign/v53/rollout_curves.json
    python3 scripts/agent_ops/feature_gate/v53_rollout_prep.py preflight-relax \\
        --tables-dir outputs/pesn/30h_campaign/data \\
        --curves outputs/pesn/30h_campaign/v53/rollout_curves.json \\
        --out outputs/pesn/30h_campaign/v53/relax_preflight.json
    python3 scripts/agent_ops/feature_gate/v53_rollout_prep.py smoke-rollout \\
        --tables-dir outputs/pesn/30h_campaign/data \\
        --curves outputs/pesn/30h_campaign/v53/rollout_curves.json \\
        --checkpoint outputs/pesn/30h_campaign/r3_runs/<run>/checkpoint_best.pt \\
        --n-steps 5 --level F2 \\
        --out outputs/pesn/30h_campaign/v53/smoke_rollout_r3.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (  # noqa: E402
    KeyAtomDatasetV3,
)

CURVE_LIST_SCHEMA: str = "pes2ts_v53_rollout_curves_v1"
PREFLIGHT_SCHEMA: str = "pes2ts_v53_relax_preflight_v1"

#: Frozen campaign parameters (config/campaigns/pes2ts_30h_20261020.yaml
#: ``v53_rollout`` — immutable in this task).
SEED_V53: int = 17
DEV_FOLDS_V53: tuple[int, ...] = (0, 1, 2)
N_CURVES_MIN_V53: int = 8
N_CURVES_MAX_V53: int = 12
NO_PHYSICS_STEPS_V53: tuple[int, ...] = (5, 10, 20)
ARMS_V53: tuple[str, ...] = ("classic", "r3", "v52")
PHYSICS_MISSING_DEFAULT: str = "physics_not_run"

#: xtb binary probe locations (ACP config ``execution.xtb.path`` fallbacks).
XTB_PROBE_PATHS: tuple[str, ...] = ("/opt/xtb/bin/xtb", "/usr/local/bin/xtb")


# ---------------------------------------------------------------------------
# curve pre-registration
# ---------------------------------------------------------------------------
def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_rollout_curves(
    tables_dir: Path,
    *,
    seed: int = SEED_V53,
    n_min: int = N_CURVES_MIN_V53,
    n_max: int = N_CURVES_MAX_V53,
) -> dict[str, Any]:
    """Deterministic fold × K-quartile stratified selection (see module doc)."""
    ds = KeyAtomDatasetV3(tables_dir)
    folds = dict(ds.folds)

    # per-curve forward/reverse intent ids + K (from the FORWARD intent)
    fwd_intents: dict[str, str] = {}
    rev_intents: dict[str, str] = {}
    k_size: dict[str, int] = {}
    n_nodes: dict[str, int] = {}
    for rec in ds.samples:
        cid = rec.curve_id
        if rec.intent.direction == "forward":
            fwd_intents.setdefault(cid, rec.intent.intent_id)
            k_size.setdefault(cid, len(rec.intent.key_rows()))
        else:
            rev_intents.setdefault(cid, rec.intent.intent_id)

    usable = sorted(
        cid for cid in fwd_intents
        if cid in folds and folds[cid] in DEV_FOLDS_V53)
    if len(usable) < n_min:
        raise SystemExit(
            f"only {len(usable)} usable Development curves (< {n_min}); "
            f"cannot pre-register a valid representative set")

    # rank-based K quartiles over the pool ordered by (K, curve_id)
    order = sorted(usable, key=lambda c: (k_size[c], c))
    n = len(order)
    quartile = {cid: min(3, (i * 4) // n) for i, cid in enumerate(order)}

    strata: dict[tuple[int, int], list[str]] = defaultdict(list)
    for cid in usable:
        strata[(folds[cid], quartile[cid])].append(cid)

    def _tiebreak(cid: str) -> str:
        return _sha256_hex(f"v53-rollout|{seed}|{cid}")

    picked: list[dict[str, Any]] = []
    for stratum_key in sorted(strata):
        if len(picked) >= n_max:
            break
        winner = min(strata[stratum_key], key=_tiebreak)
        picked.append({
            "curve_id": winner,
            "fold": folds[winner],
            "k_size": k_size[winner],
            "k_quartile": quartile[winner],
            "forward_intent_id": fwd_intents[winner],
            "reverse_intent_id": rev_intents.get(winner, ""),
            "start_node_index": None,   # filled below from irc_curve
            "n_nodes": None,
        })

    # start node = endpoint A of the canonical A→B path (plan §6.1:
    # episode 从实际 IRC 的第一个端点开始); ts_index recorded for
    # diagnostics only — never a policy input.
    import csv as _csv
    with (tables_dir / "irc_curve.csv").open(
            "r", encoding="utf-8", newline="") as fh:
        for row in _csv.DictReader(fh):
            for entry in picked:
                if row["curve_id"] == entry["curve_id"]:
                    entry["start_node_index"] = int(
                        row.get("endpoint_a_index") or 0)
                    entry["ts_index_record_only"] = int(row["ts_index"])
                    entry["n_nodes"] = int(row["n_nodes"])

    if not (n_min <= len(picked) <= n_max):
        raise SystemExit(
            f"selection produced {len(picked)} curves; frozen range is "
            f"[{n_min}, {n_max}]")

    table_files = sorted(tables_dir.glob("*.csv")) + [
        tables_dir / "data_manifest.json"]
    doc = {
        "schema": CURVE_LIST_SCHEMA,
        "frozen": True,
        "selection_rule": (
            "pool = curves with ≥1 label_status=ok sample AND fold in "
            "{0,1,2} (Development only); strata = fold × K-quartile "
            "(rank-based over (K, curve_id) order); per-stratum winner = "
            "min sha256(f'v53-rollout|{seed}|<curve_id>'); one per stratum "
            "in (fold, quartile) stratum order, capped at "
            f"{n_max}; start node = endpoint A; both intent directions "
            "recorded"),
        "seed": seed,
        "dev_folds": list(DEV_FOLDS_V53),
        "n_pool": len(usable),
        "n_strata": len(strata),
        "n_curves": len(picked),
        "no_physics_steps": list(NO_PHYSICS_STEPS_V53),
        "arms": list(ARMS_V53),
        "verified_ts_claim": "forbidden",
        "curves": sorted(picked, key=lambda e: e["curve_id"]),
        "source_sha256": {
            p.name: _sha256_file(p) for p in table_files if p.is_file()
        },
    }
    doc["curve_list_sha256"] = hashlib.sha256(
        json.dumps(doc["curves"], sort_keys=True,
                   separators=(",", ":")).encode("utf-8")).hexdigest()
    return doc


def cmd_select_curves(args: argparse.Namespace) -> int:
    doc = select_rollout_curves(Path(args.tables_dir), seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"froze {doc['n_curves']} curves -> {out}")
    for entry in doc["curves"]:
        print(f"  {entry['curve_id']}  fold={entry['fold']} "
              f"K={entry['k_size']} q{entry['k_quartile']} "
              f"start_node={entry['start_node_index']}")
    return 0


# ---------------------------------------------------------------------------
# constrained-relax bridge preflight (dry-run; NO physics execution)
# ---------------------------------------------------------------------------
def _resolve_acp_wiring() -> dict[str, Any]:
    """ACP wiring from the frozen campaign manifest (read-only)."""
    manifest = REPO_ROOT / "data/manifests/pes2ts_g2t_campaign_s3_full.json"
    if not manifest.is_file():
        return {"available": False, "reason": "acp_manifest_missing"}
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    wiring = doc.get("acp_wiring") or {}
    root = Path(str(wiring.get("root", "")))
    checks = {
        "acp_root": str(root) if root.is_dir() else None,
        "acp_cli_py": (
            str(root / "src/acp/cli.py")
            if (root / "src/acp/cli.py").is_file() else None),
        "acp_python": (
            str(wiring.get("python")) if wiring.get("python") else None),
        "acp_config": (
            str(wiring.get("config_path"))
            if wiring.get("config_path")
            and Path(str(wiring.get("config_path"))).is_file() else None),
    }
    return {"available": bool(checks["acp_root"] and checks["acp_cli_py"]),
            "checks": checks}


def _probe_xtb() -> dict[str, Any]:
    found = shutil.which("xtb")
    for candidate in XTB_PROBE_PATHS:
        if found:
            break
        if Path(candidate).is_file():
            found = candidate
    return {"xtb_binary": found, "available": bool(found)}


def _build_dry_run_request(
    tables_dir: Path, curve_entry: dict[str, Any]) -> dict[str, Any]:
    """Build one real xTB constrained-opt request payload (NOT executed).

    The request drives the intent's FIRST K-pair CV to its post-step
    target distance (current d + Δξ clipped to a one-d0 move), the same
    budget/constraints for every arm; non-key atoms are free (no fix_atoms)
    so the backend relaxes everything except the driver constraint.
    """
    from pes2ts_core.integration.acp.xtb_constraints_request import (
        DriverTarget,
        build_constraint_request,
    )
    ds = KeyAtomDatasetV3(tables_dir)
    curve_id = curve_entry["curve_id"]
    node_index = int(curve_entry["start_node_index"])
    intent_id = str(curve_entry["forward_intent_id"])

    xyz = np.asarray(ds.frame_xyz(curve_id, node_index), dtype=np.float64)
    intent = None
    for (_cid, _iid), rec in ds.intents.items():
        if _cid == curve_id and _iid == intent_id:
            intent = rec
            break
    if intent is None:
        raise SystemExit(f"no forward intent found for {curve_id}")

    key_set = set(intent.key_rows())
    k_pairs = [
        (i, j) for (i, j), (b_src, b_tgt) in sorted(intent.pair_orders().items())
        if b_src != b_tgt and i in key_set and j in key_set]
    if not k_pairs:
        raise SystemExit(f"{curve_id}: no K-pair CV to drive")

    # driver = first K-pair; target = one-d0 directed move along Δξ
    i, j = k_pairs[0]
    d_now = float(np.linalg.norm(xyz[j] - xyz[i]))
    delta_xi = float(intent.cv_deltas[0]) if intent.cv_deltas else 0.0
    d0 = 0.08
    target = max(0.5, d_now + float(np.clip(delta_xi, -1.0, 1.0)) * d0)

    # atomic numbers: from any sample of this curve
    atomic_numbers = next(
        rec.atomic_numbers for rec in ds.samples if rec.curve_id == curve_id)
    elements = [str(_ELEMENTS[z - 1]) for z in atomic_numbers]

    request = build_constraint_request(
        geometry=[[float(v) for v in row] for row in xyz],
        elements=elements,
        charge=0,
        multiplicity=1,
        targets=[DriverTarget(
            driver_id=f"kp{i}_{j}", kind="distance",
            atoms=(int(i), int(j)), target=target, unit="angstrom")],
        reaction_id=curve_id,
        point_id=f"{curve_id}:{node_index}:dryrun",
        gfn_level=2,
        force_constant=50.0,
        opt_level="normal",
        max_cycle=500,
        threads=4,
        timeout_seconds=300,
    )
    return {
        "curve_id": curve_id,
        "node_index": node_index,
        "intent_id": intent_id,
        "driver": {"atoms": [int(i), int(j)], "d_now_angstrom": d_now,
                   "delta_xi": delta_xi, "target_angstrom": target},
        "request_sha256": request["provenance"]["request_sha256"],
        "request_schema": request.get("schema_version", ""),
        "request_built": True,
        "request_executed": False,
    }


#: minimal Z→element table (H..Rn subset needed for the E1 set)
_ELEMENTS: tuple[str, ...] = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
    "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr",
    "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr")


def cmd_preflight_relax(args: argparse.Namespace) -> int:
    tables_dir = Path(args.tables_dir)
    curves_path = Path(args.curves)
    curves_doc = json.loads(curves_path.read_text(encoding="utf-8"))
    if curves_doc.get("schema") != CURVE_LIST_SCHEMA or not curves_doc.get(
            "frozen"):
        raise SystemExit(f"{curves_path} is not a frozen v53 curve list")

    acp = _resolve_acp_wiring()
    xtb = _probe_xtb()

    dry_run: dict[str, Any] | None = None
    dry_run_error: str | None = None
    if args.curve_id:
        entry = next(
            (e for e in curves_doc["curves"]
             if e["curve_id"] == args.curve_id), None)
        if entry is None:
            raise SystemExit(f"{args.curve_id} not in frozen curve list")
    else:
        entry = curves_doc["curves"][0]
    try:
        dry_run = _build_dry_run_request(tables_dir, entry)
    except Exception as exc:  # noqa: BLE001 — preflight reports, never crashes
        dry_run_error = f"{type(exc).__name__}: {exc}"

    runnable = bool(acp.get("available") and xtb.get("available"))
    doc = {
        "schema": PREFLIGHT_SCHEMA,
        "acp": acp,
        "xtb": xtb,
        "executor_runnable": runnable,
        "dry_run_request": dry_run,
        "dry_run_error": dry_run_error,
        "physics_missing": PHYSICS_MISSING_DEFAULT,
        "physics_executed": False,
        "authorization": (
            "preflight/dry-run only; real constrained-relax execution "
            "requires GPU/queue clearance + supervisor approval "
            "(30h plan §6.2); this task never executes physics"),
        "verdict": (
            "available_dry_run_only" if runnable and dry_run else
            "wiring_present_xtb_missing" if acp.get("available") else
            "wiring_incomplete"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(json.dumps({k: doc[k] for k in (
        "verdict", "executor_runnable", "physics_missing",
        "physics_executed")}, indent=2))
    print(f"-> {out}")
    return 0


def cmd_smoke_rollout(args: argparse.Namespace) -> int:
    """CPU 5-step no-physics rollout on 2 pre-registered curves (R3 arm)."""
    import sys as _sys

    import numpy as np

    from pes2ts_core.pesn.models.rollout_v3 import (  # local import: torch
        NO_PHYSICS_STEPS_V3,
        RolloutStartV3,
        load_r3_checkpoint_v3,
        run_rollout_v3,
    )

    tables_dir = Path(args.tables_dir)
    curves_doc = json.loads(
        Path(args.curves).read_text(encoding="utf-8"))
    if curves_doc.get("schema") != CURVE_LIST_SCHEMA or not curves_doc.get(
            "frozen"):
        raise SystemExit("curves file is not a frozen v53 curve list")
    n_steps = int(args.n_steps)
    if n_steps not in NO_PHYSICS_STEPS_V53:
        raise SystemExit(
            f"n_steps={n_steps} not in frozen budget {NO_PHYSICS_STEPS_V53}")

    wanted = args.curve_ids.split(",") if args.curve_ids else None
    entries = curves_doc["curves"]
    if wanted:
        entries = [e for e in entries if e["curve_id"] in wanted]
        if len(entries) != len(wanted):
            raise SystemExit("some requested curve_ids are not frozen")
    else:
        entries = entries[:2]

    ds = KeyAtomDatasetV3(tables_dir)
    model = load_r3_checkpoint_v3(args.checkpoint, device="cpu")
    level = args.level

    records: list[dict[str, Any]] = []
    for entry in entries:
        cid = entry["curve_id"]
        node = int(entry["start_node_index"])
        xyz = ds.frame_xyz(cid, node)
        intent = ds.intents.get((cid, str(entry["forward_intent_id"])))
        if intent is None:
            raise SystemExit(f"forward intent missing for {cid}")
        sample0 = next(r for r in ds.samples if r.curve_id == cid)
        start = RolloutStartV3(
            curve_id=cid, node_index=node, intent=intent,
            atomic_numbers=sample0.atomic_numbers, xyz=xyz, d0=0.08)
        result = run_rollout_v3(
            arm="r3", start=start, n_steps=n_steps, model=model,
            level=level, device="cpu")
        records.append(result.to_dict())
        s = result.summary
        print(f"{cid} fold={entry['fold']} K={entry['k_size']} "
              f"steps={result.n_steps_executed} "
              f"mean_u_max={s['mean_u_max_atom']:.4f} "
              f"mean_clip={s['mean_clip_factor']:.4f} "
              f"d_key {s['d_key_first']} -> {s['d_key_last']} "
              f"sign_agree={s['cv_sign_agree_fraction']} "
              f"collisions={s['n_collisions']} nonfinite={s['n_nonfinite']}")

    doc = {
        "schema": "pes2ts_v53_rollout_smoke_v1",
        "arm": "r3",
        "level": level,
        "n_steps": n_steps,
        "device": "cpu",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": _sha256_file(Path(args.checkpoint)),
        "curve_list_sha256": curves_doc.get("curve_list_sha256", ""),
        "claim_scope": "TS_candidate_only",
        "physics_missing": PHYSICS_MISSING_DEFAULT,
        "records": records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"-> {out} ({_sys.executable})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sel = sub.add_parser("select-curves")
    p_sel.add_argument("--tables-dir", required=True)
    p_sel.add_argument("--out", required=True)
    p_sel.add_argument("--seed", type=int, default=SEED_V53)
    p_sel.set_defaults(func=cmd_select_curves)

    p_pf = sub.add_parser("preflight-relax")
    p_pf.add_argument("--tables-dir", required=True)
    p_pf.add_argument("--curves", required=True)
    p_pf.add_argument("--out", required=True)
    p_pf.add_argument("--curve-id", default=None)
    p_pf.set_defaults(func=cmd_preflight_relax)

    p_sm = sub.add_parser("smoke-rollout")
    p_sm.add_argument("--tables-dir", required=True)
    p_sm.add_argument("--curves", required=True)
    p_sm.add_argument("--checkpoint", required=True)
    p_sm.add_argument("--out", required=True)
    p_sm.add_argument("--n-steps", type=int, default=5)
    p_sm.add_argument("--level", default="F2",
                      help="feature level; MUST match the checkpoint arm")
    p_sm.add_argument("--curve-ids", default=None,
                      help="comma-separated frozen curve ids (default: first 2)")
    p_sm.set_defaults(func=cmd_smoke_rollout)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
