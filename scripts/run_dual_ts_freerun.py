"""WP4-B free-run CLI (plan v4 §G 自由运行门; G4 §6 BINDING, C6).

Rolls the SELECTED bc_nll checkpoint out from the 8 monitor groups'
endpoint states (branch A consistently), one branch per group:

* ``a = a_det(μ)`` primary (``--sampled`` = registered optional variant);
* ``X' = X + a`` — NO reference next-frame correction anywhere;
* features RECOMPUTED at consume time with the frozen centroid rule +
  frozen spec (reuse of ``build_state_features`` machinery);
* real xTB E/g per visited state via the ACP driver receipt machinery
  (content-addressed; separately billed ledger; pre-registered QC budget
  8 groups × ≤40 steps = ≤320 single points + HARD abort at cap);
* metrics: cumulative deviation vs the teacher path where they overlap
  (post-hoc, same frozen mass-weighted Kabsch registration), per-step
  geometry quality, guard-rail terminations, oracle arc (DIAGNOSTIC
  ONLY), real cost.

Example:
  python3 scripts/run_dual_ts_freerun.py   # full 8-group real campaign
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.models import require_torch  # noqa: E402
from pes2ts_core.pesn.offline.evaluation.dual_ts_freerun import (  # noqa: E402
    FREERUN_RUN_SCHEMA,
    FREERUN_VERSION,
    FreerunError,
    FreerunObserver,
    QCBudgetExhausted,
    RolloutSettings,
    emit_rollout_artifacts,
    load_frozen_scales,
    load_rollout_start,
    load_selected_policy,
    observation_settings_digest,
    post_hoc_truth_metrics,
    run_rollout,
    summarize_rollouts,
)
from pes2ts_core.utils.jsonio import write_json  # noqa: E402
from scripts.compute_dual_ts_observations import (  # noqa: E402
    _ensure_mpi_root_override,
    _pin_orca_mpi_launcher,
    load_acp_wiring,
)

BASE = PROJECT_ROOT / "outputs/pesn/dual_ts_v2_20261015"
DEFAULT_CHECKPOINT = BASE / (
    "wp4_bc/run_bc_bc_nll_seed20261017/checkpoint_last.pt")
DEFAULT_BUNDLE = BASE / "wp4_bc/run_bc_bc_nll_seed20261017/bundle.json"
DEFAULT_MONITOR_DIR = BASE / "wp4_monitor"
DEFAULT_FEATURES_DIR = DEFAULT_MONITOR_DIR / "features_v41"
DEFAULT_OUT = BASE / "wp4_freerun"
DEFAULT_ACP_CONFIG = PROJECT_ROOT / (
    "config/pesn/dual_ts_observations_16x1_v1.yaml")
SELECTION_RECORD = BASE / "wp4_bc/selection_record.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--monitor-dir", type=Path, default=DEFAULT_MONITOR_DIR,
                    help="WP4 monitor wave dir (teachers_observed/, "
                         "observations/ receipt store)")
    ap.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR,
                    help="monitor feature shard dir (v4.1 5-channel: "
                         "wp4_monitor/features_v41); the historical 7-channel "
                         "'features' dir is NOT silently used")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--qc-budget", type=int, default=320,
                    help="pre-registered cap on BILLED single points "
                         "(8 groups x <=40 steps); hard abort at cap")
    ap.add_argument("--groups", default="all8",
                    help="'all8' or a comma-separated subset of monitor "
                         "group ids (subset of the monitor wave)")
    ap.add_argument("--acp-config", type=Path, default=DEFAULT_ACP_CONFIG)
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--sampled", action="store_true",
                    help="optional sampled-action variant "
                         "(default primary: a = a_det(mu))")
    ap.add_argument("--device", default=None,
                    help="torch device (default: cuda:0 when available, "
                         "else cpu — recorded, never silent)")
    return ap.parse_args(argv)


def resolve_groups(features_manifest: dict, groups_arg: str) -> list[str]:
    wave = sorted(str(r["reaction_group_id"])
                  for r in features_manifest["per_reaction"])
    if groups_arg.strip().lower() == "all8":
        return wave
    requested = [g.strip() for g in groups_arg.split(",") if g.strip()]
    unknown = sorted(set(requested) - set(wave))
    if unknown or not requested:
        raise SystemExit(
            f"REFUSED: --groups {sorted(unknown or requested)} not in the "
            f"monitor wave {wave}")
    return sorted(requested)


def main() -> int:
    args = parse_args()
    torch = require_torch()
    if args.device is not None:
        device = args.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("[freerun] CUDA unavailable — running the policy forward "
                  "on cpu (recorded)", flush=True)

    features_dir = Path(args.features_dir)
    observed_dir = Path(args.monitor_dir) / "teachers_observed"
    monitor_receipts = Path(args.monitor_dir) / "observations"
    for label, path in (("features", features_dir),
                        ("teachers_observed", observed_dir)):
        if not path.is_dir():
            raise SystemExit(f"REFUSED: monitor {label} missing: {path}")

    scales = load_frozen_scales(features_dir)
    print(f"[freerun] frozen scales: s_g={scales['s_g']!r} Ha/A, "
          f"s_delta_e={scales['s_delta_e']!r} Ha, "
          f"spec digest {scales['spec_digest'][:12]}…", flush=True)

    selected_recipe = None
    if SELECTION_RECORD.is_file():
        sel = json.loads(SELECTION_RECORD.read_text())
        if sel.get("status") == "SELECTED":
            selected_recipe = sel.get("selected_recipe")
    policy, identity = load_selected_policy(
        torch, checkpoint_path=args.checkpoint, bundle_path=args.bundle,
        features_dir=features_dir, selected_recipe=selected_recipe)
    if device != "cpu":
        policy.to(device)
    print(f"[freerun] model identity: run_id={identity['run_id']} "
          f"recipe={identity['recipe']} ckpt={identity['checkpoint_sha256'][:12]}… "
          f"bundle={identity['bundle_content_digest'][:12]}… "
          f"action={identity['action_version']}", flush=True)

    manifest = json.loads(
        (features_dir / "features_manifest.json").read_text())
    groups = resolve_groups(manifest, args.groups)
    print(f"[freerun] groups ({len(groups)}): {groups}", flush=True)

    acp = load_acp_wiring(args.acp_config, retry=1)
    mpi_pin = _pin_orca_mpi_launcher(acp)
    root_override = _ensure_mpi_root_override()
    print(f"[freerun] ACP: root={acp['root']} method={acp['method']} "
          f"nproc={acp['nproc']} timeout={acp['timeout_seconds']} "
          f"mpi_pin={mpi_pin} root_override={root_override}", flush=True)

    settings = RolloutSettings(
        max_steps=int(args.max_steps), qc_budget=int(args.qc_budget),
        action_mode="sampled" if args.sampled else "deterministic",
        seed=int(args.seed))
    over_budget = len(groups) * settings.max_steps > settings.qc_budget
    print(f"[freerun] settings: {settings.to_payload()} "
          f"(budget covers campaign: {not over_budget})", flush=True)

    out_dir = Path(args.out)
    observer = FreerunObserver(
        out_root=out_dir / "observations", acp=acp,
        qc_budget=settings.qc_budget, fallback_roots=[monitor_receipts])
    acp_settings_digest = observation_settings_digest(acp)

    run_id = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    started = time.perf_counter()
    records = []
    campaign_stop = None
    for gi, gid in enumerate(groups):
        try:
            start = load_rollout_start(
                features_dir=features_dir, observed_dir=observed_dir,
                group_id=gid, branch_label=settings.branch_label,
                expected_spec_digest=scales["spec_digest"])
        except FreerunError as exc:
            print(f"[freerun] {gid}: START REFUSAL {exc}", flush=True)
            campaign_stop = f"start_refusal:{gid}:{exc.code}"
            break
        try:
            record = run_rollout(
                torch=torch, model=policy,
                observer=observer, start=start, scales=scales,
                identity=identity, settings=settings, group_index=gi)
        except QCBudgetExhausted as exc:
            print(f"[freerun] {gid}: {exc} — campaign hard abort", flush=True)
            campaign_stop = f"{gid}:{exc.code}"
            break
        record = post_hoc_truth_metrics(record, observed_dir=observed_dir)
        record_path, bundle_path, record_digest = emit_rollout_artifacts(
            out_dir=out_dir, record=record, identity=identity,
            scales=scales, acp_settings_digest=acp_settings_digest)
        records.append(record)
        post = record.get("post_hoc_truth_metrics", {})
        geo = [s["geometry_next"] for s in record["steps"]]
        min_dist = min((g["min_interatomic_distance_A"] for g in geo),
                       default=float("nan"))
        print(f"[freerun] {gid}: steps={record['n_steps']} "
              f"term={record['termination']['status']}"
              f"{':' + str(record['termination']['reason']) if record['termination']['reason'] else ''} "
              f"dev_final={post.get('deviation_final_A')} "
              f"dev_cum={post.get('cumulative_deviation_mean_A')} "
              f"min_dist={min_dist:.4f} "
              f"clip={np.mean([g['a_max_clip_rate'] for g in geo]):.4f} "
              f"oracle_arc={post.get('oracle_arc_start_A')}→{post.get('oracle_arc_end_A')} "
              f"billed={record['cost']['n_billed']} "
              f"reused={record['cost']['n_reused']} "
              f"digest={record_digest[:12]}…", flush=True)

    wall = time.perf_counter() - started
    aggregate = summarize_rollouts(records, qc_budget=settings.qc_budget)
    ledger_path = observer.write_ledger(
        run_id=run_id, wall_seconds=wall,
        extra={"groups_attempted": [
            r["group_id"] for r in records],
            "campaign_stop": campaign_stop})
    run_record = {
        "schema_version": FREERUN_RUN_SCHEMA,
        "freerun_version": FREERUN_VERSION,
        "run_id": run_id,
        "settings": settings.to_payload(),
        "identity": identity,
        "device": str(device),
        "groups_requested": groups,
        "groups_completed": [r["group_id"] for r in records],
        "campaign_stop": campaign_stop,
        "acp_settings_digest": acp_settings_digest,
        "observations_ledger": str(ledger_path),
        "observation_mode": settings.to_payload()["observation_mode"],
        "qc": aggregate["qc"],
        "aggregate": aggregate,
        "rollout_records": {
            r["group_id"]: str(out_dir / "rollouts" / (
                f"rollout_{r['group_id']}_branch"
                f"{str(settings.branch_label).lower()}.json"))
            for r in records},
        "wall_seconds": round(wall, 3),
    }
    write_json(out_dir / "freerun_record.json", run_record)

    print(f"[freerun] aggregate: groups={aggregate['n_groups']} "
          f"term={aggregate['termination_reasons']} "
          f"steps={aggregate['steps_min_median_max']} "
          f"dev_final_median={aggregate['deviation_final_median_A']} "
          f"min_dist_global={aggregate['min_interatomic_distance_global_min_A']} "
          f"clip_mean_max={aggregate['a_max_clip_rate_mean_max']} "
          f"qc={aggregate['qc']}", flush=True)
    print(f"[freerun] record: {out_dir / 'freerun_record.json'} "
          f"(wall {wall:.1f}s, ledger {ledger_path})", flush=True)
    if campaign_stop:
        print(f"[freerun] CAMPAIGN STOPPED: {campaign_stop}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FreerunError as exc:
        print(f"FREERUN REFUSAL {exc}", file=sys.stderr)
        raise SystemExit(2)
