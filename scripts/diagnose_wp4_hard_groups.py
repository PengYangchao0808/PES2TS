"""G5 zero-QC diagnosis slice (analysis ONLY — no training, no new data).

HISTORICAL v4 G5 artifact: this script computes the 7-CHANNEL v2 metric
(reads ``wp3_features`` + ``wp4_monitor/features``, both 7-channel v2
shards).  The v4.1 clean 5-channel recompute lives in the new
``diagnose_v41`` script (WP3) — DO NOT point this script at v41 dirs.

Characterizes the 3 unresponsive monitor8 groups (RXN_0000004669/6957/10293)
against the 5 responsive ones using ONLY artifacts already on disk:

  (a) feature-space distance of each monitor group to its NEAREST fit group
      (wp3_features fit shards + wp4_monitor/features monitor shards).
      Documented deterministic metric ("mean node-block L2 after length
      normalization"): per state, average the 7-channel ``node_scalars``
      block over atoms (length normalization across molecule sizes); a
      group's PROFILE is the mean of those per-state atom-mean vectors over
      all its states; distance between two groups = L2 between profiles.
      A sensitive secondary variant averages, per monitor state, the L2 to
      the nearest fit-group profile (per-state resolution, same block).
  (b) per-group geometry/arc profiles from the teacher payloads (both waves):
      atom counts, branch lengths, arc-length distribution, max |a*|,
      near-TS step sizes (arc progress AND Cartesian per-atom RMS step
      inside the frozen near-TS stratum: remaining_arc <= 0.2 * branch max).
  (c) free-run divergence / energy drift for RXN_0000006957 (and all-group
      context) from the wp4_freerun rollout records.
  (d) a DIVERSITY-gap vs ARCHITECTURE/objective summary.  NOTE: no
      threshold for this slice was preregistered; the verdict is a
      descriptive comparison of the (a) distributions, reported honestly
      either way.

Writes outputs/pesn/dual_ts_v2_20261015/wp4_bc/g5_diagnosis_slice.json
and prints a short human-readable digest.  Deterministic: sorting only,
no RNG, float rounding recorded verbatim.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = PROJECT_ROOT / "outputs/pesn/dual_ts_v2_20261015"
FIT_FEATURES = BASE / "wp3_features"
FIT_OBSERVED = BASE / "wp2_fit50_observed"
MON_FEATURES = BASE / "wp4_monitor/features"
MON_OBSERVED = BASE / "wp4_monitor/teachers_observed"
FREERUN_ROLLOUTS = BASE / "wp4_freerun/rollouts"
OUT_PATH = BASE / "wp4_bc/g5_diagnosis_slice.json"

UNRESPONSIVE = ["RXN_0000004669", "RXN_0000006957", "RXN_0000010293"]
RESPONSIVE = ["RXN_0000000798", "RXN_0000001178", "RXN_0000002770",
              "RXN_0000006274", "RXN_0000009014"]
NEAR_TS_ARC_FRACTION = 0.2   # frozen monitor prereg stratum rule, reused


def group_ids_of(features_dir: Path) -> list[str]:
    return sorted(p.stem.removeprefix("features_")
                  for p in features_dir.glob("features_*.npz"))


def load_shard(features_dir: Path, group_id: str) -> dict:
    z = np.load(features_dir / f"features_{group_id}.npz",
                allow_pickle=False)
    return {k: z[k] for k in z.files}


def group_feature_profile(shard: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (profile(7,), per_state_atom_mean(n_states,7)).

    Length normalization: mean over atoms per state (states of the same
    group share n_atoms, but the mean keeps the metric well-defined for
    mixed-size comparisons across groups).

    HISTORICAL v4 artifact: operates on the 7-CHANNEL v2 ``node_scalars``
    block (this script must not be pointed at v41 5-channel shards).
    """
    ns = np.asarray(shard["node_scalars"], dtype=np.float64)  # (S,N,7)
    per_state = ns.mean(axis=1)                               # (S,7)
    return per_state.mean(axis=0), per_state


def nearest_fit_distance(
    profile: np.ndarray,
    per_state: np.ndarray,
    fit_profiles: dict[str, np.ndarray],
) -> dict:
    """Primary: profile L2 to nearest fit group.

    Secondary: mean over this group's states of the L2 from the state's
    atom-mean vector to the NEAREST fit-group profile (per-state view of
    the same metric; uses the same fit profiles, no re-fitting).
    """
    names = sorted(fit_profiles)
    prof_d = {n: float(np.linalg.norm(profile - fit_profiles[n]))
              for n in names}
    nearest_name = min(prof_d, key=lambda n: (prof_d[n], n))
    state_d = np.stack([
        np.linalg.norm(per_state - fit_profiles[n], axis=1) for n in names
    ], axis=0)                                              # (n_fit, S)
    nearest_idx = state_d.argmin(axis=0)
    per_state_nearest = state_d.min(axis=0)
    nearest_fit = names[int(nearest_idx[0])]
    counts: dict[str, int] = {}
    for i in nearest_idx:
        counts[names[i]] = counts.get(names[i], 0) + 1
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    return {
        "nearest_fit_group": nearest_fit,
        "profile_l2_to_nearest_fit": min(prof_d.values()),
        "profile_l2_per_fit_group": prof_d,
        "mean_per_state_l2_to_nearest_fit_profile":
            float(per_state_nearest.mean()),
        "max_per_state_l2_to_nearest_fit_profile":
            float(per_state_nearest.max()),
        "nearest_fit_group_top3_by_state_count": [
            {"group": n, "states": c} for n, c in top],
        # per-channel |Δprofile| to the nearest fit group — WHICH of the
        # 7 node-scalar channels drives the distance (interpretability
        # only; not a second metric).
        "per_channel_abs_diff_to_nearest_fit": [
            float(v) for v in
            np.abs(profile - fit_profiles[nearest_name])],
    }


def geometry_arc_profile(teacher: dict) -> dict:
    """(b) per-group geometry/arc profile from the teacher payload."""
    states = teacher["states"]
    labels = {l["state_id"]: l for l in teacher["labels"]}
    n_atoms = len(states[0]["atomic_numbers"])
    branches: dict[str, list[dict]] = {}
    for st in states:
        branches.setdefault(st["branch_id"], []).append(st)
    branch_lengths = {b: len(v) for b, v in sorted(branches.items())}

    # remaining-arc distribution over NON-teacher-endpoint states (the
    # drawable population; endpoints carry no action and their remaining
    # arc is 0 by construction of the path build).
    rem = [float(labels[s["state_id"]]["remaining_arc_length"])
           for s in states
           if not bool(labels[s["state_id"]]["teacher_endpoint"])]
    rem_arr = np.asarray(rem)
    max_a = 0.0
    for lab in teacher["labels"]:
        if bool(lab["teacher_endpoint"]):
            continue
        a = np.linalg.norm(np.asarray(lab["action_delta"],
                                      dtype=np.float64), axis=-1)
        max_a = max(max_a, float(a.max()))

    # per-branch near-TS stratum (frozen rule: remaining_arc <= 0.2 * the
    # branch's max remaining arc = its step-0 arc) -> step sizes.
    near_ts_arc_steps: list[float] = []
    near_ts_cart_rms: list[float] = []
    for b, sts in sorted(branches.items()):
        sts = sorted(sts, key=lambda s: s["step_index"])
        branch_arc = [float(labels[s["state_id"]]["remaining_arc_length"])
                      for s in sts]
        branch_max = max(branch_arc)
        for i in range(len(sts) - 1):
            if not bool(labels[sts[i]["state_id"]]["teacher_endpoint"]) \
                    and branch_arc[i] <= NEAR_TS_ARC_FRACTION * branch_max:
                near_ts_arc_steps.append(branch_arc[i] - branch_arc[i + 1])
                xi = np.asarray(sts[i]["coordinates"], dtype=np.float64)
                xj = np.asarray(sts[i + 1]["coordinates"],
                                dtype=np.float64)
                step = np.linalg.norm(xj - xi, axis=-1)
                near_ts_cart_rms.append(
                    float(np.sqrt((step ** 2).mean())))
    total_arcs = sorted(branch_arc0 for branch_arc0 in (
        max(float(labels[s["state_id"]]["remaining_arc_length"])
            for s in sts) for sts in branches.values()))
    nt_arc = np.asarray(near_ts_arc_steps)
    nt_cart = np.asarray(near_ts_cart_rms) if near_ts_cart_rms else None
    return {
        "n_atoms": int(n_atoms),
        "n_states": len(states),
        "n_action_states": len(rem),
        "n_branches": len(branches),
        "branch_lengths_states": branch_lengths,
        "branch_total_arc_A": {
            "min": float(total_arcs[0]),
            "median": float(np.median(total_arcs)),
            "max": float(total_arcs[-1]),
        },
        "remaining_arc_A": {
            "min": float(rem_arr.min()),
            "median": float(np.median(rem_arr)),
            "max": float(rem_arr.max()),
        },
        "max_action_norm_A": float(max_a),
        "near_ts_stratum": {
            "n_steps": len(near_ts_arc_steps),
            "arc_progress_per_step_A": {
                "median": float(np.median(nt_arc)) if nt_arc.size else None,
                "max": float(nt_arc.max()) if nt_arc.size else None,
            },
            "cartesian_rms_step_A": {
                "median": float(np.median(nt_cart)) if nt_cart is not None
                else None,
                "max": float(nt_cart.max()) if nt_cart is not None else None,
            },
        },
    }


def freerun_profile(rollout_path: Path) -> dict:
    r = json.loads(rollout_path.read_text())
    steps = r["steps"]
    de = [float(s.get("delta_e_hartree", float("nan"))) for s in steps]
    de0 = next((v for v in de if not math.isnan(v)), float("nan"))
    de_last = de[-1] if de else float("nan")
    min_d = min(float(s["geometry_next"]["min_interatomic_distance_A"])
                for s in steps)
    brk = max(float(s["geometry_next"]["broken_fraction"]) for s in steps)
    thm = r["post_hoc_truth_metrics"]
    return {
        "n_steps": int(r["n_steps"]),
        "termination": r["termination"]["status"],
        "deviation_final_A": float(thm["deviation_final_A"]),
        "cumulative_deviation_mean_A":
            float(thm["cumulative_deviation_mean_A"]),
        "oracle_arc_start_end_A": [float(thm["oracle_arc_start_A"]),
                                   float(thm["oracle_arc_end_A"])],
        "energy_drift_hartree": de_last - de0,
        "energy_first_hartree": de0,
        "energy_last_hartree": de_last,
        "min_interatomic_distance_A": min_d,
        "max_broken_bond_fraction": brk,
        "a_max_clip_rate_max": max(
            float(s["geometry_next"]["a_max_clip_rate"]) for s in steps),
    }


def main() -> int:
    fit_groups = group_ids_of(FIT_FEATURES)
    mon_groups = group_ids_of(MON_FEATURES)
    if sorted(RESPONSIVE + UNRESPONSIVE) != mon_groups:
        raise SystemExit(
            f"REFUSED: monitor groups on disk {mon_groups} != the 8 "
            f"preregistered groups")
    fit_profiles, fit_per_state = {}, {}
    for g in fit_groups:
        shard = load_shard(FIT_FEATURES, g)
        fit_profiles[g], fit_per_state[g] = group_feature_profile(shard)

    feature_distances, geometry = {}, {}
    for g in mon_groups:
        shard = load_shard(MON_FEATURES, g)
        prof, per_state = group_feature_profile(shard)
        feature_distances[g] = nearest_fit_distance(prof, per_state,
                                                    fit_profiles)
    for g in mon_groups:
        obs_dir = MON_OBSERVED
        cand = obs_dir / f"{g}.teacher.json"
        if not cand.is_file():
            cand = FIT_OBSERVED / f"{g}.teacher.json"
        geometry[g] = geometry_arc_profile(json.loads(cand.read_text()))

    freerun = {p.stem.removeprefix("rollout_").removesuffix("_brancha"):
               freerun_profile(p)
               for p in sorted(FREERUN_ROLLOUTS.glob("rollout_*_brancha.json"))
               if p.suffix == ".json"
               and not p.name.endswith(".bundle.json")}

    dead = [feature_distances[g]["profile_l2_to_nearest_fit"]
            for g in UNRESPONSIVE]
    resp = [feature_distances[g]["profile_l2_to_nearest_fit"]
            for g in RESPONSIVE]
    dead_med, resp_med = float(np.median(dead)), float(np.median(resp))
    ratio = dead_med / resp_med if resp_med > 0 else float("inf")
    if min(dead) > max(resp):
        verdict = "diversity_gap"
    elif ratio >= 2.0:
        verdict = "diversity_gap"
    elif max(dead) <= max(resp) and ratio <= 1.5:
        verdict = "architecture_or_objective"
    else:
        verdict = "inconclusive_mixed"

    record = {
        "schema_version": "pesn_wp4_g5_diagnosis_slice_v1",
        "title": "G5 zero-QC diagnosis slice: unresponsive vs responsive "
                 "monitor groups (analysis only, 0 new QC)",
        "unresponsive_groups": UNRESPONSIVE,
        "responsive_groups": RESPONSIVE,
        "metric_definition": {
            "primary": "group profile = mean over states of the per-state "
                       "atom-mean of the 7-channel node_scalars block "
                       "(length normalization); distance = L2 between "
                       "group profiles; nearest = min over the 50 fit "
                       "groups",
            "secondary": "mean over the group's states of the L2 from the "
                         "state atom-mean vector to the nearest fit-group "
                         "profile",
            "deterministic": "sorted iteration, no RNG, no sampling",
        },
        "feature_distances_to_nearest_fit_group": feature_distances,
        "geometry_arc_profiles": geometry,
        "freerun_profiles": freerun,
        "verdict": {
            "dead_profile_l2_values": dead,
            "responsive_profile_l2_values": resp,
            "dead_median": dead_med,
            "responsive_median": resp_med,
            "dead_over_responsive_median_ratio": ratio,
            "rule": "diversity_gap if min(dead) > max(responsive) or "
                    "median ratio >= 2.0; architecture_or_objective if "
                    "distributions overlap (max(dead) <= max(responsive) "
                    "and ratio <= 1.5); else inconclusive_mixed. NO "
                    "threshold for this slice was preregistered — "
                    "descriptive only.",
            "evidence_favors": verdict,
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(record, indent=2) + "\n")
    print(f"[g5-diagnosis] wrote {OUT_PATH}")
    for g in mon_groups:
        fd = feature_distances[g]
        tag = "DEAD" if g in UNRESPONSIVE else "resp"
        print(f"  {g} [{tag}] nearest_fit={fd['nearest_fit_group']} "
              f"profile_l2={fd['profile_l2_to_nearest_fit']:.6f} "
              f"state_mean_l2="
              f"{fd['mean_per_state_l2_to_nearest_fit_profile']:.6f}")
    print(f"  verdict: {verdict} (dead median {dead_med:.6f} vs responsive "
          f"median {resp_med:.6f}, ratio {ratio:.3f})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover
        print(f"DIAGNOSIS REFUSAL {exc!r}", file=sys.stderr)
        raise SystemExit(2)
