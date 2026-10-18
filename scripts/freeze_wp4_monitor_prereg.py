"""Freeze the WP4 monitor8 pre-registration (+ thresholds) as content-
digested JSON artifacts (G4 C4, plan v4 §D WP4 row).

Stage 1 (``--stage prereg``) freezes the monitor8 protocol: population (the
8 lineage monitor-wave group ids, with live leakage assertions against the
fit partition / disputed-5 / sealed families), baselines (zero-action;
same-density global MLE on fit50 labels only), metrics (all 8 per-group
values + median + sign consistency; near-TS / moving / near-rest strata
with FROZEN fit-measured thresholds; disputed-5 sensitivity slice),
sampling/epoch rule (§I.10), fixed reduction, recipe formulas, λ=1.0s,
σ floor, a_max, budget, model-selection rule.

Stage 2 (``--stage thresholds``) freezes the QUANTITATIVE gate thresholds
and the C3 trigger constants AFTER the prereg exists (it embeds the prereg
sha256; the prereg file is never modified afterwards).

Artifacts (outputs/pesn/dual_ts_v2_20261015/wp4_bc/):
  monitor_prereg_v1.json       frozen BEFORE any monitor evaluation
  monitor_thresholds_v1.json   frozen after it, before any gate signing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.contracts.dual_ts import (  # noqa: E402
    RADIAL_TRANSFORM_VERSION,
)
from pes2ts_core.pesn.offline.training.dual_ts_bc import (  # noqa: E402
    DENSITY_TARGET_VERSION,
    LAMBDA_A,
    LAMBDA_DIR,
    N_ARC_STRATA,
    NLL_DIVERGENCE_LIMIT,
    RECIPES,
)
from pes2ts_core.pesn.offline.training.dual_ts_monitor import (  # noqa: E402
    C3_TRIGGER_JOINT_COS,
    C3_TRIGGER_NEAR_TS_COS,
    MONITOR_PREREG_SCHEMA,
    MONITOR_THRESHOLDS_SCHEMA,
    assert_fit_monitor_disjoint,
    sha256_file,
)

LINEAGE_PATH = PROJECT_ROOT / (
    "outputs/pesn/dual_ts_v2_20261015/splits/dual_ts_split_lineage_v1.json")
FEATURES_MANIFEST = PROJECT_ROOT / (
    "outputs/pesn/dual_ts_v2_20261015/wp3_features/features_manifest.json")
WP3_SUMMARY = PROJECT_ROOT / (
    "outputs/pesn/dual_ts_v2_20261015/wp3_model/summary_lam1.json")
OUT_DIR = PROJECT_ROOT / "outputs/pesn/dual_ts_v2_20261015/wp4_bc"
PREREG_PATH = OUT_DIR / "monitor_prereg_v1.json"
THRESHOLDS_PATH = OUT_DIR / "monitor_thresholds_v1.json"

#: Frozen budget (justified in the artifact; plan §I.9 "frozen budget in
#: updates + epochs"; WP3 learnability ran 2k updates with small margins).
BUDGET_UPDATES = 20000
BATCH_STATES = 64
LR = 1e-3
SEED = 20261017


def freeze_prereg() -> None:
    lineage = json.loads(LINEAGE_PATH.read_text())
    manifest = json.loads(FEATURES_MANIFEST.read_text())
    wp3 = json.loads(WP3_SUMMARY.read_text())
    monitor8 = list(lineage["dual_ts_wave_selections"]["wave"]["monitor"]
                    ["group_ids"])
    fit_ids = list(lineage["dual_ts_wave_selections"]["wave"]["fit"]
                   ["group_ids"])
    disputed = list(manifest["disputed_reactions"])
    evidence = assert_fit_monitor_disjoint(
        fit_group_ids=fit_ids, monitor_group_ids=monitor8,
        lineage=lineage, disputed_ids=disputed)
    moving_threshold = float(wp3["hyperparams_common"]["moving_threshold_A"])
    spec_digest = manifest["spec_digest"]
    fit_states = manifest["population"]["states"]
    fit_branches = manifest["population"]["branches"]
    payload = {
        "schema_version": MONITOR_PREREG_SCHEMA,
        "title": "WP4 monitor8 pre-registration (frozen before any monitor "
                 "evaluation of trained models; G4 C4)",
        "frozen_utc_source": "content digest of this exact file is the "
                             "identity; recorded in monitor_thresholds_v1"
                             ".json and every WP4 run record",
        "lineage": {
            "path": str(LINEAGE_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(LINEAGE_PATH),
            "wave_key": "dual_ts_wave_selections.wave.monitor",
        },
        "population": {
            "group_ids": monitor8,
            "n_groups": len(monitor8),
            "features_pipeline": "pesn_dual_ts_features_v2 (spec digest "
                                 f"{spec_digest}) — the SAME wp2/wp3 "
                                 "pipeline as fit50",
            "expected_shard_count": len(monitor8),
            "assertions": evidence,
            "disputed5_note": "the disputed-5 reactions are all in the fit "
                              "partition; the monitor8 disputed-5 slice is "
                              "therefore an empty-membership sensitivity "
                              "slice (recorded per group as n=0)",
            "per_group_state_counts": {
                "status": "pending_monitor_observation_reconstruction",
                "note": "exact state/branch counts become measurable only "
                        "when the monitor8 teacher payloads exist (WP2-style "
                        "xTB observation batch — separately billed QC, "
                        "§F); fit50 reference: 6067 states / 100 branches "
                        "over 50 groups",
            },
        },
        "baselines": {
            "zero_action": {
                "definition": "a = 0; per-atom MSE = |a*|^2; direction "
                              "cosine undefined (reported null)",
            },
            "same_density": {
                "definition": "state-independent radial-tanh Gaussian with "
                              "GLOBAL mu_bar/sigma_bar fit by MLE on the "
                              "fit50 teacher latents ONLY "
                              "(u* = R^-1(a*/a_max); the radial log-det is "
                              "a data constant, so the MLE is the plain "
                              "Gaussian moment match: mu_bar = mean(u*), "
                              "sigma_bar^2 = mean(|u*-mu_bar|^2)/3)",
                "evaluated_with": "identical radial-tanh transform and "
                                  "identical per-atom metric definitions on "
                                  "monitor groups (isolates state "
                                  "conditioning)",
                "fit_partition_only": True,
            },
        },
        "metrics": {
            "per_group_policy": "all 8 groups reported INDIVIDUALLY "
                                "(n=8: no pooled-only claims); aggregate = "
                                "median over groups + per-group sign "
                                "consistency counts",
            "values_per_population": [
                "n_atoms",
                "mse_model_A2 (per-atom, deterministic a_det(mu))",
                "mse_zero_baseline_A2",
                "mse_ratio_vs_zero",
                "direction_cosine_absa_weighted (|a|-w, direction mask)",
                "nll_model_per_atom (teacher-latent NLL at u*)",
                "nll_same_density_per_atom",
            ],
            "strata": {
                "near_ts": "remaining_arc_length <= 0.2 * branch max "
                           "(frozen per-branch rule, reused from fit)",
                "moving": "|a*| >= frozen fit-population q75 = "
                          f"{moving_threshold!r} A (REUSED from "
                          "wp3_model/summary_lam1.json — no re-measurement)",
                "near_rest": "frozen near-zero mask ||a*|| < 0.01 A "
                             "(excluded from L_dir only; still in MSE/NLL)",
            },
            "disputed5_sensitivity_slice": "per-group count of atoms from "
                                           "disputed-5 groups (expected 0 "
                                           "for monitor8; recorded anyway)",
            "sigma_summary": "per-group sigma vs |u*| (mean/median/q90/min "
                             "and near-rest medians) + registered "
                             "pathology flags: sigma_pinned_at_floor "
                             "(fraction at floor >= 0.9), "
                             "sigma_dominates_near_rest (near-rest median "
                             "sigma > 3x median |u*|)",
        },
        "sampling": {
            "epoch_definition": "every reaction group drawn exactly once; "
                                "draw order group -> branch -> "
                                "arc-stratified within branch "
                                "(n_strata = %d equal-count strata by "
                                "remaining-arc rank; near-TS stratum "
                                "emitted first; within-stratum seeded "
                                "permutation)" % N_ARC_STRATA,
            "per_group_draw_counts": "recorded per epoch in every run "
                                     "record",
            "batch_states": BATCH_STATES,
            "accounting": "updates AND sample exposure both reported "
                          "(plan §I.10)",
        },
        "reduction": "FIXED (plan v4 §I.10): per-step valid-atom sum -> "
                     "branch mean -> per-reaction mean over available "
                     "branches -> group mean over groups present in the "
                     "step; per-atom NLL reported separately, never mixed "
                     "into the main reduction",
        "recipes": {
            "density_target_version": DENSITY_TARGET_VERSION,
            "nll": "L_NLL = -log pi_z(u*|s) at u* = R^-1(a*/a_max); "
                   "per-atom NLL = 1.5*log2pi + 3*log(sigma) + "
                   "0.5|u*-mu|^2/sigma^2 + log_det(u*) (G4 C1)",
            "compositions": {
                "bc_nll": "L = L_NLL",
                "bc_a": f"L = L_NLL + {LAMBDA_A} * L_a "
                        "(L_a = |a_det(mu) - a*|^2)",
                "bc_d": f"L = L_NLL + {LAMBDA_A} * L_a + {LAMBDA_DIR} * "
                        "L_dir (L_dir = |a*| * (1 - cos(a_det, a*)) on "
                        "|a|-normalized directions, masked by the frozen "
                        "near-zero mask)",
            },
            "lambda_a": LAMBDA_A,
            "lambda_dir": LAMBDA_DIR,
            "no_lambda_1000": True,
            "no_annealing_schedules": "primary grid only (G4: registered "
                                      "rescue ablations if monitor shows "
                                      "sigma pathology)",
        },
        "action_contract": {
            "transform": RADIAL_TRANSFORM_VERSION,
            "a_max_angstrom": 0.1,
            "sigma_floor": 0.02,
            "sigma_formula": "sigma = sigma_min + softplus(raw)",
            "any_projection_revalidation_rule": "transforms => re-verified "
                                                "log-prob (§I.8)",
        },
        "budget": {
            "updates": BUDGET_UPDATES,
            "batch_states": BATCH_STATES,
            "lr": LR,
            "optimizer": "adam",
            "seed": SEED,
            "epochs_estimate": "fit50: 5967 drawable states / 64 = ~94 "
                               "updates per full epoch => ~213 epochs",
            "justification": "WP3 learnability evidence used 2k updates "
                             "with small margins; 10x budget for the "
                             "recipe comparison stays inside the fit50 "
                             "small-data regime and ~6 min per recipe on "
                             "the RTX 4060 Ti; frozen before any fit",
            "fresh_init": True,
            "no_warm_start": "never warm-started from WP3 checkpoints "
                             "(isolation condition)",
        },
        "model_selection": {
            "primary_rule": "select on monitor8 metrics, NEVER min NLL "
                            "(plan §I.9; frozen quantitatively in "
                            "monitor_thresholds_v1.json AFTER this "
                            "pre-registration)",
            "population": "joint (both branches), all 8 groups",
        },
        "c3_signed_ablation_arm": {
            "default": "OFF for all three primary recipes",
            "registered_arm": "signed_mu_gate=true, joint population, best "
                              "primary recipe, same budget",
            "trigger": f"joint monitor |a|-w cosine < {C3_TRIGGER_JOINT_COS} "
                       f"OR near-TS cosine < {C3_TRIGGER_NEAR_TS_COS} at "
                       "budget end — monitor evidence only",
        },
        "monitor_data_status": {
            "at_freeze_time": "monitor8 xTB observations do not exist; "
                              "their computation is separately billed QC "
                              "and NOT part of this lane; the evaluation "
                              "code refuses to run (MONITOR_DATA_MISSING) "
                              "until the monitor8 teacher payloads + "
                              "features exist",
            "gate": "evaluation refuses unless prereg digest matches AND "
                    "the 8 monitor shards exist and digest-match",
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if PREREG_PATH.exists():
        raise SystemExit(
            f"REFUSING: {PREREG_PATH} already exists — the pre-registration "
            f"is frozen and must never be rewritten (freeze new versions "
            f"under a new filename)")
    PREREG_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"prereg frozen: {PREREG_PATH}")
    print(f"prereg sha256: {sha256_file(PREREG_PATH)}")


def freeze_thresholds() -> None:
    if not PREREG_PATH.exists():
        raise SystemExit(
            "prereg missing — quantitative thresholds can only be frozen "
            "AFTER the pre-registration exists (C4 order)")
    prereg_digest = sha256_file(PREREG_PATH)
    payload = {
        "schema_version": MONITOR_THRESHOLDS_SCHEMA,
        "title": "WP4 monitor8 quantitative gate thresholds — frozen AFTER "
                 "the pre-registration, BEFORE any gate signing (C4)",
        "prereg_file": PREREG_PATH.name,
        "prereg_sha256": prereg_digest,
        "gate": {
            "median_mse_ratio_max": 1.0,
            "n_groups_ratio_below_one_min": 6,
            "median_cosine_min": 0.30,
            "near_ts_cosine_min": 0.40,
            "n_groups": 8,
            "decision": "BC gate PASS iff median per-group MSE ratio vs "
                        "zero-action < 1.0 AND >= 6/8 groups individually "
                        "below 1.0 AND median |a|-w cosine >= 0.30 AND "
                        "median near-TS cosine >= 0.40; else REGISTER_"
                        "ABLATION (never silent scale-up)",
        },
        "selection_rule": {
            "eligibility_median_cosine_min": 0.0,
            "eligibility_n_cosine_positive_min": 6,
            "metric": "median per-group MSE ratio vs zero-action baseline "
                      "(ascending; monitor8 only, never NLL)",
            "tie_band": 0.01,
            "simplicity_order": ["bc_nll", "bc_a", "bc_d"],
            "exclusions": "recipes with a sigma-pathology flag on the "
                          "monitor summary are ineligible",
        },
        "c3_trigger": {
            "joint_cosine_below": C3_TRIGGER_JOINT_COS,
            "near_ts_cosine_below": C3_TRIGGER_NEAR_TS_COS,
            "evidence": "monitor8 at budget end ONLY",
            "arm_if_fired": "signed_mu_gate=true, joint population, best "
                            "primary recipe, same budget",
        },
    }
    if THRESHOLDS_PATH.exists():
        raise SystemExit(
            f"REFUSING: {THRESHOLDS_PATH} already exists — thresholds are "
            f"frozen once")
    THRESHOLDS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"thresholds frozen: {THRESHOLDS_PATH}")
    print(f"thresholds sha256: {sha256_file(THRESHOLDS_PATH)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("prereg", "thresholds"),
                    required=True)
    args = ap.parse_args()
    if args.stage == "prereg":
        freeze_prereg()
    else:
        freeze_thresholds()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
