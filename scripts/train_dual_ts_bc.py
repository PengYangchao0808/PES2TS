"""WP4-A dual-TS BC training CLI (plan v4 §D WP4; G4 C1–C3).

Trains ONE recipe (``--recipe {bc_nll,bc_a,bc_d}``) on the fit50 partition
with the frozen §I.10 sampler/reduction and the C1 teacher-latent NLL,
writes the per-run record + validated ``pesn_dual_ts_bundle_v2`` bundle +
full-identity checkpoint, then ATTEMPTS the prereg-gated monitor8
evaluation and updates the selection record.

The monitor attempt is gated (C4): it refuses unless (i) the frozen
pre-registration exists and its digest matches the one bound into this
run, (ii) the frozen thresholds file exists and references that digest,
and (iii) the monitor8 shards exist and digest-match.  Today (iii) is
false — monitor8 xTB observations are separately billed QC and do not
exist — so the attempt records an honest ``MONITOR_DATA_MISSING`` refusal
and the run still completes (training-side artifacts are unaffected).

Example:
  python3 scripts/train_dual_ts_bc.py --recipe bc_nll \
      --stage short --seed 20261017 \
      --out outputs/pesn/dual_ts_v2_20261015/wp41_bc
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.contracts.dual_ts import (  # noqa: E402
    validate_record,
)
from pes2ts_core.pesn.models import require_torch  # noqa: E402
from pes2ts_core.pesn.models.dual_egnn import (  # noqa: E402
    DualTSModelConfig,
)
from pes2ts_core.pesn.models.dual_ts_policy import (  # noqa: E402
    DEFAULT_A_MAX_ANGSTROM,
    DEFAULT_SIGMA_RANGE,
    DualTSPolicy,
)
from pes2ts_core.pesn.offline.training.dual_ts_bc import (  # noqa: E402
    NEAR_TS_ARC_FRACTION,
    RECIPES,
    DualTSBCSettings,
    DualTSBCTrainer,
    DualTSBCTrainerError,
    load_dual_ts_states,
    load_spec_and_manifest,
    resolve_lambda_registration,
    sha256_file,
)
from pes2ts_core.pesn.offline.training.dual_ts_monitor import (  # noqa: E402
    MONITOR_THRESHOLDS_SCHEMA,
    MonitorGateError,
    apply_selection_rule,
    assert_fit_monitor_disjoint,
    c3_trigger_evaluation,
    evaluate_groups_with_policy,
    fit_same_density_baseline_mle,
    gate_prereg,
    load_monitor_states,
    load_prereg,
    summarize_over_groups,
)
from pes2ts_core.utils.jsonio import write_json  # noqa: E402

# ADR-0022 §3: the v4.1 observation-only 5-channel features live in the v41
# directories (declared migration; the old v2 dirs remain untouched as
# historical artifacts).  Feature/observed/monitor defaults point here.
FEATURES_DIR = PROJECT_ROOT / (
    "outputs/pesn/dual_ts_v2_20261015/wp3_features_v41")
OBSERVED_DIR = PROJECT_ROOT / (
    "outputs/pesn/dual_ts_v2_20261015/wp2_fit50_observed")
MONITOR_FEATURES_DIR = PROJECT_ROOT / (
    "outputs/pesn/dual_ts_v2_20261015/wp4_monitor/features_v41")
MONITOR_OBSERVED_DIR = PROJECT_ROOT / (
    "outputs/pesn/dual_ts_v2_20261015/wp4_monitor/teachers_observed")
# Finding 2 (F3): a FRESH v4.1 output namespace — never the historical wp4_bc
# (which holds immutable v4 artifacts + the frozen protocol + thresholds).
DEFAULT_OUT = PROJECT_ROOT / "outputs/pesn/dual_ts_v2_20261015/wp41_bc"
WP4_BC_DIR = PROJECT_ROOT / "outputs/pesn/dual_ts_v2_20261015/wp4_bc"
DEFAULT_PREREG = WP4_BC_DIR / "monitor_prereg_v1.json"
DEFAULT_THRESHOLDS = WP4_BC_DIR / "monitor_thresholds_v1.json"
# Operative protocol: r0 (protocol_v41.json) is immutable; the operative
# registration (A/B/C/D mapping + matrix + fit50 set) lives in the r1
# revision chained to r0 by parent_freeze_sha256.
DEFAULT_PROTOCOL = WP4_BC_DIR / "protocol_v41_r1.json"

#: v4.1 BC run record schema (the ONLY schema eligible for formal selection;
#: historical v4 ``pesn_dual_ts_bc_run_v1`` records are ignored).
V41_RUN_SCHEMA = "pesn_dual_ts_bc_run_v2"
FORMAL_STAGES = ("short", "full")
CANARY_STAGES = ("fit1", "fit5")
SELECTION_RECORD_NAME = "selection_record_v41.json"


# ---------------------------------------------------------------------------
# Finding 4: protocol verification on read + revision-chain check
# ---------------------------------------------------------------------------

def compute_protocol_content_digest(payload: dict) -> str:
    """sha256 over the protocol content EXCLUDING the ``freeze`` block — the
    SAME rule the freeze tool uses (``scripts/freeze_v41_protocol.py``)."""
    core = {k: v for k, v in payload.items() if k != "freeze"}
    text = json.dumps(core, sort_keys=True, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _has_formal_amendment_contract(payload: dict) -> bool:
    """The r1+ amendment contract a FORMAL stage requires (finding 3).

    r0 (``protocol_v41.json``) carries no ``revision``/``arm_mapping``/
    ``matrix_hyperparams``/``fit50_group_ids`` and is therefore NOT a formal
    protocol — it stays usable for canaries/diagnostics only.
    """
    revision = payload.get("revision") or {}
    return (int(revision.get("n", 0)) >= 1
            and bool(payload.get("arm_mapping"))
            and bool(payload.get("matrix_hyperparams"))
            and bool(payload.get("fit50_group_ids")))


def load_verified_protocol(path: Path, *,
                           require_formal: bool = False) -> tuple[dict, str]:
    """Load + verify a protocol file; returns (payload, verified_digest).

    Typed refusal (SystemExit) if: missing / no ``freeze.sha256`` / content
    digest mismatch (tampered) / revision-chain broken.  The revision chain is
    verified TRANSITIVELY (r_n → … → r0, each parent link's freeze sha AND
    content digest recomputed — finding 2).  With ``require_formal=True`` a
    payload missing the r1+ amendment contract is refused as r0 (finding 3).
    """
    if not path.is_file():
        raise SystemExit(
            f"REFUSED: protocol file missing — {path} (a verified "
            f"protocol_v41[_rN].json is required)")
    payload = json.loads(path.read_text())
    recorded = str((payload.get("freeze") or {}).get("sha256", ""))
    recomputed = compute_protocol_content_digest(payload)
    if not recorded:
        raise SystemExit(
            f"REFUSED: protocol {path} has no freeze.sha256 (not frozen)")
    if recorded != recomputed:
        raise SystemExit(
            f"REFUSED: protocol {path} content digest mismatch — recorded "
            f"{recorded[:16]}… != recomputed {recomputed[:16]}… (tampered or "
            f"edited after freeze)")
    # -- transitive revision-chain verification (finding 2) ------------------
    current_path = path
    current = payload
    depth = 0
    while True:
        revision = current.get("revision")
        if not revision:
            break
        depth += 1
        if depth > 64:
            raise SystemExit(
                f"REFUSED: revision chain too deep at {current_path} "
                f"(possible cycle)")
        parent_file = str(revision.get("parent_file", "protocol_v41.json"))
        parent_path = current_path.parent / parent_file
        parent_sha = str(revision.get("parent_freeze_sha256", ""))
        if not parent_sha:
            raise SystemExit(
                f"REFUSED: revision protocol {current_path} has no "
                f"parent_freeze_sha256")
        if not parent_path.is_file():
            raise SystemExit(
                f"REFUSED: cannot verify revision chain — parent protocol "
                f"{parent_path} missing")
        parent_payload = json.loads(parent_path.read_text())
        parent_recorded = str(
            (parent_payload.get("freeze") or {}).get("sha256", ""))
        if parent_recorded != parent_sha:
            raise SystemExit(
                f"REFUSED: revision chain broken — {current_path} "
                f"parent_freeze_sha256 {parent_sha[:16]}… != {parent_path} "
                f"freeze.sha256 {parent_recorded[:16]}…")
        parent_recomputed = compute_protocol_content_digest(parent_payload)
        if parent_recorded != parent_recomputed:
            raise SystemExit(
                f"REFUSED: parent protocol {parent_path} content digest "
                f"mismatch (tampered after freeze)")
        current_path = parent_path
        current = parent_payload
    # -- formal amendment-contract gate (finding 3) --------------------------
    if require_formal and not _has_formal_amendment_contract(payload):
        raise SystemExit(
            f"REFUSED: protocol {path} is r0 (no revision/amendment contract) "
            f"— r0 is usable for canaries/diagnostics only; formal stages "
            f"(short/full) require a revision protocol (r1+) with "
            f"arm_mapping + matrix_hyperparams + fit50_group_ids")
    return payload, recomputed


# ---------------------------------------------------------------------------
# Finding 2: run-dir identity + overwrite refusal + settings digest
# ---------------------------------------------------------------------------

def _dir_float(x: float) -> str:
    """Compact deterministic float for a run-dir suffix."""
    return f"{float(x):g}"


def build_run_dir_name(*, recipe: str, seed: int, variant: str, stage: str,
                       lambda_a: float, lambda_dir: float,
                       diagnostic_run: bool) -> str:
    """Finding 2 (F3): stage/λ/diagnostic are PART of the run identity."""
    name = f"run_bc_{recipe}_seed{seed}{variant}_stage-{stage}"
    if float(lambda_a) != 1.0 or float(lambda_dir) != 1.0:
        name += f"_lambda{_dir_float(lambda_a)}_{_dir_float(lambda_dir)}"
    if diagnostic_run:
        name += "_diag"
    return name


def run_settings_digest(*, recipe, seed, stage, lambda_a, lambda_dir,
                        features_spec_digest, protocol_sha256,
                        budget_updates=0, lr=0.0, batch_states=0,
                        signed_mu_gate=False, signed_vector_messages=False,
                        ) -> str:
    core = {
        "recipe": str(recipe), "seed": int(seed), "stage": str(stage),
        "lambda_a": float(lambda_a), "lambda_dir": float(lambda_dir),
        "features_spec_digest": str(features_spec_digest),
        "protocol_sha256": str(protocol_sha256),
        "budget_updates": int(budget_updates),
        "lr": float(lr),
        "batch_states": int(batch_states),
        "signed_mu_gate": bool(signed_mu_gate),
        "signed_vector_messages": bool(signed_vector_messages),
    }
    text = json.dumps(core, sort_keys=True, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record_settings_digest(rec: dict) -> str:
    st = rec.get("settings") or {}
    data = rec.get("data") or {}
    return run_settings_digest(
        recipe=str(st.get("recipe", "")),
        seed=int(st.get("seed", 0)),
        stage=str(st.get("stage", "")),
        lambda_a=float(st.get("lambda_a", 1.0)),
        lambda_dir=float(st.get("lambda_dir", 1.0)),
        features_spec_digest=str(data.get("spec_digest", "")),
        protocol_sha256=str(st.get("protocol_sha256", "")),
        budget_updates=int(st.get("budget_updates", 0)),
        lr=float(st.get("lr", 0.0)),
        batch_states=int(st.get("batch_states", 0)),
        signed_mu_gate=bool(st.get("signed_mu_gate", False)),
        signed_vector_messages=bool(st.get("signed_vector_messages", False)),
    )


def load_policy_from_checkpoint(torch, ckpt_path: Path) -> DualTSPolicy:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = payload["extra"]
    policy = DualTSPolicy(
        config=DualTSModelConfig(
            light=bool(cfg["settings"]["light"]),
            signed_vector_messages=bool(
                cfg["signed_vector_messages"])),
        action_max=float(cfg["settings"]["action_max"]),
        sigma_range=(float(cfg["settings"]["sigma_min"]),
                     float(cfg["settings"]["sigma_max"])),
        signed_mu_gate=bool(cfg["signed_mu_gate"]),
    )
    policy.load_state_dicts(payload["model_state"])
    return policy


def attempt_monitor_evaluation(
    *,
    torch,
    recipe: str,
    run_dir: Path,
    ckpt_path: Path,
    prereg_sha256: str,
    thresholds_path: Path,
    moving_threshold: float,
    near_ts_arc_fraction: float,
    fit_states: list,
    same_density_cache: dict,
    prereg_path: Path = DEFAULT_PREREG,
    monitor_features_dir: Path = MONITOR_FEATURES_DIR,
    monitor_observed_dir: Path = MONITOR_OBSERVED_DIR,
    fit_features_dir: Path = FEATURES_DIR,
) -> dict:
    """Prereg-gated monitor8 evaluation attempt; NEVER silently skipped."""
    record: dict = {"attempted": True}
    try:
        prereg, digest = gate_prereg(prereg_path, prereg_sha256)
        record["prereg_sha256"] = digest
        if not thresholds_path.is_file():
            raise MonitorGateError(
                "THRESHOLDS_MISSING",
                f"frozen thresholds missing: {thresholds_path}")
        thresholds = json.loads(thresholds_path.read_text())
        if thresholds.get("schema_version") != MONITOR_THRESHOLDS_SCHEMA:
            raise MonitorGateError(
                "THRESHOLDS_SCHEMA_UNKNOWN",
                str(thresholds.get("schema_version")))
        if thresholds.get("prereg_sha256") != digest:
            raise MonitorGateError(
                "THRESHOLDS_PREREG_MISMATCH",
                "thresholds file references a different prereg digest")
        group_ids = prereg["population"]["group_ids"]
        # Frozen-spec reuse gate: the monitor feature manifest must declare
        # the SAME spec digest as the fit build (C5: scales never
        # re-measured on monitor; the whole point of --spec reuse).
        fit_manifest = json.loads(
            (fit_features_dir / "features_manifest.json").read_text())
        mon_manifest = json.loads(
            (Path(monitor_features_dir) / "features_manifest.json").read_text())
        if str(mon_manifest.get("spec_digest")) != \
                str(fit_manifest.get("spec_digest")):
            raise MonitorGateError(
                "SPEC_REUSE_VIOLATION",
                f"monitor spec digest {mon_manifest.get('spec_digest')} != "
                f"fit spec digest {fit_manifest.get('spec_digest')} — the "
                f"frozen scales must be REUSED, never re-measured")
        record["spec_reuse"] = {
            "spec_digest": fit_manifest["spec_digest"],
            "monitor_spec_provenance": mon_manifest.get("spec_provenance"),
            "monitor_spec_source_path": mon_manifest.get("spec_source_path"),
            "fit_features_manifest_sha256":
                sha256_file(fit_features_dir / "features_manifest.json"),
            "monitor_features_manifest_sha256":
                sha256_file(Path(monitor_features_dir)
                            / "features_manifest.json"),
        }
        # Bond-source evidence: the disputed-5 are all fit reactions — the
        # monitor build must carry NO covalent-fallback / disputed reaction.
        record["bond_source"] = {
            "disputed_reactions": mon_manifest.get("disputed_reactions"),
            "bond_source_counts": mon_manifest.get("bond_source_counts"),
            "expected": "no disputed reactions and no covalent_fallback "
                        "in the monitor wave (disputed-5 are all fit)",
        }
        if "same_density" not in same_density_cache:
            same_density_cache["same_density"] = \
                fit_same_density_baseline_mle(
                    fit_states, action_max=DEFAULT_A_MAX_ANGSTROM)
        same_density = same_density_cache["same_density"]
        monitor_states, _meta = load_monitor_states(
            monitor_features_dir, monitor_observed_dir,
            expected_group_ids=group_ids)
        record["monitor_data"] = {
            "n_states": int(_meta["n_states"]),
            "n_groups": int(_meta["n_groups"]),
            "n_action_states": int(_meta["n_action_states"]),
            "features_dir": str(monitor_features_dir),
            "observed_dir": str(monitor_observed_dir),
        }
        # Live leakage re-assert (evidence recorded; the prereg asserted the
        # same at freeze time): fit ∩ monitor = ∅, partition membership,
        # disputed-5 empty in monitor, sealed-family overlap empty.
        lineage_payload = json.loads(
            (PROJECT_ROOT / prereg["lineage"]["path"]).read_text())
        record["leakage_assertions"] = assert_fit_monitor_disjoint(
            fit_group_ids=[str(r["reaction_group_id"])
                           for r in fit_manifest["per_reaction"]],
            monitor_group_ids=group_ids,
            lineage=lineage_payload,
            disputed_ids=fit_manifest.get("disputed_reactions", ()))
        policy = load_policy_from_checkpoint(torch, ckpt_path)
        device = next(policy.parameters()).device
        per_group = evaluate_groups_with_policy(
            torch, policy, monitor_states, group_ids=group_ids,
            action_max=DEFAULT_A_MAX_ANGSTROM,
            moving_threshold=moving_threshold,
            same_density=same_density,
            near_ts_arc_fraction=near_ts_arc_fraction,
            device=device)
        summary = summarize_over_groups(per_group)
        evaluation = {
            "schema_version": "pesn_wp4_monitor_eval_v1",
            "recipe": recipe,
            "prereg_sha256": digest,
            "thresholds_sha256": sha256_file(thresholds_path),
            "same_density_baseline": same_density,
            "spec_reuse": record["spec_reuse"],
            "bond_source": record["bond_source"],
            "monitor_data": record["monitor_data"],
            "leakage_assertions": record["leakage_assertions"],
            "per_group": per_group,
            "summary": summary,
            "c3_trigger": c3_trigger_evaluation(summary),
        }
        eval_path = run_dir / "monitor_eval.json"
        write_json(eval_path, evaluation)
        record.update({
            "status": "evaluated", "file": eval_path.name,
            "summary": summary, "c3_trigger": evaluation["c3_trigger"],
        })
        return record
    except MonitorGateError as exc:
        record.update({
            "status": "refused", "code": exc.code, "detail": exc.detail,
        })
        return record
    except Exception as exc:  # unexpected — recorded verbatim, never silent
        record.update({
            "status": "error", "error": repr(exc),
            "traceback": traceback.format_exc(),
        })
        return record


def _monitor_gate_criteria(summary: dict, gate: dict) -> dict:
    """The four frozen monitor-gate criteria evaluated on ONE summary.

    Mirrors the ``gate_pass`` conjuncts of
    :func:`pes2ts_core.pesn.offline.training.dual_ts_monitor.apply_selection_rule`
    (median MSE ratio < max, >= min groups below 1.0, median |a|-w cosine >=
    min, median near-TS cosine >= min) so per-seed gate decisions can be
    reported and aggregated (finding 1 seed aggregation).
    """
    ratio = summary.get("mse_ratio_median")
    cos = summary.get("direction_cosine_median")
    ratio_below = summary.get("n_groups_ratio_below_one", 0)
    near_ts = (summary.get("strata") or {}).get("near_ts", {})
    near_ts_cos = near_ts.get("direction_cosine_median")
    c1 = bool(ratio is not None and math.isfinite(float(ratio))
              and float(ratio) < float(gate["median_mse_ratio_max"]))
    c2 = bool(int(ratio_below) >= int(gate["n_groups_ratio_below_one_min"]))
    c3 = bool(cos is not None
              and float(cos) >= float(gate["median_cosine_min"]))
    c4 = bool(near_ts_cos is not None
              and float(near_ts_cos) >= float(gate["near_ts_cosine_min"]))
    return {
        "mse_ratio_median_below_max": c1,
        "n_groups_ratio_below_one_min": c2,
        "direction_cosine_median_above_min": c3,
        "near_ts_cosine_above_min": c4,
        "pass": c1 and c2 and c3 and c4,
    }


def _median_across(values) -> float | None:
    vals = [float(v) for v in values
            if v is not None and math.isfinite(float(v))]
    return float(np.median(vals)) if vals else None


def _median_per_group(per_group_lists) -> list | None:
    """Element-wise median across seeds of per-group metric lists."""
    lists = [p for p in per_group_lists if p is not None]
    if not lists:
        return None
    n = len(lists[0])
    out = []
    for i in range(n):
        vals = [float(p[i]) for p in lists
                if p[i] is not None and math.isfinite(float(p[i]))]
        out.append(float(np.median(vals)) if vals else None)
    return out


def _aggregate_seed_summaries(seed_summaries: list[dict]) -> dict:
    """Median summary across seeds (element-wise per-group median; the
    per-group counts are recomputed from the medians so they stay consistent)."""
    n = len(seed_summaries)
    mse_pg = _median_per_group([s.get("mse_ratio_per_group")
                                for s in seed_summaries])
    cos_pg = _median_per_group([s.get("direction_cosine_per_group")
                                for s in seed_summaries])
    n_ratio_below = sum(1 for v in (mse_pg or [])
                        if v is not None and v < 1.0)
    n_cos_pos = sum(1 for v in (cos_pg or [])
                    if v is not None and v > 0)
    strata = {}
    for stratum in ("near_ts", "moving", "near_rest"):
        strata[stratum] = {
            "mse_ratio_median": _median_across(
                [s.get("strata", {}).get(stratum, {}).get("mse_ratio_median")
                 for s in seed_summaries]),
            "direction_cosine_median": _median_across(
                [s.get("strata", {}).get(stratum, {})
                      .get("direction_cosine_median")
                 for s in seed_summaries]),
        }
    return {
        "n_groups": seed_summaries[0].get("n_groups") if n else None,
        "n_seeds_aggregated": n,
        "mse_ratio_per_group": mse_pg,
        "mse_ratio_median": _median_across(
            [s.get("mse_ratio_median") for s in seed_summaries]),
        "mse_ratio_min": min(
            (s.get("mse_ratio_min") for s in seed_summaries
             if s.get("mse_ratio_min") is not None), default=None),
        "mse_ratio_max": max(
            (s.get("mse_ratio_max") for s in seed_summaries
             if s.get("mse_ratio_max") is not None), default=None),
        "n_groups_ratio_below_one": n_ratio_below,
        "direction_cosine_per_group": cos_pg,
        "direction_cosine_median": _median_across(
            [s.get("direction_cosine_median") for s in seed_summaries]),
        "n_groups_cosine_positive": n_cos_pos,
        "nll_per_atom_per_group": _median_per_group(
            [s.get("nll_per_atom_per_group") for s in seed_summaries]),
        "nll_same_density_per_group": _median_per_group(
            [s.get("nll_same_density_per_group") for s in seed_summaries]),
        "strata": strata,
        "pathology": {
            "sigma_pinned_at_floor_groups": sorted({
                g for s in seed_summaries
                for g in s.get("pathology", {}).get(
                    "sigma_pinned_at_floor_groups", [])}),
            "sigma_dominates_near_rest_groups": sorted({
                g for s in seed_summaries
                for g in s.get("pathology", {}).get(
                    "sigma_dominates_near_rest_groups", [])}),
        },
    }


def _arm_suffix(*, signed_mu_gate: bool, signed_vector_messages: bool) -> str:
    """Signed-flag suffix mirroring the run-dir variant naming."""
    suffix = ""
    if signed_mu_gate:
        suffix += "_signedmu"
    if signed_vector_messages:
        suffix += "_signedvecmsg"
    return suffix


def _arm_tag(recipe: str, *, signed_mu_gate: bool,
             signed_vector_messages: bool) -> str:
    """Arm-unique recipe tag: ``recipe`` + signed-flag suffix (finding 4)."""
    return f"{recipe}{_arm_suffix(signed_mu_gate=signed_mu_gate,
                                  signed_vector_messages=signed_vector_messages)}"


def _run_key(recipe: str, *, signed_mu_gate: bool,
             signed_vector_messages: bool, seed: int, stage: str) -> str:
    """Arm-unique + stage-explicit key: ``<arm>__seed<seed>__stage-<stage>``
    (finding 4: signed arms no longer collide with the unsigned recipe)."""
    return (f"{_arm_tag(recipe, signed_mu_gate=signed_mu_gate,
                        signed_vector_messages=signed_vector_messages)}"
            f"__seed{seed}__stage-{stage}")


def _resolve_fit_features_dir(protocol: dict, run_entries: dict,
                              fit_features_dir: Path) -> Path:
    """Evidence fit-features dir from the ACTIVE PROTOCOL, never a diagnostic
    build (finding 5).

    Resolution order: (1) the active protocol's ``pipeline.feature_dirs.fit``;
    (2) the latest FORMAL run's recorded ``data.features_dir``; (3) the
    ``fit_features_dir`` argument, REFUSED if it looks like a diagnostic
    (``_a45`` / ``_diag``) build.
    """
    protocol_fit = str(((protocol.get("pipeline") or {})
                        .get("feature_dirs") or {}).get("fit", "")).strip()
    if protocol_fit:
        p = Path(protocol_fit)
        return p if p.is_absolute() else PROJECT_ROOT / p
    formal_dirs = [str(e.get("features_dir", ""))
                   for e in run_entries.values() if e.get("features_dir")]
    if formal_dirs:
        p = Path(formal_dirs[-1])  # last formal run in sorted dir order
        return p if p.is_absolute() else PROJECT_ROOT / p
    name = fit_features_dir.name
    if "_a45" in name or "_diag" in name:
        raise SystemExit(
            f"REFUSED: evidence fit-features dir {fit_features_dir} is a "
            f"diagnostic build — the ACTIVE PROTOCOL's fit dir (or a FORMAL "
            f"run's) must be used, never a diagnostic run (finding 5)")
    return fit_features_dir


def finalize_selection(
    out_dir: Path,
    thresholds_path: Path,
    same_density_cache: dict,
    protocol: dict,
    *,
    fit_features_dir: Path = FEATURES_DIR,
) -> dict:
    """v4.1 selection record → ``selection_record_v41.json`` (finding 2/1/3/4/5).

    Only v4.1 runs are ingested: schema ``pesn_dual_ts_bc_run_v2`` AND
    ``data.spec_digest`` == the protocol pipeline digest AND
    ``settings.protocol_sha256`` == the ACTIVE protocol freeze sha256 (finding
    3) AND not a diagnostic run.  Stage dimension (finding 1): ``full`` runs
    feed the monitor-only selection rule + ``runs`` + ``evidence.monitor_evals``;
    ``short`` runs go to a SEPARATE ``screening`` section (never ``runs``);
    canaries (fit1/fit5) are keyed WITH stage in ``canary_record``.  Signed
    arms (``_signedvecmsg``/``_signedmu``) are keyed arm-uniquely and included
    in EVERY section (finding 4).  The registered gate outcome is separated
    from the RELATIVE argmin recipe (finding 3): ``gate_outcome`` is
    ``SELECTED`` iff ≥1 arm passes the ≥2/3-seed rule, else
    ``REGISTER_ABLATION``; ``relative_best_recipe`` is the frozen rule's argmin
    (relative ranking only, never a gate pass).  Historical v4
    ``pesn_dual_ts_bc_run_v1`` records (list-format ``selection_record.json``)
    are NEVER touched.
    """
    thresholds = json.loads(thresholds_path.read_text())
    pipeline_spec = str(
        (protocol.get("pipeline") or {}).get("spec_digest", ""))
    active_protocol_sha = str(
        (protocol.get("freeze") or {}).get("sha256", ""))
    gate = thresholds.get("gate") or {}
    selection_rule = thresholds.get("selection_rule") or {}
    run_entries: dict[str, dict] = {}       # stage == "full" only
    screening_entries: dict[str, dict] = {}  # stage == "short"
    canary_entries: dict[str, dict] = {}     # keyed <arm>__seed__stage-<stage>
    for run_dir in sorted(out_dir.glob("run_bc_*")):
        rec_path = run_dir / "run_record.json"
        if not rec_path.is_file():
            continue
        rec = json.loads(rec_path.read_text())
        # v4.1 schema only (finding 2.5: ignore v4-shaped records)
        if rec.get("schema_version") != V41_RUN_SCHEMA:
            continue
        data = rec.get("data") or {}
        if pipeline_spec and str(data.get("spec_digest", "")) != pipeline_spec:
            continue  # different pipeline build — not this protocol's matrix
        settings = rec.get("settings") or {}
        if bool(settings.get("diagnostic_run", False)):
            continue  # diagnostics excluded entirely
        # finding 3: bind per-run protocol revision to the active protocol
        if active_protocol_sha and \
                str(settings.get("protocol_sha256", "")) != active_protocol_sha:
            continue  # recorded under a different protocol revision — exclude
        stage = str(settings.get("stage", ""))
        seed = int(settings.get("seed", 0))
        recipe = str(rec.get("recipe", ""))
        lambda_a = float(settings.get("lambda_a", 1.0))
        lambda_dir = float(settings.get("lambda_dir", 1.0))
        signed_mu_gate = bool(settings.get("signed_mu_gate", False))
        signed_vector_messages = bool(
            settings.get("signed_vector_messages", False))
        key = _run_key(recipe, signed_mu_gate=signed_mu_gate,
                       signed_vector_messages=signed_vector_messages,
                       seed=seed, stage=stage)
        entry = {
            "key": key,
            "arm": _arm_tag(recipe, signed_mu_gate=signed_mu_gate,
                            signed_vector_messages=signed_vector_messages),
            "recipe": recipe, "seed": seed, "stage": stage,
            "lambda_a": lambda_a, "lambda_dir": lambda_dir,
            "run_dir": run_dir.name,
            "signed_mu_gate": signed_mu_gate,
            "signed_vector_messages": signed_vector_messages,
            "diagnostic_run": False,
            "protocol_sha256": str(settings.get("protocol_sha256", "")),
            "updates_run": rec["accounting"]["updates_run"],
            "bundle_content_digest": rec.get("bundle_content_digest"),
            "features_dir": data.get("features_dir"),
        }
        eval_path = run_dir / "monitor_eval.json"
        if eval_path.is_file():
            ev = json.loads(eval_path.read_text())
            entry["monitor_summary"] = ev["summary"]
            entry["c3_trigger"] = ev.get("c3_trigger")
        if stage == "full":
            run_entries[key] = entry
        elif stage == "short":
            screening_entries[key] = entry
        elif stage in CANARY_STAGES:
            canary_entries[key] = entry

    # -- arm aggregation over FULL-stage runs (signed arms INCLUDED) ----------
    by_arm_seed: dict[str, dict[int, dict]] = {}
    for entry in run_entries.values():
        if "monitor_summary" not in entry:
            continue
        by_arm_seed.setdefault(entry["arm"], {})[entry["seed"]] = \
            entry["monitor_summary"]

    seed_aggregation: dict[str, dict] = {}
    arm_summaries: dict[str, dict] = {}
    per_recipe: dict[str, dict] = {}
    for arm in sorted(by_arm_seed):
        seeds = sorted(by_arm_seed[arm])
        per_seed: dict[str, dict] = {}
        for s in seeds:
            summary = by_arm_seed[arm][s]
            criteria = _monitor_gate_criteria(summary, gate) if gate \
                else {"pass": False}
            per_seed[str(s)] = {
                "gate_decision": "PASS" if criteria.get("pass")
                else "BELOW_GATE",
                "gate_criteria": criteria,
                "summary": summary,
            }
        n_seeds = len(seeds)
        n_pass = sum(1 for p in per_seed.values()
                     if p["gate_decision"] == "PASS")
        majority_pass = bool(n_seeds and n_pass * 3 >= 2 * n_seeds)
        median_summary = _aggregate_seed_summaries(
            [by_arm_seed[arm][s] for s in seeds])
        arm_summaries[arm] = median_summary
        seed_aggregation[arm] = {
            "n_seeds": n_seeds,
            "n_seeds_passing": n_pass,
            "selection_status": "PASS" if majority_pass else "BELOW_GATE",
            "majority_rule": "≥ 2/3 seeds individually PASS all four "
                             "monitor-gate criteria",
            "per_seed": per_seed,
            "median_summary": median_summary,
        }
        pathology = bool(
            median_summary.get("pathology", {})
            .get("sigma_pinned_at_floor_groups")
            or median_summary.get("pathology", {})
            .get("sigma_dominates_near_rest_groups"))
        cos = median_summary.get("direction_cosine_median")
        cos_pos = median_summary.get("n_groups_cosine_positive")
        eligibility_ok = bool(
            cos is not None and math.isfinite(float(cos))
            and float(cos) > float(selection_rule.get(
                "eligibility_median_cosine_min", 0.0))
            and int(cos_pos or 0) >= int(selection_rule.get(
                "eligibility_n_cosine_positive_min", 0)))
        per_recipe[arm] = {
            "gate_decision": "PASS" if majority_pass else "BELOW_GATE",
            "selection_eligible": bool(eligibility_ok and not pathology),
            "sigma_pathology": pathology,
        }

    # -- relative best (argmin over PRIMARY recipes, NEVER a gate pass) -------
    primary_summaries = {
        arm: arm_summaries[arm] for arm in sorted(arm_summaries)
        if arm in selection_rule.get("simplicity_order", [])}
    selected = apply_selection_rule(primary_summaries, thresholds) \
        if primary_summaries else None
    relative_best_recipe = selected["selected_recipe"] if selected else None
    # finding 3: the registered gate outcome is REGISTER_ABLATION unless at
    # least one arm passes the >=2/3-seed rule.
    gate_outcome = ("SELECTED" if any(
        sa["selection_status"] == "PASS" for sa in seed_aggregation.values())
        else "REGISTER_ABLATION")

    if not arm_summaries:
        record = {
            "schema_version": "pesn_wp4_selection_record_v41",
            "gate_outcome": "NOT_EVALUABLE",
            "relative_best_recipe": None,
            "status": "NOT_EVALUABLE",
            "reason": "monitor8 evaluation blocked: the 8 monitor-group "
                      "teacher payloads + xTB observations do not exist "
                      "(separately billed QC, §F). The preregistered "
                      "selection rule is recorded and will be applied by "
                      "scripts/train_dual_ts_bc.py once monitor8 features "
                      "exist; NO selection is made on fit metrics (§I.9).",
            "c3_trigger": {
                "status": "not_evaluable",
                "reason": "C3 trigger evidence is monitor-only (joint |a|-w "
                          "cosine < 0.30 OR near-TS cosine < 0.40 at budget "
                          "end); monitor8 data missing — the trigger cannot "
                          "fire on fit numbers, so the signed arm "
                          "(signed_mu_gate=true) is NOT run (default OFF "
                          "stands).",
            },
            "prereg_sha256": thresholds["prereg_sha256"],
            "thresholds_sha256": sha256_file(thresholds_path),
            "protocol_sha256": active_protocol_sha,
            "pipeline_spec_digest": pipeline_spec,
            "selection_key": ["arm", "seed", "stage"],
            "seed_aggregation_rule": "per-seed gate decisions all reported; "
                                     "a recipe's selection status = ≥2/3 "
                                     "seeds individually PASS all four "
                                     "monitor-gate criteria; per_recipe "
                                     "summary reports median metrics across "
                                     "seeds",
            "selection_rule_frozen": thresholds["selection_rule"],
            "runs": run_entries,
            "screening": screening_entries,
            "canary_record": canary_entries,
        }
    else:
        primary_c3 = {
            key: r.get("c3_trigger") for key, r in run_entries.items()
            if not (r.get("signed_mu_gate") or r.get("signed_vector_messages"))
            and r.get("c3_trigger")}
        record = {
            "schema_version": "pesn_wp4_selection_record_v41",
            "gate_outcome": gate_outcome,
            "relative_best_recipe": relative_best_recipe,
            "relative_best_recipe_note":
                "argmin of the frozen selection rule over PRIMARY recipes "
                "(median per-group MSE ratio, tie_band 0.01, simplicity_order) "
                "— RELATIVE ranking only, NOT a gate pass; see gate_outcome "
                "for the registered gate decision.",
            "status": gate_outcome,
            "selection_key": ["arm", "seed", "stage"],
            "protocol_sha256": active_protocol_sha,
            "pipeline_spec_digest": pipeline_spec,
            "rule": thresholds.get("selection_rule"),
            "selection_metric": (selected or {}).get("selection_metric"),
            "per_recipe": per_recipe,
            "seed_aggregation": seed_aggregation,
            "c3_trigger": {
                "per_primary_run": primary_c3,
                "any_primary_triggered": any(
                    (t or {}).get("status") == "triggered"
                    for t in primary_c3.values()),
            },
            "evidence": {
                "prereg": str(DEFAULT_PREREG),
                "thresholds": str(thresholds_path),
                "fit_features_manifest": str(
                    _resolve_fit_features_dir(
                        protocol, run_entries, fit_features_dir)
                    / "features_manifest.json"),
                "monitor_evals": {
                    key: str(Path(r["run_dir"]) / "monitor_eval.json")
                    for key, r in run_entries.items()
                    if "monitor_summary" in r},
                "selection_metric_note": "monitor8 median per-group MSE "
                                         "ratio vs zero-action; NEVER fit "
                                         "metrics (§I.9)",
            },
            "runs": run_entries,
            "screening": screening_entries,
            "canary_record": canary_entries,
        }
    write_json(out_dir / SELECTION_RECORD_NAME, record)
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipe", required=True, choices=RECIPES)
    ap.add_argument("--budget-updates", type=int, default=20000)
    ap.add_argument("--batch-states", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-a", type=float, default=1.0,
                    help="L_a weight (default 1.0); must match a registered "
                         "protocol slot unless --allow-diagnostic-budget")
    ap.add_argument("--lambda-dir", type=float, default=1.0,
                    help="L_dir weight (default 1.0); arm B unfreeze — must "
                         "match a registered slot unless diagnostic")
    ap.add_argument("--stage", default="full",
                    choices=("fit1", "fit5", "short", "full"),
                    help="registered budget stage (fit1/fit5 are canaries "
                         "with any update count)")
    ap.add_argument("--allow-diagnostic-budget", action="store_true",
                    help="accept an unregistered budget/λ as a diagnostic run "
                         "(recorded diagnostic_run=true; excluded from formal "
                         "selection)")
    ap.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL,
                    help="path to the verified protocol_v41[_rN].json "
                         "(budget/λ/matrix registration)")
    ap.add_argument("--features-dir", type=Path, default=FEATURES_DIR,
                    help="fit feature shard dir (default: v41 features)")
    ap.add_argument("--observed-dir", type=Path, default=OBSERVED_DIR,
                    help="fit observed teacher table dir")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--monitor-prereg", type=Path, default=DEFAULT_PREREG)
    ap.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-monitor-attempt", action="store_true",
                    help="record the monitor attempt as skipped (never "
                         "default)")
    ap.add_argument("--monitor-features-dir", type=Path,
                    default=MONITOR_FEATURES_DIR,
                    help="monitor feature shard dir (default unchanged)")
    ap.add_argument("--monitor-observed-dir", type=Path,
                    default=MONITOR_OBSERVED_DIR,
                    help="monitor observed teacher table dir (default "
                         "unchanged)")
    ap.add_argument("--monitor-eval-only", action="store_true",
                    help="additive: skip training; re-run ONLY the C4-gated "
                         "monitor evaluation against the EXISTING checkpoint "
                         "in the recipe run dir (same gates, same prereg "
                         "digest binding read from the run record)")
    ap.add_argument("--signed-mu-gate", action="store_true",
                    help="C3 registered ablation arm (signed_mu_gate=true); "
                         "default OFF; run dir gets a _signedmu suffix and "
                         "is EXCLUDED from the primary selection rule")
    ap.add_argument("--signed-vector-messages", action="store_true",
                    help="G4 §4 registered escalation arm "
                         "(signed_vector_messages=true, the DualEGNN "
                         "message-gate variant); default OFF; run dir gets "
                         "a _signedvecmsg suffix and is EXCLUDED from the "
                         "primary selection rule")
    args = ap.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    wp3 = json.loads((PROJECT_ROOT / (
        "outputs/pesn/dual_ts_v2_20261015/wp3_model/summary_lam1.json")
    ).read_text())
    moving_threshold = float(
        wp3["hyperparams_common"]["moving_threshold_A"])

    if args.monitor_eval_only:
        variant = ""
        if args.signed_mu_gate:
            variant += "_signedmu"
        if args.signed_vector_messages:
            variant += "_signedvecmsg"
        # finding 2: run-dir identity includes stage/λ; reconstruct the dir
        # name for the EXISTING run (formal runs are non-diagnostic).
        run_dir = out_dir / build_run_dir_name(
            recipe=args.recipe, seed=args.seed, variant=variant,
            stage=args.stage, lambda_a=args.lambda_a,
            lambda_dir=args.lambda_dir, diagnostic_run=False)
        ckpt_path = run_dir / "checkpoint_last.pt"
        if not ckpt_path.is_file():
            raise SystemExit(
                f"REFUSED: --monitor-eval-only requires an existing "
                f"checkpoint; missing {ckpt_path}")
        rec = json.loads((run_dir / "run_record.json").read_text())
        recorded_prereg = str(
            (rec.get("settings") or {}).get("prereg_sha256", ""))
        protocol, _ = load_verified_protocol(
            args.protocol, require_formal=(args.stage in FORMAL_STAGES))
        print(f"[monitor-eval-only] recipe={args.recipe} "
              f"run_dir={run_dir.name} recorded_prereg="
              f"{recorded_prereg[:12]}…", flush=True)
        print(f"loading fit50 shards from {args.features_dir} "
              f"(same-density baseline population)", flush=True)
        states, meta = load_dual_ts_states(args.features_dir,
                                           args.observed_dir)
        print(f"loaded {meta['n_states']} states / {meta['n_groups']} groups",
              flush=True)
        same_density_cache: dict = {}
        evaluation = attempt_monitor_evaluation(
            torch=require_torch(), recipe=args.recipe, run_dir=run_dir,
            ckpt_path=ckpt_path, prereg_sha256=recorded_prereg,
            thresholds_path=args.thresholds,
            moving_threshold=moving_threshold,
            near_ts_arc_fraction=NEAR_TS_ARC_FRACTION,
            fit_states=states, same_density_cache=same_density_cache,
            prereg_path=args.monitor_prereg,
            monitor_features_dir=args.monitor_features_dir,
            monitor_observed_dir=args.monitor_observed_dir,
            fit_features_dir=args.features_dir)
        final = json.loads((run_dir / "run_record.json").read_text())
        final["monitor_evaluation"] = evaluation
        write_json(run_dir / "run_record.json", final)
        print(f"monitor attempt: {evaluation.get('status')}"
              f" {evaluation.get('code', '')}", flush=True)
        if evaluation.get("status") == "evaluated":
            s = evaluation["summary"]
            c3 = evaluation["c3_trigger"]
            print(f"[monitor8] recipe={args.recipe} "
                  f"median_ratio={s['mse_ratio_median']:.4f} "
                  f"ratio<1 groups={s['n_groups_ratio_below_one']}/8 "
                  f"median_cos={s['direction_cosine_median']:.4f} "
                  f"cos>0 groups={s['n_groups_cosine_positive']}/8 "
                  f"near_ts_cos="
                  f"{s['strata']['near_ts']['direction_cosine_median']}",
                  flush=True)
            print(f"[c3] {c3.get('status')} "
                  f"(joint fired={c3.get('joint_fired')}, "
                  f"near_ts fired={c3.get('near_ts_fired')})", flush=True)
        selection = finalize_selection(out_dir, args.thresholds,
                                       same_density_cache, protocol,
                                       fit_features_dir=args.features_dir)
        print(f"selection record: {out_dir / SELECTION_RECORD_NAME} "
              f"status={selection['status']}", flush=True)
        return 0

    # -- finding 4: verify the protocol on READ (content digest + chain) -----
    # finding 3: FORMAL stages additionally require the r1+ amendment contract
    # (r0 stays usable for canaries/diagnostics only).
    protocol, protocol_sha256 = load_verified_protocol(
        args.protocol, require_formal=(args.stage in FORMAL_STAGES))

    # -- protocol_v41 registration gate (budget + λ + matrix identity) -------
    # load spec + manifest FIRST (the gate binds the features spec digest +
    # population group set to the protocol).
    spec_digest, manifest = load_spec_and_manifest(args.features_dir)
    subset = None
    if args.stage == "fit1":
        subset = list(protocol["fit1_fit5_subsets"]["fit1_group_ids"])
    elif args.stage == "fit5":
        subset = list(protocol["fit1_fit5_subsets"]["fit5_group_ids"])
    if subset is not None:
        population = sorted(str(g) for g in subset)
    else:
        population = sorted(
            str(r["reaction_group_id"]) for r in manifest["per_reaction"])
    decision = resolve_lambda_registration(
        protocol, recipe=args.recipe, lambda_a=args.lambda_a,
        lambda_dir=args.lambda_dir, stage=args.stage,
        budget_updates=args.budget_updates,
        allow_diagnostic_budget=bool(args.allow_diagnostic_budget),
        seed=args.seed, features_spec_digest=spec_digest,
        population_group_ids=population,
        batch_states=args.batch_states, lr=args.lr,
        signed_vector_messages=bool(args.signed_vector_messages))
    if not decision["accepted"]:
        raise SystemExit(
            "REFUSED: run is not registered in protocol_v41 — "
            + "; ".join(decision["problems"])
            + " — re-freeze protocol_v41 or pass --allow-diagnostic-budget")
    diagnostic_run = bool(decision["diagnostic"])
    if diagnostic_run:
        print("[diagnostic] unregistered budget/λ; run recorded as "
              "diagnostic_run=true and EXCLUDED from formal selection",
              flush=True)

    # C4 gate: monitor prereg must exist before any monitor-compared run
    _, prereg_digest = load_prereg(args.monitor_prereg)

    print(f"loading shards from {args.features_dir}"
          + (f" (stage {args.stage} subset: {subset})" if subset else ""),
          flush=True)
    states, meta = load_dual_ts_states(args.features_dir, args.observed_dir,
                                       group_filter=subset)
    print(f"loaded {meta['n_states']} states / {meta['n_groups']} groups; "
          f"max |a*| = {meta['max_action_norm_angstrom']:.6f} A",
          flush=True)

    settings = DualTSBCSettings(
        recipe=args.recipe, budget_updates=args.budget_updates,
        batch_states=args.batch_states, lr=args.lr, seed=args.seed,
        lambda_a=args.lambda_a, lambda_dir=args.lambda_dir,
        device=args.device, prereg_sha256=prereg_digest,
        stage=args.stage, protocol_sha256=protocol_sha256,
        diagnostic_run=diagnostic_run,
        signed_mu_gate=bool(args.signed_mu_gate),
        signed_vector_messages=bool(args.signed_vector_messages))
    variant = ""
    if args.signed_mu_gate:
        variant += "_signedmu"
    if args.signed_vector_messages:
        variant += "_signedvecmsg"
    run_dir = out_dir / build_run_dir_name(
        recipe=args.recipe, seed=args.seed, variant=variant,
        stage=args.stage, lambda_a=args.lambda_a,
        lambda_dir=args.lambda_dir, diagnostic_run=diagnostic_run)

    # finding 2 (F3): refuse to overwrite an existing run record — no silent
    # stage/λ/diagnostic collision with itself or with historical v4 artifacts.
    if run_dir.exists():
        rec_path = run_dir / "run_record.json"
        if rec_path.is_file():
            existing = json.loads(rec_path.read_text())
            existing_digest = _record_settings_digest(existing)
            current_digest = run_settings_digest(
                recipe=args.recipe, seed=args.seed, stage=args.stage,
                lambda_a=args.lambda_a, lambda_dir=args.lambda_dir,
                features_spec_digest=spec_digest,
                protocol_sha256=protocol_sha256,
                budget_updates=args.budget_updates, lr=args.lr,
                batch_states=args.batch_states,
                signed_mu_gate=bool(args.signed_mu_gate),
                signed_vector_messages=bool(args.signed_vector_messages))
            if existing_digest == current_digest:
                raise SystemExit(
                    f"REFUSED: {run_dir.name} already contains a run_record "
                    f"with IDENTICAL settings — delete it (or rename) to "
                    f"re-run; no silent overwrite")
            raise SystemExit(
                f"REFUSED: {run_dir.name} already contains a run_record with "
                f"DIFFERENT settings — delete it to re-run; no silent "
                f"overwrite")

    trainer = DualTSBCTrainer(
        settings=settings, states=states, data_meta=meta, run_dir=run_dir)
    print(f"training recipe={args.recipe} on {trainer.device} "
          f"({trainer.policy.n_parameters} params)", flush=True)
    record = trainer.train()

    ckpt_path = run_dir / "checkpoint_last.pt"
    bundle = trainer.build_bundle(record)
    bundle_path = run_dir / "bundle.json"
    bundle_path.write_text(json.dumps(bundle.to_payload(), indent=2) + "\n")
    validate_record("pesn_dual_ts_bundle_v2", bundle.to_payload())
    record["bundle_file"] = bundle_path.name
    record["bundle_content_digest"] = bundle.content_digest()
    record["bundle_validated"] = True
    final = json.loads((run_dir / "run_record.json").read_text())
    final.update({k: record[k] for k in
                  ("bundle_file", "bundle_content_digest",
                   "bundle_validated")})
    write_json(run_dir / "run_record.json", final)
    print(f"bundle validated: {bundle_path} "
          f"(digest {bundle.content_digest()[:16]}…)", flush=True)

    fd = record["fit_diagnostics_final"]
    print(f"[fit-side diagnostic ONLY (not selection)]: "
          f"MSE/baseline={fd['mse_over_baseline']:.3f} "
          f"cos={fd['direction_cosine_absa_weighted']} "
          f"NLL/atom={fd['nll_per_atom']:.3f} "
          f"sigma_mean={fd['sigma_mean']:.4f}", flush=True)

    # monitor attempt (C4-gated) + selection record refresh
    same_density_cache: dict = {}
    if args.skip_monitor_attempt:
        record["monitor_evaluation"] = {
            "attempted": False, "status": "skipped_by_flag"}
    else:
        record["monitor_evaluation"] = attempt_monitor_evaluation(
            torch=trainer.torch, recipe=args.recipe, run_dir=run_dir,
            ckpt_path=ckpt_path, prereg_sha256=prereg_digest,
            thresholds_path=args.thresholds,
            moving_threshold=moving_threshold,
            near_ts_arc_fraction=NEAR_TS_ARC_FRACTION,
            fit_states=states, same_density_cache=same_density_cache,
            prereg_path=args.monitor_prereg,
            monitor_features_dir=args.monitor_features_dir,
            monitor_observed_dir=args.monitor_observed_dir)
        print(f"monitor attempt: {record['monitor_evaluation'].get('status')}"
              f" {record['monitor_evaluation'].get('code', '')}",
              flush=True)
    final = json.loads((run_dir / "run_record.json").read_text())
    final["monitor_evaluation"] = record["monitor_evaluation"]
    write_json(run_dir / "run_record.json", final)

    selection = finalize_selection(out_dir, args.thresholds,
                                   same_density_cache, protocol,
                                   fit_features_dir=args.features_dir)
    print(f"selection record: {out_dir / SELECTION_RECORD_NAME} "
          f"status={selection['status']}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DualTSBCTrainerError as exc:
        print(f"TRAINER REFUSAL {exc}", file=sys.stderr)
        raise SystemExit(2)
