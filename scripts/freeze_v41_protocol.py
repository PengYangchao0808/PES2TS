"""Freeze the v4.1 WP2 training protocol ``protocol_v41.json`` (P06 / F6 / F7).

This is the NEW pre-registration for the staged BC/IQL re-training matrix.  It
BINDS (never re-derives):

* the v41 features pipeline (spec digest f678f8e5… + feature dirs);
* the registered budget stages {fit1, fit5, short=2000, full=20000};
* the fixed seeds [20261017, 20261018, 20261019];
* selection keyed by (recipe, seed) — no silent multi-seed overwrite;
* the monitor8 gate thresholds — values COPIED unchanged from
  ``monitor_thresholds_v1.json`` (bound by sha256 reference + copied values;
  thresholds are inherited, NOT re-derived, NOT lowered);
* a λ registration slot (arm A frozen at (1.0, 1.0); arm B = null, to be
  frozen from WP3 gradient-scale diagnostics before full runs);
* the deterministic fit1/fit5 canary subsets and the external-confirm wave
  pre-lock (confirm8_v41).

Freeze is by content sha256: re-running with DIFFERENT content refuses;
re-running with identical content is a no-op.  Every attempt is appended to
``protocol_v41_attempts.jsonl`` (append-only, v4 pattern).

Artifact: outputs/pesn/dual_ts_v2_20261015/wp4_bc/protocol_v41.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LINEAGE_PATH = PROJECT_ROOT / (
    "outputs/pesn/dual_ts_v2_20261015/splits/dual_ts_split_lineage_v1.json")
FIT_FEATURES_DIR = PROJECT_ROOT / (
    "outputs/pesn/dual_ts_v2_20261015/wp3_features_v41")
FIT_OBSERVED_DIR = PROJECT_ROOT / (
    "outputs/pesn/dual_ts_v2_20261015/wp2_fit50_observed")
MONITOR_FEATURES_DIR = PROJECT_ROOT / (
    "outputs/pesn/dual_ts_v2_20261015/wp4_monitor/features_v41")
THRESHOLDS_PATH = PROJECT_ROOT / (
    "outputs/pesn/dual_ts_v2_20261015/wp4_bc/monitor_thresholds_v1.json")
OUT_DIR = PROJECT_ROOT / "outputs/pesn/dual_ts_v2_20261015/wp4_bc"
PROTOCOL_PATH = OUT_DIR / "protocol_v41.json"
ATTEMPTS_LOG = OUT_DIR / "protocol_v41_attempts.jsonl"

PROTOCOL_SCHEMA = "pesn_dual_ts_protocol_v41"
PIPELINE_LABEL = "pesn_dual_ts_features_v3"
SEEDS = [20261017, 20261018, 20261019]
SELECTION_SEED = 20261017
NEAR_TS_ARC_FRACTION = 0.2

#: r1 revision output (chained to r0 by parent_freeze_sha256).
REVISION_PATH = OUT_DIR / "protocol_v41_r1.json"

#: arm B λ_dir freeze bounds (finding 2; frozen r1 calibration procedure).
ARM_B_LAMBDA_BOUNDS: tuple[float, float] = (1e-3, 1e3)

#: r2 seed-aggregation amendment (finding 1): how monitor8 selection consumes
#: the 3-seed matrix.
SEED_AGGREGATION_AMENDMENT: dict = {
    "rule": "per-seed gate decisions all reported; a recipe's selection "
            "status = ≥2/3 seeds individually PASS all four monitor-gate "
            "criteria; per_recipe summary reports median metrics across seeds",
    "n_seeds_required": 3,
    "majority_threshold": "≥ 2/3",
}

#: monitor8 groups (the v4 historical regression set — never re-used for
#: external confirmation; F6).
MONITOR8 = [
    "RXN_0000000798", "RXN_0000001178", "RXN_0000002770", "RXN_0000004669",
    "RXN_0000006274", "RXN_0000006957", "RXN_0000009014", "RXN_0000010293",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Deterministic fit5 / fit1 subsets (F7 canary)
# ---------------------------------------------------------------------------

def _near_ts_fractions(per_reaction: list, features_dir: Path,
                       observed_dir: Path) -> dict[str, float]:
    """Per-group fraction of drawable states with ral <= 0.2 * branch max."""
    out: dict[str, float] = {}
    for row in per_reaction:
        gid = row["reaction_group_id"]
        shard = np.load(features_dir / row["npz_file"])
        teacher = json.loads(
            (observed_dir / f"{gid}.teacher.json").read_text())
        labels = {str(l["state_id"]): l for l in teacher["labels"]}
        state_ids = [str(s) for s in shard["state_id"]]
        branch_ids = [str(b) for b in shard["branch_id"]]
        is_teacher = [bool(t) for t in shard["is_teacher_endpoint"]]
        branch_max: dict[str, float] = {}
        for sid, bid, it in zip(state_ids, branch_ids, is_teacher):
            if it:
                continue
            ral = float(labels[sid]["remaining_arc_length"])
            branch_max[bid] = max(branch_max.get(bid, 0.0), ral)
        n_drawable = 0
        n_near = 0
        for sid, bid, it in zip(state_ids, branch_ids, is_teacher):
            if it:
                continue
            ral = float(labels[sid]["remaining_arc_length"])
            n_drawable += 1
            if ral <= NEAR_TS_ARC_FRACTION * branch_max[bid]:
                n_near += 1
        out[gid] = (n_near / n_drawable) if n_drawable else 0.0
    return out


def compute_fit_subsets(manifest: dict, features_dir: Path,
                        observed_dir: Path, seed: int) -> dict:
    """Deterministic fit5 / fit1 stratified subsets (rule frozen + reproducible).

    fit5 = 5 DISTINCT fit50 groups, one per stratum (in priority order):
      1. normal        — median path length (states count, lower median)
      2. near-TS-heavy — highest near-TS state fraction (ral <= 0.2·branch max)
      3. longest-path  — most states
      4. disputed      — one of the 5 endpoint-bond-disputed groups (the
                         lowest RXN id among ``bond_source_* ==
                         covalent_fallback`` / manifest ``disputed_reactions``)
      5. smallest      — fewest atoms
    Ties break by ascending reaction-group id.  ``seed`` is recorded for
    reproducibility (used only to make tie-breaking explicit, not to randomize
    the primary selection).  fit1 = the "normal" member of fit5.
    """
    per_reaction = manifest["per_reaction"]
    near_ts = _near_ts_fractions(per_reaction, features_dir, observed_dir)
    disputed = set(manifest.get("disputed_reactions", ()))
    stats = []
    for row in per_reaction:
        gid = row["reaction_group_id"]
        stats.append({
            "group_id": gid,
            "n_atoms": int(row["n_atoms"]),
            "n_states": int(row["states"]),
            "near_ts_fraction": near_ts.get(gid, 0.0),
            "disputed": gid in disputed,
        })

    used: set[str] = set()

    def _pick(key, *, reverse: bool = False, predicate=None):
        cands = [s for s in stats
                 if s["group_id"] not in used
                 and (predicate is None or predicate(s))]
        if not cands:
            return None
        # primary key ascending (or descending via negation); ties ALWAYS
        # break by ascending reaction-group id (frozen rule).
        def sort_key(s):
            primary = key(s)
            return (-primary if reverse else primary, s["group_id"])
        return min(cands, key=sort_key)

    # 1. normal = median path length (lower median, ties -> ascending id)
    by_states = sorted(stats, key=lambda s: (s["n_states"], s["group_id"]))
    median_idx = len(by_states) // 2
    normal = by_states[median_idx]

    members: list[dict] = [normal]
    used.add(normal["group_id"])

    near_ts_member = _pick(lambda s: s["near_ts_fraction"], reverse=True)
    if near_ts_member:
        members.append(near_ts_member)
        used.add(near_ts_member["group_id"])

    longest = _pick(lambda s: s["n_states"], reverse=True)
    if longest:
        members.append(longest)
        used.add(longest["group_id"])

    disputed_member = _pick(
        lambda s: s["group_id"], predicate=lambda s: s["disputed"])
    if disputed_member:
        members.append(disputed_member)
        used.add(disputed_member["group_id"])

    smallest = _pick(lambda s: s["n_atoms"])
    if smallest:
        members.append(smallest)
        used.add(smallest["group_id"])

    fit5 = [m["group_id"] for m in members]
    fit1 = [normal["group_id"]]
    return {
        "seed": seed,
        "rule": ("fit5 = 5 distinct fit50 groups: (1) normal = median path "
                 "length; (2) near-TS-heavy = highest near-TS state fraction "
                 "(ral <= 0.2 * branch max); (3) longest path; (4) one "
                 "endpoint-bond-disputed group (lowest RXN id among "
                 "bond_source fallback-flagged); (5) smallest atom count; "
                 "ties break by ascending reaction-group id. fit1 = the "
                 "'normal' member."),
        "fit5_group_ids": fit5,
        "fit1_group_ids": fit1,
        "membership": {
            m["group_id"]: {
                "n_atoms": m["n_atoms"], "n_states": m["n_states"],
                "near_ts_fraction": m["near_ts_fraction"],
                "disputed": m["disputed"],
            } for m in members
        },
    }


# ---------------------------------------------------------------------------
# External-confirm wave pre-lock (F6)
# ---------------------------------------------------------------------------

def select_confirm8(lineage: dict, seed: int) -> dict:
    monitor_partition = set(lineage["partitions"]["monitor"]["group_ids"])
    monitor8 = set(lineage["dual_ts_wave_selections"]["wave"]["monitor"]
                   ["group_ids"])
    fit50 = set(lineage["dual_ts_wave_selections"]["wave"]["fit"]["group_ids"])
    # assert no overlap: fit50 must never be in the monitor partition
    assert not (monitor_partition & fit50), "fit50 overlaps monitor partition"
    assert monitor8 <= monitor_partition, "monitor8 not inside monitor partition"
    candidates = sorted(monitor_partition - monitor8 - fit50)
    rng = np.random.default_rng(seed)
    chosen = sorted(str(g) for g in rng.choice(candidates, size=8,
                                               replace=False))
    return {
        "seed": seed,
        "rule": ("seeded RNG(seed) choice-8 without replacement over the "
                 "monitor partition (250) minus monitor8 (8) minus anything "
                 "ever used for tuning (fit50 — asserted disjoint); stored "
                 "sorted."),
        "candidate_pool_size": len(candidates),
        "confirm8_v41": chosen,
        "assertions": {
            "fit50_disjoint_from_monitor_partition": True,
            "monitor8_excluded": sorted(set(MONITOR8) & set(chosen)) == [],
        },
    }


# ---------------------------------------------------------------------------
# Protocol payload assembly + freeze
# ---------------------------------------------------------------------------

def _serialize(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def _content_digest(payload: dict) -> str:
    """sha256 of the SCIENTIFIC content (the ``freeze`` block excluded so the
    digest is stable across re-runs — the freeze timestamp is provenance, not
    content)."""
    core = {k: v for k, v in payload.items() if k != "freeze"}
    text = json.dumps(core, sort_keys=True, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_payload() -> dict:
    lineage = json.loads(LINEAGE_PATH.read_text())
    fit_manifest = json.loads(
        (FIT_FEATURES_DIR / "features_manifest.json").read_text())
    thresholds = json.loads(THRESHOLDS_PATH.read_text())

    spec_digest = fit_manifest["spec_digest"]
    assert spec_digest.startswith("f678f8e5"), "unexpected v41 spec digest"

    subsets = compute_fit_subsets(fit_manifest, FIT_FEATURES_DIR,
                                  FIT_OBSERVED_DIR, SELECTION_SEED)
    confirm8 = select_confirm8(lineage, SELECTION_SEED)

    payload = {
        "schema_version": PROTOCOL_SCHEMA,
        "title": "v4.1 WP2 training protocol (P06/F6/F7) — frozen BEFORE any "
                 "staged re-training",
        "pipeline": {
            "label": PIPELINE_LABEL,
            "spec_digest": spec_digest,
            "feature_dirs": {
                "fit": str(FIT_FEATURES_DIR.relative_to(PROJECT_ROOT)),
                "monitor": str(MONITOR_FEATURES_DIR.relative_to(PROJECT_ROOT)),
            },
            "note": "observation-only 5-channel node_scalars (ADR-0022); "
                    "supervision keys kept out of network inputs",
        },
        "registered_budgets": {
            "fit1": {"updates": None,
                     "note": "staged canary — any updates count allowed"},
            "fit5": {"updates": None,
                     "note": "staged canary — any updates count allowed"},
            "short": {"updates": 2000, "note": "fit50 short run"},
            "full": {"updates": 20000, "note": "fit50 full run"},
        },
        "seeds": SEEDS,
        "selection_key": ["recipe", "seed"],
        "selection_note": "selection_record.json runs keyed "
                          "{recipe}__seed{seed}; no silent multi-seed "
                          "overwrite",
        "monitor_gate": {
            "thresholds_sha256": sha256_file(THRESHOLDS_PATH),
            "thresholds_file": THRESHOLDS_PATH.name,
            "inherited": True,
            "note": "thresholds INHERITED, not re-derived, not lowered; "
                    "values copied unchanged from monitor_thresholds_v1.json "
                    "and bound by sha256 reference",
            "gate": thresholds["gate"],
            "selection_rule": thresholds["selection_rule"],
            "c3_trigger": thresholds["c3_trigger"],
        },
        "lambda_registration": {
            "slots": {
                "arm_a": {"lambda_a": 1.0, "lambda_dir": 1.0,
                          "status": "frozen"},
                "arm_b": None,
            },
            "arm_b_note": "arm_b_lambda slot = null; to be frozen from WP3 "
                          "gradient-scale diagnostics before full runs",
            "rule": "a run is accepted iff (lambda_a, lambda_dir) match a "
                    "non-null slot; else refuse unless "
                    "--allow-diagnostic-budget",
        },
        "fit1_fit5_subsets": subsets,
        "external_confirm": {
            **confirm8,
            "deferred_note": "NO teacher/observation/QC build for confirm8 "
                             "now; deferred to the WP4 external-confirmation "
                             "step.  Estimated cost ~1k xTB single points.",
        },
        "freeze": {
            "utc": datetime.now(timezone.utc).isoformat(),
            "tool": "scripts/freeze_v41_protocol.py",
        },
    }
    return payload


def append_attempt(sha: str, status: str, detail: str,
                   log_path: Path | None = None) -> None:
    log = Path(log_path) if log_path is not None else ATTEMPTS_LOG
    log.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "sha256": sha,
        "status": status,
        "detail": detail,
    }
    with log.open("a") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# r1 revision (finding 4.4 / finding 7): A/B/C/D mapping + matrix + batch
# semantics + package-comparison + fit50 population registration.
# ---------------------------------------------------------------------------

def build_r1_amendments() -> tuple[dict, list[str]]:
    fit_manifest = json.loads(
        (FIT_FEATURES_DIR / "features_manifest.json").read_text())
    fit50_group_ids = sorted(
        str(r["reaction_group_id"]) for r in fit_manifest["per_reaction"])

    arm_mapping = {
        "A": {
            "recipe": "bc_nll",
            "lambda_slot": "arm_a",
            "lambda_a": 1.0,
            "lambda_dir": 1.0,
            "signed_vector_messages": False,
            "note": "clean 5-ch baseline (NLL-only recipe)",
        },
        "B": {
            "recipe": "bc_d",
            "lambda_slot": "arm_b",
            "lambda_a": 1.0,
            "lambda_dir": None,
            "signed_vector_messages": False,
            "lambda_dir_calibration": {
                "procedure": (
                    "after the fit50 short-2000 run of A, load its final "
                    "checkpoint, measure median ||dL_nll/dtheta|| and "
                    "||dL_dir/dtheta|| over >=20 fit50 batches (seeded "
                    "20261017), lambda_dir := median ratio, bounds "
                    "[1e-3, 1e3], else escalate to orchestrator"),
                "value_frozen_in": "r2",
            },
            "note": "A + scale-calibrated direction loss",
        },
        "C": {
            "recipe": "bc_nll",
            "lambda_slot": "arm_a",
            "lambda_a": 1.0,
            "lambda_dir": 1.0,
            "signed_vector_messages": True,
            "note": "A + --signed-vector-messages",
        },
        "D": {
            "recipe": "bc_d",
            "lambda_slot": "arm_b",
            "lambda_a": 1.0,
            "lambda_dir": None,
            "signed_vector_messages": True,
            "requires": "B and C each beat A on the frozen gate metrics",
            "note": "B + C",
        },
    }

    matrix_hyperparams = {
        "batch_states": 64,
        "lr": 1e-3,
        "seeds": list(SEEDS),
        "budgets_per_stage": {"short": 2000, "full": 20000},
        "note": "matrix hyperparams: batch_states=64, lr=1e-3, 3 seeds; "
                "budget per stage short=2000, full=20000 updates",
    }

    batch_semantics = {
        "bc": {"per_update": "batch_states=64 rows",
               "call": "draw_batch(batch_states)"},
        "iql": {"per_update": "rows_per_round=n_groups rows",
                "call": "draw_batch(rows_per_round)",
                "batch_reactions_unused": True},
        "registered_difference": (
            "per-update chunk size differs (64 vs n_groups); SAME marginal "
            "draw law (group round-robin + branch uniform RNG + 3-bin arc "
            "round-robin); after the finding-3 canonical-order fix the "
            "same-seed draw SEQUENCE is identical too"),
    }

    package_comparison = {
        "A_vs_v4": (
            "PACKAGE comparison (clean 5-ch inputs + unified with-replacement "
            "group sampler vs v4 epoch traversal + 7-ch); attribution of "
            "A-vs-v4 differences to inputs alone is invalid"),
        "A_B_C_D": (
            "within v4.1 is the controlled comparison (same features, "
            "sampler, budget, batch, lr, seed)"),
    }

    amendments = {
        "arm_mapping": arm_mapping,
        "matrix_hyperparams": matrix_hyperparams,
        "batch_semantics": batch_semantics,
        "package_comparison": package_comparison,
        "fit50_group_ids": fit50_group_ids,
    }
    amendment_keys = [
        "arm_mapping: A/B/C/D -> recipe/lambda-slot/signed-flag mapping; "
        "B arm_b lambda_dir value deferred to r2 (calibration procedure "
        "frozen now)",
        "matrix_hyperparams: batch_states=64, lr=1e-3, 3 seeds, budgets "
        "short=2000/full=20000",
        "batch_semantics: BC batch_states=64/update vs IQL rows_per_round="
        "n_groups (registered difference, same marginal draw law)",
        "package_comparison: A-vs-v4 is a PACKAGE comparison; A/B/C/D "
        "within v4.1 is the controlled comparison",
        "fit50_group_ids: frozen fit50 population set (formal-run gate)",
    ]
    return amendments, amendment_keys


def _verify_full_chain(path: Path, payload: dict) -> None:
    """Transitively verify a revision chain r_n → … → r0 (finding 2).

    Every parent link is recomputed: the child's ``parent_freeze_sha256`` must
    equal the parent's ``freeze.sha256`` AND the parent's own content digest
    (``freeze`` block excluded) must match.  Walks until a payload with no
    ``revision`` block (r0).
    """
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
                f"REFUSING: revision chain too deep at {current_path} "
                f"(possible cycle)")
        parent_file = str(revision.get("parent_file", "protocol_v41.json"))
        parent_path = current_path.parent / parent_file
        parent_sha = str(revision.get("parent_freeze_sha256", ""))
        if not parent_sha:
            raise SystemExit(
                f"REFUSING: revision protocol {current_path} has no "
                f"parent_freeze_sha256")
        if not parent_path.is_file():
            raise SystemExit(
                f"REFUSING: cannot verify revision chain — parent protocol "
                f"{parent_path} missing")
        parent_payload = json.loads(parent_path.read_text())
        parent_recorded = str(
            (parent_payload.get("freeze") or {}).get("sha256", ""))
        if parent_recorded != parent_sha:
            raise SystemExit(
                f"REFUSING: revision chain broken — {current_path} "
                f"parent_freeze_sha256 {parent_sha[:16]}… != {parent_path} "
                f"freeze.sha256 {parent_recorded[:16]}…")
        parent_recomputed = _content_digest(parent_payload)
        if parent_recorded != parent_recomputed:
            raise SystemExit(
                f"REFUSING: parent protocol {parent_path} content digest "
                f"mismatch (tampered after freeze)")
        current_path = parent_path
        current = parent_payload


def _fill_arm_b_lambda(revised: dict, arm_b_lambda: float) -> list[str]:
    """Fill arm_mapping B/D λ_dir + lambda_registration.slots.arm_b (finding 2).

    Returns the r2 amendment key strings added by this fill.
    """
    arm_mapping = {k: dict(v) for k, v in
                   (revised.get("arm_mapping") or {}).items()}
    for key in ("B", "D"):
        if key in arm_mapping:
            arm_mapping[key]["lambda_dir"] = float(arm_b_lambda)
    revised["arm_mapping"] = arm_mapping

    lam = {k: (dict(v) if isinstance(v, dict) else v)
           for k, v in (revised.get("lambda_registration") or {}).items()}
    slots = {k: (dict(v) if isinstance(v, dict) else v)
             for k, v in (lam.get("slots") or {}).items()}
    slots["arm_b"] = {"lambda_a": 1.0, "lambda_dir": float(arm_b_lambda),
                      "status": "frozen"}
    lam["slots"] = slots
    revised["lambda_registration"] = lam

    revised["seed_aggregation"] = SEED_AGGREGATION_AMENDMENT
    return [
        "arm_mapping: B/D lambda_dir := arm_b_lambda (frozen from WP4a "
        "calibration)",
        "lambda_registration: slots.arm_b frozen at (1.0, arm_b_lambda)",
        "seed_aggregation: per-seed gate decisions all reported; a recipe's "
        "selection status = ≥2/3 seeds individually PASS all four "
        "monitor-gate criteria; per_recipe summary reports median metrics "
        "across seeds",
        "selection_key: runs full-only, screening short-only, canary_record "
        "keyed by stage",
    ]


def revise(*, parent: Path | None = None, out: Path | None = None,
           n: int = 1, arm_b_lambda: float | None = None) -> dict:
    """Freeze a revision ``out`` chained to ``parent`` (generic, finding 2).

    ``n == 1`` (default) reproduces r1 exactly: r0 content + the r1
    amendments.  ``n >= 2`` keeps the parent's (r1) content and only fills the
    arm B/D λ_dir slots + ``lambda_registration.slots.arm_b`` from
    ``--arm-b-lambda`` (refused outside [1e-3, 1e3]) + the seed-aggregation
    amendment.  Refuses if the parent is missing/tampered, if the parent's
    FULL chain (r_n → … → r0) does not verify, or if ``out`` already exists
    with different content.  Attempts are appended to the attempts log next to
    ``out`` (``<out dir>/protocol_v41_attempts.jsonl``).
    """
    parent = Path(parent) if parent is not None else PROTOCOL_PATH
    out = Path(out) if out is not None else REVISION_PATH
    n = int(n)
    attempts_log = out.parent / "protocol_v41_attempts.jsonl"

    if not parent.is_file():
        append_attempt("", "refused",
                       f"revise: parent protocol {parent} missing",
                       log_path=attempts_log)
        raise SystemExit(
            f"REFUSING: parent protocol {parent} missing; freeze it first")
    parent_payload = json.loads(parent.read_text())
    parent_sha = str((parent_payload.get("freeze") or {}).get("sha256", ""))
    parent_recomputed = _content_digest(parent_payload)
    if not parent_sha or parent_sha != parent_recomputed:
        append_attempt(parent_sha, "refused",
                       f"revise: parent {parent} content digest mismatch "
                       f"(tampered/unfrozen)", log_path=attempts_log)
        raise SystemExit(
            f"REFUSING: parent {parent} content digest mismatch (recorded "
            f"{parent_sha[:16]}… != recomputed {parent_recomputed[:16]}…); "
            f"cannot chain a revision to a tampered parent")
    # full chain verification (finding 2): r_n → … → r0
    _verify_full_chain(parent, parent_payload)

    revised = {k: v for k, v in parent_payload.items() if k != "freeze"}

    if n == 1:
        amendments, amendment_keys = build_r1_amendments()
        revised.update(amendments)
    elif n >= 2:
        if arm_b_lambda is None:
            raise SystemExit(
                f"REFUSING: r{n} revision requires --arm-b-lambda (the frozen "
                f"arm B λ_dir calibration value)")
        lam = float(arm_b_lambda)
        if not (ARM_B_LAMBDA_BOUNDS[0] <= lam <= ARM_B_LAMBDA_BOUNDS[1]):
            append_attempt("", "refused",
                           f"revise: arm_b_lambda {lam} outside "
                           f"{ARM_B_LAMBDA_BOUNDS}", log_path=attempts_log)
            raise SystemExit(
                f"REFUSING: --arm-b-lambda {lam} outside the frozen "
                f"calibration bounds {ARM_B_LAMBDA_BOUNDS}")
        amendment_keys = _fill_arm_b_lambda(revised, lam)
    else:
        raise SystemExit(f"REFUSING: --n must be >= 1, got {n}")

    revised["revision"] = {
        "n": n,
        "parent_freeze_sha256": parent_sha,
        "parent_file": parent.name,
        "amendments": amendment_keys,
    }
    revised["freeze"] = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "tool": f"scripts/freeze_v41_protocol.py --revise --n {n}",
    }
    content_digest = _content_digest(revised)
    revised["freeze"]["sha256"] = content_digest

    if out.exists():
        existing = json.loads(out.read_text())
        existing_digest = str(
            (existing.get("freeze") or {}).get("sha256", ""))
        if existing_digest != content_digest:
            append_attempt(content_digest, "refused",
                           f"revise: different r{n} content; existing "
                           f"{existing_digest}", log_path=attempts_log)
            raise SystemExit(
                f"REFUSING: {out} already exists with a DIFFERENT content "
                f"(existing {existing_digest[:16]}… != new "
                f"{content_digest[:16]}…).  Write a new revision under a new "
                f"filename instead of overwriting.")
        append_attempt(content_digest, "idempotent",
                       f"revise: same r{n} content already written",
                       log_path=attempts_log)
        print(f"already written (same content): {out}")
        print(f"r{n} content sha256: {content_digest}")
        return revised

    out.write_text(_serialize(revised))
    append_attempt(content_digest, "frozen",
                   f"revision r{n} (parent {parent.name} {parent_sha[:16]}…)",
                   log_path=attempts_log)
    print(f"r{n} frozen: {out}")
    print(f"r{n} content sha256: {content_digest}")
    print(f"parent ({parent.name}) freeze sha256: {parent_sha}")
    print("amendments:")
    for key in amendment_keys:
        print(f"  - {key}")
    return revised


def freeze() -> dict:
    payload = build_payload()
    content_digest = _content_digest(payload)
    payload["freeze"]["sha256"] = content_digest
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text())
        existing_digest = str(
            (existing.get("freeze") or {}).get("sha256", ""))
        if existing_digest != content_digest:
            append_attempt(content_digest, "refused",
                           f"different content: existing {existing_digest}")
            raise SystemExit(
                f"REFUSING: {PROTOCOL_PATH} already frozen with a DIFFERENT "
                f"content (existing {existing_digest[:16]}… != new "
                f"{content_digest[:16]}…).  Freeze a new protocol version "
                f"under a new filename instead of overwriting.")
        append_attempt(content_digest, "idempotent",
                       "same content; already frozen")
        print(f"already frozen (same content): {PROTOCOL_PATH}")
        print(f"protocol content sha256: {content_digest}")
        return payload
    PROTOCOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL_PATH.write_text(_serialize(payload))
    append_attempt(content_digest, "frozen", "first freeze")
    print(f"protocol frozen: {PROTOCOL_PATH}")
    print(f"protocol content sha256: {content_digest}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preview", action="store_true",
                    help="print the fit1/fit5 + confirm8 IDs and exit WITHOUT "
                         "freezing")
    ap.add_argument("--revise", action="store_true",
                    help="freeze a REVISION chained to a parent by "
                         "parent_freeze_sha256 (default: r1 chained to r0)")
    ap.add_argument("--parent", type=Path, default=PROTOCOL_PATH,
                    help="parent protocol to chain from (default: r0)")
    ap.add_argument("--out", type=Path, default=REVISION_PATH,
                    help="revision output path (default: protocol_v41_r1.json)")
    ap.add_argument("--n", type=int, default=1,
                    help="revision number (1 = r1; >=2 requires "
                         "--arm-b-lambda)")
    ap.add_argument("--arm-b-lambda", type=float, default=None,
                    help="frozen arm B/D lambda_dir (required for --n >=2; "
                         "bounds [1e-3, 1e3])")
    args = ap.parse_args()
    if args.revise:
        revise(parent=args.parent, out=args.out, n=args.n,
               arm_b_lambda=args.arm_b_lambda)
        return 0
    payload = build_payload()
    print("confirm8_v41:", payload["external_confirm"]["confirm8_v41"])
    print("fit5_group_ids:", payload["fit1_fit5_subsets"]["fit5_group_ids"])
    print("fit1_group_ids:", payload["fit1_fit5_subsets"]["fit1_group_ids"])
    if args.preview:
        return 0
    freeze()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
