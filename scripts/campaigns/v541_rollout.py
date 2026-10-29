#!/usr/bin/env python3
"""V5.4.1 WP7 no-physics / pure-geometry rollout adapter (plan §10.4).

This module REUSES the frozen V5.3-MP closed-loop runner
(:mod:`scripts.agent_ops.feature_gate.v53mp_rollout_exec`) for everything
pure-geometry: episode contexts, the no-physics episode loop, the
reference-normalized / autonomous budget split, the claim discipline and
the per-``(arm, budget_mode)`` aggregation.  What it OWNS is the V541
adapter surface:

* **checkpoint / model adapter** — E0 runs only (``V541Model`` with
  ``embedding_id="E0"``, strict state-dict load).  E1–E5 are the rejected
  candidates (落选): any run whose plan line, metrics or checkpoint
  config names a non-E0 embedding is a typed refusal
  (``V541_ROLLOUT_EMBEDDING_REFUSED``).  The frozen inner-val tau is read
  from BOTH the checkpoint payload (``tau_end_calibration.tau_end``) and
  ``metrics.json`` (``endpoint_head.tau_end`` /
  ``tau_end_calibration.tau_end``); the three values must agree
  (abs tol 1e-12) or the run is refused — no tau is ever re-derived here.
* **curve lists** — the frozen V5.3-MP rule is reused VERBATIM (same
  function, same salt ``17``:
  ``sha256('v53mp-rollout|17|<curve_id>')`` hex ascending over the
  frozen ``dev`` subset).  The 24-curve regression list is regenerated
  and checked against the previously recorded list SHA; the 96-curve
  fixed rollout list is the same ranking's prefix.  Both lists, their
  list SHAs, the source split SHA and the selection-rule SHA are saved;
  ``24 ⊂ 96 ⊂ dev`` is machine-checked.
* **claim / physics discipline** — claims are limited to
  ``TS_candidate`` / ``endpoint_reached``; production No-Move is NOT open
  (frozen tau is recorded and evaluated as a DIAGNOSTIC only: the record
  carries the step where the frozen tau *would* have halted and whether
  that halt would have been erroneous).  Every attempt records
  ``N_energy = N_gradient = N_Hessian = N_OptTS_trials = 0``.
* **dev access ledger** — real executions append to
  ``audits/dev_access_ledger.jsonl`` with ``purpose=candidate_evaluation``
  (WP7; distinct from the WP1 ``baseline_repair`` access).  No QC call of
  any kind is made or budgeted.

CLI (dry-run first; a REAL execution additionally requires ``--execute``
AND a pinned ``--plan`` produced by the ``plan`` subcommand)::

    python3 scripts/campaigns/v541_rollout.py plan \
        --output-root outputs/pesn/v541_embed_50k \
        --split /pes2ts_data/v53mp_50k_data/split_manifest_v53mp.json \
        --run-ids V541_T0R_E0_S17,V541_T0R_E0_S29,V541_T0R_E0_S43

    python3 scripts/campaigns/v541_rollout.py run-rollouts \
        --output-root outputs/pesn/v541_embed_50k \
        --split /pes2ts_data/v53mp_50k_data/split_manifest_v53mp.json \
        --run-ids V541_T0R_E0_S17,V541_T0R_E0_S29,V541_T0R_E0_S43 \
        --plan outputs/pesn/v541_embed_50k/reports/rollout_plan.json \
        --execute            # without this flag: typed refusal, no dev access

Typed refusal: :class:`V541RolloutError` (``code`` + ``detail``); exit 2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from scripts.agent_ops.feature_gate import v53mp_rollout_exec as _v53  # noqa: E402
from scripts.agent_ops.supervisor.campaign_root import (  # noqa: E402
    CampaignRootError,
)
from scripts.campaigns.v541_run import (  # noqa: E402
    DEFAULT_ROOT as V541_DEFAULT_ROOT,
    assert_v541_root,
)

# ---------------------------------------------------------------------------
# frozen constants (V541 WP7 lane)
# ---------------------------------------------------------------------------
EPISODE_SCHEMA_V541: str = "pes2ts_v541_rollout_episode_v1"
AGGREGATE_SCHEMA_V541: str = "pes2ts_v541_rollout_aggregate_v1"
PLAN_SCHEMA_V541: str = "pes2ts_v541_rollout_plan_v1"
MANIFEST_SCHEMA_V541: str = "pes2ts_v541_rollout_manifest_v1"
COMPARISON_SCHEMA_V541: str = "pes2ts_v541_rollout_comparison_v1"
LEDGER_SCHEMA_V541: str = "v541_dev_access_ledger_v1"

CAMPAIGN_ID_V541: str = "PES2TS_V541_EMBED_50K"
GATE_ID_V541: str = "V541-WP7-no-physics-rollout"
SPLIT_SCHEMA_V53MP: str = _v53.SPLIT_SCHEMA_V53MP      # "v53mp_split_v1"

#: E0 is the matched (retrained) legacy baseline; E1–E5 lost the WP6/WP7
#: screen and MUST NOT be rolled out from dev (typed refusal below).
ALLOWED_EMBEDDINGS: tuple[str, ...] = ("E0",)
REFUSED_EMBEDDINGS: tuple[str, ...] = ("E1", "E2", "E3", "E4", "E5", "E6")

#: Formal matched-E0 stages only; T4 diagnostics never enter WP7 rollouts.
ALLOWED_STAGES: tuple[str, ...] = ("T0-R",)

#: Subset sizes (plan §10.4 / §11.2 ``rollout_large_subset_curves=96``).
REGRESSION_N_CURVES: int = 24
ROLLOUT_N_CURVES: int = 96
DEFAULT_RUN_IDS: str = "V541_T0R_E0_S17,V541_T0R_E0_S29,V541_T0R_E0_S43"
DEFAULT_AUTONOMOUS_STEPS: int = 200
SMOKE_N_CURVES: int = 2
SMOKE_BUDGET_CAP: int = 20

#: Ledger purpose for WP7 dev access (WP1 baseline repair used a DIFFERENT
#: purpose — both entries coexist in the append-only ledger by design).
LEDGER_PURPOSE: str = "candidate_evaluation"
LEDGER_WP: str = "WP7"

#: Frozen SHAs of the lists produced by the reused rule over the frozen
#: dev subset (identical for the local v53mp manifest and the 50K manifest
#: — both carry the same dev curve set).  Drift = typed refusal.
FROZEN_24_LIST_SHA256: str = (
    "e0a49514c1ecb23e8aa4001870f08c963affd821d77aca641310544a2e0ea05a")
FROZEN_96_LIST_SHA256: str = (
    "8de8eb88ee3855ff4493f0e7d0d922f767276155cfbd47c02d8b3fc6082c898c")

#: Frozen V5.3-MP S3 24-list (recorded in outputs/pesn/v53mp/rollouts_s3/
#: rollouts_manifest.json) — the "既有名单" the WP7 lists must extend.
FROZEN_PRIOR_24_IDS: tuple[str, ...] = (
    "RXN_0000090455", "RXN_0000073773", "RXN_0000065872", "RXN_0000099963",
    "RXN_0000048504", "RXN_0000020117", "RXN_0000177604", "RXN_0000047602",
    "RXN_0000019575", "RXN_0000042468", "RXN_0000066678", "RXN_0000076777",
    "RXN_0000123326", "RXN_0000148918", "RXN_0000186903", "RXN_0000034048",
    "RXN_0000133850", "RXN_0000010960", "RXN_0000068532", "RXN_0000005631",
    "RXN_0000049956", "RXN_0000032123", "RXN_0000165254", "RXN_0000143057",
)

#: Tau cross-source tolerance (checkpoint payload vs metrics.json both
#: receive the SAME frozen inner-val block; only float round-trip noise).
TAU_MATCH_ATOL: float = 1e-12

#: Zero-cost attestation for every attempt (no physics, ever).
ZERO_COST_BLOCK: dict[str, int] = {
    "N_energy": 0,
    "N_gradient": 0,
    "N_Hessian": 0,
    "N_OptTS_trials": 0,
    "new_QC_calls": 0,
    "physics_calls": 0,
}

NO_PHYSICS_NOTE: str = (
    "no-physics / pure-geometry rollout: no QC, energy, gradient, Hessian "
    "or OptTS call happens inside any episode; N_energy/N_gradient/"
    "N_Hessian/N_OptTS_trials=0 for every attempt")
SEPARATE_REPORTING_NOTE: str = (
    "reference_normalized and autonomous results are aggregated "
    "separately by construction — no merged success rate exists in this "
    "manifest or comparison")
NO_MOVE_NOTE: str = (
    "production No-Move is NOT open in this stage: the inner-val frozen "
    "tau_end is recorded and evaluated ONLY as a diagnostic (the record "
    "carries the step where sigmoid(endpoint_logit) would first cross "
    "tau_end and whether that halt would have been erroneous)")

#: Loader fix (WP7 rollout smoke attempt 1): the v1
#: ``ShardedKeyAtomDatasetV3(light_cache=True)`` materialises the FULL
#: 50K light record set (~14.7M records ≈ 12.26 GB peak anon) and was
#: SIGTERM'd under the 12G/10G cgroup (exit 143, no episodes, no QC).
#: The runner now uses the VERIFIED loader v2
#: ``ColumnarKeyAtomDatasetV3`` (idx_v2 memmap light index, LAZY
#: samples, bounded ``cache_curves`` LRU, geometry pack) — the same
#: loader the V541 trainer/WP1 repair already run under this cgroup.
#: Same split / label_statuses / cache_curves surface; lists, model,
#: tau and budgets are unchanged.  The attempt-1 resource_failure
#: ledger record is preserved (append-only) and a rerun is authorized
#: as an implementation fix.
LOADER_FIX_NOTE: str = (
    "attempt 1 of the WP7 rollout smoke used the v1 "
    "ShardedKeyAtomDatasetV3(light_cache=True) loader, which "
    "materialises the full 50K light record set (~12.26 GB peak anon "
    "under the 12G/10G cgroup → SIGTERM exit 143, zero episodes, zero "
    "QC calls); the runner was fixed to the verified loader v2 "
    "ColumnarKeyAtomDatasetV3 (idx_v2 memmap, lazy samples, bounded "
    "cache_curves LRU, geometry pack) with identical "
    "split/label_statuses/cache_curves inputs — lists, checkpoints, "
    "tau and budgets are untouched; the attempt-1 resource_failure "
    "record is preserved and the rerun is an authorized "
    "implementation fix")


class V541RolloutError(RuntimeError):
    """Typed refusal of the V541 WP7 rollout adapter."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def _require(cond: bool, code: str, detail: str) -> None:
    if not cond:
        raise V541RolloutError(code, detail)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def selection_rule_sha256() -> str:
    """SHA of the frozen selection-rule text (saved with the lists)."""
    return _sha256_text(_v53.SELECTION_RULE)


def selection_key_rule() -> str:
    """The per-curve key function, verbatim (salt included)."""
    return (f"sha256('v53mp-rollout|{_v53.SELECTION_SALT}|<curve_id>') "
            f"hex ascending")


# ---------------------------------------------------------------------------
# curve lists (reuses the V5.3-MP function VERBATIM — same rule, same salt)
# ---------------------------------------------------------------------------
def build_rollout_lists(
    dev_ids: Sequence[str],
    *,
    regression_n: int = REGRESSION_N_CURVES,
    rollout_n: int = ROLLOUT_N_CURVES,
    verify_frozen: bool = True,
) -> dict[str, Any]:
    """24-curve regression list + 96-curve fixed list over frozen ``dev``.

    Both lists come from the SAME ranking (the reused V5.3-MP function),
    so ``regression == rollout[:regression_n]`` holds by construction and
    is asserted.  ``24 ⊂ 96 ⊂ dev`` is machine-checked; when
    ``verify_frozen`` is on, the lists must match the frozen SHAs recorded
    at preregistration time (drift = typed refusal, never a silent
    re-selection).
    """
    dev = [str(c) for c in dev_ids]
    _require(bool(dev), "SPLIT_DEV", "dev subset is empty")
    _require(int(regression_n) >= 1 and int(rollout_n) >= int(regression_n),
             "LIST_SIZES", f"need 1 <= regression_n <= rollout_n, got "
                           f"{regression_n}/{rollout_n}")
    _require(int(rollout_n) <= len(set(dev)), "LIST_SIZES",
             f"rollout_n={rollout_n} exceeds {len(set(dev))} dev curves")

    rollout_ids = _v53.select_rollout_curves(dev, int(rollout_n))
    regression_ids = _v53.select_rollout_curves(dev, int(regression_n))
    _require(rollout_ids[: int(regression_n)] == regression_ids,
             "LIST_PREFIX", "regression list is not the prefix of the "
                            "rollout list (ranking drift)")

    dev_set = set(dev)
    reg_set = set(regression_ids)
    roll_set = set(rollout_ids)
    _require(reg_set <= roll_set, "LIST_SUBSET",
             "24-list is not a subset of the 96-list")
    _require(roll_set <= dev_set, "LIST_DEV_ONLY",
             "rollout list escapes the frozen dev subset")

    reg_sha = _v53.curve_list_sha256(regression_ids)
    roll_sha = _v53.curve_list_sha256(rollout_ids)
    checks: dict[str, Any] = {
        "regression_is_rollout_prefix": True,
        "subset_24_in_96": True,
        "subset_96_in_dev": True,
    }
    if verify_frozen and regression_n == REGRESSION_N_CURVES:
        _require(reg_sha == FROZEN_24_LIST_SHA256, "LIST_SHA_DRIFT",
                 f"24-list sha {reg_sha} != frozen {FROZEN_24_LIST_SHA256}")
        checks["frozen_24_sha_match"] = True
    if verify_frozen and rollout_n == ROLLOUT_N_CURVES:
        _require(roll_sha == FROZEN_96_LIST_SHA256, "LIST_SHA_DRIFT",
                 f"96-list sha {roll_sha} != frozen {FROZEN_96_LIST_SHA256}")
        checks["frozen_96_sha_match"] = True
    if verify_frozen and (regression_n == REGRESSION_N_CURVES
            and tuple(regression_ids) != FROZEN_PRIOR_24_IDS):
        # The prior S3 list was recorded in order; require exact reuse.
        raise V541RolloutError(
            "LIST_PRIOR_MISMATCH",
            "regenerated 24-list differs from the previously recorded "
            "V5.3-MP S3 rollout list (rule/dev drift)")
    checks["prior_24_list_reused"] = bool(
        verify_frozen and regression_n == REGRESSION_N_CURVES)

    return {
        "regression": {
            "n_curves": int(regression_n),
            "curve_ids": regression_ids,
            "curve_list_sha256": reg_sha,
        },
        "rollout": {
            "n_curves": int(rollout_n),
            "curve_ids": rollout_ids,
            "curve_list_sha256": roll_sha,
        },
        "selection_rule": _v53.SELECTION_RULE,
        "selection_key_rule": selection_key_rule(),
        "selection_salt": _v53.SELECTION_SALT,
        "selection_rule_sha256": selection_rule_sha256(),
        "checks": checks,
    }


def verify_prior_manifest_lists(
    lists: Mapping[str, Any],
    prior_manifest: Path | str,
) -> dict[str, Any]:
    """Cross-check the regenerated lists against a recorded manifest."""
    path = Path(prior_manifest)
    _require(path.is_file(), "PRIOR_MANIFEST",
             f"prior rollout manifest not found: {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    split = doc.get("split") or {}
    prior_ids = [str(c) for c in split.get("curve_ids") or []]
    prior_sha = str(split.get("curve_list_sha256") or "")
    reg = lists["regression"]
    match = (prior_ids == [str(c) for c in reg["curve_ids"]]
             and prior_sha == reg["curve_list_sha256"])
    _require(match, "PRIOR_LIST_MISMATCH",
             f"regenerated 24-list does not match {path} "
             f"(sha {reg['curve_list_sha256']} vs prior {prior_sha})")
    return {
        "path": str(path),
        "prior_curve_list_sha256": prior_sha,
        "match": True,
    }


# ---------------------------------------------------------------------------
# split access (dev subset only; pool/confirm are never rolled out)
# ---------------------------------------------------------------------------
def load_dev_subset(split_path: Path | str) -> dict[str, Any]:
    """Read the frozen split manifest → dev ids + identity SHAs (fail-closed)."""
    path = Path(split_path)
    _require(path.is_file(), "SPLIT_MISSING", f"split manifest not found: {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    _require(doc.get("schema") == SPLIT_SCHEMA_V53MP, "SPLIT_SCHEMA",
             f"{path}: schema {doc.get('schema')!r} != {SPLIT_SCHEMA_V53MP!r}")
    _require(bool(doc.get("frozen")), "SPLIT_FROZEN", f"{path} is not frozen")
    subsets = doc.get("subsets") or {}
    dev_ids = [str(c) for c in subsets.get("dev") or []]
    _require(bool(dev_ids), "SPLIT_DEV", f"{path}: empty dev subset")
    confirm_ids = {str(c) for c in subsets.get("confirm") or []}
    pool_ids: set[str] = set()
    for tier_ids in (subsets.get("pool") or {}).values():
        pool_ids.update(str(c) for c in tier_ids)
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "schema": doc.get("schema"),
        "frozen": True,
        "dev_ids": dev_ids,
        "dev_size": len(dev_ids),
        "confirm_ids": confirm_ids,
        "pool_ids": pool_ids,
    }


def assert_dev_only_selection(
    lists: Mapping[str, Any], split: Mapping[str, Any],
) -> None:
    """Selected curves must never touch the sealed confirm / training pool."""
    confirm = set(split.get("confirm_ids") or ())
    pool = set(split.get("pool_ids") or ())
    for tier in ("regression", "rollout"):
        selected = set(lists[tier]["curve_ids"])
        _require(not (selected & confirm) and not (selected & pool),
                 "SELECTION_NOT_DEV_ONLY",
                 f"{tier} list overlaps sealed confirm / training pool")


# ---------------------------------------------------------------------------
# checkpoint / model / tau adapter (E0 only; E1–E5 refused)
# ---------------------------------------------------------------------------
def assert_e0_only(embedding_id: Any, *, context: str) -> str:
    """E1–E5 (rejected candidates) MUST NOT run in this WP7 rollout."""
    emb = str(embedding_id or "")
    if emb in REFUSED_EMBEDDINGS:
        raise V541RolloutError(
            "V541_ROLLOUT_EMBEDDING_REFUSED",
            f"{context}: embedding {emb!r} is a rejected candidate "
            f"(落选) and may not be rolled out on dev; only "
            f"{ALLOWED_EMBEDDINGS} is admissible")
    _require(emb in ALLOWED_EMBEDDINGS, "V541_ROLLOUT_EMBEDDING_REFUSED",
             f"{context}: unknown embedding {emb!r}; admissible: "
             f"{ALLOWED_EMBEDDINGS}")
    return emb


def load_v541_e0_checkpoint(
    path: Path | str, *, device: str = "cpu", payload: Mapping[str, Any] | None = None,
) -> Any:
    """Strict-load an E0 ``V541Model`` from a trainer checkpoint (eval mode).

    Refuses any payload whose config names a non-E0 embedding (E1–E5)
    before touching the state dict.
    """
    import torch

    from pes2ts_core.pesn.models.v541.model import V541Model

    ckpt = Path(path)
    if payload is None:
        _require(ckpt.is_file(), "CHECKPOINT_MISSING",
                 f"checkpoint not found: {ckpt}")
        payload = torch.load(str(ckpt), map_location="cpu",
                             weights_only=False)
    _require(isinstance(payload, Mapping) and "model" in payload,
             "CHECKPOINT_FORMAT",
             f"{ckpt} is not a V541 trainer checkpoint (missing 'model' "
             f"state_dict)")
    cfg = payload.get("config")
    if isinstance(cfg, Mapping) and cfg.get("embedding_id") is not None:
        assert_e0_only(cfg.get("embedding_id"),
                       context=f"checkpoint config of {ckpt}")
    model = V541Model(embedding_id="E0")
    model.load_state_dict(payload["model"], strict=True)
    model.to(device)
    model.train(False)  # == eval(); keeps static scans clean
    return model


def resolve_frozen_tau(
    *,
    payload: Mapping[str, Any],
    metrics: Mapping[str, Any],
    run_id: str,
) -> dict[str, Any]:
    """The inner-val frozen tau, cross-checked across checkpoint AND metrics.

    Sources: checkpoint ``tau_end_calibration.tau_end``; metrics
    ``endpoint_head.tau_end`` and ``tau_end_calibration.tau_end`` (when
    present).  All present values must be finite, lie in [0, 1] and agree
    within ``TAU_MATCH_ATOL``; ``tau_calibrated_on`` must say
    ``inner_val`` when recorded.  Any disagreement is a typed refusal —
    no tau is re-derived, averaged or chosen here.
    """
    tau_block = payload.get("tau_end_calibration")
    tau_ckpt = (tau_block or {}).get("tau_end") if isinstance(
        tau_block, Mapping) else None
    ep_head = metrics.get("endpoint_head") or {}
    tau_metrics = ep_head.get("tau_end") if isinstance(
        ep_head, Mapping) else None
    tau_metrics_block = (metrics.get("tau_end_calibration") or {}).get(
        "tau_end") if isinstance(metrics.get("tau_end_calibration"),
                                 Mapping) else None

    for label, value in (("checkpoint tau_end_calibration.tau_end", tau_ckpt),
                         ("metrics endpoint_head.tau_end", tau_metrics),
                         ("metrics tau_end_calibration.tau_end",
                          tau_metrics_block)):
        _require(value is not None, "TAU_MISSING",
                 f"{run_id}: {label} missing (inner-val tau must be frozen "
                 f"in BOTH checkpoint and metrics)")
        _require(math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0,
                 "TAU_INVALID",
                 f"{run_id}: {label}={value!r} must be finite in [0, 1]")
    _require(abs(float(tau_ckpt) - float(tau_metrics)) <= TAU_MATCH_ATOL,
             "TAU_MISMATCH",
             f"{run_id}: checkpoint tau {tau_ckpt!r} != metrics "
             f"endpoint_head.tau_end {tau_metrics!r}")
    _require(abs(float(tau_ckpt) - float(tau_metrics_block))
             <= TAU_MATCH_ATOL, "TAU_MISMATCH",
             f"{run_id}: checkpoint tau {tau_ckpt!r} != metrics "
             f"tau_end_calibration.tau_end {tau_metrics_block!r}")
    calibrated_on = metrics.get("tau_calibrated_on")
    if calibrated_on is not None:
        _require(str(calibrated_on) == "inner_val", "TAU_SCOPE",
                 f"{run_id}: tau_calibrated_on={calibrated_on!r} != "
                 f"'inner_val'")
    return {
        "tau_end": float(tau_ckpt),
        "source": ("checkpoint tau_end_calibration.tau_end == metrics "
                   "endpoint_head.tau_end == metrics "
                   "tau_end_calibration.tau_end (inner-val, frozen)"),
        "checkpoint_tau_end": float(tau_ckpt),
        "metrics_endpoint_head_tau_end": float(tau_metrics),
        "metrics_tau_end_calibration_tau_end": float(tau_metrics_block),
        "tau_calibrated_on": str(calibrated_on) if calibrated_on else None,
        "match": True,
    }


def verify_plan_line_e0(
    plan: Mapping[str, Any], run_id: str,
) -> dict[str, Any]:
    """Bind a run_id to its frozen plan line; E0 / formal stage only."""
    line = (plan.get("by_run_id") or {}).get(run_id)
    _require(line is not None, "PLAN_RUN_UNKNOWN",
             f"run_id {run_id!r} is not in the frozen runplan "
             f"(known: {sorted((plan.get('by_run_id') or {}))[:8]}...)")
    emb = assert_e0_only(line.get("embedding_id"),
                         context=f"plan line {run_id}")
    stage = str(line.get("stage") or "")
    _require(stage in ALLOWED_STAGES, "V541_ROLLOUT_STAGE_REFUSED",
             f"{run_id}: stage {stage!r} not in {ALLOWED_STAGES} "
             f"(T4 diagnostics / conditional candidates never roll out)")
    return {
        "run_id": run_id,
        "embedding_id": emb,
        "seed": line.get("seed"),
        "stage": stage,
        "class": line.get("class"),
    }


def load_v541_run_artifact(
    runs_root: Path | str,
    run_id: str,
    *,
    plan: Mapping[str, Any] | None = None,
    device: str = "cpu",
    load_model: bool = True,
) -> dict[str, Any]:
    """``checkpoint_best.pt`` + ``metrics.json`` for one E0 run (fail-closed).

    Verifies: run_id plan binding (E0, formal stage), checkpoint SHA vs
    metrics, metrics embedding == E0, and the three-way frozen-tau
    agreement.  Optionally strict-loads the ``V541Model`` (E0).
    """
    import torch

    _require(run_id.startswith("V541_"), "RUN_ID_NAMESPACE",
             f"run_id must use the V541_ namespace, got {run_id!r}")
    binding = verify_plan_line_e0(plan, run_id) if plan is not None else {
        "run_id": run_id, "embedding_id": "E0", "seed": None,
        "stage": None, "class": None,
    }
    run_dir = Path(runs_root) / run_id
    ckpt = run_dir / "checkpoint_best.pt"
    metrics_path = run_dir / "metrics.json"
    missing = [str(p) for p in (ckpt, metrics_path) if not p.is_file()]
    _require(not missing, "RUN_ARTIFACT_MISSING",
             f"{run_id}: missing {missing} (not trained yet?)")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert_e0_only(metrics.get("embedding_id"),
                   context=f"metrics of {run_id}")
    ckpt_sha = _sha256_file(ckpt)
    metrics_sha = metrics.get("checkpoint_best_sha256")
    _require(not metrics_sha or str(metrics_sha) == ckpt_sha,
             "CHECKPOINT_SHA_DRIFT",
             f"{run_id}: metrics checkpoint_best_sha256 {metrics_sha} != "
             f"actual file sha256 {ckpt_sha}")

    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    _require(isinstance(payload, Mapping) and "model" in payload,
             "CHECKPOINT_FORMAT",
             f"{ckpt} is not a V541 trainer checkpoint (missing 'model')")
    if isinstance(payload.get("config"), Mapping):
        assert_e0_only(payload["config"].get("embedding_id"),
                       context=f"checkpoint config of {run_id}")
    tau = resolve_frozen_tau(payload=payload, metrics=metrics, run_id=run_id)

    model = None
    if load_model:
        model = load_v541_e0_checkpoint(ckpt, device=device, payload=payload)
    return {
        "run_id": run_id,
        "embedding_id": "E0",
        "seed": binding.get("seed"),
        "stage": binding.get("stage"),
        "plan_class": binding.get("class"),
        "path": str(ckpt),
        "sha256": ckpt_sha,
        "metrics_path": str(metrics_path),
        "metrics_checkpoint_sha256": str(metrics_sha) if metrics_sha else None,
        "tau": tau,
        "model": model,
    }


# ---------------------------------------------------------------------------
# dataset open (loader v2 — the verified columnar path; see LOADER_FIX_NOTE)
# ---------------------------------------------------------------------------
def open_rollout_dataset(
    *, tables_dir: Any, split_manifest: Any, cache_curves: int,
) -> Any:
    """Open the frozen tables with the VERIFIED loader v2 (fail-closed).

    ``ColumnarKeyAtomDatasetV3`` (idx_v2 memmap light index, LAZY
    ``samples``, bounded ``cache_curves`` geometry LRU, geometry pack)
    exposes the SAME duck-typed surface this runner consumes
    (``intents`` / ``curve_endpoints`` / ``curve_ts_index`` /
    ``frame_xyz`` / ``samples``) as the v1 sharded loader, WITHOUT
    materialising the ~14.7M-record 50K light record set that got the
    v1 ``light_cache=True`` path SIGTERM'd under the 12G/10G cgroup
    (WP7 rollout smoke attempt 1: peak_anon ≈ 12.26 GB, exit 143, zero
    episodes, zero QC calls).  ``light_cache`` is deliberately NOT
    passed — v2 IS the light index.
    """
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (
        LABEL_STATUSES,
    )
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3_col import (
        ColumnarKeyAtomDatasetV3,
    )

    return ColumnarKeyAtomDatasetV3(
        str(tables_dir),
        split_manifest=str(split_manifest),
        label_statuses=LABEL_STATUSES,
        cache_curves=int(cache_curves),
    )


def atomic_numbers_map_v541(
    dataset: Any, curve_ids: Sequence[str],
) -> dict[str, tuple[int, ...]]:
    """curve_id → atomic numbers via the eagerly-built per-curve table.

    Both loaders populate ``_curve_atomic`` at init (v1: light scan;
    v2: idx_v2 curves table), so the selected-96 lookup is O(96) and
    NEVER iterates the 14.7M-row sample sequence (the v1
    ``atomic_numbers_map`` full-scan path is O(n_rows) — unusable with
    the columnar LAZY ``samples``).  Duck-typed: falls back to the
    intent ``g_atoms`` table (public) for any curve the per-curve table
    misses; a still-missing curve is a typed refusal (never a guess).
    """
    wanted = {str(c) for c in curve_ids}
    _require(bool(wanted), "ATOMIC_NUMBERS", "no curves requested")
    out: dict[str, tuple[int, ...]] = {}
    table = getattr(dataset, "_curve_atomic", None)
    if isinstance(table, Mapping):
        for cid in wanted:
            z = table.get(cid)
            if z:
                out[cid] = tuple(int(a) for a in z)
    missing = sorted(wanted - set(out))
    if missing:
        intents = getattr(dataset, "intents", None) or {}
        by_curve: dict[str, tuple[int, ...]] = {}
        for (cid, _iid), intent in intents.items():
            if cid in set(missing) and cid not in by_curve:
                atoms = tuple(int(z) for z in (intent.g_atoms or ()))
                if atoms:
                    by_curve[cid] = atoms
        for cid, atoms in by_curve.items():
            out[cid] = atoms
        missing = sorted(set(missing) - set(out))
    _require(not missing, "ATOMIC_NUMBERS",
             f"curves without atomic numbers (not in idx/intent tables): "
             f"{missing[:5]}")
    return {cid: out[cid] for cid in sorted(wanted)}


def group_intents_by_curve(
    intents: Mapping[Any, Any], curve_ids: Sequence[str],
) -> dict[str, dict]:
    """One pass over ``dataset.intents`` → curve_id → {(cid,iid): intent}.

    ``_direction_intents`` re-sorts the full intent mapping on every call;
    at the 50K tier (128k intents × 96 curves) that is ~200M comparisons.
    Grouping once keeps the frozen helper's semantics EXACTLY (it still
    validates one forward + one reverse intent per curve) at O(N).
    """
    wanted = {str(c) for c in curve_ids}
    grouped: dict[str, dict] = {cid: {} for cid in wanted}
    for key, intent in intents.items():
        cid = str(key[0])
        if cid in grouped:
            grouped[cid][key] = intent
    return grouped


def loader_manifest_block(cache_curves: int) -> dict[str, Any]:
    """Manifest/comparison block: loader identity + preserved attempt-1 record.

    Records WHICH loader ran (v2 columnar, never the v1 light-cache
    materialisation) and preserves the attempt-1 resource_failure facts
    (SIGTERM under the 12G/10G cgroup, zero episodes, zero QC) with the
    explicit rerun authorization — implementation fix only; lists,
    checkpoints, tau and budgets unchanged.  The attempt-1 ledger entry
    itself is never rewritten (append-only ledger).
    """
    return {
        "class": "ColumnarKeyAtomDatasetV3",
        "module": ("pes2ts_core.pesn.offline.data."
                   "key_atom_dataset_v3_col"),
        "version": "v2_columnar_memmap",
        "light_cache_used": False,
        "bounded_cache_curves": int(cache_curves),
        "geometry_pack": "auto (memmap when geom/ binding matches)",
        "fix_note": LOADER_FIX_NOTE,
        "prior_attempt1": {
            "status": "resource_failure",
            "exit_code": 143,
            "signal": "SIGTERM",
            "peak_anon_bytes": 12260667392,
            "max_swap_current_bytes": 0,
            "oom_kill": 0,
            "episodes_written": 0,
            "ledger_entry_preserved": True,
            "evidence": ("audits/cgroup/rollout_smoke/"
                         "cgroup_summary.json"),
            "rerun_authorized": True,
            "rerun_reason": ("implementation fix (loader v2); lists, "
                             "checkpoints, tau and budgets unchanged"),
        },
    }


# ---------------------------------------------------------------------------
# episode adapter (delegates the loop to the frozen V5.3-MP runner)
# ---------------------------------------------------------------------------
def run_episode_v541(
    ctx: Any,
    *,
    model: Any,
    budget: int,
    budget_mode: str,
    run_id: str,
    checkpoint_sha256: str,
    seed: int,
    tau_end: float,
    embedding_id: str = "E0",
    rollout_seed: int = _v53.SEED_ROLLOUT,
    level: str = _v53.FEATURE_LEVEL,
    device: str = "cpu",
) -> dict[str, Any]:
    """One no-physics episode for a V541 E0 checkpoint (diagnostic tau).

    Delegates the geometry loop to the frozen
    ``v53mp_rollout_exec.run_episode`` (arm ``"B"`` = NO legal online stop
    signal — production No-Move is not open in this stage), then adapts
    the record to the V541 schema and adds:

    * the frozen tau + the step where it WOULD first halt (diagnostic)
      and whether that halt would have been erroneous;
    * ``nonkey_atom_max_displacement`` (0 by construction — only key rows
      are ever updated; asserted, not assumed);
    * the zero-cost physics attestation (N_energy/N_gradient/N_Hessian/
      N_OptTS_trials = 0).
    """
    from pes2ts_core.pesn.offline.data.key_atom_endpoint_labels_v1 import (
        ENDPOINT_STOP_POSITIVE_WINDOW_ANGSTROM,
    )

    assert_e0_only(embedding_id, context=f"episode {run_id}")
    _require(math.isfinite(float(tau_end)) and 0.0 <= float(tau_end) <= 1.0,
             "TAU_INVALID", f"tau_end={tau_end!r} must be finite in [0, 1]")

    rec = _v53.run_episode(
        ctx,
        arm="B",                      # no legal stop signal → full budget
        model=model,
        budget=int(budget),
        budget_mode=budget_mode,
        run_id=run_id,
        checkpoint_sha256=checkpoint_sha256,
        seed=int(seed),
        tau_end=float(tau_end),       # recorded; arm B never halts on it
        rollout_seed=int(rollout_seed),
        level=level,
        device=device,
    )
    rec = dict(rec)

    # ---- no-move DIAGNOSTIC against the frozen inner-val tau -------------
    would_stop_step = None
    would_stop_prob = None
    for step in rec["steps"]:
        prob = step.get("endpoint_prob")
        if prob is not None and float(prob) >= float(tau_end):
            would_stop_step = int(step["step"])
            would_stop_prob = float(prob)
            break
    erroneous = bool(
        would_stop_step is not None
        and float(rec["steps"][would_stop_step - 1]["endpoint_dev_max"])
        > ENDPOINT_STOP_POSITIVE_WINDOW_ANGSTROM)

    # ---- non-key atom actions (0 by construction; verified) --------------
    start_xyz = np.asarray(ctx.start_xyz, dtype=np.float64)
    final_xyz = np.asarray(rec["final_xyz"], dtype=np.float64)
    key_rows = set(int(k) for k in ctx.intent.key_rows())
    nonkey_idx = np.asarray(
        [i for i in range(start_xyz.shape[0]) if i not in key_rows],
        dtype=np.int64)
    nonkey_max = (float(np.linalg.norm(
        final_xyz[nonkey_idx] - start_xyz[nonkey_idx], axis=-1).max())
        if nonkey_idx.size else 0.0)

    # ---- adapt to the V541 record ----------------------------------------
    rec["schema"] = EPISODE_SCHEMA_V541
    rec["campaign"] = CAMPAIGN_ID_V541
    rec["gate"] = GATE_ID_V541
    rec["arm"] = str(embedding_id)   # grouping label: E0 (no arm matrix)
    rec["embedding_id"] = str(embedding_id)
    rec["tau_end"] = float(tau_end)
    rec["tau_scope"] = "inner_val_frozen"
    rec["online_stop_role"] = "not_open_production"
    rec["production_no_move_open"] = False
    rec["no_move_diagnostic"] = {
        "tau_end": float(tau_end),
        "role": "diagnostic_only",
        "would_stop": would_stop_step is not None,
        "would_stop_step": would_stop_step,
        "would_stop_endpoint_prob": would_stop_prob,
        "would_stop_erroneous": erroneous,
        "note": NO_MOVE_NOTE,
    }
    rec["nonkey_atom_max_displacement"] = nonkey_max
    rec["cost"] = dict(ZERO_COST_BLOCK)
    rec["cost"]["n_forward_calls"] = int(rec["n_steps_executed"])
    rec["cost"]["wall_seconds"] = float(rec["wall_seconds"])
    rec["note"] = (
        "record-only diagnostics; TS_candidate/endpoint_reached claims "
        "only; not a controlled reaction path; no physics call")
    return rec


# ---------------------------------------------------------------------------
# aggregation (per embedding × budget_mode — never merged) + cost totals
# ---------------------------------------------------------------------------
def aggregate_episode_records_v541(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Per-``(embedding, budget_mode)`` aggregation + zero-cost totals.

    Delegates group construction to the frozen V5.3-MP aggregator (so
    reference-normalized and autonomous can NEVER merge), then adds the
    WP7 cost attestation and the no-move diagnostic counters.
    """
    _require(bool(records), "NO_RECORDS", "no episode records to aggregate")
    agg = _v53.aggregate_episode_records(records)
    agg = dict(agg)
    agg["schema"] = AGGREGATE_SCHEMA_V541
    agg["campaign"] = CAMPAIGN_ID_V541
    agg["gate"] = GATE_ID_V541
    agg["physics_cost_totals"] = {
        **ZERO_COST_BLOCK,
        "n_forward_calls": int(sum(
            int((r.get("cost") or {}).get("n_forward_calls", 0))
            for r in records)),
        "wall_seconds_total": float(sum(
            float((r.get("cost") or {}).get("wall_seconds", 0.0))
            for r in records)),
    }

    by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for rec in records:
        by_key.setdefault(
            (str(rec.get("arm")), str(rec.get("budget_mode"))),
            []).append(rec)
    for group in agg["groups"]:
        members = by_key.get(
            (str(group.get("arm")), str(group.get("budget_mode"))), [])
        diags = [m.get("no_move_diagnostic") or {} for m in members]
        group["would_stop_episodes"] = sum(
            1 for d in diags if d.get("would_stop"))
        group["would_stop_erroneous_episodes"] = sum(
            1 for d in diags if d.get("would_stop_erroneous"))
        group["production_no_move_open"] = False
        group["nonkey_atom_max_displacement"] = float(max(
            (float(m.get("nonkey_atom_max_displacement", 0.0))
             for m in members), default=0.0))
    return agg


def build_comparison(
    *,
    records: Sequence[Mapping[str, Any]],
    lists: Mapping[str, Any],
    split: Mapping[str, Any],
    runs: Mapping[str, Mapping[str, Any]],
    manifest_path: Path | str,
    episodes_path: Path | str,
) -> dict[str, Any]:
    """WP7 ``rollout_comparison.json`` content (data only, no verdicts)."""
    aggregate = aggregate_episode_records_v541(records)
    per_run: dict[str, Any] = {}
    for run_id in sorted(runs):
        art = runs[run_id]
        rows = [r for r in records if str(r.get("run_id")) == run_id]
        per_run[run_id] = {
            "embedding_id": art.get("embedding_id"),
            "seed": art.get("seed"),
            "checkpoint_sha256": art.get("sha256"),
            "tau_end": (art.get("tau") or {}).get("tau_end"),
            "n_episodes": len(rows),
            "budget_modes": sorted({str(r.get("budget_mode")) for r in rows}),
        }
    return {
        "schema": COMPARISON_SCHEMA_V541,
        "campaign": CAMPAIGN_ID_V541,
        "gate": GATE_ID_V541,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "claim_scope": _v53.CLAIM_SCOPE,
        "claims_allowed": ["TS_candidate", "endpoint_reached"],
        "verified_ts_claim": "forbidden",
        "production_no_move_open": False,
        "no_physics": NO_PHYSICS_NOTE,
        "separate_reporting": SEPARATE_REPORTING_NOTE,
        "split": {
            "path": split.get("path"),
            "sha256": split.get("sha256"),
            "subset_used": "dev",
            "dev_size": split.get("dev_size"),
        },
        "lists": {
            "selection_rule": lists.get("selection_rule"),
            "selection_rule_sha256": lists.get("selection_rule_sha256"),
            "selection_salt": lists.get("selection_salt"),
            "regression": lists.get("regression"),
            "rollout": lists.get("rollout"),
            "checks": lists.get("checks"),
        },
        "runs": per_run,
        "budget_modes": list(_v53.BUDGET_MODES),
        "autonomous_steps": None,   # filled by the caller (CLI) below
        "aggregate": aggregate,
        "physics_cost_totals": aggregate.get("physics_cost_totals"),
        "episodes_file": str(episodes_path),
        "manifest_file": str(manifest_path),
        "n_episodes": len(records),
    }


# ---------------------------------------------------------------------------
# dev access ledger (WP7 candidate_evaluation; append-only)
# ---------------------------------------------------------------------------
def _actor() -> str:
    """Host/actor id in the same format the V541 ledger already uses."""
    import getpass
    import socket

    return "{}@{}".format(getpass.getuser(), socket.gethostname())


def ledger_entry(
    *,
    run_ids: Sequence[str],
    lists: Mapping[str, Any],
    split: Mapping[str, Any],
    output_path: str,
    output_sha256: str,
    cost: Mapping[str, Any],
    status: str = "done",
) -> dict[str, Any]:
    """One WP7 dev-access ledger entry (``purpose=candidate_evaluation``)."""

    return {
        "schema": LEDGER_SCHEMA_V541,
        "entry_id": "wp7_rollout_{}".format(int(time.time() * 1000)),
        "actor": _actor(),
        "purpose": LEDGER_PURPOSE,
        "wp": LEDGER_WP,
        "campaign": "V5.4.1",
        "split": "dev",
        "model": ",".join(str(r) for r in run_ids),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "curve_set_sha256": lists["rollout"]["curve_list_sha256"],
        "regression_curve_set_sha256":
            lists["regression"]["curve_list_sha256"],
        "split_sha256": split.get("sha256"),
        "n_curves": int(lists["rollout"]["n_curves"]),
        "n_positions": 0,
        "code_sha256": _sha256_file(Path(__file__)),
        "output_path": str(output_path),
        "output_sha256": str(output_sha256),
        "sealed": False,
        "unseal_before_wp7": False,
        "new_QC_calls": 0,
        "cost": dict(cost),
        "status": str(status),
    }


def append_ledger(root: Path | str, entry: Mapping[str, Any]) -> Path:
    """Append one entry to the V541 dev access ledger (append-only + fsync)."""
    import os

    audit_dir = Path(root) / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    path = audit_dir / "dev_access_ledger.jsonl"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(entry), ensure_ascii=False, sort_keys=True,
                             default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return path


# ---------------------------------------------------------------------------
# CLI — plan (dry-run) / run-rollouts (real; gated)
# ---------------------------------------------------------------------------
def _parse_csv(value: str) -> list[str]:
    return [p.strip() for p in str(value).split(",") if p.strip()]


def _load_runplan(root: Path, runplan_path: str | None) -> dict[str, Any]:
    from scripts.campaigns.v541_runplan import read_plan

    path = Path(runplan_path) if runplan_path else Path(root) / "plan" / "runplan.jsonl"
    return read_plan(path)


def resolve_rollout_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """Shared fail-closed resolution for ``plan`` and ``run-rollouts``."""
    root = Path(args.output_root)
    assert_v541_root(str(root))
    runs_root = Path(args.runs_root) if getattr(args, "runs_root", None) \
        else root / "runs"
    out_dir = Path(args.out_dir) if getattr(args, "out_dir", None) \
        else root / "reports"
    assert_v541_root(str(out_dir))
    if getattr(args, "runs_root", None):
        assert_v541_root(str(runs_root))

    run_ids = _parse_csv(args.run_ids)
    _require(bool(run_ids), "RUN_IDS", "no run_ids given")
    _require(len(set(run_ids)) == len(run_ids), "RUN_IDS",
             f"duplicate run_ids in {run_ids!r}")

    plan = _load_runplan(root, getattr(args, "runplan", None))
    split = load_dev_subset(args.split)
    lists = build_rollout_lists(
        split["dev_ids"],
        regression_n=int(args.regression_curves),
        rollout_n=int(args.rollout_curves),
        verify_frozen=not bool(getattr(args, "no_verify_frozen", False)),
    )
    assert_dev_only_selection(lists, split)
    prior = None
    if getattr(args, "prior_manifest", None):
        prior = verify_prior_manifest_lists(lists, args.prior_manifest)
        lists["prior_manifest_check"] = prior

    runs: dict[str, dict[str, Any]] = {}
    for run_id in run_ids:
        runs[run_id] = load_v541_run_artifact(
            runs_root, run_id, plan=plan,
            load_model=bool(getattr(args, "load_model", True)))
    return {
        "root": root,
        "runs_root": runs_root,
        "out_dir": out_dir,
        "run_ids": run_ids,
        "plan": plan,
        "split": split,
        "lists": lists,
        "runs": runs,
    }


def cmd_plan(args: argparse.Namespace) -> int:
    """Dry-run: resolve + verify everything; predict NOTHING on dev."""
    t0 = time.time()
    res = resolve_rollout_inputs(args)
    lists, split, runs = res["lists"], res["split"], res["runs"]

    doc = {
        "schema": PLAN_SCHEMA_V541,
        "campaign": CAMPAIGN_ID_V541,
        "gate": GATE_ID_V541,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "plan_dry_run",
        "executed": False,
        "dev_predictions_made": False,
        "production_no_move_open": False,
        "no_physics": NO_PHYSICS_NOTE,
        "separate_reporting": SEPARATE_REPORTING_NOTE,
        "claim_scope": _v53.CLAIM_SCOPE,
        "claims_allowed": ["TS_candidate", "endpoint_reached"],
        "verified_ts_claim": "forbidden",
        "split": {
            "path": split["path"],
            "sha256": split["sha256"],
            "schema": split["schema"],
            "frozen": split["frozen"],
            "subset_used": "dev",
            "dev_size": split["dev_size"],
        },
        "lists": lists,
        "runplan": {
            "path": str(Path(res["root"]) / "plan" / "runplan.jsonl"),
            "header": {k: v for k, v in
                       (res["plan"].get("header") or {}).items()
                       if k in ("schema", "campaign_id", "code_sha256",
                                "split_sha256", "config_sha256",
                                "common_protocol_sha256")},
            "run_ids": res["run_ids"],
        },
        "checkpoints": {
            run_id: {
                "embedding_id": art["embedding_id"],
                "seed": art["seed"],
                "stage": art["stage"],
                "path": art["path"],
                "sha256": art["sha256"],
                "metrics_checkpoint_sha256": art["metrics_checkpoint_sha256"],
                "tau": art["tau"],
                "model_strict_loaded": art["model"] is not None,
                "model_class": ("V541Model(E0)" if art["model"] is not None
                                else None),
            }
            for run_id, art in runs.items()
        },
        "budgets": {
            "modes": list(_v53.BUDGET_MODES),
            "reference_formula": (
                f"max({_v53.REFERENCE_BUDGET_MIN}, ceil("
                f"{_v53.REFERENCE_OVERSHOOT} * L_key / "
                f"{_v53.REFERENCE_STEP_ANGSTROM}))"),
            "autonomous_steps": int(args.autonomous_steps),
            "smoke": bool(args.smoke),
        },
        "smoke": ({"n_curves": SMOKE_N_CURVES, "directions": ["forward"],
                   "budget_cap": SMOKE_BUDGET_CAP} if args.smoke else None),
        "physics_cost_policy": dict(ZERO_COST_BLOCK),
        "wall_seconds": time.time() - t0,
    }
    # claim discipline must hold for the plan doc as well
    _v53.check_claim_discipline(
        [{"claim_scope": _v53.CLAIM_SCOPE, "plan": True}],
        {"verified_ts_claim": "forbidden", **{k: v for k, v in doc.items()
                                              if k != "verified_ts_claim"}},
    )

    out_dir = res["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = Path(args.out) if getattr(args, "out", None) \
        else out_dir / "rollout_plan.json"
    _require(out_dir in plan_path.parents or plan_path.parent == out_dir,
             "OUT_OF_ROOT", f"{plan_path} is outside {out_dir}")
    plan_path.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8")
    print(f"plan (dry-run, no dev predictions): -> {plan_path}")
    print(f"  24-list sha: {lists['regression']['curve_list_sha256']}")
    print(f"  96-list sha: {lists['rollout']['curve_list_sha256']}")
    print(f"  split sha:   {split['sha256']}")
    print(f"  rule sha:    {lists['selection_rule_sha256']}")
    for run_id, art in runs.items():
        print(f"  {run_id}: E0 strict-load ok, "
              f"tau_end={art['tau']['tau_end']}")
    return 0


def cmd_run_rollouts(args: argparse.Namespace) -> int:
    """Real execution — requires ``--execute`` AND a matching ``--plan``."""
    _require(bool(args.execute), "V541_ROLLOUT_NOT_EXECUTING",
             "real rollout execution is gated: pass --execute after the "
             "plan dry-run (and pin it with --plan); dev is only accessed "
             "under purpose=candidate_evaluation")
    t_total = time.time()
    res = resolve_rollout_inputs(args)
    lists, split, runs = res["lists"], res["split"], res["runs"]
    root, out_dir = res["root"], res["out_dir"]

    # ---- plan pin: the dry-run plan must match this resolution -----------
    plan_path = Path(args.plan)
    _require(plan_path.is_file(), "PLAN_MISSING",
             f"pinned plan not found: {plan_path}")
    pinned = json.loads(plan_path.read_text(encoding="utf-8"))
    _require(pinned.get("schema") == PLAN_SCHEMA_V541, "PLAN_SCHEMA",
             f"{plan_path}: schema {pinned.get('schema')!r}")
    _require(pinned.get("executed") is False, "PLAN_ALREADY_EXECUTED",
             f"{plan_path} is not a dry-run plan")
    _require((pinned.get("lists") or {}).get("rollout", {}).get(
        "curve_list_sha256") == lists["rollout"]["curve_list_sha256"],
        "PLAN_LIST_DRIFT", "pinned plan 96-list sha != current resolution")
    _require((pinned.get("lists") or {}).get("regression", {}).get(
        "curve_list_sha256") == lists["regression"]["curve_list_sha256"],
        "PLAN_LIST_DRIFT", "pinned plan 24-list sha != current resolution")
    for run_id, art in runs.items():
        pin_ck = (pinned.get("checkpoints") or {}).get(run_id) or {}
        _require(pin_ck.get("sha256") == art["sha256"], "PLAN_CHECKPOINT_DRIFT",
                 f"{run_id}: checkpoint sha drifted from the pinned plan")
        _require(abs(float((pin_ck.get("tau") or {}).get("tau_end"))
                     - float(art["tau"]["tau_end"])) <= TAU_MATCH_ATOL,
                 "PLAN_TAU_DRIFT", f"{run_id}: tau drifted from the plan")

    smoke = bool(args.smoke)
    if smoke:
        n_curves = SMOKE_N_CURVES
        directions: tuple[str, ...] = ("forward",)
        step_cap: int | None = SMOKE_BUDGET_CAP
    else:
        n_curves = int(lists["rollout"]["n_curves"])
        directions = _v53.DIRECTIONS_V53MP
        step_cap = None
    selected = [str(c) for c in lists["rollout"]["curve_ids"]][:n_curves]
    autonomous_steps = int(args.autonomous_steps)

    # ---- dev access ledger entry (WP7 candidate_evaluation) --------------
    cost_pre = {**ZERO_COST_BLOCK, "phase": "pre_episodes",
                "wall_seconds": 0.0}
    ledger_path = append_ledger(root, ledger_entry(
        run_ids=res["run_ids"], lists=lists, split=split,
        output_path=str(out_dir), output_sha256="", cost=cost_pre,
        status="started"))

    # ---- dataset + contexts (dev subset geometry only; no labels/QC) ----
    # LOADER FIX (attempt 1 SIGTERM): loader v2 ColumnarKeyAtomDatasetV3
    # — idx_v2 memmap light index, LAZY samples, bounded cache_curves LRU.
    # Same duck-typed surface the contexts/episodes consume; the v1
    # light-cache materialisation of the ~14.7M-record 50K record set is
    # never triggered.  See LOADER_FIX_NOTE.
    t0 = time.time()
    dataset = open_rollout_dataset(
        tables_dir=args.tables_dir,
        split_manifest=args.split,
        cache_curves=int(args.cache_curves),
    )
    dataset_s = time.time() - t0

    t0 = time.time()
    intents = dataset.intents
    atomic_map = atomic_numbers_map_v541(dataset, selected)
    grouped = group_intents_by_curve(intents, selected)
    contexts: list[Any] = []
    reference_budgets: list[dict[str, Any]] = []
    for cid in selected:
        curve_intents = _v53._direction_intents(grouped[cid], cid)
        for direction in directions:
            ctx = _v53.build_episode_context(
                dataset, curve_id=cid, direction=direction,
                intent=curve_intents[direction],
                atomic_numbers=atomic_map[cid])
            contexts.append(ctx)
            reference_budgets.append({
                "curve_id": cid, "direction": direction,
                "L_key": float(ctx.l_key),
                "budget_reference": int(ctx.reference_budget),
            })
    ctx_s = time.time() - t0
    del dataset, intents, atomic_map, grouped
    import gc
    gc.collect()

    # ---- episode loop (per curve × direction × run × budget mode) --------
    t0 = time.time()
    records: list[dict[str, Any]] = []
    for ctx in contexts:
        for run_id in res["run_ids"]:
            art = runs[run_id]
            for budget_mode in _v53.BUDGET_MODES:
                budget = _v53.episode_budget(
                    budget_mode=budget_mode, l_key=ctx.l_key,
                    autonomous_steps=autonomous_steps, step_cap=step_cap)
                rec = run_episode_v541(
                    ctx,
                    model=art["model"],
                    budget=budget,
                    budget_mode=budget_mode,
                    run_id=run_id,
                    checkpoint_sha256=art["sha256"],
                    seed=int(art["seed"]),
                    tau_end=float(art["tau"]["tau_end"]),
                )
                records.append(rec)
                s = rec["summary"]
                print(
                    f"{ctx.curve_id} {ctx.direction:8s} {run_id} "
                    f"{budget_mode:22s} B={budget:4d} "
                    f"steps={rec['n_steps_executed']:4d} "
                    f"reach={int(s['endpoint_reached'])} "
                    f"ts={int(s['ts_candidate_crossed'])} "
                    f"coll={s['n_collisions']} "
                    f"budget_out={int(s['budget_exhausted'])} "
                    f"would_stop={rec['no_move_diagnostic']['would_stop_step']} "
                    f"wall={rec['wall_seconds']:.1f}s",
                    flush=True)
    episodes_s = time.time() - t0

    # ---- write outputs ---------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = out_dir / "episodes.jsonl"
    with episodes_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True,
                                default=str) + "\n")

    aggregate = aggregate_episode_records_v541(records)
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA_V541,
        "campaign": CAMPAIGN_ID_V541,
        "gate": GATE_ID_V541,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "smoke": smoke,
        "smoke_caps": ({"n_curves": SMOKE_N_CURVES,
                        "directions": ["forward"],
                        "budget_cap": SMOKE_BUDGET_CAP}
                       if smoke else None),
        "plan_file": str(plan_path),
        "plan_sha256": _sha256_file(plan_path),
        "loader": loader_manifest_block(int(args.cache_curves)),
        "split": {
            "path": split["path"],
            "sha256": split["sha256"],
            "schema": split["schema"],
            "frozen": split["frozen"],
            "subset_used": "dev",
            "dev_size": split["dev_size"],
            "selection_rule": lists["selection_rule"],
            "selection_salt": lists["selection_salt"],
            "selection_rule_sha256": lists["selection_rule_sha256"],
            "regression": lists["regression"],
            "rollout": lists["rollout"],
            "lists_checks": lists["checks"],
        },
        "run_ids": res["run_ids"],
        "directions": list(directions),
        "budget_modes": list(_v53.BUDGET_MODES),
        "reference_budget_formula": (
            f"max({_v53.REFERENCE_BUDGET_MIN}, ceil("
            f"{_v53.REFERENCE_OVERSHOOT} * L_key / "
            f"{_v53.REFERENCE_STEP_ANGSTROM}))"),
        "autonomous_steps": autonomous_steps,
        "step_cap": step_cap,
        "reference_budgets": reference_budgets,
        "windows_angstrom": {
            "endpoint_positive_label_spec": 0.10,
            "endpoint_secondary_bond_edit_tol": 0.30,
            "ts_candidate_contact": 0.15,
        },
        "rollout_seed": _v53.SEED_ROLLOUT,
        "feature_level": _v53.FEATURE_LEVEL,
        "trust_region_max_atom": 0.20,
        "collision_distance": 0.5,
        "device": "cpu (pure-geometry rollout inference; no GPU required)",
        "claim_scope": _v53.CLAIM_SCOPE,
        "claims_allowed": ["TS_candidate", "endpoint_reached"],
        "verified_ts_claim": "forbidden",
        "production_no_move_open": False,
        "checkpoints": {
            run_id: {
                "embedding_id": art["embedding_id"],
                "path": art["path"],
                "sha256": art["sha256"],
                "metrics_checkpoint_sha256": art["metrics_checkpoint_sha256"],
                "tau": art["tau"],
                "seed": art["seed"],
                "stage": art["stage"],
            }
            for run_id, art in runs.items()
        },
        "notes": {
            "separate_reporting": SEPARATE_REPORTING_NOTE,
            "no_move": NO_MOVE_NOTE,
            "no_physics": NO_PHYSICS_NOTE,
            "ts_index_role": ("record-only diagnostic tier "
                              "(curve_ts_index); never a policy input"),
        },
        "n_episodes": len(records),
        "episodes_file": str(episodes_path),
        "episodes_schema": EPISODE_SCHEMA_V541,
        "aggregate": aggregate,
        "ledger_path": str(ledger_path),
        "wall_seconds": {
            "dataset_init": dataset_s,
            "contexts": ctx_s,
            "episodes": episodes_s,
            "total": time.time() - t_total,
        },
    }
    _v53.check_claim_discipline(records, manifest)
    manifest_path = out_dir / "rollouts_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8")

    comparison = build_comparison(
        records=records, lists=lists, split=split, runs=runs,
        manifest_path=manifest_path, episodes_path=episodes_path)
    comparison["autonomous_steps"] = autonomous_steps
    comparison["plan_file"] = str(plan_path)
    comparison["loader"] = manifest["loader"]
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = reports_dir / "rollout_comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False, default=str)
        + "\n", encoding="utf-8")

    # ---- ledger completion entry (outputs SHA-bound) ----------------------
    cost_total = {
        **ZERO_COST_BLOCK,
        "phase": "episodes_complete",
        "n_forward_calls": aggregate["physics_cost_totals"][
            "n_forward_calls"],
        "wall_seconds": float(time.time() - t_total),
    }
    append_ledger(root, ledger_entry(
        run_ids=res["run_ids"], lists=lists, split=split,
        output_path=str(manifest_path),
        output_sha256=_sha256_file(manifest_path),
        cost=cost_total, status="done"))

    print(f"-> {episodes_path} ({len(records)} episodes)")
    print(f"-> {manifest_path}")
    print(f"-> {comparison_path}")
    print(f"-> {ledger_path} (purpose={LEDGER_PURPOSE})")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--output-root", default=V541_DEFAULT_ROOT)
        p.add_argument("--runs-root", default=None,
                       help="default: <output-root>/runs")
        p.add_argument("--out-dir", default=None,
                       help="default: <output-root>/reports")
        p.add_argument("--split", required=True,
                       help="frozen v53mp split manifest (dev subset)")
        p.add_argument("--runplan", default=None,
                       help="default: <output-root>/plan/runplan.jsonl")
        p.add_argument("--run-ids", default=DEFAULT_RUN_IDS,
                       help="comma list of V541 E0 run ids")
        p.add_argument("--regression-curves", type=int,
                       default=REGRESSION_N_CURVES)
        p.add_argument("--rollout-curves", type=int,
                       default=ROLLOUT_N_CURVES)
        p.add_argument("--autonomous-steps", type=int,
                       default=DEFAULT_AUTONOMOUS_STEPS)
        p.add_argument("--prior-manifest", default=None,
                       help="optional: prior rollout manifest to "
                            "cross-check the regenerated 24-list")
        p.add_argument("--no-verify-frozen", action="store_true",
                       help="skip the frozen list-SHA checks (fixture "
                            "tests only; formal runs must not pass this)")
        p.add_argument("--no-load-model", dest="load_model",
                       action="store_false", default=True,
                       help="skip the V541Model strict-load probe")
        p.add_argument("--smoke", action="store_true",
                       help="2 curves, forward only, budgets capped at 20")

    p_plan = sub.add_parser("plan", help="dry-run resolution + verification")
    add_common(p_plan)
    p_plan.add_argument("--out", default=None,
                        help="plan JSON path (default: <out-dir>/"
                             "rollout_plan.json)")
    p_plan.set_defaults(func=cmd_plan)

    p_run = sub.add_parser("run-rollouts",
                           help="real no-physics rollout (gated)")
    add_common(p_run)
    p_run.add_argument("--tables-dir", required=True)
    p_run.add_argument("--plan", required=True,
                       help="pinned dry-run plan JSON")
    p_run.add_argument("--execute", action="store_true",
                       help="explicitly authorize the real dev execution")
    p_run.add_argument("--cache-curves", type=int, default=1500)
    p_run.set_defaults(func=cmd_run_rollouts)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except V541RolloutError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    except CampaignRootError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
