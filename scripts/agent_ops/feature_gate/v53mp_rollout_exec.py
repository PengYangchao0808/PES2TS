#!/usr/bin/env python3
"""V5.3-MP S3 closed-loop (no-physics) rollout runner — gate V53MP-S3-closed-loop.

NEW runner for the V5.3-MP campaign (``config/campaigns/
pes2ts_v53mp_20261021.yaml``, plan §7.2).  The old runner
``v53_rollout_exec.py`` is bound to the frozen R3/V5.2 campaign matrix
(arms classic/r3/v52, budgets {5,10,20} enforced inside
``run_rollout_v3``); V5.3-MP needs arms A/B/C
(``KeyAtomV53MPModel`` checkpoints) and per-plan budgets up to hundreds
of steps, so this module REUSES the engine's pieces by import (start
state, frame-sample/feature/forward flow, trust-region clip, d_key /
min-interatomic helpers) but owns its own episode loop.

Budgets (plan §7.2; ALWAYS separate episodes/records, never merged):

* ``reference_normalized``: ``B = max(40, ceil(1.5 * L_key / 0.08))``
  where ``L_key`` is the reference-arc key-atom path length (Å) from the
  start endpoint node to the target endpoint node — offline teacher
  diagnostic, comparable with prior campaigns;
* ``autonomous``: fixed ``--autonomous-steps`` (default 200) for every
  episode — the formal full-path evaluation.

Arm C additionally applies its frozen online No-Move stop
(``sigmoid(endpoint_logit) >= tau_end`` halts the episode, stop step
recorded).  Arms A/B have NO legal stop signal (``q`` is
never_stop_criterion per the campaign config) and always run to budget.
The No-Move stop is evaluated ONLY as a diagnostic — training
qualification != deployment qualification.

Curve selection: candidates are the frozen split manifest's ``dev``
subset ONLY (pool = training data, confirm = sealed — neither is ever
rolled out); ordered by ``sha256("v53mp-rollout|17|<curve_id>")`` hex,
first ``--n-curves`` taken; rule + selected list + list sha256 recorded
in the manifest.

Claim discipline (campaign yaml ``evaluation_contract.rollout.claims``):
every record carries ``claim_scope = "TS_candidate_only"``; endpoint /
TS-candidate claims only.  ``verified_ts``-style claims are FORBIDDEN —
they can appear in this codebase only inside such negation notes (OptTS +
single imaginary mode + bidirectional IRC verification is out of scope).
Reference-normalized and autonomous results are aggregated separately —
no merged success rate ever.

CPU-only by design: rollout inference is small (~55k params); the GPU
stays free for the concurrent training campaign.

Usage::

    python3 scripts/agent_ops/feature_gate/v53mp_rollout_exec.py run-rollouts \
        --tables-dir outputs/pesn/v53mp/data \
        --split outputs/pesn/v53mp/data/split_manifest_v53mp.json \
        --runs-root outputs/pesn/v53mp \
        --arms A,B,C --seeds 17,29,43 \
        --n-curves 24 --autonomous-steps 200 \
        --out-dir outputs/pesn/v53mp/rollouts_s3 [--smoke]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# schemas / frozen constants (V5.3-MP lane)
# ---------------------------------------------------------------------------
EPISODE_SCHEMA_V53MP: str = "pes2ts_v53mp_rollout_episode_v1"
MANIFEST_SCHEMA_V53MP: str = "pes2ts_v53mp_rollout_manifest_v1"
AGGREGATE_SCHEMA_V53MP: str = "pes2ts_v53mp_rollout_aggregate_v1"
SPLIT_SCHEMA_V53MP: str = "v53mp_split_v1"
GATE_ID: str = "V53MP-S3-closed-loop"
CAMPAIGN_ID: str = "pes2ts_v53mp_20261021"

#: Frozen claim scope (campaign yaml: claims = [TS_candidate, endpoint_reached]).
CLAIM_SCOPE: str = "TS_candidate_only"

ARMS_V53MP: tuple[str, ...] = ("A", "B", "C")
SMOKE_ARMS: tuple[str, ...] = ("A", "B")   # C checkpoints may not exist yet
DIRECTIONS_V53MP: tuple[str, ...] = ("forward", "reverse")
BUDGET_MODES: tuple[str, ...] = ("reference_normalized", "autonomous")

#: Curve-selection salt + rule (frozen here; recorded verbatim in the manifest).
SELECTION_SALT: str = "17"
SELECTION_RULE: str = (
    "order dev candidates by sha256('v53mp-rollout|17|<curve_id>') hex "
    "ascending; take the first --n-curves")

#: Reference-normalized budget: B = max(40, ceil(1.5 * L_key / 0.08)).
REFERENCE_BUDGET_MIN: int = 40
REFERENCE_OVERSHOOT: float = 1.5
REFERENCE_STEP_ANGSTROM: float = 0.08

#: Per-episode torch/np seed (mirrors the frozen V5.3 rollout seed 17).
SEED_ROLLOUT: int = 17

#: Feature level (V5.3-MP first round keeps the F2-bidir inputs —
#: trainer constant ``FEATURE_LEVEL_V53MP``).
FEATURE_LEVEL: str = "F2"

#: Default geometry-LRU size for rollout curves (dev-only subset; keeps
#: host RAM low while training runs concurrently).
DEFAULT_CACHE_CURVES: int = 700

SMOKE_N_CURVES: int = 2
SMOKE_BUDGET_CAP: int = 20
DEFAULT_AUTONOMOUS_STEPS: int = 200
DEFAULT_N_CURVES: int = 24
DEFAULT_SEEDS: str = "17,29,43"
DEFAULT_ARMS: str = "A,B,C"

ONLINE_STOP_NOTE: str = (
    "arm C online No-Move stop (sigmoid(endpoint_logit) >= tau_end) is "
    "evaluated ONLY as a diagnostic — training qualification != "
    "deployment qualification; arms A/B have no legal stop signal "
    "(q is never_stop_criterion)")
SEPARATE_REPORTING_NOTE: str = (
    "reference_normalized and autonomous results are aggregated "
    "separately by construction — no merged success rate exists in this "
    "manifest")


class RolloutExecError(RuntimeError):
    """Typed refusal of the V5.3-MP rollout runner."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def _require(cond: bool, code: str, detail: str) -> None:
    if not cond:
        raise RolloutExecError(code, detail)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# pure helpers (unit-tested)
# ---------------------------------------------------------------------------
def select_rollout_curves(
    candidate_ids: Sequence[str], n_curves: int
) -> list[str]:
    """Deterministic dev-only curve selection (plan §7.2 S3 rollout set).

    ``candidate_ids`` MUST be the frozen split manifest's ``dev`` subset
    (the caller passes nothing else — pool is training data, confirm is
    sealed).  Candidates are ordered by ``sha256("v53mp-rollout|17|"
    "<curve_id>")`` hex ascending; the first ``n_curves`` win.  Input
    order never matters (set-deterministic); ties are broken by the curve
    id so the result is a total-order function of the candidate SET.
    """
    _require(int(n_curves) >= 1, "N_CURVES", "n_curves must be >= 1")
    unique = sorted({str(c) for c in candidate_ids})
    _require(unique, "NO_CANDIDATES", "candidate set is empty")
    _require(int(n_curves) <= len(unique), "N_CURVES",
             f"n_curves={n_curves} exceeds {len(unique)} candidates")
    ranked = sorted(
        unique,
        key=lambda cid: _sha256_text(f"v53mp-rollout|{SELECTION_SALT}|{cid}"))
    return ranked[: int(n_curves)]


def curve_list_sha256(curve_ids: Sequence[str]) -> str:
    """List identity: sha256 over the newline-joined selected ids."""
    return _sha256_text("\n".join(str(c) for c in curve_ids))


def reference_budget(l_key: float) -> int:
    """Frozen reference-normalized budget ``max(40, ceil(1.5·L/0.08))``."""
    _require(math.isfinite(float(l_key)) and float(l_key) >= 0.0,
             "L_KEY", f"L_key must be finite and >= 0, got {l_key!r}")
    raw = math.ceil(
        REFERENCE_OVERSHOOT * float(l_key) / REFERENCE_STEP_ANGSTROM)
    return max(REFERENCE_BUDGET_MIN, raw)


def l_key_from_arc(key_frames: Sequence[np.ndarray]) -> float:
    """Reference-arc key-atom path length ``L_key`` (Å, Frobenius).

    ``key_frames[f]`` is the ``[n_key, 3]`` key-atom sub-matrix of arc
    frame ``f`` (frames are consecutive true-IRC nodes, start endpoint →
    target endpoint).  ``L_key = Σ_f ||xyz_key[f+1] − xyz_key[f]||_F``.
    Non-key atoms never contribute (they are not in the sub-matrices).
    """
    _require(len(key_frames) >= 2, "ARC_FRAMES",
             "an arc needs at least 2 frames")
    total = 0.0
    for f in range(1, len(key_frames)):
        prev = np.asarray(key_frames[f - 1], dtype=np.float64)
        cur = np.asarray(key_frames[f], dtype=np.float64)
        _require(prev.shape == cur.shape and prev.ndim == 2
                 and prev.shape[1] == 3, "ARC_SHAPE",
                 f"arc frame {f} shape {cur.shape} != {prev.shape}")
        total += float(np.linalg.norm(cur - prev))
    return float(total)


def reference_arc_nodes(start_node: int, target_node: int) -> list[int]:
    """Contiguous inclusive node path start → target (either direction)."""
    if int(start_node) <= int(target_node):
        return list(range(int(start_node), int(target_node) + 1))
    return list(range(int(start_node), int(target_node) - 1, -1))


def episode_budget(
    *,
    budget_mode: str,
    l_key: float,
    autonomous_steps: int,
    step_cap: int | None = None,
) -> int:
    """Resolve one episode's step budget (``step_cap`` = smoke cap)."""
    _require(budget_mode in BUDGET_MODES, "BUDGET_MODE",
             f"unknown budget mode {budget_mode!r}; known {BUDGET_MODES}")
    if budget_mode == "reference_normalized":
        budget = reference_budget(l_key)
    else:
        _require(int(autonomous_steps) >= 1, "AUTONOMOUS_STEPS",
                 f"autonomous_steps must be >= 1, got {autonomous_steps}")
        budget = int(autonomous_steps)
    if step_cap is not None:
        budget = min(budget, int(step_cap))
    _require(budget >= 1, "BUDGET", f"resolved budget {budget} < 1")
    return budget


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return float(1.0 / (1.0 + math.exp(-x)))
    e = math.exp(x)
    return float(e / (1.0 + e))


def _dist_summary(values: Sequence[float]) -> dict[str, Any]:
    """Distribution summary for the first-No-Move distance diagnostics."""
    vals = [float(v) for v in values]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "min": None,
                "max": None}
    return {
        "n": len(vals),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


# ---------------------------------------------------------------------------
# claim discipline (machine-checked; mirrors enforce_diagnostic_labeling_v3)
# ---------------------------------------------------------------------------
def check_claim_discipline(
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any] | None = None,
) -> None:
    """Machine-check the frozen V5.3-MP claim discipline.

    * every episode record carries ``claim_scope == "TS_candidate_only"``;
    * episode records never contain the forbidden claim strings
      ``verified_ts`` / ``verified path``;
    * the manifest carries ``verified_ts_claim == "forbidden"`` (the only
      sanctioned negation note) and never contains ``verified path``.
    """
    _require(bool(records), "NO_RECORDS", "no episode records to check")
    for idx, rec in enumerate(records):
        if rec.get("claim_scope") != CLAIM_SCOPE:
            raise RolloutExecError(
                "CLAIM_SCOPE",
                f"record {idx} claim_scope={rec.get('claim_scope')!r} != "
                f"{CLAIM_SCOPE!r}")
        text = json.dumps(dict(rec), sort_keys=True).lower()
        for banned in ("verified_ts", "verified path"):
            if banned in text:
                raise RolloutExecError(
                    "CLAIM_FORBIDDEN",
                    f"record {idx} contains forbidden claim string "
                    f"{banned!r}")
    if manifest is not None:
        if manifest.get("verified_ts_claim") != "forbidden":
            raise RolloutExecError(
                "CLAIM_FORBIDDEN",
                "manifest must carry verified_ts_claim='forbidden'")
        stripped = {k: v for k, v in dict(manifest).items()
                    if k != "verified_ts_claim"}
        text = json.dumps(stripped, sort_keys=True).lower()
        for banned in ("verified_ts", "verified path"):
            if banned in text:
                raise RolloutExecError(
                    "CLAIM_FORBIDDEN",
                    f"manifest contains forbidden claim string {banned!r} "
                    f"outside the negation note")


# ---------------------------------------------------------------------------
# episode context + loop
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EpisodeContextV53MP:
    """Everything one (curve, direction) episode needs — no live dataset.

    Built ONCE from the frozen tables before the episode loop; contains
    only record-side geometry diagnostics (arc d_key series, endpoint/TS
    frames).  Nothing here is ever fed to the model: the loop re-embeds
    from the CURRENT geometry + the FIXED intent only.
    """

    curve_id: str
    direction: str
    intent: Any                      # IntentRecordV3
    start_node: int
    target_node: int
    ts_index: int                    # record-only (curve_ts_index)
    atomic_numbers: tuple[int, ...]
    start_xyz: tuple[tuple[float, float, float], ...]
    arc_nodes: tuple[int, ...]
    arc_dkey: tuple[np.ndarray, ...]     # d_key at every reference-arc node
    target_xyz: tuple[tuple[float, float, float], ...]
    ts_xyz: tuple[tuple[float, float, float], ...]
    l_key: float
    reference_budget: int            # max(40, ceil(1.5 L_key / 0.08))
    extra: Mapping[str, Any] = field(default_factory=dict)


def _predict_forward_v53mp(
    *,
    model: Any,
    start: Any,                      # RolloutStartV3
    xyz: np.ndarray,
    level: str,
    device: str,
) -> tuple[np.ndarray, float | None, float | None]:
    """One forward on the CURRENT geometry + FIXED intent.

    Identical input-construction discipline to the engine's
    ``_build_frame_sample`` + ``_predict_u_v3`` flow (imports, not
    copies): features → collate → forward-view → ``model.forward``.
    Returns ``(u [K,3], q, endpoint_logit)``; any non-finite model
    output is a TYPED REFUSAL (never a fabricated action).
    """
    import torch

    from pes2ts_core.pesn.contracts.key_atom_v3 import forward_view_v3
    from pes2ts_core.pesn.features.key_atom_graph_v3 import (
        build_sample_features_v3,
        collate_v3,
        to_torch_batch,
    )
    from pes2ts_core.pesn.models.rollout_v3 import (
        RolloutV3Error,
        _build_frame_sample,
    )

    sample = _build_frame_sample(start, xyz)
    built = build_sample_features_v3(sample, level=level)
    batch = to_torch_batch(collate_v3([built]), device=device)
    forward_batch = forward_view_v3(batch)
    with torch.no_grad():
        out = model(forward_batch)
    if not isinstance(out, Mapping) or "u" not in out:
        raise RolloutV3Error(
            "MODEL_OUTPUT",
            "model forward must return a Mapping with 'u'")
    u = np.asarray(out["u"].detach().cpu().numpy(), dtype=np.float64)
    q: float | None = None
    endpoint_logit: float | None = None
    if "q" in out:
        q = float(out["q"].detach().cpu().numpy().reshape(-1)[0])
    if "endpoint_logit" in out:
        endpoint_logit = float(
            out["endpoint_logit"].detach().cpu().numpy().reshape(-1)[0])
    if not bool(np.isfinite(u).all()):
        raise RolloutV3Error(
            "NONFINITE_MODEL_OUTPUT",
            "model produced non-finite u (typed refusal; no action "
            "fabricated)")
    if q is not None and not math.isfinite(q):
        raise RolloutV3Error(
            "NONFINITE_MODEL_OUTPUT",
            "model produced a non-finite q readout (typed refusal)")
    if endpoint_logit is not None and not math.isfinite(endpoint_logit):
        raise RolloutV3Error(
            "NONFINITE_MODEL_OUTPUT",
            "model produced a non-finite endpoint_logit (typed refusal)")
    return u, q, endpoint_logit


def run_episode(
    ctx: EpisodeContextV53MP,
    *,
    arm: str,
    model: Any,
    budget: int,
    budget_mode: str,
    run_id: str,
    checkpoint_sha256: str,
    seed: int,
    tau_end: float | None = None,
    rollout_seed: int = SEED_ROLLOUT,
    level: str = FEATURE_LEVEL,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run ONE no-physics episode and return its record.

    Own loop (the engine's ``run_rollout_v3`` freezes arms/budgets for
    the old campaign), but every geometry/engine primitive is imported
    from ``rollout_v3`` — no copied bodies:

    * seed torch+np with ``rollout_seed`` before the episode (CPU only);
    * each step: forward on the CURRENT geometry with the FIXED intent,
      trust-region clip at 0.20 Å/atom, apply to key rows only;
    * record d_key, per-CV sign agreement vs intent cv_deltas, min
      interatomic distance (collision < 0.5 Å), non-finite geometry
      flag, clip factor, displacement L1, q / endpoint_prob when the
      model emits them;
    * arm C: halt when ``sigmoid(endpoint_logit) >= tau_end`` (record
      stop step; diagnostic only) — A/B always run to budget;
    * non-finite MODEL output → typed refusal error captured in the
      record; no action is ever fabricated.
    """
    import torch

    from pes2ts_core.pesn.models.rollout_v3 import (
        CLAIM_SCOPE_V3,
        COLLISION_DISTANCE_ANGSTROM,
        RolloutStartV3,
        RolloutV3Error,
        TRUST_REGION_MAX_ATOM_ANGSTROM,
        TS_DKEY_CONTACT_TOLERANCE_ANGSTROM,
        _cv_pairs_of,
        _d_key,
        _min_interatomic,
        _step0_fingerprint,
        _trust_region_clip,
    )
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (
        kabsch_rmsd_v52,
    )
    from pes2ts_core.pesn.offline.data.key_atom_endpoint_labels_v1 import (
        BOND_EDIT_TOLERANCE_ANGSTROM_V1,
        ENDPOINT_STOP_POSITIVE_WINDOW_ANGSTROM,
    )

    _require(arm in ARMS_V53MP, "ARM", f"unknown arm {arm!r}")
    _require(budget_mode in BUDGET_MODES, "BUDGET_MODE",
             f"unknown budget mode {budget_mode!r}")
    _require(int(budget) >= 1, "BUDGET", f"budget must be >= 1, got {budget}")
    if arm == "C":
        _require(tau_end is not None and math.isfinite(float(tau_end)),
                 "TAU_END",
                 "arm C requires a frozen tau_end from metrics.json")
    _require(CLAIM_SCOPE_V3 == CLAIM_SCOPE, "CLAIM_SCOPE_DRIFT",
             f"engine claim scope {CLAIM_SCOPE_V3!r} != lane {CLAIM_SCOPE!r}")

    t0 = time.time()
    intent = ctx.intent
    key_rows = intent.key_rows()
    cv_pairs = _cv_pairs_of(intent)
    cv_deltas = np.array(
        [float(d) for d in intent.cv_deltas[: len(cv_pairs)]],
        dtype=np.float64)
    if cv_deltas.size < len(cv_pairs):
        cv_deltas = np.pad(cv_deltas, (0, len(cv_pairs) - cv_deltas.size))

    d_target = np.asarray(ctx.arc_dkey[-1], dtype=np.float64)
    ts_pos = list(ctx.arc_nodes).index(int(ctx.ts_index))
    d_ts = np.asarray(ctx.arc_dkey[ts_pos], dtype=np.float64)
    key_index = np.asarray(key_rows, dtype=np.int64)
    target_xyz = np.asarray(ctx.target_xyz, dtype=np.float64)
    ts_xyz = np.asarray(ctx.ts_xyz, dtype=np.float64)

    start = RolloutStartV3(
        curve_id=ctx.curve_id, node_index=int(ctx.start_node),
        intent=intent, atomic_numbers=ctx.atomic_numbers,
        xyz=ctx.start_xyz, d0=0.08)

    torch.manual_seed(int(rollout_seed))
    np.random.seed(int(rollout_seed))

    xyz = start.xyz_array()
    xyz0 = xyz.copy()
    d_prev = _d_key(xyz, cv_pairs)
    arc_n = len(ctx.arc_nodes)
    steps: list[dict[str, Any]] = []
    refusal: dict[str, Any] | None = None
    first_no_move_step: int | None = None
    no_move_distance: float | None = None
    no_move_arc_remaining: float | None = None
    online_halted = False
    max_excursion = 0.0

    t_exec = 1
    while t_exec <= int(budget):
        try:
            u, q, endpoint_logit = _predict_forward_v53mp(
                model=model, start=start, xyz=xyz,
                level=level, device=device)
        except RolloutV3Error as exc:
            refusal = {"code": exc.code, "detail": exc.detail,
                       "step": int(t_exec),
                       "action": "no_action_fabricated"}
            break

        norms = np.linalg.norm(u, axis=-1)
        u_max = float(norms.max()) if norms.size else 0.0
        u_rms = float(np.sqrt((norms * norms).mean())) if norms.size else 0.0
        u_clipped, factor = _trust_region_clip(
            u, TRUST_REGION_MAX_ATOM_ANGSTROM)
        disp_l1 = float(np.linalg.norm(u_clipped, axis=-1).sum())

        xyz = xyz.copy()
        xyz[key_index] += u_clipped
        geometry_finite = bool(np.isfinite(xyz).all())
        if key_index.size:
            max_excursion = max(
                max_excursion,
                float(np.linalg.norm(
                    xyz[key_index] - xyz0[key_index], axis=-1).max()))

        d_now = _d_key(xyz, cv_pairs)
        d_delta = d_now - d_prev
        sign_agree = tuple(
            bool(np.sign(dd) == np.sign(dv)) if dv != 0.0 else True
            for dd, dv in zip(d_delta, cv_deltas))
        d_prev = d_now
        min_d = _min_interatomic(xyz)

        dev_target = float(np.abs(d_now - d_target).max()) if d_now.size \
            else math.inf
        dev_ts = float(np.abs(d_now - d_ts).max()) if d_now.size \
            else math.inf
        dists = [float(np.linalg.norm(d_now - a)) for a in ctx.arc_dkey]
        nearest = int(np.argmin(dists)) if dists else 0
        arc_frac = nearest / max(1, arc_n - 1)

        endpoint_prob = (_sigmoid(endpoint_logit)
                         if endpoint_logit is not None else None)
        steps.append({
            "step": int(t_exec),
            "u_max_atom": u_max,
            "u_rms": u_rms,
            "clip_factor": float(factor),
            "disp_l1_sum": disp_l1,
            "d_key": [float(v) for v in d_now],
            "d_key_delta": [float(v) for v in d_delta],
            "cv_sign_agree": [bool(v) for v in sign_agree],
            "min_interatomic": float(min_d),
            "collision": bool(min_d < COLLISION_DISTANCE_ANGSTROM),
            "geometry_finite": geometry_finite,
            "endpoint_dev_max": dev_target,
            "ts_dev_max": dev_ts,
            "arc_progress": float(arc_frac),
            "q_pred": q,
            "endpoint_prob": endpoint_prob,
        })

        if not geometry_finite:
            refusal = {
                "code": "NONFINITE_GEOMETRY",
                "detail": f"step {t_exec}: geometry non-finite after "
                          f"applying a finite action (typed refusal; no "
                          f"action fabricated)",
                "step": int(t_exec),
                "action": "no_action_fabricated"}
            break

        if (arm == "C" and endpoint_prob is not None
                and endpoint_prob >= float(tau_end)):
            online_halted = True
            first_no_move_step = int(t_exec)
            no_move_distance = dev_target
            no_move_arc_remaining = float(1.0 - arc_frac)
            break
        t_exec += 1

    n_steps = len(steps)

    # ---- summary metrics (all record-only diagnostics) --------------------
    ep_step = next((s["step"] for s in steps
                    if s["endpoint_dev_max"] <=
                    ENDPOINT_STOP_POSITIVE_WINDOW_ANGSTROM), None)
    ep30_step = next((s["step"] for s in steps
                      if s["endpoint_dev_max"] <=
                      BOND_EDIT_TOLERANCE_ANGSTROM_V1), None)
    ts_step = next((s["step"] for s in steps
                    if s["ts_dev_max"] <=
                    TS_DKEY_CONTACT_TOLERANCE_ANGSTROM), None)
    progress = [s["arc_progress"] for s in steps]
    final_xyz = xyz

    if cv_pairs:
        d_final = _d_key(final_xyz, cv_pairs)
        dev_final: float | None = float(np.abs(d_final - d_target).max())
        ts_final: float | None = float(np.abs(d_final - d_ts).max())
    else:  # pragma: no cover — contexts are built with declared CVs
        dev_final = ts_final = None

    endpoint_reached = ep_step is not None
    summary: dict[str, Any] = {
        "n_steps_executed": n_steps,
        "endpoint_reached": bool(endpoint_reached),
        "endpoint_reached_step": ep_step,
        "endpoint_reached_tol30": ep30_step is not None,
        "endpoint_reached_tol30_step": ep30_step,
        "ts_candidate_crossed": ts_step is not None,
        "ts_candidate_crossed_step": ts_step,
        "arc_progress_series": progress,
        "final_arc_progress": (progress[-1] if progress else None),
        "first_no_move_step": first_no_move_step,
        "no_move_distance_max_dkey": no_move_distance,
        "no_move_arc_fraction_remaining": no_move_arc_remaining,
        "stopped_online": bool(online_halted),
        "n_collisions": sum(1 for s in steps if s["collision"]),
        "n_nonfinite": sum(1 for s in steps if not s["geometry_finite"]),
        "n_clip_events": sum(1 for s in steps if s["clip_factor"] < 1.0),
        "cv_sign_agree_fraction": (
            float(np.mean([a for s in steps for a in s["cv_sign_agree"]]))
            if steps and cv_pairs else None),
        "total_displacement": float(
            sum(s["disp_l1_sum"] for s in steps)),
        "mean_clip_factor": (
            float(np.mean([s["clip_factor"] for s in steps]))
            if steps else None),
        "max_atom_excursion": float(max_excursion),
        "endpoint_not_reached": not endpoint_reached,
        "budget_exhausted": bool(
            refusal is None and not online_halted and n_steps == int(budget)
            and not endpoint_reached),
        "refusal_code": refusal["code"] if refusal else None,
        "final_dev_target_max": dev_final,
        "final_dev_ts_max": ts_final,
        "final_target_rmsd_all": float(
            kabsch_rmsd_v52(final_xyz, target_xyz)),
        "final_target_rmsd_key": float(
            kabsch_rmsd_v52(final_xyz[key_index], target_xyz[key_index])),
        "final_ts_rmsd_all": float(kabsch_rmsd_v52(final_xyz, ts_xyz)),
        "final_ts_rmsd_key": float(
            kabsch_rmsd_v52(final_xyz[key_index], ts_xyz[key_index])),
        "note": ("record-only diagnostics; TS_candidate/endpoint_reached "
                 "claims only; not a controlled reaction path"),
    }

    record: dict[str, Any] = {
        "schema": EPISODE_SCHEMA_V53MP,
        "campaign": CAMPAIGN_ID,
        "gate": GATE_ID,
        "curve_id": ctx.curve_id,
        "direction": ctx.direction,
        "intent_id": str(intent.intent_id),
        "start_node_index": int(ctx.start_node),
        "target_node_index": int(ctx.target_node),
        "ts_index_record_only": int(ctx.ts_index),
        "arm": arm,
        "seed": int(seed),
        "run_id": run_id,
        "checkpoint_sha256": checkpoint_sha256,
        "tau_end": (float(tau_end) if tau_end is not None else None),
        "budget_mode": budget_mode,
        "budget": int(budget),
        "L_key": float(ctx.l_key),
        "rollout_seed": int(rollout_seed),
        "level": level,
        "device": device,
        "trust_region_max_atom": TRUST_REGION_MAX_ATOM_ANGSTROM,
        "collision_distance": COLLISION_DISTANCE_ANGSTROM,
        "n_steps_requested": int(budget),
        "n_steps_executed": n_steps,
        "input_step0_sha256": _step0_fingerprint(start, level),
        "halted_online_stop": bool(online_halted),
        "online_stop_step": first_no_move_step,
        "online_stop_role": ("diagnostic_only" if arm == "C" else None),
        "claim_scope": CLAIM_SCOPE,
        "refusal": refusal,
        "steps": steps,
        "final_xyz": [[float(v) for v in row] for row in final_xyz],
        "summary": summary,
        "wall_seconds": float(time.time() - t0),
    }
    return record


# ---------------------------------------------------------------------------
# aggregation (per arm × budget_mode — never merged)
# ---------------------------------------------------------------------------
def aggregate_episode_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate per ``(arm, budget_mode)`` — data only, no prose verdicts.

    Groups are constructed by the ``(arm, budget_mode)`` key, so
    reference-normalized and autonomous results structurally CANNOT be
    merged into one success rate.
    """
    check_claim_discipline(records)
    buckets: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for rec in records:
        buckets.setdefault(
            (str(rec.get("arm")), str(rec.get("budget_mode"))),
            []).append(rec)

    groups: list[dict[str, Any]] = []
    for (arm, budget_mode), members in sorted(buckets.items()):
        sums = [dict(m.get("summary") or {}) for m in members]
        budgets = [int(m.get("budget", 0)) for m in members]
        refusals = [m.get("refusal") for m in members]

        def _rate(field: str) -> float:
            return float(np.mean([1.0 if s.get(field) else 0.0
                                  for s in sums])) if sums else 0.0

        stops = [s for s in sums if s.get("first_no_move_step") is not None]
        groups.append({
            "arm": arm,
            "budget_mode": budget_mode,
            "n_episodes": len(members),
            "n_curves": len({str(m.get("curve_id")) for m in members}),
            "n_directions": len({str(m.get("direction")) for m in members}),
            "n_seeds": len({int(m.get("seed", -1)) for m in members}),
            "budget_min": min(budgets) if budgets else None,
            "budget_max": max(budgets) if budgets else None,
            "endpoint_arrival_rate": _rate("endpoint_reached"),
            "endpoint_arrival_rate_tol30": _rate("endpoint_reached_tol30"),
            "ts_candidate_crossing_rate": _rate("ts_candidate_crossed"),
            "first_no_move_distance_summary": _dist_summary(
                [s["no_move_distance_max_dkey"] for s in stops]),
            "first_no_move_arc_fraction_remaining_summary": _dist_summary(
                [s["no_move_arc_fraction_remaining"] for s in stops]),
            "collisions_total": int(sum(
                int(s.get("n_collisions", 0)) for s in sums)),
            "episodes_with_collision": sum(
                1 for s in sums if int(s.get("n_collisions", 0)) > 0),
            "nonfinite_total": int(sum(
                int(s.get("n_nonfinite", 0)) for s in sums)),
            "refusal_episodes": sum(1 for r in refusals if r),
            "illegal_model_output_episodes": sum(
                1 for r in refusals
                if r and r.get("code") == "NONFINITE_MODEL_OUTPUT"),
            "budget_exhaustion_fraction": _rate("budget_exhausted"),
            "online_stop_halted_episodes": sum(
                1 for s in sums if s.get("stopped_online")),
        })
    return {
        "schema": AGGREGATE_SCHEMA_V53MP,
        "grouping": "arm x budget_mode (separate reporting; never merged)",
        "groups": groups,
    }


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------
def _parse_csv_str(value: str) -> list[str]:
    return [p.strip() for p in str(value).split(",") if p.strip()]


def _parse_csv_int(value: str) -> list[int]:
    try:
        return [int(p) for p in _parse_csv_str(value)]
    except ValueError as exc:
        raise RolloutExecError("SEEDS", f"bad integer list {value!r}") from exc


def _direction_intents(
    intents: Mapping[tuple[str, str], Any], curve_id: str
) -> dict[str, Any]:
    """Exactly one forward + one reverse intent per curve (frozen data)."""
    out: dict[str, Any] = {}
    for (cid, _iid), intent in sorted(intents.items()):
        if cid != curve_id:
            continue
        direction = str(intent.direction)
        if direction in out:
            raise RolloutExecError(
                "INTENT_MULTIPLE",
                f"{curve_id}: multiple {direction!r} intents in the "
                f"frozen tables")
        out[direction] = intent
    if set(out) != set(DIRECTIONS_V53MP):
        raise RolloutExecError(
            "INTENT_MISSING",
            f"{curve_id}: expected forward+reverse intents, got "
            f"{sorted(out)}")
    return out


def atomic_numbers_map(
    dataset: Any, curve_ids: Sequence[str]
) -> dict[str, tuple[int, ...]]:
    """One pass over ``dataset.samples`` → curve_id → atomic numbers."""
    wanted = {str(c) for c in curve_ids}
    out: dict[str, tuple[int, ...]] = {}
    for rec in dataset.samples:
        cid = rec.curve_id
        if cid in wanted and cid not in out:
            out[cid] = tuple(int(z) for z in rec.atomic_numbers)
    missing = sorted(wanted - set(out))
    if missing:
        raise RolloutExecError(
            "ATOMIC_NUMBERS",
            f"curves without sample rows (not built?): {missing[:5]}")
    return out


def build_episode_context(
    dataset: Any,
    *,
    curve_id: str,
    direction: str,
    intent: Any,
    atomic_numbers: Sequence[int],
) -> EpisodeContextV53MP:
    """Assemble one (curve, direction) context from the frozen tables."""
    from pes2ts_core.pesn.models.rollout_v3 import _d_key

    endpoints = dataset.curve_endpoints(curve_id)
    a, b = int(endpoints[0]), int(endpoints[1])
    ts_index = int(dataset.curve_ts_index(curve_id))
    start_node, target_node = ((a, b) if direction == "forward"
                               else (b, a))
    if str(intent.direction) != direction:
        raise RolloutExecError(
            "INTENT_DIRECTION",
            f"{curve_id}: intent {intent.intent_id} direction "
            f"{intent.direction!r} != {direction!r}")
    key_rows = tuple(int(k) for k in intent.key_rows())
    if not key_rows:
        raise RolloutExecError(
            "KEYSET_EMPTY", f"{curve_id}/{direction}: empty K set")
    arc_nodes = reference_arc_nodes(start_node, target_node)
    if ts_index not in arc_nodes:
        raise RolloutExecError(
            "TS_NOT_ON_ARC",
            f"{curve_id}: ts_index {ts_index} not on arc "
            f"{arc_nodes[0]}..{arc_nodes[-1]}")

    frames = [np.asarray(dataset.frame_xyz(curve_id, n), dtype=np.float64)
              for n in arc_nodes]
    cv_pairs = [(int(i), int(j)) for i, j in (intent.cv_coordinates or ())]
    if not cv_pairs:
        raise RolloutExecError(
            "NO_CV_PAIRS",
            f"{curve_id}/{direction}: intent declares no cv_coordinates")
    arc_dkey = tuple(_d_key(f, cv_pairs) for f in frames)
    key_index = np.asarray(key_rows, dtype=np.int64)
    l_key = l_key_from_arc([f[key_index] for f in frames])
    ts_pos = arc_nodes.index(ts_index)
    return EpisodeContextV53MP(
        curve_id=curve_id,
        direction=direction,
        intent=intent,
        start_node=start_node,
        target_node=target_node,
        ts_index=ts_index,
        atomic_numbers=tuple(int(z) for z in atomic_numbers),
        start_xyz=tuple(tuple(float(v) for v in row)
                        for row in frames[0]),
        arc_nodes=tuple(arc_nodes),
        arc_dkey=arc_dkey,
        target_xyz=tuple(tuple(float(v) for v in row)
                         for row in frames[-1]),
        ts_xyz=tuple(tuple(float(v) for v in row)
                     for row in frames[ts_pos]),
        l_key=float(l_key),
        reference_budget=reference_budget(l_key),
    )


def load_v53mp_checkpoint(
    path: Path, *, device: str = "cpu"
) -> Any:
    """Load a V5.3-MP arm checkpoint (``KeyAtomV53MPModel``, strict, eval)."""
    import torch

    from pes2ts_core.pesn.models.key_atom_heads_v3 import KeyAtomV53MPModel
    from pes2ts_core.pesn.models.rollout_v3 import RolloutV3Error

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "model" not in payload:
        raise RolloutExecError(
            "CHECKPOINT_FORMAT",
            f"{path} is not a V5.3-MP trainer checkpoint (missing "
            f"'model' state_dict)")
    model = KeyAtomV53MPModel()
    model.load_state_dict(payload["model"], strict=True)
    model.to(device)
    model.train(False)  # == eval(); train(False) keeps static scans clean
    return model


def _load_run_artifact(
    runs_root: Path, tier: str, arm: str, seed: int
) -> dict[str, Any]:
    """checkpoint_best.pt + metrics.json for one run (fail-closed)."""
    run_id = f"v53mp_{tier}_{arm}_s{seed}"
    run_dir = runs_root / run_id
    ckpt = run_dir / "checkpoint_best.pt"
    metrics_path = run_dir / "metrics.json"
    missing = [str(p) for p in (ckpt, metrics_path) if not p.is_file()]
    if missing:
        raise RolloutExecError(
            "RUN_ARTIFACT_MISSING",
            f"{run_id}: missing {missing} (arm/seed not trained yet?)")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    ckpt_sha = _sha256_file(ckpt)
    metrics_sha = metrics.get("checkpoint_best_sha256")
    if metrics_sha and str(metrics_sha) != ckpt_sha:
        raise RolloutExecError(
            "CHECKPOINT_SHA_DRIFT",
            f"{run_id}: metrics checkpoint_best_sha256 {metrics_sha} != "
            f"actual file sha256 {ckpt_sha}")
    tau_end = (metrics.get("endpoint_head") or {}).get("tau_end")
    if arm == "C":
        _require(tau_end is not None and math.isfinite(float(tau_end)),
                 "TAU_END_MISSING",
                 f"{run_id}: metrics.json lacks endpoint_head.tau_end "
                 f"(required for the arm C online stop)")
    return {
        "run_id": run_id,
        "path": str(ckpt),
        "sha256": ckpt_sha,
        "metrics_checkpoint_sha256": str(metrics_sha) if metrics_sha else None,
        "tau_end": float(tau_end) if tau_end is not None else None,
        "arm": arm,
        "seed": int(seed),
        "model": load_v53mp_checkpoint(ckpt),
    }


def run_rollouts(args: argparse.Namespace) -> int:
    from scripts.campaigns.r3_v53mp_run import _assert_v53mp_root

    # ---- fail-closed campaign root BEFORE any output is opened -----------
    out_dir = Path(args.out_dir)
    runs_root = Path(args.runs_root)
    tables_dir = Path(args.tables_dir)
    split_path = Path(args.split)
    _assert_v53mp_root(str(out_dir))
    _assert_v53mp_root(str(runs_root))
    _assert_v53mp_root(str(tables_dir))

    t_total = time.time()
    tier = str(getattr(args, "tier", "20k"))
    smoke = bool(args.smoke)

    arms = _parse_csv_str(args.arms)
    for arm in arms:
        if arm not in ARMS_V53MP:
            raise RolloutExecError(
                "ARM", f"unknown arm {arm!r}; known {ARMS_V53MP}")
    if len(set(arms)) != len(arms):
        raise RolloutExecError("ARM", f"duplicate arms in {arms!r}")
    seeds = _parse_csv_int(args.seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise RolloutExecError("SEEDS", f"bad seeds {seeds!r}")

    if smoke:
        dropped = [a for a in arms if a not in SMOKE_ARMS]
        arms = [a for a in arms if a in SMOKE_ARMS]
        if dropped:
            print(f"smoke: excluding arm(s) {dropped} (smoke runs A,B only "
                  f"— C checkpoints may not exist yet)", file=sys.stderr)
        if not arms:
            raise RolloutExecError(
                "SMOKE_ARMS", "smoke mode supports arms A,B only")
        n_curves = SMOKE_N_CURVES
        directions = ("forward",)
        step_cap: int | None = SMOKE_BUDGET_CAP
    else:
        n_curves = int(args.n_curves)
        directions = DIRECTIONS_V53MP
        step_cap = None
    autonomous_steps = int(args.autonomous_steps)
    budget_modes = BUDGET_MODES

    # ---- frozen split: dev subset ONLY -----------------------------------
    split_doc = json.loads(split_path.read_text(encoding="utf-8"))
    if split_doc.get("schema") != SPLIT_SCHEMA_V53MP:
        raise RolloutExecError(
            "SPLIT_SCHEMA",
            f"{split_path}: schema {split_doc.get('schema')!r} != "
            f"{SPLIT_SCHEMA_V53MP!r}")
    if not split_doc.get("frozen"):
        raise RolloutExecError(
            "SPLIT_FROZEN", f"{split_path} is not frozen")
    subsets = split_doc.get("subsets") or {}
    dev_ids = [str(c) for c in subsets.get("dev") or []]
    confirm_ids = {str(c) for c in subsets.get("confirm") or []}
    pool_ids = set()
    for tier_ids in (subsets.get("pool") or {}).values():
        pool_ids.update(str(c) for c in tier_ids)
    _require(dev_ids, "SPLIT_DEV", "split manifest has an empty dev subset")

    selected = select_rollout_curves(dev_ids, n_curves)
    sel_set = set(selected)
    if sel_set & confirm_ids or sel_set & pool_ids:
        raise RolloutExecError(
            "SELECTION_NOT_DEV_ONLY",
            "selected curves overlap the sealed confirm / training pool "
            "subsets")
    list_sha = curve_list_sha256(selected)

    # ---- dataset (lazy sharded loader; CPU; bounded geometry LRU) --------
    t0 = time.time()
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (
        LABEL_STATUSES,
        ShardedKeyAtomDatasetV3,
    )

    dataset = ShardedKeyAtomDatasetV3(
        str(tables_dir),
        split_manifest=str(split_path),
        label_statuses=LABEL_STATUSES,   # ok/masked/terminal/conflict
        light_cache=True,                # manifest-bound light-index cache
        cache_curves=int(getattr(args, "cache_curves",
                                 DEFAULT_CACHE_CURVES)),
    )
    dataset_s = time.time() - t0

    # ---- per-curve contexts (start/arc/endpoint/TS diagnostics) ----------
    t0 = time.time()
    intents = dataset.intents  # single Mapping copy
    atomic_map = atomic_numbers_map(dataset, selected)
    contexts: list[EpisodeContextV53MP] = []
    reference_budgets: list[dict[str, Any]] = []
    for cid in selected:
        curve_intents = _direction_intents(intents, cid)
        for direction in directions:
            ctx = build_episode_context(
                dataset, curve_id=cid, direction=direction,
                intent=curve_intents[direction],
                atomic_numbers=atomic_map[cid])
            contexts.append(ctx)
            reference_budgets.append({
                "curve_id": cid,
                "direction": direction,
                "L_key": float(ctx.l_key),
                "budget_reference": int(ctx.reference_budget),
            })
    ctx_s = time.time() - t0

    # ---- checkpoints (A/B/C × seeds) -------------------------------------
    t0 = time.time()
    runs: dict[tuple[str, int], dict[str, Any]] = {}
    for arm in arms:
        for seed in seeds:
            runs[(arm, seed)] = _load_run_artifact(
                runs_root, tier, arm, seed)
    ckpt_s = time.time() - t0

    # Free the light index + intent map before the episode loop (host-RAM
    # discipline: a training campaign runs concurrently).  Contexts carry
    # their own geometry diagnostics; nothing downstream reads the dataset.
    del dataset, intents, atomic_map
    import gc
    gc.collect()

    # ---- episode loop (per curve × direction × arm × seed × budget mode)
    t0 = time.time()
    records: list[dict[str, Any]] = []
    for ctx in contexts:
        for arm in arms:
            for seed in seeds:
                art = runs[(arm, seed)]
                for budget_mode in budget_modes:
                    budget = episode_budget(
                        budget_mode=budget_mode,
                        l_key=ctx.l_key,
                        autonomous_steps=autonomous_steps,
                        step_cap=step_cap)
                    rec = run_episode(
                        ctx,
                        arm=arm,
                        model=art["model"],
                        budget=budget,
                        budget_mode=budget_mode,
                        run_id=art["run_id"],
                        checkpoint_sha256=art["sha256"],
                        seed=seed,
                        tau_end=art["tau_end"],
                    )
                    records.append(rec)
                    s = rec["summary"]
                    print(
                        f"{ctx.curve_id} {ctx.direction:8s} {arm} s{seed} "
                        f"{budget_mode:22s} B={budget:4d} "
                        f"steps={rec['n_steps_executed']:4d} "
                        f"reach={int(s['endpoint_reached'])} "
                        f"ts={int(s['ts_candidate_crossed'])} "
                        f"prog={s['final_arc_progress'] if s['final_arc_progress'] is None else round(s['final_arc_progress'], 3)} "
                        f"coll={s['n_collisions']} "
                        f"stop={s['first_no_move_step']} "
                        f"wall={rec['wall_seconds']:.1f}s",
                        flush=True)
    episodes_s = time.time() - t0

    # ---- discipline + aggregation ----------------------------------------
    aggregate = aggregate_episode_records(records)

    # ---- write outputs (campaign root already asserted) -------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = out_dir / "episodes.jsonl"
    with episodes_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True)
                     + "\n")

    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA_V53MP,
        "campaign": CAMPAIGN_ID,
        "gate": GATE_ID,
        "generated_at": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "smoke": smoke,
        "smoke_caps": ({"n_curves": SMOKE_N_CURVES,
                        "directions": ["forward"],
                        "budget_cap": SMOKE_BUDGET_CAP,
                        "arms": list(SMOKE_ARMS)} if smoke else None),
        "tier": tier,
        "split": {
            "path": str(split_path),
            "sha256": _sha256_file(split_path),
            "schema": split_doc.get("schema"),
            "frozen": bool(split_doc.get("frozen")),
            "subset_used": "dev",
            "dev_subset_size": len(dev_ids),
            "selection_rule": SELECTION_RULE,
            "selection_salt": SELECTION_SALT,
            "n_curves": len(selected),
            "curve_ids": selected,
            "curve_list_sha256": list_sha,
        },
        "arms": arms,
        "seeds": seeds,
        "directions": list(directions),
        "budget_modes": list(budget_modes),
        "reference_budget_formula": (
            f"max({REFERENCE_BUDGET_MIN}, ceil({REFERENCE_OVERSHOOT} * "
            f"L_key / {REFERENCE_STEP_ANGSTROM}))"),
        "autonomous_steps": autonomous_steps,
        "step_cap": step_cap,
        "reference_budgets": reference_budgets,
        "windows_angstrom": {
            "endpoint_positive_label_spec": 0.10,
            "endpoint_secondary_bond_edit_tol": 0.30,
            "ts_candidate_contact": 0.15,
        },
        "rollout_seed": SEED_ROLLOUT,
        "feature_level": FEATURE_LEVEL,
        "trust_region_max_atom": 0.20,
        "collision_distance": 0.5,
        "device": ("cpu (rollout inference is small; GPU left free for "
                   "the concurrent training campaign — documented "
                   "choice)"),
        "claim_scope": CLAIM_SCOPE,
        "claims_allowed": ["TS_candidate", "endpoint_reached"],
        "verified_ts_claim": "forbidden",
        "checkpoints": {
            art["run_id"]: {
                "path": art["path"],
                "sha256": art["sha256"],
                "metrics_checkpoint_sha256": art[
                    "metrics_checkpoint_sha256"],
                "tau_end": art["tau_end"],
                "arm": art["arm"],
                "seed": art["seed"],
            }
            for art in runs.values()
        },
        "notes": {
            "separate_reporting": SEPARATE_REPORTING_NOTE,
            "online_stop": ONLINE_STOP_NOTE,
            "ts_index_role": ("record-only diagnostic tier "
                              "(curve_ts_index); never a policy input"),
            "no_physics": ("no physics call happens inside any episode — "
                           "pure-geometry rollout"),
        },
        "n_episodes": len(records),
        "episodes_file": str(episodes_path),
        "episodes_schema": EPISODE_SCHEMA_V53MP,
        "aggregate": aggregate,
        "wall_seconds": {
            "dataset_init": dataset_s,
            "contexts": ctx_s,
            "checkpoint_load": ckpt_s,
            "episodes": episodes_s,
            "total": time.time() - t_total,
        },
    }
    check_claim_discipline(records, manifest)
    manifest_path = out_dir / "rollouts_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    print(f"-> {episodes_path} ({len(records)} episodes)")
    print(f"-> {manifest_path}")
    print(f"wall: dataset={dataset_s:.1f}s contexts={ctx_s:.1f}s "
          f"ckpt={ckpt_s:.1f}s episodes={episodes_s:.1f}s "
          f"total={time.time() - t_total:.1f}s")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run-rollouts")
    p.add_argument("--tables-dir", default="outputs/pesn/v53mp/data")
    p.add_argument("--split",
                   default="outputs/pesn/v53mp/data/split_manifest_v53mp.json")
    p.add_argument("--runs-root", default="outputs/pesn/v53mp")
    p.add_argument("--arms", default=DEFAULT_ARMS,
                   help="comma list over {A,B,C}")
    p.add_argument("--seeds", default=DEFAULT_SEEDS,
                   help="comma list of trained-run seeds")
    p.add_argument("--n-curves", type=int, default=DEFAULT_N_CURVES)
    p.add_argument("--autonomous-steps", type=int,
                   default=DEFAULT_AUTONOMOUS_STEPS)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tier", default="20k", choices=("2k", "5k", "20k"))
    p.add_argument("--cache-curves", type=int,
                   default=DEFAULT_CACHE_CURVES)
    p.add_argument("--smoke", action="store_true",
                   help="2 curves, forward only, budgets capped at 20, "
                        "arms A,B only")
    p.set_defaults(func=run_rollouts)
    args = parser.parse_args(argv)
    from scripts.agent_ops.supervisor.campaign_root import CampaignRootError

    try:
        return int(args.func(args))
    except RolloutExecError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    except CampaignRootError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
