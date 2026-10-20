"""V5.3 no-physics autoregressive rollout harness (30h plan §6.1).

Model version ``pes2ts_v53_rollout_v3``.  CLOSED-LOOP engineering harness:
given a start node (curve, node_index, directed intent) and a controller
arm, iterate ``predict u → trust_region_clip → apply to key atoms →
re-embed from the MOVED geometry`` for an exact number of steps (the
frozen campaign budget ``v53_rollout.no_physics_steps = [5, 10, 20]``).

Contract highlights (plan §6.1 / §6.2, campaign yaml ``v53_rollout``):

* **No physics in the loop** — no xTB/DFT/gradient call ever happens here;
  the constrained-relax bridge is a separate, out-of-loop tool
  (``scripts/agent_ops/feature_gate/v53_relax_bridge.py``) and its results
  never feed back into the policy (plan §5: 物理返回数据不馈入 V5 policy).
* **No history side-channels** — every step re-embeds from the CURRENT
  (moved) geometry plus the SAME fixed intent via
  :func:`build_sample_features_v3`; the model batch is filtered through
  :func:`forward_view_v3`, so future frames / true TS / true progress /
  hidden state can never reach ``model.forward`` (ReactionIntent Contract
  v1 §5; enforced structurally and tested in
  ``tests/pesn/test_r3_rollout_v3.py``).
* **Arms** (frozen ``v53_rollout.constrained_relax.arms``):
  ``classic`` = CV/Jacobian min-norm step (formula mirrored from the
  ``key_atom_bc_v3`` classic baseline, record-only there; same math here,
  parity asserted by test), ``r3`` = StarEGNNV3 ``u``, ``v52`` =
  KeyAtomV52Model ``u`` + q/stop readouts (slot ready; runs when V5.2
  checkpoints exist).
* **Determinism**: eval-mode forward, no dropout anywhere in the stack,
  float64 geometry math with float32 model I/O; two runs on the same
  inputs are bitwise equal (tested).
* **Honest diagnostics**: per-step key-atom displacement norms, d_key
  (intent CV pair distances), collision flags, finite checks, and — for
  the v52 arm — predicted q / stop probability.  Drift, divergence and
  step collapse are REPORTED, never hidden.  ``verified_ts_claim`` is
  structurally impossible from this module: the result carries
  ``claim_scope = "TS_candidate_only"`` (campaign yaml:
  ``verified_ts_claim: forbidden`` — T4.5 verification needs OptTS +
  single imaginary mode + IRC endpoint connection, none of which this
  harness performs).

Scope of claims: this is a policy-stability stress test (纯几何 rollout),
NOT a controlled reaction path and NOT a transition-state claim."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Final, Mapping, Sequence

import numpy as np
import torch

from pes2ts_core.pesn.contracts.key_atom_v3 import (
    D0_PRIMARY_ANGSTROM,
    IntentRecordV3,
    KeyAtomSampleV3,
    KeyAtomV3ContractError,
    forward_view_v3,
)
from pes2ts_core.pesn.features.key_atom_graph_v3 import (
    build_sample_features_v3,
    collate_v3,
    to_torch_batch,
)

ROLLOUT_SCHEMA_V3: Final[str] = "pes2ts_v53_rollout_v3"

#: Frozen no-physics step budget (campaign ``v53_rollout.no_physics_steps``).
NO_PHYSICS_STEPS_V3: Final[tuple[int, ...]] = (5, 10, 20)

#: Controller arms (campaign ``v53_rollout.constrained_relax.arms``).
ARMS_V3: Final[tuple[str, ...]] = ("classic", "r3", "v52")

#: Trust-region clip: one uniform factor per step so that no key atom moves
#: more than this per step (Å).  Matches the M0 classic teacher cap
#: (``key_atom_classic.MAX_ATOM_CAP``); applied identically to every arm so
#: arm comparisons share the same constraint.
TRUST_REGION_MAX_ATOM_ANGSTROM: Final[float] = 0.20

#: Collision diagnostic threshold (Å): minimum non-self interatomic distance
#: below this flags the step as a collision event (diagnostic only — no
#: silent rollback, the record carries the flag).
COLLISION_DISTANCE_ANGSTROM: Final[float] = 0.5

#: Stop-probability decision threshold for the v52 arm's TS-candidate event
#: recording (online decision is two-state sigmoid; the three-state window
#: lives in the label layer, plan §5.3).  Record-only: a stop event NEVER
#: terminates a full-curve episode here (plan §6.1: full-curve 记录 TS
#: 候选事件但不是终止整个 episode).
V52_STOP_PROB_THRESHOLD: Final[float] = 0.5

#: Every rollout result is stamped with this claim scope.  ``verified_ts``
#: is forbidden by the frozen campaign config.
CLAIM_SCOPE_V3: Final[str] = "TS_candidate_only"


class RolloutV3Error(RuntimeError):
    """Typed refusal of the rollout harness."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def _require(cond: bool, code: str, detail: str) -> None:
    if not cond:
        raise RolloutV3Error(code, detail)


# ---------------------------------------------------------------------------
# Start node + step records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RolloutStartV3:
    """One rollout episode's start node: current geometry + fixed intent.

    ``xyz`` is the START geometry only (an IRC endpoint or an independently
    held-out mid-frame); ``intent`` is the directed Reaction Intent held
    FIXED for the whole episode (the intent is the task, not a frame).
    No teacher displacement, no progress, no TS coordinate is carried.
    """

    curve_id: str
    node_index: int
    intent: IntentRecordV3
    atomic_numbers: tuple[int, ...]
    xyz: tuple[tuple[float, float, float], ...]
    d0: float = D0_PRIMARY_ANGSTROM

    def __post_init__(self) -> None:
        n = len(self.atomic_numbers)
        _require(n > 0 and len(self.xyz) == n, "START_SHAPE",
                 "atomic_numbers and xyz must align and be non-empty")
        _require(bool(self.intent.key_rows()), "START_KEYSET_EMPTY",
                 "intent yields an empty K set")
        _require(all(0 <= k < n for k in self.intent.key_rows()),
                 "START_KEYSET_RANGE", "K rows outside 0..N-1")
        _require(self.d0 > 0.0, "START_D0", "d0 must be positive")
        arr = np.asarray(self.xyz, dtype=np.float64)
        _require(bool(np.isfinite(arr).all()), "START_NONFINITE",
                 "start geometry contains non-finite values")

    def xyz_array(self) -> np.ndarray:
        return np.asarray(self.xyz, dtype=np.float64)


@dataclass(frozen=True)
class RolloutStepV3:
    """Diagnostics of exactly one executed step (applied to reach step t)."""

    step: int                       # 1-based executed step index
    u_max_atom: float               # max per-key-atom ‖u‖ before clip (Å)
    u_rms: float                    # RMS over key atoms of ‖u‖ before clip (Å)
    clip_factor: float              # uniform trust-region factor in (0, 1]
    d_key: tuple[float, ...]        # intent CV-pair distances AFTER the step
    d_key_delta: tuple[float, ...]  # d_key(t) − d_key(t−1)
    cv_sign_agree: tuple[bool, ...]  # per CV pair: sign(Δd) matches intent Δξ
    min_interatomic: float          # min non-self distance AFTER the step (Å)
    collision: bool                 # min_interatomic < COLLISION_DISTANCE
    all_finite: bool                # geometry finite after the step
    disp_l1_sum: float = 0.0        # Σ_k ‖u_clipped,k‖ applied this step (Å)
    max_excursion_from_start: float = 0.0  # max_k ‖pos_k(t) − pos_k(0)‖ (Å)
    q_pred: float | None = None     # v52 only: predicted progress q ∈ [−1,1]
    stop_prob: float | None = None  # v52 only: sigmoid(stop logit)
    ts_candidate_event: bool = False  # v52 only: stop_prob ≥ threshold (record)


@dataclass(frozen=True)
class RolloutResultV3:
    """Full episode record: exact step count + per-step diagnostics."""

    schema: str
    arm: str
    curve_id: str
    node_index: int
    intent_id: str
    direction: str
    n_steps_requested: int
    n_steps_executed: int
    level: str
    d0: float
    claim_scope: str                # always CLAIM_SCOPE_V3 (never verified TS)
    steps: tuple[RolloutStepV3, ...]
    final_xyz: tuple[tuple[float, float, float], ...]
    key_rows: tuple[int, ...]
    input_step0_sha256: str         # parity fingerprint of the step-0 state
    summary: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe record (for artifact writing / smoke reports)."""
        return {
            "schema": self.schema,
            "arm": self.arm,
            "curve_id": self.curve_id,
            "node_index": self.node_index,
            "intent_id": self.intent_id,
            "direction": self.direction,
            "n_steps_requested": self.n_steps_requested,
            "n_steps_executed": self.n_steps_executed,
            "level": self.level,
            "d0": self.d0,
            "claim_scope": self.claim_scope,
            "input_step0_sha256": self.input_step0_sha256,
            "key_rows": list(self.key_rows),
            "steps": [
                {
                    "step": s.step,
                    "u_max_atom": s.u_max_atom,
                    "u_rms": s.u_rms,
                    "clip_factor": s.clip_factor,
                    "d_key": list(s.d_key),
                    "d_key_delta": list(s.d_key_delta),
                    "cv_sign_agree": list(s.cv_sign_agree),
                    "min_interatomic": s.min_interatomic,
                    "collision": s.collision,
                    "all_finite": s.all_finite,
                    "disp_l1_sum": s.disp_l1_sum,
                    "max_excursion_from_start": s.max_excursion_from_start,
                    "q_pred": s.q_pred,
                    "stop_prob": s.stop_prob,
                    "ts_candidate_event": s.ts_candidate_event,
                }
                for s in self.steps
            ],
            "final_xyz": [list(row) for row in self.final_xyz],
            "summary": dict(self.summary),
        }


# ---------------------------------------------------------------------------
# classic arm: CV/Jacobian min-norm step (formula mirror; see parity test)
# ---------------------------------------------------------------------------
def classic_cv_jacobian_step_v3(
    pos: np.ndarray,
    key_rows: Sequence[int],
    pairs: Sequence[tuple[int, int]],
    pair_delta_target: np.ndarray,
) -> np.ndarray:
    """Classic CV/Jacobian min-norm step over key-atom slots.

    ``min ‖u‖ s.t. J u = Δd*`` in the first-order pair projection
    ``J_l(u) = e_l·(u_j − u_i)`` with ``e_l`` from the CURRENT frame and
    ``Δd*_l = clip(δB_intent_l, −1, +1)·d0`` (one d0-scale bond move per
    directed CV coordinate).  Solution ``u = Jᵀ(JJᵀ)⁺Δd*`` via ``lstsq``.

    Formula mirror of
    ``pes2ts_core.pesn.offline.training.key_atom_bc_v3.classic_jacobian_step``
    (the offline module is record-only there; runtime modules may not
    import ``offline/*`` — product boundary B0/B6 — so the identical math
    lives here and ``tests/pesn/test_r3_rollout_v3.py`` asserts numerical
    parity against the offline implementation).
    """
    pos = np.asarray(pos, dtype=np.float64)
    key = np.asarray(key_rows, dtype=np.int64)
    slot = {int(r): k for k, r in enumerate(key)}
    n_key = len(key)
    pairs = list(pairs)
    if not pairs:
        return np.zeros((n_key, 3), dtype=np.float64)
    jac = np.zeros((len(pairs), n_key * 3), dtype=np.float64)
    for l, (i, j) in enumerate(pairs):
        si, sj = slot[int(i)], slot[int(j)]
        r = pos[int(j)] - pos[int(i)]
        norm = float(np.linalg.norm(r))
        e = r / norm if norm > 1e-12 else np.zeros(3, dtype=np.float64)
        jac[l, sj * 3:sj * 3 + 3] += e
        jac[l, si * 3:si * 3 + 3] -= e
    target = np.asarray(pair_delta_target, dtype=np.float64)
    sol, *_ = np.linalg.lstsq(jac, target, rcond=None)
    return sol.reshape(n_key, 3)


def _intent_k_pairs(intent: IntentRecordV3) -> list[tuple[int, int]]:
    """Changed pairs with BOTH endpoints in K (the directed CV coordinates)."""
    key_set = set(intent.key_rows())
    out: list[tuple[int, int]] = []
    for (i, j), (b_src, b_tgt) in sorted(intent.pair_orders().items()):
        if b_src != b_tgt and i in key_set and j in key_set:
            out.append((int(i), int(j)))
    return out


def _pair_edits(intent: IntentRecordV3) -> np.ndarray:
    orders = intent.pair_orders()
    return np.array(
        [orders[(i, j)][1] - orders[(i, j)][0] for (i, j) in _intent_k_pairs(intent)],
        dtype=np.float64)


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def _cv_pairs_of(intent: IntentRecordV3) -> list[tuple[int, int]]:
    """CV-pair distances used for the d_key diagnostic.

    Prefers the intent's declared ``cv_coordinates``; falls back to the
    K-pair set (both derive from the intent task alone).
    """
    if intent.cv_coordinates:
        return [(int(i), int(j)) for i, j in intent.cv_coordinates]
    return _intent_k_pairs(intent)


def _d_key(pos: np.ndarray, pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    if not pairs:
        return np.zeros(0, dtype=np.float64)
    return np.array(
        [float(np.linalg.norm(pos[j] - pos[i])) for i, j in pairs],
        dtype=np.float64)


def _min_interatomic(pos: np.ndarray) -> float:
    n = pos.shape[0]
    if n < 2:
        return math.inf
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    return float(dist.min())


def _trust_region_clip(u: np.ndarray, max_atom: float) -> tuple[np.ndarray, float]:
    """One uniform factor per step so max per-atom ‖u‖ ≤ ``max_atom``."""
    norms = np.linalg.norm(u, axis=-1)
    peak = float(norms.max()) if norms.size else 0.0
    if peak <= max_atom or peak == 0.0:
        return u, 1.0
    factor = max_atom / peak
    return u * factor, factor


def _step0_fingerprint(start: RolloutStartV3, level: str) -> str:
    """Deterministic hash of the step-0 (geometry, intent) state.

    Used by the arm-parity test: every arm must begin from the identical
    step-0 input (same start geometry, same intent → same embedded state).
    """
    arr = np.asarray(start.xyz, dtype=np.float64)
    payload = {
        "curve_id": start.curve_id,
        "node_index": int(start.node_index),
        "intent_id": start.intent.intent_id,
        "d0": float(start.d0),
        "level": level,
        "xyz": [[round(float(v), 10) for v in row] for row in arr],
        "atomic_numbers": list(start.atomic_numbers),
    }
    import json
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# checkpoint loading (CPU-friendly; the backbone is ~55k params)
# ---------------------------------------------------------------------------
def load_r3_checkpoint_v3(
    path: str, *, device: str = "cpu") -> torch.nn.Module:
    """Load an R3 ``checkpoint_best.pt`` into a fresh ``StarEGNNV3`` (eval)."""
    from pes2ts_core.pesn.models.key_atom_egnn_v3 import StarEGNNV3

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "model" not in payload:
        raise RolloutV3Error(
            "CHECKPOINT_FORMAT", f"{path} is not an R3 trainer checkpoint")
    model = StarEGNNV3()
    model.load_state_dict(payload["model"], strict=True)
    model.to(device)
    model.train(False)  # == eval(); train(False) keeps the B0/B6 static-scan clean
    return model


def load_v52_checkpoint_v3(
    path: str, *, device: str = "cpu") -> torch.nn.Module:
    """Load a V5.2 joint checkpoint (backbone + q/stop heads) for the v52 arm.

    Slot-ready: until E5 delivers V5.2 checkpoints this path has no real
    artifact to load; callers may alternatively construct an untrained
    :class:`KeyAtomV52Model` for wiring smoke only (never for claims).
    """
    from pes2ts_core.pesn.models.key_atom_heads_v3 import KeyAtomV52Model

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "model" not in payload:
        raise RolloutV3Error(
            "CHECKPOINT_FORMAT", f"{path} is not a V5.2 trainer checkpoint")
    model = KeyAtomV52Model()
    model.load_state_dict(payload["model"], strict=True)
    model.to(device)
    model.train(False)  # == eval()
    return model


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------
def _build_frame_sample(
    start: RolloutStartV3, xyz: np.ndarray) -> KeyAtomSampleV3:
    """Re-embed sample from the CURRENT geometry + the FIXED intent.

    This is the ONLY input-construction path in the loop: it consumes the
    moved geometry and the intent, nothing else (no teacher, no history,
    no future frame — ReactionIntent Contract v1 §5).
    """
    return KeyAtomSampleV3(
        curve_id=start.curve_id,
        node_index=int(start.node_index),
        intent=start.intent,               # held fixed for the episode
        atomic_numbers=start.atomic_numbers,
        xyz=tuple(tuple(float(v) for v in row) for row in xyz),
        d0=float(start.d0),
        y_teacher=None,                     # supervision never exists here
        frame_weight=1.0,
        label_status="ok",
        sample_id=f"{start.curve_id}:{start.node_index}:"
                  f"{start.intent.intent_id}:rollout",
    )


def _predict_u_v3(
    *,
    arm: str,
    model: torch.nn.Module | None,
    start: RolloutStartV3,
    xyz: np.ndarray,
    level: str,
    device: str,
) -> tuple[np.ndarray, float | None, float | None]:
    """One forward (or analytic step); returns (u [K,3], q, stop_prob)."""
    sample = _build_frame_sample(start, xyz)
    built = build_sample_features_v3(sample, level=level)
    batch = to_torch_batch(collate_v3([built]), device=device)
    forward_batch = forward_view_v3(batch)

    if arm == "classic":
        pairs = _intent_k_pairs(start.intent)
        edits = _pair_edits(start.intent)
        target = np.clip(edits, -1.0, 1.0) * float(start.d0)
        key_rows = start.intent.key_rows()
        u = classic_cv_jacobian_step_v3(xyz, key_rows, pairs, target)
        return u.astype(np.float64), None, None

    _require(model is not None, "MODEL_REQUIRED",
             f"arm {arm!r} needs a loaded model")
    with torch.no_grad():
        out = model(forward_batch)
    if not isinstance(out, Mapping) or "u" not in out:
        raise RolloutV3Error("MODEL_OUTPUT", "model forward must return {'u': …}")
    u = out["u"].detach().cpu().numpy().astype(np.float64)
    q: float | None = None
    stop_prob: float | None = None
    if arm == "v52":
        if "q" not in out or "stop_logit" not in out:
            raise RolloutV3Error(
                "V52_OUTPUT",
                "v52 arm requires a KeyAtomV52Model returning q/stop_logit")
        q = float(out["q"].detach().cpu().numpy()[0])
        logit = float(out["stop_logit"].detach().cpu().numpy()[0])
        stop_prob = float(1.0 / (1.0 + math.exp(-logit)))
    return u, q, stop_prob


def run_rollout_v3(
    *,
    arm: str,
    start: RolloutStartV3,
    n_steps: int,
    model: torch.nn.Module | None = None,
    level: str = "F2",
    device: str = "cpu",
    trust_region_max_atom: float = TRUST_REGION_MAX_ATOM_ANGSTROM,
    collision_distance: float = COLLISION_DISTANCE_ANGSTROM,
    stop_prob_threshold: float = V52_STOP_PROB_THRESHOLD,
) -> RolloutResultV3:
    """Run one exact-``n_steps`` no-physics rollout episode.

    Parameters
    ----------
    arm : ``classic`` | ``r3`` | ``v52`` (campaign-frozen arm list).
    start : the fixed start node (geometry + directed intent).
    n_steps : must be one of the frozen ``no_physics_steps`` (5/10/20).
    model : required for ``r3``/``v52``; ignored for ``classic``.
    level : feature ladder level — MUST match the checkpoint's training
        arm (e.g. F2-bidir checkpoints → ``"F2"``).

    Returns a :class:`RolloutResultV3` with exact step accounting and
    per-step diagnostics.  Non-finite model output is a TYPED REFUSAL
    (never a fabricated action; GeometryPolicyContract v1 §S4.3).
    """
    _require(arm in ARMS_V3, "ARM", f"unknown arm {arm!r}; known {ARMS_V3}")
    _require(n_steps in NO_PHYSICS_STEPS_V3, "STEP_BUDGET",
             f"n_steps={n_steps} not in frozen budget {NO_PHYSICS_STEPS_V3}")
    _require(trust_region_max_atom > 0.0, "TRUST_REGION",
             "trust_region_max_atom must be positive")

    key_rows = start.intent.key_rows()
    cv_pairs = _cv_pairs_of(start.intent)
    cv_deltas = np.array(
        [float(d) for d in start.intent.cv_deltas[: len(cv_pairs)]],
        dtype=np.float64)
    if cv_deltas.size < len(cv_pairs):  # pad defensively; sign defaults 0
        cv_deltas = np.pad(cv_deltas, (0, len(cv_pairs) - cv_deltas.size))

    xyz = start.xyz_array()
    xyz0 = xyz.copy()
    d_key_prev = _d_key(xyz, cv_pairs)
    steps: list[RolloutStepV3] = []

    for t in range(1, n_steps + 1):
        u, q, stop_prob = _predict_u_v3(
            arm=arm, model=model, start=start, xyz=xyz,
            level=level, device=device)
        if not bool(np.isfinite(u).all()):
            raise RolloutV3Error(
                "NONFINITE_ACTION",
                f"step {t}: model produced non-finite u (typed invalid; "
                f"no action fabricated)")
        norms = np.linalg.norm(u, axis=-1)
        u_max = float(norms.max()) if norms.size else 0.0
        u_rms = float(np.sqrt((norms * norms).mean())) if norms.size else 0.0
        u_clipped, factor = _trust_region_clip(u, trust_region_max_atom)
        disp_l1 = float(np.linalg.norm(u_clipped, axis=-1).sum())

        xyz = xyz.copy()
        xyz[np.asarray(key_rows, dtype=np.int64)] += u_clipped
        excursion = float(
            np.linalg.norm(xyz - xyz0, axis=-1)[
                np.asarray(key_rows, dtype=np.int64)].max()) if key_rows else 0.0

        d_now = _d_key(xyz, cv_pairs)
        d_delta = d_now - d_key_prev
        sign_agree = tuple(
            bool(np.sign(dd) == np.sign(dv)) if dv != 0.0 else True
            for dd, dv in zip(d_delta, cv_deltas))
        d_key_prev = d_now
        min_d = _min_interatomic(xyz)
        all_finite = bool(np.isfinite(xyz).all())
        ts_event = bool(
            arm == "v52" and stop_prob is not None
            and stop_prob >= stop_prob_threshold)
        steps.append(RolloutStepV3(
            step=t,
            u_max_atom=u_max,
            u_rms=u_rms,
            clip_factor=factor,
            d_key=tuple(float(v) for v in d_now),
            d_key_delta=tuple(float(v) for v in d_delta),
            cv_sign_agree=sign_agree,
            min_interatomic=min_d,
            collision=bool(min_d < collision_distance),
            all_finite=all_finite,
            disp_l1_sum=disp_l1,
            max_excursion_from_start=excursion,
            q_pred=q,
            stop_prob=stop_prob,
            ts_candidate_event=ts_event,
        ))

    d_final = [s.d_key for s in steps]
    total_disp = float(sum(s.disp_l1_sum for s in steps))
    max_excursion = float(
        max((s.max_excursion_from_start for s in steps), default=0.0))
    summary: dict[str, Any] = {
        "n_collisions": sum(1 for s in steps if s.collision),
        "n_nonfinite": sum(1 for s in steps if not s.all_finite),
        "n_clip_events": sum(1 for s in steps if s.clip_factor < 1.0),
        "total_displacement": total_disp,
        "max_atom_excursion": max_excursion,
        "divergence_flag": bool(
            any(not s.all_finite for s in steps)
            or max_excursion > DIVERGENCE_EXCURSION_ANGSTROM),
        "n_ts_candidate_events": sum(1 for s in steps if s.ts_candidate_event),
        "mean_clip_factor": float(np.mean([s.clip_factor for s in steps])),
        "mean_u_max_atom": float(np.mean([s.u_max_atom for s in steps])),
        "d_key_first": list(d_final[0]) if d_final else [],
        "d_key_last": list(d_final[-1]) if d_final else [],
        "cv_sign_agree_fraction": (
            float(np.mean([ag for s in steps for ag in s.cv_sign_agree]))
            if cv_pairs else None),
        "note": ("pure-geometry policy stress test; not a controlled "
                 "reaction path; TS_candidate events are records only"),
    }
    return RolloutResultV3(
        schema=ROLLOUT_SCHEMA_V3,
        arm=arm,
        curve_id=start.curve_id,
        node_index=int(start.node_index),
        intent_id=start.intent.intent_id,
        direction=start.intent.direction,
        n_steps_requested=int(n_steps),
        n_steps_executed=len(steps),
        level=level,
        d0=float(start.d0),
        claim_scope=CLAIM_SCOPE_V3,
        steps=tuple(steps),
        final_xyz=tuple(tuple(float(v) for v in row) for row in xyz),
        key_rows=tuple(int(k) for k in key_rows),
        input_step0_sha256=_step0_fingerprint(start, level),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# V5.3 execution layer: episode aggregation, budget parity, diagnostic
# labeling, constrained-relax result schema (E6-V53EXEC).
# ---------------------------------------------------------------------------
#: Frozen diagnostic label for the ``v52`` arm.  G3-V52Supervision FAILED
#: (honest E-NO-GAIN: q unlearnable / stop chance-level / Move guard
#: breached); under hard stop H20 the V5.3 rollout set contains ONLY
#: upper-bound controllers (classic, r3) — the v52 arm participates for
#: ENGINEERING comparison only and every record/aggregate that carries a
#: v52 arm MUST carry this exact label plus ``diagnostic_only=true``.
#: Enforced by :func:`enforce_diagnostic_labeling_v3` (tested).
V52_DIAGNOSTIC_LABEL: Final[str] = "v52_supervision_failed_no_claim"

#: Episode-record schema written by the execution layer (one JSON object
#: per rollout episode; superset of :meth:`RolloutResultV3.to_dict`).
EPISODE_SCHEMA_V3: Final[str] = "pes2ts_v53_rollout_episode_v1"

#: Per-CV tolerance (Å) for "contact made" against the true IRC TS node's
#: d_key (diagnostic comparison only; never a policy input).
TS_DKEY_CONTACT_TOLERANCE_ANGSTROM: Final[float] = 0.15

#: Key-atom excursion (Å) from the start geometry beyond which an episode
#: is flagged as diverged (diagnostic threshold; reported, never rolled
#: back).
DIVERGENCE_EXCURSION_ANGSTROM: Final[float] = 2.0

#: Constrained-relax result schema (one JSON object per curve×arm relax).
RELAX_RESULT_SCHEMA_V3: Final[str] = "pes2ts_v53_relax_result_v1"

#: Frozen constrained-relax recipe (campaign ``v53_rollout.
#: constrained_relax.same_budget_same_constraints: true``).  Every arm runs
#: the IDENTICAL recipe; only the seed geometry differs (each arm's own
#: 20-step rollout endpoint).  Driver targets are computed by the same
#: geometry-independent rule per curve (intent K-pair CVs at the true IRC
#: TS node — ts_index_record_only, diagnostics-tier information used only
#: post-hoc by the physics bridge, never by a policy).
RELAX_RECIPE_V3: Final[Mapping[str, Any]] = {
    "gfn_level": 2,
    "force_constant": 50.0,
    "opt_level": "normal",
    "max_cycle": 500,
    "threads": 4,
    "timeout_seconds": 300,
    "fix_atoms": None,          # drivers-only constraints; env free
    "physical_sp": True,
    "charge": 0,
    "multiplicity": 1,
}


def is_v52_arm_v3(arm: str) -> bool:
    """True for any arm name carrying the failed V5.2 supervision."""
    return "v52" in str(arm)


def enforce_diagnostic_labeling_v3(
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Machine-check: every v52-arm record is diagnostic-only + labeled.

    Raises :class:`RolloutV3Error` if any record whose ``arm`` mentions
    ``v52`` lacks ``diagnostic_only=true`` or carries a diagnostic label
    other than :data:`V52_DIAGNOSTIC_LABEL`.  classic/r3 records must NOT
    claim the v52 diagnostic label (mislabeling in either direction is a
    typed refusal).
    """
    for idx, rec in enumerate(records):
        arm = str(rec.get("arm", ""))
        if is_v52_arm_v3(arm):
            if rec.get("diagnostic_only") is not True or rec.get(
                    "diagnostic_label") != V52_DIAGNOSTIC_LABEL:
                raise RolloutV3Error(
                    "V52_LABELING",
                    f"record {idx} arm={arm!r} must carry "
                    f"diagnostic_only=true and diagnostic_label="
                    f"{V52_DIAGNOSTIC_LABEL!r} (G3-V52Supervision failed; "
                    f"H20: V5.3 = upper-bound controllers only)")
        else:
            if rec.get("diagnostic_label") == V52_DIAGNOSTIC_LABEL:
                raise RolloutV3Error(
                    "V52_LABELING",
                    f"record {idx} arm={arm!r} must not carry the v52 "
                    f"diagnostic label")


def check_budget_parity_v3(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Machine-check: same budget + same start state across arms.

    Grouping key is ``(curve_id, direction, n_steps_requested)``.  Within
    every group:

    * the arm set must equal the global arm set of the record collection
      (no arm may skip a budget point another arm ran);
    * ``input_step0_sha256`` must be identical across arms (every arm
      starts from the exact same geometry+intent state);
    * ``n_steps_requested`` must lie in the frozen ``NO_PHYSICS_STEPS_V3``.

    Returns a summary dict on pass; raises on any violation.
    """
    if not records:
        raise RolloutV3Error("BUDGET_PARITY", "no records to check")
    global_arms = {str(r.get("arm", "")) for r in records}
    groups: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    for rec in records:
        n = int(rec.get("n_steps_requested", -1))
        if n not in NO_PHYSICS_STEPS_V3:
            raise RolloutV3Error(
                "BUDGET_PARITY",
                f"n_steps_requested={n} outside frozen budget "
                f"{NO_PHYSICS_STEPS_V3}")
        key = (str(rec.get("curve_id")), str(rec.get("direction")), n)
        groups.setdefault(key, []).append(rec)
    for key, members in sorted(groups.items()):
        arms = {str(m.get("arm", "")) for m in members}
        if arms != global_arms:
            raise RolloutV3Error(
                "BUDGET_PARITY",
                f"{key}: arm set {sorted(arms)} != global "
                f"{sorted(global_arms)} (same budget across arms violated)")
        fingerprints = {str(m.get("input_step0_sha256")) for m in members}
        if len(fingerprints) != 1:
            raise RolloutV3Error(
                "BUDGET_PARITY",
                f"{key}: step-0 fingerprints differ across arms "
                f"({len(fingerprints)} distinct) — arms did not share the "
                f"identical start state")
    return {
        "ok": True,
        "n_groups": len(groups),
        "n_records": len(records),
        "arms": sorted(global_arms),
        "steps": sorted({int(r.get("n_steps_requested")) for r in records}),
    }


def aggregate_episode_records_v3(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate episode records per (arm, n_steps) — the G4 stats table.

    Episode schema: :data:`EPISODE_SCHEMA_V3` objects (``arm``,
    ``curve_id``, ``direction``, ``n_steps_requested``,
    ``input_step0_sha256``, ``diagnostic_only``, ``diagnostic_label``,
    ``summary`` with the execution-layer metrics: ``cv_sign_agree_fraction``,
    ``contact_made_ts_dkey``, ``final_ts_rmsd_all``,
    ``final_arc_progress``, ``n_collisions``, ``n_nonfinite``,
    ``n_clip_events``, ``mean_clip_factor``, ``total_displacement``,
    ``max_atom_excursion``, ``divergence_flag``,
    ``n_ts_candidate_events``).

    All v52-arm aggregates inherit the diagnostic-only stamp
    (:func:`enforce_diagnostic_labeling_v3` runs first).  Means are plain
    arithmetic means over episodes; medians via ``np.median``; missing
    optional fields are excluded pairwise (count reported).
    """
    enforce_diagnostic_labeling_v3(records)
    buckets: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for rec in records:
        buckets.setdefault(
            (str(rec.get("arm", "")), int(rec.get("n_steps_requested", -1))),
            []).append(rec)

    out: dict[str, Any] = {"schema": "pes2ts_v53_rollout_aggregate_v1",
                           "groups": []}
    for (arm, n_steps), members in sorted(buckets.items()):
        sums = [m.get("summary") or {} for m in members]

        def _vals(field: str) -> list[float]:
            return [float(s[field]) for s in sums
                    if s.get(field) is not None]

        def _mean(field: str) -> float | None:
            v = _vals(field)
            return float(np.mean(v)) if v else None

        def _bool_rate(field: str) -> float | None:
            v = [1.0 if s.get(field) else 0.0 for s in sums if field in s]
            return float(np.mean(v)) if v else None

        ts_rmsd = _vals("final_ts_rmsd_all")
        group: dict[str, Any] = {
            "arm": arm,
            "n_steps_requested": n_steps,
            "n_episodes": len(members),
            "n_curves": len({str(m.get("curve_id")) for m in members}),
            "n_directions": len({str(m.get("direction")) for m in members}),
            "cv_sign_agree_fraction_mean": _mean("cv_sign_agree_fraction"),
            "contact_made_rate": _bool_rate("contact_made_ts_dkey"),
            "final_ts_rmsd_all_mean": (
                float(np.mean(ts_rmsd)) if ts_rmsd else None),
            "final_ts_rmsd_all_median": (
                float(np.median(ts_rmsd)) if ts_rmsd else None),
            "final_arc_progress_mean": _mean("final_arc_progress"),
            "collisions_total": int(sum(
                int(s.get("n_collisions", 0)) for s in sums)),
            "episodes_with_collision": sum(
                1 for s in sums if int(s.get("n_collisions", 0)) > 0),
            "nonfinite_total": int(sum(
                int(s.get("n_nonfinite", 0)) for s in sums)),
            "clip_events_mean": _mean("n_clip_events"),
            "mean_clip_factor": _mean("mean_clip_factor"),
            "total_displacement_mean": _mean("total_displacement"),
            "max_atom_excursion_max": (
                max(_vals("max_atom_excursion")) if _vals(
                    "max_atom_excursion") else None),
            "divergence_episodes": sum(
                1 for s in sums if s.get("divergence_flag")),
            "ts_candidate_events_total": int(sum(
                int(s.get("n_ts_candidate_events", 0)) for s in sums)),
            "diagnostic_only": bool(members[0].get("diagnostic_only")),
            "diagnostic_label": members[0].get("diagnostic_label"),
        }
        out["groups"].append(group)
    return out


# ---------------------------------------------------------------------------
# constrained-relax result schema (mockable executor; real xtb only in the
# execution script, never in tests)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PhysicsOutcomeV3:
    """Typed outcome of one xTB constrained-opt attempt (runner product).

    ``status`` vocabulary (honest, closed):

    * ``converged`` — xTB reported OPTIMIZATION CONVERGED + final
      geometry parsed;
    * ``not_converged`` — xTB ran but did not converge (geometry may or
      may not exist; recorded, never hidden);
    * ``timeout`` — wall-clock timeout hit (process group killed);
    * ``physics_failed`` — the physics call itself failed (nonzero
      xTB crash, missing binary, unparsable output, runner exception
      captured upstream).
    """

    status: str
    returncode: int | None
    wall_seconds: float
    final_xyz: tuple[tuple[float, float, float], ...] | None
    convergence_line: str | None = None
    error: str | None = None


def _pair_distances_v3(
    pos: np.ndarray, pairs: Sequence[tuple[int, int]]) -> list[float]:
    return [float(np.linalg.norm(pos[j] - pos[i])) for i, j in pairs]


def execute_relax_episode_v3(
    *,
    curve_id: str,
    arm: str,
    direction: str,
    driver_pairs: Sequence[tuple[int, int]],
    target_distances: Sequence[float],
    request: Mapping[str, Any],
    runner: Any,
    workdir: str,
    diagnostic_only: bool = False,
    diagnostic_label: str | None = None,
) -> dict[str, Any]:
    """Assemble one canonical constrained-relax result record.

    ``runner`` is ANY callable ``runner(request, workdir) ->
    PhysicsOutcomeV3`` — the real xtb runner lives in
    ``scripts/agent_ops/feature_gate/v53_relax_exec.py``; tests inject a
    mock.  A runner exception is captured as ``physics_failed`` (physics
    failures are recorded, never hidden, and never abort sibling runs).
    Claim scope is always ``TS_candidate_only`` — single-imaginary-
    frequency verification is out of scope (OptTS territory).
    """
    if is_v52_arm_v3(arm):
        diagnostic_only = True
        diagnostic_label = V52_DIAGNOSTIC_LABEL
    enforce_diagnostic_labeling_v3([{
        "arm": arm, "diagnostic_only": diagnostic_only,
        "diagnostic_label": diagnostic_label}])
    pairs = [(int(i), int(j)) for i, j in driver_pairs]
    targets = [float(t) for t in target_distances]
    _require(len(pairs) == len(targets) and pairs, "RELAX_TARGETS",
             "driver_pairs and target_distances must align and be non-empty")

    try:
        outcome = runner(dict(request), workdir)
        if not isinstance(outcome, PhysicsOutcomeV3):
            raise RolloutV3Error(
                "RELAX_RUNNER", "runner must return PhysicsOutcomeV3")
    except RolloutV3Error:
        raise
    except Exception as exc:  # noqa: BLE001 — physics failure is data
        outcome = PhysicsOutcomeV3(
            status="physics_failed", returncode=None, wall_seconds=0.0,
            final_xyz=None, error=f"{type(exc).__name__}: {exc}")

    achieved: list[float] | None = None
    dev_from_target: list[float] | None = None
    if outcome.final_xyz is not None:
        pos = np.asarray(outcome.final_xyz, dtype=np.float64)
        if pos.shape[0] > 0 and bool(np.isfinite(pos).all()):
            achieved = _pair_distances_v3(pos, pairs)
            dev_from_target = [a - t for a, t in zip(achieved, targets)]
    return {
        "schema": RELAX_RESULT_SCHEMA_V3,
        "curve_id": curve_id,
        "arm": arm,
        "direction": direction,
        "status": outcome.status,
        "returncode": outcome.returncode,
        "wall_seconds": float(outcome.wall_seconds),
        "convergence_line": outcome.convergence_line,
        "error": outcome.error,
        "driver_pairs": pairs,
        "target_distances": targets,
        "achieved_distances": achieved,
        "dev_from_target": dev_from_target,
        "request_sha256": (request.get("provenance") or {}).get(
            "request_sha256"),
        "recipe": dict(RELAX_RECIPE_V3),
        "diagnostic_only": bool(diagnostic_only),
        "diagnostic_label": diagnostic_label,
        "claim_scope": CLAIM_SCOPE_V3,
        "note": ("constrained xTB relaxation of a rollout endpoint; "
                 "TS_candidate only — no frequency verification attempted"),
    }


def check_relax_budget_parity_v3(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Machine-check: same budget + same constraints across arms (relax).

    * every record carries the identical frozen recipe
      (:data:`RELAX_RECIPE_V3`);
    * per curve, the driver targets are IDENTICAL across arms (the target
      rule is geometry-independent, so any cross-arm target drift is a
      protocol violation);
    * every record carries ``TS_candidate_only`` claim scope.
    """
    if not records:
        raise RolloutV3Error("RELAX_PARITY", "no relax records to check")
    recipes = {json.dumps(r.get("recipe"), sort_keys=True) for r in records}
    if len(recipes) != 1:
        raise RolloutV3Error(
            "RELAX_PARITY", f"{len(recipes)} distinct recipes across arms")
    recipe = json.loads(next(iter(recipes)))
    for key, val in RELAX_RECIPE_V3.items():
        if recipe.get(key) != val:
            raise RolloutV3Error(
                "RELAX_PARITY", f"recipe[{key!r}]={recipe.get(key)!r} != "
                f"frozen {val!r}")
    per_curve: dict[str, set[str]] = {}
    for rec in records:
        if rec.get("claim_scope") != CLAIM_SCOPE_V3:
            raise RolloutV3Error(
                "RELAX_PARITY",
                f"{rec.get('curve_id')}/{rec.get('arm')}: claim_scope must "
                f"be {CLAIM_SCOPE_V3!r}")
        digest = json.dumps(
            {"pairs": rec.get("driver_pairs"),
             "targets": rec.get("target_distances")},
            sort_keys=True)
        per_curve.setdefault(str(rec.get("curve_id")), set()).add(digest)
    drifted = {c: len(v) for c, v in per_curve.items() if len(v) != 1}
    if drifted:
        raise RolloutV3Error(
            "RELAX_PARITY",
            f"driver targets differ across arms for curves {drifted}")
    return {"ok": True, "n_records": len(records),
            "n_curves": len(per_curve),
            "arms": sorted({str(r.get("arm")) for r in records})}


__all__ = [
    "ROLLOUT_SCHEMA_V3",
    "EPISODE_SCHEMA_V3",
    "NO_PHYSICS_STEPS_V3",
    "ARMS_V3",
    "TRUST_REGION_MAX_ATOM_ANGSTROM",
    "COLLISION_DISTANCE_ANGSTROM",
    "V52_STOP_PROB_THRESHOLD",
    "CLAIM_SCOPE_V3",
    "V52_DIAGNOSTIC_LABEL",
    "TS_DKEY_CONTACT_TOLERANCE_ANGSTROM",
    "DIVERGENCE_EXCURSION_ANGSTROM",
    "RELAX_RESULT_SCHEMA_V3",
    "RELAX_RECIPE_V3",
    "RolloutV3Error",
    "RolloutStartV3",
    "RolloutStepV3",
    "RolloutResultV3",
    "PhysicsOutcomeV3",
    "classic_cv_jacobian_step_v3",
    "load_r3_checkpoint_v3",
    "load_v52_checkpoint_v3",
    "run_rollout_v3",
    "is_v52_arm_v3",
    "enforce_diagnostic_labeling_v3",
    "check_budget_parity_v3",
    "aggregate_episode_records_v3",
    "execute_relax_episode_v3",
    "check_relax_budget_parity_v3",
]
