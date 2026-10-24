#!/usr/bin/env python3
"""V5.3-MP zero-training-cost noise-injection diagnostic (plan §9.5).

DIAGNOSTIC ONLY — NOT part of S4.1; NO training is performed or consumed.
Uses the frozen S3 checkpoints and perturbs the rollout START geometry
with per-atom Gaussian noise ``N(0, σ²)`` (default σ = 0.02 Å) to
quantify exposure bias via the collision-rate change vs an unperturbed
baseline (plan ``outputs/pesn/v53mp/reports/S4_infra_优化方案_20261023.md``
§9.5: "用 S3 冻结模型，从微扰起点（σ=0.02 Å）rollout，量化 exposure bias
与碰撞率；若碰撞率随扰动急剧上升，再立项重训练").

This module REUSES ``v53mp_rollout_exec.py`` by import (episode contexts,
curve selection, the ``run_episode`` loop, checkpoint loading,
aggregation, claim discipline) — nothing in that module is edited or
copied.  The ONLY perturbation hook is a ``dataclasses.replace`` of the
episode context's ``start_xyz``; the noise is applied ONCE at t=0 (the
start frame), never re-drawn per step, and every record-side diagnostic
(arc d_key, target/TS frames, budgets) stays on the pristine reference
arc so baseline/perturbed episodes remain directly comparable.

Protocol per checkpoint (arm × seed):

1. BASELINE REPRODUCTION anchor — rerun the unperturbed episodes with
   the identical curve selection (frozen salt "17") and assert every
   per-episode record matches the stored ``rollouts_s3`` output for the
   same ``(run_id, curve_id, direction, budget_mode)`` key: all
   DETERMINISTIC quantities bitwise-exact (flags, step counts, stop
   steps, arc-progress series, hashes), continuous model outputs within
   a tight tolerance (BLAS thread-order FP noise; larger gap = harness
   bug ⇒ STOP).
2. PERTURBED — same curves / steps / seeds; start xyz += noise draw
   seeded deterministically from ``(noise_seed, curve_id)`` (one draw
   per curve, reused for both directions — same molecule, same N atoms).
3. COMPARISON — per episode: collision flags, final IRC deviation vs the
   reference endpoint, stop-step changes (arm C online No-Move stop),
   and the divergence-vs-step profile (exposure-bias curve).

Summary per checkpoint + overall: collision rate baseline vs perturbed
with exact (Clopper–Pearson) binomial CIs, median/max endpoint
deviation, fraction of episodes whose stop step changes (arm C only).
Decision rule (plan §9.5): collision rate significantly worse
(perturbed CI lower bound > baseline rate) ⇒ recommend a standalone
retraining project; else park the branch.

CPU-only by default (``--device cpu``): rollout inference is tiny and
the GPU stays free for any concurrent campaign.  The dataset is loaded
ONCE for all checkpoints.

Usage::

    python3 scripts/agent_ops/feature_gate/v53mp_noise_rollout_diag.py \
        run-noise-diag \
        --tables-dir outputs/pesn/v53mp/data \
        --split outputs/pesn/v53mp/data/split_manifest_v53mp.json \
        --runs-root outputs/pesn/v53mp \
        --arms C --seeds 17,29,43 \
        --n-curves 24 --autonomous-steps 200 \
        --sigma 0.02 --noise-seed 17 \
        --out-dir outputs/pesn/v53mp/rollouts_noise [--smoke]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# Reuse of the frozen V5.3-MP rollout runner (import, never fork).
# ---------------------------------------------------------------------------
from scripts.agent_ops.feature_gate.v53mp_rollout_exec import (  # noqa: E402
    ARMS_V53MP,
    BUDGET_MODES,
    CLAIM_SCOPE,
    DEFAULT_AUTONOMOUS_STEPS,
    DEFAULT_CACHE_CURVES,
    DEFAULT_N_CURVES,
    DEFAULT_SEEDS,
    DIRECTIONS_V53MP,
    EPISODE_SCHEMA_V53MP,
    FEATURE_LEVEL,
    SEED_ROLLOUT,
    SELECTION_SALT,
    SPLIT_SCHEMA_V53MP,
    EpisodeContextV53MP,
    RolloutExecError,
    _direction_intents,
    _load_run_artifact,
    atomic_numbers_map,
    build_episode_context,
    check_claim_discipline,
    curve_list_sha256,
    episode_budget,
    run_episode,
    select_rollout_curves,
)
from scripts.campaigns.r3_v53mp_run import _assert_v53mp_root  # noqa: E402

# ---------------------------------------------------------------------------
# schemas / constants (noise-diagnostic lane)
# ---------------------------------------------------------------------------
NOISE_EPISODE_SCHEMA: str = "pes2ts_v53mp_noise_episode_v1"
NOISE_MANIFEST_SCHEMA: str = "pes2ts_v53mp_noise_manifest_v1"
NOISE_REPORT_SCHEMA: str = "pes2ts_v53mp_noise_diag_report_v1"
DIAG_ID: str = "V53MP-S3-noise-injection"

DEFAULT_SIGMA: float = 0.02
DEFAULT_NOISE_SEED: int = 17
DEFAULT_OUT_DIR: str = "outputs/pesn/v53mp/rollouts_noise"
DEFAULT_ANCHOR_DIR: str = "outputs/pesn/v53mp/rollouts_s3"
DEFAULT_REPORT_JSON: str = (
    "outputs/pesn/v53mp/reports/noise_diag_20261023.json")
DEFAULT_REPORT_MD: str = (
    "outputs/pesn/v53mp/reports/noise_diag_20261023.md")

SMOKE_N_CURVES: int = 2
SMOKE_BUDGET_CAP: int = 20
SMOKE_DIRECTIONS: tuple[str, ...] = ("forward",)

#: Deterministic noise-seed derivation (documented in the manifest).
NOISE_SEED_FORMULA: str = (
    "rng = np.random.default_rng(int(sha256('<noise_seed>|<curve_id>')"
    ".hexdigest()[:32], 16)); noise = rng.normal(0.0, sigma, (N_atoms, 3))"
    " — one draw per curve, reused for both direction start frames")

PERTURBATION_NOTE: str = (
    "start-frame xyz += N(0, sigma^2) per atom, applied ONCE at t=0 "
    "(never re-drawn per step); record-side reference diagnostics "
    "(arc d_key, target/TS frames, budgets) stay on the pristine arc")

DIAGNOSTIC_ONLY_NOTE: str = (
    "zero-training-cost diagnostic per plan §9.5 — NOT part of S4.1; "
    "no training performed or consumed; frozen S3 checkpoints only")

VERDICT_RETRAIN: str = "recommend_standalone_retraining_project"
VERDICT_PARK: str = "park_branch"


class NoiseDiagError(RuntimeError):
    """Typed refusal of the noise-injection diagnostic."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def _require(cond: bool, code: str, detail: str) -> None:
    if not cond:
        raise NoiseDiagError(code, detail)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# pure helpers (unit-tested)
# ---------------------------------------------------------------------------
def noise_rng(noise_seed: int, curve_id: str) -> np.random.Generator:
    """Deterministic per-(noise_seed, curve_id) RNG.

    ``sha256(f"{noise_seed}|{curve_id}")`` → 128-bit integer seed for
    ``np.random.default_rng``; input order and call sites never matter.
    """
    _require(int(noise_seed) >= 0, "NOISE_SEED",
             f"noise_seed must be >= 0, got {noise_seed!r}")
    digest = hashlib.sha256(
        f"{int(noise_seed)}|{curve_id}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "big"))


def start_noise_draw(
    noise_seed: int, curve_id: str, n_atoms: int, sigma: float
) -> np.ndarray:
    """Per-atom Gaussian start-geometry noise ``[N, 3]`` (Å).

    ``sigma == 0`` short-circuits to an exact-zero array (σ=0 must be
    bitwise-identical to the unperturbed baseline).
    """
    _require(int(n_atoms) >= 1, "N_ATOMS", "n_atoms must be >= 1")
    _require(math.isfinite(float(sigma)) and float(sigma) >= 0.0,
             "SIGMA", f"sigma must be finite and >= 0, got {sigma!r}")
    if float(sigma) == 0.0:
        return np.zeros((int(n_atoms), 3), dtype=np.float64)
    return noise_rng(noise_seed, curve_id).normal(
        0.0, float(sigma), size=(int(n_atoms), 3)).astype(np.float64)


def apply_start_noise(
    ctx: EpisodeContextV53MP, *, sigma: float, noise_seed: int
) -> tuple[EpisodeContextV53MP, np.ndarray]:
    """Return ``(ctx_noisy, noise)`` — start_xyz perturbed ONCE at t=0.

    Every other context field (arc d_key series, target/TS frames,
    budgets, intent) is carried over untouched, so all record-side
    diagnostics of the perturbed episode stay comparable with the
    baseline.  The ORIGINAL ``ctx`` is never mutated.
    """
    start = np.asarray(ctx.start_xyz, dtype=np.float64)
    noise = start_noise_draw(
        noise_seed, str(ctx.curve_id), int(start.shape[0]), sigma)
    noisy = start + noise
    _require(bool(np.isfinite(noisy).all()), "NOISE_NONFINITE",
             f"{ctx.curve_id}: perturbed start geometry is non-finite")
    ctx_noisy = dataclasses.replace(
        ctx,
        start_xyz=tuple(tuple(float(v) for v in row) for row in noisy))
    return ctx_noisy, noise


def episode_key(rec: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """Match key shared with the ``rollouts_s3`` anchor episodes."""
    return (str(rec.get("run_id")), str(rec.get("curve_id")),
            str(rec.get("direction")), str(rec.get("budget_mode")))


#: Default tolerance (Å / dimensionless) for CONTINUOUS record fields in
#: the baseline reproduction anchor.  Continuous model outputs can differ
#: at ~1e-9 when the rerun process uses a different BLAS thread count
#: than the original run (reduction-order noise, amplified over a closed
#: loop); every DETERMINISTIC quantity (flags, step counts, stop steps,
#: arc-progress series, hashes) must still match BITWISE.
DEFAULT_ANCHOR_CONTINUOUS_TOL: float = 1e-6

#: Float fields that must match BITWISE even though they are floats —
#: they are functions of discrete quantities (or frozen inputs), so any
#: difference means real harness/selection drift, not FP noise.
ANCHOR_EXACT_FLOAT_FIELDS: frozenset[str] = frozenset({
    "L_key", "tau_end", "arc_progress_series", "final_arc_progress",
    "no_move_arc_fraction_remaining", "cv_sign_agree_fraction",
})


def _strip_wall(record: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in dict(record).items() if k != "wall_seconds"}


def _compare_values(
    a: Any, b: Any, *, path: str, leaf: str, tol: float,
    stats: dict[str, Any],
) -> str | None:
    """Recursive anchor comparison: exact for deterministic values,
    tolerance for continuous floats (tracked in ``stats``)."""
    if isinstance(a, bool) or isinstance(b, bool):
        if a is not b:
            return f"{path}: anchor={a!r} rerun={b!r}"
        return None
    if isinstance(a, float) or isinstance(b, float):
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            return (f"{path}: type {type(a).__name__} != "
                    f"{type(b).__name__}")
        if leaf in ANCHOR_EXACT_FLOAT_FIELDS:
            if a != b:
                return f"{path}: exact-float mismatch anchor={a!r} " \
                       f"rerun={b!r}"
            return None
        dev = abs(float(a) - float(b))
        stats["n_continuous_compared"] += 1
        stats["max_continuous_abs_dev"] = max(
            stats["max_continuous_abs_dev"], dev)
        if dev > stats["continuous_tol"]:
            stats["n_continuous_over_tol"] += 1
            return (f"{path}: continuous dev {dev:.3e} > tol "
                    f"{stats['continuous_tol']:.1e} "
                    f"(anchor={a!r} rerun={b!r})")
        return None
    if isinstance(a, int) or isinstance(b, int):
        if a != b:
            return f"{path}: anchor={a!r} rerun={b!r}"
        return None
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            return f"{path}: container type mismatch"
        for k in sorted(set(a) | set(b)):
            if k not in a:
                return f"{path}.{k}: missing in anchor"
            if k not in b:
                return f"{path}.{k}: missing in rerun"
            d = _compare_values(a[k], b[k], path=f"{path}.{k}",
                                leaf=str(k), tol=tol, stats=stats)
            if d:
                return d
        return None
    if isinstance(a, list) or isinstance(b, list):
        if not (isinstance(a, list) and isinstance(b, list)):
            return f"{path}: container type mismatch"
        if len(a) != len(b):
            return f"{path}: list len {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = _compare_values(x, y, path=f"{path}[{i}]", leaf=leaf,
                                tol=tol, stats=stats)
            if d:
                return d
        return None
    if type(a) is not type(b) or a != b:
        return f"{path}: anchor={a!r} rerun={b!r}"
    return None


def compare_anchor_records(
    stored: Mapping[str, Any], rerun: Mapping[str, Any], *,
    continuous_tol: float = DEFAULT_ANCHOR_CONTINUOUS_TOL,
) -> tuple[str | None, dict[str, Any]]:
    """One-episode anchor comparison; returns ``(diff_or_None, stats)``.

    Everything except ``wall_seconds`` is compared.  All deterministic
    quantities (bools, ints, strings, exact-float fields) must match
    bitwise; other floats must sit within ``continuous_tol``.
    """
    stats: dict[str, Any] = {
        "continuous_tol": float(continuous_tol),
        "n_continuous_compared": 0,
        "max_continuous_abs_dev": 0.0,
        "n_continuous_over_tol": 0,
    }
    diff = _compare_values(
        _strip_wall(stored), _strip_wall(rerun), path="", leaf="",
        tol=float(continuous_tol), stats=stats)
    return diff, stats


def load_anchor_episodes(
    anchor_dir: Path, run_ids: Sequence[str]
) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    """Index the stored ``rollouts_s3`` episodes for the wanted runs."""
    path = Path(anchor_dir) / "episodes.jsonl"
    _require(path.is_file(), "ANCHOR_MISSING",
             f"{path} not found (baseline reproduction anchor required)")
    wanted = {str(r) for r in run_ids}
    out: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if str(rec.get("run_id")) in wanted:
                out[episode_key(rec)] = rec
    return out


def assert_baseline_anchor(
    rerun_records: Sequence[Mapping[str, Any]],
    anchor: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    *, require_all: bool = True,
    continuous_tol: float = DEFAULT_ANCHOR_CONTINUOUS_TOL,
) -> dict[str, Any]:
    """Baseline reproduction anchor vs the stored ``rollouts_s3`` set.

    All DETERMINISTIC quantities (flags, step counts, arc-progress
    series, stop steps, hashes, refusal codes, …) must match BITWISE;
    continuous model outputs must sit within ``continuous_tol``
    (BLAS thread-order FP noise, amplified by the closed loop — any
    larger gap is a harness bug ⇒ raise and STOP).  ``wall_seconds``
    is excluded.  Returns a verification summary for the manifest.
    """
    checked = 0
    max_dev = 0.0
    n_compared = 0
    matched_keys: list[str] = []
    for rec in rerun_records:
        key = episode_key(rec)
        stored = anchor.get(key)
        if stored is None:
            if require_all:
                raise NoiseDiagError(
                    "BASELINE_ANCHOR_MISSING_EPISODE",
                    f"rerun episode {key} not present in the anchor set "
                    f"(curve selection or harness drift)")
            continue
        diff, stats = compare_anchor_records(
            stored, rec, continuous_tol=continuous_tol)
        if diff is not None:
            raise NoiseDiagError(
                "BASELINE_ANCHOR_MISMATCH",
                f"episode {key}: baseline rerun != stored anchor "
                f"(harness bug — STOP). first diff: {diff}")
        checked += 1
        max_dev = max(max_dev, stats["max_continuous_abs_dev"])
        n_compared += stats["n_continuous_compared"]
        matched_keys.append("|".join(key))
    if require_all:
        _require(checked == len(rerun_records), "BASELINE_ANCHOR_INCOMPLETE",
                 f"only {checked}/{len(rerun_records)} episodes verified")
    return {
        "n_episodes_checked": checked,
        "n_rerun": len(rerun_records),
        "deterministic_fields_bitwise_exact": True,
        "continuous_tolerance": float(continuous_tol),
        "max_continuous_abs_dev": float(max_dev),
        "n_continuous_values_compared": n_compared,
        "excluded_fields": ["wall_seconds"],
        "episode_keys_sha256": hashlib.sha256(
            "\n".join(sorted(matched_keys)).encode("utf-8")).hexdigest(),
    }


# ---------------------------------------------------------------------------
# exact (Clopper–Pearson) binomial CI — pure math, no scipy dependency
# ---------------------------------------------------------------------------
def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p) (n small: direct summation)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    total = 0.0
    for i in range(0, k + 1):
        total += math.comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i))
    return float(min(1.0, max(0.0, total)))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided Clopper–Pearson interval for a binomial rate."""
    _require(int(n) >= 1, "CI_N", f"n must be >= 1, got {n}")
    _require(0 <= int(k) <= int(n), "CI_K", f"k must be in [0, n], got {k}/{n}")
    _require(0.0 < float(alpha) < 1.0, "CI_ALPHA",
             f"alpha must be in (0, 1), got {alpha}")
    k, n = int(k), int(n)
    if k == 0:
        lower = 0.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(100):            # bisect: 1 - CDF(k-1; p) = alpha/2
            mid = 0.5 * (lo + hi)       # (increasing in p)
            if 1.0 - _binom_cdf(k - 1, n, mid) > alpha / 2.0:
                hi = mid
            else:
                lo = mid
        lower = 0.5 * (lo + hi)
    if k == n:
        upper = 1.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(100):            # bisect: CDF(k; p) = alpha/2
            mid = 0.5 * (lo + hi)
            if _binom_cdf(k, n, mid) > alpha / 2.0:
                lo = mid
            else:
                hi = mid
        upper = 0.5 * (lo + hi)
    return float(lower), float(upper)


def collision_rate_summary(flags: Sequence[bool],
                           alpha: float = 0.05) -> dict[str, Any]:
    """``k/n`` collision-episode rate with an exact binomial CI."""
    vals = [1.0 if bool(f) else 0.0 for f in flags]
    n = len(vals)
    if n == 0:
        return {"n": 0, "k": 0, "rate": None, "ci": None,
                "ci_level": 1.0 - alpha}
    k = int(sum(vals))
    lo, hi = clopper_pearson(k, n, alpha=alpha)
    return {"n": n, "k": k, "rate": float(k / n),
            "ci": [lo, hi], "ci_level": 1.0 - alpha}


def _dist(values: Sequence[float]) -> dict[str, Any]:
    vals = [float(v) for v in values]
    if not vals:
        return {"n": 0, "median": None, "mean": None, "min": None,
                "max": None}
    return {"n": len(vals), "median": float(np.median(vals)),
            "mean": float(np.mean(vals)), "min": float(min(vals)),
            "max": float(max(vals))}


def first_collision_step(record: Mapping[str, Any]) -> int | None:
    for step in record.get("steps") or ():
        if step.get("collision"):
            return int(step.get("step"))
    return None


# ---------------------------------------------------------------------------
# per-episode pairing + condition comparison
# ---------------------------------------------------------------------------
def pair_episode_records(
    baseline: Sequence[Mapping[str, Any]],
    perturbed: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Pair baseline/perturbed episodes by the shared match key."""
    base = {episode_key(r): r for r in baseline}
    pert = {episode_key(r): r for r in perturbed}
    _require(set(base) == set(pert), "PAIR_KEYS",
             f"key sets differ: baseline-only "
             f"{sorted(set(base) - set(pert))[:3]} / perturbed-only "
             f"{sorted(set(pert) - set(base))[:3]}")
    return [(base[k], pert[k]) for k in sorted(base)]


def compare_episode_pair(
    base: Mapping[str, Any], pert: Mapping[str, Any]
) -> dict[str, Any]:
    """One (baseline, perturbed) episode comparison record."""
    bs, ps = base["summary"], pert["summary"]
    base_stop = base.get("online_stop_step")
    pert_stop = pert.get("online_stop_step")
    return {
        "run_id": str(base["run_id"]),
        "curve_id": str(base["curve_id"]),
        "direction": str(base["direction"]),
        "budget_mode": str(base["budget_mode"]),
        "arm": str(base["arm"]),
        "seed": int(base["seed"]),
        "budget": int(base["budget"]),
        "baseline": {
            "has_collision": bool(bs["n_collisions"] > 0),
            "n_collisions": int(bs["n_collisions"]),
            "first_collision_step": first_collision_step(base),
            "final_dev_target_max": bs["final_dev_target_max"],
            "final_target_rmsd_all": bs["final_target_rmsd_all"],
            "online_stop_step": base_stop,
            "stopped_online": bool(bs["stopped_online"]),
            "endpoint_reached": bool(bs["endpoint_reached"]),
            "n_steps_executed": int(bs["n_steps_executed"]),
            "input_step0_sha256": str(base["input_step0_sha256"]),
        },
        "perturbed": {
            "has_collision": bool(ps["n_collisions"] > 0),
            "n_collisions": int(ps["n_collisions"]),
            "first_collision_step": first_collision_step(pert),
            "final_dev_target_max": ps["final_dev_target_max"],
            "final_target_rmsd_all": ps["final_target_rmsd_all"],
            "online_stop_step": pert_stop,
            "stopped_online": bool(ps["stopped_online"]),
            "endpoint_reached": bool(ps["endpoint_reached"]),
            "n_steps_executed": int(ps["n_steps_executed"]),
            "input_step0_sha256": str(pert["input_step0_sha256"]),
        },
        "collision_flag_changed": bool(
            (bs["n_collisions"] > 0) != (ps["n_collisions"] > 0)),
        "collision_new": bool(ps["n_collisions"] > 0 and bs["n_collisions"] == 0),
        "collision_cleared": bool(
            ps["n_collisions"] == 0 and bs["n_collisions"] > 0),
        "n_collisions_delta": int(ps["n_collisions"] - bs["n_collisions"]),
        "final_dev_target_delta": (
            float(ps["final_dev_target_max"])
            - float(bs["final_dev_target_max"])
            if bs["final_dev_target_max"] is not None
            and ps["final_dev_target_max"] is not None else None),
        "final_rmsd_delta": (
            float(ps["final_target_rmsd_all"])
            - float(bs["final_target_rmsd_all"])),
        "stop_step_changed": bool(base_stop != pert_stop),
        "start_step0_changed": bool(
            base["input_step0_sha256"] != pert["input_step0_sha256"]),
    }


def divergence_profile(
    baseline: Mapping[str, Any], perturbed: Mapping[str, Any]
) -> list[dict[str, float]]:
    """Per-step |perturbed − baseline| of (arc_progress, endpoint_dev).

    The exposure-bias curve: how far the perturbed rollout drifts from
    the unperturbed one as model exposure grows.  Steps beyond either
    episode's executed length are not comparable and are dropped.
    """
    b_steps = baseline.get("steps") or []
    p_steps = perturbed.get("steps") or []
    out: list[dict[str, float]] = []
    for i in range(min(len(b_steps), len(p_steps))):
        b, p = b_steps[i], p_steps[i]
        out.append({
            "step": int(b["step"]),
            "abs_d_arc_progress": abs(
                float(p["arc_progress"]) - float(b["arc_progress"])),
            "abs_d_endpoint_dev": abs(
                float(p["endpoint_dev_max"]) - float(b["endpoint_dev_max"])),
        })
    return out


def summarize_pairs(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *, arm: str, budget_mode: str, seed: str,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Condition comparison summary for one (arm, seed|"pooled", mode)."""
    comps = [compare_episode_pair(b, p) for b, p in pairs]
    n = len(comps)
    base_rate = collision_rate_summary(
        [c["baseline"]["has_collision"] for c in comps], alpha=alpha)
    pert_rate = collision_rate_summary(
        [c["perturbed"]["has_collision"] for c in comps], alpha=alpha)
    significant = bool(
        n > 0 and pert_rate["ci"] is not None
        and base_rate["rate"] is not None
        and pert_rate["ci"][0] > base_rate["rate"])
    dev_deltas = [abs(c["final_dev_target_delta"]) for c in comps
                  if c["final_dev_target_delta"] is not None]
    rmsd_deltas = [abs(c["final_rmsd_delta"]) for c in comps]

    # divergence-vs-step profile (median across pairs per step)
    prof_raw = [divergence_profile(b, p) for b, p in pairs]
    profile: list[dict[str, Any]] = []
    max_t = max((len(x) for x in prof_raw), default=0)
    for i in range(max_t):
        ap = [x[i]["abs_d_arc_progress"] for x in prof_raw if len(x) > i]
        ed = [x[i]["abs_d_endpoint_dev"] for x in prof_raw if len(x) > i]
        profile.append({
            "step": i + 1,
            "n_pairs": len(ap),
            "median_abs_d_arc_progress": float(np.median(ap)),
            "median_abs_d_endpoint_dev": float(np.median(ed)),
        })
    return {
        "arm": arm,
        "seed": seed,
        "budget_mode": budget_mode,
        "n_pairs": n,
        "collision_rate_baseline": base_rate,
        "collision_rate_perturbed": pert_rate,
        "collision_flag_changed": {
            "n": sum(1 for c in comps if c["collision_flag_changed"]),
            "n_new": sum(1 for c in comps if c["collision_new"]),
            "n_cleared": sum(1 for c in comps if c["collision_cleared"]),
        },
        "n_collisions_delta": _dist(
            [c["n_collisions_delta"] for c in comps]),
        "endpoint_dev_abs_delta": _dist(dev_deltas),
        "final_rmsd_abs_delta": _dist(rmsd_deltas),
        "endpoint_dev_baseline": _dist(
            [c["baseline"]["final_dev_target_max"] for c in comps]),
        "endpoint_dev_perturbed": _dist(
            [c["perturbed"]["final_dev_target_max"] for c in comps]),
        "stop_step_changed": {           # arm C online No-Move stop only
            "n_changed": sum(1 for c in comps if c["stop_step_changed"]),
            "fraction": (float(np.mean(
                [1.0 if c["stop_step_changed"] else 0.0 for c in comps]))
                if n else None),
            "baseline_online_halted": sum(
                1 for c in comps if c["baseline"]["stopped_online"]),
            "perturbed_online_halted": sum(
                1 for c in comps if c["perturbed"]["stopped_online"]),
        },
        "endpoint_reached": {
            "baseline": sum(1 for c in comps if c["baseline"]["endpoint_reached"]),
            "perturbed": sum(1 for c in comps if c["perturbed"]["endpoint_reached"]),
        },
        "start_step0_changed_fraction": (
            float(np.mean([1.0 if c["start_step0_changed"] else 0.0
                           for c in comps])) if n else None),
        "significant_collision_worsening": significant,
        "divergence_profile": profile,
        "episode_comparisons": comps,
    }


def decide_verdict(overall_summaries: Sequence[Mapping[str, Any]],
                   budget_mode: str) -> dict[str, Any]:
    """Plan §9.5 decision rule on the pooled autonomous evaluation."""
    primary = [s for s in overall_summaries
               if s["budget_mode"] == budget_mode]
    worse = [s for s in primary if s.get("significant_collision_worsening")]
    verdict = VERDICT_RETRAIN if worse else VERDICT_PARK
    return {
        "rule": ("collision rate significantly worse ⇔ perturbed "
                 "Clopper–Pearson CI lower bound > baseline collision "
                 "rate ⇒ recommend standalone retraining project; "
                 "else park the branch (plan §9.5)"),
        "evaluated_on": f"pooled seeds, budget_mode={budget_mode}",
        "arms_significantly_worse": [s["arm"] for s in worse],
        "verdict": verdict,
        "per_arm": [
            {"arm": s["arm"],
             "baseline_rate": s["collision_rate_baseline"]["rate"],
             "perturbed_ci_low": (s["collision_rate_perturbed"]["ci"] or
                                  [None, None])[0],
             "significant": bool(s["significant_collision_worsening"])}
            for s in primary],
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
        raise NoiseDiagError("SEEDS", f"bad integer list {value!r}") from exc


def load_dataset_contexts(
    tables_dir: Path, split_path: Path, *, n_curves: int,
    directions: Sequence[str], cache_curves: int,
) -> tuple[list[EpisodeContextV53MP], dict[str, Any]]:
    """Frozen split → selected curves → per-(curve, direction) contexts.

    The dataset is constructed ONCE here (the caller frees it before the
    episode loop); identical curve-selection discipline to the engine.
    """
    split_doc = json.loads(split_path.read_text(encoding="utf-8"))
    if split_doc.get("schema") != SPLIT_SCHEMA_V53MP:
        raise NoiseDiagError(
            "SPLIT_SCHEMA",
            f"{split_path}: schema {split_doc.get('schema')!r} != "
            f"{SPLIT_SCHEMA_V53MP!r}")
    if not split_doc.get("frozen"):
        raise NoiseDiagError("SPLIT_FROZEN", f"{split_path} is not frozen")
    subsets = split_doc.get("subsets") or {}
    dev_ids = [str(c) for c in subsets.get("dev") or []]
    confirm_ids = {str(c) for c in subsets.get("confirm") or []}
    pool_ids: set[str] = set()
    for tier_ids in (subsets.get("pool") or {}).values():
        pool_ids.update(str(c) for c in tier_ids)
    _require(dev_ids, "SPLIT_DEV", "split manifest has an empty dev subset")

    selected = select_rollout_curves(dev_ids, n_curves)
    if set(selected) & confirm_ids or set(selected) & pool_ids:
        raise NoiseDiagError(
            "SELECTION_NOT_DEV_ONLY",
            "selected curves overlap the sealed confirm / training pool")

    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (
        LABEL_STATUSES,
        ShardedKeyAtomDatasetV3,
    )

    dataset = ShardedKeyAtomDatasetV3(
        str(tables_dir),
        split_manifest=str(split_path),
        label_statuses=LABEL_STATUSES,
        light_cache=True,
        cache_curves=int(cache_curves),
    )
    intents = dataset.intents
    atomic_map = atomic_numbers_map(dataset, selected)
    contexts: list[EpisodeContextV53MP] = []
    for cid in selected:
        curve_intents = _direction_intents(intents, cid)
        for direction in directions:
            contexts.append(build_episode_context(
                dataset, curve_id=cid, direction=direction,
                intent=curve_intents[direction],
                atomic_numbers=atomic_map[cid]))
    del dataset, intents, atomic_map
    import gc
    gc.collect()
    meta = {
        "split": {"path": str(split_path),
                  "sha256": _sha256_file(split_path),
                  "frozen": True, "subset_used": "dev",
                  "selection_salt": SELECTION_SALT,
                  "n_curves": len(selected),
                  "curve_ids": selected,
                  "curve_list_sha256": curve_list_sha256(selected)},
        "directions": list(directions),
    }
    return contexts, meta


def run_noise_diagnostic(args: argparse.Namespace) -> int:
    t_total = time.time()
    out_dir = Path(args.out_dir)
    runs_root = Path(args.runs_root)
    tables_dir = Path(args.tables_dir)
    split_path = Path(args.split)
    anchor_dir = Path(getattr(args, "anchor_dir", DEFAULT_ANCHOR_DIR))

    # fail-closed campaign root BEFORE any output is opened
    _assert_v53mp_root(str(out_dir))
    _assert_v53mp_root(str(runs_root))
    _assert_v53mp_root(str(tables_dir))
    if not args.skip_anchor:
        _assert_v53mp_root(str(anchor_dir))

    tier = str(args.tier)
    smoke = bool(args.smoke)
    arms = _parse_csv_str(args.arms)
    for arm in arms:
        if arm not in ARMS_V53MP:
            raise NoiseDiagError(
                "ARM", f"unknown arm {arm!r}; known {ARMS_V53MP}")
    if len(set(arms)) != len(arms):
        raise NoiseDiagError("ARM", f"duplicate arms in {arms!r}")
    seeds = _parse_csv_int(args.seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise NoiseDiagError("SEEDS", f"bad seeds {seeds!r}")
    sigma = float(args.sigma)
    _require(math.isfinite(sigma) and sigma >= 0.0, "SIGMA",
             f"--sigma must be finite and >= 0, got {args.sigma!r}")
    noise_seed = int(args.noise_seed)
    _require(noise_seed >= 0, "NOISE_SEED",
             f"--noise-seed must be >= 0, got {args.noise_seed!r}")
    device = str(getattr(args, "device", "cpu"))

    if smoke:
        n_curves = SMOKE_N_CURVES
        directions = SMOKE_DIRECTIONS
        step_cap: int | None = SMOKE_BUDGET_CAP
        require_anchor = False
    else:
        n_curves = int(args.n_curves)
        directions = DIRECTIONS_V53MP
        step_cap = None
        require_anchor = not bool(args.skip_anchor)
    autonomous_steps = int(args.autonomous_steps)
    budget_modes = BUDGET_MODES

    # ---- contexts (dataset ONCE for all checkpoints) ----------------------
    t0 = time.time()
    contexts, split_meta = load_dataset_contexts(
        tables_dir, split_path, n_curves=n_curves, directions=directions,
        cache_curves=int(args.cache_curves))
    dataset_s = time.time() - t0

    # ---- checkpoints -------------------------------------------------------
    t0 = time.time()
    runs: dict[tuple[str, int], dict[str, Any]] = {}
    for arm in arms:
        for seed in seeds:
            runs[(arm, seed)] = _load_run_artifact(
                runs_root, tier, arm, seed)
    ckpt_s = time.time() - t0

    # ---- anchor (baseline reproduction) -----------------------------------
    run_ids = [art["run_id"] for art in runs.values()]
    anchor: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    if require_anchor or (anchor_dir / "episodes.jsonl").is_file():
        if not args.skip_anchor:
            anchor = load_anchor_episodes(anchor_dir, run_ids)

    # ---- per-checkpoint episodes ------------------------------------------
    t0 = time.time()
    baseline_records: list[dict[str, Any]] = []
    perturbed_records: list[dict[str, Any]] = []
    noise_meta_per_curve: dict[str, dict[str, Any]] = {}
    anchor_results: dict[str, dict[str, Any]] = {}
    noise_draws: dict[str, np.ndarray] = {}

    for (arm, seed), art in sorted(runs.items()):
        # (1) BASELINE REPRODUCTION anchor
        run_base: list[dict[str, Any]] = []
        for ctx in contexts:
            for budget_mode in budget_modes:
                budget = episode_budget(
                    budget_mode=budget_mode, l_key=ctx.l_key,
                    autonomous_steps=autonomous_steps, step_cap=step_cap)
                rec = run_episode(
                    ctx, arm=arm, model=art["model"], budget=budget,
                    budget_mode=budget_mode, run_id=art["run_id"],
                    checkpoint_sha256=art["sha256"], seed=seed,
                    tau_end=art["tau_end"], device=device)
                run_base.append(rec)
                baseline_records.append(rec)
        if anchor:
            verified = assert_baseline_anchor(
                run_base, anchor, require_all=require_anchor)
            anchor_results[art["run_id"]] = verified
            print(f"[anchor] {art['run_id']}: "
                  f"{verified['n_episodes_checked']}/{verified['n_rerun']} "
                  f"episodes bitwise-match the stored rollouts_s3 anchor",
                  flush=True)
        elif require_anchor:
            raise NoiseDiagError(
                "ANCHOR_EMPTY", f"no anchor episodes loaded for "
                f"{art['run_id']} from {anchor_dir}")

        # (2) PERTURBED — same curves/steps; noise once at t=0
        for ctx in contexts:
            if ctx.curve_id not in noise_draws:
                start = np.asarray(ctx.start_xyz, dtype=np.float64)
                noise_draws[ctx.curve_id] = start_noise_draw(
                    noise_seed, ctx.curve_id, int(start.shape[0]), sigma)
                nz = noise_draws[ctx.curve_id]
                noise_meta_per_curve[ctx.curve_id] = {
                    "sigma": sigma,
                    "noise_seed": noise_seed,
                    "n_atoms": int(nz.shape[0]),
                    "max_abs_atom_displacement": float(np.abs(nz).max()),
                    "max_atom_displacement": float(
                        np.linalg.norm(nz, axis=-1).max()),
                    "rms_atom_displacement": float(
                        np.sqrt((nz * nz).mean())),
                    "applied": "once_at_t0",
                }
            ctx_noisy, _noise = apply_start_noise(
                ctx, sigma=sigma, noise_seed=noise_seed)
            for budget_mode in budget_modes:
                budget = episode_budget(
                    budget_mode=budget_mode, l_key=ctx.l_key,
                    autonomous_steps=autonomous_steps, step_cap=step_cap)
                rec = run_episode(
                    ctx_noisy, arm=arm, model=art["model"], budget=budget,
                    budget_mode=budget_mode, run_id=art["run_id"],
                    checkpoint_sha256=art["sha256"], seed=seed,
                    tau_end=art["tau_end"], device=device)
                perturbed_records.append(rec)
        s_base = [(r["summary"]["n_collisions"] > 0) for r in run_base]
        print(f"[{art['run_id']}] baseline episodes={len(run_base)} "
              f"collision_flagged={sum(s_base)}", flush=True)
    episodes_s = time.time() - t0

    # ---- discipline + comparison ------------------------------------------
    all_records = baseline_records + perturbed_records
    check_claim_discipline(all_records)

    pairs = pair_episode_records(baseline_records, perturbed_records)
    per_run_summaries: list[dict[str, Any]] = []
    per_seed_summaries: list[dict[str, Any]] = []
    overall_summaries: list[dict[str, Any]] = []
    for arm in arms:
        for budget_mode in budget_modes:
            seed_summaries = []
            for seed in seeds:
                sub = [(b, p) for b, p in pairs
                       if b["arm"] == arm and int(b["seed"]) == seed
                       and b["budget_mode"] == budget_mode]
                s = summarize_pairs(
                    sub, arm=arm, budget_mode=budget_mode, seed=str(seed))
                per_seed_summaries.append(s)
                seed_summaries.append(s)
                per_run_summaries.append(s)
            pooled = [(b, p) for b, p in pairs
                      if b["arm"] == arm and b["budget_mode"] == budget_mode]
            overall_summaries.append(summarize_pairs(
                pooled, arm=arm, budget_mode=budget_mode, seed="pooled"))

    verdict = decide_verdict(overall_summaries, "autonomous")

    # ---- write outputs ------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    base_path = out_dir / "episodes_baseline.jsonl"
    with base_path.open("w", encoding="utf-8") as fh:
        for rec in baseline_records:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True)
                     + "\n")
    pert_path = out_dir / "episodes_perturbed.jsonl"
    with pert_path.open("w", encoding="utf-8") as fh:
        for rec in perturbed_records:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True)
                     + "\n")

    manifest: dict[str, Any] = {
        "schema": NOISE_MANIFEST_SCHEMA,
        "campaign": "pes2ts_v53mp_20261021",
        "diag_id": DIAG_ID,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "diagnostic_only_note": DIAGNOSTIC_ONLY_NOTE,
        "smoke": smoke,
        "smoke_caps": ({"n_curves": SMOKE_N_CURVES,
                        "directions": list(SMOKE_DIRECTIONS),
                        "budget_cap": SMOKE_BUDGET_CAP} if smoke else None),
        "tier": tier,
        "split": split_meta,
        "arms": arms,
        "seeds": seeds,
        "directions": list(directions),
        "budget_modes": list(budget_modes),
        "autonomous_steps": autonomous_steps,
        "step_cap": step_cap,
        "sigma_angstrom": sigma,
        "noise_seed": noise_seed,
        "noise_seed_formula": NOISE_SEED_FORMULA,
        "perturbation": PERTURBATION_NOTE,
        "rollout_seed": SEED_ROLLOUT,
        "feature_level": FEATURE_LEVEL,
        "device": device,
        "claim_scope": CLAIM_SCOPE,
        "verified_ts_claim": "forbidden",
        "checkpoints": {
            art["run_id"]: {
                "path": art["path"], "sha256": art["sha256"],
                "tau_end": art["tau_end"], "arm": art["arm"],
                "seed": art["seed"]}
            for art in runs.values()},
        "anchor": {
            "dir": str(anchor_dir),
            "skipped": bool(args.skip_anchor or not anchor),
            "per_run": anchor_results,
        },
        "noise_per_curve": noise_meta_per_curve,
        "n_episodes_baseline": len(baseline_records),
        "n_episodes_perturbed": len(perturbed_records),
        "episodes_baseline_file": str(base_path),
        "episodes_perturbed_file": str(pert_path),
        "episodes_schema": EPISODE_SCHEMA_V53MP,
        "wall_seconds": {
            "dataset_init": dataset_s,
            "checkpoint_load": ckpt_s,
            "episodes": episodes_s,
            "total": time.time() - t_total,
        },
    }
    manifest_path = out_dir / "noise_diag_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    report = build_report(
        manifest=manifest, per_seed=per_seed_summaries,
        overall=overall_summaries, verdict=verdict, pairs=pairs)
    report_json = Path(args.report_json)
    report_md = Path(args.report_md)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    report_md.write_text(render_report_md(report), encoding="utf-8")

    print(f"-> {base_path} ({len(baseline_records)} baseline episodes)")
    print(f"-> {pert_path} ({len(perturbed_records)} perturbed episodes)")
    print(f"-> {manifest_path}")
    print(f"-> {report_json}")
    print(f"-> {report_md}")
    print(f"verdict: {verdict['verdict']} "
          f"(rule: perturbed CI lower bound > baseline rate)")
    print(f"wall: dataset={dataset_s:.1f}s ckpt={ckpt_s:.1f}s "
          f"episodes={episodes_s:.1f}s total={time.time() - t_total:.1f}s")
    return 0


# ---------------------------------------------------------------------------
# report (machine + human)
# ---------------------------------------------------------------------------
def _slim_summary(s: Mapping[str, Any], *, keep_profile: bool) -> dict[str, Any]:
    out = {k: v for k, v in s.items()
           if k not in ("episode_comparisons", "divergence_profile")}
    if keep_profile:
        prof = s.get("divergence_profile") or []
        # keep a compact profile: first 10 steps + powers of two + last
        keep_idx = sorted(set(
            list(range(min(10, len(prof))))
            + [2 ** e for e in range(0, 20) if 2 ** e < len(prof)]
            + [len(prof) - 1]))
        out["divergence_profile_compact"] = [prof[i] for i in keep_idx]
    return out


def build_report(
    *, manifest: Mapping[str, Any], per_seed: Sequence[Mapping[str, Any]],
    overall: Sequence[Mapping[str, Any]], verdict: Mapping[str, Any],
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    comps = [compare_episode_pair(b, p) for b, p in pairs]
    exposure = [c for c in comps if c["collision_new"]]
    return {
        "schema": NOISE_REPORT_SCHEMA,
        "diag_id": DIAG_ID,
        "title": ("V5.3-MP noise-injection / exposure-bias diagnostic "
                  "(zero training cost, plan §9.5)"),
        "generated_at": manifest["generated_at"],
        "diagnostic_only_note": DIAGNOSTIC_ONLY_NOTE,
        "perturbation": PERTURBATION_NOTE,
        "noise_seed_formula": NOISE_SEED_FORMULA,
        "sigma_angstrom": manifest["sigma_angstrom"],
        "noise_seed": manifest["noise_seed"],
        "tier": manifest["tier"],
        "arms": manifest["arms"],
        "seeds": manifest["seeds"],
        "directions": manifest["directions"],
        "budget_modes": manifest["budget_modes"],
        "autonomous_steps": manifest["autonomous_steps"],
        "device": manifest["device"],
        "checkpoints": manifest["checkpoints"],
        "anchor": manifest["anchor"],
        "noise_per_curve": manifest["noise_per_curve"],
        "collision_exposure_episodes": {
            "n_pairs": len(comps),
            "n_collision_new": len(exposure),
            "n_collision_cleared": sum(1 for c in comps if c["collision_cleared"]),
            "examples": [
                {"run_id": c["run_id"], "curve_id": c["curve_id"],
                 "direction": c["direction"],
                 "baseline_first_collision_step": c["baseline"]["first_collision_step"],
                 "perturbed_first_collision_step": c["perturbed"]["first_collision_step"]}
                for c in exposure[:10]],
        },
        "per_seed": [_slim_summary(s, keep_profile=False) for s in per_seed],
        "overall": [_slim_summary(s, keep_profile=True) for s in overall],
        "verdict": verdict,
        "wall_seconds": manifest["wall_seconds"],
        "notes": {
            "separate_reporting": (
                "reference_normalized and autonomous results are "
                "aggregated separately — never a merged success rate"),
            "online_stop": (
                "arm C online No-Move stop (sigmoid(endpoint_logit) >= "
                "tau_end) is a diagnostic only"),
            "claim_scope": CLAIM_SCOPE,
            "no_training": (
                "frozen S3 checkpoints only; no training performed or "
                "consumed; NOT part of S4.1"),
        },
    }


def _fmt_rate(r: Mapping[str, Any] | None) -> str:
    if not r or r.get("rate") is None:
        return "n/a"
    lo, hi = r["ci"]
    return f"{r['k']}/{r['n']} = {r['rate']:.3f} (95% CI [{lo:.3f}, {hi:.3f}])"


def render_report_md(report: Mapping[str, Any]) -> str:
    v = report["verdict"]
    lines: list[str] = []
    a = lines.append
    a("# V5.3-MP 噪声注入 / exposure-bias 诊断报告（2026-10-23）")
    a("")
    a("> **诊断专用（DIAGNOSTIC ONLY）**：零训练成本（plan §9.5）；"
      "**不属于 S4.1**；仅使用 S3 冻结 checkpoint，未消耗任何训练。")
    a(f"> σ = {report['sigma_angstrom']} Å；noise_seed = "
      f"{report['noise_seed']}（{report['noise_seed_formula']}）")
    a(f"> 扰动方式：{report['perturbation']}")
    a("")
    a("## 1. 协议")
    a("")
    a("对每个 (arm × seed) checkpoint：")
    a("")
    a("1. **Baseline 复现锚**：以冻结曲线选择（salt `17`）重跑无扰动 "
      "rollout，逐 episode 与 `rollouts_s3` 存档精确比对（除 "
      "wall_seconds）；任何不一致 = harness bug ⇒ 停止。")
    a("2. **扰动组**：相同曲线 / 步数 / seed，仅起点 xyz 加 "
      "N(0, σ²) 每原子噪声（**只在 t=0 施加一次**，非逐步扰动）。")
    a("3. **比较**：碰撞 flag、终点 IRC 停差、arm C 在线停止步变化、"
      "逐步发散曲线（exposure-bias curve）。")
    a("")
    a("## 2. Baseline 复现锚")
    a("")
    a("判据：所有**确定性量**（碰撞/端点/停止 flag、步数、"
      "arc-progress 序列、stop step、input hash 等）必须与存档"
      "**逐位一致**；连续模型输出（d_key、坐标、RMSD 等）容差 "
      f"{DEFAULT_ANCHOR_CONTINUOUS_TOL:g} Å（不同 BLAS 线程数的 "
      "浮点归约序噪声，闭环放大；超出即 harness bug ⇒ 停止）。")
    a("")
    anchor = report["anchor"]
    if anchor.get("skipped"):
        a("- 本次为 smoke / 跳过模式：**未执行锚比对**。")
    else:
        for run_id, res in sorted((anchor.get("per_run") or {}).items()):
            a(f"- `{run_id}`: {res['n_episodes_checked']}/"
              f"{res['n_rerun']} episodes 确定性字段逐位一致；"
              f"连续字段最大偏差 "
              f"{res['max_continuous_abs_dev']:.2e} "
              f"（容差 {res['continuous_tolerance']:g}，共比较 "
              f"{res['n_continuous_values_compared']} 个连续值）✓")
    a("")
    a("## 3. 碰撞率：baseline vs 扰动（自主 200 步，pooled seeds）")
    a("")
    a("| arm | budget_mode | baseline 碰撞率（exact 95% CI） | "
      "扰动后碰撞率（exact 95% CI） | 显著恶化 |")
    a("|---|---|---|---|---|")
    for s in report["overall"]:
        a(f"| {s['arm']} | {s['budget_mode']} | "
          f"{_fmt_rate(s['collision_rate_baseline'])} | "
          f"{_fmt_rate(s['collision_rate_perturbed'])} | "
          f"{'是' if s['significant_collision_worsening'] else '否'} |")
    a("")
    a("显著恶化判据（plan §9.5）：扰动组 Clopper–Pearson CI 下界 > "
      "baseline 碰撞率。")
    a("")
    a("## 4. 每 checkpoint（arm × seed）碰撞率")
    a("")
    a("| run | budget_mode | baseline | 扰动后 | stop 步变化比例 |")
    a("|---|---|---|---|---|")
    for s in report["per_seed"]:
        frac = s["stop_step_changed"]["fraction"]
        a(f"| {s['arm']}_s{s['seed']} | {s['budget_mode']} | "
          f"{_fmt_rate(s['collision_rate_baseline'])} | "
          f"{_fmt_rate(s['collision_rate_perturbed'])} | "
          f"{'n/a' if frac is None else f'{frac:.3f}'} |")
    a("")
    a("## 5. 终点 IRC 偏差（|Δ final_dev_target_max|，pooled）")
    a("")
    a("| arm | budget_mode | median | max | baseline median | 扰动后 median |")
    a("|---|---|---|---|---|---|")

    def _f3(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.3f}"

    for s in report["overall"]:
        d = s["endpoint_dev_abs_delta"]
        a(f"| {s['arm']} | {s['budget_mode']} | "
          f"{_f3(d['median'])} | {_f3(d['max'])} | "
          f"{_f3(s['endpoint_dev_baseline']['median'])} | "
          f"{_f3(s['endpoint_dev_perturbed']['median'])} |")
    a("")
    a("## 6. 逐步发散曲线（exposure-bias curve，autonomous，pooled）")
    a("")
    a("| step | n_pairs | median \\|Δarc_progress\\| | "
      "median \\|Δendpoint_dev\\| (Å) |")
    a("|---|---|---|---|")
    for s in report["overall"]:
        if s["budget_mode"] != "autonomous":
            continue
        for row in s.get("divergence_profile_compact") or []:
            a(f"| {row['step']} | {row['n_pairs']} | "
              f"{row['median_abs_d_arc_progress']:.4f} | "
              f"{row['median_abs_d_endpoint_dev']:.4f} |")
    a("")
    a("## 7. 碰撞暴露 episode（baseline 无碰撞 → 扰动后出现碰撞）")
    a("")
    cx = report["collision_exposure_episodes"]
    a(f"- 可比 episode 对：{cx['n_pairs']}")
    a(f"- **新增碰撞（exposure）**：{cx['n_collision_new']}")
    a(f"- 扰动后消失的碰撞：{cx['n_collision_cleared']}")
    for ex in cx["examples"]:
        a(f"  - {ex['run_id']} {ex['curve_id']}/{ex['direction']}: "
          f"baseline 首次碰撞步={ex['baseline_first_collision_step']} → "
          f"扰动后={ex['perturbed_first_collision_step']}")
    a("")
    a("## 8. 判定（plan §9.5 decision rule）")
    a("")
    a(f"- 规则：{v['rule']}")
    a(f"- 评估口径：{v['evaluated_on']}")
    a(f"- 显著恶化 arms：{v['arms_significantly_worse'] or '无'}")
    if v["verdict"] == VERDICT_RETRAIN:
        a(f"- **结论：{v['verdict']}** —— 碰撞率随扰动显著上升，"
          "按 plan §9.5 立项独立重训练分支（noise-to-IRC）。")
    else:
        a(f"- **结论：{v['verdict']}** —— 碰撞率未随 σ=0.02 Å 扰动显著"
          "上升，噪声注入分支按 plan §9.5 暂缓（park），不立项重训练。")
    a("")
    a("## 9. 资源与产物")
    a("")
    w = report["wall_seconds"]
    a(f"- 墙钟：dataset={w['dataset_init']:.1f}s, episodes="
      f"{w['episodes']:.1f}s, total={w['total']:.1f}s；device="
      f"{report['device']}（rollout 推理极小，GPU 留给并发战役）")
    a("- 产物：`outputs/pesn/v53mp/rollouts_noise/"
      "{episodes_baseline.jsonl, episodes_perturbed.jsonl, "
      "noise_diag_manifest.json}` + 本报告")
    a("")
    a("## 10. 边界声明")
    a("")
    a("- 本诊断**只读**复用 `v53mp_rollout_exec.py`（未修改任何既有模块）；"
      "episode 循环、指标、聚合全部来自该 runner。")
    a("- 所有 episode 记录携带 `claim_scope = TS_candidate_only`；"
      "无任何 verified-TS 类声明。")
    a("- reference_normalized 与 autonomous 指标分开报告，无合并成功率。")
    a("- **本报告仅为诊断，不构成 S4.1 验收项，未消耗任何训练预算。**")
    a("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run-noise-diag")
    p.add_argument("--tables-dir", default="outputs/pesn/v53mp/data")
    p.add_argument("--split",
                   default="outputs/pesn/v53mp/data/split_manifest_v53mp.json")
    p.add_argument("--runs-root", default="outputs/pesn/v53mp")
    p.add_argument("--arms", default="C", help="comma list over {A,B,C}")
    p.add_argument("--seeds", default=DEFAULT_SEEDS,
                   help="comma list of trained-run seeds")
    p.add_argument("--n-curves", type=int, default=DEFAULT_N_CURVES)
    p.add_argument("--autonomous-steps", type=int,
                   default=DEFAULT_AUTONOMOUS_STEPS)
    p.add_argument("--sigma", type=float, default=DEFAULT_SIGMA,
                   help="per-atom Gaussian start-noise std (Å)")
    p.add_argument("--noise-seed", type=int, default=DEFAULT_NOISE_SEED)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--anchor-dir", default=DEFAULT_ANCHOR_DIR,
                   help="stored rollouts_s3 dir for the baseline "
                        "reproduction anchor")
    p.add_argument("--skip-anchor", action="store_true",
                   help="skip the baseline reproduction anchor "
                        "(diagnostic wiring only; NEVER for formal runs)")
    p.add_argument("--report-json", default=DEFAULT_REPORT_JSON)
    p.add_argument("--report-md", default=DEFAULT_REPORT_MD)
    p.add_argument("--tier", default="20k", choices=("2k", "5k", "20k"))
    p.add_argument("--cache-curves", type=int, default=DEFAULT_CACHE_CURVES)
    p.add_argument("--device", default="cpu",
                   help="model forward device (cpu default avoids GPU "
                        "contention with the training campaign)")
    p.add_argument("--smoke", action="store_true",
                   help="2 curves, forward only, budgets capped at 20; "
                        "baseline anchor check skipped")
    p.set_defaults(func=run_noise_diagnostic)
    args = parser.parse_args(argv)
    from scripts.agent_ops.supervisor.campaign_root import CampaignRootError

    try:
        return int(args.func(args))
    except (NoiseDiagError, RolloutExecError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    except CampaignRootError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
