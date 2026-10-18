"""V5 sprint key-atom BC losses + trainer + CLI (ADR-0026 §2.8, plan §4).

Schema/version: ``pesn_key_atom_bc_v1``.

Losses (plan §4, verbatim): radial Huber h_δ(e) = e²/2 (e≤δ) else
δ(e−δ/2), δ=0.25, on the normalized radial error e_i = ‖u_i−y_i‖/0.08;
per-frame degK weights w_i = 1/max(1, degK_i) normalized Σw_i=1;
L_pair = mean_l h((J_l(u)−J_l(y))/0.08) with J_l(u) = e_l·(u_j−u_i) and
e_l recomputed from the CURRENT geometry each batch; L_cos gated on the
teacher only (g_i = 1[‖y_i‖>0.02]) with soft-norm cosine (1e-4 Å);
λ_cos ramps 0 → 0.02 linearly over ``cos_ramp_updates`` after
``cos_start_update``; λ_pair = 0.5.  The batch loss is the plain mean
over frames — the reaction-balanced sampler already applies q weights,
so q is NOT multiplied again; the degK w_i normalization stays inside
each frame term.  Empty cos gate ⇒ L_cos = 0 exactly; all math is in Å;
non-finite values raise a typed abort (status ``aborted_nan``).

:class:`KeyAtomBCTrainer` implements the sprint §4/§5 recipe: fixed
update budget, Lane B pools for batch assembly (8 distinct reaction
groups × 4 frames, 2 per branch where possible, q-weighted draws from
the current cycle pool, cycle advance on exhaustion — INTEGRATION PENDING
on ``pes2ts_core.pesn.offline.data.key_atom_dataset``), paired
initialization/sample order across arms (``torch.manual_seed(seed)``
before model construction AND before sampler creation), AdamW
1e-3/1e-4 with linear warmup 250 then cosine to 1e-4, grad clip 1.0,
fp32, dev-fold DENSE eval every 250 updates with reaction-cluster
median/mean MAE/RMSE, RMSE/zero and RMSE/Classic ratios, effective-atom
cos medians (overall and near-TS), ``eval.jsonl`` /
``checkpoint_last.pt`` / ``checkpoint_final.pt`` / ``metrics.json``
(raw AND bounded via :func:`bounded_projection`) / ``status.json``.
Early stopping exists ONLY in screen mode (inner_val reaction-cluster
median RMSE/zero, patience 8, min_delta 0.005, no stop before update
500); formal mode runs the fixed budget with
``formal_outer_early_stopping=false``.  ``run_record.json`` carries the
ADR-0025 ``gv`` block validated by ``validate_gv_run_record``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping, Sequence

import numpy as np
import torch

from pes2ts_core.config_loader import load_config_mapping
from pes2ts_core.gv_registry import load_gv_map, validate_gv_run_record
from pes2ts_core.pesn.contracts.key_atom import KeyAtomBatch
from pes2ts_core.pesn.models.key_atom_classic import (
    MAX_ATOM_CAP,
    classic_pair_driver,
)
from pes2ts_core.pesn.models.key_atom_egnn import A_REF, StarEGNN

SCHEMA_VERSION: Final[str] = "pesn_key_atom_bc_v1"
RUN_RECORD_SCHEMA: Final[str] = "pesn_key_atom_bc_run_v1"
CHECKPOINT_SCHEMA: Final[str] = "pesn_key_atom_bc_checkpoint_v1"
CAMPAIGN_ID: Final[str] = "V5_SPRINT"
EXPECTED_CONFIG_SCHEMA: Final[str] = "pesn_v5_sprint_draft2"

#: Status vocabulary (plan §5.7 / sprint contract).
STATUS_COMPLETE: Final[str] = "complete"
STATUS_CENSORED_SCREEN: Final[str] = "censored_screen"
STATUS_ABORTED_DATA: Final[str] = "aborted_data"
STATUS_ABORTED_NAN: Final[str] = "aborted_nan"
STATUS_VALUES: Final[tuple[str, ...]] = (
    STATUS_COMPLETE, STATUS_CENSORED_SCREEN,
    STATUS_ABORTED_DATA, STATUS_ABORTED_NAN)


class KeyAtomBCTrainerError(RuntimeError):
    """Typed trainer refusal carrying a machine-readable code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class KeyAtomLossError(RuntimeError):
    """Typed loss failure (non-finite math or malformed tensors)."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise KeyAtomBCTrainerError(code, detail)


def _require_loss(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise KeyAtomLossError(code, detail)


# ---------------------------------------------------------------------------
# Loss configuration + λ_cos ramp (plan §4)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class KeyAtomLossConfig:
    """Plan §4 loss hyper-parameters (sprint yaml ``pesn_v5_sprint_draft2``)."""

    loss_scale_angstrom: float = A_REF      # a_ref = 0.08 Å
    huber_delta: float = 0.25
    lambda_pair: float = 0.5
    lambda_cos_max: float = 0.02
    cos_start_update: int = 250
    cos_ramp_updates: int = 250
    tau_norm_angstrom: float = 0.02
    cos_soft_norm_angstrom: float = 1e-4

    @classmethod
    def from_campaign_config(cls, config: Mapping) -> "KeyAtomLossConfig":
        return cls(
            loss_scale_angstrom=float(config.get("a_ref_angstrom", A_REF)),
            huber_delta=float(config.get("huber_delta_normalized", 0.25)),
            lambda_pair=float(config.get("lambda_pair", 0.5)),
            lambda_cos_max=float(config.get("lambda_cos", 0.02)),
            cos_start_update=int(config.get("cos_start_update", 250)),
            cos_ramp_updates=int(config.get("cos_ramp_updates", 250)),
            tau_norm_angstrom=float(config.get("tau_norm_angstrom", 0.02)),
            cos_soft_norm_angstrom=float(
                config.get("cos_soft_norm_angstrom", 1e-4)),
        )


def cos_lambda(update: int, config: KeyAtomLossConfig) -> float:
    """λ_cos at a 1-based update count (plan §4 ramp).

    Exactly 0 before ``cos_start_update``, linear to ``lambda_cos_max``
    over the next ``cos_ramp_updates`` updates, constant afterwards.
    """
    start = int(config.cos_start_update)
    ramp = max(int(config.cos_ramp_updates), 1)
    if update < start:
        return 0.0
    frac = min(1.0, (update - start) / ramp)
    return float(config.lambda_cos_max) * frac


# ---------------------------------------------------------------------------
# Loss building blocks (plan §4 EXACT)
# ---------------------------------------------------------------------------

def radial_huber(e: torch.Tensor, delta: float = 0.25) -> torch.Tensor:
    """h_δ(e) = e²/2 (e≤δ) else δ(e−δ/2); rotation-invariant by construction
    because it consumes only the radial error e = ‖Δu‖/a_ref."""
    quad = 0.5 * e * e
    linear = delta * (e - 0.5 * delta)
    return torch.where(e <= delta, quad, linear)


def degk_frame_weights(degK: torch.Tensor) -> torch.Tensor:
    """w_i = 1/max(1, degK_i) normalized so Σw_i = 1 (per frame)."""
    w = 1.0 / degK.clamp(min=1.0)
    total = w.sum()
    _require_loss(bool(torch.isfinite(total)) and float(total) > 0.0,
                  "WEIGHTS_DEGENERATE", "per-frame degK weights sum to 0")
    return w / total


def pair_unit_vectors(
    pos: torch.Tensor,
    key_index: torch.Tensor,
    pair_index: torch.Tensor,
) -> torch.Tensor:
    """e_l = (x_j − x_i)/d_ij from the CURRENT geometry (recomputed per
    batch; plan §4).  ``pair_index`` indexes key-atom slots."""
    if pair_index.numel() == 0:
        return torch.zeros(0, 3, dtype=pos.dtype, device=pos.device)
    pi = pair_index[0].long()
    pj = pair_index[1].long()
    x = pos[key_index]
    d = x[pj] - x[pi]
    dist = d.norm(dim=-1)
    safe = dist > 1e-12
    return torch.where(
        safe.unsqueeze(-1), d / dist.clamp(min=1e-12).unsqueeze(-1),
        torch.zeros_like(d))


def pair_projections(
    u: torch.Tensor, pair_index: torch.Tensor, e_l: torch.Tensor
) -> torch.Tensor:
    """J_l(u) = e_l·(u_j − u_i) — linear in u (key-slot pair indices)."""
    if pair_index.numel() == 0:
        return torch.zeros(0, dtype=u.dtype, device=u.device)
    pi = pair_index[0].long()
    pj = pair_index[1].long()
    return (e_l * (u[pj] - u[pi])).sum(dim=-1)


def soft_cosine(
    u: torch.Tensor, y: torch.Tensor, soft: float = 1e-4
) -> torch.Tensor:
    """Soft-norm cosine per atom (plan §4): the 1e-4 Å soft norm keeps the
    zero附近 gradient finite."""
    num = (u * y).sum(dim=-1)
    den = torch.sqrt(u.pow(2).sum(dim=-1) + soft * soft) * torch.sqrt(
        y.pow(2).sum(dim=-1) + soft * soft)
    return num / den.clamp(min=1e-12)


def key_atom_bc_loss(
    u: torch.Tensor,
    y: torch.Tensor,
    degK: torch.Tensor,
    key_ptr: torch.Tensor,
    pos: torch.Tensor,
    key_index: torch.Tensor,
    pair_index: torch.Tensor | None = None,
    *,
    config: KeyAtomLossConfig | None = None,
    lambda_cos: float = 0.0,
    frame_mask: torch.Tensor | None = None,
) -> dict:
    """Total loss + components (plan §4): per frame
    ``L_atom + λ_pair·L_pair + λ_cos·L_cos`` with w_i normalized INSIDE
    the frame term; batch loss = mean over unmasked frames (q weights are
    the sampler's job and are NOT re-applied here).

    Raises ``KeyAtomLossError`` on non-finite inputs or results; an empty
    cos gate yields L_cos = 0 exactly.
    """
    cfg = config or KeyAtomLossConfig()
    _require_loss(bool(torch.isfinite(u).all()),
                  "NONFINITE_PREDICTION", "u contains non-finite values")
    _require_loss(bool(torch.isfinite(y).all()),
                  "NONFINITE_TEACHER", "y contains non-finite values")
    if pair_index is not None and pair_index.numel() > 0:
        from pes2ts_core.pesn.contracts.key_atom import (
            global_pair_rows_to_slots,
        )
        pair_index = global_pair_rows_to_slots(
            pair_index, key_index, int(pos.shape[0]))
    n_graphs = int(key_ptr.shape[0]) - 1
    scale = float(cfg.loss_scale_angstrom)
    total = torch.zeros((), dtype=u.dtype, device=u.device)
    n_used = 0
    l_atom_acc = torch.zeros((), dtype=u.dtype, device=u.device)
    l_pair_acc = torch.zeros((), dtype=u.dtype, device=u.device)
    l_cos_acc = torch.zeros((), dtype=u.dtype, device=u.device)
    for b in range(n_graphs):
        if frame_mask is not None and not bool(frame_mask[b]):
            continue
        lo, hi = int(key_ptr[b]), int(key_ptr[b + 1])
        u_b, y_b, w_b = u[lo:hi], y[lo:hi], degK[lo:hi]
        m = hi - lo
        if m == 0:
            continue
        w = degk_frame_weights(w_b)
        # L_atom = Σ w_i h(||u−y||/a_ref)
        e = (u_b - y_b).norm(dim=-1) / scale
        l_atom = (w * radial_huber(e, cfg.huber_delta)).sum()
        # L_pair = mean over this frame's key pairs
        l_pair = torch.zeros((), dtype=u.dtype, device=u.device)
        if pair_index is not None and pair_index.numel() > 0:
            in_frame = ((pair_index[0] >= lo) & (pair_index[0] < hi)
                        & (pair_index[1] >= lo) & (pair_index[1] < hi))
            if bool(in_frame.any()):
                sub = pair_index[:, in_frame]
                e_l = pair_unit_vectors(pos, key_index, sub)
                dj = pair_projections(u_b, sub - lo, e_l)
                dy = pair_projections(y_b, sub - lo, e_l)
                l_pair = radial_huber(
                    (dj - dy).abs() / scale, cfg.huber_delta).mean()
        # L_cos: teacher-magnitude gate only (never the prediction)
        g = (y_b.norm(dim=-1) > float(cfg.tau_norm_angstrom)).to(u.dtype)
        wg = w * g
        denom = wg.sum()
        if float(denom) > 0.0:
            c = soft_cosine(u_b, y_b, cfg.cos_soft_norm_angstrom)
            l_cos = (wg * (1.0 - c)).sum() / denom.clamp(min=1e-8)
        else:
            l_cos = torch.zeros((), dtype=u.dtype, device=u.device)
        frame = l_atom + float(cfg.lambda_pair) * l_pair + \
            float(lambda_cos) * l_cos
        total = total + frame
        l_atom_acc = l_atom_acc + l_atom
        l_pair_acc = l_pair_acc + l_pair
        l_cos_acc = l_cos_acc + l_cos
        n_used += 1
    _require_loss(n_used > 0, "NO_ELIGIBLE_FRAMES",
                  "batch has no unmasked frames with key atoms")
    total = total / n_used
    _require_loss(bool(torch.isfinite(total)),
                  "NONFINITE_LOSS", "total loss is not finite")
    return {
        "loss": total,
        "l_atom": l_atom_acc / n_used,
        "l_pair": l_pair_acc / n_used,
        "l_cos": l_cos_acc / n_used,
        "lambda_cos": float(lambda_cos),
        "n_frames": n_used,
    }


# ---------------------------------------------------------------------------
# Bounded export (plan §4: raw kept; u_bounded = ONE uniform factor/frame)
# ---------------------------------------------------------------------------

def bounded_projection(
    u: torch.Tensor, d0: torch.Tensor | float, key_ptr: torch.Tensor, *,
    max_atom_cap: float = MAX_ATOM_CAP,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single common per-frame factor so max_atom ≤ 0.20 Å AND RMS ≤ d0.

    Returns ``(u_bounded [Σm,3], scale [B])`` — the applied factors are
    recorded so nothing is hidden by the truncation.
    """
    out = u.clone()
    n_graphs = int(key_ptr.shape[0]) - 1
    scale = torch.ones(n_graphs, dtype=torch.float32)
    d0_t = (d0 if torch.is_tensor(d0)
            else torch.full((n_graphs,), float(d0), dtype=torch.float32))
    for b in range(n_graphs):
        lo, hi = int(key_ptr[b]), int(key_ptr[b + 1])
        if hi <= lo:
            continue
        norms = u[lo:hi].norm(dim=-1)
        max_atom = float(norms.max())
        rms = float(torch.sqrt(torch.mean(norms * norms)))
        f = 1.0
        if max_atom > 1e-12:
            f = min(f, float(max_atom_cap) / max_atom)
        d0_b = float(d0_t[b])
        if rms > 1e-12 and d0_b > 0.0:
            f = min(f, d0_b / rms)
        out[lo:hi] = u[lo:hi] * f
        scale[b] = f
    return out, scale


# ---------------------------------------------------------------------------
# Eval metrics (per reaction cluster + cluster-median aggregation)
# ---------------------------------------------------------------------------

def _median(values: Sequence[float]) -> float | None:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return None
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def compute_cluster_metrics(
    per_sample: Sequence[dict], *,
    soft: float = 1e-4, tau: float = 0.02, near_ts_cut: float = 0.20,
) -> tuple[dict, dict]:
    """Reaction-cluster metrics + cluster-median aggregation.

    Each ``per_sample`` entry: ``group_id``, ``u`` [m,3], ``y`` [m,3],
    ``u_classic`` [m,3] or None, ``near_ts`` float.  Per cluster: median/
    mean MAE & RMSE (Å), RMSE/zero and RMSE/Classic ratios, effective-atom
    (‖y‖>τ) soft-cosine median overall and for near-TS samples.  The
    aggregate takes the median over clusters for every scalar.
    """
    clusters: dict[str, list[dict]] = {}
    for s in per_sample:
        clusters.setdefault(str(s["group_id"]), []).append(s)
    per_cluster: dict[str, dict] = {}
    for gid, samples in sorted(clusters.items()):
        errs: list[np.ndarray] = []
        y_norms: list[np.ndarray] = []
        cos_eff: list[float] = []
        cos_near: list[float] = []
        errs_classic: list[np.ndarray] = []
        has_classic = True
        for s in samples:
            u = np.asarray(s["u"], dtype=np.float64)
            y = np.asarray(s["y"], dtype=np.float64)
            e = np.linalg.norm(u - y, axis=1)
            errs.append(e)
            y_norms.append(np.linalg.norm(y, axis=1))
            eff = np.linalg.norm(y, axis=1) > tau
            if bool(eff.any()):
                c = soft_cosine(
                    torch.as_tensor(u[eff], dtype=torch.float64),
                    torch.as_tensor(y[eff], dtype=torch.float64),
                    soft).numpy()
                cos_eff.extend(c.tolist())
                if float(s.get("near_ts", 1.0)) <= near_ts_cut:
                    cos_near.extend(c.tolist())
            uc = s.get("u_classic")
            if uc is None:
                has_classic = False
            else:
                errs_classic.append(
                    np.linalg.norm(
                        np.asarray(uc, dtype=np.float64) - y, axis=1))
        e_all = np.concatenate(errs) if errs else np.zeros(1)
        y_all = np.concatenate(y_norms) if y_norms else np.zeros(1)
        mae = float(e_all.mean())
        rmse = float(np.sqrt(np.mean(e_all ** 2)))
        rmse_zero = float(np.sqrt(np.mean(y_all ** 2)))
        entry = {
            "n_samples": len(samples),
            "mae": mae,
            "rmse": rmse,
            "rmse_over_zero": rmse / max(rmse_zero, 1e-12),
            "cos_eff_median": _median(cos_eff),
            "cos_eff_near_ts_median": _median(cos_near),
        }
        if has_classic and errs_classic:
            e_cls = np.concatenate(errs_classic)
            rmse_cls = float(np.sqrt(np.mean(e_cls ** 2)))
            entry["rmse_over_classic"] = rmse / max(rmse_cls, 1e-12)
        else:
            entry["rmse_over_classic"] = None
        per_cluster[gid] = entry
    agg_keys = (
        "mae", "rmse", "rmse_over_zero", "rmse_over_classic",
        "cos_eff_median", "cos_eff_near_ts_median")
    aggregate = {
        k: _median([per_cluster[g][k] for g in per_cluster])
        for k in agg_keys}
    aggregate["n_clusters"] = len(per_cluster)
    return per_cluster, aggregate


# ---------------------------------------------------------------------------
# Sampler (plan §5.2: 8 groups × 4 frames, 2/branch, q-weighted, cycle pool)
# ---------------------------------------------------------------------------

def _entry_fields(entry) -> tuple[str, float]:
    """Tolerant (sample_id, q_weight) access for Lane B pool entries.

    Accepts objects with attributes or dict rows; INTEGRATION PENDING on
    the exact ``key_atom_dataset`` pool row type.
    """
    if isinstance(entry, Mapping):
        sid = entry.get("sample_id", entry.get("id"))
        q = entry.get("q_weight", entry.get("q", 1.0))
    else:
        sid = getattr(entry, "sample_id", getattr(entry, "id", None))
        q = getattr(entry, "q_weight", getattr(entry, "q", 1.0))
    _require(sid is not None, "POOL_ENTRY_INVALID",
             f"pool entry has no sample_id: {entry!r}")
    return str(sid), float(q)


class KeyAtomCycleSampler:
    """Seeded, paired reaction-balanced sampler (plan §5.2 / §4).

    Per batch: 8 distinct groups (uniform without replacement), each with
    4 frames — 2 per branch where the branch pool allows, otherwise as
    many as exist with the remainder from the other branch.  Frames are
    drawn from the group's current cycle pool; within a cycle every frame
    is drawn at most once (q-weighted order), then the cycle advances
    (fresh seeded shuffle of the same branch pool — Lane B rotation hooks
    in via ``pools`` refresh).  The RNG is seeded once per run so the
    sample sequence is identical across arms for the same seed.
    """

    def __init__(
        self,
        pools: Mapping[str, Mapping[int, Sequence]],
        *,
        seed: int,
        reactions_per_batch: int = 8,
        frames_per_reaction: int = 4,
    ) -> None:
        # pools: {group_id: {branch(int): [pool rows]}}
        self.pools: dict[str, dict[int, list]] = {}
        for g, branches in pools.items():
            self.pools[str(g)] = {
                int(b): list(rows) for b, rows in dict(branches).items()}
        if not self.pools:
            raise KeyAtomBCTrainerError(
                "EMPTY_POOLS", "sampler received no reaction groups")
        self.seed = int(seed)
        self.reactions_per_batch = int(reactions_per_batch)
        self.frames_per_reaction = int(frames_per_reaction)
        self.rng = np.random.default_rng(self.seed)
        self._order: dict[tuple[str, int], list[int]] = {}
        self._cycle: dict[tuple[str, int], int] = {}
        self.draws = 0
        self._reset_all_cycles()

    def _frames(self, group: str, branch: int) -> list[tuple[str, float]]:
        entries = self.pools.get(group, {}).get(branch, [])
        return [_entry_fields(e) for e in entries]

    def _reset_cycle(self, group: str, branch: int) -> None:
        frames = self._frames(group, branch)
        n = len(frames)
        # q-weighted without-replacement order: repeated Gumbel-top-k
        if n == 0:
            self._order[(group, branch)] = []
        else:
            q = np.asarray([max(f[1], 0.0) for f in frames], dtype=np.float64)
            if float(q.sum()) <= 0.0:
                q = np.ones(n, dtype=np.float64)
            keys = -np.log(-np.log(self.rng.random(n)) + 1e-12) - np.log(q)
            self._order[(group, branch)] = list(np.argsort(keys))
        self._cycle[(group, branch)] = \
            self._cycle.get((group, branch), 0) + 1

    def _reset_all_cycles(self) -> None:
        for g in sorted(self.pools):
            for b in self.pools[g]:
                self._reset_cycle(g, int(b))

    def _take(self, group: str, branch: int, count: int) -> list[str]:
        frames = self._frames(group, branch)
        out: list[str] = []
        while len(out) < count:
            order = self._order.get((group, branch), [])
            if not order:
                self._reset_cycle(group, branch)
                order = self._order.get((group, branch), [])
                if not order:
                    break  # branch pool empty
            idx = order.pop(0)
            out.append(frames[idx][0])
        return out

    def draw_batch(self) -> list[dict]:
        """One batch of ``reactions_per_batch × frames_per_reaction`` rows:
        ``{"group_id", "branch", "sample_id"}``."""
        groups = sorted(self.pools)
        chosen = self.rng.choice(
            len(groups), size=min(self.reactions_per_batch, len(groups)),
            replace=False)
        per_branch = self.frames_per_reaction // 2
        rows: list[dict] = []
        for gi in chosen:
            group = groups[int(gi)]
            branches = sorted(self.pools[group], key=int)
            taken: dict[int, list[str]] = {}
            for b in branches:
                taken[int(b)] = self._take(group, int(b), per_branch)
            # top up to frames_per_reaction from whichever branch remains
            deficit = self.frames_per_reaction - sum(
                len(v) for v in taken.values())
            for b in branches:
                if deficit <= 0:
                    break
                extra = self._take(group, int(b), deficit + per_branch)
                take_n = min(deficit, len(extra))
                taken[int(b)][:0] = extra[:take_n]
                deficit -= take_n
            for b in sorted(taken):
                for sid in taken[b]:
                    rows.append(
                        {"group_id": group, "branch": b, "sample_id": sid})
        self.draws += len(rows)
        return rows

    def state_dict(self) -> dict:
        return {"seed": self.seed, "draws": int(self.draws),
                "cycles": {f"{g}|{b}": c for (g, b), c in self._cycle.items()}}

    def load_state_dict(self, state: Mapping) -> None:
        if int(state["seed"]) != self.seed:
            raise KeyAtomBCTrainerError(
                "SAMPLER_SEED_MISMATCH", "cannot restore sampler state")
        self.draws = int(state["draws"])


# ---------------------------------------------------------------------------
# LR schedule: linear warmup then cosine to min_lr (plan §4)
# ---------------------------------------------------------------------------

def lr_at_update(
    update: int, *, base_lr: float, warmup: int, total: int, min_lr: float
) -> float:
    """1-based update → learning rate (linear warmup, then cosine)."""
    if warmup > 0 and update < warmup:
        return base_lr * update / warmup
    span = max(total - warmup, 1)
    t = min(1.0, (update - warmup) / span)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))


# ---------------------------------------------------------------------------
# Lane B integration (key_atom_dataset runtime adapters)
# ---------------------------------------------------------------------------

_LANE_B_CACHE: dict[tuple[str, str], dict] = {}


def _split_group_reactions(split_path: str | Path, fold: int,
                           config: Mapping | None = None):
    """Read pesn_v5_split_v1 (folds/group_fold + fit50 selection) ->
    (train_groups, dev_groups)."""
    import json as _json

    doc = _json.loads(Path(split_path).read_text(encoding="utf-8"))
    folds = doc.get("folds")
    group_fold = doc.get("group_fold")
    if not isinstance(folds, dict) or not isinstance(group_fold, dict):
        raise KeyAtomBCTrainerError(
            "SPLIT_EMPTY", f"no folds/group_fold tables in {split_path}")
    paths = dict((config or {}).get("paths", {}))
    sel_path = Path(paths.get(
        "fit50_selection",
        "outputs/pesn/direct_egnn_v1_20261015/splits/selection_direct_fit50.json"))
    sel = _json.loads(sel_path.read_text(encoding="utf-8"))
    group_rids: dict[str, list[str]] = {}
    for gid, rid in zip(sel.get("group_ids", []), sel.get("reaction_ids", [])):
        group_rids.setdefault(str(gid), []).append(str(rid))
    train: dict[str, list[str]] = {}
    dev: dict[str, list[str]] = {}
    for gid, rids in group_rids.items():
        f = group_fold.get(gid)
        if f is None:
            raise KeyAtomBCTrainerError(
                "SPLIT_EMPTY", f"group {gid} absent from group_fold table")
        bucket = dev if int(f) == int(fold) else train
        bucket[gid] = rids
    if not train and not dev:
        raise KeyAtomBCTrainerError(
            "SPLIT_EMPTY", f"no fit50 rows parsed from {split_path}")
    return train, dev


def _lane_b_runtime(config: Mapping, d0: float, split_path: str | Path):
    """Cached shard list + dataset + vocabs for one (feature_root, d0)."""
    from pes2ts_core.pesn.offline.data import key_atom_dataset as lane_b

    paths = dict(config.get("paths", {}))
    feature_root = str(paths.get("feature_root", "outputs/pesn/v5/features"))
    d0_tag = f"{float(d0):.2f}"
    key = (feature_root, d0_tag)
    cached = _LANE_B_CACHE.get(key)
    if cached is None:
        shards = sorted(
            (Path(feature_root) / f"d0_{d0_tag}").glob("*.branch*.npz"))
        if not shards:
            raise KeyAtomBCTrainerError(
                "DATA_LANE_PENDING",
                f"no feature shards under {feature_root}/d0_{d0_tag}; "
                "run stage prepare-audit first")
        dataset = lane_b.V5FeatureDataset([str(p) for p in shards])
        cached = {"shards": shards, "dataset": dataset,
                  "shard_sizes": list(dataset._shard_index[1:]
                                      - dataset._shard_index[:-1])}
        _LANE_B_CACHE[key] = cached
    return cached


def _sample_id(shard_idx: int, sample_idx: int) -> str:
    return f"s{int(shard_idx)}:i{int(sample_idx)}"


def _parse_sample_id(sid: str, runtime: dict) -> int:
    s, i = str(sid).split(":")
    shard_idx = int(s[1:])
    sample_idx = int(i[1:])
    offset = 0
    for k in range(shard_idx):
        offset += int(runtime["shard_sizes"][k])
    return offset + sample_idx


def load_lane_b_pools(config: Mapping, *, fold: int, d0: float, split_path: str):
    """Train cycle pools + dev dense rows (plan §5.2), Lane B API bridge.

    Returns ``{"train": {gid: {branch: [row]}}, "dev_dense": [row]}`` with
    rows ``{"sample_id", "q_weight"}``; ``sample_id`` encodes
    ``(shard_idx, sample_idx)`` over the sorted shard list of the d0 root.
    """
    from pes2ts_core.pesn.offline.data import key_atom_dataset as lane_b

    train_groups, dev_groups = _split_group_reactions(split_path, fold, config)
    paths = dict(config.get("paths", {}))
    feature_root = str(paths.get("feature_root", "outputs/pesn/v5/features"))
    d0_tag = f"{float(d0):.2f}"
    pools = lane_b.build_pools(feature_root, train_groups,
                               max_per_branch_cycle=int(
                                   config.get(
                                       "max_start_frames_per_branch_cycle",
                                       16)),
                               d0_tag=d0_tag)
    train: dict[str, dict[int, list[dict]]] = {}
    for gid, branches in pools.items():
        train[gid] = {}
        for branch, cycles in dict(branches).items():
            rows: list[dict] = []
            for cycle in cycles:
                for shard_idx, sample_idx, q in cycle:
                    rows.append({"sample_id": _sample_id(shard_idx, sample_idx),
                                 "q_weight": float(q)})
            train[gid][int(branch)] = rows
    runtime = _lane_b_runtime(config, d0, split_path)
    dev_rids = {rid for rids in dev_groups.values() for rid in rids}
    dev_rows: list[dict] = []
    for shard_idx, path in enumerate(runtime["shards"]):
        rid = path.name.split(".branch")[0]
        if rid not in dev_rids:
            continue
        with np.load(path, allow_pickle=False) as data:
            eligible = np.asarray(data["eligible"], dtype=bool)
            q = np.asarray(data["q_weight"], dtype=np.float64)
        for sample_idx in np.nonzero(eligible)[0]:
            dev_rows.append({"sample_id": _sample_id(shard_idx, sample_idx),
                             "q_weight": float(q[sample_idx])})
    if not train:
        raise KeyAtomBCTrainerError(
            "EMPTY_POOLS", "no train pools resolved from split")
    return {"train": train, "dev_dense": dev_rows}


class _SampleView:
    """Index-mapped view over Lane B's V5FeatureDataset for sample rows."""

    def __init__(self, dataset, indices):
        self._dataset = dataset
        self._indices = list(int(i) for i in indices)

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, j: int):
        return self._dataset[self._indices[j]]


def load_lane_b_view(*, config: Mapping, samples: Sequence[dict], fold: int,
                     d0: float, anchor: str, split_path: str, device):
    """Resolve sampler rows to (view, collate) over the shared dataset."""
    from pes2ts_core.pesn.offline.data import key_atom_dataset as lane_b

    runtime = _lane_b_runtime(config, d0, split_path)
    indices = [_parse_sample_id(row["sample_id"], runtime) for row in samples]
    view = _SampleView(runtime["dataset"], indices)
    vkey = ("vocabs", str(runtime["shards"][0].parent), f"fold{int(fold)}")
    vocabs = _LANE_B_CACHE.get(vkey)
    if vocabs is None:
        train_groups, _dev = _split_group_reactions(split_path, fold, config)
        train_rids = [rid for rids in train_groups.values() for rid in rids]
        paths = dict(config.get("paths", {}))
        feature_root = str(paths.get("feature_root",
                                     "outputs/pesn/v5/features"))
        vocabs = lane_b.build_vocabs(feature_root, train_rids)
        _LANE_B_CACHE[vkey] = vocabs
    method_vocab, solvent_vocab = vocabs
    collate = runtime["dataset"].get_collate_fn(
        anchor_mode=(str(anchor).lower() == "on"),
        method_vocab=method_vocab, solvent_vocab=solvent_vocab,
        device=device)
    return view, collate


# ---------------------------------------------------------------------------
# Provenance helpers (ADR-0025 gv block)
# ---------------------------------------------------------------------------

def _sha256_or_none(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(repo_root: Path | None = None) -> str:
    """HEAD commit via direct ``.git`` reads (no subprocess; truth-guard safe)."""
    try:
        root = Path(repo_root) if repo_root is not None else Path.cwd()
        git = root / ".git"
        if git.is_file():  # worktree pointer
            target = Path(git.read_text(encoding="utf-8").strip().split(
                "gitdir:", 1)[-1].strip())
            if not target.is_absolute():
                target = root / target
            git = target
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref = git / head[5:].strip()
            if ref.is_file():
                return ref.read_text(encoding="utf-8").strip()
            return "unknown"
        return head or "unknown"
    except OSError:
        return "unknown"


def build_gv_block(*, config_path: Path, repo_root: Path) -> dict:
    """ADR-0025 ``gv`` provenance block for a V5_SPRINT run record."""
    gv_map = load_gv_map()
    data_manifest = repo_root / "data/manifests/v5/split_manifest.json"
    map_path = Path(gv_map.path)
    return {
        "map_version": gv_map.map_version,
        "campaign_id": CAMPAIGN_ID,
        "primary_g": "G3.P.0",
        "supporting_g": ["G3.P.1", "G3.P.2", "G3.P.3", "G0", "G1"],
        "gates": ["V5-G1", "V5-G2", "V5-G3", "V5-G4"],
        "code_commit": _git_commit(),
        "config_sha256": _sha256_or_none(config_path),
        "data_manifest_sha256": _sha256_or_none(data_manifest),
        "map_sha256": _sha256_or_none(map_path),
    }


def write_run_record(
    run_dir: Path, *, payload: Mapping, gv: Mapping, path: str = "run_record.json"
) -> None:
    """Write (or refresh) the run record; the gv block is validated via the
    ADR-0025 registry before anything hits disk."""
    doc = {**dict(payload), "gv": dict(gv)}
    validate_gv_run_record(doc, load_gv_map())
    (run_dir / path).write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class KeyAtomArmSpec:
    """Resolved arm identity: prefix A = anchor on, C = anchor off."""

    arm_id: str
    anchor: str          # "on" | "off"
    d0: float

    def to_payload(self) -> dict:
        return {"arm_id": self.arm_id, "anchor": self.anchor, "d0": self.d0}


def resolve_arm(config: Mapping, arm_id: str) -> KeyAtomArmSpec:
    """Parse the arm id: A→anchor on, C→anchor off; d0 from the config
    arm_table (plan §5.2)."""
    table = (config.get("sprint", {}) or {}).get("arm_table", {}) or {}
    _require(arm_id in table, "ARM_UNKNOWN",
             f"arm {arm_id!r} not in config arm_table {sorted(table)}")
    entry = table[arm_id] or {}
    d0 = float(entry.get("d0", config.get("d0_angstrom", 0.08)))
    raw_anchor = entry.get("anchor", "off")
    # YAML 1.1 parses bare on/off as booleans — normalize both spellings
    if isinstance(raw_anchor, bool):
        anchor = "on" if raw_anchor else "off"
    else:
        anchor = str(raw_anchor).strip().lower()
        if anchor in ("true", "yes"):
            anchor = "on"
        elif anchor in ("false", "no"):
            anchor = "off"
    _require(anchor in ("on", "off"), "ARM_ANCHOR_INVALID",
             f"arm {arm_id} anchor must be on/off, got {anchor!r}")
    prefix = arm_id[0].upper()
    _require((prefix == "A") == (anchor == "on"), "ARM_PREFIX_CONFLICT",
             f"arm {arm_id}: prefix {prefix!r} disagrees with arm_table "
             f"anchor={anchor!r}")
    return KeyAtomArmSpec(arm_id=arm_id, anchor=anchor, d0=d0)


class KeyAtomBCTrainer:
    """Config-driven V5 sprint trainer (ADR-0026 §2.8, plan §4/§5).

    One trainer drives one (arm, fold, seed, mode) run with a fixed
    update budget.  Screen mode may censored-stop on the inner_val
    reaction-cluster median RMSE/zero (patience/min_delta from config);
    formal mode never early-stops.
    """

    def __init__(
        self,
        *,
        config: Mapping,
        config_path: Path,
        arm: KeyAtomArmSpec,
        fold: int,
        seed: int,
        updates: int,
        mode: str,
        run_dir: Path,
        device_override: str | None = None,
        repo_root: Path | None = None,
    ) -> None:
        _require(config.get("schema") == EXPECTED_CONFIG_SCHEMA,
                 "CONFIG_SCHEMA",
                 f"expected schema {EXPECTED_CONFIG_SCHEMA!r}, got "
                 f"{config.get('schema')!r}")
        _require(mode in ("screen", "formal"), "MODE_UNKNOWN",
                 f"mode must be screen|formal, got {mode!r}")
        _require(int(updates) > 0, "BUDGET_INVALID", "updates must be positive")
        self.config = config
        self.config_path = Path(config_path)
        self.arm = arm
        self.fold = int(fold)
        self.seed = int(seed)
        self.updates = int(updates)
        self.mode = mode
        self.run_dir = Path(run_dir)
        self.repo_root = Path(repo_root) if repo_root is not None else Path.cwd()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.loss_config = KeyAtomLossConfig.from_campaign_config(config)
        self.eval_every = int(config.get("eval_every_updates", 250))
        self.batch_size = int(config.get("batch_size", 32))
        self.reactions_per_batch = int(
            config.get("reactions_per_batch", 8))
        self.frames_per_reaction = int(config.get("frames_per_reaction", 4))
        self.grad_clip = float(config.get("grad_clip_norm", 1.0))
        self.base_lr = float(config.get("lr", 1e-3))
        self.weight_decay = float(config.get("weight_decay", 1e-4))
        self.warmup = int(config.get("warmup_updates", 250))
        self.min_lr = float(config.get("min_lr", 1e-4))
        self.early_stop_patience = int(
            config.get("early_stop_patience_checks", 8))
        self.early_stop_min_delta = float(
            config.get("early_stop_min_delta", 0.005))
        self.formal_early_stopping = bool(
            config.get("formal_outer_early_stopping", False))
        self.split_path = str(
            (config.get("paths", {}) or {}).get(
                "split_manifest", "data/manifests/v5/split_manifest.json"))

        # device: config require_cuda honored; --device cpu is smoke-only
        from pes2ts_core.pesn.offline.training.device import resolve_device
        dev_cfg = config.get("device", {}) or {}
        if device_override is not None and device_override.startswith("cpu"):
            self.torch, self.device = resolve_device(
                device=device_override, require_cuda=False)
        else:
            self.torch, self.device = resolve_device(
                device=device_override or str(dev_cfg.get("device", "cuda:0")),
                require_cuda=bool(dev_cfg.get("require_cuda", True)))

        # paired initialization: seed BEFORE model construction
        torch.manual_seed(self.seed)
        self.model = StarEGNN().to(torch.float32).to(self.device)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params, lr=self.base_lr, weight_decay=self.weight_decay)

        self.sampler = None  # created in fit() after pools load (re-seeded)
        self.updates_run = 0
        self.eval_history: list[dict] = []
        self.bounded_scales: list[dict] = []
        self.status = "running"
        self.status_detail = ""
        self._best_metric: float | None = None
        self._bad_evals = 0

    # -- data (Lane B integration points) -----------------------------------

    def _build_pools(self):
        try:
            return load_lane_b_pools(
                self.config, fold=self.fold, d0=self.arm.d0,
                split_path=self.split_path)
        except ImportError as exc:
            raise KeyAtomBCTrainerError(
                "DATA_LANE_PENDING",
                "Lane B module pes2ts_core.pesn.offline.data."
                f"key_atom_dataset not available yet: {exc}") from exc

    def _load_samples(self, sample_rows: Sequence[dict], *, dense: bool):
        try:
            view, collate = load_lane_b_view(
                config=self.config, samples=list(sample_rows), fold=self.fold,
                d0=self.arm.d0, anchor=self.arm.anchor,
                split_path=self.split_path, device=self.device)
        except ImportError as exc:
            raise KeyAtomBCTrainerError(
                "DATA_LANE_PENDING",
                "Lane B module pes2ts_core.pesn.offline.data."
                f"key_atom_dataset not available yet: {exc}") from exc
        batches = []
        for i in range(0, len(view), self.batch_size):
            chunk = [view[j] for j in range(i, min(i + self.batch_size,
                                                    len(view)))]
            if not chunk:
                continue
            batch = collate(chunk)
            batches.append(self._to_device(batch))
        return batches

    def _to_device(self, batch: KeyAtomBatch) -> KeyAtomBatch:
        for name in ("z_atomic", "h_node_inv", "pos", "pos0", "anchor_vec",
                     "edge_index", "edge_attr", "r_ij", "r0_ij", "task19",
                     "batch", "ptr", "key_index", "key_ptr", "pair_index",
                     "pair_edit", "degK", "y_teacher", "q_weight", "near_ts",
                     "terminal", "capped", "d0", "branch"):
            val = getattr(batch, name, None)
            if torch.is_tensor(val):
                if val.dtype in (torch.float64,):
                    val = val.to(torch.float32)
                setattr(batch, name, val.to(self.device))
        return batch

    # -- run record ----------------------------------------------------------

    def _write_run_record(self, *, n_parameters: int) -> None:
        payload = {
            "schema_version": RUN_RECORD_SCHEMA,
            "loss_schema_version": SCHEMA_VERSION,
            "run_id": self.run_dir.name,
            "status": "running",
            "arm": self.arm.to_payload(),
            "fold": self.fold,
            "seed": self.seed,
            "updates_budget": self.updates,
            "mode": self.mode,
            "formal_outer_early_stopping": (
                False if self.mode == "formal" else self.formal_early_stopping),
            "n_parameters": n_parameters,
            "config_path": str(self.config_path),
            "split_manifest": self.split_path,
        }
        gv = build_gv_block(
            config_path=self.config_path, repo_root=self.repo_root)
        write_run_record(self.run_dir, payload=payload, gv=gv)
        self._run_record_payload = payload
        self._run_record_gv = gv

    def _finalize_run_record(self) -> None:
        payload = {**self._run_record_payload, "status": self.status,
                   "updates_run": self.updates_run}
        write_run_record(self.run_dir, payload=payload, gv=self._run_record_gv)

    # -- checkpointing -------------------------------------------------------

    def _save_checkpoint(self, name: str) -> None:
        doc = {
            "schema_version": CHECKPOINT_SCHEMA,
            "update": self.updates_run,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "arm": self.arm.to_payload(),
            "fold": self.fold,
            "seed": self.seed,
        }
        torch.save(doc, self.run_dir / "checkpoints" / name)

    # -- eval ----------------------------------------------------------------

    def evaluate(self, dev_batches: Sequence[KeyAtomBatch]) -> dict:
        """Dense dev-fold eval: raw + bounded per-cluster metrics."""
        self.model.train(False)
        per_sample: list[dict] = []
        with torch.no_grad():
            for batch in dev_batches:
                u, _z = self.model(batch)
                y = batch.y_teacher
                if y is None:
                    continue
                u_b, scale = bounded_projection(
                    u.float(), batch.d0 if batch.d0 is not None else A_REF,
                    batch.key_ptr)
                u_cls, cls_info = classic_pair_driver(batch)
                key_ptr = batch.key_ptr.cpu().numpy()
                near = (batch.near_ts.cpu().numpy()
                        if batch.near_ts is not None
                        else np.ones(len(key_ptr) - 1))
                groups = (batch.reaction_ids if batch.reaction_ids
                          else tuple("unknown" for _ in range(len(key_ptr) - 1)))
                u_np = u.float().cpu().numpy()
                y_np = y.cpu().numpy()
                ub_np = u_b.cpu().numpy()
                uc_np = u_cls.cpu().numpy()
                for b in range(len(key_ptr) - 1):
                    lo, hi = int(key_ptr[b]), int(key_ptr[b + 1])
                    if hi <= lo:
                        continue
                    degenerate = bool(cls_info["degenerate_frames"][b])
                    per_sample.append({
                        "group_id": str(groups[b]),
                        "u": u_np[lo:hi], "y": y_np[lo:hi],
                        "u_classic": None if degenerate else uc_np[lo:hi],
                        "near_ts": float(near[b]),
                        "u_bounded": ub_np[lo:hi],
                        "bounded_scale": float(scale[b]),
                    })
        self.model.train()
        cfg = self.loss_config
        per_cluster_raw, agg_raw = compute_cluster_metrics(
            per_sample, soft=cfg.cos_soft_norm_angstrom,
            tau=cfg.tau_norm_angstrom)
        per_cluster_b, agg_b = compute_cluster_metrics(
            [{**s, "u": s["u_bounded"]} for s in per_sample],
            soft=cfg.cos_soft_norm_angstrom, tau=cfg.tau_norm_angstrom)
        return {
            "update": self.updates_run,
            "wall_s": time.time() - self._t0,
            "raw": {"per_cluster": per_cluster_raw, "aggregate": agg_raw},
            "bounded": {"per_cluster": per_cluster_b, "aggregate": agg_b},
        }

    def _record_eval(self, record: dict) -> None:
        self.eval_history.append(record)
        with (self.run_dir / "eval.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "update": record["update"],
                "wall_s": record["wall_s"],
                "per_cluster": record["raw"]["per_cluster"],
                "aggregate": record["raw"]["aggregate"],
                "per_cluster_bounded": record["bounded"]["per_cluster"],
                "aggregate_bounded": record["bounded"]["aggregate"],
            }, ensure_ascii=False) + "\n")

    def _early_stop_check(self, record: dict) -> bool:
        """Screen-mode only: inner_val reaction-cluster median RMSE/zero."""
        if self.mode != "screen":
            return False
        metric = record["raw"]["aggregate"].get("rmse_over_zero")
        if metric is None or not math.isfinite(metric):
            return False
        if self.updates_run < 500:  # no rank-based stop before update 500
            return False
        if self._best_metric is None:
            self._best_metric = metric
            return False
        if self._best_metric - metric > self.early_stop_min_delta:
            self._best_metric = metric
            self._bad_evals = 0
            return False
        self._bad_evals += 1
        return self._bad_evals >= self.early_stop_patience

    # -- main loop -----------------------------------------------------------

    def fit(self) -> str:
        """Run the fixed budget; return the final status string."""
        self._t0 = time.time()
        self._write_run_record(n_parameters=self.model.n_parameters)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        try:
            pools = self._build_pools()
            train_pools = pools["train"] if "train" in pools else pools
            dev_samples = (pools.get("dev_dense")
                           if "train" in pools else None)
            # paired sample order: seed BEFORE sampler creation
            torch.manual_seed(self.seed)
            self.sampler = KeyAtomCycleSampler(
                train_pools, seed=self.seed,
                reactions_per_batch=self.reactions_per_batch,
                frames_per_reaction=self.frames_per_reaction)
            dev_batches = self._load_samples(
                dev_samples if dev_samples is not None else [], dense=True)
        except KeyAtomBCTrainerError as exc:
            self._abort(exc.code, exc.detail,
                        STATUS_ABORTED_DATA
                        if exc.code.startswith(("DATA", "EMPTY", "POOL"))
                        else STATUS_ABORTED_NAN)
            return self.status

        self.model.train()
        try:
            for update in range(1, self.updates + 1):
                self.updates_run = update
                lr = lr_at_update(
                    update, base_lr=self.base_lr, warmup=self.warmup,
                    total=self.updates, min_lr=self.min_lr)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
                rows = self.sampler.draw_batch()
                batches = self._load_samples(rows, dense=False)
                lam_cos = cos_lambda(update, self.loss_config)
                for batch in batches:
                    u, _z = self.model(batch)
                    if batch.y_teacher is None:
                        continue
                    out = key_atom_bc_loss(
                        u.float(), batch.y_teacher.float(),
                        batch.degK.float(), batch.key_ptr, batch.pos,
                        batch.key_index, batch.pair_index,
                        config=self.loss_config, lambda_cos=lam_cos)
                    loss = out["loss"]
                    if not bool(torch.isfinite(loss)):
                        raise KeyAtomBCTrainerError(
                            "ABORT_NAN", f"non-finite loss at update {update}")
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                if update % self.eval_every == 0 or update == self.updates:
                    record = self.evaluate(dev_batches)
                    self._record_eval(record)
                    self._save_checkpoint("checkpoint_last.pt")
                    if self._early_stop_check(record):
                        self.status = STATUS_CENSORED_SCREEN
                        self.status_detail = (
                            "screen early stop: inner_val reaction-cluster "
                            "median RMSE/zero stalled (patience "
                            f"{self.early_stop_patience}, min_delta "
                            f"{self.early_stop_min_delta})")
                        break
        except KeyAtomBCTrainerError as exc:
            self._abort(exc.code, exc.detail, STATUS_ABORTED_NAN)
            return self.status
        except KeyAtomLossError as exc:
            self._abort(exc.code, exc.detail, STATUS_ABORTED_NAN)
            return self.status

        if self.status == "running":
            self.status = STATUS_COMPLETE
        self._save_checkpoint("checkpoint_final.pt")
        self._write_metrics()
        (self.run_dir / "status.json").write_text(json.dumps({
            "status": self.status, "detail": self.status_detail,
            "updates_run": self.updates_run, "mode": self.mode,
        }, indent=2) + "\n", encoding="utf-8")
        self._finalize_run_record()
        return self.status

    def _abort(self, code: str, detail: str, status: str) -> None:
        self.status = status
        self.status_detail = f"{code}: {detail}"
        (self.run_dir / "status.json").write_text(json.dumps({
            "status": status, "detail": self.status_detail,
            "updates_run": self.updates_run, "mode": self.mode,
        }, indent=2) + "\n", encoding="utf-8")
        try:
            self._finalize_run_record()
        except Exception:  # record refresh is best-effort on abort
            pass

    def _write_metrics(self) -> None:
        doc = {
            "schema_version": SCHEMA_VERSION,
            "arm": self.arm.to_payload(),
            "fold": self.fold, "seed": self.seed, "mode": self.mode,
            "updates_run": self.updates_run,
            "raw": [ {"update": r["update"], "aggregate": r["raw"]["aggregate"],
                       "per_cluster": r["raw"]["per_cluster"]}
                      for r in self.eval_history],
            "bounded": [
                {"update": r["update"], "aggregate": r["bounded"]["aggregate"],
                 "per_cluster": r["bounded"]["per_cluster"]}
                for r in self.eval_history],
        }
        (self.run_dir / "metrics.json").write_text(
            json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pes2ts_core.pesn.offline.training.key_atom_bc",
        description="V5 sprint key-atom BC single-run trainer (ADR-0026)")
    p.add_argument("--config", default="config/campaigns/v5_sprint.yaml")
    p.add_argument("--arm", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--updates", type=int, default=None)
    p.add_argument("--mode", choices=("screen", "formal"), required=True)
    p.add_argument("--out", default=None,
                   help="run dir (default outputs/pesn/v5/sprint/runs/"
                        "<arm>_f<fold>_s<seed>_<mode>)")
    p.add_argument("--device", default=None,
                   help="device override (cpu allowed for smoke only)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config_path = Path(args.config)
    config = load_config_mapping(config_path)
    arm = resolve_arm(config, args.arm)
    default_updates = (1500 if args.mode == "screen" else 3000)
    updates = int(args.updates) if args.updates is not None else default_updates
    run_id = f"{args.arm}_f{args.fold}_s{args.seed}_{args.mode}"
    run_dir = (Path(args.out) if args.out
               else Path("outputs/pesn/v5/sprint/runs") / run_id)
    trainer = KeyAtomBCTrainer(
        config=config, config_path=config_path, arm=arm, fold=args.fold,
        seed=args.seed, updates=updates, mode=args.mode, run_dir=run_dir,
        device_override=args.device)
    status = trainer.fit()
    print(json.dumps({"run_id": run_id, "status": status,
                      "run_dir": str(run_dir)}, ensure_ascii=False))
    return 0 if status in (STATUS_COMPLETE, STATUS_CENSORED_SCREEN) else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
