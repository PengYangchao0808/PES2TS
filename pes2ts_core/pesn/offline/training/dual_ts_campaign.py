"""V4.2–V4.7 campaign trainer + single-arm runner (Lane C; contract §6/§8).

Replaces the per-recipe ``DualTSBCTrainer`` flow for the new campaign: one
:class:`CampaignTrainer` drives every arm through a model-agnostic core,
reusing the frozen §I.10 sampler/reduction from ``dual_ts_sampling`` /
``dual_ts_bc`` and consuming unified npz labels (native ``action_angstrom`` /
``is_terminal`` / ``remaining_arc_length`` in the feature shard, or the same
arrays from ``wp42_campaign/legacy_labels/*.npz`` for the legacy kind).

Two heads:

* ``bc_nll_legacy`` — the historical radial-tanh NLL head (``DualTSPolicy``,
  recipe ``bc_nll``) with the ``legacy_v41`` preset, reconstructed in-place to
  reproduce the old ``DualTSBCSettings`` numeric path (a drift test asserts the
  50-step loss sequence matches within tolerance).
* ``displacement`` — the unbounded displacement head built by Lane B's
  ``build_displacement_policy``; loss ``L_xyz`` (atom-mean → state → branch →
  group) plus optional ``L_dir`` / ``L_mag`` (H2), and optional frozen
  ``a_phys`` skip (H1).

The trainer owns: full resume (model/optimizer/scheduler/EMA/sampler RNG/
exposure/update count), per-update independent checkpoints
``checkpoints/u{5000,10000,20000,…}`` with BOTH raw and EMA identities,
fit5-dev evaluation, extended exposure (per-reaction/branch draw quantiles +
near-zero/long-step/near-TS buckets), typed stop on non-finite loss/gradient,
and a ``pesn_dual_ts_bc_run_v3`` record (all v2 keys + §6 additions).
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Iterable, Mapping, Sequence

import numpy as np
import torch

from pes2ts_core.pesn.models import require_torch
from pes2ts_core.pesn.offline.training.dual_ts_bc import (
    NEAR_TS_ARC_FRACTION,
    build_targets,
    epoch_state_order,
    per_atom_nll,
    reduce_step_sums,
)
from pes2ts_core.pesn.offline.training.dual_ts_optim import (
    EMA_DECAY,
    EMATracker,
    build_optim,
    campaign_warmup_steps,
)
from pes2ts_core.pesn.offline.training.dual_ts_sampling import (
    DualTSGroupBranchSampler,
    ExposureTracker,
    build_sampler_groups,
)
from pes2ts_core.utils.jsonio import write_json

#: Version identities (contract §1 / §6).
CAMPAIGN_TRAINER_VERSION: Final[str] = "pesn_dual_ts_campaign_trainer_v1.0"
RUN_RECORD_V3_SCHEMA: Final[str] = "pesn_dual_ts_bc_run_v3"
CHECKPOINT_SCHEMA: Final[str] = "pesn_dual_ts_campaign_checkpoint_v1"
EMA_CHECKPOINT_SCHEMA: Final[str] = "pesn_dual_ts_campaign_ema_v1"

HEADS: Final[tuple[str, ...]] = ("bc_nll_legacy", "displacement")
SAMPLERS: Final[tuple[str, ...]] = ("group_balanced_v41", "epoch_traversal_v41")
DATASET_KINDS: Final[tuple[str, ...]] = ("legacy", "native")
FEATURE_VERSIONS: Final[tuple[str, ...]] = ("v3", "v4")
ENCODERS: Final[tuple[str, ...]] = ("egnn", "painn")
EMA_READS: Final[tuple[str, ...]] = ("raw", "ema", "dev_select")

#: Frozen action-max used ONLY by the legacy radial head.
DEFAULT_ACTION_MAX: Final[float] = 0.1

#: Near-zero action threshold (Å) shared with the frozen mask rule.
NEAR_ZERO_ACTION_THRESHOLD: Final[float] = 0.01

#: Model-input keys produced by ``collate_dual_states`` (the head forward
#: receives exactly these, never the supervision targets).
MODEL_INPUT_KEYS: Final[tuple[str, ...]] = (
    "z", "node_scalars", "x", "g",
    "atom_feats_a", "atom_feats_b", "edge_feats_a", "edge_feats_b",
    "edge_index", "node_counts", "n_graphs",
)


class CampaignTrainerError(RuntimeError):
    """Typed campaign trainer refusal (carries a machine-readable code)."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise CampaignTrainerError(code, detail)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CampaignDatasetSpec:
    """Data identity for one arm (contract §8 arm tuples)."""

    kind: str = "legacy"          # legacy | native
    features_dir: str = ""
    labels_dir: str = ""          # legacy labels (unused for native)
    split_path: str = ""

    def __post_init__(self) -> None:
        if self.kind not in DATASET_KINDS:
            raise CampaignTrainerError(
                "DATASET_KIND_UNKNOWN", f"kind {self.kind!r} not in "
                f"{DATASET_KINDS}")

    def to_payload(self) -> dict:
        return {
            "kind": self.kind,
            "features_dir": self.features_dir,
            "labels_dir": self.labels_dir,
            "split_path": self.split_path,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "CampaignDatasetSpec":
        return cls(
            kind=str(payload.get("kind", "legacy")),
            features_dir=str(payload.get("features_dir", "")),
            labels_dir=str(payload.get("labels_dir", "")),
            split_path=str(payload.get("split_path", "")),
        )


@dataclass(frozen=True, slots=True)
class CampaignArmConfig:
    """One pre-registered campaign arm (contract §8 + §5/§6 hyper-params)."""

    arm_id: str = ""
    dataset: CampaignDatasetSpec = field(default_factory=CampaignDatasetSpec)
    head: str = "displacement"          # bc_nll_legacy | displacement
    use_prior_skip: bool = False        # H1: a_hat = a_phys + f
    lambda_dir: float | None = None     # H2 (None = off)
    lambda_mag: float | None = None     # H2 (None = off)
    features_version: str = "v3"        # v3 | v4
    encoder: str = "egnn"               # egnn | painn
    vector_channels: int = 16
    contact_edges: bool = False         # V2
    hidden: int = 64
    n_layers: int = 3
    sampler: str = "group_balanced_v41"
    preset: str = "campaign_v1"
    seed: int = 20261017
    updates: int = 20000
    batch_states: int = 64
    ema_read: str = "raw"               # raw | ema | dev_select
    eval_points: tuple[int, ...] = (5000, 10000, 20000)
    keypair_condition: bool = False     # E2
    clip: float = 1.0
    curve_every: int = 100
    device: str = "cuda:0"
    require_cuda: bool = True
    light: bool = True                  # legacy head only (old default)
    final_fit50: bool = False           # V4.7 FINAL: train on full fit50

    def __post_init__(self) -> None:
        if self.head not in HEADS:
            raise CampaignTrainerError(
                "HEAD_UNKNOWN", f"head {self.head!r} not in {HEADS}")
        if self.sampler not in SAMPLERS:
            raise CampaignTrainerError(
                "SAMPLER_UNKNOWN", f"sampler {self.sampler!r} not in "
                f"{SAMPLERS}")
        if self.features_version not in FEATURE_VERSIONS:
            raise CampaignTrainerError(
                "FEATURES_VERSION_UNKNOWN",
                f"features_version {self.features_version!r} not in "
                f"{FEATURE_VERSIONS}")
        if self.encoder not in ENCODERS:
            raise CampaignTrainerError(
                "ENCODER_UNKNOWN", f"encoder {self.encoder!r} not in "
                f"{ENCODERS}")
        if self.ema_read not in EMA_READS:
            raise CampaignTrainerError(
                "EMA_READ_UNKNOWN", f"ema_read {self.ema_read!r} not in "
                f"{EMA_READS}")
        if self.updates <= 0 or self.batch_states <= 0:
            raise CampaignTrainerError("BUDGET_INVALID", "non-positive budget")
        for lam, name in ((self.lambda_dir, "lambda_dir"),
                          (self.lambda_mag, "lambda_mag")):
            if lam is not None and (not np.isfinite(float(lam))
                                    or float(lam) < 0.0):
                raise CampaignTrainerError(
                    "LAMBDA_INVALID", f"{name} must be finite and >= 0")
        if self.head == "displacement" and self.preset == "legacy_v41":
            raise CampaignTrainerError(
                "PRESET_HEAD_CONFLICT",
                "displacement head requires preset campaign_v1")
        for point in self.eval_points:
            if int(point) <= 0 or int(point) > self.updates:
                raise CampaignTrainerError(
                    "EVAL_POINT_INVALID",
                    f"eval point {point} outside (0, {self.updates}]")

    @property
    def is_legacy(self) -> bool:
        return self.head == "bc_nll_legacy"

    def to_payload(self) -> dict:
        return {
            "arm_id": self.arm_id,
            "dataset": self.dataset.to_payload(),
            "head": self.head,
            "use_prior_skip": self.use_prior_skip,
            "lambda_dir": self.lambda_dir,
            "lambda_mag": self.lambda_mag,
            "features_version": self.features_version,
            "encoder": self.encoder,
            "vector_channels": self.vector_channels,
            "contact_edges": self.contact_edges,
            "hidden": self.hidden,
            "n_layers": self.n_layers,
            "sampler": self.sampler,
            "preset": self.preset,
            "seed": self.seed,
            "updates": self.updates,
            "batch_states": self.batch_states,
            "ema_read": self.ema_read,
            "eval_points": list(self.eval_points),
            "keypair_condition": self.keypair_condition,
            "clip": self.clip,
            "curve_every": self.curve_every,
            "device": self.device,
            "require_cuda": self.require_cuda,
            "light": self.light,
            "final_fit50": self.final_fit50,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "CampaignArmConfig":
        return cls(
            arm_id=str(payload.get("arm_id", "")),
            dataset=CampaignDatasetSpec.from_payload(payload.get("dataset", {})),
            head=str(payload.get("head", "displacement")),
            use_prior_skip=bool(payload.get("use_prior_skip", False)),
            lambda_dir=payload.get("lambda_dir"),
            lambda_mag=payload.get("lambda_mag"),
            features_version=str(payload.get("features_version", "v3")),
            encoder=str(payload.get("encoder", "egnn")),
            vector_channels=int(payload.get("vector_channels", 16)),
            contact_edges=bool(payload.get("contact_edges", False)),
            hidden=int(payload.get("hidden", 64)),
            n_layers=int(payload.get("n_layers", 3)),
            sampler=str(payload.get("sampler", "group_balanced_v41")),
            preset=str(payload.get("preset", "campaign_v1")),
            seed=int(payload.get("seed", 20261017)),
            updates=int(payload.get("updates", 20000)),
            batch_states=int(payload.get("batch_states", 64)),
            ema_read=str(payload.get("ema_read", "raw")),
            eval_points=tuple(int(p) for p in payload.get("eval_points", (5000, 10000, 20000))),
            keypair_condition=bool(payload.get("keypair_condition", False)),
            clip=float(payload.get("clip", 1.0)),
            curve_every=int(payload.get("curve_every", 100)),
            device=str(payload.get("device", "cuda:0")),
            require_cuda=bool(payload.get("require_cuda", True)),
            light=bool(payload.get("light", True)),
            final_fit50=bool(payload.get("final_fit50", False)),
        )


# ---------------------------------------------------------------------------
# Data loading (unified npz labels)
# ---------------------------------------------------------------------------

def _scalar(v):
    if hasattr(v, "item") and not isinstance(v, str):
        return v.item()
    return v


def load_campaign_states(
    dataset: CampaignDatasetSpec,
    *,
    group_ids: Sequence[str],
) -> tuple[list[dict], dict]:
    """Load per-state records for the given reaction groups (unified npz).

    Feature shards are read from ``features_dir`` (v3 or v4 layout, consumed
    as-is).  Labels come from the same npz for ``native`` (``action_angstrom``,
    ``is_terminal``, ``remaining_arc_length``) or from
    ``labels_dir/<reaction_id>.npz`` for ``legacy``.  Produces one dict per
    state in the ``collate_dual_states``-compatible layout.
    """
    features_dir = Path(dataset.features_dir)
    labels_dir = Path(dataset.labels_dir)
    manifest = json.loads(
        (features_dir / "features_manifest.json").read_text())
    rows = sorted(manifest["per_reaction"],
                  key=lambda r: str(r["reaction_group_id"]))
    keep = {str(g) for g in group_ids}
    rows = [r for r in rows if str(r["reaction_group_id"]) in keep]

    states: list[dict] = []
    n_actions = 0
    max_action_norm = 0.0
    for row in rows:
        path = features_dir / row["npz_file"]
        shard = np.load(path)
        rid = str(_scalar(shard["reaction_id"]))
        z = np.asarray(shard["atomic_numbers"], dtype=np.int64)
        node_scalars = shard["node_scalars"]
        x = shard["x_angstrom"]
        g = shard["g_scaled"]
        edge_index = shard["edge_index"]
        atom_a, atom_b = shard["atom_feats_a"], shard["atom_feats_b"]
        edge_a, edge_b = shard["edge_feats_a"], shard["edge_feats_b"]
        # E2 key-pair whitelist needs the undirected union pairs (i < j).
        edge_pairs = shard["edge_pairs"] if "edge_pairs" in shard else None
        if dataset.kind == "legacy":
            lbl = np.load(labels_dir / f"legacy_labels_{rid}.npz")
        else:
            lbl = shard
        # v3 stores the edge block shared across states (E, 4); v4/E1 stores
        # per-state gradient-augmented blocks (n_states, E, 7).
        edge_a_per_state = edge_a.ndim == 3
        edge_b_per_state = edge_b.ndim == 3
        action = lbl["action_angstrom"]
        is_terminal = lbl["is_terminal"]
        ral = lbl["remaining_arc_length"]
        for r, sid in enumerate(shard["state_id"]):
            terminal = bool(np.asarray(is_terminal[r]).item())
            act = (None if terminal
                   else np.asarray(action[r], dtype=np.float64))
            near_zero = (np.zeros(int(z.size), dtype=bool) if act is None
                         else np.linalg.norm(act, axis=1)
                         < NEAR_ZERO_ACTION_THRESHOLD)
            if act is not None:
                n_actions += 1
                norms = np.linalg.norm(act, axis=1)
                max_action_norm = max(max_action_norm, float(norms.max()))
            states.append({
                "state_id": str(_scalar(sid)),
                "reaction_group_id": str(_scalar(shard["reaction_group_id"])),
                "branch_id": str(_scalar(shard["branch_id"][r])),
                "episode_id": str(_scalar(shard["episode_id"][r])),
                "step_index": int(shard["step_index"][r]),
                "is_teacher": terminal,
                "remaining_arc_length": float(np.asarray(ral[r]).item()),
                "action": act,
                "near_zero_mask": near_zero,
                "atomic_numbers": z,
                "node_scalars": node_scalars[r],
                "x_angstrom": x[r],
                "g_scaled": g[r],
                "atom_feats_a": atom_a,
                "atom_feats_b": atom_b,
                "edge_feats_a": edge_a[r] if edge_a_per_state else edge_a,
                "edge_feats_b": edge_b[r] if edge_b_per_state else edge_b,
                "edge_index": edge_index,
                "edge_pairs": edge_pairs,
            })
    meta = {
        "features_dir": str(features_dir),
        "labels_dir": str(labels_dir),
        "dataset_kind": dataset.kind,
        "group_ids": sorted(keep),
        "n_groups": len(keep),
        "n_states": len(states),
        "n_action_states": n_actions,
        "max_action_norm_angstrom": max_action_norm,
    }
    return states, meta


# ---------------------------------------------------------------------------
# Displacement targets + loss building blocks
# ---------------------------------------------------------------------------

def build_displacement_targets(states: Sequence[dict], torch) -> dict:
    """Alignment arrays for the displacement loss (no latent transform).

    Mirrors ``build_targets`` but never applies the radial-tanh inverse, so
    unbounded teacher actions are legal.
    """
    n_atoms = [int(np.asarray(s["atomic_numbers"]).size) for s in states]
    valid, near_zero = [], []
    actions = []
    for s, n in zip(states, n_atoms):
        valid.extend([not s["is_teacher"]] * n)
        a = (np.zeros((n, 3)) if s.get("action") is None
             else np.asarray(s["action"], dtype=np.float64))
        actions.append(a)
        near_zero.extend(np.linalg.norm(a, axis=1) < NEAR_ZERO_ACTION_THRESHOLD)
    a_star = np.concatenate(actions, axis=0)
    group_ids = [s["reaction_group_id"] for s in states]
    branch_ids = [s["branch_id"] for s in states]
    state_of_atom = np.repeat(np.arange(len(states), dtype=np.int64), n_atoms)
    cells = sorted(set(zip(group_ids, branch_ids)))
    cell_index = {c: i for i, c in enumerate(cells)}
    groups = sorted(set(group_ids))
    group_index = {g: i for i, g in enumerate(groups)}
    cell_group = np.asarray([group_index[g] for g, _ in cells], dtype=np.int64)
    return {
        "a_star": torch.as_tensor(a_star, dtype=torch.float32),
        "atom_valid": torch.as_tensor(np.asarray(valid, dtype=bool)),
        "dir_mask": torch.as_tensor(np.asarray(
            [v and not nz for v, nz in zip(valid, near_zero)], dtype=bool)),
        "cell_index": torch.as_tensor(np.asarray(
            [cell_index[(g, b)] for g, b in zip(group_ids, branch_ids)],
            dtype=np.int64)),
        "state_of_atom": torch.as_tensor(state_of_atom),
        "n_states": len(states),
        "n_cells": len(cells),
        "cell_group": torch.as_tensor(cell_group),
        "n_groups": len(groups),
        "group_ids": groups,
    }


def compute_a_s(states: Sequence[dict]) -> float:
    """Frozen positive length scale ``a_s`` (median non-zero atom |a*|).

    Estimated from fit45_train non-zero atom action norms (contract §5)."""
    norms: list[float] = []
    for s in states:
        if s.get("action") is None:
            continue
        a = np.asarray(s["action"], dtype=np.float64)
        n = np.linalg.norm(a, axis=1)
        norms.extend(n[n > 0.0].tolist())
    if not norms:
        raise CampaignTrainerError(
            "A_S_EMPTY", "no non-zero teacher atom actions to estimate a_s")
    return float(np.median(np.asarray(norms, dtype=float)))


def per_atom_cos_dir_loss(a_hat, a_star, eps: float = 1e-12):
    """``1 − cos(a_hat, a_star)`` per atom (masked upstream; contract §5)."""
    na = a_hat.norm(dim=-1)
    w = a_star.norm(dim=-1)
    cos = (a_hat * a_star).sum(dim=-1) / (na * w).clamp(min=eps)
    return 1.0 - cos.clamp(min=-1.0, max=1.0)


def per_atom_mag_loss(a_hat, a_star, *, delta: float = 1.0, eps: float = 1e-8):
    """Huber(δ=1) of the log magnitude ratio per atom (contract §5)."""
    ratio = torch.log((a_hat.norm(dim=-1) + eps)
                      / (a_star.norm(dim=-1) + eps))
    abs_r = ratio.abs()
    quad = 0.5 * ratio ** 2
    linear = delta * (abs_r - 0.5 * delta)
    return torch.where(abs_r <= delta, quad, linear)


def compute_physical_prior_action(
    state: dict,
    *,
    s_g: float,
    eps_g: float,
    tau_g: float,
    delta_s_fit: float,
    masses_for,
) -> np.ndarray:
    """Frozen ``a_phys`` for one state (contract §4; no gradient).

    ``b_i = (g_i/m_i) / sqrt(Σ_j ‖g_j‖²/m_j + eps_g²)``, ``a_phys = Δs_fit·b``.
    Below ``tau_g`` (per-state ``sqrt(Σ_j ‖g_j‖²/m_j)``) the action is all
    zero.  Raw gradient is reconstructed as ``g_scaled * s_g``.
    """
    g = np.asarray(state["g_scaled"], dtype=np.float64) * float(s_g)
    z = np.asarray(state["atomic_numbers"], dtype=np.int64)
    m = np.asarray(masses_for(z), dtype=np.float64)
    q2 = (g * g).sum(axis=-1) / m
    q = float(np.sqrt(q2.sum()))
    if q < float(tau_g):
        return np.zeros_like(g)
    denom = float(np.sqrt(q2.sum() + float(eps_g) ** 2))
    b = g / m[:, None] / denom
    return float(delta_s_fit) * b


# ---------------------------------------------------------------------------
# Epoch-traversal sampler (epoch_state_order wrapped to draw_batch)
# ---------------------------------------------------------------------------

class EpochTraversalSampler:
    """``epoch_state_order`` semantics behind the ``draw_batch`` interface.

    Returns row INDICES into the drawable list (parity with
    ``DualTSGroupBranchSampler``); one epoch = one full traversal (every
    drawable state drawn exactly once).  Fully restorable via
    ``state_dict``/``load_state_dict``.
    """

    def __init__(self, *, seed: int, states: Sequence[dict],
                 batch_states: int) -> None:
        self.seed = int(seed)
        self.batch_states = int(batch_states)
        self.drawable = [s for s in states if not s["is_teacher"]]
        if not self.drawable:
            raise CampaignTrainerError(
                "EMPTY_ACTION_POPULATION", "no non-teacher states to draw")
        self._by_id = {s["state_id"]: i for i, s in enumerate(self.drawable)}
        self.draws = 0
        self.rounds_completed = 0
        self._epoch = 0
        self._order_epoch = -1
        self._order: list[int] = []
        self._cursor = 0
        self._refill()

    @property
    def sampling_rule(self) -> str:
        return "epoch_traversal_v41"

    @property
    def within_branch_rule(self) -> str:
        return "arc_stratified_seeded_permutation_v0"

    @property
    def n_groups(self) -> int:
        return len({s["reaction_group_id"] for s in self.drawable})

    def _refill(self) -> None:
        order, _ = epoch_state_order(
            self.drawable, seed=self.seed, epoch=self._epoch)
        self._order = [self._by_id[s["state_id"]] for s in order]
        self._order_epoch = self._epoch
        self._epoch += 1
        self._cursor = 0

    def draw_batch(self, n: int) -> list[int]:
        out: list[int] = []
        for _ in range(int(n)):
            if self._cursor >= len(self._order):
                self._refill()
                self.rounds_completed += 1
            out.append(self._order[self._cursor])
            self._cursor += 1
        self.draws += len(out)
        return out

    def state_dict(self) -> dict:
        return {
            "seed": self.seed,
            "draws": int(self.draws),
            "rounds_completed": int(self.rounds_completed),
            "order_epoch": int(self._order_epoch),
            "cursor": int(self._cursor),
        }

    def load_state_dict(self, state: Mapping) -> None:
        if int(state["seed"]) != self.seed:
            raise ValueError("sampler seed mismatch on restore")
        self.draws = int(state["draws"])
        self.rounds_completed = int(state["rounds_completed"])
        self._epoch = int(state["order_epoch"])
        cursor = int(state["cursor"])
        self._refill()  # regenerates the order and resets _cursor to 0
        self._cursor = cursor  # restore the cursor position after refill


# ---------------------------------------------------------------------------
# Extended exposure accounting
# ---------------------------------------------------------------------------

class CampaignExposureTracker(ExposureTracker):
    """ExposureTracker + per-cell draw quantiles and bucket counters.

    ``row_meta`` entries additionally carry ``max_action_norm`` (per-state
    max atom action norm) so near-zero / long-step buckets can be counted;
    ``long_step_p90_threshold`` is the fit45 P90 of per-state max norms.
    """

    def __init__(self, *, row_meta: Sequence[Mapping],
                 long_step_p90_threshold: float | None = None) -> None:
        super().__init__(row_meta=row_meta)
        self.long_step_p90_threshold = long_step_p90_threshold
        self._cell_state_counts: dict[tuple, dict] = {}
        self._near_zero_state: set = set()
        self._long_step_state: set = set()
        for r in self.row_meta:
            key = (str(r["group_id"]), str(r["branch_id"]))
            self._cell_state_counts.setdefault(key, {})[str(r["state_id"])] = 0
            sid = str(r["state_id"])
            norm = r.get("max_action_norm")
            if norm is not None and float(norm) < NEAR_ZERO_ACTION_THRESHOLD:
                self._near_zero_state.add(sid)
            if (long_step_p90_threshold is not None and norm is not None
                    and float(norm) >= long_step_p90_threshold):
                self._long_step_state.add(sid)
        self._near_zero_draws = 0
        self._long_step_draws = 0

    def record_batch(self, row_indices: Sequence[int]) -> None:
        super().record_batch(row_indices)
        for i in row_indices:
            meta = self.row_meta[int(i)]
            key = (str(meta["group_id"]), str(meta["branch_id"]))
            sid = str(meta["state_id"])
            self._cell_state_counts[key][sid] += 1
            if sid in self._near_zero_state:
                self._near_zero_draws += 1
            if sid in self._long_step_state:
                self._long_step_draws += 1

    # -- resume state --------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "total_draws": int(self.total_draws),
            "distinct_all": sorted(self.distinct_all),
            "near_ts_total": int(self.near_ts_total),
            "interval_draws": int(self.interval_draws),
            "interval_distinct_all": sorted(self.interval_distinct_all),
            "interval_near_ts": int(self.interval_near_ts),
            "per_group": {
                g: {
                    "draws": pg["draws"],
                    "distinct": sorted(pg["distinct"]),
                    "near_ts": pg["near_ts"],
                    "interval_draws": pg["interval_draws"],
                    "interval_distinct": sorted(pg["interval_distinct"]),
                    "interval_near_ts": pg["interval_near_ts"],
                }
                for g, pg in self.per_group.items()
            },
            "history": list(self.history),
            "cell_state_counts": {
                f"{g}|{b}": dict(counts)
                for (g, b), counts in self._cell_state_counts.items()
            },
            "near_zero_draws": int(self._near_zero_draws),
            "long_step_draws": int(self._long_step_draws),
        }

    def load_state_dict(self, state: Mapping) -> None:
        self.total_draws = int(state["total_draws"])
        self.distinct_all = set(state["distinct_all"])
        self.near_ts_total = int(state["near_ts_total"])
        self.interval_draws = int(state["interval_draws"])
        self.interval_distinct_all = set(state["interval_distinct_all"])
        self.interval_near_ts = int(state["interval_near_ts"])
        for g, pg in state["per_group"].items():
            self.per_group[g]["draws"] = int(pg["draws"])
            self.per_group[g]["distinct"] = set(pg["distinct"])
            self.per_group[g]["near_ts"] = int(pg["near_ts"])
            self.per_group[g]["interval_draws"] = int(pg["interval_draws"])
            self.per_group[g]["interval_distinct"] = set(pg["interval_distinct"])
            self.per_group[g]["interval_near_ts"] = int(pg["interval_near_ts"])
        self.history = list(state["history"])
        for key, counts in state["cell_state_counts"].items():
            g, b = key.split("|", 1)
            self._cell_state_counts[(g, b)] = dict(counts)
        self._near_zero_draws = int(state["near_zero_draws"])
        self._long_step_draws = int(state["long_step_draws"])

    def quantile_payload(self) -> dict:
        cells = {}
        for (g, b), counts in sorted(self._cell_state_counts.items()):
            arr = np.asarray(list(counts.values()), dtype=float)
            cells[f"{g}|{b}"] = {
                "group_id": g, "branch_id": b,
                "n_states": int(arr.size),
                "draws_p10": float(np.percentile(arr, 10)) if arr.size else 0.0,
                "draws_p50": float(np.percentile(arr, 50)) if arr.size else 0.0,
                "draws_p90": float(np.percentile(arr, 90)) if arr.size else 0.0,
                "draws_max": float(arr.max()) if arr.size else 0.0,
                "draws_total": float(arr.sum()) if arr.size else 0.0,
            }
        return {
            "cells": cells,
            "near_zero_draws": int(self._near_zero_draws),
            "long_step_draws": int(self._long_step_draws),
            "long_step_p90_threshold": self.long_step_p90_threshold,
        }

    def payload(self) -> dict:
        base = super().payload()
        base["extended"] = self.quantile_payload()
        return base


# ---------------------------------------------------------------------------
# Model assembly (legacy head self-contained; displacement head via Lane B)
# ---------------------------------------------------------------------------

def _model_state_dict(model) -> dict:
    if hasattr(model, "state_dicts"):
        return model.state_dicts()
    if hasattr(model, "state_dict"):
        return model.state_dict()
    raise CampaignTrainerError(
        "MODEL_STATE_UNSUPPORTED", "model exposes no state dict")


def _model_load_state_dict(model, state: dict) -> None:
    if hasattr(model, "load_state_dicts"):
        model.load_state_dicts(state)
    elif hasattr(model, "load_state_dict"):
        model.load_state_dict(state)
    else:
        raise CampaignTrainerError(
            "MODEL_STATE_UNSUPPORTED", "model exposes no load_state_dict")


def _model_params(model) -> list:
    if hasattr(model, "parameters"):
        return list(model.parameters())
    return []


def _set_train(model, flag: bool) -> None:
    if hasattr(model, "train_mode"):
        model.train_mode(flag)
    elif hasattr(model, "train"):
        model.train(flag)


def _model_n_parameters(model) -> int:
    if hasattr(model, "n_parameters"):
        try:
            return int(model.n_parameters)
        except TypeError:
            pass
    return sum(p.numel() for p in _model_params(model))


def build_legacy_policy(torch, *, light: bool, signed_vector_messages: bool,
                        action_max: float = DEFAULT_ACTION_MAX,
                        signed_mu_gate: bool = False):
    """Reconstruct the old ``DualTSPolicy`` (radial-tanh NLL head)."""
    from pes2ts_core.pesn.models.dual_egnn import DualTSModelConfig
    from pes2ts_core.pesn.models.dual_ts_policy import (
        DEFAULT_SIGMA_RANGE,
        DualTSPolicy,
    )
    return DualTSPolicy(
        config=DualTSModelConfig(light=light,
                                 signed_vector_messages=signed_vector_messages),
        action_max=action_max,
        sigma_range=DEFAULT_SIGMA_RANGE,
        signed_mu_gate=signed_mu_gate,
    )


def build_displacement_policy(*args, **kwargs):
    """Lazy import of Lane B's displacement-policy factory (single call point).

    Tests monkeypatch this function (or inject a fake) before building a
    displacement arm.
    """
    from pes2ts_core.pesn.models.dual_ts_displacement import (
        build_displacement_policy as _build,
    )
    return _build(*args, **kwargs)


def load_dev_split(split_path):
    """Lazy import of Lane A's dev-split loader (single call point)."""
    from pes2ts_core.pesn.offline.data.dual_ts_dev_split import (
        load_dev_split as _load,
    )
    return _load(split_path)


# ---------------------------------------------------------------------------
# E2 key-pair conditioning (contract §5 E2 / plan §6)
# ---------------------------------------------------------------------------

def _reconstruct_endpoint_graphs(state: dict):
    """Reconstruct endpoint A/B ``EndpointGraph`` from the npz union topology.

    ``edge_pairs`` is the undirected union (i < j); ``edge_feats_a``/``b`` is
    the directed 4-channel block (index 0 = bond_present, 1 = bond_order,
    2 = changed_flag, 3 = stereo_known).  Both the v3 shared (E, 4) and the
    v4 per-state (E, 7) layouts share the first four channels, so the bond
    maps are reconstructed identically.
    """
    from pes2ts_core.pesn.contracts.dual_ts import EndpointBond, EndpointGraph
    z = np.asarray(state["atomic_numbers"], dtype=np.int64)
    edge_pairs = np.asarray(state["edge_pairs"], dtype=np.int64)
    edge_a = np.asarray(state["edge_feats_a"], dtype=np.float64)
    edge_b = np.asarray(state["edge_feats_b"], dtype=np.float64)
    n = int(z.size)
    atom_map_ids = tuple(f"m{i + 1:04d}" for i in range(n))
    bonds_a: list = []
    bonds_b: list = []
    for k, (i, j) in enumerate(edge_pairs):
        i, j = int(i), int(j)
        if edge_a[k, 0] > 0.5:  # bond_present (A's view) on the union pair
            bonds_a.append(EndpointBond(i=i, j=j, bond_order=float(edge_a[k, 1])))
        if edge_b[k, 0] > 0.5:
            bonds_b.append(EndpointBond(i=i, j=j, bond_order=float(edge_b[k, 1])))
    return (
        EndpointGraph(atomic_numbers=z, atom_map_ids=atom_map_ids,
                      bonds=tuple(bonds_a)),
        EndpointGraph(atomic_numbers=z, atom_map_ids=atom_map_ids,
                      bonds=tuple(bonds_b)),
    )


class KeyPairWrappedEncoder(torch.nn.Module):
    """Wrap the displacement encoder with the E2 ``KeyPairConditioner``.

    ``forward(inputs)`` returns the SAME dict structure as the raw encoder
    (``{"h", "v"}``) but with ``h`` FiLM-modulated by the per-batch key-pair
    records in ``inputs["pair_records"]`` / ``inputs["pair_indices"]``.  The
    conditioner's parameters are registered and counted by ``n_parameters``.
    """

    def __init__(self, encoder, hidden: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.hidden = int(hidden)
        from pes2ts_core.pesn.features.dual_ts_graph_v4 import (
            KeyPairConditioner,
        )
        self.conditioner = KeyPairConditioner(hidden=self.hidden)

    def forward(self, inputs: dict) -> dict:
        enc = self.encoder(inputs)
        if isinstance(enc, dict):
            h, v = enc["h"], enc["v"]
        else:
            h, v = enc
        h = self.conditioner(
            h, inputs["pair_records"], inputs["pair_indices"])
        return {"h": h, "v": v}


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class CampaignTrainer:
    """Model-agnostic campaign trainer (legacy + displacement heads)."""

    def __init__(
        self,
        *,
        arm: CampaignArmConfig,
        states: Sequence[dict],
        data_meta: dict,
        run_dir: str | Path,
        dev_states: Sequence[dict] | None = None,
        prior_spec: dict | None = None,
        deadline: float | None = None,
    ) -> None:
        self.arm = arm
        self.states = list(states)
        self.data_meta = dict(data_meta)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.dev_states = list(dev_states) if dev_states is not None else []
        self.prior_spec = dict(prior_spec) if prior_spec else None
        self.deadline = deadline

        self.drawable = [s for s in self.states if not s["is_teacher"]]
        _require(bool(self.drawable), "EMPTY_ACTION_POPULATION",
                 "no non-teacher states to draw")

        self.torch = require_torch()
        from pes2ts_core.pesn.offline.training.device import (
            resolve_device, seed_everything, set_cpu_threads)
        self.torch, self.device = resolve_device(
            device=arm.device, require_cuda=arm.require_cuda)
        set_cpu_threads(intra=2, inter=1)
        seed_everything(arm.seed)

        self.is_legacy = arm.is_legacy
        self.action_max = DEFAULT_ACTION_MAX
        self.model = self._build_model()
        self._params = _model_params(self.model)
        self.optimizer, self.scheduler = build_optim(
            self._params, preset=arm.preset, total_updates=arm.updates)
        self.ema = EMATracker(
            decay=EMA_DECAY,
            start_after_warmup=campaign_warmup_steps(arm.updates)
            if arm.preset == "campaign_v1" else 0)
        self.ema.register(self._params)

        # frozen positive length scale (displacement only)
        self.a_s = compute_a_s(self.drawable) if not self.is_legacy else 0.0

        # frozen physical prior actions (H1 skip)
        self._a_phys_cache: dict[str, np.ndarray] = {}
        if not self.is_legacy and arm.use_prior_skip:
            self._prime_physical_prior()

        # E2 key-pair conditioning (per-state records + whitelist)
        self._keypair_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if not self.is_legacy and arm.keypair_condition:
            self._prime_keypairs()

        # sampler
        self.sampler = self._build_sampler()

        # exposure (extended)
        long_step_p90 = self._long_step_p90_threshold()
        self.exposure = CampaignExposureTracker(
            row_meta=[
                {
                    "group_id": s["reaction_group_id"],
                    "state_id": s["state_id"],
                    "branch_id": s["branch_id"],
                    "remaining_arc_length": s["remaining_arc_length"],
                    "max_action_norm": (
                        float(np.linalg.norm(s["action"], axis=1).max())
                        if s.get("action") is not None else None),
                }
                for s in self.drawable
            ],
            long_step_p90_threshold=long_step_p90,
        )

        self.history: list[dict] = []
        self.diagnostics: list[dict] = []
        self.dev_evals: dict[str, dict] = {}
        self.updates_run = 0
        self.samples_exposed = 0
        self.resume_events: list[dict] = []
        self.ema_read = arm.ema_read
        self.checkpoints: list[dict] = []
        self.budget_state: dict = {"status": "pending"}
        if self.is_legacy:
            self._loss_keys: list[str] = ["nll_loss"]
        else:
            self._loss_keys = ["l_xyz"]
            if arm.lambda_dir is not None:
                self._loss_keys.append("l_dir")
            if arm.lambda_mag is not None:
                self._loss_keys.append("l_mag")

    # -- model / sampler / prior --------------------------------------------

    def _build_model(self):
        torch = self.torch
        if self.is_legacy:
            return build_legacy_policy(
                torch, light=self.arm.light,
                signed_vector_messages=False).to(
                    torch.float32).to(self.device)
        # displacement head via Lane B (real config objects; dict payloads
        # fail with AttributeError on attribute access)
        from pes2ts_core.pesn.models.dual_egnn import DualTSModelConfig
        from pes2ts_core.pesn.models.dual_ts_displacement import (
            DisplacementHeadConfig,
        )
        model_config = DualTSModelConfig(
            d_model=32, d_element=16, max_z=32,
            n_basis=32, cutoff=5.0, light=False,
            edge_feat_dim=7 if self.arm.features_version == "v4" else 4,
        )
        head_config = DisplacementHeadConfig(
            readout_hidden=64,
            use_prior_skip=self.arm.use_prior_skip)
        policy = build_displacement_policy(
            model_config, head_config,
            encoder_kind=self.arm.encoder,
            vector_channels=self.arm.vector_channels,
            contact_edges=self.arm.contact_edges,
            hidden=self.arm.hidden, n_layers=self.arm.n_layers,
            edge_feat_dim=7 if self.arm.features_version == "v4" else 4,
        )
        if self.arm.keypair_condition:
            # E2: wrap the encoder so the conditioner's params are registered
            # and counted (the head still reads policy.encoder(inputs)["h"]).
            policy.encoder = KeyPairWrappedEncoder(
                policy.encoder, hidden=self.arm.hidden)
        return policy.to(torch.float32).to(self.device)

    def _build_sampler(self):
        if self.arm.sampler == "epoch_traversal_v41":
            return EpochTraversalSampler(
                seed=self.arm.seed, states=self.states,
                batch_states=self.arm.batch_states)
        groups = build_sampler_groups(
            self.drawable,
            group_key=lambda s: s["reaction_group_id"],
            branch_key=lambda s: s["branch_id"],
            arc_key=lambda s, _i: s["remaining_arc_length"],
        )
        return DualTSGroupBranchSampler(
            seed=self.arm.seed, groups=groups, rows_per_group=1)

    def _long_step_p90_threshold(self) -> float | None:
        if self.is_legacy:
            return None
        norms = []
        for s in self.drawable:
            if s.get("action") is None:
                continue
            norms.append(float(np.linalg.norm(s["action"], axis=1).max()))
        if not norms:
            return None
        return float(np.percentile(np.asarray(norms), 90))

    def _prime_physical_prior(self) -> None:
        _require(self.prior_spec is not None, "PRIOR_SPEC_MISSING",
                 "use_prior_skip requires a physical prior spec")
        from pes2ts_core.pesn.offline.data.direct_preprocess import masses_for
        # The Lane A artifact wraps the spec payload under ``spec``.
        spec = dict(self.prior_spec.get("spec", self.prior_spec))
        s_g = float(spec.get("s_g", 0.0) or 0.0)
        eps_g = float(spec.get("eps_g", 0.0))
        tau_g = float(spec.get("tau_g", 0.0))
        delta_s_fit = float(spec.get("delta_s_fit", 0.0))
        seen = {s["state_id"] for s in self.states}
        for s in list(self.states) + [
                s for s in self.dev_states if s["state_id"] not in seen]:
            self._a_phys_cache[s["state_id"]] = compute_physical_prior_action(
                s, s_g=s_g, eps_g=eps_g, tau_g=tau_g, delta_s_fit=delta_s_fit,
                masses_for=masses_for)

    def _prime_keypairs(self) -> None:
        """Precompute per-state key-pair whitelist + records for E2 collation.

        Primes BOTH the training and dev states: ``dev_evaluate`` forwards the
        fit5-dev population, which the key-pair collation must also cover.
        """
        from pes2ts_core.pesn.features.dual_ts_graph_v4 import (
            KEY_PAIR_RECORD_DIM,
            build_key_pair_records,
            select_key_pairs,
        )
        seen = {s["state_id"] for s in self.states}
        all_states = list(self.states) + [
            s for s in self.dev_states if s["state_id"] not in seen]
        _require(all("edge_pairs" in s for s in all_states),
                 "KEYPAIR_NO_EDGE_PAIRS",
                 "keypair_condition requires edge_pairs in the feature shards")
        graphs: dict[str, tuple] = {}
        for gid in {s["reaction_group_id"] for s in all_states}:
            rep = next(s for s in all_states if s["reaction_group_id"] == gid)
            graphs[gid] = _reconstruct_endpoint_graphs(rep)
        for s in all_states:
            graph_a, graph_b = graphs[s["reaction_group_id"]]
            key_pairs = select_key_pairs(graph_a, graph_b)
            if int(key_pairs.shape[0]) == 0:
                self._keypair_cache[s["state_id"]] = (
                    np.zeros((0, 2), dtype=np.int64),
                    np.zeros((0, KEY_PAIR_RECORD_DIM), dtype=np.float64))
                continue
            records = build_key_pair_records(
                key_pairs=key_pairs,
                x=np.asarray(s["x_angstrom"], dtype=np.float64),
                topology_a=graph_a, topology_b=graph_b)
            self._keypair_cache[s["state_id"]] = (key_pairs, records)

    # -- batching / forward --------------------------------------------------

    def _collate(self, chunk: Sequence[dict]) -> dict:
        torch = self.torch
        from pes2ts_core.pesn.models.dual_egnn import collate_dual_states
        batch = collate_dual_states(torch, list(chunk))
        for key, val in batch.items():
            if torch.is_tensor(val):
                if val.dtype in (torch.float64, torch.float32):
                    val = val.to(dtype=torch.float32)
                batch[key] = val.to(self.device)
        if not self.is_legacy and self.arm.keypair_condition:
            records, indices = self._keypair_batch(chunk)
            batch["pair_records"] = records
            batch["pair_indices"] = indices
        return batch

    def _keypair_batch(self, chunk: Sequence[dict]):
        """Concatenated per-state key-pair records + offset pair indices."""
        torch = self.torch
        from pes2ts_core.pesn.features.dual_ts_graph_v4 import (
            KEY_PAIR_RECORD_DIM,
        )
        records: list[np.ndarray] = []
        indices: list[np.ndarray] = []
        offset = 0
        for s in chunk:
            key_pairs, rec = self._keypair_cache[s["state_id"]]
            n = int(key_pairs.shape[0])
            if n > 0:
                records.append(np.asarray(rec, dtype=np.float64))
                indices.append(np.asarray(key_pairs, dtype=np.int64) + offset)
            offset += int(np.asarray(s["atomic_numbers"]).size)
        if not records:
            return (
                torch.zeros(0, KEY_PAIR_RECORD_DIM, device=self.device),
                torch.zeros(0, 2, dtype=torch.long, device=self.device),
            )
        return (
            torch.as_tensor(np.concatenate(records, axis=0),
                            dtype=torch.float32).to(self.device),
            torch.as_tensor(np.concatenate(indices, axis=0),
                            dtype=torch.long).to(self.device),
        )

    def _targets(self, chunk: Sequence[dict]) -> dict:
        torch = self.torch
        if self.is_legacy:
            targets = build_targets(list(chunk), self.action_max)
            for key in ("u_star", "a_star", "atom_valid", "dir_mask",
                        "cell_index", "cell_group", "state_of_atom"):
                targets[key] = targets[key].to(self.device)
            return targets
        targets = build_displacement_targets(list(chunk), torch)
        for key in ("a_star", "atom_valid", "dir_mask", "cell_index",
                    "cell_group", "state_of_atom"):
            targets[key] = targets[key].to(self.device)
        if self.arm.use_prior_skip:
            a_phys = np.concatenate(
                [self._a_phys_cache[s["state_id"]] for s in chunk], axis=0)
            targets["a_phys"] = torch.as_tensor(
                a_phys, dtype=torch.float32).to(self.device)
        return targets

    def _forward(self, inputs: dict, targets: dict) -> dict:
        if not self.is_legacy and self.arm.use_prior_skip:
            return self.model(inputs, targets["a_phys"])
        return self.model(inputs)

    # -- loss ----------------------------------------------------------------

    def compute_losses(self, inputs: dict, targets: dict, out: dict) -> dict:
        torch = self.torch
        if self.is_legacy:
            return self._compute_legacy_losses(inputs, targets, out)
        return self._compute_displacement_losses(targets, out)

    def _compute_legacy_losses(self, inputs, targets, out) -> dict:
        torch = self.torch
        mu, log_sigma = out["mu"], out["log_sigma"]
        valid = targets["atom_valid"]
        nll_atom = per_atom_nll(self.model, mu, log_sigma, targets["u_star"])
        n_states = targets["n_states"]
        nll_state = torch.zeros(n_states, device=mu.device).index_add_(
            0, targets["state_of_atom"], nll_atom.masked_fill(~valid, 0.0))
        nll_loss, nll_group = reduce_step_sums(
            torch, nll_state, targets["cell_index"], targets["n_cells"],
            targets["cell_group"], targets["n_groups"])
        return {
            "loss": nll_loss,
            "nll_loss": nll_loss.detach(),
            "nll_group": nll_group.detach(),
            "nll_per_atom": float(
                nll_atom.detach().masked_fill(~valid, 0.0).sum()
                / valid.to(mu.dtype).sum().clamp(min=1.0)),
        }

    def compute_loss_components(self, targets: dict, out: dict) -> dict:
        """Non-detached ``l_xyz`` / ``l_dir`` / ``l_mag`` scalars (λ canary).

        Mirrors the reduction of :meth:`_compute_displacement_losses` but
        returns the differentiable tensors so the calibrator can backprop
        each component separately through the shared encoder parameters.
        """
        torch = self.torch
        a_hat = out["a_hat"]
        a_star = targets["a_star"]
        valid = targets["atom_valid"]
        state_of_atom = targets["state_of_atom"]
        n_states = targets["n_states"]
        diff2 = ((a_hat - a_star) ** 2).sum(dim=-1) / (self.a_s ** 2)
        counts = torch.zeros(n_states, device=a_hat.device).index_add_(
            0, state_of_atom, valid.to(a_hat.dtype))
        sums = torch.zeros(n_states, device=a_hat.device).index_add_(
            0, state_of_atom, diff2.masked_fill(~valid, 0.0))
        l_xyz, _ = reduce_step_sums(
            torch, sums / counts.clamp(min=1.0), targets["cell_index"],
            targets["n_cells"], targets["cell_group"], targets["n_groups"])

        dir_mask = targets["dir_mask"]
        cos_atom = per_atom_cos_dir_loss(a_hat, a_star)
        dir_state = torch.zeros(n_states, device=a_hat.device).index_add_(
            0, state_of_atom, cos_atom.masked_fill(~dir_mask, 0.0))
        dir_counts = torch.zeros(n_states, device=a_hat.device).index_add_(
            0, state_of_atom, dir_mask.to(a_hat.dtype))
        l_dir, _ = reduce_step_sums(
            torch, dir_state / dir_counts.clamp(min=1.0), targets["cell_index"],
            targets["n_cells"], targets["cell_group"], targets["n_groups"])

        mag_atom = per_atom_mag_loss(a_hat, a_star)
        mag_state = torch.zeros(n_states, device=a_hat.device).index_add_(
            0, state_of_atom, mag_atom.masked_fill(~dir_mask, 0.0))
        l_mag, _ = reduce_step_sums(
            torch, mag_state / dir_counts.clamp(min=1.0),
            targets["cell_index"], targets["n_cells"],
            targets["cell_group"], targets["n_groups"])
        return {"l_xyz": l_xyz, "l_dir": l_dir, "l_mag": l_mag}

    def _compute_displacement_losses(self, targets, out) -> dict:
        torch = self.torch
        a_hat = out["a_hat"]
        a_star = targets["a_star"]
        valid = targets["atom_valid"]
        state_of_atom = targets["state_of_atom"]
        n_states = targets["n_states"]
        # L_xyz: atom mean -> state -> branch -> group
        diff2 = ((a_hat - a_star) ** 2).sum(dim=-1) / (self.a_s ** 2)
        counts = torch.zeros(n_states, device=a_hat.device).index_add_(
            0, state_of_atom, valid.to(a_hat.dtype))
        sums = torch.zeros(n_states, device=a_hat.device).index_add_(
            0, state_of_atom, diff2.masked_fill(~valid, 0.0))
        state_mean = sums / counts.clamp(min=1.0)
        l_xyz, group_means = reduce_step_sums(
            torch, state_mean, targets["cell_index"], targets["n_cells"],
            targets["cell_group"], targets["n_groups"])

        parts: dict = {"loss": l_xyz, "l_xyz": l_xyz.detach(),
                       "l_xyz_group": group_means.detach()}
        if self.arm.lambda_dir is not None or self.arm.lambda_mag is not None:
            dir_mask = targets["dir_mask"]
            cos_atom = per_atom_cos_dir_loss(a_hat, a_star)
            dir_state = torch.zeros(n_states, device=a_hat.device).index_add_(
                0, state_of_atom, cos_atom.masked_fill(~dir_mask, 0.0))
            dir_counts = torch.zeros(n_states, device=a_hat.device).index_add_(
                0, state_of_atom, dir_mask.to(a_hat.dtype))
            dir_mean = dir_state / dir_counts.clamp(min=1.0)
            l_dir, _ = reduce_step_sums(
                torch, dir_mean, targets["cell_index"], targets["n_cells"],
                targets["cell_group"], targets["n_groups"])
            mag_atom = per_atom_mag_loss(a_hat, a_star)
            mag_state = torch.zeros(n_states, device=a_hat.device).index_add_(
                0, state_of_atom, mag_atom.masked_fill(~dir_mask, 0.0))
            mag_mean = mag_state / dir_counts.clamp(min=1.0)
            l_mag, _ = reduce_step_sums(
                torch, mag_mean, targets["cell_index"], targets["n_cells"],
                targets["cell_group"], targets["n_groups"])
            parts["l_dir"] = l_dir.detach()
            parts["l_mag"] = l_mag.detach()
            total = l_xyz
            if self.arm.lambda_dir is not None:
                total = total + self.arm.lambda_dir * l_dir
            if self.arm.lambda_mag is not None:
                total = total + self.arm.lambda_mag * l_mag
            parts["loss"] = total
        return parts

    def _finite(self, values: Iterable[float]) -> bool:
        return all(np.isfinite(float(v)) for v in values)

    # -- training ------------------------------------------------------------

    def train(self) -> dict:
        torch = self.torch
        st = self.arm
        started = time.perf_counter()
        update = self.updates_run
        epoch = self.sampler.draws // max(1, len(self.drawable))
        n_drawable = len(self.drawable)
        running: dict[str, float] = {"n": 0.0, "loss": 0.0}
        for key in self._loss_keys:
            running[key] = 0.0

        while update < st.updates:
            # hard deadline: save first, then stop (budget_censored)
            if self.deadline is not None and time.time() >= self.deadline:
                self._save_checkpoint(update)
                self.budget_state = {
                    "status": "budget_censored",
                    "updates_run": update,
                    "reason": "hard deadline reached",
                }
                return self._finalize_record(update, started)

            rows = self.sampler.draw_batch(st.batch_states)
            self.exposure.record_batch(rows)
            chunk = [self.drawable[i] for i in rows]
            inputs = self._collate(chunk)
            targets = self._targets(chunk)
            _set_train(self.model, True)
            out = self._forward(inputs, targets)
            parts = self.compute_losses(inputs, targets, out)
            self.optimizer.zero_grad(set_to_none=True)
            parts["loss"].backward()
            if not self._grads_finite():
                raise CampaignTrainerError(
                    "NONFINITE_GRADIENT", f"non-finite gradient at update "
                    f"{update + 1}")
            torch.nn.utils.clip_grad_norm_(self._params, st.clip)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.ema.update(self._params)
            update += 1
            self.updates_run = update
            self.samples_exposed += len(chunk)

            vals = {}
            for k in ("loss",) + tuple(self._loss_keys):
                if k in parts:
                    t_ = parts[k]
                    vals[k] = float(t_.detach() if t_.requires_grad else t_)
            if not self._finite(vals.values()):
                raise CampaignTrainerError(
                    "NONFINITE_LOSS", f"non-finite loss at update {update}: "
                    f"{vals}")
            running["n"] += 1
            for k in vals:
                running[k] += vals[k]

            new_epoch = self.sampler.draws // max(1, n_drawable)
            while new_epoch > epoch:
                epoch += 1
                self.exposure.snapshot(kind="epoch", update=update, epoch=epoch)
            if update % 1000 == 0:
                self.exposure.snapshot(kind="update_1000", update=update,
                                       epoch=epoch)
            if update % st.curve_every == 0 or update == st.updates:
                self.history.append(self._history_entry(
                    update, epoch, running, parts, started))
                running = {"n": 0.0, "loss": 0.0}
                for key in self._loss_keys:
                    running[key] = 0.0
            if update in st.eval_points:
                self.diagnostics.append({
                    "update": update,
                    **self.fit_diagnostics(tag=f"u{update}"),
                })
                self._save_checkpoint(update)
                if self.dev_states:
                    self.dev_evals[f"u{update}"] = self.dev_evaluate(
                        self.dev_states)

        self.exposure.snapshot(kind="final", update=update, epoch=epoch)
        self.budget_state = {"status": "completed", "updates_run": update}
        return self._finalize_record(update, started)

    def _grads_finite(self) -> bool:
        for p in self._params:
            if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
                return False
        return True

    def _history_entry(self, update, epoch, running, parts, started) -> dict:
        n = max(int(running["n"]), 1)
        entry = {
            "update": update,
            "epoch": epoch,
            "loss": running["loss"] / n,
            "samples_exposed": self.samples_exposed,
            "seconds": round(time.perf_counter() - started, 2),
        }
        for key in self._loss_keys:
            entry[key] = running[key] / n
        return entry

    def _finalize_record(self, update: int, started: float) -> dict:
        final = self.fit_diagnostics(tag="final")
        record = self._build_record(update, started, final)
        write_json(self.run_dir / "run_record.json", record)
        return record

    def _build_record(self, update: int, started: float,
                      fit_final: dict) -> dict:
        st = self.arm
        wall = time.perf_counter() - started
        timing = self._timing_payload(update, wall)
        return {
            "schema_version": RUN_RECORD_V3_SCHEMA,
            "trainer_version": CAMPAIGN_TRAINER_VERSION,
            "arm_id": st.arm_id,
            "preset": st.preset,
            "recipe": st.head,
            "settings": st.to_payload(),
            "data": {k: v for k, v in self.data_meta.items()},
            "accounting": {
                "updates_run": self.updates_run,
                "batch_states": st.batch_states,
                "samples_exposed": self.samples_exposed,
                "sampler_seed": st.seed,
                "sampler_rule": self.sampler.sampling_rule,
                "within_branch_rule": self.sampler.within_branch_rule,
                "drawable_states": len(self.drawable),
                "n_groups": self.sampler.n_groups,
                "rounds_completed": getattr(
                    self.sampler, "rounds_completed", 0),
                "a_s_angstrom": self.a_s if not self.is_legacy else None,
            },
            "exposure": self.exposure.payload(),
            "history": self.history,
            "diagnostics": self.diagnostics,
            "fit_diagnostics_final": fit_final,
            "wall_seconds": round(wall, 2),
            "checkpoints": list(self.checkpoints),
            "dev_evals": dict(self.dev_evals),
            "ema": {
                "decay": EMA_DECAY,
                "start_after_warmup": self.ema.start_after_warmup,
                "steps": int(self.ema.steps),
                "read_rule": self.ema_read,
            },
            "timing": timing,
            "resume_events": list(self.resume_events),
            "budget_state": dict(self.budget_state),
        }

    def _timing_payload(self, update: int, wall: float) -> dict:
        peak_mem_mb = 0.0
        if self.torch.cuda.is_available():
            try:
                peak_mem_mb = float(
                    self.torch.cuda.max_memory_allocated() / 1e6)
            except Exception:
                peak_mem_mb = 0.0
        return {
            "steps_per_sec": round(update / max(wall, 1e-9), 3),
            "peak_mem_mb": peak_mem_mb,
            "eval_seconds": round(self._last_eval_seconds, 3),
            "n_parameters": _model_n_parameters(self.model),
        }

    @property
    def _last_eval_seconds(self) -> float:
        return getattr(self, "_eval_seconds", 0.0)

    # -- diagnostics + dev eval ---------------------------------------------

    def fit_diagnostics(self, *, tag: str, batch_size: int = 256) -> dict:
        if self.is_legacy:
            return self._legacy_fit_diagnostics(tag=tag, batch_size=batch_size)
        a_pred = self._forward_deterministic(self.states, batch_size=batch_size)
        a_star, valid, _ = self._displacement_arrays(self.states)
        mse = float(((a_pred - a_star) ** 2).sum(-1)[valid].mean())
        zero = float((a_star ** 2).sum(-1)[valid].mean())
        return {
            "tag": tag,
            "population": "fit_partition_train_side_diagnostic_only",
            "per_atom_action_mse_A2": mse,
            "zero_baseline_action_mse_A2": zero,
            "mse_over_baseline": mse / zero if zero > 0 else None,
        }

    def _legacy_fit_diagnostics(self, *, tag: str, batch_size: int = 256) -> dict:
        torch = self.torch
        _set_train(self.model, False)
        mu_all = []
        with torch.no_grad():
            for i in range(0, len(self.states), batch_size):
                chunk = list(self.states[i:i + batch_size])
                inputs = self._collate(chunk)
                targets = self._targets(chunk)
                out = self._forward(inputs, targets)
                mu_all.append(out["mu"].cpu())
        from pes2ts_core.pesn.models.direct_policy import radial_transform_stable
        mu = torch.cat(mu_all, dim=0)
        targets = build_targets(self.states, self.action_max)
        a_det = self.action_max * radial_transform_stable(torch, mu)
        valid = targets["atom_valid"]
        diff2 = (a_det - targets["a_star"]).pow(2).sum(-1)
        mse = float(diff2[valid].mean())
        zero = float((targets["a_star"] ** 2).sum(-1)[valid].mean())
        return {
            "tag": tag,
            "population": "fit_partition_train_side_diagnostic_only",
            "per_atom_action_mse_A2": mse,
            "zero_baseline_action_mse_A2": zero,
            "mse_over_baseline": mse / zero if zero > 0 else None,
        }

    def _forward_deterministic(self, states, *, batch_size: int = 256):
        torch = self.torch
        _set_train(self.model, False)
        preds = []
        with torch.no_grad():
            for i in range(0, len(states), batch_size):
                chunk = list(states[i:i + batch_size])
                inputs = self._collate(chunk)
                targets = self._targets(chunk)
                out = self._forward(inputs, targets)
                if self.is_legacy:
                    a = self.model.deterministic(out["mu"])
                else:
                    a = out["a_hat"]
                preds.append(a.detach().cpu())
        return torch.cat(preds, dim=0).numpy().astype(np.float64)

    def _displacement_arrays(self, states):
        t = build_displacement_targets(list(states), self.torch)
        return (t["a_star"].numpy().astype(np.float64),
                t["atom_valid"].numpy(),
                t["dir_mask"].numpy())

    def dev_evaluate(self, states: Sequence[dict]) -> dict:
        """fit5-dev metrics (group median MSE/zero, cos, near-TS cos,
        near-zero atom RMSE, effective denominators, per-group)."""
        group_ids = sorted({s["reaction_group_id"] for s in states})
        by_group: dict[str, list[dict]] = {g: [] for g in group_ids}
        for s in states:
            by_group.setdefault(s["reaction_group_id"], []).append(s)
        per_group = []
        for gid in group_ids:
            gstates = by_group[gid]
            a_pred = self._forward_deterministic(gstates)
            t = build_displacement_targets(list(gstates), self.torch)
            a_star = t["a_star"].numpy().astype(np.float64)
            valid = t["atom_valid"].numpy()
            dir_mask = t["dir_mask"].numpy()
            near_ts = _compute_near_ts_mask(gstates)
            per_group.append({
                "group_id": gid,
                "n_states": len(gstates),
                "n_action_states": sum(
                    1 for s in gstates if not s["is_teacher"]),
                "all": _group_metrics_block(
                    a_pred, a_star, valid, dir_mask,
                    np.ones_like(valid, dtype=bool)),
                "near_ts": _group_metrics_block(
                    a_pred, a_star, valid, dir_mask, near_ts),
            })
        summary = _summarize_dev_groups(per_group)
        return {"per_group": per_group, "summary": summary,
                "n_groups": len(group_ids)}

    def run_dev_eval_with_timing(self, states: Sequence[dict]) -> dict:
        started = time.perf_counter()
        result = self.dev_evaluate(states)
        self._eval_seconds = time.perf_counter() - started
        return result

    # -- checkpoint ----------------------------------------------------------

    def _save_checkpoint(self, update: int) -> None:
        torch = self.torch
        ckpt_dir = self.run_dir / "checkpoints" / f"u{update}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ema_model_state = self._ema_model_state()
        raw_payload = {
            "schema_version": CHECKPOINT_SCHEMA,
            "arm": self.arm.to_payload(),
            "update": int(update),
            "model_state": _model_state_dict(self.model),
            "ema_state": self.ema.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": (self.scheduler.state_dict()
                                if self.scheduler is not None else None),
            "sampler_state": self.sampler.state_dict(),
            "exposure_state": self.exposure.state_dict(),
            "updates_run": int(self.updates_run),
            "samples_exposed": int(self.samples_exposed),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "identity": self.checkpoint_identity(),
        }
        raw_path = ckpt_dir / "raw.pt"
        torch.save(raw_payload, raw_path)
        ema_path = ckpt_dir / "ema.pt"
        torch.save({
            "schema_version": EMA_CHECKPOINT_SCHEMA,
            "arm": self.arm.to_payload(),
            "update": int(update),
            "model_state": ema_model_state,
            "identity": self.checkpoint_identity(),
        }, ema_path)
        self.checkpoints.append({
            "update": int(update),
            "dir": str(ckpt_dir.relative_to(self.run_dir)),
            "raw_file": "raw.pt",
            "ema_file": "ema.pt",
        })

    def _ema_model_state(self) -> dict:
        raw = [p.detach().clone() for p in self._params]
        self.ema.apply_to(self._params)
        try:
            return _model_state_dict(self.model)
        finally:
            for p, r in zip(self._params, raw):
                p.data.copy_(r)

    def checkpoint_identity(self) -> dict:
        st = self.arm
        return {
            "arm_id": st.arm_id,
            "head": st.head,
            "preset": st.preset,
            "seed": st.seed,
            "features_version": st.features_version,
            "encoder": st.encoder,
            "sampler": st.sampler,
            "split_sha256": self.data_meta.get("split_sha256", ""),
            "schema_version": RUN_RECORD_V3_SCHEMA,
        }

    # -- resume --------------------------------------------------------------

    def resume(self) -> dict:
        """Restore from the newest checkpoint and continue training."""
        torch = self.torch
        ckpt = self._latest_checkpoint()
        if ckpt is None:
            raise CampaignTrainerError(
                "RESUME_NO_CHECKPOINT", "no checkpoint to resume from")
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != CHECKPOINT_SCHEMA:
            raise CampaignTrainerError(
                "CHECKPOINT_SCHEMA_UNKNOWN",
                str(payload.get("schema_version")))
        if payload["arm"]["arm_id"] != self.arm.arm_id or \
                payload["arm"]["seed"] != self.arm.seed:
            raise CampaignTrainerError(
                "RESUME_CONFIG_MISMATCH", "checkpoint arm identity mismatch")
        _model_load_state_dict(self.model, payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        if self.scheduler is not None and payload.get("scheduler_state"):
            self.scheduler.load_state_dict(payload["scheduler_state"])
        self.ema.load_state_dict(payload["ema_state"], self._params)
        self.sampler.load_state_dict(payload["sampler_state"])
        self.exposure.load_state_dict(payload["exposure_state"])
        self.updates_run = int(payload.get("updates_run", 0))
        self.samples_exposed = int(payload.get("samples_exposed", 0))
        torch.set_rng_state(payload["torch_rng_state"])
        np.random.set_state(payload["numpy_rng_state"])
        random.setstate(payload["python_rng_state"])
        self.resume_events.append({
            "from": str(ckpt),
            "update": self.updates_run,
            "at": time.time(),
        })
        self.dev_evals = {}
        self.checkpoints = []
        return payload

    def _latest_checkpoint(self) -> Path | None:
        ckpts = sorted(
            (self.run_dir / "checkpoints").glob("u*/raw.pt")
        ) if (self.run_dir / "checkpoints").exists() else []
        if not ckpts:
            return None
        def _update_key(p: Path) -> int:
            return int(p.parent.name[1:])
        return max(ckpts, key=_update_key)


# ---------------------------------------------------------------------------
# Dev-eval helpers
# ---------------------------------------------------------------------------

def _compute_near_ts_mask(states: Sequence[dict]) -> np.ndarray:
    n_atoms = [int(np.asarray(s["atomic_numbers"]).size) for s in states]
    branches, ral = [], []
    for s, n in zip(states, n_atoms):
        branches.extend([s["branch_id"]] * n)
        ral.extend([s["remaining_arc_length"]] * n)
    branches = np.asarray(branches)
    ral = np.asarray(ral, dtype=float)
    near_ts = np.zeros(ral.shape[0], dtype=bool)
    for b in set(branches.tolist()):
        idx = branches == b
        near_ts[idx] = ral[idx] <= NEAR_TS_ARC_FRACTION * float(ral[idx].max())
    return near_ts


def _group_metrics_block(a_pred, a_star, valid, dir_mask, extra_mask) -> dict:
    m = valid & extra_mask
    n = int(m.sum())
    if n == 0:
        return {"n_atoms": 0}
    mse_model = float(((a_pred - a_star) ** 2).sum(-1)[m].mean())
    mse_zero = float((a_star ** 2).sum(-1)[m].mean())
    d = m & dir_mask
    if bool(d.any()):
        w = np.linalg.norm(a_star[d], axis=1)
        ad = a_pred[d]
        ast = a_star[d]
        cos = (ad * ast).sum(1) / np.maximum(
            np.linalg.norm(ad, axis=1) * w, 1e-12)
        cos_w = float((w * cos).sum() / w.sum())
        n_dir = int(d.sum())
    else:
        cos_w = None
        n_dir = 0
    nz = m & ~dir_mask
    if bool(nz.any()):
        rmse = float(np.sqrt(((a_pred - a_star) ** 2).sum(-1)[nz].mean()))
        n_nz = int(nz.sum())
    else:
        rmse = None
        n_nz = 0
    return {
        "n_atoms": n,
        "mse_model_A2": mse_model,
        "mse_zero_A2": mse_zero,
        "mse_ratio": mse_model / mse_zero if mse_zero > 0 else None,
        "direction_cosine": cos_w,
        "n_atoms_direction": n_dir,
        "near_zero_rmse_A": rmse,
        "n_atoms_near_zero": n_nz,
    }


def _summarize_dev_groups(per_group: Sequence[dict]) -> dict:
    def col(key):
        return [g["all"].get(key) for g in per_group
                if g["all"].get("n_atoms", 0) > 0]

    ratios = [v for v in col("mse_ratio") if v is not None]
    cosines = [v for v in col("direction_cosine") if v is not None]
    near_ts_cos = [g["near_ts"].get("direction_cosine") for g in per_group
                   if g["near_ts"].get("direction_cosine") is not None]
    total_model = sum(g["all"]["mse_model_A2"] * g["all"]["n_atoms"]
                      for g in per_group if g["all"].get("n_atoms"))
    total_zero = sum(g["all"]["mse_zero_A2"] * g["all"]["n_atoms"]
                     for g in per_group if g["all"].get("n_atoms"))
    rmse_vals = [g["all"]["near_zero_rmse_A"] for g in per_group
                 if g["all"].get("near_zero_rmse_A") is not None]
    n_near_zero = sum(g["all"]["n_atoms_near_zero"] for g in per_group)
    return {
        "n_groups": len(per_group),
        "mse_ratio_per_group": [g["all"].get("mse_ratio") for g in per_group],
        "mse_ratio_median": float(np.median(ratios)) if ratios else None,
        "n_groups_ratio_below_one": int(sum(1 for v in ratios if v < 1.0)),
        "direction_cosine_per_group": [
            g["all"].get("direction_cosine") for g in per_group],
        "direction_cosine_median": (
            float(np.median(cosines)) if cosines else None),
        "n_groups_cosine_positive": int(sum(1 for v in cosines if v > 0)),
        "near_ts_cosine_median": (
            float(np.median(near_ts_cos)) if near_ts_cos else None),
        "pooled_mse_ratio": (
            total_model / total_zero if total_zero > 0 else None),
        "near_zero_rmse_pooled_A": (
            float(np.sqrt(np.mean(np.asarray(rmse_vals) ** 2)))
            if rmse_vals else None),
        "effective_denominators": {
            "n_groups": len(per_group),
            "n_atoms_direction": sum(g["all"]["n_atoms_direction"]
                                      for g in per_group),
            "n_atoms_near_zero": n_near_zero,
        },
    }


# ---------------------------------------------------------------------------
# Single-arm runner
# ---------------------------------------------------------------------------

def _timing_estimate(timing: dict, full_updates: int,
                     n_evals: int) -> dict:
    """×1.25 margin estimate of a full run from a timing probe."""
    steps_per_sec = float(timing.get("steps_per_sec", 0.0) or 0.0)
    eval_seconds = float(timing.get("eval_seconds", 0.0) or 0.0)
    if steps_per_sec > 0:
        train_seconds = full_updates / steps_per_sec
    else:
        train_seconds = float("inf")
    full_seconds = (train_seconds + int(n_evals) * eval_seconds) * 1.25
    return {
        "steps_per_sec": steps_per_sec,
        "eval_seconds": eval_seconds,
        "full_updates": int(full_updates),
        "n_evals": int(n_evals),
        "estimated_full_seconds": full_seconds,
        "estimated_full_minutes": full_seconds / 60.0,
    }


def run_arm(arm: CampaignArmConfig, out_dir: str | Path, *,
            timing_only: bool = False, deadline: float | None = None,
            prior_spec: dict | None = None,
            split: dict | None = None,
            dev_states: Sequence[dict] | None = None,
            resume: bool = True) -> dict:
    """Run one arm (or a timing probe) and return the run record.

    ``split``/``dev_states`` may be injected by the driver/tests; otherwise
    the dev split is loaded through Lane A's ``load_dev_split`` and the
    feature npz through :func:`load_campaign_states`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if split is None:
        split = load_dev_split(arm.dataset.split_path)
    # Lane A's load_dev_split returns the raw artifact with train_group_ids /
    # dev_group_ids / split_sha256 (contract §3); accept both naming variants.
    train_groups = [str(g) for g in split.get(
        "train_group_ids", split.get("fit45_train_groups", []))]
    dev_groups = [str(g) for g in split.get(
        "dev_group_ids", split.get("fit5_dev_groups", []))]
    split_sha = str(split.get("split_sha256", ""))
    if arm.final_fit50:
        # V4.7 FINAL: train on the complete fit50 (dev groups re-absorbed);
        # no fit5-dev selection (contract §9.2).
        fit_groups = sorted(set(train_groups) | set(dev_groups))
        dev_groups = []
    else:
        fit_groups = train_groups

    train_states, meta = load_campaign_states(
        arm.dataset, group_ids=fit_groups)
    meta["split_sha256"] = split_sha
    meta["split_fit45_group_ids"] = train_groups
    meta["split_fit5_group_ids"] = dev_groups

    if dev_states is None:
        dev_states, _dev_meta = load_campaign_states(
            arm.dataset, group_ids=dev_groups)

    trainer = CampaignTrainer(
        arm=arm, states=train_states, data_meta=meta, run_dir=out_dir,
        dev_states=dev_states, prior_spec=prior_spec, deadline=deadline)

    if timing_only:
        record = trainer.timing_probe(full_updates=arm.updates)
    else:
        if resume:
            try:
                trainer.resume()
            except CampaignTrainerError as exc:
                if exc.code != "RESUME_NO_CHECKPOINT":
                    raise
        record = trainer.train()
    write_json(out_dir / "run_record.json", record)
    return record


# attach timing_probe method (kept out of the class body for readability)
def _timing_probe(self: CampaignTrainer, *, full_updates: int) -> dict:
    torch = self.torch
    st = self.arm
    started = time.perf_counter()
    # warmup 100 steps (discarded)
    for _ in range(100):
        rows = self.sampler.draw_batch(st.batch_states)
        chunk = [self.drawable[i] for i in rows]
        inputs = self._collate(chunk)
        targets = self._targets(chunk)
        _set_train(self.model, True)
        out = self._forward(inputs, targets)
        parts = self.compute_losses(inputs, targets, out)
        self.optimizer.zero_grad(set_to_none=True)
        parts["loss"].backward()
        torch.nn.utils.clip_grad_norm_(self._params, st.clip)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
    warmup_seconds = time.perf_counter() - started

    # time 500 steps
    t0 = time.perf_counter()
    for _ in range(500):
        rows = self.sampler.draw_batch(st.batch_states)
        chunk = [self.drawable[i] for i in rows]
        inputs = self._collate(chunk)
        targets = self._targets(chunk)
        out = self._forward(inputs, targets)
        parts = self.compute_losses(inputs, targets, out)
        self.optimizer.zero_grad(set_to_none=True)
        parts["loss"].backward()
        torch.nn.utils.clip_grad_norm_(self._params, st.clip)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
    timed_seconds = time.perf_counter() - t0

    # one dev eval
    eval_result = self.run_dev_eval_with_timing(self.dev_states)

    timing = {
        "steps_per_sec": 500.0 / max(timed_seconds, 1e-9),
        "peak_mem_mb": (float(torch.cuda.max_memory_allocated() / 1e6)
                        if torch.cuda.is_available() else 0.0),
        "eval_seconds": self._eval_seconds,
        "n_parameters": _model_n_parameters(self.model),
        "warmup_seconds": round(warmup_seconds, 3),
        "timed_seconds": round(timed_seconds, 3),
    }
    estimate = _timing_estimate(timing, full_updates,
                                n_evals=len(self.arm.eval_points))
    record = {
        "schema_version": RUN_RECORD_V3_SCHEMA,
        "trainer_version": CAMPAIGN_TRAINER_VERSION,
        "arm_id": st.arm_id,
        "preset": st.preset,
        "recipe": st.head,
        "settings": st.to_payload(),
        "data": {k: v for k, v in self.data_meta.items()},
        "timing_only": True,
        "timing": timing,
        "estimate": estimate,
        "dev_eval_sample": eval_result,
        "wall_seconds": round(time.perf_counter() - started, 2),
    }
    return record


CampaignTrainer.timing_probe = _timing_probe  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# CLI (invoked as a subprocess by the campaign driver)
# ---------------------------------------------------------------------------

def _main(argv: Sequence[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm-config", type=Path, required=True,
                    help="JSON arm config payload")
    ap.add_argument("--out", type=Path, required=True,
                    help="run directory")
    ap.add_argument("--timing-only", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--deadline", type=float, default=None,
                    help="absolute unix deadline (hard stop)")
    ap.add_argument("--prior-spec", type=Path, default=None,
                    help="physical prior spec JSON (H1/H2)")
    args = ap.parse_args(argv)

    arm = CampaignArmConfig.from_payload(
        json.loads(args.arm_config.read_text()))
    prior_spec = None
    if args.prior_spec is not None:
        prior_spec = json.loads(args.prior_spec.read_text())
    record = run_arm(
        arm, args.out, timing_only=args.timing_only,
        deadline=args.deadline, prior_spec=prior_spec,
        resume=not args.no_resume)
    print(f"[campaign] arm={arm.arm_id} seed={arm.seed} "
          f"updates={record.get('accounting', {}).get('updates_run', 'timing')}"
          f" budget={record.get('budget_state', {}).get('status', 'n/a')}",
          flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    import sys
    raise SystemExit(_main(sys.argv[1:]))


__all__ = [
    "CAMPAIGN_TRAINER_VERSION",
    "CHECKPOINT_SCHEMA",
    "DEFAULT_ACTION_MAX",
    "EMA_CHECKPOINT_SCHEMA",
    "EMA_READS",
    "ENCODERS",
    "FEATURE_VERSIONS",
    "HEADS",
    "NEAR_ZERO_ACTION_THRESHOLD",
    "RUN_RECORD_V3_SCHEMA",
    "SAMPLERS",
    "CampaignArmConfig",
    "CampaignDatasetSpec",
    "CampaignExposureTracker",
    "CampaignTrainer",
    "CampaignTrainerError",
    "EpochTraversalSampler",
    "KeyPairWrappedEncoder",
    "build_displacement_policy",
    "build_displacement_targets",
    "build_legacy_policy",
    "compute_a_s",
    "compute_physical_prior_action",
    "load_campaign_states",
    "load_dev_split",
    "per_atom_cos_dir_loss",
    "per_atom_mag_loss",
    "run_arm",
]
