"""Dual-TS BC trainer (WP4-A, plan v4 §I.8–I.10; G4 oracle prescription).

The G4 review (``.slim/deepwork/pes2ts-dualts-v4/g4-oracle-review.md``) is
BINDING for this module:

* **C1 — teacher-latent NLL.**  ``L_NLL = −log π_z(u*|s)`` evaluated AT the
  teacher latent ``u* = R⁻¹(a*/a_max)`` (inverse radial-tanh of the teacher
  action; NOT the WP3 self-sample entropy surrogate).  Per atom::

      NLL_i = 1.5·log2π + 3·log σ_i + 0.5‖u*_i − μ_i‖²/σ_i² + log_det(u*_i)

  bounded below (σ floor 0.02) and with ∂NLL/∂μ = (μ − u*)/σ², so gradients
  drive μ → u* and divergence is impossible.  The density/target version
  string ``radial_tanh_v0:bc_target=teacher_latent_v1`` is registered in
  every checkpoint and bundle.
* **C2 — three recipes; λ_a = λ_dir = 1.0 DEFAULT (arm A).**  λ is no longer
  hard-frozen (v4.1 WP2 arm B may unfreeze λ_dir): a run is accepted only if
  (recipe, λ_a, λ_dir) match a registered ``protocol_v41.json`` slot, else it
  refuses unless ``--allow-diagnostic-budget`` (recorded ``diagnostic_run``).::

      bc_nll : L = L_NLL
      bc_a   : L = L_NLL + λ_a·L_a,     L_a = ‖a_det(μ) − a*‖²
      bc_d   : bc_a + λ_dir·L_dir,      L_dir = ‖a*‖·(1 − cos(a_det, a*))

  ``a_det = a_max·tanh(‖μ‖)μ/‖μ‖``; ``L_dir`` runs on |a|-normalized
  directions, masked by the frozen near-zero mask (‖a*‖ < 0.01 Å excluded);
  near-rest atoms stay in ``L_a``.  Model selection uses monitor8 metrics,
  NEVER min NLL (implemented in :mod:`...training.dual_ts_monitor`).
* **§I.10 UNIFIED sampler (shared with IQL).**  BC draws via
  :class:`~pes2ts_core.pesn.offline.training.dual_ts_sampling.DualTSGroupBranchSampler`
  (``draw_batch``, group round-robin → branch uniform RNG → 3-bin arc
  round-robin) — the SAME surface IQL uses, so the same seed + dataset yields
  the IDENTICAL draw sequence (P06/F5).  Full per-group exposure accounting
  (draws, distinct-state coverage, repeat rate, near-TS fraction) is recorded
  in ``run_record.json`` with no truncation.  Reduction (fixed):
  per-step valid-atom SUM → branch mean → per-reaction mean over available
  branches → group mean over the groups present in the step.  Updates AND
  sample exposure are both reported.
* **C3 — signed variants default OFF**; flags surface in checkpoint config
  and bundles and are never silently flipped.
* **Fresh init** — never warm-started from WP3 checkpoints.

Monitor8 evaluation lives in
:mod:`pes2ts_core.pesn.offline.training.dual_ts_monitor` and refuses to run
without a content-digest-verified pre-registration (C4); this module only
trains on the fit partition and writes per-run records + validated
``pesn_dual_ts_bundle_v2`` bundles.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import numpy as np

from pes2ts_core.pesn.contracts.dual_ts import (
    ACTION_SCHEMA_VERSION,
    BUNDLE_SCHEMA_VERSION,
    DualTSBundle,
    DualTSFeaturesSpec,
    DualTSTeacher,
)
from pes2ts_core.pesn.features.dual_ts_graph import (
    build_teacher_supervision,
)
from pes2ts_core.pesn.models import require_torch
from pes2ts_core.pesn.models.direct_policy import (
    radial_transform_stable,
)
from pes2ts_core.pesn.models.dual_egnn import (
    DUAL_TS_EGNN_VERSION,
    DualTSModelConfig,
    collate_dual_states,
)
from pes2ts_core.pesn.models.dual_ts_policy import (
    DUAL_TS_POLICY_VERSION,
    DEFAULT_A_MAX_ANGSTROM,
    DEFAULT_SIGMA_RANGE,
    DualTSPolicy,
)
from pes2ts_core.pesn.offline.training.checkpoints import (
    BC_CHECKPOINT_SCHEMA_V2,
    save_bc_checkpoint_v2,
    save_checkpoint,
)
from pes2ts_core.pesn.offline.training.dual_ts_sampling import (
    GROUP_BRANCH_SAMPLING_RULE,
    NEAR_TS_ARC_FRACTION,
    WITHIN_BRANCH_RULE,
    DualTSGroupBranchSampler,
    ExposureTracker,
    build_sampler_groups,
)
from pes2ts_core.utils.jsonio import write_json

DUAL_TS_BC_VERSION: Final[str] = "pesn_dual_ts_bc_v2.0"

#: C1 binding: density + target identity registered in every bundle and
#: checkpoint (G4 §2 prescription).
DENSITY_TARGET_VERSION: Final[str] = (
    "radial_tanh_v0:bc_target=teacher_latent_v1")

RECIPES: Final[tuple[str, ...]] = ("bc_nll", "bc_a", "bc_d")

#: Recipe composition defaults (C2): λ_a = λ_dir = 1.0 are the DEFAULT
#: arm-A values.  λ is no longer hard-frozen — arm B may unfreeze λ_dir —
#: but a run is accepted only if (recipe, λ_a, λ_dir) match a registered
#: protocol_v41 entry (or carries ``--allow-diagnostic-budget``).
LAMBDA_A: Final[float] = 1.0
LAMBDA_DIR: Final[float] = 1.0

#: §I.9/G4 arc stratification (legacy epoch_state_order; kept for the frozen
#: v4 arc-stratum definition — the unified sampler uses its own 3-bin cursor
#: but the SAME near-TS accounting rule ``ral <= 0.2 * branch max``).
N_ARC_STRATA: Final[int] = 3

#: Hard divergence guard: the teacher-latent NLL is bounded below by
#: construction; a run exceeding this per-state level is a bug, not a
#: curve feature (G4: "verify no divergence").
NLL_DIVERGENCE_LIMIT: Final[float] = 1.0e4

_LOG_2PI: Final[float] = math.log(2.0 * math.pi)


class DualTSBCTrainerError(RuntimeError):
    """Typed dual-TS BC trainer refusal."""


def _require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise DualTSBCTrainerError(code, detail)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Data loading (WP3-A shards + WP2-C observed teacher labels)
# ---------------------------------------------------------------------------

def load_spec_and_manifest(features_dir: str | Path) -> tuple[str, dict]:
    """Re-verify the frozen features spec digest against the manifest."""
    features_dir = Path(features_dir)
    manifest = json.loads(
        (features_dir / "features_manifest.json").read_text())
    spec_payload = json.loads(
        (features_dir / "features_spec.json").read_text())
    digest = DualTSFeaturesSpec.from_payload(
        spec_payload["spec"]).content_digest()
    if digest != manifest["spec_digest"]:
        raise DualTSBCTrainerError(
            "SPEC_DIGEST_MISMATCH",
            f"recomputed {digest} != manifest {manifest['spec_digest']}")
    if digest != spec_payload["spec_digest"]:
        raise DualTSBCTrainerError(
            "SPEC_DIGEST_MISMATCH", "features_spec.json wrapper mismatch")
    return digest, manifest


def load_dual_ts_states(
    features_dir: str | Path,
    observed_dir: str | Path,
    *,
    expected_group_ids: Sequence[str] | None = None,
    group_filter: Sequence[str] | None = None,
) -> tuple[list[dict], dict]:
    """One record per state: WP3-A shard inputs + teacher action label.

    The shards are consumed AS-IS (per-state centroid-removed X is the
    frozen consume-time translation rule).  Supervision actions are the
    contract-frozen ``label.action_delta`` of the SAME bound teacher table
    the shards were built from (actions are never re-derived from shard X).
    Every shard's sha256 is re-verified against the manifest (identity
    binding).  ``expected_group_ids`` (when given) must match the shard set
    exactly — this is the monitor8 population gate.  ``group_filter`` (when
    given) restricts loading to exactly those reaction groups (fit1/fit5
    canary subsets); it is mutually exclusive with ``expected_group_ids``.
    """
    features_dir = Path(features_dir)
    observed_dir = Path(observed_dir)
    spec_digest, manifest = load_spec_and_manifest(features_dir)
    shard_rows = sorted(manifest["per_reaction"],
                        key=lambda r: r["reaction_group_id"])
    if group_filter is not None:
        keep = {str(g) for g in group_filter}
        shard_rows = [r for r in shard_rows
                      if str(r["reaction_group_id"]) in keep]
    group_ids = [str(r["reaction_group_id"]) for r in shard_rows]
    if expected_group_ids is not None and group_filter is not None:
        raise DualTSBCTrainerError(
            "LOADER_ARGS_CONFLICT",
            "expected_group_ids and group_filter are mutually exclusive")
    if expected_group_ids is not None:
        if sorted(group_ids) != sorted(str(g) for g in expected_group_ids):
            raise DualTSBCTrainerError(
                "POPULATION_MISMATCH",
                f"shard groups {sorted(group_ids)} != expected "
                f"{sorted(str(g) for g in expected_group_ids)}")
    elif group_filter is not None:
        if sorted(group_ids) != sorted(str(g) for g in group_filter):
            raise DualTSBCTrainerError(
                "SUBSET_MISMATCH",
                f"shard groups {sorted(group_ids)} != requested subset "
                f"{sorted(str(g) for g in group_filter)}")
    states: list[dict] = []
    shard_digests: list[str] = []
    n_actions = 0
    max_action_norm = 0.0
    for row in shard_rows:
        path = features_dir / row["npz_file"]
        digest = sha256_file(path)
        if digest != row["npz_sha256"]:
            raise DualTSBCTrainerError(
                "SHARD_DIGEST_MISMATCH",
                f"{path.name}: {digest} != manifest {row['npz_sha256']}")
        shard_digests.append(digest)
        shard = np.load(path)
        teacher = DualTSTeacher.from_payload(json.loads(
            (observed_dir / f"{shard['reaction_id'].item()}.teacher.json"
             ).read_text()))
        labels = {lab.state_id: lab for lab in teacher.labels}
        z = shard["atomic_numbers"]
        node_scalars = shard["node_scalars"]
        x = shard["x_angstrom"]
        g = shard["g_scaled"]
        edge_index = shard["edge_index"]
        atom_a, atom_b = shard["atom_feats_a"], shard["atom_feats_b"]
        edge_a, edge_b = shard["edge_feats_a"], shard["edge_feats_b"]
        stored_mask = shard["near_zero_action_mask"]
        is_teacher = shard["is_teacher_endpoint"]
        for r, sid in enumerate(shard["state_id"]):
            lab = labels[str(sid)]
            action = (None if lab.action_delta is None
                      else np.asarray(lab.action_delta, dtype=np.float64))
            if action is not None:
                n_actions += 1
                norms = np.linalg.norm(action, axis=1)
                max_action_norm = max(max_action_norm, float(norms.max()))
            # Recompute the supervision mask via the SINGLE frozen rule
            # (build_teacher_supervision) and require it to match the stored
            # npz key — supervision semantics are unchanged by v4.1 WP1.
            supervision = build_teacher_supervision(
                action_delta=action,
                is_teacher_endpoint=bool(is_teacher[r]),
                n_atoms=int(z.size))
            if bool(np.any(supervision.near_zero_action_mask
                            != stored_mask[r])):
                raise DualTSBCTrainerError(
                    "MASK_MISMATCH",
                    f"stored near-zero mask disagrees with the teacher "
                    f"action at state {sid}")
            states.append({
                "state_id": str(sid),
                "reaction_group_id": str(shard["reaction_group_id"].item()),
                "branch_id": str(shard["branch_id"][r]),
                "episode_id": str(shard["episode_id"][r]),
                "step_index": int(shard["step_index"][r]),
                "is_teacher": bool(is_teacher[r]),
                "remaining_arc_length": float(lab.remaining_arc_length),
                "action": action,
                "near_zero_mask": np.asarray(stored_mask[r], dtype=bool),
                "atomic_numbers": z,
                "node_scalars": node_scalars[r],
                "x_angstrom": x[r],
                "g_scaled": g[r],
                "atom_feats_a": atom_a,
                "atom_feats_b": atom_b,
                "edge_feats_a": edge_a,
                "edge_feats_b": edge_b,
                "edge_index": edge_index,
            })
    meta = {
        "features_dir": str(features_dir),
        "observed_dir": str(observed_dir),
        "spec_digest": spec_digest,
        "shard_digests": shard_digests,
        "group_ids": group_ids,
        "n_groups": len(group_ids),
        "n_states": len(states),
        "n_action_states": n_actions,
        "n_teacher_states": sum(s["is_teacher"] for s in states),
        "norm_params": manifest["norm_params"],
        "population": manifest["population"],
        "max_action_norm_angstrom": max_action_norm,
    }
    return states, meta


# ---------------------------------------------------------------------------
# §I.10 sampler: group → branch → arc-stratified within branch
# ---------------------------------------------------------------------------

def epoch_state_order(
    states: Sequence[dict],
    *,
    seed: int,
    epoch: int,
    near_ts_arc_fraction: float = NEAR_TS_ARC_FRACTION,
    n_strata: int = N_ARC_STRATA,
) -> tuple[list[dict], dict[str, int]]:
    """One epoch's draw order: every reaction group drawn exactly once.

    Within a group, branches are traversed in seeded-sorted order; within a
    branch, states are arc-stratified into ``n_strata`` equal-count strata
    by remaining-arc-length rank (ascending: the near-TS stratum — smallest
    remaining arc — is emitted FIRST, guaranteeing near-TS coverage even if
    a budget stops mid-epoch; then mid, then far-from-TS), with a seeded
    permutation inside each stratum.  Teacher-endpoint states (no action)
    are never drawn.  Returns the ordered states and the per-group draw
    counts (plan §I.10: 组内抽样数量明确记录).
    """
    rng = np.random.default_rng([int(seed), int(epoch)])
    groups = sorted({s["reaction_group_id"] for s in states})
    order: list[dict] = []
    per_group_counts: dict[str, int] = {}
    for gi in rng.permutation(len(groups)):
        gid = groups[int(gi)]
        gstates = [s for s in states if s["reaction_group_id"] == gid]
        drawable = [s for s in gstates if not s["is_teacher"]]
        if not drawable:
            per_group_counts[gid] = 0
            continue
        branches = sorted({s["branch_id"] for s in drawable})
        branch_order = [branches[i] for i in
                        rng.permutation(len(branches))]
        stream: list[dict] = []
        for bid in branch_order:
            bs = [s for s in drawable if s["branch_id"] == bid]
            ral = np.asarray([s["remaining_arc_length"] for s in bs])
            ranked = np.argsort(ral, kind="stable")
            strata = np.array_split(ranked, max(1, int(n_strata)))
            for stratum in strata:  # near-TS stratum first (see docstring)
                idx = [int(i) for i in stratum]
                perm = rng.permutation(len(idx))
                stream.extend(bs[idx[p]] for p in perm)
        order.extend(stream)
        per_group_counts[gid] = len(stream)
    return order, per_group_counts


# ---------------------------------------------------------------------------
# Targets, per-atom losses, fixed §I.10 reduction
# ---------------------------------------------------------------------------

def latent_targets(numpy, actions: np.ndarray, action_max: float):
    """``u* = R⁻¹(a*/a_max)`` (loud rejection on a_max violations)."""
    z = np.asarray(actions, dtype=np.float64) / float(action_max)
    norms = np.linalg.norm(z, axis=-1, keepdims=True)
    if float(norms.max()) >= 1.0:
        raise DualTSBCTrainerError(
            "A_MAX_VIOLATION",
            f"teacher action at/over a_max: max |a*|/a_max = "
            f"{float(norms.max()):.12f}")
    r = np.arctanh(np.clip(norms, 0.0, 1.0 - 1e-7))
    u = r * z / np.maximum(norms, 1e-12)
    return numpy.asarray(u, dtype=np.float64), z


def build_targets(states: Sequence[dict], action_max: float) -> dict:
    """Per-atom supervision arrays aligned with the collated batch order."""
    torch = require_torch()
    n_atoms = [int(np.asarray(s["atomic_numbers"]).size) for s in states]
    valid, near_zero = [], []
    actions = []
    for s, n in zip(states, n_atoms):
        valid.extend([not s["is_teacher"]] * n)
        near_zero.extend(np.asarray(s["near_zero_mask"], dtype=bool))
        actions.append(np.zeros((n, 3)) if s["action"] is None
                       else np.asarray(s["action"], dtype=np.float64))
    a_star = np.concatenate(actions, axis=0)
    u_star, _ = latent_targets(np, a_star, action_max)
    group_ids = [s["reaction_group_id"] for s in states]
    branch_ids = [s["branch_id"] for s in states]
    state_of_atom = np.repeat(np.arange(len(states), dtype=np.int64),
                              n_atoms)
    cells = sorted(set(zip(group_ids, branch_ids)))
    cell_index = {c: i for i, c in enumerate(cells)}
    groups = sorted(set(group_ids))
    group_index = {g: i for i, g in enumerate(groups)}
    cell_group = np.asarray(
        [group_index[g] for g, _ in cells], dtype=np.int64)
    return {
        "u_star": torch.as_tensor(u_star, dtype=torch.float32),
        "a_star": torch.as_tensor(a_star, dtype=torch.float32),
        "atom_valid": torch.as_tensor(
            np.asarray(valid, dtype=bool)),
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


def per_atom_nll(policy: DualTSPolicy, mu, log_sigma, u_star):
    """``−log π_z(u*|s)`` per atom — the C1 teacher-latent NLL."""
    return -policy.log_prob_z(mu, log_sigma, u_star)["per_atom"]


def per_atom_action_mse(a_det, a_star):
    """``‖a_det − a*‖²`` per atom (L_a building block)."""
    diff = a_det - a_star
    return (diff * diff).sum(dim=-1)


def per_atom_dir_loss(a_det, a_star, eps: float = 1e-12):
    """``‖a*‖·(1 − cos(a_det, a*))`` per atom (|a|-normalized directions,
    magnitude-weighted; near-zero atoms are masked out upstream)."""
    w = a_star.norm(dim=-1)
    na = a_det.norm(dim=-1)
    cos = (a_det * a_star).sum(dim=-1) / (na * w).clamp(min=eps)
    return w * (1.0 - cos.clamp(min=-1.0, max=1.0))


def reduce_step_sums(torch, state_sums, cell_index, n_cells: int,
                     cell_group, n_groups: int):
    """Fixed §I.10 reduction over one training step.

    per-step valid-atom sum (given as ``state_sums``) → branch mean (mean
    within each (group, branch) cell present in the step) → per-reaction
    mean over available branches → group mean over the groups present.
    Returns ``(loss, group_means)`` where ``group_means`` is the (n_groups,)
    vector of per-group values (``nan`` where a group is absent).
    """
    device = state_sums.device
    cell_sum = torch.zeros(n_cells, device=device).index_add_(
        0, cell_index, state_sums)
    cell_count = torch.zeros(n_cells, device=device).index_add_(
        0, cell_index, torch.ones_like(state_sums))
    cell_mean = cell_sum / cell_count.clamp(min=1.0)
    group_sum = torch.zeros(n_groups, device=device).index_add_(
        0, cell_group, cell_mean)
    group_count = torch.zeros(n_groups, device=device).index_add_(
        0, cell_group, (cell_count > 0).to(state_sums.dtype))
    group_mean = group_sum / group_count.clamp(min=1.0)
    present = group_count > 0
    loss = group_mean[present].mean()
    return loss, group_mean


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DualTSBCSettings:
    """One BC run's full hyperparameter identity (C1–C3, §I.10)."""

    recipe: str = "bc_a"
    budget_updates: int = 20000
    batch_states: int = 64
    lr: float = 1e-3
    clip: float = 1.0
    seed: int = 20261017
    lambda_a: float = LAMBDA_A
    lambda_dir: float = LAMBDA_DIR
    action_max: float = DEFAULT_A_MAX_ANGSTROM
    sigma_min: float = DEFAULT_SIGMA_RANGE[0]
    sigma_max: float = DEFAULT_SIGMA_RANGE[1]
    light: bool = True
    signed_mu_gate: bool = False            # C3: default OFF, never silent
    signed_vector_messages: bool = False    # C3: default OFF, never silent
    device: str = "cuda:0"
    require_cuda: bool = True
    near_ts_arc_fraction: float = NEAR_TS_ARC_FRACTION
    n_arc_strata: int = N_ARC_STRATA
    curve_every: int = 100
    eval_every: int = 5000
    prereg_sha256: str = ""   # monitor prereg binding (C4 gate input)
    #: v4.1 protocol_v41 binding: stage name (fit1/fit5/short/full) + protocol
    #: content sha256 + whether this run is a diagnostic (unregistered) budget.
    stage: str = "full"
    protocol_sha256: str = ""
    diagnostic_run: bool = False

    def __post_init__(self) -> None:
        if self.recipe not in RECIPES:
            raise DualTSBCTrainerError(
                "RECIPE_UNKNOWN", f"recipe {self.recipe!r} not in {RECIPES}")
        # v4.1 WP2: λ is no longer hard-frozen at 1.0 (arm B may unfreeze
        # λ_dir).  Registration is enforced by resolve_lambda_registration /
        # the CLI; here we only guard against non-finite / negative values.
        for name, val in (("lambda_a", self.lambda_a),
                          ("lambda_dir", self.lambda_dir)):
            if not np.isfinite(float(val)) or float(val) < 0.0:
                raise DualTSBCTrainerError(
                    "LAMBDA_INVALID",
                    f"{name} must be finite and >= 0, got {val!r}")
        if self.budget_updates <= 0 or self.batch_states <= 0:
            raise DualTSBCTrainerError("BUDGET_INVALID", "non-positive budget")
        if self.sigma_min < 0.02:
            raise DualTSBCTrainerError(
                "SIGMA_FLOOR", "sigma floor below the frozen 0.02")

    def to_payload(self) -> dict:
        return {
            "recipe": self.recipe,
            "budget_updates": self.budget_updates,
            "batch_states": self.batch_states,
            "lr": self.lr, "clip": self.clip, "seed": self.seed,
            "lambda_a": self.lambda_a, "lambda_dir": self.lambda_dir,
            "action_max": self.action_max,
            "sigma_min": self.sigma_min, "sigma_max": self.sigma_max,
            "light": self.light,
            "signed_mu_gate": self.signed_mu_gate,
            "signed_vector_messages": self.signed_vector_messages,
            "device": self.device, "require_cuda": self.require_cuda,
            "near_ts_arc_fraction": self.near_ts_arc_fraction,
            "n_arc_strata": self.n_arc_strata,
            "curve_every": self.curve_every, "eval_every": self.eval_every,
            "density_target_version": DENSITY_TARGET_VERSION,
            "prereg_sha256": self.prereg_sha256,
            "stage": self.stage,
            "protocol_sha256": self.protocol_sha256,
            "diagnostic_run": self.diagnostic_run,
        }


def resolve_lambda_registration(
    protocol: dict,
    *,
    recipe: str,
    lambda_a: float,
    lambda_dir: float,
    stage: str,
    budget_updates: int,
    allow_diagnostic_budget: bool,
    seed: int | None = None,
    features_spec_digest: str | None = None,
    population_group_ids: Sequence[str] | None = None,
    batch_states: int | None = None,
    lr: float | None = None,
    signed_vector_messages: bool = False,
) -> dict:
    """v4.1 WP2 λ/budget/identity registration gate (protocol_v41[_rN].json).

    Returns a decision payload.  A run is ACCEPTED (non-diagnostic) iff:

    * ``stage`` is a registered budget name in the protocol, AND
    * the budget matches the registered value for that stage (fit1/fit5 are
      staged canaries: any updates count is allowed), AND
    * ``(lambda_a, lambda_dir)`` matches a non-null ``lambda_registration``
      slot (arm A = (1.0, 1.0) today; arm B slot is null until a revision
      freezes it).

    FORMAL stages (short/full) additionally bind the whole matrix identity
    (finding 4) AND the arm TUPLE (finding 3/5):

    * ``recipe`` ∈ the frozen A/B/C/D ``arm_mapping`` recipes;
    * the tuple ``(recipe, lambda_a, lambda_dir, signed_vector_messages)``
      must EXACTLY match one ``arm_mapping`` entry (A: bc_nll 1.0/1.0/False;
      B: bc_d 1.0/<arm_b>/False; C: bc_nll 1.0/1.0/True; D: bc_d
      1.0/<arm_b>/True) — so ``bc_d`` with λ_dir=1.0 refuses (arm_a is
      bc_nll);
    * ``seed`` ∈ the protocol ``seeds``;
    * ``features_spec_digest`` == the protocol ``pipeline.spec_digest``;
    * ``population_group_ids`` == the frozen fit50 group set;
    * ``batch_states`` and ``lr`` == the protocol ``matrix_hyperparams``.

    CANARY stages (fit1/fit5) still apply the spec-digest + population checks
    (population == the frozen fit1/fit5 subset) and the loose λ-slot match, but
    ``seed``, updates and matrix values are FREE (they are tiny pre-fit smoke
    runs).

    Otherwise the run is refused unless ``allow_diagnostic_budget``, in which
    case it is accepted as a diagnostic run (excluded from formal selection).
    """
    registered_budgets = protocol.get("registered_budgets", {}) or {}
    slots = ((protocol.get("lambda_registration", {}) or {}).get("slots", {})
             or {})
    arm_mapping = protocol.get("arm_mapping", {}) or {}
    matrix = protocol.get("matrix_hyperparams", {}) or {}
    pipeline = protocol.get("pipeline", {}) or {}
    protocol_seeds = list(protocol.get("seeds", []) or [])
    fit50_ids = list(protocol.get("fit50_group_ids", []) or [])
    subsets = protocol.get("fit1_fit5_subsets", {}) or {}

    formal = stage in ("short", "full")
    canary = stage in ("fit1", "fit5")

    problems: list[str] = []
    if stage not in registered_budgets:
        problems.append(
            f"stage {stage!r} not in registered budgets "
            f"{sorted(registered_budgets)}")
    else:
        reg_updates = registered_budgets[stage].get("updates")
        if reg_updates is not None and int(budget_updates) != int(reg_updates):
            problems.append(
                f"budget {budget_updates} != registered {reg_updates} for "
                f"stage {stage!r}")

    lambda_ok = any(
        slot is not None
        and float(slot.get("lambda_a", -1.0)) == float(lambda_a)
        and float(slot.get("lambda_dir", -1.0)) == float(lambda_dir)
        for slot in slots.values()
    )

    if formal and arm_mapping:
        # finding 3/5: bind the ARM TUPLE — accepted iff (recipe, lambda_a,
        # lambda_dir, signed_vector_messages) EXACTLY matches one arm_mapping
        # entry.  bc_d with λ_dir=1.0 refuses (arm_a is bc_nll).
        tuple_match = any(
            str(cfg.get("recipe")) == str(recipe)
            and float(cfg.get("lambda_a", -1.0)) == float(lambda_a)
            and cfg.get("lambda_dir") is not None
            and float(cfg.get("lambda_dir")) == float(lambda_dir)
            and bool(cfg.get("signed_vector_messages", False))
            == bool(signed_vector_messages)
            for cfg in arm_mapping.values()
        )
        if not tuple_match:
            problems.append(
                f"(recipe={recipe}, lambda_a={lambda_a}, "
                f"lambda_dir={lambda_dir}, "
                f"signed_vector_messages={signed_vector_messages}) does not "
                f"match any frozen arm_mapping entry for a formal run")
    elif not lambda_ok:
        problems.append(
            f"lambda ({lambda_a}, {lambda_dir}) not a registered slot")

    if formal and protocol_seeds and seed is not None:
        if int(seed) not in [int(s) for s in protocol_seeds]:
            problems.append(
                f"seed {seed} not in protocol seeds {protocol_seeds}")

    protocol_spec = str(pipeline.get("spec_digest", ""))
    if protocol_spec and features_spec_digest is not None:
        if str(features_spec_digest) != protocol_spec:
            problems.append(
                f"features spec digest {str(features_spec_digest)[:12]}… != "
                f"protocol pipeline {protocol_spec[:12]}…")

    if population_group_ids is not None:
        pop = sorted(str(g) for g in population_group_ids)
        if formal:
            if fit50_ids and pop != sorted(str(g) for g in fit50_ids):
                problems.append(
                    "population group set != frozen fit50 set")
        elif canary:
            ref = subsets.get(f"{stage}_group_ids")
            if ref and pop != sorted(str(g) for g in ref):
                problems.append(
                    f"population group set != frozen {stage} subset")

    if formal and matrix:
        if batch_states is not None and \
                int(batch_states) != int(matrix.get("batch_states", -1)):
            problems.append(
                f"batch_states {batch_states} != matrix "
                f"{matrix.get('batch_states')}")
        if lr is not None and float(lr) != float(matrix.get("lr", -1.0)):
            problems.append(
                f"lr {lr} != matrix {matrix.get('lr')}")

    accepted = not problems
    if accepted:
        return {"accepted": True, "diagnostic": False, "problems": []}
    if allow_diagnostic_budget:
        return {"accepted": True, "diagnostic": True, "problems": problems}
    return {"accepted": False, "diagnostic": False, "problems": problems}


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DualTSBCTrainer:
    """Teacher-latent BC over the dual-TS policy (one recipe per run)."""

    def __init__(
        self,
        *,
        settings: DualTSBCSettings,
        states: Sequence[dict],
        data_meta: dict,
        run_dir: str | Path,
    ) -> None:
        self.settings = settings
        self.states = list(states)
        self.data_meta = dict(data_meta)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.drawable = [s for s in self.states if not s["is_teacher"]]
        _require(bool(self.drawable), "EMPTY_ACTION_POPULATION",
                 "no non-teacher states to draw")
        self.torch = require_torch()
        from pes2ts_core.pesn.offline.training.device import (
            resolve_device, seed_everything, set_cpu_threads)
        self.torch, self.device = resolve_device(
            device=settings.device, require_cuda=settings.require_cuda)
        set_cpu_threads(intra=2, inter=1)
        seed_everything(settings.seed)
        self.policy = DualTSPolicy(
            config=DualTSModelConfig(
                light=settings.light,
                signed_vector_messages=settings.signed_vector_messages),
            action_max=settings.action_max,
            sigma_range=(settings.sigma_min, settings.sigma_max),
            signed_mu_gate=settings.signed_mu_gate,
        ).to(self.torch.float32).to(self.device)
        self.optimizer = self.torch.optim.Adam(
            self.policy.parameters(), lr=settings.lr)
        # -- §I.10 unified sampler + exposure accounting (P06 / F5) ----------
        # The SAME DualTSGroupBranchSampler surface as IQL; same seed + same
        # dataset => identical draw sequence.
        self._sampler_groups = build_sampler_groups(
            self.drawable,
            group_key=lambda s: s["reaction_group_id"],
            branch_key=lambda s: s["branch_id"],
            arc_key=lambda s, _i: s["remaining_arc_length"],
        )
        self.sampler = DualTSGroupBranchSampler(
            seed=settings.seed, groups=self._sampler_groups,
            rows_per_group=1)
        self.exposure = ExposureTracker(row_meta=[
            {
                "group_id": s["reaction_group_id"],
                "state_id": s["state_id"],
                "branch_id": s["branch_id"],
                "remaining_arc_length": s["remaining_arc_length"],
            }
            for s in self.drawable
        ])
        self.history: list[dict] = []
        self.diagnostics: list[dict] = []
        self.updates_run = 0
        self.samples_exposed = 0
        self.nll_min = math.inf
        self.nll_max = -math.inf
        self._last_checkpoint_digest = ""

    # -- batching -----------------------------------------------------------

    def _batch(self, chunk: Sequence[dict]) -> dict:
        torch = self.torch
        batch = collate_dual_states(torch, list(chunk))
        for key, val in batch.items():
            if torch.is_tensor(val):
                if val.dtype in (torch.float64, torch.float32):
                    val = val.to(dtype=torch.float32)
                batch[key] = val.to(self.device)
        targets = build_targets(chunk, self.settings.action_max)
        for key in ("u_star", "a_star", "atom_valid", "dir_mask",
                    "cell_index", "cell_group", "state_of_atom"):
            targets[key] = targets[key].to(self.device)
        batch.update(targets)
        return batch

    # -- loss ---------------------------------------------------------------

    def compute_losses(self, batch: dict) -> dict:
        """Recipe loss with the fixed §I.10 reduction (C1 + C2)."""
        st = self.settings
        out = self.policy(batch)
        mu, log_sigma = out["mu"], out["log_sigma"]
        valid = batch["atom_valid"]
        state_of_atom = batch["state_of_atom"]
        nll_atom = per_atom_nll(self.policy, mu, log_sigma,
                                batch["u_star"])
        a_det = self.policy.deterministic(mu)
        mse_atom = per_atom_action_mse(a_det, batch["a_star"])
        dir_atom = per_atom_dir_loss(a_det, batch["a_star"])
        n_states = batch["n_states"]
        torch = self.torch
        nll_state = torch.zeros(n_states, device=mu.device).index_add_(
            0, state_of_atom, nll_atom.masked_fill(~valid, 0.0))
        mse_state = torch.zeros(n_states, device=mu.device).index_add_(
            0, state_of_atom, mse_atom.masked_fill(~valid, 0.0))
        dir_state = torch.zeros(n_states, device=mu.device).index_add_(
            0, state_of_atom,
            dir_atom.masked_fill(~batch["dir_mask"], 0.0))
        nll_loss, nll_group = reduce_step_sums(
            self.torch, nll_state, batch["cell_index"], batch["n_cells"],
            batch["cell_group"], batch["n_groups"])
        mse_loss, _ = reduce_step_sums(
            self.torch, mse_state, batch["cell_index"], batch["n_cells"],
            batch["cell_group"], batch["n_groups"])
        dir_loss, _ = reduce_step_sums(
            self.torch, dir_state, batch["cell_index"], batch["n_cells"],
            batch["cell_group"], batch["n_groups"])
        if st.recipe == "bc_nll":
            loss = nll_loss
        elif st.recipe == "bc_a":
            loss = nll_loss + st.lambda_a * mse_loss
        else:  # bc_d
            loss = nll_loss + st.lambda_a * mse_loss \
                + st.lambda_dir * dir_loss
        return {
            "loss": loss,
            "nll_loss": nll_loss.detach(),
            "mse_loss": mse_loss.detach(),
            "dir_loss": dir_loss.detach(),
            "nll_group": nll_group.detach(),
            "nll_per_atom": float(
                nll_atom.detach().masked_fill(~valid, 0.0).sum()
                / valid.to(mu.dtype).sum().clamp(min=1.0)),
            "mu_norm_mean": float(mu.detach().norm(dim=-1).mean()),
            "sigma_mean": float(log_sigma.detach().exp().mean()),
        }

    def _state_index(self, batch: dict):
        return batch["state_of_atom"]

    # -- training -----------------------------------------------------------

    def train(self) -> dict:
        torch = self.torch
        st = self.settings
        started = time.perf_counter()
        update = 0
        epoch = 0
        n_drawable = len(self.drawable)
        running = {"n": 0, "loss": 0.0, "nll": 0.0, "mse": 0.0, "dir": 0.0}
        while update < st.budget_updates:
            rows = self.sampler.draw_batch(st.batch_states)
            self.exposure.record_batch(rows)
            chunk = [self.drawable[i] for i in rows]
            batch = self._batch(chunk)
            self.policy.train_mode(True)
            parts = self.compute_losses(batch)
            self.optimizer.zero_grad(set_to_none=True)
            parts["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), st.clip)
            self.optimizer.step()
            update += 1
            self.updates_run = update
            self.samples_exposed += len(chunk)
            vals = {}
            for k in ("loss", "nll_loss", "mse_loss", "dir_loss"):
                t_ = parts[k]
                vals[k] = float(t_.detach() if t_.requires_grad else t_)
            if not all(np.isfinite(list(vals.values()))):
                raise DualTSBCTrainerError(
                    "NONFINITE_LOSS", f"non-finite loss at update "
                    f"{update}: {vals}")
            if vals["nll_loss"] > NLL_DIVERGENCE_LIMIT:
                raise DualTSBCTrainerError(
                    "NLL_DIVERGENCE",
                    f"teacher-latent NLL diverged at update {update}: "
                    f"{vals['nll_loss']:.4e} > {NLL_DIVERGENCE_LIMIT:.1e}")
            self.nll_min = min(self.nll_min, vals["nll_loss"])
            self.nll_max = max(self.nll_max, vals["nll_loss"])
            running["n"] += 1
            for k, key in (("loss", "loss"), ("nll", "nll_loss"),
                           ("mse", "mse_loss"), ("dir", "dir_loss")):
                running[k] += vals[key]
            # epoch = one full dataset exposure (every drawable state drawn
            # once on average): draws // n_drawable.  Snapshot per epoch.
            new_epoch = self.sampler.draws // n_drawable
            while new_epoch > epoch:
                epoch += 1
                self.exposure.snapshot(kind="epoch", update=update,
                                       epoch=epoch)
            if update % 1000 == 0:
                self.exposure.snapshot(kind="update_1000", update=update,
                                       epoch=epoch)
            if update % st.curve_every == 0 or update == st.budget_updates:
                n = max(running["n"], 1)
                self.history.append({
                    "update": update,
                    "epoch": epoch,
                    "loss": running["loss"] / n,
                    "nll_loss": running["nll"] / n,
                    "mse_loss": running["mse"] / n,
                    "dir_loss": running["dir"] / n,
                    "nll_per_atom_running": parts["nll_per_atom"],
                    "mu_norm_mean": parts["mu_norm_mean"],
                    "sigma_mean": parts["sigma_mean"],
                    "samples_exposed": self.samples_exposed,
                    "seconds": round(time.perf_counter() - started, 2),
                })
                running = {"n": 0, "loss": 0.0, "nll": 0.0,
                           "mse": 0.0, "dir": 0.0}
            if st.eval_every and update % st.eval_every == 0:
                self.diagnostics.append({
                    "update": update,
                    **self.fit_diagnostics(tag=f"u{update}"),
                })
        self.exposure.snapshot(kind="final", update=update, epoch=epoch)
        final = self.fit_diagnostics(tag="final")
        self.save_checkpoint("checkpoint_last.pt")
        record = {
            "schema_version": "pesn_dual_ts_bc_run_v2",
            "trainer_version": DUAL_TS_BC_VERSION,
            "recipe": st.recipe,
            "settings": st.to_payload(),
            "data": {k: v for k, v in self.data_meta.items()},
            "accounting": {
                "updates_run": self.updates_run,
                "batch_states": st.batch_states,
                "samples_exposed": self.samples_exposed,
                "sampler_seed": st.seed,
                "sampler_rule": GROUP_BRANCH_SAMPLING_RULE,
                "within_branch_rule": WITHIN_BRANCH_RULE,
                "drawable_states": n_drawable,
                "n_groups": self.sampler.n_groups,
                "rounds_completed": self.sampler.rounds_completed,
                "epoch_definition": "one full dataset exposure: every "
                                    "drawable state drawn once on average "
                                    "(draws // drawable_states); unified "
                                    "group->branch round-robin sampler "
                                    "(shared with IQL)",
                "nll_min_max_per_state": [self.nll_min, self.nll_max],
                "nll_divergence_limit": NLL_DIVERGENCE_LIMIT,
            },
            "exposure": self.exposure.payload(),
            "history": self.history,
            "diagnostics": self.diagnostics,
            "fit_diagnostics_final": final,
            "wall_seconds": round(time.perf_counter() - started, 2),
        }
        write_json(self.run_dir / "run_record.json", record)
        return record

    # -- fit-partition diagnostics (NOT monitor metrics) --------------------

    def fit_diagnostics(self, *, tag: str, batch_size: int = 256) -> dict:
        """Full-fit-population train-side diagnostics (clearly labeled; the
        selection rule NEVER consumes these — monitor8 only, §I.9)."""
        torch = self.torch
        self.policy.train_mode(False)
        mu_all, sigma_all = [], []
        with torch.no_grad():
            for i in range(0, len(self.states), batch_size):
                batch = self._batch(self.states[i:i + batch_size])
                out = self.policy(batch)
                mu_all.append(out["mu"].cpu())
                sigma_all.append(out["sigma"].cpu())
            mu = torch.cat(mu_all, dim=0)
            sigma = torch.cat(sigma_all, dim=0)
            targets = build_targets(self.states, self.settings.action_max)
            a_det = self.settings.action_max * radial_transform_stable(
                torch, mu)
            valid = targets["atom_valid"]
            mse = float(per_atom_action_mse(a_det, targets["a_star"])
                        [valid].mean())
            zero = float((targets["a_star"] ** 2).sum(-1)[valid].mean())
            dmask = targets["dir_mask"]
            if bool(dmask.any()):
                ad = a_det[dmask]
                ast = targets["a_star"][dmask]
                w = ast.norm(dim=-1)
                cos = (ad * ast).sum(-1) / (
                    ad.norm(dim=-1) * w).clamp(min=1e-12)
                dir_cos = float((w * cos).sum() / w.sum())
            else:
                dir_cos = None
            nll_atom = per_atom_nll(
                self.policy, mu.to(self.device),
                torch.log(sigma.to(self.device).clamp(min=1e-8)),
                targets["u_star"].to(self.device))
            nll_pa = float(nll_atom.sum()
                           / valid.to(mu.dtype).sum().clamp(min=1.0))
        return {
            "tag": tag,
            "population": "fit_partition_train_side_diagnostic_only",
            "per_atom_action_mse_A2": mse,
            "zero_baseline_action_mse_A2": zero,
            "mse_over_baseline": mse / zero,
            "direction_cosine_absa_weighted": dir_cos,
            "nll_per_atom": nll_pa,
            "sigma_mean": float(sigma.mean()),
            "sigma_min_observed": float(sigma.min()),
        }

    # -- checkpoint + bundle ------------------------------------------------

    def checkpoint_identity(self) -> dict:
        """The v2 checkpoint identity block (verified verbatim by IQL)."""
        st = self.settings
        return {
            "features_spec_digest": self.data_meta["spec_digest"],
            "action_version": ACTION_SCHEMA_VERSION,
            "seed": st.seed,
            "recipe": st.recipe,
            "lambda_a": st.lambda_a,
            "lambda_dir": st.lambda_dir,
            "sampler_config": self.sampler_config(),
            "protocol_sha256": st.protocol_sha256,
            "density_target_version": DENSITY_TARGET_VERSION,
        }

    def sampler_config(self) -> dict:
        return {
            "seed": self.sampler.seed,
            "rows_per_group": self.sampler.rows_per_group,
            "n_groups": self.sampler.n_groups,
            "sampling_rule": self.sampler.sampling_rule,
            "within_branch_rule": self.sampler.within_branch_rule,
        }

    def checkpoint_payload_meta(self) -> dict:
        st = self.settings
        return {
            "trainer_version": DUAL_TS_BC_VERSION,
            "recipe": st.recipe,
            "settings": st.to_payload(),
            "density_target_version": DENSITY_TARGET_VERSION,
            "model": DUAL_TS_EGNN_VERSION,
            "policy": DUAL_TS_POLICY_VERSION,
            "spec_digest": self.data_meta["spec_digest"],
            "shard_digests_sha256": self.data_meta["shard_digests"],
            "signed_mu_gate": st.signed_mu_gate,
            "signed_vector_messages": st.signed_vector_messages,
            "prereg_sha256": st.prereg_sha256,
        }

    def save_checkpoint(self, name: str) -> Path:
        """Write the BC checkpoint in BOTH frozen formats.

        * ``checkpoint_last.pt`` — ``pesn_checkpoint_v1`` (the generic writer
          kept for compat): the free-run ``load_selected_policy`` path reads
          this format (WP4-B migrates it to v2 separately).
        * ``checkpoint_v2.pt`` — ``pesn_dual_ts_bc_checkpoint_v2`` with the
          identity block + real ``content_digest``: the format IQL
          ``load_bc_actor`` accepts (v4.1 WP2 / requirement 7).
        """
        v1_path = self.run_dir / "checkpoint_last.pt"
        save_checkpoint(
            v1_path,
            model=self.policy,
            optimizer=self.optimizer,
            epoch=self.updates_run,
            best_metric=self.nll_min,
            config_sha256=self.data_meta["spec_digest"],
            dataset_manifest_sha256=self.data_meta["spec_digest"],
            extra=self.checkpoint_payload_meta(),
        )
        v2_path = self.run_dir / "checkpoint_v2.pt"
        self._last_checkpoint_digest = save_bc_checkpoint_v2(
            v2_path,
            policy=self.policy,
            optimizer=self.optimizer,
            epoch=self.updates_run,
            best_metric=self.nll_min,
            identity=self.checkpoint_identity(),
            sampler_state=self.sampler.state_dict(),
            extra=self.checkpoint_payload_meta(),
        )
        return v1_path

    def build_bundle(self, record: dict) -> DualTSBundle:
        st = self.settings
        ckpt = self.run_dir / "checkpoint_last.pt"
        npv = self.data_meta["norm_params"]
        s_delta_e = npv.get("s_delta_e_hartree", npv.get("s_delta_e"))
        bundle = DualTSBundle(
            run_id=(
                f"pesn_dual_ts_wp4a_bc_{st.recipe}_seed{st.seed}"
                + ("_signedmu" if st.signed_mu_gate else "")
                + ("_signedvecmsg" if st.signed_vector_messages else "")
                + "_v1"),
            # NOTE: freerun binds the checkpoint by FILE sha256 (load_selected_
            # policy); the checkpoint's own content_digest is carried inside
            # the checkpoint identity and verified by IQL load_bc_actor.
            bc_parent_checkpoint_digest=sha256_file(ckpt),
            seeds={"torch": st.seed, "numpy": st.seed, "python": st.seed},
            hyperparams={
                **st.to_payload(),
                "loss_formula": {
                    "bc_nll": "L = L_NLL",
                    "bc_a": "L = L_NLL + 1.0 * L_a",
                    "bc_d": "L = L_NLL + 1.0 * L_a + 1.0 * L_dir",
                }[st.recipe],
                "nll_definition": "teacher-latent -log pi_z(u*|s), "
                                  "u* = R^-1(a*/a_max) (G4 C1)",
                "reduction": "per-step valid-atom sum -> branch mean -> "
                             "per-reaction mean over available branches -> "
                             "group mean (plan v4 §I.10, fixed)",
                "sampler": "unified DualTSGroupBranchSampler shared with IQL; "
                           "group round-robin (each group once per round), "
                           "branch uniform RNG, 3-bin arc round-robin; "
                           "epoch = one full dataset exposure",
                "accounting": record["accounting"],
                "updates_run": self.updates_run,
            },
            dataset_version=(
                f"pesn_dual_ts_features_v3@{self.data_meta['spec_digest']}"),
            norm_version=(
                "dual_ts_frozen_scales_v0:"
                f"s_g={npv['s_g_hartree_per_angstrom']}_Ha/A:"
                f"s_delta_e={s_delta_e}_Ha"),
            split_version="dual_ts_split_lineage_v1:wave=fit50",
            action_version=(
                f"{DENSITY_TARGET_VERSION}:a_max={st.action_max}:"
                f"sigma_min={st.sigma_min}"),
            environment_version="pesn_dual_ts_env_v2:not_exercised_in_wp4a_bc",
            reward_version="pesn_dual_ts_reward_v2:not_exercised_in_wp4a_bc",
            input_eligibility={
                "spec_digest": self.data_meta["spec_digest"],
                "shard_count": len(self.data_meta["shard_digests"]),
                "shard_digests_sha256": self.data_meta["shard_digests"],
                "action_labels": "contract label.action_delta of the SAME "
                                 "bound wp2_fit50_observed teacher table",
                "near_zero_mask_rule": (
                    "frozen 0.01 A mask excludes atoms from L_dir only; "
                    "they stay in L_a and L_NLL"),
                "teacher_endpoint_rule": (
                    "teacher_endpoint_states never drawn (no action)"),
                "fresh_init": "no WP3 checkpoint warm start",
            },
            output_eligibility={
                "purpose": "wp4-a BC recipe run (plan v4 §D WP4 row); "
                           "selection is monitor8-only (blocked pending "
                           "monitor8 observations; see wp4-bc report)",
                "monitor_selection": "NOT performed on fit metrics",
                "c3_signed_variant": st.signed_mu_gate,
                "beats_zero_baseline_fit_side": (
                    record["fit_diagnostics_final"]["mse_over_baseline"] < 1.0),
            },
            production_decision_open=False,
            schema_version=BUNDLE_SCHEMA_VERSION,
        )
        bundle.validate()
        return bundle


__all__ = [
    "DENSITY_TARGET_VERSION",
    "DUAL_TS_BC_VERSION",
    "DualTSBCTrainer",
    "DualTSBCTrainerError",
    "DualTSBCSettings",
    "LAMBDA_A",
    "LAMBDA_DIR",
    "NEAR_TS_ARC_FRACTION",
    "NLL_DIVERGENCE_LIMIT",
    "RECIPES",
    "build_targets",
    "epoch_state_order",
    "latent_targets",
    "load_dual_ts_states",
    "load_spec_and_manifest",
    "per_atom_action_mse",
    "per_atom_dir_loss",
    "per_atom_nll",
    "reduce_step_sums",
    "resolve_lambda_registration",
    "sha256_file",
]
