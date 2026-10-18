#!/usr/bin/env python3
"""Shared zero-QC helpers for the WP3 v4.1 diagnostics suite.

This module is NOT a deliverable on its own — it is imported by the
``diagnose_v41_*.py`` scripts so the five old-checkpoint arms, the OLD
7-channel feature loader, the NEW 5-channel loader, the frozen metric
definitions, and the per-sample identity check all come from ONE place
(no drift between items).

Read-only analysis: nothing here invokes ACP/xtb/observer compute; it only
reads the already-frozen shards, teacher tables, checkpoints, and lineage
artifacts.

IMPORTANT LAYOUT FACTS (v4.1 WP1):
  * OLD checkpoints + OLD shards (``wp3_features`` / ``wp4_monitor/features``)
    are 7-channel node_scalars.  The OLD model's ``node_proj`` input dim is
    ``2*d_model + 7 = 71`` (verified from ``checkpoint_last.pt``).
  * The current ``dual_egnn`` module builds against ``N_NODE_SCALARS`` (now 5)
    and its consume-time width guard rejects any other width.  To load OLD
    checkpoints we patch ``dual_egnn.N_NODE_SCALARS = 7`` BEFORE building the
    policy and before collating OLD shards, then restore it.  Nothing here
    modifies any checked-in source file.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

import pes2ts_core.pesn.models.dual_egnn as _de  # noqa: E402
from pes2ts_core.pesn.features.dual_ts_spec import (  # noqa: E402
    BOND_SOURCE_COVALENT_FALLBACK,
)
from pes2ts_core.pesn.models.dual_egnn import (  # noqa: E402
    DualTSModelConfig,
    collate_dual_states,
)
from pes2ts_core.pesn.models.dual_ts_policy import (  # noqa: E402
    DEFAULT_A_MAX_ANGSTROM,
    DualTSPolicy,
)
from pes2ts_core.pesn.offline.training.dual_ts_bc import (  # noqa: E402
    build_targets,
    load_dual_ts_states,
    per_atom_action_mse,
    per_atom_dir_loss,
    per_atom_nll,
    reduce_step_sums,
)

BASE = PROJECT_ROOT / "outputs/pesn/dual_ts_v2_20261015"
OLD_FIT_FEATURES = BASE / "wp3_features"
OLD_MON_FEATURES = BASE / "wp4_monitor" / "features"
OBSERVED_FIT = BASE / "wp2_fit50_observed"
OBSERVED_MON = BASE / "wp4_monitor" / "teachers_observed"
NEW_FIT_FEATURES = BASE / "wp3_features_v41"
NEW_MON_FEATURES = BASE / "wp4_monitor" / "features_v41"
WP4_BC = BASE / "wp4_bc"
OUT_DIR = BASE / "wp41_diagnostics"
LINEAGE_PATH = BASE / "splits" / "dual_ts_split_lineage_v1.json"
GAPS_PATH = BASE / "wp2_fit50" / "gaps_registry.json"
G5_SLICE_PATH = WP4_BC / "g5_diagnosis_slice.json"

ACTION_MAX = float(DEFAULT_A_MAX_ANGSTROM)  # 0.1 A

#: The 5 OLD arms, in canonical order.  recipe/flags read from each bundle.
ARMS: tuple[dict, ...] = (
    {"name": "bc_nll", "run_dir": "run_bc_bc_nll_seed20261017",
     "recipe": "bc_nll", "signed_mu_gate": False, "signed_vector_messages": False},
    {"name": "bc_a", "run_dir": "run_bc_bc_a_seed20261017",
     "recipe": "bc_a", "signed_mu_gate": False, "signed_vector_messages": False},
    {"name": "bc_d", "run_dir": "run_bc_bc_d_seed20261017",
     "recipe": "bc_d", "signed_mu_gate": False, "signed_vector_messages": False},
    {"name": "bc_nll_signedmu", "run_dir": "run_bc_bc_nll_seed20261017_signedmu",
     "recipe": "bc_nll", "signed_mu_gate": True, "signed_vector_messages": False},
    {"name": "bc_nll_signedvecmsg", "run_dir": "run_bc_bc_nll_seed20261017_signedvecmsg",
     "recipe": "bc_nll", "signed_mu_gate": False, "signed_vector_messages": True},
)

#: Frozen disputed-5 reaction ids (task ``eligibility.bond_authority ==
#: "disputed"`` -> both endpoint graphs consumed via covalent-radius fallback).
DISPUTED_IDS: tuple[str, ...] = (
    "RXN_0000000160", "RXN_0000000756", "RXN_0000000778",
    "RXN_0000001076", "RXN_0000001174",
)


# ---------------------------------------------------------------------------
# Layout patching (OLD 7-ch vs NEW 5-ch)
# ---------------------------------------------------------------------------

def set_old_layout() -> None:
    """Patch ``dual_egnn.N_NODE_SCALARS`` to 7 for OLD checkpoint loading."""
    _de.N_NODE_SCALARS = 7


def set_new_layout() -> None:
    """Restore ``dual_egnn.N_NODE_SCALARS`` to the current (5) frozen layout."""
    from pes2ts_core.pesn.features.dual_ts_graph import N_NODE_SCALARS
    _de.N_NODE_SCALARS = int(N_NODE_SCALARS)


def device_of() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def write_json(path: Path, obj) -> None:
    """Deterministic JSON write (sorted keys, no NaN, compact indent)."""
    text = json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False,
                      allow_nan=False)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Arm checkpoint loading
# ---------------------------------------------------------------------------

def arm_checkpoint(arm: dict) -> Path:
    return WP4_BC / arm["run_dir"] / "checkpoint_last.pt"


def load_old_policy(arm: dict):
    """Build the OLD (7-ch) policy and load the arm checkpoint into it.

    Leaves ``dual_egnn.N_NODE_SCALARS == 7`` for the rest of the process so
    that ``collate_dual_states`` (and its consume-time width guard) accept the
    7-channel OLD shards.  Callers that also need the NEW 5-ch layout in the
    same process must call :func:`set_new_layout` explicitly.
    """
    set_old_layout()
    ckpt = torch.load(arm_checkpoint(arm), map_location="cpu",
                      weights_only=False)
    policy = DualTSPolicy(
        config=DualTSModelConfig(
            light=True, signed_vector_messages=arm["signed_vector_messages"]),
        action_max=ACTION_MAX,
        sigma_range=(0.02, 1.0),
        signed_mu_gate=arm["signed_mu_gate"],
    )
    policy.load_state_dicts(ckpt["model_state"])
    policy.to(torch.float32)
    policy.to(device_of())
    policy.train_mode(False)
    return policy, ckpt


# ---------------------------------------------------------------------------
# State loading (OLD 7-ch and NEW 5-ch)
# ---------------------------------------------------------------------------

def load_old_states(features_dir: Path, observed_dir: Path):
    """Load OLD 7-ch states (patched layout), verifying digest + masks.

    Leaves the OLD 7-channel layout active (see :func:`load_old_policy`).
    """
    set_old_layout()
    return load_dual_ts_states(features_dir, observed_dir)


def load_new_states(features_dir: Path, observed_dir: Path):
    """Load NEW 5-ch states (current layout)."""
    set_new_layout()
    return load_dual_ts_states(features_dir, observed_dir)


# ---------------------------------------------------------------------------
# Per-atom alignment (torch targets -> numpy)
# ---------------------------------------------------------------------------

def aligned(states, action_max: float = ACTION_MAX):
    """Aligned numpy arrays for a collated state list (order == collate order).

    Returns a dict with per-atom arrays: ``a_star`` (N,3), ``u_star`` (N,3),
    ``valid`` (N,), ``dir_mask`` (N,), ``group`` (N,), ``branch`` (N,),
    ``state_idx`` (N,), ``ral`` (N,), ``is_teacher`` (N,), ``step`` (N,),
    plus the per-state ``n_atoms`` list.
    """
    n_atoms = [int(np.asarray(s["atomic_numbers"]).size) for s in states]
    tgt = build_targets(list(states), action_max)
    a_star = tgt["a_star"].numpy().astype(np.float64)
    u_star = tgt["u_star"].numpy().astype(np.float64)
    valid = tgt["atom_valid"].numpy().astype(bool)
    dir_mask = tgt["dir_mask"].numpy().astype(bool)
    group = np.repeat(np.asarray([s["reaction_group_id"] for s in states],
                                 dtype=object), n_atoms)
    branch = np.repeat(np.asarray([s["branch_id"] for s in states],
                                  dtype=object), n_atoms)
    state_idx = np.repeat(np.arange(len(states), dtype=np.int64), n_atoms)
    ral = np.repeat(np.asarray([float(s["remaining_arc_length"]) for s in states],
                               dtype=np.float64), n_atoms)
    is_teacher = np.repeat(np.asarray([bool(s["is_teacher"]) for s in states],
                                      dtype=bool), n_atoms)
    step = np.repeat(np.asarray([int(s["step_index"]) for s in states],
                                dtype=np.int64), n_atoms)
    return {
        "a_star": a_star, "u_star": u_star, "valid": valid,
        "dir_mask": dir_mask, "group": group, "branch": branch,
        "state_idx": state_idx, "ral": ral, "is_teacher": is_teacher,
        "step": step, "n_atoms": n_atoms,
    }


# ---------------------------------------------------------------------------
# Deterministic forward (no gradients)
# ---------------------------------------------------------------------------

def forward_det(policy, states, batch_size: int = 256, action_max: float = ACTION_MAX):
    """Deterministic per-atom action ``a_det = a_max*tanh(||mu||)mu/||mu||``.

    Returns ``(a_det, mu, sigma)`` numpy arrays aligned with collate order.
    """
    torch_mod = policy.torch
    mus, sigmas = [], []
    with torch_mod.no_grad():
        for i in range(0, len(states), batch_size):
            chunk = list(states[i:i + batch_size])
            batch = collate_dual_states(torch_mod, chunk)
            for k, v in list(batch.items()):
                if torch_mod.is_tensor(v) and v.dtype in (torch.float64, torch.float32):
                    batch[k] = v.to(dtype=torch.float32)
                if torch_mod.is_tensor(v):
                    batch[k] = v.to(policy.module.encoder.node_proj.weight.device)
            out = policy(batch)
            mu = out["mu"]
            a_det = policy.deterministic(mu)
            mus.append(a_det.detach().cpu().numpy().astype(np.float64))
            sigmas.append(out["sigma"].detach().cpu().numpy().astype(np.float64))
    a_det = np.concatenate(mus, axis=0) if mus else np.zeros((0, 3))
    sigma = np.concatenate(sigmas, axis=0) if sigmas else np.zeros((0,))
    return a_det, sigma


# ---------------------------------------------------------------------------
# Metric primitives (frozen definitions mirroring dual_ts_monitor)
# ---------------------------------------------------------------------------

def atom_mse(a_det, a_star):
    return ((a_det - a_star) ** 2).sum(-1)


def atom_cos(a_det, a_star, eps: float = 1e-12):
    na = np.linalg.norm(a_det, axis=1)
    ns = np.linalg.norm(a_star, axis=1)
    return (a_det * a_star).sum(1) / np.maximum(na * ns, eps)


def atom_alpha(a_det, a_star, eps: float = 1e-12):
    return np.linalg.norm(a_det, axis=1) / np.maximum(
        np.linalg.norm(a_star, axis=1), eps)


def dir_cosine_weighted(a_det, a_star, dmask):
    """|a|-weighted direction cosine over ``dmask`` (monitor definition)."""
    if not bool(dmask.any()):
        return None
    ad, ast = a_det[dmask], a_star[dmask]
    w = np.linalg.norm(ast, axis=1)
    cos = (ad * ast).sum(1) / np.maximum(
        np.linalg.norm(ad, axis=1) * w, 1e-12)
    return float((w * cos).sum() / w.sum())


def mse_ratio_vs_zero(a_det, a_star, mask):
    """mean(||a_det-a*||^2)/mean(||a*||^2) over ``mask`` (monitor definition)."""
    if not bool(mask.any()):
        return None
    mse_model = float(atom_mse(a_det, a_star)[mask].mean())
    mse_zero = float((a_star[mask] ** 2).sum(-1).mean())
    return mse_model / mse_zero if mse_zero > 0 else None


def percentile_dict(values, name: str) -> dict:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size), "min": float(v.min()),
        "p05": float(np.percentile(v, 5)),
        "q25": float(np.percentile(v, 25)),
        "median": float(np.median(v)),
        "mean": float(v.mean()),
        "q75": float(np.percentile(v, 75)),
        "p95": float(np.percentile(v, 95)),
        "max": float(v.max()),
    }


def med_iqr(values) -> dict:
    v = np.asarray([float(x) for x in values if x is not None
                    and math.isfinite(float(x))], dtype=np.float64)
    if v.size == 0:
        return {"median": None, "q25": None, "q75": None, "n": 0}
    return {"median": float(np.median(v)),
            "q25": float(np.percentile(v, 25)),
            "q75": float(np.percentile(v, 75)), "n": int(v.size)}


# ---------------------------------------------------------------------------
# Per-sample identity check (F9: replaces invalid group-level back-inference)
# ---------------------------------------------------------------------------

def per_sample_identity(a_det, a_star, dmask):
    """Per-atom identity ``||a_det-a*||^2/||a*||^2 == 1 + alpha^2 - 2*alpha*cos``.

    Returns the per-atom residual (actual - reconstructed) over ``dmask``.
    """
    alpha = atom_alpha(a_det, a_star)
    cos = atom_cos(a_det, a_star)
    r_norm = atom_mse(a_det, a_star) / np.maximum(
        np.linalg.norm(a_star, axis=1) ** 2, 1e-30)
    recon = 1.0 + alpha ** 2 - 2.0 * alpha * cos
    resid = r_norm - recon
    return resid[dmask], alpha[dmask], cos[dmask]


# ---------------------------------------------------------------------------
# Linear-span projection (item 4 vector-expressiveness probe)
# ---------------------------------------------------------------------------

def orthonormal_basis(vectors):
    """Orthonormal basis of the span of ``vectors`` (each a 3-vector).

    Robust SVD-based: returns an (m,3) array of orthonormal rows (m <= 3)
    spanning the column space of the stacked vectors.  Dropping near-zero
    singular directions avoids the >3-vector Gram-Schmidt degeneracy.
    """
    if not vectors:
        return np.zeros((0, 3), dtype=np.float64)
    vs = np.asarray([np.asarray(v, dtype=np.float64).reshape(3)
                     for v in vectors], dtype=np.float64)
    if vs.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    vs = vs.reshape(-1, 3)
    u, s, _vt = np.linalg.svd(vs.T, full_matrices=False)  # vs.T is (3,k)
    s = np.asarray(s, dtype=np.float64)
    tol = 1e-10 * float(s.max()) if s.size and float(s.max()) > 0 else 1e-12
    mask = s > tol
    return u[:, mask].T  # (m, 3) orthonormal rows


def project_residual(direction, vectors):
    """Residual ``1 - cos(d, proj_span(d)) = 1 - ||proj||`` for a UNIT ``d``.

    ``direction`` must be a 3-vector (any norm); ``vectors`` is a list of
    3-vectors spanning the available equivariant directions at the atom.
    Returns ``{"residual", "proj_norm"}``: residual 0 = fully expressible
    (||proj|| == 1), residual 1 = no overlap (||proj|| == 0).  Pure numpy.
    """
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    dn = float(np.linalg.norm(d))
    if dn < 1e-12:
        return {"residual": 1.0, "proj_norm": 0.0}
    d = d / dn
    basis = orthonormal_basis(vectors)
    if basis.size == 0:
        return {"residual": 1.0, "proj_norm": 0.0}
    coeffs = basis @ d                       # (m,)
    proj_norm = float(np.linalg.norm(coeffs))  # == ||proj|| for orthonormal basis
    proj_norm = min(max(proj_norm, 0.0), 1.0)
    return {"residual": float(1.0 - proj_norm), "proj_norm": proj_norm}


# ---------------------------------------------------------------------------
# Inserted-midpoint / original classification (WP2 provenance)
# ---------------------------------------------------------------------------

def load_inserted_state_ids(gaps_path: Path = GAPS_PATH) -> set[str]:
    """Set of inserted-midpoint state ids from the WP2 gaps registry.

    Rule (documented): WP2 subdivided over-cap steps (|a| > 0.1 A) by linear
    midpoint insertion; the registry ``items.inserted_nodes.{A,B}`` lists each
    inserted state id, its enclosing original steps, and the trigger
    magnitude.  A state is an *inserted midpoint* iff its state_id appears in
    that list; everything else is an original teacher node.
    """
    reg = json.loads(Path(gaps_path).read_text())
    ids: set[str] = set()
    for r in reg["reactions"]:
        for side in ("A", "B"):
            for rec in r.get("items", {}).get("inserted_nodes", {}).get(side, []):
                ids.add(str(rec["inserted_state_id"]))
    return ids


# ---------------------------------------------------------------------------
# Lineage wave ids
# ---------------------------------------------------------------------------

def lineage_wave_ids() -> dict:
    return json.loads(LINEAGE_PATH.read_text())["dual_ts_wave_selections"]["wave"]


# ---------------------------------------------------------------------------
# Recipe loss with gradients (mirrors DualTSBCTrainer.compute_losses)
# ---------------------------------------------------------------------------

def _move_and_cast(torch_mod, batch, device):
    for k, v in list(batch.items()):
        if torch_mod.is_tensor(v):
            if v.dtype in (torch.float64, torch.float32):
                v = v.to(dtype=torch.float32)
            batch[k] = v.to(device)
    return batch


def recipe_loss_terms(policy, states, action_max: float, device: str):
    """Returns ``(loss, nll_loss, mse_loss, dir_loss)`` for the recipe.

    Mirrors ``DualTSBCTrainer.compute_losses`` (per-step valid-atom SUM ->
    branch mean -> group mean) using the module-level frozen functions.
    The recipe is read from ``policy``-independent arg ``recipe`` in callers;
    this function returns the three term losses; callers combine per recipe.
    """
    torch_mod = policy.torch
    batch = collate_dual_states(torch_mod, list(states))
    batch = _move_and_cast(torch_mod, batch, device)
    tgt = build_targets(list(states), action_max)
    for k in ("u_star", "a_star", "atom_valid", "dir_mask",
              "cell_index", "cell_group", "state_of_atom"):
        tgt[k] = tgt[k].to(device)
    batch.update(tgt)
    out = policy(batch)
    mu, log_sigma = out["mu"], out["log_sigma"]
    valid = batch["atom_valid"]
    state_of_atom = batch["state_of_atom"]
    nll_atom = per_atom_nll(policy, mu, log_sigma, batch["u_star"])
    a_det = policy.deterministic(mu)
    mse_atom = per_atom_action_mse(a_det, batch["a_star"])
    dir_atom = per_atom_dir_loss(a_det, batch["a_star"])
    n_states = batch["n_states"]
    nll_state = torch_mod.zeros(n_states, device=mu.device).index_add_(
        0, state_of_atom, nll_atom.masked_fill(~valid, 0.0))
    mse_state = torch_mod.zeros(n_states, device=mu.device).index_add_(
        0, state_of_atom, mse_atom.masked_fill(~valid, 0.0))
    dir_state = torch_mod.zeros(n_states, device=mu.device).index_add_(
        0, state_of_atom, dir_atom.masked_fill(~batch["dir_mask"], 0.0))
    nll_loss, _ = reduce_step_sums(
        torch_mod, nll_state, batch["cell_index"], batch["n_cells"],
        batch["cell_group"], batch["n_groups"])
    mse_loss, _ = reduce_step_sums(
        torch_mod, mse_state, batch["cell_index"], batch["n_cells"],
        batch["cell_group"], batch["n_groups"])
    dir_loss, _ = reduce_step_sums(
        torch_mod, dir_state, batch["cell_index"], batch["n_cells"],
        batch["cell_group"], batch["n_groups"])
    return nll_loss, mse_loss, dir_loss


def recipe_combine(nll_loss, mse_loss, dir_loss, recipe: str):
    if recipe == "bc_nll":
        return nll_loss
    if recipe == "bc_a":
        return nll_loss + 1.0 * mse_loss
    return nll_loss + 1.0 * mse_loss + 1.0 * dir_loss


def grad_norm(loss, policy, retain_graph: bool = False):
    """Total parameter gradient norm ``sqrt(sum_g ||grad_g||^2)`` (float64)."""
    params = list(policy.parameters())
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph,
                                allow_unused=True)
    total = 0.0
    for g in grads:
        if g is not None:
            total += float((g.double() ** 2).sum().item())
    return math.sqrt(total)


__all__ = [
    "ACTION_MAX", "ARMS", "BASE", "DISPUTED_IDS", "G5_SLICE_PATH", "GAPS_PATH",
    "LINEAGE_PATH", "NEW_FIT_FEATURES", "NEW_MON_FEATURES", "OBSERVED_FIT",
    "OBSERVED_MON", "OLD_FIT_FEATURES", "OLD_MON_FEATURES", "OUT_DIR", "WP4_BC",
    "aligned", "arm_checkpoint", "atom_alpha", "atom_cos", "atom_mse",
    "device_of", "dir_cosine_weighted", "forward_det", "grad_norm",
    "lineage_wave_ids", "load_inserted_state_ids", "load_new_states",
    "load_old_policy", "load_old_states", "med_iqr", "mse_ratio_vs_zero",
    "orthonormal_basis", "percentile_dict", "per_sample_identity",
    "project_residual", "recipe_combine", "recipe_loss_terms",
    "set_new_layout", "set_old_layout", "write_json",
]
