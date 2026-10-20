#!/usr/bin/env python3
"""V5.1a R1: gradient diagnostics probe over fold-0 inner-train batches (CPU).

Lane: campaign V5_1A_R1_REPAIR — gradient-diagnostics.

Loads the frozen B0 fold-0 best checkpoint
(``outputs/pesn/v5_1/runs/B0_f0_s17_v51/checkpoints/checkpoint_best.pt``;
arm B0 = magnitude_head off, pair_loss legacy_j) into ``StarEGNN`` and
probes per-loss-term gradients on 20 REAL fold-0 inner-train batches
(inner split: frozen seed 20261019, n_val 8 → 32 inner-train clusters;
sampler/collate reused verbatim from ``key_atom_bc``).  Update counter is
fixed at 600 (aux ramp full: λ_cos=0.02, λ_mag=0.5, λ_sign=0.2,
λ_dead=0.1 per cos_lambda/aux_ramp_fraction).

Per batch, autograd per loss term SEPARATELY (retain_graph; grads w.r.t.
ALL model params, unused params → zero vectors):

* ``l_atom`` (vector radial-huber term);
* ``l_pair`` under pair_loss='legacy_j' (first-order J term);
* reference trio under pair_loss='exact_trio' — pair_reg / sign / dead
  (the exact_trio config is built via dataclasses.replace on the campaign
  KeyAtomLossConfig, NOT via the arm table: arm B0's invariant is
  pair_loss=legacy_j, and the loss function itself accepts any
  KeyAtomLossConfig, so the config object is the legitimate switch point);
* pair_reg RESTRICTED to strong-bucket pairs (|Δd_true| ≥ 0.02 A) vs its
  complement ("weak+dead" in the R1 spec = non-strong = weak+medium+dead,
  |Δd_true| < 0.02 A) on the SAME batch, implemented by ZEROING per-pair
  huber contributions while keeping the module's per-frame denominators
  (pair_cnt.clamp(min=1)) and frame mask EXACTLY — no module is modified;
* l_atom RESTRICTED to key atoms that are endpoints of strong-changing
  pairs vs endpoints of non-strong pairs (per-atom contributions zeroed,
  per-frame degK weights unchanged; an atom may belong to both masks).

Reported per term: mean±std across batches of the grad L2 norm;
cosines: cos(l_atom, l_pair) for BOTH pair_loss modes, trio-conflict
cos(strong-restricted, non-strong-restricted pair_reg), and the analogous
restricted l_atom cosine.  Numbers must be finite (asserted).

Outputs (deterministic, re-runnable, overwritten in place):

* ``outputs/pesn/v5_1_r1/gradients/gradient_probe.json``
  (schema ``pesn_v5_1_r1_gradient_probe_v1``)
* ``outputs/pesn/v5_1_r1/gradients/gradient_attribution_data.md``
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCHEMA_VERSION: Final[str] = "pesn_v5_1_r1_gradient_probe_v1"

#: Frozen strong-bucket edge (A): pairs with |dd_true| >= this are "strong".
STRONG_EDGE: Final[float] = 0.02
#: Probe update counter (aux ramp full after update 500).
PROBE_UPDATE: Final[int] = 600
DEFAULT_N_BATCHES: Final[int] = 20
DEFAULT_SEED: Final[int] = 17
DEFAULT_INNER_VAL_SEED: Final[int] = 20261019
DEFAULT_INNER_VAL_N: Final[int] = 8
D0: Final[float] = 0.05
FOLD: Final[int] = 0

DEFAULT_CHECKPOINT = (PROJECT_ROOT / "outputs/pesn/v5_1/runs"
                      / "B0_f0_s17_v51/checkpoints/checkpoint_best.pt")
DEFAULT_CONFIG = PROJECT_ROOT / "config/campaigns/v5_1_sprint.yaml"
DEFAULT_SPLIT = PROJECT_ROOT / "data/manifests/v5/split_manifest.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs/pesn/v5_1_r1/gradients"


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------

def import_with_retry(module_name: str, *, retries: int = 3,
                      delay_s: float = 30.0):
    """Import a core module, retrying transient failures (a concurrent lane
    may be editing key_atom_bc.py)."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return __import__(module_name, fromlist=["*"])
        except Exception as exc:  # noqa: BLE001 — retry any transient error
            last = exc
            if attempt < retries - 1:
                print(f"[r1_grad] import {module_name} failed ({exc!r}); "
                      f"retrying in {delay_s:.0f}s "
                      f"({attempt + 1}/{retries})", file=sys.stderr)
                time.sleep(delay_s)
    raise ImportError(f"cannot import {module_name} after {retries} "
                      f"attempts: {last!r}")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def flat_grads(loss_terms: dict, names: Sequence[str], params: list,
               *, torch) -> dict[str, "np.ndarray"]:
    """Per-term flattened gradient vectors over ALL model params.

    Unused/unconnected params contribute zero vectors (e.g. z_head never
    reaches any BC loss term).  A term that does not require grad at all
    (e.g. l_sign when a batch has no teacher-active pairs) yields the zero
    vector, recorded via a side list by the caller through requires_grad.
    """
    out: dict[str, np.ndarray] = {}
    for name in names:
        term = loss_terms[name]
        if not bool(getattr(term, "requires_grad", False)):
            out[name] = np.zeros(sum(p.numel() for p in params),
                                 dtype=np.float64)
            continue
        grads = torch.autograd.grad(
            term, params, retain_graph=True, allow_unused=True)
        chunks = [
            (torch.zeros_like(p) if g is None else g).detach().reshape(-1)
            for p, g in zip(params, grads)]
        out[name] = torch.cat(chunks).to(torch.float64).cpu().numpy()
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return None
    return float(np.dot(a, b) / (na * nb))


def mean_std(values: Sequence[float | None]) -> dict:
    finite = [float(v) for v in values if v is not None
              and math.isfinite(float(v))]
    n_none = sum(1 for v in values if v is None)
    if not finite:
        return {"mean": None, "std": None, "n": 0, "n_undefined": n_none}
    arr = np.asarray(finite, dtype=np.float64)
    return {"mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()), "max": float(arr.max()),
            "n": len(finite), "n_undefined": n_none}


# ---------------------------------------------------------------------------
# restricted losses (zeroing per-pair / per-atom contributions; no module
# modification — mirrors key_atom_bc_loss_vectorized internals EXACTLY)
# ---------------------------------------------------------------------------

def restricted_trio_pair_terms(*, torch, key_atom_bc, pos, key_index,
                               key_ptr, pair_index, pair_edit, u, y):
    """exact_trio pair_reg split into strong vs non-strong by zeroing
    per-pair contributions.

    Mirrors the vectorized loss: global→slot pair conversion, pair_edit
    filter, per-frame huber means with ``pair_cnt.clamp(min=1)``
    denominators, active-frame mask and /n_used aggregation.  Returns
    ``(l_strong, l_non_strong, n_pairs_strong, n_pairs_non)`` — by
    construction l_strong + l_non_strong == the module's l_pair_reg.
    """
    from pes2ts_core.pesn.contracts.key_atom import (
        global_pair_rows_to_slots)
    exact_delta_d_torch = key_atom_bc.exact_delta_d_torch
    radial_huber = key_atom_bc.radial_huber
    n_graphs = int(key_ptr.shape[0]) - 1
    lo = key_ptr[:-1]
    hi = key_ptr[1:]
    active = hi > lo
    n_used = int(active.sum())
    keep = active.to(u.dtype)
    scale = 0.08  # cfg.loss_scale_angstrom (a_ref), identical for both arms

    slots = global_pair_rows_to_slots(pair_index, key_index,
                                      int(pos.shape[0]))
    keep_mask = pair_edit.to(slots.device) != 0
    slots = slots[:, keep_mask]
    atom_frame = torch.repeat_interleave(
        torch.arange(n_graphs, device=u.device),
        (hi - lo).clamp(min=0))
    sum_s = torch.zeros(n_graphs, dtype=u.dtype, device=u.device)
    sum_n = torch.zeros(n_graphs, dtype=u.dtype, device=u.device)
    cnt = torch.zeros(n_graphs, dtype=u.dtype, device=u.device)
    n_strong = n_non = 0
    if slots.numel() > 0:
        pi = slots[0].long()
        pj = slots[1].long()
        pair_frame = atom_frame[pi]
        x_key = pos[key_index]
        dd_u = exact_delta_d_torch(x_key, slots, u)
        dd_y = exact_delta_d_torch(x_key, slots, y)
        pair_h = radial_huber(((dd_u - dd_y).abs() / scale).to(u.dtype),
                              0.25)
        strong = dd_y.abs() >= STRONG_EDGE
        non = ~strong
        n_strong = int(strong.sum())
        n_non = int(non.sum())
        if bool(strong.any()):
            sum_s.index_add_(0, pair_frame[strong], pair_h[strong])
        if bool(non.any()):
            sum_n.index_add_(0, pair_frame[non], pair_h[non])
        cnt.index_add_(0, pair_frame,
                       torch.ones_like(pair_h, dtype=u.dtype))
    denom = cnt.clamp(min=1.0)
    l_s_frames = sum_s / denom
    l_n_frames = sum_n / denom
    l_strong = (l_s_frames * keep).sum() / max(n_used, 1)
    l_non = (l_n_frames * keep).sum() / max(n_used, 1)
    return l_strong, l_non, n_strong, n_non


def restricted_atom_terms(*, torch, degK, key_ptr, u, y,
                          pos, key_index, pair_index, pair_edit):
    """l_atom split into strong-pair-endpoint atoms vs non-strong-pair
    endpoint atoms by zeroing per-atom contributions; per-frame degK
    weights and the frame mask stay EXACTLY as in the module.  An atom
    that is an endpoint of both strong and non-strong pairs appears in
    BOTH masks (reported separately); atoms in no changed pair contribute
    to neither.  Returns
    ``(l_atom_strong, l_atom_non, n_atoms_s, n_atoms_n, n_atoms_both,
    n_atoms_neither)``.
    """
    from pes2ts_core.pesn.contracts.key_atom import (
        global_pair_rows_to_slots)
    radial_huber = key_atom_bc_mod_holder["mod"].radial_huber
    n_graphs = int(key_ptr.shape[0]) - 1
    lo = key_ptr[:-1]
    hi = key_ptr[1:]
    active = hi > lo
    n_used = int(active.sum())
    keep = active.to(u.dtype)
    scale = 0.08
    n_keys = int(u.shape[0])

    atom_frame = torch.repeat_interleave(
        torch.arange(n_graphs, device=u.device),
        (hi - lo).clamp(min=0))
    w_raw = 1.0 / degK.to(dtype=u.dtype).clamp(min=1.0)
    w_sum = torch.zeros(n_graphs, dtype=u.dtype, device=u.device)
    w_sum.index_add_(0, atom_frame, w_raw)
    w = w_raw / w_sum[atom_frame].clamp(min=1e-12)
    e = (u - y).norm(dim=-1) / scale
    per_atom = w * radial_huber(e, 0.25)

    slots = global_pair_rows_to_slots(pair_index, key_index,
                                      int(pos.shape[0]))
    keep_mask = pair_edit.to(slots.device) != 0
    slots = slots[:, keep_mask]
    mask_s = torch.zeros(n_keys, dtype=torch.bool, device=u.device)
    mask_n = torch.zeros(n_keys, dtype=torch.bool, device=u.device)
    if slots.numel() > 0:
        dd_y = key_atom_bc_mod_holder["exact"](
            pos[key_index], slots, y)
        strong = (dd_y.abs() >= STRONG_EDGE)
        s_slots = slots[:, strong]
        n_slots = slots[:, ~strong]
        if s_slots.numel() > 0:
            mask_s[s_slots.reshape(-1).long()] = True
        if n_slots.numel() > 0:
            mask_n[n_slots.reshape(-1).long()] = True
    both = mask_s & mask_n
    neither = ~mask_s & ~mask_n
    sum_s = torch.zeros(n_graphs, dtype=u.dtype, device=u.device)
    sum_n = torch.zeros(n_graphs, dtype=u.dtype, device=u.device)
    if bool(mask_s.any()):
        sum_s.index_add_(0, atom_frame, per_atom * mask_s.to(u.dtype))
    if bool(mask_n.any()):
        sum_n.index_add_(0, atom_frame, per_atom * mask_n.to(u.dtype))
    l_s = (sum_s * keep).sum() / max(n_used, 1)
    l_n = (sum_n * keep).sum() / max(n_used, 1)
    return (l_s, l_n, int(mask_s.sum()), int(mask_n.sum()),
            int(both.sum()), int(neither.sum()))


#: late-bound module holder (avoids threading the import everywhere)
key_atom_bc_mod_holder: dict = {}


# ---------------------------------------------------------------------------
# markdown draft
# ---------------------------------------------------------------------------

def _fmt(v, nd: int = 4) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{float(v):.{nd}f}"


def _ms_row(label: str, m: dict, nd: int = 4) -> str:
    return (f"| {label} | {_fmt(m['mean'], nd)} | {_fmt(m['std'], nd)} | "
            f"{_fmt(m.get('min'), nd)} | {_fmt(m.get('max'), nd)} | "
            f"{m['n']} / {m['n_undefined']} |")


def write_markdown(path: Path, doc: dict) -> None:
    lines: list[str] = []
    add = lines.append
    add("# R1 gradient probe — data tables (DRAFT)")
    add("")
    add(f"- schema: `{doc['schema_version']}`  generated: {doc['generated_at']}")
    add(f"- checkpoint: `{doc['inputs']['checkpoint']}` "
        f"(sha256 `{doc['inputs']['checkpoint_sha256'][:16]}…`)")
    add(f"- fold={doc['probe']['fold']} seed={doc['probe']['seed']} "
        f"update={doc['probe']['update']} batches={doc['probe']['n_batches']} "
        f"(CPU, fp32; inner-train clusters: "
        f"{doc['probe']['n_inner_train_clusters']}; inner split hash "
        f"`{doc['probe']['inner_split_hash'][:16]}…`)")
    add(f"- aux ramp at update {doc['probe']['update']}: "
        f"λ_cos={doc['probe']['lambda_cos']}, "
        f"λ_mag={doc['probe']['lambda_mag']}, "
        f"λ_sign={doc['probe']['lambda_sign']}, "
        f"λ_dead={doc['probe']['lambda_dead']}")
    add("")
    add("## Per-term gradient L2 norms (over ALL model params)")
    add("")
    add("| term | mean | std | min | max | n batches (finite/undefined) |")
    add("|---|---|---|---|---|---|")
    for label, key in (("l_atom", "l_atom"),
                       ("l_pair (legacy_j)", "l_pair_legacy"),
                       ("l_pair_reg (exact_trio)", "l_pair_reg_trio"),
                       ("l_sign (exact_trio)", "l_sign_trio"),
                       ("l_dead (exact_trio)", "l_dead_trio")):
        add(_ms_row(label, doc["grad_norms"][key]))
    add("")
    add("## Cosines")
    add("")
    add("| cosine | mean | std | min | max | n batches (finite/undefined) |")
    add("|---|---|---|---|---|---|")
    add(_ms_row("cos(l_atom, l_pair) legacy_j",
                doc["cosines"]["l_atom_vs_l_pair_legacy"]))
    add(_ms_row("cos(l_atom, l_pair_reg) exact_trio",
                doc["cosines"]["l_atom_vs_l_pair_reg_trio"]))
    add(_ms_row("trio conflict: cos(pair_reg strong, pair_reg non-strong)",
                doc["cosines"]["trio_conflict_pair_reg"]))
    add(_ms_row("cos(l_atom strong-endpoint, l_atom non-strong-endpoint)",
                doc["cosines"]["l_atom_restricted"]))
    add("")
    tc = doc["trio_conflict"]
    add(f"Trio conflict verdict (neutral rule: mean cosine < 0 ⇒ gradients "
        f"oppose): mean = {_fmt(tc['cosine']['mean'])} ⇒ "
        f"{'CONFLICT' if (tc['cosine']['mean'] is not None and tc['cosine']['mean'] < 0) else 'NO CONFLICT (aligned on average)'}"
        f".  Norm ratio strong/non-strong: mean "
        f"{_fmt(tc['norm_ratio_strong_over_non']['mean'])}, std "
        f"{_fmt(tc['norm_ratio_strong_over_non']['std'])}.")
    add("")
    add("## Batch composition (mean across batches)")
    add("")
    bc = doc["batch_composition"]
    add("| quantity | mean |")
    add("|---|---|")
    add(f"| pairs strong (|Δd_true| ≥ {STRONG_EDGE}) | "
        f"{_fmt(bc['n_pairs_strong']['mean'])} |")
    add(f"| pairs non-strong | {_fmt(bc['n_pairs_non_strong']['mean'])} |")
    add(f"| atoms in ≥1 strong pair | {_fmt(bc['n_atoms_strong']['mean'])} |")
    add(f"| atoms in ≥1 non-strong pair | "
        f"{_fmt(bc['n_atoms_non_strong']['mean'])} |")
    add(f"| atoms in both masks | {_fmt(bc['n_atoms_both']['mean'])} |")
    add(f"| atoms in no changed pair | "
        f"{_fmt(bc['n_atoms_neither']['mean'])} |")
    add("")
    add("## Observations (neutral)")
    add("")
    for obs in doc["observations"]:
        add(f"- {obs}")
    add("")
    add("_Draft data tables — interpretation belongs to the R1 lane "
        "report._")
    add("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--split-manifest", default=str(DEFAULT_SPLIT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--n-batches", type=int, default=DEFAULT_N_BATCHES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--update", type=int, default=PROBE_UPDATE)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    if not args.device.startswith("cpu"):
        raise SystemExit("r1_gradient_probe is CPU-only by contract "
                         "(pass --device cpu)")
    # config paths (feature_root / split_manifest) are repo-relative
    import os
    os.chdir(PROJECT_ROOT)

    import torch

    # bit-determinism: single-threaded reductions (multi-threaded OpenMP
    # reduction order is not reproducible across runs at the 1e-10 level)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    key_atom_bc = import_with_retry(
        "pes2ts_core.pesn.offline.training.key_atom_bc")
    key_atom_bc_mod_holder["mod"] = key_atom_bc
    key_atom_bc_mod_holder["exact"] = key_atom_bc.exact_delta_d_torch
    from pes2ts_core.config_loader import load_config_mapping
    from pes2ts_core.pesn.models.key_atom_egnn import StarEGNN

    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config)
    split_path = Path(args.split_manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config_mapping(config_path)
    device = torch.device("cpu")
    torch.manual_seed(int(args.seed))

    # --- model reconstruction from the checkpoint dict (arm B0: magnitude
    # head OFF; constructor args read from the checkpoint, defaults match
    # the frozen model class contract) ---
    doc_ck = torch.load(checkpoint_path, map_location="cpu",
                        weights_only=False)
    if str(doc_ck.get("schema_version")) != "pesn_key_atom_bc_checkpoint_v1":
        raise SystemExit(f"unexpected checkpoint schema "
                         f"{doc_ck.get('schema_version')!r}")
    arm = dict(doc_ck.get("arm") or {})
    magnitude_head = bool(doc_ck.get("magnitude_head",
                                     arm.get("magnitude_head", False)))
    magnitude_mode = str(doc_ck.get("magnitude_mode")
                         or arm.get("magnitude_mode") or "per_atom")
    model = StarEGNN(magnitude_head=magnitude_head,
                     magnitude_mode=magnitude_mode).to(
        torch.float32).to(device)
    model.load_state_dict(doc_ck["model"])
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]

    # --- fold-0 inner-train pools (frozen inner split) ---
    pools = key_atom_bc.load_lane_b_pools(
        config, fold=FOLD, d0=D0, split_path=str(split_path),
        inner_val={"n_val": DEFAULT_INNER_VAL_N, "seed": DEFAULT_INNER_VAL_SEED})
    inner_train = pools["inner_train"]
    inner_split = dict(pools["inner_split"])
    # cross-check against the training run's recorded split hash
    ckpt_split_hash = (dict(doc_ck.get("v51") or {})
                       .get("inner_val_split_hash"))
    split_hash_match = (ckpt_split_hash is None
                        or ckpt_split_hash == inner_split["hash"])

    # --- loss configs: campaign defaults (legacy_j for B0) and the
    # exact_trio REFERENCE built by replacing the config object field
    # directly (arm invariants live in the trainer's arm table; the loss
    # function takes any KeyAtomLossConfig, so the config object is the
    # legitimate switch point — documented deviation-free path) ---
    cfg_legacy = key_atom_bc.KeyAtomLossConfig.from_campaign_config(config)
    cfg_trio = dataclasses.replace(cfg_legacy, pair_loss="exact_trio")
    lam_cos = key_atom_bc.cos_lambda(int(args.update), cfg_legacy)
    aux = key_atom_bc.aux_ramp_fraction(int(args.update), cfg_legacy)
    lam_mag = cfg_legacy.lambda_mag * aux
    lam_sign = cfg_legacy.lambda_sign * aux
    lam_dead = cfg_legacy.lambda_dead * aux

    # --- 20 real batches: seeded sampler over inner-train pools ---
    sampler = key_atom_bc.KeyAtomCycleSampler(
        inner_train, seed=int(args.seed),
        reactions_per_batch=int(config.get("reactions_per_batch", 8)),
        frames_per_reaction=int(config.get("frames_per_reaction", 4)))

    batch_size = int(config.get("batch_size", 32))

    def load_batch(rows: list[dict]):
        view, collate = key_atom_bc.load_lane_b_view(
            config=config, samples=rows, fold=FOLD, d0=D0, anchor="on",
            split_path=str(split_path), device=device)
        chunk = [view[j] for j in range(len(view))]
        return collate(chunk)

    norms: dict[str, list[float]] = {
        "l_atom": [], "l_pair_legacy": [], "l_pair_reg_trio": [],
        "l_sign_trio": [], "l_dead_trio": []}
    cos_legacy: list[float | None] = []
    cos_trio: list[float | None] = []
    cos_conflict: list[float | None] = []
    cos_atom_restricted: list[float | None] = []
    conflict_norm_ratios: list[float | None] = []
    comp = {"n_pairs_strong": [], "n_pairs_non_strong": [],
            "n_atoms_strong": [], "n_atoms_non_strong": [],
            "n_atoms_both": [], "n_atoms_neither": []}
    integrity: list[float] = []

    for b_idx in range(int(args.n_batches)):
        rows = sampler.draw_batch()
        batch = load_batch(rows)
        u, _z = model(batch)
        y = batch.y_teacher.float()
        pair_edit = batch.pair_edit

        common = dict(
            config=cfg_legacy, lambda_cos=lam_cos, pair_edit=pair_edit,
            s=None, lambda_mag=lam_mag, lambda_sign=lam_sign,
            lambda_dead=lam_dead)
        out_legacy = key_atom_bc.key_atom_bc_loss_vectorized(
            u.float(), y, batch.degK.float(), batch.key_ptr, batch.pos,
            batch.key_index, batch.pair_index, **common)
        g_legacy = flat_grads(out_legacy,
                              ("l_atom", "l_pair"), params, torch=torch)

        out_trio = key_atom_bc.key_atom_bc_loss_vectorized(
            u.float(), y, batch.degK.float(), batch.key_ptr, batch.pos,
            batch.key_index, batch.pair_index,
            **{**common, "config": cfg_trio})
        g_trio = flat_grads(
            out_trio, ("l_pair_reg", "l_sign", "l_dead"), params, torch=torch)

        # restricted trio pair_reg (strong vs non-strong, zeroed pairs)
        l_s, l_ns, n_ps, n_pn = restricted_trio_pair_terms(
            torch=torch, key_atom_bc=key_atom_bc, pos=batch.pos,
            key_index=batch.key_index, key_ptr=batch.key_ptr,
            pair_index=batch.pair_index, pair_edit=pair_edit, u=u.float(),
            y=y)
        # integrity: strong + non-strong == module pair_reg
        integrity.append(abs(float((l_s + l_ns
                                    - out_trio["l_pair_reg"]).detach())))
        g_conf_s = flat_grads({"t": l_s}, ("t",), params, torch=torch)["t"]
        g_conf_n = flat_grads({"t": l_ns}, ("t",), params, torch=torch)["t"]

        # restricted l_atom (endpoints of strong vs non-strong pairs)
        (la_s, la_n, n_as, n_an, n_ab, n_a_neither) = restricted_atom_terms(
            torch=torch, degK=batch.degK.float(), key_ptr=batch.key_ptr,
            u=u.float(), y=y, pos=batch.pos, key_index=batch.key_index,
            pair_index=batch.pair_index, pair_edit=pair_edit)
        g_a_s = flat_grads({"t": la_s}, ("t",), params, torch=torch)["t"]
        g_a_n = flat_grads({"t": la_n}, ("t",), params, torch=torch)["t"]

        # --- collect ---
        norms["l_atom"].append(float(np.linalg.norm(g_legacy["l_atom"])))
        norms["l_pair_legacy"].append(
            float(np.linalg.norm(g_legacy["l_pair"])))
        norms["l_pair_reg_trio"].append(
            float(np.linalg.norm(g_trio["l_pair_reg"])))
        norms["l_sign_trio"].append(float(np.linalg.norm(g_trio["l_sign"])))
        norms["l_dead_trio"].append(float(np.linalg.norm(g_trio["l_dead"])))
        cos_legacy.append(cosine(g_legacy["l_atom"], g_legacy["l_pair"]))
        cos_trio.append(cosine(g_legacy["l_atom"], g_trio["l_pair_reg"]))
        cos_conflict.append(cosine(g_conf_s, g_conf_n))
        cos_atom_restricted.append(cosine(g_a_s, g_a_n))
        n_s_norm = float(np.linalg.norm(g_conf_s))
        n_n_norm = float(np.linalg.norm(g_conf_n))
        conflict_norm_ratios.append(
            n_s_norm / n_n_norm if n_n_norm > 0 else None)
        comp["n_pairs_strong"].append(float(n_ps))
        comp["n_pairs_non_strong"].append(float(n_pn))
        comp["n_atoms_strong"].append(float(n_as))
        comp["n_atoms_non_strong"].append(float(n_an))
        comp["n_atoms_both"].append(float(n_ab))
        comp["n_atoms_neither"].append(float(n_a_neither))

    # --- finiteness assertions ---
    bad: list[str] = []
    for key, vals in norms.items():
        for i, v in enumerate(vals):
            if not math.isfinite(v):
                bad.append(f"norm {key}[{i}] = {v}")
    for name, vals in (("cos_legacy", cos_legacy), ("cos_trio", cos_trio),
                       ("cos_conflict", cos_conflict),
                       ("cos_atom_restricted", cos_atom_restricted),
                       ("conflict_ratio", conflict_norm_ratios)):
        for i, v in enumerate(vals):
            if v is not None and not math.isfinite(v):
                bad.append(f"{name}[{i}] = {v}")
    max_integrity = max(integrity) if integrity else 0.0
    if max_integrity > 1e-6:
        bad.append(f"restricted-pair integrity max |Δ| = {max_integrity}")
    if bad:
        for msg in bad:
            print(f"[r1_grad] NON-FINITE/INTEGRITY: {msg}", file=sys.stderr)
        raise SystemExit("gradient probe produced non-finite values or "
                         "failed the restricted-loss integrity check")

    grad_norms_out = {k: mean_std(v) for k, v in norms.items()}
    cosines_out = {
        "l_atom_vs_l_pair_legacy": mean_std(cos_legacy),
        "l_atom_vs_l_pair_reg_trio": mean_std(cos_trio),
        "trio_conflict_pair_reg": mean_std(cos_conflict),
        "l_atom_restricted": mean_std(cos_atom_restricted),
    }
    conflict_cos = cosines_out["trio_conflict_pair_reg"]["mean"]
    ratio_ms = mean_std(conflict_norm_ratios)
    comp_out = {k: mean_std(v) for k, v in comp.items()}

    observations = [
        ("grad L2 norm order of magnitude per term (mean over all model "
         "params): l_atom=" + _fmt(grad_norms_out["l_atom"]["mean"], 6)
         + ", l_pair(legacy)=" + _fmt(grad_norms_out["l_pair_legacy"]["mean"], 6)
         + ", l_pair_reg(trio)=" + _fmt(grad_norms_out["l_pair_reg_trio"]["mean"], 6)
         + ", l_sign=" + _fmt(grad_norms_out["l_sign_trio"]["mean"], 6)
         + ", l_dead=" + _fmt(grad_norms_out["l_dead_trio"]["mean"], 6)),
        ("per-term norms are UNWEIGHTED; at update 600 the trainer scales "
         "them by λ_pair=0.5, λ_sign=0.2, λ_dead=0.1 (λ_cos=0.02), so the "
         "weighted l_sign contribution is ~"
         + _fmt(0.2 * (grad_norms_out["l_sign_trio"]["mean"] or 0.0), 4)
         + " and weighted l_dead ~"
         + _fmt(0.1 * (grad_norms_out["l_dead_trio"]["mean"] or 0.0), 4)
         + " per batch before cosine-annealing/LR"),
        ("cos(l_atom, l_pair) legacy_j: mean "
         + _fmt(cosines_out["l_atom_vs_l_pair_legacy"]["mean"])
         + "; exact_trio pair_reg: mean "
         + _fmt(cosines_out["l_atom_vs_l_pair_reg_trio"]["mean"])),
        ("trio conflict cos(strong-restricted, non-strong-restricted "
         "pair_reg): mean " + _fmt(conflict_cos) + " ("
         + ("negative on average — the two pair subsets push parameters "
            "in opposing directions"
            if conflict_cos is not None and conflict_cos < 0 else
            "non-negative on average — no first-order conflict detected")
         + "); strong/non-strong norm ratio mean "
         + _fmt(ratio_ms["mean"])),
        ("restricted l_atom cosine (strong-endpoint vs non-strong-endpoint "
         "atoms): mean "
         + _fmt(cosines_out["l_atom_restricted"]["mean"])
         + " (atom masks overlap: an atom in both a strong and a "
         "non-strong pair contributes to both restricted terms)"),
        ("pair-count context per batch (means): strong pairs "
         + _fmt(comp_out["n_pairs_strong"]["mean"])
         + ", non-strong pairs "
         + _fmt(comp_out["n_pairs_non_strong"]["mean"])),
        ("exact_trio reference built via dataclasses.replace on the "
         "campaign KeyAtomLossConfig (pair_loss='exact_trio'); arm B0 "
         "invariants untouched (arm table still legacy_j)"),
        ("restricted losses reproduce the module's denominators exactly; "
         "integrity |l_strong + l_non_strong - l_pair_reg| max = "
         + f"{max_integrity:.3e}"),
    ]

    doc = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "inputs": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_update": int(doc_ck.get("update", -1)),
            "checkpoint_arm": arm,
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "split_manifest": str(split_path),
            "split_manifest_sha256": sha256_file(split_path),
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
        },
        "probe": {
            "fold": FOLD,
            "seed": int(args.seed),
            "update": int(args.update),
            "n_batches": int(args.n_batches),
            "batch_size": batch_size,
            "d0": D0,
            "n_inner_train_clusters": len(inner_train),
            "inner_split_seed": DEFAULT_INNER_VAL_SEED,
            "inner_split_n_val": DEFAULT_INNER_VAL_N,
            "inner_split_hash": inner_split["hash"],
            "inner_split_hash_matches_checkpoint": bool(split_hash_match),
            "lambda_cos": lam_cos,
            "lambda_mag": float(lam_mag),
            "lambda_sign": float(lam_sign),
            "lambda_dead": float(lam_dead),
            "strong_edge_angstrom": STRONG_EDGE,
            "non_strong_definition": ("|dd_true| < 0.02 A (weak + medium + "
                                      "dead) — the complement of the strong "
                                      "bucket on the same batch"),
            "restriction_method": ("per-pair / per-atom contributions "
                                   "zeroed; module denominators (pair_cnt, "
                                   "degK weights, frame mask, /n_used) kept "
                                   "verbatim; no module modified"),
        },
        "grad_norms": grad_norms_out,
        "cosines": cosines_out,
        "trio_conflict": {
            "cosine": cosines_out["trio_conflict_pair_reg"],
            "norm_ratio_strong_over_non": ratio_ms,
            "conflict": (conflict_cos is not None and conflict_cos < 0),
            "rule": ("mean cosine < 0 ⇒ opposing first-order updates "
                     "(conflict)"),
        },
        "batch_composition": comp_out,
        "integrity": {
            "max_abs_l_strong_plus_l_non_minus_module_pair_reg":
                max_integrity,
            "tolerance": 1e-6,
        },
        "observations": observations,
    }

    json_path = out_dir / "gradient_probe.json"
    json_path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    md_path = out_dir / "gradient_attribution_data.md"
    write_markdown(md_path, doc)

    print(f"[r1_grad] wrote {json_path}")
    print(f"[r1_grad] wrote {md_path}")
    for key in ("l_atom", "l_pair_legacy", "l_pair_reg_trio",
                "l_sign_trio", "l_dead_trio"):
        m = grad_norms_out[key]
        print(f"  |grad {key:18s}| mean={_fmt(m['mean'], 6)} "
              f"std={_fmt(m['std'], 6)}")
    print(f"  cos(l_atom, l_pair) legacy: "
          f"{_fmt(cosines_out['l_atom_vs_l_pair_legacy']['mean'])}  "
          f"trio: {_fmt(cosines_out['l_atom_vs_l_pair_reg_trio']['mean'])}")
    print(f"  trio conflict cos: {_fmt(conflict_cos)} "
          f"(conflict={doc['trio_conflict']['conflict']}), "
          f"norm ratio strong/non: {_fmt(ratio_ms['mean'])}")
    print(f"  restricted l_atom cos: "
          f"{_fmt(cosines_out['l_atom_restricted']['mean'])}")
    print(f"  integrity max |Δ| = {max_integrity:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
