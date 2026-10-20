#!/usr/bin/env python
"""R1 preregistered weight screen for the new pair-bucket losses (CPU).

Protocol (``outputs/pesn/v5_1_r1/protocol_r1.json`` →
``new_losses.weight_screen``):

* measurement point: ``B0_f0_s17_v51`` ``checkpoint_best`` state, update
  counter 600 (post-ramp), ``batches`` real fold-0 inner-train batches
  drawn with the campaign sampler (default 20);
* per term (``l_strong``, ``l_dead2``) and per candidate
  λ ∈ {0.03, 0.1, 0.3}::

      ratio(λ) = λ · ‖∇_θ L_term‖₂ / ‖∇_θ l_atom‖₂     (PARAMETER grads)

  reported as mean ± std over the batches;
* pick rule (preregistered): among candidates with ratio in [1%, 30%]
  pick the SMALLEST λ with ratio ≥ 2%; reject > 30% (drowns the primary
  objective) or < 1% (no effect); if none qualifies, pick the candidate
  closest to the window (all-in-window-but-below-2% → the largest ratio,
  i.e. closest to the 2% line) and document it.

Output: ``outputs/pesn/v5_1_r1/screen/weight_screen.json`` +
the picked λ per arm printed to stdout.  CPU-only by default.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Final, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.config_loader import load_config_mapping  # noqa: E402

SCREEN_SCHEMA: Final[str] = "pesn_v5_1_r1_weight_screen_v1"
DEFAULT_CANDIDATES: Final[tuple[float, ...]] = (0.03, 0.1, 0.3)
#: Preregistered window / target (protocol_r1.json pick_rule).
RATIO_LO: Final[float] = 0.01     # reject below 1% (no effect)
RATIO_HI: Final[float] = 0.30     # reject above 30% (drowns L_atom)
RATIO_TARGET: Final[float] = 0.02  # sufficient signal


def pick_lambda(ratio_means: Mapping[float, float]) -> dict:
    """Preregistered pick rule (pure; unit-tested).

    ``ratio_means`` maps candidate λ → mean ratio(λ).  Returns the pick,
    its basis, per-candidate eligibility and rejection reasons.
    """
    candidates = sorted(float(k) for k in ratio_means)
    ratios = {float(k): float(v) for k, v in ratio_means.items()}

    def _window_state(r: float) -> str:
        if r > RATIO_HI:
            return "reject_gt_30pct"
        if r < RATIO_LO:
            return "reject_lt_1pct"
        return "eligible"

    eligible = [l for l in candidates
                if _window_state(ratios[l]) == "eligible"]
    qualified = [l for l in eligible if ratios[l] >= RATIO_TARGET]
    if qualified:
        picked = min(qualified)
        basis = ("smallest candidate with ratio >= 2% inside the "
                 "[1%, 30%] window (preregistered pick_rule)")
    elif eligible:
        # inside the window but all below 2%: closest to the 2% line
        picked = max(eligible, key=lambda l: (ratios[l], -l))
        basis = ("no candidate reaches 2% inside [1%, 30%] — picked the "
                 "largest ratio (closest to the 2% line); documented "
                 "exception per pick_rule")
    else:
        def _distance(l: float) -> float:
            r = ratios[l]
            if r < RATIO_LO:
                return RATIO_LO - r
            if r > RATIO_HI:
                return r - RATIO_HI
            return 0.0
        picked = min(candidates, key=lambda l: (_distance(l), l))
        basis = ("all candidates outside [1%, 30%] — picked the one "
                 "CLOSEST to the window; documented exception per "
                 "pick_rule")
    return {
        "picked": picked,
        "basis": basis,
        "eligible": eligible,
        "per_candidate": {
            repr(l): {
                "ratio_mean": ratios[l],
                "state": _window_state(ratios[l]),
            } for l in candidates},
    }


def _param_grad_l2(model) -> float:
    """L2 norm of the parameter-gradient snapshot (0.0 if none/empty)."""
    total = 0.0
    seen = False
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach().double()
            total += float((g * g).sum())
            seen = True
    return math.sqrt(total) if seen else 0.0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python scripts/campaigns/r1_weight_screen.py",
        description="V5.1a-R1 gradient-ratio weight screen (CPU)")
    p.add_argument("--config",
                   default="config/campaigns/v5_1_sprint.yaml",
                   help="campaign yaml used to build the B0 machinery")
    p.add_argument("--checkpoint",
                   default="outputs/pesn/v5_1/runs/B0_f0_s17_v51/"
                           "checkpoints/checkpoint_best.pt",
                   help="B0_f0_s17_v51 checkpoint_best state")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--update", type=int, default=600,
                   help="update counter of the measurement point "
                        "(post-ramp)")
    p.add_argument("--batches", type=int, default=20)
    p.add_argument("--lambdas", default=",".join(
        repr(c) for c in DEFAULT_CANDIDATES),
                   help="comma-separated candidate weights")
    p.add_argument("--device", default="cpu",
                   help="device (CPU by default — the screen never needs "
                        "a GPU)")
    p.add_argument("--out",
                   default="outputs/pesn/v5_1_r1/screen/weight_screen.json")
    p.add_argument("--scratch",
                   default="outputs/pesn/v5_1_r1/screen/_trainer_scratch",
                   help="scratch run dir for the trainer machinery")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    import torch

    from pes2ts_core.pesn.offline.training import key_atom_bc as kbc

    candidates = tuple(float(x) for x in str(args.lambdas).split(",")
                       if str(x).strip())
    if not candidates:
        print("SCREEN_INVALID: empty --lambdas", file=sys.stderr)
        return 2
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        print(f"CHECKPOINT_NOT_FOUND: {checkpoint}", file=sys.stderr)
        return 2
    config_path = Path(args.config)
    config = load_config_mapping(config_path)
    arm = kbc.resolve_arm(config, "B0")
    t0 = time.time()
    trainer = kbc.KeyAtomBCTrainer(
        config=config, config_path=config_path, arm=arm,
        fold=int(args.fold), seed=int(args.seed), updates=1,
        mode="v51", run_dir=Path(args.scratch),
        device_override=str(args.device), repo_root=PROJECT_ROOT)

    doc = torch.load(checkpoint, map_location="cpu", weights_only=False)
    head_on = bool(doc.get("magnitude_head", False))
    if head_on or str(doc.get("pair_loss")) != "legacy_j":
        print(f"CHECKPOINT_ARM_MISMATCH: magnitude_head={head_on}, "
              f"pair_loss={doc.get('pair_loss')!r} — expected B0 "
              "(magnitude_head=false, pair_loss=legacy_j)", file=sys.stderr)
        return 2
    trainer.model.load_state_dict(doc["model"])
    trainer.model.train()

    # --- real fold-0 inner-train batches (campaign sampler + collate) ---
    pools = trainer._build_pools()
    inner_train = pools["inner_train"]
    torch.manual_seed(int(args.seed))
    sampler = kbc.KeyAtomCycleSampler(
        inner_train, seed=int(args.seed),
        reactions_per_batch=trainer.reactions_per_batch,
        frames_per_reaction=trainer.frames_per_reaction)
    batches: list = []
    while len(batches) < int(args.batches):
        rows = sampler.draw_batch()
        batches.extend(trainer._load_samples(rows, dense=False))
    batches = batches[:int(args.batches)]

    cfg = trainer.loss_config
    update = int(args.update)
    lam_cos = kbc.cos_lambda(update, cfg)
    base_strong: list[float] = []
    base_dead2: list[float] = []
    atom_norms: list[float] = []
    strong_norms: list[float] = []
    dead2_norms: list[float] = []
    n_empty_strong = n_empty_dead2 = 0

    for batch in batches:
        u, _z = trainer.model(batch)
        if batch.y_teacher is None:
            continue
        out = kbc.key_atom_bc_loss_vectorized(
            u.float(), batch.y_teacher.float(), batch.degK.float(),
            batch.key_ptr, batch.pos, batch.key_index, batch.pair_index,
            config=cfg, lambda_cos=lam_cos, pair_edit=batch.pair_edit,
            s=None,
            # force the buckets ON (λ here only gates computation; the
            # screen measures the RAW term gradients)
            lambda_strong=1.0, lambda_dead2=1.0)

        def _term_norm(term: str) -> float:
            trainer.model.zero_grad(set_to_none=True)
            value = out[term]
            if not bool(getattr(value, "requires_grad", False)):
                return 0.0  # empty bucket → constant 0, no graph
            value.backward(retain_graph=True)
            return _param_grad_l2(trainer.model)

        n_atom = _term_norm("l_atom")
        n_strong = _term_norm("l_strong")
        n_dead2 = _term_norm("l_dead2")
        trainer.model.zero_grad(set_to_none=True)
        if not (math.isfinite(n_atom) and n_atom > 0.0):
            continue  # degenerate batch — never fabricate a ratio
        atom_norms.append(n_atom)
        strong_norms.append(n_strong)
        dead2_norms.append(n_dead2)
        base_strong.append(n_strong / n_atom)
        base_dead2.append(n_dead2 / n_atom)
        if int(out["n_strong_pairs"]) == 0:
            n_empty_strong += 1
        if int(out["n_dead2_pairs"]) == 0:
            n_empty_dead2 += 1

    n_used = len(base_strong)
    if n_used < max(5, len(batches) // 2):
        print(f"SCREEN_INVALID: only {n_used}/{len(batches)} usable "
              "batches (finite non-zero l_atom grad norms)", file=sys.stderr)
        return 2

    import numpy as np

    def _term_block(base: Sequence[float], empty: int) -> dict:
        base_arr = np.asarray(base, dtype=np.float64)
        per_lambda = {}
        ratio_means = {}
        for lam in candidates:
            samples = lam * base_arr
            mean = float(samples.mean())
            ratio_means[float(lam)] = mean
            per_lambda[repr(float(lam))] = {
                "lambda": float(lam),
                "ratio_mean": mean,
                # sample std over batches (ddof=1; None at n=1)
                "ratio_std": (float(samples.std(ddof=1))
                              if len(samples) > 1 else 0.0),
                "ratio_samples": [float(x) for x in samples],
            }
        pick = pick_lambda(ratio_means)
        return {
            "base_ratio_mean": float(base_arr.mean()),
            "base_ratio_std": (float(base_arr.std(ddof=1))
                               if len(base_arr) > 1 else 0.0),
            "empty_bucket_batches": int(empty),
            "per_lambda": per_lambda,
            "pick": pick,
        }

    strong_block = _term_block(base_strong, n_empty_strong)
    dead_block = _term_block(base_dead2, n_empty_dead2)
    lambda_strong = float(strong_block["pick"]["picked"])
    lambda_dead2 = float(dead_block["pick"]["picked"])

    payload = {
        "schema_version": SCREEN_SCHEMA,
        "campaign_id": "V5_1A_R1_REPAIR",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rule": (
            "ratio(λ) = λ·‖∇_θ L_term‖₂ / ‖∇_θ l_atom‖₂ over PARAMETER "
            "gradients; candidates with ratio in [1%, 30%] are admissible; "
            "pick the SMALLEST λ with ratio >= 2%; reject >30% or <1%; "
            "exceptions pick the candidate closest to the window "
            "(preregistered: protocol_r1.json new_losses.weight_screen)"),
        "measurement_point": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": kbc._sha256_or_none(checkpoint),
            "checkpoint_update": int(doc.get("update", -1)),
            "update": update,
            "fold": int(args.fold),
            "seed": int(args.seed),
            "config_path": str(config_path),
            "n_batches_requested": int(args.batches),
            "n_batches_used": n_used,
            "torch_version": str(torch.__version__),
            "device": str(trainer.device),
        },
        "candidates": [float(c) for c in candidates],
        "grad_norms": {
            "l_atom_mean": float(np.mean(atom_norms)),
            "l_strong_mean": float(np.mean(strong_norms)),
            "l_dead2_mean": float(np.mean(dead2_norms)),
        },
        "terms": {"l_strong": strong_block, "l_dead2": dead_block},
        "picks": {"lambda_strong": lambda_strong,
                  "lambda_dead2": lambda_dead2},
        "arm_map": {
            "A2": {"term": "l_strong", "lambda_strong": lambda_strong},
            "A3": {"term": "l_dead2", "lambda_dead2": lambda_dead2},
            "A4": {"term": "both",
                   "lambda_strong": lambda_strong,
                   "lambda_dead2": lambda_dead2},
        },
        "wall_s": time.time() - t0,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    print(f"weight screen -> {out_path}")
    for term in ("l_strong", "l_dead2"):
        block = payload["terms"][term]
        for lam in candidates:
            row = block["per_lambda"][repr(float(lam))]
            print(f"  {term:8s} λ={lam:<5g} ratio = "
                  f"{row['ratio_mean']:.4f} ± {row['ratio_std']:.4f}")
        print(f"  {term:8s} picked λ = {block['pick']['picked']:g} "
              f"({block['pick']['basis']})")
    print(f"picked lambda per arm: "
          f"A2 (strong) λ_strong={lambda_strong:g}; "
          f"A3 (dead)   λ_dead2={lambda_dead2:g}; "
          f"A4 (both)   λ_strong={lambda_strong:g}, "
          f"λ_dead2={lambda_dead2:g}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
