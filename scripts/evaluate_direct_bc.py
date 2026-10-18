#!/usr/bin/env python3
"""Evaluate a trained direct BC checkpoint (fit50 characterization).

Metrics (plan §7.1 first-round recording list):
* group-balanced NLL (training-consistent reduction);
* independent baselines under the same radial transform: N(0, I3) latent
  prior per atom, and the fixed-sigma ``mu=0`` baseline;
* deterministic-action direction cosine vs expert displacement
  (mass-weighted aggregate over atoms and steps);
* per-atom displacement error in angstrom and as a fraction of a_max;
* sigma statistics per atom;
* per-termination-class NLL (goal / budget / no-progress) and per-reaction
  spread.

No QC, no truth; reads checkpoints and the transition dataset only.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pes2ts_core.pesn.models.egnn import EGNNConfig, DirectEGNNEncoder, collate_direct_features  # noqa: E402
from pes2ts_core.pesn.models.direct_policy import RadialDirectPolicy, radial_inverse  # noqa: E402
from pes2ts_core.pesn.offline.data.direct_dataset import DirectTransitionDataset  # noqa: E402
from pes2ts_core.pesn.offline.training.direct_bc import sample_to_feature_inputs  # noqa: E402
from pes2ts_core.pesn.offline.data.direct_preprocess import masses_for  # noqa: E402
from pes2ts_core.pesn.offline.training.device import resolve_device  # noqa: E402


def load_policy(settings: dict, checkpoint_path: Path):
    torch, device = resolve_device(device="cuda:0", require_cuda=True)
    encoder = DirectEGNNEncoder(EGNNConfig(
        hidden=settings["hidden"], n_layers=settings["n_layers"],
        light=bool(settings.get("light", False))))
    policy = RadialDirectPolicy(encoder=encoder,
                                action_max=float(settings["action_max"]),
                                sigma_range=tuple(settings["sigma_range"])).to(device)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy.module.load_state_dict(payload["model_state"]["policy"])
    policy.train_mode(False)
    return torch, device, policy, payload


def baseline_nll_log_z(samples, a_max: float, sigma: float | None):
    """Mean per-sample log pi_z under mu=0 baselines (N(0,I3) or N(0,sigma^2))."""
    import torch
    from pes2ts_core.pesn.models.direct_policy import radial_log_det_per_atom
    total = 0.0
    n = 0
    for sample in samples:
        a = torch.as_tensor(sample.action.astype(np.float64))
        u = radial_inverse(torch, torch.as_tensor(a / a_max))
        s = 1.0 if sigma is None else float(sigma)
        resid = u / s
        per_atom = (-1.5 * math.log(2 * math.pi) - 3.0 * math.log(s)
                    - 0.5 * (resid * resid).sum(-1)
                    - radial_log_det_per_atom(torch, u))
        total += float(per_atom.sum())
        n += 1
    return total / max(n, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--selection", default="direct_fit50")
    parser.add_argument("--checkpoint", default="checkpoint_last.pt")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    import torch
    payload = torch.load(args.run_dir / args.checkpoint, map_location="cpu",
                         weights_only=False)
    settings = payload["extra"]["settings"]
    torch, device, policy, _ = load_policy(settings, args.run_dir / args.checkpoint)
    a_max = float(settings["action_max"])

    dataset = DirectTransitionDataset(args.dataset_dir,
                                      allowed_subsets=[args.selection])
    reaction_ids = dataset.reaction_ids

    per_reaction = {}
    all_nll, all_cos, all_norm_err, all_sigma = [], [], [], []
    all_cos_moving, all_n_moved = [], []
    nll_by_reason: dict[str, list[float]] = {}
    samples_all = []
    with torch.no_grad():
        for rid in reaction_ids:
            samples = [s for s in dataset.get_reaction(rid)]
            samples_all.extend(samples)
            features = [sample_to_feature_inputs(s) for s in samples]
            batch = collate_direct_features(torch, features)
            batch = {k: (v.to(device) if hasattr(v, "to") else v)
                     for k, v in batch.items()}
            out = policy(batch)
            actions = torch.cat([torch.as_tensor(s.action.astype(np.float32))
                                 for s in samples]).to(device)
            u_expert = radial_inverse(torch, actions / a_max)
            lp = policy.module.log_prob_z(out["mu"], out["log_sigma"], u_expert)
            per_atom = lp["per_atom"]
            per_sample = torch.zeros(len(samples), device=device).index_add_(
                0, batch["node_graph_id"], per_atom)
            r_nll = float(-per_sample.mean())
            per_reaction[rid] = {
                "n": len(samples),
                "mean_nll": round(r_nll, 3),
                "mean_log_sigma": round(float(out["log_sigma"].mean()), 4),
            }
            all_nll.extend((-per_sample).tolist())
            all_sigma.extend(out["log_sigma"].tolist())
            # deterministic action vs expert: cosine + norm error (per atom, mass weighted)
            det = policy.module.deterministic(out["mu"])            # (Ntot,3)
            z_exp = (actions / a_max)
            masses = torch.as_tensor(
                np.concatenate([masses_for(s.atomic_numbers) for s in samples]).astype(np.float32)
            ).to(device)
            det_sel = det
            dot = (det_sel * actions).sum(-1)
            cos = dot / (det_sel.norm(dim=-1) * actions.norm(dim=-1) + 1e-12)
            w = masses / masses.mean()
            n_moved = int((actions.norm(dim=-1) > 0.02).sum())
            all_cos.append(float((cos * w).sum() / w.sum()))
            moving = actions.norm(dim=-1) > 0.02
            all_cos_moving.append(
                float(cos[moving].mean()) if bool(moving.any()) else float("nan"))
            all_n_moved.append(n_moved)
            err = (det_sel - actions).norm(dim=-1)
            all_norm_err.append(float((err * w).sum() / w.sum()))
            for s in samples:
                nll_by_reason.setdefault(s.termination_reason, []).append(r_nll)

    import collections
    payload_out = {
        "run_dir": str(args.run_dir),
        "checkpoint_epoch": payload.get("epoch"),
        "n_reactions": len(reaction_ids),
        "mean_nll": float(np.mean(all_nll)),
        "baseline_nll_mu0_sigma1": round(baseline_nll_log_z(samples_all, a_max, None), 3),
        "baseline_nll_mu0_sigma_init": round(
            baseline_nll_log_z(samples_all, a_max, 0.5), 3),
        "direction_cosine_mass_weighted": round(float(np.mean(all_cos)), 4),
        "direction_cosine_moving_atoms_only": round(
            float(np.nanmean(all_cos_moving)), 4),
        "n_moving_atoms_per_sample": round(
            float(np.sum(all_n_moved)) / max(1, len(all_nll)), 4),
        "displacement_error_angstrom_mean": round(float(np.mean(all_norm_err)), 4),
        "displacement_error_frac_amax": round(float(np.mean(all_norm_err)) / a_max, 4),
        "log_sigma_mean": round(float(np.mean(all_sigma)), 4),
        "log_sigma_min": round(float(np.min(all_sigma)), 4),
        "log_sigma_max": round(float(np.max(all_sigma)), 4),
        "nll_by_termination_reason": {
            k: {"n": len(v), "mean_nll": round(float(np.mean(v)), 3)}
            for k, v in sorted(nll_by_reason.items())
        },
        "per_reaction": per_reaction,
        "note": ("fit50 characterization at the recorded checkpoint; not the "
                 "formal fit500/2k three-seed result and not a capability claim"),
    }
    text = json.dumps(payload_out, indent=2)
    if args.output:
        args.output.write_text(text)
    print(json.dumps({k: v for k, v in payload_out.items() if k != "per_reaction"},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
