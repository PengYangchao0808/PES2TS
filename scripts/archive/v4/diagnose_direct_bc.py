#!/usr/bin/env python3
"""Diagnose the BC plateau: clamped sigma, gradient flow, oracle floors.

Hypotheses tested:
* H1: the log-sigma head saturates at the upper clamp -> zero gradient
  through clamp.max -> sigma permanently stuck at sigma_max;
* H2: the mu path receives usable gradient signal (overfit test should
  drive NLL far below the N(0,1) baseline on a tiny subset);
* H3: the achievable per-atom floor (oracle mu, oracle sigma) explains how
  much of the loss is even learnable at this scale.

No QC, no truth; reads checkpoint + dataset only.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pes2ts_core.pesn.models.egnn import EGNNConfig, DirectEGNNEncoder, collate_direct_features  # noqa: E402
from pes2ts_core.pesn.models.direct_policy import RadialDirectPolicy, radial_inverse  # noqa: E402
from pes2ts_core.pesn.offline.data.direct_dataset import DirectTransitionDataset  # noqa: E402
from pes2ts_core.pesn.offline.training.direct_bc import sample_to_feature_inputs  # noqa: E402
from pes2ts_core.pesn.offline.training.device import resolve_device  # noqa: E402


def build_modules(settings, checkpoint):
    import torch
    torch, device = resolve_device(device="cuda:0", require_cuda=True)
    encoder = DirectEGNNEncoder(EGNNConfig(
        hidden=settings["hidden"], n_layers=settings["n_layers"]))
    policy = RadialDirectPolicy(encoder=encoder,
                                action_max=float(settings["action_max"]),
                                sigma_range=tuple(settings["sigma_range"])).to(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    policy.module.load_state_dict(payload["model_state"]["policy"])
    return torch, device, policy, payload


def batch_for(torch, device, samples):
    features = [sample_to_feature_inputs(s) for s in samples]
    batch = collate_direct_features(torch, features)
    batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
    actions = torch.cat([torch.as_tensor(s.action.astype(np.float32))
                         for s in samples]).to(device)
    return batch, actions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--selection", default="direct_fit50")
    args = parser.parse_args()

    import torch
    payload = torch.load(args.run_dir / "checkpoint_last.pt", map_location="cpu",
                         weights_only=False)
    settings = payload["extra"]["settings"]
    torch, device, policy, _ = build_modules(settings, args.run_dir / "checkpoint_last.pt")
    a_max = float(settings["action_max"])
    dataset = DirectTransitionDataset(args.dataset_dir, allowed_subsets=[args.selection])

    # --- H1: sigma head state (clamp vs softplus parametrizations) ----------
    samples = dataset.get_reaction(dataset.reaction_ids[0])
    batch, actions = batch_for(torch, device, samples)
    enc = policy.module.encoder(batch)
    sigma_min = settings["sigma_range"][0]
    if hasattr(policy.module, "raw_sigma_head"):
        raw = policy.module.raw_sigma_head(enc["h"]).squeeze(-1)
        sigma = sigma_min + torch.nn.functional.softplus(raw)
        h1 = {
            "parametrization": "softplus",
            "raw_min": round(float(raw.min()), 4),
            "raw_max": round(float(raw.max()), 4),
            "raw_mean": round(float(raw.mean()), 4),
            "sigma_min": round(float(sigma.min()), 4),
            "sigma_max": round(float(sigma.max()), 4),
            "sigma_mean": round(float(sigma.mean()), 4),
            "n_nodes": int(raw.numel()),
        }
    else:
        pre_clamp = policy.module.log_sigma_head(enc["h"]).squeeze(-1)
        h1 = {
            "parametrization": "clamp",
            "pre_clamp_min": round(float(pre_clamp.min()), 4),
            "pre_clamp_max": round(float(pre_clamp.max()), 4),
            "pre_clamp_mean": round(float(pre_clamp.mean()), 4),
            "n_at_max_clamp": int((pre_clamp >= math.log(settings["sigma_range"][1])).sum()),
            "n_at_min_clamp": int((pre_clamp <= math.log(settings["sigma_range"][0])).sum()),
            "n_nodes": int(pre_clamp.numel()),
        }

    # --- gradient flow check ------------------------------------------------
    u_expert = radial_inverse(torch, actions / a_max)
    policy.module.zero_grad(set_to_none=True)
    out = policy(batch)
    lp = policy.module.log_prob_z(out["mu"], out["log_sigma"], u_expert)
    per_sample = torch.zeros(len(samples), device=device).index_add_(
        0, batch["node_graph_id"], lp["per_atom"])
    loss = -per_sample.mean()
    loss.backward()
    grad_norms = {}
    for name, param in policy.module.named_parameters():
        if param.grad is not None:
            grad_norms[name] = round(float(param.grad.norm()), 8)
    sigma_key = ("raw_sigma_head" if hasattr(policy.module, "raw_sigma_head")
                 else "log_sigma_head")
    h2_grads = {
        "mu_gate_weight": grad_norms.get("mu_gate.weight"),
        "sigma_head_weight": grad_norms.get(f"{sigma_key}.weight"),
        "sigma_head_bias": grad_norms.get(f"{sigma_key}.bias"),
        "encoder_first_zembed": grad_norms.get("encoder.z_embed.weight"),
    }

    # --- H3: oracle floors on the full fit50 ---------------------------------
    all_u, all_shoulder = [], []
    for rid in dataset.reaction_ids:
        rs = dataset.get_reaction(rid)
        b, a = batch_for(torch, device, rs)
        ue = radial_inverse(torch, a / a_max)
        all_u.append(ue.detach().cpu())
    U = torch.cat(all_u)                      # (Ntot, 3)
    r2 = (U * U).sum(-1)                      # per-atom squared radius
    n_atoms = U.shape[0]
    from pes2ts_core.pesn.models.direct_policy import radial_log_det_per_atom
    log_det = radial_log_det_per_atom(torch, U)
    const = -1.5 * math.log(2 * math.pi)
    # mu-oracle (sigma fixed at 1.0)
    per_atom_sigma1 = const - 0.5 * 0.0 - log_det  # mu=u -> resid 0
    # sigma-oracle (mu arbitrary -> resid 0): sigma^2 = r2/3
    sigma_star2 = (r2 / 3.0).clamp(min=1e-12)
    # per-atom: -1.5 log(2pi) - 3 log(sigma*) - 0.5*||resid||^2/sigma*^2 - logdet;
    # with mu = u the residual term vanishes and 3 log sigma* = 1.5 log sigma*^2
    per_atom_sigma_star = const - 1.5 * torch.log(sigma_star2) - log_det
    h3 = {
        "n_atoms_total": int(n_atoms),
        "mean_radius": round(float(r2.sqrt().mean()), 4),
        "mean_nll_per_atom_model_equivalent": round(float(-per_atom_sigma1.mean()), 4),
        "mean_nll_per_atom_sigma_oracle": round(float(-per_atom_sigma_star.mean()), 4),
        "sigma_oracle_mean": round(float(sigma_star2.sqrt().mean()), 4),
        "improvement_available_per_atom": round(
            float(per_atom_sigma1.mean() - per_atom_sigma_star.mean()), 4),
    }

    # --- H2: overfit test on two reactions -----------------------------------
    torch.manual_seed(0)
    subset = dataset.reaction_ids[:2]
    tiny = []
    for rid in subset:
        tiny.extend(dataset.get_reaction(rid))
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    traj = []
    policy.train_mode(True)
    for step in range(400):
        b, a = batch_for(torch, device, tiny)
        ue = radial_inverse(torch, a / a_max)
        policy.module.zero_grad(set_to_none=True)
        o = policy(b)
        l = -(policy.module.log_prob_z(o["mu"], o["log_sigma"], ue)["per_atom"]
              .reshape(-1).mean())
        l.backward()
        opt.step()
        if step % 50 == 0 or step == 399:
            traj.append({"step": step, "loss_per_atom": round(float(l), 4)})
    h2_overfit = {"n_samples": len(tiny), "trajectory": traj}

    report = {
        "run_dir": str(args.run_dir),
        "h1_sigma_saturation": h1,
        "h2_gradient_norms": h2_grads,
        "h3_oracle_floors": h3,
        "h2_overfit_trajectory": h2_overfit,
    }
    out_path = args.run_dir / "diagnosis.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
