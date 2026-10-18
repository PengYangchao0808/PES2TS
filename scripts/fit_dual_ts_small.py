"""WP3-B small-data learnability fitting runs (plan v4 §D WP3 row, §I.9).

Short GPU runs (<= 2000 updates each, Adam) of the dual-TS EGNN + radial
policy head on the REAL WP3-A fit50 feature shards, with NLL +
deterministic-action MSE (plan BC-A composition), per config:

  (i)   branch_a        — branch-A states only
  (ii)  branch_b        — branch-B states only
  (iii) joint           — both branches
  (iv)  near_ts         — near-TS region only: remaining arc length
                          <= 20% of the branch maximum (teacher-endpoint
                          states still excluded from the action loss)
  (v)   moving_atoms    — all states, per-atom loss restricted to the top
                          quartile of per-atom |a| over the fit population
  (vi)  zero_baseline   — no training; zero-action baseline MSE on the
                          joint population (reference for every config)

Reported per config: final train loss (recomputed full-population with the
final weights), per-atom action MSE, direction cosine on |a|-weighted
samples (masked per the frozen spec: |a_i| < 0.01 Å excluded,
teacher-endpoint states excluded), against the zero-action baseline on the
same population.  This lane PROVES expressiveness/learnability only — full
BC recipes, monitor selection and frozen thresholds are WP4 scope.  Honest
reporting: a config that fails to beat the zero-action baseline is
diagnostic signal for WP4, not a reason to extend training.

Artifacts under outputs/pesn/dual_ts_v2_20261015/wp3_model/:
  fit_<config>.json      run record (config, spec digest, shard digests,
                         seeds, hyperparams, curve, metrics)
  ckpt_<config>.pt       checkpoint (content digest bound in the bundle)
  bundle_<config>.json   pesn_dual_ts_bundle_v2 record (validated)
  summary.json/.md       summary table
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from pes2ts_core.pesn.contracts.dual_ts import (  # noqa: E402
    BUNDLE_SCHEMA_VERSION,
    DualTSBundle,
    DualTSFeaturesSpec,
    DualTSTeacher,
    RADIAL_TRANSFORM_VERSION,
    validate_record,
)
from pes2ts_core.pesn.features.dual_ts_graph import (  # noqa: E402
    build_teacher_supervision,
)
from pes2ts_core.pesn.models import require_torch  # noqa: E402
from pes2ts_core.pesn.models.dual_egnn import (  # noqa: E402
    DUAL_TS_EGNN_VERSION,
    DualTSModelConfig,
    collate_dual_states,
)
from pes2ts_core.pesn.models.dual_ts_policy import (  # noqa: E402
    DUAL_TS_POLICY_VERSION,
    DEFAULT_A_MAX_ANGSTROM,
    DEFAULT_SIGMA_RANGE,
    DualTSPolicy,
)

torch = require_torch()

FEATURES_DIR = PROJECT_ROOT / "outputs/pesn/dual_ts_v2_20261015/wp3_features_v41"
OBSERVED_DIR = PROJECT_ROOT / "outputs/pesn/dual_ts_v2_20261015/wp2_fit50_observed"
OUT_DIR = PROJECT_ROOT / "outputs/pesn/dual_ts_v2_20261015/wp3_model"

TRAIN_CONFIGS = ["branch_a", "branch_b", "joint", "near_ts", "moving_atoms"]
ALL_CONFIGS = TRAIN_CONFIGS + ["zero_baseline"]

NEAR_TS_ARC_FRACTION = 0.2
MOVING_QUARTILE = 0.75
LAMBDA_A = 1.0
CURVE_EVERY = 25


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_spec_and_manifest() -> tuple[DualTSFeaturesSpec, dict]:
    manifest = json.loads((FEATURES_DIR / "features_manifest.json").read_text())
    spec_payload = json.loads((FEATURES_DIR / "features_spec.json").read_text())
    spec = DualTSFeaturesSpec.from_payload(spec_payload["spec"])
    digest = spec.content_digest()
    if digest != manifest["spec_digest"]:
        raise RuntimeError(
            f"SPEC_DIGEST_MISMATCH: recomputed {digest} != manifest "
            f"{manifest['spec_digest']}")
    if digest != spec_payload["spec_digest"]:
        raise RuntimeError("SPEC_DIGEST_MISMATCH: features_spec.json wrapper")
    return spec, manifest


def load_states() -> list[dict]:
    """One record per fit state: shard input arrays + teacher action label.

    The shards are consumed AS-IS (per-state centroid-removed X is the
    frozen consume-time translation rule).  The supervision action is the
    contract-frozen ``label.action_delta`` of the SAME bound teacher table
    the shards were built from (shard X is per-state centered, so actions
    must NOT be re-derived from shard X differences).
    """
    shard_paths = sorted(FEATURES_DIR.glob("features_RXN_*.npz"))
    if len(shard_paths) != 50:
        raise RuntimeError(f"SHARD_COUNT_INVALID: {len(shard_paths)} != 50")
    states: list[dict] = []
    mask_mismatches = 0
    n_actions = 0
    for path in shard_paths:
        shard = np.load(path)
        teacher = DualTSTeacher.from_payload(json.loads(
            (OBSERVED_DIR / f"{shard['reaction_id'].item()}.teacher.json"
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
        for row, sid in enumerate(shard["state_id"]):
            lab = labels[str(sid)]
            action = (None if lab.action_delta is None
                      else np.asarray(lab.action_delta, dtype=np.float64))
            if action is not None:
                n_actions += 1
                sup = build_teacher_supervision(
                    action_delta=action,
                    is_teacher_endpoint=bool(is_teacher[row]),
                    n_atoms=int(z.size))
                mask_mismatches += int(np.any(
                    sup.near_zero_action_mask != stored_mask[row]))
            states.append({
                "state_id": str(sid),
                "reaction_id": str(shard["reaction_id"].item()),
                "branch_id": str(shard["branch_id"][row]),
                "episode_id": str(shard["episode_id"][row]),
                "step_index": int(shard["step_index"][row]),
                "is_teacher": bool(is_teacher[row]),
                "remaining_arc_length": float(lab.remaining_arc_length),
                "action": action,
                "near_zero_mask": np.asarray(stored_mask[row], dtype=bool),
                "atomic_numbers": z,
                "node_scalars": node_scalars[row],
                "x_angstrom": x[row],
                "g_scaled": g[row],
                "atom_feats_a": atom_a,
                "atom_feats_b": atom_b,
                "edge_feats_a": edge_a,
                "edge_feats_b": edge_b,
                "edge_index": edge_index,
            })
    if mask_mismatches:
        raise RuntimeError(
            f"MASK_MISMATCH: stored near-zero mask disagrees with the "
            f"teacher action on {mask_mismatches} atoms")
    print(f"loaded {len(states)} states from {len(shard_paths)} shards; "
          f"{n_actions} action labels; mask check OK", flush=True)
    return states


def branch_side(state: dict) -> str:
    return state["branch_id"].split("#")[-1]


def select_population(states: list[dict], config: str,
                      near_ts_fraction: float) -> tuple[list[dict], dict]:
    if config in ("branch_a", "branch_b"):
        side = "brA" if config == "branch_a" else "brB"
        pop = [s for s in states if branch_side(s) == side]
    elif config == "joint":
        pop = list(states)
    elif config == "near_ts":
        by_branch: dict[str, list[dict]] = {}
        for s in states:
            by_branch.setdefault(s["branch_id"], []).append(s)
        pop = []
        for branch_states in by_branch.values():
            ral = np.array([s["remaining_arc_length"]
                            for s in branch_states])
            cut = ral.max() * near_ts_fraction
            pop.extend(s for s, r in zip(branch_states, ral) if r <= cut)
    elif config in ("moving_atoms", "zero_baseline"):
        pop = list(states)
    else:
        raise ValueError(f"unknown config {config}")
    meta = {
        "n_states": len(pop),
        "n_teacher_states": sum(s["is_teacher"] for s in pop),
        "n_action_states": sum(not s["is_teacher"] for s in pop),
    }
    return pop, meta


def moving_atom_threshold(states: list[dict]) -> float:
    norms = np.concatenate([
        np.linalg.norm(s["action"], axis=1)
        for s in states if not s["is_teacher"]])
    return float(np.quantile(norms, MOVING_QUARTILE))


# ---------------------------------------------------------------------------
# Batching / losses / metrics
# ---------------------------------------------------------------------------

def batch_to_device(batch: dict, device, dtype) -> dict:
    out = {}
    for key, val in batch.items():
        if torch.is_tensor(val):
            if val.dtype in (torch.float64, torch.float32):
                val = val.to(dtype)
            out[key] = val.to(device)
        else:
            out[key] = val
    return out


def state_atom_meta(states: list[dict]) -> dict:
    """Per-atom arrays aligned with the collated batch order."""
    n_atoms = [int(np.asarray(s["atomic_numbers"]).size) for s in states]
    is_teacher_atom = np.repeat([s["is_teacher"] for s in states], n_atoms)
    near_zero = np.concatenate([s["near_zero_mask"] for s in states])
    actions = np.concatenate([
        np.zeros((n, 3)) if s["action"] is None else s["action"]
        for s, n in zip(states, n_atoms)], axis=0)
    return {
        "n_atoms": n_atoms,
        "is_teacher_atom": is_teacher_atom,
        "near_zero": near_zero,
        "actions": actions,
        "action_norm": np.linalg.norm(actions, axis=1),
    }


def loss_atom_mask(meta: dict, config: str, moving_thr: float) -> np.ndarray:
    mask = ~meta["is_teacher_atom"]
    if config == "moving_atoms":
        mask = mask & (meta["action_norm"] >= moving_thr)
    return mask


def eval_population(pol: DualTSPolicy, states: list[dict], config: str,
                    moving_thr: float, device, dtype,
                    batch_size: int = 256) -> dict:
    """Full-population metrics of the CURRENT weights + zero baseline."""
    meta = state_atom_meta(states)
    lmask = loss_atom_mask(meta, config, moving_thr)
    mu_all, sigma_all = [], []
    with torch.no_grad():
        for i in range(0, len(states), batch_size):
            chunk = states[i:i + batch_size]
            batch = batch_to_device(collate_dual_states(torch, chunk),
                                    device, dtype)
            out = pol(batch)
            mu_all.append(out["mu"].detach().cpu().numpy())
            sigma_all.append(out["sigma"].detach().cpu().numpy())
    mu = np.concatenate(mu_all, axis=0)
    sigma = np.concatenate(sigma_all, axis=0)
    a = meta["actions"]
    a_det = DEFAULT_A_MAX_ANGSTROM * np.tanh(
        np.linalg.norm(mu, axis=1, keepdims=True)) * mu / np.maximum(
        np.linalg.norm(mu, axis=1, keepdims=True), 1e-12)
    err2 = ((a_det - a) ** 2).sum(axis=1)

    # direction metric population: spec mask (near-zero excluded, teacher
    # states excluded entirely), weighted by |a_i|
    dmask = (~meta["is_teacher_atom"]) & (~meta["near_zero"])
    if config == "moving_atoms":
        dmask = dmask & (meta["action_norm"] >= moving_thr)
    na = meta["action_norm"][dmask]
    cos = (a_det[dmask] * a[dmask]).sum(axis=1) / np.maximum(
        na * np.linalg.norm(a_det[dmask], axis=1), 1e-12)
    w = na
    dir_cos = float((w * cos).sum() / w.sum())
    mse = float(err2[lmask].mean())
    baseline = float((a ** 2).sum(axis=1)[lmask].mean())
    return {
        "n_atoms_loss": int(lmask.sum()),
        "n_atoms_direction": int(dmask.sum()),
        "per_atom_action_mse_A2": mse,
        "direction_cosine_absa_weighted": dir_cos,
        "zero_baseline_action_mse_A2": baseline,
        "beats_zero_baseline": bool(mse < baseline),
        "mse_over_baseline": float(mse / baseline),
        "sigma_mean": float(sigma.mean()),
    }


# ---------------------------------------------------------------------------
# Training run
# ---------------------------------------------------------------------------

def run_config(config: str, *, states: list[dict], moving_thr: float,
               updates: int, batch_size: int, lr: float, seed: int,
               device, dtype, lambda_a: float = LAMBDA_A,
               nll_weight: float = 1.0, tag: str = "") -> dict:
    t0 = time.time()
    pop, pop_meta = select_population(states, config, NEAR_TS_ARC_FRACTION)
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    random.seed(seed)
    pol = DualTSPolicy(config=DualTSModelConfig(light=True),
                       action_max=DEFAULT_A_MAX_ANGSTROM,
                       sigma_range=DEFAULT_SIGMA_RANGE).to(dtype).to(device)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    action_states = [i for i, s in enumerate(pop) if not s["is_teacher"]]
    if not action_states:
        raise RuntimeError(f"EMPTY_ACTION_POPULATION: {config}")
    curve = []
    running = {"loss": 0.0, "nll": 0.0, "mse": 0.0, "mu_norm": 0.0, "n": 0}
    update = 0
    order: list[int] = []
    while update < updates:
        if not order:
            order = list(np_rng.permutation(action_states))
        idx_batch = [order.pop() for _ in range(min(batch_size, len(order)))]
        chunk = [pop[i] for i in idx_batch]
        meta = state_atom_meta(chunk)
        lmask = torch.as_tensor(
            loss_atom_mask(meta, config, moving_thr),
            dtype=torch.bool, device=device)
        a_target = torch.as_tensor(meta["actions"], dtype=dtype, device=device)
        batch = batch_to_device(collate_dual_states(torch, chunk),
                                device, dtype)
        out = pol(batch)
        eps = torch.randn_like(out["mu"])
        u = out["mu"] + out["log_sigma"].exp().unsqueeze(-1) * eps
        lp = pol.log_prob_z(out["mu"], out["log_sigma"], u)["per_atom"]
        nll = -(lp * lmask).sum() / lmask.sum()
        a_det = pol.deterministic(out["mu"])
        mse = ((a_det - a_target) ** 2).sum(dim=1)[lmask].mean()
        loss = nll_weight * nll + lambda_a * mse
        opt.zero_grad()
        loss.backward()
        opt.step()
        running["loss"] += float(loss.detach())
        running["nll"] += float(nll.detach())
        running["mse"] += float(mse.detach())
        running["mu_norm"] += float(out["mu"].norm(dim=-1).mean().detach())
        running["n"] += 1
        update += 1
        if update % CURVE_EVERY == 0 or update == updates:
            n = max(running["n"], 1)
            curve.append({
                "update": update,
                "loss": running["loss"] / n,
                "nll_per_atom": running["nll"] / n,
                "action_mse_A2": running["mse"] / n,
                "mu_norm_mean": running["mu_norm"] / n,
            })
            running = {"loss": 0.0, "nll": 0.0, "mse": 0.0, "mu_norm": 0.0,
                       "n": 0}
    metrics = eval_population(pol, pop, config, moving_thr, device, dtype)
    ckpt_path = OUT_DIR / f"ckpt_{config}{tag}.pt"
    torch.save({
        "state_dicts": {k: {kk: vv.cpu() for kk, vv in v.items()}
                         for k, v in pol.state_dicts().items()},
        "config": {
            "hidden": pol.config.hidden, "n_layers": pol.config.n_layers,
            "n_basis": pol.config.n_basis, "cutoff": pol.config.cutoff,
            "light": pol.config.light,
            "signed_mu_gate": pol.signed_mu_gate,
            "signed_vector_messages": pol.config.signed_vector_messages,
        },
        "action_max": DEFAULT_A_MAX_ANGSTROM,
        "sigma_range": list(DEFAULT_SIGMA_RANGE),
    }, ckpt_path)
    record = {
        "config": config,
        "tag": tag.lstrip("_"),
        "population": {**pop_meta,
                       "selection": {
                           "branch_a": "branch_id suffix #brA",
                           "branch_b": "branch_id suffix #brB",
                           "joint": "all fit states",
                           "near_ts": f"remaining_arc_length <= "
                                      f"{NEAR_TS_ARC_FRACTION} * branch max",
                           "moving_atoms": f"|a_i| >= fit-population q"
                                           f"{MOVING_QUARTILE} = {moving_thr:.6f} A",
                       }[config]},
        "loss_atom_rule": (
            "teacher_endpoint excluded entirely; "
            + ("|a_i| >= moving threshold only; " if config == "moving_atoms" else "")
            + "NLL(mean per atom) + 1.0 * deterministic-action MSE"),
        "hyperparams": {
            "updates": updates, "batch_size_states": batch_size, "lr": lr,
            "lambda_a": lambda_a, "nll_weight": nll_weight,
            "optimizer": "adam",
            "model": DUAL_TS_EGNN_VERSION, "policy": DUAL_TS_POLICY_VERSION,
            "hidden": pol.config.hidden, "n_layers": pol.config.n_layers,
            "a_max_angstrom": DEFAULT_A_MAX_ANGSTROM,
            "sigma_min": DEFAULT_SIGMA_RANGE[0],
            "transform": RADIAL_TRANSFORM_VERSION,
            "composition": f"loss = {nll_weight} * NLL(log pi_z, mean/atom) "
                           f"+ {lambda_a} * deterministic-action MSE",
        },
        "seeds": {"torch": seed, "numpy": seed, "python": seed},
        "curve": curve,
        "metrics": metrics,
        "checkpoint_file": ckpt_path.name,
        "checkpoint_sha256": sha256_file(ckpt_path),
        "wall_seconds": round(time.time() - t0, 2),
    }
    return record


def zero_baseline_record(states: list[dict], moving_thr: float,
                         device, dtype) -> dict:
    pop, pop_meta = select_population(states, "zero_baseline",
                                      NEAR_TS_ARC_FRACTION)
    meta = state_atom_meta(pop)
    lmask = loss_atom_mask(meta, "joint", moving_thr)
    a = meta["actions"]
    metrics = {
        "n_atoms_loss": int(lmask.sum()),
        "n_atoms_direction": int((~meta["is_teacher_atom"]
                                  & ~meta["near_zero"]).sum()),
        "per_atom_action_mse_A2": float((a ** 2).sum(axis=1)[lmask].mean()),
        "direction_cosine_absa_weighted": None,
        "zero_baseline_action_mse_A2": float((a ** 2).sum(axis=1)[lmask].mean()),
        "beats_zero_baseline": False,
        "mse_over_baseline": 1.0,
        "sigma_mean": None,
        "note": "a=0 reference; direction cosine undefined for a zero action",
    }
    return {
        "config": "zero_baseline",
        "population": pop_meta,
        "metrics": metrics,
        "note": "no training; a=0 reference on the joint population",
    }


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

def build_bundle(record: dict, spec_digest: str, shard_digests: list[str],
                 seed: int) -> DualTSBundle:
    hp = record["hyperparams"]
    tag = record.get("tag", "")
    bundle = DualTSBundle(
        run_id=f"pesn_dual_ts_wp3b_smallfit_{record['config']}"
               f"{'_' + tag if tag else ''}_v0",
        bc_parent_checkpoint_digest=record["checkpoint_sha256"],
        seeds={"torch": seed, "numpy": seed, "python": seed},
        hyperparams={
            **hp,
            "loss": "NLL(mean per atom, log pi_z) + 1.0 * deterministic action MSE",
            "population": record["population"],
            "updates_run": record["curve"][-1]["update"] if record["curve"] else 0,
        },
        dataset_version=f"pesn_dual_ts_features_v3@{spec_digest}",
        norm_version=(
            "dual_ts_frozen_scales_v0:"
            "s_g=0.01543995635890959_Ha/A:"
            "s_delta_e=0.07888807747799866_Ha"),
        split_version=f"wp3b_smallfit_{record['config']}_population_v0",
        action_version=f"{RADIAL_TRANSFORM_VERSION}:a_max="
                       f"{DEFAULT_A_MAX_ANGSTROM}:sigma_min="
                       f"{DEFAULT_SIGMA_RANGE[0]}",
        environment_version="pesn_dual_ts_env_v2:not_exercised_in_wp3b_v0",
        reward_version="pesn_dual_ts_reward_v2:not_exercised_in_wp3b_v0",
        input_eligibility={
            "spec_digest": spec_digest,
            "shard_count": len(shard_digests),
            "shard_digests_sha256": shard_digests,
            "near_zero_mask_rule": (
                "per_atom_action_norm_below_threshold_masked_out_of_"
                "direction_loss_v0 (0.01 A)"),
            "teacher_endpoint_rule": (
                "teacher_endpoint_states_excluded_from_action_loss_v0"),
            "action_labels": "contract label.action_delta of the SAME "
                             "bound wp2_fit50_observed teacher table",
        },
        output_eligibility={
            "purpose": "wp3b expressiveness/learnability diagnostic "
                       "(plan v4 §D WP3); NOT a production BC recipe",
            "wp4_scope": "full BC recipes + frozen thresholds deferred",
            "beats_zero_baseline": record["metrics"]["beats_zero_baseline"],
        },
        production_decision_open=False,
        schema_version=BUNDLE_SCHEMA_VERSION,
    )
    bundle.validate()
    return bundle


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--updates", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--lambda-a", type=float, default=LAMBDA_A,
                    help="weight of the deterministic-action MSE term")
    ap.add_argument("--nll-weight", type=float, default=1.0,
                    help="weight of the NLL term")
    ap.add_argument("--tag", default="",
                    help="suffix tag for output files (e.g. composition id)")
    ap.add_argument("--configs", nargs="*", default=ALL_CONFIGS)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    tag = f"_{args.tag}" if args.tag else ""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spec, manifest = load_spec_and_manifest()
    spec_digest = spec.content_digest()
    shard_digests = [r["npz_sha256"] for r in manifest["per_reaction"]]
    states = load_states()
    moving_thr = moving_atom_threshold(states)
    print(f"moving-atom threshold (q{MOVING_QUARTILE} of |a| over "
          f"non-teacher fit atoms): {moving_thr:.6f} A", flush=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    dtype = torch.float32
    print(f"device={device} dtype={dtype} torch={torch.__version__} "
          f"composition: loss = {args.nll_weight}*NLL + {args.lambda_a}*MSE",
          flush=True)

    records = []
    for config in args.configs:
        if config == "zero_baseline":
            rec = zero_baseline_record(states, moving_thr, device, dtype)
        else:
            rec = run_config(
                config, states=states, moving_thr=moving_thr,
                updates=args.updates, batch_size=args.batch_size,
                lr=args.lr, seed=args.seed, device=device, dtype=dtype,
                lambda_a=args.lambda_a, nll_weight=args.nll_weight,
                tag=tag)
            bundle = build_bundle(rec, spec_digest, shard_digests, args.seed)
            bundle_path = OUT_DIR / f"bundle_{config}{tag}.json"
            bundle_path.write_text(json.dumps(
                bundle.to_payload(), indent=2) + "\n")
            rec["bundle_file"] = bundle_path.name
            rec["bundle_content_digest"] = bundle.content_digest()
            rec["schema_versions"] = {
                "bundle": BUNDLE_SCHEMA_VERSION,
                "spec": "pesn_dual_ts_features_v3",
            }
        rec["spec_digest"] = spec_digest
        rec["environment"] = {
            "device": str(device), "dtype": str(dtype),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        (OUT_DIR / f"fit_{config}{tag}.json").write_text(
            json.dumps(rec, indent=2) + "\n")
        m = rec["metrics"]
        cos = m["direction_cosine_absa_weighted"]
        print(f"[{config:>13}] states={rec['population']['n_states']:4d} "
              f"loss_atoms={m['n_atoms_loss']:6d} "
              f"MSE={m['per_atom_action_mse_A2']:.6e} "
              f"baseline={m['zero_baseline_action_mse_A2']:.6e} "
              f"ratio={m['mse_over_baseline']:.3f} "
              f"cos={'n/a' if cos is None else format(cos, '+.4f')} "
              f"{'reference' if config == 'zero_baseline' else ('BEATS' if m['beats_zero_baseline'] else 'LOSES TO')}"
              f"{' (a=0)' if config == 'zero_baseline' else ' baseline'}",
              flush=True)
        records.append(rec)

    # summary table
    rows = ["| config | states | loss atoms | per-atom action MSE [A²] | "
            "zero baseline MSE [A²] | MSE/baseline | dir cosine (|a|-w) | "
            "verdict |",
            "|---|---|---|---|---|---|---|---|"]
    for rec in records:
        m = rec["metrics"]
        cos = m["direction_cosine_absa_weighted"]
        rows.append(
            f"| {rec['config']} | {rec['population']['n_states']} | "
            f"{m['n_atoms_loss']} | {m['per_atom_action_mse_A2']:.6e} | "
            f"{m['zero_baseline_action_mse_A2']:.6e} | "
            f"{m['mse_over_baseline']:.3f} | "
            f"{'n/a (a=0)' if cos is None else format(cos, '+.4f')} | "
            f"{'reference' if rec['config'] == 'zero_baseline' else ('beats baseline' if m['beats_zero_baseline'] else 'FAILS vs baseline (WP4 signal)')} |")
    summary_md = "\n".join(rows) + "\n"
    (OUT_DIR / f"summary{tag}.md").write_text(summary_md)
    (OUT_DIR / f"summary{tag}.json").write_text(json.dumps({
        "schema_version": "pesn_dual_ts_wp3b_smallfit_summary_v0",
        "tag": tag.lstrip("_") or "primary",
        "composition": {
            "nll_weight": args.nll_weight, "lambda_a": args.lambda_a,
            "formula": f"loss = {args.nll_weight} * NLL + "
                       f"{args.lambda_a} * action-MSE",
        },
        "spec_digest": spec_digest,
        "seeds": {"torch": args.seed, "numpy": args.seed,
                  "python": args.seed},
        "hyperparams_common": {
            "updates": args.updates, "batch_size_states": args.batch_size,
            "lr": args.lr, "lambda_a": args.lambda_a,
            "nll_weight": args.nll_weight,
            "model": DUAL_TS_EGNN_VERSION, "policy": DUAL_TS_POLICY_VERSION,
            "moving_threshold_A": moving_thr,
            "near_ts_arc_fraction": NEAR_TS_ARC_FRACTION,
        },
        "runs": [{"config": r["config"],
                  "metrics": r["metrics"],
                  "bundle_content_digest": r.get("bundle_content_digest"),
                  "checkpoint_sha256": r.get("checkpoint_sha256")}
                 for r in records],
    }, indent=2) + "\n")
    print(summary_md, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
