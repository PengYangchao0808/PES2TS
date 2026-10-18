"""V4.2 λ calibration canary (Lane C; contract §5 / ADR-0024 item 3).

Fresh-inits the H0 (=D3) displacement model on fit45_train native data, draws
32 fixed group-balanced batches (seed 20261017), and backprops ``L_xyz`` /
``L_dir`` / ``L_mag`` separately through the SHARED encoder parameters to get
their gradient-norm medians, then sets::

    λ_dir = 0.3 · med‖∇L_xyz‖ / med‖∇L_dir‖
    λ_mag = 0.1 · med‖∇L_xyz‖ / med‖∇L_mag‖

An invalid/zero denominator (or a zero/non-finite median) exits non-zero.
The result (batch-list hash, per-loss norm medians, λ values) is written to
``wp42_campaign/lambda_canary.json`` and consumed by the driver for H2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.offline.training.dual_ts_campaign import (  # noqa: E402
    CampaignArmConfig,
    CampaignDatasetSpec,
    CampaignTrainer,
    load_campaign_states,
    load_dev_split,
)

WP42 = PROJECT_ROOT / "outputs/pesn/dual_ts_v2_20261015/wp42_campaign"
DEV_SPLIT_PATH = WP42 / "dev_split.json"
NATIVE_FEATURES = WP42 / "native_features"
OUT_PATH = WP42 / "lambda_canary.json"

N_BATCHES = 32
SEED = 20261017
CANARY_SCHEMA = "pesn_dual_ts_lambda_canary_v1"


def _h0_arm() -> CampaignArmConfig:
    return CampaignArmConfig(
        arm_id="H0",
        dataset=CampaignDatasetSpec(
            kind="native", features_dir=str(NATIVE_FEATURES),
            labels_dir="", split_path=str(DEV_SPLIT_PATH)),
        head="displacement", use_prior_skip=False,
        features_version="v3", encoder="egnn", vector_channels=16,
        contact_edges=False, hidden=64, n_layers=3,
        sampler="group_balanced_v41", preset="campaign_v1",
        seed=SEED, updates=20000, batch_states=64,
        device="cuda:0", require_cuda=True,
    )


def _encoder_params(trainer: CampaignTrainer) -> list:
    model = trainer.model
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return list(model.parameters())
    if hasattr(encoder, "parameters"):
        return list(encoder.parameters())
    return []


def _grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float((p.grad.detach() ** 2).sum())
    return math.sqrt(total)


def _batch_hash(batch_state_ids: list[list[str]]) -> str:
    h = hashlib.sha256()
    for ids in batch_state_ids:
        h.update(("|".join(ids) + "\n").encode())
    return h.hexdigest()


def calibrate() -> dict:
    split = load_dev_split(str(DEV_SPLIT_PATH))
    fit_groups = [str(g) for g in split["train_group_ids"]]
    states, meta = load_campaign_states(
        CampaignDatasetSpec(kind="native", features_dir=str(NATIVE_FEATURES)),
        group_ids=fit_groups)
    meta["split_sha256"] = str(split.get("split_sha256", ""))

    trainer = CampaignTrainer(
        arm=_h0_arm(), states=states, data_meta=meta,
        run_dir=WP42 / "_canary_tmp", dev_states=[])
    encoder_params = _encoder_params(trainer)

    norms = {"l_xyz": [], "l_dir": [], "l_mag": []}
    batch_ids: list[list[str]] = []
    for _ in range(N_BATCHES):
        rows = trainer.sampler.draw_batch(trainer.arm.batch_states)
        chunk = [trainer.drawable[i] for i in rows]
        batch_ids.append([s["state_id"] for s in chunk])
        inputs = trainer._collate(chunk)
        targets = trainer._targets(chunk)
        out = trainer._forward(inputs, targets)
        comps = trainer.compute_loss_components(targets, out)
        for key in ("l_xyz", "l_dir", "l_mag"):
            trainer.optimizer.zero_grad(set_to_none=True)
            comps[key].backward(retain_graph=True)
            norms[key].append(_grad_norm(encoder_params))

    med = {k: float(np.median(np.asarray(v))) for k, v in norms.items()}
    # explicit exit on invalid/near-zero denominator
    for k in ("l_xyz", "l_dir", "l_mag"):
        if not np.isfinite(med[k]) or med[k] <= 1e-12:
            print(f"REFUSED: {k} gradient-norm median is non-finite/zero "
                  f"({med[k]!r})", file=sys.stderr)
            return None
    lam_dir = 0.3 * med["l_xyz"] / med["l_dir"]
    lam_mag = 0.1 * med["l_xyz"] / med["l_mag"]
    payload = {
        "schema_version": CANARY_SCHEMA,
        "arm_id": "H0", "seed": SEED, "n_batches": N_BATCHES,
        "batch_list_sha256": _batch_hash(batch_ids),
        "grad_norm_medians": med,
        "lambda_dir": float(lam_dir),
        "lambda_mag": float(lam_mag),
        "rule": "lambda_dir=0.3*med||dL_xyz||/med||dL_dir||; "
                "lambda_mag=0.1*med||dL_xyz||/med||dL_mag|| (shared encoder params)",
        "split_sha256": meta.get("split_sha256", ""),
    }
    OUT_PATH.write_text(
        json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args(argv)
    payload = calibrate()
    if payload is None:
        return 1
    print(f"[canary] lambda_dir={payload['lambda_dir']:.4f} "
          f"lambda_mag={payload['lambda_mag']:.4f} "
          f"med||grad||={{l_xyz:{payload['grad_norm_medians']['l_xyz']:.4e}, "
          f"l_dir:{payload['grad_norm_medians']['l_dir']:.4e}, "
          f"l_mag:{payload['grad_norm_medians']['l_mag']:.4e}}}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["calibrate", "main"]
