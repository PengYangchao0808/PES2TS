"""V5.2 λq/λm gradient-share probe (plan §5.4 formalism; W-LOSS E5).

The v52 yaml freezes the loss form ``L_move + λq·L_q + λm·L_TSStop`` but
NOT the λq/λm numbers (E4 handoff §7.2: calibration is E5 work per plan
§5.4; loss defaults λq=λm=0 = move-only).  This probe documents the only
λ source for the V5.2 queue — never guessed:

* reference: the matched B0-F checkpoint (pre-registered R3 winner
  F2-bidir fold0, seed-matched) wrapped in ``KeyAtomV52Model`` with fresh
  seeded heads (exactly the E5 init);
* batches: the fold0 V5.2 TRAINING slice (dev folds != 0, inner-train
  portion of the seed split — the E4-coherent protocol the runs use);
* per batch: encoder/head gradient norms of L_move (λq=λm=0 through the
  FROZEN v52 entrypoint), L_q (∇ at λq=1 minus move) and L_TSStop
  (∇ at λm=1 minus move) — gradient linearity, no loss reimplementation;
* target: 2% encoder share (the lower anchor of the plan §5.4 2%/10%
  band — no10% V5.2 variants exist in the frozen run matrix; documented
  choice), 20-batch median per seed, campaign λ = target / median-ratio,
  capped at 0.5, <1% at cap => scale_unreachable (recorded, arm blocked).

Output: ``outputs/pesn/30h_campaign/v52_runs/v52_lambda_calibration.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agent_ops.gpu_queue.gpu_lock import acquire as gpu_acquire  # noqa: E402
from scripts.agent_ops.supervisor.campaign_root import (  # noqa: E402
    CAMPAIGN_ROOT,
    CampaignRootError,
    assert_campaign_root,
)
from scripts.campaigns.r3_gradient_probe import (  # noqa: E402
    GPU_LOCK,
    bucket_ratios,
    calibrate_lambda,
    median,
)

PROBE_SCHEMA = "pesn_v52_lambda_probe_v1"
N_BATCHES = 20
TARGET = 0.02
B0F_REFERENCE_RUNS = {17: "5372d9b12807c3f0", 29: "1e01499b933d4ad1",
                      43: "9e2ef3ef468db0af"}


def _grad_norms(model) -> Tuple[float, float]:
    """(encoder, head) gradient norms on a KeyAtomV52Model."""
    enc = head = 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        sq = float(param.grad.detach().pow(2).sum().cpu())
        if name.startswith(("progress_head.", "ts_stop_head.")):
            head += sq
        elif name.startswith("backbone."):
            enc += sq
    return enc ** 0.5, head ** 0.5


def measure_term_ratios(
    model, batch, labels_batch, *, term: str, stop_weights, lambda_cos: float
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """((move_enc, move_head), (term_enc, term_head)) gradient norms.

    Bucket/term isolation by gradient linearity with the frozen loss:
    ``∇L_term = ∇total(λ_term=1) − ∇total(0)`` computed elementwise on the
    per-parameter grads (norms alone would not compose).
    """
    from pes2ts_core.pesn.models.key_atom_loss_v52_v3 import key_atom_loss_v52
    from pes2ts_core.pesn.offline.training.key_atom_v52_train_v3 import (
        _key_ptr,
    )
    from pes2ts_core.pesn.contracts.key_atom_v3 import (
        NODE_DEGK_KEY,
        NODE_KEY_SET_KEY,
        NODE_POS_KEY,
        forward_view_v3,
    )

    def _zero():
        for p in model.parameters():
            p.grad = None

    def _grads(lq: float, lm: float):
        _zero()
        out = model(forward_view_v3(batch))
        out_loss = key_atom_loss_v52(
            out["u"], batch["y_teacher"],
            batch[NODE_DEGK_KEY][batch[NODE_KEY_SET_KEY]],
            _key_ptr(batch), batch[NODE_POS_KEY], batch[NODE_KEY_SET_KEY],
            batch["pair_index"],
            q_pred=out["q"], q_true=labels_batch["q_true"],
            q_mask=labels_batch["q_mask"],
            stop_logit=out["stop_logit"], y_stop=labels_batch["y_stop"],
            stop_mask=labels_batch["stop_mask"],
            stop_class_weights=stop_weights if lm != 0.0 else None,
            lambda_q=lq, lambda_stop=lm, lambda_cos=lambda_cos,
            frame_weight=batch["frame_weight"], pair_edit=batch["pair_edit"])
        out_loss["loss"].backward()
        return {n: (p.grad.detach().clone() if p.grad is not None else None)
                for n, p in model.named_parameters()}

    g_move = _grads(0.0, 0.0)
    g_full = _grads(1.0, 0.0) if term == "q" else _grads(0.0, 1.0)

    def _norms(gmap) -> Tuple[float, float]:
        enc = head = 0.0
        for name, grad in gmap.items():
            if grad is None:
                continue
            sq = float(grad.pow(2).sum().cpu())
            if name.startswith(("progress_head.", "ts_stop_head.")):
                head += sq
            elif name.startswith("backbone."):
                enc += sq
        return enc ** 0.5, head ** 0.5

    diff = {}
    for name, grad in g_full.items():
        base = g_move.get(name)
        if grad is None:
            continue
        diff[name] = grad if base is None else (grad - base)
    return _norms(g_move), _norms(diff)


def run_probe(args: argparse.Namespace) -> Dict[str, Any]:
    import numpy as np
    import torch
    from pes2ts_core.pesn.features.key_atom_graph_v3 import (
        collate_v3,
        to_torch_batch,
    )
    from pes2ts_core.pesn.models.key_atom_heads_v3 import KeyAtomV52Model
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (
        KeyAtomDatasetV3,
    )
    from pes2ts_core.pesn.offline.training.key_atom_bc_v3 import (
        build_feature_cache,
        set_seed,
    )
    from pes2ts_core.pesn.offline.training.key_atom_v52_train_v3 import (
        _curve_batch_positions,
        _labels_for_batch,
        load_backbone_from_r3,
        stop_class_weights_for_fold,
        v52_curve_split,
    )

    dataset = KeyAtomDatasetV3(
        args.data_dir, split_manifest=args.split, label_statuses=("ok",))
    labels = dataset.v52_labels()
    runs_root = Path(args.runs_root)
    per_seed: Dict[str, Any] = {}
    for seed in (17, 29, 43):
        ref_run = B0F_REFERENCE_RUNS[seed]
        ckpt = runs_root.parent / "r3_runs" / ref_run / "checkpoint_best.pt"
        if not ckpt.is_file():
            raise SystemExit("B0-F checkpoint missing: {}".format(ckpt))
        set_seed(seed)
        model = load_backbone_from_r3(ckpt, device=args.device)
        for p in model.parameters():
            p.requires_grad_(True)
        model.train(True)
        stop_weights = stop_class_weights_for_fold(0)
        train_idx, _, _ = v52_curve_split(dataset.samples, 0, seed)
        cache = build_feature_cache(
            dataset, train_idx, level="F2", intent_off=False)
        rng = np.random.RandomState(seed + 52)
        ratios = {"q": {"enc": [], "head": []},
                  "stop": {"enc": [], "head": []}}
        t0 = time.time()
        for _ in range(args.batches):
            positions = _curve_batch_positions(
                dataset.samples, train_idx, seed=seed, rng_state=rng)
            batch = to_torch_batch(
                collate_v3([cache[int(p)] for p in positions]),
                device=args.device)
            lab = _labels_for_batch(batch, labels, args.device)
            for term in ("q", "stop"):
                (menc, mhead), (tenc, thead) = measure_term_ratios(
                    model, batch, lab, term=term, stop_weights=stop_weights,
                    lambda_cos=0.02)
                r_enc, r_head = bucket_ratios(menc, tenc, mhead, thead)
                ratios[term]["enc"].append(r_enc)
                ratios[term]["head"].append(r_head)
        per_seed[str(seed)] = {
            "reference_run_id": ref_run,
            "elapsed_s": round(time.time() - t0, 2),
            **{f"{term}_{g}": median(ratios[term][g])
               for term in ("q", "stop") for g in ("enc", "head")},
        }
    arms: Dict[str, Any] = {}
    for term in ("q", "stop"):
        med = median([per_seed[str(s)][f"{term}_enc"] for s in (17, 29, 43)])
        cal = calibrate_lambda(med, TARGET)
        head_ratios = [per_seed[str(s)][f"{term}_head"] for s in (17, 29, 43)]
        lam = cal["lambda"]
        arms[term] = {
            "target_encoder_share": TARGET,
            "median_encoder_ratio_across_seeds": med,
            "lambda": cal["lambda"],
            "achieved_encoder_share": cal["share_at_lambda"],
            "verdict": cal["verdict"],
            "head_shares_at_lambda": [
                None if lam is None else lam * h for h in head_ratios],
        }
    return {
        "schema": PROBE_SCHEMA,
        "reference": ("B0-F = R3 winner F2-bidir fold0, seed-matched; heads "
                      "fresh-seeded; batches = fold0 V5.2 train slice "
                      "(dev folds != 0, inner-train); frozen v52 loss only"),
        "n_batches": args.batches,
        "target": TARGET,
        "target_rationale": ("lower anchor of plan 5.4 2%/10% band; the "
                             "frozen V5.2 run matrix has no 10% variants"),
        "per_seed": per_seed,
        "terms": arms,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="r3_v52_lambda_probe",
        description="Calibrate V5.2 lambda_q/lambda_m from encoder gradient "
        "shares (plan 5.4 formalism; the ONLY lambda source for E5)")
    parser.add_argument("--runs-root", default=CAMPAIGN_ROOT + "/v52_runs")
    parser.add_argument("--data-dir", default=CAMPAIGN_ROOT + "/data")
    parser.add_argument("--split",
                        default=CAMPAIGN_ROOT + "/data/split_manifest_r3.json")
    parser.add_argument("--out", default=None)
    parser.add_argument("--batches", type=int, default=N_BATCHES)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    try:
        assert_campaign_root(str(Path(args.runs_root).parent))
    except CampaignRootError as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 1
    handle = gpu_acquire(GPU_LOCK, lane="W-LOSS-E5",
                         run_id="v52-lambda-probe", cuda=True)
    if handle is None:
        sys.stderr.write("error: GPU lock busy\n")
        return 1
    try:
        result = run_probe(args)
    finally:
        handle.release()
    out = (Path(args.out) if args.out
           else Path(args.runs_root) / "v52_lambda_calibration.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps({"out": str(out),
                      "terms": {t: {"lambda": v["lambda"],
                                    "verdict": v["verdict"],
                                    "share": v["achieved_encoder_share"]}
                                for t, v in result["terms"].items()}},
                     sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
