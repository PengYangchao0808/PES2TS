#!/usr/bin/env python3
"""V5.3-MP λ (lambda_rank/lambda_q/lambda_endpoint) gradient-share probe.

Mirrors ``r3_v52_lambda_probe.py`` discipline for the V5.3-MP three-term
PROCESS loss (plan §6: λ 用 encoder 梯度占比 2–10% 校准；不做网格):

* reference model: ``KeyAtomV53MPModel`` whose backbone is the frozen R3
  F2-bidir winner (fold0 x seed17, ``5372d9b12807c3f0``) — exactly the S1
  init — with fresh seeded heads;
* batches: the 2k tier's inner-train slice (seed 17 pool split), built
  through the sharded loader with FULL label statuses;
* per batch: encoder gradient norms of L_move (all λ=0 through the frozen
  v53mp entrypoint) and of each term (λ_term=1 minus move, by gradient
  linearity — the loss is never reimplemented);
* target: 2% encoder share (lower anchor of the 2–10% band); 20-batch
  median; λ = target / median-ratio, capped at 0.5, <1% at cap ⇒
  ``scale_unreachable`` (recorded; the term stays at 0 and the arm is
  blocked rather than forced).

Output: ``outputs/pesn/v53mp/v53mp_lambda_calibration.json`` — the ONLY
source for ``training_contract.lambda_probe.values`` in the campaign yaml
(manual fill; config_sha256 changes; plan-init --force afterwards).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agent_ops.gpu_queue.gpu_lock import acquire as gpu_acquire  # noqa: E402
from scripts.campaigns.r3_gradient_probe import (  # noqa: E402
    calibrate_lambda,
    median,
)

PROBE_SCHEMA = "pesn_v53mp_lambda_probe_v1"
N_BATCHES = 20
TARGET = 0.02
R3_WINNER_RUN_ID = "5372d9b12807c3f0"
R3_WINNER_CKPT = (
    "outputs/pesn/30h_campaign/r3_runs/{}/checkpoint_best.pt"
    .format(R3_WINNER_RUN_ID))
GPU_LOCK = "/tmp/opencode/pes2ts_gpu.lock"

HEAD_PREFIXES = ("process_head.",)


def _grad_norms(model) -> Tuple[float, float]:
    enc = head = 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        sq = float(param.grad.detach().pow(2).sum().cpu())
        if name.startswith(HEAD_PREFIXES):
            head += sq
        elif name.startswith("backbone."):
            enc += sq
    return enc ** 0.5, head ** 0.5


def _zero(model) -> None:
    for param in model.parameters():
        param.grad = None


def _term_grads(model, batch, loss_kwargs_base, terms: Tuple[str, ...],
                ) -> Dict[str, Tuple[float, float]]:
    """(encoder, head) grad norms per term by gradient linearity.

    ``∇L_term = ∇total(λ_term=1) − ∇total(0)`` computed ELEMENTWISE on the
    per-parameter grads (the frozen loss is never reimplemented; norms of
    differences, never differences of norms).  Each λ setting gets its OWN
    forward pass (the graph is never backwarded twice).
    """
    from pes2ts_core.pesn.models.key_atom_loss_v53mp_v3 import (
        key_atom_loss_v53mp,
    )
    from pes2ts_core.pesn.contracts.key_atom_v3 import forward_view_v3

    def grads_with(**lams):
        _zero(model)
        fwd = model(forward_view_v3(batch))
        out = key_atom_loss_v53mp(u=fwd["u"], q_pred=fwd["q"],
                                  endpoint_logit=fwd["endpoint_logit"],
                                  **loss_kwargs_base, **lams)
        out["loss"].backward()
        return {name: (param.grad.detach().clone() if param.grad is not None
                       else None)
                for name, param in model.named_parameters()}

    base = grads_with(lambda_rank=0.0, lambda_q=0.0, lambda_endpoint=0.0)

    def norm_of(grads, prefixes):
        total = 0.0
        for name, g in grads.items():
            if g is None:
                continue
            if any(name.startswith(p) for p in prefixes):
                total += float(g.pow(2).sum().cpu())
        return total ** 0.5

    base_enc = norm_of(base, ("backbone.",))
    base_head = norm_of(base, HEAD_PREFIXES)
    per_term: Dict[str, Tuple[float, float]] = {}
    for term in terms:
        lam = {t: 0.0 for t in ("lambda_rank", "lambda_q", "lambda_endpoint")}
        lam[term] = 1.0
        full = grads_with(**lam)
        diff = {name: (full[name] - base[name]
                       if (full[name] is not None and base[name] is not None)
                       else full[name] if full[name] is not None
                       else None)
                for name in full}
        per_term[term] = (norm_of(diff, ("backbone.",)),
                          norm_of(diff, HEAD_PREFIXES))
    per_term["__move__"] = (base_enc, base_head)
    return per_term


def run_probe(tables_dir: str, split_path: str, device: str,
              out_path: str) -> dict:
    import torch

    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (
        LABEL_STATUSES,
        ShardedKeyAtomDatasetV3,
    )
    from pes2ts_core.pesn.offline.data.key_atom_endpoint_labels_v1 import (
        build_endpoint_labels_spec_v1,
    )
    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        v53mp_pool_split,
    )
    from pes2ts_core.pesn.models.key_atom_egnn_v3 import StarEGNNV3
    from pes2ts_core.pesn.models.key_atom_heads_v3 import KeyAtomV53MPModel
    from pes2ts_core.pesn.offline.training.key_atom_bc_v3 import set_seed
    from pes2ts_core.pesn.features.key_atom_graph_v3 import (
        build_sample_features_v3,
        collate_v3,
        to_torch_batch,
    )
    from pes2ts_core.pesn.contracts.key_atom_v3 import (
        NODE_DEGK_KEY,
        NODE_KEY_SET_KEY,
        NODE_POS_KEY,
        forward_view_v3,
    )

    dataset = ShardedKeyAtomDatasetV3(
        tables_dir, label_statuses=LABEL_STATUSES,
        split_manifest=split_path, light_cache=True)
    # labels are pure functions of the frozen tables: use the S0-finalize
    # cache (bound to the data-manifest sha); never silently recompute at
    # 20k scale
    from pes2ts_core.pesn.offline.data.key_atom_label_cache_v53mp import (
        LabelCacheError,
        load_label_cache,
    )
    try:
        labels, endpoint_labels = load_label_cache(
            tables_dir,
            dataset_manifest_sha256=dataset.manifest_sha256)
    except LabelCacheError:
        labels = dataset.v52_labels()
        endpoint_labels = build_endpoint_labels_spec_v1(dataset, labels)
    split_doc = json.loads(Path(split_path).read_text(encoding="utf-8"))
    pool_2k = [str(c) for c in split_doc["subsets"]["pool"]["2k"]]
    samples = dataset.samples
    train_pos, _val_pos = v53mp_pool_split(
        samples, pool_curve_ids=pool_2k, seed=17)

    set_seed(17)
    backbone = StarEGNNV3()
    payload = torch.load(R3_WINNER_CKPT, map_location="cpu",
                         weights_only=False)
    backbone.load_state_dict(payload["model"], strict=True)
    model = KeyAtomV53MPModel(backbone=backbone).to(device)

    rng = __import__("numpy").random.RandomState(17 + 52)
    # endpoint pos_weight from the inner-train pool (the same structural
    # estimate the trainer makes at run start — never from dev)
    y_all = []
    for i in train_pos:
        lab = endpoint_labels.by_sample.get(str(samples[int(i)].sample_id))
        if lab is not None and int(lab.endpoint_label) in (0, 1):
            y_all.append(int(lab.endpoint_label))
    n_pos = int(sum(y_all))
    n_neg = int(len(y_all) - n_pos)
    endpoint_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
    terms = ("lambda_rank", "lambda_q", "lambda_endpoint")
    ratios: Dict[str, List[float]] = {t: [] for t in terms}
    t0 = time.time()
    for step in range(N_BATCHES):
        # curve-balanced batch (same 8x4 recipe as the trainer)
        from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
            _curve_batch_positions,
            _endpoint_for_batch,
            _labels_for_batch,
            sample_order_pairs,
        )
        sampler_cfg = __import__("types").SimpleNamespace(
            batch_curves=8, frames_per_curve=4)
        positions = _curve_batch_positions(
            samples, train_pos, config=sampler_cfg, rng=rng)
        built = []
        for p in positions:
            feat = build_sample_features_v3(
                dataset.to_sample_v3(samples[int(p)]),
                level="F2", intent_off=False)
            # zero-pad y_teacher for terminal/masked rows (same discipline
            # as the trainer's _FeatureLRU — mixed batches never lose the
            # key; the frame_mask/weights exclude them from the move loss)
            if feat.get("y_teacher") is None:
                n_key = len(samples[int(p)].intent.key_rows())
                feat = dict(feat)
                feat["y_teacher"] = np.zeros((n_key, 3), dtype=np.float32)
            built.append(feat)
        batch = to_torch_batch(collate_v3(built), device=device)
        from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
            _frame_mask_for,
        )
        frame_mask = _frame_mask_for(
            [samples[int(p)] for p in positions], device)
        sample_ids = [str(s) for s in batch["sample_ids"]]
        lab = _labels_for_batch(sample_ids, labels, device)
        ep = _endpoint_for_batch(sample_ids, endpoint_labels, device)
        order_pairs = sample_order_pairs(
            batch, labels, order_min_gap=0.05, max_pairs=256, rng=rng)
        n_keys = batch["n_keys"].long()
        key_ptr = __import__("torch").cat(
            [__import__("torch").zeros(
                1, dtype=__import__("torch").long, device=n_keys.device),
             __import__("torch").cumsum(n_keys, dim=0)])
        # forward/backward per λ setting happens INSIDE _term_grads (one
        # graph per setting; never backwarded twice).  loss_kwargs_base
        # carries only the batch/label tensors (model outputs are produced
        # by the per-setting forward).
        loss_kwargs = dict(
            y=batch["y_teacher"],
            degK=batch[NODE_DEGK_KEY][batch[NODE_KEY_SET_KEY]],
            key_ptr=key_ptr, pos=batch[NODE_POS_KEY],
            key_index=batch[NODE_KEY_SET_KEY], pair_index=batch["pair_index"],
            q_true=lab["q_true"], q_mask=lab["q_mask"],
            order_pairs=order_pairs,
            y_endpoint=ep["y_endpoint"],
            endpoint_mask=ep["endpoint_mask"],
            endpoint_pos_weight=endpoint_pos_weight,
            lambda_cos=0.02,
            frame_mask=frame_mask,
            frame_weight=batch["frame_weight"],
            pair_edit=batch["pair_edit"],
        )
        per_term = _term_grads(model, batch, loss_kwargs, terms)
        move_enc = per_term["__move__"][0]
        for term in terms:
            term_enc = per_term[term][0]
            ratios[term].append(term_enc / (move_enc + 1e-12))
    result_terms: Dict[str, Any] = {}
    for term in terms:
        med = median(ratios[term])
        cal = calibrate_lambda(med, TARGET)
        result_terms[term.replace("lambda_", "")] = {
            "lambda": cal["lambda"],
            "median_encoder_ratio_at_1": med,
            "achieved_encoder_share": cal["share_at_lambda"],
            "verdict": cal["verdict"],
        }
    doc = {
        "schema": PROBE_SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device,
        "tables_dir": tables_dir,
        "split": split_path,
        "reference_init": R3_WINNER_RUN_ID,
        "n_batches": N_BATCHES,
        "target_encoder_share": TARGET,
        "terms": result_terms,
        "wall_clock_s": round(time.time() - t0, 1),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")
    return doc


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="r3_v53mp_lambda_probe",
        description="Calibrate V5.3-MP lambdas from encoder gradient shares "
                    "(the ONLY lambda source for the campaign yaml)")
    parser.add_argument("--tables-dir", default="outputs/pesn/v53mp/data")
    parser.add_argument("--split", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out",
                        default="outputs/pesn/v53mp/v53mp_lambda_calibration.json")
    args = parser.parse_args(argv)
    split = args.split or str(Path(args.tables_dir) / "split_manifest_v53mp.json")
    handle = gpu_acquire(GPU_LOCK, lane="V53MP-lambda-probe",
                         run_id="v53mp-lambda-probe", cuda=True)
    if handle is None:
        sys.stderr.write("error: GPU lock busy\n")
        return 1
    try:
        doc = run_probe(args.tables_dir, split, args.device, args.out)
    finally:
        handle.release()
    print(json.dumps({k: v["lambda"] for k, v in doc["terms"].items()},
                     sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
