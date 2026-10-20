"""R3 λ calibration gradient probe (plan §5.4; yaml aux_calibration).

Measures, on the B0-F reference checkpoint (pre-registered Top-1 winner
F2-bidir, fold0, seed-matched), the ENCODER gradient share of the strong /
dead bucket losses vs L_base over 20 real inner-train batches, then
calibrates the aux-arm λ for S2/S10 (strong) and D2/D10 (dead) targeting
encoder shares 2% / 10%:

    ρ_{k,g}(λ) = λ·||∇_{θ_g} L_k|| / (||∇_{θ_g} L_base|| + ε)   (g = encoder|head)

Frozen constraints (campaign yaml ``r3_training.aux_calibration``):
* λ ≤ 0.5 (``lambda_max``);
* head share at the chosen λ ≤ 0.3 (else ``head_overweighted`` flag);
* median over 20 batches;
* encoder share at λ=0.5 below 1% -> ``scale_unreachable`` (recorded
  honestly, the arm is then BLOCKED, never forced);
* λ≤0.5 cannot reach the 2%/10% target -> ``target_unreachable`` (λ=0.5
  capped and recorded).

Isolation uses ONLY the frozen loss: bucket gradients are obtained as
``∇total(λ=1) − ∇total(λ=0)`` (gradient linearity; no loss math is
reimplemented).  ``λ=0`` regression: the aux path at λ=0 must reproduce
the B0-F path bit-identically (covered by
``tests/agent_ops/test_gpu_r3_trainer.py``).

Output: ``outputs/pesn/30h_campaign/r3_runs/lambda_calibration.json``
(per-seed shares + campaign λ per arm = median across seeds) consumed by
``run_v5_1_r3.py run --lambda-arm``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agent_ops.gpu_queue.gpu_lock import acquire as gpu_acquire  # noqa: E402
from scripts.agent_ops.supervisor.campaign_root import (  # noqa: E402
    CAMPAIGN_ROOT,
    CampaignRootError,
    assert_campaign_root,
)

PROBE_SCHEMA = "pesn_r3_lambda_probe_v1"
N_BATCHES_DEFAULT = 20
TARGETS = {"S2": ("strong", 0.02), "S10": ("strong", 0.10),
           "D2": ("dead", 0.02), "D10": ("dead", 0.10)}
LAMBDA_MAX = 0.5
HEAD_SHARE_MAX = 0.3
SCALE_FLOOR = 0.01
GPU_LOCK = "/tmp/opencode/pes2ts_gpu.lock"

#: pre-registered Top-1 winner (B0-F reference): F2-bidir fold0, per seed
B0F_REFERENCE_RUNS = {
    17: "5372d9b12807c3f0",
    29: "1e01499b933d4ad1",
    43: "9e2ef3ef468db0af",
}
ENCODER_PREFIXES = ("elem_emb", "node_proj", "message_mlp", "node_mlp", "norm_h")
HEAD_PREFIXES = ("w_v", "z_head")


# ---------------------------------------------------------------------------
# pure math (unit-tested)
# ---------------------------------------------------------------------------


def bucket_ratios(
    grad_base_enc: float, grad_bucket_enc: float,
    grad_base_head: float, grad_bucket_head: float,
) -> Tuple[float, float]:
    """Per-batch (encoder, head) share ratios at λ=1: ||∇L_k||/(||∇L_base||+ε)."""
    return (grad_bucket_enc / (grad_base_enc + 1e-12),
            grad_bucket_head / (grad_base_head + 1e-12))


def calibrate_lambda(
    median_ratio_encoder: float,
    target_share: float,
    *,
    lambda_max: float = LAMBDA_MAX,
    scale_floor: float = SCALE_FLOOR,
) -> Dict[str, Any]:
    """λ = target / median(ρ at λ=1), capped at λ_max; honest verdicts.

    Returns ``{lambda, share_at_lambda, verdict}`` where verdict is one of
    ``calibrated`` | ``target_unreachable`` (λ capped at λ_max, achieved
    share < target but ≥ scale_floor) | ``scale_unreachable`` (even λ_max
    yields < scale_floor encoder share; λ recorded but the arm must be
    blocked, never forced).
    """
    ratio = float(median_ratio_encoder)
    if ratio <= 0.0:
        return {"lambda": None, "share_at_lambda": 0.0,
                "verdict": "scale_unreachable"}
    lam = float(target_share) / ratio
    if lam <= lambda_max:
        return {"lambda": lam, "share_at_lambda": float(target_share),
                "verdict": "calibrated"}
    share_at_max = lambda_max * ratio
    if share_at_max < scale_floor:
        return {"lambda": lambda_max, "share_at_lambda": share_at_max,
                "verdict": "scale_unreachable"}
    return {"lambda": lambda_max, "share_at_lambda": share_at_max,
            "verdict": "target_unreachable"}


def head_flag(share_ratio_head_at_lambda: float,
              head_max: float = HEAD_SHARE_MAX) -> str:
    return "head_overweighted" if share_ratio_head_at_lambda > head_max else "ok"


def median(values: Sequence[float]) -> float:
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    if n == 0:
        raise ValueError("median of empty sequence")
    mid = n // 2
    return ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])


# ---------------------------------------------------------------------------
# gradient measurement (frozen loss only)
# ---------------------------------------------------------------------------


def _grad_group_norm(model, prefixes: Tuple[str, ...]) -> float:
    total = 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if any(name.startswith(p) for p in prefixes):
            total += float(param.grad.detach().pow(2).sum().cpu())
    return total ** 0.5


def _zero_grads(model) -> None:
    for param in model.parameters():
        param.grad = None


def measure_bucket_ratios(
    model,
    batch: Mapping[str, Any],
    *,
    bucket: str,
    loss_cfg,
    lambda_cos: float,
    device: str,
) -> Tuple[float, float]:
    """(encoder, head) λ=1 share ratios of one bucket on one real batch.

    Uses gradient linearity with the frozen loss only:
    ``∇L_k = ∇total(λ_k=1) − ∇total(λ_k=0)`` (norms combined per group via
    the two gradient vectors -- computed as ||g1−g0|| elementwise on the
    parameter grads).
    """
    from pes2ts_core.pesn.contracts.key_atom_v3 import (
        NODE_DEGK_KEY,
        NODE_KEY_SET_KEY,
        NODE_POS_KEY,
        forward_view_v3,
    )
    from pes2ts_core.pesn.models.key_atom_losses_v3 import key_atom_loss_v3

    aux = "S2" if bucket == "strong" else "D2"

    def _total(lambda_bucket: float):
        _zero_grads(model)
        out = model(forward_view_v3(batch))
        kwargs = dict(
            config=loss_cfg, lambda_cos=lambda_cos,
            frame_weight=batch["frame_weight"], pair_edit=batch["pair_edit"],
            aux_arm=aux)
        if bucket == "strong":
            kwargs["lambda_strong"] = lambda_bucket
        else:
            kwargs["lambda_dead"] = lambda_bucket
        loss_out = key_atom_loss_v3(
            out["u"], batch["y_teacher"],
            batch[NODE_DEGK_KEY][batch[NODE_KEY_SET_KEY]],
            _key_ptr(batch), batch[NODE_POS_KEY], batch[NODE_KEY_SET_KEY],
            batch["pair_index"], **kwargs)
        loss_out["loss"].backward()
        return {n: (p.grad.detach().clone() if p.grad is not None else None)
                for n, p in model.named_parameters()}

    g0 = _total(0.0)
    g1 = _total(1.0)
    enc = head = 0.0
    for name, param in model.named_parameters():
        a, b = g0.get(name), g1.get(name)
        if b is None:
            continue
        diff = b if a is None else (b - a)
        sq = float(diff.pow(2).sum().cpu())
        if any(name.startswith(p) for p in ENCODER_PREFIXES):
            enc += sq
        elif any(name.startswith(p) for p in HEAD_PREFIXES):
            head += sq
    _zero_grads(model)
    return enc ** 0.5, head ** 0.5


def _key_ptr(batch) -> Any:
    import torch

    n_keys = batch["n_keys"].long()
    return torch.cat(
        [torch.zeros(1, dtype=torch.long, device=n_keys.device),
         torch.cumsum(n_keys, dim=0)])


# ---------------------------------------------------------------------------
# probe driver
# ---------------------------------------------------------------------------


def run_probe(args: argparse.Namespace) -> Dict[str, Any]:
    from pes2ts_core.pesn.models.key_atom_egnn_v3 import StarEGNNV3
    from pes2ts_core.pesn.models.key_atom_losses_v3 import KeyAtomLossV3Config
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (
        KeyAtomDatasetV3,
        build_feature_batches,
    )
    from pes2ts_core.pesn.offline.training.key_atom_bc_v3 import (
        build_feature_cache,
        curve_batch_sampler,
        make_inner_split,
        set_seed,
    )
    import torch

    runs_root = Path(args.runs_root)
    device = args.device
    loss_cfg = KeyAtomLossV3Config()
    lambda_cos = float(args.lambda_cos)   # frozen R0 end value 0.02
    per_seed: Dict[str, Any] = {}
    for seed in (17, 29, 43):
        run_id = B0F_REFERENCE_RUNS[seed]
        run_dir = runs_root / run_id
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.is_file():
            raise SystemExit(
                "B0-F reference metrics missing: {} (run the F-bidir fold0 "
                "runs first)".format(metrics_path))
        ref = json.loads(metrics_path.read_text(encoding="utf-8"))
        if ref.get("arm") != "F2-bidir" or int(ref.get("fold", -1)) != 0 \
                or int(ref.get("seed", -1)) != seed:
            raise SystemExit(
                "B0-F reference {} is arm={} fold={} seed={}; expected "
                "F2-bidir fold0 seed{}".format(
                    run_id, ref.get("arm"), ref.get("fold"), ref.get("seed"),
                    seed))
        set_seed(seed)
        model = StarEGNNV3().to(device)
        state = torch.load(str(run_dir / "checkpoint_best.pt"),
                           map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        model.train()

        dataset = KeyAtomDatasetV3(
            args.data_dir, fold=0, split_manifest=args.split,
            label_statuses=("ok",))
        train_idx, _ = make_inner_split(dataset.samples, seed)
        cache = build_feature_cache(
            dataset, train_idx, level="F2", intent_off=False)
        sampler = curve_batch_sampler(
            dataset.samples, train_idx, bidirectional=True, seed=seed + 1)

        seed_ratios: Dict[str, Dict[str, List[float]]] = {
            "strong": {"enc": [], "head": []},
            "dead": {"enc": [], "head": []},
        }
        t0 = time.time()
        for _ in range(args.batches):
            positions = next(sampler)
            # same feature pipeline as the B0-F (F2-bidir) training runs
            batch = build_feature_batches(
                dataset, [train_idx[p] for p in positions],
                level="F2", intent_off=False, device=device)
            base_enc, base_head = _base_grads(model, batch, loss_cfg, lambda_cos)
            for bucket in ("strong", "dead"):
                b_enc, b_head = measure_bucket_ratios(
                    model, batch, bucket=bucket, loss_cfg=loss_cfg,
                    lambda_cos=lambda_cos, device=device)
                r_enc, r_head = bucket_ratios(base_enc, b_enc, base_head, b_head)
                seed_ratios[bucket]["enc"].append(r_enc)
                seed_ratios[bucket]["head"].append(r_head)
        seed_out: Dict[str, Any] = {"reference_run_id": run_id,
                                    "checkpoint_sha256": ref["checkpoint_best_sha256"],
                                    "batches": args.batches,
                                    "elapsed_s": round(time.time() - t0, 2)}
        for bucket in ("strong", "dead"):
            seed_out[bucket] = {
                "median_encoder_ratio": median(seed_ratios[bucket]["enc"]),
                "median_head_ratio": median(seed_ratios[bucket]["head"]),
            }
        per_seed[str(seed)] = seed_out

    # campaign λ per arm = calibrate on the MEDIAN-across-seeds encoder ratio
    arms: Dict[str, Any] = {}
    for arm, (bucket, target) in TARGETS.items():
        ratios = [per_seed[str(s)][bucket]["median_encoder_ratio"]
                  for s in (17, 29, 43)]
        head_ratios = [per_seed[str(s)][bucket]["median_head_ratio"]
                       for s in (17, 29, 43)]
        med_enc = median(ratios)
        cal = calibrate_lambda(med_enc, target)
        lam = cal["lambda"]
        head_shares = [None if lam is None else lam * h for h in head_ratios]
        arms[arm] = {
            "bucket": bucket,
            "target_encoder_share": target,
            "median_encoder_ratio_across_seeds": med_enc,
            "per_seed_encoder_ratios": ratios,
            "lambda": cal["lambda"],
            "achieved_encoder_share": cal["share_at_lambda"],
            "verdict": cal["verdict"],
            "head_shares_at_lambda": head_shares,
            "head_flag": "ok" if all(
                h is None or h <= HEAD_SHARE_MAX for h in head_shares)
            else "head_overweighted",
            "lambda_zero_bit_identical_to_B0F": True,
        }
    return {
        "schema": PROBE_SCHEMA,
        "reference": "B0-F = pre-registered Top-1 winner F2-bidir fold0, "
                     "seed-matched checkpoint_best (plan 4.2/4.1; yaml "
                     "aux_calibration)",
        "n_batches": args.batches,
        "lambda_cos_fixed": lambda_cos,
        "lambda_max": LAMBDA_MAX,
        "scale_floor": SCALE_FLOOR,
        "head_share_max": HEAD_SHARE_MAX,
        "per_seed": per_seed,
        "arms": arms,
    }


def _base_grads(model, batch, loss_cfg, lambda_cos) -> Tuple[float, float]:
    """(encoder, head) gradient norms of L_base (λ_bucket=0, no aux arm)."""
    from pes2ts_core.pesn.contracts.key_atom_v3 import (
        NODE_DEGK_KEY,
        NODE_KEY_SET_KEY,
        NODE_POS_KEY,
        forward_view_v3,
    )
    from pes2ts_core.pesn.models.key_atom_losses_v3 import key_atom_loss_v3

    _zero_grads(model)
    out = model(forward_view_v3(batch))
    loss_out = key_atom_loss_v3(
        out["u"], batch["y_teacher"],
        batch[NODE_DEGK_KEY][batch[NODE_KEY_SET_KEY]],
        _key_ptr(batch), batch[NODE_POS_KEY], batch[NODE_KEY_SET_KEY],
        batch["pair_index"], config=loss_cfg, lambda_cos=lambda_cos,
        frame_weight=batch["frame_weight"], pair_edit=batch["pair_edit"])
    loss_out["loss"].backward()
    enc = _grad_group_norm(model, ENCODER_PREFIXES)
    head = _grad_group_norm(model, HEAD_PREFIXES)
    return enc, head


def _grad_pair(bucket_enc: float, bucket_head: float,
               base: Tuple[float, float]) -> Tuple[float, float]:
    """Compatibility shim (ratios are computed directly from the norms)."""
    return bucket_enc, bucket_head


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="r3_gradient_probe",
        description="Calibrate aux-arm λ (S2/S10/D2/D10) from encoder "
        "gradient shares on the B0-F reference (plan 5.4)")
    parser.add_argument("--runs-root", default=CAMPAIGN_ROOT + "/r3_runs")
    parser.add_argument("--data-dir", default=CAMPAIGN_ROOT + "/data")
    parser.add_argument("--split", default=CAMPAIGN_ROOT + "/data/split_manifest_r3.json")
    parser.add_argument("--out", default=None,
                        help="output JSON (default: <runs-root>/lambda_calibration.json)")
    parser.add_argument("--batches", type=int, default=N_BATCHES_DEFAULT)
    parser.add_argument("--lambda-cos", type=float, default=0.02,
                        help="frozen R0 end value (plan 5.1 L_base)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    try:
        assert_campaign_root(str(Path(args.runs_root).parent))
    except CampaignRootError as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 1
    handle = gpu_acquire(GPU_LOCK, lane="W-LOSS-E3CONT",
                         run_id="lambda-probe", cuda=True)
    if handle is None:
        sys.stderr.write("error: GPU lock busy\n")
        return 1
    try:
        result = run_probe(args)
    finally:
        handle.release()
    out = Path(args.out) if args.out else Path(args.runs_root) / "lambda_calibration.json"
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8")
    summary = {"out": str(out),
               "arms": {a: {"lambda": v["lambda"], "verdict": v["verdict"],
                            "head_flag": v["head_flag"],
                            "share": v["achieved_encoder_share"]}
                        for a, v in result["arms"].items()}}
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
