"""Single CLI entry point for the PES2TS V5.6 training / evaluation engine.

Usage (repository root)::

    python -m pes2ts_v5.runner train --config config/experiments/v56_baseline_gru.yaml
    python -m pes2ts_v5.runner train --config config/experiments/smoke.yaml \
        --max-epochs 1 --max-train-batches 2 --out-dir .slim/runs/smoke-test
    python -m pes2ts_v5.runner eval  --run-dir outputs/v5/runs/<run>
    python -m pes2ts_v5.runner sweep --config config/experiments/v56_arch_matrix.yaml \
        --arch M01 --arch M04 --max-epochs 1
    python -m pes2ts_v5.runner smoke --config config/experiments/v56_baseline_gru.yaml

Every subcommand prints one compact JSON object and returns exit code ``0``
(success) or ``1`` (failure, with ``error_type`` / ``code`` in the JSON).

Data honesty: a ``store_manifest`` with a ``manifest.json`` is loaded as a real
V5.5 store; when the policy is a ``synthetic_fixture*`` policy the deterministic
synthetic fixture is built at that path (this is the only fallback).  Any other
missing store raises ``V55ContractNotAvailable`` — no data is ever fabricated.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from pes2ts_v5.configs.schema import (
    SCHEMA_NAME,
    ExperimentConfig,
    config_hash,
    config_to_dict,
    load_experiment_config,
)
from pes2ts_v5.data.resolve import (
    CappedDataset,
    DataDimensionMismatch,
    resolve_dataset,
    resolve_repo_path,
)
from pes2ts_v5.evaluation.single_step import evaluate_single_step
from pes2ts_v5.losses.components import build_weighted_loss_from_config
from pes2ts_v5.models.registry import (
    assert_model_config_runnable,
    build_model_from_config,
    resolve_core_hidden_dim,
    resolve_core_name,
)
from pes2ts_v5.training._util import (
    json_safe,
    param_count,
    resolve_device,
    set_seed,
    write_json,
)
from pes2ts_v5.training.callbacks import build_callbacks
from pes2ts_v5.training.trainer import TrainConfig, V5Trainer, preflight_forward_backward

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = REPO_ROOT / "outputs" / "v5" / "runs"

class RunnerError(RuntimeError):
    """Typed runner failure (``code`` is stable for callers/tests)."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# Paths / dataset resolution (implemented once in ``pes2ts_v5.data.resolve``)
# ---------------------------------------------------------------------------

def _default_run_dir(experiment_id: str) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return DEFAULT_OUT_ROOT / f"{experiment_id}_{stamp}"


def _write_resolved_config(cfg: ExperimentConfig, path: Path) -> Path:
    """Write the validated config as canonical JSON into the run directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config_to_dict(cfg), ensure_ascii=False,
                               indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def run_train(
    cfg: ExperimentConfig,
    *,
    config_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    device: str | None = None,
    max_epochs: int | None = None,
    max_train_batches: int | None = None,
    resume: str | Path | None = None,
) -> dict:
    """Train one model described by ``cfg``; return the JSON-able summary."""
    run_dir = Path(out_dir) if out_dir is not None else _default_run_dir(cfg.experiment_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = None if config_path is None else Path(config_path)
    cfg_hash = config_hash(cfg)
    # fail fast on capability blockers (e.g. encoder input contract) before data
    assert_model_config_runnable(cfg.model)
    # seed BEFORE dataset/view/model construction so a (config_hash, seed) pair
    # always maps to identical initial weights and batches (P3 replayability)
    set_seed(int(cfg.seed))
    _write_resolved_config(cfg, run_dir / "config_resolved.json")

    dataset, data_info = resolve_dataset(cfg)
    model = build_model_from_config(
        cfg.model, frame_feat_dim=int(dataset.frame_feat_dim),
        static_feat_dim=int(dataset.static_feat_dim))
    train_cfg = TrainConfig.from_experiment(
        cfg, max_epochs=max_epochs, max_train_batches=max_train_batches,
        device=device)
    callbacks = build_callbacks(cfg.training.callbacks, out_dir=run_dir,
                                patience=cfg.training.patience, repo_root=REPO_ROOT)
    loss_fn = build_weighted_loss_from_config(cfg.loss)
    provenance = {
        "schema": SCHEMA_NAME,
        "config_path": None if config_path is None else str(config_path),
        "data": data_info,
        "experiment_id": cfg.experiment_id,
        "model": {
            "encoder": cfg.model.encoder,
            "core": cfg.model.core.name,
            "core_id": resolve_core_name(cfg.model.core.name),
            "head": cfg.model.head.name,
        },
    }
    trainer = V5Trainer(
        model, dataset, train_cfg, out_dir=run_dir,
        experiment_id=cfg.experiment_id, callbacks=callbacks, loss_fn=loss_fn,
        config_hash=cfg_hash, provenance=provenance)
    if resume is not None:
        trainer.load_checkpoint(resume)
    report = trainer.train()

    eval_results: dict[str, Any] = {}
    test_device = str(trainer.device)
    for split in list(cfg.evaluation.splits) + list(cfg.evaluation.ood):
        try:
            eval_results[split] = evaluate_single_step(
                model, dataset, split, device=test_device,
                batch_size=train_cfg.batch_size)
        except KeyError as exc:  # split absent from this dataset artifact
            eval_results[split] = {"error": f"{type(exc).__name__}: {exc}"}

    cpu_metrics: dict[str, Any] | None = None
    if cfg.evaluation.cpu_benchmark:
        from pes2ts_v5.evaluation.cpu_benchmark import benchmark_cpu_inference
        try:
            cpu_metrics = benchmark_cpu_inference(
                model, dataset, history_len=int(train_cfg.history_len))
        finally:  # the benchmark moves the model to CPU; restore training device
            model.to(trainer.device)

    checkpoints = {}
    ckpt_dir = trainer._checkpoint_dir()
    if ckpt_dir is not None:
        checkpoints = {path.name: str(path) for path in sorted(ckpt_dir.glob("*.pt"))}

    summary = {
        "status": "ok",
        "mode": "train",
        "experiment_id": cfg.experiment_id,
        "config_hash": cfg_hash,
        "run_dir": str(run_dir),
        "device": str(trainer.device),
        "data": {
            **data_info,
            "dataset_hash": str(getattr(dataset, "dataset_hash", "")),
            "dataset_version": str(getattr(dataset, "dataset_version", "")),
            "counts": getattr(dataset, "counts", {}),
        },
        "model": {
            "encoder": cfg.model.encoder,
            "core": cfg.model.core.name,
            "core_id": resolve_core_name(cfg.model.core.name),
            "core_hidden_dim": resolve_core_hidden_dim(cfg.model.core),
            "head": cfg.model.head.name,
            "total_params": param_count(model),
            "param_report": json_safe(getattr(model, "param_report", lambda: {})()),
        },
        "loss": json_safe(loss_fn.describe()),
        "training": json_safe(report),
        "evaluation": json_safe(eval_results),
        "cpu_benchmark": json_safe(cpu_metrics),
        "checkpoints": checkpoints,
        "artifacts": {
            "config_resolved": str(run_dir / "config_resolved.json"),
            "summary": str(run_dir / "summary.json"),
        },
        "callbacks": [cb.name for cb in callbacks],
    }
    write_json(run_dir / "summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def _find_checkpoint(run_dir: Path) -> Path | None:
    candidates = sorted((run_dir / "checkpoints").glob("*/*.pt"))
    for name in ("best.pt", "ema.pt", "last.pt"):
        for path in candidates:
            if path.name == name:
                return path
    return candidates[0] if candidates else None


def run_eval(run_dir: str | Path, *, device: str | None = None) -> dict:
    """Evaluate a finished run directory (writes ``eval.json``)."""
    root = Path(run_dir)
    config_file = root / "config_resolved.json"
    if not config_file.exists():
        raise RunnerError("RUN_DIR_INVALID", f"missing {config_file}")
    cfg = load_experiment_config(config_file)
    assert_model_config_runnable(cfg.model)
    dataset, data_info = resolve_dataset(cfg)
    model = build_model_from_config(
        cfg.model, frame_feat_dim=int(dataset.frame_feat_dim),
        static_feat_dim=int(dataset.static_feat_dim))
    checkpoint = _find_checkpoint(root)
    if checkpoint is None:
        raise RunnerError("CHECKPOINT_NOT_FOUND", f"no *.pt checkpoint under {root}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"])
    resolved = resolve_device(device or cfg.training.device)
    model.to(resolved)
    results: dict[str, Any] = {}
    for split in list(cfg.evaluation.splits) + list(cfg.evaluation.ood):
        try:
            results[split] = evaluate_single_step(
                model, dataset, split, device=str(resolved))
        except KeyError as exc:
            results[split] = {"error": f"{type(exc).__name__}: {exc}"}
    summary = {
        "status": "ok",
        "mode": "eval",
        "experiment_id": cfg.experiment_id,
        "config_hash": config_hash(cfg),
        "run_dir": str(root),
        "checkpoint": str(checkpoint),
        "checkpoint_weights_sha256": str(payload.get("weights_sha256", "")),
        "device": str(resolved),
        "data": data_info,
        "evaluation": json_safe(results),
    }
    write_json(root / "eval.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Sweep (delegates to the existing sweep runner; no second training loop)
# ---------------------------------------------------------------------------

def run_sweep(
    cfg: ExperimentConfig,
    *,
    archs: list[str] | None = None,
    out_dir: str | Path | None = None,
    device: str | None = None,
    max_epochs: int | None = None,
    max_train_batches: int | None = None,
) -> dict:
    """Run the architecture matrix through the single engine sweep.

    Delegates to :func:`pes2ts_v5.training.sweep.run_sweep`, which assembles every
    architecture with ``models.registry.build_model_from_config`` and trains with
    ``training.trainer.V5Trainer`` — the same code path as ``runner train``.
    """
    from pes2ts_v5.training.sweep import run_sweep as _run_sweep

    return json_safe(_run_sweep(
        cfg, archs=archs, out_dir=out_dir, device=device,
        max_epochs=max_epochs, max_train_batches=max_train_batches))


# ---------------------------------------------------------------------------
# Smoke (fast preflight: build + one forward/backward, no artifacts)
# ---------------------------------------------------------------------------

def run_smoke(cfg: ExperimentConfig, *, device: str | None = None) -> dict:
    """Build dataset+model from ``cfg`` and run one finite forward/backward."""
    from pes2ts_v5.models.registry import make_smoke_batch

    assert_model_config_runnable(cfg.model)
    set_seed(int(cfg.seed))
    dataset, data_info = resolve_dataset(cfg)
    model = build_model_from_config(
        cfg.model, frame_feat_dim=int(dataset.frame_feat_dim),
        static_feat_dim=int(dataset.static_feat_dim))
    resolved = resolve_device(device or cfg.training.device)
    batch = make_smoke_batch(frame_feat_dim=int(dataset.frame_feat_dim),
                             static_feat_dim=int(dataset.static_feat_dim),
                             q_dim=int(cfg.model.head.q_dim),
                             process_classes=int(cfg.model.head.process_classes),
                             device="cpu")
    loss_fn = build_weighted_loss_from_config(cfg.loss)
    preflight = preflight_forward_backward(
        model, batch, loss_fn=loss_fn, device=str(resolved))
    return {
        "status": "ok",
        "mode": "smoke",
        "experiment_id": cfg.experiment_id,
        "config_hash": config_hash(cfg),
        "device": str(resolved),
        "data": data_info,
        "core": cfg.model.core.name,
        "total_params": param_count(model),
        "preflight": preflight,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True,
                        help="experiment YAML under config/experiments/")
    parser.add_argument("--device", type=str, default=None,
                        help="cpu / cuda / auto (overrides training.device)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="run directory (train/sweep)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pes2ts_v5.runner",
        description="PES2TS V5.6 single training/evaluation/sweep CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train one model from an experiment YAML")
    _add_common(p_train)
    p_train.add_argument("--max-epochs", type=int, default=None)
    p_train.add_argument("--max-train-batches", type=int, default=None)
    p_train.add_argument("--resume", type=Path, default=None,
                         help="checkpoint to resume from")

    p_eval = sub.add_parser("eval", help="evaluate a finished run directory")
    p_eval.add_argument("--run-dir", type=Path, required=True)
    p_eval.add_argument("--device", type=str, default=None)

    p_sweep = sub.add_parser("sweep", help="architecture sweep over one YAML")
    _add_common(p_sweep)
    p_sweep.add_argument("--arch", action="append", default=None,
                         help="core id/alias, repeatable (default: config core)")
    p_sweep.add_argument("--max-epochs", type=int, default=None)
    p_sweep.add_argument("--max-train-batches", type=int, default=None)

    p_smoke = sub.add_parser("smoke", help="build + one forward/backward preflight")
    p_smoke.add_argument("--config", type=Path, required=True)
    p_smoke.add_argument("--device", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI; returns 0 on success, 1 on typed/handled failure."""
    parser = build_parser()
    args = parser.parse_args(argv)
    mode = str(args.command)
    try:
        if mode == "train":
            cfg = load_experiment_config(args.config)
            summary = run_train(
                cfg, config_path=args.config, out_dir=args.out_dir,
                device=args.device, max_epochs=args.max_epochs,
                max_train_batches=args.max_train_batches, resume=args.resume)
        elif mode == "eval":
            summary = run_eval(args.run_dir, device=args.device)
        elif mode == "sweep":
            cfg = load_experiment_config(args.config)
            summary = run_sweep(
                cfg, archs=args.arch, out_dir=args.out_dir, device=args.device,
                max_epochs=args.max_epochs,
                max_train_batches=args.max_train_batches)
            summary["status"] = "ok" if summary.get("n_failed", 0) == 0 else "partial"
            summary["mode"] = "sweep"
        elif mode == "smoke":
            cfg = load_experiment_config(args.config)
            summary = run_smoke(cfg, device=args.device)
        else:  # pragma: no cover - argparse enforces the choices
            raise RunnerError("UNKNOWN_COMMAND", f"unknown command {mode!r}")
    except Exception as exc:
        summary = {
            "status": "error",
            "mode": mode,
            "error_type": type(exc).__name__,
            "code": getattr(exc, "code", None),
            "error": str(exc)[:2000],
        }
        print(json.dumps(json_safe(summary), ensure_ascii=False))
        return 1
    print(json.dumps(json_safe(summary), ensure_ascii=False))
    if summary.get("status") == "error":
        return 1
    if mode == "sweep" and summary.get("status") == "partial":
        return 1
    return 0


__all__ = [
    "CappedDataset",
    "DEFAULT_OUT_ROOT",
    "DataDimensionMismatch",
    "REPO_ROOT",
    "RunnerError",
    "build_parser",
    "main",
    "resolve_dataset",
    "resolve_repo_path",
    "run_eval",
    "run_smoke",
    "run_sweep",
    "run_train",
]


if __name__ == "__main__":
    raise SystemExit(main())
