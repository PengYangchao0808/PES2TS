"""Config-driven experiment sweep + append-only experiment registry.

Runs the V5.6-A style protocol from a YAML config: build the dataset view,
train each requested architecture with the *same* Encoder/Heads/training rules,
evaluate ID/OOD + CPU cost, then write ``experiment_registry.csv`` and the
ranking-only ``leaderboard.csv``.  Failures are recorded in
``failure_taxonomy.csv`` — never silently dropped, never imputed.
"""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from pathlib import Path

import torch
import yaml

from pes2ts_v5.data.manifest import V55DatasetSpec
from pes2ts_v5.data.trajectories import V5TrajectoryDataset, build_synthetic_dataset
from pes2ts_v5.evaluation.cpu_benchmark import benchmark_cpu_inference
from pes2ts_v5.evaluation.irregular_sampling import evaluate_irregular_sampling
from pes2ts_v5.evaluation.ood import evaluate_ood
from pes2ts_v5.evaluation.rollout_q import run_rollout_q
from pes2ts_v5.evaluation.single_step import evaluate_single_step
from pes2ts_v5.models.v5_model import build_v5_model
from pes2ts_v5.reports import (
    FAILURE_COLUMNS,
    LEADERBOARD_COLUMNS,
    MODEL_SELECTION_TEMPLATE,
    REGISTRY_COLUMNS,
    append_rows,
    ensure_reports_dir,
    write_csv,
    write_model_selection_template,
)
from pes2ts_v5.training.trainer import TrainConfig, V5Trainer, set_seed

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SYNTHETIC_DIR = REPO_ROOT / "data" / "interim" / "v5_synthetic"


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def config_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hash(module_dir: str | Path | None = None) -> str:
    root = Path(module_dir or REPO_ROOT / "pes2ts_v5")
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        h.update(str(path.relative_to(root)).encode("utf-8"))
        h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


def build_dataset(cfg: dict):
    ds_cfg = dict(cfg.get("dataset") or {})
    kind = str(ds_cfg.get("kind", "synthetic")).lower()
    if kind == "synthetic":
        out = Path(ds_cfg.get("path") or DEFAULT_SYNTHETIC_DIR)
        if not out.is_absolute():
            out = REPO_ROOT / out
        spec = V55DatasetSpec.from_dict(ds_cfg["spec"]) if ds_cfg.get("spec") else None
        force = bool(ds_cfg.get("force_rebuild", True))
        if force or not (out / "manifest.json").exists():
            build_synthetic_dataset(
                out, seed=int(ds_cfg.get("seed", cfg.get("seed", 17))),
                n_trajectories_per_reaction=int(
                    ds_cfg.get("n_trajectories_per_reaction", 2)),
                spec=spec)
        return V5TrajectoryDataset(out)
    if kind == "idx_v2":
        from pes2ts_v5.data.idx_v2 import IdxV2Dataset, IdxV2Reader
        reader = IdxV2Reader(ds_cfg.get("root"))
        return IdxV2Dataset(
            reader,
            mode=str(ds_cfg.get("input_mode", "G0")),
            seed=int(ds_cfg.get("seed", cfg.get("seed", 17))),
            max_curves=dict(ds_cfg.get("max_curves") or {}),
            history_len=int(ds_cfg.get("history_len", 16)),
        )
    raise ValueError(f"unknown dataset kind {kind!r}")


def _null_metrics() -> dict:
    return {"move_rmse": float("nan"), "move_cosine": float("nan"),
            "process_acc": float("nan"), "process_auc": float("nan"),
            "progress_mae": float("nan"), "trust_auc": float("nan")}


def _sweep_archs(cfg: dict, archs: list[str] | None, limit: int | None) -> list[str]:
    selected = archs or cfg.get("archs") or []
    if not selected:
        from pes2ts_v5.models.cores import CORE_REGISTRY
        selected = sorted(CORE_REGISTRY)
    selected = [str(a).upper() for a in selected]
    return selected[:limit] if limit else selected


def run_sweep(
    config_path: str | Path,
    *,
    archs: list[str] | None = None,
    limit: int | None = None,
    out_dir: str | Path | None = None,
    device: str | None = None,
    max_epochs: int | None = None,
    max_train_batches: int | None = None,
) -> dict:
    cfg_path = Path(config_path)
    cfg = load_config(cfg_path)
    stage = str(cfg.get("stage", "V56-A"))
    seed = int(cfg.get("seed", 17))
    set_seed(seed)
    out_root = Path(out_dir or cfg.get("out_dir") or REPO_ROOT / "outputs" / "v5")
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = out_root / f"{stage}_{stamp}"
    reports_dir = run_dir / "reports"
    ensure_reports_dir(reports_dir)
    ds = build_dataset(cfg)
    ds_manifest = getattr(ds, "manifest", None)
    if ds_manifest is not None:
        ds_manifest.save(run_dir)
    artifact_manifest = {
        "stage": stage,
        "dataset_version": ds.dataset_version,
        "dataset_hash": ds.dataset_hash,
        "counts": getattr(ds, "counts", {}),
        "spec": getattr(ds.spec, "to_dict", lambda: {})(),
        "config_path": str(cfg_path),
        "config_hash": config_hash(cfg_path),
        "code_hash": code_hash(),
        "created_at": stamp,
    }
    if hasattr(ds, "reader") and hasattr(ds.reader, "manifest_doc"):
        artifact_manifest["idx_v2"] = ds.reader.manifest_doc()
    (run_dir / "dataset_manifest.json").write_text(
        json.dumps(artifact_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    model_cfg = dict(cfg.get("model") or {})
    train_cfg_doc = dict(cfg.get("training") or {})
    if max_epochs is not None:
        train_cfg_doc["max_epochs"] = max_epochs
    if max_train_batches is not None:
        train_cfg_doc["max_train_batches"] = max_train_batches
    if device is not None:
        train_cfg_doc["device"] = device
    train_cfg = TrainConfig(**{k: v for k, v in train_cfg_doc.items()
                               if k in TrainConfig.__dataclass_fields__})
    eval_cfg = dict(cfg.get("eval") or {})
    selected = _sweep_archs(cfg, archs, limit)
    registry_rows, leaderboard_rows, failures = [], [], []
    for arch in selected:
        experiment_id = f"{stage}-{arch}-s{seed:02d}"
        entry = {"experiment_id": experiment_id, "stage": stage, "arch": arch,
                 "seed": seed, "status": "started", "timestamp": stamp,
                 "dataset_version": ds.dataset_version,
                 "dataset_hash": ds.dataset_hash,
                 "config_hash": artifact_manifest["config_hash"],
                 "code_hash": artifact_manifest["code_hash"],
                 "input_mode": getattr(ds.spec, "input_mode", "G0")}
        t0 = time.time()
        try:
            model, info = build_v5_model(
                arch, model_cfg,
                frame_feat_dim=int(ds.frame_feat_dim),
                static_feat_dim=int(ds.static_feat_dim),
                q_dim=int(ds.q_dim))
            trainer = V5Trainer(
                model, ds, train_cfg,
                train_split=str(train_cfg_doc.get("train_split", "train")),
                val_split=str(train_cfg_doc.get("val_split", "valid")),
                out_dir=run_dir, experiment_id=experiment_id)
            train_report = trainer.train()
            val_metrics = trainer.evaluate()
            cpu = (benchmark_cpu_inference(model, ds,
                                           history_len=int(train_cfg.history_len),
                                           n_iters=int(eval_cfg.get("cpu_iters", 10)),
                                           n_warmup=int(eval_cfg.get("cpu_warmup", 2)))
                   if eval_cfg.get("cpu_benchmark", True) else {"latency_ms_mean": float("nan")})
            # the CPU benchmark temporarily moves the model to CPU; restore the
            # training device before further GPU/CPU-agnostic evaluation
            model.to(trainer.device)
            id_metrics = evaluate_single_step(
                model, ds, "id", device=trainer.device,
                max_batches=eval_cfg.get("max_eval_batches"))
            rollout = {}
            if eval_cfg.get("run_rollout", False) and len(ds.trajectory_indices("id")) > 0:
                try:
                    rollout = run_rollout_q(
                        model, ds, "id", device=trainer.device,
                        max_trajectories=int(eval_cfg.get("rollout_max_trajectories", 2)),
                        steps=int(eval_cfg.get("rollout_steps", 3)))
                except NotImplementedError as exc:
                    rollout = {"rollout_error": str(exc)}
            ckpt = trainer._checkpoint_dir()
            weights_file = ckpt / "best.pt" if ckpt is not None else None
            weights_sha = ""
            if weights_file is not None and weights_file.exists():
                weights_sha = hashlib.sha256(weights_file.read_bytes()).hexdigest()[:16]
            best_ckpt = ckpt / "best.pt" if ckpt is not None else None
            entry.update({
                **{k: info.get(k, "") for k in (
                    "encoder", "encoder_params", "core_params", "head_params",
                    "total_params", "core_hidden_dim", "param_budget", "rel_error")},
                "core_layers": model_cfg.get("core_layers", 2),
                "epochs_run": train_report["epochs_run"],
                "best_epoch": train_report["best_epoch"],
                "train_seconds": round(time.time() - t0, 2),
                "device": train_report["device"],
                "best_val_loss": train_report["best_val_loss"],
                "move_rmse": val_metrics.get("move_rmse"),
                "move_cosine": val_metrics.get("move_cosine"),
                "process_acc": val_metrics.get("process_acc"),
                "process_auc": val_metrics.get("process_auc"),
                "progress_mae": val_metrics.get("progress_mae"),
                "trust_auc": val_metrics.get("trust_auc"),
                "cpu_step_ms": cpu.get("latency_ms_mean"),
                "cpu_p95_ms": cpu.get("latency_ms_p95"),
                "weights_sha256": weights_sha,
                "checkpoint_path": str(best_ckpt) if best_ckpt else "",
                "status": "done",
                "notes": json.dumps({"id_move_rmse": id_metrics.get("move_rmse"),
                                     "rollout": rollout}, ensure_ascii=False),
            })
            registry_rows.append(entry)
            leaderboard_rows.append(entry)
        except Exception as exc:  # noqa: BLE001 - recorded, not hidden
            failure = {
                "experiment_id": experiment_id, "arch": arch, "stage": stage,
                "failure_class": type(exc).__name__,
                "error_message": f"{exc}"[:800], "timestamp": stamp,
            }
            failures.append(failure)
            entry.update({"status": "failed", "notes": f"{type(exc).__name__}: {exc}"[:400]})
            registry_rows.append(entry)
            (run_dir / "failures.log").open("a", encoding="utf-8").write(
                f"[{experiment_id}] {traceback.format_exc()}\n")
    append_rows(reports_dir / "experiment_registry.csv", REGISTRY_COLUMNS, registry_rows)
    if failures:
        append_rows(reports_dir / "failure_taxonomy.csv", FAILURE_COLUMNS, failures)
    ranked = sorted(
        [r for r in leaderboard_rows if r.get("status") == "done"],
        key=lambda r: (float(r.get("move_rmse") or float("inf"))))
    for i, row in enumerate(ranked, start=1):
        row["rank"] = i
    write_csv(reports_dir / "leaderboard.csv", LEADERBOARD_COLUMNS, ranked)
    write_model_selection_template(
        reports_dir / "model_selection.md",
        dataset_version=ds.dataset_version, dataset_hash=ds.dataset_hash,
        config_path=str(cfg_path), config_hash=artifact_manifest["config_hash"],
        timestamp=stamp)
    summary = {
        "stage": stage, "run_dir": str(run_dir), "reports_dir": str(reports_dir),
        "dataset_version": ds.dataset_version, "dataset_hash": ds.dataset_hash,
        "archs": selected, "n_done": len(ranked), "n_failed": len(failures),
        "leaderboard": [
            {"rank": r.get("rank"), "arch": r.get("arch"),
             "move_rmse": r.get("move_rmse"), "core_params": r.get("core_params"),
             "cpu_step_ms": r.get("cpu_step_ms"), "status": r.get("status")}
            for r in ranked],
        "failures": failures,
    }
    (run_dir / "sweep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


__all__ = ["build_dataset", "code_hash", "config_hash", "load_config", "run_sweep",
           "MODEL_SELECTION_TEMPLATE"]
