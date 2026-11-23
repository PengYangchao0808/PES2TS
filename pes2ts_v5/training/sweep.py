"""Config-driven architecture sweep + append-only experiment registry.

Every architecture goes through the **single V5.6 engine**, exactly like the
``train`` command:

``configs.load_experiment_config`` (one config contract)
-> ``data.resolve.resolve_dataset`` (one dataset resolution)
-> ``models.registry.build_model_from_config`` (one model-assembly path)
-> ``training.trainer.V5Trainer`` (one training loop).

There is no second config parser, no second model builder and no second training
loop in this module — it only runs the matrix and records results:
``experiment_registry.csv`` (append-only), ``failure_taxonomy.csv`` (failures are
never silently dropped or imputed), ranking-only ``leaderboard.csv`` and the
``model_selection.md`` template.
"""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from pathlib import Path

import torch  # noqa: F401 - re-exported for legacy sweep callers
import yaml  # noqa: F401 - re-exported for legacy sweep callers

from pes2ts_v5.configs.schema import (
    ExperimentConfig,
    load_experiment_config,
)
from pes2ts_v5.configs.schema import (
    config_hash as experiment_config_hash,
)
from pes2ts_v5.data.resolve import REPO_ROOT, resolve_dataset
from pes2ts_v5.evaluation.cpu_benchmark import benchmark_cpu_inference
from pes2ts_v5.evaluation.single_step import evaluate_single_step
from pes2ts_v5.losses.components import build_weighted_loss_from_config
from pes2ts_v5.models.registry import build_model_from_config, resolve_core_name
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
from pes2ts_v5.training._util import json_safe, set_seed
from pes2ts_v5.training.callbacks import build_callbacks
from pes2ts_v5.training.trainer import TrainConfig, V5Trainer

DEFAULT_SYNTHETIC_DIR = REPO_ROOT / "data" / "interim" / "v5_synthetic"


# ---------------------------------------------------------------------------
# Thin compatibility helpers (single source of truth: the experiment schema)
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> ExperimentConfig:
    """Load + validate the experiment YAML (the schema loader; no legacy parsing)."""
    return load_experiment_config(path)


def config_hash(path: str | Path) -> str:
    """Canonical config hash of the experiment YAML at ``path``."""
    return experiment_config_hash(load_experiment_config(path))


def code_hash(module_dir: str | Path | None = None) -> str:
    root = Path(module_dir or REPO_ROOT / "pes2ts_v5")
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        h.update(str(path.relative_to(root)).encode("utf-8"))
        h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


def build_dataset(cfg: ExperimentConfig):
    """Resolve the dataset for ``cfg`` (same resolution as the runner)."""
    dataset, _info = resolve_dataset(cfg)
    return dataset


def _sweep_archs(cfg: ExperimentConfig, archs: list[str] | None,
                 limit: int | None) -> list[str]:
    selected = list(archs) if archs else [cfg.model.core.name]
    canonical: list[str] = []
    for name in selected:
        core_id = resolve_core_name(str(name))
        if core_id not in canonical:
            canonical.append(core_id)
    return canonical[:limit] if limit else canonical


def _variant_config(cfg: ExperimentConfig, arch: str) -> ExperimentConfig:
    """Config with ``model.core.name`` replaced by one canonical architecture."""
    core = cfg.model.core.model_copy(update={"name": arch})
    model = cfg.model.model_copy(update={"core": core})
    return cfg.model_copy(update={
        "model": model,
        "experiment_id": f"{cfg.experiment_id}-{arch.lower()}-s{int(cfg.seed):02d}",
    })


def _eval_split(cfg: ExperimentConfig, dataset) -> str:
    splits = list(getattr(dataset, "splits", {}) or {})
    for candidate in ("id", *cfg.evaluation.splits):
        if candidate in splits:
            return candidate
    raise KeyError(f"none of the evaluation splits {cfg.evaluation.splits} exist "
                   f"in the dataset (available: {splits})")


def run_sweep(
    config_path: str | Path | ExperimentConfig,
    *,
    archs: list[str] | None = None,
    limit: int | None = None,
    out_dir: str | Path | None = None,
    device: str | None = None,
    max_epochs: int | None = None,
    max_train_batches: int | None = None,
) -> dict:
    """Run the architecture matrix described by one experiment config."""
    cfg = (config_path if isinstance(config_path, ExperimentConfig)
           else load_experiment_config(config_path))
    config_path_value = ("" if isinstance(config_path, ExperimentConfig)
                         else str(config_path))
    stage = cfg.experiment_id
    seed = int(cfg.seed)
    set_seed(seed)

    out_root = Path(out_dir or REPO_ROOT / "outputs" / "v5")
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = out_root / f"{stage}_{stamp}"
    reports_dir = run_dir / "reports"
    ensure_reports_dir(reports_dir)

    dataset, data_info = resolve_dataset(cfg)
    manifest = getattr(dataset, "manifest", None)
    if manifest is not None:
        manifest.save(run_dir)
    cfg_hash = experiment_config_hash(cfg)
    artifact_manifest = {
        "stage": stage,
        "config_path": config_path_value,
        "config_hash": cfg_hash,
        "code_hash": code_hash(),
        "created_at": stamp,
        "dataset_version": dataset.dataset_version,
        "dataset_hash": dataset.dataset_hash,
        "counts": getattr(dataset, "counts", {}),
        "data": json_safe(data_info),
        "dims": {
            "frame_feat_dim": int(dataset.frame_feat_dim),
            "static_feat_dim": int(dataset.static_feat_dim),
            "q_dim": int(dataset.q_dim),
        },
    }
    (run_dir / "dataset_manifest.json").write_text(
        json.dumps(artifact_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8")

    train_cfg = TrainConfig.from_experiment(
        cfg, device=device, max_epochs=max_epochs,
        max_train_batches=max_train_batches)
    loss_fn = build_weighted_loss_from_config(cfg.loss)
    eval_split = _eval_split(cfg, dataset)
    selected = _sweep_archs(cfg, archs, limit)

    registry_rows: list[dict] = []
    leaderboard_rows: list[dict] = []
    failures: list[dict] = []
    for arch in selected:
        experiment_id = f"{stage}-{arch}-s{seed:02d}"
        attempt_dir = run_dir / "experiments" / arch
        entry: dict = {
            "experiment_id": experiment_id, "stage": stage, "arch": arch,
            "seed": seed, "status": "started", "timestamp": stamp,
            "dataset_version": dataset.dataset_version,
            "dataset_hash": dataset.dataset_hash, "config_hash": cfg_hash,
            "code_hash": artifact_manifest["code_hash"],
            "input_mode": str(getattr(getattr(dataset, "spec", None),
                                      "input_mode", "G0")),
        }
        t0 = time.time()
        try:
            variant = _variant_config(cfg, arch)
            model = build_model_from_config(
                variant.model, frame_feat_dim=int(dataset.frame_feat_dim),
                static_feat_dim=int(dataset.static_feat_dim))
            callbacks = build_callbacks(
                cfg.training.callbacks, out_dir=attempt_dir,
                patience=cfg.training.patience, repo_root=REPO_ROOT)
            trainer = V5Trainer(
                model, dataset, train_cfg, out_dir=attempt_dir,
                experiment_id=experiment_id, callbacks=callbacks,
                loss_fn=loss_fn, config_hash=experiment_config_hash(variant),
                provenance={"stage": stage, "arch": arch,
                            "data": json_safe(data_info)})
            train_report = trainer.train()
            val_metrics = trainer.evaluate()
            if cfg.evaluation.cpu_benchmark:
                cpu = benchmark_cpu_inference(
                    model, dataset, history_len=int(train_cfg.history_len))
                model.to(trainer.device)  # benchmark moved it to CPU
            else:
                cpu = {"latency_ms_mean": None, "latency_ms_p95": None}
            id_metrics = evaluate_single_step(
                model, dataset, eval_split, device=str(trainer.device))
            core_info = dict(getattr(model, "core_info", {}) or {})
            param_report = model.param_report()
            checkpoint_dir = trainer._checkpoint_dir()
            best_checkpoint = (checkpoint_dir / "best.pt"
                               if checkpoint_dir is not None else None)
            weights_sha = ""
            if best_checkpoint is not None and best_checkpoint.exists():
                weights_sha = hashlib.sha256(
                    best_checkpoint.read_bytes()).hexdigest()[:16]
            entry.update({
                "encoder": variant.model.encoder,
                "encoder_params": param_report.get("encoder_params"),
                "core_params": param_report.get("core_params"),
                "head_params": param_report.get("head_params"),
                "total_params": param_report.get("total_params"),
                "core_hidden_dim": int(core_info.get(
                    "hidden_dim", trainer.state.extra.get("core_hidden_dim", 0) or 0)),
                "param_budget": int(variant.model.core.param_budget),
                "rel_error": core_info.get("rel_error"),
                "core_layers": int(variant.model.core.layers),
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
                "checkpoint_path": str(best_checkpoint) if best_checkpoint else "",
                "status": "done",
                "notes": json.dumps({
                    "eval_split": eval_split,
                    "id_move_rmse": id_metrics.get("move_rmse"),
                    "data_source": data_info.get("source"),
                    "config_hash": experiment_config_hash(variant),
                }, ensure_ascii=False),
            })
            registry_rows.append(entry)
            leaderboard_rows.append(dict(entry))
        except Exception as exc:
            failures.append({
                "experiment_id": experiment_id, "arch": arch, "stage": stage,
                "failure_class": type(exc).__name__,
                "error_message": f"{exc}"[:800], "timestamp": stamp,
            })
            entry.update({"status": "failed",
                          "notes": f"{type(exc).__name__}: {exc}"[:400]})
            registry_rows.append(entry)
            with (run_dir / "failures.log").open("a", encoding="utf-8") as fh:
                fh.write(f"[{experiment_id}] {traceback.format_exc()}\n")

    append_rows(reports_dir / "experiment_registry.csv", REGISTRY_COLUMNS,
                registry_rows)
    if failures:
        append_rows(reports_dir / "failure_taxonomy.csv", FAILURE_COLUMNS, failures)
    ranked = sorted(
        [row for row in leaderboard_rows if row.get("status") == "done"],
        key=lambda row: float(row.get("move_rmse") or float("inf")))
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    write_csv(reports_dir / "leaderboard.csv", LEADERBOARD_COLUMNS, ranked)
    write_model_selection_template(
        reports_dir / "model_selection.md",
        dataset_version=dataset.dataset_version,
        dataset_hash=dataset.dataset_hash,
        config_path=config_path_value or f"<ExperimentConfig {stage}>",
        config_hash=cfg_hash, timestamp=stamp)

    summary = {
        "stage": stage,
        "run_dir": str(run_dir),
        "reports_dir": str(reports_dir),
        "config_hash": cfg_hash,
        "dataset_version": dataset.dataset_version,
        "dataset_hash": dataset.dataset_hash,
        "data_source": data_info.get("source"),
        "dims": artifact_manifest["dims"],
        "archs": selected,
        "n_done": len(ranked),
        "n_failed": len(failures),
        "leaderboard": [{
            "rank": row.get("rank"), "arch": row.get("arch"),
            "move_rmse": row.get("move_rmse"),
            "core_params": row.get("core_params"),
            "core_hidden_dim": row.get("core_hidden_dim"),
            "cpu_step_ms": row.get("cpu_step_ms"),
            "status": row.get("status"),
        } for row in ranked],
        "failures": failures,
    }
    (run_dir / "sweep_summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8")
    return summary


__all__ = [
    "DEFAULT_SYNTHETIC_DIR",
    "build_dataset",
    "code_hash",
    "config_hash",
    "load_config",
    "run_sweep",
    "MODEL_SELECTION_TEMPLATE",
]
