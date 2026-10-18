"""Explicit offline entry points with scoped configuration (B1–B6).

The predictor keeps its own standalone configuration file (default
``config/pesn/g3_p1.yaml``); unknown keys and unknown sections fail loudly.
The Classic ``config/defaults.yaml`` is deliberately not consulted here.
Offline commands never enter the runtime import chain — ``pesn.__main__``
imports this module lazily.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import yaml

from pes2ts_core.pesn.__main__ import (
    EXIT_BACKEND_MISSING,
    EXIT_CHECKSUM_ERROR,
    EXIT_CONFIG_ERROR,
    EXIT_CONTRACT_REJECT,
    EXIT_OK,
    EXIT_PIPELINE_FAILED,
)
from pes2ts_core.utils.hashing import JSONValue, sha256_file

TRUTH_FLAG_EXIT = 23  # mirrors the repo-wide convention for missing --allow-truth

_ALLOWED_TOP: frozenset[str] = frozenset(
    {"schema_version", "paths", "seed", "inventory", "split", "build_teacher", "train", "model", "eval", "bundle"}
)
_ALLOWED_KEYS: dict[str, frozenset[str]] = {
    "paths": frozenset({"interim", "manifests", "output_root", "artifacts_root"}),
    "inventory": frozenset({"max_reactions"}),
    "split": frozenset(
        {"dev_fraction", "calibration_fraction", "family_holdout_fraction", "development_exposure_file"}
    ),
    "build_teacher": frozenset(
        {
            "train_fit_limit",
            "train_dev_limit",
            "authority_shard_size",
            "dataset_shard_size",
            "fit_reaction_ids_file",
            "dev_reaction_ids_file",
            "fit_reaction_ids_key",
            "dev_reaction_ids_key",
        }
    ),
    "train": frozenset(
        {
            "batch_reactions",
            "frames_per_reaction",
            "pairs_per_reaction",
            "pair_margin",
            "max_epochs",
            "patience",
            "lr",
            "weight_decay",
            "clip",
            "seed",
            "tau",
            "loss_weights",
            "apply_symmetry",
            "lazy_loading",
        }
    ),
    "model": frozenset({"hidden", "n_layers", "n_basis", "spatial_cutoff"}),
    "eval": frozenset({"batch_reactions", "bootstrap_iterations"}),
    "bundle": frozenset({"model_id"}),
}
_ALLOWED_LOSS_WEIGHTS: frozenset[str] = frozenset(
    {"s", "d", "rank", "direction", "activity", "symmetry"}
)


class OfflineConfigError(RuntimeError):
    """Invalid predictor configuration (unknown key / bad value)."""


def load_pesn_config(path: str | Path) -> dict[str, JSONValue]:
    config_path = Path(path)
    if not config_path.is_file():
        raise OfflineConfigError(f"config file missing: {config_path}")
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise OfflineConfigError(f"config is not a mapping: {config_path}")
    unknown = set(document) - _ALLOWED_TOP
    if unknown:
        raise OfflineConfigError(f"unknown config sections: {sorted(unknown)}")
    for section, allowed in _ALLOWED_KEYS.items():
        value = document.get(section)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise OfflineConfigError(f"config section {section!r} must be a mapping")
        unknown_keys = set(value) - allowed
        if unknown_keys:
            raise OfflineConfigError(
                f"unknown keys in section {section!r}: {sorted(unknown_keys)}"
            )
    weights = (document.get("train") or {}).get("loss_weights")
    if weights is not None:
        unknown_weights = set(weights) - _ALLOWED_LOSS_WEIGHTS
        if unknown_weights:
            raise OfflineConfigError(
                f"unknown loss weights: {sorted(unknown_weights)}"
            )
    if str(document.get("schema_version")) != "pesn_g3p1_config_v1":
        raise OfflineConfigError(
            f"config schema_version {document.get('schema_version')!r} unsupported"
        )
    return document


def _output_root(config: Mapping[str, JSONValue], override: str | None) -> Path:
    root = Path(override or str(config.get("paths", {}).get("output_root", "outputs/pesn/g3_p1")))
    root.mkdir(parents=True, exist_ok=True)
    return root


def cmd_inventory(args) -> int:
    from pes2ts_core.pesn.offline.data.inventory import build_inventory_report
    from pes2ts_core.utils.jsonio import write_json

    try:
        config = load_pesn_config(args.config)
    except OfflineConfigError as exc:
        print(f"config error: {exc}")
        return EXIT_CONFIG_ERROR
    interim = str(config.get("paths", {}).get("interim", "data/interim"))
    try:
        report = build_inventory_report(interim)
    except Exception as exc:
        print(f"inventory failed: {type(exc).__name__}: {exc}")
        return EXIT_CHECKSUM_ERROR
    out = _output_root(config, args.output_root) / "inventory"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "inventory_report.json", report)
    print(
        f"inventory: qualified_train={report['n_qualified_train']} counts={report['qualification_counts']}"
    )
    return EXIT_OK


def cmd_split(args) -> int:
    from pes2ts_core.pesn.offline.data.inventory import load_source_inventory
    from pes2ts_core.pesn.offline.splits.freeze import freeze_split, write_split_manifest
    from pes2ts_core.pesn.offline.experiments.runs import config_content_sha256
    from pes2ts_core.utils.jsonio import write_json

    try:
        config = load_pesn_config(args.config)
    except OfflineConfigError as exc:
        print(f"config error: {exc}")
        return EXIT_CONFIG_ERROR
    paths = config.get("paths", {})
    interim = str(paths.get("interim", "data/interim"))
    manifests = str(paths.get("manifests", "data/manifests"))
    split_cfg = config.get("split") or {}
    seed = int(config.get("seed", 42))

    exposure_ids: set[str] = set()
    exposure_file = split_cfg.get("development_exposure_file")
    if exposure_file:
        exposure_path = Path(str(exposure_file))
        if exposure_path.suffix == ".csv":
            import csv

            with exposure_path.open(encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    rid = row.get("reaction_id")
                    if rid:
                        exposure_ids.add(str(rid))
        else:
            exposure_ids.update(
                str(line).strip()
                for line in exposure_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )

    try:
        source = load_source_inventory(interim)
        manifest = freeze_split(
            interim,
            manifests,
            seed=seed,
            family_by_reaction=source.family_by_reaction,
            development_exposure_ids=exposure_ids,
            dev_fraction=float(split_cfg.get("dev_fraction", 0.10)),
            calibration_fraction=float(split_cfg.get("calibration_fraction", 0.10)),
            family_holdout_fraction=float(split_cfg.get("family_holdout_fraction", 0.05)),
        )
    except Exception as exc:
        print(f"split failed: {type(exc).__name__}: {exc}")
        return EXIT_PIPELINE_FAILED
    out = _output_root(config, args.output_root) / "splits"
    path = write_split_manifest(manifest, out)
    summary = {
        "split_path": str(path),
        "config_sha256": config_content_sha256(config),
        "subset_sizes": {
            key: len(value)
            for key, value in manifest["subsets"].items()
        },
        "family_holdout_reactions": manifest["family_holdout"]["n_reactions"],
        "isolated_conflicts": len(manifest["isolated_conflicts"]),
    }
    write_json(out / "split_summary.json", summary)
    print(f"split frozen: {summary['subset_sizes']} holdout={summary['family_holdout_reactions']}")
    return EXIT_OK


def _load_frozen_reaction_ids(
    path_value: object,
    *,
    allowed: set[str],
    kind: str,
    key: str | None = None,
) -> list[str]:
    """Load a frozen cohort id list, refusing any id outside *allowed*.

    The campaign cohort files are the frozen cross-task usage lists; a
    reaction that drifted out of the qualified pool is a typed refusal, not
    a silent drop.
    """
    import json

    if not path_value:
        raise OfflineConfigError(
            f"frozen {kind} cohort file required but not configured"
        )
    payload = json.loads(Path(str(path_value)).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if key is not None:
            if key not in payload or not isinstance(payload[key], list):
                raise OfflineConfigError(
                    f"frozen {kind} cohort file {path_value} lacks list key "
                    f"{key!r}"
                )
            payload = payload[key]
        else:
            payload = next(
                (
                    value
                    for value in payload.values()
                    if isinstance(value, list)
                ),
                payload,
            )
    if not isinstance(payload, list):
        raise OfflineConfigError(
            f"frozen {kind} cohort file {path_value} is not a reaction-id list"
        )
    ids = [str(rid) for rid in payload]
    violators = sorted(set(ids) - allowed)
    if violators:
        raise OfflineConfigError(
            f"frozen {kind} cohort contains reactions outside the qualified "
            f"pool ({len(violators)}: {violators[:5]}...); re-freeze cohorts"
        )
    return sorted(set(ids))


def cmd_build_teacher(args) -> int:
    from pes2ts_core.pesn.offline.data.dataset import (
        PER_REACTION_LOOKUP_THRESHOLD,
        build_dataset_manifest,
        build_label_spec,
        build_labels_manifest,
        build_reaction_record,
        build_records_streaming,
        deterministic_selection,
        task_coverage_of,
        write_shards,
    )
    from pes2ts_core.pesn.offline.data.inventory import QUALIFIED, load_source_inventory
    from pes2ts_core.pesn.offline.splits.freeze import (
        TRAIN_DEV,
        TRAIN_FIT,
        load_split_manifest,
    )
    from pes2ts_core.pesn.offline.experiments.runs import config_content_sha256
    from pes2ts_core.utils.jsonio import write_json

    try:
        config = load_pesn_config(args.config)
    except OfflineConfigError as exc:
        print(f"config error: {exc}")
        return EXIT_CONFIG_ERROR
    if not args.allow_truth:
        print("build-teacher requires --allow-truth (audited, scope-checked reads)")
        return TRUTH_FLAG_EXIT

    paths = config.get("paths", {})
    interim = Path(str(paths.get("interim", "data/interim")))
    manifests_dir = Path(str(paths.get("manifests", "data/manifests")))
    out_root = _output_root(config, args.output_root)
    dataset_dir = out_root / "teacher_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 42))
    build_cfg = config.get("build_teacher") or {}
    fit_limit = int(build_cfg.get("train_fit_limit", 2000))
    dev_limit = int(build_cfg.get("train_dev_limit", 250))
    authority_shard = int(build_cfg.get("authority_shard_size", 1000))
    dataset_shard = int(build_cfg.get("dataset_shard_size", 200))

    try:
        source = load_source_inventory(interim)
        split_manifest = load_split_manifest(out_root / "splits")
        subsets = split_manifest["subsets"]
        qualified_fit = [
            rid
            for rid in subsets[TRAIN_FIT]
            if source.qualification(rid) == QUALIFIED
        ]
        qualified_dev = [
            rid
            for rid in subsets[TRAIN_DEV]
            if source.qualification(rid) == QUALIFIED
        ]
        frozen_fit_ids = build_cfg.get("fit_reaction_ids_file")
        frozen_dev_ids = build_cfg.get("dev_reaction_ids_file")
        if frozen_fit_ids or frozen_dev_ids:
            selected_fit = _load_frozen_reaction_ids(
                frozen_fit_ids,
                allowed=set(qualified_fit),
                kind="fit",
                key=build_cfg.get("fit_reaction_ids_key"),
            )
            selected_dev = _load_frozen_reaction_ids(
                frozen_dev_ids,
                allowed=set(qualified_dev),
                kind="dev",
                key=build_cfg.get("dev_reaction_ids_key"),
            )
        else:
            selected_fit = deterministic_selection(
                qualified_fit, seed=seed, limit=fit_limit
            )
            selected_dev = deterministic_selection(
                qualified_dev, seed=seed, limit=dev_limit
            )
        selected_all = selected_fit + selected_dev
        if len(selected_all) > PER_REACTION_LOOKUP_THRESHOLD:
            records = build_records_streaming(
                selected_all,
                interim_dir=interim,
                manifests_dir=manifests_dir,
                split_dir=out_root / "splits",
                allow_truth=True,
                shard_size=authority_shard,
            )
        else:
            records = [
                build_reaction_record(
                    rid,
                    interim_dir=interim,
                    manifests_dir=manifests_dir,
                    split_dir=out_root / "splits",
                    allow_truth=True,
                    shard_size=authority_shard,
                )
                for rid in selected_all
            ]
        exclusions = sum(1 for record in records if record.exclusion is not None)
        shards = write_shards(records, dataset_dir, shard_size=dataset_shard)
        from pes2ts_core.pesn.offline.data.dataset import (
            _finalize_label_spec_hash,
        )

        label_spec = _finalize_label_spec_hash(build_label_spec())
        write_json(dataset_dir / "label_spec.json", label_spec)
        dataset_manifest = build_dataset_manifest(
            dataset_dir=dataset_dir,
            shards=shards,
            source_hashes=source.source_hashes,
            parent_split_sha256=str(split_manifest["parent_split_sha256"]),
            label_spec_sha256=str(label_spec["content_sha256"]),
            n_reactions=len(records),
            n_excluded=exclusions,
            task_coverage=task_coverage_of(records),
        )
        dataset_manifest["subset_members"] = {
            TRAIN_FIT: selected_fit,
            TRAIN_DEV: selected_dev,
        }
        write_json(dataset_dir / "dataset_manifest.json", dataset_manifest)
        dataset_sha = sha256_file(dataset_dir / "dataset_manifest.json")
        dataset_manifest["dataset_manifest_sha256"] = dataset_sha
        write_json(dataset_dir / "dataset_manifest.json", dataset_manifest)
        labels_manifest = build_labels_manifest(
            dataset_manifest_sha256=dataset_sha,
            label_spec_sha256=str(label_spec["content_sha256"]),
            shards=shards,
            n_samples=int(dataset_manifest["n_samples"]),
        )
        write_json(dataset_dir / "labels_manifest.json", labels_manifest)
    except Exception as exc:
        print(f"build-teacher failed: {type(exc).__name__}: {exc}")
        return EXIT_PIPELINE_FAILED

    print(
        f"teacher dataset: reactions={dataset_manifest['n_reactions']} "
        f"samples={dataset_manifest['n_samples']} excluded={exclusions} "
        f"coverage={dataset_manifest['task_coverage']}"
    )
    return EXIT_OK


def cmd_train(args) -> int:
    from pes2ts_core.pesn.offline.experiments.runs import (
        RunRecord,
        config_content_sha256,
        make_run_id,
    )
    from pes2ts_core.pesn.offline.splits.freeze import TRAIN_DEV, TRAIN_FIT
    from pes2ts_core.pesn.offline.training.loader import (
        LazyReactionSequence,
        LazyTeacherProvider,
        load_teacher_reactions,
    )
    from pes2ts_core.pesn.offline.training.trainer import Trainer, TrainSettings
    from pes2ts_core.utils.jsonio import read_json

    try:
        config = load_pesn_config(args.config)
    except OfflineConfigError as exc:
        print(f"config error: {exc}")
        return EXIT_CONFIG_ERROR
    try:
        import torch

        torch_version = torch.__version__
    except ImportError:
        print("training requires torch; install requirements-pesn-training.txt")
        return EXIT_BACKEND_MISSING

    out_root = _output_root(config, args.output_root)
    dataset_dir = out_root / "teacher_dataset"
    lazy_loading = bool((config.get("train") or {}).get("lazy_loading", True))
    try:
        if lazy_loading:
            provider = LazyTeacherProvider(dataset_dir)
            dataset_manifest = provider.manifest
            train_reactions = LazyReactionSequence(
                provider, provider.valid_label_reaction_ids(subset=TRAIN_FIT)
            )
            dev_reactions = LazyReactionSequence(
                provider, provider.valid_label_reaction_ids(subset=TRAIN_DEV)
            )
        else:
            reactions = load_teacher_reactions(dataset_dir)
            dataset_manifest = read_json(dataset_dir / "dataset_manifest.json")
            subsets = dataset_manifest["subset_members"]
            by_subset = {TRAIN_FIT: set(subsets[TRAIN_FIT]), TRAIN_DEV: set(subsets[TRAIN_DEV])}
            train_reactions = [r for r in reactions if r.reaction_id in by_subset[TRAIN_FIT] and r.s_mask.any()]
            dev_reactions = [r for r in reactions if r.reaction_id in by_subset[TRAIN_DEV] and r.s_mask.any()]
    except Exception as exc:
        print(f"train data load failed: {type(exc).__name__}: {exc}")
        return EXIT_CHECKSUM_ERROR

    settings = TrainSettings.from_config(config)
    config_sha = config_content_sha256(config)
    dataset_sha = sha256_file(dataset_dir / "dataset_manifest.json")

    run_record = None
    resume_path = None
    if args.resume:
        run_record = RunRecord(out_root / args.resume)
        resume_path = run_record.run_dir / "checkpoint_last.pt"
        run_id = args.resume
    else:
        attempt = 1
        run_id = make_run_id(config_sha256=config_sha, seed=settings.seed, attempt=attempt)
        while (out_root / run_id).exists():
            attempt += 1
            run_id = make_run_id(config_sha256=config_sha, seed=settings.seed, attempt=attempt)
        run_record = RunRecord.start(
            runs_root=out_root,
            run_id=run_id,
            config=config,
            config_sha256=config_sha,
            dataset_manifest_sha256=dataset_sha,
            split_manifest_sha256=str(dataset_manifest.get("parent_split_manifest_sha256", "")),
            feature_spec_version="pesn_feature_spec_v1.0",
            label_spec_sha256=str(dataset_manifest.get("label_spec_sha256", "")),
            seed=settings.seed,
            n_parameters=0,
            backend_versions={"torch": torch_version},
        )

    trainer = Trainer(
        settings=settings,
        run_record=run_record,
        config_sha256=config_sha,
        dataset_manifest_sha256=dataset_sha,
    )
    run_record.update_status(
        "running", n_parameters=trainer.model.n_parameters()
    )
    try:
        summary = trainer.train(train_reactions, dev_reactions, resume_from=resume_path)
    except Exception as exc:
        run_record.record_failure("train", str(exc))
        run_record.finalize("failed")
        print(f"train failed: {type(exc).__name__}: {exc}")
        return EXIT_PIPELINE_FAILED
    run_record.finalize("completed")
    print(f"train complete: run_id={run_id} best_epoch={summary['best_epoch']} dev_mae_s={summary['best_dev_mae_s']}")
    return EXIT_OK


def cmd_evaluate(args) -> int:
    from pes2ts_core.pesn.offline.evaluation.benchmark import (
        run_frozen_benchmark,
        write_benchmark_report,
    )
    from pes2ts_core.pesn.offline.splits.freeze import TRAIN_DEV
    from pes2ts_core.pesn.offline.training.checkpoints import load_checkpoint
    from pes2ts_core.pesn.offline.training.loader import load_teacher_reactions
    from pes2ts_core.pesn.offline.training.trainer import Trainer, TrainSettings
    from pes2ts_core.utils.jsonio import read_json

    try:
        config = load_pesn_config(args.config)
    except OfflineConfigError as exc:
        print(f"config error: {exc}")
        return EXIT_CONFIG_ERROR
    out_root = _output_root(config, args.output_root)
    dataset_dir = out_root / "teacher_dataset"
    run_dirs = sorted(
        (p for p in out_root.iterdir() if p.is_dir() and p.name.startswith("g3p1-")),
        key=lambda p: p.stat().st_mtime,
    )
    if not run_dirs:
        print("no training run found; train first")
        return EXIT_CHECKSUM_ERROR
    run_dir = run_dirs[-1]
    best = run_dir / "checkpoint_best.pt"
    if not best.is_file():
        print(f"best checkpoint missing in {run_dir}")
        return EXIT_CHECKSUM_ERROR

    settings = TrainSettings.from_config(config)
    trainer = Trainer(
        settings=settings,
        run_record=None,
        config_sha256="eval",
        dataset_manifest_sha256="eval",
    )
    load_checkpoint(
        best,
        model=trainer.model,
        optimizer=trainer.optimizer,
        expected_config_sha256=None,
        expected_dataset_sha256=None,
    )
    try:
        reactions = load_teacher_reactions(dataset_dir)
        dataset_manifest = read_json(dataset_dir / "dataset_manifest.json")
        dev_ids = set(dataset_manifest["subset_members"][TRAIN_DEV])
        dev_reactions = [r for r in reactions if r.reaction_id in dev_ids]
    except Exception as exc:
        print(f"eval data load failed: {type(exc).__name__}: {exc}")
        return EXIT_CHECKSUM_ERROR

    report = run_frozen_benchmark(
        trainer.model,
        dev_reactions,
        eval_name=f"train_dev@{run_dir.name}",
        pair_margin=settings.pair_margin,
    )
    write_benchmark_report(report, run_dir / "metrics.json")
    print(
        "eval: model mae_s={:.4f} vs rule={:.4f} constant={:.4f} | dir_cos={:.3f} | act_auc={:.3f}".format(
            report["model"]["mae_s"],
            report["baseline_rule"]["mae_s"],
            report["baseline_constant"]["mae_s"],
            report["model"]["direction_cosine_mean"],
            report["model"]["activity_pr_auc"],
        )
    )
    return EXIT_OK


def cmd_export(args) -> int:
    from pes2ts_core.pesn.offline.evaluation.benchmark import run_frozen_benchmark
    from pes2ts_core.pesn.offline.publishing.export import export_bundle
    from pes2ts_core.pesn.offline.splits.freeze import TRAIN_DEV
    from pes2ts_core.pesn.offline.training.checkpoints import load_checkpoint
    from pes2ts_core.pesn.offline.training.loader import load_teacher_reactions
    from pes2ts_core.pesn.offline.training.trainer import Trainer, TrainSettings
    from pes2ts_core.utils.jsonio import read_json

    try:
        config = load_pesn_config(args.config)
    except OfflineConfigError as exc:
        print(f"config error: {exc}")
        return EXIT_CONFIG_ERROR
    out_root = _output_root(config, args.output_root)
    dataset_dir = out_root / "teacher_dataset"
    run_dirs = sorted(
        (p for p in out_root.iterdir() if p.is_dir() and p.name.startswith("g3p1-")),
        key=lambda p: p.stat().st_mtime,
    )
    if not run_dirs:
        print("no training run found; train first")
        return EXIT_CHECKSUM_ERROR
    run_dir = run_dirs[-1]
    best = run_dir / "checkpoint_best.pt"
    metrics_path = run_dir / "metrics.json"
    if not best.is_file():
        print(f"best checkpoint missing in {run_dir}")
        return EXIT_CHECKSUM_ERROR

    settings = TrainSettings.from_config(config)
    trainer = Trainer(
        settings=settings,
        run_record=None,
        config_sha256="export",
        dataset_manifest_sha256="export",
    )
    load_checkpoint(
        best,
        model=trainer.model,
        optimizer=trainer.optimizer,
        expected_config_sha256=None,
        expected_dataset_sha256=None,
    )

    if metrics_path.is_file():
        metrics = read_json(metrics_path)
    else:
        try:
            reactions = load_teacher_reactions(dataset_dir)
            dataset_manifest = read_json(dataset_dir / "dataset_manifest.json")
            dev_ids = set(dataset_manifest["subset_members"][TRAIN_DEV])
            dev_reactions = [r for r in reactions if r.reaction_id in dev_ids]
            metrics = run_frozen_benchmark(
                trainer.model,
                dev_reactions,
                eval_name=f"train_dev@{run_dir.name}",
                pair_margin=settings.pair_margin,
            )
        except Exception as exc:
            print(f"export metrics failed: {type(exc).__name__}: {exc}")
            return EXIT_PIPELINE_FAILED

    run_doc = read_json(run_dir / "run.json")
    model_metrics = metrics.get("model", metrics)
    direction_ok = (
        isinstance(model_metrics, dict)
        and isinstance(model_metrics.get("direction_cosine_mean"), float)
        and model_metrics["direction_cosine_mean"] >= 0.70
    )
    activity_ok = (
        isinstance(model_metrics, dict)
        and isinstance(model_metrics.get("activity_pr_auc"), float)
        and model_metrics["activity_pr_auc"] >= 0.60
    )
    model_id = str((config.get("bundle") or {}).get("model_id", "pesn-irc-v0.1.0-rc.1"))
    artifacts_root = str(config.get("paths", {}).get("artifacts_root", "artifacts/pesn"))
    output_dir = Path(artifacts_root) / model_id

    try:
        bundle_path = export_bundle(
            run_dir=run_dir,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            model_id=model_id,
            metrics=metrics,
            open_direction=direction_ok,
            open_activity=activity_ok,
            run_summary={
                "run_id": run_doc.get("run_id"),
                "status": run_doc.get("status"),
                "best_epoch": next(
                    (e.get("dev_mae_s") for e in reversed(run_doc.get("epochs", [])) if e.get("dev_mae_s") is not None),
                    None,
                ),
                "n_parameters": run_doc.get("n_parameters"),
            },
            model_config={
                "hidden": settings.hidden,
                "n_layers": settings.n_layers,
                "n_basis": settings.n_basis,
            },
        )
    except Exception as exc:
        print(f"export failed: {type(exc).__name__}: {exc}")
        return EXIT_PIPELINE_FAILED
    print(f"bundle exported: {bundle_path} (direction_open={direction_ok}, activity_open={activity_ok})")
    return EXIT_OK


__all__ = [
    "EXIT_BACKEND_MISSING",
    "EXIT_CHECKSUM_ERROR",
    "EXIT_CONFIG_ERROR",
    "EXIT_CONTRACT_REJECT",
    "EXIT_OK",
    "EXIT_PIPELINE_FAILED",
    "OfflineConfigError",
    "cmd_build_teacher",
    "cmd_evaluate",
    "cmd_export",
    "cmd_inventory",
    "cmd_split",
    "cmd_train",
    "load_pesn_config",
]
